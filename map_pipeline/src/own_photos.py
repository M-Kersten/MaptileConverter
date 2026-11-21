"""Your own photos of a place, read into :class:`photos.Photo`.

For the spots Mapillary never drove past: a courtyard, a quay, the side of a
building the street does not face. Drop the pictures in a folder and point
``photos.folder`` at it. What each one needs is a position and, ideally, a
heading, and where they come from is tried in this order:

1. A sidecar next to the picture, ``IMG_0042.jpg.json`` or ``IMG_0042.json``,
   or an entry under the file's name in a ``poses.json`` in the folder. This
   is what the UI writes when someone drags a photo to where it was taken,
   and it wins over everything the camera said. Keys: ``lat``/``lon`` or RD
   ``x``/``y``, and optionally ``heading``, ``pitch``, ``roll``,
   ``height_m`` (above the ground), ``z`` (NAP), ``kind`` and ``focal``.
2. The EXIF GPS block: latitude, longitude, and ``GPSImgDirection`` when the
   phone recorded which way it pointed.
3. The GPano XMP block a 360 camera writes: ``ProjectionType`` marks an
   equirectangular image and ``PoseHeadingDegrees`` says which way its
   middle column faces.

A photo with a position but no heading is kept, flagged ``heading_known =
False``: a 360 photo still sees every wall around it, and the heading can be
solved against the model later. A photo with no position is skipped and
listed, because nothing downstream can guess where it was.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .camera import KINDS, PERSPECTIVE, SPHERICAL, Camera, rotation_from_heading
from .photos import (
    POSE_GPS_COMPASS,
    POSE_GPS_ONLY,
    POSE_MANUAL,
    SOURCE_OWN,
    Photo,
    grid_convergence_deg,
    lonlat_to_rd,
    rd_to_lonlat,
    rotation_enu_to_grid,
)

LOG = logging.getLogger(__name__)

EXTENSIONS = (".jpg", ".jpeg", ".png")

# EXIF tags, by number, so no lookup table from Pillow is needed.
TAG_ORIENTATION = 0x0112
IFD_EXIF = 0x8769
IFD_GPS = 0x8825
TAG_DATETIME_ORIGINAL = 0x9003
TAG_FOCAL_35MM = 0xA405
GPS_LAT_REF, GPS_LAT, GPS_LON_REF, GPS_LON = 1, 2, 3, 4
GPS_IMG_DIRECTION_REF, GPS_IMG_DIRECTION = 16, 17

# Magnetic declination over the Netherlands, 2025-2030: about 2.5 degrees
# east, varying by less than a degree across the country. A phone that writes
# a magnetic heading is off by more than that anyway.
DECLINATION_DEG = 2.5

# The diagonal of a 36 x 24 mm frame, which is what "35 mm equivalent" means.
FULL_FRAME_DIAGONAL_MM = float(np.hypot(36.0, 24.0))

DEFAULT_FOCAL = 0.85


@dataclass
class OwnPhotoResult:
    photos: list[Photo] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def stats(self) -> dict:
        return {
            "read": len(self.photos),
            "skipped": len(self.skipped),
            "spherical": sum(1 for p in self.photos if p.camera.kind == SPHERICAL),
            "without_heading": sum(1 for p in self.photos if not p.heading_known),
            "placed_by_hand": sum(1 for p in self.photos if p.pose == POSE_MANUAL),
            **(
                {"skipped_files": [f"{name}: {why}" for name, why in self.skipped[:50]]}
                if self.skipped
                else {}
            ),
        }


def _rational(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        numerator, denominator = value
        return float(numerator) / float(denominator)


def _dms(value, ref) -> float:
    degrees, minutes, seconds = (_rational(v) for v in value)
    out = degrees + minutes / 60.0 + seconds / 3600.0
    return -out if str(ref).upper().startswith(("S", "W")) else out


_XMP_FIELD = r"GPano:{name}\s*(?:=\s*\"([^\"]*)\"|>\s*([^<]*)<)"


def read_gpano(raw: bytes) -> dict[str, str]:
    """The GPano fields of an XMP packet, attribute or element form."""
    text = raw.decode("utf-8", errors="ignore")
    out = {}
    for name in (
        "ProjectionType",
        "PoseHeadingDegrees",
        "PosePitchDegrees",
        "PoseRollDegrees",
    ):
        match = re.search(_XMP_FIELD.format(name=name), text)
        if match:
            out[name] = (match.group(1) or match.group(2) or "").strip()
    return out


def _xmp_bytes(path: Path, image) -> bytes:
    raw = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp") or b""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if raw:
        return raw
    # Older Pillow keeps the APP1 XMP segment out of `info`. It sits near the
    # start of the file, so a bounded read finds it.
    with path.open("rb") as fh:
        head = fh.read(512 * 1024)
    start = head.find(b"<x:xmpmeta")
    end = head.find(b"</x:xmpmeta>", start)
    return head[start : end + 12] if start >= 0 and end > start else b""


def _sidecar(path: Path, folder_poses: dict) -> dict | None:
    for candidate in (path.with_name(path.name + ".json"), path.with_suffix(".json")):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                LOG.warning("ignoring %s: not valid JSON (%s)", candidate.name, exc)
    return folder_poses.get(path.name)


def _focal_from_35mm(f35: float, width: int, height: int) -> float:
    """Normalised focal from a 35 mm-equivalent focal length."""
    return float(f35) / FULL_FRAME_DIAGONAL_MM * float(np.hypot(width, height)) / max(
        width, height
    )


def read_photo(
    path: Path, folder_poses: dict | None = None
) -> tuple[Photo | None, str]:
    """One file as a Photo, or None and the reason it could not be placed."""
    from PIL import Image

    folder_poses = folder_poses or {}
    try:
        image = Image.open(path)
    except OSError as exc:
        return None, f"not an image ({exc})"

    with image:
        exif = image.getexif()
        width, height = image.size
        # A phone stores a portrait photo as landscape pixels plus a flag.
        # Downstream reads the pixels upright, so the camera has to be upright.
        if exif.get(TAG_ORIENTATION) in (5, 6, 7, 8):
            width, height = height, width
        gps = exif.get_ifd(IFD_GPS) if exif else {}
        sub = exif.get_ifd(IFD_EXIF) if exif else {}
        gpano = read_gpano(_xmp_bytes(path, image))

    side = _sidecar(path, folder_poses) or {}
    notes: list[str] = []

    # Lens.
    kind = str(side.get("kind", "")).strip().lower()
    if kind not in KINDS:
        projection = gpano.get("ProjectionType", "").lower()
        if projection == "equirectangular" or (
            width == 2 * height and width >= 2000
        ):
            kind = SPHERICAL
        else:
            kind = PERSPECTIVE
    if side.get("focal"):
        focal = float(side["focal"])
    elif sub.get(TAG_FOCAL_35MM):
        focal = _focal_from_35mm(_rational(sub[TAG_FOCAL_35MM]), width, height)
    else:
        focal = DEFAULT_FOCAL
        if kind != SPHERICAL:
            notes.append("no focal length in EXIF, assumed a 65 degree lens")
    camera = Camera(kind, width, height, focal if kind != SPHERICAL else 1.0)

    # Position.
    pose = POSE_GPS_COMPASS
    if "x" in side and "y" in side:
        x, y = float(side["x"]), float(side["y"])
        lon = lat = None
        pose = POSE_MANUAL
    elif "lat" in side and "lon" in side:
        lon, lat = float(side["lon"]), float(side["lat"])
        x, y = lonlat_to_rd(lon, lat)
        pose = POSE_MANUAL
    elif gps.get(GPS_LAT) and gps.get(GPS_LON):
        try:
            lat = _dms(gps[GPS_LAT], gps.get(GPS_LAT_REF, "N"))
            lon = _dms(gps[GPS_LON], gps.get(GPS_LON_REF, "E"))
        except (TypeError, ValueError, ZeroDivisionError):
            return None, "unreadable GPS position"
        x, y = lonlat_to_rd(lon, lat)
    else:
        return None, "no position: no GPS in the file and no sidecar"

    if lon is None:
        lon, lat = rd_to_lonlat(x, y)
    convergence = float(grid_convergence_deg(lon, lat)[0])

    # Heading. A sidecar heading is on the RD grid, since that is the map the
    # person placed it on; the camera's own is against true or magnetic north.
    heading_known = True
    pitch = float(side.get("pitch", 0.0))
    roll = float(side.get("roll", 0.0))
    if side.get("heading") is not None:
        rotation = rotation_from_heading(float(side["heading"]), pitch, roll)
    else:
        heading = None
        if kind == SPHERICAL and gpano.get("PoseHeadingDegrees"):
            heading = float(gpano["PoseHeadingDegrees"])
            pitch = float(gpano.get("PosePitchDegrees") or pitch)
            roll = float(gpano.get("PoseRollDegrees") or roll)
        elif gps.get(GPS_IMG_DIRECTION) is not None:
            heading = _rational(gps[GPS_IMG_DIRECTION])
            if str(gps.get(GPS_IMG_DIRECTION_REF, "T")).upper().startswith("M"):
                heading += DECLINATION_DEG
        if heading is None:
            heading_known = False
            heading = 0.0
            if pose != POSE_MANUAL:
                pose = POSE_GPS_ONLY
            notes.append("no heading: to be solved against the model")
        rotation = rotation_enu_to_grid(
            rotation_from_heading(heading, pitch, roll), convergence
        )

    z = float(side["z"]) if "z" in side else 0.0
    taken = str(sub.get(TAG_DATETIME_ORIGINAL, "") or "")
    try:
        taken = (
            datetime.strptime(taken, "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S")
            if taken
            else ""
        )
    except ValueError:
        taken = ""

    photo = Photo(
        id=f"own_{path.stem}",
        source=SOURCE_OWN,
        x=float(x),
        y=float(y),
        z=z,
        rotation=rotation,
        camera=camera,
        pose=pose,
        heading_known=heading_known,
        captured_at=taken,
        path=str(path),
        position_sigma_m=1.0 if pose == POSE_MANUAL else 5.0,
        height_m=float(side["height_m"]) if "height_m" in side else None,
        notes=notes,
    )
    return photo, ""


def read_folder(folder: Path) -> OwnPhotoResult:
    """Every placeable photo in `folder`, recursively, sorted by name."""
    result = OwnPhotoResult()
    if not folder.is_dir():
        result.skipped.append((str(folder), "folder does not exist"))
        return result
    poses_file = folder / "poses.json"
    folder_poses: dict = {}
    if poses_file.is_file():
        try:
            folder_poses = json.loads(poses_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            LOG.warning("ignoring %s: not valid JSON (%s)", poses_file, exc)

    seen: set[str] = set()
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        photo, why = read_photo(path, folder_poses)
        if photo is None:
            result.skipped.append((path.name, why))
            continue
        # Two files called IMG_0001 in different subfolders are two photos.
        if photo.id in seen:
            photo.id = f"{photo.id}_{len(seen)}"
        seen.add(photo.id)
        result.photos.append(photo)
    return result


__all__ = [
    "DECLINATION_DEG",
    "OwnPhotoResult",
    "read_folder",
    "read_gpano",
    "read_photo",
]
