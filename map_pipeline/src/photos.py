"""Street photos as one kind of thing, whoever took them.

A Mapillary image and a photo off your own camera arrive with different
metadata: one carries a pose from Mapillary's reconstruction, the other a GPS
fix and a compass heading in its EXIF, or nothing at all. Everything after
ingest -- which walls a photo sees, how sharp it sees them, what it shows --
only needs a position, an orientation and a lens, so both are turned into a
:class:`Photo` here and nothing downstream asks where it came from.

The world frame is the pipeline's own: RD New metres east and north, NAP
metres up. Orientation is a world-to-camera rotation in that frame (see
`camera.py` for the conventions).

RD's grid north is not true north. The difference, the meridian convergence,
is zero along 5°23' east and grows to about 1.5° at the borders; a compass
heading or a Mapillary rotation is relative to true north, so it is turned
onto the grid here. Over 30 m a degree is half a metre on a facade.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .camera import Camera, heading_of

LOG = logging.getLogger(__name__)

# How a photo's pose was obtained, best first. The later stages trust a pose
# according to this: an SfM pose is good to a metre or two, a phone's compass
# to ten or twenty degrees, and a photo with only GPS has no heading at all.
POSE_MANUAL = "manual"
POSE_SFM = "sfm"
POSE_GPS_COMPASS = "gps+compass"
POSE_GPS_ONLY = "gps"
POSE_SOURCES = (POSE_MANUAL, POSE_SFM, POSE_GPS_COMPASS, POSE_GPS_ONLY)

SOURCE_MAPILLARY = "mapillary"
SOURCE_OWN = "own"


@dataclass
class Photo:
    """One street photo with its pose in the RD/NAP frame."""

    id: str
    source: str
    x: float
    y: float
    z: float
    # World-to-camera, rows are the camera's right, down and forward axes.
    rotation: np.ndarray
    camera: Camera
    pose: str = POSE_GPS_COMPASS
    # False when nothing said which way the camera faced. The pose then holds
    # a placeholder heading that a later stage has to solve.
    heading_known: bool = True
    captured_at: str = ""
    # Where the pixels are: a URL to fetch, or a file on disk.
    image_url: str = ""
    path: str = ""
    sequence: str = ""
    creator: str = ""
    licence: str = ""
    # Metres; how far the position is believed to be off.
    position_sigma_m: float = 3.0
    # Camera height above the ground, when the photo says. Otherwise the
    # run's default for its source applies.
    height_m: float | None = None
    # Whether the pose was checked against the model and its edges agreed.
    # None until a check has been made; a photo that failed one never
    # becomes a texture, and is not trusted to measure either.
    verified: bool | None = None
    # The buildings whose own edges lined up with this photo, by identifier:
    # the walls it may paint. None until a check has been made.
    verified_buildings: list[str] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def heading_deg(self) -> float:
        return heading_of(self.rotation)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "z": round(float(self.z), 3),
            "rotation": [[round(float(v), 9) for v in row] for row in self.rotation],
            "heading_deg": round(self.heading_deg, 2),
            "camera": self.camera.to_dict(),
            "pose": self.pose,
            "heading_known": self.heading_known,
            "captured_at": self.captured_at,
            "image_url": self.image_url,
            "path": self.path,
            "sequence": self.sequence,
            "creator": self.creator,
            "licence": self.licence,
            "position_sigma_m": self.position_sigma_m,
            **({"height_m": self.height_m} if self.height_m is not None else {}),
            **({"verified": self.verified} if self.verified is not None else {}),
            **(
                {"verified_buildings": list(self.verified_buildings)}
                if self.verified_buildings is not None
                else {}
            ),
            **({"notes": list(self.notes)} if self.notes else {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Photo":
        return cls(
            id=str(data["id"]),
            source=str(data["source"]),
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
            rotation=np.asarray(data["rotation"], dtype=np.float64),
            camera=Camera.from_dict(data["camera"]),
            pose=str(data.get("pose", POSE_GPS_COMPASS)),
            heading_known=bool(data.get("heading_known", True)),
            captured_at=str(data.get("captured_at", "")),
            image_url=str(data.get("image_url", "")),
            path=str(data.get("path", "")),
            sequence=str(data.get("sequence", "")),
            creator=str(data.get("creator", "")),
            licence=str(data.get("licence", "")),
            position_sigma_m=float(data.get("position_sigma_m", 3.0)),
            height_m=(
                float(data["height_m"]) if data.get("height_m") is not None else None
            ),
            verified=data.get("verified"),
            verified_buildings=data.get("verified_buildings"),
            notes=list(data.get("notes", [])),
        )


@lru_cache(maxsize=None)
def _cached_transformer(src_crs: str, dst_crs: str):
    """One transformer per direction for the whole run.

    Building a pyproj transformer costs about 30 ms, and a Mapillary search
    returns thousands of images: making one per image was 90 seconds of a
    300 m area's two-minute photo stage.
    """
    from .geo import _transformer

    return _transformer(src_crs, dst_crs)


def grid_convergence_deg(lon, lat) -> np.ndarray:
    """Angle from RD grid north to true north, degrees clockwise.

    Measured rather than taken from a formula: step a little way north along
    the meridian and see which way that goes on the grid.
    """
    from .geo import RD_CRS, WGS84_CRS

    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    tf = _cached_transformer(WGS84_CRS, RD_CRS)
    x0, y0 = tf.transform(lon, lat)
    x1, y1 = tf.transform(lon, lat + 1e-4)
    return np.degrees(np.arctan2(np.asarray(x1) - x0, np.asarray(y1) - y0))


def enu_to_grid(convergence_deg: float) -> np.ndarray:
    """Matrix taking a local east-north-up vector onto RD grid axes.

    True north points `convergence_deg` clockwise of grid north, so a vector
    is turned clockwise by that much about the vertical.
    """
    b = np.radians(float(convergence_deg))
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_enu_to_grid(rotation_enu: np.ndarray, convergence_deg: float) -> np.ndarray:
    """Re-express a world-to-camera rotation given in ENU against RD axes.

    ``x_cam = R_enu @ v_enu`` and ``v_grid = M @ v_enu``, so
    ``x_cam = R_enu @ M.T @ v_grid``.
    """
    return np.asarray(rotation_enu) @ enu_to_grid(convergence_deg).T


def lonlat_to_rd(lon, lat):
    from .geo import RD_CRS, WGS84_CRS

    return _cached_transformer(WGS84_CRS, RD_CRS).transform(lon, lat)


def rd_to_lonlat(x, y):
    from .geo import RD_CRS, WGS84_CRS

    return _cached_transformer(RD_CRS, WGS84_CRS).transform(x, y)


def save_photos(photos: Iterable[Photo], path: Path) -> Path:
    photos = list(photos)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"photos": [p.to_dict() for p in photos]}, indent=1),
        encoding="utf-8",
    )
    return path


def load_photos(path: Path) -> list[Photo]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Photo.from_dict(item) for item in data.get("photos", [])]


def thin_photos(photos: list[Photo], spacing_m: float) -> list[Photo]:
    """Keep one photo per `spacing_m` along each sequence, and every own photo.

    A dashcam sequence takes a frame every metre or two, and neighbours see
    the same wall from nearly the same place. Keeping them all multiplies the
    work downstream without adding a view. Own photos are never thinned:
    someone chose to take each of them.
    """
    if spacing_m <= 0:
        return list(photos)
    kept: list[Photo] = []
    last_by_sequence: dict[str, np.ndarray] = {}
    for photo in sorted(photos, key=lambda p: (p.sequence, p.captured_at, p.id)):
        if photo.source != SOURCE_MAPILLARY or not photo.sequence:
            kept.append(photo)
            continue
        here = np.array([photo.x, photo.y])
        previous = last_by_sequence.get(photo.sequence)
        if previous is None or np.hypot(*(here - previous)) >= spacing_m:
            kept.append(photo)
            last_by_sequence[photo.sequence] = here
    return kept


def with_height(
    photos: list[Photo], ground_at, camera_height_m: float | dict[str, float]
) -> list[Photo]:
    """Put each photo at its camera height above the terrain under it.

    Neither GPS altitude nor Mapillary's reconstructed altitude is good to
    better than several metres, while the AHN under the camera is good to
    centimetres and a camera is held, or mounted, at a known height above it.
    `camera_height_m` is one height, or one per source; a photo that states
    its own wins. A photo placed by hand with an explicit NAP height keeps it.
    """
    if not photos:
        return []
    xs = np.array([p.x for p in photos])
    ys = np.array([p.y for p in photos])
    ground = np.asarray(ground_at(xs, ys), dtype=np.float64).reshape(-1)
    out = []
    for photo, z in zip(photos, ground):
        if photo.pose == POSE_MANUAL and photo.z != 0.0 and photo.height_m is None:
            out.append(photo)
            continue
        if photo.height_m is not None:
            above = float(photo.height_m)
        elif isinstance(camera_height_m, dict):
            above = float(camera_height_m.get(photo.source, 2.0))
        else:
            above = float(camera_height_m)
        notes = photo.notes
        if not np.isfinite(z):
            notes = notes + ["no terrain under the camera"]
            z = 0.0
        out.append(replace(photo, z=float(z) + above, notes=notes))
    return out


__all__ = [
    "POSE_GPS_COMPASS",
    "POSE_GPS_ONLY",
    "POSE_MANUAL",
    "POSE_SFM",
    "POSE_SOURCES",
    "Photo",
    "SOURCE_MAPILLARY",
    "SOURCE_OWN",
    "enu_to_grid",
    "grid_convergence_deg",
    "load_photos",
    "lonlat_to_rd",
    "rd_to_lonlat",
    "rotation_enu_to_grid",
    "save_photos",
    "thin_photos",
    "with_height",
]
