"""Street-level photos from Mapillary, as :class:`photos.Photo`.

Mapillary is open street imagery: dashcam sequences, cyclists, people walking
with a 360 camera on a pole. What makes it useful here rather than just
pretty is that Mapillary runs its own structure-from-motion over every
sequence and publishes the result per image: a corrected position, a full
camera rotation, and the lens's focal length and distortion. That is most of
the work of finding where a photo sits in the model, done already.

Three things about the API shape this module.

A bbox query returns at most 2000 images and the server refuses large boxes,
so an area is asked for in tiles, and a tile that comes back full is split
in four and asked again.

Image URLs are signed and expire within days, so they are never cached: the
metadata is, and the URL for the pixels is asked for at download time.

Every image is CC BY-SA 4.0. A wall *measured* from one carries no licence,
but a texture *made* from one is a derivative and has to be shared alike and
credited. The photo keeps its creator so whoever bakes it can say so.

The access token is the caller's, not the project's. It is read from the
``MAPILLARY_TOKEN`` environment variable or from a file outside version
control, never from the config file a run writes, and it is never logged.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import requests

from .camera import (
    FISHEYE,
    PERSPECTIVE,
    SPHERICAL,
    Camera,
    rotation_from_axis_angle,
    rotation_from_heading,
)
from .geo import BBox, rd_bbox_to_wgs84
from .http_util import ServiceError, get_with_retry
from .photos import (
    POSE_GPS_COMPASS,
    POSE_GPS_ONLY,
    POSE_SFM,
    SOURCE_MAPILLARY,
    Photo,
    grid_convergence_deg,
    lonlat_to_rd,
    rotation_enu_to_grid,
)

LOG = logging.getLogger(__name__)

API_URL = "https://graph.mapillary.com"
LICENCE = "CC BY-SA 4.0"
TOKEN_ENV = "MAPILLARY_TOKEN"
# Where a token may be kept instead of the environment, checked in order. The
# first sits next to the pipeline and is ignored by git; the second is per user.
TOKEN_FILES = (
    Path(__file__).resolve().parent.parent / "mapillary_token.txt",
    Path.home() / ".config" / "terreinlab" / "mapillary_token",
)

PAGE_LIMIT = 2000
# Degrees per tile side: about 170 m east-west and 280 m north-south here,
# small enough that a busy city centre rarely fills one.
TILE_DEG = 0.0025
MIN_TILE_DEG = TILE_DEG / 16

IMAGE_FIELDS = (
    "id",
    "captured_at",
    "camera_type",
    "camera_parameters",
    "geometry",
    "computed_geometry",
    "computed_rotation",
    "compass_angle",
    "computed_compass_angle",
    "sequence",
    "width",
    "height",
    "creator",
)

SIZES = {
    256: "thumb_256_url",
    1024: "thumb_1024_url",
    2048: "thumb_2048_url",
    0: "thumb_original_url",
}

# Used when an image carries no lens calibration. Roughly a phone's main
# camera, 65 degrees across.
DEFAULT_FOCAL = 0.85


class MissingToken(ServiceError):
    """Mapillary was asked for and no token was found."""


def find_token() -> str:
    """The access token, from the environment or a token file.

    Deliberately not from the config: a config is written by the UI, copied
    between machines and kept next to the model, and a token in it travels
    wherever the model does.
    """
    value = os.environ.get(TOKEN_ENV, "").strip()
    if value:
        return value
    for path in TOKEN_FILES:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    raise MissingToken(
        "Mapillary street photos are switched on but there is no access token. "
        f"Set the {TOKEN_ENV} environment variable, or put the token in "
        f"{TOKEN_FILES[0]}. A free token comes from "
        "https://www.mapillary.com/dashboard/developers."
    )


def session_for(token: str) -> requests.Session:
    session = requests.Session()
    session.headers["Authorization"] = f"OAuth {token}"
    return session


def _tiles(bbox_wgs84: BBox, step: float = TILE_DEG) -> Iterator[BBox]:
    nx = max(1, int(np.ceil(bbox_wgs84.width / step - 1e-9)))
    ny = max(1, int(np.ceil(bbox_wgs84.height / step - 1e-9)))
    dx, dy = bbox_wgs84.width / nx, bbox_wgs84.height / ny
    for i in range(nx):
        for j in range(ny):
            yield BBox(
                bbox_wgs84.xmin + i * dx,
                bbox_wgs84.ymin + j * dy,
                bbox_wgs84.xmin + (i + 1) * dx,
                bbox_wgs84.ymin + (j + 1) * dy,
            )


def _quarters(tile: BBox) -> list[BBox]:
    cx, cy = tile.center
    return [
        BBox(tile.xmin, tile.ymin, cx, cy),
        BBox(cx, tile.ymin, tile.xmax, cy),
        BBox(tile.xmin, cy, cx, tile.ymax),
        BBox(cx, cy, tile.xmax, tile.ymax),
    ]


def _tile_key(tile: BBox) -> str:
    return "_".join(f"{v:.6f}" for v in tile.as_tuple())


def fetch_image_metadata(
    bbox_rd: BBox,
    *,
    session: requests.Session,
    cache_dir: Path | None = None,
    api_url: str = API_URL,
    timeout: float = 60.0,
    max_retries: int = 4,
    max_age_days: float = 30.0,
) -> list[dict]:
    """Every Mapillary image record inside `bbox_rd`, deduplicated by id."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def ask(tile: BBox) -> list[dict]:
        cached = cache_dir / f"{_tile_key(tile)}.json" if cache_dir else None
        if cached is not None and cached.is_file():
            age_days = (time.time() - cached.stat().st_mtime) / 86400.0
            if age_days <= max_age_days:
                return json.loads(cached.read_text(encoding="utf-8"))
        response = get_with_retry(
            f"{api_url.rstrip('/')}/images",
            params={
                "fields": ",".join(IMAGE_FIELDS),
                "bbox": ",".join(f"{v:.7f}" for v in tile.as_tuple()),
                "limit": PAGE_LIMIT,
            },
            timeout=timeout,
            max_retries=max_retries,
            session=session,
            description="Mapillary image search",
        )
        payload = response.json()
        if "error" in payload:
            raise ServiceError(
                f"Mapillary refused the image search: "
                f"{payload['error'].get('message', payload['error'])}"
            )
        records = list(payload.get("data", []))
        if cached is not None:
            cached.write_text(json.dumps(records), encoding="utf-8")
        return records

    found: dict[str, dict] = {}
    pending = list(_tiles(rd_bbox_to_wgs84(bbox_rd)))
    while pending:
        tile = pending.pop()
        records = ask(tile)
        if len(records) >= PAGE_LIMIT and tile.width / 2 >= MIN_TILE_DEG:
            # Full: there are more than one page holds. Ask again in quarters.
            pending.extend(_quarters(tile))
            continue
        if len(records) >= PAGE_LIMIT:
            LOG.warning(
                "Mapillary tile %s is still full at the smallest tile size; "
                "some images there are missing", tile,
            )
        for record in records:
            found[str(record["id"])] = record
    return list(found.values())


def _camera(record: dict) -> Camera:
    kind = str(record.get("camera_type") or PERSPECTIVE).lower()
    if kind in ("spherical", "equirectangular"):
        kind = SPHERICAL
    elif kind != FISHEYE:
        kind = PERSPECTIVE
    width = int(record.get("width") or 0)
    height = int(record.get("height") or 0)
    if width <= 0 or height <= 0:
        width, height = (4096, 2048) if kind == SPHERICAL else (4000, 3000)
    params = record.get("camera_parameters") or []
    focal = float(params[0]) if len(params) >= 1 and params[0] else DEFAULT_FOCAL
    k1 = float(params[1]) if len(params) >= 2 else 0.0
    k2 = float(params[2]) if len(params) >= 3 else 0.0
    return Camera(kind, width, height, focal, k1, k2)


def photo_from_record(record: dict) -> Photo | None:
    """One Mapillary image record as a Photo, or None if it has no position."""
    geometry = record.get("computed_geometry") or record.get("geometry")
    if not geometry or not geometry.get("coordinates"):
        return None
    lon, lat = (float(v) for v in geometry["coordinates"][:2])
    x, y = lonlat_to_rd(lon, lat)
    convergence = float(grid_convergence_deg(lon, lat)[0])

    notes: list[str] = []
    heading_known = True
    if record.get("computed_rotation") is not None and record.get("computed_geometry"):
        rotation_enu = rotation_from_axis_angle(record["computed_rotation"])
        pose, sigma = POSE_SFM, 1.5
    else:
        compass = record.get("computed_compass_angle")
        if compass is None or compass < 0:
            compass = record.get("compass_angle")
        if compass is None or compass < 0:
            compass, heading_known = 0.0, False
            pose = POSE_GPS_ONLY
            notes.append("no heading")
        else:
            pose = POSE_GPS_COMPASS
        rotation_enu = rotation_from_heading(float(compass))
        sigma = 5.0

    captured = record.get("captured_at")
    when = (
        datetime.fromtimestamp(float(captured) / 1000.0, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
        if captured
        else ""
    )
    creator = record.get("creator") or {}
    return Photo(
        id=f"mly_{record['id']}",
        source=SOURCE_MAPILLARY,
        x=float(x),
        y=float(y),
        z=0.0,
        rotation=rotation_enu_to_grid(rotation_enu, convergence),
        camera=_camera(record),
        pose=pose,
        heading_known=heading_known,
        captured_at=when,
        sequence=str(record.get("sequence") or ""),
        creator=str(creator.get("username", "")) if isinstance(creator, dict) else "",
        licence=LICENCE,
        position_sigma_m=sigma,
        notes=notes,
    )


def mapillary_id(photo: Photo) -> str:
    return photo.id.removeprefix("mly_")


def fetch_photos(
    bbox_rd: BBox,
    *,
    photos_cfg: dict,
    work_dir: Path,
    session: requests.Session | None = None,
) -> tuple[list[Photo], dict]:
    """Mapillary photos around `bbox_rd`, and counts of what was dropped why."""
    margin = float(photos_cfg.get("max_distance_m", 40.0))
    if session is None:
        session = session_for(find_token())
    records = fetch_image_metadata(
        bbox_rd.buffered(margin),
        session=session,
        cache_dir=work_dir / "_mapillary" / "search",
        api_url=str(photos_cfg.get("mapillary_url", API_URL)),
        timeout=float(photos_cfg.get("timeout_s", 60.0)),
        max_retries=int(photos_cfg.get("max_retries", 4)),
    )

    min_year = int(photos_cfg.get("min_year", 0) or 0)
    counts = {"found": len(records), "no_position": 0, "too_old": 0}
    photos: list[Photo] = []
    for record in records:
        photo = photo_from_record(record)
        if photo is None:
            counts["no_position"] += 1
            continue
        if min_year and photo.captured_at and int(photo.captured_at[:4]) < min_year:
            counts["too_old"] += 1
            continue
        photos.append(photo)
    counts["kept"] = len(photos)
    return photos, counts


def image_urls(
    ids: list[str],
    *,
    session: requests.Session,
    size: int = 2048,
    api_url: str = API_URL,
    timeout: float = 60.0,
    max_retries: int = 4,
) -> dict[str, str]:
    """Fresh signed URLs for the pixels, asked for fifty images at a time."""
    field = SIZES.get(int(size), SIZES[2048])
    urls: dict[str, str] = {}
    for start in range(0, len(ids), 50):
        batch = ids[start : start + 50]
        response = get_with_retry(
            f"{api_url.rstrip('/')}/",
            params={"ids": ",".join(batch), "fields": field},
            timeout=timeout,
            max_retries=max_retries,
            session=session,
            description="Mapillary image URLs",
        )
        for key, value in response.json().items():
            if isinstance(value, dict) and value.get(field):
                urls[str(key)] = str(value[field])
    return urls


def download_images(
    photos: list[Photo],
    cache_dir: Path,
    *,
    session: requests.Session,
    size: int = 2048,
    api_url: str = API_URL,
    timeout: float = 60.0,
    max_retries: int = 4,
) -> list[Photo]:
    """Fetch the pixels for Mapillary photos, returning them with ``path`` set.

    The camera is rescaled to the size actually downloaded, since a 2048 px
    thumbnail of a 4000 px original has the same lens on a smaller grid.
    Photos that already have a file keep it and cost nothing.
    """
    from dataclasses import replace

    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "orig" if int(size) == 0 else str(int(size))

    def target(photo: Photo) -> Path:
        return cache_dir / f"{mapillary_id(photo)}_{tag}.jpg"

    wanted = [
        p for p in photos if p.source == SOURCE_MAPILLARY and not target(p).is_file()
    ]
    urls = (
        image_urls(
            [mapillary_id(p) for p in wanted],
            session=session,
            size=size,
            api_url=api_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        if wanted
        else {}
    )
    def fetch(photo: Photo) -> str:
        url = urls.get(mapillary_id(photo))
        if not url:
            return "no image URL"
        # The signed URL is on a CDN that knows nothing of our token.
        try:
            response = get_with_retry(
                url,
                timeout=timeout,
                max_retries=max_retries,
                expect_binary=True,
                description="Mapillary image",
            )
        except ServiceError as exc:
            return f"download failed: {exc}"
        target(photo).write_bytes(response.content)
        return ""

    # The CDN answers each image in a fraction of a second, so a handful at
    # once is what decides how long a few hundred take.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = dict(zip((p.id for p in wanted), pool.map(fetch, wanted)))

    out: list[Photo] = []
    for photo in photos:
        if photo.source != SOURCE_MAPILLARY:
            out.append(photo)
            continue
        path = target(photo)
        if not path.is_file():
            why = failures.get(photo.id) or "no image"
            out.append(replace(photo, notes=photo.notes + [why]))
            continue
        with Image.open(path) as image:
            width, height = image.size
        out.append(
            replace(
                photo,
                path=str(path),
                camera=photo.camera.scaled(width, height),
            )
        )
    return out


__all__ = [
    "API_URL",
    "LICENCE",
    "MissingToken",
    "TOKEN_ENV",
    "download_images",
    "fetch_image_metadata",
    "fetch_photos",
    "find_token",
    "image_urls",
    "photo_from_record",
    "session_for",
]
