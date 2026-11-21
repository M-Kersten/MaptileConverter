"""The page's own photos: one folder per area, and what is known of each.

The web page lets someone drop their photos of a place onto the map, put
each where it was taken, turn it the way it looked, or pin it to a wall.
All of that ends up as files the pipeline already reads: the photos in
``photos/<area>/``, and beside each one the sidecar `own_photos` describes.
This module is the page's side of that, kept out of the server so it can be
tested without one.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

ALLOWED = (".jpg", ".jpeg", ".png")
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
THUMB_PX = 320


def folder_for(root: Path, area: str) -> Path:
    """Where an area's photos live. `area` must already be a safe name."""
    return root / "photos" / area


def safe_name(name: str) -> str:
    """A file name that can only be one file in the folder."""
    base = Path(str(name)).name
    stem, suffix = Path(base).stem, Path(base).suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError(f"only {', '.join(ALLOWED)} photos can be added")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "photo"
    return f"{stem[:80]}{suffix}"


def photos_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED and not p.name.startswith(".")
    )


def sidecar_path(folder: Path, name: str) -> Path:
    return folder / f"{name}.json"


def read_sidecar(folder: Path, name: str) -> dict:
    path = sidecar_path(folder, name)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_sidecar(folder: Path, name: str, data: dict) -> None:
    sidecar_path(folder, name).write_text(json.dumps(data, indent=1), encoding="utf-8")


def save_upload(folder: Path, name: str, data: bytes) -> str:
    """Store one uploaded photo; returns the name it was stored under."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"a photo may be at most {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    if not data:
        raise ValueError("the photo was empty")
    clean = safe_name(name)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / clean
    stem, suffix = target.stem, target.suffix
    n = 1
    while target.exists():
        target = folder / f"{stem}-{n}{suffix}"
        n += 1
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - anything unreadable is refused alike
        raise ValueError(f"that is not a photo Pillow can read ({exc})") from exc
    target.write_bytes(data)
    return target.name


def describe(folder: Path) -> list[dict]:
    """Every photo in the folder, and what the pipeline will make of it."""
    from .geo import RD_CRS, WGS84_CRS, _transformer
    from .own_photos import read_photo

    poses = {}
    poses_file = folder / "poses.json"
    if poses_file.is_file():
        try:
            poses = json.loads(poses_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            poses = {}
    to_wgs = _transformer(RD_CRS, WGS84_CRS)
    out = []
    for path in photos_in(folder):
        photo, why = read_photo(path, poses)
        side = read_sidecar(folder, path.name)
        entry = {
            "name": path.name,
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "usable": photo is not None,
            "problem": why,
            "pinned": bool(side.get("pin")),
            "placed_by_hand": "x" in side or "lat" in side,
        }
        if photo is not None:
            lon, lat = to_wgs.transform(photo.x, photo.y)
            entry.update({
                "x": round(photo.x, 2),
                "y": round(photo.y, 2),
                "lon": round(float(lon), 7),
                "lat": round(float(lat), 7),
                "heading": round(photo.heading_deg, 1) if photo.heading_known else None,
                "kind": photo.camera.kind,
                "pose": photo.pose,
                "notes": photo.notes,
            })
        out.append(entry)
    return out


def set_pose(folder: Path, name: str, *, x: float, y: float, heading: float | None = None) -> dict:
    """Put a photo where it was taken, and turn it the way it looked.

    `heading` is on the RD grid, degrees clockwise from grid north, which is
    what someone turning a photo on the page's map sets.
    """
    clean = safe_name(name)
    if not (folder / clean).is_file():
        raise FileNotFoundError(clean)
    side = read_sidecar(folder, clean)
    side.pop("lat", None)
    side.pop("lon", None)
    side.update({"x": round(float(x), 3), "y": round(float(y), 3)})
    if heading is not None:
        side["heading"] = round(float(heading) % 360.0, 2)
    write_sidecar(folder, clean, side)
    return side


def set_pin(folder: Path, name: str, *, corners, x: float, y: float) -> dict:
    """Pin a photo to the wall at (x, y) by its four marked corners."""
    clean = safe_name(name)
    if not (folder / clean).is_file():
        raise FileNotFoundError(clean)
    points = [[float(u), float(v)] for u, v in corners]
    if len(points) != 4:
        raise ValueError("a pin needs four corners: top left, top right, bottom right, bottom left")
    side = read_sidecar(folder, clean)
    side["pin"] = {"corners": points, "x": round(float(x), 3), "y": round(float(y), 3)}
    write_sidecar(folder, clean, side)
    return side


def clear_pin(folder: Path, name: str) -> dict:
    clean = safe_name(name)
    side = read_sidecar(folder, clean)
    side.pop("pin", None)
    write_sidecar(folder, clean, side)
    return side


def delete_photo(folder: Path, name: str) -> None:
    clean = safe_name(name)
    for path in (folder / clean, sidecar_path(folder, clean), folder / ".thumbs" / f"{clean}.jpg"):
        if path.is_file():
            path.unlink()


def thumbnail(folder: Path, name: str, size: int = THUMB_PX) -> bytes:
    """A small upright JPEG of a photo, made once and kept."""
    from PIL import Image, ImageOps

    clean = safe_name(name)
    source = folder / clean
    cache = folder / ".thumbs" / f"{clean}.jpg"
    if cache.is_file() and cache.stat().st_mtime >= source.stat().st_mtime:
        return cache.read_bytes()
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((size, size))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=82)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(buffer.getvalue())
    return buffer.getvalue()


def mapillary_points(bbox, cache_dir: Path, *, limit: int = 4000) -> dict:
    """Where Mapillary has photos in `bbox` (RD), for drawing on the map:
    ``{"points", "total", "panos"}``.

    Thinned evenly when there are more than `limit`: the page needs to show
    where the coverage is, not every frame of every dashcam.
    """
    from .mapillary import fetch_image_metadata, find_token, photo_from_record, session_for

    records = fetch_image_metadata(bbox, session=session_for(find_token()), cache_dir=cache_dir)
    points = []
    for record in records:
        photo = photo_from_record(record)
        if photo is None:
            continue
        points.append({
            "x": round(photo.x, 1),
            "y": round(photo.y, 1),
            "pano": photo.camera.kind == "spherical",
            "year": int(photo.captured_at[:4]) if photo.captured_at else None,
        })
    total = len(points)
    panos = sum(1 for p in points if p["pano"])
    if len(points) > limit:
        step = len(points) / limit
        points = [points[int(i * step)] for i in range(limit)]
    return {"points": points, "total": total, "panos": panos}


__all__ = [
    "clear_pin",
    "delete_photo",
    "describe",
    "folder_for",
    "mapillary_points",
    "photos_in",
    "read_sidecar",
    "safe_name",
    "save_upload",
    "set_pin",
    "set_pose",
    "thumbnail",
]
