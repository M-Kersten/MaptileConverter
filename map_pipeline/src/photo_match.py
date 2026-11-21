"""Side by side: a street photo, and the model seen from the same spot.

The last check on everything the photos feed into the model. A camera is
placed where a checked photo was taken, looking at one wall, and two
pictures are made of the same view: one cut from the photo, one rendered
from the model. Put next to each other they show at a glance whether the
model's walls are where the photo's are, and whether what was read off the
photo -- colours, windows, storeys, and later the photo textures -- looks
like the place.

360 photos are cut into an ordinary perspective view first, through the
same camera code that lined them up, so the two pictures are the same
projection pixel for pixel.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .camera import PERSPECTIVE, Camera, project_world, rotation_from_heading
from .facade_capture import _bilinear
from .photos import Photo

LOG = logging.getLogger(__name__)


def view_towards(photo: Photo, target: np.ndarray, *, hfov_deg: float = 80.0, width: int = 960, height: int = 720):
    """A perspective camera at the photo, looking at `target` (RD/NAP)."""
    d = np.asarray(target, dtype=np.float64) - photo.centre
    heading = float(np.degrees(np.arctan2(d[0], d[1])))
    pitch = float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))
    focal = 0.5 / np.tan(np.radians(hfov_deg) / 2.0)
    camera = Camera(PERSPECTIVE, width, height, focal)
    return camera, rotation_from_heading(heading, pitch)


def cut_view(photo: Photo, image: np.ndarray, camera: Camera, rotation: np.ndarray) -> np.ndarray:
    """The part of a photo a virtual camera at the same spot would see."""
    from .camera import unproject_camera

    h, w = camera.height, camera.width
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = unproject_camera(camera, np.column_stack([cols.ravel(), rows.ravel()]))
    world = rays @ rotation  # camera to world: R is world to camera
    source = photo.camera.scaled(image.shape[1], image.shape[0])
    points = photo.centre + world * 10.0
    pixels, ok = project_world(source, photo.rotation, photo.centre, points)
    out = np.zeros((h * w, 3), dtype=np.float32)
    out[ok] = _bilinear(image, pixels[ok, 0], pixels[ok, 1], wrap=source.kind == "spherical")
    return out.reshape(h, w, 3)


def blender_view(name: str, photo: Photo, camera: Camera, rotation: np.ndarray, origin_rd, z_offset: float) -> dict:
    """The camera in the model's local frame and Blender's convention.

    Blender's camera looks down its own -Z with +Y up, so its axes in the
    world are right, up and backwards: the photo's right, minus its down,
    minus its forward.
    """
    right, down, forward = rotation[0], rotation[1], rotation[2]
    matrix = np.column_stack([right, -down, -forward])
    location = [
        float(photo.x - origin_rd[0]), float(photo.y - origin_rd[1]), float(photo.z - z_offset),
    ]
    return {
        "name": name,
        "location": location,
        "rotation": matrix.tolist(),
        "hfov_deg": float(np.degrees(2.0 * np.arctan(0.5 / camera.focal))),
        "width": camera.width,
        "height": camera.height,
    }


def render_matches(
    pairs: list[tuple[str, Photo, np.ndarray, np.ndarray]],
    blend_path: Path,
    out_dir: Path,
    *,
    origin_rd,
    z_offset: float,
    blender_command=None,
) -> list[Path]:
    """Photo cut-outs and model renders, side by side, one file per pair.

    `pairs` are ``(name, photo, photo pixels, target point)``.
    `blender_command(script, args)` runs a Blender script; by default the
    bpy module in a subprocess.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    views, cuts = [], {}
    for name, photo, image, target in pairs:
        camera, rotation = view_towards(photo, target)
        cuts[name] = cut_view(photo, image, camera, rotation)
        views.append(blender_view(name, photo, camera, rotation, origin_rd, z_offset))
    if not views:
        return []
    renders = out_dir / "_renders"
    renders.mkdir(parents=True, exist_ok=True)
    views_file = renders / "views.json"
    views_file.write_text(json.dumps(views), encoding="utf-8")
    script = Path(__file__).resolve().parent.parent / "blender" / "photo_match.py"
    args = ["--blend", str(blend_path), "--views", str(views_file), "--out", str(renders)]
    if blender_command is None:
        import subprocess
        import sys

        subprocess.run([sys.executable, str(script), *args], check=True, capture_output=True, text=True)
    else:
        blender_command(script, args)

    written = []
    for name in cuts:
        render_path = renders / f"{name}.png"
        if not render_path.is_file():
            continue
        left = Image.fromarray(np.clip(cuts[name], 0, 255).astype(np.uint8))
        right = Image.open(render_path).convert("RGB")
        pair = Image.new("RGB", (left.width + right.width + 8, max(left.height, right.height)), (0, 0, 0))
        pair.paste(left, (0, 0))
        pair.paste(right, (left.width + 8, 0))
        path = out_dir / f"match_{name}.jpg"
        pair.save(path, quality=85)
        written.append(path)
    return written


def choose_matches(result, textures=None, limit: int = 4):
    """Walls worth comparing: those wearing a photo first, then those read.

    Returns ``[(name, photo, target point)]`` using each wall's best checked
    view whose pixels are on disk.
    """
    coverage = result.coverage
    if coverage is None:
        return []
    by_id = {p.id: p for p in result.photos}
    textured = set(textures.placements) if textures is not None else set()
    read = {w.face for m in getattr(result, "facades", {}).values() for w in m.walls}
    order = sorted(
        range(len(coverage.faces)),
        key=lambda i: (coverage.faces[i].key not in textured, i not in read, -coverage.faces[i].area_m2),
    )
    chosen, used_photos = [], set()
    for i in order:
        face = coverage.faces[i]
        if face.key not in textured and i not in read:
            break
        for view in coverage.views[i]:
            photo = by_id.get(view.photo)
            if view.verified is True and photo is not None and photo.path and photo.id not in used_photos:
                chosen.append((str(len(chosen) + 1), photo, face.centre))
                used_photos.add(photo.id)
                break
        if len(chosen) >= limit:
            break
    return chosen


def write_photo_matches(result, textures, blend_path: Path, out_dir: Path, *, origin_rd, z_offset, blender_command=None, limit: int = 4) -> list[Path]:
    """Photo and model side by side for a few walls, into photo_checks/."""
    from .facade_capture import PhotoPixels

    pixels = PhotoPixels(max_cached=2)
    pairs = []
    for name, photo, target in choose_matches(result, textures, limit):
        try:
            pairs.append((name, photo, pixels.get(photo.path), target))
        except OSError:
            continue
    checks = out_dir / "photo_checks"
    for old in checks.glob("match_*.jpg"):
        old.unlink()
    if not pairs:
        return []
    written = render_matches(
        pairs, blend_path, checks, origin_rd=origin_rd, z_offset=z_offset,
        blender_command=blender_command,
    )
    import shutil

    shutil.rmtree(checks / "_renders", ignore_errors=True)
    return written


__all__ = [
    "blender_view",
    "choose_matches",
    "cut_view",
    "render_matches",
    "view_towards",
    "write_photo_matches",
]
