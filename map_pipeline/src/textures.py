"""Photographed wall surfaces, fetched once and cached.

The generated facade gets its *layout* right — bay spacing, storey lines,
window proportions — but its wall is value noise, and at street level value
noise reads as noise rather than as brick. A photograph of real brickwork,
sampled at its true size, fixes the part the layout cannot.

Poly Haven is the source because its textures are CC0: no attribution
required, no licence to propagate into whatever the model ends up in, and no
API key to manage. Each asset publishes the real-world size it covers, which
is what makes it possible to sample bricks at 210 mm rather than at whatever
size happens to fill the tile — scale is the first thing the eye catches.

Nothing here is required. The download happens once per area, the result is
cached under the work directory, and a run with no network falls back to the
procedural wall, so the pipeline stays repeatable offline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .http_util import get_with_retry, short_error

LOG = logging.getLogger(__name__)

API = "https://api.polyhaven.com"

# One surface per facade style, chosen to match what the style is describing.
# Dutch cities are overwhelmingly brick, which is why most of these are.
#
# Every one of these covers about two metres of wall or more. A one-metre
# texture has to repeat four times across a four-metre tile, and at that rate
# the eye picks out the repeat before it picks out the brick.
WALL_TEXTURES: dict[str, str] = {
    "historic": "medieval_red_brick",
    "interbellum": "brick_wall_10",
    "postwar": "painted_plaster_wall",
    "modern": "brick_wall_006",
    "contemporary": "concrete_wall_008",
    "monumental": "sandstone_blocks_04",
    # A Dutch bedrijfshal is blockwork far more often than it is sheet metal.
    "industrial": "concrete_block_wall",
    "ground_home": "large_red_bricks",
    "ground_retail": "concrete_wall_007",
}

# 1k is more than enough under a 512 px tile, and keeps the download to about a
# megabyte per style rather than fifty.
RESOLUTION = "1k"
FORMAT = "jpg"

LICENCE = "Poly Haven textures are CC0. https://polyhaven.com/license"


class TextureUnavailable(RuntimeError):
    """No photograph could be had, so the caller should draw its own wall."""


def _cache_dir(work_dir: Path) -> Path:
    # Above the per-area directory: the same brick serves every area, and
    # re-downloading it per run would be rude to a service giving it away.
    return work_dir.parent / "_textures"


def _diffuse_entry(files: dict) -> dict:
    """Find the albedo map, whatever this asset happens to call it.

    Poly Haven is not consistent: most assets expose ``Diffuse``, some only
    ``diff_png``, a few use ``albedo``.
    """
    for key in ("Diffuse", "diffuse", "diff", "diff_png", "albedo", "Albedo"):
        entry = files.get(key)
        if isinstance(entry, dict) and entry:
            return entry
    raise TextureUnavailable("asset publishes no diffuse map")


def _pick_file(entry: dict) -> str:
    resolutions = [RESOLUTION] + [r for r in entry if r != RESOLUTION]
    for resolution in resolutions:
        formats = entry.get(resolution) or {}
        for fmt in (FORMAT, "png", "jpg"):
            spec = formats.get(fmt)
            if isinstance(spec, dict) and spec.get("url"):
                return str(spec["url"])
    raise TextureUnavailable("asset publishes no downloadable diffuse map")


def fetch_texture(slug: str, work_dir: Path, *, timeout_s: float = 60.0) -> Path:
    """Download one texture's albedo, or return the copy already on disk."""
    cache = _cache_dir(work_dir)
    cache.mkdir(parents=True, exist_ok=True)

    for existing in cache.glob(f"{slug}_{RESOLUTION}.*"):
        return existing

    meta_path = cache / f"{slug}.json"
    if meta_path.is_file():
        files = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        response = get_with_retry(
            f"{API}/files/{slug}",
            timeout=timeout_s,
            max_retries=2,
            description=f"texture metadata for {slug}",
        )
        files = response.json()
        meta_path.write_text(json.dumps(files), encoding="utf-8")

    url = _pick_file(_diffuse_entry(files))
    suffix = Path(url).suffix or ".jpg"
    path = cache / f"{slug}_{RESOLUTION}{suffix}"

    response = get_with_retry(
        url,
        timeout=timeout_s,
        max_retries=2,
        expect_binary=True,
        description=f"texture {slug}",
    )
    path.write_bytes(response.content)
    LOG.info("cached %s (%.1f MB)", path.name, len(response.content) / 1e6)
    return path


def texture_size_m(slug: str, work_dir: Path, *, timeout_s: float = 60.0) -> tuple[float, float]:
    """How much wall the photograph covers, in metres.

    Without this the bricks come out at whatever size fills the tile, which is
    the difference between a wall and a photograph of a wall.
    """
    cache = _cache_dir(work_dir)
    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / "assets.json"

    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        response = get_with_retry(
            f"{API}/assets?t=textures",
            timeout=timeout_s,
            max_retries=2,
            description="texture index",
        )
        index = response.json()
        index_path.write_text(json.dumps(index), encoding="utf-8")

    dimensions = (index.get(slug) or {}).get("dimensions")
    if not dimensions or len(dimensions) < 2:
        # A sane default beats failing: most wall textures are a couple of
        # metres square.
        return 2.0, 2.0
    return float(dimensions[0]) / 1000.0, float(dimensions[1]) / 1000.0


def _tile_to(pixels: np.ndarray, size_px: int, repeats_x: float, repeats_y: float) -> np.ndarray:
    """Resample a texture so it repeats the given number of times across a tile.

    Sampling with wrapped indices rather than stitching whole copies means a
    fractional repeat count works, which it has to: a 4 m tile over a 3 m
    photograph is 1.33 repeats, not 1 or 2.
    """
    height, width = pixels.shape[:2]
    ys = (np.arange(size_px) * (repeats_y * height / size_px)).astype(int) % height
    xs = (np.arange(size_px) * (repeats_x * width / size_px)).astype(int) % width
    return pixels[np.ix_(ys, xs)]


def _tint(pixels: np.ndarray, target: tuple[int, int, int], strength: float) -> np.ndarray:
    """Pull a photograph towards a style's colour without flattening its grain.

    Each channel is scaled so the mean lands on the target, which moves the
    colour while leaving every brick, joint and stain exactly where it was.
    Blending back towards the original by ``strength`` keeps some of the
    photograph's own character.
    """
    if strength <= 0.0:
        return pixels

    mean = pixels.reshape(-1, 3).mean(axis=0)
    mean[mean < 1.0] = 1.0
    scaled = pixels * (np.asarray(target, dtype=np.float64) / mean)
    return np.clip(pixels * (1.0 - strength) + scaled * strength, 0, 255)


def wall_base(
    style,
    size_px: int,
    work_dir: Path,
    *,
    tile_width_m: float,
    tile_height_m: float,
    tint_strength: float = 0.6,
    timeout_s: float = 60.0,
) -> np.ndarray:
    """A photographed wall for one style, tiled and tinted to fit its tile.

    Raises :class:`TextureUnavailable` when there is nothing to load, which is
    the caller's cue to draw the procedural wall instead.
    """
    slug = WALL_TEXTURES.get(style.name)
    if not slug:
        raise TextureUnavailable(f"no texture is mapped to the {style.name} style")

    from PIL import Image

    path = fetch_texture(slug, work_dir, timeout_s=timeout_s)
    photo_w, photo_h = texture_size_m(slug, work_dir, timeout_s=timeout_s)

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64)

    tiled = _tile_to(
        pixels,
        size_px,
        repeats_x=max(tile_width_m / max(photo_w, 0.1), 0.05),
        repeats_y=max(tile_height_m / max(photo_h, 0.1), 0.05),
    )
    return _tint(tiled, style.wall_rgb, tint_strength)


def load_wall_bases(
    styles: list,
    size_px: int,
    work_dir: Path,
    *,
    tile_heights: list[float],
    tint_strength: float = 0.6,
    timeout_s: float = 60.0,
) -> dict[int, np.ndarray]:
    """Fetch what is available, and say plainly what is not.

    Partial success is the normal case worth supporting: one asset moving or
    one request timing out should cost that one style its photograph, not the
    whole run.
    """
    bases: dict[int, np.ndarray] = {}
    missed: list[str] = []

    for index, style in enumerate(styles):
        try:
            bases[index] = wall_base(
                style,
                size_px,
                work_dir,
                tile_width_m=style.tile_width_m,
                tile_height_m=tile_heights[index],
                tint_strength=tint_strength,
                timeout_s=timeout_s,
            )
        except TextureUnavailable as exc:
            missed.append(f"{style.name} ({exc})")
        except Exception as exc:  # noqa: BLE001 - never fail a run over a texture
            missed.append(f"{style.name} ({short_error(exc)})")

    if bases:
        LOG.info(
            "photographed wall surfaces for %d of %d styles", len(bases), len(styles)
        )
    if missed:
        LOG.warning(
            "drawing the wall for %s; %s",
            ", ".join(missed),
            "check the network, or set facade.photo_textures to false to stop trying",
        )
    return bases


__all__ = [
    "LICENCE",
    "RESOLUTION",
    "TextureUnavailable",
    "WALL_TEXTURES",
    "fetch_texture",
    "load_wall_bases",
    "texture_size_m",
    "wall_base",
]
