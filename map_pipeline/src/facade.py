"""Procedural facade textures, generated offline with Pillow.

The aerial photo never sees a wall, so facades have to be fabricated. The plan
called for a Blender node material, but node graphs do not survive FBX export:
Unity would receive a flat grey material and the walls would lose their pattern.
Generating the same pattern into an image keeps it procedural while staying
inside what FBX can carry, which is a texture reference.

One tile covers ``tile_width_m`` across and one storey up. Wall UVs are built
from metres, so windows land on storey lines whatever the building height, which
is what the node material was for.
"""

from __future__ import annotations

import colorsys
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class FacadeStyle:
    """One facade look: wall colour, window layout, trim."""

    name: str
    wall_rgb: tuple[int, int, int]
    trim_rgb: tuple[int, int, int]
    glass_rgb: tuple[int, int, int]
    windows_across: int
    window_width_frac: float
    window_height_frac: float
    sill_frac: float
    grain: float


# Ordered from low-rise brick to high-rise glass. Buildings are matched to a
# style by height, so taller stock reads as more modern.
STYLES: tuple[FacadeStyle, ...] = (
    FacadeStyle(
        name="brick",
        wall_rgb=(150, 96, 78),
        trim_rgb=(226, 224, 218),
        glass_rgb=(66, 82, 94),
        windows_across=2,
        window_width_frac=0.30,
        window_height_frac=0.46,
        sill_frac=0.22,
        grain=0.10,
    ),
    FacadeStyle(
        name="plaster",
        wall_rgb=(198, 190, 176),
        trim_rgb=(238, 236, 232),
        glass_rgb=(74, 88, 100),
        windows_across=2,
        window_width_frac=0.32,
        window_height_frac=0.50,
        sill_frac=0.20,
        grain=0.06,
    ),
    FacadeStyle(
        name="concrete",
        wall_rgb=(166, 166, 162),
        trim_rgb=(206, 208, 210),
        glass_rgb=(60, 76, 90),
        windows_across=3,
        window_width_frac=0.24,
        window_height_frac=0.52,
        sill_frac=0.18,
        grain=0.05,
    ),
    FacadeStyle(
        name="glass",
        wall_rgb=(108, 118, 128),
        trim_rgb=(150, 158, 166),
        glass_rgb=(84, 108, 126),
        windows_across=3,
        window_width_frac=0.28,
        window_height_frac=0.66,
        sill_frac=0.10,
        grain=0.03,
    ),
)


def _value_noise(shape: tuple[int, int], cells: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth tiling noise in 0..1, used to break up flat colour."""
    coarse = rng.random((cells, cells))
    # Wrap one row and column so the upsampled result tiles seamlessly.
    wrapped = np.pad(coarse, ((0, 1), (0, 1)), mode="wrap")

    ys = np.linspace(0, cells, shape[0], endpoint=False)
    xs = np.linspace(0, cells, shape[1], endpoint=False)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]
    # Smoothstep keeps the interpolation from looking like a grid of ramps.
    ty = ty * ty * (3 - 2 * ty)
    tx = tx * tx * (3 - 2 * tx)

    v00 = wrapped[np.ix_(y0, x0)]
    v01 = wrapped[np.ix_(y0, x0 + 1)]
    v10 = wrapped[np.ix_(y0 + 1, x0)]
    v11 = wrapped[np.ix_(y0 + 1, x0 + 1)]

    return (
        v00 * (1 - tx) * (1 - ty)
        + v01 * tx * (1 - ty)
        + v10 * (1 - tx) * ty
        + v11 * tx * ty
    )


def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Lighten or darken a colour while keeping its hue."""
    h, l, s = colorsys.rgb_to_hls(*[c / 255.0 for c in rgb])
    l = float(np.clip(l * factor, 0.0, 1.0))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def render_facade_tile(
    style: FacadeStyle, size_px: int, seed: int = 0
) -> "np.ndarray":
    """Render one seamlessly tiling storey of facade as an RGB array.

    The tile wraps in both directions: windows sit inside the tile with a margin,
    and the storey band runs along the bottom edge, so repeating the image in U
    and V produces a continuous wall.
    """
    rng = np.random.default_rng(seed)
    height = width = size_px

    canvas = np.zeros((height, width, 3), dtype=np.float64)
    canvas[:, :] = style.wall_rgb

    # Wall grain.
    noise = _value_noise((height, width), cells=max(4, size_px // 32), rng=rng)
    fine = _value_noise((height, width), cells=max(8, size_px // 8), rng=rng)
    combined = 0.65 * noise + 0.35 * fine
    canvas *= 1.0 + style.grain * (combined - 0.5)[:, :, None] * 2.0

    # Storey band along the bottom edge, which becomes the line between floors.
    band_px = max(2, int(round(0.055 * height)))
    canvas[:band_px, :] = _shade(style.wall_rgb, 0.80)
    canvas[band_px : band_px + max(1, band_px // 2), :] = _shade(style.wall_rgb, 1.10)

    # Windows.
    cell_width = width / style.windows_across
    # Capped against the cell so windows always keep a wall margin, which is
    # what makes the tile wrap seamlessly.
    win_w = min(style.window_width_frac * width, cell_width * 0.8)
    win_h = style.window_height_frac * height
    sill = style.sill_frac * height
    frame_px = max(1, int(round(0.012 * size_px)))

    for column in range(style.windows_across):
        cx = (column + 0.5) * cell_width
        x0 = int(round(cx - win_w / 2))
        x1 = int(round(cx + win_w / 2))
        y0 = int(round(sill))
        y1 = int(round(sill + win_h))
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue

        # Frame first, then glass inset into it.
        canvas[y0:y1, x0:x1] = style.trim_rgb

        gx0, gx1 = x0 + frame_px, x1 - frame_px
        gy0, gy1 = y0 + frame_px, y1 - frame_px
        if gx1 - gx0 < 2 or gy1 - gy0 < 2:
            continue

        glass = np.zeros((gy1 - gy0, gx1 - gx0, 3), dtype=np.float64)
        glass[:, :] = style.glass_rgb
        # A vertical gradient reads as sky reflected in the upper pane.
        ramp = np.linspace(1.35, 0.72, gy1 - gy0)[::-1][:, None, None]
        glass *= ramp
        # Faint per-window variation so a wall is not a perfect grid.
        glass *= 1.0 + 0.06 * (rng.random() - 0.5)
        canvas[gy0:gy1, gx0:gx1] = glass

        # Mullion splitting the pane, skipped when the window is small.
        if gx1 - gx0 > 8 * frame_px:
            mid = (gx0 + gx1) // 2
            canvas[gy0:gy1, mid : mid + frame_px] = style.trim_rgb
        if gy1 - gy0 > 8 * frame_px:
            mid = (gy0 + gy1) // 2
            canvas[mid : mid + frame_px, gx0:gx1] = style.trim_rgb

        # Sill under the window.
        sill_y0 = max(0, y0 - max(1, frame_px))
        canvas[sill_y0:y0, x0:x1] = _shade(style.trim_rgb, 0.88)

    return np.clip(canvas, 0, 255).astype(np.uint8)


def style_for_height(height_m: float, n_variants: int) -> int:
    """Pick a style index for a building height.

    Low buildings get brick, tall ones get glass, which matches how Dutch urban
    stock actually reads from the street.
    """
    if n_variants <= 1:
        return 0
    thresholds = (9.0, 18.0, 32.0)
    index = sum(height_m >= t for t in thresholds)
    return int(min(index, n_variants - 1))


def generate_facade_textures(
    work_dir: Path, *, facade_cfg: dict
) -> list[Path]:
    """Write one facade PNG per variant and return the paths in style order."""
    from PIL import Image

    work_dir.mkdir(parents=True, exist_ok=True)
    size_px = int(facade_cfg["texture_px"])
    variants = min(int(facade_cfg["variants"]), len(STYLES))
    seed = int(facade_cfg["seed"])

    paths: list[Path] = []
    for index in range(variants):
        style = STYLES[index]
        pixels = render_facade_tile(style, size_px, seed=seed + index)
        # Row 0 of the array is the bottom of the tile in UV space, but PNG rows
        # run top-down, so flip on the way out.
        image = Image.fromarray(pixels[::-1], mode="RGB")
        name = "facade.png" if variants == 1 else f"facade_{index:02d}_{style.name}.png"
        path = work_dir / name
        image.save(path, format="PNG")
        paths.append(path)
        LOG.info("wrote %s (%s, %dpx)", path, style.name, size_px)

    return paths


__all__ = [
    "STYLES",
    "FacadeStyle",
    "generate_facade_textures",
    "render_facade_tile",
    "style_for_height",
]
