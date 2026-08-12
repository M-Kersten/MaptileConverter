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
    # Newest construction year this style covers. 3DBAG carries the original
    # build year on every building, and era predicts how a facade looks far
    # better than height does: a 1890s canal house and a 1970s office block can
    # be the same height and look nothing alike.
    era_until: int = 3000
    label: str = ""
    # Round the window head. A church or a warehouse arch is the single
    # clearest cue that a wall is not a stack of domestic storeys.
    arch: float = 0.0
    # Draw the horizontal band between storeys. Off for buildings that have no
    # storeys to divide.
    storey_band: bool = True
    # Metres of wall one tile covers. Wide for industrial bays, narrow for a
    # house front.
    tile_width_m: float = 4.0
    # Metres of wall one tile covers vertically. Only used where the facade is
    # not divided into storeys.
    tile_height_m: float = 0.0


# Ordered oldest to newest. A building picks the first style whose era covers
# its construction year.
STYLES: tuple[FacadeStyle, ...] = (
    FacadeStyle(
        name="historic",
        label="pre-1920 brick, tall narrow windows",
        era_until=1920,
        wall_rgb=(138, 84, 68),
        trim_rgb=(232, 230, 224),
        glass_rgb=(58, 72, 84),
        windows_across=2,
        window_width_frac=0.26,
        window_height_frac=0.56,
        sill_frac=0.18,
        grain=0.13,
    ),
    FacadeStyle(
        name="interbellum",
        label="1920-1945 brick",
        era_until=1945,
        wall_rgb=(158, 102, 82),
        trim_rgb=(226, 224, 218),
        glass_rgb=(66, 82, 94),
        windows_across=2,
        window_width_frac=0.31,
        window_height_frac=0.46,
        sill_frac=0.22,
        grain=0.10,
    ),
    FacadeStyle(
        name="postwar",
        label="1945-1975 plaster and concrete",
        era_until=1975,
        wall_rgb=(196, 188, 174),
        trim_rgb=(238, 236, 232),
        glass_rgb=(74, 88, 100),
        windows_across=3,
        window_width_frac=0.26,
        window_height_frac=0.50,
        sill_frac=0.20,
        grain=0.06,
    ),
    FacadeStyle(
        name="modern",
        label="1975-2000 brick and panel",
        era_until=2000,
        wall_rgb=(170, 148, 130),
        trim_rgb=(214, 214, 212),
        glass_rgb=(64, 80, 94),
        windows_across=3,
        window_width_frac=0.28,
        window_height_frac=0.54,
        sill_frac=0.16,
        grain=0.07,
    ),
    FacadeStyle(
        name="contemporary",
        label="2000 onwards, glass and panel",
        era_until=3000,
        wall_rgb=(112, 122, 132),
        trim_rgb=(152, 160, 168),
        glass_rgb=(86, 110, 128),
        windows_across=3,
        window_width_frac=0.30,
        window_height_frac=0.66,
        sill_frac=0.10,
        grain=0.03,
    ),
)

# Two compositions that the era styles get badly wrong, because both describe
# buildings without ordinary storeys.
#
# A church or tower has one tall volume with arched openings, not a stack of
# three-metre rows. Before this the Dom came out as thirty-six storeys of
# domestic windows. A shed has a handful of large bays and long blank walls.
EXTRA_STYLES: dict[str, FacadeStyle] = {
    "monumental": FacadeStyle(
        name="monumental",
        label="church, tower or civic hall: stone with arched openings",
        wall_rgb=(150, 142, 124),
        trim_rgb=(196, 190, 176),
        glass_rgb=(48, 54, 58),
        windows_across=2,
        window_width_frac=0.22,
        window_height_frac=0.60,
        sill_frac=0.16,
        grain=0.13,
        arch=0.85,
        storey_band=False,
        # One tile spans a whole bay of a church wall, not a storey.
        tile_width_m=7.0,
        tile_height_m=9.0,
    ),
    "industrial": FacadeStyle(
        name="industrial",
        label="shed or depot: wide bays, mostly blank wall",
        wall_rgb=(158, 158, 154),
        trim_rgb=(188, 190, 190),
        glass_rgb=(96, 108, 112),
        windows_across=1,
        window_width_frac=0.46,
        window_height_frac=0.34,
        sill_frac=0.42,
        grain=0.05,
        storey_band=False,
        tile_width_m=9.0,
        tile_height_m=6.0,
    ),
}


# The ground storey is what makes a street read as a street: shopfronts and
# doors rather than another row of the same windows. It is a separate material
# because a wall has to be split at the first-floor line to use it.
#
# Which one a building gets comes from its BAG function. Shopfronts along a
# residential street look as wrong as a blank wall along a shopping street, so
# housing gets a door-and-window ground floor instead.
GROUND_STYLES: tuple[FacadeStyle, ...] = (
    FacadeStyle(
        name="ground_home",
        label="ground floor, doors and windows",
        wall_rgb=(146, 116, 96),
        trim_rgb=(228, 226, 220),
        glass_rgb=(56, 68, 78),
        windows_across=2,
        window_width_frac=0.26,
        window_height_frac=0.46,
        sill_frac=0.20,
        grain=0.09,
    ),
    FacadeStyle(
        name="ground_retail",
        label="ground floor, shopfronts",
        wall_rgb=(150, 142, 134),
        trim_rgb=(224, 222, 216),
        glass_rgb=(58, 70, 80),
        windows_across=2,
        window_width_frac=0.40,
        window_height_frac=0.66,
        sill_frac=0.05,
        grain=0.05,
    ),
)

# Kept for callers that want the single default ground storey.
GROUND_STYLE = GROUND_STYLES[1]


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


# Relief heights for the normal map, 0 = deepest. Glass sits behind its frame,
# the storey band and the sills stand proud of the wall.
RELIEF_WALL = 0.50
RELIEF_BAND = 0.62
RELIEF_FRAME = 0.45
RELIEF_GLASS = 0.28
RELIEF_SILL = 0.60
# How far the relief actually stands out, in tile-widths. Small: a facade is
# nearly flat, and overdoing this makes windows look like portholes.
RELIEF_DEPTH = 0.035


def render_facade_layers(
    style: FacadeStyle,
    size_px: int,
    seed: int = 0,
    base: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one storey of facade as ``(rgb, relief)``.

    The tile wraps in both directions: windows sit inside the tile with a margin,
    and the storey band runs along the bottom edge, so repeating the image in U
    and V produces a continuous wall. The relief channel drives the normal map,
    so both come out of the same geometry and stay in register.

    ``base`` replaces the flat colour and its noise with a photograph of real
    masonry, already tiled to this style's tile. Everything drawn on top of the
    wall — the windows, the band, the sills — is the same either way, because
    those are the parts a photograph of a blank wall cannot supply.
    """
    rng = np.random.default_rng(seed)
    height = width = size_px

    canvas = np.zeros((height, width, 3), dtype=np.float64)
    relief = np.full((height, width), RELIEF_WALL, dtype=np.float64)

    if base is not None:
        canvas[:, :] = base[:height, :width]
    else:
        canvas[:, :] = style.wall_rgb
        # Wall grain, standing in for the masonry a photograph would show.
        noise = _value_noise((height, width), cells=max(4, size_px // 32), rng=rng)
        fine = _value_noise((height, width), cells=max(8, size_px // 8), rng=rng)
        combined = 0.65 * noise + 0.35 * fine
        canvas *= 1.0 + style.grain * (combined - 0.5)[:, :, None] * 2.0

    # The wall as it stands before anything is cut into it. Restoring from this
    # rather than from the flat colour is what lets an arched head keep the
    # masonry around its curve.
    wall_layer = canvas.copy()

    # Storey band along the bottom edge, which becomes the line between floors.
    if style.storey_band:
        band_px = max(2, int(round(0.055 * height)))
        # Shading the wall in place keeps the band made of the same brick,
        # instead of laying a flat stripe over a photograph.
        canvas[:band_px, :] = np.clip(canvas[:band_px, :] * 0.80, 0, 255)
        lip = band_px + max(1, band_px // 2)
        canvas[band_px:lip, :] = np.clip(canvas[band_px:lip, :] * 1.10, 0, 255)
        relief[:band_px, :] = RELIEF_BAND

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

        # A rounded head turns a domestic window into a church or warehouse
        # opening. The arch occupies the top of the opening, and everything
        # outside its curve stays wall.
        arch_mask = None
        if style.arch > 0:
            span = x1 - x0
            rise = int(round(style.arch * span * 0.5))
            head = min(y1, y0 + int(round(win_h)))
            arch_top = head
            arch_base = max(y0, head - rise)
            if arch_base < arch_top:
                rows = np.arange(arch_base, arch_top)[:, None]
                cols = np.arange(x0, x1)[None, :]
                centre_x = (x0 + x1 - 1) * 0.5
                radius = span * 0.5
                # Measured from the centre of the springing line: the opening
                # is the half-disc above it, so it is full width where the
                # curve starts and closes to nothing at the crown. Row 0 of
                # this array is the bottom of the tile, so dy grows upwards.
                dy = (rows - arch_base) / max(1, arch_top - arch_base)
                dx = np.abs(cols - centre_x) / max(1e-6, radius)
                outside = (dx ** 2 + dy ** 2) > 1.0
                arch_mask = (arch_base, arch_top, outside)

        # Frame first, then glass inset into it.
        canvas[y0:y1, x0:x1] = style.trim_rgb
        relief[y0:y1, x0:x1] = RELIEF_FRAME

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
        relief[gy0:gy1, gx0:gx1] = RELIEF_GLASS

        # Mullion splitting the pane, skipped when the window is small.
        if gx1 - gx0 > 8 * frame_px:
            mid = (gx0 + gx1) // 2
            canvas[gy0:gy1, mid : mid + frame_px] = style.trim_rgb
            relief[gy0:gy1, mid : mid + frame_px] = RELIEF_FRAME
        # An arched opening is one tall light. A crossbar through it splits the
        # arch off and makes it read as a disc sitting on a rectangle.
        if gy1 - gy0 > 8 * frame_px and not style.arch:
            mid = (gy0 + gy1) // 2
            canvas[mid : mid + frame_px, gx0:gx1] = style.trim_rgb
            relief[mid : mid + frame_px, gx0:gx1] = RELIEF_FRAME

        # Sill under the window.
        sill_y0 = max(0, y0 - max(1, frame_px))
        canvas[sill_y0:y0, x0:x1] = _shade(style.trim_rgb, 0.88)
        relief[sill_y0:y0, x0:x1] = RELIEF_SILL

        # Put the wall back outside the arch, so the opening reads as rounded
        # rather than as a rectangle with a curve drawn on it.
        if arch_mask is not None:
            arch_base, arch_top, outside = arch_mask
            block = canvas[arch_base:arch_top, x0:x1]
            wall = wall_layer[arch_base:arch_top, x0:x1]
            canvas[arch_base:arch_top, x0:x1] = np.where(
                outside[:, :, None], wall, block
            )
            relief_block = relief[arch_base:arch_top, x0:x1]
            relief[arch_base:arch_top, x0:x1] = np.where(
                outside, RELIEF_WALL, relief_block
            )

    return np.clip(canvas, 0, 255).astype(np.uint8), relief


def render_facade_tile(style: FacadeStyle, size_px: int, seed: int = 0) -> np.ndarray:
    """The colour channel of :func:`render_facade_layers`."""
    return render_facade_layers(style, size_px, seed)[0]


def relief_to_normal_map(relief: np.ndarray, depth: float = RELIEF_DEPTH) -> np.ndarray:
    """Turn a relief field into a tangent-space normal map.

    Uses the OpenGL convention, green pointing up, which is what Blender and
    Unity both expect. Gradients wrap, so the normal map tiles exactly like the
    colour it came from.
    """
    scale = depth * relief.shape[0]
    # np.roll wraps, which keeps the seams of a tiling texture consistent.
    d_dx = (np.roll(relief, -1, axis=1) - np.roll(relief, 1, axis=1)) * 0.5 * scale
    d_dy = (np.roll(relief, -1, axis=0) - np.roll(relief, 1, axis=0)) * 0.5 * scale

    # Surface normal of a height field is (-dz/dx, -dz/dy, 1), normalised.
    normal = np.stack([-d_dx, -d_dy, np.ones_like(relief)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    return np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def wall_styles(n_variants: int) -> list[FacadeStyle]:
    """Wall styles in slot order: the eras, then the archetype specials.

    The extras go last so their indices stay put when the number of era
    variants changes.
    """
    return list(STYLES[:n_variants]) + [
        EXTRA_STYLES["monumental"],
        EXTRA_STYLES["industrial"],
    ]


def style_for_archetype(archetype: int, n_variants: int) -> int | None:
    """Slot for an archetype that needs its own composition, else None.

    Only the two the era styles get wrong are special-cased. Housing, offices
    and shops are all stacks of storeys, so era is the better predictor for
    them and they keep it.
    """
    # Matches src/buildings.py.
    ARCH_INDUSTRIAL, ARCH_MONUMENTAL = 4, 5
    if archetype == ARCH_MONUMENTAL:
        return n_variants
    if archetype == ARCH_INDUSTRIAL:
        return n_variants + 1
    return None


def style_for_building(
    height_m: float, build_year: int | None, n_variants: int
) -> int:
    """Pick a facade style for one building.

    Construction year decides it when 3DBAG supplies one, which it does for
    every building in practice. Height is only the fallback, and a poor proxy:
    era is what actually determines whether a wall is dark brick or glass.
    """
    if n_variants <= 1:
        return 0

    if build_year:
        for index, style in enumerate(STYLES[:n_variants]):
            if build_year <= style.era_until:
                return index
        return n_variants - 1

    thresholds = (9.0, 18.0, 32.0)
    return int(min(sum(height_m >= t for t in thresholds), n_variants - 1))


def style_for_height(height_m: float, n_variants: int) -> int:
    """Height-only style pick, kept for callers with no build year."""
    return style_for_building(height_m, None, n_variants)


def _wall_bases(
    styles: list[FacadeStyle], size_px: int, work_dir: Path, facade_cfg: dict
) -> dict[int, np.ndarray]:
    """Photographed masonry per style, when the run is asked for it.

    How much wall one tile covers vertically is what decides the brick scale,
    and it differs by style: an ordinary storey, or a whole church bay.
    """
    if not bool(facade_cfg.get("photo_textures", True)):
        return {}

    from .textures import load_wall_bases

    storey = float(facade_cfg.get("floor_height_m", 3.0))
    ground = float(facade_cfg.get("ground_floor_height_m", 3.6))
    heights = [
        style.tile_height_m
        or (ground if style.name.startswith("ground") else storey)
        for style in styles
    ]
    return load_wall_bases(
        styles,
        size_px,
        work_dir,
        tile_heights=heights,
        tint_strength=float(facade_cfg.get("photo_tint", 0.6)),
    )


def generate_facade_textures(
    work_dir: Path, *, facade_cfg: dict
) -> list[tuple[Path, Path | None]]:
    """Write the facade textures, returning ``(colour, normal)`` per variant.

    When the ground storey is enabled its texture comes last, so the Blender
    stage can address it as the final material slot.

    The normal map is named ``*_normal.png`` because Unity's importer keys off
    that suffix to set the texture type automatically.
    """
    from PIL import Image

    work_dir.mkdir(parents=True, exist_ok=True)
    size_px = int(facade_cfg["texture_px"])
    variants = min(int(facade_cfg["variants"]), len(STYLES))
    seed = int(facade_cfg["seed"])
    want_normal = bool(facade_cfg.get("normal_map", True))

    styles = wall_styles(variants)
    ground_styles: list[FacadeStyle] = []
    if bool(facade_cfg.get("ground_floor", True)):
        # Both ground variants are written when building function is available,
        # so a residential street does not get a parade of shopfronts.
        ground_styles = (
            list(GROUND_STYLES)
            if facade_cfg.get("ground_by_function")
            else [GROUND_STYLE]
        )
        styles.extend(ground_styles)
    single_ground = len(ground_styles) == 1

    bases = _wall_bases(styles, size_px, work_dir, facade_cfg)

    paths: list[tuple[Path, Path | None]] = []
    for index, style in enumerate(styles):
        pixels, relief = render_facade_layers(
            style, size_px, seed=seed + index, base=bases.get(index)
        )

        if style.name.startswith("ground"):
            # With nothing to choose between, the one ground storey is just
            # "ground" rather than being labelled with a function it does not
            # actually know.
            stem = "facade_ground" if single_ground else f"facade_{style.name}"
        elif style.name in EXTRA_STYLES:
            stem = f"facade_{index:02d}_{style.name}"
        elif variants == 1:
            stem = "facade"
        else:
            stem = f"facade_{index:02d}_{style.name}"
        # Row 0 of the array is the bottom of the tile in UV space, but PNG rows
        # run top-down, so flip on the way out.
        colour_path = work_dir / f"{stem}.png"
        Image.fromarray(pixels[::-1], mode="RGB").save(colour_path, format="PNG")

        normal_path = None
        if want_normal:
            normal = relief_to_normal_map(relief, float(facade_cfg["relief_depth"]))
            normal_path = work_dir / f"{stem}_normal.png"
            Image.fromarray(normal[::-1], mode="RGB").save(normal_path, format="PNG")

        paths.append((colour_path, normal_path))
        LOG.info(
            "wrote %s (%s, %dpx)%s",
            colour_path.name,
            style.name,
            size_px,
            " with normal map" if normal_path else "",
        )

    return paths


def generate_tree_texture(work_dir: Path, size_px: int = 512, seed: int = 7) -> Path:
    """One atlas holding bark and foliage, so all trees cost a single material.

    The left half is bark and the right half foliage. Trunk UVs sample the left,
    canopy UVs the right, which keeps every tree in the scene on one draw call.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    half = size_px // 2
    canvas = np.zeros((size_px, size_px, 3), dtype=np.float64)

    # Bark: brown with vertical grain.
    bark = np.zeros((size_px, half, 3), dtype=np.float64)
    bark[:, :] = (96, 72, 54)
    streaks = _value_noise((size_px, half), cells=max(4, half // 6), rng=rng)
    fine = _value_noise((size_px, half), cells=max(8, half // 2), rng=rng)
    bark *= 1.0 + 0.30 * ((0.7 * streaks + 0.3 * fine) - 0.5)[:, :, None] * 2.0
    canvas[:, :half] = bark

    # Foliage: mottled green, darker low down so a canopy reads as rounded.
    leaf = np.zeros((size_px, size_px - half, 3), dtype=np.float64)
    leaf[:, :] = (78, 108, 56)
    blobs = _value_noise((size_px, size_px - half), cells=max(5, half // 10), rng=rng)
    speckle = _value_noise((size_px, size_px - half), cells=max(10, half // 3), rng=rng)
    leaf *= 1.0 + 0.34 * ((0.6 * blobs + 0.4 * speckle) - 0.5)[:, :, None] * 2.0
    shade = np.linspace(0.72, 1.16, size_px)[:, None, None]
    leaf *= shade
    canvas[:, half:] = leaf

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "tree.png"
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)[::-1], mode="RGB").save(
        path, format="PNG"
    )
    LOG.info("wrote %s (bark and foliage atlas, %dpx)", path.name, size_px)
    return path


def generate_water_texture(work_dir: Path, size_px: int = 512, seed: int = 21) -> Path:
    """A seamless canal-water tile: dark green-brown with a soft ripple."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    canvas = np.zeros((size_px, size_px, 3), dtype=np.float64)
    canvas[:, :] = (58, 74, 68)

    broad = _value_noise((size_px, size_px), cells=max(3, size_px // 128), rng=rng)
    ripple = _value_noise((size_px, size_px), cells=max(8, size_px // 24), rng=rng)
    glint = _value_noise((size_px, size_px), cells=max(16, size_px // 8), rng=rng)

    canvas *= 1.0 + 0.22 * (broad - 0.5)[:, :, None] * 2.0
    canvas *= 1.0 + 0.14 * (ripple - 0.5)[:, :, None] * 2.0
    # A few brighter specks read as light catching the surface.
    highlight = np.clip((glint - 0.78) * 4.0, 0, 1)[:, :, None]
    canvas = canvas * (1 - highlight) + np.array([150, 168, 172]) * highlight

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "water.png"
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB").save(path)
    LOG.info("wrote %s (%dpx)", path.name, size_px)
    return path


# Car paint, picked to look like a Dutch street rather than a showroom: mostly
# grey, white and black, with a few colours among them.
CAR_PAINT: tuple[tuple[int, int, int], ...] = (
    (188, 190, 194),   # silver
    (232, 232, 230),   # white
    (48, 50, 54),      # near-black
    (72, 88, 118),     # blue
    (128, 44, 42),     # dark red
    (94, 104, 96),     # dark green
)


def generate_vehicle_texture(
    work_dir: Path, size_px: int = 256, seed: int = 51
) -> Path:
    """Atlas for cars, boats and mooring posts.

    The top half is one patch per car colour; the bottom half is hull white on
    the left and weathered timber on the right. One image means a few hundred
    vehicles cost a single draw call, and the car colours cost nothing extra
    because they are UV offsets rather than materials.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    canvas = np.zeros((size_px, size_px, 3), dtype=np.float64)
    half = size_px // 2

    # Row 0 of the array is the bottom of the tile: hull and post live here.
    hull = np.zeros((half, half, 3), dtype=np.float64)
    hull[:, :] = (206, 208, 206)
    grime = _value_noise((half, half), cells=max(3, half // 12), rng=rng)
    hull *= 1.0 + 0.14 * (grime - 0.5)[:, :, None] * 2.0
    canvas[:half, :half] = hull

    timber = np.zeros((half, size_px - half, 3), dtype=np.float64)
    timber[:, :] = (92, 78, 64)
    streak = _value_noise((half, size_px - half), cells=max(4, half // 5), rng=rng)
    timber *= 1.0 + 0.28 * (streak - 0.5)[:, :, None] * 2.0
    canvas[:half, half:] = timber

    # Car colours across the upper row.
    patch = size_px / len(CAR_PAINT)
    for index, colour in enumerate(CAR_PAINT):
        x0 = int(round(index * patch))
        x1 = int(round((index + 1) * patch))
        block = np.zeros((size_px - half, x1 - x0, 3), dtype=np.float64)
        block[:, :] = colour
        # Just enough variation that a row of the same colour is not flat.
        speckle = _value_noise(
            (size_px - half, x1 - x0), cells=max(2, (x1 - x0) // 8), rng=rng
        )
        block *= 1.0 + 0.07 * (speckle - 0.5)[:, :, None] * 2.0
        canvas[half:, x0:x1] = block

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "vehicle.png"
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)[::-1], mode="RGB").save(
        path, format="PNG"
    )
    LOG.info(
        "wrote %s (%d car colours, hull and timber, %dpx)",
        path.name,
        len(CAR_PAINT),
        size_px,
    )
    return path


def generate_structure_texture(
    work_dir: Path, size_px: int = 256, seed: int = 77
) -> Path:
    """Concrete for bridge decks, piers and tunnel walls.

    One tiling surface rather than four: everything here is cast concrete, and
    telling a deck from a pier is what the separate objects are for.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    canvas = np.zeros((size_px, size_px, 3), dtype=np.float64)
    canvas[:, :] = (150, 148, 144)

    broad = _value_noise((size_px, size_px), cells=max(3, size_px // 64), rng=rng)
    grit = _value_noise((size_px, size_px), cells=max(12, size_px // 6), rng=rng)
    canvas *= 1.0 + 0.16 * (broad - 0.5)[:, :, None] * 2.0
    canvas *= 1.0 + 0.10 * (grit - 0.5)[:, :, None] * 2.0

    # Shutter lines, the one thing that says cast concrete rather than stone.
    for offset in range(0, size_px, max(8, size_px // 4)):
        band = max(1, size_px // 128)
        canvas[offset : offset + band, :] *= 0.92

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "structure.png"
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB").save(path)
    LOG.info("wrote %s (concrete, %dpx)", path.name, size_px)
    return path


def generate_rail_texture(
    work_dir: Path, size_px: int = 256, seed: int = 63
) -> Path:
    """Atlas for track: ballast with sleepers on the left, rail steel right.

    The sleepers live here rather than in the geometry. There are 38 km of
    track within a kilometre of Utrecht Centraal, which at one sleeper per
    600 mm would be 63,000 boxes; as bands in a texture repeating along the
    track they cost nothing. The tile covers 2 m of track, so three bands put
    them at roughly the real spacing.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    half = size_px // 2
    canvas = np.zeros((size_px, size_px, 3), dtype=np.float64)

    # Ballast: coarse grey stone. High-frequency noise, because crushed rock
    # is the one surface out here with no structure at all.
    ballast = np.zeros((size_px, half, 3), dtype=np.float64)
    ballast[:, :] = (128, 124, 118)
    coarse = _value_noise((size_px, half), cells=max(8, half // 6), rng=rng)
    fine = _value_noise((size_px, half), cells=max(16, half // 2), rng=rng)
    ballast *= 1.0 + 0.42 * ((0.45 * coarse + 0.55 * fine) - 0.5)[:, :, None] * 2.0

    # Sleepers across the tile: three bands over the two metres it covers.
    sleepers = 3
    band = max(2, int(round(size_px * 0.09)))
    for index in range(sleepers):
        centre = int((index + 0.5) * size_px / sleepers)
        low, high = max(0, centre - band // 2), min(size_px, centre + band // 2)
        timber = np.zeros((high - low, half, 3), dtype=np.float64)
        timber[:, :] = (74, 62, 52)
        grain = _value_noise((high - low, half), cells=max(3, half // 8), rng=rng)
        timber *= 1.0 + 0.22 * (grain - 0.5)[:, :, None] * 2.0
        ballast[low:high] = timber
    canvas[:, :half] = ballast

    # Rail steel: dark, with a worn bright crown along the running surface.
    steel = np.zeros((size_px, size_px - half, 3), dtype=np.float64)
    steel[:, :] = (86, 84, 86)
    rust = _value_noise((size_px, size_px - half), cells=max(4, half // 6), rng=rng)
    steel *= 1.0 + 0.20 * (rust - 0.5)[:, :, None] * 2.0
    crown = np.linspace(0, 1, size_px - half)
    polish = np.exp(-((crown - 0.5) ** 2) / 0.02)[None, :, None]
    steel = steel * (1 - 0.55 * polish) + np.array([196, 198, 200]) * 0.55 * polish
    canvas[:, half:] = steel

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "rail.png"
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB").save(path)
    LOG.info("wrote %s (ballast with sleepers, and rail steel, %dpx)", path.name, size_px)
    return path


def generate_furniture_texture(
    work_dir: Path, size_px: int = 256, seed: int = 33
) -> Path:
    """Atlas for street furniture: dark metal on the left, wood on the right."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    half = size_px // 2
    canvas = np.zeros((size_px, size_px, 3), dtype=np.float64)

    metal = np.zeros((size_px, half, 3), dtype=np.float64)
    metal[:, :] = (62, 64, 68)
    grain = _value_noise((size_px, half), cells=max(4, half // 8), rng=rng)
    metal *= 1.0 + 0.16 * (grain - 0.5)[:, :, None] * 2.0
    canvas[:, :half] = metal

    wood = np.zeros((size_px, size_px - half, 3), dtype=np.float64)
    wood[:, :] = (124, 92, 58)
    streak = _value_noise((size_px, size_px - half), cells=max(3, half // 4), rng=rng)
    wood *= 1.0 + 0.26 * (streak - 0.5)[:, :, None] * 2.0
    canvas[:, half:] = wood

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "furniture.png"
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB").save(path)
    LOG.info("wrote %s (metal and wood atlas, %dpx)", path.name, size_px)
    return path


__all__ = [
    "CAR_PAINT",
    "GROUND_STYLE",
    "GROUND_STYLES",
    "generate_furniture_texture",
    "generate_rail_texture",
    "generate_structure_texture",
    "generate_vehicle_texture",
    "generate_tree_texture",
    "generate_water_texture",
    "RELIEF_DEPTH",
    "STYLES",
    "FacadeStyle",
    "generate_facade_textures",
    "relief_to_normal_map",
    "render_facade_layers",
    "render_facade_tile",
    "style_for_building",
    "style_for_height",
]
