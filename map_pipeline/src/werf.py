"""Werven: the wharves of Utrecht's canals and the cellar walls behind them.

Along the Oudegracht the street runs four metres above the water, the houses
on one side and an iron railing on the other, and below the railing a second
street: the wharf, a few metres wide at the waterline, with the cellars under
the street opening onto it, a door and a window each. The survey has all of
it, a sharp step of 4.0 m from the street (5.1 m NAP round the Ganzenmarkt)
down to the wharf (1.2 m). The layered ground read it off a 4 m grid and made
a bank of it: a slope from the street down to the water, with the wharf gone.

The BGT draws the wall that holds the street up: ``kunstwerkdeel``, plus-type
``keermuur``, a strip 0.29 m wide (the median of 49 round the Ganzenmarkt) in
pieces of about six metres, one per cellar. That strip is

* a region of its own in the ground (`layered`), widened to `WALL_PAD_M` either
  side so the mesher keeps it, whose edge on the street side is at the
  street's height and on the wharf side at the wharf's: the drop is the strip;
* a wall standing on it (`build_werf_walls`), from just under the wharf to the
  street, brick, a cellar door and a window to every six metres of it. Its top
  is flush with the street and covers the steep strip; its face is the wall
  the wharf sees.

Strips at ``relatieve_hoogteligging`` -1 are the walls of the passages under
the bridges and the cellar walls carried on into them: walls too, inside the
bridge's abutment or along its passage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

# How far each side of the BGT strip the wall and its ground region reach:
# a 0.29 m strip is too thin for the mesher, which folds anything under half
# a metre into its neighbour.
WALL_PAD_M = 0.16
# A wall whose street is less than this above its wharf is no werf wall.
MIN_DROP_M = 0.6
# How far beyond the wall the ground is read on either side.
READ_M = (0.35, 0.8)
# The wall's foot this far under the wharf, its top this far over the street.
FOOT_SINK_M = 0.25
TOP_LIFT_M = 0.02
# The texture: one cellar front, door and window, to this much wall.
TILE_W_M = 6.0
TILE_H_M = 4.0
VERTEX_EVERY_M = 1.0


def retaining_walls(features, bbox) -> list:
    """The BGT's retaining walls on the ground, as shapely polygons in RD."""
    import shapely
    from shapely.geometry import shape

    box = shapely.box(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
    out = []
    for feature in features:
        props = feature.get("properties") or {}
        kind = str(props.get("plus_type") or props.get("type") or "").lower()
        if "keermuur" not in kind:
            continue
        try:
            level = int(props.get("relatieve_hoogteligging") or 0)
        except (TypeError, ValueError):
            level = 0
        # Under the ground (-1) is under a bridge: the walls of the wharf's
        # passage through it, and the cellar wall carried on into it. Left
        # out, the street sloped down to the wharf beside every bridge.
        if level > 0:
            continue
        try:
            geometry = shapely.make_valid(shapely.force_2d(shape(feature.get("geometry")))).intersection(box)
        except Exception:  # noqa: BLE001 - one broken outline
            continue
        for part in getattr(geometry, "geoms", [geometry]):
            if part.geom_type == "Polygon" and part.area > 0.05:
                out.append(part)
    return out


def wall_band(polygon):
    """The strip as the wall and the ground take it: padded, simplified."""
    band = polygon.buffer(WALL_PAD_M, join_style="mitre", mitre_limit=2.0).simplify(0.05)
    if band.geom_type == "MultiPolygon":
        band = max(band.geoms, key=lambda g: g.area)
    return band


@dataclass
class WerfSet:
    """Cellar walls: triangles in RD and NAP, and their UVs in texture tiles."""

    tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    uvs: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 2)))
    counts: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.tris))

    def stats(self) -> dict:
        return {"triangles": len(self), **self.counts}


def _levels(points: np.ndarray, band, ground_at) -> tuple[np.ndarray, np.ndarray]:
    """The street over and the wharf under each point of a wall's outline.

    Read just beyond the band all round each point, leaving out what falls
    inside it: the highest is the street, the lowest the wharf.
    """
    import shapely

    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    offsets = np.concatenate([
        np.column_stack([np.cos(angles), np.sin(angles)]) * r for r in READ_M
    ])
    tops, feet = [], []
    for p in points:
        around = p + offsets
        outside = ~shapely.contains_xy(band, around[:, 0], around[:, 1])
        if not outside.any():
            tops.append(np.nan)
            feet.append(np.nan)
            continue
        z = np.asarray(ground_at(around[outside, 0], around[outside, 1]), dtype=np.float64)
        z = z[np.isfinite(z)]
        tops.append(float(z.max()) if len(z) else np.nan)
        feet.append(float(z.min()) if len(z) else np.nan)
    return np.asarray(tops), np.asarray(feet)


def _densify(ring: np.ndarray, every: float) -> np.ndarray:
    out = []
    for a, b in zip(ring[:-1], ring[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / every)))
        for t in np.arange(n) / n:
            out.append(a + (b - a) * t)
    return np.asarray(out)


def build_werf_walls(walls, ground_at, *, min_drop_m: float = MIN_DROP_M) -> WerfSet:
    """A brick wall on every retaining strip, from its wharf to its street.

    `ground_at(x, y)` is the ground as drawn: the walls meet it at its own
    heights, so the street runs onto their tops and the wharf to their feet.
    """
    import mapbox_earcut

    result = WerfSet()
    tris: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    built, low, metres = 0, 0, 0.0
    for polygon in walls:
        band = wall_band(polygon)
        if band.is_empty or band.geom_type != "Polygon":
            continue
        ring = np.asarray(band.exterior.coords)[:, :2]
        if len(ring) < 4:
            continue
        # Counter-clockwise, so every side faces out.
        if np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]) < 0:
            ring = ring[::-1]
        points = _densify(ring, VERTEX_EVERY_M)
        tops, feet = _levels(points, band, ground_at)
        ok = np.isfinite(tops) & np.isfinite(feet)
        if ok.sum() < 3:
            continue
        # A point whose reading failed takes its neighbours'.
        index = np.arange(len(points))
        tops = np.interp(index, index[ok], tops[ok])
        feet = np.interp(index, index[ok], feet[ok])
        if float(np.median(tops - feet)) < min_drop_m:
            low += 1
            continue
        # One street and one wharf along each wall: the top flat across it, so
        # its coping is level where the street is.
        top = tops + TOP_LIFT_M
        foot = feet - FOOT_SINK_M
        n = len(points)
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
        closing = float(np.hypot(*(points[0] - points[-1])))
        for i in range(n):
            j = (i + 1) % n
            u0 = along[i] / TILE_W_M
            u1 = (along[i] + (np.hypot(*(points[j] - points[i])) if j else closing)) / TILE_W_M
            a_lo = [*points[i], foot[i]]
            b_lo = [*points[j], foot[j]]
            b_hi = [*points[j], top[j]]
            a_hi = [*points[i], top[i]]
            # The door stands on the wharf: up the texture from the wall's foot
            # where it is seen, the wharf, not from under it.
            v = lambda z, base: (z - base) / TILE_H_M  # noqa: E731
            base_i, base_j = feet[i], feet[j]
            tris.append(np.array([a_lo, b_lo, b_hi]))
            uvs.append(np.array([[u0, v(foot[i], base_i)], [u1, v(foot[j], base_j)], [u1, v(top[j], base_j)]]))
            tris.append(np.array([a_lo, b_hi, a_hi]))
            uvs.append(np.array([[u0, v(foot[i], base_i)], [u1, v(top[j], base_j)], [u0, v(top[i], base_i)]]))
        # The top, the coping, flush with the street.
        try:
            caps = np.asarray(
                mapbox_earcut.triangulate_float64(points, np.array([n], dtype=np.uint32)), dtype=np.int64
            ).reshape(-1, 3)
        except Exception:  # noqa: BLE001 - a wall without a lid is still a wall
            caps = np.zeros((0, 3), dtype=np.int64)
        for a, b, c in caps:
            tri = np.array([[*points[k], top[k]] for k in (a, b, c)])
            # Up, whatever earcut wound.
            if np.cross(tri[1] - tri[0], tri[2] - tri[0])[2] < 0:
                tri = tri[[0, 2, 1]]
            tris.append(tri)
            # The coping in the texture's brick band above the door.
            uvs.append(tri[:, :2] / TILE_W_M * np.array([1.0, 0.0]) + np.array([0.0, 0.99]))
        built += 1
        metres += 0.5 * float(band.length)
    if tris:
        result.tris = np.asarray(tris, dtype=np.float64)
        result.uvs = np.asarray(uvs, dtype=np.float64)
    result.counts = {"walls": built, "walls_too_low": low, "wall_length_m": round(metres, 1)}
    if built:
        LOG.info(
            "werven: %d cellar walls, %.0f m along the wharves; %d retaining walls too low for one",
            built, metres, low,
        )
    return result


def save_werf(werf: WerfSet, path: Path) -> Path:
    np.savez_compressed(path, tris=werf.tris, uvs=werf.uvs)
    return path


def _brick(work_dir: Path, w: int, h: int, px_per_m: int) -> np.ndarray:
    """`w` by `h` pixels of old red brick at `px_per_m`: photographed where it
    can be had (Poly Haven, CC0), drawn where it cannot."""
    brick = None
    try:
        from .textures import _tile_to, _tint, fetch_texture, texture_size_m

        path = fetch_texture("medieval_red_brick", work_dir)
        bw, bh = texture_size_m("medieval_red_brick", work_dir)
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
        side = max(w, h)
        brick = _tint(_tile_to(pixels, side, side / px_per_m / bw, side / px_per_m / bh)[:h, :w], (122, 70, 54), 0.5)
    except Exception:  # noqa: BLE001 - offline: a drawn brick
        brick = None
    if brick is None or brick.shape[:2] != (h, w):
        rng = np.random.default_rng(5)
        brick = np.zeros((h, w, 3))
        course = max(2, int(0.065 * px_per_m))
        length = max(4, int(0.21 * px_per_m))
        for row in range(0, h, course):
            offset = (row // course % 2) * length // 2
            for col in range(-offset, w, length):
                tone = np.array([128, 72, 56]) * rng.uniform(0.82, 1.12)
                brick[row:row + course - 1, max(col, 0):col + length - 1] = tone
            brick[row + course - 1:row + course] = (150, 140, 128)
    return brick


def generate_brick_texture(work_dir: Path, *, px_per_m: int = 128, size_m: float = 4.0) -> tuple[Path, Path]:
    """Plain old brick, `size_m` square, for the sides and arches of bridges."""
    from PIL import Image

    from .facade import relief_to_normal_map

    n = int(size_m * px_per_m)
    colour = _brick(work_dir, n, n, px_per_m)
    relief = colour.mean(axis=2) / 255.0 * 0.4
    colour_path = Path(work_dir) / "bridge_brick.png"
    normal_path = Path(work_dir) / "bridge_brick_normal.png"
    Image.fromarray(np.clip(colour, 0, 255).astype(np.uint8)).save(colour_path)
    Image.fromarray(relief_to_normal_map(relief)).save(normal_path)
    return colour_path, normal_path


def generate_werf_texture(work_dir: Path, *, px_per_m: int = 128) -> tuple[Path, Path]:
    """The cellar front: brick, an arched door and an arched window, tiling.

    `TILE_W_M` by `TILE_H_M`, the door's foot on the bottom edge where the
    wharf is.
    """
    from PIL import Image

    from .facade import relief_to_normal_map

    w, h = int(TILE_W_M * px_per_m), int(TILE_H_M * px_per_m)
    colour = _brick(work_dir, w, h, px_per_m).copy()
    relief = colour.mean(axis=2) / 255.0 * 0.4

    yy, xx = np.mgrid[0:h, 0:w]
    up = (h - 1 - yy) / px_per_m  # metres above the wharf
    across = xx / px_per_m

    def arch(x0, width, sill, height):
        """Inside an arched opening: a rectangle with a half circle on top."""
        r = width / 2.0
        cx = x0 + r
        spring = sill + height - r
        body = (across >= x0) & (across <= x0 + width) & (up >= sill) & (up <= spring)
        crown = ((across - cx) ** 2 + (up - spring) ** 2 <= r * r) & (up > spring)
        return body | crown

    door = arch(1.2, 1.5, 0.0, 2.6)
    door_reveal = arch(1.08, 1.74, 0.0, 2.72) & ~door
    window = arch(3.9, 0.95, 1.1, 1.35)
    window_reveal = arch(3.8, 1.15, 1.0, 1.5) & ~window
    stone = (200, 192, 176)
    colour[door_reveal] = stone
    colour[window_reveal] = stone
    # A plank door, dark with age, and the window's glass with its bars.
    planks = (np.floor(across / 0.18) % 2).astype(bool)
    colour[door] = np.where(planks[door, None], (70, 46, 32), (60, 40, 28))
    colour[window] = (34, 40, 46)
    bars = window & ((np.abs(((across - 3.9) % 0.32) - 0.16) < 0.02) | (np.abs(up - 1.75) < 0.025))
    colour[bars] = (210, 206, 196)
    # The top band: the coping stone of the street's edge.
    colour[up >= TILE_H_M - 0.12] = stone
    relief[door] -= 0.25
    relief[window] -= 0.2
    relief[door_reveal | window_reveal] += 0.05
    colour_path = Path(work_dir) / "werf_wall.png"
    normal_path = Path(work_dir) / "werf_wall_normal.png"
    Image.fromarray(np.clip(colour, 0, 255).astype(np.uint8)).save(colour_path)
    Image.fromarray(relief_to_normal_map(relief)).save(normal_path)
    return colour_path, normal_path


__all__ = [
    "MIN_DROP_M",
    "TILE_H_M",
    "TILE_W_M",
    "WALL_PAD_M",
    "WerfSet",
    "build_werf_walls",
    "generate_brick_texture",
    "generate_werf_texture",
    "retaining_walls",
    "save_werf",
    "wall_band",
]
