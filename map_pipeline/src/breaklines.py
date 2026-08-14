"""The lines the terrain mesh has to fold along.

A canal bank, a kerb, the foot of a building and a contour are all places where
the ground changes, and a mesh that does not have an edge there cannot show the
change or be edited at it. This module collects those lines and hands them to
the triangulator as constraints.

Four sources, in the order they matter:

* **BGT outlines** -- water, roads, and the land-cover polygons between them.
  The BGT is a planar partition, so these outlines already tile the ground
  without gaps or overlaps, and adjacent polygons repeat their shared corners
  exactly. That is why 77k raw outline vertices over 600 m of Utrecht weld down
  to 38k: half of them are the same corner seen from both sides.
* **Building footprints**, so the terrain has a loop to select where a wall
  meets the ground.
* **Contours** off the height model, which are the only breaklines that follow
  the shape of the ground rather than something drawn on it.
* **The bbox itself**, so the mesh ends in a straight edge that tiles with a
  neighbouring area.

The triangulator cannot take two constraints that cross -- no triangulation can
satisfy both -- so everything here ends in `split_crossings`, which cuts every
segment at every intersection. BGT outlines do not cross each other, but
contours cross roads and water constantly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .cdt import WELD_M, weld
from .geo import BBox

LOG = logging.getLogger(__name__)

# How far a simplified outline may stray from the surveyed one. The BGT is
# surveyed to the centimetre and carries vertices a couple of centimetres
# apart along what is visibly a straight kerb; keeping those costs triangles
# and buys nothing anyone can see.
DEFAULT_SIMPLIFY_M = 0.15

# Contours closer together than this in height are not worth an edge loop over
# Dutch terrain, where a whole bbox often spans five metres.
DEFAULT_CONTOUR_INTERVAL_M = 0.5

# Segments shorter than this are dropped. They come from simplification leaving
# a stub, and a triangulator handed one makes a sliver.
MIN_SEGMENT_M = 0.05


@dataclass
class BreaklineSet:
    """Constraint segments, as welded points plus index pairs."""

    points: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float64)
    )
    segments: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )
    counts: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.segments))

    def stats(self) -> dict:
        return {
            "points": int(len(self.points)),
            "segments": int(len(self.segments)),
            "by_source": dict(self.counts),
        }


def douglas_peucker(ring: np.ndarray, tolerance: float) -> np.ndarray:
    """Drop vertices that sit within `tolerance` of the line they lie on."""
    ring = np.asarray(ring, dtype=np.float64)
    n = len(ring)
    if n < 3 or tolerance <= 0:
        return ring
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        span = ring[j] - ring[i]
        length = float(np.hypot(span[0], span[1]))
        rel = ring[i + 1 : j] - ring[i]
        if length < 1e-12:
            far = np.hypot(rel[:, 0], rel[:, 1])
        else:
            far = np.abs(rel[:, 0] * span[1] - rel[:, 1] * span[0]) / length
        k = int(np.argmax(far))
        if far[k] > tolerance:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return ring[keep]


def clip_ring_segments(ring: np.ndarray, bbox: BBox) -> list[np.ndarray]:
    """Cut a ring into the pieces of it that lie inside `bbox`.

    Returned as open polylines: a ring that leaves and re-enters the area comes
    back as two, which is right -- the part outside is not ours to constrain.
    """
    ring = np.asarray(ring, dtype=np.float64)
    if len(ring) < 2:
        return []
    pieces: list[list] = []
    current: list = []
    for a, b in zip(ring[:-1], ring[1:]):
        clipped = _clip_segment(a, b, bbox)
        if clipped is None:
            if len(current) >= 2:
                pieces.append(current)
            current = []
            continue
        pa, pb = clipped
        if current and np.allclose(current[-1], pa, atol=1e-9):
            current.append(pb)
        else:
            if len(current) >= 2:
                pieces.append(current)
            current = [pa, pb]
    if len(current) >= 2:
        pieces.append(current)
    return [np.asarray(p, dtype=np.float64) for p in pieces]


def _clip_segment(a, b, bbox: BBox):
    """Liang-Barsky against the bbox; None when the segment misses it."""
    x0, y0 = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - x0, float(b[1]) - y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0 - bbox.xmin),
        (dx, bbox.xmax - x0),
        (-dy, y0 - bbox.ymin),
        (dy, bbox.ymax - y0),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t1 <= t0:
        return None
    return (
        np.array([x0 + t0 * dx, y0 + t0 * dy]),
        np.array([x0 + t1 * dx, y0 + t1 * dy]),
    )


def contour_lines(
    heights: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    interval_m: float = DEFAULT_CONTOUR_INTERVAL_M,
) -> list[np.ndarray]:
    """Contours of a height grid, by marching squares.

    Each cell is handled on its own and yields at most two segments, so the
    output is a soup of short pieces rather than joined-up rings. That is all
    the triangulator wants, and joining them would only be undone by the
    splitting step anyway.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if interval_m <= 0 or heights.size == 0:
        return []
    low = float(np.floor(heights.min() / interval_m) * interval_m)
    high = float(heights.max())
    levels = np.arange(low + interval_m, high, interval_m)
    if len(levels) == 0:
        return []

    out: list[np.ndarray] = []
    z00 = heights[:-1, :-1]
    z10 = heights[:-1, 1:]
    z11 = heights[1:, 1:]
    z01 = heights[1:, :-1]
    for level in levels:
        # Corner code per cell, counter-clockwise from the bottom left.
        code = (
            (z00 > level).astype(np.uint8)
            | ((z10 > level).astype(np.uint8) << 1)
            | ((z11 > level).astype(np.uint8) << 2)
            | ((z01 > level).astype(np.uint8) << 3)
        )
        active = np.flatnonzero((code != 0) & (code != 15))
        if active.size == 0:
            continue
        rows, cols = np.unravel_index(active, code.shape)
        for r, c in zip(rows.tolist(), cols.tolist()):
            for seg in _cell_segments(
                heights, xs, ys, r, c, float(level), int(code[r, c])
            ):
                out.append(seg)
    return out


# Which pair of cell edges each marching-squares case joins. Edges are numbered
# 0 bottom, 1 right, 2 top, 3 left. The two saddle cases produce two segments.
_CASES = {
    1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),),
    5: ((3, 2), (0, 1)), 6: ((0, 2),), 7: ((3, 2),), 8: ((2, 3),),
    9: ((2, 0),), 10: ((2, 1), (0, 3)), 11: ((2, 1),), 12: ((1, 3),),
    13: ((1, 0),), 14: ((0, 3),),
}


def _cell_segments(heights, xs, ys, r, c, level, code):
    corners = (
        (heights[r, c], xs[c], ys[r]),
        (heights[r, c + 1], xs[c + 1], ys[r]),
        (heights[r + 1, c + 1], xs[c + 1], ys[r + 1]),
        (heights[r + 1, c], xs[c], ys[r + 1]),
    )

    def crossing(edge):
        a, b = corners[edge], corners[(edge + 1) % 4]
        span = b[0] - a[0]
        t = 0.5 if abs(span) < 1e-12 else (level - a[0]) / span
        t = min(max(t, 0.0), 1.0)
        return (a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))

    return [
        np.array([crossing(e0), crossing(e1)], dtype=np.float64)
        for e0, e1 in _CASES.get(code, ())
    ]


def split_crossings(points: np.ndarray, segments: np.ndarray, *, cell_m: float = 10.0):
    """Cut every segment at every point where two of them cross.

    The triangulator cannot honour two constraints that cross, and it says so
    rather than guessing, so this has to happen first. Candidate pairs come out
    of a uniform grid rather than an all-pairs sweep: at 40 000 segments the
    difference is a second against most of an hour.
    """
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.int64)
    if len(segments) == 0:
        return points, segments

    a = points[segments[:, 0]]
    b = points[segments[:, 1]]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)

    origin = points.min(axis=0)
    buckets: dict = {}
    for index in range(len(segments)):
        c0 = ((lo[index] - origin) / cell_m).astype(np.int64)
        c1 = ((hi[index] - origin) / cell_m).astype(np.int64)
        for cx in range(c0[0], c1[0] + 1):
            for cy in range(c0[1], c1[1] + 1):
                buckets.setdefault((cx, cy), []).append(index)

    cuts: dict = {}
    checked = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                p, q = members[i], members[j]
                if (p, q) in checked:
                    continue
                checked.add((p, q))
                if (
                    hi[p][0] < lo[q][0] or hi[q][0] < lo[p][0]
                    or hi[p][1] < lo[q][1] or hi[q][1] < lo[p][1]
                ):
                    continue
                if set(segments[p]) & set(segments[q]):
                    continue  # they already share an endpoint
                hit = _intersection(a[p], b[p], a[q], b[q])
                if hit is None:
                    continue
                cuts.setdefault(p, []).append(hit)
                cuts.setdefault(q, []).append(hit)

    if not cuts:
        return points, segments

    extra = [points]
    out: list = []
    next_index = len(points)
    for index in range(len(segments)):
        if index not in cuts:
            out.append(segments[index])
            continue
        start, end = a[index], b[index]
        direction = end - start
        length2 = float(direction @ direction)
        ordered = sorted(
            cuts[index],
            key=lambda h: float((h - start) @ direction) / max(length2, 1e-12),
        )
        chain = [int(segments[index][0])]
        for hit in ordered:
            extra.append(hit.reshape(1, 2))
            chain.append(next_index)
            next_index += 1
        chain.append(int(segments[index][1]))
        out.extend([[chain[k], chain[k + 1]] for k in range(len(chain) - 1)])

    LOG.info(
        "breaklines: %d segments cut at %d crossings",
        len(cuts),
        sum(len(v) for v in cuts.values()) // 2,
    )
    return np.vstack(extra), np.asarray(out, dtype=np.int64)


def resolve_crossings(points, segments, *, rounds: int = 4, cell_m: float = 10.0):
    """Split until nothing crosses, or say so and let the caller cope.

    One pass is not enough. Welding after a split is what merges the two copies
    of a shared intersection into one point, but it also drags a cut that
    landed a millimetre from an endpoint back onto that endpoint, which undoes
    the split and restores the crossing. Repeating converges in two or three
    rounds; what is left after that is a handful of contour segments meeting at
    a hair, and the triangulator is told to drop those rather than refuse the
    whole mesh over them.
    """
    for _ in range(rounds):
        before = len(segments)
        points, segments = split_crossings(points, segments, cell_m=cell_m)
        points, mapping = weld(points, WELD_M)
        segments = mapping[segments].reshape(-1, 2)
        segments = segments[segments[:, 0] != segments[:, 1]]
        segments = np.unique(np.sort(segments, axis=1), axis=0)
        if len(segments) == before:
            return points, segments
    LOG.info("breaklines: some still meet at a hair after %d rounds", rounds)
    return points, segments


def _intersection(p0, p1, q0, q1):
    """Where two segments properly cross, or None."""
    r = p1 - p0
    s = q1 - q0
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-15:
        return None  # parallel or collinear; nothing to cut at
    diff = q0 - p0
    t = (diff[0] * s[1] - diff[1] * s[0]) / denom
    u = (diff[0] * r[1] - diff[1] * r[0]) / denom
    eps = 1e-9
    if not (eps < t < 1.0 - eps and eps < u < 1.0 - eps):
        return None
    return p0 + t * r


def build_breaklines(
    bbox: BBox,
    *,
    rings_by_source: dict,
    heights: np.ndarray | None = None,
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
    simplify_m: float = DEFAULT_SIMPLIFY_M,
    contour_interval_m: float = DEFAULT_CONTOUR_INTERVAL_M,
) -> BreaklineSet:
    """Turn outlines and a height grid into non-crossing constraint segments."""
    chains: list[np.ndarray] = []
    counts: dict = {}

    for source, rings in rings_by_source.items():
        before = len(chains)
        for ring in rings:
            ring = np.asarray(ring, dtype=np.float64)[:, :2]
            for piece in clip_ring_segments(ring, bbox):
                simplified = douglas_peucker(piece, simplify_m)
                if len(simplified) >= 2:
                    chains.append(simplified)
        counts[source] = len(chains) - before

    if heights is not None and xs is not None and ys is not None:
        before = len(chains)
        for seg in contour_lines(heights, xs, ys, interval_m=contour_interval_m):
            for piece in clip_ring_segments(seg, bbox):
                if len(piece) >= 2:
                    chains.append(piece)
        counts["contours"] = len(chains) - before

    # The area's own edge, so the mesh ends straight and tiles with its
    # neighbours instead of stopping wherever the outlines happened to.
    corners = np.array(
        [
            [bbox.xmin, bbox.ymin],
            [bbox.xmax, bbox.ymin],
            [bbox.xmax, bbox.ymax],
            [bbox.xmin, bbox.ymax],
            [bbox.xmin, bbox.ymin],
        ],
        dtype=np.float64,
    )
    chains.append(corners)
    counts["bbox"] = 1

    if not chains:
        return BreaklineSet(counts=counts)

    raw = np.vstack(chains)
    welded, mapping = weld(raw, WELD_M)

    pairs: list = []
    offset = 0
    for chain in chains:
        n = len(chain)
        ids = mapping[offset : offset + n]
        offset += n
        for k in range(n - 1):
            if ids[k] != ids[k + 1]:
                pairs.append((int(ids[k]), int(ids[k + 1])))

    segments = np.asarray(sorted({tuple(sorted(p)) for p in pairs}), dtype=np.int64)
    segments = segments.reshape(-1, 2)

    # Drop stubs left by simplification before anything has to triangulate them.
    if len(segments):
        length = np.hypot(*(welded[segments[:, 1]] - welded[segments[:, 0]]).T)
        segments = segments[length >= MIN_SEGMENT_M]

    welded, segments = resolve_crossings(welded, segments)

    LOG.info(
        "breaklines: %d segments over %d points (%s)",
        len(segments),
        len(welded),
        ", ".join(f"{k} {v}" for k, v in counts.items() if v),
    )
    return BreaklineSet(points=welded, segments=segments, counts=counts)


# Which BGT collections carry an outline worth folding the terrain along.
# Order is only for the log line; they all end up in one constraint set.
OUTLINE_COLLECTIONS = {
    "water": "waterdeel",
    "roads": "wegdeel",
    "unpaved": "onbegroeidterreindeel",
    "green": "begroeidterreindeel",
    "buildings": "pand",
}


def fetch_outline_rings(bbox: BBox, *, sources=None, **fetch_kwargs) -> dict:
    """Read the BGT outlines the terrain mesh folds along.

    Responses are cached by the BGT client, so the surfaces stage reads the
    same collections again for free rather than paying for them twice.
    """
    from .bgt import fetch_current, polygon_rings

    wanted = OUTLINE_COLLECTIONS if sources is None else {
        k: v for k, v in OUTLINE_COLLECTIONS.items() if k in set(sources)
    }
    out: dict = {}
    for name, collection in wanted.items():
        try:
            features, _ = fetch_current(collection, bbox, **fetch_kwargs)
        except Exception as exc:  # noqa: BLE001 - a missing outline is not fatal
            LOG.warning("no %s outlines for the terrain mesh (%s)", name, exc)
            continue
        rings: list = []
        for feature in features:
            for group in polygon_rings(feature.get("geometry")):
                rings.extend(group)
        if rings:
            out[name] = rings
    return out


__all__ = [
    "OUTLINE_COLLECTIONS",
    "fetch_outline_rings",
    "DEFAULT_CONTOUR_INTERVAL_M",
    "DEFAULT_SIMPLIFY_M",
    "BreaklineSet",
    "build_breaklines",
    "clip_ring_segments",
    "contour_lines",
    "douglas_peucker",
    "resolve_crossings",
    "split_crossings",
]
