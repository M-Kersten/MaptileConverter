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

Two things in here exist because of what they cost the mesh downstream, and
both are worth knowing before changing anything:

* Outlines are simplified **as a network, not as rings** (`partition_chains`).
  Ring by ring, a kerb two polygons share is thinned twice, differently, and
  becomes two lines that cross each other repeatedly -- and a triangulation
  handed that can only fill the gap between them with slivers.
* Contours are **joined and then thinned against the height they cost**
  (`contour_polylines`), not emitted as the raw per-cell chords marching
  squares produces. Raw, they were 99% of every breakline in an area, put a
  vertex every grid cell along ground that needed none, and about half of them
  were tracing the laser scanner's own noise across flat fields.
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

# A contour shorter than this, once simplified, is not a feature. Over flat
# ground AHN's few centimetres of scan noise throws off little closed rings
# wherever the surface happens to sit near a contour level, and each one used
# to arrive as a constraint the mesh had to honour.
MIN_CONTOUR_LENGTH_M = 4.0

# A floor under the slope used to price a contour's simplification, so that
# dead-flat ground still puts a finite number on it. At a 0.10 m tolerance this
# floor allows a contour out on the flat to wander 100 m before it is worth a
# vertex, which is the same as saying it should be a straight line.
MIN_GRADIENT = 1e-3

# Ground flatter than this carries no contour worth having. AHN measures to a
# few centimetres and a contour level that happens to land near the height of a
# flat field will wander all over it chasing that noise -- and once simplified,
# what is left is a straight line drawn across a field at random, which shows
# up in Blender as a crease through ground that is not creased.
#
# Read it as a run: at a 0.5 m contour interval, 1 in 60 is one contour every
# 30 m, and ground flatter than that does not need contours to be described to
# within a tenth of a metre.
MIN_CONTOUR_SLOPE = 1.0 / 60.0

# Contours are pulled off a lightly smoothed copy of the height grid. The
# heights themselves are never smoothed -- the mesh has to sit on what was
# measured -- but the *shape* of a contour is meant to follow the ground, and
# on a laser scan the raw shape is half noise. One 3x3 pass halves the number
# of contour chords over flat ground and leaves a canal bank, which is metres
# deep, exactly where it was.
CONTOUR_SMOOTH_PASSES = 1


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


def simplify_by_height(ring: np.ndarray, weights: np.ndarray, tolerance_m: float):
    """Douglas-Peucker priced in metres of height rather than metres sideways.

    Plain Douglas-Peucker asks how far a vertex sits from the chord that would
    replace it. For a contour that is the wrong question: a contour is a line
    of constant height, so moving it sideways by `d` where the ground slopes at
    `g` misplaces the height by `d * g`, and nothing else about `d` matters.

    Weighting the deviation by the local gradient therefore prices every vertex
    in the units the tolerance is actually written in. It also does the right
    thing at both ends by itself: on a canal bank the gradient is steep, the
    allowance in metres is small and the bank keeps its detail; out on a flat
    field the gradient is the scanner's own noise, the allowance is tens of
    metres, and the contour that was only ever tracing that noise collapses to
    a straight line and then falls under the length floor.
    """
    ring = np.asarray(ring, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(ring)
    if n < 3 or tolerance_m <= 0:
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
            off = np.hypot(rel[:, 0], rel[:, 1])
        else:
            off = np.abs(rel[:, 0] * span[1] - rel[:, 1] * span[0]) / length
        cost = off * weights[i + 1 : j]
        k = int(np.argmax(cost))
        if cost[k] > tolerance_m:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return ring[keep]


def _gradient_grid(heights: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """Slope magnitude at every grid node, in metres of rise per metre along."""
    heights = np.asarray(heights, dtype=np.float64)
    dx = float(abs(xs[1] - xs[0])) if len(xs) > 1 else 1.0
    dy = float(abs(ys[1] - ys[0])) if len(ys) > 1 else 1.0
    gy, gx = np.gradient(heights, dy, dx)
    return np.hypot(gx, gy)


def _sample_grid(grid, xs, ys, points):
    """Nearest-node lookup. Good enough: this only sets a tolerance."""
    if len(points) == 0:
        return np.zeros(0)
    cols = np.clip(
        np.round((points[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1
    )
    rows = np.clip(
        np.round((points[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1
    )
    return grid[rows, cols]


def chain_segments(segments: list[np.ndarray], *, tolerance: float = WELD_M):
    """Join loose two-point chords into the longest polylines they make.

    Marching squares emits one chord per grid cell, and consecutive cells share
    the point where the contour crosses the edge between them, so the chords
    already form chains -- they just arrive shuffled. Walking them back into
    polylines is what makes simplification possible at all: there is nothing to
    simplify about a two-point segment, which is why contours used to reach the
    triangulator at full grid resolution while every other breakline had been
    thinned.
    """
    if not segments:
        return []
    quantum = max(tolerance, 1e-9)

    def key(p):
        return (int(round(p[0] / quantum)), int(round(p[1] / quantum)))

    # Adjacency over shared endpoints. A contour level can pass through a
    # saddle, where four chord ends meet; those are left as junctions and the
    # walk simply stops there rather than guessing which pair continues.
    ends: dict = {}
    for index, seg in enumerate(segments):
        ends.setdefault(key(seg[0]), []).append((index, 0))
        ends.setdefault(key(seg[-1]), []).append((index, 1))

    used = [False] * len(segments)
    out: list[np.ndarray] = []
    for start in range(len(segments)):
        if used[start]:
            continue
        used[start] = True
        chain = [segments[start][0], segments[start][-1]]
        # Grow from both ends until the chain closes or runs out.
        for direction in (1, 0):
            while True:
                tip = chain[-1] if direction else chain[0]
                here = ends.get(key(tip), ())
                # A contour level passing through a saddle brings four chord
                # ends to one point. Walking through would splice two branches
                # into one polyline, and the simplifier would then cut the
                # corner across the saddle. Stop instead and let them stay two.
                if len(here) != 2:
                    break
                nxt = next(((i, s) for i, s in here if not used[i]), None)
                if nxt is None:
                    break
                index, side = nxt
                used[index] = True
                far = segments[index][0] if side else segments[index][-1]
                if direction:
                    chain.append(far)
                else:
                    chain.insert(0, far)
        out.append(np.asarray(chain, dtype=np.float64))
    return out


def chains_between_junctions(points: np.ndarray, edges) -> list[np.ndarray]:
    """Break an edge network into the runs between its junctions.

    A junction is any point where something other than exactly two edges meet:
    a dead end, a corner where three polygons come together, a crossing. Those
    are the points the network's shape depends on, so they are kept and the
    simple runs between them are handed back as polylines to be thinned.
    """
    neighbours: dict = {}
    for a, b in edges:
        a, b = int(a), int(b)
        if a == b:
            continue
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)

    junctions = {v for v, near in neighbours.items() if len(near) != 2}
    seen: set = set()
    out: list[np.ndarray] = []

    def walk(start, step):
        run = [start, step]
        seen.add(frozenset((start, step)))
        previous, current = start, step
        while current not in junctions:
            following = [n for n in neighbours[current] if n != previous]
            if not following:
                break
            nxt = following[0]
            if frozenset((current, nxt)) in seen:
                break
            seen.add(frozenset((current, nxt)))
            run.append(nxt)
            previous, current = current, nxt
        return run

    for start in sorted(junctions):
        for step in sorted(neighbours[start]):
            if frozenset((start, step)) not in seen:
                out.append(np.asarray([points[i] for i in walk(start, step)]))

    # Whatever is left is a ring with no junction on it at all -- an island, a
    # pond, a building standing on its own.
    for start in sorted(neighbours):
        for step in sorted(neighbours[start]):
            if frozenset((start, step)) not in seen:
                run = walk(start, step)
                if run[-1] != start:
                    run.append(start)
                out.append(np.asarray([points[i] for i in run]))
    return out


def partition_chains(rings_by_source: dict, bbox: BBox, simplify_m: float):
    """Simplify a planar partition without pulling its shared edges apart.

    The BGT is a planar partition: a road and the pavement beside it are two
    polygons that carry the *same* boundary, vertex for vertex. Simplifying
    them one ring at a time does not keep it that way. Douglas-Peucker is
    anchored on the ends of whatever it is given, and where three polygons meet
    part-way along a kerb, that junction is a ring corner for one of them and
    an ordinary point on a smooth curve for another -- so the anchors differ,
    the two copies of one kerb are thinned to different vertices, and what was
    a single line becomes two lines up to twice the tolerance apart that cross
    each other over and over.

    Everything downstream then inherits it. The crossings get cut into a ladder
    of millimetre segments, the ladder meets at angles no triangulation can
    make a decent triangle out of, and refinement chases those corners into
    ever smaller slivers: 268 of the 283 worst triangles in a test area came
    from one kerb that had been turned into two.

    So the partition is welded into a single network first, and simplification
    runs on the runs *between* junctions. Each shared kerb then exists once,
    gets thinned once, and both polygons keep the same one.
    """
    pieces: list[np.ndarray] = []
    owners: list[str] = []
    for source, rings in rings_by_source.items():
        for ring in rings:
            ring = np.asarray(ring, dtype=np.float64)[:, :2]
            for piece in clip_ring_segments(ring, bbox):
                if len(piece) >= 2:
                    pieces.append(piece)
                    owners.append(source)

    counts = {source: 0 for source in rings_by_source}
    if not pieces:
        return [], counts

    welded, mapping = weld(np.vstack(pieces), WELD_M)
    edges: set = set()
    offset = 0
    for piece in pieces:
        ids = mapping[offset : offset + len(piece)]
        offset += len(piece)
        for k in range(len(ids) - 1):
            if ids[k] != ids[k + 1]:
                edges.add(frozenset((int(ids[k]), int(ids[k + 1]))))

    chains = []
    for run in chains_between_junctions(welded, [tuple(e) for e in edges]):
        simplified = douglas_peucker(run, simplify_m)
        if len(simplified) >= 2:
            chains.append(simplified)

    # Which source a run belongs to is only ever reported, never used, and a
    # shared kerb belongs to two of them. Counted against the first that
    # claimed a piece so the log still says where the work came from.
    for source in counts:
        counts[source] = sum(1 for o in owners if o == source)
    return chains, counts


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
    out: list[np.ndarray] = []
    for _level, segments in _contour_levels(heights, xs, ys, interval_m):
        out.extend(segments)
    return out


def _contour_levels(heights, xs, ys, interval_m):
    """Yield (level, chords) so callers can keep the levels apart.

    Chaining has to happen within a level: two chords from different levels can
    share an endpoint where the surface is flat, and joining across that would
    make a polyline that is not a contour of anything.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if interval_m <= 0 or heights.size == 0:
        return
    low = float(np.floor(heights.min() / interval_m) * interval_m)
    high = float(heights.max())
    levels = np.arange(low + interval_m, high, interval_m)
    if len(levels) == 0:
        return

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
        here: list[np.ndarray] = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            here.extend(
                _cell_segments(heights, xs, ys, r, c, float(level), int(code[r, c]))
            )
        if here:
            yield float(level), here


def contour_polylines(
    heights: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    interval_m: float = DEFAULT_CONTOUR_INTERVAL_M,
    tolerance_m: float = 0.10,
    min_length_m: float = MIN_CONTOUR_LENGTH_M,
) -> list[np.ndarray]:
    """Contours as joined, simplified polylines rather than per-cell chords.

    This is what the terrain mesh wants. Emitting the chords raw put a vertex
    every grid cell along every contour, which made the contours 99% of all the
    breaklines in an area and left the mesh a hundred times denser along them
    than on the ground either side -- and a triangulation that has to bridge a
    density step like that can only do it with long thin wedges, whatever
    triangulator you use.

    So: join the chords back into the lines they came from, then spend vertices
    on them at the rate the height tolerance justifies.
    """
    source = _smooth(np.asarray(heights, dtype=np.float64), CONTOUR_SMOOTH_PASSES)
    gradient = _gradient_grid(source, xs, ys)
    out: list[np.ndarray] = []
    for _level, chords in _contour_levels(source, xs, ys, interval_m):
        for line in chain_segments(chords):
            if len(line) < 2:
                continue
            slope = _sample_grid(gradient, xs, ys, line)
            # Not a contour of anything: the ground it crosses is flat enough
            # that the level it marks is inside the scanner's own noise.
            if float(np.median(slope)) < MIN_CONTOUR_SLOPE:
                continue
            simplified = simplify_by_height(
                line, np.maximum(slope, MIN_GRADIENT), tolerance_m
            )
            if len(simplified) < 2:
                continue
            run = float(np.hypot(*np.diff(simplified, axis=0).T).sum())
            if run < min_length_m:
                continue
            out.append(simplified)
    return out


def _smooth(grid: np.ndarray, passes: int) -> np.ndarray:
    """3x3 mean, edges held. Only ever used to decide where contours run."""
    out = grid
    for _ in range(max(int(passes), 0)):
        pad = np.pad(out, 1, mode="edge")
        out = (
            sum(pad[i : i + grid.shape[0], j : j + grid.shape[1]]
                for i in range(3) for j in range(3))
            / 9.0
        )
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
    tolerance_m: float = 0.10,
) -> BreaklineSet:
    """Turn outlines and a height grid into non-crossing constraint segments."""
    # Simplified as one network rather than ring by ring, so a boundary two
    # polygons share stays one line. See partition_chains.
    chains, counts = partition_chains(rings_by_source, bbox, simplify_m)

    if heights is not None and xs is not None and ys is not None:
        before = len(chains)
        for line in contour_polylines(
            heights,
            xs,
            ys,
            interval_m=contour_interval_m,
            tolerance_m=tolerance_m,
        ):
            for piece in clip_ring_segments(line, bbox):
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
    "MIN_CONTOUR_LENGTH_M",
    "BreaklineSet",
    "build_breaklines",
    "chain_segments",
    "chains_between_junctions",
    "clip_ring_segments",
    "contour_lines",
    "contour_polylines",
    "douglas_peucker",
    "partition_chains",
    "resolve_crossings",
    "simplify_by_height",
    "split_crossings",
]
