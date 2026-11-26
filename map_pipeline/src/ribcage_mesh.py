"""The layered style's ground as a ribcage: the 3D team's layout, and the default.

What the 3D team asked for, with the grid mesher's Hoek van Holland in front
of them: areas designed like a ribcage, where quads do the work across the
large spaces, triangles fill in at the edges where the outline needs them,
and the quads are never just the underlying grid. Then, with their own
Hazeldonk file beside ours: a large area is a few large faces there, which
is what they select and edit.

So there is no grid. Every region -- a street, a verge, a square, a field, a
canal -- is meshed in its own frame:

* **Ribs across what is narrow.** A street, a footpath, a verge: ribs run
  straight across it from station to station along its two outlines, the
  same rungs the grid mesher lays (`terrain_mesh._cross_rungs`), so it is a
  ladder of quads that follows the street round its bends. Nothing else is
  put in a ladder cell; a road whose crown or dike the survey shows gets one
  loop down its middle, as with the grid.
* **One face for a plain open space.** What the ribs leave -- a square, a
  field, a yard, a junction -- is one face, its outline only, where its
  ground lies in one plane to the tolerance (`_plain`). Where it does not, it
  is cut in two across its longer side along a line of a lattice turned to
  the space (`_own_angle`, `_cut`), and each piece tested again, down to
  pieces the station spacing across (`_open_layout`). A cut ends on a
  station of the outline, so no neighbour gains a vertex.
* **Open water has one flat bed.** A pond gets an edge loop a little way in
  from its outline, and a hand's breadth in along a quay wall, where the
  ground drops sheer; inside a ring further in, the bed is one flat face
  (`_bed_fill`). Water too narrow for that, and with ``open_faces`` off every
  open space, gets the lattice the ribcage first had: quads at twice the
  station spacing, at the station spacing where those miss the survey
  (`_refined`), joined to the outline by triangles paired into quads.

The outlines carry the same stations on both sides (`layered_mesh.
_with_stations`), so two regions meet vertex for vertex whatever each does
inside; the finishing -- welding, mending, heights, beds, diagonals -- is the
grid mesher's own (`layered_mesh._assemble`), so the two can be compared on
their layout alone. The grid is ``layered.mesher: "quads"``.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .layered_mesh import (
    LAND_ROW_M,
    NEAR_CELL_M,
    PAVED_ROW_M,
    QUAY_REACH_M,
    QUAY_MIN_M,
    QUAY_ROW_M,
    SQUARE_DEG,
    THIN_M,
    _area,
    _assemble,
    _divide_rungs,
    _parts,
    _rungs_miss,
    _with_stations,
    _without_overlaps,
    _without_thin,
)

LOG = logging.getLogger(__name__)

# The lattice in the open is this many station spacings across where the
# ground allows: the large quads doing the work of the large spaces.
OPEN_FACTOR = 2
# A cell smaller than this many station squares is filled from its outline.
SMALL_CELL_SQUARES = 2.5
# A point of the fine lattice goes in where the coarse one misses the survey
# there by more than this share of the tolerance.
REFINE_SHARE = 0.8
# Water's edge loop: this far in from the bank, so the bed is carved even in a
# ditch; along a quay, `QUAY_ROW_M`.
BANK_LOOP_M = 0.5

# An open space is one face where the ground in it is plain: its corners
# within this share of the tolerance of one plane, so the face lies flat
# whichever way it is cut into triangles ...
PLAIN_CORNER_SHARE = 0.3
# ... and the survey inside it within this share of the tolerance of that
# plane, the same margin the lattice refines at.
PLAIN_SURVEY_SHARE = REFINE_SHARE
# Where it is not, the space is cut in two across its length, on a line of
# a lattice of the station spacing turned to the space, until each piece is
# plain; a piece no more than this many stations across is not cut again.
SMALLEST_CUT_STEPS = 1.5
# A cut that leaves a piece smaller than this is not made.
CUT_MIN_M2 = 4.0
# Open water: the bed is one flat face inside a ring this far in from the
# bank, deep enough in that a bank eased to 45 degrees has come down to it.
BED_RING_M = 2.0


def _own_angle(polygon, size) -> float:
    """The direction a piece of ground runs in, degrees in [-45, 45).

    The length-weighted direction of its outline modulo 90 degrees -- a side
    and the side square to it agree -- leaving out the edge of the area,
    which runs north-south whatever the piece does. A piece with no outline
    of its own takes 0.
    """
    lengths, angles = [], []
    for part in _parts(polygon):
        for ring in [part.exterior, *part.interiors]:
            c = np.asarray(ring.coords)[:, :2]
            a, b = c[:-1], c[1:]
            d = b - a
            length = np.hypot(d[:, 0], d[:, 1])
            on_edge = np.zeros(len(a), dtype=bool)
            for i in (0, 1):
                for edge in (0.0, size[i]):
                    on_edge |= (np.abs(a[:, i] - edge) < 1e-3) & (np.abs(b[:, i] - edge) < 1e-3)
            keep = (length > 1e-6) & ~on_edge
            lengths.append(length[keep])
            angles.append(np.degrees(np.arctan2(d[keep, 1], d[keep, 0])) % 90.0)
    if not lengths:
        return 0.0
    length = np.concatenate(lengths)
    angle = np.concatenate(angles)
    if not len(length) or length.sum() <= 0:
        return 0.0
    # On the circle of 90 degrees: the mean of the doubled-up angle.
    theta = np.radians(angle * 4.0)
    mean = np.degrees(np.arctan2((length * np.sin(theta)).sum(), (length * np.cos(theta)).sum()))
    best = (mean / 4.0 + 45.0) % 90.0 - 45.0
    return float(best)


def _lattice(polygon, angle: float, spacing: float, anchor):
    """Lattice points over a polygon's bounds, turned by `angle` about
    `anchor`, with their (i, j) indices from the anchor."""
    x0, y0, x1, y1 = polygon.bounds
    reach = 0.5 * float(np.hypot(x1 - x0, y1 - y0)) + 2.0 * spacing
    span = np.hypot(*(np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)]) - anchor)) + reach
    count = int(np.ceil(span / spacing))
    steps = np.arange(-count, count + 1)
    ii, jj = np.meshgrid(steps, steps)
    ii, jj = ii.ravel(), jj.ravel()
    a = np.radians(angle)
    u = np.array([np.cos(a), np.sin(a)])
    v = np.array([-np.sin(a), np.cos(a)])
    points = anchor + spacing * (ii[:, None] * u + jj[:, None] * v)
    inside = (
        (points[:, 0] >= x0 - spacing) & (points[:, 0] <= x1 + spacing)
        & (points[:, 1] >= y0 - spacing) & (points[:, 1] <= y1 + spacing)
    )
    return points[inside], ii[inside], jj[inside]


def _refined(polygon, angle, fine: float, anchor, height_at, tolerance_m: float):
    """The fine lattice points of every coarse square that would miss the survey.

    The coarse lattice is every other point of the fine one. At the fine
    points along a coarse square's sides and in its middle, its corners give
    a height by bilinear interpolation; where any is more than
    `REFINE_SHARE` of the tolerance off the survey, the whole square goes to
    the fine spacing -- all five points, so a patch of rough ground is a
    patch of small quads, joined to the big ones by triangles round its edge.
    One point at a time scattered triangles all over a dike.
    """
    import shapely

    points, ii, jj = _lattice(polygon, angle, fine, anchor)
    if not len(points):
        return np.zeros((0, 2))
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(ii, jj))}
    heights = np.asarray(height_at(points), dtype=np.float64)
    inside = shapely.contains_xy(polygon, points[:, 0], points[:, 1])
    middles = ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2))
    chosen = set()
    for i0, j0 in {(int(i) - int(i) % 2, int(j) - int(j) % 2) for i, j in zip(ii, jj)}:
        corners = [index.get((i0 + di, j0 + dj)) for di in (0, 2) for dj in (0, 2)]
        if any(c is None for c in corners):
            continue
        z = heights[corners]
        if not np.isfinite(z).all():
            continue
        ks = [index.get((i0 + di, j0 + dj)) for di, dj in middles]
        if any(k is None for k in ks) or not any(inside[k] for k in ks):
            continue
        for (di, dj), k in zip(middles, ks):
            if not np.isfinite(heights[k]):
                continue
            s, t = di / 2.0, dj / 2.0
            guess = (z[0] * (1 - s) * (1 - t) + z[2] * s * (1 - t)
                     + z[1] * (1 - s) * t + z[3] * s * t)
            if abs(heights[k] - guess) > REFINE_SHARE * tolerance_m:
                chosen.update(ks)
                break
    keep = [k for k in chosen if inside[k]]
    return points[keep] if keep else np.zeros((0, 2))


def _edge_loop(polygon, offset: float, size, only=None):
    """Points `offset` in from each corner of a polygon's outline, along the
    inward bisector: an edge loop that follows the outline. Not along the edge
    of the area, and with `only` (a geometry) only where it is near that."""
    import shapely

    from .terrain_mesh import _on_frame, _ring_frames

    out = []
    bounds = (0.0, 0.0, float(size[0]), float(size[1]))
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords)[:-1, :2]
        if len(coords) < 3:
            continue
        bisector, cos_half, _, _, _ = _ring_frames(coords)
        usable = ~_on_frame(coords, bounds) & (cos_half >= 0.3)
        if only is not None:
            usable &= shapely.dwithin(only, shapely.points(coords), 1e-6)
        out.append(coords[usable] + offset * bisector[usable])
    if not out:
        return np.zeros((0, 2))
    points = np.vstack(out)
    if not len(points):
        return points
    inside = shapely.contains_xy(polygon, points[:, 0], points[:, 1])
    points = points[inside]
    if len(points):
        clear = shapely.distance(polygon.boundary, shapely.points(points)) > 0.6 * offset
        points = points[clear]
    return points


def _is_ladder_cell(cell) -> bool:
    """Four corners, convex: two ribs and the outline between them."""
    from .terrain_mesh import _ccw, _quad_score

    if len(cell.interiors):
        return False
    coords = np.asarray(cell.exterior.coords)[:-1, :2]
    return len(coords) == 4 and _quad_score(_ccw(coords)) is not None


def _rib_cells(region, rungs):
    """A region cut along its ribs: the ladder cells, and what is left as
    open cells.

    A rib is worth keeping where it is a rung of a ladder -- with a quad on
    at least one side of it. Across a square or a forecourt the ribs reach
    from one side to the other on a slant, cut it into polygons bounded by
    long edges with nothing on them, and the triangulation of those is fans
    off their ends. So the cells that are not ladder quads are merged back
    into one open cell wherever they touch, which takes those ribs out
    again, and the open cell gets the lattice.
    """
    import shapely

    if not rungs:
        return [region]
    lines = shapely.union_all(np.array([region.boundary, *rungs], dtype=object))
    cells = shapely.get_parts(shapely.polygonize(shapely.get_parts(lines)))
    if not len(cells):
        return [region]
    inside = shapely.contains(region, shapely.point_on_surface(cells))
    cells = [c for c in cells[inside] if c.area > 1e-6]
    if abs(sum(c.area for c in cells) - region.area) > max(1e-3 * region.area, 0.01):
        return [region]
    ladder = [c for c in cells if _is_ladder_cell(c)]
    rest = [c for c in cells if not _is_ladder_cell(c)]
    if len(rest) < 2:
        return cells
    merged = shapely.union_all(np.array(rest, dtype=object), grid_size=None)
    opened = [p for p in _parts(merged) if p.area > 1e-6]
    if abs(sum(p.area for p in opened) - sum(c.area for c in rest)) > 1e-3 * max(
            1.0, sum(c.area for c in rest)):
        return cells
    # The merge must not drop a vertex off the outline: a rib's end left on
    # the region's edge is a vertex its neighbour has too.
    keep = {(round(x, 6), round(y, 6)) for c in rest
            for x, y in np.asarray(c.exterior.coords)[:-1, :2]}
    restored = []
    for polygon in opened:
        restored.append(_with_vertices(polygon, keep))
    return ladder + restored


def _with_vertices(polygon, wanted):
    """A polygon with any of the `wanted` points that lie on its outline, but
    were dropped from it, put back as vertices."""
    import shapely

    rings = []
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords)[:, :2]
        have = {(round(x, 6), round(y, 6)) for x, y in coords}
        missing = np.array([p for p in wanted if p not in have], dtype=np.float64)
        if not len(missing):
            rings.append(coords)
            continue
        segments = shapely.linestrings(np.stack([coords[:-1], coords[1:]], axis=1))
        tree = shapely.STRtree(shapely.points(missing))
        seg, point = tree.query(segments, predicate="dwithin", distance=1e-6)
        extra: dict[int, list] = {}
        for a, b in zip(seg, point):
            extra.setdefault(int(a), []).append(missing[int(b)])
        out = []
        for k in range(len(coords) - 1):
            out.append(coords[k])
            if k in extra:
                q = np.array(extra[k])
                along = np.linalg.norm(q - coords[k], axis=1)
                out.extend(q[np.argsort(along)])
        out.append(coords[-1])
        rings.append(np.array(out))
    return shapely.Polygon(rings[0], rings[1:])


def _plain(piece, samples, height_at, tolerance_m: float) -> bool:
    """Whether one face over `piece` would draw the ground in it.

    Its corners, at their surveyed heights, must lie within
    `PLAIN_CORNER_SHARE` of the tolerance of the plane through them, and the
    survey inside it (`samples`, ``(n, 3)``) within `PLAIN_SURVEY_SHARE`.
    """
    import shapely

    ring = np.asarray(piece.exterior.coords)[:-1, :2]
    z = np.asarray(height_at(ring), dtype=np.float64)
    if len(ring) < 3 or not np.isfinite(z).all():
        return False
    centre = ring.mean(axis=0)
    a = np.column_stack([ring - centre, np.ones(len(ring))])
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    if np.abs(a @ coef - z).max() > PLAIN_CORNER_SHARE * tolerance_m:
        return False
    if len(samples):
        x0, y0, x1, y1 = piece.bounds
        near = samples[
            (samples[:, 0] >= x0) & (samples[:, 0] <= x1)
            & (samples[:, 1] >= y0) & (samples[:, 1] <= y1)
        ]
        if len(near):
            near = near[shapely.contains_xy(piece, near[:, 0], near[:, 1])]
        if len(near):
            drawn = (near[:, :2] - centre) @ coef[:2] + coef[2]
            if np.abs(drawn - near[:, 2]).max() > PLAIN_SURVEY_SHARE * tolerance_m:
                return False
    return True


def _cut(piece, u, v, at: float, anchor, fixed, size=None):
    """`piece` cut along the line across `u` at `at` from `anchor`.

    Each stretch of the line inside the piece is a cut from where it meets
    the outline to where it leaves it. On the space's own outline (`fixed`)
    an end goes to the nearer corner of the edge it meets, so the neighbour
    across that outline keeps every vertex it shares and gains none; on an
    earlier cut it stays where it meets it, and the piece across that cut
    takes the new vertex in later (`_conformed`), and so it does on the edge
    of the area, which no neighbour shares. Returns the pieces, or None where
    the line makes no clean cut.
    """
    import shapely

    x0, y0, x1, y1 = piece.bounds
    reach = float(np.hypot(x1 - x0, y1 - y0)) + 1.0
    middle = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    base = middle + (at - float((middle - anchor) @ u)) * u
    line = shapely.LineString([base - reach * v, base + reach * v])
    hit = shapely.intersection(piece.boundary, line)
    if any(g.geom_type != "Point" for g in shapely.get_parts(hit)):
        return None  # along an edge of the outline
    points = shapely.get_coordinates(hit)
    if len(points) < 2:
        return None
    order = np.argsort((points - base) @ v)
    points = points[order]
    rings = [np.asarray(r.coords)[:, :2] for r in [piece.exterior, *piece.interiors]]
    starts = np.vstack([r[:-1] for r in rings])
    ends = np.vstack([r[1:] for r in rings])

    def placed(p):
        d = ends - starts
        t = np.clip(((p - starts) * d).sum(axis=1) / np.maximum((d * d).sum(axis=1), 1e-18),
                    0.0, 1.0)
        gap = np.hypot(*(starts + t[:, None] * d - p).T)
        k = int(np.argmin(gap))
        a, b = starts[k], ends[k]
        if np.hypot(*(p - a)) < 1e-6:
            return a
        if np.hypot(*(p - b)) < 1e-6:
            return b
        on_cut = fixed is not None and not shapely.dwithin(
            fixed, shapely.Point(0.5 * (a + b)), 1e-6)
        on_frame = size is not None and any(
            abs(a[i] - edge) < 1e-6 and abs(b[i] - edge) < 1e-6
            for i in (0, 1) for edge in (0.0, size[i])
        )
        if on_cut or on_frame:
            return p
        return a if np.hypot(*(p - a)) <= np.hypot(*(p - b)) else b

    shapely.prepare(piece)
    cuts = []
    for a, b in zip(points[:-1], points[1:]):
        if not shapely.contains_xy(piece, *(0.5 * (a + b))):
            continue
        a, b = placed(a), placed(b)
        if np.hypot(*(b - a)) < 1e-3:
            continue
        chord = shapely.LineString([a, b])
        mid = 0.5 * (a + b)
        if not shapely.contains_xy(piece, *mid) or shapely.distance(
                piece.boundary, shapely.Point(mid)) < 1e-3:
            continue
        # Clear of the outline between its ends: an end met on an edge lies on
        # it only to the rounding, so the ends themselves are not asked.
        trim = 1e-4 / max(float(np.hypot(*(b - a))), 1e-9)
        inner = shapely.LineString([a + trim * (b - a), b - trim * (b - a)])
        if shapely.intersects(inner, piece.boundary):
            continue
        cuts.append(chord)
    if not cuts:
        return None
    # The ends that fall between corners made corners first: met only to the
    # rounding, the cut would not be noded to the outline.
    ends = np.array([c for cut in cuts for c in np.asarray(cut.coords)])
    piece = _with_vertices_near(piece, ends)
    lines = shapely.union_all(np.array([piece.boundary, *cuts], dtype=object))
    cells = shapely.get_parts(shapely.polygonize(shapely.get_parts(lines)))
    if len(cells) < 2:
        return None
    inside = shapely.contains_xy(
        piece, *shapely.get_coordinates(shapely.point_on_surface(cells)).T)
    cells = [c for c in cells[inside] if c.area > 1e-6]
    if len(cells) < 2 or min(c.area for c in cells) < CUT_MIN_M2:
        return None
    if abs(sum(c.area for c in cells) - piece.area) > 1e-6 * max(piece.area, 1.0) + 1e-4:
        return None
    return [shapely.orient_polygons(c) for c in cells]


def _conformed(pieces):
    """Every piece with each corner of another that lies on its outline put in.

    A cut that ends on an earlier cut is a vertex on that cut's edge in the
    piece across it too, or the two would meet in a T and crack open.
    """
    import shapely

    corners = {}
    for piece in pieces:
        for ring in [piece.exterior, *piece.interiors]:
            for x, y in np.asarray(ring.coords)[:-1, :2]:
                corners[(round(float(x), 6), round(float(y), 6))] = (float(x), float(y))
    if not corners:
        return pieces
    wanted = np.array(list(corners.values()))
    tree = shapely.STRtree(shapely.points(wanted))
    out = []
    for piece in pieces:
        near = tree.query(piece.boundary, predicate="dwithin", distance=1e-5)
        if not len(near):
            out.append(piece)
            continue
        out.append(_with_vertices_near(piece, wanted[near]))
    return out


def _with_vertices_near(polygon, points, within: float = 1e-5):
    """`_with_vertices` for points that may lie a hair off the outline."""
    import shapely

    rings = []
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords)[:, :2]
        have = {(round(x, 6), round(y, 6)) for x, y in coords}
        out = []
        for k in range(len(coords) - 1):
            a, b = coords[k], coords[k + 1]
            out.append(a)
            d = b - a
            length2 = float(d @ d)
            if length2 <= 0:
                continue
            t = ((points - a) @ d) / length2
            foot = a + np.clip(t, 0, 1)[:, None] * d
            gap = np.hypot(*(points - foot).T)
            on = (t > 1e-9) & (t < 1 - 1e-9) & (gap < within)
            for i in np.flatnonzero(on)[np.argsort(t[on])]:
                key = (round(float(points[i, 0]), 6), round(float(points[i, 1]), 6))
                if key not in have:
                    out.append(points[i])
                    have.add(key)
        out.append(coords[-1])
        rings.append(np.array(out))
    result = shapely.Polygon(rings[0], rings[1:])
    return result if result.is_valid else polygon


def _open_layout(cell, *, angle, anchor, step, height_at, tolerance_m, sample_m, size):
    """An open space cut into as few pieces as its ground allows.

    Returns ``(polygon, plain)`` per piece: a plain piece is one face, its
    outline only. Where one face would miss the ground the piece is cut in
    two across its longer side, on the line of a lattice of `step` turned to
    the space nearest its middle, so the cuts in neighbouring pieces line up;
    a piece no more than `SMALLEST_CUT_STEPS` steps across is not cut again,
    plain or not. A piece with a hole in it is cut through the hole, as a
    face cannot have one. A cut that ends on another piece's edge gives that
    piece a corner, and a plain piece whose new corners take it out of its
    plane is cut in turn.
    """
    import shapely

    a = np.radians(angle)
    u = np.array([np.cos(a), np.sin(a)])
    v = np.array([-np.sin(a), np.cos(a)])
    x0, y0, x1, y1 = cell.bounds
    gx = np.arange(x0 + 0.5 * sample_m, x1, sample_m)
    gy = np.arange(y0 + 0.5 * sample_m, y1, sample_m)
    samples = np.zeros((0, 3))
    if len(gx) and len(gy):
        sx, sy = np.meshgrid(gx, gy)
        xy = np.column_stack([sx.ravel(), sy.ravel()])
        xy = xy[shapely.contains_xy(cell, xy[:, 0], xy[:, 1])]
        if len(xy):
            z = np.asarray(height_at(xy), dtype=np.float64)
            samples = np.column_stack([xy, z])[np.isfinite(z)]

    fixed = shapely.union_all(np.array(
        [r for r in [cell.exterior, *cell.interiors]], dtype=object))
    shapely.prepare(fixed)
    smallest = SMALLEST_CUT_STEPS * step
    leaves = []
    stack = [shapely.orient_polygons(cell)]
    budget = 20000
    for _ in range(8):
        while stack:
            piece = stack.pop()
            budget -= 1
            holes = len(piece.interiors) > 0
            if not holes and _plain(piece, samples, height_at, tolerance_m):
                leaves.append((piece, True))
                continue
            coords = np.vstack([np.asarray(r.coords)[:, :2]
                                for r in [piece.exterior, *piece.interiors]]) - anchor
            pu, pv = coords @ u, coords @ v
            spans = [(pu.max() - pu.min(), u, v, pu), (pv.max() - pv.min(), v, -u, pv)]
            spans.sort(key=lambda s: -s[0])
            if budget <= 0 or (not holes and spans[0][0] <= smallest):
                leaves.append((piece, False))
                continue
            tries = []
            if holes:
                hole = shapely.point_on_surface(shapely.Polygon(piece.interiors[0]))
                h = np.asarray(hole.coords[0]) - anchor
                for _, axis, across, _ in spans:
                    tries.append((axis, across, float(h @ axis)))
            else:
                for span, axis, across, along in spans:
                    if span <= smallest:
                        continue
                    lo, hi = float(along.min()), float(along.max())
                    middle = 0.5 * (lo + hi)
                    on_lattice = sorted(
                        (k * step for k in range(int(np.floor(lo / step)),
                                                 int(np.ceil(hi / step)) + 1)
                         if lo + 0.25 * step < k * step < hi - 0.25 * step),
                        key=lambda c: abs(c - middle),
                    )
                    for at in on_lattice[:3] or [middle]:
                        tries.append((axis, across, at))
            made = None
            for axis, across, at in tries:
                made = _cut(piece, axis, across, at, anchor, fixed, size)
                if made is not None:
                    break
            if made is None:
                leaves.append((piece, False))
                continue
            stack.extend(made)
        before = [len(p.exterior.coords) for p, _ in leaves]
        polygons = _conformed([p for p, _ in leaves])
        kept, again = [], []
        for polygon, (_, plain), count in zip(polygons, leaves, before):
            if (plain and len(polygon.exterior.coords) > count
                    and not _plain(polygon, samples, height_at, tolerance_m)):
                again.append(polygon)
            else:
                kept.append((polygon, plain))
        leaves = kept
        if not again:
            break
        stack = again
    LOG.debug("open layout: %d pieces (%d plain), budget left %d, %d left uncut",
              len(leaves), sum(1 for _, plain in leaves if plain), budget, len(stack))
    return leaves + [(p, False) for p in stack]


def _bed_fill(cell, *, loop, near_m, angle, anchor, size):
    """Open water as its bank and one flat face for the bed.

    The bed is the water `BED_RING_M` in from its outline, with a station
    every `near_m` round it; it lies flat at the bed's depth, so it is one
    face, cut only through an island. The bank between the outline and the
    bed is triangulated through the edge loop along it (`loop`), and paired
    into quads. None where there is no bed that wide.
    """
    import shapely

    from .terrain_mesh import FILL, GRID_SIZE, OPEN, ROW, _ccw, _fill, _fill_pinned, _quad_score

    bed = cell.buffer(-BED_RING_M, join_style="mitre", mitre_limit=2.0)
    bed = shapely.set_precision(shapely.segmentize(bed, near_m), GRID_SIZE)
    parts = [p for p in _parts(bed) if p.area >= SMALL_CELL_SQUARES * near_m * near_m]
    if not parts:
        return None
    bed = shapely.union_all(np.array(parts, dtype=object))
    bank = shapely.difference(cell, bed)
    out = []
    for piece in _parts(bank):
        if piece.area <= 1e-6:
            continue
        faces = _fill_pinned(piece, np.zeros((0, 2)), 0.0, kept=loop)
        out.extend((c, ROW if len(c) == 4 else FILL) for c in faces)
    flat = lambda points: np.zeros(len(np.asarray(points).reshape(-1, 2)))  # noqa: E731
    for part in _parts(bed):
        for piece, _ in _open_layout(part, angle=angle, anchor=anchor, step=near_m,
                                     height_at=flat, tolerance_m=1.0, sample_m=1e9,
                                     size=size):
            ring = _ccw(np.asarray(piece.exterior.coords)[:-1, :2])
            if len(piece.interiors):
                out.extend((c, FILL) for c in _fill(piece))
            elif len(ring) == 4 and _quad_score(ring) is not None:
                out.append((ring, ROW))
            elif len(ring) > 4:
                out.append((ring, OPEN))
            else:
                out.extend((c, FILL) for c in _fill(piece))
    covered = sum(abs(float(shapely.Polygon(c).area)) for c, _ in out)
    if abs(covered - cell.area) > max(1e-3 * cell.area, 0.01):
        return None
    return out, 0


def _rib_fill(cell, *, kind, near_m, size, height_at, tolerance_m, quay_zone=None,
              laddered=False, open_faces=True, sample_m=2.0):
    """One cell as faces: a ladder rung pair as one quad, a small cell from its
    outline, an open one as a face per plain stretch of it (`open_faces`; the
    lattice where the ground is not plain) or through a lattice turned to it.
    `laddered` says the region's ribs are its ladder (narrow water: the bed is
    carved by the loop the rungs are divided at, not by a loop along its
    banks)."""
    import shapely

    from .terrain_mesh import FILL, OPEN, ROW, WATER, _ccw, _fill, _fill_pinned, _quad_score

    cell = shapely.orient_polygons(cell)
    coords = np.asarray(cell.exterior.coords)[:-1, :2]
    if not len(cell.interiors) and len(coords) == 4 and (kind != WATER or laddered):
        quad = _ccw(coords)
        if _quad_score(quad) is not None:
            return [(quad, ROW)], 0
    wet = kind == WATER
    loop = np.zeros((0, 2))
    if wet:
        loop = _edge_loop(cell, BANK_LOOP_M, size)
        if quay_zone is not None:
            along_quay = _edge_loop(cell, QUAY_ROW_M, size, only=quay_zone)
            if len(along_quay):
                near = shapely.dwithin(quay_zone, shapely.points(loop), 1e-6) if len(loop) else []
                loop = np.vstack([loop[~np.asarray(near, dtype=bool)] if len(loop) else loop,
                                  along_quay])
    if cell.area < SMALL_CELL_SQUARES * near_m * near_m and not len(loop):
        return [(c, FILL) for c in _fill(cell)], 0
    angle = _own_angle(cell, size)
    anchor = np.asarray(cell.centroid.coords[0])
    coarse = OPEN_FACTOR * near_m

    def lattice_fill(piece, anchor, kept_points):
        points, _, _ = _lattice(piece, angle, coarse, anchor)
        extra = _refined(piece, angle, near_m, anchor, height_at, tolerance_m)
        if len(extra):
            clear = shapely.distance(piece.boundary, shapely.points(extra)) > 0.45 * near_m
            extra = extra[clear]
        kept = np.vstack([extra, kept_points]) if len(kept_points) else extra
        faces = _fill_pinned(piece, points, 0.45 * coarse, kept=kept)
        return [(c, ROW if len(c) == 4 else FILL) for c in faces], len(extra)

    if not open_faces:
        return lattice_fill(cell, anchor, loop)
    if wet:
        try:
            faces = _bed_fill(cell, loop=loop, near_m=near_m, angle=angle, anchor=anchor,
                              size=size)
        except Exception as exc:  # noqa: BLE001 - the lattice always works
            LOG.debug("ribcage: open water given the lattice: %s", exc)
            faces = None
        return faces if faces else lattice_fill(cell, anchor, loop)
    out, refined = [], 0
    for piece, plain in _open_layout(cell, angle=angle, anchor=anchor, step=near_m,
                                     height_at=height_at, tolerance_m=tolerance_m,
                                     sample_m=sample_m, size=size):
        ring = _ccw(np.asarray(piece.exterior.coords)[:-1, :2])
        if len(piece.interiors):
            faces, extra = lattice_fill(piece, anchor, np.zeros((0, 2)))
            out.extend(faces)
            refined += extra
        elif len(ring) == 4 and _quad_score(ring) is not None:
            out.append((ring, ROW))
        elif len(ring) > 4 and plain:
            out.append((ring, OPEN))
        else:
            out.extend((c, FILL) for c in _fill(piece))
    return out, refined


def ribcage_terrain(terrain, pieces, bodies, *, codes_wet, depth_m,
                    paved_codes, narrow_codes=(), near_m: float = NEAR_CELL_M, quays=(),
                    tolerance_m: float = 0.5, open_faces: bool = True):
    """The terrain mesh as a ribcage, conforming to every piece's outline.

    The same inputs and outputs as `layered_mesh.quad_terrain`, so the two
    are interchangeable: `pieces` are ``(polygon, code)`` in local
    coordinates from the first grid node; `bodies` the water bodies as
    ``(local polygon, level NAP)``. Returns RD vertices with heights, face
    sizes, loops, the code and part of each face, and stats.
    """
    import shapely

    from .terrain_mesh import (
        FILL,
        GRID_SIZE,
        LAND,
        OPEN,
        WATER,
        _cross_rungs,
        _fill,
        _fill_gaps,
    )

    started = time.time()
    g = float(terrain.grid_m)
    k = max(1, int(round(float(near_m) / g)))
    near_m = k * g
    ox, oy = float(terrain.xs[0]), float(terrain.ys[0])
    width = float(np.round(terrain.xs[-1] - ox, 6))
    height = float(np.round(terrain.ys[-1] - oy, 6))
    size = (width, height)
    shift = np.array([ox, oy])
    centre = np.array([0.5 * width, 0.5 * height])
    frame = shapely.box(0.0, 0.0, width, height)
    margin_m = 0.35 * g

    filled = _fill_gaps(np.array([p for p, _ in pieces], dtype=object), frame)
    pieces = [(part, code) for polygon, (_, code) in zip(filled, pieces)
              for part in _parts(polygon)]
    pieces, overlaps = _without_overlaps(pieces)
    pieces, thin = _without_thin(pieces, THIN_M, wet_code=codes_wet)

    parts, codes = [], []
    for polygon, code in pieces:
        polygon = _with_stations(polygon, lambda mid, c: near_m, centre, size)
        polygon = shapely.orient_polygons(shapely.set_precision(polygon, GRID_SIZE))
        for piece in _parts(polygon):
            if piece.area > 1e-4:
                parts.append((piece, WATER if code == codes_wet else LAND))
                codes.append(int(code))
    quay_zone = None
    quay_parts = set()
    if quays:
        quay_zone = shapely.union_all(np.array(
            [shapely.transform(q, lambda c: c - shift) for q in quays], dtype=object
        )).buffer(QUAY_REACH_M)
        shapely.prepare(quay_zone)
        for i, (polygon, kind) in enumerate(parts):
            if kind == WATER and polygon.boundary.intersection(quay_zone).length > QUAY_MIN_M:
                quay_parts.add(i)
    # Water gets ribs like land: a ditch or a narrow canal is a ladder whose
    # rungs are divided down its middle, and that loop is the bed. Along a
    # quay it does not: the ground drops sheer there, in the loop a hand's
    # breadth from the wall, not across half the canal.
    rows = [
        PAVED_ROW_M if c in paved_codes
        else g if (c in narrow_codes or i in quay_parts)
        else max(LAND_ROW_M, g)
        for i, ((_, kind), c) in enumerate(zip(parts, codes))
    ]

    timings = {}
    t = time.time()
    parts, across = _cross_rungs(
        parts, frame, row_m=g, margin_m=margin_m, rows=rows, square_deg=SQUARE_DEG,
        snap_m=0.5 * near_m,
    )
    timings["ribs_s"] = round(time.time() - t, 1)

    def height_at(points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return np.asarray(
            terrain.sample(points[:, 0] + shift[0], points[:, 1] + shift[1]), dtype=np.float64
        )

    faces = []
    ladder_cells = open_cells = refined_points = refilled = divided = crowned = 0
    t = time.time()
    for index, (region, kind) in enumerate(parts):
        out = []
        try:
            for cell in _rib_cells(region, across.get(index, ())):
                cell_faces, extra = _rib_fill(
                    cell, kind=kind, near_m=near_m, size=size, height_at=height_at,
                    tolerance_m=tolerance_m,
                    quay_zone=quay_zone if index in quay_parts else None,
                    laddered=bool(across.get(index)),
                    open_faces=open_faces, sample_m=g,
                )
                if len(cell_faces) == 1 and len(cell_faces[0][0]) == 4:
                    ladder_cells += 1
                else:
                    open_cells += 1
                refined_points += extra
                out.extend(cell_faces)
        except Exception as exc:  # noqa: BLE001 - one region, not the terrain
            LOG.debug("ribcage: region %d filled from its outline: %s", index, exc)
            out = []
        covered = sum(_area(c) for c, _ in out)
        if not out or abs(covered - region.area) > max(1e-3 * region.area, 0.01):
            out = [(c, FILL) for c in _fill(region)]
            refilled += 1
        if kind == WATER and across.get(index):
            # At least once down the middle: the bed.
            lengths = [line.length for line in across[index]]
            count = int(np.clip(np.round(np.median(lengths) / near_m), 2, 4))
            out, n_divided = _divide_rungs(out, across[index], near_m, count=count)
            divided += n_divided
        elif kind == LAND and codes[index] not in paved_codes and across.get(index):
            out, n_divided = _divide_rungs(out, across[index], near_m)
            divided += n_divided
        elif (kind == LAND and across.get(index)
              and _rungs_miss(across[index], height_at) > tolerance_m):
            out, n_divided = _divide_rungs(out, across[index], near_m, count=2)
            divided += n_divided
            crowned += 1
        faces.extend((c, p, kind, codes[index]) for c, p in out)
    timings["regions_meshed_s"] = round(time.time() - t, 1)

    vertices, sizes, loops, face_kind, face_part, face_code, finished = _assemble(
        faces, terrain=terrain, bodies=bodies, depth_m=depth_m, shift=shift,
        size=size, quay_zone=quay_zone,
    )
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    open_area = 0.0
    for f in np.flatnonzero(face_part == OPEN):
        ring = vertices[loops[starts[f]: starts[f] + sizes[f]], :2]
        ring = ring - ring.min(axis=0)
        open_area += 0.5 * abs(float(np.sum(
            ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])))
    stats = {
        "mesher": "ribcage",
        "ribs": int(sum(len(v) for v in across.values())),
        "ladder_cells": int(ladder_cells),
        "open_cells": int(open_cells),
        "open_quad_m": float(OPEN_FACTOR * near_m),
        "open_faces": int((face_part == OPEN).sum()),
        "open_faces_m2": round(float(open_area), 1),
        "points_refined_to_station_spacing": int(refined_points),
        "rungs_divided": int(divided),
        "roads_given_a_middle_loop": int(crowned),
        "regions_filled_whole": int(refilled),
        "pieces_overlapping": int(overlaps),
        "thin_strips_folded": int(thin),
        "water_along_quays": int(len(quay_parts)),
        **finished,
        "water_faces": int((face_kind == WATER).sum()),
        **timings,
        "mesher_s": round(time.time() - started, 1),
    }
    return vertices, sizes.astype(np.int64), loops, face_code, face_part, stats
