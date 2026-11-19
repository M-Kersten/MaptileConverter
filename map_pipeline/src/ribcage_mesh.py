"""The layered style's ground as a ribcage: the 3D team's layout, beside the grid.

What the 3D team asked for, with the grid mesher's Hoek van Holland in front
of them: areas designed like a ribcage, where quads do the work across the
large spaces, triangles fill in at the edges where the outline needs them,
and the quads are never just the underlying grid.

So there is no grid. Every region -- a street, a verge, a square, a field, a
canal -- is meshed in its own frame:

* **Ribs across what is narrow.** A street, a footpath, a verge: ribs run
  straight across it from station to station along its two outlines, the
  same rungs the grid mesher lays (`terrain_mesh._cross_rungs`), so it is a
  ladder of quads that follows the street round its bends. Nothing else is
  put in a ladder cell; a road whose crown or dike the survey shows gets one
  loop down its middle, as with the grid.
* **A lattice turned to the space in what is wide.** What the ribs leave --
  a square, a field, a junction, the water -- gets a lattice of quads turned
  to that piece's own main direction (`_own_angle`), not to a grid shared
  with everything else. Twice the station spacing where the ground is plain
  enough for it; where the survey shows the ground doing something a big
  quad would miss by more than the tolerance, the station spacing there
  (`_refined`). The lattice is joined to the outline by triangles, and every
  pair of those that makes a good quad is one.
* **Banks get a loop.** Water gets an edge loop a little way in from its
  outline, so the bed is carved even in a ditch too narrow for the lattice,
  and a hand's breadth in along a quay wall, where the ground drops sheer.

The outlines carry the same stations on both sides (`layered_mesh.
_with_stations`), so two regions meet vertex for vertex whatever each does
inside; the finishing -- welding, mending, heights, beds, diagonals -- is the
grid mesher's own (`layered_mesh._assemble`), so the two can be compared on
their layout alone. Choose it with ``layered.mesher: "ribcage"``.
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


def _rib_fill(cell, *, kind, near_m, size, height_at, tolerance_m, quay_zone=None,
              laddered=False):
    """One cell as faces: a ladder rung pair as one quad, a small cell from its
    outline, an open one through a lattice turned to it. `laddered` says the
    region's ribs are its ladder (narrow water: the bed is carved by the loop
    the rungs are divided at, not by a loop along its banks)."""
    import shapely

    from .terrain_mesh import FILL, ROW, WATER, _ccw, _fill, _fill_pinned, _quad_score

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
    points, _, _ = _lattice(cell, angle, coarse, anchor)
    extra = _refined(cell, angle, near_m, anchor, height_at, tolerance_m)
    if len(extra):
        clear = shapely.distance(cell.boundary, shapely.points(extra)) > 0.45 * near_m
        extra = extra[clear]
    kept = np.vstack([extra, loop]) if len(loop) else extra
    faces = _fill_pinned(cell, points, 0.45 * coarse, kept=kept)
    return [(c, ROW if len(c) == 4 else FILL) for c in faces], len(extra)


def ribcage_terrain(terrain, pieces, bodies, *, codes_wet, depth_m,
                    paved_codes, narrow_codes=(), near_m: float = NEAR_CELL_M, quays=(),
                    tolerance_m: float = 0.5, kerbed_codes=()):
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
    pieces, thin = _without_thin(pieces, THIN_M, wet_code=codes_wet,
                                 kerbed_codes=kerbed_codes)

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
    stats = {
        "mesher": "ribcage",
        "ribs": int(sum(len(v) for v in across.values())),
        "ladder_cells": int(ladder_cells),
        "open_cells": int(open_cells),
        "open_quad_m": float(OPEN_FACTOR * near_m),
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
