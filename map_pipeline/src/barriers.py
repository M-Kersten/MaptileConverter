"""Walls, fences, hedges and the small structures bolted onto buildings.

This is the largest thing the BGT publishes that the pipeline used to ignore.
Over one square kilometre of Utrecht centre:

    scheiding_lijn        1563   fence 1174, wall 332, quay wall 36, barrier 4
    scheiding_vlak        1450   wall 1330, quay wall 120
    gebouwinstallatie      114   entrance steps 55, awning 53, stoop 6
    overigbouwwerk          82   transformer 6, canopy 4, open shed 2
    vegetatieobject_lijn    34   hedge 34

Three thousand walls and fences is not decoration. They are what divides one
plot from the next, and a street without them reads as a set of buildings
standing in a field -- which is exactly what the model looked like. The quay
walls matter more still: an Utrecht canal without its wall is a trench.

None of it carries a height. The BGT surveys the footprint of a wall and the
line of a fence, and stops there. So the heights come from a table by type,
which is honest about being an assumption and puts it somewhere a level
designer can change it, rather than from AHN -- a fence is thinner than the
half-metre height grid can see, and a wall against a building takes the
building's height, which is the same trap the tree heights had to be dug out
of.

Two families of geometry, and they need different treatment:

* **Lines** sweep into a closed prism: offset either side by half the
  thickness, drape both edges on the terrain, and close the top and the ends.
  A fence is 6 cm thick and a wall 30, but neither is a plane -- a
  zero-thickness wall shows as a hole from one side in anything with backface
  culling on.
* **Polygons** extrude: the ring becomes vertical quads and the top gets a cap.
  The cap is earcut, which fans a thin strip into slivers -- see roads for
  how much that matters on a surface. It does not matter here: the top of a
  30 cm wall is not lit, walked on, or ever seen from above, and the sides,
  which are the part you look at, are clean quads by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .bgt import fetch_current, polygon_rings
from .geo import BBox
from .rails import clip_line_to_bbox, densify, offset_normals
from .surfaces import clip_ring_to_bbox

LOG = logging.getLogger(__name__)

# How far apart to put vertices along a swept line. A garden wall follows the
# ground and the ground moves; 2 m is fine enough for a Dutch gradient and
# coarse enough that 1563 fences stay cheap.
DEFAULT_STEP_M = 2.0

# Nothing shorter than this is worth a prism. Below about a metre a fence
# fragment is a survey artefact -- the end of a run clipped by a bbox, or a
# gate post recorded as its own object.
MIN_LENGTH_M = 0.8

# A barrier sinks this far into the ground, so a prism whose base was sampled at
# one point does not float at the other end of a slope. Only for the ones that
# stand on the ground: burying an awning by 15 cm would make a 12 cm awning
# 27 cm thick and hang it lower than it belongs.
EMBED_M = 0.15


def feature_type(properties: dict) -> str:
    """Which BGT field actually holds the type, for these collections.

    It is not the same field twice. `scheiding` and `overigbouwwerk` put the
    value in `type` and leave `plus_type` explicitly null, with
    `plus_type_leeg: waardeOnbekend` to say so; `gebouwinstallatie` and
    `vegetatieobject` do the reverse, carrying `plus_type: toegangstrap` beside
    a useless `type: niet-bgt`. Reading only `plus_type` is why the first run of
    this module fetched 1563 fences and 1450 walls and drew none of them.

    So: whichever is populated, preferring the finer one.
    """
    for key in ("plus_type", "type"):
        value = properties.get(key)
        if value and str(value).lower() not in ("none", "niet-bgt"):
            return str(value)
    return ""


def _drape_points(points: np.ndarray, sampler, step_m: float, tolerance_m: float):
    """Where a swept barrier needs a vertex, and the ground height there.

    Cutting every fence at a fixed step is the obvious thing and it is pure
    waste: Dutch ground barely moves, so 907 fences came to 56,540 triangles,
    most of them holding a straight line straight. What a vertex is *for* is
    the places the ground bends.

    So the line is cut finely, the ground sampled along it, and the inserted
    points thinned against the height profile -- keeping one wherever dropping
    it would let the barrier cut into the ground, or lift off it, by more than
    the tolerance. The survey's own vertices are anchors and always survive,
    because they are the shape of the fence rather than the shape of the
    ground under it.
    """
    from .geo import douglas_peucker_keep as _douglas_peucker_keep

    dense = densify(points, step_m)
    height = np.asarray(sampler(dense[:, 0], dense[:, 1]), dtype=np.float64)
    if len(dense) < 3 or tolerance_m <= 0:
        return dense, height

    # Which of the dense points are the survey's own. densify keeps them, so
    # they are the ones that match an input vertex exactly -- absolutely, not
    # relatively: numpy's default relative tolerance is metres at RD
    # coordinates, which took every corner of a small wall for the first one
    # and the wall down to a point.
    steps = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    anchors = [0]
    at = 0
    for vertex in points[1:]:
        while at < len(dense) - 1 and not np.allclose(dense[at], vertex, rtol=0.0, atol=1e-6):
            at += 1
        if at != anchors[-1]:
            anchors.append(at)
    if anchors[-1] != len(dense) - 1:
        anchors.append(len(dense) - 1)

    # The profile as a 2D line of (distance along, height), so the ordinary
    # perpendicular-distance rule reads as vertical error. True while the
    # gradient is shallow, which in this country it is.
    keep = np.zeros(len(dense), dtype=bool)
    keep[anchors] = True
    profile = np.column_stack([along, height])
    for start, end in zip(anchors[:-1], anchors[1:]):
        if end > start + 1:
            keep[start : end + 1] |= _douglas_peucker_keep(
                profile[start : end + 1], tolerance_m
            )
    return dense[keep], height[keep]


# A surface-model reading inside a canopy's outline counts as its roof between
# these heights above the ground under it: lower is the ground or a parked
# car seen through a gap, higher is a building the canopy runs in under.
ROOF_MIN_M = 2.0
ROOF_MAX_M = 30.0
# And there must be enough of them: under this share of the outline, the
# survey saw through the roof -- glass, a mesh -- and the lift is the better
# guess.
ROOF_SHARE = 0.3
ROOF_SAMPLE_M = 1.0


def _measured_top(rings, ground, surface_at) -> float | None:
    """The height of a canopy's roof from the surface model, or None.

    The style's lift is a bicycle shelter's: 2.6 m up. Stood on that, the
    roof over the platforms at Utrecht Centraal hung a few metres above the
    platform, under its own roof in the survey. The surface model sees the
    roof itself; the median of what it reads inside the outline, at least a
    metre in from the edge where the reading is part roof and part ground, is
    the height of the top.
    """
    import shapely

    polygon = shapely.Polygon(rings[0], rings[1:])
    if not polygon.is_valid:
        polygon = shapely.make_valid(polygon)
    inner = polygon.buffer(-ROOF_SAMPLE_M)
    if inner.is_empty or inner.area < 1.0:
        inner = polygon
    xmin, ymin, xmax, ymax = inner.bounds
    xs, ys = np.meshgrid(
        np.arange(xmin + ROOF_SAMPLE_M / 2, xmax, ROOF_SAMPLE_M),
        np.arange(ymin + ROOF_SAMPLE_M / 2, ymax, ROOF_SAMPLE_M),
    )
    xs, ys = xs.ravel(), ys.ravel()
    inside = shapely.contains_xy(inner, xs, ys)
    if inside.sum() < 3:
        centre = inner.representative_point()
        xs, ys = np.array([centre.x]), np.array([centre.y])
    else:
        xs, ys = xs[inside], ys[inside]
    roof = np.asarray(surface_at(xs, ys), dtype=np.float64).reshape(xs.shape)
    under = np.asarray(ground(xs, ys), dtype=np.float64).reshape(xs.shape)
    above = roof - under
    seen = np.isfinite(above) & (above >= ROOF_MIN_M) & (above <= ROOF_MAX_M)
    if seen.sum() < max(1, int(np.ceil(ROOF_SHARE * len(xs)))):
        return None
    return float(np.median(roof[seen]))


# A canopy whose roof the survey sees at heights this far apart (10th to 90th
# percentile) is not one flat roof but several -- a station hall over its
# platform roofs -- and is draped on the survey rather than laid flat.
ROOF_SPREAD_M = 2.0
# The lattice a draped roof is built on.
ROOF_LATTICE_M = 4.0


def _roof_heights(xs, ys, ground, surface_at):
    """The survey's roof at points: the median of a few readings round each,
    held between `ROOF_MIN_M` and `ROOF_MAX_M` above the ground, NaN where
    there is no such reading."""
    offsets = np.array([[0.0, 0.0], [0.6, 0.0], [-0.6, 0.0], [0.0, 0.6], [0.0, -0.6]])
    readings = np.stack([
        np.asarray(surface_at(xs + dx, ys + dy), dtype=np.float64).reshape(xs.shape)
        for dx, dy in offsets
    ])
    under = np.asarray(ground(xs, ys), dtype=np.float64).reshape(xs.shape)
    above = readings - under[None, :]
    readings = np.where((above >= ROOF_MIN_M) & (above <= ROOF_MAX_M), readings, np.nan)
    import warnings

    with warnings.catch_warnings():
        # A point with no reading at all is NaN, which is the answer.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(readings, axis=0), under


def _draped_roof(rings, ground, surface_at, style):
    """A canopy's roof following the survey, as ``(vertices, triangles, uvs)``,
    or None where one flat roof will do.

    The roof over Utrecht Centraal's platforms is one BGT outline of 3.5
    hectares, and the survey sees it anywhere from 7 to 26 m NAP: the hall
    roof high over the tracks, the platform roofs low beside it. Laid flat at
    its median, most of it was metres off. So where the survey's readings
    spread by more than `ROOF_SPREAD_M`, the roof is a sheet on a
    `ROOF_LATTICE_M` lattice -- the ground's own constrained fill, quads where
    they fit -- with each vertex at the survey's roof there, a slab the
    style's thickness under it, and its edge closed all round.
    """
    import shapely

    from .terrain_mesh import _fill_pinned, _key

    polygon = shapely.Polygon(rings[0], rings[1:])
    if not polygon.is_valid or polygon.area < 4.0 * ROOF_LATTICE_M ** 2:
        return None
    x0, y0, x1, y1 = polygon.bounds
    gx, gy = np.meshgrid(
        np.arange(x0 + ROOF_LATTICE_M / 2, x1, ROOF_LATTICE_M),
        np.arange(y0 + ROOF_LATTICE_M / 2, y1, ROOF_LATTICE_M),
    )
    inner = np.column_stack([gx.ravel(), gy.ravel()])
    inside = shapely.contains_xy(polygon, inner[:, 0], inner[:, 1])
    if inside.sum() < 4:
        return None
    probe, _ = _roof_heights(inner[inside, 0], inner[inside, 1], ground, surface_at)
    seen = probe[np.isfinite(probe)]
    if len(seen) < ROOF_SHARE * len(probe):
        return None
    if np.percentile(seen, 90) - np.percentile(seen, 10) <= ROOF_SPREAD_M:
        return None

    faces = _fill_pinned(polygon, inner, 0.45 * ROOF_LATTICE_M)
    if not faces:
        return None
    index: dict[tuple, int] = {}
    points: list = []
    loops = []
    for face in faces:
        corners = []
        for p in np.asarray(face)[:, :2]:
            key = _key(p)
            if key not in index:
                index[key] = len(points)
                points.append(p)
            corners.append(index[key])
        loops.append(corners)
    points = np.asarray(points, dtype=np.float64)
    top, _ = _roof_heights(points[:, 0], points[:, 1], ground, surface_at)
    if not np.isfinite(top).any():
        return None
    # A vertex with no reading of its own takes its nearest neighbour's.
    missing = ~np.isfinite(top)
    if missing.any():
        have = np.flatnonzero(~missing)
        tree = shapely.STRtree(shapely.points(points[have]))
        nearest = tree.query_nearest(shapely.points(points[missing]), all_matches=False)[1]
        top[missing] = top[have[nearest]]
    n = len(points)
    vertices = np.vstack([
        np.column_stack([points, top - style.height_m]),
        np.column_stack([points, top]),
    ])
    triangles = []
    for corners in loops:
        corners = corners if _signed_area(points[corners]) >= 0 else corners[::-1]
        lid = [c + n for c in corners]
        floor = corners[::-1]
        for face in (lid, floor):
            if len(face) == 4:
                triangles.extend([[face[0], face[1], face[2]], [face[0], face[2], face[3]]])
            else:
                for k in range(1, len(face) - 1):
                    triangles.append([face[0], face[k], face[k + 1]])
    # The edge, all round: the outline's own points, which the fill kept.
    for ring in _oriented([np.asarray(r, dtype=np.float64) for r in rings]):
        ids = [index.get(_key(p)) for p in ring]
        if any(i is None for i in ids):
            return None
        for a, b in zip(ids, ids[1:] + ids[:1]):
            triangles.extend([[a, b, n + b], [a, n + b, n + a]])
    local = np.vstack([points, points]) - points.min(axis=0)
    return vertices, np.asarray(triangles, dtype=np.int64), local


def _without_repeats(ring: np.ndarray, within: float = 1e-3) -> np.ndarray:
    """A closed ring without consecutive vertices closer than `within`."""
    if len(ring) < 2:
        return ring
    step = np.linalg.norm(ring - np.roll(ring, 1, axis=0), axis=1)
    keep = step >= within
    if not keep.any():
        return ring[:1]
    return ring[keep]


def _footprint_lookup(buildings):
    """`f(polygon)`: the buildings' footprints under a polygon, as one
    geometry, or None for no buildings. Built from each building's roof
    triangles seen from above."""
    import shapely

    if buildings is None or not len(getattr(buildings, "buildings", [])):
        return None
    shapes = []
    for building in buildings.buildings:
        roof = np.asarray(getattr(building, "roof_tris", []), dtype=np.float64)
        if roof.ndim != 3 or not len(roof):
            continue
        triangles = shapely.polygons(roof[:, :, :2])
        triangles = triangles[shapely.area(triangles) > 1e-6]
        if len(triangles):
            shapes.append(shapely.union_all(triangles))
    if not shapes:
        return None
    shapes = np.array(shapes, dtype=object)
    tree = shapely.STRtree(shapes)

    def under(polygon):
        hits = tree.query(polygon, predicate="intersects")
        if not len(hits):
            return None
        return shapely.union_all(shapes[hits]).buffer(0.05)

    return under


# A canopy left smaller than this once the buildings are cut out of it is a
# sliver along a facade, not a roof.
CANOPY_MIN_M2 = 4.0


def _off_buildings(rings, footprints_under):
    """A canopy's outline with the buildings under it cut out, as rings per
    piece. The roof the survey sees there is the building's."""
    import shapely

    rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in rings]
    polygon = shapely.make_valid(shapely.Polygon(rings[0], rings[1:]))
    covered = footprints_under(polygon)
    if covered is None or covered.is_empty:
        return [rings]
    left = shapely.difference(polygon, covered)
    pieces = []
    for part in shapely.get_parts(left):
        if part.geom_type != "Polygon" or part.area < CANOPY_MIN_M2:
            continue
        part = part.simplify(0.02)
        if part.is_empty or part.geom_type != "Polygon":
            continue
        pieces.append([np.asarray(part.exterior.coords)[:, :2]]
                      + [np.asarray(r.coords)[:, :2] for r in part.interiors])
    return pieces


def _signed_area(ring: np.ndarray) -> float:
    """Positive for an anticlockwise ring, seen from above."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _oriented(rings: list[np.ndarray]) -> list[np.ndarray]:
    """The outer ring anticlockwise and the holes clockwise.

    The sides are built facing out of an anticlockwise ring. The survey does
    not promise an order, and a wall it drew the other way round came out
    inside out: its sides faced in, which is a black wall in a lit viewport
    and an invisible one with backface culling on.
    """
    out = []
    for index, ring in enumerate(rings):
        anticlockwise = _signed_area(ring) > 0.0
        want = index == 0
        out.append(ring if anticlockwise == want else ring[::-1].copy())
    return out


def _ring_drape(ring: np.ndarray, sampler, step_m: float, tolerance_m: float) -> np.ndarray:
    """A closed ring, with a vertex added only where the ground bends.

    The swept lines' rule (`_drape_points`) on a ring: every survey corner
    stays, and a cut between them survives only where leaving it out would
    let the base leave the ground by more than the tolerance. Cutting every
    edge at a fixed step instead put a vertex every metre along every wall
    and awning, holding a straight line straight.
    """
    closed = np.vstack([ring, ring[:1]])
    dense, _height = _drape_points(closed, sampler, step_m, tolerance_m)
    out = dense[:-1]
    return out if len(out) >= len(ring) else ring


# A polygon is capped as a ladder when it is a strip: this many times longer
# than it is wide. Below that it is a block -- a transformer, a stoop -- and
# the triangulation of a handful of corners is already clean.
STRIP_ASPECT = 2.5


def _strip_cap(ring: np.ndarray):
    """The top of a strip-shaped outline as a ladder of quads, or None.

    Earcut fans a strip: one corner of a wall joined to every vertex along the
    far side, a spray of slivers on top of every wall the drape gave more
    than four corners. A strip has two long sides; walked from one end to the
    other together, each step is one rung across, and two rungs and the sides
    between them are a quad. Where one side has a vertex the other has not,
    the other gets one opposite it, so the rungs meet both sides square and
    the cap is quads all the way along, with a triangle fan only at a rounded
    end.

    Returns the ring with those vertices added (the sides are built from it,
    so nothing is left hanging off an edge) and the cap faces as index lists
    into it, or None for an outline that is not a strip.
    """
    import shapely

    n = len(ring)
    if n < 4:
        return None
    box = shapely.minimum_rotated_rectangle(shapely.Polygon(ring))
    corners = np.asarray(box.exterior.coords)[:4] if box.geom_type == "Polygon" else None
    if corners is None:
        return None
    sides = np.diff(np.vstack([corners, corners[:1]]), axis=0)
    lengths = np.hypot(sides[:, 0], sides[:, 1])
    long_side = int(np.argmax(lengths[:2]))
    length, width = float(lengths[long_side]), float(lengths[1 - long_side])
    if width <= 1e-6 or length / width < STRIP_ASPECT:
        return None
    axis = sides[long_side] / length
    along = ring @ axis
    reach = 0.75 * width

    def end_run(mask):
        """The one cyclic run of vertices in `mask`, as (first, last)."""
        if mask.all() or not mask.any():
            return None
        starts = np.flatnonzero(mask & ~np.roll(mask, 1))
        if len(starts) != 1:
            return None
        first = int(starts[0])
        last = first
        while mask[(last + 1) % n]:
            last = (last + 1) % n
        return first, last

    head = end_run(along <= along.min() + reach)
    tail = end_run(along >= along.max() - reach)
    if head is None or tail is None:
        return None

    def walk(start, stop):
        out = [start]
        while out[-1] != stop:
            out.append((out[-1] + 1) % n)
        return out

    # One long side from the head to the tail, the other back again; turned
    # round, both run head to tail. A pointed end is one vertex both share.
    side_a = walk(head[1], tail[0])
    side_b = walk(tail[1], head[0])[::-1]
    shared = set(side_a) & set(side_b)
    allowed = {end[0] for end in (head, tail) if end[0] == end[1]}
    if len(side_a) < 2 or len(side_b) < 2 or shared - allowed:
        return None

    def fractions(indices):
        points = ring[indices]
        steps = np.hypot(*np.diff(points, axis=0).T)
        total = float(steps.sum())
        if total <= 1e-9:
            return None
        return np.concatenate([[0.0], np.cumsum(steps)]) / total

    t_a, t_b = fractions(side_a), fractions(side_b)
    if t_a is None or t_b is None:
        return None
    # Two vertices this close along the strip share a rung rather than each
    # getting a partner a few centimetres from the other.
    gap = max(0.2, 0.5 * width) / max(length, 1e-9)

    points = [p for p in ring]
    a = [side_a[0]]
    b = [side_b[0]]
    i = j = 0
    added_a: dict[int, list[int]] = {}
    added_b: dict[int, list[int]] = {}

    # Where each side's last rung landed, as a fraction of the strip.
    reached = {"a": 0.0, "b": 0.0}

    def partner(name, rungs, side, t_side, k, t, added):
        """The vertex opposite a station at fraction t, on edge k..k+1 of the
        other side: that side's last rung end if it is close, else a new one."""
        if t - reached[name] <= gap:
            return rungs[-1]
        span = t_side[k + 1] - t_side[k]
        f = float(np.clip((t - t_side[k]) / span, 0.0, 1.0)) if span > 1e-12 else 0.0
        start, end = ring[side[k]], ring[side[k + 1]]
        points.append(start + (end - start) * f)
        added.setdefault(k, []).append(len(points) - 1)
        reached[name] = t
        return len(points) - 1

    last_a, last_b = len(side_a) - 1, len(side_b) - 1
    while i < last_a or j < last_b:
        next_a = t_a[i + 1] if i < last_a else 2.0
        next_b = t_b[j + 1] if j < last_b else 2.0
        if abs(next_a - next_b) <= gap or next_a > 1.0 or next_b > 1.0:
            i = min(i + 1, last_a)
            j = min(j + 1, last_b)
            a.append(side_a[i])
            b.append(side_b[j])
            reached["a"], reached["b"] = t_a[i], t_b[j]
        elif next_a < next_b:
            i += 1
            a.append(side_a[i])
            reached["a"] = next_a
            b.append(partner("b", b, side_b, t_b, j, next_a, added_b))
        else:
            j += 1
            b.append(side_b[j])
            reached["b"] = next_b
            a.append(partner("a", a, side_a, t_a, i, next_b, added_a))

    # The ring again, with the new vertices in their places along it. Side B
    # was walked backwards, so its additions go in backwards.
    position = {}
    new_ring = []
    for index in range(n):
        new_ring.append(index)
        k_a = side_a.index(index) if index in side_a[:-1] else None
        if k_a is not None:
            new_ring.extend(added_a.get(k_a, []))
        if index in side_b[1:]:
            k_b = side_b.index(index) - 1
            new_ring.extend(reversed(added_b.get(k_b, [])))
    for slot, index in enumerate(new_ring):
        position[index] = slot

    faces = []
    for k in range(len(a) - 1):
        quad = [a[k], a[k + 1], b[k + 1], b[k]]
        unique = list(dict.fromkeys(quad))
        if len(unique) >= 3:
            faces.append([position[v] for v in unique])
    for first, last in (head, tail):
        run = walk(first, last)
        if len(run) >= 3:
            for k in range(1, len(run) - 1):
                faces.append([position[run[0]], position[run[k]], position[run[k + 1]]])
    out = np.asarray([points[v] for v in new_ring], dtype=np.float64)
    # Built the right way, every face runs anticlockwise and together they
    # cover the outline exactly once. A short or crooked outline can fold
    # the ladder over its own end -- three faces on one edge, on three small
    # walls over Utrecht Centraal -- and is capped by earcut instead.
    total = 0.0
    for face in faces:
        for k in range(1, len(face) - 1):
            area = _signed_area(out[[face[0], face[k], face[k + 1]]])
            if area <= 1e-9:
                return None
            total += area
    if abs(total - _signed_area(out)) > 1e-6 * max(1.0, abs(_signed_area(out))):
        return None
    return out, faces


def _cdt_cap(rings, flat):
    """A top that is not a strip, as the ground fills a face: a constrained
    Delaunay triangulation, paired into quads where they make good ones.

    Earcut clips ears off the outline in order, which on an awning drawn
    with a curve in it is a fan of slivers from one corner -- 1,773 over
    Hoek van Holland. Returns index faces into `flat`, or None where the
    outline will not triangulate cleanly, for earcut to do.
    """
    import shapely

    from .terrain_mesh import _fill, _key

    try:
        polygon = shapely.Polygon(rings[0], rings[1:])
        if not polygon.is_valid:
            return None
        faces = _fill(polygon)
    except Exception:  # noqa: BLE001 - earcut is the fallback
        return None
    if not faces:
        return None
    index = {_key(p): i for i, p in enumerate(flat)}
    out = []
    for face in faces:
        ids = [index.get(_key(p)) for p in np.asarray(face)[:, :2]]
        if any(i is None for i in ids) or len(set(ids)) != len(ids):
            return None
        out.append(ids)
    # Every face up, and together exactly the outline: a face folded over
    # its neighbour cancels the overlap in the total, so both are needed.
    areas = [_signed_area(flat[f]) for f in out]
    if min(areas) <= 1e-12:
        return None
    if abs(sum(areas) - polygon.area) > 1e-6 * max(1.0, polygon.area):
        return None
    # And no edge used twice the same way round.
    seen = set()
    for f in out:
        for a, b in zip(f, f[1:] + f[:1]):
            if (a, b) in seen:
                return None
            seen.add((a, b))
    return out


def _up(points: np.ndarray, face: list[int]) -> list[int]:
    """A cap face wound to face up."""
    return face if _signed_area(points[face]) >= 0.0 else face[::-1]


def _ring_steps(ring: np.ndarray, step_m: float) -> np.ndarray:
    """Insert vertices along a closed ring so no edge is longer than `step_m`.

    The survey's own vertices all survive: they are the shape of the wall,
    where the inserted ones are only there to let the ground be followed.
    """
    closed = np.vstack([ring, ring[:1]])
    out = [closed[:1]]
    for start, end in zip(closed[:-1], closed[1:]):
        span = float(np.hypot(*(end - start)))
        cuts = max(1, int(np.ceil(span / step_m)))
        t = np.arange(1, cuts + 1) / cuts
        out.append(start + (end - start) * t[:, None])
    return np.vstack(out)[:-1]


def _prepare_rings(group, bbox: BBox) -> list[np.ndarray]:
    """One BGT polygon into rings the extruder can use, cut to the area.

    Cut, not merely centre-tested. This used to keep any polygon whose middle
    was inside, on the grounds that a wall poking a metre past the edge was not
    worth a seam -- and then a quay wall followed its canal 227 m past the bbox
    and took the model's bounding box with it. A wall is not a building:
    cutting one at the edge shows a cut wall, which is what the edge of the
    model looks like anyway.
    """
    rings = []
    for ring in group:
        ring = np.asarray(ring, dtype=np.float64)
        if ring.ndim != 2 or len(ring) < 3:
            continue
        clipped = clip_ring_to_bbox(ring[:, :2], bbox)
        if len(clipped) >= 3:
            rings.append(clipped)
    return rings


def _span(base_z, style: BarrierStyle):
    """The bottom and top of a barrier, given where the ground is."""
    if style.lift_m > 0.0:
        return base_z + style.lift_m, base_z + style.lift_m + style.height_m
    return base_z - EMBED_M, base_z + style.height_m


@dataclass(frozen=True)
class BarrierStyle:
    """What one BGT type is drawn as."""

    name: str
    height_m: float
    thickness_m: float
    material: str
    # A fence is mostly holes. Drawn as a solid slab it reads as a hoarding,
    # and a railing and a hedge come out the same object. See-through styles
    # get their own alpha-clipped material and a texture that tiles along the
    # run, so the uprights stay the same width whatever the fence's length.
    see_through: bool = False
    # Awnings and canopies hang off a facade. Drawn from the ground they are a
    # solid block against the wall, which is worse than leaving them out.
    lift_m: float = 0.0
    # Read the top off the surface model where it can see one (`_measured_top`),
    # and fall back on `lift_m` where it cannot. A canopy is anything from a
    # bicycle shelter to the roof over a station's platforms, and no one lift
    # fits both.
    top_from_survey: bool = False


# Linear types, by collection and plus_type.
#
# A hedge is thicker than a fence and shorter than a wall, and it is the one
# entry here whose height is nearly reliable -- a Dutch hedge is clipped to
# about the height of the fence it hides.
LINE_STYLES: dict[tuple[str, str], BarrierStyle] = {
    ("scheiding_lijn", "hek"): BarrierStyle(
        "fence", 1.8, 0.06, "paint", see_through=True
    ),
    ("scheiding_lijn", "muur"): BarrierStyle("wall", 2.0, 0.24, "brick"),
    ("scheiding_lijn", "kademuur"): BarrierStyle("quay_wall", 1.1, 0.40, "concrete"),
    ("scheiding_lijn", "walbescherming"): BarrierStyle(
        "bank_protection", 0.5, 0.30, "concrete"
    ),
    ("scheiding_lijn", "geluidsscherm"): BarrierStyle(
        "noise_barrier", 3.0, 0.16, "concrete"
    ),
    ("scheiding_lijn", "damwand"): BarrierStyle("sheet_pile", 0.8, 0.20, "concrete"),
    ("vegetatieobject_lijn", "haag"): BarrierStyle("hedge", 1.5, 0.70, "hedge"),
}

# Areal types. A wall the BGT drew as a polygon is wider than half a metre, so
# its own outline gives the thickness and only the height is assumed.
AREA_STYLES: dict[tuple[str, str], BarrierStyle] = {
    ("scheiding_vlak", "muur"): BarrierStyle("wall", 2.0, 0.0, "brick"),
    ("scheiding_vlak", "kademuur"): BarrierStyle("quay_wall", 1.1, 0.0, "concrete"),
    ("gebouwinstallatie", "toegangstrap"): BarrierStyle("steps", 0.9, 0.0, "stone"),
    ("gebouwinstallatie", "bordes"): BarrierStyle("stoop", 0.4, 0.0, "stone"),
    ("gebouwinstallatie", "luifel"): BarrierStyle(
        "awning", 0.12, 0.0, "metal", lift_m=2.4
    ),
    ("overigbouwwerk", "overkapping"): BarrierStyle(
        "canopy", 0.14, 0.0, "metal", lift_m=2.6, top_from_survey=True
    ),
    ("overigbouwwerk", "lage trafo"): BarrierStyle("transformer", 2.4, 0.0, "brick"),
    ("overigbouwwerk", "open loods"): BarrierStyle("shed", 2.8, 0.0, "metal"),
    ("overigbouwwerk", "bunker"): BarrierStyle("bunker", 2.5, 0.0, "concrete"),
    ("overigbouwwerk", "opslagtank"): BarrierStyle("tank", 4.0, 0.0, "metal"),
}

COLLECTIONS = sorted(
    {collection for collection, _ in (*LINE_STYLES, *AREA_STYLES)}
)


# How thin a collider is allowed to be, for physics rather than for looks.
#
# A fence is drawn 6 cm thick because that is roughly what a fence is. Unity's
# default physics step is 50 Hz with discrete collision detection, so a body
# moving at 20 m/s travels 40 cm between steps: if it starts one side of a 6 cm
# slab and ends the other, no contact is ever generated and the drone flies
# through the fence. 30 cm covers 15 m/s, which is most survey flying, and the
# collider is invisible so the fence still looks like a fence.
#
# This is a mitigation and not the cure. Faster than that and the fix is
# Continuous Dynamic collision detection on the drone's rigidbody, which costs
# CPU but cannot tunnel at any speed. Both are in the README.
DEFAULT_COLLISION_THICKNESS_M = 0.30


def collision_style(style: BarrierStyle, min_thickness_m: float) -> BarrierStyle:
    """The same barrier, fattened to the thinnest a collider may be.

    Which dimension is the thin one depends on how the barrier was drawn, and
    they are both handled because a drone can hit either:

    * A **line** barrier is a prism swept along a run, so `thickness_m` is its
      width and a fence's 6 cm is the thin direction.
    * An **area** barrier takes its width from the surveyed outline -- the BGT
      only draws a wall as a polygon when it is wider than half a metre, so
      there is nothing to fix there -- and `height_m` is the thin direction.
      That matters for exactly one family: an awning or a canopy is a 12 cm
      plate hanging 2.4 m up, and a drone climbing a facade goes through it.

    Widening an area barrier's outline would be a polygon offset, which this
    does not attempt: nothing in AREA_STYLES is thin horizontally.
    """
    if style.thickness_m > 0.0:
        return replace(
            style, thickness_m=max(style.thickness_m, float(min_thickness_m))
        )
    return replace(style, height_m=max(style.height_m, float(min_thickness_m)))


def apply_height_override(
    style: BarrierStyle | None, heights: dict
) -> BarrierStyle | None:
    """`barriers.heights` changes a height, and deliberately nothing else.

    `replace`, rather than building a fresh BarrierStyle: this was written
    positionally, and adding `see_through` to the middle of the dataclass then
    shifted every later field along by one. An overridden fence came out solid,
    and an overridden awning lost its 2.4 m lift *and* inherited the fence's
    cutout material, so it sat on the pavement as an alpha-clipped slab against
    the wall. Nothing failed, because the knob defaults to empty.

    It is a named function rather than a closure in `build_barriers` so that a
    test can reach the real thing. Testing a local copy of this logic is how it
    went unnoticed.
    """
    if style is None:
        return None
    override = heights.get(style.name)
    if override is None:
        return style
    return replace(style, height_m=float(override))


@dataclass
class BarrierSet:
    """Triangles with a material name each, ready for the Blender stage."""

    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # Metres along the run and up from the base. The Blender stage divides by
    # the tile size, so a fence keeps the same upright spacing at any length.
    uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))
    # One per triangle: which atlas patch it wears, and which BGT type it is.
    material: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    kind: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    material_names: list[str] = field(default_factory=list)
    kind_names: list[str] = field(default_factory=list)
    # Which kinds need the alpha-clipped fence material rather than the atlas.
    see_through: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    # The same barriers, fattened to the thinnest a collider may be, and empty
    # when nothing needed it. No UVs and no materials: this geometry is never
    # drawn. See `collision_style`.
    collision_vertices: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3))
    )
    collision_triangles: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), np.int32)
    )
    collision_thickness_m: float = 0.0

    def __len__(self) -> int:
        return int(len(self.triangles))

    def stats(self) -> dict:
        out = {
            "triangles": len(self),
            "vertices": int(len(self.vertices)),
            **self.counts,
        }
        if len(self.collision_triangles):
            out["collision_triangles"] = int(len(self.collision_triangles))
            out["collision_thickness_m"] = round(self.collision_thickness_m, 3)
        return out


def _sweep_line(points: np.ndarray, sampler, style: BarrierStyle):
    """One polyline into a closed prism, as ``(vertices, triangles)``.

    The base follows the terrain and the top follows the base, so a wall up a
    slope stays the same height all the way along rather than levelling off or
    burying itself.

    The ground is sampled at each rail, not once along the centreline. A wall
    has width, and the mitre at a corner widens it further -- up to two and a
    half times the half-thickness, which on a half-metre quay wall is half a
    metre sideways. That is nothing on a pavement and a lot on a canal bank,
    which is exactly where quay walls are: sampling the middle and building the
    corners from it put a wall 0.96 m underground.
    """
    if len(points) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    normals = offset_normals(points)
    half = max(style.thickness_m, 0.02) * 0.5
    left = points + normals * half
    right = points - normals * half

    ground_left = np.asarray(sampler(left[:, 0], left[:, 1]), dtype=np.float64)
    ground_right = np.asarray(sampler(right[:, 0], right[:, 1]), dtype=np.float64)
    # Each base on its own ground, so neither side floats and neither is
    # swallowed; the top level across the width, taken from the higher side, so
    # the wall does not lean and is never shorter than it should be. Sharing
    # one height for both would have to pick, and either choice is wrong on a
    # bank: the lower buries the high side, the higher floats the low one.
    low_left, _ = _span(ground_left, style)
    low_right, _ = _span(ground_right, style)
    _, high = _span(np.maximum(ground_left, ground_right), style)

    n = len(points)
    # Four rails: left-low, left-high, right-high, right-low. Ordered around
    # the cross-section, so consecutive rails bound one face of the prism and
    # the last wraps to the first.
    rails = [
        np.column_stack([left, low_left]),
        np.column_stack([left, high]),
        np.column_stack([right, high]),
        np.column_stack([right, low_right]),
    ]
    vertices = np.vstack(rails)

    # U in metres along the run, V normalised over the height.
    #
    # The two are not the same thing on purpose. Along the run the texture has
    # to tile, so that a four-hundred-metre fence gets four hundred metres of
    # uprights rather than eight stretched across it -- and the Blender stage
    # divides U by the tile size to get there. Up the height it must not tile:
    # one fence is one texture, top rail to bottom rail, or the top rail lands
    # wherever the height happens to fall and most fences simply have no top.
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    top = np.ones(n)
    bottom = np.zeros(n)
    uvs = np.vstack([
        np.column_stack([along, bottom]),
        np.column_stack([along, top]),
        np.column_stack([along, top]),
        np.column_stack([along, bottom]),
    ])

    triangles = []
    for r in range(4):
        a0 = r * n
        b0 = ((r + 1) % 4) * n
        for i in range(n - 1):
            triangles.append([a0 + i, b0 + i, b0 + i + 1])
            triangles.append([a0 + i, b0 + i + 1, a0 + i + 1])

    # Close both ends, or the prism is a tube and reads as hollow from any
    # position that can see into it. Facing out: the start looks back down
    # the run, the end along it. These were the other way round, so every
    # wall's ends faced into the wall -- black in a lit viewport, see-through
    # with backface culling. Each as the pair (a, b, c), (a, c, d), which the
    # Blender stage joins back into one quad.
    for i, quad in (
        (0, [0, 3, 2, 1]),
        (n - 1, [0, 1, 2, 3]),
    ):
        q = [r * n + i for r in quad]
        triangles.append([q[0], q[1], q[2]])
        triangles.append([q[0], q[2], q[3]])

    return vertices, np.asarray(triangles, dtype=np.int64), uvs


def _extrude_rings(
    rings: list[np.ndarray],
    sampler,
    style: BarrierStyle,
    step_m: float = 1.0,
    *,
    tolerance_m: float = 0.05,
    top_z: float | None = None,
):
    """A polygon into a prism, as ``(vertices, triangles, uvs)``.

    The outer ring and any holes all become vertical quads. The top of a
    strip -- most walls -- is a ladder of quads across it (`_strip_cap`);
    anything else is capped by earcut. Every quad is written as the pair of
    triangles (a, b, c), (a, c, d), which the Blender stage joins back up.

    `top_z` puts the top at a measured height -- a canopy read off the
    surface model -- rather than at the style's height above the ground.

    The base is sampled per vertex, not once for the whole outline. One height
    for the whole thing is the obvious shortcut and it fails exactly where
    these objects live: a quay wall runs along a canal bank, which is the
    steepest ground in a Dutch city. On a bumpy test surface a single median
    buried one end of a wall by 0.65 m.

    A lifted piece -- an awning, a canopy -- is the exception. It hangs off a
    facade at one height, so it takes the median and stays flat; an awning
    that ripples with the pavement under it is worse than one that does not.
    """
    import mapbox_earcut

    rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in rings]
    rings = [r[:-1] if np.allclose(r[0], r[-1], rtol=0.0, atol=1e-6) else r for r in rings]
    # A survey vertex drawn twice in a row is an edge of no length: a side
    # with no width, and on top a triangle the triangulation made twice.
    rings = [_without_repeats(r) for r in rings]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2))
    rings = _oriented(rings)
    flat_piece = style.lift_m > 0.0 or top_z is not None

    # Follow the ground along the ring. A survey draws a thirty-metre wall as
    # a four-vertex rectangle, and sampling the ground only at the corners
    # leaves the base a straight chord over whatever the ground does in
    # between: measured over the Gelderland square, one ring edge in five
    # departed from its chord by more than the fifteen-centimetre embed, and
    # the worst by 3.9 m. Walls floated, steps buried themselves.
    #
    # Only where it bends, though (`_ring_drape`), and not at all for a piece
    # that is flat anyway: an awning cut every metre was a hundred triangles
    # holding a flat plate flat, 26,580 of them over Hoek van Holland.
    if step_m > 0 and not flat_piece:
        rings = [_ring_drape(ring, sampler, step_m, tolerance_m) for ring in rings]

    cap_faces = None
    if len(rings) == 1:
        strip = _strip_cap(rings[0])
        if strip is not None:
            rings[0], cap_faces = strip

    flat = np.vstack(rings)
    ground = np.asarray(sampler(flat[:, 0], flat[:, 1]), dtype=np.float64)
    if top_z is not None:
        low = np.full(len(flat), float(top_z) - style.height_m)
        high = np.full(len(flat), float(top_z))
    else:
        if style.lift_m > 0.0:
            # Flat, and clear of the ground at its worst corner. The median
            # looks like the fair choice and is not: an awning whose footprint
            # straddles a slope ends up half buried, by 1.43 m on the
            # Gelderland square. Too high at one end is a thing nobody
            # notices; underground is not.
            ground = np.full(len(flat), float(ground.max()))
        low, high = _span(ground, style)

    n = len(flat)
    vertices = np.vstack([
        np.column_stack([flat, low]),
        np.column_stack([flat, high]),
    ])

    triangles = []
    start = 0
    for ring in rings:
        count = len(ring)
        for i in range(count):
            a = start + i
            b = start + (i + 1) % count
            triangles.append([a, b, n + b])
            triangles.append([a, n + b, n + a])
        start += count

    if cap_faces is None:
        cap_faces = _cdt_cap(rings, flat)
    if cap_faces is None:
        ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        try:
            found = np.asarray(
                mapbox_earcut.triangulate_float64(flat, ends), dtype=np.int64
            ).reshape(-1, 3)
        except Exception as exc:  # noqa: BLE001 - one bad outline is not fatal
            LOG.debug("could not cap a barrier outline: %s", exc)
            found = np.zeros((0, 3), dtype=np.int64)
        cap_faces = found.tolist()
        # Earcut winds every triangle the same way; turned up together, by
        # their total, not one by one -- a near-flat sliver's own sign is
        # rounding, and turned on its own it faced the other way from its
        # neighbours.
        if cap_faces and sum(_signed_area(flat[f]) for f in cap_faces) < 0.0:
            cap_faces = [f[::-1] for f in cap_faces]
    else:
        cap_faces = [_up(flat, list(face)) for face in cap_faces]

    def split(face, lift):
        """A face as triangles, a quad as the pair (a, b, c), (a, c, d)."""
        face = [v + lift for v in face]
        if len(face) == 4:
            return [[face[0], face[1], face[2]], [face[0], face[2], face[3]]]
        return [face]

    for face in cap_faces:
        triangles.extend(split(face, n))
    # An awning has no ground under it, so it needs a floor as well as a lid.
    # A wall's underside is buried and nobody will look.
    if flat_piece:
        for face in cap_faces:
            triangles.extend(split(face[::-1], 0))

    # Metres from the outline's own corner rather than RD eastings, which
    # would be six-figure UVs for anything that ever reads them. The solid
    # styles take a flat atlas patch instead, so these only matter if one of
    # them is ever given a tiling texture of its own.
    local = flat - flat.min(axis=0)
    uvs = np.vstack([local, local])

    return vertices, np.asarray(triangles, dtype=np.int64), uvs


def build_barriers(
    bbox: BBox,
    work_dir: Path,
    *,
    barriers_cfg: dict,
    terrain,
    collision: bool = False,
    skip_styles=(),
    deck_at=None,
    surface_at=None,
    buildings=None,
) -> BarrierSet:
    """Fetch every barrier collection and turn it into draped geometry.

    `surface_at(x, y)` is the surface model (AHN DSM), NaN where it has no
    reading. A canopy's top is read from it (`_measured_top`).

    `skip_styles` names styles left to someone else: the layered style builds
    quay walls as the face of the quay (`harbour`), not a wall standing on it.

    `deck_at(x, y)` is the top of a bridge deck there, NaN off one
    (`structures.deck_surface`). A barrier the BGT puts above the ground
    (``relatieve_hoogteligging`` 1 or more) stands on it: the parapet of a
    canal bridge, stood on the ground under it, was a wall from the canal bed
    to above the deck, closing the canal under every bridge in Amsterdam.

    `collision` adds a second, fattened prism per thin run, for a simulation
    that needs a fence something cannot tunnel through. It costs another sweep
    and another copy in the npz, so it is off unless `export.collision` asked
    for it.
    """
    result = BarrierSet()
    if not bool(barriers_cfg.get("enabled", True)):
        LOG.info("barriers disabled")
        return result

    fetch_kwargs = {
        "page_limit": int(barriers_cfg["page_limit"]),
        "timeout": float(barriers_cfg["timeout_s"]),
        "max_retries": int(barriers_cfg["max_retries"]),
        "max_pages": int(barriers_cfg["max_pages"]),
    }
    step_m = float(barriers_cfg.get("step_m", DEFAULT_STEP_M))
    drape_tolerance_m = float(barriers_cfg.get("drape_tolerance_m", 0.05))
    heights = barriers_cfg.get("heights") or {}
    collision_m = float(
        barriers_cfg.get("collision_thickness_m", DEFAULT_COLLISION_THICKNESS_M)
    )

    def collider_for(style):
        """The fattened style to sweep as well, or None for neither."""
        if not collision:
            return None
        fat = collision_style(style, collision_m)
        return fat if fat != style else None

    skipped = set(skip_styles)
    footprints_under = _footprint_lookup(buildings)

    def on_deck(x, y):
        """The deck where there is one, the ground elsewhere."""
        below = np.asarray(terrain.surface(x, y), dtype=np.float64)
        deck = np.asarray(deck_at(x, y), dtype=np.float64).reshape(below.shape)
        return np.where(np.isfinite(deck), np.fmax(deck, below), below)

    def styled(table, collection, plus_type):
        style = apply_height_override(table.get((collection, plus_type)), heights)
        return None if style is not None and style.name in skipped else style

    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, BarrierStyle]] = []
    collision_chunks: list[tuple[np.ndarray, np.ndarray]] = []
    counts: dict[str, int] = {}

    for collection in COLLECTIONS:
        try:
            features, _stats = fetch_current(
                collection, bbox, class_field="plus_type", **fetch_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - one empty collection is not fatal
            LOG.warning("could not read BGT %s: %s", collection, exc)
            continue

        for feature in features:
            properties = feature.get("properties") or {}
            plus_type = feature_type(properties)
            geometry = feature.get("geometry") or {}
            geometry_type = str(geometry.get("type") or "")
            try:
                level = int(properties.get("relatieve_hoogteligging") or 0)
            except (TypeError, ValueError):
                level = 0
            if level < 0:
                # In a tunnel: the Maastunnel's dividing wall, stood on the
                # ground, ran up over the headwall and across the park on top.
                counts["below_ground_left_out"] = counts.get("below_ground_left_out", 0) + 1
                continue
            raised = level > 0
            ground = on_deck if raised and deck_at is not None else terrain.surface

            if geometry_type in ("LineString", "MultiLineString"):
                style = styled(LINE_STYLES, collection, plus_type)
                if style is None:
                    continue
                raw = (
                    [geometry.get("coordinates") or []]
                    if geometry_type == "LineString"
                    else (geometry.get("coordinates") or [])
                )
                for line in raw:
                    line = np.asarray(line, dtype=np.float64)
                    if line.ndim != 2 or len(line) < 2:
                        continue
                    for piece in clip_line_to_bbox(line[:, :2], bbox):
                        if _length(piece) < MIN_LENGTH_M:
                            continue
                        dense, _profile = _drape_points(
                            piece, ground, step_m, drape_tolerance_m
                        )
                        verts, tris, uvs = _sweep_line(
                            dense, ground, style
                        )
                        if len(tris):
                            chunks.append(
                                (verts, tris, uvs, style)
                            )
                            counts[style.name] = counts.get(style.name, 0) + 1
                            fat = collider_for(style)
                            if fat is not None:
                                c_verts, c_tris, _ = _sweep_line(
                                    dense, ground, fat
                                )
                                if len(c_tris):
                                    collision_chunks.append((c_verts, c_tris))

            elif geometry_type in ("Polygon", "MultiPolygon"):
                style = styled(AREA_STYLES, collection, plus_type)
                if style is None:
                    continue
                for group in polygon_rings(geometry):
                    rings = _prepare_rings(group, bbox)
                    if not rings:
                        continue
                    # A canopy over a building is the building's roof: cut the
                    # buildings out, or the two lie on one another at the same
                    # height and flicker (the hall at Utrecht Centraal).
                    pieces = (
                        _off_buildings(rings, footprints_under)
                        if style.top_from_survey and footprints_under is not None
                        else [rings]
                    )
                    for rings in pieces:
                        top = None
                        if style.top_from_survey and surface_at is not None:
                            open_rings = [
                                np.asarray(r, dtype=np.float64)[:, :2] for r in rings
                            ]
                            open_rings = [
                                r[:-1] if np.allclose(r[0], r[-1], rtol=0.0, atol=1e-6) else r
                                for r in open_rings
                            ]
                            draped = _draped_roof(_oriented(open_rings), ground, surface_at, style)
                            if draped is not None:
                                chunks.append((*draped, style))
                                counts[style.name] = counts.get(style.name, 0) + 1
                                key = f"{style.name}_draped"
                                counts[key] = counts.get(key, 0) + 1
                                fat = collider_for(style)
                                if fat is not None:
                                    fat_roof = _draped_roof(
                                        _oriented(open_rings), ground, surface_at, fat
                                    )
                                    if fat_roof is not None:
                                        collision_chunks.append(fat_roof[:2])
                                continue
                            top = _measured_top(rings, ground, surface_at)
                            if top is not None:
                                key = f"{style.name}_measured"
                                counts[key] = counts.get(key, 0) + 1
                        verts, tris, uvs = _extrude_rings(
                            rings, ground, style, step_m,
                            tolerance_m=drape_tolerance_m, top_z=top,
                        )
                        if len(tris):
                            chunks.append((verts, tris, uvs, style))
                            counts[style.name] = counts.get(style.name, 0) + 1
                            fat = collider_for(style)
                            if fat is not None:
                                c_verts, c_tris, _ = _extrude_rings(
                                    rings, ground, fat, step_m,
                                    tolerance_m=drape_tolerance_m, top_z=top,
                                )
                                if len(c_tris):
                                    collision_chunks.append((c_verts, c_tris))

    if not chunks:
        LOG.info("no barriers found")
        return result

    materials = sorted({style.material for _, _, _, style in chunks})
    kinds = sorted({style.name for _, _, _, style in chunks})
    material_index = {name: i for i, name in enumerate(materials)}
    kind_index = {name: i for i, name in enumerate(kinds)}

    all_vertices, all_triangles, all_uvs = [], [], []
    material_of, kind_of = [], []
    offset = 0
    for verts, tris, uvs, style in chunks:
        all_vertices.append(verts)
        all_uvs.append(uvs)
        all_triangles.append(tris + offset)
        material_of.append(
            np.full(len(tris), material_index[style.material], np.int32)
        )
        kind_of.append(np.full(len(tris), kind_index[style.name], np.int32))
        offset += len(verts)

    vertices = np.vstack(all_vertices)
    uv = np.vstack(all_uvs)
    triangles = np.vstack(all_triangles)
    material_of = np.concatenate(material_of)
    kind_of = np.concatenate(kind_of)

    # A survey outline can double a vertex, which makes a triangle with no
    # area: invisible, but it breaks normals in Blender and shows up in any
    # mesh check as a fault. Cheaper to drop here than to explain there.
    a = vertices[triangles[:, 0]]
    edge1 = vertices[triangles[:, 1]] - a
    edge2 = vertices[triangles[:, 2]] - a
    real = np.linalg.norm(np.cross(edge1, edge2), axis=1) > 1e-9
    dropped = int((~real).sum())
    if dropped:
        LOG.debug("dropped %d zero-area barrier triangles", dropped)

    result.vertices = vertices
    result.uv = uv
    result.triangles = triangles[real].astype(np.int32)
    result.material = material_of[real]
    result.kind = kind_of[real]
    result.material_names = materials
    result.kind_names = kinds
    result.see_through = sorted(
        {style.name for _, _, _, style in chunks if style.see_through}
    )
    result.counts = counts

    if collision_chunks:
        c_verts, c_tris, c_offset = [], [], 0
        for verts, tris in collision_chunks:
            c_verts.append(verts)
            c_tris.append(tris + c_offset)
            c_offset += len(verts)
        result.collision_vertices = np.vstack(c_verts)
        result.collision_triangles = np.vstack(c_tris).astype(np.int32)
        result.collision_thickness_m = collision_m

    LOG.info(
        "barriers: %s (%d triangles, %d vertices)",
        ", ".join(f"{v} {k}" for k, v in sorted(counts.items())),
        len(result.triangles),
        len(result.vertices),
    )
    if len(result.collision_triangles):
        LOG.info(
            "barrier colliders: %d triangles, nothing thinner than %.2f m",
            len(result.collision_triangles),
            collision_m,
        )
    save_barriers(result, work_dir / "barriers.npz")
    return result


def _length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def save_barriers(barriers: BarrierSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices=barriers.vertices,
        triangles=barriers.triangles,
        material=barriers.material,
        kind=barriers.kind,
        uv=barriers.uv,
        # Fixed-width strings, never object arrays. An object array is
        # pickled, and a pickle written by NumPy 2 names `numpy._core`, which
        # the NumPy 1.26 bundled with Blender 4.2 cannot import: the Blender
        # stage died on the first barrier with every other layer built.
        material_names=np.array(barriers.material_names, dtype=str),
        kind_names=np.array(barriers.kind_names, dtype=str),
        see_through=np.array(barriers.see_through, dtype=str),
        collision_vertices=barriers.collision_vertices,
        collision_triangles=barriers.collision_triangles,
        collision_thickness_m=np.float64(barriers.collision_thickness_m),
    )
    LOG.info("wrote %s (%d triangles)", path, len(barriers))
    return path


__all__ = [
    "AREA_STYLES",
    "COLLECTIONS",
    "DEFAULT_COLLISION_THICKNESS_M",
    "LINE_STYLES",
    "BarrierSet",
    "BarrierStyle",
    "apply_height_override",
    "build_barriers",
    "collision_style",
    "feature_type",
    "save_barriers",
]
