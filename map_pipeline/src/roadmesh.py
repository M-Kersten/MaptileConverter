"""Road surfaces with topology a game engine can use.

The road geometry was correct and unusable. Four separate things were wrong
with it, and they compound:

* **Nothing was simplified.** BGT surveys a kerb to the centimetre, so a
  straight street arrives carrying a vertex every few centimetres and a corner
  carrying dozens. Every one of them became geometry.
* **Earcut fans a strip into slivers.** Handed a road 8 m wide with vertices
  5 cm apart, it produces triangles 160 times longer than they are wide: a
  third of them came out under one degree.
* **Refinement left hanging nodes.** Splitting one triangle and not its
  neighbour leaves a T-junction, which is a crack and a shading seam in any
  engine that welds normals.
* **Nothing was indexed.** Every triangle carried its own three corners, so the
  vertex buffer was three times the size it needed to be with no vertex reuse
  at all.

This module fixes them in the order they matter. Rings are simplified as a
*network*, so a kerb two road parts share stays one line; the interior is
triangulated by the same constrained Delaunay code the terrain uses, so the
triangles are well shaped and the ring survives as edges; refinement inserts
points rather than cutting triangles, so the result stays conforming; and the
whole thing comes out indexed.

Refinement is where this first went badly wrong, and the way it went wrong is
worth keeping written down, because the mesh it produced was worse than no
refinement at all. Two things were true at once:

* Nothing stopped a candidate point being inserted right beside a kerb. The
  triangulator may not flip across a constraint, so such a point cannot
  improve the triangle there -- it only wedges a thinner sliver against the
  kerb.
* A circumcentre that landed outside the road was replaced by the triangle's
  centroid. A thin triangle is thin because it lies along a kerb, so its
  centroid is a hair off that kerb: this manufactured exactly the points the
  first item could not cope with.

Separately neither is fatal; together they are a feedback loop, each round
laying new slivers along the kerb for the next round to chase. Over one square
kilometre of Utrecht, refining a footpath network of 18,731 triangles:

    unrefined                        18,731 tris   median 17.9 deg   23.6% <10
    both faults (what shipped)      233,807 tris   median 22.6 deg   33.0% <10
    no encroachment split only       43,849 tris   median 26.6 deg   21.1% <10
    centroid fallback only           53,105 tris   median 35.1 deg    0.9% <10
    neither                          51,240 tris   median 35.1 deg    1.2% <10

So the fix that matters is the encroachment split: a candidate inside a kerb
segment's diametral circle is dropped and the kerb halved in its place, which
is Ruppert's rule and is what makes refinement converge. Dropping the centroid
fallback is then redundant -- the split catches those points anyway, as the
fourth row shows -- and it is gone regardless, because a rule that only ever
produced bad points is not worth keeping for the rounding.
"""

from __future__ import annotations

import logging

import numpy as np

from .cdt import WELD_M, weld

LOG = logging.getLogger(__name__)

# How far a simplified kerb may move. The BGT is surveyed to the centimetre and
# a road drawn to a quarter of a metre is still visibly the same road, while
# costing a fraction of the vertices.
DEFAULT_SIMPLIFY_M = 0.25


def _douglas_peucker_keep(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Which vertices of an open run to keep. Endpoints always survive."""
    n = len(points)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    keep[0] = keep[-1] = True
    if n < 3 or tolerance <= 0:
        keep[:] = True
        return keep
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        span = points[j] - points[i]
        length = float(np.hypot(span[0], span[1]))
        rel = points[i + 1 : j] - points[i]
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
    return keep


def simplify_ring_network(rings: list[np.ndarray], tolerance_m: float):
    """Thin a set of polygon rings without pulling their shared edges apart.

    Road parts are a planar partition like everything else in the BGT: a
    carriageway and the cycle path beside it carry the same kerb, vertex for
    vertex. Thinning them one ring at a time does not keep it that way, and the
    gap that opens between two copies of one kerb is a stripe of ground showing
    through the road.

    So the decision is made once, on the welded network. Junctions -- anywhere
    other than exactly two edges meet -- are kept unconditionally, because they
    are where the partition's shape lives. The runs between them are thinned by
    Douglas-Peucker, which is symmetric end to end, so both rings carrying a
    shared run are handed the same answer and stay welded to each other.

    Returns `(rings, source_index)`. The index says which input ring each
    survivor came from, because a ring can be dropped for no longer being a
    polygon and the caller has to be able to put the rest back in the groups
    they belong to. Matching them up by identity afterwards does not work --
    these are new arrays -- and doing it silently produced an empty road.
    """
    indexed = [
        (k, np.asarray(r, dtype=np.float64)[:, :2]) for k, r in enumerate(rings)
    ]
    indexed = [(k, r) for k, r in indexed if len(r) >= 3]
    if not indexed:
        return [], []
    source = [k for k, _ in indexed]
    usable = [r for _, r in indexed]
    if tolerance_m <= 0:
        return usable, source

    welded, mapping = weld(np.vstack(usable), WELD_M)

    # Degree over the whole network, counting each undirected edge once.
    neighbours: dict[int, set] = {}
    offset = 0
    ids_per_ring = []
    for ring in usable:
        ids = mapping[offset : offset + len(ring)]
        offset += len(ring)
        ids_per_ring.append(ids)
        for a, b in zip(ids[:-1], ids[1:]):
            if a == b:
                continue
            neighbours.setdefault(int(a), set()).add(int(b))
            neighbours.setdefault(int(b), set()).add(int(a))

    junction = {v for v, near in neighbours.items() if len(near) != 2}

    out, kept_source = [], []
    for source_index, ring, ids in zip(source, usable, ids_per_ring):
        anchors = [k for k, v in enumerate(ids) if int(v) in junction]
        # A ring with no junction on it at all -- an island of road with no
        # neighbour -- still needs two anchors to be thinned between.
        if len(anchors) < 2:
            anchors = [0, len(ring) - 1]
        if anchors[0] != 0:
            anchors.insert(0, 0)
        if anchors[-1] != len(ring) - 1:
            anchors.append(len(ring) - 1)

        keep = np.zeros(len(ring), dtype=bool)
        keep[anchors] = True
        for start, end in zip(anchors[:-1], anchors[1:]):
            run = ring[start : end + 1]
            keep[start : end + 1] |= _douglas_peucker_keep(run, tolerance_m)
        thinned = ring[keep]
        # Closed and still a polygon.
        if len(thinned) >= 3:
            if not np.allclose(thinned[0], thinned[-1]):
                thinned = np.vstack([thinned, thinned[:1]])
            if len(thinned) >= 4:
                out.append(thinned)
                kept_source.append(source_index)
    return out, kept_source


def _even_odd(rings: list[np.ndarray], points: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon against one ring group: outer minus holes."""
    inside = np.zeros(len(points), dtype=bool)
    px, py = points[:, 0], points[:, 1]
    for ring in rings:
        ax, ay = ring[:-1, 0], ring[:-1, 1]
        bx, by = ring[1:, 0], ring[1:, 1]
        for i in range(len(ax)):
            # A horizontal edge crosses no horizontal ray, so it is skipped
            # outright. Nudging the denominator instead divides by 1e-300 and
            # overflows to infinity, which is a comparison against nonsense.
            if by[i] == ay[i]:
                continue
            straddles = (ay[i] > py) != (by[i] > py)
            crosses = straddles & (
                px < (bx[i] - ax[i]) * (py - ay[i]) / (by[i] - ay[i]) + ax[i]
            )
            inside ^= crosses
    return inside


def _inside(groups: list[list[np.ndarray]], points: np.ndarray) -> np.ndarray:
    """Which points are road, over a whole set of ring groups.

    The triangulator fills the convex hull and a road is not convex -- a
    junction is an L or a T -- so something has to say which of the triangles
    it produced are actually surface.

    Even-odd *within* a group, so a roundabout's island is a hole; union
    *across* groups, so two parts that overlap stay road rather than cancelling
    each other out. Applying even-odd to everything at once is the obvious
    shortcut and it punches a hole at every place two parts meet: on a test
    network of eight crossing streets it made the surface area swing between
    11,132 and 17,236 square metres depending only on how finely it was cut.
    """
    inside = np.zeros(len(points), dtype=bool)
    for group in groups:
        inside |= _even_odd(group, points)
    return inside


# A road face bigger than this is subdivided. A carriageway is nearly flat, so
# nothing about the height demands it -- but two triangles spanning a 280 m
# street are 35 times longer than they are wide, and an engine lights, lightmaps
# and collides against triangle shape, not against area.
DEFAULT_MAX_EDGE_M = 12.0

# Refinement stops chasing shape below this. Delaunay refinement reaches about
# 20 degrees on well-formed input; a road outline has sharp corners where two
# streets meet, and no triangulation puts a good triangle in a wedge the input
# already had.
DEFAULT_MIN_ANGLE_DEG = 22.0

# Below this a thin face is left alone: it is in a corner the survey drew, and
# refining there only makes smaller thin faces.
MIN_SHAPE_EDGE_M = 1.0

# How far a kerb may be bent to meet a vertex that all but lies on it. Well
# under the quarter metre simplification is already allowed to move the same
# kerb, and far enough above the 5 mm weld to catch what the weld cannot.
#
# Not larger, and this is measured rather than taste: at 5 cm it removes four
# fifths of the sub-degree slivers and no constraint crosses another, and at
# 10 cm a bent kerb starts crossing its neighbour and the triangulator drops
# one -- which is the very failure the crossing pass above exists to prevent.
# Past 5 cm it buys almost nothing anyway: 136 slivers become 115.
DEFAULT_SNAP_M = 0.05

MAX_REFINE_ROUNDS = 8


def build_road_mesh(
    groups: list[list[np.ndarray]],
    *,
    simplify_m: float = DEFAULT_SIMPLIFY_M,
    max_edge_m: float = DEFAULT_MAX_EDGE_M,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
    snap_m: float = DEFAULT_SNAP_M,
    sampler=None,
    tolerance_m: float = 0.08,
):
    """Triangulate a road surface into shapes an engine can use.

    Returns `(points_xy, triangles)` -- indexed, conforming, and with every
    outline edge preserved.

    Earcut is what this replaces, and it was the wrong tool for the shape.
    Given a long strip it fans it into slivers, and the sliver is not an
    artefact of the dense input: simplify a 280 m street to a clean rectangle
    and earcut still hands back two triangles 35 times longer than they are
    wide. Constrained Delaunay plus refinement is what turns that into a
    surface, and because refinement inserts points rather than cutting
    individual triangles, it stays conforming -- no hanging nodes, so no cracks
    and no shading seams.
    """
    from .cdt import triangulate

    # Simplified across every group at once, so a kerb two parts share is
    # thinned once and they stay welded to each other.
    flat = [ring for group in groups for ring in group]
    sizes = [len(group) for group in groups]
    thinned, source = simplify_ring_network(flat, simplify_m)
    if not thinned:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int32)
    # A ring can be dropped for no longer being a polygon, so the grouping is
    # rebuilt from what survived rather than assumed.
    groups = _regroup(thinned, source, sizes)

    points, segments = _ring_constraints(thinned)
    if len(points) < 3:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int32)

    # No triangulation can honour two constraints that cross, and the
    # triangulator drops one of them rather than guessing. Silently, which is
    # the problem: the dropped edge is a piece of kerb, so the outline no longer
    # closes and the inside/outside test then keeps whatever it likes. On a test
    # network of crossing streets that made the surface area swing between
    # 13,140 and 20,728 square metres with nothing changed but the face size.
    #
    # BGT road parts are a planar partition and should never cross. Should.
    from .breaklines import resolve_crossings

    points, segments = resolve_crossings(points, segments)

    # Two BGT parts that share a kerb have both drawn it, and the two drawings
    # rarely agree to the millimetre. Where they disagree by less than the weld
    # tolerance the vertices merge; where they disagree by a little more, they
    # do not, and the pair becomes a ribbon of triangles five millimetres tall
    # -- 691 of them under a hundredth of a degree over one square kilometre,
    # with a median height of exactly the 5 mm weld. Welding cannot fix it,
    # because these vertices are near each other's *edges*, not near each
    # other's *vertices*. Snapping them onto those edges can.
    points, segments = _snap_to_edges(points, segments, snap_m, max_edge_m)

    # Nothing cuts the kerbs up front any more. There used to be a pass that
    # chopped every one of them to the face size before triangulating, on the
    # grounds that a 280 m kerb left as one constraint forces every triangle
    # along it to span the whole street. That was true when refinement could
    # not split a constraint; now that it can, the pass is redundant and
    # measurably worse -- over one square kilometre it cost 1,499 extra
    # triangles and nine seconds, because it put vertices along every kerb
    # whether the triangles there needed them or not.
    #
    # Constraints are carried as coordinates rather than indices from here on,
    # because refinement splits them and every round re-welds the point set.
    kerb = np.stack([points[segments[:, 0]], points[segments[:, 1]]], axis=1)
    origin = tuple(points.mean(axis=0))
    steiner = np.zeros((0, 2))
    world = points
    triangles = np.zeros((0, 3), dtype=np.int32)

    for _ in range(MAX_REFINE_ROUNDS):
        merged = np.vstack([kerb.reshape(-1, 2), steiner])
        merged, mapping = weld(merged, WELD_M)
        constraints = mapping[: 2 * len(kerb)].reshape(-1, 2)
        constraints = constraints[constraints[:, 0] != constraints[:, 1]]
        try:
            result = triangulate(
                merged, constraints, origin=origin, skip_crossing=True
            )
        except Exception as exc:  # noqa: BLE001 - one bad outline is not fatal
            LOG.debug("could not triangulate a road surface: %s", exc)
            return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int32)

        world = result.world_points()
        keep = _inside(groups, world[result.triangles].mean(axis=1))
        triangles = result.triangles[keep]
        if not len(triangles):
            break

        wanted = _points_to_add(
            world, triangles, groups, max_edge_m, min_angle_deg,
            sampler, tolerance_m,
        )
        if not len(wanted):
            break

        # Ruppert's rule, and leaving it out is what made refinement actively
        # harmful: a point that lands inside a kerb segment's diametral circle
        # cannot improve the triangle, because the triangulator may not flip
        # across a constraint. It only wedges a thinner sliver between itself
        # and the kerb, which the next round then chases closer still. Split
        # the kerb instead -- that fixes the sliver and keeps the outline.
        wanted, split = _split_encroached(kerb, wanted)
        # Only now, once the outliers have had their say about the kerb.
        wanted = wanted[_inside(groups, wanted)] if len(wanted) else wanted
        if len(split):
            kerb = split
        if not len(wanted) and not len(split):
            break
        steiner = np.vstack([steiner, wanted]) if len(wanted) else steiner

    return _compact(world, np.asarray(triangles, dtype=np.int32))


def _compact(points: np.ndarray, triangles: np.ndarray):
    """Drop points no triangle refers to, and renumber what is left.

    A candidate can be inserted and then find every triangle around it trimmed
    away as being outside the road. What it leaves behind is a vertex nothing
    draws, which still costs a slot in the buffer and shows up in Blender as a
    loose point for someone to wonder about.
    """
    if not len(triangles):
        return np.zeros((0, 2)), triangles
    used = np.unique(triangles)
    if len(used) == len(points):
        return points, triangles
    renumber = np.zeros(len(points), dtype=np.int32)
    renumber[used] = np.arange(len(used), dtype=np.int32)
    return points[used], renumber[triangles]


def _split_encroached(kerb: np.ndarray, candidates: np.ndarray):
    """Drop candidates that encroach on a kerb, halving the kerb instead.

    Returns `(surviving candidates, kerb)`, where the kerb is a new array if
    anything was split and empty if nothing was.

    A point encroaches a segment when it falls inside the circle that has the
    segment as its diameter. The grid is sized from the median segment rather
    than the longest, so one 280 m kerb among thousands of two-metre ones does
    not collapse it into a single cell; a segment wider than a cell is simply
    filed under all of them.
    """
    if not len(kerb) or not len(candidates):
        return candidates, np.zeros((0, 2, 2))

    middle = kerb.mean(axis=1)
    half = np.hypot(*(kerb[:, 1] - kerb[:, 0]).T) * 0.5
    cell = max(float(np.median(half)) * 2.0, 1e-6)

    buckets: dict = {}
    low = (middle - half[:, None]) // cell
    high = (middle + half[:, None]) // cell
    for index in range(len(kerb)):
        for cx in range(int(low[index, 0]), int(high[index, 0]) + 1):
            for cy in range(int(low[index, 1]), int(high[index, 1]) + 1):
                buckets.setdefault((cx, cy), []).append(index)

    survives = np.ones(len(candidates), dtype=bool)
    doomed: set = set()
    for index, point in enumerate(candidates):
        # The disc is inside the segment's bounding box grown by its own half
        # length, which is what was filed, so the point's own cell is enough.
        near = buckets.get((int(point[0] // cell), int(point[1] // cell)))
        if not near:
            continue
        near = np.asarray(near)
        hit = near[np.hypot(*(middle[near] - point).T) < half[near]]
        if len(hit):
            survives[index] = False
            doomed.update(int(i) for i in hit)

    if not doomed:
        return candidates, np.zeros((0, 2, 2))

    order = np.array(sorted(doomed), dtype=np.int64)
    mid = kerb[order].mean(axis=1)
    halves = np.concatenate(
        [
            np.stack([kerb[order][:, 0], mid], axis=1),
            np.stack([mid, kerb[order][:, 1]], axis=1),
        ]
    )
    intact = np.delete(kerb, order, axis=0)
    return candidates[survives], np.concatenate([intact, halves])


def _snap_to_edges(points, segments, epsilon, reach):
    """Give a segment a vertex wherever a stray point almost lies on it.

    The point is not moved; the segment is bent onto it, by at most `epsilon`.
    That is the right way round: the point may be a corner two other kerbs
    share, and moving it would pull them with it, while the segment it lands on
    has nothing else depending on where its middle runs.

    `reach` only sizes the grid. Segments are filed under every cell their
    bounding box touches rather than under their midpoint, because this runs
    before the kerbs are cut down and a 280 m one filed under its middle would
    be invisible to a point at its end.
    """
    if epsilon <= 0 or len(segments) < 2:
        return points, segments

    a = points[segments[:, 0]]
    b = points[segments[:, 1]]
    span = b - a
    length2 = np.maximum((span * span).sum(axis=1), 1e-18)
    cell = max(float(reach), 1e-6)

    buckets: dict = {}
    low = np.minimum(a, b) - epsilon
    high = np.maximum(a, b) + epsilon
    for index in range(len(segments)):
        x0, y0 = int(low[index, 0] // cell), int(low[index, 1] // cell)
        x1, y1 = int(high[index, 0] // cell), int(high[index, 1] // cell)
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                buckets.setdefault((cx, cy), []).append(index)

    by_point: dict = {}
    for index, point in enumerate(points):
        by_point.setdefault((int(point[0] // cell), int(point[1] // cell)), []).append(index)

    extra: dict = {}
    for cell_key, members in by_point.items():
        # One cell is enough: a segment is already filed under every cell its
        # bounding box reaches, and that box is grown by epsilon first.
        near = buckets.get(cell_key)
        if not near:
            continue
        near = np.asarray(near)
        chunk = np.asarray(members)
        here = points[chunk]

        t = ((here[:, None, :] - a[near][None]) * span[near][None]).sum(-1)
        t = t / length2[near][None]
        # Only the middle of a segment. A point near an end is a point near
        # that end's vertex, which is the weld's business, not this pass's.
        proj = a[near][None] + t[..., None] * span[near][None]
        gap = np.hypot(*(here[:, None, :] - proj).transpose(2, 0, 1))
        hit = (gap < epsilon) & (t > 1e-9) & (t < 1 - 1e-9)
        # Never bend a segment onto one of its own endpoints.
        hit &= chunk[:, None] != segments[near][None, :, 0]
        hit &= chunk[:, None] != segments[near][None, :, 1]
        for row, column in zip(*np.nonzero(hit)):
            extra.setdefault(int(near[column]), []).append(
                (float(t[row, column]), int(chunk[row]))
            )

    if not extra:
        return points, segments

    out = []
    for index in range(len(segments)):
        start, end = int(segments[index, 0]), int(segments[index, 1])
        cuts = extra.get(index)
        if not cuts:
            out.append((start, end))
            continue
        chain = [start]
        for _, vertex in sorted(cuts):
            if vertex != chain[-1]:
                chain.append(vertex)
        if chain[-1] != end:
            chain.append(end)
        out.extend(zip(chain[:-1], chain[1:]))
    return points, np.asarray(out, dtype=np.int64)



def _regroup(rings, source, sizes):
    """Put the surviving rings back into the groups they came from."""
    bounds, at = [], 0
    for size in sizes:
        bounds.append((at, at + size))
        at += size
    groups = [[] for _ in sizes]
    for ring, index in zip(rings, source):
        for g, (lo, hi) in enumerate(bounds):
            if lo <= index < hi:
                groups[g].append(ring)
                break
    return [g for g in groups if g]


def _ring_constraints(rings: list[np.ndarray]):
    """Ring vertices as one point array, and their edges as index pairs."""
    chunks = []
    pairs = []
    offset = 0
    for ring in rings:
        body = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
        n = len(body)
        if n < 3:
            continue
        chunks.append(body)
        for k in range(n):
            pairs.append((offset + k, offset + (k + 1) % n))
        offset += n
    if not chunks:
        return np.zeros((0, 2)), np.zeros((0, 2), dtype=np.int64)
    return np.vstack(chunks), np.asarray(pairs, dtype=np.int64)


def _points_to_add(
    points, triangles, groups, max_edge_m, min_angle_deg, sampler, tolerance_m
):
    """Where to insert, to make the faces the right size, shape and height."""
    from .terrain_mesh import _circumcentres, _longest_edge, _min_angles

    flat = np.column_stack([points, np.zeros(len(points))])
    longest = _longest_edge(flat, triangles)
    angles = _min_angles(flat, triangles)
    oversized = longest > max_edge_m
    misshapen = angles < min_angle_deg
    # Size and shape need different floors. A face at the size cap is done; a
    # thin one is not, and gating both on the same generous floor left the
    # median angle at 12 degrees because every sliver was too short to qualify.
    # Shape is chased much further down, stopping only where a road outline's
    # own sharp corners begin -- no triangulation improves a wedge the input
    # already had.
    bad = (oversized & (longest > 0.5 * max_edge_m)) | (
        misshapen & (longest > MIN_SHAPE_EDGE_M)
    )
    # And where the face has left the ground. Done here rather than by cutting
    # triangles afterwards, which is what left hanging nodes: a point inserted
    # into the triangulation is seen by every face that touches it, so the
    # surface stays conforming and there is no crack to hide.
    if sampler is not None:
        corners = points[triangles]
        centroid = corners.mean(axis=1)
        corner_z = np.asarray(
            sampler(corners[:, :, 0].ravel(), corners[:, :, 1].ravel())
        ).reshape(-1, 3)
        middle = np.asarray(sampler(centroid[:, 0], centroid[:, 1]))
        off_ground = np.abs(corner_z.mean(axis=1) - middle) > tolerance_m
        bad |= off_ground & (longest > MIN_SHAPE_EDGE_M)
    if not bad.any():
        return np.zeros((0, 2))

    centre, radius = _circumcentres(flat, triangles)
    # Circumcentres that land outside the road are kept for now, and dropped
    # by the caller after the encroachment test has seen them. They are the
    # signal that a triangle is up against the boundary, and the kerb split
    # they trigger is what fixes it.
    #
    # This used to stand the triangle's centroid in for them instead, which is
    # the worst available choice: a thin triangle is thin because it lies along
    # a kerb, so its centroid is a hair off that kerb. The encroachment split
    # would now catch such a point anyway, so removing it changes almost
    # nothing on its own -- but see the module docstring for what it did in
    # company.
    where = np.flatnonzero(bad)
    candidate, keep_radius = centre[where], radius[where]
    if not len(candidate):
        return np.zeros((0, 2))

    # Biggest offenders first, and no two new points on top of each other. The
    # separation is the local circumradius rather than a fraction of the size
    # cap: a fixed distance throttles exactly the fine work that fixing a thin
    # face needs, and left the median angle at ten degrees.
    order = np.argsort(-keep_radius)
    return _spread(candidate[order], keep_radius[order])


def _spread(points, radius):
    """Thin a batch so two new points cannot make the sliver they came to fix.

    Separation scaled to each point's own triangle, so fine work stays possible
    where it is needed and coarse faces are not over-served.
    """
    cell = max(float(np.median(radius)) * 0.5, 1e-6)
    taken: dict = {}
    keep = []
    for index, point in enumerate(points):
        want = max(0.5 * float(radius[index]), cell * 0.25)
        rings = min(int(np.ceil(want / cell)), 4)
        cx, cy = int(point[0] // cell), int(point[1] // cell)
        clash = False
        for dx in range(-rings, rings + 1):
            for dy in range(-rings, rings + 1):
                for other in taken.get((cx + dx, cy + dy), ()):
                    if np.hypot(*(point - other)) <= want:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if clash:
            continue
        taken.setdefault((cx, cy), []).append(point)
        keep.append(index)
    return points[keep]


__all__ = [
    "DEFAULT_MAX_EDGE_M",
    "DEFAULT_MIN_ANGLE_DEG",
    "DEFAULT_SIMPLIFY_M",
    "build_road_mesh",
    "simplify_ring_network",
]
