"""The layered style's terrain as quads, the way the modellers lay theirs out.

What the modellers asked for, looking at the triangulated terrain:

* **Quads, not a triangulation.** Small triangles everywhere read as noise;
  a grid of quads with the survey's own small rises and dips reads as ground.
* **Roads as whole quads across**, one after the other along the road.
* **Edge loops that follow the shape**: along an embankment, a verge, a ditch.
* **Points where the outlines meet**: round a roundabout the rungs run
  across from the island to the outer edge, so both outlines share them.
* **Ditches**, carved in.
* **One resolution everywhere.** Asked again, with the model in hand: a
  grid that coarsens away from the middle and loops cut wherever the survey
  rises left patches far denser than the ground there needs, and a mix of
  sizes is hard to edit. They divide the model themselves.

So the ground is meshed region by region with the survey style's machinery
(`terrain_mesh`): a row of quads along every outline, rungs straight across
anything too narrow for two rows -- a road up to `PAVED_ROW_M` across is one
ladder of quads -- and one grid of `NEAR_CELL_M` squares inside, the same
everywhere. The grid is turned to the way the streets run (`_grid_angle`), so
a row along a street meets it square on and the two stay quads.
"""

from __future__ import annotations

import logging
import time

import numpy as np

LOG = logging.getLogger(__name__)

# Rows this wide along paved outlines, so a road up to about twice this
# across gets rungs straight over it: one quad from kerb to kerb.
PAVED_ROW_M = 10.0
# Rows this wide along the outlines of grass and open ground, so a verge, a
# strip between carriageways or the slope of a dike up to about twice this
# across is a ladder of quads following its shape, not a grid cut across it.
LAND_ROW_M = 6.0
# The grid squares, everywhere; and the spacing of the stations along every
# outline, so the rows meet the grid square on.
NEAR_CELL_M = 4.0
# An outline along the edge of the area up to this long is left in one piece.
FRAME_WHOLE_M = 2.0 * PAVED_ROW_M
# A rung across a road meets the far kerb within this many degrees of square.
SQUARE_DEG = 30.0


def _with_stations(polygon, spacing, centre, size=None, turn=0.0):
    """A polygon with each edge cut into equal parts, spaced by distance.

    As `terrain_mesh._with_stations`: every edge is cut from the same end
    whichever ring runs along it, and its spacing is read at its middle, so
    the two regions sharing it compute the same stations. Edges along the edge
    of the area (``size`` is its width and height) are left whole up to
    `FRAME_WHOLE_M`: nothing is on the other side, and a road cut off there
    ends in one quad, not a fan. A longer one -- a field running along the
    edge -- is cut like any other, or the strip along it has no vertices.
    With the grid turned (`turn` degrees about `centre`, see `_grid_angle`)
    the polygon is in the turned frame, and the edge of the area is found by
    turning its corners back.
    """
    import shapely

    if polygon is None or polygon.is_empty or polygon.geom_type != "Polygon":
        return polygon
    rings = []
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords)[:, :2]
        level = _turn(coords, turn, centre) if turn else coords
        out = [coords[0]]
        for (a, b), (la, lb) in zip(zip(coords[:-1], coords[1:]), zip(level[:-1], level[1:])):
            if size is not None and float(np.hypot(*(b - a))) <= FRAME_WHOLE_M and any(
                abs(la[i] - edge) < 1e-6 and abs(lb[i] - edge) < 1e-6
                for i in (0, 1) for edge in (0.0, size[i])
            ):
                out.append(b)
                continue
            length = float(np.hypot(*(b - a)))
            step = spacing(0.5 * (a + b), centre)
            count = int(np.ceil(length / step - 1e-9))
            if count > 1:
                forward = (a[0], a[1]) < (b[0], b[1])
                lo, hi = (a, b) if forward else (b, a)
                steps = lo + (hi - lo) * (np.arange(1, count) / count)[:, None]
                out.extend(steps if forward else steps[::-1])
            out.append(b)
        rings.append(np.array(out))
    return shapely.Polygon(rings[0], rings[1:])


def _turn(points, degrees: float, centre):
    """Points turned anticlockwise by `degrees` about `centre`."""
    points = np.asarray(points, dtype=np.float64)
    if not degrees:
        return points.copy()
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    d = points[..., :2] - centre
    out = points.copy()
    out[..., 0] = centre[0] + c * d[..., 0] - s * d[..., 1]
    out[..., 1] = centre[1] + s * d[..., 0] + c * d[..., 1]
    return out


# The grid is turned to the outlines only when they agree on a direction:
# at least this share of their length within `ALIGN_WITHIN_DEG` of it.
ALIGN_SHARE = 0.25
ALIGN_WITHIN_DEG = 4.0


def _grid_angle(pieces, size) -> float:
    """The way the streets run, in degrees between -45 and 45; 0 for none.

    Rows follow every outline, and where a row along a street meets a grid
    square-on the two are one quad wide; where it meets it at 40 degrees, as
    across most of Hoek van Holland, every cell along the street is cut into
    triangles. So the grid is turned to the direction most of the outline
    length runs in, counted modulo 90 degrees -- a street and the side
    street square to it agree -- leaving out the edge of the area, which
    runs north-south whatever the streets do.
    """
    lengths, angles = [], []
    for polygon, _ in pieces:
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
                keep = (length > 0.5) & ~on_edge
                lengths.append(length[keep])
                angles.append(np.degrees(np.arctan2(d[keep, 1], d[keep, 0])) % 90.0)
    if not lengths:
        return 0.0
    length = np.concatenate(lengths)
    angle = np.concatenate(angles)
    if not len(length) or length.sum() <= 0:
        return 0.0
    hist = np.bincount(np.floor(angle).astype(int) % 90, weights=length, minlength=90)
    window = int(round(ALIGN_WITHIN_DEG))
    smooth = sum(np.roll(hist, k) for k in range(-window, window + 1))
    peak = float(np.argmax(smooth)) + 0.5
    # Refined to the length-weighted mean near the peak, on the circle of 90.
    off = (angle - peak + 45.0) % 90.0 - 45.0
    near = np.abs(off) <= ALIGN_WITHIN_DEG
    if length[near].sum() < ALIGN_SHARE * length.sum():
        return 0.0
    best = peak + float(np.average(off[near], weights=length[near]))
    best = (best + 45.0) % 90.0 - 45.0
    return 0.0 if abs(best) < 1.0 else round(best, 3)


def _flat_on_frame(xy, sizes, loops, size) -> np.ndarray:
    """Faces with no area whose corners all lie on one edge of the area."""
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    face_of = np.repeat(np.arange(len(sizes)), sizes)
    following = np.arange(len(loops)) + 1
    following[starts + sizes - 1] = starts
    a, b = xy[loops], xy[loops[following]]
    area = np.zeros(len(sizes))
    np.add.at(area, face_of, a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])
    flat = np.abs(0.5 * area) < 1e-6
    on_edge = np.zeros(len(sizes), dtype=bool)
    for i in (0, 1):
        for edge in (0.0, size[i]):
            there = np.abs(xy[loops, i] - edge) < 1e-6
            on_edge |= np.bincount(face_of, weights=there, minlength=len(sizes)) == sizes
    return flat & on_edge


def _rungs_miss(across, height_at) -> float:
    """How far the survey is off the straight rungs across a road, at their
    quarter, middle and three-quarter points: the 90th percentile, so one
    rung over a ditch does not split a whole street."""
    if not across:
        return 0.0
    ends = np.array([np.asarray(line.coords)[[0, -1], :2] for line in across])
    a, b = ends[:, 0], ends[:, 1]
    za, zb = height_at(a), height_at(b)
    worst = np.zeros(len(across))
    for t in (0.25, 0.5, 0.75):
        z = height_at(a + (b - a) * t)
        worst = np.maximum(worst, np.abs(z - (za + (zb - za) * t)))
    worst = worst[np.isfinite(worst)]
    return float(np.percentile(worst, 90)) if len(worst) else 0.0


def _divide_rungs(faces, across, step_m: float, count: int | None = None):
    """The rungs across a strip of ground cut into pieces about `step_m` long.

    A ladder straight across a verge or the slope of a dike has vertices only
    at its edges, and whatever the ground does in between -- the crest of the
    dike, the bottom of a dip -- is lost. Cut every rung into the same number
    of pieces and the cuts line up from rung to rung: edge loops running along
    the strip, following its shape, at the heights the survey has there.

    Returns the faces, each ladder cell split into that many quads and any
    other face with a rung edge given the cuts and filled again, and how many
    rungs were cut.
    """
    import shapely

    from .terrain_mesh import _ccw, _fill, _key, _snap

    lengths = np.array([line.length for line in across])
    if not len(lengths):
        return faces, 0
    if count is None:
        count = int(np.clip(np.round(np.median(lengths) / step_m), 1, 4))
    if count < 2:
        return faces, 0
    cuts = {}
    for line in across:
        ends = _snap(np.asarray(line.coords)[[0, -1], :2])
        ka, kb = _key(ends[0]), _key(ends[1])
        if ka == kb:
            continue
        a, b = (ends[0], ends[1]) if ka < kb else (ends[1], ends[0])
        points = _snap(a + (b - a) * (np.arange(count + 1) / count)[:, None])
        cuts[tuple(sorted((ka, kb)))] = points

    def along(p, q):
        """The points of the rung from p to q, ends included, or None."""
        key = tuple(sorted((_key(p), _key(q))))
        points = cuts.get(key)
        if points is None:
            return None
        return points if _key(points[0]) == _key(p) else points[::-1]

    out = []
    used = set()
    for coords, part in faces:
        coords = np.asarray(coords)
        n = len(coords)
        rungs = [along(coords[i], coords[(i + 1) % n]) for i in range(n)]
        hit = [r is not None for r in rungs]
        if not any(hit):
            out.append((coords, part))
            continue
        for i in range(n):
            if hit[i]:
                used.add(tuple(sorted((_key(coords[i]), _key(coords[(i + 1) % n])))))
        if n == 4 and hit[1] and hit[3] and not hit[0] and not hit[2]:
            coords = np.roll(coords, -1, axis=0)
            rungs = rungs[1:] + rungs[:1]
            hit = hit[1:] + hit[:1]
        if n == 4 and hit[0] and hit[2] and not hit[1] and not hit[3]:
            first = rungs[0]  # coords[0] to coords[1]
            second = rungs[2][::-1]  # coords[3] to coords[2]
            for i in range(count):
                quad = np.array([first[i], first[i + 1], second[i + 1], second[i]])
                out.append((_ccw(quad), part))
            continue
        ring = []
        for i in range(n):
            ring.append(coords[i][None, :])
            if hit[i]:
                ring.append(rungs[i][1:-1])
        polygon = shapely.Polygon(np.vstack(ring))
        if not polygon.is_valid or polygon.area < 1e-6:
            out.append((coords, part))
            continue
        out.extend((c, part) for c in _fill(polygon))
    return out, len(used)


# A triangle less than this high over its long edge is a sliver: its normal
# points anywhere, and whatever is shrinkwrapped onto it is thrown sideways.
SLIVER_HEIGHT_M = 0.05


def _merge_slivers(local_xy, sizes, loops, face_kind, face_part, face_code,
                   min_height=SLIVER_HEIGHT_M):
    """Every sliver triangle folded into the face across its long edge.

    The sliver's third corner is added to that face's outline, which then
    covers both, and is filled again if it is no longer a triangle or a convex
    quad. Nothing moves and no vertex is lost, so no crack opens.

    Returns the new vertices, sizes, loops, kinds, parts and codes, and how
    many slivers went.
    """
    import shapely

    from .terrain_mesh import _fill, _quad_score

    xy = np.asarray(local_xy, dtype=np.float64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    faces = [list(map(int, loops[s: s + n])) for s, n in zip(starts, sizes)]
    kind, part, code = list(face_kind), list(face_part), list(face_code)
    alive = [True] * len(faces)
    new_xy = [tuple(p) for p in xy.tolist()]
    index = {p: i for i, p in enumerate(new_xy)}

    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    edge_faces: dict = {}

    def link(f):
        v = faces[f]
        for i in range(len(v)):
            edge_faces.setdefault(ekey(v[i], v[(i + 1) % len(v)]), set()).add(f)

    def unlink(f):
        v = faces[f]
        for i in range(len(v)):
            edge_faces.get(ekey(v[i], v[(i + 1) % len(v)]), set()).discard(f)

    for f in range(len(faces)):
        link(f)

    def sliver(f):
        v = faces[f]
        if len(v) != 3:
            return None
        p = xy_of(v)
        lengths = [np.hypot(*(p[(i + 1) % 3] - p[i])) for i in range(3)]
        i = int(np.argmax(lengths))
        if lengths[i] <= 0:
            return None
        a, b = p[i], p[(i + 1) % 3]
        c = p[(i + 2) % 3]
        height = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / lengths[i]
        return i if height < min_height else None

    def xy_of(v):
        return np.array([new_xy[i] for i in v])

    merged = 0
    queue = [f for f in range(len(faces)) if len(faces[f]) == 3]
    # Refilling can make a new sliver, which is merged in turn; the budget
    # ends any cycle of the two.
    budget = 4 * len(queue) + 100
    while queue and budget > 0:
        budget -= 1
        f = queue.pop()
        if not alive[f]:
            continue
        i = sliver(f)
        if i is None:
            continue
        v = faces[f]
        a, b, c = v[i], v[(i + 1) % 3], v[(i + 2) % 3]
        across = [g for g in edge_faces.get(ekey(a, b), ()) if g != f and alive[g]]
        if not across:
            continue
        g = across[0]
        w = faces[g]
        k = next(j for j in range(len(w)) if ekey(w[j], w[(j + 1) % len(w)]) == ekey(a, b))
        ring = w[: k + 1] + [c] + w[k + 1:]
        unlink(f)
        unlink(g)
        alive[f] = alive[g] = False
        pts = xy_of(ring)
        if len(ring) == 4 and _quad_score(pts) is not None and _quad_score(pts) < 75.0:
            pieces = [ring]
        else:
            polygon = shapely.Polygon(pts)
            if not polygon.is_valid:
                # Put both back as they were: better a sliver than a hole.
                alive[f] = alive[g] = True
                link(f)
                link(g)
                continue
            pieces = []
            for corners in _fill(polygon):
                ids = []
                for p in np.asarray(corners):
                    key = (float(p[0]), float(p[1]))
                    if key not in index:
                        near = int(np.argmin(np.hypot(*(pts - p).T)))
                        key = tuple(pts[near])
                    ids.append(index[key])
                pieces.append(ids)
        for piece in pieces:
            faces.append(piece)
            kind.append(kind[g])
            part.append(part[g])
            code.append(code[g])
            alive.append(True)
            link(len(faces) - 1)
            if len(piece) == 3 and len(pieces) == 1:
                queue.append(len(faces) - 1)
        merged += 1

    keep = [f for f in range(len(faces)) if alive[f]]
    return (
        np.array(new_xy, dtype=np.float64),
        np.array([len(faces[f]) for f in keep], dtype=np.int64),
        np.array([i for f in keep for i in faces[f]], dtype=np.int64),
        np.array([kind[f] for f in keep], dtype=np.int64),
        np.array([part[f] for f in keep], dtype=np.int64),
        np.array([code[f] for f in keep], dtype=np.int64),
        merged,
    )


# Water within this of a BGT quay wall is along the quay ...
QUAY_REACH_M = 1.5
# ... if this much of its outline is, and its row there is this wide.
QUAY_MIN_M = 5.0
QUAY_ROW_M = 0.3
# Ground strips thinner than this are folded into a neighbour before meshing.
THIN_M = 0.5

# The steepest a ditch bank is cut: one metre down per metre across, 45°.
BANK_SLOPE = 1.0


def _gentle_banks(heights, loops, sizes, face_kind, xy, wet_kind, slope=BANK_SLOPE,
                  steep=None):
    """No point under water further below the bank than `slope` allows.

    A ditch a metre across cut `ditch_depth_m` below its level in the middle,
    under a bank that stands half a metre above the water, has banks over 60°
    steep: no ditch looks like that, and a layer draped on such a bank is
    lifted along its slant rather than up. The bed is raised where it has to be.
    """
    import shapely

    face_of_loop = np.repeat(np.arange(len(sizes)), sizes)
    wet = face_kind[face_of_loop] == wet_kind
    if not wet.any():
        return heights
    in_water = np.zeros(len(heights), dtype=bool)
    on_land = np.zeros(len(heights), dtype=bool)
    in_water[loops[wet]] = True
    on_land[loops[~wet]] = True
    shore = in_water & on_land
    under = np.flatnonzero(in_water & ~on_land)
    if not shore.any() or not len(under):
        return heights
    # The bank: every edge of a water face along which both ends are shore.
    # Measured to the edge, not to its ends: those can be metres apart along
    # the bank while the bank itself is a hand's breadth away.
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    nxt = np.arange(len(loops)) + 1
    nxt[starts + sizes - 1] = starts
    a_all, b_all = loops, loops[nxt]
    on_bank = wet & shore[a_all] & shore[b_all]
    if steep is not None:
        # A quay is no bank: the water goes straight down at it, and it holds
        # nothing up further out either.
        mid = 0.5 * (xy[a_all] + xy[b_all])
        on_bank &= ~shapely.intersects_xy(steep, mid[:, 0], mid[:, 1])
    a, b = a_all[on_bank], b_all[on_bank]
    if not len(a):
        return heights
    tree = shapely.STRtree(shapely.linestrings(np.stack([xy[a], xy[b]], axis=1)))
    # Every bank edge near enough to matter: a bank `reach_m` away allows a
    # drop of `slope * reach_m`, more than any ditch is cut.
    reach_m = 6.0
    u, k = tree.query(shapely.points(xy[under]), predicate="dwithin", distance=reach_m)
    p = xy[under[u]]
    pa, pb = xy[a[k]], xy[b[k]]
    d = pb - pa
    t = np.clip(np.einsum("ij,ij->i", p - pa, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-12), 0, 1)
    foot = pa + t[:, None] * d
    bank_z = (1 - t) * heights[a[k]] + t * heights[b[k]]
    distance = np.linalg.norm(p - foot, axis=1)
    floor = np.full(len(under), -np.inf)
    np.maximum.at(floor, u, bank_z - slope * distance)
    if steep is not None:
        # Along a quay wall the drop is the point.
        near_quay = shapely.intersects_xy(steep, xy[under, 0], xy[under, 1])
        floor[near_quay] = -np.inf
    out = heights.copy()
    out[under] = np.maximum(heights[under], floor)
    return out


# Vertices closer than this are one: the same point, snapped to the
# centimetre grid on one side of an outline and not on the other, or rounded
# the other way at a half centimetre.
WELD_M = 0.012


def _weld(local_xy, sizes, loops, within=WELD_M):
    """Vertices closer than `within` made one, and the faces mended round them.

    The one on the centimetre grid is kept. A face that loses a corner this
    way keeps the rest -- a quad becomes the triangle it now is -- and one
    left with fewer than three is dropped: it had no area.
    """
    import shapely

    xy = np.asarray(local_xy, dtype=np.float64)
    tree = shapely.STRtree(shapely.points(xy))
    a, b = tree.query(shapely.points(xy), predicate="dwithin", distance=within)
    pick = a < b
    a, b = a[pick], b[pick]
    if not len(a):
        return xy, sizes, loops, 0
    parent = np.arange(len(xy))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    on_grid = (np.abs(xy / 0.01 - np.rint(xy / 0.01)) < 1e-6).all(axis=1)
    for i, j in zip(a, b):
        ri, rj = root(i), root(j)
        if ri == rj:
            continue
        # The root is the one on the grid, if either is.
        if on_grid[rj] and not on_grid[ri]:
            ri, rj = rj, ri
        parent[rj] = ri
    target = np.array([root(i) for i in range(len(xy))])
    mapped = target[loops]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    new_sizes, new_loops = [], []
    for s0, n in zip(starts, sizes):
        ring = list(mapped[s0: s0 + n])
        ring = [v for k, v in enumerate(ring) if v != ring[k - 1]]
        new_sizes.append(len(ring))
        new_loops.extend(ring)
    return xy, np.array(new_sizes, dtype=np.int64), np.array(new_loops, dtype=np.int64), int((target != np.arange(len(xy))).sum())


def _floating(xy, sizes, loops) -> np.ndarray:
    """Per face: whether it is in a piece of mesh that shares no edge with the
    main sheet and lies on top of it."""
    import shapely

    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    nxt = np.arange(len(loops)) + 1
    nxt[starts + sizes - 1] = starts
    face_of_loop = np.repeat(np.arange(len(sizes)), sizes)
    key = np.sort(np.column_stack([loops, loops[nxt]]), axis=1)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    parent = np.arange(len(sizes))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    same = inverse[order[1:]] == inverse[order[:-1]]
    for i, j in zip(face_of_loop[order[:-1]][same], face_of_loop[order[1:]][same]):
        ri, rj = root(int(i)), root(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    comp = np.array([root(i) for i in range(len(sizes))])
    labels, counts = np.unique(comp, return_counts=True)
    if len(labels) < 2:
        return np.zeros(len(sizes), dtype=bool)
    main = labels[np.argmax(counts)]
    main_faces = np.flatnonzero(comp == main)
    polygons = np.array([shapely.Polygon(xy[loops[starts[f]: starts[f] + sizes[f]]])
                         for f in main_faces], dtype=object)
    tree = shapely.STRtree(polygons)
    stray = np.zeros(len(sizes), dtype=bool)
    for label in labels:
        if label == main:
            continue
        faces = np.flatnonzero(comp == label)
        points = shapely.points([xy[loops[starts[f]: starts[f] + sizes[f]]].mean(axis=0) for f in faces])
        hit, _ = tree.query(points, predicate="within")
        if len(np.unique(hit)) == len(faces):
            stray[faces] = True
    return stray


# A vertex this close to the middle of an edge it is not on is on it.
T_JUNCTION_M = 0.1


def _open_edges(xy, sizes, loops, size):
    """Per loop: the edge to the next corner, and whether no other face shares it."""
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    nxt = np.arange(len(loops)) + 1
    nxt[starts + sizes - 1] = starts
    a, b = loops, loops[nxt]
    key = np.sort(np.column_stack([a, b]), axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    lone = counts[inverse.ravel()] == 1
    mid = 0.5 * (xy[a] + xy[b])
    on_frame = (
        (np.abs(mid[:, 0]) < 1e-6) | (np.abs(mid[:, 0] - size[0]) < 1e-6)
        | (np.abs(mid[:, 1]) < 1e-6) | (np.abs(mid[:, 1] - size[1]) < 1e-6)
    )
    return a, b, lone & ~on_frame


def _mend_t_junctions(xy, sizes, loops, face_kind, face_part, face_code, size,
                      within=T_JUNCTION_M):
    """An open edge with an open vertex beside it takes that vertex in.

    Two neighbours meant to share an edge can come out with a vertex on one
    side a few centimetres off the other's straight edge: an edge each, a
    sliver between them, a crack. The face with the straight edge gets the
    vertex, and is filled again if it is no longer a triangle or a quad.
    """
    import shapely

    from .terrain_mesh import _fill, _quad_score

    a, b, is_open = _open_edges(xy, sizes, loops, size)
    if not is_open.any():
        return sizes, loops, face_kind, face_part, face_code, 0
    open_idx = np.flatnonzero(is_open)
    open_vertices = np.unique(np.concatenate([a[open_idx], b[open_idx]]))
    tree = shapely.STRtree(shapely.points(xy[open_vertices]))
    segments = shapely.linestrings(np.stack([xy[a[open_idx]], xy[b[open_idx]]], axis=1))
    seg, hit = tree.query(segments, predicate="dwithin", distance=within)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    face_of_loop = np.repeat(np.arange(len(sizes)), sizes)
    inserts: dict = {}  # loop index -> [(t, vertex)]
    for s_i, v_i in zip(seg, hit):
        loop = open_idx[s_i]
        v = int(open_vertices[v_i])
        p, q = xy[a[loop]], xy[b[loop]]
        if v in (a[loop], b[loop]):
            continue
        d = q - p
        t = float(np.dot(xy[v] - p, d) / max(float(np.dot(d, d)), 1e-12))
        if not 0.02 < t < 0.98:
            continue
        inserts.setdefault(int(loop), []).append((t, v))
    if not inserts:
        return sizes, loops, face_kind, face_part, face_code, 0
    faces = [list(map(int, loops[s0: s0 + n])) for s0, n in zip(starts, sizes)]
    changed = {}
    for loop, items in inserts.items():
        f = int(face_of_loop[loop])
        ring = changed.get(f, faces[f])
        k = ring.index(int(a[loop]))
        added = [v for _, v in sorted(items) if v not in ring]
        if not added:
            continue
        changed[f] = ring[: k + 1] + added + ring[k + 1:]
    out_faces, kind, part, code = [], [], [], []
    for f, ring in enumerate(faces):
        new = changed.get(f)
        if new is None:
            out_faces.append(ring)
            kind.append(face_kind[f]); part.append(face_part[f]); code.append(face_code[f])
            continue
        coords = xy[new]
        pieces = None
        if len(new) == 4 and _quad_score(coords) is not None and _quad_score(coords) < 75.0:
            pieces = [new]
        else:
            polygon = shapely.Polygon(coords)
            if polygon.is_valid and polygon.area > 1e-9:
                lookup = {tuple(c): i for c, i in zip(coords.tolist(), new)}
                pieces = []
                for q in _fill(polygon):
                    ids = [lookup.get(tuple(c)) for c in np.asarray(q).tolist()]
                    if None in ids:
                        pieces = None
                        break
                    pieces.append(ids)
        if not pieces:
            out_faces.append(ring)
            kind.append(face_kind[f]); part.append(face_part[f]); code.append(face_code[f])
            continue
        for piece in pieces:
            out_faces.append(piece)
            kind.append(face_kind[f]); part.append(face_part[f]); code.append(face_code[f])
    return (
        np.array([len(r) for r in out_faces], dtype=np.int64),
        np.array([i for r in out_faces for i in r], dtype=np.int64),
        np.array(kind, dtype=np.int64), np.array(part, dtype=np.int64),
        np.array(code, dtype=np.int64), len(changed),
    )


def _patch_holes(xy, sizes, loops, face_kind, face_part, face_code, size):
    """Any hole left in the sheet filled, and any face lying loose on it dropped.

    Nothing should leave either; this is the last line, because a hole in the
    terrain is a hole in the model and a place nothing can stand, and a loose
    face is ground drawn twice.
    """
    import shapely

    from .terrain_mesh import _fill

    def open_loops(sizes, loops):
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        nxt = np.arange(len(loops)) + 1
        nxt[starts + sizes - 1] = starts
        a, b = loops, loops[nxt]
        key = np.sort(np.column_stack([a, b]), axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        lone = counts[inverse.ravel()] == 1
        mid = 0.5 * (xy[a] + xy[b])
        on_frame = (
            (np.abs(mid[:, 0]) < 1e-6) | (np.abs(mid[:, 0] - size[0]) < 1e-6)
            | (np.abs(mid[:, 1]) < 1e-6) | (np.abs(mid[:, 1] - size[1]) < 1e-6)
        )
        return a, b, lone & ~on_frame

    # A piece of mesh joined to the rest by no edge, lying on top of the sheet,
    # is ground drawn twice: it goes. (A single loose face over Utrecht; a
    # cluster of six edges over Rotterdam.)
    a, b, is_open = open_loops(sizes, loops)
    stray = _floating(xy, sizes, loops)
    if stray.any():
        keep = ~stray
        loops = loops[np.repeat(keep, sizes)]
        sizes, face_kind, face_part, face_code = (
            sizes[keep], face_kind[keep], face_part[keep], face_code[keep]
        )
        a, b, is_open = open_loops(sizes, loops)
    open_edges = np.flatnonzero(is_open)
    if not len(open_edges):
        return sizes, loops, face_kind, face_part, face_code, int(stray.sum())
    face_of_loop = np.repeat(np.arange(len(sizes)), sizes)
    # Round a hole the faces' edges run the other way: from b back to a.
    step = {int(b[e]): (int(a[e]), int(face_of_loop[e])) for e in open_edges}
    new_faces, patched = [], 0
    seen = set()
    for start in list(step):
        if start in seen:
            continue
        ring, owner, v = [], None, start
        while v in step and v not in seen:
            seen.add(v)
            ring.append(v)
            v, owner = step[v]
        if v != start or len(ring) < 3:
            continue
        coords = xy[ring]
        # Walked this way a hole runs counter-clockwise; anything else is the
        # outline of faces, not of a gap between them.
        if _signed_area(coords) <= 0:
            continue
        if len(ring) == 3:
            pieces = [ring]
        else:
            polygon = shapely.Polygon(coords)
            if not polygon.is_valid:
                continue
            lookup = {tuple(c): i for c, i in zip(coords.tolist(), ring)}
            pieces = [[lookup.get(tuple(c), ring[0]) for c in np.asarray(q).tolist()]
                      for q in _fill(polygon)]
        for piece in pieces:
            if _area(xy[piece]) <= 0:
                continue
            if (xy[piece][1, 0] - xy[piece][0, 0]) * (xy[piece][2, 1] - xy[piece][0, 1]) - (
                    xy[piece][2, 0] - xy[piece][0, 0]) * (xy[piece][1, 1] - xy[piece][0, 1]) < 0:
                piece = piece[::-1]
            new_faces.append((piece, owner))
        patched += 1
    if not new_faces:
        return sizes, loops, face_kind, face_part, face_code, int(stray.sum())
    owners = np.array([o for _, o in new_faces], dtype=np.int64)
    return (
        np.concatenate([sizes, [len(f) for f, _ in new_faces]]).astype(np.int64),
        np.concatenate([loops, [i for f, _ in new_faces for i in f]]).astype(np.int64),
        np.concatenate([face_kind, face_kind[owners]]),
        np.concatenate([face_part, face_part[owners]]),
        np.concatenate([face_code, face_code[owners]]),
        patched + int(stray.sum()),
    )


def _signed_area(coords) -> float:
    c = np.asarray(coords, dtype=np.float64)
    return 0.5 * float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))


def _without_overlaps(pieces):
    """Pieces that overlap made not to: each loses what an earlier one covers.

    Returns the pieces and how many overlaps there were. Taking the earlier
    piece away leaves the later one's outline running exactly along the
    earlier one's, so the two still share every vertex of that edge.
    """
    import shapely

    from .terrain_mesh import GRID_SIZE

    polygons = [p for p, _ in pieces]
    codes = [c for _, c in pieces]
    tree = shapely.STRtree(np.array(polygons, dtype=object))
    a, b = tree.query(np.array(polygons, dtype=object), predicate="overlaps")
    pairs = sorted({(int(min(i, j)), int(max(i, j))) for i, j in zip(a, b) if i != j})
    count = 0
    for i, j in pairs:
        if polygons[i].is_empty or polygons[j].is_empty:
            continue
        shared = polygons[i].intersection(polygons[j])
        if shared.area <= 1e-4:
            continue
        cut = shapely.set_precision(polygons[j].difference(polygons[i]), GRID_SIZE)
        polygons[j] = _polygonal_only(cut)
        count += 1
    out = []
    for polygon, code in zip(polygons, codes):
        for part in _parts(polygon):
            if part.area > 1e-4:
                out.append((shapely.orient_polygons(part), code))
    return out, count


def _without_thin(pieces, thin_m: float, wet_code=None):
    """Strips thinner than `thin_m` folded into the neighbour they share most
    outline with. Returns the pieces and how many were folded.

    Thinness is twice the area over the perimeter: a strip's width. The BGT
    has borders of shrubs, paving and lawn 35 to 40 cm wide side by side along
    a quay at Hoek van Holland, narrower than any row the mesher lays, and the
    rows either side of them did not land on their outlines: 21 cracks. A
    strip that narrow is not something a modeller draws as its own surface.
    Land strips go to land where they can, so a hedge along the water does not
    become water.
    """
    import shapely

    from .terrain_mesh import GRID_SIZE

    polygons = [p for p, _ in pieces]
    codes = [c for _, c in pieces]

    def thickness(p):
        return 2.0 * p.area / p.length if p.length > 0 else 0.0

    order = sorted(
        (i for i, p in enumerate(polygons) if not p.is_empty and thickness(p) < thin_m),
        key=lambda i: thickness(polygons[i]),
    )
    if not order:
        return pieces, 0
    strips = set(order)
    tree = shapely.STRtree(np.array(polygons, dtype=object))
    folded = 0
    for i in order:
        strip = polygons[i]
        if strip.is_empty or thickness(strip) >= thin_m:
            continue
        best, best_len = None, 0.0
        for j in tree.query(strip.buffer(0.01)):
            j = int(j)
            if j == i or polygons[j].is_empty:
                continue
            if j in strips and thickness(polygons[j]) < thin_m:
                continue  # never into another strip, or two would swap
            shared = strip.boundary.intersection(polygons[j].buffer(0.01)).length
            # Land into land before land into water.
            if wet_code is not None and codes[i] != wet_code and codes[j] == wet_code:
                shared *= 0.01
            if shared > best_len:
                best, best_len = j, shared
        if best is None or best_len <= 0.0:
            continue
        merged = shapely.set_precision(shapely.union(polygons[best], strip), GRID_SIZE)
        polygons[best] = _polygonal_only(merged)
        polygons[i] = shapely.Polygon()
        folded += 1
    out = []
    for polygon, code in zip(polygons, codes):
        for part in _parts(polygon):
            if part.area > 1e-4:
                out.append((shapely.orient_polygons(part), code))
    return out, folded


def _polygonal_only(geometry):
    import shapely

    parts = _parts(geometry)
    if not parts:
        return shapely.Polygon()
    return shapely.MultiPolygon(parts) if len(parts) > 1 else parts[0]


def _area(coords) -> float:
    c = np.asarray(coords, dtype=np.float64)
    return 0.5 * abs(float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])))


def quad_terrain(terrain, pieces, bodies, *, codes_wet, depth_m,
                 paved_codes, narrow_codes=(), near_m: float = NEAR_CELL_M, quays=(),
                 tolerance_m: float = 0.5):
    """The terrain mesh, quads first, conforming to every piece's outline.

    `pieces` are ``(polygon, code)`` in local coordinates from the first grid
    node; `bodies` the water bodies as ``(local polygon, level NAP)``.
    `codes_wet` is the code water pieces carry, `paved_codes` the codes that
    get the road rows, `narrow_codes` the ones that keep rows one grid
    square wide (footpaths, yards); everything else gets `LAND_ROW_M`. Returns RD vertices with heights, face sizes, loops,
    the code and part of each face, and stats.
    """
    import shapely

    from .terrain_mesh import (
        CORE,
        GRID_SIZE,
        LAND,
        ROAD as HIDDEN_KIND,
        ROW,
        FILL,
        WATER,
        _cross_rungs,
        _fill,
        _fill_gaps,
        _mesh_region,
    )

    started = time.time()
    g = float(terrain.grid_m)
    # The grid squares are `k` survey squares across, so the grid lines up
    # with the stations along the outlines and the rows meet it square on.
    k = max(1, int(round(float(near_m) / g)))
    near_m = k * g
    ox, oy = float(terrain.xs[0]), float(terrain.ys[0])
    xs_full = np.round(np.asarray(terrain.xs, dtype=np.float64) - ox, 6)
    ys_full = np.round(np.asarray(terrain.ys, dtype=np.float64) - oy, 6)
    xs = xs_full[: ((len(xs_full) - 1) // k) * k + 1: k]
    ys = ys_full[: ((len(ys_full) - 1) // k) * k + 1: k]
    if len(xs) < 2 or len(ys) < 2:
        k, near_m, xs, ys = 1, g, xs_full, ys_full
    shift = np.array([ox, oy])
    width, height = float(xs_full[-1]), float(ys_full[-1])
    centre = np.array([0.5 * width, 0.5 * height])
    frame = shapely.box(0.0, 0.0, width, height)
    margin_m = 0.35 * g
    widest = max(PAVED_ROW_M, g)
    outer = shapely.difference(frame.buffer(4.0 * widest, join_style="mitre"), frame)

    timings = {}
    def spacing(mid, centre):
        return near_m
    # Each region is meshed on its own, so a gap between two of them -- left
    # where a hairpin was trimmed, say -- would be a hole. It goes to the
    # neighbour it shares most outline with.
    filled = _fill_gaps(np.array([p for p, _ in pieces], dtype=object), frame)
    pieces = [(part, code) for polygon, (_, code) in zip(filled, pieces)
              for part in _parts(polygon)]
    # And where two overlap, the overlap goes to the first: meshed twice it
    # was ground drawn twice, its edges matching neither neighbour (fourteen
    # over Rotterdam's Rijnhaven).
    pieces, overlaps = _without_overlaps(pieces)
    pieces, thin = _without_thin(pieces, THIN_M, wet_code=codes_wet)
    # The grid turned to the streets, by turning everything else the other
    # way and turning the mesh back once it is assembled.
    turn = _grid_angle(pieces, (width, height))
    if turn:
        pieces = [
            (shapely.transform(p, lambda c: _turn(c, -turn, centre)), code)
            for p, code in pieces
        ]
        frame = shapely.Polygon(_turn(np.asarray(frame.exterior.coords), -turn, centre))
        outer = shapely.difference(frame.buffer(4.0 * widest, join_style="mitre"), frame)
        reach = int(np.ceil((0.5 * np.hypot(width, height) + near_m) / near_m))
        xs = centre[0] + near_m * np.arange(-reach, reach + 1)
        ys = centre[1] + near_m * np.arange(-reach, reach + 1)
    parts, codes = [], []
    for polygon, code in pieces:
        polygon = _with_stations(polygon, spacing, centre, (width, height), turn=turn)
        polygon = shapely.orient_polygons(shapely.set_precision(polygon, GRID_SIZE))
        for piece in _parts(polygon):
            if piece.area > 1e-4:
                parts.append((piece, WATER if code == codes_wet else LAND))
                codes.append(int(code))
    rows = [
        PAVED_ROW_M if c in paved_codes
        else g if (kind == WATER or c in narrow_codes)
        else max(LAND_ROW_M, g)
        for (_, kind), c in zip(parts, codes)
    ]
    # Water along a quay wall gets a row a hand's breadth wide: the ground
    # drops from the quay top to the bed in that, as a quay does, rather than
    # sloping down two metres of bank.
    quay_zone = None
    if quays:
        quay_zone = shapely.union_all(np.array(
            [shapely.transform(q, lambda c: c - shift) for q in quays], dtype=object
        )).buffer(QUAY_REACH_M)
        shapely.prepare(quay_zone)
        turned_quays = (
            shapely.transform(quay_zone, lambda c: _turn(c, -turn, centre)) if turn
            else quay_zone
        )
        for i, (polygon, kind) in enumerate(parts):
            if kind == WATER and polygon.boundary.intersection(turned_quays).length > QUAY_MIN_M:
                rows[i] = QUAY_ROW_M
    t = time.time()
    # A rung lands on the station across from it where there is one within
    # half a spacing: the two sides' stations then pair up into one ladder,
    # rather than each side's rungs falling between the other's.
    parts, across = _cross_rungs(
        parts, frame, row_m=g, margin_m=margin_m, rows=rows, square_deg=SQUARE_DEG,
        snap_m=0.5 * near_m,
    )
    timings["cross_rungs_s"] = round(time.time() - t, 1)

    n = len(xs)
    taken = np.zeros((n - 1, n - 1), dtype=bool)
    owner = np.full((n - 1, n - 1), -1, dtype=np.int64)
    faces = []  # (coords local, part, mesher kind, code)
    failed = 0
    divided = 0
    crowned = 0
    refilled = 0

    def height_local(points):
        """The survey's height at points in the (turned) frame the mesher works in."""
        flat = _turn(points, turn, centre) if turn else np.asarray(points, dtype=np.float64)
        return np.asarray(
            terrain.sample(flat[:, 0] + shift[0], flat[:, 1] + shift[1]), dtype=np.float64
        )

    t = time.time()
    for index, (region, kind) in enumerate(parts):
        before = taken.copy()
        try:
            out, cells, _, _ = _mesh_region(
                region, kind, frame, outer, xs, ys, taken,
                row_m=rows[index], margin_m=margin_m, across=across.get(index, ()),
                spaced_fill=True,
            )
        except Exception as exc:  # noqa: BLE001 - one region, not the terrain
            LOG.debug("layered terrain: region %d meshed without rows: %s", index, exc)
            failed += 1
            taken[:] = before
            out, cells, _, _ = _mesh_region(
                region, HIDDEN_KIND, frame, outer, xs, ys, taken,
                row_m=rows[index], margin_m=margin_m,
            )
            out = [(c, ROW if p == FILL else p) for c, p in out]
        # A region its faces do not cover -- a sliver a few centimetres wide,
        # whose faces came out too thin to keep -- would be a hole. Filled
        # whole from its outline instead, it shares every vertex with its
        # neighbours all the same.
        covered = sum(_area(c) for c, _ in out) + len(cells) * near_m * near_m
        if abs(covered - region.area) > max(1e-3 * region.area, 0.01):
            taken[:] = before
            cells = np.zeros((0, 2), dtype=np.int64)
            out = [(c, FILL) for c in _fill(region)]
            refilled += 1
        if kind == LAND and codes[index] not in paved_codes and across.get(index):
            out, n_divided = _divide_rungs(out, across[index], near_m)
            divided += n_divided
        elif (kind == LAND and across.get(index)
              and _rungs_miss(across[index], height_local) > tolerance_m):
            # A road on a dike or with a crown the survey shows: one loop
            # down its middle, along the whole road, so it stays a ladder.
            out, n_divided = _divide_rungs(out, across[index], near_m, count=2)
            divided += n_divided
            crowned += 1
        faces.extend((c, p, kind, codes[index]) for c, p in out)
        if len(cells):
            owner[cells[:, 1], cells[:, 0]] = index
    timings["regions_meshed_s"] = round(time.time() - t, 1)

    # ---- the grid squares, one size everywhere ----------------------------
    t = time.time()
    jj, ii = np.nonzero(owner >= 0)
    for j, i in zip(jj.tolist(), ii.tolist()):
        o = int(owner[j, i])
        corners = np.array([
            (xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1]),
        ])
        faces.append((corners, CORE, parts[o][1], codes[o]))
    timings["grid_s"] = round(time.time() - t, 1)

    # ---- assemble, one vertex per position ----------------------------------
    vertices, sizes, loops, face_kind, face_part, face_code, finished = _assemble(
        faces, terrain=terrain, bodies=bodies, depth_m=depth_m, shift=shift,
        size=(width, height), turn=turn, centre=centre, quay_zone=quay_zone,
    )
    mended, patched, slivers, welded = (
        finished["t_junctions_mended"], finished["holes_patched_or_strays_dropped"],
        finished["slivers_merged"], finished["vertices_welded"],
    )

    stats = {
        "mesher": "quads",
        "grid_angle_deg": float(turn),
        "grid_squares": {f"{near_m:g} m": int(len(jj))},
        "cross_rungs": int(sum(len(v) for v in across.values())),
        "regions_meshed_without_rows": int(failed),
        "rungs_divided": int(divided),
        "roads_given_a_middle_loop": int(crowned),
        "regions_filled_whole": int(refilled),
        "pieces_overlapping": int(overlaps),
        "thin_strips_folded": int(thin),
        "t_junctions_mended": int(mended),
        "water_along_quays": int(sum(1 for r in rows if r == QUAY_ROW_M)),
        "slivers_merged": int(slivers),
        "vertices_welded": int(welded),
        "holes_patched_or_strays_dropped": int(patched),
        "water_faces": int((face_kind == WATER).sum()),
        **timings,
        "mesher_s": round(time.time() - started, 1),
    }
    return vertices, sizes.astype(np.int64), loops, face_code, face_part, stats


def _assemble(faces, *, terrain, bodies, depth_m, shift, size, turn=0.0, centre=None,
              quay_zone=None):
    """Faces in local coordinates into the finished ground.

    `faces` is ``(corners, part, mesher kind, code)`` per face. One vertex per
    position, welded; T-junctions mended, holes patched, slivers merged; the
    heights read off the survey, the beds carved and the banks eased; every
    quad's diagonal turned to fit. Returns the RD vertices with heights, face
    sizes, loops, kinds, parts and codes, and the counts of what was mended.
    """
    from .terrain_mesh import (
        GRID_SIZE,
        WATER,
        _distinct_corners,
        _orient,
        _water_heights,
    )

    width, height = size
    corner_lists = [np.asarray(c, dtype=np.float64) for c, _, _, _ in faces]
    sizes = np.array([len(c) for c in corner_lists], dtype=np.int64)
    all_corners = np.vstack(corner_lists)
    keys = np.rint(all_corners * 1000.0).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    loops = inverse.ravel().astype(np.int64)
    local_xy = all_corners[first]
    face_kind = np.array([k for _, _, k, _ in faces], dtype=np.int64)
    face_part = np.array([p for _, p, _, _ in faces], dtype=np.int64)
    face_code = np.array([c for _, _, _, c in faces], dtype=np.int64)
    if turn:
        # Back the way it came, the edge of the area put back exactly on it:
        # to the centimetre the turned outlines were rounded to, its corners
        # were off it.
        local_xy = _turn(local_xy, turn, centre)
        for i, edge in ((0, 0.0), (0, width), (1, 0.0), (1, height)):
            near_edge = np.abs(local_xy[:, i] - edge) < 2.0 * GRID_SIZE
            local_xy[near_edge, i] = edge
        local_xy[:, 0] = np.clip(local_xy[:, 0], 0.0, width)
        local_xy[:, 1] = np.clip(local_xy[:, 1], 0.0, height)
    local_xy, sizes, loops, welded = _weld(local_xy, sizes, loops)
    keep = _distinct_corners(loops, sizes) & (sizes >= 3)
    if turn:
        # A sliver at a corner of the area whose corners all went back onto
        # its edge is flat now: nothing, and only the edge of the area is
        # left open without it.
        keep &= ~_flat_on_frame(local_xy, sizes, loops, (width, height))
    if not keep.all():
        loops = loops[np.repeat(keep, sizes)]
        sizes, face_kind, face_part, face_code = (
            sizes[keep], face_kind[keep], face_part[keep], face_code[keep]
        )
    sizes, loops, face_kind, face_part, face_code, mended = _mend_t_junctions(
        local_xy, sizes, loops, face_kind, face_part, face_code, (width, height),
    )
    sizes, loops, face_kind, face_part, face_code, patched = _patch_holes(
        local_xy, sizes, loops, face_kind, face_part, face_code, (width, height),
    )
    used = np.unique(loops)
    if len(used) < len(local_xy):
        remap = np.full(len(local_xy), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        loops = remap[loops]
        local_xy = local_xy[used]

    local_xy, sizes, loops, face_kind, face_part, face_code, slivers = _merge_slivers(
        local_xy, sizes, loops, face_kind, face_part, face_code,
    )

    xy = local_xy + shift
    # Every vertex reads the survey grid, between its nodes as well: the 0.5 m
    # survey itself, read at an outline point on a retaining wall, made faces
    # standing nearly upright, slivers seen from above, on which whatever is
    # draped picks one sliver and a check the next. The grid is never steeper
    # than it is between two of its nodes.
    heights = np.asarray(terrain.sample(xy[:, 0], xy[:, 1]), dtype=np.float64)
    heights = _water_heights(heights, loops, sizes, face_kind, xy, bodies, shift, depth_m)
    heights = _gentle_banks(heights, loops, sizes, face_kind, local_xy, WATER, steep=quay_zone)
    vertices = np.column_stack([xy, heights])
    sizes, loops = _orient(vertices, sizes, loops, face_part, face_kind, terrain)
    counts = {
        "t_junctions_mended": int(mended),
        "holes_patched_or_strays_dropped": int(patched),
        "slivers_merged": int(slivers),
        "vertices_welded": int(welded),
    }
    return vertices, sizes, loops, face_kind, face_part, face_code, counts


def _parts(geometry):
    from .terrain_mesh import _parts as parts

    return parts(geometry)


__all__ = ["NEAR_CELL_M", "PAVED_ROW_M", "quad_terrain"]
