"""The layered style's terrain as quads, the way the modellers lay theirs out.

What the modellers asked for, looking at the triangulated terrain:

* **Quads, not a triangulation.** Small triangles everywhere read as noise;
  a grid of quads with the survey's own small rises and dips reads as ground.
* **Roads as whole quads across**, one after the other along the road.
* **Edge loops that follow the shape**: along an embankment, a verge, a ditch.
* **Points where the outlines meet**: round a roundabout the rungs run
  across from the island to the outer edge, so both outlines share them.
* **Ditches**, carved in.
* **Fewer points far away.**

So the ground is meshed region by region with the survey style's machinery
(`terrain_mesh`): a row of quads along every outline, rungs straight across
anything too narrow for two rows -- a road up to `PAVED_ROW_M` across is one
ladder of quads -- and the terrain grid inside. Away from the middle the grid
squares are merged into bigger ones, two by two, as far as the distance
allows and as long as the merged square stays within the tolerance of the
survey. A merged square beside smaller ones takes their corner on its side
and is split into a quad and a triangle or two, so there is never a crack.
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
# Grid squares in the middle of the area are at most this big; out to the
# focus distance twice this, beyond it `FAR_CELLS` times, where the survey
# is flat enough for it.
NEAR_CELL_M = 4.0
FAR_CELLS = 8
# A rung across a road meets the far kerb within this many degrees of square.
SQUARE_DEG = 30.0


def _spacing(focus_m: float, near_m: float):
    """Station spacing by how far out an edge is: the same on both sides of it.

    The grid's own spacing out to `focus_m`, so the rows meet the grid square
    on; twice that beyond, where fewer points matter more than tidy rows.
    """
    def spacing(mid: np.ndarray, centre: np.ndarray) -> float:
        d = float(np.max(np.abs(mid - centre)))
        return near_m if d <= focus_m else 2.0 * near_m
    return spacing


def _with_stations(polygon, spacing, centre, size=None):
    """A polygon with each edge cut into equal parts, spaced by distance.

    As `terrain_mesh._with_stations`: every edge is cut from the same end
    whichever ring runs along it, and its spacing is read at its middle, so
    the two regions sharing it compute the same stations. Edges along the edge
    of the area (``size`` is its width and height) are left whole: nothing is
    on the other side, and a road cut off there ends in one quad, not a fan.
    """
    import shapely

    if polygon is None or polygon.is_empty or polygon.geom_type != "Polygon":
        return polygon
    rings = []
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords)[:, :2]
        out = [coords[0]]
        for a, b in zip(coords[:-1], coords[1:]):
            if size is not None and any(
                abs(a[i] - edge) < 1e-6 and abs(b[i] - edge) < 1e-6
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


def _divide_rungs(faces, across, step_m: float):
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


def _loop_cuts(local_xy, sizes, loops, face_kind, face_part, face_code, *, height_at,
               nodes_xy, nodes_z, tolerance_m, skip_kind, max_cuts=4000, max_run=400,
               min_edge_m=1.0):
    """Edge loops cut through the faces that miss the survey, as a modeller would.

    The faces are laid out from the outlines and the grid; between their
    corners the ground can still do something they do not -- the ballast
    ridge of a railway, the crest of a dike in a verge. Where a face is more
    than `tolerance_m` off a survey grid node inside it, the pair of opposite
    edges whose middles bring it closest is cut, and the cut is carried on
    through the quads beside it, edge to opposite edge, as Blender's loop cut
    does: a new loop along the strip. It ends in a triangle, at a face of
    `skip_kind` (water, carved on purpose), or after `max_run` quads, with
    the last face split into a quad and a triangle so there is never a crack.

    Returns the new vertices, sizes, loops, kinds, parts and codes, and how
    many loops were cut.
    """
    import heapq

    import shapely

    xy = [tuple(p) for p in np.asarray(local_xy, dtype=np.float64).tolist()]
    z = [float(v) for v in height_at(np.asarray(local_xy, dtype=np.float64))]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    faces = [list(map(int, loops[s: s + n])) for s, n in zip(starts, sizes)]
    kind, part, code = list(face_kind), list(face_part), list(face_code)
    alive = [True] * len(faces)
    nodes_xy = np.asarray(nodes_xy, dtype=np.float64)
    nodes_z = np.asarray(nodes_z, dtype=np.float64)

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

    # Which survey nodes lie in which face.
    polygons = np.array([shapely.Polygon([xy[i] for i in v]) for v in faces], dtype=object)
    node_lists: dict = {}
    if len(faces) and len(nodes_xy):
        tree = shapely.STRtree(polygons)
        ni, fi = tree.query(shapely.points(nodes_xy), predicate="intersects")
        order = np.argsort(ni, kind="stable")
        ni, fi = ni[order], fi[order]
        first = np.ones(len(ni), dtype=bool)
        first[1:] = ni[1:] != ni[:-1]
        for n, f in zip(ni[first], fi[first]):
            node_lists.setdefault(int(f), []).append(int(n))
    node_lists = {f: np.array(v, dtype=np.int64) for f, v in node_lists.items()}

    def inside(points, ring):
        """Which points lie in the convex counter-clockwise ring."""
        ok = np.ones(len(points), dtype=bool)
        for i in range(len(ring)):
            ax, ay = ring[i]
            bx, by = ring[(i + 1) % len(ring)]
            cross = (bx - ax) * (points[:, 1] - ay) - (by - ay) * (points[:, 0] - ax)
            ok &= cross >= -1e-7
        return ok

    def tri_interp(points, a, b, c, za, zb, zc):
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            return np.full(len(points), np.nan), np.zeros(len(points), dtype=bool)
        w1 = ((b[1] - c[1]) * (points[:, 0] - c[0]) + (c[0] - b[0]) * (points[:, 1] - c[1])) / det
        w2 = ((c[1] - a[1]) * (points[:, 0] - c[0]) + (a[0] - c[0]) * (points[:, 1] - c[1])) / det
        w3 = 1.0 - w1 - w2
        hit = (w1 >= -1e-7) & (w2 >= -1e-7) & (w3 >= -1e-7)
        return w1 * za + w2 * zb + w3 * zc, hit

    def error_of(ring_ids, nodes, ring_xy=None, ring_z=None):
        if nodes is None or not len(nodes):
            return 0.0
        pts = nodes_xy[nodes]
        target = nodes_z[nodes]
        rx = ring_xy if ring_xy is not None else [xy[i] for i in ring_ids]
        rz = ring_z if ring_z is not None else [z[i] for i in ring_ids]
        if len(rx) == 3:
            value, _ = tri_interp(pts, rx[0], rx[1], rx[2], rz[0], rz[1], rz[2])
            return float(np.nanmax(np.abs(value - target), initial=0.0))
        best = np.inf
        for d in (0, 1):
            a, b, c, e = d, d + 1, d + 2, (d + 3) % 4
            v1, h1 = tri_interp(pts, rx[a], rx[b], rx[c], rz[a], rz[b], rz[c])
            v2, _ = tri_interp(pts, rx[a], rx[c], rx[e], rz[a], rz[c], rz[e])
            value = np.where(h1, v1, v2)
            best = min(best, float(np.nanmax(np.abs(value - target), initial=0.0)))
        return best

    heap: list = []

    def push(f):
        if kind[f] == skip_kind:
            return
        err = error_of(faces[f], node_lists.get(f))
        if err > tolerance_m:
            heapq.heappush(heap, (-err, f))

    def new_vertex(a, b):
        p = (0.5 * (xy[a][0] + xy[b][0]), 0.5 * (xy[a][1] + xy[b][1]))
        xy.append(p)
        z.append(float(height_at(np.array([p]))[0]))
        return len(xy) - 1

    mids: dict = {}

    def mid(a, b):
        k = ekey(a, b)
        if k not in mids:
            mids[k] = new_vertex(a, b)
        return mids[k]

    def replace(f, children):
        unlink(f)
        alive[f] = False
        nodes = node_lists.pop(f, None)
        for child in children:
            faces.append(child)
            kind.append(kind[f])
            part.append(part[f])
            code.append(code[f])
            alive.append(True)
            g = len(faces) - 1
            link(g)
            if nodes is not None and len(nodes):
                mine = inside(nodes_xy[nodes], [xy[i] for i in child])
                if mine.any():
                    node_lists[g] = nodes[mine]
                    nodes = nodes[~mine]
            push(g)

    def at(v, a, b):
        """Index i with v[i], v[i+1] being a and b in either order."""
        n = len(v)
        for i in range(n):
            if ekey(v[i], v[(i + 1) % n]) == ekey(a, b):
                return i
        return None

    def carry(a, b, m):
        """Carry a cut of edge a-b (middle m) on into the face across it."""
        steps = 0
        while True:
            across = [g for g in edge_faces.get(ekey(a, b), ()) if alive[g]]
            if not across:
                return
            g = across[0]
            v = faces[g]
            i = at(v, a, b)
            if i is None:
                return
            n = len(v)
            x, y = v[i], v[(i + 1) % n]
            if n == 3:
                c = v[(i + 2) % 3]
                replace(g, [[x, m, c], [m, y, c]])
                return
            c, d = v[(i + 2) % 4], v[(i + 3) % 4]
            if kind[g] == skip_kind or steps >= max_run:
                replace(g, [[x, m, d], [m, y, c, d]])
                return
            m2 = mid(c, d)
            replace(g, [[x, m, m2, d], [m, y, c, m2]])
            a, b, m = c, d, m2
            steps += 1

    for f in range(len(faces)):
        push(f)
    cuts = 0
    while heap and cuts < max_cuts:
        neg, f = heapq.heappop(heap)
        if not alive[f]:
            continue
        v = faces[f]
        err = -neg
        nodes = node_lists.get(f)
        if len(v) == 3:
            lengths = [np.hypot(xy[v[(i + 1) % 3]][0] - xy[v[i]][0],
                                xy[v[(i + 1) % 3]][1] - xy[v[i]][1]) for i in range(3)]
            i = int(np.argmax(lengths))
            if lengths[i] < 2.0 * min_edge_m:
                continue
            x, y, c = v[i], v[(i + 1) % 3], v[(i + 2) % 3]
            m = mid(x, y)
            replace(f, [[x, m, c], [m, y, c]])
            carry(x, y, m)
            cuts += 1
            continue
        best = None
        for i in (0, 1):
            x, y, c, d = v[i], v[i + 1], v[(i + 2) % 4], v[(i + 3) % 4]
            l1 = np.hypot(xy[y][0] - xy[x][0], xy[y][1] - xy[x][1])
            l2 = np.hypot(xy[d][0] - xy[c][0], xy[d][1] - xy[c][1])
            if min(l1, l2) < 2.0 * min_edge_m:
                continue
            p1 = (0.5 * (xy[x][0] + xy[y][0]), 0.5 * (xy[x][1] + xy[y][1]))
            p2 = (0.5 * (xy[c][0] + xy[d][0]), 0.5 * (xy[c][1] + xy[d][1]))
            h = height_at(np.array([p1, p2]))
            ring_a = [xy[x], p1, p2, xy[d]]
            ring_b = [p1, xy[y], xy[c], p2]
            za = [z[x], h[0], h[1], z[d]]
            zb = [h[0], z[y], z[c], h[1]]
            if nodes is not None and len(nodes):
                pts = nodes_xy[nodes]
                in_a = inside(pts, ring_a)
                score = max(error_of(None, nodes[in_a], ring_a, za),
                            error_of(None, nodes[~in_a], ring_b, zb))
            else:
                score = 0.0
            if best is None or score < best[0]:
                best = (score, i)
        if best is None or best[0] >= err - 0.05:
            continue  # no cut here gets it any closer
        i = best[1]
        x, y, c, d = v[i], v[i + 1], v[(i + 2) % 4], v[(i + 3) % 4]
        m1, m2 = mid(x, y), mid(c, d)
        replace(f, [[x, m1, m2, d], [m1, y, c, m2]])
        carry(x, y, m1)
        carry(c, d, m2)
        cuts += 1

    keep = [f for f in range(len(faces)) if alive[f]]
    sizes_out = np.array([len(faces[f]) for f in keep], dtype=np.int64)
    loops_out = np.array([i for f in keep for i in faces[f]], dtype=np.int64)
    return (
        np.array(xy, dtype=np.float64), sizes_out, loops_out,
        np.array([kind[f] for f in keep], dtype=np.int64),
        np.array([part[f] for f in keep], dtype=np.int64),
        np.array([code[f] for f in keep], dtype=np.int64),
        cuts,
    )


# A triangle less than this high over its long edge is a sliver: its normal
# points anywhere, and whatever is shrinkwrapped onto it is thrown sideways.
SLIVER_HEIGHT_M = 0.05


def _merge_slivers(local_xy, sizes, loops, face_kind, face_part, face_code,
                   min_height=SLIVER_HEIGHT_M):
    """Every sliver triangle folded into the face across its long edge.

    The sliver's third corner is added to that face's outline, which then
    covers both, and is filled again if it is no longer a triangle or a convex
    quad. Nothing moves and no vertex is lost, so no crack opens.

    Returns the faces as `_loop_cuts` does, and how many slivers went.
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


# The steepest a ditch bank is cut: one metre down per metre across, 45°.
BANK_SLOPE = 1.0


def _gentle_banks(heights, loops, sizes, face_kind, xy, wet_kind, slope=BANK_SLOPE):
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

    # A face sharing no edge with any other lies on top of the sheet, not in
    # it: it goes.
    a, b, is_open = open_loops(sizes, loops)
    face_of_loop = np.repeat(np.arange(len(sizes)), sizes)
    shared = np.bincount(face_of_loop, weights=~is_open, minlength=len(sizes))
    stray = shared == 0
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


def _area(coords) -> float:
    c = np.asarray(coords, dtype=np.float64)
    return 0.5 * abs(float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])))


def _cap_levels(nb: int, size: float, centre, focus_m: float, near_m: float, g: float):
    """Per block of `size`, whether its distance allows a block that big."""
    mids = (np.arange(nb) + 0.5) * size
    dx = np.abs(mids[None, :] - centre[0])
    dy = np.abs(mids[:, None] - centre[1])
    # The block's nearest point, so a big block never reaches into the middle.
    d = np.maximum(np.maximum(dx, dy) - 0.5 * size, 0.0)
    cap = np.where(d <= 0.5 * focus_m, near_m, np.where(d <= focus_m, 2 * near_m, FAR_CELLS * near_m))
    return size <= cap + 1e-9


def _blocks(owner, heights, g, centre, *, focus_m, near_m, base_level, tolerance_m):
    """The core squares as blocks: a level per survey grid square, balanced.

    `owner` is the region index per square of the survey grid (-1 where no
    grid face is), `heights` the grid's node heights, `g` its spacing. A
    block of level ``l`` is ``2**l`` survey squares across; the grid the rows
    meet is level `base_level`. A block a level up merges four owned by the
    same region where the distance allows a block that big and every survey
    node inside lies within `tolerance_m` of the merged square. A base block
    that misses the survey by more than that is split in four, unless a row
    or fill face is beside it: those carry a vertex at every base grid node
    and no more. No block is more than one level bigger than the one beside
    it, so a side carries at most one extra vertex.
    """
    rows, cols = owner.shape
    top = int(max(0, np.floor(np.log2(max(FAR_CELLS * near_m / g, 1.0)) + 1e-9)))
    finest = max(0, base_level - 1)
    free = owner < 0
    padded = np.pad(free, 1, constant_values=False)
    beside_free = (
        free | padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    )
    ok = [owner >= 0]
    for level in range(1, top + 1):
        m = 2 ** level
        nb_r, nb_c = rows // m, cols // m
        if nb_r < 1 or nb_c < 1:
            break
        prev = ok[-1]
        children = (
            prev[0: 2 * nb_r: 2, 0: 2 * nb_c: 2] & prev[1: 2 * nb_r: 2, 0: 2 * nb_c: 2]
            & prev[0: 2 * nb_r: 2, 1: 2 * nb_c: 2] & prev[1: 2 * nb_r: 2, 1: 2 * nb_c: 2]
        )
        view = owner[: nb_r * m, : nb_c * m].reshape(nb_r, m, nb_c, m)
        same = (view == view[:, :1, :, :1]).all(axis=(1, 3))
        if level <= finest:
            ok.append(children & same)
            continue
        # Every survey grid node inside the block, against the merged square.
        windows = np.lib.stride_tricks.sliding_window_view(heights, (m + 1, m + 1))[::m, ::m]
        windows = windows[:nb_r, :nb_c]
        u = np.arange(m + 1) / m
        z00 = windows[:, :, 0, 0][..., None, None]
        z10 = windows[:, :, 0, -1][..., None, None]
        z01 = windows[:, :, -1, 0][..., None, None]
        z11 = windows[:, :, -1, -1][..., None, None]
        uu, vv = np.meshgrid(u, u)
        plane = (1 - vv) * ((1 - uu) * z00 + uu * z10) + vv * ((1 - uu) * z01 + uu * z11)
        dev = np.nanmax(np.abs(windows - plane), axis=(2, 3))
        flat = np.isfinite(dev) & (dev <= tolerance_m)
        if level <= base_level:
            edge = beside_free[: nb_r * m, : nb_c * m].reshape(nb_r, m, nb_c, m).any(axis=(1, 3))
            ok.append(children & same & (flat | edge))
            continue
        cap = _cap_levels(max(nb_r, nb_c), m * g, centre, focus_m, near_m, g)[:nb_r, :nb_c]
        ok.append(children & same & cap & flat)

    def levels():
        out = np.where(owner >= 0, 0, -1)
        jj, ii = np.indices(owner.shape)
        for level in range(1, len(ok)):
            m = 2 ** level
            nb_r, nb_c = ok[level].shape
            inside = (jj < nb_r * m) & (ii < nb_c * m)
            hit = np.zeros(owner.shape, dtype=bool)
            hit[inside] = ok[level][jj[inside] // m, ii[inside] // m]
            out[hit] = level
        return out

    block_level = levels()
    for _ in range(4 * len(ok) + 4):
        # Squares with no grid face count as the base level: the faces there
        # carry a vertex at every base grid node along the edge of the grid.
        seen = np.where(block_level < 0, base_level, block_level)
        padded = np.pad(seen, 1, constant_values=10 ** 6)
        neighbour = np.minimum.reduce([
            padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:],
        ])
        too_big = (block_level > neighbour + 1) & (block_level >= 1)
        if not too_big.any():
            break
        jj, ii = np.nonzero(too_big)
        for j, i in zip(jj, ii):
            level = int(block_level[j, i])
            m = 2 ** level
            ok[level][j // m, i // m] = False
        block_level = levels()
    return block_level


def quad_terrain(terrain, pieces, bodies, *, codes_wet, tolerance_m, focus_m, depth_m,
                 paved_codes, narrow_codes=(), near_m: float = NEAR_CELL_M):
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
        _distinct_corners,
        _fill,
        _fill_gaps,
        _mesh_region,
        _orient,
        _water_heights,
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
    spacing = _spacing(float(focus_m), near_m)
    # Each region is meshed on its own, so a gap between two of them -- left
    # where a hairpin was trimmed, say -- would be a hole. It goes to the
    # neighbour it shares most outline with.
    filled = _fill_gaps(np.array([p for p, _ in pieces], dtype=object), frame)
    pieces = [(part, code) for polygon, (_, code) in zip(filled, pieces)
              for part in _parts(polygon)]
    parts, codes = [], []
    for polygon, code in pieces:
        polygon = _with_stations(polygon, spacing, centre, (width, height))
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
    refilled = 0
    t = time.time()
    for index, (region, kind) in enumerate(parts):
        before = taken.copy()
        try:
            out, cells, _, _ = _mesh_region(
                region, kind, frame, outer, xs, ys, taken,
                row_m=rows[index], margin_m=margin_m, across=across.get(index, ()),
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
        faces.extend((c, p, kind, codes[index]) for c, p in out)
        if len(cells):
            owner[cells[:, 1], cells[:, 0]] = index
    timings["regions_meshed_s"] = round(time.time() - t, 1)

    # ---- the grid squares, merged into blocks away from the middle --------
    t = time.time()
    grid_z = np.asarray(
        terrain.measured if terrain.measured is not None else terrain.heights,
        dtype=np.float64,
    )
    # On the survey grid itself, so a base square can be split where it has
    # to be. Only for a power of two: otherwise the base grid is the finest.
    split = k if (k & (k - 1)) == 0 else 1
    base_level = int(np.log2(split))
    owner_fine = np.repeat(np.repeat(owner, split, axis=0), split, axis=1)
    if split == 1:
        xs_fine, ys_fine, g_fine = xs, ys, near_m
        z_fine = grid_z[::k, ::k][: len(ys), : len(xs)]
    else:
        xs_fine, ys_fine, g_fine = xs_full, ys_full, g
        z_fine = grid_z
    block_level = _blocks(
        owner_fine, z_fine, g_fine, centre, focus_m=float(focus_m), near_m=near_m,
        base_level=base_level, tolerance_m=float(tolerance_m),
    )
    owner = owner_fine
    blocks = []  # (level, I, J)
    for level in range(int(block_level.max()) + 1 if block_level.size else 0):
        m = 2 ** level
        jj, ii = np.nonzero(block_level == level)
        first = (jj % m == 0) & (ii % m == 0)
        blocks.extend((level, int(i // m), int(j // m)) for j, i in zip(jj[first], ii[first]))

    def corner_keys(points):
        k = np.rint(np.asarray(points) * 1000.0).astype(np.int64)
        return k[:, 0] * (1 << 31) + k[:, 1]

    region_corners = (
        np.vstack([c for c, _, _, _ in faces]) if faces else np.zeros((0, 2))
    )
    block_arr = np.array(blocks, dtype=np.int64).reshape(-1, 3)
    m_arr = 2 ** block_arr[:, 0]
    i0, j0 = block_arr[:, 1] * m_arr, block_arr[:, 2] * m_arr
    i1, j1 = i0 + m_arr, j0 + m_arr
    block_corners = np.stack([
        np.column_stack([xs_fine[i0], ys_fine[j0]]), np.column_stack([xs_fine[i1], ys_fine[j0]]),
        np.column_stack([xs_fine[i1], ys_fine[j1]]), np.column_stack([xs_fine[i0], ys_fine[j1]]),
    ], axis=1) if len(block_arr) else np.zeros((0, 4, 2))
    known = np.unique(np.concatenate([
        corner_keys(region_corners), corner_keys(block_corners.reshape(-1, 2)),
    ]))
    hanging_faces = 0
    levels_arr = block_arr[:, 0] if len(block_arr) else np.zeros(0, dtype=np.int64)
    for level in np.unique(levels_arr):
        chosen = np.flatnonzero(levels_arr == level)
        m = 2 ** int(level)
        corners_l = block_corners[chosen]  # (B, 4, 2)
        owners = owner[j0[chosen], i0[chosen]]
        if m == 1:
            faces.extend(
                (corners_l[b], CORE, parts[o][1], codes[o]) for b, o in enumerate(owners)
            )
            continue
        # Every grid node along each side, and which of them a face beside
        # the block has as a corner.
        frac = (np.arange(1, m) / m)[None, None, :, None]
        a = corners_l[:, :, None, :]
        c = np.roll(corners_l, -1, axis=1)[:, :, None, :]
        between = a + (c - a) * frac  # (B, 4, m-1, 2)
        present = np.isin(corner_keys(between.reshape(-1, 2)), known).reshape(
            len(chosen), 4, m - 1
        )
        for b, o in enumerate(owners):
            kind, code = parts[o][1], codes[o]
            if not present[b].any():
                faces.append((corners_l[b], CORE, kind, code))
                continue
            ring = []
            for side in range(4):
                ring.append(corners_l[b, side][None, :])
                ring.append(between[b, side][present[b, side]])
            hanging_faces += 1
            faces.extend(
                (q, CORE, kind, code) for q in _fill(shapely.Polygon(np.vstack(ring)))
            )
    timings["grid_s"] = round(time.time() - t, 1)

    # ---- assemble, one vertex per position ----------------------------------
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
    local_xy, sizes, loops, welded = _weld(local_xy, sizes, loops)
    keep = _distinct_corners(loops, sizes) & (sizes >= 3)
    if not keep.all():
        loops = loops[np.repeat(keep, sizes)]
        sizes, face_kind, face_part, face_code = (
            sizes[keep], face_kind[keep], face_part[keep], face_code[keep]
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

    # ---- loops cut where the faces miss the survey -------------------------
    t = time.time()
    gx, gy = np.meshgrid(xs_full, ys_full)
    local_xy, sizes, loops, face_kind, face_part, face_code, cuts = _loop_cuts(
        local_xy, sizes, loops, face_kind, face_part, face_code,
        height_at=lambda p: np.asarray(
            terrain.sample(p[:, 0] + shift[0], p[:, 1] + shift[1]), dtype=np.float64
        ),
        nodes_xy=np.column_stack([gx.ravel(), gy.ravel()]), nodes_z=grid_z.ravel(),
        tolerance_m=float(tolerance_m), skip_kind=WATER,
    )
    timings["loop_cuts_s"] = round(time.time() - t, 1)
    local_xy, sizes, loops, face_kind, face_part, face_code, slivers = _merge_slivers(
        local_xy, sizes, loops, face_kind, face_part, face_code,
    )

    xy = local_xy + shift
    # Every vertex reads the survey grid, between its nodes as well: the 0.5 m
    # survey itself, read at an outline point on a retaining wall, made faces
    # standing nearly upright, slivers seen from above, on which whatever is
    # draped picks one sliver and a check the next. The grid is never steeper
    # than it is between two of its nodes, and the loop cuts already hold the
    # faces to the survey.
    heights = np.asarray(terrain.sample(xy[:, 0], xy[:, 1]), dtype=np.float64)
    heights = _water_heights(heights, loops, sizes, face_kind, xy, bodies, shift, depth_m)
    heights = _gentle_banks(heights, loops, sizes, face_kind, local_xy, WATER)
    vertices = np.column_stack([xy, heights])
    sizes, loops = _orient(vertices, sizes, loops, face_part, face_kind, terrain)

    levels_used = {
        f"{(2 ** level) * g_fine:g} m": int((block_arr[:, 0] == level).sum())
        for level in np.unique(block_arr[:, 0])
    } if len(block_arr) else {}
    stats = {
        "mesher": "quads",
        "grid_blocks": levels_used,
        "grid_blocks_split_at_a_finer_neighbour": int(hanging_faces),
        "cross_rungs": int(sum(len(v) for v in across.values())),
        "regions_meshed_without_rows": int(failed),
        "rungs_divided": int(divided),
        "regions_filled_whole": int(refilled),
        "loops_cut_to_the_survey": int(cuts),
        "slivers_merged": int(slivers),
        "vertices_welded": int(welded),
        "holes_patched_or_strays_dropped": int(patched),
        "water_faces": int((face_kind == WATER).sum()),
        **timings,
        "mesher_s": round(time.time() - started, 1),
    }
    return vertices, sizes.astype(np.int64), loops, face_code, face_part, stats


def _parts(geometry):
    from .terrain_mesh import _parts as parts

    return parts(geometry)


__all__ = ["NEAR_CELL_M", "PAVED_ROW_M", "quad_terrain"]
