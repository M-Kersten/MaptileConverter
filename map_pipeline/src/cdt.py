"""Constrained Delaunay triangulation, so mesh edges land on real features.

A grid mesh, and the bisection mesh that replaced it, both put their edges
where the algorithm wanted them: on grid lines and grid diagonals. Neither can
put an edge along a canal bank, because the bank does not run along a grid
diagonal. In Blender that is the difference between selecting the edge loop of
a quay and dragging it, and hand-picking a staircase of triangles that happens
to be dense near the water.

This triangulates a set of points together with a set of segments that must
survive as edges -- the outlines of roads, water, land cover and buildings, and
contour lines off the height model. Every constraint comes out as an edge of
the mesh, and everything else is Delaunay, which is the triangulation that
avoids thin triangles wherever it is free to choose.

The three pieces are standard, and each is here because the previous one cannot
do the job alone:

* Bowyer-Watson insertion builds a Delaunay triangulation one point at a time.
  It cannot honour a segment: an edge it wants to flip away will be flipped
  away.
* Constraint recovery puts a required edge back by deleting the triangles the
  segment crosses and retriangulating the two pockets left behind. This is what
  makes an edge follow the bank.
* A flood fill from outside then drops the triangles that are not in the
  domain -- outside the bbox, or inside a hole -- because insertion happily
  triangulates the convex hull and the domain is not convex.

Everything is float64 in local metres. Callers pass RD coordinates through
`origin`, which matters: RD northings are around 450 000, and an orientation
test on numbers that size loses most of the precision that decides whether a
sliver triangle is wound one way or the other.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

LOG = logging.getLogger(__name__)

# Points closer together than this are the same point. BGT boundaries are
# surveyed to the centimetre and adjacent polygons repeat their shared corners
# exactly, so this only ever merges what was meant to be one point.
WELD_M = 0.005

# How far out the enclosing super-triangle sits, as a multiple of the data's
# own half-span. See triangulate() for why this is not a small number.
SUPER_TRIANGLE_SPAN = 1000.0

# Below this a triangle is treated as having no area at all. Scaled for local
# metres, where a real terrain triangle is at worst a few square centimetres.
AREA_EPS = 1e-12


@dataclass
class Triangulation:
    """The mesh, plus which of its edges were required to be there."""

    points: np.ndarray  # (n, 2) float64, local metres
    triangles: np.ndarray  # (m, 3) int32, counter-clockwise
    constrained: set  # frozensets of two point indices
    origin: tuple[float, float] = (0.0, 0.0)
    # Constraints dropped because they crossed one already in place.
    skipped: int = 0

    @property
    def point_count(self) -> int:
        return int(len(self.points))

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    def world_points(self) -> np.ndarray:
        """Back into the coordinates the caller handed in."""
        return self.points + np.asarray(self.origin, dtype=np.float64)

    def edges(self) -> set:
        out = set()
        for a, b, c in self.triangles:
            out.add(frozenset((int(a), int(b))))
            out.add(frozenset((int(b), int(c))))
            out.add(frozenset((int(c), int(a))))
        return out

    def missing_constraints(self) -> set:
        """Constraints that did not survive. Should always be empty."""
        return self.constrained - self.edges()


def _orient(ax, ay, bx, by, cx, cy) -> float:
    """Twice the signed area of abc. Positive when counter-clockwise."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _in_circle(ax, ay, bx, by, cx, cy, dx, dy) -> float:
    """Positive when d is inside the circumcircle of a counter-clockwise abc."""
    adx, ady = ax - dx, ay - dy
    bdx, bdy = bx - dx, by - dy
    cdx, cdy = cx - dx, cy - dy
    return (
        (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
        - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
        + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
    )


def _hilbert_order(points: np.ndarray, bits: int = 16) -> np.ndarray:
    """Sort points along a Hilbert curve.

    Insertion order decides everything about the cost of this algorithm. Points
    in file order send the point-location walk across the whole mesh for every
    insertion; points in Hilbert order leave the next point next to the last
    one, so the walk is a handful of steps whatever the mesh size.
    """
    lo = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - lo, 1e-12)
    side = (1 << bits) - 1
    x = ((points[:, 0] - lo[0]) / span[0] * side).astype(np.int64)
    y = ((points[:, 1] - lo[1]) / span[1] * side).astype(np.int64)

    rx = np.zeros_like(x)
    ry = np.zeros_like(y)
    d = np.zeros_like(x)
    s = 1 << (bits - 1)
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # Rotate the quadrant so the curve stays connected.
        swap = ry == 0
        flip = swap & (rx == 1)
        x_f = np.where(flip, s - 1 - x, x)
        y_f = np.where(flip, s - 1 - y, y)
        x_new = np.where(swap, y_f, x_f)
        y_new = np.where(swap, x_f, y_f)
        x, y = x_new, y_new
        s >>= 1
    return np.argsort(d, kind="stable")


class _Mesh:
    """Triangles with neighbour links, mutable, indices into one point array.

    ``nbr[t][i]`` is the triangle across the edge opposite vertex ``i`` of
    triangle ``t``, or -1. ``vt[v]`` is any live triangle touching vertex
    ``v``. Both exist so that every operation here costs the size of the patch
    it changes rather than the size of the mesh: with tens of thousands of
    constraints to recover, anything that walks the whole mesh once per
    constraint is not an option.
    """

    __slots__ = (
        "px", "py", "tri", "nbr", "dead", "vt", "constrained", "real", "_free"
    )

    def __init__(self, px: list, py: list, real: int = 0) -> None:
        self.px = px
        self.py = py
        # Vertices from here up are the corners of the enclosing super-triangle
        # rather than real points.
        self.real = real
        self.tri: list[list[int]] = []
        self.nbr: list[list[int]] = []
        self.dead: list[bool] = []
        self.vt: dict[int, int] = {}
        self.constrained: set = set()
        self._free: list[int] = []

    def add(self, a: int, b: int, c: int) -> int:
        if self._free:
            t = self._free.pop()
            self.tri[t] = [a, b, c]
            self.nbr[t] = [-1, -1, -1]
            self.dead[t] = False
        else:
            self.tri.append([a, b, c])
            self.nbr.append([-1, -1, -1])
            self.dead.append(False)
            t = len(self.tri) - 1
        self.vt[a] = self.vt[b] = self.vt[c] = t
        return t

    def kill(self, t: int) -> None:
        self.dead[t] = True
        self._free.append(t)

    def link(self, t: int, i: int, u: int, j: int) -> None:
        if t >= 0:
            self.nbr[t][i] = u
        if u >= 0:
            self.nbr[u][j] = t

    def orient(self, a: int, b: int, c: int) -> float:
        px, py = self.px, self.py
        return _orient(px[a], py[a], px[b], py[b], px[c], py[c])

    def in_circle(self, t: int, d: int) -> float:
        a, b, c = self.tri[t]
        px, py = self.px, self.py
        return _in_circle(px[a], py[a], px[b], py[b], px[c], py[c], px[d], py[d])


def _edges_of(tri: list[int]):
    """The three edges, each directed as the counter-clockwise triangle sees it.

    Edge ``i`` is the one opposite vertex ``i``, which is the indexing the
    neighbour array uses.
    """
    a, b, c = tri
    return ((b, c), (c, a), (a, b))


def _third(tri: list[int], u: int, v: int) -> int:
    """The vertex of `tri` that is neither u nor v."""
    for w in tri:
        if w != u and w != v:
            return w
    raise RuntimeError(f"triangle {tri} is not incident to both {u} and {v}")


def _retriangulate(mesh: _Mesh, doomed: list[int], replacements: list) -> list[int]:
    """Swap a patch of triangles for another covering the same ground.

    The patch boundary is recorded before anything is deleted, so the new
    triangles can be stitched back to the mesh without touching -- or even
    looking at -- the rest of it. Every operation in this module that changes
    topology goes through here, which is the only reason the adjacency stays
    consistent through both point insertion and constraint recovery.
    """
    inside = set(doomed)
    outside: dict = {}
    for t in doomed:
        for i, (u, v) in enumerate(_edges_of(mesh.tri[t])):
            n = mesh.nbr[t][i]
            if n >= 0 and n not in inside:
                outside[(u, v)] = (n, mesh.nbr[n].index(t))
    for t in doomed:
        mesh.kill(t)

    made: list[int] = []
    edge_map: dict = {}
    for a, b, c in replacements:
        t = mesh.add(a, b, c)
        made.append(t)
        for i, edge in enumerate(_edges_of(mesh.tri[t])):
            edge_map[edge] = (t, i)

    for (u, v), (t, i) in edge_map.items():
        partner = edge_map.get((v, u))
        if partner is not None:
            mesh.nbr[t][i] = partner[0]
            continue
        # The patch boundary keeps the direction the old triangle gave it,
        # because both cover the same side of that edge.
        beyond = outside.get((u, v))
        mesh.nbr[t][i] = -1 if beyond is None else beyond[0]
        if beyond is not None:
            mesh.nbr[beyond[0]][beyond[1]] = t
    return made


def _fan(mesh: _Mesh, v: int):
    """Every live triangle touching `v`, walked round rather than searched."""
    start = mesh.vt.get(v, -1)
    if start < 0 or mesh.dead[start]:
        return
    t = start
    for _ in range(4096):
        yield t
        i = mesh.tri[t].index(v)
        # Rotating one way round v means crossing the edge opposite the vertex
        # two steps on, which is the edge from v to its counter-clockwise
        # neighbour in this triangle.
        t = mesh.nbr[t][(i + 2) % 3]
        if t < 0 or t == start:
            return
    raise RuntimeError(f"the triangles around vertex {v} do not close")


def _edge_exists(mesh: _Mesh, a: int, b: int) -> bool:
    return any(b in mesh.tri[t] for t in _fan(mesh, a))


def _locate(mesh: _Mesh, start: int, p: int) -> int:
    """Walk from `start` to the triangle containing point `p`."""
    t = start
    if t < 0 or mesh.dead[t]:
        t = next(i for i in range(len(mesh.tri)) if not mesh.dead[i])
    for _ in range(1_000_000):
        a, b, c = mesh.tri[t]
        if mesh.orient(b, c, p) < 0:
            nxt = mesh.nbr[t][0]
        elif mesh.orient(c, a, p) < 0:
            nxt = mesh.nbr[t][1]
        elif mesh.orient(a, b, p) < 0:
            nxt = mesh.nbr[t][2]
        else:
            return t
        if nxt < 0:
            return t
        t = nxt
    raise RuntimeError("point location did not terminate")


def _insert_point(mesh: _Mesh, p: int, hint: int) -> int:
    """Bowyer-Watson: delete every triangle p invalidates, then fan into the hole."""
    start = _locate(mesh, hint, p)

    cavity = [start]
    seen = {start}
    stack = [start]
    while stack:
        t = stack.pop()
        for i, edge in enumerate(_edges_of(mesh.tri[t])):
            n = mesh.nbr[t][i]
            if n < 0 or n in seen:
                continue
            # A constrained edge stops the cavity. Crossing it would delete a
            # triangle on the far side of a boundary that has to survive, and
            # the fan would then bridge straight over the constraint.
            if frozenset(edge) in mesh.constrained:
                continue
            if mesh.in_circle(n, p) > 0:
                seen.add(n)
                cavity.append(n)
                stack.append(n)

    inside = set(cavity)
    fan = []
    for t in cavity:
        for i, (u, v) in enumerate(_edges_of(mesh.tri[t])):
            n = mesh.nbr[t][i]
            if n < 0 or n not in inside:
                fan.append((p, u, v))
    made = _retriangulate(mesh, cavity, fan)
    return made[-1] if made else start


def _polygon_fill(mesh: _Mesh, poly: list[int], out: list) -> None:
    """Triangulate a pocket left by a removed constraint crossing.

    `poly` runs from one end of the constrained segment to the other along one
    side. The Delaunay point of the chain becomes the apex and the two halves
    recurse, which keeps the pocket Delaunay without re-running insertion in it.
    """
    if len(poly) < 3:
        return
    if len(poly) == 3:
        out.append((poly[0], poly[1], poly[2]))
        return
    a, b = poly[0], poly[-1]
    best = 1
    for i in range(2, len(poly) - 1):
        if _in_circumcircle(mesh, a, b, poly[best], poly[i]):
            best = i
    out.append((a, poly[best], b))
    _polygon_fill(mesh, poly[: best + 1], out)
    _polygon_fill(mesh, poly[best:], out)


def _in_circumcircle(mesh: _Mesh, a: int, b: int, c: int, d: int) -> bool:
    """Is d inside the circle through a, b and c, whichever way abc winds?

    The raw determinant only answers that question for a counter-clockwise
    abc; fed a clockwise one it answers the opposite. A pocket left by a
    removed constraint is routinely not convex, so the triangle formed by the
    two ends of its chain and a candidate apex winds either way, and taking
    the sign at face value picks an apex outside the pocket.
    """
    turn = mesh.orient(a, b, c)
    if turn == 0.0:
        return False
    inside = _in_circle(
        mesh.px[a], mesh.py[a],
        mesh.px[b], mesh.py[b],
        mesh.px[c], mesh.py[c],
        mesh.px[d], mesh.py[d],
    )
    return inside > 0 if turn > 0 else inside < 0


def _crossed_triangles(mesh: _Mesh, a: int, b: int) -> tuple[list, list, list]:
    """Triangles the segment a-b passes through, and the chains either side.

    Walks the segment across the mesh, tracking the edge (u, v) it is currently
    passing through with u to its left and v to its right. Each step hops to
    the neighbour across that edge; the far vertex of that neighbour replaces
    whichever of u or v is on the side the segment did not pass.
    """
    start = left0 = right0 = -1
    for t in _fan(mesh, a):
        i = mesh.tri[t].index(a)
        u, v = mesh.tri[t][(i + 1) % 3], mesh.tri[t][(i + 2) % 3]
        # A vertex sitting exactly on the segment cannot be walked past: the
        # segment has to be cut there and each half constrained separately.
        # Real data does this constantly, because BGT outlines share corners
        # and plenty of them are collinear.
        for w in (u, v):
            if _between(mesh, a, b, w):
                return w, None, None
        # In a counter-clockwise triangle (a, u, v) the vertices run clockwise
        # as seen from a, so u sits to the *right* of a ray leaving a and v to
        # its left. The segment leaves through edge (u, v) exactly when b falls
        # in that wedge.
        if mesh.orient(a, b, u) < 0 < mesh.orient(a, b, v):
            start, left0, right0 = t, v, u
            break
    if start < 0:
        raise _Unplaceable(f"no triangle at vertex {a} faces {b}")

    crossed = [start]
    left, right = [a, left0], [a, right0]
    t, u, v = start, left0, right0
    for _ in range(len(mesh.tri) + 3):
        if frozenset((u, v)) in mesh.constrained:
            raise _CrossingConstraints(a, b, u, v)
        n = mesh.nbr[t][mesh.tri[t].index(_third(mesh.tri[t], u, v))]
        if n < 0:
            raise _Unplaceable("a constraint ran off the edge of the triangulation")
        far = _third(mesh.tri[n], u, v)
        if far == b:
            crossed.append(n)
            break
        side = mesh.orient(a, b, far)
        if side == 0.0:
            # Straight through a vertex again; cut and retry both halves.
            return far, None, None
        crossed.append(n)
        if side > 0:
            left.append(far)
            u = far
        else:
            right.append(far)
            v = far
        t = n
    else:
        raise _Unplaceable("the walk along a constraint did not reach its end")

    left.append(b)
    right.append(b)
    return crossed, left, right


def _between(mesh: _Mesh, a: int, b: int, w: int) -> bool:
    """True when w lies on the open segment a-b."""
    if w == a or w == b:
        return False
    if mesh.orient(a, b, w) != 0.0:
        return False
    px, py = mesh.px, mesh.py
    dx, dy = px[b] - px[a], py[b] - py[a]
    t = (px[w] - px[a]) * dx + (py[w] - py[a]) * dy
    return 0.0 < t < dx * dx + dy * dy


class _ConstraintProblem(RuntimeError):
    """A required edge could not be put in. Skippable when the caller allows."""


class _Unplaceable(_ConstraintProblem):
    """The walk could not start or could not finish.

    Degenerate geometry rather than a logic error: a vertex sitting exactly on
    the line of a constraint but outside it leaves no triangle whose wedge
    strictly contains the far end.
    """


class _CrossingConstraints(_ConstraintProblem):
    """Two required edges cross, which no triangulation can satisfy."""

    def __init__(self, a, b, u, v):
        super().__init__(
            f"constraint {a}-{b} crosses constraint {u}-{v}. Two segments that "
            f"cross cannot both be edges; split them at their intersection "
            f"first (src/breaklines.py does this)."
        )


def _apply_constraints(mesh: _Mesh, segments: np.ndarray, skip_crossing: bool) -> int:
    """Force every segment to appear as an edge. Returns how many were skipped.

    With `skip_crossing`, a segment that crosses one already recovered is
    dropped rather than raised over. Breakline input is split at its own
    crossings first, but a cut landing a hair from an endpoint survives that,
    and a whole run is not worth losing over a few contour segments meeting at
    a millimetre.
    """
    skipped = 0
    pending = [(int(a), int(b)) for a, b in segments]
    while pending:
        a, b = pending.pop()
        if a == b:
            continue
        key = frozenset((a, b))
        if key in mesh.constrained:
            continue
        if _edge_exists(mesh, a, b):
            mesh.constrained.add(key)
            continue
        try:
            crossed, left, right = _crossed_triangles(mesh, a, b)
        except _ConstraintProblem:
            if not skip_crossing:
                raise
            skipped += 1
            continue
        if left is None:
            # The segment runs through vertex `crossed`; constrain both halves.
            pending.append((a, crossed))
            pending.append((crossed, b))
            continue
        pocket: list = []
        # Both chains run a->b, so closed by the segment they wind opposite
        # ways; winding is normalised below. What matters is that the two
        # pockets tile the crossed strip exactly once.
        _polygon_fill(mesh, left, pocket)
        _polygon_fill(mesh, right, pocket)
        wound = []
        for x, y, z in pocket:
            area = mesh.orient(x, y, z)
            if abs(area) <= AREA_EPS:
                continue
            wound.append((x, z, y) if area < 0 else (x, y, z))
        _retriangulate(mesh, crossed, wound)
        mesh.constrained.add(key)
    return skipped


def triangulate(
    points: np.ndarray,
    segments: np.ndarray | None = None,
    *,
    origin: tuple[float, float] | None = None,
    skip_crossing: bool = False,
) -> Triangulation:
    """Delaunay triangulation of `points` in which every `segment` is an edge.

    `points` is (n, 2) in the caller's units; `segments` is (m, 2) of indices
    into it. The result fills the convex hull -- trimming it to a domain is the
    caller's job, because only the caller knows which side of a constraint is
    inside.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (n, 2), got {points.shape}")
    if len(points) < 3:
        raise ValueError(f"need at least three points, got {len(points)}")

    if origin is None:
        origin = tuple(points.mean(axis=0))
    local = points - np.asarray(origin, dtype=np.float64)

    px = list(local[:, 0])
    py = list(local[:, 1])
    mesh = _Mesh(px, py, real=len(px))

    # The super-triangle has to be far enough out to be invisible to the
    # result, and "far enough" is much further than it looks. Its corners take
    # part in the Delaunay property like any other point, so a close one makes
    # real triangles along the hull genuinely non-Delaunay and they never get
    # built: at 4x the span this lost 5 of 402 triangles and 1.3% of the area,
    # silently, along the edge of the bbox. Measured exact from about 100x.
    span = float(np.abs(local).max()) * SUPER_TRIANGLE_SPAN + 1.0
    real = len(px)
    px.extend([-3.0 * span, 3.0 * span, 0.0])
    py.extend([-2.0 * span, -2.0 * span, 3.0 * span])
    root = mesh.add(real, real + 1, real + 2)

    hint = root
    for index in _hilbert_order(local):
        hint = _insert_point(mesh, int(index), hint)

    skipped = 0
    if segments is not None and len(segments):
        skipped = _apply_constraints(
            mesh, np.asarray(segments, dtype=np.int64), skip_crossing
        )
        if skipped:
            LOG.warning(
                "%d of %d constraints crossed another and were dropped",
                skipped,
                len(segments),
            )

    kept = []
    for t in range(len(mesh.tri)):
        if mesh.dead[t]:
            continue
        a, b, c = mesh.tri[t]
        if a >= real or b >= real or c >= real:
            continue
        if mesh.orient(a, b, c) <= AREA_EPS:
            continue
        kept.append((a, b, c))

    return Triangulation(
        points=local[:real],
        triangles=np.asarray(kept, dtype=np.int32).reshape(-1, 3),
        constrained={frozenset((int(x), int(y))) for x, y in mesh.constrained},
        origin=(float(origin[0]), float(origin[1])),
        skipped=skipped,
    )


def weld(points: np.ndarray, tolerance: float = WELD_M) -> tuple[np.ndarray, np.ndarray]:
    """Merge points closer than `tolerance`, returning survivors and a map.

    Two BGT polygons that share a boundary repeat its corners exactly, and a
    triangulator handed the same point twice makes a zero-area triangle and an
    adjacency that then does not close.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points.reshape(0, 2), np.zeros(0, dtype=np.int64)
    quantised = np.round(points / max(tolerance, 1e-9)).astype(np.int64)
    _, first, inverse = np.unique(
        quantised, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(np.argsort(first))
    return points[np.sort(first)], order[inverse.ravel()]


__all__ = [
    "AREA_EPS",
    "SUPER_TRIANGLE_SPAN",
    "WELD_M",
    "Triangulation",
    "triangulate",
    "weld",
]
