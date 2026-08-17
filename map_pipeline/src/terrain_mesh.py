"""Spends terrain vertices where the ground actually moves.

A regular grid charges the same price everywhere. Over a Dutch bbox that is a
bad trade: most of the area is a field or a car park that three vertices would
describe perfectly, while the dike, the canal bank and the viaduct ramp that
you can actually see get the same handful of vertices as the flat bits.

This module replaces the grid with a right-triangulated irregular network. Two
triangles cover the bbox; each is cut at the midpoint of its hypotenuse, over
and over, and a cut only happens where the ground under that triangle strays
further from it than the tolerance allows. Flat ground stops splitting almost
immediately, a canal bank keeps splitting to the grid.

Two properties make this the right shape for a mesh headed into Unity:

* No cracks. Every triangle is a right isoceles triangle on the grid diagonal,
  and the error that decides a split is stored per *vertex*, so the two
  triangles that share a hypotenuse always agree about whether to cut it.
  There are no T-junctions to seam over and no skirts to hide them.
* No slivers. Every triangle has the same three angles, which is what keeps
  normals and lightmaps well behaved where a greedy point-insertion TIN would
  hand you long thin wedges.

The error driving the split is nested: a triangle's error is the worst of its
own midpoint and everything below it. Without that a triangle can pass its own
midpoint test while hiding a dike between two of its corners.

Reference: Evans, Kirkpatrick and Townsend, "Right-triangulated irregular
networks" (2001); the level-synchronous form here follows Mapbox's martini.

`build_constrained` is the other mesh, and the one that actually ships. It puts
edges along real features by triangulating the breaklines together with the
ground, and the bisection mesh above is reduced to a way of choosing which grid
points are worth keeping.

That hand-off is where the whole thing used to come apart, and the reason is
worth stating plainly: **the bisection mesh's tolerance is a property of its own
triangles, not of its points.** It keeps a vertex because of the right triangles
*it* would have drawn between them; Delaunay draws different ones, and the
guarantee does not come with. Measured, the same points re-triangulated were
three times outside the tolerance they had been chosen for, and with the
breaklines added, fourteen times.

So the point set has to be earned back, and `build_constrained` does it with
Delaunay refinement after Ruppert: split any constraint segment that a vertex
encroaches upon or that the ground has sagged away from, and insert the
circumcentre of any triangle that is too wrong. The circumcentre matters more
than it looks -- the centroid sits among the corners it came from, so inserting
it splits a bad triangle into three of the same shape, and the loop that did
that added 132 points, then 35, then 33, then 38, and finished no closer than it
started.

Reference: Ruppert, "A Delaunay refinement algorithm for quality 2-dimensional
mesh generation" (1995).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .cdt import WELD_M

LOG = logging.getLogger(__name__)

# How many times the constrained mesh is allowed to add points and rebuild.
# Each round costs a full retriangulation. Four was enough when the loop was
# only ever going to stall anyway; a loop that converges deserves the room to.
MAX_REFINE_ROUNDS = 12

# A floor on triangle shape, and a deliberately low one: it is here to catch
# the degenerate, not to chase a quality bound.
#
# Refinement textbooks aim for 20 degrees or so, and measuring it here says
# not to. Driving on height alone already gives 8.2% of triangles under 20
# degrees; asking for 20 gives *15.7%* and 80% more triangles, because a whole
# round of circumcentres goes in at once and inserting one into a marginal
# triangle mostly makes more marginal triangles. Two degrees costs 5% more
# triangles and halves the genuinely degenerate tail, and that is the whole of
# the bargain worth taking. The shape of this mesh comes from refining on
# height and splitting encroached segments, not from an angle target.
MIN_ANGLE_DEG = 2.0

# Refinement stops at the spacing the ground was measured at, as a fraction of
# a grid step. Below that there is no more information to resolve, only more
# triangles, and no lower bound to a loop that keeps splitting.
REFINE_FLOOR_STEPS = 0.5

# A ceiling on refinement, as a multiple of the points it started with. Ruppert
# terminates on well-formed input; this is the seatbelt for input that is not,
# and it is a real risk here because a BGT planar partition is full of acute
# corners where two outlines meet.
MAX_REFINE_GROWTH = 6.0

# Ground points within this much of a breakline are dropped. Nothing is known
# about the ground at a finer spacing than it was sampled at, so a grid point
# that close is not carrying height -- it is only carrying a thin triangle,
# because the line it is crowding has its own vertices metres apart and the
# triangle between them can only be a wedge.
#
# This is the one part of "grade the density around breaklines" that survived
# contact with the measurements. Forcing the bisection mesh *finer* near a line
# was the other half of that idea and it was simply wrong: it cost 50% more
# triangles for the same shape, because refinement grades the mesh properly by
# itself and a lattice pushed up against an arbitrary polyline can only ever
# make wedges. Removing the clearance entirely is worse again -- 20081
# triangles against 13949, and 22 degenerate ones against none.
GROUND_CLEARANCE_STEPS = 1.0


@dataclass
class TerrainMesh:
    """The simplified surface, plus the grid that still describes it exactly."""

    vertices: np.ndarray  # (m, 3) float64: grid column, grid row, height in NAP
    triangles: np.ndarray  # (t, 3) int32, wound counter-clockwise
    heights: np.ndarray  # (n, n) float32, the grid sampled off this mesh
    tolerance_m: float
    grid_vertices: int
    max_error_m: float
    mean_error_m: float

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    def stats(self) -> dict:
        full_tris = 2 * (self.grid_vertices - 1) ** 2
        return {
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
            "grid_vertices": self.grid_vertices**2,
            "grid_triangles": full_tris,
            "vertex_fraction": round(self.vertex_count / self.grid_vertices**2, 5),
            "triangle_fraction": round(self.triangle_count / full_tris, 5),
            "tolerance_m": round(float(self.tolerance_m), 4),
            "max_error_m": round(float(self.max_error_m), 4),
            "mean_error_m": round(float(self.mean_error_m), 5),
        }


def is_grid_size(n: int) -> bool:
    """True when an n x n grid can carry the bisection hierarchy."""
    return n >= 3 and ((n - 1) & (n - 2)) == 0


def next_grid_size(n: int) -> int:
    """The smallest usable grid size at or above `n`.

    The hierarchy needs a side of 2**k + 1 so that every hypotenuse midpoint
    lands on a grid point. Nothing else is a valid grid.
    """
    n = max(3, int(n))
    size = 3
    while size < n:
        size = ((size - 1) << 1) + 1
    return size


def _hierarchy(k: int) -> list[np.ndarray]:
    """Every triangle in the bisection hierarchy, coarsest level first.

    A triangle is six grid indices: both ends of its hypotenuse, then its
    apex. Cutting one means splitting the hypotenuse at its midpoint -- always
    another grid point -- and handing that midpoint to both halves as their
    new apex.

    There are 2k + 1 levels. Each halves the area, so the last one covers half
    a grid cell and is the full grid triangulation. That last level is the odd
    one out: its hypotenuse runs along a cell diagonal, whose midpoint falls
    between grid points, so it can be drawn but never cut. Levels 0 to 2k - 1
    are the ones that carry a split point, which is why the error pass and the
    cut decision both stop one level short of the end.
    """
    side = 1 << k
    levels = [
        np.array(
            [
                [0, 0, side, side, side, 0],
                [side, side, 0, 0, 0, side],
            ],
            dtype=np.int32,
        )
    ]
    for _ in range(2 * k):
        ax, ay, bx, by, cx, cy = levels[-1].T
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        levels.append(
            np.concatenate(
                [
                    np.column_stack([cx, cy, ax, ay, mx, my]),
                    np.column_stack([bx, by, cx, cy, mx, my]),
                ]
            ).astype(np.int32)
        )
    return levels


def _nested_errors(heights: np.ndarray, levels: list[np.ndarray]) -> np.ndarray:
    """Worst height a split point hides, for it and everything beneath it.

    Stored per vertex rather than per triangle, and taken as the maximum over
    both triangles that share the hypotenuse. That is what keeps the mesh
    watertight: neighbours across a hypotenuse read the same number, so they
    always make the same decision about cutting it.
    """
    error = np.zeros(heights.shape, dtype=np.float64)
    # The last level has no split point of its own, so it contributes nothing.
    finest_split = len(levels) - 2
    for depth in range(finest_split, -1, -1):
        ax, ay, bx, by, cx, cy = levels[depth].T
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        own = np.abs(
            0.5 * (heights[ay, ax] + heights[by, bx]) - heights[my, mx]
        )
        if depth < finest_split:
            # The two halves are cut at the midpoints of this triangle's legs.
            own = np.maximum(own, error[(ay + cy) >> 1, (ax + cx) >> 1])
            own = np.maximum(own, error[(by + cy) >> 1, (bx + cx) >> 1])
        np.maximum.at(error, (my, mx), own)
    return error


def _wind_ccw(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Flip whichever triangles face down, so every normal points at the sky."""
    a = vertices[triangles[:, 0], :2]
    b = vertices[triangles[:, 1], :2]
    c = vertices[triangles[:, 2], :2]
    twice_area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    flipped = twice_area < 0
    out = triangles.copy()
    out[flipped] = out[flipped][:, [0, 2, 1]]
    return out


def distance_to_lines(points, segments, xs, ys) -> np.ndarray:
    """Distance from every grid node to the nearest constraint segment.

    A chamfer transform, so the answer is a couple of percent out. That is
    plenty: it only ever sets how big a triangle may be, and being 2% wrong
    about that changes nothing anyone can see.
    """
    n_rows, n_cols = len(ys), len(xs)
    step_x = float(xs[1] - xs[0]) if n_cols > 1 else 1.0
    step_y = float(ys[1] - ys[0]) if n_rows > 1 else 1.0
    spacing = 0.5 * (abs(step_x) + abs(step_y))
    far = float(n_rows + n_cols)
    dist = np.full((n_rows, n_cols), far, dtype=np.float64)

    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.int64).reshape(-1, 2)
    if len(segments):
        a, b = points[segments[:, 0]], points[segments[:, 1]]
        # Walk each segment at half a cell so no crossed cell is missed.
        run = np.hypot(*(b - a).T)
        steps = np.maximum(1, np.ceil(run / (0.5 * spacing)).astype(np.int64))
        offsets = np.concatenate([np.linspace(0.0, 1.0, s + 1) for s in steps])
        owner = np.repeat(np.arange(len(segments)), steps + 1)
        walk = a[owner] + offsets[:, None] * (b - a)[owner]
        cols = np.clip(np.round((walk[:, 0] - xs[0]) / step_x), 0, n_cols - 1)
        rows = np.clip(np.round((walk[:, 1] - ys[0]) / step_y), 0, n_rows - 1)
        dist[rows.astype(np.int64), cols.astype(np.int64)] = 0.0

    # Two sweeps over the grid, in cells. Within a row the horizontal term is
    # a running minimum of (value - index), which is the whole point: it turns
    # a sequential scan into one accumulate.
    diag = np.sqrt(2.0)
    index = np.arange(n_cols, dtype=np.float64)

    def sweep(rows_in_order, above):
        for r in rows_in_order:
            row = dist[r]
            if above is not None and 0 <= r + above < n_rows:
                near = dist[r + above]
                np.minimum(row, near + 1.0, out=row)
                np.minimum(row[1:], near[:-1] + diag, out=row[1:])
                np.minimum(row[:-1], near[1:] + diag, out=row[:-1])
            np.minimum(row, index + np.minimum.accumulate(row - index), out=row)
            flip = row[::-1]
            np.minimum(flip, index + np.minimum.accumulate(flip - index), out=flip)

    sweep(range(n_rows), -1)
    sweep(range(n_rows - 1, -1, -1), 1)
    return dist * spacing


def build_rtin(heights: np.ndarray, *, tolerance_m: float) -> TerrainMesh:
    """Simplify an (n, n) height grid to within `tolerance_m` of itself.

    Returns the mesh and, alongside it, the same grid resampled off that mesh.
    Everything downstream that drapes on the ground -- roads, rails, trees,
    water levels -- keeps reading a grid, and reading this one means it lands
    on the surface Unity will actually show rather than on the one that was
    thrown away.
    """
    heights = np.asarray(heights)
    if heights.ndim != 2 or heights.shape[0] != heights.shape[1]:
        raise ValueError(f"heights must be a square grid, got {heights.shape}")
    n = int(heights.shape[0])
    if not is_grid_size(n):
        raise ValueError(
            f"terrain grid is {n} across; the bisection hierarchy needs "
            f"2**k + 1 (the next one up is {next_grid_size(n)})"
        )
    tolerance = max(float(tolerance_m), 0.0)

    k = int(round(np.log2(n - 1)))
    levels = _hierarchy(k)
    source = heights.astype(np.float64)
    error = _nested_errors(source, levels)

    flat = source.copy()
    reachable = np.ones(len(levels[0]), dtype=bool)
    faces: list[np.ndarray] = []
    finest_split = len(levels) - 2
    for depth, level in enumerate(levels):
        if depth > finest_split:
            # Half-cell triangles: nothing left to cut, so everything the
            # descent still reaches is a leaf.
            faces.append(level[reachable])
            break
        ax, ay, bx, by, cx, cy = level.T
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        # A triangle is cut when its nested error is over budget. Because that
        # error only grows towards the root, anything worth cutting is already
        # reachable, so this needs no separate check against `reachable`.
        split = error[my, mx] > tolerance
        # Whatever is not cut collapses its own midpoint onto its hypotenuse.
        # Repeated top-down that lays every grid point inside a surviving
        # triangle exactly on its plane, which is what makes `flat` the mesh.
        held = ~split
        flat[my[held], mx[held]] = 0.5 * (
            flat[ay[held], ax[held]] + flat[by[held], bx[held]]
        )
        leaf = reachable & held
        if leaf.any():
            faces.append(level[leaf])
        reachable = np.concatenate([split, split])

    # Each row is already (ax, ay, bx, by, cx, cy), so this unpacks straight
    # into three corners per triangle, in winding order.
    corners = np.concatenate([f.reshape(-1, 2) for f in faces]).astype(np.int64)
    flat_index = corners[:, 1] * n + corners[:, 0]
    used, inverse = np.unique(flat_index, return_inverse=True)
    rows, cols = np.divmod(used, n)
    vertices = np.column_stack(
        [cols.astype(np.float64), rows.astype(np.float64), flat[rows, cols]]
    )
    triangles = _wind_ccw(vertices, inverse.reshape(-1, 3).astype(np.int32))

    residual = np.abs(flat - source)
    mesh = TerrainMesh(
        vertices=vertices,
        triangles=triangles,
        heights=flat.astype(np.float32),
        tolerance_m=tolerance,
        grid_vertices=n,
        max_error_m=float(residual.max()),
        mean_error_m=float(residual.mean()),
    )
    LOG.info(
        "terrain mesh: %d vertices and %d triangles, %.1f%% of the %dx%d grid, "
        "worst height error %.3f m",
        mesh.vertex_count,
        mesh.triangle_count,
        100.0 * mesh.triangle_count / (2 * (n - 1) ** 2),
        n,
        n,
        mesh.max_error_m,
    )
    return mesh


@dataclass
class ConstrainedMesh:
    """A terrain whose edges follow the features the ground actually has."""

    vertices: np.ndarray  # (n, 3) float64: RD easting, RD northing, NAP height
    triangles: np.ndarray  # (m, 3) int32, counter-clockwise
    breakline_edges: int
    height_points: int
    tolerance_m: float
    max_error_m: float
    counts: dict
    # How far the ground strays from a breakline, which the error above cannot
    # see: it samples inside triangles, and a constrained edge is the boundary
    # between two of them.
    max_edge_sag_m: float = 0.0
    refined_rounds: int = 0
    converged: bool = True
    # Triangles further from the ground than the height model's own resolution
    # allows. This, not the raw worst error, is the number that means the mesh
    # has a fault rather than the data having a step in it.
    triangles_over_allowance: int = 0
    breaklines_over_allowance: int = 0
    reachable_tolerance_m: float = 0.0
    # Sliver triangles, split by whose fault they are. A surveyed outline that
    # meets another at four degrees puts a four-degree triangle in the mesh and
    # no triangulation can do better -- the wedge is in the input. Only the
    # other number says the mesh has a problem.
    slivers_from_input: int = 0
    slivers_of_our_own: int = 0
    # How far this mesh stands above the height grid that roads, rails and
    # water are draped on. Anything laid on the grid has to clear this or the
    # terrain pokes through it, and it is a percentile rather than a maximum on
    # purpose: one bad triangle in four square kilometres should not lift every
    # road in the model.
    rise_above_grid_m: float = 0.0
    # The same faces, with flat pairs of triangles fused into quads. The
    # triangles above stay the source of truth -- every check measures those --
    # and this is what gets exported, because it is what an editor can work on.
    face_loops: np.ndarray | None = None
    face_sizes: np.ndarray | None = None
    quad_stats: dict = field(default_factory=dict)
    # Which edges are breaklines, as vertex index pairs. Marked sharp on export
    # so a kerb or a bank can be selected as an edge loop in Blender instead of
    # hunted for one triangle at a time.
    sharp_edges: np.ndarray | None = None

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    def stats(self) -> dict:
        return {
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
            "breakline_edges": int(self.breakline_edges),
            "height_points": int(self.height_points),
            "tolerance_m": round(float(self.tolerance_m), 4),
            "max_error_m": round(float(self.max_error_m), 4),
            "max_edge_sag_m": round(float(self.max_edge_sag_m), 4),
            "reachable_tolerance_m": round(float(self.reachable_tolerance_m), 4),
            "triangles_over_allowance": int(self.triangles_over_allowance),
            "breaklines_over_allowance": int(self.breaklines_over_allowance),
            "rise_above_grid_m": round(float(self.rise_above_grid_m), 4),
            "slivers_from_input": int(self.slivers_from_input),
            "slivers_of_our_own": int(self.slivers_of_our_own),
            **({"quads": self.quad_stats} if self.quad_stats else {}),
            "refined_rounds": int(self.refined_rounds),
            "refinement_converged": bool(self.converged),
            "breaklines_by_source": dict(self.counts),
        }


def build_constrained(
    bbox,
    heights: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    rings_by_source: dict,
    tolerance_m: float = 0.10,
    simplify_m: float = 0.15,
    contour_interval_m: float = 0.5,
    min_feature_length_m: float = 0.0,
    quads: bool = False,
    max_fold_deg: float = 12.0,
    min_quad_angle_deg: float = 25.0,
) -> ConstrainedMesh:
    """Triangulate the area so every breakline comes out as an edge.

    Two kinds of point go in. The breaklines decide where the mesh folds, and
    they come with the segments that have to survive. On top of those go the
    grid points the height model cannot do without, chosen by the same
    bisection error test the plain adaptive mesh uses -- flat ground contributes
    almost none of them, which is the whole point.
    """
    from .breaklines import build_breaklines
    from .cdt import triangulate, weld

    lines = build_breaklines(
        bbox,
        rings_by_source=rings_by_source,
        heights=heights,
        xs=xs,
        ys=ys,
        simplify_m=simplify_m,
        contour_interval_m=contour_interval_m,
        # Contours are simplified against the height they cost, not against a
        # distance, so they need to know what the mesh is aiming for.
        tolerance_m=tolerance_m,
        min_feature_length_m=min_feature_length_m,
    )

    # Grid points worth keeping for height alone. The adaptive mesh already
    # answers exactly that question, so its vertices are reused as a point set
    # rather than as a triangulation.
    #
    # Height is not the only thing they are wanted for, though. Where a
    # breakline runs through flat ground, height alone asks for no points at
    # all, and the triangulation is then left to reach from a vertex every few
    # metres along the line to the far side of an empty field. So the same pass
    # is also told how big a triangle may be near a line, and it fills in a
    # band that steps the density down instead of dropping off it.
    step = 0.5 * (
        abs(float(xs[1] - xs[0])) + abs(float(ys[1] - ys[0]))
    ) if len(xs) > 1 and len(ys) > 1 else 1.0
    gap = (
        distance_to_lines(lines.points, lines.segments, xs, ys)
        if len(lines.segments)
        else None
    )
    height_mesh = build_rtin(heights, tolerance_m=tolerance_m)
    columns = height_mesh.vertices[:, 0].astype(int)
    rows = height_mesh.vertices[:, 1].astype(int)
    if gap is not None:
        clear = gap[rows, columns] >= GROUND_CLEARANCE_STEPS * step
        columns, rows = columns[clear], rows[clear]
    height_points = np.column_stack([xs[columns], ys[rows]])

    n_break = len(lines.points)
    combined = (
        np.vstack([lines.points, height_points]) if n_break else height_points
    )
    merged, mapping = weld(combined, WELD_M)
    segments = mapping[lines.segments] if len(lines.segments) else lines.segments
    if len(segments):
        segments = segments[segments[:, 0] != segments[:, 1]]

    # Breaklines decide where the mesh folds, but they say nothing about the
    # ground between them, and the height points were chosen for a different
    # triangulation than the one they end up in -- the bisection mesh keeps a
    # point because of the triangles *it* would have drawn, and Delaunay draws
    # different ones. So the point set has to be earned back here.
    #
    # This is Delaunay refinement, after Ruppert: split any constraint segment
    # that is encroached upon or that the ground has left behind, and insert
    # the circumcentre of any triangle that is too wrong or too thin. Both
    # halves matter. Splitting segments alone leaves the interior coarse;
    # inserting circumcentres alone leaves a long breakline as one forced edge
    # with the mesh hanging off it.
    floor_m = REFINE_FLOOR_STEPS * step
    clearance = GROUND_CLEARANCE_STEPS * step
    allowance = height_allowance(heights, tolerance_m)
    reachable = float(allowance.max())
    if reachable > tolerance_m * 1.01:
        LOG.info(
            "terrain mesh: the height model steps by up to %.2f m between "
            "neighbouring samples, so a %.2f m tolerance is not reachable "
            "everywhere; up to %.2f m is allowed where it steps",
            2.0 * reachable,
            tolerance_m,
            reachable,
        )
    budget = int(MAX_REFINE_GROWTH * max(len(merged), 1))
    result = None
    rounds_used = 0
    for round_index in range(MAX_REFINE_ROUNDS):
        rounds_used = round_index + 1
        result = triangulate(
            merged, segments, origin=tuple(bbox.center), skip_crossing=True
        )
        # The mesh handed back is this one, so the segments that built it are
        # the ones to report and to measure. A round that ends on the budget
        # leaves `segments` describing points the returned mesh never got.
        built_from = segments
        world = result.world_points()
        z = _bilinear(heights, xs, ys, world[:, 0], world[:, 1])
        vertices = np.column_stack([world[:, 0], world[:, 1], z])

        length_of = (
            np.hypot(*(vertices[segments[:, 1], :2] - vertices[segments[:, 0], :2]).T)
            if len(segments)
            else np.zeros(0)
        )
        cut = (
            _segments_to_split(
                vertices, segments, result.triangles, heights, xs, ys,
                allowance, floor_m,
            )
            if len(segments)
            else np.zeros(0, dtype=bool)
        )
        extra, stuck = _refine_points(
            vertices, result.triangles, heights, xs, ys, allowance, floor_m,
            gap, floor_m,
        )
        # Anything that could not take a circumcentre falls back on splitting
        # the constraint that blocked it.
        if len(stuck) and len(segments):
            index_of = {
                (min(int(a), int(b)), max(int(a), int(b))): i
                for i, (a, b) in enumerate(segments)
            }
            for tri in result.triangles[stuck]:
                for i in range(3):
                    a, b = int(tri[i]), int(tri[(i + 1) % 3])
                    at = index_of.get((min(a, b), max(a, b)))
                    if at is not None and length_of[at] > 2.0 * floor_m:
                        cut[at] = True

        if not cut.any() and len(extra) == 0:
            break
        if len(merged) >= budget:
            LOG.info(
                "terrain mesh: refinement stopped at its %d point ceiling", budget
            )
            break

        # Segment midpoints go in as points and as two segments each, so the
        # constraint survives the split rather than being replaced by it.
        added = [merged]
        next_index = len(merged)
        keep = list(segments[~cut]) if len(segments) else []
        mids = []
        sharp = _acute_corners(merged, segments)
        for index in np.flatnonzero(cut):
            p, q = segments[index]
            mids.append(_split_at(merged, segments, index, sharp, floor_m))
            keep.append((int(p), next_index))
            keep.append((next_index, int(q)))
            next_index += 1
        if mids:
            mids = np.asarray(mids)
            added.append(mids)
            extra = _keep_clear_of(extra, mids, floor_m)
        if len(extra):
            added.append(extra)

        LOG.info(
            "terrain mesh: round %d splits %d breakline segments and adds "
            "%d points",
            round_index + 1,
            int(cut.sum()),
            len(extra),
        )
        before = len(merged)
        merged, mapping = weld(np.vstack(added), WELD_M)
        segments = (
            np.asarray(keep, dtype=np.int64).reshape(-1, 2)
            if keep
            else np.zeros((0, 2), dtype=np.int64)
        )
        if len(segments):
            segments = mapping[segments].reshape(-1, 2)
            segments = segments[segments[:, 0] != segments[:, 1]]
            segments = np.unique(np.sort(segments, axis=1), axis=0)
        if len(merged) == before:
            break
    else:
        # The loop ran out of rounds with work still applied but never
        # triangulated. Returning `result` here would hand back a mesh a round
        # older than the segments describing it.
        result = triangulate(
            merged, segments, origin=tuple(bbox.center), skip_crossing=True
        )
        built_from = segments

    world = result.world_points()
    z = _bilinear(heights, xs, ys, world[:, 0], world[:, 1])
    vertices = np.column_stack([world[:, 0], world[:, 1], z])

    # Checked here, where the mesh was built, rather than left to the
    # validation stage. A tear does not announce itself downstream: the
    # adjacency around a hole stays self-consistent, so the first sign used to
    # be an unrelated vertex thousands of constraints later.
    _assert_tiles(vertices, result.triangles, bbox)

    lost = result.missing_constraints()
    if lost:
        raise ValueError(
            f"{len(lost)} breaklines did not survive triangulation; the mesh "
            f"would not fold where the ground does"
        )

    error = _sampled_error(vertices, result.triangles, heights, xs, ys)
    sag, sagging = _edge_sag(vertices, built_from, heights, xs, ys, allowance)
    over = _over_allowance(vertices, result.triangles, heights, xs, ys, allowance)
    from_input, our_own = _count_slivers(vertices, result.triangles, built_from)
    rise = _rise_above_grid(vertices, result.triangles, heights, xs, ys)
    # Converged means the mesh reached what it was asked for. Counting how much
    # work the last round did measures the loop, not the answer, and the tail
    # of a converging refinement is always a handful of points on a mesh of
    # thousands -- which reads as "still working" and is not.
    converged = bool(over == 0 and sagging == 0)
    mesh = ConstrainedMesh(
        vertices=vertices,
        triangles=result.triangles,
        breakline_edges=len(built_from),
        height_points=len(height_points),
        tolerance_m=float(tolerance_m),
        max_error_m=error,
        counts=lines.counts,
        max_edge_sag_m=sag,
        refined_rounds=rounds_used,
        converged=converged,
        triangles_over_allowance=over,
        breaklines_over_allowance=sagging,
        reachable_tolerance_m=reachable,
        slivers_from_input=from_input,
        slivers_of_our_own=our_own,
        rise_above_grid_m=rise,
    )
    if quads:
        from .quadmesh import pair_into_quads

        protected = {frozenset((int(a), int(b))) for a, b in built_from}
        mesh.face_loops, mesh.face_sizes, mesh.quad_stats = pair_into_quads(
            vertices,
            result.triangles,
            protected,
            max_fold_deg=max_fold_deg,
            min_quad_angle_deg=min_quad_angle_deg,
        )
    mesh.sharp_edges = (
        np.asarray(built_from, dtype=np.int32).reshape(-1, 2)
        if len(built_from)
        else np.zeros((0, 2), dtype=np.int32)
    )

    LOG.info(
        "terrain mesh: %d vertices, %d triangles, %d breakline edges, "
        "worst height error %.3f m, worst breakline sag %.3f m, "
        "%d triangles and %d breaklines over what the height model allows, "
        "%d slivers of our own and %d from sharp corners in the input, "
        "%d refinement rounds%s",
        mesh.vertex_count,
        mesh.triangle_count,
        mesh.breakline_edges,
        mesh.max_error_m,
        mesh.max_edge_sag_m,
        mesh.triangles_over_allowance,
        mesh.breaklines_over_allowance,
        mesh.slivers_of_our_own,
        mesh.slivers_from_input,
        mesh.refined_rounds,
        "" if converged else " (stopped short)",
    )
    return mesh


def _assert_tiles(vertices, triangles, bbox) -> None:
    """The triangles must cover the bbox exactly once, with none facing down."""
    corners = vertices[triangles][:, :, :2]
    a = corners[:, 1] - corners[:, 0]
    b = corners[:, 2] - corners[:, 0]
    twice_area = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    want = bbox.width * bbox.height
    covered = 0.5 * float(twice_area.sum())
    if abs(covered - want) > 1e-6 * want:
        raise ValueError(
            f"the terrain mesh covers {covered:.1f} m2 of a {want:.0f} m2 "
            f"bbox, so it has a hole or an overlap"
        )
    down = int((twice_area <= 0).sum())
    if down:
        raise ValueError(f"{down} terrain triangles face down or are degenerate")


def _barycentric_probes(rounds: int = 4):
    """Sample positions inside a triangle, as barycentric weights.

    The old refinement looked at the centroid and three points a third of the
    way out to the corners, all of them huddled in the middle. A triangle that
    straddles a canal bank is at its worst near an *edge*, so those four probes
    passed it and the mesh kept the wedge. These reach out to a twentieth of
    the way from each edge, which is close enough to see it.
    """
    out = []
    for i in range(rounds + 1):
        for j in range(rounds + 1 - i):
            k = rounds - i - j
            weights = np.array([i, j, k], dtype=np.float64) / rounds
            # Pulled a little off the corners and edges: a probe exactly on an
            # edge is shared with the neighbour and says nothing about either.
            weights = 0.05 / 3.0 + 0.95 * weights
            out.append(weights)
    return np.asarray(out)


PROBES = _barycentric_probes()


def _height_error(vertices, triangles, heights, xs, ys, *, want_where=False):
    """Worst gap between each triangle and the ground under it.

    With `want_where`, also returns where in each triangle that worst gap was
    found, which is what tells refinement where to aim.
    """
    if len(triangles) == 0:
        empty = np.zeros(0)
        return (empty, np.zeros((0, 2))) if want_where else empty
    corners = vertices[triangles]
    worst = np.zeros(len(triangles))
    where = corners[:, 0, :2].copy() if want_where else None
    for weights in PROBES:
        probe = (
            weights[0] * corners[:, 0]
            + weights[1] * corners[:, 1]
            + weights[2] * corners[:, 2]
        )
        truth = _bilinear(heights, xs, ys, probe[:, 0], probe[:, 1])
        gap = np.abs(probe[:, 2] - truth)
        if want_where:
            better = gap > worst
            where[better] = probe[better, :2]
        np.maximum(worst, gap, out=worst)
    return (worst, where) if want_where else worst


def _allowance_over(vertices, triangles, allowance, xs, ys):
    """The allowance across a whole triangle, not at one point in it.

    A triangle's ability to follow the ground is limited by the roughest part
    of the ground it covers, and that is not always where its own worst error
    lands. Reading the allowance at the error's location alone flagged 2515
    triangles on a real 2 km area for a step that was inside them.
    """
    if len(triangles) == 0:
        return np.zeros(0)
    corners = vertices[triangles]
    best = np.zeros(len(triangles))
    for weights in PROBES:
        probe = (
            weights[0] * corners[:, 0]
            + weights[1] * corners[:, 1]
            + weights[2] * corners[:, 2]
        )
        np.maximum(best, _sample(allowance, xs, ys, probe), out=best)
    return best


def _snap_to_grid(points, xs, ys):
    """Move each point to the nearest node of the height grid.

    Refinement used to insert circumcentres for height as well as for shape,
    and a circumcentre is almost never a grid node. That matters more than it
    sounds: a real AHN model is full of one-cell features -- a quay wall, the
    lip of a filled building hole, the scar where a tree was taken out -- and
    the only way a mesh reproduces one is to have a vertex *at* it. Inserting
    beside it instead leaves the error exactly where it was, which is why
    refinement stalled at two and a half times its tolerance on real ground
    however fine it was allowed to cut.

    There are also finitely many grid nodes, so this is what makes the loop
    terminate at the resolution of the data instead of chasing a target that
    the data cannot express.
    """
    if len(points) == 0:
        return points
    step_x = float(xs[1] - xs[0])
    step_y = float(ys[1] - ys[0])
    cols = np.clip(np.round((points[:, 0] - xs[0]) / step_x), 1, len(xs) - 2)
    rows = np.clip(np.round((points[:, 1] - ys[0]) / step_y), 1, len(ys) - 2)
    return np.column_stack([xs[cols.astype(int)], ys[rows.astype(int)]])


def height_allowance(heights: np.ndarray, tolerance_m: float) -> np.ndarray:
    """How close to the ground the mesh can actually be asked to get, per node.

    A tolerance is a promise about a surface, and this one is a grid. Between
    two neighbouring samples the ground is whatever the interpolation says, and
    where those two samples differ by a step -- a quay wall, the lip of a filled
    building hole, the scar where a tree was taken out -- a flat triangle
    spanning the pair can only sit about half the step away from it, unless one
    of its edges happens to lie exactly along the boundary between the two
    cells. A Delaunay mesh over scattered points cannot promise that.

    So the allowance is the tolerance, or half the local step, whichever is
    larger. Asking for less is asking the mesh to reproduce detail that the
    height model does not contain, and it does real damage: refinement cannot
    ever satisfy it, so it keeps inserting points into triangles it has no way
    to improve. On a real Utrecht kilometre that ran to 701,978 triangles and
    still reported a tenth of a metre missed by a factor of five.

    Both refinement and the checks read this same number, on purpose. A target
    the mesh is driven towards and a target it is judged against have to be the
    same target.
    """
    heights = np.asarray(heights, dtype=np.float64)
    step = np.zeros_like(heights)
    for axis in (0, 1):
        difference = np.abs(np.diff(heights, axis=axis))
        pad = [(0, 0), (0, 0)]
        pad[axis] = (0, 1)
        np.maximum(step, np.pad(difference, pad, mode="edge"), out=step)
        pad[axis] = (1, 0)
        np.maximum(step, np.pad(difference, pad, mode="edge"), out=step)
    return np.maximum(float(tolerance_m), 0.5 * step)


def grid_interpolation_floor(heights: np.ndarray) -> float:
    """The best any mesh on this grid can do, in metres.

    Everything downstream reads the ground as a bilinear surface over the grid,
    and a triangle is flat. Inside one cell the two differ by the cell's twist
    over eight, and no amount of refinement removes it -- cutting a cell finer
    only interpolates the same four numbers. Asking for a tolerance under this
    is asking for detail the height model does not carry, so it is worth saying
    so rather than refining forever and failing a check.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if heights.shape[0] < 2 or heights.shape[1] < 2:
        return 0.0
    twist = np.abs(
        heights[:-1, :-1] + heights[1:, 1:] - heights[:-1, 1:] - heights[1:, :-1]
    )
    return float(twist.max()) / 8.0


def _min_angles(points, triangles):
    """Smallest angle of each triangle, in degrees."""
    if len(triangles) == 0:
        return np.zeros(0)
    corner = points[triangles][:, :, :2]
    angles = []
    for i in range(3):
        u = corner[:, (i + 1) % 3] - corner[:, i]
        v = corner[:, (i + 2) % 3] - corner[:, i]
        cos = (u * v).sum(axis=1) / np.maximum(
            np.hypot(*u.T) * np.hypot(*v.T), 1e-30
        )
        angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.min(np.stack(angles), axis=0)


def _circumcentres(points, triangles):
    """Centre of each triangle's circumcircle, and its radius."""
    a = points[triangles[:, 0], :2]
    b = points[triangles[:, 1], :2]
    c = points[triangles[:, 2], :2]
    bx, by = (b - a).T
    cx, cy = (c - a).T
    d = 2.0 * (bx * cy - by * cx)
    safe = np.where(np.abs(d) < 1e-18, 1e-18, d)
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (cy * b2 - by * c2) / safe
    uy = (bx * c2 - cx * b2) / safe
    centre = a + np.column_stack([ux, uy])
    return centre, np.hypot(ux, uy)


def _apexes_across(triangles):
    """For each edge, the opposite corner of every triangle carrying it.

    This is what makes the encroachment test cheap. A vertex inside the
    diametral circle of a constrained edge is, in a constrained Delaunay
    triangulation, always visible as the apex of one of the two triangles that
    edge belongs to, so there is no need to search the whole point set.
    """
    across: dict = {}
    for a, b, c in triangles:
        a, b, c = int(a), int(b), int(c)
        across.setdefault((min(a, b), max(a, b)), []).append(c)
        across.setdefault((min(b, c), max(b, c)), []).append(a)
        across.setdefault((min(c, a), max(c, a)), []).append(b)
    return across


def _segments_to_split(points, segments, triangles, heights, xs, ys, allowance, floor_m):
    """Which constraint segments have to be cut in half, and why.

    Two reasons, both of which the old code had no answer to at all.

    *Encroachment*: a vertex sitting inside a segment's diametral circle can
    only ever be the apex of a thin triangle on that segment. Splitting the
    segment is Ruppert's answer and it is what stops a breakline from being a
    single forced edge hundreds of metres long with the mesh draped off it.

    *Sag*: a constraint is a straight line between its endpoints, and the
    ground under it is not. Simplification is what makes this bite -- it exists
    to delete the vertices in between -- so the segment has to earn its length
    back wherever the ground disagrees with it by more than the tolerance.
    """
    if len(segments) == 0:
        return np.zeros(0, dtype=bool)

    a = points[segments[:, 0], :2]
    b = points[segments[:, 1], :2]
    length = np.hypot(*(b - a).T)
    want = np.zeros(len(segments), dtype=bool)

    across = _apexes_across(triangles)
    midpoint = 0.5 * (a + b)
    radius = 0.5 * length
    for index, (p, q) in enumerate(segments):
        for apex in across.get((min(int(p), int(q)), max(int(p), int(q))), ()):
            if np.hypot(*(points[apex, :2] - midpoint[index])) < radius[index] * 0.999:
                want[index] = True
                break

    # Sag, measured along the chord against the height grid.
    steps = np.linspace(0.1, 0.9, 9)
    z0 = points[segments[:, 0], 2]
    z1 = points[segments[:, 1], 2]
    for s in steps:
        probe = a + s * (b - a)
        truth = _bilinear(heights, xs, ys, probe[:, 0], probe[:, 1])
        allowed = _sample(allowance, xs, ys, probe)
        want |= np.abs((z0 + s * (z1 - z0)) - truth) > allowed

    # Nothing is gained by cutting below the spacing the ground was measured
    # at, and it is the only thing standing between this and a loop that never
    # ends on a segment running along a cliff.
    return want & (length > 2.0 * floor_m)


def _acute_corners(points, segments, degrees=60.0):
    """Vertices where two constraints meet at less than `degrees`.

    These are Ruppert's known bad case, and BGT is full of them: wherever one
    road splits off another, or a building footprint crosses a kerb at a
    shallow angle. Splitting one of the two segments puts a new vertex inside
    the other's diametral circle, which makes that one split, which puts a
    vertex inside the first's -- and the pair grind each other down for as long
    as they are allowed to, throwing off a sliver at every step.
    """
    at: dict = {}
    for a, b in segments:
        at.setdefault(int(a), []).append(int(b))
        at.setdefault(int(b), []).append(int(a))
    sharp = set()
    limit = np.cos(np.radians(degrees))
    for vertex, others in at.items():
        if len(others) < 2:
            continue
        arms = points[others, :2] - points[vertex, :2]
        norm = np.hypot(*arms.T)
        keep = norm > 1e-12
        arms, norm = arms[keep], norm[keep]
        if len(arms) < 2:
            continue
        unit = arms / norm[:, None]
        cosines = unit @ unit.T
        np.fill_diagonal(cosines, -1.0)
        if cosines.max() > limit:
            sharp.add(vertex)
    return sharp


def _split_at(points, segments, index, sharp, floor_m):
    """Where to cut a segment: its midpoint, or a concentric shell.

    A segment with one end at a sharp corner is cut at a power-of-two distance
    from that corner rather than in half. Both segments meeting there then get
    their cut at the same radius, so the two new vertices are as far from each
    other as they are from the corner, and neither lands inside the other's
    diametral circle. The grinding stops.

    This is Ruppert's concentric shells, in Shewchuk's formulation.
    """
    a, b = int(segments[index][0]), int(segments[index][1])
    p, q = points[a, :2], points[b, :2]
    length = float(np.hypot(*(q - p)))
    ends = (a in sharp, b in sharp)
    if ends[0] == ends[1]:
        # Neither end is sharp, or both are: halving is right, and for two
        # sharp ends there is no shell radius that suits both.
        return 0.5 * (p + q)
    corner, far = (p, q) if ends[0] else (q, p)
    shell = 2.0 ** np.floor(np.log2(max(length * 0.5, floor_m)))
    shell = float(min(max(shell, floor_m), length - floor_m))
    return corner + (far - corner) * (shell / length)


def _refine_points(
    vertices, triangles, heights, xs, ys, allowance, floor_m, gap, clearance
):
    """Circumcentres of the triangles that are too wrong or too thin.

    The circumcentre, not the centroid. That is not a detail: the centroid sits
    among the corners it came from, so inserting it splits a bad triangle into
    three more of the same shape and the loop chases its own tail -- which is
    exactly what the old one did, adding 132, then 35, then 33, then 38 points
    and stopping no closer than it started. The circumcentre is by construction
    a full circumradius from every existing vertex, which is what makes each
    insertion buy a well-shaped triangle instead of three thin ones.
    """
    if len(triangles) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)

    error, worst_at = _height_error(
        vertices, triangles, heights, xs, ys, want_where=True
    )
    angle = _min_angles(vertices, triangles)
    centre, radius = _circumcentres(vertices, triangles)

    # Where a triangle is off the ground, aim at the grid node nearest the worst
    # of it: that is a place the ground was actually measured, so the mesh can
    # reproduce what is there. Shape is a different question and still takes the
    # circumcentre, which is the point that provably improves it.
    allowed = _allowance_over(vertices, triangles, allowance, xs, ys)
    off_ground = error > allowed
    target = np.where(off_ground[:, None], _snap_to_grid(worst_at, xs, ys), centre)
    bad = off_ground | (angle < MIN_ANGLE_DEG)
    # A triangle already at the resolution of the height model cannot be
    # improved by looking harder at it.
    bad &= radius > floor_m
    if not bad.any():
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)

    # Height first, then size. Refinement runs against a budget, and a round
    # that spends it all chasing thin triangles leaves the handful that are
    # genuinely off the ground unfixed -- which is the wrong way round, because
    # a thin triangle is ugly and a triangle in the wrong place is wrong.
    where = np.flatnonzero(bad)
    order = np.lexsort((-radius[where], ~off_ground[where]))
    where = where[order]
    picked = target[where]
    keep_radius = radius[where]

    # A grid node the mesh already has welds away to nothing, so the round would
    # report progress and make none. Fall back to the circumcentre -- and do it
    # before anything is filtered, because a substitution made afterwards is one
    # nothing has checked. Doing it the other way round let circumcentres of
    # near-degenerate triangles through, and they are not nearby: the mesh came
    # out covering 672 million square metres of a 250 thousand metre bbox.
    existing = {
        (int(round(x / WELD_M)), int(round(y / WELD_M))) for x, y in vertices[:, :2]
    }
    for index, point in enumerate(picked):
        if (int(round(point[0] / WELD_M)), int(round(point[1] / WELD_M))) in existing:
            picked[index] = centre[where[index]]

    # Inside the area, and not landing on top of a breakline. The clearance
    # here is only the degenerate case, deliberately: keeping circumcentres a
    # fixed distance clear of every breakline sounds prudent and is not, since
    # a canal bank *is* a breakline and refusing to refine within six metres of
    # one leaves the bank exactly as wrong as it was. Splitting encroached
    # segments is what keeps the geometry near a line healthy, and it is
    # scale-adaptive in a way a fixed distance can never be.
    inside = (
        (picked[:, 0] > xs[0]) & (picked[:, 0] < xs[-1])
        & (picked[:, 1] > ys[0]) & (picked[:, 1] < ys[-1])
    )
    if gap is not None:
        inside &= _sample(gap, xs, ys, picked) >= clearance
    # A bad triangle whose point cannot be placed is the boundary case Ruppert's
    # rule exists for: the thing standing in its way -- the constrained edge it
    # would have crossed -- is split instead. Without this the mesh never
    # improves where a canal runs off the edge of the bbox, which is precisely
    # where it was worst.
    stuck = where[~inside]
    picked, keep_radius = picked[inside], keep_radius[inside]

    # Classic refinement inserts one point and rebuilds. Rebuilding per point
    # is far too slow here, so a whole round goes in at once -- and then two
    # points landing on top of each other would make the very sliver this is
    # trying to remove. Spaced on a grid at the local circumradius.
    return _spread(picked, keep_radius, floor_m), stuck


def _keep_clear_of(points, others, distance):
    """Drop any point sitting within `distance` of one already spoken for.

    A round inserts two kinds of point at once: midpoints of segments being
    split, and circumcentres. Each kind is spaced against its own kind and
    neither knew about the other, which is how a circumcentre came to land
    34 mm from a midpoint and leave a pair of triangles at 0.6 degrees.
    """
    if len(points) == 0 or len(others) == 0:
        return points
    cell = max(float(distance), 1e-9)
    buckets: dict = {}
    for other in others:
        buckets.setdefault((int(other[0] // cell), int(other[1] // cell)), []).append(
            other
        )
    keep = []
    for index, point in enumerate(points):
        cx, cy = int(point[0] // cell), int(point[1] // cell)
        near = [
            q
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for q in buckets.get((cx + dx, cy + dy), ())
        ]
        if all(np.hypot(*(point - q)) > distance for q in near):
            keep.append(index)
    return points[keep]


def _spread(points, radius, floor_m):
    """Thin a batch so two new points cannot make the sliver they came to fix.

    The first version of this keyed its grid by the local circumradius as well
    as by position, meaning to allow a fine point to sit near a coarse one. The
    effect was that two circumcentres of almost the same size were never
    compared with each other at all, and a pair of them landed 34 mm apart with
    a 3 m triangle either side. Separation is symmetric or it is nothing.
    """
    if len(points) == 0:
        return points
    cell = max(float(floor_m), 1e-9)
    taken: dict = {}
    keep = []
    for index in range(len(points)):
        # Scaled to the triangle it came from, floored at the resolution of the
        # height model, and capped so a huge triangle cannot make this a scan
        # of the whole grid.
        want = max(cell, 0.3 * float(radius[index]))
        rings = min(int(np.ceil(want / cell)), 8)
        cx = int(points[index, 0] // cell)
        cy = int(points[index, 1] // cell)
        clash = False
        for dx in range(-rings, rings + 1):
            for dy in range(-rings, rings + 1):
                for other in taken.get((cx + dx, cy + dy), ()):
                    if np.hypot(*(points[index] - other)) <= want:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if clash:
            continue
        taken.setdefault((cx, cy), []).append(points[index])
        keep.append(index)
    return points[keep]


def _sample(grid, xs, ys, points):
    """Nearest-node lookup on a grid, clamped to it."""
    cols = np.clip(
        np.round((points[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1
    )
    rows = np.clip(
        np.round((points[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1
    )
    return grid[rows, cols]


# A triangle under this is a sliver. An input corner under it explains one.
SLIVER_DEG = 1.0


def _count_slivers(vertices, triangles, segments):
    """Sliver triangles, split into the input's and ours.

    Two outlines that meet at half a degree -- and a BGT partition does contain
    those, where two surveyed polygons touch at a hair -- put a half-degree
    triangle in the mesh. There is nowhere to place a point that improves it:
    the wedge is the input's shape. Blaming the mesh for those hides the ones
    that are genuinely its own, which are the ones worth fixing.
    """
    if len(triangles) == 0:
        return 0, 0
    smallest = _min_angles(vertices, triangles)
    slivers = smallest < SLIVER_DEG
    if not slivers.any():
        return 0, 0
    if len(segments) == 0:
        return 0, int(slivers.sum())

    # Attributed by whether the input alone decided the triangle's shape. Every
    # corner on a constraint means it fills a gap between surveyed lines -- a
    # wedge where two outlines converge, a strip too narrow to fit anything
    # better -- and there is no point to add that would improve it. A sliver out
    # in open ground has at least one corner that refinement put there, and that
    # one is ours.
    #
    # Membership of the sharp corner itself is not enough on its own: a wedge
    # closing at a degree and a half throws slivers along its whole length, tens
    # of metres from the apex, and none of those touch it.
    on_line = np.zeros(len(vertices), dtype=bool)
    on_line[np.unique(segments)] = True
    from_input = slivers & on_line[triangles].all(axis=1)
    return int(from_input.sum()), int((slivers & ~from_input).sum())


def _rise_above_grid(vertices, triangles, heights, xs, ys) -> float:
    """How far the mesh stands above the grid, at the 99.9th percentile.

    Signed and one-sided: only a mesh standing *above* the grid buries what is
    draped on it, and a mesh dipping below is hidden by whatever is on top. A
    percentile rather than a maximum because this sets a lift applied to every
    road in the model, and taking the worst single triangle in four square
    kilometres lifted all of them by 22 cm.
    """
    if len(triangles) == 0:
        return 0.0
    corners = vertices[triangles]
    best = 0.0
    for weights in PROBES:
        probe = (
            weights[0] * corners[:, 0]
            + weights[1] * corners[:, 1]
            + weights[2] * corners[:, 2]
        )
        truth = _bilinear(heights, xs, ys, probe[:, 0], probe[:, 1])
        above = probe[:, 2] - truth
        best = max(best, float(np.percentile(above, 99.9)))
    return max(best, 0.0)


def _over_allowance(vertices, triangles, heights, xs, ys, allowance) -> int:
    """How many triangles miss the ground by more than the ground allows."""
    if len(triangles) == 0:
        return 0
    error = _height_error(vertices, triangles, heights, xs, ys)
    return int((error > _allowance_over(vertices, triangles, allowance, xs, ys)).sum())


def _edge_sag(vertices, segments, heights, xs, ys, allowance=None):
    """Worst gap between a breakline and the ground beneath it.

    A constrained edge is a straight line the mesh is obliged to keep, and the
    ground under it is under no such obligation. Nothing measured this before,
    and it is not covered by the error inside triangles: that is sampled across
    a face, and this is the face's boundary.
    """
    if len(segments) == 0:
        return 0.0, 0
    a = vertices[segments[:, 0]]
    b = vertices[segments[:, 1]]
    worst = 0.0
    over = np.zeros(len(segments), dtype=bool)
    for s in np.linspace(0.05, 0.95, 19):
        probe = a[:, :2] + s * (b[:, :2] - a[:, :2])
        truth = _bilinear(heights, xs, ys, probe[:, 0], probe[:, 1])
        chord = a[:, 2] + s * (b[:, 2] - a[:, 2])
        gap = np.abs(chord - truth)
        worst = max(worst, float(gap.max()))
        if allowance is not None:
            over |= gap > _sample(allowance, xs, ys, probe)
    return worst, int(over.sum())


def _bilinear(grid, xs, ys, x, y):
    """Height at arbitrary points, from the grid, clamped to its edges."""
    from .elevation import _bilinear_on_grid

    return _bilinear_on_grid(np.asarray(grid, dtype=np.float64), xs, ys, x, y)


def _sampled_error(vertices, triangles, heights, xs, ys) -> float:
    """Worst gap between the mesh and the grid under it.

    Sampled across each triangle rather than at its centre. Centre-only was
    cheap and wrong in the one case that matters: a long wedge laid across a
    canal bank passes through the true surface near the middle and is metres
    out at both ends, so the centre is the single best place to measure if you
    want a flattering answer.
    """
    if len(triangles) == 0:
        return 0.0
    return float(_height_error(vertices, triangles, heights, xs, ys).max())


def save_constrained_mesh(mesh: ConstrainedMesh, path, origin) -> "Path":
    """Write the mesh for the Blender stage, in local metres."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    local = mesh.vertices.copy()
    local[:, 0] -= origin[0]
    local[:, 1] -= origin[1]
    payload = dict(
        vertices=local.astype(np.float64),
        triangles=mesh.triangles.astype(np.int32),
        tolerance_m=np.float64(mesh.tolerance_m),
        max_error_m=np.float64(mesh.max_error_m),
        grid_indexed=np.array(False),
    )
    # Faces as loops and lengths, so one pair of arrays carries the quads and
    # the triangles that would not pair.
    if mesh.face_loops is not None and mesh.face_sizes is not None:
        payload["face_loops"] = mesh.face_loops.astype(np.int32)
        payload["face_sizes"] = mesh.face_sizes.astype(np.int32)
    if mesh.sharp_edges is not None:
        payload["sharp_edges"] = mesh.sharp_edges.astype(np.int32)
    np.savez(path, **payload)
    LOG.info("wrote %s", path)
    return path


def save_terrain_mesh(mesh: TerrainMesh, path) -> "Path":
    """Write the mesh for the Blender stage.

    Vertices carry grid column and row rather than metres. The Blender side
    already holds the xs/ys arrays, so this lands on exactly the same
    coordinates the grid mesh used, and it lets the water bed -- which is a
    grid -- be read straight off a vertex without a search.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vertices=mesh.vertices.astype(np.float32),
        triangles=mesh.triangles.astype(np.int32),
        tolerance_m=np.float64(mesh.tolerance_m),
        max_error_m=np.float64(mesh.max_error_m),
        grid_indexed=np.array(True),
    )
    LOG.info("wrote %s", path)
    return path


__all__ = [
    "TerrainMesh",
    "build_rtin",
    "is_grid_size",
    "next_grid_size",
    "save_terrain_mesh",
]
