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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger(__name__)

# How many times the constrained mesh is allowed to add points and rebuild to
# get closer to the ground. Each round costs a full retriangulation, and the
# gain falls off fast after the second.
MAX_REFINE_ROUNDS = 4


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
) -> ConstrainedMesh:
    """Triangulate the area so every breakline comes out as an edge.

    Two kinds of point go in. The breaklines decide where the mesh folds, and
    they come with the segments that have to survive. On top of those go the
    grid points the height model cannot do without, chosen by the same
    bisection error test the plain adaptive mesh uses -- flat ground contributes
    almost none of them, which is the whole point.
    """
    from .breaklines import build_breaklines
    from .cdt import WELD_M, triangulate, weld

    lines = build_breaklines(
        bbox,
        rings_by_source=rings_by_source,
        heights=heights,
        xs=xs,
        ys=ys,
        simplify_m=simplify_m,
        contour_interval_m=contour_interval_m,
    )

    # Grid points worth keeping for height alone. The adaptive mesh already
    # answers exactly that question, so its vertices are reused as a point set
    # rather than as a triangulation.
    height_mesh = build_rtin(heights, tolerance_m=tolerance_m)
    columns = height_mesh.vertices[:, 0].astype(int)
    rows = height_mesh.vertices[:, 1].astype(int)
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
    # triangulation. So triangulate, find the triangles that stray, drop a
    # point in each, and go round again. Two or three rounds is plenty: each
    # one roughly halves the worst gap.
    result = None
    for _ in range(MAX_REFINE_ROUNDS):
        result = triangulate(
            merged, segments, origin=tuple(bbox.center), skip_crossing=True
        )
        world = result.world_points()
        z = _bilinear(heights, xs, ys, world[:, 0], world[:, 1])
        extra = _straying_triangles(
            np.column_stack([world[:, 0], world[:, 1], z]),
            result.triangles,
            heights,
            xs,
            ys,
            tolerance_m,
        )
        if len(extra) == 0:
            break
        LOG.info("terrain mesh: %d more points to follow the ground", len(extra))
        before = len(merged)
        merged, mapping = weld(np.vstack([merged, extra]), WELD_M)
        if len(merged) == before:
            break
        segments = mapping[segments] if len(segments) else segments
        if len(segments):
            segments = segments[segments[:, 0] != segments[:, 1]]

    world = result.world_points()
    z = _bilinear(heights, xs, ys, world[:, 0], world[:, 1])
    vertices = np.column_stack([world[:, 0], world[:, 1], z])

    lost = result.missing_constraints()
    if lost:
        raise ValueError(
            f"{len(lost)} breaklines did not survive triangulation; the mesh "
            f"would not fold where the ground does"
        )

    error = _sampled_error(vertices, result.triangles, heights, xs, ys)
    mesh = ConstrainedMesh(
        vertices=vertices,
        triangles=result.triangles,
        breakline_edges=len(segments),
        height_points=len(height_points),
        tolerance_m=float(tolerance_m),
        max_error_m=error,
        counts=lines.counts,
    )
    LOG.info(
        "terrain mesh: %d vertices, %d triangles, %d breakline edges "
        "(worst height error %.3f m)",
        mesh.vertex_count,
        mesh.triangle_count,
        mesh.breakline_edges,
        mesh.max_error_m,
    )
    return mesh


def _straying_triangles(vertices, triangles, heights, xs, ys, tolerance):
    """Points to add where the mesh has drifted off the ground.

    Sampled at the centroid and the three edge midpoints rather than the
    centroid alone: a triangle spanning a ditch can pass at its centre and be
    a metre out halfway along a side.
    """
    if len(triangles) == 0:
        return np.zeros((0, 2))
    corners = vertices[triangles]
    centre = corners.mean(axis=1)
    # Every probe has to sit strictly inside its triangle. An edge midpoint
    # looks like the obvious place to check, but most edges here are
    # breaklines, and a point dropped exactly on a constraint is the one thing
    # the triangulator cannot then route around.
    probes = [centre]
    for i in range(3):
        probes.append((corners[:, i] + 2.0 * centre) / 3.0)

    wanted = []
    for probe in probes:
        truth = _bilinear(heights, xs, ys, probe[:, 0], probe[:, 1])
        gap = np.abs(probe[:, 2] - truth)
        pick = gap > tolerance
        if pick.any():
            wanted.append(probe[pick, :2])
    if not wanted:
        return np.zeros((0, 2))
    return np.vstack(wanted)


def _bilinear(grid, xs, ys, x, y):
    """Height at arbitrary points, from the grid, clamped to its edges."""
    from .elevation import _bilinear_on_grid

    return _bilinear_on_grid(np.asarray(grid, dtype=np.float64), xs, ys, x, y)


def _sampled_error(vertices, triangles, heights, xs, ys) -> float:
    """Worst gap between the mesh and the grid, measured at triangle centres.

    Only a sample -- checking every grid point against the triangle over it
    would cost more than building the mesh did -- but a triangle big enough to
    stray is big enough to stray at its centre.
    """
    if len(triangles) == 0:
        return 0.0
    corners = vertices[triangles]
    centre = corners.mean(axis=1)
    truth = _bilinear(heights, xs, ys, centre[:, 0], centre[:, 1])
    return float(np.abs(centre[:, 2] - truth).max())


def save_constrained_mesh(mesh: ConstrainedMesh, path, origin) -> "Path":
    """Write the mesh for the Blender stage, in local metres."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    local = mesh.vertices.copy()
    local[:, 0] -= origin[0]
    local[:, 1] -= origin[1]
    np.savez(
        path,
        vertices=local.astype(np.float64),
        triangles=mesh.triangles.astype(np.int32),
        tolerance_m=np.float64(mesh.tolerance_m),
        max_error_m=np.float64(mesh.max_error_m),
        grid_indexed=np.array(False),
    )
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
