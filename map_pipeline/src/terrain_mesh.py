"""The terrain as a mesh whose edges follow what is on the ground.

A plain grid ignores the outlines on the ground: the edge of a canal, a road
or a building falls wherever it falls inside a grid square, so the waterline
is a staircase of squares and none of it can be selected and moved. Here the
outlines are the mesh's edges, and the faces are built from them inwards:

* **Regions.** Road, water and building outlines cut the area into regions,
  each of one kind. Outlines closer than `CLEAN_M` are snapped together and
  every shared edge carries the same vertices on both sides.
* **Rows.** Every visible region -- land and water -- gets a row of quads
  along its outline, one grid spacing wide, with a rung at every station of
  the outline. Select an outline in Blender and the row beside it is the next
  loop in; that is the "move in from there".
* **Ladders.** Where a region is too narrow for a row on each side -- a
  front garden, a verge, a ditch -- the rungs run straight across instead, and
  the point each lands on is added to the outline on both sides of it.
* **Grid.** Inside the rows the regular grid carries on as before, the same
  squares at the same places.
* **Fill.** Between the rows and the grid's staircase edge the triangulator
  fills the strip, and the triangles are paired back into quads where the
  quad is well shaped.

The ground under roads and buildings is hidden, so those regions get no rows:
their outline is an edge loop, and inside it is grid and fill. Under water the
row runs from the waterline down to the bed, and a ditch too narrow for a row
gets a bed line down its middle.

Heights come from the survey (`TerrainResult.sample`), the waterline sits at
the water level and everything under water at the bed. Fitting the ground
under the roads happens afterwards, in `elevation.lower_under`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

LOG = logging.getLogger(__name__)

# What a stretch of ground is. A region is one kind; its outline is where the
# kind changes.
LAND, ROAD, WATER, BUILDING = 0, 1, 2, 3
KIND_NAMES = {LAND: "land", ROAD: "road", WATER: "water", BUILDING: "building"}
# Where a face sits in its region.
CORE, ROW, FILL = 0, 1, 2
PART_NAMES = {CORE: "grid", ROW: "row", FILL: "fill"}
# What an edge lies on, as bits; a vertex carries those of its edges.
WATERLINE, ROAD_EDGE, BUILDING_EDGE, ROW_EDGE, BED = 1, 2, 4, 8, 16
FEATURE_NAMES = {
    WATERLINE: "waterline",
    ROAD_EDGE: "road_edge",
    BUILDING_EDGE: "building_edge",
    ROW_EDGE: "row",
    BED: "bed",
}
# The sources a terrain can follow, as named in the config.
SOURCES = ("water", "roads", "buildings")

# Ground nobody sees. No rows inside: the outline is an edge loop, and the
# rest is grid and fill.
HIDDEN = (ROAD, BUILDING)

# Outlines are snap-rounded to this, so no two corners are closer.
SNAP_M = 0.1
# Corners and edges of different outlines closer than this become one. A
# building 30 cm from the kerb leaves a strip no face can be made of.
CLEAN_M = 0.4
# Everything inside a region meets on a centimetre grid.
GRID_SIZE = 0.01
# How far a simplified water or building outline may stray from the survey.
OUTLINE_SIMPLIFY_M = 0.2


@dataclass
class TerrainMesh:
    """The terrain as built: quads and triangles, outlines tagged."""

    vertices: np.ndarray  # (V, 3) RD easting, northing, NAP height as drawn
    # The height as surveyed, water beds applied: what the model shows before
    # the ground is fitted under the roads, and what Reset puts back.
    measured_z: np.ndarray
    face_sizes: np.ndarray  # (F,) 3 or 4
    # Vertex ids, counter-clockwise from above. A quad is drawn as two
    # triangles split from its first corner to its third, as Blender's fixed
    # split and the FBX export do.
    face_loops: np.ndarray
    face_kind: np.ndarray  # (F,) LAND, ROAD, WATER or BUILDING
    face_part: np.ndarray  # (F,) CORE, ROW or FILL
    vertex_on: np.ndarray  # (V,) bits of the outlines a vertex lies on
    edges: np.ndarray  # (E, 2) vertex ids of the edges along outlines and rows
    edge_on: np.ndarray  # (E,) bits
    grid_m: float = 2.0
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.face_sizes))

    @property
    def loop_starts(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.face_sizes)[:-1]]).astype(np.int64)

    def triangles(self) -> tuple[np.ndarray, np.ndarray]:
        """Every face as triangles: ``(T, 3)`` vertex ids and the face each is from."""
        starts = self.loop_starts
        loops = self.face_loops
        quads = np.flatnonzero(self.face_sizes == 4)
        tris = np.flatnonzero(self.face_sizes == 3)
        q = starts[quads]
        first = np.column_stack([loops[q], loops[q + 1], loops[q + 2]])
        second = np.column_stack([loops[q], loops[q + 2], loops[q + 3]])
        t = starts[tris]
        single = np.column_stack([loops[t], loops[t + 1], loops[t + 2]])
        out = np.vstack([first, second, single]).astype(np.int64)
        owner = np.concatenate([quads, quads, tris]).astype(np.int64)
        return out, owner

    def locator(self) -> "TriangleLocator":
        """Point location over the mesh's triangles, built once.

        Heights may change -- the ground is fitted under the roads after the
        mesh is made -- but where the triangles are does not.
        """
        cached = getattr(self, "_locator", None)
        if cached is None:
            triangles, _ = self.triangles()
            xy = self.vertices[:, :2]
            cached = TriangleLocator(
                xy, triangles, cell=max(self.grid_m, 0.5),
                bounds=(*xy.min(axis=0), *xy.max(axis=0)),
            )
            self._locator = cached
        return cached

    def heights_at(self, x, y) -> np.ndarray:
        """The mesh as drawn at RD coordinates, clamped into the area."""
        return self.locator().interpolate(self.vertices[:, 2], x, y)

    def to_npz_fields(self) -> dict:
        return {
            "vertices": self.vertices.astype(np.float64),
            "measured_z": self.measured_z.astype(np.float64),
            "face_sizes": self.face_sizes.astype(np.int32),
            "face_loops": self.face_loops.astype(np.int64),
            "face_kind": self.face_kind.astype(np.int8),
            "face_part": self.face_part.astype(np.int8),
            "vertex_on": self.vertex_on.astype(np.int32),
            "edges": self.edges.astype(np.int64),
            "edge_on": self.edge_on.astype(np.int32),
            "grid_m": np.float64(self.grid_m),
        }


class TriangleLocator:
    """Which triangle each point falls in, and where in it.

    A uniform grid of buckets, one grid square each: a square of the regular
    grid holds its own two triangles, and the rows and fill a handful. No
    tree, and nothing beyond numpy.
    """

    def __init__(self, xy: np.ndarray, tris: np.ndarray, *, cell: float, bounds):
        self.xy = np.asarray(xy, dtype=np.float64)
        self.tris = np.asarray(tris, dtype=np.int64)
        self.cell = float(cell)
        self.x0, self.y0, x1, y1 = (float(v) for v in bounds)
        self.x1, self.y1 = x1, y1
        self.nx = max(1, int(np.ceil((x1 - self.x0) / self.cell)))
        self.ny = max(1, int(np.ceil((y1 - self.y0) / self.cell)))
        a = self.xy[self.tris[:, 0]]
        b = self.xy[self.tris[:, 1]]
        c = self.xy[self.tris[:, 2]]
        self.a = a
        self.v0 = b - a
        self.v1 = c - a
        self.d00 = np.einsum("ij,ij->i", self.v0, self.v0)
        self.d01 = np.einsum("ij,ij->i", self.v0, self.v1)
        self.d11 = np.einsum("ij,ij->i", self.v1, self.v1)
        denom = self.d00 * self.d11 - self.d01 * self.d01
        self.ok = np.abs(denom) > 1e-12
        self.inv = np.where(self.ok, 1.0 / np.where(self.ok, denom, 1.0), 0.0)

        corners = np.stack([a, b, c], axis=1)
        lo = np.floor((corners.min(axis=1) - [self.x0, self.y0]) / self.cell).astype(np.int64)
        hi = np.floor((corners.max(axis=1) - [self.x0, self.y0]) / self.cell).astype(np.int64)
        lo[:, 0] = np.clip(lo[:, 0], 0, self.nx - 1)
        hi[:, 0] = np.clip(hi[:, 0], 0, self.nx - 1)
        lo[:, 1] = np.clip(lo[:, 1], 0, self.ny - 1)
        hi[:, 1] = np.clip(hi[:, 1], 0, self.ny - 1)
        wide = hi[:, 0] - lo[:, 0] + 1
        tall = hi[:, 1] - lo[:, 1] + 1
        counts = wide * tall
        owner = np.repeat(np.arange(len(self.tris)), counts)
        start = np.cumsum(counts) - counts
        k = np.arange(int(counts.sum())) - np.repeat(start, counts)
        w = np.repeat(wide, counts)
        bx = np.repeat(lo[:, 0], counts) + k % w
        by = np.repeat(lo[:, 1], counts) + k // w
        bucket = by * self.nx + bx
        order = np.argsort(bucket, kind="stable")
        self.bucket_tris = owner[order]
        self.bucket_start = np.searchsorted(
            bucket[order], np.arange(self.nx * self.ny + 1)
        )

    def locate(self, x, y, chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
        """``(triangle, weights)`` per point; triangle -1 where none holds it.

        Points are clamped into the bounds first, as the grid sampler does,
        so the edge of the area reads the edge of the terrain.
        """
        x = np.clip(np.asarray(x, dtype=np.float64).ravel(), self.x0, self.x1)
        y = np.clip(np.asarray(y, dtype=np.float64).ravel(), self.y0, self.y1)
        n = len(x)
        tri = np.full(n, -1, dtype=np.int64)
        weights = np.zeros((n, 3))
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            t, w = self._locate(x[begin:end], y[begin:end])
            tri[begin:end] = t
            weights[begin:end] = w
        return tri, weights

    def _locate(self, x, y):
        n = len(x)
        bx = np.clip(np.floor((x - self.x0) / self.cell).astype(np.int64), 0, self.nx - 1)
        by = np.clip(np.floor((y - self.y0) / self.cell).astype(np.int64), 0, self.ny - 1)
        bucket = by * self.nx + bx
        start = self.bucket_start[bucket]
        count = self.bucket_start[bucket + 1] - start
        point = np.repeat(np.arange(n), count)
        offset = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
        cand = self.bucket_tris[np.repeat(start, count) + offset]
        px = x[point] - self.a[cand, 0]
        py = y[point] - self.a[cand, 1]
        d20 = px * self.v0[cand, 0] + py * self.v0[cand, 1]
        d21 = px * self.v1[cand, 0] + py * self.v1[cand, 1]
        inv = self.inv[cand]
        u = (self.d11[cand] * d20 - self.d01[cand] * d21) * inv
        v = (self.d00[cand] * d21 - self.d01[cand] * d20) * inv
        w0 = 1.0 - u - v
        score = np.minimum(np.minimum(w0, u), v)
        score = np.where(self.ok[cand], score, -np.inf)
        tri = np.full(n, -1, dtype=np.int64)
        weights = np.zeros((n, 3))
        if not len(point):
            return tri, weights
        # The candidate a point is most inside of: on a shared edge either
        # triangle gives the same height.
        best = np.full(n, -np.inf)
        np.maximum.at(best, point, score)
        winner = (score == best[point]) & (score > -1e-6)
        first = np.unique(point[winner], return_index=True)[1]
        pick = np.flatnonzero(winner)[first]
        tri[point[pick]] = cand[pick]
        weights[point[pick]] = np.column_stack([w0[pick], u[pick], v[pick]])
        return tri, weights

    def interpolate(self, z: np.ndarray, x, y) -> np.ndarray:
        """Heights `z` (per vertex) at the points, NaN where no triangle is."""
        tri, w = self.locate(x, y)
        out = np.full(len(tri), np.nan)
        found = tri >= 0
        corners = self.tris[tri[found]]
        out[found] = (w[found] * np.asarray(z, dtype=np.float64)[corners]).sum(axis=1)
        return out


# ---------------------------------------------------------------------------
# Building it
# ---------------------------------------------------------------------------


def build_terrain_mesh(
    terrain,
    *,
    roads=None,
    water=(),
    buildings=None,
    sources=SOURCES,
    water_depth_m: float = 1.2,
    row_m: float | None = None,
) -> TerrainMesh:
    """The terrain mesh for `terrain`, following the outlines in `sources`.

    `roads` is the RoadMesh, `water` the WaterBody list and `buildings` the
    BuildingSet; any of them may be missing. With no outlines at all the mesh
    is the plain grid. If the outlines defeat the geometry -- they should
    not, but survey data is survey data -- the run carries on with the plain
    grid and says so, rather than failing at the terrain.
    """
    started = time.time()
    grid_m = float(terrain.grid_m)
    row_m = float(row_m) if row_m else grid_m
    sources = {str(s).strip().lower() for s in (sources or ())}
    try:
        layers, bodies = _layers(
            terrain,
            roads=roads if "roads" in sources else None,
            water=water if "water" in sources else (),
            buildings=buildings if "buildings" in sources else None,
        )
        if not layers:
            mesh = grid_mesh(terrain)
        else:
            mesh = _feature_mesh(
                terrain, layers, bodies, row_m=row_m, water_depth_m=water_depth_m
            )
    except Exception as exc:  # noqa: BLE001 - the plain grid is always possible
        LOG.warning(
            "terrain: the outlines could not be meshed (%s); using the plain grid",
            exc,
        )
        mesh = grid_mesh(terrain)
        mesh.stats["fallback"] = str(exc)[:200]
    mesh.stats["seconds"] = round(time.time() - started, 1)
    quads = int((mesh.face_sizes == 4).sum())
    LOG.info(
        "terrain mesh: %d faces (%d%% quads), %d vertices; %s",
        len(mesh),
        round(100.0 * quads / max(len(mesh), 1)),
        len(mesh.vertices),
        ", ".join(
            f"{v} {PART_NAMES[k]}"
            for k, v in sorted(
                {int(k): int((mesh.face_part == k).sum()) for k in np.unique(mesh.face_part)}.items()
            )
        ),
    )
    return mesh


def grid_mesh(terrain) -> TerrainMesh:
    """The plain grid as a TerrainMesh: every square a quad, nothing tagged."""
    xs, ys = terrain.xs, terrain.ys
    n = len(xs)
    gx, gy = np.meshgrid(xs, ys)
    heights = np.asarray(terrain.measured if terrain.measured is not None else terrain.heights,
                         dtype=np.float64)
    vertices = np.column_stack([gx.ravel(), gy.ravel(), heights.ravel()])
    cc, rr = np.meshgrid(np.arange(n - 1), np.arange(n - 1))
    base = (rr * n + cc).ravel()
    loops = np.column_stack([base, base + 1, base + n + 1, base + n]).ravel()
    faces = (n - 1) ** 2
    return TerrainMesh(
        vertices=vertices,
        measured_z=vertices[:, 2].copy(),
        face_sizes=np.full(faces, 4, dtype=np.int32),
        face_loops=loops.astype(np.int64),
        face_kind=np.zeros(faces, dtype=np.int8),
        face_part=np.zeros(faces, dtype=np.int8),
        vertex_on=np.zeros(len(vertices), dtype=np.int32),
        edges=np.zeros((0, 2), dtype=np.int64),
        edge_on=np.zeros(0, dtype=np.int32),
        grid_m=float(terrain.grid_m),
        stats={"faces": faces, "quad_share": 1.0, "outlines": 0},
    )


def _layers(terrain, *, roads, water, buildings):
    """The outline layers in RD, first winning, plus the water bodies."""
    import shapely

    layers = []
    if roads is not None and len(roads):
        from .roads import SURFACE

        starts = roads.loop_starts
        keep = np.flatnonzero((roads.face_kind == SURFACE) & (roads.face_level <= 0))
        polygons = [
            shapely.Polygon(roads.vertices[roads.face_loops[starts[i]: starts[i] + roads.face_sizes[i]], :2])
            for i in keep
        ]
        if polygons:
            layers.append((ROAD, _polygonal(shapely.union_all(
                shapely.make_valid(np.array(polygons, dtype=object))))))
    bodies = []
    if water:
        polygons = []
        for body in water:
            rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in body.rings]
            rings = [r for r in rings if len(r) >= 4]
            if not rings:
                continue
            polygon = _polygonal(shapely.make_valid(shapely.Polygon(rings[0], rings[1:])))
            if polygon.is_empty:
                continue
            polygon = polygon.simplify(OUTLINE_SIMPLIFY_M)
            polygons.append(polygon)
            bodies.append((polygon, float(body.level_nap)))
        if polygons:
            layers.append((WATER, _polygonal(shapely.union_all(np.array(polygons, dtype=object)))))
    if buildings is not None:
        roofs = [b.roof_tris for b in getattr(buildings, "buildings", []) if len(b.roof_tris)]
        if roofs:
            tris = np.concatenate(roofs)[:, :, :2]
            polygons = shapely.polygons(tris)
            polygons = polygons[shapely.area(polygons) > 1e-6]
            if len(polygons):
                footprint = _polygonal(shapely.union_all(shapely.make_valid(polygons)))
                layers.append((BUILDING, footprint.simplify(OUTLINE_SIMPLIFY_M)))
    return [(k, g) for k, g in layers if not g.is_empty], bodies


def _polygonal(geometry):
    """Only the polygons of a geometry, as one (Multi)Polygon."""
    import shapely

    if geometry is None:
        return shapely.Polygon()
    parts = _parts(shapely.make_valid(geometry))
    if not parts:
        return shapely.Polygon()
    return shapely.union_all(np.array(parts, dtype=object))


def _parts(geometry) -> list:
    """Every polygon in a geometry, however deeply it is nested.

    `make_valid` can hand back a collection holding a MultiPolygon beside a few
    stray lines. Taking only the collection's direct Polygon members threw the
    whole MultiPolygon away: one land region over Bronbeek lost its entire
    grid that way and was filled with faces forty metres across.
    """
    import shapely

    out = []
    stack = [geometry]
    while stack:
        item = stack.pop()
        if item is None or item.is_empty:
            continue
        kind = item.geom_type
        if kind == "Polygon":
            out.append(item)
        elif kind in ("MultiPolygon", "GeometryCollection"):
            stack.extend(reversed(list(shapely.get_parts(item))))
    return out


def _feature_mesh(terrain, layers, bodies, *, row_m: float, water_depth_m: float) -> TerrainMesh:
    import shapely

    xs_rd, ys_rd = terrain.xs, terrain.ys
    grid_m = float(terrain.grid_m)
    # Local coordinates from the first grid node: the grid nodes then sit
    # exactly on the snapping grid, and the numbers stay small.
    ox, oy = float(xs_rd[0]), float(ys_rd[0])
    xs = np.round(xs_rd - ox, 6)
    ys = np.round(ys_rd - oy, 6)
    shift = np.array([ox, oy])
    layers = [(k, shapely.transform(g, lambda c: c - shift)) for k, g in layers]
    bodies = [(shapely.transform(g, lambda c: c - shift), level) for g, level in bodies]
    frame = shapely.segmentize(
        shapely.box(0.0, 0.0, float(xs[-1]), float(ys[-1])), grid_m * 1.0001
    )
    outer = shapely.difference(frame.buffer(4.0 * row_m, join_style="mitre"), frame)
    station_m = 1.6 * grid_m
    margin_m = 0.35 * grid_m

    timings = {}
    t = time.time()
    parts = _regions(frame, layers, station_m=station_m)
    timings["regions_s"] = round(time.time() - t, 1)
    t = time.time()
    parts, across = _cross_rungs(parts, frame, row_m=row_m, margin_m=margin_m)
    timings["cross_rungs_s"] = round(time.time() - t, 1)

    n = len(xs)
    taken = np.zeros((n - 1, n - 1), dtype=bool)
    cell_kind = np.zeros((n - 1, n - 1), dtype=np.int64)
    faces = []  # (coords local (k, 2), part, kind, region index)
    beds = []  # (2, 2) local segments along the bed of a ditch
    t = time.time()
    failed = 0
    for index, (region, kind) in enumerate(parts):
        before = taken.copy() if kind not in HIDDEN else None
        try:
            out, cells, inset, bed = _mesh_region(
                region, kind, frame, outer, xs, ys, taken,
                row_m=row_m, margin_m=margin_m, across=across.get(index, ()),
            )
        except Exception as exc:  # noqa: BLE001 - one region, not the terrain
            # Its outline is shared with its neighbours, so it is meshed all
            # the same, only without rows: grid and fill, like a hidden one.
            LOG.debug("terrain: region %d meshed without rows: %s", index, exc)
            failed += 1
            if across.get(index) or before is None:
                raise
            # The squares the failed attempt claimed are free again.
            taken[:] = before
            out, cells, inset, bed = _mesh_region(
                region, ROAD, frame, outer, xs, ys, taken,
                row_m=row_m, margin_m=margin_m,
            )
            out = [(c, ROW if p == FILL else p) for c, p in out]
        faces.extend((c, p, kind, index) for c, p in out)
        if len(cells):
            cell_kind[cells[:, 1], cells[:, 0]] = kind
        beds.extend(bed)
    timings["regions_meshed_s"] = round(time.time() - t, 1)
    if failed:
        LOG.info("terrain: %d regions meshed without rows", failed)

    # Assemble: every corner of every face, then the grid squares, one vertex
    # per position.
    t = time.time()
    core = np.argwhere(taken.T)  # (i, j) per core square
    corner_lists = [c for c, _, _, _ in faces]
    sizes = np.array([len(c) for c in corner_lists] + [4] * len(core), dtype=np.int64)
    if len(core):
        i, j = core[:, 0], core[:, 1]
        grid_corners = np.stack(
            [
                np.column_stack([xs[i], ys[j]]),
                np.column_stack([xs[i + 1], ys[j]]),
                np.column_stack([xs[i + 1], ys[j + 1]]),
                np.column_stack([xs[i], ys[j + 1]]),
            ],
            axis=1,
        ).reshape(-1, 2)
    else:
        grid_corners = np.zeros((0, 2))
    all_corners = np.vstack([np.vstack(corner_lists) if corner_lists else np.zeros((0, 2)), grid_corners])
    keys = np.rint(all_corners * 1000.0).astype(np.int64)
    unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    local_xy = all_corners[first]
    face_kind = np.array([k for _, _, k, _ in faces], dtype=np.int64)
    face_part = np.array([p for _, p, _, _ in faces], dtype=np.int64)
    if len(core):
        face_kind = np.concatenate([face_kind, cell_kind[core[:, 1], core[:, 0]]])
        face_part = np.concatenate([face_part, np.full(len(core), CORE)])
    loops = inverse.astype(np.int64)
    timings["assemble_s"] = round(time.time() - t, 1)

    # Drop faces that collapsed onto a repeated vertex, and the vertices
    # nothing uses any more.
    keep_face = _distinct_corners(loops, sizes)
    if not keep_face.all():
        loops = loops[np.repeat(keep_face, sizes)]
        sizes = sizes[keep_face]
        face_kind = face_kind[keep_face]
        face_part = face_part[keep_face]
    used = np.unique(loops)
    if len(used) < len(local_xy):
        remap = np.full(len(local_xy), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        loops = remap[loops]
        local_xy = local_xy[used]
        unique = unique[used]

    xy = local_xy + shift
    heights = np.asarray(terrain.sample(xy[:, 0], xy[:, 1]), dtype=np.float64)
    # Off the grid nodes, read the survey itself: a vertex on an outline or in
    # a row sits on the bank or the edge it follows, and the grid between its
    # nodes is a blur of exactly that.
    off_grid = ~_on_grid_nodes(local_xy, grid_m)
    surveyed = _survey_heights(terrain, xy[off_grid])
    if surveyed is not None:
        heights[off_grid] = np.where(np.isfinite(surveyed), surveyed, heights[off_grid])
    heights = _water_heights(
        heights, loops, sizes, face_kind, xy, bodies, shift, water_depth_m
    )
    vertices = np.column_stack([xy, heights])

    sizes, loops = _orient(vertices, sizes, loops, face_part, face_kind, terrain)
    edges, edge_on, vertex_on = _outline_edges(
        loops, sizes, face_kind, face_part, len(vertices), beds, unique, shift
    )
    quads = int((sizes == 4).sum())
    stats = {
        "faces": int(len(sizes)),
        "quad_share": round(quads / max(len(sizes), 1), 3),
        "regions": {KIND_NAMES[k]: int(sum(1 for _, kk in parts if kk == k)) for k in KIND_NAMES},
        "faces_by_part": {PART_NAMES[k]: int((face_part == k).sum()) for k in PART_NAMES},
        "quad_share_by_part": {
            PART_NAMES[k]: round(float((sizes[face_part == k] == 4).mean()), 3)
            for k in PART_NAMES if (face_part == k).any()
        },
        "cross_rungs": int(sum(len(v) for v in across.values())),
        "outline_edges": {
            name: int(((edge_on & bit) > 0).sum()) for bit, name in FEATURE_NAMES.items()
        },
        **timings,
    }
    return TerrainMesh(
        vertices=vertices,
        measured_z=heights.copy(),
        face_sizes=sizes.astype(np.int32),
        face_loops=loops.astype(np.int64),
        face_kind=face_kind.astype(np.int8),
        face_part=face_part.astype(np.int8),
        vertex_on=vertex_on,
        edges=edges,
        edge_on=edge_on,
        grid_m=grid_m,
        stats=stats,
    )


def _on_grid_nodes(local_xy: np.ndarray, grid_m: float) -> np.ndarray:
    """Whether each local point is a node of the regular grid."""
    f = local_xy / grid_m
    return (np.abs(f - np.rint(f)) < 1e-6).all(axis=1)


def _survey_heights(terrain, xy: np.ndarray, half_m: float = 0.75):
    """The mean of the surveyed pixels within `half_m` of each point.

    NaN where none of them was measured -- under a building, over water --
    and None when the survey raster is not to hand.
    """
    from pathlib import Path

    path = getattr(terrain, "geotiff_path", None)
    if path is None or not Path(path).is_file() or not len(xy):
        return None
    import rasterio

    from .elevation import MAX_PLAUSIBLE_NAP, MIN_PLAUSIBLE_NAP, NODATA_CUTOFF

    with rasterio.open(path) as dataset:
        raw = dataset.read(1).astype(np.float64)
        left, bottom, right, top = dataset.bounds
    rows, cols = raw.shape
    valid = (
        np.isfinite(raw) & (raw < NODATA_CUTOFF)
        & (raw > MIN_PLAUSIBLE_NAP) & (raw < MAX_PLAUSIBLE_NAP)
    )
    values = np.where(valid, raw, 0.0)
    total = np.zeros((rows + 1, cols + 1))
    count = np.zeros((rows + 1, cols + 1))
    total[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    count[1:, 1:] = valid.astype(np.float64).cumsum(axis=0).cumsum(axis=1)
    size_x = (right - left) / cols
    size_y = (top - bottom) / rows
    c0 = np.clip(np.ceil((xy[:, 0] - half_m - left) / size_x - 0.5), 0, cols).astype(np.int64)
    c1 = np.clip(np.floor((xy[:, 0] + half_m - left) / size_x - 0.5) + 1, 0, cols).astype(np.int64)
    r0 = np.clip(np.ceil((top - (xy[:, 1] + half_m)) / size_y - 0.5), 0, rows).astype(np.int64)
    r1 = np.clip(np.floor((top - (xy[:, 1] - half_m)) / size_y - 0.5) + 1, 0, rows).astype(np.int64)
    c1 = np.maximum(c1, c0)
    r1 = np.maximum(r1, r0)
    s = total[r1, c1] - total[r0, c1] - total[r1, c0] + total[r0, c0]
    n = count[r1, c1] - count[r0, c1] - count[r1, c0] + count[r0, c0]
    return np.where(n > 0, s / np.maximum(n, 1.0), np.nan)


def _distinct_corners(loops: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Per face, whether its corners are all different vertices."""
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    keep = np.ones(len(sizes), dtype=bool)
    for size in (3, 4):
        faces = np.flatnonzero(sizes == size)
        if not len(faces):
            continue
        ring = loops[starts[faces][:, None] + np.arange(size)[None, :]]
        repeated = np.zeros(len(faces), dtype=bool)
        for i in range(size):
            for j in range(i + 1, size):
                repeated |= ring[:, i] == ring[:, j]
        keep[faces[repeated]] = False
    return keep


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


def _regions(frame, layers, *, station_m: float, sliver_m: float = 0.3) -> list:
    """Partition `frame` into ``(polygon, kind)``, the first layer winning."""
    import shapely

    layers = [(k, _polygonal(shapely.intersection(_polygonal(g), frame))) for k, g in layers]
    lines = [frame.boundary] + [g.boundary for _, g in layers if not g.is_empty]
    noded = shapely.union_all(np.array(lines, dtype=object))
    faces = np.array(_parts(shapely.polygonize(shapely.get_parts(noded))), dtype=object)
    faces = faces[shapely.area(faces) > 1e-6]
    reps = shapely.point_on_surface(faces)
    kind = np.full(len(faces), LAND, dtype=np.int64)
    done = np.zeros(len(faces), dtype=bool)
    for k, geometry in layers:
        if geometry.is_empty:
            continue
        shapely.prepare(geometry)
        inside = shapely.contains(geometry, reps) & ~done
        kind[inside] = k
        done |= inside

    # A sliver goes to the neighbour it shares most outline with.
    area = shapely.area(faces)
    perimeter = shapely.length(faces)
    thin = (2.0 * area / np.maximum(perimeter, 1e-9) < sliver_m) | (area < sliver_m ** 2)
    tree = shapely.STRtree(faces)
    for _ in range(3):
        changed = 0
        for i in np.flatnonzero(thin):
            best, best_len = None, 0.0
            for j in tree.query(faces[i], predicate="intersects"):
                if j == i or thin[j]:
                    continue
                shared = shapely.length(shapely.intersection(faces[i].boundary, faces[j].boundary))
                if shared > best_len:
                    best, best_len = j, shared
            if best is not None:
                changed += int(kind[i] != kind[best])
                kind[i] = kind[best]
                thin[i] = False
        if not changed:
            break

    by_kind = {int(k): shapely.union_all(faces[kind == k]) for k in np.unique(kind)}
    outline = shapely.union_all(
        np.array([g.boundary for g in by_kind.values()] + [frame.boundary], dtype=object),
        grid_size=SNAP_M,
    )
    final = np.array(_parts(shapely.polygonize(shapely.get_parts(outline))), dtype=object)
    final = final[shapely.area(final) > 1e-6]
    reps = shapely.point_on_surface(final)
    final_kind = np.full(len(final), LAND, dtype=np.int64)
    for k, geometry in by_kind.items():
        shapely.prepare(geometry)
        final_kind[shapely.contains(geometry, reps)] = k
    # Corners of different outlines closer than CLEAN_M become one, and every
    # shared edge identical on both sides.
    final = shapely.coverage_clean(final, gap_width=CLEAN_M, snapping_distance=CLEAN_M)
    final = _fill_gaps(final, frame)
    # Stations along every outline, then everything on the centimetre grid.
    final = np.array([_with_stations(p, station_m) for p in final], dtype=object)
    final = shapely.set_precision(final, GRID_SIZE)
    final = shapely.orient_polygons(final)
    out = []
    for polygon, k in zip(final, final_kind.tolist()):
        if polygon is None or polygon.is_empty:
            continue
        for piece in _parts(polygon):
            if piece.area > 1e-4:
                out.append((piece, int(k)))
    return out


def _fill_gaps(polygons, frame):
    """Give any sliver of the area no region covers to its best neighbour.

    Snapping corners together can pull a vertex off the edge of the area, and
    the sliver it leaves between the regions and the bbox edge is in nobody's
    outline: a hole in the terrain.
    """
    import shapely

    polygons = np.array([p if p is not None else shapely.Polygon() for p in polygons], dtype=object)
    covered = shapely.union_all(polygons[~shapely.is_empty(polygons)])
    gaps = [g for g in _parts(shapely.difference(frame, covered)) if g.area > 1e-6]
    if not gaps:
        return polygons
    tree = shapely.STRtree(polygons)
    for gap in gaps:
        best, best_len = None, 0.0
        for j in tree.query(gap, predicate="intersects"):
            if polygons[j].is_empty:
                continue
            shared = shapely.length(shapely.intersection(gap.boundary, polygons[j].boundary))
            if shared > best_len:
                best, best_len = j, shared
        if best is not None:
            polygons[best] = _polygonal(shapely.union(polygons[best], gap))
    return polygons


def _with_stations(polygon, station_m: float):
    """A polygon with its edges cut into equal parts no longer than `station_m`.

    Each edge is cut from the same end whichever way a ring runs along it, so
    the two regions sharing an edge compute bit-identical stations. Cut from
    opposite ends, a station falling on a half centimetre -- which the 10 cm
    snapping makes common -- rounded one way on one side of the outline and
    the other way on the other, and left a crack between them.
    """
    import shapely

    if polygon is None or polygon.is_empty:
        return polygon
    if polygon.geom_type == "MultiPolygon":
        return shapely.MultiPolygon([_with_stations(p, station_m) for p in polygon.geoms])
    if polygon.geom_type != "Polygon":
        return polygon
    rings = []
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords)[:, :2]
        out = [coords[0]]
        for a, b in zip(coords[:-1], coords[1:]):
            length = float(np.hypot(*(b - a)))
            count = int(np.ceil(length / station_m - 1e-9))
            if count > 1:
                forward = (a[0], a[1]) < (b[0], b[1])
                lo, hi = (a, b) if forward else (b, a)
                steps = lo + (hi - lo) * (np.arange(1, count) / count)[:, None]
                out.extend(steps if forward else steps[::-1])
            out.append(b)
        rings.append(np.array(out))
    return shapely.Polygon(rings[0], rings[1:])


def _ring_frames(coords: np.ndarray):
    """Per vertex of a ring: inward bisector, its cosine to the normals, normals."""
    previous = np.roll(coords, 1, axis=0)
    following = np.roll(coords, -1, axis=0)
    d_in = coords - previous
    d_out = following - coords
    d_in = d_in / np.maximum(np.linalg.norm(d_in, axis=1), 1e-12)[:, None]
    d_out = d_out / np.maximum(np.linalg.norm(d_out, axis=1), 1e-12)[:, None]
    # An oriented polygon has its inside on the left of every ring.
    n_in = np.column_stack([-d_in[:, 1], d_in[:, 0]])
    n_out = np.column_stack([-d_out[:, 1], d_out[:, 0]])
    bisector = n_in + n_out
    bisector = bisector / np.maximum(np.linalg.norm(bisector, axis=1), 1e-12)[:, None]
    cos_half = np.einsum("ij,ij->i", bisector, n_in)
    turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", d_in, d_out), -1.0, 1.0)))
    return bisector, cos_half, n_in, n_out, turn


def _on_frame(points: np.ndarray, bounds, tol: float = 1e-6) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return (
        (np.abs(points[:, 0] - x0) < tol) | (np.abs(points[:, 0] - x1) < tol)
        | (np.abs(points[:, 1] - y0) < tol) | (np.abs(points[:, 1] - y1) < tol)
    )


def _key(point) -> tuple[int, int]:
    return (int(round(float(point[0]) * 1000.0)), int(round(float(point[1]) * 1000.0)))


def _snap(points: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(points, dtype=np.float64) / GRID_SIZE) * GRID_SIZE


def _cross_rungs(parts, frame, *, row_m: float, margin_m: float, snap_m: float = 0.6):
    """Rungs straight across the visible regions too narrow for two rows.

    Returns the regions with the far end of each such rung added to the
    outline it lands on -- on both regions sharing that edge, so neither has
    a vertex the other lacks -- and the rungs per region index.
    """
    import shapely

    reach = 2.0 * row_m + 2.0 * margin_m
    bounds = frame.bounds
    found: dict[int, list] = {}
    for index, (polygon, kind) in enumerate(parts):
        if kind in HIDDEN:
            continue
        rings = [np.asarray(r.coords)[:-1, :2] for r in [polygon.exterior, *polygon.interiors]]
        starts, directions = [], []
        for coords in rings:
            if len(coords) < 3:
                continue
            bisector, cos_half, _, _, _ = _ring_frames(coords)
            on_frame = _on_frame(coords, bounds)
            usable = ~on_frame & (cos_half >= 0.5)
            starts.append(coords[usable])
            directions.append(bisector[usable])
        if not starts:
            continue
        starts = np.vstack(starts)
        directions = np.vstack(directions)
        if not len(starts):
            continue
        rays = shapely.linestrings(
            np.stack([starts + 1e-3 * directions, starts + reach * directions], axis=1)
        )
        hits = shapely.intersection(rays, polygon.boundary)
        stations = np.vstack(rings)
        tree = shapely.STRtree(shapely.points(stations))
        candidates = []
        for k, hit in enumerate(hits):
            if hit is None or hit.is_empty:
                continue
            points = shapely.get_coordinates(hit)
            distance = np.linalg.norm(points - starts[k], axis=1)
            j = int(np.argmin(distance))
            end = points[j]
            if _on_frame(end[None, :], bounds)[0]:
                continue
            near = tree.query_nearest(shapely.Point(end), max_distance=snap_m)
            snapped = len(near) > 0
            if snapped:
                end = stations[int(near[0])]
            candidates.append((0 if snapped else 1, float(distance[j]), starts[k], end, snapped))
        if candidates:
            found[index] = candidates

    rungs: dict[int, list] = {}
    inserts: list[np.ndarray] = []
    boundaries = {index: parts[index][0].boundary for index in found}
    for index, candidates in found.items():
        # Ones that land on a station first, then the shortest; a rung that
        # crosses or crowds one already kept is dropped.
        candidates.sort(key=lambda c: (c[0], c[1]))
        seen = set()
        kept, ends = [], []
        for _, _, start, end, snapped in candidates:
            key = tuple(sorted((_key(start), _key(end))))
            if key in seen or key[0] == key[1]:
                continue
            segment = shapely.LineString([start, end])
            if kept:
                if shapely.crosses(np.array(kept, dtype=object), segment).any():
                    continue
                prior = np.array(ends)
                if (np.linalg.norm(prior - start, axis=1) < 0.5).any() or (
                    np.linalg.norm(prior - end, axis=1) < 0.5
                ).any():
                    continue
            if _grazes([segment], boundaries[index])[0]:
                continue
            seen.add(key)
            kept.append(segment)
            ends.extend([start, end])
            if not snapped:
                inserts.append(end)
        rungs[index] = kept

    if inserts:
        # Exactly where each rung meets the edge: both regions sharing the
        # edge then round the same point to the same place when they node.
        parts = _insert_points(parts, np.array(inserts))
    return parts, rungs


def _insert_points(parts, points: np.ndarray):
    """Every region with `points` added to the outline edges they lie on."""
    import shapely

    tree = shapely.STRtree(shapely.points(points))
    out = []
    for polygon, kind in parts:
        rings = []
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = np.asarray(ring.coords)[:, :2]
            segments = shapely.linestrings(np.stack([coords[:-1], coords[1:]], axis=1))
            seg, point = tree.query(segments, predicate="dwithin", distance=1e-4)
            if not len(seg):
                rings.append(coords)
                continue
            extra: dict[int, list] = {}
            for a, b in zip(seg, point):
                extra.setdefault(int(a), []).append(points[int(b)])
            merged = []
            for k in range(len(coords) - 1):
                merged.append(coords[k])
                if k in extra:
                    q = np.array(extra[k])
                    along = np.linalg.norm(q - coords[k], axis=1)
                    length = float(np.linalg.norm(coords[k + 1] - coords[k]))
                    for m in np.argsort(along):
                        if 0.01 < along[m] < length - 0.01:
                            merged.append(q[m])
            merged.append(coords[-1])
            rings.append(np.array(merged))
        out.append((shapely.Polygon(rings[0], rings[1:]), kind))
    return out


# ---------------------------------------------------------------------------
# One region
# ---------------------------------------------------------------------------


def _grazes(lines, boundary, margin: float = 0.02) -> np.ndarray:
    """Per line, whether its middle comes within `margin` of the outline.

    A rung straight across a narrow part meets the outline at its ends only.
    Snapped onto a corner near where it lands, it can end up running along
    the outline instead, and then whatever it brings -- under water, the
    point of the bed at its middle -- goes into the outline on this side and
    not the other: a crack. Over Bronbeek, 24 of them.
    """
    import shapely

    lines = np.asarray(lines, dtype=object)
    if not len(lines):
        return np.zeros(0, dtype=bool)
    length = shapely.length(lines)
    trim = np.minimum(0.05, 0.25 * length)
    middle = shapely.line_merge(shapely.linestrings(np.stack([
        shapely.get_coordinates(shapely.line_interpolate_point(lines, trim)),
        shapely.get_coordinates(shapely.line_interpolate_point(lines, length - trim)),
    ], axis=1)))
    return shapely.dwithin(middle, boundary, margin)


def _rows(region, frame, outer, row_m: float, *, opening: float):
    """The inner outline of a region's row, and the rungs out to it."""
    import shapely

    touches = region.intersects(frame.boundary)
    # Past the edge of the area the region carries on, so no row runs along
    # the bbox: nothing there is an outline.
    extended = shapely.union(region, outer) if touches else region
    inset = extended.buffer(-row_m, join_style="mitre", mitre_limit=2.5)
    if opening > 0 and not inset.is_empty:
        inset = inset.buffer(-opening, join_style="mitre", mitre_limit=2.5).buffer(
            opening, join_style="mitre", mitre_limit=2.5
        )
    if touches and not inset.is_empty:
        inset = shapely.intersection(inset, frame)
    pieces = [p.simplify(0.05) for p in _parts(inset) if p.area >= 0.5 * row_m * row_m]
    if not pieces:
        return shapely.Polygon(), []
    inset = shapely.set_precision(shapely.union_all(np.array(pieces, dtype=object)), GRID_SIZE)
    if inset.is_empty:
        return inset, []
    boundary = inset.boundary
    shapely.prepare(boundary)
    corners = shapely.get_coordinates(boundary)
    corner_tree = shapely.STRtree(shapely.points(corners))

    bounds = frame.bounds
    candidates = []
    for ring in [region.exterior, *region.interiors]:
        coords = np.asarray(ring.coords)[:-1, :2]
        if len(coords) < 3:
            continue
        bisector, cos_half, n_in, n_out, turn = _ring_frames(coords)
        on_frame = _on_frame(coords, bounds)
        arriving = on_frame & np.roll(on_frame, 1)
        leaving = on_frame & np.roll(on_frame, -1)
        for k in range(len(coords)):
            if arriving[k] and leaving[k]:
                continue  # along the bbox edge
            if arriving[k]:
                direction, length = n_out[k], row_m
            elif leaving[k]:
                direction, length = n_in[k], row_m
            else:
                if cos_half[k] < 0.4:
                    continue  # too sharp a corner for a rung of its own
                direction, length = bisector[k], row_m / cos_half[k]
            candidates.append((turn[k], coords[k], coords[k] + direction * length))
    if not candidates:
        return inset, []
    starts = np.array([c[1] for c in candidates])
    ends = np.array([c[2] for c in candidates])
    end_points = shapely.points(ends)
    on_inset = shapely.distance(end_points, boundary) < 0.15
    # On the inner outline exactly, and on its corner where one is that close.
    projected = shapely.get_coordinates(shapely.shortest_line(end_points, boundary))[1::2]
    ends = np.where(on_inset[:, None], projected, ends)
    hit, _ = corner_tree.query_nearest(shapely.points(ends), max_distance=0.3, return_distance=True)
    ends[hit[0]] = corners[hit[1]]
    ends = _snap(ends)
    segments = shapely.linestrings(np.stack([starts, ends], axis=1))
    inside = shapely.covers(region.buffer(1e-6), segments)
    long_enough = shapely.length(segments) > 0.05
    usable = np.flatnonzero(on_inset & inside & long_enough)
    if not len(usable):
        return inset, []
    # Corners first; a rung that crosses or crowds one already kept goes.
    order = usable[np.argsort(-np.array([candidates[i][0] for i in usable]), kind="stable")]
    chosen = segments[order]
    conflicts: dict[int, list] = {}
    a, b = shapely.STRtree(chosen).query(chosen, predicate="crosses")
    for x, y in zip(a, b):
        conflicts.setdefault(int(x), []).append(int(y))
    tips = shapely.points(ends[order])
    a, b = shapely.STRtree(tips).query(tips, predicate="dwithin", distance=0.4 * row_m)
    for x, y in zip(a, b):
        if x != y:
            conflicts.setdefault(int(x), []).append(int(y))
    keep = np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        if not any(keep[j] for j in conflicts.get(i, ())):
            keep[i] = True
    return inset, list(chosen[keep])


def _core_cells(area, xs, ys, taken):
    """Grid squares whose corners, edge midpoints and centre all lie in `area`."""
    import shapely

    if area.is_empty:
        return np.zeros((0, 2), dtype=np.int64)
    x0, y0, x1, y1 = area.bounds
    g = float(xs[1] - xs[0])
    i0 = max(0, int(np.floor((x0 - xs[0]) / g)))
    i1 = min(len(xs) - 1, int(np.ceil((x1 - xs[0]) / g)))
    j0 = max(0, int(np.floor((y0 - ys[0]) / g)))
    j1 = min(len(ys) - 1, int(np.ceil((y1 - ys[0]) / g)))
    if i1 <= i0 or j1 <= j0:
        return np.zeros((0, 2), dtype=np.int64)
    shapely.prepare(area)
    hx = xs[0] + 0.5 * g * np.arange(2 * i0, 2 * i1 + 1)
    hy = ys[0] + 0.5 * g * np.arange(2 * j0, 2 * j1 + 1)
    X, Y = np.meshgrid(hx, hy)
    # On the outline counts as in, so squares reach the edge of the area.
    inside = shapely.intersects_xy(area, X, Y)
    ci, cj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1))
    li, lj = 2 * (ci - i0), 2 * (cj - j0)
    ok = np.ones(ci.shape, dtype=bool)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            ok &= inside[lj + dy, li + dx]
    ok &= ~taken[cj, ci]
    cells = np.column_stack([ci[ok], cj[ok]])
    if len(cells):
        # The samples are a metre apart, and a pointed outline can slip
        # between them; the square itself has to fit.
        boxes = shapely.box(xs[cells[:, 0]], ys[cells[:, 1]], xs[cells[:, 0] + 1], ys[cells[:, 1] + 1])
        cells = cells[shapely.covers(area, boxes)]
    return cells


def _cells_outline(cells, xs, ys, shape) -> list:
    """The outline of a set of grid squares, one line per square edge."""
    import shapely

    mask = np.zeros(shape, dtype=bool)  # rows j, columns i
    mask[cells[:, 1], cells[:, 0]] = True
    padded = np.pad(mask, 1)
    segments = []
    jj, ii = np.nonzero(padded[1:-1, 1:] != padded[1:-1, :-1])
    segments.extend(((xs[i], ys[j]), (xs[i], ys[j + 1])) for j, i in zip(jj, ii))
    jj, ii = np.nonzero(padded[1:, 1:-1] != padded[:-1, 1:-1])
    segments.extend(((xs[i], ys[j]), (xs[i + 1], ys[j])) for j, i in zip(jj, ii))
    if not segments:
        return []
    return list(shapely.linestrings(np.array(segments)))


def _spokes(rungs, inset, outline, grid_m: float, reach: float) -> list:
    """Each rung carried on from the row to the grid node in front of it."""
    import shapely

    if not rungs or not outline:
        return []
    nodes = np.unique(np.round(shapely.get_coordinates(np.array(outline, dtype=object)), 6), axis=0)
    core = shapely.union_all(shapely.polygonize(np.array(outline, dtype=object)))
    coords = np.array([np.asarray(r.coords)[[0, -1], :2] for r in rungs])
    start, tip = coords[:, 0], coords[:, 1]
    direction = tip - start
    direction /= np.maximum(np.linalg.norm(direction, axis=1), 1e-12)[:, None]
    tree = shapely.STRtree(shapely.points(nodes))
    ri, ni = tree.query(shapely.points(tip), predicate="dwithin", distance=reach)
    if not len(ri):
        return []
    offset = nodes[ni] - tip[ri]
    ahead = np.einsum("ij,ij->i", offset, direction[ri])
    aside = np.abs(offset[:, 0] * direction[ri, 1] - offset[:, 1] * direction[ri, 0])
    ok = (ahead > 0.1) & (aside <= 0.75 * grid_m)
    ri, ni, score = ri[ok], ni[ok], (ahead + 2.0 * aside)[ok]
    if not len(ri):
        return []
    order = np.lexsort((score, ri))
    ri, ni, score = ri[order], ni[order], score[order]
    first = np.ones(len(ri), dtype=bool)
    first[1:] = ri[1:] != ri[:-1]
    ri, ni, score = ri[first], ni[first], score[first]
    segments = shapely.linestrings(np.stack([tip[ri], nodes[ni]], axis=1))
    inside = shapely.covers(inset.buffer(1e-6), segments)
    through = shapely.length(shapely.intersection(segments, core)) > 1e-6
    usable = np.flatnonzero(inside & ~through)
    if not len(usable):
        return []
    segments, ni, score = segments[usable], ni[usable], score[usable]
    conflicts: dict[int, set] = {}
    a, b = shapely.STRtree(segments).query(segments, predicate="crosses")
    for x, y in zip(a, b):
        conflicts.setdefault(int(x), set()).add(int(y))
    used = set()
    keep = np.zeros(len(segments), dtype=bool)
    for i in np.argsort(score, kind="stable"):
        if int(ni[i]) in used or any(keep[j] for j in conflicts.get(int(i), ())):
            continue
        keep[i] = True
        used.add(int(ni[i]))
    return list(segments[keep])


def _mesh_region(region, kind, frame, outer, xs, ys, taken, *, row_m, margin_m, across=()):
    """The faces of one region, its grid squares, its row outline and bed."""
    import shapely

    grid_m = float(xs[1] - xs[0])
    touches = region.intersects(frame.boundary)
    if kind in HIDDEN:
        inset, rungs = shapely.Polygon(), []
    else:
        inset, rungs = _rows(region, frame, outer, row_m, opening=margin_m)
        if across and rungs:
            ends = {_key(np.asarray(r.coords)[0]) for r in across}
            ends |= {_key(np.asarray(r.coords)[-1]) for r in across}
            rungs = [r for r in rungs if _key(np.asarray(r.coords)[0]) not in ends]
            if rungs:
                arr = np.array(rungs, dtype=object)
                clash = np.zeros(len(rungs), dtype=bool)
                for line in across:
                    clash |= shapely.intersects(arr, line)
                rungs = [r for r, c in zip(rungs, clash) if not c]
    if kind not in HIDDEN and inset.is_empty:
        # Too narrow for rows, so too narrow for the grid: the ladder runs
        # straight across, and a lone square in it would only break it up.
        area = shapely.Polygon()
    else:
        base = inset if not inset.is_empty else region
        if touches:
            base = shapely.union(shapely.make_valid(base), outer)
        area = shapely.intersection(
            _polygonal(shapely.make_valid(base).buffer(-margin_m)), frame
        )
    cells = _core_cells(area, xs, ys, taken)
    if len(cells):
        taken[cells[:, 1], cells[:, 0]] = True

    outline = _cells_outline(cells, xs, ys, taken.shape) if len(cells) else []
    if across and len(cells):
        # A rung across a narrow part stays out of the grid of a wide one.
        grid = shapely.union_all(shapely.box(
            xs[cells[:, 0]], ys[cells[:, 1]], xs[cells[:, 0] + 1], ys[cells[:, 1] + 1]
        ))
        arr = np.array(across, dtype=object)
        clear = shapely.length(shapely.intersection(arr, grid)) < 1e-6
        across = [line for line, ok in zip(across, clear) if ok]
    lines = [region.boundary]
    if not inset.is_empty:
        lines.append(inset.boundary)
    lines.extend(rungs)
    lines.extend(across)
    lines.extend(outline)
    if not inset.is_empty and outline and rungs:
        lines.extend(_spokes(rungs, inset, outline, grid_m, margin_m + 2.0 * grid_m))
    noded = shapely.union_all(np.array(lines, dtype=object), grid_size=GRID_SIZE)
    faces = np.array(_parts(shapely.polygonize(shapely.get_parts(noded))), dtype=object)
    if not len(faces):
        return [], cells, inset, []
    # Nothing is dropped for being small: a face left out is a hole.
    faces = faces[shapely.area(faces) > 1e-9]
    shapely.prepare(region)
    faces = faces[shapely.covers(region, shapely.point_on_surface(faces))]
    if not len(faces):
        return [], cells, inset, []
    reps = shapely.point_on_surface(faces)
    in_core = np.zeros(len(faces), dtype=bool)
    if len(cells):
        centres = shapely.points(np.column_stack(
            [xs[cells[:, 0]] + 0.5 * grid_m, ys[cells[:, 1]] + 0.5 * grid_m]
        ))
        hit = shapely.STRtree(faces).query(centres, predicate="within")
        in_core[np.unique(hit[1])] = True
    if kind in HIDDEN:
        part_of = np.full(len(faces), FILL)
    elif inset.is_empty:
        part_of = np.full(len(faces), ROW)
    else:
        shapely.prepare(inset)
        part_of = np.where(shapely.contains(inset, reps), FILL, ROW)

    # Under water, a rung straight across a ditch is split at its middle, and
    # the middles joined: the bed of the ditch.
    middles = {}
    if kind == WATER and across:
        for line in across:
            c = _snap(np.asarray(line.coords)[[0, -1], :2])
            middles[tuple(sorted((_key(c[0]), _key(c[1]))))] = 0.5 * (c[0] + c[1])

    out, beds = [], []
    for face, core, part in zip(faces, in_core, part_of):
        if core:
            continue
        coords = np.asarray(face.exterior.coords)[:-1, :2]
        if middles:
            if len(face.interiors) == 0:
                split = _split_at_middles(coords, middles)
                if split is not None:
                    pieces, bed = split
                    out.extend((p, part) for p in pieces)
                    beds.extend(bed)
                    continue
            else:
                face = _with_middles(face, middles)
        if len(face.interiors) == 0 and len(coords) == 3:
            out.append((_ccw(coords), part))
            continue
        if len(face.interiors) == 0 and len(coords) == 4 and _good_quad(_ccw(coords)):
            out.append((_ccw(coords), part))
            continue
        out.extend((p, part) for p in _fill(face))
    if kind == WATER:
        out = _deepen(out, region)
    return out, cells, inset, beds


def _split_at_middles(coords, middles):
    """A ladder cell across a ditch, split along the bed; None if it is not one."""
    keys = [_key(c) for c in coords]
    k = len(coords)
    rung = [tuple(sorted((keys[i], keys[(i + 1) % k]))) in middles for i in range(k)]
    if not any(rung):
        return None
    if k == 4 and rung[1] and rung[3] and not rung[0] and not rung[2]:
        coords = np.roll(coords, -1, axis=0)
        keys = keys[1:] + keys[:1]
        rung = rung[1:] + rung[:1]
    if k == 4 and rung[0] and rung[2] and not rung[1] and not rung[3]:
        a, b, c, d = coords
        m_ab = middles[tuple(sorted((keys[0], keys[1])))]
        m_cd = middles[tuple(sorted((keys[2], keys[3])))]
        first = _ccw(np.array([a, m_ab, m_cd, d]))
        second = _ccw(np.array([m_ab, b, c, m_cd]))
        return [first, second], [np.array([m_ab, m_cd])]
    # Any other face with a split rung must carry the middle too, or the
    # cell beside it would have a vertex this one lacks.
    import shapely

    ring = []
    for i in range(k):
        ring.append(coords[i])
        if rung[i]:
            ring.append(middles[tuple(sorted((keys[i], keys[(i + 1) % k])))])
    polygon = shapely.Polygon(np.array(ring))
    if not polygon.is_valid or polygon.area < 1e-4:
        return None
    return list(_fill(polygon)), []


def _with_middles(face, middles):
    """A face with the middle of every split rung added to its rings."""
    import shapely

    rings = []
    for ring in [face.exterior, *face.interiors]:
        coords = np.asarray(ring.coords)[:-1, :2]
        keys = [_key(c) for c in coords]
        out = []
        for i in range(len(coords)):
            out.append(coords[i])
            pair = tuple(sorted((keys[i], keys[(i + 1) % len(coords)])))
            if pair in middles:
                out.append(middles[pair])
        rings.append(np.array(out))
    polygon = shapely.Polygon(rings[0], rings[1:])
    return polygon if polygon.is_valid else face


def _deepen(faces, region):
    """A water face with every corner on the waterline gets a middle on the bed.

    Without one it lies flat at the water level, exactly where the water
    surface is drawn, and the two flicker through each other.
    """
    import shapely

    boundary = region.boundary
    shapely.prepare(boundary)
    out = []
    for coords, part in faces:
        on_line = shapely.dwithin(shapely.points(coords), boundary, 1e-3)
        area = 0.5 * abs(float(np.sum(coords[:, 0] * np.roll(coords[:, 1], -1)
                                      - np.roll(coords[:, 0], -1) * coords[:, 1])))
        if on_line.all() and area > 0.5:
            centre = coords.mean(axis=0)
            for i in range(len(coords)):
                out.append((_ccw(np.array([coords[i], coords[(i + 1) % len(coords)], centre])), part))
        else:
            out.append((coords, part))
    return out


def _fill(face) -> list:
    """A face triangulated, and the triangles paired back into quads."""
    import shapely

    triangles = shapely.get_parts(shapely.constrained_delaunay_triangles(face))
    if not len(triangles):
        return []
    coords = shapely.get_coordinates(shapely.get_exterior_ring(triangles)).reshape(-1, 4, 2)[:, :3]
    coords = _without_flat_triangles(coords)
    if not len(coords):
        return []
    quads, rest = _recombine(_ccw_many(coords))
    return quads + rest


def _without_flat_triangles(triangles: np.ndarray) -> np.ndarray:
    """Replace each zero-area triangle by splitting its neighbour.

    The triangulator makes a flat triangle out of a vertex lying on a straight
    stretch of outline -- a station, the middle of a rung. Dropping it drops
    that vertex from the face while the face beside it keeps it: a crack. The
    triangle across the flat one's long edge is split at the vertex instead.
    """
    def area2(t):
        return (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (
            t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])

    if not len(triangles) or (np.abs(area2(triangles)) >= 1e-8).all():
        return triangles
    tris = [t for t in triangles]
    for _ in range(len(tris)):
        a2 = area2(np.array(tris))
        flat = np.flatnonzero(np.abs(a2) < 1e-8)
        if not len(flat):
            break
        f = int(flat[0])
        t = tris[f]
        # The middle vertex is the one between the other two.
        lengths = [np.hypot(*(t[(i + 2) % 3] - t[(i + 1) % 3])) for i in range(3)]
        m = int(np.argmax(lengths))  # opposite the long edge
        a, b, middle = t[(m + 1) % 3], t[(m + 2) % 3], t[m]
        key_a, key_b = _key(a), _key(b)
        partner = None
        for j, other in enumerate(tris):
            if j == f:
                continue
            keys = [_key(p) for p in other]
            if key_a in keys and key_b in keys:
                partner = j
                break
        del tris[f]
        if partner is None:
            continue
        if partner > f:
            partner -= 1
        other = tris.pop(partner)
        keys = [_key(p) for p in other]
        apex = other[[i for i, k in enumerate(keys) if k not in (key_a, key_b)][0]]
        tris.append(np.array([a, middle, apex]))
        tris.append(np.array([middle, b, apex]))
    if not tris:
        return np.zeros((0, 3, 2))
    out = np.array(tris)
    return out[np.abs(area2(out)) >= 1e-8]


def _recombine(triangles: np.ndarray):
    """Pair triangles into quads where the quad is well shaped, best first.

    `triangles` is ``(k, 3, 2)``, counter-clockwise. Returns the quads and
    the triangles left over, as lists of corner arrays.
    """
    count = len(triangles)
    if count < 2:
        return [], [t for t in triangles]
    keys = np.rint(triangles * 1000.0).astype(np.int64)
    # Each edge as (first, second) keyed by its sorted endpoints.
    a = keys
    b = np.roll(keys, -1, axis=1)
    swap = (a[:, :, 0] > b[:, :, 0]) | ((a[:, :, 0] == b[:, :, 0]) & (a[:, :, 1] > b[:, :, 1]))
    lo = np.where(swap[:, :, None], b, a).reshape(-1, 2)
    hi = np.where(swap[:, :, None], a, b).reshape(-1, 2)
    edge_keys = np.column_stack([lo, hi])
    _, inverse, counts = np.unique(edge_keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    shared = np.flatnonzero(counts[inverse] == 2)
    if not len(shared):
        return [], [t for t in triangles]
    order = shared[np.argsort(inverse[shared], kind="stable")]
    first, second = order[0::2], order[1::2]
    t1, e1 = first // 3, first % 3
    t2, e2 = second // 3, second % 3
    pa = triangles[t1, e1]
    pb = triangles[t1, (e1 + 1) % 3]
    pc = triangles[t1, (e1 + 2) % 3]
    pd = triangles[t2, (e2 + 2) % 3]
    quads = np.stack([pc, pa, pd, pb], axis=1)
    score = _quad_scores(quads)
    good = np.flatnonzero(score <= 55.0)
    good = good[np.argsort(score[good], kind="stable")]
    used = np.zeros(count, dtype=bool)
    out = []
    for i in good:
        x, y = t1[i], t2[i]
        if used[x] or used[y]:
            continue
        used[x] = used[y] = True
        out.append(quads[i])
    return out, [triangles[t] for t in np.flatnonzero(~used)]


def _quad_scores(quads: np.ndarray) -> np.ndarray:
    """Per quad ``(k, 4, 2)``: how far its worst corner is from square, in
    degrees; infinite for one that is not convex counter-clockwise."""
    previous = quads - np.roll(quads, 1, axis=1)
    following = np.roll(quads, -1, axis=1) - quads
    cross = previous[:, :, 0] * following[:, :, 1] - previous[:, :, 1] * following[:, :, 0]
    cosine = -(previous * following).sum(axis=2) / np.maximum(
        np.linalg.norm(previous, axis=2) * np.linalg.norm(following, axis=2), 1e-12
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    score = np.abs(angles - 90.0).max(axis=1)
    return np.where((cross > 1e-9).all(axis=1), score, np.inf)


def _quad_score(quad: np.ndarray):
    """How far the worst corner is from square; None for a non-convex quad."""
    score = float(_quad_scores(quad[None])[0])
    return None if not np.isfinite(score) else score


def _good_quad(quad: np.ndarray) -> bool:
    score = _quad_score(quad)
    return score is not None and score < 75.0


def _ccw_many(triangles: np.ndarray) -> np.ndarray:
    """Triangles ``(k, 3, 2)`` turned counter-clockwise where they are not."""
    t = triangles
    area = (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (
        t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])
    out = t.copy()
    flip = area < 0
    out[flip] = out[flip][:, ::-1]
    return out


def _ccw(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    x, y = coords[:, 0], coords[:, 1]
    area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return coords if area > 0 else coords[::-1].copy()


# ---------------------------------------------------------------------------
# Heights, orientation and outlines
# ---------------------------------------------------------------------------


def _water_heights(heights, loops, sizes, face_kind, xy, bodies, shift, depth_m):
    """The waterline where surveyed but never below the water; under it, the bed.

    The outline the BGT draws is the top of the bank as often as the water's
    edge, and a quay is surveyed at the top of the wall. Pinning the waterline
    to the water level pulled every bank top down to it: over Bronbeek half
    the ground touching a waterline came out more than 25 cm off the survey.
    So the waterline keeps its surveyed height, raised to the level only where
    the survey dips below it -- the water never spills over the edge -- and
    the row inside the water is the bank, from there down to the bed.
    """
    import shapely

    water_faces = np.flatnonzero(face_kind == WATER)
    if not len(water_faces) or not bodies:
        return heights
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    polygons = np.array([g for g, _ in bodies], dtype=object)
    levels = np.array([level for _, level in bodies])
    tree = shapely.STRtree(polygons)
    # The level of each water face: the body its middle lies in, or failing
    # that the nearest one.
    centres = np.array([xy[loops[starts[f]: starts[f] + sizes[f]]].mean(axis=0) for f in water_faces])
    local = shapely.points(centres - shift)
    nearest = tree.query_nearest(local, all_matches=False)
    face_level = np.full(len(sizes), np.nan)
    face_level[water_faces[nearest[0]]] = levels[nearest[1]]

    vertex_level = np.full(len(heights), np.inf)
    on_land = np.zeros(len(heights), dtype=bool)
    face_of_loop = np.repeat(np.arange(len(sizes)), sizes)
    is_water = face_kind[face_of_loop] == WATER
    np.minimum.at(vertex_level, loops[is_water], face_level[face_of_loop[is_water]])
    on_land[loops[~is_water]] = True
    wet = np.isfinite(vertex_level)
    out = heights.copy()
    shore = wet & on_land
    under = wet & ~on_land
    out[shore] = np.maximum(heights[shore], vertex_level[shore])
    out[under] = np.minimum(heights[under], vertex_level[under] - depth_m)
    return out


def _orient(vertices, sizes, loops, face_part, face_kind, terrain):
    """Every quad turned so its first-to-third diagonal fits the ground best.

    The grid squares keep the south-west to north-east split they always had.
    Elsewhere the diagonal whose middle lies nearer the surveyed ground is the
    one drawn; under water, where nothing was surveyed, the shorter one.
    """
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    quads = np.flatnonzero((sizes == 4) & (face_part != CORE))
    if not len(quads):
        return sizes, loops
    idx = starts[quads][:, None] + np.arange(4)[None, :]
    ring = loops[idx]
    p = vertices[ring]
    centre = p[:, :, :2].mean(axis=1)
    ground = np.asarray(terrain.sample(centre[:, 0], centre[:, 1]), dtype=np.float64)
    mid_a = 0.5 * (p[:, 0, 2] + p[:, 2, 2])
    mid_b = 0.5 * (p[:, 1, 2] + p[:, 3, 2])
    use_b = np.abs(mid_b - ground) < np.abs(mid_a - ground) - 1e-6
    wet = face_kind[quads] == WATER
    if wet.any():
        len_a = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
        len_b = np.linalg.norm(p[:, 1] - p[:, 3], axis=1)
        use_b = np.where(wet, len_b < len_a - 1e-6, use_b)
    rolled = np.roll(ring, -1, axis=1)
    loops = loops.copy()
    loops[idx[use_b]] = rolled[use_b]
    return sizes, loops


def _outline_edges(loops, sizes, face_kind, face_part, vertex_count, beds, keys, shift):
    """The edges along outlines, rows and beds, and the vertices on them."""
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    face_of_loop = np.repeat(np.arange(len(sizes)), sizes)
    following = np.arange(len(loops)) + 1
    ends = starts + sizes
    following[ends - 1] = starts
    a = loops
    b = loops[following]
    pair = np.sort(np.column_stack([a, b]), axis=1)
    unique, inverse = np.unique(pair, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(unique))
    first_face = np.full(len(unique), -1, dtype=np.int64)
    second_face = np.full(len(unique), -1, dtype=np.int64)
    position = np.concatenate([[0], np.cumsum(counts)[:-1]])
    first_face[:] = face_of_loop[order[position]]
    two = counts >= 2
    second_face[two] = face_of_loop[order[position[two] + 1]]

    bits = np.zeros(len(unique), dtype=np.int32)
    k1 = face_kind[first_face]
    k2 = np.where(two, face_kind[np.maximum(second_face, 0)], -1)
    p1 = face_part[first_face]
    p2 = np.where(two, face_part[np.maximum(second_face, 0)], -1)
    differ = two & (k1 != k2)
    for kind, bit in ((WATER, WATERLINE), (ROAD, ROAD_EDGE), (BUILDING, BUILDING_EDGE)):
        bits |= np.where(differ & ((k1 == kind) | (k2 == kind)), bit, 0).astype(np.int32)
    # The inner edge of a row: a row face beside the fill or the grid of the
    # same region kind.
    row_inner = two & (k1 == k2) & (
        ((p1 == ROW) & (p2 != ROW)) | ((p2 == ROW) & (p1 != ROW))
    )
    bits |= np.where(row_inner, ROW_EDGE, 0).astype(np.int32)
    bits |= np.where(row_inner & (k1 == WATER), BED, 0).astype(np.int32)
    if beds:
        index = {tuple(k): i for i, k in enumerate(keys)}
        lookup = {tuple(e): i for i, e in enumerate(unique)}
        for segment in beds:
            ka = tuple(np.rint(np.asarray(segment[0]) * 1000.0).astype(np.int64))
            kb = tuple(np.rint(np.asarray(segment[1]) * 1000.0).astype(np.int64))
            if ka in index and kb in index:
                e = tuple(sorted((index[ka], index[kb])))
                if e in lookup:
                    bits[lookup[e]] |= BED
    tagged = bits > 0
    edges = unique[tagged]
    edge_on = bits[tagged]
    vertex_on = np.zeros(vertex_count, dtype=np.int32)
    np.bitwise_or.at(vertex_on, edges[:, 0], edge_on)
    np.bitwise_or.at(vertex_on, edges[:, 1], edge_on)
    return edges.astype(np.int64), edge_on.astype(np.int32), vertex_on


__all__ = [
    "BED",
    "BUILDING",
    "BUILDING_EDGE",
    "CORE",
    "FEATURE_NAMES",
    "FILL",
    "KIND_NAMES",
    "LAND",
    "PART_NAMES",
    "ROAD",
    "ROAD_EDGE",
    "ROW",
    "ROW_EDGE",
    "SOURCES",
    "TerrainMesh",
    "TriangleLocator",
    "WATER",
    "WATERLINE",
    "build_terrain_mesh",
    "grid_mesh",
]
