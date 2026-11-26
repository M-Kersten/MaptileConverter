"""TerreinLab tools: the edits the model needs after a person has touched it.

Ships next to every model.blend, and inside it as a text block. Install it once
as an add-on (Edit > Preferences > Add-ons > Install..., pick this file) or run
the text block in the Scripting workspace. It adds a "TerreinLab" tab to the 3D
view's sidebar (N) with four buttons:

Fit terrain under roads
    After moving, widening or adding road, push the terrain down wherever it
    would show through. The same rule the pipeline used: each point of road
    surface lowers the three corners of the terrain triangle under it just
    enough, the corners under the road most. Only ever lowers, so a sculpt
    survives it; to get ground back from under a road you moved away, Reset
    first and Fit after. Works on the terrain whatever its faces: the grid,
    the rows along the outlines, or anything you have remeshed. In the
    layered style the roads, walkways and grass are the ground itself, so
    there is nothing to fit (in a model made before that, they follow the
    terrain by themselves; only one whose Shrinkwrap or drape was removed or
    applied is fitted under).

Reset terrain to survey
    Put every terrain vertex back at the surveyed height it was built from,
    undoing every fit and every sculpt -- in the layered style the roads,
    walkways and grass too, which are pieces of the same ground. Run Fit
    afterwards.

Drop selected onto ground
    Move each selected object straight down (or up) so its origin sits on the
    road, walkway, grass or terrain under it. Buildings, trees and road
    pieces all have their origin on the ground, so this is how something
    moved sideways gets back onto the surface.

Export FBX for Unity
    The pipeline's own export settings, so a re-export from an edited model
    lands in Unity exactly like the original did.

Problem areas (its own section of the tab)
    Find problems colours the faces a modeller would want to look at, over
    the model in the viewport (nothing is written into the mesh, so nothing
    reaches the FBX):

    * Slivers: long thin triangles, and quads with a spike of a corner.
    * Too dense: more vertices in one square of the ground than the budget.
    * Bent n-gons: faces with more than four sides whose corners stand off
      one plane, which every program cuts into triangles its own way. The
      ground's open stretches are n-gons on purpose, and lie flat.
    * Flipped: ground facing down, or a face wound against its neighbour.
    * Seams: an edge of one ground piece with no partner in the next -- a
      crack, a T-junction, or a step in height between two pieces.
    * Moved: vertices moved further from their surveyed height than the
      tolerance, by a sculpt or an edit.

    Previous / Next frame the view on each problem spot in turn, worst
    first; Select puts the problem faces of the active object in an edit-
    mode selection. Scans the selected objects, or with nothing selected the
    ground and the barriers.

Only bpy and numpy are used, both of which Blender bundles.
"""

bl_info = {
    "name": "TerreinLab tools",
    "author": "TerreinLab",
    "version": (1, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > TerreinLab",
    "description": "Keep roads and terrain fitted together while editing",
    "category": "Object",
}

import contextlib

import bpy
import numpy as np

# Must match src/roads.py and src/elevation.py.
TERRAIN_CLEARANCE_M = 0.05
HIDDEN_VERTEX_WEIGHT = 4.0
ROAD_CELL_M = 0.25
ROAD_COLLECTION = "Roads"
TERRAIN_NAME = "Terrain"
# What blender/process.py stores as `maptile_role` on the layered style's
# objects: the terrain, and what lies on it.
GROUND_ROLES = ("road", "walkway", "grass")


# A face of the ground under a road, in the terrain's `ground_kind` attribute.
GROUND_UNDER_ROAD = 1


def _terrain_object():
    """The terrain: the object the pipeline marked as one, else by its name."""
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.get("maptile_role") == "terrain":
            return obj
    return bpy.data.objects.get(TERRAIN_NAME)


def _follows(obj, terrain) -> bool:
    """Whether a layer needs no fitting: it is part of the ground, or kept on it.

    In the layered style the roads, walkways and grass are the ground itself,
    each the faces of it under them (`maptile_ground`), so there is nothing
    to push the terrain under. A layer from an older model is wrapped onto
    the terrain with a Shrinkwrap, or draped with Maptile_Drape geometry
    nodes, and moves with it.
    """
    if obj.get("maptile_ground"):
        return True
    for m in obj.modifiers:
        if not m.show_viewport:
            continue
        if m.type == "SHRINKWRAP" and m.target == terrain:
            return True
        group = getattr(m, "node_group", None) if m.type == "NODES" else None
        if group is not None and group.name.startswith("Maptile_Drape"):
            return True
    return False


def _road_objects(terrain):
    """What the terrain has to stay under: the roads, less any wrapped onto it."""
    found = []
    collection = bpy.data.collections.get(ROAD_COLLECTION)
    if collection is not None:
        found.extend(o for o in collection.all_objects if o.type == "MESH")
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.get("maptile_role") in ("road", "walkway"):
            if obj not in found:
                found.append(obj)
    return [o for o in found if terrain is None or not _follows(o, terrain)]


def face_triangles(co, loops, starts, totals):
    """Every face as triangles of vertex ids, ``(T, 3)``, and the face of each.

    Cut as the export cuts them (`_triangulate_on_export`): a quad from its
    first corner to its third, an n-gon by Blender's beauty fill. The fill
    keeps a concave n-gon inside its outline, where a fan from its first
    corner would reach over the ground beside it, and turns away the
    triangle of no area that three stations along one straight edge would
    make, so the middle one is a corner of the ground too. Any triangle with
    no area left in the plan is left out.
    """
    import bmesh

    co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
    loops = np.asarray(loops, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    totals = np.asarray(totals, dtype=np.int64)
    bm = bmesh.new()
    try:
        verts = [bm.verts.new(p) for p in co.tolist()]
        bm.verts.index_update()
        owner_layer = bm.faces.layers.int.new("owner")
        for f, (s, n) in enumerate(zip(starts.tolist(), totals.tolist())):
            try:
                face = bm.faces.new([verts[i] for i in loops[s: s + n].tolist()])
            except ValueError:  # the same face twice
                continue
            face[owner_layer] = f
        # The fill works in the plane of each face, which a face made here
        # does not know until it is told.
        bm.normal_update()
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="FIXED", ngon_method="BEAUTY")
        triangles = np.array([[v.index for v in face.verts] for face in bm.faces],
                             dtype=np.int64).reshape(-1, 3)
        owner = np.array([face[owner_layer] for face in bm.faces], dtype=np.int64)
    finally:
        bm.free()
    a, b, c = (co[triangles[:, k], :2] for k in range(3))
    area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    keep = area > 1e-10
    return triangles[keep], owner[keep]


def _terrain_mesh(obj):
    """The terrain as world-space vertices, triangles, and which vertices show.

    Triangles as the export draws them (`face_triangles`): a quad split from
    its first corner to its third, an n-gon by Blender's polygon fill. A
    vertex shows if
    any face round it is not ground under a road (or, in the layered style,
    under a layer); without the attribute -- a terrain someone has rebuilt --
    every vertex counts as showing.
    """
    mesh = obj.data
    count = len(mesh.vertices)
    if count < 3 or not len(mesh.polygons):
        return None
    co = np.empty(count * 3)
    mesh.vertices.foreach_get("co", co)
    matrix = np.asarray(obj.matrix_world)
    co = co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
    starts = np.empty(len(mesh.polygons), dtype=np.int64)
    totals = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", totals)
    loops = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loops)
    triangles, _ = face_triangles(co, loops, starts, totals)
    visible = np.ones(count, dtype=bool)
    covered = None
    for name, test in (
        ("ground_kind", lambda v: v == GROUND_UNDER_ROAD),
        ("ground_layer", lambda v: v != 0),
    ):
        attribute = mesh.attributes.get(name)
        if attribute is not None and attribute.domain == "FACE":
            values = np.empty(len(mesh.polygons), dtype=np.int32)
            attribute.data.foreach_get("value", values)
            covered = test(values)
            break
    if covered is not None:
        face_of_loop = np.repeat(np.arange(len(totals)), totals)
        visible[:] = False
        visible[loops[~covered[face_of_loop]]] = True
    return co, triangles, visible


class TriangleLocator:
    """Which triangle each point falls in; as src/terrain_mesh.py, verbatim."""

    def __init__(self, xy, tris, *, cell, bounds):
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
        self.bucket_start = np.searchsorted(bucket[order], np.arange(self.nx * self.ny + 1))

    def locate(self, x, y, chunk=200_000):
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
        best = np.full(n, -np.inf)
        np.maximum.at(best, point, score)
        winner = (score == best[point]) & (score > -1e-6)
        first = np.unique(point[winner], return_index=True)[1]
        pick = np.flatnonzero(winner)[first]
        tri[point[pick]] = cand[pick]
        weights[point[pick]] = np.column_stack([w0[pick], u[pick], v[pick]])
        return tri, weights


def _road_triangles(context, objects):
    """World-space triangles of every road surface, kerbs and skirts left out."""
    depsgraph = context.evaluated_depsgraph_get()
    out = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
            mesh.loop_triangles.foreach_get("vertices", tris)
            poly = np.empty(len(mesh.loop_triangles), dtype=np.int64)
            mesh.loop_triangles.foreach_get("polygon_index", poly)
            co = np.empty(len(mesh.vertices) * 3)
            mesh.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            part = mesh.attributes.get("road_part")
            keep = np.ones(len(poly), dtype=bool)
            if part is not None and part.domain == "FACE":
                values = np.empty(len(mesh.polygons), dtype=np.int32)
                part.data.foreach_get("value", values)
                keep = values[poly] == 0
            matrix = np.asarray(obj.matrix_world)
            world = co @ matrix[:3, :3].T + matrix[:3, 3]
            out.append(world[tris.reshape(-1, 3)[keep]])
        finally:
            evaluated.to_mesh_clear()
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def _underside_points(tris, cell):
    """(x, y, z) points the terrain has to stay under, as in src/roads.py."""
    if not len(tris):
        return np.zeros((0, 3))
    lo = tris[:, :, :2].min(axis=(0, 1))
    hi = tris[:, :, :2].max(axis=(0, 1))
    cols = max(1, int(np.ceil((hi[0] - lo[0]) / cell)))
    rows = max(1, int(np.ceil((hi[1] - lo[1]) / cell)))
    low = np.full((rows, cols), np.inf)
    px = (tris[:, :, 0] - lo[0]) / cell
    py = (tris[:, :, 1] - lo[1]) / cell
    pz = tris[:, :, 2]
    v0x, v0y = px[:, 1] - px[:, 0], py[:, 1] - py[:, 0]
    v1x, v1y = px[:, 2] - px[:, 0], py[:, 2] - py[:, 0]
    den = v0x * v1y - v0y * v1x
    for i in range(len(tris)):
        if abs(den[i]) < 1e-12:
            continue
        c0 = max(0, int(np.floor(px[i].min())))
        c1 = min(cols - 1, int(np.ceil(px[i].max())))
        r0 = max(0, int(np.floor(py[i].min())))
        r1 = min(rows - 1, int(np.ceil(py[i].max())))
        if c1 < c0 or r1 < r0:
            continue
        cc = np.arange(c0, c1 + 1) + 0.5
        rr = np.arange(r0, r1 + 1) + 0.5
        dx = cc[None, :] - px[i, 0]
        dy = rr[:, None] - py[i, 0]
        u = (dx * v1y[i] - dy * v1x[i]) / den[i]
        v = (dy * v0x[i] - dx * v0y[i]) / den[i]
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        if not inside.any():
            continue
        height = pz[i, 0] + u * (pz[i, 1] - pz[i, 0]) + v * (pz[i, 2] - pz[i, 0])
        block = low[r0 : r1 + 1, c0 : c1 + 1]
        np.minimum(block, np.where(inside, height, np.inf), out=block)
    r, c = np.nonzero(np.isfinite(low))
    cells = np.column_stack([lo[0] + (c + 0.5) * cell, lo[1] + (r + 0.5) * cell, low[r, c]])
    corners = tris.reshape(-1, 3)
    middles = 0.5 * (tris + np.roll(tris, -1, axis=1)).reshape(-1, 3)
    return np.vstack([cells, corners, middles])


def fit_terrain(xyz, triangles, visible, points, clearance=TERRAIN_CLEARANCE_M, cell=2.0):
    """Lower the terrain so every point clears it; as src/elevation.py.

    Each point lies in one terrain triangle, and the smallest lowering that
    clears it moves the triangle's corners in proportion to their weight, the
    ones nobody sees (under a road all round) HIDDEN_VERTEX_WEIGHT times more.
    Every corner takes the largest move any point asks of it. Only lowers.
    """
    z = np.asarray(xyz, dtype=np.float64)[:, 2].copy()
    if not len(points) or not len(triangles):
        return z, 0
    xy = np.asarray(xyz, dtype=np.float64)[:, :2]
    locator = TriangleLocator(
        xy, triangles, cell=cell, bounds=(*xy.min(axis=0), *xy.max(axis=0))
    )
    tri, weights = locator.locate(points[:, 0], points[:, 1])
    found = tri >= 0
    corners = np.asarray(triangles, dtype=np.int64)[tri[found]]
    w = weights[found]
    excess = (w * z[corners]).sum(axis=1) - (points[found, 2] - clearance)
    bad = excess > 0.0
    drop = np.zeros(len(z))
    if bad.any():
        wb = w[bad]
        share = wb * np.where(visible[corners[bad]], 1.0, HIDDEN_VERTEX_WEIGHT)
        need = excess[bad, None] * share / np.maximum((wb * share).sum(axis=1), 1e-12)[:, None]
        np.maximum.at(drop, corners[bad].ravel(), need.ravel())
    return z - drop, int((drop > 1e-6).sum())


def _write_heights(obj, heights_world):
    mesh = obj.data
    count = len(mesh.vertices)
    co = np.empty(count * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    # Back into the object's own frame; only z moves.
    inverse = np.linalg.inv(np.asarray(obj.matrix_world))
    world = co @ np.asarray(obj.matrix_world)[:3, :3].T + np.asarray(obj.matrix_world)[:3, 3]
    world[:, 2] = heights_world.ravel()
    local = world @ inverse[:3, :3].T + inverse[:3, 3]
    mesh.vertices.foreach_set("co", local.ravel())
    mesh.update()


class MAPTILE_OT_fit_terrain(bpy.types.Operator):
    bl_idname = "maptile.fit_terrain"
    bl_label = "Fit terrain under roads"
    bl_description = "Lower the terrain wherever a road surface would not clear it"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        terrain = _terrain_object()
        if terrain is None:
            self.report({"ERROR"}, f"no terrain: no object called {TERRAIN_NAME}")
            return {"CANCELLED"}
        found = _terrain_mesh(terrain)
        if found is None:
            self.report({"ERROR"}, "the terrain has no faces")
            return {"CANCELLED"}
        roads = _road_objects(terrain)
        if not roads:
            self.report({"INFO"}, "nothing to fit: every road follows the terrain")
            return {"FINISHED"}
        xyz, triangles, visible = found
        points = _underside_points(_road_triangles(context, roads), ROAD_CELL_M)
        heights, moved = fit_terrain(xyz, triangles, visible, points)
        _write_heights(terrain, heights)
        self.report({"INFO"}, f"lowered {moved} terrain vertices")
        return {"FINISHED"}


class MAPTILE_OT_reset_terrain(bpy.types.Operator):
    bl_idname = "maptile.reset_terrain"
    bl_label = "Reset terrain to survey"
    bl_description = "Put every terrain vertex back at its surveyed height"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        terrain = _terrain_object()
        # The terrain, and in the layered style every piece of the ground.
        objects = [terrain] if terrain is not None else []
        objects += [
            o for o in bpy.data.objects
            if o.type == "MESH" and o.get("maptile_ground") and o is not terrain
        ]
        reset = 0
        for obj in objects:
            measured = obj.data.attributes.get("measured_z")
            if not measured:
                continue
            values = np.empty(len(obj.data.vertices))
            measured.data.foreach_get("value", values)
            co = np.empty(len(values) * 3)
            obj.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            co[:, 2] = values
            obj.data.vertices.foreach_set("co", co.ravel())
            obj.data.update()
            reset += 1
        if not reset:
            self.report({"ERROR"}, "the terrain carries no surveyed heights")
            return {"CANCELLED"}
        self.report({"INFO"}, f"{reset} ground objects back at the surveyed heights")
        return {"FINISHED"}


class MAPTILE_OT_drop_to_ground(bpy.types.Operator):
    bl_idname = "maptile.drop_to_ground"
    bl_label = "Drop selected onto ground"
    bl_description = "Move each selected object vertically onto the road or terrain"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        depsgraph = context.evaluated_depsgraph_get()
        scene = context.scene
        targets = set()
        terrain = _terrain_object()
        if terrain is not None:
            targets.add(terrain.name)
        roads = bpy.data.collections.get(ROAD_COLLECTION)
        if roads is not None:
            targets.update(o.name for o in roads.all_objects)
        targets.update(
            o.name for o in bpy.data.objects if o.get("maptile_role") in GROUND_ROLES
        )
        moved = 0
        for obj in context.selected_objects:
            origin = obj.matrix_world.translation.copy()
            start = origin.copy()
            start.z = 10000.0
            hit, location, _, _, hit_obj, _ = (False, None, None, None, None, None)
            # Step down through whatever is above until a road or the terrain.
            probe = start
            for _ in range(16):
                hit, location, _, _, hit_obj, _ = scene.ray_cast(
                    depsgraph, probe, (0.0, 0.0, -1.0)
                )
                if not hit:
                    break
                if hit_obj.name in targets and hit_obj != obj:
                    break
                probe = location.copy()
                probe.z -= 1e-3
                hit = False
            if hit:
                world = obj.matrix_world.copy()
                world.translation.z = location.z
                obj.matrix_world = world
                moved += 1
        self.report({"INFO"}, f"dropped {moved} objects onto the ground")
        return {"FINISHED"}


@contextlib.contextmanager
def render_modifiers():
    """Every modifier as the render has it, for as long as an export runs.

    As blender/process.py: the FBX exporter applies what the viewport shows,
    so the triangulation that is on for render only is switched on for it.
    """
    switched = []
    for obj in bpy.data.objects:
        for modifier in getattr(obj, "modifiers", ()):
            if modifier.show_viewport != modifier.show_render:
                switched.append((modifier, modifier.show_viewport))
                modifier.show_viewport = modifier.show_render
    try:
        yield len(switched)
    finally:
        for modifier, shown in switched:
            modifier.show_viewport = shown


def export_unity_fbx(path):
    """The pipeline's export settings: see blender/process.py export_fbx."""
    with render_modifiers():
        bpy.ops.export_scene.fbx(
            filepath=str(path),
            use_selection=False,
            apply_unit_scale=True,
            apply_scale_options="FBX_SCALE_ALL",
            global_scale=1.0,
            axis_forward="-Z",
            axis_up="Y",
            bake_space_transform=False,
            object_types={"MESH", "EMPTY"},
            use_mesh_modifiers=True,
            use_mesh_modifiers_render=True,
            use_custom_props=True,
            mesh_smooth_type="FACE",
            use_tspace=True,
            path_mode="STRIP",
            embed_textures=False,
            bake_anim=False,
        )


class MAPTILE_OT_export_unity(bpy.types.Operator):
    bl_idname = "maptile.export_unity"
    bl_label = "Export FBX for Unity"
    bl_description = "Export with the settings the pipeline used"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = bpy.path.abspath("//model.fbx") if bpy.data.filepath else "model.fbx"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        export_unity_fbx(self.filepath)
        self.report({"INFO"}, f"exported {self.filepath}")
        return {"FINISHED"}


# --- Problem areas -----------------------------------------------------------
#
# The scan is plain numpy over the mesh arrays (`face_problems`,
# `seam_vertices`, `problem_spots`), so it can be tested without Blender; the
# operators below only gather the arrays and show the result.

# Name, label and overlay colour, in the order the panel lists them.
PROBLEMS = (
    ("sliver", "Slivers", (1.0, 0.18, 0.12)),
    ("dense", "Too dense", (1.0, 0.58, 0.0)),
    ("ngon", "Bent n-gons", (0.62, 0.36, 1.0)),
    ("flipped", "Flipped", (1.0, 0.1, 0.8)),
    ("seam", "Seams", (0.0, 0.85, 1.0)),
    ("moved", "Moved", (1.0, 0.92, 0.15)),
)
# A triangle this much longer than it is high is a sliver -- but only once it
# is long enough to see: a 30 cm triangle at the end of a kerb is not.
SLIVER_RATIO = 10.0
SLIVER_MIN_EDGE_M = 1.0
# A quad with a corner this sharp is a triangle with an extra vertex, or a spike.
SPIKE_DEG = 10.0
DENSITY_CELL_M = 5.0
DENSITY_MAX_VERTICES = 30
MOVED_M = 0.5
# An n-gon with a corner further than this off the plane of its corners is
# bent: drawn and exported, it is cut into triangles one way or another, and
# the ground in it moves with the cut. The ground's own n-gons are built
# within 0.15 m.
BENT_NGON_M = 0.25
# Ground facing further down than this is upside down.
DOWN_Z = -0.2
# Problem faces are grouped into spots this size for Previous / Next.
SPOT_M = 10.0


def _loop_geometry(co, loops, start, total):
    """Per loop: its face, the next loop round that face, and the edge to it."""
    n_faces = len(start)
    face_of = np.repeat(np.arange(n_faces), total)
    index = np.arange(len(loops))
    following = index + 1
    wrap = following == (start + total)[face_of]
    following[wrap] = start[face_of[wrap]]
    edge = co[loops[following]] - co[loops]
    return face_of, following, edge


def face_problems(
    co, loops, start, total, *, ground,
    sliver_ratio=SLIVER_RATIO, cell_m=DENSITY_CELL_M, max_vertices=DENSITY_MAX_VERTICES,
    surveyed_z=None, moved_m=MOVED_M,
):
    """Which faces of one mesh have which problem, as boolean masks.

    `co` is the vertices in world space, `loops` the vertex of each face
    corner and `start`/`total` each face's corners, as Blender stores them.
    `ground` says whether this is a piece of the ground (facing down is wrong
    for that, and for nothing else). `surveyed_z` is each vertex's built
    height in the same frame as `co`, where the mesh carries one.
    """
    co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
    loops = np.asarray(loops, dtype=np.int64)
    start = np.asarray(start, dtype=np.int64)
    total = np.asarray(total, dtype=np.int64)
    n_faces = len(start)
    out = {name: np.zeros(n_faces, dtype=bool) for name, _, _ in PROBLEMS}
    if not n_faces:
        return out, np.zeros((0, 3))

    face_of, following, edge = _loop_geometry(co, loops, start, total)
    length = np.linalg.norm(edge, axis=1)
    # Newell's normal, about each face's first corner: its length is twice
    # the area, for a triangle, a quad or anything else.
    origin = co[loops[start]][face_of]
    here = co[loops] - origin
    there = co[loops[following]] - origin
    newell = np.zeros((n_faces, 3))
    np.add.at(newell, face_of, np.cross(here, there))
    area = 0.5 * np.linalg.norm(newell, axis=1)
    normal = newell / np.maximum(2.0 * area, 1e-12)[:, None]
    centroid = np.zeros((n_faces, 3))
    np.add.at(centroid, face_of, co[loops])
    centroid /= total[:, None]

    longest = np.zeros(n_faces)
    np.maximum.at(longest, face_of, length)

    # Each corner's angle, between the edge in and the edge out.
    previous = np.empty_like(following)
    previous[following] = np.arange(len(loops))
    into = -edge[previous]
    cosine = np.einsum("ij,ij->i", into, edge) / np.maximum(
        np.linalg.norm(into, axis=1) * length, 1e-12
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    sharpest = np.full(n_faces, 180.0)
    np.minimum.at(sharpest, face_of, angle)

    triangle = total == 3
    height = 2.0 * area / np.maximum(longest, 1e-12)
    out["sliver"] = (
        (triangle & (longest > SLIVER_MIN_EDGE_M) & (longest > sliver_ratio * height))
        | ((total == 4) & (sharpest < SPIKE_DEG) & (longest > SLIVER_MIN_EDGE_M))
        | (area < 1e-6)
    )
    # Off the plane through the face's middle square to its normal.
    off_plane = np.abs(np.einsum("ij,ij->i", co[loops] - centroid[face_of], normal[face_of]))
    bent = np.zeros(n_faces)
    np.maximum.at(bent, face_of, off_plane)
    out["ngon"] = (total > 4) & (bent > BENT_NGON_M)

    if ground:
        out["flipped"] = normal[:, 2] < DOWN_Z
    else:
        # A face wound against its neighbour uses their shared edge in the
        # same direction the neighbour does.
        a = loops.astype(np.int64)
        b = loops[following].astype(np.int64)
        key = a * (len(co) + 1) + b
        _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        clash = counts[inverse.ravel()] > 1
        np.logical_or.at(out["flipped"], face_of, clash)

    # Density: places in the plan per square, against the budget. A wall's
    # top and bottom vertex are one place: counted twice, every fence was
    # "too dense" for being upright.
    cell = max(float(cell_m), 1e-3)
    places = np.unique(np.round(co[:, :2] / 0.01).astype(np.int64), axis=0) * 0.01
    keys = np.floor(np.vstack([places, centroid[:, :2]]) / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse[: len(places)], minlength=int(inverse.max()) + 1)
    out["dense"] = counts[inverse[len(places):]] > int(max_vertices)

    if surveyed_z is not None:
        moved = np.abs(co[:, 2] - np.asarray(surveyed_z, dtype=np.float64)) > moved_m
        np.logical_or.at(out["moved"], face_of, moved[loops])
    return out, centroid


def boundary_vertices(loops, start, total):
    """The vertices on an open edge of one mesh: used by one face only."""
    loops = np.asarray(loops, dtype=np.int64)
    start = np.asarray(start, dtype=np.int64)
    total = np.asarray(total, dtype=np.int64)
    if not len(loops):
        return np.zeros(0, dtype=np.int64)
    face_of = np.repeat(np.arange(len(start)), total)
    following = np.arange(len(loops)) + 1
    wrap = following == (start + total)[face_of]
    following[wrap] = start[face_of[wrap]]
    a, b = loops, loops[following]
    edges = np.sort(np.column_stack([a, b]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[counts == 1].ravel())


def seam_vertices(pieces, *, tolerance_m=0.01, step_m=0.02, border_m=0.05):
    """Where the pieces of the ground do not meet, per piece.

    `pieces` is a list of (world vertices, boundary vertex indices). The
    ground is one sheet cut into pieces by layer, so every open edge of one
    piece is an open edge of the next, vertex for vertex -- except round the
    outside of the tile. A boundary vertex with no partner at its place in
    plan is a crack or a T-junction; one whose partner is at another height
    is a step. Returns, per piece, the indices of the vertices that are
    neither.
    """
    if not pieces:
        return []
    owner, index, points = [], [], []
    for number, (co, boundary) in enumerate(pieces):
        co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
        boundary = np.asarray(boundary, dtype=np.int64)
        owner.append(np.full(len(boundary), number))
        index.append(boundary)
        points.append(co[boundary])
    owner = np.concatenate(owner)
    index = np.concatenate(index)
    points = np.vstack(points) if points else np.zeros((0, 3))
    if not len(points):
        return [np.zeros(0, dtype=np.int64) for _ in pieces]

    every = np.vstack([np.asarray(co).reshape(-1, 3) for co, _ in pieces])
    low, high = every[:, :2].min(axis=0), every[:, :2].max(axis=0)
    on_border = np.any(
        (points[:, :2] < low + border_m) | (points[:, :2] > high - border_m), axis=1
    )
    keys = np.round(points[:, :2] / tolerance_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    lowest = np.full(len(counts), np.inf)
    highest = np.full(len(counts), -np.inf)
    np.minimum.at(lowest, inverse, points[:, 2])
    np.maximum.at(highest, inverse, points[:, 2])
    alone = counts[inverse] < 2
    step = (highest - lowest)[inverse] > step_m
    bad = (alone | step) & ~on_border
    return [index[(owner == number) & bad] for number in range(len(pieces))]


def problem_spots(centroids, masks, spot_m=SPOT_M):
    """Problem faces grouped into spots, worst first.

    Each spot is (centre, how many problem faces, the problems there), so
    Previous / Next can step the view from one to the next.
    """
    hits = [(c, m) for c, m in zip(centroids, masks) if len(c)]
    points, kinds = [], []
    for centroid, mask in hits:
        any_problem = np.zeros(len(centroid), dtype=bool)
        for name, _, _ in PROBLEMS:
            any_problem |= mask[name]
        points.append(centroid[any_problem])
        kinds.append(np.column_stack([mask[name][any_problem] for name, _, _ in PROBLEMS]))
    if not points or not sum(len(p) for p in points):
        return []
    points = np.vstack(points)
    kinds = np.vstack(kinds)
    keys = np.floor(points[:, :2] / spot_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    centre = np.zeros((len(counts), 3))
    np.add.at(centre, inverse, points)
    centre /= counts[:, None]
    present = np.zeros((len(counts), len(PROBLEMS)), dtype=bool)
    np.logical_or.at(present, inverse, kinds)
    order = np.argsort(-counts, kind="stable")
    return [
        (tuple(centre[i]), int(counts[i]),
         [PROBLEMS[k][0] for k in np.flatnonzero(present[i])])
        for i in order
    ]


# What the last scan found. Not stored in the file: the overlay is a view of
# the mesh as it is now, and a stale one is worse than none.
_FOUND = {"spots": [], "counts": {}, "faces": {}, "points": [], "objects": {}}
_DRAW = {"handle": None}


def _mesh_arrays(obj):
    mesh = obj.data
    n_vertices = len(mesh.vertices)
    co = np.empty(n_vertices * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    matrix = np.array(obj.matrix_world)
    world = co @ matrix[:3, :3].T + matrix[:3, 3]
    n_faces = len(mesh.polygons)
    start = np.empty(n_faces, dtype=np.int64)
    total = np.empty(n_faces, dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", start)
    mesh.polygons.foreach_get("loop_total", total)
    loops = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loops)
    surveyed = None
    measured = mesh.attributes.get("measured_z")
    if measured is not None and measured.domain == "POINT":
        local = np.empty(n_vertices)
        measured.data.foreach_get("value", local)
        surveyed_local = co.copy()
        surveyed_local[:, 2] = local
        surveyed = (surveyed_local @ matrix[:3, :3].T + matrix[:3, 3])[:, 2]
    return world, loops, start, total, surveyed


def _is_ground(obj):
    return bool(obj.get("maptile_ground")) or obj.get("maptile_role") in (
        "terrain", *GROUND_ROLES
    )


def _scan_objects(context):
    chosen = [o for o in context.selected_objects if o.type == "MESH"]
    if chosen:
        return chosen
    return [
        o for o in context.view_layer.objects
        if o.type == "MESH" and o.visible_get()
        and (_is_ground(o) or o.name.startswith("Barriers_"))
    ]


def _fan(face_points):
    """A face as triangles for drawing: a fan from its first corner."""
    return [
        (face_points[0], face_points[k], face_points[k + 1])
        for k in range(1, len(face_points) - 1)
    ]


class MAPTILE_OT_find_problems(bpy.types.Operator):
    bl_idname = "maptile.find_problems"
    bl_label = "Find problems"
    bl_description = (
        "Colour slivers, dense patches, n-gons, flipped faces, seams and moved "
        "vertices over the model"
    )

    def execute(self, context):
        settings = context.scene.terreinlab_problems
        objects = _scan_objects(context)
        if not objects:
            self.report({"WARNING"}, "nothing to scan: select a mesh")
            return {"CANCELLED"}
        found_faces = {name: [] for name, _, _ in PROBLEMS}
        counts = {name: 0 for name, _, _ in PROBLEMS}
        centroids, masks, ground_pieces, ground_meshes = [], [], [], []
        per_object = {}
        for obj in objects:
            co, loops, start, total, surveyed = _mesh_arrays(obj)
            ground = _is_ground(obj)
            mask, centroid = face_problems(
                co, loops, start, total, ground=ground,
                sliver_ratio=settings.sliver_ratio, cell_m=settings.cell_m,
                max_vertices=settings.max_vertices, surveyed_z=surveyed,
                moved_m=settings.moved_m,
            )
            if ground:
                ground_pieces.append((co, boundary_vertices(loops, start, total)))
                ground_meshes.append((obj, co, loops, start, total, mask, centroid))
            per_object[obj.name] = (co, loops, start, total, mask)
            centroids.append(centroid)
            masks.append(mask)

        points = []
        for (obj, co, loops, start, total, mask, _c), seams in zip(
            ground_meshes, seam_vertices(ground_pieces)
        ):
            if not len(seams):
                continue
            points.extend(co[seams].tolist())
            bad = np.zeros(len(co), dtype=bool)
            bad[seams] = True
            face_of = np.repeat(np.arange(len(start)), total)
            np.logical_or.at(mask["seam"], face_of, bad[loops])

        for name, (co, loops, start, total, mask) in per_object.items():
            for problem, _, _ in PROBLEMS:
                faces = np.flatnonzero(mask[problem])
                counts[problem] += len(faces)
                for f in faces[:200000]:
                    corners = co[loops[start[f]:start[f] + total[f]]]
                    found_faces[problem].extend(_fan(corners))
        _FOUND.update(
            spots=problem_spots(centroids, masks),
            counts=counts,
            faces=found_faces,
            points=points,
            objects={name: value[4] for name, value in per_object.items()},
        )
        settings.index = 0
        _ensure_overlay()
        _redraw(context)
        summary = ", ".join(f"{counts[n]} {label.lower()}" for n, label, _ in PROBLEMS if counts[n])
        self.report(
            {"INFO"},
            f"{len(objects)} objects: {summary or 'no problems found'}",
        )
        return {"FINISHED"}


class MAPTILE_OT_clear_problems(bpy.types.Operator):
    bl_idname = "maptile.clear_problems"
    bl_label = "Clear"
    bl_description = "Hide the problem overlay"

    def execute(self, context):
        _FOUND.update(spots=[], counts={}, faces={}, points=[], objects={})
        _remove_overlay()
        _redraw(context)
        return {"FINISHED"}


class MAPTILE_OT_problem_step(bpy.types.Operator):
    bl_idname = "maptile.problem_step"
    bl_label = "Go to problem"
    bl_description = "Frame the view on the next or previous problem spot"

    step: bpy.props.IntProperty(default=1)

    def execute(self, context):
        spots = _FOUND["spots"]
        if not spots:
            self.report({"WARNING"}, "run Find problems first")
            return {"CANCELLED"}
        settings = context.scene.terreinlab_problems
        settings.index = (settings.index + self.step) % len(spots)
        centre, count, kinds = spots[settings.index]
        for area in (context.screen.areas if context.screen else ()):
            if area.type != "VIEW_3D":
                continue
            view = area.spaces.active.region_3d
            view.view_location = centre
            view.view_distance = max(SPOT_M * 2.5, 20.0)
        self.report({"INFO"}, f"{count} faces: {', '.join(kinds)}")
        return {"FINISHED"}


class MAPTILE_OT_select_problems(bpy.types.Operator):
    bl_idname = "maptile.select_problems"
    bl_label = "Select problem faces"
    bl_description = "Select the problem faces of the active object, in edit mode"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        masks = _FOUND["objects"].get(obj.name) if obj is not None else None
        if masks is None:
            self.report({"WARNING"}, "the active object was not in the last scan")
            return {"CANCELLED"}
        settings = context.scene.terreinlab_problems
        chosen = np.zeros(len(obj.data.polygons), dtype=bool)
        for name, _, _ in PROBLEMS:
            if getattr(settings, f"show_{name}") and len(masks[name]) == len(chosen):
                chosen |= masks[name]
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        mesh = obj.data
        # Faces, and their vertices and edges: edit mode rebuilds the face
        # selection from whatever the vertices still had selected, and showed
        # four thousand faces for 77.
        loops = np.empty(len(mesh.loops), dtype=np.int64)
        mesh.loops.foreach_get("vertex_index", loops)
        start = np.empty(len(mesh.polygons), dtype=np.int64)
        total = np.empty(len(mesh.polygons), dtype=np.int64)
        mesh.polygons.foreach_get("loop_start", start)
        mesh.polygons.foreach_get("loop_total", total)
        face_of = np.repeat(np.arange(len(start)), total)
        vertex_on = np.zeros(len(mesh.vertices), dtype=bool)
        vertex_on[loops[chosen[face_of]]] = True
        edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
        mesh.edges.foreach_get("vertices", edges)
        edges = edges.reshape(-1, 2)
        mesh.vertices.foreach_set("select", vertex_on)
        mesh.edges.foreach_set("select", vertex_on[edges[:, 0]] & vertex_on[edges[:, 1]])
        mesh.polygons.foreach_set("select", chosen)
        context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.object.mode_set(mode="EDIT")
        self.report({"INFO"}, f"{int(chosen.sum())} faces selected")
        return {"FINISHED"}


def _redraw(context):
    for area in context.screen.areas if context.screen else ():
        if area.type == "VIEW_3D":
            area.tag_redraw()


def _draw_overlay():
    import gpu
    from gpu_extras.batch import batch_for_shader

    settings = bpy.context.scene.terreinlab_problems
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(False)
    for name, _, colour in PROBLEMS:
        triangles = _FOUND["faces"].get(name)
        if not triangles or not getattr(settings, f"show_{name}"):
            continue
        points = np.asarray(triangles, dtype=np.float32).reshape(-1, 3)
        points[:, 2] += 0.03          # just above the face, so it is not hidden by it
        batch = batch_for_shader(shader, "TRIS", {"pos": points})
        shader.bind()
        shader.uniform_float("color", (*colour, settings.opacity))
        batch.draw(shader)
    if _FOUND["points"] and settings.show_seam:
        try:
            point_shader = gpu.shader.from_builtin("POINT_UNIFORM_COLOR")
        except (ValueError, KeyError):
            point_shader = shader
        gpu.state.point_size_set(7.0)
        points = np.asarray(_FOUND["points"], dtype=np.float32)
        batch = batch_for_shader(point_shader, "POINTS", {"pos": points})
        point_shader.bind()
        point_shader.uniform_float("color", (*PROBLEMS[4][2], 1.0))
        batch.draw(point_shader)
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set("NONE")


def _ensure_overlay():
    if _DRAW["handle"] is None:
        _DRAW["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_overlay, (), "WINDOW", "POST_VIEW"
        )


def _remove_overlay():
    if _DRAW["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW["handle"], "WINDOW")
        _DRAW["handle"] = None


def _tag_all(self, context):
    _redraw(context)


class TerreinLabProblemSettings(bpy.types.PropertyGroup):
    sliver_ratio: bpy.props.FloatProperty(
        name="Sliver ratio", default=SLIVER_RATIO, min=2.0,
        description="A triangle this many times longer than it is high is a sliver",
    )
    cell_m: bpy.props.FloatProperty(
        name="Density square", default=DENSITY_CELL_M, min=1.0, unit="LENGTH",
        description="The square of the plan the vertex budget applies to",
    )
    max_vertices: bpy.props.IntProperty(
        name="Vertices per square", default=DENSITY_MAX_VERTICES, min=4,
        description="More vertices than this in one square is too dense",
    )
    moved_m: bpy.props.FloatProperty(
        name="Moved by", default=MOVED_M, min=0.01, unit="LENGTH",
        description="Further than this from the surveyed height is moved",
    )
    opacity: bpy.props.FloatProperty(
        name="Opacity", default=0.55, min=0.1, max=1.0, update=_tag_all,
    )
    index: bpy.props.IntProperty(default=0)
    show_sliver: bpy.props.BoolProperty(name="Slivers", default=True, update=_tag_all)
    show_dense: bpy.props.BoolProperty(name="Too dense", default=True, update=_tag_all)
    show_ngon: bpy.props.BoolProperty(name="Bent n-gons", default=True, update=_tag_all)
    show_flipped: bpy.props.BoolProperty(name="Flipped", default=True, update=_tag_all)
    show_seam: bpy.props.BoolProperty(name="Seams", default=True, update=_tag_all)
    show_moved: bpy.props.BoolProperty(name="Moved", default=True, update=_tag_all)


class MAPTILE_PT_panel(bpy.types.Panel):
    bl_label = "TerreinLab"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "TerreinLab"

    def draw(self, context):
        column = self.layout.column(align=True)
        column.operator(MAPTILE_OT_fit_terrain.bl_idname, icon="MOD_SHRINKWRAP")
        column.operator(MAPTILE_OT_reset_terrain.bl_idname, icon="LOOP_BACK")
        column.operator(MAPTILE_OT_drop_to_ground.bl_idname, icon="TRIA_DOWN_BAR")
        column.separator()
        column.operator(MAPTILE_OT_export_unity.bl_idname, icon="EXPORT")


class MAPTILE_PT_problems(bpy.types.Panel):
    bl_label = "Problem areas"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "TerreinLab"
    bl_parent_id = "MAPTILE_PT_panel"

    def draw(self, context):
        settings = context.scene.terreinlab_problems
        layout = self.layout
        row = layout.row(align=True)
        row.operator(MAPTILE_OT_find_problems.bl_idname, icon="VIEWZOOM")
        row.operator(MAPTILE_OT_clear_problems.bl_idname, text="", icon="X")

        counts = _FOUND["counts"]
        column = layout.column(align=True)
        for name, label, _ in PROBLEMS:
            row = column.row(align=True)
            row.prop(settings, f"show_{name}", text=label)
            row.label(text=str(counts.get(name, "–")))

        spots = _FOUND["spots"]
        row = layout.row(align=True)
        previous = row.operator(MAPTILE_OT_problem_step.bl_idname, text="", icon="TRIA_LEFT")
        previous.step = -1
        row.label(text=f"{settings.index + 1} of {len(spots)}" if spots else "no spots")
        following = row.operator(MAPTILE_OT_problem_step.bl_idname, text="", icon="TRIA_RIGHT")
        following.step = 1
        layout.operator(MAPTILE_OT_select_problems.bl_idname, icon="RESTRICT_SELECT_OFF")

        box = layout.box()
        box.label(text="Thresholds")
        box.prop(settings, "sliver_ratio")
        box.prop(settings, "cell_m")
        box.prop(settings, "max_vertices")
        box.prop(settings, "moved_m")
        box.prop(settings, "opacity")


CLASSES = (
    TerreinLabProblemSettings,
    MAPTILE_OT_fit_terrain,
    MAPTILE_OT_reset_terrain,
    MAPTILE_OT_drop_to_ground,
    MAPTILE_OT_export_unity,
    MAPTILE_OT_find_problems,
    MAPTILE_OT_clear_problems,
    MAPTILE_OT_problem_step,
    MAPTILE_OT_select_problems,
    MAPTILE_PT_panel,
    MAPTILE_PT_problems,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.terreinlab_problems = bpy.props.PointerProperty(
        type=TerreinLabProblemSettings
    )


def unregister():
    _remove_overlay()
    del bpy.types.Scene.terreinlab_problems
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
