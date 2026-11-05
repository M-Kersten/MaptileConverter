"""Maptile tools: the edits the model needs after a person has touched it.

Ships next to every model.blend, and inside it as a text block. Install it once
as an add-on (Edit > Preferences > Add-ons > Install..., pick this file) or run
the text block in the Scripting workspace. It adds a "Maptile" tab to the 3D
view's sidebar (N) with four buttons:

Fit terrain under roads
    After moving, widening or adding road, push the terrain down wherever it
    would show through. The same rule the pipeline used: each point of road
    surface lowers the three corners of the terrain triangle under it just
    enough, the corners under the road most. Only ever lowers, so a sculpt
    survives it; to get ground back from under a road you moved away, Reset
    first and Fit after. Works on the terrain whatever its faces: the grid,
    the rows along the outlines, or anything you have remeshed. In the
    layered style the roads, walkways and grass are wrapped or draped onto
    the terrain and follow it by themselves; only one whose Shrinkwrap or
    drape was removed or applied is fitted under.

Reset terrain to survey
    Put every terrain vertex back at the surveyed height it was built from,
    undoing every fit and every sculpt. Run Fit afterwards.

Drop selected onto ground
    Move each selected object straight down (or up) so its origin sits on the
    road, walkway, grass or terrain under it. Buildings, trees and road
    pieces all have their origin on the ground, so this is how something
    moved sideways gets back onto the surface.

Export FBX for Unity
    The pipeline's own export settings, so a re-export from an edited model
    lands in Unity exactly like the original did.

Only bpy and numpy are used, both of which Blender bundles.
"""

bl_info = {
    "name": "Maptile tools",
    "author": "MaptileConverter",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Maptile",
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
    """Whether a layer is wrapped or draped onto the terrain, so moves with it.

    Roads and lawns carry a Shrinkwrap; a kerbed walkway the pipeline's
    Maptile_Drape geometry nodes, which read the terrain the same way.
    """
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


def _terrain_mesh(obj):
    """The terrain as world-space vertices, triangles, and which vertices show.

    Triangles as the export draws them: a quad split from its first corner to
    its third, a larger face fanned from its first corner. A vertex shows if
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
    fans = totals - 2
    face = np.repeat(np.arange(len(totals)), fans)
    step = np.arange(int(fans.sum())) - np.repeat(np.cumsum(fans) - fans, fans) + 1
    first = starts[face]
    triangles = np.column_stack(
        [loops[first], loops[first + step], loops[first + step + 1]]
    )
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
        measured = terrain and terrain.data.attributes.get("measured_z")
        if not measured:
            self.report({"ERROR"}, "the terrain carries no surveyed heights")
            return {"CANCELLED"}
        values = np.empty(len(terrain.data.vertices))
        measured.data.foreach_get("value", values)
        co = np.empty(len(values) * 3)
        terrain.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        co[:, 2] = values
        terrain.data.vertices.foreach_set("co", co.ravel())
        terrain.data.update()
        self.report({"INFO"}, "terrain back at the surveyed heights")
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


class MAPTILE_PT_panel(bpy.types.Panel):
    bl_label = "Maptile"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Maptile"

    def draw(self, context):
        column = self.layout.column(align=True)
        column.operator(MAPTILE_OT_fit_terrain.bl_idname, icon="MOD_SHRINKWRAP")
        column.operator(MAPTILE_OT_reset_terrain.bl_idname, icon="LOOP_BACK")
        column.operator(MAPTILE_OT_drop_to_ground.bl_idname, icon="TRIA_DOWN_BAR")
        column.separator()
        column.operator(MAPTILE_OT_export_unity.bl_idname, icon="EXPORT")


CLASSES = (
    MAPTILE_OT_fit_terrain,
    MAPTILE_OT_reset_terrain,
    MAPTILE_OT_drop_to_ground,
    MAPTILE_OT_export_unity,
    MAPTILE_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
