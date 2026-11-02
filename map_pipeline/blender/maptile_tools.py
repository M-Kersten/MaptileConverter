"""Maptile tools: the edits the model needs after a person has touched it.

Ships next to every model.blend, and inside it as a text block. Install it once
as an add-on (Edit > Preferences > Add-ons > Install..., pick this file) or run
the text block in the Scripting workspace. It adds a "Maptile" tab to the 3D
view's sidebar (N) with four buttons:

Fit terrain under roads
    After moving, widening or adding road, push the terrain down wherever it
    would show through. The same rule the pipeline used: each point of road
    surface lowers the three corners of the terrain triangle under it just
    enough, the corners under the road most. Only ever lowers.

Reset terrain to survey
    Put every terrain vertex back at the surveyed height it was built from,
    undoing every fit and every sculpt. Run Fit afterwards.

Drop selected onto ground
    Move each selected object straight down (or up) so its origin sits on the
    road or terrain under it. Buildings, trees and road pieces all have their
    origin on the ground, so this is how something moved sideways gets back
    onto the surface.

Export FBX for Unity
    The pipeline's own export settings, so a re-export from an edited model
    lands in Unity exactly like the original did.

Only bpy and numpy are used, both of which Blender bundles.
"""

bl_info = {
    "name": "Maptile tools",
    "author": "MaptileConverter",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Maptile",
    "description": "Keep roads and terrain fitted together while editing",
    "category": "Object",
}

import bpy
import numpy as np

# Must match src/roads.py and src/elevation.py.
TERRAIN_CLEARANCE_M = 0.05
HIDDEN_VERTEX_WEIGHT = 4.0
ROAD_CELL_M = 0.25
ROAD_COLLECTION = "Roads"
TERRAIN_NAME = "Terrain"


def _terrain_grid(obj):
    """The terrain's vertices as an (n, n) grid, with its axes, or None."""
    mesh = obj.data
    count = len(mesh.vertices)
    n = int(round(count ** 0.5))
    if n * n != count or n < 2:
        return None
    co = np.empty(count * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ np.asarray(obj.matrix_world)[:3, :3].T + np.asarray(
        obj.matrix_world
    )[:3, 3]
    grid = co.reshape(n, n, 3)
    xs, ys = grid[0, :, 0], grid[:, 0, 1]
    # A regular grid keeps its rows and columns. Anything else has been
    # remeshed, and the fast path below does not apply.
    if not (np.allclose(grid[:, :, 0], xs[None, :], atol=1e-3)
            and np.allclose(grid[:, :, 1], ys[:, None], atol=1e-3)):
        return None
    return grid, xs, ys


def _road_triangles(context):
    """World-space triangles of every road surface, kerbs and skirts left out."""
    collection = bpy.data.collections.get(ROAD_COLLECTION)
    if collection is None:
        return np.zeros((0, 3, 3))
    depsgraph = context.evaluated_depsgraph_get()
    out = []
    for obj in collection.all_objects:
        if obj.type != "MESH":
            continue
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


def fit_terrain(grid, xs, ys, points, clearance=TERRAIN_CLEARANCE_M):
    """Lower the grid heights so every point clears it; as src/elevation.py."""
    n = len(xs)
    heights = grid[:, :, 2].astype(np.float64).ravel()
    if not len(points):
        return heights.reshape(n, n), 0
    step_x = (xs[-1] - xs[0]) / (n - 1)
    step_y = (ys[-1] - ys[0]) / (n - 1)
    fx = np.clip((points[:, 0] - xs[0]) / step_x, 0, n - 1)
    fy = np.clip((points[:, 1] - ys[0]) / step_y, 0, n - 1)
    i0 = np.minimum(np.floor(fx).astype(np.int64), n - 2)
    j0 = np.minimum(np.floor(fy).astype(np.int64), n - 2)
    tx, ty = fx - i0, fy - j0
    lower = tx >= ty
    base = j0 * n + i0
    corners = np.stack(
        [base, np.where(lower, base + 1, base + n + 1), np.where(lower, base + n + 1, base + n)],
        axis=1,
    )
    weights = np.stack(
        [np.where(lower, 1 - tx, 1 - ty), np.where(lower, tx - ty, tx),
         np.where(lower, ty, ty - tx)],
        axis=1,
    )
    excess = (weights * heights[corners]).sum(axis=1) - (points[:, 2] - clearance)
    bad = excess > 0.0
    drop = np.zeros(heights.shape)
    if bad.any():
        nearest = np.rint(fy).astype(np.int64) * n + np.rint(fx).astype(np.int64)
        hidden = np.bincount(nearest, minlength=n * n) > 0
        w = weights[bad]
        share = w * np.where(hidden[corners[bad]], HIDDEN_VERTEX_WEIGHT, 1.0)
        need = excess[bad, None] * share / np.maximum((w * share).sum(axis=1), 1e-12)[:, None]
        np.maximum.at(drop, corners[bad].ravel(), need.ravel())
    return (heights - drop).reshape(n, n), int((drop > 1e-6).sum())


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
        terrain = bpy.data.objects.get(TERRAIN_NAME)
        if terrain is None:
            self.report({"ERROR"}, f"no object called {TERRAIN_NAME}")
            return {"CANCELLED"}
        found = _terrain_grid(terrain)
        if found is None:
            self.report({"ERROR"}, "the terrain is no longer a regular grid")
            return {"CANCELLED"}
        grid, xs, ys = found
        points = _underside_points(_road_triangles(context), ROAD_CELL_M)
        heights, moved = fit_terrain(grid, xs, ys, points)
        _write_heights(terrain, heights)
        self.report({"INFO"}, f"lowered {moved} terrain vertices")
        return {"FINISHED"}


class MAPTILE_OT_reset_terrain(bpy.types.Operator):
    bl_idname = "maptile.reset_terrain"
    bl_label = "Reset terrain to survey"
    bl_description = "Put every terrain vertex back at its surveyed height"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        terrain = bpy.data.objects.get(TERRAIN_NAME)
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
        terrain = bpy.data.objects.get(TERRAIN_NAME)
        if terrain is not None:
            targets.add(terrain.name)
        roads = bpy.data.collections.get(ROAD_COLLECTION)
        if roads is not None:
            targets.update(o.name for o in roads.all_objects)
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


def export_unity_fbx(path):
    """The pipeline's export settings: see blender/process.py export_fbx."""
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
