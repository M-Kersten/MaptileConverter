"""Headless Blender stage: intermediate files in, textured FBX out.

Runs two ways, and figures out which on its own:

    blender --background --python blender/process.py -- --work W --out O
    python  blender/process.py --work W --out O        # with the pip `bpy`

All the CityJSON and raster work already happened upstream. This script only
reads ``scene.json``, two npz files and the textures, so it stays thin and needs
no GUI add-on.

Axes: Blender is built here as X = RD easting, Y = RD northing, Z = NAP height.
The FBX exporter's Unity preset (forward -Z, up Y) turns that into Unity's
X = east, Y = up, Z = north.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np

# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> tuple[Path, Path]:
    """Read --work and --out, from after the ``--`` separator when present."""
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        # Plain `python process.py ...`: drop the script name.
        argv = argv[1:] if argv and argv[0].endswith(".py") else argv

    import argparse

    parser = argparse.ArgumentParser(description="Build the scene and export FBX.")
    parser.add_argument("--work", required=True, help="intermediate file directory")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args(argv)
    return Path(args.work), Path(args.out)


def log(message: str) -> None:
    print(f"[blender] {message}", flush=True)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def reset_scene() -> None:
    """Empty the startup file so a --background run starts from nothing."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.objects,
    ):
        for item in list(collection):
            collection.remove(item)


def make_textured_material(
    name: str, image_path: Path, roughness: float, normal_path: Path | None = None
):
    """Principled BSDF with `image_path` wired into base colour.

    FBX carries the base-colour texture reference and the scalar parameters, so
    this survives the trip into Unity as an albedo map. A normal map, when given,
    goes through a Normal Map node, which the FBX exporter writes to the
    material's bump slot.
    """
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        output = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(image_path), check_existing=True)
    texture.image.name = image_path.name
    texture.extension = "REPEAT"
    texture.location = (-400, 0)
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])

    if normal_path is not None:
        normal_texture = nodes.new("ShaderNodeTexImage")
        normal_texture.image = bpy.data.images.load(
            str(normal_path), check_existing=True
        )
        normal_texture.image.name = normal_path.name
        # Normals are vectors, not colour: reading them through sRGB would bend
        # every one of them.
        normal_texture.image.colorspace_settings.name = "Non-Color"
        normal_texture.extension = "REPEAT"
        normal_texture.location = (-700, -320)

        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.location = (-380, -320)
        links.new(normal_texture.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = roughness
    # Flat diffuse reads better than a shiny default for both aerial and facade.
    for socket in ("Specular IOR Level", "Specular"):
        if socket in bsdf.inputs:
            bsdf.inputs[socket].default_value = 0.15
            break

    return material


def build_mesh_object(
    name: str,
    vertices: np.ndarray,
    loop_vertex_indices: np.ndarray,
    loop_starts: np.ndarray,
    loop_totals: np.ndarray,
    uvs: np.ndarray,
    material_indices: np.ndarray,
    materials: list,
    shade_smooth: bool,
):
    """Create a mesh through the bulk API.

    ``foreach_set`` moves whole numpy arrays across in one call, which keeps a
    few hundred thousand triangles well under a second. Building the same mesh
    element by element takes minutes.
    """
    mesh = bpy.data.meshes.new(name)
    n_vertices = len(vertices)
    n_loops = len(loop_vertex_indices)
    n_polygons = len(loop_starts)

    mesh.vertices.add(n_vertices)
    mesh.vertices.foreach_set("co", vertices.astype(np.float32).ravel())

    mesh.loops.add(n_loops)
    mesh.loops.foreach_set("vertex_index", loop_vertex_indices.astype(np.int32))

    mesh.polygons.add(n_polygons)
    mesh.polygons.foreach_set("loop_start", loop_starts.astype(np.int32))
    try:
        # Removed in some Blender versions, where the total is implied by the
        # next polygon's loop_start.
        mesh.polygons.foreach_set("loop_total", loop_totals.astype(np.int32))
    except Exception:  # noqa: BLE001 - version difference, not a failure
        pass

    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", uvs.astype(np.float32).ravel())

    for material in materials:
        mesh.materials.append(material)
    if len(materials) > 1:
        mesh.polygons.foreach_set("material_index", material_indices.astype(np.int32))

    mesh.update()
    mesh.validate(verbose=False)

    if shade_smooth:
        mesh.shade_smooth()
    else:
        mesh.shade_flat()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def planar_uv(points_xy: np.ndarray, aerial_bbox_local: list[float]) -> np.ndarray:
    """Top-down UV from local XY, normalised to the aerial footprint.

    Both the terrain and the roofs use this, which is what lets roofs pick up
    real photo texture and blend into the ground around them.
    """
    xmin, ymin, xmax, ymax = aerial_bbox_local
    u = (points_xy[:, 0] - xmin) / (xmax - xmin)
    v = (points_xy[:, 1] - ymin) / (ymax - ymin)
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def build_terrain(scene: dict, work_dir: Path, material) -> object:
    """Grid mesh from the AHN heights, draped with the aerial photo."""
    data = np.load(work_dir / scene["terrain"]["file"])
    heights = data["heights"].astype(np.float64)
    xs = data["xs"]
    ys = data["ys"]

    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    n = heights.shape[0]

    # Row 0 of the height grid is the southern edge, so a plain meshgrid lines
    # up with Blender's +Y = north without any flipping.
    grid_x, grid_y = np.meshgrid(xs - origin_x, ys - origin_y)
    vertices = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), (heights - z_offset).ravel()]
    )

    # Quads over the grid, wound counter-clockwise so the normals point up.
    row = np.arange(n - 1)
    col = np.arange(n - 1)
    cc, rr = np.meshgrid(col, row)
    bottom_left = (rr * n + cc).ravel()
    quads = np.column_stack(
        [
            bottom_left,
            bottom_left + 1,
            bottom_left + n + 1,
            bottom_left + n,
        ]
    )

    n_quads = len(quads)
    loop_vertex_indices = quads.ravel()
    loop_starts = np.arange(0, n_quads * 4, 4)
    loop_totals = np.full(n_quads, 4)

    uvs = planar_uv(vertices[loop_vertex_indices][:, :2], scene["aerial"]["bbox_local"])

    obj = build_mesh_object(
        "Terrain",
        vertices,
        loop_vertex_indices,
        loop_starts,
        loop_totals,
        uvs,
        np.zeros(n_quads),
        [material],
        shade_smooth=True,
    )
    log(f"terrain: {n}x{n} vertices, {n_quads} quads")
    return obj


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------


def wall_uvs(
    triangles: np.ndarray,
    building_index: np.ndarray,
    ground_z_local: np.ndarray,
    floor_height: np.ndarray,
    tile_width_m: float,
) -> np.ndarray:
    """UVs that keep facade windows on storey lines.

    U runs along the wall in metres, so window spacing stays constant however
    wide the wall is. V counts storeys from the building's own ground level,
    using an effective storey height of ``height / floors`` so the top storey
    finishes flush with the roof instead of being cut mid-window.
    """
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge1, edge2)

    # Horizontal tangent of the wall: the normal rotated 90 degrees about Z.
    tangent = np.column_stack([normals[:, 1], -normals[:, 0]])
    lengths = np.linalg.norm(tangent, axis=1)
    # Near-horizontal faces have no meaningful tangent; project along +X.
    degenerate = lengths < 1e-9
    tangent[degenerate] = [1.0, 0.0]
    lengths[degenerate] = 1.0
    tangent /= lengths[:, None]

    # Broadcast the per-triangle frame across the triangle's three corners.
    tangent_per_corner = np.repeat(tangent, 3, axis=0)
    ground_per_corner = np.repeat(ground_z_local[building_index], 3)
    floor_per_corner = np.repeat(floor_height[building_index], 3)

    corners = triangles.reshape(-1, 3)
    u = (corners[:, :2] * tangent_per_corner).sum(axis=1) / tile_width_m
    v = (corners[:, 2] - ground_per_corner) / floor_per_corner
    return np.column_stack([u, v])


def build_buildings(
    scene: dict,
    work_dir: Path,
    facade_materials: list,
    aerial_material,
    ground_material=None,
):
    """One mesh holding every building, with facade and aerial material slots."""
    data = np.load(work_dir / scene["buildings"]["file"])
    wall_tris = data["wall_tris"]
    roof_tris = data["roof_tris"]
    wall_owner = data["wall_building"]
    roof_owner = data["roof_building"]
    has_ground = "ground_wall_tris" in data.files and ground_material is not None
    ground_tris = data["ground_wall_tris"] if has_ground else np.zeros((0, 3, 3))
    ground_owner = (
        data["ground_wall_building"] if has_ground else np.zeros((0,), dtype=np.int32)
    )
    ground_z_nap = data["ground_z_nap"]
    # Storeys divide the wall, not the roof ridge, so the top row of windows
    # finishes at the eaves instead of being cut in half by them.
    wall_height_m = (
        data["wall_height_m"] if "wall_height_m" in data.files else data["height_m"]
    )
    floors = data["floors"].astype(np.float64)
    style_index = data["style_index"] if "style_index" in data.files else None

    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    facade_cfg = scene["facade"]

    def to_local(tris: np.ndarray) -> np.ndarray:
        if len(tris) == 0:
            return np.zeros((0, 3, 3))
        out = tris.copy()
        out[:, :, 0] -= origin_x
        out[:, :, 1] -= origin_y
        out[:, :, 2] -= z_offset
        return out

    walls = to_local(wall_tris)
    roofs = to_local(roof_tris)
    grounds = to_local(ground_tris)
    n_walls, n_roofs, n_grounds = len(walls), len(roofs), len(grounds)
    if n_walls + n_roofs + n_grounds == 0:
        raise SystemExit("buildings.npz holds no triangles")

    ground_z_local = ground_z_nap - z_offset
    # An effective storey height makes the top storey finish flush with the top
    # of the wall rather than being clipped mid-window.
    effective_floor = np.where(
        floors > 0, wall_height_m / np.maximum(floors, 1), float(facade_cfg["floor_height_m"])
    )
    effective_floor = np.clip(effective_floor, 1.5, 8.0)

    materials = list(facade_materials) + [aerial_material]
    aerial_slot = len(facade_materials)
    if has_ground:
        materials.append(ground_material)
    ground_slot = len(materials) - 1

    parts = [chunk for chunk in (walls, roofs, grounds) if len(chunk)]
    triangles = np.concatenate(parts, axis=0)
    corners = triangles.reshape(-1, 3)

    uv_chunks = []
    if n_walls:
        uv_chunks.append(
            wall_uvs(
                walls,
                wall_owner,
                ground_z_local,
                effective_floor,
                float(facade_cfg["tile_width_m"]),
            )
        )
    if n_roofs:
        uv_chunks.append(
            planar_uv(roofs.reshape(-1, 3)[:, :2], scene["aerial"]["bbox_local"])
        )
    if n_grounds:
        # The ground storey gets exactly one tile vertically, so the shopfront
        # spans it instead of repeating inside it.
        storey = np.full(
            len(ground_z_local), float(facade_cfg.get("ground_floor_height_m", 3.6))
        )
        uv_chunks.append(
            wall_uvs(
                grounds,
                ground_owner,
                ground_z_local,
                storey,
                float(facade_cfg["tile_width_m"]),
            )
        )
    uvs = np.concatenate(uv_chunks, axis=0)

    material_indices = np.empty(n_walls + n_roofs + n_grounds, dtype=np.int32)
    if n_walls:
        if style_index is not None and len(facade_materials) > 1:
            material_indices[:n_walls] = np.clip(
                style_index[wall_owner], 0, len(facade_materials) - 1
            )
        else:
            material_indices[:n_walls] = 0
    material_indices[n_walls : n_walls + n_roofs] = aerial_slot
    material_indices[n_walls + n_roofs :] = ground_slot

    n_triangles = n_walls + n_roofs + n_grounds
    loop_vertex_indices = np.arange(n_triangles * 3)
    loop_starts = np.arange(0, n_triangles * 3, 3)
    loop_totals = np.full(n_triangles, 3)

    obj = build_mesh_object(
        "Buildings",
        corners,
        loop_vertex_indices,
        loop_starts,
        loop_totals,
        uvs,
        material_indices,
        materials,
        shade_smooth=False,
    )
    log(
        f"buildings: {len(ground_z_nap)} buildings, {n_walls} wall triangles, "
        f"{n_roofs} roof triangles, {n_grounds} ground-storey triangles"
    )
    return obj


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


def _octahedron_canopy(subdivisions: int = 1):
    """A low-poly ball: an octahedron, subdivided and pushed onto a sphere.

    One subdivision is 32 triangles, which reads as a canopy at street distance
    without putting a real sphere on every tree in the area.
    """
    verts = [
        (0, 0, 1), (1, 0, 0), (0, 1, 0),
        (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
        (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4),
    ]
    verts = [np.asarray(v, dtype=np.float64) for v in verts]

    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}
        new_faces = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                point = (verts[a] + verts[b]) * 0.5
                verts.append(point / np.linalg.norm(point))
                midpoints[key] = len(verts) - 1
            return midpoints[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        faces = new_faces

    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def build_trees(scene: dict, work_dir: Path, tree_material):
    """One mesh holding every tree: a trunk prism plus a low-poly canopy.

    Solid geometry rather than crossed billboards, so nothing depends on alpha
    settings surviving the FBX trip and being configured again in Unity.
    """
    tree_file = scene.get("trees", {}).get("file")
    if not tree_file or not (work_dir / tree_file).is_file():
        return None

    data = np.load(work_dir / tree_file)
    xy = data["xy"]
    if len(xy) == 0:
        return None

    ground = data["ground_z_nap"]
    heights = data["height_m"]
    crowns = data["crown_radius_m"]
    trunks = data["trunk_height_m"]

    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])

    canopy_verts, canopy_faces = _octahedron_canopy(1)
    trunk_sides = 5
    angles = np.linspace(0, 2 * np.pi, trunk_sides, endpoint=False)

    # UVs address one atlas: bark on the left half, foliage on the right.
    bark_uv = np.array([0.25, 0.5])
    leaf_uv = np.array([0.75, 0.5])

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_uvs: list[np.ndarray] = []
    offset = 0
    rng = np.random.default_rng(4242)

    for index in range(len(xy)):
        base_x = float(xy[index, 0]) - origin_x
        base_y = float(xy[index, 1]) - origin_y
        base_z = float(ground[index]) - z_offset
        height = float(heights[index])
        crown_r = float(crowns[index])
        trunk_h = float(trunks[index])

        # A little variation so a street of trees does not look cloned.
        spin = float(rng.random()) * 2 * np.pi
        squash = 0.85 + 0.3 * float(rng.random())

        trunk_r = max(0.08, crown_r * 0.11)
        ring_x = base_x + trunk_r * np.cos(angles + spin)
        ring_y = base_y + trunk_r * np.sin(angles + spin)
        canopy_base = base_z + trunk_h

        lower = np.column_stack([ring_x, ring_y, np.full(trunk_sides, base_z)])
        upper = np.column_stack(
            [ring_x, ring_y, np.full(trunk_sides, canopy_base + crown_r * 0.3)]
        )
        all_verts.append(np.vstack([lower, upper]))

        for side in range(trunk_sides):
            nxt = (side + 1) % trunk_sides
            all_faces.append(
                np.array(
                    [
                        [offset + side, offset + nxt, offset + trunk_sides + nxt],
                        [offset + side, offset + trunk_sides + nxt, offset + trunk_sides + side],
                    ]
                )
            )
        all_uvs.append(np.tile(bark_uv, (trunk_sides * 2 * 3, 1)))
        offset += trunk_sides * 2

        # Canopy: an ellipsoid sitting on top of the trunk.
        crown_centre = np.array(
            [base_x, base_y, canopy_base + (height - trunk_h) * 0.5]
        )
        radii = np.array(
            [crown_r, crown_r * squash, max(0.6, (height - trunk_h) * 0.5)]
        )
        all_verts.append(canopy_verts * radii + crown_centre)
        all_faces.append(canopy_faces + offset)
        all_uvs.append(np.tile(leaf_uv, (len(canopy_faces) * 3, 1)))
        offset += len(canopy_verts)

    vertices = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    uvs = np.vstack(all_uvs)

    n_triangles = len(faces)
    loop_vertex_indices = faces.reshape(-1)
    loop_starts = np.arange(0, n_triangles * 3, 3)
    loop_totals = np.full(n_triangles, 3)

    obj = build_mesh_object(
        "Trees",
        vertices,
        loop_vertex_indices,
        loop_starts,
        loop_totals,
        uvs,
        np.zeros(n_triangles),
        [tree_material],
        shade_smooth=False,
    )
    log(f"trees: {len(xy)} trees, {n_triangles} triangles")
    return obj


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_fbx(out_path: Path) -> None:
    """Write the scene to FBX using the Blender-to-Unity axis convention.

    Blender is right-handed Z-up and Unity is left-handed Y-up. Exporting with
    forward -Z and up Y sends east to Unity's X, NAP height to Y and north to Z,
    and the handedness change cancels against the axis change, so the model is
    not mirrored.

    Texture paths are stripped to bare filenames because the images are written
    next to the FBX, which is where Unity looks for them.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for obj in bpy.context.scene.objects:
        obj.select_set(True)

    bpy.ops.export_scene.fbx(
        filepath=str(out_path),
        use_selection=False,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        global_scale=1.0,
        axis_forward="-Z",
        axis_up="Y",
        # Left off deliberately: every object already carries an identity
        # transform, and the "apply transform" path is flagged experimental and
        # is a known source of subtle Unity import differences.
        bake_space_transform=False,
        object_types={"MESH"},
        use_mesh_modifiers=False,
        mesh_smooth_type="FACE",
        use_tspace=True,
        path_mode="STRIP",
        embed_textures=False,
        bake_anim=False,
    )
    log(f"exported {out_path}")


def copy_textures(
    scene: dict, work_dir: Path, out_dir: Path
) -> tuple[Path, list[tuple[Path, Path | None]], Path | None]:
    """Place the textures beside the FBX before the materials reference them."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def copy(name: str) -> Path:
        src = work_dir / name
        dst = out_dir / name
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return dst

    aerial_src = work_dir / scene["aerial"]["file"]
    aerial_dst = out_dir / scene["export"]["aerial_name"]
    if aerial_src.resolve() != aerial_dst.resolve():
        shutil.copyfile(aerial_src, aerial_dst)

    normals = scene["facade"].get("normal_files") or [None] * len(
        scene["facade"]["files"]
    )
    facades = [
        (copy(name), copy(normal) if normal else None)
        for name, normal in zip(scene["facade"]["files"], normals)
    ]

    tree_texture = scene.get("trees", {}).get("texture")
    tree_dst = copy(tree_texture) if tree_texture else None
    return aerial_dst, facades, tree_dst


def main() -> int:
    work_dir, out_dir = parse_args(list(sys.argv))
    work_dir = work_dir.resolve()
    out_dir = out_dir.resolve()

    scene_path = work_dir / "scene.json"
    if not scene_path.is_file():
        raise SystemExit(f"missing {scene_path}; run the Python stages first")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))

    log(f"building scene {scene['name']!r}")
    reset_scene()

    aerial_texture, facade_textures, tree_texture = copy_textures(
        scene, work_dir, out_dir
    )

    # The ground-storey texture is written last, so it is the trailing entry.
    has_ground_floor = bool(scene["facade"].get("ground_floor"))
    wall_textures = facade_textures[:-1] if has_ground_floor else facade_textures
    ground_texture = facade_textures[-1] if has_ground_floor else None

    aerial_material = make_textured_material("M_aerial", aerial_texture, roughness=0.9)
    facade_materials = [
        make_textured_material(
            "M_facade" if len(wall_textures) == 1 else f"M_facade_{i:02d}",
            colour,
            roughness=0.75,
            normal_path=normal,
        )
        for i, (colour, normal) in enumerate(wall_textures)
    ]
    ground_material = (
        make_textured_material(
            "M_facade_ground",
            ground_texture[0],
            roughness=0.6,
            normal_path=ground_texture[1],
        )
        if ground_texture
        else None
    )

    build_terrain(scene, work_dir, aerial_material)
    build_buildings(
        scene, work_dir, facade_materials, aerial_material, ground_material
    )

    if tree_texture is not None:
        tree_material = make_textured_material("M_tree", tree_texture, roughness=0.85)
        build_trees(scene, work_dir, tree_material)

    # Report the scene bounds so a coordinate or scale error shows up in the log
    # rather than only in Unity.
    xs, ys, zs = [], [], []
    for obj in bpy.context.scene.objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ __import__("mathutils").Vector(corner)
            xs.append(world.x)
            ys.append(world.y)
            zs.append(world.z)
    log(
        f"scene bounds X [{min(xs):.1f}, {max(xs):.1f}] "
        f"Y [{min(ys):.1f}, {max(ys):.1f}] Z [{min(zs):.1f}, {max(zs):.1f}]"
    )

    export_fbx(out_dir / scene["export"]["fbx_name"])

    summary = {
        "fbx": scene["export"]["fbx_name"],
        "bounds_local": {
            "x": [min(xs), max(xs)],
            "y": [min(ys), max(ys)],
            "z": [min(zs), max(zs)],
        },
        "objects": [obj.name for obj in bpy.context.scene.objects],
        "materials": [m.name for m in bpy.data.materials],
    }
    (work_dir / "blender_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
