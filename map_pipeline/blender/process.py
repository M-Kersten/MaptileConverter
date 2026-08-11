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

    # Mesh.shade_smooth() and shade_flat() only exist from Blender 4.1. Older
    # versions carry the flag per polygon, so fall back to setting it directly
    # rather than failing the whole run over shading.
    if hasattr(mesh, "shade_smooth") and hasattr(mesh, "shade_flat"):
        mesh.shade_smooth() if shade_smooth else mesh.shade_flat()
    else:
        mesh.polygons.foreach_set(
            "use_smooth", np.full(n_polygons, bool(shade_smooth), dtype=bool)
        )

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
# Terrain and water
# ---------------------------------------------------------------------------


def _load_surfaces(scene: dict, work_dir: Path):
    surface_file = scene.get("surfaces", {}).get("file")
    if not surface_file or not (work_dir / surface_file).is_file():
        return None
    return np.load(work_dir / surface_file)


def _sink_water_bed(
    scene: dict, work_dir: Path, heights: np.ndarray, n: int
) -> np.ndarray:
    """Push the terrain under each water body below its surface.

    The DTM has almost nothing to work from over water, so the gap filler
    interpolates inward from the banks and leaves a mound where a canal should
    be. Left alone that mound pokes straight through the water surface, so the
    grid inside a water outline is set to a flat bed below its level.
    """
    data = _load_surfaces(scene, work_dir)
    if data is None or "water_bed" not in data.files:
        return heights

    # One bed level per water body, worked out upstream: levels across an area
    # differ by metres, so a single shared bed would sit above the surface of
    # the lowest canal and poke straight through it.
    bed = data["water_bed"]
    if bed.size == 0 or bed.shape != heights.shape:
        return heights

    water = np.isfinite(bed)
    if not water.any():
        return heights

    out = heights.copy()
    out[water] = np.minimum(out[water], bed[water])
    log(
        f"terrain: sank {int(water.sum())} cells to water beds between "
        f"{np.nanmin(bed):.2f} and {np.nanmax(bed):.2f} m NAP"
    )
    return out


def build_water(scene: dict, work_dir: Path, material):
    """Flat surfaces over the BGT water outlines, one level per body.

    The outlines were triangulated upstream, where the triangulation library
    lives. This script only gets what Blender itself bundles.
    """
    data = _load_surfaces(scene, work_dir)
    if data is None or "water_tris" not in data.files:
        return None

    triangles = data["water_tris"]
    if len(triangles) == 0:
        return None

    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])

    local = triangles.copy()
    local[:, :, 0] -= origin_x
    local[:, :, 1] -= origin_y
    local[:, :, 2] -= z_offset

    corners = local.reshape(-1, 3)
    n_triangles = len(local)

    # UV in metres so a tiling water texture keeps a constant wave scale.
    uvs = np.column_stack([corners[:, 0] / 8.0, corners[:, 1] / 8.0])

    obj = build_mesh_object(
        "Water",
        corners,
        np.arange(n_triangles * 3),
        np.arange(0, n_triangles * 3, 3),
        np.full(n_triangles, 3),
        uvs,
        np.zeros(n_triangles),
        [material],
        shade_smooth=False,
    )
    bodies = len(data["water_levels"]) if "water_levels" in data.files else 0
    log(f"water: {bodies} bodies, {n_triangles} triangles")
    return obj


def build_roads(scene: dict, work_dir: Path, make_material):
    """The road surface as its own objects, one per class.

    Separate objects rather than one merged mesh, because that is what makes
    them separately addressable downstream: each becomes its own GameObject, so
    a carriageway can go on a drivable layer and a footpath on a walkable one
    without splitting anything by hand. They are cheap — six at most.

    Each carries its own material, textured with the aerial and projected the
    same way the terrain is, so the road looks exactly as it did when it was
    only pixels in the terrain's photograph. The point of the split is that the
    material is now yours to replace.
    """
    data = _load_surfaces(scene, work_dir)
    if data is None or "road_tris" not in data.files:
        return []

    triangles = data["road_tris"]
    classes = data["road_tri_class"]
    if len(triangles) == 0:
        return []

    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    names = scene.get("surfaces", {}).get("road_class_names", {})

    local = triangles.copy()
    local[:, :, 0] -= origin_x
    local[:, :, 1] -= origin_y
    local[:, :, 2] -= z_offset

    objects = []
    for code in sorted(set(int(c) for c in classes)):
        part = local[classes == code]
        if not len(part):
            continue
        corners = part.reshape(-1, 3)
        n_triangles = len(part)
        label = str(names.get(str(code), f"class_{code}"))
        # The class names already say "road_asphalt", and Roads_road_asphalt
        # reads badly in a hierarchy.
        if label.startswith("road_"):
            label = label[len("road_") :]

        objects.append(
            build_mesh_object(
                f"Roads_{label}",
                corners,
                np.arange(n_triangles * 3),
                np.arange(0, n_triangles * 3, 3),
                np.full(n_triangles, 3),
                # The same top-down projection the terrain uses, so the photo
                # lines up across the join.
                planar_uv(corners[:, :2], scene["aerial"]["bbox_local"]),
                np.zeros(n_triangles),
                [make_material(f"M_road_{label}")],
                shade_smooth=False,
            )
        )
        log(f"roads: {label}, {n_triangles} triangles")

    return objects


def build_terrain(scene: dict, work_dir: Path, material) -> object:
    """Grid mesh from the AHN heights, draped with the aerial photo."""
    data = np.load(work_dir / scene["terrain"]["file"])
    heights = data["heights"].astype(np.float64)
    xs = data["xs"]
    ys = data["ys"]

    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    n = heights.shape[0]

    heights = _sink_water_bed(scene, work_dir, heights, n)

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
    u: np.ndarray,
    triangles: np.ndarray,
    building_index: np.ndarray,
    base_z: np.ndarray,
    floor_height: np.ndarray,
) -> np.ndarray:
    """Pair a fitted U with a V that counts storeys from the wall's base.

    V uses an effective storey height of ``height / storeys`` so the top storey
    finishes flush with the eaves instead of being cut mid-window, and measures
    from ``base_z`` rather than from the ground, so a wall sitting on top of a
    split-off ground storey still starts its first row of windows on a line.
    """
    corners = triangles.reshape(-1, 3)
    base_per_corner = np.repeat(base_z[building_index], 3)
    floor_per_corner = np.repeat(floor_height[building_index], 3)
    v = (corners[:, 2] - base_per_corner) / floor_per_corner
    return np.column_stack([u, v])


def build_buildings(
    scene: dict,
    work_dir: Path,
    facade_materials: list,
    aerial_material,
    ground_materials: list | None = None,
):
    """One mesh holding every building, with facade and aerial material slots."""
    data = np.load(work_dir / scene["buildings"]["file"])
    wall_tris = data["wall_tris"]
    roof_tris = data["roof_tris"]
    wall_owner = data["wall_building"]
    roof_owner = data["roof_building"]
    has_ground = "ground_wall_tris" in data.files and bool(ground_materials)
    ground_tris = data["ground_wall_tris"] if has_ground else np.zeros((0, 3, 3))
    ground_owner = (
        data["ground_wall_building"] if has_ground else np.zeros((0,), dtype=np.int32)
    )
    ground_style = (
        data["ground_style"]
        if "ground_style" in data.files
        else np.zeros(len(data["ground_z_nap"]), dtype=np.int32)
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
    nominal_floor = float(facade_cfg["floor_height_m"])
    ground_floor_m = float(facade_cfg.get("ground_floor_height_m", 3.6))

    archetype = (
        data["archetype"]
        if "archetype" in data.files
        else np.zeros(len(ground_z_nap), dtype=np.int32)
    )
    # Matches src/buildings.py.
    ARCH_INDUSTRIAL, ARCH_MONUMENTAL = 4, 5
    special = (archetype == ARCH_MONUMENTAL) | (archetype == ARCH_INDUSTRIAL)

    # Which buildings actually had a ground storey taken off them. A church and
    # a warehouse keep their wall whole, so their base is still the ground.
    was_split = (
        has_ground & ~special & (wall_height_m > ground_floor_m * 1.2)
        if has_ground
        else np.zeros(len(ground_z_nap), dtype=bool)
    )
    cut_m = np.where(was_split, ground_floor_m, 0.0)
    wall_base_z = ground_z_local + cut_m
    # What is left for the storeys above the shopfront.
    upper_height = np.maximum(wall_height_m - cut_m, 0.1)

    # An effective storey height makes the top storey finish flush with the top
    # of the wall rather than being clipped mid-window, and measuring from the
    # first-floor line means the bottom row starts on one too.
    storeys = np.where(was_split, floors - 1.0, floors)
    storeys = np.maximum(storeys, 1.0)
    # A storey of one metre or four means the storey count is wrong, not the
    # height; fall back to dividing by the nominal storey and refit, so the top
    # of the wall still lands on a line.
    implausible = (upper_height / storeys < 1.8) | (upper_height / storeys > 6.0)
    storeys = np.where(
        implausible, np.maximum(1.0, np.round(upper_height / nominal_floor)), storeys
    )
    effective_floor = np.clip(upper_height / storeys, 1.5, 8.0)

    # A church or hall has no storeys to stack, so its wall is divided into a
    # few tall bays instead. Dividing its height by three is what turned the
    # Dom into thirty-six rows of domestic windows.
    if special.any():
        bay_height = np.where(
            archetype == ARCH_INDUSTRIAL,
            float(facade_cfg.get("industrial_bay_m", 6.0)),
            float(facade_cfg.get("monumental_bay_m", 9.0)),
        )
        # One tile per bay, with the bay count rounded so the top of the wall
        # lands on a tile edge.
        bays = np.maximum(1.0, np.round(wall_height_m / bay_height))
        effective_floor = np.where(special, wall_height_m / bays, effective_floor)

    materials = list(facade_materials) + [aerial_material]
    aerial_slot = len(facade_materials)
    ground_slot_base = len(materials)
    if has_ground:
        materials.extend(ground_materials)

    parts = [chunk for chunk in (walls, roofs, grounds) if len(chunk)]
    triangles = np.concatenate(parts, axis=0)
    corners = triangles.reshape(-1, 3)

    uv_chunks = []
    if n_walls:
        uv_chunks.append(
            wall_uvs(
                data["wall_u"],
                walls,
                wall_owner,
                wall_base_z,
                effective_floor,
            )
        )
    if n_roofs:
        uv_chunks.append(
            planar_uv(roofs.reshape(-1, 3)[:, :2], scene["aerial"]["bbox_local"])
        )
    if n_grounds:
        # The ground storey gets exactly one tile vertically, so the shopfront
        # spans it instead of repeating inside it.
        storey = np.full(len(ground_z_local), ground_floor_m)
        uv_chunks.append(
            wall_uvs(
                data["ground_wall_u"],
                grounds,
                ground_owner,
                ground_z_local,
                storey,
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
    if n_grounds:
        material_indices[n_walls + n_roofs :] = ground_slot_base + np.clip(
            ground_style[ground_owner], 0, max(0, len(ground_materials) - 1)
        )

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
# Street furniture
# ---------------------------------------------------------------------------


def _box(cx, cy, cz, sx, sy, sz, spin=0.0):
    """Axis-aligned box, optionally spun about Z, as (verts, faces)."""
    corners = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    ) * np.array([sx, sy, sz]) * 0.5

    if spin:
        cos_a, sin_a = np.cos(spin), np.sin(spin)
        rotated = corners.copy()
        rotated[:, 0] = corners[:, 0] * cos_a - corners[:, 1] * sin_a
        rotated[:, 1] = corners[:, 0] * sin_a + corners[:, 1] * cos_a
        corners = rotated

    corners += np.array([cx, cy, cz])
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return corners, faces


def build_furniture(scene: dict, work_dir: Path, material):
    """Lampposts, bollards, signs and benches as simple boxes.

    Every piece shares one material, so a couple of thousand objects cost one
    draw call rather than a couple of thousand.
    """
    furniture_file = scene.get("furniture", {}).get("file")
    if not furniture_file or not (work_dir / furniture_file).is_file():
        return None

    data = np.load(work_dir / furniture_file)
    xy = data["xy"]
    if len(xy) == 0:
        return None

    ground = data["ground_z_nap"]
    kinds = data["kind"]
    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    cfg = scene.get("furniture", {})

    lamp_h = float(cfg.get("lamp_height_m", 5.0))
    bollard_h = float(cfg.get("bollard_height_m", 0.9))
    bench_l = float(cfg.get("bench_length_m", 1.8))

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    offset = 0
    rng = np.random.default_rng(808)

    # A dark metal patch and a wood patch, addressed by UV like the tree atlas.
    metal_uv = np.array([0.25, 0.5])
    wood_uv = np.array([0.75, 0.5])

    for index in range(len(xy)):
        x = float(xy[index, 0]) - origin_x
        y = float(xy[index, 1]) - origin_y
        base = float(ground[index]) - z_offset
        kind = int(kinds[index])
        spin = float(rng.random()) * np.pi

        pieces = []
        if kind == 0:  # lamppost: column plus a short arm
            pieces.append((_box(x, y, base + lamp_h / 2, 0.14, 0.14, lamp_h, spin), metal_uv))
            pieces.append(
                (_box(x, y, base + lamp_h + 0.08, 0.7, 0.24, 0.16, spin), metal_uv)
            )
        elif kind == 1:  # bollard
            pieces.append(
                (_box(x, y, base + bollard_h / 2, 0.16, 0.16, bollard_h, spin), metal_uv)
            )
        elif kind == 2:  # bench: seat on two legs
            pieces.append((_box(x, y, base + 0.45, bench_l, 0.5, 0.08, spin), wood_uv))
            pieces.append((_box(x, y, base + 0.22, bench_l * 0.8, 0.1, 0.44, spin), metal_uv))
        else:  # sign post
            pieces.append((_box(x, y, base + 1.1, 0.09, 0.09, 2.2, spin), metal_uv))
            pieces.append((_box(x, y, base + 2.1, 0.5, 0.05, 0.4, spin), metal_uv))

        for (piece_verts, piece_faces), uv in pieces:
            vertices.append(piece_verts)
            faces.append(piece_faces + offset)
            uvs.append(np.tile(uv, (len(piece_faces) * 3, 1)))
            offset += len(piece_verts)

    all_vertices = np.vstack(vertices)
    all_faces = np.vstack(faces)
    all_uvs = np.vstack(uvs)
    n_triangles = len(all_faces)

    obj = build_mesh_object(
        "StreetFurniture",
        all_vertices,
        all_faces.reshape(-1),
        np.arange(0, n_triangles * 3, 3),
        np.full(n_triangles, 3),
        all_uvs,
        np.zeros(n_triangles),
        [material],
        shade_smooth=False,
    )
    log(f"street furniture: {len(xy)} objects, {n_triangles} triangles")
    return obj


def _hull(cx, cy, cz, length, beam, depth, freeboard, spin):
    """A boat hull: a box with the bow drawn to a point, as (verts, faces).

    Six vertices a side rather than four. A rectangular boat reads as a crate,
    and the taper is the whole difference between the two at the distance
    anyone will actually look at these from.
    """
    half_l, half_b = length * 0.5, beam * 0.5
    # Bow at +X, stern at -X. The waterline sits at z = 0.
    outline = np.array(
        [
            [half_l, 0.0],            # stem
            [half_l * 0.55, half_b],  # shoulder
            [-half_l, half_b * 0.85], # transom corner
            [-half_l, -half_b * 0.85],
            [half_l * 0.55, -half_b],
        ]
    )
    n = len(outline)
    lower = np.column_stack([outline * 0.72, np.full(n, -depth)])
    upper = np.column_stack([outline, np.full(n, freeboard)])
    corners = np.vstack([lower, upper])

    cos_a, sin_a = np.cos(spin), np.sin(spin)
    spun = corners.copy()
    spun[:, 0] = corners[:, 0] * cos_a - corners[:, 1] * sin_a
    spun[:, 1] = corners[:, 0] * sin_a + corners[:, 1] * cos_a
    spun += np.array([cx, cy, cz])

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    # Deck and bottom, fanned from the first vertex.
    for i in range(1, n - 1):
        faces.append([n, n + i, n + i + 1])
        faces.append([0, i + 1, i])
    return spun, np.array(faces, dtype=np.int64)


def build_vehicles(scene: dict, work_dir: Path, material):
    """Parked cars, moored boats and the posts they are tied to.

    All three share one material and one mesh, so a few hundred of them cost
    one draw call. The car atlas carries several body colours, because a street
    where every car is the same colour looks worse than no cars at all.
    """
    vehicle_file = scene.get("vehicles", {}).get("file")
    if not vehicle_file or not (work_dir / vehicle_file).is_file():
        return None

    data = np.load(work_dir / vehicle_file)
    xy = data["xy"]
    if len(xy) == 0:
        return None

    z_nap = data["z_nap"]
    heading = data["heading"]
    lengths = data["length"]
    widths = data["width"]
    kinds = data["kind"]
    colours = data["colour"]

    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    cfg = scene.get("vehicles", {})
    car_height = float(cfg.get("car_height_m", 1.5))
    n_colours = max(1, int(cfg.get("car_colours", 6)))

    # The atlas is one row of car colours over a row holding hull and post.
    # Sampling the middle of each patch keeps bilinear filtering off the seams.
    def car_uv(slot: int) -> np.ndarray:
        return np.array([(slot + 0.5) / n_colours, 0.75])

    hull_uv = np.array([0.25, 0.25])
    post_uv = np.array([0.75, 0.25])

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    offset = 0

    for index in range(len(xy)):
        x = float(xy[index, 0]) - origin_x
        y = float(xy[index, 1]) - origin_y
        base = float(z_nap[index]) - z_offset
        kind = int(kinds[index])
        spin = float(heading[index])
        length = float(lengths[index])
        width = float(widths[index])

        pieces = []
        if kind == 0:  # car: body with a cabin set back on it
            uv = car_uv(int(colours[index]) % n_colours)
            body_h = car_height * 0.52
            pieces.append(
                (_box(x, y, base + body_h / 2, length, width, body_h, spin), uv)
            )
            # The cabin sits behind centre, which is what makes a box read as
            # having a bonnet and therefore a front.
            cabin_h = car_height - body_h
            back_x = x - np.cos(spin) * length * 0.10
            back_y = y - np.sin(spin) * length * 0.10
            pieces.append(
                (
                    _box(
                        back_x, back_y, base + body_h + cabin_h / 2,
                        length * 0.52, width * 0.88, cabin_h, spin,
                    ),
                    uv,
                )
            )
        elif kind == 1:  # boat: hull at the waterline, with a low cabin
            draft = min(0.55, width * 0.35)
            freeboard = max(0.35, width * 0.30)
            pieces.append(
                (_hull(x, y, base, length, width, draft, freeboard, spin), hull_uv)
            )
            if length > 6.0:
                back_x = x - np.cos(spin) * length * 0.15
                back_y = y - np.sin(spin) * length * 0.15
                pieces.append(
                    (
                        _box(
                            back_x, back_y, base + freeboard + 0.35,
                            length * 0.34, width * 0.62, 0.7, spin,
                        ),
                        hull_uv,
                    )
                )
        else:  # mooring post
            pieces.append((_box(x, y, base + 0.45, 0.22, 0.22, 1.4, spin), post_uv))

        for (piece_verts, piece_faces), uv in pieces:
            vertices.append(piece_verts)
            faces.append(piece_faces + offset)
            uvs.append(np.tile(uv, (len(piece_faces) * 3, 1)))
            offset += len(piece_verts)

    if not vertices:
        return None

    all_vertices = np.vstack(vertices)
    all_faces = np.vstack(faces)
    all_uvs = np.vstack(uvs)
    n_triangles = len(all_faces)

    obj = build_mesh_object(
        "Vehicles",
        all_vertices,
        all_faces.reshape(-1),
        np.arange(0, n_triangles * 3, 3),
        np.full(n_triangles, 3),
        all_uvs,
        np.zeros(n_triangles),
        [material],
        shade_smooth=False,
    )
    counts = np.bincount(kinds, minlength=3)
    log(
        f"vehicles: {counts[0]} cars, {counts[1]} boats, {counts[2]} mooring "
        f"posts, {n_triangles} triangles"
    )
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
) -> tuple[Path, list[tuple[Path, Path | None]], dict[str, Path]]:
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

    extras: dict[str, Path] = {}
    for key, section in (
        ("tree", "trees"),
        ("water", "surfaces"),
        ("furniture", "furniture"),
        ("vehicle", "vehicles"),
    ):
        name = scene.get(section, {}).get("texture")
        if name and (work_dir / name).is_file():
            extras[key] = copy(name)
    return aerial_dst, facades, extras


def main() -> int:
    work_dir, out_dir = parse_args(list(sys.argv))
    work_dir = work_dir.resolve()
    out_dir = out_dir.resolve()

    log(f"Blender {bpy.app.version_string} on {sys.platform}")
    if bpy.app.version < (3, 3):
        log(
            f"WARNING: Blender {bpy.app.version_string} is older than 3.3 and is "
            f"not supported; the mesh and export APIs differ"
        )

    scene_path = work_dir / "scene.json"
    if not scene_path.is_file():
        raise SystemExit(f"missing {scene_path}; run the Python stages first")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))

    log(f"building scene {scene['name']!r}")
    reset_scene()

    aerial_texture, facade_textures, extra_textures = copy_textures(
        scene, work_dir, out_dir
    )

    # Ground-storey textures are written last, so they are the trailing entries.
    n_ground = int(scene["facade"].get("ground_variants", 0))
    wall_textures = facade_textures[: len(facade_textures) - n_ground] if n_ground else facade_textures
    ground_textures = facade_textures[len(facade_textures) - n_ground :] if n_ground else []

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
    ground_materials = [
        make_textured_material(
            f"M_facade_{colour.stem.replace('facade_', '')}",
            colour,
            roughness=0.6,
            normal_path=normal,
        )
        for colour, normal in ground_textures
    ]

    build_terrain(scene, work_dir, aerial_material)
    # Roads sit just above the terrain, each class its own object, so they can
    # be given their own material and their own layer downstream.
    build_roads(
        scene,
        work_dir,
        lambda name: make_textured_material(name, aerial_texture, roughness=0.85),
    )
    build_buildings(
        scene, work_dir, facade_materials, aerial_material, ground_materials
    )

    if "water" in extra_textures:
        # Water is the one smooth thing in the scene, so it gets a low
        # roughness while everything else stays matte.
        build_water(
            scene,
            work_dir,
            make_textured_material("M_water", extra_textures["water"], roughness=0.12),
        )

    if "tree" in extra_textures:
        build_trees(
            scene,
            work_dir,
            make_textured_material("M_tree", extra_textures["tree"], roughness=0.85),
        )

    if "furniture" in extra_textures:
        build_furniture(
            scene,
            work_dir,
            make_textured_material(
                "M_furniture", extra_textures["furniture"], roughness=0.55
            ),
        )

    if "vehicle" in extra_textures:
        # Car paint is the one glossy thing out here.
        build_vehicles(
            scene,
            work_dir,
            make_textured_material(
                "M_vehicle", extra_textures["vehicle"], roughness=0.35
            ),
        )

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


ERROR_FILE = "blender_error.txt"


def _run() -> int:
    """Run main and record any failure where the orchestrator can find it.

    Blender in background mode exits 0 even when the script it was given raises,
    so a traceback here would otherwise scroll past and the run would look like
    it succeeded until the FBX turned out to be missing. The traceback is
    written next to the intermediates so the orchestrator can quote it.
    """
    import traceback

    try:
        return main()
    except SystemExit:
        raise
    except BaseException:
        detail = traceback.format_exc()
        print("[blender] FAILED\n" + detail, file=sys.stderr, flush=True)
        try:
            work_dir, _ = parse_args(list(sys.argv))
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / ERROR_FILE).write_text(
                f"Blender {bpy.app.version_string} on {sys.platform}\n\n{detail}",
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001 - the original error is what matters
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
