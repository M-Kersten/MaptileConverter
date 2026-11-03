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


def make_cutout_material(name: str, image_path: Path, roughness: float):
    """Textured material whose transparent pixels are actually cut away.

    A fence is mostly holes. Wiring the image's alpha into the Principled
    BSDF's alpha socket and setting the blend method to a clip is what makes
    them holes rather than grey; without it a railing is a hoarding, and FBX
    carries the texture reference, so the receiving engine can read the same
    alpha out of the same PNG.
    """
    material = make_textured_material(name, image_path, roughness)
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    texture = next(
        (n for n in nodes if n.type == "TEX_IMAGE" and n.image
         and n.image.name == image_path.name),
        None,
    )
    if bsdf is not None and texture is not None and "Alpha" in bsdf.inputs:
        links.new(texture.outputs["Alpha"], bsdf.inputs["Alpha"])

    # The attribute moved about between versions and is gone in some; none of
    # them are worth failing the whole export over.
    for attribute, value in (
        ("blend_method", "CLIP"),
        ("shadow_method", "CLIP"),
        ("alpha_threshold", 0.5),
        ("use_backface_culling", False),
    ):
        try:
            setattr(material, attribute, value)
        except (AttributeError, TypeError):
            pass
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
    extra_uvs: dict | None = None,
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
    for uv_name, values in (extra_uvs or {}).items():
        extra = mesh.uv_layers.new(name=uv_name)
        extra.data.foreach_set("uv", np.asarray(values, dtype=np.float32).ravel())
    # The aerial projection is the one that draws; the others are for whoever
    # swaps the material.
    mesh.uv_layers.active = uv_layer
    if hasattr(uv_layer, "active_render"):
        uv_layer.active_render = True

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


def _triangulate_on_export(obj, quads: str = "FIXED") -> None:
    """Triangulate for the FBX, and only for the FBX.

    The .blend keeps its quads and n-gons, which is what an editor wants; the
    modifier is off in the viewport and on for render, and the export applies
    render modifiers. That decides the split here, where it can be matched,
    rather than leaving it to Unity's importer: the terrain's squares are cut
    first corner to third, south-west to north-east, which is the surface the
    pipeline placed every tree and lamp post on. N-gons -- junctions and the
    edges of car parks, some of them concave -- are cut by Blender's own
    polygon fill rather than by whatever an importer does with them.
    """
    if not hasattr(obj, "modifiers") or getattr(obj, "type", "MESH") != "MESH":
        return
    modifier = obj.modifiers.new("Triangulate for export", "TRIANGULATE")
    modifier.quad_method = quads
    modifier.ngon_method = "BEAUTY"
    modifier.show_viewport = False
    modifier.show_render = True


class _Layer:
    """Gather what a builder made into a collection, under one parent.

    The collection is how the .blend is organised; the parent empty is how the
    FBX is, because FBX has no collections and Unity builds its hierarchy from
    parents. Everything a builder adds without a parent of its own is put
    under the layer's empty.
    """

    def __init__(self, collection: str, parent: str | None = None):
        self.collection = collection
        self.parent = parent

    def __enter__(self):
        self.before = {obj.name for obj in bpy.data.objects}
        return self

    def __exit__(self, *exc_info):
        if exc_info[0] is not None:
            return False
        new = [obj for obj in bpy.data.objects if obj.name not in self.before]
        if not new:
            return False
        scene = bpy.context.scene
        collection = bpy.data.collections.get(self.collection)
        if collection is None:
            collection = bpy.data.collections.new(self.collection)
            scene.collection.children.link(collection)
        empty = None
        if self.parent:
            empty = bpy.data.objects.get(self.parent)
            if empty is None:
                empty = bpy.data.objects.new(self.parent, None)
                empty.empty_display_size = 5.0
                collection.objects.link(empty)
        for obj in new:
            for owner in list(obj.users_collection):
                owner.objects.unlink(obj)
            collection.objects.link(obj)
            if empty is not None and obj.parent is None:
                obj.parent = empty
        return False


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


def _water_bed_grid(scene: dict, work_dir: Path, shape=None) -> np.ndarray | None:
    """The per-body bed level on the terrain grid, or None if there is none.

    One level per water body, worked out upstream: levels across an area differ
    by metres, so a single shared bed would sit above the surface of the lowest
    canal and poke straight through it.
    """
    data = _load_surfaces(scene, work_dir)
    if data is None or "water_bed" not in data.files:
        return None
    bed = data["water_bed"]
    if bed.size == 0 or not np.isfinite(bed).any():
        return None
    if shape is not None and bed.shape != shape:
        return None
    return bed


def _sink_water_bed(
    scene: dict, work_dir: Path, heights: np.ndarray, n: int, quiet: bool = False
) -> np.ndarray:
    """Push the terrain under each water body below its surface.

    The DTM has almost nothing to work from over water, so the gap filler
    interpolates inward from the banks and leaves a mound where a canal should
    be. Left alone that mound pokes straight through the water surface, so the
    grid inside a water outline is set to a flat bed below its level.
    """
    bed = _water_bed_grid(scene, work_dir, heights.shape)
    if bed is None:
        return heights

    water = np.isfinite(bed)
    out = heights.copy()
    out[water] = np.minimum(out[water], bed[water])
    if not quiet:
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


def make_plain_material(name: str, colour, roughness: float):
    """An untextured Principled BSDF: a kerb is grey concrete, not photo."""
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
    material.diffuse_color = (*colour, 1.0)
    return material


def build_roads(scene: dict, work_dir: Path, make_material):
    """The road network: one object per street, junction and stretch of paving.

    Every street is quads cut straight across it from kerb to kerb, so a
    cross-section is an edge loop and so is every kerb; nothing sits in the
    middle of a surface. Junctions, and road no centreline reaches, are one
    face per piece of surface, bounded by its own outline. Footways stand on a
    kerb, and the edge of the whole network has a short skirt down into the
    ground. See src/roads.py.

    Each surface class is its own material slot, textured with the aerial and
    projected the way the terrain is, so the road looks as it did when it was
    pixels in the photo -- and the material is yours to replace. The second UV
    map, RoadUV, runs in metres across and along the street, for a tiling
    asphalt or brick texture and lane markings.
    """
    data = _load_surfaces(scene, work_dir)
    if data is None or "road_vertices" not in data.files:
        return []
    vertices_rd = data["road_vertices"].astype(np.float64)
    sizes = data["road_face_sizes"].astype(np.int64)
    if not len(sizes):
        return []
    loops = data["road_face_loops"].astype(np.int64)
    face_class = data["road_face_class"].astype(np.int64)
    face_object = data["road_face_object"].astype(np.int64)
    loop_uv = data["road_loop_uv"].astype(np.float64)
    object_names = [str(n) for n in data["road_object_names"]]
    object_kinds = [str(n) for n in data["road_object_kinds"]]
    face_kind = data["road_face_kind"] if "road_face_kind" in data.files else None

    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    names = scene.get("surfaces", {}).get("road_class_names", {})
    local = vertices_rd - np.array([origin_x, origin_y, z_offset])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    face_of_loop = np.repeat(np.arange(len(sizes)), sizes)

    materials: dict[int, object] = {}

    def material_for(code: int):
        if code not in materials:
            label = str(names.get(str(code), f"class_{code}"))
            if label == "kerb":
                materials[code] = make_plain_material("M_road_kerb", (0.42, 0.42, 0.40), 0.8)
            else:
                materials[code] = make_material(f"M_road_{label.replace('road_', '')}")
        return materials[code]

    objects = []
    for index, name in enumerate(object_names):
        faces = np.flatnonzero(face_object == index)
        if not len(faces):
            continue
        wanted = np.isin(face_of_loop, faces)
        own_loops = loops[wanted]
        used, remap = np.unique(own_loops, return_inverse=True)
        points = local[used]
        # The object's origin sits on the ground under the middle of it, so it
        # moves, rotates and snaps like the thing it is.
        origin = np.array(
            [0.5 * (points[:, 0].min() + points[:, 0].max()),
             0.5 * (points[:, 1].min() + points[:, 1].max()),
             points[:, 2].min()]
        )
        own_sizes = sizes[faces]
        own_starts = np.concatenate([[0], np.cumsum(own_sizes)[:-1]])
        codes = face_class[faces]
        slots = sorted(set(int(c) for c in codes))
        slot_of = {code: i for i, code in enumerate(slots)}
        obj = build_mesh_object(
            name,
            points - origin,
            remap,
            own_starts,
            own_sizes,
            # The top-down projection the terrain uses, from where the vertex
            # is in the model rather than in the object, so the photo lines up
            # across every join.
            planar_uv(points[remap][:, :2], scene["aerial"]["bbox_local"]),
            np.array([slot_of[int(c)] for c in codes]),
            [material_for(code) for code in slots],
            shade_smooth=False,
            extra_uvs={"RoadUV": loop_uv[wanted]},
        )
        obj.location = tuple(origin)
        obj["kind"] = object_kinds[index] if index < len(object_kinds) else "street"
        if face_kind is not None:
            # 0 surface, 1 kerb, 2 skirt: what the Maptile tools fit the
            # terrain under, and what they leave out.
            part = obj.data.attributes.new("road_part", "INT", "FACE")
            part.data.foreach_set("value", face_kind[faces].astype(np.int32))
        _triangulate_on_export(obj, quads="FIXED")
        objects.append(obj)

    quads = int((sizes == 4).sum())
    log(
        f"roads: {len(objects)} objects ({sum(1 for o in objects if o['kind'] == 'street')} "
        f"streets, {sum(1 for o in objects if o['kind'] == 'junction')} junctions), "
        f"{len(sizes)} faces of which {quads} quads, {len(vertices_rd)} vertices"
    )
    return objects


def _weld_corners(corners, tolerance=1e-4):
    """Merge corners that are the same point, returning points plus indices."""
    quantised = np.round(corners / tolerance).astype(np.int64)
    _, first, inverse = np.unique(
        quantised, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(np.argsort(first))
    return corners[np.sort(first)], order[inverse.ravel()].astype(np.int64)


def _tag_surface_regions(
    obj, scene, work_dir, points, loop_vertices, loop_starts, loop_totals
) -> int:
    """Label every face with what it is, and group the vertices to match.

    This is what turns "flatten the ground under this building" from a lasso
    and a prayer into select-then-flatten. The class of every square metre is
    already worked out for the land cover map; it was simply never attached to
    the terrain, so the model arrived knowing which faces were road and telling
    nobody.

    Two forms, because Blender uses them for different things. A face attribute
    is exact and is what the spreadsheet and Select Similar read. Vertex groups
    are one click in the Object Data panel and, more to the point, they drive
    proportional editing -- which is what a designer levelling a plot actually
    reaches for.
    """
    surfaces = scene.get("surfaces") or {}
    if not surfaces.get("file"):
        return 0
    try:
        data = np.load(work_dir / surfaces["file"])
    except OSError:
        return 0
    if "class_grid" not in data:
        return 0
    grid = data["class_grid"]
    if grid.size == 0:
        return 0

    names = {int(k): v for k, v in (surfaces.get("class_names") or {}).items()}
    xmin, ymin, xmax, ymax = scene["bbox_local"]
    rows, cols = grid.shape

    # One class per face, read at the face centroid.
    corner_xy = points[loop_vertices][:, :2]
    centroids = np.add.reduceat(corner_xy, loop_starts, axis=0) / loop_totals[:, None]
    col = np.clip(
        ((centroids[:, 0] - xmin) / max(xmax - xmin, 1e-9) * cols).astype(int),
        0, cols - 1,
    )
    row = np.clip(
        ((centroids[:, 1] - ymin) / max(ymax - ymin, 1e-9) * rows).astype(int),
        0, rows - 1,
    )
    face_class = grid[row, col].astype(np.int32)

    mesh = obj.data
    attribute = mesh.attributes.new("surface_class", "INT", "FACE")
    attribute.data.foreach_set("value", face_class)

    # A vertex belongs to every class touching it, which is what you want at a
    # kerb: grabbing the road takes the edge of the pavement with it.
    loop_class = np.repeat(face_class, loop_totals)
    made = 0
    for value in np.unique(face_class):
        name = names.get(int(value)) or f"class_{int(value)}"
        members = np.unique(loop_vertices[loop_class == value])
        if not len(members):
            continue
        group = obj.vertex_groups.new(name=f"ground_{name}")
        group.add(members.tolist(), 1.0, "REPLACE")
        made += 1
    return made


def _terrain_summary() -> dict:
    """Vertex and face counts of the terrain as built, for the checks to read."""
    obj = bpy.context.scene.objects.get("Terrain")
    if obj is None or obj.type != "MESH":
        return {}
    mesh = obj.data
    quads = sum(1 for polygon in mesh.polygons if len(polygon.vertices) == 4)
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.polygons),
        "quads": quads,
        # A grid is quads, every adaptive mesh is triangles. One number tells
        # you which of the two the model actually contains.
        "triangles": len(mesh.polygons) - quads,
    }


# What the terrain mesh's bits mean, as vertex groups: select the group and
# the waterline, the road edge or a building outline is selected. Matches
# src/terrain_mesh.py, which this script may not import.
OUTLINE_GROUPS = (
    (1, "outline_waterline"),
    (2, "outline_road"),
    (4, "outline_building"),
    (8, "outline_row"),
    (16, "outline_bed"),
)
# Outlines proper, marked as seams so they show in edit mode. The rows and the
# bed are loops beside them, not outlines.
OUTLINE_SEAMS = 1 | 2 | 4


def build_terrain(scene: dict, work_dir: Path, material) -> object:
    """The terrain, draped with the aerial.

    The mesh whose edges follow the outlines on the ground when the pipeline
    made one (`terrain_mesh.npz`), else the regular grid. Every face is tagged
    with the surface it lies under, so the ground below a road or a lawn is
    one click away; see `_tag_surface_regions`.
    """
    data = np.load(work_dir / scene["terrain"]["file"])
    if "face_loops" in data.files:
        return _build_terrain_mesh(scene, work_dir, material, data)
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
    groups = _tag_surface_regions(
        obj, scene, work_dir, vertices, loop_vertex_indices, loop_starts, loop_totals
    )
    # The ground as surveyed, before anything was pushed down under a road, so
    # the Maptile tools can put it back. Water beds stay sunk: that is not a
    # fit to the roads but a fix to the survey.
    if "measured" in data.files:
        measured = _sink_water_bed(
            scene, work_dir, data["measured"].astype(np.float64), n, quiet=True
        )
        attribute = obj.data.attributes.new("measured_z", "FLOAT", "POINT")
        attribute.data.foreach_set("value", (measured - z_offset).ravel().astype(np.float32))
    _triangulate_on_export(obj, quads="FIXED")
    spacing = float(xs[1] - xs[0]) if len(xs) > 1 else 0.0
    log(
        f"terrain: {n}x{n} vertices at {spacing:.2f} m, {n_quads} quads, "
        f"{groups} surface regions tagged"
    )
    return obj


def _build_terrain_mesh(scene: dict, work_dir: Path, material, data) -> object:
    """The terrain mesh: quads and triangles, its outlines tagged and grouped."""
    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    vertices = data["vertices"].astype(np.float64) - [origin_x, origin_y, z_offset]
    sizes = data["face_sizes"].astype(np.int64)
    loops = data["face_loops"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    uvs = planar_uv(vertices[loops][:, :2], scene["aerial"]["bbox_local"])
    obj = build_mesh_object(
        "Terrain",
        vertices,
        loops,
        starts,
        sizes,
        uvs,
        np.zeros(len(sizes)),
        [material],
        shade_smooth=True,
    )
    mesh = obj.data
    groups = _tag_surface_regions(obj, scene, work_dir, vertices, loops, starts, sizes)
    # The ground as surveyed, before anything was pushed down under a road, so
    # the Maptile tools can put it back. Water beds are part of it: they are
    # not a fit to the roads but a fix to the survey.
    measured = mesh.attributes.new("measured_z", "FLOAT", "POINT")
    measured.data.foreach_set(
        "value", (data["measured_z"].astype(np.float64) - z_offset).astype(np.float32)
    )
    if len(mesh.polygons) == len(sizes):
        for name, key in (("ground_kind", "face_kind"), ("ground_part", "face_part")):
            if key in data.files:
                attribute = mesh.attributes.new(name, "INT", "FACE")
                attribute.data.foreach_set("value", data[key].astype(np.int32))
    marked = _mark_outlines(obj, data)
    _triangulate_on_export(obj, quads="FIXED")
    quads = int((sizes == 4).sum())
    log(
        f"terrain: {len(vertices)} vertices, {len(sizes)} faces of which "
        f"{quads} quads; {marked} outline edges marked; {groups} surface regions tagged"
    )
    return obj


def _mark_outlines(obj, data) -> int:
    """Seams, an `outline` edge attribute and a vertex group per outline kind."""
    if "edges" not in data.files or not len(data["edges"]):
        return 0
    mesh = obj.data
    wanted = data["edges"].astype(np.int64)
    bits = data["edge_on"].astype(np.int64)
    count = len(mesh.edges)
    have = np.zeros(count * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", have)
    have = np.sort(have.reshape(-1, 2).astype(np.int64), axis=1)
    n = int(len(mesh.vertices)) + 1
    have_key = have[:, 0] * n + have[:, 1]
    order = np.argsort(have_key)
    wanted = np.sort(wanted, axis=1)
    wanted_key = wanted[:, 0] * n + wanted[:, 1]
    at = np.clip(np.searchsorted(have_key[order], wanted_key), 0, count - 1)
    found = have_key[order][at] == wanted_key
    index = order[at[found]]
    edge_bits = np.zeros(count, dtype=np.int32)
    edge_bits[index] = bits[found]
    attribute = mesh.attributes.new("outline", "INT", "EDGE")
    attribute.data.foreach_set("value", edge_bits)
    seams = (edge_bits & OUTLINE_SEAMS) > 0
    try:
        mesh.edges.foreach_set("use_seam", seams)
    except Exception:  # noqa: BLE001 - a newer Blender keeps seams elsewhere
        pass
    if "vertex_on" in data.files:
        vertex_bits = data["vertex_on"].astype(np.int64)
        for bit, name in OUTLINE_GROUPS:
            members = np.flatnonzero(vertex_bits & bit)
            if len(members):
                group = obj.vertex_groups.new(name=name)
                group.add(members.tolist(), 1.0, "REPLACE")
    return int((edge_bits > 0).sum())


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
    """One object per building, with facade and aerial material slots."""
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

    owner = np.concatenate(
        [chunk for chunk, count in ((wall_owner, n_walls), (roof_owner, n_roofs),
                                    (ground_owner, n_grounds)) if count]
    ).astype(np.int64)
    ids = (
        [str(i) for i in data["building_ids"]]
        if "building_ids" in data.files
        else [f"{i:05d}" for i in range(len(ground_z_nap))]
    )
    years = data["build_year"] if "build_year" in data.files else None
    heights = data["height_m"] if "height_m" in data.files else None

    # One object per building, so a building is something you click on, move,
    # delete or swap, and its own GameObject in Unity. The triangles of each
    # are pulled out in one sort rather than one search per building.
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(len(ground_z_nap) + 1))
    corner_uvs = uvs.reshape(-1, 3, 2)
    objects = []
    for index in range(len(ground_z_nap)):
        mine = order[bounds[index] : bounds[index + 1]]
        if not len(mine):
            continue
        tris = triangles[mine]
        points, loops = _weld_corners(tris.reshape(-1, 3))
        slots = sorted(set(int(m) for m in material_indices[mine]))
        slot_of = {m: i for i, m in enumerate(slots)}
        # The origin on the ground under the middle of the footprint, so the
        # building stands on its origin: snapping it to the terrain, or
        # swapping it for a modelled one, then just works.
        origin = np.array([
            0.5 * (points[:, 0].min() + points[:, 0].max()),
            0.5 * (points[:, 1].min() + points[:, 1].max()),
            float(ground_z_local[index]),
        ])
        identifier = ids[index]
        obj = build_mesh_object(
            "Pand_" + identifier.rsplit(".", 1)[-1],
            points - origin,
            loops,
            np.arange(0, len(mine) * 3, 3),
            np.full(len(mine), 3),
            corner_uvs[mine].reshape(-1, 2),
            np.array([slot_of[int(m)] for m in material_indices[mine]]),
            [materials[m] for m in slots],
            shade_smooth=False,
        )
        obj.location = tuple(origin)
        # Carried into the FBX as user properties, where a Unity asset
        # postprocessor can read them.
        obj["bag_id"] = identifier
        if years is not None:
            obj["build_year"] = int(years[index])
        if heights is not None:
            obj["height_m"] = round(float(heights[index]), 2)
        objects.append(obj)

    log(
        f"buildings: {len(objects)} objects, {n_walls} wall triangles, "
        f"{n_roofs} roof triangles, {n_grounds} ground-storey triangles"
    )
    return objects


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


def _tree_dice(x, y):
    """Four stable pseudo-random numbers in [0, 1) from a tree's position.

    From the position rather than from a running generator, so a tree looks the
    same whatever else is in the model: turning detection off, or rebuilding
    one group and not the other, must not reshape the trees that were already
    there. A sequential generator cannot promise that, because the order it is
    drawn in changes with the membership.
    """
    seed = np.float64(x) * 73856093.0 + np.float64(y) * 19349663.0
    out = []
    for salt in (0.0, 1.7, 3.1, 5.9):
        value = np.sin(seed * 0.0001 + salt) * 43758.5453
        out.append(float(value - np.floor(value)))
    return out



def _lobed_crown(shape, faces, lobes, crown_r, crown_h, squash, spin):
    """A crown built from overlapping lobes, as (vertices, faces).

    One deformed ellipsoid is still an ellipsoid: push it about as much as you
    like and the outline stays convex, which is what makes a lollipop a
    lollipop. A real broadleaf crown is a cluster of masses with sky showing
    between them, and the only way to an outline like that is more than one
    lump.

    Lobes are placed around the axis and at different heights, overlapping
    enough that the joins read as foliage rather than as separate balls, and
    each is deformed on its own so no two match.
    """
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0

    for lobe in range(lobes):
        turn = spin + lobe * 2.399963  # the golden angle, so nothing lines up
        # A single lobe sits on the axis; several spread around it, and the
        # spread stays well inside the crown radius or the tree reads as a
        # bunch of balloons rather than one canopy.
        half_h = crown_h * 0.5
        out = 0.0 if lobes == 1 else crown_r * 0.34
        rise = 0.0 if lobes == 1 else (lobe / (lobes - 1) - 0.5) * half_h * 0.50
        centre = np.array([out * np.cos(turn), out * np.sin(turn), rise])

        # Each lobe is a lump rather than a copy of the whole crown. Scaling a
        # lobe by the full crown height is what turns three of them into three
        # separate elongated diamonds floating above a stick: at 0.62 they are
        # wider than the gaps between their centres, so the union closes up.
        size = 1.0 if lobes == 1 else 0.62
        radii = np.array([crown_r * size, crown_r * squash * size,
                          half_h * size])

        wobble = (
            1.0
            + 0.28 * np.sin(shape[:, 0] * 2.1 + turn)
            * np.cos(shape[:, 1] * 1.8 - turn * 0.7)
            + 0.16 * np.sin(shape[:, 2] * 3.3 + shape[:, 0] * 1.4 + turn * 1.9)
        )
        all_verts.append(shape * wobble[:, None] * radii + centre)
        all_faces.append(faces + offset)
        offset += len(shape)

    crown = np.vstack(all_verts)

    # Fit the union to the crown it was asked for. Lobes overlap, so their
    # union is smaller than any one of them implies -- three at 0.62 with a
    # spread of 0.25 fill about 0.87 of the height, which made every tree a
    # sixth shorter than the canopy model measured it to be. Scaling the
    # finished cluster is robust to whatever the lobe numbers are, which
    # picking them to add up to one is not.
    low, high = crown.min(axis=0), crown.max(axis=0)
    span = high - low
    want = np.array([crown_r * 2.0, crown_r * squash * 2.0, crown_h])
    # About the middle of the box, not about the mean of the vertices: a lobed
    # crown is not symmetric, so its mean sits off-centre and scaling about it
    # leaves one side longer than asked for. That is how a 28 m tree came out
    # 28.7 m tall.
    crown = (crown - (low + high) * 0.5) * (want / np.maximum(span, 1e-6))

    return crown, np.vstack(all_faces)


# Tree shapes are built once at unit size and shared: every tree is an
# instance of one, scaled to its own crown and height and turned to its own
# angle. Two crown types by size, four trunk proportions, four shapes each.
TREE_TRUNK_FRACTIONS = (0.25, 0.35, 0.45, 0.55)
TREE_SHAPES_PER_KIND = 4


def _tree_instance(x: float, y: float, height: float, trunk_h: float) -> dict:
    """Which shape a tree is drawn with, and how it is turned and squashed.

    From the tree's position, so a tree looks the same whatever else is in the
    model. The drawn tree and its collision proxy both read this, which is
    what keeps the proxy round the tree rather than near it.
    """
    spin_d, squash_d, _spare, shape_d = _tree_dice(x, y)
    fraction = trunk_h / max(height, 1e-3)
    trunk = int(np.argmin([abs(fraction - f) for f in TREE_TRUNK_FRACTIONS]))
    return {
        "lobes": 2 if height >= 14.0 else 1,
        "trunk": trunk,
        "shape": int(shape_d * TREE_SHAPES_PER_KIND) % TREE_SHAPES_PER_KIND,
        "spin": spin_d * 2 * np.pi,
        "squash": 0.78 + 0.34 * squash_d,
    }


def _shape_lean(shape: int) -> float:
    """The lean of one unit tree shape, in units of its crown radius."""
    return (_tree_dice(31.0 * shape + 7.0, 17.0 * shape + 3.0)[3] - 0.5) * 0.06


def _unit_tree(lobes: int, trunk_fraction: float, shape: int):
    """One tree at unit size, as (vertices, faces, uvs) with its foot at 0.

    One unit tall and one unit of crown radius, so an instance scaled to
    (crown radius, crown radius x squash, height) is that tree. The same
    construction as the per-tree mesh this replaced -- tapered, flared trunk
    with a slight lean, lobed crown -- with the shape drawn from `shape`
    instead of from the tree's position.
    """
    spin = _tree_dice(31.0 * shape + 7.0, 17.0 * shape + 3.0)[0] * 2 * np.pi
    trunk_h = trunk_fraction
    crown_h = max(0.05, 1.0 - trunk_h)
    trunk_sides = 5
    angles = np.linspace(0, 2 * np.pi, trunk_sides, endpoint=False)
    cos_a, sin_a = np.cos(angles + spin), np.sin(angles + spin)
    trunk_r = 0.12
    lean = _shape_lean(shape)
    tops = np.column_stack([
        lean + trunk_r * 0.55 * cos_a, lean * 0.4 + trunk_r * 0.55 * sin_a,
        np.full(trunk_sides, trunk_h + 0.15 * crown_h),
    ])
    mids = np.column_stack([
        trunk_r * cos_a, trunk_r * sin_a, np.full(trunk_sides, trunk_h * 0.35),
    ])
    feet = np.column_stack([
        trunk_r * 1.5 * cos_a, trunk_r * 1.5 * sin_a, np.full(trunk_sides, -0.01),
    ])
    vertices = [np.vstack([feet, mids, tops])]
    faces = []
    for side in range(trunk_sides):
        nxt = (side + 1) % trunk_sides
        for lower_ring, upper_ring in ((0, 1), (1, 2)):
            a = lower_ring * trunk_sides
            b = upper_ring * trunk_sides
            faces.append([a + side, a + nxt, b + nxt])
            faces.append([a + side, b + nxt, b + side])
    uvs = [np.tile([0.25, 0.5], (len(faces) * 3, 1))]
    shape_verts, shape_faces = _octahedron_canopy(1)
    crown, crown_faces = _lobed_crown(shape_verts, shape_faces, lobes, 1.0, crown_h, 1.0, spin)
    crown = crown + np.array([lean, 0.0, trunk_h + crown_h * 0.5])
    offset = trunk_sides * 3
    vertices.append(crown)
    faces = np.vstack([np.asarray(faces), crown_faces + offset])
    uvs.append(np.tile([0.75, 0.5], (len(crown_faces) * 3, 1)))
    return np.vstack(vertices), faces, np.vstack(uvs)


def _tree_group_mesh(members, fields, origin, rng=None):
    """The trees exactly as drawn, flattened into one (vertices, faces, uvs).

    Not used to build the scene -- that instances shared shapes -- but the
    same transform the instances get, applied here, so anything that has to
    agree with the drawn trees (the collision proxies, the tests) can measure
    against what is drawn rather than against a description of it.
    """
    xy, ground, heights, crowns, trunks = fields
    origin_x, origin_y, z_offset = origin
    all_verts, all_faces, all_uvs = [], [], []
    offset = 0
    for index in np.flatnonzero(members):
        height = float(heights[index])
        crown_r = float(crowns[index])
        params = _tree_instance(xy[index, 0], xy[index, 1], height, float(trunks[index]))
        vertices, faces, uvs = _unit_tree(
            params["lobes"], TREE_TRUNK_FRACTIONS[params["trunk"]], params["shape"]
        )
        scaled = vertices * np.array([crown_r, crown_r * params["squash"], height])
        c, s_ = np.cos(params["spin"]), np.sin(params["spin"])
        turned = np.column_stack([
            scaled[:, 0] * c - scaled[:, 1] * s_,
            scaled[:, 0] * s_ + scaled[:, 1] * c,
            scaled[:, 2],
        ])
        turned += np.array([
            float(xy[index, 0]) - origin_x,
            float(xy[index, 1]) - origin_y,
            float(ground[index]) - z_offset,
        ])
        all_verts.append(turned)
        all_faces.append(faces + offset)
        all_uvs.append(uvs)
        offset += len(turned)
    if not all_verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2))
    return np.vstack(all_verts), np.vstack(all_faces), np.vstack(all_uvs)


def build_trees(scene: dict, work_dir: Path, tree_material):
    """Every tree its own object: an instance of one of a few shared shapes.

    Solid geometry rather than crossed billboards, so nothing depends on alpha
    settings surviving the FBX trip and being configured again in Unity.

    One object per tree is what makes a tree something you can select, move or
    delete in Blender, and its own GameObject in Unity. The shapes are shared
    meshes -- 32 of them -- so ten thousand trees cost ten thousand transforms,
    not ten thousand meshes: the FBX writes each shape once.

    Split into Trees_registered and Trees_detected. The BGT tree register is a
    municipal asset list rather than a survey of vegetation -- over a square
    kilometre of Utrecht centre it holds 1489 trees where the canopy model
    finds 8700 -- so most of what arrives was found rather than surveyed. The
    detector is good and not perfect, so the found ones stay their own group
    and a bad patch can be deleted together without touching a surveyed tree.
    """
    tree_file = scene.get("trees", {}).get("file")
    if not tree_file or not (work_dir / tree_file).is_file():
        return []

    data = np.load(work_dir / tree_file)
    xy = data["xy"]
    if len(xy) == 0:
        return []

    ground = data["ground_z_nap"]
    heights = data["height_m"]
    crowns = data["crown_radius_m"]
    trunks = data["trunk_height_m"]
    detected = (
        data["detected"] if "detected" in data.files
        else np.zeros(len(xy), dtype=bool)
    )
    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])

    shapes: dict[tuple, object] = {}

    def shape_for(lobes: int, trunk: int, kind: int):
        key = (lobes, trunk, kind)
        if key not in shapes:
            vertices, faces, uvs = _unit_tree(lobes, TREE_TRUNK_FRACTIONS[trunk], kind)
            n_triangles = len(faces)
            holder = build_mesh_object(
                f"TreeShape_{lobes}{trunk}{kind}",
                vertices,
                faces.reshape(-1),
                np.arange(0, n_triangles * 3, 3),
                np.full(n_triangles, 3),
                uvs,
                np.zeros(n_triangles),
                [tree_material],
                shade_smooth=False,
            )
            # Only the mesh is wanted; the object that came with it goes.
            mesh = holder.data
            bpy.data.objects.remove(holder)
            shapes[key] = mesh
        return shapes[key]

    objects = []
    for label, prefix, members in (
        ("Trees_registered", "Tree_", ~detected),
        ("Trees_detected", "Tree_d", detected),
    ):
        if not members.any():
            continue
        group = bpy.data.objects.new(label, None)
        bpy.context.scene.collection.objects.link(group)
        objects.append(group)
        for count, index in enumerate(np.flatnonzero(members), start=1):
            height = float(heights[index])
            crown_r = float(crowns[index])
            params = _tree_instance(xy[index, 0], xy[index, 1], height, float(trunks[index]))
            mesh = shape_for(params["lobes"], params["trunk"], params["shape"])
            tree = bpy.data.objects.new(f"{prefix}{count:05d}", mesh)
            tree.location = (
                float(xy[index, 0]) - origin_x,
                float(xy[index, 1]) - origin_y,
                float(ground[index]) - z_offset,
            )
            tree.rotation_euler[2] = params["spin"]
            tree.scale = (crown_r, crown_r * params["squash"], height)
            tree.parent = group
            tree["height_m"] = round(height, 2)
            tree["detected"] = bool(detected[index])
            bpy.context.scene.collection.objects.link(tree)
            objects.append(tree)
        log(f"{label}: {int(members.sum())} trees as instances of {len(shapes)} shapes")
    return objects


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


def _furniture_pieces(kind, x, y, base, spin, uv, cfg):
    """The boxes one piece of street furniture is made of.

    A proxy, not a model: three or four boxes that read correctly at the
    distance anyone will see them from. What matters is the silhouette -- a
    flagpole is tall and thin with something at the top, a bus shelter is a
    roof on posts -- because that is all the eye gets at street scale, and
    anything more would be a thousand objects nobody asked to be detailed.
    """
    metal, wood = uv["metal"], uv["wood"]
    stone, glass, paint = uv["stone"], uv["glass"], uv["paint"]
    lamp_h = float(cfg.get("lamp_height_m", 5.0))
    bollard_h = float(cfg.get("bollard_height_m", 0.9))
    bench_l = float(cfg.get("bench_length_m", 1.8))

    if kind == 0:  # lamppost: column plus a short arm
        return [
            (_box(x, y, base + lamp_h / 2, 0.14, 0.14, lamp_h, spin), metal),
            (_box(x, y, base + lamp_h + 0.08, 0.7, 0.24, 0.16, spin), metal),
        ]
    if kind == 1:  # bollard
        return [
            (_box(x, y, base + bollard_h / 2, 0.16, 0.16, bollard_h, spin), metal),
        ]
    if kind == 2:  # bench: seat on two legs
        return [
            (_box(x, y, base + 0.45, bench_l, 0.5, 0.08, spin), wood),
            (_box(x, y, base + 0.22, bench_l * 0.8, 0.1, 0.44, spin), metal),
        ]
    if kind == 3:  # sign post
        return [
            (_box(x, y, base + 1.1, 0.09, 0.09, 2.2, spin), metal),
            (_box(x, y, base + 2.1, 0.5, 0.05, 0.4, spin), metal),
        ]
    if kind == 4:  # flagpole: tall, thin, and a flag hanging off the top
        pole = float(cfg.get("flagpole_height_m", 7.0))
        return [
            (_box(x, y, base + pole / 2, 0.12, 0.12, pole, spin), metal),
            (_box(x, y, base + pole - 0.9, 0.06, 0.06, 0.3, spin), metal),
            # Hung against the pole rather than flying: a flag modelled flat
            # and horizontal reads as a mistake on a still day.
            (_box(x + 0.5 * np.cos(spin), y + 0.5 * np.sin(spin),
                  base + pole - 1.1, 1.0, 0.04, 0.7, spin), glass),
        ]
    if kind == 5:  # bus shelter: roof on four posts, one glazed side
        return [
            (_box(x, y, base + 2.45, 3.6, 1.6, 0.12, spin), metal),
            (_box(x, y, base + 1.2, 3.4, 0.06, 2.4, spin), glass),
            (_box(x, y, base + 0.5, 2.4, 0.4, 0.08, spin), wood),
        ]
    if kind == 6:  # advertising column: a drum on a plinth
        return [
            (_box(x, y, base + 0.1, 1.3, 1.3, 0.2, spin), stone),
            (_box(x, y, base + 1.5, 1.2, 1.2, 2.6, spin), glass),
            (_box(x, y, base + 2.9, 1.34, 1.34, 0.16, spin), metal),
        ]
    if kind == 7:  # art object: a plinth and something on it
        return [
            (_box(x, y, base + 0.2, 1.6, 1.6, 0.4, spin), stone),
            (_box(x, y, base + 1.5, 0.7, 0.7, 2.2, spin + 0.6), stone),
        ]
    if kind == 8:  # memorial: a wider plinth, a slab, and a step
        return [
            (_box(x, y, base + 0.15, 3.0, 2.2, 0.3, spin), stone),
            (_box(x, y, base + 0.5, 2.2, 1.4, 0.4, spin), stone),
            (_box(x, y, base + 1.7, 1.4, 0.5, 2.0, spin), stone),
        ]
    if kind == 9:  # playground: a frame with a crossbar and a slide
        return [
            (_box(x - 1.4 * np.cos(spin), y - 1.4 * np.sin(spin),
                  base + 1.1, 0.12, 0.12, 2.2, spin), paint),
            (_box(x + 1.4 * np.cos(spin), y + 1.4 * np.sin(spin),
                  base + 1.1, 0.12, 0.12, 2.2, spin), paint),
            (_box(x, y, base + 2.2, 3.0, 0.14, 0.14, spin), paint),
            (_box(x + 1.0 * np.sin(spin), y - 1.0 * np.cos(spin),
                  base + 0.6, 1.2, 0.7, 0.1, spin), paint),
        ]
    if kind == 10:  # picnic table: top and two benches
        return [
            (_box(x, y, base + 0.74, 1.8, 0.8, 0.07, spin), wood),
            (_box(x + 0.72 * np.sin(spin), y - 0.72 * np.cos(spin),
                  base + 0.45, 1.8, 0.3, 0.06, spin), wood),
            (_box(x - 0.72 * np.sin(spin), y + 0.72 * np.cos(spin),
                  base + 0.45, 1.8, 0.3, 0.06, spin), wood),
            (_box(x, y, base + 0.35, 0.2, 0.7, 0.7, spin), metal),
        ]
    if kind == 11:  # electrical cabinet: a box with a lid
        return [
            (_box(x, y, base + 0.6, 0.9, 0.45, 1.2, spin), paint),
            (_box(x, y, base + 1.23, 0.96, 0.5, 0.06, spin), metal),
        ]
    if kind == 12:  # waste collection point: a below-ground container's head
        return [
            (_box(x, y, base + 0.05, 1.5, 1.5, 0.1, spin), stone),
            (_box(x, y, base + 0.6, 0.8, 0.8, 1.0, spin), metal),
        ]
    if kind == 13:  # catenary mast: a tapered post with a bracket arm
        mast = float(cfg.get("catenary_height_m", 7.5))
        return [
            (_box(x, y, base + mast / 2, 0.24, 0.24, mast, spin), paint),
            (_box(x + 0.8 * np.cos(spin), y + 0.8 * np.sin(spin),
                  base + mast - 0.4, 1.6, 0.1, 0.1, spin), paint),
        ]
    # Anything unmapped would be a guess. Drawing nothing is the honest answer.
    return []


def build_barriers(scene: dict, work_dir: Path, material, fence_material=None):
    """Walls, fences and hedges, one object per BGT type.

    Split by type rather than delivered as one mesh, because that is the
    difference between "a wall" and "every wall": a designer can hide the
    fences to see the plots, swap the hedge material for a real one, or put
    the quay walls on their own collision layer without selecting anything by
    hand. Nearly all of them share one material, so the split costs nothing to
    draw; the see-through ones get a second, alpha-clipped one, because a fence
    is mostly holes and a solid slab reads as a hoarding.
    """
    barriers_file = scene.get("barriers", {}).get("file")
    if not barriers_file or not (work_dir / barriers_file).is_file():
        return []

    data = np.load(work_dir / barriers_file)
    vertices = data["vertices"]
    triangles = data["triangles"]
    if len(triangles) == 0:
        return []

    kind = data["kind"]
    kind_names = [str(n) for n in data["kind_names"]]
    material_of = data["material"]
    material_names = [str(n) for n in data["material_names"]]

    uv = data["uv"] if "uv" in data.files else None
    cutout = (
        {str(n) for n in data["see_through"]} if "see_through" in data.files else set()
    )
    tile_m = float(scene.get("barriers", {}).get("fence_tile_m", 2.5))
    table = scene.get("furniture", {}).get("atlas_uv") or {}
    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    local = vertices - np.array([origin_x, origin_y, z_offset])

    objects = []
    total = 0
    for index, name in enumerate(kind_names):
        mask = kind == index
        if not mask.any():
            continue
        part = triangles[mask]
        # Only the vertices this type actually uses, renumbered, so each
        # object carries its own buffer rather than a slice of everyone's.
        used = np.unique(part)
        renumber = np.zeros(len(local), dtype=np.int64)
        renumber[used] = np.arange(len(used))
        corners = local[used]
        indices = renumber[part]
        n_triangles = len(part)

        see_through = name in cutout and fence_material is not None and uv is not None
        if see_through:
            # Metres in the file, tiles here: a fence four hundred metres long
            # gets four hundred metres of uprights rather than eight stretched
            # across the whole run.
            loop_uv = uv[used][indices.reshape(-1)].copy()
            loop_uv[:, 0] /= tile_m      # U tiles along the run; V does not
        else:
            patch = np.asarray(
                table.get(material_names[int(material_of[mask][0])], (0.125, 0.25)),
                dtype=np.float64,
            )
            loop_uv = np.tile(patch, (n_triangles * 3, 1))

        objects.append(
            build_mesh_object(
                f"Barriers_{name}",
                corners,
                indices.reshape(-1),
                np.arange(0, n_triangles * 3, 3),
                np.full(n_triangles, 3),
                loop_uv,
                np.zeros(n_triangles),
                [fence_material if see_through else material],
                shade_smooth=False,
            )
        )
        total += n_triangles

    log(
        f"barriers: {len(objects)} kinds, {total} triangles "
        f"({', '.join(kind_names)})"
    )
    return objects


def build_furniture(scene: dict, work_dir: Path, material):
    """Street furniture as small groups of boxes, one shape per BGT type.

    Every piece shares one material and one atlas, so a couple of thousand
    objects cost one draw call rather than a couple of thousand.
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

    # Where each material sits in the atlas, from the scene file rather than
    # duplicated here. Falls back to the old two-patch split so an older
    # scene.json still renders.
    table = cfg.get("atlas_uv") or {}
    uv = {
        name: np.asarray(table.get(name, default), dtype=np.float64)
        for name, default in (
            ("metal", (0.125, 0.25)), ("wood", (0.375, 0.25)),
            ("brick", (0.625, 0.25)), ("concrete", (0.875, 0.25)),
            ("hedge", (0.125, 0.75)), ("paint", (0.375, 0.75)),
            ("stone", (0.625, 0.75)), ("glass", (0.875, 0.75)),
        )
    }

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    offset = 0
    rng = np.random.default_rng(808)
    drawn = 0

    for index in range(len(xy)):
        x = float(xy[index, 0]) - origin_x
        y = float(xy[index, 1]) - origin_y
        base = float(ground[index]) - z_offset
        kind = int(kinds[index])
        spin = float(rng.random()) * np.pi

        pieces = _furniture_pieces(kind, x, y, base, spin, uv, cfg)
        if pieces:
            drawn += 1

        for (piece_verts, piece_faces), uv_patch in pieces:
            vertices.append(piece_verts)
            faces.append(piece_faces + offset)
            uvs.append(np.tile(uv_patch, (len(piece_faces) * 3, 1)))
            offset += len(piece_verts)

    if not vertices:
        log("street furniture: nothing of a drawable type")
        return None

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
    log(
        f"street furniture: {drawn} of {len(xy)} objects drawn, "
        f"{n_triangles} triangles"
    )
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


def build_rails(scene: dict, work_dir: Path, material):
    """Track: a ballast bed with the two rails on it, one object per kind.

    Heavy rail, tram and metro become separate objects for the same reason the
    roads do — each is its own GameObject, so a tram route and a railway can be
    treated differently downstream without splitting anything by hand.

    The UVs were built upstream, running in metres along the track, so the
    sleepers in the ballast texture stay evenly spaced through curves.
    """
    rail_file = scene.get("rails", {}).get("file")
    if not rail_file or not (work_dir / rail_file).is_file():
        return []

    data = np.load(work_dir / rail_file)
    if not len(data["ballast_tris"]) and not len(data["rail_tris"]):
        return []

    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    names = scene.get("rails", {}).get("kind_names", {})
    # Metres of track one tile of the atlas covers.
    tile_m = float(scene.get("rails", {}).get("tile_length_m", 2.0))

    def to_local(tris):
        if not len(tris):
            return tris
        out = tris.copy()
        out[:, :, 0] -= origin_x
        out[:, :, 1] -= origin_y
        out[:, :, 2] -= z_offset
        return out

    ballast = to_local(data["ballast_tris"])
    rails = to_local(data["rail_tris"])
    ballast_kind = data["ballast_kind"]
    rail_kind = data["rail_kind"]

    # The atlas is ballast on the left half, rail steel on the right, so U is
    # squeezed into one half or the other rather than spanning the image.
    def atlas_uv(uv, left: bool):
        out = uv.copy()
        out[:, 0] = out[:, 0] * 0.46 + (0.02 if left else 0.52)
        out[:, 1] = out[:, 1] / tile_m
        return out

    objects = []
    for code in sorted(set(int(k) for k in np.concatenate([ballast_kind, rail_kind]))):
        parts, uvs = [], []
        for tris, kinds, uv, left in (
            (ballast, ballast_kind, data["ballast_uv"], True),
            (rails, rail_kind, data["rail_uv"], False),
        ):
            if not len(tris):
                continue
            mask = kinds == code
            if not mask.any():
                continue
            parts.append(tris[mask])
            uvs.append(atlas_uv(uv.reshape(-1, 3, 2)[mask].reshape(-1, 2), left))
        if not parts:
            continue

        triangles = np.concatenate(parts)
        corners = triangles.reshape(-1, 3)
        n_triangles = len(triangles)
        label = str(names.get(str(code), f"kind_{code}"))

        objects.append(
            build_mesh_object(
                f"Rails_{label}",
                corners,
                np.arange(n_triangles * 3),
                np.arange(0, n_triangles * 3, 3),
                np.full(n_triangles, 3),
                np.concatenate(uvs),
                np.zeros(n_triangles),
                [material],
                shade_smooth=False,
            )
        )
        log(f"rails: {label}, {n_triangles} triangles")

    return objects


def build_structures(scene: dict, work_dir: Path, material):
    """Bridge decks, the piers under them, and tunnels below the ground.

    One object per kind, so a deck can be given a drivable collider and a pier
    cannot, and so the tunnel is selectable on its own.
    """
    structures_file = scene.get("structures", {}).get("file")
    if not structures_file or not (work_dir / structures_file).is_file():
        return []

    data = np.load(work_dir / structures_file)
    triangles = data["tris"]
    if len(triangles) == 0:
        return []

    origin_x, origin_y = scene["origin_rd"]
    z_offset = float(scene["ground_z_offset_nap"])
    names = scene.get("structures", {}).get("kind_names", {})

    local = triangles.copy()
    local[:, :, 0] -= origin_x
    local[:, :, 1] -= origin_y
    local[:, :, 2] -= z_offset
    kinds = data["tri_kind"]

    objects = []
    for code in sorted(set(int(k) for k in kinds)):
        part = local[kinds == code]
        if not len(part):
            continue
        # Welded, so a deck arrives indexed. Every triangle used to carry its
        # own three corners, which is three times the vertex buffer with no
        # reuse at all -- and no shared normals, so an engine cannot smooth the
        # surface even where it should be smooth.
        corners, indices = _weld_corners(part.reshape(-1, 3))
        n_triangles = len(part)
        label = str(names.get(str(code), f"kind_{code}"))

        # Metre UVs: a deck and a tunnel wall both want a tiling concrete, not
        # a photo stretched over them.
        #
        # Per loop, not per vertex. Blender's uv layer holds one pair for every
        # corner of every face, so after a weld the UVs follow `indices` rather
        # than `corners` -- and the loop indices have to be `indices` too, or
        # they address a buffer that no longer exists at that length.
        loop_xy = corners[indices][:, :2]
        uvs = np.column_stack([loop_xy[:, 0] / 6.0, loop_xy[:, 1] / 6.0])
        objects.append(
            build_mesh_object(
                f"Structures_{label}",
                corners,
                indices,
                np.arange(0, n_triangles * 3, 3),
                np.full(n_triangles, 3),
                uvs,
                np.zeros(n_triangles),
                [material],
                shade_smooth=False,
            )
        )
        log(f"structures: {label}, {n_triangles} triangles")

    return objects


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------

def reused_colliders() -> dict:
    """Which visible object serves as its own collider, and why.

    A simulation needs to know what to collide with for every layer, not only
    for the two that get a proxy. Shipping a copy of a mesh that is already the
    right shape is only a bigger FBX, so these are named instead.
    """
    return {
        "Terrain": "a regular grid already; a coarser one would not be the "
                   "ground the drone is flying over",
        "Pand_*": "walls are a prism per footprint edge and the roof is a "
                   "few triangles, so there is nothing to take out",
        "Road_*, Path_*, Junction_*, Lanes_*, Paths_*, Paving": (
            "the road surface itself, kerbs included: a drone that lands wants "
            "the real camber and a car wants the real kerb"
        ),
        "Water": "one flat quad per body",
        "Structures_*": "bridge decks and piers, already low-poly prisms",
    }


def _collision_tree_mesh(members, fields, origin, crowns: bool):
    """A trunk prism and a crown box per tree, as (vertices, faces).

    This is not the drawn tree with fewer triangles -- it is the volume the
    drawn tree occupies, and the difference decides whether it is any use. The
    drawn crown is two overlapping lobes, deliberately concave and
    self-intersecting so no two trees match; a physics engine wants neither.

    **It has to enclose, and a plausible-looking proxy does not.** The first
    attempt was the ellipsoid inscribed in the crown's bounding box -- 32
    triangles, convex, visually a good fit. Measured against the drawn
    vertices, foliage stuck out of it by up to 4.3 m, because the drawn crown
    fills that box while the inscribed ellipsoid only touches it on the axes.
    A drone would have flown through the visible canopy and felt nothing.

    So the crown is the box itself. That is 12 triangles rather than 32, it
    contains the drawn crown by construction -- `_lobed_crown` scales its
    lobes to exactly this box -- and where it is wrong it is wrong in the safe
    direction: a drone stops short at a corner instead of passing through
    leaves. The trunk is a straight prism at the flare radius, widened by the
    lean so it covers all three of the drawn trunk's rings, and spun to the
    same angle so its five faces line up rather than cutting the corners off.

    `crowns=False` leaves the canopy out: 16 triangles a tree instead of 28,
    for a simulation that treats foliage as soft and only wants wood to be
    solid.
    """
    xy, ground, heights, crown_radii, trunks = fields
    origin_x, origin_y, z_offset = origin

    sides = 5
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0

    for index in np.flatnonzero(members):
        base_x = float(xy[index, 0]) - origin_x
        base_y = float(xy[index, 1]) - origin_y
        base_z = float(ground[index]) - z_offset
        height = float(heights[index])
        crown_r = float(crown_radii[index])
        trunk_h = float(trunks[index])

        # The same instance the drawn tree is, so the proxy lands on that tree
        # rather than near it. Matching it exactly is the whole job: a proxy
        # built from different numbers is a proxy for nothing.
        params = _tree_instance(xy[index, 0], xy[index, 1], height, trunk_h)
        spin = params["spin"]
        squash = params["squash"]
        fraction = TREE_TRUNK_FRACTIONS[params["trunk"]]
        crown_h = max(0.05, 1.0 - fraction) * height
        canopy_base = base_z + fraction * height
        # The unit shape's lean, in its own x, scaled by the crown radius and
        # turned with the tree.
        lean = _shape_lean(params["shape"]) * crown_r
        cos_s, sin_s = np.cos(spin), np.sin(spin)

        # Trunk. The drawn one has three rings -- flared feet, plain middle,
        # tapered top -- and the feet are the widest, so one prism at the flare
        # radius covers all three. Squashed sideways with the crown, so the
        # radius takes the wider of the two; the lean goes into the radius
        # instead of tilting the prism.
        # The top ring is pushed by the lean in the shape's own x, and by
        # four tenths of it in y, which the squash then scales.
        # And the drawn trunk is turned by its shape's own angle, not the
        # tree's, so its corners can fall anywhere round the prism: the prism
        # is sized to circumscribe the circle they lie in, not to pass through
        # them. A pentagon's sides sit at cos(36 deg) of its corner radius.
        trunk_r = (
            0.12 * 1.5 * crown_r * max(1.0, squash)
            + abs(lean) * (1.0 + 0.4 * squash)
            + 0.01
        ) / np.cos(np.pi / sides)
        cos_a = np.cos(angles + spin)
        sin_a = np.sin(angles + spin)
        rings = [
            np.column_stack([
                base_x + trunk_r * cos_a,
                base_y + trunk_r * sin_a,
                np.full(sides, z),
            ])
            for z in (base_z - 0.02 * height, canopy_base + 0.15 * crown_h)
        ]
        all_verts.append(np.vstack(rings))
        for side in range(sides):
            nxt = (side + 1) % sides
            all_faces.append(np.array([
                [offset + side, offset + nxt, offset + sides + nxt],
                [offset + side, offset + sides + nxt, offset + sides + side],
            ]))
        # Caps, so the prism is a solid rather than a pipe. A fan off the first
        # vertex is enough for a convex polygon.
        for ring, flip in ((0, True), (sides, False)):
            for side in range(1, sides - 1):
                fan = [offset + ring, offset + ring + side,
                       offset + ring + side + 1]
                all_faces.append(np.array([fan[::-1] if flip else fan]))
        offset += sides * 2

        if crowns:
            # Exactly the box the unit crown is fitted to, scaled and turned
            # as the instance is.
            box, box_faces = _box(
                base_x + lean * cos_s, base_y + lean * sin_s,
                canopy_base + crown_h * 0.5,
                crown_r * 2.0, crown_r * squash * 2.0, crown_h, spin=spin,
            )
            all_verts.append(box)
            all_faces.append(box_faces + offset)
            offset += len(box)

    if not all_verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(all_verts), np.vstack(all_faces)


def build_collision(scene: dict, work_dir: Path):
    """Physics proxies, named so a designer can set them up in one pass.

    Two objects, and only two, because only two earn their place. Everything
    else in the scene is already the right shape to collide with, and
    `reused_colliders` records which object to use instead -- a copy of a
    mesh that is already minimal is only a bigger file.

    No materials: this geometry is never drawn. In Unity the pass is select the
    Collision_* objects, add a Mesh Collider, untick Mesh Renderer.

    Returns `{object name: triangle count}` rather than the objects the other
    builders return, because the count is what the summary reports and reading
    it back off a Blender object is the one thing that cannot be exercised
    without Blender.
    """
    export_cfg = scene.get("export", {})
    if not bool(export_cfg.get("collision", False)):
        return {}

    built: dict[str, int] = {}

    def add(name: str, vertices, faces):
        n = len(faces)
        if not n:
            return
        build_mesh_object(
            name,
            vertices,
            np.asarray(faces).reshape(-1),
            np.arange(0, n * 3, 3),
            np.full(n, 3),
            np.zeros((n * 3, 2)),
            np.zeros(n),
            [],
            shade_smooth=False,
        )
        built[name] = n
        log(f"{name}: {n} triangles")

    # Barriers. The reason this layer exists: a fence is drawn 6 cm thick, and
    # at Unity's default 50 Hz discrete stepping a body doing 20 m/s crosses
    # 40 cm between steps and never touches it.
    barriers_file = scene.get("barriers", {}).get("file")
    if barriers_file and (work_dir / barriers_file).is_file():
        data = np.load(work_dir / barriers_file)
        if "collision_triangles" in data.files and len(data["collision_triangles"]):
            origin_x, origin_y = scene["origin_rd"]
            z_offset = float(scene["ground_z_offset_nap"])
            add(
                "Collision_Barriers",
                data["collision_vertices"]
                - np.array([origin_x, origin_y, z_offset]),
                data["collision_triangles"],
            )

    # Trees. The drawn crown is two overlapping lobes on purpose, which is
    # concave and self-intersecting; the proxy is the ellipsoid they fill.
    tree_file = scene.get("trees", {}).get("file")
    if tree_file and (work_dir / tree_file).is_file():
        data = np.load(work_dir / tree_file)
        xy = data["xy"]
        if len(xy):
            fields = (xy, data["ground_z_nap"], data["height_m"],
                      data["crown_radius_m"], data["trunk_height_m"])
            origin = (*scene["origin_rd"], float(scene["ground_z_offset_nap"]))
            verts, faces = _collision_tree_mesh(
                np.ones(len(xy), dtype=bool),
                fields,
                origin,
                bool(export_cfg.get("collision_tree_crowns", True)),
            )
            add("Collision_Trees", verts, faces)

    if built:
        log(
            "collision: "
            + ", ".join(sorted(built))
            + "; everything else collides with its own mesh ("
            + ", ".join(sorted(reused_colliders())) + ")"
        )
    return built


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


# The FBX settings, in one place. The Maptile tools carry a copy so a
# re-export from an edited .blend matches; a test keeps the two equal.
FBX_SETTINGS = dict(
    use_selection=False,
    apply_unit_scale=True,
    # "FBX All": every object arrives in Unity at scale 1.
    apply_scale_options="FBX_SCALE_ALL",
    global_scale=1.0,
    axis_forward="-Z",
    axis_up="Y",
    # Not Apply Transform. It would put the axis conversion into the vertices
    # and leave every object unrotated -- except that it cannot rewrite a mesh
    # several objects share, and the trees are ten thousand instances of a
    # dozen meshes: measured, each tree came out at its Blender coordinates
    # in Unity's frame, 375 m below ground. The conversion stays on the root
    # objects as a -90 degree X rotation instead, and Unity's own "Bake Axis
    # Conversion" import option takes it off, instances included.
    bake_space_transform=False,
    # Empties carry the hierarchy: Roads, Buildings, Trees and the rest are
    # parents in Unity as they are in Blender.
    object_types={"MESH", "EMPTY"},
    # The triangulation modifiers are render-only, so the export applies them
    # and the .blend keeps its quads.
    use_mesh_modifiers=True,
    use_mesh_modifiers_render=True,
    # BAG id, build year and height per building; height per tree. Unity can
    # read them in an AssetPostprocessor.
    use_custom_props=True,
    mesh_smooth_type="FACE",
    use_tspace=True,
    path_mode="STRIP",
    embed_textures=False,
    bake_anim=False,
)


def export_fbx(out_path: Path) -> None:
    """Write the scene to FBX for Unity.

    Blender is right-handed Z-up and Unity is left-handed Y-up. Exporting with
    forward -Z and up Y sends east to Unity's X, NAP height to Y and north to Z,
    and the handedness change cancels against the axis change, so the model is
    not mirrored.

    Texture paths are stripped to bare filenames because the images are written
    next to the FBX, which is where Unity looks for them.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(filepath=str(out_path), **FBX_SETTINGS)
    log(f"exported {out_path}")


def save_blend(out_dir: Path, name: str = "model.blend") -> Path:
    """Save the scene as a .blend for editing, tools included.

    Texture paths are made relative, so the folder can be moved or zipped and
    the .blend still finds its images beside it. The Maptile tools go in as a
    text block registered on load (when Blender is allowed to run scripts) and
    as a file next to the .blend to install as an add-on.
    """
    tools = Path(__file__).with_name("maptile_tools.py")
    if tools.is_file():
        shutil.copyfile(tools, out_dir / tools.name)
        text = bpy.data.texts.load(str(out_dir / tools.name))
        text.name = tools.name
        text.use_module = True
    path = out_dir / name
    # No model.blend1 backup beside it: this is a fresh file every run.
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_mainfile(compress=True)
    log(f"saved {path}")
    return path


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
        ("fence", "barriers"),
        ("vehicle", "vehicles"),
        ("rail", "rails"),
        ("structure", "structures"),
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
            # Named after the texture, so the slot says which era and which
            # look it is -- "M_facade_04_historic_1" rather than
            # "M_facade_04". With three looks an area carries seventeen wall
            # materials, and a designer reassigning one has to be able to tell
            # them apart in a list.
            "M_facade"
            if len(wall_textures) == 1
            else f"M_facade_{colour.stem.replace('facade_', '')}",
            colour,
            roughness=0.75,
            normal_path=normal,
        )
        for colour, normal in wall_textures
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

    # Each layer its own collection in the .blend and its own parent in the
    # FBX, so Unity's hierarchy reads Roads > Road_Velperweg rather than a
    # flat list of two thousand names.
    with _Layer("Terrain"):
        build_terrain(scene, work_dir, aerial_material)
    with _Layer("Roads", "Roads"):
        # Just above the terrain, one object per street, junction and stretch
        # of paving, each surface class its own material slot.
        build_roads(
            scene,
            work_dir,
            lambda name: make_textured_material(name, aerial_texture, roughness=0.85),
        )
    with _Layer("Buildings", "Buildings"):
        build_buildings(
            scene, work_dir, facade_materials, aerial_material, ground_materials
        )

    if "water" in extra_textures:
        # Water is the one smooth thing in the scene, so it gets a low
        # roughness while everything else stays matte.
        with _Layer("Water"):
            build_water(
                scene,
                work_dir,
                make_textured_material("M_water", extra_textures["water"], roughness=0.12),
            )

    if "tree" in extra_textures:
        with _Layer("Trees", "Trees"):
            build_trees(
                scene,
                work_dir,
                make_textured_material("M_tree", extra_textures["tree"], roughness=0.85),
            )

    if "furniture" in extra_textures:
        # One material for both: the atlas covers a brick wall and a wooden
        # bench alike, and sharing it keeps the whole streetscape to one draw.
        furniture_material = make_textured_material(
            "M_furniture", extra_textures["furniture"], roughness=0.55
        )
        with _Layer("Street furniture"):
            build_furniture(scene, work_dir, furniture_material)
        with _Layer("Barriers", "Barriers"):
            build_barriers(
                scene,
                work_dir,
                furniture_material,
                make_cutout_material("M_fence", extra_textures["fence"], roughness=0.6)
                if "fence" in extra_textures
                else None,
            )

    if "vehicle" in extra_textures:
        # Car paint is the one glossy thing out here.
        with _Layer("Vehicles"):
            build_vehicles(
                scene,
                work_dir,
                make_textured_material(
                    "M_vehicle", extra_textures["vehicle"], roughness=0.35
                ),
            )

    if "rail" in extra_textures:
        with _Layer("Rail", "Rails"):
            build_rails(
                scene,
                work_dir,
                make_textured_material("M_rail", extra_textures["rail"], roughness=0.7),
            )

    if "structure" in extra_textures:
        with _Layer("Structures", "Structures"):
            build_structures(
                scene,
                work_dir,
                make_textured_material(
                    "M_structure", extra_textures["structure"], roughness=0.8
                ),
            )

    # After everything visible, because the proxies are built from the same
    # intermediates and the log reads better with them at the end.
    with _Layer("Collision", "Collision"):
        collision = build_collision(scene, work_dir)
    # Present in the file and in the FBX, but out of the way while editing.
    layer = bpy.context.view_layer.layer_collection.children.get("Collision")
    if layer is not None:
        layer.hide_viewport = True

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
    blend = save_blend(out_dir, scene["export"].get("blend_name", "model.blend"))

    summary = {
        "fbx": scene["export"]["fbx_name"],
        "blend": blend.name,
        "layers": {
            collection.name: len(collection.all_objects)
            for collection in bpy.context.scene.collection.children
        },
        "bounds_local": {
            "x": [min(xs), max(xs)],
            "y": [min(ys), max(ys)],
            "z": [min(zs), max(zs)],
        },
        "objects": [obj.name for obj in bpy.context.scene.objects],
        "materials": [m.name for m in bpy.data.materials],
        "collision": {
            "objects": collision,
            # The layers with no proxy, and which object to collide with
            # instead. Without this a designer has to guess from 25 names.
            "reuses_its_own_mesh": reused_colliders(),
            "note": (
                "Add a Mesh Collider to each Collision_* object and untick its "
                "Mesh Renderer. Barrier proxies are fattened so a body cannot "
                "tunnel through a 6 cm fence at Unity's default 50 Hz discrete "
                "stepping; above about 15 m/s set the moving body's Collision "
                "Detection to Continuous Dynamic as well."
            ),
        }
        if collision
        else {},
        # What the terrain actually came out as. Every check upstream reads the
        # mesh the Python stages built in memory, which says nothing about what
        # this script drew: a scene.json missing its mesh_file shipped the
        # plain grid for weeks with every terrain check passing.
        "terrain": _terrain_summary(),
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
