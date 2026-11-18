"""The model for the page's 3D view: one binary glTF of the scene as built.

    python  blender/web_export.py --blend output/<area>/model.blend --out work/<area>/viewer.glb
    blender --background --factory-startup --python blender/web_export.py -- --blend ... --out ...

process.py calls `write` itself right after it saves the .blend, so a model
opens in the browser the moment its build finishes. The UI runs this script
for a model built before there was a 3D view, and for one edited and saved in
Blender since, which it tells by the stamp written beside the file.

It is a cache for the page, not a deliverable. It lives in work/<area>/, out
of the folder that is copied into Unity, and names the textures in
output/<area>/ by file name instead of carrying copies of them: the aerial
alone is tens of megabytes.

What the page needs, and what this writes for it:

* The scene as it renders. Every modifier is switched to its render setting
  for the write, as for the FBX, so the kerbs and the export triangulation
  are in and the view shows what Unity gets.
* Few draw calls. Every object of a layer is merged into one mesh per
  material: a thousand buildings are a few dozen primitives, not thousands.
* Instances kept as instances. A mesh many objects share -- the trees -- is
  written once with a transform per object (EXT_mesh_gpu_instancing).
* Every object still findable. Each vertex carries the number of its object
  (`_OBJECT`) and the file lists the objects with their properties, so a
  click in the view can say "Pand_0344100000000011, built 1450".
* Small. Normals are bytes (KHR_mesh_quantization), and a primitive without
  a texture has no texture coordinates.

Only what Blender bundles may be imported here; see the note in process.py.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import sys
import time
import urllib.parse
from pathlib import Path

import bpy
import numpy as np

# The layout of the file. The page asks for this number and the server holds
# the same one (VIEWER_FORMAT in ui/server.py; a test keeps them equal), so a
# change here makes every cached view be written again rather than misread.
FORMAT = 1

FILE_NAME = "viewer.glb"
STAMP_NAME = "viewer.json"

# Below this many users a shared mesh is merged like any other: an instanced
# draw for two benches costs more than the vertices it saves.
INSTANCE_MIN_USERS = 8

# Present in the file, never drawn: the physics proxies.
SKIPPED_COLLECTIONS = frozenset({"Collision"})

GEOMETRY_TYPES = frozenset({"MESH", "CURVE", "SURFACE", "FONT", "META"})

# glTF is Y up with -Z forward; Blender here is X east, Y north, Z up. A point
# (x, y, z) goes to (x, z, -y): east stays X, up becomes Y, north is -Z.
TO_GLTF = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
BYTE = 5120
UNSIGNED_SHORT = 5123
UNSIGNED_INT = 5125
FLOAT = 5126


def log(message: str) -> None:
    print(f"[viewer] {message}", flush=True)


# ---------------------------------------------------------------------------
# The binary container
# ---------------------------------------------------------------------------


class GlbBuilder:
    """Buffer views and accessors over one binary chunk.

    Every view starts on a four-byte boundary, which glTF requires of any
    view holding floats or 32-bit integers.
    """

    def __init__(self) -> None:
        self.binary = io.BytesIO()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, data: bytes, target: int | None = None, stride: int | None = None) -> int:
        padding = -self.binary.tell() % 4
        if padding:
            self.binary.write(b"\0" * padding)
        entry = {"buffer": 0, "byteOffset": self.binary.tell(), "byteLength": len(data)}
        if target is not None:
            entry["target"] = target
        if stride is not None:
            entry["byteStride"] = stride
        self.binary.write(data)
        self.views.append(entry)
        return len(self.views) - 1

    def accessor(
        self,
        array: np.ndarray,
        component: int,
        kind: str,
        *,
        target: int | None = None,
        normalized: bool = False,
        bounds: bool = False,
    ) -> int:
        """An accessor over a fresh view of `array`, one element per row."""
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[kind]
        dtype = {BYTE: np.int8, UNSIGNED_SHORT: np.uint16, UNSIGNED_INT: np.uint32,
                 FLOAT: np.float32}[component]
        rows = np.ascontiguousarray(np.asarray(array, dtype=dtype).reshape(-1, width))
        stride = None
        if component == BYTE and width == 3:
            # A vertex attribute's stride has to be a multiple of four, so a
            # byte normal is padded out to four bytes.
            padded = np.zeros((len(rows), 4), dtype=np.int8)
            padded[:, :3] = rows
            data = padded.tobytes()
            stride = 4
        else:
            data = rows.tobytes()
        entry = {
            "bufferView": self.view(data, target, stride),
            "componentType": component,
            "count": int(len(rows)),
            "type": kind,
        }
        if normalized:
            entry["normalized"] = True
        if bounds and len(rows):
            entry["min"] = [float(v) for v in rows.min(axis=0)]
            entry["max"] = [float(v) for v in rows.max(axis=0)]
        self.accessors.append(entry)
        return len(self.accessors) - 1

    def indices(self, triangles: np.ndarray) -> int:
        flat = np.asarray(triangles).reshape(-1)
        small = not len(flat) or int(flat.max()) < 65535
        return self.accessor(
            flat, UNSIGNED_SHORT if small else UNSIGNED_INT, "SCALAR",
            target=ELEMENT_ARRAY_BUFFER,
        )

    def payload(self) -> bytes:
        data = self.binary.getvalue()
        return data + b"\0" * (-len(data) % 4)


def pack_glb(document: dict, binary: bytes) -> bytes:
    """A binary glTF: header, the JSON chunk, the binary chunk."""
    text = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + (8 + len(binary) if binary else 0)
    out = bytearray(struct.pack("<III", GLB_MAGIC, 2, total))
    out += struct.pack("<II", len(text), CHUNK_JSON) + text
    if binary:
        out += struct.pack("<II", len(binary), CHUNK_BIN) + binary
    return bytes(out)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def quantize_normals(normals: np.ndarray) -> np.ndarray:
    """Unit vectors as signed bytes, the way KHR_mesh_quantization reads them."""
    normals = np.asarray(normals, dtype=np.float64)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    unit = np.divide(normals, length, out=np.zeros_like(normals), where=length > 1e-12)
    return np.clip(np.round(unit * 127.0), -127, 127).astype(np.int8)


def weld_corners(
    corner_vertex: np.ndarray, corner_normal_q: np.ndarray, corner_uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One vertex per distinct point, normal and texture coordinate.

    After the export triangulation every triangle has corners of its own, so
    without this a smooth terrain would carry each point six times over.
    Returns the corner standing for each vertex, and each corner's vertex.
    """
    key = np.zeros(
        len(corner_vertex), dtype=[("v", "<i4"), ("n", "i1", (4,)), ("u", "<f4", (2,))]
    )
    key["v"] = corner_vertex
    key["n"][:, :3] = corner_normal_q
    key["u"] = corner_uv
    flat = key.view(np.dtype((np.void, key.dtype.itemsize)))
    _, first, inverse = np.unique(flat, return_index=True, return_inverse=True)
    return first, np.asarray(inverse).reshape(-1)


def object_triangles(obj) -> dict | None:
    """An evaluated object's triangles, corner by corner, in its own space."""
    mesh = obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        n_triangles = len(mesh.loop_triangles)
        if not n_triangles:
            return None
        corners = np.empty(n_triangles * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", corners)
        slots = np.empty(n_triangles, dtype=np.int32)
        mesh.loop_triangles.foreach_get("material_index", slots)

        points = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", points)
        n_loops = len(mesh.loops)
        loop_vertex = np.empty(n_loops, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_vertex)

        normals = np.empty(n_loops * 3, dtype=np.float32)
        if hasattr(mesh, "corner_normals"):
            # Blender 4.1 and later; custom normals included.
            mesh.corner_normals.foreach_get("vector", normals)
        else:
            mesh.calc_normals_split()
            mesh.loops.foreach_get("normal", normals)

        layer = next((uv for uv in mesh.uv_layers if uv.active_render), None)
        layer = layer or mesh.uv_layers.active
        uvs = np.zeros(n_loops * 2, dtype=np.float32)
        if layer is not None and n_loops:
            layer.data.foreach_get("uv", uvs)

        materials = list(mesh.materials)
    finally:
        obj.to_mesh_clear()

    return {
        "points": points.reshape(-1, 3),
        "corner_vertex": loop_vertex[corners],
        "corner_normal": normals.reshape(-1, 3)[corners],
        "corner_uv": uvs.reshape(-1, 2)[corners],
        "slots": slots,
        "materials": materials,
    }


def triangles_as_vertices(tris: dict, matrix: np.ndarray | None) -> dict:
    """Weld the corners into vertices, placed by `matrix`, in glTF axes."""
    normal_q = quantize_normals(tris["corner_normal"])
    first, inverse = weld_corners(tris["corner_vertex"], normal_q, tris["corner_uv"])

    points = tris["points"][tris["corner_vertex"][first]].astype(np.float64)
    normals = tris["corner_normal"][first].astype(np.float64)
    triangles = inverse.reshape(-1, 3)
    if matrix is not None:
        linear = matrix[:3, :3]
        points = points @ linear.T + matrix[:3, 3]
        # Normals go through the inverse transpose, which keeps them square
        # to their faces under a scale that differs per axis.
        try:
            normals = normals @ np.linalg.inv(linear)
        except np.linalg.LinAlgError:
            pass
        if np.linalg.det(linear) < 0:
            # A mirrored object turns its triangles inside out; turn them back
            # so their winding still agrees with their normals.
            triangles = triangles[:, [0, 2, 1]]

    uv = tris["corner_uv"][first].astype(np.float32)
    # glTF reads an image top row first, Blender bottom row first.
    uv[:, 1] = 1.0 - uv[:, 1]
    return {
        "points": (points @ TO_GLTF.T).astype(np.float32),
        "normals": quantize_normals(normals @ TO_GLTF.T),
        "uv": uv,
        "triangles": triangles,
    }


def split_by_slot(vertices: dict, slots: np.ndarray):
    """Yield each material slot's triangles with just the vertices they use."""
    for slot in np.unique(slots):
        triangles = vertices["triangles"][slots == slot]
        used, remapped = np.unique(triangles, return_inverse=True)
        yield int(slot), {
            "points": vertices["points"][used],
            "normals": vertices["normals"][used],
            "uv": vertices["uv"][used],
            "triangles": np.asarray(remapped).reshape(-1, 3).astype(np.uint32),
        }


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def _surface_shader(material):
    tree = getattr(material, "node_tree", None)
    if tree is None:
        return None
    output = next(
        (n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None
    ) or next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if output is not None and output.inputs["Surface"].is_linked:
        node = output.inputs["Surface"].links[0].from_node
        if node.type == "BSDF_PRINCIPLED":
            return node
    return next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def _image_file(socket, texture_dir: Path) -> tuple[str | None, str | None]:
    """The file an image node feeding `socket` reads, if it is beside the model,
    and which of the node's outputs is used."""
    if socket is None or not socket.is_linked:
        return None, None
    link = socket.links[0]
    node = link.from_node
    if node.type != "TEX_IMAGE" or node.image is None:
        return None, None
    name = Path(bpy.path.abspath(node.image.filepath) or node.image.name).name
    if not name or not (texture_dir / name).is_file():
        return None, None
    return name, link.from_socket.name


def describe_material(material, texture_dir: Path) -> dict:
    """What the page needs of a material: a colour or a picture, a roughness,
    and whether its transparent pixels are holes."""
    description = {
        "name": material.name if material is not None else "Default",
        "color": [0.8, 0.8, 0.8, 1.0],
        "texture": None,
        "normal_texture": None,
        "roughness": 0.8,
        "metallic": 0.0,
        "cutout": False,
    }
    if material is None:
        return description
    description["color"] = [float(c) for c in material.diffuse_color]

    shader = _surface_shader(material)
    if shader is None:
        return description

    base = shader.inputs.get("Base Color")
    texture, _ = _image_file(base, texture_dir)
    if texture:
        description["texture"] = texture
        description["color"] = [1.0, 1.0, 1.0, 1.0]
    elif base is not None and not base.is_linked:
        description["color"] = [float(c) for c in base.default_value]
    elif base is not None and base.links[0].from_node.type == "RGB":
        description["color"] = [float(c) for c in base.links[0].from_node.outputs[0].default_value]

    for key, socket_name in (("roughness", "Roughness"), ("metallic", "Metallic")):
        socket = shader.inputs.get(socket_name)
        if socket is not None and not socket.is_linked:
            description[key] = float(socket.default_value)

    alpha = shader.inputs.get("Alpha")
    if alpha is not None and alpha.is_linked:
        source, output = _image_file(alpha, texture_dir)
        description["cutout"] = bool(source) and output == "Alpha"

    normal = shader.inputs.get("Normal")
    if normal is not None and normal.is_linked and normal.links[0].from_node.type == "NORMAL_MAP":
        description["normal_texture"], _ = _image_file(
            normal.links[0].from_node.inputs.get("Color"), texture_dir
        )
    return description


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def render_settings():
    """Every modifier as the render has it, for as long as the write runs.

    The same rule as the FBX export in process.py and the TerreinLab tools:
    the triangulation the model is exported with is render-only, and the
    scene is evaluated as the viewport shows it.
    """
    switched = []
    for obj in bpy.data.objects:
        for modifier in getattr(obj, "modifiers", ()):
            if modifier.show_viewport != modifier.show_render:
                switched.append((modifier, modifier.show_viewport))
                modifier.show_viewport = modifier.show_render
    try:
        bpy.context.view_layer.update()
        yield
    finally:
        for modifier, shown in switched:
            modifier.show_viewport = shown


def layer_names(scene) -> tuple[list[str], dict[str, str]]:
    """The scene's top-level collections in outliner order, and the top-level
    collection every collection sits in."""
    order: list[str] = []
    top: dict[str, str] = {}

    def walk(collection, root: str) -> None:
        for child in collection.children:
            top[child.name] = root
            walk(child, root)

    for child in scene.collection.children:
        order.append(child.name)
        top[child.name] = child.name
        walk(child, child.name)
    return order, top


def _layer_of(obj, top: dict[str, str]) -> str:
    for collection in obj.users_collection:
        if collection.name in top:
            return top[collection.name]
    return ""


def _properties(obj) -> dict:
    """The custom properties a person can read: numbers, text and switches."""
    found = {}
    for key in obj.keys():
        if key.startswith("_"):
            continue
        value = obj[key]
        if isinstance(value, (bool, int, float, str)):
            found[key] = value
    return found


def collect(depsgraph, texture_dir: Path) -> dict:
    """Everything the file holds, gathered from the evaluated scene."""
    scene = bpy.context.scene
    order, top = layer_names(scene)
    layers = {name: index for index, name in enumerate(order)}

    objects: list[list] = []
    merged: dict[tuple[int, int], list[dict]] = {}
    shared: dict[tuple[int, str], list[tuple[int, np.ndarray]]] = {}
    shared_triangles: dict[tuple[int, str], dict | None] = {}
    material_index: dict[str, int] = {}
    materials: list[dict] = []
    triangles_in = [0] * len(order)
    objects_in = [0] * len(order)

    def material_slot(material) -> int:
        key = material.name_full if material is not None else ""
        if key not in material_index:
            material_index[key] = len(materials)
            materials.append(describe_material(material, texture_dir))
        return material_index[key]

    def layer_for(name: str) -> int:
        if name not in layers:
            # Objects lying in the scene's own collection, outside any layer.
            layers[name] = len(order)
            order.append(name or "Scene")
            triangles_in.append(0)
            objects_in.append(0)
        return layers[name]

    def merge(object_id: int, layer: int, tris: dict | None, matrix: np.ndarray) -> None:
        if tris is None:
            return
        vertices = triangles_as_vertices(tris, matrix)
        triangles_in[layer] += len(vertices["triangles"])
        for slot, part in split_by_slot(vertices, tris["slots"]):
            material = tris["materials"][slot] if slot < len(tris["materials"]) else None
            part["object"] = object_id
            merged.setdefault((layer, material_slot(material)), []).append(part)

    for instance in depsgraph.object_instances:
        # The triangles are taken here and now: an instance's object is only
        # good for as long as the iteration stands on it.
        evaluated = instance.object
        if evaluated.type not in GEOMETRY_TYPES:
            continue
        original = evaluated.original
        owner = original
        if instance.is_instance and instance.parent is not None:
            owner = instance.parent.original
        layer_name = _layer_of(owner, top)
        if layer_name in SKIPPED_COLLECTIONS:
            continue
        matrix = np.array(instance.matrix_world, dtype=np.float64)
        layer = layer_for(layer_name)
        object_id = len(objects)
        objects.append([original.name, layer, _properties(original)])
        objects_in[layer] += 1
        # A mesh with no modifiers on it looks the same on every object using
        # it, so it can be drawn once per object from one copy.
        plain = original.type == "MESH" and not len(getattr(original, "modifiers", ()))
        if plain and np.linalg.det(matrix[:3, :3]) > 0:
            key = (layer, original.data.name_full)
            if key not in shared_triangles:
                shared_triangles[key] = object_triangles(evaluated)
            shared.setdefault(key, []).append((object_id, matrix))
        else:
            merge(object_id, layer, object_triangles(evaluated), matrix)

    groups = []
    for key, members in shared.items():
        tris = shared_triangles[key]
        layer = key[0]
        if tris is None:
            continue
        if len(members) < INSTANCE_MIN_USERS:
            for object_id, matrix in members:
                merge(object_id, layer, tris, matrix)
            continue
        vertices = triangles_as_vertices(tris, None)
        parts = []
        for slot, part in split_by_slot(vertices, tris["slots"]):
            material = tris["materials"][slot] if slot < len(tris["materials"]) else None
            parts.append((material_slot(material), part))
        triangles_in[layer] += len(vertices["triangles"]) * len(members)
        groups.append({"layer": layer, "mesh": key[1], "parts": parts, "members": members})

    # A layer with nothing drawn in it -- the physics proxies', an empty
    # collection left in an edited file -- is no layer to switch on or off.
    kept = [i for i, count in enumerate(objects_in) if count]
    renumber = {old: new for new, old in enumerate(kept)}
    return {
        "layers": [
            {"name": order[i], "objects": objects_in[i], "triangles": triangles_in[i]}
            for i in kept
        ],
        "objects": [[name, renumber[layer], props] for name, layer, props in objects],
        "materials": materials,
        "merged": {
            (renumber[layer], material): parts for (layer, material), parts in merged.items()
        },
        "instanced": [dict(group, layer=renumber[group["layer"]]) for group in groups],
    }


def rotation_quaternion(r: np.ndarray) -> np.ndarray:
    """A rotation matrix as a unit quaternion (x, y, z, w), glTF's order."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = [(r[2, 1] - r[1, 2]) * s, (r[0, 2] - r[2, 0]) * s, (r[1, 0] - r[0, 1]) * s, 0.25 / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def instance_transforms(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translation, rotation (x, y, z, w) and scale per instance, in glTF axes.

    A Blender object's matrix is translation, rotation and scale applied in
    that order, never a shear, so it comes apart into the three exactly.
    """
    convert = np.eye(4)
    convert[:3, :3] = TO_GLTF
    stack = convert @ np.asarray(matrices, dtype=np.float64).reshape(-1, 4, 4) @ np.linalg.inv(convert)
    translation = stack[:, :3, 3]
    linear = stack[:, :3, :3]
    scale = np.linalg.norm(linear, axis=1)
    rotation = np.array(
        [rotation_quaternion(m / s[None, :]) for m, s in zip(linear, scale)]
    ).reshape(-1, 4)
    return translation, rotation, scale


def assemble(data: dict) -> tuple[dict, bytes]:
    """The glTF document and its binary chunk."""
    builder = GlbBuilder()
    images: dict[str, int] = {}
    textures: list[dict] = []

    def texture_for(name: str | None) -> int | None:
        if not name:
            return None
        if name not in images:
            images[name] = len(images)
            textures.append({"source": images[name], "sampler": 0})
        return images[name]

    gltf_materials = []
    for material in data["materials"]:
        pbr = {
            "baseColorFactor": [round(c, 6) for c in material["color"]],
            "metallicFactor": round(material["metallic"], 4),
            "roughnessFactor": round(material["roughness"], 4),
        }
        entry = {"name": material["name"], "pbrMetallicRoughness": pbr, "doubleSided": True}
        colour = texture_for(material["texture"])
        if colour is not None:
            pbr["baseColorTexture"] = {"index": colour}
        normal = texture_for(material["normal_texture"])
        if normal is not None:
            entry["normalTexture"] = {"index": normal}
        if material["cutout"] and colour is not None:
            entry["alphaMode"] = "MASK"
            entry["alphaCutoff"] = 0.5
        gltf_materials.append(entry)

    def primitive(material: int, part: dict, object_ids: np.ndarray | None) -> dict:
        attributes = {
            "POSITION": builder.accessor(part["points"], FLOAT, "VEC3", target=ARRAY_BUFFER, bounds=True),
            "NORMAL": builder.accessor(part["normals"], BYTE, "VEC3", target=ARRAY_BUFFER, normalized=True),
        }
        if data["materials"][material]["texture"]:
            attributes["TEXCOORD_0"] = builder.accessor(part["uv"], FLOAT, "VEC2", target=ARRAY_BUFFER)
        if object_ids is not None:
            attributes["_OBJECT"] = builder.accessor(object_ids, FLOAT, "SCALAR", target=ARRAY_BUFFER)
        return {"attributes": attributes, "indices": builder.indices(part["triangles"]),
                "material": material, "mode": 4}

    meshes: list[dict] = []
    nodes: list[dict] = [
        {"name": layer["name"], "children": [], "extras": {"layer": index}}
        for index, layer in enumerate(data["layers"])
    ]

    for (layer, material), parts in sorted(data["merged"].items()):
        offsets = np.cumsum([0] + [len(p["points"]) for p in parts])
        combined = {
            "points": np.concatenate([p["points"] for p in parts]),
            "normals": np.concatenate([p["normals"] for p in parts]),
            "uv": np.concatenate([p["uv"] for p in parts]),
            "triangles": np.concatenate(
                [p["triangles"] + offsets[i] for i, p in enumerate(parts)]
            ),
        }
        object_ids = np.concatenate(
            [np.full(len(p["points"]), p["object"], dtype=np.float32) for p in parts]
        )
        meshes.append({
            "name": f"{data['layers'][layer]['name']} / {data['materials'][material]['name']}",
            "primitives": [primitive(material, combined, object_ids)],
        })
        nodes.append({"name": meshes[-1]["name"], "mesh": len(meshes) - 1})
        nodes[layer]["children"].append(len(nodes) - 1)

    for group in data["instanced"]:
        members = group["members"]
        translation, rotation, scale = instance_transforms([m for _, m in members])
        meshes.append({
            "name": group["mesh"],
            "primitives": [primitive(material, part, None) for material, part in group["parts"]],
        })
        attributes = {
            "TRANSLATION": builder.accessor(translation, FLOAT, "VEC3"),
            "ROTATION": builder.accessor(rotation, FLOAT, "VEC4"),
            "SCALE": builder.accessor(scale, FLOAT, "VEC3"),
            "_OBJECT": builder.accessor(
                np.array([object_id for object_id, _ in members], dtype=np.float32), FLOAT, "SCALAR"
            ),
        }
        nodes.append({
            "name": group["mesh"],
            "mesh": len(meshes) - 1,
            "extensions": {"EXT_mesh_gpu_instancing": {"attributes": attributes}},
        })
        nodes[group["layer"]]["children"].append(len(nodes) - 1)

    for node in nodes[: len(data["layers"])]:
        if not node["children"]:
            del node["children"]

    binary = builder.payload()
    document = {
        "asset": {"version": "2.0", "generator": f"TerreinLab web_export {FORMAT}"},
        "extensionsUsed": ["KHR_mesh_quantization", "EXT_mesh_gpu_instancing"],
        "extensionsRequired": ["KHR_mesh_quantization"],
        "scene": 0,
        "scenes": [{"nodes": list(range(len(data["layers"])))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": gltf_materials,
        "accessors": builder.accessors,
        "bufferViews": builder.views,
        "buffers": [{"byteLength": len(binary)}],
        "extras": {
            "terreinlab": {
                "format": FORMAT,
                "layers": data["layers"],
                # [name, layer, properties], indexed by the _OBJECT values.
                "objects": data["objects"],
            }
        },
    }
    if images:
        document["images"] = [
            {"uri": urllib.parse.quote(name)} for name in sorted(images, key=images.get)
        ]
        document["textures"] = textures
        # Linear with mipmaps, repeating, as every material in the model is.
        document["samplers"] = [
            {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}
        ]
    if not binary:
        del document["buffers"]
    # glTF allows no empty list anywhere: an empty scene leaves none behind.
    for key in [k for k, v in document.items() if isinstance(v, list) and not v]:
        del document[key]
    if not document["scenes"][0]["nodes"]:
        del document["scenes"][0]["nodes"]
    return document, binary


def write(out_path: Path, texture_dir: Path, source: Path | None = None,
          source_stat: os.stat_result | None = None) -> dict:
    """Write the scene to `out_path`, and the stamp beside it.

    `source` is the .blend the scene is, or was saved as; the stamp records
    its size and time so the UI can tell a view of an older save. Pass its
    stat from before it was opened if it was opened for this.
    """
    started = time.time()
    out_path = Path(out_path)
    if source is not None and source_stat is None:
        source_stat = Path(source).stat()

    with render_settings():
        data = collect(bpy.context.evaluated_depsgraph_get(), Path(texture_dir))
    document, binary = assemble(data)
    blob = pack_glb(document, binary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and moved into place, so the page never reads half a file.
    partial = out_path.with_name(out_path.name + ".part")
    partial.write_bytes(blob)
    os.replace(partial, out_path)

    stats = {
        "format": FORMAT,
        "file": out_path.name,
        "bytes": len(blob),
        "layers": len(data["layers"]),
        "objects": len(data["objects"]),
        "triangles": int(sum(layer["triangles"] for layer in data["layers"])),
        "instanced_meshes": len(data["instanced"]),
        "seconds": round(time.time() - started, 2),
        "blender": bpy.app.version_string,
    }
    stamp = dict(stats)
    if source is not None and source_stat is not None:
        stamp.update(
            source=Path(source).name,
            source_size=int(source_stat.st_size),
            source_mtime_ns=int(source_stat.st_mtime_ns),
        )
    out_path.with_name(STAMP_NAME).write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    log(
        f"wrote {out_path} ({len(blob) / 1e6:.1f} MB): {stats['triangles']} triangles, "
        f"{stats['objects']} objects in {stats['layers']} layers, {stats['seconds']} s"
    )
    return stats


# ---------------------------------------------------------------------------
# As a script
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]):
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:] if argv and argv[0].endswith(".py") else argv

    import argparse

    parser = argparse.ArgumentParser(description="Write a model.blend for the 3D view.")
    parser.add_argument("--blend", required=True, help="the model.blend to show")
    parser.add_argument("--out", required=True, help="the .glb to write")
    parser.add_argument(
        "--textures", default=None,
        help="where the model's textures are; defaults to the folder of the .blend",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(list(sys.argv))
    blend = Path(args.blend).resolve()
    if not blend.is_file():
        raise SystemExit(f"no such file: {blend}")
    stat = blend.stat()
    log(f"Blender {bpy.app.version_string}, opening {blend}")
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False)
    write(
        Path(args.out).resolve(),
        Path(args.textures).resolve() if args.textures else blend.parent,
        source=blend,
        source_stat=stat,
    )
    return 0


def _run() -> int:
    """Main, with a failure said on stderr in a way the caller can find.

    Blender in background mode exits 0 when its script raises, so the
    caller checks the file and the stamp, and quotes this.
    """
    import traceback

    try:
        return main()
    except SystemExit:
        raise
    except BaseException:
        print("[viewer] FAILED\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
