"""Render preview images of an exported FBX.

The numeric checks catch coordinate and scale mistakes. What they cannot judge
is whether the facades and the aerial drape actually look right, which is the
one check that still needs eyes. This renders a top-down and a street-level view
so that judgement can be made without opening Unity.

    python blender/preview.py --fbx output/<area>/model.fbx --out preview/
    blender --background --python blender/preview.py -- --fbx ... --out ...
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
import mathutils


def parse_args(argv: list[str]):
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:] if argv and argv[0].endswith(".py") else argv

    import argparse

    parser = argparse.ArgumentParser(description="Render previews of an FBX.")
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument(
        "--engine",
        default="auto",
        choices=("auto", "workbench", "eevee", "cycles"),
        help="auto picks workbench, which needs no GPU and still shows textures",
    )
    return parser.parse_args(argv)


def scene_bounds():
    corners = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ mathutils.Vector(corner))
    if not corners:
        raise SystemExit("scene has no meshes")
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def setup_render(engine: str, size: int) -> None:
    scene = bpy.context.scene
    available = {item.identifier for item in
                 scene.bl_rna.properties["render"].fixed_type.bl_rna
                 .properties["engine"].enum_items}

    if engine == "auto":
        # Workbench renders textured without a GPU or any light setup, which is
        # what a headless container can rely on.
        chosen = "BLENDER_WORKBENCH"
    elif engine == "workbench":
        chosen = "BLENDER_WORKBENCH"
    elif engine == "cycles":
        chosen = "CYCLES"
    else:
        chosen = next(
            (e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if e in available),
            "BLENDER_WORKBENCH",
        )

    scene.render.engine = chosen if chosen in available else "BLENDER_WORKBENCH"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    if scene.render.engine == "BLENDER_WORKBENCH":
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "TEXTURE"
        shading.show_shadows = True
        shading.show_cavity = True
    elif scene.render.engine == "CYCLES":
        scene.cycles.samples = 32

    world = bpy.data.worlds.new("PreviewWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.68, 0.85, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.4
    scene.world = world


def add_sun() -> None:
    light = bpy.data.lights.new("Sun", type="SUN")
    light.energy = 4.0
    obj = bpy.data.objects.new("Sun", light)
    obj.rotation_euler = (math.radians(50), 0, math.radians(215))
    bpy.context.scene.collection.objects.link(obj)


def place_camera(name: str, location, target, *, ortho_scale: float | None, lens: float):
    camera_data = bpy.data.cameras.new(name)
    if ortho_scale is not None:
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = ortho_scale
    else:
        camera_data.lens = lens
    camera_data.clip_end = 20000.0

    camera = bpy.data.objects.new(name, camera_data)
    camera.location = location
    direction = mathutils.Vector(target) - mathutils.Vector(location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(camera)
    return camera


def render_to(camera, path: Path) -> None:
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print(f"[preview] wrote {path}", flush=True)


def main() -> int:
    args = parse_args(list(sys.argv))
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(Path(args.fbx).resolve()))

    (min_x, min_y, min_z), (max_x, max_y, max_z) = scene_bounds()
    span = max(max_x - min_x, max_y - min_y)
    center = ((min_x + max_x) / 2, (min_y + max_y) / 2, min_z)
    print(
        f"[preview] bounds X [{min_x:.1f},{max_x:.1f}] "
        f"Y [{min_y:.1f},{max_y:.1f}] Z [{min_z:.1f},{max_z:.1f}]",
        flush=True,
    )

    setup_render(args.engine, args.size)
    add_sun()

    # Straight down: checks that the aerial drape lines up with the buildings.
    top = place_camera(
        "TopDown",
        (center[0], center[1], max_z + span),
        center,
        ortho_scale=span * 1.02,
        lens=50,
    )
    render_to(top, out_dir / "preview_top.png")

    # Low oblique: this is the view that shows facades and roof texture.
    oblique = place_camera(
        "Oblique",
        (center[0] - span * 0.42, center[1] - span * 0.52, min_z + span * 0.30),
        (center[0], center[1], min_z + (max_z - min_z) * 0.25),
        ortho_scale=None,
        lens=45,
    )
    render_to(oblique, out_dir / "preview_oblique.png")

    # Street level: close enough to judge window scale against storey height.
    street = place_camera(
        "Street",
        (center[0] - span * 0.10, center[1] - span * 0.14, min_z + 18.0),
        (center[0] + span * 0.05, center[1] + span * 0.10, min_z + 12.0),
        ortho_scale=None,
        lens=35,
    )
    render_to(street, out_dir / "preview_street.png")

    print("[preview] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
