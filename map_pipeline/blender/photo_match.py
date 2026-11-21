"""Render the model from where a street photo was taken.

Run by Blender, as ``blender -b model.blend --python photo_match.py -- --views
views.json --out folder``, or with the bpy module. `views.json` lists cameras
already in the model's local frame and Blender's convention, worked out by
the pipeline (see `src/photo_match.py`): this script only places them and
renders, so none of the maths lives where it cannot be tested.

Cycles on the CPU with a sky and a sun: slow next to Workbench, but it
needs no GPU or OpenGL, which a headless machine may not have.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import mathutils


def parse_args(argv):
    import argparse

    # Blender passes the script's own arguments after "--"; the bpy module,
    # run as a plain script, passes them straight.
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:] if argv and argv[0].endswith(".py") else argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--blend", default="")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(list(sys.argv))
    if args.blend:
        bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()))
    views = json.loads(Path(args.views).read_text(encoding="utf-8"))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    # Cycles on the CPU: Workbench and EEVEE need an OpenGL context, which a
    # headless machine without a GPU does not have.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 2
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("MatchWorld")
    if bpy.app.version < (5, 0, 0):
        world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    # Neutral light, so a wall's colour can be judged against the photo's:
    # a blue sky tints every wall in shade blue.
    background.inputs[0].default_value = (0.78, 0.78, 0.78, 1.0)
    background.inputs[1].default_value = 0.9
    scene.world = world
    sun = bpy.data.objects.new("MatchSun", bpy.data.lights.new("MatchSun", type="SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), 0.0, math.radians(200))
    scene.collection.objects.link(sun)

    for view in views:
        data = bpy.data.cameras.new(view["name"])
        data.sensor_fit = "HORIZONTAL"
        data.angle = math.radians(float(view["hfov_deg"]))
        data.clip_start = 0.2
        data.clip_end = 5000.0
        camera = bpy.data.objects.new(view["name"], data)
        rotation = mathutils.Matrix(view["rotation"])
        camera.matrix_world = mathutils.Matrix.Translation(view["location"]) @ rotation.to_4x4()
        scene.collection.objects.link(camera)
        scene.camera = camera
        scene.render.resolution_x = int(view["width"])
        scene.render.resolution_y = int(view["height"])
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(out / f"{view['name']}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[photo_match] wrote {scene.render.filepath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
