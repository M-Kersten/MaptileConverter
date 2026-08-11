#!/usr/bin/env python3
"""Orchestrator: bbox in, textured FBX plus aerial plus metadata out.

    python pipeline.py --config config.json

The Python stages fetch and prepare everything into ``work/<area>/``, then the
Blender stage builds the scene and exports:

    blender --background --python blender/process.py -- --work work/<area> --out output/<area>

Blender is found in this order: an explicit ``--blender`` path, then a
``blender`` executable on PATH, then the pip ``bpy`` module driven through this
interpreter. Any of the three produces the same output.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.buildings import build_buildings  # noqa: E402
from src.config import PipelineConfig, load_config  # noqa: E402
from src.elevation import build_terrain  # noqa: E402
from src.export import (  # noqa: E402
    build_metadata,
    write_attribution,
    write_metadata,
    write_scene_description,
)
from src.http_util import preflight  # noqa: E402
from src.sources import BY_ID, enabled_sources, health_targets  # noqa: E402
from src.facade import (  # noqa: E402
    CAR_PAINT,
    generate_facade_textures,
    generate_furniture_texture,
    generate_rail_texture,
    generate_tree_texture,
    generate_vehicle_texture,
    generate_water_texture,
)
from src.furniture import FurnitureSet, build_furniture  # noqa: E402
from src.imagery import build_aerial  # noqa: E402
from src.rails import KIND_NAMES as RAIL_KIND_NAMES  # noqa: E402
from src.rails import RailSet, build_rails, save_rails  # noqa: E402
from src.surfaces import (  # noqa: E402
    CLASS_NAMES,
    SurfaceSet,
    blend_surface_detail,
    build_surfaces,
    write_land_cover,
)
from src.trees import TreeSet, build_trees, write_tree_list  # noqa: E402
from src.usage import UsageSet, fetch_usage  # noqa: E402
from src.vehicles import (  # noqa: E402
    VehicleSet,
    build_vehicles,
    save_vehicles,
    write_spawn_list,
)
from src.validate import (  # noqa: E402
    CheckReport,
    check_aerial,
    check_bbox,
    check_buildings,
    check_buildings_on_terrain,
    check_export,
    check_fbx_reimport,
    check_furniture,
    check_rails,
    check_vehicles,
    check_surfaces,
    check_terrain,
    check_trees,
    write_report,
)

LOG = logging.getLogger("pipeline")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Filled in as the run goes, and written out at the end. The UI reads these
# back to calibrate its estimates against the machine it is actually running on.
STAGE_TIMINGS: list[dict] = []


class Stage:
    """Times a stage and labels its log output."""

    def __init__(self, name: str, index: int, total: int) -> None:
        self.name = name
        self.index = index
        self.total = total

    def __enter__(self) -> "Stage":
        LOG.info("=" * 72)
        LOG.info("Step %d/%d  %s", self.index, self.total, self.name)
        LOG.info("=" * 72)
        self.started = time.time()
        return self

    def __exit__(self, *exc_info) -> None:
        elapsed = time.time() - self.started
        LOG.info("  %s finished in %.1fs", self.name, elapsed)
        STAGE_TIMINGS.append(
            {
                "step": self.index,
                "name": self.name,
                "seconds": round(elapsed, 2),
                "failed": exc_info[0] is not None,
            }
        )


def write_timings(
    config: PipelineConfig,
    work_dir: Path,
    total_seconds: float,
    args: argparse.Namespace | None = None,
    completed: bool = False,
) -> None:
    """Record what the run cost, alongside the settings that drove it."""
    payload = {
        "name": config.name,
        "total_seconds": round(total_seconds, 2),
        "stages": STAGE_TIMINGS,
        "drivers": {
            # What the estimate scales on: area and aerial pixel count.
            "area_km2": round(config.bbox.width * config.bbox.height / 1e6, 5),
            "aerial_size_px": int(config.aerial["size_px"]),
            "aerial_megapixels": round(int(config.aerial["size_px"]) ** 2 / 1e6, 3),
            "mesh_vertices_per_side": int(config.terrain["mesh_vertices_per_side"]),
            "trees": bool(config.trees["enabled"]),
            "surfaces": bool(config.surfaces["water"] or config.surfaces["land_cover"]),
            "furniture": bool(config.furniture["enabled"]),
            "usage": bool(config.usage["enabled"]),
            # Rendering previews costs more than every data stage put together,
            # and it happens outside the timed stages, so calibration needs to
            # know whether it ran.
            "preview": bool(args.preview) if args is not None else False,
            "skip_blender": bool(args.skip_blender) if args is not None else False,
        },
        # A run that stopped at the preflight took seconds and did no work.
        # Calibrating the estimate on it would drag every prediction down.
        "completed": completed,
    }
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "timings.json").write_text(
            json.dumps(payload, indent=1) + "\n", encoding="utf-8"
        )
    except OSError as exc:  # noqa: BLE001 - timings are a convenience
        LOG.debug("could not write timings: %s", exc)


def find_blender(explicit: str | None) -> tuple[str, list[str]]:
    """Work out how to run the Blender stage.

    Returns the mode ("executable" or "bpy") and the command prefix.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file() and shutil.which(explicit) is None:
            raise SystemExit(f"--blender {explicit!r} is not an executable")
        return "executable", [explicit]

    found = shutil.which("blender")
    if found:
        return "executable", [found]

    try:
        import bpy  # noqa: F401
    except ImportError:
        raise SystemExit(
            "no Blender available. Either install the pip module "
            "(`pip install bpy`, needs CPython 3.11), or pass "
            "--blender /path/to/blender"
        ) from None
    return "bpy", [sys.executable]


def run_blender_script(
    script_name: str, script_args: list[str], blender: str | None
) -> subprocess.CompletedProcess:
    """Run a script in blender/, through whichever Blender is available."""
    script = REPO_ROOT / "blender" / script_name
    mode, prefix = find_blender(blender)

    if mode == "executable":
        command = prefix + [
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            *script_args,
        ]
    else:
        # The pip module runs the same script directly. A subprocess keeps
        # Blender's global state out of this process.
        command = prefix + [str(script), *script_args]

    LOG.info("running %s via %s", script_name, mode)
    LOG.debug("command: %s", " ".join(command))
    result = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True)

    for line in result.stdout.splitlines():
        if line.strip():
            LOG.info("  %s", line.rstrip())

    # Blender reports success even after the script it was given raises, so a
    # zero exit code is not proof. Flag it here; whoever called this decides
    # whether the run is salvageable and prints the detail.
    stderr = result.stderr or ""
    if any(
        marker in stderr for marker in ("Traceback", "[blender] FAILED")
    ):
        last = [line for line in stderr.splitlines() if line.strip()]
        LOG.warning(
            "%s raised inside Blender: %s", script_name, last[-1] if last else "?"
        )

    if result.returncode != 0:
        for line in stderr.splitlines()[-40:]:
            LOG.error("  %s", line.rstrip())
        raise SystemExit(f"{script_name} failed with exit code {result.returncode}")
    return result


def run_blender_stage(
    work_dir: Path, out_dir: Path, blender: str | None, fbx_name: str
) -> None:
    """Run the scene build, and verify it actually produced the FBX.

    Blender in background mode exits 0 even when the script it was given raises,
    so the exit code alone does not prove anything. The file it was supposed to
    write is the real test.
    """
    error_file = work_dir / "blender_error.txt"
    if error_file.exists():
        error_file.unlink()

    result = run_blender_script(
        "process.py",
        ["--work", str(work_dir), "--out", str(out_dir)],
        blender,
    )

    if (out_dir / fbx_name).is_file():
        return

    LOG.error("Blender exited %s but wrote no %s", result.returncode, fbx_name)
    if error_file.is_file():
        LOG.error("Blender reported:")
        for line in error_file.read_text(encoding="utf-8").splitlines():
            LOG.error("  %s", line)
    else:
        # No traceback file means it died before it could write one, so fall
        # back to whatever the process printed.
        for stream, label in ((result.stderr, "stderr"), (result.stdout, "stdout")):
            tail = [line for line in (stream or "").splitlines() if line.strip()][-30:]
            if tail:
                LOG.error("Blender %s tail:", label)
                for line in tail:
                    LOG.error("  %s", line)

    raise SystemExit(
        f"the Blender stage did not produce {fbx_name}. The traceback above is "
        f"from Blender itself; {error_file} has the full copy."
    )


def run(config: PipelineConfig, args: argparse.Namespace) -> int:
    work_dir = config.work_dir(REPO_ROOT)
    out_dir = config.out_dir(REPO_ROOT)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("area %r", config.name)
    LOG.info("  bbox RD      %s", config.bbox)
    LOG.info("  origin RD    %.2f, %.2f", *config.geo.origin)
    LOG.info("  work dir     %s", work_dir)
    LOG.info("  output dir   %s", out_dir)
    for warning in config.warnings:
        LOG.warning("  %s", warning)

    active = enabled_sources(config.as_dict())
    LOG.info("sources: %s", ", ".join(source.label for source in active))

    if not args.skip_preflight:
        # Cheaper to ask now than to find out at stage five. Only the sources
        # this run actually uses are checked, and only once per host.
        LOG.info("checking the services this run needs")
        targets = health_targets(config.as_dict(), active)
        labels = {
            url: ", ".join(
                BY_ID[i].label for i in ids.split(",") if i in BY_ID
            )
            for url, ids in targets.items()
        }
        down = preflight({labels[url]: url for url in targets})
        if down:
            raise SystemExit(
                "cannot start: "
                + "; ".join(down)
                + ". These are national open-data services, so this is normally "
                "an outage at their end. Check https://www.pdok.nl and "
                "https://3dbag.nl, and try again later. Turn the affected "
                "source off if it is optional, or use --skip-preflight to "
                "attempt it anyway."
            )

    want_trees = bool(config.trees["enabled"])
    want_surfaces = bool(config.surfaces["water"] or config.surfaces["land_cover"])
    want_furniture = bool(config.furniture["enabled"])
    want_usage = bool(config.usage["enabled"])
    want_vehicles = bool(config.vehicles["cars"] or config.vehicles["boats"])
    want_rails = bool(config.rails["enabled"])

    total = 5 + sum(
        (
            want_trees, want_surfaces, want_furniture, want_usage,
            want_vehicles, want_rails,
        )
    ) + (0 if args.skip_blender else 1)
    step = 0

    def next_step() -> int:
        nonlocal step
        step += 1
        return step

    with Stage("terrain (AHN DTM)", next_step(), total):
        terrain = build_terrain(config.bbox, work_dir, terrain_cfg=config.terrain)

    with Stage("aerial imagery (PDOK)", next_step(), total):
        aerial = build_aerial(config.bbox, work_dir, aerial_cfg=config.aerial)

    surfaces = SurfaceSet()
    water_texture = None
    if want_surfaces:
        with Stage("ground surfaces (BGT water and land cover)", next_step(), total):
            surfaces = build_surfaces(
                config.bbox,
                work_dir,
                surfaces_cfg=config.surfaces,
                terrain=terrain,
            )
            if surfaces.class_grid is not None:
                write_land_cover(surfaces, config.bbox, out_dir)
                blend_surface_detail(
                    aerial.path,
                    surfaces,
                    config.bbox,
                    strength=float(config.surfaces["detail_strength"]),
                )
            if surfaces.water:
                water_texture = generate_water_texture(work_dir)

    usage = UsageSet()
    if want_usage:
        with Stage("building function (BAG)", next_step(), total):
            usage = fetch_usage(config.bbox, usage_cfg=config.usage)

    with Stage("buildings (3DBAG LoD2.2)", next_step(), total):
        buildings = build_buildings(
            config.bbox,
            work_dir,
            buildings_cfg=config.buildings,
            terrain_sampler=terrain.sample,
            facade_variants=int(config.facade["variants"]),
            ground_floor_height_m=(
                float(config.facade["ground_floor_height_m"])
                if config.facade["ground_floor"]
                else None
            ),
            usage=usage if len(usage) else None,
            facade_cfg=config.facade,
        )

    trees = TreeSet()
    tree_texture = None
    if want_trees:
        with Stage("trees (BGT points, AHN heights)", next_step(), total):
            trees = build_trees(
                config.bbox,
                work_dir,
                trees_cfg=config.trees,
                terrain=terrain,
                buildings=buildings,
            )
            if len(trees):
                write_tree_list(trees, config.geo, out_dir)
                if bool(config.trees["geometry"]):
                    tree_texture = generate_tree_texture(
                        work_dir, size_px=int(config.trees["texture_px"])
                    )

    furniture = FurnitureSet()
    furniture_texture = None
    if want_furniture:
        with Stage("street furniture (BGT)", next_step(), total):
            furniture = build_furniture(
                config.bbox,
                work_dir,
                furniture_cfg=config.furniture,
                terrain=terrain,
            )
            if len(furniture):
                furniture_texture = generate_furniture_texture(work_dir)

    vehicles = VehicleSet()
    vehicle_texture = None
    if want_vehicles:
        with Stage("cars and boats (BGT)", next_step(), total):
            vehicles = build_vehicles(
                config.bbox,
                work_dir,
                vehicles_cfg=config.vehicles,
                terrain_sampler=terrain.sample,
                surfaces=surfaces if want_surfaces else None,
            )
            if len(vehicles):
                save_vehicles(vehicles, work_dir / "vehicles.npz")
                write_spawn_list(vehicles, config.geo, out_dir)
                vehicle_texture = generate_vehicle_texture(work_dir)

    rails = RailSet()
    rail_texture = None
    if want_rails:
        with Stage("railways (BGT)", next_step(), total):
            rails = build_rails(
                config.bbox, work_dir, rails_cfg=config.rails, terrain=terrain
            )
            if len(rails.ballast_tris) or len(rails.rail_tris):
                save_rails(rails, work_dir / "rails.npz")
                rail_texture = generate_rail_texture(work_dir)

    with Stage("facade textures", next_step(), total):
        facade_cfg = dict(config.facade)
        facade_cfg["ground_by_function"] = bool(len(usage))
        facade_paths = generate_facade_textures(work_dir, facade_cfg=facade_cfg)
        ground_variants = (
            (2 if len(usage) else 1) if config.facade["ground_floor"] else 0
        )

    with Stage("scene description", next_step(), total):
        write_scene_description(
            name=config.name,
            geo=config.geo,
            terrain=terrain,
            aerial=aerial,
            facade_paths=facade_paths,
            facade_cfg=config.facade,
            buildings_cfg=config.buildings,
            export_cfg=config.export,
            work_dir=work_dir,
            tree_texture=tree_texture,
            water_texture=water_texture,
            furniture_texture=furniture_texture,
            surfaces_cfg=config.surfaces,
            road_class_names={str(k): v for k, v in CLASS_NAMES.items()},
            furniture_cfg=config.furniture,
            vehicle_texture=vehicle_texture,
            vehicles_cfg=config.vehicles,
            car_colours=len(CAR_PAINT),
            rail_texture=rail_texture,
            rail_kind_names={str(k): v for k, v in RAIL_KIND_NAMES.items()},
            ground_variants=ground_variants,
        )

    if not args.skip_blender:
        with Stage("Blender scene build and FBX export", next_step(), total):
            run_blender_stage(
                work_dir, out_dir, args.blender, str(config.export["fbx_name"])
            )

    with Stage("metadata and validation", total, total):
        metadata = build_metadata(
            name=config.name,
            geo=config.geo,
            terrain=terrain,
            aerial=aerial,
            buildings=buildings,
            facade_cfg=config.facade,
            lod=str(config.buildings["lod"]),
            ahn_model=str(config.terrain["ahn_model"]),
            trees=trees,
            extra={
                "surfaces": surfaces.stats(),
                "street_furniture": furniture.stats(),
                "vehicles": vehicles.stats(),
                "railways": rails.stats(),
                "building_function": usage.stats(),
            },
        )
        write_metadata(metadata, out_dir, str(config.export["metadata_name"]))
        write_attribution(out_dir)

        report = CheckReport()
        check_bbox(report, config.bbox)
        check_terrain(report, terrain, config.bbox)
        check_buildings(
            report, buildings, max_height_m=float(config.buildings["max_height_m"])
        )
        check_buildings_on_terrain(report, buildings, terrain)
        check_trees(report, trees if want_trees else None, config.bbox)
        check_surfaces(report, surfaces if want_surfaces else None, terrain)
        check_furniture(report, furniture if want_furniture else None, config.bbox)
        check_vehicles(
            report, vehicles if want_vehicles else None, config.bbox, surfaces
        )
        check_rails(report, rails if want_rails else None, terrain)
        check_aerial(report, aerial, config.bbox)

        if not args.skip_blender:
            check_export(
                report,
                out_dir,
                work_dir=work_dir,
                bbox=config.bbox,
                fbx_name=str(config.export["fbx_name"]),
                aerial_name=str(config.export["aerial_name"]),
                metadata_name=str(config.export["metadata_name"]),
            )
            if not args.skip_reimport_check:
                check_fbx_reimport(
                    report, out_dir / str(config.export["fbx_name"]), config.bbox
                )

        write_report(report, work_dir / "validation.json")

    if args.preview and not args.skip_blender:
        # The numbers cannot judge whether the facades and the aerial drape look
        # right, so render the views that can.
        preview_dir = out_dir / "preview"
        run_blender_script(
            "preview.py",
            [
                "--fbx",
                str(out_dir / str(config.export["fbx_name"])),
                "--out",
                str(preview_dir),
            ],
            args.blender,
        )

    LOG.info("=" * 72)
    LOG.info("%s", report.summary())
    for check in report.failures:
        LOG.error("%s", check)
    for check in report.warnings:
        LOG.warning("%s", check)

    if report.ok:
        LOG.info("output written to %s", out_dir)
        for path in sorted(out_dir.iterdir()):
            LOG.info("  %s (%.2f MB)", path.name, path.stat().st_size / 1e6)
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a textured 3D model of a Dutch bbox for Unity.",
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config.json"), help="path to config.json"
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Blender executable; defaults to one on PATH, else the pip bpy module",
    )
    parser.add_argument(
        "--skip-blender",
        action="store_true",
        help="run the data stages only and stop before the scene build",
    )
    parser.add_argument(
        "--skip-reimport-check",
        action="store_true",
        help="skip re-importing the exported FBX to re-measure it",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="also render top-down, oblique and street views of the result",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="do not check the data services are reachable before starting",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    try:
        config = load_config(args.config)
    except (ValueError, FileNotFoundError) as exc:
        LOG.error("%s", exc)
        return 2

    started = time.time()
    completed = False
    try:
        code = run(config, args)
        completed = True
    except KeyboardInterrupt:
        LOG.error("interrupted")
        return 130
    finally:
        # Written even on failure: a partial run still says what the stages
        # that did complete actually cost on this machine. Only a run that
        # reached the end is used to calibrate the estimate.
        write_timings(
            config,
            config.work_dir(REPO_ROOT),
            time.time() - started,
            args,
            completed=completed,
        )
    LOG.info("total runtime %.1fs", time.time() - started)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
