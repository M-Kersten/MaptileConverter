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
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.bgt import use_cache as bgt_cache  # noqa: E402
from src.breaklines import fetch_outline_rings  # noqa: E402
from src.buildings import build_buildings  # noqa: E402
from src.config import PipelineConfig, load_config  # noqa: E402
from src.elevation import build_terrain, ensure_dsm, raster_sampler  # noqa: E402
from src.export import (  # noqa: E402
    build_metadata,
    write_attribution,
    write_metadata,
    write_scene_description,
)
from src.http_util import host_of, probe_targets  # noqa: E402
from src.sources import (  # noqa: E402
    BY_ID,
    down_sources,
    enabled_sources,
    health_targets,
)
from src.facade import (  # noqa: E402
    CAR_PAINT,
    generate_facade_textures,
    generate_furniture_texture,
    generate_rail_texture,
    generate_structure_texture,
    generate_tree_texture,
    generate_vehicle_texture,
    generate_water_texture,
)
from src.furniture import FurnitureSet, build_furniture  # noqa: E402
from src.imagery import build_aerial  # noqa: E402
from src.rails import KIND_NAMES as RAIL_KIND_NAMES  # noqa: E402
from src.rails import RailSet, build_rails, save_rails  # noqa: E402
from src.structures import KIND_NAMES as STRUCTURE_KIND_NAMES  # noqa: E402
from src.structures import (  # noqa: E402
    StructureSet,
    build_structures,
    save_structures,
)
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
    check_structures,
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


# Where Blender installs itself when nobody puts it on PATH, which is the
# normal case on macOS and Windows. A macOS .app is a directory, so the
# executable inside it can never be found by a PATH lookup however the user
# installed it -- "I have Blender installed" and "shutil.which finds blender"
# are simply different statements there.
BLENDER_GLOBS = (
    # macOS, including the versioned bundle names the installer leaves behind.
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/Applications/Blender*.app/Contents/MacOS/Blender",
    "~/Applications/Blender.app/Contents/MacOS/Blender",
    "~/Applications/Blender*.app/Contents/MacOS/Blender",
    # Windows.
    "C:/Program Files/Blender Foundation/Blender*/blender.exe",
    "C:/Program Files (x86)/Steam/steamapps/common/Blender/blender.exe",
    # Linux, for the packages that do not link into /usr/bin.
    "/usr/share/blender/blender",
    "/snap/bin/blender",
    "/var/lib/flatpak/exports/bin/org.blender.Blender",
)


def blender_in_bundle(path: Path) -> Path | None:
    """The executable inside a macOS .app, given the bundle.

    Worth handling because ``--blender /Applications/Blender.app`` is the
    obvious thing to pass on a Mac and is a directory, not a program.
    """
    if path.suffix == ".app" and path.is_dir():
        inner = path / "Contents" / "MacOS" / "Blender"
        return inner if inner.is_file() else None
    return None


def search_for_blender() -> list[str]:
    """Every Blender this machine appears to have, best first."""
    import glob as _glob

    found: list[str] = []
    for name in ("blender", "Blender", "blender.exe"):
        hit = shutil.which(name)
        if hit and hit not in found:
            found.append(hit)
    for pattern in BLENDER_GLOBS:
        for hit in sorted(_glob.glob(str(Path(pattern).expanduser())), reverse=True):
            if os.access(hit, os.X_OK) and hit not in found:
                found.append(hit)
    return found


def find_blender(explicit: str | None) -> tuple[str, list[str]]:
    """Work out how to run the Blender stage.

    Returns the mode ("executable" or "bpy") and the command prefix.
    """
    if explicit:
        path = Path(explicit).expanduser()
        inner = blender_in_bundle(path)
        if inner is not None:
            return "executable", [str(inner)]
        if not path.is_file() and shutil.which(explicit) is None:
            raise SystemExit(f"--blender {explicit!r} is not an executable")
        return "executable", [str(path) if path.is_file() else explicit]

    found = search_for_blender()
    if found:
        LOG.info("using Blender at %s", found[0])
        return "executable", [found[0]]

    try:
        import bpy  # noqa: F401
    except ImportError:
        looked = "\n  ".join(
            [shutil.which("blender") or "PATH (blender, Blender, blender.exe)"]
            + [str(Path(p).expanduser()) for p in BLENDER_GLOBS]
        )
        raise SystemExit(
            "no Blender available. Looked in:\n  "
            + looked
            + "\n\nIf Blender is installed somewhere else, pass it directly -- "
            "on a Mac the app bundle itself is fine:\n"
            "  python pipeline.py --config config.json "
            "--blender /Applications/Blender.app\n"
            "Otherwise install the pip module with `pip install bpy`, which "
            "needs CPython 3.11."
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
        settings = config.as_dict()
        targets = health_targets(settings, active)
        backups = {
            url
            for source in active
            for url in source.probes(settings)[1:]
        }
        labels = {
            url: ", ".join(BY_ID[i].label for i in ids.split(",") if i in BY_ID)
            + (" [backup]" if url in backups else "")
            for url, ids in targets.items()
        }
        results = probe_targets(labels)
        reachable = {url: ok for url, (ok, _) in results.items()}
        # A source with a second service behind it is not down until both are.
        # Aborting on the first would stop runs that the fallback would have
        # finished, which is most of what 3DBAG's API outages used to cost.
        blocked = down_sources(settings, reachable, active)
        for source in active:
            urls = source.probes(settings)
            if (
                len(urls) > 1
                and source not in blocked
                and not reachable.get(urls[0], False)
            ):
                LOG.warning(
                    "%s: %s is not answering, so this run will use %s instead",
                    source.label,
                    host_of(urls[0]),
                    host_of(urls[1]),
                )
                # Already established, so do not make the stage rediscover it.
                # The stage keeps its own probe for --skip-preflight runs.
                if source.id == "buildings":
                    config.buildings["sources"] = [
                        s for s in config.buildings["sources"] if s != "api"
                    ] or ["tiles"]
        if blocked:
            raise SystemExit(
                "cannot start: "
                + "; ".join(f"{s.label} has no service answering" for s in blocked)
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
    want_structures = bool(
        config.structures["bridges"] or config.structures["tunnels"]
    )

    total = 5 + sum(
        (
            want_trees, want_surfaces, want_furniture, want_usage,
            want_vehicles, want_rails, want_structures,
        )
    ) + (0 if args.skip_blender else 1)
    step = 0

    def next_step() -> int:
        nonlocal step
        step += 1
        return step

    with Stage("terrain (AHN DTM)", next_step(), total):
        # Cache BGT responses under this run. The terrain mesh reads outlines
        # here to fold along, and the surfaces stage reads the same collections
        # again a few steps later; without the cache that is paid for twice.
        bgt_cache(work_dir)
        breakline_rings = None
        wanted_lines = list(config.terrain.get("breaklines") or [])
        if wanted_lines:
            breakline_rings = fetch_outline_rings(
                config.bbox,
                sources=wanted_lines,
                page_limit=int(config.surfaces["page_limit"]),
                timeout=float(config.surfaces["timeout_s"]),
                max_retries=int(config.surfaces["max_retries"]),
                max_pages=int(config.surfaces["max_pages"]),
            )
        terrain = build_terrain(
            config.bbox,
            work_dir,
            terrain_cfg=config.terrain,
            breakline_rings=breakline_rings,
        )
        if terrain.constrained is not None:
            # Roads float just above the ground so the two do not fight for
            # depth. The mesh can stray from the grid the roads are draped on,
            # so the lift has to clear that or the terrain pokes through.
            clearance = round(min(terrain.constrained.max_error_m, 0.2) + 0.02, 3)
            if clearance > float(config.surfaces["road_lift_m"]):
                LOG.info(
                    "raising surfaces.road_lift_m to %.3f m to clear the "
                    "terrain mesh",
                    clearance,
                )
                config.surfaces["road_lift_m"] = clearance

    with Stage("aerial imagery (PDOK)", next_step(), total):
        aerial = build_aerial(config.bbox, work_dir, aerial_cfg=config.aerial)

    surfaces = SurfaceSet()
    water_texture = None
    if want_surfaces:
        with Stage("ground surfaces (BGT water and land cover)", next_step(), total):
            # A road on a bridge takes its height from the surface model, so
            # that has to be here before the road geometry is built.
            deck_dsm = None
            if bool(config.surfaces.get("road_geometry", True)):
                bridge_dsm = ensure_dsm(
                    terrain.bbox,
                    work_dir,
                    wcs_url=str(config.terrain["wcs_url"]),
                    resolution_m=float(config.terrain["resolution_m"]),
                    timeout=float(config.terrain["timeout_s"]),
                    max_retries=int(config.terrain["max_retries"]),
                )
                deck_dsm = raster_sampler(bridge_dsm) if bridge_dsm else None

            surfaces = build_surfaces(
                config.bbox,
                work_dir,
                surfaces_cfg=config.surfaces,
                terrain=terrain,
                deck_sampler=deck_dsm,
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

    structures = StructureSet()
    structure_texture = None
    if want_structures:
        with Stage("bridges and tunnels (BGT)", next_step(), total):
            # A bridge deck is a hard surface the surface model sees, so its
            # height is measured rather than guessed. Fetched here only if the
            # trees stage has not already brought it in.
            dsm_path = ensure_dsm(
                terrain.bbox,
                work_dir,
                wcs_url=str(config.terrain["wcs_url"]),
                resolution_m=float(config.terrain["resolution_m"]),
                timeout=float(config.terrain["timeout_s"]),
                max_retries=int(config.terrain["max_retries"]),
            )
            structures = build_structures(
                config.bbox,
                work_dir,
                structures_cfg=config.structures,
                terrain=terrain,
                dsm_sampler=raster_sampler(dsm_path) if dsm_path else None,
            )
            if len(structures.tris):
                save_structures(structures, work_dir / "structures.npz")
                structure_texture = generate_structure_texture(work_dir)

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
            structure_texture=structure_texture,
            structure_kind_names={
                str(k): v for k, v in STRUCTURE_KIND_NAMES.items()
            },
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
                "structures": {
                    **structures.stats(),
                    "deck_heights_from": "AHN DSM, median over each deck",
                    "tunnel_depth": (
                        "constructed, not surveyed: no open dataset carries "
                        "Dutch tunnel depths"
                    ),
                },
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
        check_structures(
            report, structures if want_structures else None, terrain
        )
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
