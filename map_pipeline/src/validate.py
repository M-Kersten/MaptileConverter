"""Headless checks over the pipeline output.

Nearly everything that goes wrong in a geometric pipeline is a coordinate or a
scale mistake, and those always show up in the numbers. These checks run without
anyone looking at the model, so a bad run reports itself.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import BBox

LOG = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        if self.passed:
            mark = "PASS"
        else:
            mark = "FAIL" if self.severity == "error" else "WARN"
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class CheckReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, severity: str = "error") -> None:
        # numpy predicates return np.bool_, which json cannot serialise.
        check = Check(name, bool(passed), detail, severity)
        self.checks.append(check)
        log = LOG.info if passed else (LOG.error if severity == "error" else LOG.warning)
        log("%s", check)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "passed": sum(1 for c in self.checks if c.passed),
            "failed": len(self.failures),
            "warned": len(self.warnings),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }

    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        return (
            f"{passed}/{len(self.checks)} checks passed, "
            f"{len(self.failures)} failed, {len(self.warnings)} warnings"
        )


def check_bbox(report: CheckReport, bbox: BBox) -> None:
    """Sides between 900 and 1100 m, per the plan's tolerance."""
    for label, side in (("width", bbox.width), ("height", bbox.height)):
        report.add(
            f"bbox_{label}",
            900.0 <= side <= 1100.0,
            f"{side:.1f} m (expected 900-1100 m)",
            severity="warning",
        )
    report.add(
        "bbox_in_rd_domain",
        -7000 <= bbox.xmin <= 300000 and 289000 <= bbox.ymin <= 629000,
        f"origin {bbox.center[0]:.0f}, {bbox.center[1]:.0f} is inside RD New",
    )


def check_terrain(report: CheckReport, terrain, bbox: BBox) -> None:
    """The DTM covers the bbox, has no unfilled holes, and reads as NAP."""
    heights = terrain.heights

    report.add(
        "terrain_no_nan",
        bool(np.isfinite(heights).all()),
        f"{int((~np.isfinite(heights)).sum())} non-finite height cells",
    )
    report.add(
        "terrain_grid_shape",
        heights.shape[0] == heights.shape[1] == terrain.n,
        f"{heights.shape[0]}x{heights.shape[1]} vertex grid",
    )

    # The Netherlands runs from about -7 m NAP to +325 m.
    in_range = bool(heights.min() > -25.0 and heights.max() < 400.0)
    report.add(
        "terrain_nap_plausible",
        in_range,
        f"NAP range {heights.min():.2f} to {heights.max():.2f} m",
    )

    # A DTM over a dense city is mostly holes by design, so this is a warning
    # about texture quality rather than a failure.
    report.add(
        "terrain_gap_fill",
        terrain.filled_fraction < 0.75,
        f"{100 * terrain.filled_fraction:.1f}% of grid vertices were "
        f"interpolated across nodata "
        f"({100 * terrain.nodata_fraction_raw:.1f}% of source pixels were nodata)",
        severity="warning",
    )

    grid_span_x = terrain.xs[-1] - terrain.xs[0]
    report.add(
        "terrain_covers_bbox",
        abs(grid_span_x - bbox.width) < 1e-6,
        f"grid spans {grid_span_x:.3f} m against a {bbox.width:.3f} m bbox",
    )


def check_buildings(report: CheckReport, buildings, *, max_height_m: float) -> None:
    """Buildings exist, and every height is positive and under the ceiling."""
    report.add(
        "buildings_present",
        len(buildings) > 0,
        f"{len(buildings)} buildings",
    )
    if not len(buildings):
        return

    heights = np.array([b.height_m for b in buildings.buildings])
    report.add(
        "building_heights_positive",
        bool((heights > 0).all()),
        f"lowest height {heights.min():.2f} m",
    )
    report.add(
        "building_heights_under_ceiling",
        bool((heights <= max_height_m).all()),
        f"tallest {heights.max():.2f} m (ceiling {max_height_m:.0f} m)",
    )
    missing_walls = sum(1 for b in buildings.buildings if not len(b.all_wall_tris))
    missing_roofs = sum(1 for b in buildings.buildings if not len(b.roof_tris))
    report.add(
        "buildings_have_walls_and_roofs",
        missing_walls == 0 and missing_roofs == 0,
        f"{missing_walls} without walls, {missing_roofs} without roofs",
        severity="warning",
    )
    report.add(
        "building_triangulation",
        buildings.triangulation_failures == 0,
        f"{buildings.triangulation_failures} surfaces failed to triangulate "
        f"({buildings.degenerate_surfaces} zero-area source slivers skipped "
        f"separately)",
        severity="warning",
    )


def check_buildings_on_terrain(
    report: CheckReport, buildings, terrain, *, tolerance_m: float = 0.75
) -> None:
    """No building floats above the ground it stands on.

    The wall base is compared against the lowest terrain sample under the
    footprint. A base above that would leave a visible gap, since the building's
    own GroundSurface is dropped.
    """
    floating = []
    for building in buildings.buildings:
        # After the ground-floor split the base of the building sits in the
        # ground-storey geometry, so both halves have to be considered.
        walls = building.all_wall_tris
        if not len(walls):
            continue
        points = walls.reshape(-1, 3)
        base_z = float(points[:, 2].min())
        xs = np.linspace(points[:, 0].min(), points[:, 0].max(), 3)
        ys = np.linspace(points[:, 1].min(), points[:, 1].max(), 3)
        grid_x, grid_y = np.meshgrid(xs, ys)
        terrain_min = float(np.min(terrain.sample(grid_x, grid_y)))
        if base_z > terrain_min + tolerance_m:
            floating.append((building.identifier, base_z - terrain_min))

    report.add(
        "buildings_meet_terrain",
        not floating,
        (
            f"all {len(buildings)} building bases reach the terrain"
            if not floating
            else f"{len(floating)} buildings float, worst by "
            f"{max(gap for _, gap in floating):.2f} m"
        ),
    )


def check_trees(report: CheckReport, trees, bbox: BBox) -> None:
    """Trees are inside the area, plausibly tall, and not duplicated."""
    if trees is None or not len(trees):
        report.add("trees_present", True, "no trees requested or found", severity="warning")
        return

    xs = np.array([t.x for t in trees.trees])
    ys = np.array([t.y for t in trees.trees])
    heights = np.array([t.height_m for t in trees.trees])

    report.add(
        "trees_present",
        len(trees) > 0,
        f"{len(trees)} trees ({trees.superseded_dropped} superseded BGT "
        f"versions dropped)",
    )
    report.add(
        "trees_inside_bbox",
        bool(
            (xs >= bbox.xmin).all() and (xs <= bbox.xmax).all()
            and (ys >= bbox.ymin).all() and (ys <= bbox.ymax).all()
        ),
        "every tree is inside the bbox",
    )
    report.add(
        "tree_heights_plausible",
        bool((heights > 0).all() and (heights <= 45).all()),
        f"heights {heights.min():.1f} to {heights.max():.1f} m "
        f"(median {np.median(heights):.1f})",
    )

    # Stacked duplicates are what a naive read of the BGT version history gives.
    positions = np.column_stack([np.round(xs, 2), np.round(ys, 2)])
    unique = len(np.unique(positions, axis=0))
    report.add(
        "trees_not_duplicated",
        unique == len(trees),
        f"{unique} distinct positions for {len(trees)} trees",
    )


def check_surfaces(report: CheckReport, surfaces, terrain) -> None:
    """Water sits at a sane level and its bed is below it everywhere."""
    if surfaces is None or not surfaces.water:
        report.add(
            "water_present", True, "no water in this area", severity="warning"
        )
        return

    levels = np.array([body.level_nap for body in surfaces.water])
    report.add(
        "water_present",
        True,
        f"{len(surfaces.water)} bodies covering {surfaces.water_area_m2:.0f} m2, "
        f"levels {levels.min():.2f} to {levels.max():.2f} m NAP",
    )
    report.add(
        "water_levels_plausible",
        bool((levels > -10).all() and (levels < 60).all()),
        f"levels between {levels.min():.2f} and {levels.max():.2f} m NAP",
    )

    if surfaces.water_bed is not None and surfaces.water_bed.size:
        bed = surfaces.water_bed
        water = np.isfinite(bed)
        # Each body's bed has to clear its own surface, not the average one:
        # a shared bed would poke through the lowest canal.
        report.add(
            "water_bed_below_surface",
            bool((bed[water] < np.max(levels) + 1e-6).all()),
            f"bed runs {np.nanmin(bed):.2f} to {np.nanmax(bed):.2f} m NAP "
            f"under {int(water.sum())} cells",
        )

    if surfaces.class_grid is not None:
        classified = float((surfaces.class_grid > 0).mean())
        report.add(
            "land_cover_coverage",
            classified > 0.25,
            f"{100 * classified:.0f}% of the grid carries a BGT surface class",
            severity="warning",
        )


def check_furniture(report: CheckReport, furniture, bbox: BBox) -> None:
    """Street furniture is inside the area and standing on the ground."""
    if furniture is None or not len(furniture):
        report.add(
            "furniture_present", True, "no street furniture", severity="warning"
        )
        return

    xs, ys = furniture.xy[:, 0], furniture.xy[:, 1]
    report.add(
        "furniture_present",
        True,
        ", ".join(f"{v} {k}" for k, v in furniture.counts.items() if v),
    )
    report.add(
        "furniture_inside_bbox",
        bool(
            (xs >= bbox.xmin).all() and (xs <= bbox.xmax).all()
            and (ys >= bbox.ymin).all() and (ys <= bbox.ymax).all()
        ),
        "every object is inside the bbox",
    )


def check_aerial(report: CheckReport, aerial, bbox: BBox) -> None:
    """The image has the requested size, real content, and matches the bbox."""
    from PIL import Image

    from .imagery import _allow_large_images

    _allow_large_images()
    with Image.open(aerial.path) as image:
        size = image.size
        sample = np.asarray(image.convert("RGB").resize((256, 256))).astype(np.float64)

    report.add(
        "aerial_size",
        size == (aerial.size_px, aerial.size_px),
        f"{size[0]}x{size[1]} px (requested {aerial.size_px})",
    )
    report.add(
        "aerial_has_content",
        float(sample.std()) > 5.0,
        f"pixel standard deviation {sample.std():.1f} (a flat image means the "
        f"request missed its coverage)",
    )
    report.add(
        "aerial_georef_matches_bbox",
        aerial.bbox.as_tuple() == bbox.as_tuple(),
        f"image bbox {aerial.bbox} against pipeline bbox {bbox}",
    )

    world_file = aerial.path.with_suffix(".pgw")
    if world_file.is_file():
        values = [float(v) for v in world_file.read_text().split()]
        # X and Y are checked separately: a bbox given in WGS84 is never exactly
        # square once it lands in RD, so the pixels are not square either.
        expected_x = bbox.width / aerial.size_px
        expected_y = bbox.height / aerial.size_px
        report.add(
            "aerial_world_file",
            abs(values[0] - expected_x) < 1e-6 and abs(values[3] + expected_y) < 1e-6,
            f"world file resolution {values[0]:.6f} x {-values[3]:.6f} m/px "
            f"(expected {expected_x:.6f} x {expected_y:.6f})",
        )


def check_export(
    report: CheckReport,
    out_dir: Path,
    *,
    work_dir: Path,
    bbox: BBox,
    fbx_name: str,
    aerial_name: str,
    metadata_name: str,
) -> None:
    """The promised files exist and the model bounds look right."""
    fbx = out_dir / fbx_name
    report.add(
        "fbx_written",
        fbx.is_file() and fbx.stat().st_size > 1024,
        f"{fbx.name} is {fbx.stat().st_size / 1e6:.1f} MB"
        if fbx.is_file()
        else f"{fbx} is missing",
    )
    for filename in (aerial_name, metadata_name):
        path = out_dir / filename
        report.add(
            f"output_{filename}",
            path.is_file(),
            f"{filename} present" if path.is_file() else f"{filename} is missing",
        )

    summary_path = work_dir / "blender_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        bounds = summary["bounds_local"]
        span_x = bounds["x"][1] - bounds["x"][0]
        span_y = bounds["y"][1] - bounds["y"][0]
        center_x = 0.5 * (bounds["x"][0] + bounds["x"][1])
        center_y = 0.5 * (bounds["y"][0] + bounds["y"][1])

        # Buildings that straddle the edge are returned whole, so the model can
        # reach a little past the terrain. Allowed, but bounded.
        report.add(
            "model_span",
            abs(span_x - bbox.width) < 120.0 and abs(span_y - bbox.height) < 120.0,
            f"model spans {span_x:.1f} x {span_y:.1f} m against a "
            f"{bbox.width:.0f} x {bbox.height:.0f} m bbox "
            f"(edge buildings are kept whole)",
        )
        allowed_x = _centring_tolerance(span_x, bbox.width)
        allowed_y = _centring_tolerance(span_y, bbox.height)
        report.add(
            "model_centred_on_origin",
            abs(center_x) <= allowed_x and abs(center_y) <= allowed_y,
            f"model centre at ({center_x:.1f}, {center_y:.1f}) m from origin, "
            f"within the {allowed_x:.1f} x {allowed_y:.1f} m the overhang allows",
        )


def _centring_tolerance(span: float, bbox_size: float) -> float:
    """How far off centre the model may sit, given how far it overhangs.

    The terrain always covers the bbox exactly, so the model's bounding box
    always contains it and can only ever be *larger*. That makes the offset an
    identity rather than a free parameter::

        centre = (overhang_far - overhang_near) / 2

    which is at most half the total overhang, reached when every metre of it is
    on one side. Buildings are kept whole when their centroid is inside the
    bbox, so one long warehouse near an edge is enough to put it there.

    A fixed limit was wrong twice over. It failed runs that were merely
    lopsided — 120 m of overhang is allowed by ``model_span``, which permits a
    60 m offset, while this check stopped at 30 — and it was far too lax about
    the fault it exists to catch. A wrong origin shifts the model without
    changing its span, so it leaves no overhang to spend, and the tolerance
    collapses to nothing. That is exactly the sensitivity wanted here.
    """
    # A metre of slack for floating point and for the ground skirt.
    return 0.5 * max(span - bbox_size, 0.0) + 1.0


def check_fbx_reimport(report: CheckReport, fbx_path: Path, bbox: BBox) -> None:
    """Re-import the FBX in a clean Blender scene and re-measure it.

    This is the check that catches an export-side axis or scale mistake: the
    numbers only stay sane if what was written matches what was built.
    """
    try:
        import bpy
    except ImportError:
        report.add(
            "fbx_reimport",
            True,
            "skipped: bpy is not importable in this interpreter",
            severity="warning",
        )
        return

    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    except Exception as exc:  # noqa: BLE001 - a failed import is the finding
        report.add("fbx_reimport", False, f"re-import failed: {exc}")
        return

    import mathutils

    objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objects:
        report.add("fbx_reimport", False, "re-imported file has no meshes")
        return

    corners = []
    for obj in objects:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ mathutils.Vector(corner))

    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]

    # Imported through Blender's own FBX axis conversion, so the model comes
    # back in Blender's frame: X east, Y north, Z up.
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)

    report.add(
        "fbx_reimport",
        True,
        f"{len(objects)} meshes, spans {span_x:.1f} x {span_y:.1f} x {span_z:.1f} m",
    )
    report.add(
        "fbx_reimport_scale",
        abs(span_x - bbox.width) < 150.0 and abs(span_y - bbox.height) < 150.0,
        f"horizontal spans {span_x:.1f} x {span_y:.1f} m against a "
        f"{bbox.width:.0f} x {bbox.height:.0f} m bbox",
    )
    report.add(
        "fbx_reimport_height",
        0.0 < span_z < 250.0,
        f"vertical span {span_z:.1f} m",
    )
    center_x = 0.5 * (max(xs) + min(xs))
    center_y = 0.5 * (max(ys) + min(ys))
    allowed_x = _centring_tolerance(span_x, bbox.width)
    allowed_y = _centring_tolerance(span_y, bbox.height)
    report.add(
        "fbx_reimport_centred",
        abs(center_x) <= allowed_x and abs(center_y) <= allowed_y,
        f"centre at ({center_x:.1f}, {center_y:.1f}) m, within the "
        f"{allowed_x:.1f} x {allowed_y:.1f} m the overhang allows",
    )


def write_report(report: CheckReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "Check",
    "CheckReport",
    "check_aerial",
    "check_bbox",
    "check_buildings",
    "check_buildings_on_terrain",
    "check_export",
    "check_fbx_reimport",
    "check_terrain",
    "write_report",
]
