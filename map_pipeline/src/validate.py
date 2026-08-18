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

    check_terrain_mesh(report, getattr(terrain, "mesh", None), bbox)
    check_constrained_mesh(report, getattr(terrain, "constrained", None), bbox)


def check_constrained_mesh(report: CheckReport, mesh, bbox: BBox) -> None:
    """The terrain folds along real features, and still tiles the area."""
    if mesh is None:
        return

    corners = mesh.vertices[mesh.triangles][:, :, :2]
    a = corners[:, 1] - corners[:, 0]
    b = corners[:, 2] - corners[:, 0]
    twice_area = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    covered = 0.5 * float(twice_area.sum())
    want = bbox.width * bbox.height
    report.add(
        "terrain_mesh_tiles_the_bbox",
        abs(covered - want) < 1e-6 * want,
        f"triangles cover {covered:.1f} m2 of a {want:.0f} m2 bbox",
    )
    report.add(
        "terrain_mesh_wound_up",
        bool((twice_area > 0).all()),
        f"{int((twice_area <= 0).sum())} triangles face down or are degenerate",
    )

    # The whole point: an edge where the ground has one. Without this the mesh
    # is just a Delaunay triangulation of some points and nothing folds.
    report.add(
        "terrain_folds_along_features",
        mesh.breakline_edges > 0,
        f"{mesh.breakline_edges} breakline edges from "
        f"{', '.join(f'{k} {v}' for k, v in mesh.counts.items() if v)}",
    )
    # This used to allow eight times the tolerance, or a whole metre, and only
    # warn, which let a mesh three times outside its tolerance ship in silence.
    # It now asks for the tolerance -- but counted against what the height model
    # can actually deliver, not against the number in the config.
    #
    # A tolerance is a promise about a surface, and AHN is a grid with steps in
    # it: a quay wall, the lip of a filled building hole. Where two neighbouring
    # samples differ by 60 cm, no triangle that is not aligned to the sample grid
    # gets closer than about 30 cm of the pair, however finely it is cut. Judging
    # the mesh against 0.10 m there is judging it for the data's resolution, and
    # it does harm: refinement chased it to 701,978 triangles on a real Utrecht
    # kilometre and still missed by a factor of five.
    over = getattr(mesh, "triangles_over_allowance", None)
    if over is None:
        report.add(
            "terrain_mesh_follows_the_ground",
            mesh.max_error_m <= CONSTRAINED_ERROR_SLACK * mesh.tolerance_m,
            f"worst gap between the mesh and the height grid is "
            f"{mesh.max_error_m:.3f} m against a {mesh.tolerance_m:.2f} m tolerance",
        )
    else:
        reachable = getattr(mesh, "reachable_tolerance_m", mesh.tolerance_m)
        total = len(mesh.triangles)
        detail = (
            f"{over} of {total} triangles are further from the ground than the "
            f"height model allows; worst gap {mesh.max_error_m:.3f} m, tolerance "
            f"{mesh.tolerance_m:.2f} m"
        )
        if reachable > mesh.tolerance_m * 1.01:
            detail += (
                f" (the model steps by up to {2 * reachable:.2f} m between "
                f"samples, so {reachable:.2f} m is the most it can promise "
                f"where it steps)"
            )
        report.add(
            "terrain_mesh_follows_the_ground",
            over <= MAX_OVER_ALLOWANCE_FRACTION * max(total, 1),
            detail,
        )

    quality = _triangle_quality(mesh)
    # A mesh can be perfectly accurate and still be unusable. Nothing measured
    # the shape of these triangles before, which is exactly how a terrain made
    # largely of wedges passed every check it had.
    #
    # Counted as ours or the input's, because the difference is real: two
    # surveyed outlines that meet at half a degree put a half-degree triangle in
    # the mesh and there is nowhere to put a point that improves it.
    ours = getattr(mesh, "slivers_of_our_own", quality["under_one_degree"])
    from_input = getattr(mesh, "slivers_from_input", 0)
    detail = (
        f"{ours} of {quality['count']} triangles have an angle under 1 degree "
        f"that is not explained by a sharp corner in the source outlines; "
        f"worst is {quality['worst_angle_deg']:.2f} deg, median smallest "
        f"{quality['median_angle_deg']:.1f} deg"
    )
    if from_input:
        detail += f" ({from_input} more sit in wedges the input already had)"
    report.add(
        "terrain_triangles_are_not_slivers",
        ours <= MAX_SLIVER_FRACTION * quality["count"],
        detail,
    )
    report.add(
        "terrain_triangles_are_well_shaped",
        quality["under_ten_degrees"] <= MAX_THIN_FRACTION * quality["count"],
        f"{100 * quality['under_ten_degrees'] / max(quality['count'], 1):.1f}% "
        f"of triangles have an angle under 10 degrees",
        severity="warning",
    )
    # A breakline the ground has left behind is a crease in the wrong place,
    # and it is invisible to the error check above because that measures inside
    # triangles, not along their constrained edges.
    sagging = getattr(mesh, "breaklines_over_allowance", None)
    report.add(
        "breaklines_lie_on_the_ground",
        (
            sagging <= MAX_OVER_ALLOWANCE_FRACTION * max(mesh.breakline_edges, 1)
            if sagging is not None
            else quality["worst_edge_sag_m"]
            <= CONSTRAINED_ERROR_SLACK * mesh.tolerance_m
        ),
        (
            f"{sagging} of {mesh.breakline_edges} breaklines have ground under "
            f"them further away than the height model allows; worst "
            f"{quality['worst_edge_sag_m']:.3f} m"
            if sagging is not None
            else f"the ground under the breaklines strays up to "
            f"{quality['worst_edge_sag_m']:.3f} m from them"
        ),
        severity="warning",
    )


# Refinement drives the worst height error down to the tolerance itself, so
# there is no slack to allow here beyond the arithmetic.
CONSTRAINED_ERROR_SLACK = 1.05

# A few triangles will always be beyond help: where two surveyed outlines meet
# at a sharp angle there is nowhere to put a point that improves them. This is
# the share allowed to be beyond what the height model can deliver -- one in a
# thousand, which on a 1 km area is a handful.
MAX_OVER_ALLOWANCE_FRACTION = 0.001

# Some slivers are unavoidable where two surveyed outlines meet at a sharp
# angle: no triangulation can put a good triangle in a wedge the input already
# had. A handful is the input's fault; a percent is ours.
MAX_SLIVER_FRACTION = 0.001
MAX_THIN_FRACTION = 0.05


def _ground_allowance(terrain, points, *, floor_m: float) -> np.ndarray:
    """How close anything draped on the grid can get to it, at these points.

    The same idea the terrain mesh is judged by, and deliberately the same
    function behind it: where two neighbouring height samples differ by a step,
    nothing flat spanning the pair sits closer than about half of it. Judging a
    road against a flat number instead means a single triangle crossing a quay
    wall fails a drape whose median error is 22 mm.
    """
    heights = getattr(terrain, "heights", None)
    xs = getattr(terrain, "xs", None)
    ys = getattr(terrain, "ys", None)
    if heights is None or xs is None or ys is None or len(xs) < 2:
        return np.full(len(points), floor_m)

    from .terrain_mesh import height_allowance

    allowance = height_allowance(np.asarray(heights, dtype=np.float64), floor_m)
    cols = np.clip(
        np.round((points[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1
    )
    rows = np.clip(
        np.round((points[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1
    )
    return allowance[rows, cols]


def _triangle_quality(mesh) -> dict:
    """Smallest angle per triangle, and how far the ground sags under an edge."""
    import numpy as np

    corner = mesh.vertices[mesh.triangles][:, :, :2]
    angles = []
    for i in range(3):
        u = corner[:, (i + 1) % 3] - corner[:, i]
        v = corner[:, (i + 2) % 3] - corner[:, i]
        cos = (u * v).sum(axis=1) / np.maximum(
            np.hypot(*u.T) * np.hypot(*v.T), 1e-30
        )
        angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    smallest = np.min(np.stack(angles), axis=0)
    return {
        "count": int(len(smallest)),
        "worst_angle_deg": float(smallest.min()) if len(smallest) else 0.0,
        "median_angle_deg": float(np.median(smallest)) if len(smallest) else 0.0,
        "under_one_degree": int((smallest < 1.0).sum()),
        "under_ten_degrees": int((smallest < 10.0).sum()),
        "worst_edge_sag_m": float(getattr(mesh, "max_edge_sag_m", 0.0)),
    }


# The nested error bound is measured at hypotenuse midpoints, and a vertex can
# sit on a hypotenuse whose ends were themselves moved, so the height a mesh
# gives up is bounded by the tolerance only to within a small factor. Measured
# at about 1.7x across real areas; this is the ceiling before it counts as a
# defect rather than the expected slack.
MESH_ERROR_SLACK = 3.0


def check_terrain_mesh(report: CheckReport, mesh, bbox: BBox) -> None:
    """The simplified terrain still describes the ground, and still tiles it."""
    if mesh is None:
        return

    report.add(
        "terrain_mesh_within_tolerance",
        mesh.max_error_m <= mesh.tolerance_m * MESH_ERROR_SLACK,
        f"worst height given up is {mesh.max_error_m:.3f} m against a "
        f"{mesh.tolerance_m:.2f} m tolerance "
        f"(mean {mesh.mean_error_m:.4f} m)",
    )

    # Nothing else catches a hole or a double-covered patch, and either one
    # reads as a tear in Unity rather than as a wrong height.
    corners = mesh.vertices[mesh.triangles][:, :, :2]
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 2] - corners[:, 0]
    twice_area = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    cells = (mesh.grid_vertices - 1) ** 2
    covered = 0.5 * float(twice_area.sum())
    report.add(
        "terrain_mesh_tiles_the_bbox",
        abs(covered - cells) < 1e-6 * cells,
        f"triangles cover {covered:.0f} grid cells against {cells}",
    )
    report.add(
        "terrain_mesh_wound_up",
        bool((twice_area > 0).all()),
        f"{int((twice_area <= 0).sum())} triangles face down or are degenerate",
    )
    report.add(
        "terrain_mesh_saves_vertices",
        mesh.triangle_count < 2 * cells,
        f"{mesh.triangle_count} triangles against {2 * cells} for the full "
        f"grid ({100.0 * mesh.triangle_count / (2 * cells):.1f}%)",
        severity="warning",
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

        # The bed is pushed down at mesh vertices, so a canal spanned by one
        # big triangle would keep the mound the DTM invented over it. The
        # mound is metres tall and the simplifier keeps what moves, so this
        # should never bite -- but nothing else would notice if it did.
        mesh = getattr(terrain, "mesh", None)
        if mesh is not None:
            columns = mesh.vertices[:, 0].astype(int)
            rows = mesh.vertices[:, 1].astype(int)
            inside = int(np.isfinite(bed[rows, columns]).sum())
            expected = water.mean() * len(mesh.vertices) * 0.2
            report.add(
                "water_reaches_the_terrain_mesh",
                inside >= max(len(surfaces.water), int(expected)),
                f"{inside} of {len(mesh.vertices)} terrain mesh vertices sit "
                f"inside a water outline, which is what the bed sinking moves",
            )

    if surfaces.class_grid is not None:
        classified = float((surfaces.class_grid > 0).mean())
        report.add(
            "land_cover_coverage",
            classified > 0.25,
            f"{100 * classified:.0f}% of the grid carries a BGT surface class",
            severity="warning",
        )

    if len(surfaces.road_tris):
        from .surfaces import CLASS_NAMES

        classes = np.unique(surfaces.road_tri_class)
        report.add(
            "road_surface_present",
            True,
            f"{len(surfaces.road_tris)} triangles across "
            + ", ".join(CLASS_NAMES.get(int(c), str(c)) for c in classes),
        )

        # The reason this check exists: earcut leaves slivers over 100 m long,
        # and a flat one that size cut through a canal bank by 1.86 m before
        # the refinement went in. The corners are exact by construction, so
        # the centroid is where a flat triangle misses the ground.
        # Only the roads that should be on the ground. A carriageway on a
        # bridge deliberately is not, and asserting otherwise failed the moment
        # bridges started carrying their own roads.
        at_grade = (
            surfaces.road_tri_level <= 0
            if len(surfaces.road_tri_level) == len(surfaces.road_tris)
            else np.ones(len(surfaces.road_tris), dtype=bool)
        )
        centroid = surfaces.road_tris[at_grade].mean(axis=1)
        ground = terrain.sample(centroid[:, 0], centroid[:, 1])
        # Measured against the drape, with the deliberate lift taken back off.
        # Comparing lift plus drape against one fixed number only ever worked
        # while the lift was 6 cm: raise it to clear a coarser terrain mesh and
        # the check fails for the very thing that was done to satisfy it.
        lift = float(getattr(surfaces, "road_lift_m", 0.0) or 0.0)
        error = np.abs(centroid[:, 2] - lift - ground)
        # Against what the height model can deliver, for the same reason the
        # terrain mesh is: a road crossing a quay wall spans a step between two
        # samples, and no flat triangle sits closer than about half of it
        # however finely the drape is cut. A flat 0.5 m limit failed a
        # carriageway whose median error was 22 mm for one triangle over a wall.
        allowance = _ground_allowance(terrain, centroid[:, :2], floor_m=0.5)
        over = int((error > allowance).sum())
        report.add(
            "road_surface_follows_terrain",
            over <= 0.001 * max(len(error), 1),
            f"{over} of {len(error)} road triangles are further from the ground "
            f"than the height model allows; it follows to {np.median(error):.3f} m "
            f"at the median and never more than {error.max():.2f} m, over a "
            f"{lift:.3f} m lift that keeps it from fighting the terrain for depth",
        )

        elevated = ~at_grade
        if elevated.any():
            high = surfaces.road_tris[elevated].reshape(-1, 3)
            over = high[:, 2] - terrain.sample(high[:, 0], high[:, 1])
            report.add(
                "bridge_roads_ride_their_decks",
                bool((over > 0.0).all()),
                f"{int(elevated.sum())} road triangles on bridges, running "
                f"{over.min():.2f} to {over.max():.2f} m above the ground "
                f"they cross",
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


def check_vehicles(report: CheckReport, vehicles, bbox: BBox, surfaces=None) -> None:
    """Cars are on land and boats are on water, at the right height.

    Both are placed from proxies rather than from a register of vehicles, so
    the checks are about whether the proxy held: a boat that came out on a
    street means a mooring post was matched to the wrong side.
    """
    from .vehicles import KIND_BOAT, KIND_CAR, points_in_ring

    if vehicles is None or not len(vehicles):
        report.add("vehicles_present", True, "no vehicles", severity="warning")
        return

    xs, ys = vehicles.xy[:, 0], vehicles.xy[:, 1]
    report.add(
        "vehicles_present",
        True,
        ", ".join(
            f"{value} {key}"
            for key, value in vehicles.stats().items()
            if key.endswith("s") and value
        ),
    )
    report.add(
        "vehicles_inside_bbox",
        bool(
            (xs >= bbox.xmin).all() and (xs <= bbox.xmax).all()
            and (ys >= bbox.ymin).all() and (ys <= bbox.ymax).all()
        ),
        "every vehicle is inside the bbox",
    )

    cars = vehicles.kind == KIND_CAR
    if cars.any():
        report.add(
            "cars_upright_on_ground",
            bool(np.isfinite(vehicles.z_nap[cars]).all()),
            f"{int(cars.sum())} cars between "
            f"{vehicles.z_nap[cars].min():.2f} and "
            f"{vehicles.z_nap[cars].max():.2f} m NAP",
        )

    boats = vehicles.kind == KIND_BOAT
    if boats.any() and surfaces is not None and surfaces.water:
        rings = [
            body.rings[0][:, :2] for body in surfaces.water if len(body.rings)
        ]
        afloat = 0
        for index in np.flatnonzero(boats):
            point = vehicles.xy[index][None, :]
            if any(points_in_ring(point, ring)[0] for ring in rings):
                afloat += 1
        report.add(
            "boats_are_on_water",
            afloat == int(boats.sum()),
            f"{afloat} of {int(boats.sum())} boats sit inside a water body",
        )

        levels = {round(body.level_nap, 2) for body in surfaces.water}
        boat_levels = np.unique(np.round(vehicles.z_nap[boats], 2))
        report.add(
            "boats_at_water_level",
            all(level in levels for level in boat_levels),
            f"boat heights {vehicles.z_nap[boats].min():.2f} to "
            f"{vehicles.z_nap[boats].max():.2f} m NAP, all matching a "
            f"water body's own level",
        )


def check_rails(report: CheckReport, rails, terrain) -> None:
    """Track is on the ground, the right gauge, and the right way up."""
    from .rails import KIND_NAMES, PROFILES

    if rails is None or not len(rails):
        report.add("rails_present", True, "no railway here", severity="warning")
        return

    report.add(
        "rails_present",
        True,
        f"{len(rails.lines)} tracks, {rails.length_m() / 1000:.2f} km ("
        + ", ".join(
            f"{name} {rails.length_m(kind) / 1000:.2f} km"
            for kind, name in KIND_NAMES.items()
            if rails.length_m(kind) > 0
        )
        + ")",
    )

    if not len(rails.rail_tris):
        return

    # Every rail head must sit above the bed it runs on, or the track is
    # inside out. Compared per kind: a street tram has no ballast at all and
    # runs at road level, so against the global minimum it looked as though it
    # were under the ballast of an elevated railway somewhere else entirely.
    inverted = []
    for kind, name in KIND_NAMES.items():
        beds = rails.ballast_tris[rails.ballast_kind == kind]
        heads = rails.rail_tris[rails.rail_kind == kind]
        if not len(beds) or not len(heads):
            continue
        if heads[:, :, 2].min() < beds[:, :, 2].min():
            inverted.append(name)
    report.add(
        "rails_above_ballast",
        not inverted,
        "rail heads sit above the ballast they run on"
        if not inverted
        else f"rails are below their own ballast on: {', '.join(inverted)}",
    )

    # The gauge is the one dimension a viewer will notice being wrong, and it
    # is fixed at 1435 mm for every kind of track in the country.
    heads = rails.rail_tris.reshape(-1, 3)
    report.add(
        "rails_follow_terrain",
        bool(np.isfinite(heads[:, 2]).all()),
        f"rail heads run from {heads[:, 2].min():.2f} to {heads[:, 2].max():.2f} m NAP",
    )

    at_grade = [line for line in rails.lines if line.level == 0]
    if at_grade and terrain is not None:
        points = np.vstack([line.points for line in at_grade])
        ground = terrain.sample(points[:, 0], points[:, 1])
        report.add(
            "rails_at_grade_on_the_ground",
            bool(np.isfinite(ground).all()),
            f"{len(at_grade)} tracks at grade, ground under them running "
            f"{ground.min():.2f} to {ground.max():.2f} m NAP",
        )

    elevated = int(rails.counts.get("tracks_elevated", 0))
    if elevated:
        report.add(
            "rails_elevated_noted",
            True,
            f"{elevated} of {len(rails.lines)} tracks are on a viaduct",
            severity="warning",
        )


def check_structures(report: CheckReport, structures, terrain) -> None:
    """Bridges are above what they cross; tunnels are below it."""
    from .structures import (
        KIND_DECK, KIND_PIER, KIND_TUNNEL_ROAD, KIND_TUNNEL_WALL,
    )

    if structures is None or not len(structures):
        report.add(
            "structures_present", True, "no bridges or tunnels here",
            severity="warning",
        )
        return

    report.add(
        "structures_present",
        True,
        f"{structures.count_of(KIND_DECK)} decks, "
        f"{structures.count_of(KIND_PIER)} piers, "
        f"{structures.count_of(KIND_TUNNEL_ROAD)} tunnel parts",
    )

    measured = int(structures.counts.get("decks_measured_from_dsm", 0))
    guessed = int(structures.counts.get("decks_without_a_reading", 0))
    if measured or guessed:
        report.add(
            "deck_heights_measured",
            guessed == 0,
            f"{measured} decks took their height from the surface model, "
            f"{guessed} fell back to the ordinal level",
            severity="warning",
        )

    if not len(structures.tris):
        return

    decks = structures.tri_kind == KIND_DECK
    if decks.any():
        z = structures.tris[decks][:, :, 2]
        centre = structures.tris[decks].reshape(-1, 3)
        ground = terrain.sample(centre[:, 0], centre[:, 1])
        # The reason this exists: draping a deck on the terrain sank the
        # Erasmusbrug into the Maas.
        above = float((centre[:, 2] >= ground - 0.5).mean())
        report.add(
            "decks_above_the_ground_they_cross",
            above > 0.98,
            f"{100 * above:.1f}% of deck geometry sits at or above the terrain "
            f"under it, spanning {z.min():.2f} to {z.max():.2f} m NAP",
        )

    tunnels = (structures.tri_kind == KIND_TUNNEL_ROAD)
    if tunnels.any():
        corners = structures.tris[tunnels].reshape(-1, 3)
        ground = terrain.sample(corners[:, 0], corners[:, 1])
        below = ground - corners[:, 2]
        report.add(
            "tunnel_runs_below_the_ground",
            bool((below >= -0.5).all()),
            f"tunnel road runs {below.min():.2f} to {below.max():.2f} m below "
            f"the surface (a drawn profile, not a survey)",
        )
        # Portals have to reach daylight, or the tunnel is a sealed box.
        report.add(
            "tunnel_reaches_its_portals",
            float(below.min()) < 1.0,
            f"shallowest tunnel point is {below.min():.2f} m down, so the "
            f"ramps meet the surface",
        )

    walls = structures.tri_kind == KIND_TUNNEL_WALL
    if walls.any() and tunnels.any():
        report.add(
            "tunnel_walls_reach_the_road",
            bool(
                structures.tris[walls][:, :, 2].min()
                <= structures.tris[tunnels][:, :, 2].min() + 0.5
            ),
            "tunnel walls run down to the road they enclose",
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


def check_built_terrain(report: CheckReport, work_dir: Path, terrain) -> None:
    """The terrain in the model is the one the pipeline meant to build.

    Every other terrain check reads the mesh held in memory, which says nothing
    about what the Blender stage drew. When scene.json lost its mesh_file the
    model shipped the plain grid and all of them still passed, because they
    were never looking at the model.
    """
    summary_path = work_dir / "blender_summary.json"
    if not summary_path.is_file():
        return
    try:
        built = (json.loads(summary_path.read_text(encoding="utf-8")) or {}).get(
            "terrain"
        ) or {}
    except (OSError, ValueError):
        return
    if not built:
        return

    intended = getattr(terrain, "constrained", None) or getattr(terrain, "mesh", None)
    if intended is None:
        report.add(
            "terrain_built_as_planned",
            built.get("quads", 0) > 0,
            f"no mesh was asked for, and the model has {built.get('quads', 0)} "
            f"grid quads",
        )
        return

    # Counted in triangles, because a quad in this mesh is two of them fused
    # and the fusing is the last thing that happens. Asserting that no quad
    # exists was right while every terrain was triangles and became wrong the
    # moment pairing shipped: a perfect 2 km model reported 540424 triangles
    # and 1031304 quads against 2603032 built, and 540424 + 2 x 1031304 is
    # exactly 2603032.
    #
    # This still catches what the check was written for -- the run that shipped
    # the plain grid instead of the mesh it had built -- because a grid's face
    # count does not agree with the mesh's triangle count either.
    triangles = int(built.get("triangles", 0))
    quads = int(built.get("quads", 0))
    report.add(
        "terrain_built_as_planned",
        triangles + 2 * quads == int(intended.triangle_count),
        f"the model contains {triangles} terrain triangles and {quads} quads, "
        f"which is {triangles + 2 * quads} triangles' worth against the "
        f"{intended.triangle_count} that were built for it",
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
    "check_built_terrain",
    "check_export",
    "check_fbx_reimport",
    "check_terrain",
    "check_constrained_mesh",
    "check_terrain_mesh",
    "write_report",
]
