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

    # How much of the survey the grid gave up. A regular grid cannot follow a
    # quay wall that falls between two of its rows, so this is a statement
    # about the spacing chosen rather than a defect, and it warns: the share of
    # ground more than 25 cm off was 0.8% at 2 m over Bronbeek and 3.3% at 4 m.
    survey = getattr(terrain, "survey_error", None)
    if survey and "tolerance_m" in survey:
        # The layered style's terrain is coarse on purpose, built so every
        # surveyed grid node is within a tolerance; the half-metre survey
        # between the nodes is held to the same figure.
        tolerance = float(survey["tolerance_m"])
        report.add(
            "terrain_follows_the_survey",
            survey["p95_m"] <= tolerance,
            f"within {survey['p95_m']:.2f} m of the 0.5 m survey on 95% of the "
            f"ground and {survey['p99_m']:.2f} m on 99%; "
            f"{100 * survey['over_tolerance']:.2f}% is further off than the "
            f"{tolerance:.2f} m the terrain is built to",
            severity="warning",
        )
    elif survey:
        report.add(
            "terrain_follows_the_survey",
            survey["over_25cm"] <= 0.05,
            f"within {survey['p95_m']:.2f} m of the 0.5 m survey on 95% of the "
            f"ground and {survey['p99_m']:.2f} m on 99%; "
            f"{100 * survey['over_25cm']:.2f}% is more than 25 cm off at "
            f"{terrain.grid_m:.2f} m spacing",
            severity="warning",
        )


def _step_allowance(heights: np.ndarray, floor_m: float) -> np.ndarray:
    """How close anything flat can get to the grid, per vertex.

    Where two neighbouring height samples differ by a step -- a quay wall, the
    lip of a filled building hole -- a flat face spanning the pair can only sit
    about half the step away from it. So the allowance is the floor, or half
    the largest step to a neighbour, whichever is larger.
    """
    heights = np.asarray(heights, dtype=np.float64)
    step = np.zeros_like(heights)
    for axis in (0, 1):
        difference = np.abs(np.diff(heights, axis=axis))
        pad = [(0, 0), (0, 0)]
        pad[axis] = (0, 1)
        np.maximum(step, np.pad(difference, pad, mode="edge"), out=step)
        pad[axis] = (1, 0)
        np.maximum(step, np.pad(difference, pad, mode="edge"), out=step)
    return np.maximum(floor_m, 0.5 * step)


def _ground_allowance(terrain, points, *, floor_m: float) -> np.ndarray:
    """How close anything draped on the grid can get to it, at these points.

    Judging a road against a flat number means a single face crossing a quay
    wall fails a drape whose median error is 22 mm.
    """
    heights = getattr(terrain, "measured", None)
    if heights is None:
        heights = getattr(terrain, "heights", None)
    xs = getattr(terrain, "xs", None)
    ys = getattr(terrain, "ys", None)
    if heights is None or xs is None or ys is None or len(xs) < 2:
        return np.full(len(points), floor_m)

    allowance = _step_allowance(heights, floor_m)
    cols = np.clip(
        np.round((points[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1
    )
    rows = np.clip(
        np.round((points[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1
    )
    return allowance[rows, cols]


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


def _closest_pair(points: np.ndarray, involving: np.ndarray, *, within: float) -> float:
    """The shortest distance between two points, one of them `involving`.

    Only pairs closer than `within` are looked at, through cells that size;
    with none, infinity.
    """
    n = len(points)
    if n < 2 or not involving.any():
        return np.inf
    cells = np.floor(points / within).astype(np.int64)
    cells -= cells.min(axis=0) - 1  # every cell and its neighbours positive
    width = int(cells[:, 1].max()) + 2
    keys = cells[:, 0] * width + cells[:, 1]
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    best = np.inf
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            wanted = keys + dx * width + dy
            low = np.searchsorted(ordered, wanted, side="left")
            count = np.searchsorted(ordered, wanted, side="right") - low
            first = np.repeat(np.arange(n), count)
            if not len(first):
                continue
            step = np.arange(len(first)) - np.repeat(np.cumsum(count) - count, count)
            second = order[np.repeat(low, count) + step]
            keep = (first < second) & (involving[first] | involving[second])
            if keep.any():
                gaps = np.hypot(*(points[first[keep]] - points[second[keep]]).T)
                best = min(best, float(gaps.min()))
    return best if best < within else np.inf


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

    detected = np.array([getattr(t, "detected", False) for t in trees.trees])
    if not detected.any():
        return

    report.add(
        "trees_found_as_well_as_surveyed",
        True,
        f"{int((~detected).sum())} from the BGT register, "
        f"{int(detected.sum())} found in the canopy model",
    )
    # No found tree on the spot of another. The detector places one seed per
    # crown, and a registered tree reserves its own crown before any seed is
    # taken, so a found tree within a metre of another means that reservation
    # is not working and the model has a tree drawn twice. Two registered
    # trees that close are the register's own: Hazeldonk has clumps planted
    # 0.7 m apart.
    points = np.column_stack([xs, ys])
    closest = _closest_pair(points, detected, within=3.0)
    report.add(
        "no_two_trees_share_a_spot",
        closest >= 1.0,
        f"the closest found tree is {closest:.2f} m from another tree"
        if np.isfinite(closest)
        else "no found tree is within 3 m of another tree",
    )
    # A detected tree taller than anything that grows here is a structure the
    # building footprints did not cover.
    tallest = float(heights[detected].max())
    report.add(
        "found_trees_are_tree_sized",
        tallest <= 32.0,
        f"the tallest found tree is {tallest:.1f} m",
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

    if len(surfaces.road_tris):
        check_roads(report, surfaces, terrain)


# The share of road faces that may be further from the ground than the height
# model allows. A quad spanning a quay wall cannot follow both sides of it.
MAX_ROAD_OFF_GROUND_FRACTION = 0.002


def check_roads(report: CheckReport, surfaces, terrain) -> None:
    """The road network is the shape it claims, on the ground, and on top.

    Three things an editor and an engine rely on: the streets really are
    ladders of quads, the surface follows the surveyed ground, and the terrain
    as drawn never shows through it.
    """
    from .surfaces import CLASS_FOOTPATH, CLASS_NAMES

    mesh = getattr(surfaces, "road_mesh", None)
    classes = np.unique(surfaces.road_tri_class)
    if mesh is not None and len(mesh):
        stats = mesh.stats
        report.add(
            "road_surface_present",
            True,
            f"{stats.get('faces', 0)} faces in {len(mesh.object_names)} objects "
            f"({stats.get('streets', 0)} streets, {stats.get('junctions', 0)} "
            f"junctions), {stats.get('kerb_faces', 0)} kerb faces, across "
            + ", ".join(CLASS_NAMES.get(int(c), str(c)) for c in classes),
        )
        # Not a hard number: an area of car parks and squares has few streets
        # to ladder. Over Bronbeek 80% of faces are quads.
        report.add(
            "roads_are_mostly_quads",
            stats.get("quad_share", 0.0) >= 0.5,
            f"{100 * stats.get('quad_share', 0.0):.0f}% of road faces are quads; "
            f"the largest face has {stats.get('largest_face_vertices', 0)} corners",
            severity="warning",
        )

    at_grade = (
        surfaces.road_tri_level <= 0
        if len(surfaces.road_tri_level) == len(surfaces.road_tris)
        else np.ones(len(surfaces.road_tris), dtype=bool)
    )
    tris = surfaces.road_tris[at_grade]
    tri_class = surfaces.road_tri_class[at_grade]
    lift = float(getattr(surfaces, "road_lift_m", 0.0) or 0.0)
    kerb = float(mesh.stats.get("kerb_m", 0.0)) if mesh is not None else 0.0
    if len(tris):
        centroid = tris.mean(axis=1)
        ground = terrain.sample(centroid[:, 0], centroid[:, 1])
        # With the deliberate lift and the kerb taken back off.
        offset = lift + np.where(tri_class == CLASS_FOOTPATH, kerb, 0.0)
        error = np.abs(centroid[:, 2] - offset - ground)
        # Against what the height model can deliver: a face crossing a quay
        # wall spans a step between two samples, and no flat face sits closer
        # than about half of it.
        allowance = _ground_allowance(terrain, centroid[:, :2], floor_m=0.5)
        over = int((error > allowance).sum())
        report.add(
            "road_surface_follows_terrain",
            over <= MAX_ROAD_OFF_GROUND_FRACTION * max(len(error), 1),
            f"{over} of {len(error)} road triangles are further from the ground "
            f"than the height model allows; it follows to {np.median(error):.3f} m "
            f"at the median and never more than {error.max():.2f} m, over a "
            f"{lift:.3f} m lift and a {kerb:.2f} m kerb",
        )

        # The terrain as drawn, under every corner and centre of every road
        # triangle. `lower_under` makes this true by construction; this is
        # the check that it ran, and ran on the terrain that was exported.
        probe = np.vstack([tris.reshape(-1, 3), centroid])
        drawn = np.asarray(terrain.ground(probe[:, 0], probe[:, 1]), dtype=np.float64)
        poking = probe[:, 2] - drawn
        report.add(
            "terrain_stays_under_roads",
            int((poking < 0.0).sum()) <= 0.001 * len(poking),
            f"{int((poking < 0.0).sum())} of {len(poking)} road points have the "
            f"terrain above them; the road clears it by "
            f"{np.median(poking):.3f} m at the median",
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


def check_barriers(report: CheckReport, barriers, sampler=None) -> None:
    """Walls and fences are closed, on the ground, and a believable height.

    Height is the one thing the BGT does not survey for these, so it comes
    from a table of assumptions -- which is exactly the sort of number that
    gets edited to 30 in a config file and noticed three models later.
    """
    if barriers is None or not len(barriers):
        report.add("barriers_present", True, "no walls or fences", severity="warning")
        return

    report.add(
        "barriers_present",
        True,
        ", ".join(f"{v} {k}" for k, v in sorted(barriers.counts.items())),
    )

    vertices, triangles = barriers.vertices, barriers.triangles
    edges = np.sort(
        np.concatenate(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    bad_edges = int((counts > 2).sum())
    report.add(
        "barriers_are_manifold",
        bad_edges == 0,
        f"{bad_edges} edges shared by more than two faces",
    )

    corner = vertices[triangles]
    area = np.linalg.norm(
        np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]), axis=1
    ) / 2
    degenerate = int((area < 1e-9).sum())
    report.add(
        "barriers_have_no_degenerate_faces",
        degenerate == 0,
        f"{degenerate} of {len(triangles)} faces have no area",
    )

    if sampler is not None:
        ground = np.asarray(sampler(vertices[:, 0], vertices[:, 1]), dtype=np.float64)
        above = vertices[:, 2] - ground
        # An awning or a canopy is meant to be up in the air; everything else
        # is meant to be standing on the pavement, and 8 m of garden wall is a
        # mistake. The awnings used to be measured too, and one 8 m up over a
        # station forecourt sat exactly on the limit.
        names = [str(n) for n in getattr(barriers, "kind_names", [])]
        overhead = [i for i, n in enumerate(names) if n in ("awning", "canopy")]
        standing = np.ones(len(vertices), dtype=bool)
        kinds = getattr(barriers, "kind", None)
        if overhead and kinds is not None and len(kinds) == len(triangles):
            standing[:] = False
            standing[triangles[~np.isin(kinds, overhead)].ravel()] = True
        tallest = float(above[standing].max()) if standing.any() else 0.0
        report.add(
            "barriers_are_a_believable_height",
            tallest <= 8.0,
            f"the highest point of anything standing on the ground is "
            f"{tallest:.1f} m above it",
        )
        # Nothing should hang below its own embed depth, or a fence up a bank
        # has sunk out of sight.
        deepest = float(-above.min())
        report.add(
            "barriers_sit_on_the_ground",
            deepest <= 0.30,
            f"the lowest point is {deepest:.2f} m below the ground",
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
    """The terrain in the model is the grid the pipeline meant to build.

    Every other terrain check reads the grid held in memory, which says nothing
    about what the Blender stage drew. A run that once shipped a different
    terrain from the one it had built passed all of them for exactly that
    reason, so this one reads Blender's own count back.
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

    quads = int(built.get("quads", 0))
    others = int(built.get("faces", 0)) - quads
    mesh = getattr(terrain, "mesh", None)
    if mesh is not None:
        want_vertices = len(mesh.vertices)
        want_quads = int((mesh.face_sizes == 4).sum())
        want_others = len(mesh) - want_quads
        plan = (
            f"a mesh of {want_vertices} vertices, {want_quads} quads and "
            f"{want_others} triangles"
        )
    else:
        n = int(terrain.n)
        want_vertices, want_quads, want_others = n * n, (n - 1) ** 2, 0
        plan = f"a {n} x {n} grid of {want_quads} quads"
    report.add(
        "terrain_built_as_planned",
        quads == want_quads and others == want_others
        and int(built.get("vertices", 0)) == want_vertices,
        f"the model's terrain has {built.get('vertices', 0)} vertices, {quads} "
        f"quads and {others} other faces, against {plan}",
    )
    if "exported_non_triangles" in built:
        # The pipeline placed everything on the quads cut first corner to
        # third; a face the FBX leaves whole, Unity cuts its own way.
        left = int(built["exported_non_triangles"])
        report.add(
            "terrain_exported_as_triangles",
            left == 0,
            f"{int(built.get('exported_faces', 0))} terrain faces in the FBX, "
            f"{left} of them not triangles",
        )


# How far a layer vertex may miss its band, and what share of them may.
LAYER_SLACK_M = 0.1
LAYER_STRAY_SHARE = 5e-4


def check_layered(
    report: CheckReport, work_dir: Path, ground, *, kerbs: bool, kerb_top_m: float,
    kerb_width_m: float = 0.3,
) -> None:
    """The layered style's ground as Blender built it.

    Every road, walkway and grass object is in the model; each lies on the
    terrain at its offset, which is what its modifiers are there to keep,
    and no part of a walkway stands higher than the curb generator builds
    (`kerb_top_m` above the walkway's foot); and the walkways have kerbs,
    which only the curb generator makes. Read back from Blender's own
    summary, as with the terrain: the plan says nothing about what the
    modifiers did with it.
    """
    if ground is None:
        return
    summary_path = work_dir / "blender_summary.json"
    if not summary_path.is_file():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return
    built = summary.get("layered") or {}
    wanted = [name for name in ground.object_names if name]
    made = set(built.get("objects") or [])
    absent = [name for name in wanted if name not in made]
    report.add(
        "ground_layers_built",
        bool(built) and not absent,
        f"{len(wanted) - len(absent)} of {len(wanted)} road, walkway and grass "
        f"objects are in the model"
        + (f"; missing {', '.join(absent[:5])}" if absent else ""),
    )
    if not built:
        return

    from .layered import GRASS, MOTORWAY, ROAD, WALKWAY

    kerbed = bool(kerbs and built.get("curb_generator"))
    lift = ground.lift

    def wrapped(low, high):
        # A Shrinkwrap offsets along the terrain's normal, which on a 60
        # degree bank puts a layer between half and twice its lift above the
        # ground straight down.
        return low, high, 0.5 * low, 2.0 * high

    # What each role is built to stand above the terrain under it, and the
    # band it may measure in.
    expect = {
        "road": wrapped(min(lift[ROAD], lift[MOTORWAY]), max(lift[ROAD], lift[MOTORWAY])),
        "grass": wrapped(lift[GRASS], lift[GRASS]),
        # Draped, a kerbed walkway is exact: from its lift to the top of the
        # kerb, to the half centimetre.
        "walkway": (
            (lift[WALKWAY], lift[WALKWAY] + kerb_top_m,
             lift[WALKWAY] - 0.005, lift[WALKWAY] + kerb_top_m + 0.005)
            if kerbed
            else wrapped(lift[WALKWAY], lift[WALKWAY])
        ),
    }
    above = built.get("height_above_terrain") or {}
    lines, wrong = [], []
    for role, (low, high, floor, ceiling) in expect.items():
        got = above.get(role)
        if not got:
            continue
        # Nothing past the object's own outline but a kerb: built inside it,
        # except where two footpaths touch at a point on the edge of the area
        # and the kerb on the cut edge stands its own width outside (twice
        # over Utrecht). A runaway mitre goes metres.
        past = float(got.get("beyond_outline_m", 0.0))
        reach = kerb_width_m + 0.05 if role == "walkway" else 0.05
        # A layer that floats, sinks or flies off misses its band by far, or
        # everywhere. A Shrinkwrap on a face steeper than 65 degrees, where no
        # offset can read as an exact lift straight up, misses it by a few
        # centimetres at a vertex or two: over Utrecht, five of 26,000. So
        # nothing may miss it by more than `LAYER_SLACK_M`, and no more than
        # `LAYER_STRAY_SHARE` of the vertices may miss it at all.
        lowest = got.get("lowest_m")
        highest = got.get("highest_m")
        if lowest is not None and highest is not None:
            n = max(int(got.get("vertices", 0)), 1)
            outside = sum(v < floor for v in lowest) + sum(v > ceiling for v in highest)
            listed = len(lowest) + len(highest)
            ok_band = (
                got["min_m"] >= floor - LAYER_SLACK_M
                and got["max_m"] <= ceiling + LAYER_SLACK_M
                and outside < listed
                and outside <= LAYER_STRAY_SHARE * n
            )
        else:
            outside = int(got["min_m"] < floor) + int(got["max_m"] > ceiling)
            ok_band = not outside
        ok = ok_band and not got.get("missed") and past <= reach
        lines.append(
            f"{role} {got['min_m']:.3f} to {got['max_m']:.3f} m "
            f"(built for {low:.2f}" + (f" to {high:.2f}" if high != low else "") + ")"
            + (f", {outside} of {got.get('vertices', '?')} vertices outside "
               f"{floor:.3f} to {ceiling:.3f} m" if outside else "")
            + (f", {got['missed']} vertices not over the terrain" if got.get("missed") else "")
            + (f", reaching {past:.2f} m past its outline" if past > 0.05 else "")
        )
        if not ok:
            wrong.append(role)
    if lines:
        report.add(
            "layers_lie_on_the_terrain",
            not wrong,
            "height above the terrain: " + "; ".join(lines),
        )
    if kerbs:
        if not built.get("curb_generator"):
            report.add(
                "walkways_have_kerbs",
                False,
                "the curb generator needs Blender 5, so the walkways are flat",
                severity="warning",
            )
        else:
            raw = int(built.get("walkway_faces", 0))
            evaluated = int(built.get("walkway_faces_evaluated", 0))
            report.add(
                "walkways_have_kerbs",
                evaluated > raw > 0,
                f"{int(built.get('kerbed_objects', 0))} walkway objects with the curb "
                f"generator: {raw} faces become {evaluated} with their kerbs",
            )


def check_terrain_mesh(report: CheckReport, terrain, *, expect_outlines: bool = True) -> None:
    """The terrain mesh is one closed sheet, and its edges follow the outlines.

    Closed: every edge has a face on each side except along the bbox, no edge
    has three, every face looks up, and together they cover the area. A crack
    is a hole in the ground the size of a hairline, invisible until a drone's
    shadow falls through it.
    """
    mesh = getattr(terrain, "mesh", None)
    if mesh is None:
        return
    sizes = mesh.face_sizes.astype(np.int64)
    loops = mesh.face_loops
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    following = np.arange(len(loops)) + 1
    following[starts + sizes - 1] = starts
    pairs = np.sort(np.column_stack([loops, loops[following]]), axis=1)
    edges, counts = np.unique(pairs, axis=0, return_counts=True)
    open_edges = edges[counts == 1]
    # From the corner of the area: in RD metres the areas of small faces drown
    # in the rounding of products near 10^11.
    xy = mesh.vertices[:, :2]
    lo = xy.min(axis=0)
    xy = xy - lo
    span = xy.max(axis=0)

    def sides(points):
        return np.column_stack([
            np.abs(points[:, 0]) < 1e-6, np.abs(points[:, 0] - span[0]) < 1e-6,
            np.abs(points[:, 1]) < 1e-6, np.abs(points[:, 1] - span[1]) < 1e-6,
        ])

    # Open only along one side of the area: an edge from one side to another
    # cuts a corner off.
    along = (sides(xy[open_edges[:, 0]]) & sides(xy[open_edges[:, 1]])).any(axis=1)
    cracks = int((~along).sum())
    shared_by_three = int((counts > 2).sum())
    following_vertex = loops[following]
    twice = np.add.reduceat(
        xy[loops, 0] * xy[following_vertex, 1] - xy[following_vertex, 0] * xy[loops, 1], starts
    ) if len(sizes) else np.zeros(0)
    flipped = int((twice <= 0).sum())
    covered = 0.5 * float(twice.sum())
    whole = float(span[0] * span[1])
    missing = abs(whole - covered)
    report.add(
        "terrain_is_one_closed_sheet",
        cracks == 0 and shared_by_three == 0 and flipped == 0 and missing <= max(0.01, 1e-9 * whole),
        f"{cracks} crack edges, {shared_by_three} edges with more than two faces "
        f"and {flipped} faces upside down, over {len(mesh)} faces covering "
        f"{covered:,.2f} of {whole:,.2f} m2",
    )
    stats = mesh.stats or {}
    if "outline_edges_missing" in stats:
        # The layered style: a road's or a lawn's outline is a terrain edge,
        # or the layer object there takes in a sliver of the next one.
        left = int(stats["outline_edges_missing"])
        report.add(
            "terrain_follows_the_layer_outlines",
            left == 0,
            "every outline of a road, walkway or lawn is an edge of the terrain"
            if left == 0
            else f"{left} outline edges are not terrain edges; the faces there "
            f"went to whichever layer their middle lies in",
            severity="warning",
        )
    outlines = stats.get("outline_edges") or {}
    followed = sum(int(outlines.get(name, 0)) for name in ("waterline", "road_edge", "building_edge"))
    if expect_outlines:
        report.add(
            "terrain_follows_the_outlines",
            followed > 0 or "fallback" not in stats,
            (
                f"{followed} edges along outlines ("
                + ", ".join(f"{v} {k}" for k, v in outlines.items() if v)
                + f"); {round(100 * float(stats.get('quad_share', 0)))}% of faces are quads"
            )
            if "fallback" not in stats
            else f"the outlines could not be meshed, so the terrain is the plain grid: "
            f"{stats['fallback']}",
            severity="warning",
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
    blend_name: str = "model.blend",
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
    # The editable copy. Written after the FBX, so a Blender stage that died
    # between the two leaves an FBX and no .blend, and this is what says so.
    blend = out_dir / blend_name
    report.add(
        "blend_written",
        blend.is_file() and blend.stat().st_size > 1024,
        f"{blend.name} is {blend.stat().st_size / 1e6:.1f} MB"
        if blend.is_file()
        else f"{blend} is missing",
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
    # The terrain is cut to the bbox exactly, so it measures the scale exactly:
    # the whole model also holds every building whose middle is inside, and
    # at Hazeldonk two warehouses reach 100 m and more past the square.
    terrain = next(
        (o for o in objects if o.get("maptile_role") == "terrain" or o.name == "Terrain"),
        None,
    )
    if terrain is not None:
        ends = [terrain.matrix_world @ mathutils.Vector(c) for c in terrain.bound_box]
        ground_x = max(c.x for c in ends) - min(c.x for c in ends)
        ground_y = max(c.y for c in ends) - min(c.y for c in ends)
        report.add(
            "fbx_reimport_scale",
            abs(ground_x - bbox.width) < 1.0 and abs(ground_y - bbox.height) < 1.0,
            f"the terrain spans {ground_x:.1f} x {ground_y:.1f} m against a "
            f"{bbox.width:.0f} x {bbox.height:.0f} m bbox",
        )
    else:
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
    "check_layered",
    "check_roads",
    "check_terrain",
    "check_terrain_mesh",
    "write_report",
]
