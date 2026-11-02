"""Ground surfaces: water bodies and land cover, from the BGT.

Water is the reason this module exists. Lidar does not reflect off water, so the
AHN DTM is 73% empty over a canal against 47% on land, and the few returns that
do come back scatter over several metres. The gap filler then interpolates from
the banks inward, which turns every canal into a bulge with random lumps in it.
The fix is to stop guessing: take the water outlines from the BGT, work out one
level per body, sink the bed below it and lay a flat surface on top.

Land cover comes from the same three BGT terrain collections. It gives each part
of the ground a class, which the aerial photo cannot: a photo of grass and a
photo of asphalt are just pixels. The class map is exported for Unity to use,
and drives a light per-surface detail pass over the aerial, because a 25 cm
ortho is mushy at street level whatever its pixel count.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bgt import fetch_current, polygon_rings, rasterize_rings, ring_area
from .geo import BBox, build_grid_coords

LOG = logging.getLogger(__name__)

# Class ids burned into the land-cover raster. 0 means "no BGT class here",
# which is normal: the BGT does not tile the whole country edge to edge.
CLASS_NONE = 0
CLASS_ROAD = 1
CLASS_GREEN = 2
CLASS_PAVED = 3
CLASS_UNPAVED = 4
CLASS_WATER = 5
# The BGT knows what every road surface is for and what it is made of, and a
# Dutch street is unrecognisable without that: the carriageway is brick as
# often as asphalt, the cycle path beside it is red, and the footpath is grey
# tiles. Collapsing all of it into one "road" class threw away the single most
# characteristic thing about the ground here.
CLASS_ROAD_BRICK = 6
CLASS_CYCLE = 7
CLASS_FOOTPATH = 8
CLASS_PARKING = 9
CLASS_TRANSIT = 10
CLASS_QUAY = 11
# The vertical face of a kerb, and the skirt along the outer edge of the road
# network. Never painted into the land-cover raster: it has no area from above.
CLASS_KERB = 12

CLASS_NAMES = {
    CLASS_NONE: "unclassified",
    CLASS_ROAD: "road_asphalt",
    CLASS_GREEN: "green",
    CLASS_PAVED: "paved",
    CLASS_UNPAVED: "unpaved",
    CLASS_WATER: "water",
    CLASS_ROAD_BRICK: "road_brick",
    CLASS_CYCLE: "cycle_path",
    CLASS_FOOTPATH: "footpath",
    CLASS_PARKING: "parking",
    CLASS_TRANSIT: "transit_lane",
    CLASS_QUAY: "quay",
    CLASS_KERB: "kerb",
}

# Tint and grain used to give each surface class some texture of its own on top
# of the photo. Kept subtle: the aerial still has to be the thing you see.
# `cells` is metres per grain feature, so a smaller number is a finer grain.
CLASS_DETAIL = {
    CLASS_ROAD: {"tint": (74, 76, 80), "grain": 0.55, "cells": 3.0},
    CLASS_GREEN: {"tint": (86, 116, 62), "grain": 0.85, "cells": 1.1},
    CLASS_PAVED: {"tint": (128, 124, 118), "grain": 0.60, "cells": 1.6},
    CLASS_UNPAVED: {"tint": (132, 118, 96), "grain": 0.70, "cells": 1.3},
    CLASS_ROAD_BRICK: {"tint": (118, 92, 78), "grain": 0.70, "cells": 6.0},
    # Dutch cycle paths are red asphalt, and nothing else in the street is.
    CLASS_CYCLE: {"tint": (124, 66, 54), "grain": 0.45, "cells": 3.0},
    CLASS_FOOTPATH: {"tint": (140, 138, 134), "grain": 0.55, "cells": 5.0},
    CLASS_PARKING: {"tint": (96, 94, 94), "grain": 0.60, "cells": 4.0},
    CLASS_TRANSIT: {"tint": (82, 80, 84), "grain": 0.50, "cells": 2.4},
    CLASS_QUAY: {"tint": (120, 114, 106), "grain": 0.65, "cells": 2.0},
}

# BGT `functie` on a wegdeel, mapped to what the surface actually is. Matched
# by substring because the vocabulary is finer than this ("rijbaan lokale weg",
# "rijbaan regionale weg", "voetpad op trap" and so on).
ROAD_FUNCTIONS: tuple[tuple[str, int], ...] = (
    ("parkeervlak", CLASS_PARKING),
    ("fietspad", CLASS_CYCLE),
    ("voetpad", CLASS_FOOTPATH),
    ("voetgangersgebied", CLASS_FOOTPATH),
    ("ov-baan", CLASS_TRANSIT),
    ("spoorbaan", CLASS_TRANSIT),
    ("rijbaan", CLASS_ROAD),
    ("inrit", CLASS_ROAD),
)

# Materials that make a carriageway brick rather than asphalt. Only these two
# classes change with the material; a cycle path is red whatever it is made of.
BRICK_MATERIALS = ("klinker", "sierbestrating", "betonstraatstenen", "tegels")


# Broad surfaces first, the things that sit on top of them last. A parking bay
# is cut out of a carriageway and a footpath runs along its edge, so both have
# to win where the polygons overlap.
PAINT_ORDER: tuple[int, ...] = (
    CLASS_ROAD,
    CLASS_ROAD_BRICK,
    CLASS_TRANSIT,
    CLASS_PARKING,
    CLASS_CYCLE,
    CLASS_FOOTPATH,
)


def _paint_rank(code: int) -> int:
    return PAINT_ORDER.index(code) if code in PAINT_ORDER else -1


def road_class(functie: str, material: str) -> int:
    """Which surface class one BGT road part belongs to."""
    functie = (functie or "").strip().lower()
    material = (material or "").strip().lower()

    code = CLASS_ROAD
    for token, mapped in ROAD_FUNCTIONS:
        if token in functie:
            code = mapped
            break

    # A brick carriageway is the default residential street here, not an
    # exception, so it is worth telling apart from asphalt.
    if code in (CLASS_ROAD, CLASS_PARKING) and any(
        token in material for token in BRICK_MATERIALS
    ):
        return CLASS_ROAD_BRICK
    return code


@dataclass
class WaterBody:
    """One water polygon with the level its surface sits at."""

    rings: list[np.ndarray]
    level_nap: float
    area_m2: float
    measured: bool  # False when AHN gave nothing usable and banks were used


@dataclass
class RoadPart:
    """One road polygon, kept as geometry rather than only as raster class."""

    rings: list[np.ndarray]
    surface_class: int
    # BGT relatieve_hoogteligging. Above zero the road is on a bridge, and
    # draping it on the terrain drops the carriageway off its own deck.
    level: int = 0


@dataclass
class SurfaceSet:
    water: list[WaterBody] = field(default_factory=list)
    roads: list[RoadPart] = field(default_factory=list)
    # Filled once the terrain is known, since the road surface follows it.
    road_tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    road_tri_class: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    # BGT level per road triangle, so a check can tell a carriageway that
    # should hug the ground from one riding a bridge deck.
    # The deliberate lift the roads were given, so anything judging the drape
    # can take it back off rather than counting it as error.
    road_lift_m: float = 0.03
    road_tri_level: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    # The road network as faces: quads across every street, kerbs and all.
    # `road_tris` is the same surface in triangles, for the checks.
    road_mesh: object | None = None
    class_grid: np.ndarray | None = None  # (n, n) uint8 on the terrain grid
    # (n, n) float, NaN off water: how far down the bed goes under each body.
    water_bed: np.ndarray | None = None
    counts: dict[str, int] = field(default_factory=dict)
    stats_by_collection: dict[str, str] = field(default_factory=dict)

    @property
    def water_area_m2(self) -> float:
        return float(sum(body.area_m2 for body in self.water))

    def stats(self) -> dict:
        out: dict = {
            "water_bodies": len(self.water),
            "water_area_m2": round(self.water_area_m2, 1),
            "water_levels_measured": sum(1 for b in self.water if b.measured),
        }
        if self.class_grid is not None:
            total = self.class_grid.size
            out["land_cover_fractions"] = {
                name: round(float((self.class_grid == code).sum()) / total, 4)
                for code, name in CLASS_NAMES.items()
            }
        out.update(self.counts)
        return out


def clip_ring_to_bbox(ring: np.ndarray, bbox: BBox) -> np.ndarray:
    """Clip a polygon ring to the bbox by Sutherland-Hodgman.

    Water polygons come back whole, and a canal can run a long way past the
    area: one Utrecht outline stretched the model 300 m beyond its terrain.
    Buildings are kept whole because cutting one opens it up, but a water
    surface is flat, so trimming it costs nothing and keeps the model bounded.
    The bbox is convex, which is exactly the case this algorithm handles.
    """
    edges = (
        ("x", bbox.xmin, True),
        ("x", bbox.xmax, False),
        ("y", bbox.ymin, True),
        ("y", bbox.ymax, False),
    )
    points = np.asarray(ring, dtype=np.float64)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]

    for axis, limit, keep_greater in edges:
        if len(points) == 0:
            return np.zeros((0, 2))
        index = 0 if axis == "x" else 1

        def inside(point: np.ndarray) -> bool:
            return (
                point[index] >= limit if keep_greater else point[index] <= limit
            )

        output: list[np.ndarray] = []
        for i in range(len(points)):
            current = points[i]
            previous = points[i - 1]
            current_in = inside(current)
            previous_in = inside(previous)

            if current_in != previous_in:
                span = current[index] - previous[index]
                if abs(span) > 1e-12:
                    t = (limit - previous[index]) / span
                    output.append(previous + t * (current - previous))
            if current_in:
                output.append(current)
        points = np.asarray(output) if output else np.zeros((0, 2))

    if len(points) < 3:
        return np.zeros((0, 2))
    return np.vstack([points, points[0]])


def _water_level(
    rings: list[np.ndarray],
    raster: np.ndarray,
    bounds: tuple[float, float, float, float],
    nodata_cutoff: float,
) -> tuple[float, bool]:
    """Work out the surface level of one water body.

    Where lidar did return something inside the polygon, a low percentile of
    those samples is the water surface: the high ones are boats, bridges and
    bank spill, and taking the median would ride up on them. With too few
    returns, the ring vertices give the bank height and the water sits just
    below that.
    """
    mask = rasterize_rings([rings], bounds, raster.shape)
    samples = raster[mask]
    samples = samples[np.isfinite(samples) & (samples < nodata_cutoff)]

    if len(samples) >= 25:
        return float(np.percentile(samples, 20)), True

    # Fall back to the banks: sample the raster along the outline.
    left, bottom, right, top = bounds
    rows, cols = raster.shape
    cell_x = (right - left) / cols
    cell_y = (top - bottom) / rows
    outline = rings[0]
    col = np.clip(((outline[:, 0] - left) / cell_x).astype(int), 0, cols - 1)
    row = np.clip(((top - outline[:, 1]) / cell_y).astype(int), 0, rows - 1)
    bank = raster[row, col]
    bank = bank[np.isfinite(bank) & (bank < nodata_cutoff)]
    if len(bank):
        return float(np.percentile(bank, 25) - 0.4), False
    return float("nan"), False


def build_surfaces(
    bbox: BBox,
    work_dir: Path,
    *,
    surfaces_cfg: dict,
    terrain,
    deck_sampler=None,
    network=None,
) -> SurfaceSet:
    """Fetch water and land cover, classify the terrain grid, build the roads.

    ``deck_sampler`` gives heights for road parts the BGT marks as being on a
    bridge; without one they are draped on the ground like everything else.
    ``network`` is the NWB road network the cross-sections are cut along; see
    `roads`. Without it every road is still built, from centrelines drawn down
    the middle of the road itself.
    """
    import rasterio

    from .elevation import NODATA_CUTOFF

    result = SurfaceSet()
    n = terrain.n
    grid_bounds = (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)

    with rasterio.open(terrain.geotiff_path) as dataset:
        raster = dataset.read(1).astype(np.float64)
        raster_bounds = (
            dataset.bounds.left,
            dataset.bounds.bottom,
            dataset.bounds.right,
            dataset.bounds.top,
        )

    fetch_kwargs = {
        "page_limit": int(surfaces_cfg["page_limit"]),
        "timeout": float(surfaces_cfg["timeout_s"]),
        "max_retries": int(surfaces_cfg["max_retries"]),
        "max_pages": int(surfaces_cfg["max_pages"]),
    }

    # ---- water -----------------------------------------------------------
    if bool(surfaces_cfg["water"]):
        features, stats = fetch_current("waterdeel", bbox, **fetch_kwargs)
        result.stats_by_collection["waterdeel"] = stats.summary()

        for feature in features:
            for rings in polygon_rings(feature.get("geometry")):
                # Level is read before clipping, so a body that mostly lies
                # outside the area still gets its true surface height.
                level, measured = _water_level(
                    rings, raster, raster_bounds, NODATA_CUTOFF
                )
                if not np.isfinite(level):
                    continue

                clipped = [clip_ring_to_bbox(ring, bbox) for ring in rings]
                clipped = [ring for ring in clipped if len(ring) >= 3]
                if not clipped:
                    continue
                rings = clipped

                area = ring_area(rings[0]) - sum(ring_area(r) for r in rings[1:])
                result.water.append(
                    WaterBody(
                        rings=rings,
                        level_nap=level,
                        area_m2=float(max(area, 0.0)),
                        measured=measured,
                    )
                )

        if result.water:
            levels = np.array([b.level_nap for b in result.water])
            LOG.info(
                "%d water bodies covering %.0f m2, levels %.2f to %.2f m NAP",
                len(result.water),
                result.water_area_m2,
                levels.min(),
                levels.max(),
            )

    # ---- land cover ------------------------------------------------------
    if bool(surfaces_cfg["land_cover"]):
        xs, ys = build_grid_coords(bbox, n)
        # Row 0 of the class grid is the southern edge, matching the terrain
        # grid, so it is built north-up and flipped at the end.
        class_grid = np.zeros((n, n), dtype=np.uint8)
        flip_bounds = (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)

        # Painted in this order so the more specific class wins the overlap.
        layers = [
            ("onbegroeidterreindeel", CLASS_PAVED, "fysiek_voorkomen"),
            ("begroeidterreindeel", CLASS_GREEN, "fysiek_voorkomen"),
            ("ondersteunendwaterdeel", CLASS_QUAY, "type"),
            ("wegdeel", CLASS_ROAD, "fysiek_voorkomen"),
        ]
        for collection, code, class_field in layers:
            features, stats = fetch_current(
                collection, bbox, class_field=class_field, **fetch_kwargs
            )
            result.stats_by_collection[collection] = stats.summary()
            result.counts[f"{collection}_count"] = stats.current_features

            # Rings gathered per class, so one collection can paint several.
            by_class: dict[int, list[list[np.ndarray]]] = {}
            want_geometry = collection == "wegdeel" and bool(
                surfaces_cfg.get("road_geometry", True)
            )
            for feature in features:
                properties = feature.get("properties") or {}
                physical = str(properties.get("fysiek_voorkomen") or "")

                if collection == "wegdeel":
                    target = road_class(
                        str(properties.get("functie") or ""),
                        str(properties.get("plus_fysiek_voorkomen") or physical),
                    )
                elif code == CLASS_PAVED and "onverhard" in physical:
                    target = CLASS_UNPAVED
                else:
                    target = code

                groups = polygon_rings(feature.get("geometry"))
                by_class.setdefault(target, []).extend(groups)

                if want_geometry:
                    # A tunnel drawn on the surface is simply wrong. Bridges
                    # are kept: the DTM under a canal is interpolated up to
                    # bank level, which is about where a low Dutch bridge is.
                    try:
                        level = int(properties.get("relatieve_hoogteligging") or 0)
                    except (TypeError, ValueError):
                        level = 0
                    if level < 0:
                        continue
                    for group in groups:
                        clipped = [
                            clip_ring_to_bbox(ring, bbox) for ring in group
                        ]
                        clipped = [r for r in clipped if len(r) >= 3]
                        if clipped:
                            result.roads.append(
                                RoadPart(clipped, target, level)
                            )

            # Explicit order, not dict order: road parts overlap at junctions
            # and kerbs, and whichever painted last would otherwise depend on
            # the order the API happened to return features in.
            for target in sorted(by_class, key=_paint_rank):
                rings = by_class[target]
                if rings:
                    rasterize_rings(
                        rings,
                        flip_bounds,
                        class_grid.shape,
                        out=class_grid,
                        value=target,
                    )

        # Water last: it wins over anything a road or terrain polygon claimed.
        if result.water:
            rasterize_rings(
                [b.rings for b in result.water],
                flip_bounds,
                class_grid.shape,
                out=class_grid,
                value=CLASS_WATER,
            )

        # rasterize_rings works north-up; the terrain grid is south-up.
        result.class_grid = np.flipud(class_grid)

    # Bed level per water body, on the terrain grid. It has to be per body:
    # levels across one area span metres, so a single bed taken from the median
    # would sit above the surface of the lowest canal and poke through it.
    if result.water:
        depth = float(surfaces_cfg["water_depth_m"])
        bed = np.full((n, n), np.nan, dtype=np.float64)
        for body in result.water:
            mask = rasterize_rings([body.rings], grid_bounds, bed.shape)
            level = body.level_nap - depth
            bed[mask] = np.where(
                np.isnan(bed[mask]), level, np.minimum(bed[mask], level)
            )
        result.water_bed = np.flipud(bed)

    if result.roads:
        from .roads import KERB_M, build_road_mesh

        result.road_lift_m = float(surfaces_cfg.get("road_lift_m", 0.03))
        mesh = build_road_mesh(
            result.roads,
            network,
            # The surveyed ground, not the terrain as drawn: the terrain is
            # pushed down under the roads afterwards, to fit them.
            ground=terrain.sample,
            lift_m=result.road_lift_m,
            kerb_m=float(surfaces_cfg.get("kerb_m", KERB_M)),
            # Already fetched above, and the only thing that knows where the
            # surface of each body is.
            water=result.water,
            water_clearance_m=float(surfaces_cfg.get("road_water_clearance_m", 0.25)),
            deck_sampler=deck_sampler,
            uv_origin=bbox.center,
        )
        result.road_mesh = mesh
        tris, owner = mesh.triangles()
        result.road_tris = tris
        result.road_tri_class = mesh.face_class[owner].astype(np.int32)
        result.road_tri_level = mesh.face_level[owner].astype(np.int32)
        on_bridges = sum(1 for part in result.roads if part.level > 0)
        if on_bridges:
            result.counts["road_parts_on_bridges"] = on_bridges
        result.counts["road_parts"] = len(result.roads)
        result.counts["road_faces"] = int(mesh.stats.get("faces", 0))
        result.counts["road_quad_share"] = float(mesh.stats.get("quad_share", 0.0))
        result.counts["road_kerb_faces"] = int(mesh.stats.get("kerb_faces", 0))
        result.counts["road_objects"] = len(mesh.object_names)

    # Outside the water block, and guarded: land cover and water are
    # independently switchable, so this ran on a None grid whenever water was
    # on and land cover was off.
    if result.class_grid is not None:
        fractions = {
            CLASS_NAMES[code]: float((result.class_grid == code).mean())
            for code in CLASS_NAMES
        }
        LOG.info(
            "land cover: %s",
            ", ".join(f"{k} {100 * v:.0f}%" for k, v in fractions.items() if v > 0.005),
        )

    save_surfaces(result, bbox, work_dir)
    return result


def triangulate_water(bodies: list[WaterBody]) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate the water outlines into ``(triangles, body index)``.

    This belongs here rather than in the Blender stage. That script may only use
    what Blender itself bundles, which is bpy and numpy: when Blender is a real
    application rather than the pip module, its Python is a different
    interpreter with none of this pipeline's dependencies on it.
    """
    import mapbox_earcut

    triangles: list[np.ndarray] = []
    owner: list[np.ndarray] = []

    for index, body in enumerate(bodies):
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in body.rings]
        rings = [r for r in rings if len(r) >= 3]
        if not rings:
            continue

        flat = np.vstack(rings)
        ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        try:
            indices = mapbox_earcut.triangulate_float64(flat, ring_ends)
        except Exception as exc:  # noqa: BLE001 - one bad outline is not fatal
            LOG.debug("could not triangulate a water body: %s", exc)
            continue
        if len(indices) < 3:
            continue

        corners = flat[np.asarray(indices, dtype=np.int64)].reshape(-1, 3, 2)
        # Carry the surface height with the geometry, as buildings do.
        with_z = np.dstack(
            [corners, np.full((len(corners), 3, 1), body.level_nap)]
        )
        triangles.append(with_z)
        owner.append(np.full(len(corners), index, dtype=np.int32))

    if not triangles:
        return np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int32)
    return np.concatenate(triangles), np.concatenate(owner)


# How far above the ground a bridge carriageway may plausibly be. Past this the
# surface model is reporting a building or a crane, not a deck.
MIN_BRIDGE_CLEARANCE_M = 0.5
MAX_BRIDGE_CLEARANCE_M = 40.0


def _robust_deck_level(points, deck_sampler, ground) -> float | None:
    """One deck height for a road part: the median of what plausibly is one."""
    readings = np.asarray(deck_sampler(points[:, 0], points[:, 1]), dtype=np.float64)
    clearance = readings - ground
    usable = readings[
        np.isfinite(readings)
        & (clearance > MIN_BRIDGE_CLEARANCE_M)
        & (clearance < MAX_BRIDGE_CLEARANCE_M)
    ]
    if len(usable) < max(3, len(readings) // 20):
        return None
    return float(np.median(usable))


def _hold_above_water(points, z, water, clearance_m: float):
    """Stop a path crossing a ditch from diving into it.

    Lidar does not reflect off water, so the bare-earth grid has a hole over
    every body and the gap filler interpolates across it. The result dips:
    measured over a square kilometre of Gelderland the filled ground mid-water
    sits 1.45 m below its own bank. A footpath draped on that follows it down
    and disappears under the surface it is supposed to cross.

    A path over water is a culvert or a little bridge, so it is held at the
    body's own surface level plus a clearance. Not pinned to it -- a real
    bridge rides higher, and anything already above the water is left alone.
    """
    if not water or clearance_m is None:
        return z

    out = np.asarray(z, dtype=np.float64).copy()
    px, py = points[:, 0], points[:, 1]
    for body in water:
        inside = np.zeros(len(points), dtype=bool)
        for ring in body.rings:
            ring = np.asarray(ring, dtype=np.float64)[:, :2]
            if len(ring) < 3:
                continue
            ax, ay = ring[:-1, 0], ring[:-1, 1]
            bx, by = ring[1:, 0], ring[1:, 1]
            for i in range(len(ax)):
                if by[i] == ay[i]:
                    continue
                straddles = (ay[i] > py) != (by[i] > py)
                inside ^= straddles & (
                    px < (bx[i] - ax[i]) * (py - ay[i]) / (by[i] - ay[i]) + ax[i]
                )
        if inside.any():
            floor = body.level_nap + clearance_m
            out[inside] = np.maximum(out[inside], floor)
    return out


def save_surfaces(surfaces: SurfaceSet, bbox: BBox, work_dir: Path) -> Path:
    """Write water outlines and the class grid for the Blender stage."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "surfaces.npz"

    # Rings are ragged, so they go in flattened with an index of where each one
    # starts rather than as an object array.
    ring_points: list[np.ndarray] = []
    ring_offsets: list[int] = [0]
    ring_body: list[int] = []
    body_levels: list[float] = []

    for index, body in enumerate(surfaces.water):
        body_levels.append(body.level_nap)
        for ring in body.rings:
            ring_points.append(ring)
            ring_offsets.append(ring_offsets[-1] + len(ring))
            ring_body.append(index)

    water_tris, water_tri_body = triangulate_water(surfaces.water)

    np.savez_compressed(
        path,
        # Triangulated here, not in the Blender stage: that script only has
        # what Blender bundles, and a real Blender install has its own Python.
        water_tris=water_tris,
        water_tri_body=water_tri_body,
        water_points=(
            np.vstack(ring_points) if ring_points else np.zeros((0, 2))
        ),
        water_ring_offsets=np.asarray(ring_offsets, dtype=np.int64),
        water_ring_body=np.asarray(ring_body, dtype=np.int32),
        water_levels=np.asarray(body_levels, dtype=np.float64),
        # Roads leave as their own geometry, one object per street, so they
        # can carry their own materials and sit on their own layer in Unity.
        **(surfaces.road_mesh.to_npz_fields() if surfaces.road_mesh is not None else {}),
        road_lift_m=surfaces.road_lift_m,
        class_grid=(
            surfaces.class_grid
            if surfaces.class_grid is not None
            else np.zeros((0, 0), dtype=np.uint8)
        ),
        water_bed=(
            surfaces.water_bed
            if surfaces.water_bed is not None
            else np.zeros((0, 0), dtype=np.float64)
        ),
        bbox=np.asarray(bbox.as_list(), dtype=np.float64),
    )
    LOG.info("wrote %s", path)
    return path


def write_land_cover(
    surfaces: SurfaceSet, bbox: BBox, out_dir: Path
) -> list[Path]:
    """Export the class map as a PNG plus a legend, for use in Unity."""
    if surfaces.class_grid is None:
        return []
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    # Row 0 of the grid is the south edge; PNG rows run north-down.
    image = Image.fromarray(np.flipud(surfaces.class_grid), mode="L")
    png = out_dir / "landcover.png"
    image.save(png)

    legend = out_dir / "landcover.json"
    legend.write_text(
        json.dumps(
            {
                "note": (
                    "Greyscale value is the surface class. Row 0 is the north "
                    "edge; the image covers bbox_rd exactly, like aerial.png."
                ),
                "bbox_rd": bbox.as_list(),
                "size_px": list(surfaces.class_grid.shape),
                "classes": {str(code): name for code, name in CLASS_NAMES.items()},
                "source": "BGT wegdeel, begroeidterreindeel, onbegroeidterreindeel, waterdeel",
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    LOG.info("wrote %s and %s", png.name, legend.name)
    return [png, legend]


# The noise fields are smooth by construction, so generating them larger than
# this buys nothing: they are tiled across the photo instead.
DETAIL_NOISE_PX = 512
# Features across the coarse field. The fine field carries three times as many,
# matching what the per-class fields used to do.
DETAIL_NOISE_CELLS = 64
# Rows blended at a time. Peak memory is this many rows of float32 RGB rather
# than the whole photo in float64, which is the difference between 90 MB and
# 6 GB on a 16k image.
DETAIL_STRIP_PX = 512


def _sample_tiled(
    field: np.ndarray, rows: np.ndarray, columns: np.ndarray, scale: float
) -> np.ndarray:
    """Read a tiling noise field over an image block, repeating it as needed.

    ``scale`` is how many image pixels one field pixel covers, so the grain
    keeps its size on the ground however large the area is.
    """
    size = field.shape[0]
    step = max(scale, 1e-6)
    row_index = np.floor(rows / step).astype(np.int64) % size
    col_index = np.floor(columns / step).astype(np.int64) % size
    return field[np.ix_(row_index, col_index)]


def blend_surface_detail(
    aerial_path: Path,
    surfaces: SurfaceSet,
    bbox: BBox,
    *,
    strength: float = 0.22,
    seed: int = 99,
) -> bool:
    """Mix a per-class grain into the aerial photo, in place.

    An ortho is flown at 8 cm and delivered as JPEG, so close up it is mushy
    whatever resolution it is resampled to: there is no detail left to resolve.
    Adding grain matched to what each surface actually is puts high-frequency
    texture back where the photo has none, without touching its colour or
    structure. Kept low: the photo still has to be the thing you see.
    """
    if surfaces.class_grid is None or strength <= 0:
        return False

    from PIL import Image

    from .facade import _value_noise
    from .imagery import _allow_large_images

    _allow_large_images()
    rng = np.random.default_rng(seed)

    with Image.open(aerial_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    metres_per_px = bbox.width / width

    # Two noise fields, generated once at a bounded size and reused by every
    # class. They tile, so they are repeated across the image rather than
    # generated to fit it: the field only ever holds `cells` features across,
    # so making it as large as the photo is pure waste. Generating one pair per
    # class at full size was costing 20 full-image fields — four minutes and
    # 2.7 GB apiece at 8192 px, and worse than linearly above that, which is
    # what made a large area look like a hang.
    coarse = _value_noise(
        (DETAIL_NOISE_PX, DETAIL_NOISE_PX), cells=DETAIL_NOISE_CELLS, rng=rng
    )
    fine = _value_noise(
        (DETAIL_NOISE_PX, DETAIL_NOISE_PX), cells=DETAIL_NOISE_CELLS * 3, rng=rng
    )

    classes = np.flipud(surfaces.class_grid)
    present = {
        code for code in CLASS_DETAIL if bool((surfaces.class_grid == code).any())
    }
    if not present:
        return False

    columns = np.arange(width)
    col_class = (columns * classes.shape[1] // width).clip(0, classes.shape[1] - 1)

    strips = -(-height // DETAIL_STRIP_PX)
    LOG.info(
        "blending detail for %d surface classes into %d x %d px, in %d strips",
        len(present),
        width,
        height,
        strips,
    )

    for strip_index in range(strips):
        top = strip_index * DETAIL_STRIP_PX
        bottom = min(height, top + DETAIL_STRIP_PX)
        rows = np.arange(top, bottom)

        block = np.asarray(image.crop((0, top, width, bottom)), dtype=np.float32)
        row_class = (rows * classes.shape[0] // height).clip(0, classes.shape[0] - 1)
        class_map = classes[np.ix_(row_class, col_class)]

        for code in present:
            mask = class_map == code
            if not mask.any():
                continue
            detail = CLASS_DETAIL[code]

            # Feature size held in metres rather than as a fraction of the
            # image, so the grain stays the same size on the ground whether the
            # area is one kilometre or twenty. One feature spans
            # DETAIL_NOISE_PX / DETAIL_NOISE_CELLS pixels of the field, and has
            # to come out as detail["cells"] metres on the ground.
            feature_px = DETAIL_NOISE_PX / DETAIL_NOISE_CELLS
            scale = detail["cells"] / (metres_per_px * feature_px)
            grain = (
                0.6 * _sample_tiled(coarse, rows, columns, scale)
                + 0.4 * _sample_tiled(fine, rows, columns, scale)
                - 0.5
            )[:, :, None]

            tint = np.asarray(detail["tint"], dtype=np.float32)
            amount = float(strength * detail["grain"])
            # Nudge toward the class tint, then modulate brightness with grain.
            blended = block * (1 - amount * 0.35) + tint * (amount * 0.35)
            blended = blended * (1.0 + amount * grain * 1.6)
            block[mask] = blended[mask]

        image.paste(
            Image.fromarray(np.clip(block, 0, 255).astype(np.uint8), mode="RGB"),
            (0, top),
        )
        # A big area spends minutes here, and silence is what made this look
        # like a hang rather than like work.
        if strips > 8 and (strip_index + 1) % 8 == 0:
            LOG.info("  detail blend %d/%d strips", strip_index + 1, strips)

    image.save(aerial_path)
    LOG.info("blended per-surface detail into %s", aerial_path.name)
    return True


__all__ = [
    "CLASS_CYCLE",
    "CLASS_DETAIL",
    "CLASS_FOOTPATH",
    "CLASS_GREEN",
    "CLASS_KERB",
    "CLASS_NAMES",
    "CLASS_NONE",
    "CLASS_PARKING",
    "CLASS_PAVED",
    "CLASS_QUAY",
    "CLASS_ROAD",
    "CLASS_ROAD_BRICK",
    "CLASS_TRANSIT",
    "CLASS_UNPAVED",
    "CLASS_WATER",
    "RoadPart",
    "SurfaceSet",
    "WaterBody",
    "blend_surface_detail",
    "build_surfaces",
    "clip_ring_to_bbox",
    "road_class",
    "triangulate_water",
    "save_surfaces",
    "write_land_cover",
]
