"""Street objects beyond the BGT: OpenStreetMap, and what Mapillary has seen.

The BGT's street furniture is a municipality's register, and an optional one:
Utrecht keeps 63 lampposts and 63 bollards round the Ganzenmarkt in it, while
Westerwolde and Súdwest-Fryslân keep nothing at all, so Bourtange and
Hindeloopen come out with bare streets. Two more sources fill in:

* **OpenStreetMap** (`osm`), where volunteers map benches, bins, bicycle
  parking, fountains, post boxes, statues and memorials, often more of them
  in a tourist village than in a city, since that is where people map;
* **Mapillary's map features** (`mapillary.fetch_map_features`): objects its
  detectors found in several photos and placed by triangulating them, which
  is how the shop signs, the bins and the bike racks of a street with photos
  but no mapper get in. Their positions are good to a few metres.

They are merged by trust: the BGT first, OSM where the BGT has nothing of the
same kind close by, Mapillary where neither has. A ground object that lands
inside a building is moved out to the wall when it is just inside, the error
a position triangulated from photos has, and dropped when it is deep in.
Shop signs are hung on the nearest wall: store and advertising signs flat
against it, banners out from it on a bracket. Mapillary also gives each
feature a direction, but over the Ganzenmarkt 88 of 163 shop signs point
more than 60 degrees off their wall's normal: that is the way the cameras
looked along the street, not the way the signs face, so it decides nothing.
Statues take their height from the survey: the surface model over the
plinth, less the ground.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .furniture import (
    KIND_ART,
    KIND_BENCH,
    KIND_BOLLARD,
    KIND_CABINET,
    KIND_LAMP,
    KIND_MONUMENT,
    KIND_NAMES,
    KIND_PICNIC,
    KIND_SIGN,
    KIND_WASTE,
)

LOG = logging.getLogger(__name__)

# Kinds the BGT has no type for, after furniture's own.
KIND_BIKE_RACK = 14
KIND_BIN = 15
KIND_SHOP_SIGN = 16  # flat against the wall
KIND_HANGING_SIGN = 17  # standing out from the wall on a bracket
KIND_MANHOLE = 18
KIND_FOUNTAIN = 19
KIND_PLANTER = 20
KIND_POSTBOX = 21
KIND_POLE = 22
KIND_STATUE = 23
KIND_TABLE = 24  # a terrace table and its chairs
KIND_PARASOL = 25
KIND_WALL_LAMP = 26  # a lantern on a bracket on the facade

NEW_KIND_NAMES = {
    KIND_BIKE_RACK: "bike rack",
    KIND_BIN: "litter bin",
    KIND_SHOP_SIGN: "shop sign",
    KIND_HANGING_SIGN: "hanging sign",
    KIND_MANHOLE: "manhole",
    KIND_FOUNTAIN: "fountain",
    KIND_PLANTER: "planter",
    KIND_POSTBOX: "post box",
    KIND_POLE: "pole",
    KIND_STATUE: "statue",
    KIND_TABLE: "terrace table",
    KIND_PARASOL: "parasol",
    KIND_WALL_LAMP: "wall lantern",
}
ALL_KIND_NAMES = {**KIND_NAMES, **NEW_KIND_NAMES}

SOURCE_BGT, SOURCE_OSM, SOURCE_MAPILLARY, SOURCE_DERIVED = 0, 1, 2, 3
SOURCE_NAMES = {
    SOURCE_BGT: "BGT", SOURCE_OSM: "OpenStreetMap", SOURCE_MAPILLARY: "Mapillary",
    SOURCE_DERIVED: "derived",
}

# Which kinds are the same thing for de-duplication, and how far apart two
# reports of it may be and still be one object. Mapillary's positions are
# triangulated from photos and land a few metres out; bollards stand 1.5 m
# apart in a row, so theirs is tighter.
FAMILY = {
    KIND_LAMP: "lamp", KIND_BOLLARD: "bollard", KIND_BENCH: "bench", KIND_PICNIC: "bench",
    KIND_SIGN: "sign", KIND_CABINET: "cabinet", KIND_WASTE: "bin", KIND_BIN: "bin",
    KIND_BIKE_RACK: "bike_rack", KIND_SHOP_SIGN: "shop_sign", KIND_HANGING_SIGN: "shop_sign",
    KIND_MANHOLE: "manhole", KIND_FOUNTAIN: "fountain", KIND_PLANTER: "planter",
    KIND_POSTBOX: "postbox", KIND_POLE: "pole", KIND_STATUE: "art", KIND_ART: "art",
    KIND_MONUMENT: "art", KIND_WALL_LAMP: "lamp",
}
SAME_WITHIN_M = {
    "lamp": 4.0, "bollard": 1.2, "bench": 3.0, "sign": 3.0, "cabinet": 3.0, "bin": 3.0,
    "bike_rack": 5.0, "shop_sign": 2.5, "manhole": 1.5, "fountain": 5.0, "planter": 2.0,
    "postbox": 3.0, "pole": 3.0, "art": 8.0,
}

# A ground object this far inside a building is moved out to the wall; deeper
# than this it is something mapped indoors, or wrong, and is left out.
MOVE_OUT_M = 1.5
# Shop signs further than this from any wall are not shop signs.
SIGN_WALL_M = 6.0
SIGN_HEIGHT_M = 3.1  # above the pavement: the top of a Dutch shop front
# A street light Mapillary places this close to a facade hangs on it. Round
# the Ganzenmarkt 33 of the 52 it found beyond the BGT's lampposts are within
# 2 m of a wall, where only 8 of the BGT's 63 posts stand: the old centres
# light their streets from the facades, which no register lists.
WALL_LAMP_M = 2.0
WALL_LAMP_HEIGHT_M = 4.5


@dataclass
class StreetObject:
    kind: int
    x: float
    y: float
    source: int
    spin: float = float("nan")  # radians; NaN lets the builder choose
    height_m: float = float("nan")  # the object's own height, where known
    size: float = float("nan")  # per kind: a bike rack's bikes, a fountain's width
    z_m: float = float("nan")  # height above the ground it hangs at, for signs


def osm_objects(osm, *, surface_at=None, ground_at=None) -> list[StreetObject]:
    """OpenStreetMap's points and short lines as street objects.

    `surface_at` and `ground_at` (the survey's surface and bare earth) give a
    statue its height; without them it gets a typical one.
    """
    out: list[StreetObject] = []
    for feature in getattr(osm, "features", []):
        tags = feature.tags
        x, y = (float(v) for v in feature.centroid)
        kind = feature.kind
        if kind == "bench":
            out.append(StreetObject(
                KIND_PICNIC if tags.get("leisure") == "picnic_table" else KIND_BENCH,
                x, y, SOURCE_OSM, spin=_line_spin(feature),
            ))
        elif kind == "bin":
            out.append(StreetObject(KIND_BIN, x, y, SOURCE_OSM))
        elif kind == "bike_rack":
            try:
                capacity = float(str(tags.get("capacity", "")).split(";")[0])
            except ValueError:
                capacity = float("nan")
            if tags.get("bicycle_parking") in ("building", "shed", "underground", "floor"):
                continue  # a bicycle cellar is not a rack in the street
            out.append(StreetObject(
                KIND_BIKE_RACK, x, y, SOURCE_OSM, spin=_line_spin(feature), size=capacity,
            ))
        elif kind == "bollard":
            out.append(StreetObject(KIND_BOLLARD, x, y, SOURCE_OSM))
        elif kind == "lamp":
            out.append(StreetObject(KIND_LAMP, x, y, SOURCE_OSM))
        elif kind == "fountain":
            small = tags.get("amenity") == "drinking_water" or tags.get("man_made") == "drinking_fountain"
            out.append(StreetObject(KIND_FOUNTAIN, x, y, SOURCE_OSM, size=0.0 if small else 1.0))
        elif kind == "planter":
            out.append(StreetObject(KIND_PLANTER, x, y, SOURCE_OSM))
        elif kind == "box":
            amenity = tags.get("amenity")
            out.append(StreetObject(KIND_POSTBOX if amenity == "post_box" else KIND_CABINET, x, y, SOURCE_OSM))
        elif kind == "statue":
            out.append(_statue(tags, x, y, surface_at, ground_at))
        # Hydrants are underground in the Netherlands: a sign on a post at
        # most, which the sign posts already are. Terraces and steps have
        # stages of their own.
    return out


def _line_spin(feature) -> float:
    """The direction a bench or a rack mapped as a line runs, else NaN."""
    xy = np.asarray(feature.xy, dtype=np.float64)
    if len(xy) < 2 or feature.closed:
        return float("nan")
    d = xy[-1] - xy[0]
    if np.hypot(*d) < 0.3:
        return float("nan")
    return float(np.arctan2(d[1], d[0]))


def _statue(tags: dict, x: float, y: float, surface_at, ground_at) -> StreetObject:
    """A statue, a sculpture or a memorial, as tall as the survey says."""
    artwork = tags.get("artwork_type", "")
    memorial = tags.get("memorial", "")
    if artwork in ("statue", "bust") or memorial in ("statue", "bust"):
        kind = KIND_STATUE
    elif tags.get("historic") in ("memorial", "monument") and artwork == "":
        kind = KIND_MONUMENT
    else:
        kind = KIND_ART
    height = float("nan")
    if surface_at is not None and ground_at is not None:
        # The highest return within half a metre of the node, less the
        # ground round it. Further out is the tree a statue stands under:
        # round the Ganzenmarkt the surface model rises to 4.5 to 17 m a
        # metre from three of the four.
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ring = np.concatenate([[0.0], np.full(8, 0.5)])
        theta = np.concatenate([[0.0], angles])
        sx, sy = x + ring * np.cos(theta), y + ring * np.sin(theta)
        top = np.asarray(surface_at(sx, sy), dtype=np.float64)
        # The bare earth has a hole where the statue stands: its ground is
        # read in a ring round it.
        around = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        base = np.asarray(
            ground_at(x + 2.0 * np.cos(around), y + 2.0 * np.sin(around)), dtype=np.float64
        )
        if np.isfinite(top).any() and np.isfinite(base).any():
            measured = float(np.nanmax(top) - np.nanmedian(base))
            # Under a tree the survey sees the crown: few statues are 6 m.
            if 0.6 <= measured <= 6.0:
                height = measured
    return StreetObject(kind, x, y, SOURCE_OSM, height_m=height)


def mapillary_objects(features: list[dict]) -> list[StreetObject]:
    """Mapillary's map features as street objects; the signs still loose."""
    from .mapillary import feature_kind

    out: list[StreetObject] = []
    for feature in features:
        kind = feature.get("kind") or feature_kind(str(feature.get("value") or ""))
        value = str(feature.get("value") or "")
        x, y = float(feature["x"]), float(feature["y"])
        heading = feature.get("heading")
        # Mapillary's aligned direction is a compass bearing, clockwise from
        # north; a spin is counter-clockwise from east.
        spin = float(np.radians(90.0 - float(heading))) if heading is not None else float("nan")
        mapped = {
            "lamp": KIND_LAMP, "bench": KIND_BENCH, "bin": KIND_BIN, "bike_rack": KIND_BIKE_RACK,
            "cabinet": KIND_CABINET, "pole": KIND_POLE, "bollard": KIND_BOLLARD,
            "manhole": KIND_MANHOLE, "traffic_sign": KIND_SIGN, "facade_sign": KIND_SHOP_SIGN,
        }.get(kind)
        if kind == "box":
            mapped = KIND_POSTBOX if "mailbox" in value else KIND_CABINET
        if mapped == KIND_SHOP_SIGN and "banner" in value:
            mapped = KIND_HANGING_SIGN
        if mapped is None:
            continue
        size = float("nan")
        if mapped == KIND_MANHOLE:
            # A manhole cover, a gutter's grating, a valve's lid.
            size = 0.65 if "manhole" in value else (0.2 if "valve" in value else 0.45)
        out.append(StreetObject(mapped, x, y, SOURCE_MAPILLARY, spin=spin, size=size))
    return out


def merge(objects: list[StreetObject]) -> tuple[list[StreetObject], dict]:
    """One object per thing: the most trusted report of it wins.

    Sorted by source, BGT first; an object is dropped when one of the same
    family from an earlier source, or one already kept from its own, stands
    within that family's distance. A pole Mapillary saw next to a lamppost or
    a sign post is that post.
    """
    kept: list[StreetObject] = []
    by_family: dict[str, list[tuple[float, float]]] = {}
    dropped = 0
    for obj in sorted(objects, key=lambda o: o.source):
        family = FAMILY.get(obj.kind, f"kind{obj.kind}")
        radius = SAME_WITHIN_M.get(family, 2.0)
        families = [family] + (["lamp", "sign"] if family == "pole" else [])
        duplicate = False
        for fam in families:
            near = by_family.get(fam)
            if not near:
                continue
            arr = np.asarray(near)
            if (np.hypot(arr[:, 0] - obj.x, arr[:, 1] - obj.y) < radius).any():
                duplicate = True
                break
        # Only a less trusted source is merged away. Two BGT lampposts 3 m
        # apart are two lampposts; the register said so.
        if duplicate and obj.source != SOURCE_BGT:
            dropped += 1
            continue
        kept.append(obj)
        by_family.setdefault(family, []).append((obj.x, obj.y))
    return kept, {"duplicates_dropped": dropped}


class Footprints:
    """The buildings seen from above, for inside tests and wall lookups."""

    def __init__(self, buildings) -> None:
        import shapely

        shapes = []
        for building in getattr(buildings, "buildings", []) or []:
            roof = np.asarray(getattr(building, "roof_tris", []), dtype=np.float64)
            if roof.ndim != 3 or not len(roof):
                continue
            tris = shapely.polygons(roof[:, :, :2])
            tris = tris[shapely.area(tris) > 1e-6]
            if len(tris):
                shapes.append(shapely.union_all(tris).buffer(0.02))
        # One outline for a block of terraced houses: a party wall is no wall
        # for a sign to hang on.
        self.union = shapely.union_all(np.array(shapes, dtype=object)) if shapes else None
        self.outline = self.union.boundary if self.union is not None else None
        if self.union is not None:
            shapely.prepare(self.union)

    def inside(self, x, y) -> np.ndarray:
        import shapely

        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if self.union is None:
            return np.zeros(len(x), dtype=bool)
        return shapely.contains_xy(self.union, x, y)

    def wall_at(self, x: float, y: float, within_m: float):
        """``(point on the wall, outward normal, along)`` nearest (x, y), or None."""
        import shapely

        if self.outline is None:
            return None
        p = shapely.Point(x, y)
        if self.outline.distance(p) > within_m:
            return None
        lines = getattr(self.outline, "geoms", [self.outline])
        best = min(lines, key=lambda line: line.distance(p))
        at = best.project(p)
        a = np.asarray(best.interpolate(max(at - 0.3, 0.0)).coords[0])
        b = np.asarray(best.interpolate(min(at + 0.3, best.length)).coords[0])
        foot = np.asarray(best.interpolate(at).coords[0])
        along = b - a
        if np.hypot(*along) < 1e-6:
            return None
        along /= np.hypot(*along)
        normal = np.array([-along[1], along[0]])
        if self.inside(*(foot + normal * 0.3))[0]:
            normal = -normal
        return foot, normal, along


def place(
    objects: list[StreetObject],
    footprints: Footprints | None,
    *,
    water_at=None,
) -> tuple[list[StreetObject], dict]:
    """Ground objects out of buildings and water; signs onto their walls."""
    kept: list[StreetObject] = []
    stats = {"moved_out": 0, "dropped_indoors": 0, "dropped_on_water": 0,
             "signs_hung": 0, "signs_without_wall": 0}
    for obj in objects:
        if obj.kind in (KIND_SHOP_SIGN, KIND_HANGING_SIGN) and obj.source != SOURCE_BGT:
            wall = footprints.wall_at(obj.x, obj.y, SIGN_WALL_M) if footprints else None
            if wall is None:
                stats["signs_without_wall"] += 1
                continue
            foot, normal, along = wall
            wall_spin = float(np.arctan2(along[1], along[0]))
            if obj.kind == KIND_HANGING_SIGN:
                at = foot + normal * 0.7
                kept.append(StreetObject(
                    KIND_HANGING_SIGN, float(at[0]), float(at[1]), obj.source,
                    spin=float(np.arctan2(normal[1], normal[0])), z_m=SIGN_HEIGHT_M,
                ))
            else:
                at = foot + normal * 0.06
                kept.append(StreetObject(
                    KIND_SHOP_SIGN, float(at[0]), float(at[1]), obj.source,
                    spin=wall_spin, z_m=SIGN_HEIGHT_M,
                ))
            stats["signs_hung"] += 1
            continue
        if water_at is not None and obj.kind != KIND_FOUNTAIN and bool(water_at(obj.x, obj.y)):
            stats["dropped_on_water"] += 1
            continue
        if obj.kind == KIND_LAMP and obj.source == SOURCE_MAPILLARY and footprints is not None:
            wall = footprints.wall_at(obj.x, obj.y, WALL_LAMP_M)
            if wall is not None:
                foot, normal, _ = wall
                at = foot + normal * 0.45
                kept.append(StreetObject(
                    KIND_WALL_LAMP, float(at[0]), float(at[1]), obj.source,
                    spin=float(np.arctan2(normal[1], normal[0])), z_m=WALL_LAMP_HEIGHT_M,
                ))
                stats["wall_lamps"] = stats.get("wall_lamps", 0) + 1
                continue
        if footprints is not None and footprints.inside(obj.x, obj.y)[0]:
            if obj.source == SOURCE_BGT:
                kept.append(obj)  # the register's position stands
                continue
            wall = footprints.wall_at(obj.x, obj.y, MOVE_OUT_M)
            if wall is None:
                stats["dropped_indoors"] += 1
                continue
            foot, normal, _ = wall
            at = foot + normal * 0.5
            obj.x, obj.y = float(at[0]), float(at[1])
            stats["moved_out"] += 1
        kept.append(obj)
    return kept, stats


def water_lookup(water_bodies):
    """``f(x, y) -> bool`` over the BGT water, or None without any."""
    import shapely

    from .terrain_mesh import _polygonal

    shapes = []
    for body in water_bodies or ():
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in getattr(body, "rings", ())]
        rings = [r for r in rings if len(r) >= 4]
        if rings:
            shapes.append(_polygonal(shapely.make_valid(shapely.Polygon(rings[0], rings[1:]))))
    if not shapes:
        return None
    union = shapely.union_all(np.array(shapes, dtype=object))
    shapely.prepare(union)
    return lambda x, y: bool(shapely.contains_xy(union, float(x), float(y)))


def from_furniture(furniture) -> list[StreetObject]:
    """The BGT's furniture, as street objects.

    Keeps what the furniture stage knew of each: the way it faces (turned to
    its road) and its size (a rack's bicycles, a charger's connectors).
    """
    xy = np.asarray(furniture.xy).reshape(-1, 2)
    n = len(xy)

    def per_object(values):
        values = getattr(furniture, values, None)
        if values is None or len(values) != n:
            return np.full(n, np.nan)
        return np.asarray(values, dtype=np.float64)

    spin, size = per_object("spin"), per_object("size")
    # 0 there is "one of itself"; here a size of 0 means small, NaN unknown.
    size = np.where(size > 0, size, np.nan)
    out = []
    for i, ((x, y), kind) in enumerate(zip(xy, np.asarray(furniture.kind))):
        out.append(StreetObject(
            int(kind), float(x), float(y), SOURCE_BGT, spin=float(spin[i]), size=float(size[i]),
        ))
    return out


def to_furniture(objects: list[StreetObject], surface_at, *, sources: dict | None = None):
    """Street objects back into a FurnitureSet, standing on the drawn ground."""
    from .furniture import FurnitureSet

    result = FurnitureSet()
    if not objects:
        return result
    result.xy = np.array([[o.x, o.y] for o in objects], dtype=np.float64)
    result.kind = np.array([o.kind for o in objects], dtype=np.int32)
    result.ground_z_nap = np.asarray(surface_at(result.xy[:, 0], result.xy[:, 1]), dtype=np.float64)
    result.spin = np.array([o.spin for o in objects], dtype=np.float64)
    result.height_m = np.array([o.height_m for o in objects], dtype=np.float64)
    result.size = np.array([o.size for o in objects], dtype=np.float64)
    result.z_m = np.array([o.z_m for o in objects], dtype=np.float64)
    result.source = np.array([o.source for o in objects], dtype=np.int8)
    result.counts = {
        name: int((result.kind == k).sum()) for k, name in ALL_KIND_NAMES.items()
        if (result.kind == k).any()
    }
    result.sources = sources or {}
    return result


def add_street_objects(
    furniture,
    *,
    osm=None,
    mapillary_features=None,
    buildings=None,
    areas=None,
    water=None,
    surface_at,
    survey_top=None,
    survey_ground=None,
    terraces: bool = True,
    bbox=None,
):
    """The BGT's furniture with OpenStreetMap's and Mapillary's added.

    `surface_at` is the top of the drawn ground (`TerrainResult.surface`);
    `survey_top` and `survey_ground` the survey's surface model and bare
    earth, for a statue's height. Returns a FurnitureSet and what came from
    where.
    """
    from .terraces import build_terraces, paving_for_terraces

    footprints = Footprints(buildings) if buildings is not None else None
    water_at = water_lookup(water)
    objects = from_furniture(furniture)
    found_osm = osm_objects(osm, surface_at=survey_top, ground_at=survey_ground) if osm is not None else []
    found_mly = mapillary_objects(mapillary_features) if mapillary_features else []
    merged, merge_stats = merge(objects + found_osm + found_mly)
    placed, place_stats = place(merged, footprints, water_at=water_at)
    terrace_stats: dict = {}
    if terraces and osm is not None:
        made, terrace_stats = build_terraces(
            osm, footprints, paving_ok=paving_for_terraces(areas), water_at=water_at,
        )
        placed += made
    if bbox is not None:
        placed = [
            o for o in placed
            if bbox.xmin <= o.x <= bbox.xmax and bbox.ymin <= o.y <= bbox.ymax
        ]
    stats = {
        "reported": {
            "BGT": len(objects), "OpenStreetMap": len(found_osm), "Mapillary": len(found_mly),
        },
        **merge_stats,
        **place_stats,
        "kept": summarise(placed),
    }
    if terrace_stats:
        stats["terraces"] = terrace_stats
    LOG.info(
        "street objects: %d from the BGT, %d from OpenStreetMap, %d from Mapillary; "
        "%d duplicates merged, %d moved out of buildings, %d dropped indoors, "
        "%d shop signs hung on walls",
        len(objects), len(found_osm), len(found_mly), merge_stats["duplicates_dropped"],
        place_stats["moved_out"], place_stats["dropped_indoors"], place_stats["signs_hung"],
    )
    return to_furniture(placed, surface_at, sources=stats), stats


def summarise(objects: list[StreetObject]) -> dict:
    out: dict = {}
    for source, name in SOURCE_NAMES.items():
        mine = [o for o in objects if o.source == source]
        if not mine:
            continue
        counts: dict[str, int] = {}
        for o in mine:
            label = ALL_KIND_NAMES.get(o.kind, str(o.kind))
            counts[label] = counts.get(label, 0) + 1
        out[name] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return out


__all__ = [
    "ALL_KIND_NAMES",
    "Footprints",
    "KIND_BIKE_RACK",
    "KIND_BIN",
    "KIND_FOUNTAIN",
    "KIND_HANGING_SIGN",
    "KIND_MANHOLE",
    "KIND_PARASOL",
    "KIND_PLANTER",
    "KIND_POLE",
    "KIND_POSTBOX",
    "KIND_SHOP_SIGN",
    "KIND_STATUE",
    "KIND_TABLE",
    "KIND_WALL_LAMP",
    "NEW_KIND_NAMES",
    "SOURCE_BGT",
    "SOURCE_DERIVED",
    "SOURCE_MAPILLARY",
    "SOURCE_NAMES",
    "SOURCE_OSM",
    "StreetObject",
    "add_street_objects",
    "from_furniture",
    "mapillary_objects",
    "merge",
    "osm_objects",
    "place",
    "summarise",
    "to_furniture",
    "water_lookup",
]
