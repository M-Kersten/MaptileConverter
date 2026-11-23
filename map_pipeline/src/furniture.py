"""Street furniture: the small things a street is not a street without.

None of this is structurally important, and that is rather the point. A street
with nothing on it reads as a model; the same street with lampposts along the
kerb and benches by the water reads as a place. They are cheap primitives
sharing one atlas, so a couple of thousand objects cost one draw call.

The BGT registers far more of this than the pipeline used to draw. Over one
square kilometre of Utrecht centre it has 1058 lampposts, 918 bollards and 148
benches -- which were kept -- alongside 50 playgrounds, 41 waste collection
points, 40 tram catenary masts, 8 art objects, 6 advertising columns, 51
electrical cabinets, 3 bus shelters, a memorial monument, a picnic table and
exactly one flagpole, all of which were fetched and thrown away. They are the
difference between a street and a street someone maintains.

What a municipality registers differs a great deal, though. Rotterdam's BGT
has litter bins, roll containers, parking meters, cameras, direction signs and
245 street lights hung on facades; Utrecht's has none of those. So two more
sources fill in behind it, each kept only where the BGT has nothing of the same
kind close by:

* **Charging points** from NDW's national register (CC0), with how many
  connectors each has. Neither the BGT nor most municipalities record them.
* **OpenStreetMap** (ODbL) for what volunteers map and surveyors do not:
  bicycle parking with its capacity, litter bins, post boxes, artworks, bus
  stops, cameras and the rest of `OSM_WANTED`.

Every piece is turned to face the street it stands by: a bench with its back
to the houses, a lamppost with its arm over the road, a bicycle rack along the
kerb. A light the BGT puts against a facade hangs off that facade.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bgt import fetch_current
from .geo import BBox, RD_CRS, WGS84_CRS, _transformer

LOG = logging.getLogger(__name__)

KIND_LAMP = 0
KIND_BOLLARD = 1
KIND_BENCH = 2
KIND_SIGN = 3
KIND_FLAGPOLE = 4
KIND_SHELTER = 5
KIND_COLUMN = 6
KIND_ART = 7
KIND_MONUMENT = 8
KIND_PLAYGROUND = 9
KIND_PICNIC = 10
KIND_CABINET = 11
KIND_WASTE = 12
KIND_CATENARY = 13
# The same numbers as `street_objects` for what both draw: one bicycle rack
# is kind 14 whether the BGT, OpenStreetMap or Mapillary found it.
KIND_BIKE_RACK = 14
KIND_BIN = 15
KIND_FOUNTAIN = 19
KIND_POST_BOX = 21
KIND_WALL_LAMP = 26
# Kinds only the BGT's own types and NDW's chargers give.
KIND_CONTAINER = 27
KIND_CAMERA = 28
KIND_DIRECTION_SIGN = 29
KIND_BILLBOARD = 30
KIND_INFO_BOARD = 31
KIND_PARKING_METER = 32
KIND_TOILET = 33
KIND_BIKE_LOCKER = 34
KIND_MOORING_BOLLARD = 35
KIND_HECTOMETRE = 36
KIND_CHARGER = 37
KIND_BUS_STOP = 38
KIND_TERRACE = 39

KIND_NAMES = {
    KIND_LAMP: "lamppost",
    KIND_BOLLARD: "bollard",
    KIND_BENCH: "bench",
    KIND_SIGN: "sign",
    KIND_FLAGPOLE: "flagpole",
    KIND_SHELTER: "bus shelter",
    KIND_COLUMN: "ad column",
    KIND_ART: "art object",
    KIND_MONUMENT: "monument",
    KIND_PLAYGROUND: "playground",
    KIND_PICNIC: "picnic table",
    KIND_CABINET: "cabinet",
    KIND_WASTE: "waste point",
    KIND_CATENARY: "catenary mast",
    KIND_BIN: "litter bin",
    KIND_CONTAINER: "roll container",
    KIND_CAMERA: "camera",
    KIND_DIRECTION_SIGN: "direction sign",
    KIND_BILLBOARD: "billboard",
    KIND_INFO_BOARD: "information board",
    KIND_PARKING_METER: "parking meter",
    KIND_TOILET: "public toilet",
    KIND_BIKE_LOCKER: "bicycle locker",
    KIND_MOORING_BOLLARD: "mooring bollard",
    KIND_FOUNTAIN: "fountain",
    KIND_WALL_LAMP: "wall lamp",
    KIND_HECTOMETRE: "hectometre post",
    KIND_CHARGER: "charging point",
    KIND_BIKE_RACK: "bicycle rack",
    KIND_POST_BOX: "post box",
    KIND_BUS_STOP: "bus stop",
}

# Kinds that hang off something rather than standing on the ground.
MOUNTED_KINDS = frozenset({KIND_WALL_LAMP})

# Which BGT collection and plus_type becomes which kind.
#
# "niet-bgt" appears throughout the BGT and means the surveyor recorded an
# object the standard has no type for. It is deliberately not mapped: drawing
# something specific for "unknown" is how a model ends up with 70 identical
# sheds in it.
WANTED: dict[str, dict[str, int]] = {
    "paal": {
        "lichtmast": KIND_LAMP,
        "afsluitpaal": KIND_BOLLARD,
        "poller": KIND_BOLLARD,
        "verkeersbordpaal": KIND_SIGN,
        "verkeersregelinstallatiepaal": KIND_SIGN,
        "vlaggenmast": KIND_FLAGPOLE,
        "hectometerpaal": KIND_HECTOMETRE,
    },
    "straatmeubilair": {
        "bank": KIND_BENCH,
        "abri": KIND_SHELTER,
        "reclamezuil": KIND_COLUMN,
        "kunstobject": KIND_ART,
        "herdenkingsmonument": KIND_MONUMENT,
        "speelvoorziening": KIND_PLAYGROUND,
        "speeltoestel": KIND_PLAYGROUND,
        "picknicktafel": KIND_PICNIC,
        "zitelement": KIND_BENCH,
        "betaalautomaat": KIND_PARKING_METER,
        "openbaar toilet": KIND_TOILET,
        "fietsenkluis": KIND_BIKE_LOCKER,
        "fietsenrek": KIND_BIKE_RACK,
        "bolder": KIND_MOORING_BOLLARD,
        "fontein": KIND_FOUNTAIN,
        # A light point is a lamp without its own mast: on a facade, mostly,
        # which is where it is hung below (`_hang_on_walls`).
        "lichtpunt": KIND_LAMP,
    },
    "kast": {
        "elektrakast": KIND_CABINET,
        "telkast": KIND_CABINET,
        "gaskast": KIND_CABINET,
        "cai-kast": KIND_CABINET,
        "verkeersregelinstallatiekast": KIND_CABINET,
        "openbare verlichtingkast": KIND_CABINET,
    },
    "bak": {
        "afval apart plaats": KIND_WASTE,
        "afvalbak": KIND_BIN,
        "container": KIND_CONTAINER,
    },
    "mast": {
        "bovenleidingmast": KIND_CATENARY,
        "hoogspanningsmast": KIND_CATENARY,
    },
    "bord": {
        "verkeersbord": KIND_SIGN,
        "wegwijzer": KIND_DIRECTION_SIGN,
        "reclamebord": KIND_BILLBOARD,
        "informatiebord": KIND_INFO_BOARD,
    },
    "sensor_punt": {
        "camera": KIND_CAMERA,
    },
}

# Kept for callers that used the old names.
LAMP_TYPES = {"lichtmast"}
BOLLARD_TYPES = {"afsluitpaal", "poller"}
SIGN_TYPES = {"verkeersbordpaal", "verkeersregelinstallatiepaal", "verkeersbord"}
BENCH_TYPES = {"bank"}

# OpenStreetMap tags and the kind each becomes, with how close a BGT object of
# the same kind has to be for the OpenStreetMap one to be the same object. A
# test on the node's other tags can refuse one: a mural is an artwork, but it is
# paint on a wall rather than something standing in the street.
OSM_WANTED: tuple[tuple[str, str, int, float], ...] = (
    ("amenity", "bench", KIND_BENCH, 4.0),
    ("amenity", "waste_basket", KIND_BIN, 3.0),
    ("amenity", "bicycle_parking", KIND_BIKE_RACK, 6.0),
    ("amenity", "post_box", KIND_POST_BOX, 3.0),
    ("amenity", "recycling", KIND_WASTE, 6.0),
    ("amenity", "charging_station", KIND_CHARGER, 8.0),
    ("amenity", "vending_machine", KIND_PARKING_METER, 3.0),
    ("amenity", "fountain", KIND_FOUNTAIN, 6.0),
    ("man_made", "street_cabinet", KIND_CABINET, 3.0),
    ("man_made", "surveillance", KIND_CAMERA, 5.0),
    ("tourism", "artwork", KIND_ART, 6.0),
    ("tourism", "information", KIND_INFO_BOARD, 5.0),
    ("leisure", "picnic_table", KIND_PICNIC, 4.0),
    ("highway", "street_lamp", KIND_LAMP, 3.0),
    ("highway", "bus_stop", KIND_BUS_STOP, 10.0),
    ("barrier", "bollard", KIND_BOLLARD, 1.5),
)

# A cafe that OpenStreetMap says has seats outside gets a terrace in front of
# its facade: this many metres out, a table per this many metres of front.
TERRACE_OUT_M = 2.4
TERRACE_TABLE_M = 2.2
HORECA_AMENITIES = frozenset({"restaurant", "cafe", "bar", "pub", "fast_food", "ice_cream"})

# Artwork painted on or set into a wall; there is nothing to stand in the street.
FLAT_ARTWORK = frozenset({"mural", "graffiti", "relief", "mosaic", "painting"})
# Bicycle parking inside or under something is not in the street either.
HIDDEN_PARKING = frozenset({"building", "underground", "multi-storey", "garage"})

# A bicycle every this many metres along a rack, and the most one rack holds.
BIKE_PITCH_M = 0.7
BIKES_PER_RACK = 40
# Racks in a parking area are rows this far apart.
RACK_ROW_M = 2.2
# Racks without a capacity tag are about this long.
DEFAULT_RACK_BIKES = 10

NDW_CHARGERS_URL = "https://opendata.ndw.nu/charging_point_locations.geojson.gz"

# Kinds turned to run along the street, with their front to it: a bench, a
# bicycle rack, a cabinet against the kerb.
ALONG_ROAD = frozenset({
    KIND_BENCH, KIND_SHELTER, KIND_PICNIC, KIND_CABINET, KIND_BIN,
    KIND_CONTAINER, KIND_PARKING_METER, KIND_BIKE_LOCKER, KIND_CHARGER,
    KIND_BIKE_RACK, KIND_POST_BOX, KIND_TOILET, KIND_WASTE,
})
# Kinds turned to reach towards the street: a lamp's arm, a camera, a sign's
# face looking along the road at the traffic.
TOWARDS_ROAD = frozenset({
    KIND_LAMP, KIND_SIGN, KIND_DIRECTION_SIGN, KIND_BILLBOARD,
    KIND_INFO_BOARD, KIND_CAMERA, KIND_HECTOMETRE, KIND_BUS_STOP,
})
# How far a road may be and still be the one a thing faces.
ORIENT_REACH_M = 25.0
# A light point this close to a facade is hung on it.
WALL_REACH_M = 1.5


@dataclass
class FurnitureSet:
    """Positions and kinds, ready for the Blender stage to instance.

    `spin`, `height_m`, `size` and `z_m` are per object and NaN where the
    Blender stage chooses (`street_objects.StreetObject`); `source` says
    which register or map each came from.
    """

    xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    ground_z_nap: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    # The angle of each piece's local X axis, radians anticlockwise from east;
    # NaN where nothing says which way it faces, and Blender picks one.
    spin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # How many: bicycles in a rack, connectors on a charging point. 0 for
    # everything that is one of itself.
    size: np.ndarray = field(default_factory=lambda: np.zeros(0))
    counts: dict[str, int] = field(default_factory=dict)
    # Per object where known, else None: what the street-object merge adds.
    height_m: np.ndarray | None = None
    z_m: np.ndarray | None = None
    source: np.ndarray | None = None
    sources: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.xy))

    def stats(self) -> dict:
        out = {"count": len(self), **self.counts}
        if self.sources:
            out["by_source"] = self.sources
        return out


@dataclass
class _Item:
    x: float
    y: float
    kind: int
    source: str
    spin: float = float("nan")
    size: float = 0.0


# --------------------------------------------------------------------------
# What is around a piece of furniture
# --------------------------------------------------------------------------


class Footprints:
    """Building outlines seen from above, for "is this on a wall".

    Made from each building's roof triangles, the way the barrier stage makes
    them: 3DBAG's roofs cover exactly the footprint, overhangs included.
    """

    def __init__(self, buildings=None, polygons=None) -> None:
        import shapely

        shapes = list(polygons or [])
        for building in getattr(buildings, "buildings", None) or []:
            roof = np.asarray(getattr(building, "roof_tris", []), dtype=np.float64)
            if roof.ndim != 3 or not len(roof):
                continue
            triangles = shapely.polygons(roof[:, :, :2])
            triangles = triangles[shapely.area(triangles) > 1e-6]
            if len(triangles):
                shapes.append(shapely.union_all(triangles))
        self.shapes = np.array(shapes, dtype=object)
        self.tree = shapely.STRtree(self.shapes) if len(shapes) else None

    def __bool__(self) -> bool:
        return self.tree is not None

    def inside(self, xy) -> bool:
        import shapely

        if self.tree is None:
            return False
        point = shapely.Point(float(xy[0]), float(xy[1]))
        return bool(len(self.tree.query(point, predicate="within")))

    def facade(self, xy, reach: float) -> tuple[np.ndarray, np.ndarray] | None:
        """The nearest point on a facade within `reach`, and the direction out
        of the building there."""
        found = self.wall(xy, reach)
        if found is None:
            return None
        distance, outward = found
        xy = np.asarray(xy, dtype=np.float64)
        inside = self.inside(xy)
        point = xy + outward * distance if inside else xy - outward * distance
        return point, outward

    def street_facade(self, xy, reach: float, clear: float = 1.5):
        """The nearest point on a facade within `reach` whose outside is not
        another building, and the direction out there.

        The nearest wall to a point inside a terraced house is as often the
        party wall as the front; this walks the outline for the nearest
        stretch that faces the street.
        """
        import shapely

        found = self.facade(xy, reach)
        if found is None:
            return None
        point, outward = found
        if not self.inside(point + outward * clear):
            return point, outward
        here = shapely.Point(float(xy[0]), float(xy[1]))
        best = None
        for index in self.tree.query(here.buffer(reach)):
            shape = self.shapes[int(index)]
            for polygon in shapely.get_parts(shape):
                if polygon.geom_type != "Polygon":
                    continue
                polygon = shapely.orient_polygons(polygon) if hasattr(shapely, "orient_polygons") else polygon
                ring = shapely.get_coordinates(shapely.segmentize(polygon.exterior, 0.5))[:, :2]
                for a, b in zip(ring[:-1], ring[1:]):
                    edge = b - a
                    length = float(np.linalg.norm(edge))
                    if length < 1e-6:
                        continue
                    middle = (a + b) / 2.0
                    distance = float(np.linalg.norm(middle - np.asarray(xy, dtype=np.float64)))
                    if distance > reach or (best is not None and distance >= best[0]):
                        continue
                    # Counter-clockwise: the outside is on the right.
                    out = np.array([edge[1], -edge[0]]) / length
                    if not self.inside(middle + out * clear):
                        best = (distance, middle, out)
        return None if best is None else (best[1], best[2])

    def wall(self, xy, reach: float) -> tuple[float, np.ndarray] | None:
        """Distance to the nearest facade within `reach`, and the direction
        out of the building through it."""
        import shapely

        if self.tree is None:
            return None
        point = shapely.Point(float(xy[0]), float(xy[1]))
        best = None
        for index in self.tree.query(point.buffer(reach)):
            shape = self.shapes[int(index)]
            nearest = shapely.get_coordinates(
                shapely.shortest_line(shape.boundary, point)
            )[0]
            offset = np.asarray(xy, dtype=np.float64) - nearest
            distance = float(np.linalg.norm(offset))
            if distance > reach or (best is not None and distance >= best[0]):
                continue
            if distance < 1e-6:
                continue
            outward = offset / distance
            if shape.contains(point):
                outward = -outward
            best = (distance, outward)
        return best


def _left(vector: np.ndarray) -> np.ndarray:
    return np.array([-vector[1], vector[0]])


def facing_spin(kind: int, xy, roads, reach: float = ORIENT_REACH_M) -> float:
    """Which way one piece of furniture is turned, from the road beside it.

    NaN when no road is near enough to say: a bench in the middle of a park
    faces wherever Blender's dice put it.
    """
    if roads is None or (kind not in ALONG_ROAD and kind not in TOWARDS_ROAD):
        return float("nan")
    found = roads.nearest(xy, reach)
    if found is None:
        return float("nan")
    point, tangent = found
    towards = np.asarray(point, dtype=np.float64) - np.asarray(xy, dtype=np.float64)
    distance = float(np.linalg.norm(towards))
    towards = towards / distance if distance > 0.3 else _left(tangent)
    if kind in ALONG_ROAD:
        # Local X along the road, local +Y -- the front -- towards it.
        along = tangent if float(_left(tangent) @ towards) > 0 else -tangent
        return float(np.arctan2(along[1], along[0]))
    return float(np.arctan2(towards[1], towards[0]))


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def _bgt_items(bbox: BBox, fetch_kwargs: dict, skip_types) -> list[_Item]:
    items: list[_Item] = []
    for collection, mapping in WANTED.items():
        try:
            features, _stats = fetch_current(
                collection, bbox, class_field="plus_type", **fetch_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - one empty collection is not fatal
            # kast, bak and mast are often empty outside a town centre, and a
            # street with no lampposts because the bin collection failed would
            # be a poor trade.
            LOG.warning("could not read BGT %s: %s", collection, exc)
            continue
        for feature in features:
            plus_type = str((feature.get("properties") or {}).get("plus_type") or "")
            kind = mapping.get(plus_type)
            if kind is None or plus_type in skip_types:
                continue
            geometry = feature.get("geometry") or {}
            if geometry.get("type") != "Point":
                continue
            x, y = geometry["coordinates"][:2]
            if not (bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax):
                continue
            source = "bgt_light_point" if plus_type == "lichtpunt" else "bgt"
            items.append(_Item(float(x), float(y), kind, source))
    return items


def parse_chargers(features: list[dict], bbox: BBox) -> list[_Item]:
    """NDW charging locations in the bbox, sized by their connector count."""
    rows = []
    for feature in features:
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        connectors = 0
        for availability in (feature.get("properties") or {}).get("availabilities") or []:
            try:
                connectors += int(availability.get("total") or 0)
            except (TypeError, ValueError, AttributeError):
                continue
        rows.append((float(coords[0]), float(coords[1]), max(connectors, 1)))
    if not rows:
        return []
    to_rd = _transformer(WGS84_CRS, RD_CRS)
    xs, ys = to_rd.transform([r[0] for r in rows], [r[1] for r in rows])
    return [
        _Item(float(x), float(y), KIND_CHARGER, "ndw", size=float(n))
        for x, y, (_, _, n) in zip(xs, ys, rows)
        if bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax
    ]


def fetch_chargers(
    bbox: BBox, *, url: str = NDW_CHARGERS_URL, timeout: float = 120.0,
    cache_dir: Path | None = None,
) -> list[_Item]:
    """Every public charging location in the bbox, or none on failure.

    The register is published as one national file, five megabytes gzipped,
    with no area query. What falls inside the bbox is cached.
    """
    from .http_util import get_with_retry, short_error

    cache = None
    if cache_dir is not None:
        key = f"{bbox.xmin:.1f}_{bbox.ymin:.1f}_{bbox.xmax:.1f}_{bbox.ymax:.1f}"
        cache = Path(cache_dir) / f"ndw_chargers_{key}.json.gz"
        if cache.is_file():
            with gzip.open(cache, "rt", encoding="utf-8") as handle:
                items = parse_chargers(json.load(handle), bbox)
            LOG.info("NDW: %d charging locations (cached)", len(items))
            return items
    try:
        response = get_with_retry(
            url, timeout=timeout, max_retries=3, expect_binary=True,
            description="NDW charging point locations",
        )
        payload = response.content
        if payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
        features = json.loads(payload).get("features") or []
    except Exception as exc:  # noqa: BLE001 - chargers are detail
        LOG.warning(
            "could not read NDW charging points (%s); the model gets none",
            short_error(exc),
        )
        return []

    wgs = _transformer(RD_CRS, WGS84_CRS)
    lon0, lat0 = wgs.transform(bbox.xmin - 50, bbox.ymin - 50)
    lon1, lat1 = wgs.transform(bbox.xmax + 50, bbox.ymax + 50)
    inside = [
        f for f in features
        if len((f.get("geometry") or {}).get("coordinates") or []) >= 2
        and lon0 <= f["geometry"]["coordinates"][0] <= lon1
        and lat0 <= f["geometry"]["coordinates"][1] <= lat1
    ]
    items = parse_chargers(inside, bbox)
    LOG.info("NDW: %d charging locations (of %d nationally)", len(items), len(features))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        staging = cache.with_suffix(cache.suffix + ".part")
        with gzip.open(staging, "wt", encoding="utf-8") as handle:
            json.dump(inside, handle)
        staging.replace(cache)
    return items


def _capacity(tags: dict, default: int) -> int:
    try:
        return max(1, int(str(tags.get("capacity", "")).split(";")[0]))
    except ValueError:
        return default


def osm_kind(tags: dict) -> tuple[int, float] | None:
    """The kind an OpenStreetMap node becomes, and its dedupe radius."""
    for key, value, kind, radius in OSM_WANTED:
        if tags.get(key) != value:
            continue
        if kind == KIND_ART and (
            tags.get("artwork_type") in FLAT_ARTWORK
            or tags.get("indoor") == "yes"
        ):
            return None
        if kind == KIND_INFO_BOARD and tags.get("information") not in ("board", "map"):
            return None
        if kind == KIND_WASTE and tags.get("recycling_type") not in (None, "container"):
            return None
        if kind == KIND_PARKING_METER and "parking" not in tags.get("vending", ""):
            return None
        if kind == KIND_BIKE_RACK and (
            tags.get("bicycle_parking") in HIDDEN_PARKING
            or tags.get("location") in ("underground", "indoor")
            or tags.get("indoor") == "yes"
        ):
            return None
        if kind == KIND_CAMERA and tags.get("camera:mount") in ("wall", "ceiling", "building"):
            return None
        try:
            if int(float(str(tags.get("level", "0")).split(";")[0])) < 0:
                return None
        except ValueError:
            pass
        return kind, radius
    return None


def _osm_items(osm, bbox: BBox) -> list[_Item]:
    """Nodes and bicycle parking areas from OpenStreetMap."""
    from .vehicles import principal_axes, points_in_ring

    items: list[_Item] = []
    for node in getattr(osm, "nodes", None) or []:
        x, y = node.xy
        if not (bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax):
            continue
        found = osm_kind(node.tags)
        if found is None:
            continue
        kind, _radius = found
        size = (
            float(min(_capacity(node.tags, DEFAULT_RACK_BIKES), BIKES_PER_RACK))
            if kind == KIND_BIKE_RACK else 0.0
        )
        items.append(_Item(float(x), float(y), kind, "osm", size=size))

    for area in getattr(osm, "areas", None) or []:
        tags = area.tags
        if tags.get("amenity") != "bicycle_parking" or osm_kind(tags) is None:
            continue
        ring = np.asarray(area.points, dtype=np.float64)
        if len(ring) < 4:
            continue
        long_axis, short_axis, length, width = principal_axes(ring)
        if length < 1.0:
            continue
        centre = ring.mean(axis=0)
        if not (bbox.xmin <= centre[0] <= bbox.xmax and bbox.ymin <= centre[1] <= bbox.ymax):
            continue
        rows = max(1, int(width // RACK_ROW_M))
        per_row = max(1, int(length // BIKE_PITCH_M))
        total = min(_capacity(tags, rows * per_row), rows * per_row)
        offsets = (np.arange(rows) - (rows - 1) / 2.0) * (width / rows)
        spin = float(np.arctan2(long_axis[1], long_axis[0]))
        left = total
        for offset in offsets:
            if left <= 0:
                break
            position = centre + short_axis * offset
            if not points_in_ring(position[None], ring)[0]:
                continue
            # A long row is several racks, so no one rack is longer than a
            # rack is: each is its own object, and each sits inside the area.
            row_bikes = min(per_row, left)
            pieces = int(np.ceil(row_bikes / BIKES_PER_RACK))
            span = row_bikes * BIKE_PITCH_M
            for k in range(pieces):
                n = min(BIKES_PER_RACK, row_bikes - k * BIKES_PER_RACK)
                along = -span / 2.0 + (k * BIKES_PER_RACK + n / 2.0) * BIKE_PITCH_M
                x, y = position + long_axis * along
                items.append(_Item(float(x), float(y), KIND_BIKE_RACK, "osm", spin, float(n)))
            left -= row_bikes
    return items


def terraces(osm, footprints: "Footprints", bbox: BBox) -> list[_Item]:
    """A terrace in front of every cafe OpenStreetMap seats outside."""
    out: list[_Item] = []
    if not footprints:
        return out
    for node in getattr(osm, "nodes", None) or []:
        tags = node.tags
        if tags.get("amenity") not in HORECA_AMENITIES:
            continue
        if tags.get("outdoor_seating") not in ("yes", "sidewalk", "street", "terrace", "pedestrian_zone"):
            continue
        x, y = node.xy
        if not (bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax):
            continue
        facade = footprints.street_facade((x, y), 10.0)
        if facade is None:
            continue
        point, outward = facade
        spot = point + outward * TERRACE_OUT_M
        if footprints.inside(spot) or footprints.inside(point + outward * (TERRACE_OUT_M + 1.0)):
            continue
        along = np.array([-outward[1], outward[0]])
        # Local +Y towards the street, the way the tables are set.
        spin = float(np.arctan2(along[1], along[0]))
        out.append(_Item(float(spot[0]), float(spot[1]), KIND_TERRACE, "osm", spin, 2.0))
    return out


def _near(items: list[_Item], kind: int, xy, radius: float) -> bool:
    for item in items:
        if item.kind == kind and (item.x - xy[0]) ** 2 + (item.y - xy[1]) ** 2 <= radius * radius:
            return True
    return False


def merge_sources(bgt: list[_Item], extra: list[_Item]) -> list[_Item]:
    """The BGT's own items, then anything another source adds that the BGT
    does not already have within that kind's radius."""
    radius = {kind: r for _key, _value, kind, r in OSM_WANTED}
    radius[KIND_CHARGER] = 8.0
    kept = list(bgt)
    by_kind: dict[int, list[_Item]] = {}
    for item in bgt:
        by_kind.setdefault(item.kind, []).append(item)
    for item in extra:
        nearby = by_kind.get(item.kind, [])
        if _near(nearby, item.kind, (item.x, item.y), radius.get(item.kind, 3.0)):
            continue
        kept.append(item)
        by_kind.setdefault(item.kind, []).append(item)
    return kept


def _hang_on_walls(items: list[_Item], footprints: Footprints) -> int:
    """A BGT light point by a facade is a wall lamp; elsewhere a lamppost,
    unless a registered mast already stands on the spot."""
    masts = [i for i in items if i.kind == KIND_LAMP and i.source == "bgt"]
    hung = 0
    keep = []
    for item in items:
        if item.source != "bgt_light_point":
            keep.append(item)
            continue
        if _near(masts, KIND_LAMP, (item.x, item.y), 1.0):
            continue
        wall = footprints.wall((item.x, item.y), WALL_REACH_M) if footprints else None
        if wall is not None:
            _distance, outward = wall
            item.kind = KIND_WALL_LAMP
            item.spin = float(np.arctan2(outward[1], outward[0]))
            hung += 1
        keep.append(item)
    items[:] = keep
    return hung


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def build_furniture(
    bbox: BBox,
    work_dir: Path,
    *,
    furniture_cfg: dict,
    terrain,
    skip_types: set[str] | frozenset[str] = frozenset(),
    roads=None,
    buildings=None,
    osm=None,
    chargers: list[_Item] | None = None,
    carriageway=None,
) -> FurnitureSet:
    """Fetch poles and street furniture, and put them on the terrain.

    `skip_types` are BGT types another stage has drawn already: the traffic
    stage turns signal poles into traffic lights and replaces the few sign
    posts the BGT has with the signs themselves.

    `roads` (a `traffic.RoadIndex`) turns each piece to its street;
    `buildings` hangs light points on facades and keeps OpenStreetMap's indoor
    objects out; `osm` and `chargers` are the other sources, merged behind the
    BGT. `carriageway` (a `traffic.Carriageway`) is where nothing those two
    place may stand: their positions are an address or a mapper's click, and
    a charging point a metre into the road is a charging point on the kerb.
    """
    result = FurnitureSet()
    fetch_kwargs = {
        "page_limit": int(furniture_cfg["page_limit"]),
        "timeout": float(furniture_cfg["timeout_s"]),
        "max_retries": int(furniture_cfg["max_retries"]),
        "max_pages": int(furniture_cfg["max_pages"]),
    }

    items = _bgt_items(bbox, fetch_kwargs, set(skip_types))
    footprints = Footprints(buildings) if buildings is not None else Footprints()
    hung = _hang_on_walls(items, footprints)

    extra: list[_Item] = list(chargers or [])
    if osm is not None and bool(furniture_cfg.get("osm", True)):
        found = _osm_items(osm, bbox)
        if bool(furniture_cfg.get("terraces", True)):
            found.extend(terraces(osm, footprints, bbox))
        # Anything OpenStreetMap places inside a building is indoors: a bench
        # in a station hall, a bicycle rack in a garage. A camera against a
        # facade is on the facade, and drawn on a pole it would stand in
        # front of the door.
        extra.extend(
            i for i in found
            if not footprints.inside((i.x, i.y))
            and not (i.kind == KIND_CAMERA and footprints.wall((i.x, i.y), WALL_REACH_M))
        )
    moved = 0
    if carriageway:
        for item in extra:
            pushed = carriageway.push_off((item.x, item.y))
            if pushed is not None:
                item.x, item.y = float(pushed[0]), float(pushed[1])
                moved += 1
    items = merge_sources(items, extra)
    # Pushed off the road or set out from a facade, a few can end up over the
    # edge of the area, where there is no ground to stand on.
    items = [
        i for i in items
        if bbox.xmin <= i.x <= bbox.xmax and bbox.ymin <= i.y <= bbox.ymax
    ]

    cap = int(furniture_cfg.get("max_bicycles", 3000))
    bikes = 0
    for item in items:
        if item.kind == KIND_BIKE_RACK:
            item.size = float(max(0, min(item.size or DEFAULT_RACK_BIKES, cap - bikes)))
            bikes += int(item.size)

    if not items:
        LOG.info("no street furniture found")
        return result

    for item in items:
        if not np.isfinite(item.spin):
            item.spin = facing_spin(item.kind, (item.x, item.y), roads)

    result.xy = np.array([(i.x, i.y) for i in items], dtype=np.float64)
    result.kind = np.array([i.kind for i in items], dtype=np.int32)
    result.spin = np.array([i.spin for i in items], dtype=np.float64)
    result.size = np.array([i.size for i in items], dtype=np.float64)
    result.ground_z_nap = np.asarray(
        terrain.surface(result.xy[:, 0], result.xy[:, 1]), dtype=np.float64
    )
    result.counts = {
        KIND_NAMES[k]: int((result.kind == k).sum()) for k in KIND_NAMES
    }
    result.counts["bicycles"] = int(result.size[result.kind == KIND_BIKE_RACK].sum())
    result.counts["facing_a_road"] = int(np.isfinite(result.spin).sum())
    result.counts["hung_on_facades"] = hung
    result.counts["moved_off_the_road"] = moved
    for item in items:
        result.sources[item.source] = result.sources.get(item.source, 0) + 1
    LOG.info(
        "street furniture: %s",
        ", ".join(f"{v} {k}" for k, v in result.counts.items() if v),
    )

    save_furniture(result, work_dir / "furniture.npz")
    return result


def save_furniture(furniture: FurnitureSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(furniture)

    def column(values, fill=np.nan):
        """One value per object, `fill` where a field was never set."""
        if values is None or len(values) != n:
            return np.full(n, fill, dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    np.savez_compressed(
        path,
        xy=furniture.xy,
        ground_z_nap=furniture.ground_z_nap,
        kind=furniture.kind,
        spin=column(furniture.spin),
        height_m=column(furniture.height_m),
        size=column(furniture.size, 0.0),
        z_m=column(furniture.z_m),
        source=column(furniture.source, 0.0).astype(np.int8),
    )
    LOG.info("wrote %s (%d objects)", path, n)
    return path


__all__ = [
    "ALONG_ROAD",
    "Footprints",
    "KIND_ART",
    "KIND_BENCH",
    "KIND_BIKE_RACK",
    "KIND_BIN",
    "KIND_BOLLARD",
    "KIND_CABINET",
    "KIND_CATENARY",
    "KIND_CHARGER",
    "KIND_COLUMN",
    "KIND_FLAGPOLE",
    "KIND_LAMP",
    "KIND_MONUMENT",
    "KIND_NAMES",
    "KIND_PICNIC",
    "KIND_PLAYGROUND",
    "KIND_SHELTER",
    "KIND_SIGN",
    "KIND_TERRACE",
    "KIND_WALL_LAMP",
    "KIND_WASTE",
    "MOUNTED_KINDS",
    "NDW_CHARGERS_URL",
    "OSM_WANTED",
    "TOWARDS_ROAD",
    "WANTED",
    "FurnitureSet",
    "build_furniture",
    "facing_spin",
    "fetch_chargers",
    "merge_sources",
    "osm_kind",
    "parse_chargers",
    "terraces",
    "save_furniture",
]
