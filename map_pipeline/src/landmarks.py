"""What the registers say a building is: a church, a tower, a monument.

3DBAG gives every building its shape and the BAG its use, and neither says
that one of them is a church. The facade stage then has to guess from height,
age and a missing storey count, and a low church comes out as a row of
houses. Two registers know better:

* **TOP10NL** (Kadaster, CC BY 4.0) types its buildings: kerk, toren,
  klokkentoren, watertoren, kas, parkeergarage, stationsgebouw. Over the
  Utrecht square kilometre 8 churches, 5 towers and the Domtoren by name.
  Its polygons are generalised, though, and a church in a block is often
  drawn as one polygon with the block ("huizenblok|kerk"), so a type only
  passes to a building that is most of its polygon.
* **The national monument register** (Rijksdienst voor het Cultureel
  Erfgoed, CC0) places every rijksmonument with its category: 549 over the
  same square, 21 of them religious buildings. A monument point inside a
  building's footprint is exact where TOP10NL is not.

A building either names says is a church gets the church facade, whatever
its storey count. Every rijksmonument carries its number into the model, as a
custom property next to the BAG id, and keeps a historic wall when the BAG
gives it no year.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import BBox

LOG = logging.getLogger(__name__)

TOP10NL_URL = "https://api.pdok.nl/kadaster/brt-top10nl/ogc/v1/collections"
RCE_WFS_URL = "https://data.geo.cultureelerfgoed.nl/openbaar/wfs"
RIVM_WFS_URL = "https://data.rivm.nl/geo/alo/wfs"

# What each TOP10NL type does to a building.
WORSHIP = frozenset({"kerk", "kapel", "moskee", "synagoge", "klooster"})
TOWERS = frozenset({"toren", "klokkentoren", "watertoren", "vuurtoren", "kasteel"})
SHEDS = frozenset({
    "parkeerdak, parkeerdek, parkeergarage", "fabriek", "kas, warenhuis",
    "loods", "silo", "gemaal", "elektriciteitscentrale", "werf",
})
SPECIAL = WORSHIP | TOWERS | SHEDS | frozenset({"stationsgebouw", "gemeentehuis"})

# Monument categories that are never a stack of storeys.
MONUMENTAL_CATEGORIES = ("Religieuze gebouwen",)
OLD_CATEGORIES = ("Kastelen, landhuizen en parken", "Bestuursgebouwen")

# A building has to be at least this much of a TOP10NL polygon to take its
# type: a church inside a block polygon is not the whole block.
SHARE_OF_POLYGON = 0.5


@dataclass
class Monument:
    xy: tuple[float, float]
    number: int
    category: str
    subcategory: str


@dataclass
class TypedArea:
    polygon: object
    types: tuple[str, ...]
    name: str


@dataclass
class Turbine:
    xy: tuple[float, float]
    hub_m: float
    rotor_m: float
    model: str


@dataclass
class Landmarks:
    areas: list[TypedArea] = field(default_factory=list)
    monuments: list[Monument] = field(default_factory=list)
    turbines: list[Turbine] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.areas or self.monuments or self.turbines)


def parse_top10(features: list[dict]) -> list[TypedArea]:
    import shapely
    from shapely.geometry import shape

    out = []
    for feature in features:
        props = feature.get("properties") or {}
        types = tuple(t.strip() for t in str(props.get("typegebouw") or "").split("|") if t.strip())
        if not any(t in SPECIAL for t in types):
            continue
        try:
            geometry = shape(feature.get("geometry") or {})
        except Exception:  # noqa: BLE001
            continue
        if geometry.is_empty or geometry.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        if not geometry.is_valid:
            geometry = shapely.make_valid(geometry)
        out.append(TypedArea(geometry, types, str(props.get("naam") or "")))
    return out


def parse_monuments(features: list[dict]) -> list[Monument]:
    out = []
    for feature in features:
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        try:
            number = int(props.get("rijksmonument_nummer"))
        except (TypeError, ValueError):
            continue
        out.append(
            Monument(
                xy=(float(coords[0]), float(coords[1])),
                number=number,
                category=str(props.get("hoofdcategorie") or ""),
                subcategory=str(props.get("subcategorie") or "").strip(),
            )
        )
    return out


def fetch_landmarks(
    bbox: BBox, *, cache_dir: Path | None = None, timeout: float = 120.0,
    top10_url: str = TOP10NL_URL, rce_url: str = RCE_WFS_URL,
) -> Landmarks:
    """TOP10NL's typed buildings and the rijksmonumenten in `bbox`. Either
    being away costs its improvement and nothing else."""
    import gzip
    import json

    from .http_util import get_with_retry, short_error
    from .ogcapi import fetch_items

    result = Landmarks()
    try:
        features = fetch_items(
            f"{top10_url}/gebouw_vlak/items", bbox, description="TOP10NL buildings",
            timeout=timeout, cache_dir=cache_dir, cache_name="top10nl_gebouw",
        )
        result.areas = parse_top10(features)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not read TOP10NL buildings (%s)", short_error(exc))

    cache = None
    if cache_dir is not None:
        key = f"{bbox.xmin:.1f}_{bbox.ymin:.1f}_{bbox.xmax:.1f}_{bbox.ymax:.1f}"
        cache = Path(cache_dir) / f"rce_monuments_{key}.json.gz"
    try:
        if cache is not None and cache.is_file():
            with gzip.open(cache, "rt", encoding="utf-8") as handle:
                features = json.load(handle)
        else:
            response = get_with_retry(
                rce_url,
                params={
                    "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                    "typeNames": "openbaar:rijksmonumentpunten",
                    "outputFormat": "application/json", "srsName": "EPSG:28992",
                    "bbox": f"{bbox.xmin},{bbox.ymin},{bbox.xmax},{bbox.ymax},EPSG:28992",
                    "count": 20000,
                },
                timeout=timeout, max_retries=3, description="monument register",
            )
            features = response.json().get("features") or []
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(cache, "wt", encoding="utf-8") as handle:
                    json.dump(features, handle)
        result.monuments = parse_monuments(features)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not read the monument register (%s)", short_error(exc))

    try:
        response = get_with_retry(
            RIVM_WFS_URL,
            params={
                "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                "typeNames": "alo:rivm_windturbines_ashoogte_actueel",
                "outputFormat": "application/json", "srsName": "EPSG:28992",
                "bbox": f"{bbox.xmin},{bbox.ymin},{bbox.xmax},{bbox.ymax},EPSG:28992",
                "count": 2000,
            },
            timeout=timeout, max_retries=3, description="wind turbine register",
        )
        result.turbines = parse_turbines(response.json().get("features") or [])
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not read the wind turbine register (%s)", short_error(exc))

    LOG.info(
        "landmarks: %d typed buildings in TOP10NL, %d rijksmonumenten, %d wind turbines",
        len(result.areas), len(result.monuments), len(result.turbines),
    )
    return result


def parse_turbines(features: list[dict]) -> list[Turbine]:
    out = []
    for feature in features:
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        try:
            hub, rotor = float(props.get("ash")), float(props.get("diam"))
        except (TypeError, ValueError):
            continue
        if len(coords) < 2 or not (5.0 <= hub <= 250.0) or not (2.0 <= rotor <= 250.0):
            continue
        out.append(Turbine((float(coords[0]), float(coords[1])), hub, rotor, str(props.get("wt_type") or "")))
    return out


# --------------------------------------------------------------------------
# Sails and rotors
# --------------------------------------------------------------------------

# Which way sails and rotors face: into the prevailing south-westerly.
WIND_FROM_DEG = 225.0


def _box(start, end, width: float, depth: float, up) -> np.ndarray:
    """A box from `start` to `end`, `width` across in the `up` x direction
    plane and `depth` along `up`, as twelve outward triangles."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    along = end - start
    length = float(np.linalg.norm(along))
    if length < 1e-6:
        return np.zeros((0, 3, 3))
    along /= length
    up = np.asarray(up, dtype=np.float64)
    side = np.cross(up, along)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross([1.0, 0.0, 0.0], along)
    side /= np.linalg.norm(side)
    normal = np.cross(along, side)
    a, b = side * width / 2.0, normal * depth / 2.0
    corners = [start - a - b, start + a - b, start + a + b, start - a + b,
               end - a - b, end + a - b, end + a + b, end - a + b]
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    triangles = np.array([[corners[i], corners[j], corners[k]] for i, j, k in faces])
    # Outward, whatever handedness the frame came out with.
    centre = (start + end) / 2.0
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = ((triangles.mean(axis=1) - centre) * normals).sum(axis=1) < 0
    triangles[flip] = triangles[flip][:, ::-1]
    return triangles


def _rotor_frame():
    facing = np.radians(WIND_FROM_DEG)
    out = np.array([np.sin(facing), np.cos(facing), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    across = np.cross(up, out)
    return out, up, across


def windmill_sails(building) -> np.ndarray:
    """Four sails on a windmill's cap, facing the prevailing wind, at rest in
    a cross. 3DBAG has the body and the cap; nothing scanned the sails."""
    shape = footprint(building)
    if shape is None:
        return np.zeros((0, 3, 3))
    ground, top = float(building.ground_z_nap), float(building.roof_max_nap)
    height = top - ground
    if height < 6.0:
        return np.zeros((0, 3, 3))
    centre = np.array([shape.centroid.x, shape.centroid.y])
    radius_body = float(np.sqrt(shape.area / np.pi))
    out, up, across = _rotor_frame()
    hub = np.array([*(centre + out[:2] * (radius_body + 0.8)), top - 1.5])
    reach = float(np.clip(0.45 * height + 2.0, 6.0, 14.0))
    reach = min(reach, hub[2] - ground - 0.4)
    if reach < 3.0:
        return np.zeros((0, 3, 3))
    pieces = [_box(np.array([*centre, top - 1.5]), hub + out * 0.5, 0.5, 0.5, up)]
    for angle in (45.0, 135.0, 225.0, 315.0):
        direction = np.cos(np.radians(angle)) * across + np.sin(np.radians(angle)) * up
        tip = hub + direction * reach
        pieces.append(_box(hub, tip, 0.35, 0.35, out))
        # The lattice: a narrow board along the outer three quarters, to one
        # side of the stock, as the sail frame is.
        side = np.cross(out, direction)
        base = hub + direction * reach * 0.22 + side * 1.0
        pieces.append(_box(base, tip + side * 1.0, 1.7, 0.08, out))
    return np.concatenate(pieces)


def _cylinder(base, top, r0: float, r1: float, sides: int = 16) -> np.ndarray:
    """A tapered vertical tube, closed at the top, faces outward."""
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(sides)])
    low = np.asarray(base, dtype=np.float64) + ring * r0
    high = np.asarray(top, dtype=np.float64) + ring * r1
    triangles = []
    for k in range(sides):
        n = (k + 1) % sides
        triangles.append([low[k], low[n], high[n]])
        triangles.append([low[k], high[n], high[k]])
    for k in range(1, sides - 1):
        triangles.append([high[0], high[k], high[k + 1]])
    return np.asarray(triangles)


def turbine_mesh(turbine: Turbine, ground_nap: float) -> np.ndarray:
    """A tower, a nacelle and three blades, from the register's hub height
    and rotor diameter."""
    out, up, across = _rotor_frame()
    base = np.array([turbine.xy[0], turbine.xy[1], ground_nap])
    hub = base + up * turbine.hub_m
    radius = turbine.rotor_m / 2.0
    tower_r = max(0.6, 0.022 * turbine.hub_m)
    pieces = [
        _cylinder(base, hub - up * radius * 0.03, tower_r, tower_r * 0.6),
        _box(hub - out * radius * 0.1, hub + out * radius * 0.12, radius * 0.06, radius * 0.06, up),
    ]
    face = hub + out * radius * 0.14
    for angle in (90.0, 210.0, 330.0):
        direction = np.cos(np.radians(angle)) * across + np.sin(np.radians(angle)) * up
        pieces.append(_box(face, face + direction * radius, radius * 0.06, radius * 0.015, out))
    return np.concatenate(pieces)


def landmark_meshes(buildings, landmarks: Landmarks, terrain) -> dict[str, np.ndarray]:
    """Windmill sails and wind turbines, as triangles in RD and NAP."""
    sails, turbines = [], []
    for building in getattr(buildings, "buildings", None) or []:
        if "molen" in (getattr(building, "monument_subcategory", "") or "").lower():
            sails.append(windmill_sails(building))
    for turbine in landmarks.turbines if landmarks else []:
        ground = float(np.asarray(terrain.sample(np.array([turbine.xy[0]]), np.array([turbine.xy[1]])))[0])
        turbines.append(turbine_mesh(turbine, ground))
    out = {}
    if sails and any(len(s) for s in sails):
        out["sails"] = np.concatenate([s for s in sails if len(s)])
    if turbines:
        out["turbines"] = np.concatenate(turbines)
    return out


def footprint(building):
    """A building's footprint, from its roof triangles seen from above."""
    import shapely

    roof = np.asarray(getattr(building, "roof_tris", []), dtype=np.float64)
    if roof.ndim != 3 or not len(roof):
        return None
    triangles = shapely.polygons(roof[:, :, :2])
    triangles = triangles[shapely.area(triangles) > 1e-6]
    if not len(triangles):
        return None
    shape = shapely.union_all(triangles)
    return None if shape.is_empty else shape


def apply_landmarks(buildings, landmarks: Landmarks) -> dict:
    """Tell each building what the registers say it is.

    Sets `landmark` (the TOP10NL type, or "monument") and `monument`
    (the rijksmonument number) on the buildings that have one.
    """
    import shapely

    stats = {"typed": 0, "monuments": 0}
    if not landmarks:
        return stats
    shapes = []
    owners = []
    for building in getattr(buildings, "buildings", None) or []:
        shape = footprint(building)
        if shape is not None:
            shapes.append(shape)
            owners.append(building)
    if not shapes:
        return stats
    tree = shapely.STRtree(np.array(shapes, dtype=object))

    for area in landmarks.areas:
        special = [t for t in area.types if t in SPECIAL]
        if not special:
            continue
        for index in tree.query(area.polygon, predicate="intersects"):
            shape = shapes[int(index)]
            overlap = shape.intersection(area.polygon).area
            # The building is most of the polygon, or the polygon is most of
            # the building and names nothing else.
            if overlap >= SHARE_OF_POLYGON * area.polygon.area or (
                len(area.types) == len(special) and overlap >= 0.6 * shape.area
            ):
                building = owners[int(index)]
                if not getattr(building, "landmark", ""):
                    building.landmark = special[0]
                    building.landmark_name = area.name
                    stats["typed"] += 1

    for monument in landmarks.monuments:
        point = shapely.Point(*monument.xy)
        hits = tree.query(point, predicate="dwithin", distance=0.5)
        if not len(hits):
            continue
        index = min(hits, key=lambda i: shapes[int(i)].distance(point))
        building = owners[int(index)]
        if getattr(building, "monument", None) is None:
            building.monument = monument.number
            building.monument_category = monument.category
            building.monument_subcategory = monument.subcategory
            stats["monuments"] += 1
            if not building.build_year and monument.category != "Archeologie (N)":
                # A monument the BAG has no year for is old; the historic
                # wall is the right guess.
                building.build_year = 1850
            if any(monument.category.startswith(c) for c in MONUMENTAL_CATEGORIES):
                if not getattr(building, "landmark", ""):
                    building.landmark = "kerk"
    LOG.info(
        "landmarks: %d buildings typed by TOP10NL, %d rijksmonumenten placed",
        stats["typed"], stats["monuments"],
    )
    return stats


def save_landmarks(shapes: dict[str, np.ndarray], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: tris for name, tris in shapes.items()})
    LOG.info(
        "wrote %s (%s)", path,
        ", ".join(f"{name} {len(tris)} triangles" for name, tris in shapes.items()),
    )
    return path


__all__ = [
    "save_landmarks",
    "Landmarks",
    "Turbine",
    "landmark_meshes",
    "parse_turbines",
    "turbine_mesh",
    "windmill_sails",
    "Monument",
    "SHEDS",
    "TOWERS",
    "TypedArea",
    "WORSHIP",
    "apply_landmarks",
    "fetch_landmarks",
    "footprint",
    "parse_monuments",
    "parse_top10",
]
