"""OpenStreetMap, for what the Dutch registers do not record.

The BGT knows where the carriageway is and NWB where its middle runs, and
neither says how many lanes it carries, where the zebra is or which junction
has lights. OpenStreetMap does: over one square kilometre of Utrecht centre it
tags the lane count on the main streets, every crossing with how it is marked,
and the speed tables. It is the only open national source for any of that.

Read from the main OSM API rather than Overpass. Overpass is the usual choice
for a query like this and is refused by enough office proxies that a run could
not rely on it; the main API answers a plain bounding-box download and that is
all this needs. It caps a request at 50 000 nodes, which a dense square
kilometre exceeds, so the area is read in tiles and a tile that is still too
full is split in four.

OpenStreetMap is published under the ODbL: the attribution file credits it
whenever this module contributed to a model.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import BBox, RD_CRS, WGS84_CRS, _transformer, rd_bbox_to_wgs84
from .http_util import ServiceError, get_with_retry, short_error

LOG = logging.getLogger(__name__)

OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map.json"

# A node carrying any of these is kept, with all of its tags: a bicycle
# rack's capacity and whether it is under cover are tags of their own. The
# great majority of nodes carry nothing and are only geometry for the ways.
NODE_KEYS = (
    "highway",
    "crossing",
    "crossing:markings",
    "traffic_calming",
    "traffic_signals",
    "amenity",
    "emergency",
    "man_made",
    "barrier",
    "leisure",
    "tourism",
)

# Ways that are not roads but are street detail: a bicycle parking drawn as
# an area, a planter. Everything else -- buildings above all, four fifths of a
# city download -- is the registers' business and is dropped.
AREA_TAGS = {
    "amenity": {"bicycle_parking"},
    "man_made": {"planter"},
}

# Bumped when what is kept changes, so an older cache is not read as if it
# held everything a newer run expects.
CACHE_VERSION = "osm2"


def keep_way(tags: dict) -> bool:
    if "highway" in tags:
        return True
    return any(tags.get(key) in values for key, values in AREA_TAGS.items())


@dataclass
class OsmWay:
    """One highway, in RD."""

    id: int
    points: np.ndarray  # (n, 2)
    node_ids: tuple[int, ...]
    tags: dict[str, str]

    @property
    def highway(self) -> str:
        return self.tags.get("highway", "")


@dataclass
class OsmNode:
    """One tagged node, in RD."""

    id: int
    xy: tuple[float, float]
    tags: dict[str, str]


@dataclass
class OsmData:
    ways: list[OsmWay] = field(default_factory=list)
    nodes: list[OsmNode] = field(default_factory=list)
    # Closed ways that are street detail rather than road (`AREA_TAGS`).
    areas: list[OsmWay] = field(default_factory=list)
    source: str = ""

    def __bool__(self) -> bool:
        return bool(self.ways or self.nodes or self.areas)

    def ways_through(self) -> dict[int, list[OsmWay]]:
        """Which highways pass through each node."""
        out: dict[int, list[OsmWay]] = {}
        for way in self.ways:
            for node_id in way.node_ids:
                out.setdefault(node_id, []).append(way)
        return out


def parse_elements(elements: list[dict], bbox: BBox | None = None) -> OsmData:
    """OSM JSON elements into highways and tagged nodes, in RD.

    Only ways tagged ``highway`` are kept: buildings, land use and the rest
    are the registers' job and are better there.
    """
    lonlat: dict[int, tuple[float, float]] = {}
    tagged: dict[int, dict[str, str]] = {}
    raw_ways: dict[int, dict] = {}
    for element in elements:
        kind = element.get("type")
        if kind == "node":
            try:
                node_id = int(element["id"])
                lonlat[node_id] = (float(element["lon"]), float(element["lat"]))
            except (KeyError, TypeError, ValueError):
                continue
            tags = element.get("tags") or {}
            if any(k in tags for k in NODE_KEYS):
                tagged[node_id] = {str(k): str(v) for k, v in tags.items()}
        elif kind == "way":
            tags = element.get("tags") or {}
            if keep_way(tags) and element.get("nodes"):
                raw_ways[int(element["id"])] = element

    data = OsmData(source="OpenStreetMap")
    if not lonlat:
        return data

    ids = np.fromiter(lonlat.keys(), dtype=np.int64, count=len(lonlat))
    coords = np.asarray(list(lonlat.values()), dtype=np.float64)
    to_rd = _transformer(WGS84_CRS, RD_CRS)
    xs, ys = to_rd.transform(coords[:, 0], coords[:, 1])
    rd = {int(i): (float(x), float(y)) for i, x, y in zip(ids, xs, ys)}

    for way_id, element in raw_ways.items():
        node_ids = [int(n) for n in element["nodes"] if int(n) in rd]
        if len(node_ids) < 2:
            continue
        points = np.asarray([rd[n] for n in node_ids], dtype=np.float64)
        # Coincident nodes have no direction between them.
        keep = np.concatenate(
            [[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
        )
        if keep.sum() < 2:
            continue
        target = data.ways if "highway" in (element.get("tags") or {}) else data.areas
        target.append(
            OsmWay(
                id=way_id,
                points=points[keep],
                node_ids=tuple(n for n, k in zip(node_ids, keep) if k),
                tags={str(k): str(v) for k, v in (element.get("tags") or {}).items()},
            )
        )

    for node_id, tags in tagged.items():
        x, y = rd[node_id]
        if bbox is not None and not (
            bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax
        ):
            continue
        data.nodes.append(OsmNode(id=node_id, xy=(x, y), tags=tags))
    return data


def _tiles(bbox: BBox, tile_m: float) -> list[BBox]:
    nx = max(1, int(np.ceil(bbox.width / tile_m)))
    ny = max(1, int(np.ceil(bbox.height / tile_m)))
    xs = np.linspace(bbox.xmin, bbox.xmax, nx + 1)
    ys = np.linspace(bbox.ymin, bbox.ymax, ny + 1)
    return [
        BBox(float(xs[i]), float(ys[j]), float(xs[i + 1]), float(ys[j + 1]))
        for j in range(ny)
        for i in range(nx)
    ]


def _fetch_tile(
    tile: BBox, *, url: str, timeout: float, max_retries: int, depth: int = 0
) -> list[dict]:
    """One tile's elements, splitting it when the API says it is too full."""
    wgs = rd_bbox_to_wgs84(tile)
    try:
        response = get_with_retry(
            url,
            params={
                "bbox": f"{wgs.xmin:.7f},{wgs.ymin:.7f},{wgs.xmax:.7f},{wgs.ymax:.7f}"
            },
            timeout=timeout,
            max_retries=max_retries,
            description="OpenStreetMap map download",
        )
    except ServiceError as exc:
        # "You requested too many nodes (limit is 50000)". Four smaller
        # requests answer where the one did not.
        if "too many nodes" in str(exc) and depth < 3:
            half = _tiles(tile, max(tile.width, tile.height) / 2.0 + 1e-6)
            out: list[dict] = []
            for part in half:
                out.extend(
                    _fetch_tile(
                        part, url=url, timeout=timeout,
                        max_retries=max_retries, depth=depth + 1,
                    )
                )
            return out
        raise
    return list(response.json().get("elements") or [])


def fetch_osm(
    bbox: BBox,
    *,
    url: str = OSM_API_URL,
    tile_m: float = 500.0,
    timeout: float = 120.0,
    max_retries: int = 3,
    cache_dir: Path | None = None,
) -> OsmData:
    """Highways and tagged nodes over `bbox`, or nothing on failure.

    Empty rather than raising, like the road centrelines: everything built
    from this is detail on top of a model that stands without it.
    """
    cache = None
    if cache_dir is not None:
        key = f"{bbox.xmin:.1f}_{bbox.ymin:.1f}_{bbox.xmax:.1f}_{bbox.ymax:.1f}"
        cache = Path(cache_dir) / f"{CACHE_VERSION}_{key}.json.gz"
        if cache.is_file():
            with gzip.open(cache, "rt", encoding="utf-8") as handle:
                data = parse_elements(json.load(handle), bbox)
            LOG.info(
                "OpenStreetMap: %d highways, %d areas, %d tagged nodes (cached)",
                len(data.ways), len(data.areas), len(data.nodes),
            )
            return data

    elements: dict[tuple[str, int], dict] = {}
    try:
        for tile in _tiles(bbox, tile_m):
            for element in _fetch_tile(
                tile, url=url, timeout=timeout, max_retries=max_retries
            ):
                kind = element.get("type")
                # Buildings are four fifths of a city download and none of
                # it is used; dropping them here keeps the cache small.
                if kind == "way" and not keep_way(element.get("tags") or {}):
                    continue
                if kind not in ("node", "way"):
                    continue
                elements[(kind, int(element["id"]))] = element
    except Exception as exc:  # noqa: BLE001 - see the docstring
        LOG.warning(
            "could not read OpenStreetMap (%s); lane lines, crossings and "
            "speed tables are left out",
            short_error(exc),
        )
        return OsmData(source="unavailable")

    # Only the nodes something refers to, or that say something themselves.
    used: set[int] = set()
    for (kind, _), element in elements.items():
        if kind == "way":
            used.update(int(n) for n in element.get("nodes") or [])
    kept = [
        {k: v for k, v in element.items() if k in ("type", "id", "lat", "lon", "nodes", "tags")}
        for (kind, element_id), element in elements.items()
        if kind == "way"
        or element_id in used
        or any(k in (element.get("tags") or {}) for k in NODE_KEYS)
    ]

    data = parse_elements(kept, bbox)
    LOG.info(
        "OpenStreetMap: %d highways, %d areas, %d tagged nodes",
        len(data.ways), len(data.areas), len(data.nodes),
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        staging = cache.with_suffix(cache.suffix + ".part")
        with gzip.open(staging, "wt", encoding="utf-8") as handle:
            json.dump(kept, handle)
        staging.replace(cache)
    return data


__all__ = [
    "AREA_TAGS",
    "NODE_KEYS",
    "keep_way",
    "OSM_API_URL",
    "OsmData",
    "OsmNode",
    "OsmWay",
    "fetch_osm",
    "parse_elements",
]
