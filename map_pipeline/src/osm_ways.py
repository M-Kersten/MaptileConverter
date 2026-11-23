"""OpenStreetMap, for what the Dutch registers do not record.

The BGT knows where the carriageway is and NWB where its middle runs, and
neither says how many lanes it carries, where the zebra is or which junction
has lights. OpenStreetMap does: over one square kilometre of Utrecht centre it
tags the lane count on the main streets, every crossing with how it is marked,
and the speed tables. It is the only open national source for any of that.

Read through `osm.tile_elements`: the same cached tiles of the main OSM API
the street objects come from, one download per tile for both.

OpenStreetMap is published under the ODbL: the attribution file credits it
whenever this module contributed to a model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import BBox, RD_CRS, WGS84_CRS, _transformer
from .http_util import short_error

LOG = logging.getLogger(__name__)

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
    "shop",
    "craft",
    "office",
)

# Ways that are not roads but are street detail: a bicycle parking drawn as
# an area, a planter. Everything else -- buildings above all, four fifths of a
# city download -- is the registers' business and is dropped.
AREA_TAGS = {
    "amenity": {"bicycle_parking"},
    "man_made": {"planter"},
}

def telling_building(tags: dict) -> bool:
    """A building way worth keeping: joined to the BAG on ref:bag, and saying
    more than "building=yes" (see `kinds.kind_from_osm`)."""
    from .kinds import ground_kind_from_osm, kind_from_osm

    if "building" not in tags or "ref:bag" not in tags:
        return False
    return bool(kind_from_osm(tags) or ground_kind_from_osm(tags))


def keep_way(tags: dict) -> bool:
    if "highway" in tags:
        return True
    if telling_building(tags):
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
    # Buildings whose tags say what they are, with their BAG id.
    buildings: list[OsmWay] = field(default_factory=list)
    source: str = ""

    def __bool__(self) -> bool:
        return bool(self.ways or self.nodes or self.areas or self.buildings)

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
        tags = element.get("tags") or {}
        target = (
            data.ways if "highway" in tags
            else data.buildings if telling_building(tags)
            else data.areas
        )
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


def fetch_osm_ways(
    bbox: BBox,
    *,
    cache_dir: Path,
    api_url: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
    max_age_days: float = 30.0,
) -> OsmData:
    """Highways, telling buildings and tagged nodes over `bbox`, or nothing
    on failure.

    Empty rather than raising, like the road centrelines: everything built
    from this is detail on top of a model that stands without it.
    """
    from .osm import API_URL, tile_elements

    try:
        elements, _ = tile_elements(
            bbox, cache_dir=Path(cache_dir), api_url=api_url or API_URL,
            timeout=timeout, max_retries=max_retries, max_age_days=max_age_days,
        )
    except Exception as exc:  # noqa: BLE001 - see the docstring
        LOG.warning(
            "could not read OpenStreetMap (%s); lane lines, crossings and "
            "speed tables are left out",
            short_error(exc),
        )
        return OsmData(source="unavailable")
    data = parse_elements(elements, bbox)
    LOG.info(
        "OpenStreetMap: %d highways, %d areas, %d telling buildings, %d tagged nodes",
        len(data.ways), len(data.areas), len(data.buildings), len(data.nodes),
    )
    return data


__all__ = [
    "AREA_TAGS",
    "NODE_KEYS",
    "keep_way",
    "telling_building",
    "OsmData",
    "OsmNode",
    "OsmWay",
    "fetch_osm_ways",
    "parse_elements",
]
