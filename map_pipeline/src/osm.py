"""OpenStreetMap, for what the national registers leave out.

The BGT is the map of the Netherlands' public space, and it is very good at
what a municipality maintains: kerbs, lampposts, the outline of every
pavement. What makes a square a place is often what nobody maintains:
the café terraces, the statue somebody's grandfather unveiled, the three
steps up to a church door. OpenStreetMap volunteers map exactly that:

* ``leisure=outdoor_seating`` areas, and venues tagged ``outdoor_seating``,
  are the terraces;
* ``tourism=artwork`` and ``historic=memorial`` are the statues, sculptures
  and plaques;
* ``highway=steps`` are flights of steps, with their direction;
* benches, bins, bicycle parking, bollards, street lamps, fountains and
  planters fill in where a smaller municipality never registered its own
  street furniture in the BGT, which its optional "plus" layer allows.

The data comes from the main OSM API in small tiles, each cached, and only
once per area: the API is a shared, volunteer-run service whose usage policy
allows exactly this and nothing heavier. OSM is © OpenStreetMap contributors
under the ODbL, and a model drawn from it says so.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import BBox, rd_bbox_to_wgs84
from .http_util import ServiceError, get_with_retry

LOG = logging.getLogger(__name__)

API_URL = "https://api.openstreetmap.org/api/0.6/map"
# Degrees per tile side: about 280 m by 330 m here, small enough that a busy
# city centre stays well under the API's 50,000-node limit.
TILE_DEG = 0.004
ATTRIBUTION = "© OpenStreetMap contributors (ODbL). https://www.openstreetmap.org/copyright"


@dataclass
class OsmFeature:
    """One mapped thing: a point, a line or an area, with its tags."""

    osm_id: str
    kind: str  # what the pipeline makes of it, see `classify`
    tags: dict
    # RD metres: one row for a point, a line's vertices, or a ring's.
    xy: np.ndarray
    closed: bool = False

    @property
    def centroid(self) -> np.ndarray:
        return self.xy.mean(axis=0)


@dataclass
class OsmSet:
    features: list[OsmFeature] = field(default_factory=list)
    tiles: int = 0

    def of_kind(self, *kinds: str) -> list[OsmFeature]:
        return [f for f in self.features if f.kind in kinds]

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for feature in self.features:
            counts[feature.kind] = counts.get(feature.kind, 0) + 1
        return {"features": len(self.features), "tiles": self.tiles, **dict(sorted(counts.items()))}


def classify(tags: dict, closed: bool) -> str | None:
    """What a mapped thing becomes in the model, or None to leave it out."""
    if tags.get("leisure") == "outdoor_seating" and closed:
        return "terrace"
    amenity = tags.get("amenity")
    if amenity in ("restaurant", "cafe", "bar", "pub", "biergarten", "ice_cream", "fast_food") and \
            tags.get("outdoor_seating") == "yes":
        return "terrace_venue"
    if tags.get("tourism") == "artwork":
        kind = tags.get("artwork_type", "")
        if kind in ("statue", "sculpture", "bust", "stone", "installation", ""):
            return "statue"
        return None  # murals, graffiti and the like are on walls, not on the street
    if tags.get("historic") in ("memorial", "monument") and tags.get("memorial") not in ("plaque", "stolperstein", "blue_plaque"):
        return "statue"
    if tags.get("highway") == "steps":
        return "steps"
    if amenity == "bench" or tags.get("leisure") == "picnic_table":
        return "bench"
    if amenity == "waste_basket":
        return "bin"
    if amenity == "bicycle_parking":
        return "bike_rack"
    if amenity == "fountain" or tags.get("man_made") == "drinking_fountain" or amenity == "drinking_water":
        return "fountain"
    if tags.get("barrier") == "bollard":
        return "bollard"
    if tags.get("highway") == "street_lamp":
        return "lamp"
    if tags.get("man_made") == "planter" or tags.get("landuse") == "planter":
        return "planter"
    if tags.get("emergency") == "fire_hydrant":
        return "hydrant"
    if amenity in ("post_box", "vending_machine", "parcel_locker"):
        return "box"
    return None


def parse_osm_xml(payload: bytes) -> list[tuple[str, dict, list[tuple[float, float]], bool]]:
    """``(id, tags, [(lon, lat)], closed)`` for every node and way with tags."""
    root = ET.fromstring(payload)
    nodes: dict[str, tuple[float, float]] = {}
    out = []
    for element in root:
        if element.tag != "node":
            continue
        lon, lat = float(element.get("lon")), float(element.get("lat"))
        nodes[element.get("id")] = (lon, lat)
        tags = {t.get("k"): t.get("v") for t in element.findall("tag")}
        if tags:
            out.append((f"n{element.get('id')}", tags, [(lon, lat)], False))
    for element in root:
        if element.tag != "way":
            continue
        tags = {t.get("k"): t.get("v") for t in element.findall("tag")}
        if not tags:
            continue
        refs = [nd.get("ref") for nd in element.findall("nd")]
        coords = [nodes[r] for r in refs if r in nodes]
        if len(coords) < 2:
            continue
        closed = len(refs) > 3 and refs[0] == refs[-1]
        out.append((f"w{element.get('id')}", tags, coords, closed))
    return out


def _tiles(box: BBox, step: float = TILE_DEG):
    nx = max(1, int(np.ceil(box.width / step - 1e-9)))
    ny = max(1, int(np.ceil(box.height / step - 1e-9)))
    for i in range(nx):
        for j in range(ny):
            yield BBox(
                box.xmin + i * box.width / nx, box.ymin + j * box.height / ny,
                box.xmin + (i + 1) * box.width / nx, box.ymin + (j + 1) * box.height / ny,
            )


def fetch_osm(
    bbox_rd: BBox,
    *,
    cache_dir: Path,
    api_url: str = API_URL,
    timeout: float = 120.0,
    max_retries: int = 3,
    max_age_days: float = 30.0,
    pause_s: float = 1.0,
) -> OsmSet:
    """The features of the area the model can use, from cached tiles."""
    from .photos import lonlat_to_rd

    cache_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, OsmFeature] = {}
    result = OsmSet()
    for tile in _tiles(rd_bbox_to_wgs84(bbox_rd)):
        key = "_".join(f"{v:.5f}" for v in tile.as_tuple())
        cached = cache_dir / f"{key}.json"
        if cached.is_file() and (time.time() - cached.stat().st_mtime) / 86400.0 <= max_age_days:
            raw = json.loads(cached.read_text(encoding="utf-8"))
        else:
            response = get_with_retry(
                api_url,
                params={"bbox": ",".join(f"{v:.6f}" for v in tile.as_tuple())},
                timeout=timeout,
                max_retries=max_retries,
                description="OpenStreetMap map download",
            )
            raw = [
                [osm_id, tags, coords, closed]
                for osm_id, tags, coords, closed in parse_osm_xml(response.content)
                if classify(tags, closed) is not None
            ]
            cached.write_text(json.dumps(raw), encoding="utf-8")
            # One request at a time, with room between them: the API's usage
            # policy asks for no more.
            time.sleep(pause_s)
        result.tiles += 1
        for osm_id, tags, coords, closed in raw:
            if osm_id in seen:
                continue
            kind = classify(tags, closed)
            if kind is None:
                continue
            lon = np.array([c[0] for c in coords])
            lat = np.array([c[1] for c in coords])
            x, y = lonlat_to_rd(lon, lat)
            xy = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
            seen[osm_id] = OsmFeature(osm_id, kind, tags, xy, closed)
    reach = bbox_rd.buffered(5.0)
    result.features = [
        f for f in seen.values()
        if reach.xmin <= f.centroid[0] <= reach.xmax and reach.ymin <= f.centroid[1] <= reach.ymax
    ]
    return result


__all__ = ["ATTRIBUTION", "OsmFeature", "OsmSet", "classify", "fetch_osm", "parse_osm_xml"]
