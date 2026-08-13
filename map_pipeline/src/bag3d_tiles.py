"""3DBAG from its static tiles, for when the API is not answering.

``api.3dbag.nl`` goes down often enough to be the pipeline's least reliable
required source. The same LoD2.2 data is also published as plain CityJSON
files on ``data.3dbag.nl``, which is a different service on a different host:
during the outage this module was written for, every static path answered in
under a second while the API timed out.

Two requests get you a bbox:

* the tile index, published as the ``BAG3D:tiles`` layer of a WFS that is also
  independent of the API, and answers a bbox query with the ids of the tiles
  covering it;
* the tiles themselves, gzipped CityJSON at a path built from the id.

Tiles are cached above the per-area work directory and keyed by 3DBAG version,
so they are shared by every area anyone builds. Building one city warms the
cache for its neighbours, which is the part that actually helps a team: the
second run over a region needs no network at all.

The tiles carry the whole country's tiling, not your bbox, so a small area
pulls in several thousand buildings it does not want. Every ``Building`` object
carries a ``geographicalExtent``, and skipping the ones that miss the bbox
before any geometry is touched is the difference between 20 seconds and 70.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import numpy as np
import requests

from .geo import BBox
from .http_util import ServiceError, get_with_retry, short_error

LOG = logging.getLogger(__name__)

# 3DBAG publishes a dated release and keeps the old ones. There is no "latest"
# alias and the WFS does not advertise which release it indexes, so this is
# pinned and bumped by hand; a wrong value is caught loudly below rather than
# quietly returning nothing.
DEFAULT_VERSION = "v20250903"
DEFAULT_TILE_URL = "https://data.3dbag.nl/{version}/tiles"
DEFAULT_INDEX_URL = "https://data.3dbag.nl/api/BAG3D/wfs"
CACHE_DIRNAME = "_bag3d"


def tile_ids_for_bbox(
    bbox: BBox,
    *,
    index_url: str = DEFAULT_INDEX_URL,
    timeout: float = 60.0,
    max_retries: int = 3,
    session: requests.Session | None = None,
) -> list[str]:
    """The ids of every 3DBAG tile touching `bbox`.

    3DBAG tiles are a quadtree split by building density, not a fixed grid, so
    the ids cannot be worked out from the coordinates -- the index has to be
    asked.
    """
    response = get_with_retry(
        index_url,
        params={
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": "BAG3D:tiles",
            "outputFormat": "application/json",
            "srsName": "EPSG:28992",
            "bbox": (
                f"{bbox.xmin:.3f},{bbox.ymin:.3f},{bbox.xmax:.3f},{bbox.ymax:.3f}"
                ",urn:ogc:def:crs:EPSG::28992"
            ),
        },
        timeout=timeout,
        max_retries=max_retries,
        session=session,
        description="3DBAG tile index",
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ServiceError(
            f"the 3DBAG tile index was not JSON: {response.text[:300]}"
        ) from exc

    ids = []
    for feature in payload.get("features", []):
        tile_id = (feature.get("properties") or {}).get("tile_id")
        if tile_id:
            ids.append(str(tile_id))
    if not ids:
        raise ServiceError(
            f"the 3DBAG tile index has no tiles over {bbox}; check the bbox is "
            f"inside the Netherlands and in RD New"
        )
    return sorted(set(ids))


def tile_url(tile_id: str, *, base_url: str, version: str) -> str:
    """Where a tile id lives. The id is already the path, just slash-separated."""
    parts = tile_id.split("/")
    if len(parts) != 3:
        raise ValueError(f"expected a three-part 3DBAG tile id, got {tile_id!r}")
    base = base_url.format(version=version).rstrip("/")
    return f"{base}/{parts[0]}/{parts[1]}/{parts[2]}/{'-'.join(parts)}.city.json.gz"


def cache_dir_for(work_dir: Path, version: str) -> Path:
    """One tile cache per version, above the per-area directory."""
    return work_dir.parent / CACHE_DIRNAME / version


def ensure_tile(
    tile_id: str,
    cache_dir: Path,
    *,
    base_url: str = DEFAULT_TILE_URL,
    version: str = DEFAULT_VERSION,
    timeout: float = 180.0,
    max_retries: int = 3,
    session: requests.Session | None = None,
) -> Path | None:
    """The tile on disk, downloaded only if it is not cached already.

    Returns None when the tile is missing upstream, which is what a stale
    ``version`` looks like from here. One missing tile is a gap; all of them
    missing is the caller's problem to report.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (tile_id.replace("/", "-") + ".city.json.gz")
    if path.is_file() and path.stat().st_size > 0:
        return path

    url = tile_url(tile_id, base_url=base_url, version=version)
    try:
        response = get_with_retry(
            url,
            timeout=timeout,
            max_retries=max_retries,
            session=session,
            expect_binary=True,
            description=f"3DBAG tile {tile_id}",
        )
    except ServiceError as exc:
        # A 404 here means the pinned release does not have this tile, which
        # get_with_retry raises on immediately rather than retrying.
        LOG.warning("3DBAG tile %s could not be fetched (%s)", tile_id, short_error(exc))
        return None

    # Written through a temporary name so an interrupted run cannot leave a
    # half a tile in the cache for the next one to trust.
    staging = path.with_suffix(path.suffix + ".part")
    staging.write_bytes(response.content)
    staging.replace(path)
    return path


def _extent_misses(extent, bbox: BBox) -> bool:
    """True when a building's own bounds cannot reach the bbox."""
    if not extent or len(extent) < 6:
        return False
    return (
        extent[3] < bbox.xmin
        or extent[0] > bbox.xmax
        or extent[4] < bbox.ymin
        or extent[1] > bbox.ymax
    )


def read_tile(path: Path, bbox: BBox, lod: str, result) -> int:
    """Parse one cached tile into `result`, keeping what can reach `bbox`.

    Returns how many buildings were skipped on their extent alone.
    """
    from .buildings import _extract_feature, decode_vertices

    with gzip.open(path, "rb") as handle:
        document = json.load(handle)

    transform = document.get("transform")
    if not transform:
        raise ServiceError(f"{path.name} has no transform; vertices cannot be decoded")
    objects = document.get("CityObjects") or {}

    # Decoded once for the whole tile and handed to every building in it.
    vertices: np.ndarray | None = None
    skipped = 0
    for object_id, obj in objects.items():
        if obj.get("type") != "Building":
            continue
        if _extent_misses(obj.get("geographicalExtent"), bbox):
            skipped += 1
            continue
        if vertices is None:
            vertices = decode_vertices(document.get("vertices", []), transform)
        children = obj.get("children") or []
        feature = {
            "CityObjects": {
                object_id: obj,
                **{k: objects[k] for k in children if k in objects},
            }
        }
        building = _extract_feature(feature, transform, lod, result, vertices=vertices)
        if building is not None:
            result.buildings.append(building)
    return skipped


def fetch_buildings_from_tiles(
    bbox: BBox,
    work_dir: Path,
    *,
    lod: str = "2.2",
    base_url: str = DEFAULT_TILE_URL,
    index_url: str = DEFAULT_INDEX_URL,
    version: str = DEFAULT_VERSION,
    timeout: float = 180.0,
    max_retries: int = 3,
):
    """Every 3DBAG building over `bbox`, from the static tiles."""
    from .buildings import BuildingSet

    session = requests.Session()
    tile_ids = tile_ids_for_bbox(
        bbox,
        index_url=index_url,
        timeout=min(timeout, 60.0),
        max_retries=max_retries,
        session=session,
    )
    cache = cache_dir_for(work_dir, version)
    cached = sum(1 for t in tile_ids if (cache / (t.replace("/", "-") + ".city.json.gz")).is_file())
    LOG.info(
        "3DBAG static tiles: %d over this bbox, %d already cached in %s",
        len(tile_ids),
        cached,
        cache,
    )

    paths = []
    for tile_id in tile_ids:
        path = ensure_tile(
            tile_id,
            cache,
            base_url=base_url,
            version=version,
            timeout=timeout,
            max_retries=max_retries,
            session=session,
        )
        if path is not None:
            paths.append(path)
    if not paths:
        raise ServiceError(
            f"none of the {len(tile_ids)} 3DBAG tiles over this bbox exist in "
            f"release {version}; the tile index and the release have drifted "
            f"apart, so set buildings.tiles_version to a current release "
            f"(they are listed at https://3dbag.nl/en/download)"
        )

    result = BuildingSet(lod=lod)
    result.pages_fetched = len(paths)
    result.source = "tiles"
    result.source_version = version
    skipped = 0
    for path in paths:
        skipped += read_tile(path, bbox, lod, result)
    LOG.info(
        "3DBAG tiles gave %d buildings near the bbox; %d elsewhere in the tiles "
        "were skipped on their own extent",
        len(result),
        skipped,
    )
    if not result.buildings:
        raise ServiceError(
            f"the 3DBAG tiles over {bbox} held no usable buildings at LoD {lod}"
        )
    return result


__all__ = [
    "CACHE_DIRNAME",
    "DEFAULT_INDEX_URL",
    "DEFAULT_TILE_URL",
    "DEFAULT_VERSION",
    "cache_dir_for",
    "ensure_tile",
    "fetch_buildings_from_tiles",
    "read_tile",
    "tile_ids_for_bbox",
    "tile_url",
]
