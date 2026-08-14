"""Shared client for the BGT (Basisregistratie Grootschalige Topografie).

Every BGT collection has the same two traps, so they are handled once here.

The API returns the full version history of each object. Reading a collection
naively finds roughly three times as many features as there are things on the
ground, stacked on identical positions: 4834 "trees" where 1488 stand, 6110
road surfaces where 2098 exist. Only rows with an open registration count.

The bbox is interpreted as lon/lat unless told otherwise, and an RD bbox read as
degrees selects nothing at all rather than failing, so ``bbox-crs`` and ``crs``
are always sent.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import requests

from .geo import BBox
from .http_util import ServiceError, get_with_retry

LOG = logging.getLogger(__name__)

BGT_BASE = "https://api.pdok.nl/lv/bgt/ogc/v1/collections"
RD_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"

# Where a run parks the collections it has already read. Two stages want the
# same outlines -- the terrain mesh needs them as breaklines before anything is
# draped, and the surfaces stage needs them again as geometry -- and the BGT is
# slow enough per collection that reading it twice is worth avoiding. Also
# means a re-run over the same area starts instantly.
CACHE_DIRNAME = "_bgt"
_cache_dir: Path | None = None


def use_cache(work_dir: Path | None) -> None:
    """Point the collection cache at a run's work directory, or turn it off."""
    global _cache_dir
    _cache_dir = None if work_dir is None else Path(work_dir) / CACHE_DIRNAME


def _cache_path(collection: str, bbox: BBox) -> Path | None:
    if _cache_dir is None:
        return None
    key = f"{bbox.xmin:.1f}_{bbox.ymin:.1f}_{bbox.xmax:.1f}_{bbox.ymax:.1f}"
    return _cache_dir / f"{collection}_{key}.json.gz"


def collection_url(collection: str) -> str:
    return f"{BGT_BASE}/{collection}/items"


@dataclass
class FetchStats:
    """What a collection read actually contained."""

    collection: str
    raw_features: int = 0
    current_features: int = 0
    superseded_dropped: int = 0
    pages: int = 0
    classes: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.collection}: {self.current_features} current "
            f"({self.raw_features} rows, {self.superseded_dropped} superseded)"
        )


def iter_pages(
    collection: str,
    bbox: BBox,
    *,
    page_limit: int = 1000,
    timeout: float = 180.0,
    max_retries: int = 4,
    max_pages: int = 200,
    session: requests.Session | None = None,
) -> Iterator[dict]:
    """Page through one BGT collection over `bbox`, in RD."""
    session = session or requests.Session()
    url: str | None = collection_url(collection)
    params: dict[str, Any] | None = {
        "bbox": f"{bbox.xmin:.3f},{bbox.ymin:.3f},{bbox.xmax:.3f},{bbox.ymax:.3f}",
        "bbox-crs": RD_URI,
        "crs": RD_URI,
        "limit": page_limit,
        "f": "json",
    }
    seen: set[str] = set()

    for page_index in range(max_pages):
        response = get_with_retry(
            url,
            params=params,
            timeout=timeout,
            max_retries=max_retries,
            session=session,
            description=f"BGT {collection} page {page_index}",
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ServiceError(
                f"BGT {collection} page {page_index} was not JSON: "
                f"{response.text[:200]}"
            ) from exc

        yield payload

        next_links = [
            link.get("href")
            for link in payload.get("links", [])
            if link.get("rel") == "next" and link.get("href")
        ]
        if not next_links:
            return
        if next_links[0] in seen:
            LOG.warning("BGT %s paging repeated a URL; stopping", collection)
            return
        seen.add(next_links[0])
        url, params = next_links[0], None

    LOG.warning("stopped after %d pages of BGT %s", max_pages, collection)


def is_current(properties: dict) -> bool:
    """True for the live version of an object.

    A closed registration means the row has been replaced by a newer version of
    the same object.
    """
    return not properties.get("eind_registratie")


def fetch_current(
    collection: str,
    bbox: BBox,
    *,
    class_field: str | None = None,
    page_limit: int = 1000,
    timeout: float = 180.0,
    max_retries: int = 4,
    max_pages: int = 200,
) -> tuple[list[dict], FetchStats]:
    """Read one collection and keep only the current version of each object."""
    cached = _cache_path(collection, bbox)
    if cached is not None and cached.is_file():
        with gzip.open(cached, "rt", encoding="utf-8") as handle:
            stored = json.load(handle)
        stats = FetchStats(collection=collection, **stored["stats"])
        LOG.info("%s (cached)", stats.summary())
        return stored["features"], stats

    stats = FetchStats(collection=collection)
    features: list[dict] = []
    seen_ids: set[str] = set()

    for payload in iter_pages(
        collection,
        bbox,
        page_limit=page_limit,
        timeout=timeout,
        max_retries=max_retries,
        max_pages=max_pages,
    ):
        stats.pages += 1
        for feature in payload.get("features", []):
            stats.raw_features += 1
            properties = feature.get("properties", {}) or {}

            if not is_current(properties):
                stats.superseded_dropped += 1
                continue

            identifier = properties.get("lokaal_id")
            if identifier:
                if identifier in seen_ids:
                    stats.superseded_dropped += 1
                    continue
                seen_ids.add(identifier)

            features.append(feature)
            if class_field:
                value = str(properties.get(class_field))
                stats.classes[value] = stats.classes.get(value, 0) + 1

    stats.current_features = len(features)
    LOG.info("%s", stats.summary())
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        staging = cached.with_suffix(cached.suffix + ".part")
        with gzip.open(staging, "wt", encoding="utf-8") as handle:
            json.dump(
                {
                    "features": features,
                    "stats": {
                        "pages": stats.pages,
                        "raw_features": stats.raw_features,
                        "current_features": stats.current_features,
                        "superseded_dropped": stats.superseded_dropped,
                        "classes": stats.classes,
                    },
                },
                handle,
            )
        staging.replace(cached)
    return features, stats


def polygon_rings(geometry: dict | None) -> list[list[np.ndarray]]:
    """Normalise Polygon and MultiPolygon into a list of ring groups.

    Each group is ``[outer, hole, hole, ...]`` as 2-D coordinate arrays.
    """
    if not geometry:
        return []
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []

    if kind == "Polygon":
        polygons = [coordinates]
    elif kind == "MultiPolygon":
        polygons = coordinates
    else:
        return []

    groups: list[list[np.ndarray]] = []
    for polygon in polygons:
        rings = []
        for ring in polygon:
            points = np.asarray(ring, dtype=np.float64)
            if points.ndim == 2 and len(points) >= 3:
                rings.append(points[:, :2])
        if rings:
            groups.append(rings)
    return groups


def rasterize_rings(
    ring_groups: list[list[np.ndarray]],
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    *,
    out: np.ndarray | None = None,
    value: Any = True,
) -> np.ndarray:
    """Burn polygons into a raster with the even-odd rule.

    Holes fall out of the even-odd test for free, so an inner courtyard in a
    green area does not get filled in.
    """
    left, bottom, right, top = bounds
    rows, cols = shape
    cell_x = (right - left) / cols
    cell_y = (top - bottom) / rows

    if out is None:
        out = np.zeros(shape, dtype=bool)

    for rings in ring_groups:
        stacked = np.vstack(rings)
        lo = stacked.min(axis=0)
        hi = stacked.max(axis=0)

        col0 = max(0, int((lo[0] - left) / cell_x))
        col1 = min(cols - 1, int((hi[0] - left) / cell_x) + 1)
        row0 = max(0, int((top - hi[1]) / cell_y))
        row1 = min(rows - 1, int((top - lo[1]) / cell_y) + 1)
        if col1 < col0 or row1 < row0:
            continue

        cc, rr = np.meshgrid(
            np.arange(col0, col1 + 1), np.arange(row0, row1 + 1)
        )
        px = left + (cc + 0.5) * cell_x
        py = top - (rr + 0.5) * cell_y

        inside = np.zeros(px.shape, dtype=bool)
        for ring in rings:
            x1, y1 = ring[:-1, 0], ring[:-1, 1]
            x2, y2 = ring[1:, 0], ring[1:, 1]
            for k in range(len(x1)):
                if y1[k] == y2[k]:
                    continue
                crosses = (y1[k] > py) != (y2[k] > py)
                x_at = (x2[k] - x1[k]) * (py - y1[k]) / (y2[k] - y1[k]) + x1[k]
                inside ^= crosses & (px < x_at)

        window = out[row0 : row1 + 1, col0 : col1 + 1]
        if out.dtype == bool:
            window |= inside
        else:
            window[inside] = value

    return out


def ring_area(ring: np.ndarray) -> float:
    """Absolute area of a closed ring, by the shoelace formula."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


__all__ = [
    "BGT_BASE",
    "FetchStats",
    "RD_URI",
    "collection_url",
    "fetch_current",
    "is_current",
    "iter_pages",
    "polygon_rings",
    "rasterize_rings",
    "ring_area",
]
