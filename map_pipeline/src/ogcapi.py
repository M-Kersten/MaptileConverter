"""Reading an OGC API Features collection over an RD bbox.

PDOK publishes most of its datasets this way, and every one needs the same
three things done right: the bbox and the answer both in RD (`bbox-crs` and
`crs`, without which an RD bbox is read as degrees and selects nothing), the
pages followed through their `next` links, and the result kept so a second
run over the same square does not ask again. The BGT has its own client in
`bgt`, because its version history needs filtering as well.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

from .geo import BBox
from .http_util import get_with_retry

LOG = logging.getLogger(__name__)

RD_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"


def fetch_items(
    url: str,
    bbox: BBox,
    *,
    description: str,
    timeout: float = 120.0,
    max_retries: int = 3,
    page_limit: int = 1000,
    max_pages: int = 50,
    cache_dir: Path | None = None,
    cache_name: str = "",
) -> list[dict]:
    """Every feature of the collection at `url` (its ``/items``) in `bbox`.

    Raises on failure: whether a missing dataset is fatal is the caller's
    business.
    """
    cache = None
    if cache_dir is not None and cache_name:
        key = f"{bbox.xmin:.1f}_{bbox.ymin:.1f}_{bbox.xmax:.1f}_{bbox.ymax:.1f}"
        cache = Path(cache_dir) / f"{cache_name}_{key}.json.gz"
        if cache.is_file():
            with gzip.open(cache, "rt", encoding="utf-8") as handle:
                return json.load(handle)

    params: dict[str, Any] | None = {
        "bbox": f"{bbox.xmin:.3f},{bbox.ymin:.3f},{bbox.xmax:.3f},{bbox.ymax:.3f}",
        "bbox-crs": RD_URI,
        "crs": RD_URI,
        "limit": page_limit,
        "f": "json",
    }
    features: list[dict] = []
    next_url: str | None = url
    seen: set[str] = set()
    for page in range(max_pages):
        response = get_with_retry(
            next_url,
            params=params,
            timeout=timeout,
            max_retries=max_retries,
            description=f"{description} page {page}",
        )
        payload = response.json()
        features.extend(payload.get("features") or [])
        links = [
            link.get("href")
            for link in payload.get("links") or []
            if link.get("rel") == "next" and link.get("href")
        ]
        if not links or links[0] in seen:
            break
        seen.add(links[0])
        next_url, params = links[0], None

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        staging = cache.with_suffix(cache.suffix + ".part")
        with gzip.open(staging, "wt", encoding="utf-8") as handle:
            json.dump(features, handle)
        staging.replace(cache)
    return features


__all__ = ["RD_URI", "fetch_items"]
