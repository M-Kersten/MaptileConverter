"""Building function from the BAG.

3DBAG says how tall a building is and when it was built, but not what it is for.
The BAG does: every verblijfsobject carries a ``gebruiksdoel`` (residential,
office, retail, industrial, and so on) and a ``pandidentificatie`` that is the
same building id 3DBAG uses, so the two join on a key rather than by geometry.

This matters for the ground storey. Shopfronts along a residential street look
wrong, and giving every building the same ground floor undoes much of the point
of splitting it off. With function in hand, only retail and mixed buildings get
shopfronts; housing gets doors and windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .geo import BBox
from .http_util import get_with_retry

LOG = logging.getLogger(__name__)

WFS_URL = "https://service.pdok.nl/lv/bag/wfs/v2_0"

# Coarse groups the pipeline actually acts on. The BAG has more values than
# this, and a building can carry several at once.
FUNCTION_HOME = 0
FUNCTION_RETAIL = 1
FUNCTION_OFFICE = 2
FUNCTION_INDUSTRY = 3
FUNCTION_PUBLIC = 4
FUNCTION_OTHER = 5

FUNCTION_NAMES = {
    FUNCTION_HOME: "residential",
    FUNCTION_RETAIL: "retail",
    FUNCTION_OFFICE: "office",
    FUNCTION_INDUSTRY: "industrial",
    FUNCTION_PUBLIC: "public",
    FUNCTION_OTHER: "other",
}

# Mapped from the BAG's own vocabulary.
_BAG_TO_GROUP = {
    "woonfunctie": FUNCTION_HOME,
    "winkelfunctie": FUNCTION_RETAIL,
    "kantoorfunctie": FUNCTION_OFFICE,
    "industriefunctie": FUNCTION_INDUSTRY,
    "bijeenkomstfunctie": FUNCTION_PUBLIC,
    "onderwijsfunctie": FUNCTION_PUBLIC,
    "gezondheidszorgfunctie": FUNCTION_PUBLIC,
    "sportfunctie": FUNCTION_PUBLIC,
    "logiesfunctie": FUNCTION_PUBLIC,
    "celfunctie": FUNCTION_PUBLIC,
    "overige gebruiksfunctie": FUNCTION_OTHER,
}

# When a building holds several functions, the one that decides its ground floor
# is the most street-facing of them.
_PRIORITY = [
    FUNCTION_RETAIL,
    FUNCTION_PUBLIC,
    FUNCTION_OFFICE,
    FUNCTION_INDUSTRY,
    FUNCTION_HOME,
    FUNCTION_OTHER,
]


@dataclass
class UsageSet:
    """Function group per BAG building id."""

    by_pand: dict[str, int] = field(default_factory=dict)
    objects_read: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.by_pand)

    def group_for(self, identifier: str) -> int | None:
        """Look up a 3DBAG identifier, which carries an NL.IMBAG.Pand prefix."""
        return self.by_pand.get(identifier.rsplit(".", 1)[-1])

    def stats(self) -> dict:
        return {
            "buildings_with_function": len(self.by_pand),
            "verblijfsobjecten_read": self.objects_read,
            **self.counts,
        }


def _group_from_gebruiksdoel(value: str) -> int:
    """Reduce a comma-separated gebruiksdoel to one group."""
    groups = {
        _BAG_TO_GROUP.get(part.strip().lower(), FUNCTION_OTHER)
        for part in value.split(",")
        if part.strip()
    }
    for candidate in _PRIORITY:
        if candidate in groups:
            return candidate
    return FUNCTION_OTHER


def fetch_usage(
    bbox: BBox,
    *,
    usage_cfg: dict,
) -> UsageSet:
    """Read BAG verblijfsobjecten over `bbox` and group them by building.

    The WFS pages with startIndex rather than follow-on links, so paging is
    driven by a counter until a short page comes back.
    """
    result = UsageSet()
    session = requests.Session()
    page_size = int(usage_cfg["page_limit"])
    max_pages = int(usage_cfg["max_pages"])
    seen_groups: dict[str, set[int]] = {}

    for page in range(max_pages):
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": "bag:verblijfsobject",
            "srsName": "EPSG:28992",
            "bbox": (
                f"{bbox.xmin:.3f},{bbox.ymin:.3f},"
                f"{bbox.xmax:.3f},{bbox.ymax:.3f},EPSG:28992"
            ),
            "count": page_size,
            "startIndex": page * page_size,
            "outputFormat": "application/json",
        }
        response = get_with_retry(
            WFS_URL,
            params=params,
            timeout=float(usage_cfg["timeout_s"]),
            max_retries=int(usage_cfg["max_retries"]),
            session=session,
            description=f"BAG verblijfsobject page {page}",
        )
        try:
            payload = response.json()
        except ValueError:
            LOG.warning("BAG WFS returned non-JSON; stopping at page %d", page)
            break

        features = payload.get("features", [])
        for feature in features:
            properties = feature.get("properties", {}) or {}
            pand = properties.get("pandidentificatie")
            purpose = properties.get("gebruiksdoel")
            if not pand or not purpose:
                continue
            result.objects_read += 1
            seen_groups.setdefault(str(pand), set()).add(
                _group_from_gebruiksdoel(str(purpose))
            )

        if len(features) < page_size:
            break

    # A building holding several units takes the most street-facing function.
    for pand, groups in seen_groups.items():
        for candidate in _PRIORITY:
            if candidate in groups:
                result.by_pand[pand] = candidate
                break

    result.counts = {
        FUNCTION_NAMES[group]: sum(
            1 for value in result.by_pand.values() if value == group
        )
        for group in FUNCTION_NAMES
    }
    LOG.info(
        "BAG function for %d buildings from %d units: %s",
        len(result.by_pand),
        result.objects_read,
        ", ".join(f"{v} {k}" for k, v in result.counts.items() if v),
    )
    return result


__all__ = [
    "FUNCTION_HOME",
    "FUNCTION_INDUSTRY",
    "FUNCTION_NAMES",
    "FUNCTION_OFFICE",
    "FUNCTION_OTHER",
    "FUNCTION_PUBLIC",
    "FUNCTION_RETAIL",
    "UsageSet",
    "fetch_usage",
]
