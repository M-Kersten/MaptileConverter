"""Building function from the BAG.

3DBAG says how tall a building is and when it was built, but not what it is for.
The BAG does. Its ``pand`` layer carries, per building, the purposes of every
unit in it (``gebruiksdoel``, one or several of eleven) and how many units it
has (``aantal_verblijfsobjecten``), under the same building id 3DBAG uses, so
the two join on a key rather than by geometry.

Two things come out of that:

* **Which function.** Each purpose is kept as its own group. They used to be
  six: assembly, education, health care, sport, lodging and detention were
  all "public", which made a school, a hospital and a restaurant the same
  building.
* **No function at all.** A building with no units has nothing anyone lives,
  works or shops in: a garden shed, a garage, a transformer kiosk, a barn.
  Before this it fell back on the housing facade, so every shed got a front
  door and bay windows.

The verblijfsobject layer was read before, one record per unit; the pand layer
is one record per building, which is fewer records and the only one that can
say a building has none.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import requests

from .geo import BBox
from .http_util import get_with_retry

LOG = logging.getLogger(__name__)

WFS_URL = "https://service.pdok.nl/lv/bag/wfs/v2_0"

# One group per BAG purpose. The first six keep the codes they always had;
# FUNCTION_PUBLIC is now the assembly function alone.
FUNCTION_HOME = 0
FUNCTION_RETAIL = 1
FUNCTION_OFFICE = 2
FUNCTION_INDUSTRY = 3
FUNCTION_PUBLIC = 4
FUNCTION_OTHER = 5
FUNCTION_EDUCATION = 6
FUNCTION_HEALTH = 7
FUNCTION_SPORT = 8
FUNCTION_LODGING = 9
FUNCTION_CELL = 10
# Registered, with no unit in it.
FUNCTION_NONE = 11

FUNCTION_NAMES = {
    FUNCTION_HOME: "residential",
    FUNCTION_RETAIL: "retail",
    FUNCTION_OFFICE: "office",
    FUNCTION_INDUSTRY: "industrial",
    FUNCTION_PUBLIC: "assembly",
    FUNCTION_OTHER: "other",
    FUNCTION_EDUCATION: "education",
    FUNCTION_HEALTH: "health care",
    FUNCTION_SPORT: "sport",
    FUNCTION_LODGING: "lodging",
    FUNCTION_CELL: "detention",
    FUNCTION_NONE: "no units",
}

# Mapped from the BAG's own vocabulary.
_BAG_TO_GROUP = {
    "woonfunctie": FUNCTION_HOME,
    "winkelfunctie": FUNCTION_RETAIL,
    "kantoorfunctie": FUNCTION_OFFICE,
    "industriefunctie": FUNCTION_INDUSTRY,
    "bijeenkomstfunctie": FUNCTION_PUBLIC,
    "onderwijsfunctie": FUNCTION_EDUCATION,
    "gezondheidszorgfunctie": FUNCTION_HEALTH,
    "sportfunctie": FUNCTION_SPORT,
    "logiesfunctie": FUNCTION_LODGING,
    "celfunctie": FUNCTION_CELL,
    "overige gebruiksfunctie": FUNCTION_OTHER,
}

# When a building holds several functions, the one that decides its ground
# floor is the most street-facing of them.
_PRIORITY = [
    FUNCTION_RETAIL,
    FUNCTION_PUBLIC,
    FUNCTION_LODGING,
    FUNCTION_OFFICE,
    FUNCTION_HEALTH,
    FUNCTION_EDUCATION,
    FUNCTION_SPORT,
    FUNCTION_INDUSTRY,
    FUNCTION_CELL,
    FUNCTION_HOME,
    FUNCTION_OTHER,
]


@dataclass
class PandUse:
    """What the BAG says about one building."""

    group: int
    groups: frozenset[int]
    units: int
    status: str = ""


@dataclass
class UsageSet:
    """Function per BAG building id."""

    by_pand: dict[str, int] = field(default_factory=dict)
    details: dict[str, PandUse] = field(default_factory=dict)
    objects_read: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.by_pand)

    @staticmethod
    def _key(identifier: str) -> str:
        """A 3DBAG identifier carries an NL.IMBAG.Pand prefix."""
        return identifier.rsplit(".", 1)[-1]

    def group_for(self, identifier: str) -> int | None:
        return self.by_pand.get(self._key(identifier))

    def use_for(self, identifier: str) -> PandUse | None:
        return self.details.get(self._key(identifier))

    def stats(self) -> dict:
        return {
            "buildings_with_function": len(self.by_pand),
            "buildings_read": self.objects_read,
            **self.counts,
        }


def groups_from_gebruiksdoel(value: str) -> set[int]:
    return {
        _BAG_TO_GROUP.get(part.strip().lower(), FUNCTION_OTHER)
        for part in str(value or "").split(",")
        if part.strip()
    }


def primary_group(groups) -> int:
    for candidate in _PRIORITY:
        if candidate in groups:
            return candidate
    return FUNCTION_OTHER


def _group_from_gebruiksdoel(value: str) -> int:
    """Reduce a comma-separated gebruiksdoel to one group."""
    return primary_group(groups_from_gebruiksdoel(value))


def parse_pand(properties: dict) -> tuple[str, PandUse] | None:
    """One pand record as (identifier, use), or None when it has no id."""
    identifier = str(properties.get("identificatie") or "").strip()
    if not identifier:
        return None
    try:
        units = int(properties.get("aantal_verblijfsobjecten") or 0)
    except (TypeError, ValueError):
        units = 0
    groups = groups_from_gebruiksdoel(properties.get("gebruiksdoel") or "")
    if units == 0 or not groups:
        groups = {FUNCTION_NONE}
    return identifier, PandUse(
        group=primary_group(groups) if FUNCTION_NONE not in groups else FUNCTION_NONE,
        groups=frozenset(groups),
        units=units,
        status=str(properties.get("status") or ""),
    )


def fetch_usage(
    bbox: BBox,
    *,
    usage_cfg: dict,
) -> UsageSet:
    """Read BAG buildings over `bbox` with their functions and unit counts.

    The WFS pages with startIndex rather than follow-on links, so paging is
    driven by a counter until a short page comes back.
    """
    result = UsageSet()
    session = requests.Session()
    page_size = int(usage_cfg["page_limit"])
    max_pages = int(usage_cfg["max_pages"])

    for page in range(max_pages):
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": "bag:pand",
            "srsName": "EPSG:28992",
            "bbox": (
                f"{bbox.xmin:.3f},{bbox.ymin:.3f},"
                f"{bbox.xmax:.3f},{bbox.ymax:.3f},EPSG:28992"
            ),
            "count": page_size,
            "startIndex": page * page_size,
            "outputFormat": "application/json",
            # The outline is not needed: 3DBAG has the shape, this is only
            # about what the building is for.
            "propertyName": "identificatie,gebruiksdoel,aantal_verblijfsobjecten,status",
        }
        response = get_with_retry(
            WFS_URL,
            params=params,
            timeout=float(usage_cfg["timeout_s"]),
            max_retries=int(usage_cfg["max_retries"]),
            session=session,
            description=f"BAG pand page {page}",
        )
        try:
            payload = response.json()
        except ValueError:
            LOG.warning("BAG WFS returned non-JSON; stopping at page %d", page)
            break

        features = payload.get("features", [])
        for feature in features:
            parsed = parse_pand(feature.get("properties", {}) or {})
            if parsed is None:
                continue
            identifier, use = parsed
            result.objects_read += 1
            result.details[identifier] = use
            result.by_pand[identifier] = use.group

        if len(features) < page_size:
            break

    result.counts = {
        FUNCTION_NAMES[group]: sum(
            1 for value in result.by_pand.values() if value == group
        )
        for group in FUNCTION_NAMES
    }
    LOG.info(
        "BAG function for %d buildings: %s",
        len(result.by_pand),
        ", ".join(f"{v} {k}" for k, v in result.counts.items() if v),
    )
    return result


__all__ = [
    "FUNCTION_CELL",
    "FUNCTION_EDUCATION",
    "FUNCTION_HEALTH",
    "FUNCTION_HOME",
    "FUNCTION_INDUSTRY",
    "FUNCTION_LODGING",
    "FUNCTION_NAMES",
    "FUNCTION_NONE",
    "FUNCTION_OFFICE",
    "FUNCTION_OTHER",
    "FUNCTION_PUBLIC",
    "FUNCTION_RETAIL",
    "FUNCTION_SPORT",
    "PandUse",
    "UsageSet",
    "fetch_usage",
    "groups_from_gebruiksdoel",
    "parse_pand",
    "primary_group",
]
