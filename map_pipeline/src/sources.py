"""The data sources a model can be built from.

One registry, used by everything that needs to know what a run depends on: the
pipeline checks the enabled sources are reachable before it starts, and the UI
lists them with their live status. Before this the service list was written out
once in the preflight and again as checkboxes in the page, which is exactly the
sort of duplication that goes stale without anyone noticing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Source:
    """One dataset the pipeline can pull in."""

    id: str
    label: str
    provider: str
    contributes: str
    # Config keys to flip, as a path into the config dict. Empty for sources
    # that are always on.
    toggles: tuple[tuple[str, ...], ...] = ()
    # Where in the config the service URL lives, for the reachability check.
    url_path: tuple[str, ...] = ()
    # Used when the URL is not in the config.
    fallback_url: str = ""
    required: bool = False
    note: str = ""
    # What to ask when only checking the service is alive. A feature endpoint
    # without a bbox asks the server to enumerate a whole national dataset:
    # 3DBAG takes fourteen seconds to answer that, which a reachability probe
    # reads as an outage. The collection metadata answers in three.
    probe_url: str = ""

    def url(self, config: dict) -> str:
        node: Any = config
        for key in self.url_path:
            if not isinstance(node, dict) or key not in node:
                return self.fallback_url
            node = node[key]
        return str(node) if self.url_path else self.fallback_url

    def probe(self, config: dict) -> str:
        return self.probe_url or self.url(config)


# Ordered as they appear in a run.
SOURCES: tuple[Source, ...] = (
    Source(
        id="terrain",
        label="Terrain (AHN)",
        provider="PDOK",
        contributes="ground heights, and the canopy model trees are measured against",
        url_path=("terrain", "wcs_url"),
        required=True,
    ),
    Source(
        id="aerial",
        label="Aerial imagery",
        provider="PDOK / Beeldmateriaal",
        contributes="the photo draped over the ground and the roofs",
        url_path=("aerial", "wms_url"),
        required=True,
    ),
    Source(
        id="buildings",
        label="Buildings (3DBAG)",
        provider="TU Delft / Kadaster",
        contributes="building geometry at LoD 2.2, with real roof shapes",
        url_path=("buildings", "api_url"),
        required=True,
        probe_url="https://api.3dbag.nl/collections/pand",
    ),
    Source(
        id="usage",
        label="Building function (BAG)",
        provider="PDOK",
        contributes="what each building is for, which picks its ground storey",
        toggles=(("usage", "enabled"),),
        fallback_url="https://service.pdok.nl/lv/bag/wfs/v2_0",
    ),
    Source(
        id="trees",
        label="Trees (BGT)",
        provider="PDOK",
        contributes="tree positions, given heights from AHN",
        toggles=(("trees", "enabled"),),
        url_path=("trees", "api_url"),
    ),
    Source(
        id="water",
        label="Water surfaces (BGT)",
        provider="PDOK",
        contributes="canal outlines, replacing the interpolated bulge in the DTM",
        toggles=(("surfaces", "water"),),
        url_path=("trees", "api_url"),
        note="same service as the other BGT layers",
    ),
    Source(
        id="land_cover",
        label="Land cover (BGT)",
        provider="PDOK",
        contributes="road, green and paved classes, exported and blended into the aerial",
        toggles=(("surfaces", "land_cover"),),
        url_path=("trees", "api_url"),
    ),
    Source(
        id="furniture",
        label="Street furniture (BGT)",
        provider="PDOK",
        contributes="lampposts, bollards, sign posts and benches",
        toggles=(("furniture", "enabled"),),
        url_path=("trees", "api_url"),
    ),
    Source(
        id="vehicles",
        label="Cars and boats (BGT)",
        provider="PDOK",
        contributes=(
            "cars laid out in the parking bays, boats moored between the "
            "mooring posts"
        ),
        toggles=(("vehicles", "cars"), ("vehicles", "boats")),
        url_path=("trees", "api_url"),
        note="boats need water surfaces on too",
    ),
    Source(
        id="rails",
        label="Railways (BGT)",
        provider="PDOK",
        contributes="railway, tram and metro track, on its own ballast bed",
        toggles=(("rails", "enabled"),),
        url_path=("trees", "api_url"),
    ),
)

BY_ID = {source.id: source for source in SOURCES}


def enabled_sources(config: dict) -> list[Source]:
    """Which sources a config actually uses."""
    active = []
    for source in SOURCES:
        if source.required or not source.toggles:
            active.append(source)
            continue
        if any(_read(config, path) for path in source.toggles):
            active.append(source)
    return active


def _read(config: dict, path: tuple[str, ...]) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def apply_selection(config: dict, selected: set[str]) -> dict:
    """Turn the optional sources on or off to match `selected`.

    Required sources are left alone: a model without terrain, imagery or
    buildings is not a model.
    """
    for source in SOURCES:
        if source.required or not source.toggles:
            continue
        wanted = source.id in selected
        for path in source.toggles:
            node = config
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = wanted
    return config


def health_targets(config: dict, sources: list[Source] | None = None) -> dict[str, str]:
    """Distinct URLs to probe, keyed by the sources that depend on each.

    Five of the eight sources sit behind the same two hosts, so probing per
    source would ask the same server the same question repeatedly and make an
    outage look like several separate failures.
    """
    targets: dict[str, list[str]] = {}
    for source in sources if sources is not None else list(SOURCES):
        url = source.probe(config)
        if url:
            targets.setdefault(url, []).append(source.id)
    return {url: ",".join(ids) for url, ids in targets.items()}


__all__ = [
    "BY_ID",
    "SOURCES",
    "Source",
    "apply_selection",
    "enabled_sources",
    "health_targets",
]
