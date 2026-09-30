"""Street furniture: the small things a street is not a street without.

None of this is structurally important, and that is rather the point. A street
with nothing on it reads as a model; the same street with lampposts along the
kerb and benches by the water reads as a place. They are cheap primitives
sharing one atlas, so a couple of thousand objects cost one draw call.

The BGT registers far more of this than the pipeline used to draw. Over one
square kilometre of Utrecht centre it has 1058 lampposts, 918 bollards and 148
benches -- which were kept -- alongside 50 playgrounds, 41 waste collection
points, 40 tram catenary masts, 8 art objects, 6 advertising columns, 51
electrical cabinets, 3 bus shelters, a memorial monument, a picnic table and
exactly one flagpole, all of which were fetched and thrown away. They are the
difference between a street and a street someone maintains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bgt import fetch_current
from .geo import BBox

LOG = logging.getLogger(__name__)

KIND_LAMP = 0
KIND_BOLLARD = 1
KIND_BENCH = 2
KIND_SIGN = 3
KIND_FLAGPOLE = 4
KIND_SHELTER = 5
KIND_COLUMN = 6
KIND_ART = 7
KIND_MONUMENT = 8
KIND_PLAYGROUND = 9
KIND_PICNIC = 10
KIND_CABINET = 11
KIND_WASTE = 12
KIND_CATENARY = 13

KIND_NAMES = {
    KIND_LAMP: "lamppost",
    KIND_BOLLARD: "bollard",
    KIND_BENCH: "bench",
    KIND_SIGN: "sign",
    KIND_FLAGPOLE: "flagpole",
    KIND_SHELTER: "bus shelter",
    KIND_COLUMN: "ad column",
    KIND_ART: "art object",
    KIND_MONUMENT: "monument",
    KIND_PLAYGROUND: "playground",
    KIND_PICNIC: "picnic table",
    KIND_CABINET: "cabinet",
    KIND_WASTE: "waste point",
    KIND_CATENARY: "catenary mast",
}

# Which BGT collection and plus_type becomes which kind.
#
# "niet-bgt" appears throughout the BGT and means the surveyor recorded an
# object the standard has no type for. It is deliberately not mapped: drawing
# something specific for "unknown" is how a model ends up with 70 identical
# sheds in it.
WANTED: dict[str, dict[str, int]] = {
    "paal": {
        "lichtmast": KIND_LAMP,
        "afsluitpaal": KIND_BOLLARD,
        "poller": KIND_BOLLARD,
        "verkeersbordpaal": KIND_SIGN,
        "verkeersregelinstallatiepaal": KIND_SIGN,
        "vlaggenmast": KIND_FLAGPOLE,
    },
    "straatmeubilair": {
        "bank": KIND_BENCH,
        "abri": KIND_SHELTER,
        "reclamezuil": KIND_COLUMN,
        "kunstobject": KIND_ART,
        "herdenkingsmonument": KIND_MONUMENT,
        "speelvoorziening": KIND_PLAYGROUND,
        "speeltoestel": KIND_PLAYGROUND,
        "picknicktafel": KIND_PICNIC,
        "zitelement": KIND_BENCH,
    },
    "kast": {"elektrakast": KIND_CABINET},
    "bak": {
        "afval apart plaats": KIND_WASTE,
        "afvalbak": KIND_WASTE,
    },
    "mast": {
        "bovenleidingmast": KIND_CATENARY,
        "hoogspanningsmast": KIND_CATENARY,
    },
}

# Kept for callers that used the old names.
LAMP_TYPES = {"lichtmast"}
BOLLARD_TYPES = {"afsluitpaal", "poller"}
SIGN_TYPES = {"verkeersbordpaal", "verkeersregelinstallatiepaal"}
BENCH_TYPES = {"bank"}


@dataclass
class FurnitureSet:
    """Positions and kinds, ready for the Blender stage to instance."""

    xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    ground_z_nap: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.xy))

    def stats(self) -> dict:
        return {"count": len(self), **self.counts}


def build_furniture(
    bbox: BBox,
    work_dir: Path,
    *,
    furniture_cfg: dict,
    terrain,
) -> FurnitureSet:
    """Fetch poles and street furniture, and put them on the terrain."""
    result = FurnitureSet()
    points: list[tuple[float, float]] = []
    kinds: list[int] = []

    fetch_kwargs = {
        "page_limit": int(furniture_cfg["page_limit"]),
        "timeout": float(furniture_cfg["timeout_s"]),
        "max_retries": int(furniture_cfg["max_retries"]),
        "max_pages": int(furniture_cfg["max_pages"]),
    }

    for collection, mapping in WANTED.items():
        try:
            features, stats = fetch_current(
                collection, bbox, class_field="plus_type", **fetch_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - one empty collection is not fatal
            # kast, bak and mast are often empty outside a town centre, and a
            # street with no lampposts because the bin collection failed would
            # be a poor trade.
            LOG.warning("could not read BGT %s: %s", collection, exc)
            continue
        for feature in features:
            plus_type = str((feature.get("properties") or {}).get("plus_type") or "")
            kind = mapping.get(plus_type)
            if kind is None:
                continue
            geometry = feature.get("geometry") or {}
            if geometry.get("type") != "Point":
                continue
            x, y = geometry["coordinates"][:2]
            if not (bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax):
                continue
            points.append((float(x), float(y)))
            kinds.append(kind)

    if not points:
        LOG.info("no street furniture found")
        return result

    result.xy = np.asarray(points, dtype=np.float64)
    result.kind = np.asarray(kinds, dtype=np.int32)
    result.ground_z_nap = np.asarray(
        terrain.sample(result.xy[:, 0], result.xy[:, 1]), dtype=np.float64
    )
    result.counts = {
        KIND_NAMES[k]: int((result.kind == k).sum()) for k in KIND_NAMES
    }
    LOG.info(
        "street furniture: %s",
        ", ".join(f"{v} {k}" for k, v in result.counts.items() if v),
    )

    save_furniture(result, work_dir / "furniture.npz")
    return result


def save_furniture(furniture: FurnitureSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xy=furniture.xy,
        ground_z_nap=furniture.ground_z_nap,
        kind=furniture.kind,
    )
    LOG.info("wrote %s (%d objects)", path, len(furniture))
    return path


__all__ = [
    "KIND_ART",
    "KIND_BENCH",
    "KIND_BOLLARD",
    "KIND_CABINET",
    "KIND_CATENARY",
    "KIND_COLUMN",
    "KIND_FLAGPOLE",
    "KIND_LAMP",
    "KIND_MONUMENT",
    "KIND_NAMES",
    "KIND_PICNIC",
    "KIND_PLAYGROUND",
    "KIND_SHELTER",
    "KIND_SIGN",
    "KIND_WASTE",
    "WANTED",
    "FurnitureSet",
    "build_furniture",
    "save_furniture",
]
