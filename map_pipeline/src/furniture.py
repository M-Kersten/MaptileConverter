"""Street furniture: lampposts, bollards and benches from the BGT.

None of this is structurally important, and that is rather the point. A street
with nothing on it reads as a model; the same street with lampposts along the
kerb and benches by the water reads as a place. There are about 2000 poles and
200 pieces of furniture in a square kilometre of Utrecht, so they are cheap
primitives sharing one material and one draw call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bgt import fetch_current
from .geo import BBox

LOG = logging.getLogger(__name__)

# BGT plus_type values worth drawing, and what to draw them as.
LAMP_TYPES = {"lichtmast"}
BOLLARD_TYPES = {"afsluitpaal", "poller"}
SIGN_TYPES = {"verkeersbordpaal", "verkeersregelinstallatiepaal"}
BENCH_TYPES = {"bank"}

KIND_LAMP = 0
KIND_BOLLARD = 1
KIND_BENCH = 2
KIND_SIGN = 3

KIND_NAMES = {
    KIND_LAMP: "lamppost",
    KIND_BOLLARD: "bollard",
    KIND_BENCH: "bench",
    KIND_SIGN: "sign",
}


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

    wanted: dict[str, dict[str, int]] = {
        "paal": {
            **{t: KIND_LAMP for t in LAMP_TYPES},
            **{t: KIND_BOLLARD for t in BOLLARD_TYPES},
            **{t: KIND_SIGN for t in SIGN_TYPES},
        },
        "straatmeubilair": {t: KIND_BENCH for t in BENCH_TYPES},
    }

    for collection, mapping in wanted.items():
        features, stats = fetch_current(
            collection, bbox, class_field="plus_type", **fetch_kwargs
        )
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
    "KIND_BENCH",
    "KIND_BOLLARD",
    "KIND_LAMP",
    "KIND_NAMES",
    "KIND_SIGN",
    "FurnitureSet",
    "build_furniture",
    "save_furniture",
]
