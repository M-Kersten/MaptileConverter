"""Config loading, with defaults for everything the plan leaves implicit.

``config.json`` only has to carry the interesting bits. Every knob below has a
default that works for a 1 km² area, so a minimal config stays readable.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .geo import BBox, GeoContext, parse_bbox, validate_bbox

LOG = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "name": "demo_area",
    "aerial": {
        "layer": "Luchtfoto Actueel Ortho 8cm RGB",
        "size_px": 4096,
        # The PDOK WMS advertises MaxWidth/MaxHeight of 2500 and answers
        # anything larger with a ServiceException, so the mosaic is not
        # optional above 2500 px. Kept below the cap for headroom.
        "max_request_px": 2000,
        "wms_url": "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0",
        "wmts_url": "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0",
        # The service only offers image/jpeg for GetMap. Tiles come back as
        # JPEG and are written out as a single PNG.
        "request_format": "image/jpeg",
        "timeout_s": 180,
        "max_retries": 4,
    },
    "terrain": {
        "ahn_model": "DTM",
        "resolution_m": 0.5,
        "mesh_vertices_per_side": 257,
        "wcs_url": "https://service.pdok.nl/rws/ahn/wcs/v1_0",
        "timeout_s": 300,
        "max_retries": 4,
        # DTM strips buildings out, so dense city centres come back mostly
        # nodata. Filling is a normal part of the job, not an error path.
        "max_nodata_fraction": 0.95,
        "smooth_iterations": 1,
    },
    "buildings": {
        "lod": "2.2",
        "api_url": "https://api.3dbag.nl/collections/pand/items",
        "page_limit": 500,
        "timeout_s": 180,
        "max_retries": 4,
        "max_pages": 2000,
        # Anything taller than this in a 1 km² Dutch area is a data error.
        "max_height_m": 200.0,
        "min_height_m": 1.0,
        # Buildings are pushed this far below the terrain under their footprint
        # so no gap opens up where AHN and the 3DBAG ground level disagree.
        "ground_skirt_m": 0.5,
        "merge": "single",
        # "centroid" assigns each building to the area containing its centre,
        # which keeps the model bounded without cutting geometry open.
        # "intersect" keeps every building the API returns.
        "clip_mode": "centroid",
    },
    "facade": {
        # One storey per texture tile vertically, so windows line up per floor
        # whatever the building height.
        "floor_height_m": 3.0,
        "tile_width_m": 4.0,
        "texture_px": 512,
        "variants": 1,
        "seed": 20240501,
    },
    "export": {
        "fbx_name": "model.fbx",
        "aerial_name": "aerial.png",
        "metadata_name": "metadata.json",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge `override` into `base`, recursing into nested dicts."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class PipelineConfig:
    """Fully resolved configuration plus the derived geo context."""

    name: str
    bbox: BBox
    geo: GeoContext
    aerial: dict[str, Any]
    terrain: dict[str, Any]
    buildings: dict[str, Any]
    facade: dict[str, Any]
    export: dict[str, Any]
    source_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def size_m(self) -> float:
        """Nominal square side of the area, for metadata."""
        return 0.5 * (self.bbox.width + self.bbox.height)

    def work_dir(self, root: Path) -> Path:
        return root / "work" / self.name

    def out_dir(self, root: Path) -> Path:
        return root / "output" / self.name


def load_config(path: str | Path) -> PipelineConfig:
    """Read a config file, apply defaults, and validate the bbox."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if "bbox" not in raw:
        raise ValueError(f"{path} has no 'bbox' section")

    merged = _deep_merge(DEFAULTS, raw)
    bbox = parse_bbox(raw["bbox"])
    warnings = validate_bbox(bbox)
    for warning in warnings:
        LOG.warning("%s", warning)

    name = str(merged["name"]).strip()
    if not name:
        raise ValueError("config 'name' must not be empty")

    ahn_model = str(merged["terrain"]["ahn_model"]).strip().upper()
    if ahn_model not in ("DTM", "DSM"):
        raise ValueError(
            f"terrain.ahn_model must be DTM or DSM, got {ahn_model!r}"
        )
    if ahn_model == "DSM":
        LOG.warning(
            "terrain.ahn_model is DSM: the surface model still contains "
            "buildings and vegetation, so 3DBAG buildings will sit on top of "
            "their own roofs. DTM is what this pipeline expects."
        )
    merged["terrain"]["ahn_model"] = ahn_model

    lod = str(merged["buildings"]["lod"]).strip()
    if lod != "2.2":
        LOG.warning(
            "buildings.lod is %r; only 2.2 carries real roof shapes, lower "
            "LoDs are flat extrusions",
            lod,
        )
    merged["buildings"]["lod"] = lod

    n = int(merged["terrain"]["mesh_vertices_per_side"])
    if n < 2:
        raise ValueError(
            f"terrain.mesh_vertices_per_side must be at least 2, got {n}"
        )
    merged["terrain"]["mesh_vertices_per_side"] = n

    size_px = int(merged["aerial"]["size_px"])
    if size_px < 16:
        raise ValueError(f"aerial.size_px must be at least 16, got {size_px}")
    merged["aerial"]["size_px"] = size_px

    variants = int(merged["facade"]["variants"])
    if variants < 1:
        raise ValueError(f"facade.variants must be at least 1, got {variants}")
    merged["facade"]["variants"] = variants

    merge_mode = str(merged["buildings"]["merge"]).strip().lower()
    if merge_mode not in ("single", "per_building"):
        raise ValueError(
            f"buildings.merge must be 'single' or 'per_building', got {merge_mode!r}"
        )
    merged["buildings"]["merge"] = merge_mode

    clip_mode = str(merged["buildings"]["clip_mode"]).strip().lower()
    if clip_mode not in ("centroid", "intersect"):
        raise ValueError(
            f"buildings.clip_mode must be 'centroid' or 'intersect', got {clip_mode!r}"
        )
    merged["buildings"]["clip_mode"] = clip_mode

    return PipelineConfig(
        name=name,
        bbox=bbox,
        geo=GeoContext.from_bbox(bbox),
        aerial=merged["aerial"],
        terrain=merged["terrain"],
        buildings=merged["buildings"],
        facade=merged["facade"],
        export=merged["export"],
        source_path=path,
        warnings=warnings,
    )


__all__ = ["DEFAULTS", "PipelineConfig", "load_config"]
