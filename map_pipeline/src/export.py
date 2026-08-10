"""Writes metadata.json and the scene description the Blender stage reads.

``metadata.json`` is the contract that lets Unity walk back from a local metre
position to NAP and RD, and from there to WGS84 for GPS tracks or POIs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .geo import GeoContext, rd_bbox_to_wgs84

LOG = logging.getLogger(__name__)

ATTRIBUTION = {
    "buildings": (
        "3DBAG by the 3D geoinformation research group, TU Delft, and Kadaster "
        "(CC BY 4.0). https://3dbag.nl"
    ),
    "terrain": (
        "AHN (Actueel Hoogtebestand Nederland) via PDOK (CC BY 4.0). "
        "https://www.ahn.nl"
    ),
    "aerial": (
        "Luchtfoto via PDOK / Beeldmateriaal Nederland (CC BY 4.0). "
        "https://www.beeldmateriaal.nl"
    ),
}


def build_metadata(
    *,
    name: str,
    geo: GeoContext,
    terrain,
    aerial,
    buildings,
    facade_cfg: dict,
    lod: str,
    ahn_model: str,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Assemble the metadata document."""
    bbox = geo.bbox
    wgs = rd_bbox_to_wgs84(bbox)
    center_lon, center_lat = geo.to_wgs84(*bbox.center)

    metadata: dict[str, Any] = {
        "name": name,
        "crs": "EPSG:28992",
        "bbox_rd": [round(v, 3) for v in bbox.as_list()],
        "origin_rd": [round(geo.origin_x, 3), round(geo.origin_y, 3)],
        "ground_z_offset_nap": round(float(terrain.center_z_nap), 4),
        "size_m": round(0.5 * (bbox.width + bbox.height), 3),
        "ahn_model": ahn_model,
        "aerial_layer": aerial.layer_title,
        "lod": lod,
        "unity_axis": "X=east, Y=up(NAP), Z=north",
        # Everything below is additive: the fields above are the documented
        # contract, these make the output self-describing.
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bbox_wgs84": [round(v, 8) for v in wgs.as_list()],
        "origin_wgs84": [round(center_lon, 8), round(center_lat, 8)],
        "terrain": {
            "coverage": terrain.coverage_id,
            "source_resolution_m": terrain.resolution_m,
            "mesh_vertices_per_side": terrain.n,
            "grid_spacing_m": round(bbox.width / (terrain.n - 1), 4),
            **terrain.stats(),
        },
        "aerial": aerial.to_dict(),
        "buildings": {
            **buildings.stats(),
            "lod": lod,
            "ground_surfaces": "dropped (hidden under the terrain)",
        },
        "materials": {
            "M_aerial": (
                "aerial ortho, top-down planar UV; used on terrain and roofs"
            ),
            "M_facade": (
                "generated facade, UV in metres: "
                f"{facade_cfg['tile_width_m']} m per tile across, "
                f"one storey up (nominal {facade_cfg['floor_height_m']} m)"
            ),
        },
        "unity_import": {
            "local_to_rd": "rd_x = local_x + origin_rd[0], rd_y = local_z + origin_rd[1]",
            "local_to_nap": "nap = local_y + ground_z_offset_nap",
            "note": (
                "Unity Z is RD northing and Unity Y is NAP height. Import the "
                "FBX with Scale Factor 1 and Convert Units on."
            ),
        },
        "attribution": ATTRIBUTION,
    }

    if extra:
        metadata.update(extra)
    return metadata


def write_metadata(metadata: dict, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOG.info("wrote %s", path)
    return path


def write_scene_description(
    *,
    name: str,
    geo: GeoContext,
    terrain,
    aerial,
    facade_paths: list[tuple[Path, Path | None]],
    facade_cfg: dict,
    buildings_cfg: dict,
    export_cfg: dict,
    work_dir: Path,
) -> Path:
    """Write scene.json: everything the Blender stage needs, and nothing else.

    Keeping the heavy parsing on this side is what lets ``process.py`` stay a
    thin reader instead of depending on a CityJSON add-on inside Blender.
    """
    local = geo.local_bbox()
    scene = {
        "name": name,
        "origin_rd": [geo.origin_x, geo.origin_y],
        "ground_z_offset_nap": float(terrain.center_z_nap),
        "bbox_rd": geo.bbox.as_list(),
        "bbox_local": local.as_list(),
        "terrain": {"file": "terrain_grid.npz"},
        "buildings": {
            "file": "buildings.npz",
            "merge": buildings_cfg["merge"],
        },
        "aerial": {
            "file": aerial.path.name,
            "bbox_rd": aerial.bbox.as_list(),
            # The aerial covers exactly the bbox, so its local footprint is the
            # local bbox. Kept explicit so the UV maths has a single source.
            "bbox_local": local.as_list(),
            "size_px": aerial.size_px,
        },
        "facade": {
            "files": [colour.name for colour, _ in facade_paths],
            "normal_files": [
                normal.name if normal else None for _, normal in facade_paths
            ],
            "tile_width_m": float(facade_cfg["tile_width_m"]),
            "floor_height_m": float(facade_cfg["floor_height_m"]),
            "variants": len(facade_paths),
        },
        "export": {
            "fbx_name": export_cfg["fbx_name"],
            "aerial_name": export_cfg["aerial_name"],
        },
    }

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "scene.json"
    path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    LOG.info("wrote %s", path)
    return path


def write_attribution(out_dir: Path) -> Path:
    """Drop the source credits next to the model.

    3DBAG asks for a copyright notice on reuse, and the AHN and aerial layers
    are CC BY 4.0, so the credits travel with the output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ATTRIBUTION.txt"
    lines = [
        "This model was built from open Dutch government data.",
        "",
        f"Buildings: {ATTRIBUTION['buildings']}",
        f"Terrain:   {ATTRIBUTION['terrain']}",
        f"Aerial:    {ATTRIBUTION['aerial']}",
        "",
        "Keep this notice with the model when you redistribute it.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = [
    "ATTRIBUTION",
    "build_metadata",
    "write_attribution",
    "write_metadata",
    "write_scene_description",
]
