"""Trees: BGT tree points, given real heights by AHN.

Two national sources combine into something neither has on its own. The BGT
registers individual trees as points with authoritative positions but no height.
AHN has height everywhere but does not say what is a tree. Subtracting the DTM
from the DSM leaves a canopy height model, and sampling that at each BGT point
gives every tree its own measured height.

The BGT API returns the full version history of each object, so a naive read
finds 4834 "trees" in a square kilometre of Utrecht where 1488 stand: the rest
are superseded versions stacked on the same spots. Only the current version of
each object is kept.
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import requests

from .elevation import NODATA_CUTOFF, fetch_dtm_geotiff
from .geo import BBox
from .http_util import ServiceError, get_with_retry

LOG = logging.getLogger(__name__)

RD_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"


@dataclass
class Tree:
    """One tree, positioned in RD with a height measured from AHN."""

    x: float
    y: float
    ground_z_nap: float
    height_m: float
    crown_radius_m: float
    trunk_height_m: float
    measured: bool  # False when AHN had nothing usable and a default was used

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "ground_z_nap": round(self.ground_z_nap, 3),
            "height_m": round(self.height_m, 2),
            "crown_radius_m": round(self.crown_radius_m, 2),
            "trunk_height_m": round(self.trunk_height_m, 2),
            "height_measured": self.measured,
        }


@dataclass
class TreeSet:
    trees: list[Tree] = field(default_factory=list)
    pages_fetched: int = 0
    raw_features: int = 0
    superseded_dropped: int = 0
    outside_bbox_dropped: int = 0

    def __len__(self) -> int:
        return len(self.trees)

    def stats(self) -> dict:
        if not self.trees:
            return {"count": 0}
        heights = np.array([t.height_m for t in self.trees])
        measured = sum(1 for t in self.trees if t.measured)
        return {
            "count": len(self.trees),
            "height_min_m": round(float(heights.min()), 2),
            "height_max_m": round(float(heights.max()), 2),
            "height_median_m": round(float(np.median(heights)), 2),
            "heights_measured": measured,
            "heights_defaulted": len(self.trees) - measured,
            "raw_features": self.raw_features,
            "superseded_dropped": self.superseded_dropped,
            "pages_fetched": self.pages_fetched,
        }


def iter_bgt_pages(
    bbox: BBox,
    *,
    api_url: str,
    page_limit: int = 1000,
    timeout: float = 120.0,
    max_retries: int = 4,
    max_pages: int = 100,
) -> Iterator[dict]:
    """Page through a BGT collection over `bbox`, in RD.

    ``bbox-crs`` and ``crs`` both have to be set: the OGC API default is
    lon/lat, and an RD bbox read as degrees selects nothing at all rather than
    failing.
    """
    session = requests.Session()
    url: str | None = api_url
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
            description=f"BGT page {page_index}",
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ServiceError(
                f"BGT page {page_index} was not JSON: {response.text[:300]}"
            ) from exc

        yield payload

        next_links = [
            link.get("href")
            for link in payload.get("links", [])
            if link.get("rel") == "next" and link.get("href")
        ]
        if not next_links:
            return
        next_url = next_links[0]
        if next_url in seen:
            LOG.warning("BGT paging repeated a URL; stopping")
            return
        seen.add(next_url)
        url, params = next_url, None

    LOG.warning("stopped after %d BGT pages", max_pages)


def _load_ndsm(
    bbox: BBox, work_dir: Path, dtm_geotiff: Path, trees_cfg: dict
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Canopy height model: DSM minus DTM, on the DTM's own grid.

    Both come from the same AHN service, so requesting the DSM over the DTM
    raster's exact bounds lines the two grids up cell for cell.
    """
    import rasterio

    with rasterio.open(dtm_geotiff) as dataset:
        dtm = dataset.read(1).astype(np.float64)
        bounds = (
            dataset.bounds.left,
            dataset.bounds.bottom,
            dataset.bounds.right,
            dataset.bounds.top,
        )
        # Taken from the raster rather than from config, so the DSM request is
        # split into exactly the same tiles the DTM was and the two grids line
        # up cell for cell.
        resolution = float(dataset.res[0])

    dsm_path = work_dir / "ahn_dsm.tif"
    fetch_dtm_geotiff(
        BBox(*bounds),
        dsm_path,
        wcs_url=str(trees_cfg["wcs_url"]),
        ahn_model="DSM",
        resolution_m=resolution,
        timeout=float(trees_cfg["timeout_s"]),
        max_retries=int(trees_cfg["max_retries"]),
        verify_capabilities=False,
    )

    with rasterio.open(dsm_path) as dataset:
        dsm = dataset.read(1).astype(np.float64)

    if dsm.shape != dtm.shape:
        raise ServiceError(
            f"DSM raster {dsm.shape} does not match the DTM raster {dtm.shape}; "
            f"the two AHN grids must line up to subtract them"
        )

    valid = (dsm < NODATA_CUTOFF) & (dtm < NODATA_CUTOFF)
    ndsm = np.where(valid, dsm - dtm, np.nan)
    LOG.info(
        "canopy height model over %d x %d cells, %.1f%% usable",
        ndsm.shape[1],
        ndsm.shape[0],
        100.0 * valid.mean(),
    )
    return ndsm, bounds


def _rasterize_buildings(
    buildings, bounds: tuple[float, float, float, float], shape: tuple[int, int]
) -> np.ndarray:
    """Mark every raster cell covered by a building roof.

    Without this, a tree standing a few metres from a wall takes the local
    maximum off the roof next to it and comes out as tall as the building. The
    canopy model is only meaningful away from buildings, so they are cut out of
    it before any tree is sampled.
    """
    left, bottom, right, top = bounds
    rows, cols = shape
    cell_x = (right - left) / cols
    cell_y = (top - bottom) / rows
    mask = np.zeros(shape, dtype=bool)

    for building in buildings.buildings:
        for triangle in building.roof_tris:
            # Raster coordinates of the triangle, y flipped for row order.
            px = (triangle[:, 0] - left) / cell_x
            py = (top - triangle[:, 1]) / cell_y

            col0 = max(0, int(np.floor(px.min())))
            col1 = min(cols - 1, int(np.ceil(px.max())))
            row0 = max(0, int(np.floor(py.min())))
            row1 = min(rows - 1, int(np.ceil(py.max())))
            if col1 < col0 or row1 < row0:
                continue

            cc, rr = np.meshgrid(
                np.arange(col0, col1 + 1) + 0.5, np.arange(row0, row1 + 1) + 0.5
            )
            # Barycentric test against the projected triangle.
            x1, y1 = px[0], py[0]
            x2, y2 = px[1], py[1]
            x3, y3 = px[2], py[2]
            denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if abs(denom) < 1e-12:
                continue
            a = ((y2 - y3) * (cc - x3) + (x3 - x2) * (rr - y3)) / denom
            b = ((y3 - y1) * (cc - x3) + (x1 - x3) * (rr - y3)) / denom
            c = 1.0 - a - b
            inside = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
            mask[row0 : row1 + 1, col0 : col1 + 1] |= inside

    LOG.info("masked %.1f%% of the canopy model as buildings", 100.0 * mask.mean())
    return mask


def _max_in_radius(
    grid: np.ndarray,
    bounds: tuple[float, float, float, float],
    xs: np.ndarray,
    ys: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """Highest value within `radius_m` of each point.

    A BGT tree point marks the trunk, but the canopy top is metres away from it,
    so a point sample lands on whatever the ground happens to be beside the tree
    and reads far too low. Taking the local maximum finds the crown.
    """
    left, bottom, right, top = bounds
    rows, cols = grid.shape
    cell_x = (right - left) / cols
    cell_y = (top - bottom) / rows

    col = np.clip(((xs - left) / cell_x).astype(np.int64), 0, cols - 1)
    row = np.clip(((top - ys) / cell_y).astype(np.int64), 0, rows - 1)

    radius_cells = max(1, int(round(radius_m / min(cell_x, cell_y))))
    best = np.full(len(xs), np.nan)

    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            sample = grid[
                np.clip(row + dy, 0, rows - 1), np.clip(col + dx, 0, cols - 1)
            ]
            best = np.fmax(best, sample)

    return best


def build_trees(
    bbox: BBox,
    work_dir: Path,
    *,
    trees_cfg: dict,
    terrain,
    buildings=None,
) -> TreeSet:
    """Fetch BGT trees over `bbox` and give each one a height from AHN."""
    result = TreeSet()

    LOG.info("querying BGT trees for %s", bbox)
    points: list[tuple[float, float]] = []
    seen_ids: set[str] = set()

    for payload in iter_bgt_pages(
        bbox,
        api_url=str(trees_cfg["api_url"]),
        page_limit=int(trees_cfg["page_limit"]),
        timeout=float(trees_cfg["timeout_s"]),
        max_retries=int(trees_cfg["max_retries"]),
        max_pages=int(trees_cfg["max_pages"]),
    ):
        result.pages_fetched += 1
        for feature in payload.get("features", []):
            result.raw_features += 1
            properties = feature.get("properties", {}) or {}

            # Every past version of an object is returned alongside the current
            # one. A closed registration means this row has been superseded.
            if properties.get("eind_registratie"):
                result.superseded_dropped += 1
                continue

            identifier = properties.get("lokaal_id")
            if identifier:
                if identifier in seen_ids:
                    result.superseded_dropped += 1
                    continue
                seen_ids.add(identifier)

            geometry = feature.get("geometry") or {}
            if geometry.get("type") != "Point":
                continue
            x, y = geometry["coordinates"][:2]

            if not (bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax):
                result.outside_bbox_dropped += 1
                continue
            points.append((float(x), float(y)))

    LOG.info(
        "BGT returned %d features, %d are current trees inside the bbox",
        result.raw_features,
        len(points),
    )
    if not points:
        LOG.warning("no BGT trees found for %s", bbox)
        return result

    coords = np.asarray(points, dtype=np.float64)

    ndsm, bounds = _load_ndsm(
        bbox, work_dir, terrain.geotiff_path, trees_cfg
    )
    if buildings is not None and len(buildings):
        ndsm = np.where(
            _rasterize_buildings(buildings, bounds, ndsm.shape), np.nan, ndsm
        )
    canopy = _max_in_radius(
        ndsm, bounds, coords[:, 0], coords[:, 1], float(trees_cfg["crown_search_m"])
    )

    min_h = float(trees_cfg["min_height_m"])
    max_h = float(trees_cfg["max_height_m"])
    default_h = float(trees_cfg["default_height_m"])
    crown_ratio = float(trees_cfg["crown_radius_ratio"])
    trunk_ratio = float(trees_cfg["trunk_height_ratio"])

    ground = np.asarray(terrain.sample(coords[:, 0], coords[:, 1]), dtype=np.float64)

    for index, (x, y) in enumerate(coords):
        raw = canopy[index]
        measured = bool(np.isfinite(raw) and min_h <= raw <= max_h)
        height = float(raw) if measured else default_h
        height = float(np.clip(height, min_h, max_h))

        result.trees.append(
            Tree(
                x=float(x),
                y=float(y),
                ground_z_nap=float(ground[index]),
                height_m=height,
                # Urban crowns run about a quarter of the tree's height across,
                # bounded so a young tree still reads as a tree.
                crown_radius_m=float(np.clip(crown_ratio * height, 0.8, 7.0)),
                trunk_height_m=float(np.clip(trunk_ratio * height, 0.6, 6.0)),
                measured=measured,
            )
        )

    LOG.info("%s", _summary_line(result))
    save_trees(result, work_dir / "trees.npz")
    return result


def _summary_line(result: TreeSet) -> str:
    stats = result.stats()
    return (
        f"{stats['count']} trees, heights {stats['height_min_m']}-"
        f"{stats['height_max_m']} m (median {stats['height_median_m']} m), "
        f"{stats['heights_measured']} measured from AHN, "
        f"{stats['heights_defaulted']} defaulted"
    )


def save_trees(trees: TreeSet, path: Path) -> Path:
    """Write the tree set for the Blender stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xy=np.array([[t.x, t.y] for t in trees.trees], dtype=np.float64).reshape(-1, 2),
        ground_z_nap=np.array([t.ground_z_nap for t in trees.trees], dtype=np.float64),
        height_m=np.array([t.height_m for t in trees.trees], dtype=np.float64),
        crown_radius_m=np.array(
            [t.crown_radius_m for t in trees.trees], dtype=np.float64
        ),
        trunk_height_m=np.array(
            [t.trunk_height_m for t in trees.trees], dtype=np.float64
        ),
    )
    LOG.info("wrote %s (%d trees)", path, len(trees))
    return path


def write_tree_list(trees: TreeSet, geo, out_dir: Path, filename: str = "trees.json") -> Path:
    """Write the spawn list Unity can instantiate prefabs from.

    Positions are given in the model's local frame as well as in RD, so a prefab
    can be dropped straight in without the caller redoing the origin shift.
    """
    import json

    records = []
    for tree in trees.trees:
        local_x, local_y = geo.rd_to_local(tree.x, tree.y)
        record = tree.to_dict()
        record["local"] = {
            "x": round(local_x, 3),
            # Unity Z is RD northing; Y comes from the ground height.
            "z": round(local_y, 3),
        }
        records.append(record)

    payload = {
        "count": len(records),
        "crs": "EPSG:28992",
        "note": (
            "local.x / local.z are metres in the model frame; add "
            "ground_z_nap minus metadata.ground_z_offset_nap for local Y"
        ),
        "source": "BGT vegetatieobject_punt, heights from AHN DSM minus DTM",
        "trees": records,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    LOG.info("wrote %s", path)
    return path


__all__ = [
    "Tree",
    "TreeSet",
    "build_trees",
    "iter_bgt_pages",
    "save_trees",
    "write_tree_list",
]
