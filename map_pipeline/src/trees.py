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
    # True when the canopy model found it rather than the BGT registering it.
    # The detector is good, not perfect, so these stay separable all the way
    # to the FBX and a bad patch can be deleted as a group.
    detected: bool = False

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "ground_z_nap": round(self.ground_z_nap, 3),
            "height_m": round(self.height_m, 2),
            "crown_radius_m": round(self.crown_radius_m, 2),
            "trunk_height_m": round(self.trunk_height_m, 2),
            "height_measured": self.measured,
            "detected": self.detected,
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
            "registered": sum(1 for t in self.trees if not t.detected),
            "detected": sum(1 for t in self.trees if t.detected),
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
    bbox: BBox, work_dir: Path, dtm_geotiff: Path, trees_cfg: dict, terrain
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

    # Against the terrain the model already stands on, not against the raw
    # download. This matters more than it sounds: the bare-earth DTM arrives
    # only 51% measured over this square kilometre, and its holes are exactly
    # under dense canopy, because dense canopy is what stops a ground return
    # reaching the lidar. Differencing the two rasters therefore leaves the
    # canopy model undefined in the middle of a wood -- which is the one place
    # it is most wanted. The terrain stage has already filled and smoothed that
    # surface, so using it takes the model from 51% coverage to 96% and costs
    # nothing, and the trees end up standing on the ground everything else is
    # draped on.
    rows, cols = dsm.shape
    left, bottom, right, top = bounds
    xs = left + (np.arange(cols) + 0.5) * (right - left) / cols
    ys = top - (np.arange(rows) + 0.5) * (top - bottom) / rows
    grid_x, grid_y = np.meshgrid(xs, ys)
    ground = np.asarray(
        terrain.sample(grid_x.ravel(), grid_y.ravel()), dtype=np.float64
    ).reshape(dsm.shape)

    valid = dsm < NODATA_CUTOFF
    ndsm = np.where(valid, dsm - ground, np.nan)
    LOG.info(
        "canopy height model over %d x %d cells, %.1f%% usable "
        "(the bare-earth grid alone would give %.1f%%)",
        ndsm.shape[1],
        ndsm.shape[0],
        100.0 * valid.mean(),
        100.0 * (valid & (dtm < NODATA_CUTOFF)).mean(),
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


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    """Grow a boolean mask by `cells`. No scipy in this project, so: shifts."""
    out = mask.copy()
    for _ in range(max(0, cells)):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _roughness(grid: np.ndarray, radius: int) -> np.ndarray:
    """How far the surface departs from its own local mean, per cell.

    The one measurement that tells a crown from roof clutter once the building
    footprints are already masked. Over a square kilometre of Utrecht the
    roughness at a registered tree runs to a median of 3.1 m against 0.67 m on
    a roof, so scaffolding, roof plant and the odd parked lorry come out on the
    right side of it.

    Summed-area tables, so the window size costs nothing.
    """
    valid = np.isfinite(grid)
    filled = np.where(valid, grid, 0.0)
    rows, cols = grid.shape

    def integral(values):
        out = np.zeros((rows + 1, cols + 1))
        out[1:, 1:] = values.cumsum(0).cumsum(1)
        return out

    sum_x = integral(filled)
    sum_xx = integral(filled * filled)
    sum_n = integral(valid.astype(np.float64))

    r0 = np.clip(np.arange(rows) - radius, 0, rows)
    r1 = np.clip(np.arange(rows) + radius + 1, 0, rows)
    c0 = np.clip(np.arange(cols) - radius, 0, cols)
    c1 = np.clip(np.arange(cols) + radius + 1, 0, cols)

    def window(table):
        return (
            table[np.ix_(r1, c1)] - table[np.ix_(r0, c1)]
            - table[np.ix_(r1, c0)] + table[np.ix_(r0, c0)]
        )

    count = np.maximum(window(sum_n), 1.0)
    mean = window(sum_x) / count
    variance = window(sum_xx) / count - mean * mean
    return np.where(window(sum_n) > 0, np.sqrt(np.maximum(variance, 0.0)), np.nan)


def _local_peaks(grid: np.ndarray, radius: int) -> np.ndarray:
    """Cells no lower than anything within `radius`, as a boolean mask.

    One seed per crown rather than one per cell. Without it the greedy pass
    below would have a third of a million candidates to sort through instead of
    a few thousand.
    """
    best = np.where(np.isfinite(grid), grid, -np.inf)
    peak = best.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.full_like(best, -np.inf)
            ys = slice(max(0, dy), best.shape[0] + min(0, dy))
            xs = slice(max(0, dx), best.shape[1] + min(0, dx))
            yt = slice(max(0, -dy), best.shape[0] + min(0, -dy))
            xt = slice(max(0, -dx), best.shape[1] + min(0, -dx))
            shifted[yt, xt] = best[ys, xs]
            peak = np.maximum(peak, shifted)
    return np.isfinite(grid) & (best >= peak)


def detect_trees(
    ndsm: np.ndarray,
    bounds: tuple[float, float, float, float],
    *,
    registered_xy: np.ndarray,
    registered_reach: np.ndarray | None = None,
    bbox: BBox | None = None,
    building_mask: np.ndarray | None = None,
    water_mask: np.ndarray | None = None,
    road_mask: np.ndarray | None = None,
    min_height_m: float = 2.5,
    max_height_m: float = 30.0,
    crown_radius_ratio: float = 0.25,
    min_roughness_m: float = 0.4,
    eaves_clearance_m: float = 2.0,
    clearance_per_metre: float = 0.15,
    min_spacing_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find tree crowns the BGT never registered. Returns `(xy, heights)`.

    The BGT registers street trees, which is a municipal asset register, not a
    survey of vegetation. Over one square kilometre of Utrecht centre 45% of the
    canopy above 2.5 m is nowhere near a registered tree, and that is the good
    case -- the register is at its best in a city centre and knows nothing about
    private gardens.

    Colour would be the obvious second opinion and it is not available: Dutch
    national orthos are flown deliberately leaf-off in early spring so the
    ground and the buildings are visible, and measured over the same square the
    infrared ortho gives park trees an NDVI of +0.08 against -0.18 for a roof.
    A quarter of an index point is not a detector. Height is the signal here.
    """
    left, bottom, right, top = bounds
    rows, cols = ndsm.shape
    cell_x = (right - left) / cols
    cell_y = (top - bottom) / rows
    cell = min(cell_x, cell_y)

    candidate = np.isfinite(ndsm) & (ndsm > min_height_m) & (ndsm <= max_height_m)
    if building_mask is not None:
        # Dilated, because a roof overhangs its own footprint and the eaves
        # otherwise read as a line of trees along every terrace.
        #
        # And dilated further for taller candidates, which is not a hunch: over
        # the Utrecht square, 65% of detections above 25 m sat within 5 m of a
        # building against 32% of those under 10 m, and the tallest were one
        # tight cluster around a tower the BGT footprint does not cover. The
        # taller a thing is, the likelier it is part of the building it is
        # standing against, so it has to stand further off to be believed.
        near = _dilate(building_mask, int(round(eaves_clearance_m / cell)))
        candidate &= ~near
        if clearance_per_metre > 0:
            reach = eaves_clearance_m
            step = max(cell, 1.0)
            while reach < eaves_clearance_m + clearance_per_metre * max_height_m:
                reach += step
                near = _dilate(near, int(round(step / cell)))
                # Everything this close to a roof must be shorter than the
                # clearance it has earned.
                too_tall = (reach - eaves_clearance_m) / clearance_per_metre
                candidate &= ~(near & (ndsm > too_tall))
    if water_mask is not None:
        # Lidar does not reflect off water, and the returns that do come back
        # scatter over six metres. Noise that high would be a forest.
        candidate &= ~water_mask
    if road_mask is not None:
        # A tall rough thing standing on a carriageway is a lorry, a bus or
        # scaffolding far more often than a tree: 9.5% of everything found over
        # a square kilometre of Gelderland was standing on one.
        candidate &= ~road_mask
    if not candidate.any():
        return np.zeros((0, 2)), np.zeros(0)

    if min_roughness_m > 0:
        candidate &= _roughness(ndsm, max(1, int(round(2.5 / cell)))) > min_roughness_m
    if not candidate.any():
        return np.zeros((0, 2)), np.zeros(0)

    # One seed per crown. The window is sized for the smallest tree worth
    # finding, so two trees closer than that merge -- which is the right answer
    # for a hedge row read as one crown.
    seed_radius = max(1, int(round(crown_radius_ratio * min_height_m / cell)))
    peaks = _local_peaks(np.where(candidate, ndsm, np.nan), seed_radius)
    seed_rows, seed_cols = np.nonzero(peaks)
    if not len(seed_rows):
        return np.zeros((0, 2)), np.zeros(0)

    heights = ndsm[seed_rows, seed_cols]
    xs = left + (seed_cols + 0.5) * cell_x
    ys = top - (seed_rows + 0.5) * cell_y

    # Tallest first, so where two seeds compete the real crown wins and the
    # shoulder beside it is the one dropped.
    order = np.argsort(-heights)
    xs, ys, heights = xs[order], ys[order], heights[order]

    spacing = np.maximum(crown_radius_ratio * heights, min_spacing_m)
    grid_size = max(
        float(spacing.max()),
        float(np.max(registered_reach)) if registered_reach is not None
        and len(np.atleast_1d(registered_reach)) else 0.0,
        min_spacing_m,
        cell,
    )
    taken: dict = {}

    def claim(x, y, reach, store):
        key = (int(x // grid_size), int(y // grid_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for ox, oy, oreach in taken.get((key[0] + dx, key[1] + dy), ()):
                    if np.hypot(x - ox, y - oy) < max(reach, oreach):
                        return False
        if store:
            taken.setdefault(key, []).append((x, y, reach))
        return True

    # Registered trees are placed first and never displaced: where the BGT has
    # surveyed a trunk, that position is better than anything a raster peak can
    # offer, and a detected twin beside it would be one tree drawn twice.
    #
    # Each reserves its own crown, not a flat radius. A fifteen-metre tree is
    # four metres across, and its canopy peak sits nowhere near the surveyed
    # trunk -- which is why a flat three-metre reservation left a quarter of
    # the detections sitting inside a tree that was already there.
    fixed = np.asarray(registered_xy, dtype=np.float64).reshape(-1, 2)
    if registered_reach is None:
        reserved = np.full(len(fixed), min_spacing_m)
    else:
        reserved = np.maximum(
            np.asarray(registered_reach, dtype=np.float64).ravel(), min_spacing_m
        )
    for (x, y), reach in zip(fixed, reserved):
        claim(float(x), float(y), float(reach), store=True)

    keep_x, keep_y, keep_h = [], [], []
    for x, y, height, reach in zip(xs, ys, heights, spacing):
        if not claim(float(x), float(y), float(reach), store=True):
            continue
        keep_x.append(float(x))
        keep_y.append(float(y))
        keep_h.append(float(height))

    xy = np.column_stack([keep_x, keep_y]) if keep_x else np.zeros((0, 2))
    found_h = np.asarray(keep_h)

    # Cut to the area. The canopy model covers whatever raster the AHN service
    # returned, which need not stop where the bbox does, and a tree outside the
    # terrain is a tree standing on nothing -- it also drags the model's own
    # bounding box out with it, which is what the span check measures.
    if bbox is not None and len(xy):
        inside = (
            (xy[:, 0] >= bbox.xmin) & (xy[:, 0] <= bbox.xmax)
            & (xy[:, 1] >= bbox.ymin) & (xy[:, 1] <= bbox.ymax)
        )
        if not inside.all():
            LOG.info("dropped %d detected trees outside the bbox",
                     int((~inside).sum()))
        xy, found_h = xy[inside], found_h[inside]

    LOG.info(
        "detected %d trees the register does not have, from %d canopy peaks",
        len(xy),
        len(seed_rows),
    )
    return xy, found_h


def build_trees(
    bbox: BBox,
    work_dir: Path,
    *,
    trees_cfg: dict,
    terrain,
    buildings=None,
    surfaces=None,
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
    coords = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(coords):
        LOG.warning("no BGT trees found for %s", bbox)

    want_detect = bool(trees_cfg.get("detect", True))
    if not len(coords) and not want_detect:
        return result

    min_h = float(trees_cfg["min_height_m"])
    max_h = float(trees_cfg["max_height_m"])
    default_h = float(trees_cfg["default_height_m"])
    crown_ratio = float(trees_cfg["crown_radius_ratio"])
    trunk_ratio = float(trees_cfg["trunk_height_ratio"])

    ndsm, bounds = _load_ndsm(
        bbox, work_dir, terrain.geotiff_path, trees_cfg, terrain
    )
    # Kept rather than blanked, because the detector needs to know where the
    # roofs are in order to stay off them and their eaves, and blanking loses
    # that. Only the registered-tree sampling wants them out of the way.
    building_mask = (
        _rasterize_buildings(buildings, bounds, ndsm.shape)
        if buildings is not None and len(buildings)
        else None
    )
    off_buildings = ndsm if building_mask is None else np.where(building_mask, np.nan, ndsm)

    def add(x, y, height, ground_z, measured, detected):
        height = float(np.clip(height, min_h, max_h))
        result.trees.append(
            Tree(
                x=float(x),
                y=float(y),
                ground_z_nap=float(ground_z),
                height_m=height,
                # Urban crowns run about a quarter of the tree's height across,
                # bounded so a young tree still reads as a tree.
                crown_radius_m=float(np.clip(crown_ratio * height, 0.8, 7.0)),
                trunk_height_m=float(np.clip(trunk_ratio * height, 0.6, 6.0)),
                measured=measured,
                detected=detected,
            )
        )

    if len(coords):
        canopy = _max_in_radius(
            off_buildings, bounds, coords[:, 0], coords[:, 1],
            float(trees_cfg["crown_search_m"]),
        )
        ground = np.asarray(
            terrain.sample(coords[:, 0], coords[:, 1]), dtype=np.float64
        )
        for index, (x, y) in enumerate(coords):
            raw = canopy[index]
            measured = bool(np.isfinite(raw) and min_h <= raw <= max_h)
            add(x, y, raw if measured else default_h, ground[index], measured, False)

    if want_detect:
        water_mask = _rasterize_water(surfaces, bounds, ndsm.shape)
        road_mask = _rasterize_roads(
            surfaces, bounds, ndsm.shape,
            _road_classes(trees_cfg.get("detect_off_surfaces")),
        )
        found, heights = detect_trees(
            ndsm,
            bounds,
            registered_xy=coords,
            bbox=bbox,
            registered_reach=np.array(
                [t.crown_radius_m for t in result.trees], dtype=np.float64
            ),
            building_mask=building_mask,
            water_mask=water_mask,
            road_mask=road_mask,
            min_height_m=float(trees_cfg.get("detect_min_height_m", 2.5)),
            max_height_m=float(trees_cfg.get("detect_max_height_m", 30.0)),
            crown_radius_ratio=crown_ratio,
            min_roughness_m=float(trees_cfg.get("detect_min_roughness_m", 0.4)),
            eaves_clearance_m=float(trees_cfg.get("detect_eaves_clearance_m", 2.0)),
            clearance_per_metre=float(
                trees_cfg.get("detect_clearance_per_metre", 0.15)
            ),
            min_spacing_m=float(trees_cfg.get("detect_min_spacing_m", 3.0)),
        )
        if len(found):
            ground = np.asarray(
                terrain.sample(found[:, 0], found[:, 1]), dtype=np.float64
            )
            for index, (x, y) in enumerate(found):
                add(x, y, heights[index], ground[index], True, True)

    if not result.trees:
        return result

    LOG.info("%s", _summary_line(result))
    save_trees(result, work_dir / "trees.npz")
    return result


def _road_classes(names) -> set[int]:
    """Surface-class codes from the names used in the config and the FBX.

    Named rather than numbered, because "cycle_path" survives someone reading
    it a year later and 7 does not, and because these are the same names the
    road objects already carry in the model.
    """
    from .surfaces import CLASS_NAMES

    if names is None:
        return set()
    by_name = {str(label): int(code) for code, label in CLASS_NAMES.items()}
    wanted: set[int] = set()
    for name in names:
        code = by_name.get(str(name))
        if code is None:
            LOG.warning(
                "trees.detect_off_surfaces names %r, which is not a surface "
                "class; known names are %s",
                name, ", ".join(sorted(by_name)),
            )
            continue
        wanted.add(code)
    return wanted


def _rasterize_rings(rings, bounds, shape) -> np.ndarray:
    """Even-odd scanline fill of polygon rings onto a raster."""
    left, bottom, right, top = bounds
    rows, cols = shape
    cell_x = (right - left) / cols
    cell_y = (top - bottom) / rows
    mask = np.zeros(shape, dtype=bool)

    for ring in rings:
        ring = np.asarray(ring, dtype=np.float64)
        if ring.ndim != 2 or len(ring) < 3:
            continue
        ring = ring[:, :2]
        px = (ring[:, 0] - left) / cell_x
        py = (top - ring[:, 1]) / cell_y
        ax, ay, bx, by = px[:-1], py[:-1], px[1:], py[1:]
        for row in range(
            max(0, int(np.floor(py.min()))), min(rows, int(np.ceil(py.max())) + 1)
        ):
            centre = row + 0.5
            straddles = (ay > centre) != (by > centre)
            if not straddles.any():
                continue
            t = (centre - ay[straddles]) / (by[straddles] - ay[straddles])
            crossings = np.sort(ax[straddles] + t * (bx[straddles] - ax[straddles]))
            for start, end in zip(crossings[0::2], crossings[1::2]):
                col0 = max(0, int(np.ceil(start - 0.5)))
                col1 = min(cols - 1, int(np.floor(end - 0.5)))
                if col1 >= col0:
                    mask[row, col0 : col1 + 1] = True
    return mask


def _rasterize_water(surfaces, bounds, shape) -> np.ndarray | None:
    """Mark the cells covered by water, so lidar noise is not read as forest.

    Lidar does not reflect off water: the returns that do come back scatter
    over six metres, and six metres of scatter above the ground is a tree as
    far as any height threshold can tell.
    """
    bodies = getattr(surfaces, "water", None) if surfaces is not None else None
    if not bodies:
        return None

    mask = _rasterize_rings(
        (ring for body in bodies for ring in getattr(body, "rings", ()) or ()),
        bounds, shape,
    )
    LOG.info("masked %.1f%% of the canopy model as water", 100.0 * mask.mean())
    return mask


def _rasterize_roads(surfaces, bounds, shape, classes) -> np.ndarray | None:
    """Mark the cells covered by the road classes a tree has no business on.

    A tall rough thing standing on a carriageway is a lorry, a bus, scaffolding
    or a crane far more often than it is a tree. Over a square kilometre of
    Gelderland 9.5% of everything the canopy model found was standing on a
    drivable surface, which is a lot of parked traffic to hand a level designer
    as woodland.

    Avenues do exist, and this does cost the odd real one -- a street tree
    whose crown leans far enough over the carriageway for its peak to land
    there. That trade is worth it at ten to one, and it is only applied to what
    the detector found: where the BGT has surveyed a trunk, the trunk is there,
    whatever surface the survey says it stands on.

    Footpaths are in the default set too, on second thoughts. They were left
    out to protect the tree in a pit on a pedestrianised street -- but a
    surveyed tree is never filtered by this at all, so the exclusion protected
    nothing and cost a great deal: footpath is 305 of the 871 road parts over
    the Gelderland square, and in a park every winding path through the lawns
    is one.
    """
    parts = getattr(surfaces, "roads", None) if surfaces is not None else None
    if not parts or not classes:
        return None

    wanted = set(classes)
    rings = [
        ring
        for part in parts
        if int(getattr(part, "surface_class", -1)) in wanted
        and int(getattr(part, "level", 0)) <= 0
        for ring in getattr(part, "rings", ()) or ()
    ]
    if not rings:
        return None

    mask = _rasterize_rings(rings, bounds, shape)
    LOG.info(
        "masked %.1f%% of the canopy model as road a tree would not stand on",
        100.0 * mask.mean(),
    )
    return mask


def _summary_line(result: TreeSet) -> str:
    stats = result.stats()
    return (
        f"{stats['count']} trees, heights {stats['height_min_m']}-"
        f"{stats['height_max_m']} m (median {stats['height_median_m']} m), "
        f"{stats['heights_measured']} measured from AHN, "
        f"{stats['heights_defaulted']} defaulted; "
        f"{stats['registered']} from the BGT register, "
        f"{stats['detected']} found in the canopy model"
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
        # So the Blender stage can split them into two objects. The detector is
        # good, not perfect, and a bad patch has to be deletable as a group.
        detected=np.array([t.detected for t in trees.trees], dtype=bool),
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
    "detect_trees",
    "Tree",
    "TreeSet",
    "build_trees",
    "iter_bgt_pages",
    "save_trees",
    "write_tree_list",
]
