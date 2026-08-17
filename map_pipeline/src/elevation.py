"""AHN terrain: WCS GetCoverage -> GeoTIFF -> a clean height grid.

Two things about AHN drive the design here.

The DTM is the bare ground with buildings and vegetation removed, which is
exactly what we want underneath the 3DBAG buildings. It also means a dense city
centre arrives mostly empty: every building footprint is a hole. Sixty per cent
nodata over Utrecht centre is normal, so filling holes is a routine step rather
than an error path.

Nodata is float32 max (~3.4e38), not zero and not NaN, so it has to be masked
explicitly or it poisons every average it touches.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geo import BBox, tile_edges, build_grid_coords
from .http_util import ServiceError, get_with_retry, short_error
from .terrain_mesh import (
    TerrainMesh,
    build_rtin,
    save_terrain_mesh,
)

LOG = logging.getLogger(__name__)

# AHN marks empty cells with the largest float32. Anything above this cutoff is
# nodata rather than a real height.
NODATA_CUTOFF = 1e30

# Sanity bounds for NAP heights in the Netherlands. The lowest point sits near
# -7 m NAP and the highest ground near +325 m.
MIN_PLAUSIBLE_NAP = -25.0
MAX_PLAUSIBLE_NAP = 400.0

COVERAGE_IDS = {"DTM": "dtm_05m", "DSM": "dsm_05m"}

# MapServer refuses a coverage larger than this per side, and the limit is not
# in the capabilities document, so it can only be learned by being refused:
#   "Raster size out of range, width and height of resulting coverage must be
#    no more than MAXSIZE=4000."
# At the 0.5 m AHN that caps a single request at 2000 m; measured as inclusive,
# 4000 px answers and 4001 does not. Anything larger is fetched in tiles.
MAX_COVERAGE_PX = 4000


@dataclass
class TerrainResult:
    """The height grid plus everything needed to place and check it."""

    heights: np.ndarray  # (n, n) float32, NAP metres, row 0 = south edge
    xs: np.ndarray  # (n,) RD easting per column
    ys: np.ndarray  # (n,) RD northing per row, ascending north
    bbox: BBox
    center_z_nap: float
    nodata_fraction_raw: float
    filled_fraction: float
    geotiff_path: Path | None
    coverage_id: str
    resolution_m: float
    mesh: "TerrainMesh | None" = None
    constrained: "ConstrainedMesh | None" = None

    @property
    def n(self) -> int:
        return int(self.heights.shape[0])

    def sample(self, x, y):
        """Bilinear height lookup at RD coordinates, clamped to the grid."""
        return _bilinear_on_grid(self.heights, self.xs, self.ys, x, y)

    def stats(self) -> dict:
        h = self.heights
        out = {
            "min_nap": float(h.min()),
            "max_nap": float(h.max()),
            "mean_nap": float(h.mean()),
            "center_nap": float(self.center_z_nap),
            "vertices_per_side": self.n,
            "nodata_fraction_raw": float(self.nodata_fraction_raw),
            "filled_fraction": float(self.filled_fraction),
        }
        if self.mesh is not None:
            out["mesh"] = self.mesh.stats()
        if self.constrained is not None:
            out["mesh"] = self.constrained.stats()
        return out


def _coverage_id(ahn_model: str) -> str:
    try:
        return COVERAGE_IDS[ahn_model.upper()]
    except KeyError:
        raise ValueError(
            f"unknown AHN model {ahn_model!r}; expected one of {sorted(COVERAGE_IDS)}"
        ) from None


def discover_coverages(wcs_url: str, timeout: float = 120.0) -> list[str]:
    """List coverage ids the service advertises.

    The coverage names are checked against this rather than trusted blindly,
    because PDOK has renamed them before.
    """
    response = get_with_retry(
        wcs_url,
        params={"SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCapabilities"},
        timeout=timeout,
        description="AHN WCS GetCapabilities",
    )
    import re

    return re.findall(r"<(?:\w+:)?CoverageId>([^<]+)</(?:\w+:)?CoverageId>", response.text)


def fetch_dtm_geotiff(
    bbox: BBox,
    out_path: Path,
    *,
    wcs_url: str,
    ahn_model: str = "DTM",
    resolution_m: float = 0.5,
    timeout: float = 300.0,
    max_retries: int = 4,
    verify_capabilities: bool = True,
) -> Path:
    """Download the AHN coverage for `bbox` as a GeoTIFF.

    The request is built by hand rather than through ``owslib``. owslib follows
    the operation URL advertised in the capabilities document, which currently
    points at a path that 404s, and it serialises a list-valued coverage id into
    the query string. A plain GET avoids both problems.
    """
    coverage_id = _coverage_id(ahn_model)

    if verify_capabilities:
        try:
            available = discover_coverages(wcs_url, timeout=min(timeout, 120.0))
        except Exception as exc:  # noqa: BLE001 - capabilities are advisory
            LOG.warning("could not read AHN capabilities (%s); continuing", exc)
        else:
            if available and coverage_id not in available:
                raise ServiceError(
                    f"coverage {coverage_id!r} is not offered by {wcs_url}; "
                    f"available: {', '.join(available)}"
                )

    columns = int(round(bbox.width / resolution_m))
    rows = int(round(bbox.height / resolution_m))

    if columns <= MAX_COVERAGE_PX and rows <= MAX_COVERAGE_PX:
        LOG.info("requesting AHN %s for %s", coverage_id, bbox)
        content = _get_coverage(
            bbox,
            wcs_url=wcs_url,
            coverage_id=coverage_id,
            timeout=timeout,
            max_retries=max_retries,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content)
        LOG.info("wrote %s (%.1f MB)", out_path, len(content) / 1e6)
        return out_path

    return _fetch_mosaic(
        bbox,
        out_path,
        wcs_url=wcs_url,
        coverage_id=coverage_id,
        resolution_m=resolution_m,
        columns=columns,
        rows=rows,
        timeout=timeout,
        max_retries=max_retries,
    )


def raster_sampler(path: Path):
    """Nearest-cell lookup into a GeoTIFF, with nodata as NaN.

    Bridges and elevated railways both need to ask the surface model how high
    something is, and a deck is a hard surface the DSM sees where the DTM — the
    bare ground by definition — has a hole.
    """
    import rasterio

    with rasterio.open(path) as dataset:
        band = dataset.read(1).astype(np.float64)
        left, bottom, right, top = (
            dataset.bounds.left,
            dataset.bounds.bottom,
            dataset.bounds.right,
            dataset.bounds.top,
        )
    height, width = band.shape
    band = np.where(band < NODATA_CUTOFF, band, np.nan)

    def sample(x, y):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        columns = np.clip(
            ((x - left) / max(right - left, 1e-9) * width).astype(np.int64), 0, width - 1
        )
        # Raster rows run north-down.
        rows = np.clip(
            ((top - y) / max(top - bottom, 1e-9) * height).astype(np.int64),
            0,
            height - 1,
        )
        return band[rows, columns]

    return sample


def ensure_dsm(
    bbox: BBox,
    work_dir: Path,
    *,
    wcs_url: str,
    resolution_m: float = 0.5,
    timeout: float = 300.0,
    max_retries: int = 4,
) -> Path | None:
    """The surface model over this area, fetched only if it is not already there.

    One file per area, shared by whoever needs it: the trees stage subtracts it
    from the terrain for canopy heights, and bridges and elevated track read
    deck heights straight off it.
    """
    path = work_dir / "ahn_dsm.tif"
    if path.is_file():
        return path
    try:
        fetch_dtm_geotiff(
            bbox,
            path,
            wcs_url=wcs_url,
            ahn_model="DSM",
            resolution_m=resolution_m,
            timeout=timeout,
            max_retries=max_retries,
            verify_capabilities=False,
        )
    except Exception as exc:  # noqa: BLE001 - a missing DSM is a fallback, not a failure
        LOG.warning("could not fetch the AHN surface model (%s)", short_error(exc))
        return None
    return path


def _get_coverage(
    bbox: BBox,
    *,
    wcs_url: str,
    coverage_id: str,
    timeout: float,
    max_retries: int,
) -> bytes:
    """One GetCoverage call, returning the GeoTIFF bytes."""
    params = {
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCoverage",
        "coverageId": coverage_id,
        "format": "image/tiff",
        # WCS 2.0 subsetting uses the coverage's own axis labels, which this
        # service reports as lowercase x and y in EPSG:28992.
        "subset": [
            f"x({bbox.xmin:.3f},{bbox.xmax:.3f})",
            f"y({bbox.ymin:.3f},{bbox.ymax:.3f})",
        ],
    }

    response = get_with_retry(
        wcs_url,
        params=params,
        timeout=timeout,
        max_retries=max_retries,
        expect_binary=True,
        description=f"AHN WCS GetCoverage ({coverage_id})",
    )

    if response.content[:2] not in (b"II", b"MM"):
        raise ServiceError(
            f"AHN WCS returned {len(response.content)} bytes that are not a TIFF "
            f"(content-type {response.headers.get('content-type')!r})"
        )
    return response.content


def _fetch_mosaic(
    bbox: BBox,
    out_path: Path,
    *,
    wcs_url: str,
    coverage_id: str,
    resolution_m: float,
    columns: int,
    rows: int,
    timeout: float,
    max_retries: int,
) -> Path:
    """Fetch a coverage too large for one request, in pieces.

    The service honours the requested bounds exactly and returns
    ``span / resolution`` pixels, so tiles split on whole pixels butt up with
    no seam and no overlap. Splitting in pixel space rather than in metres is
    what guarantees that: a boundary at an arbitrary coordinate would land
    mid-pixel and each side would round it differently.
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    col_spans = tile_edges(columns, MAX_COVERAGE_PX)
    row_spans = tile_edges(rows, MAX_COVERAGE_PX)
    LOG.info(
        "AHN %s over %s needs %dx%d px, past the %d px the service allows; "
        "fetching as %dx%d tiles",
        coverage_id,
        bbox,
        columns,
        rows,
        MAX_COVERAGE_PX,
        len(col_spans),
        len(row_spans),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = None
    total = len(col_spans) * len(row_spans)
    done = 0

    with tempfile.TemporaryDirectory() as scratch:
        pieces = []
        for row_index, (row_start, row_end) in enumerate(row_spans):
            for col_index, (col_start, col_end) in enumerate(col_spans):
                # Rows run north-down, so the first row is at the top.
                tile = BBox(
                    bbox.xmin + col_start * resolution_m,
                    bbox.ymax - row_end * resolution_m,
                    bbox.xmin + col_end * resolution_m,
                    bbox.ymax - row_start * resolution_m,
                )
                content = _get_coverage(
                    tile,
                    wcs_url=wcs_url,
                    coverage_id=coverage_id,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                path = Path(scratch) / f"tile_{row_index}_{col_index}.tif"
                path.write_bytes(content)
                pieces.append((path, row_start, col_start))
                done += 1
                LOG.info(
                    "  AHN tile %d/%d (%d x %d px)",
                    done,
                    total,
                    col_end - col_start,
                    row_end - row_start,
                )

        with rasterio.open(pieces[0][0]) as first:
            profile = first.profile.copy()

        profile.update(
            width=columns,
            height=rows,
            transform=from_origin(bbox.xmin, bbox.ymax, resolution_m, resolution_m),
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
        )

        with rasterio.open(out_path, "w", **profile) as mosaic:
            for path, row_start, col_start in pieces:
                with rasterio.open(path) as tile:
                    mosaic.write(
                        tile.read(1),
                        1,
                        window=Window(col_start, row_start, tile.width, tile.height),
                    )

    LOG.info(
        "wrote %s (%.1f MB, %d x %d px from %d tiles)",
        out_path,
        out_path.stat().st_size / 1e6,
        columns,
        rows,
        total,
    )
    return out_path


def _resample_to_grid(
    values: np.ndarray,
    valid: np.ndarray,
    raster_bounds: tuple[float, float, float, float],
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average raster cells into the target vertex grid.

    Each source pixel is assigned to its nearest target vertex and averaged
    there. Block averaging suits a DTM better than point sampling: it uses every
    valid measurement instead of throwing most of them away, and it closes small
    holes on its own. Target cells with no valid source pixel come back as NaN
    for the fill pass to deal with.
    """
    left, bottom, right, top = raster_bounds
    rows, cols = values.shape
    n = xs.size

    # Pixel centre coordinates of the source raster.
    px_x = left + (np.arange(cols, dtype=np.float64) + 0.5) * (right - left) / cols
    px_y = top - (np.arange(rows, dtype=np.float64) + 0.5) * (top - bottom) / rows

    # Map each pixel centre onto the nearest target vertex index.
    def nearest_index(coords: np.ndarray, axis: np.ndarray) -> np.ndarray:
        step = (axis[-1] - axis[0]) / (axis.size - 1)
        idx = np.rint((coords - axis[0]) / step).astype(np.int64)
        return np.clip(idx, 0, axis.size - 1)

    col_idx = nearest_index(px_x, xs)
    row_idx = nearest_index(px_y, ys)

    flat_idx = (row_idx[:, None] * n + col_idx[None, :]).ravel()
    flat_val = values.ravel()
    flat_ok = valid.ravel()

    counts = np.bincount(flat_idx[flat_ok], minlength=n * n)
    sums = np.bincount(flat_idx[flat_ok], weights=flat_val[flat_ok], minlength=n * n)

    grid = np.full(n * n, np.nan, dtype=np.float64)
    hit = counts > 0
    grid[hit] = sums[hit] / counts[hit]

    return grid.reshape(n, n), hit.reshape(n, n)


def _fill_holes(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill NaN cells by repeatedly averaging valid neighbours.

    Growing inward from the edges of each hole gives a smooth interpolation
    across building footprints, which is what a bare-earth surface under a
    building should look like anyway.
    """
    filled = grid.copy()
    missing = np.isnan(filled)
    total_missing = int(missing.sum())
    if total_missing == 0:
        return filled, 0

    if missing.all():
        raise ServiceError(
            "the terrain grid has no valid heights at all; the AHN coverage "
            "does not overlap the requested bbox"
        )

    # Bounded so a pathological input cannot spin forever. Each pass grows the
    # valid region by one cell, so the grid size is a hard upper bound.
    max_passes = 2 * max(grid.shape) + 8
    for _ in range(max_passes):
        missing = np.isnan(filled)
        if not missing.any():
            break

        padded = np.pad(filled, 1, mode="constant", constant_values=np.nan)
        neighbours = np.stack(
            [
                padded[:-2, 1:-1],  # south
                padded[2:, 1:-1],  # north
                padded[1:-1, :-2],  # west
                padded[1:-1, 2:],  # east
                padded[:-2, :-2],
                padded[:-2, 2:],
                padded[2:, :-2],
                padded[2:, 2:],
            ]
        )
        valid_neighbours = ~np.isnan(neighbours)
        counts = valid_neighbours.sum(axis=0)
        sums = np.where(valid_neighbours, neighbours, 0.0).sum(axis=0)

        can_fill = missing & (counts > 0)
        if not can_fill.any():
            break
        filled[can_fill] = sums[can_fill] / counts[can_fill]

    if np.isnan(filled).any():
        raise ServiceError(
            f"could not fill {int(np.isnan(filled).sum())} terrain cells; "
            f"the AHN coverage is probably partly outside the bbox"
        )

    return filled, total_missing


def _smooth(grid: np.ndarray, iterations: int) -> np.ndarray:
    """Light 3x3 box smoothing, edge-preserving at the borders.

    Filled areas are interpolations, so a pass or two takes the faceting off the
    boundary between measured and reconstructed ground.
    """
    out = grid
    for _ in range(max(0, iterations)):
        padded = np.pad(out, 1, mode="edge")
        stack = np.stack(
            [
                padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
                padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
                padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:],
            ]
        )
        out = stack.mean(axis=0)
    return out


def _bilinear_on_grid(grid: np.ndarray, xs: np.ndarray, ys: np.ndarray, x, y):
    """Bilinear sample of `grid` at RD coordinates, clamped to the grid edges."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n_x, n_y = xs.size, ys.size
    step_x = (xs[-1] - xs[0]) / (n_x - 1)
    step_y = (ys[-1] - ys[0]) / (n_y - 1)

    fx = np.clip((x - xs[0]) / step_x, 0, n_x - 1)
    fy = np.clip((y - ys[0]) / step_y, 0, n_y - 1)

    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_x - 1)
    j1 = np.minimum(j0 + 1, n_y - 1)
    tx = fx - i0
    ty = fy - j0

    v00 = grid[j0, i0]
    v10 = grid[j0, i1]
    v01 = grid[j1, i0]
    v11 = grid[j1, i1]

    return (
        v00 * (1 - tx) * (1 - ty)
        + v10 * tx * (1 - ty)
        + v01 * (1 - tx) * ty
        + v11 * tx * ty
    )


def build_terrain(
    bbox: BBox,
    work_dir: Path,
    *,
    terrain_cfg: dict,
    breakline_rings: dict | None = None,
) -> TerrainResult:
    """Fetch AHN for `bbox` and turn it into a clean NxN height grid."""
    import rasterio

    work_dir.mkdir(parents=True, exist_ok=True)
    n = int(terrain_cfg["mesh_vertices_per_side"])
    resolution = float(terrain_cfg["resolution_m"])
    ahn_model = str(terrain_cfg["ahn_model"]).upper()

    # A margin of a few cells keeps the outermost vertices interpolated from
    # real data rather than from the raster edge.
    margin = max(4.0 * resolution, (bbox.width / (n - 1)))
    request_bbox = bbox.buffered(margin)

    tif_path = work_dir / f"ahn_{ahn_model.lower()}.tif"
    fetch_dtm_geotiff(
        request_bbox,
        tif_path,
        wcs_url=str(terrain_cfg["wcs_url"]),
        ahn_model=ahn_model,
        resolution_m=resolution,
        timeout=float(terrain_cfg["timeout_s"]),
        max_retries=int(terrain_cfg["max_retries"]),
    )

    with rasterio.open(tif_path) as dataset:
        raw = dataset.read(1).astype(np.float64)
        bounds = (
            dataset.bounds.left,
            dataset.bounds.bottom,
            dataset.bounds.right,
            dataset.bounds.top,
        )
        raster_crs = dataset.crs
        raster_res = dataset.res

    if raster_crs is not None and raster_crs.to_epsg() not in (28992, None):
        raise ServiceError(
            f"AHN raster came back in {raster_crs} instead of EPSG:28992; "
            f"the pipeline assumes no reprojection is needed"
        )

    raster_bbox = BBox(*bounds)
    if not raster_bbox.contains(bbox, tol=raster_res[0]):
        raise ServiceError(
            f"AHN raster {raster_bbox} does not cover the requested bbox {bbox}"
        )

    # Mask nodata and anything physically implausible in one go.
    valid = (
        np.isfinite(raw)
        & (raw < NODATA_CUTOFF)
        & (raw > MIN_PLAUSIBLE_NAP)
        & (raw < MAX_PLAUSIBLE_NAP)
    )
    nodata_fraction = float(1.0 - valid.mean())
    LOG.info(
        "AHN %s raster %dx%d at %.2f m, %.1f%% nodata before filling",
        ahn_model,
        raw.shape[1],
        raw.shape[0],
        raster_res[0],
        100.0 * nodata_fraction,
    )

    max_nodata = float(terrain_cfg["max_nodata_fraction"])
    if nodata_fraction > max_nodata:
        raise ServiceError(
            f"AHN {ahn_model} is {100 * nodata_fraction:.1f}% nodata over this "
            f"bbox, above the {100 * max_nodata:.0f}% limit; the area is "
            f"probably outside AHN coverage"
        )

    xs, ys = build_grid_coords(bbox, n)
    grid, hit = _resample_to_grid(raw, valid, bounds, xs, ys)
    filled, filled_cells = _fill_holes(grid)
    filled = _smooth(filled, int(terrain_cfg["smooth_iterations"]))

    filled_fraction = filled_cells / float(n * n)
    LOG.info(
        "terrain grid %dx%d, %.1f%% of vertices interpolated from neighbours",
        n,
        n,
        100.0 * filled_fraction,
    )

    # Simplify before anything reads a height off this grid. Roads, rails,
    # trees and buildings all drape on `sample`, so replacing the grid here --
    # rather than at the end, next to the mesh that Blender draws -- is what
    # keeps them sitting on the surface Unity will actually show instead of on
    # the one that was thrown away.
    mesh = None
    constrained = None
    tolerance = float(terrain_cfg.get("simplify_tolerance_m", 0.0) or 0.0)
    if breakline_rings:
        from .terrain_mesh import build_constrained, save_constrained_mesh

        constrained = build_constrained(
            bbox,
            filled,
            xs,
            ys,
            rings_by_source=breakline_rings,
            tolerance_m=max(tolerance, 0.01),
            simplify_m=float(terrain_cfg.get("breakline_simplify_m", 0.15)),
            contour_interval_m=float(terrain_cfg.get("contour_interval_m", 0.5)),
            min_feature_length_m=float(
                terrain_cfg.get("min_feature_length_m", 0.0)
            ),
            max_face_m=float(terrain_cfg.get("max_face_m", 0.0)),
            quads=bool(terrain_cfg.get("quads", False)),
            max_fold_deg=float(terrain_cfg.get("quad_max_fold_deg", 12.0)),
            min_quad_angle_deg=float(terrain_cfg.get("quad_min_angle_deg", 25.0)),
        )
        save_constrained_mesh(
            constrained, work_dir / "terrain_mesh.npz", bbox.center
        )
        # The grid is left alone here on purpose. Mesh vertices take their
        # height from it, so anything else draped on it lands on the same
        # surface wherever the two share a vertex -- which, along every road
        # and water edge, is everywhere that matters.
    elif tolerance > 0.0:
        mesh = build_rtin(filled, tolerance_m=tolerance)
        filled = mesh.heights.astype(np.float64)
        save_terrain_mesh(mesh, work_dir / "terrain_mesh.npz")

    heights = filled.astype(np.float32)
    center_z = float(_bilinear_on_grid(filled, xs, ys, *bbox.center))

    np.save(work_dir / "terrain_heights.npy", heights)
    np.savez(
        work_dir / "terrain_grid.npz",
        heights=heights,
        xs=xs,
        ys=ys,
        bbox=np.asarray(bbox.as_list(), dtype=np.float64),
        center_z_nap=np.float64(center_z),
    )

    return TerrainResult(
        heights=heights,
        xs=xs,
        ys=ys,
        bbox=bbox,
        center_z_nap=center_z,
        nodata_fraction_raw=nodata_fraction,
        filled_fraction=filled_fraction,
        geotiff_path=tif_path,
        coverage_id=_coverage_id(ahn_model),
        resolution_m=resolution,
        mesh=mesh,
        constrained=constrained,
    )


__all__ = [
    "MAX_PLAUSIBLE_NAP",
    "MIN_PLAUSIBLE_NAP",
    "NODATA_CUTOFF",
    "TerrainResult",
    "build_terrain",
    "discover_coverages",
    "ensure_dsm",
    "fetch_dtm_geotiff",
    "raster_sampler",
]
