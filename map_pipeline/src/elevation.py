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

LOG = logging.getLogger(__name__)

# Spacing of the terrain grid when nothing asks for another. Measured against
# every 0.5 m AHN pixel over Bronbeek, a 2 m grid is within 11 cm for 95% of
# the ground and off by more than 25 cm on 0.8% of it, at 251k vertices. The
# adaptive triangulation it replaced held 90k vertices and was off by more than
# 25 cm on 12% of the same ground, because it was cut from a smoothed 3.9 m
# grid rather than from the survey.
DEFAULT_GRID_M = 2.0

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


class HeightRaster:
    """Heights on a regular raster over the bbox, NaN where nothing is.

    What the road surface looks like from above: where a road is, this says
    how high its top is; everywhere else it says nothing, and whoever asks
    falls back to the ground.
    """

    def __init__(self, values: np.ndarray, bbox: BBox):
        self.values = np.asarray(values, dtype=np.float64)
        self.bbox = bbox
        rows, cols = self.values.shape
        self.cell_x = bbox.width / cols
        self.cell_y = bbox.height / rows

    def __call__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rows, cols = self.values.shape
        col = np.floor((x - self.bbox.xmin) / self.cell_x).astype(np.int64)
        # Raster rows run north-down.
        row = np.floor((self.bbox.ymax - y) / self.cell_y).astype(np.int64)
        inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
        out = np.full(np.broadcast(x, y).shape, np.nan)
        out[inside] = self.values[row[inside], col[inside]]
        return out


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
    # The grid as surveyed, before anything was pushed down under a road.
    # `heights` is what the model shows; this is what the ground is.
    measured: np.ndarray | None = None
    # The top of the road surface, once the roads are built. See `surface`.
    road_top: HeightRaster | None = None
    # Grid vertices lowered under the road surface, and by how much at most.
    conformed: dict | None = None
    # How far the grid strays from every surveyed pixel it was built from.
    survey_error: dict | None = None

    @property
    def n(self) -> int:
        return int(self.heights.shape[0])

    @property
    def grid_m(self) -> float:
        return float(self.xs[1] - self.xs[0]) if self.n > 1 else 0.0

    def sample(self, x, y):
        """The surveyed ground at RD coordinates, bilinear, clamped to the grid.

        Not lowered under the roads: this is the ground as measured, which is
        what a road, a building or a check wants to know about.
        """
        grid = self.measured if self.measured is not None else self.heights
        return _bilinear_on_grid(grid, self.xs, self.ys, x, y)

    def surface(self, x, y):
        """The top of whatever the model shows at RD coordinates.

        On a road that is the road. Everywhere else it is the terrain exactly
        as it is drawn: each grid square is two flat triangles split corner to
        corner from south-west to north-east, which is the split Blender and
        the FBX export make, so a lamp post or a tree trunk stands on the face
        it will be seen standing on rather than on a curve between them.
        """
        ground = _triangulated_on_grid(self.heights, self.xs, self.ys, x, y)
        if self.road_top is None:
            return ground
        top = self.road_top(x, y)
        return np.where(np.isfinite(top), top, ground)

    def stats(self) -> dict:
        h = self.heights
        out = {
            "min_nap": float(h.min()),
            "max_nap": float(h.max()),
            "mean_nap": float(h.mean()),
            "center_nap": float(self.center_z_nap),
            "vertices_per_side": self.n,
            "grid_m": round(self.grid_m, 4),
            "nodata_fraction_raw": float(self.nodata_fraction_raw),
            "filled_fraction": float(self.filled_fraction),
        }
        if self.survey_error:
            out["error_against_survey"] = dict(self.survey_error)
        if self.conformed:
            out["lowered_under_roads"] = dict(self.conformed)
        return out

    def save(self, work_dir: Path) -> None:
        """Write the grid the Blender stage builds the terrain from."""
        np.save(work_dir / "terrain_heights.npy", self.heights.astype(np.float32))
        np.savez(
            work_dir / "terrain_grid.npz",
            heights=self.heights.astype(np.float32),
            measured=(
                self.measured if self.measured is not None else self.heights
            ).astype(np.float32),
            xs=self.xs,
            ys=self.ys,
            bbox=np.asarray(self.bbox.as_list(), dtype=np.float64),
            center_z_nap=np.float64(self.center_z_nap),
        )


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


def _triangulated_on_grid(grid: np.ndarray, xs: np.ndarray, ys: np.ndarray, x, y):
    """Height on the grid drawn as triangles, split south-west to north-east.

    A grid square is four heights that are almost never coplanar, so the
    surface between them depends on which diagonal the renderer splits it
    along. Bilinear interpolation is neither split; it is a curved patch no
    engine draws. Blender's fixed quad split and the FBX export both cut from
    the first corner to the third, which on a square wound south-west,
    south-east, north-east, north-west is the SW-NE diagonal, so that is the
    surface reported here.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n_x, n_y = xs.size, ys.size
    step_x = (xs[-1] - xs[0]) / (n_x - 1)
    step_y = (ys[-1] - ys[0]) / (n_y - 1)

    fx = np.clip((x - xs[0]) / step_x, 0, n_x - 1)
    fy = np.clip((y - ys[0]) / step_y, 0, n_y - 1)
    # The last row and column belong to the square below and left of them.
    i0 = np.minimum(np.floor(fx).astype(np.int64), n_x - 2)
    j0 = np.minimum(np.floor(fy).astype(np.int64), n_y - 2)
    tx = fx - i0
    ty = fy - j0

    v00 = grid[j0, i0]
    v10 = grid[j0, i0 + 1]
    v01 = grid[j0 + 1, i0]
    v11 = grid[j0 + 1, i0 + 1]

    lower = tx >= ty  # south-east of the diagonal
    return np.where(
        lower,
        v00 + tx * (v10 - v00) + ty * (v11 - v10),
        v00 + ty * (v01 - v00) + tx * (v11 - v01),
    )


def grid_size_for(bbox: BBox, grid_m: float) -> int:
    """Vertices per side for a grid no coarser than `grid_m` over the bbox.

    Sized on the longer side, so a slightly oblong area gets slightly finer
    spacing across its short side rather than coarser along its long one.
    """
    if grid_m <= 0:
        raise ValueError(f"terrain.grid_m must be positive, got {grid_m}")
    return max(2, int(np.ceil(max(bbox.width, bbox.height) / grid_m - 1e-9)) + 1)


def build_terrain(
    bbox: BBox,
    work_dir: Path,
    *,
    terrain_cfg: dict,
) -> TerrainResult:
    """Fetch AHN for `bbox` and turn it into a regular height grid.

    Regular because it is edited. A sculptor, a loop select and a Unity
    heightmap all expect rows and columns, and every adaptive mesh this
    pipeline tried instead traded that away for a vertex count nobody had
    asked to save.
    """
    import rasterio

    work_dir.mkdir(parents=True, exist_ok=True)
    n = grid_size_for(bbox, float(terrain_cfg.get("grid_m", DEFAULT_GRID_M)))
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
    # Each vertex is the mean of the survey pixels nearest it. Measured at 2 m
    # over Bronbeek that is within 11 cm of the survey for 95% of the ground,
    # against 13 cm for reading the single pixel under each vertex.
    grid, hit = _resample_to_grid(raw, valid, bounds, xs, ys)
    filled, filled_cells = _fill_holes(grid)
    # Smoothing takes the faceting off the ground the filler invented under
    # buildings and water, and only there. Smoothing the measured ground as
    # well doubled the error of a 2 m grid, to 21 cm at the 95th percentile.
    smoothed = _smooth(filled, int(terrain_cfg.get("smooth_iterations", 1)))
    filled = np.where(hit, filled, smoothed)

    filled_fraction = filled_cells / float(n * n)
    LOG.info(
        "terrain grid %dx%d at %.2f m, %.1f%% of vertices interpolated from "
        "neighbours",
        n,
        n,
        float(xs[1] - xs[0]),
        100.0 * filled_fraction,
    )

    heights = filled.astype(np.float32)
    center_z = float(_bilinear_on_grid(filled, xs, ys, *bbox.center))

    result = TerrainResult(
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
        measured=heights.copy(),
        survey_error=_survey_error(raw, valid, bounds, bbox, filled, xs, ys),
    )
    if result.survey_error:
        LOG.info(
            "terrain grid within %.3f m of the survey on 95%% of the ground, "
            "%.3f m on 99%%",
            result.survey_error["p95_m"],
            result.survey_error["p99_m"],
        )
    result.save(work_dir)
    return result


def _survey_error(raw, valid, bounds, bbox: BBox, grid, xs, ys) -> dict:
    """How far the grid, as drawn, strays from every surveyed pixel under it.

    Measured against the triangles the model actually shows rather than a
    bilinear patch no renderer draws, and only over measured pixels inside the
    bbox: the filled ones are the filler's own guess and agree with it by
    construction.
    """
    rows, cols = raw.shape
    left, bottom, right, top = bounds
    px = left + (np.arange(cols) + 0.5) * (right - left) / cols
    py = top - (np.arange(rows) + 0.5) * (top - bottom) / rows
    keep_c = (px > bbox.xmin) & (px < bbox.xmax)
    keep_r = (py > bbox.ymin) & (py < bbox.ymax)
    if not keep_c.any() or not keep_r.any():
        return {}
    window = valid[np.ix_(keep_r, keep_c)]
    if not window.any():
        return {}
    grid_x, grid_y = np.meshgrid(px[keep_c], py[keep_r])
    truth = raw[np.ix_(keep_r, keep_c)][window]
    drawn = _triangulated_on_grid(grid, xs, ys, grid_x[window], grid_y[window])
    error = np.abs(drawn - truth)
    p50, p95, p99 = np.percentile(error, [50, 95, 99])
    return {
        "p50_m": round(float(p50), 3),
        "p95_m": round(float(p95), 3),
        "p99_m": round(float(p99), 3),
        "over_25cm": round(float((error > 0.25).mean()), 4),
    }


def lower_under(
    terrain: TerrainResult,
    underside: HeightRaster,
    *,
    clearance_m: float,
) -> dict:
    """Push the grid down wherever a road lies over it, and nowhere else.

    The road is a separate surface laid on top, so the terrain only has to
    stay out of its way. Each vertex is held below the lowest point of road
    anywhere in the squares around it, by `clearance_m`. That is the condition
    that makes it impossible for the ground to show through: a triangle under
    a road has all three corners below the lowest road over it, so every point
    of the triangle is too. Testing only the road height at the vertex itself
    lets a bump between vertices poke through on any slope.

    Returns how many vertices moved and how far, for the metadata.
    """
    n = terrain.n
    step_x = terrain.bbox.width / (n - 1)
    step_y = terrain.bbox.height / (n - 1)
    values = underside.values
    rows, cols = values.shape

    # Lowest road under each cell of the road raster, gathered per grid square
    # first and then per vertex over the squares that share it.
    low = np.where(np.isfinite(values), values, np.inf)
    # Cell centres of the raster, mapped to the grid square they fall in.
    centre_x = terrain.bbox.xmin + (np.arange(cols) + 0.5) * underside.cell_x
    centre_y = terrain.bbox.ymax - (np.arange(rows) + 0.5) * underside.cell_y
    square_col = np.clip(((centre_x - terrain.bbox.xmin) / step_x).astype(np.int64), 0, n - 2)
    square_row = np.clip(((centre_y - terrain.bbox.ymin) / step_y).astype(np.int64), 0, n - 2)
    per_square = np.full((n - 1, n - 1), np.inf)
    flat = (square_row[:, None] * (n - 1) + square_col[None, :]).ravel()
    np.minimum.at(per_square.reshape(-1), flat, low.ravel())

    # A vertex touches up to four squares.
    padded = np.pad(per_square, 1, constant_values=np.inf)
    per_vertex = np.minimum.reduce(
        [padded[:-1, :-1], padded[:-1, 1:], padded[1:, :-1], padded[1:, 1:]]
    )
    ceiling = per_vertex - clearance_m
    before = terrain.heights.astype(np.float64)
    after = np.minimum(before, ceiling)
    moved = after < before - 1e-6
    terrain.heights = after.astype(np.float32)
    drop = (before - after)[moved]
    terrain.conformed = {
        "vertices": int(moved.sum()),
        "max_m": round(float(drop.max()), 3) if moved.any() else 0.0,
        "median_m": round(float(np.median(drop)), 3) if moved.any() else 0.0,
        "clearance_m": float(clearance_m),
    }
    LOG.info(
        "terrain: %d vertices lowered under the roads, median %.2f m, at most %.2f m",
        terrain.conformed["vertices"],
        terrain.conformed["median_m"],
        terrain.conformed["max_m"],
    )
    return terrain.conformed


__all__ = [
    "DEFAULT_GRID_M",
    "HeightRaster",
    "MAX_PLAUSIBLE_NAP",
    "MIN_PLAUSIBLE_NAP",
    "NODATA_CUTOFF",
    "TerrainResult",
    "build_terrain",
    "discover_coverages",
    "ensure_dsm",
    "fetch_dtm_geotiff",
    "grid_size_for",
    "lower_under",
    "raster_sampler",
]
