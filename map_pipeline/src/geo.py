"""Coordinate handling for the pipeline.

Everything downstream of this module works in RD New (EPSG:28992) with NAP
heights. The only reprojection that ever happens is here, at the edge, when the
caller hands in a WGS84 bounding box.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

LOG = logging.getLogger(__name__)

RD_CRS = "EPSG:28992"
WGS84_CRS = "EPSG:4326"
# 3DBAG publishes in EPSG:7415, which is RD New horizontally plus NAP heights.
# Horizontally identical to 28992, so no reprojection is needed for it.
RD_NAP_CRS = "EPSG:7415"

# A v1 area is meant to be roughly one square kilometre. Outside this range the
# pipeline still runs, it just warns.
EXPECTED_SIDE_M = 1000.0
SIDE_TOLERANCE_M = 100.0
SQUARENESS_TOLERANCE = 0.1


@dataclass(frozen=True)
class BBox:
    """An axis-aligned bounding box. Units follow the CRS it belongs to."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError(
                f"degenerate bbox: xmin={self.xmin} ymin={self.ymin} "
                f"xmax={self.xmax} ymax={self.ymax} (min must be below max)"
            )

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.xmin + self.xmax), 0.5 * (self.ymin + self.ymax))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def as_list(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def buffered(self, margin: float) -> "BBox":
        """Grow the box by `margin` on every side."""
        return BBox(
            self.xmin - margin,
            self.ymin - margin,
            self.xmax + margin,
            self.ymax + margin,
        )

    def contains(self, other: "BBox", tol: float = 1e-6) -> bool:
        return (
            self.xmin <= other.xmin + tol
            and self.ymin <= other.ymin + tol
            and self.xmax >= other.xmax - tol
            and self.ymax >= other.ymax - tol
        )

    def __str__(self) -> str:
        return (
            f"[{self.xmin:.3f}, {self.ymin:.3f}, {self.xmax:.3f}, {self.ymax:.3f}]"
        )


def _transformer(src_crs: str, dst_crs: str):
    from pyproj import Transformer

    # always_xy keeps the argument order at (x, y) / (lon, lat) regardless of
    # what the CRS authority says the axis order is. Without it EPSG:4326 comes
    # out as (lat, lon) and every coordinate silently lands in the wrong place.
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def wgs84_bbox_to_rd(bbox: BBox, edge_samples: int = 16) -> BBox:
    """Convert a lon/lat bbox to an RD New bbox.

    The projection is not affine, so a straight four-corner transform can clip
    the edges. Sampling along each edge and taking the envelope of the result
    keeps the RD box a true superset of the WGS84 box.
    """
    tf = _transformer(WGS84_CRS, RD_CRS)

    lons: list[float] = []
    lats: list[float] = []
    for i in range(edge_samples + 1):
        t = i / edge_samples
        lon = bbox.xmin + t * bbox.width
        lat = bbox.ymin + t * bbox.height
        # Bottom and top edges.
        lons.extend([lon, lon])
        lats.extend([bbox.ymin, bbox.ymax])
        # Left and right edges.
        lons.extend([bbox.xmin, bbox.xmax])
        lats.extend([lat, lat])

    xs, ys = tf.transform(lons, lats)
    return BBox(min(xs), min(ys), max(xs), max(ys))


def rd_bbox_to_wgs84(bbox: BBox) -> BBox:
    """Inverse of :func:`wgs84_bbox_to_rd`, for metadata and sanity checks."""
    tf = _transformer(RD_CRS, WGS84_CRS)
    xs = [bbox.xmin, bbox.xmax, bbox.xmin, bbox.xmax]
    ys = [bbox.ymin, bbox.ymin, bbox.ymax, bbox.ymax]
    lons, lats = tf.transform(xs, ys)
    return BBox(min(lons), min(lats), max(lons), max(lats))


def parse_bbox(spec: dict) -> BBox:
    """Read a bbox out of the config and return it in RD New.

    Accepts ``EPSG:28992`` (used as-is) or ``EPSG:4326`` (converted once, here).
    For 4326 the fields are longitude in x and latitude in y.
    """
    missing = [k for k in ("xmin", "ymin", "xmax", "ymax") if k not in spec]
    if missing:
        raise ValueError(f"bbox config is missing keys: {', '.join(missing)}")

    raw = BBox(
        float(spec["xmin"]),
        float(spec["ymin"]),
        float(spec["xmax"]),
        float(spec["ymax"]),
    )
    crs = str(spec.get("crs", RD_CRS)).strip().upper()

    if crs in (RD_CRS, "EPSG:7415", "28992", "7415"):
        return raw
    if crs in (WGS84_CRS, "4326", "WGS84"):
        converted = wgs84_bbox_to_rd(raw)
        LOG.info("converted WGS84 bbox %s to RD %s", raw, converted)
        return converted
    raise ValueError(
        f"unsupported bbox CRS {crs!r}; use EPSG:28992 (RD New) or EPSG:4326 (lon/lat)"
    )


def validate_bbox(bbox: BBox) -> list[str]:
    """Check the box is roughly the 1 km square the pipeline is tuned for.

    Returns a list of human-readable warnings; an empty list means all good.
    Anything genuinely unusable raises instead.
    """
    warnings: list[str] = []

    # RD New is only defined over the Netherlands. Well outside it means the
    # caller almost certainly mixed up a CRS somewhere upstream.
    if not (-7000 <= bbox.xmin <= 300000 and 289000 <= bbox.ymin <= 629000):
        raise ValueError(
            f"bbox {bbox} falls outside the RD New domain; check the bbox CRS "
            f"(lon/lat values need crs=EPSG:4326)"
        )

    for label, side in (("width", bbox.width), ("height", bbox.height)):
        low = EXPECTED_SIDE_M - SIDE_TOLERANCE_M
        high = EXPECTED_SIDE_M + SIDE_TOLERANCE_M
        if not (low <= side <= high):
            warnings.append(
                f"bbox {label} is {side:.1f} m, outside the tuned "
                f"{low:.0f}-{high:.0f} m range; the pipeline still runs but "
                f"cost and mesh density scale with area"
            )

    longest = max(bbox.width, bbox.height)
    shortest = min(bbox.width, bbox.height)
    if longest / shortest > 1.0 + SQUARENESS_TOLERANCE:
        warnings.append(
            f"bbox is not square: {bbox.width:.1f} x {bbox.height:.1f} m "
            f"(aspect {longest / shortest:.2f}); the aerial image is square, so "
            f"pixels will not be square on the ground"
        )

    return warnings


@dataclass(frozen=True)
class GeoContext:
    """Ties the RD bbox to the local, origin-centred metre frame.

    Local coordinates put (0, 0) at the middle of the bbox so the geometry
    reaching Unity stays in the low hundreds of metres and float32 keeps its
    precision.
    """

    bbox: BBox
    origin_x: float
    origin_y: float

    @classmethod
    def from_bbox(cls, bbox: BBox) -> "GeoContext":
        cx, cy = bbox.center
        return cls(bbox=bbox, origin_x=cx, origin_y=cy)

    @property
    def origin(self) -> tuple[float, float]:
        return (self.origin_x, self.origin_y)

    def rd_to_local(self, x, y):
        """Shift RD coordinates into local metres around the origin.

        Works for scalars and for numpy arrays.
        """
        return (x - self.origin_x, y - self.origin_y)

    def local_to_rd(self, x, y):
        """Inverse of :meth:`rd_to_local`."""
        return (x + self.origin_x, y + self.origin_y)

    def local_bbox(self) -> BBox:
        xmin, ymin = self.rd_to_local(self.bbox.xmin, self.bbox.ymin)
        xmax, ymax = self.rd_to_local(self.bbox.xmax, self.bbox.ymax)
        return BBox(xmin, ymin, xmax, ymax)

    def to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        """RD to lon/lat, for metadata and for eyeballing a location on a map."""
        tf = _transformer(RD_CRS, WGS84_CRS)
        return tf.transform(x, y)


def tile_edges(total_px: int, max_px: int) -> list[tuple[int, int]]:
    """Split `total_px` into contiguous spans of at most `max_px` pixels.

    Both PDOK services this pipeline reads from cap the size of a single
    response — the aerial WMS at 2500 px, the AHN WCS at 4000 — so both have to
    ask for large areas in pieces. Splitting evenly rather than taking full
    tiles and a remainder keeps the last piece from being a sliver.
    """
    n_tiles = max(1, -(-total_px // max_px))
    edges = [round(i * total_px / n_tiles) for i in range(n_tiles + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n_tiles)]


def build_grid_coords(bbox: BBox, n: int):
    """RD coordinates of an ``n x n`` vertex grid spanning the bbox.

    The grid includes both edges, so spacing is ``width / (n - 1)``. Returns
    ``(xs, ys)`` as 1-D arrays; ``ys`` ascends northward, so row 0 is the
    southern edge.
    """
    import numpy as np

    if n < 2:
        raise ValueError(f"grid needs at least 2 vertices per side, got {n}")
    xs = np.linspace(bbox.xmin, bbox.xmax, n, dtype=np.float64)
    ys = np.linspace(bbox.ymin, bbox.ymax, n, dtype=np.float64)
    return xs, ys


def format_warnings(warnings: Iterable[str], prefix: str = "  - ") -> str:
    return "\n".join(f"{prefix}{w}" for w in warnings)


__all__ = [
    "tile_edges",
    "BBox",
    "GeoContext",
    "RD_CRS",
    "RD_NAP_CRS",
    "WGS84_CRS",
    "build_grid_coords",
    "parse_bbox",
    "rd_bbox_to_wgs84",
    "validate_bbox",
    "wgs84_bbox_to_rd",
]
