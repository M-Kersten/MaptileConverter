"""PDOK aerial imagery: one georeferenced ortho PNG covering the bbox.

WMS is the primary path because a single mosaic of GetMap calls is far less
code than tile arithmetic. Two service limits shape the implementation:

* The service advertises MaxWidth/MaxHeight of 2500 and answers anything larger
  with a ServiceException carrying HTTP 200. A 4096 px image therefore has to be
  stitched from several requests, and error detection cannot rely on the status
  code.
* GetMap only offers ``image/jpeg``. Tiles arrive as JPEG and are written out as
  a single PNG.

The WMTS mosaic is the fallback for when WMS refuses or throttles.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from .geo import BBox, tile_edges
from .http_util import ServiceError, get_with_retry

LOG = logging.getLogger(__name__)

# OGC's fixed metres-per-pixel constant for turning a scale denominator into a
# ground resolution.
OGC_PIXEL_SIZE_M = 0.00028

# EPSG:28992 axis order is easting, northing in both WMS 1.1.1 and 1.3.0 for
# this service. Sending northing first returns a blank white image rather than
# an error, so this is verified rather than assumed.
WMS_VERSION = "1.3.0"


@dataclass
class AerialResult:
    """The written image plus the exact bbox it covers."""

    path: Path
    bbox: BBox
    size_px: int
    layer_name: str
    layer_title: str
    source: str  # "wms" or "wmts"

    @property
    def resolution_m(self) -> float:
        return self.bbox.width / self.size_px

    def to_dict(self) -> dict:
        return {
            "file": self.path.name,
            "bbox_rd": self.bbox.as_list(),
            "size_px": self.size_px,
            "layer": self.layer_title,
            "layer_name": self.layer_name,
            "source": self.source,
            "resolution_m_per_px": round(self.resolution_m, 6),
        }


def _fetch_capabilities(url: str, service: str, timeout: float) -> str:
    response = get_with_retry(
        url,
        params={"service": service, "request": "GetCapabilities"},
        timeout=timeout,
        description=f"{service} GetCapabilities",
    )
    return response.text


def resolve_layer_name(
    wms_url: str, requested: str, timeout: float = 120.0
) -> tuple[str, str]:
    """Map a configured layer to the ``(name, title)`` the service uses.

    The config names layers the way PDOK presents them in its catalogue
    ("Luchtfoto Actueel Ortho 8cm RGB"), but GetMap wants the machine name
    ("Actueel_orthoHR"). Accepts either and returns both.
    """
    try:
        xml = _fetch_capabilities(wms_url, "WMS", timeout)
    except Exception as exc:  # noqa: BLE001 - fall back to the literal value
        LOG.warning(
            "could not read WMS capabilities (%s); using layer %r as given",
            exc,
            requested,
        )
        return requested, requested

    pairs = re.findall(
        r"<Name>([^<]+)</Name>\s*<Title>([^<]+)</Title>", xml
    )
    by_name = {name.strip(): title.strip() for name, title in pairs}
    by_title = {title.strip().casefold(): name.strip() for name, title in pairs}

    wanted = requested.strip()
    if wanted in by_name:
        return wanted, by_name[wanted]
    if wanted.casefold() in by_title:
        name = by_title[wanted.casefold()]
        return name, by_name[name]

    raise ServiceError(
        f"layer {requested!r} is not offered by {wms_url}. Available layers: "
        + ", ".join(f"{n} ({t})" for n, t in sorted(by_name.items()))
    )


def _allow_large_images() -> None:
    """Lift Pillow's decompression-bomb guard.

    The guard trips at 89.5 megapixels, which is a 9459 px square: below the
    12500 px that a 1 km area needs to reach the source's native 8 cm. The guard
    exists to stop a malicious upload from exhausting memory, and here the size
    is one the caller asked for, from a known government service.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None


def fetch_aerial_wms(
    bbox: BBox,
    out_path: Path,
    *,
    layer_name: str,
    size_px: int,
    wms_url: str,
    max_request_px: int = 2000,
    request_format: str = "image/jpeg",
    timeout: float = 180.0,
    max_retries: int = 4,
) -> Path:
    """Build a `size_px` square ortho over `bbox` from tiled WMS GetMap calls.

    Each tile's sub-bbox is derived from its exact pixel span, so the tiles line
    up seam-free and the finished image maps linearly onto `bbox`.
    """
    _allow_large_images()
    from PIL import Image

    cols = tile_edges(size_px, max_request_px)
    rows = tile_edges(size_px, max_request_px)
    LOG.info(
        "fetching aerial %dpx as %dx%d WMS tiles (service caps requests at 2500px)",
        size_px,
        len(cols),
        len(rows),
    )

    canvas = Image.new("RGB", (size_px, size_px))
    session = requests.Session()
    total = len(cols) * len(rows)
    done = 0

    for row_start, row_end in rows:
        for col_start, col_end in cols:
            tile_w = col_end - col_start
            tile_h = row_end - row_start

            # Pixel column 0 is the west edge; pixel row 0 is the north edge.
            tile_xmin = bbox.xmin + bbox.width * col_start / size_px
            tile_xmax = bbox.xmin + bbox.width * col_end / size_px
            tile_ymax = bbox.ymax - bbox.height * row_start / size_px
            tile_ymin = bbox.ymax - bbox.height * row_end / size_px

            params = {
                "service": "WMS",
                "version": WMS_VERSION,
                "request": "GetMap",
                "layers": layer_name,
                "styles": "",
                "crs": "EPSG:28992",
                "bbox": f"{tile_xmin:.4f},{tile_ymin:.4f},{tile_xmax:.4f},{tile_ymax:.4f}",
                "width": tile_w,
                "height": tile_h,
                "format": request_format,
                "transparent": "false",
            }

            response = get_with_retry(
                wms_url,
                params=params,
                timeout=timeout,
                max_retries=max_retries,
                session=session,
                expect_binary=True,
                description=f"WMS GetMap tile ({col_start},{row_start})",
            )

            tile = Image.open(io.BytesIO(response.content)).convert("RGB")
            if tile.size != (tile_w, tile_h):
                raise ServiceError(
                    f"WMS returned a {tile.size} tile where {(tile_w, tile_h)} "
                    f"was requested"
                )
            canvas.paste(tile, (col_start, row_start))

            done += 1
            if done % 4 == 0 or done == total:
                LOG.info("  aerial tiles %d/%d", done, total)

    _assert_not_blank(canvas, bbox)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    LOG.info("wrote %s (%dx%d)", out_path, size_px, size_px)
    return out_path


def _assert_not_blank(image, bbox: BBox) -> None:
    """Guard against a uniformly blank result.

    A wrong axis order or an out-of-coverage bbox both come back as a flat white
    image with HTTP 200, which is otherwise indistinguishable from success.
    """
    import numpy as np

    sample = np.asarray(image.resize((256, 256))).astype(np.float64)
    if sample.std() < 0.5:
        raise ServiceError(
            f"aerial image over {bbox} is a flat colour "
            f"(mean {sample.mean():.1f}); the bbox is likely outside coverage "
            f"or the CRS axis order is wrong"
        )


def _parse_wmts_matrices(xml: str, tile_matrix_set: str) -> list[dict]:
    """Read the tile matrix definitions for one tile matrix set."""
    match = re.search(
        r"<TileMatrixSet>\s*<ows:Identifier>\s*"
        + re.escape(tile_matrix_set)
        + r"\s*</ows:Identifier>(.*?)</TileMatrixSet>",
        xml,
        re.S,
    )
    if not match:
        raise ServiceError(
            f"WMTS capabilities has no tile matrix set {tile_matrix_set!r}"
        )

    matrices = []
    for block in re.findall(r"<TileMatrix>(.*?)</TileMatrix>", match.group(1), re.S):

        def field(tag: str) -> str:
            found = re.search(rf"<{tag}>([^<]+)</{tag}>", block)
            if not found:
                raise ServiceError(f"WMTS tile matrix is missing <{tag}>")
            return found.group(1).strip()

        top_left = field("TopLeftCorner").split()
        matrices.append(
            {
                "id": field("ows:Identifier"),
                "scale_denominator": float(field("ScaleDenominator")),
                "top_left_x": float(top_left[0]),
                "top_left_y": float(top_left[1]),
                "tile_width": int(field("TileWidth")),
                "tile_height": int(field("TileHeight")),
                "matrix_width": int(field("MatrixWidth")),
                "matrix_height": int(field("MatrixHeight")),
            }
        )
    return matrices


def fetch_aerial_wmts(
    bbox: BBox,
    out_path: Path,
    *,
    layer_name: str,
    size_px: int,
    wmts_url: str,
    tile_matrix_set: str = "EPSG:28992",
    timeout: float = 180.0,
    max_retries: int = 4,
) -> Path:
    """Mosaic WMTS tiles over `bbox`. Fallback for when WMS refuses.

    Picks the coarsest zoom level that still resolves the requested pixel size,
    fetches every tile touching the bbox, then crops and resamples the mosaic to
    exactly `size_px`.
    """
    _allow_large_images()
    from PIL import Image

    xml = _fetch_capabilities(wmts_url, "WMTS", timeout)
    matrices = _parse_wmts_matrices(xml, tile_matrix_set)

    target_res = bbox.width / size_px
    usable = [
        m
        for m in matrices
        if m["scale_denominator"] * OGC_PIXEL_SIZE_M <= target_res * 1.001
    ]
    if usable:
        # Coarsest level that still meets the target: fewest tiles for the
        # resolution we actually need.
        matrix = max(usable, key=lambda m: m["scale_denominator"])
    else:
        matrix = min(matrices, key=lambda m: m["scale_denominator"])
        LOG.warning(
            "WMTS cannot reach %.3f m/px; using its finest level %s at %.3f m/px",
            target_res,
            matrix["id"],
            matrix["scale_denominator"] * OGC_PIXEL_SIZE_M,
        )

    pixel_size = matrix["scale_denominator"] * OGC_PIXEL_SIZE_M
    tile_span_x = pixel_size * matrix["tile_width"]
    tile_span_y = pixel_size * matrix["tile_height"]

    col_min = int((bbox.xmin - matrix["top_left_x"]) // tile_span_x)
    col_max = int((bbox.xmax - matrix["top_left_x"]) // tile_span_x)
    row_min = int((matrix["top_left_y"] - bbox.ymax) // tile_span_y)
    row_max = int((matrix["top_left_y"] - bbox.ymin) // tile_span_y)

    col_min = max(0, col_min)
    row_min = max(0, row_min)
    col_max = min(matrix["matrix_width"] - 1, col_max)
    row_max = min(matrix["matrix_height"] - 1, row_max)
    if col_max < col_min or row_max < row_min:
        raise ServiceError(f"bbox {bbox} falls outside the WMTS tile grid")

    n_cols = col_max - col_min + 1
    n_rows = row_max - row_min + 1
    LOG.info(
        "WMTS fallback: level %s at %.3f m/px, %dx%d tiles",
        matrix["id"],
        pixel_size,
        n_cols,
        n_rows,
    )

    template = re.search(r'<ResourceURL[^>]*template="([^"]+)"', xml)
    session = requests.Session()
    mosaic = Image.new(
        "RGB", (n_cols * matrix["tile_width"], n_rows * matrix["tile_height"])
    )

    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            if template:
                url = (
                    template.group(1)
                    .replace("{TileMatrixSet}", tile_matrix_set)
                    .replace("{TileMatrix}", matrix["id"])
                    .replace("{TileCol}", str(col))
                    .replace("{TileRow}", str(row))
                )
                # The template is per-layer; retarget it at the layer we want.
                url = re.sub(
                    r"(/wmts/v1_0/)[^/]+/", rf"\g<1>{layer_name}/", url
                )
                params = None
            else:
                url = wmts_url
                params = {
                    "service": "WMTS",
                    "request": "GetTile",
                    "version": "1.0.0",
                    "layer": layer_name,
                    "style": "default",
                    "format": "image/jpeg",
                    "tilematrixset": tile_matrix_set,
                    "tilematrix": matrix["id"],
                    "tilerow": row,
                    "tilecol": col,
                }

            response = get_with_retry(
                url,
                params=params,
                timeout=timeout,
                max_retries=max_retries,
                session=session,
                expect_binary=True,
                description=f"WMTS tile {matrix['id']}/{col}/{row}",
            )
            tile = Image.open(io.BytesIO(response.content)).convert("RGB")
            mosaic.paste(
                tile,
                (
                    (col - col_min) * matrix["tile_width"],
                    (row - row_min) * matrix["tile_height"],
                ),
            )

    # Crop the mosaic down to the exact bbox, then resample to the target size.
    mosaic_xmin = matrix["top_left_x"] + col_min * tile_span_x
    mosaic_ymax = matrix["top_left_y"] - row_min * tile_span_y

    left = (bbox.xmin - mosaic_xmin) / pixel_size
    right = (bbox.xmax - mosaic_xmin) / pixel_size
    top = (mosaic_ymax - bbox.ymax) / pixel_size
    bottom = (mosaic_ymax - bbox.ymin) / pixel_size

    cropped = mosaic.resize(
        (size_px, size_px), Image.LANCZOS, box=(left, top, right, bottom)
    )

    _assert_not_blank(cropped, bbox)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out_path, format="PNG")
    LOG.info("wrote %s (%dx%d) from WMTS", out_path, size_px, size_px)
    return out_path


def write_world_file(path: Path, bbox: BBox, size_px: int) -> Path:
    """Write an ESRI world file so the PNG opens georeferenced in GIS tools."""
    res_x = bbox.width / size_px
    res_y = bbox.height / size_px
    world_path = path.with_suffix(".pgw")
    world_path.write_text(
        "\n".join(
            [
                f"{res_x:.10f}",
                "0.0000000000",
                "0.0000000000",
                f"{-res_y:.10f}",
                # World files reference the centre of the top-left pixel.
                f"{bbox.xmin + res_x / 2:.6f}",
                f"{bbox.ymax - res_y / 2:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return world_path


def build_aerial(bbox: BBox, work_dir: Path, *, aerial_cfg: dict) -> AerialResult:
    """Fetch the ortho for `bbox`, trying WMS first and WMTS as fallback."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "aerial.png"
    size_px = int(aerial_cfg["size_px"])
    timeout = float(aerial_cfg["timeout_s"])
    max_retries = int(aerial_cfg["max_retries"])

    layer_name, layer_title = resolve_layer_name(
        str(aerial_cfg["wms_url"]), str(aerial_cfg["layer"]), timeout=min(timeout, 120)
    )
    LOG.info("aerial layer %r resolves to WMS name %r", layer_title, layer_name)

    source = "wms"
    try:
        fetch_aerial_wms(
            bbox,
            out_path,
            layer_name=layer_name,
            size_px=size_px,
            wms_url=str(aerial_cfg["wms_url"]),
            max_request_px=int(aerial_cfg["max_request_px"]),
            request_format=str(aerial_cfg["request_format"]),
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as exc:  # noqa: BLE001 - this is exactly what WMTS is for
        LOG.warning("WMS path failed (%s); falling back to WMTS tiles", exc)
        fetch_aerial_wmts(
            bbox,
            out_path,
            layer_name=layer_name,
            size_px=size_px,
            wmts_url=str(aerial_cfg["wmts_url"]),
            timeout=timeout,
            max_retries=max_retries,
        )
        source = "wmts"

    write_world_file(out_path, bbox, size_px)

    return AerialResult(
        path=out_path,
        bbox=bbox,
        size_px=size_px,
        layer_name=layer_name,
        layer_title=layer_title,
        source=source,
    )


__all__ = [
    "AerialResult",
    "build_aerial",
    "fetch_aerial_wms",
    "fetch_aerial_wmts",
    "resolve_layer_name",
    "write_world_file",
]
