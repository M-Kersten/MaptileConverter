"""Roof textures at a finer resolution than the ground, packed into atlases.

The ground's aerial covers the whole area at the detail level's resolution:
25 cm/px at Standard, where a roof seen from a drone or from the street is a
blur. The ortho itself is 8 cm. Fetching the whole area at 8 cm is what High
detail nearly does, and it costs the ground as much as the roofs. Here only
the roofs are fetched finer:

1. **Plan.** Each building's roof extent, padded by a few pixels, becomes a
   rectangle on one pixel grid anchored at the area's north-west corner.
   Rectangles that overlap are merged when the merged one is no larger than
   the two apart (a terrace becomes one strip). They are packed into pages
   of at most `roof_atlas_px` a side; if the pages add up to more than the
   budget, the resolution is coarsened until they fit, and the run says so.
2. **Fetch.** The grid is cut into tiles of the service's request size, and
   each tile that holds any roof is one GetMap request for the part of it
   the roofs need. Every pixel lands on the same grid, so a rectangle split
   across two requests joins without a seam and nothing is resampled.
3. **Map.** Each roof triangle gets UVs into its rectangle. The relighting
   stage then takes the flown sun out of the atlas exactly as it does out
   of the ground (`relight.Relighting`).

Pages are written as ``roofs_00.png`` and up beside the other textures.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import BBox

LOG = logging.getLogger(__name__)

# The ortho the service holds. Finer than this is upsampling a JPEG.
SOURCE_M_PER_PX = 0.08

# How many GetMap requests run at once. The roofs of a square kilometre at
# 10 cm are a few dozen requests; one at a time that is most of a minute of
# waiting on round trips, and four is what a browser opens to one host anyway.
FETCH_WORKERS = 4

# Pages are made a multiple of this, which block compression (DXT, BC7,
# ASTC 4x4) needs.
PAGE_ALIGN = 4

# Of a page, about this much ends up covered when rectangles of mixed sizes
# are packed, used to size a single page before packing it.
PACK_FILL = 0.85

FILE_STEM = "roofs"
UV_FILE = "roofs.npz"


def _ceil_to(value: float, step: int) -> int:
    return int(-(-int(math.ceil(value)) // step) * step)


# --------------------------------------------------------------------------
# Planning: rectangles, merging, packing, the budget
# --------------------------------------------------------------------------


def roof_extents(roofs: list[np.ndarray]) -> np.ndarray:
    """Each building's roof extent ``(xmin, ymin, xmax, ymax)`` in RD.

    NaN for a building with no roof triangles, which then gets no rectangle.
    """
    out = np.full((len(roofs), 4), np.nan)
    for index, tris in enumerate(roofs):
        if len(tris):
            xy = np.asarray(tris, dtype=np.float64).reshape(-1, 3)[:, :2]
            out[index, :2] = xy.min(axis=0)
            out[index, 2:] = xy.max(axis=0)
    return out


def grid_rects(
    extents: np.ndarray,
    origin: tuple[float, float],
    metres_per_px: float,
    padding_px: int,
) -> np.ndarray:
    """Extents as padded pixel rectangles ``(col0, row0, col1, row1)``.

    On one grid for the whole area: column 0 starts at ``origin[0]`` and row 0
    at ``origin[1]``, the north edge, rows running south. End-exclusive. Every
    roof point is at least `padding_px` pixels inside its rectangle, which is
    what keeps a mipmap from reading the neighbour packed next to it.
    """
    gx, gy = origin
    r = float(metres_per_px)
    pad = int(padding_px)
    col0 = np.floor((extents[:, 0] - gx) / r) - pad
    col1 = np.ceil((extents[:, 2] - gx) / r) + pad
    row0 = np.floor((gy - extents[:, 3]) / r) - pad
    row1 = np.ceil((gy - extents[:, 1]) / r) + pad
    rects = np.column_stack([col0, row0, col1, row1])
    rects[~np.isfinite(rects).all(axis=1)] = 0
    return rects.astype(np.int64)


def merge_rects(
    rects: np.ndarray, max_side: int
) -> tuple[np.ndarray, list[list[int]]]:
    """Merge overlapping rectangles wherever the merged one is no larger.

    A row of terraced houses has padded rectangles overlapping along every
    party wall, and the overlap would be fetched and stored twice. Two are
    merged only when their joint bounding box holds no more pixels than the
    two did apart, so a merge never costs space, and only up to `max_side`,
    so the result still fits a page. Returns the rectangles and, for each,
    the indices of the input rectangles in it.
    """
    work = np.asarray(rects, dtype=np.int64).copy()
    members: list[list[int]] = [[i] for i in range(len(work))]
    alive = np.ones(len(work), dtype=bool)

    def area(r):
        return (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])

    changed = True
    while changed:
        changed = False
        for i in range(len(work)):
            if not alive[i]:
                continue
            while True:
                a = work[i]
                overlapping = (
                    alive
                    & (work[:, 0] < a[2]) & (a[0] < work[:, 2])
                    & (work[:, 1] < a[3]) & (a[1] < work[:, 3])
                )
                overlapping[i] = False
                candidates = np.flatnonzero(overlapping)
                if not len(candidates):
                    break
                union = np.column_stack([
                    np.minimum(work[candidates, 0], a[0]),
                    np.minimum(work[candidates, 1], a[1]),
                    np.maximum(work[candidates, 2], a[2]),
                    np.maximum(work[candidates, 3], a[3]),
                ])
                cost = area(union) - area(work[candidates]) - area(a)
                fits = (
                    (cost <= 0)
                    & (union[:, 2] - union[:, 0] <= max_side)
                    & (union[:, 3] - union[:, 1] <= max_side)
                )
                if not fits.any():
                    break
                # The cheapest merge first, so the result does not depend on
                # the order the buildings arrived in more than it has to.
                pick = int(np.flatnonzero(fits)[np.argmin(cost[fits])])
                j = int(candidates[pick])
                work[i] = union[pick]
                members[i].extend(members[j])
                members[j] = []
                alive[j] = False
                changed = True
    keep = np.flatnonzero(alive)
    return work[keep], [sorted(members[i]) for i in keep]


def _skyline(
    sizes: np.ndarray, order: list[int], width: int, height: int
) -> dict[int, tuple[int, int]]:
    """Bottom-left skyline packing of `order` into one `width` by `height`.

    Returns ``{index: (x, y)}`` for each rectangle that fit; y runs down from
    the top of the page, as image rows do.
    """
    skyline = [[0, 0, width]]  # [x, top of what is stacked there, width]
    placed: dict[int, tuple[int, int]] = {}
    for index in order:
        w, h = int(sizes[index][0]), int(sizes[index][1])
        best = None
        for start in range(len(skyline)):
            x = skyline[start][0]
            if x + w > width:
                break
            y, covered, k = 0, 0, start
            while covered < w:
                y = max(y, skyline[k][1])
                covered += skyline[k][2]
                k += 1
            if y + h > height:
                continue
            if best is None or (y, x) < best:
                best = (y, x)
        if best is None:
            continue
        y, x = best
        placed[index] = (x, y)
        # Raise the skyline under the new rectangle.
        updated = []
        for sx, sy, sw in skyline:
            end = sx + sw
            if end <= x or sx >= x + w:
                updated.append([sx, sy, sw])
                continue
            if sx < x:
                updated.append([sx, sy, x - sx])
            if end > x + w:
                updated.append([x + w, sy, end - x - w])
        updated.append([x, y + h, w])
        updated.sort()
        skyline = []
        for segment in updated:
            if skyline and skyline[-1][1] == segment[1]:
                skyline[-1][2] += segment[2]
            else:
                skyline.append(segment)
    return placed


def pack_rects(
    sizes: np.ndarray, page_px: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Pack ``(width, height)`` rectangles into as few pages as they need.

    Returns ``(placement, pages)``: per rectangle its ``(page, x, y)``, and
    per page its ``(width, height)``, each a multiple of `PAGE_ALIGN` and at
    most `page_px`. One page is sized to its content rather than to
    `page_px`, so a small area does not carry a mostly empty 8192 px texture.
    """
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1, 2)
    count = len(sizes)
    placement = np.full((count, 3), -1, dtype=np.int64)
    if not count:
        return placement, []
    if (sizes > page_px).any():
        raise ValueError("a rectangle is larger than a page")
    order = sorted(range(count), key=lambda i: (-sizes[i][1], -sizes[i][0], i))

    def tight(indices: list[int]) -> tuple[int, dict] | None:
        """One page just big enough for these, if one page can hold them."""
        area = float((sizes[indices, 0] * sizes[indices, 1]).sum())
        widest = int(sizes[indices, 0].max())
        width = min(page_px, max(widest, _ceil_to(math.sqrt(area / PACK_FILL), PAGE_ALIGN)))
        while True:
            placed = _skyline(sizes, indices, width, page_px)
            if len(placed) == len(indices):
                return width, placed
            if width >= page_px:
                return None
            width = min(page_px, _ceil_to(width * 1.15, PAGE_ALIGN))

    groups: list[dict[int, tuple[int, int]]] = []
    single = tight(order)
    if single is not None:
        groups.append(single[1])
    else:
        remaining = order
        while remaining:
            placed = _skyline(sizes, remaining, page_px, page_px)
            if not placed:  # cannot happen below the size check; never spin
                raise ValueError("a rectangle does not fit an empty page")
            groups.append(placed)
            remaining = [i for i in remaining if i not in placed]
        # The last page is packed again, tight, so it is not full size for
        # the few rectangles left over.
        last = [i for i in order if i in groups[-1]]
        repacked = tight(last)
        if repacked is not None:
            groups[-1] = repacked[1]

    pages = []
    for page, placed in enumerate(groups):
        right = bottom = 0
        for index, (x, y) in placed.items():
            placement[index] = (page, x, y)
            right = max(right, x + int(sizes[index][0]))
            bottom = max(bottom, y + int(sizes[index][1]))
        pages.append(
            (min(page_px, _ceil_to(right, PAGE_ALIGN)),
             min(page_px, _ceil_to(bottom, PAGE_ALIGN)))
        )
    return placement, pages


@dataclass
class RoofPlan:
    """Where every roof's pixels come from and where they go.

    Pure geometry, no pixels: the same plan drives the fetch, the UVs and
    the checks.
    """

    origin: tuple[float, float]
    metres_per_px: float
    requested_m_per_px: float
    padding_px: int
    # Per rectangle: grid (col0, row0, col1, row1), end-exclusive.
    rects: np.ndarray
    # Per rectangle: the buildings whose roofs it holds.
    members: list[list[int]]
    # Per rectangle: (page, x, y) of its top-left pixel.
    placement: np.ndarray
    # Per page: (width, height).
    pages: list[tuple[int, int]]
    # Per building: its rectangle, or -1 when its roof stays on the aerial.
    building_rect: np.ndarray
    budget_px: int
    page_px: int
    capped: bool = False
    # Buildings left on the aerial because their roof would not fit a page.
    dropped: list[int] = field(default_factory=list)

    @property
    def total_px(self) -> int:
        return int(sum(w * h for w, h in self.pages))

    @property
    def roof_px(self) -> int:
        """Pixels inside the rectangles: the share of the pages in use."""
        r = self.rects
        return int(((r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])).sum())

    def size(self, k: int) -> tuple[int, int]:
        c0, r0, c1, r1 = (int(v) for v in self.rects[k])
        return c1 - c0, r1 - r0

    def window(self, c0: int, r0: int, c1: int, r1: int) -> BBox:
        """The RD box a span of grid pixels covers."""
        gx, gy = self.origin
        r = self.metres_per_px
        return BBox(gx + c0 * r, gy - r1 * r, gx + c1 * r, gy - r0 * r)

    def pixel_centres(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """RD x of each column and y of each row of rectangle `k`."""
        gx, gy = self.origin
        r = self.metres_per_px
        c0, r0, c1, r1 = (int(v) for v in self.rects[k])
        return (
            gx + (np.arange(c0, c1) + 0.5) * r,
            gy - (np.arange(r0, r1) + 0.5) * r,
        )

    def uv(self, points_xy: np.ndarray, k: int) -> np.ndarray:
        """UVs of RD points inside rectangle `k`, on its page.

        Blender's convention, as `planar_uv` has it: v = 0 is the bottom row.
        """
        gx, gy = self.origin
        r = self.metres_per_px
        page, x, y = (int(v) for v in self.placement[k])
        width, height = self.pages[page]
        c0, r0 = int(self.rects[k][0]), int(self.rects[k][1])
        points = np.asarray(points_xy, dtype=np.float64)
        px = x + (points[:, 0] - gx) / r - c0
        py = y + (gy - points[:, 1]) / r - r0
        return np.column_stack([px / width, 1.0 - py / height])

    def summary(self) -> dict:
        return {
            "resolution_m_per_px": round(self.metres_per_px, 4),
            "requested_m_per_px": round(self.requested_m_per_px, 4),
            "capped_by_budget": bool(self.capped),
            "budget_megapixels": round(self.budget_px / 1e6, 1),
            "pages": [list(p) for p in self.pages],
            "megapixels": round(self.total_px / 1e6, 2),
            "roof_share_of_pages": round(self.roof_px / max(self.total_px, 1), 3),
            "rectangles": int(len(self.rects)),
            "buildings": int((self.building_rect >= 0).sum()),
            "padding_px": int(self.padding_px),
            **({"left_on_the_aerial": len(self.dropped)} if self.dropped else {}),
        }


def plan_roof_atlas(
    extents: np.ndarray,
    origin: tuple[float, float],
    *,
    metres_per_px: float,
    ground_m_per_px: float,
    budget_px: int,
    page_px: int = 8192,
    padding_px: int = 4,
) -> RoofPlan | None:
    """Rectangles, packed, at the finest resolution the budget allows.

    None when the roofs would be no sharper than the ground: past that point
    an atlas only costs, and the roofs are better left on the aerial.
    """
    requested = float(metres_per_px)
    resolution = max(requested, SOURCE_M_PER_PX)
    valid = np.flatnonzero(np.isfinite(extents).all(axis=1))
    if not len(valid):
        return None
    capped = False
    plan = None
    for _ in range(12):
        if resolution >= ground_m_per_px * 0.999:
            break
        rects = grid_rects(extents[valid], origin, resolution, padding_px)
        sides = np.maximum(rects[:, 2] - rects[:, 0], rects[:, 3] - rects[:, 1])
        fits = sides <= page_px
        merged, members = merge_rects(rects[fits], page_px)
        kept = valid[fits]
        members = [[int(kept[m]) for m in group] for group in members]
        sizes = np.column_stack([merged[:, 2] - merged[:, 0], merged[:, 3] - merged[:, 1]])
        placement, pages = pack_rects(sizes, page_px)
        building_rect = np.full(len(extents), -1, dtype=np.int64)
        for k, group in enumerate(members):
            building_rect[group] = k
        plan = RoofPlan(
            origin=(float(origin[0]), float(origin[1])),
            metres_per_px=resolution,
            requested_m_per_px=requested,
            padding_px=int(padding_px),
            rects=merged,
            members=members,
            placement=placement,
            pages=pages,
            building_rect=building_rect,
            budget_px=int(budget_px),
            page_px=int(page_px),
            capped=capped,
            dropped=[int(i) for i in valid[~fits]],
        )
        if plan.total_px <= budget_px:
            return plan
        # Pixels go with the square of the resolution. A little extra, and
        # whole millimetres, so the loop settles in a step or two and every
        # request bounds is exact at the 0.1 mm the WMS is sent.
        grow = math.sqrt(plan.total_px / budget_px) * 1.01
        resolution = math.ceil(resolution * grow * 1000.0 - 1e-6) / 1000.0
        capped = True
    return None


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def fetch_windows(plan: RoofPlan, max_request_px: int) -> list[tuple[int, int, int, int]]:
    """The GetMap requests that cover every rectangle, as grid spans.

    The grid is cut into tiles of `max_request_px`, the most one request may
    ask for, and each tile any rectangle reaches becomes one request for the
    part of it the rectangles need. In a city centre that is the whole tile;
    among detached houses it is a fraction of it.
    """
    step = int(max_request_px)
    tiles: dict[tuple[int, int], list[int]] = {}
    for c0, r0, c1, r1 in plan.rects.tolist():
        for tile_row in range(r0 // step, (r1 - 1) // step + 1):
            for tile_col in range(c0 // step, (c1 - 1) // step + 1):
                part = [
                    max(c0, tile_col * step), max(r0, tile_row * step),
                    min(c1, (tile_col + 1) * step), min(r1, (tile_row + 1) * step),
                ]
                have = tiles.get((tile_row, tile_col))
                tiles[(tile_row, tile_col)] = part if have is None else [
                    min(have[0], part[0]), min(have[1], part[1]),
                    max(have[2], part[2]), max(have[3], part[3]),
                ]
    return [tuple(tiles[key]) for key in sorted(tiles)]


def assemble_pages(
    plan: RoofPlan,
    windows: list[tuple[int, int, int, int]],
    fetched,
) -> list[np.ndarray]:
    """Copy fetched windows into the pages, rectangle by rectangle.

    `fetched` yields ``(window index, pixels)`` in any order, pixels being
    ``(rows, cols, 3)`` uint8 rows north first. Whatever no rectangle covers
    is filled with the mean roof colour, so a mipmap blending a rectangle
    with its surroundings blends towards a roof tone rather than black.
    """
    pages = [np.zeros((h, w, 3), dtype=np.uint8) for w, h in plan.pages]
    used = [np.zeros((h, w), dtype=bool) for w, h in plan.pages]
    rects = plan.rects
    for k in range(len(rects)):
        page, x, y = (int(v) for v in plan.placement[k])
        w, h = plan.size(k)
        used[page][y : y + h, x : x + w] = True

    for index, pixels in fetched:
        c0, r0, c1, r1 = windows[index]
        if pixels.shape[:2] != (r1 - r0, c1 - c0):
            raise ValueError(
                f"window {index} came back {pixels.shape[1]}x{pixels.shape[0]}, "
                f"not {c1 - c0}x{r1 - r0}"
            )
        hits = np.flatnonzero(
            (rects[:, 0] < c1) & (c0 < rects[:, 2])
            & (rects[:, 1] < r1) & (r0 < rects[:, 3])
        )
        for k in hits:
            rc0, rr0, rc1, rr1 = (int(v) for v in rects[k])
            ic0, ir0 = max(c0, rc0), max(r0, rr0)
            ic1, ir1 = min(c1, rc1), min(r1, rr1)
            page, x, y = (int(v) for v in plan.placement[k])
            pages[page][
                y + ir0 - rr0 : y + ir1 - rr0, x + ic0 - rc0 : x + ic1 - rc0
            ] = pixels[ir0 - r0 : ir1 - r0, ic0 - c0 : ic1 - c0]

    for canvas, mask in zip(pages, used):
        if mask.any() and not mask.all():
            canvas[~mask] = canvas[mask][:: max(1, int(mask.sum()) // 200_000)].mean(
                axis=0
            ).astype(np.uint8)
    return pages


def _service_fetcher(aerial_cfg: dict, layer_name: str):
    """GetMap from the WMS, or WMTS tiles for a window the WMS refuses.

    Returns the fetch function and a dict it records which service answered
    in. Each worker thread keeps its own session.
    """
    import requests

    from .imagery import _fetch_capabilities, wms_image, wmts_image

    local = threading.local()
    lock = threading.Lock()
    record = {"wms": 0, "wmts": 0}
    capabilities: list[str] = []
    timeout = float(aerial_cfg.get("timeout_s", 180))
    retries = int(aerial_cfg.get("max_retries", 4))

    def session():
        if getattr(local, "session", None) is None:
            local.session = requests.Session()
        return local.session

    def fetch(window: BBox, width: int, height: int) -> np.ndarray:
        try:
            image = wms_image(
                window, width, height,
                layer_name=layer_name,
                wms_url=str(aerial_cfg["wms_url"]),
                request_format=str(aerial_cfg.get("request_format", "image/jpeg")),
                timeout=timeout,
                max_retries=retries,
                session=session(),
                description=f"WMS GetMap roofs {width}x{height}",
            )
            used = "wms"
        except Exception as exc:  # noqa: BLE001 - exactly what WMTS is for
            LOG.warning("roof window refused by the WMS (%s); trying WMTS", exc)
            with lock:
                if not capabilities:
                    capabilities.append(
                        _fetch_capabilities(str(aerial_cfg["wmts_url"]), "WMTS", timeout)
                    )
            image = wmts_image(
                window, width, height,
                layer_name=layer_name,
                wmts_url=str(aerial_cfg["wmts_url"]),
                timeout=timeout,
                max_retries=retries,
                capabilities=capabilities[0],
                session=session(),
            )
            used = "wmts"
        with lock:
            record[used] += 1
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    return fetch, record


# --------------------------------------------------------------------------
# The atlas
# --------------------------------------------------------------------------


@dataclass
class RoofAtlas:
    """The written pages, and each roof triangle's place on them."""

    plan: RoofPlan | None
    paths: list[Path] = field(default_factory=list)
    # Per roof triangle, in buildings.npz order: three UVs, and the page, or
    # -1 where the roof stays on the aerial.
    roof_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 2), np.float32))
    roof_page: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int16))
    roof_rect: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    fetch: dict = field(default_factory=dict)
    # Why there is no atlas, when there is none.
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.paths) and self.plan is not None

    @property
    def files(self) -> list[str]:
        return [path.name for path in self.paths]

    def map_rects(self, fn) -> None:
        """Rewrite every rectangle as ``fn(pixels, xs, ys)``, in place.

        `xs` and `ys` are the RD centres of the rectangle's columns and rows,
        which is all a pass over the ground needs to do the same to a roof.
        """
        from PIL import Image

        from .imagery import _allow_large_images

        if not self.ok:
            return
        _allow_large_images()
        plan = self.plan
        for page, path in enumerate(self.paths):
            with Image.open(path) as opened:
                pixels = np.array(opened.convert("RGB"), dtype=np.uint8)
            for k in np.flatnonzero(plan.placement[:, 0] == page):
                _, x, y = (int(v) for v in plan.placement[k])
                w, h = plan.size(int(k))
                xs, ys = plan.pixel_centres(int(k))
                pixels[y : y + h, x : x + w] = fn(pixels[y : y + h, x : x + w], xs, ys)
            Image.fromarray(pixels, mode="RGB").save(path)

    def save(self, work_dir: Path) -> Path:
        """The UVs for the Blender stage, beside buildings.npz."""
        path = Path(work_dir) / UV_FILE
        np.savez_compressed(
            path,
            roof_uv=self.roof_uv.reshape(-1, 2).astype(np.float32),
            roof_page=self.roof_page.astype(np.int16),
            roof_rect=self.roof_rect.astype(np.int32),
            files=np.array(self.files, dtype="U64"),
        )
        return path

    def to_dict(self) -> dict:
        if not self.ok:
            return {"note": self.note} if self.note else {}
        return {
            "files": self.files,
            **self.plan.summary(),
            "roof_triangles": int((self.roof_page >= 0).sum()),
            "roof_triangles_on_the_aerial": int((self.roof_page < 0).sum()),
            "fetch": dict(self.fetch),
        }


def roof_uvs(roofs: list[np.ndarray], plan: RoofPlan):
    """Per roof triangle: UVs, page and rectangle, in buildings.npz order."""
    uv_chunks, page_chunks, rect_chunks = [], [], []
    for index, tris in enumerate(roofs):
        if not len(tris):
            continue
        k = int(plan.building_rect[index])
        n = len(tris)
        if k < 0:
            uv_chunks.append(np.zeros((n, 3, 2), dtype=np.float32))
            page_chunks.append(np.full(n, -1, dtype=np.int16))
            rect_chunks.append(np.full(n, -1, dtype=np.int32))
            continue
        uv = plan.uv(np.asarray(tris, dtype=np.float64).reshape(-1, 3)[:, :2], k)
        uv_chunks.append(uv.reshape(n, 3, 2).astype(np.float32))
        page_chunks.append(np.full(n, int(plan.placement[k][0]), dtype=np.int16))
        rect_chunks.append(np.full(n, k, dtype=np.int32))
    if not uv_chunks:
        return (np.zeros((0, 3, 2), np.float32), np.zeros(0, np.int16),
                np.zeros(0, np.int32))
    return (np.concatenate(uv_chunks), np.concatenate(page_chunks),
            np.concatenate(rect_chunks))


def roof_settings(buildings_cfg: dict) -> dict:
    """The roof atlas settings, read once: resolution in metres, or 0 off."""
    cm = float(buildings_cfg.get("roof_cm") or 0.0)
    return {
        "metres_per_px": cm / 100.0 if cm > 0 else 0.0,
        "budget_px": int(float(buildings_cfg.get("roof_max_mpx", 48)) * 1e6),
        "page_px": int(buildings_cfg.get("roof_atlas_px", 8192)),
        "padding_px": int(buildings_cfg.get("roof_padding_px", 4)),
    }


def build_roof_atlas(
    bbox: BBox,
    work_dir: Path,
    *,
    roofs: list[np.ndarray],
    buildings_cfg: dict,
    aerial_cfg: dict,
    ground_m_per_px: float,
    layer_name: str,
    fetch=None,
) -> RoofAtlas:
    """Plan, fetch and write the roof atlas for `roofs`, one array per building.

    `fetch(window, width, height)` returns the window's pixels; the default
    asks the PDOK WMS, falling back to WMTS. Never raises for a service that
    does not answer: the roofs stay on the aerial, as they always were, and
    the note says why.
    """
    from PIL import Image

    from .imagery import _allow_large_images

    settings = roof_settings(buildings_cfg)
    started = time.time()
    if settings["metres_per_px"] <= 0:
        return RoofAtlas(None, note="switched off (buildings.roof_cm is 0)")

    extents = roof_extents(roofs)
    plan = plan_roof_atlas(
        extents,
        (bbox.xmin, bbox.ymax),
        metres_per_px=settings["metres_per_px"],
        ground_m_per_px=ground_m_per_px,
        budget_px=settings["budget_px"],
        page_px=settings["page_px"],
        padding_px=settings["padding_px"],
    )
    if plan is None:
        note = (
            f"no roofs to texture, or none sharper than the ground's "
            f"{ground_m_per_px * 100:.0f} cm/px within the "
            f"{settings['budget_px'] / 1e6:.0f} Mpx budget"
        )
        LOG.warning("roof atlas: %s; roofs stay on the aerial", note)
        return RoofAtlas(None, note=note)
    if plan.capped:
        LOG.warning(
            "roof atlas: %.1f cm/px asked, but the roofs of this area would "
            "need more than the %.0f Mpx budget there (buildings.roof_max_mpx); "
            "using %.1f cm/px",
            plan.requested_m_per_px * 100, plan.budget_px / 1e6,
            plan.metres_per_px * 100,
        )
    if plan.dropped:
        LOG.warning(
            "roof atlas: %d roofs are larger than a %d px page at %.1f cm/px "
            "and stay on the aerial",
            len(plan.dropped), plan.page_px, plan.metres_per_px * 100,
        )

    windows = fetch_windows(plan, int(aerial_cfg.get("max_request_px", 2000)))
    megapixels = sum((c1 - c0) * (r1 - r0) for c0, r0, c1, r1 in windows) / 1e6
    LOG.info(
        "roof atlas: %d roofs in %d rectangles on %d page(s) (%s) at %.1f cm/px, "
        "%.1f Mpx; fetching %.1f Mpx in %d requests",
        int((plan.building_rect >= 0).sum()), len(plan.rects), len(plan.pages),
        ", ".join(f"{w}x{h}" for w, h in plan.pages),
        plan.metres_per_px * 100, plan.total_px / 1e6, megapixels, len(windows),
    )

    record: dict = {}
    if fetch is None:
        fetch, record = _service_fetcher(aerial_cfg, layer_name)

    def one(index: int):
        c0, r0, c1, r1 = windows[index]
        return index, fetch(plan.window(c0, r0, c1, r1), c1 - c0, r1 - r0)

    fetched_at = time.time()
    try:
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            results = pool.map(one, range(len(windows)))
            pages = assemble_pages(plan, windows, results)
    except Exception as exc:  # noqa: BLE001 - the aerial is still there
        note = f"the roof imagery could not be fetched ({exc})"
        LOG.warning("roof atlas: %s; roofs stay on the aerial", note)
        return RoofAtlas(None, note=note)
    fetch_seconds = time.time() - fetched_at

    _allow_large_images()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for page, canvas in enumerate(pages):
        path = work_dir / f"{FILE_STEM}_{page:02d}.png"
        # Quick to write, because the relighting stage normally writes it
        # again: measured on a 29 Mpx page, 2.5 s at level 1 against 7.8 s
        # at the default 6, for 6% more bytes. The page that ships is the
        # relit one, at the default level, as the aerial is.
        Image.fromarray(canvas, mode="RGB").save(path, compress_level=1)
        paths.append(path)
    for stale in sorted(work_dir.glob(f"{FILE_STEM}_*.png"))[len(paths):]:
        stale.unlink()

    uv, page_of, rect_of = roof_uvs(roofs, plan)
    atlas = RoofAtlas(
        plan,
        paths=paths,
        roof_uv=uv,
        roof_page=page_of,
        roof_rect=rect_of,
        fetch={
            "requests": len(windows),
            "megapixels": round(megapixels, 2),
            "seconds": round(fetch_seconds, 1),
            "total_seconds": round(time.time() - started, 1),
            **({"source": "+".join(k for k, v in record.items() if v)} if record else {}),
        },
    )
    atlas.save(work_dir)
    LOG.info(
        "roof atlas: fetched in %.1fs, written as %s (%.1f MB)",
        fetch_seconds, ", ".join(atlas.files),
        sum(p.stat().st_size for p in paths) / 1e6,
    )
    return atlas


__all__ = [
    "FETCH_WORKERS",
    "RoofAtlas",
    "RoofPlan",
    "SOURCE_M_PER_PX",
    "assemble_pages",
    "build_roof_atlas",
    "fetch_windows",
    "grid_rects",
    "merge_rects",
    "pack_rects",
    "plan_roof_atlas",
    "roof_extents",
    "roof_settings",
    "roof_uvs",
]
