"""Straightened pictures of walls, taken from the street photos.

Both uses of a photo start here. Measuring a facade needs its windows on a
square grid, where a storey is a row and a bay is a column; painting a wall
needs its pixels laid out the way the wall's texture coordinates run. Both
are the same picture: the wall seen square-on, in its own coordinates, with
metres along the wall left to right as seen from outside and metres up.

Each pixel of that picture is a point on the wall, and its colour is read
from a photo by projecting the point through the photo's checked camera. The
pixel also carries a weight, how much that reading is worth:

* nothing where the photo does not see the point: outside the frame, in the
  band of a 360 photo that shows the person carrying it, or behind another
  building;
* little where a tree crown stands between the camera and the wall, which
  over a Dutch street is most of the ground floor half the year;
* less where the photo sees the wall obliquely or from far off, in
  proportion to the pixels per metre it can give.

Several views of one wall are then fused. A parked car, a cyclist or a
delivery van is in front of the wall in one photo and gone in the next, so
per pixel the readings that agree with the others win, and the odd one out
is dropped. The exposure of each photo is matched to the best one first: a
photo taken in sun and one in shade otherwise disagree everywhere.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .camera import SPHERICAL, project_world
from .photos import Photo

LOG = logging.getLogger(__name__)

# Occlusion and the face's outline are worked out on this coarser grid and
# stretched over the fine one. A wall seen through a gap is not decided to
# the centimetre anyway.
COARSE_M = 0.4
# Below this elevation a 360 photo shows its own mount.
PANO_FLOOR_DEG = -35.0
# How much of a reading survives a tree crown in the way. Not zero: in winter
# a crown is mostly air, and a wall seen only through branches is still
# better measured than not at all.
TREE_WEIGHT = 0.15
# A reading this far from the per-pixel consensus, in relative luminance, is
# taken to be something standing in front of the wall.
OUTLIER_LUMINANCE = 0.22
# A ray to the foot of a wall that crosses a parking bay below this height
# above the street most likely ends on a parked car. The BGT draws every
# parking bay, so the cars that hide the ground storey of half the walls in
# a Dutch street are known about before any photo is looked at.
PARKED_CAR_M = 1.8
PARKED_CAR_WEIGHT = 0.2


@dataclass
class WallImage:
    """One wall face, seen square-on."""

    face: int
    rgb: np.ndarray  # (h, w, 3) float32 0..255, row 0 at the top of the wall
    weight: np.ndarray  # (h, w) float32 0..1
    m_per_px: float
    width_m: float
    z_min: float
    z_max: float
    views: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of the wall seen with at least a usable weight."""
        return float((self.weight > 0.2).mean()) if self.weight.size else 0.0

    def z_of_row(self, row) -> np.ndarray:
        return self.z_max - (np.asarray(row) + 0.5) * self.m_per_px

    def row_of_z(self, z) -> np.ndarray:
        return (self.z_max - np.asarray(z)) / self.m_per_px - 0.5


@dataclass
class Crown:
    """A tree crown as a sphere, enough to know what it hides."""

    x: float
    y: float
    z: float
    radius: float


def parking_lookup(class_grid: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """``parking(x, y)``: whether RD points lie on a BGT parking bay.

    `class_grid` is the land-cover classes on the terrain grid, row 0 at the
    south edge, as the ground-surface stage leaves it.
    """
    from .surfaces import CLASS_PARKING

    grid = np.asarray(class_grid) == CLASS_PARKING
    x0, y0 = float(xs[0]), float(ys[0])
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    rows, cols = grid.shape

    def parking(x, y):
        c = np.rint((np.asarray(x) - x0) / dx).astype(np.int64)
        r = np.rint((np.asarray(y) - y0) / dy).astype(np.int64)
        inside = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
        out = np.zeros(np.shape(c), dtype=bool)
        out[inside] = grid[r[inside], c[inside]]
        return out

    return parking


def crowns_from_trees(trees) -> list[Crown]:
    """Tree crowns from the pipeline's trees, centred halfway up the crown."""
    out = []
    for tree in getattr(trees, "trees", []) or []:
        top = tree.ground_z_nap + tree.height_m
        base = tree.ground_z_nap + min(tree.trunk_height_m, tree.height_m * 0.6)
        radius = max(0.5 * (top - base), float(tree.crown_radius_m), 0.5)
        out.append(Crown(tree.x, tree.y, 0.5 * (top + base), radius))
    return out


class PhotoPixels:
    """Photo pixels on demand, a few kept in memory.

    A 360 photo at full size is 90 MB as an array, so they are read when
    first asked for and dropped when the cache is full.
    """

    def __init__(self, max_cached: int = 6) -> None:
        self.max_cached = max_cached
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str) -> np.ndarray:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            pixels = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        self._cache[path] = pixels
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return pixels


def _bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray, wrap: bool) -> np.ndarray:
    """Sample (h, w, 3) at fractional pixel positions; u across, v down."""
    h, w = image.shape[:2]
    x = u - 0.5
    y = np.clip(v - 0.5, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    if wrap:
        x0 %= w
        x1 = (x0 + 1) % w
    else:
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    img = image.astype(np.float32, copy=False)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def _upsample(coarse: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour stretch of a coarse grid onto a fine one."""
    rows = np.minimum((np.arange(shape[0]) * coarse.shape[0]) // shape[0], coarse.shape[0] - 1)
    cols = np.minimum((np.arange(shape[1]) * coarse.shape[1]) // shape[1], coarse.shape[1] - 1)
    return coarse[np.ix_(rows, cols)]


def face_grid(face, m_per_px: float):
    """Pixel centres of a face's straightened picture, as world points.

    Returns ``(points (h, w, 3), u (w,), z (h,))``. Row 0 is the top of the
    wall, column 0 its left end seen from outside.
    """
    width = max(1, int(np.ceil(face.width_m / m_per_px)))
    height = max(1, int(np.ceil(face.height_m / m_per_px)))
    u = (np.arange(width) + 0.5) * (face.width_m / width)
    z = face.z_max - (np.arange(height) + 0.5) * (face.height_m / height)
    tangent = (face.end - face.start) / max(face.width_m, 1e-9)
    xy = face.start[None, :] + u[:, None] * tangent[None, :]
    # A couple of centimetres proud of the wall, so the wall's own plane never
    # counts as standing in front of it.
    xy = xy + 0.02 * face.normal[None, :]
    points = np.empty((height, width, 3))
    points[:, :, 0] = xy[None, :, 0]
    points[:, :, 1] = xy[None, :, 1]
    points[:, :, 2] = z[:, None]
    return points, u, z


def face_outline(face, shape: tuple[int, int]) -> np.ndarray:
    """Which pixels of the straightened picture are wall, not the sky beside
    a gable: a gable end is a pentagon, not the rectangle around it."""
    from .photo_coverage import _inside_triangles_uz

    tangent = (face.end - face.start) / max(face.width_m, 1e-9)
    tris = face.triangles
    tris_uz = np.stack(
        [(tris[:, :, :2] - face.start[None, None, :]) @ tangent, tris[:, :, 2]], axis=-1
    )
    ch = max(2, int(np.ceil(face.height_m / COARSE_M)))
    cw = max(2, int(np.ceil(face.width_m / COARSE_M)))
    cu = (np.arange(cw) + 0.5) * face.width_m / cw
    cz = face.z_max - (np.arange(ch) + 0.5) * face.height_m / ch
    grid = np.column_stack([np.repeat(cu[None, :], ch, 0).ravel(), np.repeat(cz[:, None], cw, 1).ravel()])
    # A margin of half a coarse cell, so the wall's own edge is not clipped.
    inside = _inside_triangles_uz(grid, tris_uz)
    for du, dz in ((COARSE_M / 2, 0), (-COARSE_M / 2, 0), (0, COARSE_M / 2), (0, -COARSE_M / 2)):
        inside |= _inside_triangles_uz(grid + [du, dz], tris_uz)
    return _upsample(inside.reshape(ch, cw), shape)


def _segment_hits_spheres(origin: np.ndarray, targets: np.ndarray, crowns: list[Crown]) -> np.ndarray:
    """Does the segment from `origin` to each target pass through a crown?"""
    hit = np.zeros(len(targets), dtype=bool)
    if not crowns:
        return hit
    centres = np.array([[c.x, c.y, c.z] for c in crowns])
    radii = np.array([c.radius for c in crowns])
    d = targets - origin  # (n, 3)
    length2 = np.maximum((d * d).sum(axis=1), 1e-9)
    # Only crowns near the line of sight matter; the rest are skipped cheaply.
    reach = np.sqrt(length2.max()) + radii.max()
    near = np.linalg.norm(centres[:, :2] - origin[:2], axis=1) < reach
    for centre, radius in zip(centres[near], radii[near]):
        t = np.clip(((centre - origin) @ d.T) / length2, 0.0, 1.0)
        closest = origin + t[:, None] * d
        # Not the last stretch before the wall: a crown touching the facade is
        # in front of almost nothing a photo of the facade shows.
        hit |= (np.linalg.norm(closest - centre, axis=1) < radius) & (t < 0.97)
    return hit


def _behind_parked_cars(origin, targets, parking, street_z: float) -> np.ndarray:
    """Does the ray to each target cross a parking bay low enough to hit a car?

    Walked back from the wall in half-metre steps over the last twelve
    metres, which is as far as a row of parked cars can be from a facade.
    """
    hit = np.zeros(len(targets), dtype=bool)
    low = targets[:, 2] < street_z + PARKED_CAR_M
    if not low.any():
        return hit
    d = origin - targets[low]
    length = np.maximum(np.linalg.norm(d[:, :2], axis=1), 1e-6)
    for step in np.arange(0.5, 12.0, 0.5):
        t = np.minimum(step / length, 1.0)
        at = targets[low] + t[:, None] * d
        crosses = parking(at[:, 0], at[:, 1]) & (at[:, 2] < street_z + PARKED_CAR_M) & (t < 1.0)
        hit[np.flatnonzero(low)[crosses]] = True
    return hit


def rectify_view(
    face,
    face_index: int,
    photo: Photo,
    image: np.ndarray,
    *,
    m_per_px: float,
    faces_index=None,
    crowns: list[Crown] | None = None,
    target_px_per_m: float | None = None,
    parking=None,
    street_z: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One photo's view of one wall, straightened. Returns ``(rgb, weight)``."""
    points, u, z = face_grid(face, m_per_px)
    h, w = points.shape[:2]
    camera = photo.camera.scaled(image.shape[1], image.shape[0])
    pixels, ok = project_world(camera, photo.rotation, photo.centre, points.reshape(-1, 3))
    weight = ok.reshape(h, w).astype(np.float32)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if not weight.any():
        return rgb, weight
    rgb.reshape(-1, 3)[ok] = _bilinear(
        image, pixels[ok, 0], pixels[ok, 1], wrap=camera.kind == SPHERICAL
    )

    c = photo.centre
    if camera.kind == SPHERICAL:
        horizontal = np.linalg.norm(points[:, :, :2] - c[:2], axis=2)
        elevation = np.degrees(np.arctan2(points[:, :, 2] - c[2], horizontal))
        weight[elevation < PANO_FLOOR_DEG] = 0.0

    weight *= face_outline(face, (h, w))

    # Occlusion and trees, on a coarse grid.
    ch = max(2, int(np.ceil(face.height_m / COARSE_M)))
    cw = max(2, int(np.ceil(face.width_m / COARSE_M)))
    coarse_points = points[
        np.minimum((np.arange(ch) * h) // ch + h // (2 * ch), h - 1)
    ][:, np.minimum((np.arange(cw) * w) // cw + w // (2 * cw), w - 1)].reshape(-1, 3)
    if faces_index is not None:
        candidates = faces_index.near(c[0], c[1], float(np.linalg.norm(face.centre[:2] - c[:2])) + 5.0)
        blocked = faces_index.blocked(
            c, coarse_points, candidates, np.full(len(coarse_points), face_index)
        )
        weight *= ~_upsample(blocked.reshape(ch, cw), (h, w))
    if crowns:
        behind_tree = _segment_hits_spheres(c, coarse_points, crowns)
        weight *= np.where(_upsample(behind_tree.reshape(ch, cw), (h, w)), TREE_WEIGHT, 1.0)
    if parking is not None and street_z is not None:
        behind_car = _behind_parked_cars(c, coarse_points, parking, street_z)
        weight *= np.where(_upsample(behind_car.reshape(ch, cw), (h, w)), PARKED_CAR_WEIGHT, 1.0)

    # How sharply the photo sees each pixel, against what is asked for.
    if target_px_per_m:
        to_cam = c[None, None, :] - points
        distance = np.maximum(np.linalg.norm(to_cam, axis=2), 0.5)
        normal3 = np.array([face.normal[0], face.normal[1], 0.0])
        cos_inc = np.clip((to_cam @ normal3) / distance, 0.0, 1.0)
        sharp = camera.pixels_per_radian() * cos_inc / distance
        weight *= np.clip(sharp / target_px_per_m, 0.0, 1.0).astype(np.float32)
        weight[cos_inc <= 0.05] = 0.0

    return rgb, weight


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _edges(rgb: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Gradient magnitude, zero where the view has nothing."""
    lum = luminance(rgb)
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    return np.hypot(gx, gy) * (weight > 0.3)


def register_shift(
    reference: tuple[np.ndarray, np.ndarray],
    view: tuple[np.ndarray, np.ndarray],
    max_shift_px: int,
) -> tuple[int, int, float]:
    """How far to move `view` to lie on `reference`, by phase correlation.

    Every photo has been lined up with the model to a pixel or two, but not
    with each other, and two views of the same window a few centimetres
    apart average into a double outline. Correlating their edges in the
    straightened picture finds the residual shift directly, where it is a
    plain translation. Returns ``(rows, cols, peak)``; the peak says how
    clearly the shift stood out, and a weak one is not acted on.
    """
    a = _edges(*reference)
    b = _edges(*view)
    if a.sum() < 1 or b.sum() < 1:
        return 0, 0, 0.0
    h, w = a.shape
    fa = np.fft.rfft2(a - a.mean())
    fb = np.fft.rfft2(b - b.mean())
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-9)
    corr = np.fft.irfft2(cross, s=(h, w))
    # Only small shifts are believable: anything larger is a different
    # window lining up with this one.
    rows = np.r_[0 : max_shift_px + 1, h - max_shift_px : h]
    cols = np.r_[0 : max_shift_px + 1, w - max_shift_px : w]
    window = corr[np.ix_(rows % h, cols % w)]
    r, c = np.unravel_index(int(np.argmax(window)), window.shape)
    peak = float(window[r, c])
    dr = int(rows[r] if rows[r] <= h // 2 else rows[r] - h)
    dc = int(cols[c] if cols[c] <= w // 2 else cols[c] - w)
    return dr, dc, peak


def _shifted(array: np.ndarray, dr: int, dc: int, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(array, fill)
    h, w = array.shape[:2]
    src_r = slice(max(0, -dr), min(h, h - dr))
    dst_r = slice(max(0, dr), min(h, h + dr))
    src_c = slice(max(0, -dc), min(w, w - dc))
    dst_c = slice(max(0, dc), min(w, w + dc))
    out[dst_r, dst_c] = array[src_r, src_c]
    return out


def co_register(
    views: list[tuple[np.ndarray, np.ndarray]], m_per_px: float, max_shift_m: float = 0.35
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shift every view onto the best-covered one."""
    if len(views) < 2:
        return views
    reference = int(np.argmax([float(w.sum()) for _, w in views]))
    max_px = max(1, int(round(max_shift_m / m_per_px)))
    out = []
    for index, (rgb, weight) in enumerate(views):
        if index == reference:
            out.append((rgb, weight))
            continue
        dr, dc, peak = register_shift(views[reference], (rgb, weight), max_px)
        if peak < 0.02 or (dr == 0 and dc == 0):
            out.append((rgb, weight))
            continue
        out.append((_shifted(rgb, dr, dc), _shifted(weight, dr, dc)))
    return out


def match_exposure(views: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """Per-channel gains bringing each view to the best-covered one.

    Measured on the pixels both see well, as a ratio of medians, so a car in
    one view moves the gain hardly at all.
    """
    if not views:
        return []
    reference = int(np.argmax([float(w.sum()) for _, w in views]))
    ref_rgb, ref_w = views[reference]
    gains = []
    for index, (rgb, weight) in enumerate(views):
        if index == reference:
            gains.append(np.ones(3, dtype=np.float32))
            continue
        both = (weight > 0.3) & (ref_w > 0.3)
        if both.sum() < 200:
            gains.append(np.ones(3, dtype=np.float32))
            continue
        ratio = np.median(ref_rgb[both], axis=0) / np.maximum(np.median(rgb[both], axis=0), 1.0)
        gains.append(np.clip(ratio, 0.6, 1.6).astype(np.float32))
    return gains


def _box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1)-square window, by integral image; edges clamp."""
    if radius < 1:
        return array
    padded = np.pad(array, radius, mode="edge")
    table = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    table = np.pad(table, ((1, 0), (1, 0)))
    k = 2 * radius + 1
    total = table[k:, k:] - table[:-k, k:] - table[k:, :-k] + table[:-k, :-k]
    return total / (k * k)


def fuse_views(
    views: list[tuple[np.ndarray, np.ndarray]],
    *,
    mode: str = "blend",
    m_per_px: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """One picture from several, dropping what only one view shows.

    ``blend`` averages the views that agree, which is right for measuring:
    noise averages out and a misregistered window only widens a little.
    ``best`` takes each pixel from the single best view that agrees with the
    others and feathers the seams between views over a few decimetres, which
    is right for a texture: one sharp photo beats the average of three.
    """
    if not views:
        raise ValueError("nothing to fuse")
    if len(views) == 1:
        return views[0]
    views = co_register(views, m_per_px)
    gains = match_exposure(views)
    rgb = np.stack([v[0] * g for v, g in zip(views, gains)])  # (n, h, w, 3)
    weight = np.stack([v[1] for v in views])  # (n, h, w)

    lum = luminance(rgb)
    usable = weight > 0.05
    masked = np.where(usable, lum, np.nan)
    with np.errstate(all="ignore"):
        consensus = np.nanmedian(masked, axis=0)
    seen_twice = usable.sum(axis=0) >= 2
    # Where two or more views see a pixel, the ones far from their median
    # are something in front of the wall in that photo only.
    far = np.abs(lum - consensus[None]) > OUTLIER_LUMINANCE * np.maximum(consensus[None], 20.0)
    trust = np.where(seen_twice[None] & far, 0.05, 1.0) * weight
    if mode == "best":
        # The winner per pixel, then softened into overlapping masks so the
        # change from one photo to the next is a fade, not a cut.
        winner = np.argmax(trust, axis=0)
        radius = max(1, int(round(0.25 / m_per_px)))
        soft = np.stack([
            _box_blur((winner == k).astype(np.float32), radius) for k in range(len(views))
        ])
        trust = soft * (trust > 0.02)
    total = trust.sum(axis=0)
    fused = (rgb * trust[..., None]).sum(axis=0) / np.maximum(total, 1e-6)[..., None]
    # Matching exposures can push a bright view past white; left unclipped,
    # those values wrap to green once the picture is written as bytes.
    fused = np.clip(fused, 0.0, 255.0)
    confidence = np.clip(weight.max(axis=0), 0.0, 1.0)
    confidence = np.where(total > 1e-6, confidence, 0.0)
    return fused.astype(np.float32), confidence.astype(np.float32)


def capture_wall(
    face,
    face_index: int,
    photos: list[Photo],
    pixels: PhotoPixels,
    *,
    m_per_px: float,
    faces_index=None,
    crowns: list[Crown] | None = None,
    target_px_per_m: float | None = None,
    path_of=None,
    mode: str = "blend",
    parking=None,
    street_z: float | None = None,
) -> WallImage | None:
    """A wall face straightened from up to a handful of photos and fused.

    `path_of(photo)` says which file holds a photo's pixels; by default the
    photo's own `path`.
    """
    views, names = [], []
    for photo in photos:
        path = path_of(photo) if path_of else photo.path
        if not path:
            continue
        try:
            image = pixels.get(path)
        except OSError as exc:
            LOG.warning("photo %s could not be read: %s", photo.id, exc)
            continue
        rgb, weight = rectify_view(
            face, face_index, photo, image,
            m_per_px=m_per_px, faces_index=faces_index, crowns=crowns,
            target_px_per_m=target_px_per_m, parking=parking, street_z=street_z,
        )
        if float(weight.mean()) < 0.02:
            continue
        views.append((rgb, weight))
        names.append(photo.id)
    if not views:
        return None
    rgb, weight = fuse_views(views, mode=mode, m_per_px=m_per_px)
    return WallImage(
        face=face_index,
        rgb=rgb,
        weight=weight,
        m_per_px=m_per_px,
        width_m=face.width_m,
        z_min=face.z_min,
        z_max=face.z_max,
        views=names,
    )


def save_wall_image(wall: WallImage, path) -> None:
    """The straightened wall as a picture, unseen parts grey: for people."""
    from PIL import Image

    grey = np.full_like(wall.rgb, 128.0)
    shown = wall.rgb * wall.weight[..., None] + grey * (1 - wall.weight[..., None])
    Image.fromarray(np.clip(shown, 0, 255).astype(np.uint8)).save(path)


__all__ = [
    "Crown",
    "PhotoPixels",
    "WallImage",
    "capture_wall",
    "co_register",
    "register_shift",
    "crowns_from_trees",
    "parking_lookup",
    "face_grid",
    "face_outline",
    "fuse_views",
    "luminance",
    "match_exposure",
    "rectify_view",
    "save_wall_image",
]
