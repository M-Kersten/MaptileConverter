"""Taking the flown sun back out of the aerial photo.

A Dutch ortho is flown leaf-off in late winter, which is what makes it usable
as a ground texture at all. The same decision puts the sun low: measured over
the Bronbeek square kilometre, shadows run about 20 m to the north-east at a
sun elevation near 23 degrees, and 29% of the photo sits below a luminance of
85. Those shadows are painted onto every roof and every square metre of
ground, at one time of one day.

For a model that is going to be lit by an engine, that is wrong in a way no
amount of texture resolution fixes. At noon the scene has long winter shadows
pointing north-east that nothing in it casts, and they fight whatever the
engine's own sun is doing. It also double-counts against the canopy occlusion
baked in beside it.

So the sun is fitted and removed. Three steps, and the middle one is the
reason this can work at all:

1. **Fit the sun.** Not from flight metadata, which the WMS does not publish,
   but from the model: cast shadows across a grid of candidate sun positions
   and keep the one whose shadows best explain the dark pixels.
2. **Cast the shadow from the geometry we already have.** This pipeline knows
   where the buildings and the trees are, in three dimensions. A detector
   working from pixels alone has to guess whether a dark patch is a shadow or
   a dark roof; here the shadow is computed from the thing that casts it.
3. **Relight from the photo's own statistics.** How much darker a shadow is
   depends on the surface, so the gain is measured per land-cover class --
   shadowed grass against lit grass, shadowed asphalt against lit asphalt --
   per colour channel, which picks up the blue cast of skylight for free.

This is a mitigation and not a cure, and the limit is information: where a
shadow is deep there is nothing under it to recover and the gain only
amplifies noise, which is why it is capped.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .geo import BBox

LOG = logging.getLogger(__name__)

# Metres per pixel of the height raster the shadows are cast on. Shadow edges
# do not need the photo's resolution -- a real one is soft anyway, over the
# sun's half-degree plus scatter -- and this keeps a square kilometre to a
# 2000 px grid rather than the 12500 px the 8 cm ortho would want.
SHADOW_PIXEL_M = 0.5

# Metres per pixel while *fitting* the sun, which is coarser again. The fit
# casts a few hundred trial shadows and only has to recover an angle; doing
# that at half a metre over a square kilometre is a few hundred passes over
# four million cells, which turns a ten-second step into a ten-minute one.
# Four metres keeps a terrace of houses as a distinct block, which is all the
# direction needs.
FIT_PIXEL_M = 4.0

# The most a shadowed pixel may be brightened. Deep shade holds no detail to
# recover, so past this the gain is amplifying sensor noise and JPEG blocks.
MAX_GAIN = 3.0

# Sun elevations worth trying. PDOK flies late winter to early spring, which
# at 52 degrees north puts the sun between about 10 and 45 degrees up.
ELEVATION_RANGE = (8.0, 50.0)


def _shift(array: np.ndarray, rows: int, cols: int, fill: float) -> np.ndarray:
    """`out[i, j] = array[i + rows, j + cols]`, padded with `fill`.

    Not `np.roll`: a wrap would let a tower on the north edge shadow the south
    edge of the area.
    """
    out = np.full_like(array, fill)
    src_r0, dst_r0 = (rows, 0) if rows >= 0 else (0, -rows)
    src_c0, dst_c0 = (cols, 0) if cols >= 0 else (0, -cols)
    height = array.shape[0] - abs(rows)
    width = array.shape[1] - abs(cols)
    if height <= 0 or width <= 0:
        return out
    out[dst_r0 : dst_r0 + height, dst_c0 : dst_c0 + width] = array[
        src_r0 : src_r0 + height, src_c0 : src_c0 + width
    ]
    return out


def cast_shadow(
    surface_nap: np.ndarray,
    metres_per_px: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Where the surface cannot see the sun, as a float 0..1 raster.

    Marches toward the sun one pixel at a time, tracking the line of sight as
    it rises. A pixel is shadowed as soon as anything along that line stands
    above it. North is row 0, so moving north means moving to a lower row.

    `azimuth_deg` is where the sun is, clockwise from north: 180 is due south.
    """
    elevation = math.radians(max(elevation_deg, 0.5))
    azimuth = math.radians(azimuth_deg)
    rise_per_px = math.tan(elevation) * metres_per_px

    # One pixel per step along whichever axis moves fastest, so the march
    # never skips a cell.
    d_row = -math.cos(azimuth)
    d_col = math.sin(azimuth)
    lead = max(abs(d_row), abs(d_col))
    if lead < 1e-9:
        return np.zeros_like(surface_nap, dtype=np.float32)
    d_row, d_col = d_row / lead, d_col / lead
    step_m = metres_per_px / lead

    finite = surface_nap[np.isfinite(surface_nap)]
    if not finite.size:
        return np.zeros_like(surface_nap, dtype=np.float32)
    relief = float(finite.max() - finite.min())
    steps = int(min(2048, max(1, math.ceil(relief / max(rise_per_px, 1e-6)))))

    shadowed = np.zeros(surface_nap.shape, dtype=bool)
    for k in range(1, steps + 1):
        sampled = _shift(
            surface_nap, int(round(k * d_row)), int(round(k * d_col)), -np.inf
        )
        shadowed |= sampled > surface_nap + k * step_m * math.tan(elevation)
    return shadowed.astype(np.float32)


def _pool(array: np.ndarray, factor: int, how: str = "max") -> np.ndarray:
    """Shrink by an integer factor, keeping the maxima or the means.

    Heights pool by maximum, so a chimney-width caster is not averaged out of
    existence before it has cast anything. Brightness pools by mean, because
    that is what a coarser photo of the same ground would read.
    """
    if factor <= 1:
        return array
    rows = (array.shape[0] // factor) * factor
    cols = (array.shape[1] // factor) * factor
    if rows < factor or cols < factor:
        return array
    blocks = array[:rows, :cols].reshape(
        rows // factor, factor, cols // factor, factor
    )
    return blocks.max(axis=(1, 3)) if how == "max" else blocks.mean(axis=(1, 3))


def fit_sun(
    surface_nap: np.ndarray,
    luminance: np.ndarray,
    metres_per_px: float,
) -> tuple[float, float, float]:
    """The sun position whose shadows best explain the dark pixels.

    Returns ``(azimuth, elevation, score)``, the score being the correlation
    between "this pixel is in shadow" and "this pixel is dark". Fitting rather
    than reading the flight metadata, because the WMS does not carry it -- and
    because a fit against this area's own geometry is the thing that has to be
    right, not the almanac.

    Two passes at two resolutions. The sweep is coarse and pooled, because
    recovering a direction does not need the geometry the final mask is cast
    on. The refinement is not: pooling by maximum fattens every caster by up
    to a cell, which lengthens its shadow, and the fit answers that by
    choosing a sun too high -- measured, 2.5 to 4.5 degrees too high at a
    four-metre pool. The azimuth survives that; the elevation does not, so it
    is settled at `FIT_PIXEL_M / 4`.
    """
    def prepare(metres: float):
        factor = max(1, int(round(metres / max(metres_per_px, 1e-6))))
        heights = _pool(surface_nap, factor, "max")
        bright = _pool(luminance, factor, "mean")
        if bright.shape != heights.shape:
            bright = _upsample(bright, heights.shape)
        return heights, bright, metres_per_px * factor

    def scorer(heights, bright, metres):
        target = -(bright - bright.mean())
        spread = float(target.std())

        def score(azimuth: float, elevation: float) -> float:
            if spread < 1e-6:
                return -1.0
            mask = cast_shadow(heights, metres, azimuth, elevation)
            covered = float(mask.mean())
            # A mask with almost no variance cannot be scored: the
            # correlation's denominator goes to zero and the answer is noise.
            # Note how loose the upper bound is. An earlier 0.7 was a belief
            # about how much of a city is in shadow rather than a numerical
            # guard, and it would reject the right answer for a dense area
            # under a low sun -- Bronbeek alone comes to 44%.
            if covered < 0.01 or covered > 0.95:
                return -1.0
            deviation = mask - covered
            denominator = float(deviation.std()) * spread
            if denominator < 1e-9:
                return -1.0
            return float((deviation * target).mean() / denominator)

        return score

    coarse = scorer(*prepare(FIT_PIXEL_M))
    sweep = []
    for azimuth in range(0, 360, 10):
        for elevation in np.arange(ELEVATION_RANGE[0], ELEVATION_RANGE[1], 6.0):
            sweep.append(
                (coarse(float(azimuth), float(elevation)),
                 float(azimuth), float(elevation))
            )
    sweep.sort(reverse=True)
    if not sweep or sweep[0][0] <= 0.0:
        return 180.0, 30.0, max((s[0] for s in sweep), default=-1.0)

    # Refine the best few rather than only the best. A low sun over regular
    # streets merges its shadows into a sheet, which flattens the score
    # surface into basins a purely local refinement cannot climb out of: on a
    # grid of terraces at 14 degrees it settled 20 degrees off.
    fine = scorer(*prepare(FIT_PIXEL_M / 4.0))
    best = (180.0, 30.0, -1.0)
    seeds = [(a, e) for _, a, e in sweep[:4]]
    for seed_azimuth, seed_elevation in seeds:
        azimuth, elevation = seed_azimuth, seed_elevation
        local = (azimuth, elevation, fine(azimuth, elevation))
        for azimuth_step, elevation_step in ((6.0, 5.0), (2.0, 2.0), (1.0, 1.0)):
            for trial_azimuth in (
                azimuth - azimuth_step, azimuth, azimuth + azimuth_step
            ):
                for trial_elevation in (
                    elevation - elevation_step, elevation, elevation + elevation_step
                ):
                    if not (
                        ELEVATION_RANGE[0] <= trial_elevation <= ELEVATION_RANGE[1]
                    ):
                        continue
                    value = fine(trial_azimuth % 360.0, trial_elevation)
                    if value > local[2]:
                        local = (trial_azimuth % 360.0, trial_elevation, value)
            azimuth, elevation, _ = local
        if local[2] > best[2]:
            best = local
    return best


def burn_triangles(
    triangles: np.ndarray,
    bbox: BBox,
    raster: np.ndarray,
) -> int:
    """Raise `raster` to the height of every triangle covering each cell.

    The maximum, not the last one written: a roof and the wall under it both
    project onto the same cells, and what casts a shadow is the top.

    Roof triangles projected down are exactly the building's footprint, so
    this gets the outline and the pitch in one pass without needing the rings
    the 3DBAG reader does not keep.
    """
    if not len(triangles):
        return 0
    rows, cols = raster.shape
    cell_x = bbox.width / cols
    cell_y = bbox.height / rows

    # Triangle corners in fractional cell coordinates, north-up.
    px = (triangles[:, :, 0] - bbox.xmin) / cell_x
    py = (bbox.ymax - triangles[:, :, 1]) / cell_y
    pz = triangles[:, :, 2]

    lo_c = np.maximum(np.floor(px.min(axis=1)).astype(np.int64), 0)
    hi_c = np.minimum(np.ceil(px.max(axis=1)).astype(np.int64), cols - 1)
    lo_r = np.maximum(np.floor(py.min(axis=1)).astype(np.int64), 0)
    hi_r = np.minimum(np.ceil(py.max(axis=1)).astype(np.int64), rows - 1)

    v0x, v0y = px[:, 1] - px[:, 0], py[:, 1] - py[:, 0]
    v1x, v1y = px[:, 2] - px[:, 0], py[:, 2] - py[:, 0]
    denominator = v0x * v1y - v0y * v1x

    burned = 0
    for i in range(len(triangles)):
        if hi_c[i] < lo_c[i] or hi_r[i] < lo_r[i] or abs(denominator[i]) < 1e-12:
            continue
        cc = np.arange(lo_c[i], hi_c[i] + 1) + 0.5
        rr = np.arange(lo_r[i], hi_r[i] + 1) + 0.5
        dx = cc[None, :] - px[i, 0]
        dy = rr[:, None] - py[i, 0]
        u = (dx * v1y[i] - dy * v1x[i]) / denominator[i]
        v = (dy * v0x[i] - dx * v0y[i]) / denominator[i]
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        if not inside.any():
            continue
        height = pz[i, 0] + u * (pz[i, 1] - pz[i, 0]) + v * (pz[i, 2] - pz[i, 0])
        block = raster[lo_r[i] : hi_r[i] + 1, lo_c[i] : hi_c[i] + 1]
        np.maximum(block, np.where(inside, height, -np.inf), out=block)
        burned += 1
    return burned


def burn_canopies(trees, bbox: BBox, raster: np.ndarray) -> int:
    """Raise `raster` to the top of each tree's crown, over its crown disk.

    A disk rather than the drawn lobes: a canopy's shadow is its outline, and
    at half a metre a crown is a handful of cells wide either way.
    """
    members = getattr(trees, "trees", None) or []
    if not len(members):
        return 0
    rows, cols = raster.shape
    px_per_m_x = cols / bbox.width
    px_per_m_y = rows / bbox.height

    for tree in members:
        top = float(tree.ground_z_nap) + float(tree.height_m)
        radius = max(float(tree.crown_radius_m), 0.3)
        cx = (float(tree.x) - bbox.xmin) * px_per_m_x - 0.5
        cy = (bbox.ymax - float(tree.y)) * px_per_m_y - 0.5
        rx, ry = radius * px_per_m_x, radius * px_per_m_y
        c0 = max(0, int(math.floor(cx - rx)))
        c1 = min(cols, int(math.ceil(cx + rx)) + 1)
        r0 = max(0, int(math.floor(cy - ry)))
        r1 = min(rows, int(math.ceil(cy + ry)) + 1)
        if c1 <= c0 or r1 <= r0:
            continue
        dx = (np.arange(c0, c1) - cx) / rx
        dy = (np.arange(r0, r1) - cy) / ry
        inside = dx[None, :] ** 2 + dy[:, None] ** 2 <= 1.0
        block = raster[r0:r1, c0:c1]
        np.maximum(block, np.where(inside, top, -np.inf), out=block)
    return len(members)


def surface_raster(
    bbox: BBox,
    metres_per_px: float,
    *,
    terrain=None,
    buildings=None,
    trees=None,
) -> np.ndarray:
    """Everything that can cast a shadow, as heights in NAP.

    Absolute heights rather than heights above ground, so a slope shadows
    itself correctly -- Arnhem sits on a moraine and Bronbeek is on the side
    of it. The ground is the floor; buildings and canopies raise it.
    """
    rows = max(2, int(round(bbox.height / metres_per_px)))
    cols = max(2, int(round(bbox.width / metres_per_px)))

    raster = np.zeros((rows, cols), dtype=np.float64)
    if terrain is not None:
        xs = bbox.xmin + (np.arange(cols) + 0.5) * (bbox.width / cols)
        ys = bbox.ymax - (np.arange(rows) + 0.5) * (bbox.height / rows)
        grid_x, grid_y = np.meshgrid(xs, ys)
        raster = np.asarray(
            terrain.sample(grid_x.ravel(), grid_y.ravel()), dtype=np.float64
        ).reshape(rows, cols)

    if buildings is not None:
        roofs = [b.roof_tris for b in getattr(buildings, "buildings", []) if len(b.roof_tris)]
        if roofs:
            burn_triangles(np.concatenate(roofs, axis=0), bbox, raster)
    if trees is not None:
        burn_canopies(trees, bbox, raster)
    return raster


def _box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    """Separable moving average, for feathering the mask edge.

    A shadow edge is soft -- the sun is half a degree wide and the sky fills
    in the rest -- so a hard mask boundary would print a seam into the photo
    that was not there before.
    """
    if radius < 1:
        return array
    window = 2 * radius + 1
    out = array.astype(np.float32)
    for axis in (0, 1):
        padded = np.pad(out, [(radius + 1, radius) if a == axis else (0, 0)
                              for a in (0, 1)], mode="edge")
        summed = np.cumsum(padded, axis=axis)
        lo = np.take(summed, np.arange(0, out.shape[axis]), axis=axis)
        hi = np.take(summed, np.arange(window, window + out.shape[axis]), axis=axis)
        out = (hi - lo) / window
    return out


def _upsample(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of the shadow mask onto the photo."""
    rows = (np.arange(shape[0]) * mask.shape[0] // shape[0]).clip(
        0, mask.shape[0] - 1
    )
    cols = (np.arange(shape[1]) * mask.shape[1] // shape[1]).clip(
        0, mask.shape[1] - 1
    )
    return mask[rows[:, None], cols[None, :]]


def _class_gains(
    pixels: np.ndarray,
    classes: np.ndarray,
    shadow: np.ndarray,
    max_gain: float,
) -> dict[int, tuple[np.ndarray, float, float]]:
    """Per class: the per-channel gain, and the lit and shadowed luminance.

    Measured off the photo rather than assumed, and per channel, which is how
    the blue cast of skylight comes out in the wash: a shadow is lit by sky
    alone, so it loses more red than blue, and three separate gains undo that
    without anyone having to name a colour temperature.

    Per class because the ratio is a property of the surface. Grass in shade
    against grass in sun is a fair comparison; grass in shade against asphalt
    in sun is not, and one global gain is exactly that comparison.

    The two luminances come back as well, because they are what tells a pixel
    that merely *is* dark from one that is *in shadow*.
    """
    weights = np.array([0.2126, 0.7152, 0.0722])
    lit = shadow < 0.05
    deep = shadow > 0.95

    def measure(selector_lit, selector_dark):
        if selector_lit.sum() < 500 or selector_dark.sum() < 500:
            return None
        bright = np.median(pixels[selector_lit], axis=0)
        dark = np.median(pixels[selector_dark], axis=0)
        if np.any(dark < 1.0):
            return None
        return (
            np.clip(bright / dark, 1.0, max_gain),
            float(bright @ weights),
            float(dark @ weights),
        )

    overall = measure(lit, deep) or (np.ones(3), 255.0, 0.0)
    return {
        int(code): (measure((classes == code) & lit, (classes == code) & deep)
                    or overall)
        for code in np.unique(classes)
    }


def remove_cast_shadows(
    aerial_path,
    bbox: BBox,
    *,
    terrain=None,
    buildings=None,
    trees=None,
    class_grid=None,
    strength: float = 1.0,
    max_gain: float = MAX_GAIN,
) -> dict:
    """Fit the flown sun and take its shadows back out of the aerial, in place.

    Returns what it did, for the log and the metadata: an empty dict when
    there was nothing to work with or nothing worth changing.
    """
    from PIL import Image

    from .imagery import _allow_large_images

    if strength <= 0:
        return {}
    if buildings is None and trees is None:
        LOG.info("no geometry to cast shadows from; leaving the aerial alone")
        return {}

    _allow_large_images()
    with Image.open(aerial_path) as opened:
        image = np.asarray(opened.convert("RGB"), dtype=np.float32)
    height, width = image.shape[:2]

    surface = surface_raster(
        bbox, SHADOW_PIXEL_M, terrain=terrain, buildings=buildings, trees=trees
    )
    relief = float(surface.max() - surface.min())
    if relief < 1.0:
        LOG.info("nothing stands up in this area; no shadows to remove")
        return {}

    luminance = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    azimuth, elevation, score = fit_sun(
        surface, _upsample(luminance, surface.shape), SHADOW_PIXEL_M
    )
    if score <= 0.05:
        # The geometry explains none of the darkness. A flat overcast flight
        # has no shadows to remove, and guessing at one would only stain the
        # photo where the buildings happen to be.
        LOG.info(
            "no sun found in this aerial (best fit explains r=%.3f); leaving "
            "it alone -- an overcast flight has nothing to remove",
            score,
        )
        return {}

    shadow_small = cast_shadow(surface, SHADOW_PIXEL_M, azimuth, elevation)
    # Feather at the photo's scale, not the raster's: about a metre, which is
    # roughly where a real penumbra lands for these heights.
    shadow = _box_blur(
        _upsample(shadow_small, (height, width)),
        max(1, int(round(1.0 / (bbox.width / width)))),
    )

    if class_grid is not None:
        classes = _upsample(np.flipud(np.asarray(class_grid)), (height, width))
    else:
        classes = np.zeros((height, width), dtype=np.int32)

    # Statistics off a subsample: a median over every ninth pixel is the same
    # median, and a square kilometre at 8 cm is 150 million of them.
    step = max(1, int(round(math.sqrt(height * width / 4_000_000))))
    measured = _class_gains(
        image[::step, ::step].reshape(-1, 3),
        classes[::step, ::step].ravel(),
        shadow[::step, ::step].ravel(),
        max_gain,
    )

    gain_image = np.ones((height, width, 3), dtype=np.float32)
    lit_lum = np.full((height, width), 255.0, dtype=np.float32)
    dark_lum = np.zeros((height, width), dtype=np.float32)
    for code, (gain, bright, dark) in measured.items():
        here = classes == code
        gain_image[here] = gain.astype(np.float32)
        lit_lum[here] = bright
        dark_lum[here] = dark

    # Only brighten what actually looks dark. The mask says a shadow was cast
    # here; this says the photo agrees. Where they differ -- a tree the
    # detector missed, a building 3DBAG has a metre out of place, a car parked
    # since the flight -- the photo wins and nothing gets stained.
    confidence = np.clip(
        (lit_lum - luminance) / np.maximum(lit_lum - dark_lum, 1.0), 0.0, 1.0
    )
    applied = shadow * float(strength) * confidence

    shadowed = shadow > 0.5
    before = float(luminance[shadowed].mean()) if shadowed.any() else 0.0

    image *= 1.0 + applied[:, :, None] * (gain_image - 1.0)
    np.clip(image, 0.0, 255.0, out=image)

    after_lum = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    after = float(after_lum[shadowed].mean()) if shadowed.any() else 0.0
    lit_mean = float(after_lum[shadow < 0.05].mean()) if (shadow < 0.05).any() else 0.0

    Image.fromarray(image.astype(np.uint8), mode="RGB").save(aerial_path)
    result = {
        "sun_azimuth_deg": round(azimuth, 1),
        "sun_elevation_deg": round(elevation, 1),
        "fit_r": round(score, 3),
        "shadow_fraction": round(float(shadowed.mean()), 4),
        "mean_gain": round(
            float(np.mean([np.mean(g) for g, _, _ in measured.values()])), 3
        ),
        "shadow_luminance_before": round(before, 1),
        "shadow_luminance_after": round(after, 1),
        "lit_luminance": round(lit_mean, 1),
    }
    LOG.info(
        "de-shadowed %s: sun at %.0f deg azimuth, %.0f deg up (r=%.2f), "
        "%.0f%% of the photo in shadow, lifted from %.0f to %.0f against "
        "%.0f in the sun",
        getattr(aerial_path, "name", aerial_path),
        azimuth, elevation, score,
        result["shadow_fraction"] * 100, before, after, lit_mean,
    )
    return result


__all__ = [
    "ELEVATION_RANGE",
    "MAX_GAIN",
    "SHADOW_PIXEL_M",
    "burn_canopies",
    "burn_triangles",
    "cast_shadow",
    "fit_sun",
    "remove_cast_shadows",
    "surface_raster",
]
