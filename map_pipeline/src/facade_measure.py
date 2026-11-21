"""Reading a facade off a straightened photo of it.

The generated facades guess a wall from the building's age: a 1930s house
gets interbellum brick and two windows a bay. This module looks instead.
From a wall seen square-on (see `facade_capture`) it measures what the
facade generator needs to draw that wall rather than its era's average:

``wall_rgb``
    The colour of the wall itself: the largest colour cluster on it, read
    from its sunlit half so a cast shadow does not darken a red brick wall
    into a brown one.
``bay_m``
    The spacing of the window columns, from the periodicity across the wall
    of how unlike the wall each pixel is. Taken as the fundamental period,
    not a multiple of it: a wall of identical windows correlates with itself
    at one, two and three bays along.
``storey_m``
    The same periodicity measured upwards, which is the storey height.
``window_width_frac``, ``window_height_frac``, ``sill_frac``
    The window's share of a bay and of a storey, and where its sill sits, by
    folding the wall over one bay and one storey and seeing where the
    openings fall.

Every number comes with a confidence, and a measurement that does not stand
out from the wall's own texture is left out rather than guessed: the era's
default is a better answer than a wrong measurement.

The bottom two metres are not read. Along nearly every Dutch street that is
where the parked cars are, and they are in every photo of the wall.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

from .facade_capture import WallImage, luminance

LOG = logging.getLogger(__name__)

# Heights above the ground that are read. Below, cars and bins.
SKIP_BOTTOM_M = 2.0
# On a wall tall enough to have its ground storey split off, that storey is
# not read with the rest: a row of shopfronts seven metres apart drowned out
# the upper windows two and a half metres apart on a hotel in Arnhem, and the
# model gives the ground storey a texture of its own anyway.
GROUND_STOREY_M = 3.6
SKIP_TOP_M = 0.3
MIN_BAY_M, MAX_BAY_M = 1.0, 8.0
MIN_STOREY_M, MAX_STOREY_M = 2.4, 4.4
# A period counts when its autocorrelation reaches this.
MIN_PERIOD_CONFIDENCE = 0.15
# A profile varying less than this has no openings worth a rhythm.
MIN_PROFILE_SPREAD = 0.03
# A wall has to be this well seen before it is read at all.
MIN_COVERAGE = 0.35
MIN_PIXELS = 4000


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0..255 to CIE Lab (D65), any leading shape."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def kmeans(points: np.ndarray, k: int, *, iterations: int = 12, seed: int = 0):
    """Plain k-means, k-means++ seeded. Returns ``(centres, labels)``."""
    rng = np.random.default_rng(seed)
    n = len(points)
    k = max(1, min(k, n))
    centres = [points[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        if total <= 0:
            break
        centres.append(points[rng.choice(n, p=d2 / total)])
    centres = np.array(centres, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iterations):
        d2 = ((points[:, None, :] - centres[None]) ** 2).sum(-1)
        labels = np.argmin(d2, axis=1)
        moved = False
        for j in range(len(centres)):
            members = points[labels == j]
            if len(members):
                new = members.mean(axis=0)
                moved |= not np.allclose(new, centres[j])
                centres[j] = new
        if not moved:
            break
    return centres, labels


# --------------------------------------------------------------------------
# Periodicity
# --------------------------------------------------------------------------


def _detrend(profile: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    pad = window // 2
    padded = np.pad(profile, pad, mode="reflect")
    kernel = np.ones(window) / window
    trend = np.convolve(padded, kernel, mode="valid")
    return profile - trend[: len(profile)]


def autocorrelation(profile: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation, lag 0 first, for non-negative lags.

    The biased estimate, every lag divided by the full length: dividing by
    the shrinking number of pairs instead lets noise at long lags score as
    high as a real rhythm, and read a 4 m bay off a blank wall.
    """
    x = profile - profile.mean()
    n = len(x)
    if n < 4 or not np.any(x):
        return np.zeros(n)
    spectrum = np.fft.rfft(x, 2 * n)
    corr = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    return corr / max(corr[0], 1e-12)


def dominant_period(
    profile: np.ndarray, step_m: float, lo_m: float, hi_m: float
) -> tuple[float, float]:
    """The fundamental period of a profile within [lo_m, hi_m], and its score.

    Every local peak of the autocorrelation in range is a candidate. The
    shortest one that nearly matches the best is taken, since a period's
    multiples all correlate too and the longest of them often scores
    highest by chance. Returns ``(0, 0)`` when nothing stands out.
    """
    corr = autocorrelation(profile)
    lo = max(1, int(np.floor(lo_m / step_m)))
    hi = min(len(corr) - 2, int(np.ceil(hi_m / step_m)))
    if hi <= lo + 1:
        return 0.0, 0.0
    peaks = [
        lag
        for lag in range(lo, hi + 1)
        if corr[lag] >= corr[lag - 1] and corr[lag] >= corr[lag + 1] and corr[lag] > 0
    ]
    if not peaks:
        return 0.0, 0.0
    best = max(corr[p] for p in peaks)
    if best < MIN_PERIOD_CONFIDENCE:
        return 0.0, float(best)
    chosen = min(p for p in peaks if corr[p] >= 0.8 * best)
    # Parabolic refinement of the peak to a fraction of a pixel.
    a, b, c = corr[chosen - 1], corr[chosen], corr[chosen + 1]
    denom = a - 2 * b + c
    offset = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    return float((chosen + np.clip(offset, -0.5, 0.5)) * step_m), float(corr[chosen])


def folded(profile: np.ndarray, weight: np.ndarray, step_m: float, period_m: float, origin_m: float = 0.0, bins: int = 24) -> np.ndarray:
    """Average of a profile over one period, `bins` bins from `origin_m`."""
    positions = origin_m + (np.arange(len(profile)) + 0.5) * step_m
    phase = np.mod(positions, period_m) / period_m
    which = np.minimum((phase * bins).astype(np.int64), bins - 1)
    total = np.bincount(which, weights=profile * weight, minlength=bins)
    count = np.bincount(which, weights=weight, minlength=bins)
    return np.where(count > 0, total / np.maximum(count, 1e-9), np.nan)


def _contrast(fold: np.ndarray) -> tuple[float, float]:
    """``(range, mean)`` of a folded profile, ignoring empty bins."""
    values = fold[np.isfinite(fold)]
    if not len(values):
        return 0.0, 0.0
    return float(values.max() - values.min()), float(values.mean())


# A folded window row flatter than this is one long window, not a row of
# them: ribbon glazing, where glass and frames run on from bay to bay.
RIBBON_CONTRAST = 0.12
RIBBON_WIDTH = 0.86


def _band(fold: np.ndarray) -> tuple[float, float]:
    """Where the openings are in a folded profile: ``(start, length)`` as
    fractions of the period, the longest run above half the range."""
    values = np.nan_to_num(fold, nan=np.nanmin(fold) if np.isfinite(fold).any() else 0.0)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        return 0.0, 0.0
    above = values >= lo + 0.5 * (hi - lo)
    n = len(values)
    best_start, best_len = 0, 0
    # Runs may wrap around the end of the period.
    doubled = np.r_[above, above]
    run = 0
    for i, flag in enumerate(doubled):
        run = run + 1 if flag else 0
        if run > best_len and run <= n:
            best_len, best_start = run, (i - run + 1) % n
    return best_start / n, best_len / n


# --------------------------------------------------------------------------
# One wall
# --------------------------------------------------------------------------


def _merge_shades(centres: np.ndarray, rgb_centres: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Which cluster each centre belongs to once shades are merged.

    The sunlit and the shaded part of one wall are one surface, and k-means
    happily splits them. Shade is not only darker but bluer, lit by the sky
    rather than the sun, so the test is on hue: two coloured clusters within
    15 degrees of hue are one surface at any brightness. Two near-neutral
    clusters merge only when close in lightness, since grey glass and white
    frames are both neutral and are not one surface; and a coloured cluster
    never merges with a neutral one. Each cluster joins the largest one it
    matches directly, never through a chain.

    `centres` are Lab. Returns, per centre, the centre that speaks for it.
    """
    k = len(centres)
    chroma = np.hypot(centres[:, 1], centres[:, 2])
    hue = np.degrees(np.arctan2(centres[:, 2], centres[:, 1]))
    order = np.argsort(-sizes)
    root = np.arange(k)
    for rank, i in enumerate(order):
        for j in order[:rank]:
            if root[j] != j:
                continue
            light = abs(float(centres[i][0] - centres[j][0]))
            if chroma[i] >= 8.0 and chroma[j] >= 8.0:
                turn = abs((hue[i] - hue[j] + 180.0) % 360.0 - 180.0)
                same = turn < 15.0 and light < 45.0
            elif chroma[i] < 8.0 and chroma[j] < 8.0:
                same = light < 15.0
            else:
                same = False
            if same:
                root[i] = j
                break
    return root


def _wall_cluster(lab_all: np.ndarray, good: np.ndarray, centres: np.ndarray, merged: np.ndarray, step: float) -> int:
    """The colour cluster that is the wall: big, and everywhere.

    Glass and frames come in columns and rows; the wall is what lies between
    all of them, so it turns up in nearly every square metre. Size alone
    picked the panes over the plaster on an office whose windows fill more
    of the front than the wall does. Returns a merged cluster index.
    """
    d2 = ((lab_all[..., None, :] - centres[None, None]) ** 2).sum(-1)
    cluster = merged[np.argmin(d2, axis=-1)]
    cell = max(1, int(round(1.0 / step)))
    h, w = good.shape
    hc, wc = h // cell, w // cell
    candidates = np.unique(merged)
    sizes = {int(c): float(((cluster == c) & good).sum()) / max(1.0, float(good.sum())) for c in candidates}
    if hc < 1 or wc < 1:
        return max(sizes, key=sizes.get)
    seen = good[: hc * cell, : wc * cell].reshape(hc, cell, wc, cell).mean(axis=(1, 3)) > 0.5
    best, best_score = max(sizes, key=sizes.get), -1.0
    for c in candidates:
        mask = (cluster == c) & good
        share = mask[: hc * cell, : wc * cell].reshape(hc, cell, wc, cell).mean(axis=(1, 3))
        presence = float((share[seen] > 0.1).mean()) if seen.any() else 0.0
        score = presence * sizes[int(c)]
        if score > best_score:
            best, best_score = int(c), score
    return best


@dataclass
class WallMeasurement:
    face: int
    building_id: str
    area_m2: float
    coverage: float
    wall_rgb: tuple[int, int, int] | None = None
    trim_rgb: tuple[int, int, int] | None = None
    glass_rgb: tuple[int, int, int] | None = None
    # Share of the read area that is wall rather than opening.
    wall_share: float = 0.0
    bay_m: float = 0.0
    bay_confidence: float = 0.0
    storey_m: float = 0.0
    storey_confidence: float = 0.0
    window_width_frac: float = 0.0
    window_height_frac: float = 0.0
    sill_frac: float = 0.0
    # The brightness gain the colours were lifted by (see photo_colour).
    gain: float = 1.0
    views: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        for key in ("bay_m", "bay_confidence", "storey_m", "storey_confidence",
                    "window_width_frac", "window_height_frac", "sill_frac",
                    "coverage", "wall_share", "area_m2", "gain"):
            out[key] = round(float(out[key]), 3)
        return out


def opening_map(rgb: np.ndarray, wall_lab: np.ndarray) -> np.ndarray:
    """How unlike the wall each pixel is, 0 (wall) to 1 (certainly not)."""
    distance = np.linalg.norm(rgb_to_lab(rgb) - wall_lab, axis=-1)
    return np.clip((distance - 8.0) / 22.0, 0.0, 1.0)


def measure_wall(
    wall: WallImage,
    *,
    ground_z: float,
    wall_top_z: float,
    building_id: str = "",
    area_m2: float = 0.0,
    storey_origin_z: float | None = None,
) -> WallMeasurement:
    """Measure one straightened wall. Fields stay empty where unsure.

    `storey_origin_z` is where the storeys are counted from in the model,
    the ground or the top of a split-off ground storey; the window's sill is
    measured from it so the texture's windows land where the photo's are.
    """
    out = WallMeasurement(
        face=wall.face, building_id=building_id, area_m2=area_m2,
        coverage=wall.coverage, views=list(wall.views),
    )
    step = wall.m_per_px
    z = wall.z_of_row(np.arange(wall.rgb.shape[0]))
    bottom = SKIP_BOTTOM_M
    if wall_top_z - ground_z > GROUND_STOREY_M * 1.2 + 2 * MIN_STOREY_M:
        bottom = max(bottom, GROUND_STOREY_M + 0.2)
    rows = (z >= ground_z + bottom) & (z <= wall_top_z - SKIP_TOP_M)
    if rows.sum() < 10:
        return out
    rgb = wall.rgb[rows]
    weight = wall.weight[rows]
    good = weight > 0.3
    if good.sum() < MIN_PIXELS or float(good.mean()) < MIN_COVERAGE:
        return out

    # Wall colour: the largest cluster, read from its sunlit half.
    pixels = rgb[good]
    rng = np.random.default_rng(1)
    sample = pixels[rng.choice(len(pixels), size=min(len(pixels), 20000), replace=False)]
    lab = rgb_to_lab(sample)
    centres, labels = kmeans(lab, 4, seed=1)
    rgb_centres = np.array([
        np.median(sample[labels == j], axis=0) if np.any(labels == j) else np.zeros(3)
        for j in range(len(centres))
    ])
    merged = _merge_shades(centres, rgb_centres, np.bincount(labels, minlength=len(centres)))
    lab_all = rgb_to_lab(rgb)
    wall_cluster = _wall_cluster(lab_all, good, centres, merged, step)
    labels = merged[labels]
    sizes = np.bincount(labels, minlength=len(centres))
    members = sample[labels == wall_cluster]
    # Somewhat above the middle of the wall's own brightness: clear of the
    # shade under sills and reveals, short of the glare a low sun puts on
    # pale concrete. The sunlit half alone read a beige office as white.
    lum = luminance(members)
    level = float(np.percentile(lum, 60))
    near = np.abs(lum - level) <= max(4.0, 0.08 * level)
    wall_rgb = np.median(members[near] if near.sum() > 20 else members, axis=0)
    out.wall_rgb = tuple(int(round(v)) for v in wall_rgb)
    out.wall_share = float(sizes[wall_cluster] / max(1, len(sample)))

    openings = np.clip(
        (np.linalg.norm(lab_all - rgb_to_lab(wall_rgb), axis=-1) - 8.0) / 22.0, 0.0, 1.0
    )
    lum_all = luminance(rgb)
    lum_wall = float(luminance(wall_rgb[None])[0])
    glass = good & (openings > 0.6) & (lum_all < 0.8 * lum_wall)
    frames = good & (openings > 0.6) & (lum_all > 1.15 * lum_wall)
    if glass.sum() > 200:
        out.glass_rgb = tuple(int(round(v)) for v in np.median(rgb[glass], axis=0))
    if frames.sum() > 200:
        out.trim_rgb = tuple(int(round(v)) for v in np.median(rgb[frames], axis=0))

    w = weight * good
    # Across: the window columns, read from the upper part of the wall. The
    # upper storeys carry a building's typical rhythm; a double-height lobby
    # or a row of shopfronts does not, and on a hotel in Arnhem its seven
    # metre rhythm outvoted the two and a half of every window above it.
    upper = (z[rows] >= ground_z + max(bottom, 0.4 * (wall_top_z - ground_z)))[:, None]
    wc = w * upper
    col_total = (openings * wc).sum(axis=0)
    col_count = wc.sum(axis=0)
    seen_cols = col_count > 0.2 * max(1, int(upper.sum()))
    if seen_cols.sum() * step >= 2.5 * MIN_BAY_M:
        column = np.where(seen_cols, col_total / np.maximum(col_count, 1e-9), np.nan)
        column = np.where(np.isfinite(column), column, np.nanmean(column))
        if float(np.std(column)) < MIN_PROFILE_SPREAD:
            column = np.zeros_like(column)
        column = _detrend(column, round(MAX_BAY_M * 1.5 / step))
        bay, confidence = dominant_period(
            column, step, MIN_BAY_M, min(MAX_BAY_M, 0.5 * seen_cols.sum() * step)
        )
        if bay > 0:
            out.bay_m, out.bay_confidence = bay, confidence
            fold = folded(
                col_total / np.maximum(col_count, 1e-9), seen_cols.astype(float),
                step, bay,
            )
            spread, level = _contrast(fold)
            if spread < RIBBON_CONTRAST and level > 0.45:
                # Glass and frames all along: an office's ribbon windows read
                # as narrow slits in a grey wall until this was told apart.
                out.window_width_frac = RIBBON_WIDTH
            else:
                _, length = _band(fold)
                out.window_width_frac = float(np.clip(length, 0.12, 0.8))

    # Up: the storeys.
    row_total = (openings * w).sum(axis=1)
    row_count = w.sum(axis=1)
    seen_rows = row_count > 0.2 * rgb.shape[1]
    if seen_rows.sum() * step >= 2.0 * MIN_STOREY_M:
        profile = np.where(seen_rows, row_total / np.maximum(row_count, 1e-9), np.nan)
        profile = np.where(np.isfinite(profile), profile, np.nanmean(profile))
        if float(np.std(profile)) < MIN_PROFILE_SPREAD:
            profile = np.zeros_like(profile)
        # A wall darkens towards the eaves, or has a band of attic: a trend
        # that otherwise buries the storey rhythm.
        profile = _detrend(profile, round(MAX_STOREY_M * 1.5 / step))
        storey, confidence = dominant_period(
            profile, step, MIN_STOREY_M, min(MAX_STOREY_M, 0.6 * seen_rows.sum() * step)
        )
        if storey > 0:
            out.storey_m, out.storey_confidence = storey, confidence
            origin = ground_z if storey_origin_z is None else storey_origin_z
            # Rows run top-down; fold bottom-up from the storey origin.
            heights = z[rows][::-1] - origin
            values = (row_total / np.maximum(row_count, 1e-9))[::-1]
            weights = seen_rows[::-1].astype(float)
            phase = np.mod(heights, storey) / storey
            bins = 24
            which = np.minimum((phase * bins).astype(np.int64), bins - 1)
            total = np.bincount(which, weights=values * weights, minlength=bins)
            count = np.bincount(which, weights=weights, minlength=bins)
            fold = np.where(count > 0, total / np.maximum(count, 1e-9), np.nan)
            start, length = _band(fold)
            out.window_height_frac = float(np.clip(length, 0.2, 0.85))
            out.sill_frac = float(np.clip(start, 0.03, 0.6))
    return out


# --------------------------------------------------------------------------
# One building
# --------------------------------------------------------------------------


@dataclass
class FacadeMeasurement:
    """What the photos say about one building, its walls taken together."""

    building_id: str
    walls: list[WallMeasurement]
    wall_rgb: tuple[int, int, int] | None = None
    trim_rgb: tuple[int, int, int] | None = None
    glass_rgb: tuple[int, int, int] | None = None
    bay_m: float = 0.0
    storeys: int = 0
    storey_m: float = 0.0
    window_width_frac: float = 0.0
    window_height_frac: float = 0.0
    sill_frac: float = 0.0

    @property
    def has_colour(self) -> bool:
        return self.wall_rgb is not None

    @property
    def has_windows(self) -> bool:
        return self.bay_m > 0 and self.window_width_frac > 0

    def to_dict(self) -> dict:
        return {
            "building": self.building_id,
            "wall_rgb": self.wall_rgb,
            "trim_rgb": self.trim_rgb,
            "glass_rgb": self.glass_rgb,
            "bay_m": round(self.bay_m, 2),
            "storeys": self.storeys,
            "storey_m": round(self.storey_m, 2),
            "window_width_frac": round(self.window_width_frac, 3),
            "window_height_frac": round(self.window_height_frac, 3),
            "sill_frac": round(self.sill_frac, 3),
            "walls": [w.to_dict() for w in self.walls],
        }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def combine_walls(
    building_id: str,
    walls: list[WallMeasurement],
    *,
    ground_z: float,
    wall_top_z: float,
    model_floors: int,
) -> FacadeMeasurement:
    """One building's facade from its measured walls.

    Each number is a median over the walls that measured it, weighted by wall
    area and by how clearly the measurement stood out, so a large street
    front outvotes a sliver of side wall seen past a corner.
    """
    out = FacadeMeasurement(building_id=building_id, walls=walls)
    coloured = [w for w in walls if w.wall_rgb is not None]
    if coloured:
        # The colour of one of the walls, the one closest to all the others,
        # weighted by area: not a per-channel median, which over a white
        # front, a brick side and a glass wall made a blue no wall had.
        weights = np.array([w.area_m2 * w.coverage for w in coloured])
        lab = rgb_to_lab(np.array([w.wall_rgb for w in coloured], dtype=np.float64))
        spread = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1) @ weights
        out.wall_rgb = tuple(int(v) for v in coloured[int(np.argmin(spread))].wall_rgb)
        trims = [w for w in coloured if w.trim_rgb is not None]
        if trims:
            out.trim_rgb = tuple(int(v) for v in np.median([w.trim_rgb for w in trims], axis=0))
        glasses = [w for w in coloured if w.glass_rgb is not None]
        if glasses:
            out.glass_rgb = tuple(int(v) for v in np.median([w.glass_rgb for w in glasses], axis=0))

    bays = [w for w in walls if w.bay_m > 0]
    if bays:
        weights = np.array([w.area_m2 * w.bay_confidence for w in bays])
        out.bay_m = _weighted_median(np.array([w.bay_m for w in bays]), weights)
        out.window_width_frac = _weighted_median(
            np.array([w.window_width_frac for w in bays]), weights
        )

    storeys = [w for w in walls if w.storey_m > 0]
    if storeys:
        weights = np.array([w.area_m2 * w.storey_confidence for w in storeys])
        out.storey_m = _weighted_median(np.array([w.storey_m for w in storeys]), weights)
        out.window_height_frac = _weighted_median(
            np.array([w.window_height_frac for w in storeys]), weights
        )
        out.sill_frac = _weighted_median(np.array([w.sill_frac for w in storeys]), weights)
        height = max(wall_top_z - ground_z, 0.0)
        # The ground storey is taller than the rest, and was not read: one
        # of it, and the measured storeys above it. Dividing the whole
        # height by the upper storey made a 14.4 m block of four storeys
        # over a shop floor into six.
        if height > GROUND_STOREY_M * 1.2 + MIN_STOREY_M:
            counted = 1 + int(round((height - GROUND_STOREY_M) / out.storey_m))
        else:
            counted = int(round(height / out.storey_m)) if out.storey_m > 0 else 0
        # Trusted when it agrees with the model to within a storey: 3DBAG's
        # count comes from the height and a typical storey, and a photo that
        # sees twice as many storeys has more likely read a pattern in the
        # brick than found a building nobody measured.
        if counted >= 1 and (model_floors <= 0 or abs(counted - model_floors) <= 1):
            out.storeys = counted
    return out


__all__ = [
    "FacadeMeasurement",
    "WallMeasurement",
    "autocorrelation",
    "combine_walls",
    "dominant_period",
    "folded",
    "kmeans",
    "measure_wall",
    "opening_map",
    "rgb_to_lab",
]
