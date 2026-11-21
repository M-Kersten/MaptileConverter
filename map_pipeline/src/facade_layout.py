"""A facade's layout, measured off its photos, and the wall drawn from it.

Pasting the photographs onto the walls put a photograph's detail -- its
parked bicycles, its reflections, its people half-hidden by a planter --
next to the model's own drawn detail, and the mix is uncanny. The photos are
used here as what they are good for, a measurement, and the wall is drawn
from what they measured at the same level of detail as every other wall.

From the photos of one wall, each straightened square-on at ``M_PER_PX``
(see `facade_capture.rectify_view`), this reads

* the wall's own colour and what it is made of: brick, plaster or stone;
* its window rows and window columns, by fitting the structure rather than
  looking for a rhythm: the height is split into alternating wall and window
  bands, the width into piers and windows, each by an exact one-dimensional
  segmentation (`segment_runs`) of how unlike the wall each line of pixels
  is. A canal house two windows wide has no rhythm to find; it has two
  windows, and the segmentation finds them;
* each window: a row and a column cross in a cell, and the cell holds a
  window when it looks like one, holds wall when it plainly is wall, and
  follows its column where no photo saw it;
* the storeys, from the window rows;
* the ground storey: a shop front -- glass between piers under a fascia --
  or a house front, a door and windows, decided by what the BAG says the
  building is for and by how much glass the photos see down there;
* the colour of the window frames and of the glass.

The views of a wall are not blended into one picture, which ghosts when they
disagree by a few centimetres. Each is read on its own and their profiles
are added, weighted by how well each sees each pixel.

Every layout carries a quality. A wall the photos could not read well keeps
what they did read -- its colour, its material -- and has its openings laid
out from the building's data instead (`layout_from_data`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

LOG = logging.getLogger(__name__)

# How finely a wall is read: 40 pixels a metre, enough to see a frame.
M_PER_PX = 0.025
# A pixel counts as seen above this weight (see facade_capture).
SEEN = 0.35
# Profiles are read in bins this many metres tall or wide.
BIN_M = 0.05

# What a window band, a window column and the wall between them may measure.
ROW_MIN_M, ROW_MAX_M = 0.6, 3.6
ROW_GAP_M = 0.3
COLUMN_MIN_M, COLUMN_MAX_M = 0.4, 3.5
COLUMN_GAP_M = 0.18
# Below this height above the ground the photos see cars, people, bicycles
# and terraces. Window rows are looked for above it; the lowest of them, if
# its sill is near the street, is the ground storey's.
GROUND_SKIP_M = 0.8
# The wall's colour is read above the ground storey, out of the clutter.
COLOUR_FROM_M = 3.0
# A row whose sill is under this is the ground storey's.
GROUND_ROW_SILL_M = 1.6
# A sill stands about this high above its floor.
SILL_HEIGHT_M = 0.85
GROUND_STOREY_M = (2.8, 5.2)
DEFAULT_GROUND_M = 3.6
DEFAULT_STOREY_M = 3.0
# A cell is a window when it is this much of the way from the wall's score
# to the windows' score; wall below the lower mark.
CELL_ON, CELL_OFF = 0.55, 0.3
# Shop fronts: glass in the ground storey over this share of the width.
SHOP_GLASS_SHARE = 0.45
FASCIA_M = 0.65
FRAME_M = 0.07
BULKHEAD_M = 0.45
DOOR_W_M, DOOR_H_M = 1.05, 2.3


@dataclass
class Opening:
    """A hole in the wall: a window, a door, or a shop's glass."""

    kind: str  # "window", "door", "shop"
    # Metres along the wall from its left end, seen from outside.
    x0: float
    x1: float
    # NAP.
    z0: float
    z1: float
    # Round head: 0 square, 1 a full half circle.
    arch: float = 0.0

    @property
    def width_m(self) -> float:
        return self.x1 - self.x0

    @property
    def height_m(self) -> float:
        return self.z1 - self.z0


@dataclass
class FacadeLayout:
    """One wall, as it is to be drawn."""

    face: int
    width_m: float
    z_min: float
    z_max: float
    ground_z: float
    # Floor levels, NAP, bottom first; the top storey runs to the eaves.
    floors: list[float] = field(default_factory=list)
    eave_z: float = 0.0
    openings: list[Opening] = field(default_factory=list)
    wall_rgb: tuple = (150, 90, 70)
    frame_rgb: tuple = (235, 232, 222)
    glass_rgb: tuple = (40, 52, 64)
    material: str = "brick"  # brick, plaster, stone
    ground: str = "none"  # shop, house, none
    # The sign band over a shop front: (z0, z1, rgb).
    fascia: tuple | None = None
    door_rgb: tuple = (52, 60, 48)
    source: str = "data"  # photo, data
    quality: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def windows(self) -> list[Opening]:
        return [o for o in self.openings if o.kind == "window"]

    @property
    def storey_m(self) -> float | None:
        """The storey height its window rows measure, if two or more rows
        stand a believable storey apart."""
        sills = sorted({round(o.z0, 2) for o in self.windows if o.z0 > self.ground_z + 2.0})
        if len(sills) < 2:
            return None
        step = float(np.median(np.diff(sills)))
        return step if 2.6 <= step <= 5.5 else None

    @property
    def ground_m(self) -> float | None:
        """The ground storey's height, where window rows above it were found."""
        if len(self.floors) > 1 and self.source == "photo":
            return float(self.floors[1] - self.floors[0])
        return None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["openings"] = [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(o).items()}
            for o in self.openings
        ]
        out["floors"] = [round(f, 3) for f in self.floors]
        for key in ("width_m", "z_min", "z_max", "ground_z", "eave_z", "quality"):
            out[key] = round(float(out[key]), 3)
        return out


# --------------------------------------------------------------------------
# Small tools
# --------------------------------------------------------------------------


def box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1) square, edges held."""
    if radius < 1:
        return array
    k = 2 * radius + 1
    c = np.cumsum(np.pad(array, ((radius + 1, radius), (0, 0)), mode="edge"), axis=0)
    out = (c[k:] - c[:-k]) / k
    c = np.cumsum(np.pad(out, ((0, 0), (radius + 1, radius)), mode="edge"), axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def otsu_threshold(values: np.ndarray, lo: float = 0.12, hi: float = 0.65) -> float:
    """The split between two populations of profile values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return 0.5 * (lo + hi)
    hist, edges = np.histogram(values, bins=40, range=(0.0, 1.0))
    centres = 0.5 * (edges[:-1] + edges[1:])
    best, split = -1.0, 0.5 * (lo + hi)
    for k in range(1, len(hist)):
        w0, w1 = hist[:k].sum(), hist[k:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:k] * centres[:k]).sum() / w0
        m1 = (hist[k:] * centres[k:]).sum() / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, split = between, edges[k]
    return float(np.clip(split, lo, hi))


def segment_runs(
    profile: np.ndarray,
    *,
    tau: float,
    min_on: int,
    max_on: int,
    min_off: int,
    cost_on: float = 0.0,
) -> list[tuple[int, int]]:
    """The best split of a profile into alternating off and on runs.

    Maximises the sum over on-runs of ``profile - tau`` and over off-runs of
    ``tau - profile``, less ``cost_on`` per on-run, with every on-run between
    `min_on` and `max_on` long and every off-run between two on-runs at
    least `min_off` long; NaN counts for neither. Exact, by dynamic
    programming over where each run ends. Returns the on-runs as
    ``(start, end)`` index pairs.
    """
    f = np.asarray(profile, dtype=np.float64)
    n = len(f)
    if n == 0 or max_on < 1:
        return []
    gain = np.where(np.isfinite(f), f - tau, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(gain)])
    neg = -1e18
    best_off = np.full(n + 1, neg)
    best_on = np.full(n + 1, neg)
    back_off = np.zeros(n + 1, dtype=np.int64)
    back_on = np.zeros(n + 1, dtype=np.int64)
    best_off[0] = 0.0
    min_on = max(1, int(min_on))
    for i in range(1, n + 1):
        lengths = np.arange(min_on, min(int(max_on), i) + 1)
        if len(lengths):
            j = i - lengths
            cand = best_off[j] + (cum[i] - cum[j]) - cost_on
            k = int(np.argmax(cand))
            best_on[i], back_on[i] = cand[k], j[k]
        j = np.arange(0, i)
        free = (j == 0) | (i == n) | ((i - j) >= min_off)
        before = np.where(j == 0, 0.0, best_on[j])
        cand = np.where(free, before, neg) - (cum[i] - cum[j])
        k = int(np.argmax(cand))
        best_off[i], back_off[i] = cand[k], j[k]
    runs = []
    i, state = n, ("off" if best_off[n] >= best_on[n] else "on")
    while i > 0:
        if state == "on":
            j = int(back_on[i])
            runs.append((j, i))
            i, state = j, "off"
        else:
            j = int(back_off[i])
            if j == 0:
                break
            i, state = j, "on"
    return runs[::-1]


def tall_runs(mask: np.ndarray, rows: int) -> np.ndarray:
    """The pixels of `mask` in a vertical run at least `rows` long.

    A stone sill, a string course or a cornice is a thin horizontal band
    unlike the brick; a tree, a sign or a person is a tall blob. This keeps
    the blobs."""
    mask = np.asarray(mask, dtype=bool)
    h = mask.shape[0]
    rows = int(rows)
    if rows <= 1 or h < rows:
        return mask.copy() if rows <= 1 else np.zeros_like(mask)
    c = np.cumsum(np.pad(mask.astype(np.int32), ((1, 0), (0, 0))), axis=0)
    full = (c[rows:] - c[:-rows]) == rows  # window starting at each row is all set
    f = np.pad(full.astype(np.int32), ((rows - 1, rows - 1), (0, 0)))
    cf = np.cumsum(np.pad(f, ((1, 0), (0, 0))), axis=0)
    return ((cf[rows:] - cf[:-rows]) > 0)[:h]


def _binned(profile: np.ndarray, size: int) -> np.ndarray:
    n = len(profile) // size
    if n == 0:
        return profile.astype(np.float64)
    block = profile[: n * size].reshape(n, size)
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(block).sum(axis=1)
        sums = np.nansum(block, axis=1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _runs_in_px(runs, size: int) -> list[tuple[int, int]]:
    return [(a * size, b * size) for a, b in runs]


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------


def _lab(rgb: np.ndarray) -> np.ndarray:
    from .facade_measure import rgb_to_lab

    return rgb_to_lab(rgb)


def _lab_to_rgb(lab: np.ndarray) -> tuple[int, int, int]:
    """CIE Lab (D65) back to sRGB 0..255, one colour."""
    L, a, b = (float(v) for v in lab)
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def inv(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16.0 / 116.0) / 7.787

    xyz = np.array([inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883])
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    c = m @ xyz
    c = np.where(c > 0.0031308, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055, 12.92 * c)
    return tuple(int(v) for v in np.clip(np.round(c * 255.0), 0, 255))


def wall_colour(labs: list[np.ndarray], weights: list[np.ndarray], rows_ok: np.ndarray):
    """The wall's own colour in Lab: the largest colour cluster among the
    well-seen pixels of the rows asked for, from every view. None when too
    little is seen."""
    from .facade_measure import kmeans

    picked = []
    for lab, weight in zip(labs, weights):
        sel = (weight > 0.5) & rows_ok[:, None]
        if sel.any():
            picked.append(lab[sel])
    if not picked:
        return None
    points = np.concatenate(picked)
    if len(points) < 300:
        return None
    rng = np.random.default_rng(0)
    points = points[rng.choice(len(points), min(len(points), 8000), replace=False)]
    centres, labels = kmeans(points, 4, seed=1)
    counts = np.bincount(labels, minlength=len(centres))
    return centres[int(np.argmax(counts))]


def band_colour(lab: np.ndarray, weight: np.ndarray, wall_lab: np.ndarray):
    """A second colour the wall is made of, or None.

    A stone plinth, a plastered ground storey, a pale string course a foot
    tall: paler than the wall's main colour, and laid in bands that run
    right across it. Glass in a row of windows is darker, and a ribbon of
    it is not taken for wall."""
    from .facade_measure import kmeans

    sel = weight > 0.5
    if sel.sum() < 2000:
        return None
    points = lab[sel]
    rng = np.random.default_rng(2)
    sample = points[rng.choice(len(points), min(len(points), 8000), replace=False)]
    centres, _ = kmeans(sample, 5, seed=3)
    best = None
    for centre in centres:
        if centre[0] < wall_lab[0] + 6.0 or np.hypot(*(centre[1:] - wall_lab[1:])) > 30.0:
            continue
        if np.sqrt(((centre - wall_lab) ** 2).sum()) < 10.0:
            continue
        near = np.sqrt(((lab - centre) ** 2).sum(-1)) < 9.0
        near &= sel
        share = near.sum() / max(1, sel.sum())
        if share < 0.08:
            continue
        per_row = near.sum(1) / np.maximum(sel.sum(1), 1)
        banded = near[per_row > 0.55].sum() / max(1, near.sum())
        if banded >= 0.5 and (best is None or share > best[1]):
            best = (centre, share)
    return None if best is None else best[0]


def opening_score(lab: np.ndarray, wall_lab: np.ndarray, m_per_px: float) -> np.ndarray:
    """0..1: how unlike the wall each pixel is. Glass is darker or bluer than
    brick or plaster, and a frame paler; brick's own spread of colour is
    well under the scale used here."""
    d_chroma = np.hypot(lab[..., 1] - wall_lab[1], lab[..., 2] - wall_lab[2])
    d_light = np.abs(lab[..., 0] - wall_lab[0])
    score = np.clip(np.maximum(d_chroma / 18.0, d_light / 40.0), 0.0, 1.0)
    return box_blur(score, max(1, int(round(0.04 / m_per_px))))


def material_of(wall_lab: np.ndarray, texture: float) -> str:
    """Brick, plaster or stone, from the wall's colour and grain.

    Painted plaster is pale and close to grey; brick is red, brown or yellow
    and has the joints' grain; stone is in between, light and warm grey."""
    L, a, b = (float(v) for v in wall_lab)
    chroma = float(np.hypot(a, b))
    # Pale and close to grey or cream, and smooth: render or paint.
    if L > 60.0 and chroma < 24.0 and a < 8.0 and texture < 9.0:
        return "plaster"
    if L > 74.0:
        return "plaster"
    # Red, brown, orange or purple, darker: brick.
    if a > 5.0 and L < 64.0:
        return "brick"
    # Grey with grain: stone.
    if chroma < 12.0 and texture >= 6.0:
        return "stone"
    return "brick" if a > 3.0 else "plaster"


# --------------------------------------------------------------------------
# Reading the structure
# --------------------------------------------------------------------------


@dataclass
class _Reading:
    """Every view's score, added up on the wall's own grid."""

    score: np.ndarray  # (h, w) 0..1, NaN where unseen
    seen: np.ndarray  # (h, w) total weight
    z_of_row: np.ndarray
    m_per_px: float

    def rows_profile(self, cols: np.ndarray | None = None, rows: np.ndarray | None = None) -> np.ndarray:
        s, w = self.score, self.seen
        if cols is not None:
            s, w = s[:, cols], w[:, cols]
        good = w > SEEN
        num = np.where(good, np.nan_to_num(s) * w, 0.0).sum(axis=1)
        den = np.where(good, w, 0.0).sum(axis=1)
        enough = good.sum(axis=1) > 0.25 * s.shape[1]
        out = np.where(enough, num / np.maximum(den, 1e-9), np.nan)
        if rows is not None:
            out = np.where(rows, out, np.nan)
        return out

    def cols_profile(self, rows: np.ndarray) -> np.ndarray:
        s, w = self.score[rows], self.seen[rows]
        good = w > SEEN
        num = np.where(good, np.nan_to_num(s) * w, 0.0).sum(axis=0)
        den = np.where(good, w, 0.0).sum(axis=0)
        enough = good.sum(axis=0) > 0.25 * max(1, int(rows.sum()))
        return np.where(enough, num / np.maximum(den, 1e-9), np.nan)

    def cell(self, r0: int, r1: int, c0: int, c1: int) -> tuple[float, float]:
        """(share seen, mean score) of a block."""
        s = self.score[r0:r1, c0:c1]
        w = self.seen[r0:r1, c0:c1]
        good = w > SEEN
        if not good.size:
            return 0.0, 0.0
        share = float(good.mean())
        if not good.any():
            return share, 0.0
        return share, float(np.average(np.nan_to_num(s[good]), weights=w[good]))


def _fit_rows(reading: _Reading, rows_ok: np.ndarray, cols: np.ndarray | None) -> list[tuple[int, int]]:
    size = max(1, int(round(BIN_M / reading.m_per_px)))
    m = reading.m_per_px * size
    profile = _binned(reading.rows_profile(cols, rows_ok), size)
    tau = otsu_threshold(profile)
    runs = segment_runs(
        profile, tau=tau, min_on=int(ROW_MIN_M / m), max_on=int(ROW_MAX_M / m),
        min_off=int(ROW_GAP_M / m), cost_on=0.6,
    )
    return _runs_in_px(runs, size)


def _fit_cols(reading: _Reading, rows: np.ndarray) -> list[tuple[int, int]]:
    size = max(1, int(round(BIN_M / reading.m_per_px)))
    m = reading.m_per_px * size
    profile = _binned(reading.cols_profile(rows), size)
    tau = otsu_threshold(profile)
    runs = segment_runs(
        profile, tau=tau, min_on=int(COLUMN_MIN_M / m), max_on=int(COLUMN_MAX_M / m),
        min_off=int(COLUMN_GAP_M / m), cost_on=0.6,
    )
    return _runs_in_px(runs, size)


def _mask(n: int, runs) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for a, b in runs:
        out[a:b] = True
    return out


def _snap(values: list[float], spread: float) -> list[float]:
    """Each value to the median of those within `spread` of it."""
    arr = np.asarray(values, dtype=np.float64)
    out = []
    for v in arr:
        near = arr[np.abs(arr - v) <= spread]
        out.append(float(np.median(near)))
    return out


def measure_layout(
    views: list[tuple[np.ndarray, np.ndarray]],
    *,
    face: int = -1,
    m_per_px: float = M_PER_PX,
    width_m: float,
    z_min: float,
    z_max: float,
    ground_z: float,
    eave_z: float,
    function: int | None = None,
    year: int | None = None,
) -> FacadeLayout | None:
    """Read one wall's layout off its straightened views ``[(rgb, weight)]``.

    All views share the wall's grid: row 0 at `z_max`, column 0 at the
    wall's left end seen from outside. None when the views see too little of
    the wall to say anything.
    """
    views = [(np.asarray(r, dtype=np.float32), np.asarray(w, dtype=np.float32)) for r, w in views]
    views = [(r, w) for r, w in views if r.ndim == 3 and w.shape == r.shape[:2] and float(w.mean()) > 0.02]
    if not views:
        return None
    h, w = views[0][1].shape
    z_of_row = z_max - (np.arange(h) + 0.5) * m_per_px
    above_ground = z_of_row > ground_z + GROUND_SKIP_M
    below_eave = z_of_row < max(eave_z, ground_z + COLOUR_FROM_M + 1.0) + 0.3
    out_of_clutter = z_of_row > ground_z + COLOUR_FROM_M

    labs = [_lab(r) for r, _ in views]
    weights = [wt for _, wt in views]
    wall_lab = wall_colour(labs, weights, out_of_clutter & below_eave)
    if wall_lab is None:
        wall_lab = wall_colour(labs, weights, np.ones(h, dtype=bool))
    if wall_lab is None:
        return None

    # The structure is read off the view that sees the wall best. The others
    # are a few centimetres off it, and added in they blur a window column
    # into the pier beside it; they only stand in where it sees nothing.
    order = sorted(range(len(views)), key=lambda k: -float(weights[k].mean()))
    score = np.full((h, w), np.nan)
    seen = np.zeros((h, w), dtype=np.float64)
    texture_vals = []
    second = band_colour(labs[order[0]], weights[order[0]], wall_lab)
    for k in order:
        lab, weight = labs[k], weights[k]
        s = opening_score(lab, wall_lab, m_per_px)
        if second is not None:
            s = np.minimum(s, opening_score(lab, second, m_per_px))
        fresh = (weight > SEEN) & (seen <= SEEN)
        score = np.where(fresh, s, score)
        seen = np.where(seen > SEEN, seen, np.maximum(seen, weight))
        # The wall's grain: the spread of lightness over a few centimetres,
        # on pixels that are wall.
        wall_px = (weight > 0.5) & (s < 0.25) & out_of_clutter[:, None]
        if wall_px.sum() > 400:
            light = lab[..., 0]
            local = np.abs(light - box_blur(light, max(1, int(round(0.06 / m_per_px)))))
            texture_vals.append(float(np.median(local[wall_px])) * 2.5)
    reading = _Reading(score=score, seen=seen, z_of_row=z_of_row, m_per_px=m_per_px)

    layout = FacadeLayout(
        face=face, width_m=float(width_m), z_min=float(z_min), z_max=float(z_max),
        ground_z=float(ground_z), eave_z=float(min(max(eave_z, ground_z + 2.5), z_max)),
        source="photo",
    )
    layout.wall_rgb = _lab_to_rgb(wall_lab)
    texture = float(np.median(texture_vals)) if texture_vals else 6.0
    layout.material = material_of(wall_lab, texture)

    # Window rows and columns, twice: each pass reads one through the other.
    upper = above_ground & below_eave
    rows = [r for r in _fit_rows(reading, upper, None)]
    cols = _fit_cols(reading, _mask(h, rows)) if rows else []
    if rows and cols:
        rows = _fit_rows(reading, upper, _mask(w, cols)) or rows
        cols = _fit_cols(reading, _mask(h, rows)) or cols

    # Cells.
    windows: list[Opening] = []
    on_score, off_score = [], []
    states = {}
    for i, (r0, r1) in enumerate(rows):
        for j, (c0, c1) in enumerate(cols):
            share, mean = reading.cell(r0, r1, c0, c1)
            states[i, j] = (share, mean)
    if states:
        means = np.array([m for s, m in states.values() if s >= 0.4])
        hi = float(np.percentile(means, 85)) if len(means) else 0.6
        wall_level = reading.rows_profile(None, upper & ~_mask(h, rows))
        lo = float(np.nanmedian(wall_level)) if np.isfinite(wall_level).any() else 0.15
        span = max(hi - lo, 1e-6)
        present = {}
        for (i, j), (share, mean) in states.items():
            level = (mean - lo) / span
            if share >= 0.4:
                present[i, j] = level >= CELL_ON if level >= CELL_ON or level < CELL_OFF else None
            else:
                present[i, j] = None
        # Unseen and doubtful cells follow their column.
        for j in range(len(cols)):
            known = [present[i, j] for i in range(len(rows)) if present[i, j] is not None]
            majority = bool(known) and (sum(known) >= 0.5 * len(known))
            for i in range(len(rows)):
                if present[i, j] is None:
                    present[i, j] = majority
        for (i, j), on in present.items():
            share, mean = states[i, j]
            if on:
                r0, r1 = rows[i]
                c0, c1 = cols[j]
                windows.append(Opening(
                    "window", c0 * m_per_px, c1 * m_per_px,
                    float(z_of_row[min(r1, h) - 1] - 0.5 * m_per_px), float(z_of_row[r0] + 0.5 * m_per_px),
                ))
                if share >= 0.4:
                    on_score.append(mean)
            elif share >= 0.4:
                off_score.append(mean)

    # Line them up: one sill and one head per row, one pair of jambs per column.
    if windows:
        sills = _snap([o.z0 for o in windows], 0.25)
        heads = _snap([o.z1 for o in windows], 0.25)
        lefts = _snap([o.x0 for o in windows], 0.15)
        rights = _snap([o.x1 for o in windows], 0.15)
        for o, s, t, a, b in zip(windows, sills, heads, lefts, rights):
            o.z0, o.z1, o.x0, o.x1 = s, t, a, b
    windows = [o for o in windows if o.x1 - o.x0 >= COLUMN_MIN_M and o.z1 - o.z0 >= ROW_MIN_M]
    # A row whose sills are near the street is the ground storey's: its
    # windows, or a shop's glass.
    ground_row = [o for o in windows if o.z0 < ground_z + GROUND_ROW_SILL_M]
    windows = [o for o in windows if o.z0 >= ground_z + GROUND_ROW_SILL_M]

    # The photos saw part of the wall: carry its columns and rows on over
    # the rest, as the facade does.
    looked_cols = np.flatnonzero((seen > SEEN)[above_ground & below_eave].any(axis=0))
    if len(looked_cols):
        seen_x = (looked_cols[0] * m_per_px, (looked_cols[-1] + 1) * m_per_px)
    else:
        seen_x = (0.0, width_m)
    looked_rows = np.flatnonzero((seen > SEEN).any(axis=1) & below_eave)
    seen_top = float(z_of_row[looked_rows[0]]) if len(looked_rows) else z_max
    windows = complete_windows(windows, width_m=width_m, seen_x=seen_x, seen_top=seen_top,
                               eave_z=layout.eave_z, ground_z=ground_z)

    # Storeys from the rows: each floor a sill height under its row's sills.
    row_sills = sorted({round(o.z0, 2) for o in windows})
    row_heads = sorted({round(o.z1, 2) for o in windows})
    if row_sills:
        ground_height = float(np.clip(row_sills[0] - SILL_HEIGHT_M - ground_z, *GROUND_STOREY_M))
    else:
        ground_height = DEFAULT_GROUND_M
    floors = [ground_z, ground_z + ground_height]
    for sill in row_sills[1:]:
        below = [hd for hd in row_heads if hd < sill]
        floor = max(sill - SILL_HEIGHT_M, (below[-1] + 0.25) if below else sill - SILL_HEIGHT_M)
        if floor - floors[-1] >= 2.2:
            floors.append(floor)
    layout.floors = [f for f in floors if f < layout.eave_z - 1.0] or [ground_z]

    # Frames and glass, from the windows' pixels.
    _read_frames(layout, windows + ground_row, views, z_of_row, m_per_px)
    wall_light = float(wall_lab[0])
    frame_light = float(_lab(np.asarray(layout.frame_rgb, dtype=np.float32)[None, None])[0, 0, 0])
    if frame_light < wall_light + 18.0:
        # A frame no paler than the wall reads as wall: what was found is
        # the glass, and the window is a frame's width larger all round.
        for o in windows + ground_row:
            o.x0, o.x1 = max(0.0, o.x0 - FRAME_M), min(width_m, o.x1 + FRAME_M)
            o.z0, o.z1 = o.z0 - FRAME_M, o.z1 + FRAME_M
        layout.notes.append("dark frames: windows widened by the frame")

    # The ground storey, on its own.
    ground_top = layout.floors[1] if len(layout.floors) > 1 else ground_z + DEFAULT_GROUND_M
    _read_ground(layout, reading, views, labs, wall_lab, ground_top, function, m_per_px, windows, ground_row)

    layout.openings = windows + [o for o in layout.openings if o.kind != "window" or o.z0 < ground_top]

    # How far to believe it. The layout has to explain the picture above the
    # ground storey: its windows unlike the wall, and the wall between them
    # like the wall. A tree, a sign or a fence in front of the wall is
    # unlike the wall where the layout says wall, and that is clutter.
    upper_rows = (z_of_row > ground_top + 0.2) & below_eave
    looked = (seen > SEEN) & upper_rows[:, None]
    inside = np.zeros((h, w), dtype=bool)
    for o in windows:
        r0 = max(0, int((z_max - o.z1) / m_per_px))
        r1 = min(h, int(np.ceil((z_max - o.z0) / m_per_px)))
        inside[r0:r1, max(0, int(o.x0 / m_per_px)):min(w, int(np.ceil(o.x1 / m_per_px)))] = True
    sc = np.nan_to_num(score)
    seen_upper = float(looked.sum() / max(1, upper_rows.sum() * w))
    # The piers: wall between the windows of a row. A tree, a sign or a
    # window the reading missed stands there; a cornice or a string course,
    # between the rows, does not.
    in_rows = np.zeros(h, dtype=bool)
    for o in windows:
        r0 = max(0, int((z_max - o.z1) / m_per_px))
        r1 = min(h, int(np.ceil((z_max - o.z0) / m_per_px)))
        in_rows[r0:r1] = True
    piers = looked & ~inside & in_rows[:, None]
    if looked.any() and (looked & inside).any() and piers.any():
        contrast = float(sc[looked & inside].mean() - sc[piers].mean())
        blobs = tall_runs((sc > 0.5) & piers, int(round(0.25 / m_per_px)))
        clutter = float(blobs[piers].mean())
    else:
        contrast, clutter = 0.0, 1.0
    widths = [o.width_m for o in windows]
    regular = 1.0 - min(1.0, float(np.std(widths) / max(np.mean(widths), 1e-6))) if widths else 0.0
    # Enough of a structure to believe: one window is a shadow as often as
    # it is a window, unless the wall is only wide enough for one; and the
    # rows have to account for a fair share of the storeys the wall's height
    # holds.
    if len(windows) >= 2:
        count_term = 1.0
    elif len(windows) == 1:
        count_term = 0.4
    else:
        count_term = 0.0
    top = min(layout.eave_z, z_max)
    expected_rows = max(1, int(round((top - ground_top) / DEFAULT_STOREY_M)))
    found_rows = len({round(o.z0, 1) for o in windows})
    rows_term = float(np.clip((found_rows / expected_rows) / 0.6, 0.0, 1.0))
    shaped = [0.7 <= o.height_m / max(o.width_m, 1e-6) <= 4.0 for o in windows]
    shape_term = 0.5 + 0.5 * (sum(shaped) / len(shaped)) if shaped else 0.0
    layout.quality = float(np.clip(
        np.clip(contrast / 0.35, 0, 1)
        * np.clip(1.0 - (clutter - 0.08) / 0.25, 0, 1)
        * np.clip(seen_upper / 0.5, 0, 1)
        * (0.6 + 0.4 * regular)
        * count_term * rows_term * shape_term,
        0.0, 1.0,
    ))
    layout.notes.append(
        f"{len(rows)} rows, {len(cols)} columns, {len(windows)} windows above the ground storey; "
        f"contrast {contrast:.2f}, clutter {clutter:.0%}, upper wall seen {seen_upper:.0%}"
    )
    return layout


def _read_frames(layout, windows, views, z_of_row, m_per_px) -> None:
    """Frame colour: the pale or dark ring just inside each window; glass:
    its middle."""
    rings, middles = [], []
    for o in windows:
        r0 = int(np.searchsorted(-z_of_row, -o.z1))
        r1 = int(np.searchsorted(-z_of_row, -o.z0))
        c0, c1 = int(o.x0 / m_per_px), int(o.x1 / m_per_px)
        if r1 - r0 < 8 or c1 - c0 < 8:
            continue
        t = max(1, int(round(0.05 / m_per_px)))
        for rgb, weight in views:
            block = rgb[r0:r1, c0:c1]
            wt = weight[r0:r1, c0:c1]
            ring = np.zeros(block.shape[:2], dtype=bool)
            ring[:t, :] = ring[-t:, :] = ring[:, :t] = ring[:, -t:] = True
            inner = np.zeros_like(ring)
            inner[2 * t:-2 * t, 2 * t:-2 * t] = True
            good = wt > 0.5
            if (ring & good).sum() > 20:
                rings.append(block[ring & good])
            if (inner & good).sum() > 20:
                middles.append(block[inner & good])
    if rings:
        ring = np.concatenate(rings)
        light = ring @ np.array([0.299, 0.587, 0.114])
        # The frame is the paler part of the ring: the rest is glass.
        pale = ring[light >= np.percentile(light, 60)]
        frame = np.median(pale, axis=0)
        wall = np.asarray(layout.wall_rgb, dtype=np.float64)
        # A ring the colour of the wall is wall the window was cut a little
        # wide into: no frame was seen, and the era's stays.
        if np.abs(frame - wall).max() > 28.0:
            layout.frame_rgb = tuple(int(v) for v in frame)
    if middles:
        # Glass in a photo is the sky or the room behind it, whichever the
        # light gave; drawn, it is dark glass with a hint of that.
        seen = np.median(np.concatenate(middles), axis=0)
        base = np.array([44.0, 56.0, 68.0])
        layout.glass_rgb = tuple(int(v) for v in np.clip(0.75 * base + 0.25 * seen, 0, 90))


def _read_ground(layout, reading, views, labs, wall_lab, ground_top, function, m_per_px, windows, ground_row=()) -> None:
    """The ground storey: a shop front or a house front."""
    z = reading.z_of_row
    h, w = reading.score.shape
    width = layout.width_m
    if layout.z_min > layout.ground_z + 1.0:
        layout.ground = "none"  # this face does not reach the street
        return
    band = (z > layout.ground_z + 0.4) & (z < ground_top - 0.5)
    profile = reading.cols_profile(band) if band.any() else np.full(w, np.nan)
    size = max(1, int(round(BIN_M / m_per_px)))
    binned = _binned(profile, size)
    seen_share = float(np.isfinite(binned).mean()) if len(binned) else 0.0
    glass_runs = []
    if seen_share > 0.3:
        tau = otsu_threshold(binned)
        m = m_per_px * size
        glass_runs = _runs_in_px(segment_runs(
            binned, tau=tau, min_on=int(0.7 / m), max_on=int(8.0 / m), min_off=int(0.2 / m), cost_on=0.8,
        ), size)
    glass_share = sum(b - a for a, b in glass_runs) * m_per_px / max(width, 1e-6)
    retail = function == 1  # see usage.py
    home = function in (0, 4)  # a house, or a public building: windows and a door
    if retail or (not home and glass_share > SHOP_GLASS_SHARE):
        layout.ground = "shop"
    elif home or function is None:
        layout.ground = "house"
    else:
        layout.ground = "shop" if glass_share > SHOP_GLASS_SHARE else "house"

    upper_cols = sorted({(round(o.x0, 2), round(o.x1, 2)) for o in windows})
    g0 = layout.ground_z
    if layout.ground == "shop":
        fascia_z0 = ground_top - FASCIA_M - 0.1
        runs = [(a * m_per_px, b * m_per_px) for a, b in glass_runs if (b - a) * m_per_px >= 0.7]
        if not runs or sum(b - a for a, b in runs) < 0.35 * width:
            # Too cluttered to read: glass across the front between end piers,
            # broken where the piers of the storeys above come down.
            runs = _glass_between(upper_cols, width)
        for a, b in runs:
            layout.openings.append(Opening("shop", a, b, g0 + BULKHEAD_M, fascia_z0))
        # The shop's door: in the widest pane, at its side nearest the middle.
        widest = max(runs, key=lambda r: r[1] - r[0]) if runs else None
        if widest is not None and widest[1] - widest[0] > DOOR_W_M + 0.6:
            mid = 0.5 * width
            x0 = widest[0] if abs(widest[0] - mid) < abs(widest[1] - mid) else widest[1] - DOOR_W_M
            layout.openings.append(Opening("door", x0, x0 + DOOR_W_M, g0, g0 + DOOR_H_M + 0.2))
        # The fascia: the colour of the band over the glass where it was seen.
        rows = (z > fascia_z0) & (z < ground_top - 0.1)
        layout.fascia = (fascia_z0, ground_top - 0.1, _band_colour(views, rows, layout.wall_rgb))
        return

    # A house front: a door, and windows: those the photos saw down there,
    # or else under the columns above.
    door_x = _door_position(reading, layout, upper_cols, m_per_px)
    layout.openings.append(Opening("door", door_x, door_x + DOOR_W_M, g0, g0 + DOOR_H_M))
    seen_below = [o for o in ground_row if o.z1 <= ground_top + 0.1]
    columns = [(o.x0, o.x1) for o in seen_below] or upper_cols or _even_columns(width, max(1, int(width // 2.2)))
    sill = float(np.median([o.z0 for o in seen_below])) if seen_below else g0 + 0.9
    head = float(np.median([o.z1 for o in seen_below])) if seen_below else ground_top - 0.45
    for a, b in columns:
        if b > door_x - 0.15 and a < door_x + DOOR_W_M + 0.15:
            continue
        layout.openings.append(Opening("window", a, b, max(sill, g0 + 0.4), min(head, ground_top - 0.2)))


def _even_columns(width: float, n: int) -> list[tuple[float, float]]:
    bay = width / n
    win = min(1.3, 0.55 * bay)
    return [((k + 0.5) * bay - win / 2, (k + 0.5) * bay + win / 2) for k in range(n)]


def _glass_between(upper_cols, width: float) -> list[tuple[float, float]]:
    pier = 0.35
    if len(upper_cols) >= 2:
        # Piers under the piers above: glass from midway between columns.
        edges = [pier] + [0.5 * (upper_cols[k][1] + upper_cols[k + 1][0]) for k in range(len(upper_cols) - 1)] + [width - pier]
        runs = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a > 0.8:
                runs.append((a + 0.12, b - 0.12))
        return runs
    return [(pier, width - pier)] if width > 2 * pier + 0.8 else []


def _door_position(reading, layout, upper_cols, m_per_px) -> float:
    """Where the door is: the darkest tall slot down to the street, if a
    photo sees one; else under the first column, as a canal house has it."""
    z = reading.z_of_row
    rows = (z > layout.ground_z + 0.2) & (z < layout.ground_z + 2.0)
    width = layout.width_m
    if rows.any():
        profile = reading.cols_profile(rows)
        n = int(round(DOOR_W_M / m_per_px))
        if np.isfinite(profile).sum() > 0.5 * len(profile) and len(profile) > n:
            kernel = np.ones(n) / n
            filled = np.where(np.isfinite(profile), profile, np.nanmedian(profile))
            smooth = np.convolve(filled, kernel, mode="valid")
            k = int(np.argmax(smooth))
            if smooth[k] > np.nanmedian(filled) + 0.15:
                return float(np.clip(k * m_per_px, 0.1, max(0.1, width - DOOR_W_M - 0.1)))
    if upper_cols:
        a, b = upper_cols[0]
        return float(np.clip(0.5 * (a + b) - DOOR_W_M / 2, 0.1, max(0.1, width - DOOR_W_M - 0.1)))
    return float(max(0.1, 0.5 * width - DOOR_W_M / 2))


def _band_colour(views, rows: np.ndarray, fallback) -> tuple:
    picked = []
    for rgb, weight in views:
        sel = (weight > 0.5) & rows[:, None]
        if sel.sum() > 50:
            picked.append(rgb[sel])
    if not picked:
        return tuple(int(v * 0.45) for v in fallback)
    return tuple(int(v) for v in np.median(np.concatenate(picked), axis=0))


def complete_windows(windows, *, width_m, seen_x, seen_top, eave_z, ground_z) -> list[Opening]:
    """Carry the measured columns across what no photo saw, and the
    measured rows up to the eaves.

    A photo from across a street sees six metres of a twenty-metre front,
    or the bottom three storeys of five. A facade repeats its bays and its
    storeys; where it was not seen it is drawn on in the same rhythm, never
    over what was seen."""
    if not windows:
        return windows
    out = list(windows)
    cols = sorted({(round(o.x0, 3), round(o.x1, 3)) for o in windows})
    rows = sorted({(round(o.z0, 3), round(o.z1, 3)) for o in windows})
    if len(cols) >= 2:
        centres = np.array([0.5 * (a + b) for a, b in cols])
        steps = np.diff(centres)
        bay = float(np.median(steps))
        regular = bay > 0.8 and np.all(np.abs(steps - bay) < 0.35 * bay)
        if regular:
            half = 0.5 * float(np.median([b - a for a, b in cols]))
            extra = []
            c = centres[0] - bay
            while c - half > 0.25 and c + half < seen_x[0] + 0.05:
                extra.append(c)
                c -= bay
            c = centres[-1] + bay
            while c + half < width_m - 0.25 and c - half > seen_x[1] - 0.05:
                extra.append(c)
                c += bay
            for c in extra:
                for z0, z1 in rows:
                    out.append(Opening("window", c - half, c + half, z0, z1))
    if len(rows) >= 2 and seen_top < eave_z - 1.0:
        sills = np.array([a for a, _ in rows])
        storey = float(np.median(np.diff(sills)))
        if 2.4 <= storey <= 4.5:
            top_sill, top_head = rows[-1]
            height = top_head - top_sill
            z = top_sill + storey
            columns = sorted({(round(o.x0, 3), round(o.x1, 3)) for o in out})
            while z + height < eave_z - 0.3 and z > seen_top - 0.2:
                for a, b in columns:
                    out.append(Opening("window", a, b, z, z + height))
                z += storey
    return out


# --------------------------------------------------------------------------
# Laying out a wall from the building's data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Era:
    name: str
    until: int
    ground_house: float
    ground_shop: float
    storey: float
    window_w: float
    bay: float
    window_h_frac: float
    sill: float
    frame_rgb: tuple
    panes: str  # "sash", "casement", "plain", "ribbon"


ERAS = (
    _Era("historic", 1920, 3.8, 4.2, 3.4, 1.15, 2.15, 0.62, 0.75, (236, 233, 224), "sash"),
    _Era("interbellum", 1945, 3.0, 3.6, 2.9, 1.40, 2.40, 0.56, 0.85, (230, 226, 214), "casement"),
    _Era("postwar", 1975, 2.9, 3.4, 2.8, 1.80, 2.70, 0.50, 0.95, (226, 226, 222), "plain"),
    _Era("modern", 2000, 2.8, 3.4, 2.8, 1.60, 2.60, 0.55, 0.90, (120, 104, 92), "casement"),
    _Era("contemporary", 9999, 3.2, 4.0, 3.0, 1.80, 2.40, 0.70, 0.50, (70, 74, 78), "plain"),
)


def era_for(year: int | None) -> _Era:
    year = int(year) if year else 1900
    for era in ERAS:
        if year < era.until:
            return era
    return ERAS[-1]


def layout_from_data(
    *,
    face: int,
    width_m: float,
    z_min: float,
    z_max: float,
    ground_z: float,
    eave_z: float,
    function: int | None = None,
    year: int | None = None,
    floors: int | None = None,
    wall_rgb: tuple | None = None,
    material: str | None = None,
    frame_rgb: tuple | None = None,
    storey_m: float | None = None,
    ground_m: float | None = None,
    footprint_m2: float = 0.0,
    entrance: bool = True,
    seed: int = 0,
) -> FacadeLayout:
    """A wall laid out from what the registers say about its building.

    Its storeys from its height to the eaves -- 3DBAG's own storey count
    when that is believable, else the era's storey height -- its windows
    centred in bays of the era's width, a gable's attic windows above the
    eaves, and its ground storey from the BAG: a shop front for a shop or
    a public building, a door and windows for a house. Colours are the
    era's, or what a photo measured."""
    era = era_for(year)
    rng = np.random.default_rng(seed)
    eave = float(min(max(eave_z, ground_z + 2.6), z_max))
    layout = FacadeLayout(
        face=face, width_m=float(width_m), z_min=float(z_min), z_max=float(z_max),
        ground_z=float(ground_z), eave_z=eave, source="data",
    )
    shop = function == 1  # retail, see usage.py
    layout.ground = "none" if z_min > ground_z + 1.0 else ("shop" if shop else "house")
    if material is None:
        material = "brick" if era.name in ("historic", "interbellum", "modern") else (
            "plaster" if era.name == "postwar" else "brick")
    layout.material = material
    if wall_rgb is None:
        from .facade import STYLES

        style = next((st for st in STYLES if st.name == era.name), STYLES[0])
        choices = [style.wall_rgb, *getattr(style, "wall_alternates", ())]
        wall_rgb = tuple(int(v) for v in choices[int(rng.integers(len(choices)))])
    layout.wall_rgb = tuple(int(v) for v in wall_rgb)
    layout.frame_rgb = tuple(int(v) for v in (frame_rgb or era.frame_rgb))

    # Storeys. A shop, an office or a public building from before 1920,
    # and anything that old over fifteen metres to its eaves, was built
    # with storeys a metre taller than a house.
    ground = era.ground_shop if shop else era.ground_house
    typical = era.storey
    height = eave - ground_z
    grand = era.name == "historic" and (function in (1, 2, 4) or height >= 15.0)
    if grand:
        ground, typical = max(ground, 4.5), 4.0
        if footprint_m2 >= 600.0:
            # A warehouse, a bank, an exchange: a palace of a building.
            ground, typical = 5.5, 5.0
    if ground_m:
        ground = float(np.clip(ground_m, *GROUND_STOREY_M))
    if storey_m:
        # What the photos of the building measured beats any rule.
        typical = float(np.clip(storey_m, 2.6, 5.5))
    if height < ground + 1.5:
        ground = max(2.4, min(ground, height - 0.4))
        uppers = 0
    else:
        uppers = max(0, int(round((height - ground) / typical)))
        # 3DBAG's own count, where it is believable for the building's age
        # and nothing was measured: before 1920 a storey is rarely under
        # 3.2 m, and a grand one is near four.
        lowest = 3.7 if grand else (3.2 if era.name == "historic" else 2.6)
        if not storey_m and floors and floors > 1 and lowest <= height / floors <= 4.6:
            uppers = floors - 1
    storey = (height - ground) / uppers if uppers else typical
    if uppers and not 2.5 <= storey <= 5.6:
        uppers = max(1, int(round((height - ground) / typical)))
        storey = (height - ground) / uppers
    layout.floors = [ground_z] + [ground_z + ground + k * storey for k in range(uppers)]

    # Columns: bays of the era's width, windows centred in them.
    margin = 0.45
    usable = width_m - 2 * margin
    if usable < 1.2:
        columns = []  # a strip of wall between two others: no room for a window
    else:
        bay_m = era.bay * (1.2 if grand else 1.0)
        n = max(1, int(round(usable / bay_m)))
        bay = usable / n
        w = max(0.6, min(era.window_w * (1.2 if grand else 1.0), 0.62 * bay))
        columns = [(margin + (k + 0.5) * bay - w / 2, margin + (k + 0.5) * bay + w / 2) for k in range(n)]
    window_h = era.window_h_frac * storey
    for floor in layout.floors[1:]:
        z0 = floor + era.sill
        z1 = min(z0 + window_h, floor + storey - 0.25)
        if z1 > eave - 0.1 or z1 - z0 < 0.6:
            continue
        for a, b in columns:
            layout.openings.append(Opening("window", a, b, z0, z1))

    # A gable over the eaves gets its attic windows.
    gable = z_max - eave
    if gable > 2.4 and width_m > 2.0:
        aw, ah = min(0.9, 0.3 * width_m), min(1.1, 0.4 * gable)
        z0 = eave + 0.5
        layout.openings.append(Opening("window", width_m / 2 - aw / 2, width_m / 2 + aw / 2, z0, z0 + ah))
        if gable > 5.0:
            z0 = eave + 0.5 + ah + 0.8
            layout.openings.append(Opening("window", width_m / 2 - aw * 0.35, width_m / 2 + aw * 0.35,
                                           z0, z0 + ah * 0.7))

    # The ground storey.
    top = layout.floors[1] if len(layout.floors) > 1 else min(eave, ground_z + ground)
    g0 = ground_z
    if layout.ground == "shop":
        fascia_z0 = top - FASCIA_M - 0.1
        runs = _glass_between(columns, width_m) if columns else ([(0.3, width_m - 0.3)] if width_m > 1.4 else [])
        for a, b in runs:
            layout.openings.append(Opening("shop", a, b, g0 + BULKHEAD_M, fascia_z0))
        if runs and entrance:
            widest = max(runs, key=lambda r: r[1] - r[0])
            if widest[1] - widest[0] > DOOR_W_M + 0.6:
                x0 = 0.5 * (widest[0] + widest[1]) - DOOR_W_M / 2
                layout.openings.append(Opening("door", x0, x0 + DOOR_W_M, g0, g0 + DOOR_H_M + 0.2))
        fascia_rgb = tuple(int(v) for v in rng.choice(
            np.array([(38, 40, 44), (64, 32, 30), (30, 52, 44), (196, 190, 176), (44, 56, 82)]), axis=0))
        layout.fascia = (fascia_z0, top - 0.1, fascia_rgb)
    elif layout.ground == "house" and columns:
        # A door to every house: one along a narrow front, at its side;
        # a long front of row houses has one every three or so bays.
        bays = len(columns)
        if not entrance:
            doors = []
        elif function == 0 and bays > 3:
            doors = list(range(0, bays, 3 if bays > 4 else 2))
        elif bays <= 3:
            doors = [0]
        else:
            doors = [bays // 2]  # one entrance, in the middle
        for k, (a, b) in enumerate(columns):
            if k in doors:
                c = 0.5 * (a + b)
                layout.openings.append(Opening("door", c - DOOR_W_M / 2, c + DOOR_W_M / 2, g0, g0 + DOOR_H_M))
            else:
                layout.openings.append(Opening("window", a, b, g0 + 0.9, max(g0 + 1.8, top - 0.45)))
        layout.door_rgb = tuple(int(v) for v in rng.choice(
            np.array([(40, 58, 46), (52, 38, 32), (30, 32, 36), (96, 36, 32), (34, 48, 74)]), axis=0))
    layout.quality = 0.0
    layout.notes.append(f"from the registers: {era.name}, {len(layout.floors)} storeys, {len(columns)} bays")
    return layout


# The darkest a wall of each material plausibly is, in CIE lightness. Street
# photos are exposed for the sky and a wall in shade comes out far darker
# than it stands: the pale stone of the Winkel van Sinkel read as charcoal.
LIGHTEST_FLOOR = {"plaster": 62.0, "stone": 62.0, "brick": 36.0}


def plausible_colour(rgb, material: str) -> tuple[int, int, int]:
    """A measured wall colour with its hue kept and its lightness raised to
    what its material plausibly is."""
    lab = _lab(np.asarray(rgb, dtype=np.float32)[None, None])[0, 0].astype(np.float64)
    floor = LIGHTEST_FLOOR.get(material, 36.0)
    if lab[0] >= floor:
        return tuple(int(v) for v in rgb)
    lab[0] = floor
    return _lab_to_rgb(lab)


def merged_layout(measured: FacadeLayout | None, data: FacadeLayout, *, min_quality: float) -> FacadeLayout:
    """The measured layout when it is good enough to draw; otherwise the
    register's, wearing whatever the photos did read reliably: the wall's
    colour and material, the frames' colour."""
    if measured is not None and measured.quality >= min_quality:
        measured.wall_rgb = plausible_colour(measured.wall_rgb, measured.material)
        return measured
    if measured is not None and measured.wall_rgb:
        data.wall_rgb = plausible_colour(measured.wall_rgb, measured.material)
        data.material = measured.material
        data.frame_rgb = measured.frame_rgb
        data.source = "data+colour"
    return data


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

RELIEF_WALL = 0.50
RELIEF_FRAME = 0.47
RELIEF_GLASS = 0.26
RELIEF_SILL = 0.62
RELIEF_PLINTH = 0.55
RELIEF_DOOR = 0.36
RELIEF_FASCIA = 0.58

MASONRY = {
    ("brick", "historic"): "medieval_red_brick",
    ("brick", "interbellum"): "brick_wall_10",
    ("brick", "postwar"): "brick_wall_006",
    ("brick", "modern"): "brick_wall_006",
    ("brick", "contemporary"): "brick_wall_006",
    ("plaster", None): "painted_plaster_wall",
    ("stone", None): "sandstone_blocks_04",
}


class Masonry:
    """Photographed wall surfaces at their true size, tinted, cached."""

    def __init__(self, work_dir=None, *, tint: float = 0.8):
        self.work_dir = work_dir
        self.tint = tint
        self._cache: dict[str, tuple[np.ndarray, float, float] | None] = {}

    def _load(self, slug: str):
        if slug in self._cache:
            return self._cache[slug]
        found = None
        if self.work_dir is not None:
            try:
                from PIL import Image

                from .textures import fetch_texture, texture_size_m

                path = fetch_texture(slug, self.work_dir)
                w_m, h_m = texture_size_m(slug, self.work_dir)
                with Image.open(path) as image:
                    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
                found = (pixels, float(w_m), float(h_m))
            except Exception as exc:  # noqa: BLE001 - drawn walls fall back to a plain wall
                LOG.debug("no masonry %s: %s", slug, exc)
        self._cache[slug] = found
        return found

    def wall(self, material: str, era: str, rgb: tuple, shape: tuple[int, int], m_per_px: float,
             seed: int = 0) -> np.ndarray:
        h, w = shape
        slug = MASONRY.get((material, era)) or MASONRY.get((material, None)) or "medieval_red_brick"
        loaded = self._load(slug)
        target = np.asarray(rgb, dtype=np.float32)
        if loaded is None:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, 1.0, (max(2, h // 8 + 2), max(2, w // 8 + 2))).astype(np.float32)
            noise = np.kron(noise, np.ones((8, 8), dtype=np.float32))[:h, :w]
            return np.clip(target[None, None, :] * (1.0 + 0.05 * noise[..., None]), 0, 255)
        pixels, w_m, h_m = loaded
        ph, pw = pixels.shape[:2]
        rng = np.random.default_rng(seed)
        ox, oy = rng.integers(pw), rng.integers(ph)
        ys = ((np.arange(h) * m_per_px / max(h_m, 0.1) * ph).astype(np.int64) + oy) % ph
        xs = ((np.arange(w) * m_per_px / max(w_m, 0.1) * pw).astype(np.int64) + ox) % pw
        tile = pixels[np.ix_(ys, xs)]
        mean = tile.reshape(-1, 3).mean(axis=0)
        mean[mean < 1.0] = 1.0
        scaled = tile * (target / mean)
        return np.clip(tile * (1.0 - self.tint) + scaled * self.tint, 0, 255)


def _shade(rgb, factor):
    return tuple(int(np.clip(v * factor, 0, 255)) for v in rgb)


def draw_layout(
    layout: FacadeLayout,
    m_per_px: float = 0.04,
    *,
    masonry: Masonry | None = None,
    era: str | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """The wall as an image ``(rgb uint8, relief 0..1)``, row 0 at its top.

    Every part is drawn in metres: frames 6 cm, glazing bars 4 cm, a sill
    8 cm tall standing 5 cm proud of the window each side, so a measured
    wall and a laid-out one come out at the same level of detail."""
    rng = np.random.default_rng(seed)
    h = max(1, int(np.ceil((layout.z_max - layout.z_min) / m_per_px)))
    w = max(1, int(np.ceil(layout.width_m / m_per_px)))
    era_name = era or "historic"
    if masonry is None:
        masonry = Masonry(None)
    canvas = masonry.wall(layout.material, era_name, layout.wall_rgb, (h, w), m_per_px, seed=seed)
    relief = np.full((h, w), RELIEF_WALL, dtype=np.float32)

    def rows(z0, z1):
        r0 = int(np.clip(np.floor((layout.z_max - z1) / m_per_px), 0, h))
        r1 = int(np.clip(np.ceil((layout.z_max - z0) / m_per_px), 0, h))
        return r0, r1

    def cols(x0, x1):
        return int(np.clip(np.floor(x0 / m_per_px), 0, w)), int(np.clip(np.ceil(x1 / m_per_px), 0, w))

    px = lambda m: max(1, int(round(m / m_per_px)))  # noqa: E731

    # A plinth at the street, a cornice at the eaves, bands at the floors.
    if layout.z_min < layout.ground_z + 0.6:
        r0, r1 = rows(layout.ground_z - 0.5, layout.ground_z + 0.45)
        canvas[r0:r1] = canvas[r0:r1] * 0.78
        relief[r0:r1] = RELIEF_PLINTH
    if layout.eave_z < layout.z_max - 0.3 or abs(layout.eave_z - layout.z_max) < 0.3:
        r0, r1 = rows(layout.eave_z - 0.32, layout.eave_z)
        canvas[r0:r1] = np.clip(canvas[r0:r1] * 0.82 + 22, 0, 255)
        relief[r0:r1] = RELIEF_SILL
    if layout.material == "brick":
        for floor in layout.floors[1:]:
            r0, r1 = rows(floor - 0.08, floor)
            canvas[r0:r1] = canvas[r0:r1] * 0.9
    if layout.fascia:
        z0, z1, rgb = layout.fascia
        r0, r1 = rows(z0, z1)
        canvas[r0:r1] = np.asarray(rgb, dtype=np.float32)
        relief[r0:r1] = RELIEF_FASCIA
        # The shop's name: a paler block in the middle of the board.
        c0, c1 = cols(0.3 * layout.width_m, 0.7 * layout.width_m)
        q0, q1 = r0 + (r1 - r0) // 3, r1 - (r1 - r0) // 3
        if q1 > q0 and c1 > c0:
            canvas[q0:q1, c0:c1] = np.clip(np.asarray(rgb, np.float32) * 0.4 + 150, 0, 255)

    frame = np.asarray(layout.frame_rgb, dtype=np.float32)
    glass = np.asarray(layout.glass_rgb, dtype=np.float32)
    sill_rgb = np.asarray((206, 202, 192), dtype=np.float32)
    panes = era_for({"historic": 1900, "interbellum": 1930, "postwar": 1960, "modern": 1985}.get(era_name, 2010)).panes

    for o in sorted(layout.openings, key=lambda o: o.kind != "shop"):
        r0, r1 = rows(o.z0, o.z1)
        c0, c1 = cols(o.x0, o.x1)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        f = px(0.06)
        if o.kind == "door":
            door = np.asarray(layout.door_rgb, dtype=np.float32)
            canvas[r0:r1, c0:c1] = frame * 0.9
            relief[r0:r1, c0:c1] = RELIEF_FRAME
            # A fanlight over a house door.
            fan = px(0.45) if layout.ground == "house" else 0
            if fan and r1 - r0 > 3 * fan:
                canvas[r0 + f:r0 + fan, c0 + f:c1 - f] = glass * 1.1
                relief[r0 + f:r0 + fan, c0 + f:c1 - f] = RELIEF_GLASS
            d0 = r0 + fan + f
            if layout.ground == "shop":
                canvas[d0:r1, c0 + f:c1 - f] = glass * 1.25
                relief[d0:r1, c0 + f:c1 - f] = RELIEF_GLASS
            else:
                canvas[d0:r1, c0 + f:c1 - f] = door
                relief[d0:r1, c0 + f:c1 - f] = RELIEF_DOOR
                # Two panels.
                mid = (c0 + c1) // 2
                canvas[d0:r1, mid:mid + max(1, f // 2)] = door * 0.75
            continue
        if o.kind == "shop":
            # Glass on a bulkhead, slim dark mullions, a warm interior.
            canvas[r0:r1, c0:c1] = np.asarray((46, 46, 48), dtype=np.float32)
            relief[r0:r1, c0:c1] = RELIEF_FRAME
            g = max(1, f // 2)
            inner = canvas[r0 + g:r1 - g, c0 + g:c1 - g]
            ramp = np.linspace(1.0, 0.75, max(1, inner.shape[0]))[:, None, None]
            warm = np.asarray((92, 84, 72), dtype=np.float32)
            inner[:] = (0.55 * glass + 0.45 * warm) * ramp
            relief[r0 + g:r1 - g, c0 + g:c1 - g] = RELIEF_GLASS
            pane = px(1.6)
            for c in range(c0 + pane, c1 - pane // 2, pane):
                canvas[r0:r1, c:c + g] = 46
                relief[r0:r1, c:c + g] = RELIEF_FRAME
            b0, b1 = rows(layout.ground_z, o.z0)
            canvas[b0:b1, c0:c1] = canvas[b0:b1, c0:c1] * 0.6
            continue
        # A window: frame, glass with the sky in its upper panes, bars.
        canvas[r0:r1, c0:c1] = frame
        relief[r0:r1, c0:c1] = RELIEF_FRAME
        g0, g1, h0, h1 = r0 + f, r1 - f, c0 + f, c1 - f
        if g1 - g0 < 2 or h1 - h0 < 2:
            continue
        ramp = np.linspace(1.4, 0.72, g1 - g0)[:, None, None]
        canvas[g0:g1, h0:h1] = glass * ramp * (1.0 + 0.08 * (rng.random() - 0.5))
        relief[g0:g1, h0:h1] = RELIEF_GLASS
        bar = max(1, px(0.04))
        if panes in ("sash", "casement") and h1 - h0 > 6 * bar:
            mid = (h0 + h1) // 2
            canvas[g0:g1, mid - bar // 2:mid - bar // 2 + bar] = frame
            relief[g0:g1, mid - bar // 2:mid - bar // 2 + bar] = RELIEF_FRAME
        if panes == "sash" and g1 - g0 > 8 * bar:
            for frac in (0.5,):
                r = g0 + int(frac * (g1 - g0))
                canvas[r:r + bar, h0:h1] = frame
                relief[r:r + bar, h0:h1] = RELIEF_FRAME
        elif panes == "casement" and g1 - g0 > 8 * bar:
            r = g0 + int(0.3 * (g1 - g0))
            canvas[r:r + bar, h0:h1] = frame
            relief[r:r + bar, h0:h1] = RELIEF_FRAME
        # Sill and lintel.
        s0, s1 = rows(o.z0 - 0.08, o.z0)
        e0, e1 = cols(o.x0 - 0.05, o.x1 + 0.05)
        canvas[s0:s1, e0:e1] = sill_rgb
        relief[s0:s1, e0:e1] = RELIEF_SILL
        if layout.material == "brick":
            l0, l1 = rows(o.z1, o.z1 + 0.12)
            canvas[l0:l1, e0:e1] = canvas[l0:l1, e0:e1] * 0.85
    return np.clip(canvas, 0, 255).astype(np.uint8), relief


__all__ = [
    "ERAS",
    "FacadeLayout",
    "M_PER_PX",
    "Masonry",
    "Opening",
    "complete_windows",
    "draw_layout",
    "era_for",
    "layout_from_data",
    "measure_layout",
    "merged_layout",
    "otsu_threshold",
    "segment_runs",
]
