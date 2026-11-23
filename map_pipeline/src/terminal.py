"""What stands on a terminal: cranes on their rails and containers in stacks.

Neither is in the open data as an object. What is there:

* **Crane rails.** The BGT draws them (``spoor``, plus-function
  ``(haven)kraan``), one line per rail. Two parallel rails side by side are a
  crane track, with a gauge.
* **The surface model.** It was flown over the cranes and the stacks as they
  stood that day, at half a metre. A crane is a cluster of returns over
  `CRANE_MIN_M` above the paving on a crane track; a container stack is a
  flat top at a whole number of container heights above it.

So a crane is built where the survey saw one, as high as it saw it and of the
kind its footprint says: across its rails and no further, a portal crane; on
a wide track with its top far up, a ship-to-shore crane with its boom down
over the water or up; otherwise a slewing crane with its jib where the survey
saw the tip. And a stack is built where the survey saw one, as many
containers high. Both are the day of the flight, which is as good a day as
any other.

This module only reads; `harbour_kit` has the geometry and `harbour` puts
the two together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .harbour_kit import CONTAINER_20_M, CONTAINER_HEIGHT_M, CONTAINER_WIDTH_M

LOG = logging.getLogger(__name__)

# Two rails are one track when they run this parallel, this far apart at
# most, side by side over this share of the shorter one.
PARALLEL_DEG = 3.0
MIN_GAUGE_M = 2.5
MAX_GAUGE_M = 60.0
MIN_OVERLAP = 0.5
# A rail that bends more than this off its chord is drawn but carries no crane.
STRAIGHT_M = 1.0

# Returns this far above the paving are a crane: a stack of six is 15.5 m.
# On a track with no stacks between its rails, a shipyard's, this far.
CRANE_MIN_M = 18.0
CRANE_LOW_M = 10.0
STACKED_SHARE = 0.05
# Longer than this along its rails it is no crane but a shed over them, and
# a portal crane is no higher than this many times its gauge.
CRANE_MAX_RUN_M = 60.0
GANTRY_MAX_RATIO = 2.5
# How far either side of its rails a crane's legs and bogies reach.
CORRIDOR_M = 1.5
# The least of a crane the survey has to have seen, in half-metre cells.
CRANE_MIN_CELLS = 12
# Cells of one crane are no further apart along the track than this.
CRANE_GAP_M = 3.0
# A ship-to-shore crane: a track at least this wide with a top this high.
QUAY_CRANE_GAUGE_M = 15.0
QUAY_CRANE_MIN_M = 40.0
QUAY_CRANE_WIDTH_M = 27.0
# How far across its track a crane is looked for: a jib, a boom.
REACH_M = 45.0
QUAY_CRANE_REACH_M = 80.0

# A stack's top is a whole number of containers, each 2.59 m or a high cube's
# 2.90 m, up to this many.
MAX_TIERS = 6
TIER_M = 2.7
# Cells of one flat top differ by no more than this from their neighbours.
FLAT_M = 0.4
# The least flat top worth a container, and the fewest containers that make
# a yard: under that, they are a bus and two site huts.
MIN_STACK_M2 = 5.0
MIN_CONTAINERS = 10
MAX_CONTAINERS = 60000


@dataclass
class Track:
    """Two crane rails: the line midway between them, and their gauge."""

    start: np.ndarray
    along: np.ndarray
    length: float
    gauge: float
    level: int = 0

    @property
    def across(self) -> np.ndarray:
        return np.array([-self.along[1], self.along[0]])

    def frame(self, xy: np.ndarray):
        """Along and across the track, from its start and its middle line."""
        d = np.asarray(xy, dtype=np.float64) - self.start
        return d @ self.along, d @ self.across

    def at(self, s: float, t: float = 0.0) -> np.ndarray:
        return self.start + s * self.along + t * self.across


@dataclass
class Crane:
    kind: str  # "gantry", "jib" or "quay"
    track: Track
    centre: np.ndarray
    height_m: float  # its top, above the paving
    # gantry
    overhang_m: float = 0.0
    # jib: the jib's head, in plan and above the paving
    tip: np.ndarray | None = None
    tip_m: float = 0.0
    # quay
    water: np.ndarray | None = None
    boom_m: float = 0.0
    outreach_m: float = 0.0
    backreach_m: float = 0.0
    boom_up: bool = False


@dataclass
class Stack:
    """Containers on top of each other: one footprint, `tiers` high."""

    centre: np.ndarray
    along: np.ndarray
    length: float
    tiers: int


@dataclass
class HeightField:
    """Height above the paving on the surface model's grid; NaN is no return."""

    height: np.ndarray
    transform: object

    def xy(self, rows, cols) -> np.ndarray:
        t = self.transform
        return np.column_stack([t.c + (np.asarray(cols) + 0.5) * t.a,
                                t.f + (np.asarray(rows) + 0.5) * t.e])

    @property
    def cell_m(self) -> float:
        return float(abs(self.transform.a))

    @property
    def flat(self) -> np.ndarray:
        """The cells level with all their neighbours: a roof, a container."""
        if getattr(self, "_flat", None) is None:
            with np.errstate(invalid="ignore"):
                self._flat = _spread(self.height) <= FLAT_M
        return self._flat

    def mask(self, shapes, *, inside: bool = True) -> np.ndarray:
        """The cells whose centres are in `shapes`."""
        from rasterio.features import geometry_mask

        shapes = [g for g in shapes if g is not None and not g.is_empty]
        if not shapes:
            return np.zeros(self.height.shape, dtype=bool)
        return geometry_mask(
            [g.__geo_interface__ for g in shapes], self.height.shape, self.transform,
            invert=inside,
        )


def pair_tracks(lines, levels=None) -> list[Track]:
    """The crane tracks in a set of rails (shapely lines, RD).

    Rails that run parallel side by side are sorted across and paired with
    their neighbours, as many pairs as will go and then the widest: a
    stacking yard is rail, lane, rail, gap, rail, lane, rail, and pairing the
    nearest two first would make a track of every gap.
    """
    import shapely

    rails = []
    for index, line in enumerate(lines):
        c = shapely.get_coordinates(line)
        if len(c) < 2:
            continue
        chord = c[-1] - c[0]
        length = float(np.linalg.norm(chord))
        if length < 5.0:
            continue
        d = chord / length
        if d[0] < 0 or (d[0] == 0 and d[1] < 0):
            d = -d
            c = c[::-1]
        n = np.array([-d[1], d[0]])
        if np.abs((c - c[0]) @ n).max() > STRAIGHT_M:
            continue
        rails.append((c[0], d, length, int(levels[index]) if levels is not None else 0))
    count = len(rails)
    parent = list(range(count))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    limit = np.sin(np.radians(PARALLEL_DEG))

    def beside(i, j):
        """Rail j in rail i's frame: (across, overlap start, overlap end)."""
        p, d, length, _ = rails[i]
        q, e, other, _ = rails[j]
        if abs(d[0] * e[1] - d[1] * e[0]) > limit:
            return None
        n = np.array([-d[1], d[0]])
        s0, s1 = sorted(((q - p) @ d, (q + e * other - p) @ d))
        lo, hi = max(s0, 0.0), min(s1, length)
        if hi - lo < MIN_OVERLAP * min(length, other):
            return None
        return float((q + e * other / 2.0 - p) @ n), lo, hi

    for i in range(count):
        for j in range(i + 1, count):
            side = beside(i, j)
            if side is not None and abs(side[0]) <= MAX_GAUGE_M:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)

    tracks = []
    for members in groups.values():
        if len(members) < 2:
            continue
        ref = members[0]
        n = np.array([-rails[ref][1][1], rails[ref][1][0]])
        members.sort(key=lambda i: float((rails[i][0] + rails[i][1] * rails[i][2] / 2.0) @ n))
        # best[k]: the most pairs among the first k rails, then the most gauge.
        best = [(0, 0.0, [])] * (len(members) + 1)
        for k in range(1, len(members) + 1):
            best[k] = best[k - 1]
            if k >= 2:
                side = beside(members[k - 2], members[k - 1])
                if side is not None and MIN_GAUGE_M <= abs(side[0]) <= MAX_GAUGE_M:
                    pairs, total, chosen = best[k - 2]
                    option = (pairs + 1, total + abs(side[0]), chosen + [(members[k - 2], members[k - 1], side)])
                    if option[:2] > best[k][:2]:
                        best[k] = option
        for i, j, (offset, lo, hi) in best[-1][2]:
            p, d, _, level = rails[i]
            n = np.array([-d[1], d[0]])
            tracks.append(Track(
                start=p + d * lo + n * offset / 2.0, along=d, length=float(hi - lo),
                gauge=float(abs(offset)), level=max(level, rails[j][3]),
            ))
    return tracks


def _runs(s: np.ndarray, gap: float) -> list[tuple[float, float]]:
    """The stretches of sorted positions `s` with no hole wider than `gap`."""
    if not len(s):
        return []
    s = np.sort(s)
    cut = np.flatnonzero(np.diff(s) > gap)
    starts = np.r_[0, cut + 1]
    stops = np.r_[cut, len(s) - 1]
    return [(float(s[a]), float(s[b])) for a, b in zip(starts, stops)]


def _reach(t: np.ndarray, start: float, gap: float = 6.0) -> float:
    """How far out from `start` the values `t` go on without a hole of `gap`."""
    t = np.sort(t[t > start - gap])
    edge = start
    for value in t:
        if value - edge > gap:
            break
        edge = max(edge, value)
    return float(edge)


def find_cranes(tracks: list[Track], field: HeightField, *, wet=None, blocked=None) -> list[Crane]:
    """The cranes the survey saw on `tracks`.

    `wet` is the water (shapely), for which side of a quay crane the ship is;
    `blocked` a mask of cells that are something else, buildings.
    """
    import shapely

    if not tracks:
        return []
    zones = []
    for track in tracks:
        reach = track.gauge / 2.0 + (
            QUAY_CRANE_REACH_M if track.gauge >= QUAY_CRANE_GAUGE_M else REACH_M
        )
        zones.append(shapely.LineString([track.at(0.0), track.at(track.length)]).buffer(
            reach, cap_style="flat"
        ))
    tall = np.isfinite(field.height) & (field.height >= CRANE_LOW_M) & field.mask(zones)
    if blocked is not None:
        tall &= ~blocked
    rows, cols = np.nonzero(tall)
    if not len(rows):
        return []
    xy = field.xy(rows, cols)
    h = field.height[rows, cols]

    # Whose rails each cell is between: a neighbour's crane is not this one's
    # jib. And how high a crane starts on each track: over the stacks where
    # there are stacks between the rails.
    owner = np.full(len(xy), -1)
    frames = []
    floors = []
    for index, track in enumerate(tracks):
        s, t = track.frame(xy)
        frames.append((s, t))
        lane = field.mask([zones[index].intersection(
            shapely.LineString([track.at(0.0), track.at(track.length)]).buffer(
                track.gauge / 2.0, cap_style="flat"
            )
        )])
        stacked = lane & field.flat & (field.height >= 2.2) & (field.height < CRANE_MIN_M)
        floor = CRANE_MIN_M if stacked.sum() >= STACKED_SHARE * max(lane.sum(), 1) else CRANE_LOW_M
        floors.append(floor)
        inside = (
            (s >= -1.0) & (s <= track.length + 1.0)
            & (np.abs(t) <= track.gauge / 2.0 + CORRIDOR_M) & (h >= floor)
        )
        owner[inside & (owner < 0)] = index

    cranes = []
    for index, track in enumerate(tracks):
        s, t = frames[index]
        mine = owner == index
        if mine.sum() < CRANE_MIN_CELLS:
            continue
        wide = track.gauge >= QUAY_CRANE_GAUGE_M
        free = ((owner < 0) | mine) & (h >= floors[index])
        for lo, hi in _runs(s[mine], CRANE_GAP_M):
            run = mine & (s >= lo) & (s <= hi)
            if run.sum() < CRANE_MIN_CELLS:
                continue
            top = float(np.percentile(h[run], 99.5))
            quay = wide and top >= QUAY_CRANE_MIN_M
            if not quay and hi - lo > CRANE_MAX_RUN_M:
                continue
            # Cranes parked buffer to buffer read as one long run.
            parts = max(int(round((hi - lo) / QUAY_CRANE_WIDTH_M)), 1) if quay else 1
            edges = np.linspace(lo, hi, parts + 1)
            for a, b in zip(edges[:-1], edges[1:]):
                part = run & (s >= a) & (s <= b)
                if part.sum() < CRANE_MIN_CELLS:
                    continue
                window = free & (s >= a - 1.0) & (s <= b + 1.0)
                crane = (
                    _quay_crane(track, s, t, h, part, window, wet)
                    if quay
                    else _yard_crane(track, s, t, h, part, window, xy)
                )
                if crane is not None:
                    cranes.append(crane)
    return cranes


def _yard_crane(track, s, t, h, part, window, xy) -> Crane | None:
    """A portal crane if it keeps to its rails, a slewing crane if it does not."""
    half = track.gauge / 2.0
    out = max(_reach(t[window], half), _reach(-t[window], half))
    whole = window & (np.abs(t) <= out + 0.5)
    top = float(np.percentile(h[whole], 99.5))
    lo, hi = np.percentile(s[part], [2, 98])
    span = np.percentile(t[part], 98) - np.percentile(t[part], 2)
    # Keeping to its rails; or low, from rail to rail, with an arm out over
    # the water: a loading bridge.
    kept = out - half <= 0.35 * track.gauge + 3.0 and top <= GANTRY_MAX_RATIO * track.gauge
    bridge = span >= 0.7 * track.gauge and top <= 1.2 * track.gauge and out - half <= track.gauge
    if hi - lo <= max(12.0, track.gauge) and (kept or bridge):
        centre = track.at(float(np.clip((lo + hi) / 2.0, 0.0, track.length)))
        return Crane("gantry", track, centre, top, overhang_m=float(max(out - half - 1.0, 0.0)))
    # The foot is on the rails where the most of it is, the tower; the jib's
    # head is the high part furthest from there.
    counts, edges = np.histogram(s[part], bins=np.arange(lo - 2.0, hi + 6.0, 4.0))
    foot = track.at(float(np.clip(edges[int(np.argmax(counts))] + 2.0, 0.0, track.length)))
    high = whole & (h >= 0.5 * top)
    offset = xy[high] - foot
    away = np.linalg.norm(offset, axis=1)
    reach = float(np.percentile(away, 97))
    if reach < 6.0:
        # The jib stands nearly upright over the foot.
        return Crane("jib", track, foot, top, tip=foot + track.across * 0.3 * top, tip_m=top)
    if reach > 1.5 * top:
        return None  # further out than it is high: a roof beside the track
    way = offset[int(np.argmax(away))] / away.max()
    far = (away >= reach - 2.0) & (offset @ way > 0)
    tip = xy[high][far].mean(axis=0)
    return Crane("jib", track, foot, top, tip=tip, tip_m=float(np.percentile(h[high][far], 75)))


def _quay_crane(track, s, t, h, part, window, wet) -> Crane:
    """A ship-to-shore crane: which side the water is, and where its boom is."""
    import shapely

    half = track.gauge / 2.0
    lo, hi = np.percentile(s[part], [2, 98])
    middle = float(np.clip((lo + hi) / 2.0, 0.0, track.length))
    centre = track.at(middle)
    sides = []
    for sign in (1.0, -1.0):
        probes = np.array([track.at(middle, sign * (half + d)) for d in (10.0, 20.0, 35.0)])
        afloat = int(shapely.contains_xy(wet, probes[:, 0], probes[:, 1]).sum()) if wet is not None else 0
        sides.append((afloat, _reach(sign * t[window], half), sign))
    sign = max(sides)[2]
    tw = sign * t
    top = float(np.percentile(h[part], 99.5))

    # The girder is where most of what is between the rails is: under it
    # legs and portal beams, over it the machinery house and the A-frame.
    counts, edges = np.histogram(h[part], bins=np.arange(24.0, max(top, 30.0) + 2.0, 2.0))
    girder = float(edges[int(np.argmax(counts))] + 1.0)

    # Down, a boom is a level line out over the ship: in a strip along the
    # crane's own axis, the highest return every two metres from the rail
    # out. The first few metres are the A-frame over the waterside legs.
    strip = window & (np.abs(s - middle) <= 5.0) & (tw > half + 2.0)
    bins = np.floor((tw[strip] - half - 2.0) / 2.0).astype(int)
    peak = np.full(40, np.nan)
    for b in np.unique(bins[bins < 40]):
        peak[b] = h[strip][bins == b].max()
    span = peak[4:16]
    boom_up = True
    boom = girder
    outreach = 0.0
    if np.isfinite(span).sum() >= 9:
        level = float(np.nanmedian(span))
        level_bins = np.abs(peak - level) <= 8.0
        # ... at the height of the girder: lower, it is the ship's own deck.
        if level_bins[4:16].sum() >= 9 and abs(level - girder) <= 10.0:
            boom_up = False
            boom = level
            last = 4
            for b in range(4, 40):
                if level_bins[b]:
                    last = b
                elif b - last > 3:
                    break
            outreach = min(2.0 + 2.0 * (last + 1), QUAY_CRANE_REACH_M - 10.0)
    head = peak[:5] if not boom_up else peak[:12]
    if np.isfinite(head).any():
        top = max(top, float(np.nanmax(head)))
    back = float(np.clip(_reach(-tw[window], half) - half, 6.0, 25.0))
    return Crane(
        "quay", track, centre, top, water=sign * track.across, boom_m=boom,
        outreach_m=outreach, backreach_m=back, boom_up=boom_up,
    )


def _spread(values: np.ndarray) -> np.ndarray:
    """Per cell, the range of its 3 x 3 neighbourhood; NaN where it has none."""
    high = values.copy()
    low = values.copy()
    padded = np.pad(values, 1, constant_values=np.nan)
    rows, cols = values.shape
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            if dr == 1 and dc == 1:
                continue
            shifted = padded[dr:dr + rows, dc:dc + cols]
            np.fmax(high, shifted, out=high)
            np.fmin(low, shifted, out=low)
    return high - low


def find_containers(field: HeightField, *, yard, blocked=None, tracks=()) -> list[Stack]:
    """The container stacks the survey saw on `yard` (a mask of cells).

    A flat top at one, two ... six container heights is a stack that many
    high. Each flat top is filled with containers side by side, 2.44 m wide,
    forty-footers where two twenty-foot lengths lie in a row: along the crane
    track it lies in, otherwise along its own long side.
    """
    import shapely
    from rasterio.features import shapes
    from shapely.geometry import shape

    h = field.height
    with np.errstate(invalid="ignore"):
        tier = np.rint(h / TIER_M)
        fits = (
            np.isfinite(h) & (tier >= 1) & (tier <= MAX_TIERS)
            & (h >= tier * 2.5 - 0.3) & (h <= tier * 2.9 + 0.25)
            & field.flat
        )
    fits &= yard
    if blocked is not None:
        fits &= ~blocked
    if not fits.any():
        return []
    tiers = np.where(fits, tier, 0).astype(np.int16)
    lanes = [
        (shapely.LineString([t.at(0.0), t.at(t.length)]).buffer(t.gauge / 2.0, cap_style="flat"), t)
        for t in tracks
    ]
    lane_tree = shapely.STRtree([g for g, _ in lanes]) if lanes else None

    stacks: list[Stack] = []
    grow = field.cell_m
    for geometry, value in shapes(tiers, mask=fits, connectivity=4, transform=field.transform):
        top = shape(geometry)
        if top.area < MIN_STACK_M2:
            continue
        # The neighbourhood test took one cell off every side.
        top = top.buffer(grow, join_style="mitre")
        directions = []
        if lane_tree is not None:
            for index in lane_tree.query(top.centroid, predicate="within"):
                directions.append(lanes[int(index)][1].along)
        if not directions:
            box = shapely.get_coordinates(shapely.oriented_envelope(top))
            a, b = box[1] - box[0], box[2] - box[1]
            long, short = (a, b) if np.linalg.norm(a) >= np.linalg.norm(b) else (b, a)
            directions = [long / max(np.linalg.norm(long), 1e-9), short / max(np.linalg.norm(short), 1e-9)]
        fit = _fill(top, directions)
        if fit is None:
            continue
        for centre, along, length in fit:
            stacks.append(Stack(centre, along, length, int(value)))
    count = sum(s.tiers for s in stacks)
    if count < MIN_CONTAINERS:
        return []
    if count > MAX_CONTAINERS:
        LOG.warning("%d containers seen, more than %d: none built", count, MAX_CONTAINERS)
        return []
    return stacks


def _fill(top, directions):
    """`top` as container footprints, along the direction that fits it best."""
    import shapely

    outline = shapely.get_coordinates(top.exterior)
    best = None
    for along in directions:
        across = np.array([-along[1], along[0]])
        u, v = outline @ along, outline @ across
        length, width = np.ptp(u), np.ptp(v)
        lanes = max(int(round(width / (CONTAINER_WIDTH_M + 0.1))), 1)
        units = max(int(round(length / (CONTAINER_20_M + 0.05))), 1)
        lane_m, unit_m = width / lanes, length / units
        if not (2.0 <= lane_m <= 3.2 and 5.2 <= unit_m <= 7.0):
            continue
        error = abs(lane_m - 2.5) / 2.5 + abs(unit_m - 6.1) / 6.1
        if best is None or error < best[0]:
            best = (error, along, across, u.min(), v.min(), lanes, units, lane_m, unit_m)
    if best is None:
        return None
    _, along, across, u0, v0, lanes, units, lane_m, unit_m = best
    iu, iv = np.meshgrid(np.arange(units), np.arange(lanes))
    centres = (
        (u0 + (iu.ravel() + 0.5) * unit_m)[:, None] * along
        + (v0 + (iv.ravel() + 0.5) * lane_m)[:, None] * across
    )
    covered = shapely.contains_xy(top, centres[:, 0], centres[:, 1]).reshape(lanes, units)
    centres = centres.reshape(lanes, units, 2)
    out = []
    for lane in range(lanes):
        unit = 0
        while unit < units:
            if not covered[lane, unit]:
                unit += 1
                continue
            if unit + 1 < units and covered[lane, unit + 1]:
                centre = (centres[lane, unit] + centres[lane, unit + 1]) / 2.0
                out.append((centre, along, 2.0 * CONTAINER_20_M + 0.07))
                unit += 2
            else:
                out.append((centres[lane, unit], along, CONTAINER_20_M))
                unit += 1
    return out


__all__ = [
    "CONTAINER_HEIGHT_M", "Crane", "HeightField", "Stack", "Track", "find_containers",
    "find_cranes", "pair_tracks",
]
