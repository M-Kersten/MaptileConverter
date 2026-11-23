"""Tunnels: the mouths are measured, the inside is drawn to fit them.

The BGT draws a tunnel (``tunneldeel``) as the part that is covered. Its open
approaches are not tunnel at all: they are ordinary roads at level 0, sunk in
a cutting, and the AHN sees them all the way down to the mouth. Over the
Maastunnel the terrain model reads the south approach at −3.1 m NAP and the
north one at −5.4 m at the very edge of the tunnel polygon, with the ground on
top of the tunnel at +4 to +5 m. So where a tunnel's road begins is a
measurement, and it is 8 m below where the first version of this put it: level
with the ground, so the measured ramp ran into a wall.

What cannot be measured is everything between the mouths. That is drawn, and
drawn to fit what is known:

* **Each tunnel on its own.** The BGT splits a tunnel into parts, and one
  square kilometre of Rotterdam holds thirteen of them: the car tunnel, the
  cycle tube with its escalator shafts, four short underpasses, two metro
  sections. Touching parts are one tunnel, and every tunnel gets its own
  profile. One axis through all of them put the Maastunnel's north mouth 18 m
  down because an underpass further north was the "end" of the tunnel.
* **Portals are where the ground outside drops away.** Along a tunnel's edge,
  the survey just outside is compared with the ground just inside; where it is
  `PORTAL_DROP_M` lower, that stretch is a mouth, and the road there is at the
  height read just outside it. A tunnel cut by the edge of the area has no
  portal there: the road carries on underground.
* **Below what is above it.** The road stays a clearance and a roof
  (`CLEARANCE_M`, `ROOF_M`) under the ground on the tunnel, or under the bed of
  the water it crosses. No open dataset here measures a river bed; the bed
  deepens away from the bank to at most `structures.tunnel_water_depth_m`.
* **No steeper than a tunnel is built.** 4.5% for a road (`GRADE`), less for a
  railway, and down from each mouth without climbing again before the deepest
  point: the highest profile that meets all of that is the one drawn.

It is still the one piece of this model that is drawn rather than measured,
apart from its mouths, and `metadata.json` says so.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field

import numpy as np

LOG = logging.getLogger(__name__)

# Headroom inside, floor to ceiling. A Dutch road tunnel is built for 4.5 m
# of traffic; a train needs room for its overhead line; a metro or tram
# tunnel less; a cycle and foot tunnel 2.5 m and a little.
CLEARANCE_M = {"road": 4.7, "rail": 6.2, "metro": 4.6, "path": 2.8}
# The steepest a tunnel's floor may run.
GRADE = {"road": 0.045, "rail": 0.025, "metro": 0.04, "path": 0.05}
# Roof slab plus the least ground over it, over the ceiling.
ROOF_M = 1.5
# An edge is a mouth where the ground just outside is this much below the
# ground on the tunnel just inside, over at least `MIN_PORTAL_M` of edge.
PORTAL_DROP_M = 2.7
MIN_PORTAL_M = 3.0
# Where the edge is read: this far out for the cutting floor, this far in for
# the ground on the tunnel. Not right at the edge: the headwall of a mouth
# often stands lower than the ground behind it.
OUTSIDE_AT_M = (1.5, 3.0)
INSIDE_AT_M = (4.0, 8.0)
# A mouth found over part of a tunnel's end is carried on along that end, as
# long as the edge keeps its direction and nothing reads as closed.
EXTEND_TURN_DEG = 40.0
EXTEND_MAX_M = 15.0
CLOSED_DROP_M = 1.0
# Across one mouth the floor read in front of it is level to within this.
FLOOR_STEP_M = 1.5
# The least room a tunnel is given, and the least roof over it, whatever the
# mouths say.
MIN_CLEARANCE_M = 2.2
MIN_ROOF_M = 0.5
# Water deepens from the bank by this much per metre out, from this much at
# the bank, until `tunnel_water_depth_m`.
BANK_SLOPE = 0.25
BANK_DEPTH_M = 1.0
WATER_DEPTH_M = 12.0
# How far the drawn profile is smoothed, so the floor eases into a descent
# rather than kinking at it.
SMOOTH_M = 8.0
# Parts closer than this are one tunnel.
JOIN_M = 0.5
# How far into a tunnel from a mouth the floor is held at what the survey
# read in front of it.
MOUTH_BAND_M = max(OUTSIDE_AT_M)


@dataclass
class Portal:
    """One mouth: a stretch of a tunnel's edge open to a cutting."""

    points: np.ndarray  # (n, 2) RD, along the tunnel's edge
    floor_nap: float  # the cutting floor just outside, from the AHN
    top_nap: float  # the ground on the tunnel just inside
    outward: np.ndarray  # (2,) unit vector away from the tunnel
    # The cutting floor further out, and how far out: the apron in front of the
    # mouth follows the ramp rather than lying flat.
    ramp_nap: float = float("nan")
    ramp_at_m: float = 0.0

    @property
    def length(self) -> float:
        steps = np.diff(self.points, axis=0)
        return float(np.hypot(steps[:, 0], steps[:, 1]).sum())


@dataclass
class Tunnel:
    """One tunnel: its outline, what runs through it, and its floor."""

    polygon: object  # shapely Polygon, RD, as the BGT has it (not clipped)
    kinds: set
    portals: list[Portal] = field(default_factory=list)
    clearance_m: float = CLEARANCE_M["road"]
    grade: float = GRADE["road"]
    cover_m: float = CLEARANCE_M["road"] + ROOF_M
    # The floor on a raster over the tunnel, NaN well outside it.
    x0: float = 0.0
    y0: float = 0.0
    cell: float = 1.0
    floor: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # What the floor came out at, for the metadata.
    stats: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        for kind in ("rail", "metro", "road", "path"):
            if kind in self.kinds:
                return kind
        return "road"

    def floor_at(self, x, y) -> np.ndarray:
        """The floor at RD coordinates, bilinear; NaN off the raster."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        out = np.full(x.shape, np.nan)
        if not self.floor.size:
            return out
        ny, nx = self.floor.shape
        fx = (x - self.x0) / self.cell - 0.5
        fy = (y - self.y0) / self.cell - 0.5
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        inside = (ix >= 0) & (iy >= 0) & (ix < nx - 1) & (iy < ny - 1)
        if not inside.any():
            return out
        ix, iy, tx, ty = ix[inside], iy[inside], (fx - np.floor(fx))[inside], (fy - np.floor(fy))[inside]
        corners = np.stack([
            self.floor[iy, ix], self.floor[iy, ix + 1],
            self.floor[iy + 1, ix], self.floor[iy + 1, ix + 1],
        ], axis=1)
        weights = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)
        valid = np.isfinite(corners)
        total = (weights * valid).sum(axis=1)
        value = (np.where(valid, corners, 0.0) * weights).sum(axis=1)
        out[inside] = np.where(total > 1e-9, value / np.maximum(total, 1e-12), np.nan)
        return out


@dataclass
class TunnelPlan:
    """Every tunnel touching the area."""

    tunnels: list[Tunnel] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tunnels)

    def floor_at(self, x, y) -> np.ndarray:
        """The floor of whichever tunnel is at each point; NaN outside them."""
        import shapely

        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        out = np.full(x.shape, np.nan)
        points = shapely.points(x, y)
        for tunnel in self.tunnels:
            near = shapely.dwithin(tunnel.polygon, points, 0.5)
            if near.any():
                out[near] = tunnel.floor_at(x[near], y[near])
        return out

    def stats(self) -> list[dict]:
        return [t.stats for t in self.tunnels]


# ---------------------------------------------------------------------------
# What a tunnel is
# ---------------------------------------------------------------------------


def group_parts(polygons) -> list:
    """Touching tunnel parts as one polygon per tunnel."""
    import shapely

    polygons = [p for p in polygons if p is not None and not p.is_empty]
    if not polygons:
        return []
    merged = shapely.union_all(
        shapely.buffer(np.array(polygons, dtype=object), JOIN_M, join_style="mitre")
    )
    out = []
    for blob in shapely.get_parts(merged):
        members = [p for p in polygons if p.intersects(blob)]
        tunnel = shapely.make_valid(
            shapely.union_all(np.array(members, dtype=object))
            .buffer(JOIN_M, join_style="mitre").buffer(-JOIN_M, join_style="mitre")
        )
        parts = [g for g in shapely.get_parts(tunnel) if g.geom_type == "Polygon" and g.area > 1.0]
        if parts:
            out.append(max(parts, key=lambda g: g.area) if len(parts) == 1 else shapely.MultiPolygon(parts))
    return out


def _kinds_inside(polygon, roads, tracks) -> set:
    """What runs through a tunnel, from the BGT parts below ground in it."""
    kinds = set()
    track = False
    for geometry, function in roads:
        if not polygon.intersects(geometry) or polygon.intersection(geometry).area < 2.0:
            continue
        function = str(function or "").lower()
        if "spoor" in function:
            track = True
        elif "fiets" in function or "voet" in function or "ruiter" in function:
            kinds.add("path")
        elif "berm" not in function:
            kinds.add("road")
    # The track lines say which railway: a train needs its overhead line.
    lines = {
        "rail" if str(function or "").lower() == "trein" else "metro"
        for geometry, function in tracks if polygon.intersects(geometry)
    }
    kinds |= lines
    if track and not lines:
        kinds.add("rail")
    return kinds or {"road"}


# ---------------------------------------------------------------------------
# Portals
# ---------------------------------------------------------------------------


def _edge_samples(ring, spacing_m: float = 1.0):
    """Points every `spacing_m` along a ring, with the normal pointing away
    from the polygon on its left."""
    import shapely

    line = shapely.LineString(ring)
    length = float(line.length)
    if length < 3 * spacing_m:
        return np.zeros((0, 2)), np.zeros((0, 2))
    at = np.arange(0.0, length, spacing_m)
    points = shapely.get_coordinates(shapely.line_interpolate_point(line, at))
    ahead = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(at + 0.5, length)))
    behind = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(at - 0.5, 0.0)))
    tangent = ahead - behind
    tangent /= np.maximum(np.hypot(tangent[:, 0], tangent[:, 1]), 1e-9)[:, None]
    # The polygon is on the left of an anticlockwise ring: outward is right.
    outward = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    return points, outward


def _runs(flags: np.ndarray, gap: int = 2) -> list[np.ndarray]:
    """Index runs where `flags` holds round a closed ring, short gaps bridged."""
    n = len(flags)
    if not flags.any():
        return []
    if flags.all():
        return [np.arange(n)]
    filled = flags.copy()
    # Bridge gaps of up to `gap` samples between two flagged stretches.
    off = np.flatnonzero(~flags)
    start = int(off[0])
    order = np.roll(np.arange(n), -start)  # starts on an unflagged sample
    seq = flags[order]
    i = 0
    while i < n:
        if not seq[i]:
            j = i
            while j < n and not seq[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= gap:
                filled[order[i:j]] = True
            i = j
        else:
            i += 1
    seq = filled[order]
    runs, i = [], 0
    while i < n:
        if seq[i]:
            j = i
            while j < n and seq[j]:
                j += 1
            runs.append(order[i:j])
            i = j
        else:
            i += 1
    return runs


def _extend(run: np.ndarray, tangent: np.ndarray, closed: np.ndarray, steps: int) -> np.ndarray:
    """Carry a run on round the ring at both ends while the edge keeps the
    run's direction and nothing there reads as closed."""
    n = len(tangent)
    mean = tangent[run].mean(axis=0)
    mean /= max(float(np.hypot(*mean)), 1e-9)
    limit = np.cos(np.radians(EXTEND_TURN_DEG))
    taken = set(int(i) for i in run)
    head, tail = list(run[:1]), list(run[-1:])
    for _ in range(steps):
        i = (int(tail[-1]) + 1) % n
        if i in taken or closed[i] or float(np.dot(tangent[i], mean)) < limit:
            break
        tail.append(i)
        taken.add(i)
    for _ in range(steps):
        i = (int(head[-1]) - 1) % n
        if i in taken or closed[i] or float(np.dot(tangent[i], mean)) < limit:
            break
        head.append(i)
        taken.add(i)
    return np.concatenate([np.asarray(head[:0:-1], dtype=np.int64), run, np.asarray(tail[1:], dtype=np.int64)])


def _to_corners(line: np.ndarray, corners: np.ndarray, reach: float) -> np.ndarray:
    """A mouth carried on to the tunnel's corner at either end, if one is
    within `reach` beyond it. The edge turns at a corner, so the samples
    there read as no mouth, and a 10 m mouth came out 8 m wide."""
    if len(line) < 2 or not len(corners):
        return line
    out = line
    for at_end in (False, True):
        end = out[-1] if at_end else out[0]
        heading = (out[-1] - out[-2]) if at_end else (out[0] - out[1])
        distance = np.hypot(*(corners - end).T)
        k = int(np.argmin(distance))
        if 1e-6 < distance[k] <= reach and float(np.dot(corners[k] - end, heading)) > 0:
            out = np.vstack([out, corners[k]]) if at_end else np.vstack([corners[k], out])
    return out


def _split_at_steps(run: np.ndarray, floor: np.ndarray, step_m: float = FLOOR_STEP_M):
    """A run cut wherever the floor read in front of it jumps: a mouth opens
    onto one cutting floor, level across its width. Joined round a corner,
    the Maastunnel's south mouth took in 40 m of the street beside the
    tunnel, seven metres higher, and its roof was pressed down to the
    cutting floor along all of it. Cut there, the street is a portal of its
    own, and `consistent` throws it out."""
    pieces, current, last = [], [int(run[0])], floor[run[0]]
    for i in run[1:]:
        i = int(i)
        value = floor[i]
        if np.isfinite(value) and np.isfinite(last) and abs(value - last) > step_m:
            pieces.append(np.asarray(current, dtype=np.int64))
            current = [i]
        else:
            current.append(i)
        if np.isfinite(value):
            last = value
    pieces.append(np.asarray(current, dtype=np.int64))
    return pieces


def _merge_runs(runs: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Runs round a ring of `n` samples that overlap or touch, as one each,
    every run in ring order."""
    sets: list[set] = []
    for run in runs:
        members = set(int(i) for i in run)
        # Touching counts: the sample either side of each end.
        reach = members | {(int(run[0]) - 1) % n, (int(run[-1]) + 1) % n}
        joined = [k for k, other in enumerate(sets) if other & reach]
        for k in sorted(joined, reverse=True):
            members |= sets.pop(k)
        sets.append(members)
    out = []
    for members in sets:
        if len(members) >= n:
            out.append(np.arange(n))
            continue
        start = next(i for i in members if (i - 1) % n not in members)
        order = [start]
        while (order[-1] + 1) % n in members:
            order.append((order[-1] + 1) % n)
        out.append(np.asarray(order, dtype=np.int64))
    return out


def consistent(portals: list[Portal], grade: float, slack: float = 2.5) -> list[Portal]:
    """The portals one tunnel floor can join, lowest first.

    A street beside a tunnel can read like a mouth: the ground on the tunnel
    stands higher than the street. But it does not fit the real mouth next to
    it -- seven metres higher, three metres along, beside the Maastunnel's
    south mouth -- and no tunnel climbs that. Cuttings are low, so of two
    portals that cannot both be right the higher one goes.
    """
    import shapely

    kept: list[Portal] = []
    for portal in sorted(portals, key=lambda p: p.floor_nap):
        line = shapely.LineString(portal.points) if len(portal.points) > 1 else shapely.Point(portal.points[0])
        fits = True
        for other in kept:
            other_line = (
                shapely.LineString(other.points) if len(other.points) > 1
                else shapely.Point(other.points[0])
            )
            distance = float(line.distance(other_line))
            if abs(portal.floor_nap - other.floor_nap) > max(1.0, slack * grade * distance):
                fits = False
                break
        if fits:
            kept.append(portal)
    return kept


def find_portals(polygon, *, ground, survey=None, inside_area=None, blocked=None,
                 opening=None, spacing_m: float = 1.0) -> list[Portal]:
    """The mouths of one tunnel: stretches of its edge open to a cutting.

    `ground(x, y)` is what stands on the tunnel (`top_reader`); `survey(x, y)` the raw terrain
    model, NaN where it saw nothing (better at the cutting floor, which the
    grid smears into the roof beside it). `inside_area` is where the survey
    can be read at all -- the area being built -- and `blocked(points)` is
    True where outside is not open ground: water, or another tunnel.
    `opening(points)`, where the BGT is known, is True on a surface at ground
    level something travels on -- a road, a path, a track bed -- which is
    what lies in front of a mouth and not beside a tunnel.
    """
    import shapely

    portals: list[Portal] = []
    for part in shapely.get_parts(polygon):
        ring = np.asarray(shapely.get_coordinates(shapely.orient_polygons(part).exterior))
        points, outward = _edge_samples(ring, spacing_m)
        if not len(points):
            continue
        read = survey or ground

        def outside_at(distance):
            probe = points + distance * outward
            return probe, np.asarray(read(probe[:, 0], probe[:, 1]), dtype=np.float64)

        probes, readings = zip(*(outside_at(d) for d in OUTSIDE_AT_M))
        readings = np.column_stack(readings)
        seen = np.isfinite(readings).any(axis=1)
        outside = np.full(len(points), np.nan)
        outside[seen] = np.nanmedian(readings[seen], axis=1)
        inside = np.column_stack([
            np.asarray(ground(*(points - d * outward).T), dtype=np.float64) for d in INSIDE_AT_M
        ])
        inside = np.nanmax(inside, axis=1)

        usable = np.isfinite(outside) & np.isfinite(inside)
        if inside_area is not None:
            far_out = points + max(OUTSIDE_AT_M) * outward
            far_in = points - max(INSIDE_AT_M) * outward
            usable &= shapely.contains_xy(inside_area, far_out[:, 0], far_out[:, 1])
            usable &= shapely.contains_xy(inside_area, far_in[:, 0], far_in[:, 1])
        # Outside has to be outside: a concave tunnel can face itself.
        for probe in probes:
            usable &= ~shapely.contains_xy(polygon, probe[:, 0], probe[:, 1])
            if blocked is not None:
                usable &= ~np.asarray(blocked(probe), dtype=bool)
        drop = inside - outside
        flags = usable & (drop >= PORTAL_DROP_M)
        if opening is not None:
            flags &= np.asarray(opening(probes[0]), dtype=bool)
        closed = usable & (drop < CLOSED_DROP_M)
        tangent = np.column_stack([-outward[:, 1], outward[:, 0]])
        steps = int(round(EXTEND_MAX_M / spacing_m))

        runs = [
            run for run in _runs(flags, gap=int(round(6.0 / spacing_m)))
            if len(run) * spacing_m >= MIN_PORTAL_M
        ]
        # A mouth is the whole end of a tunnel, though the survey may read
        # only part of it: the rest is in a building's shadow, or the probes
        # there fell on another tunnel. Carried on, two parts of one mouth
        # meet, and are one.
        runs = _merge_runs([_extend(run, tangent, closed, steps) for run in runs], len(points))
        runs = [piece for run in runs for piece in _split_at_steps(run, outside)]
        for run in runs:
            # `run` is in edge order, so it is a line along the mouth; only
            # the samples that read as open count.
            read_run = run[flags[run]]
            if len(read_run) * spacing_m < MIN_PORTAL_M:
                continue
            floors = outside[read_run]
            # The cutting floor is the lowest of what is out there: the
            # footways and the foot of the retaining walls stand higher.
            low = floors[floors <= np.nanpercentile(floors, 50)]
            floor = float(np.nanmedian(low))
            top = float(np.nanmedian(inside[read_run]))
            direction = outward[run].mean(axis=0)
            direction /= max(float(np.hypot(*direction)), 1e-9)
            # Further out, for the slope of the ramp in front of the mouth.
            ramp_at = 8.0
            far = points[run] + ramp_at * direction
            ramp = np.asarray(read(far[:, 0], far[:, 1]), dtype=np.float64)
            ramp_nap = float(np.nanpercentile(ramp, 30)) if np.isfinite(ramp).any() else float("nan")
            if not np.isfinite(ramp_nap) or abs(ramp_nap - floor) > 0.1 * ramp_at:
                ramp_nap = floor
            portals.append(Portal(
                points=_to_corners(points[run], ring[:-1], 1.5 * spacing_m),
                floor_nap=floor, top_nap=top, outward=direction,
                ramp_nap=ramp_nap, ramp_at_m=ramp_at,
            ))
    return portals


# ---------------------------------------------------------------------------
# The floor between the portals
# ---------------------------------------------------------------------------

_STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbours(index: int, nx: int, ny: int):
    iy, ix = divmod(index, nx)
    for dy, dx in _STEPS:
        jy, jx = iy + dy, ix + dx
        if 0 <= jy < ny and 0 <= jx < nx:
            yield jy * nx + jx, (1.4142135623730951 if dy and dx else 1.0)


def grade_envelope(ceiling: np.ndarray, mask: np.ndarray, rise_per_cell: float) -> np.ndarray:
    """The highest surface under `ceiling` that climbs no faster than
    `rise_per_cell` from one cell to the next, moving only through `mask`.

    `min over q of (ceiling(q) + rise * distance(p, q))`, by Dijkstra from the
    lowest cells up.
    """
    ny, nx = ceiling.shape
    flat = np.where(mask, ceiling, np.inf).ravel().astype(np.float64)
    done = np.zeros(flat.size, dtype=bool)
    heap = [(float(flat[i]), int(i)) for i in np.flatnonzero(mask.ravel())]
    heapq.heapify(heap)
    valid = mask.ravel()
    while heap:
        value, index = heapq.heappop(heap)
        if done[index] or value > flat[index]:
            continue
        done[index] = True
        for other, step in _neighbours(index, nx, ny):
            if not valid[other] or done[other]:
                continue
            candidate = value + rise_per_cell * step
            if candidate < flat[other]:
                flat[other] = candidate
                heapq.heappush(heap, (candidate, other))
    return np.where(mask, flat.reshape(ny, nx), np.nan)


def descend_from(seeds: np.ndarray, surface: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """From the seed cells inward, the highest level each cell can be reached
    at without rising again: the most, over every route from a seed, of the
    lowest `surface` along it. -inf where no seed reaches.

    Along a line this is the running minimum from the mouth. Taking the higher
    of the values from two mouths gives the highest floor that only descends
    from each mouth to the deepest point, as a tunnel's does.
    """
    ny, nx = surface.shape
    flat = surface.ravel()
    best = np.full(flat.size, -np.inf)
    valid = mask.ravel()
    heap = []
    for index in np.flatnonzero(seeds.ravel() & valid):
        best[index] = flat[index]
        heap.append((-float(flat[index]), int(index)))
    heapq.heapify(heap)
    while heap:
        negative, index = heapq.heappop(heap)
        value = -negative
        if value < best[index]:
            continue
        for other, _ in _neighbours(index, nx, ny):
            if not valid[other]:
                continue
            candidate = min(value, float(flat[other]))
            if candidate > best[other]:
                best[other] = candidate
                heapq.heappush(heap, (-candidate, other))
    return best.reshape(ny, nx)


def _box_mean(values: np.ndarray, weight: np.ndarray, radius: int) -> np.ndarray:
    """Weighted mean over a square window, by summed-area tables."""
    def window_sum(a):
        padded = np.pad(a, radius + 1, mode="constant")
        table = padded.cumsum(axis=0).cumsum(axis=1)
        size = 2 * radius + 1
        ny, nx = a.shape
        return (
            table[size:size + ny, size:size + nx] - table[0:ny, size:size + nx]
            - table[size:size + ny, 0:nx] + table[0:ny, 0:nx]
        )

    total = window_sum(weight)
    return window_sum(values * weight) / np.maximum(total, 1e-12)


def _fill_outward(values: np.ndarray, rings: int) -> np.ndarray:
    """Spread the finite cells outward by `rings` cells, each new cell the
    mean of its finite neighbours, so a point on the outline reads the floor."""
    out = values.copy()
    for _ in range(rings):
        known = np.isfinite(out)
        if known.all():
            break
        padded = np.pad(np.where(known, out, 0.0), 1)
        count = np.pad(known.astype(np.float64), 1)
        total = np.zeros_like(out)
        weight = np.zeros_like(out)
        ny, nx = out.shape
        for dy, dx in _STEPS:
            total += padded[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
            weight += count[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
        grow = ~known & (weight > 0)
        out[grow] = total[grow] / weight[grow]
    return out


def water_top(points: np.ndarray, water, frame=None, max_depth_m: float = WATER_DEPTH_M):
    """The bed under each point that is in water, NaN where it is not.

    Nothing here measures a river bed, so it is drawn: `BANK_DEPTH_M` at the
    bank, deepening `BANK_SLOPE` per metre out, to `max_depth_m`. The bank is
    the edge of all the water together: the BGT cuts the Maas into several
    polygons, and the seam between two of them is not a bank. `frame` is the
    area's outline, which is where the water was cut rather than where it has
    a bank.
    """
    import shapely

    out = np.full(len(points), np.nan)
    if not water or not len(points):
        return out
    polygons, levels = [], []
    for body in water:
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in body.rings]
        rings = [r for r in rings if len(r) >= 4]
        if rings:
            polygons.append(shapely.make_valid(shapely.Polygon(rings[0], rings[1:])))
            levels.append(float(body.level_nap))
    if not polygons:
        return out
    geometry = shapely.points(points[:, 0], points[:, 1])
    level = np.full(len(points), np.inf)
    for polygon, value in zip(polygons, levels):
        inside = shapely.contains(polygon, geometry) | shapely.touches(polygon, geometry)
        level[inside] = np.minimum(level[inside], value)
    wet = np.isfinite(level)
    if not wet.any():
        return out
    bank = shapely.union_all(np.array(polygons, dtype=object)).boundary
    if frame is not None:
        bank = bank.difference(frame.exterior.buffer(0.5))
    distance = (
        np.full(int(wet.sum()), 1e6) if bank.is_empty
        else shapely.distance(bank, geometry[wet])
    )
    out[wet] = level[wet] - np.minimum(max_depth_m, BANK_DEPTH_M + BANK_SLOPE * distance)
    return out


def plan_floor(tunnel: Tunnel, *, ground, area, water=(), frame=None,
               water_depth_m: float = WATER_DEPTH_M, max_cells: int = 150_000) -> None:
    """Give a tunnel its floor (`tunnel.floor`), within `area`."""
    import shapely

    region = shapely.intersection(tunnel.polygon, area) if area is not None else tunnel.polygon
    if region.is_empty:
        return
    cell = max(1.0, float(np.sqrt(region.area / max_cells)))
    x0, y0, x1, y1 = region.bounds
    x0, y0 = x0 - 3 * cell, y0 - 3 * cell
    nx = int(np.ceil((x1 + 3 * cell - x0) / cell))
    ny = int(np.ceil((y1 + 3 * cell - y0) / cell))
    cx = x0 + (np.arange(nx) + 0.5) * cell
    cy = y0 + (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(cx, cy)
    mask = shapely.contains_xy(region, gx, gy)
    if not mask.any():
        # A sliver narrower than a cell: its nearest cells will do.
        mask = shapely.dwithin(region, shapely.points(gx, gy), cell)
    points = np.column_stack([gx[mask], gy[mask]])

    top = np.asarray(ground(points[:, 0], points[:, 1]), dtype=np.float64)
    bed = water_top(points, water, frame=frame, max_depth_m=water_depth_m)
    top = np.where(np.isfinite(bed), np.fmin(bed, top), top)
    ceiling = np.full(mask.shape, np.nan)
    ceiling[mask] = top - tunnel.cover_m

    seeds = np.zeros(mask.shape, dtype=bool)
    for portal in tunnel.portals:
        line = shapely.LineString(portal.points) if len(portal.points) > 1 else shapely.Point(portal.points[0])
        # The mouth is measured: the floor there is what the survey read. A
        # band of it, not a line: the outline runs a metre or two past the
        # roof, and in that strip the survey sees the cutting floor, which
        # read as "the ground on the tunnel" would sink the whole floor a
        # roof's depth under it.
        near = mask & shapely.dwithin(line, shapely.points(gx, gy), max(1.5 * cell, MOUTH_BAND_M))
        if not near.any():
            continue
        ceiling[near] = portal.floor_nap
        seeds |= near

    envelope = grade_envelope(ceiling, mask, tunnel.grade * cell)
    lowest = np.full(mask.shape, -np.inf)
    if seeds.any():
        floor = descend_from(seeds, envelope, mask)
        floor = np.where(np.isfinite(floor) & mask, floor, envelope)
        # A measured mouth beats a guessed river bed. From each mouth the
        # floor may fall no faster than the grade, so where the two cannot
        # both hold -- a short tunnel under deep water -- the floor keeps to
        # the mouths and the cover under the water gives.
        pins = np.where(seeds, -ceiling, np.inf)
        lowest = -grade_envelope(pins, mask, tunnel.grade * cell)
        lowest = np.where(np.isfinite(lowest), lowest, -np.inf)
        floor = np.maximum(floor, lowest)
    else:
        floor = envelope

    # Eased, and held at the mouths, which are measured.
    radius = max(1, int(round(SMOOTH_M / cell)))
    weight = mask.astype(np.float64)
    smooth = _box_mean(np.where(mask, floor, 0.0), weight, radius)
    smooth = _box_mean(np.where(mask, smooth, 0.0), weight, radius)
    if seeds.any():
        held = np.zeros(mask.shape)
        for portal in tunnel.portals:
            line = shapely.LineString(portal.points) if len(portal.points) > 1 else shapely.Point(portal.points[0])
            distance = shapely.distance(line, shapely.points(gx, gy))
            held = np.maximum(held, np.clip(1.0 - distance / (2.0 * radius * cell), 0.0, 1.0))
        smooth = held * floor + (1.0 - held) * smooth
    # Smoothing lifts the bottom of a descent a little; never above the cover,
    # or what the mouths allow.
    allowed = np.maximum(envelope, lowest) + 0.05
    floor = np.where(mask, np.minimum(smooth, np.where(seeds, np.inf, allowed)), np.nan)

    tunnel.x0, tunnel.y0, tunnel.cell = float(x0), float(y0), float(cell)
    tunnel.floor = _fill_outward(floor, 3)
    values = floor[mask]
    depth = (top - values)
    tunnel.stats = {
        "kind": tunnel.kind,
        "area_m2": round(float(tunnel.polygon.area), 1),
        "portals": len(tunnel.portals),
        "portal_floors_nap": [round(p.floor_nap, 2) for p in tunnel.portals],
        "floor_nap": [round(float(values.min()), 2), round(float(values.max()), 2)],
        "deepest_below_surface_m": round(float(depth.max()), 2),
        "clearance_m": round(float(tunnel.clearance_m), 2),
        "cover_m": round(float(tunnel.cover_m), 2),
        "grade": tunnel.grade,
    }


def top_reader(ground, survey=None, reach_m: float = 20.0):
    """What stands on a tunnel, at RD coordinates.

    The terrain model where it saw the ground. Where it did not, something is
    built there: the roof of an underpass, which the bare-earth model leaves
    out by design, or a building. The gridded terrain fills such a hole from
    whatever is around it, cutting floors included -- four metres too low over
    the underpasses by the Parkhaven, which put their floors four metres under
    their own mouths. The surface model is no better: it reads the
    Maastunnel's ventilation buildings, 19 m up. What is wanted is the ground
    level around the structure, so a hole takes the highest ground the survey
    saw within `reach_m`, and the filled grid where it saw none at all. Far
    enough to reach the street on both sides of an underpass: at 10 m the
    middle of one by the Parkhaven saw only the cutting floor at its mouth.
    """
    if survey is None:
        return ground
    angles = np.radians(np.arange(0.0, 360.0, 30.0))
    rings = np.arange(2.0, reach_m + 1e-9, 2.0)
    offsets = np.array([(r * np.cos(a), r * np.sin(a)) for r in rings for a in angles])

    def top(x, y):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        out = np.asarray(survey(x, y), dtype=np.float64).copy()
        hole = ~np.isfinite(out)
        if hole.any():
            hx, hy = x[hole], y[hole]
            around = np.full((len(hx), len(offsets)), np.nan)
            for k, (dx, dy) in enumerate(offsets):
                around[:, k] = survey(hx + dx, hy + dy)
            seen = np.isfinite(around).any(axis=1)
            filled = np.asarray(ground(hx, hy), dtype=np.float64)
            highest = np.full(len(hx), np.nan)
            highest[seen] = np.nanmax(around[seen], axis=1)
            out[hole] = np.where(seen, np.fmax(highest, filled), filled)
        return out

    return top


def plan_tunnels(
    polygons,
    *,
    ground,
    survey=None,
    area=None,
    water=(),
    roads=(),
    tracks=(),
    surface_roads=(),
    water_depth_m: float = WATER_DEPTH_M,
) -> TunnelPlan:
    """Group the BGT tunnel parts into tunnels, find their mouths, and draw
    each one's floor.

    `ground` is the gridded terrain and `survey` the raw terrain model, NaN
    where it saw no ground, each `f(x, y)` in RD (see `top_reader`).
    `polygons` are the parts as the BGT has them, not clipped to the area: a
    tunnel's far end may be outside it, and a cut is not a mouth. `area` is
    where the survey can be read (the area being built). `roads` and `tracks`
    are `(geometry, functie)` pairs of the BGT parts below ground, which say
    what a tunnel carries and so how high and how steep it is;
    `surface_roads` the geometries of those at ground level, which is what a
    mouth opens onto.
    """
    import shapely

    plan = TunnelPlan()
    groups = group_parts(polygons)
    if not groups:
        return plan
    water_polygons = []
    for body in water or ():
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in body.rings]
        rings = [r for r in rings if len(r) >= 4]
        if rings:
            water_polygons.append(shapely.make_valid(shapely.Polygon(rings[0], rings[1:])))
    wet = shapely.union_all(np.array(water_polygons, dtype=object)) if water_polygons else None
    others = shapely.union_all(np.array(groups, dtype=object))

    def blocked(probe):
        hit = shapely.contains_xy(others, probe[:, 0], probe[:, 1])
        if wet is not None:
            hit |= shapely.contains_xy(wet, probe[:, 0], probe[:, 1])
        return hit

    opening = None
    if surface_roads:
        travelled = shapely.union_all(np.array(
            [shapely.make_valid(g) for g in surface_roads], dtype=object
        ))
        shapely.prepare(travelled)

        def on_a_road(probe):
            return shapely.intersects_xy(travelled, probe[:, 0], probe[:, 1])

        opening = on_a_road

    frame = area if area is not None else None
    top = top_reader(ground, survey)
    for polygon in groups:
        kinds = _kinds_inside(polygon, roads, tracks)
        tunnel = Tunnel(polygon=polygon, kinds=kinds)
        tunnel.clearance_m = max(CLEARANCE_M[k] for k in kinds)
        tunnel.grade = min(GRADE[k] for k in kinds)
        tunnel.portals = consistent(find_portals(
            polygon, ground=top, survey=survey, inside_area=area, blocked=blocked,
            opening=opening,
        ), tunnel.grade)
        # No more room than the mouths show there is: an underpass under a
        # dike road is lower than any rule of thumb, and a measured mouth
        # wins over a guessed one.
        drops = [p.top_nap - p.floor_nap for p in tunnel.portals]
        if drops:
            tunnel.clearance_m = max(min(tunnel.clearance_m, min(drops) - MIN_ROOF_M), MIN_CLEARANCE_M)
        cover = tunnel.clearance_m + ROOF_M
        if drops:
            cover = max(min(cover, min(drops)), tunnel.clearance_m + MIN_ROOF_M)
        tunnel.cover_m = cover
        plan_floor(tunnel, ground=top, area=area, water=water, frame=frame,
                   water_depth_m=water_depth_m)
        if tunnel.floor.size:
            plan.tunnels.append(tunnel)
    LOG.info(
        "tunnels: %d (%s), %d portals read from the survey",
        len(plan.tunnels),
        ", ".join(f"{t.kind} {t.polygon.area:.0f} m2" for t in plan.tunnels),
        sum(len(t.portals) for t in plan.tunnels),
    )
    return plan


# ---------------------------------------------------------------------------
# Geometry: the tube, and the mouths
# ---------------------------------------------------------------------------

# The floor and ceiling are cut into squares this size, so they follow the
# drawn floor rather than spanning a whole tunnel in one triangle.
PIECE_M = 4.0
# How far the walls reach below the floor, so no crack shows between them.
SKIRT_M = 0.3
# A mouth's headwall carries on this far past the opening at either end, and
# stands this far over the ground on the tunnel.
WING_M = 0.6
PARAPET_M = 0.3
# The ground is cut open across a mouth; the apron in front of it and the cap
# over it reach this far from the mouth, past any face that was cut.
MOUTH_REACH_M = 6.0
APRON_LIFT_M = 0.02


@dataclass
class TunnelGeometry:
    floor: np.ndarray  # (T, 3, 3) RD + NAP, facing up
    tube: np.ndarray  # walls, ceiling and the mouths' headwalls and caps


def _earcut(polygon) -> np.ndarray:
    """Triangles of a shapely polygon (any part), 2-D, anticlockwise."""
    import shapely

    from .structures import _facing_up, _triangulate

    out = []
    for part in shapely.get_parts(polygon):
        if part.geom_type != "Polygon" or part.is_empty:
            continue
        rings = [np.asarray(part.exterior.coords)[:-1]] + [
            np.asarray(r.coords)[:-1] for r in part.interiors
        ]
        flat = _triangulate(rings)
        if len(flat):
            out.append(_facing_up(np.dstack([flat, np.zeros((len(flat), 3, 1))]))[:, :, :2])
    return np.concatenate(out) if out else np.zeros((0, 3, 2))


def _pieces(region, size: float = PIECE_M):
    """`region` cut into squares of `size`, as triangles."""
    import shapely

    x0, y0, x1, y1 = region.bounds
    xs = np.arange(np.floor(x0 / size) * size, x1, size)
    ys = np.arange(np.floor(y0 / size) * size, y1, size)
    gx, gy = np.meshgrid(xs, ys)
    boxes = shapely.box(gx.ravel(), gy.ravel(), gx.ravel() + size, gy.ravel() + size)
    pieces = shapely.intersection(boxes, region)
    pieces = pieces[~shapely.is_empty(pieces) & (shapely.area(pieces) > 1e-4)]
    chunks = [_earcut(piece) for piece in pieces]
    chunks = [c for c in chunks if len(c)]
    return np.concatenate(chunks) if chunks else np.zeros((0, 3, 2))


def _portal_line(portal: Portal, area):
    """A portal's line within the area, or None."""
    import shapely

    if len(portal.points) < 2:
        return None
    line = shapely.LineString(portal.points)
    if area is not None:
        line = shapely.intersection(line, area)
        parts = [g for g in shapely.get_parts(line) if g.geom_type == "LineString" and g.length > 0.5]
        if not parts:
            return None
        line = max(parts, key=lambda g: g.length)
    return line


def _strip(points: np.ndarray, length: float) -> np.ndarray:
    """Points along a polyline at most `length` apart."""
    import shapely

    line = shapely.segmentize(shapely.LineString(points), length)
    return np.asarray(shapely.get_coordinates(line))


def _extended(points: np.ndarray, by: float) -> tuple[np.ndarray, int, int]:
    """A polyline carried on `by` past both ends; also the index range of the
    original points in it."""
    if len(points) < 2 or by <= 0:
        return points, 0, len(points)
    head = points[0] - points[1]
    head /= max(float(np.hypot(*head)), 1e-9)
    tail = points[-1] - points[-2]
    tail /= max(float(np.hypot(*tail)), 1e-9)
    out = np.vstack([points[0] + by * head, points, points[-1] + by * tail])
    return out, 1, len(points) + 1


def tunnel_geometry(tunnel: Tunnel, *, area=None, top=None) -> TunnelGeometry:
    """The tube of one tunnel, inside `area`.

    A floor that follows the drawn profile, walls and a ceiling that look
    inward -- from inside, the back of the terrain is no ceiling: an engine
    culls it and shows the sky -- and at every mouth a headwall over the
    opening, a cap over the roof's edge and an apron on the cutting floor,
    which hide where the ground was cut open to let the road in
    (`open_mouths`). `top(x, y)` is what stands on the tunnel.
    """
    import shapely

    from .structures import _oriented, wall_strip

    region = shapely.intersection(tunnel.polygon, area) if area is not None else tunnel.polygon
    region = shapely.make_valid(region)
    if region.is_empty or not tunnel.floor.size:
        return TunnelGeometry(np.zeros((0, 3, 3)), np.zeros((0, 3, 3)))
    clearance = float(tunnel.clearance_m)

    def floor_z(xy):
        z = tunnel.floor_at(xy[:, 0], xy[:, 1])
        if not np.isfinite(z).all():
            # A point on a sliver the raster missed: the nearest floor will do.
            good = np.isfinite(z)
            fallback = float(np.nanmedian(tunnel.floor[np.isfinite(tunnel.floor)]))
            z = np.where(good, z, fallback)
        return z

    flat = _pieces(region)
    corners = flat.reshape(-1, 2)
    z = floor_z(corners).reshape(-1, 3, 1)
    floor = [np.dstack([flat, z])]
    # The ceiling: the same pieces, a clearance up, looking down.
    tube = [np.dstack([flat, z + clearance])[:, ::-1]]

    # Walls along the outline, except across a mouth and along the edge of
    # the area, where the tunnel carries on out of the model.
    lines = [_portal_line(p, None) for p in tunnel.portals]
    lines = [line for line in lines if line is not None]
    mouths = shapely.union_all(np.array(lines, dtype=object)) if lines else None
    frame = area.exterior if area is not None else None
    for part in shapely.get_parts(region):
        if part.geom_type != "Polygon":
            continue
        rings = [np.asarray(part.exterior.coords)] + [np.asarray(r.coords) for r in part.interiors]
        for index, ring in enumerate(rings):
            # The tunnel on the right of the way the ring runs, so each wall
            # looks into it.
            points = _oriented(ring, ccw=index != 0)
            points = _strip(np.vstack([points, points[:1]]), 2.0)[:-1]
            if len(points) < 3:
                continue
            step = np.roll(points, -1, axis=0) - points
            middle = shapely.points(points + 0.5 * step)
            keep = np.ones(len(points), dtype=bool)
            if mouths is not None:
                keep &= ~shapely.dwithin(mouths, middle, 0.6)
            if frame is not None:
                keep &= ~shapely.dwithin(frame, middle, 0.05)
            base = floor_z(points)
            tube.append(wall_strip(points, base + clearance, base - SKIRT_M, keep))

    aprons = []
    for portal in tunnel.portals:
        line = _portal_line(portal, area)
        if line is None:
            continue
        mouth = _mouth(portal, line, tunnel, floor_z, top, clearance)
        tube.extend(mouth["tube"])
        aprons.extend(mouth["floor"])

    floor.extend(aprons)
    floor = [f for f in floor if len(f)]
    tube = [t for t in tube if len(t)]
    return TunnelGeometry(
        np.concatenate(floor) if floor else np.zeros((0, 3, 3)),
        np.concatenate(tube) if tube else np.zeros((0, 3, 3)),
    )


def _quad_strip(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Faces between two polylines of 3-D points walked side by side,
    wound so they look the way `a`-to-next-to-`b` turns anticlockwise."""
    first = np.stack([a[:-1], a[1:], b[1:]], axis=1)
    second = np.stack([a[:-1], b[1:], b[:-1]], axis=1)
    return np.concatenate([first, second])


def _mouth(portal: Portal, line, tunnel: Tunnel, floor_z, top, clearance: float) -> dict:
    """Headwall, cap, apron and wing walls for one mouth."""
    import shapely

    from .structures import _facing_up, wall_strip

    outward = np.asarray(portal.outward, dtype=np.float64)
    points = _strip(np.asarray(shapely.get_coordinates(line)), 1.0)
    # The ring the portal came from runs with the tunnel on its left; keep
    # that, so a wall along it looks outward.
    # Outward is then on the right of the way the points run.
    tangent = points[-1] - points[0]
    if float(tangent[0] * outward[1] - tangent[1] * outward[0]) > 0:
        points = points[::-1]
    reach = MOUTH_REACH_M
    floor_at_mouth = floor_z(points)
    ceiling = floor_at_mouth + clearance

    # What the cap stands over: the highest ground on the tunnel behind the
    # mouth, and never less than a roof over the ceiling.
    if top is not None:
        behind = np.column_stack([
            np.asarray(top(*(points - d * outward).T), dtype=np.float64)
            for d in np.arange(1.0, reach + 1.0, 1.0)
        ])
        ground = np.nanmax(np.where(np.isfinite(behind), behind, -np.inf), axis=1)
        ground = np.where(np.isfinite(ground), ground, ceiling + MIN_ROOF_M)
    else:
        ground = np.full(len(points), float(portal.top_nap))
    cap = np.maximum(ground, ceiling + MIN_ROOF_M) + PARAPET_M

    wide, lo, hi = _extended(points, WING_M)
    wide_floor = np.concatenate([floor_at_mouth[:1], floor_at_mouth, floor_at_mouth[-1:]])
    wide_cap = np.concatenate([cap[:1], cap, cap[-1:]])

    tube = []
    # The headwall's face: over the opening from the ceiling up, and past its
    # ends from below the floor up, so the corners close.
    bottom = wide_floor - SKIRT_M
    bottom[lo:hi] = ceiling
    # The ends of the opening twice, once at each bottom, so the face steps
    # straight up rather than slanting across the corner.
    face_points = np.vstack([wide[:lo + 1], wide[lo:hi], wide[hi - 1:]])
    face_bottom = np.concatenate([bottom[:lo] if lo else [], [wide_floor[lo] - SKIRT_M], bottom[lo:hi],
                                  [wide_floor[hi - 1] - SKIRT_M], bottom[hi:]])
    face_top = np.concatenate([wide_cap[:lo + 1], wide_cap[lo:hi], wide_cap[hi - 1:]])
    tube.append(wall_strip(face_points, face_top, face_bottom, closed=False))

    # The cap: a block over the roof's edge, `reach` deep.
    inner = wide - reach * outward
    tube.append(_facing_up(_quad_strip(np.column_stack([wide, wide_cap]), np.column_stack([inner, wide_cap]))))
    # Its back, looking away from the mouth, down into the ground.
    back_ground = (
        np.asarray(top(*inner.T), dtype=np.float64) if top is not None else wide_cap - 1.0
    )
    back_ground = np.where(np.isfinite(back_ground), back_ground, wide_cap - 1.0)
    tube.append(wall_strip(inner[::-1], wide_cap[::-1], np.minimum(back_ground, wide_cap)[::-1] - 0.5, closed=False))
    # Its two ends.
    for end, sign in ((0, -1.0), (len(wide) - 1, 1.0)):
        # Each end looks away from the block, along the mouth.
        pair = np.array([wide[end], inner[end]]) if sign > 0 else np.array([inner[end], wide[end]])
        z_top = np.full(2, wide_cap[end])
        z_bottom = np.full(2, wide_floor[end] - SKIRT_M)
        tube.append(wall_strip(pair, z_top, z_bottom, closed=False))

    # The apron: the cutting floor in front of the mouth, following the ramp.
    rise = 0.0
    if np.isfinite(portal.ramp_nap) and portal.ramp_at_m > 0:
        rise = (portal.ramp_nap - portal.floor_nap) / portal.ramp_at_m
    outer = wide + reach * outward
    near_z = wide_floor + APRON_LIFT_M
    far_z = wide_floor + rise * reach + APRON_LIFT_M
    apron = _facing_up(_quad_strip(np.column_stack([outer, far_z]), np.column_stack([wide, near_z])))
    # Wing walls down both sides of the apron, from below it to above the
    # ground beside the cutting: they hide the cut ends. Both faces, as the
    # cutting side and the street side both show.
    for end in (0, len(wide) - 1):
        side = np.array([wide[end], outer[end]])
        beside = (
            np.asarray(top(*side.T), dtype=np.float64) if top is not None
            else np.array([wide_cap[end], wide_cap[end]])
        )
        beside = np.where(np.isfinite(beside), beside, wide_cap[end])
        z_bottom = np.array([near_z[end], far_z[end]]) - SKIRT_M
        z_top = np.maximum(beside + PARAPET_M, z_bottom + clearance * 0.5)
        z_top[0] = max(z_top[0], wide_cap[end])
        wall = wall_strip(side, z_top, z_bottom, closed=False)
        tube.append(wall)
        tube.append(wall[:, ::-1])
    return {"tube": tube, "floor": [apron]}


# ---------------------------------------------------------------------------
# Opening the ground at the mouths
# ---------------------------------------------------------------------------


@dataclass
class Mouth:
    """Where the ground has to open: a portal line, and what is either side."""

    line: object  # shapely LineString, RD
    outward: np.ndarray
    floor: np.ndarray  # along the line, as `points`
    points: np.ndarray
    ceiling_m: float  # clearance
    rise: float  # slope of the apron, per metre out
    polygon: object = None  # the tunnel's outline, which says what is inside


def mouths(plan: TunnelPlan, area=None) -> list[Mouth]:
    out = []
    for tunnel in plan.tunnels:
        for portal in tunnel.portals:
            line = _portal_line(portal, area)
            if line is None:
                continue
            points = _strip(np.asarray(line.coords), 1.0)
            floor = tunnel.floor_at(points[:, 0], points[:, 1])
            floor = np.where(np.isfinite(floor), floor, portal.floor_nap)
            rise = 0.0
            if np.isfinite(portal.ramp_nap) and portal.ramp_at_m > 0:
                rise = (portal.ramp_nap - portal.floor_nap) / portal.ramp_at_m
            out.append(Mouth(line, np.asarray(portal.outward), floor, points,
                             float(tunnel.clearance_m), rise, tunnel.polygon))
    return out


def _offsets(mouth: Mouth, xy: np.ndarray):
    """For each point: how far out of the mouth (+) or into the tunnel (-) it
    is, whether it lies across the opening's width, and the floor at the
    nearest point of the mouth."""
    import shapely

    a, b = mouth.points[0], mouth.points[-1]
    axis = b - a
    length = max(float(np.hypot(*axis)), 1e-9)
    axis = axis / length
    rel = xy - a
    along = rel @ axis
    where = shapely.points(xy[:, 0], xy[:, 1])
    distance = shapely.distance(mouth.line, where)
    if mouth.polygon is not None:
        # Inside is inside the tunnel. A mouth can bend through a right angle,
        # as the Maastunnel's south one does, and one direction for "out"
        # then puts the ground just behind its bulge in front of it.
        side = np.where(shapely.contains(mouth.polygon, where), -1.0, 1.0)
    else:
        side = np.where(rel @ mouth.outward >= 0, 1.0, -1.0)
    across = (along >= -WING_M) & (along <= length + WING_M)
    stations = np.linspace(0.0, length, len(mouth.points))
    nearest = np.clip(np.searchsorted(stations, along), 0, len(mouth.points) - 1)
    return distance * side, across, mouth.floor[nearest]


def open_mouths(mesh, mouth_list: list[Mouth], *, face_arrays=(), loop_arrays=(),
                vertex_arrays=(), reach: float = MOUTH_REACH_M - 0.5) -> dict:
    """Open the ground across each mouth, in place.

    The ground is a height field, and a height field cannot have a hole in
    its side: wherever a cutting meets a tunnel, the ground climbs straight
    from the cutting floor up onto the roof, across the opening. So:

    * outside the mouth, and on it, the ground is held down under the apron;
    * inside, within `reach`, it is held up over the ceiling, out of the way;
    * the faces between the two -- one row, spanning the mouth line from the
      inside -- go.

    The hole that leaves is all on the tunnel side of the mouth, under the
    tunnel's own headwall and cap (`tunnel_geometry`), whatever size the
    ground's faces are; the road in front of the tunnel keeps every face.

    `mesh` has `vertices`, `face_sizes`, `face_loops` and optionally
    `measured_z` and `edges`. `face_arrays`, `loop_arrays` and
    `vertex_arrays` name its other per-face, per-loop and per-vertex arrays,
    which are cut down with it.
    """
    import shapely

    if not mouth_list or not len(mesh.face_sizes):
        return {"faces_cut": 0, "vertices_moved": 0}
    vertices = mesh.vertices
    xy = vertices[:, :2]
    columns = [vertices[:, 2]]
    if getattr(mesh, "measured_z", None) is not None:
        columns.append(mesh.measured_z)

    starts = np.concatenate([[0], np.cumsum(mesh.face_sizes)[:-1]]).astype(np.int64)
    face_of_loop = np.repeat(np.arange(len(mesh.face_sizes)), mesh.face_sizes)
    drop = np.zeros(len(mesh.face_sizes), dtype=bool)
    moved = 0
    on_line = 0.05
    for mouth in mouth_list:
        x0, y0, x1, y1 = mouth.line.buffer(reach + 50.0).bounds
        near = np.flatnonzero((xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
        if not len(near):
            continue
        offset = np.full(len(xy), np.nan)
        across = np.zeros(len(xy), dtype=bool)
        floor = np.full(len(xy), np.nan)
        offset[near], across[near], floor[near] = _offsets(mouth, xy[near])

        # Faces that span the mouth from the inside go.
        loop_offset = offset[mesh.face_loops]
        loop_across = across[mesh.face_loops]
        inside = np.zeros(len(mesh.face_sizes), dtype=bool)
        outside = np.zeros(len(mesh.face_sizes), dtype=bool)
        np.logical_or.at(inside, face_of_loop, loop_across & (loop_offset < -on_line))
        np.logical_or.at(outside, face_of_loop, loop_across & (loop_offset >= -on_line))
        spanning = np.flatnonzero(inside & outside & ~drop)
        if len(spanning):
            polygons = np.array([
                shapely.Polygon(xy[mesh.face_loops[starts[f]: starts[f] + mesh.face_sizes[f]]])
                for f in spanning
            ], dtype=object)
            drop[spanning[shapely.intersects(polygons, mouth.line.buffer(on_line))]] = True

        out = across & (offset >= -on_line) & (offset <= reach)
        inn = across & (offset < -on_line) & (offset >= -reach)
        apron = floor + mouth.rise * np.maximum(np.nan_to_num(offset), 0.0)
        roof = floor + mouth.ceiling_m + MIN_ROOF_M
        for column in columns:
            low = out & (column > apron - APRON_LIFT_M)
            column[low] = apron[low] - APRON_LIFT_M
            high = inn & (column < roof)
            column[high] = roof[high]
            moved += int(low.sum() + high.sum())

    cut = int(drop.sum())
    area = 0.0
    if cut:
        # What was opened, so a check of the ground as one closed sheet can
        # tell a mouth from a crack: the area, and where it is.
        opened = []
        for f in np.flatnonzero(drop):
            ring = xy[mesh.face_loops[starts[f]: starts[f] + mesh.face_sizes[f]]]
            opened.append(shapely.make_valid(shapely.Polygon(ring)))
            ring = ring - ring.mean(axis=0)
            area += 0.5 * abs(float(np.dot(ring[:, 0], np.roll(ring[:, 1], -1))
                                    - np.dot(np.roll(ring[:, 0], -1), ring[:, 1])))
        _drop_faces(mesh, drop, face_arrays=face_arrays, loop_arrays=loop_arrays,
                    vertex_arrays=vertex_arrays)
        previous = getattr(mesh, "opened", None)
        mesh.opened = shapely.union_all(np.array(
            opened + ([previous] if previous is not None else []), dtype=object
        ))
        mesh.mouths_m2 = float(getattr(mesh, "mouths_m2", 0.0) or 0.0) + area
    if mouth_list:
        mesh.mouths = list(getattr(mesh, "mouths", []) or []) + [m.line for m in mouth_list]
    return {"faces_cut": cut, "vertices_moved": moved, "area_m2": round(area, 2)}


def _drop_faces(mesh, drop: np.ndarray, *, face_arrays=(), loop_arrays=(), vertex_arrays=()) -> None:
    """Remove faces from a mesh in place, and the vertices nothing uses then."""
    keep = ~drop
    sizes = np.asarray(mesh.face_sizes)
    loop_keep = np.repeat(keep, sizes)
    for name in ("face_loops", *loop_arrays):
        setattr(mesh, name, np.asarray(getattr(mesh, name))[loop_keep])
    mesh.face_sizes = sizes[keep]
    for name in face_arrays:
        # A name on the mesh, or (owner, name) for one kept beside it.
        owner, name = (mesh, name) if isinstance(name, str) else name
        setattr(owner, name, np.asarray(getattr(owner, name))[keep])

    used = np.zeros(len(mesh.vertices), dtype=bool)
    used[mesh.face_loops] = True
    remap = np.cumsum(used) - 1
    mesh.face_loops = remap[mesh.face_loops]
    for name in ("vertices", "measured_z", "vertex_on", *vertex_arrays):
        if getattr(mesh, name, None) is not None:
            setattr(mesh, name, np.asarray(getattr(mesh, name))[used])
    edges = getattr(mesh, "edges", None)
    if edges is not None and len(edges):
        whole = used[edges].all(axis=1)
        mesh.edges = remap[edges[whole]]
        if getattr(mesh, "edge_on", None) is not None:
            mesh.edge_on = np.asarray(mesh.edge_on)[whole]
    # Cached point location no longer matches the faces.
    if hasattr(mesh, "_locator"):
        try:
            del mesh._locator
        except AttributeError:
            pass


__all__ = [
    "CLEARANCE_M",
    "Mouth",
    "TunnelGeometry",
    "mouths",
    "open_mouths",
    "top_reader",
    "tunnel_geometry",
    "GRADE",
    "Portal",
    "ROOF_M",
    "Tunnel",
    "TunnelPlan",
    "descend_from",
    "find_portals",
    "grade_envelope",
    "group_parts",
    "plan_floor",
    "plan_tunnels",
    "water_top",
]
