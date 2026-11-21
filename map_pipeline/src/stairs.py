"""Flights of steps, cut into treads from the survey.

Three registers know where steps are, and none says how they run:

* the BGT draws a flight of public steps as a footpath, ``voetpad op trap``:
  22 of them in Bourtange's ramparts, 13 down to the wharves of the
  Oudegracht round the Ganzenmarkt;
* the BGT's ``gebouwinstallatie`` of type ``toegangstrap`` is a stoop: the few
  steps up to a front door;
* OpenStreetMap maps steps as a line, ``highway=steps``, sometimes with how
  many there are and how wide.

Drawn flat, a flight down to a wharf is a ramp, and drawn as a block of
stone 90 cm high, a stoop is a plinth. The survey has the rest. The bare
earth inside the outline gives the direction the steps climb (a plane fitted
to it), the drawn ground at the two ends gives how far, and the outline is
cut across that direction into treads, each one riser higher than the last,
the top one level with the ground above. A stoop climbs towards its
building, as high as the surface model says its landing is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger(__name__)

RISER_M = 0.17  # a comfortable step
MIN_RISE_M = 0.2  # less than this and the survey cannot tell a step from a slope
MAX_RISE_M = 8.0
STOOP_RISE_M = (0.17, 1.2)  # what a stoop to a front door can be
STOOP_DEFAULT_M = 0.51  # three steps, where the survey cannot say
OSM_WIDTH_M = 1.5
MIN_RUN_M = 0.8
MAX_STEEPNESS = 1.25  # rise over run: 51 degrees, a steep wharf stair


@dataclass
class Flight:
    """One flight's treads: each a polygon's rings and the height of its top."""

    treads: list[tuple[list[np.ndarray], float]]
    rise_m: float
    risers: int
    source: str


def _sample_inside(polygon, spacing: float = 0.25):
    import shapely

    xmin, ymin, xmax, ymax = polygon.bounds
    xs = np.arange(xmin + spacing / 2, xmax, spacing)
    ys = np.arange(ymin + spacing / 2, ymax, spacing)
    if not len(xs) or not len(ys):
        c = polygon.representative_point()
        return np.array([[c.x, c.y]])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = shapely.contains_xy(polygon, pts[:, 0], pts[:, 1])
    return pts[inside] if inside.any() else np.array([[polygon.centroid.x, polygon.centroid.y]])


def uphill(polygon, survey_at) -> np.ndarray | None:
    """The direction the ground inside `polygon` rises, from a plane fit."""
    pts = _sample_inside(polygon)
    z = np.asarray(survey_at(pts[:, 0], pts[:, 1]), dtype=np.float64)
    ok = np.isfinite(z)
    if ok.sum() < 6:
        return None
    a = np.column_stack([pts[ok] - pts[ok].mean(axis=0), np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(a, z[ok], rcond=None)
    # Once more without what lies far off the plane: a bollard's return.
    resid = np.abs(a @ coef - z[ok])
    keep = resid <= max(3.0 * np.median(resid), 0.05)
    if keep.sum() >= 6:
        coef, *_ = np.linalg.lstsq(a[keep], z[ok][keep], rcond=None)
    g = coef[:2]
    if np.hypot(*g) < 0.02:  # under 2%: no slope worth a step
        return None
    return g / np.hypot(*g)


def cut_flight(polygon, direction, z_low: float, rise: float, *, riser_m: float = RISER_M):
    """The polygon cut across `direction` into treads climbing `rise`."""
    import shapely

    from .terrain_mesh import _polygonal

    direction = np.asarray(direction, dtype=np.float64)
    across = np.array([-direction[1], direction[0]])
    coords = np.asarray(polygon.exterior.coords)[:, :2]
    s = coords @ direction
    c = coords @ across
    s0, s1 = float(s.min()), float(s.max())
    c0, c1 = float(c.min()) - 1.0, float(c.max()) + 1.0
    risers = max(1, int(round(rise / riser_m)))
    run = (s1 - s0) / risers
    treads = []
    for k in range(risers):
        a, b = s0 + k * run, s0 + (k + 1) * run
        strip = shapely.Polygon([
            direction * a + across * c0, direction * b + across * c0,
            direction * b + across * c1, direction * a + across * c1,
        ])
        piece = _polygonal(polygon.intersection(strip))
        top = z_low + (k + 1) * rise / risers
        for part in getattr(piece, "geoms", [piece]):
            if part.is_empty or part.area < 0.02:
                continue
            part = shapely.orient_polygons(part) if hasattr(shapely, "orient_polygons") else part
            rings = [np.asarray(part.exterior.coords)] + [np.asarray(r.coords) for r in part.interiors]
            treads.append((rings, float(top)))
    return treads, risers


def _ends(polygon, direction, ground_at, inset_m: float = 0.3, beyond_m: float = 0.7):
    """The ground at the low and the high end of the flight.

    Read just beyond each end, on the street the flight leaves and the wharf
    it arrives at, where there is ground there; else just inside the end.
    Inside a flight the half-metre survey is half the step and half the drop
    beside it.
    """
    coords = np.asarray(polygon.exterior.coords)[:, :2]
    s = coords @ direction
    low = coords[s <= s.min() + 0.6]
    high = coords[s >= s.max() - 0.6]

    def read(points, outward):
        for offset in (beyond_m, -inset_m):
            at = points + outward * offset
            z = np.asarray(ground_at(at[:, 0], at[:, 1]), dtype=np.float64)
            if np.isfinite(z).any():
                return float(np.nanmedian(z))
        return None

    z_low = read(low, -direction)
    z_high = read(high, direction)
    if z_low is None or z_high is None:
        return None
    return z_low, z_high


def long_axis(polygon) -> tuple[np.ndarray, float]:
    """The outline's long side as a unit vector, and how much longer it is."""
    rect = polygon.minimum_rotated_rectangle
    corners = np.asarray(rect.exterior.coords)[:4, :2]
    edges = np.diff(np.vstack([corners, corners[:1]]), axis=0)
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    longest = edges[int(np.argmax(lengths))]
    short = float(lengths.min())
    return longest / max(np.hypot(*longest), 1e-9), float(lengths.max()) / max(short, 1e-6)


def public_flight(polygon, *, ground_at, survey_at, source: str, step_count=None) -> Flight | None:
    """A flight of public steps over its outline, or None if it is flat.

    A flight climbs along its outline's long side, the treads across it, and
    where the outline is that long the survey only says which end is up: a
    plane fitted to the bare earth reads the drop of the quay wall beside a
    flight down to a wharf as the way it climbs, and at the Oudegracht cut
    every such flight across its width. Only a squat outline goes by the
    plane, or by whichever axis climbs most.
    """
    axis, aspect = long_axis(polygon)
    candidates = [axis]
    if aspect < 1.3:
        fitted = uphill(polygon, survey_at) if survey_at is not None else None
        candidates = ([fitted] if fitted is not None else []) + [axis, np.array([-axis[1], axis[0]])]
    best = None
    for direction in candidates:
        for read in (survey_at, ground_at):
            if read is None:
                continue
            ends = _ends(polygon, direction, read)
            if ends is None:
                continue
            climb = ends[1] - ends[0]
            if best is None or abs(climb) > abs(best[1]):
                best = (direction if climb >= 0 else -direction, climb)
            break
    if best is None:
        return None
    direction = best[0]
    ends = _ends(polygon, direction, ground_at) or _ends(polygon, direction, survey_at)
    if ends is None:
        return None
    z_low, z_high = ends
    rise = z_high - z_low
    if rise < MIN_RISE_M and abs(best[1]) >= MIN_RISE_M:
        # The drawn ground smooths what the survey shows: trust the survey.
        rise = abs(best[1])
    if rise < MIN_RISE_M and step_count:
        rise = RISER_M * float(step_count)
    if not MIN_RISE_M <= rise <= MAX_RISE_M:
        return None
    coords = np.asarray(polygon.exterior.coords)[:, :2] @ direction
    run = float(coords.max() - coords.min())
    # Steeper than the steepest wharf stair is the quay wall beside it read
    # as the flight, or a flight the area's edge cut down to a sliver.
    if run < MIN_RUN_M or rise > MAX_STEEPNESS * run:
        return None
    treads, risers = cut_flight(polygon, direction, z_low, rise)
    return Flight(treads, rise, risers, source) if treads else None


def stoop_flight(polygon, *, building, ground_at, surface_at=None) -> Flight | None:
    """The steps up to a front door: climbing towards `building`'s footprint."""
    import shapely

    if building is None or building.is_empty:
        return None
    centre = np.asarray(polygon.centroid.coords[0])
    near = np.asarray(shapely.shortest_line(polygon.centroid, building).coords[-1])
    direction = near - centre
    if np.hypot(*direction) < 1e-6:
        # The centre is on the wall: the side the building is on.
        inside = building.intersection(polygon.buffer(0.3))
        if inside.is_empty:
            return None
        direction = np.asarray(inside.centroid.coords[0]) - centre
        if np.hypot(*direction) < 1e-6:
            return None
    direction = direction / np.hypot(*direction)
    coords = np.asarray(polygon.exterior.coords)[:, :2]
    s = coords @ direction
    street = coords[s <= s.min() + 0.3] + direction * 0.2
    z_street = np.asarray(ground_at(street[:, 0], street[:, 1]), dtype=np.float64)
    if not np.isfinite(z_street).any():
        return None
    z_low = float(np.nanmedian(z_street))
    rise = STOOP_DEFAULT_M
    if surface_at is not None:
        pts = _sample_inside(polygon, 0.2)
        top = np.asarray(surface_at(pts[:, 0], pts[:, 1]), dtype=np.float64) - z_low
        top = top[np.isfinite(top)]
        if len(top) >= 3:
            # The lower part of what the survey sees over the stoop: above it
            # hangs whatever the facade has, a cornice or a lamp.
            measured = float(np.percentile(top, 30))
            if STOOP_RISE_M[0] <= measured <= STOOP_RISE_M[1]:
                rise = measured
    treads, risers = cut_flight(polygon, direction, z_low, rise)
    return Flight(treads, rise, risers, "stoop") if treads else None


def osm_step_outlines(osm, taken=None, *, width_m: float = OSM_WIDTH_M):
    """OpenStreetMap's steps as outlines, leaving out those the BGT drew."""
    import shapely

    out = []
    for feature in getattr(osm, "features", []) or []:
        if feature.kind != "steps" or len(feature.xy) < 2:
            continue
        try:
            width = float(str(feature.tags.get("width", "")).replace(",", ".").split()[0])
        except (ValueError, IndexError):
            width = width_m
        width = float(np.clip(width, 0.8, 6.0))
        outline = shapely.LineString(feature.xy).buffer(width / 2.0, cap_style="flat", join_style="mitre")
        if outline.is_empty or outline.area < 0.5:
            continue
        if taken is not None and not taken.is_empty and outline.intersection(taken).area > 0.3 * outline.area:
            continue
        try:
            count = int(str(feature.tags.get("step_count", "")).strip())
        except ValueError:
            count = None
        out.append((outline, count))
    return out


__all__ = [
    "Flight",
    "RISER_M",
    "cut_flight",
    "osm_step_outlines",
    "public_flight",
    "stoop_flight",
    "uphill",
]
