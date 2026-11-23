"""What the water boards know about their bridges, locks and weirs.

Every Dutch water board keeps a register (legger) of the structures in its
water, published together by Het Waterschapshuis through PDOK as IMWA. The
BGT draws a bridge's deck; the water board knows how much room a boat has
under it. Over the Utrecht centre square kilometre all fourteen of its
bridges are there, and every one is masonry, with its underside height in NAP
(4.13 to 5.07 m), the clear width of its opening (5 to 14 m), and its length
along the water.

Locks and weirs are points with a direction and a few measurements: the
width of a lock's chamber, a weir's crest height and width, what it is made
of. The geometry here (`lock_geometry`, `weir_geometry`) is built from those
and from the water polygon the structure stands in.

`richting`, the direction, is not used for bridges: it is the bridge's axis
on some and the waterway's on others. The water itself says which way is
across.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import BBox

LOG = logging.getLogger(__name__)

IMWA_URL = "https://api.pdok.nl/hwh/waterschappen-kunstwerken-imwa/ogc/v2/collections"


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _point(feature: dict) -> tuple[float, float] | None:
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates") or []
    if geometry.get("type") == "Point" and len(coords) >= 2:
        return float(coords[0]), float(coords[1])
    if geometry.get("type") in ("Polygon", "MultiPolygon", "LineString", "MultiLineString"):
        import shapely
        from shapely.geometry import shape

        try:
            centre = shapely.centroid(shape(geometry))
        except Exception:  # noqa: BLE001
            return None
        return float(centre.x), float(centre.y)
    return None


@dataclass
class WaterBridge:
    xy: tuple[float, float]
    code: str
    # The underside at its highest, in NAP: what a boat has to pass under.
    soffit_nap: float | None
    # The clear width of the opening.
    span_m: float | None
    # How far the bridge runs along the water.
    length_m: float | None
    material: str
    kind: str
    manager: str
    name: str = ""

    @property
    def masonry(self) -> bool:
        return any(word in self.material for word in ("metsel", "steen", "baksteen"))


@dataclass
class Lock:
    xy: tuple[float, float]
    name: str
    kind: str
    # The chamber's clear width.
    width_m: float | None
    # Compass degrees, either along the lock or across it: see `lock_geometry`.
    direction_deg: float | None
    sill_nap: float | None = None


@dataclass
class Weir:
    xy: tuple[float, float]
    kind: str
    crest_width_m: float | None
    crest_nap: float | None
    material: str
    direction_deg: float | None


@dataclass
class Waterworks:
    bridges: list[WaterBridge] = field(default_factory=list)
    locks: list[Lock] = field(default_factory=list)
    weirs: list[Weir] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.bridges or self.locks or self.weirs)


def parse_bridges(features: list[dict]) -> list[WaterBridge]:
    out = []
    for feature in features:
        props = feature.get("properties") or {}
        xy = _point(feature)
        if xy is None or str(props.get("statusobject") or "gerealiseerd") != "gerealiseerd":
            continue
        out.append(
            WaterBridge(
                xy=xy,
                code=str(props.get("code") or ""),
                soffit_nap=_number(props.get("hoogteonderzijde")),
                span_m=_number(props.get("breedtedoorvaart")),
                length_m=_number(props.get("lengte")),
                material=str(props.get("typemateriaal") or "").lower(),
                kind=str(props.get("typebrug") or ""),
                manager=str(props.get("waterbeheerder") or ""),
                name=str(props.get("naam") or ""),
            )
        )
    return out


def parse_locks(features: list[dict]) -> list[Lock]:
    out = []
    for feature in features:
        props = feature.get("properties") or {}
        xy = _point(feature)
        if xy is None or str(props.get("statusobject") or "gerealiseerd") != "gerealiseerd":
            continue
        out.append(
            Lock(
                xy=xy,
                name=str(props.get("naam") or ""),
                kind=str(props.get("typesluis") or ""),
                width_m=_number(props.get("doorvaartbreedte")) or _number(props.get("breedte")),
                direction_deg=_number(props.get("richting")),
                sill_nap=_number(props.get("hoogtedrempel")),
            )
        )
    # The same lock, registered by two water boards a metre apart: one entry,
    # with whatever either of them measured.
    kept: list[Lock] = []
    for lock in out:
        twin = next(
            (k for k in kept if np.hypot(lock.xy[0] - k.xy[0], lock.xy[1] - k.xy[1]) < 5.0),
            None,
        )
        if twin is None:
            kept.append(lock)
            continue
        twin.width_m = twin.width_m or lock.width_m
        twin.sill_nap = twin.sill_nap if twin.sill_nap is not None else lock.sill_nap
        twin.direction_deg = twin.direction_deg if twin.direction_deg is not None else lock.direction_deg
        if twin.kind in ("", "overig"):
            twin.kind = lock.kind
    return kept


def parse_weirs(features: list[dict]) -> list[Weir]:
    out = []
    for feature in features:
        props = feature.get("properties") or {}
        xy = _point(feature)
        if xy is None or str(props.get("statusobject") or "gerealiseerd") != "gerealiseerd":
            continue
        out.append(
            Weir(
                xy=xy,
                kind=str(props.get("typestuw") or ""),
                crest_width_m=_number(props.get("kruinbreedte")),
                crest_nap=_number(props.get("hoogteconstructie")),
                material=str(props.get("typemateriaal") or "").lower(),
                direction_deg=_number(props.get("richting")),
            )
        )
    return out


def fetch_waterworks(
    bbox: BBox,
    *,
    url: str = IMWA_URL,
    timeout: float = 120.0,
    cache_dir: Path | None = None,
    want: tuple[str, ...] = ("brug", "sluis", "stuw"),
) -> Waterworks:
    """The water boards' bridges, locks and weirs in `bbox`, or none of them
    when the register is away. Everything here improves on geometry the model
    already has, so a failure is a warning."""
    from .http_util import short_error
    from .ogcapi import fetch_items

    result = Waterworks()
    # A lock or weir just outside still holds the water inside.
    search = bbox.buffered(25.0)
    for collection, parse, target in (
        ("brug", parse_bridges, result.bridges),
        ("sluis", parse_locks, result.locks),
        ("stuw", parse_weirs, result.weirs),
    ):
        if collection not in want:
            continue
        try:
            features = fetch_items(
                f"{url}/{collection}/items",
                search,
                description=f"water board {collection}",
                timeout=timeout,
                cache_dir=cache_dir,
                cache_name=f"imwa_{collection}",
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring
            LOG.warning(
                "could not read the water boards' %s register (%s)",
                collection, short_error(exc),
            )
            continue
        target.extend(parse(features))
    LOG.info(
        "water boards: %d bridges, %d locks, %d weirs",
        len(result.bridges), len(result.locks), len(result.weirs),
    )
    return result


# --------------------------------------------------------------------------
# Reading the water
# --------------------------------------------------------------------------


def water_geometry(water) -> object | None:
    """The water surfaces as one shapely geometry."""
    import shapely

    polygons = []
    for body in water or []:
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in body.rings]
        rings = [r for r in rings if len(r) >= 4]
        if rings:
            valid = shapely.make_valid(shapely.Polygon(rings[0], rings[1:]))
            polygons.extend(p for p in shapely.get_parts(valid) if p.geom_type == "Polygon")
    if not polygons:
        return None
    return shapely.union_all(np.array(polygons, dtype=object))


def water_level_at(water, xy, default: float | None = None) -> float | None:
    """The level of the water body under `xy`, or the nearest within 30 m."""
    import shapely

    point = shapely.Point(float(xy[0]), float(xy[1]))
    best = None
    for body in water or []:
        ring = np.asarray(body.rings[0], dtype=np.float64)[:, :2]
        if len(ring) < 4:
            continue
        polygon = shapely.make_valid(shapely.Polygon(ring))
        distance = polygon.distance(point)
        if distance <= 30.0 and (best is None or distance < best[0]):
            best = (distance, float(body.level_nap))
    return best[1] if best is not None else default


def narrowest_crossing(water, centre, *, reach: float = 150.0, step_deg: float = 3.0):
    """The shortest line across the water through `centre`.

    Returns ``(length, direction, middle)``: the water's width there, the unit
    vector across it, and the middle of the crossing. None when `centre` is
    not in the water. Across a canal the shortest line through a point is the
    one at right angles to its banks, which is the way a bridge over it spans.
    """
    import shapely

    if water is None:
        return None
    c = np.asarray(centre, dtype=np.float64)
    local = shapely.intersection(water, shapely.box(c[0] - reach, c[1] - reach, c[0] + reach, c[1] + reach))
    if local.is_empty:
        return None
    point = shapely.Point(c[0], c[1])
    if local.distance(point) > 0.5:
        return None
    best = None
    for degrees in np.arange(0.0, 180.0, step_deg):
        u = np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])
        line = shapely.LineString([c - u * reach, c + u * reach])
        pieces = [
            p for p in shapely.get_parts(line.intersection(local))
            if p.geom_type == "LineString" and p.distance(point) < 0.5
        ]
        if not pieces:
            continue
        piece = min(pieces, key=lambda p: p.distance(point))
        ends = np.asarray(piece.coords)[:, :2]
        if best is None or piece.length < best[0]:
            best = (float(piece.length), u, ends.mean(axis=0))
    return best


# --------------------------------------------------------------------------
# Locks and weirs
# --------------------------------------------------------------------------

# Mitre gates meet at this angle off square, pointing at the higher water.
MITRE_DEG = 18.0
GATE_THICKNESS_M = 0.35
# A balance beam runs along the top of its leaf and this far past the heel,
# over the quay, where the lock keeper pushes it.
BEAM_OVERHANG_M = 5.0
BEAM_SECTION_M = (0.3, 0.35)
# How far from the lock's register point its heads are looked for.
LOCK_SEARCH_M = 80.0
# A narrow stretch at least this long is the chamber itself, with a gate at
# each end; shorter ones are the heads, a gate in each.
CHAMBER_MIN_M = 12.0


def oriented_box(start, end, z0: float, z1: float, thickness: float) -> np.ndarray:
    """A box from `start` to `end` in plan, `thickness` wide, z0 to z1, as
    twelve outward-facing triangles."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return np.zeros((0, 3, 3))
    direction /= length
    side = np.array([-direction[1], direction[0]]) * thickness / 2.0
    plan = [start - side, end - side, end + side, start + side]
    low = [np.array([p[0], p[1], z0]) for p in plan]
    high = [np.array([p[0], p[1], z1]) for p in plan]
    v = low + high
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces])


def _chord(water, point, across, reach: float = 60.0):
    """The stretch of water across `point` along `across`: (left, right) ends."""
    import shapely

    p = np.asarray(point, dtype=np.float64)
    line = shapely.LineString([p - across * reach, p + across * reach])
    here = shapely.Point(p[0], p[1])
    pieces = [
        g for g in shapely.get_parts(line.intersection(water))
        if g.geom_type == "LineString" and g.distance(here) < 0.3
    ]
    if not pieces:
        return None
    ends = np.asarray(min(pieces, key=lambda g: g.distance(here)).coords)[:, :2]
    order = np.argsort((ends - p) @ across)
    return ends[order[0]], ends[order[-1]]


def lock_heads(lock: "Lock", water) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]]]:
    """Where a lock's gates go: the narrow stretches of the channel.

    A lock is where a waterway is walled in to the width of its gates, so its
    heads are the narrowest places near the register's point. Returns the
    channel axis and, per gate, the two wall points it closes between and its
    distance along the axis.
    """
    crossing = narrowest_crossing(water, lock.xy, reach=LOCK_SEARCH_M + 40.0)
    if crossing is None:
        return np.zeros(2), []
    width_here, across, middle = crossing
    axis = np.array([across[1], -across[0]])

    stations = np.arange(-LOCK_SEARCH_M, LOCK_SEARCH_M + 0.5, 1.0)
    chords = []
    for t in stations:
        chord = _chord(water, middle + axis * t, across)
        width = None if chord is None else float(np.linalg.norm(chord[1] - chord[0]))
        chords.append((t, chord, width))
    # Without a registered width, the narrowest the channel gets near the
    # point is the lock's: the register's point is not always on the chamber.
    widths = [w for _, _, w in chords if w is not None and w >= 2.5]
    wanted = lock.width_m or (min(widths) if widths else width_here)
    limit = max(wanted * 1.3 + 1.0, 3.0)

    runs: list[list[tuple]] = []
    current: list[tuple] = []
    for item in chords:
        t, chord, width = item
        if width is not None and 1.5 <= width <= limit:
            current.append(item)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return axis, []

    def at(run, which):
        t, chord, _ = run[which]
        return chord[0], chord[1], float(t)

    longest = max(runs, key=len)
    if len(longest) >= CHAMBER_MIN_M and "keer" not in lock.kind:
        inset = min(2, len(longest) // 4)
        return axis, [at(longest, inset), at(longest, -1 - inset)]
    # The heads nearest the point, one each side of it where there are two.
    by_distance = sorted(runs, key=lambda r: min(abs(item[0]) for item in r))
    picked = [by_distance[0]]
    for run in by_distance[1:]:
        side = np.sign(run[len(run) // 2][0]) != np.sign(picked[0][len(picked[0]) // 2][0])
        if side:
            picked.append(run)
            break
    if "keer" in lock.kind:
        picked = picked[:1]
    return axis, [at(run, len(run) // 2) for run in picked]


def lock_geometry(lock: "Lock", water, water_levels, ground_at) -> tuple[list, dict | None]:
    """Mitre gates with balance beams at each of a lock's heads.

    `water_levels(xy)` gives the water level near a point and `ground_at`
    the quay beside the lock. Returns ``(pieces, record)``, pieces being
    ``(triangles, "gate" | "beam")``.
    """
    import shapely

    axis, heads = lock_heads(lock, water)
    if not heads:
        return [], None
    pieces = []
    # Gates point at the higher water. Without levels either side, both point
    # the way the register's direction does, or along the axis.
    ahead = np.asarray(heads[-1][0] + heads[-1][1]) / 2.0 + axis * 15.0
    behind = np.asarray(heads[0][0] + heads[0][1]) / 2.0 - axis * 15.0
    level_ahead, level_behind = water_levels(ahead), water_levels(behind)
    upstream = axis
    if level_ahead is not None and level_behind is not None and level_behind > level_ahead + 0.05:
        upstream = -axis
    levels = []
    for left, right, _t in heads:
        level = water_levels((left + right) / 2.0)
        level = 0.0 if level is None else float(level)
        levels.append(level)
        width = float(np.linalg.norm(right - left))
        across = (right - left) / max(width, 1e-6)
        outside = np.vstack([left - across * 1.5, right + across * 1.5])
        quay = np.asarray(ground_at(outside[:, 0], outside[:, 1]), dtype=np.float64)
        quay = float(np.nanmax(quay)) if np.isfinite(quay).any() else level + 1.0
        top = max(quay - 0.1, level + 0.6)
        bottom = (lock.sill_nap if lock.sill_nap is not None and lock.sill_nap < level else level - 2.5)
        mitre = (left + right) / 2.0 + upstream * (width / 2.0) * np.tan(np.radians(MITRE_DEG))
        for heel in (left, right):
            pieces.append((oriented_box(heel, mitre, bottom, top, GATE_THICKNESS_M), "gate"))
            direction = (heel - mitre) / max(float(np.linalg.norm(heel - mitre)), 1e-6)
            pieces.append((
                oriented_box(mitre + direction * 0.4, heel + direction * BEAM_OVERHANG_M,
                             top + 0.05, top + 0.05 + BEAM_SECTION_M[1], BEAM_SECTION_M[0]),
                "beam",
            ))
    centre = np.mean([(l + r) / 2.0 for l, r, _ in heads], axis=0)
    record = {
        "name": lock.name,
        "kind": lock.kind,
        "rd": [round(float(centre[0]), 3), round(float(centre[1]), 3)],
        "gates": len(heads),
        "chamber_m": round(float(abs(heads[-1][2] - heads[0][2])), 1) if len(heads) > 1 else 0.0,
        "width_m": round(float(np.mean([np.linalg.norm(r - l) for l, r, _ in heads])), 2),
        "register_width_m": lock.width_m,
        "water_nap": round(float(np.mean(levels)), 2),
    }
    return pieces, record


def weir_geometry(weir: "Weir", water, water_levels) -> tuple[list, dict | None]:
    """A weir as a plate across its watercourse, with a frame over it.

    The crest is the register's height where it is believable (within two
    metres of the water); otherwise a hand's width over the water.
    """
    crossing = narrowest_crossing(water, weir.xy, reach=40.0)
    if crossing is None:
        return [], None
    width, across, middle = crossing
    if width > 40.0:
        return [], None
    level = water_levels(weir.xy)
    level = 0.0 if level is None else float(level)
    crest = weir.crest_nap if (
        weir.crest_nap is not None and level - 1.5 < weir.crest_nap < level + 2.0
    ) else level + 0.15
    plate = width if not weir.crest_width_m else min(width, max(weir.crest_width_m, 0.8))
    left = middle - across * plate / 2.0
    right = middle + across * plate / 2.0
    bottom = min(level, crest) - 1.2
    pieces = [(oriented_box(left, right, bottom, crest, 0.2), "weir")]
    if "vast" not in weir.kind:
        # Two uprights and a beam: the frame a movable weir is wound up in.
        frame_top = crest + 1.2
        for end in (left, right):
            pieces.append((oriented_box(end - across * 0.15, end + across * 0.15,
                                        bottom, frame_top, 0.3), "weir"))
        pieces.append((oriented_box(left, right, frame_top - 0.25, frame_top, 0.3), "weir"))
    record = {
        "kind": weir.kind,
        "material": weir.material,
        "rd": [round(float(middle[0]), 3), round(float(middle[1]), 3)],
        "width_m": round(float(plate), 2),
        "crest_nap": round(float(crest), 2),
        "water_nap": round(level, 2),
    }
    return pieces, record


__all__ = [
    "IMWA_URL",
    "lock_geometry",
    "lock_heads",
    "oriented_box",
    "weir_geometry",
    "Lock",
    "WaterBridge",
    "Waterworks",
    "Weir",
    "fetch_waterworks",
    "narrowest_crossing",
    "parse_bridges",
    "parse_locks",
    "parse_weirs",
    "water_geometry",
    "water_level_at",
]
