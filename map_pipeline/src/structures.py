"""Bridges and tunnels: the parts of the ground that are not the ground.

Everything else in this pipeline sits on the terrain. These two do not, and
until now both were simply dropped — tunnels skipped outright, bridges draped
onto the surface, which sank the Erasmusbrug into the Maas.

The two need different treatment, because the data for them is not symmetrical.

**A bridge can be measured.** The BGT publishes ``overbruggingsdeel`` split into
``dek`` and ``pijler`` — the deck and the piers holding it up, as separate
polygons — and a deck is a hard surface, so the AHN surface model sees it where
the terrain model, which is bare ground by definition, does not. Over a 1 km
square of the Maas the DSM returns a reading on all fourteen decks. The height
is therefore taken, not invented.

**A tunnel can be measured at its mouths.** Its open approaches are roads at
ground level in a cutting, and the AHN sees them down to the tunnel's edge.
Between the mouths nothing looks down and sees the road, so that part is
drawn to fit the mouths and the ground over it; see `tunnels`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bgt import fetch_current, polygon_rings
from .geo import BBox
from .surfaces import clip_ring_to_bbox

LOG = logging.getLogger(__name__)

KIND_DECK = 0
KIND_PIER = 1
KIND_TUNNEL_ROAD = 2
KIND_TUNNEL_WALL = 3
KIND_RAILING = 4
KIND_BRIDGE_SIDE = 5
KIND_ARCH = 6
KIND_LOCK_GATE = 7
KIND_LOCK_BEAM = 8
KIND_WEIR = 9

KIND_NAMES = {
    KIND_DECK: "deck",
    KIND_PIER: "pier",
    KIND_TUNNEL_ROAD: "tunnel_road",
    KIND_TUNNEL_WALL: "tunnel_wall",
    KIND_RAILING: "railing",
    KIND_BRIDGE_SIDE: "bridge_side",
    KIND_ARCH: "arch",
    KIND_LOCK_GATE: "lock_gate",
    KIND_LOCK_BEAM: "lock_beam",
    KIND_WEIR: "weir",
}

# A bridge is a body, not a sheet: its deck this thick at the crown of its
# arch, the arch springing this far above the water, a passage under it for
# a wharf this high, and the body cut into strips this wide along its span so
# its arch is a curve.
ARCH_CROWN_M = 0.9
SPRING_ABOVE_WATER_M = 0.3
PASSAGE_HEIGHT_M = 2.5
MIN_ARCH_RISE_M = 0.6
STRIP_M = 0.4
# Under this much water span an arch; over it, a beam bridge: a slab.
MAX_ARCH_SPAN_M = 30.0

# BGT type_overbruggingsdeel values worth building. "sloof" is the pile cap,
# which sits at or below the waterline and is not worth its own geometry.
DECK_TYPES = {"dek"}
PIER_TYPES = {"pijler"}

# A deck reading has to be a deck. Below this above the water it is the DSM
# seeing the river rather than the bridge; above it, a mast or a crane.
MIN_DECK_NAP = -3.0
MAX_DECK_NAP = 80.0

# How deep a deck is from its top to its underside, edge beam included. A
# canal bridge in Amsterdam is about 0.6 m at its fascia, a concrete viaduct
# over a road 1 to 1.5 m; the surface model sees only the top, so this is
# chosen, not measured (`structures.deck_thickness_m`).
DECK_THICKNESS_M = 0.8


def _empty_plan():
    from .tunnels import TunnelPlan

    return TunnelPlan()


@dataclass
class StructurePart:
    """One bridge or tunnel polygon, with what the BGT says about it."""

    rings: list[np.ndarray]
    kind: int
    level: int
    movable: bool = False


@dataclass
class StructureSet:
    parts: list[StructurePart] = field(default_factory=list)
    # Built once the terrain and the surface model are known.
    tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    tri_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    counts: dict[str, int] = field(default_factory=dict)
    stats_by_collection: dict[str, str] = field(default_factory=dict)
    # Deck heights, keyed by the part index they came from, so roads and rails
    # riding on a bridge can be lifted onto it.
    deck_levels: list[tuple[np.ndarray, float]] = field(default_factory=list)
    # Every tunnel's mouths and floor (`tunnels.TunnelPlan`), for the ground
    # to open at the mouths and the track to follow the floor.
    tunnels: object = field(default_factory=lambda: _empty_plan())
    bbox: tuple | None = None
    # One record per deck over water: its opening, and the room under it.
    bridges: list[dict] = field(default_factory=list)
    locks: list[dict] = field(default_factory=list)
    weirs: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.parts)

    def count_of(self, kind: int) -> int:
        return sum(1 for part in self.parts if part.kind == kind)

    def count_of_tris(self, kind: int) -> int:
        return int((self.tri_kind == kind).sum()) if len(self.tri_kind) else 0

    def stats(self) -> dict:
        out: dict = {
            "parts": len(self.parts),
            **{f"{name}s": self.count_of(kind) for kind, name in KIND_NAMES.items()},
            "triangles": int(len(self.tris)),
        }
        if self.tris.size:
            out["height_range_nap"] = [
                round(float(self.tris[:, :, 2].min()), 2),
                round(float(self.tris[:, :, 2].max()), 2),
            ]
        out.update(self.counts)
        return out


def _triangulate(rings: list[np.ndarray]) -> np.ndarray:
    """Earcut one ring group into 2-D triangles."""
    import mapbox_earcut

    usable = [np.asarray(r, dtype=np.float64)[:, :2] for r in rings]
    usable = [r for r in usable if len(r) >= 3]
    if not usable:
        return np.zeros((0, 3, 2))

    flat = np.vstack(usable)
    ends = np.cumsum([len(r) for r in usable]).astype(np.uint32)
    try:
        indices = mapbox_earcut.triangulate_float64(flat, ends)
    except Exception as exc:  # noqa: BLE001 - one bad outline is not fatal
        LOG.debug("could not triangulate a structure part: %s", exc)
        return np.zeros((0, 3, 2))
    if len(indices) < 3:
        return np.zeros((0, 3, 2))
    return flat[np.asarray(indices, dtype=np.int64)].reshape(-1, 3, 2)


def extrude_ring(ring: np.ndarray, top, bottom) -> np.ndarray:
    """Vertical wall around a ring, from `top` down to `bottom`.

    Either edge may be one height or one per vertex, which is what lets a pier
    follow the riverbed it stands on and a tunnel wall follow the road inside
    it. A closed ring repeats its first point; that copy is dropped here, and
    from the heights with it — keeping only one of the three in step is how
    this crashed the first time it met a real pier.
    """
    points = np.asarray(ring, dtype=np.float64)[:, :2]
    n = len(points)
    top = np.full(n, float(top)) if np.isscalar(top) else np.asarray(top, dtype=np.float64)
    bottom = (
        np.full(n, float(bottom))
        if np.isscalar(bottom)
        else np.asarray(bottom, dtype=np.float64)
    )
    if len(top) != n or len(bottom) != n:
        raise ValueError(
            f"ring has {n} points but got {len(top)} top and {len(bottom)} "
            f"bottom heights"
        )

    if n > 1 and np.allclose(points[0], points[-1], rtol=0.0, atol=1e-6):
        points, top, bottom = points[:-1], top[:-1], bottom[:-1]
    if len(points) < 3:
        return np.zeros((0, 3, 3))

    upper = np.column_stack([points, top])
    lower = np.column_stack([points, bottom])

    triangles = []
    for index in range(len(points)):
        nxt = (index + 1) % len(points)
        triangles.append([lower[index], lower[nxt], upper[nxt]])
        triangles.append([lower[index], upper[nxt], upper[index]])
    return np.asarray(triangles)


def _inside_points(rings: list[np.ndarray], spacing_m: float = 0.5) -> np.ndarray:
    """A grid of points inside a polygon, at `spacing_m` (coarser on a big one).

    The deck's height used to be read at the corners of its triangles: on a
    small canal bridge that is six points, every one on the deck's edge, where
    the surface model sees the water and the quay. One Amsterdam deck read
    +0.90 m off them against +2.61 m over the deck itself, came out below the
    ground beside it, was refused, and was guessed at five metres above the
    canal instead.
    """
    import shapely

    outer = np.asarray(rings[0], dtype=np.float64)[:, :2]
    holes = [np.asarray(r, dtype=np.float64)[:, :2] for r in rings[1:] if len(r) >= 4]
    polygon = shapely.make_valid(shapely.Polygon(outer, holes))
    if polygon.is_empty or polygon.area <= 0:
        return np.zeros((0, 2))
    step = max(spacing_m, float(np.sqrt(polygon.area / 4000.0)))
    x0, y0, x1, y1 = polygon.bounds
    gx, gy = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step))
    gx, gy = gx.ravel(), gy.ravel()
    inside = shapely.contains_xy(polygon, gx, gy)
    return np.column_stack([gx[inside], gy[inside]])


def deck_height(
    rings: list[np.ndarray], dsm_sampler, ground: float | None = None
) -> float | None:
    """The height of one deck, read off the surface model.

    A median over the whole polygon rather than a single point: the DSM over a
    bridge also catches railings, gantries and whatever was driving across when
    it was flown, and a median ignores all of them.
    """
    triangles = _triangulate(rings)
    if not len(triangles):
        return None

    points = _inside_points(rings)
    if not len(points):
        points = triangles.reshape(-1, 2)
    # Cap the work on a long viaduct; a few thousand samples is plenty for a
    # median.
    if len(points) > 4000:
        points = points[:: len(points) // 4000]

    readings = np.asarray(dsm_sampler(points[:, 0], points[:, 1]), dtype=np.float64)
    usable = readings[np.isfinite(readings)]
    usable = usable[(usable > MIN_DECK_NAP) & (usable < MAX_DECK_NAP)]
    if not len(usable):
        return None

    level = float(np.median(usable))
    # A deck the surface model puts below the ground it crosses is not a deck.
    if ground is not None and level < ground - 0.5:
        return None
    return level


def _span_axis(deck, wet):
    """The direction a bridge spans: across the water under it, or its long side."""
    import shapely

    if wet is not None:
        over = deck.intersection(wet)
        if not over.is_empty and over.area > 1.0:
            canal = wet.intersection(deck.buffer(15.0))
            rect = canal.minimum_rotated_rectangle
            corners = np.asarray(rect.exterior.coords)[:4, :2]
            edges = np.diff(np.vstack([corners, corners[:1]]), axis=0)
            flow = edges[int(np.argmax(np.hypot(edges[:, 0], edges[:, 1])))]
            flow = flow / max(np.hypot(*flow), 1e-9)
            return np.array([-flow[1], flow[0]]), over
    rect = deck.minimum_rotated_rectangle
    corners = np.asarray(rect.exterior.coords)[:4, :2]
    edges = np.diff(np.vstack([corners, corners[:1]]), axis=0)
    axis = edges[int(np.argmax(np.hypot(edges[:, 0], edges[:, 1])))]
    return axis / max(np.hypot(*axis), 1e-9), shapely.Polygon()


def _earcut_polygon(polygon) -> tuple[np.ndarray, np.ndarray]:
    """Points and triangles of one polygon, holes and all; triangles face up."""
    import mapbox_earcut

    rings = [np.asarray(polygon.exterior.coords)[:-1, :2]]
    rings += [np.asarray(r.coords)[:-1, :2] for r in polygon.interiors]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    flat = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        tris = np.asarray(mapbox_earcut.triangulate_float64(flat, ends), dtype=np.int64).reshape(-1, 3)
    except Exception:  # noqa: BLE001 - one bad outline is not fatal
        return flat, np.zeros((0, 3), dtype=np.int64)
    a, b, c = flat[tris[:, 0]], flat[tris[:, 1]], flat[tris[:, 2]]
    down = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
    tris[down] = tris[down][:, [0, 2, 1]]
    return flat, tris


def bridge_body(
    rings, level: float, *, ground_at, wet=None, water_level: float | None = None,
    passages=(), piers=(), lift: float = 0.05, crown_nap: float | None = None, arch: bool = True,
):
    """A bridge as a body: ``[(triangles, kind)]`` for its deck, sides and arch.

    The deck was a sheet with no thickness at one height, its ends a step off
    the street it meets and nothing under it: from the wharf a slab floating
    over the canal. Here the deck is cut into strips across its span and each
    point given

    * a top: the deck's level, or the ground as drawn where that is higher,
      so where the deck meets the street it is at the street's height;
    * an underside: over the water an arch, springing `SPRING_ABOVE_WATER_M`
      above the water at the quays to `ARCH_CROWN_M` under the deck at its
      crown; over the land the body goes down into the ground, an abutment;
      and where the BGT draws a passage under it (``tunneldeel`` at level -1,
      a wharf going under the bridge) the underside is `PASSAGE_HEIGHT_M` over
      the passage's floor.

    A pier in the water under it (``piers``, BGT ``pijler``) splits the
    arch: the bridges over the south Oudegracht stand on one mid-canal, and
    one arch from quay to quay had the pier cutting up through its opening.
    Each stretch of water between a quay and a pier, or two piers, has its
    own arch, and the arches land on the pier at their springing.

    The water boards' register (`waterworks`), where it has the bridge, gives
    ``crown_nap``, the underside at its highest, and ``arch``: False for a
    bridge it says is not masonry, or one that opens, which is a beam.

    The sides follow the outline from the underside to the top. A bridge over
    more than `MAX_ARCH_SPAN_M` of water is a beam bridge: a slab
    `ARCH_CROWN_M` thick.
    """
    import shapely

    from .terrain_mesh import _polygonal

    outer = np.asarray(rings[0], dtype=np.float64)[:, :2]
    holes = [np.asarray(r, dtype=np.float64)[:, :2] for r in rings[1:] if len(r) >= 4]
    deck = _polygonal(shapely.make_valid(shapely.Polygon(outer, holes)))
    if deck.is_empty or deck.area < 1.0:
        return []
    span, over = _span_axis(deck, wet)
    across = np.array([-span[1], span[0]])
    coords = np.asarray(deck.exterior.coords)[:, :2]
    s_all = coords @ span
    c_all = coords @ across
    d0, d1 = float(s_all.min()), float(s_all.max())
    arches: list[tuple[float, float]] = []  # each arch's ends, along the span
    landings: list[tuple[float, float]] = []  # the piers they land on
    spring = crown = 0.0
    if not over.is_empty:
        oc = np.vstack([np.asarray(g.exterior.coords)[:, :2] for g in getattr(over, "geoms", [over])
                        if g.geom_type == "Polygon"])
        w0, w1 = float((oc @ span).min()), float((oc @ span).max())
        if water_level is None:
            probe = np.asarray(over.representative_point().coords)
            water_level = float(np.nanmin(ground_at(probe[:, 0], probe[:, 1])))
        crown = level - ARCH_CROWN_M
        if crown_nap is not None and crown_nap > float(water_level) + 0.5:
            crown = min(float(crown_nap), level - MIN_CROWN_M)
        spring = float(water_level) + SPRING_ABOVE_WATER_M
        if arch and w1 - w0 <= MAX_ARCH_SPAN_M and crown - spring >= MIN_ARCH_RISE_M:
            cuts = []
            for pier in piers or ():
                on = pier.intersection(over)
                if on.is_empty or on.area < 0.5:
                    continue
                pc = np.vstack([np.asarray(g.exterior.coords)[:, :2]
                                for g in getattr(on, "geoms", [on]) if g.geom_type == "Polygon"])
                cuts.append((max(float((pc @ span).min()), w0), min(float((pc @ span).max()), w1)))
            start = w0
            for p0, p1 in sorted(cuts):
                if p0 - start >= 1.0:
                    arches.append((start, p0))
                landings.append((p0, p1))
                start = max(start, p1)
            if w1 - start >= 1.0:
                arches.append((start, w1))
    floors = []
    for passage in passages or ():
        under = deck.intersection(passage)
        if under.is_empty or under.area < 0.5:
            continue
        probe = np.asarray(under.representative_point().coords)
        floor = np.asarray(ground_at(probe[:, 0], probe[:, 1]), dtype=np.float64)
        if np.isfinite(floor).all():
            floors.append((under.buffer(0.05), float(floor[0])))

    def top_at(xy):
        g = np.asarray(ground_at(xy[:, 0], xy[:, 1]), dtype=np.float64)
        return np.fmax(level, np.where(np.isfinite(g), g, level)) + lift

    def under_at(xy, top):
        g = np.asarray(ground_at(xy[:, 0], xy[:, 1]), dtype=np.float64)
        z = np.where(np.isfinite(g), g - 0.3, level - 3.0)
        if arches:
            s = xy @ span
            for a0, a1 in arches:
                mid, half = 0.5 * (a0 + a1), max(0.5 * (a1 - a0), 1e-6)
                inside = (s >= a0) & (s <= a1)
                curve = spring + (crown - spring) * np.sqrt(np.clip(1.0 - ((s - mid) / half) ** 2, 0.0, 1.0))
                z = np.where(inside, curve, z)
            for p0, p1 in landings:
                z = np.where((s > p0) & (s < p1), spring, z)
        elif not over.is_empty:
            # A beam bridge: a slab over the water.
            wet_here = shapely.contains_xy(over.buffer(0.05), xy[:, 0], xy[:, 1])
            z = np.where(wet_here, crown if crown_nap is not None else level - ARCH_CROWN_M, z)
        for area, floor in floors:
            inside = shapely.contains_xy(area, xy[:, 0], xy[:, 1])
            z = np.where(inside, np.fmax(z, floor + PASSAGE_HEIGHT_M), z)
        return np.fmin(z, top - 0.3)

    tops, bottoms, sides = [], [], []
    # A cut at every arch's end too, or the strip across one runs from the
    # foot of the curve to partway up it and the arch lands on a slope.
    ends = [e for pair in arches + landings for e in pair if d0 < e < d1]
    cuts = np.unique(np.concatenate([np.arange(d0, d1, STRIP_M), [d1], ends]))
    c0, c1 = float(c_all.min()) - 1.0, float(c_all.max()) + 1.0
    boundary = deck.boundary
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-6:
            continue
        strip = shapely.Polygon([span * a + across * c0, span * b + across * c0,
                                 span * b + across * c1, span * a + across * c1])
        piece = _polygonal(deck.intersection(strip))
        for part in getattr(piece, "geoms", [piece]):
            if part.is_empty or part.area < 1e-4:
                continue
            part = shapely.orient_polygons(part)
            flat, tris = _earcut_polygon(part)
            if not len(tris):
                continue
            top = top_at(flat)
            bottom = under_at(flat, top)
            tops.append(np.dstack([flat[tris], top[tris][:, :, None]]))
            bottoms.append(np.dstack([flat[tris][:, [0, 2, 1]], bottom[tris][:, [0, 2, 1], None]]))
            # The sides: this strip's edges that are the deck's own outline.
            for ring in [part.exterior, *part.interiors]:
                pts = np.asarray(ring.coords)[:, :2]
                mids = 0.5 * (pts[:-1] + pts[1:])
                on_outline = shapely.distance(boundary, shapely.points(mids)) < 1e-4
                if not on_outline.any():
                    continue
                pa, pb = pts[:-1][on_outline], pts[1:][on_outline]
                ta, tb = top_at(pa), top_at(pb)
                ba, bb = under_at(pa, ta), under_at(pb, tb)
                low_a = np.column_stack([pa, ba])
                low_b = np.column_stack([pb, bb])
                hi_a = np.column_stack([pa, ta])
                hi_b = np.column_stack([pb, tb])
                sides.append(np.stack([low_a, low_b, hi_b], axis=1))
                sides.append(np.stack([low_a, hi_b, hi_a], axis=1))
    out = []
    # Brick where there is an arch, and on a beam bridge unless the register
    # says the bridge is not masonry or it opens (`arch` False): concrete.
    brick = bool(arches) or arch
    if tops:
        out.append((np.concatenate(tops), KIND_DECK))
    if bottoms:
        out.append((np.concatenate(bottoms), KIND_ARCH if arches else (
            KIND_BRIDGE_SIDE if brick else KIND_DECK
        )))
    if sides:
        out.append((np.concatenate(sides), KIND_BRIDGE_SIDE if brick else KIND_DECK))
    return out


def deck_corner_heights(flat, level: float, terrain, lift: float) -> np.ndarray:
    """A deck's height at each corner: its own level, or the ground if higher.

    One height for the whole part, except where the ground rises through it. A
    BGT bridge part is often a whole approach, and in dike country the ground
    climbs to meet the deck at the abutment -- so a single flat level leaves the
    last stretch buried, which put a fifth of all deck geometry under the
    terrain over a 2 km area near Dordrecht.

    Held up per corner rather than moved as a whole. Levelling the part to its
    highest ground would take a bridge over a dike and stand it on stilts; this
    way the span stays flat and only the ends lift, which is what an abutment
    looks like anyway.
    """
    flat = np.asarray(flat, dtype=np.float64)
    if not len(flat):
        return np.zeros((0, 3, 1))
    ground = np.asarray(
        terrain.sample(flat[:, :, 0].ravel(), flat[:, :, 1].ravel()), dtype=np.float64
    ).reshape(len(flat), 3, 1)
    return np.maximum(np.full((len(flat), 3, 1), float(level)), ground) + float(lift)


def _oriented(ring: np.ndarray, ccw: bool) -> np.ndarray:
    """An open ring (first point not repeated), wound the way asked."""
    points = np.asarray(ring, dtype=np.float64)[:, :2]
    if len(points) > 1 and np.allclose(points[0], points[-1], rtol=0.0, atol=1e-6):
        points = points[:-1]
    if len(points) < 3:
        return points
    x, y = points[:, 0], points[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return points if (area2 > 0) == ccw else points[::-1]


def _facing_up(triangles: np.ndarray) -> np.ndarray:
    """The same triangles, every one wound counter-clockwise from above."""
    triangles = np.array(triangles, dtype=np.float64)
    if not len(triangles):
        return triangles
    a, b, c = triangles[:, 0, :2], triangles[:, 1, :2], triangles[:, 2, :2]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = area2 < 0
    triangles[flip] = triangles[flip][:, ::-1]
    return triangles


def wall_strip(points: np.ndarray, top, bottom, keep=None, closed: bool = True) -> np.ndarray:
    """Vertical faces along a line of points, from `top` down to `bottom`.

    Like `extrude_ring`, but any edge can be left out (`keep[i]` is the edge
    from point i to the next) and the line need not close. Each face looks to
    the right of the direction the points run in, which for a ring wound
    counter-clockwise is outwards.
    """
    points = np.asarray(points, dtype=np.float64)[:, :2]
    n = len(points)
    top = np.full(n, float(top)) if np.isscalar(top) else np.asarray(top, dtype=np.float64)
    bottom = np.full(n, float(bottom)) if np.isscalar(bottom) else np.asarray(bottom, dtype=np.float64)
    edges = n if closed else n - 1
    if edges < 1:
        return np.zeros((0, 3, 3))
    keep = np.ones(edges, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    i = np.flatnonzero(keep[:edges])
    j = (i + 1) % n
    # Zero-length edges make no face.
    real = np.hypot(*(points[j] - points[i]).T) > 1e-6
    i, j = i[real], j[real]
    if not len(i):
        return np.zeros((0, 3, 3))
    lower_i = np.column_stack([points[i], bottom[i]])
    lower_j = np.column_stack([points[j], bottom[j]])
    upper_i = np.column_stack([points[i], top[i]])
    upper_j = np.column_stack([points[j], top[j]])
    first = np.stack([lower_i, lower_j, upper_j], axis=1)
    second = np.stack([lower_i, upper_j, upper_i], axis=1)
    return np.concatenate([first, second])


def deck_slab(
    rings: list[np.ndarray],
    flat: np.ndarray,
    top_at,
    thickness: float,
    *,
    shared_edge=None,
) -> np.ndarray:
    """A deck as a slab: its top, its underside `thickness` lower, and the
    sides between them round the outline.

    It used to be the top alone, a sheet of paper over the canal: from the
    water, from the quay and from a car driving under a viaduct a bridge is
    mostly its edge and its underside. `top_at(x, y)` gives the top at any
    corner, so the underside follows a deck that is held up at its ends where
    the ground rises through it (`deck_corner_heights`).

    `shared_edge(points, outward)` says, per outline edge, whether another
    deck carries on level beyond it. A BGT bridge comes in several parts --
    one per carriageway, one per span -- and a side face between two of them
    at the same height is a wall inside the bridge nobody can see.
    """
    flat = np.asarray(flat, dtype=np.float64)
    if not len(flat):
        return np.zeros((0, 3, 3))
    corners = flat.reshape(-1, 2)
    top_z = np.asarray(top_at(corners[:, 0], corners[:, 1]), dtype=np.float64).reshape(-1, 3, 1)
    top = _facing_up(np.dstack([flat, top_z]))
    if thickness <= 0:
        return top
    under = top.copy()
    under[:, :, 2] -= float(thickness)
    # Wound the other way, so the underside looks down.
    under = under[:, ::-1]

    sides = []
    for index, ring in enumerate(rings):
        # Outline anticlockwise and holes clockwise, so every side looks away
        # from the deck.
        points = _oriented(ring, ccw=index == 0)
        if len(points) < 3:
            continue
        z = np.asarray(top_at(points[:, 0], points[:, 1]), dtype=np.float64)
        keep = None
        if shared_edge is not None:
            step = np.roll(points, -1, axis=0) - points
            length = np.hypot(step[:, 0], step[:, 1])
            outward = np.column_stack([step[:, 1], -step[:, 0]]) / np.maximum(length, 1e-9)[:, None]
            middle = points + 0.5 * step
            keep = ~np.asarray(shared_edge(middle, outward, 0.5 * (z + np.roll(z, -1))), dtype=bool)
        sides.append(wall_strip(points, z, z - float(thickness), keep))
    return np.concatenate([top, under, *sides]) if sides else np.concatenate([top, under])


# A point this close to a deck's outline is on the deck, for what stands on it.
DECK_EDGE_M = 0.6


def deck_outlines(bbox: BBox, *, structures_cfg: dict, water=None) -> list:
    """Every bridge deck over water, as ``(polygon, along)``: its outline and
    the direction the water runs under it, for the ground under it.

    The survey does not see under a bridge: the bare earth there is drawn in
    from the streets at either end, a ridge at street height across the
    canal, and the ground built on it put the wharf four metres up under
    every bridge and a spike out of the canal's bed. Under a deck the ground
    is what it is either side of the bridge, along the water.
    """
    import shapely

    from .terrain_mesh import _polygonal

    fetch_kwargs = dict(
        page_limit=int(structures_cfg.get("page_limit", 1000)),
        timeout=float(structures_cfg.get("timeout_s", 180)),
        max_retries=int(structures_cfg.get("max_retries", 4)),
        max_pages=int(structures_cfg.get("max_pages", 200)),
    )
    try:
        features, _ = fetch_current(
            "overbruggingsdeel", bbox, class_field="type_overbruggingsdeel", **fetch_kwargs
        )
    except Exception as exc:  # noqa: BLE001 - the ground stays as surveyed
        LOG.warning("could not read BGT bridge decks: %s", exc)
        return []
    wet, _ = _wet(water)
    if wet is None:
        return []
    out = []
    for feature in features:
        properties = feature.get("properties") or {}
        if str(properties.get("type_overbruggingsdeel") or "").lower() not in DECK_TYPES:
            continue
        try:
            level = int(properties.get("relatieve_hoogteligging") or 0)
        except (TypeError, ValueError):
            level = 0
        if level < 1:
            continue
        for group in polygon_rings(feature.get("geometry")):
            rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in group if len(r) >= 4]
            if not rings:
                continue
            deck = _polygonal(shapely.make_valid(shapely.Polygon(rings[0], rings[1:])))
            if deck.is_empty:
                continue
            span, over = _span_axis(deck, wet)
            if over.is_empty:
                continue
            out.append((deck, np.array([span[1], -span[0]])))
    return out


def deck_surface(bbox: BBox, *, structures_cfg: dict, terrain, dsm_sampler):
    """The top of every bridge deck, as a function of position: `f(x, y)`
    gives the deck's surface where a deck is and NaN everywhere else.

    For the stages built before the bridges that have something standing on
    one. A parapet on a canal bridge is a BGT wall up at the deck's level, and
    stood on the ground under it -- in the layered style the canal bed -- it
    was a wall five metres tall closing the canal under every bridge in
    Amsterdam. Heights as `build_structures` gives the decks themselves.
    """
    import shapely

    fetch_kwargs = dict(
        page_limit=int(structures_cfg.get("page_limit", 1000)),
        timeout=float(structures_cfg.get("timeout_s", 180)),
        max_retries=int(structures_cfg.get("max_retries", 4)),
        max_pages=int(structures_cfg.get("max_pages", 200)),
    )
    lift = float(structures_cfg.get("deck_lift_m", 0.05))
    try:
        features, _ = fetch_current(
            "overbruggingsdeel", bbox, class_field="type_overbruggingsdeel", **fetch_kwargs
        )
    except Exception as exc:  # noqa: BLE001 - the barriers stay on the ground
        LOG.warning("could not read BGT bridge decks: %s", exc)
        features = []
    decks, levels = [], []
    for feature in features:
        properties = feature.get("properties") or {}
        if str(properties.get("type_overbruggingsdeel") or "").lower() not in DECK_TYPES:
            continue
        for group in polygon_rings(feature.get("geometry")):
            clipped = [clip_ring_to_bbox(ring, bbox) for ring in group]
            clipped = [r[:, :2] for r in clipped if len(r) >= 3]
            if not clipped:
                continue
            flat = _triangulate(clipped)
            if not len(flat):
                continue
            corners = flat.reshape(-1, 2)
            under = np.asarray(terrain.sample(corners[:, 0], corners[:, 1]), dtype=np.float64)
            level = (
                deck_height(clipped, dsm_sampler, ground=float(np.median(under)))
                if dsm_sampler is not None else None
            )
            if level is None:
                level = float(under.max()) + float(structures_cfg.get("fallback_clearance_m", 5.0))
            polygon = shapely.make_valid(shapely.Polygon(clipped[0], clipped[1:]))
            if not polygon.is_empty:
                decks.append(polygon)
                levels.append(level)
    if not decks:
        return lambda x, y, reach=DECK_EDGE_M: np.full(np.shape(np.atleast_1d(x)), np.nan)
    tree = shapely.STRtree(np.array(decks, dtype=object))
    levels = np.asarray(levels)

    def at(x, y, reach=DECK_EDGE_M):
        """`reach`: how far outside the outline still counts as on the deck.
        A parapet wants the edge; a tree wants to be inside (0)."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        out = np.full(x.shape, np.nan)
        points = shapely.points(x, y)
        # A parapet runs along the deck's very edge, and a wall is swept on
        # both its faces: read only inside the outline, one face stood on the
        # deck and the other on the canal bed, a comb of teeth down to the
        # water. Close to the edge is on it.
        if reach > 0:
            hit_point, hit_deck = tree.query(points, predicate="dwithin", distance=reach)
        else:
            hit_point, hit_deck = tree.query(points, predicate="within")
        if len(hit_point):
            out[hit_point] = np.fmax(out[hit_point], levels[hit_deck])
            # Held up where the ground rises through the deck, as the deck is.
            ground = np.asarray(terrain.sample(x[hit_point], y[hit_point]), dtype=np.float64)
            out[hit_point] = np.fmax(out[hit_point], ground) + lift
        return out

    return at


def build_structures(
    bbox: BBox,
    work_dir: Path,
    *,
    structures_cfg: dict,
    terrain=None,
    dsm_sampler=None,
    ground_at=None,
    water=None,
    waterworks=None,
) -> StructureSet:
    """Fetch bridge and tunnel parts and build them into geometry.

    `waterworks`, the water boards' register (`waterworks.fetch_waterworks`),
    gives a bridge over water its real underside and opening, and says which
    are masonry and get an arch.

    `water`, the WaterBody list, puts railings along the sides of every deck
    that are over water and not already walled by a BGT barrier
    (`structures.railings`).

    `ground_at(x, y)`, the ground as built where that differs from the survey
    -- the layered style cuts a canal's bed below its water -- is what a pier
    stands on, so it reaches down into the water rather than stopping above it.
    """
    result = StructureSet()
    result.bbox = tuple(bbox.as_list())
    fetch_kwargs = dict(
        page_limit=int(structures_cfg.get("page_limit", 1000)),
        timeout=float(structures_cfg.get("timeout_s", 180)),
        max_retries=int(structures_cfg.get("max_retries", 4)),
        max_pages=int(structures_cfg.get("max_pages", 200)),
    )

    want_bridges = bool(structures_cfg.get("bridges", True))
    want_tunnels = bool(structures_cfg.get("tunnels", True))

    if want_bridges:
        features, stats = fetch_current(
            "overbruggingsdeel", bbox, class_field="type_overbruggingsdeel",
            **fetch_kwargs,
        )
        result.stats_by_collection["overbruggingsdeel"] = stats.summary()
        for feature in features:
            properties = feature.get("properties") or {}
            part_type = str(properties.get("type_overbruggingsdeel") or "").lower()
            if part_type in DECK_TYPES:
                kind = KIND_DECK
            elif part_type in PIER_TYPES:
                kind = KIND_PIER
            else:
                continue
            try:
                level = int(properties.get("relatieve_hoogteligging") or 0)
            except (TypeError, ValueError):
                level = 0
            movable = str(properties.get("overbrugging_is_beweegbaar") or "").lower() == "true"

            for group in polygon_rings(feature.get("geometry")):
                clipped = [clip_ring_to_bbox(ring, bbox) for ring in group]
                clipped = [r for r in clipped if len(r) >= 3]
                if clipped:
                    result.parts.append(
                        StructurePart(clipped, kind, level, movable)
                    )

    tunnel_polygons = []
    if want_tunnels:
        import shapely

        features, stats = fetch_current(
            "tunneldeel", bbox, class_field=None, **fetch_kwargs
        )
        result.stats_by_collection["tunneldeel"] = stats.summary()
        for feature in features:
            properties = feature.get("properties") or {}
            try:
                level = int(properties.get("relatieve_hoogteligging") or 0)
            except (TypeError, ValueError):
                level = -1
            for group in polygon_rings(feature.get("geometry")):
                clipped = [clip_ring_to_bbox(ring, bbox) for ring in group]
                clipped = [r for r in clipped if len(r) >= 3]
                if clipped:
                    result.parts.append(
                        StructurePart(clipped, KIND_TUNNEL_ROAD, level)
                    )
                # Whole, not clipped: where the area cuts a tunnel is not one
                # of its mouths.
                outer = np.asarray(group[0], dtype=np.float64)[:, :2]
                if len(outer) >= 4:
                    polygon = shapely.make_valid(shapely.Polygon(
                        outer, [np.asarray(r, dtype=np.float64)[:, :2] for r in group[1:] if len(r) >= 4]
                    ))
                    if not polygon.is_empty:
                        tunnel_polygons.append(polygon)

    if (not result.parts and not waterworks) or terrain is None:
        if not result.parts:
            LOG.info("no bridges or tunnels in this area")
        return result

    if tunnel_polygons:
        result.tunnels = _plan_tunnels(
            bbox, tunnel_polygons, terrain, structures_cfg, fetch_kwargs, water=water,
        )

    railing_cover = None
    if water and bool(structures_cfg.get("railings", True)):
        railing_cover = _railing_cover(bbox, water, fetch_kwargs)
    _build_geometry(
        result, terrain, dsm_sampler, structures_cfg, ground_at=ground_at,
        railing_cover=railing_cover, water=water, waterworks=waterworks,
    )
    return result


def _wet(water):
    """The water as one shapely geometry, and `f(point)` for its level there."""
    import shapely

    polygons, levels = [], []
    for body in water or ():
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in body.rings]
        rings = [r for r in rings if len(r) >= 4]
        if not rings:
            continue
        valid = shapely.make_valid(shapely.Polygon(rings[0], rings[1:]))
        for part in shapely.get_parts(valid):
            if part.geom_type == "Polygon":
                polygons.append(part)
                levels.append(float(body.level_nap))
    if not polygons:
        return None, lambda geometry: None
    tree = shapely.STRtree(np.array(polygons, dtype=object))

    def level_under(geometry):
        hits = tree.query(geometry, predicate="intersects")
        if not len(hits):
            return None
        return float(np.median([levels[i] for i in hits]))

    return shapely.union_all(np.array(polygons, dtype=object)), level_under


def _survey_of(terrain):
    """The raw terrain model under a TerrainResult, NaN where it saw nothing."""
    path = getattr(terrain, "geotiff_path", None)
    if path is None or not Path(path).is_file():
        return None
    from .elevation import raster_sampler

    return raster_sampler(Path(path))


def _plan_tunnels(bbox: BBox, polygons, terrain, structures_cfg: dict, fetch_kwargs: dict,
                  water=None):
    """Each tunnel's mouths and floor (`tunnels.plan_tunnels`).

    What runs through a tunnel decides how high and how steep it is, and
    what lies in front of a mouth is a road, a path or a track: both come
    from the BGT parts the surface stage has already read, from the cache.
    """
    import shapely

    from .tunnels import WATER_DEPTH_M, plan_tunnels

    below, at_ground, tracks = [], [], []
    for collection in ("wegdeel", "ondersteunendwegdeel"):
        try:
            features, _ = fetch_current(collection, bbox, class_field="functie", **fetch_kwargs)
        except Exception as exc:  # noqa: BLE001 - a tunnel without them is a road tunnel
            LOG.warning("could not read BGT %s for the tunnels: %s", collection, exc)
            continue
        for feature in features:
            properties = feature.get("properties") or {}
            try:
                level = int(properties.get("relatieve_hoogteligging") or 0)
            except (TypeError, ValueError):
                level = 0
            try:
                geometry = shapely.make_valid(shapely.geometry.shape(feature.get("geometry")))
            except Exception:  # noqa: BLE001 - one bad outline is not fatal
                continue
            if geometry.is_empty:
                continue
            if level < 0:
                below.append((geometry, str(properties.get("functie") or "")))
            elif level == 0 and collection == "wegdeel":
                at_ground.append(geometry)
    try:
        features, _ = fetch_current("spoor", bbox, class_field="functie", **fetch_kwargs)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not read BGT spoor for the tunnels: %s", exc)
        features = []
    for feature in features:
        properties = feature.get("properties") or {}
        try:
            level = int(properties.get("relatieve_hoogteligging") or 0)
        except (TypeError, ValueError):
            level = 0
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        line = shapely.LineString(np.asarray(geometry.get("coordinates"), dtype=np.float64)[:, :2])
        function = str(properties.get("functie") or "")
        if level < 0:
            tracks.append((line, function))
        elif level == 0:
            # A track bed in front of a mouth is as good as a road.
            at_ground.append(line.buffer(1.5))

    return plan_tunnels(
        polygons,
        ground=terrain.sample,
        survey=_survey_of(terrain),
        area=shapely.box(*bbox.as_list()),
        water=water or (),
        roads=below,
        tracks=tracks,
        surface_roads=at_ground,
        water_depth_m=float(structures_cfg.get("tunnel_water_depth_m", WATER_DEPTH_M)),
    )


def _railing_cover(bbox: BBox, water, fetch_kwargs: dict):
    """The wet area, and the area a BGT barrier already guards, for railings."""
    import shapely

    from .railings import guarded_area

    polygons = []
    for body in water:
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in body.rings]
        rings = [r for r in rings if len(r) >= 4]
        if rings:
            valid = shapely.make_valid(shapely.Polygon(rings[0], rings[1:]))
            polygons.extend(p for p in shapely.get_parts(valid) if p.geom_type == "Polygon")
    if not polygons:
        return None
    wet = shapely.union_all(np.array(polygons, dtype=object))
    return wet, guarded_area(bbox, fetch_current, fetch_kwargs)


def _deck_neighbours(decks, terrain, lift: float, reach_m: float = 0.3, same_m: float = 0.05):
    """`shared_edge(points, outward, z)` for `deck_slab`: True where another
    deck lies just beyond an edge with its top within `same_m` of `z`.

    Only a level neighbour hides the side: where the next part is higher or
    lower, the step between them is a face someone can see.
    """
    import shapely

    polygons, levels = [], []
    for rings, level in decks:
        outer = np.asarray(rings[0], dtype=np.float64)[:, :2]
        holes = [np.asarray(r, dtype=np.float64)[:, :2] for r in rings[1:] if len(r) >= 4]
        if len(outer) < 4:
            continue
        polygon = shapely.make_valid(shapely.Polygon(outer, holes))
        if polygon.is_empty:
            continue
        polygons.append(polygon)
        levels.append(float(level))
    if len(polygons) < 2:
        return None
    tree = shapely.STRtree(np.array(polygons, dtype=object))
    levels = np.asarray(levels)

    def shared(points, outward, z):
        points = np.asarray(points, dtype=np.float64)
        probe = points + reach_m * np.asarray(outward, dtype=np.float64)
        out = np.zeros(len(points), dtype=bool)
        hit_point, hit_deck = tree.query(shapely.points(probe), predicate="within")
        if not len(hit_point):
            return out
        ground = np.asarray(terrain.sample(probe[hit_point, 0], probe[hit_point, 1]), dtype=np.float64)
        level = levels[hit_deck]
        beyond = np.maximum(level, np.where(np.isfinite(ground), ground, level)) + lift
        level_with = np.abs(beyond - np.asarray(z, dtype=np.float64)[hit_point]) <= same_m
        out[hit_point[level_with]] = True
        return out

    return shared


def _deck_railings(rings, level: float, terrain, lift: float, cover, add) -> float:
    """Railings along a deck's sides over water; returns their length."""
    import shapely

    from .railings import open_runs, railing

    wet, guarded = cover
    outline = np.asarray(rings[0], dtype=np.float64)[:, :2]
    if len(outline) < 4:
        return 0.0
    # A hair inside the outline, so the railing stands on the deck's edge
    # rather than over the water beside it.
    deck = shapely.make_valid(shapely.Polygon(outline))
    inner = deck.buffer(-0.1, join_style="mitre")
    if inner.is_empty:
        return 0.0
    total = 0.0
    for ring in shapely.get_parts(shapely.get_exterior_ring(shapely.get_parts(inner))):
        for run in open_runs(ring, wet, guarded):
            points = shapely.get_coordinates(run)
            ground = np.asarray(terrain.sample(points[:, 0], points[:, 1]), dtype=np.float64)
            base = np.maximum(level, np.where(np.isfinite(ground), ground, level)) + lift
            add(railing(points, base), KIND_RAILING)
            total += run.length
    return total


# Brick over the top of an arch, at its thinnest.
MIN_CROWN_M = 0.35


def _slices(polygon, axis: np.ndarray, centre: np.ndarray, cuts) -> list:
    """`polygon` cut into strips at distances `cuts` along `axis`."""
    import shapely

    across = np.array([-axis[1], axis[0]])
    coords = shapely.get_coordinates(polygon)
    along = (coords - centre) @ axis
    low, high = float(along.min()), float(along.max())
    edges = [low, *[c for c in sorted(cuts) if low + 1e-6 < c < high - 1e-6], high]
    far = float(np.ptp(coords, axis=0).max()) + 10.0
    pieces = []
    for a, b in zip(edges[:-1], edges[1:]):
        strip = shapely.Polygon([
            centre + axis * a - across * far, centre + axis * b - across * far,
            centre + axis * b + across * far, centre + axis * a + across * far,
        ])
        for piece in shapely.get_parts(polygon.intersection(strip)):
            if piece.geom_type == "Polygon" and piece.area > 1e-4:
                pieces.append(piece)
    return pieces


def _match_water_bridge(polygon, bridges, reach: float = 2.0):
    """The water board's bridge standing on this deck, if any."""
    import shapely

    best = None
    for bridge in bridges or []:
        point = shapely.Point(*bridge.xy)
        distance = polygon.distance(point)
        if distance <= reach and (best is None or distance < best[0]):
            best = (distance, bridge)
    return best[1] if best is not None else None


def _bridge_record(outline, level, water_nap, register, made, movable) -> dict:
    """One bridge over water for waterworks.json: its deck, the water, and
    the room under it, from the register where it has the bridge."""
    inside = outline.representative_point()
    bottom = min(float(t[:, :, 2].min()) for t, kind in made if kind in (KIND_ARCH, KIND_DECK))
    under = [t for t, kind in made if kind == KIND_ARCH]
    crown = max(float(t[:, :, 2].max()) for t in under) if under else bottom
    record = {
        "rd": [round(float(inside.x), 3), round(float(inside.y), 3)],
        "deck_nap": round(float(level), 2),
        "water_nap": None if water_nap is None else round(float(water_nap), 2),
        "movable": bool(movable),
        "arched": any(kind == KIND_ARCH for _, kind in made),
        "crown_nap": round(crown, 2),
        "clearance_m": None if water_nap is None else round(crown - float(water_nap), 2),
        "source": "water board" if register is not None else "estimated",
    }
    if register is not None:
        record.update({
            "code": register.code, "material": register.material, "type": register.kind,
            "manager": register.manager,
        })
    return record


def _build_geometry(
    result: StructureSet, terrain, dsm_sampler, structures_cfg: dict, *, ground_at=None,
    railing_cover=None, water=None, waterworks=None,
) -> None:
    """Give every part a height and turn it into triangles."""
    import shapely
    tris: list[np.ndarray] = []
    kinds: list[np.ndarray] = []

    def add(triangles: np.ndarray, kind: int) -> None:
        if len(triangles):
            tris.append(triangles)
            kinds.append(np.full(len(triangles), kind, dtype=np.int32))

    lift = float(structures_cfg.get("deck_lift_m", 0.05))
    thickness = max(float(structures_cfg.get("deck_thickness_m", DECK_THICKNESS_M)), 0.0)
    measured = 0
    guessed = 0
    bodies = bool(structures_cfg.get("bridge_bodies", True))
    wet, level_under = _wet(water) if bodies else (None, lambda g: None)
    surface = getattr(terrain, "surface", None) or terrain.sample
    decks = [
        shapely.make_valid(shapely.Polygon(np.asarray(p.rings[0])[:, :2]))
        for p in result.parts if p.kind == KIND_DECK and p.level >= 1
    ]
    deck_area = shapely.union_all(np.array(decks, dtype=object)) if decks else None
    # A tunnel part under a deck is a passage under the bridge, a wharf going
    # through its abutment: no road tunnel eighteen metres deep.
    passages = []
    if bodies and deck_area is not None:
        for p in result.parts:
            if p.kind != KIND_TUNNEL_ROAD:
                continue
            outline = shapely.make_valid(shapely.Polygon(np.asarray(p.rings[0])[:, :2]))
            if outline.intersection(deck_area).area > 0.3 * max(outline.area, 1e-9):
                passages.append(outline)
                p.kind = -1  # built by the bridge over it
    result.counts["wharf_passages"] = len(passages)
    arched = 0
    piers = [
        shapely.make_valid(shapely.Polygon(np.asarray(p.rings[0])[:, :2]))
        for p in result.parts if p.kind == KIND_PIER
    ] if bodies else []
    arched_decks = []
    # The water boards' bridges, matched to a deck for its clearance.
    water_bridges = getattr(waterworks, "bridges", None) or []

    # ---- bridges ---------------------------------------------------------
    # Every part's height first, so each deck knows where its neighbours are.
    bridge_parts = []
    for part in result.parts:
        if part.kind not in (KIND_DECK, KIND_PIER):
            continue

        flat = _triangulate(part.rings)
        if not len(flat):
            continue
        corners = flat.reshape(-1, 2)
        under = np.asarray(terrain.sample(corners[:, 0], corners[:, 1]))
        ground = float(np.median(under))

        level = None
        if dsm_sampler is not None:
            level = deck_height(part.rings, dsm_sampler, ground=ground)

        if level is None:
            # Nothing measured it. The ordinal level is all that is left, and
            # a Dutch road bridge clears what it crosses by about five metres.
            # Measured from the highest ground the part covers, not from its
            # centre: a deck spanning a slope would otherwise be guessed from
            # the middle and buried at the upper end.
            level = float(under.max()) + float(
                structures_cfg.get("fallback_clearance_m", 5.0)
            ) * max(part.level, 1)
            if part.kind == KIND_DECK:
                guessed += 1
        elif part.kind == KIND_DECK:
            # Only decks are counted: a pier reads the deck above it, so
            # counting both made "17 decks, 22 measured" out of 17 and 5.
            measured += 1
        bridge_parts.append((part, flat, level))

    shared_edge = _deck_neighbours(
        [(part.rings, level) for part, _, level in bridge_parts if part.kind == KIND_DECK],
        terrain, lift,
    )
    # Decks first: a pier under an arch is built in the arch's brick.
    for part, flat, level in sorted(bridge_parts, key=lambda t: t[0].kind != KIND_DECK):
        if part.kind == KIND_DECK:
            # One height for the whole part, except where the ground rises
            # through it. A BGT bridge part is often a whole approach, and in
            # dike country the ground climbs to meet the deck at the abutment:
            # a single flat level then leaves the last stretch buried, which is
            # what put a fifth of the deck geometry under the terrain over a
            # 2 km area near Dordrecht. Held up per corner rather than moved as
            # a whole, so the span itself stays flat and only the ends lift --
            # which is what an abutment looks like anyway.
            # A canal bridge is a body, arched over its water; a deck over dry
            # ground stays a slab, so what passes under a viaduct stays open.
            made = []
            outline = None
            if bodies and part.level >= 1 and wet is not None:
                outline = shapely.make_valid(shapely.Polygon(np.asarray(part.rings[0])[:, :2]))
            if outline is not None and outline.intersects(wet):
                water_nap = level_under(outline)
                register = _match_water_bridge(outline, water_bridges)
                soffit = getattr(register, "soffit_nap", None)
                may_arch = not part.movable and (register is None or register.masonry)
                made = bridge_body(
                    part.rings, level, ground_at=surface, wet=wet,
                    water_level=water_nap, passages=passages, piers=piers, lift=lift,
                    crown_nap=soffit, arch=may_arch,
                )
                if made:
                    result.bridges.append(_bridge_record(
                        outline, level, water_nap, register, made, part.movable,
                    ))
            if made:
                for triangles, kind in made:
                    add(triangles, kind)
                if any(kind == KIND_ARCH for _, kind in made):
                    arched += 1
                    arched_decks.append(outline)
            else:
                def top_at(x, y, level=level):
                    ground = np.asarray(terrain.sample(x, y), dtype=np.float64)
                    return np.maximum(level, np.where(np.isfinite(ground), ground, level)) + lift

                add(deck_slab(part.rings, flat, top_at, thickness, shared_edge=shared_edge), KIND_DECK)
            result.deck_levels.append((flat.reshape(-1, 2), level))
            if railing_cover is not None:
                metres = _deck_railings(part.rings, level, terrain, lift, railing_cover, add)
                result.counts["railing_m"] = round(result.counts.get("railing_m", 0.0) + metres, 1)
        else:
            # A pier runs from the deck's underside down to whatever it stands
            # on, a hand's breadth into the slab so no seam shows; under a
            # brick arch it is brick, the arches landing on it.
            top = level + lift - thickness + 0.05
            kind = KIND_PIER
            if arched_decks:
                outline = shapely.make_valid(shapely.Polygon(np.asarray(part.rings[0])[:, :2]))
                if any(outline.intersects(deck) for deck in arched_decks):
                    kind = KIND_BRIDGE_SIDE
                    result.counts["piers_under_arches"] = result.counts.get("piers_under_arches", 0) + 1
            for ring in part.rings:
                base = np.asarray(terrain.sample(ring[:, 0], ring[:, 1]), dtype=np.float64)
                if ground_at is not None:
                    built = np.asarray(ground_at(ring[:, 0], ring[:, 1]), dtype=np.float64)
                    base = np.where(np.isfinite(built), np.fmin(base, built), base)
                add(extrude_ring(ring, top, np.minimum(base, top - 0.5)), kind)
    if thickness > 0:
        result.counts["deck_thickness_m"] = round(thickness, 2)

    result.counts["decks_measured_from_dsm"] = measured
    result.counts["decks_without_a_reading"] = guessed
    result.counts["bridge_bodies"] = arched

    # ---- locks and weirs --------------------------------------------------
    if waterworks is not None and water and bool(structures_cfg.get("locks_and_weirs", True)):
        from .waterworks import lock_geometry, water_geometry, water_level_at, weir_geometry

        wet_all = wet if wet is not None else water_geometry(water)

        def levels(xy):
            return water_level_at(water, xy)

        def ground(x, y):
            built = ground_at(x, y) if ground_at is not None else None
            surveyed = np.asarray(terrain.sample(x, y), dtype=np.float64)
            return surveyed if built is None else np.fmax(surveyed, np.asarray(built, dtype=np.float64))

        if wet_all is not None:
            kinds_of = {"gate": KIND_LOCK_GATE, "beam": KIND_LOCK_BEAM, "weir": KIND_WEIR}
            for lock in waterworks.locks:
                try:
                    pieces, record = lock_geometry(lock, wet_all, levels, ground)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("could not build lock %s: %s", lock.name, exc)
                    continue
                for triangles, name in pieces:
                    add(triangles, kinds_of[name])
                if record is not None:
                    result.locks.append(record)
            for weir in waterworks.weirs:
                try:
                    pieces, record = weir_geometry(weir, wet_all, levels)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("could not build a weir: %s", exc)
                    continue
                for triangles, name in pieces:
                    add(triangles, kinds_of[name])
                if record is not None:
                    result.weirs.append(record)
        result.counts["locks"] = len(result.locks)
        result.counts["weirs"] = len(result.weirs)

    # ---- tunnels ---------------------------------------------------------
    tunnel_parts = [p for p in result.parts if p.kind == KIND_TUNNEL_ROAD]
    if len(result.tunnels):
        import shapely

        from .tunnels import top_reader, tunnel_geometry

        area = shapely.box(*result.bbox) if result.bbox is not None else None
        top = top_reader(terrain.sample, _survey_of(terrain))
        tube = bool(structures_cfg.get("tunnel_walls", True))
        for tunnel in result.tunnels.tunnels:
            built = tunnel_geometry(tunnel, area=area, top=top)
            add(built.floor, KIND_TUNNEL_ROAD)
            if tube:
                add(built.tube, KIND_TUNNEL_WALL)
        result.counts["tunnel_parts"] = len(tunnel_parts)
        result.counts["tunnels"] = len(result.tunnels)
        result.counts["tunnel_portals"] = sum(len(t.portals) for t in result.tunnels.tunnels)

    if tris:
        result.tris = np.concatenate(tris)
        result.tri_kind = np.concatenate(kinds)
    if result.bridges:
        result.counts["bridges_over_water"] = len(result.bridges)
        result.counts["arched_bridges"] = sum(1 for b in result.bridges if b.get("arched"))
        result.counts["bridges_from_water_board"] = sum(
            1 for b in result.bridges if b.get("source") == "water board"
        )

    LOG.info(
        "structures: %d decks (%d measured from the surface model, %d without "
        "a reading), %d piers, %d tunnel parts, %d triangles",
        result.count_of(KIND_DECK),
        measured,
        guessed,
        result.count_of(KIND_PIER),
        len(tunnel_parts),
        len(result.tris),
    )


def write_bridge_list(
    structures: StructureSet, geo, out_dir: Path, filename: str = "waterworks.json"
) -> Path:
    """Every bridge over water, with the room a boat has under it, and the
    locks and weirs.

    `clearance_m` is from the water's surface to the underside at its highest:
    the water board's figure where it has one, the deck less its thickness
    where it does not, which `source` says.
    """
    import json

    def local(records):
        out = []
        for record in records:
            x, y = geo.rd_to_local(record["rd"][0], record["rd"][1])
            out.append({**record, "local": {"x": round(float(x), 3), "z": round(float(y), 3)}})
        return out

    payload = {
        "crs": "EPSG:28992",
        "note": (
            "local.x / local.z are metres in the model frame. Heights are NAP; "
            "subtract metadata.ground_z_offset_nap for local Y. clearance_m is "
            "crown_nap less water_nap."
        ),
        "source": "BGT overbruggingsdeel, water board register (IMWA) via PDOK",
        "bridges": local(structures.bridges),
        "locks": local(structures.locks),
        "weirs": local(structures.weirs),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    LOG.info(
        "wrote %s (%d bridges, %d locks, %d weirs)",
        path, len(structures.bridges), len(structures.locks), len(structures.weirs),
    )
    return path


def save_structures(structures: StructureSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, tris=structures.tris, tri_kind=structures.tri_kind
    )
    LOG.info("wrote %s (%d triangles)", path, len(structures.tris))
    return path


__all__ = [
    "KIND_ARCH",
    "KIND_BRIDGE_SIDE",
    "KIND_DECK",
    "KIND_LOCK_BEAM",
    "KIND_LOCK_GATE",
    "KIND_WEIR",
    "bridge_body",
    "write_bridge_list",
    "deck_corner_heights",
    "deck_outlines",
    "KIND_NAMES",
    "KIND_PIER",
    "KIND_RAILING",
    "KIND_TUNNEL_ROAD",
    "KIND_TUNNEL_WALL",
    "MAX_DECK_NAP",
    "MIN_DECK_NAP",
    "StructurePart",
    "StructureSet",
    "build_structures",
    "deck_height",
    "deck_surface",
    "extrude_ring",
    "save_structures",
]
