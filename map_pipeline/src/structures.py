"""Bridges and tunnels: the parts of the ground that are not the ground.

Everything else in this pipeline sits on the terrain. These two do not, and
until now both were simply dropped — tunnels skipped outright, bridges draped
onto the surface, which sank the Erasmusbrug into the Maas.

The two need opposite treatment, because the data for them is not symmetrical.

**A bridge can be measured.** The BGT publishes ``overbruggingsdeel`` split into
``dek`` and ``pijler`` — the deck and the piers holding it up, as separate
polygons — and a deck is a hard surface, so the AHN surface model sees it where
the terrain model, which is bare ground by definition, does not. Over a 1 km
square of the Maas the DSM returns a reading on all fourteen decks. The height
is therefore taken, not invented.

**A tunnel cannot.** Nothing looks down and sees the Maastunnel. The BGT gives
its footprint and the carriageway inside it, and an ordinal
``relatieve_hoogteligging`` of -1, but no depth in metres exists in any open
dataset. So the profile here is *constructed*: portals at ground level, ramping
down to a configured depth along the tunnel's own long axis. It is the one
piece of geometry in this pipeline that is drawn rather than measured, and it
is marked as such in the metadata so nobody mistakes it for a survey.
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
# The body under a deck: brick round an arch where the water board says the
# bridge is masonry, a concrete slab everywhere else.
KIND_MASONRY = 5
KIND_SLAB = 6
KIND_LOCK_GATE = 7
KIND_LOCK_BEAM = 8
KIND_WEIR = 9

KIND_NAMES = {
    KIND_DECK: "deck",
    KIND_PIER: "pier",
    KIND_TUNNEL_ROAD: "tunnel_road",
    KIND_TUNNEL_WALL: "tunnel_wall",
    KIND_RAILING: "railing",
    KIND_MASONRY: "masonry",
    KIND_SLAB: "slab",
    KIND_LOCK_GATE: "lock_gate",
    KIND_LOCK_BEAM: "lock_beam",
    KIND_WEIR: "weir",
}

# BGT type_overbruggingsdeel values worth building. "sloof" is the pile cap,
# which sits at or below the waterline and is not worth its own geometry.
DECK_TYPES = {"dek"}
PIER_TYPES = {"pijler"}

# A deck reading has to be a deck. Below this above the water it is the DSM
# seeing the river rather than the bridge; above it, a mast or a crane.
MIN_DECK_NAP = -3.0
MAX_DECK_NAP = 80.0


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


def tunnel_depth_profile(
    points: np.ndarray, *, depth_m: float, ramp_m: float
) -> np.ndarray:
    """How far below ground each point of a tunnel sits.

    Nothing measures this. The profile is drawn: level with the ground at the
    portals, ramping down over ``ramp_m`` to ``depth_m`` in between, along the
    tunnel's own long axis. Position along that axis is the only thing here
    that comes from the data.
    """
    points = np.asarray(points, dtype=np.float64)[:, :2]
    if len(points) < 2:
        return np.zeros(len(points))

    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    along = centred @ vt[0]

    start, end = float(along.min()), float(along.max())
    ramp = max(min(ramp_m, 0.5 * (end - start)), 1e-6)

    # Distance from whichever portal is nearer, saturating at the ramp length.
    from_start = along - start
    from_end = end - along
    fraction = np.clip(np.minimum(from_start, from_end) / ramp, 0.0, 1.0)
    # Smoothstep, so the road eases into the descent rather than kinking.
    return depth_m * (fraction * fraction * (3.0 - 2.0 * fraction))


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


# A point this close to a deck's outline is on the deck, for what stands on it.
DECK_EDGE_M = 0.6


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

    if want_tunnels:
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

    if (not result.parts and not waterworks) or terrain is None:
        if not result.parts:
            LOG.info("no bridges or tunnels in this area")
        return result

    railing_cover = None
    if water and bool(structures_cfg.get("railings", True)):
        railing_cover = _railing_cover(bbox, water, fetch_kwargs)
    _build_geometry(
        result, terrain, dsm_sampler, structures_cfg, ground_at=ground_at,
        railing_cover=railing_cover, water=water, waterworks=waterworks,
    )
    return result


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


# Under a deck that nothing measured, this much bridge.
DEFAULT_DECK_THICKNESS_M = 0.9
# Brick over the top of an arch, at its thinnest.
MIN_CROWN_M = 0.35
# An arch springs from just above the water it spans.
SPRING_ABOVE_WATER_M = 0.3
# Slices across the span, so the vault is round rather than a roof.
ARCH_SLICE_M = 0.5


def segmental_arch(s, half_span: float, crown: float, spring: float) -> np.ndarray:
    """The underside of a segmental arch at distance `s` from its middle.

    A circle through the two springing points and the crown: the shallow
    brick arch of a canal bridge rather than a Roman semicircle. Level with
    the springing beyond the span.
    """
    s = np.asarray(s, dtype=np.float64)
    rise = max(crown - spring, 0.05)
    z = np.full(s.shape, float(spring))
    inside = np.abs(s) < half_span
    if rise > half_span:
        # Taller than half its width: no circle through those three points
        # stays above the springing, so the opening is a half ellipse.
        z[inside] = spring + rise * np.sqrt(np.maximum(1.0 - (s[inside] / half_span) ** 2, 0.0))
        return z
    radius = (half_span * half_span + rise * rise) / (2.0 * rise)
    z[inside] = spring + np.sqrt(np.maximum(radius * radius - s[inside] ** 2, 0.0)) - (radius - rise)
    return z


def openings(half_span: float, piers: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The spans between the banks and the piers standing in the water.

    `piers` are intervals along the span axis; what is left of
    [-half_span, half_span] once they are taken out is where the arches go.
    """
    edges = [(-half_span, -half_span)]
    for low, high in sorted(piers):
        low, high = max(low, -half_span), min(high, half_span)
        if high > low:
            edges.append((low, high))
    edges.append((half_span, half_span))
    out = []
    for (_, end), (start, _) in zip(edges[:-1], edges[1:]):
        if start - end > 0.5:
            out.append((end, start))
    return out


def arches_underside(s, spans, crown: float, spring: float) -> np.ndarray:
    """The underside across several openings: an arch over each, level with
    the springing over the piers between them."""
    s = np.asarray(s, dtype=np.float64)
    z = np.full(s.shape, float(spring))
    for low, high in spans:
        middle, half = (low + high) / 2.0, (high - low) / 2.0
        inside = (s > low) & (s < high)
        z[inside] = segmental_arch(s[inside] - middle, half, crown, spring)
    return z


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


def bridge_body(polygon, top_at, underside_at, *, axis=None, centre=None, cuts=()) -> np.ndarray:
    """The solid under a deck: its underside, and its sides down to it.

    `top_at(xy)` and `underside_at(xy)` give heights for an (n, 2) array of
    points. With `cuts`, the underside is sliced across `axis` there first, so
    a curve between them comes out curved.
    """
    import shapely

    polygon = shapely.orient_polygons(polygon) if hasattr(shapely, "orient_polygons") else (
        shapely.geometry.polygon.orient(polygon, 1.0)
    )
    triangles = []
    pieces = (
        _slices(polygon, np.asarray(axis, dtype=np.float64), np.asarray(centre, dtype=np.float64), cuts)
        if len(cuts) else [polygon]
    )
    for piece in pieces:
        rings = [np.asarray(piece.exterior.coords)[:, :2]] + [
            np.asarray(r.coords)[:, :2] for r in piece.interiors
        ]
        flat = _triangulate(rings)
        if not len(flat):
            continue
        # Facing down: clockwise seen from above.
        a, b, c = flat[:, 0], flat[:, 1], flat[:, 2]
        upward = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) > 0
        flat[upward] = flat[upward][:, ::-1]
        z = np.asarray(underside_at(flat.reshape(-1, 2)), dtype=np.float64).reshape(-1, 3, 1)
        triangles.append(np.dstack([flat, z]))

    for ring in [polygon.exterior, *polygon.interiors]:
        points = shapely.get_coordinates(shapely.segmentize(ring, ARCH_SLICE_M))[:, :2]
        if len(points) < 4:
            continue
        top = np.asarray(top_at(points), dtype=np.float64)
        bottom = np.minimum(np.asarray(underside_at(points), dtype=np.float64), top - 0.05)
        triangles.append(extrude_ring(points, top, bottom))
    triangles = [t for t in triangles if len(t)]
    return np.concatenate(triangles) if triangles else np.zeros((0, 3, 3))


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


def _deck_body(
    part, level: float, terrain, lift: float, structures_cfg: dict, wet, water, bridges,
    piers=(),
):
    """The body under one deck, its kind, and a record when it is over water.

    `piers` are the BGT's pier outlines: one standing in the water under an
    arched bridge splits its opening in two.
    """
    import shapely

    from .waterworks import narrowest_crossing, water_level_at

    outline = np.asarray(part.rings[0], dtype=np.float64)[:, :2]
    holes = [np.asarray(r, dtype=np.float64)[:, :2] for r in part.rings[1:] if len(r) >= 4]
    polygon = shapely.make_valid(shapely.Polygon(outline, holes))
    polygons = [p for p in shapely.get_parts(polygon) if p.geom_type == "Polygon" and p.area > 0.5]
    if not polygons:
        return np.zeros((0, 3, 3)), KIND_SLAB, None
    polygon = max(polygons, key=lambda p: p.area)

    def top_at(points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        ground = np.asarray(terrain.sample(points[:, 0], points[:, 1]), dtype=np.float64)
        return np.maximum(level, np.where(np.isfinite(ground), ground, level)) + lift

    thickness = float(structures_cfg.get("deck_thickness_m", DEFAULT_DECK_THICKNESS_M))
    over = polygon.intersection(wet) if wet is not None else None
    over_water = over is not None and not over.is_empty and over.area >= 2.0
    if not over_water:
        body = bridge_body(polygon, top_at, lambda p: top_at(p) - thickness)
        return body, KIND_SLAB, None

    wet_parts = [p for p in shapely.get_parts(over) if p.geom_type == "Polygon"]
    inside = max(wet_parts, key=lambda p: p.area).representative_point()
    crossing = narrowest_crossing(wet, (inside.x, inside.y))
    bridge = _match_water_bridge(polygon, bridges)
    water_nap = water_level_at(water, (inside.x, inside.y))

    record = {
        "rd": [round(float(inside.x), 3), round(float(inside.y), 3)],
        "deck_nap": round(float(level), 2),
        "water_nap": None if water_nap is None else round(float(water_nap), 2),
        "movable": bool(part.movable),
    }
    if crossing is None:
        body = bridge_body(polygon, top_at, lambda p: top_at(p) - thickness)
        record.update({"source": "estimated", "arched": False})
        return body, KIND_SLAB, record

    width, axis, middle = crossing
    span = width
    crown = level - thickness
    source = "estimated"
    if bridge is not None:
        if bridge.span_m and 0.5 * width <= bridge.span_m <= 3.0 * width + 4.0:
            span = float(bridge.span_m)
        if bridge.soffit_nap is not None and (water_nap is None or bridge.soffit_nap > water_nap + 0.5):
            crown = min(float(bridge.soffit_nap), level - MIN_CROWN_M)
        source = "water board"
        record.update({
            "code": bridge.code, "material": bridge.material, "type": bridge.kind,
            "manager": bridge.manager,
        })
    arched = bridge is not None and bridge.masonry and not part.movable
    spring = (water_nap if water_nap is not None else crown - 0.3 * span) + SPRING_ABOVE_WATER_M
    spring = min(spring, crown - 0.3)
    half = span / 2.0

    if arched:
        standing = []
        for pier in piers:
            if pier.intersects(polygon) and pier.intersection(wet).area > 0.2:
                along = (shapely.get_coordinates(pier) - middle) @ axis
                standing.append((float(along.min()), float(along.max())))
        spans = openings(half, standing)
        if not spans:
            spans = [(-half, half)]

        def underside_at(points):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            s = (points - middle) @ axis
            z = arches_underside(s, spans, crown, spring)
            return np.minimum(z, top_at(points) - MIN_CROWN_M)

        cuts = sorted({
            float(c) for low, high in spans
            for c in np.linspace(low, high, max(6, int(np.ceil((high - low) / ARCH_SLICE_M))) + 1)
        })
        record["openings"] = len(spans)
        body = bridge_body(polygon, top_at, underside_at, axis=axis, centre=middle, cuts=cuts)
        kind = KIND_MASONRY
    else:
        flat_bottom = crown
        body = bridge_body(
            polygon, top_at, lambda p: np.minimum(top_at(p) - MIN_CROWN_M, flat_bottom)
        )
        kind = KIND_SLAB

    record.update({
        "source": source,
        "arched": bool(arched),
        "span_m": round(float(span), 2),
        "water_width_m": round(float(width), 2),
        "crown_nap": round(float(crown), 2),
        "clearance_m": None if water_nap is None else round(float(crown - water_nap), 2),
        "span_axis_deg": round(float(np.degrees(np.arctan2(axis[0], axis[1])) % 180.0), 1),
    })
    return body, kind, record


def _build_geometry(
    result: StructureSet, terrain, dsm_sampler, structures_cfg: dict, *, ground_at=None,
    railing_cover=None, water=None, waterworks=None,
) -> None:
    """Give every part a height and turn it into triangles."""
    tris: list[np.ndarray] = []
    kinds: list[np.ndarray] = []

    def add(triangles: np.ndarray, kind: int) -> None:
        if len(triangles):
            tris.append(triangles)
            kinds.append(np.full(len(triangles), kind, dtype=np.int32))

    lift = float(structures_cfg.get("deck_lift_m", 0.05))
    measured = 0
    guessed = 0
    bodies = bool(structures_cfg.get("deck_bodies", True))
    wet = None
    if bodies and water:
        from .waterworks import water_geometry

        wet = water_geometry(water)
    water_bridges = getattr(waterworks, "bridges", None) or []
    pier_shapes = []
    if bodies:
        import shapely

        for part in result.parts:
            if part.kind == KIND_PIER and len(part.rings[0]) >= 4:
                pier = shapely.make_valid(shapely.Polygon(np.asarray(part.rings[0])[:, :2]))
                if not pier.is_empty:
                    pier_shapes.append(pier)

    # ---- bridges ---------------------------------------------------------
    # Decks first, so a pier knows whether it holds up a brick arch.
    arched_decks: list = []
    ordered = sorted(
        (p for p in result.parts if p.kind in (KIND_DECK, KIND_PIER)),
        key=lambda p: 0 if p.kind == KIND_DECK else 1,
    )
    for part in ordered:

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

        if part.kind == KIND_DECK:
            # One height for the whole part, except where the ground rises
            # through it. A BGT bridge part is often a whole approach, and in
            # dike country the ground climbs to meet the deck at the abutment:
            # a single flat level then leaves the last stretch buried, which is
            # what put a fifth of the deck geometry under the terrain over a
            # 2 km area near Dordrecht. Held up per corner rather than moved as
            # a whole, so the span itself stays flat and only the ends lift --
            # which is what an abutment looks like anyway.
            deck = np.dstack([flat, deck_corner_heights(flat, level, terrain, lift)])
            add(deck, KIND_DECK)
            result.deck_levels.append((flat.reshape(-1, 2), level))
            if bodies:
                try:
                    body, body_kind, record = _deck_body(
                        part, level, terrain, lift, structures_cfg, wet, water, water_bridges,
                        pier_shapes,
                    )
                except Exception as exc:  # noqa: BLE001 - a deck without a body is still a deck
                    LOG.warning("could not build the body under a deck: %s", exc)
                    body, body_kind, record = np.zeros((0, 3, 3)), KIND_SLAB, None
                add(body, body_kind)
                if record is not None:
                    result.bridges.append(record)
                if body_kind == KIND_MASONRY:
                    import shapely

                    arched_decks.append(shapely.make_valid(shapely.Polygon(
                        np.asarray(part.rings[0], dtype=np.float64)[:, :2]
                    )))
            if railing_cover is not None:
                metres = _deck_railings(part.rings, level, terrain, lift, railing_cover, add)
                result.counts["railing_m"] = round(result.counts.get("railing_m", 0.0) + metres, 1)
        else:
            # A pier runs from under the deck down to whatever it stands on,
            # and is brick where the arches on it are.
            pier_kind = KIND_PIER
            if arched_decks:
                import shapely

                outline = shapely.make_valid(shapely.Polygon(np.asarray(part.rings[0])[:, :2]))
                if any(outline.intersects(deck) for deck in arched_decks):
                    pier_kind = KIND_MASONRY
            for ring in part.rings:
                base = np.asarray(terrain.sample(ring[:, 0], ring[:, 1]), dtype=np.float64)
                if ground_at is not None:
                    built = np.asarray(ground_at(ring[:, 0], ring[:, 1]), dtype=np.float64)
                    base = np.where(np.isfinite(built), np.fmin(base, built), base)
                add(extrude_ring(ring, level, np.minimum(base, level - 0.5)), pier_kind)

    result.counts["decks_measured_from_dsm"] = measured
    result.counts["decks_without_a_reading"] = guessed

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
    if tunnel_parts:
        depth_m = float(structures_cfg.get("tunnel_depth_m", 18.0))
        ramp_m = float(structures_cfg.get("tunnel_ramp_m", 350.0))
        wall = bool(structures_cfg.get("tunnel_walls", True))

        # One profile over the whole tunnel, so its pieces agree on where the
        # portals are and the road does not step between them.
        everything = np.vstack(
            [r for part in tunnel_parts for r in part.rings]
        )[:, :2]

        for part in tunnel_parts:
            flat = _triangulate(part.rings)
            if not len(flat):
                continue
            points = flat.reshape(-1, 2)
            ground = np.asarray(terrain.sample(points[:, 0], points[:, 1]))
            drop = _profile_against(everything, points, depth_m=depth_m, ramp_m=ramp_m)
            road_z = (ground - drop).reshape(-1, 3, 1)
            add(np.dstack([flat, road_z]), KIND_TUNNEL_ROAD)

            if wall:
                for ring in part.rings:
                    top = terrain.sample(ring[:, 0], ring[:, 1])
                    bottom = top - _profile_against(
                        everything, ring[:, :2], depth_m=depth_m, ramp_m=ramp_m
                    )
                    add(extrude_ring(ring, top, bottom), KIND_TUNNEL_WALL)

        result.counts["tunnel_parts"] = len(tunnel_parts)
        result.counts["tunnel_depth_m"] = int(depth_m)

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


def _profile_against(
    all_points: np.ndarray, points: np.ndarray, *, depth_m: float, ramp_m: float
) -> np.ndarray:
    """Depth for `points`, using the axis of the whole tunnel."""
    centred_all = all_points - all_points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred_all, full_matrices=False)
    along_all = centred_all @ vt[0]
    start, end = float(along_all.min()), float(along_all.max())

    along = (np.asarray(points, dtype=np.float64)[:, :2] - all_points.mean(axis=0)) @ vt[0]
    ramp = max(min(ramp_m, 0.5 * (end - start)), 1e-6)
    fraction = np.clip(
        np.minimum(along - start, end - along) / ramp, 0.0, 1.0
    )
    return depth_m * (fraction * fraction * (3.0 - 2.0 * fraction))


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
    "KIND_DECK",
    "KIND_LOCK_BEAM",
    "KIND_LOCK_GATE",
    "KIND_MASONRY",
    "KIND_SLAB",
    "KIND_WEIR",
    "bridge_body",
    "segmental_arch",
    "write_bridge_list",
    "deck_corner_heights",
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
    "tunnel_depth_profile",
]
