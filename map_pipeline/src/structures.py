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

KIND_NAMES = {
    KIND_DECK: "deck",
    KIND_PIER: "pier",
    KIND_TUNNEL_ROAD: "tunnel_road",
    KIND_TUNNEL_WALL: "tunnel_wall",
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

    def __len__(self) -> int:
        return len(self.parts)

    def count_of(self, kind: int) -> int:
        return sum(1 for part in self.parts if part.kind == kind)

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

    if n > 1 and np.allclose(points[0], points[-1]):
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


def build_structures(
    bbox: BBox,
    work_dir: Path,
    *,
    structures_cfg: dict,
    terrain=None,
    dsm_sampler=None,
) -> StructureSet:
    """Fetch bridge and tunnel parts and build them into geometry."""
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

    if not result.parts or terrain is None:
        if not result.parts:
            LOG.info("no bridges or tunnels in this area")
        return result

    _build_geometry(result, terrain, dsm_sampler, structures_cfg)
    return result


def _build_geometry(
    result: StructureSet, terrain, dsm_sampler, structures_cfg: dict
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

    # ---- bridges ---------------------------------------------------------
    for part in result.parts:
        if part.kind not in (KIND_DECK, KIND_PIER):
            continue

        flat = _triangulate(part.rings)
        if not len(flat):
            continue
        centre = flat.reshape(-1, 2).mean(axis=0)
        ground = float(terrain.sample(np.array([centre[0]]), np.array([centre[1]]))[0])

        level = None
        if dsm_sampler is not None:
            level = deck_height(part.rings, dsm_sampler, ground=ground)

        if level is None:
            # Nothing measured it. The ordinal level is all that is left, and
            # a Dutch road bridge clears what it crosses by about five metres.
            level = ground + float(structures_cfg.get("fallback_clearance_m", 5.0)) * max(
                part.level, 1
            )
            if part.kind == KIND_DECK:
                guessed += 1
        elif part.kind == KIND_DECK:
            # Only decks are counted: a pier reads the deck above it, so
            # counting both made "17 decks, 22 measured" out of 17 and 5.
            measured += 1

        if part.kind == KIND_DECK:
            deck = np.dstack([flat, np.full((len(flat), 3, 1), level + lift)])
            add(deck, KIND_DECK)
            result.deck_levels.append((flat.reshape(-1, 2), level))
        else:
            # A pier runs from under the deck down to whatever it stands on.
            for ring in part.rings:
                base = terrain.sample(ring[:, 0], ring[:, 1])
                add(extrude_ring(ring, level, np.minimum(base, level - 0.5)), KIND_PIER)

    result.counts["decks_measured_from_dsm"] = measured
    result.counts["decks_without_a_reading"] = guessed

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


def save_structures(structures: StructureSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, tris=structures.tris, tri_kind=structures.tri_kind
    )
    LOG.info("wrote %s (%d triangles)", path, len(structures.tris))
    return path


__all__ = [
    "KIND_DECK",
    "KIND_NAMES",
    "KIND_PIER",
    "KIND_TUNNEL_ROAD",
    "KIND_TUNNEL_WALL",
    "MAX_DECK_NAP",
    "MIN_DECK_NAP",
    "StructurePart",
    "StructureSet",
    "build_structures",
    "deck_height",
    "extrude_ring",
    "save_structures",
    "tunnel_depth_profile",
]
