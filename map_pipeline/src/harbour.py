"""The harbour: quay faces, jetties, pontoons and mooring posts, from the BGT.

Built for the layered style, where the water is a flat surface of its own and
the terrain under it is cut down:

* **Quay faces.** The BGT draws quay walls (``scheiding``, type ``kademuur``)
  as lines and strips along the waterline. Along every stretch of waterline
  within `QUAY_REACH_M` of one, a concrete strip `COPING_M` wide stands just
  out into the water, its top flush with the quay and its face running down to
  `BELOW_WATER_M` under the water level. From the water that is the quay wall;
  from above, its coping. The terrain behind it drops in a hand's breadth
  (`layered_mesh.QUAY_ROW_M`), so nothing shows through.
* **Jetties and pontoons** (``kunstwerkdeel_vlak``, ``steiger`` and plus-type
  ``ponton``) are decks: at the height the surface model saw on them where it
  saw them, otherwise at a standard height above the water.
* **Mooring posts** (``waterinrichtingselement_punt``, plus-type ``meerpaal``)
  are posts from below the water to `POST_ABOVE_M` above it.
* **Fender works** (``waterinrichtingselement_lijn``, plus-type
  ``remmingswerk`` or ``geleidewerk``) are the catwalks and dolphin rows at a
  ferry berth and the fender runs along a quay. The BGT has them only as
  lines, and the water polygon runs under them. Out in the water a line is a
  catwalk `CATWALK_WIDTH_M` wide on posts, at the height the surface model saw
  on it, with a railing down both sides; tight along the waterline it is a
  fender, a thin strip from below the water to the quay top.

* **Piles, kerbs and bollards.** A jetty stands on piles, in rows across it
  every `PILE_EVERY_M` with a headstock over each row, and has a kerb round
  its deck; a pontoon lies between guide piles. Along a quay on open water
  (wider than `2 * OPEN_WATER_M`: a harbour, not a canal) a bollard stands
  every `BOLLARD_EVERY_M` and a ladder hangs down the face every
  `LADDER_EVERY_M`.
* **Crane rails, cranes and containers** (`terminal`). The BGT draws the
  rails (``spoor``, plus-function ``(haven)kraan``); the cranes on them and
  the container stacks are where, and as high as, the surface model saw them
  the day it was flown.

All of it is heights read off the water levels, which `surfaces` reads once
per connected water from the surface model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import harbour_kit as kit
from .bgt import fetch_current
from .geo import BBox

LOG = logging.getLogger(__name__)

QUAY, JETTY, PONTOON, POST, CATWALK, FENDER, RAILING = 0, 1, 2, 3, 4, 5, 6
PILE, BOLLARD, LADDER, CRANE_RAIL, CRANE, CONTAINER = 7, 8, 9, 10, 11, 12
KIND_NAMES = {
    QUAY: "quay", JETTY: "jetty", PONTOON: "pontoon", POST: "mooring_post",
    CATWALK: "catwalk", FENDER: "fender", RAILING: "railing",
    PILE: "pile", BOLLARD: "bollard", LADDER: "ladder", CRANE_RAIL: "crane_rail",
    CRANE: "crane", CONTAINER: "container",
}
# Jetty piles: a row across the deck this often, piles in a row no further
# apart than this, and how far in from the deck's edge the outer ones stand.
PILE_EVERY_M = 4.0
PILE_RADIUS_M = 0.15
PILE_INSET_M = 0.4
HEADSTOCK_M = 0.25
# A kerb round a deck at least this wide.
KERB_M = 0.15
KERB_MIN_DECK_M = 2.5
# A pontoon at least this long lies between two guide piles.
GUIDE_PILE_MIN_M = 6.0
# Water wider than twice this is open water: its quays are harbour quays,
# with bollards and ladders. A canal's are not.
OPEN_WATER_M = 20.0
BOLLARD_EVERY_M = 20.0
BOLLARD_BACK_M = 0.8
LADDER_EVERY_M = 60.0
LADDER_BELOW_WATER_M = 1.0
# A quay lower than this over the water has neither.
FITTED_QUAY_M = 1.0
# No tree is found within this of a crane.
CRANE_CLEAR_M = 8.0
# How coarsely the ground under the surface model is sampled, for heights
# above the paving.
GROUND_STEP_M = 4.0
# Fender-work lines: the plus-types read, how wide a catwalk is, how far
# apart its posts stand, and how close to the waterline a line has to run,
# on average, to be a fender on the quay rather than a catwalk out in the water.
FENDER_TYPES = ("remmingswerk", "geleidewerk")
CATWALK_WIDTH_M = 2.0
CATWALK_ABOVE_M = 2.5
CATWALK_POST_EVERY_M = 6.0
FENDER_REACH_M = 2.0
FENDER_THICKNESS_M = 0.3

# Waterline within this of a BGT quay wall is quay.
QUAY_REACH_M = 1.5
# The quay strip: how far out into the water it stands, how far below the
# water level its face goes, and the shortest stretch worth one.
COPING_M = 0.35
BELOW_WATER_M = 1.5
MIN_QUAY_M = 2.0
# A quay top less than this above the water is a bank, not a quay.
MIN_QUAY_HEIGHT_M = 0.3
# Decks with no reading of their own: how far above the water, how thick.
DECK_ABOVE_M = {JETTY: 1.2, PONTOON: 0.5, CATWALK: CATWALK_ABOVE_M}
DECK_THICKNESS_M = {JETTY: 0.3, PONTOON: 1.0, CATWALK: 0.3}
# Surface-model heights more than this above the water are not a deck: a
# boat moored at it, a crane, a building on it.
DECK_MAX_ABOVE_M = 6.0
POST_RADIUS_M = 0.2
POST_ABOVE_M = 2.5


@dataclass
class HarbourSet:
    tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))  # RD + NAP
    tri_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    # Per triangle, which of `harbour_kit.PAINTS` it is painted, or -1 for
    # the material of its kind.
    tri_paint: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    counts: dict = field(default_factory=dict)
    # Per deck: its top, and the water level under it, for the checks.
    deck_top: np.ndarray = field(default_factory=lambda: np.zeros(0))
    deck_level: np.ndarray = field(default_factory=lambda: np.zeros(0))
    quay_bottom_below_water: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Per crane its top above the paving, per stack how many containers high.
    crane_height: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stack_tiers: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Each crane's outline in plan (shapely), and each container stack's:
    # the canopy model reads a crane as a clump of tall trees and a ragged
    # stack as a hedge, and the tree stage keeps off both.
    crane_footprints: list = field(default_factory=list)
    stack_footprints: list = field(default_factory=list)

    @property
    def no_trees(self) -> list:
        return [*self.crane_footprints, *self.stack_footprints]

    def __len__(self) -> int:
        return int(len(self.tris))

    def save(self, path: Path) -> Path:
        np.savez_compressed(
            path,
            tris=self.tris.astype(np.float64),
            tri_kind=self.tri_kind.astype(np.int8),
            tri_paint=self.tri_paint.astype(np.int8),
            paint_names=np.array(kit.PAINT_NAMES, dtype=str),
            paint_rgb=np.array([kit.PAINTS[n][0] for n in kit.PAINT_NAMES], dtype=np.float64),
            paint_rough=np.array([kit.PAINTS[n][1] for n in kit.PAINT_NAMES], dtype=np.float64),
            kind_names=np.array([KIND_NAMES[k] for k in sorted(KIND_NAMES)], dtype=str),
        )
        return path


def build_harbour(
    bbox: BBox,
    work_dir: Path,
    *,
    water,
    quays,
    surface_at,
    dsm_path=None,
    fetch_kwargs: dict | None = None,
    fittings: bool = True,
    cranes: bool = True,
    containers: bool = True,
) -> HarbourSet:
    """Quay faces, jetties, pontoons and mooring posts over this area.

    `water` is the WaterBody list, `quays` the BGT quay walls (shapely, RD),
    `surface_at(x, y)` the top of the ground as built (terrain plus layer).
    `fittings` adds piles, kerbs, bollards and ladders; `cranes` the crane
    rails and the cranes the surface model saw on them; `containers` the
    stacks it saw.
    """
    import shapely

    result = HarbourSet()
    if not water:
        return result
    fetch_kwargs = dict(fetch_kwargs or {})
    polygons = []
    levels = []
    for body in water:
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in body.rings]
        rings = [r for r in rings if len(r) >= 4]
        if not rings:
            continue
        valid = shapely.make_valid(shapely.Polygon(rings[0], rings[1:]))
        # Area only: a repaired outline can carry a stray line, and a
        # collection has no boundary to find the waterline on.
        for polygon in shapely.get_parts(valid):
            if polygon.geom_type == "Polygon" and not polygon.is_empty:
                polygons.append(polygon)
                levels.append(float(body.level_nap))
    if not polygons:
        return result
    wet = shapely.union_all(np.array(polygons, dtype=object))
    shapely.prepare(wet)
    tree = shapely.STRtree(np.array(polygons, dtype=object))
    levels = np.array(levels)

    def level_at(points: np.ndarray) -> np.ndarray:
        idx = tree.query_nearest(shapely.points(points), all_matches=False)
        out = np.full(len(points), np.nan)
        out[idx[0]] = levels[idx[1]]
        return out

    chunks: list[tuple[np.ndarray, int, int]] = []
    open_water = wet.buffer(-OPEN_WATER_M) if fittings else None

    # ---- quay faces --------------------------------------------------------
    if quays:
        box = shapely.box(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
        zone = shapely.union_all(np.array(list(quays), dtype=object)).buffer(QUAY_REACH_M)
        # The waterline, less the edge of the area, where the water is only cut.
        waterline = wet.boundary.difference(box.exterior.buffer(0.05))
        lines = shapely.get_parts(shapely.line_merge(waterline.intersection(zone)))
        bottoms = []
        for line in lines:
            if line.geom_type != "LineString" or line.length < MIN_QUAY_M:
                continue
            tris, below = _quay_strip(line, wet, level_at, surface_at)
            if len(tris):
                chunks.append((tris, QUAY, -1))
                bottoms.extend(below)
                if open_water is not None and not open_water.is_empty:
                    bollards, ladders = _quay_fittings(line, wet, open_water, level_at, surface_at)
                    for kind, parts in ((BOLLARD, bollards), (LADDER, ladders)):
                        if parts:
                            chunks.append((np.concatenate(parts), kind, -1))
                            name = KIND_NAMES[kind]
                            result.counts[name] = result.counts.get(name, 0) + len(parts)
        result.counts["quay_m"] = round(float(sum(
            g.length for g in lines if g.geom_type == "LineString" and g.length >= MIN_QUAY_M
        )), 1)
        result.quay_bottom_below_water = np.array(bottoms)

    # ---- jetties and pontoons ----------------------------------------------
    dsm = _raster(dsm_path)
    tops, deck_levels = [], []
    try:
        features, _ = fetch_current("kunstwerkdeel_vlak", bbox, **fetch_kwargs)
    except Exception as exc:  # noqa: BLE001 - decks are a refinement
        LOG.warning("could not read BGT jetties: %s", exc)
        features = []
    box = shapely.box(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
    by_kind: dict[int, list] = {}
    for feature in features:
        props = feature.get("properties") or {}
        kind_name = str(props.get("type") or "").lower()
        plus = str(props.get("plus_type") or "").lower()
        if kind_name == "steiger":
            kind = JETTY
        elif "ponton" in plus:
            kind = PONTOON
        else:
            continue
        try:
            outline = shapely.force_2d(shapely.geometry.shape(feature.get("geometry")))
        except Exception:  # noqa: BLE001
            continue
        for polygon in shapely.get_parts(shapely.make_valid(outline).intersection(box)):
            if polygon.geom_type != "Polygon" or polygon.area < 0.5:
                continue
            if polygon.intersection(wet).area < 0.2 * polygon.area:
                continue  # on land: a platform, not a jetty
            by_kind.setdefault(kind, []).append(polygon)
    pieces = []
    for kind, outlines in by_kind.items():
        # One deck per jetty, however many BGT pieces it is drawn in: each
        # piece read its own height off the surface model, and two pieces of
        # one jetty met in a step with a crack down it.
        merged = shapely.union_all(np.array(outlines, dtype=object))
        merged = merged.buffer(DECK_JOIN_M, join_style="mitre").buffer(
            -DECK_JOIN_M, join_style="mitre"
        )
        joined = int(len(outlines) - len(shapely.get_parts(merged)))
        if joined > 0:
            result.counts["deck_pieces_joined"] = result.counts.get("deck_pieces_joined", 0) + joined
        pieces.extend((polygon, kind) for polygon in shapely.get_parts(merged))
    for polygon, kind in pieces:
        if polygon.geom_type != "Polygon" or polygon.area < 0.5:
            continue
        c = polygon.representative_point()
        level = float(level_at(np.array([[c.x, c.y]]))[0])
        top = _deck_top(polygon, dsm, level, kind)
        tris = _extrude(polygon, top - DECK_THICKNESS_M[kind], top)
        if len(tris):
            chunks.append((tris, kind, -1))
            tops.append(top)
            deck_levels.append(level)
            result.counts[KIND_NAMES[kind]] = result.counts.get(KIND_NAMES[kind], 0) + 1
            if fittings and np.isfinite(level):
                piles, timber = _deck_fittings(polygon, kind, wet, level, top)
                if piles:
                    chunks.append((np.concatenate(piles), PILE, -1))
                    result.counts["pile"] = result.counts.get("pile", 0) + len(piles)
                if timber:
                    chunks.append((np.concatenate(timber), kind, -1))

    # ---- fender works: catwalks and fenders --------------------------------
    try:
        features, _ = fetch_current(
            "waterinrichtingselement_lijn", bbox, class_field="plus_type", **fetch_kwargs
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not read BGT fender works: %s", exc)
        features = []
    shore = wet.boundary.difference(box.exterior.buffer(0.05))
    for feature in features:
        props = feature.get("properties") or {}
        plus = str(props.get("plus_type") or "").lower()
        if not any(t in plus for t in FENDER_TYPES):
            continue
        try:
            line = shapely.force_2d(shapely.geometry.shape(feature.get("geometry")))
        except Exception:  # noqa: BLE001
            continue
        for part in shapely.get_parts(line.intersection(box)):
            if part.geom_type != "LineString" or part.length < 1.0:
                continue
            if shapely.distance(wet, part) > 3.0:
                continue  # a fender on land, or in water this area does not have
            stations = shapely.get_coordinates(shapely.line_interpolate_point(
                part, np.linspace(0.0, part.length, max(int(part.length), 2))
            ))
            level = float(np.nanmedian(level_at(stations)))
            gap = float(np.mean(shapely.distance(shore, shapely.points(stations))))
            if gap <= FENDER_REACH_M:
                top = _fender_top(stations, wet, surface_at, level)
                tris = _extrude(
                    part.buffer(FENDER_THICKNESS_M / 2.0, cap_style="flat"),
                    level - BELOW_WATER_M, top,
                )
                kind = FENDER
            else:
                deck = part.buffer(CATWALK_WIDTH_M / 2.0, cap_style="flat", join_style="mitre")
                top = _deck_top(deck, dsm, level, CATWALK)
                tris = _extrude(deck, top - DECK_THICKNESS_M[CATWALK], top)
                posts = shapely.get_coordinates(shapely.line_interpolate_point(
                    part, np.linspace(0.0, part.length,
                                      max(int(np.ceil(part.length / CATWALK_POST_EVERY_M)), 1) + 1)
                ))
                tris = np.concatenate([tris] + [
                    _post(x, y, level - BELOW_WATER_M, top - DECK_THICKNESS_M[CATWALK])
                    for x, y in posts
                ])
                tops.append(top)
                deck_levels.append(level)
                kind = CATWALK
                # A railing down both sides, a hand's breadth in from the edge.
                from .railings import open_runs, railing

                for side in (1.0, -1.0):
                    edge = shapely.offset_curve(part, side * (CATWALK_WIDTH_M / 2.0 - 0.1))
                    for run in open_runs(edge, wet):
                        points = shapely.get_coordinates(run)
                        rail = railing(points, np.full(len(points), top))
                        if len(rail):
                            chunks.append((rail, RAILING, -1))
                            result.counts["railing_m"] = round(
                                result.counts.get("railing_m", 0.0) + run.length, 1
                            )
            if len(tris):
                chunks.append((tris, kind, -1))
                key = f"{KIND_NAMES[kind]}_m"
                result.counts[key] = round(result.counts.get(key, 0.0) + part.length, 1)
    result.deck_top = np.array(tops)
    result.deck_level = np.array(deck_levels)

    # ---- mooring posts -------------------------------------------------------
    try:
        features, _ = fetch_current(
            "waterinrichtingselement_punt", bbox, class_field="plus_type", **fetch_kwargs
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not read BGT mooring posts: %s", exc)
        features = []
    posts = []
    for feature in features:
        props = feature.get("properties") or {}
        if "meerpaal" not in str(props.get("plus_type") or "").lower():
            continue
        coords = (feature.get("geometry") or {}).get("coordinates")
        if not coords or len(coords) < 2:
            continue
        x, y = float(coords[0]), float(coords[1])
        if not (bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax):
            continue
        if shapely.distance(wet, shapely.Point(x, y)) > 3.0:
            continue
        posts.append((x, y))
    if posts:
        p = np.array(posts)
        lv = level_at(p)
        for (x, y), level in zip(p, lv):
            chunks.append((_post(x, y, level - BELOW_WATER_M, level + POST_ABOVE_M), POST, -1))
        result.counts["mooring_post"] = len(posts)

    # ---- crane rails, cranes and containers ----------------------------------
    if cranes or containers:
        _terminal(
            result, chunks, bbox, wet, surface_at, dsm, fetch_kwargs,
            cranes=cranes, containers=containers,
        )

    if chunks:
        result.tris = np.concatenate([t for t, _, _ in chunks])
        result.tri_kind = np.concatenate([np.full(len(t), k, dtype=np.int8) for t, k, _ in chunks])
        result.tri_paint = np.concatenate([np.full(len(t), p, dtype=np.int8) for t, _, p in chunks])
    LOG.info(
        "harbour: %s",
        ", ".join(f"{k} {v}" for k, v in result.counts.items()) or "nothing",
    )
    result.save(Path(work_dir) / "harbour.npz")
    return result


def _raster(path):
    if path is None or not Path(path).is_file():
        return None
    import rasterio

    with rasterio.open(path) as dataset:
        return dataset.read(1).astype(np.float64), dataset.transform


def _shapes(collection: str, bbox: BBox, fetch_kwargs: dict, keep=None) -> list:
    """One BGT collection as shapely geometries, those `keep` says yes to."""
    import shapely

    try:
        features, _ = fetch_current(collection, bbox, **fetch_kwargs)
    except Exception as exc:  # noqa: BLE001 - every one of these is a refinement
        LOG.warning("could not read BGT %s: %s", collection, exc)
        return []
    out = []
    for feature in features:
        props = feature.get("properties") or {}
        if keep is not None and not keep(props):
            continue
        try:
            geometry = shapely.force_2d(shapely.geometry.shape(feature.get("geometry")))
        except Exception:  # noqa: BLE001 - one broken outline
            continue
        if not geometry.is_empty:
            out.append((shapely.make_valid(geometry), props))
    return out


def _terminal(result, chunks, bbox, wet, surface_at, dsm, fetch_kwargs, *, cranes, containers):
    """Crane rails, and what the surface model saw standing on the terminal."""
    import shapely

    from . import terminal

    box = shapely.box(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
    tracks = []
    if cranes:
        rails, levels = [], []
        for line, props in _shapes(
            "spoor", bbox, fetch_kwargs,
            keep=lambda p: "kraan" in str(p.get("plus_functie") or "").lower(),
        ):
            try:
                level = int(props.get("relatieve_hoogteligging") or 0)
            except (TypeError, ValueError):
                level = 0
            for part in shapely.get_parts(line.intersection(box)):
                if part.geom_type == "LineString" and part.length >= 2.0:
                    rails.append(part)
                    levels.append(level)
        length = 0.0
        for line, level in zip(rails, levels):
            count = max(int(np.ceil(line.length / 10.0)), 1)
            points = shapely.get_coordinates(
                shapely.line_interpolate_point(line, np.linspace(0.0, line.length, count + 1))
            )
            tops = np.asarray(surface_at(points[:, 0], points[:, 1]), dtype=np.float64)
            if level > 0 and dsm is not None:
                # Up on a runway: as high as the survey saw the most of it.
                seen = _sample(dsm, points)
                seen = seen[np.isfinite(seen)]
                if len(seen) and np.isfinite(tops).any() and np.median(seen) > np.nanmedian(tops) + 2.0:
                    tops = np.where(np.isfinite(tops), float(np.median(seen)), np.nan)
            ok = np.isfinite(tops)
            for start, stop in _true_runs(ok):
                if stop - start < 2:
                    continue
                tris = kit.rail(points[start:stop], tops[start:stop])
                if len(tris):
                    chunks.append((tris, CRANE_RAIL, -1))
                    length += float(np.linalg.norm(np.diff(points[start:stop], axis=0), axis=1).sum())
        if length:
            result.counts["crane_rail_m"] = round(length, 1)
        tracks = terminal.pair_tracks(rails, levels)
    if dsm is None:
        return
    field = _height_field(dsm, surface_at, wet)
    if field is None:
        return
    buildings = [g.buffer(1.0) for g, _ in _shapes("pand", bbox, fetch_kwargs)]
    blocked = field.mask(buildings)

    if tracks:
        heights = []
        for crane in terminal.find_cranes(tracks, field, wet=wet, blocked=blocked):
            ground = float(surface_at(crane.centre[0], crane.centre[1]))
            if not np.isfinite(ground):
                continue
            track = crane.track
            top = ground + crane.height_m
            if crane.kind == "gantry":
                parts = kit.gantry_crane(
                    crane.centre, track.along, track.gauge, ground, top, crane.overhang_m
                )
            elif crane.kind == "quay":
                parts = kit.quay_crane(
                    crane.centre, track.along, crane.water, track.gauge, ground, top,
                    boom_height=ground + crane.boom_m - 2.2, outreach=crane.outreach_m,
                    backreach=crane.backreach_m, boom_up=crane.boom_up,
                )
            else:
                parts = kit.jib_crane(
                    crane.centre, track.along, track.gauge, ground, top,
                    np.r_[crane.tip, ground + crane.tip_m],
                )
            for tris, paint in parts:
                chunks.append((tris, CRANE, kit.PAINT_NAMES.index(paint)))
            plan = np.concatenate([tris.reshape(-1, 3)[:, :2] for tris, _ in parts])
            result.crane_footprints.append(
                shapely.MultiPoint(plan).convex_hull.buffer(CRANE_CLEAR_M)
            )
            heights.append(crane.height_m)
            key = f"{crane.kind}_crane"
            result.counts[key] = result.counts.get(key, 0) + 1
        result.crane_height = np.array(heights)

    if containers:
        yard = [g for g, _ in _shapes("onbegroeidterreindeel", bbox, fetch_kwargs)]
        if not yard:
            return
        sheds = [g.buffer(1.0) for g, _ in _shapes("overigbouwwerk", bbox, fetch_kwargs)]
        on_yard = field.mask(yard) & ~field.mask([wet.buffer(1.0)])
        stacks = terminal.find_containers(
            field, yard=on_yard, blocked=blocked | field.mask(sheds), tracks=tracks
        )
        if not stacks:
            return
        centres = np.array([s.centre for s in stacks])
        grounds = np.asarray(surface_at(centres[:, 0], centres[:, 1]), dtype=np.float64)
        colours = [kit.PAINT_NAMES.index(n) for n in kit.CONTAINER_PAINTS]
        built = 0
        tiers = []
        by_paint: dict[int, list] = {}
        for stack, ground in zip(stacks, grounds):
            if not np.isfinite(ground):
                continue
            tiers.append(stack.tiers)
            across = np.array([-stack.along[1], stack.along[0]])
            half = stack.along * (stack.length / 2.0 + 1.0)
            side = across * (kit.CONTAINER_WIDTH_M / 2.0 + 1.0)
            result.stack_footprints.append(shapely.Polygon([
                stack.centre - half - side, stack.centre + half - side,
                stack.centre + half + side, stack.centre - half + side,
            ]))
            for tier in range(stack.tiers):
                bottom = ground + tier * kit.CONTAINER_HEIGHT_M
                tris = kit.container(stack.centre, stack.along, stack.length, bottom)
                # No floor under any of them, and a roof only on the top one.
                tris = np.concatenate([tris[:2], tris[4:]]) if tier == stack.tiers - 1 else tris[4:]
                choice = kit.pick(
                    stack.centre[0] + 0.3 * tier, stack.centre[1], len(colours),
                    kit.CONTAINER_WEIGHTS,
                )
                by_paint.setdefault(colours[choice], []).append(tris)
                built += 1
        for paint, parts in by_paint.items():
            chunks.append((np.concatenate(parts), CONTAINER, paint))
        if built:
            result.counts["container"] = built
            result.counts["container_stack"] = len(tiers)
            result.stack_tiers = np.array(tiers, dtype=np.int64)


def _true_runs(flags: np.ndarray):
    """(start, stop) of every run of True in `flags`."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(flags, dtype=np.int8), 0]))
    return zip(edges[0::2], edges[1::2])


def _sample(dsm, points: np.ndarray) -> np.ndarray:
    """The surface model at `points`; NaN off it and where it has no return."""
    raster, transform = dsm
    cols = np.floor((points[:, 0] - transform.c) / transform.a).astype(int)
    rows = np.floor((points[:, 1] - transform.f) / transform.e).astype(int)
    ok = (rows >= 0) & (rows < raster.shape[0]) & (cols >= 0) & (cols < raster.shape[1])
    out = np.full(len(points), np.nan)
    out[ok] = raster[rows[ok], cols[ok]]
    out[~np.isfinite(out) | (np.abs(out) > 1e6)] = np.nan
    return out


def _height_field(dsm, surface_at, wet):
    """The surface model as height above the ground as built.

    The ground is read every `GROUND_STEP_M` and filled in between: a
    terminal is flat, and the ground mesh is slow to ask four million times.
    Over water the ground is the nearest land's, so that a boom over a ship
    is as high above the quay as the crane it belongs to.
    """
    import shapely

    from .terminal import HeightField

    raster, transform = dsm
    surface = np.where(np.isfinite(raster) & (np.abs(raster) < 1e6), raster, np.nan)
    rows, cols = surface.shape
    step = max(int(round(GROUND_STEP_M / abs(transform.a))), 1)
    r = np.unique(np.r_[np.arange(0, rows, step), rows - 1])
    c = np.unique(np.r_[np.arange(0, cols, step), cols - 1])
    x = transform.c + (c + 0.5) * transform.a
    y = transform.f + (r + 0.5) * transform.e
    gx, gy = np.meshgrid(x, y)
    ground = np.asarray(surface_at(gx.ravel(), gy.ravel()), dtype=np.float64)
    ground[shapely.contains_xy(wet, gx.ravel(), gy.ravel())] = np.nan
    ground = ground.reshape(gx.shape)
    if not np.isfinite(ground).any():
        return None
    for _ in range(max(ground.shape)):
        holes = ~np.isfinite(ground)
        if not holes.any():
            break
        padded = np.pad(ground, 1, constant_values=np.nan)
        around = np.stack([
            padded[dr:dr + ground.shape[0], dc:dc + ground.shape[1]]
            for dr in range(3) for dc in range(3)
        ])
        known = np.isfinite(around)
        total = np.where(known, around, 0.0).sum(axis=0)
        seen = known.sum(axis=0)
        fill = holes & (seen > 0)
        ground[fill] = total[fill] / seen[fill]
    along = np.empty((len(r), cols))
    for i in range(len(r)):
        along[i] = np.interp(np.arange(cols), c, ground[i])
    full = np.empty((rows, cols))
    for j in range(cols):
        full[:, j] = np.interp(np.arange(rows), r, along[:, j])
    return HeightField(surface - full, transform)


def _quay_fittings(line, wet, open_water, level_at, surface_at):
    """Bollards along one stretch of harbour quay, and ladders down its face."""
    import shapely

    if line.length < BOLLARD_EVERY_M / 2.0:
        return [], []
    at = np.arange(BOLLARD_EVERY_M / 2.0, line.length, BOLLARD_EVERY_M)
    points = shapely.get_coordinates(shapely.line_interpolate_point(line, at))
    ahead = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(at + 0.5, line.length)))
    behind = shapely.get_coordinates(shapely.line_interpolate_point(line, at - 0.5))
    tangent = ahead - behind
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-12)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    probe = points + 0.2 * normal
    normal[~shapely.contains_xy(wet, probe[:, 0], probe[:, 1])] *= -1.0
    land = points - BOLLARD_BACK_M * normal
    top = np.asarray(surface_at(land[:, 0], land[:, 1]), dtype=np.float64)
    level = level_at(points)
    keep = (
        np.isfinite(top) & np.isfinite(level) & (top >= level + FITTED_QUAY_M)
        & ~shapely.contains_xy(wet, land[:, 0], land[:, 1])
        & (shapely.distance(open_water, shapely.points(points)) <= OPEN_WATER_M + 3.0)
    )
    bollards, ladders = [], []
    every = max(int(round(LADDER_EVERY_M / BOLLARD_EVERY_M)), 1)
    for i in np.flatnonzero(keep):
        bollards.append(kit.bollard(land[i, 0], land[i, 1], top[i]))
        if i % every == every // 2:
            # Beside the bollard, not under its lines.
            foot = points[i] + tangent[i] * 3.0 + normal[i] * (COPING_M + 0.05)
            ladders.append(kit.ladder(foot, normal[i], top[i] + 0.05, level[i] - LADDER_BELOW_WATER_M))
    return bollards, ladders


def _deck_fittings(polygon, kind: int, wet, level: float, top: float):
    """What a deck stands on and is edged with: (piles, timber).

    A jetty: rows of piles across it with a headstock over each row, and a
    kerb round the deck. A pontoon: a guide pile at either end of one side.
    """
    import shapely

    corners = shapely.get_coordinates(shapely.oriented_envelope(polygon))
    if len(corners) < 4:
        return [], []
    a, b = corners[1] - corners[0], corners[3] - corners[0]
    if np.linalg.norm(a) < np.linalg.norm(b):
        a, b = b, a
    length, width = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if length < 1.0 or width < 0.3:
        return [], []
    u, v = a / length, b / width
    origin = corners[0]
    bottom = level - BELOW_WATER_M
    piles, timber = [], []
    if kind == PONTOON:
        if length < GUIDE_PILE_MIN_M:
            return [], []
        for along in (1.0, length - 1.0):
            p = origin + u * along + v * (width + PILE_RADIUS_M + 0.1)
            if wet.contains(shapely.Point(p)) and not polygon.contains(shapely.Point(p)):
                piles.append(kit.prism(p[0], p[1], PILE_RADIUS_M + 0.05, bottom, level + POST_ABOVE_M))
        return piles, timber

    under = top - DECK_THICKNESS_M[kind]
    rows = max(int(np.ceil((length - 1.5) / PILE_EVERY_M)), 1) + 1
    inset = min(PILE_INSET_M, width / 2.0)
    across = (
        np.array([width / 2.0]) if width < 1.6
        else np.linspace(inset, width - inset, max(int(np.ceil((width - 2 * inset) / PILE_EVERY_M)), 1) + 1)
    )
    for along in np.linspace(min(0.75, length / 2.0), max(length - 0.75, length / 2.0), rows):
        row = origin + u * along + v * across[:, None]
        standing = shapely.contains_xy(polygon, row[:, 0], row[:, 1]) & shapely.contains_xy(wet, row[:, 0], row[:, 1])
        row = row[standing]
        for x, y in row:
            piles.append(kit.prism(x, y, PILE_RADIUS_M, bottom, under - HEADSTOCK_M, sides=6))
        if len(row) >= 2:
            timber.append(kit.beam(
                np.r_[row[0] - v * 0.2, under - HEADSTOCK_M / 2.0],
                np.r_[row[-1] + v * 0.2, under - HEADSTOCK_M / 2.0], HEADSTOCK_M,
            ))
        elif len(row) == 1:
            timber.append(kit.box(row[0], v, min(width, 1.2), HEADSTOCK_M, under - HEADSTOCK_M, under))
    if width >= KERB_MIN_DECK_M:
        inner = polygon.buffer(-KERB_M, join_style="mitre")
        for ring in shapely.get_parts(polygon.difference(inner)):
            if ring.geom_type == "Polygon" and ring.area > 0.05:
                kerb = _extrude(ring, top, top + KERB_M * 0.8)
                if len(kerb):
                    timber.append(kerb)
    return piles, timber


# Pieces of a jetty this close are one jetty.
DECK_JOIN_M = 0.05


def _deck_top(polygon, dsm, level: float, kind: int) -> float:
    """The deck's top: the surface model's reading on it, or a standard height."""
    default = level + DECK_ABOVE_M[kind]
    if dsm is None or not np.isfinite(level):
        return default
    from rasterio.features import geometry_mask

    raster, transform = dsm
    inner = polygon.buffer(-0.3)
    if inner.is_empty:
        inner = polygon
    mask = ~geometry_mask([inner.__geo_interface__], raster.shape, transform)
    values = raster[mask]
    values = values[np.isfinite(values) & (values > level + 0.1) & (values < level + DECK_MAX_ABOVE_M)]
    if len(values) < 10:
        return default
    # The lower half of what stands on it: the deck, not the boat beside it.
    return float(np.percentile(values, 25))


def _fender_top(stations, wet, surface_at, level: float) -> float:
    """A fender's top: the quay it hangs on, read a metre in from the water."""
    import shapely

    lines = shapely.shortest_line(shapely.points(stations), wet.boundary)
    near = shapely.get_coordinates(shapely.line_interpolate_point(lines, 1.0, normalized=True))
    # Step a metre past the waterline, away from the fender.
    step = near - stations
    step /= np.maximum(np.linalg.norm(step, axis=1), 1e-9)[:, None]
    away = near + step
    inland = ~shapely.contains_xy(wet, away[:, 0], away[:, 1])
    tops = np.asarray(surface_at(away[inland, 0], away[inland, 1]), dtype=np.float64)
    tops = tops[np.isfinite(tops) & (tops > level + MIN_QUAY_HEIGHT_M)]
    return float(np.median(tops)) if len(tops) else level + CATWALK_ABOVE_M


def _quay_strip(line, wet, level_at, surface_at):
    """One stretch of quay: a strip standing just out into the water."""
    import shapely

    length = line.length
    count = max(int(np.ceil(length / 1.0)), 1)
    points = shapely.get_coordinates(
        shapely.line_interpolate_point(line, np.linspace(0.0, length, count + 1))
    )
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-12)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    # Point the normals into the water.
    probe = points + 0.2 * normal
    inside = shapely.contains_xy(wet, probe[:, 0], probe[:, 1])
    normal[~inside] *= -1.0
    land = points - 0.4 * normal
    outer = points + COPING_M * normal
    top = np.asarray(surface_at(land[:, 0], land[:, 1]), dtype=np.float64)
    level = level_at(points)
    keep = np.isfinite(top) & np.isfinite(level) & (top > level + MIN_QUAY_HEIGHT_M)
    if keep.sum() < 2:
        return np.zeros((0, 3, 3)), []
    bottom = level - BELOW_WATER_M
    tris = []
    for i in range(len(points) - 1):
        if not (keep[i] and keep[i + 1]):
            continue
        a_in = np.r_[points[i], top[i]]
        b_in = np.r_[points[i + 1], top[i + 1]]
        a_out = np.r_[outer[i], top[i]]
        b_out = np.r_[outer[i + 1], top[i + 1]]
        a_low = np.r_[outer[i], bottom[i]]
        b_low = np.r_[outer[i + 1], bottom[i + 1]]
        a_back = np.r_[points[i], bottom[i]]
        b_back = np.r_[points[i + 1], bottom[i + 1]]
        tris.extend([
            [a_in, b_in, b_out], [a_in, b_out, a_out],      # coping
            [a_out, b_out, b_low], [a_out, b_low, a_low],   # face to the water
            [a_back, b_back, b_in], [a_back, b_in, a_in],   # back, against the quay
            [a_low, b_low, b_back], [a_low, b_back, a_back],  # underside
        ])
    # Close the ends of each run.
    runs = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0]))
    for start, stop in zip(runs[0::2], runs[1::2]):
        for i in (start, stop - 1):
            q = [np.r_[points[i], top[i]], np.r_[outer[i], top[i]],
                 np.r_[outer[i], bottom[i]], np.r_[points[i], bottom[i]]]
            tris.extend([[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
    below = list((level - bottom)[keep])
    return np.array(tris, dtype=np.float64), below


def _extrude(polygon, bottom: float, top: float) -> np.ndarray:
    """A polygon as a slab: top, bottom and sides."""
    import shapely

    pieces = shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))
    if not len(pieces):
        return np.zeros((0, 3, 3))
    flat = shapely.get_coordinates(shapely.get_exterior_ring(pieces)).reshape(-1, 4, 2)[:, :3]
    tris = []
    for t in flat:
        tris.append(np.column_stack([t, np.full(3, top)]))
        tris.append(np.column_stack([t[::-1], np.full(3, bottom)]))
    for ring in [polygon.exterior, *polygon.interiors]:
        c = np.asarray(ring.coords)[:, :2]
        for a, b in zip(c[:-1], c[1:]):
            a_t, b_t = np.r_[a, top], np.r_[b, top]
            a_b, b_b = np.r_[a, bottom], np.r_[b, bottom]
            tris.append([a_b, b_b, b_t])
            tris.append([a_b, b_t, a_t])
    return np.array(tris, dtype=np.float64)


def _post(x: float, y: float, bottom: float, top: float, sides: int = 8) -> np.ndarray:
    """An eight-sided post with a flat top."""
    angles = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    ring = np.column_stack([x + POST_RADIUS_M * np.cos(angles), y + POST_RADIUS_M * np.sin(angles)])
    tris = []
    for i in range(sides):
        a, b = ring[i], ring[(i + 1) % sides]
        tris.append([np.r_[a, bottom], np.r_[b, bottom], np.r_[b, top]])
        tris.append([np.r_[a, bottom], np.r_[b, top], np.r_[a, top]])
        tris.append([np.r_[x, y, top], np.r_[a, top], np.r_[b, top]])
    return np.array(tris, dtype=np.float64)


__all__ = ["HarbourSet", "KIND_NAMES", "build_harbour"]
