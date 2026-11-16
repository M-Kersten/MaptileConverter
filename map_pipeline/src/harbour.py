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
  on it; tight along the waterline it is a fender, a thin strip from below the
  water to the quay top.

All of it is heights read off the water levels, which `surfaces` reads once
per connected water from the surface model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bgt import fetch_current
from .geo import BBox

LOG = logging.getLogger(__name__)

QUAY, JETTY, PONTOON, POST, CATWALK, FENDER = 0, 1, 2, 3, 4, 5
KIND_NAMES = {
    QUAY: "quay", JETTY: "jetty", PONTOON: "pontoon", POST: "mooring_post",
    CATWALK: "catwalk", FENDER: "fender",
}
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
    counts: dict = field(default_factory=dict)
    # Per deck: its top, and the water level under it, for the checks.
    deck_top: np.ndarray = field(default_factory=lambda: np.zeros(0))
    deck_level: np.ndarray = field(default_factory=lambda: np.zeros(0))
    quay_bottom_below_water: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return int(len(self.tris))

    def save(self, path: Path) -> Path:
        np.savez_compressed(
            path,
            tris=self.tris.astype(np.float64),
            tri_kind=self.tri_kind.astype(np.int8),
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
) -> HarbourSet:
    """Quay faces, jetties, pontoons and mooring posts over this area.

    `water` is the WaterBody list, `quays` the BGT quay walls (shapely, RD),
    `surface_at(x, y)` the top of the ground as built (terrain plus layer).
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
        polygon = shapely.make_valid(shapely.Polygon(rings[0], rings[1:]))
        if not polygon.is_empty:
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

    chunks: list[tuple[np.ndarray, int]] = []

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
                chunks.append((tris, QUAY))
                bottoms.extend(below)
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
            c = polygon.representative_point()
            level = float(level_at(np.array([[c.x, c.y]]))[0])
            top = _deck_top(polygon, dsm, level, kind)
            tris = _extrude(polygon, top - DECK_THICKNESS_M[kind], top)
            if len(tris):
                chunks.append((tris, kind))
                tops.append(top)
                deck_levels.append(level)
                result.counts[KIND_NAMES[kind]] = result.counts.get(KIND_NAMES[kind], 0) + 1

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
            if len(tris):
                chunks.append((tris, kind))
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
            chunks.append((_post(x, y, level - BELOW_WATER_M, level + POST_ABOVE_M), POST))
        result.counts["mooring_post"] = len(posts)

    if chunks:
        result.tris = np.concatenate([t for t, _ in chunks])
        result.tri_kind = np.concatenate([np.full(len(t), k, dtype=np.int8) for t, k in chunks])
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
