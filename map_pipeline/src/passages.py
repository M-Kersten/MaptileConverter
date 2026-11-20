"""Roads and water that run under a building: a tunnel cut through its base.

3DBAG builds every building as a block standing on the ground over its whole
footprint, and that footprint is the BAG's *top view* of the building. Where a
building spans a street or a canal -- a gateway through a terrace, a
cantilevered storey over the pavement, a shopping centre straddling a road --
the block comes down on the street. Hoog Catharijne in Utrecht is the case
that makes it obvious: the Catharijnesingel flows under the shopping centre,
buses and taxis drive under it, and the model had the canal end in a brick
wall and the road run into a facade.

The BGT knows better. It draws the ground at ground level, so a road or a water
at ``relatieve_hoogteligging`` 0 inside a BAG footprint is ground the building
stands *over*, not on. So wherever one does, the building is given a tunnel:
its walls are opened below a ceiling height where the road or water crosses
them, any roof below that height is cut away over the passage, and the hole is
lined -- side walls where the passage meets the rest of the building, and a
ceiling over it. The road and the water themselves are already there, drawn at
ground level by the stages that build them. Paving and station platforms
widen a passage one of those makes -- Utrecht Centraal's hall spans its
platforms as well as its tracks -- but never make one of their own.

How high the tunnel is cannot be measured: lidar does not see under a building,
and neither 3DBAG nor the BAG records an underpass. Like the depth of a tunnel
(`structures`) it is drawn: a clearance above the ground for what passes under
(`buildings.passages.clearance_m`), held a slab below the lowest roof over it.
Where a building is too low to leave that much headroom, it is left alone and
counted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

LOG = logging.getLogger(__name__)

# What passes under, which decides how much headroom the tunnel gets. Paving
# -- a platform, a forecourt -- only ever widens a passage one of the others
# makes: a station hall spans its platforms as well as its tracks, but paving
# alone under a building is no road through it.
ROAD, PATH, RAIL, WATER, PAVED = "road", "path", "rail", "water", "paved"
KINDS = (ROAD, PATH, RAIL, WATER, PAVED)

# BGT `functie` on a wegdeel, matched by substring as `surfaces.ROAD_FUNCTIONS`
# is. Anything not listed is driven on, so it gets a road's headroom.
PATH_FUNCTIONS = ("voetpad", "fietspad", "voetgangersgebied", "ruiterpad")
RAIL_FUNCTIONS = ("spoorbaan",)

DEFAULTS = {
    "enabled": True,
    # Above the highest ground under the passage. A road in the Netherlands is
    # signed from 4.2 m and built to 4.5-4.6; a catenary wants about 6; a
    # footpath or a canal one storey.
    "clearance_m": {ROAD: 4.5, PATH: 3.0, RAIL: 6.0, WATER: 3.0, PAVED: 3.0},
    # Less than this under the building and nothing gets through: it is left
    # whole, and counted.
    "min_headroom_m": 2.2,
    # Narrower than this is where the BGT and the BAG disagree about where a
    # facade is, not a passage: a pavement reaching 30 cm under a wall.
    "min_width_m": 1.0,
    "min_area_m2": 4.0,
    # How far the ceiling stays below the lowest roof over the passage.
    "slab_m": 0.5,
}

# A wall this close to the passage is on its edge. Footprint, walls and the
# passage cut from the footprint share their coordinates to well under this.
EDGE_M = 0.005
# A wall crossing less of the passage than this only grazes a corner of it.
GRAZE_M = 0.05
# The lining is drawn in pieces no longer than this, so its top can follow a
# roof that is lower than the ceiling somewhere along it.
LINING_STEP_M = 2.0


@dataclass
class PassageArea:
    """One BGT polygon at ground level that may run under a building."""

    polygon: object  # shapely Polygon, RD
    kind: str


@dataclass
class PassageStats:
    passages: int = 0
    buildings: int = 0
    too_low: int = 0
    area_m2: float = 0.0
    by_kind: dict[str, int] = field(default_factory=dict)
    # One per passage, for the checks: the building's id, the passage seen
    # from above (shapely, RD), its ceiling and the ground under it, in NAP.
    cut: list[tuple[str, object, float, float]] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "passages": self.passages,
            "buildings_with_a_passage": self.buildings,
            "too_low_to_pass_under": self.too_low,
            "passage_area_m2": round(self.area_m2, 1),
            "by_kind": dict(sorted(self.by_kind.items())),
            "headroom": (
                "drawn, not surveyed: a clearance above the ground for what "
                "passes under, a slab below the lowest roof over it"
            ),
        }


def passage_kind(collection: str, properties: dict) -> str | None:
    """What a BGT polygon is, as something that can pass under a building.

    None for anything off the ground: a footbridge or a deck
    (``relatieve_hoogteligging`` above 0) runs over the building's ground floor
    or onto it, and a tunnel (below 0) under its foundations, where the
    `structures` stage puts it.
    """
    try:
        level = int(properties.get("relatieve_hoogteligging") or 0)
    except (TypeError, ValueError):
        level = 0
    if level != 0:
        return None
    if collection == "waterdeel":
        return WATER
    if collection == "onbegroeidterreindeel":
        return PAVED
    # A platform is a structure in the BGT, not paving, but stands under a
    # station hall the way the paving between the tracks would.
    if collection == "kunstwerkdeel_vlak":
        return PAVED if str(properties.get("type") or "").strip().lower() == "perron" else None
    if collection != "wegdeel":
        return None
    function = str(properties.get("functie") or "").strip().lower()
    if any(token in function for token in RAIL_FUNCTIONS):
        return RAIL
    if any(token in function for token in PATH_FUNCTIONS):
        return PATH
    return ROAD


def fetch_passage_areas(bbox, fetch_current, fetch_kwargs: dict) -> list[PassageArea]:
    """Every road, water, paved and platform polygon at ground level, whole,
    over `bbox`.

    Not clipped to the area: a building on the edge is kept whole, and so is
    the road under it. The surfaces stage has read the first three already,
    so on a normal run only the platforms are fetched here.
    """
    import shapely
    from shapely.geometry import shape

    areas: list[PassageArea] = []
    for collection in ("wegdeel", "waterdeel", "onbegroeidterreindeel", "kunstwerkdeel_vlak"):
        try:
            features, _ = fetch_current(collection, bbox, **fetch_kwargs)
        except Exception as exc:  # noqa: BLE001 - the buildings stay whole
            LOG.warning("could not read BGT %s for passages under buildings: %s", collection, exc)
            continue
        for feature in features:
            kind = passage_kind(collection, feature.get("properties") or {})
            if kind is None or not feature.get("geometry"):
                continue
            try:
                geometry = shapely.make_valid(shapely.force_2d(shape(feature["geometry"])))
            except Exception:  # noqa: BLE001 - one broken outline
                continue
            for part in shapely.get_parts(geometry):
                if part.geom_type == "Polygon" and part.area > 0:
                    areas.append(PassageArea(part, kind))
    return areas


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def footprint(roof_tris: np.ndarray):
    """A building's footprint: everything its roofs cover, seen from above."""
    import shapely

    if not len(roof_tris):
        return shapely.Polygon()
    xy = np.asarray(roof_tris, dtype=np.float64)[:, :, :2]
    keep = np.abs(_cross2(xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0])) > 1e-6
    if not keep.any():
        return shapely.Polygon()
    triangles = shapely.polygons(xy[keep])
    return shapely.make_valid(shapely.union_all(triangles, grid_size=1e-4))


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _polygons(geometry) -> list:
    import shapely

    if geometry is None or geometry.is_empty:
        return []
    return [p for p in shapely.get_parts(geometry) if p.geom_type == "Polygon" and not p.is_empty]


def opening(geometry, radius: float):
    """Morphological opening: what is left of `geometry` once everything
    narrower than twice `radius` is gone. Square joins keep a street's straight
    edges and right-angled corners where they were."""
    if geometry.is_empty or radius <= 0:
        return geometry
    eroded = geometry.buffer(-radius, join_style="mitre", mitre_limit=4.0)
    if eroded.is_empty:
        return eroded
    return eroded.buffer(radius, join_style="mitre", mitre_limit=4.0).intersection(geometry)


class SurfaceHeights:
    """The height of a set of triangles where they cover a point, seen from
    above: the lowest where several do. For "how much building is over here"."""

    def __init__(self, triangles: np.ndarray) -> None:
        import shapely

        triangles = np.asarray(triangles, dtype=np.float64)
        xy = triangles[:, :, :2]
        area = _cross2(xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0])
        keep = np.abs(area) > 1e-6
        self.triangles = triangles[keep]
        self.tree = (
            shapely.STRtree(shapely.polygons(self.triangles[:, :, :2]))
            if len(self.triangles) else None
        )

    def lowest(self, x, y) -> np.ndarray:
        import shapely

        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        out = np.full(x.shape, np.nan)
        if self.tree is None or not len(x):
            return out
        point, tri = self.tree.query(shapely.points(x, y), predicate="intersects")
        if not len(point):
            return out
        t = self.triangles[tri]
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        p = np.column_stack([x[point], y[point]])
        denominator = _cross2(b[:, :2] - a[:, :2], c[:, :2] - a[:, :2])
        u = _cross2(p - a[:, :2], c[:, :2] - a[:, :2]) / denominator
        v = _cross2(b[:, :2] - a[:, :2], p - a[:, :2]) / denominator
        z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
        best = np.full(x.shape, np.inf)
        np.minimum.at(best, point, z)
        hit = np.isfinite(best)
        out[hit] = best[hit]
        return out


def split_at(triangles: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """``(below, above)`` a horizontal plane, cutting the triangles across it."""
    from .buildings import split_walls_at_height

    if not len(triangles):
        empty = np.zeros((0, 3, 3), dtype=np.float64)
        return empty, empty
    return split_walls_at_height(triangles, np.full(len(triangles), float(z)))


def _clip_convex(points: np.ndarray, axis: int, limit: float, keep_greater: bool) -> np.ndarray:
    """One Sutherland-Hodgman pass: keep a convex polygon on one side of a line."""
    out = []
    count = len(points)
    for i in range(count):
        current, previous = points[i], points[i - 1]
        current_in = current[axis] >= limit if keep_greater else current[axis] <= limit
        previous_in = previous[axis] >= limit if keep_greater else previous[axis] <= limit
        if current_in != previous_in:
            span = current[axis] - previous[axis]
            if abs(span) > 1e-12:
                t = (limit - previous[axis]) / span
                out.append(previous + t * (current - previous))
        if current_in:
            out.append(current)
    return np.asarray(out) if out else np.zeros((0, points.shape[1]))


def open_walls(walls: np.ndarray, region, top: float) -> tuple[np.ndarray, int]:
    """The walls with every stretch that meets `region` taken out below `top`.

    A wall is vertical, so from above it is a line. Where that line crosses
    the passage it is in the way: an inner wall standing in the passage, or
    the party wall between two buildings over one road. Where it runs along
    the passage's edge it is the building's outline, and the passage comes
    out there -- a tunnel mouth, an arcade along a street. Both are opened.
    Only the triangles that are opened are cut at `top`: cutting the rest
    would leave their neighbours a vertex in the middle of an edge. Returns
    the walls left and how many triangles were opened.
    """
    import shapely

    from .buildings import split_walls_at_height

    walls = np.asarray(walls, dtype=np.float64)
    if not len(walls) or region.is_empty:
        return walls, 0
    edge = region.buffer(EDGE_M, join_style="mitre", mitre_limit=4.0)
    shapely.prepare(edge)
    near = _near_mask(walls, edge)

    kept = [walls[~near]]
    opened = 0
    for triangle in walls[near]:
        if triangle[:, 2].min() >= top:
            kept.append(triangle[None])
            continue
        frame = _wall_frame(triangle)
        gaps = _gaps(frame, edge) if frame is not None else []
        s = (triangle[:, :2] - frame[0]) @ frame[1] if frame is not None else None
        if not gaps or not any(b > s.min() + 1e-4 and a < s.max() - 1e-4 for a, b in gaps):
            kept.append(triangle[None])
            continue
        opened += 1
        low, high = split_walls_at_height(triangle[None], np.array([float(top)]))
        kept.append(high)
        for piece in low:
            kept.append(_without(piece, frame, gaps))
    kept = [k for k in kept if len(k)]
    return (np.concatenate(kept) if kept else np.zeros((0, 3, 3))), opened


def _wall_frame(triangle: np.ndarray):
    """``(origin, along, across)`` of a vertical triangle's plane; None if it
    is not vertical. Sloped faces are the roofs' business."""
    xy = triangle[:, :2]
    pairs = ((0, 1), (0, 2), (1, 2))
    lengths = [float(np.hypot(*(xy[j] - xy[i]))) for i, j in pairs]
    i, j = pairs[int(np.argmax(lengths))]
    length = max(lengths)
    if length < 1e-6:
        return None
    along = (xy[j] - xy[i]) / length
    across = np.array([-along[1], along[0]])
    offset = (xy - xy[i]) @ across
    if np.abs(offset).max() > 0.02:
        return None
    return xy[i] + float(offset.mean()) * across, along, across


def _gaps(frame, edge, span: float = 1e4) -> list[tuple[float, float]]:
    """The stretches of a wall's line on or in the passage, as distances along
    it. The line is drawn long; `_without` keeps to the part the wall covers."""
    import shapely

    origin, along, _ = frame
    line = shapely.LineString([origin - span * along, origin + span * along])
    out = []
    for piece in shapely.get_parts(line.intersection(edge)):
        if piece.geom_type != "LineString" or piece.length < GRAZE_M:
            continue
        ends = np.asarray(piece.coords)[[0, -1]]
        a, b = sorted(((ends - origin) @ along).tolist())
        # The margin taken back off, so the opening ends where the passage's
        # side wall starts rather than a slit past it.
        out.append((a + EDGE_M, b - EDGE_M))
    return out


def _without(triangle: np.ndarray, frame, gaps) -> np.ndarray:
    """One wall triangle with the stretches in `gaps` cut out of it.

    Clipped in the wall's own plane, (along, up), where the cuts are vertical
    lines; clipping keeps the corner order, so the face keeps its side.
    """
    origin, along, _ = frame
    s = (triangle[:, :2] - origin) @ along
    s0, s1 = float(s.min()), float(s.max())
    keep, cursor = [], s0
    for a, b in sorted(gaps):
        if b <= s0 or a >= s1:
            continue
        if a > cursor + 1e-4:
            keep.append((cursor, a))
        cursor = max(cursor, b)
    if s1 > cursor + 1e-4:
        keep.append((cursor, s1))

    flat = np.column_stack([s, triangle[:, 2]])
    out = []
    for a, b in keep:
        piece = _clip_convex(flat, 0, a, True)
        if len(piece) >= 3:
            piece = _clip_convex(piece, 0, b, False)
        if len(piece) < 3:
            continue
        lifted = np.column_stack([
            origin[0] + piece[:, 0] * along[0],
            origin[1] + piece[:, 0] * along[1],
            piece[:, 1],
        ])
        for k in range(1, len(lifted) - 1):
            out.append([lifted[0], lifted[k], lifted[k + 1]])
    out = np.asarray(out, dtype=np.float64).reshape(-1, 3, 3)
    if len(out):
        edge = np.cross(out[:, 1] - out[:, 0], out[:, 2] - out[:, 0])
        out = out[np.linalg.norm(edge, axis=1) > 1e-9]
    return out


def triangulate_polygon(polygon) -> np.ndarray:
    """Earcut one shapely polygon into ``(n, 3, 2)`` triangles."""
    import mapbox_earcut

    rings = [np.asarray(polygon.exterior.coords, dtype=np.float64)[:-1]]
    rings += [np.asarray(r.coords, dtype=np.float64)[:-1] for r in polygon.interiors]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return np.zeros((0, 3, 2))
    flat = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        indices = mapbox_earcut.triangulate_float64(flat, ends)
    except Exception:  # noqa: BLE001 - one bad outline
        return np.zeros((0, 3, 2))
    if len(indices) < 3:
        return np.zeros((0, 3, 2))
    return flat[np.asarray(indices, dtype=np.int64)].reshape(-1, 3, 2)


def cut_roofs(roofs: np.ndarray, region) -> np.ndarray:
    """The roofs with `region` cut out of them, seen from above. For the roofs
    below the ceiling: over the passage, they were the passage's ceiling."""
    import shapely

    roofs = np.asarray(roofs, dtype=np.float64)
    if not len(roofs) or region.is_empty:
        return roofs
    shapely.prepare(region)
    out = []
    for triangle in roofs:
        xy = triangle[:, :2]
        area = float(_cross2(xy[1] - xy[0], xy[2] - xy[0]))
        flat = shapely.Polygon(xy)
        if abs(area) < 1e-6 or not region.intersects(flat):
            out.append(triangle[None])
            continue
        rest = flat.difference(region)
        for polygon in _polygons(rest):
            pieces = triangulate_polygon(polygon)
            if not len(pieces):
                continue
            # Height from the triangle's own plane.
            a, b, c = triangle
            normal = np.cross(b - a, c - a)
            if abs(normal[2]) < 1e-12:
                continue
            z = a[2] - (
                normal[0] * (pieces[:, :, 0] - a[0]) + normal[1] * (pieces[:, :, 1] - a[1])
            ) / normal[2]
            lifted = np.dstack([pieces, z])
            # Earcut's winding is its own; face the way the roof faced.
            turn = _cross2(pieces[:, 1] - pieces[:, 0], pieces[:, 2] - pieces[:, 0])
            flip = np.sign(turn) != np.sign(area)
            lifted[flip] = lifted[flip][:, ::-1]
            out.append(lifted)
    out = [o for o in out if len(o)]
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def ceiling(region, z: float) -> np.ndarray:
    """The underside of the building over the passage, facing down."""
    out = []
    for polygon in _polygons(region):
        pieces = triangulate_polygon(polygon)
        if not len(pieces):
            continue
        turn = _cross2(pieces[:, 1] - pieces[:, 0], pieces[:, 2] - pieces[:, 0])
        # Facing down is clockwise from above.
        pieces[turn > 0] = pieces[turn > 0][:, ::-1]
        out.append(np.dstack([pieces, np.full(pieces.shape[:2], float(z))]))
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def lining(region, solid, heights: SurfaceHeights, *, top: float, bottom: float) -> np.ndarray:
    """The passage's side walls: wherever its edge runs through the building.

    Along an edge the building is on one side, the passage on the other, and
    the wall faces the passage. Where the edge is the building's own outline --
    a tunnel mouth, a passage along the facade -- there is no building on the
    far side and nothing is drawn. The top follows the building down where it
    is lower than the ceiling.
    """
    import shapely
    from shapely.geometry.polygon import orient

    shapely.prepare(solid)
    quads = []
    for polygon in _polygons(region):
        # Outline anticlockwise and holes clockwise: the passage on the left.
        polygon = orient(polygon, 1.0)
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = np.asarray(ring.coords, dtype=np.float64)
            for p, q in zip(coords[:-1], coords[1:]):
                length = float(np.hypot(*(q - p)))
                if length < 1e-3:
                    continue
                steps = max(1, int(np.ceil(length / LINING_STEP_M)))
                t = np.linspace(0.0, 1.0, steps + 1)
                points = p + t[:, None] * (q - p)
                right = np.array([q[1] - p[1], p[0] - q[0]]) / length
                middles = 0.5 * (points[:-1] + points[1:]) + 0.1 * right
                inside = shapely.contains_xy(solid, middles[:, 0], middles[:, 1])
                if not inside.any():
                    continue
                building_top = heights.lowest(middles[:, 0], middles[:, 1])
                for k in np.flatnonzero(inside):
                    upper = min(float(top), float(building_top[k])) if np.isfinite(building_top[k]) else float(top)
                    if upper <= bottom + 0.05:
                        continue
                    a, b = points[k], points[k + 1]
                    quads.append((a, b, upper))
    if not quads:
        return np.zeros((0, 3, 3))
    out = np.empty((2 * len(quads), 3, 3))
    for n, (a, b, upper) in enumerate(quads):
        a_low, b_low = [a[0], a[1], bottom], [b[0], b[1], bottom]
        a_high, b_high = [a[0], a[1], upper], [b[0], b[1], upper]
        # Facing left of a -> b, into the passage.
        out[2 * n] = [a_low, b_high, b_low]
        out[2 * n + 1] = [a_low, a_high, b_high]
    return out


def _sample_points(polygon, spacing_m: float = 1.0, cap: int = 3000) -> np.ndarray:
    import shapely

    step = max(spacing_m, float(np.sqrt(polygon.area / cap)))
    x0, y0, x1, y1 = polygon.bounds
    gx, gy = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step))
    gx, gy = gx.ravel(), gy.ravel()
    inside = shapely.contains_xy(polygon, gx, gy)
    points = np.column_stack([gx[inside], gy[inside]])
    if not len(points):
        centre = polygon.representative_point()
        points = np.array([[centre.x, centre.y]])
    return points


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------


def cut_passages(buildings, areas: list[PassageArea], terrain_sampler, cfg: dict | None = None) -> PassageStats:
    """Give every building a tunnel where a road or water runs under it.

    Changes the buildings in place: walls and roofs are cut, and the lining
    goes into `Building.passage_tris`. Has to run after the ground skirt, which
    sets where the walls reach down to, and before the ground storey is split
    off, which would otherwise be cut twice.
    """
    import shapely

    settings = {**DEFAULTS, **(cfg or {})}
    clearance = {**DEFAULTS["clearance_m"], **(settings.get("clearance_m") or {})}
    stats = PassageStats()
    if not areas or not bool(settings.get("enabled", True)):
        return stats

    polygons = np.array([a.polygon for a in areas], dtype=object)
    kinds = [a.kind for a in areas]
    tree = shapely.STRtree(polygons)
    radius = 0.5 * float(settings["min_width_m"])
    min_area = float(settings["min_area_m2"])
    slab = float(settings["slab_m"])
    headroom = float(settings["min_headroom_m"])

    for building in buildings.buildings:
        if not len(building.roof_tris) or not len(building.all_wall_tris):
            continue
        lo = building.roof_tris[:, :, :2].reshape(-1, 2).min(axis=0)
        hi = building.roof_tris[:, :, :2].reshape(-1, 2).max(axis=0)
        candidates = tree.query(shapely.box(lo[0], lo[1], hi[0], hi[1]))
        if not len(candidates):
            continue
        solid = footprint(building.roof_tris)
        if solid.is_empty:
            continue
        shapely.prepare(solid)
        touching = [int(i) for i in candidates if solid.intersects(polygons[i])]
        if not touching:
            continue
        under = {}
        for index in touching:
            piece = polygons[index].intersection(solid)
            if not piece.is_empty and piece.area > 1e-3:
                under.setdefault(kinds[index], []).append(piece)
        if not under:
            continue
        by_kind = {k: shapely.union_all(np.array(v, dtype=object)) for k, v in under.items()}
        everything = shapely.union_all(np.array(list(by_kind.values()), dtype=object))
        passages = [
            p for p in _polygons(opening(everything, radius))
            if p.area >= min_area
        ]
        if not passages:
            continue

        heights = SurfaceHeights(building.roof_tris)
        base = float(building.all_wall_tris[:, :, 2].min())
        made = 0
        # Largest first, so the cut a later one makes cannot starve it.
        for region in sorted(passages, key=lambda p: -p.area):
            points = _sample_points(region)
            ground_z = np.asarray(terrain_sampler(points[:, 0], points[:, 1]), dtype=np.float64)
            ground_z = ground_z[np.isfinite(ground_z)]
            ground = float(np.percentile(ground_z, 90)) if len(ground_z) else building.ground_z_nap
            roof = heights.lowest(points[:, 0], points[:, 1])
            roof = roof[np.isfinite(roof)]
            if not len(roof):
                continue
            passing = [
                k for k, shape in by_kind.items()
                if shape.intersection(region).area > 0.25 * min_area
            ] or list(by_kind)
            if set(passing) <= {PAVED}:
                continue
            wanted = max(float(clearance.get(k, clearance[ROAD])) for k in passing)
            top = min(ground + wanted, float(np.percentile(roof, 10)) - slab)
            if top - ground < headroom:
                stats.too_low += 1
                continue
            _cut(building, region, solid, heights, top=top, bottom=base)
            made += 1
            stats.passages += 1
            stats.area_m2 += float(region.area)
            for k in passing:
                stats.by_kind[k] = stats.by_kind.get(k, 0) + 1
            stats.cut.append((building.identifier, region, float(top), float(ground)))
        if made:
            stats.buildings += 1

    if stats.passages or stats.too_low:
        LOG.info(
            "passages: %d under %d buildings, %.0f m2 (%s); %d left shut, too low to pass under",
            stats.passages, stats.buildings, stats.area_m2,
            ", ".join(f"{v} {k}" for k, v in sorted(stats.by_kind.items())) or "none",
            stats.too_low,
        )
    return stats


def _cut(building, region, solid, heights: SurfaceHeights, *, top: float, bottom: float) -> None:
    """Open one passage through one building, up to `top`."""
    import shapely

    building.wall_tris, _ = open_walls(building.wall_tris, region, top)

    # Roofs below the ceiling over the passage are cut away: the passage is
    # open to the sky there, and no ceiling is drawn under it.
    roofs = building.roof_tris
    over = _near_mask(roofs, region)
    if over.any():
        flat = np.abs(_cross2(roofs[:, 1, :2] - roofs[:, 0, :2], roofs[:, 2, :2] - roofs[:, 0, :2])) > 1e-6
        candidates = np.flatnonzero(over & flat & (roofs[:, :, 2].min(axis=1) < top))
        if len(candidates):
            hit = shapely.intersects(region, shapely.polygons(roofs[candidates, :, :2]))
            over = np.zeros(len(roofs), dtype=bool)
            over[candidates[hit]] = True
        else:
            over[:] = False
    sky = shapely.Polygon()
    if over.any():
        low, high = split_at(roofs[over], top)
        if len(low):
            xy = low[:, :, :2]
            flat = np.abs(_cross2(xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0])) > 1e-6
            if flat.any():
                sky = shapely.union_all(shapely.polygons(xy[flat])).intersection(region)
        building.roof_tris = _stack([roofs[~over], cut_roofs(low, region), high])

    lid = region.difference(sky) if not sky.is_empty else region
    building.passage_tris = _stack([
        building.passage_tris,
        ceiling(lid, top),
        lining(region, solid, heights, top=top, bottom=bottom),
    ])


def blocked(building, region, top: float) -> tuple[int, int]:
    """What of a building still stands in one of its passages, below `top`:
    ``(wall triangles, roof points)``.

    A wall is in the way where it runs through the passage or along its edge
    where the building's outline is -- across a tunnel mouth, the way
    `open_walls` reads it. A roof is in the way where it is lower than the
    ceiling over a point of the passage. Both should be none once
    `cut_passages` has been.
    """
    import shapely

    edge = region.buffer(EDGE_M, join_style="mitre", mitre_limit=4.0)
    shapely.prepare(edge)
    walls = building.all_wall_tris
    walls = walls[_near_mask(walls, edge)] if len(walls) else walls
    standing = 0
    for triangle in walls:
        low = triangle[:, 2] < top - 0.05
        frame = _wall_frame(triangle)
        if low.sum() < 2 or frame is None:
            continue
        origin, along, _ = frame
        s = (triangle[:, :2] - origin) @ along
        span = shapely.LineString([origin + s[low].min() * along, origin + s[low].max() * along])
        piece = span.intersection(edge)
        if not piece.is_empty and piece.length - 2 * EDGE_M > GRAZE_M:
            standing += 1
    inner = region.buffer(-0.2, join_style="mitre", mitre_limit=4.0)
    if inner.is_empty:
        return standing, 0
    points = _sample_points(inner)
    surfaces = _stack([building.roof_tris, building.passage_tris])
    lowest = SurfaceHeights(surfaces).lowest(points[:, 0], points[:, 1]) if len(surfaces) else np.full(len(points), np.nan)
    low_roof = int((np.isfinite(lowest) & (lowest < top - 0.05)).sum())
    return standing, low_roof


def _near_mask(triangles: np.ndarray, geometry) -> np.ndarray:
    if not len(triangles):
        return np.zeros(0, dtype=bool)
    x0, y0, x1, y1 = geometry.bounds
    lo, hi = triangles[:, :, :2].min(axis=1), triangles[:, :, :2].max(axis=1)
    return (hi[:, 0] >= x0) & (lo[:, 0] <= x1) & (hi[:, 1] >= y0) & (lo[:, 1] <= y1)


def _stack(chunks) -> np.ndarray:
    chunks = [np.asarray(c, dtype=np.float64).reshape(-1, 3, 3) for c in chunks if len(c)]
    return np.concatenate(chunks) if chunks else np.zeros((0, 3, 3), dtype=np.float64)


__all__ = [
    "DEFAULTS",
    "KINDS",
    "PassageArea",
    "PassageStats",
    "SurfaceHeights",
    "blocked",
    "ceiling",
    "cut_passages",
    "cut_roofs",
    "fetch_passage_areas",
    "footprint",
    "lining",
    "open_walls",
    "opening",
    "passage_kind",
]
