"""Road surfaces as a mesh an editor can work with.

The BGT describes a street as a mosaic of surfaces: carriageway, parking bays,
cycle path, footway. Triangulating that mosaic directly gives a mesh with
vertices scattered through the middle of every surface and no structure an
editor can grab -- 23% of the road vertices over Bronbeek sat inside a
surface rather than on a kerb, and the valence ran from 4 to 11.

Road tools build roads the other way round: a centreline, and a cross-section
swept along it into strips of quads, with junctions as separate patches. This
module does that with the data that exists. The kerbs come from the BGT and
are never moved; the centrelines come from NWB (see `nwb`) and decide only
where the cross-sections are cut.

The steps, per level (a road on a bridge is meshed apart from the one under
it):

1. The BGT surfaces become a clean partition: overlaps resolved in paint
   order, shared kerbs made identical, and survey noise taken off the
   outlines without moving any kerb by more than `simplify_m`.
2. Along every centreline, stations are placed wherever a kerb has a corner
   and at least every `station_m` in between, and pulled back from each
   junction by the width of the roads meeting there.
3. At each station a line is cut straight across the street, through every
   surface that belongs to this street and no further: it stops at the first
   gap, and where a surface is nearer another centreline. Where it crosses a
   kerb it snaps onto any kerb vertex within `SNAP_M`.
4. Cuts that would cross each other are dropped, nearest the junction first.
5. The partition is split along the cuts. Between two cuts every surface
   becomes one quad; what is left -- junctions, and surfaces no centreline
   reaches -- stays one face per piece, bounded by its own outline.

So every vertex lies on a kerb or a strip edge, every cross-section is an edge
loop across the whole street, and every kerb is an edge loop along it. The
footway is raised by `kerb_m` with a vertical kerb face along every edge where
it meets a lower surface, and the outer edge of the whole network carries a
short skirt down into the ground so no gap opens against the terrain.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

import numpy as np

from .surfaces import (
    CLASS_CYCLE,
    CLASS_FOOTPATH,
    CLASS_KERB,
    PAINT_ORDER,
    RoadPart,
    _hold_above_water,
    _robust_deck_level,
)

LOG = logging.getLogger(__name__)

# Spacing of cross-sections along a straight street. Close enough that the
# road follows the ground between them -- the terrain grid is 2 m -- and far
# enough apart that a straight street is a handful of quads, not a comb.
STATION_M = 3.0
# Two cross-sections closer than this are one too many.
MIN_STATION_GAP_M = 1.0
# Road the national network does not reach -- an estate's own lanes, a path
# through a park -- gets a centreline of its own, drawn down the middle of its
# outline. Only where it is a strip: wider than this, it is a square or a car
# park, and has no "along" to cut across.
SKELETON_MAX_WIDTH_M = 16.0
# Road within this distance of an NWB centreline is that centreline's.
NWB_COVER_M = 10.0
# How far a cross-section reaches either side of its centreline.
REACH_M = 25.0
# A cut crossing a kerb this close to one of its vertices goes through the
# vertex instead, which is what keeps slivers out of the mesh.
SNAP_M = 0.30
# How far a kerb may move when the survey's noise is taken off the outlines.
SIMPLIFY_M = 0.15
# A face bigger than this with no street to cut it is cut into a grid of this
# spacing, so it follows the ground. A junction of two ordinary streets is
# well under it; a car park or a square is not.
GRID_FACE_M2 = 150.0
GRID_M = 4.0
# How far a kerb may move again when a run of it between two cross-sections is
# straightened. Together with SIMPLIFY_M this is the 25 cm the outline is
# allowed to give up.
STRAIGHTEN_M = 0.10
# A kerb vertex that turns by more than this is a corner, and gets a
# cross-section of its own.
CORNER_DEG = 20.0
# The road surface floats this far above the surveyed ground.
LIFT_M = 0.03
# Height of the kerb between a footway and whatever it meets. Dutch kerbs run
# 8 to 15 cm.
KERB_M = 0.10
# How far the skirt along the network's outer edge reaches down.
SKIRT_M = 0.35
# How far the terrain is held below the lowest road over it. Five centimetres
# plus the lift is enough depth separation at drone height for a 24-bit depth
# buffer with Unity's default near plane, out to about half a kilometre.
TERRAIN_CLEARANCE_M = 0.05
# Surfaces that sit on top of the kerb rather than at carriageway level.
RAISED_CLASSES = (CLASS_FOOTPATH,)

# Face kinds.
SURFACE, KERB, SKIRT = 0, 1, 2


@dataclass
class RoadMesh:
    """The road network as faces, ready for the Blender stage."""

    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    face_sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    face_loops: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    face_class: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    face_level: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    face_object: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    face_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    # Per loop, in metres: across and along the centreline for a street, plain
    # x and y for everything else, along and up for a kerb.
    loop_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    object_names: list[str] = field(default_factory=list)
    object_kinds: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.face_sizes))

    @property
    def loop_starts(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.face_sizes)[:-1]]).astype(np.int64)

    def face(self, index: int) -> np.ndarray:
        start = int(self.loop_starts[index])
        return self.face_loops[start : start + int(self.face_sizes[index])]

    def triangles(self, kinds=(SURFACE,)) -> tuple[np.ndarray, np.ndarray]:
        """Every face of these kinds as triangles, and which face each came from.

        Earcut per face, because a face can be a concave n-gon. Only for the
        things that need triangles -- the height rasters and the checks; the
        mesh itself keeps its quads.
        """
        import mapbox_earcut

        out: list[np.ndarray] = []
        owner: list[np.ndarray] = []
        starts = self.loop_starts
        wanted = np.isin(self.face_kind, np.asarray(kinds))
        for index in np.flatnonzero(wanted):
            loop = self.face_loops[starts[index] : starts[index] + self.face_sizes[index]]
            points = self.vertices[loop]
            if len(loop) == 3:
                tris = np.array([[0, 1, 2]])
            elif len(loop) == 4 and self.face_kind[index] == SURFACE:
                # The same split the export makes: first corner to third.
                tris = np.array([[0, 1, 2], [0, 2, 3]])
            elif self.face_kind[index] != SURFACE:
                tris = np.array([[0, 1, 2], [0, 2, 3]])
            else:
                flat = points[:, :2].astype(np.float64)
                try:
                    found = mapbox_earcut.triangulate_float64(
                        flat, np.array([len(flat)], dtype=np.uint32)
                    )
                except Exception:  # noqa: BLE001 - a bad face is skipped, not fatal
                    continue
                if len(found) < 3:
                    continue
                tris = np.asarray(found, dtype=np.int64).reshape(-1, 3)
            out.append(points[tris])
            owner.append(np.full(len(tris), index, dtype=np.int64))
        if not out:
            return np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int64)
        return np.concatenate(out), np.concatenate(owner)

    def to_npz_fields(self) -> dict:
        return {
            "road_vertices": self.vertices,
            "road_face_sizes": self.face_sizes.astype(np.int32),
            "road_face_loops": self.face_loops.astype(np.int64),
            "road_face_class": self.face_class.astype(np.int32),
            "road_face_level": self.face_level.astype(np.int32),
            "road_face_object": self.face_object.astype(np.int32),
            "road_face_kind": self.face_kind.astype(np.int8),
            "road_loop_uv": self.loop_uv.astype(np.float32),
            # Plain strings: Blender's NumPy cannot unpickle an object array.
            "road_object_names": np.array(self.object_names, dtype=str),
            "road_object_kinds": np.array(self.object_kinds, dtype=str),
        }


# ---------------------------------------------------------------------------
# Planar part: the partition, the cuts, the faces
# ---------------------------------------------------------------------------


def _paint_rank(code: int) -> int:
    return PAINT_ORDER.index(code) if code in PAINT_ORDER else -1


def _polygons_of(geometry) -> list:
    import shapely

    out = []
    for part in shapely.get_parts(geometry):
        if part.geom_type == "Polygon" and not part.is_empty and part.area > 1e-4:
            out.append(part)
        elif part.geom_type in ("MultiPolygon", "GeometryCollection"):
            out.extend(_polygons_of(part))
    return out


def clean_partition(parts: list[RoadPart], simplify_m: float = SIMPLIFY_M) -> list[tuple]:
    """Road parts as non-overlapping polygons with identical shared kerbs.

    Returns ``(polygon, surface class, level, part index)`` tuples. Overlaps
    are given to the surface that paints last, as in the land-cover raster: a
    parking bay cut out of a carriageway belongs to the bay.
    """
    import shapely
    from shapely import Polygon

    items: list[list] = []
    for index, part in enumerate(parts):
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in part.rings]
        rings = [r for r in rings if len(r) >= 4]
        if not rings:
            continue
        try:
            polygon = shapely.make_valid(Polygon(rings[0], rings[1:]))
        except Exception:  # noqa: BLE001 - one bad outline is not fatal
            continue
        for piece in _polygons_of(polygon):
            items.append([piece, int(part.surface_class), int(part.level), index])
    if not items:
        return []

    # Winners first; within a class, larger parts first so a sliver of overlap
    # is taken from the small one.
    items.sort(key=lambda it: (-_paint_rank(it[1]), -it[0].area))
    placed: list = []
    placed_bounds = np.zeros((0, 4))
    kept: list[list] = []
    for item in items:
        geometry = item[0]
        if len(placed):
            b = geometry.bounds
            near = np.flatnonzero(
                (placed_bounds[:, 0] <= b[2]) & (placed_bounds[:, 2] >= b[0])
                & (placed_bounds[:, 1] <= b[3]) & (placed_bounds[:, 3] >= b[1])
            )
            if len(near):
                geometry = geometry.difference(
                    shapely.union_all([placed[i] for i in near])
                )
        for piece in _polygons_of(geometry):
            kept.append([piece, item[1], item[2], item[3]])
            placed.append(piece)
            placed_bounds = np.vstack([placed_bounds, piece.bounds])

    geometries = np.array([k[0] for k in kept], dtype=object)
    # Shared kerbs made vertex-for-vertex identical, and the hairline gaps the
    # survey leaves between neighbouring surfaces closed.
    if hasattr(shapely, "coverage_clean"):
        geometries = shapely.coverage_clean(geometries, gap_width=0.05)
    else:  # Shapely 2.1: snapping only.
        geometries = shapely.set_precision(geometries, 0.005)
    if simplify_m > 0:
        geometries = shapely.coverage_simplify(geometries, simplify_m)

    out = []
    for geometry, (_, code, level, index) in zip(geometries, kept):
        if geometry is None or geometry.is_empty:
            continue
        for piece in _polygons_of(geometry):
            out.append((piece, code, level, index))
    return out


class _VertexHash:
    """Nearest kerb vertex to a point, within a small radius."""

    def __init__(self, points: np.ndarray, cell: float):
        self.points = np.asarray(points, dtype=np.float64)
        self.cell = cell
        self.buckets: dict[tuple[int, int], list[int]] = {}
        keys = np.floor(self.points / cell).astype(np.int64)
        for i, (kx, ky) in enumerate(keys):
            self.buckets.setdefault((int(kx), int(ky)), []).append(i)

    def nearest(self, x: float, y: float, radius: float) -> int | None:
        kx, ky = int(math.floor(x / self.cell)), int(math.floor(y / self.cell))
        best, best_d = None, radius
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in self.buckets.get((kx + dx, ky + dy), ()):
                    d = math.hypot(self.points[i, 0] - x, self.points[i, 1] - y)
                    if d <= best_d:
                        best, best_d = i, d
        return best


def _frame(line, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Point and left normal on a centreline at distance `t` along it.

    The tangent is taken over two metres rather than from the one segment the
    point falls on, so a cross-section at a polyline vertex is cut square to
    the street rather than square to whichever segment won the tie.
    """
    length = line.length
    here = line.interpolate(t)
    ahead = line.interpolate(min(length, t + 1.0))
    behind = line.interpolate(max(0.0, t - 1.0))
    d = np.array([ahead.x - behind.x, ahead.y - behind.y])
    norm = float(np.hypot(*d))
    if norm < 1e-9:
        d, norm = np.array([1.0, 0.0]), 1.0
    d /= norm
    return np.array([here.x, here.y]), np.array([-d[1], d[0]])


def _cut_pieces(stations, polygons, poly_tree, lines, line_tree, reach):
    """Where each station's cross-section runs through road, in pieces.

    Returns, per station, a sorted list of ``(s0, s1, owner, own_margin)``
    where `s` is signed distance from the centreline and each piece lies in
    one surface polygon.
    """
    import shapely

    if not stations:
        return []
    starts = np.array([p - reach * n for _, _, p, n in stations])
    ends = np.array([p + reach * n for _, _, p, n in stations])
    segments = shapely.linestrings(np.stack([starts, ends], axis=1))
    pairs = poly_tree.query(segments, predicate="intersects")
    pieces = shapely.intersection(segments[pairs[0]], polygons[pairs[1]])

    rows: list[tuple[int, float, float, np.ndarray]] = []
    for k, piece in enumerate(pieces):
        station = int(pairs[0][k])
        _, _, p, n = stations[station]
        for part in shapely.get_parts(piece):
            if part.geom_type != "LineString" or part.length < 0.02:
                continue
            coords = np.asarray(part.coords)[:, :2]
            s = (coords - p) @ n
            rows.append((station, float(s.min()), float(s.max()), coords.mean(axis=0)))

    per_station: list[list] = [[] for _ in stations]
    if not rows:
        return per_station
    mids = shapely.points(np.array([r[3] for r in rows]))
    nearest, _ = line_tree.query_nearest(mids, return_distance=True, all_matches=False)
    owner_of = np.empty(len(rows), dtype=np.int64)
    owner_of[nearest[0]] = nearest[1]
    own_lines = np.array([lines[stations[r[0]][0]] for r in rows], dtype=object)
    own_distance = shapely.distance(mids, own_lines)
    other_distance = shapely.distance(mids, lines[owner_of])
    for k, (station, s0, s1, _) in enumerate(rows):
        per_station[station].append(
            (s0, s1, int(owner_of[k]), float(own_distance[k] - other_distance[k]))
        )
    for entry in per_station:
        entry.sort()
    return per_station


# A cross-section carries on across a gap in the road this narrow -- a grass
# verge or a row of tree pits between the carriageway and the footway -- so the
# footway beyond it is cut with the rest of the street rather than left whole.
MAX_VERGE_M = 6.0


def _run_through_centre(
    pieces, line_index: int, *, gap: float = MAX_VERGE_M, approach: float = 3.0
):
    """The crossings of the run of road this street owns, across its verges.

    Starts from the piece under the centreline -- or the nearest one within
    `approach`, since a centreline can sit a metre off its carriageway -- and
    walks outward until the road stops for longer than `gap` or a surface
    belongs to another street.
    """
    if not pieces:
        return []
    centre = None
    best = approach
    for i, (s0, s1, _, _) in enumerate(pieces):
        d = 0.0 if s0 <= 0.0 <= s1 else min(abs(s0), abs(s1))
        if d <= best:
            best, centre = d, i
    if centre is None:
        return []

    def mine(piece) -> bool:
        _, _, owner, margin = piece
        return owner == line_index or margin <= 0.5

    if not mine(pieces[centre]):
        return []
    lo = hi = centre
    while lo > 0 and pieces[lo - 1][1] >= pieces[lo][0] - gap and mine(pieces[lo - 1]):
        lo -= 1
    while (
        hi < len(pieces) - 1
        and pieces[hi + 1][0] <= pieces[hi][1] + gap
        and mine(pieces[hi + 1])
    ):
        hi += 1
    crossings = sorted({round(v, 4) for s0, s1, _, _ in pieces[lo : hi + 1] for v in (s0, s1)})
    # Two crossings closer than this are one kerb seen twice.
    merged: list[float] = []
    for value in crossings:
        if not merged or value - merged[-1] > 0.02:
            merged.append(value)
    return merged


def _cut_geometry(p, n, crossings, snapper: _VertexHash | None):
    """The cut as a polyline through its crossings, snapped onto kerb vertices."""
    import shapely

    points = [p + s * n for s in crossings]
    if snapper is not None:
        for i, q in enumerate(points):
            hit = snapper.nearest(q[0], q[1], SNAP_M)
            if hit is not None:
                points[i] = snapper.points[hit].copy()
    clean = [points[0]]
    for q in points[1:]:
        if np.hypot(*(q - clean[-1])) > 1e-3:
            clean.append(q)
    if len(clean) < 2:
        return None
    # A hair past each end, so the cut certainly crosses the kerb it ends on.
    # The overhang has nothing on its far side and is dropped as a dangle.
    first = clean[0] - 0.01 * _unit(clean[1] - clean[0])
    last = clean[-1] + 0.01 * _unit(clean[-1] - clean[-2])
    return shapely.linestrings(np.vstack([first, *clean, last]))


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.hypot(*v))
    return v / norm if norm > 1e-12 else np.zeros(2)


@dataclass
class _Plan:
    """What the planar stage decided, for the 3D stage to build on."""

    faces: list = field(default_factory=list)  # (coords (k,2) ccw, class, level, part, owner key)
    lines: list = field(default_factory=list)  # shapely centrelines
    line_meta: list = field(default_factory=list)  # Centreline per line
    # How far the stations stop short of each end of each line, in line order.
    setbacks: list = field(default_factory=list)
    # Lines whose middle lies on a footway or cycle path.
    path_lines: set = field(default_factory=set)
    # Half the carriageway width at each junction, from trial cuts.
    half_width_at: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def plan_faces(
    partition: list[tuple],
    network,
    *,
    station_m: float = STATION_M,
    reach_m: float = REACH_M,
    skeleton: bool = True,
    group_start: int = 1,
) -> _Plan:
    """Cut the partition across every street and return its faces.

    The national centrelines go first and claim their streets. Whatever road
    they leave without a cross-section -- an estate's own lanes, the paths
    through a park -- then gets centrelines drawn down its own middle (see
    `skeleton_centrelines`), and is cut the same way.
    """
    import shapely
    from shapely import LineString, STRtree

    plan = _Plan()
    if not partition:
        return plan
    polygons = np.array([item[0] for item in partition], dtype=object)
    poly_tree = STRtree(polygons)
    road = shapely.union_all(polygons)
    bounds = road.bounds

    # Centrelines anywhere near the road.
    for line in getattr(network, "lines", []) or []:
        geometry = LineString(line.points)
        if geometry.length < 1.0:
            continue
        b = geometry.bounds
        if b[2] < bounds[0] - reach_m or b[0] > bounds[2] + reach_m:
            continue
        if b[3] < bounds[1] - reach_m or b[1] > bounds[3] + reach_m:
            continue
        plan.lines.append(geometry)
        plan.line_meta.append(line)

    rings = [
        np.asarray(ring.coords)[:-1, :2]
        for poly in polygons
        for ring in [poly.exterior, *poly.interiors]
    ]
    snapper = _VertexHash(np.vstack(rings), SNAP_M)
    # The kerb vertices that are corners rather than points on a curve. A
    # station at every vertex of a rounded kerb is a comb of cross-sections;
    # the regular spacing and the straightening below handle a curve.
    corners = np.vstack([ring[_turning(ring) > CORNER_DEG] for ring in rings])
    context = (polygons, poly_tree, snapper, corners, station_m, reach_m)

    national = list(range(len(plan.lines)))
    cuts = _accept(_candidate_cuts(plan, national, context), [])
    plan.stats["national_cuts"] = len(cuts)

    if skeleton:
        reached = _laddered_region(plan, cuts)
        unreached = road.difference(reached) if reached is not None else road
        # A metre-wide fringe left along a laddered street is the street's
        # kerb seen at an angle, not a road of its own.
        unreached = unreached.buffer(-0.6).buffer(0.6).intersection(road)
        own = skeleton_centrelines(unreached, group_start=group_start)
        for line in own:
            plan.lines.append(LineString(line.points))
            plan.line_meta.append(line)
        plan.stats["skeleton_centrelines"] = len(own)
        drawn = list(range(len(national), len(plan.lines)))
        if drawn:
            cuts = _accept(_candidate_cuts(plan, drawn, context), cuts)

    if plan.lines:
        middles = shapely.line_interpolate_point(
            np.array(plan.lines, dtype=object), 0.5, normalized=True
        )
        on = poly_tree.query(middles, predicate="within")
        for li, pi in zip(*on):
            if partition[pi][1] in (CLASS_FOOTPATH, CLASS_CYCLE):
                plan.path_lines.add(int(li))
    plan.stats["cuts"] = len(cuts)

    # Split the partition along the cuts.
    linework = [poly.boundary for poly in polygons] + [c[3] for c in cuts]
    faces, face_item = _split(linework, poly_tree)

    # A face with no street to cut it across -- a car park, a square, the
    # ballast strip of a railway -- is flat between its outline points, and
    # over tens of metres the ground under it is not. Cut those into a grid,
    # so they follow the ground in rows and columns rather than forcing the
    # terrain down to get under them.
    big = [faces[i] for i in np.flatnonzero(face_item >= 0) if faces[i].area > GRID_FACE_M2]
    if big:
        grid = [line for face in big for line in _grid_lines(face, GRID_M)]
        if grid:
            faces, face_item = _split(linework + grid, poly_tree)
    plan.stats["gridded_faces"] = len(big)

    for index in np.flatnonzero(face_item >= 0):
        face = faces[index]
        if face.area < 1e-4:
            continue
        polygon, code, level, part = partition[face_item[index]]
        for piece in _without_holes(face):
            coords = np.asarray(shapely.orient_polygons(piece).exterior.coords)[:-1, :2]
            coords = _distinct_corners(coords)
            if len(coords) >= 3:
                plan.faces.append((coords, code, level, part))
    before = sum(len(f[0]) for f in plan.faces)
    plan.faces = _straighten_runs(plan.faces, STRAIGHTEN_M)
    plan.stats["faces"] = len(plan.faces)
    plan.stats["vertices_straightened_away"] = before - sum(len(f[0]) for f in plan.faces)
    return plan


def _split(linework: list, poly_tree):
    """Faces of the arrangement of `linework`, and the surface each lies in."""
    import shapely

    noded = shapely.union_all(np.array(linework, dtype=object))
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(noded)))
    reps = shapely.point_on_surface(faces)
    where = poly_tree.query(reps, predicate="within")
    face_item = np.full(len(faces), -1, dtype=np.int64)
    face_item[where[0]] = where[1]
    return faces, face_item


def _grid_lines(face, step: float) -> list:
    """A grid of lines across a face, square to its longest side, clipped to it."""
    import shapely

    rectangle = np.asarray(shapely.minimum_rotated_rectangle(face).exterior.coords)[:4]
    sides = np.diff(np.vstack([rectangle, rectangle[:1]]), axis=0)
    longest = sides[int(np.argmax(np.hypot(sides[:, 0], sides[:, 1])))]
    u = longest / max(float(np.hypot(*longest)), 1e-9)
    v = np.array([-u[1], u[0]])
    coords = np.asarray(face.exterior.coords)[:, :2]
    pu, pv = coords @ u, coords @ v
    lines = []
    for axis, other, lo, hi, olo, ohi in (
        (u, v, pu.min(), pu.max(), pv.min(), pv.max()),
        (v, u, pv.min(), pv.max(), pu.min(), pu.max()),
    ):
        count = int((hi - lo) // step)
        if count < 1:
            continue
        # Centred, so the leftover is split between the two edges.
        first = lo + 0.5 * ((hi - lo) - count * step)
        for k in range(count + 1):
            at = first + k * step
            if at <= lo + 0.5 or at >= hi - 0.5:
                continue
            a = axis * at + other * (olo - 1.0)
            b = axis * at + other * (ohi + 1.0)
            clipped = shapely.intersection(shapely.LineString([a, b]), face)
            lines.extend(
                part for part in shapely.get_parts(clipped)
                if part.geom_type == "LineString" and part.length > 0.5
            )
    return lines


def _candidate_cuts(plan: _Plan, which: list[int], context) -> list[tuple]:
    """Stations and their cross-sections, for the centrelines in `which`.

    Ownership is judged against every centreline the plan holds, so a cut
    from one street stops where the surface belongs to another.
    """
    import shapely
    from shapely import STRtree

    polygons, poly_tree, snapper, corners, station_m, reach_m = context
    if not which:
        return []
    lines = np.array(plan.lines, dtype=object)
    line_tree = STRtree(lines)
    while len(plan.setbacks) < len(plan.lines):
        plan.setbacks.append((0.0, 0.0))

    degree: dict[int, int] = {}
    for meta in plan.line_meta:
        for junction in (meta.start_junction, meta.end_junction):
            degree[junction] = degree.get(junction, 0) + 1

    # Width of each street near each end, from a trial cut, so the stations
    # can be pulled back from a junction by the width of the roads meeting
    # there.
    trial = []
    trial_junction = []
    for li in which:
        geometry, meta = plan.lines[li], plan.line_meta[li]
        ends = (
            (min(8.0, geometry.length / 2), meta.start_junction),
            (max(geometry.length - 8.0, geometry.length / 2), meta.end_junction),
        )
        for t, junction in ends:
            p, n = _frame(geometry, t)
            trial.append((li, t, p, n))
            trial_junction.append(junction)
    # How far each street reaches either side of its centreline near each
    # junction: the whole street, footways and all, not just the carriageway.
    # A street stops short of a junction by the reach of the streets crossing
    # it there, so none of its cuts runs into the corner footways.
    reach_at = dict(plan.half_width_at)
    for (li, _, _, _), junction, pieces in zip(
        trial, trial_junction,
        _cut_pieces(trial, polygons, poly_tree, lines, line_tree, reach_m),
    ):
        crossings = _run_through_centre(pieces, li)
        extent = max((abs(c) for c in crossings), default=3.0)
        reach_at[(junction, li)] = max(reach_at.get((junction, li), 0.0), extent)
    plan.half_width_at = reach_at

    def setback(junction: int, li: int) -> float:
        if degree.get(junction, 0) < 3:
            return 0.0
        others = [
            extent for (j, other), extent in reach_at.items()
            if j == junction and other != li
        ]
        return float(np.clip(max(others, default=3.0) + 1.0, 3.0, 25.0))

    # Kerb corners by the street they belong to, for stations at corners.
    vertex_owner = np.full(len(corners), -1, dtype=np.int64)
    own_distance = np.full(len(corners), np.inf)
    if len(corners):
        corner_points = shapely.points(corners)
        owners, _ = line_tree.query_nearest(
            corner_points, return_distance=True, all_matches=False
        )
        vertex_owner[owners[0]] = owners[1]
        own_distance = shapely.distance(corner_points, lines[vertex_owner])

    stations = []  # (line, t, p, n)
    station_rank = []  # distance from the nearest junction end
    for li in which:
        geometry, meta = plan.lines[li], plan.line_meta[li]
        length = geometry.length
        lo = setback(meta.start_junction, li)
        hi = length - setback(meta.end_junction, li)
        plan.setbacks[li] = (lo, length - hi)
        if hi - lo < MIN_STATION_GAP_M:
            continue
        mine = np.flatnonzero((vertex_owner == li) & (own_distance <= reach_m))
        corner_t = shapely.line_locate_point(geometry, shapely.points(corners[mine]))
        corner_t = np.sort(corner_t[(corner_t > lo) & (corner_t < hi)])
        chosen = [lo]
        for t in corner_t:
            if t - chosen[-1] >= MIN_STATION_GAP_M and hi - t >= MIN_STATION_GAP_M:
                chosen.append(float(t))
        chosen.append(hi)
        filled = [chosen[0]]
        for a, b in zip(chosen[:-1], chosen[1:]):
            extra = int(math.ceil((b - a) / station_m)) - 1
            for k in range(1, extra + 1):
                filled.append(a + (b - a) * k / (extra + 1))
            filled.append(b)
        for t in filled:
            p, n = _frame(geometry, t)
            stations.append((li, float(t), p, n))
            station_rank.append(min(t, length - t))

    candidates = []
    pieces_of = _cut_pieces(stations, polygons, poly_tree, lines, line_tree, reach_m)
    for k, ((li, t, p, n), pieces) in enumerate(zip(stations, pieces_of)):
        crossings = _run_through_centre(pieces, li)
        if len(crossings) < 2:
            continue
        geometry = _cut_geometry(p, n, crossings, snapper)
        if geometry is None:
            continue
        candidates.append((station_rank[k], li, t, geometry))
    plan.stats["stations"] = plan.stats.get("stations", 0) + len(stations)
    return candidates


def _accept(candidates: list[tuple], accepted: list[tuple]) -> list[tuple]:
    """Keep the candidates that cross nothing already kept.

    Furthest from a junction first, so where two cross it is the one nearer
    the junction that goes. Cuts already accepted are never displaced.
    """
    from shapely import STRtree

    if not candidates:
        return list(accepted)
    candidates = sorted(candidates, key=lambda c: -c[0])
    everything = list(accepted) + candidates
    geoms = np.array([c[3] for c in everything], dtype=object)
    keep = np.zeros(len(everything), dtype=bool)
    keep[: len(accepted)] = True
    partners: dict[int, list[int]] = {}
    hits = STRtree(geoms).query(geoms, predicate="crosses")
    for a, b in zip(*hits):
        if a != b:
            partners.setdefault(int(a), []).append(int(b))
    for i in range(len(accepted), len(everything)):
        if not any(keep[j] for j in partners.get(i, ())):
            keep[i] = True
    return [everything[i] for i in np.flatnonzero(keep)]


def _laddered_region(plan: _Plan, cuts: list[tuple]):
    """The road already spanned by consecutive cross-sections, plus junctions."""
    import shapely
    from shapely import Polygon

    by_line: dict[int, list[tuple]] = {}
    for _, li, t, geometry in cuts:
        by_line.setdefault(li, []).append((t, geometry))
    strips = []
    for li, entries in by_line.items():
        entries.sort(key=lambda e: e[0])
        for (_, a), (_, b) in zip(entries, entries[1:]):
            ring = np.vstack([np.asarray(a.coords), np.asarray(b.coords)[::-1]])
            strip = shapely.make_valid(Polygon(ring))
            if not strip.is_empty:
                strips.append(strip)
    degree: dict[int, int] = {}
    ends: dict[int, np.ndarray] = {}
    for geometry, meta in zip(plan.lines, plan.line_meta):
        coords = np.asarray(geometry.coords)
        for junction, point in ((meta.start_junction, coords[0]), (meta.end_junction, coords[-1])):
            degree[junction] = degree.get(junction, 0) + 1
            ends[junction] = point
    for junction, count in degree.items():
        if count >= 3:
            extents = [e for (j, _), e in plan.half_width_at.items() if j == junction]
            radius = float(np.clip(max(extents, default=3.0) + 2.0, 6.0, 27.0))
            strips.append(shapely.Point(ends[junction]).buffer(radius))
    if not strips:
        return None
    return shapely.union_all(np.array(strips, dtype=object))


def _distinct_corners(coords: np.ndarray, tolerance: float = 1e-3) -> np.ndarray:
    """A ring without consecutive repeats.

    Where a cut ends on a kerb, the noding puts the crossing a few nanometres
    from the cut's own end point, and both survive into the face. Welded to the
    millimetre they become the same vertex twice in a row, which is a
    zero-length edge and turned most of the quads into pentagons.
    """
    keep = np.ones(len(coords), dtype=bool)
    last = coords[-1]
    for i, point in enumerate(coords):
        if np.hypot(*(point - last)) < tolerance:
            keep[i] = False
        else:
            last = point
    out = coords[keep]
    # The wrap-around: the first point against the last one kept.
    while len(out) > 1 and np.hypot(*(out[0] - out[-1])) < tolerance:
        out = out[:-1]
    return out


def _turning(ring: np.ndarray) -> np.ndarray:
    """How sharply a closed ring turns at each vertex, in degrees."""
    incoming = ring - np.roll(ring, 1, axis=0)
    outgoing = np.roll(ring, -1, axis=0) - ring
    a = np.arctan2(incoming[:, 1], incoming[:, 0])
    b = np.arctan2(outgoing[:, 1], outgoing[:, 0])
    return np.degrees(np.abs((b - a + np.pi) % (2 * np.pi) - np.pi))


def _straighten_runs(faces: list[tuple], tolerance: float) -> list[tuple]:
    """Drop kerb vertices that a straight line between two cuts would do as well.

    A strip between two cross-sections is a quad only if nothing lies on its
    sides between the cuts. The survey puts vertices there -- a slight bend, a
    driveway apron a few centimetres proud -- and every one turns a quad into
    a pentagon an editor cannot loop through.

    Decided per run, not per face. A run is the stretch of kerb between two
    vertices where something else meets it (a cut, a third surface, the edge
    of the network), and it is shared by the two faces either side, so both
    lose the same vertices and no crack or T-junction can open between them.
    A vertex is dropped only if the run without it stays within `tolerance`.
    """
    from .geo import douglas_peucker_keep

    if tolerance <= 0 or not faces:
        return faces
    keyed = [[(round(x * 1000), round(y * 1000)) for x, y in f[0]] for f in faces]
    position: dict[tuple, np.ndarray] = {}
    neighbours: dict[tuple, set] = {}
    for coords, keys in zip((f[0] for f in faces), keyed):
        for i, key in enumerate(keys):
            position[key] = coords[i]
            nxt = keys[(i + 1) % len(keys)]
            neighbours.setdefault(key, set()).add(nxt)
            neighbours.setdefault(nxt, set()).add(key)
    fixed = {key for key, near in neighbours.items() if len(near) != 2}

    removed: set = set()
    done: set = set()
    for keys in keyed:
        n = len(keys)
        anchors = [i for i, key in enumerate(keys) if key in fixed]
        if len(anchors) < 2:
            continue
        for a, b in zip(anchors, anchors[1:] + [anchors[0] + n]):
            run = [keys[i % n] for i in range(a, b + 1)]
            if len(run) < 3:
                continue
            canonical = tuple(run) if run[0] <= run[-1] else tuple(reversed(run))
            if canonical in done:
                continue
            done.add(canonical)
            points = np.array([position[k] for k in canonical])
            keep = douglas_peucker_keep(points, tolerance)
            removed.update(k for k, kept in zip(canonical, keep) if not kept)

    # A face may not fall below a triangle. Put back whatever it needs; the
    # vertex comes back for its neighbour too, which keeps the two matched.
    changed = True
    while changed:
        changed = False
        for keys in keyed:
            if sum(1 for k in keys if k not in removed) < 3:
                for k in keys:
                    if k in removed:
                        removed.discard(k)
                        changed = True

    out = []
    for (coords, *rest), keys in zip(faces, keyed):
        mask = np.array([k not in removed for k in keys])
        out.append((coords[mask], *rest))
    return out


def skeleton_centrelines(
    region,
    *,
    max_width_m: float = SKELETON_MAX_WIDTH_M,
    start_id: int = 1,
    group_start: int = 1,
) -> list:
    """Centrelines for road nothing else reaches, from the shape of the road.

    The medial axis of each strip of `region`: the Voronoi diagram of points
    along its outline, kept where it runs strictly inside, with the spurs every
    corner throws off pruned away. What is left runs down the middle of the
    strip and branches where the strip does, which is what a centreline is.

    Every line from one connected piece of road shares a `group`, so the
    pieces can be handed over as one object rather than as dozens.
    """
    from .nwb import Centreline

    out: list = []
    next_id = start_id
    node_ids: dict[tuple[int, int], int] = {}

    def node(point) -> int:
        key = (round(point[0] * 10), round(point[1] * 10))
        if key not in node_ids:
            node_ids[key] = -(start_id + len(node_ids))
        return node_ids[key]

    group = group_start
    for component in _polygons_of(region):
        if component.area < 15.0:
            continue
        width = 2.0 * component.area / max(component.length, 1e-9)
        if width > max_width_m:
            continue
        pieces = _medial_axis(component)
        for piece in pieces:
            coords = np.asarray(piece.coords)[:, :2]
            if len(coords) < 2:
                continue
            out.append(
                Centreline(
                    points=coords,
                    section_id=-next_id,
                    start_junction=node(coords[0]),
                    end_junction=node(coords[-1]),
                    name="",
                    kind="skeleton",
                    group=group,
                )
            )
            next_id += 1
        if pieces:
            group += 1
    return out


def _medial_axis(polygon, spacing: float = 1.0) -> list:
    """The pruned, smoothed medial axis of one polygon, as line strings."""
    import shapely

    boundary = shapely.segmentize(polygon.boundary, spacing)
    points = shapely.get_coordinates(boundary)
    if len(points) < 4:
        return []
    edges = shapely.get_parts(
        shapely.voronoi_polygons(shapely.multipoints(points), only_edges=True)
    )
    shapely.prepare(polygon)
    edges = edges[shapely.contains_properly(polygon, edges)]
    if not len(edges):
        return []
    merged = shapely.get_parts(shapely.line_merge(shapely.union_all(edges)))

    # Prune spurs: a branch that ends in the open and is shorter than the road
    # is wide where it leaves the trunk is a corner's, not the road's.
    lines = [np.asarray(m.coords)[:, :2] for m in merged if m.length > 0]
    for _ in range(4):
        ends: dict[tuple[int, int], int] = {}
        for coords in lines:
            for p in (coords[0], coords[-1]):
                key = (round(p[0] * 100), round(p[1] * 100))
                ends[key] = ends.get(key, 0) + 1
        kept = []
        pruned = False
        for coords in lines:
            a = (round(coords[0][0] * 100), round(coords[0][1] * 100))
            b = (round(coords[-1][0] * 100), round(coords[-1][1] * 100))
            length = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
            leaf = ends[a] == 1 or ends[b] == 1
            if leaf and len(lines) > 1:
                trunk = coords[-1] if ends[a] == 1 else coords[0]
                half = float(polygon.exterior.distance(shapely.Point(trunk)))
                if length < max(3.0, 2.5 * half):
                    pruned = True
                    continue
            kept.append(coords)
        lines = kept
        if not pruned:
            break
        joined = shapely.line_merge(
            shapely.union_all(np.array([shapely.LineString(c) for c in lines], dtype=object))
        )
        lines = [np.asarray(m.coords)[:, :2] for m in shapely.get_parts(joined) if m.length > 0]

    out = []
    for coords in lines:
        line = shapely.LineString(coords).simplify(0.25)
        if line.length < 4.0:
            continue
        # Cut into metre steps before smoothing, so the smoothing rounds each
        # bend over a metre or so and cannot cut an inside corner.
        out.append(_chaikin(shapely.segmentize(line, 1.0), 2))
    return out


def _chaikin(line, rounds: int):
    """Corner-cut a polyline, keeping its ends, so its tangent turns smoothly."""
    import shapely

    coords = np.asarray(line.coords)[:, :2]
    for _ in range(rounds):
        if len(coords) < 3:
            break
        q = 0.75 * coords[:-1] + 0.25 * coords[1:]
        r = 0.25 * coords[:-1] + 0.75 * coords[1:]
        middle = np.empty((2 * (len(coords) - 1), 2))
        middle[0::2] = q
        middle[1::2] = r
        coords = np.vstack([coords[:1], middle[1:-1], coords[-1:]])
    return shapely.LineString(coords)


def _without_holes(face, depth: int = 0) -> list:
    """A face with an island in it, cut so every piece is one outline.

    Blender's faces have no holes. A traffic island inside a junction would
    otherwise make the junction face unrepresentable, so a line is run through
    each island and the face re-split along it -- vertically first, then
    horizontally if that was not enough.
    """
    import shapely
    from shapely import LineString

    if not len(face.interiors):
        return [face]
    if depth >= 4:
        # Never seen; but a face is better filled than lost.
        LOG.debug("could not split the islands out of a road face; filling them")
        return [shapely.Polygon(face.exterior)]
    xmin, ymin, xmax, ymax = face.bounds
    lines = [face.boundary]
    for ring in face.interiors:
        c = shapely.Polygon(ring).representative_point()
        if depth % 2 == 0:
            lines.append(LineString([(c.x, ymin - 1.0), (c.x, ymax + 1.0)]))
        else:
            lines.append(LineString([(xmin - 1.0, c.y), (xmax + 1.0, c.y)]))
    noded = shapely.union_all(np.array(lines, dtype=object))
    pieces = [
        p for p in shapely.get_parts(shapely.polygonize(shapely.get_parts(noded)))
        if p.area > 1e-6 and face.contains(p.representative_point())
    ]
    out = []
    for piece in pieces:
        out.extend(_without_holes(piece, depth + 1) if len(piece.interiors) else [piece])
    return out


# ---------------------------------------------------------------------------
# 3D: heights, kerbs, skirts, objects
# ---------------------------------------------------------------------------


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    return cleaned or "unnamed"


def build_road_mesh(
    parts: list[RoadPart],
    network,
    *,
    ground,
    lift_m: float = LIFT_M,
    kerb_m: float = KERB_M,
    skirt_m: float = SKIRT_M,
    station_m: float = STATION_M,
    simplify_m: float = SIMPLIFY_M,
    reach_m: float = REACH_M,
    water: list | None = None,
    water_clearance_m: float = 0.25,
    deck_sampler=None,
    raised_classes: tuple[int, ...] = RAISED_CLASSES,
    uv_origin: tuple[float, float] = (0.0, 0.0),
) -> RoadMesh:
    """The whole road network as one mesh, split into objects.

    `ground` is the surveyed ground, a sampler of RD coordinates. Roads on a
    bridge (BGT level above zero) are meshed apart from those at grade and
    ride their deck, read from `deck_sampler` when there is one.
    """
    import shapely

    mesh = RoadMesh()
    if not parts:
        return mesh

    levels = sorted({1 if part.level > 0 else 0 for part in parts})
    plans: list[tuple[int, _Plan]] = []
    for group in levels:
        # Indices into `parts`, which the deck levels are keyed on.
        index_of = [i for i, p in enumerate(parts) if (1 if p.level > 0 else 0) == group]
        partition = clean_partition([parts[i] for i in index_of], simplify_m)
        partition = [(poly, code, level, index_of[i]) for poly, code, level, i in partition]
        plan = plan_faces(
            partition, network, station_m=station_m, reach_m=reach_m,
            group_start=1 + 1000 * group,
        )
        plans.append((group, plan))

    # One deck height per bridge part, measured once.
    deck_level: dict[int, float] = {}
    if deck_sampler is not None:
        for index, part in enumerate(parts):
            if part.level <= 0:
                continue
            ring = np.asarray(part.rings[0], dtype=np.float64)[:, :2]
            g = np.asarray(ground(ring[:, 0], ring[:, 1]), dtype=np.float64)
            level = _robust_deck_level(ring, deck_sampler, g)
            if level is not None:
                deck_level[index] = level

    vertex_key: dict[tuple, int] = {}
    vertices: list[tuple[float, float, float]] = []
    sizes: list[int] = []
    loops: list[int] = []
    classes: list[int] = []
    face_levels: list[int] = []
    objects_of_face: list[int] = []
    kinds: list[int] = []
    uvs: list[tuple[float, float]] = []
    object_index: dict[str, int] = {}
    object_names: list[str] = []
    object_kinds: list[str] = []

    def object_for(name: str, kind: str) -> int:
        if name not in object_index:
            object_index[name] = len(object_names)
            object_names.append(name)
            object_kinds.append(kind)
        return object_index[name]

    total_surface = 0
    quads = 0
    for group, plan in plans:
        if not plan.faces:
            continue
        coords_all = np.vstack([f[0] for f in plan.faces])
        base = np.asarray(ground(coords_all[:, 0], coords_all[:, 1]), dtype=np.float64)
        held = _hold_above_water(coords_all, base + lift_m, water, water_clearance_m)
        offsets = np.cumsum([0] + [len(f[0]) for f in plan.faces])

        # Which street, junction or paving each face belongs to.
        owner_line, owner_t, owner_d = _owners(plan, [f[0] for f in plan.faces])
        names_by_junction = _junction_names(plan)

        # Planar edge -> faces, for kerbs and skirts.
        edge_faces: dict[tuple, list[tuple[int, int, int]]] = {}
        face_vertex_ids: list[list[int]] = []
        group_face_ids: list[int] = []
        for fi, (coords, code, level, part) in enumerate(plan.faces):
            raised = code in raised_classes
            heights = held[offsets[fi] : offsets[fi + 1]].copy()
            if group > 0:
                deck = deck_level.get(part)
                if deck is not None:
                    # On the deck or on the ground, whichever is higher: a
                    # bridge part carries its approach ramp too, and where
                    # that runs onto rising ground a flat deck would sink
                    # into it.
                    heights = np.maximum(heights, deck + lift_m)
                else:
                    # Nothing usable overhead: draped like any other road,
                    # and said to be at grade, because that is where it is.
                    level = 0
            if raised:
                heights = heights + kerb_m
            ids = []
            for (x, y), z in zip(coords, heights):
                key = (round(x * 1000), round(y * 1000), group, raised)
                if key not in vertex_key:
                    vertex_key[key] = len(vertices)
                    vertices.append((x, y, float(z)))
                ids.append(vertex_key[key])

            name, kind, line = _object_name(
                plan, owner_line[fi], owner_t[fi], owner_d[fi], names_by_junction,
                level=level, reach_m=reach_m,
            )
            obj = object_for(name, kind)
            face_id = len(sizes)
            sizes.append(len(ids))
            loops.extend(ids)
            classes.append(code)
            face_levels.append(level)
            objects_of_face.append(obj)
            kinds.append(SURFACE)
            if line is not None:
                geometry = plan.lines[line]
                along = shapely.line_locate_point(geometry, shapely.points(coords))
                across = _signed_offset(geometry, coords, along)
                uvs.extend(zip(across.tolist(), along.tolist()))
            else:
                # Metres from the model origin. Raw RD would put 193 km into
                # a float32 UV and leave it centimetres to resolve with.
                uvs.extend(map(tuple, (coords - np.asarray(uv_origin)).tolist()))
            face_vertex_ids.append(ids)
            group_face_ids.append(face_id)
            total_surface += 1
            quads += len(ids) == 4

            for k in range(len(coords)):
                a = (round(coords[k][0] * 1000), round(coords[k][1] * 1000))
                b = (round(coords[(k + 1) % len(coords)][0] * 1000),
                     round(coords[(k + 1) % len(coords)][1] * 1000))
                edge = (a, b) if a < b else (b, a)
                edge_faces.setdefault(edge, []).append((fi, k, int(raised)))

        # Kerbs where a raised face meets a lower one, skirts along the edge
        # of the network.
        for edge, users in edge_faces.items():
            if len(users) == 1:
                fi, k, raised = users[0]
                _add_vertical(
                    face_vertex_ids[fi], k, plan.faces[fi][0], vertices, sizes, loops,
                    classes, face_levels, objects_of_face, kinds, uvs,
                    drop=skirt_m + (kerb_m if raised else 0.0), kind=SKIRT,
                    obj=objects_of_face[group_face_ids[fi]], level=face_levels[group_face_ids[fi]],
                )
            elif len(users) == 2 and users[0][2] != users[1][2]:
                fi, k, _ = users[0] if users[0][2] else users[1]
                _add_vertical(
                    face_vertex_ids[fi], k, plan.faces[fi][0], vertices, sizes, loops,
                    classes, face_levels, objects_of_face, kinds, uvs,
                    drop=kerb_m, kind=KERB,
                    obj=objects_of_face[group_face_ids[fi]], level=face_levels[group_face_ids[fi]],
                    reuse=vertex_key, group=group,
                )

    mesh.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    mesh.face_sizes = np.asarray(sizes, dtype=np.int32)
    mesh.face_loops = np.asarray(loops, dtype=np.int64)
    mesh.face_class = np.asarray(classes, dtype=np.int32)
    mesh.face_level = np.asarray(face_levels, dtype=np.int32)
    mesh.face_object = np.asarray(objects_of_face, dtype=np.int32)
    mesh.face_kind = np.asarray(kinds, dtype=np.int8)
    mesh.loop_uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    mesh.object_names = object_names
    mesh.object_kinds = object_kinds
    surface = mesh.face_kind == SURFACE
    mesh.stats = {
        "faces": int(surface.sum()),
        "quads": int(quads),
        "quad_share": round(quads / max(total_surface, 1), 3),
        "largest_face_vertices": int(mesh.face_sizes[surface].max()) if surface.any() else 0,
        "kerb_faces": int((mesh.face_kind == KERB).sum()),
        "skirt_faces": int((mesh.face_kind == SKIRT).sum()),
        "vertices": int(len(mesh.vertices)),
        "streets": sum(1 for k in object_kinds if k == "street"),
        "junctions": sum(1 for k in object_kinds if k == "junction"),
        "paving_objects": sum(1 for k in object_kinds if k == "paving"),
        "centrelines": sum(len(plan.lines) for _, plan in plans),
        "drawn_centrelines": sum(plan.stats.get("skeleton_centrelines", 0) for _, plan in plans),
        "kerb_m": float(kerb_m),
        "lift_m": float(lift_m),
        "cross_sections": sum(plan.stats.get("cuts", 0) for _, plan in plans),
    }
    LOG.info(
        "roads: %d faces (%.0f%% quads), %d kerb faces, %d streets, %d junctions, "
        "%d cross-sections from %d centrelines",
        mesh.stats["faces"],
        100 * mesh.stats["quad_share"],
        mesh.stats["kerb_faces"],
        mesh.stats["streets"],
        mesh.stats["junctions"],
        mesh.stats["cross_sections"],
        mesh.stats["centrelines"],
    )
    return mesh


def _add_vertical(
    ids, k, coords, vertices, sizes, loops, classes, face_levels, objects_of_face,
    kinds, uvs, *, drop, kind, obj, level, reuse=None, group=None,
):
    """A vertical face hanging off edge `k` of a face, on its right-hand side.

    The face's own loop runs anticlockwise seen from above, so its interior is
    to the left of each edge and the right is outward. For a kerb the bottom
    edge is the lower face's own vertices, shared, so the kerb is welded to
    both surfaces it joins; a skirt's bottom is new.
    """
    a_top, b_top = ids[k], ids[(k + 1) % len(ids)]
    ax, ay, az = vertices[a_top]
    bx, by, bz = vertices[b_top]
    if reuse is not None:
        a_bot = reuse.get((round(ax * 1000), round(ay * 1000), group, False))
        b_bot = reuse.get((round(bx * 1000), round(by * 1000), group, False))
        if a_bot is None or b_bot is None:
            return
    else:
        a_bot = len(vertices)
        vertices.append((ax, ay, az - drop))
        b_bot = len(vertices)
        vertices.append((bx, by, bz - drop))
    sizes.append(4)
    loops.extend([a_bot, b_bot, b_top, a_top])
    classes.append(CLASS_KERB)
    face_levels.append(level)
    objects_of_face.append(obj)
    kinds.append(kind)
    run = float(np.hypot(bx - ax, by - ay))
    uvs.extend([(0.0, 0.0), (run, 0.0), (run, drop), (0.0, drop)])


def _signed_offset(geometry, coords: np.ndarray, along: np.ndarray) -> np.ndarray:
    """Signed distance from the centreline, positive on its left."""
    out = np.empty(len(coords))
    for i, (point, t) in enumerate(zip(coords, along)):
        p, n = _frame(geometry, float(t))
        out[i] = float((point - p) @ n)
    return out


def _owners(plan: _Plan, face_coords: list[np.ndarray]):
    """Nearest centreline to each face, how far along it, and how far off."""
    import shapely
    from shapely import STRtree

    count = len(face_coords)
    if not plan.lines or not count:
        return np.full(count, -1), np.zeros(count), np.full(count, np.inf)
    # Not the vertex mean, which can fall outside a concave face.
    centres = shapely.point_on_surface(
        np.array([shapely.Polygon(c) for c in face_coords], dtype=object)
    )
    lines = np.array(plan.lines, dtype=object)
    found, distance = STRtree(lines).query_nearest(
        centres, return_distance=True, all_matches=False
    )
    line = np.full(count, -1, dtype=np.int64)
    dist = np.full(count, np.inf)
    line[found[0]] = found[1]
    dist[found[0]] = distance
    along = np.zeros(count)
    ok = line >= 0
    along[ok] = shapely.line_locate_point(lines[line[ok]], centres[ok])
    return line, along, dist


def _junction_names(plan: _Plan) -> dict[int, str]:
    by_junction: dict[int, set[str]] = {}
    for meta in plan.line_meta:
        for junction in (meta.start_junction, meta.end_junction):
            if meta.name:
                by_junction.setdefault(junction, set()).add(meta.name)
    return {
        junction: "_x_".join(_slug(n) for n in sorted(names)[:3])
        for junction, names in by_junction.items()
    }


def _object_name(plan: _Plan, line: int, t: float, distance: float, junction_names,
                 *, level: int, reach_m: float):
    """Which object a face goes in, its kind, and the centreline for its UVs."""
    if line < 0 or distance > reach_m:
        return ("Paving_bridge" if level > 0 else "Paving"), "paving", None
    meta = plan.line_meta[line]
    length = plan.lines[line].length
    lo, back = plan.setbacks[line] if line < len(plan.setbacks) else (0.0, 0.0)
    hi = length - back
    beyond = (t < lo - 1e-6) or (t > hi + 1e-6)
    if meta.kind == "skeleton":
        # Road we drew the centreline for ourselves has no names to go by,
        # so one connected network of it is one object, junctions included.
        path = _path_like(plan, line)
        name = f"{'Paths' if path else 'Lanes'}_{meta.group:03d}"
        if level > 0:
            name = f"Bridge_{name}"
        return name, "street", None if beyond else line
    for junction, edge in ((meta.start_junction, t < lo - 1e-6), (meta.end_junction, t > hi + 1e-6)):
        if edge:
            label = junction_names.get(junction) or str(junction)
            return f"Junction_{label}_{junction}", "junction", None
    street = meta.name or f"unnamed_{abs(meta.section_id)}"
    path = meta.kind == "FP" or (meta.kind == "skeleton" and _path_like(plan, line))
    prefix = "Bridge" if level > 0 else ("Path" if path else "Road")
    return f"{prefix}_{_slug(street)}", "street", line


def _path_like(plan: _Plan, line: int) -> bool:
    """Whether a centreline we drew runs down a footway or cycle path."""
    return bool(plan.path_lines and line in plan.path_lines)


def road_top(mesh: RoadMesh, bbox, cell_m: float = 0.5):
    """The top of the road seen from above, as a height raster.

    What anything placed on a road stands on. Roads on bridges are left out:
    a lamp post under a bridge stands on the quay.
    """
    from .elevation import HeightRaster
    from .relight import burn_triangles

    cols = max(1, int(math.ceil(bbox.width / cell_m)))
    rows = max(1, int(math.ceil(bbox.height / cell_m)))
    top = np.full((rows, cols), -np.inf)
    tris, owner = mesh.triangles((SURFACE,))
    if len(tris):
        burn_triangles(tris[mesh.face_level[owner] <= 0], bbox, top)
    top[~np.isfinite(top)] = np.nan
    return HeightRaster(top, bbox)


def road_underside(mesh: RoadMesh, bbox, cell_m: float = 0.25) -> np.ndarray:
    """Points the terrain has to stay under: (x, y, z), wherever there is road.

    The lowest road over every quarter-metre cell, plus every vertex and the
    midpoint of every edge of the road surface itself, so a road edge that
    grazes a cell without covering its centre still counts. Roads on bridges
    are left out: the ground under a deck needs pushing nowhere.
    """
    from .relight import burn_triangles

    tris, owner = mesh.triangles((SURFACE,))
    if not len(tris):
        return np.zeros((0, 3))
    tris = tris[mesh.face_level[owner] <= 0]
    cols = max(1, int(math.ceil(bbox.width / cell_m)))
    rows = max(1, int(math.ceil(bbox.height / cell_m)))
    low = np.full((rows, cols), -np.inf)
    flipped = tris.copy()
    flipped[:, :, 2] *= -1.0
    burn_triangles(flipped, bbox, low)
    r, c = np.nonzero(np.isfinite(low))
    cells = np.column_stack([
        bbox.xmin + (c + 0.5) * (bbox.width / cols),
        bbox.ymax - (r + 0.5) * (bbox.height / rows),
        -low[r, c],
    ])
    corners = tris.reshape(-1, 3)
    middles = 0.5 * (tris + np.roll(tris, -1, axis=1)).reshape(-1, 3)
    return np.vstack([cells, corners, middles])


__all__ = [
    "KERB",
    "KERB_M",
    "LIFT_M",
    "RAISED_CLASSES",
    "REACH_M",
    "RoadMesh",
    "SKIRT",
    "SKIRT_M",
    "STATION_M",
    "SURFACE",
    "TERRAIN_CLEARANCE_M",
    "build_road_mesh",
    "clean_partition",
    "plan_faces",
    "road_top",
    "road_underside",
    "skeleton_centrelines",
]
