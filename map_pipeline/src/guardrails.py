"""Steel guardrails along the motorway, for the layered style.

A Dutch motorway has a steel guardrail (*geleiderail*) along most of its verges
and down its central reservation, and none of it is in the open data: the BGT
draws the carriageway and the verge beside it and nothing standing between
them. This lays one along every stretch of motorway edge that borders a verge,
grass or open ground, where the ground is built (`layered`):

* **Along the carriageway, not across it.** The edges are read off the ground
  as built: where a motorway face meets a face of grass, open ground, water or
  paving that is no road (a hard verge). Where the carriageway meets another
  road part (a junction mouth, the lane into a service area, a slip road that
  is a regional road at its far end) there is no rail, and none within
  `CLEAR_M` of that road either, so a rail stops short of a mouth rather than
  turning the corner into it. Where one motorway part meets another, at a
  merge, there is no edge at all. A green gore the layered style kerbs as an
  island has its kerb for an edge, and no rail.
* **Not on a bridge.** Where a carriageway ends at a part on a bridge or in a
  tunnel (``relatieve_hoogteligging`` not 0) the edge across its end is left
  open: the structures stage owns the deck. A verge running under a bridge
  keeps its rail.
* **Not twice.** Within `GUARDED_M` of a BGT wall, fence or noise barrier the
  edge is already guarded.
* **On the verge.** The rail's face stands `SETBACK_M` onto the verge, and
  nowhere closer than that to any carriageway: in a gore, where two
  carriageways part, the two rails stop where the gore gets too narrow for
  them, rather than crossing; two rails back to back closer than `APART_M`
  are thinned to one.

The rail is a W-profile beam `RAIL_HEIGHT_M` tall, its top `RAIL_TOP_M` above
the ground, on square posts every `POST_EVERY_M`, in broad terms Rijkswaterstaat's
standard steel guardrail. The profile is folded once down its middle, which is
enough to catch the light the way the real one does: about 10 triangles a metre.
Every height is read off the ground as built, so a post stands on the verge
the model has, not on the survey under it, `POST_EMBED_M` into it under the
lowest of its corners.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

# The rail. Its top stands this far above the ground at its face, and the
# beam is this tall and this deep, front to back.
RAIL_TOP_M = 0.75
RAIL_HEIGHT_M = 0.31
RAIL_DEPTH_M = 0.08
# How far the fold down the middle of the face is set back from the two
# ridges either side of it: the W of the profile, in one fold.
RAIL_GROOVE_M = 0.04
# Posts: how far apart, how thick (square), and how far into the ground, so
# a post on a sloping verge does not float at its downhill corner.
POST_EVERY_M = 2.0
POST_SIZE_M = 0.10
POST_EMBED_M = 0.10
# The rail's face stands this far onto the verge from the carriageway edge.
SETBACK_M = 0.5
# No closer than this to another road: that is a junction, not a verge.
CLEAR_M = 0.3
# A BGT wall, fence or noise barrier this close already guards the edge.
GUARDED_M = 0.6
# An edge with both ends this close to a part on a bridge or in a tunnel lies
# along it: the outlines the ground is built from are simplified and snapped,
# so the end of a carriageway can stand a few tens of centimetres off the
# bridge part it runs on into.
ELEVATED_REACH_M = 1.0
# Nothing shorter is a guardrail: a stretch between two mouths, or a sliver.
MIN_RUN_M = 10.0
# A rail ends this far inside the edge of the area, so its posts, behind it,
# are still over the ground.
FRAME_CLEAR_M = RAIL_DEPTH_M + POST_SIZE_M + 0.05
# How far out a carriageway is looked for beyond a rail, to tell a central
# reservation (another carriageway beyond it) from a verge, for the counts.
BETWEEN_M = 30.0

RAIL, POST = 0, 1
KIND_NAMES = {RAIL: "rail", POST: "post"}


@dataclass
class GuardrailSet:
    """The guardrails as triangles, RD and NAP, with what the checks read."""

    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    # Metres along the rail and up from the ground: a tiling texture's UVs.
    uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    kind: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    # For the checks: the middle of every post's foot, the four corners of
    # it (P x 4 x 3), and the top of the rail's face at every post. All are
    # vertices of the mesh but the middle of the foot.
    post_foot: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    post_corners: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 3)))
    rail_top: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # The rail lines, (N, 2) RD each, the carriageway on their left.
    lines: list = field(default_factory=list)
    # A fattened solid along each rail for physics; empty unless asked for.
    collision_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    collision_triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    collision_thickness_m: float = 0.0
    counts: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.triangles))

    def stats(self) -> dict:
        out = {"triangles": len(self), **self.counts}
        if len(self.collision_triangles):
            out["collision_triangles"] = int(len(self.collision_triangles))
        return out

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vertices=self.vertices.astype(np.float64),
            triangles=self.triangles.astype(np.int32),
            uv=self.uv.astype(np.float64),
            kind=self.kind.astype(np.int8),
            kind_names=np.array([KIND_NAMES[k] for k in sorted(KIND_NAMES)], dtype=str),
            post_foot=self.post_foot.astype(np.float64),
            post_corners=self.post_corners.reshape(-1, 3).astype(np.float64),
            rail_top=self.rail_top.astype(np.float64),
            collision_vertices=self.collision_vertices.astype(np.float64),
            collision_triangles=self.collision_triangles.astype(np.int32),
            collision_thickness_m=np.float64(self.collision_thickness_m),
        )
        return path


# ---------------------------------------------------------------------------
# Where: the motorway's edges as built
# ---------------------------------------------------------------------------


def verge_faces(face_layer, face_material) -> np.ndarray:
    """The faces a guardrail may stand on, beside a carriageway.

    Grass and open ground (water and the gap under a bridge included), and
    paving that is no road: a hard verge. Not a road, a footpath or a
    kerbed island.
    """
    from .layered import GRASS, MATERIAL_NAMES, ROAD, TERRAIN

    layer = np.asarray(face_layer, dtype=np.int64)
    material = np.asarray(face_material, dtype=np.int64)
    hard = MATERIAL_NAMES.index("Verharding")
    return (layer == GRASS) | (layer == TERRAIN) | ((layer == ROAD) & (material == hard))


def motorway_edges(sizes, loops, face_layer, face_material):
    """Every edge round the motorway faces, and what lies across it.

    Returns ``(edges, neighbour, verge)``: vertex-id pairs ordered as their
    motorway face runs, counter-clockwise from above, so the carriageway is on
    the left of each; the face across (-1 on the edge of the area); and
    whether that face is verge (`verge_faces`).
    """
    from .layered import MOTORWAY

    sizes = np.asarray(sizes, dtype=np.int64)
    loops = np.asarray(loops, dtype=np.int64)
    layer = np.asarray(face_layer, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    following = np.arange(len(loops)) + 1
    following[starts + sizes - 1] = starts
    face_of = np.repeat(np.arange(len(sizes)), sizes)
    a, b = loops, loops[following]
    n = int(loops.max()) + 1 if len(loops) else 1
    key = np.minimum(a, b) * n + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    same = key[order][1:] == key[order][:-1]
    partner = np.full(len(loops), -1, dtype=np.int64)
    partner[order[:-1][same]] = order[1:][same]
    partner[order[1:][same]] = order[:-1][same]
    on_motorway = layer[face_of] == MOTORWAY
    across = np.where(partner >= 0, face_of[np.maximum(partner, 0)], -1)
    boundary = on_motorway & ((across < 0) | (layer[np.maximum(across, 0)] != MOTORWAY))
    edges = np.column_stack([a, b])[boundary]
    neighbour = across[boundary]
    verge = verge_faces(face_layer, face_material)
    is_verge = (neighbour >= 0) & verge[np.maximum(neighbour, 0)]
    return edges, neighbour, is_verge


def _faces_polygon(vertices, sizes, loops, members):
    """The union of some faces, as one shapely geometry."""
    import shapely

    members = np.asarray(members, dtype=np.int64)
    if not len(members):
        return shapely.Polygon()
    sizes = np.asarray(sizes, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    rings = [vertices[loops[starts[f]: starts[f] + sizes[f]], :2] for f in members.tolist()]
    polygons = np.array([shapely.Polygon(r) for r in rings], dtype=object)
    polygons = polygons[shapely.area(polygons) > 1e-9]
    if not len(polygons):
        return shapely.Polygon()
    try:
        return shapely.coverage_union_all(polygons)
    except Exception:  # noqa: BLE001 - not a clean coverage after all
        return shapely.union_all(polygons)


def rail_lines(
    vertices, sizes, loops, face_layer, face_material, *,
    frame=None, elevated=None, guarded=None,
    setback_m: float = SETBACK_M, clear_m: float = CLEAR_M, min_run_m: float = MIN_RUN_M,
) -> list:
    """Where the guardrails go: lines (N x 2) with the carriageway on the left.

    `vertices`, `sizes`, `loops`, `face_layer` and `face_material` are the
    ground as built (`LayeredGround`), `frame` the area (shapely), `elevated`
    the parts on bridges and in tunnels and `guarded` the reach of the BGT
    barriers already standing, both shapely and both optional.
    """
    import shapely

    from .layered import MOTORWAY

    vertices = np.asarray(vertices, dtype=np.float64)
    layer = np.asarray(face_layer, dtype=np.int64)
    motorway_faces = np.flatnonzero(layer == MOTORWAY)
    if not len(motorway_faces):
        return []
    edges, _, guard = motorway_edges(sizes, loops, face_layer, face_material)
    edges = edges[guard]
    if not len(edges):
        return []
    a = vertices[edges[:, 0], :2]
    b = vertices[edges[:, 1], :2]

    if elevated is not None and not elevated.is_empty:
        # Where the carriageway runs on onto a bridge or into a tunnel, the
        # edge across its end lies along the part above or below, both its
        # ends on it, with the carriageway on its near side clear of it. A
        # verge passing under a bridge crosses the bridge's outline instead,
        # and has the bridge over the carriageway beside it.
        mid = 0.5 * (a + b)
        d = b - a
        left = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(
            np.linalg.norm(d, axis=1), 1e-12
        )[:, None]
        shapely.prepare(elevated)
        inner = mid + ELEVATED_REACH_M * left
        along = (
            (shapely.distance(elevated, shapely.points(a)) <= ELEVATED_REACH_M)
            & (shapely.distance(elevated, shapely.points(b)) <= ELEVATED_REACH_M)
        )
        ends = along & ~shapely.contains_xy(elevated, inner[:, 0], inner[:, 1])
        a, b = a[~ends], b[~ends]
        if not len(a):
            return []

    # The edges joined into runs, each in the direction of its motorway face.
    runs = shapely.line_merge(
        shapely.multilinestrings(shapely.linestrings(np.stack([a, b], axis=1))),
        directed=True,
    )
    # Onto the verge: to the right of a run, away from the carriageway.
    offsets = [
        shapely.offset_curve(run, -setback_m, quad_segs=2, join_style="mitre", mitre_limit=2.0)
        for run in shapely.get_parts(runs)
        if run.length > 1e-6
    ]
    if not offsets:
        return []
    candidate = shapely.union_all(np.array(offsets, dtype=object))

    # Keep off every carriageway by the setback, and off every other road by
    # `clear_m`; inside the area, and away from what is already guarded.
    motorway = _faces_polygon(vertices, sizes, loops, motorway_faces)
    verge = verge_faces(face_layer, face_material)
    others = np.flatnonzero(~verge & (layer != MOTORWAY))
    roads = _faces_polygon(vertices, sizes, loops, others)
    keep_off = motorway.buffer(setback_m - 0.05, quad_segs=4)
    if not roads.is_empty:
        keep_off = shapely.union(keep_off, roads.buffer(clear_m, quad_segs=4))
    lines = candidate.difference(keep_off)
    if frame is not None:
        # Rail and posts wholly inside the area, over the ground it has.
        lines = lines.intersection(frame.buffer(-FRAME_CLEAR_M, join_style="mitre"))
    if guarded is not None and not guarded.is_empty:
        lines = lines.difference(guarded)
    if lines.is_empty:
        return []
    pieces = []
    for part in shapely.get_parts(shapely.line_merge(lines)):
        if part.geom_type != "LineString" or part.length < min_run_m:
            continue
        coords = _carriageway_on_the_left(np.asarray(part.coords)[:, :2], motorway)
        pieces.extend(_split_at_turns(coords))
    return _back_to_back(pieces, min_run_m)


# A rail turning further than this at one corner is two rails: the two sides
# of a gore, run into one where the offsets meet at its tip.
SPLIT_TURN_DEG = 120.0
# Two rails closer than this stand back to back with their posts in each
# other: rail and post deep twice over, and a little air.
APART_M = 2.0 * (RAIL_DEPTH_M + POST_SIZE_M) + 0.04


def _split_at_turns(coords: np.ndarray, limit_deg: float = SPLIT_TURN_DEG) -> list:
    """A line cut wherever it turns back on itself by more than `limit_deg`."""
    if len(coords) < 3:
        return [coords]
    d = np.diff(coords, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
    turn = np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(axis=1), -1.0, 1.0)))
    cuts = np.flatnonzero(turn > limit_deg) + 1
    out, start = [], 0
    for cut in cuts.tolist():
        out.append(coords[start: cut + 1])
        start = cut
    out.append(coords[start:])
    return [c for c in out if len(c) >= 2]


def _back_to_back(pieces: list, min_run_m: float) -> list:
    """Rails too close back to back thinned out: the shorter gives way.

    In a gore the rails of its two sides meet at the tip, and in a central
    reservation too narrow for two the rails of the two carriageways stand
    in each other; the longer of the two is kept whole and the other stops
    where they come within `APART_M`. A strip narrower than that gets one
    rail, facing one carriageway.
    """
    import shapely

    lines = sorted(
        (shapely.LineString(p) for p in pieces if len(p) >= 2),
        key=lambda line: -line.length,
    )
    kept: list = []
    for line in lines:
        if kept:
            near = shapely.union_all(np.array(kept, dtype=object)).buffer(APART_M, quad_segs=4)
            line = line.difference(near)
        merged = shapely.line_merge(line, directed=True) if not line.is_empty else line
        for part in shapely.get_parts(merged):
            if part.geom_type == "LineString" and part.length >= min_run_m:
                kept.append(part)
    return [np.asarray(line.coords)[:, :2] for line in kept]


def _carriageway_on_the_left(coords: np.ndarray, motorway) -> np.ndarray:
    """The line turned, if need be, so the carriageway is on its left."""
    import shapely

    if len(coords) < 2:
        return coords
    step = np.diff(coords, axis=0)
    length = np.linalg.norm(step, axis=1)
    i = int(np.argmax(length))
    mid = coords[i] + 0.5 * step[i]
    left = np.array([-step[i, 1], step[i, 0]]) / max(length[i], 1e-12)
    near = shapely.distance(motorway, shapely.Point(*(mid + 0.2 * left)))
    far = shapely.distance(motorway, shapely.Point(*(mid - 0.2 * left)))
    return coords if near <= far else coords[::-1].copy()


# ---------------------------------------------------------------------------
# What: rail and posts
# ---------------------------------------------------------------------------

# The profile across the rail, as (onto the verge from the face, up from the
# rail's foot): the two ridges of the face, the fold between them, and the
# back. Counter-clockwise in that plane, so each side's normal points out.
PROFILE = np.array([
    [0.0, RAIL_HEIGHT_M],
    [RAIL_GROOVE_M, 0.5 * RAIL_HEIGHT_M],
    [0.0, 0.0],
    [RAIL_DEPTH_M, 0.0],
    [RAIL_DEPTH_M, RAIL_HEIGHT_M],
])
# The end of the rail, fanned from the fold, which sees every other corner.
PROFILE_CAP = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 0]])


def stations(points: np.ndarray, every_m: float = POST_EVERY_M):
    """Evenly spaced points along a line, none further apart than `every_m`.

    Returns the points, how far along each is, and the unit direction across
    to the right of the line (onto the verge) at each, mitred at a bend.
    """
    from .rails import offset_normals

    points = np.asarray(points, dtype=np.float64)[:, :2]
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(step)])
    length = float(along[-1])
    count = max(int(np.ceil(length / every_m - 1e-9)), 1)
    at = np.linspace(0.0, length, count + 1)
    xy = np.column_stack([np.interp(at, along, points[:, 0]), np.interp(at, along, points[:, 1])])
    right = -offset_normals(xy)
    return xy, at, right


def guardrail(points: np.ndarray, ground_at):
    """Rail and posts along one line (N x 2), the carriageway on its left.

    `ground_at(x, y)` is the ground as built. The rail's face follows the
    line, `RAIL_TOP_M` above the ground under it at every post. Each post goes
    `POST_EMBED_M` into the ground under the lowest of its corners, so on a
    steep bank none of its foot is in the air. Returns ``(vertices,
    triangles, uv, kind, post_foot, post_corners, rail_top)``: the middle of
    every foot, its four corners, and the top of the rail's face at each post.
    """
    xy, along, right = stations(points)
    n = len(xy)
    unit = right / np.maximum(np.linalg.norm(right, axis=1), 1e-12)[:, None]
    ground = np.asarray(ground_at(xy[:, 0], xy[:, 1]), dtype=np.float64)
    foot = ground + (RAIL_TOP_M - RAIL_HEIGHT_M)

    # ---- the rail: the profile swept from post to post -------------------
    k = len(PROFILE)
    rail = np.empty((n, k, 3))
    for j, (out, up) in enumerate(PROFILE):
        rail[:, j, :2] = xy + right * out
        rail[:, j, 2] = foot + up
    vertices = [rail.reshape(-1, 3)]
    uv = [np.column_stack([np.repeat(along, k), (rail[:, :, 2] - ground[:, None]).ravel()])]
    tris = []
    for j in range(k):
        p, q = j, (j + 1) % k
        i = np.arange(n - 1)
        p0, p1, q0, q1 = i * k + p, (i + 1) * k + p, i * k + q, (i + 1) * k + q
        tris.append(np.column_stack([p0, p1, q1]))
        tris.append(np.column_stack([p0, q1, q0]))
    # The ends closed: the start faces back along the line, the end forward.
    tris.append(PROFILE_CAP)
    tris.append((n - 1) * k + PROFILE_CAP[:, ::-1])
    rail_tris = np.vstack(tris)
    kinds = [np.full(len(rail_tris), RAIL, dtype=np.int8)]
    all_tris = [rail_tris]
    offset = n * k

    # ---- posts: behind the rail, on the verge ----------------------------
    along_dir = np.column_stack([-unit[:, 1], unit[:, 0]])  # forward, along the line
    centre = xy + unit * (RAIL_DEPTH_M + 0.5 * POST_SIZE_M)
    top = ground + RAIL_TOP_M
    h = 0.5 * POST_SIZE_M
    left = -unit
    # Counter-clockwise from above.
    corners = np.stack([
        centre + h * (-along_dir - left),
        centre + h * (along_dir - left),
        centre + h * (along_dir + left),
        centre + h * (-along_dir + left),
    ], axis=1)
    under = np.asarray(ground_at(
        np.concatenate([centre[:, 0], corners[:, :, 0].ravel()]),
        np.concatenate([centre[:, 1], corners[:, :, 1].ravel()]),
    ), dtype=np.float64)
    under = np.column_stack([under[:n], under[n:].reshape(n, 4)])
    under = np.where(np.isfinite(under), under, ground[:, None])
    base = under.min(axis=1) - POST_EMBED_M
    post_vertices = np.concatenate([
        np.dstack([corners, np.repeat(base[:, None], 4, axis=1)]),
        np.dstack([corners, np.repeat(top[:, None], 4, axis=1)]),
    ], axis=1)  # (n, 8, 3): bottom ring then top ring
    box = []
    for s in range(4):
        t = (s + 1) % 4
        box += [[s, t, 4 + t], [s, 4 + t, 4 + s]]
    box += [[4, 5, 6], [4, 6, 7]]
    box = np.asarray(box, dtype=np.int64)
    post_tris = (np.arange(n)[:, None, None] * 8 + box[None]).reshape(-1, 3) + offset
    vertices.append(post_vertices.reshape(-1, 3))
    uv.append(np.column_stack([
        np.repeat(along, 8), (post_vertices[:, :, 2] - ground[:, None]).ravel(),
    ]))
    all_tris.append(post_tris)
    kinds.append(np.full(len(post_tris), POST, dtype=np.int8))

    post_foot = np.column_stack([centre, base])
    rail_top = np.column_stack([xy, top])
    return (
        np.vstack(vertices), np.vstack(all_tris), np.vstack(uv),
        np.concatenate(kinds), post_foot, post_vertices[:, :4], rail_top,
    )


def collision_solid(points: np.ndarray, ground_at, thickness_m: float):
    """A solid block along the rail for physics, as ``(vertices, triangles)``.

    From the rail's face back onto the verge, at least `thickness_m` thick and
    as deep as the rail and posts are, and from below the ground to the top of
    the rail: what `barriers.collision_style` does to a fence. The beam is
    8 cm deep, which a body at speed passes through between two physics
    steps; the block is not, and it holds the drawn rail and posts.
    """
    from .barriers import BarrierStyle, _sweep_line, collision_style

    drawn = BarrierStyle("guardrail", RAIL_TOP_M, RAIL_DEPTH_M + POST_SIZE_M, "metal")
    style = collision_style(drawn, float(thickness_m))
    xy, _, right = stations(points)
    # Swept along its own middle, so one side is the rail's face.
    centre = xy + right * (0.5 * style.thickness_m)
    vertices, triangles, _ = _sweep_line(centre, ground_at, style)
    return vertices, triangles


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def elevated_area(areas):
    """The ground areas on bridges and in tunnels, as one shapely geometry."""
    import shapely

    polygons = []
    for area in areas or ():
        if int(getattr(area, "level", 0) or 0) == 0:
            continue
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in area.rings]
        rings = [r for r in rings if len(r) >= 4]
        if not rings:
            continue
        valid = shapely.make_valid(shapely.Polygon(rings[0], rings[1:]))
        polygons.extend(p for p in shapely.get_parts(valid) if p.geom_type == "Polygon")
    if not polygons:
        return shapely.Polygon()
    return shapely.union_all(np.array(polygons, dtype=object))


def build_guardrails(
    ground,
    areas=(),
    *,
    bbox=None,
    guarded=None,
    collision_thickness_m: float | None = None,
    work_dir: Path | None = None,
) -> GuardrailSet:
    """Guardrails along the motorway of a layered ground (`LayeredGround`).

    `areas` is the GroundArea list the ground was built from: its parts on
    bridges and in tunnels are where a carriageway's end is left open.
    `guarded` is the reach of the BGT barriers already standing
    (`railings.guarded_area`, RD). `collision_thickness_m` adds the physics
    block (`collision_solid`). Written to ``guardrails.npz`` in `work_dir`.
    """
    import shapely

    result = GuardrailSet()
    mesh = ground.mesh
    frame = None
    if bbox is not None:
        frame = shapely.box(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
    lines = rail_lines(
        mesh.vertices, mesh.face_sizes, mesh.face_loops, mesh.face_kind, ground.face_material,
        frame=frame, elevated=elevated_area(areas), guarded=guarded,
    )
    result.lines = lines

    def ground_at(x, y):
        """The ground as built, read at the nearest point of the area."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if bbox is not None:
            x = np.clip(x, bbox.xmin + 1e-6, bbox.xmax - 1e-6)
            y = np.clip(y, bbox.ymin + 1e-6, bbox.ymax - 1e-6)
        return ground.surface_at(x, y)

    if lines:
        chunks, kinds, uvs, feet, corners, tops = [], [], [], [], [], []
        c_chunks = []
        offset = c_offset = 0
        for line in lines:
            verts, tris, uv, kind, foot, corner, top = guardrail(line, ground_at)
            chunks.append((verts, tris + offset))
            offset += len(verts)
            uvs.append(uv)
            kinds.append(kind)
            feet.append(foot)
            corners.append(corner)
            tops.append(top)
            if collision_thickness_m:
                c_verts, c_tris = collision_solid(line, ground_at, collision_thickness_m)
                c_chunks.append((c_verts, c_tris + c_offset))
                c_offset += len(c_verts)
        result.vertices = np.vstack([v for v, _ in chunks])
        result.triangles = np.vstack([t for _, t in chunks]).astype(np.int64)
        result.uv = np.vstack(uvs)
        result.kind = np.concatenate(kinds)
        result.post_foot = np.vstack(feet)
        result.post_corners = np.concatenate(corners)
        result.rail_top = np.vstack(tops)
        if c_chunks:
            result.collision_vertices = np.vstack([v for v, _ in c_chunks])
            result.collision_triangles = np.vstack([t for _, t in c_chunks]).astype(np.int64)
            result.collision_thickness_m = float(collision_thickness_m)
    result.counts = _counts(result, mesh)
    LOG.info(
        "guardrails: %d runs, %.0f m (%.0f m between two carriageways), %d posts, "
        "%d triangles (%.1f a metre)",
        result.counts["runs"], result.counts["length_m"],
        result.counts["between_carriageways_m"], result.counts["posts"], len(result),
        result.counts["triangles_per_m"],
    )
    if work_dir is not None:
        result.save(Path(work_dir) / "guardrails.npz")
    return result


def _counts(result: GuardrailSet, mesh) -> dict:
    """Runs, metres, posts, and how much of it runs between two carriageways."""
    import shapely

    from .layered import MOTORWAY

    length = float(sum(
        np.linalg.norm(np.diff(line, axis=0), axis=1).sum() for line in result.lines
    ))
    between = 0.0
    if result.lines:
        motorway = _faces_polygon(
            mesh.vertices, mesh.face_sizes, mesh.face_loops,
            np.flatnonzero(np.asarray(mesh.face_kind) == MOTORWAY),
        )
        shapely.prepare(motorway)
        for line in result.lines:
            xy, _, right = stations(line)
            unit = right / np.maximum(np.linalg.norm(right, axis=1), 1e-12)[:, None]
            # A carriageway straight across from the rail, beyond the verge.
            rays = shapely.linestrings(
                np.stack([xy + 0.5 * unit, xy + BETWEEN_M * unit], axis=1)
            )
            hit = shapely.intersects(motorway, rays)
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            between += float((steps * (hit[:-1] & hit[1:])).sum())
    return {
        "runs": len(result.lines),
        "length_m": round(length, 1),
        "between_carriageways_m": round(between, 1),
        "posts": int(len(result.post_foot)),
        "triangles_per_m": round(len(result) / length, 1) if length else 0.0,
    }


__all__ = [
    "GuardrailSet",
    "KIND_NAMES",
    "build_guardrails",
    "collision_solid",
    "elevated_area",
    "guardrail",
    "motorway_edges",
    "rail_lines",
    "stations",
    "verge_faces",
]
