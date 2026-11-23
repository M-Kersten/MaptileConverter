"""One door in the closed walls of a shed, a barn, a kiosk or a tower.

A building with no unit in the BAG, or one the registers call a barn, a water
tower or a transformer house, gets walls without windows (src/facade.py). A
closed wall reads as a block; one door says what it is. Drawing the door into
the wall texture would repeat it every few metres, all the way round, so it is
a piece of geometry of its own instead, set on one wall:

* the wall that looks at the nearest road, when OpenStreetMap has one near;
* on a barn, a gable end, where the big doors are;
* never a wall another building stands against.

A wooden door has a pale frame round it; a steel one, on a kiosk or a tower,
is flush. A barn's door is two leaves with a gap between them.
"""

from __future__ import annotations

import logging

import numpy as np

from .buildings import (
    ARCH_BARN,
    ARCH_CLOSED,
    ARCH_OUTBUILDING,
    ARCH_TECHNICAL,
    classify_archetype,
)

LOG = logging.getLogger(__name__)

# archetype: (width, height, leaves, mesh key)
DOORS = {
    ARCH_OUTBUILDING: (0.9, 2.0, 1, "doors"),
    ARCH_BARN: (3.2, 3.6, 2, "doors"),
    ARCH_TECHNICAL: (1.0, 2.1, 1, "steel_doors"),
    ARCH_CLOSED: (1.0, 2.2, 1, "steel_doors"),
}
# Kinds that are closed but have no door to speak of, or no walls at all.
NO_DOOR_KINDS = frozenset({"tank", "canopy", "fuel_canopy", "parking"})

LEAF_DEPTH_M = 0.06
FRAME_M = 0.08
FRAME_DEPTH_M = 0.1
LEAF_GAP_M = 0.03
# Space over the door before the eaves, and the least a door can be.
HEAD_ROOM_M = 0.2
MIN_DOOR_M = 1.6
# How far a road can be and still decide which way the door looks.
ROAD_REACH_M = 80.0
KEYS = ("doors", "steel_doors", "door_frames")


def _box(origin, along, out, width, depth, z0, height) -> np.ndarray:
    """A box standing on the wall, its back face left out: (10, 3, 3)."""
    along = np.append(np.asarray(along, dtype=np.float64), 0.0)
    out = np.append(np.asarray(out, dtype=np.float64), 0.0)
    base = np.array([origin[0], origin[1], z0], dtype=np.float64)
    up = np.array([0.0, 0.0, height])

    def corner(i, j, k):
        return base + i * width * along + j * depth * out + k * up

    quads = [
        (corner(0, 1, 0), corner(1, 1, 0), corner(1, 1, 1), corner(0, 1, 1)),  # front
        (corner(0, 0, 1), corner(0, 1, 1), corner(1, 1, 1), corner(1, 0, 1)),  # top
        (corner(0, 0, 0), corner(1, 0, 0), corner(1, 1, 0), corner(0, 1, 0)),  # bottom
        (corner(0, 0, 0), corner(0, 1, 0), corner(0, 1, 1), corner(0, 0, 1)),  # left
        (corner(1, 0, 0), corner(1, 0, 1), corner(1, 1, 1), corner(1, 1, 0)),  # right
    ]
    centre = corner(0.5, 0.5, 0.5)
    triangles = []
    for a, b, c, d in quads:
        for tri in ((a, b, c), (a, c, d)):
            tri = np.array(tri)
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if float(np.dot(normal, tri.mean(axis=0) - centre)) < 0.0:
                tri = tri[[0, 2, 1]]
            triangles.append(tri)
    return np.array(triangles)


def _edges(shape) -> list[tuple[np.ndarray, np.ndarray]]:
    """The outline's straight stretches, counter-clockwise, so the outside of
    each is on its right."""
    import shapely

    out = []
    for polygon in shapely.get_parts(shape):
        if polygon.geom_type != "Polygon":
            continue
        polygon = shapely.geometry.polygon.orient(polygon.simplify(0.25), 1.0)
        ring = np.asarray(polygon.exterior.coords)[:, :2]
        for a, b in zip(ring[:-1], ring[1:]):
            if float(np.linalg.norm(b - a)) > 1e-6:
                out.append((a, b))
    return out


def _wall_profile(building, a, b, out) -> tuple[np.ndarray, np.ndarray] | None:
    """The wall triangles in the plane of edge a-b, as (along, z) per corner."""
    walls = np.asarray(getattr(building, "wall_tris", []), dtype=np.float64)
    if walls.ndim != 3 or not len(walls):
        return None
    edge = b - a
    length = float(np.linalg.norm(edge))
    along = edge / length
    rel = walls[:, :, :2] - a
    across = rel @ out
    s = rel @ along
    near = (np.abs(across) < 0.3).all(axis=1) & (s > -0.3).all(axis=1) & (s < length + 0.3).all(axis=1)
    if not near.any():
        return None
    return s[near], walls[near][:, :, 2]


def _top_at(profile, s: float) -> float | None:
    """The highest point of the wall straight above `s` along it."""
    along, z = profile
    best = None
    for tri_s, tri_z in zip(along, z):
        lo, hi = float(tri_s.min()), float(tri_s.max())
        if hi - lo < 1e-6 or not (lo - 1e-6 <= s <= hi + 1e-6):
            continue
        # Where the vertical through s crosses the triangle's edges.
        for i in range(3):
            s0, s1 = tri_s[i], tri_s[(i + 1) % 3]
            z0, z1 = tri_z[i], tri_z[(i + 1) % 3]
            if abs(s1 - s0) < 1e-9:
                continue
            t = (s - s0) / (s1 - s0)
            if -1e-6 <= t <= 1 + 1e-6:
                top = float(z0 + t * (z1 - z0))
                best = top if best is None else max(best, top)
    return best


def _nearest_road(point, roads) -> np.ndarray | None:
    """The direction from `point` to the nearest road, or None."""
    import shapely

    if roads is None:
        return None
    here = shapely.Point(float(point[0]), float(point[1]))
    hits = roads["tree"].query(here.buffer(ROAD_REACH_M))
    if not len(hits):
        return None
    lines = roads["lines"]
    nearest = min(hits, key=lambda i: lines[int(i)].distance(here))
    target = shapely.get_coordinates(shapely.shortest_line(here, lines[int(nearest)]))[-1]
    offset = target - np.asarray(point, dtype=np.float64)
    distance = float(np.linalg.norm(offset))
    return offset / distance if distance > 1e-6 else None


def road_index(osm) -> dict | None:
    """OpenStreetMap's roads as a tree of lines, for "which way is the street"."""
    import shapely

    lines = [
        shapely.LineString(way.points)
        for way in (getattr(osm, "ways", None) or [])
        if len(way.points) >= 2 and way.tags.get("highway")
    ]
    if not lines:
        return None
    lines = np.array(lines, dtype=object)
    return {"lines": lines, "tree": shapely.STRtree(lines)}


def place_door(building, others, roads=None, terrain=None) -> dict[str, list] | None:
    """The door of one building, as triangles per mesh key, or None."""
    from .landmarks import footprint

    archetype = getattr(building, "archetype", None)
    if archetype is None:
        archetype = classify_archetype(building)
    spec = DOORS.get(archetype)
    if spec is None or getattr(building, "kind", "") in NO_DOOR_KINDS:
        return None
    width, height, leaves, key = spec
    shape = footprint(building)
    if shape is None:
        return None
    edges = _edges(shape)
    if not edges:
        return None
    longest = max(float(np.linalg.norm(b - a)) for a, b in edges)

    candidates = []
    for a, b in edges:
        length = float(np.linalg.norm(b - a))
        door_w = min(width, length - 0.6)
        if door_w < 0.75:
            continue
        along = (b - a) / length
        out = np.array([along[1], -along[0]])
        middle = (a + b) / 2.0
        if others is not None and others(middle + out * 0.8):
            continue  # a party wall
        facing = 0.0
        to_road = _nearest_road(middle, roads)
        if to_road is not None:
            facing = float(np.dot(out, to_road))
        if archetype == ARCH_BARN:
            # The gable: the short end of a long barn.
            preference = 1.0 if length <= 0.7 * longest else 0.0
        else:
            preference = length / longest
        candidates.append((facing + preference, a, b, length, door_w, along, out))
    if not candidates:
        return None
    _, a, b, length, door_w, along, out = max(candidates, key=lambda c: c[0])

    middle_s = length / 2.0
    ground = float(getattr(building, "ground_z_nap", 0.0) or 0.0)
    if terrain is not None:
        spot = (a + b) / 2.0 + out * 0.5
        try:
            sampled = float(np.asarray(terrain.sample(np.array([spot[0]]), np.array([spot[1]])))[0])
            if np.isfinite(sampled) and abs(sampled - ground) < 0.6:
                ground = sampled
        except Exception:  # noqa: BLE001 - a sampler outside its grid
            pass
    profile = _wall_profile(building, a, b, out)
    tops = []
    if profile is not None:
        for s in (middle_s - door_w / 2.0, middle_s, middle_s + door_w / 2.0):
            top = _top_at(profile, s)
            if top is not None:
                tops.append(top)
    wall_top = min(tops) if tops else float(getattr(building, "wall_top_nap", ground + 3.0))
    door_h = min(height, wall_top - ground - HEAD_ROOM_M)
    if door_h < MIN_DOOR_M:
        return None

    left = a + along * (middle_s - door_w / 2.0)
    pieces: dict[str, list] = {key: []}
    leaf_w = (door_w - LEAF_GAP_M * (leaves - 1)) / leaves
    for k in range(leaves):
        corner = left + along * (k * (leaf_w + LEAF_GAP_M))
        pieces[key].append(_box(corner, along, out, leaf_w, LEAF_DEPTH_M, ground, door_h))
    if key == "doors":
        frames = [
            _box(left - along * FRAME_M, along, out, FRAME_M, FRAME_DEPTH_M, ground, door_h + FRAME_M),
            _box(left + along * door_w, along, out, FRAME_M, FRAME_DEPTH_M, ground, door_h + FRAME_M),
            _box(left, along, out, door_w, FRAME_DEPTH_M, ground + door_h, FRAME_M),
        ]
        pieces["door_frames"] = frames
    return pieces


def door_meshes(buildings, osm=None, terrain=None) -> dict[str, np.ndarray]:
    """Every door, as triangles in RD and NAP per mesh key."""
    import shapely

    from .landmarks import footprint

    owners = list(getattr(buildings, "buildings", None) or [])
    if not owners:
        return {}
    wanted = [
        b for b in owners
        if DOORS.get(getattr(b, "archetype", None)) is not None
        and getattr(b, "kind", "") not in NO_DOOR_KINDS
    ]
    if not wanted:
        return {}
    shapes = [s for s in (footprint(b) for b in owners) if s is not None]
    tree = shapely.STRtree(np.array(shapes, dtype=object)) if shapes else None

    def others(xy) -> bool:
        if tree is None:
            return False
        return bool(len(tree.query(shapely.Point(float(xy[0]), float(xy[1])), predicate="within")))

    roads = road_index(osm)
    found: dict[str, list] = {key: [] for key in KEYS}
    doors = 0
    for building in wanted:
        pieces = place_door(building, others, roads, terrain)
        if not pieces:
            continue
        doors += 1
        for key, boxes in pieces.items():
            found[key].extend(boxes)
    out = {key: np.concatenate(boxes) for key, boxes in found.items() if boxes}
    LOG.info("doors: %d of %d closed buildings", doors, len(wanted))
    return out


__all__ = ["DOORS", "KEYS", "door_meshes", "place_door", "road_index"]
