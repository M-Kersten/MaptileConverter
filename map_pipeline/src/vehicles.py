"""Cars and boats, placed where the BGT says they belong.

Neither is a dataset. Nobody publishes where the cars are parked, and nobody
publishes where the boats are moored. But the BGT publishes the two things that
determine both, and they turn out to be enough:

*Parking bays.* ``wegdeel`` carries ``functie = parkeervlak``, one polygon per
run of bays. Over Utrecht centre the median bay is 11 m long and 2.6 m across,
which is a parallel bay holding two cars. The polygon's own shape says how the
cars in it are oriented: a strip 2.6 m across can only hold cars end to end,
one 5 m across can only hold them side by side.

*Mooring posts.* ``waterinrichtingselement_punt`` carries ``plus_type =
meerpaal``. The median gap between neighbouring posts over the same area is
6.7 m, which is a boat, because that is exactly what the spacing is for. A boat
goes between each pair, on whichever side of the line the water is.

So neither is invented. Both are read off structures that exist because the
vehicle does — which is why the result lands on the real kerbs and the real
canals rather than being scattered plausibly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bgt import fetch_current, polygon_rings
from .geo import BBox

LOG = logging.getLogger(__name__)

KIND_CAR = 0
KIND_BOAT = 1
KIND_POST = 2

KIND_NAMES = {KIND_CAR: "car", KIND_BOAT: "boat", KIND_POST: "mooring post"}

# A parking bay narrower than this can only hold cars nose to tail.
PARALLEL_MAX_W = 3.4
# Wider than this and it is a car park, not a strip of bays: rows are laid out
# across it rather than a single line.
BAY_MAX_W = 7.5

CAR_LENGTH = 4.3
CAR_WIDTH = 1.8
# Kerbside cars are not bumper to bumper.
PARALLEL_PITCH = 5.6
PERPENDICULAR_PITCH = 2.6

# Mooring posts closer than this are a pair holding one boat between them;
# further apart and they belong to different boats, or to no boat at all.
MOORING_MIN_M = 3.5
MOORING_MAX_M = 18.0
BOAT_MIN_LENGTH = 3.5
BOAT_MAX_LENGTH = 20.0
BOAT_BEAM_RATIO = 0.30


@dataclass
class VehicleSet:
    """Placed vehicles, ready for the Blender stage to instance."""

    xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    z_nap: np.ndarray = field(default_factory=lambda: np.zeros(0))
    heading: np.ndarray = field(default_factory=lambda: np.zeros(0))
    length: np.ndarray = field(default_factory=lambda: np.zeros(0))
    width: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    # Which slot of the vehicle texture atlas each one samples.
    colour: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    counts: dict[str, int] = field(default_factory=dict)
    stats_by_collection: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.xy))

    def count_of(self, kind: int) -> int:
        return int((self.kind == kind).sum()) if len(self) else 0

    def stats(self) -> dict:
        return {
            "count": len(self),
            **{
                f"{name}s": self.count_of(kind)
                for kind, name in KIND_NAMES.items()
            },
            **self.counts,
        }


def principal_axes(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The long and short axes of a polygon, and how far it runs along each.

    A bay is a long thin rectangle at whatever angle the street runs, so its
    own shape is the only thing that says which way the cars in it face.
    """
    centred = ring - ring.mean(axis=0)
    # Rows of vt are the principal directions, longest first.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    long_axis, short_axis = vt[0], vt[1]

    along = centred @ long_axis
    across = centred @ short_axis
    # np.ptp rather than the array method: NumPy 2.0 removed ndarray.ptp().
    return long_axis, short_axis, float(np.ptp(along)), float(np.ptp(across))


def points_in_ring(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test, vectorised over the points.

    Bays are not all rectangles: they bend round corners and step around trees,
    and a car placed on the polygon's bounding box rather than inside the
    polygon ends up in the road.
    """
    if len(points) == 0 or len(ring) < 3:
        return np.zeros(len(points), dtype=bool)

    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)

    x0, y0 = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)

    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        if ay == by:
            continue
        straddles = (ay > y) != (by > y)
        # Where the edge crosses this point's horizontal line.
        crossing_x = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= straddles & (x < crossing_x)
    return inside


def cars_in_bay(
    ring: np.ndarray, rng: np.random.Generator, *, occupancy: float
) -> list[tuple[float, float, float]]:
    """Lay cars out in one parking polygon, as ``(x, y, heading)``."""
    long_axis, short_axis, along, across = principal_axes(ring)
    if along < CAR_LENGTH * 0.8 or across < 1.6:
        return []

    centre = ring.mean(axis=0)

    if across <= PARALLEL_MAX_W:
        # Nose to tail down the kerb: the bay is too narrow for anything else.
        rows = [0.0]
        pitch = PARALLEL_PITCH
        facing = long_axis
    elif across <= BAY_MAX_W:
        # One row of cars nose-in, so they face across the bay.
        rows = [0.0]
        pitch = PERPENDICULAR_PITCH
        facing = short_axis
    else:
        # A car park: rows of nose-in bays back to back.
        n_rows = max(1, int(across // 5.0))
        span = (n_rows - 1) * (across / n_rows)
        rows = list(np.linspace(-span / 2, span / 2, n_rows))
        pitch = PERPENDICULAR_PITCH
        facing = short_axis

    count = int(along // pitch)
    if count < 1:
        return []
    offsets = (np.arange(count) - (count - 1) / 2) * pitch

    heading = float(np.arctan2(facing[1], facing[0]))
    placed: list[tuple[float, float, float]] = []
    for row in rows:
        positions = (
            centre
            + offsets[:, None] * long_axis
            + row * short_axis
        )
        # A car whose centre is outside the polygon is in the road.
        keep = points_in_ring(positions, ring)
        keep &= rng.random(len(positions)) < occupancy
        for position in positions[keep]:
            jitter = rng.normal(0.0, 0.10, 2)
            placed.append(
                (
                    float(position[0] + jitter[0]),
                    float(position[1] + jitter[1]),
                    heading + float(rng.normal(0.0, 0.02)),
                )
            )
    return placed


def _nearest_links(xy: np.ndarray, lo: float, hi: float) -> list[tuple[int, int]]:
    """Pair each post with its nearest neighbour, when that is a boat apart.

    Linking every pair inside the range would put a third boat across a run of
    three posts, overlapping the two real ones. Nearest-only keeps a chain of
    posts a chain of boats.
    """
    if len(xy) < 2:
        return []

    deltas = xy[:, None, :] - xy[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, np.inf)

    links: set[tuple[int, int]] = set()
    for index in range(len(xy)):
        nearest = int(np.argmin(distances[index]))
        gap = distances[index, nearest]
        if lo <= gap <= hi:
            links.add((min(index, nearest), max(index, nearest)))
    return sorted(links)


def _in_any(point: np.ndarray, rings: list[np.ndarray]) -> bool:
    return any(points_in_ring(point[None, :], ring)[0] for ring in rings)


def _float_clear_of_bank(
    midpoint: np.ndarray,
    normal: np.ndarray,
    beam: float,
    water_rings: list[np.ndarray],
) -> np.ndarray | None:
    """Push a boat off the post line until its whole beam is afloat.

    A post stands at the water's edge, so a hull centred a half-beam off it
    still has its inboard side up on the quay. Testing only the centre point
    let that through; both gunwales have to be over water, and the boat is
    walked outwards until they are.
    """
    for side in (1.0, -1.0):
        offsets = np.arange(beam * 0.6 + 0.4, beam * 0.6 + 3.0, 0.4)
        fallback = None
        for offset in offsets:
            centre = midpoint + normal * side * offset
            if not _in_any(centre, water_rings):
                continue
            fallback = fallback if fallback is not None else centre
            inboard = centre - normal * side * beam * 0.5
            outboard = centre + normal * side * beam * 0.5
            if _in_any(inboard, water_rings) and _in_any(outboard, water_rings):
                return centre
        if fallback is not None:
            # A canal narrower than the boat is beamy: better moored a little
            # over the edge than not there at all.
            return fallback
    return None


def boats_between_posts(
    xy: np.ndarray,
    water_rings: list[np.ndarray],
    rng: np.random.Generator,
    *,
    occupancy: float,
) -> list[tuple[float, float, float, float, float]]:
    """A boat per pair of posts, as ``(x, y, heading, length, beam)``.

    The posts stand on the bank, so the boat is pushed off the line between
    them, towards whichever side is water. A post with water on neither side is
    not a mooring post in any useful sense and is left alone.
    """
    placed = []
    for first, second in _nearest_links(xy, MOORING_MIN_M, MOORING_MAX_M):
        start, end = xy[first], xy[second]
        gap = float(np.linalg.norm(end - start))
        along = (end - start) / gap
        normal = np.array([-along[1], along[0]])

        # A boat moors between its posts with a little to spare.
        length = float(np.clip(gap * 0.85, BOAT_MIN_LENGTH, BOAT_MAX_LENGTH))
        beam = float(np.clip(length * BOAT_BEAM_RATIO, 1.2, 5.0))
        midpoint = 0.5 * (start + end)

        chosen = _float_clear_of_bank(midpoint, normal, beam, water_rings)
        if chosen is None or rng.random() >= occupancy:
            continue

        placed.append(
            (
                float(chosen[0]),
                float(chosen[1]),
                float(np.arctan2(along[1], along[0])),
                length,
                beam,
            )
        )
    return placed


def build_vehicles(
    bbox: BBox,
    work_dir: Path,
    *,
    vehicles_cfg: dict,
    terrain_sampler=None,
    surfaces=None,
) -> VehicleSet:
    """Fetch the bays and posts, and place what belongs on them."""
    result = VehicleSet()
    rng = np.random.default_rng(int(vehicles_cfg.get("seed", 1807)))
    fetch_kwargs = dict(
        page_limit=int(vehicles_cfg.get("page_limit", 1000)),
        timeout=float(vehicles_cfg.get("timeout_s", 180)),
        max_retries=int(vehicles_cfg.get("max_retries", 4)),
        max_pages=int(vehicles_cfg.get("max_pages", 200)),
    )

    xy: list[tuple[float, float]] = []
    heading: list[float] = []
    length: list[float] = []
    width: list[float] = []
    kind: list[int] = []

    # ---- cars --------------------------------------------------------------
    if bool(vehicles_cfg.get("cars", True)):
        features, stats = fetch_current(
            "wegdeel", bbox, class_field="functie", **fetch_kwargs
        )
        result.stats_by_collection["wegdeel"] = stats.summary()

        occupancy = float(vehicles_cfg.get("car_occupancy", 0.72))
        bays = 0
        for feature in features:
            functie = str((feature.get("properties") or {}).get("functie") or "")
            if "parkeervlak" not in functie.lower():
                continue
            for group in polygon_rings(feature.get("geometry")):
                bays += 1
                for x, y, angle in cars_in_bay(
                    group[0][:, :2], rng, occupancy=occupancy
                ):
                    xy.append((x, y))
                    heading.append(angle)
                    length.append(CAR_LENGTH)
                    width.append(CAR_WIDTH)
                    kind.append(KIND_CAR)
        result.counts["parking_bays"] = bays
        LOG.info("placed %d cars across %d parking bays", len(xy), bays)

    # ---- boats and their posts ---------------------------------------------
    if bool(vehicles_cfg.get("boats", True)):
        features, stats = fetch_current(
            "waterinrichtingselement_punt", bbox, class_field="plus_type",
            **fetch_kwargs,
        )
        result.stats_by_collection["waterinrichtingselement_punt"] = stats.summary()

        posts = [
            feature
            for feature in features
            if "meerpaal"
            in str((feature.get("properties") or {}).get("plus_type") or "").lower()
        ]
        post_xy = np.array(
            [
                feature["geometry"]["coordinates"][:2]
                for feature in posts
                if (feature.get("geometry") or {}).get("coordinates")
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        result.counts["mooring_posts"] = len(post_xy)

        water_rings = [
            body.rings[0][:, :2]
            for body in (surfaces.water if surfaces is not None else [])
            if body.rings is not None and len(body.rings)
        ]

        before = len(xy)
        if len(post_xy) and water_rings:
            for x, y, angle, boat_length, beam in boats_between_posts(
                post_xy, water_rings, rng,
                occupancy=float(vehicles_cfg.get("boat_occupancy", 0.8)),
            ):
                xy.append((x, y))
                heading.append(angle)
                length.append(boat_length)
                width.append(beam)
                kind.append(KIND_BOAT)
        elif len(post_xy):
            LOG.warning(
                "found %d mooring posts but no water surfaces to moor against; "
                "enable surfaces.water to get boats",
                len(post_xy),
            )

        for position in post_xy:
            xy.append((float(position[0]), float(position[1])))
            heading.append(0.0)
            length.append(0.25)
            width.append(0.25)
            kind.append(KIND_POST)

        LOG.info(
            "placed %d boats between %d mooring posts",
            len(xy) - before - len(post_xy),
            len(post_xy),
        )

    if not xy:
        return result

    result.xy = np.array(xy, dtype=np.float64)
    result.heading = np.array(heading, dtype=np.float64)
    result.length = np.array(length, dtype=np.float64)
    result.width = np.array(width, dtype=np.float64)
    result.kind = np.array(kind, dtype=np.int32)

    # A bay straddling the edge comes back whole, so cars laid out along it run
    # past the terrain. Buildings are kept whole in that situation because
    # cutting one opens a hole in its wall; a car has nothing to cut, and one
    # parked out over the void is just wrong.
    inside = (
        (result.xy[:, 0] >= bbox.xmin)
        & (result.xy[:, 0] <= bbox.xmax)
        & (result.xy[:, 1] >= bbox.ymin)
        & (result.xy[:, 1] <= bbox.ymax)
    )
    if not inside.all():
        dropped = int((~inside).sum())
        result.counts["dropped_outside_bbox"] = dropped
        LOG.info("dropped %d vehicles that fell outside the bbox", dropped)
        result.xy = result.xy[inside]
        result.heading = result.heading[inside]
        result.length = result.length[inside]
        result.width = result.width[inside]
        result.kind = result.kind[inside]
    # Cars get a colour each; boats and posts have one look apiece.
    result.colour = np.where(
        result.kind == KIND_CAR,
        rng.integers(0, CAR_COLOURS, len(result.xy)),
        0,
    ).astype(np.int32)

    result.z_nap = _ground_heights(result, terrain_sampler, surfaces)
    return result


# How many car colours the atlas carries.
CAR_COLOURS = 6


def _ground_heights(
    vehicles: VehicleSet, terrain_sampler, surfaces
) -> np.ndarray:
    """What each vehicle sits on: the terrain, or the water it floats in."""
    heights = np.zeros(len(vehicles), dtype=np.float64)
    if terrain_sampler is not None:
        heights = np.asarray(
            terrain_sampler(vehicles.xy[:, 0], vehicles.xy[:, 1]), dtype=np.float64
        )

    # A boat sits on its canal, which is metres away from the bank the terrain
    # samples. Levels across one area span metres, so it has to be per body.
    if surfaces is not None and surfaces.water:
        afloat = vehicles.kind == KIND_BOAT
        for index in np.flatnonzero(afloat):
            point = vehicles.xy[index][None, :]
            for body in surfaces.water:
                if len(body.rings) and points_in_ring(point, body.rings[0][:, :2])[0]:
                    heights[index] = body.level_nap
                    break
    return heights


def save_vehicles(vehicles: VehicleSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xy=vehicles.xy,
        z_nap=vehicles.z_nap,
        heading=vehicles.heading,
        length=vehicles.length,
        width=vehicles.width,
        kind=vehicles.kind,
        colour=vehicles.colour,
    )
    LOG.info("wrote %s (%d vehicles)", path, len(vehicles))
    return path


def write_spawn_list(
    vehicles: VehicleSet, geo, out_dir: Path, filename: str = "vehicles.json"
) -> Path:
    """Write the spawn list, so real car models can replace the boxes.

    Same idea as ``trees.json``: the baked geometry is there so the FBX looks
    right on its own, but anyone with proper vehicle prefabs wants positions
    and headings, not boxes.
    """
    import json

    records = []
    for index in range(len(vehicles)):
        x, y = vehicles.xy[index]
        local_x, local_y = geo.rd_to_local(float(x), float(y))
        records.append(
            {
                "kind": KIND_NAMES[int(vehicles.kind[index])],
                "rd": [round(float(x), 3), round(float(y), 3)],
                # Unity Z is RD northing; Y comes from the ground height.
                "local": {"x": round(local_x, 3), "z": round(local_y, 3)},
                "ground_z_nap": round(float(vehicles.z_nap[index]), 3),
                # Unity's Y rotation is clockwise from +Z, which is north; the
                # heading is held here as anticlockwise from east.
                "heading_deg": round(
                    (90.0 - float(np.degrees(vehicles.heading[index]))) % 360.0, 1
                ),
                "length_m": round(float(vehicles.length[index]), 2),
                "width_m": round(float(vehicles.width[index]), 2),
            }
        )

    payload = {
        "count": len(records),
        "crs": "EPSG:28992",
        "note": (
            "local.x / local.z are metres in the model frame; add "
            "ground_z_nap minus metadata.ground_z_offset_nap for local Y. "
            "heading_deg is a Unity Y rotation, clockwise from north"
        ),
        "source": (
            "cars laid out in BGT wegdeel parkeervlak polygons, boats between "
            "BGT meerpaal mooring posts"
        ),
        "vehicles": records,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    LOG.info("wrote %s", path)
    return path


__all__ = [
    "CAR_COLOURS",
    "KIND_BOAT",
    "KIND_CAR",
    "KIND_NAMES",
    "KIND_POST",
    "VehicleSet",
    "boats_between_posts",
    "build_vehicles",
    "cars_in_bay",
    "points_in_ring",
    "principal_axes",
    "save_vehicles",
    "write_spawn_list",
]
