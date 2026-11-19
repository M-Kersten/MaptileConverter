"""The harbour's parts: cranes, containers, bollards, ladders, piles and rails.

Plain geometry, each part a few boxes and beams: stand-ins with the right
size in the right place, cheap enough to stand by the hundred, for the
modellers to replace where a place deserves better. Every function returns
triangles in RD and NAP; the cranes return a list of ``(triangles, paint)``,
`paint` a name out of `PAINTS`, because a crane is more than one colour.

`harbour` decides where they stand and how large they are.
"""

from __future__ import annotations

import numpy as np

# The harbour's paints: name -> (rgb, roughness). A container and a crane
# take one each; the rest of a kind keeps the kind's own material.
PAINTS = {
    "Container_Rood": ((0.42, 0.06, 0.04), 0.6),
    "Container_Blauw": ((0.04, 0.12, 0.3), 0.6),
    "Container_Groen": ((0.05, 0.22, 0.12), 0.6),
    "Container_Oranje": ((0.65, 0.2, 0.03), 0.6),
    "Container_Grijs": ((0.35, 0.36, 0.37), 0.6),
    "Container_Wit": ((0.75, 0.75, 0.72), 0.6),
    "Container_Geel": ((0.7, 0.5, 0.05), 0.6),
    "Container_Bruin": ((0.2, 0.08, 0.05), 0.6),
    "Kraan_Blauw": ((0.03, 0.1, 0.32), 0.5),
    "Kraan_Oranje": ((0.75, 0.22, 0.03), 0.5),
    "Kraan_Groen": ((0.06, 0.28, 0.14), 0.5),
    "Kraan_Geel": ((0.8, 0.58, 0.04), 0.5),
    "Kraan_Rood": ((0.5, 0.05, 0.04), 0.5),
    "Kraan_Huis": ((0.72, 0.73, 0.74), 0.5),
    "Kraan_Kabel": ((0.05, 0.05, 0.06), 0.5),
}
PAINT_NAMES = list(PAINTS)
CONTAINER_PAINTS = [n for n in PAINT_NAMES if n.startswith("Container_")]
CRANE_PAINTS = ["Kraan_Blauw", "Kraan_Oranje", "Kraan_Groen", "Kraan_Geel", "Kraan_Rood"]
# How often each container colour turns up, in the order of CONTAINER_PAINTS.
CONTAINER_WEIGHTS = np.array([5, 5, 2, 3, 3, 2, 1, 4], dtype=np.float64)

# An ISO container: 8 ft wide, 8 ft 6 in high, 20 or 40 ft long.
CONTAINER_WIDTH_M = 2.44
CONTAINER_HEIGHT_M = 2.59
CONTAINER_20_M = 6.06
CONTAINER_40_M = 12.19

BOLLARD_HEIGHT_M = 0.6
LADDER_WIDTH_M = 0.45
RUNG_EVERY_M = 0.3
RAIL_WIDTH_M = 0.12

_EMPTY = np.zeros((0, 3, 3))


def pick(x: float, y: float, count: int, weights=None) -> int:
    """A choice out of `count` that depends only on where a thing stands, so
    a re-run paints every container and crane the colour it had."""
    seed = (int(round(x * 10.0)) * 73856093) ^ (int(round(y * 10.0)) * 19349663)
    rng = np.random.default_rng(seed & 0x7FFFFFFF)
    if weights is None:
        return int(rng.integers(count))
    w = np.asarray(weights, dtype=np.float64)
    return int(rng.choice(count, p=w / w.sum()))


def box(centre, along, length: float, width: float, bottom: float, top: float) -> np.ndarray:
    """An upright box: `centre` and `along` (a unit vector) in plan."""
    c = np.asarray(centre, dtype=np.float64)[:2]
    u = np.asarray(along, dtype=np.float64)[:2]
    v = np.array([-u[1], u[0]])
    ring = [c + sa * u * length / 2.0 + sb * v * width / 2.0
            for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    low = [np.r_[p, bottom] for p in ring]
    high = [np.r_[p, top] for p in ring]
    tris = [[high[0], high[1], high[2]], [high[0], high[2], high[3]],
            [low[0], low[2], low[1]], [low[0], low[3], low[2]]]
    for i in range(4):
        j = (i + 1) % 4
        tris.append([low[i], low[j], high[j]])
        tris.append([low[i], high[j], high[i]])
    return np.array(tris, dtype=np.float64)


def beam(a, b, width: float, height: float | None = None) -> np.ndarray:
    """A square beam from `a` to `b` in space, any way up, its ends closed."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    height = width if height is None else height
    d = b - a
    length = float(np.linalg.norm(d))
    if length <= 1e-9:
        return _EMPTY
    d = d / length
    side = np.cross([0.0, 0.0, 1.0], d)
    if np.linalg.norm(side) < 1e-6:  # upright: any side will do
        side = np.array([1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    up = np.cross(d, side)
    offsets = [(-side * width - up * height) / 2.0, (side * width - up * height) / 2.0,
               (side * width + up * height) / 2.0, (-side * width + up * height) / 2.0]
    near = [a + o for o in offsets]
    far = [b + o for o in offsets]
    tris = [[near[0], near[2], near[1]], [near[0], near[3], near[2]],
            [far[0], far[1], far[2]], [far[0], far[2], far[3]]]
    for i in range(4):
        j = (i + 1) % 4
        tris.append([near[i], near[j], far[j]])
        tris.append([near[i], far[j], far[i]])
    return np.array(tris, dtype=np.float64)


def prism(x: float, y: float, radius: float, bottom: float, top: float, sides: int = 8) -> np.ndarray:
    """An upright post of `sides` sides with a flat top."""
    angles = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    ring = np.column_stack([x + radius * np.cos(angles), y + radius * np.sin(angles)])
    tris = []
    for i in range(sides):
        a, b = ring[i], ring[(i + 1) % sides]
        tris.append([np.r_[a, bottom], np.r_[b, bottom], np.r_[b, top]])
        tris.append([np.r_[a, bottom], np.r_[b, top], np.r_[a, top]])
        tris.append([np.r_[x, y, top], np.r_[a, top], np.r_[b, top]])
    return np.array(tris, dtype=np.float64)


def bollard(x: float, y: float, ground: float) -> np.ndarray:
    """A mooring bollard: a stem and a wider head, cast iron."""
    neck = ground + BOLLARD_HEIGHT_M - 0.18
    return np.concatenate([
        prism(x, y, 0.2, ground, neck),
        prism(x, y, 0.3, neck, ground + BOLLARD_HEIGHT_M),
    ])


def ladder(foot, out, top: float, bottom: float) -> np.ndarray:
    """A quay ladder against a wall: `foot` in plan, `out` the way off the wall."""
    p = np.asarray(foot, dtype=np.float64)[:2]
    n = np.asarray(out, dtype=np.float64)[:2]
    t = np.array([-n[1], n[0]])
    left = p - t * LADDER_WIDTH_M / 2.0
    right = p + t * LADDER_WIDTH_M / 2.0
    parts = [
        beam(np.r_[left, bottom], np.r_[left, top], 0.05),
        beam(np.r_[right, bottom], np.r_[right, top], 0.05),
    ]
    for z in np.arange(bottom + RUNG_EVERY_M, top - 0.05, RUNG_EVERY_M):
        parts.append(beam(np.r_[left, z], np.r_[right, z], 0.03))
    return np.concatenate(parts)


def rail(points: np.ndarray, tops: np.ndarray) -> np.ndarray:
    """A crane rail along `points`, its head a few centimetres out of the paving."""
    parts = [
        beam(np.r_[points[i], tops[i] - 0.06], np.r_[points[i + 1], tops[i + 1] - 0.06],
             RAIL_WIDTH_M, 0.2)
        for i in range(len(points) - 1)
    ]
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else _EMPTY


def container(centre, along, length: float, bottom: float) -> np.ndarray:
    """One container, a finger narrower than its slot so a stack shows its joints."""
    return box(centre, along, length - 0.08, CONTAINER_WIDTH_M - 0.06,
               bottom, bottom + CONTAINER_HEIGHT_M - 0.03)


# ---- cranes ----------------------------------------------------------------


def _frame(centre, along, across, wheelbase, gauge, ground, top, leg):
    """Four legs on two rails, and a sill beam along each rail."""
    parts = []
    for side in (-1.0, 1.0):
        foot = centre + side * across * gauge / 2.0
        for end in (-1.0, 1.0):
            p = foot + end * along * wheelbase / 2.0
            parts.append(box(p, along, leg, leg, ground + 0.9, top))
        # The bogies and the beam over them, as one.
        parts.append(box(foot, along, wheelbase + 2.0, leg, ground + 0.1, ground + 0.9))
    return parts


def gantry_crane(centre, along, gauge: float, ground: float, height: float,
                 overhang: float = 0.0) -> list:
    """A portal crane across its two rails: legs, two girders, a trolley.

    The yard crane over a container stack and the gantry on a shipyard quay.
    `height` is its top above NAP, `overhang` how far the girders reach past
    the rails on either side.
    """
    centre = np.asarray(centre, dtype=np.float64)[:2]
    along = np.asarray(along, dtype=np.float64)[:2]
    across = np.array([-along[1], along[0]])
    paint = CRANE_PAINTS[pick(centre[0], centre[1], len(CRANE_PAINTS))]
    wheelbase = float(np.clip(gauge * 0.45, 5.0, 14.0))
    depth = float(np.clip((height - ground) * 0.08, 1.0, 2.5))
    girder_top = height - 2.0
    body = _frame(centre, along, across, wheelbase, gauge, ground, girder_top - depth, 0.7)
    span = gauge + 2.0 * overhang + 1.0
    for end in (-1.0, 1.0):
        body.append(box(centre + end * along * wheelbase / 2.0, across, span, 1.0,
                        girder_top - depth, girder_top))
        # A tie along the rail under the girders, on either side.
        body.append(box(centre + end * across * gauge / 2.0, along, wheelbase, 0.6,
                        girder_top - depth, girder_top - depth + 0.8))
    offset = (pick(centre[0], centre[1], 5) - 2) * gauge / 8.0
    trolley = box(centre + across * offset, across, 4.0, wheelbase + 1.2, girder_top, height)
    cab = box(centre + across * (offset + 3.0), across, 2.0, 2.0,
              girder_top - depth - 2.2, girder_top - depth)
    return [(np.concatenate(body), paint), (np.concatenate([trolley, cab]), "Kraan_Huis")]


def jib_crane(centre, along, gauge: float, ground: float, height: float, tip) -> list:
    """A slewing harbour crane: a portal on the rails, a house, a jib.

    `tip` is where the jib's head is, in space; `height` the top above NAP.
    """
    centre = np.asarray(centre, dtype=np.float64)[:2]
    along = np.asarray(along, dtype=np.float64)[:2]
    across = np.array([-along[1], along[0]])
    tip = np.asarray(tip, dtype=np.float64)
    paint = CRANE_PAINTS[pick(centre[0], centre[1], len(CRANE_PAINTS))]
    rise = height - ground
    wheelbase = float(np.clip(gauge, 5.0, 10.0))
    portal = ground + float(np.clip(0.25 * rise, 4.5, 10.0))
    body = _frame(centre, along, across, wheelbase, gauge, ground, portal, 0.7)
    body.append(box(centre, along, wheelbase + 0.7, gauge + 0.7, portal, portal + 0.8))
    body.append(prism(centre[0], centre[1], 1.4, portal + 0.8, portal + 2.2))

    reach = tip[:2] - centre
    flat = float(np.linalg.norm(reach))
    way = reach / flat if flat > 1e-6 else across
    house_floor = portal + 2.2
    house_top = house_floor + 3.5
    house = box(centre - way * 1.5, way, 8.0, 4.0, house_floor, house_top)
    weight = box(centre - way * 6.5, way, 2.0, 3.6, house_floor + 0.3, house_top - 0.8)

    pivot = np.r_[centre + way * 2.5, house_top - 1.0]
    apex = np.r_[centre - way * 1.0, max(house_top + 5.0, pivot[2] + 0.6 * (tip[2] - pivot[2]))]
    side = np.array([-way[1], way[0]])
    for s in (-1.0, 1.0):
        body.append(beam(np.r_[centre + way * 2.0 + s * side * 1.6, house_top], apex, 0.4))
        body.append(beam(np.r_[centre - way * 4.5 + s * side * 1.6, house_top], apex, 0.4))
    body.append(beam(pivot, tip, 1.2, 1.0))
    cables = [beam(apex, tip, 0.15)]
    hook = max(ground + 2.0, portal)
    if tip[2] - hook > 1.0:
        cables.append(beam(tip, np.r_[tip[:2], hook], 0.1))
        cables.append(box(tip[:2], way, 0.6, 0.6, hook - 0.8, hook))
    return [
        (np.concatenate(body), paint),
        (house, "Kraan_Huis"),
        (np.concatenate([weight, *cables]), "Kraan_Kabel"),
    ]


def quay_crane(centre, along, water, gauge: float, ground: float, height: float, *,
               boom_height: float, outreach: float, backreach: float, boom_up: bool) -> list:
    """A ship-to-shore container crane.

    `water` is the unit vector across the rails towards the ship. With the
    boom down it reaches `outreach` past the waterside rail at `boom_height`
    and `height` is the top of the A-frame; with the boom up, `height` is the
    boom's head.
    """
    centre = np.asarray(centre, dtype=np.float64)[:2]
    along = np.asarray(along, dtype=np.float64)[:2]
    water = np.asarray(water, dtype=np.float64)[:2]
    paint = CRANE_PAINTS[pick(centre[0], centre[1], len(CRANE_PAINTS))]
    wheelbase = float(np.clip(gauge * 0.6, 14.0, 20.0))
    leg = 1.3
    girder = 2.2
    body = _frame(centre, along, water, wheelbase, gauge, ground, boom_height, leg)
    # The portal: a beam between each pair of legs, across and along.
    portal = ground + min(14.0, 0.4 * (boom_height - ground))
    for end in (-1.0, 1.0):
        body.append(box(centre + end * along * wheelbase / 2.0, water, gauge, 1.0,
                        portal - 1.2, portal))
        body.append(box(centre + end * water * gauge / 2.0, along, wheelbase, 1.0,
                        boom_height - 1.5, boom_height))
    sea = centre + water * gauge / 2.0
    land = centre - water * gauge / 2.0
    # The fixed girder, from the back reach to the hinge at the waterside legs.
    hinge = sea + water * 2.0
    back = land - water * backreach
    top = boom_height + girder
    body.append(box((hinge + back) / 2.0, water, float(np.linalg.norm(hinge - back)), 3.0,
                    boom_height, top))
    if boom_up:
        length = max((height - top) / np.sin(np.radians(80.0)), 10.0)
        head = np.r_[hinge + water * length * np.cos(np.radians(80.0)), top + length * np.sin(np.radians(80.0))]
        apex_z = top + 0.45 * (head[2] - top)
    else:
        length = max(outreach - 2.0, 10.0)
        head = np.r_[hinge + water * length, top - girder / 2.0]
        apex_z = max(height, top + 8.0)
    body.append(beam(np.r_[hinge, top - girder / 2.0], head, 3.0, girder))
    apex = np.r_[sea - water * 2.0, apex_z]
    for end in (-1.0, 1.0):
        body.append(beam(np.r_[sea + end * along * wheelbase / 2.0, top], apex, 0.9))
        body.append(beam(np.r_[land + end * along * wheelbase / 2.0, top], apex, 0.6))
    stays = [beam(apex, np.r_[back, top], 0.3)]
    start = np.r_[hinge, top - girder / 2.0]
    for share in (0.5, 0.95):
        stays.append(beam(apex, start + share * (head - start), 0.3))
    house = box(land - water * min(backreach, 8.0) / 2.0, water, 12.0, 6.0, top, top + 5.0)
    cab_at = centre + water * (gauge / 4.0)
    cab = box(cab_at, water, 5.0, 4.0, boom_height - 3.5, boom_height)
    return [
        (np.concatenate(body), paint),
        (np.concatenate([house, cab]), "Kraan_Huis"),
        (np.concatenate(stays), "Kraan_Kabel"),
    ]


__all__ = [
    "BOLLARD_HEIGHT_M", "CONTAINER_20_M", "CONTAINER_40_M", "CONTAINER_HEIGHT_M",
    "CONTAINER_PAINTS", "CONTAINER_WEIGHTS", "CONTAINER_WIDTH_M", "PAINTS", "PAINT_NAMES",
    "beam", "bollard", "box", "container", "gantry_crane", "jib_crane", "ladder", "pick",
    "prism", "quay_crane", "rail",
]
