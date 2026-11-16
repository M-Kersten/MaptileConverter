"""Railings along the open sides of what people walk on over water.

A bridge deck over a canal and a catwalk out to a ferry berth have nothing
along their sides in the open data unless the BGT happens to draw a parapet
there; walked out of the model, they drop straight into the water. This lays
a plain railing -- posts, a handrail and a knee rail -- along every stretch of
such an edge that is over water and not already walled by a BGT barrier.

The geometry is generic and cheap, about 17 triangles a metre: posts every
`POST_EVERY_M`, rails as square tubes. It is a stand-in for the modellers to
replace with the railing of the place, in the place where it belongs.
"""

from __future__ import annotations

import numpy as np

HEIGHT_M = 1.0
KNEE_M = 0.5
POST_EVERY_M = 1.5
POST_SIZE_M = 0.06
RAIL_SIZE_M = 0.05
# An edge has to be over water for at least this long to get a railing.
MIN_RUN_M = 1.0
# A BGT wall or fence this close along an edge already guards it.
GUARDED_M = 0.6


def open_runs(edge, water, guarded=None) -> list:
    """The stretches of `edge` (a shapely line) over `water`, less `guarded`.

    `water` is the wet area; `guarded` the area within reach of a BGT barrier
    already standing there. Both shapely, both optional to be empty.
    """
    import shapely

    if edge is None or edge.is_empty:
        return []
    runs = edge.intersection(water) if water is not None else edge
    if guarded is not None and not guarded.is_empty:
        runs = runs.difference(guarded)
    merged = shapely.line_merge(runs) if not runs.is_empty else runs
    return [
        part for part in shapely.get_parts(merged)
        if part.geom_type == "LineString" and part.length >= MIN_RUN_M
    ]


def railing(points: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Triangles of a railing along `points` (N x 2), standing on `base` (N).

    Resampled so no two posts are more than `POST_EVERY_M` apart; the rails
    follow the base, so a railing on a deck that lifts at its abutment lifts
    with it.
    """
    points = np.asarray(points, dtype=np.float64)[:, :2]
    base = np.asarray(base, dtype=np.float64)
    if len(points) < 2:
        return np.zeros((0, 3, 3))
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(step)])
    length = float(along[-1])
    if length <= 0:
        return np.zeros((0, 3, 3))
    count = max(int(np.ceil(length / POST_EVERY_M)), 1)
    at = np.linspace(0.0, length, count + 1)
    xy = np.column_stack([np.interp(at, along, points[:, 0]), np.interp(at, along, points[:, 1])])
    z = np.interp(at, along, base)

    tris = []
    for (x, y), z0 in zip(xy, z):
        tris.extend(_box_post(x, y, z0, z0 + HEIGHT_M, POST_SIZE_M))
    for height in (HEIGHT_M, KNEE_M):
        for i in range(len(xy) - 1):
            tris.extend(_box_rail(
                np.r_[xy[i], z[i] + height], np.r_[xy[i + 1], z[i + 1] + height], RAIL_SIZE_M,
            ))
    return np.asarray(tris, dtype=np.float64)


def _box_post(x, y, bottom, top, size):
    h = size / 2.0
    ring = np.array([[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]])
    tris = []
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        tris.append([np.r_[a, bottom], np.r_[b, bottom], np.r_[b, top]])
        tris.append([np.r_[a, bottom], np.r_[b, top], np.r_[a, top]])
    c = [np.r_[p, top] for p in ring]
    tris.extend([[c[0], c[1], c[2]], [c[0], c[2], c[3]]])
    return tris


def _box_rail(a, b, size):
    """A square tube from `a` to `b`, its sides upright: four faces, no ends."""
    d = b - a
    flat = np.hypot(d[0], d[1])
    if flat <= 1e-9:
        return []
    side = np.array([-d[1], d[0], 0.0]) / flat * (size / 2.0)
    up = np.array([0.0, 0.0, size / 2.0])
    corners = [side + up, -side + up, -side - up, side - up]
    tris = []
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        tris.append([a + p, b + p, b + q])
        tris.append([a + p, b + q, a + q])
    return tris


def guarded_area(bbox, fetch_current, fetch_kwargs: dict):
    """Within `GUARDED_M` of any BGT wall, fence or parapet, as one shape."""
    import shapely
    from shapely.geometry import shape

    lines = []
    for collection in ("scheiding_lijn", "scheiding_vlak"):
        try:
            features, _ = fetch_current(collection, bbox, class_field="plus_type", **fetch_kwargs)
        except Exception:  # noqa: BLE001 - unguarded is the safe side
            continue
        for feature in features:
            try:
                geometry = shapely.force_2d(shape(feature.get("geometry")))
            except Exception:  # noqa: BLE001 - one broken outline
                continue
            if not geometry.is_empty:
                lines.append(geometry.boundary if geometry.geom_type.endswith("Polygon") else geometry)
    if not lines:
        return shapely.Polygon()
    return shapely.union_all(np.array(lines, dtype=object)).buffer(GUARDED_M)


__all__ = ["guarded_area", "open_runs", "railing"]
