"""Café terraces, from OpenStreetMap.

A square in summer is its terraces: tables and chairs in rows along the
facades and parasols over them. No register has them; OpenStreetMap does, in
two ways:

* ``leisure=outdoor_seating`` areas, the terrace's own outline: six round the
  Ganzenmarkt, twenty-two on the Markt in Delft. Those are filled with table
  sets on a grid turned to the outline's long side, parasols over every
  other one.
* cafés, bars and restaurants tagged ``outdoor_seating=yes`` with no outline:
  twenty-nine round the Ganzenmarkt. Each gets a row of tables along its own
  facade, on the side where the BGT says there is paving, a public street or
  square, and not a yard, a carriageway or the water.

They are summer furniture, and a model is for every season, so they are
objects of their own a designer can hide in one click.
"""

from __future__ import annotations

import logging

import numpy as np

from .street_objects import KIND_PARASOL, KIND_TABLE, SOURCE_OSM, Footprints, StreetObject

LOG = logging.getLogger(__name__)

TABLE_PITCH_M = 2.0  # one table and its chairs, centre to centre
EDGE_M = 0.6  # a table's centre this far inside the terrace's outline
PARASOL_MIN_M2 = 8.0
VENUE_REACH_M = 12.0  # a venue's node this far from its facade, at most
VENUE_ROW = (-2.0, 0.0, 2.0)  # tables along the facade, from the venue
VENUE_DEPTH_M = 1.3  # the row's distance out from the wall
OUTLINED_NEAR_M = 15.0  # a venue this close to a mapped terrace has it

# Paving a terrace may stand on: walkways of any kind, and road in anything
# but asphalt (a pedestrian street paved in brick is a road in the BGT).
ROAD_OK = {"Bakstenen", "Sierbestrating", "Tegels", "Betonstenen", "Beton_Geel", "Verharding", "Grind"}


def paving_for_terraces(areas):
    """``f(x, y) -> bool``: is this public paving a terrace may stand on?"""
    import shapely

    from .layered import ROAD, WALKWAY
    from .terrain_mesh import _polygonal

    shapes = []
    for area in areas or ():
        if getattr(area, "level", 0) != 0:
            continue
        ok = area.layer == WALKWAY or (area.layer == ROAD and area.material in ROAD_OK)
        if not ok:
            continue
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in area.rings]
        rings = [r for r in rings if len(r) >= 4]
        if rings:
            shapes.append(_polygonal(shapely.make_valid(shapely.Polygon(rings[0], rings[1:]))))
    if not shapes:
        return lambda x, y: np.zeros(len(np.atleast_1d(x)), dtype=bool)
    union = shapely.union_all(np.array(shapes, dtype=object))
    shapely.prepare(union)

    def ok(x, y):
        return shapely.contains_xy(
            union, np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
        )

    return ok


def _fill(polygon) -> tuple[list[tuple[float, float]], float]:
    """Table centres on a grid turned to the outline's long side."""
    import shapely

    rect = polygon.minimum_rotated_rectangle
    corners = np.asarray(rect.exterior.coords)[:4]
    edges = np.diff(np.vstack([corners, corners[:1]]), axis=0)
    longest = edges[int(np.argmax(np.hypot(edges[:, 0], edges[:, 1])))]
    angle = float(np.arctan2(longest[1], longest[0]))
    along = np.array([np.cos(angle), np.sin(angle)])
    across = np.array([-along[1], along[0]])
    centre = np.asarray(polygon.centroid.coords[0])
    inner = polygon.buffer(-EDGE_M)
    if inner.is_empty:
        return ([tuple(centre)] if polygon.area >= 2.0 else []), angle
    rel = corners - centre
    span_a = np.abs(rel @ along).max()
    span_c = np.abs(rel @ across).max()
    out = []
    for i in np.arange(-span_a, span_a + 1e-6, TABLE_PITCH_M):
        for j in np.arange(-span_c, span_c + 1e-6, TABLE_PITCH_M):
            p = centre + along * i + across * j
            if shapely.contains_xy(inner, p[0], p[1]):
                out.append((float(p[0]), float(p[1])))
    if not out and polygon.area >= 2.0:
        out = [tuple(centre)]
    return out, angle


def build_terraces(
    osm,
    footprints: Footprints | None,
    *,
    paving_ok=None,
    water_at=None,
) -> tuple[list[StreetObject], dict]:
    """Tables and parasols for every terrace OpenStreetMap knows of."""
    import shapely

    out: list[StreetObject] = []
    stats = {"outlined": 0, "along_facades": 0, "venues_without_room": 0, "tables": 0, "parasols": 0}
    outlines = [f for f in getattr(osm, "features", []) if f.kind == "terrace" and len(f.xy) >= 4]
    polygons = []
    for feature in outlines:
        polygon = shapely.make_valid(shapely.Polygon(feature.xy))
        if polygon.is_empty or polygon.area < 1.0:
            continue
        if footprints is not None and footprints.union is not None:
            polygon = polygon.difference(footprints.union)
        if polygon.is_empty:
            continue
        polygons.append(polygon)
        centres, angle = _fill(polygon)
        if not centres:
            continue
        stats["outlined"] += 1
        for index, (x, y) in enumerate(centres):
            out.append(StreetObject(KIND_TABLE, x, y, SOURCE_OSM, spin=angle, size=4.0))
        if polygon.area >= PARASOL_MIN_M2:
            for x, y in centres[::2]:
                out.append(StreetObject(KIND_PARASOL, x, y, SOURCE_OSM, spin=angle, size=2.6))

    covered = shapely.union_all(np.array(polygons, dtype=object)) if polygons else None
    taken = []
    for feature in getattr(osm, "features", []):
        if feature.kind != "terrace_venue" or footprints is None:
            continue
        x, y = (float(v) for v in feature.centroid)
        if covered is not None and covered.distance(shapely.Point(x, y)) < OUTLINED_NEAR_M:
            continue
        wall = footprints.wall_at(x, y, VENUE_REACH_M)
        if wall is None:
            stats["venues_without_room"] += 1
            continue
        foot, normal, along = wall
        row = [foot + normal * VENUE_DEPTH_M + along * t for t in VENUE_ROW]
        row = np.asarray(row)
        free = ~footprints.inside(row[:, 0], row[:, 1])
        if paving_ok is not None:
            free &= paving_ok(row[:, 0], row[:, 1])
            # Room for the chairs on the street side, too.
            outer = row + normal * 0.9
            free &= paving_ok(outer[:, 0], outer[:, 1])
        if water_at is not None:
            free &= ~np.array([bool(water_at(px, py)) for px, py in row])
        for other in taken:
            free &= np.hypot(row[:, 0] - other[0], row[:, 1] - other[1]) > 1.6
        if not free.any():
            stats["venues_without_room"] += 1
            continue
        stats["along_facades"] += 1
        spin = float(np.arctan2(along[1], along[0]))
        for (px, py), ok in zip(row, free):
            if ok:
                out.append(StreetObject(KIND_TABLE, float(px), float(py), SOURCE_OSM, spin=spin, size=2.0))
                taken.append((float(px), float(py)))
        middle = row[len(row) // 2]
        if free[len(row) // 2]:
            out.append(StreetObject(KIND_PARASOL, float(middle[0]), float(middle[1]), SOURCE_OSM,
                                    spin=spin, size=2.6))
    stats["tables"] = sum(1 for o in out if o.kind == KIND_TABLE)
    stats["parasols"] = sum(1 for o in out if o.kind == KIND_PARASOL)
    if out:
        LOG.info(
            "terraces: %d outlined in OpenStreetMap, %d along a café's facade, %d tables, "
            "%d parasols; %d cafés had no paving in front to put one on",
            stats["outlined"], stats["along_facades"], stats["tables"], stats["parasols"],
            stats["venues_without_room"],
        )
    return out, stats


__all__ = ["build_terraces", "paving_for_terraces"]
