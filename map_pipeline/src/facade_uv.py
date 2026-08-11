"""Fitting facade textures to the walls they land on.

The texture is a tile; the wall is whatever width the building happens to be.
Something has to reconcile them, and doing it badly is what makes a generated
facade look generated: tiling at a fixed width in metres leaves the wall ending
wherever it ends, so a 5.4 m Dutch house front on a 4 m tile shows one window
and a second sliced in half by the party wall.

Measuring each wall face and rounding its bay count to a whole number fixes
that. It lives here rather than in the Blender stage for the same reason the
triangulation does: it is plain numpy, the Blender stage may only import what
Blender bundles, and maths nobody can test without a copy of Blender is maths
nobody tests. The result is independent of where the origin sits, because every
face is measured against its own left-hand edge, so computing it in RD gives
exactly what computing it in local metres would.
"""

from __future__ import annotations

import numpy as np

# Below this fraction of a tile a face is a sliver — the return of a bay
# window, a chamfered corner — and stretching a whole bay over it would squash
# a window rather than fit one.
MIN_BAY_FRACTION = 0.55

# How finely a wall's direction and position are binned when deciding which
# triangles belong to the same face. Loose enough to hold one wall together
# through floating-point noise, tight enough to keep the front of a building
# apart from its side.
DIRECTION_BIN_DEG = 2.0
OFFSET_BIN_M = 0.5


def wall_tangents(triangles: np.ndarray) -> np.ndarray:
    """Horizontal direction along each wall: its normal turned 90° about Z."""
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge1, edge2)

    tangent = np.column_stack([normals[:, 1], -normals[:, 0]])
    lengths = np.linalg.norm(tangent, axis=1)
    # A near-horizontal face has no meaningful tangent; project it along +X.
    degenerate = lengths < 1e-9
    tangent[degenerate] = [1.0, 0.0]
    lengths[degenerate] = 1.0
    return tangent / lengths[:, None]


def wall_face_ids(
    triangles: np.ndarray, building_index: np.ndarray
) -> tuple[np.ndarray, int]:
    """Group wall triangles into the flat faces they were cut from.

    A facade arrives as a triangle soup but has to be textured as one surface,
    because bay spacing is a property of the whole wall rather than of each
    triangle. Triangles belong together when they are on the same building,
    point the same way, and lie on the same line through the plan.
    """
    if len(triangles) == 0:
        return np.zeros(0, dtype=np.int64), 0

    tangent = wall_tangents(triangles)

    azimuth = np.degrees(np.arctan2(tangent[:, 1], tangent[:, 0]))
    # The modulo folds the two ways of writing due south, ±180°, onto one bin.
    # Without it a wall facing that way splits down the middle.
    direction = np.round(azimuth / DIRECTION_BIN_DEG).astype(np.int64) % int(
        round(360.0 / DIRECTION_BIN_DEG / 2)
    )

    # How far the wall's line is from the origin, measured along its normal.
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    centroid = triangles.mean(axis=1)[:, :2]
    offset = np.round((centroid * normal).sum(axis=1) / OFFSET_BIN_M).astype(np.int64)

    keys = np.column_stack([building_index.astype(np.int64), direction, offset])
    _, face = np.unique(keys, axis=0, return_inverse=True)
    face = face.reshape(-1).astype(np.int64)
    return face, int(face.max()) + 1


def fit_wall_u(
    triangles: np.ndarray,
    building_index: np.ndarray,
    tile_width: np.ndarray,
) -> np.ndarray:
    """U per triangle corner, a whole number of bays across every wall face.

    Each face is measured, its bay count rounded, and the tile stretched
    slightly to fit: a 5.4 m house front gets one bay of 5.4 m or two of 2.7 m,
    and both its corners land on a tile edge instead of through a window.

    Faces too narrow for even one bay are centred on the tile seam, which is
    the blank pier between windows, so a half-metre sliver shows brick.
    """
    if len(triangles) == 0:
        return np.zeros(0)

    tangent = wall_tangents(triangles)
    corners = triangles.reshape(-1, 3)
    metres = (corners[:, :2] * np.repeat(tangent, 3, axis=0)).sum(axis=1)

    face, n_faces = wall_face_ids(triangles, building_index)
    face_per_corner = np.repeat(face, 3)

    low = np.full(n_faces, np.inf)
    high = np.full(n_faces, -np.inf)
    np.minimum.at(low, face_per_corner, metres)
    np.maximum.at(high, face_per_corner, metres)

    # Every triangle on a face belongs to one building, so any of them can
    # speak for the face's tile width.
    face_tile = np.zeros(n_faces)
    face_tile[face] = tile_width[building_index]

    span = np.maximum(high - low, 1e-6)
    bays = np.maximum(1.0, np.round(span / face_tile))
    fitted = span / bays

    narrow = fitted < MIN_BAY_FRACTION * face_tile
    scale = np.where(narrow, face_tile, fitted)
    origin = np.where(narrow, 0.5 * (low + high), low)

    return (metres - origin[face_per_corner]) / scale[face_per_corner]


def tile_widths(archetype: np.ndarray, facade_cfg: dict) -> np.ndarray:
    """Metres of wall one tile covers, per building.

    A church bay, a warehouse bay and a house front are all different widths,
    and giving them one spacing is most of why the first two looked wrong.
    """
    # Matches src/buildings.py.
    ARCH_INDUSTRIAL, ARCH_MONUMENTAL = 4, 5

    width = np.full(len(archetype), float(facade_cfg["tile_width_m"]))
    width = np.where(
        archetype == ARCH_MONUMENTAL,
        float(facade_cfg.get("monumental_tile_m", 7.0)),
        width,
    )
    return np.where(
        archetype == ARCH_INDUSTRIAL,
        float(facade_cfg.get("industrial_tile_m", 9.0)),
        width,
    )


__all__ = [
    "MIN_BAY_FRACTION",
    "fit_wall_u",
    "tile_widths",
    "wall_face_ids",
    "wall_tangents",
]
