"""Turns the terrain triangulation into a quad-dominant mesh.

A triangulation is what the geometry wants and quads are what an editor wants.
Blender selects, loops and subdivides along quads; a triangle fan gives you
none of that, which is why a mesh that is numerically perfect can still be
miserable to work on.

So this is the last step and not the first: the triangles are already correct
and already have their edges on the kerbs, the banks and the contours, and this
only decides which pairs of them to fuse. Nothing moves, nothing is added and
nothing is thrown away, so every accuracy figure measured on the triangles
still holds afterwards.

Two edges are never dissolved:

* **A breakline.** Fusing across one would delete the very edge the mesh exists
  to have. This is what separates the approach from a general remesher: an
  automatic quad remesher aligns edges to the curvature of the surface it is
  given, and a kerb is not a curvature feature -- it is a line from a different
  dataset that happens to lie on this surface. Told which edges those are, the
  pairing keeps every one of them.
* **A fold.** Two triangles meeting at an angle are a ridge or a ditch, and
  flattening them into one quad would smooth it away. The threshold is a
  parameter because how much of a fold counts as terrain rather than noise is a
  judgement about the area, not a fact about the mesh.

The pairing itself is a greedy maximum matching on the dual graph: score every
legal fusion by the shape of the quad it would make, take them best first, skip
any whose triangles have already been spoken for. Optimal matching is a
blossom algorithm and is not worth it here -- greedy on a Delaunay mesh reaches
the high eighties as a percentage of quads, and the last few per cent of
matching buys nothing anyone can see.
"""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger(__name__)

# A fold sharper than this stays a pair of triangles. Read it as the angle
# between the two faces: 0 is perfectly flat, and a Dutch dike face against its
# crown is around 20 degrees.
DEFAULT_MAX_FOLD_DEG = 12.0

# A quad with a corner tighter than this is worse than the two triangles it
# would replace. Anything under 20 degrees is a wedge with an extra vertex.
DEFAULT_MIN_QUAD_ANGLE_DEG = 25.0


def _quad_corners(tri_a, tri_b, u, v):
    """The four corners of the quad made by fusing two triangles.

    Both triangles are counter-clockwise, so the shared edge runs one way round
    `tri_a` and the other way round `tri_b`. Following the two outer edges of
    each in turn gives the boundary: u, then b's apex, then v, then a's apex.
    """
    apex_a = int([w for w in tri_a if w not in (u, v)][0])
    apex_b = int([w for w in tri_b if w not in (u, v)][0])
    return (u, apex_b, v, apex_a)


def _interior_angles(points):
    """The four interior angles of a polygon, in degrees."""
    n = len(points)
    out = []
    for i in range(n):
        a = points[(i - 1) % n] - points[i]
        b = points[(i + 1) % n] - points[i]
        norm = np.hypot(*a) * np.hypot(*b)
        if norm < 1e-18:
            return None
        cos = float(np.dot(a, b) / norm)
        out.append(np.degrees(np.arccos(min(max(cos, -1.0), 1.0))))
    return out


def _is_convex_ccw(points):
    """True when every turn goes the same way round, and none is degenerate."""
    n = len(points)
    for i in range(n):
        a = points[(i + 1) % n] - points[i]
        b = points[(i + 2) % n] - points[(i + 1) % n]
        if a[0] * b[1] - a[1] * b[0] <= 1e-12:
            return False
    return True


def _fold_degrees(vertices, tri_a, tri_b):
    """The angle between two triangles' faces, in degrees."""
    normals = []
    for tri in (tri_a, tri_b):
        p = vertices[list(tri)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        length = float(np.linalg.norm(n))
        if length < 1e-18:
            return 180.0
        normals.append(n / length)
    cos = float(np.dot(normals[0], normals[1]))
    return float(np.degrees(np.arccos(min(max(cos, -1.0), 1.0))))


def pair_into_quads(
    vertices: np.ndarray,
    triangles: np.ndarray,
    protected: set | None = None,
    *,
    max_fold_deg: float = DEFAULT_MAX_FOLD_DEG,
    min_quad_angle_deg: float = DEFAULT_MIN_QUAD_ANGLE_DEG,
):
    """Fuse triangle pairs into quads. Returns (loops, sizes, stats).

    `loops` is every face's vertex indices end to end and `sizes` says how long
    each face is, which is the shape Blender's mesh API wants and which lets one
    array carry both quads and the triangles that could not be paired.

    `protected` holds frozensets of two vertex indices that may not be
    dissolved -- the breaklines.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    protected = protected or set()
    if len(triangles) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), {}

    # Which two triangles share each edge, and which way round each sees it.
    sharing: dict = {}
    for index, tri in enumerate(triangles):
        for i in range(3):
            u, v = int(tri[i]), int(tri[(i + 1) % 3])
            sharing.setdefault((min(u, v), max(u, v)), []).append((index, u, v))

    candidates = []
    blocked_by_line = 0
    blocked_by_fold = 0
    blocked_by_shape = 0
    for (lo, hi), users in sharing.items():
        if len(users) != 2:
            continue  # a boundary edge, or a non-manifold one
        if frozenset((lo, hi)) in protected:
            blocked_by_line += 1
            continue
        (index_a, ua, va), (index_b, _, _) = users
        if _fold_degrees(vertices, triangles[index_a], triangles[index_b]) > max_fold_deg:
            blocked_by_fold += 1
            continue
        corners = _quad_corners(
            triangles[index_a], triangles[index_b], ua, va
        )
        flat = vertices[list(corners)][:, :2]
        if not _is_convex_ccw(flat):
            blocked_by_shape += 1
            continue
        angles = _interior_angles(flat)
        if angles is None or min(angles) < min_quad_angle_deg:
            blocked_by_shape += 1
            continue
        candidates.append((min(angles), index_a, index_b, corners))

    # Best-shaped fusions first, so a triangle that could pair two ways spends
    # itself on the better quad.
    candidates.sort(key=lambda c: -c[0])
    taken = np.zeros(len(triangles), dtype=bool)
    loops: list[int] = []
    sizes: list[int] = []
    paired = 0
    for _score, index_a, index_b, corners in candidates:
        if taken[index_a] or taken[index_b]:
            continue
        taken[index_a] = taken[index_b] = True
        loops.extend(corners)
        sizes.append(4)
        paired += 1

    for index in np.flatnonzero(~taken):
        loops.extend(int(w) for w in triangles[index])
        sizes.append(3)

    faces = paired + int((~taken).sum())
    stats = {
        "faces": faces,
        "quads": paired,
        "triangles": int((~taken).sum()),
        "quad_fraction": round(paired / max(faces, 1), 4),
        "edges_kept_for_breaklines": blocked_by_line,
        "edges_kept_for_folds": blocked_by_fold,
        "edges_refused_on_shape": blocked_by_shape,
        "max_fold_deg": float(max_fold_deg),
        "min_quad_angle_deg": float(min_quad_angle_deg),
    }
    LOG.info(
        "terrain quads: %d faces, %d quads (%.0f%%), %d triangles left; "
        "%d edges kept as breaklines, %d as folds",
        faces,
        paired,
        100.0 * paired / max(faces, 1),
        stats["triangles"],
        blocked_by_line,
        blocked_by_fold,
    )
    return (
        np.asarray(loops, dtype=np.int64),
        np.asarray(sizes, dtype=np.int64),
        stats,
    )


__all__ = [
    "DEFAULT_MAX_FOLD_DEG",
    "DEFAULT_MIN_QUAD_ANGLE_DEG",
    "pair_into_quads",
]
