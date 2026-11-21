"""A photo pinned to a wall by hand: four corners, one exact pose.

For the photos nothing else can place: a courtyard with no GPS fix, a phone
whose compass was off by forty degrees, an old picture from a shoebox. A
person marks the four corners of one wall in the photo -- top left, top
right, bottom right, bottom left, as seen -- and points at that wall on the
map. The model knows where those four corners are in the world: the wall's
two ends, at the street and at the eaves. Four points on one plane, seen
through a known lens, fix the camera completely.

The maths is the planar case of camera resection. The four image points and
the four wall points are related by a homography; with the lens known, its
first two columns are the wall's two axes seen from the camera and its third
is where the wall's corner is, which is the pose. When the photo says nothing
about its lens, the focal length is found from the same homography: the
wall's two axes must come out at right angles, and only one focal length
makes them.

A pinned photo is placed by a person, so it is trusted for the wall it was
pinned to and is not moved by the automatic alignment.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from .camera import PERSPECTIVE, Camera, unproject_camera
from .photos import POSE_MANUAL, Photo

LOG = logging.getLogger(__name__)


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 3x3 homography taking four or more 2D points `src` onto `dst`."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h


def focal_from_homography(h: np.ndarray) -> float | None:
    """The focal length, in the units of the image coordinates `h` maps to,
    that makes the plane's two axes perpendicular. None if no real one."""
    h1, h2 = h[:, 0], h[:, 1]
    denom = h1[2] * h2[2]
    if abs(denom) < 1e-12:
        return None
    f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / denom
    return float(np.sqrt(f2)) if f2 > 0 else None


def _virtual_frame(rays: np.ndarray) -> np.ndarray:
    """A rotation whose forward axis is the mean of `rays` (camera frame)."""
    forward = rays.mean(axis=0)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, -1.0, 0.0])  # camera frame: y is down
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack([right, down, forward])


def pose_from_corners(
    camera: Camera,
    pixels: np.ndarray,
    corners_world: np.ndarray,
    *,
    estimate_focal: bool = False,
):
    """Camera pose from four image points of four coplanar world points.

    `corners_world` are the wall's corners in the order `pixels` were
    clicked, all on one vertical plane. Returns ``(rotation, centre,
    camera, reprojection_px)``; `camera` has the focal length used, which
    differs from the given one when `estimate_focal` found a better one.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    world = np.asarray(corners_world, dtype=np.float64)
    origin = world[3]  # bottom left
    along = world[2] - world[3]
    along[2] = 0.0
    width = float(np.linalg.norm(along))
    a_axis = along / width
    b_axis = np.array([0.0, 0.0, 1.0])
    plane = np.column_stack([(world - origin) @ a_axis, (world - origin) @ b_axis])

    if estimate_focal and camera.kind == PERSPECTIVE:
        centred = (pixels - [camera.width / 2.0, camera.height / 2.0]) / camera.size
        f = focal_from_homography(homography(plane, centred))
        if f is not None and 0.25 < f < 4.0:
            camera = Camera(camera.kind, camera.width, camera.height, f, camera.k1, camera.k2)

    rays = unproject_camera(camera, pixels)
    q = _virtual_frame(rays)
    local = rays @ q.T
    image = local[:, :2] / local[:, 2:3]
    h = homography(plane, image)
    scale = 1.0 / np.linalg.norm(h[:, 0])
    r1, r2, t = h[:, 0] * scale, h[:, 1] * scale, h[:, 2] * scale
    if t[2] < 0:  # the wall must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    r_plane = u @ vt
    if np.linalg.det(r_plane) < 0:
        r_plane = u @ np.diag([1, 1, -1]) @ vt
    world_axes = np.column_stack([a_axis, b_axis, np.cross(a_axis, b_axis)])
    rotation_virtual = r_plane @ world_axes.T
    centre = origin - rotation_virtual.T @ t
    rotation = q.T @ rotation_virtual

    from .camera import project_camera

    # In front of the camera and back on the clicks; a corner may lie just
    # past the edge of the frame, where a person clicks the edge.
    in_camera = (world - centre) @ rotation.T
    reprojected, _ = project_camera(camera, in_camera)
    in_front = np.all(in_camera[:, 2] > 0) if camera.kind != "spherical" else True
    error = (
        float(np.max(np.linalg.norm(reprojected - pixels, axis=1)))
        if in_front and np.all(np.isfinite(reprojected)) else float("inf")
    )
    return rotation, centre, camera, error


def wall_corners(face, building) -> np.ndarray:
    """The four corners of a wall as a person sees them: top left, top
    right, bottom right, bottom left, from outside.

    The top of each end is the wall's own height there, read from its
    triangles: that is the roofline someone clicks. The building's eaves
    height is not, on a stepped building whose taller part is elsewhere,
    and a gable's apex is in the middle, not at the ends. The bottom is the
    street.
    """
    tangent = (face.end - face.start) / max(face.width_m, 1e-9)
    corners = face.triangles.reshape(-1, 3)
    along = (corners[:, :2] - face.start) @ tangent
    near_start = along <= along.min() + 0.3
    near_end = along >= along.max() - 0.3
    top_start = float(corners[near_start, 2].max())
    top_end = float(corners[near_end, 2].max())
    bottom = float(building.ground_z_nap)
    (x0, y0), (x1, y1) = face.start, face.end
    return np.array([[x0, y0, top_start], [x1, y1, top_end], [x1, y1, bottom], [x0, y0, bottom]])


def nearest_face(faces, x: float, y: float, *, exposed_only: bool = True) -> int | None:
    """The wall face nearest a point on the map."""
    best, best_d = None, np.inf
    for i, face in enumerate(faces):
        if exposed_only and not face.exposed:
            continue
        a, b = face.start, face.end
        span = b - a
        t = np.clip(((np.array([x, y]) - a) @ span) / max(float(span @ span), 1e-9), 0.0, 1.0)
        d = float(np.linalg.norm(np.array([x, y]) - (a + t * span)))
        if d < best_d:
            best, best_d = i, d
    return best if best_d < 15.0 else None


def apply_pin(photo: Photo, faces, buildings, *, max_error_px: float = 25.0):
    """The photo placed by its pin, or ``(None, reason)``."""
    pin = photo.pin or {}
    corners = pin.get("corners")
    if not corners or len(corners) != 4:
        return None, "a pin needs four corners"
    if "x" in pin and "y" in pin:
        x, y = float(pin["x"]), float(pin["y"])
    elif "lat" in pin and "lon" in pin:
        from .photos import lonlat_to_rd

        x, y = lonlat_to_rd(float(pin["lon"]), float(pin["lat"]))
    else:
        return None, "a pin needs the wall's place on the map"
    index = nearest_face(faces, x, y)
    if index is None:
        return None, "no wall within 15 m of where the pin points"
    face = faces[index]
    building = buildings.buildings[face.building]
    world = wall_corners(face, building)
    rotation, centre, camera, error = pose_from_corners(
        photo.camera, np.asarray(corners, dtype=np.float64), world,
        estimate_focal="no focal length" in " ".join(photo.notes),
    )
    if not np.isfinite(error) or error > max_error_px:
        return None, (
            f"the four corners do not fit that wall (off by {error:.0f} px): "
            f"check their order, top left first, clockwise"
        )
    placed = replace(
        photo,
        rotation=rotation,
        x=float(centre[0]), y=float(centre[1]), z=float(centre[2]),
        camera=camera,
        pose=POSE_MANUAL,
        heading_known=True,
        verified=True,
        verified_buildings=[building.identifier],
        position_sigma_m=0.3,
        height_m=None,
        notes=[n for n in photo.notes if "no heading" not in n] + [f"pinned to {building.identifier}"],
    )
    return placed, ""


__all__ = [
    "apply_pin",
    "focal_from_homography",
    "homography",
    "nearest_face",
    "pose_from_corners",
    "wall_corners",
]
