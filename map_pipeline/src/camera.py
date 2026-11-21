"""Camera models for street photos: where a point in the world lands in a picture.

Three lenses cover what Mapillary serves and what people bring back from a
walk: an ordinary *perspective* camera (phones, dashcams), a *fisheye*
(action cameras), and the *spherical* 360 photo stored as an equirectangular
image. The conventions are OpenSfM's, because that is what Mapillary's own
reconstruction hands out and translating them anywhere but here is how a
camera ends up looking backwards:

* The camera frame has x to the right of the image, y down, and z forward
  out of the lens.
* A rotation maps a *world* vector into that camera frame, ``x_cam = R @
  (X - C)`` for a camera at ``C``. Its rows are therefore the camera's right,
  down and forward axes written in world coordinates.
* Focal length is normalised by the larger image side, so the same number
  describes the 1024 px thumbnail and the 4000 px original.
* Normalised image coordinates are measured from the image centre in units
  of that larger side; pixels are ``centre + normalised * max(width, height)``.

Plain numpy, no OpenCV: the pipeline only imports what it already depends on,
and maths nobody can run without an extra native library is maths nobody
tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

PERSPECTIVE = "perspective"
FISHEYE = "fisheye"
SPHERICAL = "spherical"
KINDS = (PERSPECTIVE, FISHEYE, SPHERICAL)


@dataclass(frozen=True)
class Camera:
    """The lens and sensor of one photo. Pose lives on the photo."""

    kind: str
    width: int
    height: int
    # Normalised by max(width, height). Unused for spherical images.
    focal: float = 0.8
    k1: float = 0.0
    k2: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"camera kind must be one of {KINDS}, got {self.kind!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")

    @property
    def size(self) -> int:
        return max(self.width, self.height)

    def scaled(self, width: int, height: int) -> "Camera":
        """The same lens on a downsized copy of the image.

        Normalised focal and distortion do not change with resolution, which
        is the point of normalising them; only the pixel grid does.
        """
        return Camera(self.kind, int(width), int(height), self.focal, self.k1, self.k2)

    def pixels_per_radian(self) -> float:
        """Angular resolution at the image centre."""
        if self.kind == SPHERICAL:
            return self.width / (2.0 * np.pi)
        return self.focal * self.size

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "focal": self.focal,
            "k1": self.k1,
            "k2": self.k2,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        return cls(
            kind=str(data["kind"]),
            width=int(data["width"]),
            height=int(data["height"]),
            focal=float(data.get("focal", 0.8)),
            k1=float(data.get("k1", 0.0)),
            k2=float(data.get("k2", 0.0)),
        )


def rotation_from_axis_angle(vector) -> np.ndarray:
    """Rodrigues: a 3-vector whose length is the angle in radians."""
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def axis_angle_from_rotation(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rotation_from_axis_angle`."""
    R = np.asarray(R, dtype=np.float64)
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # Half a turn: the antisymmetric part vanishes, read the axis off the
        # symmetric part instead.
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        # Recover signs from the off-diagonal terms, anchored on the largest.
        i = int(np.argmax(axis))
        for j in range(3):
            if j != i and R[i, j] + R[j, i] < 0:
                axis[j] = -axis[j]
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * np.sin(angle)) * angle


def rotation_from_heading(
    heading_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0
) -> np.ndarray:
    """World-to-camera rotation for a camera looking along a compass heading.

    Heading is clockwise from the world's +y (north), pitch is up from the
    horizon, roll turns the image clockwise as seen by the photographer. The
    world is x east, y north, z up, which RD with NAP heights is.
    """
    h, p, r = np.radians([heading_deg, pitch_deg, roll_deg])
    forward = np.array([np.sin(h) * np.cos(p), np.cos(h) * np.cos(p), np.sin(p)])
    right = np.array([np.cos(h), -np.sin(h), 0.0])
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    if abs(r) > 0:
        # Roll about the camera's own forward axis.
        c, s = np.cos(r), np.sin(r)
        roll = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        R = roll @ R
    return R


def heading_of(R: np.ndarray) -> float:
    """Compass heading of the camera's forward axis, degrees clockwise from +y."""
    forward = np.asarray(R)[2]
    return float(np.degrees(np.arctan2(forward[0], forward[1])) % 360.0)


def _distort(camera: Camera, r2: np.ndarray) -> np.ndarray:
    return 1.0 + camera.k1 * r2 + camera.k2 * r2 * r2


@lru_cache(maxsize=1024)
def max_undistorted_radius(camera: Camera) -> float:
    """How far off-axis the distortion polynomial still means anything.

    ``r (1 + k1 r^2 + k2 r^4)`` grows with r only up to where its derivative
    ``1 + 3 k1 r^2 + 5 k2 r^4`` reaches zero. Past that the curve folds back,
    and a point well outside the lens's view lands inside the image again --
    on the wrong wall. Infinite when the curve never folds.
    """
    a, b = 5.0 * camera.k2, 3.0 * camera.k1
    roots = np.roots([a, b, 1.0]) if abs(a) > 1e-15 else (
        np.array([-1.0 / b]) if abs(b) > 1e-15 else np.array([])
    )
    positive = [float(r.real) for r in np.atleast_1d(roots) if abs(r.imag) < 1e-12 and r.real > 0]
    return float(np.sqrt(min(positive))) if positive else float("inf")


def project_camera(camera: Camera, points_cam: np.ndarray):
    """Camera-frame points to pixels. Returns ``(pixels (n, 2), valid (n,))``.

    ``valid`` is false for points the lens cannot see (behind a perspective
    camera, beyond the fisheye's half-sphere) and for points that land
    outside the image. A spherical camera sees everything, so there it only
    excludes the camera's own position.
    """
    p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    w, h, size = camera.width, camera.height, camera.size

    if camera.kind == SPHERICAL:
        lon = np.arctan2(x, z)
        lat = np.arctan2(-y, np.hypot(x, z))
        u = (lon / (2.0 * np.pi) + 0.5) * w
        v = (0.5 - lat / np.pi) * h
        valid = np.linalg.norm(p, axis=1) > 1e-9
        return np.column_stack([u, v]), valid

    if camera.kind == PERSPECTIVE:
        ahead = z > 1e-6
        safe_z = np.where(ahead, z, 1.0)
        xn, yn = x / safe_z, y / safe_z
        r2 = xn * xn + yn * yn
        d = _distort(camera, r2)
        xn, yn = xn * d * camera.focal, yn * d * camera.focal
        valid = ahead & (r2 < max_undistorted_radius(camera) ** 2)
    else:  # FISHEYE, OpenSfM's equidistant model
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        d = _distort(camera, theta * theta)
        safe_r = np.where(r > 1e-12, r, 1.0)
        scale = np.where(r > 1e-12, camera.focal * d * theta / safe_r, camera.focal)
        xn, yn = x * scale, y * scale
        valid = (theta < np.radians(95.0)) & (theta < max_undistorted_radius(camera))

    u = w / 2.0 + xn * size
    v = h / 2.0 + yn * size
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return np.column_stack([u, v]), valid & inside


def _undistort_radius(camera: Camera, distorted: np.ndarray) -> np.ndarray:
    """Solve ``t * (1 + k1 t^2 + k2 t^4) = distorted`` for t, by Newton."""
    t = distorted.copy()
    for _ in range(20):
        t2 = t * t
        f = t * (1.0 + camera.k1 * t2 + camera.k2 * t2 * t2) - distorted
        df = 1.0 + 3.0 * camera.k1 * t2 + 5.0 * camera.k2 * t2 * t2
        step = f / np.where(np.abs(df) > 1e-9, df, 1e-9)
        t = t - step
        if np.all(np.abs(step) < 1e-12):
            break
    return t


def unproject_camera(camera: Camera, pixels: np.ndarray) -> np.ndarray:
    """Pixels to unit rays in the camera frame. Inverse of :func:`project_camera`."""
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    w, h, size = camera.width, camera.height, camera.size

    if camera.kind == SPHERICAL:
        lon = (px[:, 0] / w - 0.5) * 2.0 * np.pi
        lat = (0.5 - px[:, 1] / h) * np.pi
        return np.column_stack(
            [np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)]
        )

    xn = (px[:, 0] - w / 2.0) / size / camera.focal
    yn = (px[:, 1] - h / 2.0) / size / camera.focal
    rd = np.hypot(xn, yn)

    if camera.kind == PERSPECTIVE:
        r = _undistort_radius(camera, rd)
        scale = np.where(rd > 1e-12, r / np.where(rd > 1e-12, rd, 1.0), 1.0)
        rays = np.column_stack([xn * scale, yn * scale, np.ones(len(px))])
    else:
        theta = _undistort_radius(camera, rd)
        scale = np.where(rd > 1e-12, np.sin(theta) / np.where(rd > 1e-12, rd, 1.0), 1.0)
        rays = np.column_stack([xn * scale, yn * scale, np.cos(theta)])

    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def project_world(
    camera: Camera, rotation: np.ndarray, centre: np.ndarray, points: np.ndarray
):
    """World points to pixels for a camera at ``centre`` with ``rotation``."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3) - np.asarray(centre)
    return project_camera(camera, p @ np.asarray(rotation).T)


__all__ = [
    "Camera",
    "FISHEYE",
    "KINDS",
    "PERSPECTIVE",
    "SPHERICAL",
    "axis_angle_from_rotation",
    "heading_of",
    "project_camera",
    "project_world",
    "rotation_from_axis_angle",
    "rotation_from_heading",
    "unproject_camera",
]
