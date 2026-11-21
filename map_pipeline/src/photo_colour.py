"""Putting every street photo in the aerial's colours.

Street photos come out of a dozen cameras on a dozen days: an overcast
morning makes red brick brown, a low sun makes it orange, and a dashcam's
white balance is anyone's guess. Read as they are, the walls of one street
would come out in as many colours as there were days of photographing.

The model already has a colour standard for everything that is not a wall:
the aerial photo, one flight, one camera, draped over the ground and the
roofs. And every street photo sees some of the same ground the aerial sees,
the pavement and the road around the camera. So each photo is compared with
the aerial over that ground, and the per-channel gain that makes the two
agree is the photo's exposure and white balance relative to the aerial.
Walls read through that gain sit in the same colours as the roofs and the
ground beside them, which is what matters for the model to look like one
place rather than a collage.

The ground is sampled in a ring around the camera, close enough to be
sharp and far enough to be past the mount of a 360 camera. Points are left
out where a wall stands in the way, and under tree crowns, where the aerial
sees leaves and the street photo sees pavement. Medians do the rest: a
parked car or a shadow is a minority of the ring.

**Brightness only, not colour.** Tried per channel over central Arnhem, the
aerial's street came out bluish wherever buildings had shaded it on the day
of the flight -- (70, 84, 110) against the street photo's red brick paving
-- and the gains swung from (0.96, 1.45, 1.77) to (1.75, 1.75, 1.61) within
one sequence. Applied, they would have turned walls blue. The brightness of
the two agreed far better: most sequences came out one stop darker than the
aerial, consistently, which is exactly the dark-wall problem. So the gain is
one number per photo, and each photo keeps its own colour balance.
"""

from __future__ import annotations

import logging

import numpy as np

from .camera import SPHERICAL, project_world
from .photos import Photo

LOG = logging.getLogger(__name__)

RING_M = (3.5, 14.0)
PANO_FLOOR_DEG = -35.0
MIN_POINTS = 60
GAIN_LIMITS = (0.4, 3.0)


class AerialColours:
    """The aerial photo, sampled at RD coordinates."""

    def __init__(self, image: np.ndarray, bbox) -> None:
        self.image = np.asarray(image, dtype=np.float32)
        self.bbox = bbox

    @classmethod
    def load(cls, path, bbox) -> "AerialColours":
        from PIL import Image

        from .imagery import _allow_large_images

        _allow_large_images()
        with Image.open(path) as image:
            return cls(np.asarray(image.convert("RGB")), bbox)

    def sample(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.image.shape[:2]
        col = (np.asarray(x) - self.bbox.xmin) / self.bbox.width * w
        row = (self.bbox.ymax - np.asarray(y)) / self.bbox.height * h
        inside = (col >= 0) & (col < w - 1) & (row >= 0) & (row < h - 1)
        out = np.zeros((len(col), 3), dtype=np.float32)
        c0 = np.floor(col[inside]).astype(np.int64)
        r0 = np.floor(row[inside]).astype(np.int64)
        fc = (col[inside] - c0)[:, None]
        fr = (row[inside] - r0)[:, None]
        img = self.image
        out[inside] = (
            img[r0, c0] * (1 - fc) * (1 - fr) + img[r0, c0 + 1] * fc * (1 - fr)
            + img[r0 + 1, c0] * (1 - fc) * fr + img[r0 + 1, c0 + 1] * fc * fr
        )
        return out, inside


def ground_ring(photo: Photo, ground_at, *, step_m: float = 1.0, azimuth_step_deg: float = 4.0):
    """Ground points around a camera, on the measured terrain."""
    radii = np.arange(RING_M[0], RING_M[1] + 1e-9, step_m)
    azimuths = np.radians(np.arange(0.0, 360.0, azimuth_step_deg))
    rr, aa = np.meshgrid(radii, azimuths)
    x = photo.x + rr.ravel() * np.sin(aa.ravel())
    y = photo.y + rr.ravel() * np.cos(aa.ravel())
    z = np.asarray(ground_at(x, y), dtype=np.float64).reshape(-1) + 0.03
    return np.column_stack([x, y, z])


def photo_gain(
    photo: Photo,
    image: np.ndarray,
    aerial: AerialColours,
    ground_at,
    *,
    faces_index=None,
    crowns=None,
) -> tuple[np.ndarray | None, dict]:
    """Per-channel gain taking this photo's colours to the aerial's.

    Returns ``(gain or None, details)``; None when too little shared ground
    was seen to say.
    """
    points = ground_ring(photo, ground_at)
    camera = photo.camera.scaled(image.shape[1], image.shape[0])
    pixels, ok = project_world(camera, photo.rotation, photo.centre, points)
    if camera.kind == SPHERICAL:
        horizontal = np.linalg.norm(points[:, :2] - photo.centre[:2], axis=1)
        elevation = np.degrees(np.arctan2(points[:, 2] - photo.centre[2], horizontal))
        ok &= elevation > PANO_FLOOR_DEG + 2.0
    if crowns:
        for crown in crowns:
            ok &= np.hypot(points[:, 0] - crown.x, points[:, 1] - crown.y) > crown.radius + 0.5
    if faces_index is not None and ok.any():
        candidates = faces_index.near(photo.x, photo.y, RING_M[1] + 2.0)
        hidden = np.zeros(len(points), dtype=bool)
        hidden[ok] = faces_index.blocked(
            photo.centre, points[ok], candidates, np.full(int(ok.sum()), -1)
        )
        ok &= ~hidden
    reference, inside = aerial.sample(points[:, 0], points[:, 1])
    ok &= inside
    details = {"ground_points": int(ok.sum())}
    if ok.sum() < MIN_POINTS:
        return None, details

    from .facade_capture import _bilinear

    seen = _bilinear(image, pixels[ok, 0], pixels[ok, 1], wrap=camera.kind == SPHERICAL)
    reference = reference[ok]
    # Neither clipped nor black in either picture.
    usable = (seen.min(axis=1) > 8) & (seen.max(axis=1) < 247) & (reference.min(axis=1) > 8) & (reference.max(axis=1) < 247)
    if usable.sum() < MIN_POINTS:
        return None, details
    from .facade_capture import luminance

    ratio = float(np.median(luminance(reference[usable]))) / max(
        float(np.median(luminance(seen[usable]))), 1.0
    )
    gain = np.full(3, np.clip(ratio, *GAIN_LIMITS))
    details.update({
        "usable": int(usable.sum()),
        "gain": round(float(gain[0]), 3),
        "photo_median": [round(float(v), 1) for v in np.median(seen[usable], axis=0)],
        "aerial_median": [round(float(v), 1) for v in np.median(reference[usable], axis=0)],
    })
    return gain.astype(np.float32), details


def smooth_gains(photos: dict[str, Photo], gains: dict[str, np.ndarray], radius_m: float = 40.0) -> dict[str, np.ndarray]:
    """Each photo's gain as the median over its sequence neighbours.

    One sequence is one camera on one day, so its exposure drifts slowly;
    a single photo whose ring happened to be all parked cars should not get
    a gain of its own. Photos with no measured neighbours keep their own.
    """
    out = {}
    for pid, gain in gains.items():
        photo = photos[pid]
        near = [
            gains[q] for q in gains
            if photos[q].sequence == photo.sequence and photo.sequence
            and np.hypot(photos[q].x - photo.x, photos[q].y - photo.y) <= radius_m
        ]
        out[pid] = np.median(np.array(near), axis=0) if len(near) >= 3 else gain
    return out


def wall_gain(
    wall_rgb, gain: float, *, ceiling: float = 185.0, strength: float = 0.7
) -> float:
    """The gain to apply to one wall's colours.

    The aerial gain lifts a wall photographed on a dull day to the
    brightness of a sunlit one. A model's walls are base colours that the
    renderer lights again, so the lift is taken at `strength` of its stops
    -- a gain of 2 becomes 1.6 -- and stops where the wall colour's
    luminance reaches `ceiling`, about the lightest render there is. Taken
    all the way, overcast brick came out salmon and beige render white.
    A gain below one, from a photo brighter than the aerial, is applied at
    the same strength, and a wall photographed in full sun is brought down
    to the ceiling even with no gain to apply.
    """
    from .facade_capture import luminance

    softened = float(gain) ** float(strength)
    level = float(luminance(np.asarray(wall_rgb, dtype=np.float64)[None])[0])
    # Never past the ceiling, whatever the gain: a wall in full sun can be
    # near white in the photo itself.
    return float(min(softened, ceiling / max(level, 1.0)))


__all__ = ["AerialColours", "ground_ring", "photo_gain", "smooth_gains", "wall_gain"]
