"""Lining a photo up with the model, and finding out when it will not line up.

Mapillary's reconstruction puts a camera within a degree or two and a metre
or two of where it was, which is good enough to decide what a photo sees and
not good enough to paint with: a degree at fifteen metres is a quarter of a
metre on the wall, half a window. Your own photos start further out, from a
phone's GPS and compass. So before any pixel is taken from a photo, the
photo is moved until the model's building edges -- corners, eaves, ridges --
sit on edges in the picture.

The second job matters as much as the first. A photo can be ranked the best
view of a wall and show something else entirely: a close-up of a plaque, the
inside of a shop window, a wall that has since been rebuilt. No pose makes
the model's edges land on edges in such a picture, and that is how it is
caught: a photo whose edges do not agree after refinement is not *verified*,
and an unverified photo never becomes a texture.

The match is a truncated chamfer distance with edge orientation: the image's
edges are split into four orientation bins and each gets a distance map, and
a point on a model edge looks itself up in the map of edges running its way.
A roofline then cannot snap onto a downpipe. The search is a coarse grid over
heading and position followed by a pattern search over all six parameters,
in plain numpy, as everything here is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

from .camera import SPHERICAL, Camera, project_camera
from .photos import Photo

LOG = logging.getLogger(__name__)

# Model edges are searched for within this many pixels, at WORK_WIDTH.
TRUNCATE_PX = 12
# A point within this of an edge running its way counts as matched.
INLIER_PX = 1.5
# Photos are aligned at this width. Halving a 2048 px photo costs a pixel of
# precision at the original size and makes the distance maps four times
# cheaper.
WORK_WIDTH = 1024
BINS = 4
BIN_HALF_WIDTH_DEG = 30.0
# Below this elevation a 360 photo shows the person carrying it, the bicycle
# or the car roof, and the mount.
PANO_FLOOR_DEG = -35.0
EDGE_SPACING_M = 0.4
MIN_EDGE_M = 0.6

# Verified when, after refinement, this share of the model's visible edge
# points sit on an edge in the picture, the fit clearly beats the model moved
# off it, and there are enough points to mean something.
#
# Set by eye over central Arnhem, and they are far apart by lens for a
# reason. A 360 photo sees the model all round, which pins a pose from every
# side, and a correct fit there can still score as low as 15%: a church is
# dozens of model edges behind trees, a modern front has little contrast,
# and half the scene can lie outside the area. Every false fit found was an
# ordinary photo seeing little of the model -- a plaque, a curtain wall --
# so those are held to more.
MIN_INLIERS = {"spherical": 0.12, "other": 0.25}
MIN_RATIO = {"spherical": 1.15, "other": 1.5}
# About 60 m of building edge. With less of the model in view, a glass
# curtain wall offers enough straight lines to fit almost anything: a photo
# of one with 86 model points in front of it verified at 60% on the mullions.
MIN_POINTS = 150
# The verdict is per building: a photo is trusted to paint a wall when the
# wall's own building lines up. Over central Arnhem, edges within 20 m of a
# camera landed at 26 to 33% on a good fit and edges further off at 3 to 8%,
# behind trees, market stalls and buildings outside the area, so one number
# for the whole photo rejected good fits for what stood far away.
MIN_BUILDING_POINTS = 40
# How much a move away from the starting pose costs, in units of the cost,
# per standard deviation squared of the pose source's own uncertainty. A
# correction bigger than the source's error has to be paid for in evidence.
PRIOR_WEIGHT = 0.02


# --------------------------------------------------------------------------
# Model edges
# --------------------------------------------------------------------------


def feature_edges(triangles: np.ndarray, angle_deg: float = 20.0) -> np.ndarray:
    """The edges of a triangle soup that show: boundaries and creases.

    Edges inside a flat surface are where earcut happened to cut it, and are
    invisible in a photo, so only edges whose two faces meet at more than
    `angle_deg`, or that have one face only, are kept. Returns (n, 2, 3).
    """
    if len(triangles) == 0:
        return np.zeros((0, 2, 3))
    corners = triangles.reshape(-1, 3)
    key = np.round(corners * 100.0).astype(np.int64)
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1, 3)
    vertices = unique / 100.0

    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)

    edges = np.concatenate([inverse[:, [0, 1]], inverse[:, [1, 2]], inverse[:, [2, 0]]])
    face = np.tile(np.arange(len(triangles)), 3)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, face = edges[order], face[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    starts = np.flatnonzero(np.r_[True, ~same])
    counts = np.diff(np.r_[starts, len(edges)])

    cos_limit = np.cos(np.radians(angle_deg))
    keep = counts != 2
    pairs = starts[counts == 2]
    crease = (normal[face[pairs]] * normal[face[pairs + 1]]).sum(axis=1) < cos_limit
    keep[counts == 2] = crease
    chosen = edges[starts[keep]]
    segments = vertices[chosen]
    length = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    return segments[length >= MIN_EDGE_M]


def building_edges(buildings) -> tuple[np.ndarray, np.ndarray]:
    """The edges of every building a pedestrian sees, and whose each is.

    Those are the edges of the walls: their corners, and their tops, which
    are the eaves, the parapets and the gable rakes. Edges that belong to
    the roofs alone -- ridges, the plant on a flat roof, a church's inner
    roofs -- are left out. From the street most of them sit behind a
    parapet or a gutter, and the occlusion test, which only knows walls,
    cannot tell; counted, they made correct fits over central Arnhem score
    as low as 5%.

    The line where a wall meets the street is left out too: parked cars,
    bins and kerbs hide it in nearly every street photo, and matching it
    would pull cameras towards whatever stands in front.
    """
    segments, owner = [], []
    for index, building in enumerate(buildings.buildings):
        parts = [t for t in (building.wall_tris, building.ground_wall_tris) if len(t)]
        if not parts:
            continue
        edges = feature_edges(np.concatenate(parts))
        low = edges[:, :, 2].max(axis=1) < building.ground_z_nap + 1.0
        edges = edges[~low]
        segments.append(edges)
        owner.append(np.full(len(edges), index, dtype=np.int64))
    if not segments:
        return np.zeros((0, 2, 3)), np.zeros(0, dtype=np.int64)
    return np.concatenate(segments), np.concatenate(owner)


@dataclass
class EdgeSamples:
    """Points along the model's visible edges, which way each edge runs, and
    which building each belongs to (an index into the building list)."""

    points: np.ndarray  # (n, 3)
    directions: np.ndarray  # (n, 3) unit vectors
    owner: np.ndarray  # (n,) int

    def __len__(self) -> int:
        return int(len(self.points))

    def subset(self, mask) -> "EdgeSamples":
        return EdgeSamples(self.points[mask], self.directions[mask], self.owner[mask])


def sample_edges(
    segments: np.ndarray, spacing_m: float = EDGE_SPACING_M, owner: np.ndarray | None = None
) -> EdgeSamples:
    """Points every `spacing_m` along each segment, with its direction."""
    if owner is None:
        owner = np.zeros(len(segments), dtype=np.int64)
    if not len(segments):
        return EdgeSamples(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int64))
    a, b = segments[:, 0], segments[:, 1]
    length = np.linalg.norm(b - a, axis=1)
    count = np.maximum(1, np.floor(length / spacing_m)).astype(np.int64)
    which = np.repeat(np.arange(len(segments)), count)
    # Centred in each step, so no point sits exactly on a corner, where two
    # edges running different ways would both claim it.
    offset = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    t = (offset + 0.5) / count[which]
    points = a[which] + t[:, None] * (b - a)[which]
    direction = (b - a)[which] / length[which][:, None]
    return EdgeSamples(points, direction, np.asarray(owner)[which])


# --------------------------------------------------------------------------
# Image edges
# --------------------------------------------------------------------------


def _shift_max(mask: np.ndarray) -> np.ndarray:
    """3x3 dilation."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    tmp = out.copy()
    out[:, 1:] |= tmp[:, :-1]
    out[:, :-1] |= tmp[:, 1:]
    return out


def truncated_distance(mask: np.ndarray, limit: int = TRUNCATE_PX) -> np.ndarray:
    """Chessboard distance to the nearest True pixel, capped at `limit`."""
    distance = np.full(mask.shape, limit, dtype=np.float32)
    current = mask.copy()
    distance[current] = 0
    for step in range(1, limit):
        grown = _shift_max(current)
        distance[grown & ~current] = step
        current = grown
    return distance


def _blur(gray: np.ndarray) -> np.ndarray:
    out = gray.copy()
    for axis in (0, 1):
        out = (np.roll(out, 1, axis) + out + np.roll(out, -1, axis)) / 3.0
    return out


@dataclass
class EdgeMaps:
    """Distance to the nearest edge, per edge orientation, for one photo."""

    distance: np.ndarray  # (BINS, h, w) float32
    scale: float  # work pixels per original pixel

    @property
    def height(self) -> int:
        return int(self.distance.shape[1])

    @property
    def width(self) -> int:
        return int(self.distance.shape[2])


def edge_maps(image: np.ndarray, camera: Camera, *, percentile: float = 88.0) -> EdgeMaps:
    """Find the edges in a photo, sorted by which way they run.

    `image` is an (h, w) or (h, w, 3) array at the camera's resolution.
    Edges are the strongest gradients in the picture; a fixed share of
    pixels rather than a fixed threshold, so a dull winter photo and a
    sunlit one both yield about the same density of edges.
    """
    from PIL import Image

    array = np.asarray(image)
    pil = Image.fromarray(array.astype(np.uint8))
    scale = min(1.0, WORK_WIDTH / pil.width)
    work = pil.convert("L").resize(
        (max(1, round(pil.width * scale)), max(1, round(pil.height * scale))),
        Image.BILINEAR,
    )
    gray = _blur(np.asarray(work, dtype=np.float32))
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[1:-1, 1:-1] = (
        gray[:-2, 2:] + 2 * gray[1:-1, 2:] + gray[2:, 2:]
        - gray[:-2, :-2] - 2 * gray[1:-1, :-2] - gray[2:, :-2]
    )
    gy[1:-1, 1:-1] = (
        gray[2:, :-2] + 2 * gray[2:, 1:-1] + gray[2:, 2:]
        - gray[:-2, :-2] - 2 * gray[:-2, 1:-1] - gray[:-2, 2:]
    )
    magnitude = np.hypot(gx, gy)

    valid = np.ones(gray.shape, dtype=bool)
    if camera.kind == SPHERICAL:
        rows = np.arange(gray.shape[0])
        elevation = (0.5 - (rows + 0.5) / gray.shape[0]) * 180.0
        valid[elevation < PANO_FLOOR_DEG] = False
    threshold = max(np.percentile(magnitude[valid], percentile), 20.0)
    strong = (magnitude >= threshold) & valid

    # Thin the edges a little: keep a pixel only where it is at least as
    # strong as its neighbours across the gradient. Without this a soft
    # edge is five pixels wide and matches anything near it.
    angle = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    # One pixel along the gradient, which with the angle folded into
    # [0, 180) always points down or sideways.
    across_x = np.rint(np.cos(np.radians(angle))).astype(np.int64)
    across_y = np.rint(np.sin(np.radians(angle))).astype(np.int64)
    h, w = gray.shape
    rr, cc = np.mgrid[0:h, 0:w]
    r1 = np.clip(rr + across_y, 0, h - 1)
    c1 = np.clip(cc + across_x, 0, w - 1)
    r2 = np.clip(rr - across_y, 0, h - 1)
    c2 = np.clip(cc - across_x, 0, w - 1)
    peak = (magnitude >= magnitude[r1, c1]) & (magnitude >= magnitude[r2, c2])
    strong &= peak

    maps = np.empty((BINS, h, w), dtype=np.float32)
    for k in range(BINS):
        centre = k * 180.0 / BINS
        diff = np.abs((angle - centre + 90.0) % 180.0 - 90.0)
        maps[k] = truncated_distance(strong & (diff <= BIN_HALF_WIDTH_DEG))
        maps[k][~valid] = TRUNCATE_PX
    return EdgeMaps(distance=maps, scale=scale)


# --------------------------------------------------------------------------
# Pose
# --------------------------------------------------------------------------


def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_cam(pitch_deg: float, roll_deg: float) -> np.ndarray:
    p, r = np.radians([pitch_deg, roll_deg])
    cp, sp, cr, sr = np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    roll = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])
    return roll @ pitch


def perturbed(rotation: np.ndarray, centre: np.ndarray, params: np.ndarray):
    """Pose after turning by yaw/pitch/roll degrees and moving dx/dy/dz metres.

    Yaw turns the camera about the world's vertical, as a wrong compass
    would; pitch and roll tilt it about its own axes, as an unlevel mount
    would. Positive yaw turns clockwise seen from above.
    """
    yaw, pitch, roll, dx, dy, dz = params
    turned = _rot_cam(pitch, roll) @ rotation @ _rot_z(yaw)
    return turned, centre + np.array([dx, dy, dz])


@dataclass
class Alignment:
    photo: str
    before: float  # inlier share at the starting pose
    after: float
    points: int  # model edge points in view
    params: tuple  # yaw, pitch, roll (deg), dx, dy, dz (m)
    verified: bool
    reason: str = ""
    baseline: float = 0.0  # inlier share with the model moved off the fit
    # Per building in view: points, inlier share after, with the model moved
    # off, and whether it lines up. Keyed by building index.
    buildings: dict = field(default_factory=dict)
    building_ids: dict = field(default_factory=dict)

    @property
    def verified_buildings(self) -> list:
        return [self.building_ids.get(b, b) for b, v in self.buildings.items() if v["verified"]]

    def to_dict(self) -> dict:
        return {
            "photo": self.photo,
            "inliers_before": round(self.before, 3),
            "inliers_after": round(self.after, 3),
            "edge_points": self.points,
            "correction": {
                k: round(float(v), 3)
                for k, v in zip(("yaw_deg", "pitch_deg", "roll_deg", "dx_m", "dy_m", "dz_m"), self.params)
            },
            "inliers_moved_off": round(self.baseline, 3),
            "verified": self.verified,
            **({"reason": self.reason} if self.reason else {}),
            "buildings_in_view": len(self.buildings),
            "buildings_verified": [str(b) for b in self.verified_buildings],
        }


def _bins_for(camera: Camera, rotation, centre, points, directions):
    """Which orientation bin each point's edge falls in, in the picture.

    An edge's gradient runs across it, so a line drawn at angle a in the
    image lights the bin at a + 90.
    """
    p0, ok0 = project_camera(camera, (points - centre) @ rotation.T)
    p1, ok1 = project_camera(camera, (points + 0.15 * directions - centre) @ rotation.T)
    d = p1 - p0
    angle = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 90.0) % 180.0
    k = np.round(angle / (180.0 / BINS)).astype(np.int64) % BINS
    return k, ok0 & ok1


class _Objective:
    def __init__(
        self, photo: Photo, maps: EdgeMaps, samples: EdgeSamples,
        sigma: np.ndarray | None = None,
    ) -> None:
        points, directions = samples.points, samples.directions
        self.camera = photo.camera
        self.rotation0 = photo.rotation
        self.centre0 = photo.centre
        self.maps = maps
        self.points = points
        self.directions = directions
        # Standard deviations of yaw, pitch, roll (deg), dx, dy, dz (m).
        self.sigma = sigma
        # Orientation is decided once, at the start: the corrections searched
        # are too small to turn an edge into a different bin.
        self.bins, _ = _bins_for(self.camera, self.rotation0, self.centre0, points, directions)

    def distances(self, params) -> tuple[np.ndarray, np.ndarray]:
        rotation, centre = perturbed(self.rotation0, self.centre0, np.asarray(params))
        pixels, ok = project_camera(self.camera, (self.points - centre) @ rotation.T)
        u = np.floor(pixels[:, 0] * self.maps.scale).astype(np.int64)
        v = np.floor(pixels[:, 1] * self.maps.scale).astype(np.int64)
        if self.camera.kind == SPHERICAL:
            u %= self.maps.width
        inside = ok & (u >= 0) & (u < self.maps.width) & (v >= 0) & (v < self.maps.height)
        d = np.full(len(self.points), float(TRUNCATE_PX))
        d[inside] = self.maps.distance[self.bins[inside], v[inside], u[inside]]
        return d, inside

    def cost(self, params) -> float:
        d, _ = self.distances(params)
        value = float(d.mean()) / TRUNCATE_PX
        if self.sigma is not None:
            z = np.asarray(params) / self.sigma
            value += PRIOR_WEIGHT * float(z @ z)
        return value

    def inlier_mask(self, params) -> np.ndarray:
        d, _ = self.distances(params)
        return d <= INLIER_PX

    def inliers(self, params) -> float:
        return float(self.inlier_mask(params).mean())


def drop_hidden_seams(samples: EdgeSamples, faces, tolerance_m: float = 0.3) -> EdgeSamples:
    """Leave out edge points that lie on a party wall.

    Where two terraced houses meet, the model has a corner and a roof edge
    on the wall they share, and the street sees one front running straight
    on. Those points can never land on an edge in a photo, however good the
    pose, so they only dilute the evidence.
    """
    points = samples.points
    # Party walls only. A wall facing a neighbour across a narrow alley is
    # hard to photograph, but its corner on the street is plain to see.
    hidden = [f for f in faces if f.party]
    if not hidden or not len(points):
        return samples
    keep = np.ones(len(points), dtype=bool)
    for face in hidden:
        a, b = face.start, face.end
        span = b - a
        length2 = float(span @ span)
        if length2 < 1e-9:
            continue
        rel = points[:, :2] - a
        t = np.clip(rel @ span / length2, 0.0, 1.0)
        off = np.linalg.norm(rel - t[:, None] * span, axis=1)
        on = (off <= tolerance_m) & (points[:, 2] >= face.z_min - 0.2) & (points[:, 2] <= face.z_max + 0.5)
        keep &= ~on
    return samples.subset(keep)


def visible_edge_points(
    photo: Photo, samples: EdgeSamples, faces_index, max_distance_m: float
) -> EdgeSamples:
    """The model edge points this camera could see, in front of anything else."""
    c = photo.centre
    samples = samples.subset(
        np.linalg.norm(samples.points[:, :2] - c[:2], axis=1) <= max_distance_m
    )
    if not len(samples):
        return samples
    points = samples.points
    _, in_view = project_camera(photo.camera, (points - c) @ photo.rotation.T)
    if photo.camera.kind == SPHERICAL:
        elevation = np.degrees(np.arctan2(points[:, 2] - c[2], np.linalg.norm(points[:, :2] - c[:2], axis=1)))
        in_view &= elevation > PANO_FLOOR_DEG + 3.0
        if not photo.heading_known:
            in_view[:] = elevation > PANO_FLOOR_DEG + 3.0
    samples = samples.subset(in_view)
    if not len(samples):
        return samples
    points = samples.points
    # Nudged towards the camera, so the faces an edge belongs to do not
    # count as standing in front of it.
    towards = c[None] - points
    towards /= np.maximum(np.linalg.norm(towards, axis=1, keepdims=True), 1e-9)
    probes = points + 0.25 * towards
    candidates = faces_index.near(c[0], c[1], max_distance_m)
    hidden = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), 2000):
        chunk = slice(start, start + 2000)
        hidden[chunk] = faces_index.blocked(
            c, probes[chunk], candidates, np.full(len(probes[chunk]), -1)
        )
    return samples.subset(~hidden)


def _pattern_search(objective: _Objective, start, steps, bounds, *, min_step=0.03, max_rounds=60):
    params = np.array(start, dtype=np.float64)
    steps = np.array(steps, dtype=np.float64)
    best = objective.cost(params)
    for _ in range(max_rounds):
        improved = False
        for i in range(len(params)):
            for sign in (1.0, -1.0):
                trial = params.copy()
                trial[i] += sign * steps[i]
                if abs(trial[i]) > bounds[i]:
                    continue
                value = objective.cost(trial)
                if value < best - 1e-6:
                    params, best, improved = trial, value, True
                    break
        if not improved:
            steps *= 0.5
            if np.all(steps < min_step):
                break
    return params, best


def align_photo(
    photo: Photo,
    image: np.ndarray,
    samples: EdgeSamples,
    faces_index,
    *,
    building_ids=None,
    max_distance_m: float = 40.0,
    min_inliers: float | None = None,
    min_ratio: float | None = None,
    min_points: int = MIN_POINTS,
) -> tuple[Photo, Alignment]:
    """Refine one photo's pose against the model, and say what to trust it for.

    The pose is fitted to every model edge in view. The verdict is per
    building: the photo's `verified_buildings` lists the buildings whose own
    edges line up, and only those are walls it may paint. `building_ids`
    turns building indices into the identifiers listed; without it the
    indices are listed.
    """
    seen = visible_edge_points(photo, samples, faces_index, max_distance_m)
    zero = (0.0,) * 6
    if len(seen) < min_points:
        return replace(photo, verified=False, verified_buildings=[]), Alignment(
            photo.id, 0.0, 0.0, len(seen), zero, False,
            f"only {len(seen)} model edge points in view",
        )

    maps = edge_maps(image, photo.camera)
    objective = _Objective(photo, maps, seen)
    before = objective.inliers(np.zeros(6))

    sigma = max(1.0, float(photo.position_sigma_m))
    if not photo.heading_known:
        yaw_sigma, yaw_bound = 180.0, 360.0
    elif photo.pose == "sfm":
        yaw_sigma, yaw_bound = 2.0, 6.0
    else:
        yaw_sigma, yaw_bound = 10.0, 30.0
    reach = min(2.5, 2.0 * sigma)
    objective.sigma = np.array([yaw_sigma, 2.0, 2.0, sigma, sigma, 0.5])

    start_yaw = 0.0
    if not photo.heading_known:
        # Nothing said which way the camera faced: turn it all the way round
        # first, then search position around the best heading.
        sweep = np.arange(-180.0, 180.0, 2.0)
        costs = [objective.cost(np.array([y, 0, 0, 0, 0, 0.0])) for y in sweep]
        start_yaw = float(sweep[int(np.argmin(costs))])
        yaws = start_yaw + np.arange(-4.0, 4.01, 1.0)
    else:
        step = 1.0 if photo.pose == "sfm" else 2.0
        yaws = np.arange(-yaw_bound + step, yaw_bound - step + 1e-9, step)
    shifts = np.arange(-reach, reach + 1e-9, 0.5)

    best = np.array([start_yaw, 0, 0, 0, 0, 0.0])
    best_cost = objective.cost(best)
    for yaw in yaws:
        for dx in shifts:
            for dy in shifts:
                trial = np.array([yaw, 0.0, 0.0, dx, dy, 0.0])
                value = objective.cost(trial)
                if value < best_cost:
                    best, best_cost = trial, value

    bounds = np.array([yaw_bound, 6.0, 6.0, reach, reach, 1.5])
    params, _ = _pattern_search(objective, best, [1.0, 0.5, 0.5, 0.5, 0.5, 0.3], bounds)
    after = objective.inliers(params)

    # The same model, moved a little off the fit in every direction it can
    # be wrong. Averaged, since a facade of identical bays matches one bay
    # along as well as it matches the right place.
    offsets = [
        (d, 0, 0, 0, 0, 0) for d in (-8.0, -5.0, 5.0, 8.0)
    ] + [
        (0, 0, 0, dx, dy, 0) for dx, dy in ((2.5, 0), (-2.5, 0), (0, 2.5), (0, -2.5))
    ] + [(0, d, 0, 0, 0, 0) for d in (-3.0, 3.0)]
    moved = np.mean(
        [objective.inlier_mask(params + np.array(o)) for o in offsets], axis=0
    )
    baseline = float(moved.mean())
    hits = objective.inlier_mask(params)

    pinned = [
        name
        for name, value, bound in zip(("heading", "pitch", "roll", "x", "y", "height"), params, bounds)
        if bound < 180 and abs(value) >= 0.9 * bound
    ]
    lens = "spherical" if photo.camera.kind == SPHERICAL else "other"
    if min_inliers is None:
        min_inliers = MIN_INLIERS[lens]
    if min_ratio is None:
        min_ratio = MIN_RATIO[lens]

    per_building: dict = {}
    for b in np.unique(seen.owner):
        mine = seen.owner == b
        count = int(mine.sum())
        if count < MIN_BUILDING_POINTS:
            continue
        share = float(hits[mine].mean())
        off = float(moved[mine].mean())
        per_building[int(b)] = {
            "points": count,
            "after": share,
            "moved_off": off,
            "verified": bool(share >= min_inliers and share >= min_ratio * off),
        }
    good = [b for b, v in per_building.items() if v["verified"]]

    if pinned:
        verified, reason = False, (
            f"the {', '.join(pinned)} had to move to the edge of what the pose "
            f"could be wrong by"
        )
    elif after < min_ratio * baseline:
        # A few dozen points on one building can be fitted to nearly any
        # busy picture. The photo as a whole has to agree too.
        verified, reason = False, (
            f"the fit ({100 * after:.0f}%) hardly beats the model moved off it "
            f"({100 * baseline:.0f}%) over the whole photo"
        )
    elif not per_building:
        verified, reason = False, "no building shows enough of its edges"
    elif not good:
        best = max(per_building.values(), key=lambda v: v["after"])
        if best["after"] < min_inliers:
            reason = (
                f"no building lines up: at best {100 * best['after']:.0f}% of a "
                f"building's edges land on edges in the photo"
            )
        else:
            reason = (
                f"no building lines up: the best fit ({100 * best['after']:.0f}%) "
                f"hardly beats the model moved off it ({100 * best['moved_off']:.0f}%), "
                f"a repeating facade or not this building"
            )
        verified = False
    else:
        verified, reason = True, ""
    for b in per_building:
        if not verified:
            per_building[b]["verified"] = False
    ids = {
        b: (building_ids[b] if building_ids is not None else b) for b in per_building
    }
    objective.sigma = None
    rotation, centre = perturbed(photo.rotation, photo.centre, params)
    refined = replace(
        photo,
        rotation=rotation,
        x=float(centre[0]),
        y=float(centre[1]),
        z=float(centre[2]),
        verified=verified,
        verified_buildings=[ids[b] for b in per_building if per_building[b]["verified"]],
        heading_known=photo.heading_known or verified,
    )
    return refined, Alignment(
        photo.id, before, after, len(seen), tuple(float(v) for v in params),
        verified, reason, baseline, per_building, ids,
    )


def check_sequences(
    photos: dict[str, Photo],
    alignments: dict[str, Alignment],
    *,
    radius_m: float = 60.0,
    max_shift_m: float = 1.5,
    max_turn_deg: float = 2.5,
) -> list[str]:
    """Withdraw verification from corrections their own sequence disagrees with.

    A Mapillary sequence is one reconstruction, so its errors are smooth: a
    GPS bias of a metre north holds for every photo near each other in it.
    When one photo's correction differs from what its verified neighbours
    in the same sequence needed, it fitted something it should not have.
    Needs two neighbours to judge; with fewer, the photo keeps its verdict.
    Returns the ids withdrawn.
    """
    withdrawn = []
    verified = [pid for pid, a in alignments.items() if a.verified and photos[pid].sequence]
    for pid in verified:
        photo = photos[pid]
        others = [
            q for q in verified
            if q != pid
            and photos[q].sequence == photo.sequence
            and np.hypot(photos[q].x - photo.x, photos[q].y - photo.y) <= radius_m
        ]
        if len(others) < 2:
            continue
        mine = np.array(alignments[pid].params)
        theirs = np.median(np.array([alignments[q].params for q in others]), axis=0)
        shift = float(np.hypot(*(mine[3:5] - theirs[3:5])))
        turn = float(abs(mine[0] - theirs[0]))
        if shift > max_shift_m or turn > max_turn_deg:
            a = alignments[pid]
            a.verified = False
            for stats in a.buildings.values():
                stats["verified"] = False
            a.reason = (
                f"its correction differs from its sequence's by {shift:.1f} m "
                f"and {turn:.1f} degrees"
            )
            withdrawn.append(pid)
    return withdrawn


def load_image(photo: Photo) -> np.ndarray:
    """A photo's pixels, upright, at the size its camera describes."""
    from PIL import Image, ImageOps

    with Image.open(photo.path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.size != (photo.camera.width, photo.camera.height):
            image = image.resize((photo.camera.width, photo.camera.height), Image.BILINEAR)
        return np.asarray(image)


def draw_overlay(photo: Photo, image: np.ndarray, segments: np.ndarray, path, *, before: Photo | None = None, max_distance_m: float = 45.0):
    """The photo with the model's edges drawn on: magenta before, green after."""
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    near = np.linalg.norm(segments.mean(axis=1)[:, :2] - photo.centre[:2], axis=1) < max_distance_m
    for pose, colour in ((before, (255, 40, 200)), (photo, (60, 255, 90))):
        if pose is None:
            continue
        for a, b in segments[near]:
            pts = a[None] + np.linspace(0.0, 1.0, 16)[:, None] * (b - a)[None]
            px, ok = project_camera(pose.camera, (pts - pose.centre) @ pose.rotation.T)
            for i in range(len(pts) - 1):
                if ok[i] and ok[i + 1] and abs(px[i, 0] - px[i + 1, 0]) < pose.camera.width / 2:
                    draw.line([tuple(px[i]), tuple(px[i + 1])], fill=colour, width=2)
    canvas.save(path, quality=85)
    return path


__all__ = [
    "Alignment",
    "EdgeSamples",
    "drop_hidden_seams",
    "check_sequences",
    "EdgeMaps",
    "align_photo",
    "building_edges",
    "draw_overlay",
    "edge_maps",
    "feature_edges",
    "load_image",
    "perturbed",
    "sample_edges",
    "truncated_distance",
    "visible_edge_points",
]
