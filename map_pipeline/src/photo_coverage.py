"""Which walls the street photos see, and how sharply.

The question every later step starts from. Measuring a facade (route 1)
needs a wall seen square-on enough to count its windows; wearing one as a
texture (route 2) needs it seen sharply, whole and unobstructed. Both are
decided per *wall face* -- one flat side of one building, the same grouping
the facade UVs are fitted over -- and per photo.

A view of a face is scored by three numbers:

``visible``
    The share of points spread over the face that land inside the picture
    and are not hidden behind another building.
``px_per_m``
    How many pixels of the photo one metre of wall covers, the worse of the
    horizontal and vertical directions. Horizontal includes foreshortening:
    a wall seen at 60 degrees off square gets half the pixels across it.
``incidence_deg``
    The angle between the wall's outward normal and the line to the camera.

Occlusion is tested in plan against every other face, using each face's
height range, so a ray to an upper storey passes over a low shed in front
while a ray to the ground floor does not. It is not an exact depth test --
gables and roofs are ignored, and so are trees and parked cars -- which is
right for deciding coverage and wrong for projecting pixels. Projection does
its own depth test.

A wall nobody could photograph anyway, the party wall between two terraced
houses, is not counted against coverage: a face whose outward normal runs
into another face within `EXPOSED_CLEARANCE_M` is marked unexposed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import SPHERICAL, project_world
from .facade_uv import wall_face_ids
from .photos import Photo

LOG = logging.getLogger(__name__)

EXPOSED_CLEARANCE_M = 2.5
MIN_FACE_AREA_M2 = 4.0
MIN_FACE_WIDTH_M = 1.0
# Sample points across and up each face, inset from its edges.
SAMPLES_ACROSS = 4
SAMPLES_UP = 3
# Views kept per face, best first.
VIEWS_PER_FACE = 8

UNSEEN, SEEN, MEASURE, TEXTURE = "unseen", "seen", "measure", "texture"


@dataclass
class WallFace:
    """One flat side of one building."""

    building: int
    building_id: str
    # Ends of the face along the ground, RD metres, left to right seen from
    # outside.
    start: np.ndarray
    end: np.ndarray
    z_min: float
    z_max: float
    # Horizontal unit vector pointing out of the building.
    normal: np.ndarray
    area_m2: float
    samples: np.ndarray  # (k, 3) points on the face
    triangles: np.ndarray  # (t, 3, 3) the wall triangles it was made from
    exposed: bool = True

    @property
    def width_m(self) -> float:
        return float(np.hypot(*(self.end - self.start)))

    @property
    def height_m(self) -> float:
        return self.z_max - self.z_min

    @property
    def centre(self) -> np.ndarray:
        mid = 0.5 * (self.start + self.end)
        return np.array([mid[0], mid[1], 0.5 * (self.z_min + self.z_max)])


@dataclass
class View:
    photo: str
    visible: float
    px_per_m: float
    incidence_deg: float
    distance_m: float

    def to_dict(self) -> dict:
        return {
            "photo": self.photo,
            "visible": round(self.visible, 3),
            "px_per_m": round(self.px_per_m, 1),
            "incidence_deg": round(self.incidence_deg, 1),
            "distance_m": round(self.distance_m, 1),
        }


@dataclass
class Coverage:
    faces: list[WallFace] = field(default_factory=list)
    views: list[list[View]] = field(default_factory=list)
    photos_used: int = 0
    photos_total: int = 0
    thresholds: dict = field(default_factory=dict)

    def best(self, index: int, min_visible: float) -> View | None:
        good = [v for v in self.views[index] if v.visible >= min_visible]
        return max(good, key=lambda v: v.px_per_m) if good else None

    def face_class(self, index: int) -> str:
        t = self.thresholds
        view = self.best(index, float(t.get("min_visible", 0.6)))
        if view is None:
            return SEEN if self.views[index] else UNSEEN
        if view.px_per_m >= float(t.get("texture_px_per_m", 40.0)) and view.incidence_deg <= float(
            t.get("texture_max_incidence_deg", 55.0)
        ):
            return TEXTURE
        if view.px_per_m >= float(t.get("measure_px_per_m", 12.0)):
            return MEASURE
        return SEEN

    def stats(self) -> dict:
        classes = [self.face_class(i) for i in range(len(self.faces))]
        exposed = [i for i, f in enumerate(self.faces) if f.exposed]
        area = np.array([self.faces[i].area_m2 for i in exposed])
        total_area = float(area.sum()) or 1.0

        def share(*wanted: str) -> float:
            hit = np.array([classes[i] in wanted for i in exposed], dtype=bool)
            return round(float(area[hit].sum()) / total_area, 3) if len(exposed) else 0.0

        buildings = {f.building for f in self.faces}

        def buildings_with(*wanted: str) -> int:
            return len(
                {self.faces[i].building for i in exposed if classes[i] in wanted}
            )

        return {
            "photos": self.photos_total,
            "photos_seeing_a_wall": self.photos_used,
            "wall_faces": len(self.faces),
            "exposed_faces": len(exposed),
            "exposed_wall_area_m2": round(float(area.sum()), 1),
            # Shares of exposed wall area. Texture-ready walls are also
            # measurable, so the second includes the first.
            "share_texture_ready": share(TEXTURE),
            "share_measurable": share(TEXTURE, MEASURE),
            "share_seen": share(TEXTURE, MEASURE, SEEN),
            "buildings": len(buildings),
            "buildings_with_texture_ready_wall": buildings_with(TEXTURE),
            "buildings_with_measurable_wall": buildings_with(TEXTURE, MEASURE),
            "thresholds": dict(self.thresholds),
        }


def _inside_triangles_uz(points: np.ndarray, tris_uz: np.ndarray) -> np.ndarray:
    """Which (u, z) points fall inside any of the (u, z) triangles."""
    if not len(points) or not len(tris_uz):
        return np.zeros(len(points), dtype=bool)
    a, b, c = tris_uz[:, 0], tris_uz[:, 1], tris_uz[:, 2]
    p = points[:, None, :]
    v0, v1, v2 = (c - a)[None], (b - a)[None], p - a[None]
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    ok = np.abs(denom) > 1e-12
    denom = np.where(ok, denom, 1.0)
    s = (d11 * d20 - d01 * d21) / denom
    t = (d00 * d21 - d01 * d20) / denom
    inside = ok & (s >= -1e-6) & (t >= -1e-6) & (s + t <= 1 + 1e-6)
    return inside.any(axis=1)


def wall_faces(buildings) -> list[WallFace]:
    """Every wall face of every building worth photographing."""
    chunks, owners = [], []
    for index, building in enumerate(buildings.buildings):
        tris = building.all_wall_tris
        if len(tris):
            chunks.append(tris)
            owners.append(np.full(len(tris), index, dtype=np.int64))
    if not chunks:
        return []
    tris = np.concatenate(chunks)
    owner = np.concatenate(owners)
    face_of, n_faces = wall_face_ids(tris, owner)

    edge1 = tris[:, 1] - tris[:, 0]
    edge2 = tris[:, 2] - tris[:, 0]
    cross = np.cross(edge1, edge2)
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)

    order = np.argsort(face_of, kind="stable")
    bounds = np.searchsorted(face_of[order], np.arange(n_faces + 1))
    faces: list[WallFace] = []
    for face in range(n_faces):
        members = order[bounds[face] : bounds[face + 1]]
        area = float(tri_area[members].sum())
        if area < MIN_FACE_AREA_M2:
            continue
        # CityJSON orients surfaces outward, so the area-weighted normal of
        # the face's triangles points out of the building.
        n = cross[members].sum(axis=0)[:2]
        length = float(np.hypot(*n))
        if length < 1e-9:
            continue
        normal = n / length
        # Left to right for someone standing outside looking at the wall. The
        # facade U in facade_uv.py runs the other way.
        tangent = np.array([-normal[1], normal[0]])
        corners = tris[members].reshape(-1, 3)
        along = corners[:, :2] @ tangent
        across = corners[:, :2] @ normal
        u0, u1 = float(along.min()), float(along.max())
        if u1 - u0 < MIN_FACE_WIDTH_M:
            continue
        offset = float(np.median(across))
        start = tangent * u0 + normal * offset
        end = tangent * u1 + normal * offset
        z_min, z_max = float(corners[:, 2].min()), float(corners[:, 2].max())

        # Samples on a grid over the face's bounding rectangle, kept where the
        # face actually is: a gable end is a pentagon, not a rectangle.
        fu = (np.arange(SAMPLES_ACROSS) + 0.5) / SAMPLES_ACROSS
        fz = (np.arange(SAMPLES_UP) + 0.5) / SAMPLES_UP
        gu, gz = np.meshgrid(u0 + fu * (u1 - u0), z_min + fz * (z_max - z_min))
        grid = np.column_stack([gu.ravel(), gz.ravel()])
        tris_uz = np.stack(
            [
                tris[members][:, :, :2] @ tangent,
                tris[members][:, :, 2],
            ],
            axis=-1,
        )
        keep = _inside_triangles_uz(grid, tris_uz)
        if not keep.any():
            keep[:] = True
        grid = grid[keep]
        # Lifted a few centimetres off the wall so the face does not hide its
        # own samples.
        xy = tangent[None] * grid[:, :1] + normal[None] * (offset + 0.05)
        samples = np.column_stack([xy, grid[:, 1]])

        building_index = int(owner[members[0]])
        faces.append(
            WallFace(
                building=building_index,
                building_id=buildings.buildings[building_index].identifier,
                start=start,
                end=end,
                z_min=z_min,
                z_max=z_max,
                normal=normal,
                area_m2=area,
                samples=samples,
                triangles=tris[members],
            )
        )
    return faces


class _SegmentIndex:
    """Faces as plan segments, bucketed on a grid for quick lookup."""

    def __init__(self, faces: list[WallFace], cell_m: float = 20.0) -> None:
        self.a = np.array([f.start for f in faces]).reshape(-1, 2)
        self.b = np.array([f.end for f in faces]).reshape(-1, 2)
        self.z0 = np.array([f.z_min for f in faces])
        self.z1 = np.array([f.z_max for f in faces])
        self.cell = cell_m
        self.buckets: dict[tuple[int, int], list[int]] = {}
        for i in range(len(faces)):
            lo = np.floor(np.minimum(self.a[i], self.b[i]) / cell_m).astype(int)
            hi = np.floor(np.maximum(self.a[i], self.b[i]) / cell_m).astype(int)
            for cx in range(lo[0], hi[0] + 1):
                for cy in range(lo[1], hi[1] + 1):
                    self.buckets.setdefault((cx, cy), []).append(i)

    def near(self, x: float, y: float, radius: float) -> np.ndarray:
        lo = np.floor((np.array([x, y]) - radius) / self.cell).astype(int)
        hi = np.floor((np.array([x, y]) + radius) / self.cell).astype(int)
        found: set[int] = set()
        for cx in range(lo[0], hi[0] + 1):
            for cy in range(lo[1], hi[1] + 1):
                found.update(self.buckets.get((cx, cy), ()))
        return np.array(sorted(found), dtype=np.int64)

    def blocked(
        self,
        origin: np.ndarray,
        targets: np.ndarray,
        candidates: np.ndarray,
        exclude: np.ndarray,
    ) -> np.ndarray:
        """Is the ray from `origin` to each target crossed by a candidate face?

        `exclude[k]` is the face target k lies on, which must not hide it.
        """
        if not len(targets) or not len(candidates):
            return np.zeros(len(targets), dtype=bool)
        o = origin[:2]
        d = targets[:, :2] - o  # (n, 2)
        a = self.a[candidates]  # (m, 2)
        e = self.b[candidates] - a
        # Solve o + t d = a + s e for every ray/segment pair.
        denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        ok = np.abs(denom) > 1e-12
        denom = np.where(ok, denom, 1.0)
        w = a[None] - o
        t = (w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]) / denom
        s = (w[..., 0] * d[:, None, 1] - w[..., 1] * d[:, None, 0]) / denom
        length = np.linalg.norm(d, axis=1)[:, None]
        # Stop short of the target, so a face sharing a corner with the
        # target's face does not catch rays to the target's edge.
        margin = 0.3 / np.maximum(length, 1e-6)
        z = origin[2] + t * (targets[:, None, 2] - origin[2])
        hit = (
            ok
            & (t > 1e-3)
            & (t < 1.0 - margin)
            & (s >= 0.0)
            & (s <= 1.0)
            & (z >= self.z0[candidates][None])
            & (z <= self.z1[candidates][None])
            & (candidates[None, :] != exclude[:, None])
        )
        return hit.any(axis=1)


def mark_exposed(faces: list[WallFace], index: _SegmentIndex) -> None:
    """Flag faces that open onto something a camera could stand in."""
    for i, face in enumerate(faces):
        centre = face.centre
        mid_z = face.z_min + min(2.0, 0.5 * face.height_m)
        # From just inside the wall, so a neighbour's wall on exactly the same
        # line -- the other side of a party wall -- is crossed, not missed.
        origin = np.array([centre[0], centre[1], mid_z]) - np.r_[face.normal * 0.2, 0.0]
        probe = origin + np.r_[face.normal * (EXPOSED_CLEARANCE_M + 0.2), 0.0]
        candidates = index.near(centre[0], centre[1], EXPOSED_CLEARANCE_M + 1.0)
        face.exposed = not bool(
            index.blocked(origin, probe[None], candidates, np.array([i]))[0]
        )


def score_views(
    faces: list[WallFace],
    photos: list[Photo],
    *,
    max_distance_m: float = 40.0,
    max_incidence_deg: float = 75.0,
    min_visible: float = 0.25,
    resolution: dict[str, int] | None = None,
) -> Coverage:
    """Every useful (photo, face) pair, with how well the photo sees the face.

    `resolution` gives the image width each camera kind will be downloaded
    at, keyed by kind, so the sharpness is that of the pixels that will
    actually be used rather than of an original nobody fetches. 0 or missing
    means the original.
    """
    resolution = resolution or {}
    index = _SegmentIndex(faces)
    mark_exposed(faces, index)

    centres = np.array([f.centre for f in faces]).reshape(-1, 3)
    normals = np.array([f.normal for f in faces]).reshape(-1, 2)
    cos_max = np.cos(np.radians(max_incidence_deg))
    views: list[list[View]] = [[] for _ in faces]
    used = 0

    for photo in photos:
        camera = photo.camera
        width = int(resolution.get(camera.kind, 0) or 0)
        if width and width < camera.width:
            camera = camera.scaled(width, round(camera.height * width / camera.width))
        ppr = camera.pixels_per_radian()
        c = photo.centre

        near = index.near(c[0], c[1], max_distance_m)
        if not len(near):
            continue
        to_cam = c[None, :2] - centres[near, :2]
        dist_h = np.linalg.norm(to_cam, axis=1)
        facing = (to_cam * normals[near]).sum(axis=1) / np.maximum(dist_h, 1e-6)
        keep = (dist_h <= max_distance_m) & (dist_h > 0.5) & (facing >= cos_max)
        if not photo.heading_known and camera.kind != SPHERICAL:
            # A narrow camera of unknown heading could be looking anywhere.
            keep[:] = False
        near, facing = near[keep], facing[keep]
        if not len(near):
            continue

        saw_any = False
        candidates = index.near(c[0], c[1], max_distance_m)
        for face_index, cos_inc in zip(near, facing):
            face = faces[face_index]
            samples = face.samples
            _, in_image = project_world(camera, photo.rotation, c, samples)
            if camera.kind == SPHERICAL and not photo.heading_known:
                in_image[:] = True
            if not in_image.any():
                continue
            hidden = index.blocked(
                c, samples, candidates, np.full(len(samples), face_index)
            )
            visible = float((in_image & ~hidden).mean())
            if visible < min_visible:
                continue
            centre = face.centre
            distance = float(np.linalg.norm(centre - c))
            elevation = np.arctan2(abs(centre[2] - c[2]), max(distance, 1e-6))
            across = ppr * float(cos_inc) / distance
            up = ppr * float(np.cos(elevation)) / distance
            views[face_index].append(
                View(
                    photo=photo.id,
                    visible=visible,
                    px_per_m=min(across, up),
                    incidence_deg=float(np.degrees(np.arccos(np.clip(cos_inc, -1, 1)))),
                    distance_m=distance,
                )
            )
            saw_any = True
        used += int(saw_any)

    for i, face_views in enumerate(views):
        face_views.sort(key=lambda v: (v.visible >= 0.6, v.px_per_m * v.visible), reverse=True)
        views[i] = face_views[:VIEWS_PER_FACE]

    return Coverage(faces=faces, views=views, photos_used=used, photos_total=len(photos))


def write_coverage(coverage: Coverage, path: Path) -> Path:
    """Faces, their views and the summary, for the later stages and for people."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faces = []
    for i, face in enumerate(coverage.faces):
        faces.append(
            {
                "index": i,
                "building": face.building_id,
                "start": [round(float(v), 3) for v in face.start],
                "end": [round(float(v), 3) for v in face.end],
                "z": [round(face.z_min, 2), round(face.z_max, 2)],
                "normal": [round(float(v), 4) for v in face.normal],
                "area_m2": round(face.area_m2, 1),
                "exposed": face.exposed,
                "class": coverage.face_class(i),
                "views": [v.to_dict() for v in coverage.views[i]],
            }
        )
    path.write_text(
        json.dumps({"summary": coverage.stats(), "faces": faces}, indent=1),
        encoding="utf-8",
    )
    return path


CLASS_COLOURS = {
    TEXTURE: (46, 160, 67),
    MEASURE: (240, 160, 40),
    SEEN: (200, 80, 70),
    UNSEEN: (150, 150, 150),
}


def render_coverage_map(
    coverage: Coverage,
    photos: list[Photo],
    bbox,
    path: Path,
    *,
    m_per_px: float = 0.5,
    background: Path | None = None,
) -> Path:
    """A plan of the area: walls coloured by what the photos can do with them.

    Green walls can wear a photo, orange ones can be measured, red ones are
    seen but too far or too skew for either, grey ones are not seen at all.
    Dots are the cameras. Party walls are not drawn.
    """
    from PIL import Image, ImageDraw, ImageEnhance

    width = max(1, int(round(bbox.width / m_per_px)))
    height = max(1, int(round(bbox.height / m_per_px)))
    if background is not None and background.is_file():
        image = Image.open(background).convert("RGB").resize((width, height))
        image = ImageEnhance.Brightness(ImageEnhance.Color(image).enhance(0.2)).enhance(0.55)
    else:
        image = Image.new("RGB", (width, height), (32, 34, 38))
    draw = ImageDraw.Draw(image)

    def px(x: float, y: float) -> tuple[float, float]:
        return ((x - bbox.xmin) / m_per_px, (bbox.ymax - y) / m_per_px)

    for photo in photos:
        u, v = px(photo.x, photo.y)
        colour = (90, 170, 255) if photo.source == "mapillary" else (255, 90, 220)
        draw.ellipse((u - 1.5, v - 1.5, u + 1.5, v + 1.5), fill=colour)

    line = max(2, int(round(1.2 / m_per_px)))
    for i, face in enumerate(coverage.faces):
        if not face.exposed:
            continue
        draw.line(
            [px(*face.start), px(*face.end)],
            fill=CLASS_COLOURS[coverage.face_class(i)],
            width=line,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


__all__ = [
    "Coverage",
    "MEASURE",
    "SEEN",
    "TEXTURE",
    "UNSEEN",
    "View",
    "WallFace",
    "mark_exposed",
    "render_coverage_map",
    "score_views",
    "wall_faces",
    "write_coverage",
]
