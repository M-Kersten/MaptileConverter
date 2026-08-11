"""Railways, trams and metro from the BGT.

The first linear dataset in the pipeline. Everything else arrives as polygons
to fill or points to stand something on; ``spoor`` arrives as one centreline
per track, so the geometry has to be built out sideways from a line rather than
filled in from an outline.

What comes back is one line per running track, not one per route, so a station
throat resolves into individual tracks through the points. ``functie`` separates
heavy rail from tram and light rail, which matters because they are built
differently: a railway sits on a raised ballast bed and a city tram is set flush
into the street, and drawing a gravel bed down a shopping street would be worse
than drawing nothing.

Sleepers are in the ballast texture rather than in the geometry. There are
38 km of track within a kilometre of Utrecht Centraal, which at a sleeper every
600 mm is 63,000 of them; as a texture repeating along the track they are free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bgt import fetch_current
from .geo import BBox

LOG = logging.getLogger(__name__)

KIND_TRAIN = 0
KIND_TRAM = 1
KIND_METRO = 2

KIND_NAMES = {KIND_TRAIN: "train", KIND_TRAM: "tram", KIND_METRO: "metro"}

# BGT `functie` on a spoor, mapped to how the track is built.
RAIL_FUNCTIONS: dict[str, int] = {
    "trein": KIND_TRAIN,
    "tram": KIND_TRAM,
    "sneltram": KIND_METRO,
    "metro": KIND_METRO,
}


@dataclass(frozen=True)
class TrackProfile:
    """How wide the bed is and how the rails sit on it, in metres."""

    ballast_half_width: float
    ballast_lift: float
    rail_height: float
    rail_half_width: float
    # Standard gauge, which every one of these uses in the Netherlands.
    gauge: float = 1.435


PROFILES: dict[int, TrackProfile] = {
    # A railway sits on a ballast shoulder well wider than the sleepers.
    KIND_TRAIN: TrackProfile(
        ballast_half_width=1.95, ballast_lift=0.12, rail_height=0.20,
        rail_half_width=0.04,
    ),
    # A street tram has no bed at all: the rails are set into the paving, so
    # the road surface underneath is what shows between them.
    KIND_TRAM: TrackProfile(
        ballast_half_width=0.0, ballast_lift=0.0, rail_height=0.03,
        rail_half_width=0.035,
    ),
    KIND_METRO: TrackProfile(
        ballast_half_width=1.65, ballast_lift=0.10, rail_height=0.16,
        rail_half_width=0.04,
    ),
}

# How far apart points along a track may be before the ground beneath it stops
# being followed. Rails run over embankments and under bridges, so this is what
# keeps a straight line from cutting through a slope.
DEFAULT_STEP_M = 4.0

# A miter join at a sharp vertex would run away to infinity, so the widening is
# capped. Rail curves are gentle, so this almost never binds.
MAX_MITER = 2.5


@dataclass
class RailLine:
    """One running track."""

    points: np.ndarray  # (n, 2) in RD
    kind: int
    # BGT relatieve_hoogteligging: 0 at grade, above zero on a viaduct, below
    # in a tunnel.
    level: int

    @property
    def length_m(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


@dataclass
class RailSet:
    lines: list[RailLine] = field(default_factory=list)
    # Built once the terrain is known, since the track follows the ground.
    ballast_tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    ballast_kind: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    ballast_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    rail_tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    rail_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    rail_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    counts: dict[str, int] = field(default_factory=dict)
    stats_by_collection: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.lines)

    def length_m(self, kind: int | None = None) -> float:
        return float(
            sum(
                line.length_m
                for line in self.lines
                if kind is None or line.kind == kind
            )
        )

    def stats(self) -> dict:
        out: dict = {
            "tracks": len(self.lines),
            "track_length_m": round(self.length_m(), 1),
        }
        for kind, name in KIND_NAMES.items():
            length = self.length_m(kind)
            if length:
                out[f"{name}_length_m"] = round(length, 1)
        out.update(self.counts)
        return out


def clip_line_to_bbox(points: np.ndarray, bbox: BBox) -> list[np.ndarray]:
    """Trim a polyline to the bbox, returning the pieces that survive.

    A railway does not stop at the edge of the area, and the BGT returns any
    track that touches it in full: one line out of Utrecht Centraal stretched
    the model to 1592 m across an 800 m box. Water had the same problem and the
    same answer — a building is kept whole because cutting one opens it up, but
    a line has no inside to expose.

    Several pieces come back when a track leaves the area and returns, which
    happens wherever a curve clips a corner.
    """
    points = np.asarray(points, dtype=np.float64)[:, :2]
    if len(points) < 2:
        return []

    pieces: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []

    for start, end in zip(points[:-1], points[1:]):
        span = _clip_segment(start, end, bbox)
        if span is None:
            # Outside: whatever run was open has ended.
            if len(current) >= 2:
                pieces.append(current)
            current = []
            continue

        t0, t1 = span
        delta = end - start
        entry = start + delta * t0
        exit_ = start + delta * t1

        if current and np.allclose(current[-1], entry, atol=1e-9):
            current.append(exit_)
        else:
            if len(current) >= 2:
                pieces.append(current)
            current = [entry, exit_]

        # A segment that leaves the box early ends the run.
        if t1 < 1.0 - 1e-12:
            if len(current) >= 2:
                pieces.append(current)
            current = []

    if len(current) >= 2:
        pieces.append(current)

    return [np.asarray(piece) for piece in pieces if len(piece) >= 2]


def _clip_segment(start: np.ndarray, end: np.ndarray, bbox: BBox):
    """Liang-Barsky: the parameter range of a segment inside the box."""
    delta = end - start
    t0, t1 = 0.0, 1.0

    for numerator, denominator in (
        (start[0] - bbox.xmin, -delta[0]),
        (bbox.xmax - start[0], delta[0]),
        (start[1] - bbox.ymin, -delta[1]),
        (bbox.ymax - start[1], delta[1]),
    ):
        if abs(denominator) < 1e-12:
            # Parallel to this edge: either wholly inside it or wholly out.
            if numerator < 0:
                return None
            continue
        ratio = numerator / denominator
        if denominator < 0:
            if ratio > t1:
                return None
            t0 = max(t0, ratio)
        else:
            if ratio < t0:
                return None
            t1 = min(t1, ratio)

    return (t0, t1) if t1 > t0 else None


def densify(points: np.ndarray, max_step: float = DEFAULT_STEP_M) -> np.ndarray:
    """Insert points so no gap exceeds ``max_step``, keeping the originals.

    Resampling at a fixed interval instead would round off the curves: the BGT
    puts vertices where a track actually bends, so those are the points worth
    keeping. This only fills the long straights, where the ground underneath
    still has to be followed.
    """
    points = np.asarray(points, dtype=np.float64)[:, :2]
    if len(points) < 2:
        return points

    out = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        span = float(np.linalg.norm(end - start))
        if span <= 1e-9:
            continue
        steps = max(1, int(np.ceil(span / max_step)))
        for step in range(1, steps + 1):
            out.append(start + (end - start) * (step / steps))
    return np.asarray(out)


def offset_normals(points: np.ndarray) -> np.ndarray:
    """Per-vertex sideways direction, mitred so the ribbon has no gaps.

    Using each segment's own normal would leave a notch on the outside of every
    bend. Averaging the two adjacent normals and lengthening by ``1/cos`` keeps
    the edge continuous, which is what a mitre join is.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.zeros((len(points), 2))

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    lengths[lengths < 1e-12] = 1.0
    tangents = deltas / lengths[:, None]
    segment_normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    normals = np.empty_like(points)
    normals[0] = segment_normals[0]
    normals[-1] = segment_normals[-1]
    if len(points) > 2:
        normals[1:-1] = segment_normals[:-1] + segment_normals[1:]

    norms = np.linalg.norm(normals, axis=1)
    norms[norms < 1e-12] = 1.0
    normals /= norms[:, None]

    # 1/cos(half the turn) widens the join by exactly what the corner needs.
    cosines = np.ones(len(points))
    if len(points) > 2:
        cosines[1:-1] = np.einsum("ij,ij->i", normals[1:-1], segment_normals[:-1])
    cosines = np.clip(np.abs(cosines), 1.0 / MAX_MITER, 1.0)
    return normals / cosines[:, None]


def _ribbon(
    points: np.ndarray,
    z: np.ndarray,
    normals: np.ndarray,
    centre_offset: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """A flat strip along a line, as ``(triangles, uv)``.

    ``centre_offset`` shifts the whole strip sideways, which is how the two
    rails are placed either side of the track's centre.
    """
    if len(points) < 2 or half_width <= 0.0:
        return np.zeros((0, 3, 3)), np.zeros((0, 2))

    centre = points + normals * centre_offset
    left = centre + normals * half_width
    right = centre - normals * half_width

    left_3d = np.column_stack([left, z])
    right_3d = np.column_stack([right, z])

    # Distance along the track, so a tiling texture keeps its scale through
    # curves and the sleepers stay evenly spaced.
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])

    triangles = []
    uvs = []
    for i in range(len(points) - 1):
        a, b = left_3d[i], right_3d[i]
        c, d = left_3d[i + 1], right_3d[i + 1]
        triangles.append([a, b, d])
        triangles.append([a, d, c])
        v0, v1 = along[i], along[i + 1]
        uvs.extend([[0.0, v0], [1.0, v0], [1.0, v1]])
        uvs.extend([[0.0, v0], [1.0, v1], [0.0, v1]])

    return np.asarray(triangles), np.asarray(uvs)


def build_track_geometry(
    rails: RailSet,
    sampler,
    *,
    step_m: float = DEFAULT_STEP_M,
    deck_sampler=None,
) -> None:
    """Turn centrelines into ballast and rail geometry, draped on the ground.

    ``deck_sampler`` supplies the height of track that the BGT marks as being
    above ground level. Without one, a viaduct is drawn at ground level and
    said to be.
    """
    ballast_tris, ballast_kind, ballast_uv = [], [], []
    rail_tris, rail_kind, rail_uv = [], [], []

    for line in rails.lines:
        points = densify(line.points, step_m)
        if len(points) < 2:
            continue

        profile = PROFILES.get(line.kind, PROFILES[KIND_TRAIN])
        normals = offset_normals(points)

        # One height across the track's width, because a track bed is level
        # across it. Sampling each edge separately would twist the rails.
        ground = np.asarray(sampler(points[:, 0], points[:, 1]), dtype=np.float64)
        if line.level > 0 and deck_sampler is not None:
            ground = deck_sampler(points, ground)

        if profile.ballast_half_width > 0.0:
            tris, uv = _ribbon(
                points,
                ground + profile.ballast_lift,
                normals,
                0.0,
                profile.ballast_half_width,
            )
            if len(tris):
                ballast_tris.append(tris)
                ballast_uv.append(uv)
                ballast_kind.append(np.full(len(tris), line.kind, dtype=np.int32))

        rail_z = ground + profile.ballast_lift + profile.rail_height
        for side in (-0.5, 0.5):
            tris, uv = _ribbon(
                points,
                rail_z,
                normals,
                side * profile.gauge,
                profile.rail_half_width,
            )
            if len(tris):
                rail_tris.append(tris)
                rail_uv.append(uv)
                rail_kind.append(np.full(len(tris), line.kind, dtype=np.int32))

    def stack(chunks, width):
        if not chunks:
            return np.zeros((0, 3, 3)) if width == 3 else np.zeros((0, 2))
        return np.concatenate(chunks)

    rails.ballast_tris = stack(ballast_tris, 3)
    rails.ballast_uv = stack(ballast_uv, 2)
    rails.ballast_kind = (
        np.concatenate(ballast_kind) if ballast_kind else np.zeros(0, dtype=np.int32)
    )
    rails.rail_tris = stack(rail_tris, 3)
    rails.rail_uv = stack(rail_uv, 2)
    rails.rail_kind = (
        np.concatenate(rail_kind) if rail_kind else np.zeros(0, dtype=np.int32)
    )


def deck_from_dsm(dsm_path: Path, terrain_min_clear: float = 1.5, max_clear: float = 25.0):
    """A height source for elevated track, read off the surface model.

    A viaduct deck is a hard surface, so the AHN DSM sees it where the DTM —
    which is the bare ground by definition — does not. That makes the deck
    height a measurement rather than a guess, which matters because a fifth of
    the track around Utrecht Centraal is up on one.

    The reading is only trusted where it is plausibly a deck: high enough above
    the ground to be one, low enough not to be a gantry or a passing catenary
    mast. Anywhere else the track falls back to the ground.
    """
    import rasterio

    dataset = rasterio.open(dsm_path)
    band = dataset.read(1).astype(np.float64)
    left, bottom, right, top = (
        dataset.bounds.left,
        dataset.bounds.bottom,
        dataset.bounds.right,
        dataset.bounds.top,
    )
    height, width = band.shape
    dataset.close()

    from .elevation import NODATA_CUTOFF

    def sample(points: np.ndarray, ground: np.ndarray) -> np.ndarray:
        columns = np.clip(
            ((points[:, 0] - left) / max(right - left, 1e-9) * width).astype(int),
            0,
            width - 1,
        )
        # Raster rows run north-down.
        rows = np.clip(
            ((top - points[:, 1]) / max(top - bottom, 1e-9) * height).astype(int),
            0,
            height - 1,
        )
        deck = band[rows, columns]
        clearance = deck - ground
        usable = (
            (deck < NODATA_CUTOFF)
            & (clearance > terrain_min_clear)
            & (clearance < max_clear)
        )
        return np.where(usable, deck, ground)

    return sample


def build_rails(
    bbox: BBox,
    work_dir: Path,
    *,
    rails_cfg: dict,
    terrain=None,
) -> RailSet:
    """Fetch the tracks and build them into geometry."""
    result = RailSet()

    features, stats = fetch_current(
        "spoor",
        bbox,
        class_field="functie",
        page_limit=int(rails_cfg.get("page_limit", 1000)),
        timeout=float(rails_cfg.get("timeout_s", 180)),
        max_retries=int(rails_cfg.get("max_retries", 4)),
        max_pages=int(rails_cfg.get("max_pages", 200)),
    )
    result.stats_by_collection["spoor"] = stats.summary()

    skipped_kind = 0
    tunnels = 0
    for feature in features:
        properties = feature.get("properties") or {}
        function = str(properties.get("functie") or "").strip().lower()
        kind = RAIL_FUNCTIONS.get(function)
        if kind is None:
            # Harbour crane rails and the like come through as "niet-bgt".
            skipped_kind += 1
            continue

        try:
            level = int(properties.get("relatieve_hoogteligging") or 0)
        except (TypeError, ValueError):
            level = 0
        if level < 0:
            # A tunnel drawn on the surface is a track through the pavement.
            tunnels += 1
            continue

        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        points = np.asarray(geometry.get("coordinates") or [], dtype=np.float64)
        if points.ndim != 2 or len(points) < 2:
            continue

        for piece in clip_line_to_bbox(points, bbox):
            result.lines.append(RailLine(piece, kind, level))

    result.counts["tracks_skipped_not_rail"] = skipped_kind
    result.counts["tracks_in_tunnel"] = tunnels
    elevated = sum(1 for line in result.lines if line.level > 0)
    result.counts["tracks_elevated"] = elevated

    if not result.lines:
        LOG.info("no railway in this area")
        return result

    if terrain is None:
        return result

    deck_sampler = None
    if elevated:
        dsm_path = work_dir / "ahn_dsm.tif"
        if dsm_path.is_file():
            deck_sampler = deck_from_dsm(dsm_path)
            LOG.info(
                "%d elevated tracks will take their deck height from the AHN "
                "surface model", elevated,
            )
        else:
            LOG.warning(
                "%d of %d tracks are on a viaduct, and will be drawn at ground "
                "level: the AHN surface model (work/<area>/ahn_dsm.tif) is not "
                "there to measure the deck against. Enabling trees fetches it.",
                elevated,
                len(result.lines),
            )

    build_track_geometry(
        result,
        terrain.sample,
        step_m=float(rails_cfg.get("step_m", DEFAULT_STEP_M)),
        deck_sampler=deck_sampler,
    )

    LOG.info(
        "railways: %d tracks, %.2f km (%s), %d ballast and %d rail triangles",
        len(result.lines),
        result.length_m() / 1000.0,
        ", ".join(
            f"{KIND_NAMES[k]} {result.length_m(k) / 1000.0:.2f} km"
            for k in KIND_NAMES
            if result.length_m(k) > 0
        ),
        len(result.ballast_tris),
        len(result.rail_tris),
    )
    return result


def save_rails(rails: RailSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ballast_tris=rails.ballast_tris,
        ballast_kind=rails.ballast_kind,
        ballast_uv=rails.ballast_uv,
        rail_tris=rails.rail_tris,
        rail_kind=rails.rail_kind,
        rail_uv=rails.rail_uv,
    )
    LOG.info("wrote %s (%d tracks)", path, len(rails.lines))
    return path


__all__ = [
    "DEFAULT_STEP_M",
    "KIND_METRO",
    "KIND_NAMES",
    "KIND_TRAIN",
    "KIND_TRAM",
    "PROFILES",
    "RAIL_FUNCTIONS",
    "RailLine",
    "RailSet",
    "TrackProfile",
    "build_rails",
    "build_track_geometry",
    "clip_line_to_bbox",
    "deck_from_dsm",
    "densify",
    "offset_normals",
    "save_rails",
]
