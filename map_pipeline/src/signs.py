"""Traffic signs: which sign stands where, which way it faces, and what is on it.

The BGT barely registers them. Over one square kilometre of Utrecht centre it
has 16 sign posts, each a pole with no sign on it, where the street has 570
poles carrying 696 signs. Those come from NDW, the national road-data portal,
which keeps a register of every sign on every Dutch road: its RVV code, the
number on it, the plates under it and the road section it belongs to,
collected by driving the roads with cameras and published as open data.

What the register gives and what it does not:

- **Positions per sign, not per pole.** Two signs on one pole are two records
  a few centimetres to a metre apart. Poles are rebuilt by joining records
  closer than ``signs.pole_merge_m``.
- **A facing rounded to the compass.** 94% of bearings are 0, 90, 180 or 270,
  so a sign on a street running at 30 degrees would face 30 degrees off. The
  bearing is the direction the traffic it addresses drives -- measured over
  Utrecht centre, 453 of 459 signs within 45 degrees of their own road
  section's direction -- so the sign faces the other way, and the exact angle
  is taken from the NWB road section NDW links it to.
- **No heights or sizes.** Those are the regulation's: a 60 cm disc, a 90 cm
  triangle, the lowest plate 2.2 m up so a cyclist passes under it. All of it
  is in ``signs`` in the config.
- **Overhead signs** (``placement: ABOVE``) hang over the road on something the
  register does not describe: a span wire, a gantry, a bridge's face. They are
  hung at ``signs.overhead_clearance_m`` from a short hanger, and a height
  limit sign (C19) just above the limit it shows, which is where the bridge it
  is fixed to is.

The register is not complete. The ministry bought a selection of RVV codes,
the drive-by survey is yearly, and about one record in twenty has no code
(``onbekend``). Where the BGT has a sign post with no NDW pole near it, the
furniture stage's plain post stays.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import BBox
from .http_util import get_with_retry, short_error
from .sign_faces import (
    SHAPE_PLATE,
    SHAPE_RECT,
    SHAPE_WIDTH,
    UNKNOWN,
    Atlas,
    CommonsClient,
    black_text,
    build_atlas,
    code_parts,
    rank_of,
)

LOG = logging.getLogger(__name__)

NDW_URL = (
    "https://data.ndw.nu/api/rest/static-road-data/traffic-signs/v5/current-state"
)
# A municipality that does not exist: answers in a second with an empty list,
# where the bare endpoint starts streaming every sign in the country.
NDW_PROBE_URL = NDW_URL + "?countyCode=GM0000"
MUNICIPALITIES_URL = (
    "https://api.pdok.nl/kadaster/bestuurlijkegebieden/ogc/v1/"
    "collections/gemeentegebied/items"
)
RD_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"
ATTRIBUTION = (
    "Traffic signs from the NDW traffic sign register (Nationaal Dataportaal "
    "Wegverkeer), open data. https://docs.ndw.nu/producten/verkeersborden/"
)

POLE_SIDES = 8


@dataclass
class Sign:
    """One sign from the register, in RD."""

    x: float
    y: float
    code: str
    black: str = ""
    # (code, text) per onderbord, top first.
    plates: list = field(default_factory=list)
    # The way the face looks, radians anticlockwise from east.
    facing: float = 0.0
    # Where the facing came from: "section", "nearest road" or "bearing".
    facing_from: str = "bearing"
    overhead: bool = False
    ndw_id: str = ""
    road_section: int | None = None
    # As the register has them: the compass bearing the addressed traffic
    # drives, rounded to a quadrant, and FORTH or BACK along the section.
    bearing: float | None = None
    direction: str = ""
    pole: int = -1
    # Filled when laid out: the plate's centre height above the ground.
    centre_z: float = 0.0


@dataclass
class SignSet:
    """Everything the Blender stage draws, and what the checks look at."""

    signs: list = field(default_factory=list)
    pole_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pole_ground: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pole_top: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    loops: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    atlas: Atlas | None = None
    counts: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.signs)

    def stats(self) -> dict:
        return {"count": len(self), **self.counts}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def municipalities(bbox: BBox, *, timeout: float = 60.0, cache_dir: Path | None = None) -> list[str]:
    """The CBS codes (``GM0344``) of every municipality the bbox touches.

    NDW serves its register per municipality and has no bbox filter, so a
    square on a border reads both.
    """
    cache = None
    if cache_dir is not None:
        key = f"{bbox.xmin:.0f}_{bbox.ymin:.0f}_{bbox.xmax:.0f}_{bbox.ymax:.0f}"
        cache = Path(cache_dir) / f"municipalities_{key}.json"
        if cache.is_file():
            try:
                return list(json.loads(cache.read_text(encoding="utf-8")))
            except ValueError:
                pass
    response = get_with_retry(
        MUNICIPALITIES_URL,
        params={
            "bbox": f"{bbox.xmin:.3f},{bbox.ymin:.3f},{bbox.xmax:.3f},{bbox.ymax:.3f}",
            "bbox-crs": RD_URI,
            "f": "json",
            "limit": 50,
        },
        timeout=timeout,
        max_retries=3,
        description="PDOK municipalities",
    )
    codes = sorted(
        {
            str((f.get("properties") or {}).get("identificatie") or "")
            for f in response.json().get("features") or []
        }
        - {""}
    )
    if cache is not None and codes:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(codes), encoding="utf-8")
    return codes


def fetch_register(
    code: str,
    cache_dir: Path,
    *,
    url: str = NDW_URL,
    timeout: float = 300.0,
    max_retries: int = 3,
    cache_days: float = 30.0,
) -> list[dict]:
    """Every sign in one municipality, as NDW's GeoJSON features.

    Kept for `cache_days` under ``work/_ndw/``, shared by every area in the
    municipality: NDW rebuilds the register once a month, and Utrecht alone is
    18 MB.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{code}.json.gz"
    if path.is_file() and (time.time() - path.stat().st_mtime) < cache_days * 86400:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            features = json.load(handle)
        LOG.info("NDW %s: %d signs (cached)", code, len(features))
        return features
    started = time.time()
    response = get_with_retry(
        url,
        params={"countyCode": code},
        timeout=timeout,
        max_retries=max_retries,
        description=f"NDW traffic signs {code}",
    )
    features = response.json().get("features") or []
    cache_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(features, handle)
    LOG.info(
        "NDW %s: %d signs, %.1f MB in %.0f s",
        code,
        len(features),
        len(response.content) / 1e6,
        time.time() - started,
    )
    return features


def parse_signs(features: list[dict], bbox: BBox) -> list[Sign]:
    """The register's records inside the bbox, in RD."""
    from pyproj import Transformer

    rows = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if geometry.get("type") != "Point" or len(coords) < 2:
            continue
        rows.append((float(coords[0]), float(coords[1]), feature))
    if not rows:
        return []
    transform = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    xs, ys = transform.transform(
        np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    )
    out = []
    seen = set()
    for x, y, (_, _, feature) in zip(xs, ys, rows):
        if not (bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax):
            continue
        props = feature.get("properties") or {}
        ndw_id = str(feature.get("id") or "")
        # A municipality boundary runs along a road now and then, and a sign
        # on it can be in both registers.
        if ndw_id and ndw_id in seen:
            continue
        seen.add(ndw_id)
        plates = [
            (str(s.get("signCode") or ""), str(s.get("text") or ""))
            for s in sorted(props.get("supplementarySigns") or [], key=lambda s: s.get("index", 0))
        ]
        bearing = props.get("bearing")
        section = props.get("roadSectionId")
        sign = Sign(
            x=float(x),
            y=float(y),
            code=str(props.get("rvvCode") or UNKNOWN),
            black=str(props.get("blackCode") or ""),
            plates=plates,
            overhead=str(props.get("placement") or "").upper() == "ABOVE",
            ndw_id=ndw_id,
            road_section=int(section) if section not in (None, "") else None,
            bearing=float(bearing) if bearing is not None else None,
            direction=str(props.get("drivingDirection") or "").upper(),
        )
        out.append(sign)
    return out


# ---------------------------------------------------------------------------
# Facing
# ---------------------------------------------------------------------------


def compass_to_angle(bearing_deg: float) -> float:
    """Compass degrees (clockwise from north) to radians anticlockwise from east."""
    return math.radians(90.0 - float(bearing_deg))


def _nearest_on(points: np.ndarray, x: float, y: float) -> tuple[float, np.ndarray]:
    """Distance to a polyline and the direction of the segment nearest."""
    a = points[:-1]
    ab = points[1:] - a
    length2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    t = np.clip(((np.array([x, y]) - a) * ab).sum(axis=1) / length2, 0.0, 1.0)
    nearest = a + ab * t[:, None]
    d = np.hypot(nearest[:, 0] - x, nearest[:, 1] - y)
    i = int(np.argmin(d))
    return float(d[i]), ab[i] / math.sqrt(length2[i])


def set_facing(signs: list[Sign], network, *, search_m: float = 25.0) -> dict[str, int]:
    """Turn every sign to face the traffic it addresses, at the road's angle.

    The register's bearing is the direction that traffic drives, rounded to
    the compass. The road section it names gives the true direction: along the
    section where NDW says FORTH, against it where it says BACK, and where it
    says neither, whichever of the two the bearing is nearer. A sign with no
    section takes the nearest road in the same way, and only one with no road
    within `search_m` keeps the rounded bearing.
    """
    lines = list(getattr(network, "lines", None) or [])
    by_section = {line.section_id: line for line in lines}
    tree = geoms = None
    if lines:
        import shapely

        geoms = [shapely.LineString(line.points) for line in lines]
        tree = shapely.STRtree(geoms)

    counts = {"section": 0, "nearest road": 0, "bearing": 0}
    for sign in signs:
        travel = None
        if sign.bearing is not None:
            b = compass_to_angle(sign.bearing)
            travel = np.array([math.cos(b), math.sin(b)])
        direction = sign.direction

        tangent, source = None, "bearing"
        line = by_section.get(sign.road_section) if sign.road_section is not None else None
        if line is not None and len(line.points) >= 2:
            _, tangent = _nearest_on(np.asarray(line.points, float), sign.x, sign.y)
            source = "section"
        elif tree is not None:
            import shapely

            index = tree.query_nearest(
                shapely.Point(sign.x, sign.y), max_distance=search_m, all_matches=False
            )
            if len(index):
                points = np.asarray(lines[int(index[0])].points, float)
                if len(points) >= 2:
                    _, tangent = _nearest_on(points, sign.x, sign.y)
                    source = "nearest road"
                    direction = ""  # FORTH and BACK are about the named section

        if tangent is not None:
            if direction == "BACK":
                tangent = -tangent
            elif direction != "FORTH" and travel is not None and float(tangent @ travel) < 0:
                tangent = -tangent
            elif direction != "FORTH" and travel is None:
                source = "bearing"
                tangent = None
        if tangent is not None:
            sign.facing = math.atan2(-tangent[1], -tangent[0])
        elif travel is not None:
            sign.facing = math.atan2(-travel[1], -travel[0])
            source = "bearing"
        else:
            source = "bearing"
        sign.facing_from = source
        counts[source] += 1
    return counts


# ---------------------------------------------------------------------------
# Poles
# ---------------------------------------------------------------------------


def group_poles(signs: list[Sign], merge_m: float) -> np.ndarray:
    """Join signs closer than `merge_m` into one pole. Returns pole centres.

    Single linkage over a grid, so a pole of five signs spread a metre along
    the kerb is one pole however its records happen to be ordered. Overhead
    signs hang on their own and get no pole.
    """
    standing = [i for i, s in enumerate(signs) if not s.overhead]
    parent = {i: i for i in standing}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cell = max(merge_m, 1e-3)
    grid: dict[tuple[int, int], list[int]] = {}
    for i in standing:
        key = (int(math.floor(signs[i].x / cell)), int(math.floor(signs[i].y / cell)))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((key[0] + dx, key[1] + dy), ()):
                    if math.hypot(signs[i].x - signs[j].x, signs[i].y - signs[j].y) <= merge_m:
                        parent[root(i)] = root(j)
        grid.setdefault(key, []).append(i)

    poles: dict[int, int] = {}
    centres: list[list[float]] = []
    members: list[list[int]] = []
    for i in standing:
        r = root(i)
        if r not in poles:
            poles[r] = len(centres)
            centres.append([0.0, 0.0])
            members.append([])
        members[poles[r]].append(i)
        signs[i].pole = poles[r]
    for p, group in enumerate(members):
        centres[p] = [
            float(np.mean([signs[i].x for i in group])),
            float(np.mean([signs[i].y for i in group])),
        ]
    return np.asarray(centres, dtype=np.float64).reshape(-1, 2)


def _angle_gap(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


@dataclass
class Plate:
    """One rectangle to draw: a sign or an onderbord, with its face."""

    sign: int
    key: tuple
    width: float
    height: float
    facing: float
    bottom: float = 0.0  # above the ground, filled by the layout


def plate_size(face, shape: str, size_m: float, plate_width_m: float | None = None) -> tuple[float, float]:
    """Width and height in metres, from the regulation's series and the art."""
    aspect = max(0.2, float(getattr(face, "aspect", 1.0) or 1.0))
    if shape == SHAPE_PLATE:
        width = plate_width_m if plate_width_m else size_m
        return width, width / aspect
    width = size_m * SHAPE_WIDTH.get(shape, 1.0)
    height = width / aspect
    # A rectangle whose art is taller than wide is held to the height of the
    # series rather than allowed to grow with it. Only a rectangle: a 90 cm
    # triangle is 78 cm tall, and held to 72 it came out 83 cm wide.
    if shape == SHAPE_RECT and height > size_m * 1.2:
        height = size_m * 1.2
        width = height * aspect
    return width, height


def layout_pole(plates_by_sign: list[list[Plate]], *, gap_m: float, clearance_m: float) -> float:
    """Stack the plates on one pole, and return the pole's height.

    Signs facing the same way share a stack, most important on top, each
    followed by its own onderborden. Two stacks facing opposite ways hang back
    to back at the same heights; a stack at an angle to another goes below it,
    where the two cannot pass through each other.
    """
    stacks: list[dict] = []
    for plates in plates_by_sign:
        facing = plates[0].facing
        for stack in stacks:
            if _angle_gap(stack["facing"], facing) < math.radians(35):
                stack["plates"].extend(plates)
                break
        else:
            stacks.append({"facing": facing, "plates": list(plates)})

    levels: list[list[dict]] = []
    for stack in stacks:
        for level in levels:
            if len(level) == 1 and _angle_gap(level[0]["facing"], stack["facing"]) > math.radians(145):
                level.append(stack)
                break
        else:
            levels.append([stack])

    top = 0.0
    for level in levels:
        lowest = top
        for stack in level:
            z = top
            for plate in stack["plates"]:
                plate.bottom = z - plate.height
                z = plate.bottom - gap_m
            lowest = min(lowest, z + gap_m)
        top = lowest - gap_m * 2
    bottom = min(p.bottom for s in stacks for p in s["plates"])
    lift = clearance_m - bottom
    for stack in stacks:
        for plate in stack["plates"]:
            plate.bottom += lift
    return max(p.bottom + p.height for s in stacks for p in s["plates"])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class _Mesh:
    def __init__(self) -> None:
        self.vertices: list = []
        self.uv: list = []
        self.loops: list = []
        self.sizes: list = []

    def face(self, corners, uvs) -> None:
        start = len(self.vertices)
        self.vertices.extend(corners)
        self.uv.extend(uvs)
        self.loops.extend(range(start, start + len(corners)))
        self.sizes.append(len(corners))

    def prism(self, x, y, z0, z1, radius, uv) -> None:
        """An upright octagonal tube, capped at the top."""
        ring = [
            (x + radius * math.cos(2 * math.pi * k / POLE_SIDES),
             y + radius * math.sin(2 * math.pi * k / POLE_SIDES))
            for k in range(POLE_SIDES)
        ]
        for k in range(POLE_SIDES):
            a, b = ring[k], ring[(k + 1) % POLE_SIDES]
            self.face(
                [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)],
                [uv] * 4,
            )
        self.face([(p[0], p[1], z1) for p in ring], [uv] * POLE_SIDES)

    def plate(self, cx, cy, facing, bottom, width, height, front_uv, back_uv, offset, thickness) -> None:
        """A sign: its face looking along `facing`, its back behind it."""
        fx, fy = math.cos(facing), math.sin(facing)
        rx, ry = -fy, fx  # the right hand of someone reading it
        top = bottom + height
        half = width / 2
        u0, v0, u1, v1 = front_uv
        px, py = cx + fx * offset, cy + fy * offset
        self.face(
            [
                (px - rx * half, py - ry * half, bottom),
                (px + rx * half, py + ry * half, bottom),
                (px + rx * half, py + ry * half, top),
                (px - rx * half, py - ry * half, top),
            ],
            [(u0, v0), (u1, v0), (u1, v1), (u0, v1)],
        )
        # The back, a little behind and facing the other way. Its outline is
        # the face's, so the same corner of the cell goes on the same corner of
        # the plate: seen from behind it is the face's outline, mirrored as it
        # should be.
        bx, by = cx + fx * (offset - thickness), cy + fy * (offset - thickness)
        b0, w0, b1, w1 = back_uv
        self.face(
            [
                (bx + rx * half, by + ry * half, bottom),
                (bx - rx * half, by - ry * half, bottom),
                (bx - rx * half, by - ry * half, top),
                (bx + rx * half, by + ry * half, top),
            ],
            [(b1, w0), (b0, w0), (b0, w1), (b1, w1)],
        )

    def arrays(self):
        return (
            np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3),
            np.asarray(self.loops, dtype=np.int32),
            np.asarray(self.sizes, dtype=np.int32),
            np.asarray(self.uv, dtype=np.float64).reshape(-1, 2),
        )


def _on_deck(signs_xy: np.ndarray, deck_at, roads, terrain_z: np.ndarray) -> np.ndarray:
    """Which poles stand on a bridge deck rather than under it.

    A deck's outline covers both the road on it and, for a viaduct, the road
    under it. The nearest BGT road surface settles which: one at
    ``relatieve_hoogteligging`` above zero is on the bridge. With no road
    surfaces to ask, a deck too low to drive under is stood on.
    """
    deck = np.asarray(deck_at(signs_xy[:, 0], signs_xy[:, 1]), dtype=np.float64).reshape(-1)
    on = np.isfinite(deck)
    if not on.any():
        return np.full(len(signs_xy), np.nan)
    raised = None
    if roads:
        import shapely

        polygons, levels = [], []
        for part in roads:
            try:
                rings = [np.asarray(r, float) for r in part.rings]
                polygon = shapely.Polygon(rings[0], rings[1:])
            except Exception:  # noqa: BLE001 - one bad ring is not a reason to stop
                continue
            polygons.append(polygon)
            levels.append(int(getattr(part, "level", 0)))
        if polygons:
            tree = shapely.STRtree(polygons)
            raised = np.zeros(len(signs_xy), dtype=bool)
            for i in np.nonzero(on)[0]:
                hit = tree.query_nearest(
                    shapely.Point(signs_xy[i]), max_distance=8.0, all_matches=False
                )
                raised[i] = bool(len(hit)) and levels[int(hit[0])] > 0
    if raised is None:
        raised = (deck - terrain_z) < 4.0
    out = np.full(len(signs_xy), np.nan)
    keep = on & raised
    out[keep] = deck[keep]
    return out


def build_signs(
    bbox: BBox,
    work_dir: Path,
    *,
    signs_cfg: dict,
    terrain,
    network=None,
    deck_at=None,
    roads=None,
    session=None,
) -> SignSet:
    """Read the register, turn and stack every sign, and build the geometry."""
    result = SignSet()
    shared = work_dir.parent / "_ndw"
    timeout = float(signs_cfg.get("timeout_s", 300))
    codes = [str(c) for c in (signs_cfg.get("municipalities") or [])]
    if not codes:
        try:
            codes = municipalities(bbox, timeout=min(timeout, 60.0), cache_dir=work_dir / "_bgt")
        except Exception as exc:  # noqa: BLE001 - no signs is not a failed run
            LOG.warning("could not tell which municipality this is: %s", short_error(exc))
            return result
    features: list[dict] = []
    for code in codes:
        try:
            features.extend(
                fetch_register(
                    code,
                    shared,
                    url=str(signs_cfg.get("api_url") or NDW_URL),
                    timeout=timeout,
                    max_retries=int(signs_cfg.get("max_retries", 3)),
                    cache_days=float(signs_cfg.get("cache_days", 30)),
                )
            )
        except Exception as exc:  # noqa: BLE001 - a missing municipality is not fatal
            LOG.warning("could not read the NDW register for %s: %s", code, short_error(exc))
    signs = parse_signs(features, bbox)
    if not bool(signs_cfg.get("unknown", True)):
        signs = [s for s in signs if s.code != UNKNOWN]
    if not signs:
        LOG.info("no traffic signs in the register for this area")
        return result

    facing_counts = set_facing(signs, network, search_m=float(signs_cfg.get("road_search_m", 25.0)))
    pole_xy = group_poles(signs, float(signs_cfg.get("pole_merge_m", 0.5)))

    # The faces first: a plate's size follows from its artwork's proportions.
    wanted = [(s.code, s.black) for s in signs]
    plates_wanted = [p for s in signs for p in s.plates]
    client = CommonsClient(
        work_dir.parent / "_signs",
        enabled=bool(signs_cfg.get("commons", True)),
        pause_s=float(signs_cfg.get("commons_pause_s", 1.0)),
        session=session,
    )
    atlas = build_atlas(wanted, plates_wanted, work_dir / "signs.png", client=client)
    result.atlas = atlas

    size_m = float(signs_cfg.get("size_m", 0.6))
    gap_m = float(signs_cfg.get("gap_m", 0.05))
    clearance_m = float(signs_cfg.get("clearance_m", 2.2))
    pole_r = float(signs_cfg.get("pole_diameter_m", 0.06)) / 2
    thickness = 0.012
    offset = pole_r + 0.02

    def plates_for(index: int) -> list[Plate]:
        sign = signs[index]
        face = atlas.faces[("sign", sign.code, sign.black)]
        width, height = plate_size(face, face.shape, size_m)
        out = [Plate(index, face.key, width, height, sign.facing)]
        for code, text in sign.plates:
            sub = atlas.faces[("plate", code, text)]
            # As wide as the sign it explains, within reason.
            pw, ph = plate_size(sub, SHAPE_PLATE, size_m, plate_width_m=max(size_m, min(width, size_m * 1.5)))
            out.append(Plate(index, sub.key, pw, ph, sign.facing))
        return out

    # Ground under each pole: the terrain, or a bridge deck where the pole
    # stands on one.
    n_poles = len(pole_xy)
    ground = (
        np.asarray(terrain.surface(pole_xy[:, 0], pole_xy[:, 1]), dtype=np.float64)
        if n_poles
        else np.zeros(0)
    )
    on_deck = 0
    if deck_at is not None and n_poles:
        deck = _on_deck(pole_xy, lambda x, y: deck_at(x, y, reach=0.0), roads, ground)
        lifted = np.isfinite(deck)
        on_deck = int(lifted.sum())
        ground = np.where(lifted, deck, ground)

    mesh = _Mesh()
    metal = atlas.metal_uv
    metal_uv = (metal[0], metal[1], metal[0], metal[1])
    members: dict[int, list[int]] = {}
    for i, sign in enumerate(signs):
        if sign.pole >= 0:
            members.setdefault(sign.pole, []).append(i)
    pole_top = np.zeros(n_poles)
    for p in range(n_poles):
        order = sorted(members.get(p, []), key=lambda i: (rank_of(signs[i].code), signs[i].ndw_id))
        groups = [plates_for(i) for i in order]
        top = layout_pole(groups, gap_m=gap_m, clearance_m=clearance_m)
        pole_top[p] = top
        x, y = pole_xy[p]
        base = float(ground[p])
        mesh.prism(x, y, base - 0.15, base + top + 0.04, pole_r, metal)
        for plates in groups:
            for plate in plates:
                face = atlas.faces[plate.key]
                back = atlas.faces[face.back].uv if face.back else metal_uv
                mesh.plate(x, y, plate.facing, base + plate.bottom, plate.width, plate.height,
                           face.uv, back, offset, thickness)
            signs[plates[0].sign].centre_z = plates[0].bottom + plates[0].height / 2

    # Overhead: hung from a short hanger over the road, at the clearance, or
    # for a height limit just above the limit it shows.
    overhead = [i for i, s in enumerate(signs) if s.overhead]
    if overhead:
        xy = np.array([[signs[i].x, signs[i].y] for i in overhead])
        under = np.asarray(terrain.surface(xy[:, 0], xy[:, 1]), dtype=np.float64)
        default_clearance = float(signs_cfg.get("overhead_clearance_m", 4.6))
        for k, i in enumerate(overhead):
            sign = signs[i]
            plates = plates_for(i)
            clearance = default_clearance
            letter, number, _ = code_parts(sign.code)
            if letter == "C" and number == 19:
                try:
                    clearance = float(str(sign.black).replace(",", ".")) + 0.1
                except ValueError:
                    pass
            layout_pole([plates], gap_m=gap_m, clearance_m=clearance)
            top = max(p.bottom + p.height for p in plates)
            base = float(under[k])
            mesh.prism(sign.x, sign.y, base + top - 0.02, base + top + 0.5, 0.012, metal)
            for plate in plates:
                face = atlas.faces[plate.key]
                back = atlas.faces[face.back].uv if face.back else metal_uv
                mesh.plate(sign.x, sign.y, plate.facing, base + plate.bottom, plate.width, plate.height,
                           face.uv, back, 0.0, thickness)
            sign.centre_z = plates[0].bottom + plates[0].height / 2

    result.signs = signs
    result.pole_xy = pole_xy
    result.pole_ground = ground
    result.pole_top = pole_top
    result.vertices, result.loops, result.sizes, result.uv = mesh.arrays()
    faces = atlas.counts()
    result.counts = {
        "poles": int(n_poles),
        "overhead": len(overhead),
        "with_onderbord": sum(1 for s in signs if s.plates),
        "onderborden": sum(len(s.plates) for s in signs),
        "codes": len({s.code for s in signs}),
        "unknown_code": sum(1 for s in signs if s.code == UNKNOWN),
        "faced_by_road_section": facing_counts["section"],
        "faced_by_nearest_road": facing_counts["nearest road"],
        "faced_by_bearing_only": facing_counts["bearing"],
        "on_bridge_decks": on_deck,
        "faces_from_commons": faces.get("commons", 0),
        "faces_drawn": faces.get("drawn", 0),
        "plates_drawn": faces.get("plate", 0),
        "municipalities": codes,
    }
    LOG.info(
        "traffic signs: %d on %d poles and %d overhead, %d codes; faced by road "
        "section %d, nearest road %d, bearing only %d; faces %d from Commons, %d drawn",
        len(signs), n_poles, len(overhead), result.counts["codes"],
        facing_counts["section"], facing_counts["nearest road"], facing_counts["bearing"],
        faces.get("commons", 0), faces.get("drawn", 0),
    )
    return result


def save_signs(signs: SignSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices=signs.vertices,
        loops=signs.loops,
        sizes=signs.sizes,
        uv=signs.uv,
        pole_xy=signs.pole_xy,
        pole_ground=signs.pole_ground,
        pole_top=signs.pole_top,
    )
    LOG.info("wrote %s (%d signs, %d faces)", path, len(signs), len(signs.sizes))
    return path


def drop_covered_posts(furniture, signs: SignSet, *, radius_m: float = 3.0, sign_kind: int = 3) -> int:
    """Take the BGT's bare sign posts out where an NDW pole stands.

    The BGT surveyed the post; NDW surveyed the signs on it. Drawn twice it is
    two poles a metre apart, one of them with nothing on it.
    """
    if furniture is None or not len(furniture) or not len(signs.pole_xy):
        return 0
    is_post = furniture.kind == sign_kind
    if not is_post.any():
        return 0
    # A few dozen posts against a few hundred poles: a plain distance matrix.
    posts = furniture.xy[is_post]
    distance = np.full(len(posts), np.inf)
    for start in range(0, len(signs.pole_xy), 2048):
        chunk = signs.pole_xy[start : start + 2048]
        d = np.hypot(posts[:, None, 0] - chunk[None, :, 0], posts[:, None, 1] - chunk[None, :, 1])
        distance = np.minimum(distance, d.min(axis=1))
    covered = np.zeros(len(furniture), dtype=bool)
    covered[np.nonzero(is_post)[0][distance <= radius_m]] = True
    if not covered.any():
        return 0
    keep = ~covered
    furniture.xy = furniture.xy[keep]
    furniture.kind = furniture.kind[keep]
    furniture.ground_z_nap = furniture.ground_z_nap[keep]
    return int(covered.sum())


def write_sign_list(signs: SignSet, geo, out_dir: Path, filename: str = "signs.json") -> Path:
    """Every sign with its code, its plates and its facing, for an engine.

    The FBX draws them; a driving simulation wants to know what they say.
    Same frame and conventions as ``vehicles.json``.
    """
    records = []
    for sign in signs.signs:
        local_x, local_y = geo.rd_to_local(sign.x, sign.y)
        ground = float(signs.pole_ground[sign.pole]) if sign.pole >= 0 else None
        records.append(
            {
                "code": sign.code,
                "value": black_text(sign.code, sign.black) or None,
                "black_code": sign.black or None,
                "onderborden": [
                    {"code": code or None, "text": text or None} for code, text in sign.plates
                ],
                "rd": [round(sign.x, 3), round(sign.y, 3)],
                "local": {"x": round(local_x, 3), "z": round(local_y, 3)},
                "pole": sign.pole if sign.pole >= 0 else None,
                "overhead": sign.overhead,
                "ground_z_nap": round(ground, 3) if ground is not None else None,
                "centre_above_ground_m": round(sign.centre_z, 2),
                # The way the face looks, as a Unity Y rotation: clockwise
                # from north.
                "facing_deg": round((90.0 - math.degrees(sign.facing)) % 360.0, 1),
                "facing_from": sign.facing_from,
                "ndw_id": sign.ndw_id,
                "nwb_road_section": sign.road_section,
            }
        )
    payload = {
        "count": len(records),
        "crs": "EPSG:28992",
        "note": (
            "local.x / local.z are metres in the model frame. facing_deg is the "
            "direction the face looks, as a Unity Y rotation (clockwise from "
            "north); traffic reading the sign drives the opposite way"
        ),
        "source": ATTRIBUTION,
        "signs": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("wrote %s", path)
    return path


__all__ = [
    "ATTRIBUTION",
    "MUNICIPALITIES_URL",
    "NDW_PROBE_URL",
    "NDW_URL",
    "Plate",
    "Sign",
    "SignSet",
    "build_signs",
    "compass_to_angle",
    "drop_covered_posts",
    "fetch_register",
    "group_poles",
    "layout_pole",
    "municipalities",
    "parse_signs",
    "plate_size",
    "save_signs",
    "set_facing",
    "write_sign_list",
]
