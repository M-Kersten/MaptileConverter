"""Walls, fences, hedges and the small structures bolted onto buildings.

This is the largest thing the BGT publishes that the pipeline used to ignore.
Over one square kilometre of Utrecht centre:

    scheiding_lijn        1563   fence 1174, wall 332, quay wall 36, barrier 4
    scheiding_vlak        1450   wall 1330, quay wall 120
    gebouwinstallatie      114   entrance steps 55, awning 53, stoop 6
    overigbouwwerk          82   transformer 6, canopy 4, open shed 2
    vegetatieobject_lijn    34   hedge 34

Three thousand walls and fences is not decoration. They are what divides one
plot from the next, and a street without them reads as a set of buildings
standing in a field -- which is exactly what the model looked like. The quay
walls matter more still: an Utrecht canal without its wall is a trench.

None of it carries a height. The BGT surveys the footprint of a wall and the
line of a fence, and stops there. So the heights come from a table by type,
which is honest about being an assumption and puts it somewhere a level
designer can change it, rather than from AHN -- a fence is thinner than the
half-metre height grid can see, and a wall against a building takes the
building's height, which is the same trap the tree heights had to be dug out
of.

Two families of geometry, and they need different treatment:

* **Lines** sweep into a closed prism: offset either side by half the
  thickness, drape both edges on the terrain, and close the top and the ends.
  A fence is 6 cm thick and a wall 30, but neither is a plane -- a
  zero-thickness wall shows as a hole from one side in anything with backface
  culling on.
* **Polygons** extrude: the ring becomes vertical quads and the top gets a cap.
  The cap is earcut, which fans a thin strip into slivers -- see roadmesh for
  how much that matters on a surface. It does not matter here: the top of a
  30 cm wall is not lit, walked on, or ever seen from above, and the sides,
  which are the part you look at, are clean quads by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bgt import fetch_current, polygon_rings
from .geo import BBox
from .rails import clip_line_to_bbox, densify, offset_normals
from .surfaces import clip_ring_to_bbox

LOG = logging.getLogger(__name__)

# How far apart to put vertices along a swept line. A garden wall follows the
# ground and the ground moves; 2 m is fine enough for a Dutch gradient and
# coarse enough that 1563 fences stay cheap.
DEFAULT_STEP_M = 2.0

# Nothing shorter than this is worth a prism. Below about a metre a fence
# fragment is a survey artefact -- the end of a run clipped by a bbox, or a
# gate post recorded as its own object.
MIN_LENGTH_M = 0.8

# A barrier sinks this far into the ground, so a prism whose base was sampled at
# one point does not float at the other end of a slope. Only for the ones that
# stand on the ground: burying an awning by 15 cm would make a 12 cm awning
# 27 cm thick and hang it lower than it belongs.
EMBED_M = 0.15


def feature_type(properties: dict) -> str:
    """Which BGT field actually holds the type, for these collections.

    It is not the same field twice. `scheiding` and `overigbouwwerk` put the
    value in `type` and leave `plus_type` explicitly null, with
    `plus_type_leeg: waardeOnbekend` to say so; `gebouwinstallatie` and
    `vegetatieobject` do the reverse, carrying `plus_type: toegangstrap` beside
    a useless `type: niet-bgt`. Reading only `plus_type` is why the first run of
    this module fetched 1563 fences and 1450 walls and drew none of them.

    So: whichever is populated, preferring the finer one.
    """
    for key in ("plus_type", "type"):
        value = properties.get(key)
        if value and str(value).lower() not in ("none", "niet-bgt"):
            return str(value)
    return ""


def _drape_points(points: np.ndarray, sampler, step_m: float, tolerance_m: float):
    """Where a swept barrier needs a vertex, and the ground height there.

    Cutting every fence at a fixed step is the obvious thing and it is pure
    waste: Dutch ground barely moves, so 907 fences came to 56,540 triangles,
    most of them holding a straight line straight. What a vertex is *for* is
    the places the ground bends.

    So the line is cut finely, the ground sampled along it, and the inserted
    points thinned against the height profile -- keeping one wherever dropping
    it would let the barrier cut into the ground, or lift off it, by more than
    the tolerance. The survey's own vertices are anchors and always survive,
    because they are the shape of the fence rather than the shape of the
    ground under it.
    """
    from .roadmesh import _douglas_peucker_keep

    dense = densify(points, step_m)
    height = np.asarray(sampler(dense[:, 0], dense[:, 1]), dtype=np.float64)
    if len(dense) < 3 or tolerance_m <= 0:
        return dense, height

    # Which of the dense points are the survey's own. densify keeps them, so
    # they are the ones that match an input vertex exactly.
    steps = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    anchors = [0]
    at = 0
    for vertex in points[1:]:
        while at < len(dense) - 1 and not np.allclose(dense[at], vertex, atol=1e-9):
            at += 1
        if at != anchors[-1]:
            anchors.append(at)
    if anchors[-1] != len(dense) - 1:
        anchors.append(len(dense) - 1)

    # The profile as a 2D line of (distance along, height), so the ordinary
    # perpendicular-distance rule reads as vertical error. True while the
    # gradient is shallow, which in this country it is.
    keep = np.zeros(len(dense), dtype=bool)
    keep[anchors] = True
    profile = np.column_stack([along, height])
    for start, end in zip(anchors[:-1], anchors[1:]):
        if end > start + 1:
            keep[start : end + 1] |= _douglas_peucker_keep(
                profile[start : end + 1], tolerance_m
            )
    return dense[keep], height[keep]


def _prepare_rings(group, bbox: BBox) -> list[np.ndarray]:
    """One BGT polygon into rings the extruder can use, cut to the area.

    Cut, not merely centre-tested. This used to keep any polygon whose middle
    was inside, on the grounds that a wall poking a metre past the edge was not
    worth a seam -- and then a quay wall followed its canal 227 m past the bbox
    and took the model's bounding box with it. A wall is not a building:
    cutting one at the edge shows a cut wall, which is what the edge of the
    model looks like anyway.
    """
    rings = []
    for ring in group:
        ring = np.asarray(ring, dtype=np.float64)
        if ring.ndim != 2 or len(ring) < 3:
            continue
        clipped = clip_ring_to_bbox(ring[:, :2], bbox)
        if len(clipped) >= 3:
            rings.append(clipped)
    return rings


def _span(base_z, style: BarrierStyle):
    """The bottom and top of a barrier, given where the ground is."""
    if style.lift_m > 0.0:
        return base_z + style.lift_m, base_z + style.lift_m + style.height_m
    return base_z - EMBED_M, base_z + style.height_m


@dataclass(frozen=True)
class BarrierStyle:
    """What one BGT type is drawn as."""

    name: str
    height_m: float
    thickness_m: float
    material: str
    # Awnings and canopies hang off a facade. Drawn from the ground they are a
    # solid block against the wall, which is worse than leaving them out.
    lift_m: float = 0.0


# Linear types, by collection and plus_type.
#
# A hedge is thicker than a fence and shorter than a wall, and it is the one
# entry here whose height is nearly reliable -- a Dutch hedge is clipped to
# about the height of the fence it hides.
LINE_STYLES: dict[tuple[str, str], BarrierStyle] = {
    ("scheiding_lijn", "hek"): BarrierStyle("fence", 1.8, 0.06, "paint"),
    ("scheiding_lijn", "muur"): BarrierStyle("wall", 2.0, 0.24, "brick"),
    ("scheiding_lijn", "kademuur"): BarrierStyle("quay_wall", 1.1, 0.40, "concrete"),
    ("scheiding_lijn", "walbescherming"): BarrierStyle(
        "bank_protection", 0.5, 0.30, "concrete"
    ),
    ("scheiding_lijn", "geluidsscherm"): BarrierStyle(
        "noise_barrier", 3.0, 0.16, "concrete"
    ),
    ("scheiding_lijn", "damwand"): BarrierStyle("sheet_pile", 0.8, 0.20, "concrete"),
    ("vegetatieobject_lijn", "haag"): BarrierStyle("hedge", 1.5, 0.70, "hedge"),
}

# Areal types. A wall the BGT drew as a polygon is wider than half a metre, so
# its own outline gives the thickness and only the height is assumed.
AREA_STYLES: dict[tuple[str, str], BarrierStyle] = {
    ("scheiding_vlak", "muur"): BarrierStyle("wall", 2.0, 0.0, "brick"),
    ("scheiding_vlak", "kademuur"): BarrierStyle("quay_wall", 1.1, 0.0, "concrete"),
    ("gebouwinstallatie", "toegangstrap"): BarrierStyle("steps", 0.9, 0.0, "stone"),
    ("gebouwinstallatie", "bordes"): BarrierStyle("stoop", 0.4, 0.0, "stone"),
    ("gebouwinstallatie", "luifel"): BarrierStyle(
        "awning", 0.12, 0.0, "metal", lift_m=2.6
    ),
    ("overigbouwwerk", "overkapping"): BarrierStyle(
        "canopy", 0.14, 0.0, "metal", lift_m=2.8
    ),
    ("overigbouwwerk", "lage trafo"): BarrierStyle("transformer", 2.4, 0.0, "brick"),
    ("overigbouwwerk", "open loods"): BarrierStyle("shed", 2.8, 0.0, "metal"),
    ("overigbouwwerk", "bunker"): BarrierStyle("bunker", 2.5, 0.0, "concrete"),
    ("overigbouwwerk", "opslagtank"): BarrierStyle("tank", 4.0, 0.0, "metal"),
}

COLLECTIONS = sorted(
    {collection for collection, _ in (*LINE_STYLES, *AREA_STYLES)}
)


@dataclass
class BarrierSet:
    """Triangles with a material name each, ready for the Blender stage."""

    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))
    # One per triangle: which atlas patch it wears, and which BGT type it is.
    material: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    kind: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    material_names: list[str] = field(default_factory=list)
    kind_names: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.triangles))

    def stats(self) -> dict:
        return {"triangles": len(self), "vertices": int(len(self.vertices)), **self.counts}


def _sweep_line(points: np.ndarray, sampler, style: BarrierStyle):
    """One polyline into a closed prism, as ``(vertices, triangles)``.

    The base follows the terrain and the top follows the base, so a wall up a
    slope stays the same height all the way along rather than levelling off or
    burying itself.

    The ground is sampled at each rail, not once along the centreline. A wall
    has width, and the mitre at a corner widens it further -- up to two and a
    half times the half-thickness, which on a half-metre quay wall is half a
    metre sideways. That is nothing on a pavement and a lot on a canal bank,
    which is exactly where quay walls are: sampling the middle and building the
    corners from it put a wall 0.96 m underground.
    """
    if len(points) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    normals = offset_normals(points)
    half = max(style.thickness_m, 0.02) * 0.5
    left = points + normals * half
    right = points - normals * half

    ground_left = np.asarray(sampler(left[:, 0], left[:, 1]), dtype=np.float64)
    ground_right = np.asarray(sampler(right[:, 0], right[:, 1]), dtype=np.float64)
    # Each base on its own ground, so neither side floats and neither is
    # swallowed; the top level across the width, taken from the higher side, so
    # the wall does not lean and is never shorter than it should be. Sharing
    # one height for both would have to pick, and either choice is wrong on a
    # bank: the lower buries the high side, the higher floats the low one.
    low_left, _ = _span(ground_left, style)
    low_right, _ = _span(ground_right, style)
    _, high = _span(np.maximum(ground_left, ground_right), style)

    n = len(points)
    # Four rails: left-low, left-high, right-high, right-low. Ordered around
    # the cross-section, so consecutive rails bound one face of the prism and
    # the last wraps to the first.
    rails = [
        np.column_stack([left, low_left]),
        np.column_stack([left, high]),
        np.column_stack([right, high]),
        np.column_stack([right, low_right]),
    ]
    vertices = np.vstack(rails)

    triangles = []
    for r in range(4):
        a0 = r * n
        b0 = ((r + 1) % 4) * n
        for i in range(n - 1):
            triangles.append([a0 + i, b0 + i, b0 + i + 1])
            triangles.append([a0 + i, b0 + i + 1, a0 + i + 1])

    # Close both ends, or the prism is a tube and reads as hollow from any
    # position that can see into it.
    for i, sign in ((0, 1), (n - 1, -1)):
        quad = [0 * n + i, 1 * n + i, 2 * n + i, 3 * n + i]
        if sign > 0:
            triangles.append([quad[0], quad[1], quad[2]])
            triangles.append([quad[0], quad[2], quad[3]])
        else:
            triangles.append([quad[0], quad[2], quad[1]])
            triangles.append([quad[0], quad[3], quad[2]])

    return vertices, np.asarray(triangles, dtype=np.int64)


def _extrude_rings(
    rings: list[np.ndarray], sampler, style: BarrierStyle
):
    """A polygon into a prism, as ``(vertices, triangles)``.

    The outer ring and any holes all become vertical quads; the top is capped
    by earcut. See the module docstring for why a sliver in the cap is
    acceptable here and would not be on a road.

    The base is sampled per vertex, not once for the whole outline. One height
    for the whole thing is the obvious shortcut and it fails exactly where
    these objects live: a quay wall runs along a canal bank, which is the
    steepest ground in a Dutch city. On a bumpy test surface a single median
    buried one end of a wall by 0.65 m.

    A lifted piece -- an awning, a canopy -- is the exception. It hangs off a
    facade at one height, so it takes the median and stays flat; an awning
    that ripples with the pavement under it is worse than one that does not.
    """
    import mapbox_earcut

    rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in rings]
    rings = [r[:-1] if np.allclose(r[0], r[-1]) else r for r in rings]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    flat = np.vstack(rings)
    ground = np.asarray(sampler(flat[:, 0], flat[:, 1]), dtype=np.float64)
    if style.lift_m > 0.0:
        ground = np.full(len(flat), float(np.median(ground)))
    low, high = _span(ground, style)

    n = len(flat)
    vertices = np.vstack([
        np.column_stack([flat, low]),
        np.column_stack([flat, high]),
    ])

    triangles = []
    start = 0
    for ring in rings:
        count = len(ring)
        for i in range(count):
            a = start + i
            b = start + (i + 1) % count
            triangles.append([a, b, n + b])
            triangles.append([a, n + b, n + a])
        start += count

    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        cap = np.asarray(
            mapbox_earcut.triangulate_float64(flat, ends), dtype=np.int64
        ).reshape(-1, 3)
    except Exception as exc:  # noqa: BLE001 - one bad outline is not fatal
        LOG.debug("could not cap a barrier outline: %s", exc)
        cap = np.zeros((0, 3), dtype=np.int64)
    if len(cap):
        triangles.extend((cap + n).tolist())
        # An awning has no ground under it, so it needs a floor as well as a
        # lid. A wall's underside is buried and nobody will look.
        if style.lift_m > 0.0:
            triangles.extend(cap[:, ::-1].tolist())

    return vertices, np.asarray(triangles, dtype=np.int64)


def build_barriers(
    bbox: BBox,
    work_dir: Path,
    *,
    barriers_cfg: dict,
    terrain,
) -> BarrierSet:
    """Fetch every barrier collection and turn it into draped geometry."""
    result = BarrierSet()
    if not bool(barriers_cfg.get("enabled", True)):
        LOG.info("barriers disabled")
        return result

    fetch_kwargs = {
        "page_limit": int(barriers_cfg["page_limit"]),
        "timeout": float(barriers_cfg["timeout_s"]),
        "max_retries": int(barriers_cfg["max_retries"]),
        "max_pages": int(barriers_cfg["max_pages"]),
    }
    step_m = float(barriers_cfg.get("step_m", DEFAULT_STEP_M))
    drape_tolerance_m = float(barriers_cfg.get("drape_tolerance_m", 0.05))
    heights = barriers_cfg.get("heights") or {}

    def styled(table, collection, plus_type):
        style = table.get((collection, plus_type))
        if style is None:
            return None
        override = heights.get(style.name)
        if override is None:
            return style
        return BarrierStyle(
            style.name, float(override), style.thickness_m,
            style.material, style.lift_m,
        )

    chunks: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    counts: dict[str, int] = {}

    for collection in COLLECTIONS:
        try:
            features, _stats = fetch_current(
                collection, bbox, class_field="plus_type", **fetch_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - one empty collection is not fatal
            LOG.warning("could not read BGT %s: %s", collection, exc)
            continue

        for feature in features:
            plus_type = feature_type(feature.get("properties") or {})
            geometry = feature.get("geometry") or {}
            geometry_type = str(geometry.get("type") or "")

            if geometry_type in ("LineString", "MultiLineString"):
                style = styled(LINE_STYLES, collection, plus_type)
                if style is None:
                    continue
                raw = (
                    [geometry.get("coordinates") or []]
                    if geometry_type == "LineString"
                    else (geometry.get("coordinates") or [])
                )
                for line in raw:
                    line = np.asarray(line, dtype=np.float64)
                    if line.ndim != 2 or len(line) < 2:
                        continue
                    for piece in clip_line_to_bbox(line[:, :2], bbox):
                        if _length(piece) < MIN_LENGTH_M:
                            continue
                        dense, _profile = _drape_points(
                            piece, terrain.sample, step_m, drape_tolerance_m
                        )
                        verts, tris = _sweep_line(dense, terrain.sample, style)
                        if len(tris):
                            chunks.append((verts, tris, style.material, style.name))
                            counts[style.name] = counts.get(style.name, 0) + 1

            elif geometry_type in ("Polygon", "MultiPolygon"):
                style = styled(AREA_STYLES, collection, plus_type)
                if style is None:
                    continue
                for group in polygon_rings(geometry):
                    rings = _prepare_rings(group, bbox)
                    if not rings:
                        continue
                    verts, tris = _extrude_rings(rings, terrain.sample, style)
                    if len(tris):
                        chunks.append((verts, tris, style.material, style.name))
                        counts[style.name] = counts.get(style.name, 0) + 1

    if not chunks:
        LOG.info("no barriers found")
        return result

    materials = sorted({material for _, _, material, _ in chunks})
    kinds = sorted({kind for _, _, _, kind in chunks})
    material_index = {name: i for i, name in enumerate(materials)}
    kind_index = {name: i for i, name in enumerate(kinds)}

    all_vertices, all_triangles, material_of, kind_of = [], [], [], []
    offset = 0
    for verts, tris, material, kind in chunks:
        all_vertices.append(verts)
        all_triangles.append(tris + offset)
        material_of.append(np.full(len(tris), material_index[material], np.int32))
        kind_of.append(np.full(len(tris), kind_index[kind], np.int32))
        offset += len(verts)

    vertices = np.vstack(all_vertices)
    triangles = np.vstack(all_triangles)
    material_of = np.concatenate(material_of)
    kind_of = np.concatenate(kind_of)

    # A survey outline can double a vertex, which makes a triangle with no
    # area: invisible, but it breaks normals in Blender and shows up in any
    # mesh check as a fault. Cheaper to drop here than to explain there.
    a = vertices[triangles[:, 0]]
    edge1 = vertices[triangles[:, 1]] - a
    edge2 = vertices[triangles[:, 2]] - a
    real = np.linalg.norm(np.cross(edge1, edge2), axis=1) > 1e-9
    dropped = int((~real).sum())
    if dropped:
        LOG.debug("dropped %d zero-area barrier triangles", dropped)

    result.vertices = vertices
    result.triangles = triangles[real].astype(np.int32)
    result.material = material_of[real]
    result.kind = kind_of[real]
    result.material_names = materials
    result.kind_names = kinds
    result.counts = counts

    LOG.info(
        "barriers: %s (%d triangles, %d vertices)",
        ", ".join(f"{v} {k}" for k, v in sorted(counts.items())),
        len(result.triangles),
        len(result.vertices),
    )
    save_barriers(result, work_dir / "barriers.npz")
    return result


def _length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def save_barriers(barriers: BarrierSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices=barriers.vertices,
        triangles=barriers.triangles,
        material=barriers.material,
        kind=barriers.kind,
        material_names=np.array(barriers.material_names, dtype=object),
        kind_names=np.array(barriers.kind_names, dtype=object),
    )
    LOG.info("wrote %s (%d triangles)", path, len(barriers))
    return path


__all__ = [
    "AREA_STYLES",
    "COLLECTIONS",
    "LINE_STYLES",
    "BarrierSet",
    "BarrierStyle",
    "build_barriers",
    "feature_type",
    "save_barriers",
]
