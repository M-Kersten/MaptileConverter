"""The ground the way the modellers build it: a coarse terrain, flat layers on top.

The hand-made Hazeldonk model the modellers work from is built in layers,
each a separate object a person can grab:

* **Terrain.** One sheet, with as few faces as keep it close to the survey --
  2,115 faces for their square kilometre -- and the aerial on it.
* **Roads, the motorway, walkways and grass**, each a separate set of faces
  lying on the terrain a few centimetres up: grass at 2 cm, everything paved
  at 5 cm. Their own colour, no photo.
* **Kerbs** are not modelled at all: every walkway and traffic island is a flat
  outline carrying the modellers' curb generator, which raises the walkway and
  builds the kerb round it. Edit the outline and the kerb follows.

Here the same layers come from the BGT. The terrain is a triangulation of the
survey: start from the outlines of every layer, add the surveyed point that is
furthest from the surface until none is further than `tolerance_m`. The
outlines are edges of it, so each layer is simply the terrain's own faces
under it, lifted -- no face of a road can dip under the ground or float off it.

Water is left out, as the modellers left it out: the terrain and the aerial
carry the ditches. Bridges and tunnels are left to the structures stage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .terrain_mesh import (
    GRID_SIZE,
    TerrainMesh,
    TriangleLocator,
    _parts,
    _polygonal,
    _recombine,
    _regions,
)

LOG = logging.getLogger(__name__)

# What a stretch of ground is drawn as. Numbered clear of terrain_mesh's
# WATER (2) and BUILDING (3), so a face kind means the same thing to every
# check that reads one; ROAD is the same ground in both: under a road.
TERRAIN, ROAD, MOTORWAY, WALKWAY, GRASS = 0, 1, 4, 5, 6
LAYER_NAMES = {
    TERRAIN: "terrain", ROAD: "road", MOTORWAY: "motorway",
    WALKWAY: "walkway", GRASS: "grass",
}
# The word each layer's objects are named with, as in the hand-made file.
LAYER_WORDS = {ROAD: "Wegen", MOTORWAY: "Snelweg", WALKWAY: "Voetpaden", GRASS: "Gras"}

# Flat colours, named and coloured as in the hand-made file where it has one:
# linear RGB and roughness. Asfalt, Bakstenen, Beton, Beton_Geel and Gras are
# theirs; Fietspad and Verharding are the two surfaces their square did not
# need a colour for.
MATERIALS: dict[str, tuple[tuple[float, float, float], float]] = {
    "Asfalt": ((0.0496, 0.0496, 0.0496), 0.5),
    "Bakstenen": ((0.1414, 0.0969, 0.0822), 0.5),
    "Fietspad": ((0.2, 0.045, 0.035), 0.6),
    "Verharding": ((0.25, 0.25, 0.24), 0.7),
    "Beton_Geel": ((0.6146, 0.4983, 0.1433), 0.5),
    "Beton": ((0.8, 0.8, 0.8), 0.5),
    "Gras": ((0.0697, 0.1031, 0.0007), 1.0),
}
MATERIAL_NAMES = tuple(MATERIALS)

# Which outline wins where two overlap, best first. The BGT is a partition,
# so this only settles the slivers where two surveys disagree.
PRIORITY = (
    (MOTORWAY, "Asfalt"),
    (ROAD, "Asfalt"), (ROAD, "Bakstenen"), (ROAD, "Fietspad"),
    (WALKWAY, "Beton_Geel"),
    (ROAD, "Verharding"),
    (GRASS, "Gras"),
)

# A green verge or island counts as an island -- raised, kerbed all round --
# when paving surrounds at least this much of it and it is no bigger than
# `ISLAND_MAX_M2`. A field between two roads is not an island.
ISLAND_SHARE = 0.9
ISLAND_MAX_M2 = 2500.0

# How far each layer's top is above the terrain, for what stands on it: a
# lamp post on a footpath stands on the walkway the curb generator raised.
DEFAULT_OFFSETS = {ROAD: 0.05, MOTORWAY: 0.05, WALKWAY: 0.05, GRASS: 0.02}


def _code(layer: int, material: str) -> int:
    return layer * 16 + MATERIAL_NAMES.index(material)


def _split_code(code):
    code = np.asarray(code, dtype=np.int64)
    return code // 16, code % 16


# ---------------------------------------------------------------------------
# What each BGT polygon is
# ---------------------------------------------------------------------------


@dataclass
class GroundArea:
    """One BGT polygon as the layered ground reads it."""

    rings: list[np.ndarray]  # RD; the first is the outline, the rest holes
    layer: int
    material: str
    level: int = 0  # BGT relatieve_hoogteligging


def classify(collection: str, properties: dict) -> tuple[int, str] | None:
    """``(layer, material)`` for one BGT polygon; None leaves it to the terrain.

    The motorway is apart because the modellers keep it apart. Footpaths and
    paved traffic islands are walkways: the curb generator raises them. Green
    traffic islands start as grass and become islands when paving surrounds
    them (`ISLAND_SHARE`). Paving that is no road -- a forecourt, a hard verge
    -- is laid flat in grey, without kerbs. Yards, sand and water stay terrain.
    """
    from .surfaces import CLASS_CYCLE, CLASS_FOOTPATH, CLASS_ROAD_BRICK, road_class

    functie = str(properties.get("functie") or "").strip().lower()
    physical = str(properties.get("fysiek_voorkomen") or "").strip().lower()
    plus = str(properties.get("plus_fysiek_voorkomen") or "").strip().lower()
    paved = "verhard" in physical and "onverhard" not in physical
    green = "groen" in physical or "gras" in plus or "begroeid" in physical

    if collection == "wegdeel":
        if "rijbaan" in functie and ("autosnelweg" in functie or "autoweg" in functie):
            return MOTORWAY, "Asfalt"
        code = road_class(functie, plus or physical)
        if code == CLASS_FOOTPATH:
            return WALKWAY, "Beton_Geel"
        if code == CLASS_CYCLE:
            return ROAD, "Fietspad"
        if code == CLASS_ROAD_BRICK:
            return ROAD, "Bakstenen"
        if "onverhard" in physical:
            return None  # a track across a field is the field
        return ROAD, "Asfalt"
    if collection == "ondersteunendwegdeel":
        if "verkeerseiland" in functie:
            return (WALKWAY, "Beton_Geel") if paved else (GRASS, "Gras")
        if green:
            return GRASS, "Gras"
        if paved:
            return ROAD, "Verharding"
        return None
    if collection == "begroeidterreindeel":
        return GRASS, "Gras"
    if collection == "onbegroeidterreindeel":
        return (ROAD, "Verharding") if paved else None
    return None


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass
class LayeredGround:
    """The terrain and the layers lying on it, ready for Blender."""

    mesh: TerrainMesh  # the terrain; face_kind is the layer on each face
    face_material: np.ndarray  # (F,) index into MATERIAL_NAMES
    face_object: np.ndarray  # (F,) index into object_names; -1 terrain only
    object_names: list[str]
    object_layer: list[int]
    terrain_name: str
    # How far each layer's object sits above the terrain, and how far above
    # the terrain its top is once Blender has run (a walkway is raised).
    lift: dict[int, float]
    top: dict[int, float]
    origin: tuple[float, float] = (0.0, 0.0)  # local frame origin, RD
    stats: dict = field(default_factory=dict)

    def face_top(self) -> np.ndarray:
        layers = np.asarray(self.mesh.face_kind, dtype=np.int64)
        table = np.zeros(max(LAYER_NAMES) + 1)
        for layer, value in self.top.items():
            table[layer] = value
        return table[layers]

    def surface_at(self, x, y) -> np.ndarray:
        """The top of what is drawn at RD coordinates: terrain plus the layer."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(x, y).shape
        x = np.broadcast_to(x, shape).ravel()
        y = np.broadcast_to(y, shape).ravel()
        locator = self.mesh.locator()
        tri, weights = locator.locate(x, y)
        _, owner = self.mesh.triangles()
        z = self.mesh.vertices[:, 2]
        out = np.full(len(x), np.nan)
        found = tri >= 0
        corners = locator.tris[tri[found]]
        out[found] = (weights[found] * z[corners]).sum(axis=1)
        out[found] += self.face_top()[owner[tri[found]]]
        return out.reshape(shape)

    def to_npz_fields(self) -> dict:
        fields = self.mesh.to_npz_fields()
        rgb = np.array([MATERIALS[name][0] for name in MATERIAL_NAMES], dtype=np.float64)
        rough = np.array([MATERIALS[name][1] for name in MATERIAL_NAMES], dtype=np.float64)
        fields.update(
            face_layer=np.asarray(self.mesh.face_kind, dtype=np.int8),
            face_material=self.face_material.astype(np.int16),
            face_object=self.face_object.astype(np.int32),
            # Plain unicode arrays: Blender reads these without pickle.
            object_names=np.array(self.object_names or [""], dtype=str),
            object_layer=np.array(self.object_layer or [0], dtype=np.int8),
            object_lift=np.array(
                [self.lift.get(layer, 0.0) for layer in (self.object_layer or [0])],
                dtype=np.float64,
            ),
            material_names=np.array(MATERIAL_NAMES, dtype=str),
            material_rgb=rgb,
            material_roughness=rough,
            terrain_name=np.array(self.terrain_name, dtype=str),
        )
        return fields

    def save(self, work_dir: Path) -> Path:
        path = Path(work_dir) / "layered_ground.npz"
        np.savez_compressed(path, **self.to_npz_fields())
        return path


# ---------------------------------------------------------------------------
# Building it
# ---------------------------------------------------------------------------


def build_layered_ground(
    terrain,
    areas: list[GroundArea],
    *,
    prefix: str,
    tolerance_m: float = 0.5,
    outline_m: float = 0.25,
    focus_m: float = 250.0,
    lift: dict[int, float] | None = None,
    walkway_raise_m: float = 0.13,
    max_rounds: int = 60,
) -> LayeredGround:
    """The coarse terrain and its layers, from the survey and the BGT areas.

    `tolerance_m` is how far any surveyed grid node may be from the terrain;
    `outline_m` how far a simplified outline may stray from the BGT's. Faces
    in the middle `focus_m` of the area are grouped east and west of the
    centre, the rest by quadrant, as the modellers group theirs.
    """
    import shapely

    started = time.time()
    lift = {**DEFAULT_OFFSETS, **(lift or {})}
    xs = np.asarray(terrain.xs, dtype=np.float64)
    ys = np.asarray(terrain.ys, dtype=np.float64)
    ox, oy = float(xs[0]), float(ys[0])
    width, height = float(xs[-1] - ox), float(ys[-1] - oy)
    frame = shapely.box(0.0, 0.0, width, height)

    # ---- regions: the BGT cut into one layer and material each -----------
    by_code: dict[int, list] = {}
    skipped_levels = 0
    for area in areas:
        if area.level != 0:
            skipped_levels += 1
            continue
        rings = [np.asarray(r, dtype=np.float64)[:, :2] - [ox, oy] for r in area.rings]
        rings = [r for r in rings if len(r) >= 4]
        if not rings:
            continue
        polygon = _polygonal(shapely.Polygon(rings[0], rings[1:]))
        if not polygon.is_empty:
            by_code.setdefault(_code(area.layer, area.material), []).append(polygon)
    order = [_code(layer, material) for layer, material in PRIORITY]
    layers = [
        (code, _polygonal(shapely.union_all(np.array(by_code[code], dtype=object))))
        for code in order
        if code in by_code
    ]
    pieces = _regions(frame, layers, station_m=1e9) if layers else [(frame, 0)]
    pieces = _simplify(pieces, outline_m, frame)
    pieces = _islands(pieces)
    pieces, hairpins = _without_hairpins(pieces)
    timings = {"regions_s": round(time.time() - started, 1)}

    # ---- the terrain: a triangulation refined until it fits the survey -----
    t = time.time()
    gx, gy = np.meshgrid(xs - ox, ys - oy)
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    grid_z = np.asarray(terrain.measured if terrain.measured is not None else terrain.heights,
                        dtype=np.float64).ravel()
    points, segments = _constraints(pieces)

    def heights_at(xy):
        return np.asarray(terrain.sample(xy[:, 0] + ox, xy[:, 1] + oy), dtype=np.float64)

    points, z, tris, missing, rounds = _fit(
        points, segments, heights_at, grid_xy, grid_z,
        tolerance_m=float(tolerance_m), bounds=(0.0, 0.0, width, height),
        max_rounds=max_rounds,
    )
    timings["terrain_s"] = round(time.time() - t, 1)

    # ---- faces: each triangle takes the region it lies in -----------------
    t = time.time()
    tri_code = _classify_triangles(points, tris, pieces)
    faces, codes = _faces(points, tris, tri_code)
    timings["faces_s"] = round(time.time() - t, 1)

    sizes = np.array([len(f) for f in faces], dtype=np.int64)
    loops = np.concatenate(faces).astype(np.int64) if faces else np.zeros(0, np.int64)
    face_layer, face_material = _split_code(codes)
    face_material = np.where(face_layer == TERRAIN, -1, face_material)

    vertices = np.column_stack([points[:, 0] + ox, points[:, 1] + oy, z])
    edges, edge_on, vertex_on = _layer_outlines(loops, sizes, face_layer, len(points))
    mesh = TerrainMesh(
        vertices=vertices,
        measured_z=z.copy(),
        face_sizes=sizes.astype(np.int32),
        face_loops=loops,
        face_kind=face_layer.astype(np.int8),
        face_part=np.zeros(len(sizes), dtype=np.int8),
        vertex_on=vertex_on,
        edges=edges,
        edge_on=edge_on,
        grid_m=float(terrain.grid_m),
        locator_cell=8.0,
    )

    # ---- objects, named and grouped as the modellers do --------------------
    face_object, names, object_layer = _objects(
        points, loops, sizes, face_layer, prefix=prefix,
        centre=(0.5 * width, 0.5 * height), focus_m=float(focus_m),
    )
    top = dict(lift)
    top[WALKWAY] = lift[WALKWAY] + float(walkway_raise_m)

    # ---- how well it fits -------------------------------------------------
    locator = TriangleLocator(points, tris, cell=8.0, bounds=(0.0, 0.0, width, height))
    drawn = locator.interpolate(z, grid_xy[:, 0], grid_xy[:, 1])
    error = np.abs(drawn - grid_z)
    error = error[np.isfinite(error)]
    quads = int((sizes == 4).sum())
    stats = {
        "faces": int(len(sizes)),
        "vertices": int(len(points)),
        "quad_share": round(quads / max(len(sizes), 1), 3),
        "tolerance_m": float(tolerance_m),
        "grid_error_max_m": round(float(error.max()), 3) if len(error) else 0.0,
        "grid_error_p95_m": round(float(np.percentile(error, 95)), 3) if len(error) else 0.0,
        "refinement_rounds": int(rounds),
        "outline_edges_missing": int(missing),
        "walkway_hairpins_removed": int(hairpins),
        "faces_by_layer": {
            LAYER_NAMES[k]: int((face_layer == k).sum()) for k in LAYER_NAMES
        },
        "objects": {name: int((face_object == i).sum()) for i, name in enumerate(names)},
        "parts_on_bridges_or_tunnels": int(skipped_levels),
        **timings,
        "seconds": round(time.time() - started, 1),
    }
    mesh.stats = dict(stats)
    LOG.info(
        "layered ground: terrain %d faces (%d%% quads), %d vertices, every grid node "
        "within %.2f m (95%% within %.2f m); %d objects; %s",
        len(sizes), round(100 * stats["quad_share"]), len(points),
        stats["grid_error_max_m"], stats["grid_error_p95_m"], len(names),
        ", ".join(f"{v} {k}" for k, v in stats["faces_by_layer"].items() if v),
    )
    if missing:
        LOG.warning(
            "layered ground: %d outline edges could not be kept as terrain edges; "
            "the faces there were given to whichever layer their middle lies in",
            missing,
        )
    return LayeredGround(
        mesh=mesh,
        face_material=face_material.astype(np.int64),
        face_object=face_object,
        object_names=names,
        object_layer=object_layer,
        terrain_name=f"{prefix}_Terrein",
        lift={k: float(v) for k, v in lift.items()},
        top={k: float(v) for k, v in top.items()},
        origin=(ox, oy),
        stats=stats,
    )


def _simplify(pieces, tolerance_m: float, frame):
    """The outlines simplified together, so a shared edge stays shared."""
    import shapely

    if tolerance_m <= 0 or len(pieces) < 2:
        return pieces
    geoms = np.array([p for p, _ in pieces], dtype=object)
    codes = [k for _, k in pieces]
    try:
        simple = shapely.coverage_simplify(geoms, tolerance_m, simplify_boundary=False)
    except Exception as exc:  # noqa: BLE001 - simplification is an economy
        LOG.debug("layered ground: outlines not simplified: %s", exc)
        return pieces
    simple = shapely.set_precision(simple, GRID_SIZE)
    simple = shapely.orient_polygons(simple)
    out = []
    for polygon, code in zip(simple, codes):
        for part in _parts(polygon):
            if part.area > 1e-3:
                out.append((part, code))
    # Simplifying a valid coverage keeps it one; if it ever did not, the
    # unsimplified pieces are the safe answer.
    covered = float(sum(p.area for p, _ in out))
    if abs(covered - frame.area) > 1e-3 * frame.area:
        LOG.debug("layered ground: simplified outlines lost area; kept the originals")
        return pieces
    return out


def _islands(pieces):
    """Green pieces paving surrounds become islands: raised and kerbed."""
    import shapely

    layers = np.array([k // 16 for _, k in pieces])
    paved = np.flatnonzero(np.isin(layers, (ROAD, MOTORWAY, WALKWAY)))
    if not len(paved):
        return pieces
    paved_geoms = np.array([pieces[i][0] for i in paved], dtype=object)
    tree = shapely.STRtree(paved_geoms)
    island = _code(WALKWAY, "Gras")
    out = list(pieces)
    for i, (polygon, code) in enumerate(pieces):
        if code // 16 != GRASS or polygon.area > ISLAND_MAX_M2:
            continue
        ring = polygon.exterior
        near = tree.query(polygon.buffer(0.05), predicate="intersects")
        if not len(near):
            continue
        around = shapely.union_all(paved_geoms[near]).buffer(0.05)
        if ring.intersection(around).length >= ISLAND_SHARE * ring.length:
            out[i] = (polygon, island)
    return out


HAIRPIN_DEG = 20.0


# ... and how much area taking one corner away may move between pieces: a
# needle or a slit is trimmed, a footpath that tapers over twenty metres is not.
HAIRPIN_MAX_M2 = 2.0
# Below this a corner is trimmed whatever it moves: measured on the curb
# generator, a 2 degree tip throws its kerb 3 m, a 3 degree notch 10 m.
HAIRPIN_ALWAYS_DEG = 6.0
# A walkway narrower than twice this -- the kerb's width -- has no room for a
# kerb on each side; one no larger than SLIVER_MAX_M2 with a corner under
# HAIRPIN_ALWAYS_DEG is a sliver too.
SLIVER_M = 0.3
SLIVER_MAX_M2 = 50.0


def _hairpins(polygon, limit_deg: float = HAIRPIN_DEG, max_m2: float | None = None) -> set:
    """Grid keys of the corners where an outline turns back on itself.

    With `max_m2`, only those whose triangle with the two corners either side
    -- what taking the corner away moves -- is no larger than that, unless
    the corner is sharper than `HAIRPIN_ALWAYS_DEG`.
    """
    keys = set()
    cos_limit = np.cos(np.radians(limit_deg))
    for ring in [polygon.exterior, *polygon.interiors]:
        c = np.asarray(ring.coords)[:-1, :2]
        if len(c) < 3:
            continue
        back = np.roll(c, 1, axis=0) - c
        ahead = np.roll(c, -1, axis=0) - c
        lengths = np.hypot(*back.T) * np.hypot(*ahead.T)
        cosine = np.where(
            lengths > 0, (back * ahead).sum(axis=1) / np.maximum(lengths, 1e-18), -1.0
        )
        sharp = cosine > cos_limit
        if max_m2 is not None:
            area = 0.5 * np.abs(back[:, 0] * ahead[:, 1] - back[:, 1] * ahead[:, 0])
            sharp &= (area <= max_m2) | (cosine > np.cos(np.radians(HAIRPIN_ALWAYS_DEG)))
        for point in c[sharp]:
            keys.add(tuple(np.rint(point / GRID_SIZE).astype(np.int64)))
    return keys


def _without_hairpins(pieces, limit_deg: float = HAIRPIN_DEG):
    """Walkway corners sharper than `limit_deg` taken out of every piece.

    The curb generator insets its kerb by mitring each corner, and at a
    corner of a few degrees the mitre runs away: at Hazeldonk a kerb vertex
    landed 825 m off a 2.7 degree needle tip in a footpath outline. The tip
    goes from the walkway and from whatever shares it, so the pieces still
    tile the area; the sliver changes hands. Returns the pieces and how many
    corners went.
    """
    import shapely

    def without(polygon, drop):
        rings = []
        for ring in [polygon.exterior, *polygon.interiors]:
            c = np.asarray(ring.coords)[:-1, :2]
            keys = [tuple(k) for k in np.rint(c / GRID_SIZE).astype(np.int64)]
            kept = c[[k not in drop for k in keys]]
            rings.append(kept)
        if len(rings[0]) < 3:
            return None
        made = shapely.Polygon(rings[0], [r for r in rings[1:] if len(r) >= 3])
        return made if made.is_valid and made.area > 0 else None

    removed = 0
    # Taking a needle's tip leaves the corner behind it as the new tip, so
    # this goes round until none is left.
    for _ in range(50):
        drop = set()
        for polygon, code in pieces:
            if code // 16 == WALKWAY:
                drop |= _hairpins(polygon, limit_deg, HAIRPIN_MAX_M2)
        if not drop:
            break
        # A corner whose removal would break any piece stays.
        for _ in range(len(drop) + 1):
            out, broken = [], set()
            for polygon, code in pieces:
                corners = {
                    tuple(k) for ring in [polygon.exterior, *polygon.interiors]
                    for k in np.rint(np.asarray(ring.coords)[:, :2] / GRID_SIZE).astype(np.int64)
                }
                if not corners & drop:
                    out.append((polygon, code))
                    continue
                made = without(polygon, drop)
                if made is None:
                    broken |= corners & drop
                out.append((made, code))
            if not broken:
                break
            drop -= broken
        if not drop or broken:
            break
        pieces = out
        removed += len(drop)

    # A walkway still with one, and too thin for a kerb either side -- or
    # small and with a corner sharp enough to throw its kerb -- is a sliver
    # that cannot lose its tip without vanishing: it joins the neighbour it
    # shares most outline with.
    walkways = [
        i for i, (p, k) in enumerate(pieces)
        if k // 16 == WALKWAY and _hairpins(p, limit_deg) and (
            shapely.maximum_inscribed_circle(p).length < SLIVER_M
            or (p.area <= SLIVER_MAX_M2 and _hairpins(p, HAIRPIN_ALWAYS_DEG))
        )
    ]
    if walkways:
        others = [i for i, (_, k) in enumerate(pieces) if k // 16 != WALKWAY]
        tree = shapely.STRtree(np.array([pieces[i][0] for i in others], dtype=object))
        pieces = list(pieces)
        for i in walkways:
            polygon = pieces[i][0]
            near = tree.query(polygon, predicate="intersects")
            shared = [
                (polygon.boundary.intersection(pieces[others[j]][0].boundary).length, others[j])
                for j in near
            ]
            if shared:
                _, best = max(shared)
                pieces[i] = (polygon, pieces[best][1])
                removed += 1
    return pieces, removed


def _constraints(pieces):
    """Every outline vertex once, and every outline edge once, as index pairs."""
    coords = []
    segments = []
    offset = 0
    for polygon, _ in pieces:
        for ring in [polygon.exterior, *polygon.interiors]:
            c = np.asarray(ring.coords)[:-1, :2]
            if len(c) < 3:
                continue
            idx = np.arange(len(c)) + offset
            coords.append(c)
            segments.append(np.column_stack([idx, np.roll(idx, -1)]))
            offset += len(c)
    if not coords:
        return np.zeros((0, 2)), np.zeros((0, 2), dtype=np.int64)
    coords = np.vstack(coords)
    keys = np.rint(coords / GRID_SIZE).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    points = unique.astype(np.float64) * GRID_SIZE
    segs = inverse[np.vstack(segments)]
    segs = np.sort(segs, axis=1)
    segs = segs[segs[:, 0] != segs[:, 1]]
    segs = np.unique(segs, axis=0)
    return points, segs


def _delaunay(points: np.ndarray) -> np.ndarray:
    """Delaunay triangles over `points`, as ``(T, 3)`` counter-clockwise ids."""
    import shapely

    keys = np.rint(points / GRID_SIZE).astype(np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    triangles = shapely.delaunay_triangles(shapely.multipoints(points))
    parts = shapely.get_parts(triangles)
    if not len(parts):
        return np.zeros((0, 3), dtype=np.int64)
    coords = shapely.get_coordinates(parts).reshape(len(parts), 4, 2)[:, :3]
    flat = np.rint(coords.reshape(-1, 2) / GRID_SIZE).astype(np.int64)
    # Each corner back to the point it came from.
    at = np.searchsorted(sorted_keys[:, 0] * (1 << 32) + sorted_keys[:, 1],
                         flat[:, 0] * (1 << 32) + flat[:, 1])
    at = np.clip(at, 0, len(order) - 1)
    ids = order[at].reshape(-1, 3)
    cross = _cross(points, ids)
    ids = np.where((cross < 0)[:, None], ids[:, ::-1], ids)
    return _without_flat(points, ids)


def _cross(points, tris):
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])


def _without_flat(points, tris, eps: float = 1e-9):
    """Each zero-area triangle replaced by splitting its neighbour.

    The triangulator returns one wherever three outline points lie on a line
    -- an edge and the middle added to keep it. Dropping it leaves that middle
    on one side of the outline and not the other: a crack. The triangle
    across its long edge is split at the middle instead.
    """
    for _ in range(100):
        flat = np.flatnonzero(np.abs(_cross(points, tris)) <= eps)
        if not len(flat):
            return tris
        edges: dict[tuple[int, int], list[int]] = {}
        for t, (p, q, r) in enumerate(tris.tolist()):
            for u, w in ((p, q), (q, r), (r, p)):
                edges.setdefault((min(u, w), max(u, w)), []).append(t)
        gone: set[int] = set()
        added = []
        for f in flat.tolist():
            if f in gone:
                continue
            t = tris[f]
            corners = points[t]
            lengths = [
                float(np.hypot(*(corners[(i + 1) % 3] - corners[(i + 2) % 3])))
                for i in range(3)
            ]
            i = int(np.argmax(lengths))
            middle, a, b = int(t[i]), int(t[(i + 1) % 3]), int(t[(i + 2) % 3])
            gone.add(f)
            across = [n for n in edges.get((min(a, b), max(a, b)), []) if n != f and n not in gone]
            if not across:
                continue  # on the edge of the area: nothing on the other side
            n = across[0]
            gone.add(n)
            ring = [int(v) for v in tris[n]]
            j = next(k for k in range(3) if {ring[k], ring[(k + 1) % 3]} == {a, b})
            u, w, c = ring[j], ring[(j + 1) % 3], ring[(j + 2) % 3]
            added += [(u, middle, c), (middle, w, c)]
        keep = np.array([t not in gone for t in range(len(tris))], dtype=bool)
        tris = np.vstack([tris[keep], np.array(added, dtype=np.int64).reshape(-1, 3)])
    return tris[np.abs(_cross(points, tris)) > eps]


def _edge_keys(tris: np.ndarray, n: int) -> np.ndarray:
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e[:, 0] * n + e[:, 1])


def _conform(points, z, segments, heights_at, max_splits: int = 30):
    """Triangulate, splitting every outline edge the triangulation missed.

    A Delaunay triangulation of the outline points takes most outline edges
    as its own; where it cuts across one, adding the edge's middle makes it
    take the halves. Returns the triangles and how many edges were still
    missing when it gave up.
    """
    for _ in range(max_splits):
        tris = _delaunay(points)
        n = len(points)
        have = _edge_keys(tris, n)
        keys = segments[:, 0] * n + segments[:, 1]
        missing = ~np.isin(keys, have)
        if not missing.any():
            return points, z, segments, tris, 0
        bad = segments[missing]
        middle = np.rint(0.5 * (points[bad[:, 0]] + points[bad[:, 1]]) / GRID_SIZE) * GRID_SIZE
        # An edge too short to split is as good as it gets.
        new_keys = np.rint(middle / GRID_SIZE).astype(np.int64)
        old_keys = np.rint(points / GRID_SIZE).astype(np.int64)
        fresh = ~_rows_in(new_keys, old_keys)
        if not fresh.any():
            return points, z, segments, tris, int(missing.sum())
        bad, middle = bad[fresh], middle[fresh]
        middle_z = heights_at(middle)
        ids = np.arange(len(points), len(points) + len(middle))
        points = np.vstack([points, middle])
        z = np.concatenate([z, middle_z])
        keep = segments[~missing]
        unsplit = segments[missing][~fresh]
        halves = np.vstack([np.column_stack([bad[:, 0], ids]), np.column_stack([ids, bad[:, 1]])])
        segments = np.unique(np.sort(np.vstack([keep, unsplit, halves]), axis=1), axis=0)
    tris = _delaunay(points)
    n = len(points)
    missing = ~np.isin(segments[:, 0] * n + segments[:, 1], _edge_keys(tris, n))
    return points, z, segments, tris, int(missing.sum())


def _rows_in(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    a = rows[:, 0] * (1 << 32) + rows[:, 1]
    b = np.sort(table[:, 0] * (1 << 32) + table[:, 1])
    at = np.clip(np.searchsorted(b, a), 0, max(len(b) - 1, 0))
    return (b[at] == a) if len(b) else np.zeros(len(a), dtype=bool)


def _fit(points, segments, heights_at, grid_xy, grid_z, *, tolerance_m, bounds, max_rounds):
    """Add the worst surveyed point of every triangle off by more than the
    tolerance, until none is; the outlines kept as edges throughout."""
    import shapely

    z = heights_at(points) if len(points) else np.zeros(0)
    finite = np.isfinite(grid_z)
    grid_xy, grid_z = grid_xy[finite], grid_z[finite]
    grid_keys = np.rint(grid_xy / GRID_SIZE).astype(np.int64)
    rounds = 0
    missing = 0
    tris = np.zeros((0, 3), dtype=np.int64)
    for rounds in range(1, max_rounds + 1):
        points, z, segments, tris, missing = _conform(points, z, segments, heights_at)
        locator = TriangleLocator(points, tris, cell=8.0, bounds=bounds)
        tri, weights = locator.locate(grid_xy[:, 0], grid_xy[:, 1])
        found = tri >= 0
        drawn = np.full(len(grid_z), np.nan)
        drawn[found] = (weights[found] * z[tris[tri[found]]]).sum(axis=1)
        error = np.where(found, np.abs(drawn - grid_z), 0.0)
        over = np.flatnonzero(error > tolerance_m)
        if not len(over):
            break
        # The worst node of each triangle over the tolerance.
        order = over[np.lexsort((-error[over], tri[over]))]
        first = np.ones(len(order), dtype=bool)
        first[1:] = tri[order][1:] != tri[order][:-1]
        pick = order[first]
        pick = pick[~_rows_in(grid_keys[pick], np.rint(points / GRID_SIZE).astype(np.int64))]
        if not len(pick):
            break
        new_xy = grid_xy[pick]
        new_z = grid_z[pick]
        # Nothing within a few centimetres of a vertex already there: it would
        # only make a sliver. The triangle is refined elsewhere next round.
        near_vertex, _ = shapely.STRtree(shapely.points(points)).query(
            shapely.points(new_xy), predicate="dwithin", distance=0.05
        )
        keep_new = np.ones(len(new_xy), dtype=bool)
        keep_new[near_vertex] = False
        if not keep_new.any():
            break
        new_xy, new_z = new_xy[keep_new], new_z[keep_new]
        keep_new = np.ones(len(new_xy), dtype=bool)
        if len(segments):
            # A node on an outline splits that outline: a vertex beside an
            # edge it nearly touches makes a sliver no face can be made of.
            lines = shapely.linestrings(
                np.stack([points[segments[:, 0]], points[segments[:, 1]]], axis=1)
            )
            tree = shapely.STRtree(lines)
            hit_point, hit_line = tree.query(
                shapely.points(new_xy), predicate="dwithin", distance=0.02
            )
            if len(hit_point):
                keep_seg = np.ones(len(segments), dtype=bool)
                taken_points: set[int] = set()
                halves = []
                for p, line in zip(hit_point.tolist(), hit_line.tolist()):
                    keep_new[p] = False
                    if not keep_seg[line] or p in taken_points:
                        continue  # one split per edge per round; the rest wait
                    taken_points.add(p)
                    keep_seg[line] = False
                    halves.append((p, int(segments[line, 0]), int(segments[line, 1])))
                split_xy = np.array([new_xy[p] for p, _, _ in halves]).reshape(-1, 2)
                split_z = np.array([new_z[p] for p, _, _ in halves])
                ids = len(points) + np.arange(len(halves))
                pairs = [(a, i) for (_, a, _), i in zip(halves, ids)]
                pairs += [(i, b) for (_, _, b), i in zip(halves, ids)]
                segments = segments[keep_seg]
                if pairs:
                    segments = np.unique(
                        np.sort(np.vstack([segments, np.array(pairs, dtype=np.int64)]), axis=1),
                        axis=0,
                    )
                points = np.vstack([points, split_xy])
                z = np.concatenate([z, split_z])
        points = np.vstack([points, new_xy[keep_new]])
        z = np.concatenate([z, new_z[keep_new]])
    else:
        points, z, segments, tris, missing = _conform(points, z, segments, heights_at)
    return points, z, tris, missing, rounds


def _classify_triangles(points, tris, pieces) -> np.ndarray:
    """The region code each triangle's middle lies in; 0 where none."""
    import shapely

    if not len(tris):
        return np.zeros(0, dtype=np.int64)
    centres = shapely.points(points[tris].mean(axis=1))
    geoms = np.array([p for p, _ in pieces], dtype=object)
    codes = np.array([k for _, k in pieces], dtype=np.int64)
    tree = shapely.STRtree(geoms)
    which, region = tree.query(centres, predicate="intersects")
    out = np.zeros(len(tris), dtype=np.int64)
    first = np.unique(which, return_index=True)[1]
    out[which[first]] = codes[region[first]]
    return out


def _faces(points, tris, tri_code):
    """Triangles paired into quads within each region, as vertex-id loops.

    Each quad starts at an end of the edge its two triangles shared, so its
    first-to-third diagonal is that edge: split as Blender draws a quad and
    as the export cuts it, a quad is the two triangles the fit measured.
    """
    keys = np.rint(points / GRID_SIZE).astype(np.int64)
    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(keys)}
    faces, codes = [], []
    for code in np.unique(tri_code):
        members = tris[tri_code == code]
        quads, singles = _recombine(points[members])
        # _recombine gives (c, a, d, b) for triangles (a, b, c) and (b, a, d).
        quads = [np.roll(q, -1, axis=0) for q in quads]
        for corners in quads + list(singles):
            k = np.rint(np.asarray(corners) / GRID_SIZE).astype(np.int64)
            faces.append(np.array([lookup[(int(a), int(b))] for a, b in k], dtype=np.int64))
            codes.append(int(code))
    return faces, np.array(codes, dtype=np.int64)


def _layer_outlines(loops, sizes, face_layer, n_vertices):
    """Edges where the layer changes, tagged with the layers on either side."""
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    following = np.arange(len(loops)) + 1
    following[starts + sizes - 1] = starts
    face_of = np.repeat(np.arange(len(sizes)), sizes)
    pairs = np.sort(np.column_stack([loops, loops[following]]), axis=1)
    key = pairs[:, 0] * n_vertices + pairs[:, 1]
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    same_next = np.zeros(len(key), dtype=bool)
    same_next[:-1] = key_sorted[1:] == key_sorted[:-1]
    a = order[:-1][same_next[:-1]]
    b = order[1:][same_next[:-1]]
    la = face_layer[face_of[a]]
    lb = face_layer[face_of[b]]
    differ = la != lb
    edges = pairs[a[differ]]
    # A bit per layer on either side: 1 << layer.
    edge_on = ((1 << la[differ]) | (1 << lb[differ])).astype(np.int32)
    vertex_on = np.zeros(n_vertices, dtype=np.int32)
    np.bitwise_or.at(vertex_on, edges[:, 0], edge_on)
    np.bitwise_or.at(vertex_on, edges[:, 1], edge_on)
    return edges.astype(np.int64), edge_on, vertex_on


def zone_of(dx: float, dy: float, focus_m: float) -> tuple[bool, str]:
    """Which part of the area a point is in, relative to the centre.

    Inside the middle `focus_m` it is the centre's east or west half; outside
    it, a quadrant -- the modellers' VP_Oost and Wegen_Afstand_NO.
    """
    if abs(dx) <= focus_m and abs(dy) <= focus_m:
        return True, "Centrum_" + ("Oost" if dx >= 0 else "West")
    return False, "Afstand_" + ("N" if dy >= 0 else "Z") + ("O" if dx >= 0 else "W")


def object_name(prefix: str, layer: int, zone: tuple[bool, str] | None) -> str:
    word = LAYER_WORDS[layer]
    if layer == MOTORWAY or zone is None:
        return f"{prefix}_{word}"
    central, name = zone
    return f"{prefix}_{name}_{word}" if central else f"{prefix}_{word}_{name}"


def _objects(points, loops, sizes, face_layer, *, prefix, centre, focus_m):
    """Which object each face belongs to.

    Roads and grass by where each face is; a walkway by where its whole
    island is, because the curb generator kerbs every edge where an object
    ends -- splitting one footpath across two objects would kerb it in half.
    """
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    face_of = np.repeat(np.arange(len(sizes)), sizes)
    xy = points[loops]
    centroid = np.zeros((len(sizes), 2))
    np.add.at(centroid, face_of, xy)
    centroid /= np.maximum(sizes, 1)[:, None]
    dx, dy = centroid[:, 0] - centre[0], centroid[:, 1] - centre[1]

    # Walkway islands: faces joined across shared edges.
    walk = np.flatnonzero(face_layer == WALKWAY)
    island = np.full(len(sizes), -1, dtype=np.int64)
    if len(walk):
        parent = {int(f): int(f) for f in walk}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        following = np.arange(len(loops)) + 1
        following[starts + sizes - 1] = starts
        is_walk = face_layer[face_of] == WALKWAY
        e = np.sort(np.column_stack([loops, loops[following]]), axis=1)[is_walk]
        owners = face_of[is_walk]
        key = e[:, 0] * (len(points) + 1) + e[:, 1]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        same = np.flatnonzero(ks[1:] == ks[:-1])
        for i in same:
            a, b = find(int(owners[order[i]])), find(int(owners[order[i + 1]]))
            if a != b:
                parent[a] = b
        roots = np.array([find(int(f)) for f in walk])
        island[walk] = roots

    names: list[str] = []
    layers: list[int] = []
    index: dict[str, int] = {}
    face_object = np.full(len(sizes), -1, dtype=np.int64)

    def assign(faces, name, layer):
        if name not in index:
            index[name] = len(names)
            names.append(name)
            layers.append(layer)
        face_object[faces] = index[name]

    for layer in (MOTORWAY, ROAD, WALKWAY, GRASS):
        members = np.flatnonzero(face_layer == layer)
        if not len(members):
            continue
        if layer == MOTORWAY:
            assign(members, object_name(prefix, layer, None), layer)
            continue
        if layer == WALKWAY:
            for root in np.unique(island[members]):
                faces = members[island[members] == root]
                area = np.abs(_areas(points, loops, starts, sizes, faces))
                w = area / max(area.sum(), 1e-9)
                zone = zone_of(float((dx[faces] * w).sum()), float((dy[faces] * w).sum()), focus_m)
                assign(faces, object_name(prefix, layer, zone), layer)
            continue
        zones = [zone_of(float(a), float(b), focus_m) for a, b in zip(dx[members], dy[members])]
        for zone in sorted(set(zones)):
            faces = members[[z == zone for z in zones]]
            assign(faces, object_name(prefix, layer, zone), layer)
    return face_object, names, layers


def _areas(points, loops, starts, sizes, faces):
    out = np.zeros(len(faces))
    for i, f in enumerate(faces):
        ring = points[loops[starts[f]: starts[f] + sizes[f]]]
        x, y = ring[:, 0], ring[:, 1]
        out[i] = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return out


__all__ = [
    "GRASS",
    "GroundArea",
    "LAYER_NAMES",
    "LayeredGround",
    "MATERIALS",
    "MATERIAL_NAMES",
    "MOTORWAY",
    "ROAD",
    "TERRAIN",
    "WALKWAY",
    "build_layered_ground",
    "classify",
    "object_name",
    "zone_of",
]
