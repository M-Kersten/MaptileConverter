"""The ground the way the modellers build it: one sheet, in layers they can grab.

The hand-made Hazeldonk model the modellers work from is built in layers,
each a separate object a person can grab:

* **Terrain**, with the aerial on it.
* **Roads, the motorway, walkways and grass**, each a separate set of faces.
  Their own colour, no photo.
* **Kerbs** are not modelled at all: every walkway and traffic island is a flat
  outline carrying the modellers' curb generator, which raises the walkway and
  builds the kerb round it. Edit the outline and the kerb follows.

Their file lays the layers a few centimetres over a full terrain. With our
model in hand they asked for the ground once: a terrain under every road is
ground drawn twice, and deleting it leaves holes wherever no layer is. So
here the ground is one sheet cut into those objects. It is laid out in quads
along the outlines of every layer (`layered_mesh`): rungs straight across the
roads, loops along the verges, and one grid of the same squares everywhere
else, turned to the way the streets run. The outlines are edges of it, so
each layer is simply the faces under it and the terrain object what is left
-- they meet edge to edge, sharing every vertex, and none lies on another.

Water is cut into the terrain, with a flat water object over it, and along
quay walls the ground drops straight down (`harbour` builds the quay faces,
jetties and mooring posts). Bridges and tunnels are left to the structures
stage.
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
# need a colour for. Sierbestrating, Tegels, Betonstenen and Grind are what the
# BGT says a street is paved with where it says (`paving_material`): a city
# centre is brick and setts from wall to wall, not yellow concrete.
MATERIALS: dict[str, tuple[tuple[float, float, float], float]] = {
    "Asfalt": ((0.0496, 0.0496, 0.0496), 0.5),
    "Bakstenen": ((0.1414, 0.0969, 0.0822), 0.5),
    "Fietspad": ((0.2, 0.045, 0.035), 0.6),
    "Verharding": ((0.25, 0.25, 0.24), 0.7),
    "Beton_Geel": ((0.6146, 0.4983, 0.1433), 0.5),
    "Beton": ((0.8, 0.8, 0.8), 0.5),
    "Gras": ((0.0697, 0.1031, 0.0007), 1.0),
    "Sierbestrating": ((0.2159, 0.1878, 0.15), 0.7),
    "Tegels": ((0.305, 0.2961, 0.2705), 0.6),
    "Betonstenen": ((0.2307, 0.2159, 0.1878), 0.65),
    "Grind": ((0.3515, 0.2961, 0.2016), 0.9),
}
MATERIAL_NAMES = tuple(MATERIALS)

# What a BGT surface is paved with, by its `plus_fysiek_voorkomen` (matched by
# substring: "gebakken klinkers", "betonstraatstenen") and failing that its
# `fysiek_voorkomen`. Over the Ganzenmarkt in Utrecht 116 of 153 road parts
# say which; in Bourtange 73 of 73.
PAVINGS: tuple[tuple[str, str], ...] = (
    ("klinker", "Bakstenen"),
    ("sierbestrating", "Sierbestrating"),
    ("natuursteen", "Sierbestrating"),
    ("betonstraatst", "Betonstenen"),  # -steen, -stenen
    ("grasbeton", "Betonstenen"),
    ("tegel", "Tegels"),
    ("asfalt", "Asfalt"),
    ("beton", "Verharding"),
    ("grind", "Grind"),
    ("schelpen", "Grind"),
    ("puin", "Grind"),
    ("half verhard", "Grind"),
)

# The surfaces each layer can be drawn in. A road takes what it is paved with;
# a walkway too, and Beton_Geel, the modellers' own walkway, where the BGT does
# not say. A walkway the survey shows level with the road beside it is laid as
# road (`flush_walkways`), in its own material.
ROAD_SURFACES = ("Asfalt", "Bakstenen", "Sierbestrating", "Betonstenen", "Tegels", "Fietspad")
WALKWAY_SURFACES = (
    "Beton_Geel", "Bakstenen", "Sierbestrating", "Betonstenen", "Tegels", "Asfalt",
    "Verharding", "Grind",
)

# Which outline wins where two overlap, best first. The BGT is a partition,
# so this only settles the slivers where two surveys disagree.
PRIORITY = (
    ((MOTORWAY, "Asfalt"),)
    + tuple((ROAD, material) for material in ROAD_SURFACES)
    + tuple((WALKWAY, material) for material in WALKWAY_SURFACES)
    + ((ROAD, "Beton_Geel"), (ROAD, "Verharding"), (ROAD, "Grind"), (GRASS, "Gras"))
)

# How far a walkway may stand above the road beside it, measured in the
# survey, and still be level with it: no kerb. A Dutch kerb shows 8 to 15 cm;
# the old centres' low ones 5 to 8. Over the Ganzenmarkt 21 of 53 footpaths
# beside a road are within 3 cm of it, and in Hindeloopen 24 of 25.
FLUSH_BELOW_M = 0.05

# A green verge or island counts as an island -- raised, kerbed all round --
# when paving surrounds at least this much of it and it is no bigger than
# `ISLAND_MAX_M2`. A field between two roads is not an island.
ISLAND_SHARE = 0.9
ISLAND_MAX_M2 = 2500.0

# How far each layer lies above the ground: not at all. The roads, walkways
# and grass are the ground, each its own object, sharing the vertices along
# every edge where two meet; nothing is lifted onto a terrain under it. Only
# a walkway's top is higher, by what the curb generator raises it, and a lamp
# post on a footpath stands on that.
DEFAULT_OFFSETS = {ROAD: 0.0, MOTORWAY: 0.0, WALKWAY: 0.0, GRASS: 0.0}

# Ground under water: terrain, carved down to a bed, with no layer on it. A
# terrain code of its own (material slot 15, which no material uses) only so
# the mesher can tell it apart; it comes out as plain terrain.
WATER_CODE = TERRAIN * 16 + 15
# Under a retaining wall: terrain, slot 14, carrying the drop from the street
# on one side to the wharf on the other (`werf`), under the wall standing on
# it. Plain terrain once built.
WALL_CODE = TERRAIN * 16 + 14
# The survey's ridge under a bridge spills past the deck's outline: the
# bare earth is drawn in across the gap and read on a half-metre grid. Water
# points this near a deck are put on the bed like those under it.
NEAR_DECK_M = 3.0
# How far under the water level the bed of a ditch is cut. No water surface
# is drawn over it, so this is the dip itself, not a depth for a plane to
# hide: enough to read as a ditch from the air.
DITCH_DEPTH_M = 0.6


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
    functie: str = ""  # the BGT's own word for it: "voetpad op trap" is a stair


def paving_material(physical: str, plus: str, default: str) -> str:
    """What a surface is paved with, from the BGT's words for it."""
    for token, material in PAVINGS:
        if token in plus:
            return material
    if "half verhard" in physical:
        return "Grind"
    return default


def is_stair(functie: str) -> bool:
    """A BGT footpath that is a flight of steps: "voetpad op trap"."""
    return "trap" in str(functie or "").lower()


def classify(collection: str, properties: dict) -> tuple[int, str] | None:
    """``(layer, material)`` for one BGT polygon; None leaves it to the terrain.

    The motorway is apart because the modellers keep it apart. Footpaths and
    paved traffic islands are walkways: the curb generator raises them. Green
    traffic islands start as grass and become islands when paving surrounds
    them (`ISLAND_SHARE`). Paving that is no road -- a forecourt, a hard verge
    -- is laid flat without kerbs. Yards, sand and water stay terrain. Each
    takes the material the BGT says it is paved with (`paving_material`).

    A flight of steps ("voetpad op trap") is laid flat as road, under the
    steps the barriers stage stands on it: a kerb round a stair is no kerb.
    """
    from .surfaces import CLASS_CYCLE, CLASS_FOOTPATH, road_class

    functie = str(properties.get("functie") or "").strip().lower()
    physical = str(properties.get("fysiek_voorkomen") or "").strip().lower()
    plus = str(properties.get("plus_fysiek_voorkomen") or "").strip().lower()
    paved = "verhard" in physical and "onverhard" not in physical
    green = "groen" in physical or "gras" in plus or "begroeid" in physical

    if collection == "wegdeel":
        if "rijbaan" in functie and ("autosnelweg" in functie or "autoweg" in functie):
            return MOTORWAY, "Asfalt"
        if is_stair(functie):
            return ROAD, paving_material(physical, plus, "Verharding")
        code = road_class(functie, plus or physical)
        if code == CLASS_FOOTPATH:
            closed = "gesloten" in physical
            return WALKWAY, paving_material(physical, plus, "Verharding" if closed else "Beton_Geel")
        if code == CLASS_CYCLE:
            material = paving_material(physical, plus, "Fietspad")
            # A cycle path in brick is brick; in asphalt it is red asphalt.
            return ROAD, ("Fietspad" if material in ("Asfalt", "Verharding") else material)
        if "onverhard" in physical:
            return None  # a track across a field is the field
        # Open paving is laid in elements, and asphalt is closed paving by
        # definition: an open carriageway the BGT says no more of is brick,
        # which is what most of them are.
        material = paving_material(physical, plus, "Bakstenen" if "open" in physical else "Asfalt")
        if material in ("Verharding", "Beton_Geel"):
            material = "Asfalt"
        return ROAD, material
    if collection == "ondersteunendwegdeel":
        if "verkeerseiland" in functie:
            return (WALKWAY, paving_material(physical, plus, "Beton_Geel")) if paved else (GRASS, "Gras")
        if green:
            return GRASS, "Gras"
        if paved:
            return ROAD, paving_material(physical, plus, "Verharding")
        return None
    if collection == "begroeidterreindeel":
        return GRASS, "Gras"
    if collection == "onbegroeidterreindeel":
        if not paved:
            return None
        # A forecourt in open paving is pavers; in closed paving, concrete.
        return ROAD, paving_material(physical, plus, "Betonstenen" if "open" in physical else "Verharding")
    return None


def flush_walkways(
    areas: list[GroundArea],
    ground_at,
    *,
    flush_below_m: float = FLUSH_BELOW_M,
    min_stations: int = 6,
) -> dict:
    """Lay as road every walkway the survey shows level with the road beside it.

    The curb generator raises every walkway and builds a kerb round it, which
    is what a suburban footpath is. An old centre's pedestrian street is brick
    from wall to wall, the footpath level with the carriageway: Ganzenmarkt
    drawn with kerbs is a road with pavements it does not have. The survey
    tells them apart. Every half metre along where a walkway meets a road, the
    bare ground 0.75 to 1.25 m inside the walkway is compared with the same
    distance into the road; a walkway whose median step is under
    `flush_below_m` is laid as road, in its own material, and has no kerb.

    `ground_at(x, y)` is the bare-earth survey at its own resolution: a grid
    averaged to a metre smooths a 6 cm kerb into a slope. Walkways meeting no
    road, or too little of one to measure, keep their kerb. Changes `areas`
    in place; returns what it measured.
    """
    import shapely

    from .terrain_mesh import _polygonal

    def polygon_of(area):
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in area.rings]
        rings = [r for r in rings if len(r) >= 4]
        if not rings:
            return None
        polygon = _polygonal(shapely.make_valid(shapely.Polygon(rings[0], rings[1:])))
        return None if polygon.is_empty else polygon

    roads = [polygon_of(a) for a in areas if a.layer == ROAD and a.level == 0]
    roads = [r for r in roads if r is not None]
    stats = {"walkways": 0, "measured": 0, "flush": 0, "steps_m": []}
    if not roads:
        return stats
    road = shapely.union_all(np.array(roads, dtype=object))
    near_road = road.buffer(0.4)
    shapely.prepare(road)
    shapely.prepare(near_road)
    offsets = np.array([0.75, 1.0, 1.25])
    for area in areas:
        if area.layer != WALKWAY or area.level != 0:
            continue
        stats["walkways"] += 1
        polygon = polygon_of(area)
        if polygon is None or not polygon.intersects(near_road):
            continue
        shared = polygon.boundary.intersection(near_road)
        lines = [g for g in getattr(shared, "geoms", [shared]) if g.geom_type == "LineString"]
        stations, normals = [], []
        for line in lines:
            coords = np.asarray(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                length = float(np.hypot(*(b - a)))
                if length < 0.25:
                    continue
                along = (b - a) / length
                normal = np.array([-along[1], along[0]])
                for t in np.arange(0.25, length, 0.5):
                    stations.append(a + along * t)
                    normals.append(normal)
        if len(stations) < min_stations:
            continue
        stations = np.asarray(stations)
        normals = np.asarray(normals)
        # Point every normal into the walkway.
        probe = stations + normals * 0.6
        inward = shapely.contains_xy(polygon, probe[:, 0], probe[:, 1])
        normals[~inward] *= -1.0
        probe = stations + normals * 0.6
        keep = shapely.contains_xy(polygon, probe[:, 0], probe[:, 1])
        inside = stations[:, None, :] + normals[:, None, :] * offsets[None, :, None]
        outside = stations[:, None, :] - normals[:, None, :] * offsets[None, :, None]
        on_road = shapely.contains_xy(road, outside[..., 0], outside[..., 1]).all(axis=1)
        keep &= on_road
        if keep.sum() < min_stations:
            continue
        z_in = np.asarray(ground_at(inside[keep, :, 0].ravel(), inside[keep, :, 1].ravel()),
                          dtype=np.float64).reshape(-1, len(offsets))
        z_out = np.asarray(ground_at(outside[keep, :, 0].ravel(), outside[keep, :, 1].ravel()),
                           dtype=np.float64).reshape(-1, len(offsets))
        good = (np.isfinite(z_in).sum(axis=1) >= 2) & (np.isfinite(z_out).sum(axis=1) >= 2)
        if good.sum() < min_stations:
            continue
        with np.errstate(all="ignore"):
            steps = np.nanmedian(z_in[good], axis=1) - np.nanmedian(z_out[good], axis=1)
        step = float(np.median(steps))
        stats["measured"] += 1
        stats["steps_m"].append(round(step, 3))
        if step < flush_below_m:
            area.layer = ROAD
            stats["flush"] += 1
    steps = np.asarray(stats["steps_m"])
    stats["step_median_m"] = round(float(np.median(steps)), 3) if len(steps) else None
    LOG.info(
        "kerbs: %d of %d walkways meet a road; %d of them level with it, laid as road "
        "without a kerb (median step %s m)",
        stats["measured"], stats["walkways"], stats["flush"], stats["step_median_m"],
    )
    stats["steps_m"] = stats["steps_m"][:200]
    return stats


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
    lift: dict[int, float] | None = None,
    walkway_raise_m: float = 0.13,
    max_rounds: int = 60,
    water=(),
    ditch_depth_m: float = DITCH_DEPTH_M,
    mesher: str = "quads",
    quays=(),
    square_m: float | None = None,
    ground_at=None,
    flush_below_m: float = FLUSH_BELOW_M,
    walls=(),
    under_decks=(),
) -> LayeredGround:
    """The terrain and its layers, from the survey and the BGT areas.

    `tolerance_m` is how far the terrain may be from the survey where it is
    simplified; `outline_m` how far a simplified outline may stray from the
    BGT's. `water` is the WaterBody list: carved into the terrain as ditches
    and canals `ditch_depth_m` below their level. Each layer is one object;
    the modellers divide the model themselves.

    `square_m` is the size of the grid squares and the spacing of the
    stations along every outline (`layered_mesh.NEAR_CELL_M` if not given),
    the same everywhere; a whole number of survey grid squares.

    `ground_at(x, y)`, the bare-earth survey at its own resolution, lets
    walkways level with the road beside them be laid as road, without a kerb
    (`flush_walkways`); without it every walkway is kerbed.

    `walls` are retaining walls (shapely, RD): each is a region of its own,
    its edge on the high side at the ground there and on the low side at the
    ground there, read just beyond it (`werf`). A canal street four metres
    over its wharf drops at the wall, rather than down a bank of grid squares.

    `under_decks` are bridge decks over water, ``(polygon, along)`` in RD
    (`structures.deck_outlines`): the ground under each is read beside the
    bridge along the water rather than off a survey that cannot see there.

    `mesher` "quads" lays the terrain out in quads (`layered_mesh`);
    "ribcage" lays it out as the 3D team draws it: ribs across what is
    narrow, a lattice turned to each space in what is wide, triangles at the
    edges, and no grid (`ribcage_mesh`); "triangles" is the triangulation
    refined until every grid node is within `tolerance_m`, which is also what
    a failed quad or ribcage mesh falls back to.
    """
    import shapely

    started = time.time()
    lift = {**DEFAULT_OFFSETS, **(lift or {})}
    xs = np.asarray(terrain.xs, dtype=np.float64)
    ys = np.asarray(terrain.ys, dtype=np.float64)
    ox, oy = float(xs[0]), float(ys[0])
    width, height = float(xs[-1] - ox), float(ys[-1] - oy)
    frame = shapely.box(0.0, 0.0, width, height)

    kerbs = (
        flush_walkways(areas, ground_at, flush_below_m=flush_below_m)
        if ground_at is not None else None
    )

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
    bodies = []
    for body in water or ():
        rings = [np.asarray(r, dtype=np.float64)[:, :2] - [ox, oy] for r in body.rings]
        rings = [r for r in rings if len(r) >= 4]
        if not rings:
            continue
        polygon = _polygonal(shapely.Polygon(rings[0], rings[1:]))
        if polygon.is_empty:
            continue
        by_code.setdefault(WATER_CODE, []).append(polygon)
        bodies.append((polygon.simplify(0.2), float(body.level_nap)))
    wall_bands = []
    if walls:
        from .werf import wall_band

        for wall in walls:
            band = wall_band(wall)
            if band.is_empty:
                continue
            wall_bands.append(band)
            local = shapely.transform(band, lambda c: c - [ox, oy])
            by_code.setdefault(WALL_CODE, []).append(_polygonal(local))
    order = [_code(layer, material) for layer, material in PRIORITY]
    # Water after the paving, so a road over a culvert stays a road.
    order.insert(order.index(_code(GRASS, "Gras")), WATER_CODE)
    # A wall before everything: it is where the street and the wharf meet.
    order.insert(0, WALL_CODE)
    # Anything PRIORITY does not rank still gets drawn, last.
    order += sorted(code for code in by_code if code not in order)
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

    gx, gy = np.meshgrid(xs - ox, ys - oy)
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    grid_z = np.asarray(terrain.measured if terrain.measured is not None else terrain.heights,
                        dtype=np.float64).ravel()
    missing, rounds = 0, 0
    mesh_stats: dict = {}
    face_part = None
    built = None
    if mesher in ("quads", "ribcage"):
        from .layered_mesh import quad_terrain
        from .ribcage_mesh import ribcage_terrain

        generate = ribcage_terrain if mesher == "ribcage" else quad_terrain
        t = time.time()
        try:
            built = generate(
                terrain, pieces, bodies, codes_wet=WATER_CODE,
                depth_m=float(ditch_depth_m), tolerance_m=float(tolerance_m),
                # Roads get rows wide enough to ladder straight across;
                # yards, forecourts and footpaths the grid's.
                paved_codes={_code(MOTORWAY, "Asfalt")}
                | {_code(ROAD, material) for material in ROAD_SURFACES},
                narrow_codes={_code(WALKWAY, material) for material in WALKWAY_SURFACES}
                | {_code(ROAD, m) for m in ("Beton_Geel", "Verharding", "Grind")},
                # What the curb generator kerbs: no thin strip of anything
                # else goes into it while it has another neighbour.
                kerbed_codes={k for _, k in pieces if k // 16 == WALKWAY},
                quays=quays,
                **({"near_m": float(square_m)} if square_m else {}),
            )
        except Exception as exc:  # noqa: BLE001 - the triangulation always works
            LOG.warning(
                "layered ground: the %s terrain could not be built (%s); "
                "using the triangulation", mesher, exc,
            )
            mesh_stats["quad_failure"] = str(exc)[:200]
        timings["terrain_s"] = round(time.time() - t, 1)
    if built is not None:
        vertices_rd, sizes, loops, codes, face_part, quad_stats = built
        mesh_stats.update(quad_stats)
        points = vertices_rd[:, :2] - [ox, oy]
        z = vertices_rd[:, 2].copy()
    else:
        # ---- a triangulation refined until it fits the survey -------------
        t = time.time()
        points, segments = _constraints(pieces)

        def heights_at(xy):
            return np.asarray(terrain.sample(xy[:, 0] + ox, xy[:, 1] + oy), dtype=np.float64)

        points, z, tris, missing, rounds = _fit(
            points, segments, heights_at, grid_xy, grid_z,
            tolerance_m=float(tolerance_m), bounds=(0.0, 0.0, width, height),
            max_rounds=max_rounds,
        )
        timings["terrain_s"] = round(time.time() - t, 1)

        # ---- faces: each triangle takes the region it lies in -------------
        t = time.time()
        tri_code = _classify_triangles(points, tris, pieces)
        faces, codes = _faces(points, tris, tri_code)
        timings["faces_s"] = round(time.time() - t, 1)
        mesh_stats["mesher"] = "triangles"
        sizes = np.array([len(f) for f in faces], dtype=np.int64)
        loops = np.concatenate(faces).astype(np.int64) if faces else np.zeros(0, np.int64)
    walled = 0
    if wall_bands:
        z, walled = _wall_edges(
            points + [ox, oy], z, loops, sizes, np.asarray(codes) == WALL_CODE,
            ground_at if ground_at is not None else terrain.sample,
        )
    shaded = 0
    if under_decks:
        def water_level_at(point, reach_m=30.0):
            """The level of the water nearest a point, RD, within `reach_m`."""
            local = shapely.Point(point[0] - ox, point[1] - oy)
            near = [(poly.distance(local), level) for poly, level in bodies]
            near = [n for n in near if n[0] <= reach_m]
            return min(near)[1] if near else None

        z, shaded = _under_decks(
            points + [ox, oy], z, loops, sizes, (np.asarray(codes) // 16 == TERRAIN) & (np.asarray(codes) % 16 == 15),
            under_decks, ground_at if ground_at is not None else terrain.sample,
            water_level_at=water_level_at,
        )
    face_layer, face_material = _split_code(codes)
    face_material = np.where(face_layer == TERRAIN, -1, face_material)
    face_layer, face_material, blunted = _blunt_walkway_tips(
        points, loops, sizes, face_layer, face_material,
    )

    vertices = np.column_stack([points[:, 0] + ox, points[:, 1] + oy, z])
    edges, edge_on, vertex_on = _layer_outlines(loops, sizes, face_layer, len(points))
    mesh = TerrainMesh(
        vertices=vertices,
        measured_z=z.copy(),
        face_sizes=sizes.astype(np.int32),
        face_loops=loops,
        face_kind=face_layer.astype(np.int8),
        face_part=(
            np.zeros(len(sizes), dtype=np.int8) if face_part is None
            else np.asarray(face_part, dtype=np.int8)
        ),
        vertex_on=vertex_on,
        edges=edges,
        edge_on=edge_on,
        grid_m=float(terrain.grid_m),
        locator_cell=8.0,
    )

    # ---- objects, named and grouped as the modellers do --------------------
    face_object, names, object_layer = _objects(face_layer, prefix=prefix)
    top = dict(lift)
    top[WALKWAY] = lift[WALKWAY] + float(walkway_raise_m)

    # ---- how well it fits -------------------------------------------------
    locator = mesh.locator()
    drawn = locator.interpolate(mesh.vertices[:, 2], grid_xy[:, 0] + ox, grid_xy[:, 1] + oy)
    error = np.abs(drawn - grid_z)
    # Under water the terrain is carved below the survey on purpose.
    wet = np.zeros(len(grid_xy), dtype=bool)
    if bodies:
        wet_area = shapely.union_all(np.array([b for b, _ in bodies], dtype=object))
        shapely.prepare(wet_area)
        wet = shapely.intersects_xy(wet_area, grid_xy[:, 0], grid_xy[:, 1])
    error = error[np.isfinite(error) & ~wet]
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
        "walkway_tips_blunted": int(blunted),
        "retaining_walls": len(wall_bands),
        "wall_edge_points_set": int(walled),
        "points_under_bridges_set": int(shaded),
        "water_bodies": int(len(bodies)),
        **mesh_stats,
        "faces_by_layer": {
            LAYER_NAMES[k]: int((face_layer == k).sum()) for k in LAYER_NAMES
        },
        "objects": {name: int((face_object == i).sum()) for i, name in enumerate(names)},
        "parts_on_bridges_or_tunnels": int(skipped_levels),
        **({"kerbs": kerbs} if kerbs is not None else {}),
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


def _under_decks(xy, z, loops, sizes, wet_face, decks, ground_at, *, beyond_m: float = 0.6,
                 reach_m: float = 60.0, water_level_at=None):
    """The ground under bridge decks at its height either side of the bridge.

    Each point under a deck that is not wholly under water is read where a
    line from it along the water leaves the deck, both ways, `beyond_m`
    past its edge: the wharf under a bridge is the wharf beside it, the
    street at an abutment the street beside it. Where the two disagree the
    lower wins: a wharf that goes under the bridge on one side and meets a
    stair on the other still goes under it. Where both are water, the point
    is under the water too, at the depth a canal bed is cut (`water_level_at`).
    Returns the heights and how many points were set.
    """
    import shapely

    xy = np.asarray(xy, dtype=np.float64)[:, :2]
    z = np.asarray(z, dtype=np.float64).copy()
    sizes = np.asarray(sizes, dtype=np.int64)
    loops = np.asarray(loops, dtype=np.int64)
    face_of = np.repeat(np.arange(len(sizes)), sizes)
    dry = np.zeros(len(xy), dtype=bool)
    dry[loops[~np.asarray(wet_face, dtype=bool)[face_of]]] = True
    changed = 0
    for deck, along in decks:
        along = np.asarray(along, dtype=np.float64)
        under = shapely.contains_xy(deck.buffer(-0.05), xy[:, 0], xy[:, 1])
        # Under water only: the bed, which easing its banks against the
        # ridge had lifted into a mound across the canal under the bridge.
        # Near the deck too: the ridge spills past its outline, and a water
        # point 0.6 m beside a bridge on the Oudegracht stood at 1.8 m, its
        # faces a shoal across the canal under the arch.
        if water_level_at is not None:
            near = shapely.contains_xy(deck.buffer(NEAR_DECK_M), xy[:, 0], xy[:, 1])
            for point in np.flatnonzero(near & ~dry).tolist():
                level = water_level_at(xy[point])
                if level is not None and z[point] > float(level) - DITCH_DEPTH_M:
                    z[point] = float(level) - DITCH_DEPTH_M
                    changed += 1
        inside = under & dry
        for point in np.flatnonzero(inside).tolist():
            readings = []
            for sign in (1.0, -1.0):
                ray = shapely.LineString([xy[point], xy[point] + sign * along * reach_m])
                inner = ray.intersection(deck)
                if inner.is_empty:
                    continue
                parts = getattr(inner, "geoms", [inner])
                # The stretch of the ray from the point to where it leaves.
                first = min(parts, key=lambda g: g.distance(shapely.Point(xy[point])))
                exit_at = max(np.hypot(*(np.asarray(first.coords) - xy[point]).T))
                at = xy[point] + sign * along * (exit_at + beyond_m)
                value = float(np.asarray(ground_at(np.array([at[0]]), np.array([at[1]])), dtype=np.float64)[0])
                if np.isfinite(value):
                    readings.append(value)
            if readings:
                z[point] = float(min(readings))
                changed += 1
            elif water_level_at is not None:
                level = water_level_at(xy[point])
                if level is not None:
                    z[point] = min(float(z[point]), float(level) - DITCH_DEPTH_M)
                    changed += 1
    return z, changed


def _wall_edges(xy, z, loops, sizes, on_wall, ground_at, *, reach_m: float = 0.8):
    """Every point of a retaining wall's region at the ground beside it.

    A point on the wall's edge is shared with the ground on one side: the
    street or the wharf. It takes that ground's height, read `reach_m` out
    towards each face beside it that is not wall, from the survey at its own
    half metre: the grid the mesher reads smooths a four-metre step into a
    slope. A point inside the region, shared with nothing else, takes the
    highest of its neighbours: it is under the wall's top. Returns the heights
    and how many points were set.
    """
    xy = np.asarray(xy, dtype=np.float64)[:, :2]
    z = np.asarray(z, dtype=np.float64).copy()
    sizes = np.asarray(sizes, dtype=np.int64)
    loops = np.asarray(loops, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    face_of = np.repeat(np.arange(len(sizes)), sizes)
    on_wall = np.asarray(on_wall, dtype=bool)
    centroids = np.zeros((len(sizes), 2))
    np.add.at(centroids, face_of, xy[loops])
    centroids /= sizes[:, None]
    wall_points = np.unique(loops[on_wall[face_of]])
    if not len(wall_points):
        return z, 0
    # For each wall point, the faces round it that are not wall.
    beside: dict[int, list[int]] = {}
    for loop in np.flatnonzero(np.isin(loops, wall_points) & ~on_wall[face_of]).tolist():
        beside.setdefault(int(loops[loop]), []).append(int(face_of[loop]))
    readings = []
    for point, faces in beside.items():
        out = centroids[faces] - xy[point]
        norm = np.hypot(out[:, 0], out[:, 1])
        out = out / np.maximum(norm, 1e-9)[:, None]
        at = xy[point] + out * reach_m
        readings.append((point, at, faces))
    if readings:
        flat = np.vstack([at for _, at, _ in readings])
        values = np.asarray(ground_at(flat[:, 0], flat[:, 1]), dtype=np.float64)
        k = 0
        for point, at, faces in readings:
            v = values[k:k + len(at)]
            k += len(at)
            if np.isfinite(v).any():
                z[point] = float(np.nanmedian(v))
            else:
                # No survey there: the ground's own points round it, off the wall.
                others = np.unique(np.concatenate([loops[starts[f]:starts[f] + sizes[f]] for f in faces]))
                others = others[~np.isin(others, wall_points)]
                if len(others):
                    z[point] = float(np.median(z[others]))
    inside = np.setdiff1d(wall_points, np.fromiter(beside.keys(), dtype=np.int64))
    for point in inside.tolist():
        faces = face_of[loops == point]
        near = np.unique(np.concatenate([loops[starts[f]:starts[f] + sizes[f]] for f in faces]))
        near = near[near != point]
        if len(near):
            z[point] = float(np.max(z[near]))
    return z, len(beside) + len(inside)


def _blunt_walkway_tips(points, loops, sizes, face_layer, face_material, *,
                        limit_deg: float | None = None, rounds: int = 12):
    """Needles and slits in the walkway's outline, taken out after meshing.

    `_without_hairpins` takes the needles out of the BGT's outlines, but the
    mesher makes its own: an outline simplified one way on one side and
    another on the other leaves a wedge between them, and over the
    Ganzenmarkt one 0.6 m wide and 5 m long ended in a 7 degree tip, off
    which the curb generator's mitre flew 4.5 m. A tip is a vertex where the
    walkway's faces meet in less than `limit_deg` all told, and its walkway
    faces become whatever ground shares most edge with them. A slit is the
    same the other way round, a sliver of road into the walkway, and its
    faces become walkway. It goes round again, since the next vertex down
    may be the new tip.
    Returns the layers, the materials and how many faces changed hands.
    """
    limit_deg = HAIRPIN_DEG if limit_deg is None else float(limit_deg)
    sizes = np.asarray(sizes, dtype=np.int64)
    loops = np.asarray(loops, dtype=np.int64)
    face_layer = np.asarray(face_layer, dtype=np.int64).copy()
    face_material = np.asarray(face_material, dtype=np.int64).copy()
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    face_of = np.repeat(np.arange(len(sizes)), sizes)
    following = np.arange(len(loops)) + 1
    following[starts + sizes - 1] = starts
    previous = np.arange(len(loops)) - 1
    previous[starts] = starts + sizes - 1
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    # The inside angle of every face at every one of its vertices, past 180
    # where a face is not convex: the turn from the edge in to the edge out,
    # read the way the face winds.
    into = xy[loops] - xy[loops[previous]]
    out = xy[loops[following]] - xy[loops]
    turn = np.arctan2(into[:, 0] * out[:, 1] - into[:, 1] * out[:, 0], (into * out).sum(axis=1))
    cross = xy[loops, 0] * xy[loops[following], 1] - xy[loops[following], 0] * xy[loops, 1]
    winding = np.sign(np.bincount(face_of, weights=cross, minlength=len(sizes)))
    winding[winding == 0] = 1.0
    corner = np.degrees(np.pi - turn * winding[face_of])
    # Each edge, by its two vertices, and the faces either side of it.
    pairs = np.sort(np.column_stack([loops, loops[following]]), axis=1)
    _, edge_id = np.unique(pairs, axis=0, return_inverse=True)
    edge_id = edge_id.ravel()
    order = np.argsort(edge_id, kind="stable")
    total_all = np.bincount(loops, weights=corner, minlength=len(xy))
    # A vertex the faces go all the way round; on the area's edge they do not.
    interior = total_all > 359.0
    changed = 0

    def across(face, want_walkway: bool):
        """(layer, material) of the faces across this face's edges."""
        out = []
        for loop in range(starts[face], starts[face] + sizes[face]):
            same = order[np.searchsorted(edge_id[order], edge_id[loop]):]
            for other in same[:4].tolist():
                if edge_id[other] != edge_id[loop]:
                    break
                neighbour = face_of[other]
                if neighbour != face and (face_layer[neighbour] == WALKWAY) == want_walkway:
                    out.append((int(face_layer[neighbour]), int(face_material[neighbour])))
        return out

    for _ in range(rounds):
        walk = face_layer[face_of] == WALKWAY
        total = np.bincount(loops[walk], weights=corner[walk], minlength=len(xy))
        touched = np.bincount(loops[walk], minlength=len(xy)) > 0
        # A needle of walkway: its faces meet in less than the limit.
        tips = touched & (total < limit_deg)
        # A slit into the walkway, the same needle the other way round: what
        # is not walkway meets in less than the limit, inside the area, and
        # the kerb's mitre runs away up it just the same.
        rest = total_all - total
        slits = touched & interior & (rest > 1e-6) & (rest < limit_deg)
        if not tips.any() and not slits.any():
            break
        moved = 0
        for face in np.unique(face_of[walk & tips[loops]]).tolist():
            found = across(face, want_walkway=False)
            if found:
                values, counts = np.unique(np.array(found), axis=0, return_counts=True)
                face_layer[face], face_material[face] = values[int(np.argmax(counts))]
                moved += 1
        for face in np.unique(face_of[~walk & slits[loops]]).tolist():
            found = across(face, want_walkway=True)
            if found:
                values, counts = np.unique(np.array(found), axis=0, return_counts=True)
                face_layer[face], face_material[face] = values[int(np.argmax(counts))]
                moved += 1
        if not moved:
            break
        changed += moved
    if changed:
        LOG.info("layered ground: %d faces at needle tips and slits in the walkway changed hands", changed)
    return face_layer, face_material, changed


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


def object_name(prefix: str, layer: int) -> str:
    """The one object a layer is: Hazeldonk_Wegen, Hazeldonk_Voetpaden."""
    return f"{prefix}_{LAYER_WORDS[layer]}"


def _objects(face_layer, *, prefix):
    """Which object each face belongs to: one per layer.

    The modellers used to have the ground split into an east and a west half
    round the middle and quadrants beyond, as their hand-made file is. With
    the model in hand they asked for it whole: they divide it themselves, for
    their own culling. The terrain's own faces belong to no layer object (-1).
    """
    names: list[str] = []
    layers: list[int] = []
    face_object = np.full(len(face_layer), -1, dtype=np.int64)
    for layer in (MOTORWAY, ROAD, WALKWAY, GRASS):
        members = np.flatnonzero(face_layer == layer)
        if not len(members):
            continue
        face_object[members] = len(names)
        names.append(object_name(prefix, layer))
        layers.append(layer)
    return face_object, names, layers


__all__ = [
    "FLUSH_BELOW_M",
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
    "PAVINGS",
    "build_layered_ground",
    "classify",
    "flush_walkways",
    "is_stair",
    "object_name",
    "paving_material",
]
