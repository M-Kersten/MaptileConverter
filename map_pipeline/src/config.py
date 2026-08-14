"""Config loading, with defaults for everything the plan leaves implicit.

``config.json`` only has to carry the interesting bits. Every knob below has a
default that works for a 1 km² area, so a minimal config stays readable.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .geo import BBox, GeoContext, parse_bbox, validate_bbox
from .terrain_mesh import is_grid_size, next_grid_size

LOG = logging.getLogger(__name__)

# The aerial layer is an 8 cm ortho, so a 1 km area resolves fully at about
# 12500 px. Above that the pipeline is upsampling a JPEG, not gaining detail.
AERIAL_SOURCE_M_PER_PX = 0.08

DEFAULTS: dict[str, Any] = {
    "name": "demo_area",
    "aerial": {
        "layer": "Luchtfoto Actueel Ortho 8cm RGB",
        "size_px": 4096,
        # The PDOK WMS advertises MaxWidth/MaxHeight of 2500 and answers
        # anything larger with a ServiceException, so the mosaic is not
        # optional above 2500 px. Kept below the cap for headroom.
        "max_request_px": 2000,
        "wms_url": "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0",
        "wmts_url": "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0",
        # The service only offers image/jpeg for GetMap. Tiles come back as
        # JPEG and are written out as a single PNG.
        "request_format": "image/jpeg",
        "timeout_s": 180,
        "max_retries": 4,
    },
    "terrain": {
        "ahn_model": "DTM",
        "resolution_m": 0.5,
        "mesh_vertices_per_side": 257,
        "wcs_url": "https://service.pdok.nl/rws/ahn/wcs/v1_0",
        "timeout_s": 300,
        "max_retries": 4,
        # DTM strips buildings out, so dense city centres come back mostly
        # nodata. Filling is a normal part of the job, not an error path.
        "max_nodata_fraction": 0.95,
        "smooth_iterations": 1,
        # How far the terrain mesh may stray from the height grid, in metres.
        # Above zero the regular grid is replaced by a triangulation that keeps
        # its vertices where the ground moves and drops them where it does not,
        # which is most of a Dutch bbox. 0.10 m is roughly AHN's own vertical
        # accuracy (5 cm systematic plus 5 cm stochastic), so at the default
        # the mesh gives up nothing the source could resolve in the first
        # place. Set to 0 for the old full grid.
        "simplify_tolerance_m": 0.10,
        # Fold the terrain along real features instead of along grid diagonals,
        # so a canal bank or a kerb is an edge loop you can select in Blender
        # rather than a staircase of triangles that happens to be dense there.
        # Empty turns it off and leaves the bisection mesh.
        "breaklines": ["water", "roads", "unpaved", "green", "buildings"],
        # How far a simplified outline may stray from the surveyed one. The BGT
        # carries vertices centimetres apart along a visibly straight kerb.
        "breakline_simplify_m": 0.15,
        # Contour spacing. 0 leaves contours out; they are the only breaklines
        # that follow the ground rather than something drawn on it, and also
        # the ones that cost the most triangles.
        "contour_interval_m": 0.5,
    },
    "buildings": {
        "lod": "2.2",
        "api_url": "https://api.3dbag.nl/collections/pand/items",
        # 3DBAG publishes the same LoD2.2 data twice, on two separate services.
        # api.3dbag.nl is the one that goes down, so a run that cannot reach it
        # falls through to the static CityJSON tiles on data.3dbag.nl instead of
        # failing. Put "tiles" first to skip the API entirely; the tiles are
        # cached across areas and runs, so a warm cache needs no network.
        "sources": ["api", "tiles"],
        "tiles_url": "https://data.3dbag.nl/{version}/tiles",
        "tiles_index_url": "https://data.3dbag.nl/api/BAG3D/wfs",
        # 3DBAG keeps every dated release and publishes no "latest" alias, so
        # this is bumped by hand. Releases are listed at 3dbag.nl/en/download.
        "tiles_version": "v20250903",
        # A short knock on the API before committing the full timeout budget:
        # without it a dead API costs timeout x retries before the fallback
        # gets a turn, which is long enough that people kill the run instead.
        "probe_url": "https://api.3dbag.nl/collections/pand",
        "probe_timeout_s": 8.0,
        "page_limit": 500,
        "timeout_s": 180,
        "max_retries": 4,
        "max_pages": 2000,
        # Anything taller than this in a 1 km² Dutch area is a data error.
        "max_height_m": 200.0,
        "min_height_m": 1.0,
        # Buildings are pushed this far below the terrain under their footprint
        # so no gap opens up where AHN and the 3DBAG ground level disagree.
        "ground_skirt_m": 0.5,
        "merge": "single",
        # "centroid" assigns each building to the area containing its centre,
        # which keeps the model bounded without cutting geometry open.
        # "intersect" keeps every building the API returns.
        "clip_mode": "centroid",
    },
    "facade": {
        # One storey per texture tile vertically, so windows line up per floor
        # whatever the building height.
        "floor_height_m": 3.0,
        "tile_width_m": 4.0,
        "texture_px": 512,
        # One style per era. 3DBAG supplies a construction year on effectively
        # every building, and era predicts a facade far better than height.
        # Set to 1 for a single wall material.
        "variants": 5,
        "seed": 20240501,
        # A normal map gives the windows and storey bands real relief under a
        # moving light. It costs one extra texture and exports through FBX.
        "normal_map": True,
        "relief_depth": 0.035,
        # A distinct ground storey is what stops a facade reading as a
        # repeating grid. Walls are cut at this height to carry it.
        "ground_floor": True,
        "ground_floor_height_m": 3.6,
        # Buildings without ordinary storeys get their own spacing. A church
        # bay and a warehouse bay are both much larger than a domestic one.
        "monumental_tile_m": 7.0,
        "monumental_bay_m": 9.0,
        "industrial_tile_m": 9.0,
        "industrial_bay_m": 6.0,
        # Photographed masonry under the generated windows, from Poly Haven
        # (CC0). Downloaded once and cached under work/_textures, so only the
        # first run needs the network; set false to stay fully procedural.
        "photo_textures": True,
        # How far the photograph is pulled towards the era's colour. 0 keeps
        # the photograph as shot, 1 lands it exactly on the palette.
        "photo_tint": 0.6,
    },
    "trees": {
        "enabled": True,
        # BGT registers individual trees as points. Heights come from AHN.
        "api_url": (
            "https://api.pdok.nl/lv/bgt/ogc/v1/collections/"
            "vegetatieobject_punt/items"
        ),
        "wcs_url": "https://service.pdok.nl/rws/ahn/wcs/v1_0",
        "page_limit": 1000,
        "timeout_s": 180,
        "max_retries": 4,
        "max_pages": 200,
        # A BGT point marks the trunk; the crown top is metres off it, so the
        # canopy height is the local maximum within this radius.
        "crown_search_m": 3.0,
        "min_height_m": 2.0,
        "max_height_m": 40.0,
        "default_height_m": 7.0,
        "crown_radius_ratio": 0.26,
        "trunk_height_ratio": 0.38,
        "geometry": True,
        "texture_px": 512,
    },
    "surfaces": {
        # Water is the reason this exists: lidar does not reflect off it, so
        # the DTM is mostly empty over a canal and the gap filler turns every
        # one into a bulge. BGT outlines replace the guesswork.
        "water": True,
        "land_cover": True,
        # Roads as their own geometry, one object per surface class, so they
        # carry their own material and their own layer in Unity instead of
        # being pixels in the terrain's texture.
        "road_geometry": True,
        # How far the road surface floats above the terrain, to keep the two
        # from fighting for the same depth.
        "road_lift_m": 0.06,
        # How far a flat road triangle may miss the ground under it before it
        # gets split. Earcut leaves slivers over 100 m long, and one that size
        # cuts straight through a canal bank.
        "road_drape_tolerance_m": 0.08,
        # How far the bed is sunk below the water surface.
        "water_depth_m": 1.2,
        # Grain mixed into the aerial per surface class, to counter how mushy
        # an ortho looks close up. 0 disables it.
        "detail_strength": 0.22,
        "page_limit": 1000,
        "timeout_s": 180,
        "max_retries": 4,
        "max_pages": 200,
    },
    "furniture": {
        "enabled": True,
        "page_limit": 1000,
        "timeout_s": 180,
        "max_retries": 4,
        "max_pages": 200,
        "lamp_height_m": 5.0,
        "bollard_height_m": 0.9,
        "bench_length_m": 1.8,
    },
    "vehicles": {
        # Nobody publishes where cars are parked or boats are moored, but the
        # BGT publishes the parking bays and the mooring posts, which is the
        # same information one step back.
        "cars": True,
        "boats": True,
        # Not every bay holds a car and not every pair of posts holds a boat.
        "car_occupancy": 0.72,
        "boat_occupancy": 0.8,
        "car_height_m": 1.5,
        "seed": 1807,
        "page_limit": 1000,
        "timeout_s": 180,
        "max_retries": 4,
        "max_pages": 200,
    },
    "structures": {
        # Bridges and tunnels: the parts of the ground that are not the ground.
        "bridges": True,
        "tunnels": True,
        # A deck floats this far over its measured height, so it does not fight
        # the road riding on it for the same depth.
        "deck_lift_m": 0.05,
        # Used only when the surface model has no reading over a deck, which is
        # rare. A Dutch road bridge clears what it crosses by about this much.
        "fallback_clearance_m": 5.0,
        # Nothing measures how deep a tunnel runs, and no open dataset carries
        # it, so this profile is drawn rather than surveyed: portals at ground
        # level, ramping down to this depth in between. The Maastunnel road
        # deck sits about 20 m under the Maas.
        "tunnel_depth_m": 18.0,
        "tunnel_ramp_m": 350.0,
        # Side walls up to ground level, so a tunnel reads as a cutting rather
        # than as a road floating underground. No ceiling, deliberately: a
        # roofed tunnel is invisible in the model it was added to.
        "tunnel_walls": True,
        "page_limit": 1000,
        "timeout_s": 180,
        "max_retries": 4,
        "max_pages": 200,
    },
    "rails": {
        # BGT spoor: one centreline per running track, with the function that
        # says whether it is heavy rail, a street tram or light rail.
        "enabled": True,
        # How far apart points along a track may get before the ground under
        # it stops being followed.
        "step_m": 4.0,
        "page_limit": 1000,
        "timeout_s": 180,
        "max_retries": 4,
        "max_pages": 200,
    },
    "usage": {
        # BAG building function, joined to 3DBAG on the building id.
        "enabled": True,
        "page_limit": 1000,
        "timeout_s": 180,
        "max_retries": 4,
        "max_pages": 60,
    },
    "export": {
        "fbx_name": "model.fbx",
        "aerial_name": "aerial.png",
        "metadata_name": "metadata.json",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge `override` into `base`, recursing into nested dicts."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class PipelineConfig:
    """Fully resolved configuration plus the derived geo context."""

    name: str
    bbox: BBox
    geo: GeoContext
    aerial: dict[str, Any]
    terrain: dict[str, Any]
    buildings: dict[str, Any]
    facade: dict[str, Any]
    trees: dict[str, Any]
    surfaces: dict[str, Any]
    furniture: dict[str, Any]
    vehicles: dict[str, Any]
    rails: dict[str, Any]
    structures: dict[str, Any]
    usage: dict[str, Any]
    export: dict[str, Any]
    source_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def size_m(self) -> float:
        """Nominal square side of the area, for metadata."""
        return 0.5 * (self.bbox.width + self.bbox.height)

    def as_dict(self) -> dict[str, Any]:
        """The resolved settings as plain nested dicts.

        The source registry addresses config by key path, so it needs the
        sections back in the shape they had in the file.
        """
        return {
            "name": self.name,
            "aerial": self.aerial,
            "terrain": self.terrain,
            "buildings": self.buildings,
            "facade": self.facade,
            "trees": self.trees,
            "surfaces": self.surfaces,
            "furniture": self.furniture,
            "vehicles": self.vehicles,
            "rails": self.rails,
            "structures": self.structures,
            "usage": self.usage,
            "export": self.export,
        }

    def work_dir(self, root: Path) -> Path:
        return root / "work" / self.name

    def out_dir(self, root: Path) -> Path:
        return root / "output" / self.name


def load_config(path: str | Path) -> PipelineConfig:
    """Read a config file, apply defaults, and validate the bbox."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if "bbox" not in raw:
        raise ValueError(f"{path} has no 'bbox' section")

    merged = _deep_merge(DEFAULTS, raw)
    bbox = parse_bbox(raw["bbox"])
    warnings = validate_bbox(bbox)
    for warning in warnings:
        LOG.warning("%s", warning)

    name = str(merged["name"]).strip()
    if not name:
        raise ValueError("config 'name' must not be empty")

    ahn_model = str(merged["terrain"]["ahn_model"]).strip().upper()
    if ahn_model not in ("DTM", "DSM"):
        raise ValueError(
            f"terrain.ahn_model must be DTM or DSM, got {ahn_model!r}"
        )
    if ahn_model == "DSM":
        LOG.warning(
            "terrain.ahn_model is DSM: the surface model still contains "
            "buildings and vegetation, so 3DBAG buildings will sit on top of "
            "their own roofs. DTM is what this pipeline expects."
        )
    merged["terrain"]["ahn_model"] = ahn_model

    lod = str(merged["buildings"]["lod"]).strip()
    if lod != "2.2":
        LOG.warning(
            "buildings.lod is %r; only 2.2 carries real roof shapes, lower "
            "LoDs are flat extrusions",
            lod,
        )
    merged["buildings"]["lod"] = lod

    n = int(merged["terrain"]["mesh_vertices_per_side"])
    if n < 2:
        raise ValueError(
            f"terrain.mesh_vertices_per_side must be at least 2, got {n}"
        )

    tolerance = float(merged["terrain"].get("simplify_tolerance_m", 0.0) or 0.0)
    if tolerance < 0:
        raise ValueError(
            f"terrain.simplify_tolerance_m cannot be negative, got {tolerance}"
        )
    merged["terrain"]["simplify_tolerance_m"] = tolerance
    if tolerance > 0 and not is_grid_size(n):
        # The bisection hierarchy needs every hypotenuse midpoint to land on a
        # grid point, which only holds for 2**k + 1. Rounding up rather than
        # down because the extra rows are source detail for the simplifier to
        # choose from, and it discards whatever it does not need anyway.
        snapped = next_grid_size(n)
        LOG.info(
            "terrain.mesh_vertices_per_side %d -> %d, the next size the "
            "adaptive mesh can subdivide",
            n,
            snapped,
        )
        n = snapped
    merged["terrain"]["mesh_vertices_per_side"] = n

    # Both dials have a ceiling set by the source data. Past it you are
    # interpolating, not resolving, so the run just costs more.
    native_terrain = int(round(bbox.width / float(merged["terrain"]["resolution_m"]))) + 1
    if n > native_terrain:
        LOG.warning(
            "terrain.mesh_vertices_per_side is %d, finer than the %.2f m AHN "
            "source supports over this bbox (native is about %d); the extra "
            "vertices are interpolated",
            n,
            float(merged["terrain"]["resolution_m"]),
            native_terrain,
        )

    size_px = int(merged["aerial"]["size_px"])
    if size_px < 16:
        raise ValueError(f"aerial.size_px must be at least 16, got {size_px}")
    merged["aerial"]["size_px"] = size_px

    native_aerial = int(round(bbox.width / AERIAL_SOURCE_M_PER_PX))
    if size_px > native_aerial:
        LOG.warning(
            "aerial.size_px is %d, finer than the %.0f cm source supports over "
            "this bbox (native is about %d px); the image is upsampled",
            size_px,
            AERIAL_SOURCE_M_PER_PX * 100,
            native_aerial,
        )

    variants = int(merged["facade"]["variants"])
    if variants < 1:
        raise ValueError(f"facade.variants must be at least 1, got {variants}")
    merged["facade"]["variants"] = variants

    merge_mode = str(merged["buildings"]["merge"]).strip().lower()
    if merge_mode not in ("single", "per_building"):
        raise ValueError(
            f"buildings.merge must be 'single' or 'per_building', got {merge_mode!r}"
        )
    merged["buildings"]["merge"] = merge_mode

    clip_mode = str(merged["buildings"]["clip_mode"]).strip().lower()
    if clip_mode not in ("centroid", "intersect"):
        raise ValueError(
            f"buildings.clip_mode must be 'centroid' or 'intersect', got {clip_mode!r}"
        )
    merged["buildings"]["clip_mode"] = clip_mode

    return PipelineConfig(
        name=name,
        bbox=bbox,
        geo=GeoContext.from_bbox(bbox),
        aerial=merged["aerial"],
        terrain=merged["terrain"],
        buildings=merged["buildings"],
        facade=merged["facade"],
        trees=merged["trees"],
        surfaces=merged["surfaces"],
        furniture=merged["furniture"],
        vehicles=merged["vehicles"],
        rails=merged["rails"],
        structures=merged["structures"],
        usage=merged["usage"],
        export=merged["export"],
        source_path=path,
        warnings=warnings,
    )


__all__ = ["DEFAULTS", "PipelineConfig", "load_config"]
