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

from .facade import MAX_LOOKS_PER_ERA
from .geo import BBox, GeoContext, parse_bbox, validate_bbox

LOG = logging.getLogger(__name__)

# The aerial layer is an 8 cm ortho, so a 1 km area resolves fully at about
# 12500 px. Above that the pipeline is upsampling a JPEG, not gaining detail.
AERIAL_SOURCE_M_PER_PX = 0.08

DEFAULTS: dict[str, Any] = {
    "name": "demo_area",
    # How the ground is built. "layered" is the way the modellers build it by
    # hand: a coarse terrain with roads, walkways and grass as separate flat
    # objects on top and the curb generator on every walkway (see
    # src/layered.py). "survey" is the dense terrain that follows the survey
    # to the centimetre, with every street its own quad-strip object.
    "style": "layered",
    "layered": {
        # What every ground object's name starts with; empty takes the area
        # name ("hazeldonk" gives Hazeldonk_Terrein, Hazeldonk_Snelweg).
        "prefix": "",
        # How far any surveyed point may be from the terrain. The hand-made
        # Hazeldonk terrain is within 0.98 m on 95% of the ground; 0.5 m
        # everywhere costs about 27k faces over a square kilometre.
        "tolerance_m": 0.5,
        # How far a simplified outline may stray from the BGT's.
        "outline_m": 0.25,
        # The middle of the area, grouped into an east and a west object per
        # layer; outside it, one per quadrant. The modellers' VP_Oost_Wegen
        # and Wegen_Afstand_NO.
        "focus_m": 250.0,
        # How far each layer is lifted off the terrain, as the modellers'
        # shrinkwrap offsets do: enough to keep it from flickering.
        "road_lift_m": 0.05,
        "grass_lift_m": 0.02,
        # How the terrain is laid out: "quads", rows along the outlines and a
        # grid that coarsens away from the middle, or "triangles", the
        # triangulation refined until it is within tolerance_m everywhere.
        "mesher": "quads",
        # How far below the water level a ditch is cut. No water surface is
        # drawn in this style: this is the dip in the ground.
        "ditch_depth_m": 0.6,
        # The curb generator on every walkway and traffic island. Needs
        # Blender 5; with an older one the walkways stay flat.
        "kerbs": True,
        # The modellers' own settings for it.
        "curb": {
            "width_m": 0.3,
            "height_m": 0.15,
            "height_offset_curb_m": 0.02,
            "slant": 0.6,
            "bevel_m": 0.02,
            "bevel_inner_m": 0.015,
            "walkway_height_offset_m": 0.05,
            "bevel_curb_outer": True,
            "bevel_curb_inner": True,
            "smooth_angle_deg": 46.0,
            # Empty keeps each walkway's own material (Beton_Geel, or Gras on
            # a green island); a name puts that material on every walkway.
            "walkway_material": "",
            "curb_material": "Beton",
        },
    },
    "aerial": {
        "layer": "Luchtfoto Actueel Ortho 8cm RGB",
        "size_px": 4096,
        # How much of the flown sun to take back out, 0 to 1. The ortho is
        # shot on one morning in late winter, so every roof and every square
        # metre of ground carries a hard shadow at one sun angle -- measured
        # over Bronbeek, about 20 m long at 23 degrees elevation, across 29%
        # of the photo. An engine lighting the scene at any other hour then
        # has two suns disagreeing. 0 leaves the photo as flown.
        "deshadow": 1.0,
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
        # Spacing of the terrain grid, in metres. A regular grid of quads,
        # because it is edited: sculpting, loop selection and subdivision all
        # expect rows and columns. Measured against the 0.5 m survey over
        # Bronbeek: 2 m is within 11 cm on 95% of the ground at 251k vertices;
        # 4 m within 22 cm at 63k; 1 m within 6 cm at 1M.
        "grid_m": 2.0,
        # Outlines the terrain's edges follow: the waterline, the road edge,
        # the building outline, each with a row of quads along it and the grid
        # inside. Empty gives the plain grid.
        "feature_edges": ["water", "roads", "buildings"],
        "wcs_url": "https://service.pdok.nl/rws/ahn/wcs/v1_0",
        "timeout_s": 300,
        "max_retries": 4,
        # DTM strips buildings out, so dense city centres come back mostly
        # nodata. Filling is a normal part of the job, not an error path.
        "max_nodata_fraction": 0.95,
        # Passes of smoothing over the ground the filler invented under
        # buildings and water. Measured ground is never smoothed.
        "smooth_iterations": 1,
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
        # The most looks any one era may get. Looks are handed out per era in
        # proportion to how many buildings that era actually holds, so this is
        # a ceiling rather than a count: an area with sixteen pre-1920 houses
        # spends one texture on them however high this goes. Measured over
        # Bronbeek -- 1322 buildings, 73% of them interbellum -- a flat 12
        # would cost 62 wall textures where the allocation costs 20.
        # 1 switches the whole thing off and gives one wall per era.
        "looks_per_era": 12,
        # How many buildings of one era should share a texture. The lever that
        # actually decides how repetitive a street looks; lower buys variety
        # with textures. At Bronbeek 80 puts 81 houses on the busiest wall,
        # against 967 before any of this.
        "buildings_per_look": 80,
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
        # How far the aerial is darkened under a canopy, 0 to 1. Dutch orthos
        # are flown leaf-off on purpose, so the ground under a tree is
        # photographed in full sun and the leafy tree the model then stands
        # over it reads as not touching the ground. Ambient occlusion, not a
        # sun shadow: it is correct at any time of day and adds to a
        # real-time shadow instead of fighting it. 0 leaves the photo alone.
        "ground_shade": 0.35,
        # The BGT tree register is a municipal asset list, not a survey of
        # vegetation: over one square kilometre of Utrecht centre 45% of the
        # canopy above 2.5 m is nowhere near a registered tree, and the
        # register is at its best in a city centre. The rest is found in the
        # canopy height model. Detected trees stay flagged, and arrive as their
        # own object, because the detector is good rather than perfect.
        "detect": True,
        # Below this a crown is a shrub, and shrubs are not worth a tree mesh.
        "detect_min_height_m": 3.0,
        # And above this it is a structure, not a tree. Urban trees top out
        # around thirty metres; the forty-metre "trees" the first run found
        # were one tower the building footprints do not cover.
        "detect_max_height_m": 30.0,
        # Extra clearance from a building per metre of height. Measured: 65% of
        # detections above 25 m sat within 5 m of a building against 32% of
        # those under 10 m, so height near a roof is the tell.
        "detect_clearance_per_metre": 0.15,
        # How rough the surface has to be. Once the building footprints are
        # masked this is what rejects scaffolding, roof plant and lorries:
        # roughness at a registered tree runs to a median of 3.1 m against
        # 0.67 m on a roof.
        "detect_min_roughness_m": 0.4,
        # A roof overhangs its own footprint, and without this clearance the
        # eaves read as a line of trees along every terrace.
        "detect_eaves_clearance_m": 2.0,
        # No two detected trees closer than this, whatever their crowns say.
        "detect_min_spacing_m": 3.0,
        # Surfaces a found tree is not allowed to stand on. A tall rough thing
        # on a carriageway is a lorry, a bus or scaffolding far more often than
        # a tree: 9.5% of everything found over a square kilometre of
        # Gelderland was standing on one. Avenues exist and this costs the odd
        # real one, which is a good trade at ten to one -- and it is only
        # applied to what the detector found, never to a surveyed trunk.
        #
        # Footpaths included, on second thoughts. They were left out to protect
        # the tree in a pit on a pedestrianised street -- but a surveyed tree is
        # never filtered by this at all, so the exclusion protected nothing and
        # cost a great deal: footpath is 305 of the 871 road parts over the
        # Gelderland square, and in a park every winding path through the lawns
        # is one. Trees were standing on them.
        "detect_off_surfaces": [
            "road_asphalt", "road_brick", "cycle_path", "parking",
            "transit_lane", "footpath",
        ],
        "geometry": True,
        "texture_px": 512,
    },
    "surfaces": {
        # Water is the reason this exists: lidar does not reflect off it, so
        # the DTM is mostly empty over a canal and the gap filler turns every
        # one into a bulge. BGT outlines replace the guesswork.
        "water": True,
        "land_cover": True,
        # Roads as their own geometry: quads across every street, cut along
        # the national road network (NWB), one object per street.
        "road_geometry": True,
        "nwb_url": (
            "https://api.pdok.nl/rws/nationaal-wegenbestand-wegen/ogc/v1/"
            "collections/wegvakken/items"
        ),
        # How far the road surface floats above the surveyed ground. The
        # terrain is pushed down under it as well, so this only has to keep
        # the two apart, not clear the terrain's error.
        "road_lift_m": 0.03,
        # Height of the kerb between a footway and the carriageway beside it.
        "kerb_m": 0.10,
        # How far above a water surface a path crossing it is held. The
        # bare-earth grid has a hole over every body and the filler dips across
        # it -- 1.45 m below its own bank over the Gelderland square -- so a
        # footpath draped on that follows it under the water.
        "road_water_clearance_m": 0.25,
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
        # The BGT gives a position and a type but no size, so the proxies need
        # a height from somewhere. These two are the only ones tall enough to
        # be worth arguing about.
        "flagpole_height_m": 7.0,
        "catenary_height_m": 7.5,
    },
    "barriers": {
        # Walls, fences, hedges and the small structures bolted onto buildings.
        # The largest thing the BGT publishes that the model used to ignore:
        # over one square kilometre of Utrecht centre, 1563 separations drawn
        # as lines and 1450 more drawn as polygons.
        "enabled": True,
        "page_limit": 1000,
        "timeout_s": 180,
        "max_retries": 4,
        "max_pages": 200,
        # How finely a swept line is cut before the height profile is thinned.
        "step_m": 1.0,
        # How far a barrier may stray from the ground between vertices. Cutting
        # at a fixed step instead more than doubled the geometry to hold
        # straight lines straight.
        "drape_tolerance_m": 0.05,
        # The BGT surveys where a wall is, never how tall. These are the
        # assumption, in one place, for someone to argue with. Keys are the
        # names in barriers.LINE_STYLES and AREA_STYLES.
        "heights": {},
        # How long one tile of the fence texture is. The uprights keep this
        # spacing however long the run, which is the point of tiling it.
        "fence_tile_m": 2.5,
        # The thinnest a physics proxy may be, which is not the thinnest a
        # fence may look. Only used when export.collision is on; see
        # barriers.DEFAULT_COLLISION_THICKNESS_M for why 30 cm.
        "collision_thickness_m": 0.30,
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
        # The same scene for editing: collections per layer, quads and
        # n-gons kept, the Maptile tools inside.
        "blend_name": "model.blend",
        "aerial_name": "aerial.png",
        "metadata_name": "metadata.json",
        # Collision_* objects alongside the visible ones, for a simulation
        # rather than a picture. Off by default because they cost FBX size and
        # most imports do not want them.
        "collision": False,
        # Whether a tree's collider includes its canopy. True encloses the
        # crown, which is what stops a drone flying through foliage; false
        # leaves the trunk alone at a quarter of the triangles, for a sim that
        # treats leaves as soft.
        "collision_tree_crowns": True,
    },
}


# One choice in place of a page of sliders. Each level sets how sharp the
# aerial is (as ground resolution, so it means the same over any area), how
# closely spaced the terrain grid is, and how large the wall images are. A
# config file names one with a top-level "detail"; anything it also sets
# explicitly wins. The UI shows the same table (ui/index.html, DETAIL), and a
# test keeps the two equal.
DETAIL_PRESETS: dict[str, dict] = {
    "draft": {"aerial_cm": 50, "grid_m": 4.0, "facade_px": 256,
              "buildings_per_look": 160, "normal_map": False},
    "standard": {"aerial_cm": 25, "grid_m": 2.0, "facade_px": 512,
                 "buildings_per_look": 80, "normal_map": True},
    "high": {"aerial_cm": 12, "grid_m": 1.0, "facade_px": 1024,
             "buildings_per_look": 40, "normal_map": True},
}


# What each style changes in the other sections, applied before the file's
# own settings so anything set explicitly still wins. The modellers' ground
# has no street-by-street road objects, and their file has no parked cars.
# Water is fetched, but as ditches and canals cut into the ground, not as a
# surface of its own: see `layered`.
STYLE_PRESETS: dict[str, dict] = {
    "layered": {
        "surfaces": {"road_geometry": False},
        "terrain": {"feature_edges": []},
        "vehicles": {"cars": False, "boats": False},
    },
    "survey": {},
}

# What a style cannot do at all, whatever the file says. The layered ground
# makes its own roads, so the survey style's would lie on top of them.
STYLE_LOCKED: dict[str, dict[tuple[str, str], object]] = {
    "layered": {("surfaces", "road_geometry"): False},
    "survey": {},
}


# Settings that no longer do anything, and what took their place. A config
# that still carries one loads, and says so, rather than failing or being
# silently ignored.
RETIRED: dict[str, str] = {
    "terrain.mesh_vertices_per_side": "the terrain is a regular grid; set terrain.grid_m",
    "terrain.simplify_tolerance_m": "the terrain is a regular grid; set terrain.grid_m",
    "terrain.breaklines": "the terrain is a regular grid with the roads laid on top",
    "terrain.breakline_simplify_m": "the terrain is a regular grid with the roads laid on top",
    "terrain.contour_interval_m": "the terrain is a regular grid with the roads laid on top",
    "terrain.min_feature_length_m": "the terrain is a regular grid with the roads laid on top",
    "terrain.max_face_m": "the terrain is a regular grid; set terrain.grid_m",
    "terrain.quads": "the terrain is quads already",
    "terrain.quad_max_fold_deg": "the terrain is quads already",
    "terrain.quad_min_angle_deg": "the terrain is quads already",
    "surfaces.road_drape_tolerance_m": "roads are cut across along their centrelines",
    "surfaces.road_simplify_m": "roads are cut across along their centrelines",
    "surfaces.road_max_edge_m": "roads are cut across along their centrelines",
    "surfaces.road_min_angle_deg": "roads are cut across along their centrelines",
    "surfaces.road_snap_m": "roads are cut across along their centrelines",
    "buildings.merge": "every building is its own object, under the Buildings parent",
}


def _retired_keys(raw: dict) -> list[str]:
    """Warnings for each retired setting a config file still sets."""
    out = []
    for path, instead in RETIRED.items():
        section, key = path.split(".", 1)
        if isinstance(raw.get(section), dict) and key in raw[section]:
            out.append(f"{path} no longer does anything: {instead}")
    return out


def detail_settings(detail: str, side_m: float) -> dict:
    """The config sections a detail level stands for, over an area this wide."""
    try:
        preset = DETAIL_PRESETS[str(detail).strip().lower()]
    except KeyError:
        raise ValueError(
            f"detail must be one of {', '.join(DETAIL_PRESETS)}, got {detail!r}"
        ) from None
    # A multiple of 4 px: the mosaic splits the photo into tiles and whole
    # numbers keep every tile boundary on an exact pixel.
    size_px = max(256, int(round(side_m / (preset["aerial_cm"] / 100.0) / 4.0)) * 4)
    return {
        "aerial": {"size_px": size_px},
        "terrain": {"grid_m": float(preset["grid_m"])},
        "facade": {
            "texture_px": int(preset["facade_px"]),
            "buildings_per_look": int(preset["buildings_per_look"]),
            "normal_map": bool(preset["normal_map"]),
        },
    }


def default_prefix(name: str) -> str:
    """Object-name prefix from an area name: "hazeldonk_west" -> "Hazeldonk_West"."""
    import re
    import unicodedata

    # Accents dropped rather than the letters: "Café" -> "Cafe".
    plain = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    words = [w for w in re.split(r"[\s_\-]+", plain) if w]
    cleaned = "_".join(w[:1].upper() + w[1:] for w in words)
    # Blender and Unity both take anything, but a file system path does not.
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", cleaned)
    return cleaned or "Area"


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
    style: str
    layered: dict[str, Any]
    aerial: dict[str, Any]
    terrain: dict[str, Any]
    buildings: dict[str, Any]
    facade: dict[str, Any]
    trees: dict[str, Any]
    surfaces: dict[str, Any]
    furniture: dict[str, Any]
    barriers: dict[str, Any]
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
            "style": self.style,
            "layered": self.layered,
            "aerial": self.aerial,
            "terrain": self.terrain,
            "buildings": self.buildings,
            "facade": self.facade,
            "trees": self.trees,
            "surfaces": self.surfaces,
            "furniture": self.furniture,
            "barriers": self.barriers,
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

    bbox = parse_bbox(raw["bbox"])
    retired = _retired_keys(raw)
    for warning in retired:
        LOG.warning("%s", warning)
    base = DEFAULTS
    if raw.get("detail") is not None:
        # The level first, then whatever the file sets explicitly on top.
        base = _deep_merge(
            DEFAULTS, detail_settings(raw["detail"], max(bbox.width, bbox.height))
        )
    style = str(raw.get("style", DEFAULTS["style"]) or "").strip().lower()
    if style not in STYLE_PRESETS:
        raise ValueError(
            f"style must be one of {', '.join(STYLE_PRESETS)}, got {raw.get('style')!r}"
        )
    base = _deep_merge(base, STYLE_PRESETS[style])
    merged = _deep_merge(base, {k: v for k, v in raw.items() if k != "detail"})
    merged["style"] = style
    for (section, key), value in STYLE_LOCKED[style].items():
        if merged.get(section, {}).get(key, value) != value:
            LOG.warning(
                "%s.%s cannot be %r in the %s style; using %r",
                section, key, merged[section][key], style, value,
            )
        merged.setdefault(section, {})[key] = value
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

    grid_m = float(merged["terrain"].get("grid_m", 2.0))
    if not grid_m > 0:
        raise ValueError(f"terrain.grid_m must be positive, got {grid_m}")
    merged["terrain"]["grid_m"] = grid_m

    edges = merged["terrain"].get("feature_edges")
    if edges is None or edges is False:
        edges = []
    elif edges is True:
        edges = ["water", "roads", "buildings"]
    elif isinstance(edges, str):
        edges = [edges]
    known = ("water", "roads", "buildings")
    cleaned = []
    for source in edges:
        source = str(source).strip().lower()
        if source not in known:
            raise ValueError(
                f"terrain.feature_edges names {source!r}; it can follow "
                f"{', '.join(known)}"
            )
        if source not in cleaned:
            cleaned.append(source)
    merged["terrain"]["feature_edges"] = cleaned

    # Both dials have a ceiling set by the source data. Past it you are
    # interpolating, not resolving, so the run just costs more.
    if grid_m < float(merged["terrain"]["resolution_m"]):
        LOG.warning(
            "terrain.grid_m is %.2f m, finer than the %.2f m AHN source; the "
            "extra vertices are interpolated",
            grid_m,
            float(merged["terrain"]["resolution_m"]),
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

    looks = int(merged["facade"]["looks_per_era"])
    if looks < 1:
        raise ValueError(
            f"facade.looks_per_era must be at least 1, got {looks}"
        )
    if looks > MAX_LOOKS_PER_ERA:
        # Clamped rather than refused: the extra textures would be duplicates
        # of earlier ones, so the only cost of asking for them is memory.
        LOG.warning(
            "facade.looks_per_era %d is above the %d the palettes can fill; "
            "using %d",
            looks,
            MAX_LOOKS_PER_ERA,
            MAX_LOOKS_PER_ERA,
        )
        looks = MAX_LOOKS_PER_ERA
    merged["facade"]["looks_per_era"] = looks

    clip_mode = str(merged["buildings"]["clip_mode"]).strip().lower()
    if clip_mode not in ("centroid", "intersect"):
        raise ValueError(
            f"buildings.clip_mode must be 'centroid' or 'intersect', got {clip_mode!r}"
        )
    merged["buildings"]["clip_mode"] = clip_mode

    layered = merged["layered"]
    for key in ("tolerance_m", "focus_m"):
        value = float(layered[key])
        if not value > 0:
            raise ValueError(f"layered.{key} must be positive, got {value}")
        layered[key] = value
    mesher = str(layered.get("mesher", "quads")).strip().lower()
    if mesher not in ("quads", "triangles"):
        raise ValueError(f"layered.mesher must be 'quads' or 'triangles', got {mesher!r}")
    layered["mesher"] = mesher
    for key in ("outline_m", "road_lift_m", "grass_lift_m", "ditch_depth_m"):
        value = float(layered[key])
        if value < 0:
            raise ValueError(f"layered.{key} must not be negative, got {value}")
        layered[key] = value
    prefix = str(layered.get("prefix") or "").strip()
    layered["prefix"] = prefix or default_prefix(name)
    curb = layered["curb"]
    if not 0 < float(curb["height_offset_curb_m"]) < float(curb["height_m"]):
        raise ValueError(
            "layered.curb.height_offset_curb_m must be between 0 and the kerb height"
        )

    return PipelineConfig(
        name=name,
        bbox=bbox,
        geo=GeoContext.from_bbox(bbox),
        style=style,
        layered=layered,
        aerial=merged["aerial"],
        terrain=merged["terrain"],
        buildings=merged["buildings"],
        facade=merged["facade"],
        trees=merged["trees"],
        surfaces=merged["surfaces"],
        furniture=merged["furniture"],
        barriers=merged["barriers"],
        vehicles=merged["vehicles"],
        rails=merged["rails"],
        structures=merged["structures"],
        usage=merged["usage"],
        export=merged["export"],
        source_path=path,
        warnings=warnings + retired,
    )


__all__ = [
    "DEFAULTS",
    "DETAIL_PRESETS",
    "PipelineConfig",
    "RETIRED",
    "STYLE_LOCKED",
    "STYLE_PRESETS",
    "default_prefix",
    "detail_settings",
    "load_config",
]
