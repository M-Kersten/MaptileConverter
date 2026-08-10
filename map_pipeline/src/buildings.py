"""3DBAG buildings: paginated CityJSON -> semantically split triangle soup.

Three properties of the 3DBAG API shape this module.

The API answers with a FeatureCollection of CityJSONFeature objects, and each
page carries its own ``transform``. Vertices are integer-quantised against that
per-page transform, so pages have to be decoded before they are merged. Merging
the raw pages first and applying one transform afterwards silently warps every
page but the first.

``numberMatched`` and ``numberReturned`` count CityObjects, not features, and a
building contributes one Building plus one BuildingPart per part. Paging is
therefore driven off the ``next`` links rather than off those counts.

LoD2.2 geometry lives on the BuildingPart children, while the useful attributes
(ground level, roof height, storey count) live on the Building parent, so the
two have to be read together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import requests

from .geo import BBox
from .http_util import ServiceError, get_with_retry

LOG = logging.getLogger(__name__)

WALL = "WallSurface"
ROOF = "RoofSurface"
GROUND = "GroundSurface"
# 3DBAG also emits OuterFloorSurface for overhanging parts. It faces downward
# and is treated as roof-like so it picks up aerial texture rather than facade.
OUTER_FLOOR = "OuterFloorSurface"

DEFAULT_FLOOR_HEIGHT_M = 3.0


@dataclass
class Building:
    """One 3DBAG building, ready for meshing."""

    identifier: str
    wall_tris: np.ndarray  # (n, 3, 3) float64, RD x / RD y / NAP z
    roof_tris: np.ndarray  # (m, 3, 3) float64
    ground_z_nap: float
    roof_max_nap: float
    floors: int
    # Top of the walls. On a pitched roof this is the eaves, well below the
    # ridge, and it is what the facade has to divide into storeys.
    wall_top_nap: float = 0.0
    build_year: int | None = None
    # Filled in when the ground storey is split off; carries its own material.
    ground_wall_tris: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3, 3), dtype=np.float64)
    )

    @property
    def height_m(self) -> float:
        return self.roof_max_nap - self.ground_z_nap

    @property
    def wall_height_m(self) -> float:
        return max(self.wall_top_nap - self.ground_z_nap, 0.0)

    @property
    def all_wall_tris(self) -> np.ndarray:
        """Every wall triangle, whichever side of the ground-floor cut it fell.

        After the split the base of the building lives in ``ground_wall_tris``,
        so anything reasoning about where the building meets the ground has to
        look at both.
        """
        chunks = [c for c in (self.wall_tris, self.ground_wall_tris) if len(c)]
        if not chunks:
            return np.zeros((0, 3, 3), dtype=np.float64)
        return np.concatenate(chunks, axis=0)

    @property
    def n_triangles(self) -> int:
        return int(len(self.wall_tris) + len(self.roof_tris))


@dataclass
class BuildingSet:
    """Every building in the area plus the counters used for validation."""

    buildings: list[Building] = field(default_factory=list)
    skipped_no_geometry: int = 0
    skipped_degenerate: int = 0
    degenerate_surfaces: int = 0
    triangulation_failures: int = 0
    pages_fetched: int = 0
    lod: str = "2.2"

    def __len__(self) -> int:
        return len(self.buildings)

    def stats(self) -> dict:
        heights = [b.height_m for b in self.buildings]
        return {
            "count": len(self.buildings),
            "wall_triangles": int(sum(len(b.wall_tris) for b in self.buildings)),
            "roof_triangles": int(sum(len(b.roof_tris) for b in self.buildings)),
            "height_min_m": round(float(min(heights)), 2) if heights else 0.0,
            "height_max_m": round(float(max(heights)), 2) if heights else 0.0,
            "height_mean_m": round(float(np.mean(heights)), 2) if heights else 0.0,
            "pages_fetched": self.pages_fetched,
            "skipped_no_geometry": self.skipped_no_geometry,
            "skipped_degenerate": self.skipped_degenerate,
            "degenerate_source_surfaces": self.degenerate_surfaces,
            "triangulation_failures": self.triangulation_failures,
        }


# --------------------------------------------------------------------------
# API access
# --------------------------------------------------------------------------


def iter_api_pages(
    bbox: BBox,
    *,
    api_url: str,
    page_limit: int = 500,
    timeout: float = 180.0,
    max_retries: int = 4,
    max_pages: int = 2000,
) -> Iterator[dict]:
    """Yield each page of the 3DBAG item response for `bbox`.

    Paging follows the ``next`` links. Repeating URLs are treated as the end of
    the run so a server-side paging quirk cannot spin forever.
    """
    session = requests.Session()
    url: str | None = api_url
    params: dict[str, Any] | None = {
        "bbox": f"{bbox.xmin:.3f},{bbox.ymin:.3f},{bbox.xmax:.3f},{bbox.ymax:.3f}",
        "limit": page_limit,
    }
    seen_urls: set[str] = set()

    for page_index in range(max_pages):
        response = get_with_retry(
            url,
            params=params,
            timeout=timeout,
            max_retries=max_retries,
            session=session,
            description=f"3DBAG page {page_index}",
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ServiceError(
                f"3DBAG page {page_index} was not JSON: {response.text[:300]}"
            ) from exc

        yield payload

        next_links = [
            link.get("href")
            for link in payload.get("links", [])
            if link.get("rel") == "next" and link.get("href")
        ]
        if not next_links:
            return
        next_url = next_links[0]
        if next_url in seen_urls:
            LOG.warning("3DBAG paging repeated %s; stopping", next_url)
            return
        seen_urls.add(next_url)
        url, params = next_url, None

    LOG.warning("stopped after %d pages; raise buildings.max_pages if truncated", max_pages)


def decode_vertices(vertices: Sequence[Sequence[float]], transform: dict) -> np.ndarray:
    """Undo CityJSON quantisation into real RD + NAP coordinates.

    Vertices are stored as integers; without the per-page scale and translate
    everything collapses onto a millimetre grid near the origin.
    """
    scale = np.asarray(transform["scale"], dtype=np.float64)
    translate = np.asarray(transform["translate"], dtype=np.float64)
    if not vertices:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(vertices, dtype=np.float64) * scale + translate


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _newell_normal(points: np.ndarray) -> np.ndarray:
    """Best-fit polygon normal. Handles non-planar rings, which LoD2.2 has."""
    rolled = np.roll(points, -1, axis=0)
    normal = np.array(
        [
            np.sum((points[:, 1] - rolled[:, 1]) * (points[:, 2] + rolled[:, 2])),
            np.sum((points[:, 2] - rolled[:, 2]) * (points[:, 0] + rolled[:, 0])),
            np.sum((points[:, 0] - rolled[:, 0]) * (points[:, 1] + rolled[:, 1])),
        ]
    )
    length = np.linalg.norm(normal)
    if length < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return normal / length


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal in-plane axes forming a right-handed frame with `normal`.

    Right-handedness matters: it makes a counter-clockwise ring in 3D project to
    a positively-wound polygon in 2D, which keeps the triangle winding that
    earcut produces pointing the same way as the source surface.
    """
    reference = (
        np.array([1.0, 0.0, 0.0])
        if abs(normal[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _clean_ring(points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Drop consecutive duplicate vertices, including a repeated closing point."""
    if len(points) < 2:
        return points
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > tol:
            keep.append(i)
    cleaned = points[keep]
    # CityJSON rings are implicitly closed; drop an explicit closing vertex.
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tol:
        cleaned = cleaned[:-1]
    return cleaned


def is_degenerate_surface(rings: list[np.ndarray]) -> bool:
    """True when a surface has no area to triangulate.

    3DBAG carries a small number of sliver faces whose ring collapses to two
    distinct points once duplicates are removed. They are source-data noise, not
    a triangulation problem, so they are counted separately.
    """
    if not rings:
        return True
    outer = _clean_ring(np.asarray(rings[0], dtype=np.float64))
    return len(outer) < 3


def triangulate_surface(rings: list[np.ndarray]) -> np.ndarray:
    """Triangulate one CityJSON surface into an (n, 3, 3) triangle array.

    The first ring is the outer boundary and the rest are holes. Surfaces are
    projected onto their best-fit plane before earcut runs, so non-planar roof
    faces and rings with holes both work.
    """
    import mapbox_earcut

    cleaned = [_clean_ring(np.asarray(r, dtype=np.float64)) for r in rings]
    cleaned = [r for r in cleaned if len(r) >= 3]
    if not cleaned:
        return np.zeros((0, 3, 3), dtype=np.float64)

    outer = cleaned[0]
    normal = _newell_normal(outer)
    u, v = _plane_basis(normal)
    origin = outer[0]

    flat = np.concatenate(cleaned, axis=0)
    relative = flat - origin
    projected = np.column_stack([relative @ u, relative @ v])

    ring_ends = np.cumsum([len(r) for r in cleaned]).astype(np.uint32)

    try:
        indices = mapbox_earcut.triangulate_float64(projected, ring_ends)
    except Exception as exc:  # noqa: BLE001 - one bad surface must not stop the run
        LOG.debug("earcut failed on a surface with %d rings: %s", len(cleaned), exc)
        return np.zeros((0, 3, 3), dtype=np.float64)

    if len(indices) < 3:
        return np.zeros((0, 3, 3), dtype=np.float64)

    triangles = flat[np.asarray(indices, dtype=np.int64)].reshape(-1, 3, 3)

    # Drop slivers, then align winding with the surface normal so the faces are
    # consistently outward.
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    keep = areas > 1e-9
    triangles = triangles[keep]
    cross = cross[keep]
    if len(triangles) == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)

    flipped = (cross @ normal) < 0
    triangles[flipped] = triangles[flipped][:, ::-1, :]
    return triangles


def _iter_solid_surfaces(
    boundaries: Any, semantics_values: Any
) -> Iterator[tuple[Any, Any]]:
    """Walk a Solid or MultiSurface, pairing each surface with its semantic index.

    A Solid nests one level deeper than a MultiSurface (shells of surfaces), and
    its semantics array nests to match.
    """
    if not boundaries:
        return

    # A Solid's first element is a shell: a list of surfaces, each a list of
    # rings, each a list of vertex indices. Four levels of nesting.
    def depth(node: Any) -> int:
        count = 0
        while isinstance(node, (list, tuple)) and node:
            count += 1
            node = node[0]
        return count

    nesting = depth(boundaries)

    if nesting >= 4:  # Solid (or CompositeSolid flattened one level at a time)
        for shell_index, shell in enumerate(boundaries):
            shell_semantics = (
                semantics_values[shell_index]
                if isinstance(semantics_values, (list, tuple))
                and shell_index < len(semantics_values)
                else None
            )
            for surface_index, surface in enumerate(shell):
                value = (
                    shell_semantics[surface_index]
                    if isinstance(shell_semantics, (list, tuple))
                    and surface_index < len(shell_semantics)
                    else None
                )
                yield surface, value
    else:  # MultiSurface
        for surface_index, surface in enumerate(boundaries):
            value = (
                semantics_values[surface_index]
                if isinstance(semantics_values, (list, tuple))
                and surface_index < len(semantics_values)
                else None
            )
            yield surface, value


def _semantic_type(surfaces: list[dict] | None, value: Any) -> str | None:
    if not surfaces or value is None:
        return None
    if not isinstance(value, int) or not (0 <= value < len(surfaces)):
        return None
    return surfaces[value].get("type")


def _pick_geometry(city_object: dict, lod: str) -> dict | None:
    """Find the geometry at the requested LoD, preferring a Solid."""
    candidates = [
        geometry
        for geometry in city_object.get("geometry", [])
        if str(geometry.get("lod", "")).strip() == lod
    ]
    if not candidates:
        return None
    for geometry in candidates:
        if geometry.get("type") in ("Solid", "CompositeSolid"):
            return geometry
    return candidates[0]


def _extract_feature(
    feature: dict, transform: dict, lod: str, counters: BuildingSet
) -> Building | None:
    """Turn one CityJSONFeature into a :class:`Building`."""
    city_objects = feature.get("CityObjects", {})
    if not city_objects:
        return None

    vertices = decode_vertices(feature.get("vertices", []), transform)
    if len(vertices) == 0:
        counters.skipped_no_geometry += 1
        return None

    # The Building parent holds the attributes; the BuildingPart children hold
    # the LoD geometry.
    parent_id, parent = None, None
    for object_id, obj in city_objects.items():
        if obj.get("type") == "Building":
            parent_id, parent = object_id, obj
            break
    if parent is None:
        parent_id, parent = next(iter(city_objects.items()))

    attributes = parent.get("attributes", {}) or {}

    part_ids = parent.get("children") or [
        object_id
        for object_id, obj in city_objects.items()
        if obj.get("type") == "BuildingPart"
    ]
    if not part_ids:
        part_ids = [parent_id]

    wall_chunks: list[np.ndarray] = []
    roof_chunks: list[np.ndarray] = []
    saw_geometry = False

    for part_id in part_ids:
        part = city_objects.get(part_id)
        if part is None:
            continue
        geometry = _pick_geometry(part, lod)
        if geometry is None:
            continue
        saw_geometry = True

        semantics = geometry.get("semantics") or {}
        surface_defs = semantics.get("surfaces")
        semantic_values = semantics.get("values")

        for surface, value in _iter_solid_surfaces(
            geometry.get("boundaries"), semantic_values
        ):
            surface_type = _semantic_type(surface_defs, value)

            # The building's own footprint sits under the terrain and is never
            # visible, so it is dropped rather than meshed.
            if surface_type == GROUND:
                continue

            try:
                rings = [vertices[np.asarray(ring, dtype=np.int64)] for ring in surface]
            except (IndexError, TypeError, ValueError):
                counters.triangulation_failures += 1
                continue
            if not rings:
                continue

            if is_degenerate_surface(rings):
                counters.degenerate_surfaces += 1
                continue

            triangles = triangulate_surface(rings)
            if len(triangles) == 0:
                counters.triangulation_failures += 1
                continue

            if surface_type == WALL:
                wall_chunks.append(triangles)
            elif surface_type in (ROOF, OUTER_FLOOR):
                roof_chunks.append(triangles)
            elif surface_type is None:
                # No semantics: fall back to orientation. Near-vertical faces
                # are walls, the rest take aerial texture.
                normal = _newell_normal(rings[0])
                (wall_chunks if abs(normal[2]) < 0.35 else roof_chunks).append(
                    triangles
                )

    if not saw_geometry:
        counters.skipped_no_geometry += 1
        return None

    walls = (
        np.concatenate(wall_chunks, axis=0)
        if wall_chunks
        else np.zeros((0, 3, 3), dtype=np.float64)
    )
    roofs = (
        np.concatenate(roof_chunks, axis=0)
        if roof_chunks
        else np.zeros((0, 3, 3), dtype=np.float64)
    )
    if len(walls) == 0 and len(roofs) == 0:
        counters.skipped_degenerate += 1
        return None

    all_z = np.concatenate(
        [chunk[:, :, 2].ravel() for chunk in (walls, roofs) if len(chunk)]
    )

    ground_z = attributes.get("b3_h_maaiveld")
    ground_z = float(ground_z) if ground_z is not None else float(all_z.min())

    roof_max = attributes.get("b3_h_dak_max")
    roof_max = float(roof_max) if roof_max is not None else float(all_z.max())

    # Storeys are counted against the wall, not the ridge. Using the ridge on a
    # pitched roof stretches the storey height so the top row of windows is cut
    # in half by the eaves.
    wall_top = float(walls[:, :, 2].max()) if len(walls) else roof_max
    wall_height = max(wall_top - ground_z, 0.0)

    storeys = attributes.get("b3_bouwlagen")
    floors = 0
    if storeys is not None and int(storeys) > 0:
        # Trust the BAG storey count only when it implies a believable storey
        # height; it is occasionally set for the whole block rather than a part.
        implied = wall_height / int(storeys)
        if 2.2 <= implied <= 5.0:
            floors = int(storeys)
    if floors == 0:
        floors = max(1, int(round(wall_height / DEFAULT_FLOOR_HEIGHT_M)))

    build_year = attributes.get("oorspronkelijkbouwjaar")
    try:
        build_year = int(build_year) if build_year else None
    except (TypeError, ValueError):
        build_year = None

    return Building(
        identifier=str(parent_id),
        wall_tris=walls,
        roof_tris=roofs,
        ground_z_nap=ground_z,
        roof_max_nap=roof_max,
        floors=floors,
        wall_top_nap=wall_top,
        build_year=build_year,
    )


def fetch_buildings(bbox: BBox, *, buildings_cfg: dict) -> BuildingSet:
    """Fetch and parse every 3DBAG building intersecting `bbox`."""
    lod = str(buildings_cfg["lod"])
    result = BuildingSet(lod=lod)
    seen: set[str] = set()

    LOG.info("querying 3DBAG for %s at LoD %s", bbox, lod)
    for payload in iter_api_pages(
        bbox,
        api_url=str(buildings_cfg["api_url"]),
        page_limit=int(buildings_cfg["page_limit"]),
        timeout=float(buildings_cfg["timeout_s"]),
        max_retries=int(buildings_cfg["max_retries"]),
        max_pages=int(buildings_cfg["max_pages"]),
    ):
        result.pages_fetched += 1

        # Each page carries its own quantisation grid, so it must be decoded
        # with its own transform before anything is merged.
        transform = (payload.get("metadata") or {}).get("transform")
        if not transform:
            raise ServiceError(
                "3DBAG page has no metadata.transform; vertices cannot be decoded"
            )

        for feature in payload.get("features", []):
            feature_id = feature.get("id")
            if feature_id in seen:
                continue
            if feature_id:
                seen.add(feature_id)

            building = _extract_feature(feature, transform, lod, result)
            if building is not None:
                result.buildings.append(building)

        if result.pages_fetched % 5 == 0:
            LOG.info(
                "  %d pages, %d buildings so far", result.pages_fetched, len(result)
            )

    LOG.info(
        "3DBAG returned %d buildings over %d pages",
        len(result),
        result.pages_fetched,
    )
    if not result.buildings:
        raise ServiceError(
            f"3DBAG returned no usable buildings for {bbox} at LoD {lod}; "
            f"check the bbox is over built-up land and in RD New"
        )
    return result


def _footprint_center(building: Building) -> tuple[float, float]:
    chunks = [c for c in (building.wall_tris, building.roof_tris) if len(c)]
    points = np.concatenate([c.reshape(-1, 3) for c in chunks])
    return (
        0.5 * (float(points[:, 0].min()) + float(points[:, 0].max())),
        0.5 * (float(points[:, 1].min()) + float(points[:, 1].max())),
    )


def filter_to_bbox(
    buildings: BuildingSet, bbox: BBox, mode: str
) -> tuple[BuildingSet, str]:
    """Decide which of the returned buildings belong to this area.

    The API returns every building that *intersects* the bbox, and it returns
    them whole. Around Utrecht Centraal that pulls in the Jaarbeurs halls, a
    single 296 x 462 m object whose centroid sits 300 m outside the box, which
    on its own stretches the model half a kilometre past the terrain.

    ``centroid`` is the usual tiling rule: a building belongs to the tile
    containing its centre, so each one appears exactly once across adjacent
    tiles and geometry is never cut open. ``intersect`` keeps everything the API
    returned, which gives full coverage at the cost of an unbounded overhang.
    """
    if mode == "intersect":
        return buildings, f"kept all {len(buildings)} buildings that touch the bbox"

    kept = []
    for building in buildings.buildings:
        cx, cy = _footprint_center(building)
        if bbox.xmin <= cx <= bbox.xmax and bbox.ymin <= cy <= bbox.ymax:
            kept.append(building)

    dropped = len(buildings) - len(kept)
    buildings.buildings = kept
    return buildings, (
        f"kept {len(kept)} buildings whose centre is inside the bbox, "
        f"dropped {dropped} that only clip its edge"
    )


def filter_implausible(
    buildings: BuildingSet, *, min_height_m: float, max_height_m: float
) -> tuple[BuildingSet, list[str]]:
    """Drop buildings whose height is outside the plausible range."""
    kept: list[Building] = []
    notes: list[str] = []
    too_low, too_high = 0, 0

    for building in buildings.buildings:
        height = building.height_m
        if height < min_height_m:
            too_low += 1
        elif height > max_height_m:
            too_high += 1
        else:
            kept.append(building)

    if too_low:
        notes.append(f"dropped {too_low} buildings under {min_height_m:.1f} m tall")
    if too_high:
        notes.append(f"dropped {too_high} buildings over {max_height_m:.1f} m tall")

    buildings.buildings = kept
    return buildings, notes


def apply_ground_skirt(
    buildings: BuildingSet,
    terrain_sampler: Callable[[Any, Any], Any],
    *,
    skirt_m: float = 0.5,
) -> int:
    """Push each building's wall base below the terrain under its footprint.

    3DBAG's ground level and the AHN DTM come from the same source data, but the
    DTM has holes wherever a building stands and those holes are filled by
    interpolation. The reconstructed ground can therefore sit slightly above a
    building's base. Since the GroundSurface is dropped, that would show as a
    gap under the walls, so the wall base is extended down past the lowest
    terrain sample under the building.
    """
    adjusted = 0

    for building in buildings.buildings:
        if len(building.wall_tris) == 0:
            continue

        walls = building.wall_tris
        points = walls.reshape(-1, 3)
        base_z = float(points[:, 2].min())

        # Sample terrain on a small grid over the footprint and take the lowest.
        xs = np.linspace(points[:, 0].min(), points[:, 0].max(), 4)
        ys = np.linspace(points[:, 1].min(), points[:, 1].max(), 4)
        grid_x, grid_y = np.meshgrid(xs, ys)
        terrain_min = float(np.min(terrain_sampler(grid_x, grid_y)))

        target = min(base_z, terrain_min) - skirt_m
        if target >= base_z - 1e-6:
            continue

        # Only the vertices sitting on the building's base plane move.
        on_base = np.abs(walls[:, :, 2] - base_z) <= 0.05
        if not on_base.any():
            continue
        walls[:, :, 2] = np.where(on_base, target, walls[:, :, 2])
        adjusted += 1

    LOG.info("extended the wall base on %d buildings to meet the terrain", adjusted)
    return adjusted


def split_walls_at_height(
    triangles: np.ndarray, split_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cut wall triangles along a horizontal line, returning ``(lower, upper)``.

    A repeating grid of identical windows is the clearest sign a facade was
    generated. Real streets have a different ground storey, and giving it its
    own material means the wall has to be cut at the first-floor line: a
    triangle spanning both storeys cannot switch texture partway through,
    because UVs only exist at its corners.

    Each triangle is clipped against its own building's split height, so the
    cut follows the terrain rather than one global plane.
    """
    if len(triangles) == 0:
        empty = np.zeros((0, 3, 3), dtype=np.float64)
        return empty, empty

    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []

    for triangle, z_cut in zip(triangles, split_z):
        below = triangle[:, 2] < z_cut
        count = int(below.sum())

        if count == 3:
            lower.append(triangle)
            continue
        if count == 0:
            upper.append(triangle)
            continue

        # One or two corners are below the cut. Interpolate along the two edges
        # that cross it; the result is a triangle on one side and a quad on the
        # other, and the quad becomes two triangles.
        if count == 1:
            lone = int(np.flatnonzero(below)[0])
        else:
            lone = int(np.flatnonzero(~below)[0])

        apex = triangle[lone]
        other_a = triangle[(lone + 1) % 3]
        other_b = triangle[(lone + 2) % 3]

        def crossing(start: np.ndarray, end: np.ndarray) -> np.ndarray:
            span = end[2] - start[2]
            if abs(span) < 1e-12:
                return start.copy()
            t = float(np.clip((z_cut - start[2]) / span, 0.0, 1.0))
            return start + t * (end - start)

        cut_a = crossing(apex, other_a)
        cut_b = crossing(apex, other_b)

        apex_side = lower if count == 1 else upper
        quad_side = upper if count == 1 else lower

        apex_side.append(np.array([apex, cut_a, cut_b]))
        quad_side.append(np.array([cut_a, other_a, other_b]))
        quad_side.append(np.array([cut_a, other_b, cut_b]))

    def stack(chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, 3, 3), dtype=np.float64)
        return np.stack(chunks).astype(np.float64)

    return stack(lower), stack(upper)


def apply_ground_floor_split(
    buildings: BuildingSet, ground_floor_height_m: float
) -> int:
    """Separate each building's ground storey from the storeys above it."""
    split_count = 0

    for building in buildings.buildings:
        if len(building.wall_tris) == 0:
            continue
        # Nothing to split on a single-storey building: it is all ground floor.
        if building.wall_height_m <= ground_floor_height_m * 1.2:
            building.ground_wall_tris = building.wall_tris
            building.wall_tris = np.zeros((0, 3, 3), dtype=np.float64)
            split_count += 1
            continue

        cut = np.full(
            len(building.wall_tris), building.ground_z_nap + ground_floor_height_m
        )
        lower, upper = split_walls_at_height(building.wall_tris, cut)
        building.ground_wall_tris = lower
        building.wall_tris = upper
        split_count += 1

    LOG.info("split the ground storey off %d buildings", split_count)
    return split_count


def save_buildings(
    buildings: BuildingSet, path: Path, style_index: np.ndarray | None = None
) -> Path:
    """Write the building set to a compact npz for the Blender stage.

    Triangles are stored as one soup per surface class with a per-triangle
    building index. That keeps the Blender side to array slicing instead of
    CityJSON parsing.
    """
    wall_chunks, roof_chunks, ground_chunks = [], [], []
    wall_owner, roof_owner, ground_owner = [], [], []

    for index, building in enumerate(buildings.buildings):
        if len(building.wall_tris):
            wall_chunks.append(building.wall_tris)
            wall_owner.append(np.full(len(building.wall_tris), index, dtype=np.int32))
        if len(building.roof_tris):
            roof_chunks.append(building.roof_tris)
            roof_owner.append(np.full(len(building.roof_tris), index, dtype=np.int32))
        if len(building.ground_wall_tris):
            ground_chunks.append(building.ground_wall_tris)
            ground_owner.append(
                np.full(len(building.ground_wall_tris), index, dtype=np.int32)
            )

    def stack(chunks: list[np.ndarray]) -> np.ndarray:
        if chunks:
            return np.concatenate(chunks, axis=0).astype(np.float64)
        return np.zeros((0, 3, 3), dtype=np.float64)

    def stack_ids(chunks: list[np.ndarray]) -> np.ndarray:
        if chunks:
            return np.concatenate(chunks, axis=0)
        return np.zeros((0,), dtype=np.int32)

    if style_index is None:
        style_index = np.zeros(len(buildings.buildings), dtype=np.int32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        style_index=np.asarray(style_index, dtype=np.int32),
        wall_tris=stack(wall_chunks),
        wall_building=stack_ids(wall_owner),
        roof_tris=stack(roof_chunks),
        roof_building=stack_ids(roof_owner),
        ground_wall_tris=stack(ground_chunks),
        ground_wall_building=stack_ids(ground_owner),
        build_year=np.array(
            [b.build_year or 0 for b in buildings.buildings], dtype=np.int32
        ),
        # Fixed-width unicode rather than object dtype, so loading the archive
        # never needs allow_pickle.
        building_ids=np.array(
            [b.identifier for b in buildings.buildings], dtype="U64"
        ),
        ground_z_nap=np.array(
            [b.ground_z_nap for b in buildings.buildings], dtype=np.float64
        ),
        roof_max_nap=np.array(
            [b.roof_max_nap for b in buildings.buildings], dtype=np.float64
        ),
        height_m=np.array(
            [b.height_m for b in buildings.buildings], dtype=np.float64
        ),
        wall_height_m=np.array(
            [b.wall_height_m for b in buildings.buildings], dtype=np.float64
        ),
        floors=np.array([b.floors for b in buildings.buildings], dtype=np.int32),
        lod=np.array(buildings.lod),
    )
    LOG.info("wrote %s (%d buildings)", path, len(buildings))
    return path


def build_buildings(
    bbox: BBox,
    work_dir: Path,
    *,
    buildings_cfg: dict,
    terrain_sampler: Callable[[Any, Any], Any] | None = None,
    facade_variants: int = 1,
    ground_floor_height_m: float | None = None,
) -> BuildingSet:
    """Fetch, clean, ground, and cache the buildings for `bbox`."""
    result = fetch_buildings(bbox, buildings_cfg=buildings_cfg)

    result, note = filter_to_bbox(
        result, bbox, str(buildings_cfg["clip_mode"]).strip().lower()
    )
    LOG.info("%s", note)

    result, notes = filter_implausible(
        result,
        min_height_m=float(buildings_cfg["min_height_m"]),
        max_height_m=float(buildings_cfg["max_height_m"]),
    )
    for note in notes:
        LOG.warning("%s", note)

    if not result.buildings:
        raise ServiceError(
            "every 3DBAG building was filtered out as implausible; "
            "check buildings.min_height_m / max_height_m"
        )

    if terrain_sampler is not None:
        apply_ground_skirt(
            result,
            terrain_sampler,
            skirt_m=float(buildings_cfg["ground_skirt_m"]),
        )

    # Split before saving: the ground storey has to be its own geometry to
    # carry its own material.
    if ground_floor_height_m:
        apply_ground_floor_split(result, ground_floor_height_m)

    from .facade import style_for_building

    style_index = np.array(
        [
            style_for_building(b.height_m, b.build_year, facade_variants)
            for b in result.buildings
        ],
        dtype=np.int32,
    )
    years = [b.build_year for b in result.buildings if b.build_year]
    if years:
        LOG.info(
            "construction years on %d/%d buildings, %d to %d",
            len(years),
            len(result.buildings),
            min(years),
            max(years),
        )

    save_buildings(result, work_dir / "buildings.npz", style_index=style_index)
    return result


__all__ = [
    "Building",
    "BuildingSet",
    "apply_ground_floor_split",
    "apply_ground_skirt",
    "split_walls_at_height",
    "build_buildings",
    "decode_vertices",
    "fetch_buildings",
    "filter_implausible",
    "filter_to_bbox",
    "is_degenerate_surface",
    "iter_api_pages",
    "save_buildings",
    "triangulate_surface",
]
