"""Every exposed wall drawn from its own layout, onto a few atlas pages.

The era styles give every wall of an era the same tile, repeated: windows
cut through by the corner, storeys out of step with the building's real
height, no door anywhere. Here each wall is drawn for itself
(`facade_layout`): its storeys fitted to its height, its windows centred in
its own bays, its door and its shop front, from the registers; and where the
street photos read the wall well enough, from what they measured instead.

The walls are packed onto atlas pages, a colour page and its normal map,
read by the model through the same per-triangle page and UVs the photo
textures used (`photo_textures.Placement`), so nothing downstream changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .facade_layout import (
    FacadeLayout,
    Masonry,
    draw_layout,
    era_for,
    layout_from_data,
    merged_layout,
)
from .photo_textures import PAD_PX, PhotoTextureSet, Placement, _padded_paste, pack

LOG = logging.getLogger(__name__)

DEFAULT_PX_PER_M = 25.0
DEFAULT_MAX_WALLS = 1500
DEFAULT_MIN_QUALITY = 0.5
MIN_WIDTH_M = 1.5
MIN_HEIGHT_M = 2.5
# How deep the drawn relief reads: glass sits this far behind a sill.
RELIEF_DEPTH_M = 0.25


def wall_normals(relief: np.ndarray, m_per_px: float, depth_m: float = RELIEF_DEPTH_M) -> np.ndarray:
    """Tangent-space normal map (OpenGL, green up) of a drawn wall's relief,
    in metres: a sill 8 cm tall throws the same edge at any resolution."""
    height = relief.astype(np.float64) * depth_m
    d_dy, d_dx = np.gradient(height, m_per_px)
    # Row 0 is the top of the wall, so up is minus the row direction.
    normal = np.stack([-d_dx, d_dy, np.ones_like(height)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def walls_to_draw(faces, *, measured: dict | None, limit: int, skip_keys=()) -> list[int]:
    """The faces to draw: exposed walls big enough to hold a window, the
    ones the photos measured first, then the largest."""
    measured = measured or {}
    skip = set(skip_keys)
    candidates = [
        i for i, face in enumerate(faces)
        if face.exposed and face.width_m >= MIN_WIDTH_M and face.height_m >= MIN_HEIGHT_M
        and tuple(face.key) not in skip
    ]
    candidates.sort(key=lambda i: (i not in measured, -faces[i].area_m2))
    return candidates[: max(0, int(limit))]


def building_colours(faces, layouts: dict, *, min_quality: float) -> dict[int, FacadeLayout]:
    """Per building, the measured wall that says most about its colours:
    the best read, if any read at all."""
    best: dict[int, FacadeLayout] = {}
    for index, layout in layouts.items():
        if layout is None or not layout.wall_rgb or index >= len(faces):
            continue
        building = faces[index].building
        if building not in best or layout.quality > best[building].quality:
            best[building] = layout
    return best


# A reading has to be at least this good for its storey height to be used.
STOREY_QUALITY = 0.25


def building_storeys(faces, layouts: dict, facades: dict | None = None,
                     ids: dict | None = None) -> dict[int, tuple[float, float | None]]:
    """Per building, the storey height its photos measured, and its ground
    storey's: the window rows are the steadiest thing a photo shows, even
    where its columns could not be told from the clutter. `facades` are the
    wall readings by building id (`facade_measure`), whose storey heights
    count too; `ids` maps building index to id."""
    found: dict[int, list] = {}
    grounds: dict[int, list] = {}
    for index, layout in layouts.items():
        if layout is None or index >= len(faces) or layout.storey_m is None:
            continue
        if layout.quality < STOREY_QUALITY:
            continue
        found.setdefault(faces[index].building, []).append(layout.storey_m)
        if layout.ground_m:
            grounds.setdefault(faces[index].building, []).append(layout.ground_m)
    for building, identifier in (ids or {}).items():
        reading = (facades or {}).get(identifier)
        storey = float(getattr(reading, "storey_m", 0.0) or 0.0)
        if 2.6 <= storey <= 5.5:
            found.setdefault(building, []).append(storey)
    return {
        building: (float(np.median(values)), float(np.median(grounds[building])) if building in grounds else None)
        for building, values in found.items()
    }


def draw_walls(
    buildings,
    faces,
    work_dir: Path,
    out_dir: Path | None = None,
    *,
    facade_cfg: dict,
    layouts: dict | None = None,
    facades: dict | None = None,
    skip_keys=(),
) -> PhotoTextureSet:
    """Draw every chosen wall and pack the drawings onto atlas pages.

    `faces` are `photo_coverage.WallFace`; `layouts` the walls the street
    photos measured, by face index (`facade_layout.measure_layout`);
    `facades` the street photos' wall readings by building id
    (`facade_measure`), whose colour is read from the sunlit half of each
    wall and is preferred; `skip_keys` faces that already wear something
    else.
    """
    from PIL import Image

    textures = PhotoTextureSet()
    layouts = layouts or {}
    px_per_m = float(facade_cfg.get("drawn_px_per_m", DEFAULT_PX_PER_M))
    m_per_px = 1.0 / max(px_per_m, 1.0)
    min_quality = float(facade_cfg.get("drawn_min_quality", DEFAULT_MIN_QUALITY))
    chosen = walls_to_draw(
        faces, measured=layouts, limit=int(facade_cfg.get("drawn_walls_max", DEFAULT_MAX_WALLS)),
        skip_keys=skip_keys,
    )
    if not chosen:
        return textures
    masonry = Masonry(work_dir if facade_cfg.get("photo_textures", True) else None)
    colours = building_colours(faces, layouts, min_quality=min_quality)
    ids = {i: b.identifier for i, b in enumerate(buildings.buildings)}
    storeys = building_storeys(faces, layouts, facades, ids)
    sunlit = {
        i: tuple(int(v) for v in facades[identifier].wall_rgb)
        for i, identifier in ids.items()
        if facades and identifier in facades and getattr(facades[identifier], "wall_rgb", None)
    }

    # Where the doors go. 3DBAG cuts a long front into a face at every
    # change of roof, and a door on each made a row of front doors along a
    # public building: a building that is not housing has one entrance, on
    # its widest wall to the street; a house front has its own.
    widest: dict[int, int] = {}
    for index in chosen:
        face = faces[index]
        building = buildings.buildings[face.building]
        if face.z_min > building.ground_z_nap + 1.0:
            continue
        if face.building not in widest or face.width_m > faces[widest[face.building]].width_m:
            widest[face.building] = index

    images, normals, used, reports = [], [], [], []
    counts = {"measured": 0, "data+colour": 0, "data": 0}
    for index in chosen:
        face = faces[index]
        building = buildings.buildings[face.building]
        eave = float(building.wall_top_nap or building.roof_max_nap)
        floors = int(building.floors) if getattr(building, "storeys_known", True) else None
        data = layout_from_data(
            face=index, width_m=face.width_m, z_min=face.z_min, z_max=face.z_max,
            ground_z=building.ground_z_nap, eave_z=eave, function=building.function,
            year=building.build_year, floors=floors,
            storey_m=storeys.get(face.building, (None, None))[0],
            ground_m=storeys.get(face.building, (None, None))[1],
            footprint_m2=float(getattr(building, "footprint_m2", 0.0) or 0.0),
            entrance=(
                widest.get(face.building) == index
                or (building.function == 0 and face.width_m >= 3.5)
            ),
            # One palette per building, not per wall.
            seed=face.building,
        )
        measured = layouts.get(index)
        if measured is None and face.building in colours:
            # A wall the photos did not read takes its building's colours
            # from one they did.
            measured = colours[face.building]
            if measured.face != index:
                measured = _colours_only(measured)
        if measured is not None and face.building in sunlit:
            # The colour read from the sunlit half of the building's walls.
            from dataclasses import replace

            measured = replace(measured, wall_rgb=sunlit[face.building])
        layout = merged_layout(measured, data, min_quality=min_quality)
        source = "measured" if layout.source == "photo" else layout.source
        counts[source] = counts.get(source, 0) + 1
        rgb, relief = draw_layout(
            layout, m_per_px, masonry=masonry, era=era_for(building.build_year).name, seed=index,
        )
        images.append(rgb)
        normals.append(wall_normals(relief, m_per_px))
        used.append((index, face))
        reports.append({"face": index, "building": building.identifier, **layout.to_dict()})

    page_px = int(facade_cfg.get("drawn_page_px", 4096))
    spots, scales = pack([(im.shape[1], im.shape[0]) for im in images], page_px)
    n_pages = max(s[0] for s in spots) + 1
    pages = [np.zeros((page_px, page_px, 3), dtype=np.uint8) for _ in range(n_pages)]
    normal_pages = [np.full((page_px, page_px, 3), (128, 128, 255), dtype=np.uint8) for _ in range(n_pages)]
    for image, normal, (index, face), (page, x, y), scale in zip(images, normals, used, spots, scales):
        if scale < 1.0:
            size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
            image = np.asarray(Image.fromarray(image).resize(size, Image.LANCZOS))
            normal = np.asarray(Image.fromarray(normal).resize(size, Image.LANCZOS))
        _padded_paste(pages[page], image, x, y, PAD_PX)
        _padded_paste(normal_pages[page], normal, x, y, PAD_PX)
        tangent = (face.end - face.start) / max(face.width_m, 1e-9)
        textures.placements[tuple(face.key)] = Placement(
            page=page, x=x, y=y, width=image.shape[1], height=image.shape[0],
            start=(float(face.start[0]), float(face.start[1])),
            tangent=(float(tangent[0]), float(tangent[1])),
            z_max=float(face.z_max),
            m_per_px_u=float(face.width_m) / image.shape[1],
            m_per_px_z=float(face.height_m) / image.shape[0],
        )
        textures.sources[tuple(face.key)] = []
    for i, (page, normal) in enumerate(zip(pages, normal_pages)):
        used_h = max(
            (p.y + p.height + PAD_PX for p in textures.placements.values() if p.page == i), default=256
        )
        height = int(min(page_px, int(np.ceil(used_h / 256.0)) * 256))
        path = work_dir / f"drawn_wall_{i:02d}.jpg"
        normal_path = work_dir / f"drawn_wall_{i:02d}_normal.png"
        Image.fromarray(page[:height]).save(path, quality=88)
        Image.fromarray(normal[:height]).save(normal_path, optimize=True)
        textures.pages.append(path)
        textures.normal_pages.append(normal_path)
        textures.page_sizes.append((page_px, height))
    textures.drawn = {"walls": len(textures.placements), **counts}
    if out_dir is not None:
        (Path(out_dir) / "facade_layouts.json").write_text(
            json.dumps({"walls": reports}, indent=1), encoding="utf-8"
        )
    LOG.info(
        "drawn walls: %d walls on %d pages at %.0f px/m; %d drawn as the photos measured them, "
        "%d laid out from the registers in the colours the photos read, %d from the registers alone",
        len(textures.placements), len(textures.pages), px_per_m, counts.get("measured", 0),
        counts.get("data+colour", 0), counts.get("data", 0),
    )
    return textures


def _colours_only(layout: FacadeLayout) -> FacadeLayout:
    """A measured layout reduced to what carries over to another wall of the
    same building: its colours and material, never its windows."""
    from dataclasses import replace

    return replace(layout, openings=[], quality=0.0)


def combine(first: PhotoTextureSet, second: PhotoTextureSet) -> PhotoTextureSet:
    """Two atlases as one: the second's pages numbered after the first's,
    and where both hold a wall, the first's."""
    from dataclasses import replace

    out = PhotoTextureSet(
        pages=list(first.pages) + list(second.pages),
        page_sizes=list(first.page_sizes) + list(second.page_sizes),
        placements=dict(first.placements),
        sources=dict(first.sources),
        credits=dict(first.credits),
        seen_share=dict(first.seen_share),
    )
    out.normal_pages = list(getattr(first, "normal_pages", []) or [None] * len(first.pages)) + list(
        getattr(second, "normal_pages", []) or [None] * len(second.pages)
    )
    out.drawn = dict(getattr(first, "drawn", {}) or getattr(second, "drawn", {}) or {})
    offset = len(first.pages)
    for key, placement in second.placements.items():
        if key not in out.placements:
            out.placements[key] = replace(placement, page=placement.page + offset)
            out.sources[key] = second.sources.get(key, [])
    return out


__all__ = ["combine", "draw_walls", "wall_normals", "walls_to_draw"]
