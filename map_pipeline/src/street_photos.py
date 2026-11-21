"""The street photos stage: gather the photos, place them, see what they see.

Runs after the buildings, because what a photo is worth is decided by the
walls in front of it. Mapillary and a folder of your own pictures are read
into one list of :class:`photos.Photo`, each put at camera height on the
measured ground, and scored against every wall face (see `photo_coverage`).

What it leaves behind:

``work/<area>/photos.json``
    Every photo kept, with its pose in RD/NAP.
``work/<area>/photo_coverage.json``
    Every wall face, its best views, and what the photos can do for it.
``work/<area>/alignment.json``
    For every photo checked against the model: how well its edges agreed
    before and after its pose was refined, the correction, and the verdict.
``output/<area>/photo_coverage.png``
    The same as a plan over the aerial, for a person to look at.
``output/<area>/photo_checks/``
    A dozen of the checked photos with the model drawn over them, magenta
    where Mapillary put the camera and green where the check moved it: the
    best fits, and the rejections, so a person can see what was decided.

Refinement is what makes a photo fit to paint with. Only the photos that are
among the best views of some wall are checked, since the rest are never
used; each is downloaded once and cached under ``work/<area>/_mapillary``.

A run where Mapillary is switched on but cannot be reached, or has no token,
carries on without it: the photos add detail to a model that is complete
without them, so they must never be the reason a build fails.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .geo import BBox
from .http_util import ServiceError
from .photo_coverage import (
    MEASURE,
    TEXTURE,
    Coverage,
    _SegmentIndex,
    render_coverage_map,
    score_views,
    wall_faces,
    write_coverage,
)
from .photos import SOURCE_MAPILLARY, SOURCE_OWN, Photo, save_photos, thin_photos, with_height

LOG = logging.getLogger(__name__)


@dataclass
class StreetPhotoResult:
    photos: list[Photo] = field(default_factory=list)
    coverage: Coverage | None = None
    mapillary: dict = field(default_factory=dict)
    own: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    coverage_map: Path | None = None
    alignments: dict = field(default_factory=dict)
    # Building id -> facade_measure.FacadeMeasurement.
    facades: dict = field(default_factory=dict)
    # Photo id -> brightness gain against the aerial.
    gains: dict = field(default_factory=dict)
    # Face index -> facade_layout.FacadeLayout: each read wall's layout.
    layouts: dict = field(default_factory=dict)

    @property
    def uses_mapillary(self) -> bool:
        return any(p.source == SOURCE_MAPILLARY for p in self.photos)

    def stats(self) -> dict:
        out: dict = {
            "photos": len(self.photos),
            "from_mapillary": sum(1 for p in self.photos if p.source == SOURCE_MAPILLARY),
            "own": sum(1 for p in self.photos if p.source == SOURCE_OWN),
        }
        if self.mapillary:
            out["mapillary"] = self.mapillary
        if self.own:
            out["own_photos"] = self.own
        if self.alignments:
            checked = list(self.alignments.values())
            ok = [a for a in checked if a.verified]
            out["pose_check"] = {
                "checked": len(checked),
                "verified": len(ok),
                "median_inliers_before": round(float(np.median([a.before for a in ok])), 3) if ok else None,
                "median_inliers_after": round(float(np.median([a.after for a in ok])), 3) if ok else None,
                "median_shift_m": round(float(np.median([np.hypot(*a.params[3:5]) for a in ok])), 2) if ok else None,
                "median_turn_deg": round(float(np.median([abs(a.params[0]) for a in ok])), 2) if ok else None,
            }
        if self.facades:
            measured = list(self.facades.values())
            out["facades"] = {
                "buildings_read": len(measured),
                "with_colour": sum(1 for m in measured if m.has_colour),
                "with_window_columns": sum(1 for m in measured if m.has_windows),
                "with_storeys_counted": sum(1 for m in measured if m.storeys > 0),
                "walls_read": sum(len(m.walls) for m in measured),
            }
        if self.gains:
            values = np.array([float(g) for g in self.gains.values()])
            out["brightness_from_aerial"] = {
                "photos": len(values),
                "median_gain": round(float(np.median(values)), 3),
                "range": [round(float(values.min()), 3), round(float(values.max()), 3)],
            }
        if self.coverage is not None:
            out["coverage"] = self.coverage.stats()
        if self.problems:
            out["problems"] = list(self.problems)
        return out


def resolve_folder(folder: str, base: Path) -> Path | None:
    if not str(folder or "").strip():
        return None
    path = Path(str(folder)).expanduser()
    return path if path.is_absolute() else (base / path)


def build_street_photos(
    bbox: BBox,
    work_dir: Path,
    out_dir: Path,
    *,
    photos_cfg: dict,
    buildings,
    terrain,
    base_dir: Path,
    aerial_path: Path | None = None,
    session=None,
    trees=None,
    deck_at=None,
) -> StreetPhotoResult:
    """`deck_at(x, y)`, the top of a bridge deck there or NaN, puts a photo
    taken on a bridge on its deck: the survey under a bridge is the wharf or
    the water, four metres down, and the camera under the deck saw nothing
    the photo shows."""
    result = StreetPhotoResult()
    gathered: list[Photo] = []

    if photos_cfg.get("mapillary"):
        from .mapillary import fetch_photos

        try:
            found, counts = fetch_photos(
                bbox, photos_cfg=photos_cfg, work_dir=work_dir, session=session
            )
        except ServiceError as exc:
            # MissingToken is a ServiceError too, and says how to fix itself.
            LOG.warning("Mapillary skipped: %s", exc)
            result.problems.append(f"Mapillary skipped: {exc}")
        else:
            thinned = thin_photos(found, float(photos_cfg.get("spacing_m", 4.0)))
            counts["after_thinning"] = len(thinned)
            result.mapillary = counts
            LOG.info(
                "Mapillary: %d images around the area, %d kept, %d after thinning "
                "to one per %.0f m of each sequence",
                counts["found"], counts["kept"], len(thinned),
                float(photos_cfg.get("spacing_m", 4.0)),
            )
            gathered.extend(thinned)

    folder = resolve_folder(photos_cfg.get("folder", ""), base_dir)
    if folder is not None:
        from .own_photos import read_folder

        own = read_folder(folder)
        result.own = own.stats()
        for name, why in own.skipped:
            LOG.warning("own photo %s skipped: %s", name, why)
        LOG.info(
            "own photos: %d read from %s, %d skipped", len(own.photos), folder,
            len(own.skipped),
        )
        gathered.extend(own.photos)

    pinned = [p for p in gathered if p.pin]
    if pinned:
        from .photo_coverage import mark_exposed
        from .photo_pin import apply_pin

        pin_faces = wall_faces(buildings)
        mark_exposed(pin_faces, _SegmentIndex(pin_faces))
        placed = {}
        for photo in pinned:
            moved, why = apply_pin(photo, pin_faces, buildings)
            if moved is None:
                LOG.warning("pin on %s not used: %s", photo.id, why)
                result.problems.append(f"pin on {photo.id} not used: {why}")
            else:
                placed[photo.id] = moved
        gathered = [placed.get(p.id, p) for p in gathered]
        if placed:
            LOG.info("%d photos placed by their pins", len(placed))

    margin = float(photos_cfg.get("max_distance_m", 40.0))
    reach = bbox.buffered(margin)
    inside = [
        p for p in gathered
        if reach.xmin <= p.x <= reach.xmax and reach.ymin <= p.y <= reach.ymax
    ]
    outside = len(gathered) - len(inside)
    if outside:
        LOG.info("%d photos lie too far outside the area to see it", outside)

    heights = {
        SOURCE_MAPILLARY: float(photos_cfg.get("mapillary_camera_height_m", 2.0)),
        SOURCE_OWN: float(photos_cfg.get("own_camera_height_m", 1.6)),
    }
    def ground_at(x, y):
        z = np.asarray(terrain.sample(x, y), dtype=np.float64)
        if deck_at is None:
            return z
        deck = np.asarray(deck_at(x, y, 0.0), dtype=np.float64).reshape(z.shape)
        return np.where(np.isfinite(deck), np.fmax(deck, z), z)

    result.photos = with_height(inside, ground_at, heights)
    save_photos(result.photos, work_dir / "photos.json")

    if not result.photos:
        LOG.info("no street photos for this area")
        return result

    faces = wall_faces(buildings)
    resolution = {
        "perspective": int(photos_cfg.get("image_size", 2048)),
        "fisheye": int(photos_cfg.get("image_size", 2048)),
        "spherical": int(photos_cfg.get("pano_image_size", 0)),
    }
    thresholds = {
        "min_visible": float(photos_cfg.get("min_visible", 0.6)),
        "measure_px_per_m": float(photos_cfg.get("measure_px_per_m", 12.0)),
        "texture_px_per_m": float(photos_cfg.get("texture_px_per_m", 40.0)),
        "texture_max_incidence_deg": float(
            photos_cfg.get("texture_max_incidence_deg", 55.0)
        ),
    }

    def score(photos: list[Photo]) -> Coverage:
        coverage = score_views(
            faces,
            photos,
            max_distance_m=margin,
            max_incidence_deg=float(photos_cfg.get("max_incidence_deg", 75.0)),
            resolution=resolution,
        )
        coverage.thresholds = dict(thresholds)
        return coverage

    coverage = score(result.photos)
    if photos_cfg.get("refine", True):
        refined = refine_poses(
            result, coverage, buildings, work_dir, out_dir,
            photos_cfg=photos_cfg, session=session,
        )
        if refined:
            result.photos = refined
            save_photos(result.photos, work_dir / "photos.json")
            coverage = score(result.photos)
            coverage.thresholds["texture_needs_verified"] = True

    result.coverage = coverage
    write_coverage(coverage, work_dir / "photo_coverage.json")

    # Read when some photo is trusted: checked against the model, or pinned
    # to a wall by hand, which skips the check.
    if photos_cfg.get("read_facades", True) and any(p.verified for p in result.photos):
        read_facades(
            result, coverage, buildings, terrain, work_dir, out_dir,
            photos_cfg=photos_cfg, bbox=bbox, aerial_path=aerial_path, trees=trees,
        )
    result.coverage_map = render_coverage_map(
        coverage, result.photos, bbox, out_dir / "photo_coverage.png",
        background=aerial_path,
    )

    s = coverage.stats()
    LOG.info(
        "street photos: %d of %d photos see a wall; of the exposed wall area "
        "%.0f%% can wear a photo, %.0f%% can be measured, %.0f%% is seen at all",
        s["photos_seeing_a_wall"], s["photos"], 100 * s["share_texture_ready"],
        100 * s["share_measurable"], 100 * s["share_seen"],
    )
    LOG.info(
        "  %d of %d buildings have a wall that can wear a photo, %d one that "
        "can be measured",
        s["buildings_with_texture_ready_wall"], s["buildings"],
        s["buildings_with_measurable_wall"],
    )
    return result


def bake_photo_textures(
    result: StreetPhotoResult,
    buildings,
    work_dir: Path,
    out_dir: Path,
    *,
    photos_cfg: dict,
    terrain=None,
    trees=None,
    surfaces=None,
    observed=None,
    n_variants: int = 5,
    session=None,
):
    """Route 2: the walls that can wear their photos get them."""
    from .facade_capture import crowns_from_trees, parking_lookup
    from .photo_textures import PhotoTextureSet, bake_textures

    if result.coverage is None or str(photos_cfg.get("textures", "off")).lower() == "off":
        return PhotoTextureSet()
    parking = None
    if surfaces is not None and getattr(surfaces, "class_grid", None) is not None and terrain is not None:
        parking = parking_lookup(surfaces.class_grid, terrain.xs, terrain.ys)
    return bake_textures(
        result,
        result.coverage,
        buildings,
        work_dir,
        out_dir,
        photos_cfg=photos_cfg,
        observed=observed,
        n_variants=n_variants,
        faces_index=_SegmentIndex(result.coverage.faces),
        crowns=crowns_from_trees(trees) if trees is not None else [],
        parking=parking,
        session=session,
    )


def _read_layout(face, face_index, building, photos, pixels, *, index, crowns, gain, photos_cfg):
    """One wall's layout (`facade_layout`), read off each of its photos
    straightened on its own at four centimetres a pixel, its colours brought
    to the aerial's brightness like the wall's measured colour is."""
    from .facade_capture import rectify_view
    from .facade_layout import M_PER_PX, measure_layout
    from .photo_colour import wall_gain

    views = []
    for photo in photos[:4]:
        try:
            image = pixels.get(photo.path)
        except OSError:
            continue
        rgb, weight = rectify_view(
            face, face_index, photo, image, m_per_px=M_PER_PX, faces_index=index,
            crowns=crowns, target_px_per_m=1.0 / M_PER_PX,
        )
        if float(weight.mean()) >= 0.02:
            views.append((rgb, weight))
    if not views:
        return None
    layout = measure_layout(
        views, face=face_index, m_per_px=M_PER_PX, width_m=face.width_m,
        z_min=face.z_min, z_max=face.z_max, ground_z=building.ground_z_nap,
        eave_z=float(building.wall_top_nap or building.roof_max_nap),
        function=building.function, year=building.build_year,
    )
    if layout is None:
        return None
    # All the way to the aerial's sunlit brightness: the drawn wall is lit
    # by the model's own sun, and a wall photographed in shade would be
    # shaded twice.
    lift = wall_gain(layout.wall_rgb, gain, strength=1.0)
    layout.wall_rgb = _scaled(layout.wall_rgb, lift)
    layout.frame_rgb = _scaled(layout.frame_rgb, lift)
    if layout.fascia:
        z0, z1, rgb = layout.fascia
        layout.fascia = (z0, z1, _scaled(rgb, lift))
    return layout


def walls_to_read(coverage: Coverage, buildings, limit: int, per_building: int = 4):
    """The walls worth reading, with the views to read them from.

    Exposed walls that some photo verified for the wall's own building may
    see, largest first, at most `per_building` of one building. Returns
    ``[(face index, [photo ids])]``.
    """
    chosen: list[tuple[int, list[str]]] = []
    taken: dict[int, int] = {}
    order = sorted(range(len(coverage.faces)), key=lambda i: -coverage.faces[i].area_m2)
    for i in order:
        face = coverage.faces[i]
        if not face.exposed or coverage.face_class(i) not in (MEASURE, TEXTURE):
            continue
        views = [v.photo for v in coverage.views[i] if v.verified is True][:4]
        if not views or taken.get(face.building, 0) >= per_building:
            continue
        taken[face.building] = taken.get(face.building, 0) + 1
        chosen.append((i, views))
        if len(chosen) >= limit:
            break
    return chosen


def read_facades(
    result: StreetPhotoResult,
    coverage: Coverage,
    buildings,
    terrain,
    work_dir: Path,
    out_dir: Path,
    *,
    photos_cfg: dict,
    bbox: BBox,
    aerial_path: Path | None,
    trees=None,
) -> None:
    """Measure the facades the checked photos see well (route 1)."""
    from .facade_capture import PhotoPixels, capture_wall, crowns_from_trees, save_wall_image
    from .facade_measure import combine_walls, measure_wall
    from .photo_colour import AerialColours, photo_gain, smooth_gains, wall_gain

    chosen = walls_to_read(coverage, buildings, int(photos_cfg.get("read_max_walls", 400)))
    if not chosen:
        return
    by_id = {p.id: p for p in result.photos}
    index = _SegmentIndex(coverage.faces)
    crowns = crowns_from_trees(trees) if trees is not None else []
    pixels = PhotoPixels()

    # Brightness against the aerial, per photo and then per sequence.
    gains: dict[str, np.ndarray] = {}
    if photos_cfg.get("colour_from_aerial", True) and aerial_path and Path(aerial_path).is_file():
        aerial = AerialColours.load(aerial_path, bbox)
        needed = sorted({pid for _, views in chosen for pid in views})
        for pid in needed:
            photo = by_id.get(pid)
            if photo is None or not photo.path:
                continue
            try:
                gain, _ = photo_gain(
                    photo, pixels.get(photo.path), aerial, terrain.sample,
                    faces_index=index, crowns=crowns,
                )
            except OSError:
                continue
            if gain is not None:
                gains[pid] = gain
        gains = smooth_gains(by_id, gains)
        result.gains = {pid: float(g[0]) for pid, g in gains.items()}

    walls_by_building: dict[int, list] = {}
    sheet = []
    walls_dir = work_dir / "walls"
    walls_dir.mkdir(parents=True, exist_ok=True)
    m_per_px = float(photos_cfg.get("read_m_per_px", 0.05))
    for face_index, views in chosen:
        face = coverage.faces[face_index]
        building = buildings.buildings[face.building]
        photos = [by_id[pid] for pid in views if pid in by_id and by_id[pid].path]
        wall = capture_wall(
            face, face_index, photos, pixels,
            m_per_px=m_per_px, faces_index=index, crowns=crowns,
            target_px_per_m=1.0 / m_per_px,
        )
        if wall is None:
            continue
        measurement = measure_wall(
            wall,
            ground_z=building.ground_z_nap,
            wall_top_z=building.wall_top_nap or building.roof_max_nap,
            building_id=building.identifier,
            area_m2=face.area_m2,
        )
        used = [g[0] for pid, g in gains.items() if pid in wall.views] or [1.0]
        if measurement.wall_rgb is not None:
            gain = wall_gain(
                measurement.wall_rgb, float(np.median(used)),
                strength=float(photos_cfg.get("brightness_strength", 0.7)),
            )
            measurement.gain = gain
            measurement.wall_rgb = _scaled(measurement.wall_rgb, gain)
            measurement.trim_rgb = _scaled(measurement.trim_rgb, gain)
            measurement.glass_rgb = _scaled(measurement.glass_rgb, gain)
            wall.rgb = np.clip(wall.rgb * gain, 0.0, 255.0)
        walls_by_building.setdefault(face.building, []).append(measurement)
        if len(sheet) < 12 and measurement.wall_rgb is not None:
            sheet.append((wall, measurement, building))
        save_wall_image(wall, walls_dir / f"{building.identifier}_{face_index}.jpg")
        if photos_cfg.get("read_layouts", True):
            layout = _read_layout(
                face, face_index, building, photos, pixels, index=index, crowns=crowns,
                gain=float(np.median(used)), photos_cfg=photos_cfg,
            )
            if layout is not None:
                result.layouts[face_index] = layout

    for building_index, walls in walls_by_building.items():
        building = buildings.buildings[building_index]
        result.facades[building.identifier] = combine_walls(
            building.identifier,
            walls,
            ground_z=building.ground_z_nap,
            wall_top_z=building.wall_top_nap or building.roof_max_nap,
            model_floors=int(building.floors),
        )

    (out_dir / "facades.json").write_text(
        json.dumps(
            {"buildings": [m.to_dict() for m in result.facades.values()]}, indent=1
        ),
        encoding="utf-8",
    )
    _draw_wall_checks(sheet, out_dir / "photo_checks")
    LOG.info(
        "facades: read %d walls of %d buildings; %d with a wall colour, %d with "
        "window columns, %d with storeys counted",
        sum(len(w) for w in walls_by_building.values()), len(result.facades),
        sum(1 for m in result.facades.values() if m.has_colour),
        sum(1 for m in result.facades.values() if m.has_windows),
        sum(1 for m in result.facades.values() if m.storeys > 0),
    )


def _scaled(rgb, gain: float):
    if rgb is None:
        return None
    return tuple(int(min(255, round(v * gain))) for v in rgb)


def _draw_wall_checks(walls, folder: Path) -> None:
    """Straightened walls with the measured bays and storeys drawn on."""
    from PIL import Image, ImageDraw

    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("wall_*.jpg"):
        old.unlink()
    for wall, measurement, building in walls:
        shown = wall.rgb * wall.weight[..., None] + 128.0 * (1 - wall.weight[..., None])
        image = Image.fromarray(np.clip(shown, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        step = wall.m_per_px
        if measurement.bay_m > 0:
            u = 0.0
            while u < wall.width_m:
                draw.line([(u / step, 0), (u / step, image.height)], fill=(0, 230, 255))
                u += measurement.bay_m
        if measurement.storey_m > 0:
            z = building.ground_z_nap
            while z < wall.z_max:
                row = float(wall.row_of_z(z))
                draw.line([(0, row), (image.width, row)], fill=(255, 230, 0))
                z += measurement.storey_m
        swatch = tuple(int(v) for v in measurement.wall_rgb)
        draw.rectangle([4, 4, 44, 44], fill=swatch, outline=(255, 255, 255))
        image.save(folder / f"wall_{building.identifier}_{wall.face}.jpg", quality=85)


def refine_candidates(coverage: Coverage, limit: int) -> list[str]:
    """The photos worth checking: the two best views of every wall that
    could be measured or textured, most wall area first."""
    worth: dict[str, float] = {}
    for i, face in enumerate(coverage.faces):
        if not face.exposed or coverage.face_class(i) not in (MEASURE, TEXTURE):
            continue
        for view in coverage.views[i][:2]:
            worth[view.photo] = worth.get(view.photo, 0.0) + face.area_m2
    ranked = sorted(worth, key=lambda pid: -worth[pid])
    return ranked[: max(0, int(limit))]


def refine_poses(
    result: StreetPhotoResult,
    coverage: Coverage,
    buildings,
    work_dir: Path,
    out_dir: Path,
    *,
    photos_cfg: dict,
    session=None,
) -> list[Photo] | None:
    """Check the photos that matter against the model, and refine their poses.

    Returns the full photo list with refined poses and verdicts, or None when
    nothing could be checked.
    """
    from .pose_refine import (
        align_photo,
        building_edges,
        check_sequences,
        draw_overlay,
        drop_hidden_seams,
        load_image,
        sample_edges,
    )

    wanted = set(refine_candidates(coverage, int(photos_cfg.get("refine_max", 250))))
    if not wanted:
        return None
    by_id = {p.id: p for p in result.photos}
    result_camera = {p.id: p.camera for p in result.photos}
    chosen = [by_id[pid] for pid in wanted]

    from_mapillary = [p for p in chosen if p.source == SOURCE_MAPILLARY]
    if from_mapillary:
        from .mapillary import download_images, find_token, session_for

        try:
            http = session or session_for(find_token())
            fetched = download_images(
                from_mapillary,
                work_dir / "_mapillary" / "images",
                session=http,
                size=int(photos_cfg.get("refine_image_size", 2048)),
                api_url=str(photos_cfg.get("mapillary_url", "https://graph.mapillary.com")),
                timeout=float(photos_cfg.get("timeout_s", 60.0)),
                max_retries=int(photos_cfg.get("max_retries", 4)),
            )
        except ServiceError as exc:
            LOG.warning("Mapillary images could not be fetched: %s", exc)
            result.problems.append(f"Mapillary images could not be fetched: {exc}")
            fetched = []
        for photo in fetched:
            by_id[photo.id] = photo

    segments, owner = building_edges(buildings)
    samples = drop_hidden_seams(sample_edges(segments, owner=owner), coverage.faces)
    building_ids = [b.identifier for b in buildings.buildings]
    index = _SegmentIndex(coverage.faces)
    margin = float(photos_cfg.get("max_distance_m", 40.0))

    alignments = {}
    originals = {}
    images = {}
    for pid in sorted(wanted):
        photo = by_id[pid]
        if not photo.path:
            continue
        if photo.pose == "manual" and photo.verified and photo.pin:
            # Pinned by hand: exact for its wall, and not to be moved by
            # edges that a tree or a sign may have lined up wrongly.
            continue
        try:
            image = load_image(photo)
        except OSError as exc:
            LOG.warning("photo %s could not be read: %s", pid, exc)
            continue
        originals[pid] = photo
        refined, alignment = align_photo(
            photo, image, samples, index, building_ids=building_ids,
            max_distance_m=margin,
        )
        # Checked on a downsized copy, but the photo is still the photo: its
        # sharpness, and what it can paint, is that of the original.
        by_id[pid] = replace(refined, camera=result_camera[pid])
        alignments[pid] = alignment
        images[pid] = None  # loaded again for the few that are drawn

    withdrawn = check_sequences(by_id, alignments)
    for pid in withdrawn:
        by_id[pid] = replace(by_id[pid], verified=False, verified_buildings=[])
    if not alignments:
        return None
    result.alignments = alignments

    (work_dir / "alignment.json").write_text(
        json.dumps(
            {"photos": [alignments[pid].to_dict() for pid in sorted(alignments)]},
            indent=1,
        ),
        encoding="utf-8",
    )

    # A dozen to look at: the best fits and the most telling rejections.
    checks = out_dir / "photo_checks"
    if checks.is_dir():
        for old in checks.glob("*.jpg"):
            old.unlink()
    checks.mkdir(parents=True, exist_ok=True)
    good = sorted((a for a in alignments.values() if a.verified), key=lambda a: -a.after)[:6]
    bad = sorted(
        (a for a in alignments.values() if not a.verified and a.points >= 150),
        key=lambda a: -a.points,
    )[:6]
    for alignment in good + bad:
        tag = "ok" if alignment.verified else "rejected"
        try:
            # Drawn on the copy that was checked, with the camera that copy has.
            checked = originals[alignment.photo]
            draw_overlay(
                replace(by_id[alignment.photo], camera=checked.camera),
                load_image(checked),
                segments,
                checks / f"{tag}_{alignment.photo}.jpg",
                before=originals[alignment.photo],
            )
        except OSError:
            continue

    verified = sum(1 for a in alignments.values() if a.verified)
    LOG.info(
        "pose check: %d of %d photos line up with the model after refinement%s",
        verified, len(alignments),
        f" ({len(withdrawn)} withdrawn for disagreeing with their sequence)"
        if withdrawn else "",
    )
    return [by_id[p.id] for p in result.photos]


__all__ = [
    "StreetPhotoResult",
    "bake_photo_textures",
    "read_facades",
    "walls_to_read",
    "build_street_photos",
    "refine_candidates",
    "refine_poses",
    "resolve_folder",
]
