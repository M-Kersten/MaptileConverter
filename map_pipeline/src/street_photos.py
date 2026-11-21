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
) -> StreetPhotoResult:
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
    result.photos = with_height(inside, terrain.sample, heights)
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
    "build_street_photos",
    "refine_candidates",
    "refine_poses",
    "resolve_folder",
]
