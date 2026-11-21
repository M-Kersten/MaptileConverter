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
``output/<area>/photo_coverage.png``
    The same as a plan over the aerial, for a person to look at.

A run where Mapillary is switched on but cannot be reached, or has no token,
carries on without it: the photos add detail to a model that is complete
without them, so they must never be the reason a build fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .geo import BBox
from .http_util import ServiceError
from .photo_coverage import Coverage, render_coverage_map, score_views, wall_faces, write_coverage
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
    coverage = score_views(
        faces,
        result.photos,
        max_distance_m=margin,
        max_incidence_deg=float(photos_cfg.get("max_incidence_deg", 75.0)),
        resolution={
            "perspective": int(photos_cfg.get("image_size", 2048)),
            "fisheye": int(photos_cfg.get("image_size", 2048)),
            "spherical": int(photos_cfg.get("pano_image_size", 0)),
        },
    )
    coverage.thresholds = {
        "min_visible": float(photos_cfg.get("min_visible", 0.6)),
        "measure_px_per_m": float(photos_cfg.get("measure_px_per_m", 12.0)),
        "texture_px_per_m": float(photos_cfg.get("texture_px_per_m", 40.0)),
        "texture_max_incidence_deg": float(
            photos_cfg.get("texture_max_incidence_deg", 55.0)
        ),
    }
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


__all__ = ["StreetPhotoResult", "build_street_photos", "resolve_folder"]
