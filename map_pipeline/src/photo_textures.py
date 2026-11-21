"""Walls that wear their own photographs.

Route 2 of the street photos. Where a checked photo sees a wall sharply and
square-on, the wall's texture is the photo itself: straightened (see
`facade_capture`), its views fused pixel by pixel from whichever saw each
part best, its brightness matched to the aerial, and packed with every other
such wall onto a few large atlas pages that the model's materials read.

What no photo saw has to come from somewhere, and there are three places,
tried in order:

1. **The wall's own rhythm.** Facades repeat: the bay one along, or the
   storey above, usually shows what a tree or a parked van hides. Holes are
   copied from the same spot one or more bays across, or storeys up, using
   the periods the facade reading measured. A ground storey does not repeat
   upwards, so it only copies sideways.
2. **The wall's generated facade.** What is left is drawn as the model
   would have drawn the wall without photos: its measured style, in the
   same bay and storey grid.
3. Both blended in over a few decimetres, so no hole has an edge.

Mapillary photos are CC BY-SA 4.0, and a texture made from one is a
derivative: it is shared alike and credited. ``photos.textures`` decides
whose photos may become textures: ``own`` (the default) only yours, ``all``
Mapillary's too, ``off`` none. Every texture's photos and their authors are
listed beside the model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .facade_capture import PhotoPixels, _box_blur, capture_wall, luminance
from .photos import SOURCE_MAPILLARY, SOURCE_OWN, Photo

LOG = logging.getLogger(__name__)

PAGE_PX = 4096
PAD_PX = 4
# A pixel with less weight than this is a hole.
HOLE_WEIGHT = 0.3
# Sharpness asked of a photo texture, pixels per metre of wall.
MIN_PX_PER_M, MAX_PX_PER_M = 24.0, 60.0
# Up to this height above the street, views that disagree are taken to show
# something in front of the wall: the cars parked along the kerb where the
# BGT draws no parking bay, a cyclist, a bin. Higher up, glass looks
# different from every angle and is left alone.
STREET_BAND_M = 2.2
DISAGREE = 0.5


@dataclass
class Placement:
    """Where one wall's texture lies on its atlas page."""

    page: int
    x: int
    y: int
    width: int
    height: int
    # The wall's own frame: its left end seen from outside, the direction
    # along it, and its top; and how many metres of wall one texture pixel
    # covers across and up, which differ a little once rounded to pixels.
    start: tuple
    tangent: tuple
    z_max: float
    m_per_px_u: float
    m_per_px_z: float

    def uv(self, corners: np.ndarray, page_size: tuple[int, int]) -> np.ndarray:
        """Texture coordinates of world points on this wall."""
        start = np.asarray(self.start)
        tangent = np.asarray(self.tangent)
        u_m = (corners[..., :2] - start) @ tangent
        px = self.x + u_m / self.m_per_px_u
        py = self.y + (self.z_max - corners[..., 2]) / self.m_per_px_z
        w, h = page_size
        # Blender's V runs up from the bottom of the image.
        return np.stack([px / w, 1.0 - py / h], axis=-1)

    def to_dict(self) -> dict:
        return {
            "page": self.page, "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "start": list(self.start), "tangent": list(self.tangent),
            "z_max": self.z_max, "m_per_px_u": self.m_per_px_u, "m_per_px_z": self.m_per_px_z,
        }


@dataclass
class PhotoTextureSet:
    pages: list[Path] = field(default_factory=list)
    page_sizes: list[tuple[int, int]] = field(default_factory=list)
    # Face key -> placement.
    placements: dict = field(default_factory=dict)
    # Face key -> the photos it was made from.
    sources: dict = field(default_factory=dict)
    credits: dict = field(default_factory=dict)
    seen_share: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.placements)

    def stats(self) -> dict:
        shares = list(self.seen_share.values())
        return {
            "walls": len(self.placements),
            "pages": len(self.pages),
            "photos": len({pid for ids in self.sources.values() for pid in ids}),
            "median_photographed_share": round(float(np.median(shares)), 3) if shares else 0.0,
        }


def may_texture(photo: Photo, setting: str) -> bool:
    """Whether a photo may become a texture under ``photos.textures``."""
    setting = str(setting or "own").lower()
    if setting == "off":
        return False
    if setting == "all":
        return True
    return photo.source == SOURCE_OWN


def texture_walls(coverage, photos: dict, setting: str, limit: int, per_wall: int = 3):
    """``[(face index, [photo ids])]``: walls to texture, largest first."""
    from .photo_coverage import TEXTURE

    chosen = []
    order = sorted(range(len(coverage.faces)), key=lambda i: -coverage.faces[i].area_m2)
    for i in order:
        face = coverage.faces[i]
        if not face.exposed or coverage.face_class(i) != TEXTURE:
            continue
        views = [
            v.photo for v in coverage.views[i]
            if v.verified is True and v.photo in photos and may_texture(photos[v.photo], setting)
        ][:per_wall]
        if views:
            chosen.append((i, views))
        if len(chosen) >= limit:
            break
    return chosen


# --------------------------------------------------------------------------
# Lighting and holes
# --------------------------------------------------------------------------


def flatten_light(rgb: np.ndarray, weight: np.ndarray, m_per_px: float, *, strength: float = 0.5, radius_m: float = 2.5) -> np.ndarray:
    """Even out light that changes over metres, keep what changes over centimetres.

    A building's shadow across half a facade, or the sun on one end of it,
    is lighting the renderer will add again; a window, a joint or a sill is
    the wall itself. Dividing by a heavily blurred luminance, at `strength`
    of the way, keeps the second and softens the first.
    """
    if strength <= 0:
        return rgb
    lum = luminance(rgb)
    seen = weight > HOLE_WEIGHT
    if seen.sum() < 100:
        return rgb
    radius = max(1, int(round(radius_m / m_per_px)))
    # Blur only what was seen, so holes do not drag the light level down.
    total = _box_blur(np.where(seen, lum, 0.0), radius)
    count = _box_blur(seen.astype(np.float64), radius)
    low = total / np.maximum(count, 1e-3)
    target = float(np.median(lum[seen]))
    ratio = np.clip(target / np.maximum(low, 1.0), 0.6, 1.7) ** strength
    return np.clip(rgb * ratio[..., None], 0, 255).astype(np.float32)


def fill_from_rhythm(
    rgb: np.ndarray,
    weight: np.ndarray,
    *,
    m_per_px: float,
    bay_m: float,
    storey_m: float,
    z_of_row: np.ndarray,
    ground_top_z: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill holes from the same spot a whole number of bays or storeys away.

    Returns the filled picture and the weight, filled pixels at 0.6: known,
    but not seen. A ground storey copies only sideways.
    """
    rgb = rgb.copy()
    weight = weight.copy()
    hole = weight < HOLE_WEIGHT
    if not hole.any() or (bay_m <= 0 and storey_m <= 0):
        return rgb, weight
    h, w = hole.shape
    zone = (z_of_row < ground_top_z).astype(np.int8) if ground_top_z is not None else np.zeros(h, np.int8)
    offsets = []
    for k in (0, 1, -1, 2, -2, 3, -3):
        for m in (0, 1, -1, 2, -2, 3, -3, 4, -4):
            if (k, m) == (0, 0):
                continue
            if k and storey_m <= 0 or m and bay_m <= 0:
                continue
            offsets.append((abs(k) * 1.5 + abs(m), k, m))
    offsets.sort()
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    good = ~hole
    for _, k, m in offsets:
        dr = int(round(-k * storey_m / m_per_px))  # k storeys up is rows above
        dc = int(round(m * bay_m / m_per_px))
        src_r = rows + dr
        src_c = cols + dc
        valid = (src_r >= 0) & (src_r < h) & (src_c >= 0) & (src_c < w)
        src_r_c = np.clip(src_r, 0, h - 1)
        src_c_c = np.clip(src_c, 0, w - 1)
        same_zone = zone[np.clip(rows[:, 0] + dr, 0, h - 1)][:, None] == zone[:, None]
        take = hole & valid & good[src_r_c, src_c_c] & same_zone
        if not take.any():
            continue
        rgb[take] = rgb[src_r_c, src_c_c][take]
        weight[take] = 0.6
        hole &= ~take
        if not hole.any():
            break
    return rgb, weight


def generated_wall(
    shape: tuple[int, int],
    *,
    m_per_px: float,
    width_m: float,
    z_of_row: np.ndarray,
    style,
    tile_width_m: float,
    storey_base_z: float,
    storey_m: float,
    ground_style=None,
    ground_z: float | None = None,
    ground_height_m: float = 3.6,
    size_px: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """The wall as the generator would draw it, at this picture's pixels.

    The same grid the model's UVs use: a whole number of bays across the
    wall, storeys counted from their base, and the ground storey in its own
    style below it.
    """
    from .facade import render_facade_layers

    h, w = shape
    tile, _ = render_facade_layers(style, size_px, seed=seed)
    bays = max(1, int(round(width_m / max(tile_width_m, 0.5))))
    bay = width_m / bays
    u = (np.arange(w) + 0.5) * m_per_px
    tu = np.minimum((np.mod(u, bay) / bay * size_px).astype(np.int64), size_px - 1)
    tv = np.minimum((np.mod(z_of_row - storey_base_z, storey_m) / storey_m * size_px).astype(np.int64), size_px - 1)
    out = tile[tv[:, None], tu[None, :]].astype(np.float32)
    if ground_style is not None and ground_z is not None:
        ground, _ = render_facade_layers(ground_style, size_px, seed=seed + 1)
        below = z_of_row < ground_z + ground_height_m
        gv = np.minimum((np.clip(z_of_row - ground_z, 0, None) / ground_height_m * size_px).astype(np.int64), size_px - 1)
        out[below] = ground[gv[below][:, None], tu[None, :]]
    return out


def blend_fill(rgb: np.ndarray, weight: np.ndarray, fill: np.ndarray, m_per_px: float) -> np.ndarray:
    """Lay `fill` under the picture where it is weak, with a soft edge.

    Nothing of a hole shows through: a reading behind a parked car keeps a
    fifth of its weight, and a fifth of a car is a ghost of one, floating on
    the wall. The fill takes over completely below the hole weight.
    """
    trust = np.clip((weight - HOLE_WEIGHT) / 0.3, 0.0, 1.0)
    trust = _box_blur(trust, max(1, int(round(0.15 / m_per_px))))
    return (rgb * trust[..., None] + fill * (1.0 - trust[..., None])).astype(np.float32)


# --------------------------------------------------------------------------
# Atlas
# --------------------------------------------------------------------------


def pack(sizes: list[tuple[int, int]], page_px: int = PAGE_PX, pad: int = PAD_PX):
    """Shelf-pack rectangles onto square pages, tallest first.

    Returns ``(placements [(page, x, y)], scales)``: a rectangle larger than
    a page is scaled down to fit, and its scale returned.
    """
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0]))
    result: list = [None] * len(sizes)
    scales = [1.0] * len(sizes)
    page, x, y, shelf = 0, 0, 0, 0
    for i in order:
        w, h = sizes[i]
        scale = min(1.0, (page_px - 2 * pad) / max(w, h, 1))
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
        scales[i] = scale
        if x + w + 2 * pad > page_px:
            x, y, shelf = 0, y + shelf, 0
        if y + h + 2 * pad > page_px:
            page, x, y, shelf = page + 1, 0, 0, 0
        result[i] = (page, x + pad, y + pad)
        x += w + 2 * pad
        shelf = max(shelf, h + 2 * pad)
    return result, scales


def _padded_paste(page: np.ndarray, image: np.ndarray, x: int, y: int, pad: int) -> None:
    """Paste with the edge pixels repeated into the padding, so filtering
    at a wall's edge never reaches its neighbour on the page."""
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    h, w = padded.shape[:2]
    ph, pw = page.shape[:2]
    y0, x0 = y - pad, x - pad
    page[max(0, y0):min(ph, y0 + h), max(0, x0):min(pw, x0 + w)] = padded[
        max(0, -y0):h - max(0, y0 + h - ph), max(0, -x0):w - max(0, x0 + w - pw)
    ]


# --------------------------------------------------------------------------
# The bake
# --------------------------------------------------------------------------


def bake_textures(
    result,
    coverage,
    buildings,
    work_dir: Path,
    out_dir: Path,
    *,
    photos_cfg: dict,
    observed=None,
    n_variants: int = 5,
    faces_index=None,
    crowns=None,
    parking=None,
    session=None,
) -> PhotoTextureSet:
    """Make the photo textures for every wall that can wear one."""
    from .facade import GROUND_STYLES, STYLES, style_for_building
    from .photo_colour import wall_gain

    textures = PhotoTextureSet()
    setting = str(photos_cfg.get("textures", "own"))
    by_id = {p.id: p for p in result.photos}
    chosen = texture_walls(coverage, by_id, setting, int(photos_cfg.get("textures_max_walls", 300)))
    if not chosen:
        LOG.info("photo textures: no wall can wear a photo under photos.textures=%r", setting)
        return textures

    # Full-size pixels for the photos used: a 2048 px 360 photo gives a wall
    # ten metres off a third of what the original does.
    texture_paths = _fetch_texture_pixels(
        [by_id[pid] for _, views in chosen for pid in views], work_dir, photos_cfg, session
    )

    measured_walls = {
        w.face: w for m in result.facades.values() for w in m.walls
    } if getattr(result, "facades", None) else {}
    pixels = PhotoPixels(max_cached=3)
    images, keys, faces_used = [], [], []
    m_per_px_of = {}
    for face_index, views in chosen:
        face = coverage.faces[face_index]
        building = buildings.buildings[face.building]
        photos = [by_id[pid] for pid in views if pid in texture_paths]
        if not photos:
            continue
        best = max(
            (v.px_per_m for v in coverage.views[face_index] if v.photo in views), default=MIN_PX_PER_M
        )
        m_per_px = 1.0 / float(np.clip(best, MIN_PX_PER_M, float(photos_cfg.get("texture_max_px_per_m", MAX_PX_PER_M))))
        wall = capture_wall(
            face, face_index, photos, pixels,
            m_per_px=m_per_px, faces_index=faces_index, crowns=crowns,
            target_px_per_m=0.6 / m_per_px, path_of=lambda p: texture_paths.get(p.id, ""),
            mode="best", parking=parking, street_z=building.ground_z_nap,
        )
        if wall is None:
            continue
        if wall.agreement is not None:
            # Translucent cars on a ground floor are two photos' different
            # cars averaged; better a hole the rhythm or the generated
            # ground storey can fill.
            z_rows = wall.z_of_row(np.arange(wall.rgb.shape[0]))
            street = (z_rows < building.ground_z_nap + STREET_BAND_M)[:, None]
            wall.weight = np.where(street & (wall.agreement < DISAGREE), wall.weight * 0.2, wall.weight)
        measured = measured_walls.get(face_index)
        gains = [result.gains[pid] for pid in wall.views if pid in getattr(result, "gains", {})]
        # How far this wall may be lifted is asked of its colour as
        # photographed: the measured colour, with its own lift undone, or the
        # median of what the photos saw.
        if measured is not None and measured.wall_rgb:
            colour = tuple(c / max(measured.gain, 1e-3) for c in measured.wall_rgb)
        elif (wall.weight > HOLE_WEIGHT).any():
            colour = tuple(np.median(wall.rgb[wall.weight > HOLE_WEIGHT], axis=0))
        else:
            colour = (128.0, 128.0, 128.0)
        gain = wall_gain(
            colour, float(np.median(gains)) if gains else 1.0,
            strength=float(photos_cfg.get("brightness_strength", 0.7)),
        )
        wall.rgb = np.clip(wall.rgb * gain, 0, 255)
        rgb = flatten_light(wall.rgb, wall.weight, m_per_px,
                            strength=float(photos_cfg.get("texture_flatten_light", 0.5)))
        seen_share = float((wall.weight > HOLE_WEIGHT).mean())

        z_rows = wall.z_of_row(np.arange(rgb.shape[0]))
        split = building.wall_height_m > 3.6 * 1.2 + 2.4
        ground_top = building.ground_z_nap + 3.6 if split else None
        bay = measured.bay_m if measured is not None else 0.0
        storey = measured.storey_m if measured is not None else 0.0
        rgb, weight = fill_from_rhythm(
            rgb, wall.weight, m_per_px=m_per_px, bay_m=bay, storey_m=storey,
            z_of_row=z_rows, ground_top_z=ground_top,
        )
        if (weight < HOLE_WEIGHT).any():
            era = STYLES[style_for_building(building.height_m, building.build_year, max(1, n_variants))]
            style = era
            if observed is not None:
                look = observed.face_look.get(face.key, observed.look_of.get(face.building))
                if look is not None:
                    style = observed.styles[look]
            base = ground_top if ground_top is not None else building.ground_z_nap
            floors = max(1, int(building.floors) - (1 if split else 0))
            fill = generated_wall(
                rgb.shape[:2], m_per_px=m_per_px, width_m=face.width_m, z_of_row=z_rows,
                style=style,
                tile_width_m=(bay * (1 if style.windows_across == 1 else style.windows_across)) if bay > 0 else float(photos_cfg.get("fallback_tile_m", 4.0)),
                storey_base_z=base,
                storey_m=storey if storey > 0 else max(1.5, (building.wall_top_nap - base) / floors),
                ground_style=GROUND_STYLES[1 if building.function in (1, 4) else 0] if split else None,
                ground_z=building.ground_z_nap if split else None,
            )
            rgb = blend_fill(rgb, weight, fill, m_per_px)
        images.append(np.clip(rgb, 0, 255).astype(np.uint8))
        keys.append(face.key)
        faces_used.append((face_index, face))
        m_per_px_of[face.key] = m_per_px
        textures.sources[face.key] = list(wall.views)
        textures.seen_share[face.key] = seen_share

    if not images:
        return textures
    page_px = int(photos_cfg.get("texture_page_px", PAGE_PX))
    spots, scales = pack([(im.shape[1], im.shape[0]) for im in images], page_px)
    n_pages = max(s[0] for s in spots) + 1
    pages = [np.zeros((page_px, page_px, 3), dtype=np.uint8) for _ in range(n_pages)]
    from PIL import Image

    for image, key, (face_index, face), (page, x, y), scale in zip(images, keys, faces_used, spots, scales):
        if scale < 1.0:
            image = np.asarray(Image.fromarray(image).resize(
                (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), Image.LANCZOS
            ))
        _padded_paste(pages[page], image, x, y, PAD_PX)
        tangent = (face.end - face.start) / max(face.width_m, 1e-9)
        textures.placements[key] = Placement(
            page=page, x=x, y=y, width=image.shape[1], height=image.shape[0],
            start=(float(face.start[0]), float(face.start[1])),
            tangent=(float(tangent[0]), float(tangent[1])),
            z_max=float(face.z_max),
            m_per_px_u=float(face.width_m) / image.shape[1],
            m_per_px_z=float(face.height_m) / image.shape[0],
        )
    # Trim each page to what it holds, rounded up to a multiple of 256.
    for index, page in enumerate(pages):
        used_h = max(
            (p.y + p.height + PAD_PX for p in textures.placements.values() if p.page == index), default=256
        )
        height = int(min(page_px, int(np.ceil(used_h / 256.0)) * 256))
        path = work_dir / f"photo_wall_{index:02d}.jpg"
        Image.fromarray(page[:height]).save(path, quality=90)
        textures.pages.append(path)
        textures.page_sizes.append((page_px, height))

    _write_credits(textures, by_id, out_dir)
    LOG.info(
        "photo textures: %d walls on %d pages from %d photos; median %.0f%% of "
        "each wall photographed, the rest filled",
        len(textures.placements), len(textures.pages), textures.stats()["photos"],
        100 * textures.stats()["median_photographed_share"],
    )
    return textures


def _fetch_texture_pixels(photos: list[Photo], work_dir: Path, photos_cfg: dict, session) -> dict:
    """Photo id -> file holding its pixels at texture resolution."""
    out = {}
    wanted = {p.id: p for p in photos}
    own = [p for p in wanted.values() if p.source == SOURCE_OWN and p.path]
    for photo in own:
        out[photo.id] = photo.path
    mly = [p for p in wanted.values() if p.source == SOURCE_MAPILLARY]
    if not mly:
        return out
    from .http_util import ServiceError
    from .mapillary import download_images, find_token, session_for

    sizes = {
        "spherical": int(photos_cfg.get("pano_image_size", 0)),
        "perspective": int(photos_cfg.get("image_size", 2048)),
        "fisheye": int(photos_cfg.get("image_size", 2048)),
    }
    try:
        http = session or session_for(find_token())
    except ServiceError as exc:
        LOG.warning("photo textures: no Mapillary token for full-size images (%s)", exc)
        http = None
    for kind, size in sizes.items():
        group = [p for p in mly if p.camera.kind == kind]
        if not group:
            continue
        if http is None:
            fetched = [p for p in group if p.path]
        else:
            try:
                fetched = download_images(
                    group, work_dir / "_mapillary" / "images", session=http, size=size,
                    timeout=float(photos_cfg.get("timeout_s", 60.0)),
                    max_retries=int(photos_cfg.get("max_retries", 4)),
                )
            except ServiceError as exc:
                LOG.warning("photo textures: full-size images unavailable (%s); using what is cached", exc)
                fetched = [p for p in group if p.path]
        for photo in fetched:
            if photo.path:
                out[photo.id] = photo.path
    return out


def _write_credits(textures: PhotoTextureSet, photos: dict, out_dir: Path) -> None:
    """Which photos each texture page was made from, and whose they are."""
    used = sorted({pid for ids in textures.sources.values() for pid in ids})
    rows = []
    for pid in used:
        photo = photos.get(pid)
        if photo is None:
            continue
        rows.append({
            "photo": pid,
            "source": photo.source,
            "creator": photo.creator,
            "licence": photo.licence or ("your own" if photo.source == SOURCE_OWN else ""),
            "captured_at": photo.captured_at,
        })
    textures.credits = {"photos": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "photo_credits.json").write_text(json.dumps(textures.credits, indent=1), encoding="utf-8")


def triangle_photo_uvs(triangles: np.ndarray, keys: np.ndarray, textures: PhotoTextureSet):
    """Per triangle: the atlas page (-1 for none) and its corners' UVs.

    `keys` are the triangles' face keys, (n, 3) rows as from
    facade_uv.wall_face_keys.
    """
    page = np.full(len(triangles), -1, dtype=np.int32)
    uv = np.zeros((len(triangles), 3, 2), dtype=np.float32)
    if not textures.placements or not len(triangles):
        return page, uv
    lookup = textures.placements
    for i, key in enumerate(map(tuple, keys.tolist())):
        placement = lookup.get(key)
        if placement is None:
            continue
        page[i] = placement.page
        uv[i] = placement.uv(triangles[i], textures.page_sizes[placement.page])
    return page, uv


__all__ = [
    "Placement",
    "PhotoTextureSet",
    "bake_textures",
    "blend_fill",
    "fill_from_rhythm",
    "flatten_light",
    "generated_wall",
    "may_texture",
    "pack",
    "texture_walls",
    "triangle_photo_uvs",
]
