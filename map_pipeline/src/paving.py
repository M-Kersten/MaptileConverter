"""Photographed paving for the layered ground, coloured from the aerial.

The layered ground draws every road, walkway and lawn in one flat colour, the
way the modellers' hand-made file does. From a street that reads as a board
game: the Ganzenmarkt in Utrecht is red-brown brick in a herringbone from wall
to wall, and flat mauve with yellow pavements is nothing like it.

The BGT says what each surface is paved with (`layered.paving_material`), and
Poly Haven has each of those photographed, CC0, with the real size it covers:
brick in a herringbone, setts, 30 cm concrete tiles, concrete pavers in half
bond, asphalt, gravel, lawn. Each material gets its photograph at true size,
with its normal map, so the joints catch the light.

Then the colour. A photograph of brick is somebody else's brick, and Dutch
paving comes in every shade from salmon to purple-grey. The aerial has the
real one: the median of the aerial over every surface in that material, pulled
part of the way from a typical colour for it, since an aerial flown in late
winter through haze reads duller than the street. Each channel is scaled, so
every brick, joint and stain stays where the photograph has it.

Nothing here is required: with no network the ground keeps its flat colours.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .textures import LICENCE, TextureUnavailable, _tint, fetch_texture, texture_size_m

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paving:
    """What one ground material looks like up close."""

    slug: str  # the Poly Haven texture
    rgb: tuple[int, int, int]  # a typical colour for it, sRGB
    roughness: float
    aerial_share: float = 0.6  # how far the aerial's colour wins over `rgb`


# By the layered ground's material names (`layered.MATERIALS`). Beton, the
# kerb, stays plain: the curb generator builds it and lays its own UVs.
PAVINGS: dict[str, Paving] = {
    # Gebakken klinkers in a herringbone, 21 x 10 cm: the Dutch street.
    "Bakstenen": Paving("herringbone_pavement_03", (122, 78, 62), 0.85),
    # Setts. "Sierbestrating" is anything ornamental, mostly natural stone.
    "Sierbestrating": Paving("square_cobblestone", (124, 116, 104), 0.85),
    # The 30 x 30 cm stoeptegel.
    "Tegels": Paving("concrete_tiles", (150, 148, 142), 0.8),
    # Betonstraatstenen, 21 x 10.5 cm, in half bond.
    "Betonstenen": Paving("concrete_pavers_02", (130, 126, 118), 0.85),
    # A walkway the BGT gives no paving for is most likely tiles too.
    "Beton_Geel": Paving("concrete_tiles", (150, 148, 142), 0.8),
    "Verharding": Paving("concrete_floor_01", (138, 136, 130), 0.85),
    "Asfalt": Paving("clean_asphalt", (72, 72, 74), 0.9),
    # Red asphalt, and only the cycle path is.
    "Fietspad": Paving("clean_asphalt", (142, 66, 54), 0.9, aerial_share=0.4),
    "Grind": Paving("gravel_floor", (160, 148, 124), 0.95),
    # Lawn. The aerial is flown in late winter, when grass is brown: half of it.
    "Gras": Paving("grass_ground", (84, 104, 52), 1.0, aerial_share=0.3),
}


@dataclass
class PavingTexture:
    material: str
    colour: Path
    normal: Path | None
    tile_m: float  # metres of ground the texture covers, each way
    roughness: float
    rgb: tuple[int, int, int]  # the colour it was given
    origin: str = "typical"  # where that colour came from


def aerial_colours(
    areas, aerial_path: Path, world_path: Path, *, max_samples: int = 20000, seed: int = 0
) -> dict[str, tuple[int, int, int]]:
    """The median aerial colour under each material's surfaces, sRGB.

    Sampled at random points inside the BGT polygons, the outer half metre left
    out: a polygon's edge is where the kerb, the gutter and the next surface
    are.
    """
    import shapely
    from PIL import Image

    from .terrain_mesh import _polygonal

    if not Path(aerial_path).is_file() or not Path(world_path).is_file():
        return {}
    a, _, _, e, c, f = [float(v) for v in Path(world_path).read_text().split()[:6]]
    with Image.open(aerial_path) as image:
        pixels = np.asarray(image.convert("RGB"))
    height, width = pixels.shape[:2]
    by_material: dict[str, list] = {}
    for area in areas:
        if getattr(area, "level", 0) != 0:
            continue
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in area.rings]
        rings = [r for r in rings if len(r) >= 4]
        if not rings:
            continue
        polygon = _polygonal(shapely.make_valid(shapely.Polygon(rings[0], rings[1:])))
        inner = polygon.buffer(-0.5)
        if not inner.is_empty:
            by_material.setdefault(area.material, []).append(inner)
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[int, int, int]] = {}
    for material, polygons in by_material.items():
        union = shapely.union_all(np.array(polygons, dtype=object))
        if union.is_empty or union.area < 4.0:
            continue
        xmin, ymin, xmax, ymax = union.bounds
        n = int(min(max_samples, max(500, union.area * 4)))
        xs = rng.uniform(xmin, xmax, n * 3)
        ys = rng.uniform(ymin, ymax, n * 3)
        inside = shapely.contains_xy(union, xs, ys)
        xs, ys = xs[inside][:n], ys[inside][:n]
        cols = ((xs - c) / a).astype(np.int64)
        rows = ((ys - f) / e).astype(np.int64)
        ok = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        if ok.sum() < 50:
            continue
        samples = pixels[rows[ok], cols[ok]].astype(np.float64)
        # Leave out what is no paving: shade the deshadowing missed, white
        # markings, a parked car's roof.
        brightness = samples.mean(axis=1)
        low, high = np.percentile(brightness, [15, 85])
        keep = (brightness >= low) & (brightness <= high)
        out[material] = tuple(int(round(v)) for v in np.median(samples[keep], axis=0))
    return out


def _material_lookup(areas, inset_m: float = 0.4):
    """``f(x, y) -> material index or -1`` over the BGT areas, and the names.

    Each polygon is shrunk by `inset_m`: a point on the edge between brick
    and asphalt says nothing about either.
    """
    import shapely

    from .terrain_mesh import _polygonal

    by_material: dict[str, list] = {}
    for area in areas:
        if getattr(area, "level", 0) != 0:
            continue
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in area.rings]
        rings = [r for r in rings if len(r) >= 4]
        if not rings:
            continue
        polygon = _polygonal(shapely.make_valid(shapely.Polygon(rings[0], rings[1:])))
        inner = polygon.buffer(-inset_m)
        if not inner.is_empty:
            by_material.setdefault(area.material, []).append(inner)
    names = sorted(by_material)
    unions = []
    for name in names:
        union = shapely.union_all(np.array(by_material[name], dtype=object))
        shapely.prepare(union)
        unions.append(union)

    def lookup(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        out = np.full(len(x), -1, dtype=np.int64)
        for index, union in enumerate(unions):
            free = out < 0
            if not free.any():
                break
            hit = shapely.contains_xy(union, x[free], y[free])
            out[np.flatnonzero(free)[hit]] = index
        return out

    return lookup, names


def photo_colours(
    photos,
    areas,
    ground_at,
    pixels,
    *,
    faces_index=None,
    crowns=None,
    max_photos: int = 160,
    per_photo: int = 300,
    min_samples: int = 300,
    min_photos: int = 3,
) -> dict[str, dict]:
    """The paving's colour as the street photos see it, per material.

    The aerial sees Utrecht's red-brown brick as grey, (129, 125, 119): a
    winter flight through haze, eight centimetres a pixel, half of each brick
    its sandy joint. The photos taken from the street see it the colour it
    is. Around each checked photo the ground is sampled in a ring 3.5 to 14 m
    out, on the survey, and every point that a wall or a tree crown hides is
    left out; what is left is looked up in the BGT for what it is paved with.
    Each photo gives each material at most `per_photo` points, so one
    sequence of a hundred photos down one street does not decide a town.

    Returns per material ``{"rgb", "samples", "photos"}``, sRGB, for the
    materials seen at least `min_samples` times in `min_photos` photos.
    """
    from .camera import SPHERICAL, project_world
    from .facade_capture import _bilinear, luminance
    from .photo_colour import PANO_FLOOR_DEG, ground_ring

    lookup, names = _material_lookup(areas)
    if not names:
        return {}
    usable = [p for p in photos if p.verified and p.path]
    if len(usable) > max_photos:
        # Spread over the area rather than the first sequence's worth.
        order = np.random.default_rng(0).permutation(len(usable))[:max_photos]
        usable = [usable[i] for i in sorted(order)]
    samples: dict[int, list] = {}
    seen_by: dict[int, int] = {}
    for photo in usable:
        try:
            image = pixels.get(photo.path)
        except OSError:
            continue
        points = ground_ring(photo, ground_at, step_m=0.5, azimuth_step_deg=3.0)
        camera = photo.camera.scaled(image.shape[1], image.shape[0])
        px, ok = project_world(camera, photo.rotation, photo.centre, points)
        if camera.kind == SPHERICAL:
            horizontal = np.linalg.norm(points[:, :2] - photo.centre[:2], axis=1)
            elevation = np.degrees(np.arctan2(points[:, 2] - photo.centre[2], horizontal))
            ok &= elevation > PANO_FLOOR_DEG + 2.0
        else:
            # The bottom of a dashcam's frame is its own bonnet.
            ok &= px[:, 1] < 0.88 * image.shape[0]
        if crowns:
            for crown in crowns:
                ok &= np.hypot(points[:, 0] - crown.x, points[:, 1] - crown.y) > crown.radius + 0.5
        if faces_index is not None and ok.any():
            candidates = faces_index.near(photo.x, photo.y, 16.0)
            hidden = np.zeros(len(points), dtype=bool)
            hidden[ok] = faces_index.blocked(
                photo.centre, points[ok], candidates, np.full(int(ok.sum()), -1)
            )
            ok &= ~hidden
        if not ok.any():
            continue
        material = lookup(points[ok, 0], points[ok, 1])
        rgb = _bilinear(image, px[ok, 0], px[ok, 1], wrap=camera.kind == SPHERICAL)
        good = (material >= 0) & (rgb.min(axis=1) > 6) & (rgb.max(axis=1) < 250)
        for index in np.unique(material[good]):
            picked = rgb[good & (material == index)]
            if len(picked) < 20:
                continue
            if len(picked) > per_photo:
                picked = picked[np.random.default_rng(len(picked)).choice(len(picked), per_photo, replace=False)]
            samples.setdefault(int(index), []).append(picked)
            seen_by[int(index)] = seen_by.get(int(index), 0) + 1
    out: dict[str, dict] = {}
    for index, chunks in samples.items():
        if seen_by.get(index, 0) < min_photos:
            continue
        values = np.concatenate(chunks).astype(np.float64)
        if len(values) < min_samples:
            continue
        # The lit half of the paving: shade is the light, not the stone.
        lum = luminance(values)
        low, high = np.percentile(lum, [40, 92])
        lit = values[(lum >= low) & (lum <= high)]
        out[names[index]] = {
            "rgb": [int(round(v)) for v in np.median(lit, axis=0)],
            "samples": int(len(values)),
            "photos": int(seen_by[index]),
        }
    if out:
        LOG.info(
            "paving as the street photos see it: %s",
            "; ".join(f"{k} {v['rgb']} ({v['photos']} photos)" for k, v in sorted(out.items())),
        )
    return out


def target_colours(
    materials,
    *,
    photo: dict[str, dict] | None = None,
    aerial: dict[str, tuple[int, int, int]] | None = None,
    strength: float = 0.7,
    ceiling: float = 185.0,
) -> dict[str, tuple[tuple[int, int, int], str]]:
    """The colour each material's photograph is given, and where it came from.

    The street photos give the hue; the aerial, the model's one standard for
    brightness, gives the brightness, softened by `strength` and never above
    `ceiling`, as the walls are (`photo_colour`). Without a street photo the
    hue is the material's typical one (`PAVINGS`).
    """
    from .facade_capture import luminance

    photo = photo or {}
    aerial = aerial or {}
    out: dict[str, tuple[tuple[int, int, int], str]] = {}
    for material in materials:
        paving = PAVINGS.get(material)
        if paving is None:
            continue
        if material in photo:
            base, source = np.asarray(photo[material]["rgb"], dtype=np.float64), "street photos"
        else:
            base, source = np.asarray(paving.rgb, dtype=np.float64), "typical"
            if material in aerial and paving.aerial_share > 0:
                # Part of the way to what the aerial sees: a typical brick
                # brightened to Bourtange's aerial came out orange.
                share = 0.8 * paving.aerial_share
                base = base * (1.0 - share) + np.asarray(aerial[material], dtype=np.float64) * share
                source = "typical, toward the aerial"
        if material in aerial and paving.aerial_share > 0:
            ratio = float(luminance(np.asarray(aerial[material], dtype=np.float64)[None])[0]) / max(
                float(luminance(base[None])[0]), 1.0
            )
            gain = ratio ** (strength * paving.aerial_share / 0.6)
            base = base * gain
            source += ", brightness from the aerial"
            top = float(base.max())
            if top > ceiling:
                base = base * (ceiling / top)
        out[material] = (tuple(int(round(v)) for v in np.clip(base, 0, 255)), source)
    return out


def build_paving_textures(
    materials: list[str],
    work_dir: Path,
    *,
    targets: dict[str, tuple[tuple[int, int, int], str]] | None = None,
    timeout_s: float = 60.0,
) -> dict[str, PavingTexture]:
    """A coloured photograph and a normal map for each material that has one.

    Written to the work directory as ``paving_<material>.jpg`` and
    ``paving_<material>_normal.jpg``. A material whose photograph cannot be
    had is left out, and keeps its flat colour.
    """
    from PIL import Image

    targets = targets or {}
    made: dict[str, PavingTexture] = {}
    missed: list[str] = []
    for material in materials:
        paving = PAVINGS.get(material)
        if paving is None:
            continue
        try:
            source = fetch_texture(paving.slug, work_dir, timeout_s=timeout_s)
            width_m, height_m = texture_size_m(paving.slug, work_dir, timeout_s=timeout_s)
        except (TextureUnavailable, Exception) as exc:  # noqa: BLE001 - optional
            missed.append(f"{material} ({exc})")
            continue
        try:
            normal_source = fetch_texture(paving.slug, work_dir, timeout_s=timeout_s, kind="normal")
        except Exception:  # noqa: BLE001 - a flat surface is still a surface
            normal_source = None

        target_rgb, origin = targets.get(material, (paving.rgb, "typical"))
        target = np.asarray(target_rgb, dtype=np.float64)
        with Image.open(source) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
        coloured = _tint(pixels, tuple(target), 1.0)
        colour_path = Path(work_dir) / f"paving_{material}.jpg"
        Image.fromarray(np.clip(coloured, 0, 255).astype(np.uint8)).save(colour_path, quality=92)
        normal_path = None
        if normal_source is not None:
            normal_path = Path(work_dir) / f"paving_{material}_normal{normal_source.suffix}"
            shutil.copyfile(normal_source, normal_path)
        made[material] = PavingTexture(
            material=material,
            colour=colour_path,
            normal=normal_path,
            tile_m=float(max(0.5, (width_m + height_m) / 2.0)),
            roughness=paving.roughness,
            rgb=tuple(int(round(v)) for v in target),
            origin=origin,
        )
    if made:
        LOG.info(
            "paving: %s",
            "; ".join(
                f"{t.material} {PAVINGS[t.material].slug} {t.tile_m:.1f} m, rgb {t.rgb} ({t.origin})"
                for t in made.values()
            ),
        )
    if missed:
        LOG.warning("paving: no photograph for %s; flat colour instead", ", ".join(missed))
    return made


def scene_entry(textures: dict[str, PavingTexture]) -> dict:
    """What the Blender stage reads: per material, its files and its size."""
    return {
        name: {
            "colour": t.colour.name,
            "normal": t.normal.name if t.normal else None,
            "tile_m": round(t.tile_m, 4),
            "roughness": t.roughness,
            "rgb": list(t.rgb),
            "colour_from": t.origin,
        }
        for name, t in textures.items()
    }


__all__ = [
    "LICENCE",
    "PAVINGS",
    "Paving",
    "PavingTexture",
    "aerial_colours",
    "build_paving_textures",
    "photo_colours",
    "scene_entry",
    "target_colours",
]
