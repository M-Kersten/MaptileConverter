"""What tells a driver what to do: signs, lights, paint and the lids in the road.

The BGT draws the carriageway and stops there. A street built from it alone is
grey from kerb to kerb with eight blank sign posts on it, where the same square
kilometre of Utrecht centre has 695 traffic signs, 24 marked crossings, a dozen
sets of lights, 31 speed humps and three thousand drain covers. Four more
sources fill that in:

* **Traffic signs** from NDW's national sign register (CC0): every sign with
  its RVV code, the road section it belongs to and which way its traffic
  drives. Each becomes a plate of the right shape with the right face on it.
* **Lanes, crossings and humps** from OpenStreetMap (ODbL): the lane count
  becomes lane lines, a crossing its zebra or its dashed edges.
* **Traffic lights** from the BGT's own signal poles, which used to be drawn
  as one more blank sign post. Where a municipality did not survey them, the
  junctions OpenStreetMap tags as signalled get a pole per approach.
* **Manholes and gullies** from the BGT's ``put`` collection.

Give-way teeth and stop lines are not in any register, but the signs that
require them are: every B6 gets its row of teeth across the lane it governs,
every B7 its line.

Nothing here measures paint. A lane line is put where the lane count says it
must be, between kerbs the BGT surveyed, and left out wherever the carriageway
is not the width that count implies -- so the lines stop at a junction rather
than running across it.

The meshes are finished on this side and handed to Blender as plain arrays, so
everything in this file runs, and is tested, without Blender.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .geo import BBox, RD_CRS, WGS84_CRS, _transformer
from .http_util import get_with_retry, short_error

LOG = logging.getLogger(__name__)

NDW_SIGNS_URL = (
    "https://data.ndw.nu/api/rest/static-road-data/traffic-signs/v4/current-state"
)
MUNICIPALITY_URL = (
    "https://api.pdok.nl/kadaster/bestuurlijkegebieden/ogc/v1/"
    "collections/gemeentegebied/items"
)
RD_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"

# --------------------------------------------------------------------------
# The atlas
# --------------------------------------------------------------------------

ATLAS_GRID = 16
ATLAS_CELL_PX = 128
# How much of a cell a sign face fills. The rest is a border in the face's own
# colour, so a mip level that bleeds across the edge bleeds nothing visible.
FACE_FILL = 0.9

# Flat colours, one cell each, addressed by their centre.
FLAT_CELLS: dict[str, tuple[int, tuple[int, int, int]]] = {
    "white": (0, (236, 236, 230)),
    "yellow": (1, (238, 190, 40)),
    "cover": (2, (52, 50, 48)),
    "grate": (3, (40, 40, 42)),
    "housing": (4, (22, 22, 24)),
    "red": (5, (225, 40, 30)),
    "amber": (6, (245, 165, 25)),
    "green": (7, (40, 190, 80)),
    "post": (8, (150, 154, 158)),
    "back": (9, (120, 124, 128)),
}
FIRST_FACE_CELL = 16

RED = (200, 16, 30)
BLUE = (0, 84, 166)
WHITE = (250, 250, 250)
BLACK = (20, 20, 20)
ORANGE = (245, 160, 20)

# Unit outlines, counter-clockwise, in a frame with +u to the viewer's right
# and +v up. The same outline cuts the plate and places its face in the atlas.
SHAPES: dict[str, np.ndarray] = {
    "disc": np.array(
        [[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    ),
    "octagon": np.array(
        [
            [np.cos(a), np.sin(a)]
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
        ]
    ),
    "square": np.array([[-0.92, -0.92], [0.92, -0.92], [0.92, 0.92], [-0.92, 0.92]]),
    "diamond": np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
    "tri_up": np.array([[-1.0, -0.8], [1.0, -0.8], [0.0, 0.93]]),
    "tri_down": np.array([[-1.0, 0.8], [0.0, -0.93], [1.0, 0.8]]),
}
# Half the plate's extent in metres: a 60 cm disc, a 90 cm triangle.
SHAPE_HALF_M = {
    "disc": 0.30,
    "octagon": 0.33,
    "square": 0.30,
    "diamond": 0.36,
    "tri_up": 0.45,
    "tri_down": 0.45,
}

# code -> (shape, recipe, glyph). A recipe is how the face is painted, a glyph
# what is drawn on it. Signs not listed fall back on their series letter, which
# gets shape and colours right and leaves the pictogram out.
SIGN_FACES: dict[str, tuple[str, str, str]] = {
    "A1": ("disc", "ring", ""),
    "A1-15": ("disc", "ring", "15"),
    "A1-20": ("disc", "ring", "20"),
    "A1-30": ("disc", "ring", "30"),
    "A1-50": ("disc", "ring", "50"),
    "A1-60": ("disc", "ring", "60"),
    "A1-70": ("disc", "ring", "70"),
    "A1-80": ("disc", "ring", "80"),
    "A1-100": ("disc", "ring", "100"),
    "B1": ("diamond", "priority", ""),
    "B6": ("tri_down", "yield", ""),
    "B7": ("octagon", "stop", "STOP"),
    "C1": ("disc", "ring", ""),
    "C2": ("disc", "no_entry", ""),
    "C3": ("square", "blue", "arrow_up"),
    "C4": ("square", "blue", "arrow_right"),
    "C6": ("disc", "ring", "car"),
    "C7": ("disc", "ring", "truck"),
    "C12": ("disc", "ring", "car"),
    "C15": ("disc", "ring", "bike"),
    "C16": ("disc", "ring", "walker"),
    "D1": ("disc", "blue", "roundabout"),
    "D2": ("disc", "blue", "arrow_down_right"),
    "D3": ("disc", "blue", "arrow_down_both"),
    "D4": ("disc", "blue", "arrow_up"),
    "D5": ("disc", "blue", "arrow_right"),
    "E1": ("disc", "no_parking", "1"),
    "E2": ("disc", "no_parking", "2"),
    "E3": ("disc", "no_parking", "1"),
    "E4": ("square", "blue", "P"),
    "G7": ("disc", "blue", "walker"),
    "G11": ("disc", "blue", "bike"),
    "G12a": ("disc", "blue", "bike"),
    "G13": ("disc", "blue", "bike"),
    "L2": ("square", "blue", "zebra"),
    "L8": ("square", "blue", "dead_end"),
    # Series fallbacks.
    "A": ("disc", "ring", ""),
    "B": ("tri_up", "warning", "junction"),
    "C": ("disc", "ring", ""),
    "D": ("disc", "blue", "arrow_up"),
    "E": ("square", "blue", "P"),
    "F": ("disc", "ring", ""),
    "G": ("square", "blue", ""),
    "H": ("square", "blue", ""),
    "J": ("tri_up", "warning", "!"),
    "K": ("square", "blue", ""),
    "L": ("square", "blue", ""),
}
_FACE_ORDER = list(SIGN_FACES)


# Road symbols, white on a transparent cell, read through an alpha-cut
# material: a symbol drawn opaque would cover the asphalt round it with a
# square of grey.
SYMBOLS = ("bike", "bus", "arrow")


def symbol_cell(name: str) -> int:
    return FIRST_FACE_CELL + len(SIGN_FACES) + SYMBOLS.index(name)


def symbol_uv(name: str) -> np.ndarray:
    """Per-corner UVs for a symbol quad, in `GroundPaint.quad`'s corner order:
    near right, far right, far left, near left."""
    u0, v0, u1, v1 = cell_rect(symbol_cell(name))
    inset = 0.5 / (ATLAS_GRID * ATLAS_CELL_PX)
    u0, v0, u1, v1 = u0 + inset, v0 + inset, u1 - inset, v1 - inset
    return np.array([[u1, v0], [u1, v1], [u0, v1], [u0, v0]])


def face_key(code: str, speed: str = "") -> str | None:
    """Which atlas face a sign uses, or None for one nobody could classify."""
    code = (code or "").strip()
    if not code or not code[0].isalpha():
        return None
    if code == "A1" and f"A1-{speed}" in SIGN_FACES:
        return f"A1-{speed}"
    # "E8-vracht", "C22b": the base sign decides the face.
    for candidate in (code, code.split("-")[0]):
        if candidate in SIGN_FACES:
            return candidate
    letter = code[0].upper()
    return letter if letter in SIGN_FACES else None


def face_cell(key: str) -> int:
    return FIRST_FACE_CELL + _FACE_ORDER.index(key)


def cell_rect(cell: int) -> tuple[float, float, float, float]:
    """A cell's (u0, v0, u1, v1), with V measured from the bottom of the image."""
    col, row = cell % ATLAS_GRID, cell // ATLAS_GRID
    size = 1.0 / ATLAS_GRID
    return (col * size, 1.0 - (row + 1) * size, (col + 1) * size, 1.0 - row * size)


def cell_centre(cell: int) -> tuple[float, float]:
    u0, v0, u1, v1 = cell_rect(cell)
    return ((u0 + u1) / 2.0, (v0 + v1) / 2.0)


def flat_uv(name: str) -> tuple[float, float]:
    return cell_centre(FLAT_CELLS[name][0])


def face_uv(key: str, outline: np.ndarray) -> np.ndarray:
    """Atlas UVs for a unit outline on the face `key`."""
    u, v = cell_centre(face_cell(key))
    half = 0.5 / ATLAS_GRID * FACE_FILL
    return np.column_stack([u + outline[:, 0] * half, v + outline[:, 1] * half])


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow before 10.1 has only the 11 px bitmap font
        return ImageFont.load_default()


def _draw_glyph(draw, glyph: str, cx: float, cy: float, r: float, colour) -> None:
    """A pictogram, drawn in a box of half-size `r` around (cx, cy)."""

    def at(u: float, v: float) -> tuple[float, float]:
        return (cx + u * r, cy - v * r)

    def arrow(points: list[tuple[float, float]]) -> None:
        # A shaft along `points`, and a head on the last segment.
        draw.line([at(*p) for p in points], fill=colour, width=max(2, int(r * 0.22)))
        (ax, ay), (bx, by) = points[-2], points[-1]
        d = np.array([bx - ax, by - ay], dtype=float)
        d /= max(np.linalg.norm(d), 1e-9)
        n = np.array([-d[1], d[0]])
        tip = np.array([bx, by]) + d * 0.22
        base = np.array([bx, by]) - d * 0.22
        draw.polygon(
            [at(*tip), at(*(base + n * 0.32)), at(*(base - n * 0.32))], fill=colour
        )

    def disc(u: float, v: float, radius: float, **kwargs) -> None:
        x, y = at(u, v)
        draw.ellipse(
            [x - radius * r, y - radius * r, x + radius * r, y + radius * r], **kwargs
        )

    def rect(a, b, fill) -> None:
        (x0, y0), (x1, y1) = at(*a), at(*b)
        draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=fill)

    line_w = max(2, int(r * 0.12))
    if not glyph:
        return
    if glyph == "arrow_up":
        arrow([(0, -0.6), (0, 0.45)])
    elif glyph == "arrow_right":
        arrow([(-0.6, 0), (0.45, 0)])
    elif glyph == "arrow_down_right":
        arrow([(-0.4, 0.4), (0.3, -0.3)])
    elif glyph == "arrow_down_both":
        arrow([(0, 0.5), (0.4, -0.3)])
        arrow([(0, 0.5), (-0.4, -0.3)])
    elif glyph == "roundabout":
        for turn in range(3):
            a = turn * 2 * np.pi / 3
            arrow(
                [
                    (0.55 * np.cos(a + t), 0.55 * np.sin(a + t))
                    for t in (0.0, 0.5, 1.0, 1.4)
                ]
            )
    elif glyph == "bike":
        disc(-0.42, -0.2, 0.3, outline=colour, width=line_w)
        disc(0.42, -0.2, 0.3, outline=colour, width=line_w)
        draw.line(
            [at(-0.42, -0.2), at(-0.1, 0.3), at(0.3, 0.3), at(0.42, -0.2)],
            fill=colour, width=line_w,
        )
        draw.line([at(-0.1, 0.3), at(0.05, -0.2), at(0.3, 0.3)], fill=colour, width=line_w)
        draw.line([at(-0.2, 0.42), at(0.0, 0.42)], fill=colour, width=line_w)
    elif glyph == "walker":
        disc(0.0, 0.62, 0.16, fill=colour)
        draw.line([at(0, 0.45), at(0, -0.1)], fill=colour, width=int(r * 0.22))
        draw.line([at(0, -0.1), at(-0.3, -0.7)], fill=colour, width=int(r * 0.16))
        draw.line([at(0, -0.1), at(0.3, -0.7)], fill=colour, width=int(r * 0.16))
        draw.line([at(-0.35, 0.1), at(0, 0.35), at(0.35, 0.1)], fill=colour, width=line_w)
    elif glyph in ("car", "truck"):
        top = 0.35 if glyph == "car" else 0.5
        rect((-0.6, top - 0.1), (0.6, -0.25), colour)
        rect((-0.38, top + 0.2), (0.38, top - 0.1), colour)
        disc(-0.36, -0.3, 0.16, fill=colour)
        disc(0.36, -0.3, 0.16, fill=colour)
    elif glyph == "zebra":
        draw.polygon([at(-0.8, -0.75), at(0.8, -0.75), at(0, 0.8)], fill=WHITE)
        _draw_glyph(draw, "walker", cx, cy + 0.1 * r, r * 0.5, BLACK)
    elif glyph == "dead_end":
        rect((-0.14, -0.75), (0.14, 0.3), WHITE)
        rect((-0.5, 0.3), (0.5, 0.62), RED)
    elif glyph == "junction":
        draw.line([at(0, -0.45), at(0, 0.35)], fill=colour, width=int(r * 0.2))
        draw.line([at(-0.35, -0.1), at(0.35, -0.1)], fill=colour, width=line_w)
    else:  # text: a speed, P, STOP, !
        size = int(r * (1.5 if len(glyph) <= 1 else 1.15 if len(glyph) == 2 else 0.8))
        font = _font(max(8, size))
        box = draw.textbbox((0, 0), glyph, font=font)
        draw.text(
            (cx - (box[0] + box[2]) / 2.0, cy - (box[1] + box[3]) / 2.0),
            glyph, fill=colour, font=font,
        )


def _draw_face(draw, key: str, x0: float, y0: float, size: float) -> None:
    """Paint one sign face into its cell."""
    shape, recipe, glyph = SIGN_FACES[key]
    outline = SHAPES[shape]
    cx, cy = x0 + size / 2.0, y0 + size / 2.0
    full = size / 2.0 * FACE_FILL

    def poly(scale: float, colour, squash: float = 1.0) -> None:
        if shape == "disc":
            r = full * scale
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)
            return
        centre = outline.mean(axis=0)
        pts = centre + (outline - centre) * scale
        draw.polygon(
            [(cx + u * full, cy - v * full) for u, v in pts], fill=colour
        )

    def background(colour) -> None:
        draw.rectangle([x0, y0, x0 + size, y0 + size], fill=colour)

    # Where the pictogram sits: a triangle's centre of mass is not its box's.
    gy = cy - outline.mean(axis=0)[1] * full
    if recipe == "ring":
        background(RED)
        poly(1.0, RED)
        poly(0.76, WHITE)
        _draw_glyph(draw, glyph, cx, gy, full * 0.52, BLACK)
    elif recipe == "no_entry":
        background(RED)
        poly(1.0, RED)
        draw.rectangle(
            [cx - full * 0.68, cy - full * 0.15, cx + full * 0.68, cy + full * 0.15],
            fill=WHITE,
        )
    elif recipe == "blue":
        background(BLUE)
        poly(1.0, BLUE)
        _draw_glyph(draw, glyph, cx, gy, full * 0.62, WHITE)
    elif recipe == "no_parking":
        background(RED)
        poly(1.0, RED)
        poly(0.78, BLUE)
        w = int(full * 0.2)
        k = full * 0.6
        draw.line([cx - k, cy - k, cx + k, cy + k], fill=RED, width=w)
        if glyph == "2":
            draw.line([cx - k, cy + k, cx + k, cy - k], fill=RED, width=w)
    elif recipe in ("warning", "yield"):
        background(RED)
        poly(1.0, RED)
        poly(0.62, WHITE)
        if recipe == "warning":
            _draw_glyph(draw, glyph, cx, gy, full * 0.4, BLACK)
    elif recipe == "stop":
        background(RED)
        poly(1.0, WHITE)
        poly(0.9, RED)
        _draw_glyph(draw, glyph, cx, gy, full * 0.62, WHITE)
    elif recipe == "priority":
        background(WHITE)
        poly(1.0, WHITE)
        poly(0.66, ORANGE)


def _draw_symbol(draw, name: str, x0: float, y0: float, size: float) -> None:
    """One road symbol, white, its top towards the far end of the paint.

    Painted symbols are stretched along the road, because the driver sees
    them foreshortened. The bicycle lies along the lane, front wheel ahead.
    """
    cx = x0 + size / 2.0
    white = (250, 250, 250, 255)
    w = max(3, int(size * 0.045))
    if name == "bike":
        r = size * 0.17
        for wy in (y0 + size * 0.24, y0 + size * 0.76):
            draw.ellipse([cx - r, wy - r, cx + r, wy + r], outline=white, width=w)
        top, bottom = y0 + size * 0.24, y0 + size * 0.76
        draw.line([(cx, top), (cx - size * 0.12, y0 + size * 0.42),
                   (cx - size * 0.12, y0 + size * 0.62), (cx, bottom)], fill=white, width=w)
        draw.line([(cx - size * 0.12, y0 + size * 0.42), (cx, y0 + size * 0.62)],
                  fill=white, width=w)
        draw.line([(cx - size * 0.2, y0 + size * 0.3), (cx + size * 0.08, y0 + size * 0.3)],
                  fill=white, width=w)
    elif name == "bus":
        font = _font(int(size * 0.42))
        box = draw.textbbox((0, 0), "BUS", font=font)
        draw.text(
            (cx - (box[0] + box[2]) / 2.0, y0 + size / 2.0 - (box[1] + box[3]) / 2.0),
            "BUS", fill=white, font=font,
        )
    elif name == "arrow":
        shaft = size * 0.12
        draw.rectangle([cx - shaft / 2, y0 + size * 0.4, cx + shaft / 2, y0 + size * 0.95],
                       fill=white)
        draw.polygon([(cx, y0 + size * 0.05), (cx - size * 0.28, y0 + size * 0.42),
                      (cx + size * 0.28, y0 + size * 0.42)], fill=white)


def generate_traffic_texture(work_dir: Path, cell_px: int = ATLAS_CELL_PX) -> Path:
    """The atlas: flat paints in the top rows, one sign face per cell below,
    then the road symbols on transparent cells."""
    from PIL import Image, ImageDraw

    # Drawn large and reduced, which is all the anti-aliasing a disc needs.
    over = 4
    big = cell_px * over
    image = Image.new(
        "RGBA", (ATLAS_GRID * big, ATLAS_GRID * big), (*FLAT_CELLS["back"][1], 255)
    )
    draw = ImageDraw.Draw(image)
    for cell, colour in FLAT_CELLS.values():
        col, row = cell % ATLAS_GRID, cell // ATLAS_GRID
        draw.rectangle(
            [col * big, row * big, (col + 1) * big - 1, (row + 1) * big - 1], fill=colour
        )
    for key in _FACE_ORDER:
        cell = face_cell(key)
        col, row = cell % ATLAS_GRID, cell // ATLAS_GRID
        _draw_face(draw, key, col * big, row * big, big)
    for name in SYMBOLS:
        cell = symbol_cell(name)
        col, row = cell % ATLAS_GRID, cell // ATLAS_GRID
        # Transparent, with the colour of the paint under it, so the edge a
        # mip level blends in is white rather than black.
        draw.rectangle(
            [col * big, row * big, (col + 1) * big - 1, (row + 1) * big - 1],
            fill=(250, 250, 250, 0),
        )
        _draw_symbol(draw, name, col * big, row * big, big)
    image = image.resize(
        (ATLAS_GRID * cell_px, ATLAS_GRID * cell_px), Image.LANCZOS
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "traffic.png"
    image.save(path)
    LOG.info("wrote %s (%d sign faces)", path, len(_FACE_ORDER))
    return path


# --------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------


class MeshBuilder:
    """Triangles with a UV per vertex, collected piece by piece."""

    def __init__(self) -> None:
        self._vertices: list[np.ndarray] = []
        self._faces: list[np.ndarray] = []
        self._uv: list[np.ndarray] = []
        self._count = 0

    def add(self, vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> None:
        vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim == 1:
            uv = np.tile(uv, (len(vertices), 1))
        self._vertices.append(vertices)
        self._faces.append(np.asarray(faces, dtype=np.int64).reshape(-1, 3) + self._count)
        self._uv.append(uv)
        self._count += len(vertices)

    def fan(self, vertices: np.ndarray, uv: np.ndarray) -> None:
        """A convex polygon, as a fan from its first corner."""
        n = len(vertices)
        faces = np.array([[0, i, i + 1] for i in range(1, n - 1)])
        self.add(vertices, faces, uv)

    def box(self, centre, size, right: np.ndarray, uv) -> None:
        """A box whose local X runs along the horizontal unit vector `right`."""
        cx, cy, cz = centre
        sx, sy, sz = (s / 2.0 for s in size)
        forward = np.array([-right[1], right[0]])
        corners = []
        for dz in (-sz, sz):
            for dx, dy in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)):
                xy = np.array([cx, cy]) + right * dx + forward * dy
                corners.append([xy[0], xy[1], cz + dz])
        faces = np.array(
            [
                [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
            ]
        )
        self.add(np.array(corners), faces, np.asarray(uv, dtype=np.float64))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._vertices:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2))
        return np.vstack(self._vertices), np.vstack(self._faces), np.vstack(self._uv)

    def __len__(self) -> int:
        return sum(len(f) for f in self._faces)


PAINT_PIECE_M = 1.5
# Paint stops this far short of the kerb, as it does on the street: a stripe
# run right up to it has two corners on the footway.
KERB_INSET_M = 0.15


class GroundPaint:
    """Flat polygons lying on the road, heights filled in all at once."""

    def __init__(self) -> None:
        self._polygons: list[np.ndarray] = []
        # A flat colour's one UV, or a UV per corner for a textured symbol.
        self._uv: list[np.ndarray] = []

    def quad(self, centre, along, half_length: float, half_width: float, uv) -> None:
        centre = np.asarray(centre, dtype=np.float64)
        along = np.asarray(along, dtype=np.float64)
        across = np.array([-along[1], along[0]])
        b = across * half_width
        # In pieces, so a long stripe follows the camber of the road it lies
        # on instead of bridging it corner to corner.
        pieces = max(1, int(np.ceil(2.0 * half_length / PAINT_PIECE_M)))
        edges = np.linspace(-half_length, half_length, pieces + 1)
        for start, end in zip(edges[:-1], edges[1:]):
            p, q = centre + along * start, centre + along * end
            self.polygon(np.array([p - b, q - b, q + b, p + b]), uv)

    def polygon(self, xy: np.ndarray, uv) -> None:
        """A convex polygon, counter-clockwise seen from above."""
        self._polygons.append(np.asarray(xy, dtype=np.float64))
        self._uv.append(np.asarray(uv, dtype=np.float64))

    def symbol(self, centre, along, half_length: float, half_width: float, name: str) -> None:
        """A road symbol, its top towards `along`. One quad, not pieces: the
        texture is laid over all of it."""
        centre = np.asarray(centre, dtype=np.float64)
        along = np.asarray(along, dtype=np.float64)
        b = np.array([-along[1], along[0]]) * half_width
        a = along * half_length
        self.polygon(
            np.array([centre - a - b, centre + a - b, centre + a + b, centre - a + b]),
            symbol_uv(name),
        )

    def __len__(self) -> int:
        return len(self._polygons)

    def mesh(self, height_at: Callable, lift: float) -> MeshBuilder:
        out = MeshBuilder()
        if not self._polygons:
            return out
        flat = np.vstack(self._polygons)
        z = np.asarray(height_at(flat[:, 0], flat[:, 1]), dtype=np.float64) + lift
        start = 0
        for polygon, uv in zip(self._polygons, self._uv):
            n = len(polygon)
            out.fan(np.column_stack([polygon, z[start:start + n]]), uv)
            start += n
        return out


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def along(points: np.ndarray, s) -> tuple[np.ndarray, np.ndarray]:
    """Point and unit tangent at distance `s` along a polyline."""
    seg = np.diff(points, axis=0)
    length = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(length)])
    s = np.clip(np.atleast_1d(np.asarray(s, dtype=np.float64)), 0.0, cum[-1])
    index = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(length) - 1)
    tangent = seg[index] / length[index][:, None]
    return points[index] + tangent * (s - cum[index])[:, None], tangent


def polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _left(vector: np.ndarray) -> np.ndarray:
    return np.array([-vector[1], vector[0]])


class Carriageway:
    """The road surface as one shape, for measuring it kerb to kerb."""

    def __init__(self, geometry=None) -> None:
        import shapely

        self.geometry = geometry
        if geometry is not None and not shapely.is_empty(geometry):
            shapely.prepare(geometry)
        else:
            self.geometry = None

    def __bool__(self) -> bool:
        return self.geometry is not None

    def sections(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        *,
        reach: float = 16.0,
        slack: float = 1.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Where the road ends either side of each point, along its normal.

        Returns ``(lo, hi)``: signed distances along the normal, NaN where the
        point is not on a road. A cut that reaches the end of its line never
        found a kerb -- it is looking down a side street -- and is NaN too.
        """
        import shapely

        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 2)
        lo = np.full(len(points), np.nan)
        hi = np.full(len(points), np.nan)
        if self.geometry is None or not len(points):
            return lo, hi
        lines = shapely.linestrings(
            np.stack([points - normals * reach, points + normals * reach], axis=1)
        )
        for k, cut in enumerate(shapely.intersection(lines, self.geometry)):
            if cut is None or shapely.is_empty(cut):
                continue
            best = None
            for part in shapely.get_parts(cut):
                if part.geom_type != "LineString":
                    continue
                d = (np.asarray(part.coords)[:, :2] - points[k]) @ normals[k]
                low, high = float(d.min()), float(d.max())
                gap = 0.0 if low <= 0.0 <= high else min(abs(low), abs(high))
                if gap <= slack and (best is None or gap < best[0]):
                    best = (gap, low, high)
            if best is None or best[1] < -(reach - 0.5) or best[2] > reach - 0.5:
                continue
            lo[k], hi[k] = best[1], best[2]
        return lo, hi

    def push_off(self, xy, margin: float = 0.6, reach: float = 4.0):
        """A point on the road moved to the kerb and `margin` past it, or None
        when it is not on the road, or further in than `reach`."""
        import shapely

        if self.geometry is None:
            return None
        point = shapely.Point(float(xy[0]), float(xy[1]))
        if not self.geometry.contains(point):
            return None
        edge = shapely.get_coordinates(
            shapely.shortest_line(self.geometry.boundary, point)
        )[0]
        out = edge - np.asarray(xy, dtype=np.float64)
        distance = float(np.linalg.norm(out))
        if distance > reach or distance < 1e-9:
            return None
        return edge + out / distance * margin

    def section(self, point, normal, **kwargs) -> tuple[float, float] | None:
        lo, hi = self.sections(np.asarray(point)[None], np.asarray(normal)[None], **kwargs)
        if not np.isfinite(lo[0]):
            return None
        return float(lo[0]), float(hi[0])


ROAD_FUNCTIONS = ("rijbaan", "ov-baan", "woonerf", "overweg")
CYCLE_FUNCTIONS = ("fietspad",)


def carriageways(features: list[dict]) -> tuple[Carriageway, Carriageway]:
    """BGT road surfaces as (motor traffic, cycle tracks)."""
    import shapely
    from shapely.geometry import shape

    road, cycle = [], []
    for feature in features:
        props = feature.get("properties") or {}
        function = str(props.get("functie") or "").lower()
        try:
            if int(props.get("relatieve_hoogteligging") or 0) < 0:
                continue  # in a tunnel: nothing up here to paint
        except (TypeError, ValueError):
            pass
        if function.startswith(ROAD_FUNCTIONS):
            bucket = road
        elif function.startswith(CYCLE_FUNCTIONS):
            bucket = cycle
        else:
            continue
        try:
            geometry = shape(feature.get("geometry") or {})
        except Exception:  # noqa: BLE001 - one bad outline is not worth a run
            continue
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            geometry = shapely.make_valid(geometry)
        bucket.append(geometry)

    def merged(parts):
        if not parts:
            return None
        # A hair of overlap, so neighbours that share an edge to the
        # millimetre become one surface rather than two that touch.
        return shapely.union_all(shapely.buffer(np.array(parts, dtype=object), 0.02))

    return Carriageway(merged(road)), Carriageway(merged(cycle))


# --------------------------------------------------------------------------
# Signs
# --------------------------------------------------------------------------


@dataclass
class Sign:
    xy: tuple[float, float]
    code: str
    bearing: float | None = None  # compass degrees its traffic drives
    section_id: int | None = None  # NWB wvk_id
    direction: str = ""  # H with the section's digitising, T against
    speed: str = ""
    text: str = ""


def parse_signs(features: list[dict], bbox: BBox | None = None) -> list[Sign]:
    """NDW sign features into roadside signs in RD."""
    rows = []
    for feature in features:
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        if str(props.get("status") or "PLACED") != "PLACED":
            continue
        # "B" hangs over the road on a gantry; this draws posts.
        if str(props.get("placement") or "L") != "L":
            continue
        code = str(props.get("rvvCode") or "").strip()
        if face_key(code) is None:
            continue
        rows.append((float(coords[0]), float(coords[1]), code, props))
    if not rows:
        return []
    to_rd = _transformer(WGS84_CRS, RD_CRS)
    xs, ys = to_rd.transform([r[0] for r in rows], [r[1] for r in rows])
    signs = []
    for x, y, (_, _, code, props) in zip(xs, ys, rows):
        if bbox is not None and not (
            bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax
        ):
            continue
        try:
            section = int(props["roadSectionId"])
        except (KeyError, TypeError, ValueError):
            section = None
        bearing = props.get("bearing")
        texts = [
            str(t.get("text") or "").strip()
            for t in props.get("textSigns") or []
            if isinstance(t, dict)
        ]
        signs.append(
            Sign(
                xy=(float(x), float(y)),
                code=code,
                bearing=float(bearing) if isinstance(bearing, (int, float)) else None,
                section_id=section,
                direction=str(props.get("drivingDirection") or ""),
                speed=str(props.get("blackCode") or "").strip(),
                text="; ".join(t for t in texts if t),
            )
        )
    return signs


def municipality_codes(bbox: BBox, *, url: str = MUNICIPALITY_URL, timeout: float = 60.0) -> list[str]:
    """The municipalities a bbox touches, as GM codes."""
    response = get_with_retry(
        url,
        params={
            "bbox": f"{bbox.xmin:.1f},{bbox.ymin:.1f},{bbox.xmax:.1f},{bbox.ymax:.1f}",
            "bbox-crs": RD_URI,
            "limit": 50,
            "f": "json",
        },
        timeout=timeout,
        max_retries=3,
        description="municipality lookup",
    )
    codes = []
    for feature in response.json().get("features") or []:
        code = str((feature.get("properties") or {}).get("identificatie") or "")
        if code.startswith("GM") and code not in codes:
            codes.append(code)
    return codes


def fetch_signs(
    bbox: BBox,
    *,
    url: str = NDW_SIGNS_URL,
    municipality_url: str = MUNICIPALITY_URL,
    timeout: float = 300.0,
    cache_dir: Path | None = None,
) -> list[Sign]:
    """Every roadside sign in the bbox, or none when the register is away.

    NDW publishes per municipality and has no bounding-box query, so a run
    downloads the whole municipality -- 18 MB for Utrecht -- and keeps what
    falls inside. Only that is cached.
    """
    cache = None
    if cache_dir is not None:
        key = f"{bbox.xmin:.1f}_{bbox.ymin:.1f}_{bbox.xmax:.1f}_{bbox.ymax:.1f}"
        cache = Path(cache_dir) / f"ndw_signs_{key}.json.gz"
        if cache.is_file():
            with gzip.open(cache, "rt", encoding="utf-8") as handle:
                signs = parse_signs(json.load(handle), bbox)
            LOG.info("NDW: %d traffic signs (cached)", len(signs))
            return signs

    try:
        codes = municipality_codes(bbox, url=municipality_url)
        features: list[dict] = []
        for code in codes:
            response = get_with_retry(
                url,
                params={"countyCode": code},
                timeout=timeout,
                max_retries=3,
                description=f"NDW traffic signs {code}",
            )
            features.extend(response.json().get("features") or [])
    except Exception as exc:  # noqa: BLE001 - signs are detail, not structure
        LOG.warning(
            "could not read the NDW traffic sign register (%s); the model "
            "gets no traffic signs",
            short_error(exc),
        )
        return []

    # Keep only what is inside, with a margin for the cache to be honest
    # about: a sign 2 m outside is not in the model.
    inside = []
    wgs = _transformer(RD_CRS, WGS84_CRS)
    lon0, lat0 = wgs.transform(bbox.xmin - 50, bbox.ymin - 50)
    lon1, lat1 = wgs.transform(bbox.xmax + 50, bbox.ymax + 50)
    for feature in features:
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        if len(coords) >= 2 and lon0 <= coords[0] <= lon1 and lat0 <= coords[1] <= lat1:
            inside.append(feature)
    signs = parse_signs(inside, bbox)
    LOG.info(
        "NDW: %d traffic signs (of %d in %s)", len(signs), len(features), ", ".join(codes)
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        staging = cache.with_suffix(cache.suffix + ".part")
        with gzip.open(staging, "wt", encoding="utf-8") as handle:
            json.dump(inside, handle)
        staging.replace(cache)
    return signs


class RoadIndex:
    """Road lines, for asking which one something stands beside."""

    def __init__(self, lines: list[np.ndarray], sections: dict[int, np.ndarray] | None = None):
        import shapely

        self.lines = [np.asarray(line, dtype=np.float64) for line in lines if len(line) >= 2]
        self.sections = sections or {}
        self._shapes = [shapely.LineString(line) for line in self.lines]
        self._tree = shapely.STRtree(self._shapes) if self._shapes else None

    def nearest(self, xy, reach: float) -> tuple[np.ndarray, np.ndarray] | None:
        """(point on road, unit tangent) of the nearest line within `reach`."""
        import shapely

        if self._tree is None:
            return None
        point = shapely.Point(float(xy[0]), float(xy[1]))
        index = self._tree.nearest(point)
        shape = self._shapes[int(index)]
        if shape.distance(point) > reach:
            return None
        p, t = along(self.lines[int(index)], shape.project(point))
        return p[0], t[0]

    def on_section(self, section_id: int | None, xy) -> tuple[np.ndarray, np.ndarray] | None:
        import shapely

        line = self.sections.get(section_id) if section_id is not None else None
        if line is None or len(line) < 2:
            return None
        s = shapely.LineString(line).project(shapely.Point(float(xy[0]), float(xy[1])))
        p, t = along(np.asarray(line, dtype=np.float64), s)
        return p[0], t[0]


def road_index(network=None, osm=None) -> RoadIndex:
    """The NWB sections and OpenStreetMap's roads and cycle paths, as one index.

    NWB carries the section ids the sign register refers to; OpenStreetMap
    has the estate lanes and cycle paths NWB leaves out.
    """
    lines = [
        w.points for w in (getattr(osm, "ways", None) or [])
        if w.highway in DRIVABLE or w.highway == "cycleway"
    ]
    sections: dict[int, np.ndarray] = {}
    for line in getattr(network, "lines", None) or []:
        if getattr(line, "kind", "") == "skeleton":
            continue
        sections[int(line.section_id)] = np.asarray(line.points, dtype=np.float64)
        lines.append(sections[int(line.section_id)])
    return RoadIndex(lines, sections)


def travel_direction(sign: Sign, roads: RoadIndex, reach: float = 15.0):
    """Which way the traffic a sign speaks to is driving.

    Returns ``(direction, point on the road or None)``. The register's own
    bearing is rounded to the nearest compass point, so it only ever chooses
    between a road's two directions; the direction itself comes from the road.
    """
    hint = None
    if sign.bearing is not None:
        radians = np.radians(sign.bearing)
        hint = np.array([np.sin(radians), np.cos(radians)])

    found = roads.on_section(sign.section_id, sign.xy)
    if found is not None and sign.direction in ("H", "T"):
        point, tangent = found
        return (tangent if sign.direction == "H" else -tangent), point
    if found is None:
        found = roads.nearest(sign.xy, reach)
    if found is None:
        return (hint if hint is not None else np.array([0.0, 1.0])), None

    point, tangent = found
    if hint is not None and abs(float(tangent @ hint)) > 0.3:
        return (tangent if tangent @ hint > 0 else -tangent), point
    # No usable hint: a sign stands to the right of the traffic it is for.
    offset = np.asarray(sign.xy) - point
    return (tangent if _left(tangent) @ offset < 0 else -tangent), point


def _group_posts(positions: np.ndarray, directions: np.ndarray, radius: float = 0.8):
    """Signs that share a post: close together and facing the same way."""
    groups: list[list[int]] = []
    cells: dict[tuple[int, int], list[int]] = {}
    for index, (xy, direction) in enumerate(zip(positions, directions)):
        cx, cy = int(np.floor(xy[0] / radius)), int(np.floor(xy[1] / radius))
        placed = False
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for group_index in cells.get((gx, gy), ()):
                    head = groups[group_index][0]
                    if (
                        np.linalg.norm(positions[head] - xy) <= radius
                        and directions[head] @ direction > 0.5
                    ):
                        groups[group_index].append(index)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            cells.setdefault((cx, cy), []).append(len(groups))
            groups.append([index])
    return groups


SIGN_TOP_M = 2.45
SIGN_STEP_M = 0.72
SIGNS_PER_POST = 3


def sign_mesh(
    signs: list[Sign], directions: np.ndarray, ground: np.ndarray
) -> tuple[MeshBuilder, list[dict]]:
    """Posts with their plates. `directions` is the travel direction per sign."""
    mesh = MeshBuilder()
    records: list[dict] = []
    if not signs:
        return mesh, records
    positions = np.array([s.xy for s in signs], dtype=np.float64)
    post_uv, back_uv = flat_uv("post"), flat_uv("back")

    for group in _group_posts(positions, directions):
        head = group[0]
        x, y = positions[head]
        base = float(ground[head])
        direction = directions[head]
        normal = -direction  # the face looks back at its traffic
        right = np.array([-normal[1], normal[0]])
        members = group[:SIGNS_PER_POST]
        top = SIGN_TOP_M
        mesh.box(
            (x, y, base + (top + 0.4) / 2.0), (0.06, 0.06, top + 0.4), right, post_uv
        )
        for slot, index in enumerate(members):
            sign = signs[index]
            key = face_key(sign.code, sign.speed)
            shape = SIGN_FACES[key][0]
            outline = SHAPES[shape]
            half = SHAPE_HALF_M[shape]
            centre_z = base + top - slot * SIGN_STEP_M
            front = np.column_stack(
                [
                    x + normal[0] * 0.045 + right[0] * outline[:, 0] * half,
                    y + normal[1] * 0.045 + right[1] * outline[:, 0] * half,
                    centre_z + outline[:, 1] * half,
                ]
            )
            mesh.fan(front, face_uv(key, outline))
            back = front[::-1].copy()
            back[:, 0] -= normal[0] * 0.012
            back[:, 1] -= normal[1] * 0.012
            mesh.fan(back, np.asarray(back_uv))
            records.append(
                {
                    "code": sign.code,
                    "face": key,
                    "rd": [round(float(x), 3), round(float(y), 3)],
                    "ground_z_nap": round(base, 3),
                    "height_m": round(top - slot * SIGN_STEP_M, 2),
                    # Compass degrees the plate faces.
                    "facing_deg": round(float(np.degrees(np.arctan2(normal[0], normal[1])) % 360), 1),
                    **({"speed": sign.speed} if sign.speed else {}),
                    **({"text": sign.text} if sign.text else {}),
                }
            )
    return mesh, records


# --------------------------------------------------------------------------
# Traffic lights
# --------------------------------------------------------------------------

SIGNAL_POLE_M = 3.3


def signal_mesh(
    positions: np.ndarray, directions: np.ndarray, ground: np.ndarray
) -> MeshBuilder:
    """A pole with a three-lamp head, facing the traffic that drives `directions`."""
    mesh = MeshBuilder()
    lamp = SHAPES["octagon"] * 0.09
    for (x, y), direction, base in zip(positions, directions, ground):
        normal = -np.asarray(direction, dtype=np.float64)
        right = np.array([-normal[1], normal[0]])
        base = float(base)
        mesh.box((x, y, base + SIGNAL_POLE_M / 2.0), (0.11, 0.11, SIGNAL_POLE_M), right, flat_uv("post"))
        hx, hy = x + normal[0] * 0.16, y + normal[1] * 0.16
        mesh.box((hx, hy, base + 2.85), (0.28, 0.22, 0.86), right, flat_uv("housing"))
        # A backboard, which is most of what one sees of a signal from afar.
        mesh.box((x + normal[0] * 0.07, y + normal[1] * 0.07, base + 2.85), (0.5, 0.03, 1.1), right, flat_uv("housing"))
        fx, fy = hx + normal[0] * 0.115, hy + normal[1] * 0.115
        for colour, height in (("red", 3.12), ("amber", 2.85), ("green", 2.58)):
            face = np.column_stack(
                [
                    fx + right[0] * lamp[:, 0],
                    fy + right[1] * lamp[:, 0],
                    base + height + lamp[:, 1],
                ]
            )
            mesh.fan(face, np.asarray(flat_uv(colour)))
    return mesh


DRIVABLE = frozenset(
    {
        "motorway", "motorway_link", "trunk", "trunk_link", "primary",
        "primary_link", "secondary", "secondary_link", "tertiary",
        "tertiary_link", "unclassified", "residential", "living_street",
        "service", "busway", "road",
    }
)


def _is_oneway(tags: dict) -> bool:
    return tags.get("oneway") in ("yes", "1", "true", "-1") or tags.get("junction") == "roundabout"


def _node_on_way(way, node_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Position and tangent of a way at one of its nodes."""
    try:
        index = way.node_ids.index(node_id)
    except ValueError:
        return None
    points = way.points
    a = points[max(index - 1, 0)]
    b = points[min(index + 1, len(points) - 1)]
    tangent = b - a
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        return None
    return points[index], tangent / norm


# A signal OpenStreetMap puts on the junction itself stops its traffic this
# far back from the middle: half the road it crosses and the crossing before it.
SIGNAL_SETBACK_M = 6.0


def place_signals(
    poles: np.ndarray, osm, roads: RoadIndex, road: Carriageway, *, lone_m: float = 30.0,
    stops: list | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Where the lights stand and which way their traffic drives.

    Surveyed poles first. A junction OpenStreetMap tags as signalled with no
    surveyed pole near it gets one per approach -- not per way through it: a
    way that leaves the junction has no traffic arriving that the light is
    for -- on the kerb to the right, set back from the middle of the junction.

    `stops`, when given, is filled with the point on the road where each
    light's traffic stops, or None where no road was found, in the same order.
    """
    positions: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    stopping: list = []
    for xy in np.asarray(poles, dtype=np.float64).reshape(-1, 2):
        found = roads.nearest(xy, 25.0)
        if found is None:
            direction = np.array([0.0, 1.0])
            stopping.append(None)
        else:
            point, tangent = found
            direction = tangent if _left(tangent) @ (xy - point) < 0 else -tangent
            stopping.append(np.asarray(point, dtype=np.float64))
        positions.append(xy)
        directions.append(direction)
    surveyed = len(positions)

    through = osm.ways_through() if osm else {}
    for node in osm.nodes if osm else []:
        if node.tags.get("highway") != "traffic_signals":
            continue
        xy = np.asarray(node.xy)
        if surveyed and np.min(np.linalg.norm(np.asarray(positions[:surveyed]) - xy, axis=1)) < lone_m:
            continue
        drivable = [w for w in through.get(node.id, []) if w.highway in DRIVABLE]
        junction = len(drivable) > 1
        for way in drivable:
            found = _node_on_way(way, node.id)
            if found is None:
                continue
            point, tangent = found
            index = way.node_ids.index(node.id)
            wanted = node.tags.get("traffic_signals:direction", "")
            headings = []
            # Forward traffic arrives from the node before; backward traffic
            # from the node after. A way that starts or ends here brings
            # traffic in one way only.
            if wanted != "backward" and index > 0:
                headings.append(tangent)
            if (
                wanted != "forward" and not _is_oneway(way.tags)
                and index < len(way.node_ids) - 1
            ):
                headings.append(-tangent)
            for heading in headings:
                stop = point - heading * (SIGNAL_SETBACK_M if junction else 0.0)
                rightward = -_left(heading)
                cut = road.section(stop, rightward)
                reach = (cut[1] if cut else 3.2) + 0.7
                positions.append(stop + rightward * reach)
                directions.append(heading)
                stopping.append(stop)

    if stops is not None:
        stops[:] = stopping
    stats = {"surveyed": surveyed, "from_osm": len(positions) - surveyed}
    if not positions:
        return np.zeros((0, 2)), np.zeros((0, 2)), stats
    return np.asarray(positions), np.asarray(directions), stats


# --------------------------------------------------------------------------
# Paint
# --------------------------------------------------------------------------

CROSSING_LENGTH_M = 4.0
LANE_MIN_M = 2.4
LANE_MAX_M = 4.6
LINE_WIDTH_M = 0.10


def _lane_count(tags: dict) -> int:
    try:
        return int(str(tags.get("lanes", "")).split(";")[0])
    except ValueError:
        return 0


def paint_lanes(
    paint: GroundPaint,
    osm,
    road: Carriageway,
    bbox: BBox | None = None,
    crossings: list[dict] | None = None,
) -> dict:
    """Lines between lanes, wherever the kerbs are as far apart as the lanes need.

    `crossings` are the ones already painted: a lane line stops short of a
    zebra, it does not run through it.
    """
    taken = np.array([c["rd"] for c in crossings or []], dtype=np.float64).reshape(-1, 2)
    stats = {"lane_line_m": 0.0, "laned_ways": 0}
    white = flat_uv("white")
    for way in osm.ways if osm else []:
        tags = way.tags
        lanes = _lane_count(tags)
        if way.highway not in DRIVABLE or lanes < 2:
            continue
        if tags.get("lane_markings") == "no" or tags.get("tunnel") == "yes" or tags.get("area") == "yes":
            continue
        length = polyline_length(way.points)
        try:
            speed = float(tags.get("maxspeed", "50"))
        except ValueError:
            speed = 50.0
        # 3 m of line and 9 m of gap is the Dutch lane line outside town;
        # inside, the gap closes up to the length of the line.
        dash, gap = (3.0, 9.0) if speed > 50 else (3.0, 3.0)
        stations = np.arange(dash / 2.0 + gap / 2.0, length, dash + gap)
        if not len(stations):
            continue
        points, tangents = along(way.points, stations)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        lo, hi = road.sections(points, normals)
        width = hi - lo
        ok = np.isfinite(width) & (width >= lanes * LANE_MIN_M) & (width <= lanes * LANE_MAX_M)
        if bbox is not None:
            # A way that crosses the area is delivered whole, however far
            # beyond the edge it runs.
            ok &= (
                (points[:, 0] >= bbox.xmin) & (points[:, 0] <= bbox.xmax)
                & (points[:, 1] >= bbox.ymin) & (points[:, 1] <= bbox.ymax)
            )
        if len(taken):
            nearest = np.linalg.norm(points[:, None, :] - taken[None, :, :], axis=2).min(axis=1)
            ok &= nearest > CROSSING_LENGTH_M / 2.0 + dash / 2.0 + 0.5
        if not ok.any():
            continue
        # Counted from the left kerb. On a two-way road the lanes left of the
        # way's direction are the ones coming back.
        if _is_oneway(tags):
            centre = None
        else:
            try:
                centre = int(tags["lanes:backward"])
            except (KeyError, ValueError):
                centre = lanes // 2
        stats["laned_ways"] += 1
        for boundary in range(1, lanes):
            solid = boundary == centre and lanes >= 4
            for k in np.where(ok)[0]:
                offset = hi[k] - width[k] * boundary / lanes
                centre_xy = points[k] + normals[k] * offset
                half = (dash + gap) / 2.0 if solid else dash / 2.0
                paint.quad(centre_xy, tangents[k], half, LINE_WIDTH_M / 2.0, white)
                stats["lane_line_m"] += half * 2.0
    stats["lane_line_m"] = round(stats["lane_line_m"], 1)
    return stats


def crossing_style(tags: dict) -> str:
    """How a crossing is painted: "zebra", "dashes" or "" for bare road."""
    markings = tags.get("crossing:markings", "")
    kind = tags.get("crossing", "")
    if markings in ("no",) or kind in ("unmarked", "informal", "no"):
        return ""
    if markings.startswith("zebra") or kind == "zebra":
        return "zebra"
    if markings:
        return "dashes"
    # Untagged paint. A marked crossing without lights is a zebra in the
    # Netherlands; one with lights is two dashed lines.
    if kind in ("marked", "uncontrolled"):
        return "zebra"
    if kind == "traffic_signals":
        return "dashes"
    return ""


def paint_crossings(
    paint: GroundPaint, osm, road: Carriageway, cycle: Carriageway
) -> tuple[dict, list[dict]]:
    """Zebras and dashed crossings where OpenStreetMap has one on a road."""
    stats = {"zebras": 0, "dashed_crossings": 0, "humps": 0}
    records: list[dict] = []
    if not osm:
        return stats, records
    white = flat_uv("white")
    through = osm.ways_through()
    painted: list[tuple[np.ndarray, float]] = []
    for node in osm.nodes:
        tags = node.tags
        calming = tags.get("traffic_calming", "")
        style = crossing_style(tags) if ("crossing" in tags or tags.get("highway") == "crossing") else ""
        hump = calming in ("hump", "bump")
        if not style and not hump:
            continue
        ways = through.get(node.id, [])
        carriers = [w for w in ways if w.highway in DRIVABLE] or [
            w for w in ways if w.highway == "cycleway"
        ]
        if not carriers:
            continue
        way = carriers[0]
        found = _node_on_way(way, node.id)
        if found is None:
            continue
        point, tangent = found
        normal = _left(tangent)
        surface = road if way.highway in DRIVABLE else cycle
        cut = surface.section(point, normal)
        if cut is None:
            continue
        lo, hi = cut
        width = hi - lo
        if not 2.0 <= width <= 18.0:
            continue
        # A road drawn as two ways has two nodes for one crossing. Where no
        # island parts them, both measure the same kerbs, and one is enough.
        middle = point + normal * (lo + hi) / 2.0
        if any(np.linalg.norm(middle - m) < max(w / 2.0, CROSSING_LENGTH_M) for m, w in painted):
            continue
        painted.append((middle, width))
        lo, hi, width = lo + KERB_INSET_M, hi - KERB_INSET_M, width - 2.0 * KERB_INSET_M

        if style == "zebra":
            # Half-metre stripes with half-metre gaps, centred on the road.
            count = max(1, int((width + 0.5) // 1.0))
            first = lo + (width - (count - 0.5)) / 2.0 + 0.25
            for k in range(count):
                paint.quad(point + normal * (first + k), tangent, CROSSING_LENGTH_M / 2.0, 0.25, white)
            stats["zebras"] += 1
        elif style == "dashes":
            count = max(1, int((width + 0.5) // 1.0))
            first = lo + (width - (count - 0.5)) / 2.0 + 0.25
            for side in (-1.0, 1.0):
                edge = point + tangent * side * CROSSING_LENGTH_M / 2.0
                for k in range(count):
                    paint.quad(edge + normal * (first + k), normal, 0.25, 0.05, white)
            stats["dashed_crossings"] += 1
        if hump and not style:
            # The row of white blocks on the ramp of a Dutch speed hump.
            count = max(1, int((width + 0.5) // 1.0))
            first = lo + (width - (count - 0.5)) / 2.0 + 0.25
            for k in range(count):
                paint.quad(point + normal * (first + k), tangent, 0.4, 0.25, white)
            stats["humps"] += 1
        if style:
            records.append(
                {
                    "style": style,
                    "rd": [round(float(point[0]), 3), round(float(point[1]), 3)],
                    # Kerb to kerb; the paint stops short of both.
                    "width_m": round(float(width + 2.0 * KERB_INSET_M), 2),
                    # Compass degrees the road runs, either way.
                    "road_deg": round(float(np.degrees(np.arctan2(tangent[0], tangent[1])) % 180), 1),
                    "signals": tags.get("crossing") == "traffic_signals",
                }
            )
    return stats, records


TOOTH_BASE_M = 0.5
TOOTH_HEIGHT_M = 0.6
TOOTH_PITCH_M = 0.75
# A carriageway narrower than this carries one direction, and the line
# crosses all of it.
TWO_WAY_MIN_M = 4.6


def paint_give_way(
    paint: GroundPaint,
    signs: list[Sign],
    directions: np.ndarray,
    road_points: list,
    road: Carriageway,
    cycle: Carriageway,
) -> dict:
    """Teeth under every B6 and a line under every B7."""
    stats = {"give_way_lines": 0, "stop_lines": 0}
    white = flat_uv("white")
    done: list[tuple[np.ndarray, np.ndarray]] = []
    for sign, direction, point in zip(signs, directions, road_points):
        base = sign.code.split("-")[0]
        if base not in ("B6", "B7") or point is None:
            continue
        # Two signs either side of the same lane ask for one line.
        if any(np.linalg.norm(p - point) < 4.0 and d @ direction > 0.5 for p, d in done):
            continue
        normal = _left(direction)
        cut = road.section(point, normal) or cycle.section(point, normal)
        if cut is None:
            continue
        lo, hi = cut
        if hi - lo > 18.0:
            continue
        if hi - lo >= TWO_WAY_MIN_M:
            hi = (lo + hi) / 2.0  # the half its traffic drives on: the right
        lo += KERB_INSET_M
        span = hi - lo
        if span < 1.0:
            continue
        done.append((point, direction))
        if base == "B7":
            paint.quad(point + normal * (lo + span / 2.0), normal, span / 2.0 - 0.1, 0.15, white)
            stats["stop_lines"] += 1
            continue
        count = max(1, int((span - TOOTH_BASE_M) // TOOTH_PITCH_M) + 1)
        first = lo + (span - ((count - 1) * TOOTH_PITCH_M + TOOTH_BASE_M)) / 2.0 + TOOTH_BASE_M / 2.0
        for k in range(count):
            centre = point + normal * (first + k * TOOTH_PITCH_M)
            # Points at the driver it is telling to wait.
            paint.polygon(
                np.array(
                    [
                        centre - normal * TOOTH_BASE_M / 2.0,
                        centre + normal * TOOTH_BASE_M / 2.0,
                        centre - direction * TOOTH_HEIGHT_M,
                    ]
                ),
                white,
            )
        stats["give_way_lines"] += 1
    return stats


STOP_LINE_M = 0.3
# A light stands a little past the line its traffic stops at.
STOP_BEFORE_M = 1.5


def paint_stop_lines(
    paint: GroundPaint, stops: list, directions: np.ndarray, road: Carriageway,
    cycle: Carriageway,
) -> dict:
    """A solid line across the lanes each traffic light stops.

    One per approach: a light on either kerb, or a repeat head over the lane,
    stops the same traffic at the same line.
    """
    stats = {"stop_lines_at_lights": 0}
    white = flat_uv("white")
    done: list[tuple[np.ndarray, np.ndarray]] = []
    for stop, direction in zip(stops, directions):
        if stop is None:
            continue
        direction = np.asarray(direction, dtype=np.float64)
        point = np.asarray(stop, dtype=np.float64) - direction * STOP_BEFORE_M
        if any(np.linalg.norm(p - point) < 8.0 and d @ direction > 0.7 for p, d in done):
            continue
        normal = _left(direction)
        cut = road.section(point, normal) or cycle.section(point, normal)
        if cut is None:
            continue
        lo, hi = cut
        if hi - lo > 18.0:
            continue
        if hi - lo >= TWO_WAY_MIN_M:
            hi = (lo + hi) / 2.0
        lo += KERB_INSET_M
        span = hi - lo
        if span < 1.0:
            continue
        done.append((point, direction))
        paint.quad(point + normal * (lo + span / 2.0), normal, span / 2.0, STOP_LINE_M / 2.0, white)
        stats["stop_lines_at_lights"] += 1
    return stats


# Parking stalls, laid out the way the vehicle stage parks its cars in them.
STALL_LINE_M = 0.1


def paint_parking_bays(paint: GroundPaint, features: list[dict], bbox: BBox) -> dict:
    """White lines between the stalls of every BGT parking bay.

    The BGT draws a row of bays as one polygon, and says nothing of where one
    stall ends. The stalls are cut the way the vehicle stage parks its cars,
    so a car stands between two lines rather than on one: nose to tail in a
    strip by the kerb, side by side across a deeper bay, back to back in a
    car park.
    """
    import shapely
    from shapely.geometry import shape

    from .vehicles import (
        BAY_MAX_W, PARALLEL_MAX_W, PARALLEL_PITCH, PERPENDICULAR_PITCH, principal_axes,
    )

    stats = {"parking_bays": 0, "stall_lines": 0}
    white = flat_uv("white")
    for feature in features:
        props = feature.get("properties") or {}
        if not str(props.get("functie") or "").lower().startswith("parkeervlak"):
            continue
        try:
            geometry = shape(feature.get("geometry") or {})
        except Exception:  # noqa: BLE001
            continue
        for polygon in shapely.get_parts(geometry):
            if polygon.geom_type != "Polygon" or polygon.is_empty:
                continue
            ring = np.asarray(polygon.exterior.coords)[:, :2]
            centre = ring.mean(axis=0)
            if not (bbox.xmin <= centre[0] <= bbox.xmax and bbox.ymin <= centre[1] <= bbox.ymax):
                continue
            long_axis, short_axis, along, across = principal_axes(ring)
            if along < 4.0 or across < 1.6:
                continue
            if across <= PARALLEL_MAX_W:
                pitch, rows = PARALLEL_PITCH, [(0.0, across)]
            elif across <= BAY_MAX_W:
                pitch, rows = PERPENDICULAR_PITCH, [(0.0, across)]
            else:
                n_rows = max(1, int(across // 5.0))
                depth = across / n_rows
                pitch = PERPENDICULAR_PITCH
                rows = [((k - (n_rows - 1) / 2.0) * depth, depth) for k in range(n_rows)]
            count = int(along // pitch)
            if count < 1:
                continue
            # Centred on the bay's middle, like the cars.
            middle = centre + long_axis * float(
                ((ring - centre) @ long_axis).max() + ((ring - centre) @ long_axis).min()
            ) / 2.0
            inner = polygon.buffer(-0.1)
            if inner.is_empty:
                continue
            stats["parking_bays"] += 1
            for offset, depth in rows:
                for k in range(count + 1):
                    b = (k - count / 2.0) * pitch
                    mid = middle + long_axis * b + short_axis * offset
                    line = shapely.LineString(
                        [mid - short_axis * depth / 2.0, mid + short_axis * depth / 2.0]
                    )
                    piece = line.intersection(inner)
                    parts = [p for p in shapely.get_parts(piece) if p.geom_type == "LineString"]
                    if not parts:
                        continue
                    part = max(parts, key=lambda p: p.length)
                    if part.length < 1.0:
                        continue
                    ends = np.asarray(part.coords)[:, :2]
                    paint.quad(
                        ends.mean(axis=0), short_axis, part.length / 2.0,
                        STALL_LINE_M / 2.0, white,
                    )
                    stats["stall_lines"] += 1
    return stats


SYMBOL_EVERY_M = 50.0
BIKE_SYMBOL_M = (0.9, 1.6)   # across, along
BUS_SYMBOL_M = (2.2, 3.4)


def paint_symbols(
    paint: GroundPaint, osm, network, road: Carriageway, cycle: Carriageway,
    bbox: BBox | None = None,
) -> dict:
    """A bicycle on every cycle track, "BUS" on every bus lane, now and again.

    Cycle tracks are the BGT's own, found along NWB's cycle path sections
    and OpenStreetMap's cycleways; bus lanes along OpenStreetMap's busways.
    A track wide enough for both directions gets a symbol each way.
    """
    stats = {"bike_symbols": 0, "bus_symbols": 0}
    placed: list[tuple[np.ndarray, np.ndarray]] = []
    lines: list[tuple[np.ndarray, str, bool]] = []
    for line in getattr(network, "lines", None) or []:
        if getattr(line, "kind", "") == "FP":
            lines.append((np.asarray(line.points, dtype=np.float64), "bike", False))
    for way in getattr(osm, "ways", None) or []:
        if way.highway == "cycleway":
            lines.append((way.points, "bike", _is_oneway(way.tags)))
        elif way.highway == "busway":
            lines.append((way.points, "bus", _is_oneway(way.tags)))

    for points, name, oneway in lines:
        length = polyline_length(points)
        if length < SYMBOL_EVERY_M / 2.0:
            continue
        surface = cycle if name == "bike" else road
        across_m, along_m = BIKE_SYMBOL_M if name == "bike" else BUS_SYMBOL_M
        stations = np.arange(SYMBOL_EVERY_M / 4.0, length, SYMBOL_EVERY_M)
        centres, tangents = along(points, stations)
        for centre, tangent in zip(centres, tangents):
            if bbox is not None and not (
                bbox.xmin <= centre[0] <= bbox.xmax and bbox.ymin <= centre[1] <= bbox.ymax
            ):
                continue
            normal = _left(tangent)
            cut = surface.section(centre, normal)
            if cut is None:
                continue
            lo, hi = cut
            width = hi - lo
            if width < across_m + 0.3 or width > (8.0 if name == "bus" else 6.0):
                continue
            middle = centre + normal * (lo + hi) / 2.0
            if any(np.linalg.norm(p - middle) < SYMBOL_EVERY_M * 0.4 for p, _ in placed):
                continue
            two_way = not oneway and width >= 2.0 * across_m + 0.6
            if two_way:
                # Keep right: the half on the right of each direction.
                quarter = width / 4.0
                for heading, side in ((tangent, -1.0), (-tangent, 1.0)):
                    paint.symbol(middle + normal * side * quarter, heading,
                                 along_m / 2.0, across_m / 2.0, name)
                    stats[f"{name}_symbols"] += 1
            else:
                paint.symbol(middle, tangent, along_m / 2.0, across_m / 2.0, name)
                stats[f"{name}_symbols"] += 1
            placed.append((middle, tangent))
    return stats


COVER_TYPES = {
    "inspectie- / rioolput": "cover",
    "waterleidingput": "cover",
    "drainageput": "cover",
    "kolk": "grate",
    # Dutch hydrants are under the pavement: a small lid, and a yellow frame
    # round it so the fire brigade finds it.
    "brandkraan / -put": "hydrant",
}

HYDRANT_LID_M = (0.3, 0.45)
HYDRANT_FRAME_M = 0.08


def paint_covers(paint: GroundPaint, features: list[dict], bbox: BBox, roads: RoadIndex) -> dict:
    """Manhole lids, gully grates and hydrant lids from the BGT."""
    stats = {"manholes": 0, "gullies": 0, "hydrants": 0}
    lid = SHAPES["octagon"] * 0.35
    for feature in features:
        kind = COVER_TYPES.get(str((feature.get("properties") or {}).get("plus_type") or ""))
        geometry = feature.get("geometry") or {}
        if kind is None or geometry.get("type") != "Point":
            continue
        x, y = (float(v) for v in geometry["coordinates"][:2])
        if not (bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax):
            continue
        if kind == "cover":
            paint.polygon(lid + np.array([x, y]), flat_uv("cover"))
            stats["manholes"] += 1
        elif kind == "hydrant":
            found = roads.nearest((x, y), 20.0)
            along_road = found[1] if found is not None else np.array([1.0, 0.0])
            across_road = np.array([-along_road[1], along_road[0]])
            half_w, half_l = HYDRANT_LID_M[0] / 2.0, HYDRANT_LID_M[1] / 2.0
            frame = HYDRANT_FRAME_M
            centre = np.array([x, y])
            paint.quad(centre, along_road, half_l, half_w, flat_uv("cover"))
            # The frame as four strips round the lid, not a plate under it:
            # two coplanar faces fight over every pixel in a depth buffer.
            for side in (-1.0, 1.0):
                paint.quad(
                    centre + along_road * side * (half_l + frame / 2.0),
                    along_road, frame / 2.0, half_w + frame, flat_uv("yellow"),
                )
                paint.quad(
                    centre + across_road * side * (half_w + frame / 2.0),
                    along_road, half_l, frame / 2.0, flat_uv("yellow"),
                )
            stats["hydrants"] += 1
        else:
            # A gully lies along its kerb, which runs the way the road does.
            found = roads.nearest((x, y), 20.0)
            tangent = found[1] if found is not None else np.array([1.0, 0.0])
            paint.quad((x, y), tangent, 0.225, 0.15, flat_uv("grate"))
            stats["gullies"] += 1
    return stats


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------

OBJECTS = ("signs", "signals", "markings", "symbols", "covers")


@dataclass
class TrafficSet:
    """Four finished meshes in RD and NAP, and what went into them."""

    meshes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    signs: list[dict] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    crossings: list[dict] = field(default_factory=list)
    used_osm: bool = False
    # BGT pole types this stage drew, for the furniture stage to leave alone.
    replaces: set[str] = field(default_factory=set)
    # What it read, for the furniture stage to turn its pieces by and fill in
    # from, so neither is read twice. Not saved.
    osm: object = None
    roads: object = None
    carriageway: object = None

    def __len__(self) -> int:
        return sum(len(faces) for _, faces, _ in self.meshes.values())

    def triangles(self, name: str) -> int:
        return len(self.meshes[name][1]) if name in self.meshes else 0

    def stats(self) -> dict:
        return {"triangles": len(self), **self.counts}


def build_traffic(
    bbox: BBox,
    work_dir: Path,
    *,
    traffic_cfg: dict,
    terrain,
    deck_at: Callable | None = None,
    network=None,
    fetch_bgt: Callable | None = None,
    osm=None,
) -> TrafficSet:
    """Fetch what is switched on and build the meshes.

    `osm` is an OpenStreetMap read already made; without one the stage reads
    its own when lines or lights need it.
    """
    from .bgt import CACHE_DIRNAME, fetch_current
    from .osm import OSM_API_URL, OsmData, fetch_osm

    fetch_bgt = fetch_bgt or fetch_current
    cache_dir = work_dir / CACHE_DIRNAME
    want_signs = bool(traffic_cfg.get("signs", True))
    want_signals = bool(traffic_cfg.get("signals", True))
    want_markings = bool(traffic_cfg.get("markings", True))
    want_covers = bool(traffic_cfg.get("covers", True))
    result = TrafficSet()

    def height_at(x, y):
        z = np.asarray(terrain.surface(x, y), dtype=np.float64)
        if deck_at is not None:
            deck = np.asarray(deck_at(x, y), dtype=np.float64)
            z = np.where(np.isfinite(deck) & (deck > z), deck, z)
        return z

    def bgt(collection: str) -> list[dict]:
        try:
            features, _ = fetch_bgt(
                collection,
                bbox,
                class_field="plus_type",
                page_limit=int(traffic_cfg.get("page_limit", 1000)),
                timeout=float(traffic_cfg.get("timeout_s", 180)),
                max_retries=int(traffic_cfg.get("max_retries", 4)),
                max_pages=int(traffic_cfg.get("max_pages", 200)),
            )
            return features
        except Exception as exc:  # noqa: BLE001 - detail, not structure
            LOG.warning("could not read BGT %s: %s", collection, short_error(exc))
            return []

    if osm is None:
        osm = OsmData()
        if want_markings or want_signals:
            osm = fetch_osm(
                bbox.buffered(30.0),
                url=str(traffic_cfg.get("osm_url", OSM_API_URL)),
                timeout=float(traffic_cfg.get("timeout_s", 180)),
                cache_dir=cache_dir,
            )
    result.osm = osm
    # The nodes this stage paints or lights; the read itself stays whole.
    osm = OsmData(
        ways=osm.ways,
        nodes=[
            n for n in osm.nodes
            if bbox.xmin <= n.xy[0] <= bbox.xmax and bbox.ymin <= n.xy[1] <= bbox.ymax
        ],
        areas=osm.areas,
        source=osm.source,
    )

    road, cycle = Carriageway(), Carriageway()
    wegdeel: list[dict] = []
    if want_markings or want_signals:
        wegdeel = bgt("wegdeel")
        road, cycle = carriageways(wegdeel)

    roads = road_index(network, osm)
    result.roads = roads
    result.carriageway = road

    signs: list[Sign] = []
    directions = np.zeros((0, 2))
    road_points: list = []
    signal_stops: tuple = ([], np.zeros((0, 2)))
    if want_signs:
        signs = fetch_signs(
            bbox,
            url=str(traffic_cfg.get("signs_url", NDW_SIGNS_URL)),
            municipality_url=str(traffic_cfg.get("municipality_url", MUNICIPALITY_URL)),
            timeout=float(traffic_cfg.get("signs_timeout_s", 300)),
            cache_dir=cache_dir,
        )
    if signs:
        resolved = [travel_direction(sign, roads) for sign in signs]
        directions = np.array([d for d, _ in resolved], dtype=np.float64)
        road_points = [p for _, p in resolved]
        xy = np.array([s.xy for s in signs])
        mesh, result.signs = sign_mesh(signs, directions, height_at(xy[:, 0], xy[:, 1]))
        result.meshes["signs"] = mesh.arrays()
        result.replaces.update({"verkeersbordpaal", "verkeersbord"})
        result.counts["signs"] = len(result.signs)

    if want_signals:
        poles = np.array(
            [
                f["geometry"]["coordinates"][:2]
                for f in bgt("paal")
                if (f.get("properties") or {}).get("plus_type") == "verkeersregelinstallatiepaal"
                and (f.get("geometry") or {}).get("type") == "Point"
                and bbox.xmin <= f["geometry"]["coordinates"][0] <= bbox.xmax
                and bbox.ymin <= f["geometry"]["coordinates"][1] <= bbox.ymax
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        stop_points: list = []
        positions, headings, signal_stats = place_signals(
            poles, osm, roads, road, stops=stop_points
        )
        keep = (
            (positions[:, 0] >= bbox.xmin) & (positions[:, 0] <= bbox.xmax)
            & (positions[:, 1] >= bbox.ymin) & (positions[:, 1] <= bbox.ymax)
        ) if len(positions) else np.zeros(0, dtype=bool)
        positions, headings = positions[keep], headings[keep]
        stop_points = [s for s, k in zip(stop_points, keep) if k]
        signal_stops = (stop_points, headings)
        # The surveyed poles are drawn here now, whether or not any stand in
        # this square.
        result.replaces.add("verkeersregelinstallatiepaal")
        if len(positions):
            ground = height_at(positions[:, 0], positions[:, 1])
            result.meshes["signals"] = signal_mesh(positions, headings, ground).arrays()
            result.signals = [
                {
                    "rd": [round(float(x), 3), round(float(y), 3)],
                    "ground_z_nap": round(float(z), 3),
                    "facing_deg": round(float(np.degrees(np.arctan2(-d[0], -d[1])) % 360), 1),
                }
                for (x, y), d, z in zip(positions, headings, ground)
            ]
            result.counts["signals"] = len(positions)
            result.counts["signals_from_osm"] = int(signal_stats["from_osm"])
            if signal_stats["from_osm"]:
                result.used_osm = True

    if want_markings:
        paint = GroundPaint()
        crossing_stats, result.crossings = paint_crossings(paint, osm, road, cycle)
        lane_stats = paint_lanes(paint, osm, road, bbox, result.crossings)
        give_way = paint_give_way(paint, signs, directions, road_points, road, cycle)
        stop_stats = paint_stop_lines(paint, *signal_stops, road, cycle)
        bay_stats = (
            paint_parking_bays(paint, wegdeel, bbox)
            if bool(traffic_cfg.get("parking_bays", True)) else {}
        )
        if len(paint):
            result.meshes["markings"] = paint.mesh(
                height_at, float(traffic_cfg.get("marking_lift_m", 0.03))
            ).arrays()
        symbols = GroundPaint()
        symbol_stats = paint_symbols(symbols, osm, network, road, cycle, bbox)
        if len(symbols):
            result.meshes["symbols"] = symbols.mesh(
                height_at, float(traffic_cfg.get("marking_lift_m", 0.03))
            ).arrays()
        result.counts.update({
            **lane_stats, **crossing_stats, **give_way, **stop_stats, **bay_stats,
            **symbol_stats,
        })
        if lane_stats["laned_ways"] or result.crossings or crossing_stats["humps"]:
            result.used_osm = True

    if want_covers:
        paint = GroundPaint()
        result.counts.update(paint_covers(paint, bgt("put"), bbox, roads))
        if len(paint):
            result.meshes["covers"] = paint.mesh(
                height_at, float(traffic_cfg.get("cover_lift_m", 0.015))
            ).arrays()

    LOG.info(
        "traffic: %s",
        ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in result.counts.items() if v)
        or "nothing found",
    )
    if len(result):
        save_traffic(result, work_dir / "traffic.npz")
    return result


def save_traffic(traffic: TrafficSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, (vertices, faces, uv) in traffic.meshes.items():
        arrays[f"{name}_vertices"] = vertices
        arrays[f"{name}_faces"] = faces.astype(np.int32)
        arrays[f"{name}_uv"] = uv.astype(np.float32)
    np.savez_compressed(path, **arrays)
    LOG.info("wrote %s (%d triangles)", path, len(traffic))
    return path


def write_traffic_list(traffic: TrafficSet, geo, out_dir: Path, filename: str = "traffic.json") -> Path:
    """Signs, lights and crossings as a list, for prefabs and for logic.

    The baked meshes make the FBX look right; this says what each thing *is*,
    which a traffic simulation or a swap for real prefabs needs and a mesh
    cannot tell it.
    """

    def with_local(record: dict) -> dict:
        x, y = geo.rd_to_local(record["rd"][0], record["rd"][1])
        # Unity Z is RD northing; Y comes from the ground height.
        return {**record, "local": {"x": round(float(x), 3), "z": round(float(y), 3)}}

    payload = {
        "crs": "EPSG:28992",
        "note": (
            "local.x / local.z are metres in the model frame; add "
            "ground_z_nap minus metadata.ground_z_offset_nap for local Y. "
            "facing_deg is the compass direction a plate or lamp faces."
        ),
        "sources": {
            "signs": "NDW verkeersborden (RVV codes)",
            "signals": "BGT paal (verkeersregelinstallatiepaal), OpenStreetMap where not surveyed",
            "crossings": "OpenStreetMap",
        },
        "signs": [with_local(r) for r in traffic.signs],
        "signals": [with_local(r) for r in traffic.signals],
        "crossings": [with_local(r) for r in traffic.crossings],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    LOG.info("wrote %s", path)
    return path


__all__ = [
    "Carriageway",
    "GroundPaint",
    "MeshBuilder",
    "NDW_SIGNS_URL",
    "OBJECTS",
    "RoadIndex",
    "SIGN_FACES",
    "Sign",
    "TrafficSet",
    "build_traffic",
    "carriageways",
    "crossing_style",
    "face_key",
    "fetch_signs",
    "generate_traffic_texture",
    "paint_covers",
    "paint_crossings",
    "paint_give_way",
    "paint_lanes",
    "paint_parking_bays",
    "paint_stop_lines",
    "paint_symbols",
    "parse_signs",
    "place_signals",
    "road_index",
    "save_traffic",
    "sign_mesh",
    "signal_mesh",
    "travel_direction",
    "write_traffic_list",
]
