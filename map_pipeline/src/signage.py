"""Name boards over the shops: what a shopping street is recognised by.

The facade stage gives a building with shops in it a shopfront ground storey,
and every shopfront is the same shopfront. What tells one street from the next
is the names over them. OpenStreetMap has them: over the Utrecht centre
square kilometre 745 named shops, cafes and restaurants, each a point at or
just inside its own front.

Each becomes a board on the facade nearest it, above the shop window, facing
the street: the nearest wall whose outside is not another building. The name
is drawn into an atlas of its own (`shopsigns.png`), one cell per name, in a
colour picked from the name so a chain wears the same board everywhere.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .geo import BBox

LOG = logging.getLogger(__name__)

# Amenities that hang their name over the door.
SIGN_AMENITIES = frozenset({
    "restaurant", "cafe", "fast_food", "bar", "pub", "bank", "pharmacy",
    "ice_cream", "cinema", "theatre", "nightclub", "biergarten",
    "bureau_de_change", "dentist", "veterinary", "library",
})

CELL_W, CELL_H = 256, 64
# Board colours, background and lettering, picked by the name.
BOARDS = (
    ((24, 52, 38), (236, 226, 196)),    # bottle green, cream
    ((16, 28, 54), (240, 240, 236)),    # navy, white
    ((20, 20, 22), (230, 196, 120)),    # black, gold
    ((110, 24, 32), (244, 236, 222)),   # burgundy, off-white
    ((236, 232, 222), (30, 30, 34)),    # cream, black
    ((246, 246, 244), (150, 28, 36)),   # white, red
    ((42, 86, 120), (250, 250, 250)),   # steel blue, white
    ((86, 54, 36), (238, 220, 180)),    # brown, sand
)
BOARD_HEIGHT_M = 0.55
BOARD_DEPTH_M = 0.08
# The bottom of the board, above the pavement: over a shop window.
BOARD_BOTTOM_M = 2.9
# The furthest a named point may be from its facade.
FACADE_REACH_M = 8.0


@dataclass
class Shop:
    name: str
    kind: str
    xy: tuple[float, float]


@dataclass
class Board:
    shop: Shop
    point: np.ndarray       # on the facade, in plan
    outward: np.ndarray     # unit, out of the building
    width: float
    bottom_nap: float


def shops(osm, bbox: BBox) -> list[Shop]:
    """Named shops, cafes and the like inside `bbox`."""
    out = []
    for node in getattr(osm, "nodes", None) or []:
        tags = node.tags
        name = str(tags.get("name") or "").strip()
        if not name:
            continue
        kind = (
            tags.get("shop") or tags.get("craft")
            or (tags.get("amenity") if tags.get("amenity") in SIGN_AMENITIES else "")
        )
        if not kind or tags.get("disused:shop") or tags.get("level", "0") not in ("0", "0;1"):
            continue
        x, y = node.xy
        if bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax:
            out.append(Shop(name=name[:40], kind=str(kind), xy=(float(x), float(y))))
    return out


def board_width(name: str) -> float:
    return float(np.clip(0.2 * len(name) + 0.8, 1.4, 4.5))


def place_boards(
    found: list[Shop], footprints, ground_at: Callable, *, bottom_m: float = BOARD_BOTTOM_M,
) -> list[Board]:
    """A board for each shop on its facade, kept clear of its neighbours."""
    boards: list[Board] = []
    for shop in found:
        # The outside of the wall has to be the street, not the next house.
        facade = footprints.street_facade(shop.xy, FACADE_REACH_M)
        if facade is None:
            continue
        point, outward = facade
        width = board_width(shop.name)
        along = np.array([-outward[1], outward[0]])
        # Shift along the facade past any board already there.
        for _ in range(6):
            clash = next(
                (
                    b for b in boards
                    if float(b.outward @ outward) > 0.9
                    and abs(float((b.point - point) @ outward)) < 0.5
                    and abs(float((b.point - point) @ along)) < (b.width + width) / 2.0 + 0.2
                ),
                None,
            )
            if clash is None:
                break
            side = 1.0 if float((point - clash.point) @ along) >= 0 else -1.0
            point = clash.point + along * side * ((clash.width + width) / 2.0 + 0.25)
        ground = float(np.asarray(ground_at(np.array([point[0]]), np.array([point[1]])))[0])
        boards.append(Board(shop, point, outward, width, ground + bottom_m))
    return boards


def _colours(name: str):
    digest = int(hashlib.md5(name.lower().encode("utf-8")).hexdigest(), 16)
    return BOARDS[digest % len(BOARDS)]


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def atlas_layout(n_cells: int) -> tuple[int, int, int]:
    """Columns, and the atlas's width and height in pixels, for `n_cells`."""
    columns = 8 if n_cells <= 8 * 32 else 16
    rows = int(np.ceil(n_cells / columns))
    height = CELL_H
    while height < rows * CELL_H:
        height *= 2
    return columns, columns * CELL_W, height


def cell_rect(cell: int, columns: int, width: int, height: int) -> tuple[float, float, float, float]:
    """(u0, v0, u1, v1), V from the bottom, half a pixel in from the edges."""
    col, row = cell % columns, cell // columns
    u0 = (col * CELL_W + 0.5) / width
    u1 = ((col + 1) * CELL_W - 0.5) / width
    v1 = 1.0 - (row * CELL_H + 0.5) / height
    v0 = 1.0 - ((row + 1) * CELL_H - 0.5) / height
    return u0, v0, u1, v1


def generate_sign_texture(names: list[str], work_dir: Path) -> tuple[Path, dict[str, int], tuple]:
    """One cell per name, after cell 0, which is the board's dark back."""
    from PIL import Image, ImageDraw

    unique = list(dict.fromkeys(names))
    columns, width, height = atlas_layout(len(unique) + 1)
    image = Image.new("RGB", (width, height), (34, 34, 36))
    draw = ImageDraw.Draw(image)
    cells: dict[str, int] = {}
    for index, name in enumerate(unique, start=1):
        col, row = index % columns, index // columns
        x0, y0 = col * CELL_W, row * CELL_H
        background, letters = _colours(name)
        draw.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], fill=background)
        size = 42
        font = _font(size)
        box = draw.textbbox((0, 0), name, font=font)
        while (box[2] - box[0] > CELL_W - 20 or box[3] - box[1] > CELL_H - 14) and size > 10:
            size -= 2
            font = _font(size)
            box = draw.textbbox((0, 0), name, font=font)
        draw.text(
            (x0 + (CELL_W - (box[0] + box[2])) / 2.0, y0 + (CELL_H - (box[1] + box[3])) / 2.0),
            name, fill=letters, font=font,
        )
        cells[name] = index
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "shopsigns.png"
    image.save(path, optimize=True)
    LOG.info("wrote %s (%d names, %dx%d)", path.name, len(unique), width, height)
    return path, cells, (columns, width, height)


def board_mesh(boards: list[Board], cells: dict[str, int], layout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every board as a box: the name on the front, dark everywhere else."""
    columns, width, height = layout
    back = cell_rect(0, columns, width, height)
    back_uv = np.array([(back[0] + back[2]) / 2.0, (back[1] + back[3]) / 2.0])
    vertices, faces, uvs = [], [], []
    count = 0
    for board in boards:
        out = np.asarray(board.outward, dtype=np.float64)
        right = np.array([-out[1], out[0]])
        centre = np.asarray(board.point, dtype=np.float64) + out * (BOARD_DEPTH_M / 2.0 + 0.02)
        half_w, depth = board.width / 2.0, BOARD_DEPTH_M / 2.0
        z0, z1 = board.bottom_nap, board.bottom_nap + BOARD_HEIGHT_M
        # Front face first, then the box's other five.
        corners = {}
        for name, (a, b) in {
            "fbl": (-half_w, depth), "fbr": (half_w, depth),
            "rbl": (-half_w, -depth), "rbr": (half_w, -depth),
        }.items():
            corners[name] = centre + right * a + out * b
        quads = [
            ("front", [("fbl", z0), ("fbr", z0), ("fbr", z1), ("fbl", z1)]),
            ("back", [("rbr", z0), ("rbl", z0), ("rbl", z1), ("rbr", z1)]),
            ("side", [("rbl", z0), ("fbl", z0), ("fbl", z1), ("rbl", z1)]),
            ("side", [("fbr", z0), ("rbr", z0), ("rbr", z1), ("fbr", z1)]),
            ("top", [("fbl", z1), ("fbr", z1), ("rbr", z1), ("rbl", z1)]),
            ("bottom", [("rbl", z0), ("rbr", z0), ("fbr", z0), ("fbl", z0)]),
        ]
        rect = cell_rect(cells[board.shop.name], columns, width, height)
        for kind, quad in quads:
            points = np.array([[*corners[c], z] for c, z in quad])
            vertices.append(points)
            faces.append(np.array([[0, 1, 2], [0, 2, 3]]) + count)
            if kind == "front":
                u0, v0, u1, v1 = rect
                uvs.append(np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]))
            else:
                uvs.append(np.tile(back_uv, (4, 1)))
            count += 4
    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2))
    return np.vstack(vertices), np.vstack(faces), np.vstack(uvs)


@dataclass
class SignSet:
    boards: list[Board]
    texture: Path | None = None
    mesh: tuple | None = None

    def __len__(self) -> int:
        return len(self.boards)

    def stats(self) -> dict:
        return {"boards": len(self.boards), "names": len({b.shop.name for b in self.boards})}


def build_shop_signs(
    bbox: BBox, work_dir: Path, *, osm, buildings, ground_at: Callable,
) -> SignSet:
    """Boards for every named shop on its facade, and their atlas."""
    from .furniture import Footprints

    found = shops(osm, bbox)
    if not found:
        LOG.info("shop signs: no named shops here")
        return SignSet([])
    footprints = Footprints(buildings)
    boards = place_boards(found, footprints, ground_at)
    if not boards:
        LOG.info("shop signs: %d named shops, none on a facade", len(found))
        return SignSet([])
    texture, cells, layout = generate_sign_texture([b.shop.name for b in boards], work_dir)
    vertices, faces, uv = board_mesh(boards, cells, layout)
    np.savez_compressed(
        work_dir / "shopsigns.npz",
        signs_vertices=vertices, signs_faces=faces.astype(np.int32), signs_uv=uv.astype(np.float32),
    )
    LOG.info("shop signs: %d boards for %d named shops", len(boards), len(found))
    return SignSet(boards, texture, (vertices, faces, uv))


def write_shop_list(signs: SignSet, geo, out_dir: Path, filename: str = "shops.json") -> Path:
    records = []
    for board in signs.boards:
        x, y = geo.rd_to_local(float(board.point[0]), float(board.point[1]))
        records.append({
            "name": board.shop.name,
            "kind": board.shop.kind,
            "rd": [round(float(board.point[0]), 3), round(float(board.point[1]), 3)],
            "local": {"x": round(float(x), 3), "z": round(float(y), 3)},
            "board_bottom_nap": round(float(board.bottom_nap), 2),
            "facing_deg": round(float(np.degrees(np.arctan2(board.outward[0], board.outward[1])) % 360), 1),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps({
        "crs": "EPSG:28992",
        "source": "OpenStreetMap (ODbL)",
        "note": "One board per named shop, on the facade nearest it; facing_deg is the compass direction it faces.",
        "shops": records,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = [
    "SIGN_AMENITIES",
    "Board",
    "Shop",
    "SignSet",
    "atlas_layout",
    "board_mesh",
    "build_shop_signs",
    "generate_sign_texture",
    "place_boards",
    "shops",
    "write_shop_list",
]
