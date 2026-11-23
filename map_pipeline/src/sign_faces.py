"""What the traffic signs show: one atlas holding every face an area needs.

The artwork is the RVV 1990's own, as Wikimedia Commons publishes it: one SVG
per sign, named ``Nederlands verkeersbord C2.svg`` and so on. The signs are an
appendix to a regulation, and a regulation carries no copyright in the
Netherlands (Auteurswet article 11), which is why Commons marks them public
domain and why nothing about them has to travel with the model.

Commons renders every SVG to PNG at a fixed address on upload.wikimedia.org,
derived from the MD5 of the file name. Fetching that thumbnail rather than the
SVG means no SVG renderer to install, which on Windows is the difference
between a dependency and a support ticket. The API is not used at all: it is
rate limited per address, and a shared office or cloud address hits that limit
after a request or two.

Each file is fetched once, at one request a second, and kept under
``work/_signs/`` for every later area. A name Commons does not have is
remembered too, so it is not asked for again. Whatever cannot be had -- no
network, a code Commons names differently, a limit that will not lift -- is
drawn here instead: the right shape and colours, the number on a speed sign,
a white disc with a red ring for a prohibition. From across a street that
reads as the sign; up close it is a stand-in, and the summary says how many
there were.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

COMMONS_THUMB = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/{a}/{ab}/{name}/{px}px-{name}.png"
)
# Commons renders thumbnails at a fixed set of widths; these two are on it.
THUMB_WIDTHS = (500, 250)
LICENCE = (
    "Traffic sign artwork: RVV 1990 signs from Wikimedia Commons, public domain "
    "(Auteurswet article 11). https://commons.wikimedia.org"
)

# One cell of the atlas, in pixels. A sign is 60 cm across, so 256 px is about
# 2.3 mm a pixel: sharper than anyone standing on the pavement can see.
CELL_PX = 256
# Supersampling for the drawn faces: drawn at four times the size and shrunk,
# so a red ring has a soft edge rather than a staircase.
SUPERSAMPLE = 4
PAD_PX = 4

# The RVV colours, near enough. Sign red, sign blue and sign yellow are RAL
# 3020, 5017 and 1023 in the regulation.
RED = (196, 30, 34)
BLUE = (0, 84, 150)
YELLOW = (250, 194, 0)
WHITE = (250, 250, 248)
BLACK = (24, 24, 24)
GREY = (120, 120, 120)
# The back of a sign and its pole: bare aluminium.
METAL = (150, 153, 156)

UNKNOWN = "onbekend"

# The shape of each sign, by code. It sets the plate's size as much as its
# outline: a warning triangle is half as wide again as a round sign.
SHAPE_ROUND = "round"
SHAPE_TRIANGLE = "triangle"
SHAPE_TRIANGLE_DOWN = "triangle_down"
SHAPE_OCTAGON = "octagon"
SHAPE_DIAMOND = "diamond"
SHAPE_SQUARE = "square"
SHAPE_RECT = "rect"
SHAPE_PLATE = "plate"  # an onderbord: a white plate under the sign

# Width of each shape against a round sign's diameter. Dutch signs come in
# series (a 600 mm disc goes with a 900 mm triangle), which is what this is.
SHAPE_WIDTH = {
    SHAPE_ROUND: 1.0,
    SHAPE_OCTAGON: 1.0,
    SHAPE_DIAMOND: 1.0,
    SHAPE_SQUARE: 1.0,
    SHAPE_TRIANGLE: 1.5,
    SHAPE_TRIANGLE_DOWN: 1.5,
    SHAPE_RECT: 1.6,
    SHAPE_PLATE: 1.0,
}

# Order on a pole, top first: priority, then prohibitions and limits, then
# what is compulsory, then the rest. Below its own sign, each onderbord.
CATEGORY_RANK = {c: i for i, c in enumerate("BACFDJGELHK")}


def code_parts(code: str) -> tuple[str, int, str]:
    """``"G12a"`` -> ``("G", 12, "a")``; anything else -> ``("", 0, code)``."""
    match = re.fullmatch(r"([A-Z]{1,2})(\d+)([a-z]?)(?:-.*)?", str(code or "").strip())
    if not match:
        return "", 0, str(code or "")
    return match.group(1), int(match.group(2)), match.group(3)


def shape_of(code: str) -> str:
    """The outline of a sign, from its code alone."""
    letter, number, _ = code_parts(code)
    if letter == "A":
        return SHAPE_SQUARE if number in (4, 5) else SHAPE_ROUND
    if letter == "B":
        if number in (1, 2):
            return SHAPE_DIAMOND
        if number == 6:
            return SHAPE_TRIANGLE_DOWN
        if number == 7:
            return SHAPE_OCTAGON
        return SHAPE_TRIANGLE
    if letter in ("C", "D"):
        return SHAPE_ROUND
    if letter == "E":
        return SHAPE_ROUND if number <= 3 else SHAPE_SQUARE
    if letter == "F":
        return SHAPE_ROUND if number <= 5 else SHAPE_SQUARE
    if letter == "G":
        return SHAPE_ROUND if 7 <= number <= 13 else SHAPE_RECT
    if letter == "J":
        return SHAPE_TRIANGLE
    if letter == "L":
        return SHAPE_SQUARE
    return SHAPE_RECT


def rank_of(code: str) -> int:
    letter, number, _ = code_parts(code)
    return CATEGORY_RANK.get(letter[:1], len(CATEGORY_RANK)) * 100 + min(number, 99)


def black_text(code: str, black: str) -> str:
    """What a sign's black code puts on its face, as the sign prints it.

    NDW gives speed, length, width, height and weight as bare numbers, with a
    decimal point and sometimes a trailing ``.0``. The sign itself writes
    ``2,2 m``.
    """
    black = str(black or "").strip()
    if not black:
        return ""
    letter, number, _ = code_parts(code)
    try:
        value = float(black.replace(",", "."))
    except ValueError:
        return black if letter == "H" else ""
    text = f"{value:g}".replace(".", ",")
    if letter == "C" and number in (17, 18, 19):
        return f"{text} m"
    if letter == "C" and number in (20, 21):
        return f"{text} t"
    return text


def commons_names(code: str, black: str = "") -> list[str]:
    """File names to try on Commons for a sign, most specific first."""
    base = str(code or "").strip()
    if not base or base == UNKNOWN:
        return []
    names = []
    black = str(black or "").strip()
    if black and re.fullmatch(r"[0-9A-Za-z.,]+", black):
        names.append(f"Nederlands verkeersbord {base}-{black.replace(',', '.')}.svg")
    names.append(f"Nederlands verkeersbord {base}.svg")
    # NDW appends qualifiers the regulation does not have ("E8-vracht").
    plain = re.sub(r"-.*$", "", base)
    if plain != base:
        names.append(f"Nederlands verkeersbord {plain}.svg")
    return names


def sub_commons_names(code: str) -> list[str]:
    code = str(code or "").strip()
    if not re.fullmatch(r"OB\d+[a-z]?", code, flags=re.I):
        return []
    return [f"Nederlands verkeersbord {code}.svg", f"Nederlands onderbord {code}.svg"]


def thumb_url(name: str, width: int) -> str:
    """Where Commons keeps its PNG rendering of a file, without asking."""
    file_name = name.replace(" ", "_")
    digest = hashlib.md5(file_name.encode("utf-8")).hexdigest()
    return COMMONS_THUMB.format(a=digest[0], ab=digest[:2], name=file_name, px=width)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


class CommonsClient:
    """Thumbnails from Commons, paced, cached, and given up on gracefully.

    One request a second, and a 429 waits as long as Commons asks, up to a
    limit. After that many refusals in a row the rest of the run draws its own
    faces rather than spend minutes being told no.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        enabled: bool = True,
        pause_s: float = 1.0,
        timeout_s: float = 30.0,
        max_wait_s: float = 30.0,
        give_up_after: int = 3,
        session=None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.pause_s = pause_s
        self.timeout_s = timeout_s
        self.max_wait_s = max_wait_s
        self.give_up_after = give_up_after
        self.session = session
        self.refusals = 0
        self.fetched = 0
        self.from_cache = 0
        self._last = 0.0
        self._missing_path = self.cache_dir / "missing.json"
        try:
            self._missing = set(json.loads(self._missing_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            self._missing = set()

    @property
    def given_up(self) -> bool:
        return self.refusals >= self.give_up_after

    def _cached(self, name: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        return self.cache_dir / f"{safe}.png"

    def get(self, names: list[str]):
        """The first of `names` Commons has, as an RGBA image, or None."""
        from PIL import Image

        for name in names:
            path = self._cached(name)
            if path.is_file():
                try:
                    image = Image.open(path).convert("RGBA")
                    self.from_cache += 1
                    return image
                except OSError:
                    path.unlink(missing_ok=True)
        if not self.enabled:
            return None
        for name in names:
            if name in self._missing or self.given_up:
                continue
            payload = self._download(name)
            if payload is None:
                continue
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cached(name)
            path.write_bytes(payload)
            try:
                return Image.open(path).convert("RGBA")
            except OSError:
                path.unlink(missing_ok=True)
        return None

    def _download(self, name: str) -> bytes | None:
        import requests

        from .http_util import USER_AGENT

        http = self.session or requests
        for width in THUMB_WIDTHS:
            while not self.given_up:
                wait = self.pause_s - (time.monotonic() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
                try:
                    response = http.get(
                        thumb_url(name, width),
                        timeout=self.timeout_s,
                        headers={"User-Agent": USER_AGENT},
                    )
                except requests.RequestException as exc:
                    LOG.info("Commons: %s: %s", name, exc)
                    self.refusals += 1
                    continue
                status = response.status_code
                if status == 200 and response.content[:8] == b"\x89PNG\r\n\x1a\n":
                    self.refusals = 0
                    self.fetched += 1
                    return response.content
                if status == 429 or status >= 500:
                    self.refusals += 1
                    try:
                        after = float(response.headers.get("Retry-After", "5"))
                    except ValueError:
                        after = 5.0
                    if self.given_up:
                        LOG.warning(
                            "Commons refused %d times in a row; drawing the "
                            "remaining sign faces instead",
                            self.refusals,
                        )
                        return None
                    time.sleep(min(max(after, 1.0), self.max_wait_s))
                    continue
                # 404: not at this width, or not on Commons at all. Try the
                # other width before calling the name missing.
                break
        if not self.given_up:
            self._missing.add(name)
            self._save_missing()
        return None

    def _save_missing(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._missing_path.write_text(
                json.dumps(sorted(self._missing), indent=0), encoding="utf-8"
            )
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _font(size: int, bold: bool = True):
    from PIL import ImageFont

    for name in (
        ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf")
        if bold
        else ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf")
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow before 10.1 has one size only
        return ImageFont.load_default()


def _fit_text(draw, text: str, box_w: float, box_h: float, start: int, bold=True):
    """The largest font, no bigger than `start`, that fits `text` in the box."""
    size = start
    while size > 8:
        font = _font(size, bold)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        if right - left <= box_w and bottom - top <= box_h:
            return font
        size = int(size * 0.88)
    return _font(8, bold)


def _centred_text(draw, text, cx, cy, box_w, box_h, colour, start, bold=True):
    font = _fit_text(draw, text, box_w, box_h, start, bold)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (cx - (left + right) / 2, cy - (top + bottom) / 2), text, fill=colour, font=font
    )


def _arrow(draw, cx, cy, length, angle_deg, colour, width):
    """A plain arrow, `angle_deg` anticlockwise from pointing right."""
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), -math.sin(a)
    x0, y0 = cx - dx * length / 2, cy - dy * length / 2
    x1, y1 = cx + dx * length * 0.22, cy + dy * length * 0.22
    draw.line([(x0, y0), (x1, y1)], fill=colour, width=int(width))
    tip = (cx + dx * length / 2, cy + dy * length / 2)
    side = (-dy, dx)
    head = length * 0.32
    base = (tip[0] - dx * head, tip[1] - dy * head)
    draw.polygon(
        [
            tip,
            (base[0] + side[0] * head * 0.7, base[1] + side[1] * head * 0.7),
            (base[0] - side[0] * head * 0.7, base[1] - side[1] * head * 0.7),
        ],
        fill=colour,
    )


def _regular(cx, cy, r, n, rotation_deg):
    return [
        (
            cx + r * math.cos(math.radians(rotation_deg + 360.0 * i / n)),
            cy - r * math.sin(math.radians(rotation_deg + 360.0 * i / n)),
        )
        for i in range(n)
    ]


def draw_face(code: str, black: str = ""):
    """A stand-in for a sign's artwork: its shape, its colours, its number.

    Drawn when Commons cannot supply the real one. It gets right what is seen
    from across a street -- a red-ringed disc, a blue square with a P, a stop
    sign -- and makes no attempt at the pictograms.
    """
    from PIL import Image, ImageDraw

    size = CELL_PX * SUPERSAMPLE
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    letter, number, suffix = code_parts(code)
    shape = shape_of(code) if code != UNKNOWN else SHAPE_SQUARE
    label = black_text(code, black)
    c = size / 2
    r = size * 0.48
    ring = size * 0.09

    def disc(fill, outline=None, rim=0.0):
        if outline is not None:
            draw.ellipse([c - r, c - r, c + r, c + r], fill=outline)
            draw.ellipse([c - r + rim, c - r + rim, c + r - rim, c + r - rim], fill=fill)
        else:
            draw.ellipse([c - r, c - r, c + r, c + r], fill=fill)

    def square(fill, rim_colour=WHITE, rim=size * 0.03):
        draw.rounded_rectangle([c - r, c - r, c + r, c + r], radius=size * 0.05, fill=rim_colour)
        draw.rounded_rectangle(
            [c - r + rim, c - r + rim, c + r - rim, c + r - rim], radius=size * 0.04, fill=fill
        )

    def triangle(down=False):
        rot = -90 if down else 90
        outer = _regular(c, c + (-1 if down else 1) * size * 0.08, r * 1.12, 3, rot)
        inner = _regular(c, c + (-1 if down else 1) * size * 0.08, r * 1.12 - ring * 2.1, 3, rot)
        draw.polygon(outer, fill=RED)
        draw.polygon(inner, fill=WHITE)

    if code == UNKNOWN:
        square(WHITE, GREY, size * 0.025)
    elif shape == SHAPE_OCTAGON:
        draw.polygon(_regular(c, c, r, 8, 22.5), fill=WHITE)
        draw.polygon(_regular(c, c, r - size * 0.02, 8, 22.5), fill=RED)
        _centred_text(draw, "STOP", c, c, r * 1.5, r * 0.7, WHITE, int(size * 0.3))
    elif shape == SHAPE_DIAMOND:
        draw.polygon(_regular(c, c, r, 4, 90), fill=BLACK)
        draw.polygon(_regular(c, c, r - size * 0.015, 4, 90), fill=WHITE)
        draw.polygon(_regular(c, c, r * 0.62, 4, 90), fill=YELLOW)
        if number == 2:
            draw.line([(c - r * 0.5, c + r * 0.5), (c + r * 0.5, c - r * 0.5)],
                      fill=BLACK, width=int(size * 0.06))
    elif shape in (SHAPE_TRIANGLE, SHAPE_TRIANGLE_DOWN):
        triangle(down=shape == SHAPE_TRIANGLE_DOWN)
        if shape == SHAPE_TRIANGLE:
            cy = c + size * 0.14
            if letter == "B" and number == 3:
                draw.rectangle([c - size * 0.03, cy - size * 0.17, c + size * 0.03, cy + size * 0.13], fill=BLACK)
                draw.rectangle([c - size * 0.12, cy - size * 0.05, c + size * 0.12, cy + size * 0.01], fill=BLACK)
            else:
                _centred_text(draw, "!", c, cy, size * 0.3, size * 0.32, BLACK, int(size * 0.34))
    elif letter == "A" and shape == SHAPE_ROUND:
        if number == 2:
            disc(WHITE, BLACK, size * 0.012)
            _centred_text(draw, label, c, c, r * 1.3, r * 0.9, GREY, int(size * 0.42))
            for k in (-1, 0, 1):
                off = k * size * 0.07
                draw.line([(c - r * 0.65 + off, c + r * 0.65 + off), (c + r * 0.65 + off, c - r * 0.65 + off)],
                          fill=BLACK, width=int(size * 0.025))
        else:
            disc(WHITE, RED, ring)
            if label:
                _centred_text(draw, label, c, c, r * 1.3, r * 0.95, BLACK, int(size * 0.5))
    elif letter == "A":  # A4, A5: advised speed, blue square
        square(BLUE)
        if label:
            _centred_text(draw, label, c, c, r * 1.4, r * 1.0, WHITE, int(size * 0.5))
        if number == 5:
            draw.line([(c - r * 0.8, c + r * 0.8), (c + r * 0.8, c - r * 0.8)], fill=RED, width=int(size * 0.05))
    elif letter == "C" and number == 2:
        disc(RED)
        draw.rectangle([c - r * 0.72, c - r * 0.17, c + r * 0.72, c + r * 0.17], fill=WHITE)
    elif letter == "C":
        disc(WHITE, RED, ring)
        if label:
            _centred_text(draw, label, c, c, r * 1.3, r * 0.8, BLACK, int(size * 0.34))
        elif number not in (1,):
            # A pictogram's weight in the middle, without pretending to be one.
            draw.rounded_rectangle([c - r * 0.45, c - r * 0.2, c + r * 0.45, c + r * 0.25],
                                   radius=size * 0.03, fill=BLACK)
    elif letter == "D":
        disc(BLUE, WHITE, size * 0.012)
        angle = {"RO": -45, "LO": -135, "R": 0, "L": 180}.get(str(black).upper(), 90)
        if number == 2 and angle == 90:
            angle = -45
        _arrow(draw, c, c, r * 1.15, angle, WHITE, size * 0.09)
    elif letter == "E" and number <= 3:
        disc(BLUE, RED, ring)
        width = int(size * 0.075)
        draw.line([(c - r * 0.62, c - r * 0.62), (c + r * 0.62, c + r * 0.62)], fill=RED, width=width)
        if number == 2:
            draw.line([(c - r * 0.62, c + r * 0.62), (c + r * 0.62, c - r * 0.62)], fill=RED, width=width)
    elif letter == "E":
        square(BLUE)
        _centred_text(draw, "P", c, c - size * 0.02, r * 1.2, r * 1.2, WHITE, int(size * 0.7))
    elif letter == "F" and shape == SHAPE_ROUND:
        disc(WHITE, RED, ring)
    elif letter == "F":
        square(BLUE)
        _arrow(draw, c - r * 0.3, c, r * 1.1, 90, WHITE, size * 0.07)
        _arrow(draw, c + r * 0.3, c, r * 1.1, -90, RED, size * 0.07)
    elif letter == "G" and shape == SHAPE_ROUND:
        disc(BLUE, WHITE, size * 0.012)
        if number in (11, 12, 13):
            for dx in (-0.38, 0.38):
                draw.ellipse([c + r * dx - r * 0.28, c + r * 0.05 - r * 0.28,
                              c + r * dx + r * 0.28, c + r * 0.05 + r * 0.28],
                             outline=WHITE, width=int(size * 0.035))
            draw.line([(c - r * 0.38, c + r * 0.05), (c, c - r * 0.35), (c + r * 0.38, c + r * 0.05)],
                      fill=WHITE, width=int(size * 0.035))
        else:
            draw.ellipse([c - r * 0.12, c - r * 0.62, c + r * 0.12, c - r * 0.38], fill=WHITE)
            draw.line([(c, c - r * 0.35), (c, c + r * 0.15)], fill=WHITE, width=int(size * 0.07))
            draw.line([(c, c + r * 0.15), (c - r * 0.25, c + r * 0.6)], fill=WHITE, width=int(size * 0.06))
            draw.line([(c, c + r * 0.15), (c + r * 0.25, c + r * 0.6)], fill=WHITE, width=int(size * 0.06))
    elif letter == "L":
        square(BLUE)
        draw.rectangle([c - r * 0.62, c - r * 0.62, c + r * 0.62, c + r * 0.62], fill=WHITE)
        if number == 8:
            draw.rectangle([c - r * 0.12, c - r * 0.1, c + r * 0.12, c + r * 0.55], fill=BLACK)
            draw.rectangle([c - r * 0.45, c - r * 0.35, c + r * 0.45, c - r * 0.12], fill=RED)
        else:
            draw.polygon(_regular(c, c + r * 0.08, r * 0.5, 3, 90), fill=BLACK)
    else:  # G1-G6, H, K and everything unplaced: a blue or white rectangle
        fill = WHITE if letter == "K" else BLUE
        text_colour = BLACK if letter == "K" else WHITE
        draw.rounded_rectangle([c - r, c - r * 0.62, c + r, c + r * 0.62], radius=size * 0.04, fill=WHITE)
        draw.rounded_rectangle([c - r + size * 0.025, c - r * 0.62 + size * 0.025,
                                c + r - size * 0.025, c + r * 0.62 - size * 0.025],
                               radius=size * 0.03, fill=fill)
        if label:
            _centred_text(draw, label, c, c, r * 1.7, r * 0.8, text_colour, int(size * 0.2))
    return image.resize((CELL_PX, CELL_PX), Image.LANCZOS)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = str(text).split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


# Texts NDW records for an onderbord nobody could read, or a placeholder. The
# plate is real; what is on it is not known.
BLANK_TEXTS = {"", "onleesbaar", "autodate", "onbekend", "none"}


# NDW writes a pictogram on a plate as its name in double brackets.
PICTOGRAMS = {"route-links": "\u2190", "route-rechts": "\u2192"}
MAX_PLATE_LINES = 6


def clean_text(text) -> str:
    """An onderbord's text as the plate prints it.

    Six texts in Utrecht's register were stored as UTF-8 read as Latin-1, some
    of them twice over, so "bij groen één auto" arrived as a string of accented
    capitals. Undone a layer at a time until it stops changing. A pictogram,
    "[[bus]]", becomes its name, or an arrow for the two route arrows.
    """
    text = str(text or "")
    for _ in range(4):
        if not re.search("[\u00c2\u00c3\u00c4\u00e2]", text):
            break
        repaired = None
        for codec in ("cp1252", "latin-1"):
            try:
                repaired = text.encode(codec).decode("utf-8")
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        if repaired is None or repaired == text:
            break
        text = repaired
    text = re.sub(
        r"\[\[([^\]]+)\]\]",
        lambda m: PICTOGRAMS.get(m.group(1).strip().lower(), m.group(1).replace("-", " ")),
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def plate_lines(text: str) -> list[str]:
    """The text in lines, as many as it needs up to six, none of it dropped.

    Half of Utrecht's plates say ten characters or fewer; one says 258. A long
    one gets a taller plate and longer lines rather than a font nobody can read
    or a sentence that stops halfway.
    """
    text = clean_text(text)
    if text.lower() in BLANK_TEXTS:
        return []
    width = 16
    lines = _wrap(text, width)
    while len(lines) > MAX_PLATE_LINES:
        width += 4
        lines = _wrap(text, width)
    return lines


def draw_plate(text: str):
    """An onderbord: a white plate, a thin black edge, the text on it.

    Three to one, or taller for three lines, as the plates under Dutch signs
    are. The text is what NDW recorded -- "uitgezonderd", "ma t/m za
    08.00-18.00 h" -- and is the part of a street that tells you which city
    you are in.
    """
    from PIL import Image, ImageDraw

    lines = plate_lines(text)
    rows = max(1, len(lines))
    width = CELL_PX * SUPERSAMPLE
    height = int(width * (0.34 if rows == 1 else 0.24 + 0.16 * rows))
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    rim = width * 0.012
    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=width * 0.03, fill=BLACK)
    draw.rounded_rectangle([rim, rim, width - 1 - rim, height - 1 - rim], radius=width * 0.025, fill=WHITE)
    if lines:
        line_h = (height - 4 * rim) / rows
        longest = max(lines, key=len)
        font = _fit_text(draw, longest, width * 0.88, line_h * 0.8, int(line_h * 0.75), bold=False)
        for i, line in enumerate(lines):
            left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
            cy = 2 * rim + line_h * (i + 0.5)
            draw.text((width / 2 - (left + right) / 2, cy - (top + bottom) / 2), line, fill=BLACK, font=font)
    return image.resize((CELL_PX, max(8, height // SUPERSAMPLE)), Image.LANCZOS)


def overlay_number(image, code: str, black: str):
    """Put a speed sign's own number on Commons artwork that shows another.

    Commons has one A1 with a sample number on it; the street has 30, 50 and
    15. The disc inside the red ring is painted white and the number drawn.
    """
    from PIL import ImageDraw

    letter, number, _ = code_parts(code)
    label = black_text(code, black)
    if not (letter == "A" and number == 1 and label):
        return image
    image = image.copy()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    alpha = np.asarray(image)[:, :, 3]
    ys, xs = np.nonzero(alpha > 128)
    if not len(xs):
        return image
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    r = (xs.max() - xs.min()) / 2 * 0.74
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=WHITE + (255,))
    _centred_text(draw, label, cx, cy, r * 1.6, r * 1.15, BLACK, int(r * 1.2))
    return image


def silhouette(image, colour=METAL):
    """The back of a sign: its outline, in bare aluminium."""
    from PIL import Image

    rgba = np.asarray(image.convert("RGBA")).copy()
    rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2] = colour
    return Image.fromarray(rgba, "RGBA")


def _bleed(rgba: np.ndarray, passes: int = 6) -> np.ndarray:
    """Spread colour into the clear pixels around a shape.

    A clear pixel still has a colour, and mipmapping and filtering average it
    in at the edge: left black, every round sign gets a dark fringe at a
    distance. Each pass gives a clear pixel the mean of its opaque neighbours.
    """
    out = rgba.astype(np.float32)
    solid = out[:, :, 3] > 8
    for _ in range(passes):
        if solid.all():
            break
        weight = solid.astype(np.float32)
        acc = np.zeros(out.shape[:2] + (3,), np.float32)
        count = np.zeros(out.shape[:2], np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            acc += np.roll(out[:, :, :3] * weight[:, :, None], (dy, dx), axis=(0, 1))
            count += np.roll(weight, (dy, dx), axis=(0, 1))
        grow = (~solid) & (count > 0)
        out[grow, :3] = acc[grow] / count[grow, None]
        solid = solid | grow
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# The atlas
# ---------------------------------------------------------------------------


@dataclass
class Face:
    """One picture in the atlas, and where it went."""

    key: tuple
    image: object  # PIL RGBA
    source: str  # "commons", "drawn" or "plate"
    shape: str
    uv: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # u0, v0, u1, v1
    aspect: float = 1.0  # width over height of what is drawn
    back: tuple | None = None  # key of the silhouette behind it


@dataclass
class Atlas:
    faces: dict
    metal_uv: tuple[float, float]
    path: Path | None = None
    size_px: tuple[int, int] = (0, 0)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for face in self.faces.values():
            out[face.source] = out.get(face.source, 0) + 1
        return out


def _crop(image):
    """Trim the clear margin, so the plate is the size of the sign."""
    alpha = np.asarray(image)[:, :, 3]
    ys, xs = np.nonzero(alpha > 16)
    if not len(xs):
        return image
    return image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def _to_cell(image, max_w: int = CELL_PX, max_h: int = CELL_PX):
    from PIL import Image

    image = _crop(image)
    w, h = image.size
    scale = min(max_w / w, max_h / h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def build_atlas(
    signs: list[tuple[str, str]],
    plates: list[tuple[str, str]],
    out_path: Path,
    *,
    client: CommonsClient | None = None,
    width_px: int = 2048,
) -> Atlas:
    """Fetch or draw every face, pack them, and write the atlas.

    `signs` are (code, black code) pairs; `plates` are (onderbord code, text)
    pairs. The backs come free: one silhouette per distinct outline, which is
    a handful however many signs there are.
    """
    from PIL import Image

    faces: dict = {}
    for code, black in dict.fromkeys(signs):
        key = ("sign", code, black)
        image = None
        if client is not None and code != UNKNOWN:
            image = client.get(commons_names(code, black))
        source = "commons"
        if image is None:
            image, source = draw_face(code, black), "drawn"
        else:
            image = overlay_number(image, code, black)
        faces[key] = Face(key, _to_cell(image), source, shape_of(code) if code != UNKNOWN else SHAPE_SQUARE)

    for code, text in dict.fromkeys(plates):
        key = ("plate", code, text)
        image, source = None, "plate"
        if client is not None and not plate_lines(text):
            image = client.get(sub_commons_names(code))
            source = "commons" if image is not None else "plate"
        if image is None:
            image = draw_plate(text)
        faces[key] = Face(key, _to_cell(image), source, SHAPE_PLATE)

    # The backs: one per distinct outline, matched on a coarse copy of it.
    backs: dict = {}
    for face in list(faces.values()):
        mask = np.asarray(face.image.resize((24, 24)))[:, :, 3] > 128
        if mask.all():
            continue  # rectangular: the plain metal patch does
        signature = ("back", face.image.size, hashlib.md5(mask.tobytes()).hexdigest()[:12])
        face.back = signature
        if signature not in backs:
            backs[signature] = Face(signature, silhouette(face.image), "back", face.shape)
    faces.update(backs)

    # Shelf packing, tallest first, so the rows waste little.
    order = sorted(faces.values(), key=lambda f: (-f.image.size[1], -f.image.size[0], str(f.key)))
    x = y = PAD_PX
    row_h = 0
    placed = []
    metal_box = (PAD_PX, PAD_PX, 32, 32)
    x = metal_box[2] + 2 * PAD_PX
    row_h = metal_box[3]
    for face in order:
        w, h = face.image.size
        if x + w + PAD_PX > width_px:
            x = PAD_PX
            y += row_h + 2 * PAD_PX
            row_h = 0
        placed.append((face, x, y))
        x += w + 2 * PAD_PX
        row_h = max(row_h, h)
    # A multiple of 64 rather than a power of two: rounding 4500 px up to 8192
    # nearly doubled the atlas for nothing, and every engine since about 2010
    # takes a texture of any size.
    height_px = int(math.ceil((y + row_h + PAD_PX) / 64.0)) * 64

    canvas = np.zeros((height_px, width_px, 4), np.uint8)
    mx, my, mw, mh = metal_box
    canvas[my - PAD_PX : my + mh + PAD_PX, mx - PAD_PX : mx + mw + PAD_PX] = METAL + (255,)
    for face, px, py in placed:
        rgba = np.asarray(face.image.convert("RGBA"))
        h, w = rgba.shape[:2]
        # Padded on every side, then bled, so filtering at the edge of a cell
        # reads the cell's own colour rather than its neighbour's.
        block = np.zeros((h + 2 * PAD_PX, w + 2 * PAD_PX, 4), np.uint8)
        block[PAD_PX : PAD_PX + h, PAD_PX : PAD_PX + w] = rgba
        block = _bleed(block, passes=PAD_PX + 2)
        canvas[py - PAD_PX : py + h + PAD_PX, px - PAD_PX : px + w + PAD_PX] = block
        # Half a texel in from the cell's edge, V counted up from the bottom.
        face.uv = (
            (px + 0.5) / width_px,
            1.0 - (py + h - 0.5) / height_px,
            (px + w - 0.5) / width_px,
            1.0 - (py + 0.5) / height_px,
        )
        face.aspect = w / h
    # Fully clear pixels stay clear, whatever colour bled into them.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, "RGBA").save(out_path, optimize=True)
    metal_uv = ((mx + mw / 2) / width_px, 1.0 - (my + mh / 2) / height_px)
    atlas = Atlas(faces=faces, metal_uv=metal_uv, path=out_path, size_px=(width_px, height_px))
    LOG.info(
        "sign atlas: %d faces (%s), %d x %d px",
        len(faces),
        ", ".join(f"{v} {k}" for k, v in sorted(atlas.counts().items())),
        width_px,
        height_px,
    )
    return atlas


__all__ = [
    "Atlas",
    "CommonsClient",
    "Face",
    "LICENCE",
    "SHAPE_PLATE",
    "SHAPE_RECT",
    "SHAPE_WIDTH",
    "UNKNOWN",
    "black_text",
    "build_atlas",
    "clean_text",
    "code_parts",
    "commons_names",
    "draw_face",
    "draw_plate",
    "plate_lines",
    "rank_of",
    "shape_of",
    "thumb_url",
]
