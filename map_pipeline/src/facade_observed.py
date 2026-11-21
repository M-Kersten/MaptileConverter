"""Facade styles made from what the street photos measured.

The generator draws a wall from a :class:`facade.FacadeStyle`: a wall colour,
a window layout, a bay width. The era styles pick those by construction
year. An *observed* style takes them from the building's own photos (see
`facade_measure`) and keeps the era's choices only for what the photos could
not say: the brick photograph under the paint, the grain, the frame colour
when no frame stood out.

Buildings that measured alike share a style, so a terrace of identical
houses is one texture rather than twelve, and the number of textures stays
bounded however many buildings the photos reach. The clustering is greedy
and ordered, largest facade first, so the same area gives the same styles
every run.

Two things are per building rather than per style and are returned
separately: the bay width, which sets how wide one window column is drawn
on that building's walls, and the storey count, which sets how many rows of
windows its walls are divided into.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from .facade import STYLES, FacadeStyle, style_for_building
from .facade_measure import FacadeMeasurement, rgb_to_lab

LOG = logging.getLogger(__name__)

# Two buildings share a style when their walls are this close in colour
# (CIE76 delta E) and their windows this close in proportion.
SAME_COLOUR_DE = 7.0
SAME_WINDOW = 0.07
# How hard the photographed masonry under an observed style is pulled to
# the measured colour. The era styles use 0.6, which keeps more of the
# photograph's own; here the colour is the point.
OBSERVED_TINT = 0.9
# Archetypes that keep their own composition whatever the photos say: a
# church wall is not a stack of storeys. Matches src/buildings.py.
ARCH_INDUSTRIAL, ARCH_MONUMENTAL = 4, 5


@dataclass
class ObservedPlan:
    """What the measurements change, ready to apply."""

    styles: list[FacadeStyle]
    # Building index -> index into `styles`: the look of its largest measured
    # wall, worn by its walls nobody photographed.
    look_of: dict[int, int]
    # Face key -> index into `styles`, for each wall measured on its own.
    face_look: dict[tuple, int]
    # Face key -> metres of wall one texture tile covers across.
    face_tile_m: dict[tuple, float]
    # Building index -> storeys, where the photos counted them.
    floors: dict[int, int]

    @property
    def tile_width_m(self) -> dict[tuple, float]:
        return self.face_tile_m

    def summary(self) -> dict:
        return {
            "observed_styles": len(self.styles),
            "buildings_with_observed_style": len(self.look_of),
            "walls_with_observed_style": len(self.face_look),
            "walls_with_measured_bays": len(self.face_tile_m),
            "buildings_with_counted_storeys": len(self.floors),
        }


def _style_from(wall, era: FacadeStyle, name: str) -> FacadeStyle:
    """A style from one measured wall, the era filling what it lacks."""
    windows = wall.bay_m > 0 and wall.window_width_frac > 0
    storeys = wall.window_height_frac > 0
    return replace(
        era,
        name=name,
        base_name=era.family,
        label=f"measured from street photos, on {era.label or era.name}",
        wall_rgb=tuple(int(v) for v in wall.wall_rgb),
        trim_rgb=tuple(int(v) for v in wall.trim_rgb) if wall.trim_rgb else era.trim_rgb,
        # The glass a photo sees is mostly the reflected street; darkened a
        # little it reads as glass under the model's own sky.
        glass_rgb=(
            tuple(int(v * 0.85) for v in wall.glass_rgb) if wall.glass_rgb else era.glass_rgb
        ),
        windows_across=1 if windows else era.windows_across,
        window_width_frac=wall.window_width_frac if windows else era.window_width_frac,
        window_height_frac=wall.window_height_frac if storeys else era.window_height_frac,
        sill_frac=wall.sill_frac if storeys else era.sill_frac,
        tile_width_m=wall.bay_m if windows else era.tile_width_m,
        wall_alternates=(),
        trim_alternates=(),
        tint=OBSERVED_TINT,
    )


def _signature(style: FacadeStyle) -> np.ndarray:
    lab = rgb_to_lab(np.array(style.wall_rgb, dtype=np.float64))
    return np.array([
        *lab,
        style.window_width_frac * (1 if style.windows_across == 1 else 0.5),
        style.window_height_frac,
        style.sill_frac,
        float(style.windows_across == 1),
    ])


def _alike(a: np.ndarray, b: np.ndarray) -> bool:
    return (
        float(np.linalg.norm(a[:3] - b[:3])) < SAME_COLOUR_DE
        and float(np.max(np.abs(a[3:6] - b[3:6]))) < SAME_WINDOW
        and a[6] == b[6]
    )


def plan_observed(
    measurements: dict[str, FacadeMeasurement],
    buildings,
    *,
    n_variants: int,
    max_styles: int = 24,
    face_keys: dict[int, tuple] | None = None,
) -> ObservedPlan:
    """Styles, and per-wall and per-building changes, from the measurements.

    `face_keys` maps a measured wall's face index to its face key; walls
    without one are left to their building's look.
    """
    face_keys = face_keys or {}
    index_of = {b.identifier: i for i, b in enumerate(buildings.buildings)}
    walls = []
    for building_id, measurement in measurements.items():
        index = index_of.get(building_id)
        if index is None:
            continue
        building = buildings.buildings[index]
        if building.archetype in (ARCH_INDUSTRIAL, ARCH_MONUMENTAL):
            continue
        for wall in measurement.walls:
            walls.append((wall.area_m2 * wall.coverage, index, wall))
    # Largest wall first, then by index and face, so the result does not
    # depend on dictionary order.
    walls.sort(key=lambda w: (-w[0], w[1], w[2].face))

    styles: list[FacadeStyle] = []
    signatures: list[np.ndarray] = []
    plan = ObservedPlan(styles=styles, look_of={}, face_look={}, face_tile_m={}, floors={})
    for building_id, measurement in measurements.items():
        index = index_of.get(building_id)
        if index is not None and measurement.storeys > 0:
            plan.floors[index] = int(measurement.storeys)

    for _, index, wall in walls:
        building = buildings.buildings[index]
        era = STYLES[style_for_building(building.height_m, building.build_year, max(1, n_variants))]
        key = face_keys.get(wall.face)
        windows = wall.bay_m > 0 and wall.window_width_frac > 0
        if windows and key is not None:
            # An observed style draws one window per tile. Without a colour
            # the wall keeps its era's style, which draws several, so the
            # tile spans that many bays.
            per_tile = 1 if wall.wall_rgb is not None else era.windows_across
            plan.face_tile_m[key] = float(wall.bay_m) * per_tile
        if wall.wall_rgb is None:
            continue
        style = _style_from(wall, era, f"observed_{len(styles):02d}")
        signature = _signature(style)
        match = next((k for k, other in enumerate(signatures) if _alike(signature, other)), None)
        if match is None and len(styles) < max_styles:
            styles.append(style)
            signatures.append(signature)
            match = len(styles) - 1
        elif match is None:
            # Out of textures: the nearest style there is.
            match = int(np.argmin([np.linalg.norm(signature[:3] - s[:3]) for s in signatures]))
        if key is not None:
            plan.face_look[key] = match
        # Walls come largest first, so the first one a building gets is the
        # look its unphotographed walls wear.
        plan.look_of.setdefault(index, match)
    return plan


def apply_floors(buildings, plan: ObservedPlan) -> int:
    """Set the counted storeys on the buildings. Returns how many changed."""
    changed = 0
    for index, floors in plan.floors.items():
        building = buildings.buildings[index]
        if building.floors != floors:
            building.floors = floors
            changed += 1
        building.storeys_known = True
    return changed


__all__ = [
    "ObservedPlan",
    "apply_floors",
    "plan_observed",
]
