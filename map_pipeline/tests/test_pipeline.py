"""Checks for the parts that are easy to get subtly wrong.

Run everything (network tests included):

    python tests/test_pipeline.py

Run only the offline tests:

    python tests/test_pipeline.py --offline
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.buildings import (  # noqa: E402
    _newell_normal,
    _plane_basis,
    decode_vertices,
    is_degenerate_surface,
    triangulate_surface,
)
from src.config import load_config  # noqa: E402
from src.elevation import _bilinear_on_grid, _fill_holes, _resample_to_grid  # noqa: E402
from src.facade import STYLES, render_facade_tile, style_for_height  # noqa: E402
from src.geo import (  # noqa: E402
    BBox,
    GeoContext,
    build_grid_coords,
    parse_bbox,
    validate_bbox,
    wgs84_bbox_to_rd,
)
from src.http_util import extract_service_exception, looks_like_xml  # noqa: E402
from src.geo import tile_edges  # noqa: E402

RUN_NETWORK = "--offline" not in sys.argv



def cross2d(a: np.ndarray, b: np.ndarray):
    """Z component of the cross product of 2-D vectors.

    NumPy 2.0 deprecated passing 2-D vectors to np.cross and will remove it,
    so the determinant is spelled out rather than relying on that path.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class TestGeo(unittest.TestCase):
    def test_bbox_rejects_inverted(self):
        with self.assertRaises(ValueError):
            BBox(10, 10, 0, 20)

    def test_local_frame_round_trips(self):
        bbox = BBox(136000, 455000, 137000, 456000)
        geo = GeoContext.from_bbox(bbox)
        self.assertEqual(geo.origin, (136500.0, 455500.0))

        local = geo.rd_to_local(136000.0, 456000.0)
        self.assertAlmostEqual(local[0], -500.0)
        self.assertAlmostEqual(local[1], 500.0)

        back = geo.local_to_rd(*local)
        self.assertAlmostEqual(back[0], 136000.0)
        self.assertAlmostEqual(back[1], 456000.0)

    def test_wgs84_bbox_converts_to_utrecht(self):
        # A box over Utrecht city centre in lon/lat.
        bbox = parse_bbox(
            {
                "crs": "EPSG:4326",
                "xmin": 5.108,
                "ymin": 52.086,
                "xmax": 5.123,
                "ymax": 52.096,
            }
        )
        # Utrecht sits near RD 136000, 455000.
        self.assertTrue(133000 < bbox.xmin < 139000, bbox)
        self.assertTrue(453000 < bbox.ymin < 458000, bbox)

    def test_wgs84_envelope_is_a_superset(self):
        """Edge sampling must not clip the box the way four corners can."""
        wgs = BBox(5.10, 52.08, 5.14, 52.11)
        rd = wgs84_bbox_to_rd(wgs)
        corners_only = wgs84_bbox_to_rd(wgs, edge_samples=1)
        self.assertLessEqual(rd.xmin, corners_only.xmin + 1e-9)
        self.assertGreaterEqual(rd.xmax, corners_only.xmax - 1e-9)

    def test_lonlat_without_crs_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_bbox(BBox(5.10, 52.08, 5.14, 52.11))

    def test_non_square_bbox_warns(self):
        warnings = validate_bbox(BBox(136000, 455000, 137000, 455400))
        self.assertTrue(any("not square" in w for w in warnings), warnings)

    def test_grid_spans_bbox_inclusively(self):
        bbox = BBox(0, 0, 1000, 1000)
        xs, ys = build_grid_coords(bbox, 257)
        self.assertEqual(xs[0], 0.0)
        self.assertEqual(xs[-1], 1000.0)
        self.assertAlmostEqual(float(xs[1] - xs[0]), 1000.0 / 256)


class TestCityJSON(unittest.TestCase):
    def test_transform_is_applied(self):
        """Quantised vertices must come back as real RD coordinates."""
        transform = {"scale": [0.001, 0.001, 0.001], "translate": [136000.0, 455000.0, 0.0]}
        decoded = decode_vertices([[1000, 2000, 3000]], transform)
        np.testing.assert_allclose(decoded[0], [136001.0, 455002.0, 3.0])

    def test_newell_normal_points_up_for_ccw_ring(self):
        ring = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        np.testing.assert_allclose(_newell_normal(ring), [0, 0, 1], atol=1e-9)

    def test_plane_basis_is_right_handed(self):
        for normal in (
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81]),
        ):
            u, v = _plane_basis(normal)
            np.testing.assert_allclose(np.cross(u, v), normal, atol=1e-9)

    def test_triangulates_square(self):
        ring = np.array([[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float)
        tris = triangulate_surface([ring])
        self.assertEqual(len(tris), 2)
        area = sum(
            0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) for t in tris
        )
        self.assertAlmostEqual(area, 100.0)

    def test_triangulates_ring_with_hole(self):
        outer = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)
        hole = np.array([[3, 3, 0], [3, 7, 0], [7, 7, 0], [7, 3, 0]], dtype=float)
        tris = triangulate_surface([outer, hole])
        area = sum(
            0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) for t in tris
        )
        # 100 for the square minus 16 for the hole.
        self.assertAlmostEqual(area, 84.0, places=6)

    def test_triangulates_vertical_wall(self):
        """Walls are vertical, which is the degenerate case for XY projection."""
        wall = np.array(
            [[0, 0, 0], [0, 0, 10], [10, 0, 10], [10, 0, 0]], dtype=float
        )
        tris = triangulate_surface([wall])
        area = sum(
            0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) for t in tris
        )
        self.assertAlmostEqual(area, 100.0)

    def test_winding_follows_surface_normal(self):
        ring = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)
        expected = _newell_normal(ring)
        for tri in triangulate_surface([ring]):
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            self.assertGreater(float(normal @ expected), 0.0)

    def test_non_planar_surface_still_triangulates(self):
        """LoD2.2 roof faces are not exactly planar."""
        ring = np.array(
            [[0, 0, 0], [10, 0, 0.4], [10, 10, 0], [0, 10, -0.3]], dtype=float
        )
        self.assertEqual(len(triangulate_surface([ring])), 2)

    def test_degenerate_surface_is_detected(self):
        sliver = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=float)
        self.assertTrue(is_degenerate_surface([sliver]))
        self.assertFalse(
            is_degenerate_surface(
                [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)]
            )
        )


class TestGroundFloorSplit(unittest.TestCase):
    """Cutting a wall at the first-floor line must not lose or move any surface."""

    @staticmethod
    def _area(triangles):
        if len(triangles) == 0:
            return 0.0
        return float(
            sum(
                0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))
                for t in triangles
            )
        )

    def test_area_is_conserved(self):
        from src.buildings import split_walls_at_height

        rng = np.random.default_rng(11)
        walls = rng.random((500, 3, 3)) * np.array([12.0, 12.0, 9.0])
        lower, upper = split_walls_at_height(walls, np.full(len(walls), 3.6))
        self.assertAlmostEqual(
            self._area(walls), self._area(lower) + self._area(upper), places=6
        )

    def test_nothing_crosses_the_cut(self):
        from src.buildings import split_walls_at_height

        rng = np.random.default_rng(12)
        walls = rng.random((300, 3, 3)) * np.array([10.0, 10.0, 8.0])
        lower, upper = split_walls_at_height(walls, np.full(len(walls), 4.0))
        if len(lower):
            self.assertLessEqual(lower[:, :, 2].max(), 4.0 + 1e-9)
        if len(upper):
            self.assertGreaterEqual(upper[:, :, 2].min(), 4.0 - 1e-9)

    def test_wall_entirely_below_or_above_is_untouched(self):
        from src.buildings import split_walls_at_height

        below = np.array([[[0, 0, 0], [2, 0, 0], [0, 0, 1]]], dtype=float)
        lower, upper = split_walls_at_height(below, np.array([3.0]))
        self.assertEqual(len(lower), 1)
        self.assertEqual(len(upper), 0)

        above = np.array([[[0, 0, 5], [2, 0, 5], [0, 0, 7]]], dtype=float)
        lower, upper = split_walls_at_height(above, np.array([3.0]))
        self.assertEqual(len(lower), 0)
        self.assertEqual(len(upper), 1)

    def test_each_building_is_cut_at_its_own_height(self):
        """The cut follows the terrain, so it cannot be one global plane."""
        from src.buildings import split_walls_at_height

        walls = np.array(
            [
                [[0, 0, 0], [2, 0, 0], [0, 0, 10]],
                [[0, 0, 0], [2, 0, 0], [0, 0, 10]],
            ],
            dtype=float,
        )
        lower, _ = split_walls_at_height(walls, np.array([2.0, 8.0]))
        self.assertLess(self._area(lower[:1]), self._area(lower[1:]))


class TestArchetypes(unittest.TestCase):
    """What a building is decides its facade, where its era cannot."""

    @staticmethod
    def _building(**kwargs):
        from src.buildings import Building

        defaults = dict(
            identifier="NL.IMBAG.Pand.0001",
            wall_tris=np.zeros((0, 3, 3)),
            roof_tris=np.zeros((0, 3, 3)),
            ground_z_nap=0.0,
            roof_max_nap=10.0,
            wall_top_nap=10.0,
            floors=3,
        )
        defaults.update(kwargs)
        return Building(**defaults)

    def test_a_church_is_monumental(self):
        """The Dom: 104 m, built 1382, and no storey count in the BAG."""
        from src.buildings import ARCH_MONUMENTAL, classify_archetype

        dom = self._building(
            roof_max_nap=104.3,
            wall_top_nap=104.3,
            build_year=1382,
            storeys_known=False,
            footprint_m2=1100.0,
        )
        self.assertEqual(classify_archetype(dom), ARCH_MONUMENTAL)

    def test_a_modern_tower_is_not_monumental(self):
        """Height alone must not make something historic."""
        from src.buildings import ARCH_MONUMENTAL, classify_archetype

        tower = self._building(
            roof_max_nap=90.0,
            wall_top_nap=90.0,
            build_year=2017,
            storeys_known=False,
            footprint_m2=900.0,
        )
        self.assertNotEqual(classify_archetype(tower), ARCH_MONUMENTAL)

    def test_a_wide_low_building_is_industrial(self):
        from src.buildings import ARCH_INDUSTRIAL, classify_archetype

        shed = self._building(
            roof_max_nap=8.0,
            wall_top_nap=8.0,
            build_year=1995,
            footprint_m2=4200.0,
        )
        self.assertEqual(classify_archetype(shed), ARCH_INDUSTRIAL)

    def test_a_small_home_is_a_house(self):
        from src.buildings import ARCH_HOUSE, classify_archetype

        house = self._building(
            roof_max_nap=9.0,
            wall_top_nap=7.0,
            build_year=1912,
            function=0,
            footprint_m2=78.0,
        )
        self.assertEqual(classify_archetype(house), ARCH_HOUSE)

    def test_a_large_home_is_an_apartment_block(self):
        from src.buildings import ARCH_APARTMENT, classify_archetype

        block = self._building(
            roof_max_nap=28.0,
            wall_top_nap=28.0,
            build_year=1975,
            function=0,
            footprint_m2=900.0,
        )
        self.assertEqual(classify_archetype(block), ARCH_APARTMENT)

    def test_the_special_archetypes_get_their_own_style_slots(self):
        from src.facade import style_for_archetype, wall_styles

        self.assertIsNone(style_for_archetype(0, 5))  # house: era decides
        self.assertEqual(style_for_archetype(5, 5), 5)  # monumental
        self.assertEqual(style_for_archetype(4, 5), 6)  # industrial

        styles = wall_styles(5)
        self.assertEqual(len(styles), 7)
        self.assertEqual(styles[5].name, "monumental")
        self.assertEqual(styles[6].name, "industrial")

    def test_slots_stay_put_when_the_era_count_changes(self):
        """Blender addresses these by index, so they cannot shift about."""
        from src.facade import style_for_archetype, wall_styles

        for variants in (1, 3, 5):
            styles = wall_styles(variants)
            self.assertEqual(
                styles[style_for_archetype(5, variants)].name, "monumental"
            )
            self.assertEqual(
                styles[style_for_archetype(4, variants)].name, "industrial"
            )

    def test_a_church_keeps_its_wall_whole(self):
        """A cathedral has no shopfront to split off."""
        from src.buildings import (
            ARCH_HOUSE,
            ARCH_MONUMENTAL,
            BuildingSet,
            apply_ground_floor_split,
        )

        wall = np.array([[[0, 0, 0], [6, 0, 0], [0, 0, 20]]], dtype=float)
        church = self._building(
            wall_tris=wall.copy(), roof_max_nap=20.0, wall_top_nap=20.0
        )
        church.archetype = ARCH_MONUMENTAL
        house = self._building(
            wall_tris=wall.copy(), roof_max_nap=20.0, wall_top_nap=20.0
        )
        house.archetype = ARCH_HOUSE

        apply_ground_floor_split(BuildingSet([church, house], lod="2.2"), 3.6)

        self.assertEqual(len(church.ground_wall_tris), 0)
        self.assertEqual(len(church.wall_tris), 1)
        self.assertGreater(len(house.ground_wall_tris), 0)


class TestBayFitting(unittest.TestCase):
    """Every wall face must end on a tile edge, not through a window."""

    @staticmethod
    def _wall(x0, x1, height=9.0, y=0.0):
        """Two triangles making a rectangular wall facing -Y."""
        a, b = [x0, y, 0.0], [x1, y, 0.0]
        c, d = [x1, y, height], [x0, y, height]
        return np.array([[a, b, c], [a, c, d]], dtype=float)

    def test_a_house_front_gets_a_whole_number_of_bays(self):
        from src.facade_uv import fit_wall_u

        # 5.4 m front on a 4 m tile: fixed tiling would cut a window in half.
        walls = self._wall(0.0, 5.4)
        u = fit_wall_u(walls, np.zeros(len(walls), dtype=np.int32), np.array([4.0]))

        self.assertAlmostEqual(u.min(), 0.0, places=9)
        span = u.max() - u.min()
        self.assertAlmostEqual(span, round(span), places=9)
        self.assertGreaterEqual(span, 1.0)

    def test_every_face_of_a_real_range_of_widths_lands_on_an_edge(self):
        from src.facade_uv import fit_wall_u

        rng = np.random.default_rng(4)
        widths = rng.uniform(3.0, 40.0, 60)
        walls, owners = [], []
        for index, width in enumerate(widths):
            # Spread them out so each is its own face.
            walls.append(self._wall(0.0, width, y=index * 20.0))
            owners.append(np.full(2, index, dtype=np.int32))

        u = fit_wall_u(
            np.concatenate(walls), np.concatenate(owners), np.full(len(widths), 4.0)
        )
        per_face = u.reshape(len(widths), -1)
        for index, row in enumerate(per_face):
            span = row.max() - row.min()
            self.assertAlmostEqual(
                span, round(span), places=8, msg=f"wall {widths[index]:.2f} m"
            )

    def test_bays_stay_close_to_the_requested_width(self):
        from src.facade_uv import fit_wall_u

        for width in (4.4, 7.9, 12.3, 25.1):
            walls = self._wall(0.0, width)
            u = fit_wall_u(walls, np.zeros(2, dtype=np.int32), np.array([4.0]))
            bays = u.max() - u.min()
            self.assertAlmostEqual(width / bays, 4.0, delta=1.0)

    def test_a_sliver_shows_wall_rather_than_a_squashed_window(self):
        """A 0.4 m return must not have a whole window stretched over it."""
        from src.facade_uv import fit_wall_u

        walls = self._wall(0.0, 0.4)
        u = fit_wall_u(walls, np.zeros(2, dtype=np.int32), np.array([4.0]))

        # Centred on the tile seam, which is the pier between windows.
        self.assertLess(abs(u.min() + u.max()), 1e-9)
        self.assertLess(u.max() - u.min(), 0.2)

    def test_the_ground_storey_agrees_with_the_wall_above_it(self):
        from src.buildings import split_walls_at_height
        from src.facade_uv import fit_wall_u

        walls = self._wall(0.0, 13.7)
        lower, upper = split_walls_at_height(walls, np.full(len(walls), 3.6))

        combined = np.concatenate([upper, lower])
        owners = np.zeros(len(combined), dtype=np.int32)
        u = fit_wall_u(combined, owners, np.array([4.0]))

        upper_u = u[: len(upper) * 3]
        lower_u = u[len(upper) * 3 :]
        self.assertAlmostEqual(upper_u.min(), lower_u.min(), places=9)
        self.assertAlmostEqual(upper_u.max(), lower_u.max(), places=9)

    def test_the_result_does_not_depend_on_where_the_origin_is(self):
        """It is computed in RD and consumed in local metres."""
        from src.facade_uv import fit_wall_u

        walls = self._wall(0.0, 9.3)
        owners = np.zeros(2, dtype=np.int32)
        tile = np.array([4.0])

        shifted = walls + np.array([136500.0, 455500.0, 12.0])
        np.testing.assert_allclose(
            fit_wall_u(walls, owners, tile), fit_wall_u(shifted, owners, tile), atol=1e-9
        )

    def test_the_sides_of_a_building_are_separate_faces(self):
        from src.facade_uv import wall_face_ids

        front = self._wall(0.0, 8.0, y=0.0)
        back = self._wall(0.0, 8.0, y=6.0)
        side = np.array(
            [[[0, 0, 0], [0, 6, 0], [0, 6, 9]], [[0, 0, 0], [0, 6, 9], [0, 0, 9]]],
            dtype=float,
        )
        walls = np.concatenate([front, back, side])
        _, count = wall_face_ids(walls, np.zeros(len(walls), dtype=np.int32))
        self.assertEqual(count, 3)

    def test_one_wall_stays_one_face_however_it_was_triangulated(self):
        from src.facade_uv import wall_face_ids

        walls = np.concatenate(
            [self._wall(0.0, 4.0), self._wall(4.0, 9.0), self._wall(9.0, 11.0)]
        )
        _, count = wall_face_ids(walls, np.zeros(len(walls), dtype=np.int32))
        self.assertEqual(count, 1)

    def test_the_same_wall_on_two_buildings_stays_two_faces(self):
        """A terrace is a row of houses, not one continuous ribbon."""
        from src.facade_uv import wall_face_ids

        walls = np.concatenate([self._wall(0.0, 5.4), self._wall(5.4, 10.8)])
        owners = np.array([0, 0, 1, 1], dtype=np.int32)
        _, count = wall_face_ids(walls, owners)
        self.assertEqual(count, 2)

    def test_tile_width_follows_the_archetype(self):
        from src.facade_uv import tile_widths

        cfg = {
            "tile_width_m": 4.0,
            "monumental_tile_m": 7.0,
            "industrial_tile_m": 9.0,
        }
        widths = tile_widths(np.array([0, 4, 5]), cfg)
        np.testing.assert_allclose(widths, [4.0, 9.0, 7.0])


class TestEraStyles(unittest.TestCase):
    def test_build_year_picks_the_era(self):
        from src.facade import STYLES, style_for_building

        n = len(STYLES)
        self.assertEqual(style_for_building(12.0, 1890, n), 0)   # historic
        self.assertEqual(style_for_building(12.0, 1935, n), 1)   # interbellum
        self.assertEqual(style_for_building(12.0, 1960, n), 2)   # postwar
        self.assertEqual(style_for_building(12.0, 1985, n), 3)   # modern
        self.assertEqual(style_for_building(12.0, 2015, n), 4)   # contemporary

    def test_same_height_different_era_gives_different_styles(self):
        from src.facade import STYLES, style_for_building

        n = len(STYLES)
        self.assertNotEqual(
            style_for_building(20.0, 1890, n), style_for_building(20.0, 2015, n)
        )

    def test_missing_year_falls_back_to_height(self):
        from src.facade import STYLES, style_for_building

        n = len(STYLES)
        self.assertEqual(style_for_building(4.0, None, n), 0)
        self.assertGreater(style_for_building(40.0, None, n), 0)

    def test_single_variant_collapses_to_one_style(self):
        from src.facade import style_for_building

        self.assertEqual(style_for_building(40.0, 2015, 1), 0)


class TestTrees(unittest.TestCase):
    def test_superseded_versions_are_dropped(self):
        """The BGT returns every past version; only the current one is a tree."""
        from src.trees import TreeSet

        features = [
            {"properties": {"lokaal_id": "a", "eind_registratie": "2022-01-01"},
             "geometry": {"type": "Point", "coordinates": [136100, 455100]}},
            {"properties": {"lokaal_id": "a", "eind_registratie": None},
             "geometry": {"type": "Point", "coordinates": [136100, 455100]}},
            {"properties": {"lokaal_id": "b", "eind_registratie": None},
             "geometry": {"type": "Point", "coordinates": [136200, 455200]}},
        ]
        result = TreeSet()
        kept = []
        seen = set()
        for feature in features:
            properties = feature["properties"]
            if properties.get("eind_registratie"):
                result.superseded_dropped += 1
                continue
            if properties["lokaal_id"] in seen:
                result.superseded_dropped += 1
                continue
            seen.add(properties["lokaal_id"])
            kept.append(feature)

        self.assertEqual(len(kept), 2)
        self.assertEqual(result.superseded_dropped, 1)

    def test_building_mask_covers_roofs(self):
        from src.buildings import Building, BuildingSet
        from src.trees import _rasterize_buildings

        roof = np.array([[[10, 10, 5], [30, 10, 5], [30, 30, 5]]], dtype=float)
        building = Building(
            identifier="x",
            wall_tris=np.zeros((0, 3, 3)),
            roof_tris=roof,
            ground_z_nap=0.0,
            roof_max_nap=5.0,
            floors=1,
        )
        mask = _rasterize_buildings(
            BuildingSet(buildings=[building]), (0.0, 0.0, 40.0, 40.0), (40, 40)
        )
        # Inside the triangle is masked, well outside it is not.
        self.assertTrue(mask[int(40 - 20), 25])
        self.assertFalse(mask[int(40 - 35), 5])

    def test_local_maximum_finds_a_crown_beside_the_point(self):
        """A BGT point marks the trunk, so a point sample misses the canopy."""
        from src.trees import _max_in_radius

        grid = np.zeros((40, 40))
        grid[20, 22] = 14.0  # crown top, two cells east of the trunk
        bounds = (0.0, 0.0, 40.0, 40.0)
        x = np.array([20.5])
        y = np.array([40 - 20.5])

        point_like = _max_in_radius(grid, bounds, x, y, 0.5)
        wider = _max_in_radius(grid, bounds, x, y, 3.0)
        self.assertEqual(float(point_like[0]), 0.0)
        self.assertEqual(float(wider[0]), 14.0)

    def test_registered_trees_may_stand_close_found_ones_may_not(self):
        """Hazeldonk's register has clumps planted 0.7 m apart, which is the
        register's business. A found tree that close to another is the
        detector drawing one twice -- including a pair the old check, which
        only compared neighbours in order of x, could not see."""
        from src.geo import BBox
        from src.validate import CheckReport, check_trees

        class Trees:
            def __init__(self, points):
                self.trees = [types_namespace(x=x, y=y, height_m=8.0, detected=found)
                              for x, y, found in points]
                self.superseded_dropped = 0

            def __len__(self):
                return len(self.trees)

        def verdict(points):
            report = CheckReport()
            check_trees(report, Trees(points), BBox(0.0, 0.0, 100.0, 100.0))
            return {c.name: c for c in report.checks}["no_two_trees_share_a_spot"]

        far = [(80.0, 80.0, True), (90.0, 10.0, True)]
        self.assertTrue(verdict([(10.0, 10.0, False), (10.5, 10.5, False), *far]).passed)
        self.assertFalse(verdict([(10.0, 10.0, False), (10.5, 10.5, True), *far]).passed)
        hidden = [(20.0, 20.0, True), (20.5, 70.0, False), (20.9, 20.3, True), *far]
        self.assertFalse(verdict(hidden).passed)
        self.assertIn("0.95 m", verdict(hidden).detail)


class TestTreeDetection(unittest.TestCase):
    """Finding the trees the BGT never registered.

    The register is a municipal asset list, not a survey of vegetation: over a
    square kilometre of Utrecht centre it holds 1489 trees where 45% of the
    canopy above 2.5 m is nowhere near one of them, and that is the good case.
    Colour cannot help -- Dutch national orthos are flown deliberately leaf-off
    so the ground stays visible, and the infrared ortho gives park trees an
    NDVI of +0.08 against -0.18 for a roof. Height is the signal.
    """

    BOUNDS = (0.0, 0.0, 100.0, 100.0)

    @staticmethod
    def _blank(rows=200, cols=200):
        return np.zeros((rows, cols))

    @classmethod
    def _put(cls, grid, x, y, height, radius=2.0, rough=0.0, seed=0):
        """Drop a dome of `height` at RD (x, y), optionally roughened."""
        left, bottom, right, top = cls.BOUNDS
        rows, cols = grid.shape
        cx = (x - left) / (right - left) * cols
        cy = (top - y) / (top - bottom) * rows
        rr, cc = np.mgrid[0:rows, 0:cols]
        cell = (right - left) / cols
        dist = np.hypot((cc - cx) * cell, (rr - cy) * cell)
        dome = np.where(dist < radius, height * np.sqrt(
            np.maximum(1 - (dist / radius) ** 2, 0)), 0.0)
        if rough:
            noise = np.random.default_rng(seed).normal(0, rough, grid.shape)
            dome = np.where(dome > 0, dome + noise, dome)
        return np.maximum(grid, dome)

    def test_a_crown_becomes_one_tree_not_a_hundred(self):
        """Every cell of a crown is above the height threshold. Without a peak
        search and a crown-sized reservation, each would be its own tree."""
        from src.trees import detect_trees

        # A big tree, so the crown it reserves (0.26 x 22 = 5.7 m) is clearly
        # wider than the three-metre floor. With a tree barely above the floor
        # the two rules agree and the fixture proves nothing.
        grid = self._put(self._blank(), 50, 50, 22.0, radius=7.0, rough=0.9)
        found, heights = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)),
            min_height_m=2.5, crown_radius_ratio=0.26, min_roughness_m=0.3,
        )
        self.assertEqual(len(found), 1, f"one crown became {len(found)} trees")
        self.assertAlmostEqual(found[0][0], 50.0, delta=2.0)
        self.assertAlmostEqual(found[0][1], 50.0, delta=2.0)
        self.assertGreater(heights[0], 18.0)

    def test_a_registered_tree_is_not_found_twice(self):
        """A crown's canopy peak sits nowhere near the surveyed trunk, so the
        register has to reserve its own crown, not a token radius -- a flat
        three metres left a quarter of the detections inside a tree that was
        already in the model."""
        from src.trees import detect_trees

        grid = self._put(self._blank(), 50, 50, 20.0, radius=5.0, rough=0.9)
        surveyed = np.array([[50.0, 50.0]])

        # A crown reservation the size of the crown keeps the detector out.
        covered, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=surveyed,
            registered_reach=np.array([0.26 * 20.0]),
            min_height_m=2.5, crown_radius_ratio=0.26, min_roughness_m=0.3,
        )
        self.assertEqual(len(covered), 0, "the surveyed tree was found again")

        # A token reservation does not, which is the bug this replaced: a flat
        # three metres left a quarter of the detections inside a tree the model
        # already had.
        token, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=surveyed,
            registered_reach=np.array([0.5]), min_spacing_m=0.5,
            min_height_m=2.5, crown_radius_ratio=0.26, min_roughness_m=0.3,
        )
        self.assertGreater(len(token), 0, "the fixture cannot show a difference")

    def test_a_smooth_tall_thing_is_not_a_tree(self):
        """What roughness is for. A crown is a mess of leaves at half-metre
        scale where a made thing is a surface: at a registered tree roughness
        runs to a median of 3.1 m against 0.67 m on a roof.

        It cleans up what the footprint mask misses rather than replacing it --
        measured over the Utrecht square, a roughness cut alone still keeps
        64% of roof area, because a roof *edge* is as rough as anything.
        """
        from src.trees import detect_trees

        # A rough crown, and a smooth flat-topped slab as tall as it is.
        grid = self._put(self._blank(), 25, 25, 9.0, radius=4.0, rough=0.9)
        grid[40:80, 120:160] = 9.0          # RD x 60..80, y 60..80

        found, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)),
            min_height_m=2.5, min_roughness_m=0.4, crown_radius_ratio=0.26,
        )
        self.assertGreater(len(found), 0, "the rough crown was missed too")
        inside = [(x, y) for x, y in found if 63 < x < 77 and 63 < y < 77]
        self.assertEqual(
            inside, [], f"{len(inside)} trees growing on a flat roof",
        )
        # The edges of that slab are as rough as any crown -- measured at 4.3
        # against 0.0 in the middle -- so they are still detected here. That is
        # the building mask's job, not this one's, and the point of saying so
        # is that raising the roughness cut will not fix a roof edge.
        self.assertTrue(
            any(x > 55 for x, _ in found),
            "the fixture no longer has a roof edge to leak",
        )

    def test_a_roof_and_its_eaves_are_both_excluded(self):
        """A roof overhangs its own footprint, and without clearance the eaves
        read as a line of trees along every terrace."""
        from src.trees import detect_trees

        grid = self._blank()
        grid[60:140, 60:140] = 10.0
        rng = np.random.default_rng(3)
        grid[60:140, 60:140] += rng.normal(0, 1.2, (80, 80))
        mask = np.zeros(grid.shape, dtype=bool)
        mask[62:138, 62:138] = True     # the footprint, inside the roofline

        loose, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)),
            building_mask=mask, eaves_clearance_m=0.0, clearance_per_metre=0.0,
            min_height_m=2.5, min_roughness_m=0.4, crown_radius_ratio=0.26,
        )
        clear, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)),
            building_mask=mask, eaves_clearance_m=2.0, clearance_per_metre=0.0,
            min_height_m=2.5, min_roughness_m=0.4, crown_radius_ratio=0.26,
        )
        self.assertGreater(len(loose), 0, "the fixture has no eaves to find")
        self.assertEqual(len(clear), 0, f"{len(clear)} trees along the eaves")

    def test_a_tall_thing_against_a_building_is_the_building(self):
        """Measured over the Utrecht square: 65% of detections above 25 m sat
        within 5 m of a building against 32% of those under 10 m, and the
        tallest were one cluster around a tower the footprints do not cover."""
        from src.trees import detect_trees

        grid = self._blank()
        mask = np.zeros(grid.shape, dtype=bool)
        mask[80:120, 80:120] = True
        grid[80:120, 80:120] = 20.0
        # A tall rough spire 4 m off the roof, and a short tree 4 m off it too.
        grid = self._put(grid, 62, 50, 26.0, radius=2.0, rough=1.2, seed=1)
        grid = self._put(grid, 50, 38, 6.0, radius=2.0, rough=0.9, seed=2)

        found, heights = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)),
            building_mask=mask, eaves_clearance_m=2.0,
            clearance_per_metre=0.15, max_height_m=30.0,
            min_height_m=2.5, min_roughness_m=0.4, crown_radius_ratio=0.26,
        )
        self.assertTrue(
            all(h < 20.0 for h in heights),
            f"kept a {max(heights, default=0):.0f} m spire beside the roof",
        )

    def test_water_is_not_a_forest(self):
        """Lidar does not reflect off water and the returns that do come back
        scatter over six metres, which any height threshold reads as canopy."""
        from src.trees import detect_trees

        grid = self._blank()
        rng = np.random.default_rng(7)
        grid[20:80, 20:80] = np.abs(rng.normal(0, 3.0, (60, 60)))
        mask = np.zeros(grid.shape, dtype=bool)
        mask[20:80, 20:80] = True

        noisy, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)),
            min_height_m=2.5, min_roughness_m=0.4, crown_radius_ratio=0.26,
        )
        masked, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)), water_mask=mask,
            min_height_m=2.5, min_roughness_m=0.4, crown_radius_ratio=0.26,
        )
        self.assertGreater(len(noisy), 10, "the fixture is not noisy enough")
        self.assertEqual(len(masked), 0, f"{len(masked)} trees on the canal")

    def test_the_canopy_model_is_measured_from_the_filled_ground(self):
        """The bare-earth grid arrives 51% measured over the Utrecht square and
        its holes are exactly under dense canopy, because dense canopy is what
        stops a ground return. Differencing the two rasters therefore leaves the
        model undefined in the middle of a wood, which is where it is wanted."""
        import inspect

        from src import trees as trees_module

        source = inspect.getsource(trees_module._load_ndsm)
        self.assertIn(
            "terrain.sample", source,
            "the canopy model is differenced against the raw download again",
        )
        self.assertNotIn(
            "dsm - dtm", source.replace(" ", " "),
            "the canopy model still subtracts the unfilled bare-earth grid",
        )

    def test_detected_trees_stay_separable(self):
        """The detector is good and not perfect, so a bad patch has to be
        deletable as a group rather than hunted tree by tree."""
        import tempfile

        from src.trees import Tree, TreeSet, save_trees

        result = TreeSet()
        for index in range(4):
            result.trees.append(
                Tree(x=float(index), y=0.0, ground_z_nap=0.0, height_m=8.0,
                     crown_radius_m=2.0, trunk_height_m=2.0, measured=True,
                     detected=index >= 2)
            )
        stats = result.stats()
        self.assertEqual(stats["registered"], 2)
        self.assertEqual(stats["detected"], 2)

        with tempfile.TemporaryDirectory() as folder:
            path = save_trees(result, Path(folder) / "trees.npz")
            data = np.load(path)
            self.assertIn("detected", data.files)
            self.assertEqual(list(data["detected"]), [False, False, True, True])

    def test_a_found_tree_outside_the_area_is_dropped(self):
        """The canopy model covers whatever raster the AHN service returned,
        which need not stop where the bbox does. A tree outside the terrain is
        a tree standing on nothing, and it stretches the model with it."""
        from src.trees import detect_trees

        # Two crowns: one inside a 60 m area, one well outside it.
        grid = self._put(self._blank(), 30, 30, 12.0, radius=4.0, rough=0.9)
        grid = self._put(grid, 85, 85, 12.0, radius=4.0, rough=0.9, seed=4)
        found, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)),
            min_height_m=2.5, crown_radius_ratio=0.26, min_roughness_m=0.3,
        )
        self.assertGreater(
            len(found), 1, "the detector works over the whole raster, as it must"
        )
        self.assertTrue(
            any(x > 60 or y > 60 for x, y in found),
            "the fixture has nothing outside the area to drop",
        )

        from src.geo import BBox

        clipped, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)),
            bbox=BBox(0.0, 0.0, 60.0, 60.0),
            min_height_m=2.5, crown_radius_ratio=0.26, min_roughness_m=0.3,
        )
        self.assertTrue(len(clipped), "the clip dropped everything")
        self.assertLess(len(clipped), len(found), "nothing was clipped")
        self.assertTrue(
            all(0 <= x <= 60 and 0 <= y <= 60 for x, y in clipped),
            "a tree survived the clip from outside the area",
        )

    def test_a_tree_does_not_stand_on_a_carriageway(self):
        """A tall rough thing on a road is a lorry, a bus or scaffolding far
        more often than a tree: 9.5% of everything found over a square
        kilometre of Gelderland was standing on one.

        Avenues exist and this costs the odd real one, which is a good trade at
        ten to one -- but only for what the detector found. Where the BGT has
        surveyed a trunk, the trunk is there.
        """
        from src.trees import detect_trees

        # A crown over the road, and another on the verge beside it.
        grid = self._put(self._blank(), 50, 50, 11.0, radius=3.0, rough=0.9)
        grid = self._put(grid, 50, 70, 11.0, radius=3.0, rough=0.9, seed=5)
        road = np.zeros(grid.shape, dtype=bool)
        road[np.ix_(range(80, 120), range(0, 200))] = True   # RD y 40..60

        loose, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)),
            min_height_m=2.5, crown_radius_ratio=0.26, min_roughness_m=0.3,
        )
        kept, _ = detect_trees(
            grid, self.BOUNDS, registered_xy=np.zeros((0, 2)), road_mask=road,
            min_height_m=2.5, crown_radius_ratio=0.26, min_roughness_m=0.3,
        )
        self.assertTrue(
            any(40 < y < 60 for _, y in loose),
            "the fixture has nothing on the road to drop",
        )
        self.assertEqual(
            [(x, y) for x, y in kept if 40 < y < 60], [],
            "a tree is still standing on the carriageway",
        )
        self.assertTrue(
            any(y > 60 for _, y in kept), "the verge tree went with it"
        )

    def test_the_surfaces_are_named_not_numbered(self):
        """"cycle_path" survives someone reading the config a year later and 7
        does not, and these are the names the road objects already carry."""
        from src.config import DEFAULTS
        from src.surfaces import CLASS_CYCLE, CLASS_FOOTPATH, CLASS_ROAD
        from src.trees import _road_classes

        codes = _road_classes(DEFAULTS["trees"]["detect_off_surfaces"])
        self.assertIn(CLASS_ROAD, codes)
        self.assertIn(CLASS_CYCLE, codes)
        # Footpaths too. They were left out to protect the tree in a pit on a
        # pedestrianised street -- but a surveyed tree is never filtered by
        # this at all, so the exclusion protected nothing and cost a great
        # deal: footpath is 305 of the 871 road parts over the Gelderland
        # square, and in a park every winding path through the lawns is one.
        self.assertIn(CLASS_FOOTPATH, codes)
        # A name that is not a surface class is dropped, not silently obeyed.
        self.assertEqual(_road_classes(["road_asphalt", "nonsense"]), {CLASS_ROAD})
        self.assertEqual(_road_classes(None), set())

    def test_build_trees_hands_the_detector_the_area(self):
        """The clip is only worth anything if the caller asks for it."""
        import inspect

        from src import trees as trees_module

        source = inspect.getsource(trees_module.build_trees)
        self.assertIn(
            "bbox=bbox", source,
            "build_trees no longer tells the detector what area it is in",
        )

    def test_the_settings_reach_the_config(self):
        from src.config import DEFAULTS

        self.assertTrue(DEFAULTS["trees"]["detect"])
        self.assertEqual(DEFAULTS["trees"]["detect_min_height_m"], 3.0)
        self.assertEqual(DEFAULTS["trees"]["detect_max_height_m"], 30.0)


class TestSurfaces(unittest.TestCase):
    @staticmethod
    def _area(ring):
        if len(ring) < 3:
            return 0.0
        x, y = ring[:, 0], ring[:, 1]
        return 0.5 * abs(
            float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        )

    def test_clipping_bounds_a_canal_that_leaves_the_area(self):
        from src.surfaces import clip_ring_to_bbox

        bbox = BBox(0, 0, 100, 100)
        # A canal running well past the area on both sides.
        ring = np.array(
            [[-500, 40], [600, 40], [600, 60], [-500, 60], [-500, 40]], dtype=float
        )
        clipped = clip_ring_to_bbox(ring, bbox)
        self.assertGreater(len(clipped), 3)
        self.assertGreaterEqual(clipped[:, 0].min(), -1e-9)
        self.assertLessEqual(clipped[:, 0].max(), 100 + 1e-9)
        self.assertAlmostEqual(self._area(clipped), 100 * 20, places=6)

    def test_clipping_leaves_an_interior_ring_alone(self):
        from src.surfaces import clip_ring_to_bbox

        bbox = BBox(0, 0, 100, 100)
        ring = np.array([[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]], float)
        self.assertAlmostEqual(
            self._area(clip_ring_to_bbox(ring, bbox)), self._area(ring), places=6
        )

    def test_polygon_entirely_outside_disappears(self):
        from src.surfaces import clip_ring_to_bbox

        ring = np.array([[200, 200], [300, 200], [300, 300], [200, 200]], float)
        self.assertEqual(len(clip_ring_to_bbox(ring, BBox(0, 0, 100, 100))), 0)

    def test_rasterizer_respects_holes(self):
        from src.bgt import rasterize_rings

        outer = np.array([[1, 1], [9, 1], [9, 9], [1, 9], [1, 1]], float)
        hole = np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]], float)
        mask = rasterize_rings([[outer, hole]], (0.0, 0.0, 10.0, 10.0), (10, 10))
        self.assertTrue(mask[8, 2])    # inside the outer ring
        self.assertFalse(mask[5, 5])   # inside the hole
        self.assertFalse(mask[0, 0])   # outside everything

    def test_polygon_rings_normalises_multipolygon(self):
        from src.bgt import polygon_rings

        square = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        single = polygon_rings({"type": "Polygon", "coordinates": [square]})
        multi = polygon_rings(
            {"type": "MultiPolygon", "coordinates": [[square], [square]]}
        )
        self.assertEqual(len(single), 1)
        self.assertEqual(len(multi), 2)
        self.assertEqual(polygon_rings({"type": "Point", "coordinates": [0, 0]}), [])


class TestBgtVersioning(unittest.TestCase):
    """Every BGT collection returns its full version history."""

    def test_only_the_open_registration_is_current(self):
        from src.bgt import is_current

        self.assertTrue(is_current({"lokaal_id": "a"}))
        self.assertTrue(is_current({"lokaal_id": "a", "eind_registratie": None}))
        self.assertFalse(
            is_current({"lokaal_id": "a", "eind_registratie": "2022-01-14"})
        )


class TestBuildingFunction(unittest.TestCase):
    def test_gebruiksdoel_maps_to_a_group(self):
        from src.usage import (
            FUNCTION_HOME,
            FUNCTION_OFFICE,
            FUNCTION_RETAIL,
            _group_from_gebruiksdoel,
        )

        self.assertEqual(_group_from_gebruiksdoel("woonfunctie"), FUNCTION_HOME)
        self.assertEqual(_group_from_gebruiksdoel("winkelfunctie"), FUNCTION_RETAIL)
        self.assertEqual(_group_from_gebruiksdoel("kantoorfunctie"), FUNCTION_OFFICE)

    def test_mixed_use_takes_the_street_facing_function(self):
        """A shop under flats is a shop at street level."""
        from src.usage import FUNCTION_RETAIL, _group_from_gebruiksdoel

        self.assertEqual(
            _group_from_gebruiksdoel("woonfunctie,winkelfunctie"), FUNCTION_RETAIL
        )

    def test_lookup_strips_the_3dbag_prefix(self):
        from src.usage import UsageSet

        usage = UsageSet(by_pand={"0344100000071252": 1})
        self.assertEqual(usage.group_for("NL.IMBAG.Pand.0344100000071252"), 1)
        self.assertIsNone(usage.group_for("NL.IMBAG.Pand.0000000000000000"))


class TestElevation(unittest.TestCase):
    def test_nodata_is_masked_not_averaged(self):
        """AHN nodata is ~3.4e38; averaging it in would ruin the whole tile."""
        values = np.array([[1.0, 3.4028235e38], [2.0, 3.0]])
        valid = values < 1e30
        grid, hit = _resample_to_grid(
            values, valid, (0.0, 0.0, 2.0, 2.0), np.array([0.5, 1.5]), np.array([0.5, 1.5])
        )
        self.assertTrue(np.all(np.nan_to_num(grid, nan=0.0) < 10.0))

    def test_holes_are_filled_from_neighbours(self):
        grid = np.full((9, 9), 5.0)
        grid[3:6, 3:6] = np.nan
        filled, missing = _fill_holes(grid)
        self.assertEqual(missing, 9)
        self.assertFalse(np.isnan(filled).any())
        np.testing.assert_allclose(filled, 5.0)

    def test_fill_raises_when_everything_is_nodata(self):
        with self.assertRaises(Exception):
            _fill_holes(np.full((4, 4), np.nan))

    def test_bilinear_sampling(self):
        grid = np.array([[0.0, 10.0], [20.0, 30.0]])
        xs = np.array([0.0, 100.0])
        ys = np.array([0.0, 100.0])
        self.assertAlmostEqual(float(_bilinear_on_grid(grid, xs, ys, 0.0, 0.0)), 0.0)
        self.assertAlmostEqual(float(_bilinear_on_grid(grid, xs, ys, 100.0, 0.0)), 10.0)
        self.assertAlmostEqual(float(_bilinear_on_grid(grid, xs, ys, 50.0, 50.0)), 15.0)
        # Outside the grid clamps rather than extrapolating.
        self.assertAlmostEqual(float(_bilinear_on_grid(grid, xs, ys, -50.0, 0.0)), 0.0)


class TestImagery(unittest.TestCase):
    def test_tiles_cover_every_pixel_exactly_once(self):
        for total, cap in ((4096, 2000), (8192, 2000), (1000, 2000), (2500, 2500)):
            spans = tile_edges(total, cap)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], total)
            for a, b in zip(spans, spans[1:]):
                self.assertEqual(a[1], b[0])
            self.assertTrue(all(end - start <= cap for start, end in spans))

    def test_service_exception_is_recognised(self):
        payload = (
            b'<?xml version="1.0"?>\n<ServiceExceptionReport version="1.3.0">'
            b"<ServiceException>image size too large</ServiceException>"
            b"</ServiceExceptionReport>"
        )
        self.assertTrue(looks_like_xml(payload))
        self.assertEqual(extract_service_exception(payload), "image size too large")

    def test_jpeg_is_not_mistaken_for_xml(self):
        self.assertFalse(looks_like_xml(b"\xff\xd8\xff\xe0\x00\x10JFIF"))


class TestServiceFailures(unittest.TestCase):
    """A dead service should be cheap to discover and obvious to read."""

    def test_connect_is_bounded_separately_from_read(self):
        """One shared timeout makes a dead host cost the whole read budget."""
        from src.http_util import CONNECT_TIMEOUT_S

        self.assertLessEqual(CONNECT_TIMEOUT_S, 15.0)
        # At the configured 180 s read budget, four attempts against a host
        # that is not listening must not run into the minutes.
        worst_case = 4 * CONNECT_TIMEOUT_S + (2 + 4 + 8)
        self.assertLess(worst_case, 60.0)

    def test_unreachable_is_told_apart_from_a_refusal(self):
        import requests

        from src.http_util import is_unreachable

        self.assertTrue(is_unreachable(requests.ConnectionError("boom")))
        self.assertTrue(is_unreachable(requests.ConnectTimeout("boom")))
        # A refusal reached a server, so it is a different kind of problem.
        self.assertFalse(is_unreachable(requests.HTTPError("400")))
        self.assertFalse(is_unreachable(ValueError("nope")))

    def test_error_text_is_readable(self):
        from src.http_util import short_error

        raw = (
            "HTTPSConnectionPool(host='api.3dbag.nl', port=443): Max retries "
            "exceeded with url: /collections/pand/items (Caused by "
            "ConnectTimeoutError(<HTTPSConnection object at 0x10a41c690>, "
            "'Connection to api.3dbag.nl timed out. (connect timeout=180.0)'))"
        )
        self.assertEqual(short_error(Exception(raw)), "connection timed out")
        self.assertEqual(
            short_error(Exception("HTTPSConnectionPool(...): Read timed out.")),
            "connected, but the server never replied",
        )
        self.assertEqual(
            short_error(Exception("Connection reset by peer")),
            "connection reset by the server",
        )
        # Anything unrecognised still comes through, just bounded.
        self.assertLess(len(short_error(Exception("x" * 500))), 165)

    def test_host_is_extracted_for_the_message(self):
        from src.http_util import host_of

        self.assertEqual(host_of("https://api.3dbag.nl/collections/pand"), "api.3dbag.nl")

    def test_preflight_reports_every_service_that_is_down(self):
        from src import http_util

        original = http_util.check_reachable
        try:
            http_util.check_reachable = lambda url, timeout=8.0: (
                ("3dbag" not in url), "connection timed out"
            )
            down = http_util.preflight(
                {
                    "buildings (3DBAG)": "https://api.3dbag.nl/collections/pand/items",
                    "AHN terrain (PDOK)": "https://service.pdok.nl/rws/ahn/wcs/v1_0",
                }
            )
        finally:
            http_util.check_reachable = original

        self.assertEqual(len(down), 1)
        self.assertIn("api.3dbag.nl", down[0])
        self.assertIn("buildings (3DBAG)", down[0])


class TestShadowCasting(unittest.TestCase):
    """Shadows cast from the geometry the model already has.

    A shadow detector working from pixels alone cannot tell a dark roof from
    a dark roof's shadow. This pipeline knows where the buildings and the
    trees are in three dimensions, so it computes the shadow from the thing
    that casts it, which is the whole reason the relighting can be trusted
    not to stain a dark surface.
    """

    @staticmethod
    def _one_box(height=10.0, size=100):
        surface = np.zeros((size, size))
        surface[size // 2 - 2 : size // 2 + 2, size // 2 - 2 : size // 2 + 2] = height
        return surface

    def _shadow_centre(self, mask):
        rows, cols = np.where(mask > 0.5)
        if not len(rows):
            return None
        half = mask.shape[0] / 2.0
        # East and north offsets in cells from the middle of the raster.
        return float(cols.mean() - half), float(half - rows.mean())

    def test_the_shadow_falls_away_from_the_sun(self):
        from src.relight import cast_shadow

        surface = self._one_box()
        for azimuth, east, north in (
            (180.0, 0.0, +1.0),   # sun due south, shadow to the north
            (0.0, 0.0, -1.0),     # sun due north, shadow to the south
            (90.0, -1.0, 0.0),    # sun due east, shadow to the west
            (270.0, +1.0, 0.0),   # sun due west, shadow to the east
            (225.0, +1.0, +1.0),  # sun south-west, shadow north-east
        ):
            centre = self._shadow_centre(cast_shadow(surface, 1.0, azimuth, 45.0))
            self.assertIsNotNone(centre, f"no shadow at azimuth {azimuth}")
            got_east, got_north = centre
            self.assertGreater(
                got_east * east + got_north * north, 1.0,
                f"sun at {azimuth} deg put the shadow {got_east:+.1f} east "
                f"{got_north:+.1f} north",
            )

    def test_a_lower_sun_casts_a_longer_shadow(self):
        """And by the right amount: height over the tangent of the elevation."""
        from src.relight import cast_shadow

        surface = self._one_box(height=10.0, size=160)
        for elevation, expect in ((45.0, 10.0), (26.565, 20.0), (18.435, 30.0)):
            mask = cast_shadow(surface, 1.0, 180.0, elevation)
            rows = np.where(mask.any(axis=1))[0]
            # The box occupies four rows; the shadow runs north of it.
            length = (160 // 2 - 2) - rows.min()
            self.assertAlmostEqual(
                length, expect, delta=2.0,
                msg=f"at {elevation} deg the shadow ran {length} m, not {expect}",
            )

    def test_a_taller_building_casts_a_longer_shadow(self):
        from src.relight import cast_shadow

        short = cast_shadow(self._one_box(5.0), 1.0, 180.0, 30.0)
        tall = cast_shadow(self._one_box(20.0), 1.0, 180.0, 30.0)
        self.assertGreater(tall.sum(), short.sum() * 2.0)

    def test_nothing_shadows_itself_round_the_edge(self):
        """A roll would let a tower on the north edge darken the south one."""
        from src.relight import cast_shadow

        surface = np.zeros((80, 80))
        surface[0:3, 38:42] = 30.0  # a tower on the north edge
        # Sun in the south, so the shadow would run north -- off the raster.
        mask = cast_shadow(surface, 1.0, 180.0, 20.0)
        self.assertEqual(
            float(mask[40:].sum()), 0.0,
            "the shadow wrapped round to the far side of the area",
        )

    def test_flat_ground_casts_nothing(self):
        from src.relight import cast_shadow

        self.assertEqual(
            float(cast_shadow(np.zeros((60, 60)), 1.0, 180.0, 30.0).sum()), 0.0
        )

    def test_a_slope_shadows_itself(self):
        """Heights are absolute NAP, not height above ground, so a hillside
        shades its own far slope. Arnhem is on a moraine and Bronbeek is on
        the side of it."""
        from src.relight import cast_shadow

        rows = np.arange(100)
        # A ridge: climbs to the middle, drops away north of it.
        surface = np.minimum(rows, 100 - rows)[:, None] * np.ones((1, 100)) * 0.8
        mask = cast_shadow(surface, 1.0, 180.0, 20.0)
        self.assertGreater(
            float(mask[:50].mean()), 0.3,
            "the north-facing slope is not in its own shadow",
        )
        self.assertLess(float(mask[55:].mean()), 0.05, "the sunlit side is shaded")

    def test_a_pitched_roof_is_burned_at_its_real_slope(self):
        """Averaging a triangle's three corner heights gets a flat roof right
        and a pitched one wrong, which is most Dutch houses."""
        from src.geo import BBox
        from src.relight import burn_triangles

        bbox = BBox(0.0, 0.0, 100.0, 100.0)
        raster = np.zeros((100, 100))
        # A slope climbing from 4 m at y=20 to 16 m at y=60.
        pitch = np.array([
            [[20.0, 20.0, 4.0], [80.0, 20.0, 4.0], [80.0, 60.0, 16.0]],
            [[20.0, 20.0, 4.0], [80.0, 60.0, 16.0], [20.0, 60.0, 16.0]],
        ])
        burn_triangles(pitch, bbox, raster)
        low = float(raster[78, 50])    # y ~ 22, near the eaves
        high = float(raster[42, 50])   # y ~ 58, near the ridge
        self.assertAlmostEqual(low, 4.0, delta=1.0)
        self.assertAlmostEqual(high, 16.0, delta=1.0)
        self.assertGreater(
            high - low, 8.0,
            "the roof was burned flat; a pitch of 12 m came out as "
            f"{high - low:.1f} m",
        )

    def test_only_the_triangle_is_burned_not_its_bounding_box(self):
        """A thin diagonal covers an eighth of the box it sits in. Filling the
        box instead would have every building shadow a square."""
        from src.geo import BBox
        from src.relight import burn_triangles

        bbox = BBox(0.0, 0.0, 100.0, 100.0)
        raster = np.zeros((100, 100))
        wedge = np.array([
            [[10.0, 10.0, 9.0], [90.0, 10.0, 9.0], [90.0, 20.0, 9.0]],
        ])
        burn_triangles(wedge, bbox, raster)
        covered = float((raster > 1.0).sum())
        # Half of an 80 x 10 m box.
        self.assertAlmostEqual(covered, 400.0, delta=80.0)
        self.assertLess(
            covered, 600.0,
            f"{covered:.0f} cells burned for a triangle of about 400",
        )
        # The corner outside the hypotenuse stays clear.
        self.assertAlmostEqual(float(raster[82, 15]), 0.0, places=6)

    def test_a_low_sun_over_regular_streets_is_still_found(self):
        """Shadows merge into a sheet there, which flattens the score surface
        into basins. Refining only the best coarse candidate settled 20
        degrees off this case."""
        from src.relight import cast_shadow, fit_sun

        rng = np.random.default_rng(3)
        surface = np.zeros((600, 600))
        for row in range(30, 570, 44):
            for col in range(30, 570, 70):
                surface[row : row + 20, col : col + 50] = rng.uniform(7, 14)
        mask = cast_shadow(surface, 0.5, 300.0, 14.0)
        luminance = np.full(surface.shape, 150.0)
        luminance[surface > 0.5] = 90.0
        luminance -= 55.0 * mask
        luminance += rng.normal(0, 8, luminance.shape)

        azimuth, elevation, _ = fit_sun(surface, luminance, 0.5)
        gap = min(abs(azimuth - 300.0), 360.0 - abs(azimuth - 300.0))
        self.assertLess(gap, 8.0, f"azimuth {azimuth} against 300")
        self.assertLess(abs(elevation - 14.0), 8.0)

    def test_a_mask_that_covers_everything_scores_nothing(self):
        """A mask with almost no variance cannot be scored -- the
        correlation's denominator goes to zero and the answer is noise. So a
        sun low enough to shadow the whole area must never win, however well
        'everything' happens to correlate with a dark photo.

        The bound is deliberately loose. A tighter one would be a belief about
        how much of a city is in shadow rather than a numerical guard, and
        would reject the right answer for a dense area under a low sun:
        Bronbeek alone comes to 44%.
        """
        from src.relight import cast_shadow, fit_sun

        rng = np.random.default_rng(17)
        # One tall wall along the south edge. At the bottom of the elevation
        # range its shadow runs the length of the area and covers all of it.
        surface = np.zeros((500, 500))
        surface[-10:, :] = 60.0
        saturating = float(cast_shadow(surface, 0.5, 180.0, 8.0).mean())
        self.assertGreater(
            saturating, 0.95, "the fixture cannot saturate; nothing to guard"
        )
        # A photo that is dark nearly everywhere, for reasons that are not
        # this shadow -- so 'shadow everything' correlates with it.
        luminance = np.full(surface.shape, 58.0) + rng.normal(0, 7, surface.shape)
        luminance[surface > 1.0] = 150.0

        azimuth, elevation, score = fit_sun(surface, luminance, 0.5)
        if score > 0.05:
            covered = float(cast_shadow(surface, 0.5, azimuth, elevation).mean())
            self.assertLess(
                covered, 0.95,
                f"the fit accepted a sun shadowing {covered:.0%} of the area",
            )

    def test_a_legitimately_shadowed_area_is_still_fitted(self):
        """The other side of that bound: a real answer may cover a lot of
        ground, and must not be thrown away for it."""
        from src.relight import cast_shadow, fit_sun

        rng = np.random.default_rng(23)
        surface = np.zeros((500, 500))
        for row in range(20, 480, 26):
            for col in range(20, 480, 30):
                surface[row : row + 14, col : col + 20] = rng.uniform(14, 22)
        mask = cast_shadow(surface, 0.5, 200.0, 16.0)
        self.assertGreater(
            float(mask.mean()), 0.55,
            "the fixture is not densely enough shadowed to test the bound",
        )
        luminance = np.full(surface.shape, 150.0)
        luminance[surface > 0.5] = 95.0
        luminance -= 58.0 * mask
        luminance += rng.normal(0, 7, luminance.shape)

        azimuth, _, score = fit_sun(surface, luminance, 0.5)
        self.assertGreater(score, 0.05, "a heavily shadowed area found no sun")
        gap = min(abs(azimuth - 200.0), 360.0 - abs(azimuth - 200.0))
        self.assertLess(gap, 10.0, f"azimuth {azimuth} against 200")

    def test_the_sun_is_recovered_from_the_shadows_it_cast(self):
        """No flight metadata: the WMS does not publish a capture time, so
        the sun is fitted against the area's own geometry."""
        from src.relight import cast_shadow, fit_sun

        rng = np.random.default_rng(3)
        surface = np.zeros((600, 600))
        for row in range(30, 570, 44):
            for col in range(30, 570, 70):
                surface[row : row + 20, col : col + 50] = rng.uniform(7, 14)

        for true_azimuth, true_elevation in ((225.0, 23.0), (150.0, 30.0)):
            mask = cast_shadow(surface, 0.5, true_azimuth, true_elevation)
            luminance = np.full(surface.shape, 150.0)
            luminance[surface > 0.5] = 90.0      # dark roofs, to mislead
            luminance -= 55.0 * mask
            luminance += rng.normal(0, 8, luminance.shape)

            azimuth, elevation, score = fit_sun(surface, luminance, 0.5)
            gap = min(
                abs(azimuth - true_azimuth), 360.0 - abs(azimuth - true_azimuth)
            )
            self.assertLess(gap, 6.0, f"azimuth {azimuth} against {true_azimuth}")
            self.assertLess(
                abs(elevation - true_elevation), 6.0,
                f"elevation {elevation} against {true_elevation}",
            )
            self.assertGreater(score, 0.05)

    def test_an_overcast_flight_fits_no_sun(self):
        """Nothing to remove, and a confident guess would stain the photo
        wherever the buildings happen to be."""
        from src.relight import fit_sun

        rng = np.random.default_rng(11)
        surface = np.zeros((400, 400))
        for row in range(30, 370, 44):
            for col in range(30, 370, 70):
                surface[row : row + 20, col : col + 50] = 10.0
        flat_light = np.full(surface.shape, 140.0) + rng.normal(0, 8, surface.shape)
        _, _, score = fit_sun(surface, flat_light, 0.5)
        self.assertLess(score, 0.05, f"found a sun in an overcast photo: r={score}")

    def test_a_roof_lands_at_its_own_height_over_its_own_footprint(self):
        from src.geo import BBox
        from src.relight import burn_triangles

        bbox = BBox(0.0, 0.0, 100.0, 100.0)
        raster = np.zeros((100, 100))
        roof = np.array([
            [[20.0, 20.0, 12.0], [40.0, 20.0, 12.0], [40.0, 40.0, 12.0]],
            [[20.0, 20.0, 12.0], [40.0, 40.0, 12.0], [20.0, 40.0, 12.0]],
        ])
        burn_triangles(roof, bbox, raster)
        # North is up: y=20..40 is rows 60..80, x=20..40 is columns 20..40.
        self.assertAlmostEqual(float(raster[65, 25]), 12.0, places=6)
        self.assertAlmostEqual(float(raster[10, 10]), 0.0, places=6)
        self.assertAlmostEqual(float(raster.max()), 12.0, places=6)
        self.assertAlmostEqual(
            float((raster > 6.0).sum()), 400.0, delta=45.0,
            msg="the 20 x 20 m roof did not cover about 400 cells",
        )

    def test_the_tallest_thing_over_a_cell_wins(self):
        """A roof and the wall under it project onto the same cells, and what
        casts a shadow is the top."""
        from src.geo import BBox
        from src.relight import burn_triangles

        bbox = BBox(0.0, 0.0, 100.0, 100.0)
        raster = np.zeros((100, 100))
        low = np.array([[[10.0, 10.0, 3.0], [60.0, 10.0, 3.0], [60.0, 60.0, 3.0]]])
        high = np.array([[[10.0, 10.0, 18.0], [60.0, 10.0, 18.0], [60.0, 60.0, 18.0]]])
        burn_triangles(np.concatenate([high, low]), bbox, raster)
        self.assertAlmostEqual(float(raster.max()), 18.0, places=6)
        burn_triangles(low, bbox, raster)
        self.assertAlmostEqual(
            float(raster.max()), 18.0, places=6,
            msg="a later, lower triangle overwrote a taller one",
        )

    def test_a_canopy_stands_at_the_top_of_its_crown(self):
        from src.geo import BBox
        from src.relight import burn_canopies

        class Trees:
            trees = [type("T", (), {
                "x": 50.0, "y": 50.0, "ground_z_nap": 4.0,
                "height_m": 16.0, "crown_radius_m": 6.0,
            })()]

        bbox = BBox(0.0, 0.0, 100.0, 100.0)
        raster = np.zeros((100, 100))
        burn_canopies(Trees(), bbox, raster)
        self.assertAlmostEqual(float(raster[50, 50]), 20.0, places=6)
        self.assertAlmostEqual(float(raster[10, 10]), 0.0, places=6)
        self.assertAlmostEqual(
            float((raster > 1.0).sum()), np.pi * 36.0, delta=20.0
        )

    def test_the_ground_is_the_floor_and_things_stand_on_it(self):
        from src.geo import BBox
        from src.relight import surface_raster

        class Sloped:
            def sample(self, x, y):
                return 2.0 + 0.05 * np.asarray(x)

        class Buildings:
            buildings = [type("B", (), {"roof_tris": np.array([
                [[40.0, 40.0, 30.0], [60.0, 40.0, 30.0], [60.0, 60.0, 30.0]],
            ])})()]

        bbox = BBox(0.0, 0.0, 100.0, 100.0)
        raster = surface_raster(bbox, 1.0, terrain=Sloped(), buildings=Buildings())
        self.assertAlmostEqual(float(raster[50, 5]), 2.0 + 0.05 * 5.5, delta=0.1)
        self.assertAlmostEqual(float(raster.max()), 30.0, places=6)


class TestDeShadowing(unittest.TestCase):
    """Taking the flown sun back out of the photo.

    A Dutch ortho is shot on one morning in late winter, so every roof and
    every square metre of ground carries a hard shadow at one sun angle. An
    engine lighting the model at any other hour then has two suns disagreeing.
    """

    PX = 480
    SPAN = 200.0
    AZIMUTH, ELEVATION = 225.0, 25.0
    # How much darker, per channel, a shadow is in these fixtures. Bluer than
    # it is dark: a shadow is lit by sky alone, so it loses more red.
    SHADE_RGB = (0.42, 0.48, 0.62)
    WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

    class _Flat:
        def sample(self, x, y):
            return np.zeros(len(np.atleast_1d(x)))

    @staticmethod
    def _buildings(boxes, height=10.0):
        tris = []
        for x0, y0, x1, y1 in boxes:
            tris.append([[x0, y0, height], [x1, y0, height], [x1, y1, height]])
            tris.append([[x0, y0, height], [x1, y1, height], [x0, y1, height]])
        roof = np.array(tris, dtype=float)
        return type("Buildings", (), {
            "buildings": [type("B", (), {"roof_tris": roof})()]
        })()

    def _scene(self, folder, *, overcast=False, dark_roofs=True,
               classes=None, inverted=None):
        """A photo of flat ground with boxes on it, shadows and all.

        Returns `(path, shadow mask, roof mask)` with the masks at photo
        resolution, so a test can say what should and should not have moved.
        """
        from PIL import Image

        from src.geo import BBox
        from src.relight import cast_shadow, surface_raster

        bbox = BBox(0.0, 0.0, self.SPAN, self.SPAN)
        boxes = [
            (x, y, x + 12.0, y + 8.0)
            for x in range(20, 180, 30) for y in range(20, 180, 30)
        ]
        buildings = self._buildings(boxes)

        surface = surface_raster(bbox, 0.5, terrain=self._Flat(), buildings=buildings)
        mask = cast_shadow(surface, 0.5, self.AZIMUTH, self.ELEVATION) > 0.5
        roof = surface_raster(bbox, 0.5, buildings=buildings) > 0.5

        def grow(small):
            rows = np.arange(self.PX) * small.shape[0] // self.PX
            cols = np.arange(self.PX) * small.shape[1] // self.PX
            return small[rows[:, None], cols[None, :]]

        shadow, roofs = grow(mask), grow(roof)
        rng = np.random.default_rng(5)
        image = np.zeros((self.PX, self.PX, 3))
        image[:, :] = (110.0, 130.0, 85.0)
        if classes is not None:
            # A second surface with its own albedo, so the per-class gains
            # have something to tell apart.
            image[grow(classes) == 1] = (150.0, 148.0, 145.0)
        if dark_roofs:
            # Darker than the shadowed grass below, on purpose: that is the
            # case a brightness threshold cannot get right.
            image[roofs] = (45.0, 40.0, 38.0)
        if not overcast:
            image = image * np.where(shadow[:, :, None], self.SHADE_RGB, 1.0)
        if inverted is not None:
            # Painted after the shadow, so this class comes out *brighter*
            # where it is shadowed than where it is lit: its measured gain is
            # below one.
            image[inverted & shadow] = (170.0, 168.0, 165.0)
            image[inverted & ~shadow] = (100.0, 98.0, 96.0)
            # And a minority of its shadowed pixels are genuinely dark, so
            # they are the ones a gain below one would actually darken.
            # Without them the photometric guard alone keeps the clamp from
            # ever being reached, and the clamp goes untested.
            speckle = np.zeros(inverted.shape, dtype=bool)
            speckle[::3] = True
            image[inverted & shadow & speckle] = (30.0, 29.0, 28.0)
        image += rng.normal(0, 4, image.shape)

        path = Path(folder) / "aerial.png"
        Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)
        return path, shadow, roofs, buildings, bbox

    def _luminance(self, path):
        from PIL import Image

        return np.asarray(Image.open(path), dtype=np.float64) @ self.WEIGHTS

    def test_shadowed_ground_is_brought_back_up_to_lit_ground(self):
        import tempfile

        from src.relight import remove_cast_shadows

        with tempfile.TemporaryDirectory() as tmp:
            path, shadow, roofs, buildings, bbox = self._scene(tmp)
            before = self._luminance(path)
            info = remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings
            )
            after = self._luminance(path)

        ground = ~roofs
        lit, shaded = ground & ~shadow, ground & shadow
        target = float(before[lit].mean())
        was = float(before[shaded].mean())
        now = float(after[shaded].mean())
        self.assertLess(was, target * 0.7, "the fixture has no shadow in it")
        self.assertGreater(
            now, target * 0.85,
            f"shadowed ground only came back to {now:.0f} against {target:.0f}",
        )
        self.assertGreater(info["fit_r"], 0.05)

    def test_sunlit_ground_is_left_alone(self):
        """It rewrites the photo, and the photo is the thing you see."""
        import tempfile

        from src.relight import remove_cast_shadows

        with tempfile.TemporaryDirectory() as tmp:
            path, shadow, roofs, buildings, bbox = self._scene(tmp)
            before = self._luminance(path)
            remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings
            )
            after = self._luminance(path)

        lit = (~roofs) & (~shadow)
        self.assertAlmostEqual(
            float(after[lit].mean()), float(before[lit].mean()), delta=1.0
        )

    def test_a_dark_roof_in_the_sun_is_not_mistaken_for_a_shadow(self):
        """The property that makes this safe. These roofs are *darker* than
        the shadowed grass, so anything thresholding on brightness alone
        brightens them; the geometry says no shadow was cast there, and the
        photo agrees they are not shadow-coloured."""
        import tempfile

        from src.relight import remove_cast_shadows

        with tempfile.TemporaryDirectory() as tmp:
            path, shadow, roofs, buildings, bbox = self._scene(tmp)
            before = self._luminance(path)
            roof_lit = roofs & (~shadow)
            ground_shaded = (~roofs) & shadow
            self.assertLess(
                float(before[roof_lit].mean()),
                float(before[ground_shaded].mean()),
                "the fixture's roofs are not darker than its shadows",
            )
            remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings
            )
            after = self._luminance(path)

        lifted = float(after[roof_lit].mean()) / float(before[roof_lit].mean())
        self.assertLess(
            lifted, 1.15,
            f"a lit dark roof was brightened {lifted:.2f}x as if it were shade",
        )

    def test_the_blue_cast_of_skylight_comes_out_too(self):
        """A shadow is lit by sky alone, so it loses more red than blue. Three
        separate channel gains undo that without naming a colour temperature;
        one luminance gain would leave every shadow blue."""
        import tempfile

        from PIL import Image

        from src.relight import remove_cast_shadows

        with tempfile.TemporaryDirectory() as tmp:
            path, shadow, roofs, buildings, bbox = self._scene(tmp)
            before = np.asarray(Image.open(path), dtype=np.float64)
            remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings
            )
            after = np.asarray(Image.open(path), dtype=np.float64)

        ground = ~roofs
        def blueness(image, selector):
            return float(
                (image[selector][:, 2] / np.maximum(image[selector][:, 0], 1)).mean()
            )

        lit = blueness(before, ground & ~shadow)
        was = blueness(before, ground & shadow)
        now = blueness(after, ground & shadow)
        self.assertGreater(was, lit * 1.2, "the fixture's shadows are not blue")
        self.assertLess(
            abs(now - lit), abs(was - lit) * 0.4,
            f"shadow blue/red went {was:.2f} -> {now:.2f}, against {lit:.2f} lit",
        )

    def test_each_surface_gets_its_own_gain(self):
        """How much darker a shadow is depends on what is under it. One gain
        for the whole photo is grass-in-shade measured against asphalt-in-sun.
        """
        import tempfile

        from src.relight import remove_cast_shadows

        # The right half is a brighter surface, as its own class.
        classes = np.zeros((400, 400), dtype=np.int32)
        classes[:, 200:] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path, shadow, roofs, buildings, bbox = self._scene(tmp, classes=classes)
            before = self._luminance(path)
            # `class_grid` is handed over the way the surfaces stage holds it,
            # with row 0 at the south edge.
            remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings,
                class_grid=np.flipud(classes),
            )
            after = self._luminance(path)

        half = self.PX // 2
        for name, columns in (("left", slice(0, half)), ("right", slice(half, None))):
            side = np.zeros((self.PX, self.PX), dtype=bool)
            side[:, columns] = True
            ground = side & ~roofs
            lit, shaded = ground & ~shadow, ground & shadow
            target = float(before[lit].mean())
            now = float(after[shaded].mean())
            self.assertGreater(
                now, target * 0.85,
                f"the {name} surface came back to {now:.0f} of {target:.0f}",
            )

    def test_an_overcast_photo_is_left_alone(self):
        """Nothing to remove. Guessing at a sun would stain the photo
        wherever the buildings happen to be."""
        import tempfile

        from src.relight import remove_cast_shadows

        with tempfile.TemporaryDirectory() as tmp:
            path, _, _, buildings, bbox = self._scene(tmp, overcast=True)
            before = self._luminance(path)
            info = remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings
            )
            after = self._luminance(path)
        self.assertEqual(info, {})
        np.testing.assert_array_equal(after, before)

    def test_it_can_be_switched_off(self):
        import tempfile

        from src.relight import remove_cast_shadows

        with tempfile.TemporaryDirectory() as tmp:
            path, _, _, buildings, bbox = self._scene(tmp)
            before = self._luminance(path)
            info = remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings, strength=0.0
            )
            after = self._luminance(path)
        self.assertEqual(info, {})
        np.testing.assert_array_equal(after, before)

    def test_with_nothing_standing_up_there_is_nothing_to_do(self):
        """And it does not go looking. Fitting a sun means a few hundred
        trial shadow casts; over open country where nothing stands up they
        all come back empty, which is half a minute spent proving it.

        The fit is replaced with a trap rather than merely checking the
        answer, because returning `{}` the slow way looks identical from
        outside.
        """
        import tempfile

        import src.relight as relight_module

        with tempfile.TemporaryDirectory() as tmp:
            path, _, _, _, bbox = self._scene(tmp)
            before = self._luminance(path)
            # Buildings present but flat: no geometry can cast anything.
            flat = self._buildings([(10.0, 10.0, 90.0, 90.0)], height=0.2)
            real_fit = relight_module.fit_sun
            relight_module.fit_sun = lambda *a, **k: self.fail(
                "fitted a sun over ground with nothing standing up on it"
            )
            try:
                info = relight_module.remove_cast_shadows(
                    path, bbox, terrain=self._Flat(), buildings=flat
                )
            finally:
                relight_module.fit_sun = real_fit
            after = self._luminance(path)
        self.assertEqual(info, {})
        np.testing.assert_array_equal(after, before)

    def test_no_geometry_means_no_guessing(self):
        import tempfile

        from src.geo import BBox
        from src.relight import remove_cast_shadows

        with tempfile.TemporaryDirectory() as tmp:
            path, _, _, _, _ = self._scene(tmp)
            before = self._luminance(path)
            info = remove_cast_shadows(
                path, BBox(0.0, 0.0, self.SPAN, self.SPAN), terrain=self._Flat()
            )
            after = self._luminance(path)
        self.assertEqual(info, {})
        np.testing.assert_array_equal(after, before)

    def test_the_gain_is_capped(self):
        """Deep shade holds no detail to recover, so past the cap the gain is
        amplifying sensor noise and JPEG blocks."""
        import tempfile

        from PIL import Image

        from src.relight import remove_cast_shadows

        with tempfile.TemporaryDirectory() as tmp:
            path, shadow, roofs, buildings, bbox = self._scene(tmp)
            before = np.asarray(Image.open(path), dtype=np.float64)
            info = remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings,
                max_gain=1.4,
            )
            after = np.asarray(Image.open(path), dtype=np.float64)

        self.assertLessEqual(info["mean_gain"], 1.4 + 1e-6)
        ratio = float(
            after[shadow & ~roofs].mean() / before[shadow & ~roofs].mean()
        )
        self.assertLessEqual(
            ratio, 1.45, f"shadow was lifted {ratio:.2f}x against a 1.4 cap"
        )

    def test_it_reports_what_it_did(self):
        import tempfile

        from src.relight import remove_cast_shadows

        with tempfile.TemporaryDirectory() as tmp:
            path, _, _, buildings, bbox = self._scene(tmp)
            info = remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings
            )
        gap = min(
            abs(info["sun_azimuth_deg"] - self.AZIMUTH),
            360.0 - abs(info["sun_azimuth_deg"] - self.AZIMUTH),
        )
        self.assertLess(gap, 8.0, f"reported azimuth {info['sun_azimuth_deg']}")
        self.assertLess(abs(info["sun_elevation_deg"] - self.ELEVATION), 8.0)
        self.assertGreater(info["shadow_fraction"], 0.05)
        self.assertGreater(
            info["shadow_luminance_after"], info["shadow_luminance_before"]
        )

    def test_half_strength_does_half_the_work(self):
        """0 and 1 are the easy cases; the knob has to mean something in
        between, and a pass that ignored it would still satisfy both ends."""
        import tempfile

        from src.relight import remove_cast_shadows

        lifts = {}
        for strength in (0.5, 1.0):
            with tempfile.TemporaryDirectory() as tmp:
                path, shadow, roofs, buildings, bbox = self._scene(tmp)
                before = self._luminance(path)
                remove_cast_shadows(
                    path, bbox, terrain=self._Flat(), buildings=buildings,
                    strength=strength,
                )
                after = self._luminance(path)
            shaded = shadow & ~roofs
            lifts[strength] = float(
                after[shaded].mean() - before[shaded].mean()
            )
        self.assertGreater(lifts[0.5], 5.0, "half strength did nothing")
        self.assertAlmostEqual(
            lifts[0.5] / lifts[1.0], 0.5, delta=0.12,
            msg=f"half strength lifted {lifts[0.5]:.1f} against "
                f"{lifts[1.0]:.1f} at full",
        )

    def test_the_mask_edge_is_feathered(self):
        """A hard mask boundary prints a seam into the photo that was not
        there before. Across a shadow edge the correction has to ramp."""
        import tempfile

        from src.relight import remove_cast_shadows

        with tempfile.TemporaryDirectory() as tmp:
            path, shadow, roofs, buildings, bbox = self._scene(tmp)
            before = self._luminance(path)
            remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings
            )
            after = self._luminance(path)

        lift = after - before
        # Along the boundary of the mask, how many pixels got a partial lift?
        edge = shadow ^ np.roll(shadow, 1, axis=0)
        touched = lift > 1.0
        full = float(np.percentile(lift[shadow & ~roofs], 75))
        partial = ((lift > 1.0) & (lift < full * 0.6))[edge & ~roofs]
        self.assertGreater(
            float(partial.mean()), 0.12,
            "the correction steps straight from nothing to full at the edge",
        )

    def test_a_shadow_is_never_made_darker(self):
        """The gain is clamped at the bottom as well as the top.

        A class whose statistics come out backwards -- too few samples, an odd
        surface, something repainted between the flight and the survey -- has
        a measured ratio below one. Applying it would *darken* ground the
        geometry says is already in shadow, which is the opposite of the job.
        """
        import tempfile

        from PIL import Image

        from src.relight import remove_cast_shadows

        # A class painted bright where it is shadowed and dark where it is lit.
        inverted = np.zeros((self.PX, self.PX), dtype=bool)
        inverted[self.PX // 3 : 2 * self.PX // 3] = True
        classes = np.zeros((400, 400), dtype=np.int32)
        classes[400 // 3 : 2 * 400 // 3] = 1

        with tempfile.TemporaryDirectory() as tmp:
            path, shadow, roofs, buildings, bbox = self._scene(
                tmp, inverted=inverted
            )
            before = np.asarray(Image.open(path), dtype=np.float64)
            weights = self.WEIGHTS
            backwards = inverted & shadow
            forwards = inverted & ~shadow
            self.assertGreater(
                float(np.median(before[backwards] @ weights)),
                float(np.median(before[forwards] @ weights)),
                "the fixture's odd class is not actually backwards",
            )
            # The dark minority inside it: these are what a gain below one
            # would darken, and what the clamp has to protect.
            darkest = before[backwards] @ weights
            self.assertGreater(
                float((darkest < 60.0).mean()), 0.2,
                "the fixture has no dark pixels inside the backwards class",
            )
            remove_cast_shadows(
                path, bbox, terrain=self._Flat(), buildings=buildings,
                class_grid=np.flipud(classes),
            )
            after = np.asarray(Image.open(path), dtype=np.float64)

        darkened = float((after < before - 3.0).mean())
        self.assertLess(
            darkened, 0.01,
            f"{darkened:.1%} of the photo was made darker by a de-shadow pass",
        )

    def test_terrain_alone_is_not_enough_to_start_relighting(self):
        """Sloped ground self-shades, so the fit would find something. It is
        still not a reason to rewrite the photo when the thing that casts most
        of the shadow -- the buildings, the trees -- was not fetched."""
        import tempfile

        from src.relight import remove_cast_shadows

        class Ridge:
            def sample(self, x, y):
                x = np.asarray(x, dtype=float)
                return 20.0 - np.abs(x - 100.0) * 0.3

        with tempfile.TemporaryDirectory() as tmp:
            path, _, _, _, bbox = self._scene(tmp)
            before = self._luminance(path)
            info = remove_cast_shadows(path, bbox, terrain=Ridge())
            after = self._luminance(path)
        self.assertEqual(info, {})
        np.testing.assert_array_equal(after, before)

    def test_the_pipeline_relights_in_the_right_order(self):
        """Three passes rewrite the aerial and each reads what the one before
        wrote. The sun comes out first, while the photo holds only what the
        camera saw; the grain goes onto evenly lit ground rather than being
        amplified where a shadow was; the canopy occlusion goes on last,
        because it is the one shading term that belongs in a ground texture.
        """
        import inspect

        import pipeline

        source = inspect.getsource(pipeline.run)
        order = [
            source.index("remove_cast_shadows("),
            source.index("blend_surface_detail("),
            source.index("bake_canopy_shade("),
        ]
        self.assertEqual(order, sorted(order), "the aerial passes are out of order")

    def test_the_metadata_says_the_photo_no_longer_carries_a_sun(self):
        """Anyone lighting this model needs to know the texture is not
        carrying its own sun any more -- and, when it still is, at what angle,
        so they can match it."""
        import inspect

        import pipeline

        source = inspect.getsource(pipeline.run)
        self.assertIn('"aerial_relighting": shadow_info', source)
        # Not reported as an empty block when nothing was done: that reads as
        # a feature that failed rather than one that stood down.
        self.assertIn("if shadow_info else {}", source)

    def test_the_knob_exists_and_reaches_the_pipeline(self):
        import inspect

        import pipeline
        from src.config import DEFAULTS

        self.assertEqual(DEFAULTS["aerial"]["deshadow"], 1.0)
        self.assertIn(
            'config.aerial.get("deshadow"', inspect.getsource(pipeline.run)
        )


class TestCanopyShade(unittest.TestCase):
    """The aerial is flown leaf-off and the trees are leafy.

    That is not an accident on either side: Dutch orthos are flown in winter so
    the ground is visible, which is what makes the photo usable as a ground
    texture and what the tree detector reads the canopy model against. The
    model then stands a leafy tree over pavement photographed in full sun, and
    the eye reads a tree that is not touching the ground.
    """

    PX = 240

    def _aerial(self, folder: Path, value=150):
        from PIL import Image

        path = folder / "aerial.png"
        Image.fromarray(
            np.full((self.PX, self.PX, 3), value, np.uint8), mode="RGB"
        ).save(path)
        return path

    @staticmethod
    def _trees(*spec):
        from src.trees import Tree, TreeSet

        return TreeSet(trees=[
            Tree(x=float(x), y=float(y), ground_z_nap=2.0, height_m=14.0,
                 crown_radius_m=float(r), trunk_height_m=5.0, measured=True)
            for x, y, r in spec
        ])

    def _bake(self, folder, trees, strength=0.35, bbox=None):
        from PIL import Image

        from src.geo import BBox
        from src.trees import bake_canopy_shade

        path = self._aerial(folder)
        before = np.asarray(Image.open(path), dtype=np.float64)
        changed = bake_canopy_shade(
            path, trees, bbox or BBox(0.0, 0.0, 100.0, 100.0), strength=strength
        )
        after = np.asarray(Image.open(path), dtype=np.float64)
        return changed, before, after

    def _at(self, image, x, y, bbox_size=100.0):
        """The pixel at an RD position, north-up."""
        col = int(x / bbox_size * self.PX)
        row = int((bbox_size - y) / bbox_size * self.PX)
        return float(image[row, col].mean())

    def test_the_ground_under_a_tree_gets_darker(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            changed, before, after = self._bake(
                Path(tmp), self._trees((50.0, 50.0, 6.0))
            )
        self.assertTrue(changed)
        self.assertLess(
            self._at(after, 50.0, 50.0), self._at(before, 50.0, 50.0) * 0.8,
            "the ground under the canopy is as bright as open pavement",
        )

    def test_it_lands_where_the_tree_is(self):
        """North is up in an image and up in RD, so the row axis runs the
        other way. Getting that wrong mirrors every shadow in the model about
        the middle of the area, which on a symmetric fixture looks fine."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _, before, after = self._bake(
                Path(tmp), self._trees((25.0, 75.0, 6.0))
            )
        self.assertLess(self._at(after, 25.0, 75.0), self._at(before, 25.0, 75.0) * 0.8)
        # The point this would land on if north were flipped, or x and y swapped.
        for x, y in ((25.0, 25.0), (75.0, 75.0), (75.0, 25.0)):
            self.assertAlmostEqual(
                self._at(after, x, y), self._at(before, x, y), delta=0.5,
                msg=f"shade appeared at ({x}, {y}) as well as at (25, 75)",
            )

    def test_ground_away_from_any_tree_is_untouched(self):
        """It darkens the photo, and a photo is the thing you see. A pass that
        dims the whole image is worse than no pass."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _, before, after = self._bake(
                Path(tmp), self._trees((50.0, 50.0, 4.0))
            )
        untouched = np.isclose(after, before).all(axis=2)
        self.assertGreater(
            untouched.mean(), 0.8,
            f"only {untouched.mean():.0%} of the photo was left alone",
        )
        for x, y in ((5.0, 5.0), (95.0, 95.0), (5.0, 95.0)):
            self.assertAlmostEqual(
                self._at(after, x, y), self._at(before, x, y), delta=0.5
            )

    def test_two_crowns_overlapping_are_one_canopy(self):
        """Summing coverage drives a copse to black: three trees over one spot
        would take 1 - 3 * strength, which at 0.35 is negative. One canopy has
        one sky above it, so the strongest cover wins."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _, before, alone = self._bake(
                Path(tmp), self._trees((50.0, 50.0, 8.0))
            )
            _, _, clumped = self._bake(
                Path(tmp),
                self._trees((50.0, 50.0, 8.0), (51.0, 50.0, 8.0),
                            (50.0, 51.0, 8.0), (49.0, 49.0, 8.0)),
            )
        self.assertAlmostEqual(
            self._at(clumped, 50.0, 50.0), self._at(alone, 50.0, 50.0), delta=0.5,
            msg="four overlapping crowns shaded harder than one",
        )
        self.assertGreater(
            clumped.min(), 0.0, "the shade drove pixels to black"
        )

    def test_the_strength_is_what_it_says(self):
        import tempfile

        for strength in (0.2, 0.35, 0.6):
            with tempfile.TemporaryDirectory() as tmp:
                _, before, after = self._bake(
                    Path(tmp), self._trees((50.0, 50.0, 10.0)), strength=strength
                )
            ratio = self._at(after, 50.0, 50.0) / self._at(before, 50.0, 50.0)
            self.assertAlmostEqual(
                ratio, 1.0 - strength, delta=0.02,
                msg=f"strength {strength} left {ratio:.3f} of the brightness",
            )

    def test_zero_leaves_the_photo_alone(self):
        """The knob has to reach all the way to off, or someone who does not
        want their aerial touched has no way out."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            changed, before, after = self._bake(
                Path(tmp), self._trees((50.0, 50.0, 8.0)), strength=0.0
            )
        self.assertFalse(changed)
        np.testing.assert_array_equal(after, before)

    def test_no_trees_is_not_an_error(self):
        import tempfile

        from src.trees import TreeSet

        with tempfile.TemporaryDirectory() as tmp:
            changed, before, after = self._bake(Path(tmp), TreeSet())
        self.assertFalse(changed)
        np.testing.assert_array_equal(after, before)

    def test_the_edge_is_soft(self):
        """A hard disk reads as paint on grass rather than as shade. Measured
        across the crown edge, the darkening has to fall off gradually."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _, before, after = self._bake(
                Path(tmp), self._trees((50.0, 50.0, 10.0))
            )
        ratios = [
            self._at(after, 50.0 + d, 50.0) / self._at(before, 50.0 + d, 50.0)
            for d in (0.0, 6.0, 10.0, 13.0, 16.0, 19.0)
        ]
        # Monotone from dark at the trunk to untouched outside the feather.
        for near, far in zip(ratios, ratios[1:]):
            self.assertLessEqual(near, far + 1e-6, f"not monotone: {ratios}")
        self.assertLess(ratios[0], 0.7, "the middle is not shaded")
        self.assertGreater(ratios[-1], 0.99, "the shade never stops")
        # And genuinely gradual rather than one step: at least three distinct
        # levels across the edge.
        self.assertGreaterEqual(
            len({round(r, 2) for r in ratios}), 4, f"a hard edge: {ratios}"
        )

    def test_a_tree_outside_the_area_does_not_wrap_round(self):
        """Trees are kept slightly outside the bbox on purpose -- a canopy
        overhangs the edge. Its shade has to clip, not appear on the far
        side."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _, before, after = self._bake(
                Path(tmp), self._trees((-3.0, 50.0, 5.0))
            )
        # It reaches in from the west edge.
        self.assertLess(self._at(after, 1.0, 50.0), self._at(before, 1.0, 50.0))
        # And not from the east.
        self.assertAlmostEqual(
            self._at(after, 98.0, 50.0), self._at(before, 98.0, 50.0), delta=0.5
        )

    def test_a_rectangular_area_is_not_assumed_square(self):
        """Metres per pixel differ between the axes when the bbox is not
        square, and using one for both both stretches every canopy and puts it
        in the wrong place. Both halves are checked: an aspect ratio on its own
        passes when the mark has been pushed off the edge and clipped."""
        import tempfile

        from src.geo import BBox

        with tempfile.TemporaryDirectory() as tmp:
            _, before, after = self._bake(
                Path(tmp), self._trees((50.0, 120.0, 8.0)),
                bbox=BBox(0.0, 0.0, 100.0, 200.0),
            )
        touched = ~np.isclose(after, before).all(axis=2)
        rows = np.where(touched.any(axis=1))[0]
        cols = np.where(touched.any(axis=0))[0]
        self.assertTrue(len(rows) and len(cols), "nothing was shaded at all")

        # Centred on the tree: x=50 of 100 is half way across, y=120 of 200 is
        # 40% down from the top.
        self.assertAlmostEqual(
            float(rows.mean()) / self.PX, 0.40, delta=0.03,
            msg="the mark is not centred on the tree's northing",
        )
        self.assertAlmostEqual(
            float(cols.mean()) / self.PX, 0.50, delta=0.03,
            msg="the mark is not centred on the tree's easting",
        )
        # 8 m is 8/100 of the width but 8/200 of the height, so in pixels the
        # mark is twice as wide as it is tall.
        self.assertAlmostEqual(
            len(cols) / len(rows), 2.0, delta=0.25,
            msg=f"{len(cols)} columns by {len(rows)} rows is not a circle on "
                f"the ground",
        )

    def test_the_shaded_patch_is_round_and_not_square(self):
        """Each tree is drawn into the box its shade can reach, and the
        rounding happens inside that box. A falloff that never reaches zero
        leaves the whole box faintly dark, which is a square halo round every
        tree -- and sampling along an axis never crosses one."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _, before, after = self._bake(
                Path(tmp), self._trees((50.0, 50.0, 10.0))
            )
        from src.trees import SHADE_FEATHER

        # Just inside the corner of the box, so well outside the circle.
        edge = 10.0 * (1.0 + SHADE_FEATHER) * 0.92
        for dx, dy in ((edge, edge), (-edge, edge), (edge, -edge), (-edge, -edge)):
            self.assertAlmostEqual(
                self._at(after, 50.0 + dx, 50.0 + dy),
                self._at(before, 50.0 + dx, 50.0 + dy), delta=0.5,
                msg=f"the corner of the patch box at ({dx:+.1f}, {dy:+.1f}) "
                    f"is shaded, so the mark is a square",
            )

    def test_the_falloff_eases_out_rather_than_stopping(self):
        """Smoothstep, not a linear ramp. A linear falloff has the same slope
        right up to where it stops, which leaves a visible ring at the outer
        edge; smoothstep flattens into it. Both are monotone and both reach
        zero, so only the shape tells them apart."""
        import tempfile

        from src.trees import SHADE_FEATHER

        crown = 10.0
        with tempfile.TemporaryDirectory() as tmp:
            _, before, after = self._bake(
                Path(tmp), self._trees((50.0, 50.0, crown))
            )

        def coverage(distance_m):
            ratio = (
                self._at(after, 50.0 + distance_m, 50.0)
                / self._at(before, 50.0 + distance_m, 50.0)
            )
            return (1.0 - ratio) / 0.35

        # A tenth of the way in from the outer edge. Linear would give 0.1
        # there; smoothstep gives 3 * 0.1^2 - 2 * 0.1^3 = 0.028.
        reach = crown * (1.0 + SHADE_FEATHER)
        near_edge = coverage(reach - 0.1 * SHADE_FEATHER * crown)
        self.assertLess(
            near_edge, 0.06,
            f"coverage {near_edge:.3f} just inside the edge is a linear ramp, "
            f"not an ease",
        )
        # And symmetrically flat at the inner end of the feather.
        inner = coverage(crown + 0.1 * SHADE_FEATHER * crown)
        self.assertGreater(
            inner, 0.94, f"coverage {inner:.3f} just inside the crown is not full"
        )

    def test_a_canopy_crossing_a_strip_boundary_is_whole(self):
        """The photo is shaded in horizontal strips to bound memory, and a
        tree is only drawn into the strips its shade reaches. Testing the
        reach with an image shorter than one strip exercises none of that."""
        import tempfile

        from PIL import Image

        from src.geo import BBox
        from src.trees import SHADE_STRIP_PX, bake_canopy_shade

        height = SHADE_STRIP_PX * 2 + 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aerial.png"
            Image.fromarray(
                np.full((height, 256, 3), 150, np.uint8), mode="RGB"
            ).save(path)
            before = np.asarray(Image.open(path), dtype=np.float64)
            # Metres per pixel on the long axis, so the tree sits exactly on
            # the first strip boundary.
            bbox = BBox(0.0, 0.0, 256.0, float(height))
            y = float(height) - SHADE_STRIP_PX
            bake_canopy_shade(
                path, self._trees((128.0, y, 40.0)), bbox, strength=0.35
            )
            after = np.asarray(Image.open(path), dtype=np.float64)

        column = after[:, 128].mean(axis=1) / np.maximum(
            before[:, 128].mean(axis=1), 1
        )
        above = column[SHADE_STRIP_PX - 30 : SHADE_STRIP_PX]
        below = column[SHADE_STRIP_PX : SHADE_STRIP_PX + 30]
        self.assertLess(above.max(), 0.9, "nothing above the boundary")
        self.assertLess(below.max(), 0.9, "nothing below the boundary")
        self.assertAlmostEqual(
            float(above.mean()), float(below[::-1].mean()), delta=0.02,
            msg="the canopy is cut off at the strip boundary",
        )

    def test_the_pipeline_shades_after_the_detail_pass(self):
        """Both write the aerial in place, so the order is load-bearing: the
        detail pass reads the photo it is given, and reading a shaded one
        would mix the grain into the shade."""
        import inspect

        import pipeline

        source = inspect.getsource(pipeline.run)
        self.assertIn("bake_canopy_shade(", source)
        self.assertLess(
            source.index("blend_surface_detail("),
            source.index("bake_canopy_shade("),
            "the canopy shade is baked before the surface detail",
        )

    def test_the_knob_exists_and_reaches_the_pipeline(self):
        import inspect

        import pipeline
        from src.config import DEFAULTS

        self.assertGreater(DEFAULTS["trees"]["ground_shade"], 0.0)
        self.assertLess(DEFAULTS["trees"]["ground_shade"], 1.0)
        self.assertIn(
            'config.trees.get("ground_shade"', inspect.getsource(pipeline.run)
        )


class TestFacadeLooks(unittest.TestCase):
    """A street is not one house drawn twelve times, and it was.

    Era decides the character of a facade and that part was right -- build year
    predicts a wall far better than height does. But era also decided the
    pixels, so every building of an age shared one texture and a terrace came
    out identical along its whole length. Each era now has several looks, and a
    building picks one from its own BAG identifier.
    """

    @staticmethod
    def _distinct(styles):
        """How many of these differ in something other than their name."""
        import dataclasses

        seen = set()
        for style in styles:
            fields = dataclasses.asdict(style)
            fields.pop("name")
            fields.pop("base_name")
            seen.add(tuple(sorted((k, str(v)) for k, v in fields.items())))
        return len(seen)

    def test_one_look_is_the_era_untouched(self):
        """`looks_per_era: 1` has to give exactly what it gave before any of
        this existed, or every area built so far changes under its owner."""
        from src.facade import STYLES, looks_for_style

        for style in STYLES:
            looks = looks_for_style(style, 1)
            self.assertEqual(len(looks), 1)
            self.assertIs(looks[0], style)

    def test_look_zero_is_always_the_era_itself(self):
        from src.facade import STYLES, looks_for_style

        for style in STYLES:
            self.assertIs(looks_for_style(style, 5)[0], style)

    def test_the_looks_of_one_era_differ(self):
        """Not just in name: a texture is generated per look, so two looks
        that agree on every field are two copies of one image."""
        from src.facade import STYLES, looks_for_style

        for style in STYLES:
            looks = looks_for_style(style, 3)
            self.assertEqual(
                self._distinct(looks), 3,
                f"the three {style.name} looks are not three looks",
            )

    def test_every_field_meant_to_vary_actually_varies(self):
        """Recolouring alone does not break up a terrace: the window rhythm is
        what the eye reads as "the same house again". Each field is checked on
        its own, because a tuple of three only has to differ in one of them --
        freezing any two of these passed a combined check.
        """
        from src.facade import STYLES, looks_for_style

        for field in ("wall_rgb", "trim_rgb", "windows_across",
                      "window_width_frac", "window_height_frac", "sill_frac",
                      "tile_width_m"):
            for style in STYLES:
                looks = looks_for_style(style, 4)
                seen = {
                    getattr(s, field) if not isinstance(getattr(s, field), float)
                    else round(getattr(s, field), 6)
                    for s in looks
                }
                self.assertGreater(
                    len(seen), 1,
                    f"all four {style.name} looks share one {field}: {seen}",
                )

    def test_a_look_is_not_one_change_away_from_its_neighbour(self):
        """Three cycles of different lengths -- walls 4, trims 2 or 3,
        geometry 3 -- so consecutive looks differ in more than one thing."""
        from src.facade import STYLES, looks_for_style

        for style in STYLES:
            looks = looks_for_style(style, 4)
            for a, b in zip(looks, looks[1:]):
                changed = sum((
                    a.wall_rgb != b.wall_rgb,
                    a.trim_rgb != b.trim_rgb,
                    (a.windows_across, a.tile_width_m)
                    != (b.windows_across, b.tile_width_m),
                ))
                self.assertGreaterEqual(
                    changed, 2,
                    f"{a.name} and {b.name} differ in only {changed} of "
                    f"wall, trim and layout",
                )

    def test_nothing_repeats_before_the_palette_runs_out(self):
        """Four wall colours against a three-entry geometry table gives
        twelve. A four-entry table would turn over in step with the colours
        and look 4 would be look 0 in a different window frame."""
        from src.facade import MAX_LOOKS_PER_ERA, STYLES, looks_for_style

        self.assertEqual(MAX_LOOKS_PER_ERA, 12)
        for style in STYLES:
            looks = looks_for_style(style, MAX_LOOKS_PER_ERA)
            self.assertEqual(
                self._distinct(looks), MAX_LOOKS_PER_ERA,
                f"{style.name} runs out of looks before {MAX_LOOKS_PER_ERA}",
            )

    def test_every_era_carries_enough_colours_to_fill_them(self):
        from src.facade import STYLES

        for style in STYLES:
            self.assertGreaterEqual(
                len(style.wall_alternates), 3,
                f"{style.name} has no alternative wall colours",
            )
            self.assertGreaterEqual(
                len(style.trim_alternates), 1,
                f"{style.name} has no alternative trim colours",
            )

    def test_the_slot_layout_holds(self):
        """Blender addresses a style by index, so the layout is a contract:
        era * looks + look, then the two specials. An extra look must not move
        an era, and an extra era must not move the specials."""
        from src.facade import style_for_archetype, wall_styles

        ARCH_INDUSTRIAL, ARCH_MONUMENTAL = 4, 5
        for variants in (1, 3, 5):
            for looks in (1, 2, 4):
                styles = wall_styles(variants, looks)
                self.assertEqual(len(styles), variants * looks + 2)
                monumental = style_for_archetype(
                    ARCH_MONUMENTAL, variants, looks
                )
                industrial = style_for_archetype(
                    ARCH_INDUSTRIAL, variants, looks
                )
                self.assertEqual(styles[monumental].name, "monumental")
                self.assertEqual(styles[industrial].name, "industrial")
                self.assertIsNone(style_for_archetype(0, variants, looks))

    def test_the_old_two_argument_calls_still_work(self):
        """Both functions are called with one argument elsewhere, and a
        default of one look is what keeps that meaning the same thing."""
        from src.facade import style_for_archetype, wall_styles

        self.assertEqual(len(wall_styles(5)), 7)
        self.assertEqual(style_for_archetype(5, 5), 5)
        self.assertEqual(style_for_archetype(4, 5), 6)

    def test_a_building_keeps_its_look(self):
        """From the identifier, not a counter, so re-running an area or
        fetching it in another order does not reshuffle the street."""
        from src.facade import look_for_building

        identifier = "NL.IMBAG.Pand.0362100012345678"
        first = look_for_building(identifier, 3)
        for _ in range(5):
            self.assertEqual(look_for_building(identifier, 3), first)

    def test_the_look_survives_a_fresh_process(self):
        """`hash()` is salted per process, so a style index built with it
        would differ between two runs of the same pipeline -- and the textures
        are generated once and referred to by index."""
        import subprocess

        code = (
            "import sys; sys.path.insert(0, '.');"
            "from src.facade import look_for_building;"
            "print([look_for_building(f'NL.IMBAG.Pand.{n}', 3) for n in range(12)])"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", code], cwd=REPO_ROOT,
                capture_output=True, text=True, check=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            ).stdout.strip()
            for seed in ("0", "1", "random")
        }
        self.assertEqual(
            len(runs), 1, f"the look depends on the hash seed: {runs}"
        )

    def test_a_terrace_does_not_all_get_the_same_look(self):
        """The whole point. Twelve neighbours of one era, twelve identifiers
        a digit apart."""
        from src.facade import look_for_building

        ids = [f"NL.IMBAG.Pand.03621000{n:06d}" for n in range(12)]
        looks = [look_for_building(i, 3) for i in ids]
        self.assertGreater(
            len(set(looks)), 1, f"a whole terrace took one look: {looks}"
        )

    def test_the_looks_are_used_about_evenly(self):
        """A hash that piles 90% of buildings onto one look has broken up
        nothing. Not a uniformity proof -- just a check that it is not
        lopsided."""
        from src.facade import look_for_building

        for n_looks in (2, 3, 5):
            counts = [0] * n_looks
            for n in range(3000):
                counts[look_for_building(f"NL.IMBAG.Pand.{n}", n_looks)] += 1
            share = min(counts) / 3000.0
            self.assertGreater(
                share, 0.5 / n_looks,
                f"with {n_looks} looks the rarest got {share:.1%}: {counts}",
            )

    def test_one_look_means_slot_zero(self):
        from src.facade import look_for_building

        for identifier in ("NL.IMBAG.Pand.1", "NL.IMBAG.Pand.2", None, ""):
            self.assertEqual(look_for_building(identifier, 1), 0)

    def test_a_building_with_no_identifier_still_gets_one(self):
        """3DBAG supplies an identifier for everything in practice, but a
        missing one must not put every nameless building on look 0."""
        from src.facade import look_for_building

        self.assertEqual(look_for_building(None, 3, fallback=4), 1)
        self.assertEqual(look_for_building("", 3, fallback=5), 2)

    def test_every_look_resolves_to_its_era_photograph(self):
        """The photographed wall surface is keyed by style, and a look is a
        new style name. Before `family` existed, `historic_1` found no
        photograph and fell back to drawn brick -- so one house in three had
        real grain and its neighbours did not."""
        from src.facade import GROUND_STYLES, wall_styles
        from src.textures import WALL_TEXTURES

        for style in list(wall_styles(5, 4)) + list(GROUND_STYLES):
            self.assertIn(
                style.family, WALL_TEXTURES,
                f"{style.name} resolves to {style.family!r}, which has no "
                f"photographed wall",
            )

    def test_the_texture_loader_resolves_a_look_to_its_era(self):
        """The map having the right keys is not the same as `wall_base` using
        them. It asks by style, and a look is a new style name -- checking the
        map alone left that unexercised."""
        import tempfile

        from src.facade import STYLES, looks_for_style
        from src.textures import WALL_TEXTURES, TextureUnavailable, wall_base

        reached = []

        def fake_fetch(slug, work_dir, *, timeout_s=60.0):
            reached.append(slug)
            raise TextureUnavailable("stopped before the network")

        import src.textures as textures_module

        real = textures_module.fetch_texture
        textures_module.fetch_texture = fake_fetch
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for look in looks_for_style(STYLES[0], 4):
                    reached.clear()
                    with self.assertRaises(TextureUnavailable):
                        wall_base(
                            look, 64, Path(tmp),
                            tile_width_m=look.tile_width_m, tile_height_m=3.0,
                        )
                    # It got as far as asking for the era's photograph, rather
                    # than giving up because "historic_1" is not in the map.
                    self.assertEqual(
                        reached, [WALL_TEXTURES["historic"]],
                        f"{look.name} asked for {reached} instead of its "
                        f"era's brick",
                    )
        finally:
            textures_module.fetch_texture = real

    def test_a_look_borrows_its_era_brick(self):
        from src.facade import STYLES, looks_for_style

        for style in STYLES:
            for look in looks_for_style(style, 4)[1:]:
                self.assertEqual(look.family, style.name)
                self.assertNotEqual(look.name, style.name)
        # An era is its own family, so nothing else has to special-case it.
        for style in STYLES:
            self.assertEqual(style.family, style.name)

    def test_the_textures_really_are_different_images(self):
        """Different parameters are not different pixels. Measured: looks of
        one era differ by 19-52 mean absolute levels, where two completely
        different eras differ by 50."""
        import tempfile

        from PIL import Image

        from src.config import DEFAULTS
        from src.facade import generate_facade_textures

        cfg = dict(
            DEFAULTS["facade"], texture_px=128, looks_per_era=3,
            photo_textures=False, normal_map=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_facade_textures(Path(tmp), facade_cfg=cfg)
            images = {
                colour.stem: np.asarray(Image.open(colour), dtype=np.float64)
                for colour, _ in paths
            }
            for era in ("historic", "postwar", "contemporary"):
                names = sorted(k for k in images if era in k)
                self.assertEqual(len(names), 3, f"{era} did not get 3 looks")
                base = next(k for k in names if k.endswith(era))
                for other in names:
                    if other == base:
                        continue
                    gap = float(np.abs(images[base] - images[other]).mean())
                    self.assertGreater(
                        gap, 8.0,
                        f"{other} is the same image as {base} (gap {gap:.1f})",
                    )

    def test_each_texture_gets_its_own_noise(self):
        """Two looks differ in colour and layout, but the grain is noise and
        with one seed they would share it, so a street would repeat a brick
        pattern it never repeats in life. The seed advances per slot."""
        from src.facade import STYLES, render_facade_layers

        first, _ = render_facade_layers(STYLES[0], 96, seed=100)
        second, _ = render_facade_layers(STYLES[0], 96, seed=101)
        gap = float(np.abs(first.astype(float) - second.astype(float)).mean())
        self.assertGreater(
            gap, 0.5, "the seed does not change the texture at all"
        )

        import inspect

        from src import facade

        source = inspect.getsource(facade.generate_facade_textures)
        self.assertIn("seed=seed + index", source)

    def test_the_texture_set_is_the_size_the_slots_expect(self):
        """One texture per slot, in slot order, or a building wears the wrong
        wall."""
        import tempfile

        from src.config import DEFAULTS
        from src.facade import generate_facade_textures, wall_styles

        for looks in (1, 3):
            cfg = dict(
                DEFAULTS["facade"], texture_px=64, looks_per_era=looks,
                photo_textures=False, normal_map=False,
            )
            with tempfile.TemporaryDirectory() as tmp:
                paths = generate_facade_textures(Path(tmp), facade_cfg=cfg)
            # The walls, plus one ground storey.
            self.assertEqual(len(paths), len(wall_styles(5, looks)) + 1)

    @staticmethod
    def _buildings(spec):
        """A BuildingSet from (identifier, year, archetype) triples."""
        from src.buildings import Building, BuildingSet

        tri = np.zeros((1, 3, 3))
        return BuildingSet(buildings=[
            Building(
                identifier=ident, wall_tris=tri, roof_tris=tri,
                ground_z_nap=0.0, roof_max_nap=9.0, floors=3,
                wall_top_nap=7.0, build_year=year, archetype=arch,
            )
            for ident, year, arch in spec
        ])

    def test_buildings_spread_one_era_across_its_looks(self):
        """`assign_facade_slots` is where the slot is actually chosen, and a
        test of `look_for_building` alone says nothing about whether it is
        used. This calls the real rule."""
        from src.buildings import assign_facade_slots

        terrace = self._buildings([
            (f"NL.IMBAG.Pand.03621000{n:06d}", 1890, 0) for n in range(300)
        ])
        slots, plan = assign_facade_slots(
            terrace, 5, {"looks_per_era": 12, "buildings_per_look": 80}
        )
        self.assertEqual(len(slots), 300)
        # 300 buildings at 80 apiece is four looks, all inside era 0.
        self.assertEqual(plan.per_era[0], 4)
        self.assertEqual(set(np.unique(slots).tolist()), {0, 1, 2, 3})

    def test_the_looks_go_where_the_buildings_are(self):
        """The whole point of allocating rather than spreading evenly. Over
        the real Bronbeek mix -- 73% of 1322 buildings interbellum -- an even
        allocation spends a dozen textures on sixteen pre-1920 houses and
        still leaves the interbellum era repeating."""
        from src.buildings import assign_facade_slots

        # The measured Bronbeek era counts.
        mix = []
        for era_year, count in ((1890, 16), (1933, 967), (1960, 240),
                                (1985, 27), (2010, 72)):
            mix += [
                (f"NL.IMBAG.Pand.{era_year}{n:07d}", era_year, 0)
                for n in range(count)
            ]
        slots, plan = assign_facade_slots(
            self._buildings(mix), 5,
            {"looks_per_era": 12, "buildings_per_look": 80},
        )
        self.assertEqual(
            plan.per_era, (1, 12, 3, 1, 1),
            "the looks were not allocated by how many buildings wear them",
        )
        # 20 textures, not the 62 a flat twelve would cost.
        self.assertEqual(plan.total_slots, 20)
        busiest = max(np.bincount(slots))
        self.assertLess(
            busiest, 120,
            f"the busiest wall is still worn by {busiest} buildings",
        )

    def test_a_sparse_era_is_not_given_looks_it_cannot_use(self):
        from src.buildings import assign_facade_slots

        # Three pre-1920 houses and nothing else.
        _, plan = assign_facade_slots(
            self._buildings([
                (f"NL.IMBAG.Pand.{n}", 1890, 0) for n in range(3)
            ]),
            5, {"looks_per_era": 12, "buildings_per_look": 80},
        )
        self.assertEqual(plan.per_era, (1, 1, 1, 1, 1))
        self.assertEqual(plan.total_slots, 7)

    def test_an_archetype_special_takes_no_era_look(self):
        """A church does not make the houses around it repeat less."""
        from src.buildings import assign_facade_slots

        ARCH_MONUMENTAL = 5
        mix = [(f"NL.IMBAG.Pand.h{n}", 1933, 0) for n in range(160)]
        mix += [(f"NL.IMBAG.Pand.church{n}", 1450, ARCH_MONUMENTAL)
                for n in range(40)]
        slots, plan = assign_facade_slots(
            self._buildings(mix), 5,
            {"looks_per_era": 12, "buildings_per_look": 80},
        )
        # 160 houses is two looks; the 40 churches must not buy a third.
        self.assertEqual(plan.per_era[1], 2)
        self.assertEqual(
            set(slots[160:].tolist()), {plan.archetype_slot(ARCH_MONUMENTAL)},
            "the monumental buildings did not all land on the monumental slot",
        )

    def test_the_slots_index_the_textures_that_get_generated(self):
        """A building carries a slot number, not a style. Generating a
        different set of textures than the slots were numbered against puts
        the wrong wall on every building after the first era whose count
        changed -- and nothing would look broken, just wrong."""
        import tempfile

        from src.buildings import assign_facade_slots
        from src.config import DEFAULTS
        from src.facade import generate_facade_textures

        mix = []
        for era_year, count in ((1890, 16), (1933, 400), (1960, 90)):
            mix += [(f"NL.IMBAG.Pand.{era_year}{n:07d}", era_year, 0)
                    for n in range(count)]
        slots, plan = assign_facade_slots(
            self._buildings(mix), 5,
            {"looks_per_era": 12, "buildings_per_look": 80},
        )
        cfg = dict(DEFAULTS["facade"], texture_px=64, photo_textures=False,
                   normal_map=False, ground_floor=False)
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_facade_textures(
                Path(tmp), facade_cfg=cfg, plan=plan
            )
        self.assertEqual(
            len(paths), plan.total_slots,
            "the texture set is not the size the slots address",
        )
        self.assertLess(
            int(slots.max()), len(paths),
            f"slot {slots.max()} addresses a texture that was never written",
        )

    def test_both_allocation_levers_do_something(self):
        """The cap and the budget are separate knobs and every test here used
        one setting of each, so freezing either passed. The cap only binds
        once an era is dense enough to ask for more than it allows."""
        from src.facade import LookPlan

        # 2000 buildings in one era: 25 looks at a budget of 80, which the
        # cap has to cut down.
        dense = [0, 2000, 0, 0, 0]
        self.assertEqual(LookPlan.from_counts(dense, cap=12, target=80).per_era[1], 12)
        self.assertEqual(LookPlan.from_counts(dense, cap=4, target=80).per_era[1], 4)
        self.assertEqual(LookPlan.from_counts(dense, cap=1, target=80).per_era[1], 1)

        # The budget decides it below the cap.
        moderate = [0, 480, 0, 0, 0]
        for target, expect in ((40, 12), (80, 6), (120, 4), (240, 2), (960, 1)):
            self.assertEqual(
                LookPlan.from_counts(moderate, cap=12, target=target).per_era[1],
                expect,
                f"480 buildings at {target} apiece should be {expect} looks",
            )

    def test_the_cap_cannot_exceed_what_the_palettes_fill(self):
        """Past twelve the looks repeat, so the textures would be duplicates
        taking a material slot each."""
        from src.facade import MAX_LOOKS_PER_ERA, LookPlan

        plan = LookPlan.from_counts([0, 100000, 0, 0, 0], cap=500, target=1)
        self.assertEqual(plan.per_era[1], MAX_LOOKS_PER_ERA)

    def test_the_shipped_budget_is_the_measured_one(self):
        """80 is not arbitrary: it is what puts the busiest Bronbeek wall on
        81 of 1322 buildings rather than 967, at 20 textures."""
        from src.config import DEFAULTS
        from src.facade import LookPlan

        budget = DEFAULTS["facade"]["buildings_per_look"]
        self.assertEqual(budget, 80)
        plan = LookPlan.from_counts(
            [16, 967, 240, 27, 72],
            cap=DEFAULTS["facade"]["looks_per_era"], target=budget,
        )
        self.assertEqual(plan.per_era, (1, 12, 3, 1, 1))
        self.assertEqual(plan.total_slots, 20)
        self.assertLess(max(967 / 12, 240 / 3), 100)

    def test_the_pipeline_hands_the_plan_to_the_texture_stage(self):
        """Both sides have to use one allocation. The buildings stage
        measures it; the facade stage has to be given it rather than
        recomputing from config and hoping they agree."""
        import inspect

        import pipeline

        source = inspect.getsource(pipeline.run)
        self.assertIn("LookPlan(buildings.facade_looks)", source)
        self.assertIn("plan=plan", source)

    def test_build_buildings_uses_the_shared_rule(self):
        import inspect

        from src import buildings

        source = inspect.getsource(buildings.build_buildings)
        self.assertIn("assign_facade_slots(", source)
        self.assertIn("result.facade_looks = plan.per_era", source)

    def test_the_count_is_clamped_to_what_the_palettes_fill(self):
        from src.facade import MAX_LOOKS_PER_ERA, looks_per_era

        self.assertEqual(looks_per_era({}), 1, "the default here is one look")
        self.assertEqual(looks_per_era({"looks_per_era": 3}), 3)
        self.assertEqual(looks_per_era({"looks_per_era": 0}), 1)
        self.assertEqual(looks_per_era({"looks_per_era": -4}), 1)
        self.assertEqual(
            looks_per_era({"looks_per_era": 99}), MAX_LOOKS_PER_ERA
        )

    def test_the_config_default_asks_for_more_than_one(self):
        """Shipping the feature switched off would be shipping nothing."""
        from src.config import DEFAULTS

        self.assertGreater(DEFAULTS["facade"]["looks_per_era"], 1)

    def test_the_config_rejects_nonsense_and_clamps_excess(self):
        import tempfile

        from src.facade import MAX_LOOKS_PER_ERA

        def load(looks):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "c.json"
                path.write_text(json.dumps({
                    "name": "t",
                    "bbox": {"xmin": 136000, "ymin": 455000,
                             "xmax": 137000, "ymax": 456000},
                    "facade": {"looks_per_era": looks},
                }))
                return load_config(path)

        self.assertEqual(load(4).facade["looks_per_era"], 4)
        self.assertEqual(
            load(500).facade["looks_per_era"], MAX_LOOKS_PER_ERA
        )
        with self.assertRaises(ValueError):
            load(0)


class TestFacade(unittest.TestCase):
    def test_tile_wraps_seamlessly(self):
        """A wall repeats the tile, so opposite edges have to match."""
        tile = render_facade_tile(STYLES[0], 128, seed=7).astype(int)
        left_gap = np.abs(tile[:, 0] - tile[:, -1]).mean()
        self.assertLess(left_gap, 12.0, "left and right edges do not match")

    def test_styles_scale_with_height(self):
        self.assertEqual(style_for_height(6.0, 4), 0)
        self.assertEqual(style_for_height(40.0, 4), 3)
        # With a single variant every building shares the one material.
        self.assertEqual(style_for_height(40.0, 1), 0)

    def test_windows_are_darker_than_wall(self):
        tile = render_facade_tile(STYLES[0], 256, seed=3).astype(float)
        self.assertLess(tile.min(), tile.mean())

    def test_normal_map_is_mostly_flat_and_tiles(self):
        from src.facade import relief_to_normal_map, render_facade_layers

        _, relief = render_facade_layers(STYLES[0], 128, seed=5)
        normal = relief_to_normal_map(relief)

        # A facade is nearly flat, so the map sits close to (128, 128, 255).
        means = normal.reshape(-1, 3).mean(axis=0)
        self.assertAlmostEqual(means[0], 127.5, delta=6)
        self.assertAlmostEqual(means[1], 127.5, delta=6)
        self.assertGreater(means[2], 235)

        # Gradients wrap, so the normal map tiles as exactly as the colour does.
        left_right = np.abs(normal[:, 0].astype(int) - normal[:, -1].astype(int))
        top_bottom = np.abs(normal[0].astype(int) - normal[-1].astype(int))
        self.assertEqual(left_right.max(), 0)
        self.assertEqual(top_bottom.max(), 0)

    def test_relief_and_colour_stay_in_register(self):
        from src.facade import render_facade_layers

        rgb, relief = render_facade_layers(STYLES[0], 128, seed=9)
        self.assertEqual(rgb.shape[:2], relief.shape)
        # The recessed glass must be the deepest thing in the tile.
        self.assertLess(relief.min(), 0.4)
        self.assertGreater(relief.max(), 0.55)

    def test_normal_map_written_alongside_colour(self):
        from src.facade import generate_facade_textures

        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "texture_px": 64, "variants": 1, "seed": 1,
                "normal_map": True, "relief_depth": 0.035,
                "ground_floor": False,
            }
            pairs = generate_facade_textures(Path(tmp), facade_cfg=cfg)
            # One era variant, plus the monumental and industrial styles that
            # every run carries.
            self.assertEqual(len(pairs), 3)
            colour, normal = pairs[0]
            self.assertTrue(colour.is_file())
            self.assertIsNotNone(normal)
            # Unity keys off the _normal suffix to set the texture type.
            self.assertTrue(normal.name.endswith("_normal.png"))

            cfg["normal_map"] = False
            _, without = generate_facade_textures(Path(tmp), facade_cfg=cfg)[0]
            self.assertIsNone(without)

    def test_ground_floor_texture_is_written_last(self):
        """The Blender stage addresses it as the trailing material slot."""
        from src.facade import generate_facade_textures

        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "texture_px": 64, "variants": 2, "seed": 1,
                "normal_map": False, "relief_depth": 0.035,
                "ground_floor": True,
            }
            pairs = generate_facade_textures(Path(tmp), facade_cfg=cfg)
            # Two era variants, the two archetype styles, then the ground.
            self.assertEqual(len(pairs), 5)
            self.assertEqual(pairs[-1][0].name, "facade_ground.png")


class TestPhotoTextures(unittest.TestCase):
    """Photographed masonry under the generated windows."""

    def test_every_style_has_a_texture_mapped(self):
        from src.facade import GROUND_STYLES, STYLES, wall_styles
        from src.textures import WALL_TEXTURES

        for style in wall_styles(len(STYLES)) + list(GROUND_STYLES):
            self.assertIn(style.name, WALL_TEXTURES, f"{style.name} has no surface")

    def test_a_texture_is_tiled_to_its_real_world_size(self):
        """Bricks at the wrong scale are the first thing the eye catches."""
        from src.textures import _tile_to

        # A photograph with one white row, so repeats are countable.
        photo = np.zeros((100, 100, 3), dtype=np.float64)
        photo[0, :] = 255

        # A 3 m photograph over a 6 m tile has to repeat twice.
        tiled = _tile_to(photo, 200, repeats_x=1.0, repeats_y=2.0)
        white_rows = np.flatnonzero(tiled[:, 0, 0] > 128)
        self.assertEqual(len(white_rows), 2)

    def test_tinting_moves_the_colour_but_keeps_the_grain(self):
        from src.textures import _tint

        rng = np.random.default_rng(3)
        photo = rng.uniform(40, 210, (64, 64, 3))
        tinted = _tint(photo, (150, 90, 70), strength=1.0)

        np.testing.assert_allclose(
            tinted.reshape(-1, 3).mean(axis=0), (150, 90, 70), atol=6.0
        )
        # Detail survives exactly: each channel is only rescaled, so every
        # brick, joint and stain stays where it was.
        for channel in range(3):
            self.assertGreater(
                np.corrcoef(
                    photo[:, :, channel].ravel(), tinted[:, :, channel].ravel()
                )[0, 1],
                0.9999,
            )

    def test_no_tint_leaves_the_photograph_alone(self):
        from src.textures import _tint

        photo = np.full((8, 8, 3), 120.0)
        np.testing.assert_array_equal(_tint(photo, (10, 200, 30), 0.0), photo)

    def test_a_missing_style_falls_back_rather_than_failing(self):
        from src.facade import FacadeStyle
        from src.textures import TextureUnavailable, wall_base

        unknown = FacadeStyle(
            name="not_a_real_style",
            wall_rgb=(1, 2, 3), trim_rgb=(1, 2, 3), glass_rgb=(1, 2, 3),
            windows_across=1, window_width_frac=0.2, window_height_frac=0.5,
            sill_frac=0.2, grain=0.1,
        )
        with self.assertRaises(TextureUnavailable):
            wall_base(unknown, 64, Path("/nonexistent/work/area"),
                      tile_width_m=4.0, tile_height_m=3.0)

    def test_a_base_replaces_the_wall_but_not_the_windows(self):
        from src.facade import STYLES, render_facade_layers

        style = STYLES[0]
        base = np.zeros((64, 64, 3), dtype=np.float64)
        base[:, :] = (10, 200, 10)  # unmistakable green

        pixels, _ = render_facade_layers(style, 64, seed=1, base=base)
        greenest = pixels[:, :, 1].astype(int) - pixels[:, :, 0].astype(int)
        self.assertGreater(greenest.max(), 100, "the photograph is not showing")
        # The glass is still drawn on top, so not everything is green.
        self.assertLess(greenest.min(), 20)

    def test_photo_textures_can_be_turned_off(self):
        from src.facade import _wall_bases, wall_styles

        self.assertEqual(
            _wall_bases(wall_styles(1), 64, Path("/tmp"), {"photo_textures": False}),
            {},
        )


class TestArchedOpenings(unittest.TestCase):
    """A rounded head is the clearest sign a wall is not a stack of storeys."""

    @staticmethod
    def _monumental():
        from src.facade import EXTRA_STYLES

        return EXTRA_STYLES["monumental"]

    def test_the_arch_is_widest_at_the_springing_and_closes_at_the_crown(self):
        from src.facade import render_facade_layers

        style = self._monumental()
        pixels, relief = render_facade_layers(style, 256, seed=1)

        # Glass sits deepest in the relief, so it marks the opening.
        opening = relief < 0.35
        rows = np.flatnonzero(opening.any(axis=1))
        self.assertTrue(len(rows) > 10, "no opening was drawn")

        widths = opening.sum(axis=1)[rows]
        # Row 0 is the bottom of the tile, so the last rows are the crown.
        self.assertGreater(
            widths[: len(widths) // 3].mean(),
            widths[-3:].mean(),
            "the arch is upside down: it narrows towards the sill, not the head",
        )

    def test_the_crown_is_narrower_than_the_opening_below_it(self):
        from src.facade import render_facade_layers

        _, relief = render_facade_layers(self._monumental(), 256, seed=1)
        opening = relief < 0.35
        rows = np.flatnonzero(opening.any(axis=1))
        top_row, mid_row = rows[-1], rows[len(rows) // 2]
        self.assertLess(opening[top_row].sum(), opening[mid_row].sum())

    def test_an_unarched_style_has_square_openings(self):
        from src.facade import STYLES, render_facade_layers

        _, relief = render_facade_layers(STYLES[0], 256, seed=1)
        opening = relief < 0.35
        rows = np.flatnonzero(opening.any(axis=1))
        widths = opening.sum(axis=1)[rows]
        # Every row of a rectangular window is the same width.
        self.assertEqual(len(set(widths.tolist())), 1)


class TestCoverageTiling(unittest.TestCase):
    """The AHN WCS caps a coverage at 4000 px, so a large area needs tiles."""

    def test_a_small_area_is_one_request(self):
        from src.elevation import MAX_COVERAGE_PX
        from src.geo import tile_edges

        # 1 km at 0.5 m is 2000 px, well inside the cap.
        self.assertEqual(tile_edges(2000, MAX_COVERAGE_PX), [(0, 2000)])

    def test_the_reported_three_kilometre_area_splits_two_by_two(self):
        from src.elevation import MAX_COVERAGE_PX
        from src.geo import tile_edges

        # 3 km at 0.5 m is 6000 px, which is what the service refused.
        spans = tile_edges(6000, MAX_COVERAGE_PX)
        self.assertEqual(spans, [(0, 3000), (3000, 6000)])

    def test_tiles_are_contiguous_and_cover_everything(self):
        from src.geo import tile_edges

        for total in (1, 999, 4000, 4001, 6000, 25000):
            spans = tile_edges(total, 4000)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], total)
            for (_, end), (start, _) in zip(spans, spans[1:]):
                self.assertEqual(end, start, "a gap or an overlap between tiles")
            for start, end in spans:
                self.assertLessEqual(end - start, 4000)

    def test_the_split_is_even_rather_than_leaving_a_sliver(self):
        from src.geo import tile_edges

        spans = tile_edges(4100, 4000)
        widths = [end - start for start, end in spans]
        self.assertEqual(widths, [2050, 2050])

    def test_tile_bounds_land_on_whole_pixels(self):
        """A boundary mid-pixel would be rounded differently either side."""
        from src.geo import BBox, tile_edges

        bbox = BBox(77632.59, 433641.39, 80632.59, 436641.39)
        resolution = 0.5
        columns = int(round(bbox.width / resolution))

        edges = []
        for start, end in tile_edges(columns, 4000):
            left = bbox.xmin + start * resolution
            right = bbox.xmin + end * resolution
            # Each tile spans a whole number of source cells.
            self.assertAlmostEqual((right - left) / resolution % 1.0, 0.0, places=9)
            edges.append((left, right))

        # And they join exactly, with no gap for a seam to appear in.
        self.assertAlmostEqual(edges[0][1], edges[1][0], places=9)
        self.assertAlmostEqual(edges[-1][1], bbox.xmax, places=9)


class TestServiceExceptionDetail(unittest.TestCase):
    """The reason a request was refused must survive into the error."""

    def test_the_ogc_exception_text_is_not_cut_off_by_the_xml_preamble(self):
        from src.http_util import extract_service_exception, looks_like_xml

        # The body the AHN WCS actually returned, preamble and all.
        body = (
            b'<?xml version="1.0" encoding="UTF-8"?>\n'
            b'<ows:ExceptionReport xmlns:ows="http://www.opengis.net/ows/2.0" '
            b'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="2.0.1" '
            b'xml:lang="en-US" xsi:schemaLocation="http://www.opengis.net/ows/2.0 '
            b'http://schemas.opengis.net/ows/2.0/owsExceptionReport.xsd">\n'
            b'  <ows:Exception exceptionCode="InvalidParameterValue" locator="size">\n'
            b"    <ows:ExceptionText>msWCSGetCoverage20(): WCS server error. "
            b"Raster size out of range, width and height of resulting coverage "
            b"must be no more than MAXSIZE=4000.</ows:ExceptionText>\n"
            b"  </ows:Exception>\n</ows:ExceptionReport>\n"
        )
        self.assertTrue(looks_like_xml(body))

        # Truncating the body at 400 characters stopped at "msWCS", which is
        # what hid the limit.
        self.assertNotIn("MAXSIZE", body.decode()[:400])

        detail = extract_service_exception(body)
        self.assertIn("MAXSIZE=4000", detail)


class TestDetailBlend(unittest.TestCase):
    """The detail pass has to stay affordable as the area grows."""

    @staticmethod
    def _scene(px: int, side_m: float):
        from PIL import Image

        from src.geo import BBox
        from src.surfaces import SurfaceSet

        rng = np.random.default_rng(5)
        image = Image.fromarray(
            (rng.random((px, px, 3)) * 255).astype(np.uint8), mode="RGB"
        )
        grid = rng.integers(0, 12, (65, 65)).astype(np.uint8)
        return image, SurfaceSet(class_grid=grid), BBox(0, 0, side_m, side_m)

    def test_the_grain_keeps_its_size_on_the_ground_whatever_the_area(self):
        """It used to be a fraction of the image, so it grew with the area."""
        from src.surfaces import (
            CLASS_DETAIL, CLASS_ROAD, DETAIL_NOISE_CELLS, DETAIL_NOISE_PX,
        )

        feature_px = DETAIL_NOISE_PX / DETAIL_NOISE_CELLS
        wanted = CLASS_DETAIL[CLASS_ROAD]["cells"]

        for side_m, px in ((600.0, 3072), (3000.0, 4096), (20000.0, 8192)):
            metres_per_px = side_m / px
            scale = wanted / (metres_per_px * feature_px)
            on_ground = feature_px * scale * metres_per_px
            self.assertAlmostEqual(on_ground, wanted, places=6)

    def test_a_tiling_field_repeats_rather_than_running_out(self):
        from src.surfaces import _sample_tiled

        field = np.arange(16, dtype=np.float64).reshape(4, 4)
        rows = np.arange(16)
        columns = np.arange(16)
        sampled = _sample_tiled(field, rows, columns, scale=1.0)

        self.assertEqual(sampled.shape, (16, 16))
        # Wraps every four pixels at this scale.
        np.testing.assert_array_equal(sampled[:4, :4], sampled[4:8, 4:8])

    def test_sampling_stretches_the_field_by_the_scale(self):
        from src.surfaces import _sample_tiled

        field = np.arange(16, dtype=np.float64).reshape(4, 4)
        sampled = _sample_tiled(field, np.arange(8), np.arange(8), scale=2.0)
        # Each field pixel now covers two image pixels.
        self.assertEqual(sampled[0, 0], sampled[0, 1])
        self.assertEqual(sampled[0, 0], sampled[1, 0])
        self.assertNotEqual(sampled[0, 0], sampled[0, 2])

    def test_strips_leave_no_seam(self):
        """Sampling by absolute coordinate is what keeps strips continuous."""
        from PIL import Image

        from src.surfaces import DETAIL_STRIP_PX, blend_surface_detail

        image, surfaces, bbox = self._scene(px=1024, side_m=600.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aerial.png"
            image.save(path)
            blend_surface_detail(path, surfaces, bbox, strength=0.22)
            blended = np.asarray(Image.open(path).convert("RGB")).astype(float)

        def row_jump(row):
            return np.abs(blended[row] - blended[row - 1]).mean()

        boundaries = [
            row_jump(r) for r in range(DETAIL_STRIP_PX, blended.shape[0], DETAIL_STRIP_PX)
        ]
        ordinary = [row_jump(r) for r in range(5, blended.shape[0] - 5, 13)]
        self.assertTrue(boundaries)
        self.assertLessEqual(max(boundaries), np.percentile(ordinary, 99))

    def test_the_photo_is_changed_gently(self):
        from PIL import Image

        from src.surfaces import blend_surface_detail

        image, surfaces, bbox = self._scene(px=512, side_m=600.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aerial.png"
            image.save(path)
            before = np.asarray(Image.open(path).convert("RGB")).astype(int)
            blend_surface_detail(path, surfaces, bbox, strength=0.22)
            after = np.asarray(Image.open(path).convert("RGB")).astype(int)

        change = np.abs(after - before).mean()
        self.assertGreater(change, 0.1, "the blend did nothing")
        self.assertLess(change, 20.0, "the blend overwhelmed the photo")

    def test_zero_strength_leaves_the_photo_alone(self):
        from PIL import Image

        from src.surfaces import blend_surface_detail

        image, surfaces, bbox = self._scene(px=256, side_m=600.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aerial.png"
            image.save(path)
            before = np.asarray(Image.open(path).convert("RGB"))
            self.assertFalse(
                blend_surface_detail(path, surfaces, bbox, strength=0.0)
            )
            np.testing.assert_array_equal(
                np.asarray(Image.open(path).convert("RGB")), before
            )

    def test_no_land_cover_is_not_an_error(self):
        from PIL import Image

        from src.geo import BBox
        from src.surfaces import SurfaceSet, blend_surface_detail

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aerial.png"
            Image.new("RGB", (64, 64)).save(path)
            self.assertFalse(
                blend_surface_detail(
                    path, SurfaceSet(), BBox(0, 0, 600, 600), strength=0.22
                )
            )


class TestStructures(unittest.TestCase):
    """Bridges are measured; tunnels are drawn. Both have to be marked."""

    @staticmethod
    def _square(cx=0.0, cy=0.0, side=20.0):
        h = side / 2
        return np.array(
            [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]]
        )

    def test_a_ring_and_its_heights_stay_in_step(self):
        """A closed ring repeats its first point; dropping only one crashed."""
        from src.structures import extrude_ring

        closed = np.vstack([self._square(), self._square()[:1]])
        heights = np.arange(len(closed), dtype=float)
        walls = extrude_ring(closed, 10.0, heights)
        self.assertGreater(len(walls), 0)

    def test_mismatched_heights_are_refused_rather_than_broadcast(self):
        from src.structures import extrude_ring

        with self.assertRaises(ValueError):
            extrude_ring(self._square(), 10.0, np.zeros(99))

    def test_a_wall_spans_from_top_to_bottom(self):
        from src.structures import extrude_ring

        walls = extrude_ring(self._square(), 12.0, 2.0)
        z = walls[:, :, 2]
        self.assertAlmostEqual(float(z.max()), 12.0)
        self.assertAlmostEqual(float(z.min()), 2.0)

    def test_a_deck_takes_the_median_of_what_the_dsm_sees(self):
        """The DSM over a bridge also holds railings and passing lorries."""
        from src.structures import deck_height

        def sampler(x, y):
            readings = np.full(len(np.atleast_1d(x)), 9.0)
            # A gantry and a lorry, which a mean would follow and a median not.
            readings[::7] = 40.0
            return readings

        self.assertAlmostEqual(
            deck_height([self._square()], sampler, ground=0.0), 9.0, places=6
        )

    def test_a_deck_below_the_ground_it_crosses_is_not_a_deck(self):
        from src.structures import deck_height

        low = lambda x, y: np.full(len(np.atleast_1d(x)), 1.0)  # noqa: E731
        self.assertIsNone(deck_height([self._square()], low, ground=8.0))

    def test_a_deck_with_no_reading_returns_nothing(self):
        from src.structures import deck_height

        nothing = lambda x, y: np.full(len(np.atleast_1d(x)), np.nan)  # noqa: E731
        self.assertIsNone(deck_height([self._square()], nothing, ground=0.0))

    def test_a_mast_is_not_mistaken_for_a_deck(self):
        from src.structures import MAX_DECK_NAP, deck_height

        mast = lambda x, y: np.full(  # noqa: E731
            len(np.atleast_1d(x)), MAX_DECK_NAP + 20.0
        )
        self.assertIsNone(deck_height([self._square()], mast, ground=0.0))

    def test_the_tunnel_is_level_with_the_ground_at_its_portals(self):
        from src.structures import tunnel_depth_profile

        # A 2 km tunnel running east-west.
        along = np.linspace(0, 2000, 201)
        points = np.column_stack([along, np.zeros_like(along)])
        depth = tunnel_depth_profile(points, depth_m=18.0, ramp_m=350.0)

        self.assertAlmostEqual(float(depth[0]), 0.0, places=6)
        self.assertAlmostEqual(float(depth[-1]), 0.0, places=6)

    def test_the_tunnel_reaches_its_full_depth_in_the_middle(self):
        from src.structures import tunnel_depth_profile

        along = np.linspace(0, 2000, 201)
        points = np.column_stack([along, np.zeros_like(along)])
        depth = tunnel_depth_profile(points, depth_m=18.0, ramp_m=350.0)

        self.assertAlmostEqual(float(depth.max()), 18.0, places=6)
        # The deepest point is in the middle, not at an end.
        self.assertGreater(float(depth[len(depth) // 2]), 17.9)

    def test_the_descent_has_no_kink_in_it(self):
        """Smoothstep, so the road eases into the ramp."""
        from src.structures import tunnel_depth_profile

        along = np.linspace(0, 2000, 401)
        points = np.column_stack([along, np.zeros_like(along)])
        depth = tunnel_depth_profile(points, depth_m=18.0, ramp_m=350.0)

        slope = np.diff(depth)
        # The gradient changes smoothly rather than stepping.
        self.assertLess(float(np.abs(np.diff(slope)).max()), 0.05)

    def test_a_short_tunnel_still_reaches_its_depth(self):
        from src.structures import tunnel_depth_profile

        along = np.linspace(0, 80, 41)
        points = np.column_stack([along, np.zeros_like(along)])
        depth = tunnel_depth_profile(points, depth_m=18.0, ramp_m=350.0)
        self.assertGreater(float(depth.max()), 17.0)
        self.assertAlmostEqual(float(depth[0]), 0.0, places=6)


class TestBridgeDecks(unittest.TestCase):
    """A deck may not sink into the ground it crosses.

    A BGT bridge part is often a whole approach, and in dike country the ground
    climbs to meet the deck at the abutment. One flat level for the part then
    leaves the last stretch buried: a fifth of all deck geometry went under the
    terrain over a 2 km area near Dordrecht.
    """

    class _Ramp:
        """Ground rising steeply across the part, as an approach does."""

        @staticmethod
        def sample(x, y):
            return 0.9 * np.atleast_1d(np.asarray(x, dtype=float))

    @staticmethod
    def _flat():
        # Two triangles spanning x = 0 to 20, so the ramp climbs 18 m over them.
        return np.array(
            [
                [[0.0, 0.0], [20.0, 0.0], [20.0, 8.0]],
                [[0.0, 0.0], [20.0, 8.0], [0.0, 8.0]],
            ]
        )

    def test_a_deck_over_rising_ground_is_not_buried(self):
        from src.structures import deck_corner_heights

        z = deck_corner_heights(self._flat(), level=9.0, terrain=self._Ramp(), lift=0.05)
        corners = self._flat().reshape(-1, 2)
        ground = self._Ramp.sample(corners[:, 0], corners[:, 1])
        self.assertTrue(
            (z.reshape(-1) >= ground - 1e-9).all(),
            f"{int((z.reshape(-1) < ground - 1e-9).sum())} deck corners are under "
            f"the ground they cross",
        )

    def test_the_span_itself_stays_flat(self):
        """Only the ends lift. Levelling the whole part to its highest ground
        would take a bridge over a dike and stand it on stilts."""
        from src.structures import deck_corner_heights

        z = deck_corner_heights(self._flat(), level=9.0, terrain=self._Ramp(), lift=0.05)
        self.assertAlmostEqual(float(z.min()), 9.05, places=6)

    def test_ground_below_the_deck_leaves_it_alone(self):
        from src.structures import deck_corner_heights

        class Low:
            @staticmethod
            def sample(x, y):
                return np.zeros(len(np.atleast_1d(x)))

        z = deck_corner_heights(self._flat(), level=9.0, terrain=Low(), lift=0.05)
        np.testing.assert_allclose(z, 9.05)


class TestRoadClasses(unittest.TestCase):
    """A Dutch street is brick, and its cycle path is red. Both matter."""

    def test_function_decides_the_class(self):
        from src.surfaces import (
            CLASS_CYCLE, CLASS_FOOTPATH, CLASS_PARKING, CLASS_ROAD,
            CLASS_TRANSIT, road_class,
        )

        self.assertEqual(road_class("rijbaan lokale weg", "asfalt"), CLASS_ROAD)
        self.assertEqual(road_class("fietspad", "asfalt"), CLASS_CYCLE)
        self.assertEqual(road_class("voetpad", ""), CLASS_FOOTPATH)
        self.assertEqual(road_class("voetpad op trap", ""), CLASS_FOOTPATH)
        self.assertEqual(road_class("OV-baan", "asfalt"), CLASS_TRANSIT)
        self.assertEqual(road_class("parkeervlak", "asfalt"), CLASS_PARKING)

    def test_a_brick_carriageway_is_told_apart_from_asphalt(self):
        from src.surfaces import CLASS_ROAD, CLASS_ROAD_BRICK, road_class

        self.assertEqual(
            road_class("rijbaan lokale weg", "gebakken klinkers"), CLASS_ROAD_BRICK
        )
        self.assertEqual(
            road_class("rijbaan lokale weg", "betonstraatstenen"), CLASS_ROAD_BRICK
        )
        self.assertEqual(road_class("rijbaan lokale weg", "asfalt"), CLASS_ROAD)

    def test_a_cycle_path_is_red_whatever_it_is_made_of(self):
        from src.surfaces import CLASS_CYCLE, road_class

        self.assertEqual(road_class("fietspad", "gebakken klinkers"), CLASS_CYCLE)

    def test_an_unknown_function_falls_back_to_carriageway(self):
        from src.surfaces import CLASS_ROAD, road_class

        self.assertEqual(road_class("", ""), CLASS_ROAD)
        self.assertEqual(road_class("iets nieuws", ""), CLASS_ROAD)

    def test_narrow_surfaces_paint_over_the_broad_ones(self):
        """Junctions overlap, so the order cannot be whatever the API returned."""
        from src.surfaces import (
            CLASS_CYCLE, CLASS_FOOTPATH, CLASS_ROAD, _paint_rank,
        )

        self.assertLess(_paint_rank(CLASS_ROAD), _paint_rank(CLASS_CYCLE))
        self.assertLess(_paint_rank(CLASS_CYCLE), _paint_rank(CLASS_FOOTPATH))


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float)


def _street_fixture(*, with_network=True):
    """A T-junction: a main street running east, a side street going north.

    Carriageways 6 m wide with a 2 m footway either side, the main street's
    north footway broken where the side street joins. Centrelines as NWB
    gives them: one section per stretch between junctions, junction ids shared
    where they meet.
    """
    from src.nwb import Centreline, Network
    from src.surfaces import CLASS_FOOTPATH, CLASS_ROAD, RoadPart

    parts = [
        RoadPart([_rect(0, -3, 60, 3)], CLASS_ROAD, 0),
        RoadPart([_rect(0, -5, 60, -3)], CLASS_FOOTPATH, 0),
        RoadPart([_rect(0, 3, 25, 5)], CLASS_FOOTPATH, 0),
        RoadPart([_rect(35, 3, 60, 5)], CLASS_FOOTPATH, 0),
        RoadPart([_rect(27, 3, 33, 40)], CLASS_ROAD, 0),
        RoadPart([_rect(25, 5, 27, 40)], CLASS_FOOTPATH, 0),
        RoadPart([_rect(33, 5, 35, 40)], CLASS_FOOTPATH, 0),
        # The corners where the footways meet.
        RoadPart([_rect(25, 3, 27, 5)], CLASS_FOOTPATH, 0),
        RoadPart([_rect(33, 3, 35, 5)], CLASS_FOOTPATH, 0),
    ]
    lines = []
    if with_network:
        lines = [
            Centreline(np.array([[-5.0, 0.0], [30.0, 0.0]]), 1, 1, 2, "Hoofdstraat", "RB"),
            Centreline(np.array([[30.0, 0.0], [65.0, 0.0]]), 2, 2, 3, "Hoofdstraat", "RB"),
            Centreline(np.array([[30.0, 0.0], [30.0, 45.0]]), 3, 2, 4, "Zijstraat", "RB"),
        ]
    return parts, Network(lines=lines)


def _build_street(**kwargs):
    from src.roads import build_road_mesh

    parts, network = _street_fixture(with_network=kwargs.pop("with_network", True))
    ground = kwargs.pop("ground", lambda x, y: np.full(np.shape(x), 2.0))
    return parts, build_road_mesh(parts, network, ground=ground, **kwargs)


class TestRoadLadders(unittest.TestCase):
    """The roads are meshed so an editor can work with them.

    What that means, concretely, and what each test pins down: no vertex in
    the middle of a surface; every street a ladder of quads whose rungs run
    kerb to kerb across the whole street; junctions as patches of their own;
    a real kerb where a footway meets the carriageway; and the outline the
    BGT surveyed, not an approximation of it.
    """

    @classmethod
    def setUpClass(cls):
        cls.parts, cls.mesh = _build_street(lift_m=0.03, kerb_m=0.10)

    def _surface_faces(self, mesh=None):
        from src.roads import SURFACE

        mesh = mesh or self.mesh
        return [i for i in range(len(mesh)) if mesh.face_kind[i] == SURFACE]

    def test_no_vertex_lies_inside_a_surface(self):
        import shapely

        edges = shapely.union_all(
            [shapely.Polygon(p.rings[0]).boundary for p in self.parts]
        )
        used = np.unique(np.concatenate([self.mesh.face(i) for i in self._surface_faces()]))
        off = shapely.distance(shapely.points(self.mesh.vertices[used, :2]), edges)
        self.assertLess(float(off.max()), 1e-6, "a vertex sits in the middle of a surface")

    def test_streets_are_ladders_of_quads(self):
        streets = [
            i for i in self._surface_faces()
            if self.mesh.object_names[self.mesh.face_object[i]].startswith("Road_")
        ]
        self.assertGreater(len(streets), 30)
        sizes = self.mesh.face_sizes[streets]
        self.assertTrue((sizes == 4).all(), f"street faces of {sorted(set(sizes))} corners")

    def test_every_rung_runs_kerb_to_kerb_across_the_whole_street(self):
        """A cut across the carriageway carries on through both footways, so a
        cross-section is one edge loop rather than three that do not meet."""
        v = self.mesh.vertices
        main = [
            i for i in self._surface_faces()
            if self.mesh.object_names[self.mesh.face_object[i]] == "Road_Hoofdstraat"
        ]
        rungs = sorted({round(float(x), 3) for i in main for x in v[self.mesh.face(i), 0]})
        # Away from the side street, where the north footway is unbroken.
        checked = 0
        for x in rungs:
            if 22.0 < x < 38.0:
                continue
            ys = {round(float(y), 3) for xx, y in v[:, :2] if abs(xx - x) < 1e-6}
            for edge in (-5.0, -3.0, 3.0, 5.0):
                self.assertIn(edge, ys, f"the rung at x={x} stops short of y={edge}")
            checked += 1
        self.assertGreater(checked, 10)
        gaps = np.diff([x for x in rungs if x < 22.0])
        self.assertLessEqual(float(gaps.max()), 3.0 + 1e-6, "rungs further apart than asked")

    def test_the_junction_is_a_patch_of_its_own(self):
        names = set(self.mesh.object_names)
        self.assertIn("Road_Hoofdstraat", names)
        self.assertIn("Road_Zijstraat", names)
        junctions = [n for n in names if n.startswith("Junction_")]
        self.assertEqual(len(junctions), 1, names)
        self.assertIn("Hoofdstraat_x_Zijstraat", junctions[0])
        # It is where the streets meet.
        index = self.mesh.object_names.index(junctions[0])
        faces = [i for i in self._surface_faces() if self.mesh.face_object[i] == index]
        centre = np.vstack([self.mesh.vertices[self.mesh.face(i)] for i in faces])[:, :2].mean(axis=0)
        self.assertLess(float(np.hypot(*(centre - [30.0, 0.0]))), 6.0)

    def test_the_footway_stands_on_a_kerb(self):
        from src.roads import KERB
        from src.surfaces import CLASS_FOOTPATH, CLASS_ROAD

        v = self.mesh.vertices
        for i in self._surface_faces():
            z = v[self.mesh.face(i), 2]
            want = 2.0 + 0.03 + (0.10 if self.mesh.face_class[i] == CLASS_FOOTPATH else 0.0)
            np.testing.assert_allclose(z, want, atol=1e-9)
            self.assertIn(self.mesh.face_class[i], (CLASS_FOOTPATH, CLASS_ROAD))
        kerbs = [i for i in range(len(self.mesh)) if self.mesh.face_kind[i] == KERB]
        length = sum(
            float(np.hypot(*(v[self.mesh.face(i)][1, :2] - v[self.mesh.face(i)][0, :2])))
            for i in kerbs
        )
        # Both sides of the main street, less the side street's mouth, plus
        # both sides of the side street.
        self.assertGreater(length, 2 * 60 - 6 + 2 * 35 - 10)
        # Welded to the faces it joins: its bottom edge is the carriageway's
        # own vertices, its top the footway's.
        tops = {int(k) for i in kerbs for k in self.mesh.face(i)[2:]}
        bottoms = {int(k) for i in kerbs for k in self.mesh.face(i)[:2]}
        surface = {int(k) for i in self._surface_faces() for k in self.mesh.face(i)}
        self.assertTrue(tops <= surface and bottoms <= surface)

    def test_a_kerb_faces_the_carriageway(self):
        """Wound so its normal points at the lower side, or it is invisible
        from the road with backface culling on."""
        from src.roads import KERB

        v = self.mesh.vertices
        for i in [i for i in range(len(self.mesh)) if self.mesh.face_kind[i] == KERB][:40]:
            a, b, c = v[self.mesh.face(i)[:3]]
            normal = np.cross(b - a, c - b)
            mid = (a + b) / 2
            # The carriageway side of a main-street kerb is towards y = 0; of a
            # side-street kerb towards x = 30.
            towards = np.array([30.0, 0.0]) - mid[:2] if mid[1] > 4.5 else np.array([0.0, -mid[1]])
            self.assertGreater(float(normal[:2] @ towards), 0.0)

    def test_the_outline_is_the_surveyed_one(self):
        import shapely

        tris, _ = self.mesh.triangles()
        u = tris[:, 1, :2] - tris[:, 0, :2]
        w = tris[:, 2, :2] - tris[:, 0, :2]
        area = 0.5 * np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]).sum()
        want = shapely.union_all([shapely.Polygon(p.rings[0]) for p in self.parts]).area
        self.assertAlmostEqual(area, want, delta=0.005 * want)

    def test_the_network_edge_has_a_skirt(self):
        from src.roads import SKIRT

        v = self.mesh.vertices
        skirts = [i for i in range(len(self.mesh)) if self.mesh.face_kind[i] == SKIRT]
        self.assertGreater(len(skirts), 20)
        for i in skirts[:20]:
            z = v[self.mesh.face(i), 2]
            self.assertGreater(float(z.max() - z.min()), 0.3)

    def test_road_uvs_run_across_and_along_the_street(self):
        index = self.mesh.object_names.index("Road_Hoofdstraat")
        starts = self.mesh.loop_starts
        across, along = [], []
        for i in self._surface_faces():
            if self.mesh.face_object[i] != index:
                continue
            uv = self.mesh.loop_uv[starts[i] : starts[i] + self.mesh.face_sizes[i]]
            across.extend(uv[:, 0])
            along.extend(uv[:, 1])
        self.assertAlmostEqual(min(across), -5.0, places=3)
        self.assertAlmostEqual(max(across), 5.0, places=3)
        self.assertGreater(max(along) - min(along), 20.0)

    def test_without_centrelines_the_road_draws_its_own(self):
        """Private lanes and park paths are not in NWB. They are cut along a
        centreline drawn down their own middle instead of left whole."""
        _, mesh = _build_street(with_network=False)
        self.assertGreater(mesh.stats["drawn_centrelines"], 0)
        self.assertGreater(mesh.stats["quad_share"], 0.6)
        self.assertTrue(any(n.startswith("Lanes_") for n in mesh.object_names))

    def test_no_roads_is_an_empty_mesh(self):
        from src.nwb import Network
        from src.roads import build_road_mesh

        mesh = build_road_mesh([], Network(), ground=lambda x, y: np.zeros(np.shape(x)))
        self.assertEqual(len(mesh), 0)
        self.assertEqual(mesh.triangles()[0].shape, (0, 3, 3))

    def test_saved_without_pickles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.npz"
            np.savez(path, **self.mesh.to_npz_fields())
            data = np.load(path)
            for key in data.files:
                self.assertNotEqual(data[key].dtype, object, key)
            self.assertIn("Road_Hoofdstraat", [str(n) for n in data["road_object_names"]])


def _ladder(parts, lines):
    from src.nwb import Network
    from src.roads import build_road_mesh

    return build_road_mesh(
        parts, Network(lines=lines),
        ground=lambda x, y: np.full(np.shape(x), 2.0), lift_m=0.03, kerb_m=0.10,
    )


def _off_the_kerbs(mesh, parts):
    """How far the furthest surface vertex is from every surveyed edge."""
    import shapely

    from src.roads import SURFACE

    edges = shapely.union_all(
        [shapely.Polygon(p.rings[0], p.rings[1:]).boundary for p in parts]
    )
    used = np.unique(np.concatenate(
        [mesh.face(i) for i in range(len(mesh)) if mesh.face_kind[i] == SURFACE]
    ))
    return float(shapely.distance(shapely.points(mesh.vertices[used, :2]), edges).max())


def _street_faces(mesh, prefix="Road_"):
    from src.roads import SURFACE

    return [
        i for i in range(len(mesh))
        if mesh.face_kind[i] == SURFACE
        and mesh.object_names[mesh.face_object[i]].startswith(prefix)
    ]


class TestRoadLadderCases(unittest.TestCase):
    """The shapes a real street throws at the cutter, one at a time."""

    @staticmethod
    def _line(points, ident, a, b, name):
        from src.nwb import Centreline

        return Centreline(np.asarray(points, dtype=float), ident, a, b, name, "RB")

    def test_a_street_does_not_cut_across_its_neighbour(self):
        """Two streets side by side, footways touching. Each street's rungs
        stop at its own edge; otherwise both sets land on both streets and
        every face is cut twice."""
        from src.surfaces import CLASS_FOOTPATH, CLASS_ROAD, RoadPart

        parts = [
            RoadPart([_rect(0, -3, 60, 3)], CLASS_ROAD, 0),
            RoadPart([_rect(0, -5, 60, -3)], CLASS_FOOTPATH, 0),
            RoadPart([_rect(0, 3, 60, 5)], CLASS_FOOTPATH, 0),
            # The second street is a metre shorter at each end, so its
            # kerb corners -- and the stations at them -- fall elsewhere.
            # Sixty centimetres of verge between the footways keeps every
            # corner nearer its own street's centreline than the other's.
            RoadPart([_rect(1, 5.6, 59, 8)], CLASS_FOOTPATH, 0),
            RoadPart([_rect(1, 8, 59, 14)], CLASS_ROAD, 0),
            RoadPart([_rect(1, 14, 59, 16)], CLASS_FOOTPATH, 0),
        ]
        mesh = _ladder(parts, [
            self._line([[-5, 0], [65, 0]], 1, 1, 2, "Zuid"),
            self._line([[-5, 11], [65, 11]], 2, 3, 4, "Noord"),
        ])
        v = mesh.vertices
        far = {round(float(x), 3) for x, y in v[:, :2] if abs(y - 16.0) < 1e-6}
        near = {round(float(x), 3) for x, y in v[:, :2] if abs(y + 5.0) < 1e-6}
        # Each outer kerb carries its own street's rungs and no more: about
        # one every 3 m, not the two streets' sets interleaved.
        self.assertLessEqual(len(far), 60 / 3.0 + 4)
        self.assertLessEqual(len(near), 60 / 3.0 + 4)
        self.assertNotEqual(far, near)

    def test_a_footway_beyond_a_verge_is_cut_with_the_street(self):
        from src.surfaces import CLASS_FOOTPATH, CLASS_ROAD, RoadPart

        parts = [
            RoadPart([_rect(0, -3, 60, 3)], CLASS_ROAD, 0),
            # Two metres of grass, then the footway.
            RoadPart([_rect(0, 5, 60, 7)], CLASS_FOOTPATH, 0),
        ]
        mesh = _ladder(parts, [self._line([[-5, 0], [65, 0]], 1, 1, 2, "Laan")])
        footway = [
            i for i in _street_faces(mesh)
            if mesh.face_class[i] == CLASS_FOOTPATH
        ]
        self.assertGreater(len(footway), 15, "the footway past the verge was left whole")
        self.assertTrue((mesh.face_sizes[footway] == 4).all())

    def test_a_slightly_wavy_kerb_still_gives_quads(self):
        """The survey wobbles: a kerb vertex every metre and a half, eight
        centimetres off the line. Kept, every one turns a quad into a
        pentagon; straightened, the kerb moves by less than ten centimetres."""
        from src.surfaces import CLASS_ROAD, RoadPart

        xs = np.arange(0.0, 60.01, 1.5)
        wobble = 3.0 + 0.08 * (np.arange(len(xs)) % 2)
        ring = np.vstack([[[0.0, -3.0], [60.0, -3.0]], np.column_stack([xs, wobble])[::-1],
                          [[0.0, -3.0]]])
        parts = [RoadPart([ring], CLASS_ROAD, 0)]
        mesh = _ladder(parts, [self._line([[-5, 0], [65, 0]], 1, 1, 2, "Kade")])
        faces = _street_faces(mesh)
        self.assertGreater(len(faces), 15)
        self.assertGreater(float((mesh.face_sizes[faces] == 4).mean()), 0.9)
        self.assertLess(_off_the_kerbs(mesh, parts), 0.25 + 1e-6)

    def test_a_parking_bay_corner_gets_a_rung(self):
        """A kerb corner is a vertex that cannot be straightened away, so a
        rung has to pass through it or the face beside it is a pentagon."""
        from src.surfaces import CLASS_FOOTPATH, CLASS_PARKING, CLASS_ROAD, RoadPart

        parts = [
            RoadPart([_rect(0, -3, 60, 3)], CLASS_ROAD, 0),
            # A bay cut into the footway, from x = 20.5 to 31.3.
            RoadPart([_rect(20.5, 3, 31.3, 5)], CLASS_PARKING, 0),
            RoadPart([np.array([[0, 3], [20.5, 3], [20.5, 5], [31.3, 5], [31.3, 3],
                                [60, 3], [60, 7], [0, 7], [0, 3]], dtype=float)],
                     CLASS_FOOTPATH, 0),
        ]
        mesh = _ladder(parts, [self._line([[-5, 0], [65, 0]], 1, 1, 2, "Singel")])
        faces = _street_faces(mesh)
        # Where the bay's back kerb meets a rung the footway face beside it
        # has that point on its straight side: a T, which no cut can avoid.
        # Every face still has four corners.
        corners = []
        for i in faces:
            ring = mesh.vertices[mesh.face(i)][:, :2]
            before = ring - np.roll(ring, 1, axis=0)
            after = np.roll(ring, -1, axis=0) - ring
            cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
            scale = np.hypot(*before.T) * np.hypot(*after.T)
            corners.append(int((np.abs(cross) > 0.02 * scale).sum()))
        self.assertEqual(set(corners), {4}, f"faces of {sorted(set(corners))} corners")
        rungs = {round(float(x), 3) for x, y in mesh.vertices[:, :2] if abs(y + 3.0) < 1e-6}
        self.assertIn(20.5, rungs)
        self.assertIn(31.3, rungs)

    def test_rungs_round_a_bend_never_cross(self):
        """On the inside of a sharp bend the cross-sections converge, and two
        that cross put a vertex in the middle of the carriageway."""
        import shapely

        from src.surfaces import CLASS_ROAD, RoadPart

        bend = shapely.union_all([
            shapely.Polygon(_rect(0, -3, 33, 3)), shapely.Polygon(_rect(27, -3, 33, 40)),
        ])
        ring = np.asarray(bend.exterior.coords)
        parts = [RoadPart([ring], CLASS_ROAD, 0)]
        mesh = _ladder(parts, [self._line([[-5, 0], [30, 0], [30, 45]], 1, 1, 2, "Bocht")])
        self.assertLess(_off_the_kerbs(mesh, parts), 1e-6, "a vertex inside the carriageway")

    def test_rungs_of_a_narrow_fork_never_cross(self):
        """Where a road forks at a narrow angle the two branches share one
        carriageway for a while, and each branch's rungs run across both.
        Kept together, they cross in the middle of the road."""
        import shapely

        from src.surfaces import CLASS_ROAD, RoadPart

        a = np.radians(15.0)
        stem, left, right = [[-30, 0], [0, 0]], [[0, 0], [50 * np.cos(a), 50 * np.sin(a)]], \
            [[0, 0], [50 * np.cos(a), -50 * np.sin(a)]]
        road = shapely.union_all([
            shapely.LineString(c).buffer(3.0, cap_style="flat") for c in (stem, left, right)
        ]).intersection(shapely.box(-25, -40, 45, 40))
        parts = [RoadPart([np.asarray(road.exterior.coords)], CLASS_ROAD, 0)]
        mesh = _ladder(parts, [
            self._line([[-35, 0], [0, 0]], 1, 9, 1, "Stam"),
            self._line([[0, 0], [60 * np.cos(a), 60 * np.sin(a)]], 2, 1, 7, "Links"),
            self._line([[0, 0], [60 * np.cos(a), -60 * np.sin(a)]], 3, 1, 8, "Rechts"),
        ])
        self.assertLess(_off_the_kerbs(mesh, parts), 1e-6, "a vertex inside the carriageway")

    def test_a_lamp_post_under_a_bridge_stands_on_the_quay(self):
        from src.geo import BBox
        from src.nwb import Network
        from src.roads import build_road_mesh, road_top
        from src.surfaces import CLASS_ROAD, RoadPart

        mesh = build_road_mesh(
            [RoadPart([_rect(0, 0, 20, 10)], CLASS_ROAD, 1),
             RoadPart([_rect(0, 20, 20, 30)], CLASS_ROAD, 0)],
            Network(), ground=lambda x, y: np.zeros(np.shape(x)),
            deck_sampler=lambda x, y: np.full(np.shape(x), 8.0),
        )
        top = road_top(mesh, BBox(-5, -5, 25, 35))
        under, beside = top(np.array([10.0, 10.0]), np.array([5.0, 25.0]))
        self.assertTrue(np.isnan(under), "the deck counted as the ground under it")
        self.assertAlmostEqual(float(beside), 0.03, places=3)


class TestRoadHeights(unittest.TestCase):
    """Roads on bridges ride their deck; paths over water stay above it."""

    @staticmethod
    def _deck_part(level):
        from src.surfaces import CLASS_ROAD, RoadPart

        return RoadPart([_rect(0, 0, 20, 10)], CLASS_ROAD, level)

    def _build(self, part, ground, deck):
        from src.nwb import Network
        from src.roads import build_road_mesh

        return build_road_mesh([part], Network(), ground=ground, lift_m=0.06, deck_sampler=deck)

    def test_a_road_on_a_bridge_is_lifted_to_its_deck(self):
        flat = lambda x, y: np.zeros(np.shape(x))  # noqa: E731
        deck = lambda x, y: np.full(np.shape(x), 12.0)  # noqa: E731
        mesh = self._build(self._deck_part(1), flat, deck)
        top = mesh.vertices[np.concatenate([mesh.face(i) for i in range(len(mesh)) if mesh.face_kind[i] == 0])]
        np.testing.assert_allclose(top[:, 2], 12.06, atol=1e-6)
        self.assertTrue((mesh.face_level[mesh.face_kind == 0] > 0).all())

    def test_a_reading_far_too_high_is_not_a_deck(self):
        flat = lambda x, y: np.zeros(np.shape(x))  # noqa: E731
        tower = lambda x, y: np.full(np.shape(x), 95.0)  # noqa: E731
        mesh = self._build(self._deck_part(1), flat, tower)
        self.assertLess(float(mesh.vertices[:, 2].max()), 1.0)
        # And it is said to be at grade, because that is where it is: the
        # terrain is pushed under it like under any other road.
        self.assertTrue((mesh.face_level == 0).all())

    def test_an_unmeasurable_bridge_road_follows_the_ground(self):
        slope = lambda x, y: 0.2 * np.asarray(x, dtype=float)  # noqa: E731
        blank = lambda x, y: np.full(np.shape(x), np.nan)  # noqa: E731
        mesh = self._build(self._deck_part(1), slope, blank)
        top = np.concatenate([mesh.face(i) for i in range(len(mesh)) if mesh.face_kind[i] == 0])
        v = mesh.vertices[top]
        np.testing.assert_allclose(v[:, 2], slope(v[:, 0], v[:, 1]) + 0.06, atol=1e-9)

    def test_a_path_over_a_ditch_stays_above_the_water(self):
        from src.nwb import Network
        from src.roads import build_road_mesh
        from src.surfaces import CLASS_FOOTPATH, RoadPart, WaterBody

        ditch = WaterBody(rings=[_rect(40, -50, 60, 150)], level_nap=4.0,
                          area_m2=4000.0, measured=True)

        def dips(x, y):
            across = np.clip((np.asarray(x, dtype=float) - 40.0) / 20.0, 0.0, 1.0)
            return 6.0 - 3.4 * np.sin(np.pi * across)

        path = RoadPart([_rect(0, 45, 100, 48)], CLASS_FOOTPATH, 0)
        mesh = build_road_mesh(
            [path], Network(), ground=dips, lift_m=0.0, kerb_m=0.0,
            water=[ditch], water_clearance_m=0.25,
        )
        from src.roads import SURFACE

        top = np.unique(np.concatenate(
            [mesh.face(i) for i in range(len(mesh)) if mesh.face_kind[i] == SURFACE]
        ))
        v = mesh.vertices[top]
        over = (v[:, 0] > 40.5) & (v[:, 0] < 59.5)
        self.assertTrue(over.any(), "no vertex over the ditch to test")
        self.assertTrue((v[over, 2] >= 4.25 - 1e-9).all())


class TestRoadRastersAndTerrain(unittest.TestCase):
    """The terrain gets out of the road's way, and stays out of it."""

    def test_after_lowering_no_ground_shows_through(self):
        from src.elevation import TerrainResult, lower_under
        from src.geo import BBox
        from src.roads import road_top, road_underside
        from src.validate import CheckReport, check_roads
        from src.surfaces import SurfaceSet

        bbox = BBox(-10.0, -20.0, 70.0, 60.0)
        axis_x = np.linspace(bbox.xmin, bbox.xmax, 41)
        axis_y = np.linspace(bbox.ymin, bbox.ymax, 41)
        gx, gy = np.meshgrid(axis_x, axis_y)
        # Lumpy ground, so the road between its vertices is not the ground.
        heights = (2.0 + 0.25 * np.sin(gx / 3.0) * np.cos(gy / 4.0)).astype(np.float32)
        terrain = TerrainResult(
            heights=heights.copy(), xs=axis_x, ys=axis_y, bbox=bbox,
            center_z_nap=2.0, nodata_fraction_raw=0.0, filled_fraction=0.0,
            geotiff_path=None, coverage_id="t", resolution_m=0.5,
            measured=heights.copy(),
        )
        _, mesh = _build_street(ground=terrain.sample, lift_m=0.03, kerb_m=0.10)
        surfaces = SurfaceSet()
        surfaces.road_mesh = mesh
        surfaces.road_lift_m = 0.03
        tris, owner = mesh.triangles()
        surfaces.road_tris = tris
        surfaces.road_tri_class = mesh.face_class[owner]
        surfaces.road_tri_level = mesh.face_level[owner]

        report = CheckReport()
        check_roads(report, surfaces, terrain)
        before = {c.name: c for c in report.checks}["terrain_stays_under_roads"]
        self.assertFalse(before.passed, "the fixture's ground never rose above the road")

        lower_under(terrain, road_underside(mesh, bbox), clearance_m=0.05)
        terrain.road_top = road_top(mesh, bbox)
        report = CheckReport()
        check_roads(report, surfaces, terrain)
        verdicts = {c.name: c for c in report.checks}
        self.assertTrue(verdicts["terrain_stays_under_roads"].passed,
                        verdicts["terrain_stays_under_roads"].detail)
        self.assertTrue(verdicts["road_surface_follows_terrain"].passed,
                        verdicts["road_surface_follows_terrain"].detail)
        # What stands on the road stands on the road.
        on_footway = terrain.surface(np.array([10.0]), np.array([4.0]))
        self.assertAlmostEqual(float(on_footway[0]), float(terrain.sample(10.0, 4.0)) + 0.13, delta=0.05)


class TestBigFacesFollowTheGround(unittest.TestCase):
    """A car park has no street to cut it across, and is not flat."""

    def test_a_car_park_is_cut_into_a_grid_that_follows_a_mound(self):
        from src.nwb import Network
        from src.roads import GRID_M, SURFACE, build_road_mesh
        from src.surfaces import CLASS_PARKING, RoadPart

        def mound(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            return 1.0 * np.exp(-((x - 20) ** 2 + (y - 20) ** 2) / 120.0)

        park = RoadPart([_rect(0, 0, 40, 40)], CLASS_PARKING, 0)
        mesh = build_road_mesh([park], Network(), ground=mound, lift_m=0.0, kerb_m=0.0)
        faces = [i for i in range(len(mesh)) if mesh.face_kind[i] == SURFACE]
        self.assertGreater(len(faces), 50, "the car park was left as one face")
        tris, owner = mesh.triangles()
        u = tris[:, 1, :2] - tris[:, 0, :2]
        w = tris[:, 2, :2] - tris[:, 0, :2]
        area = np.zeros(len(mesh))
        np.add.at(area, owner, 0.5 * np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]))
        self.assertLessEqual(float(area.max()), (GRID_M + 1.0) ** 2)
        # The middle of the mound is a vertex, not a flat face over it.
        self.assertGreater(float(mesh.vertices[:, 2].max()), 0.8)
        # And it is a grid: rows and columns, mostly quads.
        self.assertGreater(float((mesh.face_sizes[faces] == 4).mean()), 0.7)


class TestMedialAxis(unittest.TestCase):
    def test_an_l_shaped_path_gets_one_centreline_without_spurs(self):
        import shapely

        from src.roads import skeleton_centrelines

        path = shapely.union_all([
            shapely.Polygon(_rect(0, 0, 40, 3)), shapely.Polygon(_rect(37, 0, 40, 30)),
        ])
        lines = skeleton_centrelines(path)
        # One line, end to end: the corners throw no spurs of their own.
        self.assertEqual(len(lines), 1, f"{len(lines)} pieces, so spurs survived")
        ends = sorted(tuple(np.round(p, 0)) for p in (lines[0].points[0], lines[0].points[-1]))
        self.assertLess(abs(ends[0][1] - 1.5), 1.6)
        self.assertGreater(ends[1][1], 25.0)
        total = sum(float(np.linalg.norm(np.diff(l.points, axis=0), axis=1).sum()) for l in lines)
        # About the length down the middle, 38.5 + 28.5, and no corner spurs.
        self.assertGreater(total, 55.0)
        self.assertLess(total, 75.0)
        for line in lines:
            self.assertEqual(line.kind, "skeleton")
            inside = shapely.contains(path, shapely.points(line.points))
            self.assertTrue(inside.all())

    def test_a_square_is_not_a_street(self):
        import shapely

        from src.roads import skeleton_centrelines

        self.assertEqual(skeleton_centrelines(shapely.Polygon(_rect(0, 0, 60, 60))), [])


class TestRailways(unittest.TestCase):
    """Track built sideways out of a centreline, the first linear dataset."""

    @staticmethod
    def _bbox():
        from src.geo import BBox

        return BBox(0.0, 0.0, 100.0, 100.0)

    def test_a_track_running_out_of_the_area_is_trimmed(self):
        """One line out of Utrecht CS stretched the model to 1592 m."""
        from src.rails import clip_line_to_bbox

        line = np.array([[-500.0, 50.0], [600.0, 50.0]])
        pieces = clip_line_to_bbox(line, self._bbox())

        self.assertEqual(len(pieces), 1)
        np.testing.assert_allclose(pieces[0], [[0.0, 50.0], [100.0, 50.0]])

    def test_a_track_that_leaves_and_returns_comes_back_in_pieces(self):
        from src.rails import clip_line_to_bbox

        line = np.array(
            [[10.0, 50.0], [50.0, 150.0], [90.0, 50.0]]
        )
        pieces = clip_line_to_bbox(line, self._bbox())
        self.assertEqual(len(pieces), 2)
        for piece in pieces:
            self.assertTrue((piece[:, 1] <= 100.0 + 1e-9).all())

    def test_a_track_wholly_inside_is_untouched(self):
        from src.rails import clip_line_to_bbox

        line = np.array([[10.0, 10.0], [40.0, 40.0], [80.0, 20.0]])
        pieces = clip_line_to_bbox(line, self._bbox())
        self.assertEqual(len(pieces), 1)
        np.testing.assert_allclose(pieces[0], line)

    def test_a_track_wholly_outside_disappears(self):
        from src.rails import clip_line_to_bbox

        line = np.array([[200.0, 200.0], [300.0, 300.0]])
        self.assertEqual(clip_line_to_bbox(line, self._bbox()), [])

    def test_densify_keeps_the_original_vertices(self):
        """Resampling at a fixed interval would round off the curves."""
        from src.rails import densify

        line = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 8.0]])
        dense = densify(line, max_step=4.0)

        for vertex in line:
            self.assertTrue(
                np.isclose(dense, vertex).all(axis=1).any(),
                f"{vertex} was dropped",
            )
        steps = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        self.assertLessEqual(steps.max(), 4.0 + 1e-9)

    def test_densify_leaves_a_short_line_alone(self):
        from src.rails import densify

        line = np.array([[0.0, 0.0], [2.0, 0.0]])
        np.testing.assert_allclose(densify(line, max_step=4.0), line)

    def test_the_ribbon_edge_is_continuous_round_a_bend(self):
        """Per-segment normals would leave a notch on every curve."""
        from src.rails import offset_normals

        # A right-angle bend, the worst case rails ever present.
        line = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        normals = offset_normals(line)

        # The corner normal bisects the turn and is lengthened to reach it.
        corner = normals[1]
        self.assertAlmostEqual(
            abs(corner[0]), abs(corner[1]), places=6, msg="not bisecting the turn"
        )
        self.assertGreater(np.linalg.norm(corner), 1.0, "no mitre widening")

    def test_the_mitre_is_capped_at_a_hairpin(self):
        from src.rails import MAX_MITER, offset_normals

        line = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.001]])
        normals = offset_normals(line)
        self.assertLessEqual(np.linalg.norm(normals[1]), MAX_MITER + 1e-6)

    def test_a_railway_gets_ballast_and_a_tram_does_not(self):
        """A gravel bed down a shopping street is worse than nothing."""
        from src.rails import (
            KIND_TRAIN, KIND_TRAM, RailLine, RailSet, build_track_geometry,
        )

        line = np.array([[0.0, 0.0], [40.0, 0.0]])
        flat = lambda x, y: np.zeros_like(np.asarray(x, dtype=float))  # noqa: E731

        train = RailSet(lines=[RailLine(line, KIND_TRAIN, 0)])
        build_track_geometry(train, flat)
        self.assertGreater(len(train.ballast_tris), 0)
        self.assertGreater(len(train.rail_tris), 0)

        tram = RailSet(lines=[RailLine(line, KIND_TRAM, 0)])
        build_track_geometry(tram, flat)
        self.assertEqual(len(tram.ballast_tris), 0)
        self.assertGreater(len(tram.rail_tris), 0)

    def test_the_rails_are_a_standard_gauge_apart(self):
        from src.rails import (
            KIND_TRAIN, PROFILES, RailLine, RailSet, build_track_geometry,
        )

        # Running due east, so the rails separate in Y.
        line = np.array([[0.0, 0.0], [40.0, 0.0]])
        flat = lambda x, y: np.zeros_like(np.asarray(x, dtype=float))  # noqa: E731
        rails = RailSet(lines=[RailLine(line, KIND_TRAIN, 0)])
        build_track_geometry(rails, flat)

        ys = rails.rail_tris.reshape(-1, 3)[:, 1]
        gauge = PROFILES[KIND_TRAIN].gauge
        # Two rails centred on ±gauge/2, each a few centimetres wide.
        self.assertAlmostEqual(ys.max() + ys.min(), 0.0, places=6)
        self.assertAlmostEqual(
            ys.max(), gauge / 2 + PROFILES[KIND_TRAIN].rail_half_width, places=6
        )

    def test_the_rail_head_sits_above_the_ballast(self):
        from src.rails import KIND_TRAIN, RailLine, RailSet, build_track_geometry

        line = np.array([[0.0, 0.0], [40.0, 0.0]])
        flat = lambda x, y: np.full_like(np.asarray(x, dtype=float), 5.0)  # noqa: E731
        rails = RailSet(lines=[RailLine(line, KIND_TRAIN, 0)])
        build_track_geometry(rails, flat)

        self.assertGreater(
            rails.rail_tris[:, :, 2].min(), rails.ballast_tris[:, :, 2].max()
        )

    def test_the_track_follows_a_slope(self):
        from src.rails import KIND_TRAIN, RailLine, RailSet, build_track_geometry

        line = np.array([[0.0, 0.0], [100.0, 0.0]])
        ramp = lambda x, y: 0.05 * np.asarray(x, dtype=float)  # noqa: E731
        rails = RailSet(lines=[RailLine(line, KIND_TRAIN, 0)])
        build_track_geometry(rails, ramp, step_m=4.0)

        z = rails.ballast_tris[:, :, 2]
        self.assertGreater(z.max() - z.min(), 4.0)

    def test_the_uv_runs_in_metres_so_sleepers_stay_evenly_spaced(self):
        from src.rails import KIND_TRAIN, RailLine, RailSet, build_track_geometry

        line = np.array([[0.0, 0.0], [40.0, 0.0]])
        flat = lambda x, y: np.zeros_like(np.asarray(x, dtype=float))  # noqa: E731
        rails = RailSet(lines=[RailLine(line, KIND_TRAIN, 0)])
        build_track_geometry(rails, flat)

        # V is distance along the track, so it reaches the track's length.
        self.assertAlmostEqual(rails.ballast_uv[:, 1].max(), 40.0, places=6)
        self.assertAlmostEqual(rails.ballast_uv[:, 0].min(), 0.0, places=6)
        self.assertAlmostEqual(rails.ballast_uv[:, 0].max(), 1.0, places=6)

    def test_only_rail_functions_are_kept(self):
        from src.rails import RAIL_FUNCTIONS

        self.assertIn("trein", RAIL_FUNCTIONS)
        self.assertIn("tram", RAIL_FUNCTIONS)
        self.assertIn("sneltram", RAIL_FUNCTIONS)
        # Harbour crane rails come through as niet-bgt and are not track.
        self.assertNotIn("niet-bgt", RAIL_FUNCTIONS)


class TestVehiclePlacement(unittest.TestCase):
    """Cars and boats are read off the structures that exist because of them."""

    @staticmethod
    def _rect(cx, cy, length, width, angle=0.0):
        half_l, half_w = length / 2, width / 2
        corners = np.array(
            [[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]]
        )
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return corners @ rotation.T + np.array([cx, cy])

    def test_the_long_axis_of_a_rotated_bay_is_found(self):
        from src.vehicles import principal_axes

        ring = self._rect(0, 0, 20.0, 2.5, angle=np.radians(30))
        long_axis, _, along, across = principal_axes(ring)

        self.assertAlmostEqual(along, 20.0, places=6)
        self.assertAlmostEqual(across, 2.5, places=6)
        self.assertAlmostEqual(
            abs(np.degrees(np.arctan2(long_axis[1], long_axis[0]))) % 180, 30.0, places=4
        )

    def test_point_in_polygon(self):
        from src.vehicles import points_in_ring

        ring = self._rect(0, 0, 10.0, 4.0)
        points = np.array([[0.0, 0.0], [4.9, 1.9], [5.1, 0.0], [0.0, 2.1]])
        np.testing.assert_array_equal(
            points_in_ring(points, ring), [True, True, False, False]
        )

    def test_a_narrow_bay_parks_cars_nose_to_tail(self):
        from src.vehicles import cars_in_bay

        # 22 m of kerbside bay, 2.4 m deep, running east-west.
        ring = self._rect(0, 0, 22.0, 2.4)
        placed = cars_in_bay(ring, np.random.default_rng(1), occupancy=1.0)

        self.assertGreaterEqual(len(placed), 3)
        for _, _, heading in placed:
            # Aligned with the bay, so along +x.
            self.assertAlmostEqual(np.sin(heading), 0.0, places=1)

    def test_a_deep_bay_parks_cars_nose_in(self):
        from src.vehicles import cars_in_bay

        # 22 m long, 5 m deep: cars face across it, not along it.
        ring = self._rect(0, 0, 22.0, 5.0)
        placed = cars_in_bay(ring, np.random.default_rng(2), occupancy=1.0)

        self.assertGreaterEqual(len(placed), 6)
        for _, _, heading in placed:
            self.assertAlmostEqual(abs(np.cos(heading)), 0.0, places=1)

    def test_no_car_is_placed_outside_the_bay(self):
        from src.vehicles import cars_in_bay, points_in_ring

        ring = self._rect(5.0, -3.0, 18.0, 2.6, angle=np.radians(65))
        placed = cars_in_bay(ring, np.random.default_rng(3), occupancy=1.0)

        self.assertTrue(placed)
        # Jitter can nudge a centre just over the line; a car length cannot.
        centres = np.array([[x, y] for x, y, _ in placed])
        distances = np.linalg.norm(centres - ring.mean(axis=0), axis=1)
        self.assertLess(distances.max(), 9.0)

    def test_a_bay_too_small_for_a_car_gets_none(self):
        from src.vehicles import cars_in_bay

        self.assertEqual(
            cars_in_bay(self._rect(0, 0, 2.0, 2.0), np.random.default_rng(4),
                        occupancy=1.0),
            [],
        )

    def test_occupancy_leaves_gaps(self):
        from src.vehicles import cars_in_bay

        ring = self._rect(0, 0, 200.0, 2.4)
        full = cars_in_bay(ring, np.random.default_rng(5), occupancy=1.0)
        half = cars_in_bay(ring, np.random.default_rng(5), occupancy=0.5)
        self.assertGreater(len(full), len(half))

    def test_a_chain_of_posts_gives_one_boat_per_gap(self):
        """Not one per pair: that would lay a third boat over the other two."""
        from src.vehicles import _nearest_links

        posts = np.array([[0.0, 0.0], [7.0, 0.0], [14.0, 0.0]])
        self.assertEqual(_nearest_links(posts, 3.5, 18.0), [(0, 1), (1, 2)])

    def test_posts_too_far_apart_are_not_a_mooring(self):
        from src.vehicles import _nearest_links

        posts = np.array([[0.0, 0.0], [90.0, 0.0]])
        self.assertEqual(_nearest_links(posts, 3.5, 18.0), [])

    def test_a_boat_goes_on_the_water_side_of_its_posts(self):
        from src.vehicles import boats_between_posts

        # Canal occupying y in [0, 12]; posts along its southern edge.
        canal = np.array([[-50.0, 0.0], [50.0, 0.0], [50.0, 12.0], [-50.0, 12.0]])
        posts = np.array([[0.0, 0.2], [8.0, 0.2]])

        placed = boats_between_posts(
            posts, [canal], np.random.default_rng(6), occupancy=1.0
        )
        self.assertEqual(len(placed), 1)
        x, y, heading, length, beam = placed[0]
        self.assertGreater(y, 0.0, "the boat was put on the bank, not the water")
        self.assertAlmostEqual(x, 4.0, places=1)
        self.assertAlmostEqual(np.sin(heading), 0.0, places=6)
        self.assertAlmostEqual(length, 8.0 * 0.85, places=6)

    def test_a_boat_floats_clear_of_the_bank(self):
        """Testing only the centre left the inboard gunwale up on the quay."""
        from src.vehicles import _float_clear_of_bank, points_in_ring

        canal = np.array([[-50.0, 0.0], [50.0, 0.0], [50.0, 12.0], [-50.0, 12.0]])
        beam = 3.0
        centre = _float_clear_of_bank(
            np.array([0.0, 0.0]), np.array([0.0, 1.0]), beam, [canal]
        )
        self.assertIsNotNone(centre)
        inboard = centre - np.array([0.0, beam * 0.5])
        outboard = centre + np.array([0.0, beam * 0.5])
        self.assertTrue(points_in_ring(inboard[None, :], canal)[0])
        self.assertTrue(points_in_ring(outboard[None, :], canal)[0])

    def test_posts_with_no_water_get_no_boat(self):
        from src.vehicles import boats_between_posts

        posts = np.array([[0.0, 0.0], [8.0, 0.0]])
        self.assertEqual(
            boats_between_posts(posts, [], np.random.default_rng(7), occupancy=1.0),
            [],
        )

    def test_the_spawn_list_heading_is_a_unity_rotation(self):
        """Held anticlockwise from east; Unity wants clockwise from north."""
        import json

        from src.geo import BBox, GeoContext
        from src.vehicles import KIND_CAR, VehicleSet, write_spawn_list

        vehicles = VehicleSet(
            xy=np.array([[136500.0, 455500.0]]),
            z_nap=np.array([2.0]),
            # Pointing north: 90 degrees anticlockwise from east.
            heading=np.array([np.pi / 2]),
            length=np.array([4.3]),
            width=np.array([1.8]),
            kind=np.array([KIND_CAR], dtype=np.int32),
            colour=np.array([0], dtype=np.int32),
        )
        geo = GeoContext.from_bbox(BBox(136000, 455000, 137000, 456000))

        with tempfile.TemporaryDirectory() as tmp:
            path = write_spawn_list(vehicles, geo, Path(tmp))
            payload = json.loads(path.read_text(encoding="utf-8"))

        record = payload["vehicles"][0]
        self.assertEqual(record["kind"], "car")
        self.assertAlmostEqual(record["heading_deg"], 0.0, places=3)


class TestCentringCheck(unittest.TestCase):
    """A lopsided model is not an off-origin model."""

    @staticmethod
    def _centre_of(overhang_near: float, overhang_far: float, bbox: float = 1000.0):
        """The model bounds a given overhang produces, as (centre, span)."""
        low = -bbox / 2 - overhang_near
        high = bbox / 2 + overhang_far
        return 0.5 * (low + high), high - low

    def test_a_symmetric_overhang_is_centred(self):
        from src.validate import _centring_tolerance

        centre, span = self._centre_of(40.0, 40.0)
        self.assertAlmostEqual(centre, 0.0)
        self.assertLessEqual(abs(centre), _centring_tolerance(span, 1000.0))

    def test_all_the_overhang_on_one_side_still_passes(self):
        """One long warehouse near an edge is enough to do this."""
        from src.validate import _centring_tolerance

        centre, span = self._centre_of(80.0, 0.0)
        self.assertAlmostEqual(centre, -40.0)
        self.assertLessEqual(abs(centre), _centring_tolerance(span, 1000.0))

    def test_the_reported_failure_would_now_pass(self):
        """31 m off centre, which the old fixed 30 m limit rejected."""
        from src.validate import _centring_tolerance

        centre, span = self._centre_of(70.0, 7.8)
        self.assertAlmostEqual(centre, -31.1)
        self.assertLessEqual(abs(centre), _centring_tolerance(span, 1000.0))

    def test_a_wrong_origin_is_still_caught(self):
        """It shifts the model without changing its span, so it cannot hide."""
        from src.validate import _centring_tolerance

        # Terrain displaced 200 m east: same span, badly off centre.
        span = 1000.0
        self.assertGreater(200.0, _centring_tolerance(span, 1000.0))

    def test_a_small_offset_is_caught_when_there_is_no_overhang_to_spend(self):
        from src.validate import _centring_tolerance

        # Nothing hangs over, so almost nothing is allowed.
        self.assertLess(_centring_tolerance(1000.0, 1000.0), 1.5)

    def test_the_tolerance_never_exceeds_what_the_span_check_allows(self):
        from src.validate import _centring_tolerance

        # model_span caps the overhang at 120 m, so this caps out near 60.
        self.assertLess(_centring_tolerance(1120.0, 1000.0), 62.0)


class TestFindingBlender(unittest.TestCase):
    """Blender is installed in places PATH never mentions."""

    def test_a_macos_bundle_resolves_to_its_executable(self):
        """`--blender /Applications/Blender.app` is the obvious thing to pass
        on a Mac, and it is a directory rather than a program."""
        import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "Blender.app"
            inner = bundle / "Contents" / "MacOS" / "Blender"
            inner.parent.mkdir(parents=True)
            inner.write_text("#!/bin/sh\n")
            self.assertEqual(pipeline.blender_in_bundle(bundle), inner)
            mode, command = pipeline.find_blender(str(bundle))
            self.assertEqual(mode, "executable")
            self.assertEqual(command, [str(inner)])

    def test_a_bundle_without_the_executable_is_not_accepted(self):
        import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "Blender.app"
            bundle.mkdir()
            self.assertIsNone(pipeline.blender_in_bundle(bundle))

    def test_the_search_covers_the_places_installers_use(self):
        import pipeline

        joined = " ".join(pipeline.BLENDER_GLOBS)
        # A macOS .app can never be found by a PATH lookup, so it has to be
        # named explicitly; same for the Windows installer's own directory.
        self.assertIn("/Applications/Blender.app/Contents/MacOS/Blender", joined)
        self.assertIn("Blender*.app", joined)
        self.assertIn("Blender Foundation", joined)

    def test_the_failure_says_where_it_looked(self):
        """The old message claimed no Blender to someone who had just installed
        it, and named neither the bundle path nor how to point at one."""
        import pipeline

        import unittest.mock as mock

        with mock.patch.object(pipeline, "search_for_blender", return_value=[]), \
                mock.patch.object(pipeline, "pip_bpy_version", return_value=None):
            with self.assertRaises(SystemExit) as caught:
                pipeline.find_blender(None)

        message = str(caught.exception)
        self.assertIn("Looked in", message)
        self.assertIn("/Applications/Blender.app", message)
        self.assertIn("CPython 3.13", message)


class TestChoosingBlender(unittest.TestCase):
    """Blender 5 wherever there is one; an older one only with a warning."""

    def _choose(self, installed, module):
        import unittest.mock as mock

        import pipeline

        with mock.patch.object(pipeline, "search_for_blender", return_value=list(installed)), \
                mock.patch.object(pipeline, "blender_version", side_effect=installed.get), \
                mock.patch.object(pipeline, "pip_bpy_version", return_value=module):
            with self.assertLogs("pipeline", level="INFO") as logs:
                mode, command = pipeline.find_blender(None)
        return mode, command, " ".join(logs.output)

    def test_an_installed_blender_5_comes_first(self):
        mode, command, said = self._choose(
            {"/opt/blender-4.2/blender": (4, 2, 3), "/opt/blender-5.2/blender": (5, 2, 3)},
            (5, 2, 3),
        )
        self.assertEqual((mode, command), ("executable", ["/opt/blender-5.2/blender"]))
        self.assertIn("5.2.3", said)

    def test_the_pip_module_wins_over_an_old_install(self):
        """The machine that still has Blender 4.2 in Program Files."""
        import sys

        mode, command, said = self._choose({"C:/Blender 4.2/blender.exe": (4, 2, 0)}, (5, 2, 3))
        self.assertEqual((mode, command), ("bpy", [sys.executable]))

    def test_an_old_blender_is_used_but_named(self):
        mode, command, said = self._choose({"/usr/bin/blender": (4, 2, 0)}, None)
        self.assertEqual(command, ["/usr/bin/blender"])
        self.assertIn("WARNING", said)
        self.assertIn("no kerbs", said)

    def test_the_versions_are_read_without_loading_blender(self):
        import pipeline

        version = pipeline.pip_bpy_version()
        self.assertTrue(version is None or isinstance(version, tuple))
        self.assertNotIn("bpy", sys.modules)


class TestBag3dTiles(unittest.TestCase):
    """The way round an api.3dbag.nl outage."""

    def test_a_tile_id_becomes_its_path(self):
        from src.bag3d_tiles import tile_url

        self.assertEqual(
            tile_url(
                "10/490/596",
                base_url="https://data.3dbag.nl/{version}/tiles",
                version="v20250903",
            ),
            "https://data.3dbag.nl/v20250903/tiles/10/490/596/"
            "10-490-596.city.json.gz",
        )

    def test_a_malformed_tile_id_is_refused(self):
        from src.bag3d_tiles import tile_url

        with self.assertRaises(ValueError):
            tile_url("10/490", base_url="x/{version}", version="v1")

    def test_the_cache_is_shared_between_areas(self):
        """The point of it: one build warms the cache for the next."""
        from src.bag3d_tiles import cache_dir_for

        dom = cache_dir_for(Path("work/utrecht_dom"), "v20250903")
        cs = cache_dir_for(Path("work/utrecht_cs"), "v20250903")
        self.assertEqual(dom, cs)
        self.assertEqual(dom.name, "v20250903")

    def test_versions_do_not_share_a_cache(self):
        from src.bag3d_tiles import cache_dir_for

        self.assertNotEqual(
            cache_dir_for(Path("work/a"), "v20250903"),
            cache_dir_for(Path("work/a"), "v20241216"),
        )

    def test_buildings_outside_the_bbox_are_skipped_on_their_extent(self):
        """A tile covers far more than the bbox; parsing it all is the cost."""
        from src.bag3d_tiles import _extent_misses
        from src.geo import BBox

        bbox = BBox(136000, 455000, 137000, 456000)
        inside = [136400, 455400, 0.0, 136420, 455430, 12.0]
        overlapping = [136990, 455400, 0.0, 137400, 455430, 12.0]
        east = [137200, 455400, 0.0, 137300, 455430, 12.0]
        south = [136400, 454000, 0.0, 136420, 454500, 12.0]
        self.assertFalse(_extent_misses(inside, bbox))
        self.assertFalse(_extent_misses(overlapping, bbox))
        self.assertTrue(_extent_misses(east, bbox))
        self.assertTrue(_extent_misses(south, bbox))
        # No extent means it cannot be ruled out, so it has to be parsed.
        self.assertFalse(_extent_misses(None, bbox))

    def test_an_unknown_source_name_is_refused(self):
        from src.buildings import fetch_buildings
        from src.geo import BBox

        with self.assertRaises(ValueError) as caught:
            fetch_buildings(
                BBox(136000, 455000, 137000, 456000),
                buildings_cfg={"lod": "2.2", "sources": ["api", "smoke-signal"]},
            )
        self.assertIn("smoke-signal", str(caught.exception))

    def test_the_default_order_tries_the_api_then_the_tiles(self):
        from src.config import DEFAULTS

        self.assertEqual(DEFAULTS["buildings"]["sources"], ["api", "tiles"])

    def test_the_shared_vertex_array_is_used_as_given(self):
        """Decoding a tile's vertices once per building costs more than the
        download, so _extract_feature has to accept them pre-decoded."""
        import inspect

        from src.buildings import _extract_feature

        self.assertIn("vertices", inspect.signature(_extract_feature).parameters)


class TestDependencies(unittest.TestCase):
    """A package added to requirements.txt reaches environments set up before it."""

    def test_the_checked_packages_are_the_required_ones(self):
        import re

        from src.dependencies import REQUIRED

        listed = {}
        for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ";" in line:  # bpy carries a marker and is optional
                continue
            name, minimum = re.match(r"([A-Za-z0-9_.-]+)\s*>=\s*([0-9.]+)", line).groups()
            listed[name.lower()] = minimum
        checked = {package.lower(): minimum for _, package, minimum in REQUIRED}
        self.assertEqual(checked, listed)

    def test_a_missing_or_old_package_is_named(self):
        import importlib.metadata
        import importlib.util
        from unittest import mock

        from src import dependencies

        self.assertEqual(dependencies.missing_dependencies(), [])
        real_find, real_version = importlib.util.find_spec, importlib.metadata.version
        with mock.patch.object(
            importlib.util, "find_spec",
            lambda name, *a: None if name == "shapely" else real_find(name, *a),
        ), mock.patch.object(
            importlib.metadata, "version",
            lambda name: "1.23.5" if name == "numpy" else real_version(name),
        ):
            problems = dependencies.missing_dependencies()
        self.assertEqual(problems, [
            "numpy 1.23.5 is too old; 1.24 or newer is needed",
            "shapely is not installed",
        ])

    def test_the_pipeline_stops_at_once_and_says_how_to_fix_it(self):
        import subprocess

        hide = (
            "import importlib.util as u, runpy, sys\n"
            "real = u.find_spec\n"
            "u.find_spec = lambda n, *a: None if n == 'shapely' else real(n, *a)\n"
            "sys.argv = ['pipeline.py', '--config', 'config.json']\n"
            "runpy.run_path('pipeline.py', run_name='__main__')\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", hide], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("shapely is not installed", done.stderr)
        self.assertIn("-m pip install -r", done.stderr)

    def test_the_windows_launcher_can_install_python_itself(self):
        """Python 3.13 is the step colleagues get stuck on, so the launcher
        offers to install it: per user, PATH untouched, and only a file whose
        hash matches the one python.org published."""
        raw = (REPO_ROOT / "Start Map Pipeline.bat").read_bytes()
        # cmd.exe reads a batch file in blocks and can miss a label in an
        # LF-only file, so every line ends CRLF.
        self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"))
        text = raw.decode("ascii")
        labels = set(re.findall(r"(?im)^:([A-Za-z_]+)\s*$", text))
        targets = set(re.findall(r"(?i)\b(?:goto|call)\s+:([A-Za-z_]+)", text)) - {"eof"}
        self.assertEqual(targets - labels, set(), "a goto or call with no label")
        self.assertIn("choice /c YN", text)
        for flag in ("InstallAllUsers=0", "PrependPath=0", "Include_launcher=0"):
            self.assertIn(flag, text)
        digest = re.search(r'set "PY_SHA256=([0-9a-f]{64})"', text)
        self.assertIsNotNone(digest, "the installer is not pinned to a hash")
        self.assertIn("Get-FileHash", text)
        # The hash check sits between the download and the install.
        self.assertLess(text.index("Get-FileHash"), text.index("start \"\" /wait \"%PY_SETUP%\""))
        self.assertIn(r"%LOCALAPPDATA%\Programs\Python\Python313\python.exe", text)
        self.assertIn('set "PY_VERSION=3.13.', text)

    def test_the_launchers_all_ask_for_the_python_blender_5_needs(self):
        """bpy 5.1 and later is published for CPython 3.13 alone. A launcher
        still asking for 3.11 builds an environment pip cannot put Blender 5
        in, and one left from before keeps its old Blender: each rebuilds an
        environment on another Python."""
        requirement = next(
            line for line in (REPO_ROOT / "requirements.txt").read_text().splitlines()
            if line.startswith("bpy")
        )
        self.assertIn(">=5.2", requirement)
        self.assertIn('python_version >= "3.13" and python_version < "3.14"', requirement)
        for launcher in ("Start Map Pipeline.bat", "start-map-pipeline.sh",
                         "Start Map Pipeline.command"):
            text = (REPO_ROOT / launcher).read_text()
            with self.subTest(launcher=launcher):
                self.assertIn("sys.version_info[:2]==(3,13)", text)
                self.assertNotIn("3.11", text)
                self.assertNotIn("(3,11)", text)
                self.assertIn("Rebuilding the private Python environment", text)

    def test_the_launchers_reinstall_when_requirements_change(self):
        """They used to install once and stamp it done, so an update that
        added shapely never reached a team member's existing environment."""
        for launcher, compare in (
            ("Start Map Pipeline.bat", 'fc /b requirements.txt "%STAMP%"'),
            ("start-map-pipeline.sh", 'cmp -s requirements.txt "$STAMP"'),
            ("Start Map Pipeline.command", 'cmp -s requirements.txt "$STAMP"'),
        ):
            text = (REPO_ROOT / launcher).read_text()
            self.assertIn(compare, text, launcher)
            self.assertIn("requirements.txt", text.split(compare, 1)[1].split("pip install -r", 1)[1])
            self.assertNotIn("echo installed", text, launcher)


class TestConfig(unittest.TestCase):
    def test_shipped_config_loads(self):
        config = load_config(REPO_ROOT / "config.json")
        self.assertEqual(config.bbox.width, 1000.0)
        self.assertEqual(config.geo.origin, (136500.0, 455500.0))
        self.assertEqual(config.warnings, [])
        self.assertEqual(config.terrain["grid_m"], 2.0)

    def test_an_old_config_still_loads_and_says_what_changed(self):
        """A config written for the adaptive terrain and the old road mesher
        loads, and every setting that now does nothing is named, with what
        took its place, rather than being ignored without a word."""
        old = {
            "name": "old",
            "bbox": {"crs": "EPSG:28992", "xmin": 136000, "ymin": 455000,
                     "xmax": 137000, "ymax": 456000},
            "terrain": {"mesh_vertices_per_side": 513, "quads": True,
                        "breaklines": ["roads"]},
            "surfaces": {"road_max_edge_m": 8.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(json.dumps(old))
            with self.assertLogs("src.config", level="WARNING"):
                config = load_config(path)
        said = " ".join(config.warnings)
        for key in ("terrain.mesh_vertices_per_side", "terrain.quads",
                    "terrain.breaklines", "surfaces.road_max_edge_m"):
            self.assertIn(key, said)
        self.assertIn("grid_m", said)
        self.assertEqual(config.terrain["grid_m"], 2.0)

    def test_a_detail_level_in_a_config_file_and_an_explicit_setting(self):
        """The level fills in; whatever the file also sets wins."""
        raw = {
            "name": "lvl",
            "bbox": {"crs": "EPSG:28992", "xmin": 136000, "ymin": 455000,
                     "xmax": 137000, "ymax": 456000},
            "detail": "draft",
            "terrain": {"grid_m": 3.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(raw))
            config = load_config(path)
        self.assertEqual(config.terrain["grid_m"], 3.0)
        self.assertEqual(config.aerial["size_px"], 2000)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(dict(raw, detail="extreme")))
            with self.assertRaises(ValueError):
                load_config(path)

    def _load(self, raw: dict):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(raw))
            return load_config(path)

    BBOX = {"crs": "EPSG:28992", "xmin": 136000, "ymin": 455000,
            "xmax": 137000, "ymax": 456000}

    def test_the_default_style_is_the_modellers(self):
        """Layered unless asked: water fetched (for the ditches), no
        street-by-street road meshes, no parked cars, and object names
        prefixed after the area."""
        config = self._load({"name": "hazeldonk_west", "bbox": self.BBOX})
        self.assertEqual(config.style, "layered")
        self.assertTrue(config.surfaces["water"])
        self.assertFalse(config.surfaces["road_geometry"])
        self.assertEqual(config.terrain["feature_edges"], [])
        self.assertFalse(config.vehicles["cars"])
        self.assertEqual(config.layered["prefix"], "Hazeldonk_West")
        self.assertEqual(config.as_dict()["style"], "layered")
        survey = self._load({"name": "x", "bbox": self.BBOX, "style": "Survey"})
        self.assertEqual(survey.style, "survey")
        self.assertTrue(survey.surfaces["water"])
        self.assertTrue(survey.surfaces["road_geometry"])

    def test_a_style_preset_gives_way_except_where_it_cannot(self):
        """Cars can be asked for in the layered style; the survey style's road
        meshes cannot, since the layered ground makes its own roads, and the
        file is told so."""
        raw = {"name": "x", "bbox": self.BBOX, "vehicles": {"cars": True},
               "surfaces": {"road_geometry": True}}
        with self.assertLogs("src.config", level="WARNING") as logs:
            config = self._load(raw)
        self.assertTrue(config.vehicles["cars"])
        self.assertFalse(config.surfaces["road_geometry"])
        self.assertIn("surfaces.road_geometry", " ".join(logs.output))
        self.assertTrue(
            self._load({**raw, "style": "survey"}).surfaces["road_geometry"]
        )

    def test_the_layered_settings_are_checked(self):
        base = {"name": "x", "bbox": self.BBOX}
        with self.assertRaises(ValueError):
            self._load({**base, "style": "handmade"})
        for bad in ({"tolerance_m": 0}, {"focus_m": -1}, {"road_lift_m": -0.1},
                    {"curb": {"height_offset_curb_m": 0.2}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self._load({**base, "layered": bad})
        config = self._load({**base, "layered": {"prefix": " Tankstation ",
                                                 "tolerance_m": "0.25"}})
        self.assertEqual(config.layered["prefix"], "Tankstation")
        self.assertEqual(config.layered["tolerance_m"], 0.25)
        self.assertEqual(config.layered["curb"]["height_m"], 0.15)

    def test_an_area_name_makes_a_clean_prefix(self):
        from src.config import default_prefix

        self.assertEqual(default_prefix("hazeldonk_west"), "Hazeldonk_West")
        self.assertEqual(default_prefix("A12 knooppunt-Gouwe"), "A12_Knooppunt_Gouwe")
        self.assertEqual(default_prefix("café de Ĳssel/2"), "Cafe_De_IJssel2")
        self.assertEqual(default_prefix("__"), "Area")

    def test_defaults_fill_in_missing_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "minimal",
                        "bbox": {
                            "crs": "EPSG:28992",
                            "xmin": 136000,
                            "ymin": 455000,
                            "xmax": 137000,
                            "ymax": 456000,
                        },
                    }
                )
            )
            config = load_config(path)
            self.assertEqual(config.aerial["size_px"], 4096)
            self.assertEqual(config.buildings["lod"], "2.2")
            self.assertEqual(config.terrain["ahn_model"], "DTM")

    def test_bad_ahn_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "bad",
                        "bbox": {
                            "xmin": 136000,
                            "ymin": 455000,
                            "xmax": 137000,
                            "ymax": 456000,
                        },
                        "terrain": {"ahn_model": "LIDAR"},
                    }
                )
            )
            with self.assertRaises(ValueError):
                load_config(path)


class TestBlenderStageIsolation(unittest.TestCase):
    """The Blender scripts may only use what Blender itself bundles.

    When Blender is a real application rather than the pip module, it runs its
    own Python: none of this pipeline's dependencies are importable there. That
    cannot be caught by running the pipeline here, because the pip-bpy path
    shares this interpreter and every import resolves, so it is checked
    statically instead.
    """

    BUNDLED = {
        "bpy", "numpy", "mathutils", "bmesh", "gpu", "aud",
        "addon_utils", "bpy_extras", "bl_ui",
    }

    @staticmethod
    def _imported_modules(path: Path) -> set[str]:
        import ast

        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    modules.add(node.module.split(".")[0])
                elif node.level:
                    modules.add(f"<relative:{node.module}>")
        return modules

    def test_blender_scripts_import_nothing_extra(self):
        allowed = self.BUNDLED | set(sys.stdlib_module_names)
        scripts = sorted((REPO_ROOT / "blender").glob("*.py"))
        self.assertTrue(scripts, "no Blender scripts found")

        for script in scripts:
            extra = self._imported_modules(script) - allowed
            self.assertEqual(
                extra,
                set(),
                f"{script.name} imports {sorted(extra)}, which Blender does not "
                f"bundle. Move that work into src/ and pass the result through "
                f"the intermediate files.",
            )

    def test_blender_scripts_do_not_import_the_pipeline(self):
        """They read intermediate files, so they never need src/."""
        for script in sorted((REPO_ROOT / "blender").glob("*.py")):
            modules = self._imported_modules(script)
            self.assertNotIn("src", modules, f"{script.name} imports src")
            self.assertFalse(
                any(m.startswith("<relative:") for m in modules),
                f"{script.name} uses a relative import",
            )


def _blender_mesh_calls():
    """Run every Blender builder against synthetic input, offline.

    Yields `(function name, calls, error)`. Each call records the shapes handed
    to `build_mesh_object`, so the contract can be checked without Blender.

    The Blender stage is the one part of this pipeline the test suite cannot
    execute: it needs `bpy`, which is 880 MB and one CPython only. So the
    functions are exec'd against a stubbed `build_mesh_object` and stubbed
    materials instead, which is enough to catch the shape mistakes -- and one
    of those shipped: `build_structures` welded its corners and then handed
    Blender loop indices for the unwelded buffer, which killed a real run with
    "Array length mismatch (got 210, expected more)".
    """
    import re
    import tempfile

    source = (REPO_ROOT / "blender" / "process.py").read_text(encoding="utf-8")
    calls: list[dict] = []

    class FakeAttributes:
        """Mesh attributes: created and filled, never read back here."""

        @staticmethod
        def new(name, kind, domain):
            class Attribute:
                class data:  # noqa: N801 - mirrors bpy
                    @staticmethod
                    def foreach_set(key, values):
                        pass
            return Attribute

    class FakeMesh:
        attributes = FakeAttributes

    class Stub(str):
        """What the builders get back: a name they can also set things on."""

        data = FakeMesh

        def __setitem__(self, key, value):
            self.__dict__[key] = value

        def __getitem__(self, key):
            return self.__dict__[key]

    def record(name, vertices, loops, starts, totals, uvs, mats, materials, **kw):
        extra = kw.get("extra_uvs") or {}
        calls.append({
            "object": name,
            "vertices": len(vertices),
            "loops": len(loops),
            "uvs": len(uvs),
            # Every extra UV map is per loop too.
            "extra_uvs": [len(values) for values in extra.values()],
            "polygons": len(starts),
            "totals": len(totals),
            "loop_total_sum": int(np.sum(totals)),
            "material_indices": len(mats),
            "materials": len(materials),
            "max_index": int(np.max(loops)) if len(loops) else -1,
        })
        return Stub(name)

    class FakeObject(dict):
        """Enough of a Blender object for a builder to place and name it."""

        def __init__(self, name, data=None):
            super().__init__()
            self.name = name
            self.data = data
            self.location = (0.0, 0.0, 0.0)
            self.rotation_euler = [0.0, 0.0, 0.0]
            self.scale = (1.0, 1.0, 1.0)
            self.parent = None

    class FakeBpy:
        class data:  # noqa: N801 - mirrors bpy.data
            class objects:  # noqa: N801
                @staticmethod
                def new(name, data):
                    return FakeObject(name, data)

                @staticmethod
                def remove(obj):
                    pass

        class context:  # noqa: N801 - mirrors bpy.context
            class scene:  # noqa: N801
                class collection:  # noqa: N801
                    class objects:  # noqa: N801
                        @staticmethod
                        def link(obj):
                            pass

    namespace = {
        "np": np, "Path": Path, "log": lambda *a: None, "bpy": FakeBpy,
        "build_mesh_object": record,
        "make_material": lambda *a, **k: "M",
        "make_textured_material": lambda *a, **k: "M",
        "make_plain_material": lambda *a, **k: "M",
    }
    # Pull in every helper the builders lean on, in file order.
    for match in re.finditer(r"\ndef (\w+)\(.*?\n(?=\ndef |\n# ---|\Z)", source, re.S):
        # Stubbed above, and the real ones need bpy.
        if match.group(1) in ("build_mesh_object", "make_plain_material"):
            continue
        try:
            exec(match.group(0), namespace)  # noqa: S102 - our own source
        except Exception:  # noqa: BLE001 - helpers needing bpy are not our target
            pass

    rng = np.random.default_rng(11)
    folder = Path(tempfile.mkdtemp())
    n = 12
    xy = np.column_stack([rng.uniform(0, 200, n), rng.uniform(0, 200, n)])
    # A strip sharing corners along its length, so a weld has something to
    # merge. Random triangles share nothing and would let a broken weld pass.
    strip = np.array([[i * 4.0, 0.0, 0.0] for i in range(n // 2 + 1)]
                     + [[i * 4.0, 6.0, 0.0] for i in range(n // 2 + 1)])
    top = n // 2 + 1
    quads = [[i, i + 1, top + i + 1, top + i] for i in range(n // 2)]
    triangles = np.array(
        [[strip[a], strip[b], strip[c]] for a, b, c, d in quads]
        + [[strip[a], strip[c], strip[d]] for a, b, c, d in quads]
    )
    n = len(triangles)

    np.savez(folder / "trees.npz", xy=xy, ground_z_nap=np.zeros(n),
             height_m=rng.uniform(3, 20, n), crown_radius_m=rng.uniform(1, 4, n),
             trunk_height_m=rng.uniform(1, 3, n), detected=np.arange(n) >= 5)
    np.savez(folder / "furniture.npz", xy=xy, ground_z_nap=np.zeros(n),
             kind=np.arange(n) % 14)
    verts = triangles.reshape(-1, 3)
    np.savez(folder / "barriers.npz", vertices=verts,
             triangles=np.arange(len(verts)).reshape(-1, 3).astype(np.int32),
             material=np.zeros(n, np.int32), kind=(np.arange(n) % 3).astype(np.int32),
             material_names=np.array(["brick", "hedge", "metal"], dtype=str),
             kind_names=np.array(["wall", "hedge", "fence"], dtype=str),
             # The fattened physics proxy, in its own buffer: no UVs and no
             # materials, because it is never drawn.
             collision_vertices=verts + np.array([0.0, 0.0, 0.0]),
             collision_triangles=np.arange(len(verts)).reshape(-1, 3).astype(np.int32),
             collision_thickness_m=np.float64(0.30))
    np.savez(folder / "structures.npz", tris=triangles,
             tri_kind=(np.arange(n) % 2).astype(np.int32))
    np.savez(folder / "rails.npz",
             ballast_tris=triangles, ballast_kind=np.zeros(n, np.int32),
             ballast_uv=rng.uniform(0, 1, (n * 3, 2)),
             rail_tris=triangles, rail_kind=np.zeros(n, np.int32),
             rail_uv=rng.uniform(0, 1, (n * 3, 2)))
    np.savez(folder / "vehicles.npz", xy=xy, z_nap=np.zeros(n),
             heading=rng.uniform(0, 6, n), length=np.full(n, 4.2),
             width=np.full(n, 1.8), kind=np.zeros(n, np.int32),
             colour=rng.integers(0, 4, n))
    # A real road network, from the real mesher: a T-junction.
    _, street = _build_street()
    np.savez(folder / "roads.npz", **street.to_npz_fields(), road_lift_m=0.03)

    scene = {
        "origin_rd": [100.0, 100.0],
        "ground_z_offset_nap": 0.0,
        "aerial": {"bbox_local": [-200.0, -200.0, 200.0, 200.0]},
        "trees": {"file": "trees.npz", "texture": "t.png"},
        "furniture": {"file": "furniture.npz", "texture": "f.png",
                      "atlas_uv": {}},
        "barriers": {"file": "barriers.npz", "texture": "f.png"},
        "structures": {"file": "structures.npz", "kind_names": {"0": "bridge"}},
        "rails": {"file": "rails.npz", "kind_names": {"0": "track"}},
        "vehicles": {"file": "vehicles.npz", "texture": "v.png"},
        "surfaces": {"file": "roads.npz",
                     "road_class_names": {"1": "road_asphalt", "8": "footpath",
                                          "12": "kerb"}},
        "export": {"collision": True, "collision_tree_crowns": True},
    }

    for name, args in (
        ("build_trees", (scene, folder, "M")),
        ("build_furniture", (scene, folder, "M")),
        ("build_barriers", (scene, folder, "M")),
        ("build_structures", (scene, folder, "M")),
        ("build_rails", (scene, folder, "M")),
        ("build_vehicles", (scene, folder, "M")),
        ("build_roads", (scene, folder, lambda *a, **k: "M")),
        ("build_collision", (scene, folder)),
    ):
        function = namespace.get(name)
        if function is None:
            yield name, [], "not found in process.py"
            continue
        calls.clear()
        try:
            function(*args)
        except TypeError as exc:
            # A signature this harness does not know about is worth saying so
            # rather than passing quietly.
            yield name, list(calls), f"{type(exc).__name__}: {exc}"
            continue
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            yield name, list(calls), f"{type(exc).__name__}: {exc}"
            continue
        yield name, list(calls), None


class TestTreeShape(unittest.TestCase):
    """A tree used to be a sphere on a prism, which reads as a lollipop.

    None of this is about triangle count for its own sake -- the budget barely
    moved, from 42 to 46 triangles a tree on a real height mix. It is about a
    wooded square kilometre not being fifteen thousand copies of one ball.
    """

    @staticmethod
    def _shapes():
        import re

        source = (REPO_ROOT / "blender" / "process.py").read_text(encoding="utf-8")
        namespace = {"np": np}
        for constant in ("TREE_TRUNK_FRACTIONS", "TREE_SHAPES_PER_KIND"):
            line = re.search(rf"\n{constant} = .*\n", source)
            assert line is not None, f"{constant} moved"
            exec(line.group(0), namespace)  # noqa: S102 - our own source
        for name in ("_octahedron_canopy", "_tree_dice", "_lobed_crown",
                     "_tree_instance", "_shape_lean", "_unit_tree",
                     "_tree_group_mesh"):
            body = re.search(rf"\ndef {name}\(.*?\n(?=\ndef |\n# )", source, re.S)
            assert body is not None, f"{name} moved"
            exec(body.group(0), namespace)  # noqa: S102 - our own source
        return namespace

    @staticmethod
    def _one(namespace, x, y, height):
        fields = (
            np.array([[x, y]]), np.zeros(1), np.array([float(height)]),
            np.array([float(np.clip(0.26 * height, 0.8, 7.0))]),
            np.array([float(np.clip(0.38 * height, 0.6, 6.0))]),
        )
        return namespace["_tree_group_mesh"](
            np.ones(1, dtype=bool), fields, (x, y, 0.0), np.random.default_rng(1)
        )

    def test_a_tree_is_the_same_tree_every_run(self):
        """Shape comes from the position, not from a running generator. Turning
        detection off, or rebuilding one group and not the other, must not
        reshape the trees that were already there -- and a sequential generator
        cannot promise that, because the order it is drawn in changes with the
        membership."""
        namespace = self._shapes()
        dice = namespace["_tree_dice"]

        self.assertEqual(dice(193627.9, 445301.6), dice(193627.9, 445301.6))
        self.assertNotEqual(dice(193627.9, 445301.6), dice(193628.9, 445301.6))

        rolls = np.array([dice(x, y) for x, y in
                          zip(np.linspace(0, 5000, 400), np.linspace(700, 90, 400))])
        self.assertTrue((rolls >= 0).all() and (rolls < 1).all())
        for column in range(rolls.shape[1]):
            self.assertGreater(
                float(rolls[:, column].std()), 0.2,
                f"die {column} barely varies, so every tree gets the same roll",
            )

    def test_a_mature_crown_is_not_convex(self):
        """One deformed ellipsoid is still an ellipsoid: push it about as much
        as you like and the outline stays convex, which is what makes a
        lollipop a lollipop. The reference is the billowy mass photogrammetry
        gives you, and the only way to an outline like that is more than one
        lump."""
        namespace = self._shapes()
        big, _, _ = self._one(namespace, 5000.0, 900.0, 20.0)
        small, _, _ = self._one(namespace, 5000.0, 900.0, 8.0)
        crown_big = big[15:]
        crown_small = small[15:]

        # Two lobes on the mature tree, one on the young one, so the vertex
        # count doubles rather than the shape merely stretching.
        self.assertEqual(
            len(crown_big), 2 * len(crown_small),
            f"a 20 m crown has {len(crown_big)} vertices against "
            f"{len(crown_small)} on an 8 m one, so it is not lobed",
        )
        # And the lobes are offset from each other rather than stacked.
        half = len(crown_big) // 2
        first = crown_big[:half].mean(axis=0)
        second = crown_big[half:].mean(axis=0)
        self.assertGreater(
            float(np.linalg.norm(first - second)), 1.0,
            "the two lobes sit on top of one another, which is one lobe",
        )
        # Sideways, not merely one above the other: two lobes stacked on the
        # axis are still a single round outline seen from above, and the
        # billowy look is what this is for.
        self.assertGreater(
            float(np.linalg.norm(first[:2] - second[:2])), 1.0,
            f"the lobes are only {np.linalg.norm(first[:2] - second[:2]):.2f} m "
            f"apart in plan, so the crown is round from above",
        )

    def test_no_two_crowns_are_the_same_shape(self):
        """Two trees of the same height next to each other have to differ, or a
        row of them reads as one asset repeated."""
        namespace = self._shapes()
        # Two broadleaves of identical size, at different positions.
        found = []
        for x in np.arange(1000.0, 1400.0, 3.0):
            if namespace["_tree_dice"](x, 500.0)[2] >= 0.34:
                found.append(x)
            if len(found) == 2:
                break
        self.assertEqual(len(found), 2, "could not find two broadleaves")

        first, _, _ = self._one(namespace, found[0], 500.0, 14.0)
        second, _, _ = self._one(namespace, found[1], 500.0, 14.0)
        # Compare the crowns about their own centres, so position is not the
        # difference being measured. The trunk is three rings of five vertices
        # and the crown is appended after it, so the tail is exactly the crown
        # -- a height cut picks different counts once the crowns differ, which
        # is the thing being measured.
        a, b = first[15:], second[15:]
        self.assertEqual(len(a), len(b), "the two crowns are not comparable")
        spread = np.abs((a - a.mean(axis=0)) - (b - b.mean(axis=0))).max()
        self.assertGreater(
            float(spread), 0.15,
            f"two crowns differ by {spread:.3f} m, which is one asset twice",
        )

    def test_a_crown_is_not_a_ball(self):
        """The deformation has to actually deform. An ellipsoid is a lollipop
        at any distance you can still see the trunk at."""
        namespace = self._shapes()
        for x in np.arange(2000.0, 2400.0, 3.0):
            if namespace["_tree_dice"](x, 800.0)[2] >= 0.34:
                break
        # A young tree, deliberately: it gets one lobe, so the deformation is
        # the only thing that can make its crown irregular. On a lobed crown
        # the lobes supply the spread and the wobble could be gone entirely.
        vertices, _, _ = self._one(namespace, x, 800.0, 9.0)
        crown = vertices[15:]
        centre = crown.mean(axis=0)
        radius = np.linalg.norm(crown - centre, axis=1)
        # A sphere has one radius; a lobed crown has a spread of them.
        self.assertGreater(
            float(radius.std() / radius.mean()), 0.12,
            "every crown vertex is the same distance out, so it is a ball",
        )

    def test_a_tree_stands_on_the_ground(self):
        """Whatever the species, the trunk reaches the ground and the crown is
        above it. A tree floating is worse than a tree missing."""
        namespace = self._shapes()
        for height in (3.0, 5.0, 9.0, 18.0, 28.0):
            for x in (1500.0, 1503.0):
                vertices, faces, uvs = self._one(namespace, x, 600.0, height)
                # The flared foot sinks a hundredth of the tree's height in,
                # so it meets the ground rather than standing on it.
                self.assertAlmostEqual(
                    float(vertices[:, 2].min()), -0.01 * height, places=6,
                    msg=f"a {height:.0f} m tree does not meet the ground",
                )
                self.assertLessEqual(
                    float(vertices[:, 2].max()), height + 0.6,
                    f"a {height:.0f} m tree is {vertices[:, 2].max():.1f} m tall",
                )
                self.assertGreater(float(vertices[:, 2].max()), height * 0.6)
                self.assertEqual(len(uvs), len(faces) * 3)

    def test_detail_follows_size(self):
        """A sapling is not worth the geometry of the oak next to it. The
        saving is in the lobe count rather than in a coarser primitive: a
        cheaper ball only ever looked like a gem on a stick."""
        namespace = self._shapes()
        counts = {
            height: len(self._one(namespace, 3000.0, 400.0, height)[1])
            for height in (5.0, 20.0)
        }
        self.assertLess(
            counts[5.0], counts[20.0],
            f"a young tree costs {counts[5.0]} triangles and a mature one "
            f"{counts[20.0]}",
        )

    def test_nothing_comes_out_pointy(self):
        """Leafy, not coniferous. A wood of spikes reads as a Christmas tree
        farm, and these models are of Dutch parks and streets.

        A cone is narrow at the top and wide at the foot all the way down; a
        leafy crown is widest in its middle. That is the difference, and it is
        the one worth asserting.
        """
        namespace = self._shapes()
        for height in (6.0, 12.0, 20.0):
            for x in (6000.0, 6003.0, 6006.0):
                vertices, _, _ = self._one(namespace, x, 200.0, height)
                crown = vertices[15:]
                axis = crown[:, :2].mean(axis=0)
                radius = np.linalg.norm(crown[:, :2] - axis, axis=1)
                low, high = crown[:, 2].min(), crown[:, 2].max()

                def widest(lo, hi):
                    band = (crown[:, 2] >= low + (high - low) * lo) & (
                        crown[:, 2] < low + (high - low) * hi
                    )
                    return float(radius[band].max()) if band.any() else 0.0

                foot, middle = widest(0.0, 0.3), widest(0.35, 0.65)
                self.assertGreater(
                    middle, foot * 0.85,
                    f"a {height:.0f} m crown is {foot:.1f} m across at the "
                    f"foot and {middle:.1f} m in the middle, which is a cone",
                )

class TestBlenderMeshContract(unittest.TestCase):
    """Every mesh handed to Blender has to be shaped the way Blender wants.

    This exists because a real run died on it. `build_structures` welded its
    corners -- turning 210 loose corners into 105 shared vertices -- and then
    told Blender the loop indices still ran 0..209 and handed it one UV per
    welded vertex instead of one per loop. Blender's answer was "Array length
    mismatch (got 210, expected more)", six stack frames from the mistake.

    The rules are small and none of them are checkable by reading:

      * one UV per loop, not per vertex -- a weld breaks the two apart
      * every loop index inside the vertex buffer
      * the loop totals adding up to the number of loops
    """

    def test_every_builder_hands_blender_a_well_formed_mesh(self):
        checked = 0
        for name, calls, error in _blender_mesh_calls():
            with self.subTest(builder=name):
                self.assertIsNone(error, f"{name} raised: {error}")
                for call in calls:
                    checked += 1
                    label = f"{name} -> {call['object']}"
                    self.assertEqual(
                        call["uvs"], call["loops"],
                        f"{label}: {call['uvs']} UVs for {call['loops']} loops. "
                        f"UVs are per loop; after a weld that is corners"
                        f"[indices], not corners.",
                    )
                    for extra in call.get("extra_uvs", []):
                        self.assertEqual(extra, call["loops"], f"{label}: extra UV map")
                    self.assertLess(
                        call["max_index"], call["vertices"],
                        f"{label}: loop index {call['max_index']} into a buffer "
                        f"of {call['vertices']} vertices. After a weld the loop "
                        f"indices are the weld's own, not arange(3n).",
                    )
                    self.assertEqual(
                        call["loop_total_sum"], call["loops"],
                        f"{label}: loop totals sum to "
                        f"{call['loop_total_sum']} for {call['loops']} loops",
                    )
                    self.assertEqual(
                        call["totals"], call["polygons"],
                        f"{label}: {call['totals']} loop totals for "
                        f"{call['polygons']} polygons",
                    )
        self.assertGreater(checked, 8, f"only {checked} meshes were exercised")

    def test_the_welded_builders_really_are_welded(self):
        """A weld that returns as many vertices as loops has done nothing, and
        the whole point of it was to stop every triangle carrying its own three
        corners."""
        welded = {"build_roads", "build_structures"}
        seen = set()
        for name, calls, error in _blender_mesh_calls():
            if name not in welded or error:
                continue
            for call in calls:
                seen.add(name)
                self.assertLess(
                    call["vertices"], call["loops"],
                    f"{name} -> {call['object']}: {call['vertices']} vertices "
                    f"for {call['loops']} loops, so nothing was shared",
                )
        self.assertEqual(seen, welded, f"only exercised {sorted(seen)}")


class TestSourceRegistry(unittest.TestCase):
    """One registry drives the preflight and the UI's source picker."""

    def setUp(self):
        import copy

        from src.config import DEFAULTS

        self.config = copy.deepcopy(DEFAULTS)

    def test_structural_sources_cannot_be_turned_off(self):
        from src.sources import SOURCES, apply_selection, enabled_sources

        required = {s.id for s in SOURCES if s.required}
        self.assertEqual(required, {"terrain", "aerial", "buildings"})

        # Selecting nothing still leaves the three a model cannot do without.
        apply_selection(self.config, set())
        self.assertEqual(
            {s.id for s in enabled_sources(self.config)}, required
        )

    def test_selection_flips_the_matching_config_keys(self):
        from src.sources import apply_selection, enabled_sources

        apply_selection(self.config, {"trees", "water"})
        self.assertTrue(self.config["trees"]["enabled"])
        self.assertTrue(self.config["surfaces"]["water"])
        self.assertFalse(self.config["surfaces"]["land_cover"])
        self.assertFalse(self.config["furniture"]["enabled"])
        self.assertFalse(self.config["usage"]["enabled"])

        self.assertEqual(
            {s.id for s in enabled_sources(self.config)},
            {"terrain", "aerial", "buildings", "trees", "water"},
        )

    def test_shared_hosts_are_probed_once(self):
        """Eight BGT layers behind one host must not look like eight outages."""
        from src.sources import health_targets

        targets = health_targets(self.config)
        bgt = [ids for url, ids in targets.items() if "bgt" in url]
        self.assertEqual(len(bgt), 1)
        self.assertEqual(
            set(bgt[0].split(",")),
            {"trees", "water", "land_cover", "furniture", "barriers",
             "vehicles", "rails", "structures"},
        )

    def test_disabled_sources_are_not_checked(self):
        from src.sources import apply_selection, enabled_sources, health_targets

        apply_selection(self.config, set())
        targets = health_targets(self.config, enabled_sources(self.config))
        joined = " ".join(targets)
        self.assertNotIn("/bgt/", joined)
        # "bag" is a substring of "3dbag", so match the BAG service path.
        self.assertNotIn("/lv/bag/", joined)
        self.assertIn("api.3dbag.nl", joined)
        # Terrain, aerial, and both of the buildings services.
        self.assertIn("data.3dbag.nl", joined)
        self.assertEqual(len(targets), 4)

    def test_the_3dbag_api_being_down_does_not_block_a_run(self):
        """It is the outage the fallback exists for, so it must not abort."""
        from src.sources import BY_ID, down_sources

        buildings = BY_ID["buildings"]
        api, tiles = buildings.probes(self.config)
        blocked = down_sources(self.config, {api: False, tiles: True}, [buildings])
        self.assertEqual(blocked, [])

    def test_both_3dbag_services_down_does_block(self):
        from src.sources import BY_ID, down_sources

        buildings = BY_ID["buildings"]
        api, tiles = buildings.probes(self.config)
        blocked = down_sources(self.config, {api: False, tiles: False}, [buildings])
        self.assertEqual([s.id for s in blocked], ["buildings"])

    def test_a_source_with_one_service_still_blocks_on_it(self):
        from src.sources import BY_ID, down_sources

        terrain = BY_ID["terrain"]
        self.assertEqual(len(terrain.probes(self.config)), 1)
        blocked = down_sources(
            self.config, {terrain.probes(self.config)[0]: False}, [terrain]
        )
        self.assertEqual([s.id for s in blocked], ["terrain"])

    def test_every_source_can_name_its_service(self):
        from src.sources import SOURCES

        for source in SOURCES:
            self.assertTrue(
                source.url(self.config).startswith("http"),
                f"{source.id} has no service URL",
            )
            self.assertTrue(
                source.probe(self.config).startswith("http"),
                f"{source.id} has nothing to probe",
            )
            self.assertTrue(source.contributes, f"{source.id} says nothing")

    def test_reachability_does_not_ask_for_the_whole_national_dataset(self):
        """A feature endpoint with no bbox enumerates the country.

        3DBAG takes fourteen seconds to answer that and the probe gives up
        after six, so the check reported an outage on a service that was up.
        """
        from src.sources import BY_ID, health_targets

        buildings = BY_ID["buildings"]
        self.assertTrue(buildings.url(self.config).endswith("/items"))
        self.assertFalse(buildings.probe(self.config).endswith("/items"))

        self.assertIn(buildings.probe(self.config), health_targets(self.config))


class TestSurfaceRegions(unittest.TestCase):
    """The terrain has to arrive knowing which faces are road and which grass.

    Flattening a plot for a building is the operation, and it was a lasso and a
    prayer: the class of every square metre was already worked out for the land
    cover map and simply never attached to the ground.
    """

    def test_the_class_names_reach_the_blender_stage(self):
        import inspect

        from src.export import write_scene_description

        self.assertIn(
            "surface_class_names",
            inspect.signature(write_scene_description).parameters,
        )
        source = inspect.getsource(write_scene_description)
        self.assertIn('"class_names"', source)

    def test_the_pipeline_passes_the_real_names(self):
        import inspect

        import pipeline
        from src.surfaces import CLASS_NAMES

        self.assertIn("surface_class_names=", inspect.getsource(pipeline.run))
        # The names have to be usable as vertex group suffixes.
        for name in CLASS_NAMES.values():
            self.assertRegex(name, r"^[a-z][a-z0-9_]*$", f"{name!r} is not a safe name")

    def test_the_grid_terrain_is_tagged(self):
        """Wired into the terrain Blender actually builds, or the model looks
        right and is still unworkable."""
        import ast

        source = (REPO_ROOT / "blender" / "process.py").read_text()
        tree = ast.parse(source)
        build = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_terrain"
        )
        called = {
            node.func.id for node in ast.walk(build)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_tag_surface_regions", called)
        # A face attribute for exactness, vertex groups for one-click select and
        # for proportional editing, which is what levelling a plot uses.
        self.assertIn('mesh.attributes.new("surface_class", "INT", "FACE")', source)
        self.assertIn("obj.vertex_groups.new", source)


def _height_by_brute_force(vertices, triangles, query):
    """The height of a triangulation at each query point, found the slow way.

    Every triangle tested against every point, no spatial index, no shared code
    with the sampler under test. It is the reference that makes "the sampler
    picked a triangle the point is not inside" a visible failure rather than a
    plausible-looking height.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    out = np.full(len(query), np.nan)
    for row, (px, py) in enumerate(np.asarray(query, dtype=np.float64)):
        for tri in triangles:
            a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            v0, v1 = b[:2] - a[:2], c[:2] - a[:2]
            cross = v0[0] * v1[1] - v0[1] * v1[0]
            if abs(cross) <= 1e-12:
                continue
            dx, dy = px - a[0], py - a[1]
            u = (dx * v1[1] - dy * v1[0]) / cross
            v = (dy * v0[0] - dx * v0[1]) / cross
            if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
                out[row] = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])
                break
    return out


def _fake_ahn(bbox, raw_fn, *, res=0.5, margin=8.0, holes=None):
    """A stand-in for the AHN GeoTIFF: `raw_fn(x, y)` on a 0.5 m raster.

    `holes` is a function of (x, y) that is True where the survey has no
    return, which is how a building footprint arrives.
    """
    edge = bbox.buffered(margin)
    nx = int(round((edge.xmax - edge.xmin) / res))
    ny = int(round((edge.ymax - edge.ymin) / res))
    rx = edge.xmin + (np.arange(nx) + 0.5) * res
    ry = edge.ymax - (np.arange(ny) + 0.5) * res
    gx, gy = np.meshgrid(rx, ry)
    raw = np.asarray(raw_fn(gx, gy), dtype=np.float64)
    if holes is not None:
        raw = np.where(holes(gx, gy), np.float32(3.4028234663852886e38), raw)

    class FakeDataset:
        def __init__(self):
            self.bounds = type(
                "B", (), {"left": edge.xmin, "bottom": edge.ymin,
                          "right": edge.xmax, "top": edge.ymax},
            )()
            self.crs = None
            self.res = (res, res)

        def read(self, _band):
            return raw

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    return FakeDataset


def _build_fake_terrain(bbox, raw_fn, *, grid_m=2.0, holes=None, smooth=1):
    import unittest.mock as mock

    from src import elevation

    cfg = {
        "grid_m": grid_m, "resolution_m": 0.5, "ahn_model": "DTM",
        "wcs_url": "https://example.invalid/wcs", "timeout_s": 1.0,
        "max_retries": 1, "max_nodata_fraction": 0.9,
        "smooth_iterations": smooth,
    }
    dataset = _fake_ahn(bbox, raw_fn, holes=holes)
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(elevation, "fetch_dtm_geotiff"), \
            mock.patch("rasterio.open", return_value=dataset()):
        terrain = elevation.build_terrain(bbox, Path(tmp), terrain_cfg=cfg)
        saved = dict(np.load(Path(tmp) / "terrain_grid.npz"))
    return terrain, saved


def _call_keywords(path: Path, function: str, call: str) -> dict:
    """The keyword arguments of `call(...)` inside `function`, evaluated."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and ast.unparse(inner.func).endswith(call):
                    return {k.arg: ast.literal_eval(k.value) for k in inner.keywords
                            if k.arg != "filepath"}
    raise AssertionError(f"{call} not found in {function}")


def _module_assignment(path: Path, name: str):
    import ast

    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            if isinstance(node.value, ast.Call):
                return {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path.name}")


class TestUnityExport(unittest.TestCase):
    """What the FBX promises Unity, pinned where it can be read."""

    PROCESS = REPO_ROOT / "blender" / "process.py"
    TOOLS = REPO_ROOT / "blender" / "maptile_tools.py"

    def test_the_settings_are_the_ones_measured_to_work(self):
        settings = _module_assignment(self.PROCESS, "FBX_SETTINGS")
        self.assertEqual(settings["axis_forward"], "-Z")
        self.assertEqual(settings["axis_up"], "Y")
        self.assertEqual(settings["apply_scale_options"], "FBX_SCALE_ALL")
        # Apply Transform cannot rewrite a mesh the tree instances share, and
        # left them at their Blender coordinates in Unity's frame.
        self.assertFalse(settings["bake_space_transform"])
        # Empties carry the hierarchy; render modifiers carry the triangulation.
        self.assertEqual(settings["object_types"], {"MESH", "EMPTY"})
        self.assertTrue(settings["use_mesh_modifiers"])
        self.assertTrue(settings["use_mesh_modifiers_render"])
        self.assertTrue(settings["use_custom_props"])

    def test_a_re_export_from_the_tools_matches_the_pipeline(self):
        settings = _module_assignment(self.PROCESS, "FBX_SETTINGS")
        tools = _call_keywords(self.TOOLS, "export_unity_fbx", "export_scene.fbx")
        self.assertEqual(tools, settings)

    def test_the_export_splits_quads_the_way_things_were_placed(self):
        """Trees and lamp posts were placed on each terrain square split south-
        west to north-east; Blender's FIXED method splits first corner to
        third, and the terrain's squares are wound from the south-west."""
        import ast

        source = self.PROCESS.read_text(encoding="utf-8")
        self.assertIn('modifier.quad_method = quads', source)
        tree = ast.parse(source)
        for name in ("build_terrain", "build_roads"):
            body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [
                c for c in ast.walk(body)
                if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_triangulate_on_export"
            ]
            self.assertTrue(calls, f"{name} does not triangulate for export")
            for call in calls:
                quads = {k.arg: ast.literal_eval(k.value) for k in call.keywords}.get("quads")
                self.assertEqual(quads, "FIXED", name)
        # And the winding: bottom-left, bottom-right, top-right, top-left.
        self.assertIn("bottom_left + 1,\n            bottom_left + n + 1,", source)

    def test_every_layer_is_gathered_under_a_parent(self):
        source = self.PROCESS.read_text(encoding="utf-8")
        for layer in ('_Layer("Roads", "Roads")', '_Layer("Buildings", "Buildings")',
                      '_Layer("Trees", "Trees")', '_Layer("Collision", "Collision")'):
            self.assertIn(layer, source)


class TestMaptileTools(unittest.TestCase):
    """The tools in the .blend fit the terrain the way the pipeline did."""

    @staticmethod
    def _tools():
        import re

        source = (REPO_ROOT / "blender" / "maptile_tools.py").read_text(encoding="utf-8")
        namespace = {"np": np}
        # The plain constants: a number or a quoted name on one line.
        for line in re.findall(r"\n([A-Z_]+ = (?:[-0-9.]+|\"[^\"\n]*\"))(?=\n)", source):
            exec(line, namespace)  # noqa: S102 - our own source
        for pattern in (r"\nclass TriangleLocator:.*?\n(?=\ndef |\nclass )",
                        r"\ndef fit_terrain\(.*?\n(?=\ndef |\nclass )"):
            body = re.search(pattern, source, re.S)
            exec(body.group(0), namespace)  # noqa: S102 - our own source
        return namespace

    def test_the_constants_agree_with_the_pipeline(self):
        from src import elevation, roads

        tools = self._tools()
        self.assertEqual(tools["TERRAIN_CLEARANCE_M"], roads.TERRAIN_CLEARANCE_M)
        self.assertEqual(tools["HIDDEN_VERTEX_WEIGHT"], elevation.HIDDEN_VERTEX_WEIGHT)

    def test_fitting_in_blender_is_fitting_in_the_pipeline(self):
        """On the terrain mesh, outlines and all: same point location, same
        correction, the same vertices kept back for being in sight."""
        import copy

        from src.elevation import lower_under
        from src.roads import road_underside
        from src.terrain_mesh import ROAD, build_terrain_mesh

        terrain, roads, water, buildings = _outline_terrain()
        terrain.mesh = build_terrain_mesh(terrain, roads=roads, water=water, buildings=buildings)
        mesh = copy.deepcopy(terrain.mesh)
        points = road_underside(roads, terrain.bbox)
        lower_under(terrain, points, clearance_m=0.05)

        # The triangles as the tool reads them from Blender: each face fanned
        # from its first corner, which for a quad is the export's own split.
        sizes = mesh.face_sizes.astype(np.int64)
        fans = sizes - 2
        face = np.repeat(np.arange(len(sizes)), fans)
        step = np.arange(int(fans.sum())) - np.repeat(np.cumsum(fans) - fans, fans) + 1
        first = mesh.loop_starts[face]
        loops = mesh.face_loops
        triangles = np.column_stack([loops[first], loops[first + step], loops[first + step + 1]])
        visible = np.zeros(len(mesh.vertices), dtype=bool)
        face_of_loop = np.repeat(np.arange(len(sizes)), sizes)
        visible[loops[mesh.face_kind[face_of_loop] != ROAD]] = True

        fitted, moved = self._tools()["fit_terrain"](
            mesh.vertices, triangles, visible, points, 0.05, mesh.grid_m
        )
        np.testing.assert_allclose(fitted, terrain.mesh.vertices[:, 2], atol=1e-6)
        self.assertGreater(moved, 0)

    def test_the_tools_find_points_as_the_pipeline_does(self):
        from src.terrain_mesh import TriangleLocator

        terrain, roads, water, buildings = _outline_terrain()
        from src.terrain_mesh import build_terrain_mesh

        mesh = build_terrain_mesh(terrain, roads=roads, water=water, buildings=buildings)
        triangles, _ = mesh.triangles()
        xy = mesh.vertices[:, :2]
        bounds = (*xy.min(axis=0), *xy.max(axis=0))
        rng = np.random.default_rng(8)
        x = rng.uniform(bounds[0] - 5, bounds[2] + 5, 4000)
        y = rng.uniform(bounds[1] - 5, bounds[3] + 5, 4000)
        ours = TriangleLocator(xy, triangles, cell=2.0, bounds=bounds).locate(x, y)
        theirs = self._tools()["TriangleLocator"](xy, triangles, cell=2.0, bounds=bounds).locate(x, y)
        np.testing.assert_array_equal(ours[0], theirs[0])
        np.testing.assert_allclose(ours[1], theirs[1])


class TestBarrierHeightCheck(unittest.TestCase):
    """Walls stand on the ground; awnings and canopies hang over it."""

    @staticmethod
    def _barriers(kind_name, top):
        from src.barriers import BarrierSet

        vertices = np.array([[0, 0, 0], [4, 0, 0], [4, 0, top], [0, 0, top]], dtype=float)
        if kind_name == "awning":
            vertices[:, 2] += top - 0.1  # a plate up in the air
        return BarrierSet(
            vertices=vertices, uv=np.zeros((4, 2)),
            triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            material=np.zeros(2, np.int32), kind=np.zeros(2, np.int32),
            material_names=["brick"], kind_names=[kind_name], see_through=[],
        )

    def _verdict(self, barriers):
        from src.validate import CheckReport, check_barriers

        report = CheckReport()
        check_barriers(report, barriers, lambda x, y: np.zeros(np.shape(x)))
        return {c.name: c for c in report.checks}["barriers_are_a_believable_height"]

    def test_a_nine_metre_wall_is_a_mistake(self):
        self.assertFalse(self._verdict(self._barriers("wall", 9.0)).passed)

    def test_an_awning_eight_metres_up_is_not(self):
        self.assertTrue(self._verdict(self._barriers("awning", 8.5)).passed)


def _outline_terrain():
    """A small area with everything the terrain mesh follows.

    Lumpy ground 2 m above NAP, the T-junction street from `_street_fixture`,
    a house beside it, a pond at 1 m NAP, and a ditch 4 m wide: too narrow
    for a row on each bank, so it is laddered with a bed line down it.
    """
    import types

    from src.elevation import TerrainResult
    from src.geo import BBox
    from src.surfaces import WaterBody

    bbox = BBox(-20.0, -30.0, 100.0, 90.0)
    axis_x = np.linspace(bbox.xmin, bbox.xmax, 61)
    axis_y = np.linspace(bbox.ymin, bbox.ymax, 61)
    gx, gy = np.meshgrid(axis_x, axis_y)
    heights = (2.0 + 0.3 * np.sin(gx / 7.0) * np.cos(gy / 9.0)).astype(np.float32)
    terrain = TerrainResult(
        heights=heights.copy(), xs=axis_x, ys=axis_y, bbox=bbox, center_z_nap=2.0,
        nodata_fraction_raw=0.0, filled_fraction=0.0, geotiff_path=None,
        coverage_id="t", resolution_m=0.5, measured=heights.copy(),
    )
    _, roads = _build_street(ground=terrain.sample)
    angle = np.linspace(0.0, 2.0 * np.pi, 41)[:-1]
    pond = np.column_stack([70.0 + 14.0 * np.cos(angle), 60.0 + 9.0 * np.sin(angle)])
    pond = np.vstack([pond, pond[:1]])
    ditch = _rect(-10.0, -20.0, 80.0, -16.0)
    water = [
        WaterBody(rings=[pond], level_nap=1.0, area_m2=0.0, measured=True),
        WaterBody(rings=[ditch], level_nap=1.2, area_m2=0.0, measured=True),
    ]
    house = np.array([[5.0, 10.0], [15.0, 10.0], [15.0, 20.0], [5.0, 20.0]])
    roof = np.array([
        [[*house[0], 8.0], [*house[1], 8.0], [*house[2], 8.0]],
        [[*house[0], 8.0], [*house[2], 8.0], [*house[3], 8.0]],
    ])
    buildings = types.SimpleNamespace(buildings=[types.SimpleNamespace(roof_tris=roof)])
    return terrain, roads, water, buildings


class TestTerrainMesh(unittest.TestCase):
    """The terrain's edges follow what is on the ground.

    What the person editing it gets: the waterline, the road edge and each
    building outline as edges, a row of quads along each, the plain grid
    further in -- and one closed sheet, with every height where it belongs.
    """

    @classmethod
    def setUpClass(cls):
        from src.terrain_mesh import build_terrain_mesh

        cls.terrain, cls.roads, cls.water, cls.buildings = _outline_terrain()
        cls.mesh = build_terrain_mesh(
            cls.terrain, roads=cls.roads, water=cls.water, buildings=cls.buildings,
            water_depth_m=1.2,
        )

    def _edges(self, mesh=None):
        mesh = mesh or self.mesh
        sizes = mesh.face_sizes.astype(np.int64)
        starts = mesh.loop_starts
        following = np.arange(len(mesh.face_loops)) + 1
        following[starts + sizes - 1] = starts
        pairs = np.sort(np.column_stack([mesh.face_loops, mesh.face_loops[following]]), axis=1)
        return np.unique(pairs, axis=0, return_counts=True)

    def test_it_is_one_closed_sheet(self):
        from src.validate import CheckReport, check_terrain_mesh

        self.assertNotIn("fallback", self.mesh.stats)
        edges, counts = self._edges()
        self.assertFalse((counts > 2).any(), "an edge with three faces")
        xy = self.mesh.vertices[:, :2]
        open_edges = edges[counts == 1]
        bbox = self.terrain.bbox

        def on_bbox(p):
            return (np.isclose(p[:, 0], bbox.xmin) | np.isclose(p[:, 0], bbox.xmax)
                    | np.isclose(p[:, 1], bbox.ymin) | np.isclose(p[:, 1], bbox.ymax))

        self.assertTrue((on_bbox(xy[open_edges[:, 0]]) & on_bbox(xy[open_edges[:, 1]])).all(),
                        "a crack inside the area")
        triangles, _ = self.mesh.triangles()
        t = xy[triangles]
        area = 0.5 * ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                      - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
        self.assertTrue((area > 0).all(), "a face upside down")
        self.assertAlmostEqual(float(area.sum()), bbox.width * bbox.height, places=3)
        report = CheckReport()
        check_terrain_mesh(report, types_namespace(mesh=self.mesh))
        verdicts = {c.name: c for c in report.checks}
        self.assertTrue(verdicts["terrain_is_one_closed_sheet"].passed,
                        verdicts["terrain_is_one_closed_sheet"].detail)

    def test_the_check_finds_a_hole(self):
        from src.terrain_mesh import TerrainMesh
        from src.validate import CheckReport, check_terrain_mesh

        mesh = self.mesh
        keep = np.ones(len(mesh), dtype=bool)
        keep[len(mesh) // 2] = False
        holed = TerrainMesh(
            vertices=mesh.vertices, measured_z=mesh.measured_z,
            face_sizes=mesh.face_sizes[keep],
            face_loops=mesh.face_loops[np.repeat(keep, mesh.face_sizes)],
            face_kind=mesh.face_kind[keep], face_part=mesh.face_part[keep],
            vertex_on=mesh.vertex_on, edges=mesh.edges, edge_on=mesh.edge_on,
            grid_m=mesh.grid_m, stats=dict(mesh.stats),
        )
        report = CheckReport()
        check_terrain_mesh(report, types_namespace(mesh=holed))
        self.assertFalse({c.name: c for c in report.checks}["terrain_is_one_closed_sheet"].passed)

    def test_the_check_finds_a_corner_cut_off(self):
        """Both ends of the cut lie on the bbox -- on two different sides."""
        from src.terrain_mesh import TerrainMesh, grid_mesh
        from src.validate import CheckReport, check_terrain_mesh

        plain = grid_mesh(self.terrain)
        report = CheckReport()
        check_terrain_mesh(report, types_namespace(mesh=plain))
        self.assertTrue({c.name: c for c in report.checks}["terrain_is_one_closed_sheet"].passed)
        # The corner square (0, 1, n + 1, n) loses the triangle at the corner.
        n = len(self.terrain.xs)
        cut = TerrainMesh(
            vertices=plain.vertices, measured_z=plain.measured_z,
            face_sizes=np.concatenate([[3], plain.face_sizes[1:]]).astype(np.int32),
            face_loops=np.concatenate([[1, n + 1, n], plain.face_loops[4:]]),
            face_kind=plain.face_kind, face_part=plain.face_part,
            vertex_on=plain.vertex_on, edges=plain.edges, edge_on=plain.edge_on,
            grid_m=plain.grid_m, stats=dict(plain.stats),
        )
        report = CheckReport()
        check_terrain_mesh(report, types_namespace(mesh=cut))
        verdict = {c.name: c for c in report.checks}["terrain_is_one_closed_sheet"]
        self.assertFalse(verdict.passed)
        self.assertTrue(verdict.detail.startswith("1 crack edges"), verdict.detail)
        self.assertIn("covering 14,398.00 of 14,400.00 m2", verdict.detail)

    def test_the_waterline_is_an_edge_on_the_bank_never_under_water(self):
        """The bank top stays where it was surveyed; the waterline is only
        raised to the water level where the survey dips below it."""
        import copy

        from src.terrain_mesh import WATER, WATERLINE, build_terrain_mesh

        mesh = self.mesh
        line = np.flatnonzero(mesh.vertex_on & WATERLINE)
        self.assertGreater(len(line), 20)
        xy = mesh.vertices[line, :2]
        on_pond = np.hypot((xy[:, 0] - 70.0) / 14.0, (xy[:, 1] - 60.0) / 9.0) < 1.2
        surveyed = self.terrain.sample(xy[on_pond, 0], xy[on_pond, 1])
        np.testing.assert_allclose(mesh.measured_z[line[on_pond]], surveyed, atol=1e-6)
        # The same pond with the water standing above its banks: the
        # waterline comes up to the water, not the water down to the bank.
        flooded = copy.deepcopy(self.water)
        flooded[0].level_nap = 3.0
        high = build_terrain_mesh(self.terrain, roads=self.roads, water=flooded,
                                  buildings=self.buildings)
        edge = np.flatnonzero(high.vertex_on & WATERLINE)
        near = np.hypot((high.vertices[edge, 0] - 70.0) / 14.0,
                        (high.vertices[edge, 1] - 60.0) / 9.0) < 1.2
        np.testing.assert_allclose(high.measured_z[edge[near]], 3.0, atol=1e-9)
        # Everything strictly under water lies on or below the bed.
        face_of_loop = np.repeat(np.arange(len(mesh)), mesh.face_sizes)
        dry = np.zeros(len(mesh.vertices), dtype=bool)
        dry[mesh.face_loops[mesh.face_kind[face_of_loop] != WATER]] = True
        wet = np.zeros(len(mesh.vertices), dtype=bool)
        wet[mesh.face_loops[mesh.face_kind[face_of_loop] == WATER]] = True
        under = wet & ~dry
        self.assertTrue(under.any())
        pond = under & (np.hypot((mesh.vertices[:, 0] - 70.0) / 14.0,
                                 (mesh.vertices[:, 1] - 60.0) / 9.0) < 1.0)
        self.assertTrue((mesh.vertices[pond, 2] <= 1.0 - 1.2 + 1e-9).all())

    def test_a_row_of_quads_runs_along_each_outline(self):
        import shapely

        from src.terrain_mesh import (
            BUILDING_EDGE, LAND, ROAD_EDGE, ROW, ROW_EDGE, WATERLINE,
        )

        mesh = self.mesh
        for bit in (WATERLINE, ROAD_EDGE, BUILDING_EDGE, ROW_EDGE):
            self.assertTrue((mesh.edge_on & bit).any(), f"no edges tagged {bit}")
        row = (mesh.face_part == ROW) & (mesh.face_kind == LAND)
        self.assertGreater(float((mesh.face_sizes[row] == 4).mean()), 0.75)
        # The inner edge of the row is a row's width in from the outline.
        outline = shapely.MultiLineString([
            mesh.vertices[e, :2] for e, bits in zip(mesh.edges, mesh.edge_on)
            if bits & (WATERLINE | ROAD_EDGE | BUILDING_EDGE)
        ])
        inner = np.unique(mesh.edges[(mesh.edge_on & ROW_EDGE) > 0])
        distance = shapely.distance(shapely.points(mesh.vertices[inner, :2]), outline)
        self.assertAlmostEqual(float(np.median(distance)), mesh.grid_m, delta=0.35)

    def test_a_narrow_ditch_is_a_ladder_with_a_bed_line(self):
        from src.terrain_mesh import BED, WATER

        mesh = self.mesh
        bed = np.unique(mesh.edges[(mesh.edge_on & BED) > 0])
        in_ditch = bed[(mesh.vertices[bed, 1] > -19.5) & (mesh.vertices[bed, 1] < -16.5)]
        self.assertGreater(len(in_ditch), 10, "no bed line down the ditch")
        np.testing.assert_allclose(mesh.vertices[in_ditch, 1], -18.0, atol=0.3)
        self.assertTrue((mesh.vertices[in_ditch, 2] <= 1.2 - 1.2 + 1e-9).all())
        cells = np.add.reduceat(mesh.vertices[mesh.face_loops, :2], mesh.loop_starts) / \
            mesh.face_sizes[:, None]
        ditch = (mesh.face_kind == WATER) & (cells[:, 1] < -16.0) & (cells[:, 1] > -20.0) \
            & (cells[:, 0] > 0.0) & (cells[:, 0] < 70.0)
        self.assertGreater(float((mesh.face_sizes[ditch] == 4).mean()), 0.9)
        from src.terrain_mesh import CORE

        self.assertFalse((ditch & (mesh.face_part == CORE)).any(),
                         "a lone grid square in a ditch too narrow for rows")

    def test_open_ground_is_still_the_grid(self):
        from src.terrain_mesh import CORE, WATER

        mesh = self.mesh
        # Under the pond the grid carries on, on the bed rather than the survey.
        core = np.flatnonzero((mesh.face_part == CORE) & (mesh.face_kind != WATER))
        self.assertGreater(len(core), 1000)
        corners = mesh.vertices[mesh.face_loops[mesh.loop_starts[core][:, None] + np.arange(4)]]
        step = self.terrain.grid_m
        np.testing.assert_allclose(corners[:, 1, 0] - corners[:, 0, 0], step, atol=1e-6)
        np.testing.assert_allclose(corners[:, 3, 1] - corners[:, 0, 1], step, atol=1e-6)
        # Its heights are the surveyed grid's own.
        node = corners[:, 0]
        i = np.rint((node[:, 0] - self.terrain.xs[0]) / step).astype(int)
        j = np.rint((node[:, 1] - self.terrain.ys[0]) / step).astype(int)
        np.testing.assert_allclose(mesh.measured_z[mesh.face_loops[mesh.loop_starts[core]]],
                                   self.terrain.measured[j, i], atol=1e-5)

    def test_with_nothing_to_follow_it_is_the_plain_grid(self):
        from src.terrain_mesh import build_terrain_mesh

        mesh = build_terrain_mesh(self.terrain, roads=self.roads, water=self.water,
                                  buildings=self.buildings, sources=[])
        n = len(self.terrain.xs)
        self.assertEqual(len(mesh), (n - 1) ** 2)
        self.assertTrue((mesh.face_sizes == 4).all())
        self.assertEqual(len(mesh.edges), 0)

    def test_outlines_that_cannot_be_meshed_leave_the_plain_grid(self):
        from unittest import mock

        from src import terrain_mesh

        with mock.patch.object(terrain_mesh, "_feature_mesh", side_effect=RuntimeError("bad")):
            mesh = terrain_mesh.build_terrain_mesh(
                self.terrain, roads=self.roads, water=self.water, buildings=self.buildings,
            )
        self.assertEqual(len(mesh), (len(self.terrain.xs) - 1) ** 2)
        self.assertIn("bad", mesh.stats["fallback"])

    def test_points_are_found_in_their_triangle(self):
        mesh = self.mesh
        triangles, _ = mesh.triangles()
        rng = np.random.default_rng(3)
        bbox = self.terrain.bbox
        x = rng.uniform(bbox.xmin, bbox.xmax, 5000)
        y = rng.uniform(bbox.ymin, bbox.ymax, 5000)
        tri, w = mesh.locator().locate(x, y)
        self.assertTrue((tri >= 0).all())
        corners = mesh.vertices[triangles[tri], :2]
        back = (w[:, :, None] * corners).sum(axis=1)
        np.testing.assert_allclose(back, np.column_stack([x, y]), atol=1e-6)
        self.assertTrue((w > -1e-6).all())
        # At a vertex the drawn height is that vertex's.
        some = np.arange(0, len(mesh.vertices), 97)
        np.testing.assert_allclose(
            mesh.heights_at(mesh.vertices[some, 0], mesh.vertices[some, 1]),
            mesh.vertices[some, 2], atol=1e-6,
        )

    def test_the_ground_drawn_is_the_mesh_and_the_fit_clears_the_road(self):
        import copy

        from src.elevation import lower_under
        from src.roads import road_top, road_underside

        terrain = copy.deepcopy(self.terrain)
        terrain.mesh = copy.deepcopy(self.mesh)
        points = road_underside(self.roads, terrain.bbox)
        lower_under(terrain, points, clearance_m=0.05)
        ground = terrain.ground(points[:, 0], points[:, 1])
        self.assertTrue((ground <= points[:, 2] - 0.05 + 1e-6).all())
        self.assertIn("visible_over_half_metre", terrain.conformed)
        terrain.road_top = road_top(self.roads, terrain.bbox)
        np.testing.assert_allclose(
            terrain.surface(np.array([45.0]), np.array([0.0])),
            terrain.road_top(np.array([45.0]), np.array([0.0])),
        )
        # The survey is untouched by the fit.
        np.testing.assert_allclose(terrain.mesh.measured_z, self.mesh.measured_z)

    def test_blender_builds_it_with_its_outlines_selectable(self):
        import re

        from src.terrain_mesh import BUILDING_EDGE, WATERLINE

        source = (REPO_ROOT / "blender" / "process.py").read_text(encoding="utf-8")
        built = {}

        class Attribute:
            def __init__(self, name):
                self.name, self.values = name, None
                outer = self

                class data:  # noqa: N801 - mirrors bpy
                    @staticmethod
                    def foreach_set(key, values):
                        outer.values = np.asarray(values)
                self.data = data

        class Edges:
            def __init__(self, pairs):
                self.pairs = pairs
                self.seams = None

            def __len__(self):
                return len(self.pairs)

            def foreach_get(self, key, out):
                out[:] = self.pairs.ravel()

            def foreach_set(self, key, values):
                self.seams = np.asarray(values)

        class Group:
            def __init__(self, name):
                self.name, self.members = name, []

            def add(self, members, weight, mode):
                self.members = list(members)

        class Groups(list):
            def new(self, name):
                group = Group(name)
                self.append(group)
                return group

        def build_mesh_object(name, vertices, loops, starts, totals, *rest, **kw):
            following = np.arange(len(loops)) + 1
            following[starts + totals - 1] = starts
            pairs = np.unique(np.sort(np.column_stack([loops, loops[following]]), axis=1), axis=0)

            class Mesh:
                pass

            mesh = Mesh()
            mesh.vertices = [None] * len(vertices)
            mesh.polygons = [None] * len(starts)
            mesh.edges = Edges(pairs[::-1].copy())  # Blender's own order
            created = {}

            class Attributes:
                @staticmethod
                def new(attr_name, kind, domain):
                    created[attr_name] = Attribute(attr_name)
                    return created[attr_name]

            mesh.attributes = Attributes
            obj = type("Object", (), {})()
            obj.name, obj.data, obj.vertex_groups = name, mesh, Groups()
            built.update(obj=obj, attributes=created, vertices=len(vertices),
                         polygons=len(starts))
            return obj

        namespace = {"np": np, "Path": Path, "log": lambda *a: None,
                     "build_mesh_object": build_mesh_object,
                     "_triangulate_on_export": lambda *a, **k: None,
                     "_tag_surface_regions": lambda *a, **k: 0}
        for constant in ("OUTLINE_GROUPS", "OUTLINE_SEAMS"):
            line = re.search(rf"\n{constant} = .*?\n(?=\n)", source, re.S)
            exec(line.group(0), namespace)  # noqa: S102 - our own source
        for name in ("planar_uv", "_mark_outlines", "_build_terrain_mesh", "build_terrain"):
            body = re.search(rf"\ndef {name}\(.*?\n(?=\ndef |\n# ---|\n[A-Z_]+ = )", source, re.S)
            exec(body.group(0), namespace)  # noqa: S102 - our own source
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / "terrain_mesh.npz", **self.mesh.to_npz_fields())
            scene = {"terrain": {"file": "terrain_mesh.npz"}, "origin_rd": [40.0, 30.0],
                     "ground_z_offset_nap": 2.0,
                     "aerial": {"bbox_local": [-60.0, -60.0, 60.0, 60.0]}}
            namespace["build_terrain"](scene, Path(tmp), "M")
        self.assertEqual(built["vertices"], len(self.mesh.vertices))
        self.assertEqual(built["polygons"], len(self.mesh))
        outline = built["attributes"]["outline"].values
        self.assertEqual(int((outline > 0).sum()), len(self.mesh.edges))
        seams = built["obj"].data.edges.seams
        self.assertEqual(int(seams.sum()), int(((self.mesh.edge_on & 7) > 0).sum()))
        groups = {g.name: g.members for g in built["obj"].vertex_groups}
        np.testing.assert_array_equal(
            sorted(groups["outline_waterline"]),
            np.flatnonzero(self.mesh.vertex_on & WATERLINE),
        )
        np.testing.assert_array_equal(
            sorted(groups["outline_building"]),
            np.flatnonzero(self.mesh.vertex_on & BUILDING_EDGE),
        )
        for name in ("measured_z", "ground_kind", "ground_part"):
            self.assertIn(name, built["attributes"])

    def test_the_config_names_what_the_terrain_follows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            base = {"name": "x", "style": "survey",
                    "bbox": {"crs": "EPSG:28992", "xmin": 136000,
                             "ymin": 455000, "xmax": 137000, "ymax": 456000}}
            path.write_text(json.dumps(base))
            self.assertEqual(load_config(path).terrain["feature_edges"],
                             ["water", "roads", "buildings"])
            # The layered style's terrain follows its own layers' outlines.
            path.write_text(json.dumps({**base, "style": "layered"}))
            self.assertEqual(load_config(path).terrain["feature_edges"], [])
            path.write_text(json.dumps({**base, "terrain": {"feature_edges": False}}))
            self.assertEqual(load_config(path).terrain["feature_edges"], [])
            path.write_text(json.dumps({**base, "terrain": {"feature_edges": ["Water"]}}))
            self.assertEqual(load_config(path).terrain["feature_edges"], ["water"])
            path.write_text(json.dumps({**base, "terrain": {"feature_edges": ["hedges"]}}))
            with self.assertRaises(ValueError):
                load_config(path)


class TestTerrainMeshHelpers(unittest.TestCase):
    """Each of these made a crack, an overlap or a hole over Bronbeek once.

    The small test area never trips them, so each is pinned down here with
    the shape that did.
    """

    def test_a_shared_edge_gets_identical_stations_from_both_sides(self):
        import shapely

        from src.terrain_mesh import _with_stations

        rng = np.random.default_rng(5)
        for _ in range(50):
            a = rng.uniform(0, 500, 2).round(1)
            b = a + rng.uniform(5, 30, 2).round(1)
            c = a + np.array([0.0, -40.0])
            d = a + np.array([0.0, 40.0])
            left = _with_stations(shapely.Polygon([a, b, d]), 3.2)
            right = _with_stations(shapely.Polygon([b, a, c]), 3.2)
            ab = b - a

            def on_ab(poly):
                return {
                    tuple(p) for p in np.asarray(poly.exterior.coords)
                    if abs(ab[0] * (p - a)[1] - ab[1] * (p - a)[0]) < 1e-6 * np.hypot(*ab)
                }

            self.assertEqual(on_ab(left), on_ab(right))
            self.assertGreater(len(on_ab(left)), 2)

    def test_a_flat_triangle_keeps_its_middle_vertex(self):
        from src.terrain_mesh import _without_flat_triangles

        a, m, b, c = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([1.0, 1.0])
        out = _without_flat_triangles(np.array([[a, m, b], [a, b, c]]))
        self.assertEqual(len(out), 2)
        used = {tuple(p) for t in out for p in t}
        self.assertIn(tuple(m), used)
        area = 0.5 * np.abs((out[:, 1, 0] - out[:, 0, 0]) * (out[:, 2, 1] - out[:, 0, 1])
                            - (out[:, 2, 0] - out[:, 0, 0]) * (out[:, 1, 1] - out[:, 0, 1]))
        self.assertAlmostEqual(float(area.sum()), 1.0)

    def test_a_square_a_pointed_outline_slips_into_is_not_grid(self):
        import shapely

        from src.terrain_mesh import _core_cells

        axis = np.arange(0.0, 12.01, 2.0)
        notch = shapely.Polygon([(3.45, 12.0), (3.55, 12.0), (3.5, 4.5)])
        area = shapely.box(0.0, 0.0, 12.0, 12.0).difference(notch)
        taken = np.zeros((len(axis) - 1, len(axis) - 1), dtype=bool)
        cells = {tuple(c) for c in _core_cells(area, axis, axis, taken)}
        # The square from (2, 4) to (4, 6): its nine samples miss the notch.
        self.assertNotIn((1, 2), cells)
        self.assertIn((3, 2), cells)

    def test_a_gap_left_at_the_bbox_edge_is_given_away(self):
        import shapely

        from src.terrain_mesh import _fill_gaps

        frame = shapely.box(0.0, 0.0, 10.0, 10.0)
        left = shapely.Polygon([(0, 0), (5, 0), (5, 10), (0.3, 10), (0, 9)])
        right = shapely.box(5.0, 0.0, 10.0, 10.0)
        filled = _fill_gaps([left, right], frame)
        self.assertAlmostEqual(shapely.union_all(filled).area, 100.0, places=6)

    def test_a_rung_grazing_the_outline_is_flagged(self):
        import shapely

        from src.terrain_mesh import _grazes

        # An L-shaped region: a rung from one arm to the other can pass a
        # hair's breadth from the inner corner at (5, 5).
        outline = shapely.Polygon([(0, 0), (10, 0), (10, 10), (5, 10), (5, 5), (0, 5)]).boundary
        clear = shapely.LineString([(0.0, 2.0), (10.0, 2.0)])
        grazing = shapely.LineString([(0.0, 4.895), (10.0, 5.095)])  # 5 mm below the corner
        self.assertEqual(list(_grazes([clear, grazing], outline)), [False, True])

    def test_a_rung_middle_goes_into_the_hole_of_a_face_too(self):
        import shapely

        from src.terrain_mesh import _key, _with_middles

        hole = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]
        face = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [hole])
        middle = np.array([5.0, 4.0])
        middles = {tuple(sorted((_key((4.0, 4.0)), _key((6.0, 4.0))))): middle}
        withd = _with_middles(face, middles)
        self.assertIn(tuple(middle), {tuple(p) for p in np.asarray(withd.interiors[0].coords)})

    def test_a_strip_too_narrow_for_rows_gets_no_grid_square(self):
        """4.5 m of ground between two outlines: a 2 m square fits in it, a
        row along each side does not. A lone square would only break up the
        ladder straight across it."""
        import shapely

        from src.terrain_mesh import LAND, ROW, _mesh_region

        axis = np.arange(0.0, 60.01, 2.0)
        frame = shapely.segmentize(shapely.box(0.0, 0.0, 60.0, 60.0), 2.0001)
        outer = shapely.difference(frame.buffer(8.0, join_style="mitre"), frame)
        strip = shapely.box(10.0, 9.3, 40.0, 13.8)
        taken = np.zeros((len(axis) - 1, len(axis) - 1), dtype=bool)
        faces, cells, inset, _ = _mesh_region(
            strip, LAND, frame, outer, axis, axis, taken, row_m=2.0, margin_m=0.7,
        )
        self.assertTrue(inset.is_empty)
        self.assertEqual(len(cells), 0)
        self.assertFalse(taken.any())
        self.assertTrue(faces)
        self.assertTrue(all(part == ROW for _, part in faces))

    def test_nested_polygons_are_all_found(self):
        import shapely

        from src.terrain_mesh import _parts

        nested = shapely.GeometryCollection([
            shapely.MultiPolygon([shapely.box(0, 0, 1, 1), shapely.box(2, 0, 3, 1)]),
            shapely.LineString([(0, 0), (5, 5)]),
        ])
        self.assertEqual(len(_parts(nested)), 2)


class TestTerrainMeshAwkwardShapes(unittest.TestCase):
    """Shapes that cracked the terrain while it was being built, one by one.

    Each is meshed on its own, and the sheet has to stay closed. The guard
    against each crack sits in `src/terrain_mesh.py`; the tests above check
    each guard works, these that it is used.
    """

    @classmethod
    def setUpClass(cls):
        cls.terrain = _outline_terrain()[0]

    def _closed(self, **layers):
        from src.terrain_mesh import build_terrain_mesh
        from src.validate import CheckReport, check_terrain_mesh

        mesh = build_terrain_mesh(self.terrain, **layers)
        self.assertNotIn("fallback", mesh.stats)
        report = CheckReport()
        check_terrain_mesh(report, types_namespace(mesh=mesh))
        verdict = {c.name: c for c in report.checks}["terrain_is_one_closed_sheet"]
        self.assertTrue(verdict.passed, verdict.detail)
        return mesh

    @staticmethod
    def _water(*rings, level=1.2):
        from src.surfaces import WaterBody

        rings = [np.array(r + r[:1], dtype=np.float64) for r in rings]
        return [WaterBody(rings=rings, level_nap=level, area_m2=0.0, measured=True)]

    def test_a_house_corner_just_inside_the_bbox(self):
        """20 cm from the edge. Snapping the corners together pulled the
        ground off the edge of the area and left a sliver of nothing."""
        corners = np.array([[90.0, 2.0], [99.8, 7.1], [95.0, 16.0], [85.0, 11.0]])
        roof = np.concatenate([corners, np.full((4, 1), 8.0)], axis=1)
        house = types_namespace(roof_tris=roof[[[0, 1, 2], [0, 2, 3]]])
        self._closed(buildings=types_namespace(buildings=[house]))

    def test_a_small_island_in_a_ditch(self):
        """No rung can cross the island, so the ladder cell around it has it
        as a hole -- and still needs the bed points of the rungs on either
        side, which the cells beside it are split at."""
        from src.terrain_mesh import BED

        ditch = [(10.0, 40.0), (50.0, 40.0), (50.0, 44.4), (10.0, 44.4)]
        island = [(29.5, 41.7), (30.5, 41.7), (30.5, 42.7), (29.5, 42.7)]
        mesh = self._closed(water=self._water(ditch, island))
        self.assertTrue((mesh.edge_on & BED).any())

    def test_a_ditch_that_tapers_to_half_a_metre(self):
        """The rung across the end was snapped onto the far corner, so it
        ran along the outline, and its bed point went into the water's side
        of the bank only."""
        ditch = [(10.0, 40.0), (40.0, 41.75), (40.0, 42.25), (10.0, 44.0)]
        self._closed(water=self._water(ditch))


def types_namespace(**kwargs):
    import types

    return types.SimpleNamespace(**kwargs)


def _rect_ring(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float)


def _layered_fixture():
    """A 160 m square with everything the layered ground is made of.

    Lumpy ground with the motorway on a 2 m bank along the south, a street
    across the middle with a footpath either side and a brick side street
    off it to the north, a lawn, a green traffic island in the street (a
    hole in the street's polygon, as the BGT has it), and a bridge deck at
    level 1, which the layers leave to the structures stage.
    """
    from src.elevation import TerrainResult
    from src.geo import BBox
    from src.layered import GRASS, MOTORWAY, ROAD, WALKWAY, GroundArea

    bbox = BBox(1000.0, 2000.0, 1160.0, 2160.0)
    axis_x = np.linspace(bbox.xmin, bbox.xmax, 81)
    axis_y = np.linspace(bbox.ymin, bbox.ymax, 81)
    gx, gy = np.meshgrid(axis_x, axis_y)
    heights = (
        3.0 + 1.5 * np.sin((gx - 1000.0) / 23.0) * np.cos((gy - 2000.0) / 31.0)
        + np.clip((20.0 - np.abs(gy - 2020.0)) / 5.0, 0.0, 2.0)
    ).astype(np.float32)
    terrain = TerrainResult(
        heights=heights.copy(), xs=axis_x, ys=axis_y, bbox=bbox, center_z_nap=3.0,
        nodata_fraction_raw=0.0, filled_fraction=0.0, geotiff_path=None,
        coverage_id="t", resolution_m=0.5, measured=heights.copy(),
    )
    island = _rect_ring(1120.0, 2078.0, 1130.0, 2082.0)
    areas = [
        GroundArea([_rect_ring(1000.0, 2010.0, 1160.0, 2030.0)], MOTORWAY, "Asfalt"),
        GroundArea([_rect_ring(1000.0, 2075.0, 1160.0, 2085.0), island[::-1]], ROAD, "Asfalt"),
        GroundArea([_rect_ring(1075.0, 2085.0, 1085.0, 2160.0)], ROAD, "Bakstenen"),
        GroundArea([_rect_ring(1000.0, 2072.0, 1160.0, 2075.0)], WALKWAY, "Beton_Geel"),
        GroundArea([_rect_ring(1000.0, 2085.0, 1075.0, 2088.0)], WALKWAY, "Beton_Geel"),
        GroundArea([_rect_ring(1085.0, 2085.0, 1160.0, 2088.0)], WALKWAY, "Beton_Geel"),
        GroundArea([island], GRASS, "Gras"),
        GroundArea([_rect_ring(1100.0, 2100.0, 1150.0, 2150.0)], GRASS, "Gras"),
        GroundArea([_rect_ring(1030.0, 2000.0, 1040.0, 2045.0)], ROAD, "Asfalt", level=1),
    ]
    return terrain, areas


class TestLayeredSurfaces(unittest.TestCase):
    """What the layered style fetches besides its own ground."""

    def _build(self, surfaces_cfg: dict, keep_areas: bool):
        from types import SimpleNamespace
        from unittest import mock

        import rasterio
        from rasterio.transform import from_origin

        import src.surfaces as surfaces_module
        from src.geo import BBox

        bbox = BBox(1000.0, 2000.0, 1100.0, 2100.0)
        square = {"type": "Polygon", "coordinates": [[
            [1020, 2020], [1060, 2020], [1060, 2030], [1020, 2030], [1020, 2020],
        ]]}
        ditch = {"type": "Polygon", "coordinates": [[
            [1010, 2060], [1090, 2060], [1090, 2063], [1010, 2063], [1010, 2060],
        ]]}
        features = {
            "wegdeel": [{"geometry": square, "properties": {
                "functie": "parkeervlak", "fysiek_voorkomen": "gesloten verharding"}}],
            "waterdeel": [{"geometry": ditch, "properties": {"type": "watergang"}}],
        }

        def fetch(collection, _bbox, **_kwargs):
            stats = SimpleNamespace(summary=lambda: {}, current_features=0)
            return features.get(collection, []), stats

        with tempfile.TemporaryDirectory() as tmp:
            tif = Path(tmp) / "dtm.tif"
            with rasterio.open(
                tif, "w", driver="GTiff", width=200, height=200, count=1,
                dtype="float32", transform=from_origin(1000.0, 2100.0, 0.5, 0.5),
            ) as dataset:
                dataset.write(np.full((1, 200, 200), 5.0, dtype=np.float32))
            terrain = SimpleNamespace(
                n=51, geotiff_path=tif,
                sample=lambda x, y: np.full(np.shape(x), 5.0),
            )
            cfg = {"page_limit": 1000, "timeout_s": 1, "max_retries": 0,
                   "max_pages": 1, "water": True, "land_cover": True,
                   "water_depth_m": 1.0, **surfaces_cfg}
            with mock.patch.object(surfaces_module, "fetch_current", fetch), \
                    mock.patch.object(surfaces_module, "build_road_mesh", create=True):
                return surfaces_module.build_surfaces(
                    bbox, Path(tmp), surfaces_cfg=cfg, terrain=terrain,
                    keep_areas=keep_areas,
                )

    def test_the_outlines_the_tree_detector_needs_are_kept(self):
        """Trees stood on parking lots in the layered style: it built no road
        mesh, so it kept no road outlines, and the detector had nothing to keep
        off. The outlines are kept now, the survey style's road mesh still
        not built, and the water is there for the water mask and the ditches."""
        from src.surfaces import CLASS_PARKING
        from src.trees import _rasterize_roads, _rasterize_water

        surfaces = self._build({"road_geometry": False}, keep_areas=True)
        self.assertEqual([p.surface_class for p in surfaces.roads], [CLASS_PARKING])
        self.assertIsNone(surfaces.road_mesh)
        self.assertEqual(len(surfaces.water), 1)
        bounds, shape = (1000.0, 2000.0, 1100.0, 2100.0), (100, 100)
        roads = _rasterize_roads(surfaces, bounds, shape, {CLASS_PARKING})
        self.assertAlmostEqual(roads.sum(), 400, delta=60)
        self.assertAlmostEqual(_rasterize_water(surfaces, bounds, shape).sum(), 240, delta=60)

        # Without the layered ground nothing asks for the outlines.
        self.assertEqual(self._build({"road_geometry": False}, keep_areas=False).roads, [])


class TestQuadTerrain(unittest.TestCase):
    """The layered terrain laid out in quads, as the modellers asked for."""

    @classmethod
    def setUpClass(cls):
        from src.layered import build_layered_ground
        from src.surfaces import WaterBody

        cls.terrain, cls.areas = _layered_fixture()
        cls.ditch = WaterBody(
            rings=[_rect_ring(1100.0, 2040.0, 1150.0, 2043.0)], level_nap=2.0,
            area_m2=150.0, measured=True,
        )
        cls.ground = build_layered_ground(
            cls.terrain, cls.areas, prefix="T", water=[cls.ditch]
        )

    def _faces(self, ground, layer):
        mesh = ground.mesh
        starts = mesh.loop_starts
        for f in np.flatnonzero(mesh.face_kind == layer):
            yield mesh.vertices[mesh.face_loops[starts[f]: starts[f] + mesh.face_sizes[f]]]

    def test_the_terrain_is_quads(self):
        """"Complete triangulation < some quads": the default is the quad mesh."""
        stats = self.ground.stats
        self.assertEqual(stats["mesher"], "quads")
        self.assertGreater(stats["quad_share"], 0.8)
        self.assertTrue(set(np.unique(self.ground.mesh.face_sizes)) <= {3, 4})

    def test_a_road_is_one_quad_from_kerb_to_kerb(self):
        """"Prefer whole continuous quads for roads": away from the junction
        and the island, every face of the 10 m street runs right across it."""
        from src.layered import ROAD

        across = []
        for corners in self._faces(self.ground, ROAD):
            x, y = corners[:, 0], corners[:, 1]
            if 1010.0 < x.min() and x.max() < 1060.0 and y.min() >= 2074.9 and y.max() <= 2085.1:
                across.append(len(corners) == 4 and y.min() < 2075.1 and y.max() > 2084.9)
        self.assertGreater(len(across), 5)
        self.assertTrue(all(across))

    def test_a_ditch_is_cut_into_the_ground(self):
        """"Missing ditches": the middle of a ditch is cut below its banks --
        `ditch_depth_m` below the water level, but no steeper than 45° from
        the bank, as no ditch is -- and no water surface is drawn."""
        v = self.ground.mesh.vertices
        middle = (v[:, 0] > 1102) & (v[:, 0] < 1148) & (np.abs(v[:, 1] - 2041.5) < 0.2)
        banks = (v[:, 0] > 1102) & (v[:, 0] < 1148) & (
            (np.abs(v[:, 1] - 2040.0) < 0.05) | (np.abs(v[:, 1] - 2043.0) < 0.05))
        self.assertTrue(middle.any())
        self.assertTrue(banks.any())
        for x, y, z in v[middle]:
            near = banks & (np.abs(v[:, 0] - x) < 4.0)
            drop = v[near, 2] - z
            self.assertTrue((drop > 0.3).all())
            run = np.hypot(v[near, 0] - x, v[near, 1] - y)
            self.assertTrue((drop <= run + 1e-6).all())
        self.assertGreater(self.ground.stats["water_faces"], 0)
        self.assertEqual(self.ground.stats["water_bodies"], 1)

    def test_the_survey_is_kept_within_the_tolerance(self):
        """"Height missing": loops are cut where a face misses the survey."""
        self.assertLessEqual(self.ground.stats["grid_error_max_m"], 0.5 + 1e-6)
        self.assertGreater(self.ground.stats["loops_cut_to_the_survey"], 0)

    def test_far_away_the_grid_is_coarser(self):
        """"Simplifying the amount of points for far away terrain"."""
        from src.layered import build_layered_ground

        far = build_layered_ground(self.terrain, self.areas, prefix="T", focus_m=40.0)
        self.assertEqual(set(self.ground.stats["grid_blocks"]), {"4 m"})
        self.assertIn("16 m", far.stats["grid_blocks"])
        self.assertLess(far.stats["faces"], self.ground.stats["faces"])
        # Bigger squares beside smaller ones share every corner: still one
        # closed sheet.
        self._assert_closed(far.mesh)

    def test_the_quad_terrain_is_one_closed_sheet(self):
        self._assert_closed(self.ground.mesh)

    def _assert_closed(self, mesh):
        """Every inner edge is shared by two faces, and the faces cover the area."""
        starts = mesh.loop_starts
        nxt = np.arange(len(mesh.face_loops)) + 1
        nxt[starts + mesh.face_sizes - 1] = starts
        pairs = np.sort(np.column_stack([mesh.face_loops, mesh.face_loops[nxt]]), axis=1)
        _, counts = np.unique(pairs, axis=0, return_counts=True)
        self.assertLessEqual(counts.max(), 2)
        tris, _ = mesh.triangles()
        p = mesh.vertices[:, :2]
        a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                            - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        self.assertAlmostEqual(area.sum(), 160.0 * 160.0, delta=1.0)
        # Open edges only along the edge of the area.
        once = np.unique(pairs, axis=0, return_counts=True)
        edges = once[0][once[1] == 1]
        mid = 0.5 * (p[edges[:, 0]] + p[edges[:, 1]])
        on_frame = (
            (np.abs(mid[:, 0] - 1000) < 1e-3) | (np.abs(mid[:, 0] - 1160) < 1e-3)
            | (np.abs(mid[:, 1] - 2000) < 1e-3) | (np.abs(mid[:, 1] - 2160) < 1e-3)
        )
        self.assertTrue(on_frame.all())

    def test_a_loop_is_cut_along_a_ridge(self):
        """A crest between two edges gets a loop along it, quads either side."""
        from src.layered_mesh import _loop_cuts

        def ridge(p):
            return np.maximum(0.0, 1.0 - np.abs(p[:, 0] - 4.0) / 2.0)

        xy = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
        gx, gy = np.meshgrid(np.arange(9.0), np.arange(9.0))
        nodes = np.column_stack([gx.ravel(), gy.ravel()])
        out = _loop_cuts(
            xy, np.array([4]), np.arange(4), np.array([0]), np.array([0]), np.array([0]),
            height_at=ridge, nodes_xy=nodes, nodes_z=ridge(nodes), tolerance_m=0.6,
            skip_kind=2,
        )
        new_xy, sizes, loops, _, _, _, cuts = out
        self.assertEqual(cuts, 1)
        self.assertEqual(list(sizes), [4, 4])
        crest = new_xy[np.unique(loops)]
        self.assertEqual(int((np.abs(crest[:, 0] - 4.0) < 1e-9).sum()), 2)

    def test_a_gap_between_two_pieces_is_not_a_hole(self):
        """Regions are meshed one by one, so a sliver no piece covers -- left
        where a hairpin was trimmed -- went to nobody: at Hazeldonk, 23 crack
        edges. It goes to the neighbour sharing most outline with it."""
        import shapely

        from src.layered_mesh import quad_terrain
        from src.terrain_mesh import TerrainMesh

        west = shapely.Polygon([(0, 0), (80, 0), (79.8, 160), (0, 160)])
        east = shapely.Polygon([(80, 0), (160, 0), (160, 160), (80, 160)])
        vertices, sizes, loops, codes, part, _ = quad_terrain(
            self.terrain, [(west, 0), (east, 6 * 16 + 6)], [], codes_wet=15,
            tolerance_m=0.5, focus_m=250.0, depth_m=0.6, paved_codes=set(),
        )
        mesh = TerrainMesh(
            vertices=vertices, measured_z=vertices[:, 2], face_sizes=sizes,
            face_loops=loops, face_kind=np.zeros(len(sizes), dtype=np.int8),
            face_part=part.astype(np.int8), vertex_on=np.zeros(len(vertices), dtype=np.int32),
            edges=np.zeros((0, 2), dtype=np.int64), edge_on=np.zeros(0, dtype=np.int32),
        )
        self._assert_closed(mesh)

    def test_a_ditch_bank_is_held_to_its_slope_against_the_bank_edge(self):
        """The bank's vertices can be metres apart along it while the bank is
        30 cm away: measured to the vertices, the bed dropped 0.9 m in those
        30 cm, a wall 70° steep, and Shrinkwrap threw the grass beside it 13 cm
        up. Measured to the bank edge, the drop is at most the run."""
        from src.layered_mesh import _gentle_banks

        xy = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, -0.3], [0.0, 5.0], [6.0, 5.0]])
        heights = np.array([1.0, 1.0, -0.5, 1.0, 1.0])
        # A water triangle under the bank edge 0-1, and land above it.
        loops = np.array([1, 0, 2, 0, 1, 4, 3])
        sizes = np.array([3, 4])
        kind = np.array([2, 0])
        out = _gentle_banks(heights, loops, sizes, kind, xy, 2)
        self.assertAlmostEqual(out[2], 1.0 - 0.3, places=6)
        self.assertTrue(np.allclose(out[[0, 1, 3, 4]], 1.0))

    def test_the_triangulation_can_still_be_asked_for(self):
        from src.layered import build_layered_ground

        ground = build_layered_ground(self.terrain, self.areas, prefix="T", mesher="triangles")
        self.assertEqual(ground.stats["mesher"], "triangles")
        self.assertLessEqual(ground.stats["grid_error_max_m"], 0.5 + 1e-6)


class TestLayeredGround(unittest.TestCase):
    """The modellers' ground: a coarse terrain and the layers lying on it.

    What the person opening the model gets: a terrain no further from the
    survey than asked, every road, walkway and lawn an object of its own
    whose outline is an edge of that terrain, named the way the hand-made
    file names them, and every placed thing standing on the layer under it.
    """

    TOLERANCE = 0.3

    @classmethod
    def setUpClass(cls):
        from src.layered import build_layered_ground

        cls.terrain, cls.areas = _layered_fixture()
        cls.ground = build_layered_ground(
            cls.terrain, cls.areas, prefix="T", tolerance_m=cls.TOLERANCE, focus_m=50.0,
        )
        cls.mesh = cls.ground.mesh

    def _at(self, x, y):
        """The face under RD point (x, y): its layer and material name."""
        from src.layered import MATERIAL_NAMES

        tri, _ = self.mesh.locator().locate(np.array([x]), np.array([y]))
        _, owner = self.mesh.triangles()
        face = int(owner[tri[0]])
        material = int(self.ground.face_material[face])
        return int(self.mesh.face_kind[face]), (MATERIAL_NAMES[material] if material >= 0 else None)

    def test_the_bgt_is_read_the_way_the_modellers_draw_it(self):
        from src.layered import GRASS, MOTORWAY, ROAD, WALKWAY, classify

        cases = [
            ("wegdeel", {"functie": "rijbaan autosnelweg", "fysiek_voorkomen": "gesloten verharding"},
             (MOTORWAY, "Asfalt")),
            ("wegdeel", {"functie": "rijbaan autoweg"}, (MOTORWAY, "Asfalt")),
            ("wegdeel", {"functie": "rijbaan lokale weg", "fysiek_voorkomen": "gesloten verharding"},
             (ROAD, "Asfalt")),
            ("wegdeel", {"functie": "rijbaan lokale weg", "plus_fysiek_voorkomen": "klinkers"},
             (ROAD, "Bakstenen")),
            ("wegdeel", {"functie": "fietspad"}, (ROAD, "Fietspad")),
            ("wegdeel", {"functie": "voetpad"}, (WALKWAY, "Beton_Geel")),
            ("wegdeel", {"functie": "voetgangersgebied"}, (WALKWAY, "Beton_Geel")),
            ("wegdeel", {"functie": "ruiterpad", "fysiek_voorkomen": "onverhard"}, None),
            ("ondersteunendwegdeel", {"functie": "verkeerseiland", "fysiek_voorkomen": "open verharding"},
             (WALKWAY, "Beton_Geel")),
            ("ondersteunendwegdeel", {"functie": "verkeerseiland", "fysiek_voorkomen": "groenvoorziening"},
             (GRASS, "Gras")),
            ("ondersteunendwegdeel", {"functie": "berm", "fysiek_voorkomen": "groenvoorziening"},
             (GRASS, "Gras")),
            ("ondersteunendwegdeel", {"functie": "berm", "fysiek_voorkomen": "half verhard"},
             (ROAD, "Verharding")),
            ("ondersteunendwegdeel", {"functie": "berm", "fysiek_voorkomen": "onverhard"}, None),
            ("begroeidterreindeel", {"fysiek_voorkomen": "grasland overig"}, (GRASS, "Gras")),
            ("onbegroeidterreindeel", {"fysiek_voorkomen": "gesloten verharding"},
             (ROAD, "Verharding")),
            ("onbegroeidterreindeel", {"fysiek_voorkomen": "zand"}, None),
            ("waterdeel", {"type": "watergang"}, None),
        ]
        for collection, properties, want in cases:
            with self.subTest(collection=collection, properties=properties):
                self.assertEqual(classify(collection, properties), want)

    def test_the_terrain_is_one_closed_sheet(self):
        from src.validate import CheckReport, check_terrain_mesh

        report = CheckReport()
        check_terrain_mesh(report, types_namespace(mesh=self.mesh), expect_outlines=False)
        failed = [str(c) for c in report.checks if not c.passed]
        self.assertEqual(failed, [])
        names = {c.name for c in report.checks}
        self.assertIn("terrain_follows_the_layer_outlines", names)
        self.assertEqual(self.ground.stats["outline_edges_missing"], 0)

    def test_no_face_is_flat(self):
        """A zero-area triangle dropped from the triangulation leaves its
        neighbours a T-junction apart: Hazeldonk had three such cracks."""
        xy = self.mesh.vertices[:, :2] - self.mesh.vertices[:, :2].min(axis=0)
        loops = self.mesh.face_loops
        following = np.arange(len(loops)) + 1
        starts = self.mesh.loop_starts
        following[starts + self.mesh.face_sizes - 1] = starts
        twice = np.add.reduceat(
            xy[loops, 0] * xy[loops[following], 1] - xy[loops[following], 0] * xy[loops, 1], starts
        )
        self.assertGreater(float(twice.min()), 1e-6)

    def test_every_surveyed_node_is_within_the_tolerance(self):
        """Measured on the faces as drawn, each quad split first corner to
        third as Blender draws it and the export cuts it."""
        stats = self.ground.stats
        self.assertLessEqual(stats["grid_error_max_m"], self.TOLERANCE + 1e-9)
        gx, gy = np.meshgrid(self.terrain.xs, self.terrain.ys)
        drawn = self.mesh.locator().interpolate(
            self.mesh.vertices[:, 2], gx.ravel(), gy.ravel()
        )
        error = np.abs(drawn - self.terrain.measured.ravel())
        self.assertLessEqual(float(np.nanmax(error)), self.TOLERANCE + 1e-6)
        # Coarse: far fewer vertices than the grid it was fitted to.
        self.assertLess(len(self.mesh.vertices), 0.5 * gx.size)
        self.assertGreater(stats["quad_share"], 0.2)

    def test_a_flat_triangle_is_mended_not_dropped(self):
        """Three outline points on a line, (0,0), (1,0) and (2,0): the
        triangulator gave Hazeldonk a flat triangle along them, under a
        triangle that spans the line whole. Dropped, the middle point is a
        corner below the line and not above it: a crack. Mended, the
        triangle above is split at the middle point."""
        from src.layered import _without_flat

        points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, -1.0]])
        a, b, c, m, d = range(5)
        tris = np.array([[a, b, c], [a, m, b], [a, d, m], [m, d, b]])
        mended = _without_flat(points, tris)
        corners = points[mended]
        twice = ((corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
                 - (corners[:, 2, 0] - corners[:, 0, 0]) * (corners[:, 1, 1] - corners[:, 0, 1]))
        self.assertTrue((twice > 0).all(), "a flat or flipped triangle is left")
        self.assertAlmostEqual(float(twice.sum()) / 2.0, 2.0)
        edges = np.sort(np.vstack([mended[:, [0, 1]], mended[:, [1, 2]], mended[:, [2, 0]]]), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        outside = {tuple(sorted(e)) for e in ((a, c), (c, b), (b, d), (d, a))}
        for edge, count in zip(map(tuple, unique), counts):
            self.assertEqual(count, 1 if edge in outside else 2, f"edge {edge}")

    def test_the_triangulation_has_no_flat_triangle_in_it(self):
        """Points on the 1 cm grid the outlines are snapped to, for which the
        triangulator returns a flat triangle along (-5.65, 3.57), (-3.67,
        5.88) and (-1.75, 8.12), inside the hull (found by search: about one
        set in two thousand). Dropped, it leaves three cracks."""
        import shapely

        from src.layered import _delaunay

        points = np.array([
            [-5.65, 3.57], [-3.67, 5.88], [-2.99, 6.68], [-1.75, 8.12], [-0.71, 20.15],
            [0.86, 11.17], [1.88, -4.39], [10.35, -4.0], [11.42, 24.51], [13.41, 4.61],
            [16.14, 11.69],
        ])
        tris = _delaunay(points)
        corners = points[tris]
        twice = ((corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
                 - (corners[:, 2, 0] - corners[:, 0, 0]) * (corners[:, 1, 1] - corners[:, 0, 1]))
        self.assertGreater(float(twice.min()), 1e-9)
        hull = shapely.convex_hull(shapely.multipoints(points)).area
        self.assertAlmostEqual(float(twice.sum()) / 2.0, hull, places=9)
        edges = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertLessEqual(int(counts.max()), 2)
        # An edge with a face on one side only is on the hull: inside it, that
        # is a crack, the middle point a corner on one side and not the other.
        rim = shapely.convex_hull(shapely.multipoints(points)).exterior
        middles = shapely.points(points[unique[counts == 1]].mean(axis=1))
        self.assertLess(float(shapely.distance(rim, middles).max()), 1e-9)

    @staticmethod
    def _coverage(walkway):
        """A 20 m square: one walkway, road everywhere else."""
        import shapely

        from src.layered import ROAD, WALKWAY, _code

        frame = shapely.box(0.0, 0.0, 20.0, 20.0)
        road = shapely.set_precision(frame.difference(walkway), 0.01)
        return [(walkway, _code(WALKWAY, "Beton_Geel")), (road, _code(ROAD, "Asfalt"))]

    def test_a_needle_on_a_walkway_is_trimmed_off(self):
        """A 1.4 degree needle tip: measured on the curb generator, a tip that
        sharp throws its kerb tens of metres (825 m at Hazeldonk). The tip
        goes to the road beside it; nothing else moves."""
        import shapely

        from src.layered import WALKWAY, _hairpins, _without_hairpins

        walkway = shapely.Polygon([(0, 0), (10, 0), (10, 1.9), (18, 2.0), (10, 2.1),
                                   (10, 4), (0, 4)])
        pieces, removed = _without_hairpins(self._coverage(walkway))
        self.assertGreaterEqual(removed, 1)
        trimmed = next(p for p, k in pieces if k // 16 == WALKWAY)
        self.assertEqual(_hairpins(trimmed), set())
        self.assertAlmostEqual(trimmed.area, 40.0, places=6)
        self.assertAlmostEqual(sum(p.area for p, _ in pieces), 400.0, places=6)
        self.assertTrue(shapely.union_all([p for p, _ in pieces]).equals(
            shapely.box(0.0, 0.0, 20.0, 20.0)))

    def test_a_tapering_path_keeps_its_taper(self):
        """A path narrowing to a point over 15 m is a path: trimming it corner
        by corner once ate 900 m2 of Hazeldonk's footpaths."""
        import shapely

        from src.layered import WALKWAY, _without_hairpins

        walkway = shapely.Polygon([(0, 0), (4, 0), (19, 1.5), (4, 3), (0, 3)])
        pieces, _ = _without_hairpins(self._coverage(walkway))
        kept = next(p for p, k in pieces if k // 16 == WALKWAY)
        self.assertAlmostEqual(kept.area, walkway.area, places=6)

    def test_a_walkway_too_thin_for_kerbs_joins_its_neighbour(self):
        import shapely

        from src.layered import ROAD, _without_hairpins

        walkway = shapely.Polygon([(2, 5), (12, 5.2), (2, 5.4)])
        pieces, removed = _without_hairpins(self._coverage(walkway))
        self.assertEqual([k // 16 for _, k in pieces], [ROAD, ROAD])
        self.assertAlmostEqual(sum(p.area for p, _ in pieces), 400.0, places=6)

    def test_a_quad_is_the_two_triangles_that_were_fitted(self):
        from src.layered import _faces

        # Two triangles sharing the edge (0, 0)-(2, 2), the long way across
        # the quad: split the other way, the quad is a different surface.
        points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        tris = np.array([[0, 1, 2], [0, 2, 3]])
        faces, codes = _faces(points, tris, np.zeros(2, dtype=np.int64))
        self.assertEqual(len(faces), 1)
        quad = list(faces[0])
        self.assertEqual({quad[0], quad[2]}, {0, 2})
        self.assertEqual(sorted(quad), [0, 1, 2, 3])

    def test_each_area_is_its_layer(self):
        from src.layered import GRASS, MOTORWAY, ROAD, TERRAIN, WALKWAY

        probes = {
            (1080.0, 2020.0): (MOTORWAY, "Asfalt"),
            (1050.0, 2080.0): (ROAD, "Asfalt"),
            (1080.0, 2120.0): (ROAD, "Bakstenen"),
            (1050.0, 2073.5): (WALKWAY, "Beton_Geel"),
            (1125.0, 2125.0): (GRASS, "Gras"),
            (1010.0, 2150.0): (TERRAIN, None),
        }
        for (x, y), want in probes.items():
            with self.subTest(x=x, y=y):
                self.assertEqual(self._at(x, y), want)

    def test_a_green_island_in_the_street_is_raised_and_kerbed(self):
        """Paving all round, small: a walkway, so the curb generator kerbs it,
        still green on top. The lawn beside the side street stays grass."""
        from src.layered import GRASS, WALKWAY

        self.assertEqual(self._at(1125.0, 2080.0), (WALKWAY, "Gras"))
        self.assertEqual(self._at(1110.0, 2110.0), (GRASS, "Gras"))

    def test_a_bridge_deck_is_left_to_the_structures(self):
        from src.layered import TERRAIN

        self.assertEqual(self.ground.stats["parts_on_bridges_or_tunnels"], 1)
        self.assertEqual(self._at(1035.0, 2040.0), (TERRAIN, None))

    def test_the_objects_are_named_and_grouped_as_the_modellers_do(self):
        from src.layered import TERRAIN

        names = self.ground.object_names
        pattern = re.compile(
            r"^T_(Snelweg|Centrum_(Oost|West)_(Wegen|Voetpaden|Gras)"
            r"|(Wegen|Voetpaden|Gras)_Afstand_[NZ][OW])$"
        )
        for name in names:
            self.assertRegex(name, pattern)
        self.assertEqual(len(set(names)), len(names))
        self.assertIn("T_Snelweg", names)
        self.assertTrue(any("Centrum" in n for n in names))
        self.assertTrue(any("Afstand" in n for n in names))
        self.assertEqual(self.ground.terrain_name, "T_Terrein")
        layer = np.asarray(self.mesh.face_kind)
        owner = self.ground.face_object
        np.testing.assert_array_equal(owner < 0, layer == TERRAIN)
        for index, object_layer in enumerate(self.ground.object_layer):
            np.testing.assert_array_equal(layer[owner == index], object_layer)

    def test_no_walkway_is_split_between_two_objects(self):
        """The curb generator kerbs every edge where an object ends, so a
        footpath cut in two would get a kerb across it."""
        from src.layered import WALKWAY

        mesh = self.mesh
        sizes = mesh.face_sizes.astype(np.int64)
        face_of = np.repeat(np.arange(len(sizes)), sizes)
        following = np.arange(len(mesh.face_loops)) + 1
        following[mesh.loop_starts + sizes - 1] = mesh.loop_starts
        edges = np.sort(np.column_stack([mesh.face_loops, mesh.face_loops[following]]), axis=1)
        walk = mesh.face_kind[face_of] == WALKWAY
        seen: dict = {}
        shared = 0
        for edge, face in zip(map(tuple, edges[walk]), face_of[walk]):
            if edge in seen:
                shared += 1
                self.assertEqual(self.ground.face_object[seen[edge]],
                                 self.ground.face_object[face])
            seen[edge] = face
        self.assertGreater(shared, 0)

    def test_things_stand_on_the_layer_under_them(self):
        from src.layered import GRASS, ROAD, TERRAIN, WALKWAY

        top = self.ground.top
        self.assertAlmostEqual(top[WALKWAY], 0.05 + 0.13)
        for (x, y), layer in {
            (1050.0, 2080.0): ROAD, (1050.0, 2073.5): WALKWAY,
            (1125.0, 2125.0): GRASS, (1010.0, 2150.0): TERRAIN,
        }.items():
            with self.subTest(layer=layer):
                ground = self.mesh.locator().interpolate(
                    self.mesh.vertices[:, 2], np.array([x]), np.array([y])
                )[0]
                self.assertAlmostEqual(
                    float(self.ground.surface_at(x, y)), ground + top.get(layer, 0.0), places=6
                )

    def test_the_file_for_blender_needs_no_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.ground.save(Path(tmp))
            data = np.load(path)  # allow_pickle defaults to False
            for key in data.files:
                self.assertNotEqual(data[key].dtype, object, key)
            self.assertEqual([str(n) for n in data["object_names"]], self.ground.object_names)
            self.assertEqual(str(data["terrain_name"]), "T_Terrein")
            self.assertEqual(len(data["face_object"]), len(self.mesh))

    def test_zones_and_names(self):
        from src.layered import GRASS, MOTORWAY, ROAD, WALKWAY, object_name, zone_of

        self.assertEqual(zone_of(10.0, -10.0, 250.0), (True, "Centrum_Oost"))
        self.assertEqual(zone_of(-0.1, 300.0, 250.0), (False, "Afstand_NW"))
        self.assertEqual(zone_of(300.0, -300.0, 250.0), (False, "Afstand_ZO"))
        self.assertEqual(object_name("P", ROAD, (True, "Centrum_West")), "P_Centrum_West_Wegen")
        self.assertEqual(object_name("P", GRASS, (False, "Afstand_NO")), "P_Gras_Afstand_NO")
        self.assertEqual(object_name("P", WALKWAY, (True, "Centrum_Oost")),
                         "P_Centrum_Oost_Voetpaden")
        self.assertEqual(object_name("P", MOTORWAY, (False, "Afstand_ZW")), "P_Snelweg")


def _process_namespace(names, extra=None):
    """Functions and constants from blender/process.py, without Blender.

    Each is exec'd on its own into one namespace, with whatever stand-ins for
    bpy the caller passes in `extra`.
    """
    import ast
    import contextlib

    source = (REPO_ROOT / "blender" / "process.py").read_text(encoding="utf-8")
    namespace = {"np": np, "Path": Path, "log": lambda *a: None, "contextlib": contextlib}
    namespace.update(extra or {})
    for node in ast.parse(source).body:
        wanted = (
            isinstance(node, ast.FunctionDef) and node.name in names
        ) or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(n, ast.Name) and n.id in names
                for target in node.targets for n in ast.walk(target)
            )
        )
        if wanted:
            exec(compile(ast.Module([node], []), "process.py", "exec"), namespace)  # noqa: S102
    return namespace


class _FakeModifiers(list):
    def new(self, name, kind):
        modifier = _FakeModifier(name=name, type=kind, show_viewport=True, show_render=True)
        self.append(modifier)
        return modifier


class _FakeModifier(dict):
    """A modifier: attributes as Blender's, item access as Blender 4's inputs."""

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)


class _FakeMesh:
    def __init__(self, vertices, loops, sizes):
        self.co = np.asarray(vertices, dtype=np.float64)
        self.loops = np.asarray(loops)
        self.polygons = list(range(len(sizes)))
        self.vertices = list(range(len(self.co)))
        self.made = {}
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        following = np.arange(len(self.loops)) + 1
        following[starts + np.asarray(sizes) - 1] = starts
        pairs = np.sort(np.column_stack([self.loops, self.loops[following]]), axis=1)
        self.edge_pairs = np.unique(pairs, axis=0)
        mesh = self

        class Attributes:
            @staticmethod
            def new(name, kind, domain):
                values = {}
                mesh.made[name] = values

                class data:  # noqa: N801 - mirrors bpy
                    @staticmethod
                    def foreach_set(key, array):
                        values[key] = np.array(array)

                return types_namespace(data=data)

        class Edges(list):
            @staticmethod
            def foreach_get(key, out):
                out[:] = mesh.edge_pairs.ravel()

            @staticmethod
            def foreach_set(key, array):
                mesh.made["edge_" + key] = np.array(array)

        self.attributes = Attributes
        self.edges = Edges(range(len(self.edge_pairs)))


class _FakeMeshObject(dict):
    def __init__(self, name, vertices, loops, starts, sizes, uvs, slots, materials):
        super().__init__()
        self.name = name
        self.type = "MESH"
        self.data = _FakeMesh(vertices, loops, sizes)
        self.modifiers = _FakeModifiers()
        self.groups = {}
        self.uvs, self.slots, self.materials = len(uvs), np.asarray(slots), list(materials)
        owner = self

        class Groups:
            @staticmethod
            def new(name):
                members = []
                owner.groups[name] = members
                return types_namespace(add=lambda ids, w, mode: members.extend(ids))

        self.vertex_groups = Groups


class TestLayeredInBlender(unittest.TestCase):
    """The Blender stage's half of the layered style, without Blender.

    The functions are exec'd from blender/process.py against stand-ins that
    record what Blender would be asked to make: which objects, their
    vertices, and the modifiers stacked on each.
    """

    @classmethod
    def setUpClass(cls):
        from src.layered import build_layered_ground

        terrain, areas = _layered_fixture()
        cls.ground = build_layered_ground(terrain, areas, prefix="T", focus_m=50.0)
        cls.folder = Path(tempfile.mkdtemp())
        cls.ground.save(cls.folder)

    @staticmethod
    def _group(name, sockets):
        items = [types_namespace(item_type="SOCKET", in_out="INPUT", name=n, identifier=i)
                 for n, i in sockets]
        items.append(types_namespace(item_type="SOCKET", in_out="OUTPUT", name="Geometry",
                                     identifier="Socket_0"))
        return types_namespace(name=name, interface=types_namespace(items_tree=items))

    @classmethod
    def _curb_group(cls):
        return cls._group("CurbGenerator", [
            ("Width", "Socket_2"), ("Height", "Socket_3"),
            ("Height Offset Curb", "Socket_4"), ("Walkway Material", "Socket_5"),
            ("Curb Material", "Socket_6"),
        ])

    def _build(self, curb_group):
        made = []

        def build_mesh_object(name, vertices, loops, starts, sizes, uvs, slots, materials, **kw):
            obj = _FakeMeshObject(name, vertices, loops, starts, sizes, uvs, slots, materials)
            made.append(obj)
            return obj

        class _Layer:
            def __init__(self, *args):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        namespace = _process_namespace(
            {"build_layered", "_layer_objects", "_mark_layer_outlines", "_shrinkwrap",
             "split_pinches",
             "_weighted_normals", "_triangulate_on_export", "set_modifier_input",
             "planar_uv", "LAYER_ROAD", "LAYER_MOTORWAY", "LAYER_WALKWAY", "LAYER_GRASS",
             "LAYER_ROLES"},
            {"build_mesh_object": build_mesh_object, "_Layer": _Layer,
             "make_plain_material": lambda name, rgb, rough: f"material {name}",
             "load_curb_generator": lambda: curb_group,
             "_flatten_group": lambda: self._group("Maptile_Flatten", []),
             "_drape_group": lambda: self._group(
                 "Maptile_Drape", [("Terrain", "Socket_1"), ("Lift", "Socket_2")])},
        )
        # Unpacked from the tuple assignment above.
        self.assertIn("LAYER_WALKWAY", namespace)
        scene = {
            "origin_rd": [1080.0, 2080.0], "ground_z_offset_nap": 0.0,
            "aerial": {"bbox_local": [-80.0, -80.0, 80.0, 80.0]},
            "layered": {
                "file": "layered_ground.npz", "prefix": "T", "kerbs": True,
                "curb": {"Width": 0.3, "Height": 0.15, "Height Offset Curb": 0.02},
                "walkway_material": "", "curb_material": "Beton",
            },
        }
        summary = namespace["build_layered"](scene, self.folder, "aerial")
        return summary, {obj.name: obj for obj in made}

    def test_one_object_for_the_terrain_and_one_per_layer_name(self):
        summary, made = self._build(self._curb_group())
        self.assertEqual(summary["terrain"], "T_Terrein")
        self.assertEqual(summary["objects"], self.ground.object_names)
        self.assertEqual(set(made), {"T_Terrein", *self.ground.object_names})
        terrain = made["T_Terrein"]
        self.assertEqual(terrain["maptile_role"], "terrain")
        self.assertEqual(len(terrain.data.polygons), len(self.ground.mesh))
        np.testing.assert_array_equal(
            terrain.data.made["ground_layer"]["value"], self.ground.mesh.face_kind
        )
        self.assertTrue(terrain.data.made["edge_use_seam"].any())
        self.assertIn("outline_walkway", terrain.groups)
        for obj in made.values():
            with self.subTest(obj=obj.name):
                self.assertEqual(obj.uvs, len(obj.data.loops), "one UV per loop")
                self.assertLess(int(obj.data.loops.max()), len(obj.data.co))

    def test_each_layer_is_the_terrain_under_it_lifted_and_kept_on_it(self):
        """A road or lawn is wrapped onto the terrain. A kerbed walkway is
        laid flat for the curb generator and draped back on after: the
        generator, made for level outlines, threw spikes up to 0.65 m on
        ones that follow the survey."""
        from src.layered import WALKWAY

        _, made = self._build(self._curb_group())
        terrain = made["T_Terrein"]
        ground_z = {tuple(np.round(p[:2], 6)): p[2] for p in terrain.data.co}
        for index, name in enumerate(self.ground.object_names):
            obj = made[name]
            layer = self.ground.object_layer[index]
            lift = self.ground.lift[layer]
            with self.subTest(obj=name):
                below = np.array([ground_z[tuple(np.round(p[:2], 6))] for p in obj.data.co])
                np.testing.assert_allclose(obj.data.co[:, 2] - below, lift, atol=1e-9)
                kinds = [m.type for m in obj.modifiers]
                # The triangulation last, so it cuts what the others made.
                self.assertEqual(kinds[-1], "TRIANGULATE")
                self.assertEqual(obj.modifiers[-1].quad_method, "FIXED")
                self.assertFalse(obj.modifiers[-1].show_viewport)
                self.assertTrue(obj.modifiers[-1].show_render)
                if layer == WALKWAY:
                    self.assertEqual(obj["maptile_role"], "walkway")
                    self.assertEqual(kinds, ["NODES", "NODES", "NODES", "WEIGHTED_NORMAL",
                                             "TRIANGULATE"])
                    flatten, curb, drape = obj.modifiers[:3]
                    self.assertEqual(flatten.node_group.name, "Maptile_Flatten")
                    self.assertEqual(curb.node_group.name, "CurbGenerator")
                    self.assertEqual(drape.node_group.name, "Maptile_Drape")
                    self.assertIs(drape["Socket_1"], terrain)
                    self.assertAlmostEqual(drape["Socket_2"], lift)
                    continue
                self.assertEqual(kinds, ["SHRINKWRAP", "TRIANGULATE"])
                wrap = obj.modifiers[0]
                self.assertIs(wrap.target, terrain)
                # Not the modellers' "On Surface", which leaves a road under
                # terrain raised past it.
                self.assertEqual(wrap.wrap_mode, "OUTSIDE_SURFACE")
                self.assertTrue(wrap.use_project_z)
                self.assertTrue(wrap.use_negative_direction and wrap.use_positive_direction)
                self.assertAlmostEqual(wrap.offset, lift)

    def test_the_curb_generator_gets_its_settings(self):
        summary, made = self._build(self._curb_group())
        walkways = [n for n, layer in zip(self.ground.object_names, self.ground.object_layer)
                    if made[n]["maptile_role"] == "walkway"]
        self.assertTrue(walkways)
        self.assertEqual(summary["kerbed_objects"], len(walkways))
        self.assertTrue(summary["curb_generator"])
        for name in walkways:
            curb = made[name].modifiers[1]
            self.assertEqual(curb["Socket_2"], 0.3)
            self.assertEqual(curb["Socket_3"], 0.15)
            self.assertEqual(curb["Socket_4"], 0.02)
            self.assertEqual(curb["Socket_6"], "material Beton")
            self.assertNotIn("Socket_5", curb, "no walkway material was asked for")

    def test_without_the_curb_generator_the_walkways_lie_flat(self):
        summary, made = self._build(None)
        self.assertEqual(summary["kerbed_objects"], 0)
        self.assertFalse(summary["curb_generator"])
        for obj in made.values():
            self.assertNotIn("NODES", [m.type for m in obj.modifiers])
            if obj.get("maptile_role") == "walkway":
                self.assertEqual([m.type for m in obj.modifiers], ["SHRINKWRAP", "TRIANGULATE"])

    def test_footpaths_touching_at_a_corner_get_a_vertex_each(self):
        """Two footpaths meeting at one corner share a vertex with four outline
        edges, and the curb generator, following each outline, paired the
        wrong two: over Utrecht kerbs flew 800 m off. Split, each outline is
        a simple loop; nothing moves."""
        split = _process_namespace({"split_pinches"})["split_pinches"]
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [2, 1, 0], [2, 2, 0], [1, 2, 0]], dtype=float)
        loops = np.array([0, 1, 2, 3, 2, 4, 5, 6])
        starts, sizes = np.array([0, 4]), np.array([4, 4])
        new_points, new_loops = split(points, loops, starts, sizes)
        self.assertEqual(len(new_points), 8)
        self.assertEqual(new_loops[:4].tolist(), [0, 1, 2, 3])
        self.assertNotEqual(new_loops[4], 2)
        np.testing.assert_array_equal(new_points[new_loops[4]], points[2])
        # Every outline vertex now has exactly two outline edges.
        following = np.array([1, 2, 3, 0, 5, 6, 7, 4])
        pairs = np.sort(np.column_stack([new_loops, new_loops[following]]), axis=1)
        _, inverse, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
        per_vertex = np.bincount(pairs[counts[inverse.ravel()] == 1].ravel())
        self.assertEqual(set(per_vertex[per_vertex > 0].tolist()), {2})
        # Two faces sharing an edge through the corner stay one fan.
        joined = np.array([0, 1, 2, 3, 1, 4, 2])
        same_points, same_loops = split(
            np.vstack([points[:3], points[3:4], [[2, 0, 0]]]), joined,
            np.array([0, 4]), np.array([4, 3]))
        self.assertEqual(same_loops.tolist(), joined.tolist())

    def test_inputs_are_set_the_way_each_blender_keeps_them(self):
        namespace = _process_namespace({"set_modifier_input"})
        group = self._curb_group()
        old = _FakeModifier(node_group=group)
        self.assertTrue(namespace["set_modifier_input"](old, "Height", 0.2))
        self.assertEqual(old["Socket_3"], 0.2)
        self.assertFalse(namespace["set_modifier_input"](old, "Geometry", 1))
        self.assertFalse(namespace["set_modifier_input"](old, "Colour", 1))
        inputs = types_namespace(Socket_3=types_namespace(value=None))
        new = _FakeModifier(node_group=group, properties=types_namespace(inputs=inputs))
        self.assertTrue(namespace["set_modifier_input"](new, "Height", 0.2))
        self.assertEqual(inputs.Socket_3.value, 0.2)
        self.assertNotIn("Socket_3", new)

    def test_the_export_switches_on_what_only_the_render_shows(self):
        """Blender's FBX exporter applies the modifiers the viewport shows,
        whatever its render option says: measured on 4.2 and 5.2, a
        render-only triangulation never reached the file."""
        for path in (REPO_ROOT / "blender" / "process.py",
                     REPO_ROOT / "blender" / "maptile_tools.py"):
            with self.subTest(file=path.name):
                import ast

                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
                function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                                and n.name in ("export_fbx", "export_unity_fbx"))
                blocks = [n for n in ast.walk(function) if isinstance(n, ast.With)]
                self.assertTrue(any(
                    "render_modifiers" in ast.unparse(item.context_expr)
                    and any("export_scene.fbx" in ast.unparse(c) for c in ast.walk(block))
                    for block in blocks for item in block.items
                ), f"{function.name} exports outside render_modifiers()")

        objects = []
        for flags in ((False, True), (True, False), (True, True), (False, False)):
            modifiers = _FakeModifiers()
            modifier = modifiers.new("m", "TRIANGULATE")
            modifier.show_viewport, modifier.show_render = flags
            objects.append(types_namespace(modifiers=modifiers))
        objects.append(types_namespace())  # an empty: no modifiers at all
        bpy = types_namespace(data=types_namespace(objects=objects))
        namespace = _process_namespace({"render_modifiers"}, {"bpy": bpy})
        before = [(o.modifiers[0].show_viewport, o.modifiers[0].show_render)
                  for o in objects[:4]]
        with namespace["render_modifiers"]() as switched:
            self.assertEqual(switched, 2)
            for obj in objects[:4]:
                self.assertEqual(obj.modifiers[0].show_viewport, obj.modifiers[0].show_render)
        after = [(o.modifiers[0].show_viewport, o.modifiers[0].show_render) for o in objects[:4]]
        self.assertEqual(after, before)

    def test_the_tools_find_the_layered_objects(self):
        import ast

        source = (REPO_ROOT / "blender" / "maptile_tools.py").read_text(encoding="utf-8")
        terrain = types_namespace(type="MESH", name="T_Terrein", get=lambda k, d=None: (
            "terrain" if k == "maptile_role" else d), modifiers=[])

        def layer(name, role, wrapped):
            wrap = types_namespace(type="SHRINKWRAP", target=terrain, show_viewport=True)
            return types_namespace(type="MESH", name=name, modifiers=[wrap] if wrapped else [],
                                   get=lambda k, d=None: role if k == "maptile_role" else d)

        road = layer("T_Centrum_Oost_Wegen", "road", True)
        applied = layer("T_Wegen_Afstand_NO", "road", False)
        grass = layer("T_Gras_Afstand_NO", "grass", False)
        walkway = layer("T_Centrum_Oost_Voetpaden", "walkway", False)
        walkway.modifiers.append(types_namespace(
            type="NODES", show_viewport=True, node_group=types_namespace(name="Maptile_Drape")))
        other = types_namespace(type="MESH", name="Terrain", modifiers=[],
                                get=lambda k, d=None: d)
        bpy = types_namespace(data=types_namespace(
            objects=[other, road, applied, grass, walkway, terrain],
            collections=types_namespace(get=lambda name: None),
        ))
        namespace = {"bpy": bpy, "np": np}
        for node in ast.parse(source).body:
            if (isinstance(node, ast.FunctionDef) and node.name in
                    ("_terrain_object", "_follows", "_road_objects")) or (
                    isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in
                    ("ROAD_COLLECTION", "TERRAIN_NAME", "GROUND_ROLES")):
                exec(compile(ast.Module([node], []), "maptile_tools.py", "exec"), namespace)
        self.assertIs(namespace["_terrain_object"](), terrain)
        # The wrapped road and the draped walkway follow the terrain; the
        # road whose wrap was applied does not, and is what the terrain must
        # stay under.
        self.assertEqual(namespace["_road_objects"](terrain), [applied])
        self.assertIn("grass", namespace["GROUND_ROLES"])


class TestIntermediatesNeedNoPickle(unittest.TestCase):
    """Blender reads these files with its own NumPy, not this one.

    Blender 4.2 bundles NumPy 1.26. An object array is pickled, and a pickle
    written by NumPy 2 refers to `numpy._core`, which 1.26 cannot import: the
    Blender stage died on the first barrier with everything else built.
    """

    def test_barrier_names_load_without_pickle(self):
        from src.barriers import BarrierSet, save_barriers

        barriers = BarrierSet(
            vertices=np.zeros((3, 3)), uv=np.zeros((3, 2)),
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
            material=np.zeros(1, np.int32), kind=np.zeros(1, np.int32),
            material_names=["brick", "paint"], kind_names=["muur", "hek"],
            see_through=["hek"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_barriers(barriers, Path(tmp) / "barriers.npz")
            data = np.load(path)  # allow_pickle defaults to False
            self.assertEqual([str(n) for n in data["kind_names"]], ["muur", "hek"])
            self.assertEqual([str(n) for n in data["see_through"]], ["hek"])
            for key in data.files:
                self.assertNotEqual(data[key].dtype, object, key)

    def test_blender_never_unpickles(self):
        self.assertNotIn(
            "allow_pickle=True",
            (REPO_ROOT / "blender" / "process.py").read_text(encoding="utf-8"),
        )


class TestTerrainGrid(unittest.TestCase):
    """The terrain is a regular grid of quads, because it is edited.

    What is checked here is what an editor and an engine rely on: the spacing
    asked for is the spacing delivered, the surveyed ground is not smoothed
    away, the surface reported to everything placed on it is the surface the
    model draws, and the ground stays out from under the roads.
    """

    def test_the_spacing_asked_for_is_the_spacing_delivered(self):
        from src.elevation import grid_size_for
        from src.geo import BBox

        self.assertEqual(grid_size_for(BBox(0, 0, 1000, 1000), 2.0), 501)
        self.assertEqual(grid_size_for(BBox(0, 0, 1000, 1000), 4.0), 251)
        # Never coarser than asked: 999 m at 2 m is 500 squares, not 499.
        self.assertEqual(grid_size_for(BBox(0, 0, 999, 999), 2.0), 501)
        # An oblong area is sized on its long side.
        self.assertEqual(grid_size_for(BBox(0, 0, 1000, 800), 2.0), 501)
        with self.assertRaises(ValueError):
            grid_size_for(BBox(0, 0, 1000, 1000), 0.0)

    def test_the_surface_is_the_triangles_the_model_draws(self):
        """Each square split south-west to north-east, which is the split
        Blender's fixed quad method and the FBX export make. A bilinear patch
        is no renderer's surface, and the other diagonal is the other one."""
        from src.elevation import _bilinear_on_grid, _triangulated_on_grid

        xs = np.array([0.0, 2.0])
        ys = np.array([0.0, 2.0])
        # A saddle: only the north-east corner is raised.
        grid = np.array([[0.0, 0.0], [0.0, 1.0]])
        # South-east of the diagonal, a quarter of the way up.
        x, y = 1.0, 0.5
        self.assertAlmostEqual(float(_triangulated_on_grid(grid, xs, ys, x, y)), 0.25)
        self.assertAlmostEqual(float(_bilinear_on_grid(grid, xs, ys, x, y)), 0.125)
        # North-west of it, by symmetry.
        self.assertAlmostEqual(float(_triangulated_on_grid(grid, xs, ys, 0.5, 1.0)), 0.25)

        # And against an explicit triangulation of a bumpy grid, point by point.
        n = 9
        axis = np.linspace(0.0, 16.0, n)
        gx, gy = np.meshgrid(axis, axis)
        bumpy = np.sin(gx / 3.0) * np.cos(gy / 4.0) * 2.0
        vertices = np.column_stack([gx.ravel(), gy.ravel(), bumpy.ravel()])
        triangles = []
        for row in range(n - 1):
            for col in range(n - 1):
                bl = row * n + col
                br, tl = bl + 1, bl + n
                tr = tl + 1
                triangles += [[bl, br, tr], [bl, tr, tl]]
        rng = np.random.default_rng(3)
        q = rng.uniform(0.1, 15.9, (300, 2))
        want = _height_by_brute_force(vertices, np.asarray(triangles), q)
        got = _triangulated_on_grid(bumpy, axis, axis, q[:, 0], q[:, 1])
        self.assertLess(float(np.abs(got - want).max()), 1e-9)

    def test_the_last_row_and_column_are_reachable(self):
        from src.elevation import _triangulated_on_grid

        axis = np.array([0.0, 1.0, 2.0])
        grid = np.arange(9.0).reshape(3, 3)
        self.assertAlmostEqual(float(_triangulated_on_grid(grid, axis, axis, 2.0, 2.0)), 8.0)
        self.assertAlmostEqual(float(_triangulated_on_grid(grid, axis, axis, 2.0, 0.0)), 2.0)

    def test_measured_ground_is_not_smoothed(self):
        """Smoothing the survey doubled a 2 m grid's error. Only the ground the
        filler invents under a building is smoothed."""
        from src.geo import BBox

        bbox = BBox(0.0, 0.0, 100.0, 100.0)
        # A quay wall: one metre down at x = 50. Smoothing would round it off.
        quay = lambda x, y: np.where(x < 50.0, 3.0, 2.0)
        hole = lambda x, y: (x > 70.0) & (x < 80.0) & (y > 70.0) & (y < 80.0)
        terrain, _ = _build_fake_terrain(bbox, quay, holes=hole, smooth=3)
        col_dry = int(np.searchsorted(terrain.xs, 46.0))
        col_wet = int(np.searchsorted(terrain.xs, 54.0))
        row = int(np.searchsorted(terrain.ys, 20.0))
        self.assertAlmostEqual(float(terrain.heights[row, col_dry]), 3.0, places=5)
        self.assertAlmostEqual(float(terrain.heights[row, col_wet]), 2.0, places=5)
        # The hole was filled and is finite.
        self.assertTrue(np.isfinite(terrain.heights).all())

    def test_a_built_terrain_reports_its_own_accuracy(self):
        from src.geo import BBox

        bbox = BBox(0.0, 0.0, 120.0, 120.0)
        gentle = lambda x, y: 4.0 + 0.01 * x + 0.3 * np.sin(y / 15.0)
        terrain, saved = _build_fake_terrain(bbox, gentle)
        self.assertAlmostEqual(terrain.grid_m, 2.0)
        self.assertEqual(terrain.n, 61)
        self.assertLess(terrain.survey_error["p95_m"], 0.01)
        # Saved for Blender, with the surveyed copy alongside.
        self.assertEqual(saved["heights"].shape, (61, 61))
        self.assertIn("measured", saved)

        # A ground the grid cannot follow says so.
        rough = lambda x, y: 4.0 + 0.6 * np.sin(x * 2.1) * np.cos(y * 1.7)
        coarse, _ = _build_fake_terrain(bbox, rough, grid_m=4.0)
        self.assertGreater(coarse.survey_error["over_25cm"], 0.05)
        from src.validate import CheckReport, check_terrain

        report = CheckReport()
        check_terrain(report, coarse, bbox)
        verdict = {c.name: c for c in report.checks}["terrain_follows_the_survey"]
        self.assertFalse(verdict.passed)

    @staticmethod
    def _flat_terrain(level=5.0):
        from src.elevation import TerrainResult
        from src.geo import BBox

        bbox = BBox(0.0, 0.0, 40.0, 40.0)
        axis = np.linspace(0.0, 40.0, 21)
        flat = np.full((21, 21), level, dtype=np.float32)
        return TerrainResult(
            heights=flat.copy(), xs=axis, ys=axis, bbox=bbox, center_z_nap=level,
            nodata_fraction_raw=0.0, filled_fraction=0.0, geotiff_path=None,
            coverage_id="test", resolution_m=0.5, measured=flat.copy(),
        )

    def test_the_ground_never_shows_through_a_road(self):
        """Every point under a road ends up with the terrain as drawn below it,
        including where the road dips between two grid vertices: testing the
        road height at the vertex alone lets that dip sink into the ground."""
        from src.elevation import lower_under

        terrain = self._flat_terrain()
        # A road strip from x = 11 to 19, sagging to 4.7 m in the middle of a
        # grid square (x = 15) and back up to 5.03 m at its edges.
        rng = np.random.default_rng(5)
        px = rng.uniform(11.0, 19.0, 4000)
        py = rng.uniform(0.0, 40.0, 4000)
        road = lambda x: 4.7 + 0.33 * np.abs(x - 15.0) / 4.0  # noqa: E731
        info = lower_under(terrain, np.column_stack([px, py, road(px)]), clearance_m=0.05)

        q = np.column_stack([rng.uniform(11.2, 18.8, 500), rng.uniform(1.0, 39.0, 500)])
        ground = terrain.surface(q[:, 0], q[:, 1])
        self.assertTrue((ground <= road(q[:, 0]) - 0.05 + 0.01).all(),
                        f"ground above the road by {float((ground - road(q[:, 0])).max()):.3f} m")
        # Only what is near a road moved; the far field is untouched.
        self.assertAlmostEqual(float(terrain.heights[10, 2]), 5.0)
        self.assertAlmostEqual(float(terrain.heights[10, 18]), 5.0)
        self.assertGreater(info["vertices"], 0)
        # The survey is kept, and is what `sample` reports.
        self.assertAlmostEqual(float(np.ravel(terrain.sample(15.0, 20.0))[0]), 5.0)

    def test_a_bank_beside_a_road_gives_up_less_than_the_road_side(self):
        """The vertex under the road takes the correction; the one on the bank
        beside it, which everyone sees, moves much less. Holding every vertex
        below the lowest road in the squares around it pulled embankment tops
        down by metres beside the Europaweg."""
        from src.elevation import lower_under

        terrain = self._flat_terrain(level=6.0)
        # A sunken road from x = 10 to 13 at 5 m, the bank rising to 6 m beside
        # it; the vertex at x = 14 is the bank top, a metre from the road.
        rng = np.random.default_rng(8)
        px = rng.uniform(10.0, 13.0, 6000)
        py = rng.uniform(2.0, 38.0, 6000)
        lower_under(terrain, np.column_stack([px, py, np.full(len(px), 5.0)]), clearance_m=0.05)
        bank_top = 6.0 - float(terrain.heights[10, 7])  # x = 14
        road_side = 6.0 - float(terrain.heights[10, 6])  # x = 12, under the road
        # Holding both below the road would have dropped the bank top by the
        # full 1.05 m, the same as the vertex under the road.
        self.assertGreater(road_side, 1.05)
        self.assertLess(bank_top, 0.7)
        self.assertLess(bank_top, 0.5 * road_side)

    def test_surface_reads_the_road_where_there_is_one(self):
        from src.elevation import HeightRaster, TerrainResult
        from src.geo import BBox

        bbox = BBox(0.0, 0.0, 10.0, 10.0)
        axis = np.linspace(0.0, 10.0, 6)
        terrain = TerrainResult(
            heights=np.full((6, 6), 1.0, dtype=np.float32), xs=axis, ys=axis,
            bbox=bbox, center_z_nap=1.0, nodata_fraction_raw=0.0,
            filled_fraction=0.0, geotiff_path=None, coverage_id="t",
            resolution_m=0.5,
        )
        top = np.full((20, 20), np.nan)
        top[:, :10] = 1.2  # the western half is road, rows run north-down
        terrain.road_top = HeightRaster(top, bbox)
        got = terrain.surface(np.array([2.0, 8.0]), np.array([5.0, 5.0]))
        self.assertAlmostEqual(float(got[0]), 1.2)
        self.assertAlmostEqual(float(got[1]), 1.0)

    def test_everything_placed_on_the_ground_uses_the_drawn_surface(self):
        """A tree, a post, a fence rail, a sleeper, a parked car: placed at a
        height, with no lift to absorb a difference, so it has to be the
        height the model draws -- the road top where there is one."""
        for module, count in (
            ("barriers", 3), ("furniture", 1), ("rails", 1), ("trees", 2),
        ):
            text = (REPO_ROOT / "src" / f"{module}.py").read_text(encoding="utf-8")
            self.assertGreaterEqual(
                text.count("terrain.surface"), count,
                f"src/{module}.py places objects on the survey, not on the model",
            )

    def test_the_built_terrain_is_checked_against_the_grid(self):
        from src.validate import CheckReport, check_built_terrain

        class Grid:
            n = 5

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for faces, quads, verts, ok in (
                (16, 16, 25, True), (20, 16, 25, False), (16, 16, 24, False),
            ):
                (work / "blender_summary.json").write_text(json.dumps(
                    {"terrain": {"faces": faces, "quads": quads, "vertices": verts}}
                ))
                report = CheckReport()
                check_built_terrain(report, work, Grid())
                verdict = {c.name: c for c in report.checks}["terrain_built_as_planned"]
                self.assertEqual(verdict.passed, ok, (faces, quads, verts))
                self.assertNotIn(
                    "terrain_exported_as_triangles", {c.name for c in report.checks}
                )

    def test_the_exported_terrain_is_checked_for_whole_quads(self):
        """A quad the FBX leaves whole is cut by Unity, not along the
        diagonal everything was placed on."""
        from src.validate import CheckReport, check_built_terrain

        class Grid:
            n = 5

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for left, ok in ((0, True), (16, False)):
                (work / "blender_summary.json").write_text(json.dumps({"terrain": {
                    "faces": 16, "quads": 16, "vertices": 25,
                    "exported_faces": 32 - left, "exported_non_triangles": left,
                }}))
                report = CheckReport()
                check_built_terrain(report, work, Grid())
                verdict = {c.name: c for c in report.checks}["terrain_exported_as_triangles"]
                self.assertEqual(verdict.passed, ok, left)

    def test_a_coarse_terrain_is_held_to_its_own_tolerance(self):
        from src.validate import CheckReport, check_terrain

        terrain = self._flat_terrain()
        for p95, ok in ((0.31, True), (0.62, False)):
            terrain.survey_error = {"p50_m": 0.1, "p95_m": p95, "p99_m": 0.7,
                                    "over_25cm": 0.09, "tolerance_m": 0.5,
                                    "over_tolerance": 0.004}
            report = CheckReport()
            check_terrain(report, terrain, terrain.bbox)
            verdict = {c.name: c for c in report.checks}["terrain_follows_the_survey"]
            self.assertEqual(verdict.passed, ok, p95)
            self.assertIn("0.50 m the terrain is built to", verdict.detail)

    def test_the_layered_ground_is_checked_as_blender_built_it(self):
        from src.layered import GRASS, MOTORWAY, ROAD, WALKWAY
        from src.validate import CheckReport, check_layered

        ground = types_namespace(
            object_names=["P_Snelweg", "P_Centrum_Oost_Wegen", "P_Centrum_Oost_Voetpaden",
                          "P_Gras_Afstand_NO"],
            lift={ROAD: 0.05, MOTORWAY: 0.05, WALKWAY: 0.05, GRASS: 0.02},
        )
        good = {
            "objects": list(ground.object_names), "kerbed_objects": 1,
            "curb_generator": True, "walkway_faces": 100, "walkway_faces_evaluated": 700,
            "height_above_terrain": {
                "road": {"min_m": 0.049, "max_m": 0.058},
                "walkway": {"min_m": 0.05, "max_m": 0.2},
                "grass": {"min_m": 0.0199, "max_m": 0.021},
            },
        }

        def run(layered, kerbs=True):
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                summary = {"terrain": {}}
                if layered is not None:
                    summary["layered"] = layered
                (work / "blender_summary.json").write_text(json.dumps(summary))
                report = CheckReport()
                check_layered(report, work, ground, kerbs=kerbs, kerb_top_m=0.18)
                return {c.name: c for c in report.checks}

        verdicts = run(good)
        self.assertEqual([str(v) for v in verdicts.values() if not v.passed], [])
        self.assertEqual(set(verdicts), {"ground_layers_built", "layers_lie_on_the_terrain",
                                         "walkways_have_kerbs"})
        floating = json.loads(json.dumps(good))
        floating["height_above_terrain"]["road"]["max_m"] = 0.11
        self.assertFalse(run(floating)["layers_lie_on_the_terrain"].passed)
        sunk = json.loads(json.dumps(good))
        sunk["height_above_terrain"]["grass"]["min_m"] = -0.01
        self.assertFalse(run(sunk)["layers_lie_on_the_terrain"].passed)
        # What the first Hazeldonk run measured: kerb spikes on outlines that
        # follow the survey.
        spiked = json.loads(json.dumps(good))
        spiked["height_above_terrain"]["walkway"] = {"min_m": -0.383, "max_m": 0.653}
        self.assertFalse(run(spiked)["layers_lie_on_the_terrain"].passed)
        # And the kerb mitre that ran 825 m off a needle tip: right height,
        # wrong place.
        runaway = json.loads(json.dumps(good))
        runaway["height_above_terrain"]["walkway"]["beyond_outline_m"] = 825.0
        verdict = run(runaway)["layers_lie_on_the_terrain"]
        self.assertFalse(verdict.passed)
        self.assertIn("825.00 m past its outline", verdict.detail)
        # Utrecht: five of 26,000 vertices a few centimetres out, on faces
        # steeper than 65 degrees. Not a layer gone wrong.
        steep = json.loads(json.dumps(good))
        steep["height_above_terrain"]["road"] = {
            "min_m": 0.014, "max_m": 0.11, "vertices": 120000,
            "lowest_m": [0.014] + [0.049] * 49, "highest_m": [0.11] + [0.058] * 49,
        }
        verdict = run(steep)["layers_lie_on_the_terrain"]
        self.assertTrue(verdict.passed, verdict.detail)
        self.assertIn("2 of 120000 vertices outside", verdict.detail)
        # But a road sunk by more than the slack anywhere still fails ...
        deep = json.loads(json.dumps(steep))
        deep["height_above_terrain"]["road"]["min_m"] = -0.2
        deep["height_above_terrain"]["road"]["lowest_m"][0] = -0.2
        self.assertFalse(run(deep)["layers_lie_on_the_terrain"].passed)
        # ... and so does a share of strays, each only a little out.
        many = json.loads(json.dumps(steep))
        many["height_above_terrain"]["road"]["lowest_m"] = [0.02] * 50
        many["height_above_terrain"]["road"]["vertices"] = 20000
        self.assertFalse(run(many)["layers_lie_on_the_terrain"].passed)
        # A kerb a kerb's width past its outline, where two footpaths touch
        # on the edge of the area, passes; more does not.
        edge = json.loads(json.dumps(good))
        edge["height_above_terrain"]["walkway"]["beyond_outline_m"] = 0.3
        verdict = run(edge)["layers_lie_on_the_terrain"]
        self.assertTrue(verdict.passed, verdict.detail)
        self.assertIn("0.30 m past its outline", verdict.detail)
        edge["height_above_terrain"]["walkway"]["beyond_outline_m"] = 0.6
        self.assertFalse(run(edge)["layers_lie_on_the_terrain"].passed)
        flat = dict(good, walkway_faces_evaluated=100)
        self.assertFalse(run(flat)["walkways_have_kerbs"].passed)
        old_blender = dict(good, curb_generator=False, kerbed_objects=0)
        old_blender["height_above_terrain"] = dict(
            good["height_above_terrain"], walkway={"min_m": 0.05, "max_m": 0.051})
        verdicts = run(old_blender)
        self.assertEqual(verdicts["walkways_have_kerbs"].severity, "warning")
        self.assertFalse(verdicts["walkways_have_kerbs"].passed)
        self.assertTrue(verdicts["layers_lie_on_the_terrain"].passed)
        self.assertNotIn("walkways_have_kerbs", run(good, kerbs=False))
        missing = dict(good, objects=good["objects"][:2])
        self.assertFalse(run(missing)["ground_layers_built"].passed)
        self.assertFalse(run(None)["ground_layers_built"].passed)


class TestPathsOverWater(unittest.TestCase):
    """Lidar does not reflect off water, so the bare-earth grid has a hole over
    every body and the filler interpolates across it. The result dips: measured
    over a square kilometre of Gelderland the filled ground mid-water sits
    1.45 m below its own bank, and a footpath draped on that follows it down
    and disappears under the surface it is meant to cross."""

    @staticmethod
    def _body(level, ring):
        from src.surfaces import WaterBody

        return WaterBody(rings=[np.asarray(ring, dtype=float)],
                         level_nap=level, area_m2=100.0, measured=True)

    def test_a_path_does_not_dive_into_the_ditch_it_crosses(self):
        from src.surfaces import _hold_above_water

        ditch = self._body(
            4.0, [[10, -5], [20, -5], [20, 25], [10, 25], [10, -5]]
        )
        points = np.array([[5.0, 10.0], [15.0, 10.0], [25.0, 10.0]])
        # Bank, dip, bank: what the gap filler leaves behind.
        draped = np.array([5.0, 2.6, 5.0])

        held = _hold_above_water(points, draped, [ditch], 0.25)
        self.assertAlmostEqual(held[1], 4.25, places=6)
        # The banks are above water already and are left exactly alone.
        self.assertAlmostEqual(held[0], 5.0, places=6)
        self.assertAlmostEqual(held[2], 5.0, places=6)

    def test_a_bridge_already_clear_of_the_water_is_left_alone(self):
        """Held, not pinned: a real bridge rides higher than a culvert."""
        from src.surfaces import _hold_above_water

        ditch = self._body(4.0, [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
        points = np.array([[5.0, 5.0]])
        held = _hold_above_water(points, np.array([9.0]), [ditch], 0.25)
        self.assertAlmostEqual(held[0], 9.0, places=6)

    def test_each_body_holds_its_own_level(self):
        """Levels across one area run to metres. A shared floor would lift a
        low canal's path into the air to clear a high one."""
        from src.surfaces import _hold_above_water

        low = self._body(1.0, [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
        high = self._body(8.0, [[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]])
        points = np.array([[5.0, 5.0], [25.0, 5.0]])
        held = _hold_above_water(points, np.array([-2.0, -2.0]), [low, high], 0.25)
        self.assertAlmostEqual(held[0], 1.25, places=6)
        self.assertAlmostEqual(held[1], 8.25, places=6)

    def test_no_water_changes_nothing(self):
        from src.surfaces import _hold_above_water

        z = np.array([1.0, 2.0, 3.0])
        points = np.zeros((3, 2))
        self.assertTrue(np.array_equal(_hold_above_water(points, z, [], 0.25), z))
        self.assertTrue(np.array_equal(_hold_above_water(points, z, None, 0.25), z))

    def test_the_setting_reaches_the_config(self):
        from src.config import DEFAULTS

        self.assertEqual(DEFAULTS["surfaces"]["road_water_clearance_m"], 0.25)


class TestCollisionLayer(unittest.TestCase):
    """Physics proxies, for a simulation rather than a picture.

    One property decides whether any of this is worth shipping: a collider has
    to *contain* what it stands in for. A proxy that is cheaper and smaller is
    worse than no proxy at all, because the drone flies into visible foliage
    and nothing happens, and it looks entirely reasonable in the viewport
    while doing it. The first tree proxy here was an inscribed ellipsoid and
    drawn foliage stuck 4.3 m out of it.
    """

    SIDES = 5

    @staticmethod
    def _tree_ns():
        import re

        source = (REPO_ROOT / "blender" / "process.py").read_text(encoding="utf-8")
        namespace = {"np": np}
        for constant in ("TREE_TRUNK_FRACTIONS", "TREE_SHAPES_PER_KIND"):
            line = re.search(rf"\n{constant} = .*\n", source)
            assert line is not None, f"{constant} moved"
            exec(line.group(0), namespace)  # noqa: S102 - our own source
        for name in ("_octahedron_canopy", "_box", "_tree_dice", "_lobed_crown",
                     "_tree_instance", "_shape_lean", "_unit_tree",
                     "_tree_group_mesh", "_collision_tree_mesh"):
            body = re.search(rf"\ndef {name}\(.*?\n(?=\ndef |\n# )", source, re.S)
            assert body is not None, f"{name} moved"
            exec(body.group(0), namespace)  # noqa: S102 - our own source
        return namespace

    @staticmethod
    def _fields(x, y, height):
        return (
            np.array([[x, y]]), np.zeros(1), np.array([float(height)]),
            np.array([float(np.clip(0.26 * height, 0.8, 7.0))]),
            np.array([float(np.clip(0.38 * height, 0.6, 6.0))]),
        )

    def _outside_by(self, points, proxy, has_crown):
        """How far the worst point lies outside the proxy, in metres. Exact.

        The trunk is a convex prism, so a point is inside it when it is behind
        every side plane and between the two ends; the crown is an axis-aligned
        box. Deliberately not sharing any code with the thing under test.
        """
        sides = self.SIDES
        lo_ring = proxy[:sides]
        hi_ring = proxy[sides:sides * 2]
        z_lo, z_hi = lo_ring[0, 2], hi_ring[0, 2]
        centre = lo_ring[:, :2].mean(axis=0)

        planes = []
        for i in range(sides):
            a, b = lo_ring[i, :2], lo_ring[(i + 1) % sides, :2]
            edge = b - a
            normal = np.array([edge[1], -edge[0]])
            normal = normal / np.linalg.norm(normal)
            if normal @ (a - centre) < 0:
                normal = -normal
            planes.append((normal, normal @ a))

        if has_crown:
            box = proxy[sides * 2:]
            box_lo, box_hi = box.min(axis=0), box.max(axis=0)

        worst = 0.0
        for p in points:
            d = max(
                [n @ p[:2] - off for n, off in planes]
                + [z_lo - p[2], p[2] - z_hi]
            )
            if d <= 1e-9:
                continue
            if has_crown:
                over = np.maximum(np.maximum(box_lo - p, p - box_hi), 0.0)
                d = min(d, float(np.linalg.norm(over)))
                if d <= 1e-9:
                    continue
            worst = max(worst, d)
        return worst

    # A position whose dice come out at the extreme of the lean, found by
    # search. A tall tree leaning its hardest is the only case where the trunk
    # proxy needs the lean added to its radius -- the flare covers the rest --
    # and a sweep of ordinary positions never produces one, so it was missed:
    # dropping the allowance passed every test until this went in.
    LEANING = (104114.24, 556464.31)

    def test_the_proxy_contains_every_drawn_vertex(self):
        namespace = self._tree_ns()
        rng = np.random.default_rng(3)
        places = [tuple(rng.uniform(100000, 200000, 2)) for _ in range(9)]
        for height, (x, y) in zip(
            (3.0, 4.0, 7.0, 11.0, 13.9, 14.0, 18.0, 25.0, 32.0), places
        ):
            fields = self._fields(x, y, height)
            drawn, _, _ = namespace["_tree_group_mesh"](
                np.ones(1, bool), fields, (x, y, 0.0), np.random.default_rng(1)
            )
            proxy, _ = namespace["_collision_tree_mesh"](
                np.ones(1, bool), fields, (x, y, 0.0), True
            )
            out = self._outside_by(drawn, proxy, True)
            self.assertLess(
                out, 1e-6,
                f"a {height:.0f} m tree pokes {out:.2f} m out of its own "
                f"collider, so a drone would fly through visible foliage",
            )

    def test_the_trunk_proxy_contains_the_drawn_trunk(self):
        """With crowns off, wood still has to be solid."""
        namespace = self._tree_ns()
        rng = np.random.default_rng(8)
        places = [tuple(rng.uniform(100000, 200000, 2)) for _ in range(4)]
        for height, (x, y) in zip((4.0, 12.0, 22.0, 30.0), places):
            fields = self._fields(x, y, height)
            drawn, _, _ = namespace["_tree_group_mesh"](
                np.ones(1, bool), fields, (x, y, 0.0), np.random.default_rng(1)
            )
            proxy, _ = namespace["_collision_tree_mesh"](
                np.ones(1, bool), fields, (x, y, 0.0), False
            )
            # The drawn trunk is the first three rings of five vertices.
            out = self._outside_by(drawn[: self.SIDES * 3], proxy, False)
            self.assertLess(out, 1e-6, f"the trunk pokes {out:.2f} m out")

    def test_the_hardest_leaning_tree_is_still_contained(self):
        """The trunk proxy is a straight prism and the drawn trunk's top ring
        is pushed sideways by the lean, which is why the lean goes into the
        radius. On most trees the flare already covers it; on a tall one
        leaning its hardest it does not, and dropping the allowance passed
        every other test here."""
        namespace = self._tree_ns()
        x, y = self.LEANING
        _spin, _squash, _spare, lean_d = namespace["_tree_dice"](x, y)
        self.assertLess(
            min(lean_d, 1.0 - lean_d), 0.01,
            "this position no longer gives an extreme lean; search for another",
        )
        for height in (28.0, 30.0, 32.0):
            fields = self._fields(x, y, height)
            drawn, _, _ = namespace["_tree_group_mesh"](
                np.ones(1, bool), fields, (x, y, 0.0), np.random.default_rng(1)
            )
            for crowns in (True, False):
                proxy, _ = namespace["_collision_tree_mesh"](
                    np.ones(1, bool), fields, (x, y, 0.0), crowns
                )
                points = drawn if crowns else drawn[: self.SIDES * 3]
                out = self._outside_by(points, proxy, crowns)
                self.assertLess(
                    out, 1e-6,
                    f"a {height:.0f} m tree leaning hard pokes {out:.2f} m out "
                    f"of its collider (crowns={crowns})",
                )

    def test_it_is_cheaper_than_what_it_replaces(self):
        """Containing the tree is not enough on its own -- the drawn mesh
        contains it perfectly and is what this exists to avoid."""
        namespace = self._tree_ns()
        drawn_total = proxy_total = trunk_total = 0
        for height in (5.0, 9.0, 16.0, 24.0):
            fields = self._fields(150000.0, 450000.0, height)
            _, drawn_faces, _ = namespace["_tree_group_mesh"](
                np.ones(1, bool), fields, (150000.0, 450000.0, 0.0),
                np.random.default_rng(1),
            )
            _, proxy_faces = namespace["_collision_tree_mesh"](
                np.ones(1, bool), fields, (150000.0, 450000.0, 0.0), True
            )
            _, trunk_faces = namespace["_collision_tree_mesh"](
                np.ones(1, bool), fields, (150000.0, 450000.0, 0.0), False
            )
            drawn_total += len(drawn_faces)
            proxy_total += len(proxy_faces)
            trunk_total += len(trunk_faces)
        self.assertEqual(proxy_total, 28 * 4, "28 triangles a tree: 16 + 12")
        self.assertEqual(trunk_total, 16 * 4)
        self.assertGreater(
            drawn_total / proxy_total, 1.8,
            f"{drawn_total} drawn against {proxy_total} proxy is not a saving",
        )

    def test_the_proxy_follows_the_tree_it_stands_for(self):
        """Both read the same dice. A proxy built from its own random numbers
        is a proxy for a different tree in the same place."""
        namespace = self._tree_ns()
        # Two trees a metre apart get different spin, squash and lean, so a
        # proxy that ignored position would contain one and miss the other.
        for x, y in ((150000.0, 450000.0), (150001.0, 450000.0)):
            fields = self._fields(x, y, 20.0)
            drawn, _, _ = namespace["_tree_group_mesh"](
                np.ones(1, bool), fields, (x, y, 0.0), np.random.default_rng(1)
            )
            mine, _ = namespace["_collision_tree_mesh"](
                np.ones(1, bool), fields, (x, y, 0.0), True
            )
            other_fields = self._fields(x + 7.0, y + 13.0, 20.0)
            theirs, _ = namespace["_collision_tree_mesh"](
                np.ones(1, bool), other_fields, (x + 7.0, y + 13.0, 0.0), True
            )
            self.assertLess(self._outside_by(drawn, mine, True), 1e-6)
            # The neighbour's proxy, moved here, should not fit: if it does,
            # the proxy is not reading the tree's own dice at all.
            shifted = theirs - np.array([7.0, 13.0, 0.0])
            self.assertGreater(
                self._outside_by(drawn, shifted, True), 1e-6,
                "any tree's proxy fits any tree, so none of them is specific",
            )

    def test_a_fence_collider_is_too_thick_to_tunnel_through(self):
        from src.barriers import (
            DEFAULT_COLLISION_THICKNESS_M,
            LINE_STYLES,
            collision_style,
        )

        fence = LINE_STYLES[("scheiding_lijn", "hek")]
        self.assertLess(fence.thickness_m, 0.1, "the drawn fence got thick")
        fat = collision_style(fence, DEFAULT_COLLISION_THICKNESS_M)
        self.assertGreaterEqual(fat.thickness_m, DEFAULT_COLLISION_THICKNESS_M)
        # Everything else about it is untouched: the collider is the same
        # fence, in the same place, at the same height.
        self.assertEqual(fat.height_m, fence.height_m)
        self.assertEqual(fat.lift_m, fence.lift_m)

    def test_a_wall_already_thick_enough_is_left_alone(self):
        """Fattening a 70 cm hedge would only move it off its own outline."""
        from src.barriers import LINE_STYLES, collision_style

        hedge = LINE_STYLES[("vegetatieobject_lijn", "haag")]
        self.assertEqual(collision_style(hedge, 0.30), hedge)
        quay = LINE_STYLES[("scheiding_lijn", "kademuur")]
        self.assertEqual(collision_style(quay, 0.30), quay)

    def test_an_awning_is_thickened_downwards_not_sideways(self):
        """An area barrier takes its width from the surveyed outline, so the
        thin direction is its height: a canopy is a 12 cm plate 2.4 m up, and
        a drone climbing a facade goes straight through it."""
        from src.barriers import AREA_STYLES, collision_style

        awning = AREA_STYLES[("gebouwinstallatie", "luifel")]
        self.assertLess(awning.height_m, 0.2)
        fat = collision_style(awning, 0.30)
        self.assertGreaterEqual(fat.height_m, 0.30)
        self.assertEqual(fat.thickness_m, awning.thickness_m)
        self.assertEqual(fat.lift_m, awning.lift_m, "it still hangs off the wall")

    def test_a_shed_is_already_tall_enough(self):
        from src.barriers import AREA_STYLES, collision_style

        for key in (("overigbouwwerk", "bunker"), ("overigbouwwerk", "open loods")):
            style = AREA_STYLES[key]
            self.assertEqual(collision_style(style, 0.30), style)

    def test_the_swept_collider_really_comes_out_thicker(self):
        """`collision_style` only changes a number. What matters is that the
        geometry built from it is wider on the ground than the drawn fence."""
        from src.barriers import LINE_STYLES, _sweep_line, collision_style

        fence = LINE_STYLES[("scheiding_lijn", "hek")]
        run = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        flat = lambda x, y: np.zeros(len(np.atleast_1d(x)))

        thin, _, _ = _sweep_line(run, flat, fence)
        fat, _, _ = _sweep_line(run, flat, collision_style(fence, 0.30))

        # The run is along X, so the width is the Y span.
        thin_w = float(thin[:, 1].max() - thin[:, 1].min())
        fat_w = float(fat[:, 1].max() - fat[:, 1].min())
        self.assertAlmostEqual(thin_w, fence.thickness_m, places=6)
        self.assertAlmostEqual(fat_w, 0.30, places=6)
        # And it still stands in the same place, at the same height.
        self.assertAlmostEqual(
            float(thin[:, 2].max()), float(fat[:, 2].max()), places=6
        )
        self.assertAlmostEqual(
            float(thin[:, 0].max()), float(fat[:, 0].max()), places=6
        )

    def test_both_geometry_paths_build_a_collider(self):
        """A fence is a line and an awning is a polygon, and they go through
        different code. Wiring up only the line path is the obvious way to get
        this half-right, and counting call sites in the source does not catch
        it -- the count moves whenever the code is tidied."""
        import tempfile

        for only in ("line", "area"):
            with tempfile.TemporaryDirectory() as tmp:
                result = self._run_barriers(True, Path(tmp), only=only)
                self.assertGreater(
                    len(result.triangles), 0, f"the {only} fixture drew nothing"
                )
                self.assertGreater(
                    len(result.collision_triangles), 0,
                    f"the {only} path builds no collider",
                )

    @staticmethod
    def _run_barriers(collision: bool | None, folder: Path, only: str | None = None):
        """`build_barriers` against a stubbed BGT, so the sweep is real.

        `collision=None` leaves the argument out altogether, which is how the
        default gets exercised rather than asserted. `only` narrows the fixture
        to one geometry kind: "line" for a fence and a quay wall, "area" for an
        awning.
        """
        import src.barriers as barriers_module
        from src.geo import BBox

        lines = [
            {"properties": {"type": "hek"},
             "geometry": {"type": "LineString",
                          "coordinates": [[10, 10], [50, 10], [50, 50]]}},
            {"properties": {"type": "kademuur"},
             "geometry": {"type": "LineString",
                          "coordinates": [[10, 90], [80, 90]]}},
        ]
        areas = [
            {"properties": {"type": "luifel"},
             "geometry": {"type": "Polygon", "coordinates": [
                 [[20, 20], [26, 20], [26, 24], [20, 24], [20, 20]]]}},
        ]

        def fake_fetch(collection, bbox, class_field=None, **kw):
            if collection == "scheiding_lijn" and only != "area":
                return lines, {}
            if collection == "gebouwinstallatie" and only != "line":
                return areas, {}
            return [], {}

        class Terrain:
            def surface(self, x, y):
                return 2.0 + 0.02 * np.atleast_1d(np.asarray(x, float))

        real = barriers_module.fetch_current
        barriers_module.fetch_current = fake_fetch
        try:
            return barriers_module.build_barriers(
                BBox(0.0, 0.0, 100.0, 100.0), folder,
                barriers_cfg={"page_limit": 10, "timeout_s": 1,
                              "max_retries": 1, "max_pages": 1},
                terrain=Terrain(),
                **({} if collision is None else {"collision": collision}),
            )
        finally:
            barriers_module.fetch_current = real

    def test_no_collider_is_swept_when_it_was_not_asked_for(self):
        """Another sweep of every thin run, and another copy in the npz, on a
        run that will never export it."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            off = self._run_barriers(False, Path(tmp))
            self.assertGreater(len(off.triangles), 0, "the fixture drew nothing")
            self.assertEqual(len(off.collision_triangles), 0)
            self.assertEqual(len(off.collision_vertices), 0)
            self.assertNotIn("collision_triangles", off.stats())

    def test_the_collider_is_swept_when_it_is(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            on = self._run_barriers(True, Path(tmp))
            self.assertGreater(len(on.collision_triangles), 0)
            self.assertEqual(on.collision_thickness_m, 0.30)
            # Indices address the collider's own buffer, which is the mistake
            # that killed a real Blender run once already.
            self.assertLess(
                int(on.collision_triangles.max()), len(on.collision_vertices)
            )
            # The quay wall is already 40 cm, so it gets no second prism: the
            # collider is smaller than the drawn geometry, not a copy of it.
            self.assertLess(
                len(on.collision_triangles), len(on.triangles),
                "every kind got a collider, including the thick ones",
            )

    def test_a_caller_that_says_nothing_gets_the_cheap_path(self):
        """The default has to be the one that costs nothing, so forgetting the
        argument cannot quietly add a sweep and a copy to every run."""
        import inspect
        import tempfile

        from src import barriers

        default = inspect.signature(barriers.build_barriers).parameters[
            "collision"
        ].default
        self.assertIs(default, False)
        with tempfile.TemporaryDirectory() as tmp:
            # Called the same way, with the argument simply left out.
            result = self._run_barriers(None, Path(tmp))
            self.assertGreater(len(result.triangles), 0)
            self.assertEqual(len(result.collision_triangles), 0)

    def test_the_pipeline_asks_for_it_only_when_exporting_it(self):
        import inspect

        import pipeline

        source = inspect.getsource(pipeline.run)
        self.assertIn('collision=bool(config.export.get("collision"', source)

    def test_the_collider_rides_along_in_the_npz(self):
        """Saved and loaded, not grepped for. Writing the right key with the
        wrong value in it looks identical in the source."""
        import tempfile

        from src.barriers import BarrierSet, save_barriers

        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                          [5.0, 5.0, 2.0], [6.0, 5.0, 2.0], [5.0, 6.0, 2.0]])
        barriers = BarrierSet(
            vertices=verts[:3],
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
            uv=np.zeros((3, 2)),
            material=np.zeros(1, np.int32),
            kind=np.zeros(1, np.int32),
            material_names=["paint"],
            kind_names=["fence"],
            collision_vertices=verts[3:],
            collision_triangles=np.array([[0, 1, 2]], dtype=np.int32),
            collision_thickness_m=0.30,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_barriers(barriers, Path(tmp) / "barriers.npz")
            data = np.load(path, allow_pickle=True)
            self.assertEqual(len(data["collision_triangles"]), 1)
            np.testing.assert_allclose(data["collision_vertices"], verts[3:])
            self.assertAlmostEqual(float(data["collision_thickness_m"]), 0.30)

    def test_an_empty_collider_round_trips_too(self):
        """Most runs have export.collision off, so the arrays are empty and
        the Blender stage has to read that without tripping."""
        import tempfile

        from src.barriers import BarrierSet, save_barriers

        barriers = BarrierSet(
            vertices=np.zeros((3, 3)),
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
            uv=np.zeros((3, 2)),
            material=np.zeros(1, np.int32), kind=np.zeros(1, np.int32),
            material_names=["paint"], kind_names=["fence"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = np.load(
                save_barriers(barriers, Path(tmp) / "b.npz"), allow_pickle=True
            )
            self.assertIn("collision_triangles", data.files)
            self.assertEqual(len(data["collision_triangles"]), 0)

    @staticmethod
    def _collision_ns(on_build=None):
        """`build_collision` and its helpers, with Blender stubbed out."""
        import re

        source = (REPO_ROOT / "blender" / "process.py").read_text(encoding="utf-8")
        namespace = {
            "np": np, "Path": Path, "log": lambda *a: None,
            "build_mesh_object": on_build or (lambda name, *a, **k: name),
        }
        for constant in ("TREE_TRUNK_FRACTIONS", "TREE_SHAPES_PER_KIND"):
            line = re.search(rf"\n{constant} = .*\n", source)
            assert line is not None, f"{constant} moved"
            exec(line.group(0), namespace)  # noqa: S102 - our own source
        for name in ("_octahedron_canopy", "_box", "_tree_dice", "_lobed_crown",
                     "_tree_instance", "_shape_lean",
                     "_collision_tree_mesh", "reused_colliders",
                     "build_collision"):
            body = re.search(
                rf"\ndef {name}\(.*?\n(?=\ndef |\n# ---)", source, re.S
            )
            assert body is not None, f"{name} moved"
            exec(body.group(0), namespace)  # noqa: S102 - our own source
        return namespace

    @staticmethod
    def _collision_work_dir(folder: Path):
        """A work dir with both inputs present, so "built nothing" can only
        mean the flag was honoured and not that there was nothing to read."""
        verts = np.array([[100010.0, 450010.0, 3.0], [100012.0, 450010.0, 3.0],
                          [100010.0, 450012.0, 4.0]])
        np.savez(
            folder / "barriers.npz", vertices=verts,
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
            collision_vertices=verts,
            collision_triangles=np.array([[0, 1, 2]], dtype=np.int32),
            collision_thickness_m=np.float64(0.30),
        )
        np.savez(
            folder / "trees.npz",
            xy=np.array([[100100.0, 450100.0], [100120.0, 450140.0]]),
            ground_z_nap=np.zeros(2), height_m=np.array([8.0, 19.0]),
            crown_radius_m=np.array([2.1, 4.9]),
            trunk_height_m=np.array([3.0, 5.0]),
        )
        return {
            "origin_rd": [100000.0, 450000.0],
            "ground_z_offset_nap": 0.0,
            "barriers": {"file": "barriers.npz"},
            "trees": {"file": "trees.npz"},
        }

    def test_nothing_is_built_unless_it_is_asked_for(self):
        """The proxies cost FBX size, and most imports want a picture.

        Both inputs are on disk, so this cannot pass by there being nothing to
        build -- which is how an earlier version of it passed with the flag
        check removed entirely.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            scene = self._collision_work_dir(folder)
            namespace = self._collision_ns(
                on_build=lambda *a, **k: self.fail(
                    "built a collider with export.collision off"
                )
            )
            for export_cfg in ({}, {"collision": False}):
                scene["export"] = export_cfg
                self.assertEqual(
                    namespace["build_collision"](scene, folder), {}
                )
            scene.pop("export")
            self.assertEqual(namespace["build_collision"](scene, folder), {})

    def test_both_proxies_are_built_when_it_is_asked_for(self):
        """The other half: the flag on, from the same inputs."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            scene = self._collision_work_dir(folder)
            scene["export"] = {"collision": True, "collision_tree_crowns": True}
            namespace = self._collision_ns()
            built = namespace["build_collision"](scene, folder)
            self.assertEqual(
                set(built), {"Collision_Barriers", "Collision_Trees"}
            )
            self.assertEqual(built["Collision_Barriers"], 1)
            self.assertEqual(built["Collision_Trees"], 28 * 2)

            scene["export"]["collision_tree_crowns"] = False
            trunks = namespace["build_collision"](scene, folder)
            self.assertEqual(trunks["Collision_Trees"], 16 * 2)

    def test_the_barrier_collider_lands_in_local_coordinates(self):
        """Everything in the FBX is relative to the bbox centre. A collider
        left in six-figure RD would sit 100 km from the fence it is for."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            scene = self._collision_work_dir(folder)
            scene["export"] = {"collision": True}
            seen = {}
            namespace = self._collision_ns(
                on_build=lambda name, vertices, *a, **k: seen.setdefault(
                    name, np.asarray(vertices)
                )
            )
            namespace["build_collision"](scene, folder)
            for name, vertices in seen.items():
                self.assertLess(
                    float(np.abs(vertices[:, :2]).max()), 10000.0,
                    f"{name} is still in RD coordinates",
                )

    def test_the_default_is_off_and_the_knobs_exist(self):
        from src.config import DEFAULTS

        self.assertFalse(DEFAULTS["export"]["collision"])
        self.assertTrue(DEFAULTS["export"]["collision_tree_crowns"])
        self.assertEqual(DEFAULTS["barriers"]["collision_thickness_m"], 0.30)

    def test_the_scene_carries_the_flags_to_blender(self):
        """The Blender stage only gets scene.json, so a knob that does not
        reach it is a knob that does nothing.

        This writes a real scene.json and reads the flags back out of it.
        Checking the source for the string instead let a mutation that renamed
        the key to "collisionx" pass, because the name still appeared on the
        line that read the config.
        """
        import json
        import tempfile

        for asked, crowns in ((True, True), (True, False), (False, True)):
            with tempfile.TemporaryDirectory() as tmp:
                scene = json.loads(
                    self._write_scene(
                        Path(tmp),
                        {"collision": asked, "collision_tree_crowns": crowns},
                    ).read_text(encoding="utf-8")
                )
                self.assertIs(scene["export"]["collision"], asked)
                self.assertIs(scene["export"]["collision_tree_crowns"], crowns)

    def test_a_scene_written_without_the_flags_defaults_to_off(self):
        """An older config, or one written by hand, must not turn this on."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            scene = json.loads(
                self._write_scene(Path(tmp), {}).read_text(encoding="utf-8")
            )
            self.assertIs(scene["export"]["collision"], False)
            self.assertIs(scene["export"]["collision_tree_crowns"], True)

    @staticmethod
    def _write_scene(work_dir: Path, extra_export: dict) -> Path:
        """A minimal real call to `write_scene_description`."""
        from src.config import DEFAULTS
        from src.elevation import TerrainResult
        from src.export import write_scene_description
        from src.geo import BBox, GeoContext
        from src.imagery import AerialResult

        bbox = BBox(100000.0, 450000.0, 100500.0, 450500.0)
        xs = np.linspace(bbox.xmin, bbox.xmax, 9)
        ys = np.linspace(bbox.ymin, bbox.ymax, 9)
        terrain = TerrainResult(
            heights=np.full((9, 9), 3.0, dtype=np.float32), xs=xs, ys=ys,
            bbox=bbox, center_z_nap=3.0, nodata_fraction_raw=0.0,
            filled_fraction=0.0, geotiff_path=None, coverage_id="dtm_05m",
            resolution_m=0.5,
        )
        aerial = AerialResult(
            path=work_dir / "aerial.png", bbox=bbox, size_px=256,
            layer_name="Actueel_ortho25", layer_title="Luchtfoto",
            source="wms",
        )
        facade = work_dir / "facade_00.png"
        facade.write_bytes(b"")
        (work_dir / "aerial.png").write_bytes(b"")
        return write_scene_description(
            name="test",
            geo=GeoContext.from_bbox(bbox),
            terrain=terrain,
            aerial=aerial,
            facade_paths=[(facade, None)],
            # The real defaults, not a hand-rolled subset: a stub that
            # happens to carry the keys this function reads today is a stub
            # that breaks the next time it reads one more.
            facade_cfg=DEFAULTS["facade"],
            buildings_cfg=DEFAULTS["buildings"],
            export_cfg={
                "fbx_name": "model.fbx", "aerial_name": "aerial.png",
                **extra_export,
            },
            work_dir=work_dir,
        )

    def test_the_metadata_beside_the_fbx_says_what_to_collide_with(self):
        """`blender_summary.json` stays in work/; the designer reads
        output/<area>/metadata.json. The guidance has to be in the second."""
        import tempfile

        import pipeline

        block = {
            "objects": {"Collision_Trees": 364204, "Collision_Barriers": 5120},
            "reuses_its_own_mesh": {"Terrain": "already simplified"},
            "note": "add a Mesh Collider and untick the renderer",
        }
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "blender_summary.json").write_text(
                json.dumps({"collision": block}), encoding="utf-8"
            )
            got = pipeline._collision_metadata(work)
        self.assertEqual(got, {"collision": block})

    def test_no_collision_block_when_there_are_no_colliders(self):
        """An empty "collision": {} in the metadata reads as a feature that
        failed. Nothing built means the key stays out."""
        import tempfile

        import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            # No Blender run at all, e.g. --skip-blender.
            self.assertEqual(pipeline._collision_metadata(work), {})
            for summary in ({}, {"collision": {}}, {"collision": {"objects": {}}}):
                (work / "blender_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                self.assertEqual(pipeline._collision_metadata(work), {})
            # And a half-written summary is not worth failing a run over.
            (work / "blender_summary.json").write_text("{ truncated", encoding="utf-8")
            self.assertEqual(pipeline._collision_metadata(work), {})

    def test_the_pipeline_puts_the_block_in_the_metadata(self):
        import inspect

        import pipeline

        source = inspect.getsource(pipeline.run)
        self.assertIn("_collision_metadata(work_dir)", source)

    def test_the_barrier_stats_report_the_collider(self):
        """metadata.json carries barriers.stats(), which is where the thickness
        actually used belongs -- a number the reader cannot otherwise know."""
        from src.barriers import BarrierSet

        bare = BarrierSet(triangles=np.zeros((4, 3), np.int32))
        self.assertNotIn("collision_triangles", bare.stats())

        with_proxy = BarrierSet(
            triangles=np.zeros((4, 3), np.int32),
            collision_triangles=np.zeros((6, 3), np.int32),
            collision_thickness_m=0.30,
        )
        stats = with_proxy.stats()
        self.assertEqual(stats["collision_triangles"], 6)
        self.assertEqual(stats["collision_thickness_m"], 0.30)

    def test_the_layers_without_a_proxy_say_what_to_use(self):
        """A designer with 25 objects and two colliders needs to be told what
        the other 23 are for."""
        import re

        source = (REPO_ROOT / "blender" / "process.py").read_text(encoding="utf-8")
        body = re.search(r"\ndef reused_colliders\(.*?\n(?=\ndef )", source, re.S)
        self.assertIsNotNone(body, "reused_colliders moved")
        namespace: dict = {}
        exec(body.group(0), namespace)  # noqa: S102 - our own source
        reused = namespace["reused_colliders"]()
        self.assertIn("Terrain", reused)
        # One object per building now, named after its BAG pand.
        self.assertIn("Pand_*", reused)
        for name, why in reused.items():
            self.assertGreater(len(why), 20, f"{name} has no reason given")


class TestBarrierHeightOverride(unittest.TestCase):
    """`barriers.heights` changes a height and nothing else.

    It used to change rather more. The override built a fresh BarrierStyle
    positionally, so inserting `see_through` into the middle of the dataclass
    shifted every later field along one: an overridden fence came out solid,
    and an overridden awning lost its 2.4 m lift *and* picked up the fence's
    cutout material, landing on the pavement as an alpha-clipped slab against
    the wall. Nothing failed, because the default for the knob is empty.
    """

    @staticmethod
    def _styled(name, table_key, table, override):
        """The real override, not a local copy of it.

        A first draft of these tests called `dataclasses.replace` here. All
        three passed against the broken code, because the broken code was in
        `build_barriers` and this helper never went near it.
        """
        from src.barriers import apply_height_override

        return apply_height_override(table[table_key], {name: override})

    def test_a_taller_fence_is_still_see_through(self):
        from src.barriers import LINE_STYLES

        got = self._styled(
            "fence", ("scheiding_lijn", "hek"), LINE_STYLES, 2.5
        )
        self.assertEqual(got.height_m, 2.5)
        self.assertIs(got.see_through, True, "a taller fence came out solid")
        self.assertEqual(got.lift_m, 0.0)
        self.assertEqual(got.material, "paint")

    def test_a_thinner_awning_still_hangs_off_the_wall(self):
        from src.barriers import AREA_STYLES

        got = self._styled(
            "awning", ("gebouwinstallatie", "luifel"), AREA_STYLES, 0.2
        )
        self.assertEqual(got.height_m, 0.2)
        self.assertEqual(
            got.lift_m, 2.4, "an overridden awning fell to the ground"
        )
        self.assertIs(
            got.see_through, False, "an awning is not a fence"
        )

    def test_the_override_is_the_only_thing_that_changes(self):
        """Whatever fields BarrierStyle grows next, they survive an override."""
        import dataclasses

        from src.barriers import AREA_STYLES, LINE_STYLES

        for key, table in (
            (("scheiding_lijn", "hek"), LINE_STYLES),
            (("gebouwinstallatie", "luifel"), AREA_STYLES),
            (("vegetatieobject_lijn", "haag"), LINE_STYLES),
        ):
            before = table[key]
            after = self._styled(before.name, key, table, 9.5)
            for f in dataclasses.fields(before):
                if f.name == "height_m":
                    continue
                self.assertEqual(
                    getattr(after, f.name), getattr(before, f.name),
                    f"overriding the height of {before.name} also changed "
                    f"{f.name}",
                )

    def test_build_barriers_applies_overrides_through_it(self):
        """The tests above reach the real override only while `build_barriers`
        still calls it, and the whole fault was logic living in a closure that
        no test could see."""
        import inspect

        from src import barriers

        source = inspect.getsource(barriers.build_barriers)
        self.assertIn("apply_height_override(", source)
        self.assertNotIn(
            "BarrierStyle(", source,
            "build_barriers constructs a style of its own again",
        )

    def test_an_unmapped_type_stays_unmapped(self):
        from src.barriers import apply_height_override

        self.assertIsNone(apply_height_override(None, {"fence": 2.0}))

    def test_no_override_hands_back_the_table_entry(self):
        """Untouched means untouched: the same object, not a copy of it."""
        from src.barriers import LINE_STYLES, apply_height_override

        style = LINE_STYLES[("scheiding_lijn", "muur")]
        self.assertIs(apply_height_override(style, {}), style)
        self.assertIs(apply_height_override(style, {"fence": 9.0}), style)


class TestStreetDetail(unittest.TestCase):
    """The small things a street is not a street without.

    Three thousand walls and fences per square kilometre, and they were all
    being fetched and thrown away. What these check is not that the numbers
    are right -- the BGT does not survey how tall a wall is, so the heights
    are an assumption -- but that the geometry closes, sits on the ground, and
    comes from the field the data actually uses.
    """

    @staticmethod
    def _slope(gradient=0.05, bump=0.0):
        def sample(x, y):
            x = np.asarray(x, dtype=float)
            return gradient * x + bump * np.sin(x / 7.0)
        return sample

    @staticmethod
    def _closed(vertices, triangles):
        """Boundary edges, non-manifold edges. A closed prism has none of either."""
        edges = np.sort(
            np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                            triangles[:, [2, 0]]]),
            axis=1,
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return int((counts == 1).sum()), int((counts > 2).sum())

    def test_the_type_comes_from_whichever_field_carries_it(self):
        """The bug that made the first run of this draw nothing.

        It is not the same field twice. scheiding and overigbouwwerk put the
        value in `type` and leave `plus_type` explicitly null; gebouwinstallatie
        and vegetatieobject do the reverse, carrying a real `plus_type` beside a
        useless `type: niet-bgt`. Reading only `plus_type` fetched 1563 fences
        and 1450 walls over one square kilometre and built none of them.
        """
        from src.barriers import feature_type

        # A fence, as scheiding_lijn really returns it.
        self.assertEqual(
            feature_type({"plus_type": None, "plus_type_leeg": "waardeOnbekend",
                          "type": "hek"}),
            "hek",
        )
        # Entrance steps, as gebouwinstallatie really returns them.
        self.assertEqual(
            feature_type({"plus_type": "toegangstrap", "type": "niet-bgt"}),
            "toegangstrap",
        )
        # A surveyed object the standard has no type for is not a type.
        self.assertEqual(feature_type({"plus_type": None, "type": "niet-bgt"}), "")
        self.assertEqual(feature_type({}), "")

    def test_a_swept_fence_is_a_closed_prism(self):
        """A wall modelled as a plane is a hole from one side in anything with
        backface culling on, which is every engine."""
        from src.barriers import LINE_STYLES, _sweep_line

        points = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 6.0], [34.0, 6.0]])
        style = LINE_STYLES[("scheiding_lijn", "hek")]
        vertices, triangles, _uv = _sweep_line(
            points, lambda x, y: np.zeros(len(np.atleast_1d(x))), style
        )

        boundary, nonmanifold = self._closed(vertices, triangles)
        self.assertEqual(boundary, 0, f"{boundary} open edges: the prism is a tube")
        self.assertEqual(nonmanifold, 0)
        # And it has the thickness it claims, not zero.
        width = vertices[:, :2].max(axis=0) - vertices[:, :2].min(axis=0)
        self.assertGreater(min(width), 0.0)

    def test_a_barrier_keeps_its_height_up_a_slope(self):
        """Sampling the ground once and extruding from there gives a wall that
        levels off on a hill, which on a canal bank means it disappears."""
        from src.barriers import EMBED_M, LINE_STYLES, _drape_points, _sweep_line

        style = LINE_STYLES[("scheiding_lijn", "muur")]
        line = np.array([[0.0, 0.0], [60.0, 0.0]])
        points, _profile = _drape_points(line, self._slope(0.05), 1.0, 0.05)
        vertices, _, _uv = _sweep_line(points, self._slope(0.05), style)

        n = len(points)
        # The four rails are stacked in order: left-low, left-high, ...
        low, high = vertices[:n, 2], vertices[n : 2 * n, 2]
        heights = high - low
        self.assertAlmostEqual(float(heights.min()), style.height_m + EMBED_M, places=6)
        self.assertAlmostEqual(float(heights.max()), style.height_m + EMBED_M, places=6)
        # And it really did climb, rather than staying flat.
        self.assertGreater(float(high.max() - high.min()), 2.0)

    def test_the_drape_spends_vertices_where_the_ground_bends(self):
        """Cutting every fence at a fixed step holds straight lines straight at
        full price: over one square kilometre it was 203,020 triangles against
        88,324 for the same walls followed to five centimetres."""
        from src.barriers import _drape_points

        line = np.array([[0.0, 0.0], [200.0, 0.0]])
        flat, _ = _drape_points(line, self._slope(0.02, bump=0.0), 1.0, 0.05)
        bumpy, _ = _drape_points(line, self._slope(0.02, bump=0.5), 1.0, 0.05)
        uniform, _ = _drape_points(line, self._slope(0.02, bump=0.0), 1.0, 0.0)

        self.assertLess(
            len(flat), 10,
            f"{len(flat)} vertices to hold a straight line straight",
        )
        self.assertGreater(
            len(bumpy), 4 * len(flat),
            "rolling ground got no more vertices than flat ground",
        )
        self.assertGreater(len(uniform), 100, "the fixed-step path is the baseline")

    def test_the_drape_stays_inside_its_tolerance(self):
        """Thinning is only allowed to cost what it was told it could cost."""
        from src.barriers import _drape_points

        sampler = self._slope(0.02, bump=0.5)
        line = np.array([[0.0, 0.0], [200.0, 0.0]])
        kept, height = _drape_points(line, sampler, 1.0, 0.05)

        # Walk the kept polyline at a fine step and compare to the real ground.
        worst = 0.0
        for (x0, _), (x1, _), z0, z1 in zip(kept[:-1], kept[1:], height[:-1], height[1:]):
            xs = np.linspace(x0, x1, 25)
            along = (xs - x0) / max(x1 - x0, 1e-9)
            worst = max(worst, float(np.abs(z0 + (z1 - z0) * along - sampler(xs, xs)).max()))
        self.assertLess(worst, 0.08, f"the barrier strays {worst:.3f} m from the ground")

    def test_an_awning_hangs_off_the_facade(self):
        """Drawn from the ground it is a solid block against the wall, which is
        worse than leaving it out."""
        from src.barriers import AREA_STYLES, _extrude_rings

        ring = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 1.4], [0.0, 1.4]])
        style = AREA_STYLES[("gebouwinstallatie", "luifel")]
        vertices, triangles, _uv = _extrude_rings([ring], lambda x, y: np.full(len(x), 3.0), style)

        self.assertGreater(
            float(vertices[:, 2].min()) - 3.0, 2.0,
            "the awning is sitting on the pavement",
        )
        # Its own thickness, not its thickness plus the ground embed. Burying a
        # 12 cm awning by 15 cm more than doubles it and hangs it too low.
        thickness = float(vertices[:, 2].max() - vertices[:, 2].min())
        self.assertAlmostEqual(thickness, style.height_m, places=6)
        # Floating, so it needs a floor as well as a lid.
        boundary, nonmanifold = self._closed(vertices, triangles)
        self.assertEqual(boundary, 0, "the awning has no underside")
        self.assertEqual(nonmanifold, 0)

    def test_a_wall_polygon_follows_the_ground_under_it(self):
        """One height for a whole outline fails where these objects live: a quay
        wall runs along a canal bank, and a single median buried one end by
        0.65 m on a test surface."""
        from src.barriers import AREA_STYLES, _extrude_rings

        ring = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 0.4], [0.0, 0.4]])
        style = AREA_STYLES[("scheiding_vlak", "kademuur")]
        sampler = self._slope(0.05)
        vertices, _, _uv = _extrude_rings([ring], sampler, style)

        below = vertices[:, 2] - sampler(vertices[:, 0], vertices[:, 1])
        self.assertLess(
            float(-below.min()), 0.30,
            f"a corner is {-below.min():.2f} m underground",
        )
        self.assertGreater(
            float(vertices[:, 2].max() - vertices[:, 2].min()), 1.9,
            "the wall came out level across a two-metre rise",
        )

    def test_every_furniture_kind_has_a_shape(self):
        """A type that reaches the export with no proxy is an invisible object
        that still costs a fetch. A type with no *name* is worse."""
        import re

        from src.furniture import KIND_NAMES, WANTED

        source = (REPO_ROOT / "blender" / "process.py").read_text()
        namespace = {"np": np}
        for name in ("_box", "_furniture_pieces"):
            body = re.search(rf"\ndef {name}\(.*?\n(?=\ndef )", source, re.S)
            self.assertIsNotNone(body, f"{name} moved")
            exec(body.group(0), namespace)  # noqa: S102 - our own source, in a test

        uv = {n: np.array([0.5, 0.5]) for n in
              ("metal", "wood", "brick", "concrete", "hedge", "paint", "stone", "glass")}
        for kind in sorted(KIND_NAMES):
            pieces = namespace["_furniture_pieces"](kind, 0.0, 0.0, 0.0, 0.3, uv, {})
            self.assertTrue(pieces, f"{KIND_NAMES[kind]} has no proxy shape")
            corners = np.vstack([p[0][0] for p in pieces])
            self.assertAlmostEqual(
                float(corners[:, 2].min()), 0.0, places=6,
                msg=f"{KIND_NAMES[kind]} does not stand on the ground",
            )
            self.assertLess(
                float(corners[:, 2].max()), 12.0,
                f"{KIND_NAMES[kind]} is {corners[:, 2].max():.0f} m tall",
            )

        # Every mapped BGT type lands on a kind that has a name.
        for collection, mapping in WANTED.items():
            for plus_type, kind in mapping.items():
                self.assertIn(kind, KIND_NAMES, f"{collection}/{plus_type}")

    def test_an_unmapped_type_draws_nothing(self):
        """The BGT is full of "niet-bgt". Drawing something specific for
        "unknown" is how a model ends up with seventy identical sheds."""
        import re

        source = (REPO_ROOT / "blender" / "process.py").read_text()
        namespace = {"np": np}
        for name in ("_box", "_furniture_pieces"):
            body = re.search(rf"\ndef {name}\(.*?\n(?=\ndef )", source, re.S)
            exec(body.group(0), namespace)  # noqa: S102 - our own source, in a test

        uv = {n: np.array([0.5, 0.5]) for n in
              ("metal", "wood", "brick", "concrete", "hedge", "paint", "stone", "glass")}
        self.assertEqual(namespace["_furniture_pieces"](99, 0, 0, 0, 0, uv, {}), [])

    def test_the_atlas_has_a_patch_for_every_material_in_use(self):
        """The Blender stage addresses the atlas by name through the scene file.
        A style naming a patch that is not there would take the fallback UV and
        paint a hedge in dark metal."""
        from src.barriers import AREA_STYLES, LINE_STYLES
        from src.facade import FURNITURE_PATCHES, furniture_uv

        available = {name for name, *_rest in FURNITURE_PATCHES}
        for style in (*LINE_STYLES.values(), *AREA_STYLES.values()):
            self.assertIn(style.material, available, style.name)
            u, v = furniture_uv(style.material)
            self.assertTrue(0.0 < u < 1.0 and 0.0 < v < 1.0)

    def test_a_wall_on_a_bank_does_not_bury_itself(self):
        """A wall has width, and the mitre at a corner widens it further -- up
        to two and a half times the half-thickness, which on a half-metre quay
        wall is half a metre sideways.

        That is nothing on a pavement and a lot on a canal bank, which is
        exactly where quay walls are. Sampling the ground along the centreline
        and building the corners from it put a wall 0.96 m underground on a
        real run.
        """
        from src.barriers import EMBED_M, LINE_STYLES, _sweep_line

        # A bank: flat on one side, dropping steeply on the other.
        def bank(x, y):
            y = np.asarray(y, dtype=float)
            return np.clip(y, 0.0, 3.0) - 3.0

        style = LINE_STYLES[("scheiding_lijn", "kademuur")]
        # Along the top of the bank, with a corner to engage the mitre.
        line = np.array([[0.0, 3.0], [20.0, 3.0], [26.0, 9.0]])
        vertices, _, _uv = _sweep_line(line, bank, style)

        # The four rails are stacked in order: left-low, left-high, right-high,
        # right-low. Only the two low ones are meant to touch the ground.
        n = len(line)
        base = np.concatenate([vertices[:n], vertices[3 * n : 4 * n]])
        offset = base[:, 2] - bank(base[:, 0], base[:, 1])
        self.assertGreaterEqual(
            float(offset.min()), -(EMBED_M + 0.05),
            f"a corner is {-offset.min():.2f} m underground, against a "
            f"{EMBED_M:.2f} m embed",
        )
        # And not floating either, which is the same mistake the other way up:
        # sampling the middle and building the corners from it lifts the low
        # side clear of the bank instead of burying the high one.
        self.assertLessEqual(
            float(offset.max()), 0.0,
            f"a corner floats {offset.max():.2f} m above the bank",
        )
        # The top is level across the width and taken from the higher side, so
        # the wall is never shorter than it claims on either bank.
        top = vertices[n : 2 * n, 2]
        higher = np.maximum(
            bank(vertices[:n, 0], vertices[:n, 1]),
            bank(vertices[3 * n : 4 * n, 0], vertices[3 * n : 4 * n, 1]),
        )
        self.assertGreaterEqual(
            float((top - higher).min()), style.height_m - 1e-6,
            f"the wall is only {(top - higher).min():.2f} m tall on its high "
            f"side, against {style.height_m:.2f} m asked for",
        )

    def test_a_wall_is_cut_at_the_edge_of_the_area(self):
        """A quay wall follows its canal, and the canal does not stop at the
        bbox. Keeping any polygon whose middle was inside took the model 227 m
        past its own terrain."""
        from src.barriers import AREA_STYLES, _extrude_rings, _prepare_rings
        from src.geo import BBox

        bbox = BBox(0.0, 0.0, 100.0, 100.0)
        # A wall running from inside the area to 200 m north of it.
        # Centre inside the area, reaching 200 m past it: exactly the shape a
        # centre test waves through.
        ring = np.array([[49.0, -50.0], [51.0, -50.0], [51.0, 200.0], [49.0, 200.0]])
        self.assertTrue(
            bbox.ymin <= ring[:, 1].mean() <= bbox.ymax,
            "the fixture's centre is outside, so a centre test would catch it",
        )

        rings = _prepare_rings([ring], bbox)
        self.assertEqual(len(rings), 1, "the wall was dropped rather than cut")
        vertices, _, _uv = _extrude_rings(
            rings, lambda x, y: np.zeros(len(x)),
            AREA_STYLES[("scheiding_vlak", "kademuur")],
        )
        self.assertLessEqual(
            float(vertices[:, 1].max()), bbox.ymax + 1e-6,
            f"the wall reaches {vertices[:, 1].max():.0f} m into a "
            f"{bbox.ymax:.0f} m area",
        )

    def test_a_long_wall_follows_the_ground_between_its_corners(self):
        """A survey draws a thirty-metre wall as a four-vertex rectangle.

        Sampling the ground only at those corners leaves the base a straight
        chord over whatever the ground does in between: measured over the
        Gelderland square, one ring edge in five departed from its chord by
        more than the fifteen-centimetre embed and the worst by 3.9 m. The
        swept lines had always been cut up before sampling; polygons were the
        ones left out.
        """
        from src.barriers import AREA_STYLES, EMBED_M, _extrude_rings

        def hummocky(x, y):
            x = np.asarray(x, dtype=float)
            return 1.5 * np.sin(x / 5.0)

        # A 40 m wall, four corners, over ground that rolls underneath it.
        ring = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 0.5], [0.0, 0.5]])
        style = AREA_STYLES[("scheiding_vlak", "muur")]

        coarse, _, _uv = _extrude_rings([ring], hummocky, style, step_m=0.0)
        fine, _, _uv = _extrude_rings([ring], hummocky, style, step_m=1.0)

        def worst_float(vertices):
            """How far the base *surface* rises above the ground.

            Measured along the edges, not at the vertices: the whole fault is
            that the base is a straight chord between two corners that are
            themselves correct, so a vertex sample cannot see it.
            """
            half = len(vertices) // 2
            base = vertices[:half]
            worst = -np.inf
            for start, end in zip(base, np.roll(base, -1, axis=0)):
                t = np.linspace(0, 1, 60)[:, None]
                along = start + (end - start) * t
                gap = along[:, 2] - hummocky(along[:, 0], along[:, 1])
                worst = max(worst, float(gap.max()))
            return worst

        self.assertGreater(
            worst_float(coarse), 0.5,
            "the fixture's ground is too flat to show the difference",
        )
        self.assertLessEqual(
            worst_float(fine), 0.0,
            f"the wall still floats {worst_float(fine):.2f} m above the ground",
        )
        gap = fine[:, 2] - hummocky(fine[:, 0], fine[:, 1])
        self.assertGreaterEqual(
            float(gap.min()), -(EMBED_M + 1e-6),
            f"and buries itself {-gap.min():.2f} m at the worst corner",
        )

    def test_an_awning_clears_the_ground_at_every_corner(self):
        """An awning is flat, and its footprint can straddle a slope. Taking
        the median ground looks like the fair choice and is not: it left one
        half buried by 1.43 m on the Gelderland square. Too high at one end is
        a thing nobody notices; underground is not."""
        from src.barriers import AREA_STYLES, _extrude_rings

        # Steep enough that the median really does bury the high end: the
        # footprint has to span more than twice the lift, or the two rules
        # agree and the fixture proves nothing.
        def slope(x, y):
            return np.asarray(x, dtype=float) * 0.8

        ring = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 2.0], [0.0, 2.0]])
        style = AREA_STYLES[("gebouwinstallatie", "luifel")]
        vertices, _, _uv = _extrude_rings([ring], slope, style, step_m=1.0)

        above = vertices[:, 2] - slope(vertices[:, 0], vertices[:, 1])
        self.assertGreater(
            float(above.min()), 0.0,
            f"the awning is {-above.min():.2f} m underground at its low end",
        )
        # Still flat: an awning that ripples with the pavement is worse than
        # one that does not.
        self.assertLess(
            float(vertices[:, 2].max() - vertices[:, 2].min()),
            style.height_m + 1e-6,
            "the awning follows the slope instead of hanging level",
        )

    def test_a_fence_is_mostly_holes(self):
        """Drawn as a solid slab a railing is a hoarding, and a fence and a
        hedge come out the same object."""
        import tempfile

        from PIL import Image

        from src.facade import generate_fence_texture

        with tempfile.TemporaryDirectory() as folder:
            path = generate_fence_texture(Path(folder), size_px=128)
            image = Image.open(path)
            self.assertEqual(image.mode, "RGBA", "the fence has no alpha at all")
            alpha = np.asarray(image)[:, :, 3]

        open_fraction = float((alpha <= 127).mean())
        self.assertGreater(open_fraction, 0.4, "the fence is nearly solid")
        self.assertLess(open_fraction, 0.85, "the fence is nearly nothing")
        # Uprights: a row across the middle alternates between bar and gap
        # several times, which a plain slab or a plain hole does not.
        middle = alpha[int(len(alpha) * 0.3)] > 127
        crossings = int(np.count_nonzero(np.diff(middle.astype(np.int8))))
        self.assertGreaterEqual(
            crossings, 6, f"only {crossings} bar edges across the tile",
        )

    def test_only_the_fences_are_see_through(self):
        """A wall is not. Nor is a quay wall, a noise barrier or a hedge."""
        from src.barriers import AREA_STYLES, LINE_STYLES

        by_name = {
            style.name: style
            for style in (*LINE_STYLES.values(), *AREA_STYLES.values())
        }
        self.assertTrue(by_name["fence"].see_through)
        for solid in ("wall", "quay_wall", "noise_barrier", "hedge", "awning"):
            self.assertFalse(
                by_name[solid].see_through, f"{solid} should not be see-through"
            )

    def test_the_fence_texture_tiles_along_the_run(self):
        """Stretched instead of tiled, a four-hundred-metre fence gets eight
        uprights. Up the height it must not tile: one fence is one texture, top
        rail to bottom rail, or most fences simply have no top."""
        from src.barriers import LINE_STYLES, _sweep_line

        style = LINE_STYLES[("scheiding_lijn", "hek")]
        flat = lambda x, y: np.zeros(len(np.atleast_1d(x)))

        short = np.array([[0.0, 0.0], [10.0, 0.0]])
        long = np.array([[0.0, 0.0], [200.0, 0.0]])
        _, _, short_uv = _sweep_line(short, flat, style)
        _, _, long_uv = _sweep_line(long, flat, style)

        self.assertAlmostEqual(float(short_uv[:, 0].max()), 10.0, delta=0.5)
        self.assertAlmostEqual(float(long_uv[:, 0].max()), 200.0, delta=0.5)
        for uv in (short_uv, long_uv):
            self.assertAlmostEqual(float(uv[:, 1].min()), 0.0, places=6)
            self.assertAlmostEqual(float(uv[:, 1].max()), 1.0, places=6)

    def test_the_settings_reach_the_config(self):
        from src.config import DEFAULTS

        self.assertTrue(DEFAULTS["barriers"]["enabled"])
        self.assertEqual(DEFAULTS["barriers"]["fence_tile_m"], 2.5)
        self.assertEqual(DEFAULTS["barriers"]["drape_tolerance_m"], 0.05)
        self.assertEqual(DEFAULTS["furniture"]["flagpole_height_m"], 7.0)

    def test_a_height_override_reaches_the_geometry(self):
        """The heights are an assumption. Putting them in the config is the
        whole point, so the config has to actually win."""
        from src.barriers import LINE_STYLES, BarrierStyle

        style = LINE_STYLES[("scheiding_lijn", "geluidsscherm")]
        override = BarrierStyle(style.name, 6.0, style.thickness_m,
                                style.material, style.lift_m)
        self.assertNotEqual(override.height_m, style.height_m)
        self.assertEqual(override.name, "noise_barrier")


class TestUIServer(unittest.TestCase):
    """The UI writes a config and shells out to pipeline.py, so the translation
    from form fields to config is the part worth pinning down."""

    def setUp(self):
        sys.path.insert(0, str(REPO_ROOT / "ui"))
        import server

        self.server = server

    def test_form_payload_becomes_a_loadable_config(self):
        payload = {
            "name": "somewhere",
            "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
            "size_px": 2048,
            "terrain_grid_m": 4.0,
            "facade_variants": 4,
            "clip_mode": "intersect",
        }
        config = self.server.build_config(payload)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            loaded = load_config(path)

        self.assertEqual(loaded.name, "somewhere")
        self.assertEqual(loaded.bbox.width, 1000.0)
        self.assertEqual(loaded.aerial["size_px"], 2048)
        self.assertEqual(loaded.terrain["grid_m"], 4.0)
        self.assertEqual(loaded.facade["variants"], 4)
        self.assertEqual(loaded.buildings["clip_mode"], "intersect")

    def test_defaults_apply_when_the_form_omits_fields(self):
        config = self.server.build_config(
            {
                "name": "bare",
                "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
            }
        )
        self.assertEqual(config["detail"], "standard")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            loaded = load_config(path)
        # Standard: 25 cm a pixel over a kilometre, a 2 m ground grid.
        self.assertEqual(loaded.aerial["size_px"], 4000)
        self.assertEqual(loaded.terrain["grid_m"], 2.0)
        self.assertEqual(loaded.terrain["ahn_model"], "DTM")
        self.assertEqual(loaded.buildings["lod"], "2.2")

    def test_the_collision_toggle_reaches_a_loadable_config(self):
        """The checkbox has to survive the whole trip: form payload, config
        file, `load_config`, and on into scene.json."""
        payload = {
            "name": "sim",
            "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
            "collision": True,
            "collision_tree_crowns": False,
            "collision_thickness_m": 0.45,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(self.server.build_config(payload)))
            loaded = load_config(path)
        self.assertTrue(loaded.export["collision"])
        self.assertFalse(loaded.export["collision_tree_crowns"])
        self.assertEqual(loaded.barriers["collision_thickness_m"], 0.45)

    def test_the_facade_variety_picker_reaches_the_config(self):
        """The form asks how many buildings may share a wall, because that is
        the question someone can answer; how many textures that costs depends
        on what is in the area."""
        payload = {
            "name": "street",
            "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
            "facade_variants": 5,
            "facade_buildings_per_look": 40,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(self.server.build_config(payload)))
            loaded = load_config(path)
        self.assertEqual(loaded.facade["buildings_per_look"], 40)
        self.assertEqual(loaded.facade["looks_per_era"], 12)
        self.assertEqual(loaded.facade["variants"], 5)

    def test_turning_the_variety_off_reaches_the_config(self):
        """The one choice a budget-shaped control could silently overrule."""
        config = self.server.build_config(
            {
                "name": "flat",
                "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
                "facade_buildings_per_look": 0,
            }
        )
        self.assertEqual(config["facade"]["looks_per_era"], 1)

    def test_the_older_form_field_still_works(self):
        config = self.server.build_config(
            {
                "name": "old",
                "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
                "facade_looks_per_era": 3,
            }
        )
        self.assertEqual(config["facade"]["looks_per_era"], 3)

    def test_the_deshadow_toggle_reaches_the_config(self):
        for asked, expect in ((True, 1.0), (False, 0.0)):
            config = self.server.build_config({
                "name": "sun",
                "bbox": {"xmin": 136000, "ymin": 455000,
                         "xmax": 137000, "ymax": 456000},
                "deshadow": 1.0 if asked else 0,
            })
            self.assertEqual(config["aerial"]["deshadow"], expect)

    def test_deshadowing_is_on_when_the_form_says_nothing(self):
        config = self.server.build_config({
            "name": "bare",
            "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            loaded = load_config(path)
        self.assertEqual(loaded.aerial["deshadow"], 1.0)
        self.assertEqual(loaded.trees["ground_shade"], 0.35)

    def test_collision_is_off_when_the_form_does_not_ask(self):
        config = self.server.build_config(
            {
                "name": "bare",
                "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
            }
        )
        self.assertFalse(config["export"]["collision"])
        loaded = self._load(config)
        self.assertTrue(loaded.export["collision_tree_crowns"])
        self.assertEqual(loaded.barriers["collision_thickness_m"], 0.30)

    def _load(self, config):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            return load_config(path)

    def test_a_detail_level_sets_photo_ground_and_walls(self):
        from src.config import DETAIL_PRESETS

        for level, preset in DETAIL_PRESETS.items():
            loaded = self._load(self.server.build_config({
                "name": "lvl", "detail": level,
                "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
            }))
            self.assertEqual(loaded.terrain["grid_m"], preset["grid_m"], level)
            self.assertEqual(loaded.facade["texture_px"], preset["facade_px"], level)
            # Ground resolution, so the pixel count follows the area.
            want = int(round(1000 / (preset["aerial_cm"] / 100) / 4)) * 4
            self.assertEqual(loaded.aerial["size_px"], want, level)

    def test_area_summary_is_none_for_unknown_area(self):
        self.assertIsNone(self.server.area_summary("no_such_area_xyz"))

    def test_quality_fields_reach_the_config(self):
        config = self.server.build_config(
            {
                "name": "q",
                "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
                "size_px": 12288,
                "terrain_grid_m": 1.0,
                "facade_texture_px": 2048,
                "facade_normal_map": False,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            loaded = load_config(path)

        self.assertEqual(loaded.aerial["size_px"], 12288)
        self.assertEqual(loaded.terrain["grid_m"], 1.0)
        self.assertEqual(loaded.facade["texture_px"], 2048)
        self.assertFalse(loaded.facade["normal_map"])

    def test_estimate_grows_with_area_pixels_and_features(self):
        estimate = self.server.estimate_seconds
        small = {"bbox": {"xmin": 0, "ymin": 0, "xmax": 500, "ymax": 500}, "size_px": 2048}
        large = {"bbox": {"xmin": 0, "ymin": 0, "xmax": 1000, "ymax": 1000}, "size_px": 2048}
        self.assertGreater(estimate(large)["seconds"], estimate(small)["seconds"])

        coarse = dict(large, size_px=2048)
        fine = dict(large, size_px=8192)
        self.assertGreater(estimate(fine)["seconds"], estimate(coarse)["seconds"])

        # Previews cost more than every data stage put together.
        with_preview = dict(large, preview=True)
        without = dict(large, preview=False)
        self.assertGreater(
            estimate(with_preview)["seconds"], 1.5 * estimate(without)["seconds"]
        )

        # Turning a source off has to make the estimate smaller.
        for flag in ("trees", "furniture", "usage", "water"):
            self.assertLess(
                estimate({**without, flag: False, "land_cover": flag != "water"})["seconds"],
                estimate(without)["seconds"],
                f"disabling {flag} did not reduce the estimate",
            )

    def test_estimate_matches_measured_runs(self):
        """Calibrated against real runs over Utrecht; keep it honest."""
        estimate = self.server.estimate_seconds

        def seconds(side, px, **kw):
            payload = {
                "bbox": {"xmin": 0, "ymin": 0, "xmax": side, "ymax": side},
                "size_px": px,
            }
            payload.update(kw)
            return estimate(payload)["seconds"]

        # The Utrecht runs predate the terrain mesh stage, so its own measured
        # cost -- 48 s a square kilometre over Bronbeek -- is added to them.
        mesh = 48.0
        measured = [
            (seconds(1000, 4096, preview=True), 468 + mesh),
            (seconds(1000, 4096, preview=False), 218 + mesh),
            (seconds(500, 2048, preview=False), 75 + mesh / 4),
        ]
        for predicted, actual in measured:
            self.assertLess(
                abs(predicted - actual) / actual,
                0.30,
                f"estimate {predicted:.0f}s is more than 30% off the measured {actual}s",
            )

    def test_speed_factor_is_bounded(self):
        speed, samples = self.server.measured_speed_factor()
        self.assertGreaterEqual(speed, 0.25)
        self.assertLessEqual(speed, 4.0)
        self.assertGreaterEqual(samples, 0)

    def test_progress_follows_the_step_headers(self):
        """The pipeline already prints its stages; progress reads those."""
        job = self.server.Job("id", "area", {}, estimate=100.0)
        job.status = "running"

        job.log("12:00:00 INFO pipeline | Step 3/10  ground surfaces (BGT water)")
        snapshot = job.snapshot(0)
        self.assertEqual(snapshot["step"], 3)
        self.assertEqual(snapshot["total_steps"], 10)
        self.assertEqual(snapshot["stage"], "ground surfaces (BGT water)")
        self.assertGreater(snapshot["fraction"], 0.0)
        self.assertLess(snapshot["fraction"], 1.0)

        # Previews run after the last numbered stage and dominate the runtime,
        # so they get named rather than looking like a stall.
        job.log("12:05:00 INFO pipeline | running preview.py via bpy")
        self.assertEqual(job.snapshot(0)["stage"], "rendering previews")

        job.status = "done"
        job.finished = job.started + 42
        self.assertEqual(job.snapshot(0)["fraction"], 1.0)

    def test_progress_stays_indeterminate_before_the_first_stage(self):
        job = self.server.Job("id", "area", {}, estimate=100.0)
        job.status = "starting"
        snapshot = job.snapshot(0)
        self.assertEqual(snapshot["total_steps"], 0)
        self.assertEqual(snapshot["fraction"], 0.0)

    def test_oversampling_past_the_source_only_warns(self):
        """Asking for more than the source holds is allowed, but flagged."""
        config = self.server.build_config(
            {
                "name": "q",
                "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
                "size_px": 16384,     # finer than the 8 cm ortho
                "terrain_grid_m": 0.25,  # finer than the 0.5 m DTM
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            with self.assertLogs("src.config", level="WARNING") as captured:
                loaded = load_config(path)

        messages = " ".join(captured.output)
        self.assertIn("finer than the 8 cm source", messages)
        self.assertIn("finer than the 0.50 m AHN source", messages)
        self.assertEqual(loaded.terrain["grid_m"], 0.25)
        self.assertEqual(loaded.aerial["size_px"], 16384)


class _PanelScan(HTMLParser):
    """Reads the panel the way a browser does.

    A regex over the markup is not good enough here: the help text is HTML in
    an attribute, so it contains its own angle brackets and stops any
    `<span[^>]*>` pattern in the middle of a tip.
    """

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
            "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.tips: list[str | None] = []
        self.explained: set[str] = set()
        self.details_open: list[str] = []
        self.step_numbers: list[str] = []
        self.step_titles: list[str] = []
        self._stack: list[dict] = []
        self._collect: str | None = None

    # A help marker explains whichever group it sits in directly: the <label>
    # that names a control, or the row that holds a checkbox. Counting it
    # against every enclosing element instead would let one "?" in a fieldset
    # vouch for every control in it.
    def _group(self):
        return self._stack[-1] if self._stack else None

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        classes = (d.get("class") or "").split()
        if "id" in d:
            self.ids.append(d["id"])
        if tag == "details" and "open" in d:
            self.details_open.append(str(d))

        group = self._group()
        if "help" in classes:
            self.tips.append(d.get("data-tip"))
            if group:
                group["helps"] += 1
        if tag in ("input", "select", "textarea") and "id" in d and group:
            group["controls"].append(d["id"])
        if "stepnum" in classes or "steptitle" in classes:
            self._collect = "stepnum" if "stepnum" in classes else "steptitle"

        if tag not in self.VOID:
            controls = [d["for"]] if tag == "label" and "for" in d else []
            self._stack.append({"tag": tag, "controls": controls, "helps": 0})

    def handle_data(self, data):
        if self._collect == "stepnum":
            self.step_numbers.append(data.strip())
        elif self._collect == "steptitle":
            self.step_titles.append(data.strip())

    def handle_endtag(self, tag):
        if tag in self.VOID or not self._stack:
            return
        self._collect = None
        frame = self._stack.pop()
        if frame["helps"]:
            self.explained.update(frame["controls"])


class TestUIPage(unittest.TestCase):
    """The page is one file holding both the markup and the script that drives
    it, so an edit to either can move the two apart. Nothing here renders the
    page — these are the joins that break silently when it is rearranged."""

    @classmethod
    def setUpClass(cls):
        cls.html = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
        cls.page = _PanelScan()
        cls.page.feed(cls.html)

    def test_every_element_the_script_reaches_for_exists(self):
        """`$("x")` on a renamed element throws and stops the page dead, and
        the browser reports it nowhere the user is going to look."""
        wanted = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', self.html))
        self.assertTrue(wanted, "found no element lookups; the pattern moved")
        missing = sorted(wanted - set(self.page.ids))
        self.assertEqual(missing, [], f"the script looks up ids that are gone: {missing}")

    def test_no_id_is_used_twice(self):
        """Two elements sharing an id means half the script talks to the wrong
        one, and only sometimes."""
        used = self.page.ids
        repeated = sorted({i for i in used if used.count(i) > 1})
        self.assertEqual(repeated, [], f"duplicate ids: {repeated}")

    def test_every_help_marker_says_something(self):
        """A "?" that opens an empty bubble is worse than no "?" at all."""
        tips = self.page.tips
        self.assertGreaterEqual(len(tips), 7, "help markers have gone missing")
        for tip in tips:
            self.assertIsNotNone(tip, "a help marker carries no data-tip")
            self.assertGreater(len(tip), 40, f"a help marker only says {tip!r}")

    def test_the_settings_that_matter_are_explained(self):
        """Every control a first-time user has to answer carries its own help,
        rather than one note at the top of the group."""
        for control in ("q", "size", "name", "detail", "preview", "collision"):
            self.assertIn(
                control, self.page.explained, f"the control {control} has no help marker"
            )

    def test_the_panel_reads_as_numbered_steps(self):
        self.assertEqual(
            self.page.step_titles,
            ["Choose an area", "Choose the style and detail",
             "Choose what to include", "Build"],
        )
        self.assertEqual(
            self.page.step_numbers, ["1", "2", "3", "4"], "the steps are misnumbered"
        )

    def test_advanced_settings_start_folded_away(self):
        """The point of the section is that a newcomer never has to open it."""
        self.assertEqual(
            self.page.details_open, [], "an advanced section ships open"
        )

    def test_the_detail_levels_match_the_server(self):
        """The page shows what a level means; the server applies it. Two
        tables, so they are held equal here, or the numbers on the page stop
        being the model that gets built."""
        from src.config import DETAIL_PRESETS

        body = re.search(r"const DETAIL = \{(.*?)\n\};", self.html, re.S).group(1)
        shown = {
            name: {k: float(v) for k, v in re.findall(r"(\w+): ([\d.]+)", values)}
            for name, values in re.findall(r"(\w+):\s*\{([^}]*)\}", body)
        }
        self.assertEqual(set(shown), set(DETAIL_PRESETS))
        for name, preset in DETAIL_PRESETS.items():
            self.assertEqual(shown[name]["aerial_cm"], preset["aerial_cm"], name)
            self.assertEqual(shown[name]["grid_m"], preset["grid_m"], name)
            self.assertEqual(shown[name]["facade_px"], preset["facade_px"], name)
        sentences = re.search(r"const DETAIL_PLAIN = \{(.*?)\n\};", self.html, re.S).group(1)
        self.assertEqual(set(re.findall(r"^\s*(\w+):", sentences, re.M)), set(DETAIL_PRESETS))
        options = set(re.findall(r'<option value="(\w+)"', re.search(
            r'<select id="detail">(.*?)</select>', self.html, re.S).group(1)))
        self.assertEqual(options, set(DETAIL_PRESETS))

    def test_there_are_no_sliders_left(self):
        """The ask was fewer controls. What is left: where, which style, how
        much detail, which layers, and whether to add collision shapes and
        previews."""
        self.assertNotIn('type="range"', self.html)
        controls = set(re.findall(r'<(?:input|select)[^>]*\bid="(\w+)"', self.html))
        self.assertEqual(
            controls, {"q", "size", "name", "style", "detail", "preview", "collision"}
        )

    def test_the_styles_match_the_server(self):
        """The page greys out what a style cannot have and unticks what it
        starts without; the config decides both. Held equal here: a source
        the page lets through in a style that cannot build it would be
        dropped by the config with only a log line to say so."""
        from src.config import STYLE_LOCKED, STYLE_PRESETS
        from src.sources import SOURCES

        body = re.search(r"const STYLE = \{(.*?)\n\};", self.html, re.S).group(1)
        shown = {
            name: {
                key: set(re.findall(r'"(\w+)"', ids))
                for key, ids in re.findall(r"(\w+): \[([^\]]*)\]", values)
            }
            for name, values in re.findall(r"(\w+):\s*\{([^}]*)\}", body)
        }
        self.assertEqual(set(shown), set(STYLE_PRESETS))
        options = set(re.findall(r'<option value="(\w+)"', re.search(
            r'<select id="style">(.*?)</select>', self.html, re.S).group(1)))
        self.assertEqual(options, set(STYLE_PRESETS))

        def value(tree: dict, path) -> object:
            for key in path:
                tree = tree.get(key, {}) if isinstance(tree, dict) else {}
            return tree if not isinstance(tree, dict) or tree else None

        for style in STYLE_PRESETS:
            locked = {
                source.id for source in SOURCES
                if source.toggles and all(
                    STYLE_LOCKED[style].get(tuple(path)) is False for path in source.toggles
                )
            }
            unticked = {
                source.id for source in SOURCES
                if source.toggles and source.id not in locked and all(
                    value(STYLE_PRESETS[style], path) is False for path in source.toggles
                )
            }
            self.assertEqual(shown[style]["locked"], locked, style)
            self.assertEqual(shown[style]["unticked"], unticked, style)
        sentences = re.search(r"const STYLE_PLAIN = \{(.*?)\n\};", self.html, re.S).group(1)
        self.assertEqual(set(re.findall(r"^\s*(\w+):", sentences, re.M)), set(STYLE_PRESETS))


@unittest.skipUnless(RUN_NETWORK, "network tests disabled with --offline")
class TestLiveServices(unittest.TestCase):
    """Checks against the real services, for the assumptions that can drift."""

    def test_wms_layer_title_resolves_to_name(self):
        from src.config import DEFAULTS
        from src.imagery import resolve_layer_name

        name, title = resolve_layer_name(
            DEFAULTS["aerial"]["wms_url"], "Luchtfoto Actueel Ortho 8cm RGB"
        )
        self.assertEqual(name, "Actueel_orthoHR")
        self.assertEqual(title, "Luchtfoto Actueel Ortho 8cm RGB")

    def test_ahn_still_offers_the_expected_coverages(self):
        from src.config import DEFAULTS
        from src.elevation import discover_coverages

        coverages = discover_coverages(DEFAULTS["terrain"]["wcs_url"])
        self.assertIn("dtm_05m", coverages)
        self.assertIn("dsm_05m", coverages)

    def test_wmts_fallback_produces_a_usable_image(self):
        """The fallback only runs when WMS fails, so exercise it directly."""
        from PIL import Image

        from src.config import DEFAULTS
        from src.imagery import fetch_aerial_wmts

        bbox = BBox(136200, 455200, 136500, 455500)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "wmts.png"
            fetch_aerial_wmts(
                bbox,
                out,
                layer_name="Actueel_orthoHR",
                size_px=512,
                wmts_url=DEFAULTS["aerial"]["wmts_url"],
            )
            with Image.open(out) as image:
                self.assertEqual(image.size, (512, 512))
                self.assertGreater(np.asarray(image.convert("RGB")).std(), 5.0)

    def test_3dbag_pages_decode_with_their_own_transform(self):
        """Each page carries its own quantisation grid, and it does differ."""
        from src.buildings import iter_api_pages
        from src.config import DEFAULTS

        cfg = DEFAULTS["buildings"]
        translates = []
        for index, page in enumerate(
            iter_api_pages(
                BBox(136000, 455000, 136100, 455100),
                api_url=cfg["api_url"],
                page_limit=2,
                max_pages=4,
            )
        ):
            translates.append(tuple(page["metadata"]["transform"]["translate"]))
            if index >= 2:
                break

        self.assertGreater(len(translates), 1)
        self.assertGreater(
            len(set(translates)),
            1,
            "pages shared a transform; merging raw pages would be safe, but the "
            "pipeline no longer assumes that",
        )


if __name__ == "__main__":
    unittest.main(argv=[a for a in sys.argv if a != "--offline"], verbosity=2)
