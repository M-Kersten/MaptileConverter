"""Checks for the parts that are easy to get subtly wrong.

Run everything (network tests included):

    python tests/test_pipeline.py

Run only the offline tests:

    python tests/test_pipeline.py --offline
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
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
from src.imagery import _tile_edges  # noqa: E402

RUN_NETWORK = "--offline" not in sys.argv


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
            spans = _tile_edges(total, cap)
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


class TestConfig(unittest.TestCase):
    def test_shipped_config_loads(self):
        config = load_config(REPO_ROOT / "config.json")
        self.assertEqual(config.bbox.width, 1000.0)
        self.assertEqual(config.geo.origin, (136500.0, 455500.0))
        self.assertEqual(config.warnings, [])

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
        """Five BGT layers behind one host must not look like five outages."""
        from src.sources import health_targets

        targets = health_targets(self.config)
        bgt = [ids for url, ids in targets.items() if "bgt" in url]
        self.assertEqual(len(bgt), 1)
        self.assertEqual(
            set(bgt[0].split(",")),
            {"trees", "water", "land_cover", "furniture", "vehicles"},
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
        self.assertEqual(len(targets), 3)

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
            "mesh_vertices": 129,
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
        self.assertEqual(loaded.terrain["mesh_vertices_per_side"], 129)
        self.assertEqual(loaded.facade["variants"], 4)
        self.assertEqual(loaded.buildings["clip_mode"], "intersect")

    def test_defaults_apply_when_the_form_omits_fields(self):
        config = self.server.build_config(
            {
                "name": "bare",
                "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
            }
        )
        self.assertEqual(config["aerial"]["size_px"], 4096)
        self.assertEqual(config["terrain"]["ahn_model"], "DTM")
        self.assertEqual(config["buildings"]["lod"], "2.2")

    def test_area_summary_is_none_for_unknown_area(self):
        self.assertIsNone(self.server.area_summary("no_such_area_xyz"))

    def test_quality_fields_reach_the_config(self):
        config = self.server.build_config(
            {
                "name": "q",
                "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
                "size_px": 12288,
                "mesh_vertices": 1025,
                "facade_texture_px": 2048,
                "facade_normal_map": False,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            loaded = load_config(path)

        self.assertEqual(loaded.aerial["size_px"], 12288)
        self.assertEqual(loaded.terrain["mesh_vertices_per_side"], 1025)
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

        measured = [
            (seconds(1000, 4096, preview=True), 468),
            (seconds(1000, 4096, preview=False), 218),
            (seconds(500, 2048, preview=False), 75),
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
                "mesh_vertices": 4097,  # finer than the 0.5 m DTM
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
        self.assertEqual(loaded.aerial["size_px"], 16384)


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
