"""Facades read off photos as layouts, and walls drawn from layouts."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src import facade_layout as fl

M = 0.025  # metres a pixel


def _facade(width_m=8.0, height_m=12.0, *, columns=(1.0, 3.5, 6.0), win_w=1.2, rows=(4.2, 7.4),
            win_h=1.8, wall=(150, 80, 62), frame=(235, 232, 225), glass=(40, 50, 62)):
    """A straightened wall: brick with framed windows, row 0 at the top."""
    h, w = int(height_m / M), int(width_m / M)
    rng = np.random.default_rng(0)
    rgb = np.empty((h, w, 3), dtype=np.float32)
    rgb[:] = wall
    rgb += rng.normal(0, 6, rgb.shape).astype(np.float32)  # the brick's grain
    for z0 in rows:
        for x0 in columns:
            r0, r1 = int((height_m - z0 - win_h) / M), int((height_m - z0) / M)
            c0, c1 = int(x0 / M), int((x0 + win_w) / M)
            rgb[r0:r1, c0:c1] = frame
            rgb[r0 + 3:r1 - 3, c0 + 3:c1 - 3] = glass
    return np.clip(rgb, 0, 255), np.ones((h, w), dtype=np.float32)


def _measure(rgb, weight, width_m=8.0, height_m=12.0, **kw):
    return fl.measure_layout(
        [(rgb, weight)], face=1, m_per_px=M, width_m=width_m, z_min=0.0, z_max=height_m,
        ground_z=0.0, eave_z=height_m, **kw,
    )


class SegmentTests(unittest.TestCase):
    def test_runs_are_found_where_the_profile_rises(self):
        profile = np.zeros(100)
        for a, b in ((10, 25), (40, 55), (70, 85)):
            profile[a:b] = 1.0
        runs = fl.segment_runs(profile, tau=0.5, min_on=5, max_on=30, min_off=3)
        self.assertEqual(runs, [(10, 25), (40, 55), (70, 85)])

    def test_unseen_stretches_count_for_neither(self):
        profile = np.zeros(60)
        profile[10:20] = 1.0
        profile[30:40] = np.nan
        runs = fl.segment_runs(profile, tau=0.5, min_on=5, max_on=30, min_off=3, cost_on=1.0)
        self.assertEqual(runs, [(10, 20)])

    def test_a_run_shorter_than_the_least_is_not_taken(self):
        profile = np.zeros(50)
        profile[10:13] = 1.0
        # Three high values cannot pay for the five low ones a run of eight
        # would have to take in with them.
        self.assertEqual(fl.segment_runs(profile, tau=0.5, min_on=8, max_on=30, min_off=3), [])


class MeasureTests(unittest.TestCase):
    def test_a_grid_of_windows_is_read_back(self):
        rgb, weight = _facade()
        layout = _measure(rgb, weight)
        windows = sorted(layout.windows, key=lambda o: (o.z0, o.x0))
        self.assertEqual(len(windows), 6)
        for o, (x0, z0) in zip(windows, [(x, z) for z in (4.2, 7.4) for x in (1.0, 3.5, 6.0)]):
            self.assertAlmostEqual(o.x0, x0, delta=0.12)
            self.assertAlmostEqual(o.x1, x0 + 1.2, delta=0.12)
            self.assertAlmostEqual(o.z0, z0, delta=0.12)
            self.assertAlmostEqual(o.z1, z0 + 1.8, delta=0.12)
        self.assertGreater(layout.quality, 0.5)
        self.assertEqual(layout.material, "brick")
        # The frames are pale: read as they are.
        self.assertGreater(sum(layout.frame_rgb), 600)
        # A storey above the ground storey for each row.
        self.assertEqual(len(layout.floors), 3)

    def test_a_tree_in_front_of_the_wall_is_not_believed(self):
        rgb, weight = _facade()
        rows, cols = rgb.shape[:2]
        yy, xx = np.mgrid[:rows, :cols]
        crown = (yy - rows * 0.45) ** 2 + (xx - cols * 0.4) ** 2 < (cols * 0.35) ** 2
        rgb[crown] = (60, 110, 50)
        layout = _measure(rgb, weight)
        self.assertLess(layout.quality, 0.5)

    def test_unseen_columns_carry_on_in_the_wall_rhythm(self):
        windows = [fl.Opening("window", x, x + 1.0, z, z + 1.6) for x in (1.0, 3.0) for z in (4.0, 7.0)]
        out = fl.complete_windows(windows, width_m=10.0, seen_x=(0.0, 4.5), seen_top=12.0,
                                  eave_z=12.0, ground_z=0.0)
        columns = sorted({round(o.x0, 2) for o in out})
        self.assertEqual(columns, [1.0, 3.0, 5.0, 7.0])

    def test_nothing_seen_reads_nothing(self):
        rgb, weight = _facade()
        self.assertIsNone(_measure(rgb, weight * 0.0))


class DataLayoutTests(unittest.TestCase):
    def test_a_canal_house_from_the_registers(self):
        layout = fl.layout_from_data(face=0, width_m=6.0, z_min=0.0, z_max=14.0, ground_z=0.0,
                                     eave_z=11.0, function=0, year=1880)
        self.assertEqual(layout.ground, "house")
        # Ground storey and two above it, under eaves at 11 m.
        self.assertEqual(len(layout.floors), 3)
        doors = [o for o in layout.openings if o.kind == "door"]
        self.assertEqual(len(doors), 1)
        upper = [o for o in layout.windows if o.z0 > layout.floors[1]]
        centres = sorted({round((o.x0 + o.x1) / 2, 2) for o in upper if o.z1 < 11.0})
        # Centred in their bays: symmetric about the middle of the front.
        self.assertAlmostEqual(centres[0] + centres[-1], 6.0, delta=0.02)
        # The gable over the eaves has its attic window.
        self.assertTrue(any(o.z0 > 11.0 for o in layout.windows))

    def test_a_shop_gets_a_shop_front(self):
        layout = fl.layout_from_data(face=0, width_m=8.0, z_min=0.0, z_max=10.0, ground_z=0.0,
                                     eave_z=10.0, function=1, year=1900)
        self.assertEqual(layout.ground, "shop")
        self.assertTrue(any(o.kind == "shop" for o in layout.openings))
        self.assertIsNotNone(layout.fascia)

    def test_a_poor_reading_keeps_only_its_colours(self):
        data = fl.layout_from_data(face=0, width_m=8.0, z_min=0.0, z_max=10.0, ground_z=0.0, eave_z=10.0)
        measured = fl.FacadeLayout(face=0, width_m=8.0, z_min=0.0, z_max=10.0, ground_z=0.0,
                                   wall_rgb=(200, 190, 170), material="plaster", quality=0.2)
        merged = fl.merged_layout(measured, data, min_quality=0.5)
        self.assertIs(merged, data)
        self.assertEqual(merged.wall_rgb, (200, 190, 170))
        self.assertEqual(merged.material, "plaster")
        good = fl.FacadeLayout(face=0, width_m=8.0, z_min=0.0, z_max=10.0, ground_z=0.0, quality=0.8)
        self.assertIs(fl.merged_layout(good, data, min_quality=0.5), good)


class DrawTests(unittest.TestCase):
    def test_windows_are_drawn_where_the_layout_has_them(self):
        layout = fl.FacadeLayout(face=0, width_m=4.0, z_min=0.0, z_max=6.0, ground_z=0.0, eave_z=6.0,
                                 floors=[0.0, 3.0], wall_rgb=(150, 80, 62))
        layout.openings = [fl.Opening("window", 1.0, 2.2, 3.8, 5.4)]
        rgb, relief = fl.draw_layout(layout, 0.04, masonry=fl.Masonry(None))
        self.assertEqual(rgb.shape, (150, 100, 3))
        # In a pane: off the glazing bars down the middle and across.
        pane = (int((6.0 - 5.0) / 0.04), int(1.3 / 0.04))
        glass = rgb[pane]
        wall = rgb[int((6.0 - 2.0) / 0.04), int(3.5 / 0.04)]
        self.assertLess(int(glass.sum()), int(wall.sum()))
        self.assertLess(relief[pane], fl.RELIEF_WALL)


class AtlasTests(unittest.TestCase):
    def test_every_exposed_wall_is_drawn_and_placed(self):
        from src.photo_coverage import WallFace
        from src.wall_atlas import draw_walls

        def face(i, x0, building=0):
            return WallFace(
                building=building, building_id=f"b{building}", start=np.array([x0, 0.0]),
                end=np.array([x0 + 6.0, 0.0]), z_min=0.0, z_max=10.0, normal=np.array([0.0, -1.0]),
                area_m2=60.0, samples=np.zeros((0, 3)), triangles=np.zeros((0, 3, 3)),
                key=(building, i, 0), exposed=True,
            )

        faces = [face(0, 0.0), face(1, 10.0), face(2, 20.0)]
        faces[2].exposed = False  # a party wall: not drawn
        building = SimpleNamespace(
            identifier="b0", ground_z_nap=0.0, wall_top_nap=9.0, roof_max_nap=12.0, floors=3,
            storeys_known=True, function=0, build_year=1900,
        )
        buildings = SimpleNamespace(buildings=[building])
        with tempfile.TemporaryDirectory() as tmp:
            textures = draw_walls(buildings, faces, Path(tmp), Path(tmp),
                                  facade_cfg={"photo_textures": False, "drawn_px_per_m": 10.0})
            self.assertEqual(len(textures.placements), 2)
            self.assertTrue(all(p.is_file() for p in textures.pages))
            self.assertTrue(all(p.is_file() for p in textures.normal_pages))
            self.assertTrue((Path(tmp) / "facade_layouts.json").is_file())
        placement = textures.placements[(0, 0, 0)]
        corners = np.array([[0.0, 0.0, 10.0], [6.0, 0.0, 0.0]])
        uv = placement.uv(corners, textures.page_sizes[placement.page])
        self.assertTrue(np.all((uv >= 0.0) & (uv <= 1.0)))

    def test_photo_textures_are_off_and_walls_drawn_by_default(self):
        from src.config import DEFAULTS

        self.assertEqual(DEFAULTS["photos"]["textures"], "off")
        self.assertTrue(DEFAULTS["facade"]["drawn_walls"])


if __name__ == "__main__":
    unittest.main()
