"""Public space: paving, kerbs, street objects, terraces, statues and steps.

What a square looks like from the middle of it, which the registers describe
in pieces: the BGT what the ground is paved with and where the steps are,
OpenStreetMap the terraces and statues, Mapillary what stands on the street.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _ring(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float)


def _building(x0, y0, x1, y1, z=5.0):
    """A box building as `Footprints` reads it: roof triangles seen from above."""
    roof = np.array([
        [[x0, y0, z], [x1, y0, z], [x1, y1, z]],
        [[x0, y0, z], [x1, y1, z], [x0, y1, z]],
    ], dtype=float)
    return SimpleNamespace(roof_tris=roof)


class NeedleTipTests(unittest.TestCase):
    def test_a_walkway_needle_tip_goes_to_the_ground_beside_it(self):
        """The mesher's own needle: a 7 degree wedge of walkway between two
        road faces, whose kerb mitre flew 4.5 m off the Ganzenmarkt."""
        from src.layered import ROAD, WALKWAY, _blunt_walkway_tips

        points = np.array([
            [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],  # a walkway square
            [5.0, 10.0], [4.4, 15.0], [4.4, 10.0],  # and a needle off its top
            [0.0, 16.0], [10.0, 16.0],
        ])
        faces = [
            [0, 1, 2, 4, 6, 3],  # walkway
            [6, 4, 5],  # the needle, walkway too
            [3, 6, 5, 7],  # road west of the needle
            [4, 2, 8, 5],  # road east of it
        ]
        sizes = np.array([len(f) for f in faces])
        loops = np.concatenate(faces)
        layer = np.array([WALKWAY, WALKWAY, ROAD, ROAD])
        material = np.array([0, 0, 1, 1])
        new_layer, new_material, changed = _blunt_walkway_tips(points, loops, sizes, layer, material)
        self.assertEqual(int(new_layer[1]), ROAD)
        self.assertEqual(int(new_material[1]), 1)
        self.assertEqual(int(new_layer[0]), WALKWAY)  # the square keeps its corners
        self.assertEqual(changed, 1)


class PavingTests(unittest.TestCase):
    def test_the_street_photos_give_the_hue_and_the_aerial_the_brightness(self):
        from src.paving import target_colours

        photo = {"Bakstenen": {"rgb": [130, 80, 60], "samples": 900, "photos": 12}}
        aerial = {"Bakstenen": (129, 125, 119), "Tegels": (120, 120, 118)}
        out = target_colours(["Bakstenen", "Tegels", "Gras"], photo=photo, aerial=aerial)
        brick, source = out["Bakstenen"]
        self.assertIn("street photos", source)
        # Still red: the aerial's grey only moved its brightness.
        self.assertGreater(brick[0], brick[2] + 30)
        self.assertGreater(sum(brick), 130 + 80 + 60)
        tiles, source = out["Tegels"]
        self.assertTrue(source.startswith("typical"))
        self.assertLessEqual(max(tiles), 185)

    def test_a_texture_is_coloured_and_sized_from_its_photograph(self):
        from PIL import Image

        import src.paving as paving

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "brick_1k.jpg"
            Image.fromarray(np.full((32, 32, 3), 90, dtype=np.uint8)).save(source)
            with mock.patch.object(paving, "fetch_texture", lambda slug, work_dir, **k: source), \
                    mock.patch.object(paving, "texture_size_m", lambda slug, work_dir, **k: (2.4, 2.4)):
                made = paving.build_paving_textures(
                    ["Bakstenen", "Nothing"], tmp, targets={"Bakstenen": ((150, 90, 70), "street photos")},
                )
            self.assertEqual(set(made), {"Bakstenen"})
            texture = made["Bakstenen"]
            self.assertAlmostEqual(texture.tile_m, 2.4)
            mean = np.asarray(Image.open(texture.colour)).reshape(-1, 3).mean(axis=0)
            np.testing.assert_allclose(mean, (150, 90, 70), atol=3)
            entry = paving.scene_entry(made)["Bakstenen"]
            self.assertEqual(entry["colour"], "paving_Bakstenen.jpg")
            self.assertEqual(entry["colour_from"], "street photos")

    def test_ground_uvs_are_metres_over_the_texture(self):
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from test_pipeline import _process_namespace

        ns = _process_namespace({"_paving_uv", "planar_uv"})
        points = np.array([[0.0, 0.0], [4.8, 2.4], [10.0, 10.0]])
        uv = ns["_paving_uv"](points, np.array([0, 0, 1]), np.array([2.4, 0.0]), [-10, -10, 10, 10])
        np.testing.assert_allclose(uv[1], [2.0, 1.0])  # a textured face: metres / 2.4
        np.testing.assert_allclose(uv[2], [1.0, 1.0])  # a flat colour: the aerial's UVs


class StreetObjectTests(unittest.TestCase):
    def test_the_bgt_wins_and_a_report_close_by_is_the_same_object(self):
        from src.street_objects import (
            KIND_BIN, KIND_POLE, SOURCE_BGT, SOURCE_MAPILLARY, SOURCE_OSM, StreetObject, merge,
        )
        from src.furniture import KIND_BENCH, KIND_LAMP

        objects = [
            StreetObject(KIND_LAMP, 0.0, 0.0, SOURCE_BGT),
            StreetObject(KIND_LAMP, 3.0, 0.0, SOURCE_BGT),  # two BGT lamps stay two
            StreetObject(KIND_LAMP, 1.5, 1.0, SOURCE_MAPILLARY),  # the first, seen
            StreetObject(KIND_POLE, 0.5, 0.5, SOURCE_MAPILLARY),  # its pole
            StreetObject(KIND_BENCH, 20.0, 0.0, SOURCE_OSM),
            StreetObject(KIND_BENCH, 21.0, 0.0, SOURCE_MAPILLARY),
            StreetObject(KIND_BIN, 40.0, 0.0, SOURCE_MAPILLARY),
        ]
        kept, stats = merge(objects)
        self.assertEqual(stats["duplicates_dropped"], 3)
        self.assertEqual(
            sorted((o.kind, o.source) for o in kept),
            sorted([(KIND_LAMP, SOURCE_BGT), (KIND_LAMP, SOURCE_BGT), (KIND_BENCH, SOURCE_OSM),
                    (KIND_BIN, SOURCE_MAPILLARY)]),
        )

    def test_signs_hang_on_walls_and_things_indoors_come_out(self):
        from src.furniture import KIND_LAMP
        from src.street_objects import (
            KIND_BIN, KIND_HANGING_SIGN, KIND_SHOP_SIGN, KIND_WALL_LAMP, SOURCE_MAPILLARY,
            Footprints, StreetObject, place,
        )

        footprints = Footprints(SimpleNamespace(buildings=[_building(0, 0, 10, 10)]))
        objects = [
            StreetObject(KIND_SHOP_SIGN, 5.0, 12.0, SOURCE_MAPILLARY),  # 2 m out from the north wall
            StreetObject(KIND_HANGING_SIGN, 12.0, 5.0, SOURCE_MAPILLARY),  # east of the east wall
            StreetObject(KIND_SHOP_SIGN, 5.0, 30.0, SOURCE_MAPILLARY),  # no wall near
            StreetObject(KIND_LAMP, 5.0, 11.0, SOURCE_MAPILLARY),  # on the facade
            StreetObject(KIND_BIN, 9.5, 5.0, SOURCE_MAPILLARY),  # just inside
            StreetObject(KIND_BIN, 5.0, 5.0, SOURCE_MAPILLARY),  # deep inside
        ]
        kept, stats = place(objects, footprints)
        by_kind = {o.kind: o for o in kept}
        flat = by_kind[KIND_SHOP_SIGN]
        self.assertAlmostEqual(flat.y, 10.06, delta=0.05)  # on the wall's face
        self.assertAlmostEqual(abs(np.cos(flat.spin)), 1.0, delta=1e-6)  # along it
        hanging = by_kind[KIND_HANGING_SIGN]
        self.assertAlmostEqual(hanging.x, 10.7, delta=0.05)  # out on its bracket
        self.assertIn(KIND_WALL_LAMP, by_kind)
        self.assertEqual(stats["signs_without_wall"], 1)
        self.assertEqual(stats["moved_out"], 1)
        self.assertEqual(stats["dropped_indoors"], 1)
        self.assertGreater(by_kind[KIND_BIN].x, 10.0)

    def test_a_statue_is_as_tall_as_the_survey_over_it(self):
        from src.osm import OsmFeature
        from src.street_objects import KIND_STATUE, osm_objects

        statue = OsmFeature("n1", "statue", {"tourism": "artwork", "artwork_type": "statue"},
                            np.array([[100.0, 100.0]]))

        def surface(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            r = np.hypot(x - 100.0, y - 100.0)
            # A 3.4 m statue, a tree crown at 12 m two metres off.
            return np.where(r < 0.6, 5.4, np.where(r > 1.5, 14.0, 2.0))

        def ground(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            # No bare earth under the statue itself: the survey has a hole.
            return np.where(np.hypot(x - 100.0, y - 100.0) < 1.0, np.nan, 2.0)

        (made,) = osm_objects(SimpleNamespace(features=[statue]), surface_at=surface, ground_at=ground)
        self.assertEqual(made.kind, KIND_STATUE)
        self.assertAlmostEqual(made.height_m, 3.4, places=3)

    def test_mapillary_classes_become_objects(self):
        from src.furniture import KIND_SIGN
        from src.street_objects import KIND_HANGING_SIGN, KIND_MANHOLE, KIND_SHOP_SIGN, mapillary_objects

        made = mapillary_objects([
            {"kind": "facade_sign", "value": "object--sign--store", "x": 0, "y": 0, "heading": 90.0},
            {"kind": "facade_sign", "value": "object--banner", "x": 0, "y": 0, "heading": None},
            {"kind": "manhole", "value": "object--catch-basin", "x": 0, "y": 0, "heading": None},
            {"kind": "traffic_sign", "value": "regulatory--no-entry--g1", "x": 0, "y": 0, "heading": 0.0},
            {"kind": None, "value": "object--cctv-camera", "x": 0, "y": 0, "heading": None},
        ])
        self.assertEqual([o.kind for o in made], [KIND_SHOP_SIGN, KIND_HANGING_SIGN, KIND_MANHOLE, KIND_SIGN])
        self.assertAlmostEqual(made[0].spin, 0.0)  # a bearing of 90 is east
        self.assertAlmostEqual(made[2].size, 0.45)


class TerraceTests(unittest.TestCase):
    def test_an_outlined_terrace_is_filled_and_a_cafe_gets_a_row_out_front(self):
        from src.layered import GroundArea, ROAD, WALKWAY
        from src.osm import OsmFeature, OsmSet
        from src.street_objects import KIND_PARASOL, KIND_TABLE, Footprints
        from src.terraces import build_terraces, paving_for_terraces

        footprints = Footprints(SimpleNamespace(buildings=[
            _building(0, 0, 10, 10), _building(30, 0, 40, 10),
        ]))
        outline = _ring(60, 0, 68, 6)  # far enough off to be no one else's
        osm = OsmSet(features=[
            OsmFeature("w1", "terrace", {"leisure": "outdoor_seating"}, outline, closed=True),
            # A café in the second building, its door on the north side.
            OsmFeature("n2", "terrace_venue", {"amenity": "cafe", "outdoor_seating": "yes"},
                       np.array([[35.0, 9.0]])),
            # One whose front is a carriageway in asphalt: no terrace there.
            OsmFeature("n3", "terrace_venue", {"amenity": "bar", "outdoor_seating": "yes"},
                       np.array([[5.0, 1.0]])),
        ])
        areas = [
            GroundArea([_ring(-5, 10, 45, 20)], WALKWAY, "Bakstenen"),
            GroundArea([_ring(-5, -10, 25, 0)], ROAD, "Asfalt"),
        ]
        made, stats = build_terraces(osm, footprints, paving_ok=paving_for_terraces(areas))
        tables = [o for o in made if o.kind == KIND_TABLE]
        self.assertEqual(stats["outlined"], 1)
        self.assertEqual(stats["along_facades"], 1)
        self.assertEqual(stats["venues_without_room"], 1)
        inside = [t for t in tables if 60 <= t.x <= 68 and 0 <= t.y <= 6]
        self.assertGreaterEqual(len(inside), 4)
        row = [t for t in tables if 25 < t.x < 45]
        self.assertTrue(row and all(10.5 < t.y < 12.5 for t in row))
        self.assertTrue(any(o.kind == KIND_PARASOL for o in made))


class StairTests(unittest.TestCase):
    def test_a_flight_down_a_slope_is_cut_into_treads(self):
        import shapely

        from src.stairs import public_flight

        # 1.5 m wide, 4 m long, the ground climbing 2 m along it, and a quay
        # drop beside it that a plane fit would read as the way it climbs.
        outline = shapely.box(0.0, 0.0, 4.0, 1.5)

        def ground(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            return np.clip(x, 0.0, 4.0) * 0.5 - np.where(y < 0.0, 3.0, 0.0)

        flight = public_flight(outline, ground_at=ground, survey_at=ground, source="bgt")
        self.assertIsNotNone(flight)
        self.assertAlmostEqual(flight.rise_m, 2.0, delta=0.05)
        self.assertEqual(flight.risers, 12)
        tops = [top for _, top in flight.treads]
        self.assertTrue(all(b > a for a, b in zip(tops, tops[1:])))
        self.assertAlmostEqual(tops[-1], 2.0, delta=0.05)
        lowest = min(flight.treads, key=lambda t: t[1])[0][0]
        self.assertLess(np.asarray(lowest)[:, 0].max(), 0.5)  # the first tread at the low end

    def test_flat_ground_and_impossible_steepness_make_no_flight(self):
        import shapely

        from src.stairs import public_flight

        outline = shapely.box(0.0, 0.0, 4.0, 1.5)
        flat = public_flight(outline, ground_at=lambda x, y: np.zeros(np.shape(x)),
                             survey_at=None, source="bgt")
        self.assertIsNone(flat)
        sliver = shapely.box(0.0, 0.0, 0.6, 1.5)
        cliff = public_flight(sliver, ground_at=lambda x, y: np.asarray(y, float) * 4.0,
                              survey_at=None, source="bgt")
        self.assertIsNone(cliff)

    def test_a_stoop_climbs_to_its_front_door(self):
        import shapely

        from src.stairs import stoop_flight

        building = shapely.box(0.0, 2.0, 10.0, 10.0)
        stoop = shapely.box(4.0, 0.8, 6.0, 2.0)

        def surface(x, y):
            return np.full(np.shape(x), 0.68)

        flight = stoop_flight(stoop, building=building, ground_at=lambda x, y: np.zeros(np.shape(x)),
                              surface_at=surface)
        self.assertEqual(flight.risers, 4)
        tops = sorted(flight.treads, key=lambda t: t[1])
        # Highest against the wall.
        self.assertGreater(np.asarray(tops[-1][0][0])[:, 1].mean(), np.asarray(tops[0][0][0])[:, 1].mean())


class OsmTests(unittest.TestCase):
    XML = b"""<?xml version="1.0"?>
<osm version="0.6">
  <node id="1" lat="52.0910" lon="5.1180"><tag k="amenity" v="bench"/></node>
  <node id="2" lat="52.0911" lon="5.1181"><tag k="tourism" v="artwork"/><tag k="artwork_type" v="statue"/></node>
  <node id="3" lat="52.0912" lon="5.1182"><tag k="historic" v="memorial"/><tag k="memorial" v="stolperstein"/></node>
  <node id="4" lat="52.0913" lon="5.1183"/>
  <node id="5" lat="52.0914" lon="5.1183"/>
  <node id="6" lat="52.0914" lon="5.1185"/>
  <node id="7" lat="52.0913" lon="5.1185"/>
  <way id="10"><nd ref="4"/><nd ref="5"/><nd ref="6"/><nd ref="7"/><nd ref="4"/>
    <tag k="leisure" v="outdoor_seating"/></way>
  <way id="11"><nd ref="4"/><nd ref="5"/><tag k="highway" v="steps"/></way>
  <node id="8" lat="52.0915" lon="5.1186"><tag k="amenity" v="cafe"/><tag k="outdoor_seating" v="yes"/></node>
</osm>"""

    def test_what_the_map_has_becomes_what_the_model_draws(self):
        from src.osm import classify, parse_osm_xml

        kinds = {i: classify(tags, closed) for i, tags, _c, closed in parse_osm_xml(self.XML)}
        self.assertEqual(kinds["n1"], "bench")
        self.assertEqual(kinds["n2"], "statue")
        self.assertIsNone(kinds["n3"])  # a stone in the pavement, not a statue
        self.assertEqual(kinds["w10"], "terrace")
        self.assertEqual(kinds["w11"], "steps")
        self.assertEqual(kinds["n8"], "terrace_venue")


class SourceTests(unittest.TestCase):
    def test_openstreetmap_is_a_soft_source_on_by_default(self):
        from src.config import DEFAULTS
        from src.sources import BY_ID, enabled_sources

        osm = BY_ID["osm"]
        self.assertTrue(osm.soft)
        self.assertTrue(osm.default_on)
        self.assertIn(osm, enabled_sources(DEFAULTS))
        self.assertTrue(DEFAULTS["layered"]["paving"])
        self.assertTrue(DEFAULTS["layered"]["flush_walkways"])


if __name__ == "__main__":
    unittest.main()
