"""Street photos: camera maths, Mapillary and own-photo ingest, wall coverage.

Offline throughout. The one real Mapillary record below was fetched once and
is kept verbatim, because the convention its rotation follows is the single
fact everything else depends on, and a synthetic record cannot check that.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.buildings import Building, BuildingSet  # noqa: E402
from src.camera import (  # noqa: E402
    FISHEYE,
    PERSPECTIVE,
    SPHERICAL,
    Camera,
    axis_angle_from_rotation,
    heading_of,
    project_camera,
    project_world,
    rotation_from_axis_angle,
    rotation_from_heading,
    unproject_camera,
)
from src.geo import BBox  # noqa: E402
from src.photo_coverage import (  # noqa: E402
    MEASURE,
    TEXTURE,
    UNSEEN,
    render_coverage_map,
    score_views,
    wall_faces,
)
from src.photos import (  # noqa: E402
    POSE_MANUAL,
    POSE_SFM,
    SOURCE_MAPILLARY,
    SOURCE_OWN,
    Photo,
    grid_convergence_deg,
    load_photos,
    lonlat_to_rd,
    save_photos,
    thin_photos,
    with_height,
)

# Fetched from graph.mapillary.com on 2026-11-21: a phone photo on the
# Oudegracht in Utrecht. computed_compass_angle is Mapillary's own statement
# of where its reconstructed camera points.
UTRECHT_RECORD = {
    "id": "1624887455022621",
    "captured_at": 1719564563000,
    "camera_type": "perspective",
    "camera_parameters": [0.73686988645114, 0.098946606750975, -0.1339014300404],
    "compass_angle": 31.568,
    "computed_compass_angle": 47.299891540902,
    "computed_geometry": {"type": "Point", "coordinates": [5.1210721408681, 52.089703259779]},
    "geometry": {"type": "Point", "coordinates": [5.1210428, 52.0897076]},
    "computed_rotation": [1.3964990445457, -0.85331459346798, 0.42623099812092],
    "sequence": "ytrwGlpZfU7Fqn4OQ9Khji",
    "width": 4000,
    "height": 3000,
    "creator": {"username": "someone", "id": "1"},
}


def box_building(
    identifier: str, x0: float, y0: float, x1: float, y1: float, height: float = 9.0
) -> Building:
    """A flat-roofed box with outward-wound walls, as 3DBAG delivers them."""
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # anticlockwise from above
    walls = []
    for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1]):
        a, b = np.array([ax, ay, 0.0]), np.array([bx, by, 0.0])
        at, bt = a + [0, 0, height], b + [0, 0, height]
        walls += [[a, b, bt], [a, bt, at]]
    roof = [
        [[x0, y0, height], [x1, y0, height], [x1, y1, height]],
        [[x0, y0, height], [x1, y1, height], [x0, y1, height]],
    ]
    return Building(
        identifier=identifier,
        wall_tris=np.array(walls, dtype=np.float64),
        roof_tris=np.array(roof, dtype=np.float64),
        ground_z_nap=0.0,
        roof_max_nap=height,
        floors=3,
        wall_top_nap=height,
    )


def photo_at(
    x: float, y: float, heading: float, *, kind: str = PERSPECTIVE, z: float = 2.0,
    ident: str = "p", source: str = SOURCE_MAPILLARY, sequence: str = "",
) -> Photo:
    camera = Camera(kind, 2048, 1024 if kind == SPHERICAL else 1536, 0.8)
    return Photo(
        id=ident, source=source, x=x, y=y, z=z,
        rotation=rotation_from_heading(heading), camera=camera, sequence=sequence,
    )


class CameraTests(unittest.TestCase):
    def test_round_trip_every_lens(self) -> None:
        rng = np.random.default_rng(3)
        for kind, k1, k2 in (
            (PERSPECTIVE, 0.09, -0.13),
            (FISHEYE, -0.05, 0.01),
            (SPHERICAL, 0.0, 0.0),
        ):
            camera = Camera(kind, 2000, 1000 if kind == SPHERICAL else 1500, 0.7, k1, k2)
            rays = rng.normal(size=(200, 3))
            if kind != SPHERICAL:
                rays[:, 2] = np.abs(rays[:, 2]) + 1.5  # in front of the lens
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            pixels, valid = project_camera(camera, rays * 7.0)
            back = unproject_camera(camera, pixels[valid])
            np.testing.assert_allclose(back, rays[valid], atol=1e-6, err_msg=kind)
            self.assertGreater(valid.mean(), 0.3, kind)

    def test_distortion_never_folds_back_into_the_image(self) -> None:
        # With these coefficients the curve turns over at about 1.2 radians
        # of normalised radius. A point beyond it must not reappear inside.
        camera = Camera(PERSPECTIVE, 2000, 1500, 0.7, 0.09, -0.13)
        _, valid = project_camera(camera, np.array([[3.0, 0.0, 1.0], [0.3, 0.0, 1.0]]))
        self.assertEqual(list(valid), [False, True])

    def test_perspective_centre_and_behind(self) -> None:
        camera = Camera(PERSPECTIVE, 1000, 800, 0.8)
        pixels, valid = project_camera(camera, np.array([[0, 0, 5.0], [0, 0, -5.0]]))
        np.testing.assert_allclose(pixels[0], [500, 400])
        self.assertEqual(list(valid), [True, False])

    def test_spherical_layout(self) -> None:
        # The middle column looks forward, the left edge backward, and the top
        # row straight up.
        camera = Camera(SPHERICAL, 4000, 2000)
        pixels, _ = project_camera(
            camera, np.array([[0, 0, 1.0], [1.0, 0, 0], [0, -1.0, 0.0001]])
        )
        np.testing.assert_allclose(pixels[0], [2000, 1000], atol=1e-6)
        np.testing.assert_allclose(pixels[1], [3000, 1000], atol=1e-6)
        self.assertLess(pixels[2][1], 1.0)

    def test_heading_rotation(self) -> None:
        for heading in (0, 47.3, 90, 200, 359):
            R = rotation_from_heading(heading, pitch_deg=5)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R), 1.0)
            self.assertAlmostEqual(heading_of(R), heading % 360, places=6)
            # Image-down points down the world.
            self.assertLess(R[1][2], -0.9)
        # A point due east of a camera facing east lands mid-image.
        pixels, valid = project_world(
            Camera(PERSPECTIVE, 100, 100, 0.8), rotation_from_heading(90),
            np.zeros(3), np.array([[10.0, 0, 0]]),
        )
        self.assertTrue(valid[0])
        np.testing.assert_allclose(pixels[0], [50, 50], atol=1e-9)

    def test_axis_angle_round_trip(self) -> None:
        for v in ([0.3, -0.2, 0.9], [1.3965, -0.8533, 0.4262], [0, 0, np.pi - 1e-8], [0, 0, 0]):
            R = rotation_from_axis_angle(v)
            np.testing.assert_allclose(
                rotation_from_axis_angle(axis_angle_from_rotation(R)), R, atol=1e-6
            )


class MapillaryRecordTests(unittest.TestCase):
    def test_real_record_points_where_mapillary_says(self) -> None:
        from src.mapillary import photo_from_record

        photo = photo_from_record(UTRECHT_RECORD)
        self.assertEqual(photo.pose, POSE_SFM)
        convergence = float(grid_convergence_deg(5.1210721408681, 52.089703259779)[0])
        # Utrecht is west of RD's central meridian, so true north is turned a
        # fraction of a degree clockwise of grid north there.
        self.assertGreater(convergence, 0.0)
        self.assertLess(convergence, 0.5)
        self.assertAlmostEqual(
            photo.heading_deg, (47.299891540902 + convergence) % 360, delta=0.05
        )
        # Upright, and looking slightly down at the canal.
        self.assertLess(photo.rotation[1][2], -0.95)
        x, y = lonlat_to_rd(5.1210721408681, 52.089703259779)
        self.assertAlmostEqual(photo.x, x, places=3)
        self.assertAlmostEqual(photo.y, y, places=3)
        self.assertEqual(photo.camera.focal, 0.73686988645114)
        self.assertEqual(photo.licence, "CC BY-SA 4.0")
        self.assertEqual(photo.creator, "someone")
        self.assertTrue(photo.captured_at.startswith("2024-06-28"))

    def test_record_without_reconstruction_uses_compass(self) -> None:
        from src.mapillary import photo_from_record

        record = dict(UTRECHT_RECORD)
        record.pop("computed_rotation")
        record.pop("computed_geometry")
        photo = photo_from_record(record)
        self.assertEqual(photo.pose, "gps+compass")
        self.assertAlmostEqual(photo.heading_deg, 47.3, delta=0.6)

        record.pop("computed_compass_angle")
        record["compass_angle"] = -1
        photo = photo_from_record(record)
        self.assertFalse(photo.heading_known)

    def test_spherical_camera(self) -> None:
        from src.mapillary import photo_from_record

        record = dict(UTRECHT_RECORD, camera_type="spherical", width=5760, height=2880)
        record.pop("camera_parameters")
        self.assertEqual(photo_from_record(record).camera.kind, SPHERICAL)


class _FakeResponse:
    def __init__(self, payload, status: int = 200, content: bytes = b"") -> None:
        self._payload = payload
        self.status_code = status
        self.ok = status < 400
        self.content = content or json.dumps(payload).encode()
        self.text = self.content.decode(errors="ignore")

    def json(self):
        return self._payload


class _FakeSession:
    """Answers image searches from a fixed set of points, 2000 at most."""

    def __init__(self, points: list[tuple[float, float]]) -> None:
        self.points = points
        self.calls = 0
        self.headers: dict = {}

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls += 1
        xmin, ymin, xmax, ymax = (float(v) for v in params["bbox"].split(","))
        hits = [
            dict(UTRECHT_RECORD, id=str(i), computed_geometry={"type": "Point", "coordinates": [lon, lat]})
            for i, (lon, lat) in enumerate(self.points)
            if xmin <= lon < xmax and ymin <= lat < ymax
        ]
        return _FakeResponse({"data": hits[: int(params["limit"])]})


class MapillarySearchTests(unittest.TestCase):
    def test_full_tiles_are_split(self) -> None:
        from src.mapillary import PAGE_LIMIT, fetch_image_metadata
        from src.geo import wgs84_bbox_to_rd

        rng = np.random.default_rng(1)
        # 2600 images in one corner of one tile: more than a page holds.
        lon = 5.1201 + rng.random(2600) * 0.0008
        lat = 52.0901 + rng.random(2600) * 0.0008
        session = _FakeSession(list(zip(lon, lat)))
        area = wgs84_bbox_to_rd(BBox(5.120, 52.090, 5.1225, 52.0925))
        with tempfile.TemporaryDirectory() as tmp:
            records = fetch_image_metadata(area, session=session, cache_dir=Path(tmp))
            self.assertEqual(len(records), 2600)
            calls = session.calls
            self.assertGreater(calls, 4)
            # A second search over the same area comes from the cache.
            again = fetch_image_metadata(area, session=session, cache_dir=Path(tmp))
            self.assertEqual(len(again), 2600)
            self.assertEqual(session.calls, calls)
        self.assertEqual(PAGE_LIMIT, 2000)

    def test_missing_token_says_how_to_fix_it(self) -> None:
        import os
        from unittest import mock

        from src import mapillary

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            mapillary, "TOKEN_FILES", (Path("/nonexistent/token"),)
        ):
            with self.assertRaises(mapillary.MissingToken) as caught:
                mapillary.find_token()
        self.assertIn("MAPILLARY_TOKEN", str(caught.exception))

    def test_token_from_environment(self) -> None:
        import os
        from unittest import mock

        from src import mapillary

        with mock.patch.dict(os.environ, {"MAPILLARY_TOKEN": " MLY|1|abc \n"}):
            self.assertEqual(mapillary.find_token(), "MLY|1|abc")
            session = mapillary.session_for(mapillary.find_token())
        self.assertEqual(session.headers["Authorization"], "OAuth MLY|1|abc")


def _jpeg(path: Path, size=(400, 300), *, exif=None, xmp: bytes | None = None) -> Path:
    from PIL import Image

    image = Image.new("RGB", size, (120, 110, 100))
    kwargs = {}
    if exif is not None:
        kwargs["exif"] = exif
    if xmp is not None:
        kwargs["xmp"] = xmp
    image.save(path, "JPEG", **kwargs)
    return path


def _gps_exif(lat: float, lon: float, heading: float | None = None, f35: float | None = None):
    from PIL import Image

    def dms(value: float):
        d = int(value)
        m = int((value - d) * 60)
        s = (value - d - m / 60) * 3600
        return (float(d), float(m), round(s, 4))

    exif = Image.Exif()
    gps = {1: "N", 2: dms(lat), 3: "E", 4: dms(lon)}
    if heading is not None:
        gps.update({16: "T", 17: float(heading)})
    exif[0x8825] = gps
    if f35 is not None:
        exif[0x8769] = {0xA405: int(f35), 0x9003: "2026:05:01 10:30:00"}
    return exif


class OwnPhotoTests(unittest.TestCase):
    def test_gps_and_heading_from_exif(self) -> None:
        from src.own_photos import read_folder

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _jpeg(folder / "a.jpg", (4000, 3000), exif=_gps_exif(52.0897, 5.1210, 90.0, 26))
            result = read_folder(folder)
        self.assertEqual(len(result.photos), 1, result.skipped)
        photo = result.photos[0]
        x, y = lonlat_to_rd(5.1210, 52.0897)
        self.assertAlmostEqual(photo.x, x, delta=0.05)
        self.assertAlmostEqual(photo.y, y, delta=0.05)
        convergence = float(grid_convergence_deg(5.1210, 52.0897)[0])
        self.assertAlmostEqual(photo.heading_deg, 90.0 + convergence, delta=0.01)
        self.assertTrue(photo.heading_known)
        self.assertEqual(photo.camera.kind, PERSPECTIVE)
        # 26 mm equivalent is a 75 degree diagonal, which on a 4:3 sensor is
        # 67 degrees across.
        across = 2 * np.degrees(np.arctan(0.5 / photo.camera.focal))
        self.assertAlmostEqual(across, 67.3, delta=0.5)
        self.assertEqual(photo.captured_at, "2026-05-01T10:30:00")
        self.assertEqual(photo.source, SOURCE_OWN)

    def test_equirectangular_with_gpano_heading(self) -> None:
        from src.own_photos import read_folder

        xmp = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
            b'xmlns:GPano="http://ns.google.com/photos/1.0/panorama/" '
            b'GPano:ProjectionType="equirectangular" '
            b'GPano:PoseHeadingDegrees="200.0"/></rdf:RDF></x:xmpmeta>'
        )
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _jpeg(folder / "pano.jpg", (2048, 1024), exif=_gps_exif(51.98, 5.91), xmp=xmp)
            photo = read_folder(folder).photos[0]
        self.assertEqual(photo.camera.kind, SPHERICAL)
        convergence = float(grid_convergence_deg(5.91, 51.98)[0])
        self.assertAlmostEqual(photo.heading_deg, (200.0 + convergence) % 360, delta=0.01)

    def test_pano_without_heading_is_kept_and_flagged(self) -> None:
        from src.own_photos import read_folder

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _jpeg(folder / "pano.jpg", (4096, 2048), exif=_gps_exif(51.98, 5.91))
            photo = read_folder(folder).photos[0]
        self.assertEqual(photo.camera.kind, SPHERICAL)
        self.assertFalse(photo.heading_known)

    def test_sidecar_wins_and_places_a_photo_without_gps(self) -> None:
        from src.own_photos import read_folder

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _jpeg(folder / "court.jpg", exif=_gps_exif(52.0, 5.0, 10.0))
            (folder / "court.jpg.json").write_text(
                json.dumps({"x": 136500.0, "y": 455500.0, "heading": 270, "height_m": 1.4})
            )
            _jpeg(folder / "quay.jpg")
            (folder / "poses.json").write_text(
                json.dumps({"quay.jpg": {"lat": 52.0897, "lon": 5.1210, "heading": 45}})
            )
            _jpeg(folder / "lost.jpg")
            result = read_folder(folder)
        by_id = {p.id: p for p in result.photos}
        court = by_id["own_court"]
        self.assertEqual(court.pose, POSE_MANUAL)
        self.assertEqual((court.x, court.y), (136500.0, 455500.0))
        # A sidecar heading is on the map it was placed on: the RD grid.
        self.assertAlmostEqual(court.heading_deg, 270.0, places=6)
        self.assertEqual(court.height_m, 1.4)
        self.assertEqual(by_id["own_quay"].pose, POSE_MANUAL)
        self.assertEqual([name for name, _ in result.skipped], ["lost.jpg"])


class PhotoListTests(unittest.TestCase):
    def test_thinning_keeps_own_photos(self) -> None:
        sequence = [
            photo_at(1000.0 + i, 2000.0, 0.0, ident=f"m{i}", sequence="s") for i in range(20)
        ]
        for i, photo in enumerate(sequence):
            photo.captured_at = f"2024-06-01T10:00:{i:02d}Z"
        own = [photo_at(1000.0, 2000.0, 0.0, ident=f"o{i}", source=SOURCE_OWN) for i in range(3)]
        kept = thin_photos(sequence + own, 4.0)
        self.assertEqual(sum(p.source == SOURCE_MAPILLARY for p in kept), 5)
        self.assertEqual(sum(p.source == SOURCE_OWN for p in kept), 3)

    def test_height_above_ground(self) -> None:
        photos = [
            photo_at(0.0, 0.0, 0.0, ident="m"),
            photo_at(0.0, 0.0, 0.0, ident="o", source=SOURCE_OWN),
        ]
        photos[1].height_m = 1.2
        placed = with_height(photos, lambda x, y: np.full(len(x), 3.0), {SOURCE_MAPILLARY: 2.0})
        self.assertEqual([p.z for p in placed], [5.0, 4.2])

    def test_save_and_load(self) -> None:
        photo = photo_at(1.0, 2.0, 33.0, ident="x")
        photo.height_m = 1.5
        with tempfile.TemporaryDirectory() as tmp:
            path = save_photos([photo], Path(tmp) / "photos.json")
            (back,) = load_photos(path)
        np.testing.assert_allclose(back.rotation, photo.rotation, atol=1e-8)
        self.assertEqual(back.camera, photo.camera)
        self.assertEqual(back.height_m, 1.5)


class CoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        # Two terraced houses sharing a party wall, and a lone house further
        # east with a shed in front of it.
        self.buildings = BuildingSet(
            buildings=[
                box_building("left", 0, 0, 6, 8),
                box_building("right", 6, 0, 12, 8),
                box_building("east", 40, 0, 50, 8),
                box_building("shed", 43, -6, 47, -4, height=2.5),
            ]
        )
        self.faces = wall_faces(self.buildings)

    def face(self, building: str, normal) -> int:
        for i, f in enumerate(self.faces):
            if f.building_id == building and np.allclose(f.normal, normal, atol=1e-6):
                return i
        raise AssertionError(f"no {normal} face on {building}")

    def test_faces_point_out_and_party_walls_are_not_exposed(self) -> None:
        self.assertEqual(len(self.faces), 16)
        south = self.faces[self.face("left", [0, -1])]
        self.assertAlmostEqual(south.width_m, 6.0)
        self.assertAlmostEqual(south.height_m, 9.0)
        # Left to right for someone standing south of it: west to east.
        self.assertLess(south.start[0], south.end[0])
        score_views(self.faces, [])
        self.assertFalse(self.faces[self.face("left", [1, 0])].exposed)
        self.assertFalse(self.faces[self.face("right", [-1, 0])].exposed)
        self.assertTrue(self.faces[self.face("left", [-1, 0])].exposed)

    def test_camera_in_front_sees_the_front_sharply(self) -> None:
        # Far enough back that the 50 degree vertical field takes in the eaves.
        photo = photo_at(6.0, -18.0, 0.0, ident="front")
        coverage = score_views(self.faces, [photo])
        coverage.thresholds = {"min_visible": 0.6, "measure_px_per_m": 12, "texture_px_per_m": 40}
        front = self.face("left", [0, -1])
        (view,) = coverage.views[front]
        self.assertEqual(view.photo, "front")
        self.assertGreater(view.visible, 0.9)
        # 0.8 * 2048 px per radian at about eleven metres.
        self.assertAlmostEqual(view.px_per_m, 0.8 * 2048 / view.distance_m * np.cos(np.radians(view.incidence_deg)), delta=1.0)
        self.assertEqual(coverage.face_class(front), TEXTURE)
        # The back is not seen, and neither is anything behind the camera.
        self.assertEqual(coverage.views[self.face("left", [0, 1])], [])
        self.assertEqual(coverage.face_class(self.face("left", [0, 1])), UNSEEN)

    def test_camera_facing_away_sees_nothing(self) -> None:
        coverage = score_views(self.faces, [photo_at(6.0, -10.0, 180.0)])
        self.assertTrue(all(not v for v in coverage.views))
        self.assertEqual(coverage.photos_used, 0)

    def test_shed_hides_only_the_ground_floor(self) -> None:
        # Low camera behind the shed: the lower samples of the house front
        # are hidden, the upper ones are not.
        coverage = score_views(self.faces, [photo_at(45.0, -12.0, 0.0, z=1.5)])
        (view,) = coverage.views[self.face("east", [0, -1])]
        self.assertGreater(view.visible, 0.3)
        self.assertLess(view.visible, 1.0)
        # From higher up and looking down a little, the shed hides nothing.
        drone = photo_at(45.0, -12.0, 0.0, z=6.0)
        drone.rotation = rotation_from_heading(0.0, pitch_deg=-12.0)
        coverage = score_views(self.faces, [drone])
        (view,) = coverage.views[self.face("east", [0, -1])]
        self.assertGreater(view.visible, 0.99)

    def test_far_or_skew_views_measure_but_do_not_texture(self) -> None:
        # 34 m off and 43 degrees off square: 35 px per metre.
        photo = photo_at(-20.0, -25.0, 43.0, ident="far")
        coverage = score_views(self.faces, [photo])
        coverage.thresholds = {"min_visible": 0.6, "measure_px_per_m": 12, "texture_px_per_m": 40}
        front = self.face("left", [0, -1])
        self.assertEqual(coverage.face_class(front), MEASURE)

    def test_pano_of_unknown_heading_sees_around_itself(self) -> None:
        photo = photo_at(25.0, 4.0, 0.0, kind=SPHERICAL, ident="pano")
        photo.heading_known = False
        coverage = score_views(self.faces, [photo])
        seen = {self.faces[i].building_id for i, v in enumerate(coverage.views) if v}
        # Everything with a wall towards it, except the house behind its
        # neighbour.
        self.assertEqual(seen, {"right", "east", "shed"})

    def test_summary_and_map(self) -> None:
        photos = [photo_at(6.0, -18.0, 0.0, ident="front"), photo_at(45.0, -18.0, 0.0, ident="e")]
        coverage = score_views(self.faces, photos)
        coverage.thresholds = {"min_visible": 0.6, "measure_px_per_m": 12, "texture_px_per_m": 40}
        stats = coverage.stats()
        self.assertEqual(stats["photos_seeing_a_wall"], 2)
        self.assertEqual(stats["exposed_faces"], 14)
        self.assertGreater(stats["share_texture_ready"], 0.0)
        self.assertLessEqual(stats["share_texture_ready"], stats["share_measurable"])
        # Both houses in front of a camera, and the shed in front of the second.
        self.assertEqual(stats["buildings_with_texture_ready_wall"], 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = render_coverage_map(coverage, photos, BBox(-10, -20, 60, 20), Path(tmp) / "c.png")
            from PIL import Image

            with Image.open(path) as image:
                self.assertEqual(image.size, (140, 80))


class _Terrain:
    def sample(self, x, y):
        return np.zeros(np.shape(x))


class StageTests(unittest.TestCase):
    def test_stage_with_mapillary_and_own_photos(self) -> None:
        from src.geo import RD_CRS, WGS84_CRS, _transformer
        from src.street_photos import build_street_photos

        buildings = BuildingSet(buildings=[box_building("a", 136000, 455000, 136010, 455008)])
        to_wgs = _transformer(RD_CRS, WGS84_CRS)
        lon, lat = to_wgs.transform(136005.0, 454990.0)
        session = _FakeSession([(lon, lat)])
        bbox = BBox(135980, 454980, 136030, 455030)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = tmp / "mine"
            folder.mkdir()
            _jpeg(folder / "x.jpg")
            (folder / "x.jpg.json").write_text(
                json.dumps({"x": 136005.0, "y": 454992.0, "heading": 0})
            )
            config = {
                "mapillary": True, "folder": str(folder), "max_distance_m": 40.0,
                "spacing_m": 4.0, "min_year": 2016,
            }
            result = build_street_photos(
                bbox, tmp / "work", tmp / "out", photos_cfg=config,
                buildings=buildings, terrain=_Terrain(), base_dir=tmp, session=session,
            )
            self.assertTrue((tmp / "work" / "photos.json").is_file())
            self.assertTrue((tmp / "work" / "photo_coverage.json").is_file())
            self.assertTrue((tmp / "out" / "photo_coverage.png").is_file())
            saved = json.loads((tmp / "work" / "photo_coverage.json").read_text())
        stats = result.stats()
        self.assertEqual(stats["from_mapillary"], 1)
        self.assertEqual(stats["own"], 1)
        self.assertTrue(result.uses_mapillary)
        # The Mapillary camera looks 47 degrees east of the house front, which
        # leaves a corner of it in the frame: too little to count. Before the
        # distortion limit, points far outside its view folded back in and it
        # appeared to see the whole front.
        self.assertEqual(stats["coverage"]["photos_seeing_a_wall"], 1)
        self.assertEqual(len(saved["faces"]), 4)
        # Own photo placed at 1.6 m on the flat test terrain.
        own = [p for p in result.photos if p.source == SOURCE_OWN][0]
        self.assertAlmostEqual(own.z, 1.6)

    def test_mapillary_failure_does_not_fail_the_stage(self) -> None:
        import os
        from unittest import mock

        from src import mapillary
        from src.street_photos import build_street_photos

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {}, clear=True
        ), mock.patch.object(mapillary, "TOKEN_FILES", (Path("/nonexistent"),)):
            tmp = Path(tmp)
            result = build_street_photos(
                BBox(0, 0, 50, 50), tmp / "w", tmp / "o", photos_cfg={"mapillary": True},
                buildings=BuildingSet(), terrain=_Terrain(), base_dir=tmp,
            )
        self.assertEqual(result.photos, [])
        self.assertIn("MAPILLARY_TOKEN", result.problems[0])


class ConfigTests(unittest.TestCase):
    def test_token_in_config_is_dropped(self) -> None:
        from src.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({
                "name": "t",
                "bbox": {"crs": "EPSG:28992", "xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
                "photos": {"mapillary": True, "mapillary_token": "MLY|1|secret"},
            }))
            with self.assertLogs("src.config", logging.WARNING) as logs:
                config = load_config(path)
        self.assertNotIn("mapillary_token", config.photos)
        self.assertNotIn("secret", json.dumps(config.as_dict()))
        self.assertTrue(any("MAPILLARY_TOKEN" in line for line in logs.output))
        self.assertTrue(config.photos["mapillary"])

    def test_mapillary_is_a_soft_optional_source(self) -> None:
        from src.config import DEFAULTS
        from src.sources import BY_ID, enabled_sources

        source = BY_ID["mapillary"]
        self.assertTrue(source.soft)
        self.assertFalse(DEFAULTS["photos"]["mapillary"])
        self.assertNotIn(source, enabled_sources(DEFAULTS))
        self.assertIn(source, enabled_sources({"photos": {"mapillary": True}}))


def _server():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "server_under_test_photos", REPO_ROOT / "ui" / "server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _server()
        self.area = {
            "name": "t",
            "bbox": {"xmin": 136000, "ymin": 455000, "xmax": 137000, "ymax": 456000},
        }

    def test_photos_are_off_unless_asked_for(self) -> None:
        # An older caller sending one flag per source, and saying nothing
        # about photos, must not get a run that needs a key.
        config = self.server.build_config(dict(self.area))
        self.assertFalse(config.get("photos", {}).get("mapillary", False))
        config = self.server.build_config(dict(self.area, mapillary=True))
        self.assertTrue(config["photos"]["mapillary"])
        config = self.server.build_config(dict(self.area, sources=["mapillary"]))
        self.assertTrue(config["photos"]["mapillary"])

    def test_estimate_counts_photos_only_when_on(self) -> None:
        estimate = self.server.estimate_seconds
        base = {"bbox": self.area["bbox"], "size_px": 2048, "preview": False}
        self.assertNotIn("photos", estimate(base)["breakdown"])
        self.assertIn("photos", estimate(dict(base, mapillary=True))["breakdown"])
        self.assertIn("photos", estimate(dict(base, sources=["mapillary"]))["breakdown"])


if __name__ == "__main__":
    unittest.main()
