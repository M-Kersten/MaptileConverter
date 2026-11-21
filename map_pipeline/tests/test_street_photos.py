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
        self.assertTrue(self.faces[self.face("left", [1, 0])].party)
        self.assertTrue(self.faces[self.face("left", [-1, 0])].exposed)
        self.assertFalse(self.faces[self.face("left", [-1, 0])].party)

    def test_an_alley_wall_is_unexposed_but_no_party_wall(self) -> None:
        faces = wall_faces(BuildingSet(buildings=[
            box_building("w", 0, 0, 6, 8), box_building("e", 7.5, 0, 14, 8),
        ]))
        score_views(faces, [])
        alley = [f for f in faces if f.building_id == "w" and np.allclose(f.normal, [1, 0])][0]
        self.assertFalse(alley.exposed)
        self.assertFalse(alley.party)

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
                # The photos here are blank; checking them against the model
                # is tested on its own below.
                "refine": False,
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

    def test_a_photo_taken_on_a_bridge_stands_on_its_deck(self) -> None:
        """The survey under a canal bridge is the wharf or the water, four
        metres below the deck. A photo taken on the bridge was placed down
        there, inside the bridge, and saw none of what it shows."""
        from src.street_photos import build_street_photos

        def deck_at(x, y, reach=0.5):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            return np.where((x > 136000) & (x < 136010), 4.0, np.nan)

        buildings = BuildingSet(buildings=[box_building("a", 136020, 455000, 136030, 455008)])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = tmp / "mine"
            folder.mkdir()
            for name, x in (("on_bridge", 136005.0), ("on_street", 136015.0)):
                _jpeg(folder / f"{name}.jpg")
                (folder / f"{name}.jpg.json").write_text(
                    json.dumps({"x": x, "y": 454995.0, "heading": 0})
                )
            result = build_street_photos(
                BBox(135980, 454980, 136050, 455030), tmp / "work", tmp / "out",
                photos_cfg={"mapillary": False, "folder": str(folder), "refine": False},
                buildings=buildings, terrain=_Terrain(), base_dir=tmp, deck_at=deck_at,
            )
        heights = {Path(p.id).stem.split(":")[-1]: p.z for p in result.photos}
        on_bridge = [z for name, z in heights.items() if "on_bridge" in name]
        on_street = [z for name, z in heights.items() if "on_street" in name]
        self.assertAlmostEqual(on_bridge[0], 4.0 + 1.6)
        self.assertAlmostEqual(on_street[0], 1.6)

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


def _street() -> BuildingSet:
    """Four detached houses of different heights in a row, and one across.

    Detached so each shows its corners: where terraced houses meet there is
    no edge to see, and the check rightly leaves those lines out.
    """
    return BuildingSet(
        buildings=[
            box_building("a", 0, 0, 6, 9, height=9.0),
            box_building("b", 8, 0, 14, 9, height=12.0),
            box_building("c", 16, 0, 22, 9, height=7.5),
            box_building("e", 24, 0, 30, 9, height=10.0),
            box_building("d", 2, -30, 18, -22, height=10.0),
        ]
    )


def _render_edges(photo: Photo, segments, *, width: int = 2) -> np.ndarray:
    """A picture of the model's edges from `photo`, white on a grey street."""
    from PIL import Image, ImageDraw

    from src.camera import project_camera

    image = Image.new("RGB", (photo.camera.width, photo.camera.height), (90, 90, 90))
    draw = ImageDraw.Draw(image)
    for a, b in segments:
        pts = a[None] + np.linspace(0.0, 1.0, 24)[:, None] * (b - a)[None]
        px, ok = project_camera(photo.camera, (pts - photo.centre) @ photo.rotation.T)
        for i in range(len(pts) - 1):
            if ok[i] and ok[i + 1]:
                draw.line([tuple(px[i]), tuple(px[i + 1])], fill=(235, 235, 235), width=width)
    return np.asarray(image)


class PoseRefineTests(unittest.TestCase):
    def setUp(self) -> None:
        from src.photo_coverage import _SegmentIndex
        from src.pose_refine import building_edges, sample_edges

        self.buildings = _street()
        from src.pose_refine import drop_hidden_seams

        self.segments, self.owner = building_edges(self.buildings)
        faces = wall_faces(self.buildings)
        score_views(faces, [])  # marks the party walls
        # As the stage does: the line where two houses meet is no edge.
        self.samples = drop_hidden_seams(sample_edges(self.segments, owner=self.owner), faces)
        self.index = _SegmentIndex(faces)
        truth = photo_at(15.0, -14.0, 0.0, z=2.0, ident="t")
        truth.camera = Camera(PERSPECTIVE, 1600, 1200, 0.55)
        truth.rotation = rotation_from_heading(4.0, pitch_deg=8.0)
        truth.pose = POSE_SFM
        truth.position_sigma_m = 1.5
        self.truth = truth

    def test_feature_edges_are_creases_not_triangulation(self) -> None:
        from src.pose_refine import feature_edges

        house = box_building("x", 0, 0, 10, 8, height=9.0)
        edges = feature_edges(np.concatenate([house.wall_tris, house.roof_tris]))
        # Twelve box edges; the diagonals that split each face are flat.
        self.assertEqual(len(edges), 12)

    def test_wall_foot_is_left_out(self) -> None:
        from src.pose_refine import building_edges

        segments, owner = building_edges(BuildingSet(buildings=[box_building("x", 0, 0, 10, 8)]))
        self.assertEqual(len(segments), 8)
        self.assertTrue(np.all(segments[:, :, 2].max(axis=1) > 1.0))

    def test_truncated_distance(self) -> None:
        from src.pose_refine import truncated_distance

        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True
        d = truncated_distance(mask, limit=3)
        self.assertEqual(d[4, 4], 0)
        self.assertEqual(d[4, 6], 2)
        self.assertEqual(d[0, 0], 3)

    def test_recovers_a_displaced_camera(self) -> None:
        from src.pose_refine import align_photo

        image = _render_edges(self.truth, self.segments)
        start = Photo(**{**self.truth.__dict__})
        start.rotation = rotation_from_heading(6.5, pitch_deg=8.0)
        start.x, start.y = self.truth.x + 0.9, self.truth.y - 0.6
        refined, alignment = align_photo(
            start, image, self.samples, self.index
        )
        self.assertTrue(alignment.verified, alignment.to_dict())
        self.assertGreater(alignment.after, alignment.before + 0.3)
        self.assertGreater(alignment.after, 0.6)
        self.assertLess(abs(refined.heading_deg - 4.0), 0.6)
        self.assertLess(np.hypot(refined.x - self.truth.x, refined.y - self.truth.y), 0.6)
        self.assertTrue(refined.verified)

    def test_verdict_is_per_building(self) -> None:
        from src.pose_refine import align_photo

        ids = [b.identifier for b in self.buildings.buildings]
        # House c has been pulled down, or is behind scaffolding: it is not
        # in the picture. Its neighbours are.
        drawn = self.segments[self.owner != ids.index("c")]
        refined, alignment = align_photo(
            self.truth, _render_edges(self.truth, drawn), self.samples, self.index,
            building_ids=ids,
        )
        self.assertTrue(alignment.verified, alignment.to_dict())
        self.assertIn("b", refined.verified_buildings)
        self.assertNotIn("c", refined.verified_buildings)

        faces = wall_faces(self.buildings)
        coverage = score_views(faces, [refined])
        by_building = {}
        for face, views in zip(faces, coverage.views):
            for view in views:
                by_building.setdefault(face.building_id, set()).add(view.verified)
        self.assertEqual(by_building["c"], {False})
        self.assertIn(True, by_building["b"])

    def test_a_photo_of_something_else_is_not_verified(self) -> None:
        from src.pose_refine import align_photo

        rng = np.random.default_rng(5)
        noise = (rng.random((1200, 1600, 3)) * 255).astype(np.uint8)
        refined, alignment = align_photo(
            self.truth, noise, self.samples, self.index
        )
        self.assertFalse(alignment.verified)
        self.assertFalse(refined.verified)
        self.assertTrue(alignment.reason)

    def test_too_little_model_in_view(self) -> None:
        from src.pose_refine import align_photo

        away = Photo(**{**self.truth.__dict__})
        away.rotation = rotation_from_heading(184.0)
        _, alignment = align_photo(
            away, _render_edges(away, self.segments), self.samples, self.index
        )
        self.assertFalse(alignment.verified)
        self.assertIn("model edge points in view", alignment.reason)

    def test_sequence_outlier_is_withdrawn(self) -> None:
        from src.pose_refine import Alignment, check_sequences

        photos, alignments = {}, {}
        for i, (dx, yaw) in enumerate(((0.5, 0.2), (0.6, 0.1), (0.4, 0.3), (2.9, 0.2))):
            pid = f"p{i}"
            photos[pid] = photo_at(10.0 * i, 0.0, 0.0, ident=pid, sequence="s")
            alignments[pid] = Alignment(pid, 0.2, 0.6, 300, (yaw, 0, 0, dx, 0, 0), True)
        withdrawn = check_sequences(photos, alignments)
        self.assertEqual(withdrawn, ["p3"])
        self.assertFalse(alignments["p3"].verified)
        self.assertTrue(alignments["p0"].verified)

    def test_rejected_photos_do_not_count_and_texture_needs_a_check(self) -> None:
        faces = wall_faces(self.buildings)
        good = photo_at(11.0, -12.0, 0.0, ident="good")
        good.verified = True
        unchecked = photo_at(3.0, -12.0, 0.0, ident="unchecked")
        bad = photo_at(19.0, -12.0, 0.0, ident="bad")
        bad.verified = False
        coverage = score_views(faces, [good, unchecked, bad])
        coverage.thresholds = {
            "min_visible": 0.5, "measure_px_per_m": 12, "texture_px_per_m": 40,
            "texture_needs_verified": True,
        }
        used = {v.photo for views in coverage.views for v in views}
        self.assertNotIn("bad", used)
        self.assertEqual(used, {"good", "unchecked"})
        # Every face's best view is a checked one where there is one.
        for views in coverage.views:
            if any(v.photo == "good" for v in views):
                self.assertEqual(views[0].photo, "good")


# ---------------------------------------------------------------------------
# Phase 3: straightened walls, measured facades, observed styles
# ---------------------------------------------------------------------------


def _paint_wall(photo: Photo, face, pattern, background=(70, 120, 200)) -> np.ndarray:
    """A picture of one wall from `photo`, coloured by pattern(u, z).

    Every pixel's ray is followed to the wall's plane; where it lands inside
    the wall it takes the pattern's colour, elsewhere the sky's.
    """
    from src.camera import unproject_camera

    cam = photo.camera
    cols, rows = np.meshgrid(np.arange(cam.width) + 0.5, np.arange(cam.height) + 0.5)
    rays = unproject_camera(cam, np.column_stack([cols.ravel(), rows.ravel()])) @ photo.rotation
    normal = np.array([face.normal[0], face.normal[1], 0.0])
    origin = np.array([face.start[0], face.start[1], 0.0])
    denom = rays @ normal
    t = ((origin - photo.centre) @ normal) / np.where(np.abs(denom) > 1e-9, denom, np.nan)
    hit = photo.centre + rays * t[:, None]
    tangent = (face.end - face.start) / face.width_m
    u = (hit[:, :2] - face.start) @ tangent
    z = hit[:, 2]
    inside = (t > 0) & (u >= 0) & (u <= face.width_m) & (z >= face.z_min) & (z <= face.z_max)
    out = np.tile(np.array(background, dtype=np.float64), (len(u), 1))
    out[inside] = pattern(u[inside], z[inside])
    return out.reshape(cam.height, cam.width, 3).astype(np.uint8)


def _facade_pattern(bay=2.5, storey=3.0, win_w=0.4, win_h=0.5, sill=0.25,
                    wall=(150, 80, 60), glass=(40, 50, 60)):
    def pattern(u, z):
        in_bay = np.mod(u, bay) / bay
        in_storey = np.mod(z, storey) / storey
        window = (np.abs(in_bay - 0.5) < win_w / 2) & (in_storey >= sill) & (in_storey < sill + win_h)
        out = np.tile(np.array(wall, dtype=np.float64), (len(u), 1))
        out[window] = glass
        return out
    return pattern


def _wall_image(width_m=20.0, height_m=15.0, m_per_px=0.05, **pattern_args):
    from src.facade_capture import WallImage

    w, h = int(width_m / m_per_px), int(height_m / m_per_px)
    u = (np.arange(w) + 0.5) * m_per_px
    z = height_m - (np.arange(h) + 0.5) * m_per_px
    uu, zz = np.meshgrid(u, z)
    rgb = _facade_pattern(**pattern_args)(uu.ravel(), zz.ravel()).reshape(h, w, 3)
    rng = np.random.default_rng(0)
    rgb = np.clip(rgb + rng.normal(0, 4, rgb.shape), 0, 255).astype(np.float32)
    return WallImage(face=0, rgb=rgb, weight=np.ones((h, w), np.float32),
                     m_per_px=m_per_px, width_m=width_m, z_min=0.0, z_max=height_m)


class FacadeCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.buildings = BuildingSet(buildings=[box_building("x", 0, 0, 12, 8, height=9.0)])
        self.faces = wall_faces(self.buildings)
        self.front = next(i for i, f in enumerate(self.faces) if np.allclose(f.normal, [0, -1]))

    def test_a_wall_comes_out_straight_and_the_right_way_round(self) -> None:
        from src.facade_capture import PhotoPixels, capture_wall

        face = self.faces[self.front]
        # Left half red, right half green, seen from the street.
        def pattern(u, z):
            out = np.zeros((len(u), 3))
            out[u < 6.0] = (200, 40, 40)
            out[u >= 6.0] = (40, 200, 40)
            out[(z > 7.0) & (u < 6.0)] = (40, 40, 200)
            return out
        photo = photo_at(5.0, -14.0, 0.0, z=2.0, ident="front")
        photo.rotation = rotation_from_heading(10.0, pitch_deg=12.0)
        photo.camera = Camera(PERSPECTIVE, 1600, 1200, 0.6)
        image = _paint_wall(photo, face, pattern)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "front.png"
            from PIL import Image

            Image.fromarray(image).save(path)
            photo.path = str(path)
            wall = capture_wall(face, self.front, [photo], PhotoPixels(), m_per_px=0.1)
        self.assertGreater(wall.coverage, 0.9)
        rgb = wall.rgb
        h, w = rgb.shape[:2]
        left_low = rgb[int(h * 0.8), int(w * 0.25)]
        right_low = rgb[int(h * 0.8), int(w * 0.75)]
        left_high = rgb[int(h * 0.1), int(w * 0.25)]
        self.assertGreater(left_low[0], 150)  # red on the left
        self.assertGreater(right_low[1], 150)  # green on the right
        self.assertGreater(left_high[2], 150)  # blue above 7 m, row 0 at the top

    def test_registration_finds_a_shift(self) -> None:
        from src.facade_capture import register_shift

        a = _wall_image(width_m=10, height_m=8, m_per_px=0.05)
        rgb = np.roll(a.rgb, (3, -5), axis=(0, 1))
        dr, dc, peak = register_shift((a.rgb, a.weight), (rgb, a.weight), 10)
        self.assertEqual((dr, dc), (-3, 5))
        self.assertGreater(peak, 0.05)

    def test_fusion_drops_what_only_one_view_shows(self) -> None:
        from src.facade_capture import fuse_views

        a = _wall_image(width_m=10, height_m=8)
        views = [(a.rgb.copy(), a.weight.copy()) for _ in range(3)]
        views[1][0][100:140, 50:120] = (250, 250, 250)  # a white van in one photo
        for mode in ("blend", "best"):
            fused, weight = fuse_views(views, mode=mode, m_per_px=0.05)
            patch = fused[100:140, 50:120]
            self.assertLess(float(np.abs(patch - a.rgb[100:140, 50:120]).mean()), 12.0, mode)
            self.assertTrue(np.all(fused <= 255.0))

    def test_disagreement_is_reported(self) -> None:
        from src.facade_capture import fuse_views

        a = _wall_image(width_m=10, height_m=8)
        views = [(a.rgb.copy(), a.weight.copy()) for _ in range(2)]
        views[0][0][120:150, 50:120] = (250, 250, 250)  # a white car in one
        views[1][0][120:150, 50:120] = (20, 20, 30)  # a black car in the other
        _, _, agreement = fuse_views(views, mode="best", m_per_px=0.05, with_agreement=True)
        self.assertLess(float(agreement[120:150, 50:120].mean()), 0.2)
        self.assertGreater(float(agreement[20:60, 50:120].mean()), 0.9)

    def test_something_in_front_of_the_wall_is_inconsistent(self) -> None:
        from src.facade_capture import photo_consistency

        a = _wall_image(width_m=10, height_m=8)
        rng = np.random.default_rng(4)
        car = rng.uniform(20, 230, size=(30, 70, 3)).astype(np.float32)
        views = [(a.rgb.copy(), a.weight.copy()) for _ in range(2)]
        # The same car, seen from two places, lands 0.5 m apart on the wall.
        views[0][0][120:150, 40:110] = car
        views[1][0][120:150, 50:120] = car
        consistency = photo_consistency(views, 0.05)
        self.assertLess(float(np.median(consistency[125:145, 55:105])), 0.5)
        self.assertGreater(float(np.median(consistency[20:60, :])), 0.8)

    def test_gable_sky_is_masked(self) -> None:
        from src.facade_capture import face_outline

        # A gable end: a pentagon, 8 m wide, eaves at 6 m, ridge at 10 m.
        a, b = np.array([0.0, 0, 0]), np.array([8.0, 0, 0])
        e1, e2, top = np.array([8.0, 0, 6]), np.array([0.0, 0, 6]), np.array([4.0, 0, 10])
        tris = np.array([[a, b, e1], [a, e1, e2], [e2, e1, top]])
        building = Building("g", tris, np.zeros((0, 3, 3)), 0.0, 10.0, 2, wall_top_nap=6.0)
        face = wall_faces(BuildingSet(buildings=[building]))[0]
        mask = face_outline(face, (100, 80))
        self.assertFalse(mask[2, 2])  # top left corner is sky
        self.assertTrue(mask[2, 40])  # under the ridge is wall
        self.assertTrue(mask[90, 5])

    def test_tree_crown_in_the_way(self) -> None:
        from src.facade_capture import Crown, _segment_hits_spheres

        origin = np.array([0.0, -10.0, 2.0])
        targets = np.array([[0.0, 0.0, 3.0], [8.0, 0.0, 3.0]])
        hits = _segment_hits_spheres(origin, targets, [Crown(0.0, -5.0, 3.0, 1.5)])
        self.assertEqual(list(hits), [True, False])


class FacadeMeasureTests(unittest.TestCase):
    def test_reads_a_regular_facade(self) -> None:
        from src.facade_measure import measure_wall

        wall = _wall_image(width_m=25.0, height_m=15.0, bay=2.5, storey=3.0,
                           win_w=0.4, win_h=0.5, sill=0.25)
        m = measure_wall(wall, ground_z=0.0, wall_top_z=15.0, building_id="b", area_m2=375)
        self.assertAlmostEqual(m.bay_m, 2.5, delta=0.1)
        self.assertAlmostEqual(m.storey_m, 3.0, delta=0.12)
        self.assertAlmostEqual(m.window_width_frac, 0.4, delta=0.1)
        self.assertAlmostEqual(m.window_height_frac, 0.5, delta=0.1)
        self.assertAlmostEqual(m.sill_frac, 0.25, delta=0.08)
        for got, want in zip(m.wall_rgb, (150, 80, 60)):
            self.assertAlmostEqual(got, want, delta=10)

    def test_fundamental_not_a_multiple(self) -> None:
        from src.facade_measure import dominant_period

        x = np.arange(0, 40, 0.05)
        profile = (np.mod(x, 2.0) < 0.8).astype(float)
        period, score = dominant_period(profile, 0.05, 1.0, 8.0)
        self.assertAlmostEqual(period, 2.0, delta=0.06)
        self.assertGreater(score, 0.5)

    def test_wide_windows_read_as_wide(self) -> None:
        from src.facade_measure import measure_wall

        # Windows 85% of each bay, between slim piers. Past about 90% glass
        # and wall cover the same share of the front, and which is the wall
        # is a question the colours cannot settle: that is a facade for a
        # photo texture, not for measuring.
        wall = _wall_image(width_m=24.0, height_m=15.0, bay=1.7, storey=3.6,
                           win_w=0.85, win_h=0.5, sill=0.3, wall=(90, 95, 105), glass=(160, 190, 210))
        m = measure_wall(wall, ground_z=0.0, wall_top_z=15.0)
        self.assertGreaterEqual(m.window_width_frac, 0.75)

    def test_a_blank_wall_has_no_windows(self) -> None:
        from src.facade_capture import WallImage
        from src.facade_measure import measure_wall

        rng = np.random.default_rng(2)
        rgb = np.clip(np.full((300, 400, 3), (180, 170, 150.0)) + rng.normal(0, 5, (300, 400, 3)), 0, 255)
        wall = WallImage(0, rgb.astype(np.float32), np.ones((300, 400), np.float32), 0.05, 20.0, 0.0, 15.0)
        m = measure_wall(wall, ground_z=0.0, wall_top_z=15.0)
        self.assertEqual(m.bay_m, 0.0)
        self.assertIsNotNone(m.wall_rgb)

    def test_combining_walls(self) -> None:
        from src.facade_measure import WallMeasurement, combine_walls

        walls = [
            WallMeasurement(1, "b", 300.0, 1.0, wall_rgb=(240, 240, 238), storey_m=2.7, storey_confidence=0.6),
            WallMeasurement(2, "b", 200.0, 1.0, wall_rgb=(150, 80, 60), storey_m=2.7, storey_confidence=0.6),
            WallMeasurement(3, "b", 100.0, 1.0, wall_rgb=(60, 90, 160)),
        ]
        m = combine_walls("b", walls, ground_z=0.0, wall_top_z=14.4, model_floors=5)
        # One of the walls' own colours, never a mixture of them.
        self.assertIn(m.wall_rgb, [w.wall_rgb for w in walls])
        # A ground storey and four of 2.7 m above it.
        self.assertEqual(m.storeys, 5)
        far = combine_walls("b", walls, ground_z=0.0, wall_top_z=14.4, model_floors=12)
        self.assertEqual(far.storeys, 0)

    def test_brightness_gain_rules(self) -> None:
        from src.photo_colour import wall_gain

        self.assertAlmostEqual(wall_gain((80, 60, 50), 2.0, strength=1.0), 2.0)
        self.assertAlmostEqual(wall_gain((80, 60, 50), 2.0, strength=0.7), 2.0 ** 0.7)
        # A pale wall stops at the ceiling.
        g = wall_gain((170, 165, 150), 2.0)
        self.assertLess(g, 1.15)
        # A wall in full sun is brought down even with no gain.
        self.assertLess(wall_gain((250, 248, 245), 1.0), 1.0)

    def test_photo_gain_against_the_aerial(self) -> None:
        from src.camera import project_camera  # noqa: F401
        from src.geo import BBox
        from src.photo_colour import AerialColours, photo_gain

        aerial = AerialColours(np.full((200, 200, 3), 160.0), BBox(-50, -50, 50, 50))
        photo = photo_at(0.0, 0.0, 0.0, kind=SPHERICAL, z=2.0, ident="p")
        photo.camera = Camera(SPHERICAL, 1024, 512)
        # Ground at half the aerial's brightness, sky white.
        image = np.full((512, 1024, 3), 80, dtype=np.uint8)
        image[:200] = 250
        gain, details = photo_gain(photo, image, aerial, lambda x, y: np.zeros(np.shape(x)))
        self.assertIsNotNone(gain, details)
        self.assertAlmostEqual(float(gain[0]), 2.0, delta=0.1)


class ObservedStyleTests(unittest.TestCase):
    def test_walls_get_looks_and_buildings_their_largest(self) -> None:
        from src.facade_measure import FacadeMeasurement, WallMeasurement
        from src.facade_observed import plan_observed

        buildings = BuildingSet(buildings=[
            box_building("a", 0, 0, 10, 8), box_building("b", 20, 0, 30, 8),
        ])
        for b in buildings.buildings:
            b.build_year = 1930
        brick = dict(wall_rgb=(150, 80, 60), bay_m=2.4, window_width_frac=0.4,
                     window_height_frac=0.5, sill_frac=0.25)
        measurements = {
            "a": FacadeMeasurement("a", [
                WallMeasurement(10, "a", 90.0, 1.0, **brick),
                WallMeasurement(11, "a", 30.0, 1.0, wall_rgb=(230, 230, 228)),
            ], storeys=3),
            # Same brick on b: shares a's style.
            "b": FacadeMeasurement("b", [WallMeasurement(20, "b", 80.0, 1.0, **dict(brick, wall_rgb=(152, 82, 61)))]),
        }
        keys = {10: (0, 1, 2), 11: (0, 3, 4), 20: (1, 1, 2)}
        plan = plan_observed(measurements, buildings, n_variants=5, max_styles=24, face_keys=keys)
        self.assertEqual(len(plan.styles), 2)
        self.assertEqual(plan.face_look[(0, 1, 2)], plan.face_look[(1, 1, 2)])
        self.assertNotEqual(plan.face_look[(0, 1, 2)], plan.face_look[(0, 3, 4)])
        self.assertEqual(plan.look_of[0], plan.face_look[(0, 1, 2)])
        self.assertAlmostEqual(plan.face_tile_m[(0, 1, 2)], 2.4)
        self.assertEqual(plan.floors, {0: 3})
        style = plan.styles[plan.face_look[(0, 1, 2)]]
        self.assertEqual(style.windows_across, 1)
        self.assertEqual(style.wall_rgb, (150, 80, 60))
        self.assertEqual(style.family, "interbellum")

    def test_buildings_file_carries_per_wall_styles(self) -> None:
        from src.buildings import save_buildings

        buildings = BuildingSet(buildings=[box_building("a", 0, 0, 10, 8)])
        faces = wall_faces(buildings)
        front = next(f for f in faces if np.allclose(f.normal, [0, -1]))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_buildings(
                buildings, Path(tmp) / "b.npz", style_index=np.array([3]),
                facade_cfg={"tile_width_m": 4.0},
                face_style={front.key: 9}, face_tile_width={front.key: 2.5},
            )
            data = np.load(path)
            wall_style = data["wall_style"]
            normals = np.cross(data["wall_tris"][:, 1] - data["wall_tris"][:, 0],
                               data["wall_tris"][:, 2] - data["wall_tris"][:, 0])
            facing_south = normals[:, 1] < 0
            self.assertTrue(np.all(wall_style[facing_south] == 9))
            self.assertTrue(np.all(wall_style[~facing_south] == 3))
            # 10 m front at a measured 2.5 m bay: four bays across.
            u = data["wall_u"].reshape(-1, 3)[facing_south]
            self.assertAlmostEqual(float(u.max() - u.min()), 4.0, places=6)

    def test_facade_u_runs_left_to_right_from_outside(self) -> None:
        from src.facade_uv import fit_wall_u

        # The south wall of a box, seen from the south: west is left.
        house = box_building("a", 0, 0, 10, 8)
        south = house.wall_tris[:2]
        u = fit_wall_u(south, np.zeros(2, dtype=np.int32), np.array([4.0])).reshape(-1)
        x = south.reshape(-1, 3)[:, 0]
        self.assertGreater(np.corrcoef(x, u)[0, 1], 0.99)


# ---------------------------------------------------------------------------
# Phase 4: photo textures
# ---------------------------------------------------------------------------


class PhotoTextureTests(unittest.TestCase):
    def test_licence_rule(self) -> None:
        from src.photo_textures import may_texture

        mine = photo_at(0, 0, 0, ident="o", source=SOURCE_OWN)
        theirs = photo_at(0, 0, 0, ident="m")
        self.assertTrue(may_texture(mine, "own"))
        self.assertFalse(may_texture(theirs, "own"))
        self.assertTrue(may_texture(theirs, "all"))
        self.assertFalse(may_texture(mine, "off"))

    def test_packing_keeps_walls_apart(self) -> None:
        from src.photo_textures import PAD_PX, pack

        rng = np.random.default_rng(3)
        sizes = [(int(w), int(h)) for w, h in rng.integers(50, 900, size=(60, 2))] + [(5000, 300)]
        spots, scales = pack(sizes, page_px=2048)
        rects = []
        for (w, h), (page, x, y), scale in zip(sizes, spots, scales):
            w, h = max(1, int(w * scale)), max(1, int(h * scale))
            self.assertLessEqual(x + w + PAD_PX, 2048)
            self.assertLessEqual(y + h + PAD_PX, 2048)
            rects.append((page, x - PAD_PX, y - PAD_PX, x + w + PAD_PX, y + h + PAD_PX))
        self.assertLess(scales[-1], 1.0)  # wider than a page: shrunk to fit
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] != b[0]:
                    continue
                overlap = min(a[3], b[3]) > max(a[1], b[1]) and min(a[4], b[4]) > max(a[2], b[2])
                self.assertFalse(overlap, (a, b))

    def test_holes_filled_from_the_next_bay(self) -> None:
        from src.photo_textures import fill_from_rhythm

        wall = _wall_image(width_m=20.0, height_m=12.0, bay=2.5, storey=3.0)
        truth = wall.rgb.copy()
        weight = wall.weight.copy()
        weight[60:100, 100:140] = 0.0  # a tree in front
        z = wall.z_of_row(np.arange(wall.rgb.shape[0]))
        damaged = wall.rgb.copy()
        damaged[60:100, 100:140] = 0.0
        filled, w = fill_from_rhythm(damaged, weight, m_per_px=0.05, bay_m=2.5, storey_m=3.0,
                                     z_of_row=z, ground_top_z=None)
        self.assertLess(float(np.abs(filled[60:100, 100:140] - truth[60:100, 100:140]).mean()), 8.0)
        self.assertTrue(np.all(w[60:100, 100:140] == 0.6))

    def test_ground_storey_never_copies_from_above(self) -> None:
        from src.photo_textures import fill_from_rhythm

        wall = _wall_image(width_m=10.0, height_m=12.0, bay=2.5, storey=3.0)
        weight = wall.weight.copy()
        z = wall.z_of_row(np.arange(wall.rgb.shape[0]))
        ground = z < 3.6
        weight[ground] = 0.0  # every bay of the ground storey hidden
        filled, w = fill_from_rhythm(wall.rgb, weight, m_per_px=0.05, bay_m=2.5, storey_m=3.0,
                                     z_of_row=z, ground_top_z=3.6)
        self.assertTrue(np.all(w[ground] == 0.0))

    def test_flatten_light_keeps_detail(self) -> None:
        from src.photo_textures import flatten_light
        from src.facade_capture import luminance

        wall = _wall_image(width_m=20.0, height_m=12.0)
        shaded = wall.rgb.copy()
        shaded[:, : shaded.shape[1] // 2] *= 0.5  # a building's shadow over half
        out = flatten_light(shaded, wall.weight, 0.05, strength=1.0)
        left = luminance(out[:, 20:150]).mean()
        right = luminance(out[:, -150:-20]).mean()
        self.assertLess(abs(left - right) / right, 0.2)
        # The windows are still darker than the wall beside them.
        self.assertGreater(float(np.std(luminance(out))), 10.0)

    def test_placement_uvs_land_on_the_wall(self) -> None:
        from src.photo_textures import Placement

        placement = Placement(page=0, x=100, y=50, width=200, height=90,
                              start=(10.0, 0.0), tangent=(1.0, 0.0), z_max=9.0,
                              m_per_px_u=0.05, m_per_px_z=0.1)
        corners = np.array([[10.0, 0.0, 9.0], [20.0, 0.0, 9.0], [10.0, 0.0, 0.0]])
        uv = placement.uv(corners, (1000, 500))
        np.testing.assert_allclose(uv[0], [100 / 1000, 1 - 50 / 500])
        np.testing.assert_allclose(uv[1], [300 / 1000, 1 - 50 / 500])
        np.testing.assert_allclose(uv[2], [100 / 1000, 1 - 140 / 500])

    def test_bake_end_to_end(self) -> None:
        from src.photo_coverage import _SegmentIndex
        from src.photo_textures import bake_textures, triangle_photo_uvs
        from src.facade_uv import wall_face_keys
        from src.street_photos import StreetPhotoResult

        buildings = BuildingSet(buildings=[box_building("x", 0, 0, 12, 8, height=9.0)])
        for b in buildings.buildings:
            b.wall_top_nap = 9.0
        faces = wall_faces(buildings)
        front_index = next(i for i, f in enumerate(faces) if np.allclose(f.normal, [0, -1]))
        front = faces[front_index]
        photo = photo_at(6.0, -12.0, 0.0, z=2.0, ident="own_front", source=SOURCE_OWN)
        photo.rotation = rotation_from_heading(0.0, pitch_deg=12.0)
        photo.camera = Camera(PERSPECTIVE, 1600, 1200, 0.6)
        photo.verified = True
        photo.verified_buildings = ["x"]
        image = _paint_wall(photo, front, _facade_pattern())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            from PIL import Image

            Image.fromarray(image).save(tmp / "front.png")
            photo.path = str(tmp / "front.png")
            result = StreetPhotoResult(photos=[photo])
            coverage = score_views(faces, [photo])
            coverage.thresholds = {"min_visible": 0.6, "measure_px_per_m": 12,
                                   "texture_px_per_m": 40, "texture_needs_verified": True}
            textures = bake_textures(
                result, coverage, buildings, tmp, tmp, photos_cfg={"textures": "own"},
                faces_index=_SegmentIndex(faces),
            )
            self.assertEqual(list(textures.placements), [front.key])
            self.assertTrue(textures.pages[0].is_file())
            credits = json.loads((tmp / "photo_credits.json").read_text())
            self.assertEqual(credits["photos"][0]["photo"], "own_front")
            page = np.asarray(Image.open(textures.pages[0]).convert("RGB"))
            tris = buildings.buildings[0].wall_tris
            face, keys = wall_face_keys(tris, np.zeros(len(tris), dtype=np.int64))
            pages, uv = triangle_photo_uvs(tris, keys[face], textures)
        on = pages >= 0
        self.assertEqual(int(on.sum()), 2)  # the two triangles of the front
        # The middle of the front, read back through its UVs, is the facade:
        # brick or glass, never the sky.
        mid = uv[on].reshape(-1, 2).mean(axis=0)
        h, w = page.shape[:2]
        colour = page[int((1 - mid[1]) * h), int(mid[0] * w)]
        self.assertTrue(colour[2] < 120, colour)


class PinTests(unittest.TestCase):
    def _wall(self):
        buildings = BuildingSet(buildings=[box_building("x", 0, 0, 12, 8, height=9.0)])
        buildings.buildings[0].wall_top_nap = 9.0
        faces = wall_faces(buildings)
        score_views(faces, [])
        front = next(i for i, f in enumerate(faces) if np.allclose(f.normal, [0, -1]))
        return buildings, faces, front

    def test_pose_from_four_corners(self) -> None:
        from src.camera import project_world
        from src.photo_pin import pose_from_corners, wall_corners

        buildings, faces, front = self._wall()
        corners = wall_corners(faces[front], buildings.buildings[0])
        for kind in (PERSPECTIVE, SPHERICAL):
            truth = photo_at(4.0, -11.0, 0.0, kind=kind, z=1.6, ident="t")
            truth.rotation = rotation_from_heading(12.0, pitch_deg=9.0, roll_deg=2.0)
            truth.camera = Camera(kind, 2000, 1000 if kind == SPHERICAL else 1500, 0.7)
            pixels, ok = project_world(truth.camera, truth.rotation, truth.centre, corners)
            self.assertTrue(ok.all())
            rotation, centre, _, error = pose_from_corners(truth.camera, pixels, corners)
            self.assertLess(error, 0.5, kind)
            np.testing.assert_allclose(centre, truth.centre, atol=0.02, err_msg=kind)
            np.testing.assert_allclose(rotation, truth.rotation, atol=1e-3, err_msg=kind)

    def test_focal_found_when_unknown(self) -> None:
        from src.camera import project_world
        from src.photo_pin import pose_from_corners, wall_corners

        buildings, faces, front = self._wall()
        corners = wall_corners(faces[front], buildings.buildings[0])
        truth = photo_at(2.0, -9.0, 0.0, z=1.6, ident="t")
        truth.rotation = rotation_from_heading(25.0, pitch_deg=14.0)
        truth.camera = Camera(PERSPECTIVE, 2000, 1500, 0.62)
        pixels, _ = project_world(truth.camera, truth.rotation, truth.centre, corners)
        guess = Camera(PERSPECTIVE, 2000, 1500, 0.85)
        rotation, centre, camera, error = pose_from_corners(guess, pixels, corners, estimate_focal=True)
        self.assertAlmostEqual(camera.focal, 0.62, delta=0.02)
        np.testing.assert_allclose(centre, truth.centre, atol=0.1)

    def test_pin_places_a_photo_without_gps(self) -> None:
        from src.camera import project_world
        from src.own_photos import read_folder
        from src.photo_pin import apply_pin, wall_corners

        buildings, faces, front = self._wall()
        corners = wall_corners(faces[front], buildings.buildings[0])
        truth = photo_at(6.0, -10.0, 0.0, z=1.5, ident="t")
        truth.rotation = rotation_from_heading(-5.0, pitch_deg=10.0)
        truth.camera = Camera(PERSPECTIVE, 1600, 1200, 0.7)
        pixels, _ = project_world(truth.camera, truth.rotation, truth.centre, corners)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _jpeg(folder / "yard.jpg", (1600, 1200), exif=_gps_exif(0.0, 0.0, None, None) if False else None)
            (folder / "yard.jpg.json").write_text(json.dumps({
                "focal": 0.7,
                "pin": {"corners": pixels.tolist(), "x": 6.0, "y": 0.5},
            }))
            photo = read_folder(folder).photos[0]
        self.assertEqual(photo.pose, POSE_MANUAL)
        placed, why = apply_pin(photo, faces, buildings)
        self.assertIsNotNone(placed, why)
        np.testing.assert_allclose(placed.centre, truth.centre, atol=0.05)
        self.assertTrue(placed.verified)
        self.assertEqual(placed.verified_buildings, ["x"])

    def test_corners_in_the_wrong_order_are_refused(self) -> None:
        from src.camera import project_world
        from src.photo_pin import apply_pin, wall_corners

        buildings, faces, front = self._wall()
        corners = wall_corners(faces[front], buildings.buildings[0])
        truth = photo_at(6.0, -10.0, 0.0, z=1.5, ident="t")
        truth.rotation = rotation_from_heading(0.0, pitch_deg=10.0)
        truth.camera = Camera(PERSPECTIVE, 1600, 1200, 0.7)
        pixels, _ = project_world(truth.camera, truth.rotation, truth.centre, corners)
        truth.pin = {"corners": pixels[[0, 2, 1, 3]].tolist(), "x": 6.0, "y": 0.5}
        placed, why = apply_pin(truth, faces, buildings)
        self.assertIsNone(placed)
        self.assertIn("corners", why)


class PhotoFolderTests(unittest.TestCase):
    def _jpeg_bytes(self, size=(400, 300), exif=None) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", size, (100, 120, 140)).save(buffer, "JPEG", **({"exif": exif} if exif is not None else {}))
        return buffer.getvalue()

    def test_upload_place_pin_and_delete(self) -> None:
        from src import photo_folder

        with tempfile.TemporaryDirectory() as tmp:
            folder = photo_folder.folder_for(Path(tmp), "arnhem")
            name = photo_folder.save_upload(folder, "../../Hof van IMG 1.JPG", self._jpeg_bytes(exif=_gps_exif(51.98, 5.91, 90.0, 26)))
            self.assertEqual(name, "Hof_van_IMG_1.jpg")
            again = photo_folder.save_upload(folder, "Hof van IMG 1.jpg", self._jpeg_bytes())
            self.assertEqual(again, "Hof_van_IMG_1-1.jpg")
            listed = {p["name"]: p for p in photo_folder.describe(folder)}
            self.assertTrue(listed[name]["usable"])
            self.assertIsNotNone(listed[name]["heading"])
            self.assertFalse(listed[again]["usable"])  # no GPS, no sidecar
            photo_folder.set_pose(folder, again, x=190900.0, y=443400.0, heading=45.0)
            listed = {p["name"]: p for p in photo_folder.describe(folder)}
            self.assertTrue(listed[again]["usable"])
            self.assertAlmostEqual(listed[again]["heading"], 45.0, places=3)
            self.assertTrue(listed[again]["placed_by_hand"])
            photo_folder.set_pin(folder, again, corners=[[1, 2], [3, 4], [5, 6], [7, 8]], x=190910.0, y=443410.0)
            side = photo_folder.read_sidecar(folder, again)
            self.assertEqual(side["pin"]["corners"][2], [5.0, 6.0])
            self.assertEqual(side["heading"], 45.0)  # the pose survives the pin
            photo_folder.clear_pin(folder, again)
            self.assertNotIn("pin", photo_folder.read_sidecar(folder, again))
            thumb = photo_folder.thumbnail(folder, name)
            self.assertTrue(thumb.startswith(b"\xff\xd8"))
            photo_folder.delete_photo(folder, again)
            self.assertEqual([p["name"] for p in photo_folder.describe(folder)], [name])

    def test_refuses_what_is_not_a_photo(self) -> None:
        from src import photo_folder

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "a"
            with self.assertRaises(ValueError):
                photo_folder.save_upload(folder, "script.py", b"print(1)")
            with self.assertRaises(ValueError):
                photo_folder.save_upload(folder, "fake.jpg", b"not an image at all")
            with self.assertRaises(ValueError):
                photo_folder.set_pin(folder, "x.exe", corners=[], x=0, y=0)

    def test_build_uses_the_area_photos(self) -> None:
        from unittest import mock

        server = _server()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from src import photo_folder

            photo_folder.save_upload(photo_folder.folder_for(root, "plein"), "a.jpg", self._jpeg_bytes())
            with mock.patch.object(server, "REPO_ROOT", root):
                config = server.build_config({
                    "name": "plein", "bbox": {"xmin": 0, "ymin": 0, "xmax": 300, "ymax": 300},
                    "photo_textures": "all",
                })
                bare = server.build_config({"name": "leeg", "bbox": {"xmin": 0, "ymin": 0, "xmax": 300, "ymax": 300}})
        self.assertEqual(config["photos"]["folder"], str(root / "photos" / "plein"))
        self.assertEqual(config["photos"]["textures"], "all")
        self.assertNotIn("folder", bare.get("photos", {}))


class PhotoMatchTests(unittest.TestCase):
    def test_textured_walls_are_compared_first(self) -> None:
        from src.photo_match import choose_matches, write_photo_matches
        from src.photo_textures import PhotoTextureSet
        from src.street_photos import StreetPhotoResult

        buildings = BuildingSet(buildings=[box_building("a", 0, 0, 10, 8), box_building("b", 20, 0, 30, 8)])
        faces = wall_faces(buildings)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            photos = []
            for ident, x in (("pa", 5.0), ("pb", 25.0)):
                photo = photo_at(x, -12.0, 0.0, ident=ident, source=SOURCE_OWN)
                photo.camera = Camera(PERSPECTIVE, 640, 480, 0.6)
                photo.verified = True
                _jpeg(tmp / f"{ident}.jpg", (640, 480))
                photo.path = str(tmp / f"{ident}.jpg")
                photos.append(photo)
            result = StreetPhotoResult(photos=photos, coverage=score_views(faces, photos))
            front_b = next(f for f in faces if f.building_id == "b" and np.allclose(f.normal, [0, -1]))
            textures = PhotoTextureSet()
            textures.placements[front_b.key] = object()
            chosen = choose_matches(result, textures, limit=4)
            self.assertEqual(chosen[0][1].id, "pb")  # the textured wall first
            self.assertEqual(len(chosen), 1)  # nothing read, nothing else to compare

            def fake_blender(script, args):
                from PIL import Image

                views = json.loads(Path(args[args.index("--views") + 1]).read_text())
                out = Path(args[args.index("--out") + 1])
                for view in views:
                    Image.new("RGB", (view["width"], view["height"]), (90, 90, 90)).save(out / f"{view['name']}.png")

            written = write_photo_matches(result, textures, tmp / "model.blend", tmp,
                                          origin_rd=(0.0, 0.0), z_offset=0.0, blender_command=fake_blender)
            self.assertEqual([w.name for w in written], ["match_1.jpg"])
            self.assertFalse((tmp / "photo_checks" / "_renders").exists())


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
