# Offline 3D map pipeline (3DBAG + AHN + PDOK aerial → Unity)

Takes a bounding box in the Netherlands and bakes a textured 3D model you import
into Unity yourself. Everything runs locally and the result needs no network at
runtime.

```
output/<area_name>/
├── model.fbx        terrain, roads, rails, bridges, tunnels, water, buildings,
│                    trees, furniture, cars, boats
├── aerial.png       aerial ortho of the same area
├── metadata.json    bbox, origin offset, CRS, source versions
├── trees.json       tree positions and heights, for spawning Unity prefabs
├── vehicles.json    car and boat positions and headings, same idea
├── landcover.png    surface class per pixel, registered to the same bbox
├── landcover.json   what the class values mean
├── facade*.png      generated facade textures, referenced by the FBX
├── tree.png         bark and foliage atlas
├── water.png        canal water
├── furniture.png    metal and wood atlas
├── vehicle.png      car paint, hull and timber atlas
├── rail.png         ballast with sleepers, and rail steel
├── structure.png    concrete for decks, piers and tunnel walls
└── ATTRIBUTION.txt  source credits
```

The pipeline stops there. No Unity layer, no viewer, no WebGL build.

## Running it

There is a small web UI, and there is the command line. They do the same thing:
the UI writes a config file and shells out to `pipeline.py`, so it cannot drift
away from the CLI.

### Without a terminal

Colleagues who would rather not open a terminal can double-click a launcher in
this folder:

| | |
| --- | --- |
| Windows | **Start Map Pipeline.bat** |
| macOS | **Start Map Pipeline.command** |
| Linux | **start-map-pipeline.sh** |

It creates a private Python environment beside itself, installs what is
missing, starts the UI and opens the browser. Nothing is typed and nothing is
installed system-wide; deleting the `.venv` folder undoes it completely.

**The first run downloads about a gigabyte and takes several minutes.** 848 MB
of that is Blender, which ships as a Python library here so nobody has to
install Blender separately. Every run after that opens the page in seconds.

**It needs CPython 3.11, and only 3.11.** Blender publishes its Python library
for that version alone. On anything else pip installs the other dependencies
happily and the run then fails minutes later at the Blender stage, with nothing
that points at the version, so the launchers check first and send you to the
download page instead.

On macOS the first double-click is blocked by Gatekeeper because the file came
from the internet: right-click it, choose **Open**, and confirm once.

**The lightest option of all is not to install it on their machines.** The UI is
a web page, so one person can run the launcher and share the address — start it
with `--host 0.0.0.0` and colleagues open `http://<your-machine>:8765`. Only do
that on a network you trust: there is no login, and anyone who can reach the
page can start runs and read the output folder.

### The UI

```bash
pip install -r requirements.txt
python ui/server.py --open
```

Open <http://127.0.0.1:8765>. The panel runs top to bottom in four numbered
steps — **choose an area**, **choose how much detail**, **choose what to
include**, **build** — and the defaults in all four are the ones to use. Search
for a place or click the map, press **Build model**, and watch the log. When it
finishes you get the check results, the preview renders and download links.

The page is written for someone who has never opened a GIS or a 3D package.
Every setting carries a **?** that says what it does in words rather than in
units, each slider prints a plain sentence under it — *"kerbs and small steps
come through"* rather than *"1 m"* — and everything a first-time user should
not have to decide is folded into a collapsed **Fine tuning** section: the
ground triangle budget, the wall texture size, and how buildings on the edge of
the square are treated. A closing **What you get, and how to open it** section
names the four output files and gives the import steps for Blender and Unity.
The technical readouts have not gone anywhere; they sit beside the plain ones,
so the same page serves both readers.

Step 3 lists every dataset a model can be built from, each with a checkbox and
a live status light, so you pick what goes in and see what is answering right
now. Terrain, aerial imagery and buildings are marked required and cannot be
unticked — without them there is no model — and if one of those is down the
Build button is disabled with the reason given. An optional source that is down
is only a warning: untick it and the run proceeds without it.

`src/sources.py` is the single registry behind this. The pipeline reads it to
check the services a run needs before it starts, and the UI reads it to draw the
panel, so the two cannot drift apart. Sources sharing a host — the four BGT
layers do — are probed once rather than four times, so one outage reads as one
failure.

The Build button carries a time estimate that updates as you change settings,
and hovering it breaks the figure down by stage. A progress bar then follows the
run stage by stage — the pipeline already announces every stage as
`Step N/M`, so progress needs no extra protocol between the two.

Treat the estimate as a rough figure. It comes from a cost model calibrated on
measured runs over Utrecht, within about 15% there, but the largest term is the
3DBAG fetch and that varies with how busy their servers are. Every run records
what it actually cost to `work/<area>/timings.json`, and the estimate rescales
itself from those, so it converges on the machine it is running on rather than
the one it was calibrated on. Failed and partial runs are left out of that.

Rough costs for a 1 km² area, previews included: Draft about 3 minutes,
Standard about 7, High about 10, Maximum about 15. Previews alone are over half
of that, so turning them off roughly halves the wait.

The map works directly in RD New, so the square you position *is* the bbox that
gets built — no reprojection anywhere in the page. It is plain HTML with no
JavaScript dependencies, and the server is standard library only.

Tiles come straight from PDOK. If the browser cannot reach it — behind a
corporate proxy, or on a machine where only the pipeline process has network —
the page detects that at startup and routes tiles through the local server
instead.

### The command line

```bash
pip install -r requirements.txt
python pipeline.py --config config.json
```

Add `--preview` to also render top-down, oblique and street-level images of the
result, which is the one check the numbers cannot make for you.

Under the hood the Python stages write intermediates to `work/<area>/`, then
Blender builds the scene:

```bash
blender --background --python blender/process.py -- --work work/<area> --out output/<area>
```

Blender is located in this order: an explicit `--blender`, then `blender` on
`PATH`, then the places the installers actually put it, then the pip `bpy`
module (`pip install bpy`, CPython 3.11). All of them produce the same output.

**On macOS, Blender is never on `PATH`.** It installs as `Blender.app`, and the
executable lives inside the bundle at `Contents/MacOS/Blender`, so a `PATH`
lookup cannot find it however it was installed — "Blender is installed" and
"`which blender` finds it" are simply different statements there. The search
covers `/Applications` and `~/Applications`, including versioned bundle names,
and the same for `C:\Program Files\Blender Foundation` on Windows. If yours is
somewhere else, pass the bundle itself:

```bash
python pipeline.py --config config.json --blender /Applications/Blender.app
```

Useful flags: `--skip-blender` runs the data stages only, `--verbose` turns on
debug logging, `--skip-reimport-check` drops the FBX round-trip check.

```bash
python tests/test_pipeline.py            # includes live checks against PDOK/3DBAG
python tests/test_pipeline.py --offline  # pure logic only
```

**Run the tests against both NumPy majors before trusting a change.** NumPy 2
removed a pile of long-deprecated API, and a pipeline developed on 1.x will
import cleanly and then fail deep inside a run on a machine with 2.x —
`ndarray.ptp()` did exactly that, hours into a large build. Passing on one
major says nothing about the other:

```bash
python -m venv .np2 --system-site-packages
.np2/bin/pip install --upgrade "numpy>=2"
.np2/bin/python -W error::DeprecationWarning -m unittest discover -s tests
```

`-W error::DeprecationWarning` is the part that earns its keep: it catches what
NumPy has scheduled for removal rather than what it has already removed. Two-
dimensional `np.cross` is on that list today.

## Configuration

`config.json` carries the interesting bits; everything else has a default in
`src/config.py`.

```json
{
  "name": "demo_area",
  "bbox": { "crs": "EPSG:28992", "xmin": 136000, "ymin": 455000,
            "xmax": 137000, "ymax": 456000 },
  "aerial":    { "layer": "Luchtfoto Actueel Ortho 8cm RGB", "size_px": 4096 },
  "terrain":   { "ahn_model": "DTM", "resolution_m": 0.5,
                 "mesh_vertices_per_side": 257 },
  "buildings": { "lod": "2.2" }
}
```

A `"crs": "EPSG:4326"` bbox is accepted too, with longitude in `xmin`/`xmax` and
latitude in `ymin`/`ymax`. It is converted to RD New once, in `geo.py`, and
everything downstream stays in RD. The bbox is validated as roughly square and
roughly 1 km per side, and warns rather than fails when it is not.

## Resolution and quality

The UI has a preset plus three sliders. Detail is set as ground resolution
rather than pixel count, so a setting means the same thing whatever the area
size, and the pixel counts are derived from the box.

| Preset | Aerial | Terrain | Facade | Normal map | 1 km² run |
| --- | --- | --- | --- | --- | --- |
| Draft | 50 cm/px | 8 m | 256 px | off | ~1 min |
| Standard | 25 cm/px | 4 m | 512 px | on | ~2.5 min |
| High | 12 cm/px | 2 m | 1024 px | on | ~5 min |
| Maximum | 8 cm/px | 1 m | 2048 px | on | ~8 min |

**Both dials hit a source-data ceiling**, and going past it costs time without
adding information. The aerial layer is an 8 cm ortho, so a 1 km area is fully
resolved at about 12500 px. The AHN DTM is 0.5 m, so the same area is fully
resolved at about 2001 vertices per side. The pipeline warns instead of
refusing; the UI shows the derived numbers and flags when a setting has gone
past what the data holds.

Three things are worth knowing before you reach for the top of the sliders.

**The aerial jump from 25 to 12 cm/px is large; from 12 cm to native 8 cm it is
small.** At 25 cm/px cars are blobs. At 12 cm/px you can see individual cars,
kerb lines and bare branches. Native 8 cm adds little over that for double the
file (92 MB against 174 MB for 1 km²). 12 cm/px is the sweet spot.

**Terrain has a sweet spot, not a "more is better" curve.** The DTM has a hole
wherever a building stands — 49% of source pixels over Utrecht centre. At 4 m
spacing each output vertex averages many real measurements and 33% end up
interpolated. At native 0.5 m spacing that rises to 86%: you get sixteen times
the vertices, and most of them carry reconstructed ground rather than measured
ground. Around 1-2 m is where the extra vertices still buy real detail.

**Grid spacing and mesh detail are separate dials.** Spacing decides how finely
the ground is *measured*; `terrain.simplify_tolerance_m` decides how many
vertices are *spent* describing what was measured. Finer spacing with the
default tolerance costs far less than the vertex count suggests, because the
extra rows are only kept where the ground actually moves — see
[Terrain mesh](#terrain-mesh).

**Unity has its own limits.** Textures over 16384 px cannot be imported at full
size, which a 2 km area at native resolution would exceed. Imported textures
are also capped at 2048 by default, so raise *Max Size* on `aerial.png` or none
of this is visible. A terrain over 65k vertices needs a 32-bit index buffer,
which Unity sets automatically.

**The services cap a single response, so large areas are fetched in pieces.**
Both caps are handled automatically and neither limits how big an area you can
build; they only decide how many requests it takes.

| Service | Cap per request | What that is on the ground |
| --- | --- | --- |
| Aerial WMS | 2500 px | requested in ≤2000 px tiles |
| AHN WCS | 4000 px | 2000 m at the 0.5 m AHN |

The AHN limit is not in its capabilities document — it can only be learned by
being refused — so it is a constant here, verified as inclusive: 4000 px is
answered and 4001 is not. A 3 km area needs 6000 px and arrives as 2×2 tiles.
Tiles are split in pixel space rather than in metres, which is what makes them
join exactly: the service honours requested bounds to the millimetre and
returns `span / resolution` pixels, so a boundary at an arbitrary coordinate
would land mid-pixel and each side would round it differently. Measured across
the join on a 3 km area, the height difference between neighbouring cells is
*smaller* than between ordinary neighbouring cells nearby, so there is no seam.

The facade normal map is the one quality lever that is not about resolution: it
gives window reveals, sills and storey bands real relief under a moving light.
It costs one small extra texture and survives FBX as the material's bump slot.
It is written as `*_normal.png` because Unity keys off that suffix to set the
texture type automatically.

Other knobs worth knowing:

| Key | Default | What it does |
| --- | --- | --- |
| `facade.variants` | `1` | How many era styles to assign by construction year, up to `5`. The monumental and industrial styles are always present on top of these. |
| `facade.ground_floor` | `true` | Split walls at the first-floor line and give the ground storey its own material. Churches and sheds are left whole. |
| `facade.ground_floor_height_m` | `3.6` | Where that cut sits above each building's own ground level. |
| `facade.photo_textures` | `true` | Photographed CC0 masonry under the generated windows, downloaded once and cached in `work/_textures/`. |
| `facade.photo_tint` | `0.6` | How far each photograph is pulled towards its era colour. `0` keeps it as shot, `1` lands it exactly on the palette. |
| `facade.monumental_tile_m` | `7.0` | Bay width for churches, towers and civic halls. |
| `facade.monumental_bay_m` | `9.0` | Bay height for the same — a church bay, not a storey. |
| `facade.industrial_tile_m` | `9.0` | Bay width for sheds and depots. |
| `facade.industrial_bay_m` | `6.0` | Bay height for the same. |
| `trees.enabled` | `true` | Fetch BGT trees and give them AHN heights. |
| `trees.geometry` | `true` | Also bake low-poly tree meshes into the FBX. `trees.json` is written either way. |
| `trees.crown_search_m` | `3.0` | Radius the canopy height is taken as a maximum over. |
| `surfaces.water` | `true` | Replace the interpolated canal bulge with real water surfaces. |
| `surfaces.road_geometry` | `true` | Build roads as their own objects, one per class, instead of only as raster class. |
| `surfaces.road_lift_m` | `0.06` | How far the road surface floats above the terrain, so the two do not fight for depth. |
| `surfaces.road_drape_tolerance_m` | `0.08` | How far a flat road triangle may miss the ground before it is split. |
| `surfaces.land_cover` | `true` | Classify the ground and export the class map. |
| `surfaces.detail_strength` | `0.22` | Per-class grain mixed into the aerial. 0 disables it. |
| `surfaces.water_depth_m` | `1.2` | How far each bed is sunk below its own water level. |
| `furniture.enabled` | `true` | Lampposts, bollards, sign posts and benches. |
| `vehicles.cars` | `true` | Lay cars out in the BGT parking bays. |
| `vehicles.boats` | `true` | Moor boats between the BGT mooring posts. Needs `surfaces.water`. |
| `vehicles.car_occupancy` | `0.72` | How full the bays are. 1 parks a car in every space. |
| `vehicles.boat_occupancy` | `0.8` | The same for moorings. |
| `rails.enabled` | `true` | Railway, tram and metro track from BGT `spoor`. |
| `rails.step_m` | `4.0` | How far apart points along a track may get before the ground under it stops being followed. |
| `structures.bridges` | `true` | Bridge decks and piers from BGT `overbruggingsdeel`, heights read off the AHN surface model. |
| `structures.tunnels` | `true` | Tunnels from BGT `tunneldeel`, on a constructed depth profile. |
| `structures.tunnel_depth_m` | `18.0` | How deep the drawn profile goes. Nothing measures this; see below. |
| `structures.tunnel_ramp_m` | `350.0` | How long the descent from each portal is. |
| `structures.tunnel_walls` | `true` | Side walls up to ground level, so a tunnel reads as a cutting. No ceiling: a roofed tunnel is invisible. |
| `usage.enabled` | `true` | BAG building function, deciding the ground storey. |
| `facade.floor_height_m` | `3.0` | Nominal storey height for the window grid. |
| `facade.texture_px` | `512` | Pixels per storey tile. 512 over a 4 m tile is 128 px/m. |
| `facade.normal_map` | `true` | Write a normal map beside each facade texture. |
| `facade.relief_depth` | `0.035` | How far window reveals and storey bands stand out. Small on purpose: a facade is nearly flat. |
| `buildings.clip_mode` | `centroid` | `centroid` keeps buildings whose centre is inside the bbox. `intersect` keeps every building the API returns. |
| `buildings.sources` | `["api", "tiles"]` | Which of 3DBAG's two services to try, in order. `["tiles"]` skips the flaky API entirely. See [When 3DBAG is down](#when-3dbag-is-down). |
| `buildings.tiles_version` | `v20250903` | Which dated 3DBAG release the static tiles come from. Bumped by hand; releases are listed at 3dbag.nl/en/download. |
| `buildings.probe_timeout_s` | `8.0` | How long to knock on the API before giving up and using the tiles. |
| `buildings.merge` | `single` | One merged buildings mesh. `per_building` gives one object each. |
| `terrain.mesh_vertices_per_side` | `257` | 257 → 66k terrain vertices before simplification. Rounded up to 2^k + 1 when simplification is on. |
| `terrain.breaklines` | roads, water, land cover, buildings | Which outlines the terrain folds along. Empty falls back to the bisection mesh. |
| `terrain.breakline_simplify_m` | `0.15` | How far a simplified outline may stray from the surveyed one. Applied to the whole outline network at once, not ring by ring, so a boundary two polygons share stays one line. |
| `terrain.contour_interval_m` | `0.5` | Contour spacing. Contours are then thinned against `simplify_tolerance_m`, so flat ground contributes none. `0` leaves them out entirely. |
| `terrain.simplify_tolerance_m` | `0.10` | How far the terrain mesh may stray from the height grid. Drops the vertices sitting on ground their neighbours already describe, which over a Dutch bbox is most of them. `0` keeps the full grid. |
| `aerial.max_request_px` | `2000` | Tile size for the WMS mosaic; the service caps requests at 2500. |

## How it fits together

```
src/geo.py         bbox parsing, WGS84 to RD, origin offset
src/elevation.py   AHN WCS -> GeoTIFF -> height grid
src/buildings.py   3DBAG API -> CityJSON -> semantic mesh data
src/imagery.py     PDOK WMS/WMTS -> aerial.png + georeference
src/facade.py      generated facade, tree, water and furniture textures
src/facade_uv.py   fitting whole window bays across each wall face
src/textures.py    photographed CC0 wall surfaces, fetched once and cached
src/sources.py     the registry of datasets a model can be built from
src/bgt.py         shared BGT client: paging, version filter, rasterising
src/trees.py       BGT tree points, heights from AHN DSM minus DTM
src/surfaces.py    BGT water bodies and land cover
src/furniture.py   BGT lampposts, bollards, signs and benches
src/vehicles.py    cars in the parking bays, boats between the mooring posts
src/rails.py       railway, tram and metro track from BGT spoor
src/structures.py  bridge decks and piers, and tunnels below ground
src/usage.py       BAG building function, joined to 3DBAG on the building id
src/export.py      metadata.json and the Blender scene description
src/validate.py    the headless checks
blender/process.py builds the scene, assigns materials, exports FBX
blender/preview.py renders preview images of an exported FBX
ui/server.py       local web UI, standard library only
ui/index.html      the page: RD map picker, settings, live log, results
```

The heavy CityJSON and raster work all happens on the Python side.
`blender/process.py` only reads `scene.json`, the npz intermediates and the
textures, so nothing depends on a GUI add-on being present in headless mode.

**The Blender scripts may only import what Blender itself bundles** — `bpy`,
`numpy`, `mathutils` and the standard library. When Blender is a real
application rather than the pip module it runs its own Python, where none of
this pipeline's dependencies exist. Anything else, triangulation included,
belongs in `src/` with its result passed through the intermediate files. A test
enforces this statically, because running the pipeline here cannot catch it: the
pip-`bpy` path shares the caller's interpreter, so every import resolves.

## Coordinates

Everything is RD New (EPSG:28992) horizontally with NAP heights. 3DBAG publishes
EPSG:7415, which is the same thing horizontally, so no layer is ever
reprojected.

The origin sits at the centre of the bbox, so geometry comes out in metres
around (0, 0, 0) and float precision stays comfortable in Unity.

Blender is built with X = RD easting, Y = RD northing, Z = NAP height. The FBX
export uses forward `-Z`, up `Y`, which lands in Unity as:

```
RD easting   -> Unity X
NAP height   -> Unity Y
RD northing  -> Unity Z
```

Blender is right-handed Z-up and Unity is left-handed Y-up. The handedness
change cancels against the axis change, so the model is not mirrored: in a Unity
top-down view east is still right and north is still up. Verify it in Unity
anyway — `metadata.json` records `origin_rd` and `ground_z_offset_nap` so you can
walk any local position back to RD and NAP:

```
rd_x = local_x + origin_rd[0]
rd_y = local_z + origin_rd[1]
nap  = local_y + ground_z_offset_nap
```

3DBAG heights and the AHN DTM are both NAP, so buildings land on the terrain
without any fitting. Both get the same `ground_z_offset_nap` subtracted. Much of
the Netherlands sits below NAP, so negative heights are normal and are never
clamped away.

## When 3DBAG is down

`api.3dbag.nl` is the least reliable service the pipeline depends on, and
buildings are not optional, so an outage there used to stop a run dead. It no
longer does: 3DBAG publishes the same LoD2.2 data twice, on two separate hosts.

| | Service | What it is |
| --- | --- | --- |
| `api` | `api.3dbag.nl` | OGC API Features, paged, returns exactly the bbox |
| `tiles` | `data.3dbag.nl` | static gzipped CityJSON tiles plus a WFS tile index |

`buildings.sources` is the order to try them in, default `["api", "tiles"]`.
Whatever the first one raises is logged and the next one gets a turn.

**The API is knocked on before it is trusted.** With `timeout_s` at 180 and four
retries, a dead API costs twelve minutes before the fallback would get a turn —
long enough that people kill the run instead of letting it recover itself. An
8-second probe decides first. The preflight does the same check even earlier and
takes the API out of the chain for the whole run when it does not answer, so the
stage does not rediscover it.

**A source with a backup is not down until both are.** The preflight and the
UI's source panel both treat 3DBAG as available when either host answers; the
panel shows an amber dot and "on the backup service" so it is visible without
being alarming. Only both being unreachable blocks a run.

**Tiles are cached above the area directory**, in `work/_bag3d/<version>/`,
so every area anyone builds shares them. That is the part that helps a team
most: the first Rotterdam build downloads about 10 MB of tiles, and every
neighbouring area after that needs no network for buildings at all.

**A tile covers far more than your bbox.** 3DBAG tiles are a quadtree split by
building density — a 600 m bbox pulls in 6 tiles holding 5,683 buildings to keep
1,229. Every `Building` carries a `geographicalExtent`, and skipping the ones
that cannot reach the bbox before touching any geometry takes the parse from
70 s to 20 s.

Measured over the Utrecht Dom bbox during a real API outage: 6 tiles, 10.3 MB,
**1,229 buildings — the same count the API returns**, in 35 s cold and 20 s with
the cache warm.

**The tiles are a dated release.** 3DBAG keeps every release and publishes no
`latest` alias, and the WFS does not advertise which one it indexes, so
`buildings.tiles_version` is pinned and bumped by hand. If the index and the
release drift apart every tile 404s, and the run says so and names the setting.
`metadata.json` records which service actually answered, and its version, so a
model built during an outage can be told apart from one built the week before.

To skip the API entirely — worth it if your team hits outages often, since the
cache makes repeat runs faster than the API anyway:

```json
"buildings": { "sources": ["tiles"] }
```

## Terrain mesh

The terrain folds along the features the ground actually has: a canal bank, a
kerb, a building footprint and a contour are all real edge loops you can select
and drag in Blender, not a patch of triangles that happens to be dense there.

**Why the grid could not do this.** A grid puts its edges on grid lines and the
bisection mesh below puts them on grid diagonals. A bank runs along neither. No
amount of extra detail fixes that -- it only makes a finer staircase.

**Breaklines.** `src/breaklines.py` collects the lines the mesh has to fold
along and `src/cdt.py` triangulates so that every one of them survives as an
edge. Four sources: BGT roads, land cover and water; building footprints;
contours off the DTM; and the bbox edge.

**The BGT is a planar partition, and simplification has to keep it one.** A road
and the pavement beside it are two polygons carrying the *same* boundary, vertex
for vertex. Simplifying them one ring at a time does not preserve that:
Douglas-Peucker is anchored on the ends of whatever it is given, and where three
polygons meet part way along a kerb, that junction is a ring corner for one of
them and an ordinary point on a smooth curve for another. The anchors differ,
the two copies of the kerb keep different vertices, and one line becomes two
that cross each other repeatedly.

Everything downstream inherits it. The crossings get cut into a ladder of
millimetre segments, the ladder meets at angles no triangulation can make a
decent triangle out of, and refinement chases those corners into ever smaller
slivers. Measured: **268 of the 283 worst triangles in a test area came from one
kerb that had been turned into two.** So `partition_chains` welds the outlines
into a single network first and simplifies the runs *between* junctions. Each
shared kerb then exists once and is thinned once.

**Contours are joined before they are thinned.** Marching squares emits one
chord per grid cell, and emitting those raw made contours **99.8% of every
breakline in an area** — a vertex every 1.5 m on a 2 m grid, along ground that
mostly needed none. That put the mesh a hundred times denser along a contour
than on the field beside it, and a triangulation asked to bridge a density step
like that can only do it with long thin wedges, whatever triangulator you use.

`contour_polylines` chains the chords back into lines and then thins them
against the **height** they cost rather than the distance: a contour is a line of
constant height, so moving it sideways by `d` where the ground slopes at `g`
misplaces the height by `d * g`. That prices every vertex in the units the
tolerance is written in, and does the right thing at both ends by itself — a
canal bank keeps its detail, and a contour out on a flat field, where the only
slope is the scanner's own noise, is allowed to wander a hundred metres and so
collapses and drops out. Roughly half of the raw contours over a Dutch area were
tracing that noise; 37% of them were out on flat ground with no feature to fold
along at all.

**Nothing may cross.** No triangulation can honour two constraints that cross,
so every segment is cut at every intersection first. Contours cross roads and
water constantly. Splitting has to be repeated, because welding afterwards is
what merges the two copies of a shared intersection and also drags a cut that
landed a millimetre from an endpoint back onto it, undoing the split.

**Then it refines, and this is where the mesh is actually made.** The bisection
mesh below chooses which grid points are worth keeping, but **its tolerance is a
property of its own triangles, not of its points**: it keeps a vertex because of
the right triangles *it* would have drawn, and Delaunay draws different ones.
Measured, the same points re-triangulated were three times outside the tolerance
they were chosen for, and with the breaklines added, fourteen times.

So the point set is earned back with Delaunay refinement, after Ruppert:

* **Split a constraint segment** when a vertex encroaches on it — sits inside
  its diametral circle — or when the ground under its chord has sagged more than
  the tolerance away from it. Without this a simplified breakline stays a single
  forced edge with the mesh draped off it; a 339 m one showed up in a mesh whose
  median edge was 2 m.
* **Insert the circumcentre** of any triangle that is further than the tolerance
  from the ground. The circumcentre, not the centroid: the centroid sits among
  the corners it came from, so inserting it splits a bad triangle into three of
  the same shape. The loop that did that added 132 points, then 35, then 33,
  then 38, and stopped no closer than it started.
* When a circumcentre falls **outside the area**, split the constrained edge
  blocking it instead. This is the boundary case, and without it the mesh never
  improves where a canal runs off the edge of the bbox — which is exactly where
  it was worst.

Ground points are also kept **clear of breaklines** by one grid step. A grid
node crowding a line carries no height that was ever measured, and the line's
own vertices are metres apart, so the triangle between them can only be a wedge.
Forcing the bisection mesh *finer* near lines is the intuitive version of this
idea and it is simply wrong — it cost 50% more triangles for the same shape,
because refinement grades the mesh properly by itself and a lattice pushed up
against an arbitrary polyline can only make wedges.

**A tolerance is a promise about a surface, and AHN is a grid with steps in it.**
A quay wall, the lip of a filled building hole, the scar where a tree was taken
out: 60 cm between neighbouring samples is ordinary. Where two samples differ by
a step, no triangle that is not aligned to the sample grid gets closer than about
half of it, however finely it is cut — a flat face cannot follow a kink that
falls between cells, and a Delaunay mesh over scattered points cannot promise an
edge exactly on a cell boundary.

So refinement and the checks share one notion of *how close the mesh can be
asked to get*: the tolerance, or half the local step, whichever is larger
(`height_allowance`). This is not a relaxation, it is the difference between a
target and a wish. Asking for the wish did real damage — refinement could never
satisfy it, so it kept inserting points into triangles it had no way to improve,
and on a real Utrecht kilometre that ran to **701,978 triangles while still
missing the tolerance by a factor of five**. Refinement also now stops at any
triangle already smaller than one grid cell, because everything inside a cell is
the interpolation's opinion rather than a measurement.

The allowance is read across a **whole triangle**, not at the point where its
own error happens to peak: a triangle's ability to follow the ground is limited
by the roughest ground it covers, and reading it at one point flagged 2515
triangles on a 2 km area for a step that was inside them.

Two more things are reported honestly rather than blamed on the mesh:

* **Where a point should go.** A triangle off the ground now takes the *grid
  node* nearest the worst of it, not a circumcentre. A circumcentre is almost
  never a grid node, and a one-cell feature is only reproduced by a vertex
  standing on it. Circumcentres are still what fixes *shape*.
* **Whose sliver it is.** Two surveyed outlines meeting at half a degree put a
  half-degree triangle in the mesh, and there is nowhere to put a point that
  improves it. A sliver with every corner on a constraint is filling a gap the
  input already had; one with a corner refinement placed itself is ours. Only
  the second count fails a check.

Measured over one synthetic Dutch area, before and after all of the above:

| | before | after |
| --- | --- | --- |
| Triangles | 25,267 | **12,884** |
| Smallest angle anywhere | 0.0025° | **1.22°** |
| Triangles under 1° | 628 | **0** |
| Triangles under 5° | 9.9% | **0.5%** |
| Triangles under 10° | 17.6% | **1.9%** |
| Median smallest angle | 29.7° | **40.4°** |
| Worst height error (0.10 m asked) | 0.459 m | **0.100 m** |
| Worst sag under a breakline | never measured | **0.100 m** |

### Quads, and controlling how busy the ground is

Everything above makes the ground *correct*. It does not make it pleasant to
edit, and those are different problems. A triangulation is what the geometry
wants; quads are what Blender wants, because selecting, looping and subdividing
all work along quads and none of them work on a triangle fan.

So `src/quadmesh.py` runs last and only decides which pairs of triangles to
fuse. Nothing moves, nothing is added, nothing is dropped — so every accuracy
figure measured on the triangles still holds afterwards, and the triangles stay
the source of truth that the checks read. Two edges are never dissolved:

* **A breakline.** Fusing across one would delete the edge the mesh exists to
  have. This is the difference from an automatic quad remesher: those align
  edges to the *curvature* of the surface they are handed, and a kerb is not a
  curvature feature — it is a line from another dataset that happens to lie on
  this surface. Often it has no dihedral angle at all, because both sides take
  their height from the same grid cell. Told which edges those are, the pairing
  keeps every one.
* **A fold.** Two triangles meeting at an angle are a ridge or a ditch, and
  fusing them would smooth it away. `quad_max_fold_deg` is where that line sits.

Greedy matching on the dual graph, best-shaped quads first. Optimal matching is
a blossom algorithm and buys nothing visible; greedy reaches 60–90% quads.

The breakline edges are also **marked sharp and as UV seams on export**, so in
Blender *Select Sharp Edges* hands you the kerb, the canal bank or a building
footprint as a loop instead of leaving you to hunt for it face by face.

Five settings control how busy the ground is, all of them in the UI under
**Terrain shape**, and none of them affect accuracy:

| Setting | Config key | What it does |
| --- | --- | --- |
| Make quads | `terrain.quads` | Fuse flat triangle pairs. |
| Keep folds sharper than | `terrain.quad_max_fold_deg` | 3° keeps every crease; 60° fuses almost everything. |
| Straighten the features | `terrain.breakline_simplify_m` | 0.15 m is as surveyed; 5 m leaves only the layout. The biggest single lever on visual busyness. |
| Ignore features shorter than | `terrain.min_feature_length_m` | Drops traffic islands, kerb stubs, driveway aprons. A planar partition has thousands, and each one is an edge loop. |
| Contour lines | `terrain.contour_interval_m` | The only edges that follow the ground itself. Also the costliest. Flat ground contributes none regardless. |
| Which features become edges | `terrain.breaklines` | Roads, water, land cover, buildings — independently. |
| Even face size | `terrain.max_face_m` | Caps how big a face may be regardless of whether the ground needs it. |

### What the terrain arrives knowing

Two things ship with the mesh that make it editable rather than merely correct,
and both are information the pipeline already had and used to throw away.

**Every face is labelled with what it is.** The surface class of every square
metre is worked out for `landcover.png`; it is now also sampled per terrain face
into an integer attribute `surface_class`, and turned into vertex groups —
`ground_road_asphalt`, `ground_water`, `ground_green`, one per class present. So
"flatten the ground for this building" is select-group-then-flatten rather than a
lasso and a prayer. The vertex groups matter more than the attribute in practice,
because they also drive proportional editing, which is what levelling a plot
actually uses.

**Every breakline is marked sharp and as a UV seam.** *Select Sharp Edges* hands
you the kerb, the canal bank or a footprint as a loop. A footprint is a closed
ring, so selecting it and then *Select → Inner Region* gives you the plot.

**A note on field-aligned quadrangulation**, since it is the obvious next thing
to reach for and it is the wrong tool here. A field-aligned quad mesh is regular
only *between* its singularities, and loop select, grid select and proportional
editing all terminate at one. The field must align to the constraints, a
cross-field carries one direction pair per point, and every junction where two
streets meet at anything but 90 degrees forces a singularity. A real 1 km area
carries around 10,400 BGT and building features, so on the order of 7,000
junctions — against roughly 40,000 quads at a 5 m target. That is a singularity
every handful of faces: not a regular sheet, and harder to work with than this.
It looks clean in published examples because those are smooth organic surfaces
or open countryside, where the field has almost nothing to satisfy.

**Roads still float above it.** They are separate objects with their own
materials, and they share the terrain's edges rather than being part of it, so
the 6 cm lift is raised just enough to clear how far the mesh *rises above* the
grid the roads were draped on — a one-sided 99.9th percentile, not the worst
error in either direction. Both halves of that matter: a mesh dipping below the
grid buries nothing and needs no clearance, and the worst single triangle is not
a statistic to apply to every road in a model. Taking the maximum pinned the
lift to its 0.2 m cap over 4 km² and left every carriageway floating 22 cm.

This is a mitigation, not a cure. The real fix is to drape roads, rails and
water on the terrain mesh itself rather than on the grid, so the two agree by
construction and the lift can go back to 6 cm.

## Adaptive terrain mesh (no breaklines)

A regular grid charges the same price everywhere, which over a Dutch bbox is a
bad trade: a car park that three vertices would describe perfectly costs exactly
as much as the canal bank next to it. `terrain.simplify_tolerance_m` replaces
the grid with a triangulation that keeps vertices only where the ground moves.

**How it decides.** Two triangles cover the bbox. Each is cut at the midpoint of
its hypotenuse — always another grid point — and a cut only happens where the
ground under that triangle strays further from it than the tolerance allows.
The error driving the decision is *nested*: a triangle's error is the worst of
its own midpoint and everything beneath it. Without that a triangle passes its
own midpoint test while hiding a dike between two of its corners.

**Why this shape and not a greedy TIN.** Two properties matter more than raw
vertex efficiency:

- **No cracks.** The error is stored per *vertex*, not per triangle, and taken
  as the maximum over both triangles sharing a hypotenuse. Neighbours therefore
  always agree about whether to cut it, so the mesh is watertight by
  construction — no T-junctions to seam over, no skirts to hide them.
- **No slivers.** Every triangle is a right isoceles triangle on a grid
  diagonal, so they all have the same three angles. Greedy point insertion gets
  fewer triangles for the same error and hands you long thin wedges for it,
  which normals and lightmaps do not enjoy.

Measured over 600 m of central Utrecht, against the 131,072 triangles of the
full 257² grid:

| Tolerance | Triangles | Share of the grid | Worst height given up |
| --- | --- | --- | --- |
| 0.02 m | 54,750 | 41.8% | 0.03 m |
| 0.05 m | 30,700 | 23.4% | 0.08 m |
| **0.10 m** (default) | **18,584** | **14.2%** | **0.17 m** |
| 0.25 m | 8,922 | 6.8% | 0.44 m |
| 0.50 m | 4,467 | 3.4% | 0.87 m |

The default is 0.10 m because that is roughly AHN's own vertical accuracy — 5 cm
systematic plus 5 cm stochastic — so it gives up nothing the source could
resolve in the first place. Set it to `0` for the old full grid.

**The worst error runs to about 1.7× the tolerance, not 1.0×.** A vertex can sit
on a hypotenuse whose own ends were moved, so the deviations compose. The bound
is a budget, not a guarantee; `terrain_mesh_within_tolerance` allows 3× before
calling it a defect.

**Simplification happens before anything drapes on the ground.** Roads, rails,
trees, buildings and water levels all read heights through the same sampler, and
that sampler is switched to the simplified surface inside `build_terrain`. Doing
it later — next to the mesh Blender actually draws — would leave every one of
them sitting on a surface that was thrown away, and roads only float 6 cm.

**The grid does not go away.** It is still written, still classified for land
cover, and still carries the water bed. Mesh vertices are addressed by grid
column and row, which is what lets the bed be read straight off a vertex.

**The grid side has to be 2^k + 1.** Halving is the only way the hierarchy
subdivides, so every hypotenuse midpoint has to land on a grid point. Sizes that
do not fit are rounded *up* — the extra rows are source detail for the
simplifier to choose from, and it discards whatever it does not need.

## Ground surfaces

**Water is the reason this exists.** Lidar does not reflect off water, so the AHN
DTM is 73% empty over a canal against 47% on land, and the returns that do come
back scatter over six metres. The gap filler then interpolates inward from the
banks, which turns every canal into a bulge with lumps in it — the worst
geometric artifact in the model. The BGT has the outlines, so the pipeline stops
guessing: it takes a level per body from a low percentile of whatever lidar did
return inside it, sinks the bed below that, and lays a flat surface on top.

The level is per body, not shared. Levels across one square kilometre of Utrecht
run from −0.57 to 1.00 m NAP, so a single bed taken from the median would sit
above the surface of the lowest canal and poke straight through it. A check
asserts every bed clears its own surface.

Water polygons are clipped to the bbox, unlike buildings. A canal runs a long
way past the area and one Utrecht outline stretched the model 300 m beyond its
terrain. Cutting a building open would show its inside; a water surface is flat,
so trimming it costs nothing.

**Land cover** comes from three more BGT collections: `wegdeel`, plus vegetated
and unvegetated terrain, with `ondersteunendwaterdeel` for quays. Over the demo
area that classifies about 46% of the ground into eleven classes. It leaves in
two forms:

- `landcover.png` and `landcover.json` — a class map registered to the same
  bbox as `aerial.png`, for driving materials or walkable/drivable logic
  in Unity.
- A light per-class grain mixed into the aerial itself. An ortho is flown at
  8 cm and delivered as JPEG, so close up it is mushy no matter what resolution
  it is resampled to — there is simply no detail left to resolve. Adding grain
  matched to what each surface actually is puts high-frequency texture back
  where the photo has none. `surfaces.detail_strength` controls it; 0 turns it
  off.

**Roads are their own layer.** `surfaces.road_geometry` builds the road surface
as geometry rather than leaving it as pixels in the terrain's photograph — one
object per class, so the FBX arrives with `Roads_asphalt`, `Roads_brick`,
`Roads_cycle_path`, `Roads_footpath`, `Roads_parking` and `Roads_transit_lane`
as separate GameObjects, each carrying its own material. That is what makes them
separately addressable: a carriageway can go on a drivable layer and a footpath
on a walkable one without splitting anything by hand.

Nothing about the look changes by default. Each road material is the aerial
photo under the same top-down projection the terrain uses, so the surface is
pixel-identical to what it replaced — including road markings and crossings,
which a tiling texture would lose. The point of the split is that the material
is now yours to replace.

Two details make it work:

- **The road surface is draped, not flat.** Earcut turns a road strip into long
  slivers — a quarter of the edges over Utrecht are longer than 10 m and the
  longest is 163 m — and sampling the ground only at their corners left one
  cutting through a canal bank by 1.86 m. Triangles are now split until a flat
  one no longer misses the ground beneath it, testing the error at the centroid
  where a plane through the corners is exactly their mean. It is adaptive, not
  uniform: flat streets stay coarse and only slopes get subdivided, which costs
  9% more geometry rather than several times as much, and takes 0.2 s. The worst
  error over the demo area drops from 1.86 m to the 8 cm tolerance. A check
  asserts it.
- **It floats 6 cm above the terrain**, so the two do not fight for the same
  depth. Splitting one triangle and not its neighbour leaves a hanging node and
  so a crack no wider than the tolerance, which is harmless here and only here:
  the terrain sits directly underneath wearing the same photograph, so a crack
  shows the ground rather than a hole.

Tunnels (`relatieve_hoogteligging` below zero) are left out — a tunnel drawn on
the surface is simply wrong. Bridges are kept, because the DTM under a canal is
interpolated up to bank level, which is about where a low Dutch bridge sits.

**Roads are not one class.** The BGT knows what every road surface is for and
what it is made of, and a Dutch street is unrecognisable without both: the
carriageway is brick as often as asphalt, the cycle path beside it is red, and
the footpath is grey tiles. `wegdeel.functie` and `plus_fysiek_voorkomen`
separate carriageway, brick street, cycle path, footpath, parking bay and tram
lane, each with its own tint and grain. Over Utrecht centre brick carriageway
(6.8%) actually outnumbers asphalt (5.5%), and footpath is the largest class at
17% because the old centre is mostly pedestrianised. Road parts overlap at
junctions and kerbs, so they are painted in an explicit order — broad surfaces
first, the things cut out of them last — rather than in whatever order the API
returned them.

## Street furniture

About 2000 poles and 200 pieces of furniture per square kilometre: lampposts,
bollards, sign posts and benches, from BGT `paal` and `straatmeubilair`. None of
it is structurally important, which is the point — a street with nothing on it
reads as a model. They are simple boxes sharing one material, so a couple of
thousand objects cost one draw call.

## Bridges and tunnels

The parts of the ground that are not the ground. Until now both were dropped:
tunnels skipped outright, bridges draped onto the surface, which sank the
Erasmusbrug into the Maas. They need opposite treatment, because the data for
them is not symmetrical.

**A bridge can be measured.** BGT `overbruggingsdeel` comes split into `dek` and
`pijler` — the deck and the piers holding it up, as separate polygons — and a
deck is a hard surface, so the AHN *surface* model sees it where the terrain
model, which is bare ground by definition, does not. Over a square kilometre of
the Maas the DSM returns a reading on all seventeen decks. The height is a
median over each deck rather than a point sample, because the DSM also caught
the railings, the gantries and whatever was driving across when it was flown.
Piers are extruded from the deck down to whatever they stand on.

**The road on a bridge rides its deck.** Draping it on the terrain left the deck
at its real height with its own carriageway lying on the water. Elevated road
parts (`relatieve_hoogteligging` above zero) now take one robust height per BGT
part — not a surface to chase, because refining a road mesh against a raster of
whatever the lidar hit drove it from 35k triangles to 510k and put one
carriageway 95 m up. Where no plausible deck reading exists, the part is draped
on the ground like any other road and the run says how many.

**A tunnel cannot be measured.** Nothing looks down and sees the Maastunnel. The
BGT gives its footprint and the carriageway inside it, and an ordinal level of
-1, but no depth in metres exists in any open dataset. So the profile here is
**constructed**: portals at ground level, ramping down over `tunnel_ramp_m` to
`tunnel_depth_m`, along the tunnel's own long axis — which is the only part of
it that comes from the data. Over the Maastunnel that puts the road at −22 m NAP
mid-river against a real figure of about −20. It is the one piece of geometry in
this pipeline that is drawn rather than measured, and `metadata.json` says so.

Tunnels get side walls but no ceiling, deliberately: a roofed tunnel is
invisible in the model it was just added to.

## Railways

BGT `spoor`, and the first linear dataset here. Everything else arrives as
polygons to fill or points to stand something on; a track arrives as a
centreline, so the geometry is built out sideways from a line rather than
filled in from an outline. What comes back is one line per *running track*, not
per route, so a station throat resolves into the individual tracks through the
points: 265 of them and 17.7 km within 800 m of Utrecht Centraal.

`functie` separates heavy rail from tram and light rail, and that matters
because they are built differently. A railway sits on a raised ballast bed; a
city tram is set flush into the street. Drawing a gravel bed down a shopping
street would be worse than drawing nothing, so a tram gets rails and no bed, and
the road surface underneath shows between them. Each kind is its own object —
`Rails_train`, `Rails_tram`, `Rails_metro` — for the same reason the roads are.

Three details are worth knowing:

- **Sleepers are in the texture, not the geometry.** There are 38 km of track
  within a kilometre of Utrecht Centraal, which at one sleeper per 600 mm is
  63,000 of them. As bands in a tile repeating along the track they cost
  nothing. The UVs run in metres along the centreline, so the spacing holds
  through curves.
- **The ribbon is mitred.** Using each segment's own normal would leave a notch
  on the outside of every bend; averaging the two adjacent normals and widening
  by `1/cos` keeps the edge continuous. The widening is capped, so a hairpin
  cannot send a corner off to infinity.
- **Elevated track takes its height from the AHN DSM.** A fifth of the track
  around Utrecht Centraal is up on a viaduct, and drawing that at ground level
  would put a railway through the street. A deck is a hard surface, so the
  surface model sees it where the DTM — bare ground by definition — does not,
  which makes the height a measurement rather than a guess. The reading is only
  trusted where it is plausibly a deck: high enough above the ground to be one,
  low enough not to be a gantry. The DSM arrives with the trees stage; without
  it, elevated track is drawn at ground level and the run says so.

Tracks are clipped to the bbox. A railway does not stop at the edge of the area
and the BGT returns any track that touches it in full, which stretched an 800 m
model to 1592 m across before the clip went in.

## Cars and boats

Neither is a dataset. Nobody publishes where cars are parked or boats are
moored. But the BGT publishes the two structures that exist *because* of them,
and those turn out to be enough — which is why the result lands on the real
kerbs and the real canals instead of being scattered plausibly.

**Parking bays.** `wegdeel` carries `functie = parkeervlak`, one polygon per run
of bays. Over Utrecht centre the median bay is 11 m long and 2.6 m across, which
is a parallel bay holding two cars. The polygon's own shape decides how the cars
in it are oriented, because it has to: a strip 2.6 m across can only hold cars
end to end, one 5 m across can only hold them side by side, and anything wider
is a car park and gets rows. Cars are placed on the bay's long axis at a 5.6 m
pitch, tested against the polygon rather than its bounding box — bays bend round
corners and step around trees, and a car placed on the box ends up in the road.
Not every bay is full, so `vehicles.car_occupancy` leaves gaps.

**Mooring posts.** `waterinrichtingselement_punt` carries `plus_type =
meerpaal`. The median gap between neighbouring posts over the same area is
6.7 m, which is a boat, because that is exactly what the spacing is for. Each
post links to its nearest neighbour in range and a hull goes between them —
nearest-only, because linking every pair inside the range lays a third boat
across a run of three, overlapping the two real ones. Posts stand at the water's
edge, so the boat is walked outwards from the line between them until both
gunwales are over water, and a post with water on neither side is left alone.
Boats float at the level of the water body they are in, which has to be per body
because levels across one area span metres.

Both leave as geometry in the FBX and as `vehicles.json`, a spawn list with
position, heading and size, so real vehicle models can replace the boxes. Cars
that fall outside the bbox are dropped: a bay straddling the edge comes back
whole, and unlike a building — where cutting one opens a hole in its wall — a
car has nothing to cut and one parked out over the void is simply wrong.

Four checks cover the part that could silently go wrong: that boats are inside a
water polygon, that their heights match a real water level, that cars are on
ground, and that nothing landed outside the area.

## Building function

3DBAG says how tall a building is and when it was built, but not what it is for.
The BAG does: every verblijfsobject carries a `gebruiksdoel` and a
`pandidentificatie`, which is the same building id 3DBAG uses, so the two join
on a key rather than by geometry. Over the demo area that resolves a function
for about 2200 buildings from 6500 units.

It decides which ground storey a building gets. Shopfronts along a residential
street look as wrong as a blank wall along a shopping street, so only retail and
public buildings get them; housing gets doors and windows. Where a building holds
several uses, the most street-facing one wins — a shop under flats is a shop at
street level.

## Trees

Two national sources combine into something neither has alone. The BGT registers
individual trees as points with authoritative positions but no height. AHN has
height everywhere but does not say what is a tree. Subtracting the DTM from the
DSM leaves a canopy height model, and sampling that at each BGT point gives every
tree its own measured height. Over a square kilometre of Utrecht that is 1488
trees, 91% of them with a measured height.

Two things make the difference between plausible trees and nonsense:

**The BGT returns the full version history of every object.** A naive read finds
4834 "trees" in that same square kilometre, because superseded versions stack up
on the same spots. Only rows with no closed registration are kept.

**A tree point marks the trunk, not the crown.** Sampling the canopy model at the
point lands on whatever is beside the tree and reads far too low, so the height
is the local maximum within a few metres. Building roofs are cut out of the
canopy model first, otherwise a tree standing near a wall inherits the height of
the building next to it.

Trees leave in two forms, and you get both:

- Low-poly geometry in the FBX — a trunk prism and a subdivided-octahedron
  canopy, about 42 triangles each, all sharing one `M_tree` material so the whole
  set is a single draw call. Solid geometry rather than crossed billboards, so
  nothing depends on alpha settings surviving FBX and being set up again.
- `trees.json`, a spawn list with position, height, crown radius and trunk
  height, in both RD and the model's local frame, for dropping in real Unity
  tree prefabs instead.

Set `trees.enabled` to `false` to skip them entirely.

## Materials

Two materials, by default.

**`M_aerial`** carries `aerial.png` with a top-down planar UV taken from world XY
normalised to the aerial bbox. It goes on the terrain *and* on the roof surfaces,
so roofs pick up real photo texture for free and buildings blend into the ground.

**`M_facade`** carries a generated facade. V counts storeys from the base of the
wall, at a storey height of `wall_height / storeys`, so the top row of windows
finishes flush with the eaves instead of being cut in half. Where a ground
storey has been split off, V is measured from the first-floor line rather than
from the ground, so the bottom row starts on a storey line too.

**Every wall face gets a whole number of window bays.** U is not a fixed number
of metres per tile. Each flat face of each building is measured, its bay count
rounded, and the tile stretched slightly to fit, so both corners land on a tile
edge. This is most of what makes a generated facade look designed rather than
papered: at a fixed 4 m tile a 5.4 m Dutch house front shows one window and a
second sliced in half by the party wall. Now it gets one bay of 5.4 m or two of
2.7 m. Faces too narrow for even one bay — the return of a bay window, a
chamfered corner — are centred on the tile seam, which is the blank pier between
windows, so a half-metre sliver shows brick rather than a squashed window. The
ground storey is fitted in the same pass as the wall above it, so a shopfront's
divisions line up with the windows over it. `src/facade_uv.py`.

**The facade follows the building's era, not its height.** 3DBAG carries an
original construction year on every building — 100% coverage in practice, ranging
from 1250 to 2022 in Utrecht centre — and era predicts how a wall looks far
better than height does. A 1890s canal house and a 1970s office block can be the
same height and look nothing alike. Five styles run from pre-1920 brick with
tall narrow windows, through interbellum brick, post-war plaster, and 1975-2000
panel, to contemporary glass. Height is only the fallback when a year is missing.

**Two kinds of building are not a stack of storeys, and get their own
composition.** A church has one tall volume with arched openings; dividing its
height by three metres turned the Dom into thirty-six rows of domestic windows.
A warehouse has a handful of large bays and long blank walls. Each building is
classified from its height, footprint, storey count and BAG function into one of
six archetypes — house, apartment, office, retail, industrial, monumental — and
the two that the era styles get wrong are routed to their own style, tile width
and bay height. Neither gets a ground-storey split: a cathedral has no
shopfront. Over Utrecht centre that is about 5% monumental and 4% industrial.
Classification is `classify_archetype` in `src/buildings.py`; the extra styles
are `EXTRA_STYLES` in `src/facade.py`.

**The wall under the windows is a photograph.** The generated layout gets the
bay spacing, storey lines and window proportions right, but its wall was value
noise, and at street level value noise reads as noise rather than as brick. One
CC0 texture per style — from [Poly Haven](https://polyhaven.com) — is downloaded
once, cached under `work/_textures/`, tiled to the real size the asset publishes
so the bricks come out at 210 mm rather than at whatever fills the tile, and
tinted towards the style's colour so the era palette survives. The windows,
frames, sills and storey bands are still drawn on top, because those are the
parts a photograph of a blank wall cannot supply. CC0 means no attribution
obligation travels into whatever the model ends up in. A run with no network
falls back to the procedural wall and says so; set `facade.photo_textures` to
`false` to stay fully procedural. `src/textures.py`.

**`M_facade_ground`** is the ground storey: shopfronts and doors rather than
another row of the same windows. A repeating grid of identical storeys is the
clearest sign a facade was generated, and the ground floor is what makes a
street read as a street. Because a triangle spanning two storeys cannot switch
texture partway through, the walls are cut along the first-floor line — per
building, so the cut follows the terrain. Set `facade.ground_floor` to `false` to
skip the split.

Building `GroundSurface` faces are dropped — they sit under the terrain.

## Choices worth knowing about

Some of these differ from the original plan, because the services or the export
format turned out to require it.

**The facade is a generated image, not a Blender node material.** Procedural node
graphs do not survive FBX export; Unity would receive a flat grey material and
the walls would lose their pattern entirely. `src/facade.py` generates the same
kind of pattern offline with Pillow, which keeps it procedural and parametric
while staying inside what FBX can carry. This is the plan's own stated fallback,
promoted to the default because it is the only version that reaches Unity.

**Every 3DBAG page is decoded with its own transform.** The API returns a
`transform` per page and *it changes between pages*. Merging raw CityJSON pages
and applying one transform afterwards silently warps every page but the first.
`buildings.py` decodes each page to real RD coordinates before merging anything.

**The aerial image is a mosaic, always.** The PDOK WMS advertises
MaxWidth/MaxHeight of 2500 and refuses a 4096 px request with `image size too
large` — delivered as an XML document under HTTP 200, so the status code alone
never proves success. A 4096 px image is stitched from 3x3 tiles. The WMTS
mosaic fallback exists and is covered by a test, since a fallback nothing
exercises is a fallback that does not work.

**The WMS layer name is not its title.** `Luchtfoto Actueel Ortho 8cm RGB` is the
title; GetMap wants `Actueel_orthoHR`. The config accepts either and resolves it
against the capabilities document. GetMap only offers `image/jpeg`, so tiles
arrive as JPEG and are written out as one PNG.

**EPSG:28992 is easting-first in both WMS 1.1.1 and 1.3.0** for this service.
Sending northing first returns a blank white image rather than an error, so
`imagery.py` also rejects a uniformly flat result.

**owslib is not used.** For the AHN WCS it follows the operation URL advertised in
the capabilities document, which currently 404s, and it serialises a list-valued
coverage id into the query string. A plain GET avoids both.

**A DTM over a city centre is mostly holes, by design.** Buildings are removed
from the bare-earth model, so around 49% of the source pixels over Utrecht
centre are nodata (the marker is float32 max, ~3.4e38, not zero and not NaN).
Filling those is a routine step: source pixels are block-averaged onto the
target grid, then remaining gaps grow inward from their edges.

**Buildings are assigned to the area containing their centre.** The API returns
every building that *intersects* the bbox, whole. Near Utrecht Centraal that
pulls in the Jaarbeurs halls — one 296 x 462 m object whose centre lies 300 m
outside the box — which on its own stretched the model 360 m past the terrain.
The centroid rule is the usual tiling convention: each building belongs to
exactly one tile, and geometry is never cut open. Set
`buildings.clip_mode: "intersect"` for full coverage with an unbounded overhang.

**Buildings are one merged mesh by default.** A 1 km² of Utrecht is ~2500
buildings; one mesh with two material slots is two draw calls in Unity instead of
thousands of objects. `buildings.merge: "per_building"` gives one object each.

**Wall bases are pushed below the terrain.** Because the DTM's building-shaped
holes are filled by interpolation, the reconstructed ground can sit slightly
above a building's own base. Since the `GroundSurface` is dropped, that would
show as a gap under the walls, so each wall base is extended down past the lowest
terrain sample under its footprint.

## Validation

Every run writes `work/<area>/validation.json` and exits non-zero if anything
fails. The checks cover the bbox size and RD domain, terrain coverage, gap fill
and NAP plausibility, building count, heights and triangulation, whether every
building actually reaches the terrain, aerial size, content and georeference, the
output files, model span and centring, and a re-import of the exported FBX to
re-measure it.

A run over the shipped Utrecht bbox passes 27 of 27 in about 140 s: 2557
buildings, 232k triangles, a 26 MB FBX.

## Attribution

The sources are open but they ask for credit, and 3DBAG requires a copyright
notice on reuse. `ATTRIBUTION.txt` is written next to every model.

- **3DBAG** — 3D geoinformation research group, TU Delft, and Kadaster (CC BY 4.0)
- **AHN** — Actueel Hoogtebestand Nederland via PDOK (CC BY 4.0)
- **Aerial** — PDOK / Beeldmateriaal Nederland (CC BY 4.0)
- **Trees, water, land cover, road surfaces, railway track, bridges, tunnels,
  street furniture, parking bays and mooring posts** — BGT (Basisregistratie
  Grootschalige Topografie) via PDOK (CC BY 4.0)
- **Building function** — BAG (Basisregistratie Adressen en Gebouwen) via PDOK
  (CC BY 4.0)
- **Wall surfaces** — [Poly Haven](https://polyhaven.com) (CC0). No attribution
  is required for these; they are credited because it is the decent thing, and
  listed here so you know nothing in the model carries an obligation you did not
  choose.
