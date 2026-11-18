<picture>
  <source media="(prefers-color-scheme: dark)" srcset="ui/logo-dark.svg">
  <img src="ui/logo-light.svg" alt="TerreinLab, van locatie naar 3D-demo" width="330">
</picture>

# TerreinLab

*Van locatie naar 3D-demo.* An internal tool for scouting a location and
working it out as a demo: pick a square of the Netherlands and it builds the
ground, the buildings and the street detail there from the open data, as a 3D
model (3DBAG + AHN + PDOK aerial → Blender and Unity).

Takes a bounding box in the Netherlands and bakes a textured 3D model you edit
in Blender and import into Unity. Everything runs locally and the result needs
no network at runtime.

```
output/<area_name>/
├── model.blend      the model for editing: one collection per layer, one object
│                    per building, street, junction and tree
├── model.fbx        the same model for Unity: terrain, roads, rails, bridges,
│                    tunnels, water, buildings, trees, furniture, cars, boats
├── maptile_tools.py Blender add-on: fit the ground under edited roads, re-export
├── aerial.png       aerial ortho of the same area
├── roofs_00.png …   the roofs from the same ortho at a finer resolution, packed
│                    per building into one or a few pages (see Roofs below)
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

The pipeline stops there: no Unity scene and no WebGL build. A Unity package
in `unity/` sets up the import; see [Importing into Unity](#importing-into-unity).
To look round a model without opening Blender or Unity, the UI opens it in 3D
in the browser; see [The 3D view](#the-3d-view).

**Two styles of ground.** By default the ground is built the way our 3D
modellers build an area by hand — the *layered* style: a terrain of quads
whose edge loops follow the roads, verges and ditches, coarser away from the
middle, with every road, walkway and lawn a flat object of its own lying on it, named and grouped as in their files, and
kerbs made by their curb generator so they stay editable. The *survey* style is
the ground as measured: a fine terrain whose edges follow every waterline, road
edge and building, the roads as meshes of their own, and water. See
[Two styles of ground](#two-styles-of-ground).

## Running it

There is a small web UI, and there is the command line. They do the same thing:
the UI writes a config file and shells out to `pipeline.py`, so it cannot drift
away from the CLI.

### Without a terminal

Colleagues who would rather not open a terminal can double-click a launcher in
this folder:

| | |
| --- | --- |
| Windows | **Start TerreinLab.bat** |
| macOS | **Start TerreinLab.command** |
| Linux | **start-terreinlab.sh** |

It creates a private Python environment beside itself, installs what is
missing, starts the UI and opens the browser. Nothing is typed and nothing is
installed system-wide; deleting the `.venv` folder undoes it completely.

It keeps a copy of the `requirements.txt` it last installed, and runs pip again
whenever the two differ, so an update that adds a package installs it on the
next start. A pipeline run checks the same list before it does anything else,
and stops with the exact command to fix it if a package is missing.

**The first run downloads about a gigabyte and takes several minutes.** 880 MB
of that is Blender 5, which ships as a Python library here so nobody has to
install Blender separately. Every run after that opens the page in seconds.

**It needs CPython 3.13, and only 3.13.** The model is built with Blender 5,
and Blender publishes its Python library for one Python at a time: from
Blender 5.1 that is 3.13. On anything else pip installs the other dependencies
happily and the run then fails minutes later at the Blender stage, with nothing
that points at the version, so the launchers check first.

**On Windows the launcher installs it for you.** When it finds no Python 3.13 it
asks, and on *Y* downloads the official 3.13.17 installer from python.org (30 MB),
checks it against the SHA-256 pinned in the `.bat`, and installs it for your
Windows user only: no administrator rights, nothing added to PATH, any other
Python left alone. It lands in `%LOCALAPPDATA%\Programs\Python\Python313`,
where the launcher looks for it next time. On *N*, or if anything fails, it
opens the download page instead. 3.13.17 is the newest 3.13 release python.org
built a Windows installer for. The macOS and Linux launchers still point you at
`brew install python@3.13` or your package manager, which need your password.

**An environment from before Blender 5 is rebuilt.** One made on Python 3.11
holds Blender 4.2, which pip cannot replace there, so each launcher checks the
environment's Python and makes it again when it is not 3.13 — one more long
first start. Intel Macs get no Blender 5 Python library at all: there, install
the Blender application, which the pipeline finds by itself.

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

Open <http://127.0.0.1:8765>. The page is in Dutch and has three views.

- **Modellen** (the start page) shows every area built so far as a card with
  its preview render, style, size, date and check result. **In 3D**, or a
  click on the picture, opens the model in the browser (see
  [The 3D view](#the-3d-view)); **Bekijken** opens
  the renders, the check results and the download links; **Opnieuw bouwen**
  opens the map with that area's square, name, style, detail and collision
  setting filled in, and **Verwijderen** deletes the model, its renders and its
  intermediates after asking. **Alles downloaden als zip** in the dialog packs
  `output/<area>/` into one file. A build that is running shows on top as a
  card with a progress bar, the log and a stop button, also after a page
  reload, as the page asks the server which runs are going (`/api/jobs`).
- **Nieuw gebied** opens the map full screen: a search field top left, the
  layers drawer on the right, and a build bar along the bottom with name,
  width, style, detail, the time estimate and **Model bouwen**. Click the map to
  move the square, drag to pan, scroll to zoom. The defaults are the ones to
  use. Building takes you back to Modellen, where the run card follows it.

**Builds queue.** One model is built at a time, as two Blender runs side by side
only slow each other down. While one runs, the button reads **In de wachtrij
zetten** and the next area waits under **Wachtrij** on Modellen, where **Uit de
wachtrij** takes it off again. The queue lives on disk: every run keeps its
settings and log in `work/<area>/run.json` and `run.log`, so a server that is
stopped and started again picks the waiting runs back up. Stopping the server,
with Ctrl+C or a plain `kill`, stops the run that is building, Blender
included, and the next server shows it as *onderbroken*, with **Opnieuw
starten** to build it again from its own settings. The stop button on a run
card stops Blender too, so the next run in the queue has the machine to
itself.

The page uses the Virtuele Brigade colours. It names Rijksoverheid Sans in its
font stack but does not ship the font; without it the browser's sans-serif is
used.

The style is *Layered* (as the modellers build it) or *Survey* (the ground as
measured). In the layered style water is cut into the ground and covered by a
flat water object, harbours get quay faces, jetties and mooring posts, and cars
and boats start unticked, as the modellers' files
have none; tick them to add them.

The page is written for someone who has never opened a GIS or a 3D package.
**Detail is one choice, not a panel of sliders**: Draft, Standard or High (see
[Resolution and quality](#resolution-and-quality)). Under it the page prints
what that means in words — *"cars and pavements read in the photo"* — and the
numbers it stands for: photo resolution, ground grid spacing, wall texture
size. There used to be a dozen sliders here — ground triangle budget, feature
straightening, contour spacing, road face size and more — and every one of them
was a way to make the model worse that a newcomer had to learn to leave alone.
They are gone; a config file that still sets one loads and says what replaced
it. The **?** button opens *Wat je krijgt, en hoe je het opent*, which names the
output files, gives the steps for Blender and Unity and has the Unity importer
package to download (see [Importing into Unity](#importing-into-unity)).

The layers drawer lists every optional dataset with a switch and a live status
light, so you pick what goes in and see what is answering right now. Terrain,
aerial imagery and buildings are always in — without them there is no model —
so the drawer only says so; if one of those is down the build button is
disabled with the reason given. An optional source that is down
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

Rough costs for a 1 km² area, previews included: Draft about 4 minutes,
Standard about 6, High about 11. Previews are a large share of that, so turning
them off cuts the wait noticeably.

The map works directly in RD New, so the square you position *is* the bbox that
gets built — no reprojection anywhere in the page. It is plain HTML with no
JavaScript dependencies, and the server is standard library only. The 3D view
is `ui/viewer.js`, WebGL2 written for this page, without a library either.

Tiles come straight from PDOK. If the browser cannot reach it — behind a
corporate proxy, or on a machine where only the pipeline process has network —
the page detects that at startup and routes tiles through the local server
instead.

### The 3D view

**In 3D** on a model's card, in its dialog or on a finished run opens the model
full screen: the scene as built, with the aerial, the facades and the trees,
under a low sun with shadows. It is for looking at a model without opening
Blender or Unity, and for showing it to someone who has neither.

| | |
| --- | --- |
| Drag | move over the map; the ground under the cursor stays under it |
| Right-drag, or Shift + drag | turn and tilt |
| Scroll | zoom towards the cursor |
| Double-click | fly to that point |
| Click | what it is and where: name, layer, properties, faces, RD and NAP |
| Z, or **Wireframe** | the wireframe on and off; **Quads** or **Triangles** beside it |
| Shift + Z | the wireframe as triangles, the way the FBX and the graphics card see it |
| M, or **Meten** | measure: click two points for the distance, the level distance and the height difference |
| Search field | find an object by name or by a property, a BAG id say, and fly to it |
| **Link** | copy the address of this view, camera included |
| Esc | stop measuring, close the search |
| Touch | one finger moves, two fingers zoom, turn and tilt |
| Keys | arrows move, Shift + arrows turn and tilt, + and − zoom, Home shows the whole area |

**Overzicht** and **Boven** frame the whole area, at an angle or straight from
above with north up, and the compass turns the view back to north. **Lagen**
lists every collection of the `.blend` with a switch: the trees off to see the
street, the buildings off to see the ground. The collision shapes are a layer
too, off at first; switched on they lie over the model in translucent magenta,
so a wall proxy that misses its wall or a barrier without one shows at a
glance. A click on a building gives its
BAG id, construction year and height from 3DBAG; on a tree its height and
whether it came from the BGT register or was found in the height model. Under
the cursor the page reads out RD and NAP all the time, so the view also measures
a model against the map.

**The wireframe is the mesh as Blender shows it.** **Wireframe** draws the
edges of every face over the model, with the textures off and the faces plain grey: quads as quads, n-gons whole, without the
diagonals the export triangulation adds, so the 4 m squares, the rungs across a
road and the rows along a verge can be checked in the browser as in Blender's
edit mode. The lines are dark on the grey faces; the clicked object's are
orange. Far off, where an edge of a few metres shrinks to
a pixel or two, the lines fade, or the whole model would turn black: zoom in
to read them. Switch layers off to look at one alone: the
ground with the buildings and trees off is the layout the modellers asked for.
A click also counts the object's faces and vertices as Blender does, with its
modifiers and before the triangulation: the kerbed walkways of Utrecht centre
are 44,166 quads, 2,504 triangles and 6 n-gons. **Triangles** beside the
switch, or Shift + Z, draws the same mesh as the FBX carries it, every quad cut
in two: the diagonals Unity will see.

**Meten** turns a click into a point: the first click sets one end, the second
the other, and the card gives the straight distance, the distance over the
ground plan and the height difference, in metres to the centimetre, with RD and
NAP of both ends. A third click starts again. The search field top left finds
objects by name or by a text property, so a BAG id taken from the BAG viewer
finds its building, and a click on a result flies there and selects it.

The address carries the model and the camera:
`http://<your-machine>:8765/#3d/utrecht_dom/c=12.4,3.1,-40.2,180.0,0.785,0.611`
opens that model from that spot (the point looked at in the model's own metres,
the distance to it, then turn and tilt in radians), as the page keeps the
address up to date while you look round. **Link** copies it, so a colleague sees the same corner. The
browser's back button leaves the 3D view.

**What it reads.** The Blender stage writes the view as its last step, to
`work/<area>/viewer.glb`: a binary glTF of the scene with every modifier as the
render has it, so the kerbs and the export triangulation are in, as in the FBX.
The objects of a layer are merged per material and per tile, a grid of six by
six over the area, so a thousand buildings are a few hundred draws and the page
skips the tiles outside the view. Street furniture, barriers and vehicles are
also left out beyond 450 m, where they would be a pixel each. The trees, ten thousand objects sharing a dozen meshes, stay
instances (`EXT_mesh_gpu_instancing`), and normals are bytes
(`KHR_mesh_quantization`). Every vertex carries the number of its object and
the file lists the objects with their custom properties, face counts and
bounds, which is what a click reads back and the search flies to. The wireframe costs a byte per triangle: which of
its three sides are edges of the mesh. The triangulation is render-only and
keeps every vertex and its number, so the edges are read from the scene as the
viewport has it and looked up among the triangles' sides; the page makes the
lines from that the first time it is switched on. The textures are not in the
file: it names them, and the page loads them from `output/<area>/`, so the
aerial is not stored twice. Over a 500 m square of Utrecht centre the file is
17.9 MB for 399,219 triangles and 1,927 objects, written in under three
seconds, and the browser has it on screen in under one.

It lives in `work/` because it is a cache for the page and no part of the
model: `output/<area>/` stays what goes into Unity, and deleting the view
loses nothing.

**When there is none, or it is out of date, the server writes it.** A
`viewer.json` beside the view stamps it with the size and time of the
`model.blend` it shows, and with the layout of the file, which the page and the
server agree on. A model built before there was a 3D view, one whose `work/`
folder was cleared, one edited in Blender and saved since, and one whose view
is of an older layout (from before the wireframe, say) all miss that stamp,
and opening them runs `blender/web_export.py` on the `.blend`
through the same Blender the pipeline uses, one model at a time; the page says
*3D-weergave voorbereiden* meanwhile, about three seconds over that square. So
the next time it opens, the 3D view shows what was saved in Blender. A `.blend`
that Blender cannot open is reported with Blender's own message and tried again
only when asked.

The view needs WebGL2, which every current browser has. A texture larger than
8192 px, an aerial over two kilometres at High detail, is scaled down to fit
the graphics card.

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
module (`pip install -r requirements.txt`, CPython 3.13) — and among those, a
Blender 5 first: an installed one, else the pip module. An older Blender is used
only when there is nothing newer, and the run says the walkways will have no
kerbs. The Blender found, and its version, are in the log.

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
  "style": "layered",
  "detail": "standard",
  "export": { "collision": true }
}
```

`style` is `layered` (the default) or `survey`; see
[Two styles of ground](#two-styles-of-ground). Like `detail`, a style only
fills in defaults, so anything the file sets wins — with one exception the
layered style cannot honour: it makes its own roads, so it has no survey-style
road meshes, and a file that asks for them is told so in the log.

`detail` is `draft`, `standard` or `high` and fills in the photo resolution,
the ground grid and the wall textures (table below). Anything set explicitly in
the file wins over it, so `"detail": "high", "terrain": {"grid_m": 2.0}` is High
with a 2 m ground.

Settings that no longer exist — the terrain triangulation and road mesher knobs
in particular — still load: the run logs one line per retired key saying what
replaced it, and otherwise ignores them. `RETIRED` in `src/config.py` is the
list.

A `"crs": "EPSG:4326"` bbox is accepted too, with longitude in `xmin`/`xmax` and
latitude in `ymin`/`ymax`. It is converted to RD New once, in `geo.py`, and
everything downstream stays in RD. The bbox is validated as roughly square and
roughly 1 km per side, and warns rather than fails when it is not.

## Resolution and quality

One choice, three levels. Detail is set as ground resolution rather than pixel
count, so a level means the same thing whatever the area size, and the pixel
counts are derived from the box.

| Detail | Aerial | Roofs | Ground grid | Facade | Normal map | Buildings per wall look |
| --- | --- | --- | --- | --- | --- | --- |
| Draft | 50 cm/px | on the aerial | 4 m | 256 px | off | 160 |
| Standard | 25 cm/px | 10 cm/px, up to 48 Mpx | 2 m | 512 px | on | 80 |
| High | 12 cm/px | 8 cm/px, up to 128 Mpx | 1 m | 1024 px | on | 40 |

**Both dials hit a source-data ceiling**, and going past it costs time without
adding information. The aerial layer is an 8 cm ortho, so a 1 km area is fully
resolved at about 12500 px. The AHN DTM is 0.5 m. The pipeline warns instead of
refusing; the UI shows the derived numbers and flags when a setting has gone
past what the data holds.

**The aerial jump from 25 to 12 cm/px is large; from 12 cm to native 8 cm it is
small.** At 25 cm/px cars are blobs. At 12 cm/px you can see individual cars,
kerb lines and bare branches. Native 8 cm adds little over that for double the
file (92 MB against 174 MB for 1 km²). The roofs are the exception to the
column: at Standard and High they have a photo of their own, finer than the
ground's (see [Roofs](#roofs)).

**The ground grid is a trade between faithful and editable.** How far the
drawn ground strays from every surveyed AHN pixel under it, measured over the
Bronbeek square kilometre:

| Grid | Vertices | Median | 95% | 99% | Pixels off by > 25 cm |
| --- | --- | --- | --- | --- | --- |
| 4 m | 63k | 2.9 cm | 22 cm | 44 cm | 4.1% |
| **2 m** | **251k** | **1.7 cm** | **11 cm** | **23 cm** | **0.8%** |
| 1 m | 1.0M | 1.1 cm | 5.8 cm | 12 cm | 0.2% |

AHN's own vertical accuracy is about 5 cm systematic plus 5 cm stochastic, so
at 2 m the grid gives up little the survey could resolve, and it is the spacing
a person can still sculpt and select by hand. 1 m is for a smaller area or
close-up work. The run reports these figures for its own area as
`terrain.survey_error` in `validation.json`.

**Unity has its own limits.** Textures over 16384 px cannot be imported at full
size, which a 2 km area at native resolution would exceed. Imported textures
are also capped at 2048 by default, so raise *Max Size* on `aerial.png` or none
of this is visible; the Unity package does that by itself. A terrain over 65k vertices needs a 32-bit index buffer,
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
It is written as `*_normal.png`, the suffix the Unity package keys off to
import it as a normal map; without the package, set its Texture Type to
*Normal map* by hand.

Other knobs worth knowing:

| Key | Default | What it does |
| --- | --- | --- |
| `facade.variants` | `5` | How many era styles to assign by construction year, up to `5`. The monumental and industrial styles are always present on top of these. |
| `facade.looks_per_era` | `12` | The most looks any one era may get. A ceiling, not a count: looks are handed out per era in proportion to how many buildings wear them, so a sparse era spends one texture however high this goes. `1` switches the whole thing off. Clamped to 12. |
| `facade.buildings_per_look` | `80` | How many buildings of one era should share a wall. The lever that actually decides how repetitive a street looks; lower buys variety at one texture each. |
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
| `trees.ground_shade` | `0.35` | How far the aerial is darkened under each canopy. The photo is flown leaf-off, so without this the ground under a tree is lit as if the tree were not there. Ambient occlusion, not a sun shadow. `0` leaves the photo alone. |
| `surfaces.water` | `true` | Replace the interpolated canal bulge with real water surfaces. |
| `surfaces.road_geometry` | `true` | Build roads as their own surface, one object per street, instead of only as raster class. See [Roads](#roads). |
| `surfaces.road_lift_m` | `0.03` | How far the carriageway sits above the surveyed ground. The terrain is then fitted under it, so this is a height, not a depth-fighting margin. |
| `surfaces.kerb_m` | `0.10` | How far footways stand above the carriageway, with a vertical kerb face between. `0` lays everything flat. |
| `surfaces.nwb_url` | PDOK NWB | Where the national road centrelines come from. Without them the roads draw their own centrelines and nothing else changes. |
| `surfaces.land_cover` | `true` | Classify the ground and export the class map. |
| `surfaces.detail_strength` | `0.22` | Per-class grain mixed into the aerial. 0 disables it. |
| `surfaces.water_depth_m` | `1.2` | How far each bed is sunk below its own water level. |
| `furniture.enabled` | `true` | Lampposts, bollards, sign posts and benches. |
| `export.collision` | `false` | Add invisible `Collision_*` shapes for physics. See [Collision shapes](#collision-shapes). |
| `export.collision_tree_crowns` | `true` | Whether a tree's collider includes its canopy. `false` leaves the trunk alone, at a quarter of the triangles. |
| `barriers.collision_thickness_m` | `0.30` | The thinnest a collider may be, which is not the thinnest a fence may look: a 6 cm panel is thin enough to tunnel through at 50 Hz. |
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
| `buildings.roof_cm` | `10` | Resolution of the roofs' own photo, set by `detail`. `0` puts the roofs on the aerial as before. Finer than the 8 cm source is clamped to 8; no finer than the aerial is the same as `0`. See [Roofs](#roofs). |
| `buildings.roof_max_mpx` | `48` | The most pixels all roof pages may hold together, set by `detail`. Where the roofs need more at `roof_cm`, they are fetched coarser, the log warns and `roof_resolution_as_asked` says by how much. |
| `buildings.roof_atlas_px` | `8192` | The largest side of a page. WebGL and many GPUs stop at 8192; Unity imports up to 16384, and anything above 8192 is warned about. |
| `buildings.roof_padding_px` | `4` | Photo kept round each roof inside its rectangle, so the first mipmaps do not blend it with the rectangle packed next to it. |
| `terrain.grid_m` | `2.0` | Ground grid spacing; set by `detail`. See [Terrain](#terrain). |
| `terrain.feature_edges` | `["water", "roads", "buildings"]` | Which outlines the terrain's edges follow, each with a row of quads alongside. `[]` gives the plain grid. |
| `aerial.deshadow` | `1.0` | How much of the flown sun to take back out of the photo, 0 to 1. See [The flown sun](#the-flown-sun-and-taking-it-back-out). `0` leaves it as flown. |
| `aerial.max_request_px` | `2000` | Tile size for the WMS mosaic; the service caps requests at 2500. |

### Roofs

A roof is most of what a drone sees, and at Standard's 25 cm a pixel is a
quarter of a metre of tiles: a dormer is a few pixels and a skylight is none.
The ortho is 8 cm. Fetching the whole area at that is nearly what High does,
and it spends most of the pixels on ground that is not the problem. So the
roofs get a photo of their own, fetched only where they are (`src/roofs.py`):

* **Rectangles on one grid.** Each building's roof extent, plus
  `roof_padding_px` (4) of photo round it, becomes a rectangle of pixels on
  one grid anchored at the area's north-west corner. Where two overlap and one
  rectangle round both holds no more pixels than the two apart, they become
  one: a terrace is one strip rather than a row of slivers each carrying its
  neighbours' party walls.
* **Packed into pages.** Tallest first, bottom-left along a skyline, into pages
  of at most `roof_atlas_px` (8192) a side. A page is sized to what it holds,
  in multiples of 4 for block compression, and there is one page when one will
  do.
* **A budget.** When the pages would hold more than `roof_max_mpx`, the
  resolution is coarsened in whole millimetres until they fit; the log warns
  and `roof_resolution_as_asked` says by how much. If not even the ground's
  own resolution would fit, the roofs stay on the aerial.
* **Few requests.** The grid is cut into tiles of `aerial.max_request_px`
  (2000 px), and each tile a rectangle reaches is one GetMap for the part of it
  the rectangles need, four at a time. Every pixel lands on the same grid, so a
  rectangle split between two requests joins without a seam and nothing is
  resampled. A window the WMS refuses is mosaicked from WMTS tiles; if neither
  answers, the roofs stay on the aerial and the run warns.
* **Mapped and relit.** Each roof triangle's UV points into its rectangle
  (`work/<area>/roofs.npz`, read by the Blender stage), and the relighting
  stage takes the flown sun out of the pages exactly as it does out of the
  ground (see [The flown sun](#the-flown-sun-and-taking-it-back-out)).

Measured over a 500 m square of Utrecht centre round the Dom, at Standard, the
same area built with `roof_cm: 0` against the default:

| | `roof_cm: 0` | `roof_cm: 10` |
| --- | --- | --- |
| roofs | on the 2000 px aerial, 25 cm/px | one 5504 x 5204 page, 10 cm/px |
| fetched for the roofs | nothing | 27.2 Mpx in 16 requests, 3.6 s |
| run time | 137 s | 18 s more: 5.9 s roof stage, 10 s relighting, 2 s checks |
| output folder | 51 MB | 89 MB, of which `roofs_00.png` 38 MB |
| detail on the roofs | 4.1 | 12.2 |

The last row is the mean absolute Laplacian of luminance over the roof pixels,
both read at 10 cm: the finest detail in the picture, three times what the
upsampled aerial has. Most of the time is writing PNG, not the network: a
29 Mpx page takes 7.8 s to compress and arrives in under 4.

The centre of Utrecht is the hard case for this. Roofs cover 53% of the square
there, and with streets running every way their rectangles, merged, add up to
more than the square itself: 241,000 m² against 133,000 m² of roof. So the page
holds about as many pixels as the whole square would at 10 cm. Where roofs
cover less of the ground the page shrinks with them, while the ground's photo
does not.

**At a square kilometre the budget decides.** The square kilometre of the
default `config.json`, which overlaps most of the 500 m square above, holds
2,557 roofs: at 10 cm they would need more than the 48 Mpx Standard allows. They come out at 13.6 cm/px instead, on
one 7164 x 6700 page: 54 Mpx fetched in 23 requests in 8.4 s, 30 s added to the
fetch and relighting stages of a 400 s run, and a 71 MB page beside a 30 MB
aerial. The log and `roof_resolution_as_asked` both say so. That is still
nearly twice as sharp as the ground in each direction; for 10 cm over a square
kilometre of city centre, raise `roof_max_mpx` and expect a second page.

Two things were weighed and left out. Rectangles turned to each building would
pack tighter where the streets run every way, but every roof pixel would then
be resampled off the service's grid, which costs some of the sharpness this is
for. And the pages are PNG like every other texture here: the photo has been
relit, so a JPEG would be a second lossy generation. As JPEG at quality 92 the
page above would be 7.6 MB rather than 38; on the graphics card the two are the
same size.

`roof_cm: 0`, and Draft, give the previous model: no stage, no page, and the
roofs on `M_aerial` with the aerial's planar UV.

## How it fits together

```
src/geo.py         bbox parsing, WGS84 to RD, origin offset
src/elevation.py   AHN WCS -> GeoTIFF -> regular height grid; the fit under roads
src/terrain_mesh.py the terrain mesh: outlines as edges, rows, grid, fill
src/buildings.py   3DBAG API -> CityJSON -> semantic mesh data
src/imagery.py     PDOK WMS/WMTS -> aerial.png + georeference
src/roofs.py       the roofs at a finer resolution: rectangles, packing, roofs_00.png
src/facade.py      generated facade, tree, water and furniture textures
src/facade_uv.py   fitting whole window bays across each wall face
src/textures.py    photographed CC0 wall surfaces, fetched once and cached
src/sources.py     the registry of datasets a model can be built from
src/bgt.py         shared BGT client: paging, version filter, rasterising
src/trees.py       BGT tree points, heights from AHN DSM minus DTM
src/surfaces.py    BGT water bodies, land cover and road surfaces
src/nwb.py         NWB-Wegen road centrelines, names and junctions
src/roads.py       road ladders: stations, cross-sections, kerbs, objects
src/furniture.py   BGT lampposts, bollards, signs and benches
src/vehicles.py    cars in the parking bays, boats between the mooring posts
src/rails.py       railway, tram and metro track from BGT spoor
src/structures.py  bridge decks and piers, and tunnels below ground
src/usage.py       BAG building function, joined to 3DBAG on the building id
src/export.py      metadata.json and the Blender scene description
src/validate.py    the headless checks
blender/process.py builds the scene, assigns materials, saves .blend, exports FBX
blender/maptile_tools.py  the Blender add-on shipped beside the model
blender/preview.py renders preview images of an exported FBX
blender/web_export.py  the scene as one binary glTF, for the 3D view
ui/server.py       local web UI, standard library only
ui/index.html      the page: models gallery, full-screen RD map with build bar, live log, results
ui/viewer.js       the 3D view: WebGL2, reads what web_export.py writes
unity/com.terreinlab.importer/  Unity package: imports every model with the
                   right settings, its properties and RD/NAP on components
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

## Opening it in Blender

Open `model.blend`. It is the same scene the FBX was exported from, saved with
relative texture paths, so the output folder can be moved or zipped whole.

**Every layer is a collection, and the big ones a parent too.** Collections are
how Blender organises a file; FBX has none, and Unity builds its hierarchy from
parent objects, so a layer of many objects is both — a collection in the
outliner and an empty that everything in it hangs from:

| Collection / parent | What is in it |
| --- | --- |
| `<Prefix>_Terrain` | layered style: the terrain `<Prefix>_Terrein` and the road, motorway and walkway objects, together one sheet of ground (see [Two styles of ground](#two-styles-of-ground)) |
| `Terrain Cutout` | layered style: the grass object |
| `Terrain` | survey style: one object, the ground; its edges follow the outlines (see [Terrain](#terrain)) |
| `Roads` | survey style: one object per street, junction and paving area (see [Roads](#roads)) |
| `Buildings` | one object per building, `Pand_<BAG id>`, with `bag_id`, `build_year` and `height_m` as custom properties |
| `Water` | water surfaces and beds |
| `Trees` | `Trees_registered` and `Trees_detected`, one object per tree |
| `Street furniture`, `Barriers`, `Vehicles`, `Rail`, `Structures` | as named; in the layered style `Barriers` also holds the motorway's guardrails, `<Prefix>_Geleiderail` |
| `Collision` | the physics proxies, hidden; see [Collision shapes](#collision-shapes) |

**Trees are instances.** Ten thousand trees share twelve meshes: three kinds,
four shapes each. A tree is an object with its own position, height, crown size
and spin, pointing at one of those meshes. Edit one shared mesh and every tree
of that shape changes; move or scale a tree and only it does.

**Quads stay quads in Blender.** The ground and the roads carry a *Triangulate*
modifier set to apply on render and export only, splitting every quad the same
way the pipeline measured heights on. In the viewport you edit the quads.
Blender's FBX exporter applies only what the viewport shows, whatever its
render option says — measured on 4.2 and 5.2, a render-only modifier never
reached the file, so until this was found Unity cut every quad its own way —
and so the export, and the TerreinLab tools' re-export, switch every modifier to
its render setting while they write and back after.
`terrain_exported_as_triangles` checks the result.

**The terrain's outlines are one click away.** Every outline is a vertex group
on the `Terrain` object, so *Object Data Properties → Vertex Groups → Select*
picks a whole waterline or every road edge at once:

| Vertex group | What it selects |
| --- | --- |
| `outline_waterline` | the edge of the water: bank above, bed below |
| `outline_road` | where the ground meets the road surface |
| `outline_building` | the footprint of every building |
| `outline_row` | the inner edge of the row along an outline: the next loop in |
| `outline_bed` | the bottom of the bank inside the water, and a ditch's centre line |

The outlines are also marked as seams, so they show in edit mode, and carried
as attributes for anything scripted: `outline` on each edge (1 waterline, 2 road
edge, 4 building, 8 row, 16 bed, added together where they meet),
`ground_kind` on each face (0 land, 1 road, 2 water, 3 building) and
`ground_part` (0 grid, 1 row, 2 fill). To work on one outline, select its
group, then *Select Linked* or *Select Loops* from there; with proportional
editing on, moving a waterline takes the bank with it.

### The TerreinLab tools

`maptile_tools.py` is written beside every `model.blend`. **Install it once as
an add-on** — *Edit → Preferences → Add-ons*, the ⌄ menu at the top right,
*Install from Disk…*, pick the file, tick it — and the **TerreinLab** tab is in the
sidebar (N) of every model from then on. It is also inside each `.blend` as a
text block: to use the tools without installing, open it in the Text Editor and
press *Run Script*. The file does not run it by itself on opening, because
Blender halts every file that asks to with a security warning. Four buttons:

* **Fit terrain under roads.** After moving, widening or raising a road, the
  ground under it is wrong. This redoes the pipeline's fit: the same points,
  the same 5 cm clearance, the same preference for lowering vertices hidden
  under the road. It only ever lowers, from the heights the terrain has now,
  so a sculpt survives it and running it twice changes nothing the second
  time; to get the ground back from under a road you moved away, Reset first
  and Fit after. It works on the terrain whatever its faces, including any you
  have remeshed yourself. Kerb and skirt faces are left out. In the layered
  style there is nothing to fit: the roads, walkways and lawns are pieces of
  the ground itself.
* **Reset terrain to survey.** Puts every terrain vertex back on its measured
  height (`measured_z`), roads ignored; in the layered style every ground
  piece, roads included, since they are the ground.
* **Drop selected onto ground.** Moves each selected object vertically onto the
  road, walkway, lawn or terrain under it — for a lamppost after a road moved,
  or anything you added.
* **Export FBX for Unity.** The pipeline's own export settings, so a re-export
  after editing lands in Unity exactly like the original did.

## Importing into Unity

**Install the TerreinLab Importer package once per Unity project**, and every
model imports with the right settings by itself. It needs Unity 2021.3 or
later. In *Window → Package Manager*, click **+** at the top left, choose *Add
package from disk…* and pick `unity/com.terreinlab.importer/package.json` from
this folder. Unity uses the package where it lies rather than copying it, so
put the folder somewhere it will stay before you add it. On a machine without
this folder, the **?** dialog of the UI offers the package as a zip: unpack it
and pick its `package.json` the same way.

Then copy the output folder into `Assets/`, all but `model.blend`. The FBX
finds its textures by name beside it. Unity imports a `.blend` itself when
Blender is installed on the machine, and the model would come in twice.

The package treats an `.fbx` as a TerreinLab model when the same folder holds
`metadata.json` and `ATTRIBUTION.txt` and the metadata gives the RD origin
(`crs`, `origin_rd` and `ground_z_offset_nap`). It leaves every other asset in
the project alone. For a TerreinLab model it:

- **sets the model up as the list below says**: Scale Factor 1 with Convert
  Units on, Bake Axis Conversion on, Generate Lightmap UVs off, Index Format
  Auto, the materials kept inside the model and pointing at the textures beside
  it, and no rig or animation, since the model has none;
- **sets up the textures beside it**: `aerial.png` at Max Size 8192, or 16384
  when the aerial is larger than 8192 px (Android, iOS, tvOS and WebGL stay at
  8192); every `roofs_*.png` the same way, and kept at its own size rather than
  scaled to a power of two; every `*_normal.png` as a normal map; `landcover.png` as data, which
  means no filtering, mipmaps or compression, and readable from scripts; and
  the rest with mipmaps and repeat wrapping. `fence.png` keeps its alpha
  coverage in every mipmap, so fences do not thin out with distance;
- **turns on GPU instancing** in every material of the model. `M_tree` is the
  one that matters: ten thousand trees share twelve meshes and that material,
  and with instancing each shape is drawn in one batch. Marking the
  `Buildings` and `Roads` hierarchies *Static* is still yours to do;
- **keeps the custom properties** of each object on a `TerreinLabProperties`
  component: `bag_id`, `build_year` and `height_m` on a building, `kind` on a
  road, `height_m` and `detected` on a tree, `maptile_role` on the pieces of
  ground. The Inspector lists them, and scripts read them as `BagId`,
  `BuildYear`, `HeightM`, `Kind`, `Detected` and `MaptileRole`;
- **puts a `TerreinLabModel` component on the model's root**, holding the area
  name, `origin_rd` and `ground_z_offset_nap` from `metadata.json`, with
  methods that turn a Unity position into RD and NAP and back (see
  [Coordinates](#coordinates)). They go through the root's transform, so they
  still hold after you move, turn or scale the model:

  ```csharp
  using TerreinLab;

  var model = GetComponentInParent<TerreinLabModel>();
  RdNapPosition here = model.WorldToRdNap(transform.position);  // here.RdX, here.RdY, here.Nap
  Vector3 there = model.RdNapToWorld(136900.0, 455850.0, 5.2);
  ```

  A script in an assembly definition of its own needs a reference to
  `TerreinLab.Importer` to see these.

The package applies these settings on every import, so a model cannot arrive
half set up, and changing one of them in the Inspector has no effect while it
is installed. To import a model differently, to bake lightmaps for example, use
a copy of `model.fbx` in a folder without `metadata.json`: to the package that
is an ordinary FBX.

**Models already in the project** before the package was installed, or before
it was updated, keep their old import until you choose *Tools → TerreinLab →
Reimport models*. It reimports every TerreinLab model in the project and the
textures beside it, writes the settings into their import settings so the
Inspector shows them, and turns on GPU instancing in any material you have
extracted from a model, which an import cannot reach. If you copy a model in
while Unity is open and its `metadata.json` or `ATTRIBUTION.txt` arrives after
the FBX, the package reimports the FBX and the textures by itself.

**Without the package**, select `model.fbx` and set these in the Inspector.
They are also what to look at when a model imported with the package looks
wrong.

1. **Model → Scale Factor: 1, Convert Units: on.** The model is in metres.
2. **Model → Bake Axis Conversion: on.** Without it the root object arrives
   with a −90° X rotation, which is harmless until a script reads positions or
   something is parented to the model. With it every transform is clean:
   X east, Y up, Z north.
3. **Model → Generate Lightmap UVs: only if you will bake lighting.** It
   overwrites the second UV channel, which on the roads is `RoadUV`, the
   along-and-across mapping for road textures. If you need both, duplicate the
   road objects for lightmapping.
4. **Model → Index Format: Auto.** The ground has more than 65k vertices and
   needs 32-bit indices; Auto handles it.
5. **`aerial.png` → Max Size: 8192** (or 16384). Unity caps imported textures at
   2048 by default, and the whole aerial is shrunk to that otherwise. The same
   for each `roofs_*.png`, and set its **Non-Power of 2** to *None*: a page is
   sized to what it holds (5504 x 5204 over Utrecht centre), and the default
   *To nearest* would scale that down to 4096 x 4096.
6. **Tree materials → Enable GPU Instancing.** Ten thousand trees share twelve
   meshes; instancing draws each shape in one batch. Marking the `Buildings` and
   `Roads` hierarchies *Static* lets Unity batch those too.

Custom properties (`kind` on roads, `bag_id` and `build_year` on buildings)
arrive as FBX user properties either way. Without the package, an
`AssetPostprocessor` of your own reads them in
`OnPostprocessGameObjectWithUserProperties`, which is what the package does.

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

## Two styles of ground

The modellers sent a hand-made model of Tankstation Hazeldonk to say how they
want an area built. It is built in layers, each a separate object a person can
grab: a **terrain** of about 2,100 faces for the square kilometre with the
aerial on it; **roads, the motorway, walkways and grass** as flat faces lying a
few centimetres above it, each in a flat colour of its own; and **no modelled
kerbs**: every walkway and traffic island is a flat outline carrying their
*CurbGenerator* geometry-nodes modifier, which raises it and builds the kerb
round it, so editing the outline edits the kerb. The *layered* style builds
that from the open data, and it is the default.

With the Hoek van Holland model in hand they asked for three changes, which
are how it is built now. **The ground once**: a terrain under every road is
the ground drawn twice, and deleting it left holes wherever no layer was, so
the ground is one sheet cut into the terrain, road, walkway and grass objects,
meeting edge to edge. **No split by compass point**: one object per layer;
they divide the model themselves, for culling. **One resolution**: no
coarser squares far out and no denser patches where the survey rises, roads
as ladders, and no fans of slivers. The *survey* style is what the pipeline built before; the
sections from [Terrain](#terrain) to [Street detail](#street-detail) describe
it.

### The layered style

How it is built (`src/layered.py`):

* **What each BGT polygon is.** Road parts, supporting road parts and the
  vegetated and unvegetated terrain parts are read into five layers:

  | BGT | Layer | Material |
  | --- | --- | --- |
  | `wegdeel`, carriageway of a motorway (*autosnelweg*, *autoweg*) | motorway | `Asfalt` |
  | `wegdeel`, footpath or pedestrian area | walkway | `Beton_Geel` |
  | `wegdeel`, cycle path | road | `Fietspad` |
  | `wegdeel`, brick or tiled carriageway | road | `Bakstenen` |
  | `wegdeel`, unpaved | terrain | — |
  | `wegdeel`, any other | road | `Asfalt` |
  | `ondersteunendwegdeel`, traffic island, paved / green | walkway / grass | `Beton_Geel` / `Gras` |
  | `ondersteunendwegdeel`, verge, green / paved | grass / road | `Gras` / `Verharding` |
  | `begroeidterreindeel` | grass | `Gras` |
  | `onbegroeidterreindeel`, paved / not | road / terrain | `Verharding` / — |

  `Asfalt`, `Bakstenen`, `Beton`, `Beton_Geel` and `Gras` are the modellers'
  own materials, colours included; `Fietspad` and `Verharding` are the two
  surfaces their square did not need. A green polygon of at most 2,500 m² with
  paving round at least 90% of its edge is an island: a walkway, so it gets
  kerbs, still green on top. Parts on bridges and in tunnels
  (`relatieve_hoogteligging` not 0) are left to the structures stage. BGT
  water (`waterdeel`) is a region of terrain of its own, cut into the ground,
  with a flat water surface over it (see *Water* below).
* **Outlines.** The polygons are cut into one region per spot, simplified
  together by up to `outline_m` (25 cm) so that an edge two regions share stays
  shared, and snapped to a 1 cm grid.
* **Hairpins.** The curb generator mitres its kerb at every corner, and at a
  corner that turns back on itself the mitre runs away: measured, a 2° tip
  throws the kerb 3 m, a 3° notch 10 m, and at Hazeldonk a needle in a
  footpath outline sent one kerb vertex 825 m off. A walkway corner under 20°
  is taken out where that moves no more than 2 m² (under 6°, always), from the
  walkway and the piece beside it alike, so the pieces still tile the area; a
  walkway too thin for a kerb on each side, or a small one keeping a corner
  under 6°, becomes part of its neighbour. At Hazeldonk that is 0.3% of the
  footpath area.
* **Ground, in quads** (`src/layered_mesh.py`). The modellers' feedback on
  the first version, a triangulation refined until it fitted the survey, was:
  quads rather than a complete triangulation, with minimal but realistic height
  variation; whole continuous quads along the roads; edge loops that follow
  the shape; points where outlines meet (round a roundabout); the ditches; and,
  on the next version, one resolution everywhere. So each region is meshed the
  way they lay theirs out:
  * *Rungs across.* Every outline gets stations 4 m apart, everywhere; only an
    outline up to 20 m long on the edge of the area is left whole, so a road
    cut off there ends in one quad. A road narrower than about 20 m gets rungs straight across from
    kerb to kerb — one quad across, one after the other along the road, and
    radial quads round a roundabout. A rung must meet the far kerb within 30°
    of square, and lands on the station across from it where one is within
    2 m, so the two kerbs' stations pair up into one ladder.
  * *Loops along.* Grass and open ground narrower than about 13 m — a verge,
    the strip between two carriageways, the slope of a dike — get rungs across
    too, each cut into pieces about 4 m long, so the cuts line up into edge
    loops running along the strip.
  * *Rows and grid.* Wider ground gets a row of quads along each outline (6 m
    on grass and open ground, 10 m along a road, 2 m along a footpath, a yard
    or water) and a grid of 4 m squares inside, the same everywhere, land and
    water bed alike. The squares used to merge up to 32 m in the far fields
    and split to 2 m, and loops were cut wherever a face missed the survey;
    the modellers found the patches of dense faces that left more work than
    detail, and the mix of sizes hard to edit.
  * *Turned to the streets.* The grid is turned to the direction most of the
    outline length runs in, counted modulo 90° (a street and the one square to
    it agree), so a row along a street meets it square on. Across Hoek van
    Holland the streets run 36° off north, and every row along one met a
    north–south grid in a band of triangles. Where no direction holds a
    quarter of the outline length within 4°, the grid stays north–south.
  * *Junctions through the grid.* What the rows leave of a junction is a big
    polygon with a station every 4 m round it and nothing inside, and its
    triangulation was fans of long slivers from the corners. Bigger than two
    grid squares, it is filled through the grid nodes inside it instead, and
    the triangles paired back into quads the size of a square.
  * *A loop down a road on a dike.* A road's rungs run straight from kerb to
    kerb. Where the survey across their middle is more than `tolerance_m` off
    them (90th percentile), every rung of that road is cut in two: one loop
    down its whole length, so it stays a ladder.
  * *A closed sheet.* Regions are meshed one by one, so a gap between two
    pieces of the outline coverage goes to the neighbour sharing most edge
    with it first; afterwards vertices within 1.2 cm of each other are made
    one, sliver triangles are folded into the face beside them, and any hole
    still left is filled.
  * Heights are the survey grid's at every vertex, read between its nodes
    too. The 0.5 m survey read at an outline point on a retaining wall made
    faces standing nearly upright, on which a draped kerb and the check
    measuring it picked different faces.

  Every outline is an edge of the sheet, so each layer is the faces under it
  and the terrain object is what is left — the pieces share every vertex
  along the edges where they meet, and none lies on another.
  `layered.mesher: "triangles"` gives the old triangulation back, and is what
  a quad mesh that fails falls back to.
* **Water.** Water is meshed like the rest, with rungs across a ditch split
  at the middle: the middle line is the bed, `ditch_depth_m` (0.6 m) below the
  water level (one per connected water, see [Ground surfaces](#ground-surfaces)),
  and the bank keeps its surveyed height; no bank is cut steeper than 45°. A
  wider pond or canal gets a row for its bank and a flat bed inside. Over it
  lies `<Prefix>_Water`, one flat object at the water levels plus
  `water_level_offset_m`: a tide. Move it up or down in Blender for another.
  Trees are kept off water, roads and parking as in the survey style.
* **Harbours** (`src/harbour.py`). Water along a BGT quay wall (`scheiding`,
  type `kademuur`) gets a 0.3 m row instead of 2 m, so the ground drops from
  the quay top almost straight down. In front of that drop stands
  `<Prefix>_Kademuren`, a 35 cm concrete strip along every stretch of
  waterline within 1.5 m of a quay wall, its top flush with the quay and its
  face running 1.5 m under the water: from the water the quay wall, from above
  its coping. The barrier stage's 1.1 m wall on top of the quay is left out.
  Jetties and pontoons (`kunstwerkdeel_vlak`, `steiger` and plus-type
  `ponton`) are decks, `<Prefix>_Steigers` and `<Prefix>_Pontons`, the BGT's
  pieces of one jetty joined into one deck (each piece read its own height,
  and two met in a step with a crack down it), at the height the surface
  model saw on them (the lower quarter of its readings, so
  a moored boat does not count), else 1.2 m (jetty) or 0.5 m (pontoon) above
  the water. Mooring posts (`meerpaal`) are posts from 1.5 m under to 2.5 m
  above the water, `<Prefix>_Meerpalen`. Where a jetty deck stands over or
  within 3 m of the waterline, that waterline is quay too: the BGT draws the
  pier but no wall under it, and the bank otherwise sloped from the quay top
  down to the bed beside the deck, a grey wedge out of the water. Fender works
  (`waterinrichtingselement_lijn`, plus-type `remmingswerk` or `geleidewerk`)
  are the catwalks and dolphin rows at a ferry berth and the fender runs along
  a quay, which the BGT keeps only as lines with the water polygon under them.
  More than 2 m out from the waterline on average, a line is a 2 m catwalk on
  posts every 6 m, `<Prefix>_Loopbruggen`, at the surface model's height or
  2.5 m above the water; closer in, a 30 cm fender from under the water to the
  quay top, `<Prefix>_Remmingswerk`. Ships and cranes are not in the open data
  as 3D objects and are left to the modellers. Over Rotterdam
  Rijnhaven–Maashaven: 2.78 km of quay, 12 jetties, 17 pontoons and 43 mooring
  posts; at the Hoek van Holland ferry terminal 82 m of catwalk and 175 m of
  fender.
* **Railings** (`src/railings.py`). Along every side of a bridge deck that is
  over water, and down both sides of every catwalk, unless a BGT wall, fence
  or parapet already stands within 0.6 m of the edge: posts every 1.5 m, a
  handrail at 1.0 m and a knee rail at 0.5 m, in plain steel, about 17
  triangles a metre. Never across a deck's ends on the quays. Bridge railings
  are `Structures_railing`, catwalk railings `<Prefix>_Leuningen`; a stand-in
  for the railing of the place. `structures.railings` turns the bridge ones
  off.
* **Guardrails** (`src/guardrails.py`). The BGT draws a motorway's
  carriageways and verges and nothing standing between them, so a steel
  guardrail is laid wherever a motorway carriageway meets grass, open ground,
  water or a hard verge in the ground as built: along every verge and down
  both sides of the central reservation. Not where the carriageway meets
  another road part, a junction mouth or the lane into a service area: the
  rail stops 0.3 m short of that road rather than turning the corner into it.
  Not at a merge, where two motorway parts share an edge and there is none to
  follow. Not across the end of a carriageway that runs onto a bridge or into
  a tunnel, whose deck the structures stage builds: an edge with both ends
  within a metre of a part on a bridge (`relatieve_hoogteligging` not 0) and
  the carriageway beside it clear of that part is such an end. The metre has
  room to spare: in the ground as built the ends of the ramps at Princeville
  (below) stand up to 0.33 m off their bridge parts, and a test 10 cm out
  missed 11 of their 37 edges and ran a rail across them. A verge passing
  under a bridge keeps its rail. Not on a green gore the
  layered style kerbs as an island: its kerb is its edge. And not within
  0.6 m of a BGT wall, fence or noise barrier, which guards the edge already.
  The rail's face stands 0.5 m onto the verge and no closer than that to any
  carriageway, so in a gore the rails of its two sides stop where it narrows;
  where two would stand back to back closer than 0.4 m, at a gore's tip or in
  a reservation too narrow for two, the shorter one stops, and a strip under a
  metre wide has room for none. Nothing shorter than 10 m is kept, and a rail
  stops 0.23 m inside the edge of the area, so its posts are over the ground.
  The rail is Rijkswaterstaat's steel guardrail in broad terms: a W-profile
  beam 0.31 m tall and 8 cm deep, its top 0.75 m above the ground at its face,
  on 10 cm square posts. The posts stand every 2 m, a common spacing for Dutch
  steel guardrail (a stiffer stretch has them closer, a softer one further
  apart), which is also twice to every 4 m square of the ground, so the rail
  follows the verge as built. Each post goes 10 cm into the ground under the
  lowest of its corners: on a steep bank a post set by its middle had one
  corner in the air. The beam is folded once down its middle, which catches
  the light the way the W does: about 10 triangles a metre, against the
  railings' 17. Every height is read off the ground the layers are, so a post
  stands on the verge the model has, not on the survey under it. It is one
  object, `<Prefix>_Geleiderail`, in the railings' `Staal`, kept with the
  walls and fences in `Barriers` (the 3D view names that layer already); with
  collision shapes on, `Collision_Guardrails` is its proxy (see
  [Collision shapes](#collision-shapes)). `layered.guardrails` turns it off.
  Over 500 m of the A16 at Tankstation Hazeldonk (RD 109,980 to 110,480 by
  390,170 to 390,670), with 3.4 km of motorway outline, that is 3,229 m of
  rail in 10 runs, 2,144 m of it between two carriageways, on 1,630 posts:
  32,560 triangles, laid out in 0.2 s. Read back from Blender, every corner of
  every post foot is 0.10 to 0.14 m into the ground as built and the top of
  the rail's face 0.750 m above it; every post stands on grass. Over the
  A16/A58 interchange at Princeville (RD 110,200 to 110,700 by 393,050 to
  393,550), where the A58's ramps cross the A16 and the railway on five
  bridge parts, it is 5,125 m in 21 runs on 2,593 posts, none across a ramp's
  end; the A16's verges keep their rails under the ramps. There 274 m of
  motorway edge borders green gores kerbed as islands and has none, and on
  the banks up to the bridges a post goes in up to 0.43 m at its uphill corner.
* **Thin strips.** A ground polygon thinner than 0.5 m (twice its area over
  its perimeter) is folded into the neighbour it shares most outline with
  before meshing, land into land where it can. Three BGT borders 35–40 cm
  wide side by side along a Hoek van Holland quay were narrower than any row
  the mesher lays, and the rows beside them missed their outlines: 21 cracks.
* **Objects.** One object per layer: `<Prefix>_Snelweg`, `<Prefix>_Wegen`,
  `<Prefix>_Voetpaden` and `<Prefix>_Gras`, and `<Prefix>_Terrein` for the
  ground none of them covers — bare ground, the ground under buildings, the
  water beds. The ground used to be split east and west of the middle and by
  quadrant beyond, as the hand-made file is; the modellers divide it
  themselves. `<Prefix>` is `layered.prefix`, or the area name:
  `hazeldonk_west` gives `Hazeldonk_West_Terrein`.
* **Placing things.** Trees, lamp posts and fences stand on the top of the layer
  under them: the ground on a road or lawn, plus the curb generator's raise
  (13 cm) on a walkway.

In Blender nothing holds a layer onto anything: it is the ground. The
modifiers:

| Modifier | On | What it does |
| --- | --- | --- |
| *CurbGenerator* (`TerreinLab_Kerbs_CurbGenerator`) | walkways | lays the walkway flat, runs the modellers' curb generator on it, and puts every point of the result back at the walkway's height straight below it |
| *WeightedNormal* | walkways, terrain | the modellers' shading |
| *Triangulate for export* (render only) | every layer, terrain | cuts quads the same way in every piece, so they stay matched along their shared edges |

The curb generator was made for the modellers' level outlines: on outlines
that follow the survey it threw kerb spikes — over Hazeldonk 1.5% of kerb
vertices stood up to 0.65 m off the ground, at slopes and at outline points a
few centimetres lower than their neighbours — while the same outlines laid
flat came out exact. So the modifier wraps it: the walkway goes in flat, and
every point that comes out is raised by the height of the walkway as it came
in, read straight below it. Every one of the generator's inputs is passed
through, so the modifier panel shows them as it would show the generator's
own, and the generator itself is unchanged inside it. Move the outline, or
raise a vertex of it, and the kerb follows. (It used to read the height off
the terrain under the walkway; there is none now. Carrying each outline
point's own height onto the kerb points made from it was tried and missed by
up to half a metre on steep ground, where a kerb point lands away from the
outline point it came from.)

One more thing the generator needs: every outline vertex with exactly two
outline edges. Where two footpaths touch at a single corner — common in a
city — four meet at one vertex, the generator pairs the wrong two, and the kerb
flies off: over Utrecht by 800 m, elsewhere a metre into the air. Each
walkway object is built with such a vertex split, one copy per side in the same
place, so every outline is a simple loop.

The terrain carries the aerial, seams along its edges where a layer begins
and a vertex group per layer outline (`outline_road`, `outline_motorway`,
`outline_walkway`, `outline_grass`), so *Select* on a group picks every
walkway edge at once. Every ground piece carries its surveyed heights
(`measured_z`), which *Reset terrain to survey* puts back.

**Kerbs need Blender 5.** The curb generator is built on the *Bevel* geometry
node, which only Blender 5 has. `blender/assets/curb_generator.blend` holds it
as the modellers made it, taken unchanged from their Hazeldonk file, and it is
appended into every model, so every walkway in a file shares one node group: change
the group and every kerb changes; change a modifier's inputs and only that
object's do. The pip `bpy` module the launchers install is Blender 5.2, so
this works out of the box; an installed Blender 5 is used in preference. With
an older Blender the walkways are left flat and the run says so
(`walkways_have_kerbs` warns).

The layered style's settings, all optional:

| Setting | Default | What it does |
| --- | --- | --- |
| `layered.prefix` | the area name | what every ground object's name starts with |
| `layered.tolerance_m` | 0.5 | how far the survey may be off a road's rungs before a loop goes down its middle, and what the checks warn past |
| `layered.mesher` | `"quads"` | `"quads"`, or `"triangles"` for the triangulation refined to `tolerance_m` |
| `layered.square_m` | 4 | the size of the grid squares and the spacing of the points along every outline, the same everywhere, in whole survey grid squares; 8 has about a third of the faces over open ground |
| `layered.ditch_depth_m` | 0.6 | how far below its water level a ditch is cut |
| `layered.water_surface` | true | the flat water object over every BGT water |
| `layered.water_level_offset_m` | 0 | added to every water level: a tide (the survey's is whatever the tide was when it was flown) |
| `layered.harbour` | true | quay faces, jetties, pontoons, mooring posts, catwalks and fenders |
| `layered.guardrails` | true | steel guardrails along the motorway's verges and central reservation, `<Prefix>_Geleiderail` |
| `layered.outline_m` | 0.25 | how far a simplified outline may stray from the BGT's |
| `layered.kerbs` | true | the curb generator on every walkway |
| `layered.curb.*` | the modellers' own | width 0.3 m, height 0.15 m, slant, bevels, materials |

Over the Hazeldonk square kilometre the ground is 84,867 faces, 77% of them
quads, on 75,479 vertices: 4 m squares everywhere, turned 15° to the roads,
rungs across every road and loops along every verge, ditches cut in. 95% of
the 0.5 m survey is within 0.28 m of it, and 1.7% further than half a metre.
The version before, with squares up to 32 m in the far fields and loops cut
to the survey, was 61,810 faces, 58% quads, within 0.24 m on 95%: the even
grid costs faces in the open fields, which `square_m: 8` gives back (41,692
faces, a little further from the survey, and fewer of them quads). The modellers' own terrain has about 2,100
faces and is within 0.98 m on 95%. There are 4 layer objects; the walkways'
3,929 faces become 25,619 with their kerbs. The ground takes 23 seconds to
build; the whole run about 3½ minutes, most of it relighting the aerial, and
it passes all 65 checks.

Over the Hoek van Holland square the modellers commented on, the ground is
82,348 faces, 79% quads (63% before), turned 36° to the streets; 34,542 of
them are the beds of the harbour and the river, under the water object. With
`square_m: 8` it is 38,383.

Over Utrecht centre — quays, retaining walls, underpasses — the terrain was
100,657 faces. A kerb mitred round a sharp corner can stand a few centimetres
outside its band, which `layers_lie_on_the_terrain` allows up to 0.05% of the
vertices and 10 cm each. And where two footpaths touch at a single point on
the edge of the area, the kerb on the cut edge stands its own width outside
it, which the check also allows: a runaway kerb goes metres.

Not in it yet: the modellers' hand-made buildings. The buildings are
3DBAG's, as in the survey style.

## Terrain

**The ground's edges follow what is on it.** The waterline, the edge of every
road and the outline of every building are edges of the terrain mesh, each with
a row of quads running alongside it. Further from any outline the ground is the
regular grid of 2 m quads it was before: same squares, same places, same
surveyed heights.

Why: a plain grid ignores the outlines. A waterline falls wherever it falls
inside a square, so the bank of a pond came out as a staircase of squares and
there was nothing to select. Editing terrain in Blender works from the outlines
inward: select the waterline, move or scale it, and proportional editing takes
the bank with it; the row beside it is the next loop in. That needs the
outlines to be edges and the faces laid out along them.

How it is built (`src/terrain_mesh.py`):

* **Regions.** At-grade road surfaces, water and building footprints cut the
  area into regions of one kind each; where they overlap, road wins over
  water, and water over building. Corners of different outlines closer than
  40 cm become one corner, and an edge two regions share has the same
  vertices on both sides, so the sheet has no cracks.
* **Rows.** Every region you can see — land and water — gets a row of quads
  along its outline, one grid spacing wide, with a rung from every station on
  the outline: every corner, and no more than 3.2 m apart between them.
* **Ladders.** A region too narrow for a row on each side — a front garden, a
  verge, a ditch — gets rungs straight across instead, and the point each one
  lands on is added to the outline on both sides of it. Under water the rungs
  are split at their middle and the middles joined: a ditch has a bed line
  down its centre.
* **Grid.** Inside the rows, the regular grid carries on.
* **Fill.** Between the rows and the grid's staircase edge a strip is
  triangulated, and the triangles are paired back into quads wherever the quad
  comes out well shaped. Each rung is carried on to the grid node in front of
  it first, which cuts the strip into small cells.
* **Hidden ground.** Under roads and buildings nobody sees the ground, so it
  gets no rows: its outline is an edge loop and the inside is grid and fill.

Heights come from the survey, as before. The waterline is never below the
water's own level, and everything under water lies at least `water_depth_m`
below that level, so the inside row of a pond is its bank, from the waterline
down to the bed. (Water beds used to be sunk square by square in Blender,
which left a stepped wall at the edge of every pond.)

Over the Bronbeek square kilometre:

| | |
| --- | --- |
| Faces | 263,601, 84% quads |
| Grid squares / row faces / fill faces | 149,179 / 39,635 / 74,787 |
| Quads among the row faces / fill faces | 81% / 54% |
| Edges along a waterline / road edge / building outline | 4,422 / 24,382 / 13,563 |
| Edges along the inner side of a row / a bed | 22,168 / 2,153 |
| Cracks, edges shared by three faces, faces upside down | 0 / 0 / 0 |
| Faces with an angle under 10° | 1,502 (0.6%) |
| Within the survey, 95% / 99% of the ground | 12 cm / 26 cm (the plain grid: 11 cm / 23 cm) |
| Time | 55 s |

The cracks line is a check every run makes (`terrain_is_one_closed_sheet`):
the faces have to cover the bbox exactly once.

The mesh gives up a little accuracy to the plain grid, and it gives it up
along the outlines: a row sits exactly on a bank, a kerb or the foot of a wall,
where any surface of flat faces is furthest from the survey. Two things keep
that small. The vertices off the grid read the 0.5 m survey directly rather
than the grid between its nodes. And the waterline keeps its surveyed height,
raised to the water level only where the survey dips below it; pinned to the
water level, it pulled every bank top down with it and left half the ground
beside the water more than 25 cm off.

**Choosing what it follows.** `terrain.feature_edges` names the outlines:
`["water", "roads", "buildings"]` by default, `["water"]` for only the
waterlines, `[]` for the plain grid. If the outlines ever defeat the geometry,
the run carries on with the plain grid and says so in a warning rather than
failing at the terrain.

**Every quad is split one way, on purpose.** A quad of four heights is not
flat, so the renderer draws it as two triangles, and which diagonal it picks
moves the ground by up to half the height difference across it. Blender's export
triangulation is told to cut every quad from its first corner to its third
(*Triangulate* modifier, quad method *Fixed*, applied on export only), and the
pipeline writes each quad so that is the diagonal it wants: south-west to
north-east on a grid square, and elsewhere the diagonal whose middle lies
nearer the surveyed ground. Every height the pipeline places something at — a
tree, a lamppost, a car — is read off those same triangles. Bilinear
interpolation, which is what "the height between grid points" usually means, is
a surface nothing draws; on lumpy ground it put posts a few centimetres in the
air or under it.

**How the grid is made.** AHN is fetched at 0.5 m with a few cells of margin,
block-averaged onto the grid — every output vertex is the mean of the surveyed
pixels around it, not one pixel picked by interpolation — and the holes where
buildings, water and dense vegetation stood are filled inward from their edges.
Only the filled cells are smoothed; surveyed ground is left as surveyed. A
`grid_m` that does not divide the area is rounded so the spacing is never
coarser than asked. The mesh's other vertices take their height from that grid.

**Two surfaces, kept apart.** `TerrainResult` carries both:

| | Reads | Used by |
|---|---|---|
| `terrain.sample` | the surveyed grid, before anything was moved | road heights, buildings, bridges, and every check |
| `terrain.surface` | what the model draws: the terrain mesh's triangles, or the road surface where there is road | trees, furniture, barriers, rails, cars |

The split follows whether a thing is **measured against the ground** or
**placed on what you see**. A road takes its height from the survey, because
that is where the street is; a lamppost on the footway stands on the footway,
kerb included, because that is what it visibly stands on. The checks read the
survey, because a check against the drawn surface measures the model against
itself.

### The ground gets out of the road's way

The road surface follows the survey; the ground can only bend at its own
vertices. Between them a 2 m quad bulges up through a carriageway on any
cambered street, and a 0.10 m kerb leaves the ground showing through the
footway. So once the terrain mesh is built, it is **fitted under the roads**:
every road vertex, edge midpoint and cell under the road is a point the ground
must stay at least 5 cm below, and the vertices around each offending point
are lowered just enough to clear it. Because the road edge is now an edge of
the terrain too, the ground beside a road no longer shares a face with the
ground under it, and the verge stays where it was surveyed.

Two details decide whether this looks right:

* **Lower the vertices you cannot see.** A point under a road lies in a
  triangle with three corners, and the drop needed can be shared between them
  in any proportion. Grid vertices under the road surface are hidden; the ones
  beside it are the verge, the bank, the field. The correction is weighted four
  to one towards hidden vertices, so the road sinks into the ground rather than
  the ground around the road sinking. On Bronbeek the fit moves 48,452
  vertices, by 3 cm at the median. 210 drop by more than half a metre, and the
  60 of those in sight are all on the road edge itself, where the ground meets
  a road lying lower than the survey beside it. Beyond the road edge not one
  does; on the plain grid, 38 verge and bank vertices did, and about 1,800
  before this weighting and the car-park grid below.
* **The survey is kept.** The lowered mesh is what is drawn; the surveyed
  heights are kept beside it (`terrain_mesh.npz`, and the `measured_z`
  attribute on the Blender terrain), so the fit can be undone and redone after
  someone moves a road. That is what the
  [TerreinLab tools](#opening-it-in-blender) do.

Big flat surfaces need help too: a car park or a square has no street to cut it
across and would otherwise be one face tens of metres wide, flat between its
corners while the ground under it is not. Any road face over 150 m² is cut into
a 4 m grid of its own, so it follows the ground in rows and columns instead of
forcing the terrain metres down to get under it.

A small face can have the same trouble when its corners straddle a height step:
a ramp down to a canal wharf, or the wharf path along a quay wall. Flat between
its corners, it hangs metres below the top of the wall or rises over the wharf.
Any face at grade that the ground strays more than 50 cm from is cut along the
terrain's own grid lines, so each piece lies within one cell of the height model
and follows it as closely as anything flat can. Over Utrecht's Vismarkt that is
91 faces at standard detail; before, 32 road triangles there were up to 2.1 m
off the ground and `road_surface_follows_terrain` failed.

The checks `terrain_stays_under_roads` and `road_surface_follows_terrain` hold
both ends of this: no ground through any road, and no road far off the survey.

## Ground surfaces

**Water is the reason this exists.** Lidar does not reflect off water, so the AHN
DTM is 73% empty over a canal against 47% on land, and the returns that do come
back scatter over six metres. The gap filler then interpolates inward from the
banks, which turns every canal into a bulge with lumps in it — the worst
geometric artifact in the model. The BGT has the outlines, so the pipeline stops
guessing: it takes a level per water from the lidar that did return inside
it, sinks the bed below that, and lays a flat surface on top.

**One level per connected water, from the surface model.** The bare-earth DTM
filters the water's own returns out and keeps mostly what stands on the water:
over Rotterdam's Maashaven 1,236 returns, nearly all moored barges, against
41,022 in the surface model (DSM), which has the water too. So the levels are
read from the DSM where the run has one (it is fetched before the surfaces
whenever water is on). And the BGT cuts a river or a harbour into polygons
where its managers' plots meet, lines that mean nothing on the water: read one
by one, the Nieuwe Waterweg at Hoek van Holland came out at −0.99 to +3.55 m
NAP, the high one off the ferries moored in it. Now every polygon's returns
are pooled with those of the polygons it touches (within 1 m), and the level
is the median of the returns from 10 cm below to 15 cm above the lowest (the
1st percentile, `SHIP_CLEARANCE_M`): the water's own returns lie within a few
centimetres of each other, and anything higher is a boat, a pontoon or a
jetty. The band was a metre at first, which let Rotterdam's barges out but
kept Amsterdam's canal boats in: they lie 0.7 m above the water and outnumber
its returns ten to one, and every canal read +0.32 m NAP against a real −0.40.
Now the main canals read −0.39 and −0.41 m. A tidal water flown at two tides
has two water peaks, and this reads the lower.
A water with too few returns takes its banks' level; one with neither takes
the median of the others, rather than being dropped and drawn as land (38,800
m² of the Nieuwe Waterweg was). `one_level_per_water` checks it.

The water runs on under bridges. The BGT cuts every pier of a canal bridge
(`overbruggingsdeel` on the ground) out of the water polygon, 38 holes and 4
notches over Amsterdam's centre; taken for islands, they were filled in from
the quays at about +1 m with the bed sloping up to them, a mound across the
canal under every bridge. A ground-level bridge part with at least half its
outline on water is now water itself, so the surface and the cut bed pass
under the deck; the piers are still built, from the deck down to the bed (the
ground as the layered style builds it, not the survey, so they reach into the
water). A bridge's parapets are BGT walls up at the deck's level
(`relatieve_hoogteligging` 1); stood on the ground under them they were walls
from the canal bed to above the deck, closing the canal under every bridge.
A barrier above the ground now stands on the deck under it, read as the
bridges stage reads the decks (`structures.deck_surface`). A deck's height is
the median of the surface model over a 0.5 m grid inside it; read at the
corners of its triangles, a small canal bridge was six points on its edges,
water and quay, and two of Amsterdam's 43 decks fell back to a guess five
metres over the canal.

Separate waters keep separate levels: levels across one square kilometre of
Utrecht run from −0.57 to 1.00 m NAP, so a single bed taken from the median
would sit above the surface of the lowest canal and poke straight through it.
Waters that meet only outside the area can differ too: Rotterdam's Maashaven
reads +0.26 m and the Nieuwe Maas +0.86 m, flown on different days at
different tides. A lock between two waters whose polygons touch is the one
place this is wrong: both sides get one level. A check asserts every bed
clears its own surface.

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

## Roads

The roads are built to be edited: **a ladder of quads down every street**, a
rung across the whole street every three metres or so, kerb to kerb, with no
vertex anywhere in the middle of a carriageway. Every street is its own object,
named after it.

```
Roads (empty)
├── Road_Velperweg                a street, all its sections between junctions
├── Junction_<street names>_<id>  where they meet, named after them
├── Path_<name>                   a named footpath or cycle path
├── Bridge_<name>                 the part of a street on a bridge deck
├── Lanes_004, Paths_017          estate roads and park paths the national
│                                 road network does not have
└── Paving                        squares and car parks belonging to no street
```

### Where the rungs come from

A rung is a cross-section of the street: a line square to its centreline, from
one outer kerb to the other. So the first thing needed is the centreline, and
the BGT only has outlines.

**The national road network has them.** NWB-Wegen (Rijkswaterstaat, via PDOK,
CC0) carries a centreline for every public road in the country, split into
sections at every junction, with the street name and the id of the junction at
each end. Its geometry is derived from the BGT, so it lies on the BGT roads.
That gives each street a spine, a name, and its junctions.

**Then the stations.** Along each centreline the cutter places a station at
every kerb corner the street owns — a parking bay, a driveway apron, the start
of a bus stop — and fills between them at three metres. A station at a corner
matters: a corner is a vertex no straightening can remove, and without a rung
through it the face beside it is a pentagon.

**Each rung stops where the street does.** The cross-section is clipped to the
road surface, and walks outward from the centreline across carriageway, cycle
path and footway — across a verge of up to 6 m too, so the footway beyond the
grass is cut with the street — until it reaches road that belongs to a
*different* street, which is judged by which centreline is nearer. Without that,
two parallel streets with touching footways cut each other's footways twice.

**At a junction the streets stand back.** Each street's stations stop short of a
junction by the reach of the streets crossing it there, so no rung runs into the
corner footways, and what is left between the streets is the junction: its own
object, named after the streets that meet in it.

**No two rungs may cross.** Where they would — a fork at a narrow angle, where
both branches share one carriageway for a while — the one nearer a junction is
dropped. A rung that has run round a sharp bend and up the next leg of the same
street is dropped too: it is no longer square to anything.

**Where there is no centreline, the road draws its own.** NWB has public roads
only. A private estate's lanes, the paths through a park and a hospital's
service roads have no centreline, and before this they came out as single faces
with hundreds of corners. Whatever road the national centrelines leave without a
rung is skeletonised: the medial axis of the leftover surface, pruned of the
spurs every corner grows, smoothed, and then cut exactly like a street. On
Bronbeek 173 national sections and 418 drawn centrelines give 6,671 rungs.

### What a street is made of

The rungs and the BGT outlines together cut the surface into faces. The
outlines are surveyed to the centimetre and wobble by a few centimetres between
rungs, and every wobble kept is one more corner on a face, so each kerb run is
straightened between the rungs it meets — within 10 cm, and once per kerb, so
the two surfaces sharing it stay welded. A face over 150 m² with no street to cut
it (a car park, a square) is cut into a 4 m grid, and a face whose corners
straddle a step in the ground (a quay wall) is cut along the terrain grid. On
Bronbeek:

| | |
| --- | --- |
| Surface faces | 17,652 |
| Quads | 78% of faces |
| Surface vertices off a surveyed kerb or outline | 0 |
| Kerb faces (the vertical step at a footway) | 4,831 |
| Street objects / junction objects | 94 / 92 |

The rest are faces with five or more corners, mostly at junctions and where a
kerb meets a rung at a T, which no cut can avoid.

**Kerbs are real.** Footways stand `surfaces.kerb_m` (10 cm) above the
carriageway, with a vertical kerb face between them in its own material,
`M_road_kerb`. The open edge of the network — where a road runs off the area, or
a path ends in grass — gets a 35 cm skirt down into the ground so nothing shows
daylight at a grazing angle.

**Heights come from the survey**, per vertex: the AHN grid under it plus
`surfaces.road_lift_m` (3 cm), footways plus the kerb. Then the terrain is
fitted under the roads (see [Terrain](#the-ground-gets-out-of-the-roads-way)),
rather than the roads being floated above the terrain to clear it.

**Two UV maps.** `UVMap` is the aerial photo under the same top-down
projection the terrain uses, so the surface looks exactly like the photo,
markings and crossings included. `RoadUV` runs in metres: on a street, U across
it from the centreline and V along it, so a tiling asphalt or brick texture, or
a lane marking, follows the curve; on a junction or paving, plain metres east
and north.

Every road object carries a `kind` custom property (`street`, `junction`,
`paving`) that survives into Unity, and every face carries `road_part`
(0 surface, 1 kerb, 2 skirt) as an attribute.

**A path crossing water does not dive into it.** Lidar does not reflect off
water, so the bare-earth grid has a hole over every body and the gap filler
interpolates across it — and the result dips: over the Gelderland square the
filled ground mid-water sits 1.45 m below its own bank. A footpath draped on
that follows it down and vanishes under the surface it is meant to cross. Any
road over a body is now held at that body's own level plus
`surfaces.road_water_clearance_m`, per body rather than shared, because levels
across one area run to metres. Held, not pinned: a real bridge rides higher and
is left alone.

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

Tunnels (`relatieve_hoogteligging` below zero) are left out of the road surface
— a tunnel drawn on the surface is simply wrong — and get their own geometry
(see [Bridges and tunnels](#bridges-and-tunnels)). Roads on a bridge ride its
deck.

## Street detail

None of this is structurally important, which is exactly the point. A street
with nothing on it reads as a model; the same street with lampposts along the
kerb, a wall between the plots and a hedge behind it reads as a place. Everything
here shares one 8-patch atlas and one material, so a few thousand objects cost
one draw call.

The BGT registers far more of it than the model used to draw. These are real
counts for one square kilometre of Utrecht centre:

| BGT collection | count | drawn as |
|---|---|---|
| `paal` | 1,996 | lampposts 1058, bollards 918, sign posts 16, **flagpole 1** |
| `scheiding_vlak` | 1,450 | walls 1330, quay walls 120 |
| `scheiding_lijn` | 1,563 | fences 1174, walls 332, quay walls 36, noise barriers 4 |
| `straatmeubilair` | 217 | benches 148, **playgrounds 50, art 8, ad columns 6, shelters 3, monument 1, picnic table 1** |
| `gebouwinstallatie` | 114 | **entrance steps 55, awnings 53, stoops 6** |
| `overigbouwwerk` | 82 | **transformer housings 6, canopies 4, open sheds 2** |
| `kast` | 69 | **electrical cabinets 51** |
| `bak` | 41 | **waste collection points 41** |
| `mast` | 40 | **tram catenary masts 40** |
| `vegetatieobject_lijn` | 34 | **hedges 34** |

Bold is what was being fetched and thrown away. The walls and fences are the
bulk of it — three thousand objects that divide one plot from the next, without
which a street is a row of buildings standing in a field. The quay walls matter
more still: an Utrecht canal without its wall is a trench.

**Nothing here carries a height.** The BGT surveys where a wall is and never how
tall, so heights come from a table by type in `barriers.LINE_STYLES` and
`AREA_STYLES`, overridable per type under `barriers.heights`. That is an
assumption and it is kept in one place rather than spread through the geometry.
AHN cannot help: a fence is thinner than the half-metre height grid can see, and
a wall against a building takes the building's height — the same trap the tree
heights had to be dug out of.

Three things about the geometry are worth knowing:

- **A barrier is a closed prism, never a plane.** A line is swept: offset either
  side by half its thickness, both edges draped, top and both ends closed. A
  fence is 6 cm thick and a wall 24, and neither is zero — a zero-thickness wall
  is a hole from one side in anything with backface culling on, which is every
  engine.
- **Vertices go where the ground bends, not every two metres.** Cutting each
  line at a fixed step holds straight lines straight at full price: 203,020
  triangles against 88,324 for the same walls followed to 5 cm. The survey's own
  vertices are anchors and always survive, because those are the shape of the
  fence rather than the shape of the ground under it.
- **A polygon is cut up before its ground is sampled.** A survey draws a 30 m
  wall as a four-vertex rectangle, and sampling only at the corners leaves the
  base a straight chord over whatever the ground does in between: one ring edge
  in five departed from its chord by more than the 15 cm embed, and the worst by
  3.9 m. The swept lines had always been cut up first; polygons were the ones
  left out.
- **A polygon's base is sampled per vertex.** One height for a whole outline
  fails exactly where these objects live — a quay wall runs along a canal bank,
  the steepest ground in a Dutch city — and on a test surface a single median
  buried one end of a wall by 0.65 m. Awnings are the exception: they hang off a
  facade at one height, so they stay flat — taken from the *highest* ground
  under their footprint, not the median. The median looks like the fair choice
  and is not: an awning whose footprint straddles a slope ends up half buried,
  by 1.43 m on the Gelderland square. Too high at one end is a thing nobody
  notices; underground is not.
- **A fence is see-through.** It gets its own alpha-clipped material and a
  palisade texture that tiles along the run — about two thirds open, so the
  uprights read as uprights. Drawn as a solid slab a railing is a hoarding, and
  a fence and a hedge come out the same object. `barriers.fence_tile_m` sets how
  long one tile is; the texture maps once over the height, so every fence gets
  its top rail wherever that height falls.

Over the demo square kilometre that comes to 2,799 objects and 88,329 triangles,
with no non-manifold edges and no degenerate faces. They arrive as one object per
type — `Barriers_fence`, `Barriers_quay_wall`, `Barriers_hedge` and so on — so a
designer can hide the fences to see the plots, or swap the hedge material,
without selecting anything by hand.

**A word on why this is not Google Earth.** Google's 3D is photogrammetry: a
mesh reconstructed automatically from oblique imagery, which has hedges and
flagpoles in it because it does not know what anything *is*. That is also why it
cannot be used here — no semantics, no separable objects, trees fused into
rooflines, topology far worse than anything in this pipeline, and a licence that
does not permit extraction. What is above is the other route to the same
density: every object is a known thing, in a named group, that a level designer
can take apart.

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
- **A curve is cut like a straight.** The BGT draws a curve as a vertex every
  few centimetres, and the track was cut at every one: a comb of hundreds of
  edges beside straights cut every 4 m. Each line is simplified to within
  3 cm first and then cut every 4 m at most, and each step of a bed or a
  rail is one quad, the strip one piece. Over Hoek van Holland the metro's
  ballast went from 8,078 triangles to 667 quads.
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

### Finding the trees nobody registered

The BGT tree register is a municipal asset list, not a survey of vegetation. It
has the street trees the council maintains and knows nothing about the tree in
someone's garden. Over the demo square kilometre **45% of the canopy above 2.5 m
is nowhere near a registered tree** — and that is the flattering case, because
the register is at its best in a city centre.

So the canopy model is now read as a source of trees rather than only as a
lookup at 1,489 points. Over the same square: **1,489 registered, 7,211 found,
8,700 in the model**, in about three seconds.

**Colour is not the way to do it, in this country.** The obvious approach is
Google Earth's apparent one — find the parts that are tall and green. PDOK
publishes a real colour-infrared ortho at 8 cm, so a proper NDVI is available
rather than a guess at greenness, and it does not work here:

| | NDVI |
|---|---|
| park trees | **+0.075** |
| water | −0.005 |
| big roof | −0.181 |
| road | −0.184 |

Summer foliage would read +0.6. Dutch national orthos are flown deliberately
**leaf-off** in early spring, so that the ground and the buildings are visible
rather than hidden under canopy — which makes them excellent for everything else
this pipeline does and useless for finding trees by colour. A quarter of an index
point between a tree and a road is not a detector. Height is the signal.

### The flown sun, and taking it back out

A Dutch ortho is shot on one morning. Over Bronbeek the pipeline fits that
morning at **140 degrees azimuth, 25 degrees up** — a mid-morning late-winter
flight at 52 degrees north — and 29% of the photo sits below a luminance of
85. Those shadows are painted onto every roof and every square metre of
ground, at one time of one day. An engine lighting the model at any other hour
then has two suns disagreeing, and for a drone sim the hour is the user's to
set. `aerial.deshadow` takes the flown one out.

**The shadow is computed from the thing that casts it.** That is the whole
reason this can be trusted. A detector working from pixels alone cannot tell a
dark roof from a dark roof's shadow, and will brighten slate, asphalt and
water. This pipeline already knows where the buildings and the trees are in
three dimensions, so it marches a ray toward the sun across a height raster
built from the terrain, the 3DBAG roofs and the detected canopies. Heights are
absolute NAP rather than above-ground, so a hillside shades its own far slope —
Arnhem is on a moraine and Bronbeek is on the side of it.

**The sun is fitted, not looked up.** The WMS publishes no capture time, so a
few hundred trial shadows are cast across a grid of sun positions and the one
whose shadows best explain the dark pixels wins. Two resolutions: the sweep is
pooled to 4 m because recovering a *direction* needs no more, and the
refinement runs at 1 m because pooling by maximum fattens every caster, which
lengthens its shadow, which the fit answers by choosing a sun 2.5 to 4.5
degrees too high. The best four coarse candidates are refined rather than the
best one — a low sun over regular streets merges its shadows into a sheet and
flattens the score surface into basins a local refinement cannot climb out of,
and on a test grid of terraces at 14 degrees it settled 20 degrees off.

The fit is worth trusting over intuition here. A crude correlation of dark
against bright pixels on this same photo says the sun is in the south-west;
looking at the Bronbeek building, its shadow plainly falls north-west, so the
sun is south-east and the fit is right. The correlation was reading the tree
canopy, not the buildings. Adding 11,921 detected trees to the height raster
moves the fitted sun not at all — 140/25 either way — which is the better
check.

**The gain is measured off the photo, per land-cover class, per channel.** How
much darker a shadow is depends on what is under it: grass in shade against
grass in sun is a fair comparison, grass in shade against asphalt in sun is
not, and one global gain is exactly that comparison. Per *channel* because a
shadow is lit by sky alone, so it loses more red than blue, and three separate
gains undo the cast without anyone naming a colour temperature.

Two guards:

* **Capped at 3x.** Deep shade holds no detail to recover, so past that the
  gain amplifies sensor noise and JPEG blocks.
* **The photo has a veto.** A pixel is only brightened to the extent it
  actually looks shadowed, measured against its own class's lit and shadowed
  medians. Where the geometry and the photo disagree — a tree the detector
  missed, a building 3DBAG has a metre out of place, a car parked since the
  flight — nothing gets stained. Tested with roofs painted *darker* than the
  shadows around them: they move by under 15%.

Measured over Bronbeek with buildings and trees, 41.6 s:

| | before | after |
|---|---|---|
| shadowed ground, luminance | 96.0 | **115.4** (sunlit: 123.4) |
| below luminance 85 | 29.4% | **15.3%** |
| contrast, p90 − p10 | 97.4 | 80.7 |

It is a mitigation, not a cure, and the limit is information rather than
effort. `aerial.deshadow: 0` leaves the photo as flown.

**The roofs get the ground's correction, not a fit of their own.** The roof
atlas (see [Roofs](#roofs)) is the same flight cut into rectangles, with no
ground round its roofs to measure a shadow against. So nothing is fitted on it:
the sun, the cast shadow and the gains per class and channel are the ones
fitted on the ground, read at each roof pixel's own RD position, with the same
veto from the photo. The shadow is feathered at its own 0.5 m scale and read
bilinearly, which comes to the same metre or so of ramp the ground gets. Over
Utrecht centre a roof's mean luminance in the atlas and the same roof's in the
aerial differ by 0.3 levels at the median and 2.1 at the 90th percentile, over
338 roofs and terraces. Fetching the whole area at the roofs' resolution and
relighting that instead would have cost the ground the same pixels again,
which is what the atlas is there to avoid.

The atlas does not get the ground's other two passes. The grain is per ground
surface, and a roof has none. The canopy occlusion is drawn under trees, which
in a street mostly stand lower than the roof next to them, so on a roof it
would darken the wrong side of the crown. One limit carries over: the shadow is
known at 0.5 m, so at 10 cm a thin dark rim can stay along a hard shadow edge
on a roof, as it can at 25 cm on the ground.

### The leaf-off photo, under the leafy trees

That same decision leaves the model disagreeing with itself. The ground under a
tree is photographed in full sun through bare branches, and then a leafy canopy
is stood over it, so the eye gets a tree that is not touching the ground. The
missing term is **ambient occlusion**, and `trees.ground_shade` bakes it into
the aerial: the photo is darkened under each canopy, to 65% of its brightness
by default.

**Occlusion, not a sun shadow, and the distinction is the whole design.** AO is
how much sky a patch of ground can see, so it is right at any hour and *adds*
to a real-time shadow. Baking a sun shadow at one angle would give every scene
lit from another angle two shadows pointing different ways, which is worse than
none — and for a drone sim the time of day is the user's to set.

Three details:

* **It comes from the trees that are drawn**, not from the canopy height model
  they were found in, so the mark on the ground and the thing above it cannot
  disagree. Same reasoning as draping on the terrain mesh rather than the grid.
* **Overlapping crowns take the strongest cover, not the sum.** A copse is one
  canopy with one sky above it; adding four crowns at 0.35 each would take the
  ground past black.
* **The falloff is a smoothstep over 0.7 of the crown radius.** Picked by
  looking at it: at 0.35 the mark is a disk, and a disk on grass reads as paint
  rather than shade; at 1.1 an avenue merges into one band and the trees above
  stop lining up with anything. Smoothstep rather than a linear ramp because a
  linear falloff has the same slope right up to where it stops, leaving a
  visible ring.

Measured over the Gelderland square kilometre — 13,043 trees — it takes 3.3 s
at 4096 px and 11.2 s at 8192 px, and shades 77% of the photo, which is what a
wooded kilometre looks like. `trees.ground_shade: 0` leaves the aerial alone.

**Difference against the filled ground, not the raw download.** The bare-earth
DTM arrives **51% measured** over this square, and its holes are exactly under
dense canopy, because dense canopy is what stops a ground return reaching the
lidar. Subtracting one raster from the other therefore leaves the canopy model
undefined in the middle of a wood, which is the one place it is most wanted.
Using the terrain surface the pipeline has already filled and smoothed takes
coverage from 51% to 96%, costs nothing, and has the trees standing on the same
ground everything else is draped on.

Four filters turn canopy into trees, and each is there because of something that
went wrong without it:

- **Off the building footprints, and 2 m clear of the eaves.** A roof overhangs
  its own footprint; without the clearance the eaves come out as a line of trees
  along every terrace.
- **Extra clearance the taller the candidate.** Measured: 65% of detections above
  25 m sat within 5 m of a building against 32% of those under 10 m, and the
  tallest were one tight cluster around a tower the footprints do not cover. The
  taller a thing is, the likelier it is part of the building it leans on.
- **Rough, not smooth.** A crown is a mess of leaves at half-metre scale where a
  made thing is a surface: roughness at a registered tree runs to a median of
  3.1 m against 0.67 m on a roof. This cleans up what the footprint mask misses;
  it does not replace it, because a roof *edge* is as rough as any crown.
- **One seed per crown.** Every cell of a crown clears the height threshold, so
  the peaks are found first and each tree then reserves its own crown. Registered
  trees reserve theirs before any seed is taken — with a token three-metre
  reservation instead, a quarter of the detections landed inside a tree the model
  already had.

**A found tree is not allowed to stand on a carriageway.** A tall rough thing
on a road is a lorry, a bus, scaffolding or a crane far more often than a tree:
over a square kilometre of Gelderland **9.5% of everything the canopy model
found was standing on a drivable surface**. Avenues do exist and this costs the
odd real one — a street tree whose crown leans far enough over the road for its
peak to land there — which is a good trade at ten to one. It applies only to
what the detector found: where the BGT has surveyed a trunk, the trunk is
there, whatever surface the survey says it stands on. Footpaths are in the set too: they
were left out at first to protect the tree in a pit on a pedestrianised street,
but a surveyed tree is never filtered by this at all, so the exclusion protected
nothing and cost a great deal — footpath is 305 of the 871 road parts over that
square, and in a park every winding path through the lawns is one.
`trees.detect_off_surfaces` names the classes.

**On a yard, a found tree needs a crown, not a flat top.** Lorry parks, depot
yards and big car parks are not parking bays in the BGT but *erf* or paving, so
the road mask misses them, and a parked trailer is 4 m tall and rough at its
edges. Over the Hazeldonk square 991 found trees stood on yards, 91% of them
grey in the photo and a median 4.06 m tall: a lorry park full of them. Masking
yards outright would take the gardens and the woodland the BGT also files as
*erf*, so on those surfaces a peak is refused only when its top is flat: more
than 15% of the 4.5 m window around it within 25 cm of the peak. A round crown
runs about 1% of that window; a trailer, container or van roof 20 to 60%.
`trees.detect_flat_top_on` names the surfaces (`paved` by default), and
`detect_flat_top_share`, `detect_flat_top_tolerance_m` and
`detect_flat_top_window_m` set the test.

**Detected trees arrive under their own parent.** The model gets
`Trees_registered` and `Trees_detected`, each holding one object per tree
(`Tree_00001`, `Tree_d00001`), sharing one material. The detector is good rather
than perfect — scaffolding, roof plant and the odd parked lorry will get through
— and a bad patch has to be deletable as a group, or a tree at a time, rather
than cut out of a merged mesh. Turn
it off with `trees.detect`, and the thresholds are all under `trees.detect_*`.

**Point clouds would not fix this.** They would improve crown *shape* and let the
understory be separated from the canopy, but the 0.5 m surface model already
resolves crowns well enough to place a tree, and AHN has no vegetation class at
all — vegetation sits in "unclassified", so the work would be the same
height-above-ground reasoning at higher resolution and considerably more cost.

### What a tree looks like

A tree used to be a sphere on a prism, which reads as a lollipop from any
distance you can still see the trunk at, and as eight identical lollipops in a
row from closer. Three things fix that, and together they cost four triangles a
tree — 42 to 46 on a real height mix:

- **The crown is pushed out of round** by two frequencies of noise, so it lobes
  rather than merely leans.
- **A mature tree gets a second lobe**, offset sideways and vertically. One
  deformed ellipsoid is still an ellipsoid — push it about as much as you like
  and the outline stays convex, which is what makes a lollipop a lollipop. The
  billowy mass photogrammetry gives you needs more than one lump, and stacking
  the second lobe on the axis is not enough: from above that is still one round
  crown.
- **The trunk tapers, flares at the foot and leans a little**, because a
  cylinder meets the ground like a pipe.

Everything is broadleaf. An earlier version made a third of them conifers, which
is a fair description of a Dutch verge and the wrong thing for these models — a
wood of spikes reads as a Christmas tree farm.

All of it is seeded from the tree's **own position** rather than from a running
generator, so the same tree is the same shape every run. Turning detection off,
or rebuilding one group and not the other, must not reshape the trees that were
already there — and a sequential generator cannot promise that, because the
order it is drawn in changes with the membership.

Detail follows size through the lobe count rather than through a coarser
primitive: one lobe under fourteen metres, two above. A cheaper ball was tried
for saplings and only ever looked like a gem on a stick. The finished cluster is
then scaled to exactly the crown it was asked for — lobes overlap, so their
union is smaller than any one of them implies, and without the fit every tree
came out a sixth shorter than the canopy model measured it to be.

Over a wooded square kilometre that comes to **63 triangles a tree**, against 42
for the sphere-on-a-prism it replaced.

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

Three materials, by default.

**`M_aerial`** carries `aerial.png` with a top-down planar UV taken from world XY
normalised to the aerial bbox. It goes on the terrain.

**`M_roof`** carries `roofs_00.png`, the same flight over the roofs only, at a
finer resolution (see [Roofs](#roofs)). Each roof triangle's UV is top-down too,
into its building's rectangle on the page, so a roof still lines up with the
ground beside it. With more than one page the materials are `M_roof_00`,
`M_roof_01` and so on, one per page, and a building's roof is always on one
page. `buildings.roof_cm: 0`, and the Draft level, put the roofs back on
`M_aerial` as before, with the planar UV the terrain has.

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

**And the era is not the whole story, because it was also the whole texture.**
Era deciding the character of a wall is right. Era deciding its *pixels* meant
every building of an age was the same building: a terrace of twelve 1890s
houses came out identical along its whole length, which is exactly what reads
as generated however good one house looks.

So each era now gets several looks. A look varies the wall colour, the trim
colour, the window count across, the window proportions and the bay width.
Measured, two looks of one era differ by 19–52 mean absolute levels where two
completely *different* eras differ by 50 — these are different buildings, not
tints.

**The looks are allocated, not spread evenly, because buildings are not.**
Over the Bronbeek square kilometre: 1,322 buildings, **73% of them built
between 1920 and 1945**, 580 of those between 1930 and 1934 alone. It is one
interwar estate built in two waves. Giving every era the same count there
spends a dozen textures on sixteen pre-1920 houses that cannot repeat, and
still leaves the interbellum era with eighty-one buildings per wall.

Each era instead gets one look per `facade.buildings_per_look` buildings *of
that era*, capped at `facade.looks_per_era`. On Bronbeek that comes out
`historic 1, interbellum 12, postwar 3, modern 1, contemporary 1` — 20 wall
textures against the 62 a flat twelve would cost:

| | textures | busiest wall |
|---|---|---|
| one wall per era | 7 | **966** buildings (73%) |
| a flat three | 17 | 345 (26%) |
| allocated | **20** | **91** (6.9%) |

`facade.LookPlan` owns the slot arithmetic — `offsets[era] + look`, specials
after all of them — because three places used to do it independently and a
layout only works while they all agree. The buildings stage measures the
allocation and the facade stage is *handed* it rather than recomputing it from
config: a building carries a slot number, not a style, so two sides disagreeing
about the layout puts the wrong wall on every building after the first era
whose count changed, and nothing looks broken.

A building picks its look from **its own BAG identifier**, not from a counter,
so it keeps the same wall when the area is rebuilt, re-tiled, or fetched in a
different order — the same reasoning as the trees taking their shape from their
position. The hash is FNV-1a over the identifier rather than `hash()`, which is
salted per process and would hand the same building a different wall on every
run while the textures it indexes stayed put.

Three details worth knowing:

* **The colours are hand-picked per era, not jittered.** Dutch brick runs red
  to purple-brown to yellow IJsselsteen, which is a range with gaps in it, and
  a random walk through RGB lands in the gaps.
* **The geometry table has three entries against four wall colours.** Four
  would turn over in step with them, and look 4 would come out as look 0
  wearing a different window frame. Three against four gives twelve distinct
  combinations, which is `MAX_LOOKS_PER_ERA`; asking for more is clamped.
* **A look borrows its era's photograph.** The photographed wall surface below
  is keyed by style, and a look is a new style name, so `historic_1` found no
  photograph and fell back to drawn brick — one house in three with real grain
  and its neighbours without. `FacadeStyle.family` resolves a look back to its
  era, and only the tint differs.

Slots are laid out `era * looks + look`, with the two archetype specials after
all of them, so an extra look does not move an era and an extra era does not
move the specials. Materials are named after their texture —
`M_facade_04_historic_1` — because with seventeen of them a designer
reassigning one has to be able to tell them apart in a list.

`facade.looks_per_era: 1` gives exactly what this gave before any of it
existed, down to the texture filenames.

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

## Collision shapes

Off by default (`export.collision`, or "Add simplified shapes for physics" in
the UI). This is for driving or flying something through the model rather than
looking at it, and it costs FBX size that most imports do not want.

Two extra objects appear, and a third in the layered style where the motorway
has guardrails; only these earn their place:

| Object | What it is | Against the drawn mesh |
|---|---|---|
| `Collision_Trees` | a trunk prism and a box round the canopy, per tree | 28 triangles against 52–84, so **2.3×** cheaper over a real height mix — or 16 with `export.collision_tree_crowns` off, **4×** |
| `Collision_Barriers` | the same walls and fences, fattened | one extra prism per thin run |
| `Collision_Guardrails` | a block along each guardrail, from the rail's face back over its posts and from below the ground to the rail's top, fattened as a fence is | 8 triangles per post against the drawn 20; 13,000 over the Hazeldonk A16 square |

Everything else in the scene is already the right shape to collide with, and
`metadata.json` names which object to use for each layer instead of shipping a
copy of a mesh that is already minimal:

```json
"collision": {
  "objects": {"Collision_Trees": 365204, "Collision_Barriers": 5120},
  "reuses_its_own_mesh": {"Terrain": "...", "Pand_*": "...", "Road_*, Path_*, ...": "..."},
  "note": "Add a Mesh Collider to each Collision_* object and untick ..."
}
```

In Unity: select the `Collision_*` objects, add a Mesh Collider, untick Mesh
Renderer. That is the whole setup, which is the point of the naming.

**A fence is 6 cm thick, and that is thin enough to fly through.** This is the
reason the barrier proxies exist. Unity's default physics step is 50 Hz with
discrete collision detection, so a body doing 20 m/s moves 40 cm between steps:
it starts one side of the panel, finishes the other, and no contact is ever
generated. `barriers.collision_thickness_m` defaults to 0.30 m, which covers
15 m/s and is invisible because the drawn fence stays 6 cm. Above that, set the
moving body's Collision Detection to **Continuous Dynamic** as well — that
cannot tunnel at any speed, and thickening is a mitigation rather than the cure.
A guardrail's beam is 8 cm deep, rail and post 18 cm together, so its block is
fattened to the same 0.30 m, onto the verge behind the rail and never into the
carriageway in front of it, where a car would meet a wall it cannot see.

Which dimension gets fattened depends on how the barrier was surveyed. A line
barrier is a swept prism, so its `thickness_m` is the thin direction. An area
barrier takes its width from its own outline — the BGT only draws a wall as a
polygon when it is wider than half a metre — so there the thin direction is
`height_m`, and that matters for one family: an awning is a 12 cm plate hanging
2.4 m up, and something climbing a facade goes straight through it.

**A collider has to contain what it stands in for, and a plausible-looking one
does not.** The first tree proxy here was the ellipsoid inscribed in the
canopy's bounding box: 32 triangles, convex, a good visual fit. Measured
against the drawn vertices, foliage stuck up to **4.3 m** out of it, because
the drawn crown fills that box while an inscribed ellipsoid only touches it on
the axes. A drone would have flown through visible leaves and felt nothing.

So the canopy proxy is the box itself — fewer triangles, and it contains the
drawn crown by construction, since that is the box the lobes were scaled to
fit. Where it is wrong it is wrong in the safe direction. The cost is invisible
margin at the corners: 2.2 m on a 7 m tree, 8.9 m on a 25 m one, measured to
the nearest drawn foliage. If flying close to trees matters more than never
clipping them, turn `export.collision_tree_crowns` off and only wood is solid.

A test contains every drawn vertex inside its own proxy, over a height sweep
and at the one position in the whole hash where a tall tree leans its hardest
— the only case where the trunk prism needs the lean added to its radius, and
one that a sweep of ordinary positions never produces.

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

**Every building is its own object.** One merged mesh was cheaper to draw and
impossible to edit: deleting a building, or moving one aside for a scenario,
meant cutting it out of a mesh holding a thousand others. Unity's static
batching recovers the draw calls for buildings that do not move.

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

A run over the Bronbeek square kilometre in Arnhem, collision shapes on, passes
62 of 62 in just over 5 minutes: a terrain of 263,601 faces following 42,367
outline edges, 188 road objects of 17,652 faces, 1,250 buildings, 10,743
trees, 1,007 parked cars, a 64 MB FBX and a 29 MB `.blend`. The Blender stage
is under half a minute of that.

The road and terrain checks are the ones to read after a change to either:
`roads_are_mostly_quads`, `road_surface_follows_terrain` (no road far off the
survey), `terrain_stays_under_roads` (no ground through a road),
`terrain_follows_the_survey`, `terrain_is_one_closed_sheet` (no cracks, no
overlaps, nothing upside down) and `terrain_follows_the_outlines`.

In the layered style the road checks give way to its own, read back from what
Blender built: `ground_layers_built` (every road, walkway and grass object is
in the model), `layers_lie_on_the_terrain` (each layer's vertices, after the
modifiers, stand their lift above the terrain, and a walkway no higher than its
kerb), `walkways_have_kerbs` (the curb generator ran),
`terrain_follows_the_layer_outlines` and `terrain_follows_the_survey`, which
holds the coarse terrain to its own tolerance rather than to the survey
style's 25 cm. In both styles `terrain_exported_as_triangles` checks the FBX
holds the terrain cut as the pipeline cut it.

When the roofs have their own photo, four checks read it back:
`roof_atlas_written` (every page on disk at its planned size, with content in
it), `roof_atlas_within_size_cap` (no page over `roof_atlas_px`, and all of them
within `roof_max_mpx`), `roof_uvs_inside_their_rectangles` (every corner of
every roof triangle inside its own building's rectangle, at least the padding
from its edge) and `roofs_use_the_atlas` (Blender put as many roof triangles on
`M_roof` as were planned). `roof_resolution_as_asked` warns when the budget
made the roofs coarser than `roof_cm`. An atlas that could not be fetched is a
warning too: the roofs are then on the aerial, as they were before.

The guardrails have three: `guardrails_present` (how many metres, posts and
triangles, and a warning where a motorway with verges has none),
`guardrails_on_the_verge` (every post and every point of the rail's face on a
face of grass, open ground or a hard verge, read off the ground's own faces
rather than the distances the rails were laid out by) and
`guardrails_stand_on_the_ground` (read back from Blender: the object has every
post foot and rail top the layout placed, no corner of any foot is less than
8 cm into the ground as built, so none is in the air, and the top of the
rail's face is within 2 cm of 0.75 m above that ground at every post).

`web_view_written` says whether the 3D view was written. It only ever warns:
the model is whole without it, and the page writes the view from the `.blend`
when it finds none.

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
- **Road centrelines and street names** — NWB-Wegen (Nationaal Wegenbestand),
  Rijkswaterstaat, via PDOK (CC0)
- **Wall surfaces** — [Poly Haven](https://polyhaven.com) (CC0). No attribution
  is required for these; they are credited because it is the decent thing, and
  listed here so you know nothing in the model carries an obligation you did not
  choose.
