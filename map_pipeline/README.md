# Offline 3D map pipeline (3DBAG + AHN + PDOK aerial → Unity)

Takes a bounding box in the Netherlands and bakes a textured 3D model you import
into Unity yourself. Everything runs locally and the result needs no network at
runtime.

```
output/<area_name>/
├── model.fbx        terrain + buildings, one file
├── aerial.png       aerial ortho of the same area
├── metadata.json    bbox, origin offset, CRS, source versions
├── facade.png       generated facade texture, referenced by the FBX
└── ATTRIBUTION.txt  source credits
```

The pipeline stops there. No Unity layer, no viewer, no WebGL build.

## Running it

There is a small web UI, and there is the command line. They do the same thing:
the UI writes a config file and shells out to `pipeline.py`, so it cannot drift
away from the CLI.

### The UI

```bash
pip install -r requirements.txt
python ui/server.py
```

Open <http://127.0.0.1:8765>. Search for a place or click the map, set the side
length, press **Build model**, and watch the log. When it finishes you get the
check results, the preview renders and download links.

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

Blender is located in this order: an explicit `--blender /path/to/blender`, then
a `blender` on `PATH`, then the pip `bpy` module (`pip install bpy`, CPython
3.11). All three produce the same output.

Useful flags: `--skip-blender` runs the data stages only, `--verbose` turns on
debug logging, `--skip-reimport-check` drops the FBX round-trip check.

```bash
python tests/test_pipeline.py            # includes live checks against PDOK/3DBAG
python tests/test_pipeline.py --offline  # pure logic only
```

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

**Unity has its own limits.** Textures over 16384 px cannot be imported at full
size, which a 2 km area at native resolution would exceed. Imported textures
are also capped at 2048 by default, so raise *Max Size* on `aerial.png` or none
of this is visible. A terrain over 65k vertices needs a 32-bit index buffer,
which Unity sets automatically.

The facade normal map is the one quality lever that is not about resolution: it
gives window reveals, sills and storey bands real relief under a moving light.
It costs one small extra texture and survives FBX as the material's bump slot.
It is written as `*_normal.png` because Unity keys off that suffix to set the
texture type automatically.

Other knobs worth knowing:

| Key | Default | What it does |
| --- | --- | --- |
| `facade.variants` | `1` | `1` gives exactly two materials. Up to `4` assigns brick/plaster/concrete/glass by height. |
| `facade.floor_height_m` | `3.0` | Nominal storey height for the window grid. |
| `facade.texture_px` | `512` | Pixels per storey tile. 512 over a 4 m tile is 128 px/m. |
| `facade.normal_map` | `true` | Write a normal map beside each facade texture. |
| `facade.relief_depth` | `0.035` | How far window reveals and storey bands stand out. Small on purpose: a facade is nearly flat. |
| `buildings.clip_mode` | `centroid` | `centroid` keeps buildings whose centre is inside the bbox. `intersect` keeps every building the API returns. |
| `buildings.merge` | `single` | One merged buildings mesh. `per_building` gives one object each. |
| `terrain.mesh_vertices_per_side` | `257` | 257 → 66k terrain vertices. |
| `aerial.max_request_px` | `2000` | Tile size for the WMS mosaic; the service caps requests at 2500. |

## How it fits together

```
src/geo.py         bbox parsing, WGS84 to RD, origin offset
src/elevation.py   AHN WCS -> GeoTIFF -> height grid
src/buildings.py   3DBAG API -> CityJSON -> semantic mesh data
src/imagery.py     PDOK WMS/WMTS -> aerial.png + georeference
src/facade.py      generated facade textures
src/export.py      metadata.json and the Blender scene description
src/validate.py    the headless checks
blender/process.py builds the scene, assigns materials, exports FBX
blender/preview.py renders preview images of an exported FBX
ui/server.py       local web UI, standard library only
ui/index.html      the page: RD map picker, settings, live log, results
```

The heavy CityJSON and raster work all happens on the Python side.
`blender/process.py` only reads `scene.json`, two npz files and the textures, so
nothing depends on a GUI add-on being present in headless mode.

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

## Materials

Two materials, by default.

**`M_aerial`** carries `aerial.png` with a top-down planar UV taken from world XY
normalised to the aerial bbox. It goes on the terrain *and* on the roof surfaces,
so roofs pick up real photo texture for free and buildings blend into the ground.

**`M_facade`** carries a generated facade. Wall UVs are in metres: U runs along
the wall divided by `tile_width_m`, V counts storeys from the building's own
ground level. The storey height is `wall_height / floors`, so the top row of
windows finishes flush with the eaves instead of being cut in half.

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
