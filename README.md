# Helios

A Rust + WebGPU (wgpu) rendering library that implements the core ideas behind Unreal Engine 5's
**Nanite** (virtual geometry with streaming and a software rasteriser, skinned meshes, voxel foliage LODs and
assemblies), **Virtual Shadow Maps**, **Lumen** (software ray-traced global illumination and reflections, plus a
Lumen Lite irradiance field for weak GPUs), **MegaLights** (stochastic shadowed lighting from hundreds of local
lights), **Substrate**-style layered materials with material graphs, heterogeneous volumes, GPU particles and a
dynamic world of moving instances, with its material / lighting / post stack — running natively and in the
browser, where a UE-style level editor with scene files and a PCG graph sits around the live viewport.

**Live showcase:** https://mp3pintyo.github.io/helios/ (needs a WebGPU browser: recent Chrome, Edge or Firefox; the
browser compiles the renderer's GPU shaders on the way: the first image after ~9 s in Chrome and ~11–15 s in Firefox,
shadows, Lumen and the rest joining as their shaders arrive — see gap 15).
Redeploy with `python site/build.py` then `python site/deploy_pages.py`.
Round 4's design notes, measurements and screenshots, one file per feature: [`docs/round4/`](docs/round4/).

```
crates/
  helios-geometry   pure Rust, no GPU: clustering, METIS-style grouping, seam-aware QEM simplification,
                    LOD DAG + streaming pages, skin weights, voxel LODs, assemblies, mesh distance fields,
                    procedural modelling and vegetation
  helios-asset      pure Rust: glTF 2.0 import (.gltf / .glb) -> meshes, node instances, PBR materials (with the
                    clearcoat / sheen / ior / specular / anisotropy / transmission / unlit extensions), skins and
                    animation clips, RGBA8 images
  helios-material   pure Rust: material graphs (serde data) validated and compiled to WGSL, with screen derivatives
  helios-pcg        pure Rust: deterministic procedural content generation graphs (sample, filter, prune, spawn)
  helios-render     wgpu renderer: GPU-driven culling, HW + SW + voxel raster, streaming, materials, VSM, Lumen,
                    MegaLights, volumes, particles, OIT, post
  helios-demo       four demo scenes (valley, cathedral, forest, gallery) + glTF viewer: native window, headless PNG
                    capture, wasm; the editor core (scene documents, scene files, PCG volumes)
web/                index.html, editor.html + editor/ (ES modules, no build step), generated pkg/ for the browser build
```

## Quick start

```powershell
# Native (interactive). Scenes: valley, cathedral, forest, gallery. Detail: medium | high.
cargo run -p helios-demo --release -- --scene cathedral
cargo run -p helios-demo --release -- --scene gallery                         # the material gallery

# Headless screenshot (CI / regression images)
cargo run -p helios-demo --release -- --scene forest --screenshot out.png --size 1600x900 --frames 120 --detail medium

# Browser (needs WebGPU: Chrome/Edge 113+, Safari 26+, Firefox 141+)
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130
./build-web.ps1
python -m http.server 8080 --directory web     # open http://localhost:8080/?scene=cathedral
#   ?shaders=serial|parallel: how the browser's background shader compile is worked (default: by browser)

# The level editor (browser): outliner, details, gizmos, content browser, PCG volumes, material graphs
python serve.py 8080 web
#   http://localhost:8080/editor.html?scene=valley                 (or Tab in index.html)
#   http://localhost:8080/editor.html?scene-file=my.helios.json    open a saved scene file (also ?scene-file= in index.html)
cargo run -p helios-demo --release -- --scene-file my.helios.json     # natively (Tab: editor tools, no panels)

# Lighting options: GI final gather, MegaLights, the candle-lit cathedral (every flame a shadow-casting light)
cargo run -p helios-demo --release -- --scene cathedral --gi lite                      # lumen | lite | auto (default)
cargo run -p helios-demo --release -- --scene cathedral --lights 1000 --megalights --sun 200,-30
#   browser: ?gi=lite, ?megalights=1, ?lights=300 (or heliosConfig.gi / .megaLights / .lights)

# Your own content: a glTF 2.0 model (.glb, or .gltf with its files); skinned models play their animations
cargo run -p helios-demo --release -- --gltf path/to/Sponza.gltf      # [--no-ground]
cargo run -p helios-demo --release -- --gltf path/to/Fox.glb          # - (Minus) crossfades to the next clip
#   browser: http://localhost:8080/?gltf=models/DamagedHelmet.glb (fetched relative to the page)
#   browser resolution: dynamic resolution (14 ms GPU budget) is on by default; ?sp=67 fixes the screen
#   percentage, ?budget=10 changes the budget, ?budget=0 turns it off
#   ?bake-workers=0: Lumen's displaced variants bake on the render thread instead of Web Workers (comparisons)

# Streaming geometry from pre-baked .hvm files instead of baking at startup
cargo run -p helios-demo --release -- --scene forest --bake-hvm hvm/forest      # bake once
cargo run -p helios-demo --release -- --stream-from hvm/forest                  # stream from disk
helios --bake-hvm web/forest --scene forest --detail medium; python serve.py 8080 web
                                                # open http://localhost:8080/?stream=forest (HTTP Range)
```

Controls: `WASD`/`QE` fly, drag to look, wheel = speed, `Shift` = fast, `N` next scene, `V` cycle debug views
(or `2`–`9`), `1` local light shadow maps (off: distance-field shadows), `[` / `]` screen percentage down / up (50, 67,
75, 100 %; the HUD shows it), `\` dynamic resolution on / off, `,` / `.` LOD error, `G` GI, `H` shadows, `T` temporal
accumulation (TSR history; off = spatial upscale only), `O` occlusion culling, `K` software raster, `I` texture streaming,
`F` aerial perspective, `C` clouds, `U` translucency, `/` per-pixel translucency layers, `L` wind, `M` motion blur, `B` depth of field, `` ` `` local exposure, `Z` / `X` zoom (a longer lens
makes the depth of field shallower), arrow keys move the sun (down past the horizon into the night),
`P` animates the sun, `J` runs a day-and-night time-lapse, `Y` starts a camera tour, `F12` saves a screenshot, `R` resets
the camera. The camera stays above the ground but may dive into the valley's lake (`Q` down over the water) for the
underwater view. Round 4 (all letters were taken): `0` pauses / resumes the demo motion (swinging chandeliers,
turning sculpture, drifting raft), `-` next animation clip of a skinned model (0.4 s crossfade; after the last one
it pauses), `;` GI final gather Lumen ↔ Lumen Lite (ends the automatic choice), `=` MegaLights, `'` volumes +
particles, `Insert` voxel LODs (off: the triangle-only cut of the same DAG), `PageUp` Substrate slabs (off: every
material renders as its plain base), `PageDown` Nanite tessellation (off: displaced materials render undisplaced), `End` coloured sunlight
through glass (off: it lands white), `Tab` editor (natively: the editor tools without panels — click to select,
`Del`, `Ctrl+D`, `F`, `Ctrl+Z` / `Ctrl+Y`, `Ctrl+S` saves the scene file; in `editor.html`: play mode).

Editor (`web/editor.html`, viewport focused, right mouse button not held): `W` / `E` / `R` move / rotate / scale,
`Space` cycles them, `` Ctrl+` `` world / local axes, `F` frames the selection, `Del` delete, `Ctrl+D` duplicate,
`Ctrl+Z` / `Ctrl+Y` undo / redo, `Ctrl+S` save, `Ctrl+O` open, `Esc` cancels a drag or deselects; right mouse +
`WASD`/`QE` flies, the wheel dollies; `Shift` / `Ctrl` click adds / toggles; drag content into the viewport, drop a
material onto an object to assign it.

Headless flags: `--scene`, `--detail`, `--size WxH`, `--frames N` (the log ends with means over the last two thirds
of the frames: GPU and per-pass times, shadow pages and draws, moving lights' faces re-anchored), `--view i` (debug view by index, 0–25; also
`heliosSetView(i)`; 14 = Lumen mesh cards, 15 = local light shadow pages, 16 = invalidation, 17 = skin weights,
18 / 19 = MegaLights samples / samples | denoised, 20 = Lumen Lite probes, 21 = volume march steps and particle
overdraw, 22 = voxel clusters, 23 = assembly parts, 24 = material features, 25 = tessellation depth and
micro-triangles), `--eye x,h,z` /
`--at x,h,z` (heights relative to the ground), `--camera x,y,z,yaw,pitch`, `--lod px`, `--pool-mb N`
(geometry pool size, to exercise streaming), `--texture-pool-mb N` (texture tile atlases), `--no-texture-compression` (uncompressed tiles; in
the browser `?texture-compression=0`), `--no-occlusion`, `--no-sw-raster`,
`--no-rvt`, `--no-glass-shadows` (sunlight through glass stays white), `--no-svt` (image textures from the 512² arrays only), `--animate-sun`, `--sun AZ,EL` (sun azimuth and
elevation in degrees; e.g. `200,-30` for a starry night),
`--keys KeyG,KeyH` (press hotkeys after loading; with `--keys-at N` at frame N instead), `--hold KeyW` (hold keys every frame: a moving camera),
`--screen-percentage N` (render at N % of the output, 25–100, and upscale with TSR), `--gpu-budget-ms N` (dynamic
resolution: pick the percentage, 50–100 % or up to `--screen-percentage`, that keeps the GPU frame under N ms),
`--no-local-exposure` (global exposure only), `--local-exposure HIGHLIGHT,SHADOW[,BLEND]` (its contrasts and the
blurred luminance blend), `--switch-to SCENE` (reload another scene halfway), `--dump-dir DIR [--dump-count N]` (save the last N frames, e.g. to
measure temporal stability), `--bake-hvm DIR` (write the scene's baked meshes and exit), `--stream-from DIR` (build
the scene from them, streaming page payloads from disk; scene and detail come from the library), `--io-latency-ms N`
(emulate slow storage), `--fov DEG`, `--move-light` (the shadow-casting light nearest the camera circles every frame,
so its shadow map is re-rendered every frame). Round 4: `--scene-file PATH` (build a `.helios.json` scene file; relative
glTF paths resolve against it), `--select i,j` (outline these instances, as the editor's selection),
`--gi lumen|lite|auto`, `--megalights`, `--lights N` (the candle-lit cathedral: the first N of up to 1008 candle flames
lit, each a shadow-casting light), `--no-skinning` (import a glTF model's skinned meshes as static bind-pose meshes, for
cost comparisons), `--full-invalidation` (every move invalidates everything, as before per-instance invalidation),
`--moving-light-maps` (moving local lights' shadow maps re-rendered every frame they move: the reference for the
cached default), `--moving-light-fields` (distance-field shadows while they move, round 4's default),
`--moving-light-pages N` (pages of moving lights' shadow-map faces rendered from new origins per frame, default 32;
round 5's `--moving-light-renders` counted whole lights), `--lantern` / `?lantern=1` (the cathedral with a lantern
carried down the nave: a fast, sharp moving light; with `--movers 1` only the lantern moves),
`--movers N` (only the first N moving prop groups move), `--static-motion` (the props move as `Mobility::Static`
instances: the one-off edit path), `--no-tessellation`, `--dice-rate N` (pixels per micro-triangle edge, default 2),
`--speed N` (fly speed in m/s for held keys, e.g. a slow dolly: `--hold KeyW --speed 0.3`). Next to the HUD (whose
per-pass times are smoothed), a headless run logs `GPU minima over N frames`: every pass's minimum, and the minimum
frame total, over the last two thirds of the frames — on a GPU shared with other work, a pass's quiet time.

**Streaming from files.** `--stream-from` / `?stream=` build the same procedural scene but substitute every baked
mesh with its pre-baked counterpart: only the header (cluster and page records, dependencies, bounds, distance field)
and the root pages are loaded; all other pages are read on demand. Forest (high): 43 MiB of geometry in RAM when
baked vs 3.7 MiB streamed, scene ready in 0.9 s instead of 8 s. In the browser, `python -m http.server` ignores
`Range` headers — it still works (each `.hvm` is then downloaded whole, once), but `python serve.py` (or any server
honouring `Range`, e.g. a CDN) streams pages on demand. Imported glTF models are not streamed (they bake at load).

## The demo scenes

| Scene | What it shows | Size |
|---|---|---|
| `valley` | Mountain valley, 600 boulders, a gold sculpture (turning once in 45 s) and a small lake with a log raft that drifts, bobs and rolls on it, procedural (not simulated) mist over the water; terrain textured through a runtime virtual texture; at the lake's north end, out of the initial view, a glass pane and two crossing tinted panels standing in the water (layered translucency test); behind the initial view a cobblestone path and displaced boulders (Nanite tessellation; one turning slowly on its plinth: a movable displaced instance), e.g. `--eye -26,1.8,39 --at -33,0.2,53` | 613 instances, 50 M (medium) / 199 M (high) source triangles |
| `cathedral` | Gothic basilica: 11-bay nave with clustered piers, triforium, stained-glass clerestory and aisles, quadripartite rib vaults, flying buttresses, polygonal apse with half dome, west front with rose window and twin spired towers; pews, altar, candelabra and eleven chandeliers swinging on their 12 m chains (7 s pendulums, 1–3°, their candle lights moving along), 33 candle lights in all; dust motes in the clerestory's light shafts and incense rising from a censer before the altar; alpha-tested wrought-iron choir screens; at the west end of the north aisle, behind the initial view, a stained-glass screen and two crossing tinted glass panes (layered translucency test). **Candle-lit** with `--lights N` / `?lights=N`: every candle flame its own shadow-casting light (60 candelabrum, 88 chandelier and 2 altar candles) plus votive racks of tea lights against the aisle walls (858), the first N lit, up to 1008. With `--lantern` / `?lantern=1`: a hand lantern (iron frame, candle, a 2 cm flame light) carried down the central aisle and back at walking pace, swinging from the hand — a fast, sharp moving light | 458 instances, 4 M (medium) / 18 M (high) |
| `forest` | Mixed broadleaf / conifer forest with real leaf geometry, bushes, ferns, grass, mossy boulders, fallen logs and mushrooms at golden hour, swaying in a gusty breeze; conifers built from shared needle-spray assemblies, foliage with voxel LODs; a campfire in the clearing in front of the initial view (simulated fire and smoke, sparks that bounce, embers that settle and cool, a flickering fire light; best at night, `--sun 200,-30 --eye 3.5,1.4,1.0 --at 7,1.0,-1`) | 8 278 instances (≈ 26 400 assembly parts when expanded), 82 M (medium) source triangles |
| `gallery` | The material gallery: a paved courtyard at golden hour with a row of plinths — clear-coat car paint, velvet with sheen, subsurface skin, a lit wax candle and a marble egg, a bush whose leaves let a lamp's light through, an anisotropic brushed-metal pot, dielectrics of IOR 1.33 / 2.4 and a tinted specular, an unlit sign — and seven material graphs: animated lava (a pool and an egg), stylised water (two instances of one graph with different parameters), a terrace of wet stone with puddles and, in front of it, three beds displaced by their graphs through Nanite tessellation: a brick terrace with carved mortar joints (`--eye 0.9,0.45,10.9 --at 0,0,9.2`), left of it a bed of mushrooms (a vector displacement: caps overhanging their stems, `--eye -5.9,0.3,10.4 --at -6.5,0.08,9.2`) and right of it a rippling cloth (an animated displacement, `--eye 4.4,0.6,10.9 --at 6.3,0.15,9.0`); behind the default view a platform of combed tufts turns on a drum (a tangent-space vector displacement that turns with it, `--eye 1.9,1.1,15.6 --at 0,0.1,13.4`) | 50 instances |

These are procedural: meshes come from `helios_geometry::modeling` / `vegetation`, textures from GPU
texture synthesis (`TexturePattern`). `0` pauses the moving props; `Y` tours each procedural scene, the gallery
included.

`model` (`--gltf FILE`, `?gltf=URL`) builds a scene from a glTF 2.0 file: every unique mesh (glTF primitive) is baked
once into virtual geometry — in parallel, with a distance field sized to its world extent — and placed by the file's
node instances; materials become metallic-roughness PBR materials whose base colour, normal, metallic-roughness and
occlusion images are resampled into one texture layer and streamed at their native resolution (see *Streaming virtual
textures*). `N` cycles through it with the procedural scenes. Beyond the core: a second UV set (`TEXCOORD_1`) for any
texture, `KHR_texture_transform` (offset, rotation, scale, `texCoord` override) and sampler wrap modes per texture,
vertex colours (`COLOR_0`), ambient occlusion maps (indirect light only), `KHR_materials_emissive_strength` and
`KHR_mesh_quantization` (8/16-bit normalized or integer attributes, interleaved or sparse accessors). Checked against
the Khronos sample models TextureTransformTest, TextureTransformMultiTest (every row but clearcoat and unlit, checked
before those extensions were imported),
MultiUVTest, VertexColorTest, TextureCoordinateTest, EmissiveStrengthTest, the quantized Lantern and Avocado,
DamagedHelmet and WaterBottle. Alpha-masked materials (`alphaMode: MASK`: foliage
cards, chains) are alpha-tested; `BLEND` materials become `MaterialKind::Translucent` (forward translucency pass,
coverage = base colour alpha × texture alpha); their meshes get no distance field and their instances cast no shadow.
The PBR extensions `KHR_materials_clearcoat`, `_sheen`, `_ior`, `_specular`, `_unlit` and `_anisotropy` (factors) become
Substrate-lite slabs and `KHR_materials_transmission` (+ `_volume` attenuation) glass (see *Substrate-lite*); checked
against ClearCoatTest, ClearcoatWicker, SheenChair, SheenTestGrid, SpecularTest, IORTestGrid, TransmissionTest, UnlitTest,
AnisotropyBarnLamp and CarConcept. Skinned meshes (`JOINTS_0` / `WEIGHTS_0`, any component type) play the file's
animations: the first clip starts on load, `-` (Minus) crossfades to the next clip and pauses after the last
(single-clip models: play / pause); the HUD shows the clip, its time and whether it is paused. Checked against Fox,
CesiumMan, BrainStem, RiggedFigure, RiggedSimple and SimpleSkin. The model is placed by its posed bounds at the start
of the first clip.

## Showcase site

`site/index.html` is a bilingual (Hungarian / English) page that embeds the engine: a pinned live viewport with live
telemetry (GPU time, resolution, triangles, per-pass GPU time against a 16.7 ms frame) next to chapters whose
buttons drive it (scene, camera, sun, debug views, feature toggles), before/after sliders, camera-tour videos and
the story of how Helios was built. It talks to the engine through the page API in `crates/helios-demo/src/web.rs`:
`globalThis.heliosConfig = { scene, embedded, screenPercentage, gpuBudgetMs, gi, megaLights, lights, statusElement }`
before loading the module, then `heliosLoadScene`, `heliosLoadModel(url)` (fetch a glTF model and make it the scene;
a promise), `heliosSetCandleLights(n)` (the cathedral's candle lights from its next build; 0: the 33 fixture lights),
`heliosKey`, `heliosLookAt`, `heliosSetSun`, `heliosSetView`, `heliosSetPaused`, `heliosHud`, `heliosScene` and
`heliosBusy` (what the engine waits for — `Compiling GPU shaders… 4 s (31 / 52)` while the browser compiles the
core pipelines of the first image, `Compiling GPU shaders… 2 s` for a new scene's or a feature's that a frame needs at
once, the canvas keeping its last frame — or empty while it renders), `heliosFrames` (frames presented; 0 until the
first image) and `heliosShaders` (the background compile as JSON: pipelines compiled and in all, the features still
waiting for theirs, the time to the first image and to full quality). The site shows `heliosBusy` on its veil until
the first frame, then a note with the features still compiling.
`embedded: true` keeps keyboard input on the focused canvas and leaves the mouse wheel to the page; `statusElement`
names the element the engine writes its (English) progress and errors into instead of `#status`, so a page can show
its own, translated text. The editor uses a second set of exports (`heliosEditor*`, `heliosMaterialNodes`; see
*Editor*), and `editor.html` ships next to the page.

```powershell
./build-web.ps1                     # web/pkg
cargo build --release -p helios-demo
python site/capture.py all          # stills + tour videos into site/media-src (needs ffmpeg; HELIOS_ASSETS = glTF samples)
uv run --with playwright --with pillow python -I site/capture_editor.py   # editor stills + video (headless Chromium)
python site/build.py                # site/dist: page, pkg (with pkg/snippets/: the bake worker), editor, media
python serve.py 8080 site/dist      # open http://localhost:8080/preview.html
```

## Using the library

```rust
use helios_render::{geometry::{build_virtual_mesh, primitives, BuildConfig}, *};

let mut scene = Scene::new();
let mesh = scene.add_mesh(build_virtual_mesh(&primitives::rock(1, 7), &BuildConfig::default()));
let rock = scene.add_texture(TextureDesc::new(TexturePattern::Rock, glam::Vec3::splat(0.3), glam::Vec3::splat(0.15)));
let material = scene.add_material(Material::textured(MaterialKind::Rock, rock, TextureMapping::ObjectTriplanar, 0.5));
scene.add_instance(mesh, material, glam::Mat4::IDENTITY);
scene.add_light(Light::point(glam::Vec3::new(0.0, 3.0, 0.0), glam::Vec3::new(1.0, 0.6, 0.3), 2.0, 10.0));

let mut renderer = Renderer::new(&device, &queue, surface_format, width, height);
renderer.settings.lod_threshold = 1.0; // max geometric error in pixels
// every frame:
renderer.render(&scene, &camera, &surface_view, delta_seconds);
```

A complete program — scene, offscreen render, PNG — is `crates/helios-render/examples/headless.rs`
(`cargo run -p helios-render --release --example headless -- out.png`). Atmosphere and camera effects are scene /
settings fields: `scene.clouds = CloudLayer::cumulus(0.4)`, `scene.fog = FogVolume { density: 0.01, .. }`,
`scene.set_time_of_day(azimuth, elevation, intensity)` (below the horizon the moon and stars take over),
`renderer.settings.depth_of_field = Some(DepthOfField { f_stop: 1.8, ..Default::default() })`,
`renderer.settings.motion_blur = 0.5` (shutter fraction).

Round 4's scene API: `scene.set_transform(instance, m)` moves an instance and invalidates only what it touches, and
`scene.set_mobility(instance, Mobility::Movable)` marks one that moves often (a dynamic shadow caster);
`scene.add_skin` / `set_instance_skin` / `set_skin_pose` (every frame) animate skinned meshes, with
`helios_asset::animation::AnimationPlayer` producing the palette from glTF clips; `scene.set_assembly(base, parts)`
draws a mesh (`helios_geometry::assembly::build_assembly_base`) as placed part meshes up close; `scene.volumes` /
`scene.particles` hold `VolumeEmitter`s and `ParticleEmitter`s; `Material::slab` (`Slab`: clear coat, sheen,
subsurface, transmission, anisotropy, IOR, specular, unlit) and `Material::graph` (a `helios_material::MaterialGraph`
added with `scene.add_material_graph`) extend a material, and `scene.set_material_parameter` overrides a graph
parameter per material (a material instance). Settings: `mega_lights`, `gi_method` (`GiMethod::ScreenProbes` /
`LumenLite`), `gi_lite_fallback`, `voxel_lods`, `substrate`, `volumes`, `particles`, `per_instance_invalidation`.

Any indexed triangle mesh works (`helios_geometry::Mesh`); `Mesh::weld` merges identical vertices while keeping UV
seams and hard edges as separate wedges. Baking is deterministic and can run offline: `VirtualMesh::to_bytes` /
`from_bytes` (`.hvm`, version 4: compressed page payloads last, so a reader can stream page byte ranges; voxel LOD
payloads and skin sections; version 2 files, which held float vertices, and version 3 files, which had no voxel LODs,
are rejected with a request to bake again). Foliage meshes opt into voxel LODs with `BuildConfig::voxels =
Some(VoxelConfig { voxels_per_triangle })`.

To stream a baked mesh instead of keeping it in memory, load its header and root pages and give the scene a
`PageSource` for the rest (`FileSource` natively, `HttpSource` in the browser, or your own):

```rust
// `read(range)` returns those bytes of the .hvm file.
let header_len = VirtualMesh::header_len(&read(0..VirtualMesh::PREFIX_BYTES as u64))?;
let mut mesh = VirtualMesh::from_header_bytes(&read(0..header_len as u64))?;     // header only
let roots = mesh.root_tail_start();
mesh.set_tail_payload(roots, &read(mesh.tail_byte_range(roots)))?;               // root pages: one range
let source = Arc::new(FileSource::open(&[path], 2)?);                             // I/O threads
scene.add_streamed_mesh(mesh, source, 0);                                         // file 0 of the source
```

---

## Architecture

### 1. Virtual geometry — `helios-geometry`

| Step | What happens | Where |
|---|---|---|
| Clustering | Greedy, connectivity-driven meshlets (≤ 128 triangles / vertices) grown across UV seams; when the connected frontier runs dry (leaves, needles, grass blades) growth continues with the nearest free triangle among the Morton-order neighbours, so foliage clusters are full instead of one leaf each | `cluster.rs` |
| Grouping | **Multilevel graph partitioning** (METIS scheme): heavy-edge matching, greedy-growing bisection, Fiduccia–Mattheyses refinement, recursive bisection; nodes weighted by triangles, weak spatial edges for disconnected geometry | `partition.rs` |
| Simplification | Quadric error metric on **position classes**: wedges (seams/hard edges) collapse consistently or not at all, open borders slide only along themselves (boundary quadrics), normal/UV deviation steers the collapse order. **Component pruning** removes the smallest disconnected pieces (leaves) and scales survivors to preserve area; its error is scaled down (`prune_error_scale`, stochastic simplification of aggregate detail) because thinning a canopy changes coverage, not the silhouette | `simplify.rs` |
| DAG | Simplified groups are re-clustered until one root remains; each cluster stores its own and its parent's error and LOD sphere | `dag.rs` |
| Pages | Member clusters of whole groups are packed into **streaming pages** with explicit parent-page dependencies | `dag.rs` |
| Compression | Positions snap to a **per-mesh grid** (power-of-two step ≈ 1/64 of the mean shortest triangle edge, ≤ 1 mm), UVs to a 2^-15 grid; each cluster stores offsets from its minimum grid point with just enough bits per axis, 10+10-bit octahedral normals and `⌈log2 vertices⌉`-bit indices — about 10 bytes per vertex instead of 24, 2.1–2.5× smaller pages overall | `encoding.rs` |
| Skin weights | Imported joint influences (≤ 4 per vertex) are renormalised and quantised **once per vertex** to 8-bit weights summing to exactly 255, then welded, clustered, simplified and compressed like any attribute: a cluster stores its joints as offsets from its smallest joint and the weights of all but the heaviest influence with just enough bits, in a skin section behind the vertex stream (a header bit; meshes without skin encode bit-identically). Simplification adds a **weight-deviation cost** (the weight a collapse hands to other joints), so strong weight seams collapse last | `skin.rs`, `encoding.rs`, `simplify.rs` |
| Voxel LODs | For aggregate meshes (`BuildConfig::voxels`), every DAG group also gets a voxel candidate built from the full-detail triangles under it; where the voxels' error is below the simplified triangles' (where pruning starts deleting leaves) the group switches to voxels at the triangles' error, and both representations stay in the DAG (see [Nanite Foliage](docs/architecture/foliage.md)) | `voxel.rs`, `dag.rs` |
| Assemblies | A base mesh built from placed part meshes at their roots, merged in place and simplified (and voxelised) further, its errors offset by the parts' worst root error so they continue where the parts end | `assembly.rs` |
| Distance fields | Narrow-band mesh SDFs (flood-filled sign that seals sub-voxel cracks, pseudonormals only at the surface; a half-voxel shell around parts thinner than a voxel; two-sided for genuinely open surfaces) or heightfields for terrain and large slabs | `sdf.rs` |

**Why it is crack-free.** Every cluster created from group *G* stores *G*'s error/sphere as its own; every cluster
consumed by *G* stores them as its parent's. They grow monotonically up the DAG, so the per-cluster test

```
resident(page) && (projected(own) <= threshold || !resident(child_page)) && projected(parent) > threshold
```

selects exactly one side of every group boundary — for every residency state that respects page dependencies.
Tests check watertight LOD cuts at 8 distances and under random partial residency.

**Why compression keeps it crack-free.** Quantising each cluster to its own bounding box would round a vertex
shared by two clusters (or by two LOD levels across a locked group border) differently on each side. Instead the
grid is the mesh's: a shared vertex has the same integer coordinates in every cluster, which only stores them
relative to its own minimum, and the decoder rebuilds `f32(min + offset) · 2^e` — at most 24 bits times a power of
two, exact in `f32` — so both sides get bit-identical positions on the CPU and on the GPU. The watertightness tests
run on decoded positions, including on a deliberately coarse grid.

### 2. GPU pipeline — `helios-render`

```
sky LUTs ─ fire-driven light colours (volume emission) ─ light culling (froxels, ≤ 255 lights each)
assembly expansion: assembly bases too coarse for the main view → part instances + one culling record each (every view)
cull instances (+ assembly records) → page batches → pages (residency + LOD prune) → clusters, small pages' one by one
  (EARLY: visible last frame)
  → HW raster (big triangles; alpha-masked ones in a second bin) + SW raster (tiny triangles)
    + voxel raster (foliage voxel clusters, compute)
    + tessellation (displaced clusters: patches → split passes → dice → SW raster, compute) → merge → HZB
  → cull clusters (LATE vs HZB) → HW + SW + voxel raster → merge                  ⇒ visibility buffer + depth
material classification (8×8 tiles) → one resolve pipeline per shading model, extended pipelines per
  (shading model, material graph) for Substrate slabs and graphs              ⇒ G-buffer (+ slab texels)
pick (editor: one thread, on request)
runtime virtual texture: allocate + render requested landscape pages
virtual shadow maps: mark (sun + local lights) → page cache (+ moved casters' footprints) → clear dirty pages
  → cull per clipmap level and once for all local lights → (cluster, page) draws → HW + SW + voxel + tessellated page raster
Lumen: global SDF clipmap (+ recompose bricks under moved instances) → surface cache (mesh cards + voxel lighting)
  → radiance cache → final gather: screen probes, or Lumen Lite's irradiance field → reflections → GI temporal
clouds: half-resolution checkerboard march → temporal · cloud shadow map
volumetric fog: froxel injection (sun and local lights via VSM, radiance cache) → temporal → front-to-back integration
MegaLights (when on): sample lights per 2×2 block → shadow rays (screen, VSM march, distance fields) → shade
  → temporal + à-trous denoise
deferred lighting (sun + VSM + cloud shadows, clustered local lights + VSM / SDF shadows or MegaLights' result, GI,
  reflections, sky + stars + clouds, fog; Substrate slabs where flagged)
  → subsurface diffusion (horizontal compute + vertical full-screen blur, added back)
  → volumes (fluid simulation · lighting cache) · particles (simulation · per-particle lighting)
  → nearest translucent depth → volume march + temporal resolve → particles into OIT targets (front / back)
  → composite of what lies behind the nearest translucent surface
  → translucency (depth-peel overlapping layers → copy lit scene + mips → forward glass / water / blended draws,
    per layer back to front)
  → composite of what lies in front of it (+ TSR reactive mask)
  → TSR (internal → output resolution) → depth of field → motion blur → histogram exposure → local exposure → bloom
  → ACES → selection outline (editor, only while something is selected)
```

Everything before TSR runs at the internal resolution (the screen percentage), everything after it at the output
resolution.

**Measurements.** Unless stated otherwise, costs are GPU times at 1600×900 on an RTX 3090. The method and the
default views' frame times round by round are in [`docs/performance.md`](docs/performance.md); each round's design
notes and full tables are in [`docs/round4/`](docs/round4/) … [`docs/round7/`](docs/round7/).

Each subsystem is described in its own page under [`docs/architecture/`](docs/architecture/):

| Page | What it covers |
|---|---|
| [Nanite core](docs/architecture/nanite.md) | Page streaming, software rasteriser, alpha-masked (programmable) raster, two-pass occlusion culling |
| [Materials](docs/architecture/materials.md) | GPU-synthesised and glTF materials, Substrate-lite slabs, subsurface, material graphs and instances |
| [Virtual textures](docs/architecture/virtual-textures.md) | Runtime virtual texture for the landscape, streaming virtual textures, BC-compressed tiles |
| [Virtual shadow maps](docs/architecture/shadows.md) | The sun's 16k² clipmap, page cache, static / dynamic caster layers |
| [Motion](docs/architecture/motion.md) | Wind (World Position Offset), skeletal meshes, the dynamic world (moving instances, caches, motion vectors) |
| [Nanite Foliage](docs/architecture/foliage.md) | Voxel LODs and assemblies |
| [Nanite Tessellation](docs/architecture/tessellation.md) | Texture and graph displacement, patch splitting, dicing, animated displacement |
| [Lumen](docs/architecture/lumen.md) | Global distance field, surface cache, radiance cache, final gather, reflections, displaced surfaces, Lumen Lite |
| [Local lights](docs/architecture/lights.md) | Froxel light culling, local light virtual shadow maps (cached, moving lights), MegaLights |
| [Atmosphere](docs/architecture/atmosphere.md) | Volumetric clouds, volumetric fog, time of day |
| [Translucency and water](docs/architecture/translucency.md) | Depth-peeled forward translucency, coloured light through glass, water bodies |
| [Volumes and particles](docs/architecture/volumes-particles.md) | Heterogeneous volumes (fluid simulation), GPU particles, order-independent transparency |
| [Post-processing](docs/architecture/post.md) | Temporal stability, TSR and dynamic resolution, camera, local exposure, profiling |

### 3. Editor, scene files and PCG — `helios-demo`, `helios-pcg`, `web/editor`

The browser level editor (outliner, details, gizmos, content browser), `.helios.json` scene files, PCG volumes and the
material graph editor: [`docs/architecture/editor.md`](docs/architecture/editor.md).

### WebGPU constraints that shaped the design

| Constraint | Consequence |
|---|---|
| No 64-bit atomics | Software raster writes depth and payload in two passes |
| No storage-texture atomics, no storage depth textures | Shadow software raster writes a u32 buffer that the page clear copies into the depth atlas; the same buffer keeps the pages' static layers |
| No multi-draw-indirect / `first_instance` | One instanced indirect draw per pass; late passes offset slots via the early count |
| 8 storage buffers per stage | Culling packs feedback + visibility bits into one buffer; materials live in a uniform array; skin palettes live in an `Rgba32Float` texture read with `textureLoad` (the G-buffer resolve was at 8); Substrate's slab table and graph parameters are uniform blocks |
| No bindless textures | Material textures are layers of two texture arrays; image detail beyond 512² lives in tile atlases behind a page table; glTF extension textures are repacked into two extra array layers per material |
| 16 sampled textures per stage (default limit) | The extended (Substrate / graph) resolve needs 17: the device requests up to 24 where the adapter offers them, otherwise slab materials render as their plain base |
| No viewport arrays | VSM pages are rasterised with per-draw tile transforms and clip distances (optional feature) or fragment clipping |
| No forward-progress guarantee | Hierarchy is flattened into instance → page-batch → page → cluster dispatches; tessellation's PatchSplit is a fixed number of indirect split passes over a ping-pong queue, sized by the deepest split of recent frames |
| Indirect dispatch ≤ 65535 per dimension | Large dispatches are 2D |
| No float atomics | Local exposure's bilateral grid and the fire light's summed emission use fixed-point workgroup atomics |
| Read-write storage textures only for 32-bit single-channel formats | Subsurface diffusion adds its result to the rgba16f scene colour with a full-screen pass and additive blending instead of a compute read-modify-write |
| No hardware ray tracing | MegaLights' shadow rays march the depth buffer, the lights' virtual shadow maps and Lumen's distance fields |
| No derivatives in compute shaders (the material resolve is compute over the visibility buffer) | Analytic barycentric UV derivatives; material graphs carry screen derivatives through their math by forward-mode differentiation |
| Browsers compile every pipeline in their GPU process before running what follows it (Firefox one at a time, asynchronous or not, on every visit); wgpu calls only the synchronous `create*Pipeline` | Progressive start-up (`pipelines.rs`): every pipeline belongs to a feature group and is queued in the browser; the renderer turns a feature off until its group is compiled, so the first image needs only the core set. Chromium: each pipeline is warmed through `create*PipelineAsync` (Dawn compiles in parallel) and made when warm, a cache hit. Firefox: the core set compiles while the scene bakes, the rest one per frame. The frame loop renders into its own texture and presents only frames that queued no compile, so the page never stalls behind one; content pipelines (skinning, masks, a translucent model…) are made on first use or queued once the scene has the content (`Lazy`) |
| No wasm threads without cross-origin isolation (`SharedArrayBuffer` needs COOP/COEP headers, which GitHub Pages cannot send) | Lumen's displaced-variant bakes run on Web Workers that each instantiate the page's compiled wasm module and exchange bytes with it (`bake_worker.js`); other start-up work (scene build, texture encoding) runs on the render thread |

## Status vs. Unreal Engine 5 — what is still simpler

The honest list of what remains simpler than UE5 — streaming, Lumen, shadows, materials, translucency, the browser
build and more — is in [`docs/status.md`](docs/status.md).
