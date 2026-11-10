# Helios

A Rust + WebGPU (wgpu) rendering library that implements the core ideas behind Unreal Engine 5's
**Nanite** (virtual geometry with streaming and a software rasteriser, skinned meshes, voxel foliage LODs and
assemblies), **Virtual Shadow Maps**, **Lumen** (software ray-traced global illumination and reflections, plus a
Lumen Lite irradiance field for weak GPUs), **MegaLights** (stochastic shadowed lighting from hundreds of local
lights), **Substrate**-style layered materials with material graphs, heterogeneous volumes, GPU particles and a
dynamic world of moving instances, with its material / lighting / post stack — running natively and in the
browser, where a UE-style level editor with scene files and a PCG graph sits around the live viewport.

**Live showcase:** https://mp3pintyo.github.io/helios/ (needs a WebGPU browser, e.g. recent Chrome or Edge).
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
material renders as its plain base), `PageDown` Nanite tessellation (off: displaced materials render undisplaced), `Tab` editor (natively: the editor tools without panels — click to select,
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
`--no-rvt`, `--no-svt` (image textures from the 512² arrays only), `--animate-sun`, `--sun AZ,EL` (sun azimuth and
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
`heliosKey`, `heliosLookAt`, `heliosSetSun`, `heliosSetView`, `heliosSetPaused`, `heliosHud` and `heliosScene`.
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
| Voxel LODs | For aggregate meshes (`BuildConfig::voxels`), every DAG group also gets a voxel candidate built from the full-detail triangles under it; where the voxels' error is below the simplified triangles' (where pruning starts deleting leaves) the group switches to voxels at the triangles' error, and both representations stay in the DAG (see *Nanite Foliage*) | `voxel.rs`, `dag.rs` |
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

**Measurements.** Unless stated otherwise, costs are GPU times at 1600×900 on an RTX 3090. Those added in round 4
were taken while the GPU was shared with up to six other agents' builds and renders: they are minima (medians where
stated) of interleaved on/off runs, whole-frame totals were at times swamped by the other load, and per-pass minima
are the reliable numbers. The per-feature notes in [`docs/round4/`](docs/round4/) hold the full tables. Default views
at 2560×1440, 100 % / 67 % screen percentage, round 3 → round 4 (final build on an idle GPU, minimum of 3 interleaved
runs; [`perf-regressions.md`](docs/round4/perf-regressions.md)): valley 8.1 / 5.6 → 8.7 / 5.5 ms, cathedral
10.9 / 6.9 → 10.3 / 6.4 ms, forest 14.5 / 11.0 → 18.3 / 12.2 ms (the campfire's shadow-casting light and Nanite
Foliage's assemblies); new in round 4: gallery 8.0 / 5.0 ms (with the graph-displaced brick terrace in view), cathedral
with 300 lights under MegaLights 15.3 / 9.2 ms. Round 5 ([`docs/round5/`](docs/round5/); final build on an idle GPU, minimum of 3 interleaved runs): valley
8.8 / 5.5 ms, cathedral 10.7 / 6.7 ms (the swinging chandelier lights have shadow maps again, cached), forest
17.0 / 11.3 ms (local light pages), gallery 8.0 / 5.0 ms, cathedral with 300 lights under MegaLights 15.2 / 9.0 ms.
Round 6 ([`docs/round6/`](docs/round6/); final build on an idle GPU, minimum of 3 interleaved runs against round 5):
valley 8.7 / 5.4 → 8.8 / 5.5 ms (Lumen's final gather sees the displaced boulders), cathedral 10.8 / 6.6 → 10.9 /
6.7 ms (an eleventh swinging chandelier light gets a shadow map), forest 16.9 / 11.4 → 15.8 / 10.6 ms (assembly
culling), gallery 8.0 / 5.0 → 8.1 / 5.0 ms, cathedral with 300 lights under MegaLights 15.0 / 9.0 → 15.2 / 9.0 ms.

* **Streaming** (`streaming.rs`, `page_source/`). Cluster and page records are always resident; vertex/triangle data
  lives in a fixed GPU pool. Culling stamps "a resident parent wants to refine into this page" into a feedback buffer,
  which is read back asynchronously; the CPU requests needed pages coarse-first together with their missing
  dependency pages, uploads arrived pages within a per-frame budget once their dependencies are resident (the
  invariant that keeps the cut crack-free), and evicts least-recently-needed leaf pages when the pool is full. A
  missing page just means its parents stay drawn. Page payloads come from memory or from an asynchronous
  `PageSource` (absent → requested → arrived → resident): native files are read by I/O threads with positioned
  reads, the browser uses `fetch` with HTTP `Range`; adjacent pages are coalesced into one read, in-flight reads are
  bounded per source, failed reads are retried later. Page payloads stay compressed in the pool (`geometry.wgsl`
  decodes vertices on the fly in every consumer), so resident memory and streamed bytes are both 2.1–2.5× smaller
  (valley 118 → 48 MiB, cathedral 167 → 80 MiB, forest 85 → 40 MiB).
* **Software rasteriser** (`sw_raster.wgsl`). Clusters whose triangles are smaller than ~2.5 px are rasterised in
  compute: a workgroup per cluster, exact sub-pixel edge functions with the top-left rule. Without 64-bit atomics the
  depth and payload are written in two passes (32-bit `atomicMax` of depth, then the ids of pixels that won); a merge
  pass combines hardware and software results. Images with it on/off differ by at most 5/255.
* **Alpha-masked materials** (`alpha_mask.rs`, `alpha_mask.wgsl`; Nanite's "programmable raster"). A material with an
  `alpha_cutoff` is cut out where its base colour image's alpha is below it — in all four rasterisers (visibility
  buffer hardware and software, shadow pages hardware and software), so leaves and grilles also cast cut-out shadows.
  Culling bins alpha-masked hardware clusters into their own list, drawn by a second pipeline whose fragment shader
  samples and discards, so opaque clusters keep a trivial fragment shader and early depth testing. The compute
  rasterisers get an alpha-testing specialisation dispatched over the same clusters (each skips the other's, a
  workgroup-uniform branch), so the opaque kernels keep their registers and occupancy; it tests each covered pixel
  with the triangle's UV gradients. Scenes without masked materials skip the extra draws and dispatches. The masks live in a separate R8 array with **coverage-preserving mips** (each mip is rescaled to keep
  the fraction of texels passing the test, so distant foliage does not dissolve). Occlusion culling needs no special
  case: the HZB is built from the alpha-tested depth. In Lumen, masked surfaces are partially transparent: each
  (mesh, material) pair gets a **coverage** — the area-weighted fraction of its surface that passes the alpha test,
  sampled once on the CPU from the mesh UVs and the image (Sponza's foliage 41–53 %, the cathedral's choir screens
  26 %) — and near-field traces hit the instance with that probability (stochastic transparency, as UE5 dithers the
  distance-field traces of two-sided foliage; 64 rays per probe and the temporal filter average it), while
  distance-field soft shadows (local lights; the sun on mesh cards) let `1 − coverage` of the light through. Masked
  instances get mesh cards, so hits on leaves read their lighting, but stay out of the global distance field, whose
  ≥ 0.25 m voxels would merge leaves and chain links into solid blobs.
* **Two-pass occlusion culling** (as in Nanite) with persistent visibility bits; hierarchical page culling skips whole
  pages whose coarsest parent is already detailed enough. Cluster culling takes a 64-lane workgroup per surviving
  page, except for pages of at most 16 clusters (assembly parts, grass, small props), which list their clusters one by
  one as *cluster candidates* after the page work items, 64 to a workgroup (round 6: the forest's needle sprays have
  17 clusters over 6 pages, so their pages kept ~5 % of a workgroup's lanes busy); beyond the candidates' capacity a
  page falls back to a work item. The sixteen shadow clipmap levels, culled one after another, share one work buffer.
* **Materials** (`textures.rs`, `gbuffer.wgsl`). Textures are synthesised on the GPU (18 patterns: ashlar, marble
  tiles, wood, slate, stained glass, bark, leaves, moss, forest floor, …) into two texture arrays with full mip
  chains (no bindless in WebGPU); imported images are area-resampled into the same layout (base colour; normal, roughness,
  metallic), one layer per material. **Ambient occlusion** maps take the base colour's alpha in that layer: opaque
  materials never read it (alpha tests have their own mask array), so AO costs no memory, no atlas and no extra texture
  fetch (BC7 and the sRGB arrays keep alpha linear; sharing a BC7 block costs the base colour a little: 47.2 → 43.8 dB
  PSNR on DamagedHelmet, 54.4 → 53.2 dB on WaterBottle, the AO itself 42 / 53 dB). It reaches the lighting in the fraction of the G-buffer's
  shading-flags channel and, like UE's material AO, darkens indirect light only — Lumen's diffuse GI or the sky, and
  reflections through the specular occlusion of `sky visibility × AO` — never direct light. AO sampled with other
  coordinates than the base colour (e.g. the second UV set) or on blended materials (whose alpha is coverage) gets a
  layer of its own, read from its 512² array layer (not streamed: AO is low-frequency, and a second streamed lookup
  would cost every pixel of the resolve). Each texture slot (main layer, emission, occlusion) has its own **UV transform**: UV set 0 or 1, an
  affine `KHR_texture_transform` (folded with the material's tiling) and wrap modes applied in the shader
  (clamp-to-edge, mirrored repeat; arrays and atlases repeat), so streamed tiles are requested for the coordinates
  actually sampled, alpha tests use the base colour's, and per-triangle tangent frames come from the transformed UVs
  (rotated normal maps stay right). **Vertex colours** multiply the base colour (their alpha the coverage of blended
  materials). The second UV set and vertex colours are optional attributes of the compressed clusters
  (`encoding.rs`): a header bit adds 4 words (bit widths, UV1 and colour minima) and per-vertex offsets with just enough
  bits (RGBA8 colours, RGB on a square-root curve); meshes without them encode bit-identically to before. The
  rarer features — second UV set, wrap modes other than repeat, occlusion layers of their own — live in a second
  permutation of the resolve pipelines (`UV_EXTENSIONS`), compiled on first use for scenes whose materials need it;
  other scenes pay for transforms, AO and the optional-attribute check only (Sponza resolve 0.28 → 0.31 ms, best of
  8 at 1600×900).
  Pixels are classified by shading model per 8×8 tile and each model (standard, rock,
  terrain, metal, foliage) is resolved by its own pipeline over just its tiles (classification slots 0–7; slots 8–23
  are the extended pipelines of *Substrate-lite* and *Material graphs*). UV mapping uses analytic
  barycentric derivatives for proper filtering; per-triangle tangent frames replace stored tangents; triplanar
  mapping is available in world or object space. The G-buffer resolve and the hardware shadow page raster
  compile skinning only into a `SKINNING` permutation used while the scene has skinned instances (see *Skeletal
  meshes*), and the debug views' recolouring of the G-buffer only into a `DEBUG_VIEWS` permutation used while a
  debug view is shown: kept in the lit pipelines, a dozen debug colourings kept their inputs (cluster, triangle,
  instance record, barycentrics) alive across the material evaluation, and the resolve ran 1.4× slower (cathedral at
  1440p 1.01 → 0.73 ms, forest 0.99 → 0.85 ms).
* **Substrate-lite** (`substrate.rs`, `substrate_gpu.rs`, `substrate.wgsl`, `brdf.wgsl`, `resolve_extended.wgsl`,
  `subsurface.rs`, `subsurface.wgsl`; UE 5.7's Substrate reduced to one slab per material). `Material::slab`
  (`Slab`) extends the metallic-roughness base with the dielectric's F0 / F90 from an **index of refraction** and a
  **specular colour / factor** (`KHR_materials_ior`, `_specular`), **anisotropic** GGX (Burley's distribution,
  Heitz's height-correlated visibility, a tangent rotated by `anisotropy_rotation`), a **clear coat** (a second,
  colourless GGX interface with its own roughness and normal), a **sheen** (Charlie distribution with Neubelt's
  visibility, as `KHR_materials_sheen`), **subsurface scattering** (wrapped diffuse plus a screen-space diffusion)
  and **thin diffuse transmission** (leaves, paper), and an **unlit** flag. Layering is energy-aware: the coat
  takes `coat · F(n·v)` and `coat · F(n·l)` from everything below it, the sheen its directional albedo, both from
  the view and the light side. **Data path.** The G-buffer stays as it is for plain materials. A material with slab
  features (or a graph, below) is classified into an *extended resolve pipeline* — one per distinct (shading model,
  graph), so its 8×8 tiles are resolved apart from the plain ones (the classification bitmask grew from 8 to 16
  slots, the tile buffer accordingly). The extended pipeline writes one **rgba32uint slab texel** per pixel
  (features, material index, clear coat; base normal octahedral under a coat or the anisotropy tangent; sheen colour
  and roughness; F0 and the subsurface / transmission amount; 16 bytes, allocated only while some material is
  extended) and sets a `SURFACE_SUBSTRATE` shading flag, so lighting reads it only there. With a clear coat the
  G-buffer's normal and roughness are the coat's, so Lumen traces reflections for the top interface; the base's go
  into the texel. Per-material constants (F90, subsurface colour and radius, transmission colour, texture layers) sit
  in a 28 KiB uniform slab table. Voxel foliage pixels (*Nanite Foliage*) in an extended pipeline evaluate their
  material the same way, slab texel included. **Every lighting path, one BRDF.** The slab lobes are functions of
  `brdf.wgsl` (`shade_slab`, `evaluate_slab`, `slab_ambient_specular`), the file the classic light loop and
  MegaLights share; the deferred lighting (sun with its VSM, clustered local lights, transmitted back light with
  the back side's shadow, image-based light: the coat reflects Lumen's traced reflections, the base layer under it
  the prefiltered sky / GI along its own — for anisotropy bent — normal) and MegaLights' shading evaluate them. The
  passes that read slabs are *variants*: the same source composed with `substrate_read.wgsl` in place of
  `substrate_read_off.wgsl` (`shader::compose_variant`), compiled the first time a scene has slab materials and used
  only while it does — scenes without them run the unchanged shaders. **Subsurface scattering** (UE's subsurface
  profile, Jimenez's separable SSS): the lighting writes the diffuse light of scattering pixels apart, demodulated
  and attenuated by the fog in front of them; a horizontal compute pass and a vertical full-screen pass blur it
  with a two-Gaussian approximation of Burley's diffusion profile, per channel scaled by the material's scattering
  colour and radius (projected to pixels per depth), 7 jittered taps per side that skip other surfaces and depth
  steps, and the vertical pass adds it back × the diffuse colour through additive blending (storage textures cannot
  read-modify-write the HDR format), so specular light and texture detail stay sharp. `PageUp`
  (`RenderSettings::substrate`) switches slabs off — every material renders as its plain base — for A/B
  comparisons. Debug view **24** shows each pixel's features (clear coat blue, sheen magenta, subsurface red,
  transmission green, anisotropy yellow, specular cyan, unlit white, blended; brighter per layer; striped where a
  graph produced them; plain grey). **glTF**: `KHR_materials_clearcoat`, `_sheen`, `_ior`, `_specular`, `_unlit`
  and `_anisotropy` (factors) become slabs; their textures are rearranged on the CPU into two optional texture
  layers per material (sheen colour + sheen roughness + coat normal + coat roughness + coat; specular colour +
  specular), sampled from the 512² arrays with the main texture's coordinates; `KHR_materials_transmission` makes
  the material glass (the forward translucency pass, refracting), tinted by the base colour and
  `KHR_materials_volume`'s attenuation over its thickness. Lumen's surface cache keeps the average albedo (slabs
  don't change GI bounces). The extended path needs 17 sampled textures per shader stage (WebGPU's default limit is
  16; device creation requests up to 24 where the adapter offers them): with 16, slab materials render as their plain
  base. Cost: scenes without slabs render the same images (valley, cathedral, forest and Sponza differ from the
  previous build by 61–70 dB PSNR, as much as two runs of it differ) and cost the same; the gallery (eleven slab materials, three graphs, about half the screen
  extended) resolves in 0.41 ms and lights in 0.42 ms with slabs, 0.38 ms lighting with them off, the subsurface
  passes take 0.02 + 0.03 ms. Memory only while used: the slab texels 16 bytes per internal pixel (23 MB at
  1600×900) while some material is extended, the subsurface light and its diffusion 2 × 8 bytes per pixel while some
  slab scatters; the classification's tile lists doubled (1.4 instead of 0.7 MB at 1600×900).
* **Material graphs** (`crates/helios-material`, `material_graphs.rs`, `material_graph.wgsl`; UE's Material Editor
  and material instances). A `MaterialGraph` is plain serde data — nodes (`Op`) with canvas positions and the
  constants of their unconnected inputs, links from output to input pins by name — stored in scene files.
  45 node types: constants, scalar / vector / texture **parameters**, texture sample (the material's layers,
  streamed through the SVT path), triplanar, mesh UVs (both sets, tiling), world / object position, vertex normal
  and colour, time, camera vector, math (add … divide, lerp, power, min, max, clamp, saturate, one-minus, abs,
  sine, cosine, frac, floor, step, smoothstep, dot, cross, normalize, length, append, component mask), footprint-
  filtered fractal **noise**, **fresnel**, **panner**, reoriented **normal blend**, **if**, and the output (base
  colour, metallic, roughness, specular, tangent-space normal, emissive, ambient occlusion, the slab parameters:
  clear coat and its roughness, sheen colour and roughness, subsurface, transmission, anisotropy — and the
  **displacement**, a height or a vector, which drives Nanite tessellation: see there). **Compilation**
  (`helios_material::compile`): ids, links, pins, a single output and cycles are validated, value widths inferred
  (float to vec4, scalars broadcast) and every problem reported with its node ("node #4: Add: cannot combine a
  vec2<f32> with a vec3<f32>"); the result is one WGSL function `graph_eval(m, input, params, vertex_color,
  defaults) -> GraphOutput`, every operation parenthesised and every identifier generated (Tint-safe), plus the
  parameter slots and the connected outputs. **Derivatives without hardware**: the resolve is a compute shader over
  the visibility buffer, so values that feed a texture coordinate, a triplanar or noise position carry their screen
  derivatives through the math by the chain rule (forward-mode differentiation from the resolve's analytic UV and
  position derivatives) — texture filtering, the SVT's tile requests and noise filtering work in graphs; other
  values pay nothing. **Pipelines**: a graph's code replaces `material_graph_off.wgsl` in the extended resolve,
  compiled at runtime (WGSL shader modules are created in the browser too) and cached by the code's FNV hash, so
  editing a graph compiles one module, identical graphs share one, and moving nodes compiles nothing.
  **Material instances**: `Scene::set_material_parameter` overrides a parameter for one material; values live in a
  16 KiB uniform block (one vec4 per parameter, the material's offset in its slab record), so changing them
  recompiles nothing (the gallery's two water spheres are one graph with different colours). Unset texture
  parameters sample the material's own texture, so a graph assigned to an imported material picks up its images.
  Unit tests compile every node type (on a texture-coordinate path, so its derivative code too) and every preset
  into the extended resolve and validate it with naga and the Tint bitwise-precedence check. **Displacement**: the
  output's Displacement pin (a height along the normal) or its VectorDisplacement pin (a world-space offset in
  units of the material's magnitude, its length clamped to 1: overhangs; one of the two, both is an error) compiles
  to a second function, `graph_displacement(m, input, params) -> f32` or `-> vec3<f32>`, from the nodes it depends
  on (sharing the node code and the parameter slots; nodes feeding only one of the two functions stay out of the
  other), which the tessellator calls per micro-triangle vertex. Its inputs (`DisplacementSurface`: mesh UVs, world
  / object position, normal, time) carry the size of a dice segment as their "derivatives", so texture samples (the
  512² arrays, not streamed tiles) and noise are filtered for the micro-triangles. `Time` (and a panner left on the
  clock) is allowed: the compiler reports the displacement as animated (`CompiledGraph::displacement_animated`,
  next to `displacement_kind`), which makes the material's instances dynamic shadow casters, and reads the clock
  from the function's input so the renderer can evaluate the previous frame's surface for motion vectors; a
  static graph pays for none of it. Round 6: the compiler also bounds how fast an animated displacement moves
  (`CompiledGraph::displacement_speed`, an interval and time-rate analysis of its nodes, `helios_material::motion`)
  for each material instance's parameter values, and a **TransformVector** node (UE's Transform: World, Object =
  Local, Tangent) turns directions between spaces in both functions (see *Nanite Tessellation*). Still rejected,
  with an error on the node: `CameraVector` / `Fresnel` (every
  view must displace alike), vertex colours, UV set 1 and the triplanar tangent normal (the tessellator does not
  read them). Texture sample and triplanar nodes gained a `Height` output (a
  procedural texture's height). The node editor is part of the browser editor (see *Material graph editor*).
* **Runtime virtual texture** (`rvt.rs`). The layered landscape material (height-blended texture layers driven by
  slope, altitude and noise) is rendered on demand into 128² pages of a 128k² virtual texture, cached in a 24×24-page
  atlas by a generic GPU **page cache** (request → keep/free → allocate → dirty list). It cut the valley's material
  pass from 0.53 to 0.21 ms.
* **Streaming virtual textures** (`svt/`, `svt.wgsl`). Image layers keep their 512² array layer as the always-resident
  coarse levels; the finer levels up to the images' native resolution (≤ 4096²) form a virtual texture of 128² tiles
  with 8-texel borders, streamed into three atlases (sRGB albedo; normal xy; roughness, metallic) as UE5's Streaming
  Virtual Textures do. Where the G-buffer resolve samples a layer with mesh UVs (base colour, normal, roughness,
  metallic, emission), it computes the anisotropic lod in native texels; one pixel per 4×4 block (rotating each
  frame) writes the tile of its finer trilinear level into a feedback texture that is read back asynchronously. The CPU loads requested tiles with their coarser ancestors, coarse first and within a
  per-frame budget, cutting them from mip pyramids built once at load (linear-light colour, Toksvig roughness), and
  evicts least recently needed tiles. A page-table texture maps every virtual tile to its finest resident
  ancestor-or-self plus a minimum lod, so sampling blends trilinearly between resident levels (the array is the level
  above the coarsest tile), never misses, and fades new tiles in instead of popping. Hardware anisotropic filtering
  works inside a tile because gradients are limited to its border. Sponza (24 image layers of 1024²): ~270 of 1536
  tiles resident for the atrium view in a 63 MiB pool (`RendererConfig::texture_pool_bytes`), material pass
  0.20 → 0.23 ms at 1600×900; triplanar mappings and procedural layers keep using the arrays.
* **Block-compressed texture tiles** (`svt/bc.rs`). Where the GPU offers BC formats (desktop browsers, D3D12,
  Vulkan, Metal on Macs) the tile atlases are BC7 sRGB (base colour or emission, with alpha), BC5 (normal xy; z is
  reconstructed) and BC5 (roughness, metallic): 3 instead of 8 bytes per texel, so the same 64 MiB pool holds 1024
  instead of 400 tiles (Sponza with a 16 MiB pool: 256 instead of 100, i.e. the atrium view's tiles stay resident
  instead of falling back to the 512² arrays). WebGPU can't write compressed textures from shaders, so a CPU
  encoder fits each 4×4 block — BC7 modes 6 (one RGBA line, 16 levels: no banding), 1 (two partitioned lines for
  edges between unrelated colours) and 5 (separate alpha line for coverage masks); BC5 as two least-squares BC4
  lines. As UE does for streaming virtual textures, tiles are compressed ahead of time rather than per upload:
  natively all levels are encoded on worker threads at load (Sponza: 1.6 M blocks per atlas in ~0.9 s on 12
  threads), and an upload is then a copy of blocks (~0.1 ms per frame); in the browser (no threads) a tile's blocks
  are encoded on first request — resumable, within a 2 ms per-frame budget — and spare budget encodes ahead,
  coarse levels first. Tiles, borders and 8-texel encoding chunks are block-aligned, so border blocks are
  bit-identical to the neighbouring tiles' (no seams) and the shader is unchanged apart from a third atlas.
  Quality (PSNR vs the uncompressed tiles): BC7 42–48 dB on Sponza / DamagedHelmet base colour (55 dB emission;
  Pillow's BC1 encoder gets 34–38 dB on the same images), BC5 46–54 dB normals, 45–67 dB metallic-roughness; final
  frames differ from uncompressed ones by ~51 dB with no visible block artefacts. Devices without BC (mobile:
  ETC2 / ASTC only) and `--no-texture-compression` use RGBA8 / RG8 / RG8 atlases with the same layout.
* **Virtual shadow maps** (`vsm.rs`). A clipmap of 16k² virtual shadow maps (≈1 mm texels near the camera) paged into
  a 4k² physical atlas through the same page cache. Pixels request the pages they sample; only new or invalidated pages
  are rendered — each level's clusters are culled against that level's dirty-page rectangle and bit mask, expanded into
  (cluster, page) draws and rasterised into their tile. With a static sun, almost nothing is re-rendered per frame.
  As in the main view, clusters whose triangles are tiny in the level's texels are **software-rasterised**: one
  32-bit `atomicMax` of depth per texel into a u32 mirror of the atlas (shadows need no payload pass), which the
  page clear copies into the depth atlas before the hardware clusters are drawn on top. Hardware page draws cull
  triangles that miss their page in the vertex shader (no viewport arrays: otherwise large clusters were
  rasterised over every page they touch, once per page); where the device has WebGPU's optional `clip-distances`
  feature, four clip distances clip every page draw to its tile (`vsm_draw_clipped.wgsl`), so triangles larger than
  a page — common near local lights, and above all those crossing a light's near plane, which project across the
  whole atlas — no longer rasterise discarded fragments (a moving candle light near the camera: page raster 4.8 →
  1.3 ms). Streaming, wind, skeletal animation and moved instances invalidate only the cached pages under
  the instances concerned (see *Wind*, *Dynamic world*).
  **Static and dynamic casters** are cached separately, as in UE5's "static separate" page cache: casters the wind
  moves this frame, `Mobility::Movable` instances and skins whose pose changed are *dynamic* (one definition,
  `dynamic_caster.wgsl`), all others *static*. A page that must be refreshed only because dynamic casters
  moved keeps the depth of its static casters as a *static layer* in its tile of the software raster's u32 mirror —
  otherwise just scratch space between the software raster and the page clear, so the layers cost no memory of their
  own — and its clear writes that layer instead of the far plane; culling drops static instances at the instance test
  (each level keeps a second dirty-page mask and rectangle for pages rendered in full, each local light a second box),
  only dynamic clusters are expanded and drawn, and their software-rasterised ones go to one of 256 scratch tiles after
  the atlas' tiles (16 MiB), which the page clear max-merges over the static layer. A page gets its static layer at
  its first such refresh: it is rendered in full once more, static casters first, whose depth is then copied into the
  mirror (`vsm_static.wgsl`) before the dynamic ones are drawn. Anything that changes static casters — the sun or a
  light moving, a structural scene change, a static instance being moved (its footprint only), streaming of static
  instances, the wind being switched on or off (per instance: no global invalidation), voxel LODs being toggled —
  frees the page, so it is rendered in full. A caster fading into or out of the wind as the
  camera approaches or leaves it is at rest then: the cached depth stays right, so the pages under it only drop their
  static layer and are rendered in full at their next refresh (re-rendering them at once doubled the shadow work of
  a camera flying through the forest). The sampled atlas always holds the merged depth, so sampling is unchanged. Sun
  and local light pages alike.
* **Wind** (`wind.rs`, `wind.wgsl`; UE5's foliage World Position Offset, as Nanite supports it). `Scene::wind`
  (direction, strength, gusts rolling through the vegetation as travelling waves, turbulence) moves materials with a
  `Material::wind` response: the whole plant *bends* about its origin (∝ height², a cantilever), branches *sway*
  (∝ distance from the plant's axis) and leaves *flutter*; rocks and buildings have none. The offset is a pure function
  of the vertex's rest position, the instance transform and the time, evaluated identically by all four rasterisers
  (visibility buffer hardware/software, shadow pages hardware/software) and by the G-buffer resolve, so clusters and
  LOD levels sharing a vertex move it identically (no cracks), and wood and leaves sharing a transform move as one;
  motion vectors evaluate the previous frame's offset, so TSR and motion blur follow it. Instance, page and cluster
  bounding spheres are padded by a bound of the displacement for frustum, HZB and shadow-page culling (a unit test
  checks the bound against the displacement). Like UE's *WPO disable distance*, an instance stops moving beyond
  `Wind::disable_distance` (80 m in the forest) or closer, where its motion bound would cover fewer than 4 pixels (grass
  before trees); it fades out over the last quarter, so nothing pops. Cached shadow pages under moving instances
  re-render their dynamic casters over their cached static layer (see *Virtual shadow maps*) — but only once their
  geometry may have moved 2 texels since the page was rendered (a speed bound of the displacement), so fine levels
  refresh every frame or two, coarse ones every few frames and levels whose texels dwarf the motion never. Lumen's
  distance fields stay static (the motion is far below their voxels). `L` toggles it. Culling and page expansion test
  a dirty-page bit mask besides the dirty rectangle, without which the scattered refreshed pages cost another ~1 ms.
  Forest (high) at 1600×900, default view, wind on, versus re-rendering the static casters of refreshed pages too (as
  before the static layers; interleaved runs, RTX 3090): ~155 pages refreshed per frame, all over their static layer;
  shadow triangles 4.5 → 3.7 M, (cluster, page) draws 93 k → 77 k, shadow page passes 1.83 → 1.62 ms (the wind adds
  ~1.6 instead of ~1.8 ms of shadow work). Almost every shadow caster in the forest sways (trees included); only
  terrain, rocks and logs are static, so the remaining cost is that of the moving vegetation. A camera flying through
  the forest renders the same number of pages, with 70 k instead of 80 k draws.
* **Skeletal meshes** (`skin.rs`, `skinning.wgsl`, `skin_palette.wgsl`; UE 5.5's Nanite skinned meshes). A
  `SkinHandle` is a joint palette (`Scene::add_skin`, `Scene::set_skin_pose` every frame,
  `Scene::set_instance_skin`); each matrix maps a rest vertex to its posed position in the instance's object space.
  Skinning — the linear blend of the vertex's joint matrices by its quantised weights, zero weights skipped, in slot
  order — is a pure function of the decoded rest vertex, its weights and the palette, evaluated by all four
  rasterisers (visibility buffer and shadow pages, hardware and software) and by the G-buffer resolve, as the wind's
  offset is: clusters and LOD levels sharing a vertex put it on the same bits, so cuts stay watertight under animation
  (the geometry tests pose a three-joint limb with four-influence vertices and check every LOD cut, full and partial
  residency, for open edges on the posed positions). Normals use the blended matrix's cofactor; motion vectors read the
  previous frame's palette (per-corner previous positions interpolated with the barycentrics — exact, skinning is
  affine per triangle), so TSR and motion blur follow the limbs (TSR vs. no-TAA on CesiumMan's walk at 100 % and 50 %:
  differences only along edges, no trails). The palettes live in one `Rgba32Float` texture (`textureLoad`, so no
  shader stage gains a storage buffer; the G-buffer resolve was at WebGPU's 8): per skin a header and six texels per
  joint (current and previous rows), rewritten each frame (24 joints: 2.3 KiB); instances find theirs through
  `GpuInstance::flags` bits 12..32 (no layout change). **Culling bounds**: per frame and skin the CPU derives, from
  per-joint rest spheres of the vertices each joint influences, a *posed sphere* (every joint's posed rest sphere — a
  blended vertex lies in the convex hull of its joints' transforms of it) and a *reach* `max |P c − c| + ‖A − I‖₂ r`
  (spectral norms in closed form) that bounds every vertex's displacement; frustum, HZB and shadow-page tests use the
  rest cluster / page / instance sphere padded by the reach, or the posed sphere where that is smaller (unit tests
  check both against random poses with scale). The LOD cut keeps the rest spheres, which keeps the DAG's monotonic
  errors (crack-free). **Shadows**: a skin whose pose changed is a dynamic caster in the static/dynamic page cache, as
  vegetation in the wind: cached pages under it re-render their dynamic casters over the static layer once it may have
  moved 2 texels, bounded per frame by the joints' motion `|ΔP c| + ‖ΔA‖₂ r`; the footprint is the rest sphere padded
  by the largest reach so far (it covers every pose a cached page may still show); a paused skin returns to the static
  layer. **Lumen**: as in UE, skinned meshes get no distance field or cards (the demo bakes them without); screen
  traces still see them. Skinned meshes never get voxel LODs (voxels carry no weights). **Animation**
  (`helios_asset::animation`): glTF skins, node hierarchy and clips (translation, rotation, scale; `STEP`, `LINEAR`
  with shortest-arc slerp, `CUBICSPLINE` Hermite), an `AnimationPlayer` (clip, time, speed, looping, pause,
  crossfade). The palette is `inverse(mesh node) · joint · inverse bind matrix`, i.e. relative to the node that places
  the mesh, so the rest-pose bounds and LOD stay meaningful while `node · palette` is exactly glTF's pose. Debug view
  **17** colours each joint and blends the colours by weight (grey: unskinned). **Permutations**: the G-buffer resolve
  and the hardware shadow page raster compile the skinning code only into a `SKINNING` permutation used while the
  scene has skinned instances (like `UV_EXTENSIONS`): unconditionally it took the forest's resolve from 0.39 to
  0.74 ms and its page raster 5 % slower, although no pixel there is skinned. Instance, page and cluster culling are
  specialised the same way (two pipeline sets): with ~26 000 assembly parts in the forest, the palette lookups cost
  registers in passes that test every part (forest high, medians of 8 interleaved runs: cull instances 0.36 →
  0.31 ms, pages 0.30 → 0.27, clusters 0.55 → 0.53; images unchanged). The visibility rasterisers and page expansion carry it at no measurable cost.
  Cost (shared GPU; min / median of 5 interleaved runs, BrainStem — 61 k triangles, 59 meshes, 18 joints — animated
  vs. the same model imported as a static mesh with `--no-skinning`): main-view raster unchanged (hw 0.02, sw
  0.06–0.07 ms), cull 0.20 → 0.23 ms, material resolve 0.31 → 0.35 ms (the skinning permutation; paused: also 0.35),
  vsm expand 0.09 → 0.12 ms, shadow page raster (hw + sw + resolve) 0.02 → 0.29 ms — the moving caster's pages
  re-rendered every frame, which a paused model doesn't (0.02 ms: its pages stay cached); GPU frame 4.16 → 4.41 ms
  (paused 4.14). Compressed payload: +2.5 bytes per cluster vertex on the Fox, +3.3 on CesiumMan (+19 / +25 %), +0.3 on
  the mostly rigid BrainStem (+3 %). Scenes without skins: every pass within ±0.02 ms of the build before skinning,
  images within run-to-run noise.
* **Dynamic world** (`motion.rs`, `vsm_motion.rs`, `lumen_dynamic.rs`, `dynamic_debug.rs`; UE5's movable
  primitives). Any instance can move every frame without invalidating the world around it. `Scene::set_transform`
  stamps the instance with a version instead of bumping the scene's instance version, and `Scene::set_mobility`
  declares it `Mobility::Static` (placed once; moving it is a one-off edit) or `Mobility::Movable` (moves often) as UE's
  component mobility. Each frame a `MotionTracker` turns the stamps into the list of moved instances with their
  previous transforms (a scan of the instances only in frames where something moved), and every subsystem applies
  just those moves; adding or hiding instances, reassigning materials, toggling shadow casting or new geometry are
  still structural changes that rebuild everything once. **GPU scene**: only the moved records are rewritten (one
  `write_buffer` per run of consecutive instances, previously all instances every frame something moved), with the
  previous frame's transform as `prev_world`, so motion vectors (TSR, motion blur) follow rigid motion; an instance
  that stops is rewritten once more to drop its motion. Temporal history is never reset by moves. **Virtual shadow
  maps**: movable instances are *dynamic casters* like wind-animated vegetation (the `MOVABLE` instance flag), so they
  stay out of the pages' static layers. The CPU writes each moved caster's footprint into the invalidation list
  before the GPU appends its streaming and wind footprints: the light-space rectangle around the instance's oriented
  box in its old and new pose (one swept footprint for a small move, one per pose for a jump) and its bounding
  spheres for local lights' faces and mips (a skinned instance's padded by its skin's largest reach). A long box is
  cut into up to 8 roughly cubic slices along its longest axis, each with a footprint of its own (a chandelier's
  iron, a 2.8 m ring on a 12 m chain: five of 2.4 m instead of one 6.3 m sphere over most of every nearby candle
  light's cube). A movable caster's footprints also carry a speed bound per slice — its largest corner displacement
  per frame, held with a slow decay — so, as for the wind, a page is refreshed once the slice may have moved 2 of its
  texels since the page was rendered: the slices near a pendulum's pivot hardly ever, the swinging ring's fine pages
  every frame and coarse mips every few. Skipped pages may show the caster up to 2 texels off, so the GPU pads move
  footprints by that much, and a caster that stops is refreshed once more at rest. Pages under a movable caster
  re-render only their dynamic casters over their cached static layer; pages under a static instance that was moved
  (or a jump) are freed and rendered in full. At rest a movable instance costs nothing. **Lumen**: the instance's
  distance-field record is rewritten in place and the 8³-voxel bricks of each global-field clipmap level under its
  old and new box (plus the narrow band) are recomposed — distance, albedo and the per-voxel closest card instance —
  by two new passes over a
  brick list (`cull_listed`, `compose_listed` in `lumen_sdf.wgsl`, one workgroup per brick), within a budget of 1024
  bricks per frame (the rest queue up, finest level first). A level recomposes under an instance only once some
  point of it moved a quarter of that level's voxel since the level last composed it, so the 0.25 m level follows a
  swinging chandelier every few frames and the 2 m level rarely. Movable instances stay in the global field, as in
  UE5 (it updates movable primitives' distance fields rather than excluding them): a fast mover only costs more
  bricks per frame and, beyond the budget, lags a few frames. The near field rewrites the instance's record and
  rebuilds its culling grid only when the instance's box crossed into other 2 m cells; mesh cards move with their
  instance (its transform in the card records; nothing is recaptured) and its card pages are relit first, within half
  the card lighting budget so the round-robin refresh keeps going. The voxel surface cache and the radiance cache are
  not invalidated: they are refreshed continuously (a quarter of the voxels and an eighth of the probes per frame)
  from the updated field. **Translucency** re-places moved instances' candidates (and water bodies when a water
  instance moves) instead of rebuilding its lists. Growing the scene bounds (the shadow clipmap's depth range) is the
  one move that invalidates every cached shadow page. `RenderSettings::per_instance_invalidation = false`
  (`--full-invalidation`) restores the old behaviour (every move a structural change) for comparisons. Debug view
  16 (*Invalidation*) shows what moves cost this frame: movable instances orange, sun shadow pages rendered this
  frame hatched red, recomposed distance-field bricks (finest level at each pixel) cyan with their outlines; the HUD
  has a "moving (0)" line with the moved instances, shadow casters and recomposed bricks. The demo scenes move (`0`
  pauses): the cathedral's chandeliers swing, the valley's sculpture turns and its raft drifts.
  Cost (high detail, shared GPU; minima of 3 interleaved runs, per-pass minima where whole-frame totals were swamped).
  Cathedral, sun shadows only (local lights' shadow maps off): GPU frame 5.88 ms at rest; moving 1 / 3 / 11
  chandeliers (3 / 9 / 33 instances) 6.04 / 6.05 / 6.04 ms with 7 / 32 / 57 sun pages re-drawing only their dynamic
  casters, versus 7.87 / 7.63 / 8.03 ms with full invalidation (all 123 cached pages rendered in full, page raster
  0.83–0.92 ms instead of 0.01–0.02 ms, and all 4 distance-field levels recomposed every frame, 0.93–0.98 ms, instead
  of ~24 bricks). With the chandeliers' candle lights' shadow maps on, the lights moving with them dominate: 1 / 3 /
  11 chandeliers render 96 / 314 / 520 pages (88 / 288 / 440 dynamic-only), page raster 0.28 / 1.11 / 3.62 ms,
  versus 609 full pages and 8.6 / 7.9 / 13.5 ms with full invalidation (138 k instead of 391 k (cluster, page) draws
  for all eleven); one uncontended pair: GPU frame 14.7 ms moving vs 25.8 ms full invalidation. Valley (sculpture +
  raft): 5.00 ms at rest, 5.66 ms with both moving (25 dynamic-only pages) vs 6.44 ms full; the sculpture alone 5.28
  vs 7.89 ms. Distance-field bricks: without the quarter-voxel threshold the chandeliers recomposed ~390 bricks a frame
  (0.42 ms), with it 24–120 (≤ 0.07 ms). CPU: a scan of the instances in frames where something moved plus a few
  record writes. Temporal stability (valley, static camera, pixels changing by more than 24/255 outside the movers and
  their shadow pages): 0.049 % moving vs 0.042 % paused (noise: water waves). Scenes without moving instances render
  as before. Slices and speed bounds (2560×1440): the swinging cathedral's static candle lights refreshed ~820 pages a
  frame under the chandeliers (`vsm expand` 0.68 ms), now ~25 (0.23 ms); the valley's sculpture and raft 38 → 15 sun
  pages. The chandeliers' own lights' shadow maps are no longer re-rendered every frame (round 4: distance-field
  shadows while they move, cathedral GPU frame 14.86 → 10.17 ms at 100 %; round 5: cached around an origin near the
  light, 10.8 ms, *Local light virtual shadow maps*).
* **Nanite Foliage** (`helios-geometry/src/voxel.rs`, `assembly.rs`; `helios-render/src/assembly.rs`,
  `voxel.wgsl`, `voxel_raster.wgsl`, `vsm_voxel_raster.wgsl`, `voxel_resolve.wgsl`, `assembly.wgsl`; UE 5.7's
  experimental Nanite Foliage). Aggregate geometry — canopies, needles, fronds, grass — cannot be simplified as
  triangles without deleting pieces: component pruning thins it, and distant forests lose coverage and colour (the
  conifers turned into reddish skeletons of bare branches). Two parts:
  **Voxel LODs.** Meshes built with `BuildConfig::voxels` (the forest's foliage, ferns and grass) give every
  DAG group a voxel candidate: the *full-detail* triangles under the group, voxelised on a grid as fine as its
  budget allows (`voxels_per_triangle`, 4 per triangle the group keeps). Each voxel stores what the geometry
  inside it does to light, not its shape: its coverage seen along each axis — each *piece* (a connected leaf,
  needle or blade) covers its projected share of the face (`max` of the areas facing either way, so a two-sided
  or folded leaf counts once), independent pieces multiply what they leave open — an octahedral normal blending
  the leaves' principal orientation with the canopy's outward direction (the coverage gradient over the group),
  and its brick's mean texture coordinate and spread. Seen from a direction, a voxel's coverage is the leaves'
  projection over the cube's footprint, in optical depth (`sqrt(Σ(d_i τ_i)²) / Σ|d_i|`, exact for leaves facing
  an axis and for isotropic tangles; adding the axes linearly counts a leaf seen diagonally up to √3 times).
  Clusters hold up to 128 bricks of 4³ voxels (occupancy bits, position, first voxel, uv) and one 32-bit word per
  voxel. Where the voxels' error is below the simplified triangles' the group switches to voxels — at the
  triangles' error, so the LOD switches where it did — and every output cluster gets *voxel alternates* standing for
  the same source triangles, with the same LOD values; every group above is voxelised too. Both representations
  stay in the DAG, so `Insert` switches between the voxel cut and the triangle-only cut of the same DAG instantly,
  and all its invariants (monotonic errors, one side of every group boundary, page dependencies) hold in both; voxel
  groups' member sets get pages of their own per mode, so a mode never streams the other's data. Voxel clusters are
  always rasterised in compute: a workgroup per cluster takes its voxels (up to 8192) in turn, projects each cube
  from its centre (moved by the wind like a vertex there: voxels shift rigidly, they don't flutter) and the
  instance's half-voxel axes, rejects voxels between pixel centres with a conservative bound, and covers a pixel
  where one of its three front faces does, at that face's depth, with the probability of the voxel's coverage along
  the view (stochastic transparency: interleaved gradient noise animated per frame and decorrelated per voxel,
  averaged by TSR). It shares the software rasteriser's depth / payload passes and target (payload = cluster slot,
  brick), so the merge with the hardware and software triangle paths is unchanged; the G-buffer resolve finds the
  voxel from the merged depth (the brick cell under the hit point, with its exact wind offset) and shades it with
  its normal (facing the viewer like a two-sided leaf) and the material sampled at the brick's uv with its spread as
  the footprint (the leaves' average colour) — through its shading model's pipeline, or the extended one (slab,
  graph) where its material has one. Shadow pages (sun clipmap and local light faces) rasterise voxels the same way
  in page texels, with the triangles' slope-scaled bias and hashed, static noise per absolute texel (cached and
  refreshed pages agree). Culling treats voxel clusters like triangle clusters (frustum, HZB, LOD, streaming);
  `.hvm` is version 4. **Assemblies** (`Scene::set_assembly`): a mesh can be made of *parts* — meshes placed by
  transforms in its space — so the forest's conifers share six needle sprays per species (one per branch span, as
  many copies as branches) instead of each species baking all its needles (`vegetation::conifer_assembly`;
  `broadleaf_assembly` builds broadleaf trees from leaf sprays the same way). The parts are resolved on the GPU
  before culling: each instance of an assembly tests its *base* — one ordinary virtual mesh built from the parts at
  their roots, merged in place and simplified (and voxelised) further, its errors offset by the parts' worst root
  error so they continue where the parts end — with the main view's LOD rule; where the base's finest level is too
  coarse, the instance writes one record per part into part slots after the scene's instances (its transform times
  the part's, the assembly's bounds and plant origin for the wind, and per-(assembly instance, part) visibility bits
  for two-pass occlusion) and flags its base (`EXPANDED`) so culling skips it. Every view culls the same parts, so
  shadows always match the view. Culling is hierarchical (round 6, [`foliage-culling.md`](docs/round6/foliage-culling.md)):
  views never test the part records as instances; each assembly instance owns one more *culling record* between the
  scene's instances and the part slots (`PART_LIST`), which the expansion fills with what its parts share (the
  assembly's bounds, wind and plant origin, flags) and hides when the base is drawn. Instance culling tests that
  record once per view, and its batches are its parts' pages — `stride` lanes per part, the most pages of any part —
  each lane resolved to its part's page and tested with the part's transform, so a conifer's 176 sprays of 6 pages
  are culled in 17 page workgroups instead of 176, and their small pages' clusters as cluster candidates. Parts inherit `MOVABLE`,
  hidden bases expand nothing, and part slots record their owning instance (`part_owner.wgsl`), so picking returns
  the tree and the selection outline covers its parts. Up to 32768 part slots: when the parts requested near them,
  the expansion's LOD threshold is raised just enough (CPU feedback with hysteresis, like the culling budget) so
  the nearest assemblies keep their parts. Debug views 22 (voxel clusters coloured, triangles grey) and 23 (assembly
  parts coloured, bases light grey). The wind takes a plant origin from the instance (spare `w` of the normal
  matrix), so a part sways about its tree's origin exactly as the baked foliage did.
  Measured (forest high, shared GPU, interleaved runs, minimum of 4, before → after): GPU frame 8.38 → 10.09 ms —
  voxels on / off (`Insert`) 10.09 / 9.40 ms: software raster + merge 1.47 → 1.99 ms, material resolve 0.38 →
  0.45 ms, shadow software raster 0.52 → 0.72 ms; the conifer assemblies (≈ 26 400 part instances) cost the rest,
  mostly culling (0.48 → 0.91 ms, late pass 0.06 → 0.12 ms; the non-skinning culling variant later took
  0.02–0.05 ms per culling pass back; round 6's culling records and cluster candidates cut the parts' culling to
  ≈ 0.1 ms: forest, 2560×1440, `cull` + `cull (late)` 1.19 → 0.56 ms at 100 %, 1.14 → 0.52 ms at 67 %, images
  unchanged) and the parts' finer triangles. Default view: 5.23 M triangles → 3.97 M triangles + 4.3 M voxels (shadows
  4.46 M → 2.7 M triangles + 2.8 M voxels). Geometry 43.3 → 39.6 MiB in RAM and in `.hvm`; conifer needles 7.9 MiB as
  baked meshes with voxels → 1.1 MiB as assemblies, baked in 0.5 instead of 2.0 s (whole scene 6.9 → 8.2 s:
  voxelisation). Broadleaf trees as leaf-spray assemblies would save 6.6 → 2.9 MiB for the same ≈ 0.7 ms of GPU time
  (their 144-triangle sprays are poor parts), so the forest bakes them whole. Distant canopy (albedo averaged over 16
  frames from 30 m above the forest, `--eye 0,30,0 --at 200,0,-60`, canopy 150–400 m away, vs. a full-detail
  reference at a 1/8 px LOD threshold): mean colour error +2.0/+1.3/+1.0 % → −0.2/−0.5/−1.2 % (RGB), 16×16-block error
  1.75 → 1.53/255, 40×40 1.19 → 0.88/255; the distant conifers keep their needles instead of showing bare red
  branches. Coverage tests on a leaf cloud: within 3.4 % of the full mesh along each axis at 30–2000 m (9 %
  diagonally at the finest voxels), where the triangle cut is off by up to 13 %. Temporal stability (2560×1440,
  static camera, % px > 24/255): 0.077 → 0.083 % with wind, 0.027 → 0.033 % calm.
* **Nanite Tessellation** (`tessellation.rs`, `displacement.wgsl`, `displacement_height.wgsl`, `tess_core.wgsl`,
  `tess_queue.wgsl`, `tess.wgsl`, `vsm_tess.wgsl`, `tess_resolve.wgsl`; UE 5.4+'s Nanite tessellation as Brian Karis
  describes it). A material's `Material::displacement` (`Displacement`: a height channel of a texture layer — a
  procedural `TexturePattern`'s height, or any channel of an image layer — sampled through the material's own
  mapping, a magnitude and a centre) moves its surface along the normal at render time; the meshes stay smooth
  and small, the detail comes from the texture. **Culling**: displaced instances carry a flag and every bounding
  sphere (instance, page, cluster; frustum, HZB, shadow pages and the shadow page expansion) is padded by the
  material's displacement bound `magnitude · max(centre, 1 − centre)`, as for the wind; the LOD cut keeps the
  undisplaced errors (as UE). Their triangle clusters always go to the software list. **Patches**: every triangle
  of a displaced cluster is a patch, a triangle in its base triangle's barycentric space with fixed-point
  corners (15 fractional bits, so midpoints are exact). Per edge, a tessellation factor is the *undisplaced*
  edge length projected to pixels as if it faced the camera (so displacement towards the viewer still gets
  diced), over the **dice rate** (`RenderSettings::dice_rate`, 2 px as in UE); edges are oriented by their end
  points' positions, so both triangles of an edge — across clusters and LOD levels, whose shared vertices are
  bit-identical — compute the same factor. **PatchSplit**: patches with an edge factor above 8 are split at the
  midpoints of exactly those edges (one, two or three: two, three or four children — red-green refinement), so a
  split depends on the edge alone and neighbours agree; the children go through a global queue in a storage
  buffer, ping-ponged between split passes. WebGPU has no forward-progress guarantee, so instead of persistent
  threads each pass is an indirect dispatch sized by a one-thread pass (as UE does on all platforms), and the
  number of passes follows the deepest split of recent frames (one spare; patches still too coarse after the last
  are diced as they are). **Dicing**: an edge with factor c gets 2^ceil(log2(c/√2)) segments (lengths within √2
  of the dice rate), the inside a uniform grid of F = 2^level segments per side with the finest edge's level
  (F = 1, 2, 4, 8 — up to 64 micro-triangles per patch); grid vertices on an edge with fewer segments snap to its
  segments (a zipper of degenerate triangles). Vertices new at a level are **geomorphed**: they grow out of the
  coarser segment they split, from its midpoint to their displaced place as the factor goes from 2^(level−1) to
  2^level — in displaced space (as CDLOD does for terrain): morphing the parameter instead slides the
  displacement across the surface, which swims. Both sides of an edge snap and morph its vertices alike, so the
  micro-mesh is watertight and changes continuously as the camera moves (a CPU mirror of the split and dice
  logic checks shared edges vertex for vertex — parameters and displaced, morphed positions — on a jittered mesh
  from near and far eyes, and that edge vertices stay on the previous frame's displaced edge while the eye moves
  through doublings and splits: < 0.01 with the morph, 0.031 without it for a 0.15-amplitude displacement of
  unit edges). **Displacement**: a vertex
  moves along its interpolated, normalised normal by `magnitude · (height − centre)`; the height is filtered for
  the size of a dice segment at the vertex (UE picks the displacement mip from the mesh's UV density likewise:
  the dicer has no screen derivatives; `Displacement::uv_density_of` measures a mesh's), read from the 512²
  array layer, a pure function of the vertex, so every pass and both sides of an edge agree. The wind applies
  after the displacement (WPO on the displaced position). **Raster**: patches are diced by grid level — one
  thread per patch for F ≤ 2, 16 or 64 threads per patch for F = 4, 8 (vertices displaced and projected into
  shared memory, then one thread per micro-triangle; every grid vertex is displaced once and a morphing vertex
  mixes its own result with its coarser segment's end vertices' — grid vertices of the same patch — instead of
  evaluating three heights) — and rasterised with the software rasteriser's rules into
  its target and two passes (depth `atomicMax`, then payload), alongside the other software clusters. All
  micro-triangles go through compute, as in UE (whose patches never take the hardware path either): the dice
  rate keeps them at a few pixels; ones stretched over 4096 pixels by a cliff in the height are dropped, and ones
  crossing the near plane are culled. **Visibility payload**: unchanged — `(cluster slot << 7 | base triangle)`,
  as for any software triangle — so material classification, picking, selection and Lumen's screen traces see an
  ordinary id. Next to it the payload pass stores the **barycentrics on the base triangle** of the point
  that was displaced to the pixel — the grid vertices' own, interpolated perspective-correctly over the
  micro-triangle, two 16-bit fractions in a per-pixel R32Uint target allocated with the first displaced instance.
  The G-buffer resolve (a `TESSELLATION` permutation, used only while the scene has displaced instances) reads
  that point instead of searching for it (round 4 projected the depth's hit onto the base plane iteratively,
  which only converges while the displaced surface is a height field over that plane), takes the world position
  from the merged depth, shades the base triangle's attributes there — the material is evaluated at the
  *undisplaced* point, as the displacement is, so a pattern that drives both stays attached — and tilts the
  interpolated normal by the surface gradient of the displacement (height differences a pixel apart along both
  edges), so shading is smooth across micro-triangles; motion vectors carry the hit's own position (otherwise the tiny
  offset between the solved point and the micro-triangle read as motion, and TSR plus motion blur flickered on a
  still surface: 0.45 % → 0.003 % of pixels changing by > 24/255 per frame). **Shadows**: `vsm_tess.wgsl` runs the
  same emit / split / dice on the (cluster, page) software items of displaced clusters, with factors in the
  page's texels (clipmap level or local light face mip at the point), patches culled against the page and
  micro-triangles rasterised like `vsm_raster.wgsl` (24.8 fixed point, page clip, slope bias) into the atlas'
  mirror, scratch tiles included: the static / dynamic page cache treats displaced instances like any other —
  static ones are cached, moving ones (dynamic world) and ones whose displacement is animated (below) re-render
  over their static layer; heights are the main view's, so shadows fall from the surface the camera sees. **Interplay**: skinned instances and voxel
  clusters are not tessellated; alpha-masked or translucent displaced materials render undisplaced; Lumen sees
  the displaced surface through baked variants of the meshes' distance fields and cards (round 6, see *Lumen*).
  Debug view **25** colours displaced surfaces by tessellation depth (splits, then dice level: blue → green →
  red) and shades every micro-triangle; `PageDown` toggles tessellation. The valley gained a cobblestone path
  and displaced boulders behind the initial view (one turning on its plinth: a movable displaced instance).
  Measured (RTX 3090 shared with up to six other agents: interleaved runs, minima of 3; valley high, tessellation
  on vs `--no-tessellation`): path view (`--eye -26,1.8,39 --at -33,0.2,53`) at 1600×900: 1.82 M micro-triangles
  (69 k patches; 0.05 M in refreshed shadow pages), GPU frame 4.17 → 5.87 ms — software raster + merge 0.08 →
  1.62 ms (emit, split, dice and both raster passes; late pass 0.04 → 0.12), material resolve 0.46 → 0.53 ms,
  shadow page tessellation 0.14 ms; at 2560×1440: 4.19 M micro-triangles, 7.85 → 11.87 ms (raster 0.11 → 3.44 ms,
  resolve 1.06 → 1.24 ms). Grazing close-up (`--eye -27.5,0.35,41.5 --at -31,0.12,49`, the path filling the
  screen): 3.42 M micro-triangles, 3.92 → 7.13 ms (raster 2.86 ms) at 1600×900; 7.76 M, 6.93 → 13.63 ms (raster
  5.89 ms) at 2560×1440 — about 0.85 ms per million micro-triangles for both raster passes, and grazing views
  dice finer than the screen needs (the factor measures edges as if they faced the camera, as in UE). Centring
  segment lengths on the dice rate (the 1/√2 in the factor) halved the micro-triangles (6.5 → 3.4 M in the
  grazing view) at no visible loss. Scenes without displacement are unchanged: forest (medium) and cathedral
  (high) every pass within ±0.03 ms of the base build (frame 8.80 → 8.98 / 9.11 → 9.14 ms min, noise), images
  differ from the base build no more than two runs of the base build differ from each other (forest: 0.25 % vs
  0.27 % of pixels over 24/255, from its wind; cathedral and valley 0.000 %). The valley's default view (the displaced content behind the camera, none on screen): 4.89 → 5.15 ms —
  the empty tessellation dispatches (≈ 0.08 ms over both culling passes and the shadow pages) and the resolve's
  `TESSELLATION` permutation (0.36 → 0.45 ms: its registers cost every pixel once a scene has displaced
  instances). Temporal stability (sunlit view, 16 frames, pixels changing by > 24/255 per frame): static camera
  0.003 % (tessellation off 0.000 %). Under a slowly changing factor (dice rate ramped 1 % per frame, TAA off)
  frames change at 0.11 % per frame with the morph and 0.09 % with snapping instead, vs 0.03 % for a fixed rate:
  at a 2 px dice rate the per-frame change is dominated by the height filter following the segment size, and
  a pop moves a surface by less than a pixel; the morph is what keeps the edges watertight and continuous (the
  CPU mirror's continuity test fails without it), not a visible difference in this view. A slow dolly (0.3 m/s)
  changes 2.8 % of the path's pixels per frame (0.6 % without displacement): the parallax of real micro-geometry,
  which TSR resolves like any other.
  **Material graphs drive the displacement** (`tessellation/graphs.rs`, `displacement_graph.wgsl`,
  `displacement_graph_resolve.wgsl`; UE: the material's Displacement output). When a material's graph links its
  Output node's Displacement pin, the graph's height (0..1, clamped, so `Displacement::bound` still bounds it;
  magnitude and centre stay the material's) replaces the texture channel everywhere a height is evaluated —
  dicing and geomorphing, shadow pages, the resolve's hit search and normal tilt. Tessellation dices every
  material in one dispatch, so the scene's *set* of displacing graphs makes one permutation of the dicing
  pipelines: a generated module with every such graph's `graph_displacement` (named by graph id) and a `switch` on
  the material's graph id (`TessMaterial::source.zw`: graph id, parameter offset), compiled at runtime — natively
  and in the browser — and cached by the hash of the set (emit and split keep their pipelines; scenes without
  displacing graphs keep the base ones; the texture path pays one uniform branch). The G-buffer resolve runs the
  graph's own function in the graph's extended pipeline. `ExtendedMaterials` patches the graph id and parameter
  offset only for materials whose graph has a resolve pipeline, so the tessellator and the resolve always agree
  (a material that loses its pipeline stays flat rather than displacing differently in the two). Graph
  parameters feeding the displacement live in the material instances' parameter block (bound to the dicing
  passes): editing one recompiles nothing; a changed displacement — table, displacing code or parameter values —
  invalidates the cached shadow pages (editing a texture-driven magnitude used to leave them stale). Crack-free:
  the compiler only lets the height depend on the undisplaced point (object / world position, normal, UV set 0)
  and per-frame constants, and the CPU mirror's shared-edge test also runs a brick-pattern height (floors,
  fractions, a steep bevel). The gallery's brick terrace (preset *Brick pavement*: running bond, chipped bevelled
  edges, per-brick heights and tints, mortar joints 2.8 cm deep) is displaced by its graph alone. Measured
  (1600×900, RTX 3090 shared with other agents, minima of 3 interleaved runs; same micro-triangles, the terrace's
  height from its graph vs a texture channel of the same magnitude in a temporary build): close-up
  (`--scene gallery --eye 0.9,0.45,10.9 --at 0,0,9.2`, 2.91 M micro-triangles) software raster + merge 1.43 ms
  (texture) → 1.89 ms (graph: three noise layers, five octaves per height), material resolve 0.51 → 0.95 ms, GPU
  frame 5.70 → 6.62 ms (3.88 ms with tessellation off); grazing (`--eye 3.6,0.12,10.0 --at -1,0.02,8.6`, 2.75 M)
  raster 1.29 → 1.73 ms, resolve 0.40 → 0.71 ms. Displacing each grid vertex once (above) took the texture-driven
  valley path from 1.62 to 1.04 ms of raster (grazing 2.83 → 1.83 ms) and the graph-driven terrace from 4.05 to
  1.89 ms, with images within run-to-run noise. Static camera: 0.0011 % of pixels change by > 24/255 per frame
  (tessellation off 0.0012 %). Scenes without displacement and the texture path render as before (forest,
  cathedral, valley: image differences and pass times within run-to-run noise); the gallery's initial view, which
  now shows the terrace's near edge, went from 3.76 to 4.58 ms.
  **Vector and animated displacement** (round 5; `docs/round5/displacement.md` has the design). A graph's
  **VectorDisplacement** pin moves the surface by a vector instead of a height along the normal: world space (the
  space of `VertexNormal` and `WorldPosition`, so `VertexNormal × height` is the height pin's displacement and a
  tangent frame can be built in the graph; a per-triangle UV tangent frame would differ across an edge — a crack),
  in units of the material's magnitude times the instance scale, zero in place, its length clamped to 1 so the
  magnitude bounds it: culling pads by `magnitude · scale` (a height's bound is `magnitude · max(centre, 1 −
  centre)`). The tessellators turn it into an object-space offset with the instance's normal matrix as a row
  transform (the inverse linear part, no inverse in the shader). The dicing permutation's switch wraps each
  graph's function by what it returns, so heights and vectors dice in one dispatch; pipelines without a vector
  graph compile the vector path out. Surfaces can now **fold over themselves** — the reason the resolve reads
  stored barycentrics (above): under an overhang several base points project to the same place and a search
  lands on the wrong one. A vector's shading normal is the cross product of the displaced surface's tangents (the
  edge steps in the plane of the interpolated normal plus the offset's finite differences a pixel apart), right
  on undersides too; displaced pixels are no longer flipped as back-facing by their *base* triangle. Shared edges
  stay bit-identical (the CPU mirror runs a folding mushroom field), and the mirror checks that the stored
  barycentrics name every grid vertex's own point. **Animated**: a displacement that reads the time flags its
  instances `INSTANCE_ANIMATED`: they are dynamic shadow casters like movable or wind-blown ones (kept out of the
  pages' static layers), and the invalidation's collect pass appends their footprint every frame — bounds padded
  by the displacement's reach, change 2 · reach — so the pages under them re-render only their dynamic casters
  over the cached static layer, and only levels whose texels the displacement can change by two (coarser levels
  never; finer ones every frame: an arbitrary graph has no speed bound). The resolve of an animated graph
  evaluates the pixel's base point once more at the previous frame's time for its motion vector (TSR and motion
  blur follow the surface); static graphs and texture displacement compile both out and keep their fully cached
  shadows. Lumen represents an animated displacement by its mean surface (see *Lumen*: displaced surfaces);
  screen traces see the moving one. Presets *Mushrooms (vector displacement)* — a disc of ground around each cell centre wrapped over a
  cap, its underside and a stem — and *Rippling cloth (animated displacement)* — travelling sine waves and a
  scrolling noise, pinned at the edges —, both as 3 m beds in the gallery.
  Measured (1600×900, RTX 3090 shared with another agent, minima of 3 interleaved runs against the previous
  build): the stored barycentrics cost the payload pass about 0.1 ms per 2.9 M micro-triangles and save the
  resolve three displacement evaluations per pixel — brick close-up raster 1.90 → 2.00 ms, resolve 0.91 → 0.69 ms,
  frame 6.56 → 6.45 ms; valley path (texture height) raster 1.04 → 1.06 ms, resolve 0.49 → 0.48 ms, frame 5.01 →
  5.04 ms; valley default view, forest and cathedral unchanged in every pass (±0.03 ms) and image (within
  run-to-run noise). Mushrooms, low view (`--eye -5.9,0.3,10.4 --at -6.5,0.08,9.2`): 2.11 M micro-triangles, GPU
  frame 3.77 ms (tessellation off) → 5.38 ms — raster 1.30 ms, resolve 0.41 → 0.48 ms, no shadow page rendered
  (cached); 5.47 M and 10.58 ms at 2560×1440. Cloth (`--eye 4.4,0.6,10.9 --at 6.3,0.15,9.0`): 1.37 M
  micro-triangles in the view and 3.99 M in the 90 shadow pages that re-render their dynamic casters every frame,
  3.60 → 5.75 ms — raster 0.83 ms, shadow tessellation 0.88 ms, resolve 0.36 → 0.51 ms; the gallery's initial view
  (all three beds at a distance) 3.77 → 4.40 ms with 9 pages a frame. Static camera, pixels changing by > 24/255
  per frame: mushrooms 0.0002 %, bricks 0.0011 % (as before), around the cloth 0.0014 %. One trap was found by
  those dumps: the payload pass matched the depth pass's bits only while both compiled the same arithmetic — the
  barycentrics are computed apart from it and the match has one step of slack now.
  **Speed bound and vector spaces** (round 6; `docs/round6/displacement-motion.md`). The material compiler bounds
  how fast an animated displacement moves (`CompiledGraph::displacement_speed`, `helios_material::motion`): an
  abstract interpretation of the displacement's nodes giving every value an interval and a bound of its time
  derivative — products by the factors' largest magnitudes, `sin`/`cos` keep their argument's rate, a smoothstep's
  slope is at most 1.5 / (max − min), a Panner adds |speed|, noise is Lipschitz (slope 4.7 for any gradients),
  rotations keep a vector's speed; discontinuities in the time (`Frac`, `Floor`, `Step`, `If`), scrolled
  textures and the like are unbounded. It is evaluated per material instance with its parameter values, carried in
  the tessellation uniform (`TessMaterial::motion`, patched as parameters change) and turned per instance into the
  footprint's per-frame motion (speed · Δt) by the invalidation's collect pass, so the cached pages under an
  animated displacement refresh — as for the wind — once it may have moved two of their texels: the cloth's 1.64 ×
  magnitude per second (26 cm/s) still refreshes the finest levels every frame but coarser ones every few frames.
  Unbounded graphs keep the every-frame refresh; `--no-displacement-speed-bound` restores it as a reference. A
  **TransformVector** node (UE's Transform: World, Object = Local, Tangent; two drop-downs in the graph editor)
  re-expresses directions in either function: a vector built in object or tangent space and turned into world
  space follows its instance. Object space is the instance's linear part over its largest scale, so the
  tessellator's conversion back makes the offset exactly `vector · magnitude` in object units, whatever the
  rotation — the tessellators, the shadow pages, the resolve at its stored barycentrics and the previous frame's
  pose (motion vectors) agree. Meshes store no tangents and a per-triangle UV frame would crack, so the
  displacement's tangent frame is the interpolated object normal with the object X axis projected into its plane
  (the mesh's U direction on planes and terrain); the surface function's is the shading frame. The gallery's
  *combed tufts* (`presets::combed_tufts`, a tangent-space vector) grow on a platform turning behind the default
  view (`--eye 1.9,1.1,15.6 --at 0,0.1,13.4`); they made the gallery's ninth extended pipeline, so classification
  now has 16 extended slots (8–23). Measured (2560×1440, minima of 3 interleaved runs against main;
  pages and shadow micro-triangles averaged over the last 60 frames): cloth close-up 172 → 140 pages a frame, shadow
  micro-triangles 9.95 → 9.55 M, shadow tessellation 2.07 → 2.04 ms (its finest levels' 2–4 mm thresholds are below
  the cloth's 4.4 mm a frame, so they still refresh every frame); the cloth from 7 m 83 → 45 pages, 0.34 → 0.13 M,
  0.17 → 0.12 ms; the gallery's default view 9 → 2 pages a frame. Against the every-frame reference, a static
  camera's frames match at 64.5 dB (no pixel off by more than 24/255: no stale shadow); scenes without animated
  displacement are unchanged (valley, cathedral, forest, gallery at 100 % and 67 %: every pass within noise). The
  turning platform's own view: 1.79 M micro-triangles, 9.78 ms.
* **Lumen-style GI** (`lumen.rs`). A 4-level global distance field (0.25–2 m voxels) is composited on the GPU from
  per-mesh SDFs; a voxel **surface cache** stores the radiance leaving nearby surfaces (sun with SDF soft shadows,
  sky with SDF occlusion, local lights, emission, one bounce from the **radiance cache** of world probes), so bounces
  accumulate. The final gather places a **screen probe** per 16×16 tile, traces 64 rays each (screen → distance
  field → surface cache → sky), filters them across neighbours and projects to SH for per-pixel integration.
  **Adaptive probes** (UE5's adaptive placement): per 8×8 cell, the pixels the integration shades are tested against
  the four uniform probes around them, and where none lies on a pixel's surface (window mullions, gaps between pews,
  chandelier rings, distant detail) the cell requests up to two extra probes on the missed surfaces. Requests are
  granted up to half the uniform probe count — evenly thinned at random when there are more, so no region is starved
  every frame — appended behind the uniform probes in the probe buffer and atlas with a per-cell indirection, traced
  and filtered like the others (indirect dispatch) and interpolated on their finer grid. Without them such pixels
  flipped between the radiance-cache fallback and whichever uniform probe the per-frame jitter put on their surface;
  **reflections** trace GGX rays the same way. Stained glass feeds coloured light into the cathedral.
  **Near field** (`near_field.rs`, `lumen_mesh_sdf.wgsl`): as in UE5, the first 2 m of every gather, reflection
  and local-light shadow ray sphere-trace the per-mesh SDFs (packed into one 3D atlas, heightfields for terrain)
  of the instances listed in a camera-centred 2 m culling grid; the global field takes over beyond. Small
  objects (candles, pew legs) occlude and cast contact shadows, and a light's own fixture no longer blurs into
  a voxel blob around it.
  **Mesh cards** (`cards.rs`, `lumen_card_light.wgsl`, `lumen_cards.wgsl`; UE5's card-based surface cache): every mesh
  with a volume distance field is captured on the CPU by up to 6 axis directions × 3 depth layers of orthographic cards
  (depth + normal per ~10 cm texel, rasterised from a LOD cut; depth peeling captures e.g. a courtyard's inner walls
  behind its outer ones). Each instance's cards are lit at a mip chosen from its distance (texels ≈ 1 % of it) in 8×8
  pages of a 2048² atlas behind per-card page tables, a global LOD bias keeping the atlas within budget; 1024 pages a
  frame are lit — new allocations first, then a round-robin refresh — with the sun through the virtual shadow map where
  the texel's page is resident (distance-field shadows otherwise), local lights, sky occlusion, one bounce from the
  radiance cache and emission. Gather, reflection and radiance-cache probe hits project into the hit instance's cards and keep the texel
  whose captured depth and normal match the hit (the near field reports the instance it hit; the global field stores
  the closest card instance per voxel); the voxel lighting remains where no card matches. Lighting stays on the
  surface it belongs to: no sunlight through thin walls, no terrain colour bleeding up through a floor that covers it,
  sun patches through windows bounce. Debug view 14 shows the card lighting each pixel would read (purple: voxels).
  GPU memory: 32 MiB lighting atlas, 16 MiB instance ids (16 bits per global-SDF voxel), a capture atlas within 16 MiB
  (cathedral 7.7, Sponza 9.4 MiB).
  **Radiance cache refresh** (`update_probes` in `lumen_light.wgsl`): it ran one thread per probe tracing its 32 rays
  in turn — ~3500 long serial threads, most of the GPU idle; it now runs a workgroup per probe, a thread per ray,
  with a shared-memory reduction of the SH: the "lumen surface + radiance cache" pass went from 1.42–1.49 to
  0.22–0.35 ms (cathedral 1.48 → 0.35, forest 1.42 → 0.24, Sponza 1.49 → 0.22, valley 1.43 → 0.22; shared GPU,
  minimums of interleaved runs), images unchanged within noise. Both final gathers profit. Moved instances update
  the global field, cards and near field in place (see *Dynamic world*).
  **Displaced surfaces** (`lumen_displaced.rs`, `lumen_displace.wgsl`; round 6, `docs/round6/lumen-displacement.md`):
  Lumen follows Nanite tessellation. Every instance whose material displaces its mesh is represented by a
  *displaced variant* of the mesh — a distance field (the mesh's own kind and voxel size) and mesh cards baked from
  the displaced surface — so the global field, the near field, both surface caches, indirect light, sky occlusion
  and reflections see the relief: texture heights, graph heights and vectors (overhangs) alike. Variants are per
  (mesh, material), and per instance where the displacement depends on its placement (world-space graphs and
  vectors, world triplanar heights). The mesh is refined into points at most ¾ of a voxel apart (welded, on a
  worker thread), the GPU displaces them with the tessellator's own `displacement_position` — for graphs through
  the dicing passes' generated module — filtered for the point spacing, and the field and cards are baked from the
  read-back positions on worker threads with the same code as every mesh's. In the browser, which has no threads
  without cross-origin isolation, the refinement and the bake run on **Web Workers** that each run a second instance
  of the wasm module (`lumen_offload.rs`, `web_bake.rs`, `bake_worker.js`; `docs/round6/web-bake-worker.md`): the
  render thread only hands over the inputs and takes in the results, one per frame, so a scene's variants no longer
  stall frames for 30–920 ms each (worst frame while the gallery's four variants bake 718–920 → 6–7 ms, against
  2–4 ms without baking, and all of them in after 1.5–1.6 instead of 2.3–3.3 s; `?bake-workers=0` runs them on the
  render thread as before, as does a browser where a worker cannot start). An
  animated displacement is represented by its mean surface (32 times in 8 s): the gallery's cloth moves at most
  3.6 cm RMS from it at 3.3 cm voxels, where a snapshot would be √2 times as far off and a re-bake could never keep
  up with the waves. Until a variant is ready (0.5–1.3 s of CPU per variant) instances keep their previous one;
  the variants of a scene arrive in one Lumen re-upload. Debug view 14 shows the gallery's mushroom bed and cloth
  in card lighting where they were voxel fallback (purple) before, and fewer unmatched card texels on the
  valley's displaced boulders (19 → 14 % of their pixels). Scenes without displacement upload and render as before.
* **Lumen Lite** (`lumen_lite.rs`, `lumen_lite.wgsl`, `lumen_trace.wgsl`; UE 5.8's `r.Lumen.FinalGatherMethod 0`).
  An alternative final gather for weak GPUs (`RenderSettings::gi_method = GiMethod::LumenLite`): instead of tracing
  64 rays from every 16×16 screen tile (plus adaptive probes) every frame, diffuse GI is gathered from a world-space
  **irradiance field** (DDGI-like) that changes slowly and is shared by every pixel near it. 4 camera-centred clipmap
  levels of 16³ probes (1, 2, 4, 8 m spacing: ±8 … ±64 m) scroll by whole spacings with a little hysteresis;
  probe slots are addressed toroidally (world coordinate mod 16), so a move keeps every probe still inside a window
  and only the slab that entered is new. Each probe holds an octahedral irradiance map (6×6 texels: cosine-weighted
  mean radiance + sky visibility) and an octahedral map of ray-distance moments (14×14: mean and mean square in
  probe spacings) for Chebyshev visibility, each with a one-texel octahedral border in its atlas so shading samples
  them bilinearly (rgba16f atlases of 40 MiB, plus 3 MiB of buffers, allocated on first use). **Update budget**: 512
  probes a frame are picked on the GPU without readback — a pass scores every probe (new probes first, finest levels
  first; then age × 4 on screen × ⅛ for *sleeping* probes × 1/64 for *buried* ones) into a 64-bucket histogram, one
  workgroup finds the threshold that fills the budget and a third pass appends the winners (indirect dispatches from
  there). **Classification**: before tracing, the global distance field moves a probe that sits within 0.1 spacing
  of a surface along its gradient (≤ 0.45 spacing), disables (*buries*) it if it cannot get out — inside walls, under
  terrain — and puts it to *sleep* where no surface lies within 1.8 spacings (no pixel interpolates it; it is still
  refreshed now and then, as foliage is not in the global field). **Tracing**: 128 rays per probe (spherical
  Fibonacci, randomly rotated per refresh) through Lumen's own tracing — near-field mesh SDFs, global distance field,
  mesh-card and voxel surface caches, radiance-cache far field, sky — one shared WGSL function
  (`trace_world_radiance`, which the screen probes call after their screen trace), so the irradiance field sees the
  same surface cache and the same multi-bounce light. The rays are projected into the probe's texels with hysteresis
  0.85 (halved where the irradiance changed by more than half: switched lights, a moving sun; new probes take their
  first rays as they are), then copied into the atlases with their borders. **Shading**: per half-resolution pixel,
  the 8 probes around a shading point biased 0.15 spacing along the normal and 0.3 towards the eye, weighted by
  trilinear × smooth backface × Chebyshev (cubed, crushed below 0.2, as DDGI) terms, from the finest level that holds
  the pixel, blended into the next coarser level over its window's last 2 cells and into Lumen's radiance cache beyond
  the last level. The result replaces the screen-probe gather in `gi_raw` (same format), so the GI denoiser, sky
  visibility (specular occlusion), material AO and the lighting pass are unchanged. **Reflections**: rough ones come
  from the irradiance field — the lighting pass already blends specular towards the gathered irradiance between
  roughness 0.2 and 0.5 and fades the traced reflection out between 0.35 and 0.55; glossy ones keep Lumen's traced
  reflections (one GGX ray per half-resolution pixel), since a probe's 6×6 irradiance cannot show a reflected image.
  **Selection**: `--gi lumen|lite|auto` natively, `?gi=lumen|lite` in the browser (page config `gi`), `;` toggles. By
  default (`auto`) integrated and software adapters start in Lumen Lite, and `RenderSettings::gi_lite_fallback`
  switches to it when dynamic resolution has sat at its minimum percentage for 180 frames (the screen-probe gather is
  the largest GI cost that does not shrink with the resolution) — browsers do not report the adapter type, so there
  the fallback decides. Debug view 20 shows the probes as spheres shaded with their irradiance (sleeping: blue tint,
  buried: red, not yet traced: magenta) over the irradiance each pixel gets.
  **Cost** (shared GPU, interleaved runs, sum of per-pass minimums of 3–6 runs; "GI" = mesh cards + surface /
  radiance cache + final gather + reflections + GI temporal): the final gather drops from 0.77–1.07 ms to
  0.36–0.40 ms (probe update 0.28–0.35 ms, independent of the resolution, + per-pixel gather 0.05–0.08 ms), all of GI
  from 1.20–1.71 to 0.74–1.04 ms (1.5–1.8× cheaper; 2.5–3.4× cheaper than round 3's 2.48–2.91 ms):

  | Scene | final gather: screen probes → Lumen Lite | GI round 3 | GI Lumen | GI Lumen Lite |
  |---|---|---|---|---|
  | valley | 0.77 → 0.40 (probes 0.35 + gather 0.05) | 2.61 | 1.20 | 0.81 |
  | cathedral | 0.91 → 0.36 (0.28 + 0.08) | 2.72 | 1.62 | 1.04 |
  | forest (medium) | 0.92 → 0.37 (0.30 + 0.07) | 2.48 | 1.29 | 0.74 |
  | Sponza (atrium) | 1.07 → 0.39 (0.32 + 0.07) | 2.91 | 1.71 | 0.95 |

  Images: the cathedral's lit frame differs from Lumen by 36.5 dB PSNR (valley 33.8, forest 43.4); its indirect
  light is 7 % darker on average, with more contrast where the probes see occlusion (pew undersides: 5th percentile
  0.20 vs 0.30) and less where the screen probes resolved it (pillar bases, vault ribs); Sponza's atrium is 9 % darker
  in indirect light (the far end of the gallery most; 23.6 dB); no light leaks through the 1.2 m aisle walls next to
  the sunlit exterior. Temporal stability (static camera, 24 frames, % of pixels changing by > 24/255): cathedral
  0.002 % for both, valley 0.042 % (Lumen 0.053 %), forest 0.056 % (Lumen 0.060 %); the indirect light alone changes
  by more than 4/255 in 1.05 % of the pixels (Lumen 0.30 %) — a fine uniform grain from the probe refreshes rather
  than the screen probes' blotches. Moving camera: within 2 % of Lumen's frame-to-frame change, no pops as slabs
  scroll in.
* **Local lights** (`lights.rs`). Point and spot lights are culled into 16×9×24 froxels of up to 255 lights each
  (63 before MegaLights). Lights are also injected into the surface cache, so they bounce.
* **Local light virtual shadow maps** (`vsm_local.rs`, `vsm_local_common.wgsl`, `vsm_local_sample.wgsl`). As in UE5,
  up to 32 shadow-casting point and spot lights per frame — the most influential on screen (range sphere coverage ×
  brightness, with hysteresis so slots and their pages stick), static lights first: a light that moved within the
  last 30 frames takes a slot they leave over, and keeps its map cached around an origin near it (*Cached shadow
  maps for moving lights* below; under MegaLights, which traces every light's shadow anyway, it gets none) — get
  virtual shadow maps: six cube faces for a point
  light, one perspective face for a spot light, each a 4096² virtual map with a mip chain down to one 128² page.
  Their pages are entries of the sun's page cache after its clipmap levels and live in the same physical atlas, so
  the sun and all lights share one pool, one allocation, one dirty list and the same page renderer (the (cluster,
  page) expansion and the hardware and software page rasterisers take a perspective projection per face; the hardware
  path clips at the light's near plane, software items must lie in front of it). Every lit pixel requests, for each
  light of its froxel with a shadow map, the page of the mip whose texels match its screen footprint (half a mip
  coarser by default, `local_shadow_resolution_bias`; lights with a large source radius coarser still) plus the
  coarsest page of that face; volumetric fog requests coarse pages. Pages stay cached while the map's origin and the
  geometry are static: a new origin or shape, a scene change or a slot handed to another light drops that light's
  pages. Streaming, moves and skins refresh local pages like the sun's (*Wind*, *Dynamic world*), in each light's
  terms: an instance's footprint refreshes the pages it projects onto in a light's faces and mips, once it may have
  moved 2 of their texels at its depth inside the face. The wind's refreshes are scheduled by what a page holds
  instead (round 5): an instance's bounding sphere says little about a light inside or next to it — a campfire among
  trees refreshed every page of its light every frame, at the crowns' speed — so whenever a local page is rendered,
  its expansion records how fast and how far the wind can move the clusters actually drawn into it, in the page's
  texels at each cluster's depth (two words per physical page in the info buffer), and the page is refreshed once
  they may have moved 2 texels: pages with near ferns and grass every frame or two, trunks and distant crowns every
  few frames or never; only their swaying casters are re-rendered, over the cached static layer. A page under a
  caster entering or leaving the wind, and every local page when the wind itself changes, measures again. A
  sphere reaching a cube face's near plane no longer covers the whole face (`local_sphere_rect` bounds the part in
  front of the plane: wholly to one side of the face axis, it projects outside the face or onto a band along one
  edge), which most clusters around a light did for some face: forest 198 k → 137 k page draws a frame (local
  lights' share 86 k → 20 k), GPU frame 18.4–18.5 → 16.7–17.6 ms at 2560×1440 (`vsm raster` 3.6–3.9 → 2.2 ms),
  67 %: 12.05 → 11.33 ms. One
  culling view serves all lights: it keeps clusters that overlap a world box around some light's dirty pages and
  selects LOD for the most demanding light's finest dirty mip, so a frame without new pages costs the same with 1 or
  32 lights. A pass of its own computes the shadows of each pixel's eight strongest shadow-mapped lights into a
  screen-space mask (4 bits each, dithered for TAA; as UE projects shadows into a mask), so the filter's cost and
  registers stay out of the lighting: contact-hardening soft shadows for the four strongest (PCSS: a blocker search
  over the light's source radius sets the filter radius, taps slide onto the receiver plane, and taps inside the
  centre's page reuse its lookup), one bilinear comparison for the next four and, in the lighting, for any further
  ones. Translucent surfaces filter them inline, volumetric fog takes one tap per light, so shafts fall between
  pillars and through the choir screens. Lights beyond the budget keep cone-traced distance-field soft shadows for
  the strongest few per pixel (mesh SDFs within 2 m of the receiver and of the light, the global field between).
  Debug view 15 shows the sampled mip and page grid of each pixel's strongest shadow-mapped light.
  Slot order matters with moving lights: in the candle-lit cathedral (300 lights, MegaLights) the swinging
  chandeliers' candles took slots and re-rendered ~970 pages a frame (page raster 11–15 ms); static lights first cut
  that to 713 dynamic-only refreshes (0.11 ms), but near the altar six swinging candles still took the slots left
  over (11.6 ms), hence no slots for moving lights under MegaLights (0.06 ms). In the clustered loop the default
  cathedral's eleven swinging chandelier lights take 10 spare slots; re-rendering their cubes every frame cost
  ~80 full pages and 3.1 ms of `vsm raster` at 1440p, which round 4 avoided with distance-field shadows (42 dB PSNR
  against the shadow maps: the soft bands under the pier capitals fainter).
  **Cached shadow maps for moving lights** (`vsm_anchor.rs`, round 5,
  [`local-shadow-cache.md`](docs/round5/local-shadow-cache.md)). A light is not a point: its penumbra spans the
  shadows cast from every point of its source, so a moving light's map keeps the *origin* it was rendered from while
  the light stays within a tenth of its source radius of it (at least 1 cm). The pages keep their static layer; the
  casters moving with the light (a chandelier's ring, candles and chain) are redrawn into them at their current pose
  by their move footprints, so a fixture never shadows itself with a stale pose. Receivers correct the parallax: an
  occluder at 1 / t of the receiver's distance casts its shadow (t − 1) · (light − origin) away from where the held
  map has it, and the PCSS filter, which knows t from its blocker search, moves its taps by as much (slid onto the
  receiver's plane; nothing at contact) — shadows follow the light every frame, and a re-render moves nothing
  visible. When the light leaves the tolerance its requested pages are rendered from a new origin placed 0.75
  tolerances ahead on its path (a chandelier light: every ~60 frames), the furthest out first; a light gets a map
  only if the budget can keep up with its recent top speed, and one that outruns it keeps distance-field shadows
  until it rests, so lights don't flip between the two. A moving light's map is one mip coarser (UE's
  `ResolutionLodBiasLocalMoving`). Round 6 ([`fast-light-shadows.md`](docs/round6/fast-light-shadows.md)) made the
  budget fit fast, sharp lights: **every cube face has its own origin** and is re-anchored on its own, the budget is
  **pages per frame** (`RenderSettings::moving_light_pages`, 32), a face costing the pages the view requests of it
  (counted on the GPU by `dirty_info`, read back per face), so a face nobody looks at costs its one fallback page and
  six faces due together spread over the next frames; a light that crosses its tolerance sphere every frame is
  rendered every frame **from where it is** (the lead shrinks with the speed, to none); one the budget cannot follow
  at its tolerance is held at twice or four times it rather than given up; and a moving light competes for a slot at
  half its influence instead of only for the slots static lights leave. The lantern of `--lantern` (a 2 cm flame at
  up to 1.6 m/s) is re-rendered every frame, ~17 pages at the moving mip bias (the reference at full resolution:
  33): 44.8 dB against the every-frame reference — all of it the mip bias — where its distance-field shadows were
  33.9 dB; with the chandeliers swinging too, all twelve moving lights keep their maps for 1000 frames (round 5:
  the lantern took the one render a frame and 8 chandelier lights fell back to distance fields), ~28 pages
  a frame rendered from new origins against 161 re-rendering every moving light every frame. HUD: `moving lights`
  line. Cathedral, 2560×1440 (round 5): GPU
  frame 10.27 ms (distance fields, round 4) → 10.77 ms, against 13.75 ms re-rendered every frame (lighting +0.22 ms
  for ten more mapped lights, marking +0.08, page raster +0.06, culling and expansion +0.10; ~42 dynamic-only pages
  a frame); image against the every-frame reference 48.9 dB (distance fields 42.2 dB), from under a chandelier
  42.9 dB (34.4 dB), no added flicker (0.0063 % of pixels over 24/255 per frame in all three modes).
  `RenderSettings::moving_light_shadows` selects `Cached`, `DistanceFields` or `EveryFrame`.
* **MegaLights** (`megalights.rs`, `megalights_sample.wgsl`, `megalights_shade.wgsl`, `megalights_denoise.wgsl`,
  `megalights_common.wgsl`, `megalights_surface.wgsl`; UE 5.5's MegaLights). `RenderSettings::mega_lights` (`=`)
  replaces the clustered light loop — which can shadow only the 32 shadow-mapped lights plus four distance-field
  traced ones per pixel and lets every other light leak through walls — by a fixed budget per pixel in which *every*
  local light casts soft shadows. **Sampling**: per 2×2 block, at a representative pixel that moves every frame, four
  lights are drawn from the froxel's light list in proportion to their unshadowed contribution through the shared BRDF
  (`brdf.wgsl`, the same functions the classic loop calls), by systematic resampling over at most 32 candidates;
  longer lists are thinned to a stratified subset in which every light has the same chance, its weight scaled by the
  inverse, so the estimate stays unbiased and the cost per pixel constant. A small floor keeps lights behind the
  representative's normal possible for the rest of its block. **Visibility guiding**: per 8×8 tile, bit masks (light
  index mod 256) of the lights some ray was traced to and of those some ray reached; a light whose rays were all
  blocked in last frame's (reprojected) tile keeps a tenth of its weight. **Shadow rays**: one per sample, to a random
  point of the light's disk (the source radius makes the penumbra): a 1 m screen trace against the depth buffer
  (contact shadows), then over the whole ray the light's virtual shadow map where it has one — shadow-map ray marching
  (UE's SMRT, 8 steps; the first is the receiver's classic test, the later ones see occluders off to the side as the
  ray converges on its point of the light, within an assumed thickness) — otherwise Lumen's distance fields (mesh SDFs
  within 2 m of either end, the global field between, narrow cones). The screen trace treats the depth buffer as 3 cm
  shells (a thicker assumption shadowed walls behind any object in front of them) and ignores emissive pixels (candle
  flames are in the depth buffer but cast no shadow). **Shading**: every pixel evaluates the BRDF of its block's four
  samples (Substrate slabs through a variant) with their unbiased contribution weights and visibilities and writes
  its direct light *demodulated* (diffuse without the diffuse colour, specular divided by the environment BRDF), so the
  denoiser never blurs texture detail. **Denoising** (SVGF): temporal accumulation with reprojection, depth-checked
  disocclusion, a 2.5σ variance clip and up to 24 (diffuse) / 12 (specular) frames, with luminance moments; then four
  à-trous iterations (3×3 binomial, strides 1–8) with depth-plane, normal and luminance edge-stopping scaled by the
  variance of the accumulated mean (so converged pixels keep normal-mapped lighting detail); the last one modulates
  the result back into one texture, which the lighting pass adds instead of running its loop. Debug views **18**
  (each sample's light, one hue per light, dark where blocked) and **19** (this frame's samples | denoised).
  Cost (cathedral medium at night, shared GPU, minimum of 4 interleaved runs; "direct" = light culling + local light
  shadow mask + lighting + MegaLights):

  | Lights | Classic direct | MegaLights direct | MegaLights passes | Classic lighting pass | Frame (classic → ML) |
  |---|---|---|---|---|---|
  | 33 (default scene) | 1.03 | 1.84 | 1.37 | 0.88 | 6.38 → 7.15 |
  | 100 (`--lights 100`) | 2.45 | 2.52 | 1.95 | 1.95 | 8.49 → 8.60 |
  | 300 | 2.78 | 2.54 | 1.99 | 2.15 | 9.03 → 8.71 |
  | 1000 | 3.53 | 2.74 | 1.95 | 2.50 | 10.55 → 9.75 |
  | 1000, aisle view | 3.63 | 2.61 | 1.86 | 2.52 | 10.11 → 9.08 |

  At 33 lights the MegaLights passes split into sample 0.16, trace 0.40, shade 0.08, temporal 0.14, spatial 0.59 ms;
  from 100 to 1000 lights only the share of distance-field rays grows, and the lighting pass itself drops to 0.46 ms
  — while the classic path shadows only 36 of those lights per pixel. Image vs the classic path at 33 lights: 39.7 dB
  PSNR (nave), 39.8 dB (candelabrum close-up; two classic runs differ by 66.7 dB); the differences are shadows the
  classic PCSS blurs away. Static camera, pixels changing by more than 24/255 frame to frame: nave 0.0093 → 0.0091 %,
  close-up 0.0017 → 0.0009 %, 1000-light aisle 0.0005 → 0.0007 % (classic → MegaLights). Browser (headless Chromium,
  1280×720, 300 lights): 1.2 ms. Memory (only while on): 56 bytes per internal pixel (two rgba16f raw, two sets of
  rgba16f diffuse + specular + r32uint history) plus 36 bytes per 2×2 block of samples.
* **Volumetric clouds** (`clouds.rs`). A cumulus layer around the planet, ray-marched where the depth buffer shows
  sky: tileable Perlin–Worley shape and Worley detail volumes generated once on the GPU, a weather field for coverage,
  adaptive stepping (long steps through clear air, short ones inside clouds), sun light marched through the cloud with
  multiple-scattering octaves and a dual-lobe phase, ambient from the sky's SH. Half resolution, checkerboarded, with
  temporal reprojection by cloud distance; a 16 km cloud shadow map darkens the ground under moving clouds.
* **Volumetric fog** (`fog.rs`). Height fog with drifting noise in a 16 px × 64-slice froxel volume: each froxel
  scatters the sun (virtual shadow map visibility, cloud shadows), the clustered local lights and Lumen's radiance
  cache (so interiors don't glow with sky light); jittered, temporally reprojected and integrated front to back.
* **Translucency** (`translucency.rs`, `translucency.wgsl`, `water.wgsl`). As in UE5, a separate forward pass after
  the deferred lighting: `Glass`, `Water` and `Translucent` (glTF `BLEND`) instances are skipped by the main view's
  culling (they never occlude through the visibility buffer / HZB) and drawn as plain indexed meshes — the finest
  in-memory level of their virtual mesh, decoded once — one draw per instance, frustum culled and sorted back to
  front on the CPU (GPU cluster lists are unordered and drawn by one indirect call). The lit HDR image is copied with
  a 5-level blurred mip chain; refracting surfaces read it at a displaced position and replace the pixel, blended
  ones are premultiplied "over". **Layers** are composited in per-pixel depth order wherever translucent surfaces may
  overlap (depth peeling, so every surface keeps its own shading model, refraction included): the CPU bounds the
  layers any pixel can see — the most projected bounding boxes of visible instances stacked at one point, a
  non-convex mesh counting as all (convexity / flatness is tested once per decoded mesh) — and for *n* > 1 layers,
  depth-only draws peel the nearest *n* − 1 into a depth texture array (each keeps the nearest fragment behind the
  previous layer). Then everything behind them is drawn in instance order as before, and each peeled layer over
  it, back to front, after a fresh copy of the scene colour — so water refracts the glass and panels standing in it,
  glass shows the glass behind it, intersecting surfaces cross correctly, and refraction never fetches a layer that
  lies in front. Up to 4 layers are exact (deeper ones fall back to instance order); views without overlap (the
  scenes' initial ones) peel nothing and cost what they did. `/` (`RenderSettings::translucency_layers`) toggles it.
  Cost (1600×900, RTX 3090, medians of 4, all translucency passes): the initial views unchanged (valley 0.43 → 0.42 ms,
  cathedral 0.40 → 0.39 ms), one peeled layer (cathedral apse) 0.42 → 0.49 ms, three (lake test corner) 1.86 → 2.02 ms
  and (aisle glass screens) 0.45 → 0.89 ms; the layer depths take 4 bytes per pixel and layer at the internal
  resolution's capacity (17 MB for three at 1600×900), allocated once a view needs them.
  Shading: GGX sun with VSM + cloud shadows, clustered local lights, Lumen radiance
  cache for ambient light, reflections from a screen-space trace against the opaque depth and colour, then Lumen's
  global distance field + surface cache, then sky with the on-screen clouds; aerial perspective and volumetric fog are
  applied at the surface depth exactly (`A·surface + B·(1 − T) + T·background`, the background already carries the
  fog behind). **Water**: 16 directional waves (3 m down to 5 cm, directions fanning out wider for shorter ones,
  each crest bent sideways by its own slow sinusoids, evaluated on a gently warped domain) with sharpened crests and
  deep-water dispersion as a per-pixel normal
  (sub-pixel waves fade into roughness), Beer–Lambert absorption + single scattering over the refracted path to the
  bottom, shore foam, sun glints. **Stained glass**: lead came opaque; glass tints the blurred scene behind and
  diffuses sky (scaled by how open the far side is, from the radiance cache) and sun — matching what Lumen's surface
  cache emits into the interior, which still gets the panes' coloured light through their distance fields. Untextured
  `Glass` is a clear pane, tinted by its base colour, its view through blurred by its roughness. Glass and blended
  surfaces under water reflect the water around them rather than the sky, and seen from under water they are lit
  and fogged through the water like the opaque scene. The pass writes the surfaces' own motion vectors for TSR.
  `RenderSettings::translucency` / `U` toggles it.
* **Water bodies** (`water.rs`, `water.wgsl`, `shore.wgsl`; UE's Water system for lakes). Every `Water` instance is a
  flat *water body* (its world rectangle and level, from its bounds); one uniform carries them with the medium.
  **Lake bed and shore**: the G-buffer resolve turns the landscape into sediment where a body covers it — sand and
  gravel near the water line, dark silt with scattered, silt-coated stones and algae patches deeper, ripples in the
  shallows (procedural, footprint-filtered: lumpy pebble domes on jittered grids, fbm) — and soaks a band just above
  the water (darker, glossy). It runs per pixel after the runtime virtual texture lookup (or the direct layering), at
  the mesh's exact height, so the bed meets the water line exactly whichever path shaded the landscape (the RVT's
  heightfield is 1.8 m coarse). **Caustics**: the sun's light on the bed is `1 / det(I + d·(1 − 1/n)·H)`, H the
  Hessian of the same 16 waves at the point the refracted sun ray entered (one fixed-point step for the waves' own
  displacement), each wave fading where its pattern is finer than the bed's pixel or the sun disk's blur over the
  depth. The bent crests make the few dominant short waves cross at angles that change from place to place, so the
  caustics form irregular cells rather than a lattice of parallelograms (straight crests did); the bends only vary
  across each wave's travel direction, so they travel with it and nothing shimmers. Caustics they scale only the bed's direct-sun share `(E_sun·c + E_amb)/(E_sun + E_amb)` (no albedo needed), with E_sun
  through the bed's virtual shadow map and cloud shadows — so no caustics in shadow. **Underwater**: with the camera
  in a body, a full-screen draw fogs the opaque scene by absorption and single scattering along the view (to the
  surface where the ray leaves the water) and adds caustics on the bed. The in-scattered light is the sun (through the
  surface's Fresnel transmission and its shadows) and the sky / GI irradiance (Lumen's radiance cache) entering the
  surface above the middle of the ray, fading with depth: since the depth changes linearly along the ray, the single
  scattering integral has a closed form, so looking up is brighter than looking down and deep water darkens. Lake
  water scatters mostly forwards (a quarter isotropic, three quarters Henyey–Greenstein: g 0.8 for the sun beam, a
  broad 0.5 lobe for the diffuse sky light from above), so water seen sideways stays clear and the refracted sun
  glows. The default medium is a clear mountain lake: pure water's absorption with a little dissolved organic matter
  (0.40, 0.06, 0.09 /m) and light particle scattering (0.03–0.038 /m), about 10 m of visibility; water surfaces are drawn two-sided and from
  below show Snell's window (the world above through refraction, screen trace → distance field → sky, with exact
  dielectric Fresnel) and total internal reflection of the deep water around it, no sky leaking; air fog is skipped.
  Reflection rays leaving a water surface ignore screen-space hits below it (the submerged bed is in the depth buffer).
  Exposure under water keeps Snell's window in its metering: the histogram otherwise drops the brightest 10 % of the
  view (for sun disks and flames), which under water is the window, the scene's light source — the camera then
  exposed for the dim water alone and burnt the window out to white.
  Cost (valley high, 1600×900, RTX 3090, min of 5): looking down at the lake filling the screen, translucency 2.00 →
  2.40 ms and the material pass 0.26 → 0.48 ms (GPU 5.93 → 6.64 ms); the default view (lake ~10 % of the screen)
  4.54 → 4.59 ms; scenes without water are unchanged. Under water (full-screen pass + surface from below): 1.97 ms.
* **Heterogeneous volumes** (`volumes.rs`, `volume_sim.wgsl`, `volume_render.wgsl`, `media.rs`, `media_common.wgsl`;
  UE's Heterogeneous Volumes). `Scene::volumes` holds `VolumeEmitter`s: boxes of soot density, temperature and fuel on
  a voxel grid (the campfire 40×80×40 over 1.6×3.2×1.6 m, the incense 32×112×32). A `FluidSimulation` runs on the GPU
  every frame in 3D textures: vorticity, forces (Boussinesq buoyancy — heat lifts, soot weighs down —, vorticity
  confinement that restores the eddies numerical diffusion removes, a source sphere whose injection flickers with
  animated noise, ambient wind, damping), divergence, 12–16 red-black SOR sweeps of the pressure Poisson equation
  (warm-started from the last frame; solid floor, open sides and top), projection, semi-Lagrangian self-advection of
  the velocity and **MacCormack** advection of the state (back-advected, corrected by half the round-trip error and
  clamped to the departure point's 8 texels, so smoke keeps detail without overshoot; half of the plain
  semi-Lagrangian result is mixed back in — pure MacCormack left voxel-sharp fronts that read as blocks wherever the
  smoke turns opaque), then combustion (fuel burns into heat and soot), cooling, dissipation and a fade at the open
  boundaries. A `ProceduralVolume` (the valley's lake mist) fills the grid from animated noise instead — no
  simulation for distant or static media. Rendering: a **lighting cache** at half the grid resolution stores per
  voxel the sun's transmittance (16 steps through the volume towards the sun × the virtual shadow map at the fog's
  page level × cloud shadows) and the isotropically scattered light: ambient from Lumen's radiance cache (the sky
  above occluded by the volume itself, an 8-step march up) and up to 8 local lights with their shadow maps. The
  **march** runs once per pixel of the visible volumes' screen rectangle (up to 4 boxes, ordered per pixel) at the
  internal resolution: trilinear density and temperature (each sample offset randomly by up to half a voxel —
  stochastic filtering that turns the trilinear filter's voxel staircases into a smooth falloff once the temporal
  resolve averages it), the cache with a Henyey–Greenstein phase for the sun plus a second, softer and less attenuated
  octave standing in for multiple scattering (Wrenninge's approximation; without it thick smoke turned black away
  from the sun), **blackbody emission** from temperature (Planck's law at three wavelengths: dull red edges, a
  yellow-white core ~40× brighter), energy-conserving integration, strides that grow 4× through clear air and step
  back at the medium's edge, early exit at 0.3 % transmittance; each part is fogged at its opacity-weighted depth
  (froxel fog + aerial perspective, as translucency). Step offsets are jittered per pixel and frame; a **temporal
  resolve** reprojects the last result with the depth of the volume each pixel sees, clamps it to the current 3×3
  neighbourhood and blends 25 % — the jitter noise averages out, the flames don't ghost. The ray is **split at the
  nearest translucent surface** (see *Order-independent transparency*), so glass and water show the smoke behind
  them. A fire **drives a scene light** (`VolumeEmitter::light`): each frame, before the light culling, a
  one-thread-per-light pass sets its colour to the scene's colour × the simulation's summed blackbody emission
  (workgroup sums into fixed-point atomics) relative to its running average — the flicker the viewer sees, on the
  clearing, the logs' shadows and Lumen's surface cache, a frame late and without a read-back.
* **GPU particles** (`particles.rs`, `particles.wgsl`; a Niagara-lite). `Scene::particles` holds `ParticleEmitter`s
  (spawn rate and bursts, sphere / box / disc shapes, lifetime range, a velocity cone, gravity, drag, curl-noise
  turbulence, collision, size and a 4-key colour/opacity gradient over life, lit or emissive — colour or blackbody
  cooling over life —, billboard or velocity-stretched quads, soft distance, phase anisotropy). Every emitter owns a
  fixed range of slots in one buffer; the CPU only turns rates and bursts into a spawn count per frame (a carried
  fraction), dead slots claim spawns with one atomic counter per emitter — no free lists, compaction or read-back;
  an emitter's first frame pre-warms its sustained population at random ages. **Simulation** (one dispatch for all
  emitters): gravity, drag, curl noise (the curl of a 3-component gradient-noise potential: divergence-free swirls),
  and **collision with the depth buffer**: a particle that went behind the visible surface (within a thickness that
  grows with its speed) is put back onto it and bounces off the G-buffer normal with restitution and friction, or
  sticks (embers settle on the ground and keep glowing). **Lighting** once per particle per frame (UE's per-particle
  translucent lighting): sun through the virtual shadow map and cloud shadows with a Henyey–Greenstein phase (dust
  glints in the clerestory's shafts when looking towards the sun), Lumen's radiance cache, the clustered local lights
  with one shadow-map tap, emission, then the fog in front — the premultiplied radiance each fragment scales. **Draw**:
  one instanced draw of every slot (dead ones collapse in the vertex shader), camera-facing or stretched along the
  screen-space velocity by the distance covered in 40 ms (sparks), procedural glow / puff sprites, cut by the opaque
  depth and faded in front of it (soft particles), bright emitters bloom.
* **Order-independent transparency** (`oit.rs`, `oit.wgsl`, `effects.rs`). Particles use **weighted blended OIT**
  (McGuire and Bavoil 2013): each fragment adds premultiplied colour × a depth weight to an rgba16f accumulation
  target and multiplies an r16f revealage target by 1 − opacity; a full-screen pass composites average colour ×
  coverage — any number of layers (a shower of sparks, thousands of motes), no sorting. **Order with the translucency
  pass**: when translucent surfaces are drawn, their nearest depth is rendered first (the translucency module's first
  peeling layer, on its own); particles write a *front* or *back* pair of targets and the volume march splits its
  rays at that depth. The back bucket (particles over volume over scene) is composited before the translucency pass —
  glass tints and water refracts the mist and sparks behind it —, the front bucket after it. **TSR**: the front
  composite writes a reactive mask into the motion target's z (free by then; UE's responsive AA): there TSR keeps at
  most 1.5 frames of history (mixed by coverage; volumes at half strength, they have their own temporal filter), a
  tight clip box and no thin-feature lock, so sparks and flames leave no trails. Scenes without emitters, or with `'`
  off, record none of these passes and render as before (valley with `'` off: max 5/255, within run-to-run noise).
  Debug view 21: particle overdraw (red → yellow) and volume march steps (blue → cyan) over a grey albedo;
  empty-space skipping leaves the clear air of a box at a quarter of the steps.
  Cost of volumes + particles + OIT (shared GPU, medium detail, interleaved on/off runs, medians of 3; "passes" = every
  pass the feature adds):

  | View | 1600×900: GPU off → on | passes | 2560×1440: GPU off → on | passes |
  |---|---|---|---|---|
  | Forest campfire close-up (`--eye 4.3,1.3,0.6 --at 7,1.0,-1`) | 9.19 → 9.65 ms | 0.58 ms | 16.4 → 17.2 ms (min) | 1.19 ms |
  | Cathedral censer (`--eye 66.2,1.8,0.4 --at 68.4,2.4,1.9`) | 6.26 → 6.46 ms | 0.38 ms | 10.88 → 11.18 ms (min) | 0.59 ms |
  | Cathedral initial view (24 000 motes, incense far) | 6.36 → 6.81 ms (min) | 0.37 ms (min) | 11.56 → 11.89 ms (min) | 0.45 ms (min) |
  | Valley initial view (lake mist, 100×10×91 cells) | 5.19 → 5.86 ms | 0.53 ms | 8.75 → 9.72 ms | 1.24 ms |

  Per pass: campfire simulation (16 SOR sweeps) 0.10–0.11 ms, incense 0.08 ms (both resolution-independent);
  lighting caches 0.02–0.03 ms; march + temporal resolve 0.15–0.45 ms at 1600×900 (the valley mist covers a third of
  the screen), 0.29–1.07 ms at 2560×1440; particle simulation + lighting 0.02 ms (1 280 fire particles) – 0.09 ms
  (24 000 motes); OIT draw 0.02–0.04 ms; nearest translucent depth 0.01 ms; composites 0.04 / 0.10 ms. GPU memory
  while on screen: 72 bytes per internal pixel (104 MiB at 1600×900) plus the grids (campfire: seven 1 MiB rgba16f
  fields, 1 MiB of pressure and divergence, a 0.1 MiB cache). Temporal stability (static camera, pixels changing by
  more than 24/255): inside the campfire's rectangle 0.11 → 0.42 % (the flames and sparks move), outside it 0.070 →
  0.083 %; cathedral 0.002 → 0.005 %. Not in these numbers (the light stays with `'` off): the campfire's 18 m
  shadow-casting light refreshed every one of its cube-map pages every frame under the swaying ferns, grass, bushes
  and trees of the clearing (+2.0–2.3 ms of `vsm raster` in the forest's default view at 2560×1440: 18.4 ms, 16.6 ms
  with its shadows off); since round 5 its pages are refreshed by the motion of the clusters they hold and drawn
  with tighter face rectangles (16.7–17.6 ms, *Local light virtual shadow maps*).
* **Temporal stability.** The renderer coarsens the LOD threshold when visible work nears the culling list
  capacities, instead of letting the GPU drop arbitrary clusters (which flickers). Lumen clamps ray radiance relative
  to the exposure (as UE's `MaxRayIntensity`: tiny bright emitters such as candle flames otherwise tint random probes
  each frame), places adaptive probes where no uniform screen probe matches a surface (and falls back to the radiance
  cache, sky visibility included, where none does), never clips GI history tighter than ±25 % (probe noise is spatially
  coherent, so a 3×3 neighbourhood underestimates it), filters reflections with
  Karis weights and variance clipping, and traces distance-field shadows of the strongest lights, stopping short of
  each light's own sub-voxel fixture. TSR widens its clip box where the image is still, so sub-pixel foliage averages
  instead of flickering. `helios --dump-dir` + a frame-difference script measure all this (a static camera changes
  ≤ 0.02 % of pixels by more than 24/255 in the valley and cathedral).
* **Temporal super resolution** (`passes/tsr.rs`, `tsr.wgsl`, `resolution.rs`; UE5's TSR and `r.ScreenPercentage` /
  `r.DynamicRes`). `RenderSettings::screen_percentage` renders the scene at 25–100 % of the output; TSR reconstructs
  the output from jittered internal frames. Internal targets are allocated for the largest percentage the settings
  allow and a frame renders into their top-left corner, so dynamic resolution never reallocates: shaders that sample
  internal targets by screen uv map it into that corner (`viewport_uv` in `common.wgsl`), histories at internal
  resolution (GI, reflections, clouds) are reprojected with the previous frame's viewport, the HZB is built and read
  over the covered corner only. Jitter is Halton (2, 3) with `8 / ratio²` phases (8 at 100 %, 32 at 50 %). Per output
  pixel TSR gathers the 3×3 internal samples around it with Gaussian weights of their jittered distance *in output
  pixels* and accumulates them by weight (history alpha counts the samples it holds), so the result converges to the
  scene filtered at output resolution; the history is read with Catmull–Rom, clipped to the neighbourhood's YCoCg
  variance box (wide when still, tight under motion), rejected where the surface's motion differs from the motion
  the history pixel was written with (disocclusion), blended with exposure-aware luminance weights, and pixels with
  less than one frame's worth of samples fill in a wider anti-aliased filter (the responsive path). Below 100 %,
  bright sub-pixel features (candle flames) that only some jitter phases hit are *locked* for one jitter cycle — not
  clipped or luminance-weighted (a lock is carried over only while the pixel is still) — or the clip would erase them. At 100 % it is the former TAA
  (same blend rates and clip box) with a proper reconstruction filter. Texture sampling gets the mip bias
  log2(internal / output) (G-buffer derivatives are scaled, so streamed texture tiles and the RVT follow); Nanite's
  error threshold is in output pixels (TSR resolves sub-internal-pixel geometry, and the cut and streaming don't change
  with the dynamic percentage) while the software-raster heuristic and HZB use internal pixels (what is rasterised);
  shadow pages — the sun's clipmap and the local lights' mips — are selected for the internal resolution's upper
  bound (a fixed percentage scales shadow cost with it, dynamic changes don't invalidate cached pages); pages are
  marked from internal pixels and the local-light shadow mask is computed at internal resolution. Stars keep their
  output-pixel size. **Dynamic resolution** (`DynamicResolution { gpu_budget_ms,
  min_percentage, max_percentage }`, off by default natively, on in the browser) averages the GPU timestamp totals of
  six frames, holds inside 80–100 % of the budget and otherwise steps towards 90 % assuming time ∝ area (which
  under-corrects, so it approaches from one side), in whole percents, ignoring the frames still in flight after a
  change. Lumen, clouds and fog run at the internal resolution (fog froxels per 16 px of the upper bound). Pixels the
  particle / volume composite marks *reactive* keep at most 1.5 frames of history (see *Order-independent
  transparency*).
* **Camera** (`passes/post_fx.rs`). A thin-lens camera (focal length from the field of view and sensor size,
  f-number, autofocus on the screen centre) drives scatter-as-gather depth of field; motion blur reconstructs along
  per-tile dominant velocities with depth-aware weights (McGuire et al. 2012) and a 180° shutter by default. Exposure
  comes from a 64-bin luminance histogram that ignores the darkest and brightest 10 %, so a sun disk or a candle
  flame cannot pump it.
* **Local exposure** (`passes/local_exposure.rs`, as UE5's Local Exposure). One global exposure cannot hold Snell's
  window over the dim lake, sunlit windows over dark aisles or a dawn sky over a shaded valley. A bilateral grid of the
  output image's log luminance (cells of 1/45 of the screen height × 32 bins spanning ±10 stops around the metered
  average; built per cell with linear splatting into workgroup fixed-point atomics — WebGPU has no float atomics —
  then Gaussian-blurred in space and luminance) gives each pixel an edge-aware *base*: the average of its neighbours
  of similar brightness, which does not bleed across a window frame or the rim of Snell's window. Blended with a
  wide plain blur of the log luminance (40 %, against the flat "HDR look" of a purely bilateral base), the base's
  distance from the metered average is scaled by `highlight_contrast` (0.6) above it and `shadow_contrast` (0.85)
  below it, while the detail on top of the base keeps its full contrast. The grid is anchored on the adapted
  exposure, not the frame's content, so it does not pump. It runs after the exposure metering (which still sees
  the scene) and before bloom (a highlight brought down glows less) and tone mapping; `RenderSettings::local_exposure`
  (`None` = off).
* **Time of day.** `Scene::set_time_of_day` keeps the sun as the light through twilight, then hands over to a
  (brightened, bluish) full moon opposite it and fades in a procedural star field (power-law magnitudes, colour
  temperatures, scintillation) behind the clouds; the demo lowers the exposure at night so it stays a night.
* **Profiling** (`profiler.rs`). Every pass is timed with timestamp queries (when the adapter supports them) and shown
  per pass in the HUD — in the browser too.

### 3. Editor, scene files and PCG — `helios-demo`, `helios-pcg`, `web/editor`

* **Editor** (`web/editor.html`, `web/editor/*.js|css`, `crates/helios-demo/src/editor/`, `web_editor.rs`,
  `app_editor.rs`; UE's Level Editor). A dark, UE5-style editor around the live WebGPU viewport, plain ES modules
  without a build step or CDN: toolbar (file, undo / redo, W/E/R gizmo modes, world / local axes, grid, rotation and
  scale snapping, scene switch, Play, Material Graph), a Content browser (engine shapes and nature prefabs, point /
  spot lights, PCG presets, the scene's materials and meshes, glTF import by URL or by dropping a `.glb`), a World
  Outliner (World → Environment / Render settings, lights, PCG volumes, the instances grouped by mesh or material,
  search, eye toggles; virtualised rows, so the forest's 8 278 instances scroll smoothly) and a Details panel
  (transform with scrubbable X/Y/Z fields and a uniform-scale lock, mesh, material slot and the material's
  parameters, graph, Substrate slab and displacement (magnitude, centre, height channel), rendering; light colour, intensity, attenuation radius, source radius,
  shadows, spot aim and cones; PCG volume, surface, seed and every node's parameters; sun / fog / clouds / wind;
  render settings). The engine owns the scene: the UI reads it as JSON and sends edit operations through the wasm
  page API (`heliosEditorState`, `heliosEditorApply`, `heliosEditorPreview`, `heliosEditorPick`, `heliosEditorCamera`,
  `heliosEditorSetSelection`, `heliosEditorSave`, `heliosEditorLoad`, `heliosEditorImport`, …). **Edits are
  documents**: every operation produces a new `SceneDoc` (one undo step; slider drags carry a merge key and form one
  step), and `Editor::sync` reconciles the scene with it, touching only what differs (an instance's transform, a
  material's parameters, one PCG volume's instances), so undo / redo are document swaps and loading a file is the
  same reconcile from an empty document. Instances are never removed — `Scene::set_visible` hides them (the
  `HIDDEN` instance flag, culled for every view; translucency, water bodies, Lumen and assembly expansion skip hidden
  instances) and hidden slots are reused for the next instance of the same mesh — so handles in the outliner, the
  selection and picking stay valid. Transform edits and gizmo previews go through `Scene::set_transform`, i.e. the
  per-instance path of *Dynamic world* (edited instances stay `Static`, so the shadow pages under them are rendered
  in full); hiding, deleting, duplicating, reassigning materials and PCG regeneration are structural changes.
  **Picking** reads the visibility buffer: a request maps the clicked output pixel to the internal resolution, a
  one-thread compute pass (`pick.wgsl`) decodes the visible-cluster entry into its instance (an assembly part into
  its owning tree) and reconstructs the world position and normal, and the 48-byte result is read back
  asynchronously (it arrives 1–2 frames later) — exact per pixel, so alpha-tested holes, swaying leaves, voxels and
  software-rasterised clusters pick what is drawn; translucent surfaces (not in the visibility buffer) are tested
  against their oriented bounds on the CPU and win when nearer. PCG-spawned instances select their volume. The
  **selection outline** (`selection.rs`, `selection.wgsl`) is a full-screen pass after tone mapping that tints
  selected instances and draws a 2-px orange outline from eight visibility taps per output pixel, against a
  per-instance selection bit set; it is only recorded while something is selected. The **gizmo** (translate arrows
  + plane handles + screen-plane centre, rotation rings with angle readout, local-axis scale handles + uniform
  centre) is drawn in a 2D overlay canvas from the engine's unjittered view-projection and hit-tested in screen
  space; drags apply a delta to every selected object around the primary's pivot, preview through
  `heliosEditorPreview` (no undo step) and commit one transform edit on release (Esc cancels); axes pointing at the
  camera switch to screen-space dragging. The overlay also shows light icons (clickable), spot cones, light ranges,
  PCG volume boxes with labels and the bounds of selected instances. Placing content picks the surface under the
  viewport centre (or the drop point) on the GPU. Natively, Tab switches the same editor core on without panels.
  Cost (shared GPU): selection outline 0.05 ms (a first version with 20 taps cost 0.34–0.57 ms), pick ≈ 0 (one
  thread, once per click); with nothing selected the frame is unchanged.
* **Scene files** (`.helios.json`, `editor/doc.rs`, serde). A base scene — a procedural scene and detail (both change
  the instance list) or a glTF URL — plus only the edits: base-instance overrides keyed by index with the mesh name
  for validation (transform as TRS, visibility, material, deleted), added instances by mesh and material *name*,
  material parameter overrides by name, material graphs (`graphs`) and slabs, base light definitions by index and
  added lights, the environment, render-setting overrides by key, imported models (name + URL), PCG volumes (volume,
  surface, graph) **and their generated output** (parts + per-instance TRS), so a file reloads with exactly the
  instances it was saved with. Meshes get stable names where scenes create them (`mesh_store::add` names every baked
  mesh, duplicates become `rock #2`; glTF imports are `MODEL/MESH`), materials get explicit names in the valley and
  the forest and `<shading model> <n>` elsewhere. Save downloads the file (Ctrl+S), Open uses a file picker (Ctrl+O),
  `?scene-file=URL` builds the file's base scene directly; natively `--scene-file PATH` (relative glTF paths resolve
  against the file). The render settings the Details panel shows come from a data-driven registry
  (`editor/settings.rs`: key, label, group, kind, getter / setter) that also stores them in scene files — a new
  setting is one entry.
* **PCG** (`crates/helios-pcg`, `editor/pcg.rs`; UE 5.7's Procedural Content Generation framework). A deterministic
  point graph: `SurfaceSampler` (on the landscape heightfield, one jittered candidate per cell of a world-anchored
  grid of `1/density` m², or on any mesh instance's triangles, area-weighted), filters (`SlopeFilter`,
  `HeightFilter`, `NoiseDensity` — world-space fractal value noise —, `DensityFilter` with stochastic thinning,
  `DistanceFilter` to points / splines with falloff, `BoundsFilter`), `SelfPruning` (Poisson-disk thinning by
  priority = density then seed, hash-grid accelerated), `Difference` and `Merge` (graphs are DAGs: nodes read any
  earlier nodes), `Transform` (scale range and jitter, random yaw, tilt, align to normal, offset) and a weighted
  `Spawner` of prefabs (several mesh / material parts per entry: trunk + foliage, stem + petals). Every random
  number is a hash of the graph seed, the node and the point's own seed (from its world grid cell or triangle), never
  of processing order, so output is bit-identical run to run and moving a volume keeps the points in the area it still
  covers (unit tests: determinism, pruning distance and order independence, slope / distance / bounds filters,
  density vs. area, weights, DAG difference, JSON round trip, presets). In the editor, a PCG volume (oriented box,
  movable / scalable / yaw-rotatable with the gizmo) runs a preset graph — *Rocks on slopes*, *Flowers in meadows*,
  *Forest edge* (trees thickening away from an edge spline, bushes and ferns along it, cleared around the trunks)
  — on built-in engine content (`editor/content.rs`, baked on first use), and regenerates live while a node's
  parameter is dragged (at most once per frame, one undo step per gesture); spawned instances reuse the volume's
  instance slots per mesh. Browser (wasm, headless Chromium): flowers in a 28×28 m volume 3 170 instances, regenerate
  15–20 ms; forest edge (80×50 m) 408 instances, 7–12 ms; rocks on slopes 3–8 ms.
* **Material graph editor** (`web/editor/graph.js`, `editor/materials.rs`). A dark node canvas docks over the
  viewport (toolbar *Material Graph*): pan (right / middle drag), zoom (wheel), drag nodes, drag pin to pin to link
  (an input takes one link), Alt+click unlinks, right-click or double-click opens a searchable palette built from the
  engine's catalog (`heliosMaterialNodes`), inline editors for constants, colours, parameter names and defaults, UV
  sets, channels; *New from preset* (lava, wet stone, stylised water, brick pavement, blank), *Assign to material*; compile errors
  mark their nodes and list below. Every change is a scene-document edit (`{op: "graph"}`: undo / redo, saved in
  `.helios.json` under `graphs`, gestures merged into one undo step) and recompiles live. The Details panel shows a
  material's graph (assign / open), its parameters as sliders and colour pickers (material instance values, saved
  per material), its Substrate slab (clear coat, sheen, subsurface and radius, transmission, anisotropy, IOR,
  specular, unlit) and its displacement — magnitude, centre and the height channel, or *Driven by graph* (opening
  it) while one of the graph's displacement pins is linked: *Height* or *Vector* (no centre), marked *animated*
  when the graph reads the time; linking a node to such a pin re-dices the surface live. The Output node sets its
  two geometry pins (Displacement, VectorDisplacement) apart with a tooltip each, and the status line names what
  the graph does to geometry ("vector displacement", "animated displacement"; round 6: the spaces a TransformVector
  turns into world space — "tangent-space vector displacement" — and an animation's speed bound, "≤ 1.64 ×
  magnitude/s", or "no speed bound"). TransformVector nodes pick their spaces from two drop-downs; the Details
  panel's note on an animated material gives its speed bound in cm/s (or says its shadows refresh every frame).

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
| No wasm threads without cross-origin isolation (`SharedArrayBuffer` needs COOP/COEP headers, which GitHub Pages cannot send) | Lumen's displaced-variant bakes run on Web Workers that each instantiate the page's compiled wasm module and exchange bytes with it (`bake_worker.js`); other start-up work (scene build, texture encoding) runs on the render thread |
## Status vs. Unreal Engine 5 — what is still simpler

Honest differences that remain:

1. **Streaming**: page payloads use a quantised, bit-packed GPU format like Nanite's, but are not entropy-coded on
   disk on top of it (Nanite adds Oodle): measured per page, LZ4 would save only 0.4–5 % and DEFLATE 3–16 %, so `.hvm`
   files store the GPU format as is. Triangle indices are plain fixed-width (no strip / delta coding). Every mesh's
   header — including all cluster records — stays resident. Cached shadow pages are invalidated per instance for
   moves, streaming, wind and skeletal animation; structural scene changes (instances added or hidden, materials
   reassigned, shadow casting toggled, new geometry), moves that grow the scene bounds and toggling voxel LODs or tessellation
   invalidate them globally.
2. **Lumen**'s mesh cards are axis-aligned (6 directions × 3 depth layers per mesh, no per-mesh card fitting as in UE5),
   carry the material's *average* albedo (no per-texel textures), are lit with one bounce from the radiance cache
   rather than card-to-card radiosity, and local lights shadow them through the global field only; surfaces no card
   sees (deep concavities, sub-texel slivers) and instances beyond 100 m use the voxel surface cache, which still leaks
   a little light through thin walls; translucent surfaces get no Lumen
   front-layer reflections (screen traces, the global field and the radiance cache instead). Mesh SDFs have at most
   48 voxels along their longest axis, so very large meshes are no sharper in the near field than in the global one.
   Adaptive screen probes are capped at half the uniform count, and detailed views exceed that (at 1600×900 the
   cathedral, the valley and the forest all request more), so they are thinned at random each frame; they cost
   0.3–0.4 ms at 1600×900 (the final gather goes from 0.5–0.55 to 0.8–0.95 ms). Remaining slow flicker on thin geometry
   comes from the surface caches' incremental updates (cards, voxels, radiance cache), not from probe placement.
   The surface caches see neither skinned meshes, volumes nor particles (only a fire's driven light), and follow
   moved instances over a few frames (item 9); they see displaced surfaces to the resolution of their baked
   variants, animated ones as their mean surface (item 14). **Lumen Lite**'s irradiance field covers ±64 m around
   the camera (4 levels of 16³ probes; beyond, Lumen's radiance cache with an estimated sky visibility); probes are
   refreshed within a fixed budget of 512 a frame (128 rays each), so lighting changes reach distant and off-screen probes over a
   few hundred milliseconds and a refreshed probe changes in small steps (fine grain in the indirect light: 1.05 % of
   pixels change by > 4/255 per frame on a static camera vs 0.30 % with screen probes); classification uses the
   global distance field only (alpha-masked foliage is not in it, so probes in tree crowns can sleep); there is no
   per-probe variability-driven update rate or probe-to-probe irradiance feedback into the surface cache (the bounce
   still comes from the radiance cache); glossy reflections still trace per pixel; the adapter-based default can only
   see integrated GPUs natively (browsers hide the adapter type, so there only the dynamic-resolution fallback
   switches).
3. **Materials**: streamed image tiles are block-compressed only where BC is available (no ETC2 / ASTC encoders: mobile
   GPUs get uncompressed tiles), are encoded at load (natively) or on demand (web) rather than cooked offline, and are
   cut from decoded images kept in CPU memory (no tiled on-disk cache); the 512² texture arrays are uncompressed;
   triplanar mappings and passes other than the G-buffer resolve (Lumen, landscape RVT) see the 512² layers;
   procedural textures are not virtualised. glTF: the images of one layer (base colour, normal, metallic-roughness)
   must share their coordinates — a normal or metallic-roughness map with another UV set or transform than the base
   colour is sampled with the base colour's (with a warning); UV sets beyond `TEXCOORD_1`, `KHR_materials_iridescence`,
   `_dispersion` and `anisotropyTexture` (the anisotropy factors are) are not imported; the clear-coat, sheen and
   specular textures are sampled with the main texture's coordinates from the 512² arrays (not streamed);
   `KHR_materials_transmission` becomes refracting glass (refraction of the screen, not a thin-walled BTDF;
   `transmissionTexture` and `thicknessTexture` are ignored). Alpha tests ignore wrap modes and vertex alpha. Lumen's
   cards and voxels see a material's average albedo, without vertex colours, texture transforms, AO, slabs or graphs
   (a lava graph glows on screen but bounces its base colour). Simplification keeps second-UV and colour seams but
   does not weigh their deviation. Alpha tests read image textures with mesh UVs only (not procedural textures or
   triplanar mapping), with 8-bit cutoffs. In Lumen an alpha-masked instance is partially transparent by
   one average coverage (its partial shadows and occlusion are uniform, not shaped like its holes), only in the near
   field: rays and local-light shadows see it within 2 m of their ends, beyond which the global field lets light
   through; its cards are captured from the geometry, holes and all, with the material's average albedo.
   **Substrate-lite** has one slab per material: no vertical layering of arbitrary slabs, no horizontal mixing (UE's
   Substrate mixes / layers / blends any number, with per-pixel slab counts and a closure-count–adaptive G-buffer); no
   hazy gloss, iridescence, dispersion, glint or fuzz beyond sheen, no specular profiles, no volumetric media. A slab
   with a clear coat ignores anisotropy (they share the texel's direction), and subsurface + transmission share one
   per-pixel amount (transmission then takes the material constant). Subsurface diffusion is screen-space (no light
   from behind thick objects except through thin transmission, no subsurface for translucent materials), uses the
   material's colour and radius rather than a measured profile, and under MegaLights the local lights' diffuse light
   of subsurface pixels is not diffused (MegaLights keeps diffuse light whole). Translucent materials (glass, water,
   `BLEND`) ignore slabs. Unlit colours are scaled to a white surface in full sun. Devices limited to WebGPU's default
   16 sampled textures per stage render slab materials as their plain base. **Material graphs** run in the G-buffer
   resolve and, for their displacement (a height or a vector), in the tessellator: no graph-driven opacity mask, world position
   offset, translucent or decal materials (raster and translucency shaders are fixed); at most 16 distinct (shading model, graph) pipelines per frame (further
   materials render without their graph, with a warning) and 1024 parameter slots in all; no custom HLSL/WGSL node,
   material functions, static switches, texture objects or per-node previews; noise and texture outputs carry no
   derivatives (a texture read at coordinates computed from another texture or from noise filters as if they were
   constant, i.e. at its finest level there); texture parameters are not editable in the UI (they default to the
   material's texture) or stored in scene files; time is the renderer's clock.
4. **Local lights**: at most 32 lights per frame get virtual shadow maps, sharing the sun's 1024-page physical pool
   (UE's default pool is four times larger), so local lights request pages half a mip coarser than the screen
   footprint; a moving light's shadow map is not re-rendered every frame as UE's is but each cube face is held at an
   origin within 0.1 source radii and parallax-corrected in the soft-shadow filter (lights faster than that are
   re-rendered every frame, as in UE, while a budget of pages per frame pays for it): exact at contact and for
   occluders the blocker search finds, while shadows of occluders right next to a large moving light stay where the
   origin casts them, single-tap lookups (fog, lights beyond a pixel's four filtered ones) are uncorrected, the map is
   a mip coarser, lights the budget cannot follow are held at up to four times their tolerance (beyond that,
   distance fields until they rest), and the budget counts pages, not their cost in draws. With the classic
   clustered loop, the rest get distance-field
   shadows for the strongest few per pixel and no shadows beyond those (light leaks through walls), and shadow-map
   soft shadows are PCSS-filtered, not ray-marched. **MegaLights** shadows every light but traces no hardware rays
   (WebGPU has none): beyond the 1 m
   screen trace, rays use the light's virtual shadow map (only the ≤ 32 static lights with one; SMRT-style, 8 steps) or
   Lumen's distance fields, which are coarser than triangles (mesh SDFs of at most 48 voxels per axis, 0.25 m global
   voxels beyond 2 m from both ends). The screen trace marches the depth buffer linearly (8 steps, no HZB traversal:
   the HZB holds the farthest depth, made for culling). Samples come from one representative pixel per 2×2 block (all
   four pixels shade with its pdf; edges between surfaces converge over the rotating representative), always four per
   block (no adaptive sample count, no "skip rays outside penumbrae" of UE 5.8, no per-light-type budget); guiding uses
   a 256-bit hash per tile (lights 256 indices apart share a bit). The sun stays on its classic path (virtual shadow
   map), and so do volumetric fog (the clustered lights, shadowed only by the shadow-mapped ones), translucent
   surfaces (forward clustered loop) and particles: with hundreds of lights they still see the unshadowed ones.
   Lumen's surface cache still lights every card texel and voxel with every light in range (one global-field shadow
   cone each), which becomes the dominant light-count-dependent cost (cathedral 33 → 1000 lights: surface + radiance
   cache 1.5 → 2.4 ms, cards 0.07 → 0.35 ms, in both modes). Light culling stays brute force per froxel (0.01 →
   0.30 ms at 1000 lights; up to 255 lights per froxel, 1024 per scene). The denoiser lags behind moving lights by up to
   24 frames where the variance clip does not catch the change, and its spatial filter softens lighting detail on
   fresh (disoccluded) pixels.
5. **Translucency** is ordered per pixel up to 4 layers deep (depth peeling; deeper layers fall back to per-instance
   order). The layer count comes from projected bounding boxes, so boxes that overlap on screen cost up to three extra
   depth-only draws, scene-colour copies and shading passes even where the surfaces themselves don't. Refraction reads
   the screen-space copy (what is off screen or hidden is not refracted; where a nearer layer covers the displaced
   lookup it falls back to the pixel behind); translucent surfaces don't request shadow pages (where the
   opaque scene didn't request theirs, they fall back to coarser resident levels) and are drawn without LOD. **Water**
   is a normal-mapped plane (no displacement or tessellation, no FFT spectrum, no simulation or interaction with
   objects, no flow maps or rivers / oceans, no spline-shaped bodies: a body is its instance's bounding rectangle, at
   most 4 per scene); the waves ignore `Scene::wind`. Caustics come from the analytic waves (a single preimage, so
   overlapping folds are not summed; no chromatic dispersion) and only light the bed seen through the water or from
   inside it, not the fog or other translucent surfaces. Underwater fog is lit by the light entering the surface above
   the middle of each view ray (no light shafts or volumetric shadowing along it), total internal reflection shows
   deep water rather than the bed, and a camera straddling the surface gets no split waterline view. The shore
   material is procedural (no texture layers, no wet-band wave run-up animation).
6. **Temporal super resolution** keeps one history (colour + sample weight, motion + lock) at output resolution, not
   UE's 200 % history option or its separate anti-flicker luma history; disocclusion is judged by motion vectors only
   (no depth history); translucency, Lumen and fog have no separate upscaling (they render at the internal resolution
   and TSR resolves them with everything else); alpha tests in the rasterisers don't get the mip bias (sharper cut-outs
   would mostly add sub-pixel shimmer to foliage). Dynamic resolution only reacts to GPU time (no CPU-bound
   detection) and needs timestamp queries.
7. **Wind** is procedural (no pivot-painter data per branch or leaf, no simulation) and does not rotate normals;
   leaves flutter around their position rather than their stem. Shadow pages cache static casters separately (as
   UE's static cache); the wind, `Mobility::Movable` and animated skins make a caster dynamic. Refreshes are decided
   per instance and redraw every dynamic caster in the
   page, so in the forest — where trees, bushes, ferns and grass all sway and only terrain, rocks and logs are
   static — the wind still costs most of what it did (see *Wind*); a page gets its static layer by being rendered in
   full once more at its first dynamic refresh, and only 256 pages a frame have scratch space for software-rasterised
   dynamic casters (beyond that they take the hardware path). The wind doesn't move distance fields, the surface cache
   or translucent materials (rigid moves do).
8. **Skeletal meshes**: linear blend skinning with up to 4 influences per vertex (glTF's first joint set; UE allows
   8–12) and 8-bit weights; no dual-quaternion skinning, morph targets (imported files with them render their base
   shape, with a warning), cloth or physics. Culling bounds are per instance (posed joint spheres) and per cluster only
   as the rest sphere padded by the instance's largest displacement — UE bounds clusters by their own influencing
   bones, which is tighter for large motions (a waving arm makes every cluster's sphere grow by the arm's reach). The
   LOD error is that of the rest pose: the weight-deviation cost only orders the collapses (coarse levels keep weight
   seams longest; a test measures half the posed deviation), it does not enter the error. Skinned meshes get no
   distance field, mesh cards or voxel LODs (as in UE: Lumen sees them through screen traces only), and translucent
   (`BLEND`) skinned surfaces are drawn by the forward pass in their rest pose. Shadow pages under a moving skin
   re-render its whole caster set each refresh (as for the wind); the footprint grows with the largest reach seen, so
   a figure that once stretched far refreshes a larger area. Root motion is not extracted (animated root joints move
   the posed bounds, not the instance).
9. **Dynamic world**: moves are rigid transforms of whole instances (deformation is skinning, item 8). A moved
   caster's shadow footprint is its oriented box in light space, cut into at most 8 slices along its longest axis,
   so a thin caster with a wide part (a chandelier's chain above its ring) still refreshes some pages its geometry
   doesn't touch; UE's instance-culled page invalidation is tighter. A move's speed bound is its current displacement
   held with a decay, not a true bound: a caster that slows down sharply can leave a page a few texels stale until it
   moves on or stops. A fixture around its own moving light redraws that light's pages every frame (item 4). The
   global distance field lags by up to a quarter voxel per level (and further for a fast mover beyond the brick
   budget); an instance that stops within that distance of where a level last composed it
   stays there. The voxel surface cache and radiance cache follow moves over a few frames instead of being
   invalidated locally. Instance records are rewritten with one `write_buffer` per contiguous run, not a GPU scatter
   of dirty records (UE's GPU-scene update); adding, hiding or removing instances still rebuilds all instance data.
   The landscape (runtime virtual texture) and water bodies' lake beds don't follow a moved terrain instance.
10. **Nanite Foliage**: voxels carry per-axis coverage, one normal and their brick's mean texture coordinate — no
    per-voxel colour, so textures varying within a brick, vertex colours and alpha-masked (card) foliage's mask
    coverage are averaged away or ignored; coverage along a diagonal is modelled from the three axes (exact for
    axis-aligned and isotropic leaves). Voxels are rasterised as stochastic cubes (no ray-marched depth buckets or
    front-to-back ordering), rely on TSR to average their dithering, move rigidly with the wind at their centre (no
    flutter, no bone-driven wind), and have no hardware path: in shadow pages refreshed over a static layer without a
    scratch tile, or split pages, they sit out that refresh. Voxels win only where pruning reports a larger error than
    the voxel budget allows (4 voxels per kept triangle: the forest's conifers and ferns from mid-distance, the
    broadleaf canopies near their roots). Assemblies are one level deep (no nested parts), their parts take the
    instance's material (wood and leaves are separate instances), and the base ↔ parts switch is per assembly
    instance, not per part (UE resolves part instances inside the cluster hierarchy): an expanded tree still writes
    every part as an instance record each frame (≈ 0.03 ms for the forest's ≈ 26 k conifer parts) and is culled
    through one record per tree whose lanes are its parts' pages (round 6: the parts' culling cost fell from ≈ 0.6 to
    ≈ 0.1 ms), and parts are only as coarse as their own small DAG allows, so the expanded range draws more
    triangles and fewer voxels than one merged mesh would. When more parts are wanted than the 32768 slots, the
    expansion threshold rises, so trees near the edge of the expanded range draw their base up to ~1.5 px coarse.
    Parts are rigid copies (a needle spray follows its branch span, not each branch's own curve); part instances'
    shadow pages are not invalidated by their pages streaming in (the base's are).
11. **Volumes and particles**: volumes are dense grids per emitter (no sparse / VDB volumes, no streaming of
    sequences, at most 8 emitters simulated and 4 marched per frame, each up to 192 steps per ray); the fluid solver
    is a single-grid red-black SOR (no multigrid), collocated (no MAC grid), with a solid floor only (no obstacles from
    the scene's geometry, no interaction with other emitters or particles) and a fixed time step clamped to 1/30 s.
    The lighting cache holds the sun's transmittance plus isotropic ambient and up to 8 local lights (no
    self-shadowing for local lights; multiple scattering is one approximate octave of the sun, not simulated), the
    march samples it trilinearly at half the grid resolution; close up, a 4 cm campfire grid shows its resolution in
    the smoke's edges. Volumes don't cast shadows on the scene or into the virtual shadow map, don't feed Lumen (only
    the fire's driven light does) and are not seen by reflections or refraction outside the screen. Overlapping
    volume boxes are composited box by box in entry order, not interleaved; particles go over the volume of their
    bucket (sparks inside the smoke are not attenuated by the smoke in front of them). Particles: weighted blended OIT
    approximates the order of layers at similar depths and saturates for very many opaque layers; one translucent
    surface (the nearest) splits the order — particles between two glass layers are composited with the farther
    bucket; collision only against the visible depth buffer (no off-screen or occluded collisions, no distance-field
    collision), no ribbons, meshes, sub-UV flipbooks, events or GPU sorting; lighting is per particle (no per-pixel
    normal-mapped lighting). Memory: 72 bytes per internal pixel while volumes and particles are on screen (volume
    march + history front/back, OIT targets, nearest translucent depth: 104 MiB at 1600×900).
12. **Editor**: no actors or hierarchy — a tree's trunk and foliage are two instances, selected separately; deleted
    instances and lights stay as hidden / disabled slots (reused, never compacted); undo history is in memory (200
    steps); no landscape sculpting or painting, no multi-viewport, no editor-only billboards in the renderer (light and
    volume icons are a 2D overlay without depth test), the gizmo draws over everything; typed transform values apply to
    the primary selection only; the native window has the editor core without panels. Edited instances stay
    `Static`, so the shadow pages under a dragged object are rendered in full every preview frame, and hiding,
    deleting, duplicating, reassigning materials and PCG regeneration are structural changes (all instance data
    re-uploaded, Lumen's instance lists rebuilt). Scene files address base instances by build order (a changed scene
    builder is detected by mesh name and the edit skipped with a warning); dropped `.glb` files are referenced by file
    name and must be hosted next to the page to reload; glTF imports bake on the main thread in the browser.
13. **PCG**: a fixed node set without a graph canvas (nodes and their parameters are edited in the Details panel; the
    graph's structure comes from the presets / the file), no attribute system or custom nodes, no spline editing in
    the viewport (distance targets are volume-local data), spawners reference meshes and materials by name, generation
    runs on the main thread (no partitioned / hierarchical generation, no runtime generation around the camera), and
    spawned instances are not individually editable. Native and wasm builds may differ in the last bits of `sin` /
    `cos`, so a point exactly at a filter threshold could flip between them; saved output sidesteps it.
14. **Nanite Tessellation**: displacement is a height along the interpolated normal — a texture channel through
    the material's mapping or a material graph's Displacement output — or a graph's world-space vector, which a
    TransformVector node can build in object or tangent space (round 6). The displacement's tangent frame is not
    UE's: meshes store no tangents, and a per-triangle UV frame would crack, so it is the interpolated normal with
    the object X axis projected into its plane — the mesh's U direction on planes and terrain, elsewhere another
    orientation, and it swirls around a curved surface's points facing ±X. Tessellation factors come from the
    undisplaced edges, so where a vector stretches the surface micro-triangles grow with it (silhouettes coarsen;
    shading stays smooth). Textures are read from the 512² texture array layer (streamed image tiles are not
    consulted) at a mip chosen from the dice segment size, so heights finer than the layer's texels are lost. A
    graph's displacement may read the time but not the camera, vertex colours, UV set 1 or the shading tangent
    frame (Triplanar's TangentNormal; compile errors). An animated displacement's speed bound (round 6) is the
    material compiler's conservative analysis of the graph: graphs it cannot bound (the time through `Frac` /
    `Floor` / `Step` / `If`, scrolled textures, `Time` to a power) refresh every cached shadow page whose texels are
    finer than their reach every frame, the bound is per instance rather than per page, and close up a fast surface
    still refreshes its finest levels every frame (UE re-renders WPO-animated casters every frame). Lumen follows
    displacement through baked *variants* (distance field and cards per mesh and displacement): accurate to the
    mesh's voxels and the cards' ~10 cm texels, so relief finer than those (mortar joints, cobbles) only reaches it
    through screen traces; an animated displacement is represented by its mean surface (8 s average), not the moving
    one; a variant takes a few frames to bake after a displacement changes (0.5–1.3 s of CPU time per variant, on
    worker threads natively and on Web Workers in the browser, where the render thread still copies each result in —
    up to ≈ 5 ms for a 124 k-triangle variant — and Lumen re-uploads its scene when a scene's variants are in, as on
    any structural change); a world-dependent displacement of a moving instance keeps the variant baked where the instance stood;
    meshes baked without a distance field (`SdfMode::None`) stay out of Lumen, displaced or not. A graph displaces
    only while it has one of the 16 extended resolve pipelines (else its materials stay flat); every displacing
    graph of the scene runs in one dicing permutation behind a switch on the material, compiled when the set
    changes (a hitch while a graph is edited). Dicing uses a uniform 2^k grid per patch with
    snapped edges instead of UE's precomputed table of patterns for every factor triple (more micro-triangles for long
    thin patches; F is set by the finest edge), and splitting is binary at edge midpoints (UE splits with wider
    factors). Hard edges (split normals) and UV seams under a UV-mapped height crack, as in UE: the two sides
    displace a shared position differently. Patches are culled by frustum (main view) or page (shadows) only, not by
    the HZB or backfacing. Micro-triangles larger than 4096 pixels (a cliff in the height) are dropped and ones
    crossing the near plane culled — there is no hardware fallback for them; the shadow pages' hardware fallbacks
    (local light items reaching the near plane, dynamic pages without a scratch tile, dynamic casters of split pages)
    draw displaced clusters undisplaced. Lists hold 512 k patches per split queue and 256 k per dice level: when
    they overflow, the dice rate is raised until they fit (HUD). The visibility payload names the base triangle
    and a second per-pixel target the point on it (16-bit barycentrics, where UE's visibility buffer encodes the
    patch and micro-triangle); texture derivatives are the base plane's. Skinned, alpha-masked, translucent and
    voxel geometry is not tessellated; the wind does not
    rotate displaced normals.
