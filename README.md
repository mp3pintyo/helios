# Helios

A Rust + WebGPU (wgpu) rendering library that implements the core ideas behind Unreal Engine 5's
**Nanite** (virtual geometry with streaming and a software rasteriser), **Virtual Shadow Maps**,
**Lumen** (software ray-traced global illumination and reflections) and its material / lighting
stack, running natively and in the browser.

**Live showcase:** https://mp3pintyo.github.io/helios/ (needs a WebGPU browser, e.g. recent Chrome or Edge).
Redeploy with `python site/build.py` then `python site/deploy_pages.py`.

```
crates/
  helios-geometry   pure Rust, no GPU: clustering, METIS-style grouping, seam-aware QEM simplification,
                    LOD DAG + streaming pages, mesh distance fields, procedural modelling and vegetation
  helios-asset      pure Rust: glTF 2.0 import (.gltf / .glb) -> meshes, node instances, PBR materials, RGBA8 images
  helios-render     wgpu renderer: GPU-driven culling, HW + SW raster, streaming, materials, VSM, Lumen, post
  helios-demo       three demo scenes (valley, cathedral, forest): native window, headless PNG capture, wasm
web/                index.html + generated pkg/ for the browser build
```

## Quick start

```powershell
# Native (interactive). Scenes: valley, cathedral, forest. Detail: medium | high.
cargo run -p helios-demo --release -- --scene cathedral

# Headless screenshot (CI / regression images)
cargo run -p helios-demo --release -- --scene forest --screenshot out.png --size 1600x900 --frames 120 --detail medium

# Browser (needs WebGPU: Chrome/Edge 113+, Safari 26+, Firefox 141+)
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130
./build-web.ps1
python -m http.server 8080 --directory web     # open http://localhost:8080/?scene=cathedral

# Your own content: a glTF 2.0 model (.glb, or .gltf with its files)
cargo run -p helios-demo --release -- --gltf path/to/Sponza.gltf      # [--no-ground]
#   browser: http://localhost:8080/?gltf=models/DamagedHelmet.glb (fetched relative to the page)
#   browser resolution: dynamic resolution (14 ms GPU budget) is on by default; ?sp=67 fixes the screen
#   percentage, ?budget=10 changes the budget, ?budget=0 turns it off

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
underwater view.

Headless flags: `--scene`, `--detail`, `--size WxH`, `--frames N`, `--view i` (debug view; 14 = Lumen mesh cards, 15 = local light shadow pages), `--eye x,h,z` /
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
so its shadow map is re-rendered every frame).

**Streaming from files.** `--stream-from` / `?stream=` build the same procedural scene but substitute every baked
mesh with its pre-baked counterpart: only the header (cluster and page records, dependencies, bounds, distance field)
and the root pages are loaded; all other pages are read on demand. Forest (high): 43 MiB of geometry in RAM when
baked vs 3.7 MiB streamed, scene ready in 0.9 s instead of 8 s. In the browser, `python -m http.server` ignores
`Range` headers — it still works (each `.hvm` is then downloaded whole, once), but `python serve.py` (or any server
honouring `Range`, e.g. a CDN) streams pages on demand. Imported glTF models are not streamed (they bake at load).

## The demo scenes

| Scene | What it shows | Size |
|---|---|---|
| `valley` | Mountain valley, 600 boulders, a gold sculpture and a small lake; terrain textured through a runtime virtual texture; at the lake's north end, out of the initial view, a glass pane and two crossing tinted panels standing in the water (layered translucency test) | 612 instances, 50 M (medium) / 199 M (high) source triangles |
| `cathedral` | Gothic basilica: 11-bay nave with clustered piers, triforium, stained-glass clerestory and aisles, quadripartite rib vaults, flying buttresses, polygonal apse with half dome, west front with rose window and twin spired towers; pews, altar, candelabra and chandeliers with 33 candle lights; alpha-tested wrought-iron choir screens; at the west end of the north aisle, behind the initial view, a stained-glass screen and two crossing tinted glass panes (layered translucency test) | 457 instances, 4 M (medium) / 18 M (high) |
| `forest` | Mixed broadleaf / conifer forest with real leaf geometry, bushes, ferns, grass, mossy boulders, fallen logs and mushrooms at golden hour, swaying in a gusty breeze | 8 261 instances, 82 M (medium) source triangles |

These are procedural: meshes come from `helios_geometry::modeling` / `vegetation`, textures from GPU
texture synthesis (`TexturePattern`).

`model` (`--gltf FILE`, `?gltf=URL`) builds a scene from a glTF 2.0 file: every unique mesh (glTF primitive) is baked
once into virtual geometry — in parallel, with a distance field sized to its world extent — and placed by the file's
node instances; materials become metallic-roughness PBR materials whose base colour, normal, metallic-roughness and
occlusion images are resampled into one texture layer and streamed at their native resolution (see *Streaming virtual
textures*). `N` cycles through it with the procedural scenes. Beyond the core: a second UV set (`TEXCOORD_1`) for any
texture, `KHR_texture_transform` (offset, rotation, scale, `texCoord` override) and sampler wrap modes per texture,
vertex colours (`COLOR_0`), ambient occlusion maps (indirect light only), `KHR_materials_emissive_strength` and
`KHR_mesh_quantization` (8/16-bit normalized or integer attributes, interleaved or sparse accessors). Checked against
the Khronos sample models TextureTransformTest, TextureTransformMultiTest (every row but clearcoat and unlit),
MultiUVTest, VertexColorTest, TextureCoordinateTest, EmissiveStrengthTest, the quantized Lantern and Avocado,
DamagedHelmet and WaterBottle. Alpha-masked materials (`alphaMode: MASK`: foliage
cards, chains) are alpha-tested; `BLEND` materials become `MaterialKind::Translucent` (forward translucency pass,
coverage = base colour alpha × texture alpha); their meshes get no distance field and their instances cast no shadow.

## Showcase site

`site/index.html` is a bilingual (Hungarian / English) page that embeds the engine: a pinned live viewport with live
telemetry (GPU time, resolution, triangles, per-pass GPU time against a 16.7 ms frame) next to chapters whose
buttons drive it (scene, camera, sun, debug views, feature toggles), before/after sliders, camera-tour videos and
the story of how Helios was built. It talks to the engine through the page API in `crates/helios-demo/src/web.rs`:
`globalThis.heliosConfig = { scene, embedded, screenPercentage, gpuBudgetMs }` before loading the module, then
`heliosLoadScene`, `heliosKey`, `heliosLookAt`, `heliosSetSun`, `heliosSetView`, `heliosSetPaused`, `heliosHud` and
`heliosScene`. `embedded: true` keeps keyboard input on the focused canvas and leaves the mouse wheel to the page.

```powershell
./build-web.ps1                     # web/pkg
cargo build --release -p helios-demo
python site/capture.py all          # stills + tour videos into site/media-src (needs ffmpeg; HELIOS_ASSETS = glTF samples)
python site/build.py                # site/dist: page, pkg, the media the page references
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

Any indexed triangle mesh works (`helios_geometry::Mesh`); `Mesh::weld` merges identical vertices while keeping UV
seams and hard edges as separate wedges. Baking is deterministic and can run offline: `VirtualMesh::to_bytes` /
`from_bytes` (`.hvm`, version 3: compressed page payloads last, so a reader can stream page byte ranges; version 2
files, which held float vertices, are rejected with a request to bake again).

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
sky LUTs ─ light culling (froxels)
cull instances → page batches → pages (residency + LOD prune) → clusters (EARLY: visible last frame)
  → HW raster (big triangles; alpha-masked ones in a second bin) + SW raster (tiny triangles) → merge → HZB
  → cull clusters (LATE vs HZB) → HW + SW raster → merge                          ⇒ visibility buffer + depth
material classification (8×8 tiles) → one resolve pipeline per shading model     ⇒ G-buffer
runtime virtual texture: allocate + render requested landscape pages
virtual shadow maps: mark (sun + local lights) → page cache → clear dirty pages → cull per clipmap level and once
  for all local lights → (cluster, page) draws
Lumen: global SDF clipmap → surface cache (mesh cards + voxel lighting) → radiance cache → screen probes → reflections
clouds: half-resolution checkerboard march → temporal · cloud shadow map
volumetric fog: froxel injection (sun and local lights via VSM, radiance cache) → temporal → front-to-back integration
deferred lighting (sun + VSM + cloud shadows, clustered local lights + VSM / SDF shadows, GI, reflections, sky +
  stars + clouds, fog)
  → translucency (depth-peel overlapping layers → copy lit scene + mips → forward glass / water / blended draws,
    per layer back to front)
  → TSR (internal → output resolution) → depth of field → motion blur → histogram exposure → local exposure → bloom
  → ACES
```

Everything before TSR runs at the internal resolution (the screen percentage), everything after it at the output
resolution.

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
  pages whose coarsest parent is already detailed enough.
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
  terrain, metal, foliage) is resolved by its own pipeline over just its tiles. UV mapping uses analytic
  barycentric derivatives for proper filtering; per-triangle tangent frames replace stored tangents; triplanar
  mapping is available in world or object space.
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
  1.3 ms). Streaming and wind invalidate only the cached pages under
  the instances concerned (see *Wind*).
  **Static and dynamic casters** are cached separately, as in UE5's "static separate" page cache: casters the wind
  moves this frame are *dynamic*, all others *static*. A page that must be refreshed only because dynamic casters
  moved keeps the depth of its static casters as a *static layer* in its tile of the software raster's u32 mirror —
  otherwise just scratch space between the software raster and the page clear, so the layers cost no memory of their
  own — and its clear writes that layer instead of the far plane; culling drops static instances at the instance test
  (each level keeps a second dirty-page mask and rectangle for pages rendered in full, each local light a second box),
  only dynamic clusters are expanded and drawn, and their software-rasterised ones go to one of 256 scratch tiles after
  the atlas' tiles (16 MiB), which the page clear max-merges over the static layer. A page gets its static layer at
  its first such refresh: it is rendered in full once more, static casters first, whose depth is then copied into the
  mirror (`vsm_static.wgsl`) before the dynamic ones are drawn. Anything that changes static casters — the sun or a
  light moving, a scene change, streaming of static instances, the wind being switched on or off (per instance: no
  global invalidation) — frees the page, so it is rendered in full. A caster fading into or out of the wind as the
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
* **Local lights** (`lights.rs`). Point and spot lights are culled into 16×9×24 froxels. Lights are also injected into
  the surface cache, so they bounce.
* **Local light virtual shadow maps** (`vsm_local.rs`, `vsm_local_common.wgsl`, `vsm_local_sample.wgsl`). As in UE5,
  up to 32 shadow-casting point and spot lights per frame — the most influential on screen (range sphere coverage ×
  brightness, with hysteresis so slots and their pages stick) — get virtual shadow maps: six cube faces for a point
  light, one perspective face for a spot light, each a 4096² virtual map with a mip chain down to one 128² page.
  Their pages are entries of the sun's page cache after its clipmap levels and live in the same physical atlas, so
  the sun and all lights share one pool, one allocation, one dirty list and the same page renderer (the (cluster,
  page) expansion and the hardware and software page rasterisers take a perspective projection per face; the hardware
  path clips at the light's near plane, software items must lie in front of it). Every lit pixel requests, for each
  light of its froxel with a shadow map, the page of the mip whose texels match its screen footprint (half a mip
  coarser by default, `local_shadow_resolution_bias`; lights with a large source radius coarser still) plus the
  coarsest page of that face; volumetric fog requests coarse pages. Pages stay cached while the light and the
  geometry are static: a light that moves or changes shape, a scene change or a slot handed to another light drops
  that light's pages. Streaming and the wind refresh local pages like the sun's (*Wind*), in each light's terms: an
  instance's footprint (padded by its wind reach) refreshes the pages it projects onto in a light's faces and mips,
  once it may have moved 2 of their texels at its depth along the face — lights' fine pages near swaying foliage every
  frame, coarse ones rarely (forest at night, lantern among ferns and grass: ~65 local pages a frame, re-rendering
  only their swaying casters over the cached terrain; hardware page raster of the whole night frame, moonlight
  included, 0.93 → 0.80 ms). One
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
  change. Lumen, clouds and fog run at the internal resolution (fog froxels per 16 px of the upper bound).
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

### WebGPU constraints that shaped the design

| Constraint | Consequence |
|---|---|
| No 64-bit atomics | Software raster writes depth and payload in two passes |
| No storage-texture atomics, no storage depth textures | Shadow software raster writes a u32 buffer that the page clear copies into the depth atlas; the same buffer keeps the pages' static layers |
| No multi-draw-indirect / `first_instance` | One instanced indirect draw per pass; late passes offset slots via the early count |
| 8 storage buffers per stage | Culling packs feedback + visibility bits into one buffer; materials live in a uniform array |
| No bindless textures | Material textures are layers of two texture arrays; image detail beyond 512² lives in tile atlases behind a page table |
| No viewport arrays | VSM pages are rasterised with per-draw tile transforms and clip distances (optional feature) or fragment clipping |
| No forward-progress guarantee | Hierarchy is flattened into instance → page-batch → page → cluster dispatches |
| Indirect dispatch ≤ 65535 per dimension | Large dispatches are 2D |

## Status vs. Unreal Engine 5 — what is still simpler

Everything that was on the roadmap is implemented. Honest differences that remain:

1. **Streaming**: page payloads use a quantised, bit-packed GPU format like Nanite's, but are not entropy-coded on
   disk on top of it (Nanite adds Oodle): measured per page, LZ4 would save only 0.4–5 % and DEFLATE 3–16 %, so `.hvm`
   files store the GPU format as is. Triangle indices are plain fixed-width (no strip / delta coding). Every mesh's
   header — including all cluster records — stays resident. Cached shadow pages are invalidated globally when the
   scene changes (per instance only for streaming and wind).
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
3. **Materials**: streamed image tiles are block-compressed only where BC is available (no ETC2 / ASTC encoders: mobile
   GPUs get uncompressed tiles), are encoded at load (natively) or on demand (web) rather than cooked offline, and are
   cut from decoded images kept in CPU memory (no tiled on-disk cache); the 512² texture arrays are uncompressed; triplanar mappings and passes other than the G-buffer resolve (Lumen,
   landscape RVT) see the 512² layers; procedural textures are not virtualised. glTF: the images of
   one layer (base colour, normal, metallic-roughness) must share their coordinates — a normal or metallic-roughness
   map with another UV set or transform than the base colour is sampled with the base colour's (with a warning); UV
   sets beyond `TEXCOORD_1`, `KHR_materials_unlit`, `KHR_materials_clearcoat` (no G-buffer room: albedo + metallic,
   normal + roughness and motion + emission + flags/AO are full; UE spends extra G-buffer targets on it),
   `KHR_materials_ior` / `_specular` (dielectric F0 is a constant 0.04; a per-pixel specular would need another
   G-buffer channel too) and transmission / volume are not imported. Alpha tests ignore wrap modes and vertex alpha.
   Lumen's cards and voxels see a material's average albedo, without vertex colours, texture transforms or AO.
   Simplification keeps second-UV and colour seams but does not weigh their deviation. Alpha tests read image
   textures with mesh UVs only (not procedural textures or triplanar mapping), with 8-bit cutoffs.
   In Lumen an alpha-masked instance is partially transparent by
   one average coverage (its partial shadows and occlusion are uniform, not shaped like its holes), only in the near
   field: rays and local-light shadows see it within 2 m of their ends, beyond which the global field lets light
   through; its cards are captured from the geometry, holes and all, with the material's average albedo.
4. **Local lights**: at most 32 lights per frame get virtual shadow maps (the rest distance-field shadows), sharing
   the sun's 1024-page physical pool (UE's default pool is four times larger), so local lights request pages half a
   mip coarser than the screen footprint. Soft shadows are PCSS-filtered, not ray-marched (no SMRT); a light's pages
   are all re-rendered when it moves.
5. **Translucency** is ordered per pixel up to 4 layers deep (depth peeling; deeper layers fall back to per-instance
   order). The layer count comes from projected bounding boxes, so boxes that overlap on screen cost up to three extra
   depth-only draws, scene-colour copies and shading passes even where the surfaces themselves don't; no
   order-independent blending for many-layer effects such as particles. Refraction reads the screen-space copy (what
   is off screen or hidden is not refracted; where a nearer layer covers the displaced lookup it falls back to the
   pixel behind); translucent surfaces don't request shadow pages (where the
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
   UE's static cache), but only the wind makes a caster dynamic: an instance whose transform changes still
   invalidates every cached page (item 1). Refreshes are decided per instance and redraw every dynamic caster in the
   page, so in the forest — where trees, bushes, ferns and grass all sway and only terrain, rocks and logs are
   static — the wind still costs most of what it did (see *Wind*); a page gets its static layer by being rendered in
   full once more at its first dynamic refresh, and only 256 pages a frame have scratch space for software-rasterised
   dynamic casters (beyond that they take the hardware path). Distance fields, the surface cache and translucent
   materials don't move.
