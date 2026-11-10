# Lumen mesh cards capture their materials (round 6 follow-up: `card-albedo`)

Design note. The README carries the user-facing text (the *Mesh cards* paragraph of *Lumen-style GI*, the
*Material graphs* bullet, gaps 2 and 3 of "Status vs. Unreal Engine 5"); this file records why it is built the way
it is, and the measurements.

## What

Lumen's mesh cards stored, per ~10 cm texel, the depth and normal of the surface they see, and the card lighting
reflected each texel's light with its material's *average* albedo — base colour × the texture's 1×1 mip, the
number the global distance field stores per voxel. Bounce light therefore had the average colour wherever a
material's colour varies: the gallery's mushroom bed (one graph material: red caps with white spots, white stems,
gills, green moss) bounced moss green under its red caps (`docs/round6/lumen-displacement.md`, "Remaining limits"),
a stained-glass window glowed in the average of its panes, a lava graph bounced its base colour.

Cards now carry the material per texel, as UE5's surface cache does:

* **Per-texel inputs** (CPU, with the geometry capture): the card rasteriser records which triangle each texel sees
  and its barycentrics (`CardConfig::sources`, `Card::sources` in `helios-geometry`); `cards.rs` turns them into
  the texel's texture coordinates (UV set 0), its UV footprint and — for a displaced variant — the offset of the
  *base* point the texel's surface point was displaced from (`SurfaceCapture::attributes`, 16 bytes per texel:
  uv as 2 × f32, offset as 3 × f16, footprint as f16).
* **Material capture** (GPU, `lumen_card_capture.wgsl`): per (surface, material) pair, the material is evaluated at
  every texel exactly as the G-buffer resolve evaluates it — `material.wgsl`'s shading models (standard image and
  procedural textures, rock, metal, foliage, glass), and for graph materials the graph's generated surface code
  (`graph_eval` + `graph_surface`, with its parameter values) — filtered for the texel's footprint, into an rgba8
  **albedo atlas**: `sqrt(albedo)` and the emission as `e / (1 + e)` (`e` = emitted radiance as a multiple of the
  albedo, the G-buffer's emissive channel).
* **Lighting** (`light_cards`): a lit texel reflects its light with the captured albedo — at coarse lighting mips
  the average of the up to four captured texels it samples — and adds the captured emission. Pairs not captured
  yet (and terrain / water materials) keep the average, so nothing changes for them.

Nothing downstream changes: gather, reflections, radiance-cache probes and Lumen Lite read the card lighting as
before, which now has the surfaces' colours.

## Design

### What is captured, and how often

* **Per (surface, material) pair.** Cards were captured per mesh (or displaced variant) and shared by every instance;
  the albedo depends on the material too. Each pair gets card records of its own (the surface's cards, pointing at
  the shared depth/normal texels and at its own albedo region); an instance's record points at its pair's. Pairs
  are cheap: 18 card records each, plus the albedo texels.
* **Per instance where the material depends on the placement**: textures projected in world space, a rock's moss
  (it grows on faces pointing up *in the world*), graphs that read the world position, the world normal or tangent
  frame, the camera or object axes (`placement_dependent`: the mushrooms read `WorldPosition` for their cells). Such
  instances get their own pair while those pairs fit 1 M texels (`PLACED_BUDGET`); further instances share their
  pair's capture, made at its first instance's placement (statistically the same texture; a rotated rock's moss is
  then on the wrong side).
* **When**: after a scene (re-)upload, within 1 M texels a frame (`CAPTURE_TEXELS_PER_FRAME`, at least one pair);
  when a graph or a material's parameter values change (`sync_materials` compares a content hash per pair: the
  material, the scene's textures, the graph code and parameter values, the placement); other material and texture
  edits re-upload Lumen's scene anyway. A re-upload keeps the albedo of unchanged pairs: their regions are copied
  from the old atlas into the new one (one `copy_texture_to_texture` per card). A displaced variant is a new
  surface (its own capture serial), so a new variant's pairs are captured fresh.
* **Budgets**: the albedo atlas holds at most 4 M texels (16 MiB, `ALBEDO_BUDGET`, the capture atlas's budget);
  pairs beyond keep the average. Texel density is the cards' own (~10 cm, coarser for very large meshes): the
  albedo has exactly the depth/normal capture's texels.

### Evaluating the material where the resolve does

The capture rebuilds the resolve's `SurfaceInput` per texel: object position (the card texel's surface point, plus
the base offset), world position, world normal and the normal matrix through the capturing instance's transform,
a tangent frame from the normal, UV and its footprint as the "screen derivatives" (and the card texel's two axes
on the surface's tangent plane as position derivatives), so texture mips and footprint-filtered noise average the
material over the texel instead of point-sampling it. The pipelines are specialised like the resolve's: one per
shading model (`MATERIAL_KIND`) and one per graph (its surface code in place of `material_graph_off.wgsl`, the
displacement function cut off — the geometry is already displaced). The graph pipelines bind the slab table (for
the material's parameter offset) and the graph parameter block; a graph the resolve does not run (device limits,
more than 16 graph pipelines) is captured with its material's own shading model, as the resolve renders it then.

**Displaced variants**: the resolve evaluates a displaced pixel's material at the *base* point it was displaced
from (positions, UVs and derivatives of the undisplaced surface), so one pattern can drive both the displacement and
the colours. The variant's refined points already carry their base position and UV; the bake keeps them
(`DisplacedVariant::base_points`, `corners`) and captures its cards with them, so a texel on a mushroom's cap is
evaluated at the base point that became the cap — red — not at the displaced position, which lies over the next
ring of the pattern.

### Alternatives rejected

* **Rendering cards through the visibility-buffer pipeline** (a card view per card through culling, raster and the
  material classify/resolve, as UE renders card captures): exact by construction, but every card would need a
  culling view, the streamed clusters resident, and the resolve's 30-odd bindings (visibility buffer, SVT feedback,
  RVT requests, G-buffer targets) adapted to atlas regions — a second renderer path for a one-off capture. The CPU
  card rasteriser already knows which triangle every texel sees; interpolating attributes there and evaluating the
  same material functions on the GPU gives the same result for a fraction of the code.
* **Evaluating the material in `light_cards`** (no atlas): every texel of every page lit per frame (64 k a frame)
  would evaluate its material, and the card lighting pass would need every graph's code — one pipeline per graph
  for a pass that lights all instances together.
* **Storing irradiance and multiplying by the albedo at sample time** (UE's final-lighting combine): sharper
  colour at coarse lighting mips, but every pass that samples cards (screen probes, reflections, the radiance-cache
  update, Lumen Lite) would bind the albedo atlas and fall back to the average for uncaptured pairs (the materials
  table and texture arrays in each), paying a texture load per card tap on every ray; lighting with the albedo
  costs it once per lit texel.
* **A CPU material evaluation**: graphs exist only as generated WGSL (a second evaluator would drift), and
  procedural textures are generated on the GPU.
* **Keeping the texel inputs on the GPU** (an attribute atlas next to the capture atlas): 16 bytes per texel of GPU
  memory for data needed only while capturing; they stay on the CPU (with the captures, which re-uploads need
  anyway) and are uploaded per capture batch.

## Measurements

RTX 3090 shared with three other agents' builds and renders (single runs of one configuration varied up to 1.6×:
valley at 67 %, 5.5–9.0 ms). Interleaved runs of a build of main (219e759) and this branch, 2560×1440, 150 frames,
**6 runs per build and configuration, minima** (GPU frame: median of the run's frames; passes: means over its last
two thirds). ms.

**Steady state** — the capture pass is not dispatched once every pair is captured; the card lighting reads up to
four albedo texels per lit texel:

| Scene | GPU frame | lumen cards (lighting) | surface + radiance cache | final gather | reflections |
|---|---|---|---|---|---|
| valley, 100 % / 67 % | 8.62 → 8.68 / 5.53 → 5.50 | 0.029 → 0.031 / 0.027 → 0.028 | 0.220 → 0.213 / 0.232 → 0.219 | 1.70 → 1.75 / 0.94 → 0.91 | 0.160 → 0.164 / 0.118 → 0.116 |
| cathedral, 100 % / 67 % | 10.78 → 10.88 / 6.62 → 6.57 | 0.067 → 0.069 / 0.065 → 0.067 | 0.349 → 0.354 / 0.350 → 0.356 | 2.27 → 2.28 / 1.04 → 1.04 | 0.462 → 0.474 / 0.256 → 0.247 |
| forest, 100 % / 67 % | 15.79 → 15.82 / 10.57 → 10.57 | 0.032 → 0.033 / 0.030 → 0.031 | 0.258 → 0.258 / 0.267 → 0.261 | 2.04 → 2.16 / 1.01 → 1.05 | 0.096 → 0.093 / 0.051 → 0.051 |
| gallery, 100 % / 67 % | 8.04 → 8.10 / 4.98 → 4.96 | 0.054 → 0.060 / 0.055 → 0.052 | 0.225 → 0.227 / 0.221 → 0.220 | 1.92 → 1.99 / 0.95 → 0.94 | 0.525 → 0.523 / 0.257 → 0.258 |

Every difference is within the run-to-run noise of this GPU (the final gather, unchanged code, moves by up to
0.13 ms between the builds' minima in both directions); the card lighting pass costs 0.001–0.006 ms more.

**Capture** (`lumen card capture`): about 0.3 ns per texel — the cathedral's 2.06 M texels take ≈ 0.6 ms of GPU
time spread over its first 3 frames after loading (measured as the pass's mean over a run that switches from the
gallery to the cathedral halfway: 0.006 ms × 100 frames, three runs alike); recapturing continuously (a test-only
switch) cost 0.016–0.042 ms per frame at 128 k texels a frame. A frame captures at most 1 M texels (≤ ~0.3 ms).

**Scene upload** (CPU, Lumen's card upload incl. the capture of every mesh; minima of 3 loads, native): gallery
6 → 9 ms, valley 83 → 116 ms, cathedral 92 → 123 ms, forest 315 → 371 ms (the texel sources and their
attributes); a re-upload for new displaced variants (no recapture of the meshes) 1 → 2 ms (gallery), 6 → 7 ms
(valley). In the browser (headless Chromium, wasm): cathedral 14.7 → 15.0 s.

**Memory**:

| Scene | (surface, material) pairs (per instance) | albedo texels | albedo atlas (GPU) | capture data on the CPU (texels + inputs) |
|---|---|---|---|---|
| gallery | 32 (17) | 0.53 M | 2.6 MiB | 1.2 MiB |
| valley | 12 (6) | 2.0 M | 9.1 MiB | 19.8 MiB |
| cathedral | 81 (39) | 2.1 M | 10.7 MiB | 19.2 MiB |
| forest | 47 (31) | 2.7 M | 12.7 MiB | 30.8 MiB |

The per-instance budget (1 M texels) is reached in the valley (6 boulder instances get their own capture, the other
~600 rock instances share their pair's).

**Images** (lit output, PSNR vs main and pixels off by more than 24/255; base-vs-base noise in brackets):

* Standard views, 2560×1440, 100 % / 67 %: valley 55.9 / 54.2 dB (noise 54.7 / 59.9), cathedral 55.9 / 55.2 dB
  (60.6 / 59.8), forest 41.3 / 41.9 dB (39.4 / 41.5), gallery 43.6 / 44.1 dB, 0.12 / 0.09 % (65.5 / 62.7 dB): the
  gallery, whose materials vary most within a material (graphs, the checkered plinths, the stained-glass sign),
  changes visibly; the cathedral, valley and forest default views change by about the noise.
* 1280×720 views: mushrooms (`--eye -5.9,0.3,10.4 --at -6.5,0.08,9.2`) 50.6 dB (noise 58.6), indirect light (view 7)
  43.9 dB: the bed's indirect light gains red where it was moss green — mean red/green of the lower screen half
  0.800 → 0.833 in view 7, 1.292 → 1.306 in the lit image (`mushrooms-indirect-difference.jpg`, difference ×8);
  the gallery row towards the lava egg (`--eye 8.6,1.1,5.4 --at 6.75,0.4,3.0`) 42.4 dB (noise 63.0), 0.19 % of pixels
  over 24/255, indirect red/green 0.923 → 0.946 (the lava egg, grey in the cards before, now glows in them);
  cathedral default view 55.8 dB (60.2), valley 59.2 dB (61.1).

**Lumen's view** (view 14, `card-albedo/*.jpg`): the mushroom bed's cards were uniformly moss green and now show
red caps, white stems and the moss (`mushrooms-cards.jpg`, from above `mushrooms-bed-top-cards.jpg`); the plinths
show their checker texture, the stained-glass sign its pattern and the lava egg its glow (`gallery-row-cards.jpg`);
the cathedral's windows their panes and the walls their stone courses (`cathedral-cards.jpg`); the valley's rocks
their moss (`valley-cards.jpg`).

**Browser** (headless Chromium 1248, WebGPU via D3D12, with the `web-bake-worker` branch merged): gallery and
cathedral load, the capture pipelines (shading models and the gallery's graphs) compile, no console, WGSL or
validation errors; once the displaced variants arrive the mushroom bed's cards match the native ones
(`browser-mushrooms-cards.jpg`, with `?bake-workers=0`). In headless Chromium the Web Worker bakes did not finish
within 60–160 s on a fresh page (on the `web-bake-worker` branch alone too), so the workers' path was checked by its
round-trip test only.

## Displaced variants and the bake offload (merged `web-bake-worker`)

The browser bakes displaced variants on Web Workers (`lumen_offload.rs`): a bake's inputs and its variant cross as
bytes. The card material inputs ride along without new inputs: `lumen_displaced::bake` takes the refinement
(`&Refined`: its points carry the base positions and UVs the material is evaluated at) — natively in process
(`BakeMesh::Refined`), offloaded by refining again in the worker from the LOD cut (`BakeMesh::Refine`, as before:
no points cross to the worker). The variant's `cards` are a `SurfaceCapture` (cards + per-texel attributes), which
`Output`'s encoding carries (and each `Card`'s `sources`, empty once converted); the variant keeps no base points,
so a recapture at another texel or scale than the bake's has no material inputs and keeps the average albedo.
`offloaded_work_is_bit_identical` checks the attributes too.

## Follow-ups: depth-aware card filtering, captured albedo in the voxels

### Depth-aware filtering of the card lighting (fix)

With per-texel albedo, a weakness of the card lookup became visible: `card_sample` checked that the hit lies on the
captured surface at the hit's own texel, then filtered bilinearly over the four nearest *lit* texels, skipping only
texels without a surface. A ground hit next to a mushroom mixed in the cap texel's light (red halos around every cap
in view 14); a floor hit at the foot of a wall mixed in the wall's. Now every lit texel stores the depth of the surface
it was lit at (`light_cards` writes `1 + stored depth / 65534` into the lighting atlas's alpha; 0 still means no
surface), and each tap is weighted by how well that depth lies on the hit's tangent plane: the depth the hit's plane
has at the tap's centre (the hit's depth plus the slope from its normal, clamped to ±4) against the tap's, within
the hit's tolerance + half the field's voxel + half a lighting texel. Taps on another surface drop out, and the
confidence counts the agreeing taps (`saturate(4 × weight)`: one agreeing nearest tap is full confidence).

* The depth costs nothing to store (the alpha was a constant 1) and one subtraction per tap to check; at rgba16f the
  stored depth has 10 bits over the card's depth range (a 3 m bed: 3 mm), far below the tolerance.
* Rejected: reading the captured depth of each tap from the capture atlas (4 more texture loads per tap at mip 0,
  and at coarse mips it is not the depth the texel was lit at); comparing normals (a cap's top and the ground both
  face up — depth is what separates them).

### Captured albedo in the global distance field and the voxel surface cache (feature)

The voxel surface cache lights the albedo the global distance field stores per voxel, which was the closest
instance's material average (base colour × texture 1×1 mip): a graph's colours never reached it. The capture pass
now also sums each pair's captured texels (8-bit albedo and emission, count; a workgroup reduction and one atomic add
per workgroup and word, `pair_sums`), every sdf instance record names its pair (`SdfInstance::pair`, written after
the card upload), and `compose` stores the pair's mean albedo where it is captured (glass and self-luminous materials
keep their sign). When the captures pending after a load or an edit are done, Lumen recomposes the field's levels
once, as after any scene change; the voxel lighting follows over its usual refresh. The near field's estimate for
hits without cards reads the same field albedo.

* One mean per instance and material, not per voxel: voxels are 0.25–2 m, so a per-voxel lookup into the cards would
  have to average many texels per voxel (or alias); the mean is exact for the instance as a whole and covers what was
  wrong — graphs and textured materials whose average is not the base colour times the 1×1 mip.
* Rejected: a per-voxel card lookup in `compose` (projecting each voxel into the closest instance's cards: three more
  bindings in a pass at the 8-storage-buffer limit, and an average over the voxel's footprint per voxel and level);
  reading the means back to the CPU (latency, and the CPU does not need them).

### Follow-up measurements

Interleaved against a build of the card-albedo state before the follow-ups (e7b4f35), 2560×1440, 6 runs per build,
minima (ms). Steady state — the field is recomposed once after captures finish, never per frame (the "lumen sdf"
pass is 0 in both builds' steady-state means):

| Scene | GPU frame 100 % / 67 % | lumen cards | final gather 100 % / 67 % | surface + radiance cache |
|---|---|---|---|---|
| valley | 8.69 → 8.71 / 5.51 → 5.46 | 0.030 → 0.030 | 1.75 → 1.79 / 0.90 → 0.87 | 0.216 → 0.224 |
| cathedral | 10.93 → 10.84 / 6.80 → 6.61 | 0.071 → 0.069 | 2.32 → 2.40 / 1.09 → 1.08 | 0.356 → 0.363 |
| forest | 16.88 → 16.19 / 10.42 → 10.64 | 0.034 → 0.034 | 2.51 → 2.30 / 0.99 → 1.03 | 0.263 → 0.265 |
| gallery | 7.96 → 8.06 / 4.97 → 4.96 | 0.053 → 0.055 | 1.88 → 2.02 / 0.95 → 0.93 | 0.221 → 0.231 |

The depth check adds a subtraction and a multiply per card tap in every card lookup (gather, reflections, radiance
cache): the final gather moves by −0.21 to +0.14 ms between the builds' minima, the same spread as between main and
the first card-albedo build, whose gather code was identical — within this GPU's noise. The sums add one workgroup
reduction to the capture (load frames only). A full recompose of the field's four levels after captures finish
costs what a scene change's does (it could not be timed separately: the shared GPU ran at 70–210 ms frames while
this was measured).

Images (1280×720; `card-albedo/followup-*.jpg`):

* Card view (14), mushrooms `--eye -5.9,0.3,10.4 --at -6.5,0.08,9.2`: 27.6 dB against the build before, 14.3 % of
  pixels over 24/255 — the red halos around the caps are gone, the ground between them is moss again, the caps are
  evenly red (`followup-mushrooms-cards.jpg`); from above 27.0 dB (`followup-bed-top-cards.jpg`).
* Voxel surface cache (view 11): the bed's voxels change from moss green to the mean of caps, stems, gills and moss —
  mean colour of the lower screen half (122.6, 126.1, 87.2) → (164.1, 143.4, 113.6) (`followup-mushrooms-voxels.jpg`).
* Indirect light (view 7): red/green of the bed region 0.800 (main) → 0.833 (cards' albedo) → 0.870 (both follow-ups)
  (`followup-mushrooms-indirect-vs-main.jpg`, difference ×8); lit image 50.6 dB against the build before.
* Elsewhere: cathedral lit 56.0 dB, view 11 51.7 dB, view 14 36.1 dB (1 % of pixels: wall/floor edges); the valley's
  card view at 300 frames 38.9 dB, 0.26 % (at 150 frames the boulders' variants arrive at different frames in the two
  runs).
* Browser (headless Chromium, `?bake-workers=0`): gallery card view and cathedral voxel view render, no console,
  WGSL or validation errors.

## Tests

* `helios-geometry`: every texel's source interpolates its triangle's corners to the point the texel sees (layer 1
  under a second slab included); nothing is recorded without the flag.
* `helios-render` (all shaders validate, including the capture with its workgroup reduction and the compose
  reading the pair sums): a tiled quad's texels carry their UVs, the base offset (decoded from f16) and the UV footprint;
  the offloaded bake round-trips the captured attributes bit-identically;
  the UV footprint follows the projection (0 edge-on); half-float rounding and saturation; a graph's surface code
  leaves out the displacement function; the capture shader validates (naga + the Tint bitwise check) without a
  graph and with every preset and every node type's test graph.

## Remaining limits

* Texel density is the cards' ~10 cm: detail finer than a texel bounces as its footprint's average (a mushroom cap
  of 8–17 cm is one or two texels). At coarse lighting mips a lit texel stands for up to 2^mip × 2^mip captured texels
  with one depth (its first surface), so the depth-aware filter separates surfaces only to that resolution.
* Not captured: vertex colours and the second UV set (UV set 0 stands in); a graph's time and camera inputs are
  frozen at capture (an animated lava bounces one moment of its glow).
* Placement-dependent materials beyond 1 M texels of per-instance pairs share the first instance's capture.
* The voxel surface cache and the global distance field see one mean captured albedo per instance and material
  (not per voxel), and no emission (as before: the voxel cache lights glass as windows and ignores emission).
* The capture inputs stay on the CPU: 16 bytes per captured texel, 20 with the depth/normal texels (1.2 MiB in the
  gallery, 19–31 MiB in the valley, cathedral and forest).
* A displaced variant recaptured at another texel or scale than its bake's keeps the average albedo (the base points
  are not kept with the variant).
* In headless Chromium the Web Worker bakes did not deliver variants within minutes on a fresh page (the
  `web-bake-worker` branch alone behaves the same); the workers' path is covered by the round-trip test, the
  render-thread path (`?bake-workers=0`) was checked in the browser.
* Terrain and water materials keep the average (heightfields have no cards anyway).
* A capture is as fresh as its scene upload: until a displaced variant arrives (seconds in the browser), the
  undisplaced bed's cards show the mushrooms' pattern evaluated on the flat bed.
