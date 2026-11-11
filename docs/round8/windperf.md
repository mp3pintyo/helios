# Dynamic wind per instance: a pose pass and lean pipelines (round 8: `wind-bones-per-instance`)

Design note for backlog item **236** `wind-bones-per-instance`, round 8, third wave. It follows
[`dynamic-wind.md`](dynamic-wind.md) (item 217), whose user-facing description is the *Dynamic wind* paragraph of
[`docs/architecture/motion.md`](../architecture/motion.md); what is still simpler than UE is item 7 of
[`docs/status.md`](../status.md). Images: `windperf/`.

## The problem

The dynamic wind with plant bones cost about 1 ms in the forest's default view (8.45 → 9.40 ms at 1600×900 in
item 217's runs): ~0.45 ms even with `--no-wind-bones`, attributed to the bone indices in the clusters, the voxel bone
words, the instance record grown from 240 to 256 bytes and more code in the rasterisers and the resolve; and
~0.5–0.7 ms for evaluating each vertex's bone chain — its own bone's arc, then every parent's frame at its attachment
down to the root, each with its rotation in closed form — in every pass that places geometry (hardware and software
visibility rasterisers, shadow-page rasterisers, the G-buffer resolve, the voxel passes). UE evaluates a plant's
bones once per instance and skins.

## What the cost actually was

Before designing, variants of the dynamic-wind build were measured against each other (forest default view,
interleaved, per-pass minima; *x* builds are experiments, not kept):

| variant | raster sw | raster hw | resolve | page raster hw | what it shows |
|---|---|---|---|---|---|
| round 8 before the dynamic wind (4c72f94) | 2.034–2.043 | 0.221 | 0.489–0.492 | 0.347–0.365 | reference |
| dynamic wind, `--no-wind-bones` | 2.091–2.105 | 0.221 | 0.550–0.552 | 0.391–0.403 | the *off* overhead |
| x4: the same with the plant code compiled out (`plant_animated` → `false`) | 2.024 | 0.220 | 0.496 | 0.352 | = reference |
| dynamic wind, bones on | 2.227–2.250 | 0.268–0.271 | 0.637–0.645 | 0.263–0.282 | |
| x3: bones on, no posing at run time (bounds and code stay) | 2.158–2.176 | 0.222 | 0.553–0.557 | 0.135–0.152 | posing ≈ 0.3 ms |
| x1: posing without the per-vertex `plant_air` | −0.02 vs the chain build | ±0 | – | −0.03 | air ≈ 0.05 ms |

* **The *off* overhead is code, not data.** With the plant code compiled out, every pass is back at the pre-wind
  numbers although the clusters still carry their bone indices, the voxels their bone words and the instances 256
  bytes: the plant path's registers lower the occupancy of the hot rasterisers and of the resolve even where it never
  runs. So `GpuInstance` stays at 256 bytes (its spare word now holds the pose entry, below) and no attribute moves.
* **Posing costs ~0.3 ms with the bones on**, the largest part in the hardware shadow-page rasteriser (0.13 ms): it
  poses every corner of every (cluster, page) draw, 384 vertex invocations per cluster and page, while the software
  path poses each vertex once per draw. The looser fitted bounds add ~0.05–0.08 ms of software raster (more clusters
  survive culling).
* **The chain's depth was not the main cost in the default view.** Most posed vertices belong to distant plants,
  whose leaf and branch levels have faded: the chain walk skips them (two texel reads each) and evaluates one root.

## UE reference

UE 5.9's Dynamic Wind simulates (or, in `Mode 0`, evaluates) bone transforms once per plant instance on the GPU and
the Nanite rasterisers skin vertices with them: one bone index per vertex, one transform lookup. Nanite's culling
splits instances into simple and complex work and runs lean shaders for the simple ones.

## Design

### 1. The pose pass (`plant_pose.rs`, `plant_pose.wgsl`)

After the assembly expansion and the simulation, before anything draws:

* **Ranked bones.** Per bone table (`plant_wind.rs` `bone_links`), a bone that hangs from another and is not a leaf
  at the end of its chain is *ranked*: branches, whorl branches, the chain an assembly part hangs from. Leaves —
  8 640 of a broadleaf crown's 8 671 bones — and roots are not. The table gains a header texel (bones, ranked bones,
  their classes) and the list rank → bone after the bones.
* **`allocate`**, one thread per plant instance (the scene's, and the assembly part slots the expansion handed out
  this frame): an instance whose ranked bones move — the branches within their distance fade (sway), a promoted
  plant (simulation), bones that never fade — takes one *pose entry* per ranked bone from a pool of 65 536 and
  writes its first entry + 1 into `GpuInstance::plant.w` (the record's spare word); the others write 0. An
  instance beyond the branches' fade takes nothing.
* **`evaluate`**, one thread per entry: the bone's rotation (faded by class, plus the simulated deviation) and the
  frames of all its ancestors at its attachment composed into one rigid transform (quaternion + translation), this
  frame's and the previous frame's (motion vectors), and the bone's pivot, axis and length — eight `Rgba32Float`
  texels. The entries live in the plant motion texture after the simulation's 256 rows (binding **171**, formerly
  the simulation texture; the simulation now writes `Rgba32Float` and keeps the previous frame's deviation in its
  state buffer too, which the pose pass reads): the rasterisers bind nothing new.

The forest's default view hands out ~11 k entries (conifers' whorls within ~35 m, the assembly parts' chains near
them, broadleaf branches, bushes); the pass takes 0.03 ms. Overflowing the pool leaves an instance without entries:
it then poses as if its branches were at rest (they are, beyond the fade; within it the branches stop swaying) — the
pool is sized for the forest with room to spare.

### 2. One lookup per vertex (`plant_wind.wgsl`, `plant_motion.wgsl`)

A vertex reads its bone's four texels at once — pivot and length; axis and class + phase; amplitude, frequency,
attachment, parent; its carrier's entry, root, attachment on the root and a leaf flag — so nothing it needs depends
on another bone's texels, and then at most one pose entry:

* a **leaf at the end of its chain** evaluates its own flutter (closed form, as before) and rides on its parent, the
  *carrier*, at its attachment; any other bone is its own carrier;
* a **ranked carrier**: its arc with the entry's rotation, then the entry's rigid transform;
* a **root carrier** (grass blades, fern fronds, trunks: no ancestors) is evaluated in place, and so is the root of
  an instance without entries: its frame where the chain hangs from it (texel 3), every ranked bone being at rest.

The resolve fetches a shared bone once for a triangle's three corners (`plant_pose_corners`) and the previous pose
once at the pixel. The result is the chain walk's, composed in a different order: a CPU mirror of the pose pass and
the vertex path (`pose_entries_match_the_chain_walk`, `wind.rs`) checks every 5th vertex of grass, fern, broadleaf
wood and foliage, bush, conifer foliage and an assembly part's table against the chain walk with all fade
combinations, with and without entries (within 2·10⁻⁵ of the plant's size). The culling bounds are untouched.

### 3. Lean pipelines where no plant is drawn (`PLANTS`)

An override in `plant_wind.wgsl`: compiled `false`, `plant_animated` is constant and the plant code drops out.

* **Rasterisers** (visibility hardware and software, opaque and alpha-masked; shadow pages hardware and software):
  one set per scene — with the plant code when the scene has instances with bones and the bones are on
  (`GpuScene::plant_code`), without it otherwise (cathedral, `--no-wind-bones`, every scene without procedural
  plants).
* **G-buffer resolve**: per *shading model* — a `PLANTS` bit in the resolve permutation, set only for the models of
  the materials of the instances in the wind (`GpuScene::plant_kinds`). In the forest the terrain, rock and log pixels resolve
  without the plant code, the foliage's with it. Extended (Substrate / graph) slots take it where any plant has bones.
* **Browser start-up**: each scene needs exactly one set. The set the first scene is expected to need compiles in
  the background from the start, while the scene bakes (`helios_render::expect_plant_bones`, set by the demo for
  the forest and the gallery — the browser's first frames draw the forest's meshes before their final bake, without
  bones, so the scene's content alone would pick the lean set first); the other set is made on demand and queued by
  `Culling::prefetch` / `VirtualShadowMap::prefetch` when a scene needs it (`GpuScene::plant_code`). The forest
  compiles 178 pipelines (180 before), the cathedral 197 (198). The voxel rasterisers keep the plant code (they run
  only where voxel LODs are).

### 4. Plants in shadow pages take the software path more readily (`cull.wgsl`)

The hardware page rasteriser poses each corner of each (cluster, page) draw; item 217 already sent plant clusters to
the software page path up to 2.5× the usual triangle size and 96 texels of radius. With the cheaper vertex path the
balance moved: at 4× and 128 texels the page rasterisers of the fern and grass close-ups dropped by 0.24 and 0.16 ms
(hardware −0.26 / −0.21, software +0.03 / +0.05) and the default view's by 0.04 ms. The same for the main view was
measured and rejected: larger triangles cost the software visibility raster more (+0.18 to +0.48 ms) than the
hardware path saved (−0.04 to −0.14 ms).

## Alternatives considered

* **A pose entry per leaf** (one lookup for every vertex): a broadleaf crown within the leaves' fade would take 8 671
  entries (seven such crowns near the default view: 61 k); leaves evaluate their flutter themselves and ride on the
  carrier's entry, so the pool holds branches only.
* **Moving the plant data out of `GpuInstance`** (back to 240 bytes): the experiment with the code compiled out shows
  the record's size costs nothing measurable; a side table would add a lookup per vertex.
* **A per-batch raster permutation** (plant and non-plant clusters in separate culling bins, as UE splits complex
  from simple instances): it would also spare the forest's terrain and rock clusters the plant code in the
  rasterisers, but changes the culling's bins, counters and dispatch arguments shared with every other feature; the
  scene-level split takes the *off* case, the per-model split takes the resolve.
* **Per-instance air** (wind directions, gust, phases computed once per instance): ~0.05 ms at most (experiment
  x1) for a table of every animated instance's air, this frame's and the previous one.
* **A workgroup per allocation** in `evaluate` instead of an owner per entry: most allocations are assembly parts
  with one ranked bone, and the pass went from 0.030 to 0.037 ms.
* **Allocating only visible / shadow-casting instances**: the shadow views see instances behind the camera; the
  distance criterion (the branches' fade) is what the vertex path's exactness rests on, and it fits the pool.

## Measurements

RTX 3090 shared with other agents' builds and renders: per-pass **minima** of interleaved runs (default view 3 rounds,
the close-ups and the cathedral 2), forest at high detail, 1600×900, 300 frames. *pre-wind* is round 8 before the
dynamic wind (4c72f94), *before* the merged main this branch starts from (4e391e7), *after* this branch; *off* is
`--no-wind-bones`. Whole-frame minima moved by up to 1 ms between rounds of the same build (other agents' work); the
per-pass minima are the reliable numbers.

| forest default view (ms) | pre-wind | before off | after off | before on | after on |
|---|---|---|---|---|---|
| GPU frame (minimum) | 8.73 | 8.93 | **8.54** | 9.32 | **9.07** |
| raster hw | 0.223 | 0.222 | 0.223 | 0.271 | 0.249 |
| raster sw + merge | 2.077 | 2.182 | 2.092 | 2.249 | 2.263 |
| material classify + resolve | 0.496 | 0.564 | 0.479 | 0.643 | 0.584 |
| vsm raster sw / hw | 0.197 / 0.347 | 0.194 / 0.393 | 0.164 / 0.377 | 0.417 / 0.302 | 0.457 / 0.162 |
| plant pose | – | – | – | – | 0.030 |
| sum of these passes | 3.34 | 3.56 | 3.34 | 3.88 | 3.75 |

| close-ups (ms): before → after | fern (`--eye 5.2,0.7,1.2 --at 6.2,0.0,0.2`), off | fern, on | grass (`--eye 4.6,0.4,-1.6 --at 5.7,0.12,-2.7`), off | grass, on |
|---|---|---|---|---|
| raster hw | 0.084 → 0.084 | 0.103 → 0.100 | 0.242 → 0.241 | 0.307 → 0.276 |
| raster sw + merge | 0.653 → 0.618 | 0.679 → 0.692 | 2.345 → 2.266 | 2.467 → 2.433 |
| material classify + resolve | 0.497 → 0.413 | 0.559 → 0.504 | 0.583 → 0.489 | 0.670 → 0.595 |
| vsm raster sw / hw | 0.054 / 0.443 → 0.056 / 0.408 | 0.078 / 0.642 → 0.132 / 0.243 | 0.026 / 0.315 → 0.023 / 0.244 | 0.057 / 0.395 → 0.081 / 0.133 |
| plant pose | – | 0.030 | – | 0.030 |
| sum of these passes | 1.73 → 1.58 | 2.06 → 1.70 | 3.51 → 3.26 | 3.90 → 3.56 |

| cathedral (no plants), ms | pre-wind | before | after |
|---|---|---|---|
| GPU frame (minimum) | 6.41 | 6.66 | 6.39 |
| material classify + resolve | 0.465 | 0.507 | 0.397 |
| raster hw / sw, vsm raster sw / hw | 0.097 / 0.086, 0.023 / 0.013 | 0.096 / 0.086, 0.023 / 0.017 | 0.095 / 0.085, 0.022 / 0.014 |

* **Bones off and scenes without plants: back to the pre-wind cost** — the forest's passes sum to 3.34 ms as before
  the dynamic wind (they were 3.56). The cathedral's resolve is even 0.07 ms below the pre-wind build's (not analysed
  further: that build's resolve also differs in the other rounds' changes since).
* **Bones on: the dynamic wind's extra cost over the pre-wind build drops from 0.54 to 0.41 ms of these passes in the
  default view, and by 0.36 / 0.35 ms in the fern and grass close-ups** — mostly the shadow pages (posing in the
  hardware page raster, and its triangles moving to the software one: hardware −0.14 / −0.40 / −0.26 ms, software
  +0.04 / +0.05 / +0.02 ms) and the resolve (−0.06 / −0.06 / −0.08 ms: the plant code only in the plants' shading
  models, a bone fetched at once). The main view's rasterisers are within ±0.03 ms: their remaining cost is the plant
  code's registers in every cluster's pipeline, the per-vertex arc of leaves, blades and roots, and the looser bounds.
* **Pool**: 11.2 k entries in the default view, 10.9 k in the fern close-up (of 65 536).
* **Motion**: the stills and 0.5 s frame sequences of the fern and grass close-ups match the build before (images
  `windperf/fern-before-after.jpg`, `grass-before-after.jpg`: frames 1/6 s apart and the per-pixel change over
  0.5 s). Two runs of the same build differ too — the campfire's flicker and sparks, the shadow refresh budget and the
  read-backs depend on timing: the fern sequence's frames are 34.5 / 37.5 dB (min / mean) against the build before, 36.1 / 38.2 dB between two runs of that build; the grass sequence 39.3 / 40.9 dB; the same number of pixels
  move by more than 24/255 over the sequence (138 047 → 139 184 fern, 72 816 → 72 113 grass). The CPU mirror test
  matches the chain walk to 2·10⁻⁵ of the plant's size.
* **Memory**: bone tables 2.24 → 3.01 MiB (four texels a bone, headers, rank lists); the plant motion texture
  12 MiB (the simulation texture it replaces was 2 MiB); the simulation state 4 → 6 MiB (the previous deviation).
* **Browser** (headless Chromium 1248, WebGPU via D3D11, `./build-web.ps1`, `?scene=forest&detail=medium`): no WGSL or
  validation errors (only Chrome's `powerPreference` notice); the `plant pose` pass runs (0.036 ms by Chrome's coarse
  timestamps), `Numpad1` starts a gust. Pipelines: forest 178 (before 180), cathedral 197 (198); first image with the
  parallel compiler 5.2–5.9 s (before 5.4–5.6 s); with the serial compiler Firefox uses (`?shaders=serial&compile-batch=1`,
  Chromium) the forest's first image 12.5 s (11.2 s) and full set 27.6 s (25.6 s), the per-pipeline times of the
  unchanged pipelines varying as much between the runs. Playwright's Firefox build exposes no WebGPU canvas
  headlessly, so Firefox itself was not measured. The first version made the lean rasterisers lazy everywhere and
  the forest's first image waited 2 s for the plant set (its frames before the final bake have no bones yet, so the
  lean set was compiled first): `expect_plant_bones` fixed that.
* **Tests**: `pose_entries_match_the_chain_walk` (new), the dynamic wind's mirror tests, `all_shaders_validate`, the
  bitwise check; the workspace's 375 tests pass.

## Remaining limits

* Leaves, grass blades, fern fronds and the roots of instances beyond the branches' fade are still evaluated per
  vertex in closed form (one arc each, with the plant's air computed per vertex); UE skins every vertex from
  per-instance bone transforms.
* The rasterisers' plant split is per scene: in a scene with plant bones every cluster (terrain, rocks) runs the
  rasteriser with the plant code; only the resolve splits per shading model.
* The pose pool is a fixed 65 536 entries (8 MiB of texels; ~11 k used in the forest); an instance that does not fit
  poses with its branches at rest, and nothing reads its use back (no HUD statistic yet).
* The fitted (looser) culling bounds of plants with bones are unchanged: ~30 % more (cluster, page) shadow draws and
  ~0.05–0.08 ms of software raster in the forest.
