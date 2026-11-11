# Browser platform — start-up, bake workers, device limits, f16 shaders

The browser build (`crates/helios-demo/src/web.rs`, `wasm32-unknown-unknown` + wasm-bindgen) runs the same renderer
as the native one. What is specific to the browser is how it starts (pipelines compile in the browser's GPU process,
the CPU work runs without threads) and what it asks of the device. Design notes: `docs/round6/firefox-freeze.md`,
`docs/round6/browser-startup.md`, `docs/round6/web-bake-worker.md`, `docs/round8/platform.md`.

## Start-up order

1. **Device** (`GpuContext::new`): the adapter's limits, capped (storage buffers as offered, 8 storage textures,
   24 sampled textures, 16k textures) and the optional features the renderer uses when present — timestamp
   queries, BC textures, clip distances, `shader-f16`.
2. **Renderer first**: its core pipelines are queued at once (`helios_render::pipelines`, `Group::Core`); the
   other groups (shadows, GI, clouds, fog, translucency, tessellation, occlusion culling, effects) compile in the
   background and join when ready. Chromium compiles them in parallel (warmed through `create*PipelineAsync`),
   Firefox one at a time (core set at once, then one batch per presented frame).
3. **Scene bake on Web Workers** (`scene_bake.rs`, round 8). Building a procedural scene is mostly baking its meshes
   (cluster DAG, pages, distance field: 93 % of the valley's build). The page sends one task per bake worker; every
   worker runs the scene's ordinary builder with a `Shard` mesh store (`mesh_store.rs`), which bakes only that
   worker's share and adds empty stand-ins for the rest. The share is decided without communication: every worker
   sees the same meshes in the same order and gives each to the least loaded worker so far (load = source
   triangles; meshes that read earlier meshes — a forest assembly's base and its sprays — are a `unit` and stay
   together). The meshes come back as `.hvm` bytes (the streaming format) and the page builds the scene once more
   with a `Baked` store that takes them instead of baking: the materials, instances and lights are the builder's
   own, the meshes bit-identical to a page bake (`scene_bake::tests`). The page's thread is free meanwhile; it only
   runs the builder without bakes (≈ 0.3 s for the valley) and decodes the meshes.
4. **Content pipelines during the bake**: as the bake starts the page builds a *stand-in* of the scene (every mesh
   empty, `Shard::stand_in`) and lets the renderer sync it without rendering (`Renderer::prefetch_scene`), which
   queues what the scene's materials need — texture patterns, the material resolve's shading models and
   tessellation permutation, translucent models, volumes. Before, these were found only by the first frame after
   the bake.
5. **Frames**: each frame renders into an engine texture that is copied to the canvas only if the frame created no
   pipeline; otherwise the page waits for the GPU process (`onSubmittedWorkDone`) without presenting (round 6).

Scene switches (`heliosLoadScene`, async since round 8; the embedded page's `N`) bake in the workers too while the
current scene keeps running, and resolve once the new scene is in. Imported glTF models bake on the page (their
meshes would have to cross to the workers first). Lumen's displaced variants use the same workers (round 6).

## Frames that wait for pipelines (round 8, `webperf`)

A frame's scene sync is where new content asks for pipelines: a new scene's shading models, graph materials,
texture patterns, its tessellator's dicing permutations, Substrate's slab lighting, an editor change. If the sync
queued a **core** pipeline, or one of a group whose feature this frame renders with (the groups ready at the
frame's start are recorded: `Renderer::render_frame`), the frame **stops after the sync** — at any frame, not only
before the first image — and the page keeps its last image while the queue compiles in the background (Chromium
in parallel; Firefox the core ones at once, the rest in batches). The next frame chooses its features again: the
new tessellator's group is not ready, so it renders undisplaced until the dicing arrives, as the first image does.
The held frame's scene changes are applied once more by the next one (marked structural), so shadow pages, Lumen
and translucency still see the switch. The status line shows "Compiling GPU shaders… N s" with the time held;
the engine logs `frames held T s`. Before (round 6), such a pipeline was created in the frame that needed it:
Chromium compiled those one at a time in the GPU process (wgpu's synchronous call), and the switch from the valley
to the gallery held the canvas 18–26 s for its 34 new pipelines (Firefox 33 s); now 2.7–4.1 s (Firefox 12–16 s).

**Compiling the next scene during its bake** (round 8, `webperf2`). A switch (`heliosLoadScene`) builds a stand-in
of the new scene (its meshes empty) as the bake starts, and `Renderer::prefetch_next_scene` derives from its
description alone — without syncing it into the GPU scene, material and texture state the current scene renders
from — the pipelines it will need: the material resolve's shading models, permutation (UV extensions, skinning,
tessellation) and extended slab / graph pipelines (`ExtendedMaterials::plan`, the slot assignment without its
uploads), texture patterns, Substrate's slab lighting and subsurface passes, the tessellator's dicing permutation of
its displacing graphs, Lumen's card captures of its materials, translucency and the volumes and particles. They are
queued **ahead** (`pipelines::ahead`): they compile like the rest (Chromium in parallel, Firefox in background
batches from one pipeline up, the current scene presenting between them) but count for no group's readiness, so
the current scene neither holds nor loses a feature; `pipelines::claim_ahead` hands them to the new scene when it
is in, and its first frame waits only for the ones still compiling. A pipeline the current scene asks for meanwhile
is claimed at once. Valley → gallery held the canvas 2.5–2.8 s in Chromium, now 1.3–1.5 s (Firefox 11–20 s → 9–14
s); gallery → forest 1.5–2.2 → 0.35 s (Firefox 4.9–8.0 → 1.3–1.4 s); forest → cathedral 1.2–1.8 s → none (Firefox
0.1–0.2 s). It saves at most the bake's length, and in Firefox the old scene presents one frame per compile batch
during the bake. Only what the preprocessed meshes reveal is still found by the first frame (the forest's voxel
LODs); Lumen's card captures are queued for every visible material (the gallery compiles 3 it doesn't use).
`docs/round8/webperf2.md`.

Pipelines created while recording wait too: Lumen's card capture of a new material (the captures stay pending,
the cards lit without albedo meanwhile) and Lumen's displaced-variant pipeline. Lumen Lite is created before the
hold (when asked for, or chosen by the frame as its fallback); MegaLights' slab variant only once MegaLights is
asked for; Firefox's background batch restarts at one pipeline after a hold (a batch grown to 32 on the last
scene's small pipelines held the canvas 18 s on the gallery's).

## Fewer pipelines (round 8, `webperf`)

Firefox compiles one pipeline at a time, each a WGSL → HLSL → DXC compile of 0.1–2.7 s. The valley's set went
from 214 to 200 pipelines and its background compile from 44.8 to 38.5 s of compiling, full quality 60 → 51 s in
one interleaved pair (62 → 47 s in a busier one; `docs/round8/webperf.md`):

* **The material resolve's `TESSELLATION` permutation from the start**: where tessellation is asked for and the
  scene has displacing materials, the browser uses the displaced permutation for the first image too (it renders
  undisplaced pixels as the plain one does) instead of compiling the plain one for the frames before the
  tessellator is in and then the displaced one: 5 resolve pipelines fewer on the valley, 10 on the gallery.
* **Tessellation dicing**: two grid levels per dispatch (`GRID` override: levels 0 and 1 a thread per patch, 2 and
  3 a workgroup per patch group; `tess_dice_level`) instead of a `LEVEL` override per level: 4 + 2 pipelines
  instead of 8 + 4 per displacing-graph permutation, the same GPU time (merging all four levels into one pipeline
  cost the graph-displaced terrace 0.57 ms: the thread path's registers took the shared-memory path's occupancy).
* **Translucency with AVBOIT**: only refracting surfaces (water, textured / rough / bumped glass) are drawn by the
  classic pass, so only their classic models (and peel variants) are queued; the others compile once AVBOIT is
  switched off.
* **AVBOIT's front and back buckets in one pipeline** (`webperf2`): `fs_oit` picks the bucket from the pass's layer
  range (the split depth only behind the nearest refracting layer) and the back pass binds the motion target too,
  writing it with weight 0: two shading pipelines instead of four, no new binding.
* **The underwater view** (`Group::OnDemand`): queued when the camera comes within 10 m of a water body, compiled
  last, gating no feature (a frame under water before it is in skips the underwater fog). The valley's default
  view is that close to its lake, so it still compiles there; views away from water (and scenes whose water the
  camera never approaches) skip it.

## The bake workers

`bake_worker.js` is both ends: imported by the engine's glue it is the page's pool, started as a module worker it
instantiates the page's already-compiled `WebAssembly.Module` (no second download or compile) and runs
`heliosBakeTask` — a scene shard (`scene_bake::run_task`) or a Lumen variant (`helios_render::run_bake_task`), told
apart by the task's first word. Workers share no memory (no `SharedArrayBuffer`, so no cross-origin isolation, which
GitHub Pages cannot provide); inputs and results are transferred `ArrayBuffer`s. Pool size `hardwareConcurrency − 2`,
1–4 workers; a worker idle for 20 s is retired.

Nothing a worker does can lose work: a worker that does not report its instance ready within 8 s, errors, traps, or
**stays silent for 10 s** while it runs a task is terminated and its task runs on the page. Silence, not duration:
the bakes report progress (`helios_geometry::progress::tick`, each simplified DAG group and every 1024 triangles of a
distance field), and the worker passes it on to the page as a heartbeat at most once a second, so a long bake on a
slow device (the valley's terrain takes 2.5 s on this machine's page thread) is not mistaken for a hung worker.
After three failures the pool stops and everything bakes on the page. `?bake-workers=0` turns the workers off.

## Device limits

`maxSampledTexturesPerShaderStage`: the deferred lighting pass binds 20 sampled textures (with the VSM moment atlas
and the volume shadow atlas of round 8), the translucency shading 18, AVBOIT 19, the Substrate slab lighting 21.
WebGPU guarantees 16; desktop adapters offer 48 (Chromium) or 64 (Firefox) on an RTX 3090, and the device asks for
up to 24. `GpuContext::MIN_SAMPLED_TEXTURES` (20) is checked before the device is made: below it the start fails with
a message naming the limit instead of a validation error. Between 20 and the optional variants' counts, AVBOIT
(weighted blended OIT instead) and Substrate's slab lighting are off, and their layouts are not created.
`--max-sampled-textures N` / `?max-sampled-textures=N` caps the device for testing these paths.

## Half precision (`shader-f16`)

The device requests `shader-f16` (wgpu `SHADER_F16`) when the adapter has it — Chromium and Firefox both offer it
here. A shader that includes `half.wgsl` writes the math that tolerates 11-bit mantissas in `hf`, `hf2`, `hf3`,
`hf4`; `shader::create_module` composes it with `half_f16.wgsl` (the f16 aliases, `enable f16;` moved to the top)
on a device with f16 shaders and with the f32 aliases otherwise — one variant per device, no extra pipelines.
`--no-shader-f16` / `?f16=0` force f32. No pass uses it after round 8's measurements: TSR and MegaLights' spatial
denoiser in f16 were no faster on the RTX 3090 (natively on Vulkan, in Chromium and Firefox) and moved the image by
42–52 dB PSNR; the conversion is kept as `docs/round8/platform/f16-tsr-megalights.patch` for GPUs with faster
half-precision math (packed f16 on AMD, mobile GPUs).

## Measuring in the browser

* `?compile-batch=N` fixes Firefox's background batch size, `?compile-log=1` logs each background batch's
  pipelines and compile time (with `compile-batch=1`, each pipeline's own).
* `heliosShaders()` (compile progress and the first-image / full-quality times), `heliosGpuPasses(reset)` (per-pass
  GPU minima since the last reset, JSON), `heliosBusy()`, `heliosFrames()`.
* The engine logs `first image at`, `every pipeline compiled at`, `scene bake: N meshes from W workers in T s` and
  `scene bake: built on the page in T s`.
