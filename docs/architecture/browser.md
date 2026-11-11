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
