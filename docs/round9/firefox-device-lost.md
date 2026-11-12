# Round 9 — Firefox lost the device after ≈ 800 frames

The user-facing description is in
[`docs/architecture/browser.md`](../architecture/browser.md#gpu-memory-round-8-gpu-memory-budget) ("Firefox on Windows").

## Problem

In Firefox the showcase's live view froze about 10 s after it reached full quality. Round 8
(`docs/round8/gpumem.md`) had put this down to a Firefox leak of presented canvas surfaces that Helios could only
report. Measured again (headed Firefox 155, valley, 1600×900, frames counted from the page): the device was lost at
780, 803, 805 and 806 presented frames in four runs — a frame count, not a time or a memory level. On a 160 Hz display
that is 5 s.

## What it was

Two things, found by wrapping the page's `GPUDevice.create*` calls:

1. **Bind groups.** 118 114 `createBindGroup` calls in 805 frames: ≈ 149 a frame (cull 14, cull args 14, Lumen 14,
   VSM expand 13, HZB 9, translucency downsample 8, bloom 11, local exposure 5, …), nearly all over the same resources
   as the frame before. Firefox's D3D12 backend allocates a bind group's descriptors in a shader-visible heap of 1 M
   descriptors and frees them when the group is dropped — which in the browser is when the garbage collector
   finalizes the JS object, and the page makes too little JS garbage for that to happen in time. ≈ 150 groups of
   ≈ 8 descriptors a frame fill the heap in ≈ 800 frames; wgpu reports "Not enough memory left" and the device is
   lost. A run with extra JS allocations in the page (the counting hook's strings) lasted 1112 frames.
2. **The canvas format.** With the bind groups fixed, Firefox's GPU process still grew ≈ 5 MiB a frame
   (20.7 → 45 GB in 75 s, the counter `GPU Process Memory\Dedicated Usage`) with the browser's preferred
   `bgra8unorm` canvas, and stayed flat with `rgba8unorm`. A minimal page (clear, copy to the canvas, present) grows
   12.5 MiB a frame with `bgra8unorm` — also with `premultiplied` alpha, rendering straight into the canvas, or
   `unconfigure` / `configure` every 30 frames (every 5: 7.4 MiB) — and not at all with `rgba8unorm`.

## Changes

* `util::bind_group` keeps the groups it makes in a cache keyed on the layout and every entry's resource (wgpu 30's
  handles are `Eq + Hash` on their identity; buffer offset and size are part of the key). A group not asked for in
  120 frames is dropped (checked every 30 frames, `Renderer::render` ends the cache's frame). The cache holds clones of
  the resources, so an identity can't be reused while a key refers to it; a replaced resource simply stops matching.
  Retired resources are still destroyed on time (`gpu_memory::destroy_retired`), and a group that refers to one is
  never asked for again. Thread-local, so native tests on parallel threads don't share it.
* The page keeps the frame target's view (`frame_view`) instead of creating one a frame, which also made the
  tone mapper's and TSR's groups new every frame.
* `canvas_format` (web.rs): `rgba8unorm` in Firefox, the preferred format elsewhere; `?canvas-format=rgba|bgra`.

## Measurements

* Firefox, valley, 1600×900: 0.0 bind groups a frame in the steady state (was 148.8); 6640 frames in 110 s without a
  lost device (was ≈ 800), the GPU process flat at +0.9 GB over the start.
* Chromium 1248 and Edge (headed, vsync on a 160 Hz display, two rounds): `rgba8unorm` 158.3–160.5 fps, HUD GPU
  12.9–13.6 ms; `bgra8unorm` 158.3–160.6 fps, 13.0–13.6 ms. No difference, so they keep `bgra8unorm`.
* Native (Vulkan, 1600×900, 200 frames, base vs new): valley 60.4 dB, cathedral 57.3 dB, gallery 65.2 dB, forest
  39.1 dB (its run-to-run noise); no validation errors. The switch chain valley → gallery → forest → cathedral →
  valley holds exactly the same GPU memory before each switch (1314.1 / 1031.2 / 1385.8 / 1401.9 / 1322.2 MiB).

## Remaining limits

A group whose resources change every frame (none in the steady state now) still makes a new group a frame. The cache
keeps a dropped resource alive up to 120 frames natively.
