# Round 9 — quality levels for weak GPUs

The user-facing description is in
[`docs/architecture/browser.md`](../architecture/browser.md#quality-levels-for-weak-gpus-round-9-quality-levels) and, for
sites, in [`ittkezdjed/README.md`](../../ittkezdjed/README.md) ("Weak GPUs").

## Problem

Visitors of the sites built on Helios reported stutter. The engine had been tuned on an RTX 3090. Most visitors have
a laptop's or a desktop CPU's integrated GPU.

Measured on the iGPU of the development machine (AMD Ryzen 5 7600X: "AMD Radeon(TM) Graphics", 2 RDNA 2 compute
units, the weakest current iGPU), Chromium, 1600×900, the default settings:

| Scene | GPU per frame | Frame rate |
|---|---|---|
| valley | 637 ms | 2.5 fps |
| cathedral | 820 ms | 1.9 fps |
| forest | 814 ms | 1.7 fps |
| gallery | 599 ms | 2.2 fps |

Three things made it this bad:

1. **The adapter type never reached the renderer in the browser.**
   * wgpu's WebGPU backend reports every adapter as `DeviceType::Other` (only the fallback adapter as `Cpu`).
   * So `GiMethod::for_adapter`, which picks Lumen Lite on integrated GPUs natively, never did in the browser.
2. **Dynamic resolution did not help.**
   * It decides from windows of six frames and steps at most 0.75 per axis.
   * At 2 fps it took about half a minute to go from 100 % to 50 %.
   * The pages above still showed 100 % after 20 s.
3. **The Lumen Lite fallback was too slow to matter.** It waits for 180 frames at the minimum resolution: 90 s at
   2 fps.

## How to test on an iGPU next to a discrete GPU

Chromium binds WebGPU to the adapter of its GPU process: `powerPreference: "low-power"` still returned the NVIDIA
card, and `--use-webgpu-power-preference=force-low-power` returned no adapter at all.

What works is `--use-adapter-luid=<high>,<low>`, with the LUID that `chrome://gpu` lists for the iGPU in decimal (here
`{0,68667}`, so `--use-adapter-luid=0,68667`). WebGPU then reports `amd` / `rdna-2` and the WebGL renderer
`ANGLE (AMD, AMD Radeon(TM) Graphics (0x0000164E) Direct3D11 …)`. Timestamp queries work, so per-pass GPU times are
available. The LUID changes between boots.

## What each feature costs on the iGPU

Each feature was switched off on its own at a fixed 50 % (`?sp=50&budget=0`). The numbers are GPU minima over 4 s
(the frames are few, so the minimum is the stable statistic). The changes are relative to the scene's baseline.

| Switched off | valley (160 ms) | cathedral (208) | forest (278) | gallery (159) |
|---|---|---|---|---|
| GI entirely | −69 | −80 | −63 | −85 |
| screen probes → Lumen Lite | −67 | −52 | −15 | −30 |
| shadows | −14 | −58 | −42 | −7 |
| local light shadow maps | −2 | −46 | −13 | +1 |
| volumetric clouds | −24 | −1 | −21 | −22 |
| volumetric fog | −5 | −16 | −1 | +1 |
| AVBOIT (→ weighted blended + peeling) | −14 | −15 | +6 | 0 |
| coloured light through glass | −4 | −27 | +4 | +3 |
| local exposure | −7 | −28 | −5 | −2 |
| motion blur | −4 | −21 | 0 | −2 |
| tessellation | −4 | −22 | +7 | +3 |
| Nanite error 1 → 2 px | −5 | −17 | −26 | +4 |
| software rasteriser | −5 | +5 | **+120** | +2 |
| short-range AO | −2 | −13 | +24 | −5 |
| TSR history | −12 | −21 | −7 | −2 |

* The cathedral's baseline was noisier than the others' (its swinging lights).
* The software rasteriser stays on: without it the forest's foliage goes to the hardware rasteriser at 2.4× the cost.
* At the low level the largest cost left was the same in every scene and did not shrink with the resolution:
  `lumen surface + radiance cache`, 25–27 ms.
  * It relights 64³ × 4 voxels of Lumen's surface cache every frame.
  * It refreshes one in eight of the 24³ × 2 radiance probes every frame.

## What was built

* **`gpu_info.js`.** It reads the WebGPU adapter's `info` and the WebGL renderer string (`WEBGL_debug_renderer_info`;
  the context is released at once).
* **`quality::classify`.** It turns that into discrete / integrated / Apple / mobile / software / unknown.
  * Intel: `xe-hpg` or `xe2-hpg` architectures and Arc A- and B-series are discrete; everything else from Intel is
    integrated.
  * AMD: "Radeon Graphics", three-digit M models and Vega … Graphics are integrated; Strix Halo's 8060S / 8050S /
    8040S count as discrete for their speed.
  * Firefox's generic "…, or similar" names are trusted only for Intel and NVIDIA.
  * 22 real browser strings are in the unit test.
* **Three levels, applied on top of what the scene, the page and the editor asked for.**
  * `medium`: Lumen Lite, weighted blended OIT, no volume shadows, Nanite error 1.5 px, no local exposure or fog
    scattering, surface cache relit every 2nd frame, dynamic resolution down to 50 %.
  * `low`: on top of that no clouds, volumetric fog, glass light, short-range AO, local light shadow maps, MegaLights,
    translucency layers, tessellation, motion blur, depth of field or wind simulation; Nanite error 2 px, shadow
    pages one level coarser, surface cache relit every 4th frame, dynamic resolution down to 35 %.
  * Settings changed while a level is on (editor edits, a scene file's overrides) survive the next change of level.
    This uses `editor/settings.rs`'s key table.
* **`lumen_surface_interval` (renderer setting, 1–8).**
  * The voxel relighting runs every N-th frame.
  * The radiance probes refresh one in 8 × N per frame.
  * A `voxels_latest` index replaces the frame parity that chose the ping-pong half.
* **`quality::Governor`, which decides on seconds, not frames.**
  * One level down after 3 s over 1.25 × the budget at dynamic resolution's minimum, or over 2.5 × the budget
    anywhere.
  * One level up after 8 s under 0.55 × the budget at the maximum, never back to a level it left.
  * Without timestamp queries the frame interval decides (over 45 ms).
  * Pauses (gaps over 1 s) restart its clocks.
* **Dynamic resolution.**
  * A step may halve each axis when the GPU is over twice the budget.
  * It starts at 50 % (`low`) or 70 % (`medium`) when the level is known at start.

## Results

The iGPU with the default (`auto`) start; the level detected is `low`. GPU per frame and frame rate after the
resolution settled:

| Scene | Before | Low (first cut) | Low + surface cache interval |
|---|---|---|---|
| valley | 637 ms / 2.5 fps | 108 ms / 10 fps | **78 ms / 15 fps** |
| cathedral | 820 ms / 1.9 fps | 108 ms / 10 fps | **82 ms / 12 fps** |
| forest | 814 ms / 1.7 fps | 199 ms / 6 fps | **169 ms / 6.7 fps** |
| gallery | 599 ms / 2.2 fps | 82 ms / 12 fps | **54 ms / 20 fps** |

* The surface cache pass went from 25–27 ms to 4.6–5.9 ms.
* The `JensenHuang_Opus55` site (gallery, two spot lights, lasers) on the same iGPU: old engine 2.0 fps, new engine
  13–14 fps at 35 %.
* With the start at 50 % it reached 35 % 3 s after its first image (24 s before).

On the RTX 3090 `auto` detects `high`. Every scene runs at 60 fps with 8.8–12.3 ms of GPU (valley 9.5, cathedral 9.7,
forest 12.3 at 75 %, gallery 9.2). A forced `?quality=low` takes 5.1–8.0 ms there.

The governor, checked in the browser:

* **Disguised iGPU** (`?gpu-class=discrete`, valley): starts `high` at 680 ms. It goes to `medium` 4 s after the
  first frames and to `low` 9.6 s later, then holds at 70 ms.
* **Disguised RTX 3090** (`?gpu-class=integrated`, gallery): starts `low` at 5.1 ms. It goes to `medium` after 8.6 s
  (7.0 ms) and to `high` after another 11 s (9.3 ms).

A typical laptop iGPU (Intel Iris Xe 96 EU, Radeon 680M / 780M) has roughly 4–6 times the 2-CU part's throughput,
so these scenes should run at about 40–60 fps there at `low`. That is an estimate; only the 2-CU iGPU was measured.

## Not done

* **Foliage density.** The forest at `low` is still dominated by its foliage: software raster 23 ms, shadow raster
  17 ms. A level could thin the scattered instances; the levels don't touch scene content.
* **Firefox.** Firefox reports neither the WebGPU vendor nor a precise WebGL model, so there the starting level comes
  from the generic name or is `high`, and the governor moves it. Not measured in Firefox on the iGPU.
* **The fixed cost of TSR.** 7–11 ms at 35 % on the iGPU, because it works at the output resolution. A low level
  could cap the output (device pixel ratio) instead.
