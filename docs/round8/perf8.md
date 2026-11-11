# Round-8 close: performance pass (`round8-perf-pass`, order 239)

Round 8 added a dozen features to every scene (short-range AO, the Lumen world cache, AVBOIT, local lights through
glass, simulated rivers on a mesh terrain, plant bones, the MegaLights froxel volume, …). On an idle GPU at 2560×1440
the default views got 0.3–1.7 ms slower than round 7. This pass measures where each millisecond went, wins back
what can be won without a visible change, and records the rest per feature.

## Method

RTX 3090, **idle GPU** (no other agent running), 2560×1440, `--frames 240`, the summary line's *frames 80..240 mean:
GPU* (the per-pass numbers are that line's means too), minimum of interleaved runs: round 7 (06306df), round 8 before
this pass (main 217be44) and after it, run in turn, so slow drift hits all three. Seven views: the default views of
the valley, cathedral, forest and gallery, the cathedral with 300 candle lights (`--lights 300`) without and with
MegaLights, and the lantern (`--lantern`). Per-pass diffs come from the same runs. Where a pass bundles several
dispatches (the material resolve runs one per shading model), a scratch build split them into passes of their own
(round 7 and round 8 both), and single costs were measured by scratch builds that removed one call each (river caustics,
SSR, the glass-light lookup, …). Those builds are not committed. Images: the default views and the
foliage close-ups before and after (screenshots of the last run), PSNR against round 8 before, next to the PSNR of two
runs of the same build (the campfire's flicker, sparks and wind make no two runs identical).

## Where round 8's time went (round 7 → round 8 before, 100 %)

| Pass (ms) | valley | cathedral | forest | gallery | 300 lights | MegaLights 300 | lantern |
|---|---|---|---|---|---|---|---|
| GPU frame | 8.79 → 10.49 | 10.88 → 12.05 | 16.15 → 16.40 | 8.10 → 9.15 | 14.83 → 16.10 | 15.19 → 16.33 | 11.61 → 12.78 |
| lighting | 0.92 → 1.19 | 2.24 → 2.96 | 1.49 → 1.88 | 1.07 → 1.70 | 4.33 → 5.50 | 1.22 → 1.74 | 2.29 → 3.05 |
| lumen world cache (new) | 0.27 | 0.48 | 0.34 | 0.31 | 0.62 | 0.63 | 0.47 |
| lumen final gather | −0.12 | −0.37 | −0.16 | +0.19 | −0.38 | −0.38 | −0.37 |
| lumen cards + surface/radiance cache | — | −0.07 | — | −0.05 | −0.60 | −0.59 | −0.07 |
| material classify + resolve | 0.84 → 1.12 | 0.73 → 0.88 | 0.85 → 1.25 | — | 0.73 → 0.90 | 0.74 → 0.89 | 0.73 → 0.89 |
| translucency (+ scene colour) | 0.76 → 1.54 | — | — | — | — | 0.67 → 0.60 | — |
| oit composite → avboit composites | 0.08 → 0.18 | 0.08 → 0.17 | 0.08 → 0.11 | — | 0.08 → 0.17 | 0.08 → 0.17 | 0.08 → 0.17 |
| glass light (new) | — | 0.07 | — | — | 0.10 | 0.11 | 0.07 |
| megalights volume (new) / volumetric fog | — | — | — | — | — | 0.69 / −0.23 | — |
| vsm raster sw / hw | — | — | 0.86 / 2.01 → 1.49 / 0.51 | — | — | — | — |
| raster sw + merge / hw | — | — | 3.02 / 0.54 → 3.33 / 0.63 | — | — | — | — |

(— : within ±0.04 ms.) Per feature:

* **Short-range AO** (foliage grounding): full resolution, inline in the lighting pass, 16 depth taps a pixel. The
  scratch build without it: lighting −0.30 (valley) … −0.61 ms (gallery) — more than the taps' own cost: inline,
  its loop kept the lighting's registers busy across the whole shading.
* **Local lights through glass**: the per-light layer lookup and gathers in the light loop: 0.12 ms of lighting in the
  cathedral, 0.27 ms with 300 lights, plus the maps' pass (0.07–0.11 ms).
* **Lumen world cache**: 0.27–0.63 ms, paid back by the final gather's −0.12…−0.38 ms and the cards' lighting (gone
  in *replace* mode). One path per 8×8 tile: 2.5 times as many paths at 1440p as at 1600×900, where it was tuned.
* **AVBOIT** replaced the weighted-blended composite: two full-screen draws a bucket (modulate, add) instead of one,
  +0.03–0.10 ms in every scene, also where nothing is composited in most pixels.
* **Material resolve**, per shading model (split build, 100 %): cathedral *standard* 0.63 → 0.79 ms (the plant-data
  decode and the plant code's registers in the shared resolve — removing the plant-data fetch alone made it *slower*,
  0.91 ms: the compiler's register allocation, not the work, decides here); forest *foliage* 0.36 → 0.63, *standard*
  (wood) 0.16 → 0.28 ms (plant bones: with `plant_animated` compiled false they drop to 0.39 / 0.18 ms); valley
  *terrain* 0.44 → 0.74 ms (the mesh terrain's per-pixel triplanar layering — it has no runtime virtual texture — and
  the river's shore terms).
* **Valley river**: translucency 0.67 → 1.36 ms, scene colour 0.09 → 0.17 ms (the river and the lake overlap on
  screen: one peel more). Scratch builds: the river surface's own shading 0.61 ms (shading replaced by a pass-through),
  of which screen-space reflections 0.28 ms (all water), caustics on the river bed 0.09 ms, the distance-field
  reflection trace 0.045 ms, the second flow-map phase 0.01 ms, foam 0.01 ms; the rest is the per-pixel water model
  the lake shares (waves, sun shadow, radiance cache, refraction).
* **Forest geometry**: the plant bones' looser bounds and the plant code in the rasterisers (raster sw + merge +0.31,
  hw +0.10 ms); the shadow pages moved to the software path (vsm raster sw +0.63, hw −1.50 ms: −0.87 ms net).
* **MegaLights volume** (300 lights with MegaLights): 0.69 ms for a froxel grid of 16-pixel columns — 160×90×64 at
  1440p, 2.5× the 1600×900 grid — of which the fog's own light loop gives back 0.23 ms.

## What this pass changed

1. **Short-range AO at half resolution** (`short_range_ao.wgsl`, `passes/shading.rs`, `targets.rs`, `lighting.wgsl`).
   A compute pass `short-range ao` before the lighting evaluates the AO once per 2×2 pixels — at the even pixel the
   GI's half-resolution grid uses, as UE's `r.Lumen.ScreenProbeGather.ShortRangeAO.DownsampleFactor` 2 — into
   `FrameTargets::short_range_ao` (R32Float, half resolution). The lighting upsamples it inside the GI's bilateral loop
   (`sample_half`: one set of depth and normal weights, the GI's four depth and normal loads shared; binding **153**),
   so each pixel blends up to four differently rotated estimates, which let the steps per direction drop from 4 to 3
   with the same look; TSR still averages the rotation over frames. With GI off the same loop runs for the AO alone.
   Lighting now binds 21 sampled textures (`MIN_SAMPLED_TEXTURES` 21, Substrate 22; adapters offer 48–64).
   Cost: the pass 0.06–0.10 ms at 1440p (0.095 in the cathedral), the lighting −0.28…−0.71 ms.
2. **World cache: paths per frame capped** (`world_cache.rs path_tile`, `lumen_world_cache.wgsl wc_path_tile`). One
   path per tile of 8×8 pixels or larger, the smallest size that keeps the paths within 24 000 a frame: 8×8 up to
   1600×900 (unchanged there, where the accuracy against the reference was measured), 9×9 at 1715×965 (67 % of
   1440p), 13×13 at 2560×1440. The cells — 20 cm within 8 m, growing with the distance — need the samples, not the
   pixels: a 1440p screen sees the same cells as a 900p one. −0.07…−0.14 ms.
3. **AVBOIT composites skip empty pixels** (`oit.wgsl`): the modulating draw discards where the transmittance is 1,
   the back bucket's light draw where it adds nothing. Exact (multiplying by 1 and adding 0 were no-ops); saves the
   blend's read and write of the HDR pixel over most of the screen: −0.03…−0.04 ms where AVBOIT composites.
4. **Glass-light lookup reads colours only behind a pane** (`glass_light.wgsl`): the distance gather first; the three
   colour gathers only where a texel's pane lies between the light and the point. Exact; −0.045 ms in the cathedral.

## Results

GPU frame (ms), 2560×1440, 100 % / 67 %, minimum of 3 interleaved runs of *before* and *after* (round 7: the figures
measured on the same idle GPU at the start of the close; the 67 % ones of the last three views in a run of their own):

| view | round 7 | round 8 before | round 8 after | after − before |
|---|---|---|---|---|
| valley | 8.79 / 5.54 | 10.69 / 6.63 | **10.42 / 6.46** | −0.27 / −0.17 |
| cathedral | 10.87 / 6.68 | 12.05 / 7.38 | **11.48 / 7.27** | −0.57 / −0.11 |
| forest | 16.35 / 10.69 | 16.71 / 11.49 | **16.29 / 11.34** | −0.42 / −0.15 |
| gallery | 8.18 / 5.05 | 9.06 / 5.53 | **8.66 / 5.29** | −0.40 / −0.24 |
| cathedral, 300 lights | 15.06 / 8.94 | 16.59 / 9.66 | **15.69 / 9.21** | −0.90 / −0.45 |
| cathedral, 300 lights, MegaLights | 15.33 / 9.04 | 16.34 / 9.50 | **15.88 / 9.34** | −0.46 / −0.16 |
| lantern | 11.62 / 7.25 | 12.92 / 7.97 | **12.27 / 7.76** | −0.65 / −0.21 |

(The *before* column runs up to 0.5 ms above the first measurement of the same build at the start of the close — the
frame totals moved with the GPU's clocks over the two hours of runs; the interleaving keeps *before* and *after*
comparable.) The forest's −0.42 includes −0.26 ms of the clouds pass, which varies by that much between runs
of one build; without it the forest gains −0.16 ms.

Per pass at 100 % (ms, before → after):

| pass | valley | cathedral | forest | gallery | 300 lights | MegaLights 300 | lantern |
|---|---|---|---|---|---|---|---|
| lighting | 1.20 → 0.92 | 2.98 → 2.46 | 1.93 → 1.58 | 1.71 → 1.12 | 5.70 → 4.99 | 1.75 → 1.30 | 3.09 → 2.56 |
| short-range ao (new pass) | 0.06 | 0.10 | 0.07 | 0.07 | 0.10 | 0.10 | 0.10 |
| lumen world cache | 0.27 → 0.20 | 0.48 → 0.34 | 0.35 → 0.28 | 0.31 → 0.23 | 0.65 → 0.50 | 0.64 → 0.50 | 0.47 → 0.35 |
| avboit composites (front + behind) | 0.18 → 0.14 | 0.17 → 0.14 | 0.11 → 0.10 | — | 0.17 → 0.14 | 0.18 → 0.14 | 0.17 → 0.14 |

At 67 % the short-range AO pass costs 0.03–0.05 ms and the lighting gains 0.13–0.35 ms; the world cache changes
little there (9×9 tiles, 20.6 k instead of 26 k paths). The glass-light early-out is −0.045 ms of lighting in the
cathedral (scratch runs; within the noise with 300 lights, where the per-light layer lookup, not the gathers, costs
the 0.27 ms).

**Images** (last run of each build, PSNR of after against before, next to two runs of *before*): valley 52.5 dB
(52.98 same build), gallery 53.5 (65.5), forest 39.2 (38.7), cathedral 39.5 (39.5 — the swinging chandeliers), lantern
41.4 (56.0), 300 lights 37.1 (54.8), MegaLights 300 36.1 (52.4). In the three candle-lit views the change is a
uniform ±1 % of brightness over the floor and the fog (mean of the lower third 75.8 → 76.8 / 255 with 300 lights),
from the world cache's different path set (same expectation, other samples: a build with the old path count and the new AO lands at
77.6) and the half-resolution AO; side by side the crops are indistinguishable, far inside the world cache's 12–15 %
distance from the reference indoors. The foliage close-ups (fern and grass, `ao-*-off-full-half.jpg`: AO off / the
round-8 full-resolution AO / this pass) show the same darkening under the fern's crown and the tufts, 37.4 / 47.2 dB
against the full-resolution AO (the AO itself moves them by 34.2 / 45.6 dB).

**Tests**: `cargo test --workspace --release` green (new: `path_count_does_not_grow_with_the_resolution`, the
budget constant checked against the shader). **Browser** (headless Chromium, WebGPU/D3D11): forest, cathedral and
valley start and render, no WGSL or validation errors.

## Rejected / measured and left

* **Adaptive SSR step count** for water (steps from the reflected ray's screen length, 12–40): +0.02 ms — the
  dynamic loop bound costs the unrolled loop more than the skipped steps save.
* **Caustics off on rivers**: −0.09 ms and invisible in the default view (60.8 dB), but they are part of the river's
  look in closer, sunnier views; kept.
* **Skipping the plant-data decode in non-plant shading models**: slower (cathedral standard 0.79 → 0.91 ms): the
  shader compiler's register allocation, not the decode, sets the resolve's cost there.
* **4 steps per direction at half resolution**: 0.113 ms; 3 steps 0.094 ms with no visible difference (the upsample
  blends four texels).
* Not attempted: a per-batch raster permutation for plant clusters (windperf.md's rejected alternative), the
  MegaLights volume's froxel size at high resolutions (it shares the fog's grid), a virtual texture for the mesh
  terrain (its cliffs need the triplanar layering), the main view's SW-raster threshold for plant clusters at 1440p
  (windperf measured raising it as a loss at 900p; at 1440p triangles are larger in pixels, so the hardware path
  already takes more of them).

## Remaining limits

* Short-range AO at half resolution: a pixel whose four texels all lie on other surfaces (a thin blade's edge) takes
  their AO by the GI's fallback weights; no history of its own (TSR averages it).
* The world cache's per-frame sample count is fixed: at 1440p each cell gets as many samples as at 900p, not more.
* The river surface's shading (0.6 ms at 1440p in the valley's default view), the cathedral's standard resolve
  (+0.16 ms), the plant bones in the forest (+0.35 ms of resolve, +0.4 ms of rasterisers) and the MegaLights volume
  (0.69 ms at 1440p) remain as the features' cost.
