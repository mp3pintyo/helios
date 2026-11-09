# Lumen Lite — README text (round 4, agent `lumen-lite`)

## Architecture bullet (for "2. GPU pipeline", after the Lumen bullet)

* **Lumen Lite** (`lumen_lite.rs`, `lumen_lite.wgsl`, `lumen_trace.wgsl`; UE 5.8's `r.Lumen.FinalGatherMethod 0`).
  An alternative final gather for weak GPUs (`RenderSettings::gi_method = GiMethod::LumenLite`): instead of tracing
  64 rays from every 16×16 screen tile (plus adaptive probes) every frame, diffuse GI is gathered from a world-space
  **irradiance field** (DDGI-like) that changes slowly and is shared by every pixel near it. 4 camera-centred clipmap
  levels of 16³ probes (1, 2, 4, 8 m spacing: ±8 … ±64 m) scroll by whole spacings with a little hysteresis;
  probe slots are addressed toroidally (world coordinate mod 16), so a move keeps every probe still inside a window
  and only the slab that entered is new. Each probe holds an octahedral irradiance map (6×6 texels: cosine-weighted
  mean radiance + sky visibility) and an octahedral map of ray-distance moments (14×14: mean and mean square in
  probe spacings) for Chebyshev visibility, each with a one-texel octahedral border in its atlas so shading samples
  them bilinearly (rgba16f atlases of 40 MiB, plus 3 MiB of buffers, allocated on first use). **Update budget**: 512 probes a frame are
  picked on the GPU without readback — a pass scores every probe (new probes first, finest levels first; then age ×
  4 on screen × ⅛ for *sleeping* probes × 1/64 for *buried* ones) into a 64-bucket histogram, one workgroup finds
  the threshold that fills the budget and a third pass appends the winners (indirect dispatches from there).
  **Classification**: before tracing, the global distance field moves a probe that sits within 0.1 spacing of a
  surface along its gradient (≤ 0.45 spacing), disables (*buries*) it if it cannot get out — inside walls, under
  terrain — and puts it to *sleep* where no surface lies within 1.8 spacings (no pixel interpolates it; it is
  still refreshed now and then, as foliage is not in the global field). **Tracing**: 128 rays per probe (spherical
  Fibonacci, randomly rotated per refresh) through Lumen's own tracing — near-field mesh SDFs, global distance field,
  mesh-card and voxel surface caches, radiance-cache far field, sky — now one shared WGSL function
  (`trace_world_radiance`, which the screen probes call after their screen trace), so the irradiance field sees
  the same surface cache and the same multi-bounce light. The rays are projected into the probe's texels with
  hysteresis 0.85 (halved where the irradiance changed by more than half: switched lights, a moving sun; new probes
  take their first rays as they are), then copied into the atlases with their borders. **Shading**: per
  half-resolution pixel, the 8 probes around a shading point biased 0.15 spacing along the normal and 0.3 towards the
  eye, weighted by trilinear × smooth backface × Chebyshev (cubed, crushed below 0.2, as DDGI) terms, from the finest
  level that holds the pixel, blended into the next coarser level over its window's last 2 cells and into Lumen's
  radiance cache beyond the last level. The result replaces the screen-probe gather in `gi_raw` (same format), so the
  GI denoiser, sky visibility (specular occlusion), material AO and the lighting pass are unchanged. **Reflections**:
  rough ones come from the irradiance field — the lighting pass already blends specular towards the gathered
  irradiance between roughness 0.2 and 0.5 and fades the traced reflection out between 0.35 and 0.55; glossy ones keep Lumen's traced reflections (one GGX ray per half-resolution pixel), since a
  probe's 6×6 irradiance cannot show a reflected image and the radiance cache is coarser still.
  **Selection**: `--gi lumen|lite|auto` natively, `?gi=lumen|lite` in the browser (page config `gi`), `;` toggles.
  By default (`auto`) integrated and software adapters start in Lumen Lite, and `RenderSettings::gi_lite_fallback`
  switches to it when dynamic resolution has sat at its minimum percentage for 180 frames (the screen-probe gather
  is the largest GI cost that does not shrink with the resolution) — browsers do not report the adapter type, so
  there the fallback decides. Debug view 20 (`--view 20`) shows the probes as spheres shaded with their irradiance
  (sleeping: blue tint, buried: red, not yet traced: magenta) over the irradiance each pixel gets.
  **Cost** (1600×900, RTX 3090 shared with six other agents, interleaved runs, minimum of 3; see the table below):
  the final gather drops from 0.77–1.07 ms to 0.36–0.40 ms (probe update 0.28–0.35 ms, independent of the
  resolution, + per-pixel gather 0.05–0.08 ms), all of GI from 1.20–1.71 to 0.74–1.04 ms (1.5–1.8× cheaper; 2.5–3.4×
  cheaper than round 3). Images: the cathedral's lit frame differs from Lumen by 36.5 dB PSNR (valley 33.8, forest
  43.4); its indirect light is 7 % darker on average, with more contrast where the probes see occlusion (pew
  undersides: 5th percentile 0.20 vs 0.30) and less where the screen probes resolved it (pillar bases, vault ribs);
  Sponza's atrium is 9 % darker in indirect light (the far end of the gallery most), which shows in the lit frame
  (23.6 dB); no light leaks through the 1.2 m aisle walls next to the sunlit exterior. Temporal stability
  (static camera, 24 frames, % of pixels changing by > 24/255): cathedral 0.002 % for both, valley 0.042 % (Lumen
  0.053 %), forest 0.056 % (Lumen 0.060 %); the indirect light alone changes by more than 4/255 in 1.05 % of the
  pixels (Lumen 0.30 %) — a fine uniform grain from the probe refreshes rather than the screen probes' blotches.
  Moving camera (24 frames walking down the nave / strafing): within 2 % of Lumen's frame-to-frame change, no pops
  as slabs scroll in.

  **Radiance cache fix**: the radiance cache's probe refresh (`update_probes` in `lumen_light.wgsl`) ran one
  thread per probe tracing its 32 rays in turn — ~3500 long serial threads, most of the GPU idle; it now runs a
  workgroup per probe, a thread per ray, with a shared-memory reduction of the SH: the "lumen surface + radiance
  cache" pass goes from 1.42–1.49 to 0.22–0.35 ms (cathedral 1.48 → 0.35, forest 1.42 → 0.24, Sponza 1.49 → 0.22,
  valley 1.43 → 0.22; minimums of interleaved runs), images unchanged within noise (60.7 dB vs
  the baseline, which differs from itself by 61.2 dB). Both final gathers profit.

GPU time, ms, 1600×900, RTX 3090 shared with six other agents; interleaved runs of the round-3 binary, Lumen and
Lumen Lite, sum of per-pass minimums of 3–6 runs (medians were dominated by the other agents' load). "GI" = mesh cards
+ surface/radiance cache + final gather + reflections + GI temporal; Sponza from the atrium (`--eye -9,1.6,0 --at
9,3.5,0`; its screen-probe gather taken from the faster of the identical round-3 and Lumen runs).

| Scene | final gather: screen probes → Lumen Lite | GI round 3 | GI Lumen | GI Lumen Lite |
|---|---|---|---|---|
| valley | 0.77 → 0.40 (probes 0.35 + gather 0.05) | 2.61 | 1.20 | 0.81 |
| cathedral | 0.91 → 0.36 (0.28 + 0.08) | 2.72 | 1.62 | 1.04 |
| forest (medium) | 0.92 → 0.37 (0.30 + 0.07) | 2.48 | 1.29 | 0.74 |
| Sponza | 1.07 → 0.39 (0.32 + 0.07) | 2.91 | 1.71 | 0.95 |

## Status vs. Unreal Engine 5 — gaps to add (Lumen item)

Lumen Lite's irradiance field covers ±64 m around the camera (4 levels of 16³ probes; beyond, Lumen's radiance
cache with an estimated sky visibility); probes are refreshed within a fixed budget of 512 a frame (128 rays each), so
lighting changes reach distant and off-screen probes over a few hundred milliseconds and a refreshed probe changes in
small steps (fine grain in the indirect light: 1.05 % of pixels change by > 4/255 per frame on a static camera vs
0.30 % with screen probes); classification uses the global distance field only (alpha-masked foliage is not in it,
so probes in tree crowns can sleep); there is no per-probe variability-driven update rate or probe-to-probe
irradiance feedback into the surface cache (the bounce still comes from the radiance cache); glossy reflections still
trace per pixel; the adapter-based default can only see integrated GPUs natively (browsers hide the adapter type).

## Flags, hotkeys, URL options

* `;` (Semicolon) — toggle the GI final gather: Lumen (screen probes) ↔ Lumen Lite (ends the automatic choice).
* `--gi lumen|lite|auto` (native), `?gi=lumen|lite` / `heliosConfig.gi` (browser). Default `auto`.
* Debug view 20 (`--view 20`, `heliosSetView(20)`, `V` cycles to it): Lumen Lite probes.
* Library: `RenderSettings::gi_method: GiMethod`, `RenderSettings::gi_lite_fallback`, `GiMethod::for_adapter`,
  `LiteFallback`, `DebugView::from_index` (`--view` and `heliosSetView` now look views up by number, so indices may
  have gaps).
