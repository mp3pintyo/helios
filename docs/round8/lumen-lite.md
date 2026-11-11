# Round 8 — Lumen Lite 5.8 and sampled surface-cache lights

Two backlog items in one branch (`r8-lumen-lite`), both on Lumen's lighting: `lumen-lite-58` (UE 5.8's Lumen
Medium / Lite and the standard-Lumen 5.8 items) and `lumen-card-light-sampling` (stochastic local lights in the
surface caches). Architecture: [`lumen.md`](../architecture/lumen.md); UE gaps: item 2 and item 4 of
[`status.md`](../status.md).

**Measurement method.** RTX 3090 shared with four other agents' builds and renders; per-pass GPU minima over the
last 80 of 120 frames, minimum of two interleaved runs per configuration (same binary, the new paths switched off by
their flags for "before"); 1600×900, default views. Image comparisons are PSNR of the tonemapped 8-bit output.

## 1. Lumen Lite 5.8 (`lumen-lite-58`)

**UE reference.** UE 5.8 adds Lumen *Medium / Lite* (beta), ~2× faster than High: an irradiance-field final gather
with probe occlusion for opaque GI, **reflections that fall back to SSR**, cheaper GI for translucency and fog, a
cheaper Lumen Scene. Standard Lumen 5.8 adds `ScreenTraces.HZBTraversal.SkipUnlitHits` (screen traces pass hits on
unlit pixels and go on), height fog on reflection hits (`r.Lumen.HeightFog 1`) and better disocclusion handling.

**Before.** Helios' Lumen Lite (round 4, `lumen_lite.rs`) had the irradiance field (4 clipmap levels of 16³ DDGI-like
probes, 512 refreshed a frame) but glossy reflections still traced one GGX ray per half-resolution pixel through the
screen, the near field (mesh SDFs) and the global distance field; the probe schedule ranked probes by age, screen and
kind only; screen-trace hits read last frame's image whatever it held there; reflections of distant geometry
ignored the fog between the reflector and what it reflects.

**Design.**
* *SSR-first reflections in Lite* (`reflect_field`, `lumen_lite.wgsl`). The same half-resolution pixels and GGX
  samples as Lumen's `reflect_trace`, so `reflect_temporal` (spatial Karis filter + clipped history) and the lighting
  pass are reused unchanged: Lite writes Lumen's `reflection_raw` and calls `Lumen::record_reflection_temporal`;
  Lumen skips its own reflection trace (`LumenOptions::traced_reflections`). The ray walks the closest-depth HZB of
  the final depth buffer (`hzb_trace.wgsl` from wave 1; 32 iterations, 30 m + the view depth, a 15 cm shell growing
  5 %/m, 2 pixels skipped at the start). Lite builds its own pyramid (`ClosestHzb` at the targets' capacity, 0.03 ms):
  MegaLights builds one only while it runs, later in the frame. A hit reads last frame's lit image (TSR history,
  reprojected), faded into the fallback over the screen's outer 8 %. The fallback is the irradiance field at the
  surface looked up in the reflected direction — its texels hold the cosine-weighted mean radiance around a direction,
  i.e. a very rough reflection — with the field's sky part (sky visibility × the cosine-blurred sky) replaced by the
  sharp sky, so open sky reflects crisply outdoors and interiors reflect their walls' light, not the sky. No world
  trace at all: the near field and the global field are not touched by reflections in Lite.
* *SkipUnlitHits* (`screen_hit_radiance` in `lumen_reflect.wgsl`). A screen-trace hit is *unlit* when last frame's
  image cannot answer it: the reprojected position leaves the screen, the history is invalid (first frame, camera
  cut) or the pixel is black there (exposed luminance < 10⁻⁴). Such a hit is passed and the trace continues — in
  Lumen's linear march (screen probes and reflections, `trace_screen`) to the next step and, past the screen trace,
  into the world trace; in Lite's HZB trace through `hzb_trace_accept`. Before, these hits returned black or, for a
  reprojection off the screen, the clamped colour of the screen's edge.
* *Height fog on reflection hits* (`reflection_height_fog`, `lumen_reflect.wgsl`; both modes). Reflections that do
  not come from the screen — Lumen's world-trace hits (near field, global field, radiance-cache far field), the sky,
  Lite's field fallback — get the volumetric fog's exponential height density integrated analytically along the
  reflected segment (the hit distance; the fog's depth for the sky; for the field the distance blends from 10 m to
  the trace distance with its sky visibility) and the light the fog scatters towards the reflecting point: the sun
  (transmittance LUT, Henyey-Greenstein phase with the fog's anisotropy) shadowed by one wide global-field cone from
  the segment's middle (two field levels coarser than the point's finest, 16 steps: fog shafts are soft; the first
  version with the surface caches' 48-step cone cost 0.058 ms in the cathedral, this one 0.025), and the radiance
  cache's ambient light above and below that point (as `fog.wgsl` injects it).
  Screen hits are left alone: last frame's image already holds the fog between the camera and the hit. Bindings
  150–152 of `lumen_gather.wgsl` (fog uniform, transmittance LUT, gather options).
* *Variability-driven probe refresh* (`priority_bucket`, `update_field`). Each refresh measures how much the new
  rays changed the probe (mean absolute luminance change of its 36 irradiance texels over their mean luminance) and
  keeps a smoothed *variability* in the top byte of the probe's status; new probes start at the maximum. The
  scheduling score (age × screen × kind) is multiplied by 0.5 for probes whose refreshes change them by ≤ 25 % (the
  refresh noise of a settled probe: 128 rays) rising to 4 for ≥ 60 %, so the fixed budget of 512 a frame goes where
  the lighting moves (swinging chandeliers, a moving sun, switched lights) and settled probes are revisited less
  often. No probe is refreshed before its weighted age reaches 24 (a settled probe on screen waits 12 frames, off
  screen 48, sleeping ones 384): a steady scene then refreshes fewer probes than the budget. Measured in the
  cathedral (temporary read-back of the probe states, not committed): a settled probe's refresh changes it by
  0.1–0.25 (mean variability 0.15 after 270 frames); the budget is used in full during the first ~200 frames and
  then 295–496 probes a frame (probe update 0.28 → 0.24 ms after 300 frames; the 120-frame benchmarks below still
  catch the start-up). Stability of the indirect light on a static camera (debug view 7, demo motion paused, 24
  frames, pixels changing by > 4/255): fixed rate 0.610 %, variability-driven 0.633 %, with the minimum age 0.611 %
  (within the run-to-run noise); convergence after the scene loads (PSNR of frames 60–140 against the last) +0.4 to
  +1.2 dB with the variability-driven schedule.
* *Reflection temporal filter.* `reflect_temporal`'s Karis weights `1 / (1 + luminance)` were taken in absolute
  radiance, where they turned the 3×3 filter into a harmonic mean: dark samples (SSR misses) swamped the hits. They
  now use the exposed luminance, as the TSR does (exposure bound at 18); Lumen's traced reflections change by 54.6 dB
  (their hits rarely miss).

Settings: `RenderSettings::lumen_lite_ssr`, `lumen_skip_unlit_hits`, `lumen_reflection_fog`,
`lumen_lite_adaptive_rate` (all default on); flags `--lite-traced-reflections`, `--lumen-no-skip-unlit`,
`--lumen-no-reflection-fog`, `--lite-fixed-rate`; editor toggles; hotkey **6** switches Lite's reflections between
screen traces and the traced path.

**Cost** (Lumen Lite, default views; "reflections" = HZB build + Lite reflections + temporal filter, before: Lumen's
traced reflections + temporal filter, without fog; ms):

| Scene | reflections before → after (without fog) | probe update | Lite GI total before → after |
|---|---|---|---|
| cathedral | 0.215 → 0.157 (0.131) | 0.246 → 0.248 | 0.964 → 0.909 |
| valley | 0.099 → 0.106 (0.098) | 0.250 → 0.216 | 0.683 → 0.654 |
| forest (medium) | 0.037 → 0.064 (0.063) | 0.272 → 0.233 | 0.720 → 0.709 |
| gallery | 0.238 → 0.130 (0.129) | 0.229 → 0.261 | 0.838 → 0.762 |

(GI total = cards + surface/radiance cache + probe update + gather + reflections + GI temporal.) The screen trace
pays where many pixels are glossy (cathedral floor, gallery puddles: −27 % and −45 %); where few are (forest), the
fixed 0.027 ms of the closest-depth pyramid outweighs the saved world traces; with MegaLights on, both build the
same pyramid (not shared yet). The first version (64 HZB iterations, the 48-step fog cone) cost 0.176 ms in the
cathedral and was no faster than tracing; 32 iterations at full resolution keep the hits (a half-resolution
variant that stopped at 2×2 cells hit the floor itself at grazing angles and was dropped). UE's "2× faster than
High" compares against screen probes; Helios' Lite already gathered from the irradiance field in round 4.

**Images.** Lit frames before → after: cathedral 45.1 dB, valley 47.8, forest 41.1 (wind), gallery 39.8 dB; the
fog alone 49.0 / 50.9 / – / 57.4 dB. The gallery's puddles
([`lumen-lite/gallery-puddles-traced-vs-ssr.jpg`](lumen-lite/gallery-puddles-traced-vs-ssr.jpg), `--eye 0,0.6,6
--at 0,0.5,0`, traced left, SSR right; 31.9 dB in the floor's crop): the screen trace reflects what is on screen
— pedestals, the candle, the stained glass — but more dimly than the world trace's surface-cache hits, rays that
leave the screen (the sphere at the left edge) fade into the field, and the backs of objects are never reflected.
[`lumen-lite/cathedral-lite-before-after.jpg`](lumen-lite/cathedral-lite-before-after.jpg): the default view.
*SkipUnlitHits* changes nothing measurable on static views (the history is valid everywhere); after a camera cut
(`R` from the north aisle, frames 100–111 against the converged view) skip on / off are within 0.1 dB — it removes
a class of artefacts (black or edge-clamped screen hits) rather than shifting the image.

**Browser** (headless Chromium, WebGPU, `?scene=cathedral&gi=lite&lights=300` and `?scene=gallery&gi=lumen`): all
202 pipelines compile in Tint, no console errors, `6` switches Lite's reflections
([`lumen-lite/browser-lite-300-lights.jpg`](lumen-lite/browser-lite-300-lights.jpg)).

**Not done.** UE Lite's cheaper GI for translucency and fog (they already read Lumen's radiance cache, not the
final gather, in Helios), a separate irradiance-field path for front-layer translucency; probe-to-probe irradiance
feedback into the surface cache (the bounce still comes from the radiance cache); classification of foliage probes
(alpha-masked foliage is still not in the global field). The "faster Lumen Scene" part of Lite is item 2 below (it
helps both modes).

## 2. Stochastic local lights in the surface caches (`lumen-card-light-sampling`)

**Problem.** The voxel surface cache (`update_voxels`) and the mesh cards (`light_cards`) looped over *every* light
of the scene per texel or voxel — not even the clustered grid: it is a camera froxel grid and the caches are world
space — and shadowed each light in range with a global-field cone. In the candle-lit cathedral at night this was the
largest light-count-dependent GI cost: surface + radiance cache 0.30 → 1.20 ms and cards 0.035 → 0.29 ms from 33
to 1000 lights (status item 4).

**UE reference.** MegaLights: a few importance-sampled lights per pixel from a candidate list, shadow rays only for
those, temporal accumulation. UE's surface cache lights cards per tile with a light list and batches of shadowed
lights; there is no 1000-light surface-cache path in UE either — this applies MegaLights' idea to Helios' caches.

**Design** (`lumen_light_grid.rs`, `lumen_light_grid.wgsl`, `lumen_light_grid_common.wgsl`,
`lumen_light_sample.wgsl`).
* *World light grid.* A camera-centred clipmap of 4 levels × 8³ cells over the global distance field's extents (4,
  8, 16, 32 m cells; 2048 cells), rebuilt every frame, one 64-thread workgroup per cell. Every light whose range
  reaches the cell's box gets a weight: luminance × falloff at its distance from the box (at least half a cell) — an
  upper bound of what it gives a point there. If at most 32 lights reach the cell they are all listed (weight 1,
  *exact*); otherwise 32 entries are drawn by systematic resampling in proportion to the weights (two passes over
  the lights in chunks of 64 with a workgroup prefix sum; each entry is assigned to exactly one light, no atomics),
  each weighted by 1 / (32 × its probability), with a fresh random offset per cell and frame. The list is then an
  unbiased estimate of the cell's full light set, whatever the light count. 66 words per cell (540 KiB).
* *Sampling per texel / voxel.* The point's cell (finest level that holds it; beyond the last level the exhaustive
  loop) gives ≤ 32 candidates, whose unshadowed irradiance (falloff, spot cone, n·l, entry weight) is evaluated; two
  are picked in proportion to it by systematic resampling and only those are shadowed (repeated picks of a light
  share its cone). The estimate — the candidates' total times the picks' visibility, colour from each pick — is
  unbiased; it is noisy only where visibility differs between candidates (penumbrae, lights behind walls). Where
  at most two candidates contribute, each is shadowed and the result is exact. A first version kept the candidates'
  contributions in a 32-element private array; it is re-evaluated in a second pass instead (cheaper for shader
  compilers than dynamically indexed private arrays).
* *Accumulation.* Voxels already blend each refresh half into their previous value (ping-pong). Card texels are
  relit only when their page is scheduled and their lighting atlas is write-only in the pass (rgba16f has no
  read-write storage in WebGPU), so their sampled local light is kept separately: one rgb9e5 word per atlas texel
  (4 M words, 16 MiB, behind the grid in the same storage buffer — `light_cards` was at 7 of WebGPU's 8 storage
  buffers per stage), blended half per refresh unless the estimate was exact or the page was just (re)allocated
  (`FRESH_PAGE`, bit 31 of the update record's third word, set by `cards.rs::assign`).
* Bindings 153 (grid uniform) and 154 (grid + card words) in `lumen_light.wgsl` and `lumen_card_light.wgsl` (not
  shared shaders). `RenderSettings::lumen_light_sampling` (default on), `--lumen-exhaustive-lights`, editor toggle.

**Alternatives.** The camera froxel grid MegaLights uses does not cover off-screen voxels and cards. Capped exact
lists (UE-style per-cell light lists) would drop lights at 1000 candles per 16 m cell, i.e. bias. Accumulating a
visibility *ratio* instead of radiance would follow intensity changes instantly but not the candidate resampling's
noise. A ping-pong card atlas would cost another 32 MiB.

**Cost** (cathedral at night, `--detail medium --sun 200,-30 --lights N`, classic light loop; ms):

| Lights | light grid | cards: all lights → sampled | surface + radiance cache: all → sampled | frame (min) |
|---|---|---|---|---|
| 33 | 0.008 | 0.035 → 0.037 | 0.300 → 0.281 | 5.68 → 5.73 |
| 300 | 0.018 | 0.123 → 0.063 | 0.717 → 0.364 | 7.83 → 7.42 |
| 1000 | 0.045 | 0.290 → 0.063 | 1.201 → 0.371 | 9.21 → 8.21 |

With MegaLights (`--megalights`) the caches cost the same (they do not use MegaLights): 300 lights cards
0.124 → 0.066, surface + radiance cache 0.719 → 0.364, frame 8.01 → 7.66 ms; 1000 lights 0.292 → 0.063, 1.204 →
0.371, frame 8.88 → 7.75 ms.

The caches' cost is now flat from 300 lights on (the candidate count is capped at 32 per cell); the grid's build grows
linearly with the light count but stays small (2048 workgroups × 16 chunks at 1000 lights).

**Images.** 1000 lights, 120 frames: the lit frame is 49.5 dB from the exhaustive path's, mean brightness
+0.4 % ([`lumen-lite/lights-1000-exhaustive-vs-sampled.jpg`](lumen-lite/lights-1000-exhaustive-vs-sampled.jpg)).
The caches themselves after 300 frames: card lighting (debug view 14) 36.8 dB, voxel lighting (view 11) 42.4 dB;
per-texel noise shows there, and through the tone curve noisy texels read ~5 % darker on average, while the gather,
which averages many texels, matches. Static-camera stability with 1000 lights (16 frames, % of pixels changing by
more than 4 / 24 levels): 26.46 / 0.0143 % exhaustive, 26.37 / 0.0148 % sampled (the demo motion, dust and fog
noise dominate; sampling adds nothing visible). How fast a moving light's bounce follows on the cards (two of their
page's refreshes) was not measured separately.

**Limits.** Two shadow cones per point and refresh: lights that are individually faint but many behind a wall
converge over a few refreshes, and card texels follow a changed light over two of their page's refreshes (cards are
refreshed round robin within 1024 pages a frame); voxels over ~3 refreshes (every 4th frame). Points beyond the grid
(> 128 m from the camera) keep the exhaustive loop. The grid's 32 candidates per cell are resampled every frame,
so a cell with hundreds of lights adds selection noise on top of visibility noise.
