# Round 8 — MegaLights sampling, volumes, HZB traces and light culling

Four backlog items on the local lights, in one branch (`r8-lights`) because they share
`megalights_sample.wgsl` and the light grid: `megalights-58-sampling`, `megalights-fog-translucency`,
`hzb-min-depth-traces`, `hierarchical-light-culling`. Architecture: [`lights.md`](../architecture/lights.md).

**Measurement method.** RTX 3090 shared with other agents' builds and renders; per-pass GPU minima over the last 80
of 120 frames, minimum of three interleaved runs per configuration; 1600×900, cathedral `--detail medium
--sun 200,-30` (night), default view, `--megalights` unless stated. "base" is main at 06306df built from the same
tree. Ray and list counts come from a temporary read-back (not committed).

## 1. MegaLights sampling (`megalights-58-sampling`)

**Problem.** One representative pixel per 2×2 block, always four samples and four rays per block. Where a light is
picked twice (the systematic resampling gives a dominant light several picks), each pick traced its own ray, also
where the answer was certain. Every light of a froxel's list was a candidate, however faint.

**UE reference.** UE 5.8: "no multiple rays to the same light outside penumbras" (0.3–1 ms saved on console), and an
early cull of lights by power and falloff before sampling (~20 % fewer light samples;
`r.MegaLights.LightAttenuationFalloff`; can clip distant specular).

**Design.**
* *Shared rays outside penumbrae.* The guide masks per 8×8 tile gain a third word set: besides "a ray was traced to
  this light" and "a ray reached it", "a ray was blocked" (visibility < 0.75). A light is *agreed* in a tile when
  last frame traced it and its rays either all reached it or were all blocked. In `generate`, a pick of the same
  light as the previous pick (systematic resampling visits the candidates in order, so repeats are consecutive)
  shares that pick's ray when the light is agreed; the header stores, per sample, the sample whose ray it uses (2
  bits each), and shading reads visibility through it (`ml_sample_visibility`). Penumbra tiles keep one ray per
  sample (each to its own random point of the light's disk). Sharing keeps the estimate unbiased: each sample's
  visibility is still an unbiased estimate, merely correlated with its twin's.
* *Ray compaction.* Saved rays only save time if no thread waits for them: `generate` appends the rays its 8×8
  blocks need to a list (one atomic per workgroup reserves a range), a one-thread pass turns the count into the
  trace's indirect dispatch (`ray_args_main`), and `trace` runs one thread per listed ray.
* *Storage-buffer limit.* WebGPU allows 8 storage buffers per stage; the ray list and its arguments needed two more,
  so the two guide buffers became the halves of one (`guide_masks`, the half chosen by frame parity), and the
  indirect arguments are group 1 of `generate` / `ray_args` only (an indirect buffer must not be bound writable in
  the dispatch it drives).
* *Importance cull* (`light_cull.wgsl`). First built as a relative per-froxel cull (a light fainter everywhere in
  the froxel than 1 % of the darkest point of the froxel's strongest light): mean list 29.4 → 28.5 lights at 300
  lights, for +0.11 ms of culling — froxels are large, so "darkest point of the strongest light" is far away.
  Replaced by UE's per-light form: each light's bounding sphere shrinks to where `power × exposure / d²` falls below
  `RenderSettings::mega_lights_importance_cull` (default 0.01, last frame's exposure, display white ≈ 1), one sqrt
  per light at load time. In the cathedral it culls nothing at the default: candle ranges already end where they are
  still visible (0.003, 0.01 and 0.03 give identical lists; 0.1 shortens the mean list at 1000 lights 35.5 → 22.6
  but darkens the image visibly, 39.2 dB, since dozens of individually faint lights add up). It stays as a cheap
  guard for scenes whose ranges are generous (e.g. imported lights).

**Rays traced** (share of all samples, sky blocks included):

| Lights | one per sample | shared | saved |
|---|---|---|---|
| 33 | 1.34 M (93.4 %) | 1.11 M (77.2 %) | 17 % |
| 300 | 1.23 M (85.5 %) | 1.16 M (80.7 %) | 6 % |
| 1000 | 1.23 M (85.7 %) | 1.15 M (79.7 %) | 7 % |

With many lights the four picks of a block are mostly four different lights, so there is little to share.

**Cost** (ms; `trace` includes the HZB screen traces of item 3):

| Lights | trace base | trace new | new, rays not shared | new, linear screen traces | sample base → new |
|---|---|---|---|---|---|
| 33 | 0.745 | 0.722 | 0.825 | 0.696 | 0.151 → 0.154 |
| 300 | 0.782 | 0.833 | 0.858 | 0.791 | 0.305 → 0.284 |
| 1000 | 0.789 | 0.837 | 0.869 | 0.790 | 0.309 → 0.289 |

Shared rays save 0.10 ms of tracing at 33 lights and 0.03 ms at 300–1000; the compacted list itself costs ~0.05 ms
against base's direct dispatch (ray order within a workgroup is the atomics' order), so with linear screen traces
the trace is at base's cost from 300 lights on and 0.05 ms faster at 33. Image vs base: 44.3 dB with a light circling
the candelabrum close-up (`--move-light`, [`lights/moving-light-base-vs-new.jpg`](lights/moving-light-base-vs-new.jpg)),
no new noise or lag visible; the static default view matches within the run-to-run noise.

**Not done.** UE's adaptive *sample* count (more samples in penumbra blocks) would need a variable block layout; the
sample count stays four (it also sets the light-selection noise, not only visibility). No per-light-type budget.

## 2. Volumetric fog and translucency through MegaLights (`megalights-fog-translucency`)

**Problem.** With MegaLights on, the fog injected every clustered light shadowed only by the ≤ 32 shadow-mapped
ones, and translucent surfaces looped over all of them with one distance-field trace: with hundreds of candles,
light leaked through the walls into the aisles' fog and onto the glass.

**UE reference.** UE 5.8's MegaLights volume: lit translucency and volumetric fog share one froxel volume of
MegaLights lighting (`r.MegaLights.Volume.GridPixelSize`, `GridSizeZ`); the front translucent layer may get a
per-pixel pass (`r.MegaLights.FrontLayerTranslucency`).

**Design** (`megalights_volume.rs`, `megalights_volume.wgsl`, `megalights_volume_common.wgsl`). One volume on the
fog's grid (16-pixel columns, 64 exponential slices; the fog's depth range when the scene has fog, so its froxels
match, else 0.5–80 m). Per froxel and frame: a point jittered through the froxel, the froxel's cluster list thinned
to 16 stratified candidates weighted by their unshadowed radiance (softened by the cell's size as the fog's loop
does), one light picked, one shadow ray to a random point of its disk — `megalights_rays.wgsl`, the same SMRT
shadow-map march (pages 4 mips coarser, as the fog requests them) or distance-field cones the pixel rays use (no
screen trace: a froxel is no surface) — and the estimate blended 10 % into the reprojected history. It stores:
* the in-scattered light towards the camera (Henyey-Greenstein with half the fog's anisotropy, as the loop did);
  `fog.wgsl` adds it instead of its loop (binding 140; a 1×1×1 texture while off, tested by its depth);
* the incident light as SH bands 0 and 1 with one colour: `sh0` the radiance sum, `sh1` its luminance-weighted
  direction. `translucency.wgsl` (bindings 130–132) shades a directional part (`|sh1| / lum(sh0)` of it, through
  the same GGX/Lambert BRDF, from the mean direction) and an ambient rest (a quarter of it diffusely, the mean of
  `max(n·l, 0)` over the sphere), instead of its loop, inside the volume's range.

`RenderSettings::mega_lights_volume` (default on, only with MegaLights), `--ml-no-volume`, editor toggle, **Home**.
Memory: 6 rgba16f volumes (ping-pong), 17.5 MB at 1600×900, only while on.

**Cost** (ms):

| Lights | volume | fog | translucency | frame without → with the volume |
|---|---|---|---|---|
| 33 | 0.240 | 0.115 → 0.080 | 0.278 → 0.269 | 6.53 → 6.76 |
| 300 | 0.311 | 0.188 → 0.079 | 0.317 → 0.265 | 7.56 → 7.66 |
| 1000 | 0.314 | 0.237 → 0.080 | 0.342 → 0.273 | 8.52 → 8.52 |

At 1000 lights the glass screens of the north aisle (`--eye 11,1.6,9.6 --at 2,2.0,10.4`): translucency 1.06 →
0.78 ms. Two samples per froxel cost 0.15 ms more for a static image within 60 dB of one.

Images: [`lights/fog-volume-off-vs-on.jpg`](lights/fog-volume-off-vs-on.jpg) (default view, 300 lights: the aisles'
fog no longer lit through the arcade walls by the votive racks behind them, shafts between the piers more defined;
38.9 dB between the two), [`lights/glass-volume-off-vs-on.jpg`](lights/glass-volume-off-vs-on.jpg) (1000 lights:
the stained-glass screen loses light from candles it cannot see; the votive rack's highlight on the lead is softer).

**Limits.** Froxel resolution: translucent surfaces get lighting at 16-pixel, slice-sized resolution (no contact
shadows, light from a flame next to a pane is softened by the cell size); one dominant direction per froxel, so
specular highlights of two lights on a pane merge into one; no per-pixel front-layer pass as in UE; outside the
volume's depth range translucency keeps the clustered loop; particles still use their own path; temporal lag of ~10
frames when lights move (the history is not clipped).

## 3. Closest-depth HZB and HZB screen traces (`hzb-min-depth-traces`)

**Problem.** The HZB holds the farthest depth of the early pass (occlusion culling); MegaLights' contact-shadow
screen trace marched 8 quadratic steps over 1 m and missed thin occluders between steps.

**Design.** `hzb.wgsl` gains `from_depth_closest` / `downsample_closest` (reverse-Z maximum), and `passes/hzb.rs` a
`ClosestHzb` pyramid built from the *final* depth buffer (`Hzb::record_closest`, the same level uniforms; 0.027 ms),
only while MegaLights runs. `hzb_trace.wgsl` is a reusable traversal: the segment is walked in screen space, where
reverse-Z depth is linear; at level L > 0 a cell is skipped (and the next one looked up a level coarser) if the
ray's depth over its span stays in front of the cell's closest depth, otherwise the walk goes a level finer; at the
pixel level a hit is a span that crosses the surface's shell (the surface and `thickness` behind it), filtered by
a caller-supplied `hzb_trace_accept(pixel)` (MegaLights: emissive flames don't occlude). Every pixel the ray
crosses is considered; at most `ML_HZB_STEPS` = 32 iterations. MegaLights' `screen_trace` uses it (same 1 m, same
3 cm shells growing 10 %/m), selected by the override `ML_HZB_TRACE` (`mega_lights_hzb_traces`,
`--ml-linear-traces`). Lumen's screen traces are not switched (not measured).

**Result.** +0.026 (33 lights) to +0.047 ms (1000) of tracing over the linear march. Close-up of a candelabrum
(`--eye 11.8,1.3,3.4 --at 14,0.5,5.1`, 300 lights): 52 dB between linear and HZB; where they differ, contact
shadows the march stepped over (a pew end's shadow on the pier base is filled in,
[`lights/hzb-trace-linear-vs-hzb.jpg`](lights/hzb-trace-linear-vs-hzb.jpg)). The trace could now go further than
1 m for little more cost; not done (the shadow maps and fields cover the rest).

## 4. Hierarchical light culling (`hierarchical-light-culling`)

**Problem.** One thread per froxel (3456 threads, 54 workgroups) looped over every light of the scene, transforming
each into view space: 0.25 ms at 1000 lights.

**Design** (`light_cull.wgsl`). One workgroup of 64 threads per coarse cell of 4 × 3 tiles × 4 slices (72 cells, 48
froxels each). Lights stream through shared memory 64 at a time: each thread loads one, moves it into view space
and tests its sphere against the cell's box, setting a bit of the chunk's mask; then each froxel's thread walks the
mask's bits in order and tests those lights against its own box. Lights stay in index order, so lists are the same
as brute force (including which lights the 255 cap drops; classic images 64.7 dB apart, two base runs 66.7 dB).
Two parity masks need only two barriers per chunk. Alternatives: z-binning (needs lights sorted by depth, which
changes the indices MegaLights' guide masks and the shadow-map slots use) and a light BVH (more than this grid
needs at 1024 lights).

| Lights | base | hierarchical |
|---|---|---|
| 33 | 0.011 | 0.008 |
| 300 | 0.079–0.092 | 0.030 |
| 1000 | 0.231–0.247 | 0.068 |

The lighting pass is unchanged (same lists). The cap of 255 lights per froxel and 1024 per scene remain.

## Overall (MegaLights frame, ms)

| Lights | base | new (all on) | new without the volume |
|---|---|---|---|
| 33 | 6.55 | 6.76 | 6.53 |
| 300 | 7.78 | 7.66 | 7.56 |
| 1000 | 8.69 | 8.52 | 8.52 |

Classic loop (no MegaLights): 33 lights 5.60 → 5.60, 300 7.46 → 7.38, 1000 9.10 → 8.79 (light culling only).
Browser (headless Chromium, WebGPU, 1280×720, `?scene=cathedral&lights=300&megalights=1`): all pipelines compile
(Tint), no console errors, `Home` toggles the volume ([`lights/browser-megalights.jpg`](lights/browser-megalights.jpg)).
