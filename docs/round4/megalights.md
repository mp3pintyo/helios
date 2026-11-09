# MegaLights — README text (round 4, agent `megalights`)

## Architecture bullet (section 2, after *Local light virtual shadow maps*)

* **MegaLights** (`megalights.rs`, `megalights_sample.wgsl`, `megalights_shade.wgsl`, `megalights_denoise.wgsl`,
  `megalights_common.wgsl`, `megalights_surface.wgsl`; UE 5.5's MegaLights). `RenderSettings::mega_lights` (`=`)
  replaces the clustered light loop — which can shadow only the 32 shadow-mapped lights plus four distance-field
  traced ones per pixel and lets every other light leak through walls — by a fixed budget per pixel in which *every*
  local light casts soft shadows. **Sampling**: per 2×2 block, at a representative pixel that moves every frame, four
  lights are drawn from the froxel's light list in proportion to their unshadowed contribution through the shared BRDF
  (`brdf.wgsl`, the same functions the classic loop now calls), by systematic resampling over at most 32 candidates;
  longer lists are thinned to a stratified subset in which every light has the same chance, its weight scaled by the
  inverse, so the estimate stays unbiased and the cost per pixel constant. A small floor keeps lights behind the
  representative's normal possible for the rest of its block. **Visibility guiding**: per 8×8 tile, bit masks (light
  index mod 256) of the lights some ray was traced to and of those some ray reached; a light whose rays were all
  blocked in last frame's (reprojected) tile keeps a tenth of its weight. **Shadow rays**: one per sample, to a random
  point of the light's disk (the source radius makes the penumbra): a 1 m screen trace against the depth buffer
  (contact shadows), then over the whole ray the light's virtual shadow map where it has one — shadow-map ray marching
  (UE's SMRT, 8 steps; the first is the receiver's classic test, the later ones see occluders off to the side as the
  ray converges on its point of the light, within an assumed thickness) — otherwise Lumen's distance fields (mesh SDFs
  within 2 m of either end, the global field between, narrow cones). The screen trace treats the depth buffer as 3 cm
  shells (a thicker assumption shadowed walls behind any object in front of them) and ignores emissive pixels (candle
  flames are in the depth buffer but cast no shadow). **Shading**: every pixel evaluates the BRDF of its block's four
  samples with their unbiased contribution weights and visibilities and writes its direct light *demodulated* (diffuse
  without the diffuse colour, specular divided by the environment BRDF), so the denoiser never blurs texture detail.
  **Denoising** (SVGF): temporal accumulation with reprojection, depth-checked disocclusion, a 2.5σ variance clip and
  up to 24 (diffuse) / 12 (specular) frames, with luminance moments; then four à-trous iterations (3×3 binomial,
  strides 1–8) with depth-plane, normal and luminance edge-stopping scaled by the variance of the accumulated mean (so
  converged pixels keep normal-mapped lighting detail); the last one modulates the result back into one texture, which
  the lighting pass adds instead of running its loop. The froxel lists now hold 255 lights (were 63). Debug views
  **18** (each sample's light, one hue per light, dark where blocked) and **19** (this frame's samples | denoised).
  Cost (cathedral at night, 1600×900, RTX 3090 shared with other jobs, minimum of 4 interleaved runs): the MegaLights
  passes take 1.37 ms at 33 lights (sample 0.16, trace 0.40, shade 0.08, temporal 0.14, spatial 0.59) and 1.95–1.99 ms
  from 100 to 1000 lights (only the share of distance-field rays grows), and the lighting pass drops from 0.88–2.50 ms
  to 0.46 ms; direct lighting in all (light culling + shadow mask + lighting + MegaLights) classic → MegaLights: 33
  lights 1.03 → 1.84 ms, 100: 2.45 → 2.52, 300: 2.78 → 2.54, 1000: 3.53 → 2.74 ms (aisle view 3.63 → 2.61) — while the
  classic path shadows only 36 of those lights per pixel. Image vs the classic path at 33 lights: 39.7 dB PSNR (nave),
  39.8 dB (candelabrum close-up); the differences are shadows the classic PCSS blurs away (a candelabrum's side candle
  on the pier behind it). Static camera, frame-to-frame pixels changing by more than 24/255: nave 0.0093 → 0.0091 %,
  close-up 0.0017 → 0.0009 %, 1000-light aisle 0.0005 → 0.0007 % (classic → MegaLights). Browser (headless Chromium,
  1280×720, 300 lights): 1.2 ms. Memory (only while on): 56 bytes per internal pixel (two rgba16f raw, two sets of
  rgba16f diffuse + specular + r32uint history) plus 36 bytes per 2×2 block of samples.

## Status vs. Unreal Engine 5 — new gaps (item 4, *Local lights*)

* **MegaLights** traces no hardware rays (WebGPU has none): beyond the 1 m screen trace, rays use the light's virtual
  shadow map (only the 32 lights with one) or Lumen's distance fields, which are coarser than triangles (mesh SDFs of at
  most 48 voxels per axis, 0.25 m global voxels beyond 2 m from both ends). The screen trace marches the depth buffer
  linearly (8 steps, no HZB traversal: the HZB holds the farthest depth, made for culling). Samples come from one
  representative pixel per 2×2 block (all four pixels shade with its pdf; edges between surfaces converge over the
  rotating representative), always four per block (no adaptive sample count, no "skip rays outside penumbrae" of UE
  5.8, no per-light-type budget); guiding uses a 256-bit hash per tile (lights 256 indices apart share a bit). The
  sun stays on its classic path (virtual shadow map), and so do volumetric fog (the clustered lights, shadowed only by
  the shadow-mapped ones) and translucent surfaces (forward clustered loop): with hundreds of lights they still see
  the unshadowed ones. Lumen's surface cache still lights every card texel and voxel with every light in range (one
  global-field shadow cone each), which becomes the dominant light-count-dependent cost (cathedral 33 → 1000 lights:
  surface + radiance cache 1.5 → 2.4 ms, cards 0.07 → 0.35 ms, in both modes). Light culling stays brute force per
  froxel (0.01 → 0.30 ms at 1000 lights). The denoiser lags behind moving lights by up to 24 frames where the variance
  clip does not catch the change, and its spatial filter softens lighting detail on fresh (disoccluded) pixels.

## Flags and hotkeys

* Hotkey **`=`** (`Equal`): MegaLights on / off.
* Debug views **18** (`--view 18`, `heliosSetView(18)`): MegaLights samples; **19**: MegaLights lighting, samples |
  denoised.
* `--megalights` (native) / `?megalights=1` / `heliosConfig.megaLights` (browser): start with MegaLights on.
* `--lights N` (native) / `?lights=N` / `heliosConfig.lights` (browser): the **candle-lit cathedral** — every candle
  flame its own shadow-casting light (60 candelabrum candles, 88 chandelier candles, 2 altar candles) plus votive
  racks (three tiers of 13 tea lights against the aisle walls of each bay, 858 candles), the first N lit (up to 1008).
  Default (no flag): the 33 fixture lights as before. E.g. `--scene cathedral --lights 1000 --megalights --sun 200,-30`.

## Measurements

Cathedral (medium detail) at night (`--sun 200,-30`), 1600×900, default view unless noted, `--frames 60`; GPU pass
times are the profiler's smoothed values, minimum of 4 interleaved runs (the RTX 3090 was shared with up to six other
agents' jobs, medians are noisier). "Direct" = light culling + local light shadow mask + lighting (+ MegaLights).

| Lights | Classic direct | MegaLights direct | MegaLights passes | Classic lighting | Frame (classic → ML) |
|---|---|---|---|---|---|
| 33 (default scene) | 1.03 | 1.84 | 1.37 | 0.88 | 6.38 → 7.15 |
| 100 (`--lights 100`) | 2.45 | 2.52 | 1.95 | 1.95 | 8.49 → 8.60 |
| 300 | 2.78 | 2.54 | 1.99 | 2.15 | 9.03 → 8.71 |
| 1000 | 3.53 | 2.74 | 1.95 | 2.50 | 10.55 → 9.75 |
| 1000, aisle view | 3.63 | 2.61 | 1.86 | 2.52 | 10.11 → 9.08 |

Image quality: PSNR of the MegaLights frame against the classic one (two classic runs differ by 66.7 dB). Stability:
`--dump-dir` sequences of the last 12 frames of a static camera, compared frame to frame (see the bullet above).
