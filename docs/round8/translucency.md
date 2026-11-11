# Order-independent translucency (AVBOIT) and local lights through glass (round 8: `avboit`, `glass-local-lights`)

Design note. The user-facing text is in [`docs/architecture/translucency.md`](../architecture/translucency.md)
(*Adaptive voxel-based OIT*, *Local lights through glass*) and
[`volumes-particles.md`](../architecture/volumes-particles.md); the remaining gaps are items 5, 11 and 16 of
[`docs/status.md`](../status.md). Images: `translucency/`.

## 1. Adaptive voxel-based OIT (`avboit`, order 208)

### Problem

Round 7 composited translucency three ways at once: glass, water and blended meshes by depth peeling (up to 4
layers; the layer count from overlapping *bounding boxes*, so overlapping boxes cost up to three extra depth-only
draws, scene-colour copies and shading passes), particles by weighted blended OIT (a depth guess that saturates), and
the volume march per box. The three only met at one split, the nearest translucent surface: particles between two
panes went with the farther bucket, sparks inside smoke were never dimmed by the smoke in front of them, smoke behind
a tinted panel was not tinted by it (both were in the same bucket, composited particles-over-volume-over-scene).

### UE reference

UE 5.9's Adaptive Voxel-Based OIT (`r.OIT.Method=1`, experimental): every translucent primitive is drawn at low
resolution into a camera-aligned voxel grid of light transmittance whose slices adapt to where translucency is; the
full-resolution translucency then weights each surface by the transmittance in front of its depth, so blending is
order independent; coloured transmittance; additive, modulate and alpha composite; no refraction; same-slice surfaces
blend as a weighted average; seams at cell scale. Its material contract: a translucent material outputs only its own
lighting / specular / emissive; a material that samples SceneColor adds the background twice and looks washed out.

### Design

`avboit.rs`, `avboit_common.wgsl` / `avboit_read.wgsl` / `avboit_write.wgsl` / `avboit.wgsl`, hooks in `effects.rs`,
`oit.rs`, `particles.rs/.wgsl`, `volumes.rs` / `volume_render.wgsl`, `translucency.rs/.wgsl`.

* **The grid.** Columns of 4×4 internal pixels (8×8 above 2048 px wide: 320×180 columns at 2560×1440), 16 cells each.
  Per frame: clear → **bounds** (a low-resolution draw of the order-independent surfaces and the particles, viewport =
  columns, each fragment tested against the opaque depth at its column's centre pixel, `atomicMax` of the depth's bits
  for near and far) → **dilation** (each non-empty column takes the bounds of its 3×3 neighbourhood, rounded to half
  precision — the precision readers get them in — so writers and readers agree on the cells, and neighbours space
  their cells alike: without it a surface crossing columns changed cells from column to column and left streaks) →
  **transmittance** (the volumes integrate their extinction per cell, 4 samples per box and cell, and in front of the
  near bound; then the same low-resolution draw adds each fragment's `−ln t` per channel) → **resolve** (per column the
  exclusive product: the transmittance in front of each cell, rgba16f 3D texture with the bounds in slice 0; the
  surfaces-only part in rgba8 for the march).
* **Atomics instead of ROVs.** WebGPU has no rasteriser-ordered views, but optical depth adds, so order doesn't
  matter: per cell two `atomic<u32>` hold red | green and blue as 16-bit fixed point (1/512 per unit, ≤ 16 per
  fragment) — the "packing per channel" — plus one word for the volumes' f32 optical depth (written by one thread per
  column, no atomics needed). A field above 0xc000 takes no more (optical depth 96: long opaque), so concurrent adds
  can't carry into the neighbouring field.
* **Adaptive slices.** Logarithmic between the column's bounds. Volumes don't widen the bounds: the valley's lake mist
  covers the whole lake, and with it in the bounds the crossing panels at the lake's north end landed in one or two
  cells and averaged (`lake-panels-...`: first attempt); their optical depth in front of the near bound is kept per
  column instead.
* **Within a cell** the medium counts as spread evenly: a fragment at fractional position `f` of cell k is weighted by
  `T_start(k) · (T_cell / t_own)^f` (clamped to `[T_start · T_cell, T_start]`), its own transmittance taken out. This
  is continuous in depth (a fragment crossing a cell boundary doesn't pop, which UE's start-of-cell weight does) and
  orders fragments sharing a cell by their position; UE's weighted average remains for fragments at the same depth.
  A surface steep in depth reaches cells in front of and behind the fragment within the columns the bilinear lookup
  reads (the grid holds it at the column centre's depth); it looks its weight up `1.5 · tile · fwidth(depth)` in front
  of itself so it doesn't dim itself (it did: dark streaks along the columns on the red panel).
* **Full resolution.** Particles (`fs_particle_avboit`) and order-independent surfaces (`fs_oit_front/back`) add
  `own light × weight` to the accumulation target and multiply the (now rgba16f) transmittance target by their own
  transmittance — coloured for glass; alpha carries the particles' own transmittance for TSR's reactive mask, so a
  clear pane doesn't make TSR distrust its history. The volume march multiplies each step's light by the surfaces in
  front of it, from one column per pixel picked at random with the bilinear weights (its temporal resolve averages
  the pick; 4 taps per step would cost more than the march). The composite is two draws in one pass: background ×
  transmittance (`Zero, Src` blending), then + light (`One, One`).
* **The split and the material contract.** `Candidate::refracts`: water, and glass with a texture, bump or roughness
  above 1/16 (the roughness from which `shade_glass` blurs the background). Non-refracting glass and blended surfaces
  contribute `surface · fog + inscatter · (1 − t)` and `t` — never `s.background_uv` (stated at the top of the
  AVBOIT section of translucency.wgsl). Refracting surfaces stay in the forward pass, which composites only the
  **nearest refracting layer per pixel** (the effects' nearest-depth target, now drawn from refracting surfaces only,
  bound as a one-layer array: `LayerRange {far: 0}`); refracting *glass* behind that layer joins the back bucket as
  surface light + transmittance (its blur and refraction would be seen through the nearest layer's blur anyway). The
  back bucket — those, the order-independent surfaces and particles behind the split, the volume behind it — is
  weighted by `T(d) / T(split)` (only what lies between the split and the fragment; the front composite dims the
  rest) and composited before the refracting layer, which then refracts and blurs it; the front bucket after. Water
  can't follow the contract (its radiance holds the refracted bed), so where water overlaps another refracting
  surface on screen the refracting surfaces are depth-peeled as in round 7 (and the order-independent ones, never
  peeled, split at the nearest).
* **Fallbacks.** The surfaces' OIT pipelines sample 17 textures (the grid is the 17th); devices offering fewer use
  the round-7 path. `RenderSettings::avboit`, `4`, `--wboit`, `?oit=wboit`, the editor's *Voxel-based OIT*.

### Alternatives considered

* **A global (per-frame) depth range** instead of per-column bounds: one hardware-filtered texture fetch per lookup,
  but the cathedral's motes span 5–60 m and a pane 2 m away would share its cells with them.
* **Rendering into a 3D texture with multiplicative blending:** one render pass per slice (WebGPU attaches one depth
  slice at a time), and a 32-byte limit per sample for the colour attachments.
* **One u32 per channel** (no packing): 50 % more grid memory; the packing's carry is guarded instead.
* **UE's per-cell normalisation** (cell opacity / summed coverage): implemented first; crossing panels showed
  stair-stepped seams where they moved between "same cell" and "different cells". The even-medium interpolation
  replaced it (no extra coverage field either).
* **Depth peeling everything** (round 7) remains as the A/B path; peeling only the nearest refracting layer replaces
  up to three peel draws, copies and shading passes by one.

### Measurements

RTX 3090 (shared with other agents' builds), 1600×900, `--detail medium`, 200 frames, three interleaved runs per path,
minima of the per-pass minima (GPU ms). "Translucency + OIT" sums every pass either path records for translucency,
particles' OIT draw, the grid and the composites (not the volume march, listed apart).

| View | GPU frame (min) AVBOIT / round 7 | Translucency + OIT | Volume march |
|---|---|---|---|
| Valley lake, crossing panels (`--eye -2.8,3.2,27 --at -1.5,2.6,30.6`) | 5.37 / 5.63 | 0.92 / 1.25 (peel 0.026, two copies 0.157, shading 1.004 → one copy 0.079, OIT surfaces 0.50, nearest refracting layer 0.21, grid 0.06) | 0.713 / 0.650 |
| Cathedral aisle glass screens (`--eye 14,1.7,10.3 --at 2.5,2.5,10.5`) | 5.37 / 5.51 | 0.68 / 0.82 | – |
| Cathedral initial view (24 000 motes) | 5.93 / 5.77 | 0.47 / 0.38 (grid 0.065: bounds 0.024 + transmittance 0.024 for the motes) | 0.060 / 0.057 |
| Valley initial view (lake mist) | 4.77 / 4.67 | 0.47 / 0.43 (the grid 0.015, the composites' second draws; an empty OIT surface pass and its scene-colour copy, 0.04, before off-screen surfaces were skipped) | 0.474 / 0.441 |
| Forest campfire, night (`--sun 200,-30 --eye 3.5,1.4,1.0 --at 7,1.0,-1`) | 7.46 / 7.45 | 0.10 / 0.04 | 0.173 / 0.162 |
| Forest campfire, day close-up (`--eye 4.3,1.3,0.6 --at 7,1.0,-1`) | 7.61 / 7.57 | 0.10 / 0.04 | 0.226 / 0.211 |

So where translucent surfaces overlap, AVBOIT is cheaper (no peeling: one copy instead of two, one shading pass of the
refracting layer); where only particles and volumes are drawn it costs 0.05–0.10 ms more (grid 0.04–0.07 ms, the
composite's second draw 0.014 ms, the march's grid taps 0.01–0.06 ms). Memory at 1600×900: the grid 36 MiB
(400×225 columns: cells 17 MiB, front 13 MiB, surfaces 6 MiB, bounds 1.4 MiB), allocated once a frame uses it; the
transmittance targets went from r16f to rgba16f (+17 MiB, both paths).

### Remaining limits

See status items 5 and 11: order within a cell approximate (even-medium model), low-resolution transmittance at pane
edges (a few pixels), sub-column sparks may miss the low-resolution draw, what lies between two refracting layers
composites behind both (only the nearest refracting layer splits), refracting glass behind the nearest refracting layer
loses its own blur, the volume's grid samples (4 per cell) are coarser than its march, no Nanite translucency.

## 2. Local lights through glass (`glass-local-lights`, order 209)

### Problem

Round 7 coloured only the sun through glass (`glass_shadow.rs`). Candles and chandeliers shone through stained glass
white, and translucent surfaces and particles behind glass got the sun white.

### Design

* **Per-light cube maps** (`glass_light.rs`, `glass_light.wgsl`, `vs/fs_glass_light` in translucency.wgsl): the 8
  local lights nearest the camera whose range reaches a glass pane (sphere vs. pane box) get a 6 × 256² rgba16f cube
  layer: the panes in range drawn from the light (left-handed 90° faces matching WebGPU's cube lookup; a unit test
  checks the orientation), per texel the transmittance of the pane nearest the light (as `fs_glass_shadow`: colour ×
  texture × stained factor × Fresnel × coverage, lead came 0) and its distance. A light keeps its layer while it stays
  picked; a layer is redrawn only when its light moves or the panes / scene change.
* **The hook** `glass_light_transmission(light index, light position, p)` (shared include, bindings 92–94): a
  uniform maps light index → layer (8 bits per light, 1024 lights); four texels gathered and depth-compared each (a
  2×2 box filter: a filtered distance blended pane edges with the empty texels and drew coloured lines on pillars in
  front of windows), their transmittances averaged. Called in the deferred lighting's clustered loop and in MegaLights'
  `ml_shade_samples` (one line each; MegaLights' sampling is untouched).
* **Translucent surfaces and particles behind glass** take the sun through the sun's glass map: `direct_light` and the
  sun behind stained glass in translucency.wgsl (`glass_transmission_soft`), the particles' per-particle lighting
  (`media_sun_glass`, one tap; the media bindings gained 90/91).

### Why per-light maps

* A **colour layer in the local lights' virtual shadow maps** would follow their paging and caching, but those pages
  are depth-only, filtered for depth and drawn by the GPU-driven cluster passes; glass is a few hundred triangles drawn
  conventionally, and only lights near glass need it.
* A **world-space glass transmittance volume** would be shared by all lights but can't tell which side of a pane a light
  is on, nor resolve 4 cm lead came.
* Per-light maps cost memory (24 MiB for 8) and a redraw per moving light (6 small passes), nothing per receiver but
  four gathers for lights that have a map.

### Measurements

Same method (three interleaved runs, `--no-glass-shadows` as the reference, which also turns the sun's map off — at
night it has none), 1600×900, cathedral at night (`--sun 200,-30`):

| View | GPU frame (min) glass / white | Pass |
|---|---|---|
| Aisle screens, 33 candle lights (`--eye 14,1.7,10.3 --at 2.5,2.5,10.5`) | 5.23 / 5.10 | lighting 0.700 → 0.757 ms; maps redrawn every frame (the chandeliers swing): 0.058 ms |
| Candle-lit, 300 lights, behind the stained-glass screen (`--lights 300 --eye 0.8,1.6,7.0 --at 2.0,0.4,12.0`) | 7.30 / 7.17 | lighting 1.203 → 1.301 ms; maps drawn once (votive candles don't move) |
| The same with MegaLights | 7.88 / 7.81 | MegaLights shade 0.079 → 0.099 ms |

Memory: 24 MiB (8 × 6 × 256² rgba16f + one 256² depth), allocated once a light near glass exists.

Images: `translucency/candle-screen-white-vs-glass.jpg` (candle-lit, `--lights 1008 --sun 200,-30 --eye 0.8,1.6,7.0
--at 2.0,0.4,12.0`: the votive rack's candles 30 cm behind the stained-glass screen; left white, right through the
glass — the warm candlelight through the screen's blue and red panes lands dim and cool).

### Remaining limits

Only 8 lights (the candle-lit cathedral's 1008 candles: the 8 nearest the camera near glass); one pane per texel; no
blur with the distance behind the pane; the fog, particles, volumes and Lumen see local lights through glass as clear;
spot lights render all six faces.

## Browser

Headless Chromium (WebGPU on D3D11), `?scene=forest` and `?scene=cathedral`: every pipeline compiles in Tint (180 in
the cathedral), no console errors or validation messages; the AVBOIT key (`Home` then, `4` after the merge) and `End` toggle
(`translucency/browser-cathedral.jpg`, `browser-forest.jpg`).
