# Translucency, coloured glass shadows and water

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Translucency** (`translucency.rs`, `translucency.wgsl`, `water.wgsl`). As in UE5, a separate forward pass after
  the deferred lighting: `Glass`, `Water` and `Translucent` (glTF `BLEND`) instances are skipped by the main view's
  culling (they never occlude through the visibility buffer / HZB) and drawn as plain indexed meshes — the finest
  in-memory level of their virtual mesh, decoded once — one draw per instance, frustum culled and sorted back to
  front on the CPU (GPU cluster lists are unordered and drawn by one indirect call). The lit HDR image is copied with
  a 5-level blurred mip chain; refracting surfaces read it at a displaced position and replace the pixel, blended
  ones are premultiplied "over". **Layers** are composited in per-pixel depth order wherever translucent surfaces may
  overlap (depth peeling, so every surface keeps its own shading model, refraction included): the CPU bounds the
  layers any pixel can see — the most projected bounding boxes of visible instances stacked at one point, a
  non-convex mesh counting as all (convexity / flatness is tested once per decoded mesh) — and for *n* > 1 layers,
  depth-only draws peel the nearest *n* − 1 into a depth texture array (each keeps the nearest fragment behind the
  previous layer). Then everything behind them is drawn in instance order as before, and each peeled layer over
  it, back to front, after a fresh copy of the scene colour — so water refracts the glass and panels standing in it,
  glass shows the glass behind it, intersecting surfaces cross correctly, and refraction never fetches a layer that
  lies in front. Up to 4 layers are exact (deeper ones fall back to instance order); views without overlap (the
  scenes' initial ones) peel nothing and cost what they did. `/` (`RenderSettings::translucency_layers`) toggles it.
  Cost (1600×900, RTX 3090, medians of 4, all translucency passes): the initial views unchanged (valley 0.43 → 0.42 ms,
  cathedral 0.40 → 0.39 ms), one peeled layer (cathedral apse) 0.42 → 0.49 ms, three (lake test corner) 1.86 → 2.02 ms
  and (aisle glass screens) 0.45 → 0.89 ms; the layer depths take 4 bytes per pixel and layer at the internal
  resolution's capacity (17 MB for three at 1600×900), allocated once a view needs them.
  Shading: GGX sun with VSM + cloud shadows, clustered local lights (with MegaLights: MegaLights' froxel volume,
  every light shadowed, [lights](lights.md)), Lumen radiance
  cache for ambient light, reflections from a screen-space trace against the opaque depth and colour, then Lumen's
  global distance field + surface cache, then sky with the on-screen clouds (glass seen face on reflects only a few
  percent and takes the radiance cache's blurry view instead of the distance-field trace; above 10 % reflectance —
  grazing angles, a pane's edge seen from just above it — it traces, round 8); aerial perspective and volumetric fog are
  applied at the surface depth exactly (`A·surface + B·(1 − T) + T·background`, the background already carries the
  fog behind). **Water**: 16 directional waves (3 m down to 5 cm, directions fanning out wider for shorter ones,
  each crest bent sideways by its own slow sinusoids, evaluated on a gently warped domain) with sharpened crests and
  deep-water dispersion as a per-pixel normal
  (sub-pixel waves fade into roughness), Beer–Lambert absorption + single scattering over the refracted path to the
  bottom, shore foam, sun glints. **Stained glass**: lead came opaque; glass tints the blurred scene behind and
  diffuses sky (scaled by how open the far side is, from the radiance cache) and sun — matching what Lumen's surface
  cache emits into the interior, which still gets the panes' coloured light through their distance fields — and the
  sun through the panes lands in their colours (next bullet). Untextured
  `Glass` is a clear pane, tinted by its base colour, its view through blurred by its roughness. Glass and blended
  surfaces under water reflect the water around them rather than the sky, and seen from under water they are lit
  and fogged through the water like the opaque scene. The pass writes the surfaces' own motion vectors for TSR.
  `RenderSettings::translucency` / `U` toggles it.
* **Adaptive voxel-based OIT** (round 8; `avboit.rs`, `avboit*.wgsl`, `oit.rs`; UE 5.9's experimental
  `r.OIT.Method=1`). The default where the device samples 19 textures per shader stage; `4` /
  `RenderSettings::avboit` / `--wboit` / `?oit=wboit` / the editor's *Voxel-based OIT* switch back to the round-7 path
  (depth peeling + weighted blended particles). A camera-aligned grid of **columns** (one per 4² internal pixels up to
  2048 wide, 8² above) with **16 cells** each, spaced logarithmically between the nearest and farthest translucent
  fragment of the column's 3×3 neighbourhood (*adaptive*: the cells sit where translucency is; neighbours space them
  alike). A low-resolution draw of every order-independent surface and particle first records the bounds, then adds
  each fragment's optical depth `−ln t` per colour channel to its cell with 32-bit atomics on 16-bit fixed-point
  fields (no rasteriser-ordered views in WebGPU; a sum is order independent); the visible volumes integrate their
  extinction over the cells (4 samples per box and cell, plus what lies in front of the near bound; volumes never
  widen the bounds — a lake's mist would spread the cells over metres). A resolve turns each column into the
  transmittance in front of each cell (rgba16f 3D texture, plus the surfaces-only part in rgba8 for the volume
  march). Full-resolution fragments then **add** their own light weighted by the transmittance in front of them
  (bilinear between four columns; within a cell the medium counts as spread evenly, the fragment's own share taken
  out — continuous in depth, and fragments sharing a cell are ordered by their position in it) and **multiply** the
  background's transmittance by their own (coloured). The volume march weights each step by the surfaces and
  particles in front of it (one column per pixel, picked at random with the bilinear weights; its temporal resolve
  averages the pick). So sparks inside smoke are dimmed by the smoke in front of them, smoke behind a tinted panel is
  tinted, particles between two panes are composited between them, and crossing panels order per pixel without
  peeling. **The material contract** (as UE's): an order-independent surface contributes only its own radiance and
  its transmittance, never a read of the scene colour behind it (the composite applies the background once).
  Therefore **refracting surfaces** — water, stained or frosted glass, glass with bump — keep the depth-peeled pass,
  but only their *nearest layer per pixel*: the nearest refracting depth splits the effects into a back and a front
  part (as before), the back part (order-independent surfaces, refracting glass behind the nearest layer as surface
  light + transmittance, particles, the volume) is composited before the refracting layer, which refracts and blurs
  it, the front part after. Where water overlaps another refracting surface (water can't follow the contract: its
  radiance holds the refracted bed), the refracting surfaces are peeled as before. Non-refracting surfaces in front
  write their motion vectors; TSR's reactive mask comes from the particles' own transmittance (alpha of the
  transmittance target). Cost (1600×900, RTX 3090, minima of 3 interleaved runs, all translucency + OIT passes): the
  lake's crossing panels 1.25 → 0.92 ms (no peeling), the cathedral's aisle screens 0.82 → 0.68 ms, views with only
  particles and volumes +0.05–0.10 ms (the grid 0.02–0.07 ms, the composite's second draw, the march's grid taps);
  the grid takes 36 MiB at 1600×900. Details and limits: `docs/round8/translucency.md`. In the browser only the
  refracting surfaces' classic shading models (and peel variants) are queued while AVBOIT is on (the others compile
  when it is switched off: `webperf`, 1–2 s each in Firefox).
* **Coloured light through glass** (`glass_shadow.rs`, `glass_shadow.wgsl`; UE5's translucent shadows). Panes that
  let the sun through (not shadow casters) would light the interior white; instead the visible `Glass` instances are
  drawn from the sun into a map of their own — an orthographic view along the sunlight fitted to their boxes, 256²
  to 2048² texels (the cathedral's 84 m of windows: 4.1 cm) — keeping per texel the pane nearest the sun: what it
  transmits (base colour × texture, the stained-glass factor, Fresnel, none through lead came) and its depth along
  the light. A point behind its texel's pane gets the sun filtered by it: the deferred lighting (blurred with the
  distance behind the pane, ~10 cm under a pane 10 m up), the volumetric fog (coloured shafts) and Lumen's card
  lighting (coloured bounce). Drawn only when the sun, the panes or the scene change (0.034 ms a frame with an
  animated sun); the lighting pays ~0.05 ms at 1440p. `RenderSettings::glass_shadows` / `End` /
  `--no-glass-shadows` toggles it (`docs/round7/glass-shadows.md`). Since round 8 the translucent surfaces (glass
  behind glass, blended panels) and the particles' per-particle lighting take the sun through the map too.
* **Local lights through glass** (round 8; `glass_light.rs`, `glass_light.wgsl`). The 8 local lights nearest the
  camera whose range reaches a glass pane get a cube map of their own (6 × 256², rgba16f): the panes within range
  drawn from the light (`fs_glass_light`), per texel the transmittance of the pane nearest the light and its distance.
  A receiver farther from the light than its texel's pane (5 cm bias) gets the light in the pane's colour: the
  deferred lighting's clustered loop and MegaLights' shading call `glass_light_transmission(light, position, p)` per
  light (one bilinear tap; a uniform maps light index → layer). A map is redrawn only when its light moves, the
  panes or the scene change (the swinging chandeliers: every frame, 0.06 ms). Lighting pays 0.06–0.10 ms at night
  (33 / 300 candles), MegaLights' shading 0.02 ms. Same switch as the sun's (`End`). 24 MiB once
  a light near glass exists. Lights beyond the eight, the volumetric fog, particles and Lumen's surface cache see
  glass as clear for local lights.
* **Rivers** are simulated shallow water drawn by the same water shading (flow-mapped waves, simulated foam): see
  [terrain and rivers](terrain.md).
* **Water bodies** (`water.rs`, `water.wgsl`, `shore.wgsl`; UE's Water system for lakes). Every `Water` instance is a
  flat *water body* (its world rectangle and level, from its bounds); one uniform carries them with the medium.
  **Lake bed and shore**: the G-buffer resolve turns the landscape into sediment where a body covers it — sand and
  gravel near the water line, dark silt with scattered, silt-coated stones and algae patches deeper, ripples in the
  shallows (procedural, footprint-filtered: lumpy pebble domes on jittered grids, fbm) — and soaks a band just above
  the water (darker, glossy). It runs per pixel after the runtime virtual texture lookup (or the direct layering), at
  the mesh's exact height, so the bed meets the water line exactly whichever path shaded the landscape (the RVT's
  heightfield is 1.8 m coarse). **Caustics**: the sun's light on the bed is `1 / det(I + d·(1 − 1/n)·H)`, H the
  Hessian of the same 16 waves at the point the refracted sun ray entered (one fixed-point step for the waves' own
  displacement), each wave fading where its pattern is finer than the bed's pixel or the sun disk's blur over the
  depth. The bent crests make the few dominant short waves cross at angles that change from place to place, so the
  caustics form irregular cells rather than a lattice of parallelograms (straight crests did); the bends only vary
  across each wave's travel direction, so they travel with it and nothing shimmers. Caustics they scale only the bed's direct-sun share `(E_sun·c + E_amb)/(E_sun + E_amb)` (no albedo needed), with E_sun
  through the bed's virtual shadow map and cloud shadows — so no caustics in shadow. **Underwater**: with the camera
  in a body (in the browser its pipeline compiles once the camera comes within 10 m of a body, round 8 `webperf`;
  a frame under water before that skips the fog), a full-screen draw fogs the opaque scene by absorption and single scattering along the view (to the
  surface where the ray leaves the water) and adds caustics on the bed. The in-scattered light is the sun (through the
  surface's Fresnel transmission and its shadows) and the sky / GI irradiance (Lumen's radiance cache) entering the
  surface above the middle of the ray, fading with depth: since the depth changes linearly along the ray, the single
  scattering integral has a closed form, so looking up is brighter than looking down and deep water darkens. Lake
  water scatters mostly forwards (a quarter isotropic, three quarters Henyey–Greenstein: g 0.8 for the sun beam, a
  broad 0.5 lobe for the diffuse sky light from above), so water seen sideways stays clear and the refracted sun
  glows. The default medium is a clear mountain lake: pure water's absorption with a little dissolved organic matter
  (0.40, 0.06, 0.09 /m) and light particle scattering (0.03–0.038 /m), about 10 m of visibility; water surfaces are drawn two-sided and from
  below show Snell's window (the world above through refraction, screen trace → distance field → sky, with exact
  dielectric Fresnel) and total internal reflection of the deep water around it, no sky leaking; air fog is skipped.
  Reflection rays leaving a water surface ignore screen-space hits below it (the submerged bed is in the depth buffer).
  Exposure under water keeps Snell's window in its metering: the histogram otherwise drops the brightest 10 % of the
  view (for sun disks and flames), which under water is the window, the scene's light source — the camera then
  exposed for the dim water alone and burnt the window out to white.
  Cost (valley high, 1600×900, RTX 3090, min of 5): looking down at the lake filling the screen, translucency 2.00 →
  2.40 ms and the material pass 0.26 → 0.48 ms (GPU 5.93 → 6.64 ms); the default view (lake ~10 % of the screen)
  4.54 → 4.59 ms; scenes without water are unchanged. Under water (full-screen pass + surface from below): 1.97 ms.
