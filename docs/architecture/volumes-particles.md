# Heterogeneous volumes, GPU particles and OIT

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Heterogeneous volumes** (`volumes.rs`, `volume_sim.wgsl`, `volume_render.wgsl`, `media.rs`, `media_common.wgsl`;
  UE's Heterogeneous Volumes). `Scene::volumes` holds `VolumeEmitter`s: boxes of soot density, temperature and fuel on
  a voxel grid (the campfire 40×80×40 over 1.6×3.2×1.6 m, the incense 32×112×32). A `FluidSimulation` runs on the GPU
  every frame in 3D textures: vorticity, forces (Boussinesq buoyancy — heat lifts, soot weighs down —, vorticity
  confinement that restores the eddies numerical diffusion removes, a source sphere whose injection flickers with
  animated noise, ambient wind, damping), divergence, 12–16 red-black SOR sweeps of the pressure Poisson equation
  (warm-started from the last frame; solid floor, open sides and top), projection, semi-Lagrangian self-advection of
  the velocity and **MacCormack** advection of the state (back-advected, corrected by half the round-trip error and
  clamped to the departure point's 8 texels, so smoke keeps detail without overshoot; half of the plain
  semi-Lagrangian result is mixed back in — pure MacCormack left voxel-sharp fronts that read as blocks wherever the
  smoke turns opaque), then combustion (fuel burns into heat and soot), cooling, dissipation and a fade at the open
  boundaries. A `ProceduralVolume` (the valley's lake mist) fills the grid from animated noise instead — no
  simulation for distant or static media. Rendering: a **lighting cache** at half the grid resolution stores per
  voxel the sun's transmittance (16 steps through the volume towards the sun × the virtual shadow map at the fog's
  page level × cloud shadows) and the isotropically scattered light: ambient from Lumen's radiance cache (the sky
  above occluded by the volume itself, an 8-step march up) and up to 8 local lights with their shadow maps. The
  **march** runs once per pixel of the visible volumes' screen rectangle (up to 4 boxes, ordered per pixel) at the
  internal resolution: trilinear density and temperature (each sample offset randomly by up to half a voxel —
  stochastic filtering that turns the trilinear filter's voxel staircases into a smooth falloff once the temporal
  resolve averages it), the cache with a Henyey–Greenstein phase for the sun plus a second, softer and less attenuated
  octave standing in for multiple scattering (Wrenninge's approximation; without it thick smoke turned black away
  from the sun), **blackbody emission** from temperature (Planck's law at three wavelengths: dull red edges, a
  yellow-white core ~40× brighter), energy-conserving integration, strides that grow 4× through clear air and step
  back at the medium's edge, early exit at 0.3 % transmittance; each part is fogged at its opacity-weighted depth
  (froxel fog + aerial perspective, as translucency). Step offsets are jittered per pixel and frame; a **temporal
  resolve** reprojects the last result with the depth of the volume each pixel sees, clamps it to the current 3×3
  neighbourhood and blends 25 % — the jitter noise averages out, the flames don't ghost. The ray is **split at the
  nearest translucent surface** (see *Order-independent transparency*), so glass and water show the smoke behind
  them. A fire **drives a scene light** (`VolumeEmitter::light`): each frame, before the light culling, a
  one-thread-per-light pass sets its colour to the scene's colour × the simulation's summed blackbody emission
  (workgroup sums into fixed-point atomics) relative to its running average — the flicker the viewer sees, on the
  clearing, the logs' shadows and Lumen's surface cache, a frame late and without a read-back.
* **GPU particles** (`particles.rs`, `particles.wgsl`; a Niagara-lite). `Scene::particles` holds `ParticleEmitter`s
  (spawn rate and bursts, sphere / box / disc shapes, lifetime range, a velocity cone, gravity, drag, curl-noise
  turbulence, collision, size and a 4-key colour/opacity gradient over life, lit or emissive — colour or blackbody
  cooling over life —, billboard or velocity-stretched quads, soft distance, phase anisotropy). Every emitter owns a
  fixed range of slots in one buffer; the CPU only turns rates and bursts into a spawn count per frame (a carried
  fraction), dead slots claim spawns with one atomic counter per emitter — no free lists, compaction or read-back;
  an emitter's first frame pre-warms its sustained population at random ages. **Simulation** (one dispatch for all
  emitters): gravity, drag, curl noise (the curl of a 3-component gradient-noise potential: divergence-free swirls),
  and **collision with the depth buffer**: a particle that went behind the visible surface (within a thickness that
  grows with its speed) is put back onto it and bounces off the G-buffer normal with restitution and friction, or
  sticks (embers settle on the ground and keep glowing). **Lighting** once per particle per frame (UE's per-particle
  translucent lighting): sun through the virtual shadow map and cloud shadows with a Henyey–Greenstein phase (dust
  glints in the clerestory's shafts when looking towards the sun), Lumen's radiance cache, the clustered local lights
  with one shadow-map tap, emission, then the fog in front — the premultiplied radiance each fragment scales. **Draw**:
  one instanced draw of every slot (dead ones collapse in the vertex shader), camera-facing or stretched along the
  screen-space velocity by the distance covered in 40 ms (sparks), procedural glow / puff sprites, cut by the opaque
  depth and faded in front of it (soft particles), bright emitters bloom.
* **Adaptive voxel-based OIT** (round 8, the default; `avboit.rs`): particles, the volume march and the
  non-refracting translucent surfaces are weighted by a measured transmittance grid instead of the depth guess below,
  so sparks inside smoke are dimmed by the smoke in front of them and particles between two panes composite between
  them; the buckets split at the nearest *refracting* surface. Described in
  [translucency.md](translucency.md#translucency-coloured-glass-shadows-and-water); `--wboit` / `Home` select the
  path below.
* **Order-independent transparency** (`oit.rs`, `oit.wgsl`, `effects.rs`). Particles use **weighted blended OIT**
  (McGuire and Bavoil 2013): each fragment adds premultiplied colour × a depth weight to an rgba16f accumulation
  target and multiplies an r16f revealage target by 1 − opacity; a full-screen pass composites average colour ×
  coverage — any number of layers (a shower of sparks, thousands of motes), no sorting. **Order with the translucency
  pass**: when translucent surfaces are drawn, their nearest depth is rendered first (the translucency module's first
  peeling layer, on its own); particles write a *front* or *back* pair of targets and the volume march splits its
  rays at that depth. The back bucket (particles over volume over scene) is composited before the translucency pass —
  glass tints and water refracts the mist and sparks behind it —, the front bucket after it. **TSR**: the front
  composite writes a reactive mask into the motion target's z (free by then; UE's responsive AA): there TSR keeps at
  most 1.5 frames of history (mixed by coverage; volumes at half strength, they have their own temporal filter), a
  tight clip box and no thin-feature lock, so sparks and flames leave no trails. Scenes without emitters, or with `'`
  off, record none of these passes and render as before (valley with `'` off: max 5/255, within run-to-run noise).
  Debug view 21: particle overdraw (red → yellow) and volume march steps (blue → cyan) over a grey albedo;
  empty-space skipping leaves the clear air of a box at a quarter of the steps.
  Cost of volumes + particles + OIT (shared GPU, medium detail, interleaved on/off runs, medians of 3; "passes" = every
  pass the feature adds):

  | View | 1600×900: GPU off → on | passes | 2560×1440: GPU off → on | passes |
  |---|---|---|---|---|
  | Forest campfire close-up (`--eye 4.3,1.3,0.6 --at 7,1.0,-1`) | 9.19 → 9.65 ms | 0.58 ms | 16.4 → 17.2 ms (min) | 1.19 ms |
  | Cathedral censer (`--eye 66.2,1.8,0.4 --at 68.4,2.4,1.9`) | 6.26 → 6.46 ms | 0.38 ms | 10.88 → 11.18 ms (min) | 0.59 ms |
  | Cathedral initial view (24 000 motes, incense far) | 6.36 → 6.81 ms (min) | 0.37 ms (min) | 11.56 → 11.89 ms (min) | 0.45 ms (min) |
  | Valley initial view (lake mist, 100×10×91 cells) | 5.19 → 5.86 ms | 0.53 ms | 8.75 → 9.72 ms | 1.24 ms |

  Per pass: campfire simulation (16 SOR sweeps) 0.10–0.11 ms, incense 0.08 ms (both resolution-independent);
  lighting caches 0.02–0.03 ms; march + temporal resolve 0.15–0.45 ms at 1600×900 (the valley mist covers a third of
  the screen), 0.29–1.07 ms at 2560×1440; particle simulation + lighting 0.02 ms (1 280 fire particles) – 0.09 ms
  (24 000 motes); OIT draw 0.02–0.04 ms; nearest translucent depth 0.01 ms; composites 0.04 / 0.10 ms. GPU memory
  while on screen: 72 bytes per internal pixel (104 MiB at 1600×900) plus the grids (campfire: seven 1 MiB rgba16f
  fields, 1 MiB of pressure and divergence, a 0.1 MiB cache). Temporal stability (static camera, pixels changing by
  more than 24/255): inside the campfire's rectangle 0.11 → 0.42 % (the flames and sparks move), outside it 0.070 →
  0.083 %; cathedral 0.002 → 0.005 %. Not in these numbers (the light stays with `'` off): the campfire's 18 m
  shadow-casting light refreshed every one of its cube-map pages every frame under the swaying ferns, grass, bushes
  and trees of the clearing (+2.0–2.3 ms of `vsm raster` in the forest's default view at 2560×1440: 18.4 ms, 16.6 ms
  with its shadows off); since round 5 its pages are refreshed by the motion of the clusters they hold and drawn
  with tighter face rectangles (16.7–17.6 ms, *Local light virtual shadow maps*).
