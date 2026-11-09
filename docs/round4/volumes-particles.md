# Heterogeneous volumes, GPU particles and order-independent transparency (round 4)

README text for the merge: an Architecture bullet (section 2, after *Volumetric fog* / *Translucency*), a pipeline
line, a "Status vs. Unreal Engine 5" item, the hotkey and the debug view.

## Pipeline diagram (section 2) — insert after the deferred lighting line

```
  → volumes (fluid simulation · lighting cache) · particles (simulation · per-particle lighting)
  → nearest translucent depth → volume march + temporal resolve → particles into OIT targets
  → composite of what lies behind the nearest translucent surface
  → translucency (…)
  → composite of what lies in front of it (+ TSR reactive mask)
```

## Architecture bullet

* **Heterogeneous volumes** (`volumes.rs`, `volume_sim.wgsl`, `volume_render.wgsl`, `media.rs`, `media_common.wgsl`;
  UE's Heterogeneous Volumes). `Scene::volumes` holds `VolumeEmitter`s: boxes of soot density, temperature and fuel on a
  voxel grid (the campfire 40×80×40 over 1.6×3.2×1.6 m, the incense 32×112×32). A `FluidSimulation` runs on the GPU
  every frame in 3D textures: vorticity, forces (Boussinesq buoyancy — heat lifts, soot weighs down —, vorticity
  confinement that restores the eddies numerical diffusion removes, a source sphere whose injection flickers with
  animated noise, ambient wind, damping), divergence, 12–16 red-black SOR sweeps of the pressure Poisson equation
  (warm-started from the last frame; solid floor, open sides and top), projection, semi-Lagrangian self-advection of
  the velocity and **MacCormack** advection of the state (back-advected, corrected by half the round-trip error and
  clamped to the departure point's 8 texels, so smoke keeps detail without overshoot; half of the plain
  semi-Lagrangian result is mixed back in — pure MacCormack left voxel-sharp fronts that read as blocks wherever the
  smoke turns opaque), then combustion (fuel
  burns into heat and soot), cooling, dissipation and a fade at the open boundaries. A `ProceduralVolume` (the valley's
  lake mist) fills the grid from animated noise instead — no simulation for distant or static media. Rendering: a
  **lighting cache** at half the grid resolution stores per voxel the sun's transmittance (16 steps through the volume
  towards the sun × the virtual shadow map at the fog's page level × cloud shadows) and the isotropically scattered
  light: ambient from Lumen's radiance cache (the sky above occluded by the volume itself, an 8-step march up) and up to
  8 local lights with their shadow maps. The **march** runs once per pixel of the visible volumes' screen rectangle
  (up to 4 boxes, ordered per pixel) at the internal resolution: trilinear density and temperature (each sample
  offset randomly by up to half a voxel — stochastic filtering that turns the trilinear filter's voxel staircases into
  a smooth falloff once the temporal resolve averages it), the cache with a Henyey–Greenstein phase for the sun plus a
  second, softer and less attenuated octave standing in for multiple scattering (Wrenninge's approximation; without
  it thick smoke turned black away from the sun), **blackbody emission** from temperature (Planck's law at three wavelengths:
  dull red edges, a yellow-white core ~40× brighter), energy-conserving integration, strides that grow 4× through clear
  air and step back at the medium's edge, early exit at 0.3 % transmittance; each part is fogged at its
  opacity-weighted depth (froxel fog + aerial perspective, as translucency). Step offsets are jittered per pixel and
  frame; a **temporal resolve** reprojects the last result with the depth of the volume each pixel sees, clamps it to
  the current 3×3 neighbourhood and blends 25 % — the jitter noise averages out, the flames don't ghost. The ray is
  **split at the nearest translucent surface** (see *OIT*), so glass and water show the smoke behind them. A fire
  **drives a scene light** (`VolumeEmitter::light`): each frame, before the light culling, a one-thread-per-light pass
  sets its colour to the scene's colour × the simulation's summed blackbody emission (workgroup sums into fixed-point
  atomics) relative to its running average — the flicker the viewer sees, on the clearing, the logs' shadows and
  Lumen's surface cache, a frame late and without a read-back. Cost: see the table below.
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
* **Order-independent transparency** (`oit.rs`, `oit.wgsl`, `effects.rs`). Particles use **weighted blended OIT**
  (McGuire and Bavoil 2013): each fragment adds premultiplied colour × a depth weight to an rgba16f accumulation
  target and multiplies an r16f revealage target by 1 − opacity; a full-screen pass composites average colour ×
  coverage — any number of layers (a shower of sparks, thousands of motes), no sorting. **Order with the translucency
  pass**: when translucent surfaces are drawn, their nearest depth is rendered first (the translucency module's first
  peeling layer, on its own); particles write a *front* or *back* pair of targets and the volume march splits its
  rays at that depth. The back bucket (particles over volume over scene) is composited before the translucency pass —
  glass tints and water refracts the mist and sparks behind it —, the front bucket after it. **TSR**: the front
  composite writes a reactive mask into the motion target's z (free by then; UE's responsive AA): with
  `flags::REACTIVE` TSR keeps at most 1.5 frames of history (mixed by coverage; volumes at half strength, they have
  their own temporal filter), a tight clip box and no thin-feature lock there, so sparks and flames leave no trails.
  Scenes without emitters, or with `'` off, record none of these passes and render as before (valley with `'` off vs
  before: max 5/255, within the 5/255 run-to-run noise).

## Measured cost

RTX 3090 shared with up to 6 other agents' builds and renders: interleaved on/off runs (`'` off = `--keys Quote`),
medium detail, 240 frames; medians of 3 runs at 1600×900, 2 at 2560×1440 (minimums within 0.1 ms of them unless
noted). Effect passes = every pass the feature adds.

| View | 1600×900: GPU off → on | effect passes | 2560×1440: GPU off → on | effect passes |
|---|---|---|---|---|
| Forest campfire close-up (`--eye 4.3,1.3,0.6 --at 7,1.0,-1`) | 9.19 → 9.65 ms | 0.58 ms | 16.4 → 17.2 ms (min) | 1.19 ms |
| Cathedral censer (`--eye 66.2,1.8,0.4 --at 68.4,2.4,1.9`) | 6.26 → 6.46 ms | 0.38 ms | 10.88 → 11.18 ms (min) | 0.59 ms |
| Cathedral initial view (24 000 motes, incense far) | 6.36 → 6.81 ms (min) | 0.37 ms (min) | 11.56 → 11.89 ms (min) | 0.45 ms (min) |
| Valley initial view (lake mist, 100×10×91 cells) | 5.19 → 5.86 ms | 0.53 ms | 8.75 → 9.72 ms | 1.24 ms |

Per pass (1600×900 / 2560×1440): campfire simulation 40×80×40 with 16 SOR sweeps 0.10–0.11 ms, incense 32×112×32
0.08 ms (both resolution-independent); lighting caches 0.02–0.03 ms; march + temporal resolve: campfire close-up
0.36 / 0.94 ms, censer 0.15 / 0.29 ms, valley mist 0.45 / 1.07 ms (it covers a third of the screen); particle
simulation + lighting 0.02 ms (1 280 fire particles) – 0.09 ms (24 000 motes); OIT draw 0.02–0.04 ms; nearest
translucent depth 0.01 ms; composites 0.04 / 0.10 ms. Debug view 21 shows where the march spends its steps: empty-space
skipping leaves the clear air of a box at a quarter of the steps. GPU memory while on screen: 72 bytes per internal
pixel (104 MiB at 1600×900) plus the grids (campfire: seven 1 MiB rgba16f fields, 1 MiB of pressure and divergence, a 0.1 MiB cache;
the incense about the same).

Temporal stability (static camera, 24 frames after 216, share of pixels changing by more than 24/255 per frame):
forest initial view 0.074 % → 0.117 % overall — inside the campfire's rectangle 0.11 → 0.42 % (the flames and
sparks move), outside it 0.070 → 0.083 %; cathedral 0.002 → 0.005 %; censer close-up 0.001 → 0.002 %; valley
0.066 → 0.014 % (run-to-run noise). Bit-exactness with the feature off: valley before vs. after with `'` off differs
by at most 5/255 in 6.3 % of pixels — the same as two runs of the old build (5/255, 5.4 %); the cathedral (which
gained a censer mesh) by ≤ 22/255 in 0.001 % of pixels. Browser (Chrome headless, WebGPU): no WGSL or validation
errors; forest, cathedral, valley and debug view 21 render as natively.

## Status vs. Unreal Engine 5 — new item

N. **Volumes and particles**: volumes are dense grids per emitter (no sparse / VDB volumes, no streaming of
   sequences, at most 8 emitters simulated and 4 marched per frame, each up to 192 steps per ray); the fluid solver
   is a single-grid red-black SOR (no multigrid), collocated (no MAC grid), with a solid floor only (no obstacles from
   the scene's geometry, no interaction with other emitters or particles) and a fixed time step clamped to 1/30 s.
   The lighting cache holds the sun's transmittance plus isotropic ambient and up to 8 local lights (no
   self-shadowing for local lights; multiple scattering is one approximate octave of the sun, not simulated), the
   march samples it trilinearly at half the grid resolution; close up, a 4 cm campfire grid shows its resolution in
   the smoke's edges. Volumes don't cast shadows on the scene or into the virtual shadow map, don't feed Lumen (only
   the fire's driven light does) and are not seen by reflections or refraction outside the screen. Overlapping
   volume boxes are composited box by box in entry order, not interleaved; particles go over the volume of their
   bucket (sparks inside the smoke are not attenuated by the smoke in front of them). Particles: weighted blended OIT
   approximates the order of layers at similar depths and saturates for very many opaque layers; one translucent
   surface (the nearest) splits the order — particles between two glass layers are composited with the farther
   bucket; collision only against the visible depth buffer (no off-screen or occluded collisions, no distance-field
   collision), no ribbons, meshes, sub-UV flipbooks, events or GPU sorting; lighting is per particle (no per-pixel
   normal-mapped lighting). Memory: 72 bytes per internal pixel while volumes and particles are on screen (volume
   march + history front/back, OIT targets, nearest translucent depth: 104 MiB at 1600×900).

## Controls

* Hotkey `'` (`Quote`): volumes + particles on / off (`RenderSettings::volumes`, `RenderSettings::particles`).
* Debug view **21** (`--view 21`, `V` cycle): particle overdraw (red → yellow, 1 per fragment) and volume march steps
  (blue → cyan) over a grey albedo.
* Demo: the forest's campfire in the clearing in front of the initial view (simulated fire and smoke, sparks that
  bounce, embers that settle and cool, a flickering fire light; best at night: `--sun 200,-30`, `--eye 3.5,1.4,1.0
  --at 7,1.0,-1`), dust motes in the cathedral nave's light shafts and incense rising from a censer before the altar
  (`--eye 66.2,1.8,0.4 --at 68.4,2.4,1.9`), mist over the valley lake (behind the translucency corner's glass panes).
