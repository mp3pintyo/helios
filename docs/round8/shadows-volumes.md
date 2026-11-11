# Prefiltered distant shadows and volume shadows (round 8 wave 2: `vsm-distant-prefiltered-moments`, `volume-shadows-lumen`)

Design note. The user-facing descriptions are in [`docs/architecture/shadows.md`](../architecture/shadows.md)
(*Prefiltered distant levels*) and [`docs/architecture/volumes-particles.md`](../architecture/volumes-particles.md)
(*Volume shadows*); what stays simpler than UE5 is in [`docs/status.md`](../status.md) items 7 and 11.

## 1. Prefiltered distant shadows (`vsm-distant-prefiltered-moments`)

### Problem

Wave 1 ([vsm.md](vsm.md)) gave the distant clipmap levels (from level 7) their own update rate under a moving sun —
the half of UE 5.8's *Prefiltered Distant* that saves re-rendering. The other half was missing: distant casters
still rendered at full resolution, so (a) every distant page that re-aims under `P` / `J` costs as much as before, a
whole stand of trees per page, and (b) thin distant geometry — needles, twigs, the canopy seen from above — is
sampled by a 12-tap comparison filter of binary depth tests: aliasing that crawls whenever a level re-renders.

### UE reference

UE 5.8's experimental *Prefiltered Distant* virtual shadow maps render the far clipmap levels at a lower resolution
into a prefiltered representation that is filtered linearly (no per-texel comparisons) and temporally reprojected;
5.9 adds their own update rate (wave 1).

### Design

* **Lower resolution.** `vsm_level` (vsm_common.wgsl), which marking and every sampler share, gives receivers whose
  level would be a distant one (`level ≥ Vsm::refresh.z`) a level `shadow_distant_bias` coarser (1: half the texel
  density, a quarter of the pages). It is an addressing change only: page size, rasterisers, culling and the page
  cache are untouched, and every sampler stays consistent (they all choose the level the same way).
* **Moment shadow map per page** (`vsm_moments.wgsl`). After the page rasters (also after the dynamic-caster
  refreshes), one 16×16 workgroup per physical page converts each *distant page rendered this frame*
  (`entries.w == frame`) into a 64² tile of an rgba16float atlas laid out like the depth atlas (tile *p* at
  (p mod 32, p div 32)): each texel the mean of the moments (z, z², z³, z⁴) of 2×2 depth texels (the box prefilter),
  plus mips 1 and 2 (32², 16²) built in the same workgroup. Cached pages keep their moments like their depth.
* **Per-page depth range.** Half precision over the scene's depth range along the sun (hundreds of metres) quantises
  depth to ~0.5 m: contact shadows of boulders would leak. Each page therefore normalises to its own range
  [zmin, zmax] of the texels that hold a caster (a workgroup reduction; empty texels count as farthest), stored in
  mip 6 of the same texture (one texel per tile, so no extra binding or buffer) after rounding outwards to half
  precision — the reader normalises the receiver with exactly the stored values.
* **Filtering** (`vsm_prefiltered.wgsl`, in the deferred lighting only). The filter width is the receiver's footprint
  (distance × pixel angle, stretched by up to 4× at grazing view angles), at least one moment texel; the mip is
  log₂(width / moment texel), blended between two mips (the shared linear sampler's mip filter is nearest). Four
  Vogel taps over 0.75 × width, rotated per pixel and frame, each kept on the receiver's plane (offset along the sun
  by −n·t / n·l, clamped); each tap is a separate page lookup (its own range), turned into a visibility by the
  **Hamburger 4MSM** (Peters and Klein 2015, algorithm 3: Cholesky solve of the 3×3 Hankel system, the two roots, the
  three cases) with a moment bias of 3·10⁻⁴ (rgba16f moments need ten times the 32-bit value) and a light bleeding
  reduction of 0.25. The receiver is lifted towards the sun by its slope (capped at 45°) times half a tap — how far
  its own surface's depth spreads over the area a tap averages — on top of the comparison filter's normal and sun
  offsets.
* Switching the feature or the bias invalidates the distant levels (their pages are rendered again).

### Alternatives considered

* **True half-resolution page rendering** (64² pages for distant levels). Every rasteriser (hardware, software,
  voxel, tessellation, clipped draws, page clear, static layers) and the culling's texel metric would need per-level
  page sizes. Addressing one level coarser gives the same texel density from the same pages with none of that.
* **EVSM** (exponential variance). Simpler, but in rgba16f the exponent is limited to ~5.5, and with a page's range
  of tens of metres the leaking between a boulder and the slope under it is far worse than with four moments.
* **A separate range buffer or texture.** Another storage buffer in the lighting layout (7 of 8 used) or another
  sampled texture (the layout is at 21 of 24 with this round's additions); mip 6 of the moment atlas costs nothing.
* **Moments in every sampler** (fog, particles, translucency, Lumen's cards). They sample a single comparison tap of a
  coarser level already; the moment path adds a texture binding to each layout, two of which are at the sampled-
  texture limit. Only the deferred lighting, where distant aliasing is seen, filters moments.
* **Temporal reprojection / accumulation of the moments** (UE). The build cannot read the tile it writes (WebGPU
  forbids sampling and storing the same subresource in a dispatch; read-write storage of rgba16f is not core), so it
  needs a second 43 MiB atlas or a copy pass per frame, and a re-aimed tile's history is in another light view and
  depth range (re-normalising four moments is a binomial transform; reprojecting needs the previous aim per level).
  Wave 1's levels stay exact for the light view they were rendered with, and the prefilter plus the footprint filter
  remove most of the crawl, so the accumulation's extra lag was not worth it. Left out.

## 2. Volume shadows (`volume-shadows-lumen`)

### Problem

Heterogeneous volumes (the campfire's smoke, the incense, the valley's lake mist) were lit (their lighting cache holds
the sun's transmittance through themselves) but cast no shadow: not on the scene, not into the fog, not on particles
or on each other; the fire's driven light shone through its own smoke column unattenuated.

### UE reference

UE's heterogeneous volumes cast shadows on the scene through an adaptive volumetric shadow map (AVSM: per texel of a
light-space map a few (depth, transmittance) nodes), built for the directional light and for local lights.

### Design

* **Transmittance atlas** (`volume_shadow.rs`, `volume_shadow.wgsl`). One 3D rgba8unorm texture; each emitter that is
  simulated this frame gets a region at its lighting cache's resolution (half the simulation grid; regions side by
  side along x, stable per emitter). Per voxel: r = the medium's transmittance from the voxel to the box's boundary
  towards the sun, g = towards the light the emitter drives, 16 steps through the density grid each. The driven
  light sits inside its fire: the hot gas is its source, not an occluder, so its march weights the soot by
  `1 − smoothstep(0.05, 0.3, temperature)` and stops at the light's source radius. The pass runs after the
  simulation, before the lighting caches.
* **Receivers** (`volume_shadow_sample.wgsl`): for each casting box (≤ 8), intersect the ray towards the sun; if it
  crosses the box, sample r where it enters (or at the receiver inside it). The transmittance stored there is the rest
  of the way through the box — one trilinear tap, valid for a receiver at any distance (a low sun's 12 m smoke shadow
  is as cheap as one under the fire). For the driven light, sample g where the segment from the light leaves the box
  towards the receiver. Used by: the deferred lighting (sun shadow incl. foliage back-lighting, the clustered light
  loop's driven light), the froxel fog (sun shafts and the fire light's in-scattering), the particles
  (`media_sun_visibility` / `media_clustered_lights`), and the volumes' lighting caches (the other boxes' shadows —
  each skips its own box, whose shadow its own 16-step march already has — and the driven light's self-shadowing).
* The opaque lighting and the fog run before the simulation in the frame: they see last frame's atlas (a frame late,
  like the fire light's flicker). A region is only sampled once it has been built.
* Bindings 151 (uniform: boxes and regions) and 152 (atlas): one more sampled texture in the lighting, fog and media
  layouts.

### Lumen (part c): not done, why

* The fire's glow on its surroundings already reaches Lumen through the light the simulation drives
  (`VolumeEmitter::light`, colour from the summed emission): Lumen's surface cache is lit by it and bounces it.
  Injecting the volume's emission into the radiance cache as well would count the same energy twice.
* What is missing is second-order: sunlight scattered by a bright smoke column onto its surroundings, and the smoke's
  occlusion of sky light / GI. A coarse injection (the volume's mean scattered radiance as a small emitter in the
  radiance cache's probes near the box) would need the probe update to know about volumes (another binding in
  `lumen_gather.wgsl`, which another wave-2 agent is changing) and would flicker with the simulation unless filtered
  over many frames. For a 1.6 m smoke column the effect is a few percent of the ambient light around it — not worth
  the instability. Documented as a gap (status item 11).

### Alternatives considered

* **Volumes into the virtual shadow map** (depth of a density threshold). Binary — no soft, partial smoke shadow —
  and it would invalidate every shadow page under the plume every frame (the simulation moves).
* **A sun-space adaptive volumetric shadow map** (UE). Handles any number of volumes in one map, but needs its own
  light-space raster of the volumes per frame and a node-merging pass; with at most 8 small boxes, per-box grids that
  reuse the lighting cache's resolution are cheaper and exact in the box's own space.
* **Sampling the lighting caches directly.** Their alpha holds the sun's transmittance *times* the VSM and cloud
  shadow at the voxel, which would double-shadow receivers, and a cache per emitter means a texture binding per
  emitter in every receiver layout; one atlas is one binding.
* **Shadows of every local light through every volume.** A transmittance per (light, volume) pair, or a march per
  pixel and light: only the light a fire drives is attenuated (its smoke is the case that matters).

## Measurements

RTX 3090, 1600×900, 240 frames, interleaved on/off runs (3 each). **The GPU was heavily shared during this session**
(other agents' renders and browsers: frame times of 40–75 ms instead of the usual 9 ms), so per-pass GPU times are
not reliable here; the page counts are exact and deterministic.

| Scene, view | pages requested, off → on | pages rendered / frame, off → on | notes |
|---|---|---|---|
| forest, `--animate-sun` (`P`), default view | 286 → 249 | 186 → 177 | distant pages are the expensive ones (whole stands of trees) |
| forest, 30 m above the canopy (`--eye 0,30,0 --at 200,0,-60`), `P` | 105 → 55 | 43 → 30 | half the pages in a distant-dominated view |
| forest, same view, static sun | 104 → 56 | — | lighting 0.50 → 0.38 ms, `vsm raster sw` 0.31 → 0.26 ms (single quiet run) |
| valley, default view | 123 → 92 | 1.6 → 1.4 | lighting 0.38 → 0.32 ms (single quiet run) |

* `vsm moments` (the build pass, 1024 workgroups, most exit at once): 0.005 ms with no distant page rendered,
  0.015–0.02 ms when a distant level re-aims under `P`.
* Image: valley default view on vs off differs by a mean |Δ| of 0.6 / 255 (distant shadow edges softer); the canopy
  from 30 m up by 2.8 / 255 (~1.7 % brighter: less fine self-shadowing at half resolution and moment light bleeding).
* Memory: the moment atlas 2048² rgba16f with 7 mips, 42.7 MiB, allocated the first time the feature is on.
* Volume shadows: `volume shadows` pass 0.007–0.013 ms (campfire 20×40×20 + incense/mist regions); the receivers'
  box tests are not measurable in the lighting / fog passes under this contention. Atlas: < 1 MiB.
* Campfire, default smoke: the shadow is faint (the smoke's optical depth along the sun is a few hundredths); with
  `--campfire-smoke 8 --sun 143,40` a plain 4 m smoke shadow on the clearing (`smoke-shadow-off-vs-on.jpg`).
* Browser (headless Chromium, WebGPU, forest): Tint compiles every shader, no WGSL / validation errors; `7` cycles the
  states (`browser-forest.jpg`).

Images: [`shadows-volumes/`](shadows-volumes/).

## Remaining limits

* Prefiltered distant shadows: in the deferred lighting only; a resolution step at the first distant level; light
  bleeding where layers of casters overlap in a filter (reduced, not removed: distant canopies come out ~2 % brighter
  than with the comparison filter); no temporal accumulation; 43 MiB of moments allocated once on.
* Volume shadows: 8-bit transmittance at half the simulation resolution; only the sun and each fire's own light;
  a frame late in the opaque lighting and fog; not on translucent surfaces, Lumen's surface cache or under MegaLights;
  emitters beyond 250 m cast none. Volumes still don't feed Lumen.
