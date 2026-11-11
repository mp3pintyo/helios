# Round 8 — natural cliff rock on the mesh terrain, a clearer river

Backlog item 235 `terrain-rock-material` (fix). How the result works is described in
[architecture/terrain.md](../architecture/terrain.md) (Shading, Simulation, Surface); this note is the
why, the alternatives and the numbers.

## Problem

Looking at the valley's new mesh terrain, the user found that "the big rock the water flows through
has a rather strange texture". The cliff band with the cave, the overhanging shelf and the arch were
shaded with the landscape's own layers, projected triplanar; their rock layer is the landscape's
`TexturePattern::Rock`, whose height and colour are ridged noise *plus a Voronoi crack network*
(7 cells per tile). On the mountains it is far away and sampled through the runtime virtual texture;
on the mesh terrain it tiles every 8.3 m over walls 13 m high, at one scale, so the faces showed a
uniform net of polygonal cells — crazy paving, or a cracked glaze — with mossy streaks, nothing like
rock (`rockmat/closeups-before-after.jpg`, left column). The river had a second problem: below the
cave it was a white sheet of foam (`rockmat/cave-river-before-after.jpg`), and the foam's streaks
were broad blotches.

## UE reference

UE landscapes and Megascans cliffs are layered materials: a large-scale macro texture and a detail
texture at different tilings (often with macro variation and distance blending to hide repetition),
slope- and normal-driven layers (moss and grass on up-facing surfaces, dirt in cavities from baked
curvature / AO), and world-aligned (triplanar) projection for cliff meshes. Strata usually come from
the sculpt or the scan. UE's water uses scattering and absorption coefficients by depth, Fresnel
reflections, and foam from the water simulation's velocity and the shoreline distance field.

## Design

**Cliff texture** (`texgen.wgsl` `cliff`, `TexturePattern::Cliff`, procedural like every other
material texture): what makes a cliff face read as rock is a mix of scales without a closed network
of lines — rounded masses (fBm flattened along the bedding), faint lamination (ridged noise),
**broad fracture planes and small chips as tilted planar facets** (each Voronoi cell gets a random
plane; neighbouring cells meet at creases of different light, not at drawn lines), fractures drawn
along only some of the large cells' borders (a noise mask fades them in and out, so no polygon ever
closes), dark streaks running down the face, and fine grain. It is made for side projections (v up).
The mesh terrain's material takes it as its rock layer (`valley_terrain::material`); the landscape
and its mountains keep the old layer.

**Two scales, one texture** (`rvt.wgsl` `cliff_rock`): 12 m tiles for masses, planes and streaks,
2.7 m tiles for chips and grain (ratio 4.46: the two never repeat together); the detail modulates the
large scale's colour through its height (only the detail's data texels are fetched; the height's
mean is about 0.5 and its mips filter towards it where the detail is sub-pixel) and adds its normals;
it is skipped altogether where a pixel covers more than 1/32 of its tile. The detail projection is turned 53° about the vertical:
walls keep their bedding horizontal, but on ledges and undersides (the top projection) its grain
crosses the large scale's instead of doubling it into parallel smears. `triplanar_sparse` skips
projections under 2 % weight, which on walls leaves one or two of three; the two scales fetch fewer
texels than the one full triplanar sample before.

**Strata from the geometry.** The ledges are geometry: `rock_relief` adds `0.25 · sin³(2.2 · q.y)`
to each rock modifier's distance, with `q` the domain-warped position. Colour bands that don't sit on
those ledges would look painted on, and the shader can't recompute the CPU warp (different noise).
So `MeshTerrain::shading_attributes` writes per vertex, into the UVs the triplanar material doesn't
use, the **strata coordinate** `q.y · 2.2 / π` (ledges at its integers; odd beds protrude — hard —,
even ones are recessed — soft; blended between rocks by distance, plain height away from them, so
it is continuous across modifier bounds) and a **cavity** (density probed 0.5 and 1.5 m out along the
normal: 0 open, up to 1 in a crevice; only near volume modifiers — on the heightfield ground it is
0). Shared border vertices of neighbouring chunks get bit-identical values (same position, same
normal), so there are no seams; a test checks it. The shader gives each bed its own tone (±13 %,
warmer or greyer) and the hard beds 7 % more light and less roughness, draws fine laminae within a
bed (faded by the coordinate's screen derivative once sub-pixel) and a dark parting at bedding planes
— open on some planes (a hash per plane) and coming and going along each (noise and the texture's
height).

**Weathering.** Dirt (the dirt layer's mean colour, modulated by the rock's detail) where the cavity
is high and the surface faces up or sideways; moss and lichen in the boulders' moss colour on
up-facing rock, more in hollows and on soft beds; steep faces stay clean and undersides bare and a
little darker. Grass and dirt still cover the level tops through the landscape's slope weights.

**River foam** (`river_sim.wgsl` `depth_step`): foam used to be born wherever the speed exceeded
0.9 m/s (and saturated: 0.6 per second × 5 s life) and wherever the surface sloped, so the whole
fast stretch below the cave turned white. It is now born from **turbulence**: convergence
(negative divergence of the velocity: the pool under the step, the bow of a boulder), shear (curl:
the wake behind a boulder), fast water (> 1.5 m/s) breaking over a steep surface, and a little for
very fast water (> 2.5 m/s); dry neighbours count as the cell's own velocity, so the banks neither
converge nor shear. It fades over 3 s instead of 5. The water shader draws it as a **lace of thin
lines** — the zero crossings of the flow-advected noise, thickening with the amount — with patches
only where it is thick, and the lake's shore foam keeps only its thin line on the river.

**River reflections.** Running water is rougher than its resolved (normal-mapped) waves:
`roughness² += 0.03 · speed²`, so fast stretches blur the sky they reflect while pools stay mirrors;
Fresnel, absorption (clear mountain water: 0.38 / 0.05 / 0.035 per metre) and the traced reflections
are the lake's. With the foam gone the bed showed the lake's sand ripples, drawn across the lake's
wave direction as regular combs; `shore_surface` now leaves them out where the river's level is the
higher (its bed is gravel). In the shallows the bed shows through; where the river runs in the shade
of its left bank, at a grazing angle, it reflects the open sky — that is Fresnel (≈ 20–35 % at
10–20°), not an error, and it now reads as rippled water rather than a mirror of the clouds.

## Alternatives considered

* *A scanned / hand-painted rock bitmap.* Better micro detail, but every other material in Helios is
  procedural (texgen) and the browser build would have to download it; the task asked to extend the
  generator.
* *Fully procedural 3D rock in the shader* (the boulders' `evaluate_rock`: 5-octave fBm with a
  4-sample gradient, plus mottling). Seamless and never repeating, but ~20 noise evaluations per
  pixel on screen-filling walls would have cost the material resolve more than the texture fetches
  do; the boulders can afford it because they are small on screen.
* *Recomputing the strata in the shader* from world height and a GPU noise warp: cheap, but the
  bands would not coincide with the geometric ledges (the CPU and GPU noises differ), which is the
  thing that makes strata convincing. A per-vertex attribute is free at run time and exact.
* *Stochastic (hex-tiling) texture sampling* to kill repetition: three samples per projection per
  scale; the two incommensurate scales plus geometry-driven colour already hide the period at these
  distances.
* *Curvature from the mesh* (per-vertex mean curvature) for dirt: noisy on surface-nets meshes at
  0.25 m; the density probe measures enclosure at the scale that matters (0.5–1.5 m).
* *Darkening the river's reflections or lowering its Fresnel*: would make it inconsistent with the
  lake and wrong at grazing angles; rougher running water is the physical reason streams don't
  mirror the sky.

## Measurements

RTX 3090 shared with other agents' builds and renders; 1600×900, 240 frames; minima over the last two
thirds of each run, three interleaved runs of the main-branch build (`before`) and this branch
(`after`), per-pass GPU minima from the profiler; the lowest of the three is given (the spread
between runs was 0.01–0.02 ms per pass).

| View | Frame (before → after) | Material classify + resolve | Translucency | River simulation |
| --- | --- | --- | --- | --- |
| Initial view | 5.55 → 5.53 ms | 0.480 → 0.482 ms | 0.654 → 0.655 ms | 0.041 → 0.042 ms |
| River `--eye -2,3,-6 --at -9,0,-20` | 5.77 → 5.78 ms | 0.649 → 0.712 ms | 0.918 → 0.870 ms | 0.042 → 0.042 ms |
| Overhang `--eye 7,1.8,-25 --at -3.9,-3,-38.9` | 3.74 → 3.75 ms | 0.335 → 0.438 ms | 0.071 → 0.072 ms | 0.041 → 0.041 ms |
| Arch `--eye 20.7,1.8,-9.8 --at 34,-5.5,-22` | 3.92 → 4.01 ms | 0.439 → 0.538 ms | — | 0.041 → 0.041 ms |

Frame times are flat in the initial and the river view and up to 0.1 ms where the cliff fills the
screen. The cliff shading itself is cheap — sparse projections, the detail scale without colour
fetches: with the UV attributes replaced by constants the overhang's resolve measured 0.351 ms
(+0.016 over before). The rest (~0.085 ms on a screen of rock) is decoding the vertices' UVs for the
mesh terrain's pixels: the resolve used to drop the UV decode for the whole Terrain pipeline, since
nothing read it. A first version decoded them for the landscape's pixels too (+0.013 ms in the
initial view); now only world-mapped Terrain materials fetch them. The river's lighter foam and
rougher reflections made its translucency pass 0.05 ms cheaper in its close-up.

Building: the mesh terrain's surface nets plus the attributes took 2.31 s against 2.23 s (one run
each, shared machine: +4 %, the cavity's two density probes per vertex near rock); one more 512²
texture is generated (9 instead of 8 in the valley).

Browser (Chromium headless, WebGPU, D3D11): no WGSL or validation errors, the valley renders the cliff
rock and the clearer river as natively (`rockmat/browser-chromium.jpg`). Tests: `cargo test --workspace
--release` 362 passed (new: `shading_attributes_follow_the_strata_and_find_crevices`).

## Images

* `rockmat/default-before-after.jpg` — the initial view (top) and the same at a low sun (bottom),
  before left, after right.
* `rockmat/closeups-before-after.jpg` — the overhang (`--eye 7,1.8,-25 --at -3.9,-3,-38.9`), the
  arch (`--eye 20.7,1.8,-9.8 --at 34,-5.5,-22`) and the shelf's underside from its foot
  (`--eye -1,2,-33 --at -2,4,-42`).
* `rockmat/cave-river-before-after.jpg` — the cave (`--eye -14,2.2,-28 --at -12.5,1.8,-40`) and the
  river (`--eye -2,3,-6 --at -9,0,-20`).
* `rockmat/low-sun-before-after.jpg` — the cave and the arch at a low sun (`--sun 1.3,0.12`,
  `--sun 2.2,0.12`).

## Remaining limits

See `docs/status.md` item 18: one procedural texture at two scales (no scanned materials, no
stochastic tiling), strata only as fine as the 0.25 m vertices (finer laminae are the texture's),
weathering from the normal and a cavity probe (no erosion or deposition simulation), no displacement
of the cliff faces; foam is a heuristic of divergence, shear, slope and speed (no aeration model, no
spray at the step). The overhang's underside takes a strong green from the sunlit meadow under it —
that is Lumen's bounce light, as before.
