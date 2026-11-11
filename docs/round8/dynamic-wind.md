# Dynamic wind: plant bones, parametric and simulated (round 8: `dynamic-wind-bones`)

Design note for backlog item **217** `dynamic-wind-bones`, round 8, third wave. The user-facing description is the
*Dynamic wind* paragraph of [`docs/architecture/motion.md`](../architecture/motion.md); what is still simpler than UE
is item 7 of [`docs/status.md`](../status.md). Images: `dynamic-wind/`.

## The problem

Helios' wind (`wind.wgsl`) was UE4-style procedural World Position Offset: every vertex moved by a function of its
rest position, its instance's origin and the time — a whole-plant *bend* (∝ height²), a branch *sway* (∝ distance from
the plant's axis) and a leaf *flutter* (a 3D wobble with a short spatial wavelength), scaled by a per-vertex wind
weight (round 8, wave 1) that holds roots. That has no notion of the plant's structure:

* leaves flutter about their own position, not their stem; a branch does not carry its twigs and leaves, every
  vertex just shifts by a smooth field, so nothing *rotates* — long blades shear sideways instead of bending, and
  normals never turn (lighting on a swaying fern stays frozen);
* every part of a plant obeys the same three terms; thick limbs and thin twigs move alike;
* voxel LODs moved rigidly at their centre with weight 1, assemblies (conifer needle sprays) as rigid copies.

## UE reference

UE 5.9's *Dynamic Wind* for Nanite foliage (experimental plugin): skeletal plant meshes whose bones are simulated on
the GPU — each bone a spring with **sway** (slow, drag-driven) and **flutter** (fast, vortex shedding), its mass and
wind-catching area derived from the mesh. Two tiers: instances share simulations until a force (radial, directional,
swirl, cone, event bursts), a **push capsule** or a velocity volume reaches them; then they are **promoted** to a
simulation of their own (a budget per mesh, 256 active instances by default). `DynamicWind.Mode 0` is a stateless,
parametric mode for low-end hardware. Each bone is a spring — there is no constraint solver between bones.

## Design

### 1. Bones from the generators (`helios_geometry::plant_bones`, `vegetation.rs`)

The generators know each plant's structure, so they emit it:

| plant | bones (class) |
|---|---|
| broadleaf tree, bush | trunk (bend), branches of depth 1 and 2 (sway; deeper twigs ride on their depth-2 ancestor), one per leaf (flutter) |
| conifer | trunk (bend), one per whorl branch (sway), one per needle blade (flutter) |
| conifer / broadleaf assembly | the wood's bones; each spray's leaves are bones without a parent, hanging from the part's plant bone |
| fern | one per frond, root below the ground to the tip (bend; fronds of one fern a little out of step), one per leaflet (flutter) |
| grass clump | one per blade (bend, each with its own phase; longer blades slower) |

A bone is a pivot, a unit axis, a length, a parent and its *attachment* (distance along the parent where it hangs),
a class, an amplitude (rotation per unit of the material's response) and a frequency fraction. Amplitudes:
stems `2·L` per unit *bend* (an arc of angle θ moves its tip by θL/2 — the procedural cantilever's `bend·h²`); leaves
`2.5 / length` per unit *flutter* (their tip moves 2.5× the response, about 5–15°); branches from their **mechanics**
— a cantilever loaded by the wind: stiffness ∝ r⁴/L³, mass ∝ r²L plus 3·10⁻⁴ × the leaf area it carries, drag area
2rL + that leaf area; the static deflection (drag / stiffness / L) relative to the plant's median sets the amplitude
(compressed: 0.5–2.5 rad per unit sway), the natural frequency √(k/m) relative to the fastest branch the frequency
(0.4–1). Thin, bare twigs whip quickly; heavy, leafy limbs swing slowly and far.

Every vertex is bound rigidly to one bone through the **skin attribute** (`Mesh::skin`, joint = bone): the cluster
encoding (joint minimum + offset bits per cluster), the DAG's simplification and pruning copies and the crack-free
LOD already carry it, so no new vertex format was needed — a leaf's 10 vertices share a bone, clusters cover runs of
consecutive leaf bones (a few bits a vertex). The table (`Mesh::bones` → `VirtualMesh::bones`) goes into the `.hvm`:
**version 6** adds a seventh section count and a bone section after the statistics (13 words a bone plus 6 words of
bounds); versions 4 and 5 still load (`plant_bones_round_trip_and_version_5_loads`). Wood and foliage of one plant
share the branch bones (same indices), so they move as one; meshes with bones get voxel LODs again (skinned meshes
don't).

### 2. Parametric mode (UE's `Mode 0`): `plant_wind.wgsl`

Per vertex, in every pass that places geometry — hardware and software visibility rasterisers, hardware and software
shadow-page rasterisers, the G-buffer resolve (corners, normals, motion vectors), the voxel rasterisers and resolve:

* the plant's **air** in the instance's object space: wind direction, side and up, the gust at the plant's origin,
  its phase and time (shared by all its bones);
* each bone's **rotation vector**, closed form, with the procedural layers' time functions by class: *bend* leans with
  the gust and sways at the plant's frequency (`cross(axis, wind·lean + side·lateral)`), *sway* is pushed downwind and
  bobs mostly up and down at up to 2.3× that frequency, *flutter* turns about two axes across the leaf and twists
  about its stem at ~2.6 Hz — each scaled by the bone's amplitude, the material's response and the instance's
  strength;
* the **chain**: the vertex's own bone bends as a **circular arc** — its frame at distance s along it turns by θ·s/L
  about the bone's rotation axis while the axis point follows `∫₀ˢ R(ωu/L)·a du` (closed form with sinc terms) — and
  each parent's frame at the attachment carries the result, down to the root. Arcs and rotations keep lengths
  (`roots_hold_and_blades_bend` checks edges change < 3 % at 0.2 rad), roots hold (s = 0 is the identity), leaves turn
  about their stems, normals turn with their bone.

All of it is a pure function of the rest vertex, its bone, the instance and the time, so passes, clusters and LOD
levels sharing a vertex put it on the same bits, as before. The tables live in one `Rgba32Float` texture read with
`textureLoad` (binding **170** (150 before the merge with the river): no stage gains a storage buffer): per bone three texels, per (mesh, material) a header
texel with the material's response, per assembly part its own table (`part_bone_table`: the spray's leaves, then the
chain of plant bones it hangs from, transformed into the part's space — the parts are drawn with their own transforms,
the base's is not available to them). `GpuInstance` grew from 240 to 256 bytes: `plant` = (header + 1, table, sim
slot + 1, –); the assembly expansion gives each part record its table (`AssemblyPart::bones`, a former padding word).

**Bounds without touching culling.** Frustum, HZB and shadow-page culling, page expansion and shadow invalidation use
`wind_reach` / `wind_speed` — the three procedural layers' bounds per sphere (`bend·h² + sway·ρ + flutter`, and their
time derivatives with each layer's rate). Rather than teach all of them a new bound, an instance with bones carries in
`wind` the procedural layers that *bound its bones*: per mesh at bake time, every vertex's displacement per unit
response of each class is summed over its chain (a rigid frame is an isometry, so the levels' displacements simply
add; each level moves a point by ≤ θ·(s²/2L + s/L·|offset from the axis point|)), and `BoneBounds::fit` finds
coefficients of h², ρ and a constant holding them — the bend class may spill into the ρ and constant terms, sway into
the constant, never into a slower layer, so the speed bounds (which multiply each layer by its rate) hold too, since the
bones use the same time functions. `equivalent_wind` turns them into world units for the instance's scale. A CPU
mirror of the shader (`bone_motion_stays_within_the_culling_bounds`) checks displacement and speed of every 7th vertex
of grass, fern, tree wood and foliage, bush and conifer foliage at random strengths up to 2, scales 0.6–1.6, times and
both turbulence extremes. The parallel Nanite-culling work stays merge-safe: no culling shader changed.

**Distance LOD.** Leaf flutter fades out where a leaf tip's motion would cover under a pixel (2.5 × the flutter
response at `WIND_MIN_PIXELS` / 4: ~20 m), branch sway where a branch 0.4 × the plant's radius long, turning by its
sway response, would (~35 m for an 18 m tree). A faded level is exactly the identity, so it is skipped — only its parent link (two texels) is read. The
fade depends on the instance and the camera only, so all passes and views agree, and it is continuous (no popping).

**G-buffer resolve.** Per pixel, the three corners: when they share a bone (a leaf, a blade, a branch) the parents'
frames are evaluated once and applied to all three (`plant_pose_corners`; only the own bone's arc is per corner);
motion vectors evaluate the previous pose once, at the pixel's interpolated rest position.

**Shadow pages.** The hardware page rasteriser poses every corner of every (cluster, page) draw — 384 chain walks per
cluster and page, where the software one poses each vertex once into shared memory. Culling for shadow views sends a
plant's clusters to the software path up to 2.5× the usual triangle size (and up to 96 instead of 48 texels of radius):
in the forest's default view the page rasterisers' cost with bones went from 0.99 to 0.64 ms (bones off: 0.60).

### 3. Simulated tier (UE's promoted instances): `plant_sim.rs`, `plant_sim.wgsl`

Plants with at most 512 bones (grass, ferns, bushes) are **promoted** when a force reaches them — the **push capsule**
that follows the camera (from the eye 2.2 m down, 0.45 m thick: a walker's legs and body) touches their bounds, or a
**gust burst**'s ring passes them (`Numpad1`, or `Renderer::wind_gust`: a ring travelling out at 7 m/s from 5 m ahead of
the camera for 3.5 s) — into one of **256** slots, and stay promoted until 2 s after the force last reached them (their
springs have settled). A compute pass (`plant sim`, one workgroup per slot) keeps per non-leaf bone a spring state: its
**deviation δ** from the parametric rotation, driven towards the contact target (the bone's outer part pushed out of
the capsule horizontally) and the gust target (bent away from the ring's centre), stiffer while touched, otherwise back
to 0 with a little overshoot (damping ratio 0.3, its frequency from the bone's, four semi-implicit Euler steps a frame).
The rasterisers add δ to the parametric rotation (a second texture, binding **171** (was 151), Rgba16Float, two texels a bone:
this frame's and the previous frame's for motion vectors). An undisturbed promoted plant therefore moves exactly like an
unpromoted one: **promotion and demotion never pop**. δ stays within 3× the bone's parametric bound and its rate within
3× the bone's speed bound, and a promoted instance's layers are scaled by 4: culling and shadow invalidation stay
valid. Leaves keep their parametric flutter; as in UE, each bone is a spring of its own.

### 4. Voxel LODs and assemblies

Voxel clusters of a plant carry a bone per brick — the bone with the most area in the brick, mapped to its nearest
non-leaf ancestor (`voxel_bone`: leaf flutter is far below a voxel): header word 4 bit 31 flags it, 16 bits a brick
follow the voxels' words (`voxel_brick_bone`; clusters without the flag decode as before). Rasterisers and the resolve
pose each brick's centre once (per workgroup in the rasterisers) and move its voxels rigidly with it, so far voxels
follow the same branches as the triangles they replace: the LOD switch shows no jump in position (only the leaves'
flutter is dropped). Assembly parts hang from their plant bone (their own table); the assembly's base (its far
representation) binds every part's root geometry to that bone, and its bounds are fitted to all parts at full detail
including their leaves, since the parts are culled with the base's wind.

## Alternatives considered

* **Per-instance bone matrices computed per frame** (a palette like `skin.rs`, then plain skinning): exact and cheap
  per vertex, but the forest has ~16 000 plant instances and up to ~9 000 bones a tree (leaves) — tens of millions of
  matrices a frame. Only the simulated tier keeps per-bone state, for a budget of instances.
* **Rigid bones** (each bone tilts as a stick, curvature only from more bones): grass blades and fronds would need
  several bones each and still kink; the arc is ~10 extra ALU a level.
* **A new bound in every culling shader**: invasive (cull, page expansion, invalidation, assembly records) and in
  conflict with the parallel culling work; the fitted equivalent layers keep them untouched.
* **Pivot-painter textures** (UE4-style pivots and axes per element addressed by UV): the same data as bones but no
  hierarchy beyond two levels and a UV channel spent on it.
* **Simulating every plant near the camera** (a first version promoted all within 9 m): it inflated their shadow
  bounds — promoted bounds also grow — and refreshed their pages every frame; promotion by force keeps the simulated
  set to the plants actually touched.

## Measurements

RTX 3090 shared with other agents' renders (several runs per configuration were visibly contended: lighting, which
the wind doesn't touch, ran 2–3× slow): per-pass **minima** of 3–4 interleaved runs per configuration, forest at high
detail, 1600×900, 300 frames; *main* is the round-8 wave-2 build (75b098f), *off* this branch with
`--no-wind-bones` (the same plants, bone data in their clusters, moved by the procedural wind), *on* the bones.

| Pass (ms) | default view: main → off → on | fern close-up (`--eye 5.2,0.7,1.2 --at 6.2,0.0,0.2`) | grass close-up (`--eye 4.6,0.4,-1.6 --at 5.7,0.12,-2.7`) |
|---|---|---|---|
| GPU frame (minimum) | 8.45 → 8.91 → 9.40 | 6.60 → 7.06 → 7.58 | 8.48 → 8.71 → 9.41 |
| raster hw / sw + merge | 0.224 / 2.088 → 0.223 / 2.165 → 0.275 / 2.281 | 0.082 / 0.611 → 0.083 / 0.635 → 0.102 / 0.677 | 0.236 / 2.196 → 0.236 / 2.277 → 0.304 / 2.400 |
| material classify + resolve | 0.496 → 0.560 → 0.651 | 0.436 → 0.492 → 0.555 | 0.501 → 0.571 → 0.659 |
| vsm raster sw / hw | 0.206 / 0.353 → 0.245 / 0.439 → 0.445 / 0.275 | 0.056 / 0.377 → 0.058 / 0.469 → 0.071 / 0.647 | 0.021 / 0.198 → 0.025 / 0.241 → 0.052 / 0.394 |
| (cluster, page) shadow draws | 41.3 k → 41.4 k → 55.1 k | 35.0 k → 34.9 k → 45.9 k | 31.6 k → 31.7 k → 42.5 k |
| plant sim | 0.007–0.010 (a few plants under the capsule) | 0.010–0.012 | – (nothing touched) |

* **Bones on vs off**: +0.5 ms of GPU frame in the default and fern views, +0.7 ms with grass filling the view: +0.05 ms hardware and +0.12 software
  visibility raster, +0.06–0.09 ms resolve, shadow pages +0.04 ms (default; the software page path took over most
  plant clusters) to +0.18 ms (ferns filling the shadow pages near the camera, whose fronds stay on the hardware path).
  The fitted bounds give ~30 % more (cluster, page) draws; the same run with the procedural bounds (wrong, for the
  experiment) had 47 k instead of 55 k draws and the same page raster time within noise — the cost is the per-vertex
  chain, not the padding.
* **Off vs main**: +0.45 ms although nothing evaluates bones: the plant meshes carry a skin (bone index) per vertex
  (a few to ~10 bits), voxel bricks their bone words, `GpuInstance` 16 more bytes, and the resolve and rasterisers more
  code (registers). A scene without plants pays nothing measurable (cathedral: every pass within ±0.02 ms of main).
* **Steps measured on the way** (default view, on): the first version cost +1.06 ms over main (resolve 0.89 ms,
  hardware page raster 0.65 ms); sharing the parent chain across a triangle's corners and posing the previous frame once
  per pixel took the resolve to 0.64 ms; sending plant clusters to the software page path in shadow views took the page
  rasterisers from 0.99 to 0.64 ms. A cheaper arc (one transcendental less per level) and fading leaves and branches
  at 2 instead of 1 pixel were within noise, so the 1-pixel fade stayed.
* **Simulated tier**: the `plant sim` pass costs ~0.01 ms with a handful of promoted plants; nothing is promoted until
  a force reaches a plant.
* **Bake**: fitting `BoneBounds` (an (h, ρ) envelope and nested ternary searches) and the bone tables add little: the
  forest's scene build took 8.6–9.1 s on main and 8.8–9.8 s here in the uncontended runs (CPU shared too).
* **Memory**: the bone tables 2.24 MiB (147 k texels) for the forest's meshes and 353 assembly parts; the simulation
  texture 1 MiB (256 slots × 1024 Rgba16Float texels) and its state buffer 4 MiB.
* **Browser** (headless Chromium, WebGPU, `?scene=forest&detail=medium`): Tint compiles every changed shader, no
  console errors; `Numpad1` starts a gust and the `plant sim` pass runs.
* **Tests**: `bone_motion_stays_within_the_culling_bounds`, `roots_hold_and_blades_bend` (CPU mirror of the shader),
  `plant_bones_round_trip_and_version_5_loads`, `bones_round_trip_and_chain_reaches_add_up`,
  `fitted_bounds_hold_every_point`, `part_tables_hang_the_part_on_its_bone`; the workspace's 354 tests pass.
* **Images** (`dynamic-wind/`): `grass-procedural-vs-bones.jpg` and `fern-procedural-vs-bones.jpg` (frames 1/6 s
  apart and the per-pixel change over 0.5 s: blades curve and swing individually, leaflets turn about the rachis and
  their shading changes with them — the fern view's change image also holds the campfire's flicker on the ground; motion blur, on by default, follows the bones' exact motion vectors), `walk-through-sim-off-on.jpg` (walking into a grass tuft at 1 m/s: the push capsule parts
  it), `gust-burst.jpg` (a gust burst over the clearing, `--eye 3.0,2.6,2.5 --at 7.0,0.0,-2.0 --keys Numpad1 --keys-at 100`,
  and its per-pixel difference from the same frames without it — the frames are deterministic: the ferns and bushes bend
  as the ring passes at 0.25–0.6 s and spring back; up to 25 000 pixels change by more than 24/255), `far-motion.jpg`
  (the default view, its voxel-cluster debug view 22 and the change over 0.5 s: the distant conifers' needle bases and
  voxels sway with their branches; a visual check, no per-pixel LOD-switch measurement).

## Remaining limits

* Only the procedural vegetation has bones: imported glTF foliage keeps the procedural wind (no skeletal plant
  import; UE's Dynamic Wind takes skeletal meshes authored with bones).
* A vertex follows one bone, rigidly (no blending at joints); twigs deeper than two branch levels ride on their
  depth-2 ancestor; assembly sprays hang rigidly on their bone's frame apart from their own leaves.
* The parametric rotations reuse the procedural layers' time functions — no drag, no vortex shedding, no per-bone wind
  sampling (a gust is one travelling wave per plant) — so the bounds of culling stay valid without new code; UE's
  sway/flutter are physically derived.
* The simulated tier knows two forces: the capsule following the camera and a radial gust burst (UE: radial,
  directional, swirl, cone forces, event bursts, velocity volumes; push capsules on any actor). It simulates plants of
  at most 512 bones (grass, ferns, bushes — not trees), springs only non-leaf bones (leaves keep their flutter), and
  a spring per bone without constraints between them (as UE). The promoted instances' bounds are scaled by 4 while
  promoted (their shadow pages refresh more often while something pushes them).
* Voxel LODs move rigidly per brick with the branch most of its leaves hang from: no flutter, and a brick straddling
  two branches follows one.
* Every pass evaluates the bone chain per vertex (UE evaluates bones once per instance and skins): ~0.3–0.5 ms in the
  forest at 1600×900. The fitted bounds are looser than the procedural ones (a trunk's rotation carries the crown
  sideways, an h·ρ term the h²/ρ basis covers loosely): ~30 % more (cluster, page) shadow draws.
* The wind weight of round 8's grounding is not used for plants with bones (their pivots hold the roots); tessellated
  surfaces keep the procedural wind.
