# Lumen follows displacement (round 6: `lumen-displacement`)

Design note. The README carries the user-facing text (the *Lumen-style GI* bullet's *Displaced surfaces*
paragraph, the *Nanite Tessellation* bullet and gap 14 of "Status vs. Unreal Engine 5"); this file records why it
is built the way it is, and the measurements.

## What

Nanite tessellation displaces surfaces at render time — texture heights (the valley's cobblestone path and
boulders), graph heights (the gallery's brick terrace), graph vectors (the mushrooms) and animated graphs (the
rippling cloth). Until now Lumen's geometric representations — the per-mesh distance fields, the mesh cards and,
built from those, the global distance field, the near field and the voxel and card surface caches — were baked
from the undisplaced meshes; only screen traces saw the relief.

Now every instance whose material displaces its mesh is represented in Lumen by a **displaced variant** of the
mesh: a distance field and a set of mesh cards baked from the displaced surface. Everything downstream follows
without changes of its own: the global field composes the variant's field, the near field packs it into its
atlas, hits project into the variant's cards, the voxel surface cache lights what the global field holds.

* **Static displacement** (texture or graph, height or vector) is baked as it is rendered, accurate to the
  variant's voxels (the mesh's own voxel size) and the cards' texels.
* **Animated displacement** (graphs that read the time) is represented by its **mean surface**: the displacement
  averaged over 32 times in 8 s.
* **Scenes and meshes without displacement** request no variant; Lumen uploads the meshes' own fields exactly as
  before (same order, same captures).

## Why per (mesh, displacement), and when per instance

Displacement is a property of the material, not of the mesh: the gallery's 3 m `bed` mesh grows into mushrooms
with one material and ripples as cloth with another. A variant is therefore keyed by a *slot* — (mesh, material,
placement) — and a *content* hash of everything the displacement reads (the material's `Displacement`, its
texture, mapping and UV transform, the graph's displacement code and parameters, the texture version).

*Placement* is the instance transform, and only for displacements that depend on it: world-triplanar texture
heights, graphs that read `WorldPosition` / its derivatives or the world `VertexNormal` (found in the compiled
displacement code: `input.world_position`, `input.dpdx`, `input.dpdy`, `input.normal`), and every vector
displacement (world-space vectors become object offsets through the instance's normal matrix). Object-space ones —
mesh-UV and object-triplanar heights, the valley's boulders — share one variant among all instances of the mesh
with that material. All three gallery graphs read the world position, so each bed and the terrace has its own.

A content change keeps the slot's old variant in use until the new one is baked (an edited parameter does not
make the surface pop back to undisplaced), and at most one bake per slot runs at a time (a dragged slider rebakes
at the rate bakes finish, not every frame). With tessellation off (`PageDown`) instances get their meshes'
own fields at once; the variants stay cached for turning it back on.

## How a variant is baked

1. **Refine** (CPU, worker thread): the mesh's triangles — a LOD cut finer than the voxel, from the pages in
   memory — are split by a uniform barycentric grid into pieces at most ¾ of a voxel long. Segments per edge are
   rounded up to a power of two, so where a large triangle meets a small one the coarser side's edge points are a
   subset of the finer side's; points are computed from the edge's canonical end and welded by their attribute
   bits, so shared edges produce the same points (the finer side's extra points leave sub-voxel T-junctions,
   which the field bake tolerates like any hairline crack). A cap of 512 k triangles grows the spacing when needed.
2. **Displace** (GPU, one dispatch per variant): `lumen_displace.wgsl` moves every point with
   `displacement_position(instance, displacement_input(...))` — the function the dicer calls per micro-triangle
   vertex — through a scene instance that has the variant (its transform, scale and material). For graph
   displacement the shader is composed exactly as the dicing pipelines are: `displacement_graphs_off.wgsl`
   replaced by `graphs::scene_dispatch_module(scene, &graphs::displacing_graphs(scene))`, cached by
   `graphs::set_hash`. The job's own copy of the tessellation uniform carries the scene's material table (copied
   on the GPU from `GpuScene::tessellation`, so graph sources and parameter offsets are the ones the dicer reads)
   behind a header that makes `displacement_segment` the point spacing everywhere: textures and noise are filtered
   for the points, not for a camera. Animated graphs are evaluated at 32 times and averaged; w returns the RMS
   distance of the moving surface from the mean (computed from offsets to the first sample, so large coordinates do
   not cancel it).
3. **Bake** (CPU, worker thread): the positions are read back and the field baked with the code every mesh uses —
   `MeshSdf::bake_volume` or `bake_heightfield` (the mesh's own kind) with a resolution that keeps the mesh's voxel
   size over the larger displaced box; the cards later capture the displaced triangles (area-weighted normals of
   the welded surface) like any mesh's LOD cut. Heightfield variants keep no triangles (they get no cards).

Variants that finish together are uploaded together: Lumen re-uploads its scene once nothing is in flight, so a
scene's variants arrive in one rebuild. In the browser, which has no threads, refinements and bakes run one per
frame.

### Why evaluate on the GPU and bake on the CPU

* **The GPU evaluates the displacement** because only the GPU can: a graph's displacement is WGSL generated by the
  material compiler, run by the tessellator through a generated dispatch module. Re-implementing the graph
  semantics (noise, texture filtering, every node) on the CPU would be a second compiler that drifts from the
  first; evaluating the very function the dicer calls is exact by construction — texture heights, graph heights
  and vectors alike — and costs one small dispatch.
* **The CPU bakes the field and cards** because that code exists, is tested (sign flood fill, crack sealing,
  two-sided shells, depth-peeled cards) and is what every other mesh's representation comes from: a displaced
  variant is then exactly what an undisplaced mesh of that shape would bake to. A GPU field bake would have to
  re-solve the sign problem the CPU bake solves with a flood fill. The price is latency (0.5–1.3 s on a worker
  thread per variant) and the read-back, both one-off.

### Rejected

* **Displacing in the trace** (offsetting distance samples by the displacement at the closest undisplaced point):
  only exact for small heights along the normal, wrong for vectors and overhangs, and every distance-field sample
  in every Lumen pass would pay a displacement evaluation.
* **Padding the undisplaced field by the displacement bound**: a conservative shell blocks light that real relief
  lets through (mortar joints, the space under the mushroom caps).
* **Baking variants at scene build** (in `helios-geometry` with the mesh): the build does not know which materials
  a mesh is rendered with, and graph displacement exists only as GPU code.

## Animated displacement: the mean surface

The options were a rest pose (one time), the mean over time, a bound of the motion (union of all poses) and a
periodic re-bake. Measured on the gallery's cloth (the bake logs the RMS distance of the moving surface from its
mean at the point that moves most): **3.6 cm**, against **3.3 cm** voxels.

* A **snapshot** (rest pose) is on average √2 times as far from the moving surface as the mean (5.1 cm RMS here),
  and is wrong in a fixed direction where the mean is not.
* A **bound** (the swept volume) is a 16 cm thick slab for the cloth: it blocks the light under and around it that
  the real cloth lets through most of the time.
* A **periodic re-bake** is a snapshot of age *a* ≥ the bake latency. For a travelling wave of angular frequency ω
  a snapshot beats the mean only while 2(1 − cos ωa) < 1, i.e. for *a* < π / (3ω): 0.55 s for the cloth's long
  wave (ω = 1.9) and 0.34 s for its short one (ω = 3.1). The bake alone takes 0.46–0.52 s on a worker thread, and
  every refresh costs a Lumen scene re-upload (all clipmap levels recomposed, card pages relit); the refresh would
  never catch up with the motion.

So the mean surface it is: no cost after the bake, the smallest average error of any static surface, within about
a voxel of the moving cloth. Screen traces still see the moving surface.

## Gallery and valley content

The gallery's terrace and beds were baked without distance fields (`SdfMode::None`): flat sheets a few centimetres
above the courtyard's heightfield would only have added a two-sided shell. Their displaced shapes are worth
representing, so the terrace is now a heightfield of 1.6 cm cells (512 along its 8 m) and the bed mesh a 96-voxel
volume (3.3 cm voxels over 3 m). The valley's displaced boulders already had distance fields and cards; their
variants follow the crags. The cobblestone path stays without a field (the cobbles are far below any voxel the
path's 30 m extent would get; the terrain's heightfield stands for it).

## Measurements

RTX 3090 shared with another agent's renders that kept it at 80–95 % utilisation for the whole session: the same
configuration's frame time varied by 2–3× between runs (valley at 2560×1440: 14.9–45.8 ms), so frame totals do
not resolve differences below several milliseconds. Interleaved runs against a build of main (6eb1c4d), 2560×1440,
240–300 frames, 6–8 runs per build and configuration; per-pass minima (ms) of the Lumen passes:

| View | final gather | reflections | surface + radiance cache | cards | Lumen share of the frame (median) |
|---|---|---|---|---|---|
| valley, 100 % / 67 % | 2.70 → 4.38 / 0.92 → 2.17 | 0.22 → 0.17 / 0.12 → 0.12 | 0.25 → 0.24 / 0.22 → 0.25 | 0.04 → 0.03 / 0.03 → 0.03 | 27.6 → 28.1 % / 27.5 → 28.2 % |
| cathedral (no displacement), 100 % / 67 % | 2.46 → 2.41 / 1.24 → 1.21 | 0.50 → 0.48 / 0.29 → 0.24 | 0.37 → 0.37 / 0.43 → 0.45 | 0.08 → 0.08 / 0.08 → 0.07 | 32.9 → 31.8 % / 29.8 → 31.0 % |
| forest (no displacement), 100 % / 67 % | 2.56 → 2.57 / 1.69 → 1.02 | 0.10 → 0.10 / 0.06 → 0.06 | 0.29 → 0.26 / 0.30 → 0.27 | 0.03 → 0.03 / 0.03 → 0.03 | 14.7 → 15.0 % / 11.2 → 11.4 % |
| gallery, 100 % / 67 % | 2.11 → 3.18 / 0.95 → 0.99 | 0.66 → 0.99 / 0.26 → 0.26 | 0.23 → 0.23 / 0.26 → 0.80 | 0.05 → 0.05 / 0.05 → 0.05 | 31.9 → 34.4 % / 30.3 → 31.5 % |
| mushrooms `--eye -5.9,0.3,10.4 --at -6.5,0.08,9.2` | 2.14 → 2.03 | 0.48 → 0.54 | 0.23 → 0.24 | 0.06 → 0.06 | 27.5 → 28.0 % |
| cloth `--eye 4.4,0.6,10.9 --at 6.3,0.15,9.0` | 4.96 → 5.87 | 2.10 → 1.96 | 0.41 → 0.35 | 0.06 → 0.06 | 24.2 → 24.6 % |
| bricks `--eye 0.9,0.45,10.9 --at 0,0,9.2` | 7.66 → 7.76 | 0.82 → 1.02 | 0.95 → 0.29 | 0.06 → 0.06 | 24.7 → 24.8 % |
| valley path `--eye -26,1.8,39 --at -33,0.2,53` | 6.95 → 6.86 | 0.11 → 0.10 | 0.68 → 0.64 | 0.04 → 0.04 | 29.7 → 28.5 % |

Even the minima move by up to ±1 ms in both directions in scenes the change cannot affect (forest 67 % final
gather 1.69 → 1.02; the passes outside Lumen — material resolve, lighting — move as much), and Lumen's share of
the frame, which cancels contention that scales every pass, stays within its run-to-run spread of about ±1.5
points everywhere (gallery 100 %: +2.5 points, with three more instances in the near field and cards). What can be
stated without the noise:

* **Scenes without displacement** run the same GPU work: no variant is requested, no new pass is recorded, the
  upload builds the same buffers in the same order. Images (PSNR new vs main / main vs main, % of pixels over
  24/255): cathedral 55.2 / 56.3 dB, 0.007 / 0.005 %; forest (wind) 41.3 / 40.7 dB, 0.195 / 0.289 %; at 67 %
  55.6 / 55.7 dB and 41.8 / 42.2 dB.
* **Valley default view** (displaced boulders behind the camera): 58.2 / 54.7 dB, 0.001 / 0.001 % (67 %: 57.3 /
  59.4 dB) — within noise.
* **Steady state with displacement**: the variants replace the meshes' fields one for one (same voxel size; the
  gallery's beds and terrace add three instances to the near field and cards), so Lumen traces the same amount of
  data; the per-frame cost is unchanged within the noise above.
* **One-off cost** when a scene's displacement changes (log of the native build, 1280×720):

  | Variant | Refined triangles | Field | Bake (worker thread, incl. card capture) |
  |---|---|---|---|
  | valley boulder (texture height, object triplanar), ×2 meshes | 96 k / 106 k | 45×29×49 / 47×31×49, 5.6 cm voxels | 0.68–1.26 s / 0.77–1.22 s |
  | gallery brick terrace (graph height) | 516 k (no cards) | heightfield 512×193, 1.6 cm cells | 25–80 ms |
  | gallery mushrooms (graph vector) | 124 k | 96×8×96, 3.3 cm voxels | 0.65–0.74 s |
  | gallery cloth (animated graph, 32 samples) | 124 k | 96×6×96, 3.3 cm voxels | 0.46–0.52 s |

  On the render thread: the LOD cut of the meshes (5–11 ms for the gallery's three variants in one frame), one
  GPU dispatch per variant and the read-back, and one Lumen re-upload once all variants are in. Moving the
  refinement (175 ms for the gallery's 385 k points) and the variants' card capture (most of the 58–62 ms Lumen
  re-upload) onto the worker threads brought the worst CPU frame after loading from 61–65 ms to 7–13 ms (gallery, valley; main 5–10 ms).
  In the browser the same work runs one task per frame on the render thread (bakes of 35–710 ms measured in
  headless Chromium).

**Lumen's view** (1280×720, debug views; `lumen-displacement/*.jpg`):

* Mesh cards (view 14) — share of the bed's pixels lit by the voxel fallback (purple) instead of cards: mushroom
  bed 98.5 → 14.0 % (the bed was not in Lumen; the mushrooms, their caps' shadows and the ground between them are
  now card-lit), cloth view 80.6 → 28.8 % (the cloth's mean surface in cards), the valley's large displaced boulder
  19.1 → 14.3 % (the crags now match their cards).
* Indirect light (view 7): the caps occlude the sky and bounce sunlight onto the ground between the mushrooms
  (`mushrooms-indirect.jpg`); the brick terrace's bevels get their own near-field occlusion (`bricks-indirect.jpg`).
* Lit image: mushrooms 37.8 dB vs main (main vs main 40.5 dB), 0.52 % of pixels over 24/255 — the stems and the
  ground under the caps darker, the shadowed ground greener; bricks 42.8 dB; cloth 49.9 dB.

**Browser** (headless Chromium 1248, WebGPU via D3D11): the gallery loads, bakes its three variants (logged in
the console), the mushroom view, its card view and the cloth view render; no console, WGSL or validation errors.
A first run found an alignment panic in the read-back (casting the mapped bytes to `Vec4`, which the browser's
copy does not align): fixed by copying without a cast.

## Remaining limits

* Accuracy is the variant's voxels and the cards' texels: relief finer than those (the bricks' 1 cm mortar joints
  against 1.6 cm heightfield cells, cobbles) reaches Lumen only through screen traces. The near field shows the
  rest within its first 2 m; the global field (0.25 m voxels and up) mostly cannot.
* The surface caches still see a material's average albedo, not the graph's colours: the mushrooms bounce the bed's
  green base colour, not their red caps (README gap 3).
* An animated displacement is represented by its mean surface; card lighting reads the sun's shadow map, which
  holds the moving surface, so the mean surface's card texels see the waves' moving shadows.
* Latency: a variant is ready a few frames after its displacement changes (refine, GPU, read-back, bake: 0.5–1.3 s
  of CPU per variant natively on worker threads). In the browser the bakes run on the render thread, one per frame
  — each a hitch of that length when a scene loads or a displacement is edited.
* Every variant arrival rebuilds Lumen's scene (as any structural change does): all clipmap levels recompose and the
  card atlas is reallocated.
* A displacement that depends on the placement is baked for the transform the instance had at the last structural
  scene change; a movable instance with such a displacement keeps that variant while it moves.
* Meshes baked without a distance field (`SdfMode::None`) stay out of Lumen, displaced or not.

## Entry points relied on (for merging with `displacement-motion`)

`tessellation::{displacement_source, displacement_animated, DisplacementSource, TessView, UNIFORM_SIZE,
BINDING_TESSELLATION}`, `tessellation::graphs::{displacing_graphs, scene_dispatch_module, set_hash, BINDING_ALBEDO,
BINDING_GRAPH_PARAMETERS}`, `CompiledGraph::{displacement, displacement_kind, displacement_hash, parameters}` and,
in WGSL, `displacement_position` / `displacement_input` of `displacement_height.wgsl` with the `Tessellation`
uniform of `displacement.wgsl`. If the other branch adds object- or tangent-space vectors (which do not depend on
the placement), `displacement_content` in `lumen_displaced.rs` should stop treating such vectors as
placement-dependent; treating them as placement-dependent stays correct, only less shared.
