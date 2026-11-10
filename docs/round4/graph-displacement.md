# Material graphs drive Nanite displacement (round 4: `graph-displacement`)

Design note. The README carries the user-facing text (the *Nanite Tessellation* and *Material graphs*
bullets, gaps 3 and 14 of "Status vs. Unreal Engine 5"); this file records why it is built the way it is.

## What

UE feeds Nanite tessellation from the material's Displacement output: any material expression. Helios
displaced by one texture channel (`Material::displacement`: layer, channel, magnitude, centre). Now a material
graph's Output node has a **Displacement** pin: a scalar height, 0..1 around the material's centre and scaled by
its magnitude (`Material::displacement` keeps those two, as UE's material settings do). When the pin is
connected and the graph compiles, the graph's height replaces the texture channel everywhere the height is
evaluated: dicing and geomorphing in the main view (`tess.wgsl`), the shadow pages (`vsm_tess.wgsl`) and the
G-buffer resolve's displaced-hit search and normal tilt (`tess_resolve.wgsl`).

## Code generation (helios-material)

`compile` now emits two functions from one graph, sharing the node code generator and the parameter slots:
`graph_eval` (unchanged, for the resolve) and `graph_displacement(m, input: DisplacementSurface, params) -> f32`
from the nodes the Displacement pin depends on (`CompiledGraph::displacement`, `displacement_function(name)`,
`displacement_hash`). Nodes that only feed the displacement stay out of `graph_eval` and vice versa. A
`Context` (surface / displacement) picks the helpers the code calls (`graph_sample` vs
`displacement_graph_sample`, ...). `DisplacementSurface` uses the field names of the resolve's `SurfaceInput`
(`uv`, `world_position`, `object_position`, `normal` and their `_dx`/`_dy` derivatives), so every node emits
the same expression in both contexts.

**Filtering without screen derivatives.** The tessellator has none, so the displacement context's
"derivatives" span one dice segment at the point, along two tangent directions (`displacement_surface` in
`displacement_graph.wgsl`: world segment = distance to the main camera × dice rate / pixels per unit, scaled
by the instance and the mesh UV density for object positions and UVs). They flow through the graph's math by
the existing forward-mode differentiation, so texture samples (`textureSampleGrad` on the 512² arrays) and noise
are filtered for the micro-triangles that sample them — the same rule as the texture-driven height. Streamed
texture tiles are not used: their residency differs between frames and passes; the height must not.

**What the displacement context rejects** (compile errors on the node, shown by the graph editor):

* `Time`, and a `Panner` whose Time input is left at its default: **time-dependent displacement is
  forbidden** rather than turning the instances into dynamic casters. A displaced surface must be static: the
  VSM's static page cache keeps displaced instances' shadows, the geomorphing assumes heights move only with
  the camera, and the resolve's motion vectors assume a still surface. Animated displacement would need
  per-frame shadow re-rendering of every page under the surface and velocity from the height's time
  derivative; UE allows it but warns about the cost. The error says so ("displaced surfaces are static").
* `CameraVector`, `Fresnel`: view-dependent; every view and the shadow pages must displace a point alike.
* `VertexColor`, `TexCoord` set 1, Triplanar's `TangentNormal`: attributes the tessellator does not read.

## Dispatch in one pass (helios-render)

Tessellation processes many materials in one dispatch, so its pipelines cannot be specialised per material
the way the resolve is. Instead the **set of displacing graphs** (graphs some displaced material takes its
height from) makes one permutation of the *dicing* pipelines (`tessellation/graphs.rs`): a generated module
replaces `displacement_graphs_off.wgsl` with every such graph's function (`displacement_graph_<id>`) and a
`switch` on the material's graph id (`TessMaterial::source.z`, graph index + 1; `source.w` is the parameter
offset). Only dicing evaluates heights — emit, split and the argument passes keep their base pipelines, so a
permutation is 2 modules and 12 pipelines. Permutations are compiled at runtime (natively and in the
browser) and cached by the hash of the set (at most 8 kept); scenes without displacing graphs use the base
pipelines, and the texture path takes one extra uniform branch.

In the G-buffer resolve the graph is known per pipeline: a graph material is resolved by its graph's extended
pipeline, whose module gets `graph_displacement` and `displacement_graph_resolve.wgsl` (calling it) in place of
the stand-in. Plain pipelines never see a graph-displaced pixel.

**Consistency.** A material displaces by its graph only while the graph has an extended resolve pipeline
(the 8 slots, the parameter block's capacity): `ExtendedMaterials` patches the graph id and offset into the
tessellation uniform for exactly those materials, as it patches resolve slots into the material uniform, so the
tessellator and the resolve always agree; a material that loses its pipeline stays flat (its height layer is
`DISPLACEMENT_NO_LAYER`) rather than displacing differently in the two places.

**Crack-free.** The height is a pure function of the point's interpolated attributes (object and world
position, normal, UV set 0), the per-frame tessellation uniform and the parameter block — the compiler's
context rules guarantee it — so both sides of a shared edge compute bit-identical vertices. The CPU mirror's
shared-edge test now also runs with a brick-pattern height (floors, fractions, a steep bevel: non-smooth),
and a shader test validates every node type allowed in the displacement context inside both tessellation
passes (naga and the Tint bitwise-precedence check).

**Bounds.** The graph's height is clamped to [0, 1], so `magnitude · max(centre, 1 − centre)` still bounds the
displacement for culling and the shadow page expansion.

**Material instances.** Graph parameters that feed the displacement live in the same 16 KiB parameter block
(bound to the dicing passes at binding 68): changing one recompiles nothing. Because a change reshapes a static
surface, the renderer hashes the displacement table and each graph-displaced material's code and parameter
values, and invalidates the cached shadow pages when that changes (this also fixed editing a texture-driven
displacement's magnitude, which left stale cached shadows).

## Editor and scenes

The Output node's catalog gained the Displacement pin (and TextureSample / Triplanar a `Height` output — the
data texel's w, a procedural texture's height); the Details panel's Displacement section shows
"Driven by graph · name" (opening the graph) instead of the height channel while the material's graph drives
it, and hints to set a magnitude. Graph links are saved in scene files as before. Preset
`brick_pavement` (*Brick pavement (displaced)*); the gallery has a brick terrace in front of the wet stone
(8 × 3 m, magnitude 3.5 cm, centre 0.9, framed by a kerb): running bond, chipped bevelled edges, per-brick
heights and tints, mortar joints 2.8 cm deep.

## Cost, and one height per vertex

A graph height costs more than a texture tap (the brick preset evaluates three noise layers, five octaves).
Measuring it exposed that the dicer evaluated up to three heights per grid vertex — its own and, for a vertex that
morphs, both ends of the coarser segment it splits. Those ends are grid vertices of the same patch, so the dicers
(`tess.wgsl`, `vsm_tess.wgsl`, both paths) now displace every grid vertex once — into shared memory, or a local
array for F ≤ 2 — and morph from the neighbours' results (`tess_grid_morph` returns their grid indices); the CPU
mirror does the same and its shared-edge and continuity tests still pass. This helps the texture path as much.

Measured (1600×900, RTX 3090 shared with other agents, minima of 3 interleaved runs):

| View | Micro-tris | Raster sw + merge: texture → graph | Resolve: texture → graph | GPU frame (tess. off) |
|---|---|---|---|---|
| gallery close-up `--eye 0.9,0.45,10.9 --at 0,0,9.2` | 2.91 M | 1.43 → 1.89 ms | 0.51 → 0.95 ms | 5.70 → 6.62 ms (3.88) |
| gallery grazing `--eye 3.6,0.12,10.0 --at -1,0.02,8.6` | 2.75 M | 1.29 → 1.73 ms | 0.40 → 0.71 ms | 5.06 → 5.87 ms (3.45) |

(texture: the same terrace displaced by a texture channel of the same magnitude, in a temporary build.) One
height per vertex: valley path raster 1.62 → 1.04 ms, valley grazing 2.83 → 1.83 ms, the brick terrace 4.05 →
1.89 ms; images within run-to-run noise (PSNR 60.5 / 61.1 dB vs base, base vs base 60.9 / 60.5 dB). Scenes
without displacement unchanged: forest 8.73 → 8.67 ms, cathedral 9.05 → 9.04 ms, valley default view 5.07 →
4.90 ms (min GPU frame; images within noise). Static camera on the terrace: 0.0011 % of pixels change by > 24/255
per frame (tessellation off: 0.0012 %). Browser (headless Chromium, WebGPU): the gallery, view 25, `PageDown` and
the editor run without console, WGSL or validation errors; screenshots in `graph-displacement/`.

## Remaining limits

* Scalar height along the normal only (no vector displacement); static only (time, the camera, vertex colours,
  UV set 1 and the tangent frame are compile errors in the displacement); textures from the 512² arrays.
* A displacing graph's materials displace only while it has an extended resolve pipeline (8 slots).
* The permutation is compiled when the set of displacing graphs changes — a hitch per graph edit, as for the
  resolve modules; at most 8 permutations are kept.
* The resolve still evaluates the height six times per displaced pixel (four hit-search steps, two gradient taps):
  the graph path's resolve costs about twice the texture path's.
