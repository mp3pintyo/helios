# Round 8 — terrain and water: the lake's edge line, mesh terrain, shallow-water rivers

Three backlog items (229 `lake-edge-line`, 221 `mesh-terrain`, 222 `shallow-water-rivers`), in this
order because the terrain is the river's bed. How the result works is described in
[architecture/terrain.md](../architecture/terrain.md); this note is the why, the alternatives and the
numbers.

## 1. The bright line at the lake's north end (229)

**Problem.** From `--scene valley --eye -2.8,3.2,27 --at -1.5,2.6,30.6` a thin, bright, straight line
crossed the landscape and the coloured panels, already in round 7
(`docs/round8/translucency/lake-panels-peeled-vs-avboit.jpg`). Suspects were the water body's
rectangle edge, shore foam and a seam in the water or shore shader.

**Cause.** None of those: removing the clear glass pane that stands in the water (translucency test
content) removed the line. The camera is 20 cm above the pane's top, so the line is the pane's
**4 cm top face** seen at grazing incidence (~4–8°). There a dielectric reflects about half the light,
and `shade_glass` took its reflections from the radiance cache's *irradiance* around the reflected ray
— a shortcut chosen because pane reflections are "faint (a few percent)". Around a near-horizontal ray
that hemisphere is half sky, so the top face reflected bright sky where the true reflection is the
landscape behind it; where the screen-space trace found the rocks instead, the line broke into dark
dashes (the dashes in the round-7 image). The water rectangle, foam and shore were innocent.

**Fix.** Glass reflections stronger than 10 % (`GLASS_TRACED_REFLECTANCE`: the env-BRDF at grazing
angles) trace Lumen's global distance field as water's reflections do; face-on panes keep the cheap
cache lookup. The top face now shows the landscape it reflects: a faint, slightly darker edge line
(the pane's real edge, aliased into a few-pixel staircase), not a bright stroke
(`terrain-water/lake-edge-line-before-after.jpg`, top before, bottom after). Cost: only grazing glass
pixels take the trace — a few thousand pixels here.

## 2. Mesh terrain (221)

**UE reference.** UE 5.8's experimental Mesh Terrain: terrain as a mesh (cliffs, overhangs, tunnels),
resolution raised where needed, a non-destructive modifier stack (sculpt, paint), integrated with
Nanite, virtual textures and PCG. Forum reports: it doesn't work with the water / shallow-water
systems and a 16 k terrain takes about a minute to load. Helios' terrain had to carry its river.

**Design** (details in the architecture page):

* A **density field** = base heightfield + ordered modifier stack. Height modifiers (channel,
  flatten) change the heightfield so heightfield readers see them (the river's bed, the landscape's
  distance field, the RVT heightfield); volume modifiers (rock with taper, capsule tunnel) are smooth
  CSG with noise and strata on rock faces. The ground term is divided by `√(1 + |∇h|²)` so it is a
  distance, and the smooth unions blend evenly on slopes.
* **Surface nets per chunk** with edge ownership that makes same-resolution neighbours watertight
  without stitching; **two resolutions** (0.25 m around the volume modifiers, 0.5 m elsewhere) joined
  by overlapping skirts that reach across the half-cell gaps; chunks are **ordinary virtual meshes**
  (Nanite DAG, streaming, culling, own distance fields where rocky), borders locked and on one
  absolute quantisation grid.
* A **seam** that replaces a rectangle of the existing landscape exactly: the mesh terrain follows
  the landscape mesh's own triangles at the hole's edge and dips under it outside — the landscape,
  its RVT and its heightfield field stay as they were around it.
* **Shading** through the landscape's material with world-triplanar mapping (no RVT for walls and
  undersides), **collision** through a triangle grid (`TerrainCollider`; the camera's new
  `floor_below`), **PCG** sampling the generated triangles.

**Alternatives considered.**

* *Marching cubes / dual contouring.* Marching cubes makes many slivers and more triangles for the
  same cell size; dual contouring keeps sharp edges but needs Hermite data and a QEF solve per cell.
  Natural rock has few sharp edges and Nanite simplifies anyway: surface nets is the cheapest
  watertight option with good triangles.
* *Transvoxel transitions* between resolutions instead of skirts: crack-free by construction but a
  large case table and a third kind of cell; skirts work because transitions are kept on
  heightfield-like ground.
* *Replacing the whole landscape by a mesh terrain.* A 900 m valley at 0.5 m would be ~6 M triangles to
  mesh and bake at every start (UE's minute at 16 k is the warning); keeping the heightfield where it
  is a heightfield and meshing a region costs ~1 s.
* *Holes by sinking landscape vertices* instead of removing cells: a pit in the landscape's
  heightfield field and RVT, overdraw, z-fighting where both surfaces meet. Removing the cells and
  dipping the mesh terrain under the edge avoids all three.
* *The RVT for the mesh terrain*: its pages are top-down projections of a heightfield; walls would
  stretch and undersides would show the ground above them.

**Showcase.** The valley, visible from the initial view: a cliff band (27 m × 13 m) with a cave, a
shelf overhanging the meadow by ~5 m, a free-standing fin with a 6.4 m arch east of the sculpture
and the river's channel. Views: `--eye 7,1.8,-25 --at -3.9,-3,-38.9` (overhang),
`--eye 20.7,1.8,-9.8 --at 34,-5.5,-22` (arch), `--eye -14,2.2,-28 --at -12.5,1.8,-40` (cave).

## 3. Shallow-water rivers (222)

**UE reference.** UE 5.9: rivers driven by a shallow-water simulation, paintable water input,
collision resolution against actors, actors excluded from the capture.

**Design.** A `River` in the scene (grid, bed, initial depth, outlets, sources, Manning n, surface
instance); a GPU **virtual-pipes** simulation (flux step, depth step with sources, outlets, border
drain, velocity and semi-Lagrangian foam), fixed 1/120 s steps, ≤ 6 per frame, a 15 s warm-up on the
first frame. The **surface** is a grid mesh whose vertices the simulation rewrites every frame inside
the translucency pass's own vertex buffer (no new geometry path, no new bindings in the translucency
layout): height, normal, velocity (UV1), foam and depth (colour). The water shader advects its waves
with a two-phase flow map and draws the foam as drifting streaks. A dilated surface-height texture
gives the G-buffer resolve the river's water line, so its banks get the lake's wet band and sediment.
The bed comes from the mesh terrain's density (cave floor under its roof, walls), the stream's
boulders are baked into it (the flow splits around them and foams behind them), and the lake is the
outlet the river drains into.

**Alternatives considered.**

* *A grid along the river's spline* (curvilinear): fewer cells for a long thin river, but
  neighbour fluxes need the metric terms and the surface mesh and the shore texture an inverse
  mapping. The rectangle costs 19 k cells, of which most are dry — at 0.03 ms it was not worth it.
* *Semi-Lagrangian shallow water with momentum advection* (Chentanez & Müller): hydraulic jumps and
  eddies, but more passes and more tuning for stability; the pipes model is monotone and robust on
  wet/dry fronts, which a river in a carved channel is mostly about.
* *A separate render pass for the river.* It would have had to duplicate refraction, the scene-colour
  copy, ordering with glass and the AVBOIT composite; writing the vertices into the existing forward
  path keeps the river a water instance like the lake.
* *Closed grid borders.* Water that spilled once (a design error found on the way: the first river
  course ran above the meadow to its east) piled up against the edge into a flat sheet; the border now
  drains.

**Showcase.** The initial view: the river leaves the cave over a foaming step and runs past the
sculpture into the lake. Close-ups: `--eye -2,3,-6 --at -9,0,-20` (current, boulders, foam,
wet banks), `--eye -16,2.5,14 --at -8,0,4` (the mouth). `Numpad2` pauses the simulation.

## Measurements

RTX 3090 shared with other agents' builds and renders; 1600×900; minima of 3 interleaved runs (and a second set
of 3 after the optimisations below) of this branch with and without the mesh terrain and the river (a temporary
switch building the old landscape), per-pass GPU minima from the profiler.

| View | Frame (without → with) | Changed passes |
| --- | --- | --- |
| Initial view | 4.94 → 5.49 ms | translucency 0.33 → 0.67 ms (river surface), material resolve 0.38 → 0.48 ms (per-pixel triplanar layering of the mesh terrain), translucency scene colour 0.04 → 0.08 ms (the river and the lake overlap on screen: one peel), river simulation 0.04 ms |
| River close-up `--eye -2,3,-6 --at -9,0,-20` | 4.31 → 5.70 ms | translucency 0.00 → 0.92 ms, material resolve 0.37 → 0.66 ms |
| Arch `--eye 20.7,1.8,-9.8 --at 34,-5.5,-22` | 3.88 → 3.85 ms | material resolve 0.34 → 0.44 ms |
| Lake edge (item 1) | — | translucency 0.256 → 0.229 ms (noise: only the pane's top face traces) |

The first version of the river surface cost 0.97 ms of translucency in the initial view: the dry part of
its rectangle was rasterised (and discarded under the ground) and every river pixel evaluated all 16 waves
twice. Per-quad vertices that collapse where a quad is dry and starting the river's waves at the fifth
(1 m and shorter) brought it to 0.67 ms. The simulation itself is 0.03–0.04 ms (19 k cells, 1–2 steps a
frame at 60 fps; the 1800-step warm-up runs once).

Building: the mesh terrain meshes in ~1–2 s natively (250 k triangles in 25 chunks, 16 fine; the machine was
shared, so these vary) and its chunks' virtual meshes take ~2 s more; in Chromium (`--detail medium`,
WebAssembly, one thread) meshing takes 1.6 s, the chunks 1.2 s, the river's bed and surface 0.2 s, and the
whole valley 7.4 s. The holed landscape bakes as before (its heightfield field from the filled grid takes
~40 ms).

Browser (Chromium headless, WebGPU, D3D11): no WGSL or validation errors; the initial view, the river, the
arch and the lake edge render as natively (`terrain-water/browser-chromium.jpg`). Tests: `cargo test
--workspace --release` 353 passed (new: four mesh terrain tests, one river test).

## Remaining limits

See `docs/status.md` item 18 (terrain and rivers) and item 5 (water bodies).
