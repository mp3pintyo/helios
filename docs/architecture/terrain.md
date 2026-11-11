# Terrain: mesh terrain and shallow-water rivers

Round 8 (`mesh-terrain`, `shallow-water-rivers`). The valley's landscape is a heightfield
(`primitives::terrain`, a regular grid baked into a virtual mesh, shaded through the runtime virtual
texture — see [virtual textures](virtual-textures.md)); north of the lake a rectangle of it is a
**mesh terrain** that can hold cliffs, overhangs, a cave and an arch, and a **river** simulated as
shallow water runs from the cave down its carved channel into the lake. Water bodies (the lake) are
described in [translucency and water](translucency.md).

## Mesh terrain (`helios_geometry::mesh_terrain`, `scenes/valley_terrain.rs`)

UE 5.8's experimental Mesh Terrain replaces the landscape heightfield by a mesh edited with
non-destructive modifiers. Helios' version:

* **Representation: a density field and a modifier stack.** `MeshTerrain` holds a base height
  function (the landscape's) and an ordered list of `Modifier`s. *Height modifiers* reshape the
  heightfield itself — `Channel` carves a river bed along a polyline of (x, bed height, z) points
  (flat bed, banks rising over `bank` metres, only ever lowering the ground), `Flatten` levels a
  pad — so whatever reads the ground as a heightfield (the river's bed, the landscape's distance
  field, the runtime virtual texture's heightfield) sees them. *Volume modifiers* are smooth CSG on
  a density (negative in the ground, about the signed distance near the surface): `Rock` merges a
  rounded, rotated box whose footprint narrows (`taper` > 0) or widens towards its top (negative:
  faces that lean out) into the ground; `Tunnel` removes a capsule (a cave, an arch's opening). Rock
  faces get `roughness` metres of domain-warped 3D noise and horizontal strata (ledges). The ground
  term is `(y − h(x, z)) / √(1 + |∇h|²)`, a distance rather than a height difference, so the smooth
  union with the rock's true distances blends evenly on slopes.
* **Meshing: surface nets per chunk.** `build_chunks` splits the region into 16 m chunks and meshes
  each with surface nets: densities on a lattice whose positions come from global indices (so
  neighbouring chunks compute bit-identical samples), a vertex per crossed cell at the mean of its
  edge crossings with the normal of the cell's trilinear interpolant, a quad per crossed lattice
  edge. A chunk owns the edges *along* an axis in its cells 0…n−1 and *across* it on lattice lines
  1…n — so neighbouring chunks of the same resolution share their border cells and are watertight by
  construction, without stitching. A chunk's vertical range is that of its columns' heights plus
  the volume modifiers over it (sparse in y).
* **Adaptive resolution.** Chunks within `margin` (4 m) of a volume modifier's bounds — the points of
  interest — mesh at the fine cell size (0.25 m), the rest at the coarse one (0.5 m); Nanite's DAG
  then simplifies each chunk for distance. Where a fine chunk meets a coarse one both stop half a
  cell short of the border and each hangs a *skirt* across the gap: a strip reaching 0.475 m
  outwards while sinking 12 cm, so the two strips overlap under each other and close the gap from
  above and the side. The margin guarantees transitions only cross heightfield-like ground (no
  overhang ever meets a skirt).
* **Nanite integration.** Every chunk is an ordinary virtual mesh (clusters, DAG, pages, streaming,
  culling) on one absolute quantisation grid (`position_precision: Some(8)`) with locked borders
  (`SimplifyOptions::lock_borders`), so neighbouring chunks stay crack-free at every level of detail.
  Chunks holding rock get their own volume distance field (40³) for Lumen; flat chunks need none.
* **The seam with the landscape** (`LandscapeSeam`). The landscape leaves out its grid cells inside
  a rectangle aligned to its grid lines (at both detail levels). Inside the rectangle the mesh
  terrain's base height follows the *landscape mesh's own triangles* (its vertex heights, linearly
  interpolated with its diagonal) at the edge and blends to the true height function one landscape
  cell inside; outside, over a band one coarse landscape cell wide, it dips 15 cm under the
  landscape. The two surfaces therefore coincide exactly along the edge and the mesh terrain hides
  under the landscape beyond it: no crack, no z-fighting, and the landscape keeps its resolution,
  its RVT shading and its heightfield distance field. That field (and the RVT's heightfield) is
  baked from a *filled* landscape whose heights inside the rectangle are the mesh terrain's
  heightfield part (channel included); the rock is in the chunks' own fields.
* **Shading.** The mesh terrain's material is the landscape's (`MaterialKind::Terrain`, same four
  layers) with `TextureMapping::WorldTriplanar`, which the G-buffer resolve reads as "volumetric
  terrain": no runtime virtual texture (its top-down pages can't hold a wall or an overhang's
  underside), the layers evaluated per pixel with every layer projected triplanar and the texel
  footprint measured in 3D. Slope selects rock on walls and undersides as on mountains; the shore
  material (lake bed, wet band) applies as on the landscape.
* **Collision and placement** (`TerrainCollider`). The chunks' triangles go into a 2 m grid of
  buckets: `raycast` walks the cells a ray's ground projection crosses (2D DDA), `ground_below(p)`
  returns the highest surface at or under a point (the floor under the arch, the arch's top above
  it), `top(x, z)` the highest. The valley's camera uses `ground_below` (`SceneSetup::floor_below`,
  new): the camera can fly through the arch and the cave and stand on the cliff. `MeshTerrain::
  surface_below` answers the same from the density field directly (the river's bed uses it).
  **PCG**: loose rock on the cliff and around the arch is a PCG graph (surface sampler → slope filter
  → self pruning → transform → spawner) evaluated on the generated triangles (`TriangleSurface`), so
  points land on the cliff top, ledges and the meadow at its foot, never on a wall or under the
  shelf; the valley's 600 scattered boulders skip the reshaped ground.
* **The valley's content.** A cliff band 27 m long and 13 m high facing the initial view (faces leaning out, strata),
  a rock shelf near its top overhanging the meadow by ~5 m, a cave through it whose floor carries
  the river's spring, a free-standing fin on the hillside to the east pierced by a 6.4 m arch, and
  the river's channel down to the lake. Region 80 × 80 m, 25 chunks (16 fine).

## Shallow-water rivers (`helios_render::river`, `shaders/river_sim.wgsl`)

UE 5.9's rivers are simulation driven (shallow water with paintable water input). Helios' river:

* **Scene description** (`Scene::rivers`, `River`): a grid (origin, cell size, cells per axis) with
  the bed height per cell, the initial depth, *outlet* cells held at a level (the lake), sources
  (position, radius, m³/s), Manning's roughness, and the water instance drawing the surface. The
  valley builds it from the mesh terrain: the bed is the channel's heightfield, under rock the floor
  below the lowest point above the ground that lies in the open (the cave's floor under its roof),
  walls where all of them are inside the rock; boulders standing in the stream are instances whose
  tops are rasterised into the bed, so the current parts around them. The river starts filled with
  the channel's design depth and runs 1800 steps (15 s) on its first frame, so a scene opens on a
  settled river. 72 × 264 cells of 0.25 m (18 × 66 m).
* **Simulation**: the virtual-pipes model (O'Brien & Hodgins, Mei et al.): per cell a depth and the
  flow through four pipes to its neighbours. `flux_step` accelerates each pipe by the difference in
  water surface height (`dt · g · cell · Δh`), slows it by Manning bed friction
  (`g n² |v| / d^(4/3)`) plus a little damping, and scales the outflow so a cell never gives more
  water than it holds; `depth_step` moves the water, adds the sources, holds outlet cells at the
  lake's level, drains the grid's border (spills leave instead of piling up), derives the velocity
  from the flow through the cell and advects foam semi-Lagrangian along it — foam fades over 5 s and
  is born where the flow is fast or the water surface breaks steeply between wet cells (behind
  boulders, at the cave's step). Fixed steps of 1/120 s, at most 6 per frame (a slow frame runs the
  river slow rather than unstable); compute only, two dispatches per step over 19 k cells.
* **Surface**: the river's water instance is a grid mesh over the cell centres whose quads have
  vertices of their own (decoded once into the translucency pass's vertex buffer like any forward
  mesh; a vertex's UV names its quad and corner). `vertex_step` rewrites its vertices every frame in
  that buffer (now created with `STORAGE`): a quad with no water at any corner collapses to a point
  (most of the rectangle is dry; rasterising it under the ground and the full wave set cost
  another 0.3 ms in the initial view); otherwise the height (the water where wet; under the
  bed where dry or an outlet, so the shoreline is the surface's intersection with the ground and the
  lake's own sheet shows where the river ends), the normal of the water surface, the velocity in the
  second UV set and foam, depth and speed in the vertex colour, alpha 0 marking simulated water. The
  forward water shading (`shade_water`) then advects its waves along the velocity with a two-phase
  flow map (period 1.6 s; the lake's waves from the fifth, 1 m and shorter — no swell on a stream;
  ripples choppier where faster), tilts them onto the sloped surface and
  draws the simulated foam as streaks drifting with the current; refraction, absorption, caustics,
  reflections and shore foam are the lake's. The river's surface counts as a single layer for
  depth peeling and is not a flat water body (no lake-bed rectangle, no underwater view).
* **Shores**: `surface_step` writes the water surface height per cell — on dry cells within three
  cells of water the highest neighbouring water surface — to an `R32Float` texture the G-buffer
  resolve reads (binding 150, `Water::river` gives the grid); `shore_surface` takes the higher of
  the lake's and the river's level, so the river's bed and banks get the lake's sediment, pebbles and
  wet band along the simulated water line.
* **Settings**: `RenderSettings::river_simulation` (default on; `--no-river-sim`, `Numpad2`, editor
  toggle under Atmosphere) pauses the simulation; the surface keeps its last state.

## Costs

See [the design note](../round8/terrain-water.md) for the measurements: building the mesh terrain,
the river's GPU passes and the frame times of the default views.
