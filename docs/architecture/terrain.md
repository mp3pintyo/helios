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
* **Shading.** The mesh terrain's material is the landscape's (`MaterialKind::Terrain`, its grass
  and dirt layers) with `TextureMapping::WorldTriplanar`, which the G-buffer resolve reads as
  "volumetric terrain": no runtime virtual texture (its top-down pages can't hold a wall or an
  overhang's underside), the layers evaluated per pixel by `mesh_terrain_layering` (rvt.wgsl) with
  every layer projected triplanar and the texel footprint measured in 3D. Slope selects rock on
  walls and undersides as on mountains; the shore material (lake bed, wet band) applies as on the
  landscape. The rock is **cliff rock** (round 8, `terrain-rock-material`):
  * *Texture*: its rock layer is `TexturePattern::Cliff` (texgen.wgsl `cliff`), made for walls
    (v up): rounded masses flattened along the bedding, faint lamination, broad fracture planes and
    small chips as randomly tilted planar facets per Voronoi cell (not crack lines), fractures along
    only some plane borders (a noise mask fades them in and out), dark streaks running down the
    face and fine grain. The landscape keeps its cracked `Rock` layer (mountains, far away).
  * *Two scales*: `cliff_rock` samples it at 0.083 and 0.37 repeats per metre (12 m and 2.7 m
    tiles, a ratio that is not a whole number, so they never repeat together); the detail scale is
    projected turned about the vertical so on ledges and undersides its grain crosses the large
    scale's; only its data texels are fetched: its height (mean ≈ 0.5, mip-filtered towards it
    where the detail is sub-pixel) modulates the large scale's colour and its normals add to the
    large scale's. It is skipped where a pixel covers more than 1/32 of its tile. `triplanar_sparse`
    skips projections whose weight is under 2 % (on a wall one or two of three), so the two scales
    fetch fewer texels than the one full triplanar sample before.
  * *Strata from the geometry*: `MeshTerrain::shading_attributes` stores two values per vertex in
    the UVs (the triplanar material needs no texture coordinates): the **strata coordinate** — the
    warped height the rock modifiers cut their ledges at (`rock_warp`, `STRATA_FREQUENCY`), in half
    periods, so the ledges lie at its integers and the parity tells the protruding (hard) beds from
    the recessed (soft) ones; blended between rocks by distance, plain height away from them — and
    the **cavity**, the density probed 0.5 and 1.5 m out along the normal (0 open, up to 1 in a
    crevice; only near volume modifiers). The resolve decodes UVs for Terrain pixels only when the
    material is world-mapped (the landscape needs none), so only the mesh terrain pays for them. The shader gives every bed its own tone (±13 %, warmer or
    greyer), hard beds 7 % lighter and smoother, fine laminae (faded once sub-pixel, from the
    coordinate's screen derivative), and a thin dark parting at bedding planes that is open on some
    planes and comes and goes along each. Because the coordinate comes from the same warp as the
    geometry, colour bands and ledges coincide.
  * *Weathering*: dirt (the dirt layer's mean colour) settles where the cavity is high and the
    surface faces up or sideways; moss and lichen (the boulders' moss colour) on up-facing rock,
    more in hollows and on soft beds; undersides and steep faces stay bare.
  * *Ground* (round 8, `river-cave-artifacts`; the landscape's and every scene's dirt layer,
    `TexturePattern::Dirt`, texgen.wgsl `dirt`): lumpy soil with two populations of stones — gravel
    (60 cells per tile) and stones (23) — that gather in patches (a low-frequency density field,
    sparse elsewhere), vary in size (a few large, many small), shape and turn (`stone_dome`) and in
    hue (warm to cool greys), and lie partly buried: each sinks by its own share into the soil's
    mean level, so its outline is wherever it rises out of the clods, not a circle, and it is
    dusted with the soil and coated where it disappears under it. The old pattern — one pebble size
    in 30 % of a 40 × 40 Voronoi grid, a dome with a step at its rim — read as an evenly spaced
    polka-dot of dark-ringed discs.
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
  walls where all of them are inside the rock; a dry cell's surface hides just under its bed, and
  only under a wall at the heightfield's ground inside the rock (round 8: hiding every cell at the
  heightfield ground, far below the cave's floor, made dry corners plunge through the floor and
  stand as sheets in front of the step at the cave's mouth); boulders standing in the stream are
  instances whose tops are rasterised into the bed, so the current parts around them. The river starts filled with
  the channel's design depth and runs 1800 steps (15 s) on its first frame, so a scene opens on a
  settled river. 72 × 264 cells of 0.25 m (18 × 66 m).
* **Simulation**: the virtual-pipes model (O'Brien & Hodgins, Mei et al.): per cell a depth and the
  flow through four pipes to its neighbours. `flux_step` accelerates each pipe by the difference in
  water surface height (`dt · g · cell · Δh`), slows it by Manning bed friction
  (`g n² |v| / d^(4/3)`) plus a little damping, and scales the outflow so a cell never gives more
  water than it holds; `depth_step` moves the water, adds the sources, holds outlet cells at the
  lake's level, drains the grid's border (spills leave instead of piling up), derives the velocity
  from the flow through the cell and advects foam semi-Lagrangian along it — foam fades over 5 s and
  is born only where the water is turbulent (round 8, `terrain-rock-material`): where the flow
  converges or decelerates sharply (negative divergence of the velocity: the pool under the cave's
  step, the bow of a boulder), where it shears (the curl: the wake behind a boulder; dry neighbours
  count as the cell's own velocity, so banks don't shear), where fast water (> 1.5 m/s) breaks
  over a steep surface, and a little where it runs very fast (> 2.5 m/s); it fades over 3 s. Fixed steps of 1/120 s, at most 6 per frame (a slow frame runs the
  river slow rather than unstable); compute only, two dispatches per step over 19 k cells.
* **Surface**: the river's water instance is a grid mesh over the cell centres whose quads have
  vertices of their own (decoded once into the translucency pass's vertex buffer like any forward
  mesh; a vertex's UV names its quad and corner). `vertex_step` rewrites its vertices every frame in
  that buffer (now created with `STORAGE`): a quad with no water at any corner collapses to a point
  (most of the rectangle is dry; rasterising it under the ground and the full wave set cost
  another 0.3 ms in the initial view); otherwise the height (the water where wet; a dry corner
  carries the highest neighbouring water surface on, level, unless its ground is lower than that —
  then 12 cm under the ground — so the bank cuts a level sheet exactly at the water line; outlet
  corners hide under the bed so the lake's own sheet shows where the river ends; round 8,
  `river-cave-artifacts`: dry corners used to dip under their bed, tilting the shore's triangles
  into the bank, and where one rose out of the ground at a grazing view its back face showed as a
  blue sliver), the normal of the water surface, the velocity in the
  second UV set and foam, depth and speed in the vertex colour, alpha 0 marking simulated water. The
  forward water shading (`shade_water`) then advects its waves along the velocity with a two-phase
  flow map (period 1.6 s; the lake's waves from the fifth, 1 m and shorter — no swell on a stream;
  ripples choppier where faster), tilts them onto the sloped surface and
  draws the simulated foam as a lace of thin lines drifting with the current (the zero crossings of
  an advected noise, thickening with the amount; patches only where the foam is thick);
  refraction, absorption, caustics, reflections and shore foam are the lake's, except that running
  water is rougher than its resolved waves (roughness² + 0.03 · speed², so a fast stretch blurs
  its sky reflection while a pool stays a mirror) and only the shore foam's thin line shows on it.
  Steep water (the chute over the cave's step: the vertex normal's y under 0.85) is aerated white
  water whatever the simulated foam. The river's surface counts as a single layer for
  depth peeling and is not a flat water body (no lake-bed rectangle, no underwater view: a back
  face of its sheet — a fold at a bank — is shaded as the water seen from above, not by
  `shade_water_below`, whose deep-water colour drew it as a saturated blue line).
* **Shores**: `surface_step` writes the water surface height per cell — on dry cells within three
  cells of water the highest neighbouring water surface *that lies below the cell's ground* (ground
  below a pool's level, under the cave's step, is not that pool's bank; taking the pool's level there
  dressed the step's face as submerged bed in blocks of whole cells) — to an `R32Float` texture the
  G-buffer resolve reads (binding 150, `Water::river` gives the grid); `shore_surface` takes the
  higher of the lake's and the river's level, so the river's bed and banks get the lake's wet band
  along the simulated water line. Sediment settles on level ground only (fading out between normal
  y 0.65 and 0.35): steep faces stay the rock they are, wet. Pebbles (`pebbles`) vary in size and
  sink into the sediment by their own share plus a buried-ness that varies over the ground, so they
  gather in patches with irregular outlines. Where the river's level is the higher its bed is a
  stream bed rather than the lake's sand: a coarse gravel matrix packed with gravel and cobbles (an
  armoured bed, no sand ripples), an olive-brown film of algae and silt (periphyton) that thickens
  with depth and in patches — so the bed darkens and greens the deeper the water — and wet stones
  that deepen in colour as well as darken. Under the lake's clear water this reads as clear shallow
  water over a visible bed; the old pale sand under it read as a milky grey sheet.
* **Settings**: `RenderSettings::river_simulation` (default on; `--no-river-sim`, `Numpad2`, editor
  toggle under Atmosphere) pauses the simulation; the surface keeps its last state.

## Costs

See [the design note](../round8/terrain-water.md) for the measurements: building the mesh terrain,
the river's GPU passes and the frame times of the default views.
