# Vector and animated displacement (round 5: `displacement`)

Design note. The README carries the user-facing text (the *Nanite Tessellation* and *Material graphs* bullets, gap
14 of "Status vs. Unreal Engine 5"); this file records why it is built the way it is, and the measurements.

## What

Round 4 left two gaps against UE's Nanite tessellation: displacement was a scalar height along the normal, and the
graph compiler rejected `Time` on the displacement branch, because the shadow page cache and the motion vectors
assumed a displaced surface never moves. Both are closed:

* **Vector displacement.** A material graph's Output node has a second geometry pin, **VectorDisplacement**
  (float3). The surface moves by that vector, so it can lean, shift sideways and fold over itself: overhangs.
* **Animated displacement.** `Time` (and a `Panner` on its default clock) may feed either displacement pin. The
  instances of such a material are dynamic shadow casters, and their pixels get motion vectors from the previous
  frame's surface. Graphs that do not read the time keep the static, fully cached behaviour.

Still rejected in the displacement, with an error on the node: `CameraVector` and `Fresnel` (every view and the
shadow pages must displace a point alike), vertex colours, UV set 1 and Triplanar's tangent-space normal
(attributes the tessellator does not read).

## The vector: which space, which units

**World space, in units of the material's magnitude times the instance's scale; (0, 0, 0) stays in place; the
length is clamped to 1.**

* *World space* because that is the space graph nodes speak: `VertexNormal` is the world normal and
  `WorldPosition` the world position in both of a graph's functions, so `VertexNormal × height` displaces exactly as
  the Displacement pin does (minus the centre) and a tangent frame can be built in the graph
  (`normalize(cross(VertexNormal, axis))`). UE 4's World Displacement was world space for the same reason. A
  tangent-space vector is not offered: the tessellator has no tangent frame, and one derived per triangle from the
  UVs differs on the two sides of an edge — a crack. Object-space vectors would need a transform node the graph
  does not have (see the limits).
* *Units of the magnitude* so `Material::displacement` keeps its meaning — the one number that bounds the
  displacement — and a material instance scales the whole shape. The centre is not used by a vector.
* *Clamped to unit length* so the bound is a bound: `magnitude · scale` pads every culling sphere (instance, page,
  cluster; frustum, HZB, shadow pages, the page expansion, the patch culling of both tessellators), where a height's
  bound is `magnitude · max(centre, 1 − centre)` (`tessellation::displacement_bound`). A graph that overshoots is
  clipped rather than culled wrongly.

A graph connects Displacement **or** VectorDisplacement; both is a compile error on the Output node. The compiler
reports what it built as `CompiledGraph::displacement_kind` (`Height` / `Vector`) and `displacement_animated`, and
emits the two as constants next to `graph_displacement` (`GRAPH_DISPLACEMENT_VECTOR`, `GRAPH_DISPLACEMENT_ANIMATED`).

The object-space offset the tessellators add is `(vector · magnitude) · N` with `N` the instance's normal matrix
used as a row transform: `N` is the inverse transpose of the instance's linear part times its largest scale, so the
product is the inverse linear part applied to `vector · magnitude · scale` — no matrix inverse in the shader, and
the wind still applies to an object-space position afterwards.

**Crack-free** as before: the vector is a pure function of the undisplaced point (object and world position,
normal, UV set 0), the per-frame tessellation uniform (dice rate, camera, time) and the parameter block, so both
sides of a shared edge compute bit-identical vertices. The CPU mirror's shared-edge test runs a third field, a
mushroom pattern that provably folds (a point moves sideways past where its neighbour stays).

**One permutation for all graphs.** The dicing permutation's switch wraps each graph's function by what it
returns (`displacement_from_height` / `displacement_from_vector`), so heights and vectors, static and animated
graphs dice in the same dispatch. `DISPLACEMENT_VECTORS` (some graph of the module is a vector) compiles the vector
path out of the base pipelines and of every resolve pipeline that cannot meet one.

## The resolve: stored barycentrics instead of a search

The visibility payload names the base triangle, not the micro-triangle. Round 4's resolve therefore *searched* for
the base point that the displacement moves onto the pixel's depth: a fixed-point iteration projecting the hit onto
the base plane after taking off the displacement. That converges only while the displaced surface is a height field
over the base plane. An overhang is exactly where it is not: several base points project to the same place, the
iteration lands on the wrong one (under a mushroom's cap: on the ground below, a local minimum of the distance) and
no better iteration helps without a starting point on the right fold.

So the rasteriser now tells the resolve. The dicer knows every grid vertex's barycentrics on the base triangle
(`tess_grid_lambda`: the grid point, snapped to its edge's segments as the position is; a morphing vertex keeps
its own, being the midpoint of the segment it grows out of). The payload pass interpolates them over the
micro-triangle — perspective-correct: with reverse-Z and an infinite far plane the depth is near / w, so the edge
weights times the corners' depths are the weights over w and their sum is the pixel's depth — and stores them,
packed as two 16-bit fractions (`tess_pack_lambda`), in a per-pixel R32Uint target (`FrameTargets::tess_lambda`,
allocated once a scene has displaced instances). The resolve reads the pixel's point, displaces it once and shades
it. No search, exact on every fold, and the same code for heights and vectors.

*Why a texture written by `textureStore`:* the resolve already binds eight storage buffers, WebGPU's guaranteed
limit, so it cannot read the software raster target; a texture binding is free. Writing it from the dicer (rather
than as a third part of the raster buffer copied by the merge) was measured to cost the same and needs no extra
pass. Concurrent writes to a texel are a race only between micro-triangles of exactly equal depth at a pixel centre
— the same race the payload has had since the software rasteriser exists.

*Precision:* 16 bits per barycentric is 1/65536 of the base triangle's edge — finer than the patches' own 15-bit
corners can split. The world position still comes from the depth, so the quantisation only moves where attributes
are sampled, by a fraction of a micro-triangle.

*A trap on the way:* the payload pass writes where its recomputed depth equals the depth pass's bits. That held
because both passes compiled the same expression. The barycentrics first reused the products the depth is summed
from; the compiler then rounded the payload pass's z differently and about half the pixels matched no triangle —
they kept a stale payload (harmless before: the same base triangle) and stale barycentrics (a speckle that TSR
smeared into noise: 2–4 % of the brick terrace's pixels changing per frame with a static camera, against 0.001 %).
The barycentrics are now computed from the normalised weights, and the payload pass accepts one step of slack
below the stored depth, so the match no longer depends on how a compiler contracts the surrounding code.

**Normal.** A height keeps round 4's formula (the interpolated normal tilted by the height's gradient, two taps a
pixel apart along the edges). A vector's normal is the cross product of the displaced surface's tangents: the steps
along both edges, taken in the plane of the interpolated normal (so an undisplaced point of a curved mesh keeps its
smooth normal), plus the finite difference of the offset over them. It is exact where the surface folds: the
underside of a cap gets a normal that points down. Displaced pixels are no longer treated as back-facing when their
*base* triangle faces away from the camera (the dicer culls back-facing micro-triangles; the displaced surface
decides).

**The material sees the undisplaced point.** For a displaced pixel, `WorldPosition`, `ObjectPosition` and the UVs
(and triplanar mapping, for texture-driven displacement) are now those of the base point, as in the displacement
function — not the displaced position. With a height along the normal the two differ little; with a vector they
differ by the whole shift, and a pattern that drives both displacement and colour (the mushroom's cap, gills and
stem are functions of the distance to the cell centre) would otherwise slide off the geometry it shaped. Lighting,
the view vector and motion vectors use the displaced position from the depth.

The resolve now evaluates the displacement three times per displaced pixel (the point and two taps) instead of six
(four search steps and two taps).

## Animated displacement

`Time` compiles to `input.time` in the displacement function (the surface function keeps `frame.time`), and the
tessellation uniform carries this frame's and the previous frame's time, so a caller chooses when.

**Shadows.** An instance whose material's displacement is animated carries `INSTANCE_ANIMATED` (set on the CPU with
`INSTANCE_DISPLACED`; assembly parts inherit it). `dynamic_caster` (the one predicate culling, page expansion and
invalidation agree on) treats it like a movable instance or one in the wind: it stays out of the pages' static
layers. The invalidation's collect pass — the dynamic-world machinery — appends its footprint every frame: the
instance's bounds padded by the displacement's reach, with *change* = 2 · reach (how far any point can be from where
a cached page shows it). Pages under it re-render only their dynamic casters over the cached static layer, and
only where the displacement can have changed them by two texels: clipmap levels (and local lights' mips) whose
texels are larger than the reach are never refreshed for it; the others are refreshed every frame (see the limits:
there is no speed bound). The footprint is not flagged as the wind's: local light pages judge the wind by the
motion their last rendering measured per page, which knows nothing of displacement, so an animated caster
invalidates them through its footprint, as a move does. The pass runs while such a caster exists (`VsmFrame::animated_displacement`); a scene
without one records nothing new. A graph that starts or stops reading the time bumps the graphs' displacement
version: instance flags are re-uploaded and the casters change layer.

**Motion vectors.** A graph resolve whose displacement is animated (`DISPLACEMENT_ANIMATED`, a constant of the
graph's module) evaluates the pixel's base point once more at the previous frame's time: the previous position is
that point's, through the previous instance transform and wind. TSR and motion blur follow the surface instead of
smearing it. Static graphs and texture displacement compile that evaluation out.

**The dicing** needs nothing: factors come from undisplaced edges, and the geomorph is computed from the current
frame's displaced positions.

**Not updated:** Lumen's distance fields, mesh cards and surface cache keep the undisplaced geometry (as for static
displacement: the displacement is far below their voxels); screen traces see the animated surface. The near field
and translucency placement do not move. Shadow page fallbacks that draw displaced clusters undisplaced (dynamic
pages without a scratch tile, dynamic casters of split pages, local-light items reaching the near plane) now can
apply to animated casters every frame rather than only while a displaced instance is moved.

## Editor, scenes, presets

* The Output node lists the two geometry pins set apart (a rule above them, amber labels) with a tooltip each;
  the graph editor's status line names what the graph does to geometry ("vector displacement", "animated
  displacement"); linking both pins marks the Output node with the compile error.
* The Details panel's Displacement section shows *Vector* or *Height* · *Driven by graph · name* (· *animated*),
  hides the centre for a vector (it has none) and notes that an animated material's shadows are re-rendered.
  `heliosEditorState` reports `displacement_mode` / `displacement_animated` per material and `displacement` /
  `displacement_animated` per graph.
* Scene files need nothing new: links are stored by pin name. A test round-trips both presets.
* Presets (`helios_material::presets`): **`mushrooms`** — *Mushrooms (vector displacement)*: staggered cells of
  world xz; around each centre a disc of the ground is wrapped over a mushroom's silhouette (dome of the cap, its
  underside back in to the stem, down the stem, the rest stretched back over the ground), per-cell size, a "Stem
  Height" parameter, colours along the same pieces, faded out towards the bed's edges; magnitude
  `MUSHROOM_MAGNITUDE` = 11 cm. **`rippling_cloth`** — *Rippling cloth (animated displacement)*: two travelling
  sine waves and a scrolling noise (a Panner on the clock), pinned along the edges by the mesh UVs.
* The gallery has both as 3 m beds beside the brick terrace: mushrooms on the left (`--eye -5.9,0.3,10.4 --at
  -6.5,0.08,9.2`), the cloth on the right (`--eye 4.4,0.6,10.9 --at 6.3,0.15,9.0`, magnitude 16 cm, centre 0.5).

## Tests

* `helios-material`: the vector pin compiles to a `vec3<f32>` function, a scalar broadcasts, the wrong width and
  both pins are errors; `Time` and a Panner on the clock animate (and a Panner on a constant does not); the clock in
  the surface function alone leaves the displacement static; the camera, vertex colours, UV set 1 and the tangent
  normal are rejected under both pins.
* `helios-render`: every node type allowed in the displacement, as a height and as a vector, plus the presets,
  validates in both tessellation passes and in its own resolve (naga + the Tint bitwise check); only an animated
  graph's resolve evaluates `tess.time.y`; bounds and flags of height / vector / animated graphs; the displacement
  version bumps when a graph changes kind or starts moving.
* CPU mirror (`tessellation/crack_tests.rs`): shared edges bit-identical under the folding vector field; the stored
  barycentrics reproduce every grid vertex's undisplaced position (snapped edge vertices included) and survive the
  16-bit packing.
* `helios-demo`: the editor reports the mode, and scene files round-trip a vector and an animated graph.

## Measurements

1600×900, RTX 3090 shared with one other agent's builds and renders; interleaved runs against main (11dfc7d),
minima of 3. GPU pass times in ms.

**Existing content** (base → this branch):

| View | GPU frame | Raster sw + merge | Material resolve |
|---|---|---|---|
| gallery bricks close-up (graph height, 2.9 M micro-triangles) `--eye 0.9,0.45,10.9 --at 0,0,9.2` | 6.56 → 6.45 | 1.90 → 2.00 | 0.91 → 0.69 |
| gallery bricks grazing `--eye 3.6,0.12,10.0 --at -1,0.02,8.6` | 5.82 → 5.89 | 1.73 → 1.83 | 0.67 → 0.53 |
| valley path (texture height, 1.82 M) `--eye -26,1.8,39 --at -33,0.2,53` | 5.01 → 5.04 | 1.04 → 1.06 | 0.49 → 0.48 |
| valley default view (displaced content off screen) | 4.74 → 4.74 | 0.12 → 0.12 | 0.37 → 0.37 |
| forest (medium), no displacement | 7.99 → 8.00 | 1.10 → 1.10 | 0.44 → 0.44 |
| cathedral, no displacement | 5.64 → 5.68 | 0.09 → 0.09 | 0.34 → 0.34 |

The stored barycentrics cost the payload pass about 0.1 ms per 2.9 M micro-triangles (0.02 on the valley path) and
save the resolve three displacement evaluations per pixel: a clear gain where the displacement is a graph
(−0.2 ms), neutral where it is a texture tap. (The gallery views now also contain the two new beds; the grazing
view's shadow pages under the cloth refresh, 28 a frame.) Scenes without displacement: every pass within ±0.03 ms,
images within run-to-run noise (PSNR base vs new / base vs base: valley default 65.4 / 64.4 dB, forest 41.2 / 41.1
dB — its wind —, cathedral 61.4 / 61.2 dB). Displaced surfaces differ from main by design (the material is
evaluated at the undisplaced point; base triangles no longer flip displaced normals): valley path 38.7 dB.

**New content** (tessellation off → on, this branch):

| View | Micro-triangles (main / shadows) | GPU frame | Raster sw + merge | Resolve | Shadow tessellation | Shadow pages / frame |
|---|---|---|---|---|---|---|
| mushrooms, low `--eye -5.9,0.3,10.4 --at -6.5,0.08,9.2` | 2.11 M / 0 | 3.77 → 5.38 | 0.06 → 1.30 | 0.41 → 0.48 | 0.06 | 0 (cached) |
| mushrooms, from above `--eye -4.2,1.1,11.6 --at -6.3,0.05,9.0` | 0.66 M / 0 | 3.85 → 4.75 | 0.05 → 0.58 | 0.40 → 0.46 | 0.06 | 0 (cached) |
| rippling cloth `--eye 4.4,0.6,10.9 --at 6.3,0.15,9.0` | 1.37 M / 3.99 M | 3.60 → 5.75 | 0.05 → 0.83 | 0.36 → 0.51 | 0.88 | 90, dynamic casters only |
| gallery initial view (all three beds, far) | 0.23 M / 0.02 M | 3.77 → 4.40 | 0.05 → 0.30 | 0.40 → 0.47 | 0.08 | 9, dynamic casters only |

At 2560×1440: mushrooms low 5.47 M micro-triangles, 10.58 ms (raster 2.84, resolve 1.04); cloth 3.64 M + 9.67 M in
168 shadow pages a frame, 11.78 ms (raster 1.98, resolve 1.08, shadow tessellation 2.08). The static vector
displacement costs what a height costs per micro-triangle and nothing in shadows once cached. The animated cloth's
shadows cost as much as its main view while it fills the screen: every page finer than its 8 cm reach under 9 m²
of surface is re-diced every frame (the static layers of those pages stay cached; with full invalidation every
caster on them would be redrawn).

**Temporal stability** (static camera, 16 frames, pixels changing by more than 24/255 per frame): mushrooms
0.0002 % (tessellation off 0.0002 %); brick terrace 0.0011 % (main 0.0011 %); valley path, sunlit 0.0026 % (main
0.0034 %); the cloth view outside the cloth 0.0014 % (0.0016 % over the whole image: the waves move well under a
pixel a frame, and TSR follows them by their motion vectors).

**Motion vectors**: a temporary build wrote the resolve's motion into the albedo (`displacement/motion-vectors.jpg`):
bands of opposite sign travel with the cloth's waves, zero at its pinned edges, peaks of about 0.3 pixels a frame
(expected from the long wave: 4.3 cm · 1.9 rad/s at 3.5 m), and exactly zero on the static bricks.

**Browser** (headless Chromium 1248, WebGPU on D3D11 via ANGLE's flag set of the brief): the gallery's mushroom,
cloth and brick views, debug view 25 and the editor (graph editor with both presets, Details panel, the both-pins
error) run without console, WGSL or validation errors; the cloth refreshes 66 shadow pages a frame there
(1280×720). Screenshots in `displacement/`.

## Remaining limits

* The vector is world space only: on a rotating (movable) instance a graph built from constants does not turn with
  it (build it from `VertexNormal`, which does). There is no transform node for object- or tangent-space vectors.
* Tessellation factors come from the *undisplaced* edges (as in UE): where a vector stretches the surface — a
  stem drawn out of a 2 cm ring of ground — micro-triangles are that much larger than the dice rate. Shading stays
  smooth (the normal is the displaced surface's); silhouettes get coarser. `--dice-rate` or a larger footprint in
  the graph are the remedies.
* An animated displacement has no speed bound (the graph is arbitrary code), so every cached shadow page whose
  texels are finer than the displacement's reach re-renders its dynamic casters every frame, as UE re-renders
  WPO-animated casters; the wind, which has a speed bound, refreshes coarse levels every few frames.
* Lumen (distance fields, cards, surface cache) does not follow displacement, static or animated; only screen
  traces do.
* The stored barycentrics cost one more store per covered pixel in the payload pass, also for texture-driven
  displacement, whose cheap heights gain little from the shorter resolve (see the measurements).
* Round 4's limits otherwise stand: 8 extended resolve pipelines, one dicing permutation per set of displacing
  graphs (a hitch per graph edit), textures from the 512² arrays, hard edges and UV seams crack.
