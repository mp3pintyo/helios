# Nanite Tessellation

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Nanite Tessellation** (`tessellation.rs`, `displacement.wgsl`, `displacement_height.wgsl`, `tess_core.wgsl`,
  `tess_queue.wgsl`, `tess.wgsl`, `vsm_tess.wgsl`, `tess_resolve.wgsl`; UE 5.4+'s Nanite tessellation as Brian Karis
  describes it). A material's `Material::displacement` (`Displacement`: a height channel of a texture layer — a
  procedural `TexturePattern`'s height, or any channel of an image layer — sampled through the material's own
  mapping, a magnitude and a centre) moves its surface along the normal at render time; the meshes stay smooth
  and small, the detail comes from the texture. **Culling**: displaced instances carry a flag and every bounding
  sphere (instance, page, cluster; frustum, HZB, shadow pages and the shadow page expansion) is padded by the
  material's displacement bound `magnitude · max(centre, 1 − centre)`, as for the wind; the LOD cut keeps the
  undisplaced errors (as UE). Their triangle clusters always go to the software list. **Patches**: every triangle
  of a displaced cluster is a patch, a triangle in its base triangle's barycentric space with fixed-point
  corners (15 fractional bits, so midpoints are exact). Per edge, a tessellation factor is the *undisplaced*
  edge length projected to pixels as if it faced the camera (so displacement towards the viewer still gets
  diced), over the **dice rate** (`RenderSettings::dice_rate`, 2 px as in UE); edges are oriented by their end
  points' positions, so both triangles of an edge — across clusters and LOD levels, whose shared vertices are
  bit-identical — compute the same factor. **PatchSplit**: patches with an edge factor above 8 are split at the
  midpoints of exactly those edges (one, two or three: two, three or four children — red-green refinement), so a
  split depends on the edge alone and neighbours agree; the children go through a global queue in a storage
  buffer, ping-ponged between split passes. WebGPU has no forward-progress guarantee, so instead of persistent
  threads each pass is an indirect dispatch sized by a one-thread pass (as UE does on all platforms), and the
  number of passes follows the deepest split of recent frames (one spare; patches still too coarse after the last
  are diced as they are). **Dicing**: an edge with factor c gets 2^ceil(log2(c/√2)) segments (lengths within √2
  of the dice rate), the inside a uniform grid of F = 2^level segments per side with the finest edge's level
  (F = 1, 2, 4, 8 — up to 64 micro-triangles per patch); grid vertices on an edge with fewer segments snap to its
  segments (a zipper of degenerate triangles). Levels 0 and 1 are diced by a thread per patch, 2 and 3 by a workgroup per patch group in shared memory; since round 8 (`webperf`) one indirect dispatch dices two levels (`tess_dice_level` finds a workgroup's level from the lists' counts), one pipeline per path and raster pass (`GRID` override) instead of one per level: 6 pipelines per view instead of 12 for the browser to compile, the same GPU time, the shadow pages' dicing 0.052 → 0.045 ms and the late pass 0.099 → 0.084 ms (fewer dispatches and barriers). Vertices new at a level are **geomorphed**: they grow out of the
  coarser segment they split, from its midpoint to their displaced place as the factor goes from 2^(level−1) to
  2^level — in displaced space (as CDLOD does for terrain): morphing the parameter instead slides the
  displacement across the surface, which swims. Both sides of an edge snap and morph its vertices alike, so the
  micro-mesh is watertight and changes continuously as the camera moves (a CPU mirror of the split and dice
  logic checks shared edges vertex for vertex — parameters and displaced, morphed positions — on a jittered mesh
  from near and far eyes, and that edge vertices stay on the previous frame's displaced edge while the eye moves
  through doublings and splits: < 0.01 with the morph, 0.031 without it for a 0.15-amplitude displacement of
  unit edges). **Displacement**: a vertex
  moves along its interpolated, normalised normal by `magnitude · (height − centre)`; the height is filtered for
  the size of a dice segment at the vertex (UE picks the displacement mip from the mesh's UV density likewise:
  the dicer has no screen derivatives; `Displacement::uv_density_of` measures a mesh's), read from the 512²
  array layer, a pure function of the vertex, so every pass and both sides of an edge agree. The wind applies
  after the displacement (WPO on the displaced position). **Raster**: patches are diced by grid level — one
  thread per patch for F ≤ 2, 16 or 64 threads per patch for F = 4, 8 (vertices displaced and projected into
  shared memory, then one thread per micro-triangle; every grid vertex is displaced once and a morphing vertex
  mixes its own result with its coarser segment's end vertices' — grid vertices of the same patch — instead of
  evaluating three heights) — and rasterised with the software rasteriser's rules into
  its target and two passes (depth `atomicMax`, then payload), alongside the other software clusters. All
  micro-triangles go through compute, as in UE (whose patches never take the hardware path either): the dice
  rate keeps them at a few pixels; ones stretched over 4096 pixels by a cliff in the height are dropped, and ones
  crossing the near plane are culled. **Visibility payload**: unchanged — `(cluster slot << 7 | base triangle)`,
  as for any software triangle — so material classification, picking, selection and Lumen's screen traces see an
  ordinary id. Next to it the payload pass stores the **barycentrics on the base triangle** of the point
  that was displaced to the pixel — the grid vertices' own, interpolated perspective-correctly over the
  micro-triangle, two 16-bit fractions in a per-pixel R32Uint target allocated with the first displaced instance.
  The G-buffer resolve (a `TESSELLATION` permutation, used only while the scene has displaced instances) reads
  that point instead of searching for it (round 4 projected the depth's hit onto the base plane iteratively,
  which only converges while the displaced surface is a height field over that plane), takes the world position
  from the merged depth, shades the base triangle's attributes there — the material is evaluated at the
  *undisplaced* point, as the displacement is, so a pattern that drives both stays attached — and tilts the
  interpolated normal by the surface gradient of the displacement (height differences a pixel apart along both
  edges), so shading is smooth across micro-triangles; motion vectors carry the hit's own position (otherwise the tiny
  offset between the solved point and the micro-triangle read as motion, and TSR plus motion blur flickered on a
  still surface: 0.45 % → 0.003 % of pixels changing by > 24/255 per frame). **Shadows**: `vsm_tess.wgsl` runs the
  same emit / split / dice on the (cluster, page) software items of displaced clusters, with factors in the
  page's texels (clipmap level or local light face mip at the point), patches culled against the page and
  micro-triangles rasterised like `vsm_raster.wgsl` (24.8 fixed point, page clip, slope bias) into the atlas'
  mirror, scratch tiles included: the static / dynamic page cache treats displaced instances like any other —
  static ones are cached, moving ones (dynamic world) and ones whose displacement is animated (below) re-render
  over their static layer; heights are the main view's, so shadows fall from the surface the camera sees. **Interplay**: skinned instances and voxel
  clusters are not tessellated; alpha-masked or translucent displaced materials render undisplaced; Lumen sees
  the displaced surface through baked variants of the meshes' distance fields and cards (round 6, see [*Lumen*](lumen.md)).
  Debug view **25** colours displaced surfaces by tessellation depth (splits, then dice level: blue → green →
  red) and shades every micro-triangle; `PageDown` toggles tessellation. The valley gained a cobblestone path
  and displaced boulders behind the initial view (one turning on its plinth: a movable displaced instance).
  Measured (RTX 3090 shared with up to six other agents: interleaved runs, minima of 3; valley high, tessellation
  on vs `--no-tessellation`): path view (`--eye -26,1.8,39 --at -33,0.2,53`) at 1600×900: 1.82 M micro-triangles
  (69 k patches; 0.05 M in refreshed shadow pages), GPU frame 4.17 → 5.87 ms — software raster + merge 0.08 →
  1.62 ms (emit, split, dice and both raster passes; late pass 0.04 → 0.12), material resolve 0.46 → 0.53 ms,
  shadow page tessellation 0.14 ms; at 2560×1440: 4.19 M micro-triangles, 7.85 → 11.87 ms (raster 0.11 → 3.44 ms,
  resolve 1.06 → 1.24 ms). Grazing close-up (`--eye -27.5,0.35,41.5 --at -31,0.12,49`, the path filling the
  screen): 3.42 M micro-triangles, 3.92 → 7.13 ms (raster 2.86 ms) at 1600×900; 7.76 M, 6.93 → 13.63 ms (raster
  5.89 ms) at 2560×1440 — about 0.85 ms per million micro-triangles for both raster passes, and grazing views
  dice finer than the screen needs (the factor measures edges as if they faced the camera, as in UE). Centring
  segment lengths on the dice rate (the 1/√2 in the factor) halved the micro-triangles (6.5 → 3.4 M in the
  grazing view) at no visible loss. Scenes without displacement are unchanged: forest (medium) and cathedral
  (high) every pass within ±0.03 ms of the base build (frame 8.80 → 8.98 / 9.11 → 9.14 ms min, noise), images
  differ from the base build no more than two runs of the base build differ from each other (forest: 0.25 % vs
  0.27 % of pixels over 24/255, from its wind; cathedral and valley 0.000 %). The valley's default view (the displaced content behind the camera, none on screen): 4.89 → 5.15 ms —
  the empty tessellation dispatches (≈ 0.08 ms over both culling passes and the shadow pages) and the resolve's
  `TESSELLATION` permutation (0.36 → 0.45 ms: its registers cost every pixel once a scene has displaced
  instances). Temporal stability (sunlit view, 16 frames, pixels changing by > 24/255 per frame): static camera
  0.003 % (tessellation off 0.000 %). Under a slowly changing factor (dice rate ramped 1 % per frame, TAA off)
  frames change at 0.11 % per frame with the morph and 0.09 % with snapping instead, vs 0.03 % for a fixed rate:
  at a 2 px dice rate the per-frame change is dominated by the height filter following the segment size, and
  a pop moves a surface by less than a pixel; the morph is what keeps the edges watertight and continuous (the
  CPU mirror's continuity test fails without it), not a visible difference in this view. A slow dolly (0.3 m/s)
  changes 2.8 % of the path's pixels per frame (0.6 % without displacement): the parallax of real micro-geometry,
  which TSR resolves like any other.
  **Material graphs drive the displacement** (`tessellation/graphs.rs`, `displacement_graph.wgsl`,
  `displacement_graph_resolve.wgsl`; UE: the material's Displacement output). When a material's graph links its
  Output node's Displacement pin, the graph's height (0..1, clamped, so `Displacement::bound` still bounds it;
  magnitude and centre stay the material's) replaces the texture channel everywhere a height is evaluated —
  dicing and geomorphing, shadow pages, the resolve's hit search and normal tilt. Tessellation dices every
  material in one dispatch, so the scene's *set* of displacing graphs makes one permutation of the dicing
  pipelines: a generated module with every such graph's `graph_displacement` (named by graph id) and a `switch` on
  the material's graph id (`TessMaterial::source.zw`: graph id, parameter offset), compiled at runtime — natively
  and in the browser — and cached by the hash of the set (emit and split keep their pipelines; scenes without
  displacing graphs keep the base ones; the texture path pays one uniform branch). The G-buffer resolve runs the
  graph's own function in the graph's extended pipeline. `ExtendedMaterials` patches the graph id and parameter
  offset only for materials whose graph has a resolve pipeline, so the tessellator and the resolve always agree
  (a material that loses its pipeline stays flat rather than displacing differently in the two). Graph
  parameters feeding the displacement live in the material instances' parameter block (bound to the dicing
  passes): editing one recompiles nothing; a changed displacement — table, displacing code or parameter values —
  invalidates the cached shadow pages (editing a texture-driven magnitude used to leave them stale). Crack-free:
  the compiler only lets the height depend on the undisplaced point (object / world position, normal, UV set 0)
  and per-frame constants, and the CPU mirror's shared-edge test also runs a brick-pattern height (floors,
  fractions, a steep bevel). The gallery's brick terrace (preset *Brick pavement*: running bond, chipped bevelled
  edges, per-brick heights and tints, mortar joints 2.8 cm deep) is displaced by its graph alone. Measured
  (1600×900, RTX 3090 shared with other agents, minima of 3 interleaved runs; same micro-triangles, the terrace's
  height from its graph vs a texture channel of the same magnitude in a temporary build): close-up
  (`--scene gallery --eye 0.9,0.45,10.9 --at 0,0,9.2`, 2.91 M micro-triangles) software raster + merge 1.43 ms
  (texture) → 1.89 ms (graph: three noise layers, five octaves per height), material resolve 0.51 → 0.95 ms, GPU
  frame 5.70 → 6.62 ms (3.88 ms with tessellation off); grazing (`--eye 3.6,0.12,10.0 --at -1,0.02,8.6`, 2.75 M)
  raster 1.29 → 1.73 ms, resolve 0.40 → 0.71 ms. Displacing each grid vertex once (above) took the texture-driven
  valley path from 1.62 to 1.04 ms of raster (grazing 2.83 → 1.83 ms) and the graph-driven terrace from 4.05 to
  1.89 ms, with images within run-to-run noise. Static camera: 0.0011 % of pixels change by > 24/255 per frame
  (tessellation off 0.0012 %). Scenes without displacement and the texture path render as before (forest,
  cathedral, valley: image differences and pass times within run-to-run noise); the gallery's initial view, which
  now shows the terrace's near edge, went from 3.76 to 4.58 ms.
  **Vector and animated displacement** (round 5; `docs/round5/displacement.md` has the design). A graph's
  **VectorDisplacement** pin moves the surface by a vector instead of a height along the normal: world space (the
  space of `VertexNormal` and `WorldPosition`, so `VertexNormal × height` is the height pin's displacement and a
  tangent frame can be built in the graph; a per-triangle UV tangent frame would differ across an edge — a crack),
  in units of the material's magnitude times the instance scale, zero in place, its length clamped to 1 so the
  magnitude bounds it: culling pads by `magnitude · scale` (a height's bound is `magnitude · max(centre, 1 −
  centre)`). The tessellators turn it into an object-space offset with the instance's normal matrix as a row
  transform (the inverse linear part, no inverse in the shader). The dicing permutation's switch wraps each
  graph's function by what it returns, so heights and vectors dice in one dispatch; pipelines without a vector
  graph compile the vector path out. Surfaces can now **fold over themselves** — the reason the resolve reads
  stored barycentrics (above): under an overhang several base points project to the same place and a search
  lands on the wrong one. A vector's shading normal is the cross product of the displaced surface's tangents (the
  edge steps in the plane of the interpolated normal plus the offset's finite differences a pixel apart), right
  on undersides too; displaced pixels are no longer flipped as back-facing by their *base* triangle. Shared edges
  stay bit-identical (the CPU mirror runs a folding mushroom field), and the mirror checks that the stored
  barycentrics name every grid vertex's own point. **Animated**: a displacement that reads the time flags its
  instances `INSTANCE_ANIMATED`: they are dynamic shadow casters like movable or wind-blown ones (kept out of the
  pages' static layers), and the invalidation's collect pass appends their footprint every frame — bounds padded
  by the displacement's reach, change 2 · reach — so the pages under them re-render only their dynamic casters
  over the cached static layer, and only levels whose texels the displacement can change by two (coarser levels
  never; finer ones every frame: an arbitrary graph has no speed bound). The resolve of an animated graph
  evaluates the pixel's base point once more at the previous frame's time for its motion vector (TSR and motion
  blur follow the surface); static graphs and texture displacement compile both out and keep their fully cached
  shadows. Lumen represents an animated displacement by its mean surface (see [*Lumen*](lumen.md): displaced surfaces);
  screen traces see the moving one. Presets *Mushrooms (vector displacement)* — a disc of ground around each cell centre wrapped over a
  cap, its underside and a stem — and *Rippling cloth (animated displacement)* — travelling sine waves and a
  scrolling noise, pinned at the edges —, both as 3 m beds in the gallery.
  Measured (1600×900, RTX 3090 shared with another agent, minima of 3 interleaved runs against the previous
  build): the stored barycentrics cost the payload pass about 0.1 ms per 2.9 M micro-triangles and save the
  resolve three displacement evaluations per pixel — brick close-up raster 1.90 → 2.00 ms, resolve 0.91 → 0.69 ms,
  frame 6.56 → 6.45 ms; valley path (texture height) raster 1.04 → 1.06 ms, resolve 0.49 → 0.48 ms, frame 5.01 →
  5.04 ms; valley default view, forest and cathedral unchanged in every pass (±0.03 ms) and image (within
  run-to-run noise). Mushrooms, low view (`--eye -5.9,0.3,10.4 --at -6.5,0.08,9.2`): 2.11 M micro-triangles, GPU
  frame 3.77 ms (tessellation off) → 5.38 ms — raster 1.30 ms, resolve 0.41 → 0.48 ms, no shadow page rendered
  (cached); 5.47 M and 10.58 ms at 2560×1440. Cloth (`--eye 4.4,0.6,10.9 --at 6.3,0.15,9.0`): 1.37 M
  micro-triangles in the view and 3.99 M in the 90 shadow pages that re-render their dynamic casters every frame,
  3.60 → 5.75 ms — raster 0.83 ms, shadow tessellation 0.88 ms, resolve 0.36 → 0.51 ms; the gallery's initial view
  (all three beds at a distance) 3.77 → 4.40 ms with 9 pages a frame. Static camera, pixels changing by > 24/255
  per frame: mushrooms 0.0002 %, bricks 0.0011 % (as before), around the cloth 0.0014 %. One trap was found by
  those dumps: the payload pass matched the depth pass's bits only while both compiled the same arithmetic — the
  barycentrics are computed apart from it and the match has one step of slack now.
  **Speed bound and vector spaces** (round 6; `docs/round6/displacement-motion.md`). The material compiler bounds
  how fast an animated displacement moves (`CompiledGraph::displacement_speed`, `helios_material::motion`): an
  abstract interpretation of the displacement's nodes giving every value an interval and a bound of its time
  derivative — products by the factors' largest magnitudes, `sin`/`cos` keep their argument's rate, a smoothstep's
  slope is at most 1.5 / (max − min), a Panner adds |speed|, noise is Lipschitz (slope 4.7 for any gradients),
  rotations keep a vector's speed; discontinuities in the time (`Frac`, `Floor`, `Step`, `If`), scrolled
  textures and the like are unbounded. It is evaluated per material instance with its parameter values, carried in
  the tessellation uniform (`TessMaterial::motion`, patched as parameters change) and turned per instance into the
  footprint's per-frame motion (speed · Δt) by the invalidation's collect pass, so the cached pages under an
  animated displacement refresh — as for the wind — once it may have moved two of their texels: the cloth's 1.64 ×
  magnitude per second (26 cm/s) still refreshes the finest levels every frame but coarser ones every few frames.
  Unbounded graphs keep the every-frame refresh; `--no-displacement-speed-bound` restores it as a reference. A
  **TransformVector** node (UE's Transform: World, Object = Local, Tangent; two drop-downs in the graph editor)
  re-expresses directions in either function: a vector built in object or tangent space and turned into world
  space follows its instance. Object space is the instance's linear part over its largest scale, so the
  tessellator's conversion back makes the offset exactly `vector · magnitude` in object units, whatever the
  rotation — the tessellators, the shadow pages, the resolve at its stored barycentrics and the previous frame's
  pose (motion vectors) agree. Meshes store no tangents and a per-triangle UV frame would crack, so the
  displacement's tangent frame is the interpolated object normal with the object X axis projected into its plane
  (the mesh's U direction on planes and terrain); the surface function's is the shading frame. The gallery's
  *combed tufts* (`presets::combed_tufts`, a tangent-space vector) grow on a platform turning behind the default
  view (`--eye 1.9,1.1,15.6 --at 0,0.1,13.4`); they made the gallery's ninth extended pipeline, so classification
  now has 16 extended slots (8–23). Measured (2560×1440, minima of 3 interleaved runs against main;
  pages and shadow micro-triangles averaged over the last 60 frames): cloth close-up 172 → 140 pages a frame, shadow
  micro-triangles 9.95 → 9.55 M, shadow tessellation 2.07 → 2.04 ms (its finest levels' 2–4 mm thresholds are below
  the cloth's 4.4 mm a frame, so they still refresh every frame); the cloth from 7 m 83 → 45 pages, 0.34 → 0.13 M,
  0.17 → 0.12 ms; the gallery's default view 9 → 2 pages a frame. Against the every-frame reference, a static
  camera's frames match at 64.5 dB (no pixel off by more than 24/255: no stale shadow); scenes without animated
  displacement are unchanged (valley, cathedral, forest, gallery at 100 % and 67 %: every pass within noise). The
  turning platform's own view: 1.79 M micro-triangles, 9.78 ms.
