# Nanite Tessellation and displacement (round 4: `nanite-tessellation`)

README text for the merge: an Architecture bullet (section 2, after *Wind* / *Nanite Foliage*), the gaps for
"Status vs. Unreal Engine 5", the new flags and keys, and a row for "WebGPU constraints".

## Architecture bullet (section 2)

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
  segments (a zipper of degenerate triangles). Vertices new at a level are **geomorphed**: they grow out of the
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
  shared memory, then one thread per micro-triangle) — and rasterised with the software rasteriser's rules into
  its target and two passes (depth `atomicMax`, then payload), alongside the other software clusters. All
  micro-triangles go through compute, as in UE (whose patches never take the hardware path either): the dice
  rate keeps them at a few pixels; ones stretched over 4096 pixels by a cliff in the height are dropped, and ones
  crossing the near plane are culled. **Visibility payload**: unchanged — `(cluster slot << 7 | base triangle)`,
  as for any software triangle — so material classification, picking, selection and Lumen's screen traces see an
  ordinary id. The G-buffer resolve (a `TESSELLATION` permutation, used only while the scene has displaced
  instances) recovers the displaced point from the merged depth: it projects the hit onto the base triangle's
  plane after taking off the displacement there (a fixed-point iteration: the offset runs along the nearly face
  normal, so four steps converge), shades the base triangle's attributes there and tilts the interpolated
  normal by the surface gradient of the displacement (height differences a pixel apart along both edges), so
  shading is smooth across micro-triangles; motion vectors carry the hit's own position (otherwise the tiny
  offset between the solved point and the micro-triangle read as motion, and TSR plus motion blur flickered on a
  still surface: 0.45 % → 0.003 % of pixels changing by > 24/255 per frame). **Shadows**: `vsm_tess.wgsl` runs the
  same emit / split / dice on the (cluster, page) software items of displaced clusters, with factors in the
  page's texels (clipmap level or local light face mip at the point), patches culled against the page and
  micro-triangles rasterised like `vsm_raster.wgsl` (24.8 fixed point, page clip, slope bias) into the atlas'
  mirror, scratch tiles included: displacement is static, so the static / dynamic page cache treats displaced
  instances like any other and moving ones (dynamic world) re-render over their static layer; heights are the
  main view's, so shadows fall from the surface the camera sees. **Interplay**: skinned instances and voxel
  clusters are not tessellated; alpha-masked or translucent displaced materials render undisplaced; Lumen's
  distance fields and mesh cards stay undisplaced (the displacement is far below their voxels; screen traces see
  it). Debug view **25** colours displaced surfaces by tessellation depth (splits, then dice level: blue → green →
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

## Status vs. Unreal Engine 5 (new item)

**Nanite Tessellation**: displacement is a height along the interpolated normal from one texture channel
through the material's mapping (UE: any material graph output, including vector displacement); the height is
read from the 512² texture array layer (streamed image tiles are not consulted) at a mip chosen from the dice
segment size, so heights finer than the layer's texels are lost. Dicing uses a uniform 2^k grid per patch with
snapped edges instead of UE's precomputed table of patterns for every factor triple (more micro-triangles for long
thin patches; F is set by the finest edge), and splitting is binary at edge midpoints (UE splits with wider
factors). Hard edges (split normals) and UV seams under a UV-mapped height crack, as in UE: the two sides
displace a shared position differently. Patches are culled by frustum (main view) or page (shadows) only, not by
the HZB or backfacing. Micro-triangles larger than 4096 pixels (a cliff in the height) are dropped and ones
crossing the near plane culled — there is no hardware fallback for them; the shadow pages' hardware fallbacks
(local light items reaching the near plane, dynamic pages without a scratch tile, dynamic casters of split pages)
draw displaced clusters undisplaced. Lists hold 512 k patches per split queue and 256 k per dice level: when
they overflow, the dice rate is raised until they fit (HUD). The visibility payload names the base triangle:
the resolve finds the displaced point by an iterative projection, which is approximate where the displacement
folds over itself, and texture derivatives are the base plane's. Lumen sees displaced surfaces only through
screen traces. Skinned, alpha-masked, translucent and voxel geometry is not tessellated; the wind does not
rotate displaced normals.

## WebGPU constraints table (new row)

| No forward-progress guarantee (persistent threads) | Tessellation's PatchSplit is a fixed number of indirect split passes over a ping-pong queue, sized by the deepest split of recent frames |

## Flags and hotkeys

* `PageDown`: tessellation on / off (`RenderSettings::tessellation`; off: displaced materials render undisplaced).
* `--view 25` (`V`, `heliosSetView(25)`): tessellation depth and micro-triangles.
* `--no-tessellation`, `--dice-rate N` (pixels per micro-triangle edge, default 2; `RenderSettings::dice_rate`).
* `--speed N`: fly speed in m/s for held keys (e.g. a slow dolly: `--hold KeyW --speed 0.3`).
* Library: `Displacement` (`Displacement::height`, `bound`, `uv_density_of`), `Material::displacement`,
  `TexturePattern::Cobblestone`, `Renderer::tessellation_stats` (`TessellationStats`).
* HUD: micro-triangles and patches (main view, shadows), dice rate, split passes.
* Demo cameras: path `--eye -26,1.8,39 --at -33,0.2,53`, grazing `--eye -27.5,0.35,41.5 --at -31,0.12,49`, sunlit
  `--eye -27,1.2,41 --at -31,0.1,48 --sun 200,28` (valley).
