# Wind, skeletal meshes and the dynamic world

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Wind** (`wind.rs`, `wind.wgsl`; UE5's foliage World Position Offset, as Nanite supports it). `Scene::wind`
  (direction, strength, gusts rolling through the vegetation as travelling waves, turbulence) moves materials with a
  `Material::wind` response: the whole plant *bends* about its origin (∝ height², a cantilever), branches *sway*
  (∝ distance from the plant's axis) and leaves *flutter*; rocks and buildings have none. The offset is a pure function
  of the vertex's rest position, the instance transform and the time, evaluated identically by all four rasterisers
  (visibility buffer hardware/software, shadow pages hardware/software) and by the G-buffer resolve, so clusters and
  LOD levels sharing a vertex move it identically (no cracks), and wood and leaves sharing a transform move as one;
  motion vectors evaluate the previous frame's offset, so TSR and motion blur follow it. Instance, page and cluster
  bounding spheres are padded by a bound of the displacement for frustum, HZB and shadow-page culling (a unit test
  checks the bound against the displacement). Like UE's *WPO disable distance*, an instance stops moving beyond
  `Wind::disable_distance` (80 m in the forest) or closer, where its motion bound would cover fewer than 4 pixels (grass
  before trees); it fades out over the last quarter, so nothing pops. Cached shadow pages under moving instances
  re-render their dynamic casters over their cached static layer (see [*Virtual shadow maps*](shadows.md)) — but only once their
  geometry may have moved 2 texels since the page was rendered (a speed bound of the displacement), so fine levels
  refresh every frame or two, coarse ones every few frames and levels whose texels dwarf the motion never. Lumen's
  distance fields stay static (the motion is far below their voxels). `L` toggles it. Culling and page expansion test
  a dirty-page bit mask besides the dirty rectangle, without which the scattered refreshed pages cost another ~1 ms.
  Forest (high) at 1600×900, default view, wind on, versus re-rendering the static casters of refreshed pages too (as
  before the static layers; interleaved runs, RTX 3090): ~155 pages refreshed per frame, all over their static layer;
  shadow triangles 4.5 → 3.7 M, (cluster, page) draws 93 k → 77 k, shadow page passes 1.83 → 1.62 ms (the wind adds
  ~1.6 instead of ~1.8 ms of shadow work). Almost every shadow caster in the forest sways (trees included); only
  terrain, rocks and logs are static, so the remaining cost is that of the moving vegetation. A camera flying through
  the forest renders the same number of pages, with 70 k instead of 80 k draws.
  **Wind weights** (round 8, [`foliage-grounding.md`](../round8/foliage-grounding.md)): every vertex's displacement is
  scaled by its *wind weight* — the plant data's attachment (`Mesh::plant.x`, baked by the vegetation generators from
  the distance along the plant from its root: 0 where a blade or stalk leaves the soil or a trunk meets the ground, 1
  where it moves freely), read per vertex by `vertex_wind_weight` in all four rasterisers and the G-buffer resolve
  (corners and motion vectors). Grass blades' and ferns' bases no longer move at ground level (the sway and flutter
  terms reached them), while a frond tip resting near the ground still sways. The weight only reduces the offset, so
  every bound above holds unchanged; voxels and tessellated surfaces move with weight 1, meshes without plant data
  (imports) too.
* **Skeletal meshes** (`skin.rs`, `skinning.wgsl`, `skin_palette.wgsl`; UE 5.5's Nanite skinned meshes). A
  `SkinHandle` is a joint palette (`Scene::add_skin`, `Scene::set_skin_pose` every frame,
  `Scene::set_instance_skin`); each matrix maps a rest vertex to its posed position in the instance's object space.
  Skinning — the linear blend of the vertex's joint matrices by its quantised weights, zero weights skipped, in slot
  order — is a pure function of the decoded rest vertex, its weights and the palette, evaluated by all four
  rasterisers (visibility buffer and shadow pages, hardware and software) and by the G-buffer resolve, as the wind's
  offset is: clusters and LOD levels sharing a vertex put it on the same bits, so cuts stay watertight under animation
  (the geometry tests pose a three-joint limb with four-influence vertices and check every LOD cut, full and partial
  residency, for open edges on the posed positions). Normals use the blended matrix's cofactor; motion vectors read the
  previous frame's palette (per-corner previous positions interpolated with the barycentrics — exact, skinning is
  affine per triangle), so TSR and motion blur follow the limbs (TSR vs. no-TAA on CesiumMan's walk at 100 % and 50 %:
  differences only along edges, no trails). The palettes live in one `Rgba32Float` texture (`textureLoad`, so no
  shader stage gains a storage buffer; the G-buffer resolve was at WebGPU's 8): per skin a header and six texels per
  joint (current and previous rows), rewritten each frame (24 joints: 2.3 KiB); instances find theirs through
  `GpuInstance::flags` bits 12..32 (no layout change). **Culling bounds**: per frame and skin the CPU derives, from
  per-joint rest spheres of the vertices each joint influences, a *posed sphere* (every joint's posed rest sphere — a
  blended vertex lies in the convex hull of its joints' transforms of it) and a *reach* `max |P c − c| + ‖A − I‖₂ r`
  (spectral norms in closed form) that bounds every vertex's displacement; frustum, HZB and shadow-page tests use the
  rest cluster / page / instance sphere padded by the reach, or the posed sphere where that is smaller (unit tests
  check both against random poses with scale). The LOD cut keeps the rest spheres, which keeps the DAG's monotonic
  errors (crack-free). **Shadows**: a skin whose pose changed is a dynamic caster in the static/dynamic page cache, as
  vegetation in the wind: cached pages under it re-render their dynamic casters over the static layer once it may have
  moved 2 texels, bounded per frame by the joints' motion `|ΔP c| + ‖ΔA‖₂ r`; the footprint is the rest sphere padded
  by the largest reach so far (it covers every pose a cached page may still show); a paused skin returns to the static
  layer. **Lumen**: as in UE, skinned meshes get no distance field or cards (the demo bakes them without); screen
  traces still see them. Skinned meshes never get voxel LODs (voxels carry no weights). **Animation**
  (`helios_asset::animation`): glTF skins, node hierarchy and clips (translation, rotation, scale; `STEP`, `LINEAR`
  with shortest-arc slerp, `CUBICSPLINE` Hermite), an `AnimationPlayer` (clip, time, speed, looping, pause,
  crossfade). The palette is `inverse(mesh node) · joint · inverse bind matrix`, i.e. relative to the node that places
  the mesh, so the rest-pose bounds and LOD stay meaningful while `node · palette` is exactly glTF's pose. Debug view
  **17** colours each joint and blends the colours by weight (grey: unskinned). **Permutations**: the G-buffer resolve
  and the hardware shadow page raster compile the skinning code only into a `SKINNING` permutation used while the
  scene has skinned instances (like `UV_EXTENSIONS`): unconditionally it took the forest's resolve from 0.39 to
  0.74 ms and its page raster 5 % slower, although no pixel there is skinned. Instance, page and cluster culling are
  specialised the same way (two pipeline sets): with ~26 000 assembly parts in the forest, the palette lookups cost
  registers in passes that test every part (forest high, medians of 8 interleaved runs: cull instances 0.36 →
  0.31 ms, pages 0.30 → 0.27, clusters 0.55 → 0.53; images unchanged). The visibility rasterisers and page expansion carry it at no measurable cost.
  Cost (shared GPU; min / median of 5 interleaved runs, BrainStem — 61 k triangles, 59 meshes, 18 joints — animated
  vs. the same model imported as a static mesh with `--no-skinning`): main-view raster unchanged (hw 0.02, sw
  0.06–0.07 ms), cull 0.20 → 0.23 ms, material resolve 0.31 → 0.35 ms (the skinning permutation; paused: also 0.35),
  vsm expand 0.09 → 0.12 ms, shadow page raster (hw + sw + resolve) 0.02 → 0.29 ms — the moving caster's pages
  re-rendered every frame, which a paused model doesn't (0.02 ms: its pages stay cached); GPU frame 4.16 → 4.41 ms
  (paused 4.14). Compressed payload: +2.5 bytes per cluster vertex on the Fox, +3.3 on CesiumMan (+19 / +25 %), +0.3 on
  the mostly rigid BrainStem (+3 %). Scenes without skins: every pass within ±0.02 ms of the build before skinning,
  images within run-to-run noise.
* **Dynamic world** (`motion.rs`, `vsm_motion.rs`, `lumen_dynamic.rs`, `dynamic_debug.rs`; UE5's movable
  primitives). Any instance can move every frame without invalidating the world around it. `Scene::set_transform`
  stamps the instance with a version instead of bumping the scene's instance version, and `Scene::set_mobility`
  declares it `Mobility::Static` (placed once; moving it is a one-off edit) or `Mobility::Movable` (moves often) as UE's
  component mobility. Each frame a `MotionTracker` turns the stamps into the list of moved instances with their
  previous transforms (a scan of the instances only in frames where something moved), and every subsystem applies
  just those moves; adding or hiding instances, reassigning materials, toggling shadow casting or new geometry are
  still structural changes that rebuild everything once. **GPU scene**: only the moved records are rewritten (one
  `write_buffer` per run of consecutive instances, previously all instances every frame something moved), with the
  previous frame's transform as `prev_world`, so motion vectors (TSR, motion blur) follow rigid motion; an instance
  that stops is rewritten once more to drop its motion. Temporal history is never reset by moves. **Virtual shadow
  maps**: movable instances are *dynamic casters* like wind-animated vegetation (the `MOVABLE` instance flag), so they
  stay out of the pages' static layers. The CPU writes each moved caster's footprint into the invalidation list
  before the GPU appends its streaming and wind footprints: the light-space rectangle around the instance's oriented
  box in its old and new pose (one swept footprint for a small move, one per pose for a jump) and its bounding
  spheres for local lights' faces and mips (a skinned instance's padded by its skin's largest reach). A long box is
  cut into up to 8 roughly cubic slices along its longest axis, each with a footprint of its own (a chandelier's
  iron, a 2.8 m ring on a 12 m chain: five of 2.4 m instead of one 6.3 m sphere over most of every nearby candle
  light's cube). A movable caster's footprints also carry a speed bound per slice — its largest corner displacement
  per frame, held with a slow decay — so, as for the wind, a page is refreshed once the slice may have moved 2 of its
  texels since the page was rendered: the slices near a pendulum's pivot hardly ever, the swinging ring's fine pages
  every frame and coarse mips every few. Skipped pages may show the caster up to 2 texels off, so the GPU pads move
  footprints by that much, and a caster that stops is refreshed once more at rest. Pages under a movable caster
  re-render only their dynamic casters over their cached static layer; pages under a static instance that was moved
  (or a jump) are freed and rendered in full. At rest a movable instance costs nothing. **Lumen**: the instance's
  distance-field record is rewritten in place and the 8³-voxel bricks of each global-field clipmap level under its
  old and new box (plus the narrow band) are recomposed — distance, albedo and the per-voxel closest card instance —
  by two new passes over a
  brick list (`cull_listed`, `compose_listed` in `lumen_sdf.wgsl`, one workgroup per brick), within a budget of 1024
  bricks per frame (the rest queue up, finest level first). A level recomposes under an instance only once some
  point of it moved a quarter of that level's voxel since the level last composed it, so the 0.25 m level follows a
  swinging chandelier every few frames and the 2 m level rarely. Movable instances stay in the global field, as in
  UE5 (it updates movable primitives' distance fields rather than excluding them): a fast mover only costs more
  bricks per frame and, beyond the budget, lags a few frames. The near field rewrites the instance's record and
  rebuilds its culling grid only when the instance's box crossed into other 2 m cells; mesh cards move with their
  instance (its transform in the card records; nothing is recaptured) and its card pages are relit first, within half
  the card lighting budget so the round-robin refresh keeps going. The voxel surface cache and the radiance cache are
  not invalidated: they are refreshed continuously (a quarter of the voxels and an eighth of the probes per frame)
  from the updated field. **Translucency** re-places moved instances' candidates (and water bodies when a water
  instance moves) instead of rebuilding its lists. Growing the scene bounds (the shadow clipmap's depth range) is the
  one move that invalidates every cached shadow page. `RenderSettings::per_instance_invalidation = false`
  (`--full-invalidation`) restores the old behaviour (every move a structural change) for comparisons. Debug view
  16 (*Invalidation*) shows what moves cost this frame: movable instances orange, sun shadow pages rendered this
  frame hatched red, recomposed distance-field bricks (finest level at each pixel) cyan with their outlines; the HUD
  has a "moving (0)" line with the moved instances, shadow casters and recomposed bricks. The demo scenes move (`0`
  pauses): the cathedral's chandeliers swing, the valley's sculpture turns and its raft drifts.
  Cost (high detail, shared GPU; minima of 3 interleaved runs, per-pass minima where whole-frame totals were swamped).
  Cathedral, sun shadows only (local lights' shadow maps off): GPU frame 5.88 ms at rest; moving 1 / 3 / 11
  chandeliers (3 / 9 / 33 instances) 6.04 / 6.05 / 6.04 ms with 7 / 32 / 57 sun pages re-drawing only their dynamic
  casters, versus 7.87 / 7.63 / 8.03 ms with full invalidation (all 123 cached pages rendered in full, page raster
  0.83–0.92 ms instead of 0.01–0.02 ms, and all 4 distance-field levels recomposed every frame, 0.93–0.98 ms, instead
  of ~24 bricks). With the chandeliers' candle lights' shadow maps on, the lights moving with them dominate: 1 / 3 /
  11 chandeliers render 96 / 314 / 520 pages (88 / 288 / 440 dynamic-only), page raster 0.28 / 1.11 / 3.62 ms,
  versus 609 full pages and 8.6 / 7.9 / 13.5 ms with full invalidation (138 k instead of 391 k (cluster, page) draws
  for all eleven); one uncontended pair: GPU frame 14.7 ms moving vs 25.8 ms full invalidation. Valley (sculpture +
  raft): 5.00 ms at rest, 5.66 ms with both moving (25 dynamic-only pages) vs 6.44 ms full; the sculpture alone 5.28
  vs 7.89 ms. Distance-field bricks: without the quarter-voxel threshold the chandeliers recomposed ~390 bricks a frame
  (0.42 ms), with it 24–120 (≤ 0.07 ms). CPU: a scan of the instances in frames where something moved plus a few
  record writes. Temporal stability (valley, static camera, pixels changing by more than 24/255 outside the movers and
  their shadow pages): 0.049 % moving vs 0.042 % paused (noise: water waves). Scenes without moving instances render
  as before. Slices and speed bounds (2560×1440): the swinging cathedral's static candle lights refreshed ~820 pages a
  frame under the chandeliers (`vsm expand` 0.68 ms), now ~25 (0.23 ms); the valley's sculpture and raft 38 → 15 sun
  pages. The chandeliers' own lights' shadow maps are no longer re-rendered every frame (round 4: distance-field
  shadows while they move, cathedral GPU frame 14.86 → 10.17 ms at 100 %; round 5: cached around an origin near the
  light, 10.8 ms, *Local light virtual shadow maps*).
