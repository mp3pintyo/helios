# Dynamic world (UE "Movable" mobility): per-instance invalidation

README text for the round-4 feature `dynamic-world` (branch of agent dynamic-world).

## Architecture bullet (section 2, after *Wind*)

* **Dynamic world** (`motion.rs`, `vsm_motion.rs`, `lumen_dynamic.rs`, `dynamic_debug.rs`; UE5's movable
  primitives). Any instance can move every frame without invalidating the world around it. `Scene::set_transform`
  stamps the instance with a version instead of bumping the scene's instance version, and `Scene::set_mobility`
  declares it `Mobility::Static` (placed once; moving it is a one-off edit) or `Mobility::Movable` (moves often) as UE's
  component mobility. Each frame a `MotionTracker` turns the stamps into the list of moved instances with their
  previous transforms (a scan of the instances only in frames where something moved), and every subsystem applies
  just those moves; adding instances, toggling shadow casting or new geometry are still structural changes that
  rebuild everything once. **GPU scene**: only the moved records are rewritten (one `write_buffer` per run of
  consecutive instances, previously all instances every frame something moved), with the previous frame's transform
  as `prev_world`, so motion vectors (TSR, motion blur) follow rigid motion; an instance that stops is rewritten once
  more to drop its motion. Temporal history is never reset by moves. **Virtual shadow maps**: movable instances are
  *dynamic casters* like wind-animated vegetation (`dynamic_caster` in `wind.wgsl` now also tests the instance's
  `MOVABLE` flag), so they stay out of the pages' static layers. The CPU writes each moved caster's footprint into
  the invalidation list before the GPU appends its streaming and wind footprints: the light-space rectangle around the
  instance's oriented box in its old and new pose (one swept footprint for a small move, one per pose for a jump) —
  for a chandelier on its 12 m chain a third of the area of its bounding circle — and its bounding spheres for
  local lights' faces and mips. Pages under a movable caster re-render only their dynamic casters over their cached
  static layer; pages under a static instance that was moved are freed and rendered in full. At rest a movable
  instance costs nothing. **Lumen**: the instance's distance-field record is rewritten in place and the 8³-voxel
  bricks of each global-field clipmap level under its old and new box (plus the narrow band) are recomposed —
  distance, albedo and the per-voxel closest card instance — by two new passes over a brick list (`cull_listed`,
  `compose_listed` in `lumen_sdf.wgsl`, one workgroup per brick), within a budget of 1024 bricks per frame (the rest
  queue up, finest level first). A level recomposes under an instance only once some point of it moved a quarter of
  that level's voxel since the level last composed it, so the 0.25 m level follows a swinging chandelier every few
  frames and the 2 m level rarely. Movable instances stay in the global field, as in UE5 (it updates movable
  primitives' distance fields rather than excluding them): a fast mover only costs more bricks per frame and, beyond
  the budget, lags a few frames. The near field rewrites the instance's record and rebuilds its culling grid only when
  the instance's box crossed into other 2 m cells; mesh cards move with their instance (its transform in the card
  records; nothing is recaptured) and its card pages are relit first, within half the card lighting budget so the
  round-robin refresh keeps going. The voxel surface cache and the radiance cache are not invalidated: they are
  refreshed continuously (a quarter of the voxels and an eighth of the probes per frame) from the updated field.
  **Translucency** re-places moved instances' candidates (and water bodies when a water instance moves) instead of
  rebuilding its lists. Growing the scene bounds (the shadow clipmap's depth range) is the one move that invalidates
  every cached shadow page. `RenderSettings::per_instance_invalidation = false` (`--full-invalidation`) restores the old
  behaviour (every move a structural change) for comparisons. Debug view 16 (*Invalidation*) shows what moves cost
  this frame: movable instances orange, sun shadow pages rendered this frame hatched red, recomposed distance-field
  bricks (finest level at each pixel) cyan with their outlines. The demo scenes move: the cathedral's eleven
  chandeliers swing on their chains (pendulums with a 7 s period, 1–3° amplitude, each with its own phase and plane;
  their candle lights move along), the valley's gold sculpture turns once in 45 s and a log raft drifts, bobs and rolls
  on the lake; `0` pauses and resumes it.
  Cost (high detail, 1600×900, RTX 3090 shared with up to six other agents: minima of 3 interleaved runs; per-pass
  minima where whole-frame totals were swamped by the other agents). Cathedral, sun shadows only (local lights' shadow
  maps off): GPU frame 5.88 ms at rest; moving 1 / 3 / 11 chandeliers (3 / 9 / 33 instances) 6.04 / 6.05 / 6.04 ms
  with 7 / 32 / 57 sun pages re-drawing only their dynamic casters, versus 7.87 / 7.63 / 8.03 ms with the old full
  invalidation (all 123 cached pages rendered in full, page raster 0.83–0.92 ms instead of 0.01–0.02 ms, and all 4
  distance-field levels recomposed every frame, 0.93–0.98 ms, instead of ~24 bricks). With the chandeliers' candle
  lights' shadow maps on, the lights moving with them dominate: 1 / 3 / 11 chandeliers render 96 / 314 / 520 pages
  (88 / 288 / 440 dynamic-only), page raster 0.28 / 1.11 / 3.62 ms, versus 609 full pages and 8.6 / 7.9 / 13.5 ms
  with full invalidation (138 k instead of 391 k (cluster, page) draws for all eleven); one uncontended pair: GPU
  frame 14.7 ms moving vs 25.8 ms full invalidation. Valley (sculpture + raft): 5.00 ms at rest, 5.66 ms with both
  moving (25 dynamic-only pages) vs 6.44 ms full; the sculpture alone 5.28 vs 7.89 ms. Distance-field bricks: without the
  quarter-voxel threshold the chandeliers recomposed ~390 bricks a frame (0.42 ms), with it 24–120 (≤ 0.07 ms).
  CPU: the moves cost a scan of the instances in frames where something moved plus a few record writes (full
  invalidation also re-uploads the near field and mesh cards, which reset their allocations, every frame). Temporal
  stability (valley, static camera, 16 frames, pixels changing by more than 24/255 outside the movers and their
  shadow pages): 0.049 % moving vs 0.042 % paused (noise: water waves). Scenes without moving instances render as
  before (forest and paused cathedral differ from the previous build only by run-to-run noise: max 9–10/255 either
  way, no pixel over 8/255).

## "Status vs. Unreal Engine 5" — replace the dynamic-world parts

* In item 1 (*Streaming*), replace "Cached shadow pages are invalidated globally when the scene changes (per instance
  only for streaming and wind)." by: "Cached shadow pages are invalidated per instance for moves, streaming and wind;
  structural scene changes (instances added, shadow casting toggled, new geometry) and moves that grow the scene
  bounds invalidate them globally."
* In item 7 (*Wind*), replace "but only the wind makes a caster dynamic: an instance whose transform changes still
  invalidates every cached page (item 1)." by "the wind and `Mobility::Movable` make a caster dynamic." and replace
  "Distance fields, the surface cache and translucent materials don't move." by "The wind doesn't move distance
  fields, the surface cache or translucent materials (rigid moves do)."
* New item **Dynamic world**: Moves are rigid transforms of whole instances (no per-bone or per-cluster updates: a
  skinned or deforming mesh is the skeletal-animation work). A moved caster's shadow footprint is its whole oriented
  box in light space, so a long, thin caster (a chandelier's chain) refreshes pages its geometry doesn't touch; UE
  invalidates per primitive too, but its instance-culled page invalidation is tighter. Rigid moves have no bounded
  speed, so every page under a moving caster is refreshed every frame it moves (the wind skips pages it cannot have
  changed by two texels). A moving local light still re-renders all of its pages every frame (in the demo the
  chandeliers' lights dominate the cost of their motion). The global distance field lags by up to a quarter voxel per
  level (and further for a fast mover beyond the brick budget); an instance that stops within that distance of where
  a level last composed it stays there. The voxel surface cache and radiance cache follow moves over a few frames
  instead of being invalidated locally. Instance records are rewritten with one `write_buffer` per contiguous run, not
  a GPU scatter of dirty records (UE's GPU-scene update); adding or removing instances still rebuilds all instance
  data. The landscape (runtime virtual texture) and water bodies' lake beds don't follow a moved terrain instance.

## New flags and hotkeys

* Hotkey `0`: pause / resume the demo motion (all scenes; the HUD shows a "moving (0)" line with the moved
  instances, shadow casters and recomposed bricks).
* Debug view 16: *Invalidation* (`--view 16`, `V` cycle).
* `--full-invalidation`: every move invalidates everything (the behaviour before per-instance invalidation).
* `--movers N`: keep only the first N moving prop groups moving (cathedral: chandeliers; valley: sculpture, raft).
* `--static-motion`: the props move as `Mobility::Static` instances (the one-off edit path: pages under them are
  rendered in full).
* Library: `Mobility`, `Scene::set_mobility`, `Scene::mobility`, `Renderer::dynamic_stats` (`DynamicStats`),
  `RenderSettings::per_instance_invalidation`.
* Help line: `0 demo motion`.
