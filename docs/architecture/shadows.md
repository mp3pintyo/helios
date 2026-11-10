# Virtual shadow maps (sun)

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Virtual shadow maps** (`vsm.rs`). A clipmap of 16k² virtual shadow maps (≈1 mm texels near the camera) paged into
  a 4k² physical atlas through the same page cache. Pixels request the pages they sample; only new or invalidated pages
  are rendered — each level's clusters are culled against that level's dirty-page rectangle and bit mask, expanded into
  (cluster, page) draws and rasterised into their tile. With a static sun, almost nothing is re-rendered per frame.
  As in the main view, clusters whose triangles are tiny in the level's texels are **software-rasterised**: one
  32-bit `atomicMax` of depth per texel into a u32 mirror of the atlas (shadows need no payload pass), which the
  page clear copies into the depth atlas before the hardware clusters are drawn on top. Hardware page draws cull
  triangles that miss their page in the vertex shader (no viewport arrays: otherwise large clusters were
  rasterised over every page they touch, once per page); where the device has WebGPU's optional `clip-distances`
  feature, four clip distances clip every page draw to its tile (`vsm_draw_clipped.wgsl`), so triangles larger than
  a page — common near local lights, and above all those crossing a light's near plane, which project across the
  whole atlas — no longer rasterise discarded fragments (a moving candle light near the camera: page raster 4.8 →
  1.3 ms). Streaming, wind, skeletal animation and moved instances invalidate only the cached pages under
  the instances concerned (see [*Wind*](motion.md), *Dynamic world*).
  **Static and dynamic casters** are cached separately, as in UE5's "static separate" page cache: casters the wind
  moves this frame, `Mobility::Movable` instances and skins whose pose changed are *dynamic* (one definition,
  `dynamic_caster.wgsl`), all others *static*. A page that must be refreshed only because dynamic casters
  moved keeps the depth of its static casters as a *static layer* in its tile of the software raster's u32 mirror —
  otherwise just scratch space between the software raster and the page clear, so the layers cost no memory of their
  own — and its clear writes that layer instead of the far plane; culling drops static instances at the instance test
  (each level keeps a second dirty-page mask and rectangle for pages rendered in full, each local light a second box),
  only dynamic clusters are expanded and drawn, and their software-rasterised ones go to one of 256 scratch tiles after
  the atlas' tiles (16 MiB), which the page clear max-merges over the static layer. A page gets its static layer at
  its first such refresh: it is rendered in full once more, static casters first, whose depth is then copied into the
  mirror (`vsm_static.wgsl`) before the dynamic ones are drawn. Anything that changes static casters — the sun or a
  light moving, a structural scene change, a static instance being moved (its footprint only), streaming of static
  instances, the wind being switched on or off (per instance: no global invalidation), voxel LODs being toggled —
  frees the page, so it is rendered in full. A caster fading into or out of the wind as the
  camera approaches or leaves it is at rest then: the cached depth stays right, so the pages under it only drop their
  static layer and are rendered in full at their next refresh (re-rendering them at once doubled the shadow work of
  a camera flying through the forest). The sampled atlas always holds the merged depth, so sampling is unchanged. Sun
  and local light pages alike.
