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
* **Refresh budget** (round 8, `vsm.wgsl`'s refresh queue; UE 5.8's `r.Shadow.Virtual.DeferredInvalidationBudget`).
  Cached pages whose casters moved (wind, skins, `Mobility::Movable` instances, animated displacement) or streamed are
  not all re-rendered in the frame they are flagged: at most `RenderSettings::shadow_refresh_budget` pages a frame
  (96 by default, `--shadow-refresh-budget N`, 0 = all at once) are, by priority. The invalidation records with each
  flag how far the page's casters may have moved since it was rendered, *in its own texels* — since clipmap texels
  match screen pixels, that is how far its shadow is off on screen — and the queue ranks pages into 64 buckets by that
  distance, doubled for every frame a page has waited; `refresh_plan` cuts the histogram at the first bucket that no
  longer fits (as the page cache's priorities do), `refresh` appends the pages above the cut to the dirty list. A page
  past the budget keeps its cached depth — a shadow a few texels behind its caster, never a hole — and waits with a
  growing priority; one that has waited `shadow_max_staleness` frames (4) is refreshed regardless of the budget, so
  staleness stays bounded. With a budget, **streaming** (a new cluster cut of the static casters, UE's case for the
  budget: Nanite LOD changes) no longer frees the pages under the instances concerned: they keep their depth and are
  queued for a full render. New pages, and pages whose light moved, are never deferred (they have nothing valid to
  show). Forest, default view (the wind flags ~157 pages a frame): 96 refreshed, ~72 waiting, none past 4 frames;
  shadow page passes 1.33 → 0.85 ms (minima; [round 8 note](../round8/vsm.md)). Scenes with fewer refreshes than the
  budget are unchanged.
* **Distant levels at their own rate** (round 8; UE 5.9 updates far shadows at their own rate). A moving sun (`P`,
  the `J` time-lapse) invalidates every page of the clipmap every frame. Levels from
  `RenderSettings::shadow_distant_level` on (7 by default: 12.5 cm texels, receivers beyond ~50 m at 1600×900;
  `--shadow-distant level,interval`, 0 = off) keep the sun direction they were rendered for: each level has its own
  light view (`Vsm::level_light`, used by addressing, culling, rasterisers and sampling) and takes the current sun
  every `shadow_distant_interval` frames (4), the levels staggered so that about one re-renders per frame — at once if
  it has fallen more than 0.5° behind (a sun dragged by hand), so the seam between levels stays below a texel at the
  `P` / `J` speeds. The page cache's invalidation became a bit mask of levels. Wind refreshes of distant pages likewise
  wait for their level's turn (they rarely occur: a distant texel dwarfs most sway). Forest with `P`: pages rendered
  241 → 186 a frame, shadow page passes 5.90 → 2.53 ms (the distant pages hold whole trees). Distant casters still
  render at full resolution: no prefiltered (moment) representation as in UE 5.8's *Prefiltered Distant*.
* **Shadow overview** (round 8; UE 5.9's *Virtual Shadow Map Overview*): debug view 26 (`--view 26`, `V` cycles,
  the editor's *View mode*) shows four panels side by side, each the frame at half size: the clipmap level each pixel
  samples with the page grid (distant levels hatched, magenta where no page is resident), what its page did this frame
  (rendered in full red, dynamic casters over the static layer yellow, split orange, deferred by the budget blue —
  brighter the longer it waited —, a distant page waiting for its turn cyan, cached grey), why it was last
  invalidated (fading over 60 frames: moving dynamic casters yellow, layer change magenta, static change red,
  streaming cyan, new or light moved white), and the physical page pool (32 × 32 tiles in the level palette or a colour
  per local light; dark: cached, bright: rendered this frame, a blue dot: refresh pending) with bars for requests
  against the pool (red: requests without a page), pages kept unrequested, pages rendered against the refresh budget,
  refreshes deferred / held / forced, and the pool's split between the sun and the local lights. Its own pass
  (`vsm_overview.rs`), compiled when first selected: nothing runs while it is off. `2` toggles the refresh budget
  and the distant levels together (off: every invalidated page at once, every level every frame).
