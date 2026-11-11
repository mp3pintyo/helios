# Virtual shadow maps: refresh budget, distant levels at their own rate, overview (round 8: `vsm-deferred-invalidation`, `vsm-prefiltered-distant`, `vsm-overview-debug`)

Design note. The user-facing description is in [`docs/architecture/shadows.md`](../architecture/shadows.md) (the
*Refresh budget*, *Distant levels at their own rate* and *Shadow overview* bullets); the remaining gaps are in
[`docs/status.md`](../status.md) items 4 and 7. This file records why it is built this way and the measurements.

## Problem

Cached virtual shadow map pages are re-rendered in two situations that cost a lot in the demo scenes:

* **Moving casters.** The wind makes nearly every caster of the forest dynamic. A page is flagged for a refresh once its
  casters may have moved 2 of its texels since it was rendered (`vsm_invalidate.wgsl`), and every flagged page that is
  needed is re-rendered in that frame (its dynamic casters over its static layer). In the forest's default view that
  is ~157 pages every frame, ~1.3–1.8 ms of shadow page passes: "the wind still costs most of what it did" (status
  item 7).
* **A moving sun.** `P` (the sun turning) or `J` (the day-night time-lapse) change the light direction every frame,
  and the page cache then re-renders every requested page of every level, every frame: ~241 pages, ~6 ms of shadow
  page passes in the forest, most of it in the distant levels, whose pages each hold whole stands of trees.

And there was no single view of what the shadow map is doing: which level a pixel samples (debug view 8), what was
invalidated (debug view 16), what the pool holds (only numbers in the HUD) were separate.

## UE reference

* **UE 5.8 `r.Shadow.Virtual.DeferredInvalidationBudget`**: invalidated pages are re-rendered under a per-frame
  budget instead of all at once; UE uses it to spread out the invalidations that Nanite LOD changes (streaming)
  cause.
* **UE 5.8 "Prefiltered Distant" (experimental) / UE 5.9 far shadows at their own update rate**: distant casters go
  into the clipmap at a lower resolution, prefiltered and temporally reprojected; 5.9 lets far shadows update at
  their own rate.
* **UE 5.9 "Virtual Shadow Map Overview"**: several shadow debug visualisations tiled on screen.

## Design

### Refresh queue with a budget (`vsm-deferred-invalidation`)

The per-physical-page refresh flag (`INFO_PAGE_REFRESH`) used to be 0 / 1. It now carries a **priority**: how far the
page's casters may have moved since it was rendered, in the page's own texels (× 16, at least 1), plus a
`REFRESH_FULL` bit. `invalidate_pages` takes the largest distance over the footprints that hit the page (a workgroup
`atomicMax`): for a sun page `min(change, speed bound × age) / texel`, for a local light page the same in its texels
at the footprint's depth, or the wind motion bounds its last rendering measured. Since a clipmap level is chosen so
that its texels match the screen's pixels, the distance in texels is roughly how far the cached shadow is off on
screen — the right measure of what a deferred refresh costs in quality, and comparable across levels and between the
sun and local lights.

The single `refresh` pass that appended every pending, needed page to the dirty list became three:

1. `refresh_count` — per physical page with a pending refresh that is needed this frame: its bucket among 64,
   `63 − ⌊2 · (log₂ texels + frames waited + 4)⌋` clamped to 1…63 (more texels and longer waits come first; every
   frame of waiting counts as much as a doubling of the error), or **bucket 0** once it has waited
   `shadow_max_staleness` frames. A histogram in the info buffer (cleared every frame).
2. `refresh_plan` — one thread walks the buckets from 1 to the first one that no longer fits the budget (bucket 0
   is always taken), as `cache_plan` does for the page pool: the cut-off bucket and how many of its pages still fit.
3. `refresh` — appends the pages above the cut (and the cut-off bucket's first ones, by an atomic counter) to the
   page cache's dirty list exactly as before (`PAGE_DYNAMIC` over the static layer, `PAGE_SPLIT`, or `PAGE_FULL` for
   `REFRESH_FULL`); the others keep their flag and their cached depth and are counted as deferred.

A deferred page shows its last rendering: a shadow some texels behind its swaying caster, never a hole, and its
priority grows every frame it waits; after `shadow_max_staleness` (4) frames it is refreshed regardless of the
budget, so the lag is bounded. New pages and pages whose light moved (the page cache's dirty list) are never deferred:
they have no valid content.

**Streaming** (UE's own use of the budget): an instance whose mesh pages changed residency changes its cluster cut,
and the pages under it used to be **freed** (`FOOTPRINT_STATIC`), i.e. re-rendered in full at once. It now gets its
own footprint kind, `FOOTPRINT_STREAMED`: with a budget the pages keep their depth and get `REFRESH_FULL` (a full
render that drops their static layer), queued like the others (with the highest distance: a new cut can move a
silhouette by anything); without a budget they are freed as before. Static casters that were *moved* still free their
pages at once (an edit must show immediately), as do the wind being switched on or off and footprint overflow.

Settings: `RenderSettings::shadow_refresh_budget` (96; `--shadow-refresh-budget N`, 0 = all at once, the editor's
*Shadow page refreshes per frame*), `shadow_max_staleness` (4; `--shadow-max-staleness N`). Statistics: the draws
buffer header's free words 6, 7, 11 carry the deferred, held and forced counts (`RenderStats::shadow_refresh_*`, a HUD
line, the headless `frames a..b mean` report).

Why 96 pages and 4 frames: the forest's wind flags ~157 pages a frame; 96 refresh every flagged page on average every
1.6 frames, and none reached the 4-frame bound in any run. The other demo scenes flag far fewer (valley 1.5, cathedral
~35 a frame) and are not throttled at all — the budget only engages where many casters move. 4 frames is 67 ms at
60 fps, under TSR's history length, so a lag of a texel or two per frame is smeared rather than seen.

### Distant levels at their own rate (`vsm-prefiltered-distant`)

Each clipmap level now has its own light view (`Vsm::level_light`, 16 more matrices in the uniform): the sun's,
except for the distant levels (from `shadow_distant_level`, 7 by default), which keep the direction they were rendered
for (`VirtualShadowMap::level_sun`) and take the current one only on their turn — every `shadow_distant_interval`
(4) frames, staggered by level (`(frame + level) % interval == 0`, so with five distant levels about one re-renders a
frame) — or at once when they lag more than 0.5° (the sun dragged with the arrow keys or the editor). The page cache's
`invalidate` became a bit mask of levels, so only the levels that re-aimed are re-rendered. Everything that addresses a
level uses its own view: `vsm_address` (marking and every sampler), the culling views (each level's `CullViewUniform`
carries its light view and planes), the expansion, the hardware / software / voxel / tessellation page rasterisers,
and the invalidation, which shifts a footprint's light-space centre by the difference between the sun's light space
and the distant level's. Filter kernels keep the sun's axes (a fraction of a degree apart).

Wind refreshes of distant pages likewise wait for their level's turn (`refresh_held`; counted as *held*), but in the
demo scenes they hardly occur: with 12.5 cm texels and the 2-texel threshold, a distant page is only refreshed for
sway larger than 25 cm.

Why level 7, every 4 frames: at `P`'s 0.08 rad/s the sun turns 0.3° in 4 frames, at `J`'s time-lapse about as much;
a 20 m tree's shadow then moves by ~10 cm — under one 12.5 cm texel of level 7, where receivers are beyond ~50 m at
1600×900 (60° field of view). A lower first level would save more but put the seam where it could be seen; level 6
(6 cm texels) would show a 1.6-texel step at that speed.

**What is not done: prefiltering.** UE's Prefiltered Distant renders distant casters into a lower-resolution,
prefiltered representation (to avoid aliasing and shimmering of thin distant geometry) with temporal reprojection.
Helios's distant levels render at full resolution, only less often; no moment / ESM representation, no reprojection
(a lagging level is exact for the direction it was rendered for, so it needs none). A prefiltered representation
would need a second atlas format and a filtered lookup path in every sampler; the measured cost was in the re-rendering
under a moving sun, which the rate alone removes.

### Shadow overview (`vsm-overview-debug`)

`DebugView::ShadowOverview` (26): a compute pass (`vsm_overview.rs`, `vsm_overview.wgsl`) after the lighting pass
overwrites the HDR target with four panels, each the frame at half size (pixel `2q` of the frame for panel pixel
`q`): (1) clipmap level and page grid, shaded by the shadow, distant levels hatched, magenta where nothing is
resident; (2) the page's work this frame from the info buffer's page state (full / dynamic / split), pending refresh
flags (deferred, held) and the frames waited; (3) the reason of its last invalidation, a new per-physical-page word
`INFO_PAGE_REASON` (`HIT_*` bits | frame << 8) written by `invalidate_pages`, fading over 60 frames, white for pages
rendered in full without a footprint (new, or the light moved); (4) the physical pool as a 32 × 32 grid (owner's level
or light colour, brightness by cached / rendered, a dot for a pending refresh) and five bars (requests vs. pool with
the unallocated part in red, retained pages, pages rendered vs. the budget, deferred / held / forced, sun vs. local
light pages). The pipeline is `lazy_compute_pipeline` (compiled when first selected) and the pass is recorded only
while the view is selected — the same pattern as the invalidation view (16) — so it costs nothing when off; no hot
shader got a branch. `2` toggles the budget and the distant levels together for A/B comparisons.

## Alternatives considered

* **Budget the page cache's dirty list too** (new pages, light moved). A new page has nothing to show, and a sun
  direction change invalidates every page at once; deferring those would show holes or wrong shadows. The distant
  levels' rate handles the sun instead.
* **Rank by clipmap level** (fine levels first). Simple, but a fine page whose grass swayed 2 texels would beat a
  coarse page under a falling tree. Texels moved is the actual error and covers local light pages too.
* **Rank by screen coverage** (pixels that sample the page, UE's importance). The marking pass would have to count
  requests per page (an atomic per pixel); texels moved already folds in the level and the motion. Left out — noted
  in status item 7.
* **A FIFO / round robin over pending pages** (no priority): bounded but ignores how far off a page is.
* **Sun motion: per-level angular quantisation** (distant levels snap the sun to a coarser angle grid): every distant
  level would re-render in the same frame at each step — a spike. Staggered turns spread it.
* **Temporal reprojection of distant pages under the new sun** (UE 5.8): unnecessary while each level is sampled
  with its own light view, which keeps its pages exact for the direction they were rendered for.

## Measurements

RTX 3090, 1600×900, default views, 240 frames per run (means and minima over the last 160), GPU shared with other
agents' builds and renders: **minima of three interleaved off / on runs**; *off* = `--shadow-refresh-budget 0
--shadow-distant 0` (round 7's behaviour), *on* = the defaults. "Shadow passes" is the sum of the `vsm *` passes'
minima.

| Scene, view | pages rendered / frame, off → on | on: deferred / held / forced | shadow passes (min), off → on | `vsm raster` + `vsm raster sw` (min) | GPU frame median (best of 3) |
|---|---|---|---|---|---|
| forest, default view (wind) | 156.8 → 96.0 | 71.6 / 0 / 0 | **1.33 → 0.85 ms** | 0.95 → 0.48 ms | 9.19 → 8.83 ms |
| forest, `--sun 60,10` (low sun, long shadows) | 130.1 → 96.0 | 40.7 / 0 / 0 | **1.16 → 0.88 ms** | 0.78 → 0.50 ms | 9.00 → 8.46 ms |
| forest, `--animate-sun` (`P`) | 240.8 → 186.0 | 0.6 / 0 / 0 | **5.90 → 2.53 ms** | 5.24 → 2.03 ms | 15.02 → 11.47 ms (frame min 13.39 → 9.71) |
| valley, default view | 1.5 → 1.5 | 0 / 0 / 0 | 0.21 → 0.22 ms | 0.01 → 0.01 ms | 4.83 → 5.10 ms (noise; frame min 4.71 → 4.71) |
| cathedral, default view (swinging chandeliers) | 35.2 → 35.2 | 0 / 0 / 0 | 0.39 → 0.40 ms | 0.04 → 0.04 ms | noise (other agents' load) |

* The forest's wind: the budget binds (157 pages flagged a frame, 96 refreshed); no page reached the 4-frame bound
  (*forced* 0). The shadow page passes drop by a third; the hardware and software page rasters, which scale with the
  pages refreshed, by half. With a low sun fewer pages are flagged (130) and the saving is smaller.
* A moving sun: the distant levels' turns cut the pages rendered from 241 to 186 a frame, but the shadow passes by
  57 %: the distant pages are the expensive ones (each holds whole stands of trees, software-rasterised). The wind's
  distant refreshes (*held*) never occurred in these views.
* Scenes in which fewer pages move than the budget (valley, cathedral with its swinging chandeliers: ~35 pages) are
  unchanged apart from the queue's two extra dispatches: `vsm dirty info` 0.013–0.022 → 0.025–0.032 ms (~+0.01 ms).
* Images: the forest's last frame, on vs. off, differs by a mean |Δ| of 0.63 / 255 (2.2 % of pixels by more than
  8) — as much as two runs with the budget off differ (0.58, 2.1 %: the fire and particles are not deterministic run
  to run). In stills the budget is invisible; in motion a deferred shadow lags its swaying caster by one to four
  frames, under TSR's history.
* The overview itself, when selected: 0.25–0.26 ms at 1600×900 (forest, cathedral); nothing when off.
* Memory: the uniform grows by 1040 bytes (the per-level light views), the info buffer by 4.3 KiB (reasons and the
  queue); the overview allocates nothing.

Images: [`vsm/`](vsm/) — `overview-forest.jpg` (the overview in the forest's default view: the budget's deferred pages
blue in panel 2, the wind's invalidations in panel 3, the pool and its bars in panel 4), `overview-sun.jpg` (with `P`:
everything near rendered in full every frame, the distant trees cached), `overview-cathedral.jpg` (the pool
full of local light pages, the chandeliers' moves in panel 3), `forest-off-vs-on.jpg` / `sun-off-vs-on.jpg`
(crops of the lit image with and without, same frame), `browser-overview.jpg` (the overview in headless Chromium).

## Remaining limits

* Where the budget binds, a swaying or moving shadow lags its caster by up to `shadow_max_staleness` frames; UE keeps
  WPO-animated pages current and throttles only Nanite's LOD invalidations. The priority ignores how many pixels
  sample a page.
* Distant levels lag a moving sun by up to 4 frames or 0.5°; there is a seam where a distant level meets the level
  before it (below a texel at the demo's sun speeds, visible if the interval is raised a lot).
* No prefiltered distant representation and no temporal reprojection (UE 5.8's Prefiltered Distant); distant casters
  render at full resolution, so thin distant geometry aliases as before.
* The overview's panels are colours and bars without text; per-pixel panels cover the sun's clipmap only (local light
  pages: debug view 15).
