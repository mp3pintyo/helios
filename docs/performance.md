# Performance measurements

Back to the [README](../README.md).

**Measurements.** Unless stated otherwise, costs are GPU times at 1600×900 on an RTX 3090. Those added in round 4
were taken while the GPU was shared with up to six other agents' builds and renders: they are minima (medians where
stated) of interleaved on/off runs, whole-frame totals were at times swamped by the other load, and per-pass minima
are the reliable numbers. The per-feature notes in [`docs/round4/`](round4/) hold the full tables. Default views
at 2560×1440, 100 % / 67 % screen percentage, round 3 → round 4 (final build on an idle GPU, minimum of 3 interleaved
runs; [`perf-regressions.md`](round4/perf-regressions.md)): valley 8.1 / 5.6 → 8.7 / 5.5 ms, cathedral
10.9 / 6.9 → 10.3 / 6.4 ms, forest 14.5 / 11.0 → 18.3 / 12.2 ms (the campfire's shadow-casting light and Nanite
Foliage's assemblies); new in round 4: gallery 8.0 / 5.0 ms (with the graph-displaced brick terrace in view), cathedral
with 300 lights under MegaLights 15.3 / 9.2 ms. Round 5 ([`docs/round5/`](round5/); final build on an idle GPU, minimum of 3 interleaved runs): valley
8.8 / 5.5 ms, cathedral 10.7 / 6.7 ms (the swinging chandelier lights have shadow maps again, cached), forest
17.0 / 11.3 ms (local light pages), gallery 8.0 / 5.0 ms, cathedral with 300 lights under MegaLights 15.2 / 9.0 ms.
Round 6 ([`docs/round6/`](round6/); final build on an idle GPU, minimum of 3 interleaved runs against round 5):
valley 8.7 / 5.4 → 8.8 / 5.5 ms (Lumen's final gather sees the displaced boulders), cathedral 10.8 / 6.6 → 10.9 /
6.7 ms (an eleventh swinging chandelier light gets a shadow map), forest 16.9 / 11.4 → 15.8 / 10.6 ms (assembly
culling), gallery 8.0 / 5.0 → 8.1 / 5.0 ms, cathedral with 300 lights under MegaLights 15.0 / 9.0 → 15.2 / 9.0 ms.
Round 6 follow-ups (browser start-up, Web Worker bakes, per-texel card albedo, the candle redraw budget and the page
pool's priorities; same method, against round 6): valley 8.7 / 5.4 → 8.7 / 5.5 ms, cathedral 10.8 / 6.7 → 10.8 / 6.7
ms, forest 15.8 / 10.5 → 15.9 / 10.6 ms, gallery 8.0 / 5.0 → 8.1 / 5.0 ms, cathedral with 300 lights without
MegaLights 16.0 / 9.7 → 14.7 / 8.8 ms (no candle left without a shadow page), with MegaLights 15.0 / 9.0 → 15.1 /
8.9 ms, the lantern 11.4 / 7.3 → 11.4 / 7.3 ms. In the browser the first image of a fresh visit takes ~8 s in
Chromium and in Firefox (was 31–50 s), Firefox's full set ~33 s; the displaced-variant bakes' worst frame 718–920 →
6–31 ms.
Round 8 grounded foliage ([`foliage-grounding.md`](round8/foliage-grounding.md); 1600×900, shared GPU, per-pass minima):
forest default view 8.87 → 9.72 ms (plant data and stalks +0.3–0.4 ms of shadow rasterisation and resolve,
short-range AO +0.15 ms of lighting).
Round 8 terrain and rivers ([`terrain-water.md`](round8/terrain-water.md); 1600×900, shared GPU, minima of 3 interleaved
runs with and without the mesh terrain and river): valley initial view 4.94 → 5.49 ms (river surface +0.34 ms of
translucency, mesh terrain's per-pixel triplanar layering +0.10 ms of resolve, simulation 0.04 ms), river close-up
(`--eye -2,3,-6 --at -9,0,-20`) 4.31 → 5.70 ms (translucency 0.92 ms), arch view 3.88 → 3.85 ms. Building the
mesh terrain: ~2 s natively, 2.8 s in Chromium (`--detail medium`, scene 7.4 s in all).
Round 8 dynamic wind ([`dynamic-wind.md`](round8/dynamic-wind.md); 1600×900, shared GPU, minima of interleaved runs):
forest default view 8.45 → 9.40 ms (8.91 with `--no-wind-bones`: bone data in the clusters and larger instance
records), fern close-up 6.60 → 7.58 ms, grass close-up 8.48 → 9.41 ms; the bones themselves cost ~0.5–0.7 ms (per-vertex chains in the rasterisers,
the resolve and the shadow pages). Scenes without plants are unchanged (cathedral within ±0.02 ms per pass).
Round 8 wind per instance ([`windperf.md`](round8/windperf.md); 1600×900, shared GPU, per-pass minima of 3 interleaved
runs): forest default view 9.32 → 9.07 ms with the bones (pre-wind build 8.73), 8.93 → 8.54 ms with `--no-wind-bones`
(the passes placing geometry back at the pre-wind cost); fern and grass close-ups −0.36 / −0.35 ms with the bones
(shadow page raster −0.40 / −0.26 ms), −0.15 / −0.25 ms without; cathedral 6.66 → 6.39 ms (resolve 0.51 → 0.40 ms).

**Round 8 close** ([`perf8.md`](round8/perf8.md); idle GPU, 2560×1440, 100 % / 67 %, minimum of 3 interleaved runs,
round 7 → round 8 before the close → **round 8 final**): valley 8.79 / 5.54 → 10.69 / 6.63 → **10.42 / 6.46** ms,
cathedral 10.87 / 6.68 → 12.05 / 7.38 → **11.48 / 7.27**, forest 16.35 / 10.69 → 16.71 / 11.49 → **16.29 / 11.34**,
gallery 8.18 / 5.05 → 9.06 / 5.53 → **8.66 / 5.29**, cathedral with 300 lights 15.06 / 8.94 → 16.59 / 9.66 →
**15.69 / 9.21**, with MegaLights 15.33 / 9.04 → 16.34 / 9.50 → **15.88 / 9.34**, the lantern 11.62 / 7.25 → 12.92 /
7.97 → **12.27 / 7.76**. Where round 8's time went (100 %): the short-range AO inline in the lighting 0.30–0.61 ms
(more than its taps: it held the lighting's registers); the Lumen world cache 0.27–0.65 ms, of which the final
gather and the cards' lighting give back 0.12–0.97 ms; local lights through glass 0.12 ms of lighting (0.27 with 300
lights) + 0.07–0.11 ms of maps; AVBOIT's two composite draws a bucket +0.10 ms; the material resolve +0.16 (cathedral
standard model: plant-data decode and register pressure), +0.40 (forest: plant bones in foliage and wood), +0.28 ms
(valley: the mesh terrain's per-pixel triplanar layers and river shores); the valley's river 0.69 ms of
translucency + 0.09 of scene colour (its shading 0.61 ms, of which SSR 0.28 for all water, caustics 0.09); the
forest's plant bones in the rasterisers +0.4 ms and its shadow pages −0.87 ms (moved to the software path); the
MegaLights froxel volume 0.69 ms (−0.23 ms of fog). Recovered: the short-range AO at half resolution in a pass of its
own (0.06–0.10 ms; lighting −0.28…−0.71 ms), world-cache paths capped at 24 000 a frame (−0.07…−0.15 ms), AVBOIT
composites skipping empty pixels (−0.04 ms), glass-light colours read only behind a pane (−0.05 ms in the cathedral).

**Round 9 quality levels** ([`quality-levels.md`](round9/quality-levels.md); Chromium, 1600×900, default `auto`
start).
* iGPU: AMD Radeon Graphics of a Ryzen 5 7600X, 2 RDNA 2 CUs, through `--use-adapter-luid`. GPU per frame and frame
  rate, before → `low`:
  * valley 637 ms / 2.5 fps → 78 ms / 15 fps;
  * cathedral 820 / 1.9 → 82 / 12;
  * forest 814 / 1.7 → 169 / 6.7;
  * gallery 599 / 2.2 → 54 / 20.
* Lumen's surface cache and radiance probes were the cost left at `low` that does not shrink with the resolution.
  Relit every 4th frame, it went from 25–27 to 4.6–5.9 ms.
* RTX 3090: `auto` = `high`, 8.8–12.3 ms; a forced `low` takes 5.1–8.0 ms.

**Round 9 Firefox frame budget** ([`browser.md`](architecture/browser.md#firefoxs-frame-budget-round-9-firefox-frame-budget);
Firefox 1543 (Playwright), RTX 3090, 1600×900, the Opus55 site's gallery at `high`). Firefox does not overlap
frames, so its frame rate depends on the GPU budget:

| GPU budget | GPU time | Frame rate |
|---|---|---|
| 14 ms (the old default) | 12.5 ms at 100 % | 27 fps, GPU 40 % busy |
| 10 ms | ~9 ms | 46–60 fps |
| 8 ms (Firefox's new cap) | ~7 ms at ~62 % | 60 fps |

**Round 9 held pipeline groups** ([`browser.md`](architecture/browser.md#only-the-features-a-level-uses-are-compiled-round-9-compile-only-enabled-features);
Firefox 1543, RTX 3090, the Opus55 site's gallery):

| `low` | Before | After |
|---|---|---|
| First image | 28.7 s | 17.9 s |
| Every pipeline compiled | 53.4 s | 31.7 s |
| Pipelines | 212 | 152 |

`high` is unchanged: 19.0 / 42.3 s before, 19.8 / 43.2 s after. After a switch from `low` to `high` the missing
pipelines compile in ~24 s in Firefox (no frozen canvas) and ~5 s in Chromium.

**Round 9 Firefox still compile** ([`browser.md`](architecture/browser.md#firefox-the-canvas-holds-still-while-it-compiles-round-9-firefox-still-compile);
Firefox 160, RTX 3090, the Opus55 gallery at `high`):
* `compileView: "live"`: first image 21.9 s, every pipeline 46.5 s.
* `"still"` with frame-counted bursts: 20.9 s and 49.1 s.
* `"still"` with 0.6 s bursts: 57.6 s.
* Firefox 160 test page, 8 pipelines of ~530 ms: compiled serially, also asynchronously; 0 GPU frames completed
  meanwhile; a second visit with the same profile was no faster.
