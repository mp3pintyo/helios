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
