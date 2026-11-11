# Local lights, their shadow maps and MegaLights

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Local lights** (`lights.rs`). Point and spot lights are culled into 16×9×24 froxels of up to 255 lights each
  (63 before MegaLights). Lights are also injected into the surface cache, so they bounce. Culling is hierarchical
  (round 8, [`lights.md`](../round8/lights.md)): one workgroup per coarse cell of 4 × 3 tiles × 4 slices streams the
  lights through shared memory 64 at a time, tests each against the cell once and only the survivors against the
  cell's 48 froxels, in index order (the same lists as brute force): 0.25 → 0.07 ms at 1000 lights, 0.08 → 0.03 at
  300.
* **Local light virtual shadow maps** (`vsm_local.rs`, `vsm_local_common.wgsl`, `vsm_local_sample.wgsl`). As in UE5,
  up to 32 shadow-casting point and spot lights per frame — the most influential on screen (range sphere coverage ×
  brightness, with hysteresis so slots and their pages stick), static lights first: a light that moved within the
  last 30 frames takes a slot they leave over, and keeps its map cached around an origin near it (*Cached shadow
  maps for moving lights* below; under MegaLights, which traces every light's shadow anyway, it gets none) — get
  virtual shadow maps: six cube faces for a point
  light, one perspective face for a spot light, each a 4096² virtual map with a mip chain down to one 128² page.
  Their pages are entries of the sun's page cache after its clipmap levels and live in the same physical atlas, so
  the sun and all lights share one pool, one allocation, one dirty list and the same page renderer (the (cluster,
  page) expansion and the hardware and software page rasterisers take a perspective projection per face; the hardware
  path clips at the light's near plane, software items must lie in front of it). Every lit pixel requests, for each
  light of its froxel with a shadow map, the page of the mip whose texels match its screen footprint (half a mip
  coarser by default, `local_shadow_resolution_bias`; lights with a large source radius coarser still) plus the
  coarsest page of that face; volumetric fog requests coarse pages. Pages stay cached while the map's origin and the
  geometry are static: a new origin or shape, a scene change or a slot handed to another light drops that light's
  pages. Streaming, moves and skins refresh local pages like the sun's ([*Wind*](motion.md), [*Dynamic world*](motion.md)), in each light's
  terms: an instance's footprint refreshes the pages it projects onto in a light's faces and mips, once it may have
  moved 2 of their texels at its depth inside the face. The wind's refreshes are scheduled by what a page holds
  instead (round 5): an instance's bounding sphere says little about a light inside or next to it — a campfire among
  trees refreshed every page of its light every frame, at the crowns' speed — so whenever a local page is rendered,
  its expansion records how fast and how far the wind can move the clusters actually drawn into it, in the page's
  texels at each cluster's depth (two words per physical page in the info buffer), and the page is refreshed once
  they may have moved 2 texels: pages with near ferns and grass every frame or two, trunks and distant crowns every
  few frames or never; only their swaying casters are re-rendered, over the cached static layer. A page under a
  caster entering or leaving the wind, and every local page when the wind itself changes, measures again. A
  sphere reaching a cube face's near plane no longer covers the whole face (`local_sphere_rect` bounds the part in
  front of the plane: wholly to one side of the face axis, it projects outside the face or onto a band along one
  edge), which most clusters around a light did for some face: forest 198 k → 137 k page draws a frame (local
  lights' share 86 k → 20 k), GPU frame 18.4–18.5 → 16.7–17.6 ms at 2560×1440 (`vsm raster` 3.6–3.9 → 2.2 ms),
  67 %: 12.05 → 11.33 ms. One
  culling view serves all lights: it keeps clusters that overlap a world box around some light's dirty pages and
  selects LOD for the most demanding light's finest dirty mip, so a frame without new pages costs the same with 1 or
  32 lights. A pass of its own computes the shadows of each pixel's eight strongest shadow-mapped lights into a
  screen-space mask (4 bits each, dithered for TAA; as UE projects shadows into a mask), so the filter's cost and
  registers stay out of the lighting: contact-hardening soft shadows for the four strongest (PCSS: a blocker search
  over the light's source radius sets the filter radius, taps slide onto the receiver plane, and taps inside the
  centre's page reuse its lookup), one bilinear comparison for the next four and, in the lighting, for any further
  ones. Translucent surfaces filter them inline, volumetric fog takes one tap per light, so shafts fall between
  pillars and through the choir screens. Lights beyond the budget keep cone-traced distance-field soft shadows for
  the strongest few per pixel (mesh SDFs within 2 m of the receiver and of the light, the global field between).
  Debug view 15 shows the sampled mip and page grid of each pixel's strongest shadow-mapped light.
  **The page pool serves what matters first** ([`vsm-pool-fairness.md`](../round6/vsm-pool-fairness.md)). The sun
  and the 32 maps share 1024 pages, and the candle-lit cathedral requests ~1860 a frame. The cache placed requests
  first come, first served — in entry order, so the slots with the highest indices got none: the six candelabrum
  candles nearest the camera held slots without a single resident page and cast no shadow. When the requests do not
  fit, the page cache now serves them by priority buckets (`page_cache.wgsl`: each mapped light's coarsest page per
  face, then the sun's pages, then the local lights' pages by their light's rank on screen, coarse mips first) and
  evicts resident pages below the cut instead of leaving the better requests without a page; and a pool resolution
  bias (`PoolPressure`, UE's virtual shadow map dynamic resolution) makes the local lights request coarser mips while
  the requests exceed 95 % of the pool — spread by rank, none for the most influential light, twice the bias for the
  faintest — and relaxes below 70 %. Candle-lit view at 2560×1440: 963 pages requested and none
  unallocated (before: ~1860, 840), no map without a page, bias 1.46 mips; the image matches a 2304-page pool's
  within noise (60.2 dB; before 43.0 dB) — what the faintest lights lose falls on receivers where their coarse pages
  are as fine as the screen. The six candles cost their filtering again (`local light shadows` +0.37 ms); the 33-light
  cathedral (55 pages short) and the lantern (94) now fit too, at 0.2 and 0.34 mips, images within noise.
  Slot order matters with moving lights: in the candle-lit cathedral (300 lights, MegaLights) the swinging
  chandeliers' candles took slots and re-rendered ~970 pages a frame (page raster 11–15 ms); static lights first cut
  that to 713 dynamic-only refreshes (0.11 ms), but near the altar six swinging candles still took the slots left
  over (11.6 ms), hence no slots for moving lights under MegaLights (0.06 ms). In the clustered loop the default
  cathedral's eleven swinging chandelier lights take 10 spare slots; re-rendering their cubes every frame cost
  ~80 full pages and 3.1 ms of `vsm raster` at 1440p, which round 4 avoided with distance-field shadows (42 dB PSNR
  against the shadow maps: the soft bands under the pier capitals fainter).
  **Cached shadow maps for moving lights** (`vsm_anchor.rs`, round 5,
  [`local-shadow-cache.md`](../round5/local-shadow-cache.md)). A light is not a point: its penumbra spans the
  shadows cast from every point of its source, so a moving light's map keeps the *origin* it was rendered from while
  the light stays within a tenth of its source radius of it (at least 1 cm). The pages keep their static layer; the
  casters moving with the light (a chandelier's ring, candles and chain) are redrawn into them at their current pose
  by their move footprints, so a fixture never shadows itself with a stale pose. Receivers correct the parallax: an
  occluder at 1 / t of the receiver's distance casts its shadow (t − 1) · (light − origin) away from where the held
  map has it, and the PCSS filter, which knows t from its blocker search, moves its taps by as much (slid onto the
  receiver's plane; nothing at contact) — shadows follow the light every frame, and a re-render moves nothing
  visible. When the light leaves the tolerance its requested pages are rendered from a new origin placed 0.75
  tolerances ahead on its path (a chandelier light: every ~60 frames), the furthest out first; a light gets a map
  only if the budget can keep up with its recent top speed, and one that outruns it keeps distance-field shadows
  until it rests, so lights don't flip between the two. A moving light's map is one mip coarser (UE's
  `ResolutionLodBiasLocalMoving`). Round 6 ([`fast-light-shadows.md`](../round6/fast-light-shadows.md)) made the
  budget fit fast, sharp lights: **every cube face has its own origin** and is re-anchored on its own, the budget is
  **pages per frame** (`RenderSettings::moving_light_pages`, 32), a face costing the pages the view requests of it
  (counted on the GPU by `dirty_info`, read back per face), so a face nobody looks at costs its one fallback page and
  six faces due together spread over the next frames; a light that crosses its tolerance sphere every frame is
  rendered every frame **from where it is** (the lead shrinks with the speed, to none); one the budget cannot follow
  at its tolerance is held at twice or four times it rather than given up; and a moving light competes for a slot at
  up to half its influence instead of only for the slots static lights leave — less the more its map costs per frame
  ([`candle-redraw-budget.md`](../round6/candle-redraw-budget.md)): casters moving with a light (its own fixture)
  refresh its pages every frame outside the page budget, so the GPU counts the pages each slot renders and requests
  (resident or not: a map the full pool starves is not free), and the claim is `0.5 / (1 + (pages a frame / 4)²)` of
  the influence. A chandelier light at its ring's centre (~3.5 pages a frame) keeps 0.28 of it; a candle flame inside
  its swinging fixture redraws its whole map, 20–50 pages a frame, and keeps under 2 %, so in the candle-lit
  cathedral's classic loop the static candles keep the slots as in round 5 (shadow pages ~140 → 14 a frame, GPU
  frame 15.29 → 14.22 ms at 2560×1440: `vsm raster` + expand 0.27 → 0.10 ms, marking 0.36 → 0.21, lighting
  4.73 → 4.22 — round 5's level; 39.3 dB against the every-frame reference instead of 43.4), while the lantern,
  hundreds of times more influential than the faint candles holding the last slots, still takes one. The lantern
  of `--lantern` (a 2 cm flame at
  up to 1.6 m/s) is re-rendered every frame, ~17 pages at the moving mip bias (the reference at full resolution:
  33): 44.8 dB against the every-frame reference — all of it the mip bias — where its distance-field shadows were
  33.9 dB; with the chandeliers swinging too, all twelve moving lights keep their maps for 1000 frames (round 5:
  the lantern took the one render a frame and 8 chandelier lights fell back to distance fields), ~28 pages
  a frame rendered from new origins against 161 re-rendering every moving light every frame. HUD: `moving lights`
  line. Cathedral, 2560×1440 (round 5): GPU
  frame 10.27 ms (distance fields, round 4) → 10.77 ms, against 13.75 ms re-rendered every frame (lighting +0.22 ms
  for ten more mapped lights, marking +0.08, page raster +0.06, culling and expansion +0.10; ~42 dynamic-only pages
  a frame); image against the every-frame reference 48.9 dB (distance fields 42.2 dB), from under a chandelier
  42.9 dB (34.4 dB), no added flicker (0.0063 % of pixels over 24/255 per frame in all three modes).
  `RenderSettings::moving_light_shadows` selects `Cached`, `DistanceFields` or `EveryFrame`.
* **MegaLights** (`megalights.rs`, `megalights_sample.wgsl`, `megalights_shade.wgsl`, `megalights_denoise.wgsl`,
  `megalights_common.wgsl`, `megalights_surface.wgsl`; UE 5.5's MegaLights). `RenderSettings::mega_lights` (`=`)
  replaces the clustered light loop — which can shadow only the 32 shadow-mapped lights plus four distance-field
  traced ones per pixel and lets every other light leak through walls — by a fixed budget per pixel in which *every*
  local light casts soft shadows. **Sampling**: per 2×2 block, at a representative pixel that moves every frame, four
  lights are drawn from the froxel's light list in proportion to their unshadowed contribution through the shared BRDF
  (`brdf.wgsl`, the same functions the classic loop calls), by systematic resampling over at most 32 candidates;
  longer lists are thinned to a stratified subset in which every light has the same chance, its weight scaled by the
  inverse, so the estimate stays unbiased and the cost per pixel constant. A small floor keeps lights behind the
  representative's normal possible for the rest of its block. **Visibility guiding**: per 8×8 tile, bit masks (light
  index mod 256) of the lights some ray was traced to and of those some ray reached; a light whose rays were all
  blocked in last frame's (reprojected) tile keeps a tenth of its weight. **Shadow rays**: one per sample, to a random
  point of the light's disk (the source radius makes the penumbra): a 1 m screen trace against the depth buffer
  (contact shadows), then over the whole ray the light's virtual shadow map where it has one — shadow-map ray marching
  (UE's SMRT, 8 steps; the first is the receiver's classic test, the later ones see occluders off to the side as the
  ray converges on its point of the light, within an assumed thickness) — otherwise Lumen's distance fields (mesh SDFs
  within 2 m of either end, the global field between, narrow cones). The screen trace treats the depth buffer as 3 cm
  shells (a thicker assumption shadowed walls behind any object in front of them) and ignores emissive pixels (candle
  flames are in the depth buffer but cast no shadow). **Shading**: every pixel evaluates the BRDF of its block's four
  samples (Substrate slabs through a variant) with their unbiased contribution weights and visibilities and writes
  its direct light *demodulated* (diffuse without the diffuse colour, specular divided by the environment BRDF), so the
  denoiser never blurs texture detail. **Denoising** (SVGF): temporal accumulation with reprojection, depth-checked
  disocclusion, a 2.5σ variance clip and up to 24 (diffuse) / 12 (specular) frames, with luminance moments; then four
  à-trous iterations (3×3 binomial, strides 1–8) with depth-plane, normal and luminance edge-stopping scaled by the
  variance of the accumulated mean (so converged pixels keep normal-mapped lighting detail); the last one modulates
  the result back into one texture, which the lighting pass adds instead of running its loop. Debug views **18**
  (each sample's light, one hue per light, dark where blocked) and **19** (this frame's samples | denoised).
  Cost (cathedral medium at night, shared GPU, minimum of 4 interleaved runs; "direct" = light culling + local light
  shadow mask + lighting + MegaLights):

  | Lights | Classic direct | MegaLights direct | MegaLights passes | Classic lighting pass | Frame (classic → ML) |
  |---|---|---|---|---|---|
  | 33 (default scene) | 1.03 | 1.84 | 1.37 | 0.88 | 6.38 → 7.15 |
  | 100 (`--lights 100`) | 2.45 | 2.52 | 1.95 | 1.95 | 8.49 → 8.60 |
  | 300 | 2.78 | 2.54 | 1.99 | 2.15 | 9.03 → 8.71 |
  | 1000 | 3.53 | 2.74 | 1.95 | 2.50 | 10.55 → 9.75 |
  | 1000, aisle view | 3.63 | 2.61 | 1.86 | 2.52 | 10.11 → 9.08 |

  At 33 lights the MegaLights passes split into sample 0.16, trace 0.40, shade 0.08, temporal 0.14, spatial 0.59 ms;
  from 100 to 1000 lights only the share of distance-field rays grows, and the lighting pass itself drops to 0.46 ms
  — while the classic path shadows only 36 of those lights per pixel. Image vs the classic path at 33 lights: 39.7 dB
  PSNR (nave), 39.8 dB (candelabrum close-up; two classic runs differ by 66.7 dB); the differences are shadows the
  classic PCSS blurs away. Static camera, pixels changing by more than 24/255 frame to frame: nave 0.0093 → 0.0091 %,
  close-up 0.0017 → 0.0009 %, 1000-light aisle 0.0005 → 0.0007 % (classic → MegaLights). Browser (headless Chromium,
  1280×720, 300 lights): 1.2 ms. Memory (only while on): 56 bytes per internal pixel (two rgba16f raw, two sets of
  rgba16f diffuse + specular + r32uint history) plus 36 bytes per 2×2 block of samples.

  **Round 8** ([`lights.md`](../round8/lights.md)). *Shared rays outside penumbrae* (UE 5.8): the guide masks also
  record per tile which lights had a ray blocked; a light picked twice in a block whose rays in the tile all agreed
  last frame (all reached it or all were blocked) traces one ray for both picks (the header stores which sample's
  ray each uses), and `generate` compacts the remaining rays per workgroup into a list that sizes the trace's
  indirect dispatch: 17 % fewer rays at 33 lights (trace 0.10 ms faster), 6–7 % at 300–1000
  (`mega_lights_shared_rays`, `--ml-no-shared-rays`). *HZB screen traces*: the contact-shadow trace walks a
  closest-depth HZB of the final depth buffer (`hzb_trace.wgsl`, reusable; `ClosestHzb` in `passes/hzb.rs`, 0.027 ms
  to build) instead of eight linear steps, so it sees every pixel the ray crosses (+0.03–0.05 ms;
  `mega_lights_hzb_traces`, `--ml-linear-traces`). *Importance cull* (`mega_lights_importance_cull`, 0.01): each
  light's reach in the grid ends where its radiance at last frame's exposure falls below the threshold (UE's
  attenuation falloff; distant specular of culled lights is lost) — in the cathedral, whose candle ranges already
  end where they are visible, it culls nothing at the default. *Volumes*: volumetric fog and translucent surfaces
  take their local light from a froxel volume MegaLights samples and shadows (`megalights_volume.rs`, on the fog's
  16 px × 64-slice grid): per froxel and frame one light drawn by its unshadowed radiance at a jittered point, one
  shadow ray (`megalights_rays.wgsl`: the same shadow-map march or distance-field cones, pages four mips coarser),
  blended 10 % into the reprojected history; the fog adds its in-scattering (phase towards the camera) instead of
  its loop, translucent surfaces shade its SH bands 0–1 as a directional part through the BRDF plus an ambient rest
  (`mega_lights_volume`, `--ml-no-volume`, **Home**). 0.24–0.31 ms, of which the fog (−0.04 to −0.16) and
  translucency loops (−0.01 to −0.07; aisle glass at 1000 lights 1.06 → 0.78) win back much; 17.5 MB at 1600×900.
  MegaLights frame at 300 / 1000 lights 7.78 → 7.66 / 8.69 → 8.52 ms.
