# Temporal stability, TSR, camera, exposure and profiling

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Temporal stability.** The renderer coarsens the LOD threshold when visible work nears the culling list
  capacities, instead of letting the GPU drop arbitrary clusters (which flickers). Lumen clamps ray radiance relative
  to the exposure (as UE's `MaxRayIntensity`: tiny bright emitters such as candle flames otherwise tint random probes
  each frame), places adaptive probes where no uniform screen probe matches a surface (and falls back to the radiance
  cache, sky visibility included, where none does), never clips GI history tighter than ±25 % (probe noise is spatially
  coherent, so a 3×3 neighbourhood underestimates it), filters reflections with
  Karis weights and variance clipping, and traces distance-field shadows of the strongest lights, stopping short of
  each light's own sub-voxel fixture. TSR widens its clip box where the image is still, so sub-pixel foliage averages
  instead of flickering. `helios --dump-dir` + a frame-difference script measure all this (a static camera changes
  ≤ 0.02 % of pixels by more than 24/255 in the valley and cathedral).
* **Temporal super resolution** (`passes/tsr.rs`, `tsr.wgsl`, `resolution.rs`; UE5's TSR and `r.ScreenPercentage` /
  `r.DynamicRes`). `RenderSettings::screen_percentage` renders the scene at 25–100 % of the output; TSR reconstructs
  the output from jittered internal frames. Internal targets are allocated for the largest percentage the settings
  allow and a frame renders into their top-left corner, so dynamic resolution never reallocates: shaders that sample
  internal targets by screen uv map it into that corner (`viewport_uv` in `common.wgsl`), histories at internal
  resolution (GI, reflections, clouds) are reprojected with the previous frame's viewport, the HZB is built and read
  over the covered corner only. Jitter is Halton (2, 3) with `8 / ratio²` phases (8 at 100 %, 32 at 50 %). Per output
  pixel TSR gathers the 3×3 internal samples around it with Gaussian weights of their jittered distance *in output
  pixels* and accumulates them by weight (history alpha counts the samples it holds), so the result converges to the
  scene filtered at output resolution; the history is read with Catmull–Rom, clipped to the neighbourhood's YCoCg
  variance box (wide when still, tight under motion), rejected where the surface's motion differs from the motion
  the history pixel was written with (disocclusion) or where the *depth history* says the history pixel showed
  another surface (round 8: each history pixel keeps the view depth of the closest sample it was written with; that
  point, rebuilt with the previous camera, must lie within 2 % of the depth of the current surface's plane, its normal
  from the depth neighbours — a plane distance, so grazing surfaces don't fail it; the test acts in proportion to a
  motion difference of 0.05–0.5 px, where the motion test, which needs 1–3 px, is blind, and not at all where surface
  and history moved alike, since a sub-pixel twig the jitter missed looks exactly like a disocclusion;
  `RenderSettings::tsr_depth_history`, `--no-tsr-depth-history`, no measurable cost), blended with exposure-aware luminance weights, and pixels with
  less than one frame's worth of samples fill in a wider anti-aliased filter (the responsive path). Below 100 %,
  bright sub-pixel features (candle flames) that only some jitter phases hit are *locked* for one jitter cycle — not
  clipped or luminance-weighted (a lock is carried over only while the pixel is still) — or the clip would erase them. At 100 % it is the former TAA
  (same blend rates and clip box) with a proper reconstruction filter. Texture sampling gets the mip bias
  log2(internal / output) (G-buffer derivatives are scaled, so streamed texture tiles and the RVT follow); Nanite's
  error threshold is in output pixels (TSR resolves sub-internal-pixel geometry, and the cut and streaming don't change
  with the dynamic percentage) while the software-raster heuristic and HZB use internal pixels (what is rasterised);
  shadow pages — the sun's clipmap and the local lights' mips — are selected for the internal resolution's upper
  bound (a fixed percentage scales shadow cost with it, dynamic changes don't invalidate cached pages); pages are
  marked from internal pixels and the local-light shadow mask is computed at internal resolution. Stars keep their
  output-pixel size. **Dynamic resolution** (`DynamicResolution { gpu_budget_ms,
  min_percentage, max_percentage }`, off by default natively, on in the browser) averages the GPU timestamp totals of
  six frames, holds inside 80–100 % of the budget and otherwise steps towards 90 % assuming time ∝ area (which
  under-corrects, so it approaches from one side), in whole percents, ignoring the frames still in flight after a
  change. Lumen, clouds and fog run at the internal resolution (fog froxels per 16 px of the upper bound). Pixels the
  particle / volume composite marks *reactive* keep at most 1.5 frames of history (see *Order-independent
  transparency*).
* **Camera** (`passes/post_fx.rs`). A thin-lens camera (focal length from the field of view and sensor size,
  f-number, autofocus on the screen centre) drives scatter-as-gather depth of field (spread over the circle of
  confusion's radius; until round 8 its diameter, twice the blur, which photo mode's lens samples exposed); motion blur reconstructs along
  per-tile dominant velocities with depth-aware weights (McGuire et al. 2012) and a 180° shutter by default. Exposure
  comes from a 64-bin luminance histogram that ignores the darkest and brightest 10 %, so a sun disk or a candle
  flame cannot pump it.
* **Photo mode** (`photo.rs`, `photo.wgsl`; UE's Movie Render Graph accumulation and its Accumulation DOF). With the
  camera still, `Renderer::start_photo(n)` (Enter, `--photo N` with `--screenshot`, `heliosPhoto(n)` /
  `heliosPhotoProgress()` in the browser) sums `n` frames into one still: every frame gets a new sub-pixel jitter
  (Halton 2, 3, no cycle) and, with depth of field on, a lens sample — the camera moves to a point of the aperture
  disc (Halton 5, 7 mapped concentrically; diameter focal length / f-number, the real-time depth of field's lens) and
  the projection is sheared so the focus plane (the setting's distance, or the autofocus distance read back from the
  GPU when the photo starts) stays put. TSR's photo permutation reconstructs each frame on its own (its Gaussian over
  this frame's samples, with their weight), `photo.wgsl` sums colour × weight and weight in two `rgba32float`
  targets, and the mean goes through exposure, bloom and tone mapping. The post-process depth of field and motion blur
  are off, the scene's clock stands still (wind, props, sun, exposure), Lumen, MegaLights and the fog keep converging
  and their noise averages too. Bright out-of-focus highlights keep their energy as real bokeh discs, which the
  gather-based depth of field dims. The still stays on screen until the camera moves or the photo is stopped.
  0.05 ms a frame for the sum; 128 frames of the cathedral take 2.2 s natively (58 MiB of targets at 1600×900, made
  on the first photo).
* **Local exposure** (`passes/local_exposure.rs`, as UE5's Local Exposure). One global exposure cannot hold Snell's
  window over the dim lake, sunlit windows over dark aisles or a dawn sky over a shaded valley. A bilateral grid of the
  output image's log luminance (cells of 1/45 of the screen height × 32 bins spanning ±10 stops around the metered
  average; built per cell with linear splatting into workgroup fixed-point atomics — WebGPU has no float atomics —
  then Gaussian-blurred in space and luminance) gives each pixel an edge-aware *base*: the average of its neighbours
  of similar brightness, which does not bleed across a window frame or the rim of Snell's window. Blended with a
  wide plain blur of the log luminance (40 %, against the flat "HDR look" of a purely bilateral base), the base's
  distance from the metered average is scaled by `highlight_contrast` (0.6) above it and `shadow_contrast` (0.85)
  below it, while the detail on top of the base keeps its full contrast. The grid is anchored on the adapted
  exposure, not the frame's content, so it does not pump. It runs after the exposure metering (which still sees
  the scene) and before bloom (a highlight brought down glows less) and tone mapping; `RenderSettings::local_exposure`
  (`None` = off).

* **Profiling** (`profiler.rs`). Every pass is timed with timestamp queries (when the adapter supports them) and shown
  per pass in the HUD — in the browser too.
