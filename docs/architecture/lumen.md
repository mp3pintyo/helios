# Lumen GI and Lumen Lite

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Lumen-style GI** (`lumen.rs`). A 4-level global distance field (0.25–2 m voxels) is composited on the GPU from
  per-mesh SDFs; a voxel **surface cache** stores the radiance leaving nearby surfaces (sun with SDF soft shadows,
  sky with SDF occlusion, local lights, emission, one bounce from the **radiance cache** of world probes), so bounces
  accumulate. The final gather places a **screen probe** per 16×16 tile, traces 64 rays each (screen → distance
  field → surface cache → sky), filters them across neighbours and projects to SH for per-pixel integration.
  **Adaptive probes** (UE5's adaptive placement): per 8×8 cell, the pixels the integration shades are tested against
  the four uniform probes around them, and where none lies on a pixel's surface (window mullions, gaps between pews,
  chandelier rings, distant detail) the cell requests up to two extra probes on the missed surfaces. Requests are
  granted up to half the uniform probe count — evenly thinned at random when there are more, so no region is starved
  every frame — appended behind the uniform probes in the probe buffer and atlas with a per-cell indirection, traced
  and filtered like the others (indirect dispatch) and interpolated on their finer grid. Without them such pixels
  flipped between the radiance-cache fallback and whichever uniform probe the per-frame jitter put on their surface;
  **reflections** trace GGX rays the same way. Stained glass feeds coloured light into the cathedral.
  **Near field** (`near_field.rs`, `lumen_mesh_sdf.wgsl`): as in UE5, the first 2 m of every gather, reflection
  and local-light shadow ray sphere-trace the per-mesh SDFs (packed into one 3D atlas, heightfields for terrain)
  of the instances listed in a camera-centred 2 m culling grid; the global field takes over beyond. Small
  objects (candles, pew legs) occlude and cast contact shadows, and a light's own fixture no longer blurs into
  a voxel blob around it.
  **Mesh cards** (`cards.rs`, `lumen_card_light.wgsl`, `lumen_cards.wgsl`; UE5's card-based surface cache): every mesh
  with a volume distance field is captured on the CPU by up to 6 axis directions × 3 depth layers of orthographic cards
  (depth + normal per ~10 cm texel, rasterised from a LOD cut; depth peeling captures e.g. a courtyard's inner walls
  behind its outer ones). Each instance's cards are lit at a mip chosen from its distance (texels ≈ 1 % of it) in 8×8
  pages of a 2048² atlas behind per-card page tables, a global LOD bias keeping the atlas within budget; 1024 pages a
  frame are lit — new allocations first, then a round-robin refresh — with the sun through the virtual shadow map where
  the texel's page is resident (distance-field shadows otherwise), local lights, sky occlusion, one bounce from the
  radiance cache and emission. Gather, reflection and radiance-cache probe hits project into the hit instance's cards and keep the texel
  whose captured depth and normal match the hit (the near field reports the instance it hit; the global field stores
  the closest card instance per voxel); the voxel lighting remains where no card matches. Lighting stays on the
  surface it belongs to: no sunlight through thin walls, no terrain colour bleeding up through a floor that covers it,
  sun patches through windows bounce. Debug view 14 shows the card lighting each pixel would read (purple: voxels).
  GPU memory: 32 MiB lighting atlas, 16 MiB instance ids (16 bits per global-SDF voxel), a capture atlas within 16 MiB
  (cathedral 7.7, Sponza 9.4 MiB).
  **Material capture** (`lumen_card_capture.wgsl`; round 6, `docs/round6/card-albedo.md`; UE5's card capture into
  the surface cache's albedo and emissive atlases): cards carry the material per texel, not its average. The CPU
  capture records which triangle each texel sees, from which the texel's texture coordinates and — for displaced
  variants — the base point it was displaced from are interpolated; the GPU then evaluates the material at every
  texel as the G-buffer resolve does (image and procedural textures through the 512² arrays, the shading models'
  procedural detail, material graphs' generated surface code with their parameters), filtered for the texel's
  footprint, into an rgba8 albedo atlas (albedo and emission), once per (surface, material) pair — per instance
  where the material depends on the placement (world-space textures and graphs, a rock's moss), within 1 M texels.
  The card lighting reflects each texel's light with that albedo and adds its emission, so bounce light takes the
  colours of the surfaces it leaves: the gallery's mushroom caps bounce red, not their bed's green; the cathedral's
  stained glass glows in its panes' colours, its walls and pews in their textures'. Captures run within 1 M texels a
  frame (at most ~0.3 ms) after a scene loads (the cathedral's 2.1 M in 3 frames) and again for the pairs whose material,
  graph or parameters change; until then a pair uses the average; pairs that did not change keep their albedo
  through scene re-uploads. Memory: the albedo atlas within 16 MiB (gallery 2.6, cathedral 10.7 MiB), the texels'
  material inputs kept on the CPU for recaptures (16 bytes per captured texel). The capture also sums each pair's
  texels: the global distance field stores, per voxel, the mean captured albedo of the closest instance's pair
  instead of the material's average, so the voxel surface cache (and the near field's estimate from it) bounces a
  graph's colours too — the mushroom bed's voxels went from moss green to the caps', stems' and moss's mean; the
  field recomposes once captures finish. Card lighting is filtered **depth-aware**: every lit texel stores the depth
  of the surface it was lit at, and a hit's bilinear taps keep only texels on its tangent plane, so a mushroom cap's
  light no longer tints the ground beside it (nor a wall's the floor at its foot).
  **Radiance cache refresh** (`update_probes` in `lumen_light.wgsl`): it ran one thread per probe tracing its 32 rays
  in turn — ~3500 long serial threads, most of the GPU idle; it now runs a workgroup per probe, a thread per ray,
  with a shared-memory reduction of the SH: the "lumen surface + radiance cache" pass went from 1.42–1.49 to
  0.22–0.35 ms (cathedral 1.48 → 0.35, forest 1.42 → 0.24, Sponza 1.49 → 0.22, valley 1.43 → 0.22; shared GPU,
  minimums of interleaved runs), images unchanged within noise. Both final gathers profit. Moved instances update
  the global field, cards and near field in place (see [*Dynamic world*](motion.md)).
  **Displaced surfaces** (`lumen_displaced.rs`, `lumen_displace.wgsl`; round 6, `docs/round6/lumen-displacement.md`):
  Lumen follows Nanite tessellation. Every instance whose material displaces its mesh is represented by a
  *displaced variant* of the mesh — a distance field (the mesh's own kind and voxel size) and mesh cards baked from
  the displaced surface — so the global field, the near field, both surface caches, indirect light, sky occlusion
  and reflections see the relief: texture heights, graph heights and vectors (overhangs) alike. Variants are per
  (mesh, material), and per instance where the displacement depends on its placement (world-space graphs and
  vectors, world triplanar heights). The mesh is refined into points at most ¾ of a voxel apart (welded, on a
  worker thread), the GPU displaces them with the tessellator's own `displacement_position` — for graphs through
  the dicing passes' generated module — filtered for the point spacing, and the field and cards are baked from the
  read-back positions on worker threads with the same code as every mesh's. In the browser, which has no threads
  without cross-origin isolation, the refinement and the bake run on **Web Workers** that each run a second instance
  of the wasm module (`lumen_offload.rs`, `web_bake.rs`, `bake_worker.js`; `docs/round6/web-bake-worker.md`): the
  render thread only hands over the inputs and takes in the results, one per frame, so a scene's variants no longer
  stall frames for 30–920 ms each (worst frame while the gallery's four variants bake 718–920 → 6–7 ms, against
  2–4 ms without baking, and all of them in after 1.5–1.6 instead of 2.3–3.3 s; `?bake-workers=0` runs them on the
  render thread as before, as does a browser where a worker cannot start). A watchdog keeps a worker from ever
  leaving a variant missing: a worker that does not report ready within 8 s, does not answer a task within 10 s or
  raises an error is terminated and its task runs on the render thread. An
  animated displacement is represented by its mean surface (32 times in 8 s): the gallery's cloth moves at most
  3.6 cm RMS from it at 3.3 cm voxels, where a snapshot would be √2 times as far off and a re-bake could never keep
  up with the waves. Until a variant is ready (0.5–1.3 s of CPU per variant) instances keep their previous one;
  the variants of a scene arrive in one Lumen re-upload. Debug view 14 shows the gallery's mushroom bed and cloth
  in card lighting where they were voxel fallback (purple) before, and fewer unmatched card texels on the
  valley's displaced boulders (19 → 14 % of their pixels). Scenes without displacement upload and render as before.
* **Lumen Lite** (`lumen_lite.rs`, `lumen_lite.wgsl`, `lumen_trace.wgsl`; UE 5.8's `r.Lumen.FinalGatherMethod 0`).
  An alternative final gather for weak GPUs (`RenderSettings::gi_method = GiMethod::LumenLite`): instead of tracing
  64 rays from every 16×16 screen tile (plus adaptive probes) every frame, diffuse GI is gathered from a world-space
  **irradiance field** (DDGI-like) that changes slowly and is shared by every pixel near it. 4 camera-centred clipmap
  levels of 16³ probes (1, 2, 4, 8 m spacing: ±8 … ±64 m) scroll by whole spacings with a little hysteresis;
  probe slots are addressed toroidally (world coordinate mod 16), so a move keeps every probe still inside a window
  and only the slab that entered is new. Each probe holds an octahedral irradiance map (6×6 texels: cosine-weighted
  mean radiance + sky visibility) and an octahedral map of ray-distance moments (14×14: mean and mean square in
  probe spacings) for Chebyshev visibility, each with a one-texel octahedral border in its atlas so shading samples
  them bilinearly (rgba16f atlases of 40 MiB, plus 3 MiB of buffers, allocated on first use). **Update budget**: 512
  probes a frame are picked on the GPU without readback — a pass scores every probe (new probes first, finest levels
  first; then age × 4 on screen × ⅛ for *sleeping* probes × 1/64 for *buried* ones) into a 64-bucket histogram, one
  workgroup finds the threshold that fills the budget and a third pass appends the winners (indirect dispatches from
  there). **Classification**: before tracing, the global distance field moves a probe that sits within 0.1 spacing
  of a surface along its gradient (≤ 0.45 spacing), disables (*buries*) it if it cannot get out — inside walls, under
  terrain — and puts it to *sleep* where no surface lies within 1.8 spacings (no pixel interpolates it; it is still
  refreshed now and then, as foliage is not in the global field). **Tracing**: 128 rays per probe (spherical
  Fibonacci, randomly rotated per refresh) through Lumen's own tracing — near-field mesh SDFs, global distance field,
  mesh-card and voxel surface caches, radiance-cache far field, sky — one shared WGSL function
  (`trace_world_radiance`, which the screen probes call after their screen trace), so the irradiance field sees the
  same surface cache and the same multi-bounce light. The rays are projected into the probe's texels with hysteresis
  0.85 (halved where the irradiance changed by more than half: switched lights, a moving sun; new probes take their
  first rays as they are), then copied into the atlases with their borders. **Shading**: per half-resolution pixel,
  the 8 probes around a shading point biased 0.15 spacing along the normal and 0.3 towards the eye, weighted by
  trilinear × smooth backface × Chebyshev (cubed, crushed below 0.2, as DDGI) terms, from the finest level that holds
  the pixel, blended into the next coarser level over its window's last 2 cells and into Lumen's radiance cache beyond
  the last level. The result replaces the screen-probe gather in `gi_raw` (same format), so the GI denoiser, sky
  visibility (specular occlusion), material AO and the lighting pass are unchanged. **Reflections**: rough ones come
  from the irradiance field — the lighting pass already blends specular towards the gathered irradiance between
  roughness 0.2 and 0.5 and fades the traced reflection out between 0.35 and 0.55; glossy ones keep Lumen's traced
  reflections (one GGX ray per half-resolution pixel), since a probe's 6×6 irradiance cannot show a reflected image.
  **Selection**: `--gi lumen|lite|auto` natively, `?gi=lumen|lite` in the browser (page config `gi`), `;` toggles. By
  default (`auto`) integrated and software adapters start in Lumen Lite, and `RenderSettings::gi_lite_fallback`
  switches to it when dynamic resolution has sat at its minimum percentage for 180 frames (the screen-probe gather is
  the largest GI cost that does not shrink with the resolution) — browsers do not report the adapter type, so there
  the fallback decides. Debug view 20 shows the probes as spheres shaded with their irradiance (sleeping: blue tint,
  buried: red, not yet traced: magenta) over the irradiance each pixel gets.
  **Cost** (shared GPU, interleaved runs, sum of per-pass minimums of 3–6 runs; "GI" = mesh cards + surface /
  radiance cache + final gather + reflections + GI temporal): the final gather drops from 0.77–1.07 ms to
  0.36–0.40 ms (probe update 0.28–0.35 ms, independent of the resolution, + per-pixel gather 0.05–0.08 ms), all of GI
  from 1.20–1.71 to 0.74–1.04 ms (1.5–1.8× cheaper; 2.5–3.4× cheaper than round 3's 2.48–2.91 ms):

  | Scene | final gather: screen probes → Lumen Lite | GI round 3 | GI Lumen | GI Lumen Lite |
  |---|---|---|---|---|
  | valley | 0.77 → 0.40 (probes 0.35 + gather 0.05) | 2.61 | 1.20 | 0.81 |
  | cathedral | 0.91 → 0.36 (0.28 + 0.08) | 2.72 | 1.62 | 1.04 |
  | forest (medium) | 0.92 → 0.37 (0.30 + 0.07) | 2.48 | 1.29 | 0.74 |
  | Sponza (atrium) | 1.07 → 0.39 (0.32 + 0.07) | 2.91 | 1.71 | 0.95 |

  Images: the cathedral's lit frame differs from Lumen by 36.5 dB PSNR (valley 33.8, forest 43.4); its indirect
  light is 7 % darker on average, with more contrast where the probes see occlusion (pew undersides: 5th percentile
  0.20 vs 0.30) and less where the screen probes resolved it (pillar bases, vault ribs); Sponza's atrium is 9 % darker
  in indirect light (the far end of the gallery most; 23.6 dB); no light leaks through the 1.2 m aisle walls next to
  the sunlit exterior. Temporal stability (static camera, 24 frames, % of pixels changing by > 24/255): cathedral
  0.002 % for both, valley 0.042 % (Lumen 0.053 %), forest 0.056 % (Lumen 0.060 %); the indirect light alone changes
  by more than 4/255 in 1.05 % of the pixels (Lumen 0.30 %) — a fine uniform grain from the probe refreshes rather
  than the screen probes' blotches. Moving camera: within 2 % of Lumen's frame-to-frame change, no pops as slabs
  scroll in.
* **Short-range AO** (round 8, `short_range_ao.wgsl`, [`foliage-grounding.md`](../round8/foliage-grounding.md); UE's
  Lumen short range AO). The occlusion within 0.4 m of a pixel is what both GI methods miss: screen probes sit one
  per 16² pixels at half resolution and their filter smears it, the distance fields hold no thin foliage, and a grass
  tuft or a fern stood on fully lit ground. The lighting pass marches, per pixel, 4 screen directions (rotated per
  pixel and frame by interleaved gradient noise; TSR averages them) in 4 steps within 0.4 m (at most 64 px at 900
  rows) through the depth buffer, keeps each direction's horizon — the highest sample's sine of elevation above the
  tangent plane, `cos(v, n)`, biased by 0.15 and faded towards the radius — and takes their mean as the occluded part
  of the cosine-weighted hemisphere (horizon-based AO, as HBAO / GTAO without their slice integral). It multiplies the
  material AO, so it darkens Lumen's diffuse light (or the sky's) and the specular occlusion, never direct light.
  Samples beyond the radius are skipped: a blade in front of distant ground casts no halo. `RenderSettings::short_range_ao`
  (on), `--no-short-range-ao`, `9`, editor *Short-range AO*; frame flag 4096. Cost (forest, 1600×900, shared GPU, minima of interleaved runs): lighting +0.15 ms in the default view,
  +0.19 ms with grass filling the screen.
