# Status vs. Unreal Engine 5 — what is still simpler

Back to the [README](../README.md).


Honest differences that remain:

1. **Streaming**: page payloads use a quantised, bit-packed GPU format like Nanite's, but are not entropy-coded on
   disk on top of it (Nanite adds Oodle): measured per page, LZ4 would save only 0.4–5 % and DEFLATE 3–16 %, so `.hvm`
   files store the GPU format as is. Triangle indices are plain fixed-width (no strip / delta coding). Every mesh's
   header — including all cluster records — stays resident. Cached shadow pages are invalidated per instance for
   moves, streaming, wind and skeletal animation; structural scene changes (instances added or hidden, materials
   reassigned, shadow casting toggled, new geometry), moves that grow the scene bounds and toggling voxel LODs or tessellation
   invalidate them globally.
2. **Lumen**'s mesh cards are axis-aligned (6 directions × 3 depth layers per mesh, no per-mesh card fitting as in UE5),
   carry their materials per ~10 cm texel (the bounce of detail finer than that is averaged by the texel's
   footprint; the voxel surface cache sees one mean per instance and material), are lit with one bounce from the radiance cache
   rather than card-to-card radiosity, and local lights shadow them through the global field only; surfaces no card
   sees (deep concavities, sub-texel slivers) and instances beyond 100 m use the voxel surface cache, which still leaks
   a little light through thin walls; translucent surfaces get no Lumen
   front-layer reflections (screen traces, the global field and the radiance cache instead). Mesh SDFs have at most
   48 voxels along their longest axis, so very large meshes are no sharper in the near field than in the global one.
   Adaptive screen probes are capped at half the uniform count, and detailed views exceed that (at 1600×900 the
   cathedral, the valley and the forest all request more), so they are thinned at random each frame; they cost
   0.3–0.4 ms at 1600×900 (the final gather goes from 0.5–0.55 to 0.8–0.95 ms). Remaining slow flicker on thin geometry
   comes from the surface caches' incremental updates (cards, voxels, radiance cache), not from probe placement.
   The **world cache** (round 8, default; UE 5.9's Lumen World Cache) replaces the cards and the voxel lighting at ray
   hits with sparse world-space cells filled by paths from the screen — indoors within 12–15 % of a path-traced
   reference where the cards and voxels were 3–6 times too bright (their 15 % open-sky term and the voxel leak through
   thin walls) — but it is traced against the distance fields, not hardware ray tracing: rays still miss skinned
   meshes, particles and volumes (their cells exist on screen only), paths shade two vertices with one sampled local
   light each and global-field shadows, cells are 2.5–5 % of their distance across (finer bounce detail is averaged),
   a surface no path reached in 64 frames falls back to the voxel lighting (and its sky term) — briefly after a cut —,
   and there is no far field beyond the clipmap. With the world cache off or behind the cards (A/B), what this item
   says about the cards and the voxel lighting applies as before.
   The surface caches see neither skinned meshes, volumes nor particles (only a fire's driven light), and follow
   moved instances over a few frames (item 9); they see displaced surfaces to the resolution of their baked
   variants, animated ones as their mean surface (item 14). **Lumen Lite**'s irradiance field covers ±64 m around
   the camera (4 levels of 16³ probes; beyond, Lumen's radiance cache with an estimated sky visibility); probes are
   refreshed within a fixed budget of 512 a frame (128 rays each), so lighting changes reach distant and off-screen probes over a
   few hundred milliseconds and a refreshed probe changes in small steps (fine grain in the indirect light: 1.05 % of
   pixels change by > 4/255 per frame on a static camera vs 0.30 % with screen probes); classification uses the
   global distance field only (alpha-masked foliage is not in it, so probes in tree crowns can sleep); there is no
   probe-to-probe irradiance feedback into the surface cache (the bounce still comes from the radiance cache); the
   refresh rate follows each probe's variability (round 8) but only through the schedule's score (UE also
   classifies probes by it); Lite's glossy reflections are screen traces (round 8): off-screen and back-facing
   objects come from the irradiance field (a cosine-wide lobe, no reflected image), the closest-depth pyramid is
   built separately from MegaLights' one, and where few pixels are glossy it costs more than the traced path did
   (forest +0.03 ms); height fog on reflections leaves out the fog's noise and local lights; the adapter-based default can only
   see integrated GPUs natively (browsers hide the adapter type, so there only the dynamic-resolution fallback
   switches).
3. **Materials**: streamed image tiles are block-compressed only where BC is available (no ETC2 / ASTC encoders: mobile
   GPUs get uncompressed tiles), are encoded at load (natively) or on demand (web) rather than cooked offline, and are
   cut from decoded images kept in CPU memory (no tiled on-disk cache); the 512² texture arrays are uncompressed;
   triplanar mappings and passes other than the G-buffer resolve (Lumen, landscape RVT) see the 512² layers;
   procedural textures are not virtualised. glTF: the images of one layer (base colour, normal, metallic-roughness)
   must share their coordinates — a normal or metallic-roughness map with another UV set or transform than the base
   colour is sampled with the base colour's (with a warning); UV sets beyond `TEXCOORD_1`, `KHR_materials_iridescence`,
   `_dispersion` and `anisotropyTexture` (the anisotropy factors are) are not imported; the clear-coat, sheen and
   specular textures are sampled with the main texture's coordinates from the 512² arrays (not streamed);
   `KHR_materials_transmission` becomes refracting glass (refraction of the screen, not a thin-walled BTDF;
   `transmissionTexture` and `thicknessTexture` are ignored). Alpha tests ignore wrap modes and vertex alpha. Lumen's
   cards capture materials per texel (graphs included) but without vertex colours or the second UV set, and graphs
   that read the camera or time are captured as they were at capture; its voxels and the global distance field see
   one mean captured albedo per instance and material (no emission). Simplification keeps second-UV and colour seams but
   does not weigh their deviation. Alpha tests read image textures with mesh UVs only (not procedural textures or
   triplanar mapping), with 8-bit cutoffs. In Lumen an alpha-masked instance is partially transparent by
   one average coverage (its partial shadows and occlusion are uniform, not shaped like its holes), only in the near
   field: rays and local-light shadows see it within 2 m of their ends, beyond which the global field lets light
   through; its cards are captured from the geometry, holes and all, with the material's average albedo.
   **Substrate-lite** has one slab per material: no vertical layering of arbitrary slabs, no horizontal mixing (UE's
   Substrate mixes / layers / blends any number, with per-pixel slab counts and a closure-count–adaptive G-buffer); no
   hazy gloss, iridescence, dispersion, glint or fuzz beyond sheen, no specular profiles, no volumetric media. A slab
   with a clear coat ignores anisotropy (they share the texel's direction), and subsurface + transmission share one
   per-pixel amount (transmission then takes the material constant). Subsurface diffusion is screen-space (no light
   from behind thick objects except through thin transmission, no subsurface for translucent materials), uses the
   material's colour and radius rather than a measured profile, and under MegaLights the local lights' diffuse light
   of subsurface pixels is not diffused (MegaLights keeps diffuse light whole). Translucent materials (glass, water,
   `BLEND`) ignore slabs. Unlit colours are scaled to a white surface in full sun. Devices limited to WebGPU's default
   16 sampled textures per stage render slab materials as their plain base. **Material graphs** run in the G-buffer
   resolve and, for their displacement (a height or a vector), in the tessellator: no graph-driven opacity mask, world position
   offset, translucent or decal materials (raster and translucency shaders are fixed); at most 16 distinct (shading model, graph) pipelines per frame (further
   materials render without their graph, with a warning) and 1024 parameter slots in all; no custom HLSL/WGSL node,
   material functions, static switches, texture objects or per-node previews; noise and texture outputs carry no
   derivatives (a texture read at coordinates computed from another texture or from noise filters as if they were
   constant, i.e. at its finest level there); texture parameters are not editable in the UI (they default to the
   material's texture) or stored in scene files; time is the renderer's clock.
4. **Local lights**: at most 32 lights per frame get virtual shadow maps, sharing the sun's 1024-page physical pool
   (UE's default pool is four times larger), so local lights request pages half a mip coarser than the screen
   footprint; a moving light's shadow map is not re-rendered every frame as UE's is but each cube face is held at an
   origin within 0.1 source radii and parallax-corrected in the soft-shadow filter (lights faster than that are
   re-rendered every frame, as in UE, while a budget of pages per frame pays for it): exact at contact and for
   occluders the blocker search finds, while shadows of occluders right next to a large moving light stay where the
   origin casts them, single-tap lookups (fog, lights beyond a pixel's four filtered ones) are uncorrected, the map is
   a mip coarser, lights the budget cannot follow are held at up to four times their tolerance (beyond that,
   distance fields until they rest), and the budget counts pages, not their cost in draws. Casters moving with a
   light (its own fixture) refresh its pages every frame outside that budget; such a light's claim on a slot falls
   with the pages its map costs per frame, so candles inside swinging fixtures keep distance-field shadows where
   static lights compete for the slots (a map without its fixture, traced in the distance fields instead, would
   be cheap but no better than distance fields: the fixture's shadow is what such a candle casts). The pool is
   too small for the candle-lit cathedral's 32 maps (~1860 pages requested a frame): it serves them by priority
   (every map's coarsest pages, the sun, then lights by rank) and coarsens the fainter lights' maps until the
   requests fit (up to 2.5 mips on average), where UE's larger pool would hold them. With the classic
   clustered loop, the rest get distance-field
   shadows for the strongest few per pixel and no shadows beyond those (light leaks through walls), and shadow-map
   soft shadows are PCSS-filtered, not ray-marched. **MegaLights** shadows every light but traces no hardware rays
   (WebGPU has none): beyond the 1 m
   screen trace, rays use the light's virtual shadow map (only the ≤ 32 static lights with one; SMRT-style, 8 steps) or
   Lumen's distance fields, which are coarser than triangles (mesh SDFs of at most 48 voxels per axis, 0.25 m global
   voxels beyond 2 m from both ends). The screen trace walks a closest-depth HZB but only over 1 m (32 iterations at most;
   Lumen's screen traces still march linearly). Samples come from one representative pixel per 2×2 block (all four
   pixels shade with its pdf; edges between surfaces converge over the rotating representative), always four per
   block: rays to a light picked twice are shared outside penumbrae (UE 5.8), but there is no adaptive *sample* count
   and no per-light-type budget, and the compacted ray list costs ~0.05 ms, so the saving is small with many lights
   (6–7 % fewer rays at 300–1000); guiding uses a 256-bit hash per tile (lights 256 indices apart share a bit). The
   importance cull is a per-light radius at last frame's exposure (it cuts nothing where ranges are already tight).
   Volumetric fog and translucent surfaces get MegaLights' lighting from a froxel volume (16-pixel columns, one
   sample per froxel and frame, ~10 frames of history without clipping, so it lags moving lights; one dominant
   direction per froxel for translucency, no per-pixel front-layer pass as UE has, the clustered loop beyond the
   volume's depth range); particles keep their own path. The sun stays on its classic path (virtual shadow map).
   Lumen's surface cache samples two local lights per texel / voxel and refresh from a world light grid (round 8,
   32 candidates per cell, resampled where more lights reach it) and accumulates over refreshes: its cost is flat
   from 300 lights on, but light changes reach card texels only over two of their page's refreshes, cells with
   hundreds of lights add selection noise, and shadows are still one global-field cone per sampled light (no mesh
   SDFs, no shadow maps); points beyond 128 m from the camera keep the loop over every light. Light culling is a two-level grid (coarse cells of 4 × 3 tiles × 4 slices,
   0.07 ms at 1000 lights), not UE's z-binning or a light BVH; still up to 255 lights per froxel, 1024 per scene. The denoiser lags behind moving lights by up to
   24 frames where the variance clip does not catch the change, and its spatial filter softens lighting detail on
   fresh (disoccluded) pixels. The shadow overview (debug view 26, round 8) shows the shared pool's pages per light and its pressure as
   colours and bars, without the numbers and per-light labels of UE's overview (the HUD and the headless report carry
   the counts), and its per-pixel panels cover the sun's clipmap only (local lights: debug view 15).
5. **Translucency** (round 8, `avboit`): non-refracting surfaces, particles and volumes are ordered by adaptive
   voxel-based OIT — a 16-cell transmittance grid per 4×4 (8×8 above 2048 px) pixel column, so the order is
   approximate within a cell (the medium counts as spread evenly in it; UE averages instead), the grid's low-resolution
   transmittance blends over a few pixels at pane edges, and sub-column sparks may miss the low-resolution draw. UE's
   AVBOIT also covers Nanite translucency and per-pixel lit volumetric fog; ours has no Nanite translucency at all
   (forward meshes). It needs 19 sampled textures per stage (else the round-7 path below). Refracting surfaces composite
   only their nearest layer (refracting glass behind it is seen unblurred through it, its own blur and refraction
   dropped); only where water overlaps another refracting surface are they peeled — then up to 4 layers deep (deeper
   layers fall back to per-instance order), the layer count from projected bounding boxes (overlapping boxes cost up
   to three extra depth-only draws, scene-colour copies and shading passes even where the surfaces don't), and
   anything order-independent between two refracting layers is composited behind both. Refraction reads
   the screen-space copy (what is off screen or hidden is not refracted; where a nearer layer covers the displaced
   lookup it falls back to the pixel behind); translucent surfaces don't request shadow pages (where the
   opaque scene didn't request theirs, they fall back to coarser resident levels) and are drawn without LOD. **Water**
   bodies (lakes) are normal-mapped planes (no displacement or tessellation, no FFT spectrum, no spline-shaped bodies:
   a body is its instance's bounding rectangle, at most 4 per scene, no oceans); rivers are simulated (item 18); the
   waves ignore `Scene::wind`. Caustics come from the analytic waves (a single preimage, so
   overlapping folds are not summed; no chromatic dispersion) and only light the bed seen through the water or from
   inside it, not the fog or other translucent surfaces. Underwater fog is lit by the light entering the surface above
   the middle of each view ray (no light shafts or volumetric shadowing along it), total internal reflection shows
   deep water rather than the bed, and a camera straddling the surface gets no split waterline view. The shore
   material is procedural (no texture layers, no wet-band wave run-up animation).
6. **Temporal super resolution** keeps one history (colour + sample weight, motion + lock) at output resolution, not
   UE's 200 % history option or its separate anti-flicker luma history; the depth history (round 8) keeps one closest
   depth per output pixel (no dilated depth scattered forward as UE's) and stays out where surface and history moved
   alike, so thin features missed by the jitter are not mistaken for disocclusions — UE has a dedicated thin-geometry
   detection for that; translucency, Lumen and fog have no separate upscaling (they render at the internal resolution
   and TSR resolves them with everything else); alpha tests in the rasterisers don't get the mip bias (sharper cut-outs
   would mostly add sub-pixel shimmer to foliage). Dynamic resolution only reacts to GPU time (no CPU-bound
   detection) and needs timestamp queries.
   **Photo mode** accumulates jitter and lens samples of one instant (no temporal samples — motion blur from
   sub-frames — no warm-up of history-dependent effects per sample, no tiling or high-resolution output beyond the
   window, a round aperture without blades). **Fog screen space scattering** spreads the forward lobe at
   single-scattering angles (many orders widen it in reality), weights the kernel per pixel only through τ (not per
   source depth), and ignores aerial perspective and underwater fog.
7. **Wind** is procedural (no pivot-painter data per branch or leaf, no simulation) and does not rotate normals;
   leaves flutter around their position rather than their stem. A per-vertex wind weight (round 8) holds plants at
   their roots, but only the procedural vegetation bakes it (imported foliage moves with weight 1, as do voxel LODs and
   tessellated surfaces), and it is one scalar per vertex, not a bone hierarchy (UE 5.9's skeletal *dynamic wind*). Shadow pages cache static casters separately (as
   UE's static cache); the wind, `Mobility::Movable` and animated skins make a caster dynamic. Refreshes are decided
   per instance and redraw every dynamic caster in the
   page, so in the forest — where trees, bushes, ferns and grass all sway and only terrain, rocks and logs are
   static — the wind still costs most of what it did (see [*Wind*](architecture/motion.md)); a page gets its static layer by being rendered in
   full once more at its first dynamic refresh, and only 256 pages a frame have scratch space for software-rasterised
   dynamic casters (beyond that they take the hardware path). The wind doesn't move distance fields, the surface cache
   or translucent materials (rigid moves do). Round 8 (`vsm-deferred-invalidation`) caps the refreshes at a budget per
   frame (96 pages), by how many texels a page's shadow may be off and how long it has waited: the forest's wind costs
   about a third less, but where the budget binds a swaying shadow lags its caster by up to 4 frames (UE throttles only
   Nanite LOD changes this way and keeps WPO-animated pages current); the priority ignores how many pixels sample a
   page. Distant clipmap levels (`vsm-prefiltered-distant`, from level 7) follow a moving sun every 4 frames in turns,
   not every frame: their shadows lag the sun by up to 4 frames or 0.5° (a seam where they meet the level before them,
   below a texel at the demo's sun speeds). Since wave 2 (`vsm-distant-prefiltered-moments`) they render one level
   coarser and are filtered as moment shadow maps (Hamburger 4MSM over a per-page normalised rgba16f moment tile) in
   the deferred lighting only — fog, particles, translucency, Lumen's cards and MegaLights still compare depths of the
   coarser pages; there is a resolution step where the last full-resolution level meets the first distant one; moments
   let some light through overlapping layers of casters (a light bleeding reduction trades it for darker penumbrae, and
   distant canopies come out ~2 % brighter than with the comparison filter); no temporal reprojection or accumulation
   of the moments (each distant level stays exact for the light view it was rendered with); the moment atlas costs
   43 MiB whether the distant levels hold few pages or many.
8. **Skeletal meshes**: linear blend skinning with up to 4 influences per vertex (glTF's first joint set; UE allows
   8–12) and 8-bit weights; no dual-quaternion skinning, morph targets (imported files with them render their base
   shape, with a warning), cloth or physics. Culling bounds are per instance (posed joint spheres) and per cluster only
   as the rest sphere padded by the instance's largest displacement — UE bounds clusters by their own influencing
   bones, which is tighter for large motions (a waving arm makes every cluster's sphere grow by the arm's reach). The
   LOD error is that of the rest pose: the weight-deviation cost only orders the collapses (coarse levels keep weight
   seams longest; a test measures half the posed deviation), it does not enter the error. Skinned meshes get no
   distance field, mesh cards or voxel LODs (as in UE: Lumen sees them through screen traces only), and translucent
   (`BLEND`) skinned surfaces are drawn by the forward pass in their rest pose. Shadow pages under a moving skin
   re-render its whole caster set each refresh (as for the wind); the footprint grows with the largest reach seen, so
   a figure that once stretched far refreshes a larger area. Root motion is not extracted (animated root joints move
   the posed bounds, not the instance).
9. **Dynamic world**: moves are rigid transforms of whole instances (deformation is skinning, item 8). A moved
   caster's shadow footprint is its oriented box in light space, cut into at most 8 slices along its longest axis,
   so a thin caster with a wide part (a chandelier's chain above its ring) still refreshes some pages its geometry
   doesn't touch; UE's instance-culled page invalidation is tighter. A move's speed bound is its current displacement
   held with a decay, not a true bound: a caster that slows down sharply can leave a page a few texels stale until it
   moves on or stops. A fixture around its own moving light redraws that light's pages every frame (the light's
   slot claim pays for it, item 4). The
   global distance field lags by up to a quarter voxel per level (and further for a fast mover beyond the brick
   budget); an instance that stops within that distance of where a level last composed it
   stays there. The voxel surface cache and radiance cache follow moves over a few frames instead of being
   invalidated locally. Instance records are rewritten with one `write_buffer` per contiguous run, not a GPU scatter
   of dirty records (UE's GPU-scene update); adding, hiding or removing instances still rebuilds all instance data.
   The landscape (runtime virtual texture) and water bodies' lake beds don't follow a moved terrain instance.
10. **Nanite Foliage**: grounded foliage (round 8) relies on content: the procedural grass, ferns, bushes and trees
    bake their wind weights and ambient occlusion and the forest places them (and its PCG volumes) on its terrain
    mesh's height, computed from the scene's own height function; imported plants get no plant data, and there is no
    Landscape-style height query of the rendered, LOD'd terrain for arbitrary scenes, and
    the short-range AO is screen space (nothing off screen or behind nearer geometry, no thickness, 4 × 4 noisy taps
    averaged by TSR; UE's comes with the screen-probe gather and a bent normal). Voxels carry per-axis coverage, one
    normal and their brick's mean texture coordinate — no per-voxel colour, so textures varying within a brick, vertex colours and alpha-masked (card) foliage's mask
    coverage are averaged away or ignored; coverage along a diagonal is modelled from the three axes (exact for
    axis-aligned and isotropic leaves). Voxels are rasterised as stochastic cubes (no ray-marched depth buckets or
    front-to-back ordering), rely on TSR to average their dithering, move rigidly with the wind at their centre (no
    flutter, no bone-driven wind), and have no hardware path: in shadow pages refreshed over a static layer without a
    scratch tile, or split pages, they sit out that refresh. Voxels win only where pruning reports a larger error than
    the voxel budget allows (4 voxels per kept triangle: the forest's conifers and ferns from mid-distance, the
    broadleaf canopies near their roots). Assemblies are one level deep (no nested parts), their parts take the
    instance's material (wood and leaves are separate instances), and the base ↔ parts switch is per assembly
    instance, not per part (UE resolves part instances inside the cluster hierarchy): an expanded tree still writes
    every part as an instance record each frame (≈ 0.03 ms for the forest's ≈ 26 k conifer parts) and is culled
    through one record per tree whose lanes are its parts' pages (round 6: the parts' culling cost fell from ≈ 0.6 to
    ≈ 0.1 ms), and parts are only as coarse as their own small DAG allows, so the expanded range draws more
    triangles and fewer voxels than one merged mesh would. When more parts are wanted than the 32768 slots, the
    expansion threshold rises, so trees near the edge of the expanded range draw their base up to ~1.5 px coarse.
    Parts are rigid copies (a needle spray follows its branch span, not each branch's own curve); part instances'
    shadow pages are not invalidated by their pages streaming in (the base's are).
11. **Volumes and particles**: volumes are dense grids per emitter (no sparse / VDB volumes, no streaming of
    sequences, at most 8 emitters simulated and 4 marched per frame, each up to 192 steps per ray); the fluid solver
    is a single-grid red-black SOR (no multigrid), collocated (no MAC grid), with a solid floor only (no obstacles from
    the scene's geometry, no interaction with other emitters or particles) and a fixed time step clamped to 1/30 s.
    The lighting cache holds the sun's transmittance plus isotropic ambient and up to 8 local lights (self-shadowing
    only for the fire's own driven light; multiple scattering is one approximate octave of the sun, not simulated), the
    march samples it trilinearly at half the grid resolution; close up, a 4 cm campfire grid shows its resolution in
    the smoke's edges. Volume shadows (round 8, `volume-shadows-lumen`) are a per-emitter transmittance grid at the
    lighting cache's resolution (8 bits, towards the sun and towards the emitter's own driven light only), not UE's
    adaptive volumetric shadow map: other local lights (shadow-mapped or not) pass through smoke unattenuated, the
    shadow is a frame late in the opaque lighting and the fog, emitters beyond the 250 m simulation distance cast none,
    and translucent surfaces (AVBOIT / forward translucency: their layouts are at the sampled-texture limit), Lumen's
    surface cache and MegaLights ignore it. Volumes still don't write into the virtual shadow map and don't feed Lumen
    (only the fire's driven light does: its light is the emission's share of the scene) and are not seen by
    reflections or refraction outside the screen. Overlapping
    volume boxes are composited box by box in entry order, not interleaved. With AVBOIT (round 8) particles, volume
    steps and non-refracting surfaces are weighted by the transmittance in front of them (sparks inside smoke are
    dimmed by the smoke in front, smoke behind a panel is tinted by it), at the grid's resolution: the volume enters
    the grid with 4 samples per cell, the march picks one grid column per pixel at random (its temporal filter averages
    it). Only the nearest refracting surface splits the order (what lies between two refracting layers is composited
    behind both). The round-7 path (`--wboit`): weighted blended OIT approximates the order of layers at similar depths
    and saturates for very many opaque layers, particles go over the volume of their bucket; collision only against the visible depth buffer (no off-screen or occluded collisions, no distance-field
    collision), no ribbons, meshes, sub-UV flipbooks, events or GPU sorting; lighting is per particle (no per-pixel
    normal-mapped lighting). Memory: 72 bytes per internal pixel while volumes and particles are on screen (volume
    march + history front/back, OIT targets, nearest translucent depth: 104 MiB at 1600×900).
12. **Editor**: no actors or hierarchy — a tree's trunk and foliage are two instances, selected separately; deleted
    instances and lights stay as hidden / disabled slots (reused, never compacted); undo history is in memory (200
    steps); no landscape sculpting or painting, no multi-viewport, no editor-only billboards in the renderer (light and
    volume icons are a 2D overlay without depth test), the gizmo draws over everything; typed transform values apply to
    the primary selection only; the native window has the editor core without panels. Edited instances stay
    `Static`, so the shadow pages under a dragged object are rendered in full every preview frame, and hiding,
    deleting, duplicating, reassigning materials and PCG regeneration are structural changes (all instance data
    re-uploaded, Lumen's instance lists rebuilt). Scene files address base instances by build order (a changed scene
    builder is detected by mesh name and the edit skipped with a warning); dropped `.glb` files are referenced by file
    name and must be hosted next to the page to reload; glTF imports bake on the main thread in the browser.
13. **PCG**: a fixed node set without a graph canvas (nodes and their parameters are edited in the Details panel; the
    graph's structure comes from the presets / the file), no attribute system or custom nodes, no spline editing in
    the viewport (distance targets are volume-local data), spawners reference meshes and materials by name, generation
    runs on the main thread (no partitioned / hierarchical generation, no runtime generation around the camera), and
    spawned instances are not individually editable. Native and wasm builds may differ in the last bits of `sin` /
    `cos`, so a point exactly at a filter threshold could flip between them; saved output sidesteps it.
14. **Nanite Tessellation**: displacement is a height along the interpolated normal — a texture channel through
    the material's mapping or a material graph's Displacement output — or a graph's world-space vector, which a
    TransformVector node can build in object or tangent space (round 6). The displacement's tangent frame is not
    UE's: meshes store no tangents, and a per-triangle UV frame would crack, so it is the interpolated normal with
    the object X axis projected into its plane — the mesh's U direction on planes and terrain, elsewhere another
    orientation, and it swirls around a curved surface's points facing ±X. Tessellation factors come from the
    undisplaced edges, so where a vector stretches the surface micro-triangles grow with it (silhouettes coarsen;
    shading stays smooth). Textures are read from the 512² texture array layer (streamed image tiles are not
    consulted) at a mip chosen from the dice segment size, so heights finer than the layer's texels are lost. A
    graph's displacement may read the time but not the camera, vertex colours, UV set 1 or the shading tangent
    frame (Triplanar's TangentNormal; compile errors). An animated displacement's speed bound (round 6) is the
    material compiler's conservative analysis of the graph: graphs it cannot bound (the time through `Frac` /
    `Floor` / `Step` / `If`, scrolled textures, `Time` to a power) refresh every cached shadow page whose texels are
    finer than their reach every frame, the bound is per instance rather than per page, and close up a fast surface
    still refreshes its finest levels every frame (UE re-renders WPO-animated casters every frame). Lumen follows
    displacement through baked *variants* (distance field and cards per mesh and displacement): accurate to the
    mesh's voxels and the cards' ~10 cm texels, so relief finer than those (mortar joints, cobbles) only reaches it
    through screen traces; an animated displacement is represented by its mean surface (8 s average), not the moving
    one; a variant takes a few frames to bake after a displacement changes (0.5–1.3 s of CPU time per variant, on
    worker threads natively and on Web Workers in the browser, where the render thread still copies each result in —
    up to ≈ 5 ms for a 124 k-triangle variant — and Lumen re-uploads its scene when a scene's variants are in, as on
    any structural change); a world-dependent displacement of a moving instance keeps the variant baked where the instance stood;
    meshes baked without a distance field (`SdfMode::None`) stay out of Lumen, displaced or not. A graph displaces
    only while it has one of the 16 extended resolve pipelines (else its materials stay flat); every displacing
    graph of the scene runs in one dicing permutation behind a switch on the material, compiled when the set
    changes (a hitch while a graph is edited). Dicing uses a uniform 2^k grid per patch with
    snapped edges instead of UE's precomputed table of patterns for every factor triple (more micro-triangles for long
    thin patches; F is set by the finest edge), and splitting is binary at edge midpoints (UE splits with wider
    factors). Hard edges (split normals) and UV seams under a UV-mapped height crack, as in UE: the two sides
    displace a shared position differently. Patches are culled by frustum (main view) or page (shadows) only, not by
    the HZB or backfacing. Micro-triangles larger than 4096 pixels (a cliff in the height) are dropped and ones
    crossing the near plane culled — there is no hardware fallback for them; the shadow pages' hardware fallbacks
    (local light items reaching the near plane, dynamic pages without a scratch tile, dynamic casters of split pages)
    draw displaced clusters undisplaced. Lists hold 512 k patches per split queue and 256 k per dice level: when
    they overflow, the dice rate is raised until they fit (HUD). The visibility payload names the base triangle
    and a second per-pixel target the point on it (16-bit barycentrics, where UE's visibility buffer encodes the
    patch and micro-triangle); texture derivatives are the base plane's. Skinned, alpha-masked, translucent and
    voxel geometry is not tessellated; the wind does not
    rotate displaced normals.
15. **Browser start-up** (round 6, `firefox-freeze`, `browser-startup`): the browser compiles every pipeline in its
    GPU process (WGSL → HLSL → DXC on Windows) — Firefox one at a time on every visit, Chrome through wgpu's
    synchronous calls one at a time too on a first visit. UE ships precompiled shaders and a PSO cache. Helios
    compiles progressively instead, like UE's asynchronous compilation with fallbacks (`pipelines.rs`,
    `docs/round6/browser-startup.md`): the first image needs only the core pipelines, and shadows, Lumen, clouds,
    fog, translucency, tessellation and occlusion culling come on as theirs arrive; Chrome compiles them in parallel
    (warmed through `create*PipelineAsync`, then made from Dawn's cache), Firefox the core set while the scene bakes
    and the rest one batch per frame. The first image now takes ~9 s in a fresh Chrome profile (was 26–41 s) and
    11–15 s in Firefox (was 36–50 s) on this shared Ryzen 5 7600X; the full set ~11 s in Chrome and 33–51 s in
    Firefox, about when it used to show its first image (a slower CPU waits proportionally longer). Still simpler:
    the core set itself is ~4 s of Firefox compiling (the material resolve 0.6–1.2 s per shading model), the scene's
    content pipelines are found only after the bake, a new scene or a debug view compiles its new pipelines in the
    frame that needs them (the canvas holding its last frame), and the scene bake (3–6 s) runs on the main thread.
    Round 8 (`browser-bake-worker`, `docs/round8/platform.md`): the scene's meshes bake on Web Workers, spread
    over up to four by a deterministic least-loaded assignment (the page only runs the builder without bakes,
    ≈ 0.3 s), and a stand-in of the scene queues its content pipelines while they bake; scene switches bake there
    too. Still simpler than UE, which cooks offline: every visit bakes the scene again (a bake takes as long as its
    largest mesh: the valley's terrain, 2.5 s on a page thread), imported glTF models bake on the page, and the
    device needs 20 sampled textures per
    shader stage (WebGPU guarantees 16; desktop browsers offer 48–64) — there is no reduced variant for smaller
    adapters. `shader-f16` is requested and wired up (`half.wgsl`), but no pass uses it: TSR and MegaLights'
    denoiser gained nothing on the RTX 3090, where UE 5.8.1 runs TSR in 16-bit by default.
    Round 8 (`webperf`, `docs/round8/webperf.md`): a frame whose scene sync queues pipelines (a new scene, an
    editor change) is held — the canvas keeps its last image — while they compile in the background, instead of
    compiling them inside the frame (a switch to the gallery held the canvas 18–26 s in Chrome, now ≈ 3–4 s); the
    valley's Firefox set shrank from 214 to 200 pipelines (45 → 38.5 s of compiling: the resolve's tessellation
    permutation used from the start, the dicing two levels per pipeline, classic translucency only for refracting
    surfaces, the underwater view on demand). Still simpler: the new scene's pipelines compile after its bake,
    not during it (its stand-in cannot be synced while the old scene renders: one GPU scene), so a switch still
    shows a frozen frame for the compile (Firefox 12–16 s for the gallery's ten resolve pipelines, one at a time), and
    AVBOIT's front / back variants (≈ 10 s of the valley's Firefox compile) are separate pipelines.
16. **Coloured light through glass** (round 7, `glass-shadows`): the sun's map of the glass panes keeps one pane
    per texel (light through two panes in a row takes the nearer one's colour) and spans every visible pane (glass
    spread over a large world gets coarse texels); Lumen's voxel surface cache and the volumes' lighting cache still
    see glass as clear for the sun (`docs/round7/glass-shadows.md`). Local lights (round 8): only the 8 nearest the
    camera whose range reaches a pane get a glass map (one pane per texel as for the sun, 256² per cube face, a 2×2
    box filter, no blur with the distance behind the pane); the rest, and the fog, particles, volumes and Lumen for
    every local light, see glass as clear; spot lights render all six faces (`docs/round8/translucency.md`).
17. **Nanite culling** (round 8, `nanite-culling-split`): no persistent-thread BVH traversal — every view runs a
    fixed chain of six dispatches (instances → pages → clusters with their argument passes), the twelve sun clipmap
    levels one after another, so fixed per-view costs dominate shadow culling (a clean level still costs its
    dispatches and barriers). UE 5.9's simple/complex split is per frame (a lean specialisation when nothing
    deforms), not per instance: windy frames cull every instance with the deformation bounds code. There is no
    multi-level DAG step: a per-mesh level table skipping finer levels at once cost as much as it saved and was
    removed; coarse pages above the LOD cut still send every cluster to the cluster test
    (`docs/round8/nanite-culling.md`).
18. **Terrain and rivers** (round 8, `mesh-terrain`, `shallow-water-rivers`): the mesh terrain replaces a rectangle of
    the landscape, not the whole landscape (no world-sized mesh terrain, no streaming of terrain regions, no paging of
    modifiers); its modifier stack is code (rock boxes with taper, capsule tunnels, polyline channels, flatten — no
    sculpt or paint brushes, no editor tools, no spline modifiers, no erosion), and it is meshed once when the scene is
    built (no live re-meshing of edited chunks; ~1 s of surface nets for the valley's 80 × 80 m natively). Resolution
    is two fixed levels per chunk (0.25 / 0.5 m) joined by overlapping skirts rather than a crack-free transition mesh
    (Transvoxel); the seam with the landscape relies on the landscape's grid. Its material is the landscape's layers
    evaluated per pixel (no runtime virtual texture for the mesh terrain, no per-region paint layers or weight maps);
    its cliff rock is one procedural texture at two scales plus colour from per-vertex strata and cavity (no
    stochastic or texture-bombed sampling, no scanned rock materials, no displacement of the cliff faces by Nanite
    tessellation; moss and dirt follow the normal and a cavity probe, not an erosion or deposition simulation; the
    strata colour is only as fine as the 0.25 m vertices — finer laminae are the texture's).
    Collision is the CPU triangle grid (no physics engine; the camera and placement use it). Rivers: one simulated
    river per scene on a fixed grid (no spline-defined river, no grid following the river's course, no paintable water
    input in the editor — sources are code); the shallow-water model is the virtual-pipes scheme (no momentum
    advection, so no hydraulic jumps or eddies behind obstacles beyond what the pipes' flow does; Manning friction only;
    first-order, a little diffusive); foam is a heuristic of divergence, shear, surface slope and speed (no aeration
    model, no spray or splash particles at the cave's step). Obstacles are static (boulders baked into the bed; the raft and other moving
    instances don't push water, nothing floats on the river), the river has no underwater view, Lumen and shadows don't
    see its water, and its surface is the sim grid (no tessellation beyond it; waves are the lake's normal-mapped ones
    advected by a two-phase flow map, which can show its period as gentle pulsing on slow water).
