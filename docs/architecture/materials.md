# Materials, Substrate-lite and material graphs

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Materials** (`textures.rs`, `gbuffer.wgsl`). Textures are synthesised on the GPU (18 patterns: ashlar, marble
  tiles, wood, slate, stained glass, bark, leaves, moss, forest floor, …) into two texture arrays with full mip
  chains (no bindless in WebGPU); imported images are area-resampled into the same layout (base colour; normal, roughness,
  metallic), one layer per material. **Ambient occlusion** maps take the base colour's alpha in that layer: opaque
  materials never read it (alpha tests have their own mask array), so AO costs no memory, no atlas and no extra texture
  fetch (BC7 and the sRGB arrays keep alpha linear; sharing a BC7 block costs the base colour a little: 47.2 → 43.8 dB
  PSNR on DamagedHelmet, 54.4 → 53.2 dB on WaterBottle, the AO itself 42 / 53 dB). It reaches the lighting in the fraction of the G-buffer's
  shading-flags channel and, like UE's material AO, darkens indirect light only — Lumen's diffuse GI or the sky, and
  reflections through the specular occlusion of `sky visibility × AO` — never direct light. AO sampled with other
  coordinates than the base colour (e.g. the second UV set) or on blended materials (whose alpha is coverage) gets a
  layer of its own, read from its 512² array layer (not streamed: AO is low-frequency, and a second streamed lookup
  would cost every pixel of the resolve). Each texture slot (main layer, emission, occlusion) has its own **UV transform**: UV set 0 or 1, an
  affine `KHR_texture_transform` (folded with the material's tiling) and wrap modes applied in the shader
  (clamp-to-edge, mirrored repeat; arrays and atlases repeat), so streamed tiles are requested for the coordinates
  actually sampled, alpha tests use the base colour's, and per-triangle tangent frames come from the transformed UVs
  (rotated normal maps stay right). **Vertex colours** multiply the base colour (their alpha the coverage of blended
  materials). The second UV set and vertex colours are optional attributes of the compressed clusters
  (`encoding.rs`): a header bit adds 4 words (bit widths, UV1 and colour minima) and per-vertex offsets with just enough
  bits (RGBA8 colours, RGB on a square-root curve); meshes without them encode bit-identically to before. Procedural
  vegetation's *plant data* (wind weight, baked ambient occlusion, 2 × 5 bits; [*Grounded foliage*](foliage.md)) is a
  third one (extension bit 29); its occlusion multiplies the material's AO. The
  rarer features — second UV set, wrap modes other than repeat, occlusion layers of their own — live in a second
  permutation of the resolve pipelines (`UV_EXTENSIONS`), compiled on first use for scenes whose materials need it;
  other scenes pay for transforms, AO and the optional-attribute check only (Sponza resolve 0.28 → 0.31 ms, best of
  8 at 1600×900).
  Pixels are classified by shading model per 8×8 tile and each model (standard, rock,
  terrain, metal, foliage) is resolved by its own pipeline over just its tiles (classification slots 0–7; slots 8–23
  are the extended pipelines of *Substrate-lite* and *Material graphs*). UV mapping uses analytic
  barycentric derivatives for proper filtering; per-triangle tangent frames replace stored tangents; triplanar
  mapping is available in world or object space. The G-buffer resolve and the hardware shadow page raster
  compile skinning only into a `SKINNING` permutation used while the scene has skinned instances (see *Skeletal
  meshes*), and the debug views' recolouring of the G-buffer only into a `DEBUG_VIEWS` permutation used while a
  debug view is shown: kept in the lit pipelines, a dozen debug colourings kept their inputs (cluster, triangle,
  instance record, barycentrics) alive across the material evaluation, and the resolve ran 1.4× slower (cathedral at
  1440p 1.01 → 0.73 ms, forest 0.99 → 0.85 ms).
* **Substrate-lite** (`substrate.rs`, `substrate_gpu.rs`, `substrate.wgsl`, `brdf.wgsl`, `resolve_extended.wgsl`,
  `subsurface.rs`, `subsurface.wgsl`; UE 5.7's Substrate reduced to one slab per material). `Material::slab`
  (`Slab`) extends the metallic-roughness base with the dielectric's F0 / F90 from an **index of refraction** and a
  **specular colour / factor** (`KHR_materials_ior`, `_specular`), **anisotropic** GGX (Burley's distribution,
  Heitz's height-correlated visibility, a tangent rotated by `anisotropy_rotation`), a **clear coat** (a second,
  colourless GGX interface with its own roughness and normal), a **sheen** (Charlie distribution with Neubelt's
  visibility, as `KHR_materials_sheen`), **subsurface scattering** (wrapped diffuse plus a screen-space diffusion)
  and **thin diffuse transmission** (leaves, paper), and an **unlit** flag. Layering is energy-aware: the coat
  takes `coat · F(n·v)` and `coat · F(n·l)` from everything below it, the sheen its directional albedo, both from
  the view and the light side. **Data path.** The G-buffer stays as it is for plain materials. A material with slab
  features (or a graph, below) is classified into an *extended resolve pipeline* — one per distinct (shading model,
  graph), so its 8×8 tiles are resolved apart from the plain ones (the classification bitmask grew from 8 to 16
  slots, the tile buffer accordingly). The extended pipeline writes one **rgba32uint slab texel** per pixel
  (features, material index, clear coat; base normal octahedral under a coat or the anisotropy tangent; sheen colour
  and roughness; F0 and the subsurface / transmission amount; 16 bytes, allocated only while some material is
  extended) and sets a `SURFACE_SUBSTRATE` shading flag, so lighting reads it only there. With a clear coat the
  G-buffer's normal and roughness are the coat's, so Lumen traces reflections for the top interface; the base's go
  into the texel. Per-material constants (F90, subsurface colour and radius, transmission colour, texture layers) sit
  in a 28 KiB uniform slab table. Voxel foliage pixels ([*Nanite Foliage*](foliage.md)) in an extended pipeline evaluate their
  material the same way, slab texel included. **Every lighting path, one BRDF.** The slab lobes are functions of
  `brdf.wgsl` (`shade_slab`, `evaluate_slab`, `slab_ambient_specular`), the file the classic light loop and
  MegaLights share; the deferred lighting (sun with its VSM, clustered local lights, transmitted back light with
  the back side's shadow, image-based light: the coat reflects Lumen's traced reflections, the base layer under it
  the prefiltered sky / GI along its own — for anisotropy bent — normal) and MegaLights' shading evaluate them. The
  passes that read slabs are *variants*: the same source composed with `substrate_read.wgsl` in place of
  `substrate_read_off.wgsl` (`shader::compose_variant`), compiled the first time a scene has slab materials and used
  only while it does — scenes without them run the unchanged shaders. **Subsurface scattering** (UE's subsurface
  profile, Jimenez's separable SSS): the lighting writes the diffuse light of scattering pixels apart, demodulated
  and attenuated by the fog in front of them; a horizontal compute pass and a vertical full-screen pass blur it
  with a two-Gaussian approximation of Burley's diffusion profile, per channel scaled by the material's scattering
  colour and radius (projected to pixels per depth), 7 jittered taps per side that skip other surfaces and depth
  steps, and the vertical pass adds it back × the diffuse colour through additive blending (storage textures cannot
  read-modify-write the HDR format), so specular light and texture detail stay sharp. `PageUp`
  (`RenderSettings::substrate`) switches slabs off — every material renders as its plain base — for A/B
  comparisons. Debug view **24** shows each pixel's features (clear coat blue, sheen magenta, subsurface red,
  transmission green, anisotropy yellow, specular cyan, unlit white, blended; brighter per layer; striped where a
  graph produced them; plain grey). **glTF**: `KHR_materials_clearcoat`, `_sheen`, `_ior`, `_specular`, `_unlit`
  and `_anisotropy` (factors) become slabs; their textures are rearranged on the CPU into two optional texture
  layers per material (sheen colour + sheen roughness + coat normal + coat roughness + coat; specular colour +
  specular), sampled from the 512² arrays with the main texture's coordinates; `KHR_materials_transmission` makes
  the material glass (the forward translucency pass, refracting), tinted by the base colour and
  `KHR_materials_volume`'s attenuation over its thickness. Lumen's mesh cards capture the base albedo per texel
  (slabs don't change GI bounces). The extended path needs 17 sampled textures per shader stage (WebGPU's default limit is
  16; device creation requests up to 24 where the adapter offers them): with 16, slab materials render as their plain
  base. Cost: scenes without slabs render the same images (valley, cathedral, forest and Sponza differ from the
  previous build by 61–70 dB PSNR, as much as two runs of it differ) and cost the same; the gallery (eleven slab materials, three graphs, about half the screen
  extended) resolves in 0.41 ms and lights in 0.42 ms with slabs, 0.38 ms lighting with them off, the subsurface
  passes take 0.02 + 0.03 ms. Memory only while used: the slab texels 16 bytes per internal pixel (23 MB at
  1600×900) while some material is extended, the subsurface light and its diffusion 2 × 8 bytes per pixel while some
  slab scatters; the classification's tile lists doubled (1.4 instead of 0.7 MB at 1600×900).
* **Material graphs** (`crates/helios-material`, `material_graphs.rs`, `material_graph.wgsl`; UE's Material Editor
  and material instances). A `MaterialGraph` is plain serde data — nodes (`Op`) with canvas positions and the
  constants of their unconnected inputs, links from output to input pins by name — stored in scene files.
  45 node types: constants, scalar / vector / texture **parameters**, texture sample (the material's layers,
  streamed through the SVT path), triplanar, mesh UVs (both sets, tiling), world / object position, vertex normal
  and colour, time, camera vector, math (add … divide, lerp, power, min, max, clamp, saturate, one-minus, abs,
  sine, cosine, frac, floor, step, smoothstep, dot, cross, normalize, length, append, component mask), footprint-
  filtered fractal **noise**, **fresnel**, **panner**, reoriented **normal blend**, **if**, and the output (base
  colour, metallic, roughness, specular, tangent-space normal, emissive, ambient occlusion, the slab parameters:
  clear coat and its roughness, sheen colour and roughness, subsurface, transmission, anisotropy — and the
  **displacement**, a height or a vector, which drives Nanite tessellation: see there). **Compilation**
  (`helios_material::compile`): ids, links, pins, a single output and cycles are validated, value widths inferred
  (float to vec4, scalars broadcast) and every problem reported with its node ("node #4: Add: cannot combine a
  vec2<f32> with a vec3<f32>"); the result is one WGSL function `graph_eval(m, input, params, vertex_color,
  defaults) -> GraphOutput`, every operation parenthesised and every identifier generated (Tint-safe), plus the
  parameter slots and the connected outputs. **Derivatives without hardware**: the resolve is a compute shader over
  the visibility buffer, so values that feed a texture coordinate, a triplanar or noise position carry their screen
  derivatives through the math by the chain rule (forward-mode differentiation from the resolve's analytic UV and
  position derivatives) — texture filtering, the SVT's tile requests and noise filtering work in graphs; other
  values pay nothing. **Pipelines**: a graph's code replaces `material_graph_off.wgsl` in the extended resolve,
  compiled at runtime (WGSL shader modules are created in the browser too) and cached by the code's FNV hash, so
  editing a graph compiles one module, identical graphs share one, and moving nodes compiles nothing. The same
  surface code (without the displacement function) is composed into Lumen's card capture
  (`lumen_card_capture.wgsl`), so a graph's colours and emission reach the surface cache and its bounce light;
  editing a parameter recaptures that material's cards.
  **Material instances**: `Scene::set_material_parameter` overrides a parameter for one material; values live in a
  16 KiB uniform block (one vec4 per parameter, the material's offset in its slab record), so changing them
  recompiles nothing (the gallery's two water spheres are one graph with different colours). Unset texture
  parameters sample the material's own texture, so a graph assigned to an imported material picks up its images.
  Unit tests compile every node type (on a texture-coordinate path, so its derivative code too) and every preset
  into the extended resolve and validate it with naga and the Tint bitwise-precedence check. **Displacement**: the
  output's Displacement pin (a height along the normal) or its VectorDisplacement pin (a world-space offset in
  units of the material's magnitude, its length clamped to 1: overhangs; one of the two, both is an error) compiles
  to a second function, `graph_displacement(m, input, params) -> f32` or `-> vec3<f32>`, from the nodes it depends
  on (sharing the node code and the parameter slots; nodes feeding only one of the two functions stay out of the
  other), which the tessellator calls per micro-triangle vertex. Its inputs (`DisplacementSurface`: mesh UVs, world
  / object position, normal, time) carry the size of a dice segment as their "derivatives", so texture samples (the
  512² arrays, not streamed tiles) and noise are filtered for the micro-triangles. `Time` (and a panner left on the
  clock) is allowed: the compiler reports the displacement as animated (`CompiledGraph::displacement_animated`,
  next to `displacement_kind`), which makes the material's instances dynamic shadow casters, and reads the clock
  from the function's input so the renderer can evaluate the previous frame's surface for motion vectors; a
  static graph pays for none of it. Round 6: the compiler also bounds how fast an animated displacement moves
  (`CompiledGraph::displacement_speed`, an interval and time-rate analysis of its nodes, `helios_material::motion`)
  for each material instance's parameter values, and a **TransformVector** node (UE's Transform: World, Object =
  Local, Tangent) turns directions between spaces in both functions (see [*Nanite Tessellation*](tessellation.md)). Still rejected,
  with an error on the node: `CameraVector` / `Fresnel` (every
  view must displace alike), vertex colours, UV set 1 and the triplanar tangent normal (the tessellator does not
  read them). Texture sample and triplanar nodes gained a `Height` output (a
  procedural texture's height). The node editor is part of the browser editor (see [*Material graph editor*](editor.md)).
