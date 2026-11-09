# Round 4 — materials: Substrate-lite slabs and material graphs

README text for the merge (Architecture bullets, "Status vs. Unreal Engine 5" gaps, flags and
hotkeys). Branch `w2-materials`.

## Quick start (README "Quick start" additions)

```powershell
cargo run -p helios-demo --release -- --scene gallery                 # the material gallery (also in the N cycle)
cargo run -p helios-demo --release -- --gltf ClearCoatTest.glb         # glTF PBR extensions
#   browser: ?scene=gallery; editor.html?scene=gallery → toolbar "Material Graph"
```

## The demo scenes (new table row)

| `gallery` | The material gallery: a paved courtyard at golden hour with a row of plinths — clear-coat car paint, velvet with sheen, subsurface skin, a lit wax candle and a marble egg, a bush whose leaves let a lamp's light through, an anisotropic brushed-metal pot, dielectrics of IOR 1.33 / 2.4 and a tinted specular, an unlit sign — and three material graphs: animated lava (a pool and an egg), stylised water (two instances of one graph with different parameters) and a terrace of wet stone with puddles | 33 instances |

## Architecture (new bullets for section 2, after *Materials*)

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
  features (or a graph, below) is classified into an *extended resolve pipeline* — slots 8–15 of the material
  classification, one per distinct (shading model, graph), so its 8×8 tiles are resolved apart from the plain
  ones (the classification bitmask grew from 8 to 16 slots, the tile buffer accordingly). The extended
  pipeline writes one **rgba32uint slab texel** per pixel (features, material index, clear coat; base normal
  octahedral under a coat or the anisotropy tangent; sheen colour and roughness; F0 and the subsurface /
  transmission amount; 16 bytes, allocated only while some material is extended) and sets a `SURFACE_SUBSTRATE`
  shading flag, so lighting reads it only there. With a clear coat the G-buffer's normal and roughness are the
  coat's, so Lumen traces reflections for the top interface; the base's go into the texel. Per-material constants
  (F90, subsurface colour and radius, transmission colour, texture layers) sit in a 28 KiB uniform slab table.
  **Every lighting path, one BRDF.** The slab lobes are functions of `brdf.wgsl` (`shade_slab`, `evaluate_slab`,
  `slab_ambient_specular`); the deferred lighting (sun with its VSM, clustered local lights, transmitted back light
  with the back side's shadow, image-based light: the coat reflects Lumen's traced reflections, the base layer
  under it the prefiltered sky / GI along its own — for anisotropy bent — normal) and MegaLights' shading evaluate
  them. The passes that read slabs are *variants*: the same source composed with `substrate_read.wgsl` in place of
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
  `KHR_materials_volume`'s attenuation over its thickness. Checked against ClearCoatTest, ClearcoatWicker,
  SheenChair, SheenTestGrid, SpecularTest, IORTestGrid, TransmissionTest, UnlitTest, AnisotropyBarnLamp and
  CarConcept. Lumen's surface cache keeps the average albedo (slabs don't change GI bounces).
  Cost: COST_TABLE
* **Material graphs** (`crates/helios-material`, `material_graphs.rs`, `material_graph.wgsl`; UE's Material Editor
  and material instances). A `MaterialGraph` is plain serde data — nodes (`Op`) with canvas positions and the
  constants of their unconnected inputs, links from output to input pins by name — stored in scene files.
  45 node types: constants, scalar / vector / texture **parameters**, texture sample (the material's layers,
  streamed through the SVT path), triplanar, mesh UVs (both sets, tiling), world / object position, vertex normal
  and colour, time, camera vector, math (add … divide, lerp, power, min, max, clamp, saturate, one-minus, abs,
  sine, cosine, frac, floor, step, smoothstep, dot, cross, normalize, length, append, component mask), footprint-
  filtered fractal **noise**, **fresnel**, **panner**, reoriented **normal blend**, **if**, and the output (base
  colour, metallic, roughness, specular, tangent-space normal, emissive, ambient occlusion and the slab parameters:
  clear coat and its roughness, sheen colour and roughness, subsurface, transmission, anisotropy). **Compilation**
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
  editing a graph compiles one module, identical graphs share one, and moving nodes compiles nothing.
  **Material instances**: `Scene::set_material_parameter` overrides a parameter for one material; values live in a
  16 KiB uniform block (one vec4 per parameter, the material's offset in its slab record), so changing them
  recompiles nothing (the gallery's two water spheres are one graph with different colours). Unset texture
  parameters sample the material's own texture, so a graph assigned to an imported material picks up its images.
  Unit tests compile every node type (on a texture-coordinate path, so its derivative code too) and every preset
  into the extended resolve and validate it with naga and the Tint bitwise-precedence check.
* **Material graph editor** (`web/editor/graph.js`, `editor/materials.rs`). In the browser editor a dark node
  canvas docks over the viewport (toolbar *Material Graph*): pan (right / middle drag), zoom (wheel), drag nodes,
  drag pin to pin to link (an input takes one link), Alt+click unlinks, right-click or double-click opens a
  searchable palette built from the engine's catalog (`heliosMaterialNodes`), inline editors for constants,
  colours, parameter names and defaults, UV sets, channels; *New from preset* (lava, wet stone, stylised water,
  blank), *Assign to material*; compile errors mark their nodes and list below. Every change is a scene-document
  edit (`{op: "graph"}`: undo / redo, saved in `.helios.json` under `graphs`, gestures merged into one undo step)
  and recompiles live. The Details panel shows a material's graph (assign / open), its parameters as sliders and
  colour pickers (material instance values, saved per material) and its Substrate slab (clear coat, sheen, subsurface
  and radius, transmission, anisotropy, IOR, specular, unlit).

## Status vs. Unreal Engine 5 (new gaps, item 3 *Materials*)

* **Substrate-lite** has one slab per material: no vertical layering of arbitrary slabs, no horizontal mixing
  (UE's Substrate mixes / layers / blends any number, with per-pixel slab counts and a closure-count–adaptive
  G-buffer); no hazy gloss, iridescence (`KHR_materials_iridescence`), dispersion, glint or fuzz beyond sheen, no
  specular profiles, no volumetric media. A slab with a clear coat ignores anisotropy (they share the texel's
  direction), and subsurface + transmission share one per-pixel amount (transmission then takes the material
  constant). Subsurface diffusion is screen-space (no light from behind thick objects except through thin
  transmission, no subsurface for translucent materials), uses the material's colour and radius rather than a
  measured profile, and under MegaLights the local lights' diffuse light of subsurface pixels is not diffused
  (MegaLights keeps diffuse light whole). Translucent materials (glass, water, `BLEND`) ignore slabs; glTF
  transmission is glass (refraction of the screen, not a thin-walled BTDF; `transmissionTexture` and
  `thicknessTexture` are ignored). The extension textures are sampled with the main texture's coordinates from the
  512² arrays (not streamed); `anisotropyTexture` is not supported (the factors are). Lumen's surface cache and
  mesh cards still see the material's average albedo (no slab, no graph): a lava graph glows on screen but bounces
  its base colour. Unlit colours are scaled to a white surface in full sun.
* **Material graphs** run in the G-buffer resolve only: no graph-driven opacity mask, world position offset,
  tessellation, translucent or decal materials (raster and translucency shaders are fixed); at most 8 distinct
  (shading model, graph) pipelines per frame (further materials render without their graph, with a warning) and
  1024 parameter slots in all; no custom HLSL/WGSL node, material functions, static switches, texture objects or
  per-node previews; noise and texture outputs carry no derivatives (a texture read at coordinates computed from
  another texture or from noise filters as if they were constant, i.e. at its finest level there); texture
  parameters are not editable in the UI
  (they default to the material's texture) or stored in scene files; time is the renderer's clock.

## Flags and hotkeys

* Scene `gallery` (`--scene gallery`, `?scene=gallery`, in the `N` cycle and the editor's scene switcher), with a
  camera tour (`Y`).
* **PageUp**: Substrate slabs on / off (`RenderSettings::substrate`, editor render setting *Substrate slabs*).
* Debug view **24** (`--view 24`, `heliosSetView(24)`): material features.
* Editor: toolbar *Material Graph*; Details → *Material graph*, *Substrate slab*. Page API `heliosMaterialNodes()`;
  edit ops `{op: "graph", name, graph}` and `{op: "material", params: {graph, parameters, slab, …}}`.

## Measurements

MEASUREMENTS
