# Lumen mesh cards capture their materials (round 6 follow-up: `card-albedo`)

Design note. The README carries the user-facing text (the *Mesh cards* paragraph of *Lumen-style GI*, the
*Material graphs* bullet, gaps 2 and 3 of "Status vs. Unreal Engine 5"); this file records why it is built the way
it is, and the measurements.

## What

Lumen's mesh cards stored, per ~10 cm texel, the depth and normal of the surface they see, and the card lighting
reflected each texel's light with its material's *average* albedo — base colour × the texture's 1×1 mip, the
number the global distance field stores per voxel. Bounce light therefore had the average colour wherever a
material's colour varies: the gallery's mushroom bed (one graph material: red caps with white spots, white stems,
gills, green moss) bounced moss green under its red caps (`docs/round6/lumen-displacement.md`, "Remaining limits"),
a stained-glass window glowed in the average of its panes, a lava graph bounced its base colour.

Cards now carry the material per texel, as UE5's surface cache does:

* **Per-texel inputs** (CPU, with the geometry capture): the card rasteriser records which triangle each texel sees
  and its barycentrics (`CardConfig::sources`, `Card::sources` in `helios-geometry`); `cards.rs` turns them into
  the texel's texture coordinates (UV set 0), its UV footprint and — for a displaced variant — the offset of the
  *base* point the texel's surface point was displaced from (`SurfaceCapture::attributes`, 16 bytes per texel:
  uv as 2 × f32, offset as 3 × f16, footprint as f16).
* **Material capture** (GPU, `lumen_card_capture.wgsl`): per (surface, material) pair, the material is evaluated at
  every texel exactly as the G-buffer resolve evaluates it — `material.wgsl`'s shading models (standard image and
  procedural textures, rock, metal, foliage, glass), and for graph materials the graph's generated surface code
  (`graph_eval` + `graph_surface`, with its parameter values) — filtered for the texel's footprint, into an rgba8
  **albedo atlas**: `sqrt(albedo)` and the emission as `e / (1 + e)` (`e` = emitted radiance as a multiple of the
  albedo, the G-buffer's emissive channel).
* **Lighting** (`light_cards`): a lit texel reflects its light with the captured albedo — at coarse lighting mips
  the average of the up to four captured texels it samples — and adds the captured emission. Pairs not captured
  yet (and terrain / water materials) keep the average, so nothing changes for them.

Nothing downstream changes: gather, reflections, radiance-cache probes and Lumen Lite read the card lighting as
before, which now has the surfaces' colours.

## Design

### What is captured, and how often

* **Per (surface, material) pair.** Cards were captured per mesh (or displaced variant) and shared by every instance;
  the albedo depends on the material too. Each pair gets card records of its own (the surface's cards, pointing at
  the shared depth/normal texels and at its own albedo region); an instance's record points at its pair's. Pairs
  are cheap: 18 card records each, plus the albedo texels.
* **Per instance where the material depends on the placement**: textures projected in world space, a rock's moss
  (it grows on faces pointing up *in the world*), graphs that read the world position, the world normal or tangent
  frame, the camera or object axes (`placement_dependent`: the mushrooms read `WorldPosition` for their cells). Such
  instances get their own pair while those pairs fit 1 M texels (`PLACED_BUDGET`); further instances share their
  pair's capture, made at its first instance's placement (statistically the same texture; a rotated rock's moss is
  then on the wrong side).
* **When**: after a scene (re-)upload, within 1 M texels a frame (`CAPTURE_TEXELS_PER_FRAME`, at least one pair);
  when a graph or a material's parameter values change (`sync_materials` compares a content hash per pair: the
  material, the scene's textures, the graph code and parameter values, the placement); other material and texture
  edits re-upload Lumen's scene anyway. A re-upload keeps the albedo of unchanged pairs: their regions are copied
  from the old atlas into the new one (one `copy_texture_to_texture` per card). A displaced variant is a new
  surface (its own capture serial), so a new variant's pairs are captured fresh.
* **Budgets**: the albedo atlas holds at most 4 M texels (16 MiB, `ALBEDO_BUDGET`, the capture atlas's budget);
  pairs beyond keep the average. Texel density is the cards' own (~10 cm, coarser for very large meshes): the
  albedo has exactly the depth/normal capture's texels.

### Evaluating the material where the resolve does

The capture rebuilds the resolve's `SurfaceInput` per texel: object position (the card texel's surface point, plus
the base offset), world position, world normal and the normal matrix through the capturing instance's transform,
a tangent frame from the normal, UV and its footprint as the "screen derivatives" (and the card texel's two axes
on the surface's tangent plane as position derivatives), so texture mips and footprint-filtered noise average the
material over the texel instead of point-sampling it. The pipelines are specialised like the resolve's: one per
shading model (`MATERIAL_KIND`) and one per graph (its surface code in place of `material_graph_off.wgsl`, the
displacement function cut off — the geometry is already displaced). The graph pipelines bind the slab table (for
the material's parameter offset) and the graph parameter block; a graph the resolve does not run (device limits,
more than 16 graph pipelines) is captured with its material's own shading model, as the resolve renders it then.

**Displaced variants**: the resolve evaluates a displaced pixel's material at the *base* point it was displaced
from (positions, UVs and derivatives of the undisplaced surface), so one pattern can drive both the displacement and
the colours. The variant's refined points already carry their base position and UV; the bake keeps them
(`DisplacedVariant::base_points`, `corners`) and captures its cards with them, so a texel on a mushroom's cap is
evaluated at the base point that became the cap — red — not at the displaced position, which lies over the next
ring of the pattern.

### Alternatives rejected

* **Rendering cards through the visibility-buffer pipeline** (a card view per card through culling, raster and the
  material classify/resolve, as UE renders card captures): exact by construction, but every card would need a
  culling view, the streamed clusters resident, and the resolve's 30-odd bindings (visibility buffer, SVT feedback,
  RVT requests, G-buffer targets) adapted to atlas regions — a second renderer path for a one-off capture. The CPU
  card rasteriser already knows which triangle every texel sees; interpolating attributes there and evaluating the
  same material functions on the GPU gives the same result for a fraction of the code.
* **Evaluating the material in `light_cards`** (no atlas): every texel of every page lit per frame (64 k a frame)
  would evaluate its material, and the card lighting pass would need every graph's code — one pipeline per graph
  for a pass that lights all instances together.
* **Storing irradiance and multiplying by the albedo at sample time** (UE's final-lighting combine): sharper
  colour at coarse lighting mips, but every pass that samples cards (screen probes, reflections, the radiance-cache
  update, Lumen Lite) would bind the albedo atlas and fall back to the average for uncaptured pairs (the materials
  table and texture arrays in each), paying a texture load per card tap on every ray; lighting with the albedo
  costs it once per lit texel.
* **A CPU material evaluation**: graphs exist only as generated WGSL (a second evaluator would drift), and
  procedural textures are generated on the GPU.
* **Keeping the texel inputs on the GPU** (an attribute atlas next to the capture atlas): 16 bytes per texel of GPU
  memory for data needed only while capturing; they stay on the CPU (with the captures, which re-uploads need
  anyway) and are uploaded per capture batch.

## Measurements

{{MEASUREMENTS}}

## Tests

* `helios-geometry`: every texel's source interpolates its triangle's corners to the point the texel sees (layer 1
  under a second slab included); nothing is recorded without the flag.
* `helios-render`: a tiled quad's texels carry their UVs, the base offset (decoded from f16) and the UV footprint;
  the UV footprint follows the projection (0 edge-on); half-float rounding and saturation; a graph's surface code
  leaves out the displacement function; the capture shader validates (naga + the Tint bitwise check) without a
  graph and with every preset and every node type's test graph.

## Remaining limits

* Texel density is the cards' ~10 cm: detail finer than a texel bounces as its footprint's average (a mushroom cap
  of 8–17 cm is one or two texels). The card lighting's bilinear filter does not check depth across texels, so a
  coloured texel's light also tints its neighbours (red halos around the caps in view 14).
* Not captured: vertex colours and the second UV set (UV set 0 stands in); a graph's time and camera inputs are
  frozen at capture (an animated lava bounces one moment of its glow).
* Placement-dependent materials beyond 1 M texels of per-instance pairs share the first instance's capture.
* The voxel surface cache and the global distance field keep the average albedo (distant and uncarded surfaces).
* The capture inputs stay on the CPU: 16 bytes per captured texel ({{CPU_MEMORY}}).
* Terrain and water materials keep the average (heightfields have no cards anyway).
* A capture is as fresh as its scene upload: until a displaced variant arrives (seconds in the browser), the
  undisplaced bed's cards show the mushrooms' pattern evaluated on the flat bed.
