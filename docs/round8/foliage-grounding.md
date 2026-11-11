# Grounded, rooted foliage and campfire shadows (round 8: `foliage-grounding`)

Design note for three backlog items of round 8, first wave: **225** `campfire-foliage-shadows`, **226**
`foliage-root-pinning`, **227** `foliage-grounding`. The user-facing descriptions are the *Grounded foliage* bullet of
[`docs/architecture/foliage.md`](../architecture/foliage.md), the *Wind weights* paragraph of
[`docs/architecture/motion.md`](../architecture/motion.md) and the *Short-range AO* bullet of
[`docs/architecture/lumen.md`](../architecture/lumen.md); the remaining gaps are items 2, 7 and 10 of
[`docs/status.md`](../status.md). Images: `foliage-grounding/`.

## The complaint

Looking at the forest, the user reported (translated): *"the fire doesn't seem to cast shadows on the plants next to
it"*; *"the plants, e.g. the grass, look as if they float above the ground; their bottom also wiggles, which it
shouldn't — the part in the ground can't move, the part touching the soil must be fixed"*; *"the shadows are kind of
nothing, as if the plant weren't real but just pasted there; it clashes strongly with its surroundings"*.

Every one of these had a concrete cause:

| Symptom | Cause found |
|---|---|
| grass floats | plants were placed at the analytic `ground_height`, but the terrain is a 384² / 768² grid mesh (2.34 / 1.17 m cells) interpolating it: on every crest the mesh lies below the analytic height — **up to 14 cm** (medium) / **3.3 cm** (high) over the area where grass grows, mean 1.8 / 0.5 cm (`plants_stand_on_the_terrain_mesh`). Grass had no sink at all. Blades were also diamonds tapering to a point *at* the ground, spread over a ±15 cm square: a clump stood on 60 needle points. |
| bases wiggle | `wind_offset`'s *sway* grows with the horizontal distance from the instance origin and its *flutter* is unweighted, so blade bases (up to 21 cm from the clump origin) and a fern's crown moved at ground level (≈ 3–5 mm of flutter, 1–2 cm of sway on tree root flares). |
| pasted look | no contact cue at all: blades were as bright at the ground as at the tip; the ground under a clump got no occlusion (Lumen's distance fields hold no thin foliage; its screen probes, one per 16² pixels at half resolution, smear the little they see); nothing darkened a clump's crowded base. |
| no fire shadows | the fern and grass shadows **were** in the campfire's local virtual shadow map, but drowned: the ground was covered with a dark blotch pattern that looked like shadow. It is neither shadow nor acne — see item 225. |

## 226 — root pinning: a per-vertex wind weight

**Design.** The displacement of every vertex is now `wind_offset(...) * weight`, where the *weight* is a per-vertex
quantity baked by the vegetation generators and carried through the whole geometry pipeline:

* `Mesh::plant: Vec<Vec2>` (empty or one per vertex): `x` = the **wind weight** (attachment), 0 where the plant is held
  — its root, the part in the soil — rising to 1 along the plant where it moves freely; `y` = the baked ambient
  occlusion (item 227). The weight is defined by the *distance along the plant from its root*, not by height: a fern
  frond drooping with its tip near the ground still moves (weight 1), a grass blade's base does not.
* Generators (`vegetation.rs`): grass blades 0 at their root (4 cm below the ground) rising to 1 within the lower third
  of the visible blade; fern fronds `smoothstep(0.03, 0.35, s)` of the fraction `s` along the frond (leaflets take the
  stalk's value at their base and tip); trees, conifers and bushes `root_weights(mesh, free_height)` — 0 at y = 0,
  1 from 0.8 m (or a quarter of a small plant's height) up — on the wood, and on bushes' leaves too, so a leaf moves with
  the twig it sits on. `MeshBuilder::plant_vertex`, `add_leaf_with_plant` and `add_strip` carry it.
* Encoding (`encoding.rs`): an optional cluster attribute like the second UV set and the colours — extension word bit
  29, then per vertex 2 × 5 bits (wind weight, occlusion; fixed width, no per-cluster range: a handful of values per
  blade) after the colour. Meshes without plant data encode exactly as before; the DAG builder carries it through
  simplification and pruning copies like the colours (`plant_data_survives_simplification`). `.hvm` → **version 5**
  (`HVM5`; version 4 files have no bit 29 set and still load).
* Shaders: `geometry.wgsl` `vertex_plant` / `vertex_wind_weight` (only read when the instance moves:
  `wind_strength > 0`); `wind.wgsl` `wind_world_position(instance, strength, object, weight)` and
  `wind_previous_world_position(..., weight)` — every caller passes it: the hardware and software visibility
  rasterisers (`raster.wgsl`, `sw_raster.wgsl`), the shadow page vertices of both page rasterisers (`vsm_page.wgsl`),
  the G-buffer resolve's corners and its motion vectors (`gbuffer.wgsl`, interpolated weight at the pixel), so all
  passes still compute bit-identical positions. Voxels (`voxel_raster`, `vsm_voxel_raster`, `voxel_resolve`) and
  tessellated (displaced) surfaces pass 1: voxels carry no plant data and move rigidly at their centre (they stand in
  where a held base moves less than a voxel), and vegetation is not displaced.
* **Bounds stay valid untouched**: the weight lies in [0, 1] and multiplies the whole offset, so
  `|offset · weight| ≤ |offset| ≤ wind_reach` — culling (`wind_padded`), the shadow invalidation speed bound
  (`wind_speed`) and `WIND_BEND_BOUND` / `WIND_FLUTTER_BOUND` are unchanged, which keeps the parallel Nanite-culling work
  merge-safe.

**For the later bone-based dynamic wind** (UE 5.9): the weight is a clean, documented per-vertex scalar — the
attachment of a vertex to the rigid ground — independent of the procedural layers. A skeletal wind would put the
plant's bones into the existing skin attributes and keep this weight as the root's influence (or derive bone weights
from the same "distance along the plant" the generators already compute).

**Alternatives rejected.** A height heuristic in the shader (`smoothstep` of the height above the origin) fails
drooping fern fronds and slopes; storing the weight in vertex colour alpha would overload the base colour's coverage;
a per-cluster weight cannot vary along a blade.

**Result** (`foliage-grounding/wind-before-after.jpg`: the per-pixel change between two frames 0.5 s apart with
shadows, TAA and motion blur off, so only geometry motion shows): before, the blade edges change down to the ground;
after, the bottom of the tuft is still and the motion grows towards the tips.

## 227 — grounding

* **On the mesh, not the analytic height** (`forest.rs` `terrain_mesh_height`): the scene places every scattered
  object (and the campfire's stones, coals and logs) on the terrain *mesh* — `ground_height` sampled at the grid's
  vertices and interpolated over the grid triangle holding the point, exactly as the mesh is rasterised (each quad
  split along its (i+1, j)–(i, j+1) diagonal). The analytic height floated up to 14 / 3.3 cm (medium / high detail)
  above it. Nanite's simplification of the terrain adds its own error only where it is below a pixel.
* **Plants grow out of the soil**: grass blades and fern stalks start `ROOT_DEPTH` = 4 cm below the plant's origin, so
  a slope or a residual terrain error never shows a floating base.
* **Grass tufts** (`grass_clump`): blades rise from a tuft 6 cm in radius (they used to stand on a ±15 cm square) and
  fan out — outer blades lean further — widest at the base and tapering to the tip, arching away from the centre, as
  a strip of 3 segments (10 triangles a blade instead of 8). **Ferns** get a tapering stalk (rachis) following each
  frond from below the ground to its tip; before, a straight half-length blade left the leaflets floating.
* **Baked occlusion** (`Mesh::plant.y`): darker towards the crowded base of a tuft (inner blades darkest: 0.35–0.6 at
  the root → 1 at 70 % of the blade) and a fern's crown (0.35 → 1 at half the frond). The G-buffer resolve multiplies
  the material's AO by it, so it darkens indirect light only, like UE's foliage vertex AO into the material AO.
* **Short-range AO** (`short_range_ao.wgsl`, `RenderSettings::short_range_ao`, default on, `--no-short-range-ao`,
  `Home`, editor *Short-range AO*): UE's Lumen applies a screen-space short-range AO on top of its screen probes for
  exactly this — the occlusion of nearby geometry the probes and distance fields miss. In the lighting pass, per
  pixel, 4 screen directions (rotated per pixel and frame, TSR averages them) × 4 steps within 0.4 m (≤ 64 px at 900
  rows) keep each direction's horizon, the highest sample's `cos(v, n)` above the tangent plane (biased 0.15, faded
  out towards the radius); the mean of the horizons is the occluded fraction of the cosine-weighted hemisphere
  (horizon-based AO as HBAO/GTAO, without their slice integration). It multiplies the occlusion that darkens Lumen's
  diffuse light and the specular occlusion. Samples beyond the radius are ignored, so a blade in front of distant
  ground casts no halo. The ground under a tuft, around a fern's crown and where rocks and logs meet it darkens.
* **Terrain bump** (content): the forest floor's height-to-normal strengths 1.0 / 0.6 / 0.6 (litter, moss, dirt) →
  0.3 / 0.2 / 0.25, see item 225 — the ground's lumpy relief also read as noise next to the plants by day.
* **Sun shadows of grass and ferns** (checked, unchanged): the VSM resolves blade shadows near the camera (soft
  streaks at the 15° golden-hour sun, `foliage-grounding/side-after.jpg`); grass beyond ~20 m casts through its voxel
  LODs (stochastic cubes, static hashed noise per texel), fine at that distance. The foliage is real geometry (no
  alpha-masked cards), so no mask resolution issue arises. A screen-space contact shadow for the sun was not needed
  once the bases are rooted and occluded.

## 225 — campfire shadows of the plants

**Diagnosis.** `--view 15` (local shadow pages) showed the ground around the fire covered with dark grey blotches,
and the lit image with dark speckles that looked like the light's shadow. They are **neither shadow nor acne**:

* a receiver bias 4× larger left them unchanged (`foliage-grounding/campfire-debug-before-after.jpg`), and replacing
  the shading normal by the depth buffer's geometric normal for the filter's receiver plane did too;
* the debug view paints a pixel `0.08` grey where the strongest light contributes nothing, and that is
  `saturate(dot(n, l)) = 0`: with the fire 0.8 m high, the light reaches the ground at 9–25° from 2–5 m away, and the
  forest floor's normal maps (litter bump 1.0, moss and dirt 0.6) tilt a large share of the ground's shading normals
  away from it. Shading with the geometric normal instead removed the blotches and made the ferns' shadows plainly
  visible — it is the normal-mapped *shading* at grazing incidence, at the scale of the fern leaflets' shadows (10–20
  cm), that drowned them.

**Fix.** Content, not shader: the forest floor's bump strengths 1.0 / 0.6 / 0.6 → 0.3 / 0.2 / 0.25 (still visible
relief by day), and the fire's `source_radius` 0.25 → 0.15 m (a ~30 cm wide flame; the PCSS penumbra of a leaflet 0.3 m
above the ground 1.5 m from the fire is now ≈ 2–3 cm, so fronds and blades cast soft but readable shadows). With the
plants rooted (227), the fronds' and tufts' shadows now radiate from the fire across the clearing; the stones' and the
teepee logs' shadows were already right and are unchanged. **No change to `vsm_local*.rs/.wgsl`.**

**Rejected:** flattening normal maps for local lights at grazing angles in the lighting (a geometric normal per pixel
from the depth buffer; physically a bumpy ground under a grazing light *is* contrasty, and the cost lands in every
scene), a larger receiver bias (no effect, as shown).

## Measurements

TBD

## Remaining limits

* Voxel LODs move with weight 1 (no plant data in voxels; invisible at their distance), tessellated surfaces likewise.
* Plant data exists for the procedural vegetation; imported glTF foliage has none (weight 1, no baked AO) — UE content
  bakes the equivalent into vertex colours.
* The short-range AO is screen space: it misses occluders off screen or hidden behind nearer geometry, has no
  thickness model (a thin blade occludes like a wall within its 0.4 m), is noisy per frame (4 × 4 taps) and relies on
  TSR to average it; UE integrates it with the screen-probe gather and a bent normal.
* Plant instances stand on the terrain's grid mesh; a Nanite LOD of the terrain coarser than the full grid (far away)
  can still differ by its sub-pixel error.
* The terrain bump fix is content for the forest; normal-mapped ground under any other grazing light shows the same
  contrast (physically plausible for a heightfield; real moss and litter scatter light more softly).
