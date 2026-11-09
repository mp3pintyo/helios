# Skeletal animation on virtual geometry (round 4: `skeletal-animation`)

README text for the merge: an Architecture bullet (section 2, after *Wind*), a row for the geometry table
(section 1), the glTF paragraph addition, the gaps for "Status vs. Unreal Engine 5" and the new flags / keys.

## Geometry table row (section 1)

| Skin weights | Imported joint influences (≤ 4 per vertex) are renormalised and quantised **once per vertex** to 8-bit weights summing to exactly 255 (`skin.rs`), then welded, clustered, simplified and compressed like any attribute: a cluster stores its joints as offsets from its smallest joint and the weights of all but the heaviest influence with just enough bits, in a skin section behind the vertex stream (a header bit; meshes without skin encode bit-identically). Simplification adds a **weight-deviation cost** (the weight a collapse hands to other joints), so strong weight seams collapse last | `skin.rs`, `encoding.rs`, `simplify.rs` |

## Architecture bullet (section 2, after *Wind*)

* **Skeletal meshes** (`skin.rs`, `skinning.wgsl`, `skin_palette.wgsl`; UE 5.5's Nanite skinned meshes). A
  `SkinHandle` is a joint palette (`Scene::add_skin`, `Scene::set_skin_pose` every frame,
  `Scene::set_instance_skin`); each matrix maps a rest vertex to its posed position in the instance's object space.
  Skinning — the linear blend of the vertex's joint matrices by its quantised weights, zero weights skipped, in slot
  order — is a pure function of the decoded rest vertex, its weights and the palette, evaluated by all four
  rasterisers (visibility buffer and shadow pages, hardware and software) and by the G-buffer resolve, as the wind's
  offset is: clusters and LOD levels sharing a vertex put it on the same bits, so cuts stay watertight under animation
  (the geometry tests pose a three-joint limb with four-influence vertices and check every LOD cut, full and partial
  residency, for open edges on the posed positions). Normals use the blended matrix's cofactor; motion vectors read the
  previous frame's palette (per-corner previous positions interpolated with the barycentrics — exact, skinning is
  affine per triangle), so TSR and motion blur follow the limbs (TSR vs. no-TAA on CesiumMan's walk at 100 % and 50 %:
  differences only along edges, no trails). The palettes live in one `Rgba32Float` texture (`textureLoad`, so no
  shader stage gains a storage buffer; the G-buffer resolve was at WebGPU's 8): per skin a header and six texels per
  joint (current and previous rows), rewritten each frame (24 joints: 2.3 KiB); instances find theirs through
  `GpuInstance::flags` bits 12..32 (no layout change). **Culling bounds**: per frame and skin the CPU derives, from
  per-joint rest spheres of the vertices each joint influences, a *posed sphere* (every joint's posed rest sphere — a
  blended vertex lies in the convex hull of its joints' transforms of it) and a *reach* `max |P c − c| + ‖A − I‖₂ r`
  (spectral norms in closed form) that bounds every vertex's displacement; frustum, HZB and shadow-page tests use the
  rest cluster / page / instance sphere padded by the reach, or the posed sphere where that is smaller (unit tests
  check both against random poses with scale). The LOD cut keeps the rest spheres, which keeps the DAG's monotonic
  errors (crack-free). **Shadows**: a skin whose pose changed is a dynamic caster in the static/dynamic page cache, as
  vegetation in the wind: cached pages under it re-render their dynamic casters over the static layer once it may have
  moved 2 texels, bounded per frame by the joints' motion `|ΔP c| + ‖ΔA‖₂ r`; the footprint is the rest sphere padded
  by the largest reach so far (it covers every pose a cached page may still show); a paused skin returns to the static
  layer. **Lumen**: as in UE, skinned meshes get no distance field or cards (the demo bakes them without); screen
  traces still see them. **Animation** (`helios_asset::animation`): glTF skins, node hierarchy and clips (translation,
  rotation, scale; `STEP`, `LINEAR` with shortest-arc slerp, `CUBICSPLINE` Hermite), an `AnimationPlayer` (clip, time,
  speed, looping, pause, crossfade). The palette is `inverse(mesh node) · joint · inverse bind matrix`, i.e. relative
  to the node that places the mesh, so the rest-pose bounds and LOD stay meaningful while `node · palette` is exactly
  glTF's pose. Debug view **17** colours each joint and blends the colours by weight.
  The G-buffer resolve and the hardware shadow page raster compile the skinning code only into a `SKINNING`
  permutation used while the scene has skinned instances (like `UV_EXTENSIONS`): unconditionally it took the forest's
  resolve from 0.39 to 0.74 ms and its page raster 5 % slower, although no pixel there is skinned; the visibility
  rasterisers, culling and page expansion carry it at no measurable cost.
  Cost (RTX 3090 shared with other agents; min / median of 5 interleaved runs, 1600×900, BrainStem — 61 k triangles,
  59 meshes, 18 joints) animated vs. the same model imported as a static mesh (`--no-skinning`): main-view raster
  unchanged (hw 0.02, sw 0.06–0.07 ms), cull 0.20 → 0.23 ms, material resolve 0.31 → 0.35 ms (the skinning
  permutation; paused: also 0.35), vsm expand 0.09 → 0.12 ms, shadow page raster (hw + sw + resolve) 0.02 → 0.29 ms —
  the moving caster's pages re-rendered every frame, which a paused model doesn't (0.02 ms: its pages stay cached);
  GPU frame 4.16 → 4.41 ms (paused 4.14). Compressed payload: +2.5 bytes per cluster vertex on the Fox, +3.3 on
  CesiumMan (+19 / +25 %), +0.3 on the mostly rigid BrainStem (+3 %). Scenes without skins (forest medium, 6
  interleaved runs vs. the base build): every pass within ±0.02 ms (resolve 0.39 → 0.41, page raster 0.88 → 0.87,
  frame 7.50 → 7.48 ms min); images differ from the base build no more than two runs of the base build differ from
  each other.

## glTF paragraph (section "The demo scenes", after the list of supported features)

Skinned meshes (`JOINTS_0` / `WEIGHTS_0`, any component type) play the file's animations: the first clip starts on
load, `-` (Minus) crossfades to the next clip and pauses after the last. Checked against Fox, CesiumMan, BrainStem,
RiggedFigure, RiggedSimple and SimpleSkin. The model is placed by its posed bounds at the start of the first clip.

## Status vs. Unreal Engine 5 (new item)

**Skeletal meshes**: linear blend skinning with up to 4 influences per vertex (glTF's first joint set; UE allows 8–12)
and 8-bit weights; no dual-quaternion skinning, morph targets (imported files with them render their base shape, with
a warning), cloth or physics. Culling bounds are per instance (posed joint spheres) and per cluster only as the rest
sphere padded by the instance's largest displacement — UE bounds clusters by their own influencing bones, which is
tighter for large motions (a waving arm makes every cluster's sphere grow by the arm's reach). The LOD error is that of
the rest pose: the weight-deviation cost only orders the collapses (coarse levels keep weight seams longest; a test
measures half the posed deviation), it does not enter the error. Skinned meshes get no distance field or mesh cards
(as in UE: Lumen sees them through screen traces only), and translucent (`BLEND`) skinned surfaces are drawn by the
forward pass in their rest pose. Shadow pages under a moving skin re-render its whole caster set each refresh (as for
the wind); the footprint grows with the largest reach seen, so a figure that once stretched far refreshes a larger
area. Root motion is not extracted (animated root joints move the posed bounds, not the instance).

## Flags and keys

* `-` (Minus): next animation clip with a 0.4 s crossfade; after the last clip the animation pauses, the next press
  starts the first clip again (single-clip models: play / pause).
* Debug view 17 (`--view 17`, `heliosSetView(17)`): skin weights (joint colours blended by weight; grey: unskinned).
* `--no-skinning`: import a model without its skins and weights (the bind pose as a static mesh; for cost
  comparisons).
* The HUD shows the clip, its time and whether it is paused.
