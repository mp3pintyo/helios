# Nanite Foliage

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Nanite Foliage** (`helios-geometry/src/voxel.rs`, `assembly.rs`; `helios-render/src/assembly.rs`,
  `voxel.wgsl`, `voxel_raster.wgsl`, `vsm_voxel_raster.wgsl`, `voxel_resolve.wgsl`, `assembly.wgsl`; UE 5.7's
  experimental Nanite Foliage). Aggregate geometry — canopies, needles, fronds, grass — cannot be simplified as
  triangles without deleting pieces: component pruning thins it, and distant forests lose coverage and colour (the
  conifers turned into reddish skeletons of bare branches). Two parts:
  **Voxel LODs.** Meshes built with `BuildConfig::voxels` (the forest's foliage, ferns and grass) give every
  DAG group a voxel candidate: the *full-detail* triangles under the group, voxelised on a grid as fine as its
  budget allows (`voxels_per_triangle`, 4 per triangle the group keeps). Each voxel stores what the geometry
  inside it does to light, not its shape: its coverage seen along each axis — each *piece* (a connected leaf,
  needle or blade) covers its projected share of the face (`max` of the areas facing either way, so a two-sided
  or folded leaf counts once), independent pieces multiply what they leave open — an octahedral normal blending
  the leaves' principal orientation with the canopy's outward direction (the coverage gradient over the group),
  and its brick's mean texture coordinate and spread. Seen from a direction, a voxel's coverage is the leaves'
  projection over the cube's footprint, in optical depth (`sqrt(Σ(d_i τ_i)²) / Σ|d_i|`, exact for leaves facing
  an axis and for isotropic tangles; adding the axes linearly counts a leaf seen diagonally up to √3 times).
  Clusters hold up to 128 bricks of 4³ voxels (occupancy bits, position, first voxel, uv) and one 32-bit word per
  voxel. Where the voxels' error is below the simplified triangles' the group switches to voxels — at the
  triangles' error, so the LOD switches where it did — and every output cluster gets *voxel alternates* standing for
  the same source triangles, with the same LOD values; every group above is voxelised too. Both representations
  stay in the DAG, so `Insert` switches between the voxel cut and the triangle-only cut of the same DAG instantly,
  and all its invariants (monotonic errors, one side of every group boundary, page dependencies) hold in both; voxel
  groups' member sets get pages of their own per mode, so a mode never streams the other's data. Voxel clusters are
  always rasterised in compute: a workgroup per cluster takes its voxels (up to 8192) in turn, projects each cube
  from its centre (moved by the wind like a vertex there: voxels shift rigidly, they don't flutter) and the
  instance's half-voxel axes, rejects voxels between pixel centres with a conservative bound, and covers a pixel
  where one of its three front faces does, at that face's depth, with the probability of the voxel's coverage along
  the view (stochastic transparency: interleaved gradient noise animated per frame and decorrelated per voxel,
  averaged by TSR). It shares the software rasteriser's depth / payload passes and target (payload = cluster slot,
  brick), so the merge with the hardware and software triangle paths is unchanged; the G-buffer resolve finds the
  voxel from the merged depth (the brick cell under the hit point, with its exact wind offset) and shades it with
  its normal (facing the viewer like a two-sided leaf) and the material sampled at the brick's uv with its spread as
  the footprint (the leaves' average colour) — through its shading model's pipeline, or the extended one (slab,
  graph) where its material has one. Shadow pages (sun clipmap and local light faces) rasterise voxels the same way
  in page texels, with the triangles' slope-scaled bias and hashed, static noise per absolute texel (cached and
  refreshed pages agree). Culling treats voxel clusters like triangle clusters (frustum, HZB, LOD, streaming);
  `.hvm` is version 4 (5 since round 8's plant data, below; version 4 files still load). **Assemblies** (`Scene::set_assembly`): a mesh can be made of *parts* — meshes placed by
  transforms in its space — so the forest's conifers share six needle sprays per species (one per branch span, as
  many copies as branches) instead of each species baking all its needles (`vegetation::conifer_assembly`;
  `broadleaf_assembly` builds broadleaf trees from leaf sprays the same way). The parts are resolved on the GPU
  before culling: each instance of an assembly tests its *base* — one ordinary virtual mesh built from the parts at
  their roots, merged in place and simplified (and voxelised) further, its errors offset by the parts' worst root
  error so they continue where the parts end — with the main view's LOD rule; where the base's finest level is too
  coarse, the instance writes one record per part into part slots after the scene's instances (its transform times
  the part's, the assembly's bounds and plant origin for the wind, and per-(assembly instance, part) visibility bits
  for two-pass occlusion) and flags its base (`EXPANDED`) so culling skips it. Every view culls the same parts, so
  shadows always match the view. Culling is hierarchical (round 6, [`foliage-culling.md`](../round6/foliage-culling.md)):
  views never test the part records as instances; each assembly instance owns one more *culling record* between the
  scene's instances and the part slots (`PART_LIST`), which the expansion fills with what its parts share (the
  assembly's bounds, wind and plant origin, flags) and hides when the base is drawn. Instance culling tests that
  record once per view, and its batches are its parts' pages — `stride` lanes per part, the most pages of any part —
  each lane resolved to its part's page and tested with the part's transform, so a conifer's 176 sprays of 6 pages
  are culled in 17 page workgroups instead of 176, and their small pages' clusters as cluster candidates. Parts inherit `MOVABLE`,
  hidden bases expand nothing, and part slots record their owning instance (`part_owner.wgsl`), so picking returns
  the tree and the selection outline covers its parts. Up to 32768 part slots: when the parts requested near them,
  the expansion's LOD threshold is raised just enough (CPU feedback with hysteresis, like the culling budget) so
  the nearest assemblies keep their parts. Debug views 22 (voxel clusters coloured, triangles grey) and 23 (assembly
  parts coloured, bases light grey). The wind takes a plant origin from the instance (spare `w` of the normal
  matrix), so a part sways about its tree's origin exactly as the baked foliage did.
  Measured (forest high, shared GPU, interleaved runs, minimum of 4, before → after): GPU frame 8.38 → 10.09 ms —
  voxels on / off (`Insert`) 10.09 / 9.40 ms: software raster + merge 1.47 → 1.99 ms, material resolve 0.38 →
  0.45 ms, shadow software raster 0.52 → 0.72 ms; the conifer assemblies (≈ 26 400 part instances) cost the rest,
  mostly culling (0.48 → 0.91 ms, late pass 0.06 → 0.12 ms; the non-skinning culling variant later took
  0.02–0.05 ms per culling pass back; round 6's culling records and cluster candidates cut the parts' culling to
  ≈ 0.1 ms: forest, 2560×1440, `cull` + `cull (late)` 1.19 → 0.56 ms at 100 %, 1.14 → 0.52 ms at 67 %, images
  unchanged) and the parts' finer triangles. Default view: 5.23 M triangles → 3.97 M triangles + 4.3 M voxels (shadows
  4.46 M → 2.7 M triangles + 2.8 M voxels). Geometry 43.3 → 39.6 MiB in RAM and in `.hvm`; conifer needles 7.9 MiB as
  baked meshes with voxels → 1.1 MiB as assemblies, baked in 0.5 instead of 2.0 s (whole scene 6.9 → 8.2 s:
  voxelisation). Broadleaf trees as leaf-spray assemblies would save 6.6 → 2.9 MiB for the same ≈ 0.7 ms of GPU time
  (their 144-triangle sprays are poor parts), so the forest bakes them whole. Distant canopy (albedo averaged over 16
  frames from 30 m above the forest, `--eye 0,30,0 --at 200,0,-60`, canopy 150–400 m away, vs. a full-detail
  reference at a 1/8 px LOD threshold): mean colour error +2.0/+1.3/+1.0 % → −0.2/−0.5/−1.2 % (RGB), 16×16-block error
  1.75 → 1.53/255, 40×40 1.19 → 0.88/255; the distant conifers keep their needles instead of showing bare red
  branches. Coverage tests on a leaf cloud: within 3.4 % of the full mesh along each axis at 30–2000 m (9 %
  diagonally at the finest voxels), where the triangle cut is off by up to 13 %. Temporal stability (2560×1440,
  static camera, % px > 24/255): 0.077 → 0.083 % with wind, 0.027 → 0.033 % calm.
