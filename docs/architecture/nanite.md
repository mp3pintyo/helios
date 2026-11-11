# Nanite core: streaming, rasterisation, culling

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Streaming** (`streaming.rs`, `page_source/`). Cluster and page records are always resident; vertex/triangle data
  lives in a fixed GPU pool. Culling stamps "a resident parent wants to refine into this page" into a feedback buffer,
  which is read back asynchronously; the CPU requests needed pages coarse-first together with their missing
  dependency pages, uploads arrived pages within a per-frame budget once their dependencies are resident (the
  invariant that keeps the cut crack-free), and evicts least-recently-needed leaf pages when the pool is full. A
  missing page just means its parents stay drawn. Page payloads come from memory or from an asynchronous
  `PageSource` (absent → requested → arrived → resident): native files are read by I/O threads with positioned
  reads, the browser uses `fetch` with HTTP `Range`; adjacent pages are coalesced into one read, in-flight reads are
  bounded per source, failed reads are retried later. Page payloads stay compressed in the pool (`geometry.wgsl`
  decodes vertices on the fly in every consumer), so resident memory and streamed bytes are both 2.1–2.5× smaller
  (valley 118 → 48 MiB, cathedral 167 → 80 MiB, forest 85 → 40 MiB).
* **Software rasteriser** (`sw_raster.wgsl`). Clusters whose triangles are smaller than ~2.5 px are rasterised in
  compute: a workgroup per cluster, exact sub-pixel edge functions with the top-left rule. Without 64-bit atomics the
  depth and payload are written in two passes (32-bit `atomicMax` of depth, then the ids of pixels that won); a merge
  pass combines hardware and software results. Images with it on/off differ by at most 5/255.
* **Alpha-masked materials** (`alpha_mask.rs`, `alpha_mask.wgsl`; Nanite's "programmable raster"). A material with an
  `alpha_cutoff` is cut out where its base colour image's alpha is below it — in all four rasterisers (visibility
  buffer hardware and software, shadow pages hardware and software), so leaves and grilles also cast cut-out shadows.
  Culling bins alpha-masked hardware clusters into their own list, drawn by a second pipeline whose fragment shader
  samples and discards, so opaque clusters keep a trivial fragment shader and early depth testing. The compute
  rasterisers get an alpha-testing specialisation dispatched over the same clusters (each skips the other's, a
  workgroup-uniform branch), so the opaque kernels keep their registers and occupancy; it tests each covered pixel
  with the triangle's UV gradients. Scenes without masked materials skip the extra draws and dispatches. The masks live in a separate R8 array with **coverage-preserving mips** (each mip is rescaled to keep
  the fraction of texels passing the test, so distant foliage does not dissolve). Occlusion culling needs no special
  case: the HZB is built from the alpha-tested depth. In Lumen, masked surfaces are partially transparent: each
  (mesh, material) pair gets a **coverage** — the area-weighted fraction of its surface that passes the alpha test,
  sampled once on the CPU from the mesh UVs and the image (Sponza's foliage 41–53 %, the cathedral's choir screens
  26 %) — and near-field traces hit the instance with that probability (stochastic transparency, as UE5 dithers the
  distance-field traces of two-sided foliage; 64 rays per probe and the temporal filter average it), while
  distance-field soft shadows (local lights; the sun on mesh cards) let `1 − coverage` of the light through. Masked
  instances get mesh cards, so hits on leaves read their lighting, but stay out of the global distance field, whose
  ≥ 0.25 m voxels would merge leaves and chain links into solid blobs.
* **Two-pass occlusion culling** (as in Nanite) with persistent visibility bits; hierarchical page culling skips whole
  pages whose coarsest parent is already detailed enough. Cluster culling takes a 64-lane workgroup per surviving
  page, except for pages of at most 16 clusters (assembly parts, grass, small props), which list their clusters one by
  one as *cluster candidates* after the page work items, 64 to a workgroup (round 6: the forest's needle sprays have
  17 clusters over 6 pages, so their pages kept ~5 % of a workgroup's lanes busy); beyond the candidates' capacity a
  page falls back to a work item. The sixteen shadow clipmap levels, culled one after another, share one work buffer.
  Round 8 ([`nanite-culling.md`](../round8/nanite-culling.md); UE 5.9's culling rework): instances with at most 16
  pages likewise list their pages as *page candidates*, 64 to a page workgroup from any instances, instead of a
  64-lane batch each (not the local lights' view, whose lanes of one instance share its loads, nor assembly culling
  records; forest main view 5 432 → 1 070 batches, page pass 0.028 → 0.020 ms); a sun clipmap level without any
  dirty page leaves the instance pass before a thread loads its instance (most levels in calm scenes); and culling
  is specialised on deformation (`DEFORMING` in `cull.wgsl`): frames in which nothing deforms — no wind blowing on an
  instance with a wind response, no displaced or skinned instance (`GpuScene::deforms`) — use pipelines without the
  wind, displacement and skin bound code (UE's simple/complex split, per frame rather than per instance). Automatic,
  no settings; results identical. `cull` + `cull (late)` at 1600×900: valley 0.210 → 0.195 ms, cathedral
  0.277 → 0.263 ms, forest 0.472 → 0.469 ms. `HELIOS_CULL_STAGES=1` times every stage of every view as its own pass
  (`cull instances [main]`, `cull pages [shadow]` …) for profiling. Testing several DAG levels at once (a per-mesh
  level table skipping the finer levels of distant instances in the instance pass) was built, proved conservative and
  measured: it cost the instance pass what it saved the page pass, so it was removed.
