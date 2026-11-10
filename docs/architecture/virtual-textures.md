# Virtual textures

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Runtime virtual texture** (`rvt.rs`). The layered landscape material (height-blended texture layers driven by
  slope, altitude and noise) is rendered on demand into 128² pages of a 128k² virtual texture, cached in a 24×24-page
  atlas by a generic GPU **page cache** (request → keep/free → allocate → dirty list). It cut the valley's material
  pass from 0.53 to 0.21 ms.
* **Streaming virtual textures** (`svt/`, `svt.wgsl`). Image layers keep their 512² array layer as the always-resident
  coarse levels; the finer levels up to the images' native resolution (≤ 4096²) form a virtual texture of 128² tiles
  with 8-texel borders, streamed into three atlases (sRGB albedo; normal xy; roughness, metallic) as UE5's Streaming
  Virtual Textures do. Where the G-buffer resolve samples a layer with mesh UVs (base colour, normal, roughness,
  metallic, emission), it computes the anisotropic lod in native texels; one pixel per 4×4 block (rotating each
  frame) writes the tile of its finer trilinear level into a feedback texture that is read back asynchronously. The CPU loads requested tiles with their coarser ancestors, coarse first and within a
  per-frame budget, cutting them from mip pyramids built once at load (linear-light colour, Toksvig roughness), and
  evicts least recently needed tiles. A page-table texture maps every virtual tile to its finest resident
  ancestor-or-self plus a minimum lod, so sampling blends trilinearly between resident levels (the array is the level
  above the coarsest tile), never misses, and fades new tiles in instead of popping. Hardware anisotropic filtering
  works inside a tile because gradients are limited to its border. Sponza (24 image layers of 1024²): ~270 of 1536
  tiles resident for the atrium view in a 63 MiB pool (`RendererConfig::texture_pool_bytes`), material pass
  0.20 → 0.23 ms at 1600×900; triplanar mappings and procedural layers keep using the arrays.
* **Block-compressed texture tiles** (`svt/bc.rs`). Where the GPU offers BC formats (desktop browsers, D3D12,
  Vulkan, Metal on Macs) the tile atlases are BC7 sRGB (base colour or emission, with alpha), BC5 (normal xy; z is
  reconstructed) and BC5 (roughness, metallic): 3 instead of 8 bytes per texel, so the same 64 MiB pool holds 1024
  instead of 400 tiles (Sponza with a 16 MiB pool: 256 instead of 100, i.e. the atrium view's tiles stay resident
  instead of falling back to the 512² arrays). WebGPU can't write compressed textures from shaders, so a CPU
  encoder fits each 4×4 block — BC7 modes 6 (one RGBA line, 16 levels: no banding), 1 (two partitioned lines for
  edges between unrelated colours) and 5 (separate alpha line for coverage masks); BC5 as two least-squares BC4
  lines. As UE does for streaming virtual textures, tiles are compressed ahead of time rather than per upload:
  natively all levels are encoded on worker threads at load (Sponza: 1.6 M blocks per atlas in ~0.9 s on 12
  threads), and an upload is then a copy of blocks (~0.1 ms per frame); in the browser (no threads) a tile's blocks
  are encoded on first request — resumable, within a 2 ms per-frame budget — and spare budget encodes ahead,
  coarse levels first. Tiles, borders and 8-texel encoding chunks are block-aligned, so border blocks are
  bit-identical to the neighbouring tiles' (no seams) and the shader is unchanged apart from a third atlas.
  Quality (PSNR vs the uncompressed tiles): BC7 42–48 dB on Sponza / DamagedHelmet base colour (55 dB emission;
  Pillow's BC1 encoder gets 34–38 dB on the same images), BC5 46–54 dB normals, 45–67 dB metallic-roughness; final
  frames differ from uncompressed ones by ~51 dB with no visible block artefacts. Devices without BC (mobile:
  ETC2 / ASTC only) and `--no-texture-compression` use RGBA8 / RG8 / RG8 atlases with the same layout.
