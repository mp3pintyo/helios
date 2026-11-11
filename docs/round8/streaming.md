# Round 8 — streaming: entropy-coded page payloads (`.hvm` version 7)

Backlog 223, `streaming-entropy-coding`. Code: `crates/helios-geometry/src/page_codec.rs` (the codec),
`dag.rs` (`.hvm` version 7: `to_coded_bytes`, `page_stored`, decoding in `from_bytes` / `set_tail_payload`),
`crates/helios-render/src/page_decode.rs` (decode threads, bake-worker tasks, render-thread fallback),
`streaming.rs` (the `Decoding` page state), `page_source/file.rs` (emulated network link),
`crates/helios-demo/src/mesh_store.rs` / `main.rs` (`--bake-hvm` writes version 7, `--hvm-raw`,
`--io-bandwidth-mbps`).

## Problem

Streamed page payloads are stored exactly as the geometry pool holds them: the quantised, bit-packed cluster format of
`encoding.rs` (Nanite's GPU format: per-cluster minimum + just enough bits per field, fixed-width triangle indices).
That is already 2.1–2.5× smaller than float vertices, but it is a *fixed-width* code: every vertex of a cluster spends
the same bits on each field however predictable it is, so generic compressors find almost nothing to remove
(measured per page on the demo scenes: LZ4 0.3–6 %, DEFLATE −9 4–16 %, zstd −19 5–20 %). The browser fetches pages
over HTTP, where bytes are what costs time.

## UE reference

Nanite cooks its streaming pages into a *disk format* that differs from the GPU format (byte-aligned, delta-coded
streams that general-purpose compressors handle well), compresses them with Oodle (Kraken) in the pak/IoStore, and
transcodes them into the GPU format in a compute pass when they arrive (`NaniteTranscode.usf`). The point is the same
as here: the representation on disk is chosen for entropy, the one in the pool for random access by the rasterisers.

## Design

**One self-contained blob per page.** `page_codec::encode_page(words, cluster records)` returns a mode byte and either
the words as they are (mode 0) or the coded page (mode 1). The coded page carries its own cluster structure (counts,
kinds, quantisation parameters), so `decode_page(bytes)` needs nothing else — it runs on any thread or Web Worker. The
decoder reads the field values back and packs them with `encode_cluster`, the very function the bake uses, so the
result is bit-identical; the encoder decodes every page it codes and stores it raw if that does not reproduce it
exactly or does not shrink it. Nothing about the GPU format or the shaders changes.

**The model** (values in the order the decoder needs them):

* *Triangles.* Vertices are in first-use order (the builder's), so a corner is either "the next new vertex" (0) or a
  distance back from it. Triangles mostly neighbour recent ones: a ring of the last 32 triangles' directed edges; a
  triangle that runs one of them backwards is coded as the slot (context: whether the previous triangle hit) plus its
  rotation (the GPU order of its corners must be kept) plus its third corner; otherwise its three corners.
* *Repeats.* A quarter of a page's vertices repeat an earlier vertex of the page (cluster and group borders: the same
  grid point with the same attributes). They are a repeat kind (all attributes / position only / none, by the previous
  vertex's kind) plus a reference: a distance back, or after a repeat the step from the previous one's source.
* *Positions* (grid integers). The vertex's first triangle gives the predictor: with two earlier corners a, b and a
  triangle across a–b whose third corner c is earlier, the parallelogram a + b − c; where several triangles around the
  vertex offer one, their mean (Draco's multi-parallelogram); else the midpoint, the one earlier corner, the previous
  vertex, or (first vertex of a cluster) the first vertex of the page's previous cluster. Residuals are coded per axis
  and predictor; y and z pick their statistics by the size of the residuals already coded for the vertex.
* *Normals* (octahedral grid). Predicted from the decoded geometry — the area-weighted sum of the face normals around
  the vertex, mapped to the octahedral grid — or from the earlier corners' normals; the encoder picks per cluster
  (foliage cards with bent normals prefer the neighbours, hard-surface and terrain the geometry).
* *UVs* (grid integers). The position predictors in texture space, or *mapped*: the vertex's position expressed in the
  plane of the parallelogram's triangle (a, b, c) as `a + s(b − a) + t(c − a)` and applied to their UVs, exact for
  planar and box mapping (the cathedral's walls). Per-cluster choice again (the valley's heightfield prefers the
  parallelogram). The second UV set, vertex colours, plant data and skin weights are predicted from one neighbour.
* *Voxel clusters*: the 8 header words raw, brick and voxel words byte by byte as the difference to the previous
  brick's or voxel's.

**Entropy coding.** A residual becomes a token — exact below 8, else its bit length and the next bit (66 tokens) —
plus raw extra bits in a separate bit stream; small alphabets (edge slots, rotations, repeat kinds, modes) are coded as
symbols of their own. Tokens are coded with rANS (32-bit state, byte-wise renormalisation, 12-bit probabilities),
encoded backwards so the decoder reads forwards with one table lookup per symbol. Every context has a static
frequency table per page, stored in the page: frequencies are quantised to 31 levels (steps of 2^⅓ relative to the
most frequent token) and the levels are delta-coded from token to token; a context whose own table would cost more
than it saves shares its parent's (rarely used predictors fall back to the parallelogram's, the z-by-size contexts to
y's, …), decided by the encoder per page from exact costs. Tables are ~2 % of the coded bytes.

**Determinism across platforms.** The predictions use integers, `f64` conversions and the IEEE basic operations
(`+ − × ÷`, `abs`, `round`) only — no fused or transcendental functions — so native and wasm decoders compute the same
predictions bit for bit; the encoder uses `log2` only to choose table levels, which are then stored.

**`.hvm` version 7.** The version-5 header, followed by `pages + 1` byte offsets of the stored pages, then the stored
pages. `VirtualMesh::to_coded_bytes()` writes it (coding pages on all cores), `to_bytes()` still writes version 5 (the
bake workers' transfer format, and `--hvm-raw`); `from_bytes`, `from_header_bytes`, `header_len`, `page_byte_range`
and `tail_byte_range` read both (versions 4 and 5 as before), `set_tail_payload` decodes the root pages, and
`VirtualMesh::is_coded` tells the streamer to decode what it reads. (The wind/bones agent may bump the raw version to
6 in parallel; version 7 is "the current raw layout + coded pages", so at merge it becomes the next number after
theirs, or stays 7.)

**Decoding while streaming.** A completed read of coded pages becomes one decode job; its pages are `Decoding`
(a new state between `Requested` and `Arrived`) until the payload words are back, then they are staged and uploaded
as before (the staging budget always counts decoded sizes; source budgets count stored bytes). Natively up to 4
decode threads (`helios-decode-N`, cores − 2) take the jobs. In the browser the jobs go to the bake workers
(`lumen_offload.rs` dispatches `HPDT` tasks to `page_decode::run_decode_task`): the read's stored pages cross as one
message, the decoded words come back; a worker that fails, or a page without workers (`?bake-workers=0`), is decoded
on the render thread for at most 2 ms per frame. A page that does not decode is a failed read (requested again 60
frames later). Stale decodes after a scene change are dropped by ticket, like reads.

## Alternatives considered

* **Generic compressors on the GPU format** (measured per page, all demo scenes): LZ4 0.3–6 %, DEFLATE −9 4–16 %,
  zstd −19 5–20 % smaller, against 52–62 % for the model. `ruzstd` decodes at 90–470 MB/s, lz4 at 3–7 GB/s — fast,
  but there is too little redundancy left in fixed-width fields for any of them.
* **A byte-aligned disk format + a generic codec** (UE's approach) would need its own transform anyway; the model
  codes the transformed values directly with contexts instead, without a C dependency for wasm.
* **Adaptive models** (no tables) code a little tighter but decode slower, and rANS wants static tables; per-page
  static tables with the parent sharing above cost ~2 %.
* **Parallelogram contexts by triangle size** did not pay for their tables; the y/z-by-residual contexts did.
* **Reordering triangles** in the bake for better connectivity coding would change the GPU pages (and the builder);
  left for later — the codec codes the builder's order as it is.
* **GPU transcoding** (UE) would decode in the GPU's spare time instead of a CPU thread; the CPU decoders were simpler
  to make bit-exact and run in parallel with the render thread here.

## Measurements

RTX 3090 machine shared with other agents' builds (timings vary by ±30 %; minima of repeated runs). Page payloads of
every page of every mesh of the scene libraries (`--bake-hvm`):

| Scene, detail | GPU format | Coded (v7) | LZ4 | DEFLATE −9 | zstd −19 | `.hvm` files v5 → v7 |
|---|---|---|---|---|---|---|
| valley, medium | 12.5 MiB | 5.74 MiB (45.9 %) | 99.7 % | 95.3 % | 94.5 % | 14.4 → 7.6 MiB |
| valley, high | 49.1 MiB | 22.0 MiB (44.9 %) | 99.7 % | 96.2 % | 95.5 % | 53.9 → 26.9 MiB |
| forest, medium | 13.1 MiB | 6.28 MiB (47.9 %) | 94.5 % | 84.0 % | 80.3 % | 15.0 → 8.2 MiB |
| forest, high | 36.5 MiB | 15.5 MiB (42.6 %) | 96.7 % | 88.5 % | 85.7 % | 40.0 → 19.0 MiB |
| cathedral, medium | 20.7 MiB | 7.99 MiB (38.6 %) | 98.8 % | 90.1 % | 88.2 % | 23.2 → 10.5 MiB |
| cathedral, high | 80.3 MiB | 30.4 MiB (37.8 %) | 98.9 % | 91.3 % | 89.5 % | 86.8 → 36.9 MiB |
| gallery, medium | 3.08 MiB | 1.31 MiB (42.6 %) | 93.9 % | 88.1 % | 86.7 % | 3.7 → 2.0 MiB |
| gallery, high | 9.51 MiB | 4.27 MiB (44.9 %) | 97.3 % | 94.4 % | 93.9 % | 10.8 → 5.5 MiB |

Every v7 file decodes to exactly the v5 file's mesh (all eight libraries compared word for word). Where the coded
bits go (cathedral / valley, medium): triangles 26 / 29 %, positions 36 / 38 %, normals 14 / 20 %, UVs 16 / 6 %,
repeats 7 / 6 %, voxels (forest) 8 %, tables ~2 %. Headers — cluster and page records, dependencies and the distance
field — are not coded and are now 18–35 % of a file.

**Speed.** Decoding: 50–65 MB/s of GPU pages per core natively; the same code compiled to wasm decodes 41–53 MB/s in
Node 24 (V8), 50–51 MB/s in Chromium 148 and 48 MB/s in Firefox 155 (a decode-only page over the cathedral,
valley and forest medium pages, best of 5). Encoding: 10–20 MB/s per core (the bake codes pages on every core; cathedral high adds ~2 s
to a 33–44 s bake).

**Streaming latency** (`--stream-from`, default view, 1600×900; "view" = every page the culling asks for is resident,
"idle" = the prefetch is done too):

| | valley high v5 → v7 | cathedral high v5 → v7 | forest high v5 → v7 |
|---|---|---|---|
| local SSD: view | 0.52 → 0.79 s | 0.86 → 0.91 s | |
| 100 Mbit/s, 20 ms: view | 2.31 → 1.65 s | 2.48 → 1.76 s | 2.09 → 1.39 s |
| 100 Mbit/s, 20 ms: idle | 4.17 → 2.00 s | 6.78 → 2.64 s | 3.23 → 1.40 s |
| bytes read | 49.1 → 22.0 MiB | 80.1 → 30.3 MiB | 36.4 → 15.5 MiB |

From a fast local disk the decode threads are the bottleneck (49 MiB at ~50 MB/s on ≤ 4 threads), so the view
completes a quarter of a second later; over a link the smaller reads win. The first image is unchanged: headers and
root pages load as before (root pages are decoded with the header: valley high 4 ms for 4.9 MiB of header and roots).

**Browser** (headless Chromium, WebGPU, `python serve.py`, `?stream=`, 1600×900; the decode jobs run on the bake
workers): every library loads and streams completely, no console errors (images: `streaming/cathedral-high-v7-chromium.jpg`,
`streaming/valley-high-v7-chromium-100mbit.jpg`). Locally (no throttling) first image / streaming idle: valley
medium 8.2 / 9.2 s (v5) → 7.9 / 8.4 s (v7), cathedral high 7.1 / 7.6 → 6.5 / 7.1 s — page load and pipeline
compilation dominate, decoding on the workers does not hold anything up. Throttled to 100 Mbit/s + 20 ms with
Chromium's network emulation (which also throttles the 6.3 MiB wasm, the headers and the textures), valley high:
first image 14.5 → 9.0 s, streaming idle 18.2 → 10.0 s. (The first browser run showed `ERR_CONNECTION_REFUSED`
on some of the ~50 parallel range requests: `serve.py`'s listen backlog was Python's default 5; it is now 128.)

**Memory.** The pool and the staged pages hold the GPU format as before; reads keep their (smaller) stored bytes
until decoded; `page_stored` adds 4 bytes per page to the resident header.

## Remaining limits

* Decoding is a CPU job at ~50 MB/s per core: an Oodle-class decoder (or UE's GPU transcode) is an order of magnitude
  faster. A disk faster than the decode threads makes streaming slightly later than with raw pages.
* Headers (cluster records, distance fields) are not compressed and dominate small libraries.
* The model codes the builder's triangle order as it is; reordering triangles for connectivity coding (and vertex
  cache locality) in the bake would shrink the triangle share (about a quarter of the bits) further.
* Imported glTF models are still baked at load, not streamed (unchanged).
