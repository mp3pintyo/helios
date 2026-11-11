# Lumen world cache (round 8, wave 2: `lumen-world-cache`, order 214)

Design note for `crates/helios-render/src/world_cache.rs`, `shaders/lumen_world_cache_common.wgsl` (cells, read side)
and `shaders/lumen_world_cache.wgsl` (paths, resolve, debug view, reference path tracer). User-facing description:
[docs/architecture/lumen.md](../architecture/lumen.md) (*World cache*). Images: `docs/round8/world-cache/`.

## Problem

Lumen's ray hits are shaded from two surface caches. The **mesh cards** (per-texel captured material, lit per
texel) cover what their axis-aligned captures see and what lies within 100 m; everything else — surfaces no card sees,
instances beyond 100 m, card pages not yet lit — falls back to the **voxel lighting**: a 64³-per-level volume with one
mean albedo per instance × material, lit at the voxel's centre, sampled half a lighting voxel in front of the hit. It
leaks: a 1.2 m aisle wall is thinner than a 0.5–4 m lighting voxel, so the sunlit exterior shows on the interior face.
Both caches also add 15 % of the *unoccluded-looking* sky irradiance (`mix(sky × AO, radiance cache, 0.85)`, with an
AO that only sees 1.5–6 m): in a closed hall this is light that cannot be there.

## UE reference

UE 5.9's **Lumen World Cache** (experimental, hardware ray tracing only): a world-space lighting cache built at
runtime by path tracing into sparse world-space cells instead of per-mesh cards. Each frame up to three bounces are
traced from one random pixel per 8×8 screen tile; cells average up to 16 frames and are evicted after 64 unused
frames; it covers small, large, spline, skeletal and WPO-deformed meshes; when active it replaces the surface cache
for the view (cards are not captured, lit or kept). Unsupported: water, RT translucency, foliage and subsurface
transmission, far field; translucent surfaces are not cached.

## Design

**Cells** (`lumen_world_cache_common.wgsl`). A hash table of 2¹⁸ slots, linear probing over 8 slots, keyed by
- the *level*: 20 cm cells within 8 m of the camera, doubling with every doubling of the distance (2.5–5 % of it);
- the *normal bucket*: dominant axis and sign of the surface normal — the two faces of a thin wall, a floor and the
  ceiling below it are different cells, so nothing leaks between them **by construction**;
- the grid coordinates on that level; along the normal's axis cells are twice as deep and centred on the grid planes
  (where floors and walls usually lie), so a hit found through a distance field a little off the true surface still
  lands in the cell of the surface.
A 32-bit slot hash and an independent 32-bit checksum identify a key. A resolved cell is one `rgba32uint` texel
(checksum, irradiance f16×3, emission f16, albedo rgb8 + frames averaged + "albedo seen on screen") in a 512×512
texture: readers bind one sampled texture (binding 150), which keeps Lumen Lite's trace stage within the 8 storage
buffers WebGPU guarantees. Readers also look in the second bucket for normals between two axes and on the other side
of a level boundary.

**Paths** (`trace_paths`, one thread per 8×8 tile). A random pixel of the tile is the first vertex: position, normal,
albedo and emission from the G-buffer. A cosine ray (near-field mesh SDFs, then the global field) finds the second
vertex; a second cosine ray from it is shaded by the caches (`wc_cached_radiance`: the world cache itself, so light
keeps bouncing from frame to frame, the cards in front of it unless it replaces them, the voxel lighting where nothing
else has the surface; sky and far field for misses). Every vertex receives direct light — the sun through the
virtual shadow map where its page is resident (distance-field shadows otherwise), coloured by glass
(`glass_transmission`), and **one** local light picked by weighted reservoir sampling among those in reach in
proportion to its unshadowed irradiance, shadowed through the global field and scaled by sum / weight (an unbiased
estimate of all of them: a shadow ray per light in reach — a dozen candles around a pew — cost more than the rest of
the path). Back along the path each vertex's irradiance (direct + π × the radiance arriving along its ray, clamped to
4 exposed units per ray like the screen probes' 1.5) is added to its cell: fixed point, scaled by the current
exposure, 256 steps per exposed unit, stochastic rounding, `atomicAdd`. A third shaded vertex cost 0.2 ms more in the
cathedral and adds what the cells already carry over from frame to frame, so paths shade two vertices.

Behind the cards (*Fallback*), a hit whose cards match with confidence ≥ 0.5 ends the path with the cards' radiance —
that is what the screen probes read there anyway, and the cells that matter are the ones the cards miss.

**Material at a hit — the hard part.** There are no hit shaders. A cell created from a visible pixel knows that
pixel's albedo exactly, so the albedo is a property of the cell, learnt from the screen and kept:
1. the albedo the hit's cell learnt from the screen (G-buffer: textures, material graphs, skinned and displaced
   surfaces, wind-blown foliage, emission) — exact, at the cell's resolution, for everything ever seen;
2. else the mesh cards' **captured texel** at the hit (`wc_card_albedo`: the hit is matched against the card's
   captured depth and normal as `card_sample` does, then the albedo atlas texel is read; the cards' material capture
   works whether or not their lighting is updated);
3. else the global field's mean albedo of the instance's (surface, material) pair.
Estimates feed the cell with weight 1, screen samples with weight 16; the first screen sample replaces any estimate
at once and marks the cell, after which estimates are ignored. Glass and emissive voxels end the path with the voxel
lighting (UE does not cache translucent surfaces either).

**Resolve** (`resolve`, one thread per slot): this frame's sums become a mean blended into the cell over up to 16
frames (a change seen by ≥ 8 samples that moves the luminance by more than 60 % shortens the history to 4 frames, so
a switched light or the sun moving behind a wall is followed within a few frames); cells no path visited for 64
frames are evicted (the slot is cleared; an entry orphaned behind it is found by nobody and evicted in turn).

**Readers.** `trace_world_radiance` (screen probes, reflections, Lumen Lite's probes) and `update_probes` (the radiance
cache) blend the cell over the voxel lighting by its confidence (frames / 4). `RenderSettings::world_cache`:
- `Off` — round 7: cards, voxel lighting where none matches;
- `Fallback` (`--world-cache on`) — cards first; the world cache in place of the voxel lighting;
- `Replace` (`--world-cache replace`, UE's behaviour) — the world cache in place of cards and voxels; the card
  lighting pass is skipped (the cards' capture still runs: its albedo feeds the estimates).

**`Replace` is the default**: it is the mode that agrees with the reference indoors (below), and it costs only
+0.04–0.15 ms because the gather loses its card lookups.
Key `5` cycles them, editor *Lighting → Lumen world cache*, browser `?world-cache=on|replace`. Debug view 27 shows the
radiance each pixel's cell holds (purple-tinted voxel lighting where none). The scene changing clears all cells.

**Reference** (`reference`, `HELIOS_WORLD_CACHE_REFERENCE=1`, native only): per half-resolution pixel and frame a
path of up to 8 bounces (Russian roulette after 2) through the same distance fields, material estimates (cards' texel,
field mean) and lights, without any cache; accumulated while the camera rests and written into `gi_raw` in place of
the final gather, so view 7 and the lit view show it through the same GI filter and tone mapping. It is unbiased for
the scene *as Lumen represents it* — the comparisons below measure the caches, not the distance fields.

## Alternatives considered

- **Cards' radiance only at hits (no learnt albedo).** Cheaper, but leaves exactly the surfaces the cards miss
  (the point of the cache) with the field's per-instance mean.
- **Albedo per cell from the hits alone** (field mean): the cathedral's floor would bounce the mean of black and white
  marble; skinned and displaced surfaces never appear in the fields at all.
- **Cells storing exitant radiance** (SHaRC's default): samples from the screen (exact albedo) and from hits
  (estimated) would be averaged into one colour; storing irradiance and albedo separately lets a screen sample correct
  every earlier estimate at once.
- **Ping-pong cell textures** for the resolve: the resolved state lives in the record buffer instead (the texture is
  write-only from the resolve, sampled by everyone else).
- **Query-driven cells** (gather misses insert cells that a later pass lights from a stored point): needs read-write
  bindings in every Lumen trace and a position per cell; the paths' second vertices already land where the gather's
  rays land.
- **Three shaded vertices** (UE's "up to 3 bounces"): +0.2 ms, no visible difference — the cells carry the further
  bounces from frame to frame.

## Measurements

RTX 3090 shared with four other agents, 1600×900. **Accuracy** (indirect light, view 7, fixed exposure, mean linear
luminance; reference = 400 paths per half-res pixel; props paused):

| View | Off (round 7) | Behind cards | Replaces cards | Reference |
|---|---|---|---|---|
| cathedral nave (EV 0) | 0.0260 (+183 %) | 0.0129 (+40 %) | 0.0106 (+15 %) | 0.0092 |
| cathedral aisle (EV 0) | 0.0234 (+457 %) | 0.0078 (+86 %) | 0.0047 (+12 %) | 0.0042 |
| valley (EV −2) | 0.0667 (0 %) | 0.0636 (−4 %) | 0.0635 (−4 %) | 0.0664 |
| forest, medium (EV −2) | 0.0357 (−3 %) | 0.0355 (−4 %) | 0.0356 (−3 %) | 0.0368 |
| gallery (EV −1) | 0.1851 (−6 %) | 0.1823 (−7 %) | 0.1820 (−7 %) | 0.1965 |

Outdoors all three agree with the reference within 3–7 %. Indoors round 7's caches are 3–6 times too bright — the
interior face of the sunlit aisle wall glows (the voxel leak) and every surface gets the 15 % of open sky — while the
world cache replacing them is within 12–15 % of the reference; behind the cards the cards' own sky term still carries
40–86 % too much. (aisle-*.jpg, nave-*.jpg)

**Cost** (minimum of 3 interleaved runs per mode, per-pass minima, ms; "GI" = cards + surface/radiance cache + world
cache + final gather + reflections + GI temporal):

| Scene | world cache pass (replace) | final gather off → replace | GI off | GI behind cards | GI replaces cards | live cells |
|---|---|---|---|---|---|---|
| cathedral | 0.31 | 0.90 → 0.75 | 1.55 | 1.95 | 1.62 | 46 k / 58 k |
| valley | 0.18 | 0.73 → 0.68 | 1.11 | 1.34 | 1.20 | 15 k |
| forest (medium) | 0.26 | 0.89 → 0.76 | 1.26 | 1.66 | 1.35 | 51 k |
| gallery | 0.21 | 0.80 → 0.79 | 1.33 | 1.58 | 1.48 | 14 k |

The pass is latency-bound (one thread per 8×8 tile: ~14–22 k threads); each shaded vertex costs ~0.1–0.2 ms
(one: 0.12 ms, two: 0.31 ms, three: 0.52 ms in the cathedral). Replacing the cards removes their lighting pass
(0.02–0.05 ms) and the gather's card lookups (up to 0.15 ms), so the net cost is +0.04–0.15 ms. Memory: 20 MiB
(2¹⁸ slots × 64-byte records + a 512² `rgba32uint` cell texture), allocated when first enabled; at most 22 % of the
slots live (`HELIOS_WORLD_CACHE_STATS=1`).

**Temporal stability** (static camera, props paused, lit frames, 24 frames; % of pixels changing by > 4/255 · > 24/255):

| Scene | Off | Behind cards | Replaces cards |
|---|---|---|---|
| cathedral | 4.72 · 0.0048 | 4.98 · 0.0052 | 4.63 · 0.0082 |
| valley | 3.91 · 0.0137 | 4.82 · 0.0228 | 4.79 · 0.0228 |
| forest (medium) | 9.23 · 0.1108 | 9.16 · 0.1130 | 9.18 · 0.1185 |

(The 4/255 level is dominated by TSR and the tone mapper's dither in every mode.) **Moving sun** (`--animate-sun`,
cathedral, indirect light, 60 frames): the mean follows smoothly in every mode (largest frame-to-frame step 0.16 %
off, 0.21–0.22 % with the cache; 60-frame drift −4.7 / −5.8 / −5.1 %), per-pixel changes 1.5 → 2.7–2.9 / 255 (the
cells' noise). **Moving camera** (walking down the nave, 40 frames): mean absolute change per frame 5.04 → 5.62
(behind cards) / 5.87 / 255 (replace), largest step of the mean 1.33 → 1.60 / 1.74 % — cells fill in as surfaces come
into view; until then the voxel lighting stands in.

**Browser** (headless Chromium, WebGPU/D3D11): the cathedral with `?world-cache=on` / `replace`, with Lumen and Lumen
Lite: Tint compiles every stage, no console errors or validation warnings; key `5` cycles the modes (browser-on.jpg).

## Remaining limits

- Rays still only hit what the distance fields hold: skinned meshes, particles and volumes get cells on screen (exact
  albedo) but no ray ever hits them (UE's hardware ray tracing does); far field beyond the clipmap uses the radiance
  cache as before.
- Cells are 2.5–5 % of their distance across (20 cm within 8 m): the bounce of detail finer than that is averaged
  (cards keep ~10 cm texels, so *behind the cards* keeps their detail where they match).
- Cells only exist where paths went; a gather ray hitting a surface no path reached in 64 frames reads the voxel
  lighting (and its sky term) instead — after a cut to a new view, the first frames.
- Local lights are sampled one per vertex with global-field shadows (no near-field contact shadows, as the cards).
- The fixed-point accumulation is relative to the current exposure: a sudden exposure change within a frame is not
  possible (exposure is read once per frame), but extreme HDR (> 4096 exposed units of irradiance) clamps.
- The reference shares the cache's direct-light and material estimates, so it validates the caching, not those.
