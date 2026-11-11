// Procedural material texture synthesis (Substance-style), run once on the GPU.
//
// Every layer of the material texture arrays is generated from a pattern recipe. All noise is
// periodic on the unit square, so every texture tiles seamlessly.
//
//   generate : pattern -> albedo (sRGB) and a temporary (height, roughness) image
//   normals  : Sobel normals from the height image -> data = (normal.xy, roughness, height)
//   downsample_albedo / downsample_data : mip chain (albedo averaged in linear space)
//
// Layers made from images skip the first two steps:
//   pack_images : area-resample base colour / normal / metallic-roughness images to the layer
//                 size -> albedo (sRGB, alpha) and data = (normal.xy, roughness, metallic);
//                 with PACK_OCCLUSION the occlusion image's R replaces the alpha

struct Recipe {
    pattern: u32,
    seed: u32,
    scale: f32,
    layer: u32,
    color_a: vec4<f32>,   // rgb, roughness
    color_b: vec4<f32>,   // rgb, roughness
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> recipe: Recipe;
@group(0) @binding(1) var out_albedo: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var out_height: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var in_height: texture_2d<f32>;
@group(0) @binding(4) var out_data: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var in_mip: texture_2d<f32>;
@group(0) @binding(6) var in_base_color: texture_2d<f32>;          // sRGB view: samples are linear
@group(0) @binding(7) var in_normal: texture_2d<f32>;
@group(0) @binding(8) var in_metallic_roughness: texture_2d<f32>;
@group(0) @binding(9) var image_sampler: sampler;                   // bilinear, repeat
@group(0) @binding(10) var in_occlusion: texture_2d<f32>;           // R: ambient occlusion

override PACK_OCCLUSION: bool = false;
// `generate` is specialised per pattern: one pipeline holds one pattern's code (60-100 ms to
// compile in the browser instead of 1.5 s for all of them), and a scene compiles the ones it uses.
override PATTERN: u32 = 0u;

const PATTERN_ASHLAR: u32 = 0u;
const PATTERN_FLAGSTONE: u32 = 1u;
const PATTERN_MARBLE_TILES: u32 = 2u;
const PATTERN_WOOD: u32 = 3u;
const PATTERN_SLATE: u32 = 4u;
const PATTERN_STAINED_GLASS: u32 = 5u;
const PATTERN_FABRIC: u32 = 6u;
const PATTERN_BARK: u32 = 7u;
const PATTERN_LEAF: u32 = 8u;
const PATTERN_MOSS: u32 = 9u;
const PATTERN_FOREST_FLOOR: u32 = 10u;
const PATTERN_GRASS: u32 = 11u;
const PATTERN_DIRT: u32 = 12u;
const PATTERN_ROCK: u32 = 13u;
const PATTERN_SNOW: u32 = 14u;
const PATTERN_HAMMERED: u32 = 15u;
const PATTERN_PLASTER: u32 = 16u;
const PATTERN_CARVED_STONE: u32 = 17u;
const PATTERN_COBBLESTONE: u32 = 18u;
const PATTERN_CLIFF: u32 = 19u;

// ---- Periodic noise -----------------------------------------------------------------------------

fn hash_u(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

fn hash2(c: vec2<i32>, seed: u32) -> u32 {
    return hash_u((u32(c.x) * 73856093u) ^ hash_u((u32(c.y) * 19349663u) ^ seed));
}

fn rand2(c: vec2<i32>, seed: u32) -> f32 {
    return f32(hash2(c, seed) & 0xffffffu) / 16777216.0;
}

fn wrap(c: vec2<i32>, period: vec2<i32>) -> vec2<i32> {
    return ((c % period) + period) % period;
}

fn gradient(c: vec2<i32>, seed: u32) -> vec2<f32> {
    let a = rand2(c, seed) * 6.2831853;
    return vec2<f32>(cos(a), sin(a));
}

// Gradient noise in [-1, 1] with integer period `period` (in lattice cells).
fn pnoise(p: vec2<f32>, period: vec2<f32>, seed: u32) -> f32 {
    let per = vec2<i32>(max(period, vec2<f32>(1.0)));
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let g00 = dot(gradient(wrap(i, per), seed), f);
    let g10 = dot(gradient(wrap(i + vec2<i32>(1, 0), per), seed), f - vec2<f32>(1.0, 0.0));
    let g01 = dot(gradient(wrap(i + vec2<i32>(0, 1), per), seed), f - vec2<f32>(0.0, 1.0));
    let g11 = dot(gradient(wrap(i + vec2<i32>(1, 1), per), seed), f - vec2<f32>(1.0, 1.0));
    return mix(mix(g00, g10, u.x), mix(g01, g11, u.x), u.y) * 1.414;
}

// fBm on the unit square: `base` = integer frequency of the first octave.
fn pfbm(uv: vec2<f32>, base: vec2<f32>, octaves: u32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = base;
    var norm = 0.0;
    for (var o = 0u; o < octaves; o++) {
        sum += pnoise(uv * freq, freq, seed + o * 1013u) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum / norm;
}

fn pridged(uv: vec2<f32>, base: vec2<f32>, octaves: u32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = base;
    var norm = 0.0;
    for (var o = 0u; o < octaves; o++) {
        let n = 1.0 - abs(pnoise(uv * freq, freq, seed + o * 733u));
        sum += n * n * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum / norm;
}

struct Voronoi {
    f1: f32,
    f2: f32,
    id: u32,
    center: vec2<f32>,
}

// Periodic Voronoi with `cells` cells per side.
fn pvoronoi(uv: vec2<f32>, cells: vec2<f32>, jitter: f32, seed: u32) -> Voronoi {
    let p = uv * cells;
    let per = vec2<i32>(cells);
    let i = vec2<i32>(floor(p));
    var v = Voronoi(1e9, 1e9, 0u, vec2<f32>(0.0));
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = i + vec2<i32>(x, y);
            let w = wrap(c, per);
            let h = hash2(w, seed);
            let offset = vec2<f32>(f32(h & 0xffffu), f32(h >> 16u)) / 65536.0;
            let center = vec2<f32>(c) + 0.5 + (offset - 0.5) * jitter;
            let d = distance(p, center);
            if d < v.f1 {
                v.f2 = v.f1;
                v.f1 = d;
                v.id = h;
                v.center = center;
            } else if d < v.f2 {
                v.f2 = d;
            }
        }
    }
    return v;
}

fn idf(id: u32, salt: u32) -> f32 {
    return f32(hash_u(id ^ salt) & 0xffffu) / 65536.0;
}

// ---- Patterns -------------------------------------------------------------------------------------

struct Texel {
    albedo: vec3<f32>,
    height: f32,
    roughness: f32,
}

// Rectangular bond with mortar: returns (distance to block edge in block units, block id).
fn blocks(uv: vec2<f32>, counts: vec2<f32>, stagger: f32, seed: u32) -> vec4<f32> {
    let row = floor(uv.y * counts.y);
    let shift = select(0.0, stagger, (u32(row) & 1u) == 1u) + rand2(vec2<i32>(i32(row), 0), seed) * 0.2;
    let x = uv.x * counts.x + shift;
    let column = floor(x);
    let local = vec2<f32>(fract(x), fract(uv.y * counts.y));
    let edge = min(min(local.x, 1.0 - local.x) * counts.y / counts.x, min(local.y, 1.0 - local.y));
    let id = f32(hash2(wrap(vec2<i32>(i32(column), i32(row)), vec2<i32>(i32(counts.x), i32(counts.y))), seed) & 0xffffu) / 65536.0;
    return vec4<f32>(edge, id, column, row);
}

fn ashlar(uv: vec2<f32>, carved: bool) -> Texel {
    let counts = vec2<f32>(3.0, 6.0) * recipe.scale;
    let b = blocks(uv, counts, 0.5, recipe.seed);
    let mortar = 0.035 * recipe.params.x + 0.01;
    let chips = pfbm(uv, counts * 4.0, 5u, recipe.seed + 7u);
    let stain = pfbm(uv, vec2<f32>(3.0), 5u, recipe.seed + 3u);
    let bevel = smoothstep(mortar, mortar + 0.07, b.x + chips * 0.02);
    var t: Texel;
    var stone = mix(recipe.color_a.rgb, recipe.color_b.rgb, saturate(b.y * 0.8 + stain * 0.6 + 0.2));
    stone *= 0.85 + 0.25 * chips;
    if carved {
        // Engraved quatrefoil-like relief inside each block.
        let local = fract(uv * counts) - 0.5;
        let r = length(local * vec2<f32>(counts.y / counts.x, 1.0));
        let petals = 0.5 + 0.5 * cos(atan2(local.y, local.x) * 4.0);
        let relief = smoothstep(0.32, 0.28, r - 0.06 * petals) * smoothstep(0.12, 0.16, r);
        t.height = bevel * (0.75 + 0.15 * chips) + relief * 0.18;
    } else {
        t.height = bevel * (0.8 + 0.2 * chips);
    }
    let mortar_color = recipe.color_b.rgb * 0.55;
    t.albedo = mix(mortar_color, stone, bevel);
    t.roughness = mix(0.95, recipe.color_a.w + 0.1 * chips, bevel);
    return t;
}

fn flagstone(uv: vec2<f32>, checker: bool) -> Texel {
    let n = recipe.scale * 4.0;
    let warp = vec2<f32>(pfbm(uv, vec2<f32>(4.0), 3u, recipe.seed), pfbm(uv, vec2<f32>(4.0), 3u, recipe.seed + 11u)) * 0.02;
    let p = (uv + warp) * n;
    let cell = floor(p);
    let local = fract(p);
    let edge = min(min(local.x, 1.0 - local.x), min(local.y, 1.0 - local.y));
    let id = rand2(wrap(vec2<i32>(cell), vec2<i32>(i32(n))), recipe.seed);
    let groove = smoothstep(0.01, 0.035, edge);
    let wear = pfbm(uv, vec2<f32>(8.0), 5u, recipe.seed + 5u);
    var t: Texel;
    var color = mix(recipe.color_a.rgb, recipe.color_b.rgb, id);
    if checker {
        let odd = (u32(cell.x) + u32(cell.y)) % 2u == 1u;
        color = select(recipe.color_a.rgb, recipe.color_b.rgb, odd);
        // Marble veining.
        let veins = abs(sin((uv.x * 3.0 + uv.y * 5.0) * 6.2831 + 6.0 * pfbm(uv, vec2<f32>(4.0), 5u, recipe.seed + 9u)));
        color *= 0.85 + 0.15 * smoothstep(0.0, 0.25, veins);
    }
    t.albedo = mix(color * 0.4, color * (0.9 + 0.2 * wear), groove);
    t.height = groove * (0.9 + 0.1 * wear);
    t.roughness = mix(0.9, mix(recipe.color_a.w, recipe.color_b.w, id) + 0.08 * wear, groove);
    return t;
}

fn wood(uv: vec2<f32>) -> Texel {
    let planks = max(1.0, floor(recipe.scale * 4.0));
    let x = uv.x * planks;
    let plank = floor(x);
    let local = fract(x);
    let id = rand2(vec2<i32>(i32(plank), 1), recipe.seed);
    let grain_warp = pfbm(uv + vec2<f32>(id, 0.0), vec2<f32>(4.0, 1.0), 4u, recipe.seed + 2u);
    let rings = fract((local + grain_warp * 1.5 + id * 3.0) * 7.0 + pfbm(uv, vec2<f32>(2.0, 16.0), 3u, recipe.seed) * 0.6);
    let fibers = pfbm(uv, vec2<f32>(64.0 * planks, 4.0), 3u, recipe.seed + 4u);
    let gap = smoothstep(0.0, 0.025, min(local, 1.0 - local));
    var t: Texel;
    let base = mix(recipe.color_a.rgb, recipe.color_b.rgb, smoothstep(0.2, 0.9, rings) * 0.7 + id * 0.3);
    t.albedo = base * (0.85 + 0.25 * fibers) * mix(0.35, 1.0, gap);
    t.height = gap * (0.85 + 0.1 * fibers + 0.05 * rings);
    t.roughness = mix(recipe.color_a.w, recipe.color_b.w, rings) + 0.1 * (1.0 - gap);
    return t;
}

fn slate(uv: vec2<f32>) -> Texel {
    let counts = vec2<f32>(5.0, 10.0) * recipe.scale;
    let b = blocks(uv, counts, 0.5, recipe.seed);
    let row_t = fract(uv.y * counts.y);
    let n = pfbm(uv, counts * 2.0, 4u, recipe.seed + 1u);
    var t: Texel;
    // Each course overlaps the one below: height rises towards the bottom edge.
    t.height = (1.0 - row_t) * 0.7 + 0.3 * smoothstep(0.0, 0.06, b.x) + n * 0.05;
    let shade = mix(recipe.color_a.rgb, recipe.color_b.rgb, b.y);
    t.albedo = shade * (0.8 + 0.3 * n) * mix(0.5, 1.0, smoothstep(0.0, 0.05, b.x));
    t.roughness = recipe.color_a.w + 0.15 * n;
    return t;
}

fn glass_palette(h: f32) -> vec3<f32> {
    let colors = array<vec3<f32>, 6>(
        vec3<f32>(0.05, 0.12, 0.65), vec3<f32>(0.7, 0.06, 0.05), vec3<f32>(0.85, 0.6, 0.08),
        vec3<f32>(0.08, 0.45, 0.15), vec3<f32>(0.35, 0.08, 0.55), vec3<f32>(0.1, 0.35, 0.75),
    );
    var c = colors;
    return c[u32(h * 6.0) % 6u];
}

fn stained_glass(uv: vec2<f32>) -> Texel {
    let v = pvoronoi(uv, vec2<f32>(6.0, 10.0) * recipe.scale, 0.85, recipe.seed);
    let lead = smoothstep(0.03, 0.08, v.f2 - v.f1);
    // Panel border and a medallion in the middle.
    let border = smoothstep(0.02, 0.04, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
    let medallion = abs(length(uv - 0.5) - 0.3);
    let ring = smoothstep(0.012, 0.03, medallion);
    let streaks = pfbm(uv, vec2<f32>(4.0, 32.0), 3u, recipe.seed + 3u);
    var t: Texel;
    let glass = glass_palette(idf(v.id, 7u)) * (0.75 + 0.35 * streaks);
    let mask = lead * border * ring;
    t.albedo = mix(vec3<f32>(0.02), glass, mask);
    t.height = mix(1.0, 0.4, mask);
    // Roughness channel doubles as the glass mask: < 0.2 = glass.
    t.roughness = mix(0.7, 0.05, mask);
    return t;
}

fn fabric(uv: vec2<f32>) -> Texel {
    let threads = 96.0 * recipe.scale;
    let warp = sin(uv.x * threads * 3.14159) * sin(uv.y * threads * 0.5 * 3.14159);
    let n = pfbm(uv, vec2<f32>(8.0), 4u, recipe.seed);
    // Woven border bands.
    let band = smoothstep(0.02, 0.0, abs(fract(uv.y * 2.0) - 0.08)) + smoothstep(0.02, 0.0, abs(fract(uv.y * 2.0) - 0.92));
    var t: Texel;
    t.albedo = mix(recipe.color_a.rgb, recipe.color_b.rgb, saturate(band)) * (0.85 + 0.1 * warp + 0.15 * n);
    t.height = 0.5 + 0.25 * warp + 0.1 * n;
    t.roughness = recipe.color_a.w;
    return t;
}

fn bark(uv: vec2<f32>) -> Texel {
    let ridges = pridged(uv + vec2<f32>(pfbm(uv, vec2<f32>(2.0, 1.0), 3u, recipe.seed) * 0.15, 0.0), vec2<f32>(8.0, 2.0) * recipe.scale, 5u, recipe.seed + 1u);
    let plates = pvoronoi(uv, vec2<f32>(10.0, 3.0) * recipe.scale, 0.9, recipe.seed + 2u);
    let fissure = smoothstep(0.0, 0.15, plates.f2 - plates.f1);
    let moss = smoothstep(0.1, 0.5, pfbm(uv, vec2<f32>(3.0, 2.0), 4u, recipe.seed + 9u) + recipe.params.x - 0.4);
    var t: Texel;
    let h = ridges * 0.6 + fissure * 0.4;
    t.albedo = mix(recipe.color_b.rgb, recipe.color_a.rgb, h) * (0.7 + 0.4 * h);
    t.albedo = mix(t.albedo, vec3<f32>(0.12, 0.17, 0.05), moss * 0.8 * smoothstep(0.3, 0.7, h));
    t.height = h;
    t.roughness = mix(recipe.color_b.w, recipe.color_a.w, h);
    return t;
}

fn leaf(uv: vec2<f32>) -> Texel {
    let c = uv - vec2<f32>(0.5, 0.5);
    let midrib = smoothstep(0.012, 0.0, abs(c.x));
    let side = abs(fract((uv.y - abs(c.x) * 0.9) * 9.0) - 0.5);
    let veins = smoothstep(0.06, 0.0, side) * smoothstep(0.45, 0.05, abs(c.x));
    let n = pfbm(uv, vec2<f32>(6.0), 4u, recipe.seed);
    let tint = pfbm(uv, vec2<f32>(2.0), 3u, recipe.seed + 5u);
    var t: Texel;
    let base = mix(recipe.color_a.rgb, recipe.color_b.rgb, saturate(0.5 + tint * 1.4));
    t.albedo = base * (0.85 + 0.2 * n) * (1.0 + 0.35 * max(midrib, veins * 0.6));
    t.height = 0.5 + 0.2 * n - 0.15 * max(midrib, veins);
    t.roughness = recipe.color_a.w;
    return t;
}

fn moss(uv: vec2<f32>) -> Texel {
    let tufts = pvoronoi(uv, vec2<f32>(24.0) * recipe.scale, 1.0, recipe.seed);
    let fine = pfbm(uv, vec2<f32>(32.0) * recipe.scale, 4u, recipe.seed + 1u);
    let large = pfbm(uv, vec2<f32>(3.0), 4u, recipe.seed + 2u);
    var t: Texel;
    let h = (1.0 - tufts.f1) * 0.6 + fine * 0.3 + 0.1;
    t.albedo = mix(recipe.color_b.rgb, recipe.color_a.rgb, saturate(h + large * 0.5)) * (0.8 + 0.3 * fine);
    t.height = h;
    t.roughness = recipe.color_a.w;
    return t;
}

fn forest_floor(uv: vec2<f32>) -> Texel {
    let leaves = pvoronoi(uv, vec2<f32>(18.0) * recipe.scale, 1.0, recipe.seed);
    let leaves2 = pvoronoi(uv + 0.37, vec2<f32>(11.0) * recipe.scale, 1.0, recipe.seed + 3u);
    let soil = pfbm(uv, vec2<f32>(16.0), 5u, recipe.seed + 1u);
    let twig_line = abs(sin((uv.x * 7.0 + uv.y * 3.0 + pfbm(uv, vec2<f32>(4.0), 3u, recipe.seed + 8u)) * 25.0));
    let twig = smoothstep(0.04, 0.0, twig_line) * step(0.75, pfbm(uv, vec2<f32>(5.0), 2u, recipe.seed + 6u) + 0.6);
    var t: Texel;
    let leaf_a = mix(recipe.color_a.rgb, vec3<f32>(0.45, 0.22, 0.06), idf(leaves.id, 1u));
    let leaf_b = mix(recipe.color_a.rgb * 0.7, vec3<f32>(0.3, 0.27, 0.08), idf(leaves2.id, 2u));
    let cover = smoothstep(0.55, 0.35, leaves.f1) ;
    let cover2 = smoothstep(0.5, 0.3, leaves2.f1) * (1.0 - cover);
    var albedo = recipe.color_b.rgb * (0.7 + 0.4 * soil);
    albedo = mix(albedo, leaf_b, cover2);
    albedo = mix(albedo, leaf_a, cover);
    albedo = mix(albedo, vec3<f32>(0.18, 0.12, 0.07), twig);
    t.albedo = albedo;
    t.height = 0.3 * soil + 0.35 * cover + 0.25 * cover2 + 0.3 * twig;
    t.roughness = recipe.color_a.w;
    return t;
}

fn grass(uv: vec2<f32>) -> Texel {
    let blades = pfbm(uv, vec2<f32>(96.0, 24.0) * recipe.scale, 3u, recipe.seed);
    let clumps = pfbm(uv, vec2<f32>(6.0), 4u, recipe.seed + 1u);
    var t: Texel;
    t.albedo = mix(recipe.color_b.rgb, recipe.color_a.rgb, saturate(0.5 + clumps * 1.2)) * (0.75 + 0.4 * blades);
    t.height = 0.5 + 0.3 * blades + 0.2 * clumps;
    t.roughness = recipe.color_a.w;
    return t;
}

// A stone of Voronoi cell `v` at `p` (cell units): turned and squashed by its hash, sized from
// a few large to many small. Returns its dome (0 at its outline, 1 on top; negative outside).
fn stone_dome(v: Voronoi, p: vec2<f32>) -> f32 {
    let a = idf(v.id, 4u) * 6.2831853;
    let axis = vec2<f32>(cos(a), sin(a));
    let d = p - v.center;
    let local = vec2<f32>(dot(d, axis), dot(d, vec2<f32>(-axis.y, axis.x)) * (1.0 + 0.8 * idf(v.id, 5u)));
    let r = mix(0.16, 0.44, pow(idf(v.id, 6u), 1.6));
    let t = dot(local, local) / (r * r);
    return select(-1.0, sqrt(max(1.0 - t, 0.0)), t < 1.0);
}

// Earth with gravel and stones, made to look like ground rather than a dot pattern: the stones
// gather in patches (sparse elsewhere), come in two populations of varied sizes and shapes,
// and lie partly buried - each sinks by its own amount into a lumpy soil, so its visible outline
// is wherever it rises out of the clods, not a circle. Stones are dusted with the soil and
// darker where it covers them.
fn dirt(uv: vec2<f32>) -> Texel {
    let seed = recipe.seed;
    let n = pfbm(uv, vec2<f32>(12.0), 5u, seed + 1u);
    let clods = pfbm(uv, vec2<f32>(40.0) * recipe.scale, 3u, seed + 2u);
    let gather = smoothstep(-0.35, 0.45, pfbm(uv, vec2<f32>(5.0), 3u, seed + 3u));
    let warp = vec2<f32>(pfbm(uv, vec2<f32>(24.0), 2u, seed + 4u), pfbm(uv, vec2<f32>(24.0), 2u, seed + 5u)) * 0.004;
    let soil_height = 0.42 + 0.1 * n + 0.07 * clods;
    let soil = recipe.color_a.rgb * (0.75 + 0.4 * n) * (0.92 + 0.16 * clods);
    var height = soil_height;
    var albedo = soil;
    var cover = 0.0;
    for (var layer = 0u; layer < 2u; layer++) {
        // Gravel (many, small, low) and stones (fewer, larger, higher).
        let cells = select(vec2<f32>(60.0), vec2<f32>(23.0), layer == 1u) * recipe.scale;
        let density = select(0.5, 0.3, layer == 1u) * (0.15 + 1.1 * gather);
        let amplitude = select(0.16, 0.32, layer == 1u);
        let q = uv + warp;
        let v = pvoronoi(q, cells, 0.8, seed + 20u + 7u * layer);
        if idf(v.id, 3u) > density {
            continue;
        }
        let dome = stone_dome(v, q * cells);
        if dome < 0.0 {
            continue;
        }
        // Each stone sinks by its own share of its height into the soil's mean level.
        let bury = 0.15 + 0.6 * idf(v.id, 7u);
        let top = 0.42 + amplitude * (dome - bury);
        let above = top - soil_height;
        if above <= 0.0 || top <= height {
            continue;
        }
        height = top;
        let tone = idf(v.id, 8u);
        let hue = mix(vec3<f32>(1.06, 0.98, 0.88), vec3<f32>(0.92, 0.97, 1.04), idf(v.id, 9u));
        var stone = recipe.color_b.rgb * (0.6 + 0.65 * tone) * hue * (0.9 + 0.2 * clods);
        // Dusted with the soil, and coated with it where it is about to disappear under it.
        let coat = 0.3 + 0.6 * (1.0 - smoothstep(0.0, 0.05, above));
        stone = mix(stone, soil, coat);
        albedo = stone;
        cover = smoothstep(0.0, 0.03, above);
    }
    var t: Texel;
    t.albedo = mix(soil, albedo, cover);
    t.height = height;
    t.roughness = mix(recipe.color_a.w, recipe.color_b.w, cover);
    return t;
}

fn rock(uv: vec2<f32>) -> Texel {
    let r = pridged(uv, vec2<f32>(4.0) * recipe.scale, 6u, recipe.seed);
    let cracks = pvoronoi(uv, vec2<f32>(7.0) * recipe.scale, 1.0, recipe.seed + 1u);
    let crack = smoothstep(0.0, 0.05, cracks.f2 - cracks.f1);
    let n = pfbm(uv, vec2<f32>(16.0), 4u, recipe.seed + 2u);
    var t: Texel;
    t.height = r * 0.7 + crack * 0.3;
    t.albedo = mix(recipe.color_b.rgb, recipe.color_a.rgb, saturate(r + n * 0.3)) * mix(0.5, 1.0, crack);
    t.roughness = mix(recipe.color_b.w, recipe.color_a.w, r);
    return t;
}

fn snow(uv: vec2<f32>) -> Texel {
    let n = pfbm(uv, vec2<f32>(8.0), 5u, recipe.seed);
    var t: Texel;
    t.albedo = recipe.color_a.rgb * (0.95 + 0.05 * n);
    t.height = 0.5 + 0.3 * n;
    t.roughness = recipe.color_a.w + 0.1 * n;
    return t;
}

fn hammered(uv: vec2<f32>) -> Texel {
    let dents = pvoronoi(uv, vec2<f32>(20.0) * recipe.scale, 1.0, recipe.seed);
    let n = pfbm(uv, vec2<f32>(16.0), 3u, recipe.seed + 1u);
    var t: Texel;
    t.height = 1.0 - dents.f1 * dents.f1;
    t.albedo = recipe.color_a.rgb * (0.92 + 0.08 * n);
    t.roughness = recipe.color_a.w + 0.1 * dents.f1 + 0.05 * n;
    return t;
}

fn plaster(uv: vec2<f32>) -> Texel {
    let n = pfbm(uv, vec2<f32>(6.0) * recipe.scale, 6u, recipe.seed);
    let stain = pfbm(uv, vec2<f32>(2.0), 4u, recipe.seed + 1u);
    var t: Texel;
    t.albedo = mix(recipe.color_a.rgb, recipe.color_b.rgb, saturate(0.5 + stain)) * (0.95 + 0.08 * n);
    t.height = 0.5 + 0.15 * n;
    t.roughness = recipe.color_a.w;
    return t;
}

// Rounded setts in sandy joints (a jittered cell grid): domed stones, so displacement makes them
// stand out of the joints. `param`: joint width.
fn cobblestone(uv: vec2<f32>) -> Texel {
    let cells = vec2<f32>(8.0) * recipe.scale;
    let warp = vec2<f32>(pfbm(uv, vec2<f32>(4.0), 3u, recipe.seed + 21u), pfbm(uv, vec2<f32>(4.0), 3u, recipe.seed + 22u)) * 0.012;
    let v = pvoronoi(uv + warp, cells, 0.7, recipe.seed);
    let gap = 0.05 + 0.12 * recipe.params.x;
    let border = v.f2 - v.f1;
    let stone = smoothstep(gap * 0.4, gap * 1.6, border);
    let dome = sqrt(saturate(1.0 - (v.f1 * v.f1 * 1.6)));
    let chips = pfbm(uv, cells * 6.0, 4u, recipe.seed + 3u);
    let sand = pfbm(uv, cells * 3.0, 4u, recipe.seed + 4u);
    let tone = idf(v.id, 11u);
    var t: Texel;
    t.height = mix(0.12 + 0.1 * sand, 0.55 + 0.35 * dome * (0.8 + 0.2 * tone) + 0.05 * chips, stone);
    let rock = mix(recipe.color_a.rgb, recipe.color_b.rgb, tone) * (0.8 + 0.35 * chips) * (0.85 + 0.2 * dome);
    let joint = mix(recipe.color_b.rgb * 0.45, vec3<f32>(0.16, 0.17, 0.08), saturate(0.5 + sand));
    t.albedo = mix(joint, rock, stone);
    t.roughness = mix(0.95, recipe.color_a.w - 0.12 * dome, stone);
    return t;
}

// A planar facet per Voronoi cell: the height of a randomly tilted plane through the cell's
// centre (conchoidal chips, fracture planes), falling to 0 towards the cell's border.
fn facet(v: Voronoi, p: vec2<f32>, salt: u32) -> f32 {
    let a = idf(v.id, salt) * 6.2831853;
    let tilt = vec2<f32>(cos(a), sin(a)) * (0.3 + 0.7 * idf(v.id, salt + 1u));
    return dot(p - v.center, tilt) * smoothstep(0.0, 0.3, v.f2 - v.f1);
}

// Weathered cliff rock for the side projections of a triplanar mapping (u horizontal, v up):
// rounded masses flattened along the bedding, horizontal lamination, broad fracture planes and
// small chips (tilted facets, not crack lines), a few fractures that come and go, dark streaks
// running down the face and fine grain. No closed network of cracks: tiled at two scales it
// must not read as paving.
fn cliff(uv: vec2<f32>) -> Texel {
    let seed = recipe.seed;
    let warp = vec2<f32>(pfbm(uv, vec2<f32>(3.0), 3u, seed + 7u), pfbm(uv, vec2<f32>(3.0), 3u, seed + 8u));
    let w = uv + warp * 0.05;
    let mass = pfbm(w, vec2<f32>(3.0, 5.0), 5u, seed);
    let lamination = pridged(w, vec2<f32>(4.0, 9.0), 4u, seed + 1u);
    let planes_cells = vec2<f32>(3.0, 2.0) * recipe.scale;
    let planes = pvoronoi(w, planes_cells, 0.9, seed + 2u);
    let plane = facet(planes, w * planes_cells, 3u);
    let chips_cells = vec2<f32>(11.0, 9.0) * recipe.scale;
    let chips = pvoronoi(w + warp * 0.03, chips_cells, 1.0, seed + 4u);
    let chip = facet(chips, (w + warp * 0.03) * chips_cells, 5u);
    let grain = pfbm(uv, vec2<f32>(64.0), 3u, seed + 9u);
    // Fractures along some fracture-plane borders only (a mask fades them in and out).
    let mask = smoothstep(0.1, 0.4, pfbm(uv, vec2<f32>(4.0, 2.0), 3u, seed + 10u));
    let fracture = (1.0 - smoothstep(0.0, 0.05, planes.f2 - planes.f1)) * mask;
    let streak = pfbm(w, vec2<f32>(12.0, 3.0), 4u, seed + 11u);
    var t: Texel;
    t.height = 0.5 + 0.2 * mass + 0.1 * (lamination - 0.5) + 0.1 * plane + 0.07 * chip + 0.03 * grain - 0.22 * fracture;
    let tone = saturate(0.55 + 0.9 * mass + 0.5 * (lamination - 0.5) + 0.15 * plane);
    var albedo = mix(recipe.color_b.rgb, recipe.color_a.rgb, tone);
    albedo *= (0.93 + 0.14 * idf(chips.id, 6u)) * (0.9 + 0.2 * grain);
    albedo *= 1.0 - 0.22 * smoothstep(0.05, 0.5, streak);
    albedo *= mix(1.0, 0.5, fracture);
    t.albedo = albedo;
    t.roughness = saturate(mix(recipe.color_a.w, recipe.color_b.w, 1.0 - tone) + 0.05 * grain - 0.06 * smoothstep(0.05, 0.5, streak));
    return t;
}

fn evaluate(uv: vec2<f32>) -> Texel {
    switch PATTERN {
        case PATTERN_ASHLAR: { return ashlar(uv, false); }
        case PATTERN_CARVED_STONE: { return ashlar(uv, true); }
        case PATTERN_FLAGSTONE: { return flagstone(uv, false); }
        case PATTERN_MARBLE_TILES: { return flagstone(uv, true); }
        case PATTERN_WOOD: { return wood(uv); }
        case PATTERN_SLATE: { return slate(uv); }
        case PATTERN_STAINED_GLASS: { return stained_glass(uv); }
        case PATTERN_FABRIC: { return fabric(uv); }
        case PATTERN_BARK: { return bark(uv); }
        case PATTERN_LEAF: { return leaf(uv); }
        case PATTERN_MOSS: { return moss(uv); }
        case PATTERN_FOREST_FLOOR: { return forest_floor(uv); }
        case PATTERN_GRASS: { return grass(uv); }
        case PATTERN_DIRT: { return dirt(uv); }
        case PATTERN_ROCK: { return rock(uv); }
        case PATTERN_SNOW: { return snow(uv); }
        case PATTERN_HAMMERED: { return hammered(uv); }
        case PATTERN_COBBLESTONE: { return cobblestone(uv); }
        case PATTERN_CLIFF: { return cliff(uv); }
        default: { return plaster(uv); }
    }
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@compute @workgroup_size(8, 8)
fn generate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_albedo);
    if any(gid.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    let t = evaluate(uv);
    textureStore(out_albedo, gid.xy, vec4<f32>(linear_to_srgb(saturate(t.albedo)), 1.0));
    textureStore(out_height, gid.xy, vec4<f32>(saturate(t.height), saturate(t.roughness), 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(in_height);
    if any(gid.xy >= size) {
        return;
    }
    let s = vec2<i32>(size);
    let p = vec2<i32>(gid.xy);
    var heights: array<f32, 9>;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            heights[(y + 1) * 3 + x + 1] = textureLoad(in_height, wrap(p + vec2<i32>(x, y), s), 0).r;
        }
    }
    let dx = (heights[2] + 2.0 * heights[5] + heights[8]) - (heights[0] + 2.0 * heights[3] + heights[6]);
    let dy = (heights[6] + 2.0 * heights[7] + heights[8]) - (heights[0] + 2.0 * heights[1] + heights[2]);
    // Bump strength scales with resolution so the relief is resolution independent.
    let strength = recipe.params.w * f32(size.x) / 128.0;
    let n = normalize(vec3<f32>(-dx * strength, dy * strength, 1.0));
    let center = textureLoad(in_height, p, 0);
    textureStore(out_data, gid.xy, vec4<f32>(n.xy * 0.5 + 0.5, center.g, center.r));
}

@compute @workgroup_size(8, 8)
fn downsample_albedo(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_albedo);
    if any(gid.xy >= size) {
        return;
    }
    let base = vec2<i32>(gid.xy * 2u);
    var sum = vec4<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        let texel = textureLoad(in_mip, base + vec2<i32>(i & 1, i >> 1), 0);
        sum += vec4<f32>(srgb_to_linear(texel.rgb), texel.a);
    }
    textureStore(out_albedo, gid.xy, vec4<f32>(linear_to_srgb(sum.rgb * 0.25), sum.a * 0.25));
}

// Shorter averaged normals mean more variance below the texel: fold it into roughness
// (Toksvig) so distant detail turns rough instead of aliasing. `len` is the length of the
// average of unit normals.
fn toksvig_roughness(roughness: f32, len: f32) -> f32 {
    return sqrt(saturate(roughness * roughness + (1.0 - len) / max(len, 1e-3) * 0.5));
}

// Bilinear taps per axis that cover one destination texel's footprint in a source image
// (an exact box filter for integer ratios; one tap = bilinear when magnifying).
fn footprint_taps(source: vec2<u32>, dest: vec2<u32>) -> vec2<u32> {
    return clamp((source + dest - 1u) / dest, vec2<u32>(1u), vec2<u32>(16u));
}

fn tap_uv(texel: vec2<u32>, size: vec2<u32>, tap: vec2<u32>, taps: vec2<u32>) -> vec2<f32> {
    return (vec2<f32>(texel) + (vec2<f32>(tap) + 0.5) / vec2<f32>(taps)) / vec2<f32>(size);
}

@compute @workgroup_size(8, 8)
fn pack_images(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_albedo);
    if any(gid.xy >= size) {
        return;
    }
    // Base colour, averaged in linear light (the sRGB view decodes; the store re-encodes).
    var color = vec4<f32>(0.0);
    let color_taps = footprint_taps(textureDimensions(in_base_color), size);
    for (var y = 0u; y < color_taps.y; y++) {
        for (var x = 0u; x < color_taps.x; x++) {
            color += textureSampleLevel(in_base_color, image_sampler, tap_uv(gid.xy, size, vec2<u32>(x, y), color_taps), 0.0);
        }
    }
    color /= f32(color_taps.x * color_taps.y);
    if PACK_OCCLUSION {
        var occlusion = 0.0;
        let occlusion_taps = footprint_taps(textureDimensions(in_occlusion), size);
        for (var y = 0u; y < occlusion_taps.y; y++) {
            for (var x = 0u; x < occlusion_taps.x; x++) {
                occlusion += textureSampleLevel(in_occlusion, image_sampler, tap_uv(gid.xy, size, vec2<u32>(x, y), occlusion_taps), 0.0).r;
            }
        }
        color.a = occlusion / f32(occlusion_taps.x * occlusion_taps.y);
    }

    // Normals: average unit vectors; the shortening becomes roughness (as in the mip chain).
    var n = vec3<f32>(0.0);
    let normal_taps = footprint_taps(textureDimensions(in_normal), size);
    for (var y = 0u; y < normal_taps.y; y++) {
        for (var x = 0u; x < normal_taps.x; x++) {
            let raw = textureSampleLevel(in_normal, image_sampler, tap_uv(gid.xy, size, vec2<u32>(x, y), normal_taps), 0.0).xyz;
            n += normalize(raw * 2.0 - 1.0 + vec3<f32>(0.0, 0.0, 1e-4));
        }
    }
    let len = length(n) / f32(normal_taps.x * normal_taps.y);

    var mr = vec2<f32>(0.0);
    let mr_taps = footprint_taps(textureDimensions(in_metallic_roughness), size);
    for (var y = 0u; y < mr_taps.y; y++) {
        for (var x = 0u; x < mr_taps.x; x++) {
            mr += textureSampleLevel(in_metallic_roughness, image_sampler, tap_uv(gid.xy, size, vec2<u32>(x, y), mr_taps), 0.0).gb;
        }
    }
    mr /= f32(mr_taps.x * mr_taps.y);

    let nn = normalize(n + vec3<f32>(0.0, 0.0, 1e-6));
    textureStore(out_albedo, gid.xy, vec4<f32>(linear_to_srgb(saturate(color.rgb)), color.a));
    textureStore(out_data, gid.xy, vec4<f32>(nn.xy * 0.5 + 0.5, toksvig_roughness(mr.x, len), mr.y));
}

@compute @workgroup_size(8, 8)
fn downsample_data(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_data);
    if any(gid.xy >= size) {
        return;
    }
    let base = vec2<i32>(gid.xy * 2u);
    var n = vec3<f32>(0.0);
    var rough = 0.0;
    var height = 0.0;
    for (var i = 0; i < 4; i++) {
        let d = textureLoad(in_mip, base + vec2<i32>(i & 1, i >> 1), 0);
        let xy = d.xy * 2.0 - 1.0;
        n += vec3<f32>(xy, sqrt(saturate(1.0 - dot(xy, xy))));
        rough += d.z;
        height += d.w;
    }
    let nn = normalize(n);
    textureStore(out_data, gid.xy, vec4<f32>(nn.xy * 0.5 + 0.5, toksvig_roughness(rough * 0.25, length(n) * 0.25), height * 0.25));
}
