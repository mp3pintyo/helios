// Virtual shadow map addressing (shared by marking, rendering and sampling).
//
// The sun's shadow is a clipmap of LEVELS levels, each a 16k x 16k virtual texture made of
// 128 x 128 pages of 128^2 texels; level i covers 2^i times the area of level 0. Pages are
// addressed by absolute light-space page coordinates (x along light X, y along light -Y), so
// they stay valid while the camera moves; the page table slot is the coordinate modulo 128.
// Resident pages live in one physical depth atlas.
//
// Static / dynamic caching (as UE5's "static separate" cache): a page that only needs refreshing
// because dynamic casters moved (wind) keeps its static casters in a static layer, the u32
// software raster mirror of its tile (vsm.rs), and re-renders just the dynamic casters over a
// copy of it. Each page rendered in a frame has a kind (info buffer, PAGE_STATE).

// Word offsets in the VSM info buffer (vsm.wgsl; mirrored by vsm.rs, cull.wgsl and
// vsm_local_common.wgsl).
const INFO_ARGS: u32 = 128u;
// Pages rendered this frame: dynamic-only, split; scratch tiles handed out.
const INFO_COUNTERS: u32 = 132u;
// A bit per clipmap page table entry for every page rendered this frame, then one for those
// rendered in full (static casters included).
const INFO_DIRTY_MASK: u32 = 136u;
const INFO_FULL_MASK: u32 = 8328u;
// Per physical page: this frame's PAGE_* kind | scratch tile << 8 (valid for pages rendered this
// frame); the entry whose static layer the mirror holds (or INVALID_ID); a pending dynamic
// refresh (vsm_invalidate.wgsl); motion bounds (two words, below).
const INFO_PAGE_STATE: u32 = 22152u;
const INFO_PAGE_STATIC: u32 = 23176u;
const INFO_PAGE_REFRESH: u32 = 24200u;
const INFO_PAGE_MOTION: u32 = 25224u;
// INFO_PAGE_STATIC of a page whose static layer is out of date (a caster changed layer): the
// mirror still holds it, so the page's next render clears it.
const STALE_STATIC_LAYER: u32 = 0xfffffffeu;

// How a page is rendered this frame.
// Every caster, no static layer kept (new pages; pages whose static casters changed).
const PAGE_FULL: u32 = 1u;
// Every caster; the static ones are captured into the page's static layer first (the first
// dynamic refresh of a page without one).
const PAGE_SPLIT: u32 = 2u;
// Dynamic casters only, over a copy of the static layer.
const PAGE_DYNAMIC: u32 = 3u;
const PAGE_KIND_MASK: u32 = 3u;
// The page's mirror still holds an older static layer: clear it before rasterising.
const PAGE_CLEAR: u32 = 4u;
// Dynamic pages: software-rasterised dynamic casters go to a scratch tile of the mirror after
// the atlas' tiles (tile index physical_count + scratch); NO_SCRATCH if none was left (they take
// the hardware path).
const NO_SCRATCH: u32 = 0xffffffu;

// Local light pages keep bounds of their wind-swayed casters' motion (INFO_PAGE_MOTION, two
// words per physical page: speed in texels per second and the largest distance between two
// poses in texels, both in 1 / PAGE_MOTION_UNITS). The page's rendering measures them
// (vsm_expand.wgsl), its invalidation schedules the next refresh by them (vsm_invalidate.wgsl).
const PAGE_MOTION_UNITS: f32 = 16.0;

fn page_kind(state: u32) -> u32 {
    return state & PAGE_KIND_MASK;
}

fn page_scratch(state: u32) -> u32 {
    return state >> 8u;
}

struct Vsm {
    light_view: mat4x4<f32>,
    level_view_proj: array<mat4x4<f32>, 16>,
    // xy: absolute page coordinate of the level's first page.
    level_origin: array<vec4<i32>, 16>,
    // x: page world size, y: texel world size.
    level_info: array<vec4<f32>, 16>,
    level_count: u32,
    page_texels: u32,
    level_pages: u32,
    tiles_per_row: u32,
    atlas_size: f32,
    first_texel: f32,
    lod_bias: f32,
    frame: u32,
    // World size of one screen pixel at distance 1 (for level selection).
    pixel_angle: f32,
    physical_count: u32,
    enabled: u32,
    draw_capacity: u32,
    eye: vec4<f32>,
}

struct VsmAddress {
    valid: bool,
    entry: u32,
    texel: vec2<f32>,   // texel position inside the page, [0, page_texels)
    depth: f32,
}

// Clipmap level whose texels match the screen footprint at `world`.
fn vsm_level(world: vec3<f32>) -> u32 {
    let footprint = distance(world, vsm.eye.xyz) * vsm.pixel_angle;
    let level = ceil(log2(max(footprint / vsm.first_texel, 1e-6)) + vsm.lod_bias);
    return u32(clamp(level, 0.0, f32(vsm.level_count - 1u)));
}

fn vsm_address(world: vec3<f32>, level: u32) -> VsmAddress {
    var a: VsmAddress;
    let ls = (vsm.light_view * vec4<f32>(world, 1.0)).xyz;
    let v = vec2<f32>(ls.x, -ls.y) / vsm.level_info[level].x;
    let page = vec2<i32>(floor(v));
    let local = page - vsm.level_origin[level].xy;
    let n = i32(vsm.level_pages);
    a.valid = all(local >= vec2<i32>(0)) && all(local < vec2<i32>(n));
    let wrapped = ((page % n) + n) % n;
    a.entry = level * vsm.level_pages * vsm.level_pages + u32(wrapped.y) * vsm.level_pages + u32(wrapped.x);
    a.texel = (v - vec2<f32>(page)) * f32(vsm.page_texels);
    a.depth = (vsm.level_view_proj[level] * vec4<f32>(world, 1.0)).z;
    return a;
}

fn vsm_tile(physical: u32) -> vec2<f32> {
    return vec2<f32>(f32(physical % vsm.tiles_per_row), f32(physical / vsm.tiles_per_row)) * f32(vsm.page_texels);
}

fn decode_tag(tag: u32) -> vec2<i32> {
    return vec2<i32>(i32(tag << 16u) >> 16u, i32(tag) >> 16u);
}
