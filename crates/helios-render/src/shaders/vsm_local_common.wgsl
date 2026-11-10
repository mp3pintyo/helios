// Virtual shadow maps of local lights (see vsm_local.rs): addressing shared by marking,
// rendering and sampling. Requires vsm_common.wgsl and a `local_vsm: LocalVsm` uniform.
//
// A shadowed point light has six cube faces, a spot light one perspective face. Each face is a
// 4096² virtual shadow map with a mip chain (32² pages of 128² texels at mip 0 down to a single
// page); a pixel samples the mip whose texels match its screen footprint, as the sun's clipmap
// levels do. Pages are entries of the sun's page cache placed after its clipmap levels
// (`entry_base`) and live in the same physical atlas: all lights share one pool, one allocation
// and one page renderer. Faces use an infinite reverse-Z perspective, depth = near / w with w the
// distance along the face axis.

const LOCAL_SLOTS: u32 = 32u;
const LOCAL_FACES: u32 = 6u;
const LOCAL_MIPS: u32 = 6u;
const LOCAL_PAGES: u32 = 32u;          // pages per face side at mip 0
const LOCAL_FACE_ENTRIES: u32 = 1365u; // 32² + 16² + 8² + 4² + 2² + 1
// Page draws (vsm_expand.wgsl) store their view in a `level` field: sun clipmap levels below
// LOCAL_LEVEL, local pages LOCAL_LEVEL + (slot * LOCAL_FACES + face) * LOCAL_MIPS + mip.
const LOCAL_LEVEL: u32 = 16u;
// Word offsets of the local lights' dirty-page data in the VSM info buffer (vsm.wgsl): 32 words
// per slot after the sun's clipmap data (its dirty page bit masks end at word 136 + 2 * 8192),
// then 4 words per face mip.
const LOCAL_INFO_SLOTS: u32 = 16520u;
const LOCAL_INFO_SLOT_WORDS: u32 = 32u;
const LOCAL_INFO_RECTS: u32 = 17544u;
// Upper bound of filter radii in texels (sampling) and of the footprint marking requests.
const LOCAL_MAX_FILTER_TEXELS: f32 = 10.0;

struct LocalShadowSlot {
    view_proj: array<mat4x4<f32>, 6>,
    inv_view_proj: array<mat4x4<f32>, 6>,
    // xyz: light position, w: range
    position_range: vec4<f32>,
    // x: face count (0: unused slot, 1: spot, 6: point), y: near plane, z: focal length
    // (1 / tan(half field of view)), w: mip bias (soft lights need less resolution)
    params: vec4<f32>,
    // x: 1 = cached pages are stale (the map's origin or shape changed, or the slot was reassigned)
    flags: vec4<u32>,
    // xyz: the map's origin (position_range.xyz) minus the light's position. Zero unless the
    // light moves: its map then stays at the origin it was rendered from while the light is
    // near it (vsm_anchor.rs), and receivers correct the parallax (local_filter_follow).
    anchor_offset: vec4<f32>,
}

struct LocalVsm {
    slots: array<LocalShadowSlot, 32>,
    // First page cache entry of slot 0.
    entry_base: u32,
    enabled: u32,
    lod_bias: f32,
    _pad: u32,
}

struct LocalAddress {
    valid: bool,
    entry: u32,
    page: vec2<u32>,
    texel: vec2<f32>,  // texel position inside the page, [0, page_texels)
    depth: f32,
}

// Cube face whose frustum contains `world` (always 0 for spot lights).
fn local_face(slot: u32, world: vec3<f32>) -> u32 {
    if local_vsm.slots[slot].params.x < 6.0 {
        return 0u;
    }
    let d = world - local_vsm.slots[slot].position_range.xyz;
    let a = abs(d);
    if a.x >= a.y && a.x >= a.z {
        return select(1u, 0u, d.x > 0.0);
    }
    if a.y >= a.z {
        return select(3u, 2u, d.y > 0.0);
    }
    return select(5u, 4u, d.z > 0.0);
}

// Distance of `world` along the face axis (clip w).
fn local_face_depth(slot: u32, face: u32, world: vec3<f32>) -> f32 {
    return (local_vsm.slots[slot].view_proj[face] * vec4<f32>(world, 1.0)).w;
}

// World size of a texel of `mip` at face depth `w`.
fn local_texel(slot: u32, w: f32, mip: u32) -> f32 {
    let resolution = f32((LOCAL_PAGES >> mip) * vsm.page_texels);
    return 2.0 * w / (local_vsm.slots[slot].params.z * resolution);
}

// Mip whose texels match the screen footprint at `world` (face depth `w`), as vsm_level does for
// the sun.
fn local_mip(slot: u32, world: vec3<f32>, w: f32) -> u32 {
    let footprint = distance(world, vsm.eye.xyz) * vsm.pixel_angle;
    let bias = local_vsm.lod_bias + local_vsm.slots[slot].params.w;
    let mip = ceil(log2(max(footprint / local_texel(slot, w, 0u), 1e-6)) + bias);
    return u32(clamp(mip, 0.0, f32(LOCAL_MIPS - 1u)));
}

fn local_mip_pages(mip: u32) -> u32 {
    return LOCAL_PAGES >> mip;
}

// First entry of `mip` inside a face: the sizes of the finer mips, (4^6 - 4^(6 - mip)) / 3.
fn local_mip_offset(mip: u32) -> u32 {
    return (4096u - (4096u >> (2u * mip))) / 3u;
}

fn local_entry(slot: u32, face: u32, mip: u32, page: vec2<u32>) -> u32 {
    let face_base = local_vsm.entry_base + (slot * LOCAL_FACES + face) * LOCAL_FACE_ENTRIES;
    return face_base + local_mip_offset(mip) + page.y * local_mip_pages(mip) + page.x;
}

fn local_address(slot: u32, face: u32, world: vec3<f32>, mip: u32) -> LocalAddress {
    var a: LocalAddress;
    let clip = local_vsm.slots[slot].view_proj[face] * vec4<f32>(world, 1.0);
    let w = max(clip.w, 1e-6);
    let uv = vec2<f32>(clip.x, -clip.y) / w * 0.5 + 0.5;
    a.valid = clip.w > 0.0 && all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0));
    let pages = f32(local_mip_pages(mip));
    let v = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * pages;
    a.page = vec2<u32>(min(v, vec2<f32>(pages - 1.0)));
    a.texel = (v - vec2<f32>(a.page)) * f32(vsm.page_texels);
    a.entry = local_entry(slot, face, mip, a.page);
    a.depth = clip.z / w;
    return a;
}

// Homogeneous page-texel position (x, y, depth, 1) * w of a world point in a local light page,
// for the page rasterisers (vsm_page.wgsl); `view_mip` = (slot * LOCAL_FACES + face) *
// LOCAL_MIPS + mip.
fn local_page_vertex(world: vec3<f32>, view_mip: u32, page: vec2<f32>) -> vec4<f32> {
    let view = view_mip / LOCAL_MIPS;
    let mip = view_mip % LOCAL_MIPS;
    let clip = local_vsm.slots[view / LOCAL_FACES].view_proj[view % LOCAL_FACES] * vec4<f32>(world, 1.0);
    let size = f32(local_mip_pages(mip) * vsm.page_texels);
    let texel = vec2<f32>(clip.x + clip.w, clip.w - clip.y) * (0.5 * size) - page * f32(vsm.page_texels) * clip.w;
    return vec4<f32>(texel, clip.z, clip.w);
}

// Where a receiver samples a light's shadow map: face, mip and the biased filter centre (a normal
// offset and a step towards the light, in texels of the mip, as for the sun).
struct LocalReceiver {
    face: u32,
    w: f32,
    mip: u32,
    texel: f32,
    p: vec3<f32>,
    l: vec3<f32>,
}

fn local_receiver(slot: u32, world: vec3<f32>, n: vec3<f32>, mip_bias: u32) -> LocalReceiver {
    var r: LocalReceiver;
    r.l = normalize(local_vsm.slots[slot].position_range.xyz - world);
    r.face = local_face(slot, world);
    r.w = local_face_depth(slot, r.face, world);
    r.mip = min(local_mip(slot, world, r.w) + mip_bias, LOCAL_MIPS - 1u);
    r.texel = local_texel(slot, r.w, r.mip);
    let n_dot_l = saturate(dot(n, r.l));
    r.p = world + n * r.texel * (1.0 + 2.5 * (1.0 - n_dot_l)) + r.l * r.texel * 1.5;
    return r;
}

// Unit vector perpendicular to `l`: with cross(l, it) the axes of filter disks facing the light.
fn local_tangent(l: vec3<f32>) -> vec3<f32> {
    return normalize(cross(l, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(l.y) > 0.9)));
}

// A local light page cache entry decoded.
struct LocalPageId {
    slot: u32,
    face: u32,
    mip: u32,
    page: vec2<u32>,
}

fn local_page_id(entry: u32) -> LocalPageId {
    var id: LocalPageId;
    let e = entry - local_vsm.entry_base;
    let view = e / LOCAL_FACE_ENTRIES;
    id.slot = view / LOCAL_FACES;
    id.face = view % LOCAL_FACES;
    let in_face = e % LOCAL_FACE_ENTRIES;
    id.mip = 0u;
    while id.mip + 1u < LOCAL_MIPS && in_face >= local_mip_offset(id.mip + 1u) {
        id.mip++;
    }
    let pages = local_mip_pages(id.mip);
    let index = in_face - local_mip_offset(id.mip);
    id.page = vec2<u32>(index % pages, index / pages);
    return id;
}

// Face uv bounds of a world sphere: x / w over the sphere's bounding box; `valid` = false if it
// misses the face. Of a sphere that reaches the near plane, the part in front of the plane
// counts: unbounded towards the side the sphere straddles the face axis on, but a sphere wholly
// to one side of the axis still projects beyond x / (its far depth) there - which lies outside
// the face for most of the spheres around a light that cross a face's near plane.
struct LocalSphereRect {
    valid: bool,
    lo: vec2<f32>,
    hi: vec2<f32>,
    // Face depth of the sphere's nearest point.
    front: f32,
    // Lower bound of the face depth of its points inside the face: no nearer than `front`, and
    // inside the face |x| <= w, so a sphere off to one side is at least that far along the axis.
    inside: f32,
}

fn local_sphere_rect(slot: u32, face: u32, sphere: vec4<f32>) -> LocalSphereRect {
    var r: LocalSphereRect;
    let near = local_vsm.slots[slot].params.y;
    let focal = local_vsm.slots[slot].params.z;
    let clip = local_vsm.slots[slot].view_proj[face] * vec4<f32>(sphere.xyz, 1.0);
    r.front = clip.w - sphere.w;
    r.valid = clip.w + sphere.w > near;
    var lo = vec2<f32>(-1.0);
    var hi = vec2<f32>(1.0);
    let a = clip.xy - focal * sphere.w;
    let b = clip.xy + focal * sphere.w;
    let back = clip.w + sphere.w;
    if r.front > near {
        lo = max(min(a / r.front, a / back), lo);
        hi = min(max(b / r.front, b / back), hi);
    } else if r.valid {
        lo = select(lo, max(a / back, lo), a > vec2<f32>(0.0));
        hi = select(hi, min(b / back, hi), b < vec2<f32>(0.0));
    }
    r.valid = r.valid && all(hi >= lo);
    r.inside = max(max(r.front, near), max(max(a.x, -b.x), max(a.y, -b.y)));
    r.lo = vec2<f32>(lo.x, -hi.y) * 0.5 + 0.5;
    r.hi = vec2<f32>(hi.x, -lo.y) * 0.5 + 0.5;
    return r;
}
