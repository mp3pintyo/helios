// Shared types and helpers. Every struct here mirrors a `#[repr(C)]` Rust struct in
// `gpu_types.rs`; keep them in sync.

const PI: f32 = 3.14159265359;
const INVALID_ID: u32 = 0xffffffffu;
const TRIANGLE_BITS: u32 = 7u;
const TRIANGLE_MASK: u32 = 0x7fu;
const MAX_CLUSTER_TRIANGLES: u32 = 128u;
const ROOT_ERROR: f32 = 3.0e38;

// Debug views (RenderSettings::debug_view).
const DEBUG_NONE: u32 = 0u;
const DEBUG_CLUSTERS: u32 = 1u;
const DEBUG_TRIANGLES: u32 = 2u;
const DEBUG_LOD: u32 = 3u;
const DEBUG_INSTANCES: u32 = 4u;
const DEBUG_NORMALS: u32 = 5u;
const DEBUG_ALBEDO: u32 = 6u;
const DEBUG_INDIRECT: u32 = 7u;
const DEBUG_SHADOW_PAGES: u32 = 8u;
const DEBUG_RASTER_PATH: u32 = 9u;
const DEBUG_MATERIALS: u32 = 10u;
const DEBUG_SURFACE_CACHE: u32 = 11u;
const DEBUG_SDF: u32 = 12u;
const DEBUG_REFLECTIONS: u32 = 13u;
const DEBUG_SURFACE_CARDS: u32 = 14u;
const DEBUG_LOCAL_SHADOW_PAGES: u32 = 15u;
const DEBUG_INVALIDATION: u32 = 16u;
const DEBUG_SKIN_WEIGHTS: u32 = 17u;
const DEBUG_MEGALIGHTS_SAMPLES: u32 = 18u;
const DEBUG_MEGALIGHTS_DENOISING: u32 = 19u;
const DEBUG_VOXELS: u32 = 22u;
const DEBUG_ASSEMBLY_PARTS: u32 = 23u;
const DEBUG_TESSELLATION: u32 = 25u;

/// Debug views that show lit (exposed, tonemapped) radiance rather than raw attributes.
fn is_lit_view(view: u32) -> bool {
    return view == DEBUG_NONE || view == DEBUG_INDIRECT || view == DEBUG_REFLECTIONS || view == DEBUG_MEGALIGHTS_DENOISING;
}

// Frame flags.
const FLAG_SHADOWS: u32 = 1u;
const FLAG_GI: u32 = 2u;
const FLAG_TAA: u32 = 4u;
const FLAG_AUTO_EXPOSURE: u32 = 8u;
const FLAG_FOG: u32 = 16u;
const FLAG_HISTORY_VALID: u32 = 32u;
const FLAG_UNDERWATER: u32 = 64u;
// Local lights are shaded by MegaLights (megalights.rs) instead of the clustered loop.
const FLAG_MEGALIGHTS: u32 = 128u;
// gbuffer_motion.z holds TSR's reactive mask (oit.wgsl; otherwise emission, consumed by then).
const FLAG_REACTIVE: u32 = 1024u;

const INVALID_PAGE: u32 = 0xffffffffu;
const INSTANCE_CAST_SHADOW: u32 = 1u;
const INSTANCE_ALPHA_MASKED: u32 = 2u;
const INSTANCE_TRANSLUCENT: u32 = 4u;   // drawn by the forward translucency pass only
const INSTANCE_HIDDEN: u32 = 8u;        // hidden in an editor: culled for every view
const INSTANCE_MOVABLE: u32 = 16u;      // Mobility::Movable: a dynamic shadow caster
const INSTANCE_ASSEMBLY: u32 = 32u;     // an assembly's base (assembly.wgsl)
const INSTANCE_EXPANDED: u32 = 64u;     // the assembly is drawn as its parts this frame
const INSTANCE_PART: u32 = 128u;        // an assembly part written by the expansion
const INSTANCE_DISPLACED: u32 = 256u;   // its material displaces it: Nanite tessellation (tess.wgsl)
// Bits 12..32: the skin palette index (INSTANCE_SKIN_SHIFT, skin_palette.wgsl).

// Visible cluster entry flags (entry.z).
const CLUSTER_SOFTWARE: u32 = 1u;
// A main-view hardware cluster of an alpha-masked material: drawn by the masked pipeline.
const CLUSTER_MASKED: u32 = 2u;
// A voxel cluster (always with CLUSTER_SOFTWARE: voxels are rasterised in compute).
const CLUSTER_VOXEL: u32 = 4u;

struct Cluster {
    bounds: vec4<f32>,
    lod_bounds: vec4<f32>,
    parent_lod_bounds: vec4<f32>,
    lod_error: f32,
    parent_error: f32,
    page: u32,
    child_page: u32,
    vertex_offset: u32,      // compressed cluster: header + vertex bit stream (geometry.wgsl)
    triangle_offset: u32,    // triangle bit stream (voxel clusters: voxel attributes)
    counts: u32,        // vertex_count | triangle_count << 8 | lod_level << 16
    // Voxel clusters (voxel.wgsl): bricks | voxels << 8. Triangle clusters: 0, or
    // CLUSTER_VOXEL_ALTERNATE where voxel clusters replace them with voxel LODs on.
    voxel: u32,
}

const CLUSTER_VOXEL_ALTERNATE: u32 = 0x80000000u;

fn cluster_vertex_count(c: Cluster) -> u32 { return c.counts & 0xffu; }
fn cluster_triangle_count(c: Cluster) -> u32 { return (c.counts >> 8u) & 0xffu; }
fn cluster_lod_level(c: Cluster) -> u32 { return c.counts >> 16u; }
fn cluster_voxel_bricks(c: Cluster) -> u32 { return c.voxel & 0xffu; }
fn cluster_voxel_count(c: Cluster) -> u32 { return (c.voxel >> 8u) & 0x3fffu; }
fn cluster_is_voxel(c: Cluster) -> bool { return cluster_voxel_bricks(c) != 0u; }

// Whether a cluster takes part in the LOD cut with voxel LODs on or off: voxel clusters and the
// triangle clusters they stand in for are alternatives.
fn cluster_in_mode(c: Cluster, voxels: bool) -> bool {
    return select(!cluster_is_voxel(c), (c.voxel & CLUSTER_VOXEL_ALTERNATE) == 0u, voxels);
}

// Triangle clusters of displaced instances are tessellated and rasterised by tess.wgsl and
// vsm_tess.wgsl (voxel clusters are not displaced); the other rasterisers skip them.
fn is_tessellated(instance_flags: u32, c: Cluster) -> bool {
    return (instance_flags & INSTANCE_DISPLACED) != 0u && !cluster_is_voxel(c);
}

struct Page {
    bounds: vec4<f32>,
    parent_lod_bounds: vec4<f32>,
    max_parent_error: f32,
    cluster_offset: u32,
    cluster_count: u32,
    lod_level: u32,
    data_words: u32,
    root: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Instance {
    world: mat4x4<f32>,
    prev_world: mat4x4<f32>,
    // Columns of the scaled inverse transpose of world; w: the plant origin of the wind
    // (wind.wgsl), the instance's position or an assembly part's assembly's.
    normal_matrix_0: vec4<f32>,
    normal_matrix_1: vec4<f32>,
    normal_matrix_2: vec4<f32>,
    bounds: vec4<f32>,
    page_offset: u32,
    page_count: u32,
    cluster_offset: u32,
    cluster_count: u32,
    material: u32,
    visibility_offset: u32,
    max_scale: f32,
    flags: u32,
    wind: vec4<f32>,  // wind response (wind.wgsl): bend, sway, flutter, frequency; 0 = static
}

// Mesh UV set -> texture coordinates (glTF texCoord, KHR_texture_transform, sampler wrap):
// uv' = (dot(u, (uv, 1)), dot(v, (uv, 1))) of set `uv_set` (0 or 1), then wrapped per axis
// (wrap = u mode | v mode << 4: 0 repeat, 1 mirrored repeat, 2 clamp to edge).
struct UvTransform {
    u: vec3<f32>,
    uv_set: u32,
    v: vec3<f32>,
    wrap: u32,
}

struct Material {
    base_color: vec4<f32>,
    emissive: vec4<f32>,  // xyz: emitted colour, w: emission as a multiple of the base colour
    roughness: f32,
    metallic: f32,
    kind: u32,
    flags: u32,           // MATERIAL_FLAG_*
    texture: u32,
    uv_scale: f32,
    normal_strength: f32,
    mapping: u32,
    detail_scale: f32,
    detail_strength: f32,
    layers: u32,
    emissive_texture: u32,
    // Texture coordinates of texture (tiling included), emissive_texture and occlusion_texture.
    // Named fields rather than an array: an array member can keep shader compilers from
    // splitting the material copy into registers.
    uv_texture: UvTransform,
    uv_emissive: UvTransform,
    uv_occlusion: UvTransform,
    occlusion_texture: u32,     // layer whose albedo alpha is ambient occlusion
    occlusion_strength: f32,
    _pad: vec2<u32>,
}

// The transform's UV set out of a point's two (meshes without a second set repeat the first).
fn uv_set_of(t: UvTransform, uv0: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
    return select(uv0, uv1, t.uv_set == 1u);
}

fn uv_transform_point(t: UvTransform, uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(dot(t.u, vec3<f32>(uv, 1.0)), dot(t.v, vec3<f32>(uv, 1.0)));
}

// Derivatives and other differences transform without the offset.
fn uv_transform_vector(t: UvTransform, d: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(dot(t.u.xy, d), dot(t.v.xy, d));
}

// Clamped coordinates stay half a texel of the 512² array layer inside the edges: textures are
// sampled with repeat addressing, whose filter would blend in the opposite edge.
const UV_CLAMP_MARGIN: f32 = 0.5 / 512.0;

fn wrap_axis(x: f32, mode: u32) -> f32 {
    switch mode {
        case 1u: {
            return 1.0 - abs((fract(x * 0.5) * 2.0) - 1.0);
        }
        case 2u: {
            return clamp(x, UV_CLAMP_MARGIN, 1.0 - UV_CLAMP_MARGIN);
        }
        default: {
            return x;
        }
    }
}

// Applies the sampler's wrap modes to interpolated coordinates (per pixel: wrapping does not
// commute with interpolation).
fn uv_wrap(t: UvTransform, uv: vec2<f32>) -> vec2<f32> {
    if t.wrap == 0u {
        return uv;
    }
    return vec2<f32>(wrap_axis(uv.x, t.wrap & 15u), wrap_axis(uv.y, t.wrap >> 4u));
}

const MAX_MATERIALS: u32 = 256u;
// The material's texture is made from images: data.w holds metallic (procedural textures store
// height there) and texture values multiply the factors (glTF metallic-roughness semantics).
const MATERIAL_FLAG_IMAGE_TEXTURE: u32 = 1u;
// The ambient occlusion is the alpha of the material's own texture sample.
const MATERIAL_FLAG_OCCLUSION_IN_TEXTURE: u32 = 2u;

// G-buffer motion.w: the shading flags (SURFACE_* in material.wgsl) in the integer part and the
// material's ambient occlusion in the fraction (0: unoccluded), which only indirect lighting
// reads. rgba16float keeps ~1/1000 of precision there for flags below 2.
const SURFACE_OCCLUSION_SCALE: f32 = 0.75;

fn pack_surface_flags(flags: f32, occlusion: f32) -> f32 {
    return flags + (1.0 - saturate(occlusion)) * SURFACE_OCCLUSION_SCALE;
}

fn surface_flags_of(packed: f32) -> u32 {
    return u32(packed);
}

fn surface_occlusion_of(packed: f32) -> f32 {
    return 1.0 - fract(packed) / SURFACE_OCCLUSION_SCALE;
}
// Alpha-tested materials (alpha_mask.wgsl): flags bits 8..16 hold the alpha cutoff (unorm8),
// bits 16..32 the alpha mask layer + 1 (0: opaque).
const MATERIAL_ALPHA_CUTOFF_SHIFT: u32 = 8u;
const MATERIAL_ALPHA_MASK_SHIFT: u32 = 16u;

// A culling view: the main camera or one shadow cascade.
struct CullView {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    planes: array<vec4<f32>, 6>,
    position: vec4<f32>,     // xyz: eye, w: near plane
    viewport: vec4<f32>,     // width, height, 1/width, 1/height
    lod_scale: f32,
    lod_threshold: f32,
    orthographic: u32,
    instance_count: u32,
    work_capacity: u32,
    cluster_capacity: u32,
    hzb_mips: u32,
    frame_index: u32,
    sw_raster_threshold: f32,
    kind: u32,               // 0 = main view, 1 = virtual shadow map level
    batch_capacity: u32,
    masked_capacity: u32,    // alpha-masked hardware draw list (main view only)
    vsm: vec4<f32>,          // level, page world size, absolute origin page (x, y)
    flags: u32,              // VIEW_*
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// CullView.flags: voxel LODs are on (voxel clusters replace their triangle alternates).
const VIEW_VOXELS: u32 = 1u;

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    unjittered_view_proj: mat4x4<f32>,
    prev_unjittered_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec4<f32>,  // xyz, w: near
    viewport: vec4<f32>,
    jitter: vec4<f32>,
    sun_direction: vec4<f32>,    // towards the sun, w: angular radius
    sun_illuminance: vec4<f32>,  // rgb illuminance outside the atmosphere
    ground_albedo: vec4<f32>,
    frame_index: u32,
    debug_view: u32,
    flags: u32,
    light_count: u32,
    exposure_compensation: f32,
    gi_intensity: f32,
    bloom_intensity: f32,
    time: f32,
    misc: vec4<f32>,              // x: delta time, y: aerial perspective scale, z: gi ray length, w: star radiance
    prev_viewport: vec4<f32>,     // the previous frame's internal viewport (dynamic resolution)
    upscale: vec4<f32>,           // x, y: internal / output pixels (TSR), z, w: output size
}

// ---- Internal resolution -------------------------------------------------------------------------
// The scene renders at an internal resolution (`frame.viewport`, a screen percentage of the
// output) into targets allocated for the largest percentage: a frame occupies their top-left
// corner, the previous one `frame.prev_viewport` (dynamic resolution changes it without
// reallocating). Pixel loads need nothing special; screen uvs (0..1 across the view) must be
// mapped before sampling such a target.

// Texels a `viewport` (frame.viewport or frame.prev_viewport) covers in a target downscaled by
// `divisor` (1 = full internal resolution).
fn viewport_extent(viewport: vec4<f32>, divisor: u32) -> vec2<u32> {
    return (vec2<u32>(viewport.xy) + divisor - 1u) / divisor;
}

// Texture coordinates of screen `uv` in a texture of size `dims` whose top-left texels show the
// view at `texels_per_uv` texels across it (e.g. `frame.viewport.xy * 0.5` for a half-resolution
// target), clamped half a texel inside the covered region so filtering never reads past it.
fn viewport_uv(uv: vec2<f32>, texels_per_uv: vec2<f32>, dims: vec2<u32>) -> vec2<f32> {
    let texel = clamp(uv * texels_per_uv, vec2<f32>(0.5), max(ceil(texels_per_uv) - 0.5, vec2<f32>(0.5)));
    return texel / vec2<f32>(dims);
}

// Software-rasterised clusters in a visible cluster list (cull state words: hardware and
// software counts, the early pass's software count) as cull_args.wgsl clamps them: the early
// pass's, or all of them for the late pass.
fn software_cluster_count(late: bool, hardware: u32, software: u32, early_software: u32, capacity: u32) -> u32 {
    return select(early_software, min(software, capacity - min(hardware, capacity)), late);
}

fn has_flag(flags: u32, flag: u32) -> bool {
    return (flags & flag) != 0u;
}

fn transform_sphere(world: mat4x4<f32>, sphere: vec4<f32>, max_scale: f32) -> vec4<f32> {
    return vec4<f32>((world * vec4<f32>(sphere.xyz, 1.0)).xyz, sphere.w * max_scale);
}

fn pcg_hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_color(id: u32) -> vec3<f32> {
    let h = pcg_hash(id);
    return vec3<f32>(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0 * 0.8 + 0.2;
}

// Jimenez's interleaved gradient noise, animated per frame.
fn interleaved_gradient_noise(pixel: vec2<f32>, frame: u32) -> f32 {
    let p = pixel + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Builds an orthonormal basis around `n` (Duff et al. 2017).
fn basis(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

fn cosine_sample_hemisphere(u: vec2<f32>) -> vec3<f32> {
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    return vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

// Reconstructs a world position from a reverse-Z depth value and pixel uv.
fn world_from_depth(uv: vec2<f32>, depth: f32, inv_view_proj: mat4x4<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let p = inv_view_proj * ndc;
    return p.xyz / p.w;
}

// View direction through a pixel (valid for sky pixels where depth = 0).
fn view_direction(uv: vec2<f32>, inv_view_proj: mat4x4<f32>, eye: vec3<f32>) -> vec3<f32> {
    let p = world_from_depth(uv, 1.0, inv_view_proj);
    return normalize(p - eye);
}
