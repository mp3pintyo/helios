// Lumen near field: sphere tracing per-mesh distance fields for the first metres of a ray.
//
// The global distance field is 0.25 m per voxel at best: anything smaller vanishes from it or
// becomes a voxel-sized blob. As in UE5 Lumen, rays first trace the mesh SDFs of the instances
// around their origin - listed per cell of a camera-centred culling grid (near_field.rs) - and
// continue in the global field beyond lumen.mesh_sdf_trace.x. Each instance is traced in its
// own field space (atlas texels for volumes, cells for heightfields) with world-space ray
// distances, so the closest hit over the list is a plain minimum.
//
// Alpha-masked instances (foliage cards, chains, grilles) are partially transparent: their
// fields are built from the geometry, holes and all, so each carries its coverage - the solid
// fraction of its surface. As UE5 dithers the distance-field traces of two-sided foliage, a ray
// hits such an instance with a probability equal to its coverage (the gather's many rays and
// temporal filtering average that out), and soft shadows let 1 - coverage of the light through.
//
// Includers declare, besides lumen_common.wgsl's bindings: mesh_sdf_data: array<vec4<u32>>
// (instance records, heightfield heights, culling grid) and mesh_sdf_atlas: texture_3d<f32>
// (volume SDFs, r8snorm distance / band).

const MESH_SDF_RECORD_VEC4S: u32 = 8u;
// Volume fields are traced for soft shadows this many voxels beyond their box, where the
// penumbra fades out (PENUMBRA_MARGIN in near_field.rs).
const MESH_SDF_PENUMBRA_MARGIN: f32 = 2.0;
// Sphere-tracing steps per instance.
const MESH_SDF_STEPS: u32 = 24u;
// Hits are accepted this close to a surface (a quarter voxel of the mesh's field, at most):
// enough to catch features thinner than a voxel, small enough that huge coarse fields - a
// cathedral floor - don't produce hits well above their surface.
const MESH_SDF_MAX_HIT_DISTANCE: f32 = 0.05;

// An instance record is a world-space bounding sphere (xyz centre, w radius; penumbra margin
// included), then the MeshSdf below, then the instance's index in the mesh-card surface cache
// (lumen_cards.wgsl; x, 0xffffffff if it has no cards) and its coverage (y, f32 bits).
struct MeshSdf {
    // World -> field space, as matrix rows.
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    // Field-space box min; w = world distance per stored unit (volume: per distance / band,
    // heightfield: per cell).
    lo: vec4<f32>,
    // Field-space box max; w = world-space voxel size.
    hi: vec4<f32>,
    // Heightfields: first word of the heights, cells x | cells z << 16 (0 for volumes), height
    // range (f32 bits, cells).
    heightfield: vec4<u32>,
}

fn mesh_sdf_word(i: u32) -> u32 {
    return mesh_sdf_data[i >> 2u][i & 3u];
}

// Whether the ray stretch [t_min, t_max] passes through instance `index`'s bounding sphere:
// one load to reject most of a cell's list.
fn mesh_sdf_near_ray(index: u32, origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32) -> bool {
    let sphere = bitcast<vec4<f32>>(mesh_sdf_data[index * MESH_SDF_RECORD_VEC4S]);
    let c = sphere.xyz - origin;
    let t = clamp(dot(c, dir), t_min, t_max);
    let gap = c - dir * t;
    return dot(gap, gap) <= sphere.w * sphere.w;
}

// The solid fraction of instance `index`'s surface (1 unless alpha-masked).
fn mesh_sdf_coverage(index: u32) -> f32 {
    return bitcast<f32>(mesh_sdf_data[index * MESH_SDF_RECORD_VEC4S + 7u].y);
}

// Whether a ray hits partially transparent instance `index` at all (probability: its
// coverage). `ray` identifies the ray; mixing in the frame lets the outcome change every frame
// even for rays that don't (e.g. reflections of a static camera), so it averages over time.
fn mesh_sdf_ray_blocked(index: u32, ray: u32) -> bool {
    let coverage = mesh_sdf_coverage(index);
    if coverage >= 1.0 {
        return true;
    }
    let h = pcg_hash(ray ^ (index * 0x9e3779b9u) ^ (lumen.frame * 0x85ebca6bu));
    return f32(h >> 8u) * (1.0 / 16777216.0) < coverage;
}

// A hash of a ray, for stochastic transparency.
fn mesh_sdf_ray_id(origin: vec3<f32>, dir: vec3<f32>) -> u32 {
    let o = bitcast<vec3<u32>>(origin);
    let d = bitcast<vec3<u32>>(dir);
    return pcg_hash(o.x ^ pcg_hash(o.y ^ pcg_hash(o.z ^ pcg_hash(d.x ^ pcg_hash(d.y ^ pcg_hash(d.z))))));
}

fn mesh_sdf_load(index: u32) -> MeshSdf {
    let base = index * MESH_SDF_RECORD_VEC4S + 1u;
    var s: MeshSdf;
    s.row0 = bitcast<vec4<f32>>(mesh_sdf_data[base]);
    s.row1 = bitcast<vec4<f32>>(mesh_sdf_data[base + 1u]);
    s.row2 = bitcast<vec4<f32>>(mesh_sdf_data[base + 2u]);
    s.lo = bitcast<vec4<f32>>(mesh_sdf_data[base + 3u]);
    s.hi = bitcast<vec4<f32>>(mesh_sdf_data[base + 4u]);
    s.heightfield = mesh_sdf_data[base + 5u];
    return s;
}

fn mesh_sdf_point(s: MeshSdf, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(s.row0, h), dot(s.row1, h), dot(s.row2, h));
}

fn mesh_sdf_vector(s: MeshSdf, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(s.row0.xyz, v), dot(s.row1.xyz, v), dot(s.row2.xyz, v));
}

fn mesh_sdf_height(s: MeshSdf, x: u32, z: u32) -> f32 {
    let i = x + (s.heightfield.y & 0xffffu) * z;
    let h = (mesh_sdf_word(s.heightfield.x + (i >> 1u)) >> ((i & 1u) * 16u)) & 0xffffu;
    return mix(bitcast<f32>(s.heightfield.z), bitcast<f32>(s.heightfield.w), f32(h) / 65535.0);
}

// Field-space distance above a heightfield: the vertical gap scaled by the slope of the
// bilinear patch below.
fn mesh_sdf_heightfield_distance(s: MeshSdf, q: vec3<f32>) -> f32 {
    let res = vec2<u32>(s.heightfield.y & 0xffffu, s.heightfield.y >> 16u);
    let g = clamp(q.xz, vec2<f32>(0.0), vec2<f32>(res - 1u));
    let i = min(vec2<u32>(g), res - 2u);
    let f = g - vec2<f32>(i);
    let h00 = mesh_sdf_height(s, i.x, i.y);
    let h10 = mesh_sdf_height(s, i.x + 1u, i.y);
    let h01 = mesh_sdf_height(s, i.x, i.y + 1u);
    let h11 = mesh_sdf_height(s, i.x + 1u, i.y + 1u);
    let h = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
    let slope = vec2<f32>(mix(h10 - h00, h11 - h01, f.y), mix(h01 - h00, h11 - h10, f.x));
    return (q.y - h) * inverseSqrt(1.0 + dot(slope, slope));
}

// World-space distance (a lower bound under non-uniform scale) at field-space point `q`.
fn mesh_sdf_distance(s: MeshSdf, q: vec3<f32>, inv_atlas: vec3<f32>) -> f32 {
    if s.heightfield.y != 0u {
        return mesh_sdf_heightfield_distance(s, q) * s.lo.w;
    }
    // Clamped to the instance's own voxels (neighbours in the atlas belong to other meshes);
    // outside them the distance to the voxel box bounds the distance from below.
    let texel = clamp(q, s.lo.xyz, s.hi.xyz);
    let inside = textureSampleLevel(mesh_sdf_atlas, lumen_sampler, texel * inv_atlas, 0.0).r * s.lo.w;
    return max(inside, length(q - texel) * s.hi.w);
}

// Distances saturate at the band of a volume field (heightfields are exact everywhere).
fn mesh_sdf_band(s: MeshSdf) -> f32 {
    return select(s.lo.w, 1e9, s.heightfield.y != 0u);
}

fn mesh_sdf_hit_distance(s: MeshSdf) -> f32 {
    return min(s.hi.w * 0.25, MESH_SDF_MAX_HIT_DISTANCE);
}

// The stretch [t_min, t_max] of the ray (field-space origin `qo`, direction `qd`, world
// distances) inside the instance's field box grown by `margin` field units; empty when x > y.
fn mesh_sdf_span(s: MeshSdf, qo: vec3<f32>, qd: vec3<f32>, t_min: f32, t_max: f32, margin: f32) -> vec2<f32> {
    let inv = 1.0 / select(qd, vec3<f32>(1e-12), abs(qd) < vec3<f32>(1e-12));
    let a = (s.lo.xyz - margin - qo) * inv;
    let b = (s.hi.xyz + margin - qo) * inv;
    let near = min(a, b);
    let far = max(a, b);
    return vec2<f32>(max(max(near.x, near.y), max(near.z, t_min)), min(min(far.x, far.y), min(far.z, t_max)));
}

// The list of instances the grid cell containing `p` holds: (first word, count); empty outside
// the grid.
fn mesh_sdf_cell(p: vec3<f32>) -> vec2<u32> {
    let grid = lumen.mesh_sdf_grid;
    if grid.w <= 0.0 {
        return vec2<u32>(0u);
    }
    let g = (p - grid.xyz) / grid.w;
    let dims = lumen.mesh_sdf_dims.xyz;
    if any(g < vec3<f32>(0.0)) || any(g >= vec3<f32>(dims)) {
        return vec2<u32>(0u);
    }
    let c = vec3<u32>(g);
    let table = lumen.mesh_sdf_dims.w * 4u;
    let word = mesh_sdf_word(table + c.x + dims.x * (c.y + dims.y * c.z));
    return vec2<u32>(table + (word >> 8u), word & 0xffu);
}

// Whether the near field covers rays starting at `p`.
fn mesh_sdf_covers(p: vec3<f32>) -> bool {
    let grid = lumen.mesh_sdf_grid;
    let g = (p - grid.xyz) / max(grid.w, 1e-6);
    return grid.w > 0.0 && all(g >= vec3<f32>(0.0)) && all(g < vec3<f32>(lumen.mesh_sdf_dims.xyz));
}

fn mesh_sdf_inv_atlas() -> vec3<f32> {
    return 1.0 / vec3<f32>(textureDimensions(mesh_sdf_atlas));
}

// Sphere-traces one instance over `span`; returns the hit distance or -1. A ray leaving a
// surface (`from_surface`) first has to get clear of that surface's own field, which a coarse
// mesh SDF may place a little off the true surface - otherwise rays would hit where they start.
fn mesh_sdf_trace_instance(s: MeshSdf, qo: vec3<f32>, qd: vec3<f32>, span: vec2<f32>, from_surface: bool, inv_atlas: vec3<f32>) -> f32 {
    let threshold = mesh_sdf_hit_distance(s);
    var clear = !from_surface;
    var t = span.x;
    for (var i = 0u; i < MESH_SDF_STEPS; i++) {
        if t > span.y {
            break;
        }
        let d = mesh_sdf_distance(s, qo + qd * t, inv_atlas);
        if d < threshold {
            if clear {
                return t;
            }
        } else {
            clear = true;
        }
        t += max(d, threshold);
    }
    return -1.0;
}

// World-space surface normal of an instance's field at field-space point `q` (tetrahedral
// gradient, transformed by the transpose of the world -> field matrix).
fn mesh_sdf_normal(s: MeshSdf, q: vec3<f32>, inv_atlas: vec3<f32>) -> vec3<f32> {
    let k = vec2<f32>(0.5, -0.5);
    let g = k.xyy * mesh_sdf_distance(s, q + k.xyy, inv_atlas) + k.yyx * mesh_sdf_distance(s, q + k.yyx, inv_atlas)
        + k.yxy * mesh_sdf_distance(s, q + k.yxy, inv_atlas) + k.xxx * mesh_sdf_distance(s, q + k.xxx, inv_atlas);
    let n = s.row0.xyz * g.x + s.row1.xyz * g.y + s.row2.xyz * g.z;
    return normalize(n + vec3<f32>(0.0, 1e-7, 0.0));
}

struct MeshSdfHit {
    hit: bool,
    t: f32,
    // World-space normal, facing the ray.
    normal: vec3<f32>,
    // Surface-cache (card) instance of the surface hit.
    instance: u32,
}

// Closest hit within [t_min, t_max] among the mesh SDFs near `origin` (rays leaving a surface).
fn trace_mesh_sdfs(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32) -> MeshSdfHit {
    var h: MeshSdfHit;
    h.hit = false;
    h.t = t_max;
    let cell = mesh_sdf_cell(origin);
    let inv_atlas = mesh_sdf_inv_atlas();
    let ray = mesh_sdf_ray_id(origin, dir);
    var closest = 0u;
    for (var k = 0u; k < cell.y; k++) {
        let index = mesh_sdf_word(cell.x + k);
        if !mesh_sdf_near_ray(index, origin, dir, t_min, h.t) || !mesh_sdf_ray_blocked(index, ray) {
            continue;
        }
        let s = mesh_sdf_load(index);
        let qo = mesh_sdf_point(s, origin);
        let qd = mesh_sdf_vector(s, dir);
        let span = mesh_sdf_span(s, qo, qd, t_min, h.t, 0.0);
        if span.x > span.y {
            continue;
        }
        let t = mesh_sdf_trace_instance(s, qo, qd, span, span.x <= t_min, inv_atlas);
        if t >= 0.0 && t < h.t {
            h.hit = true;
            h.t = t;
            closest = index;
        }
    }
    if h.hit {
        let s = mesh_sdf_load(closest);
        let n = mesh_sdf_normal(s, mesh_sdf_point(s, origin + dir * h.t), inv_atlas);
        h.normal = select(n, -n, dot(n, dir) > 0.0);
        h.instance = mesh_sdf_data[closest * MESH_SDF_RECORD_VEC4S + 7u].x;
    }
    return h;
}

// Penumbra visibility of one instance over `span` (as sdf_soft_shadow_span). The penumbra
// cannot be wider than the field's band: beyond it every distance reads as the band, and a
// wider cone would darken the whole voxel box. A receiver does not shadow itself within two
// voxels of its own field: the field's interpolation and quantisation ripples there would show
// as bands on curved surfaces, which N.L shades anyway.
fn mesh_sdf_shadow_instance(s: MeshSdf, qo: vec3<f32>, qd: vec3<f32>, span: vec2<f32>, softness: f32, from_surface: bool, inv_atlas: vec3<f32>) -> f32 {
    let threshold = mesh_sdf_hit_distance(s);
    let band = mesh_sdf_band(s);
    var visibility = 1.0;
    var t = select(span.x, span.x + 2.0 * s.hi.w, from_surface);
    for (var i = 0u; i < MESH_SDF_STEPS; i++) {
        if t > span.y {
            break;
        }
        let d = mesh_sdf_distance(s, qo + qd * t, inv_atlas);
        visibility = min(visibility, saturate(d / clamp(softness * t, 1e-4, band) + 0.15));
        if visibility <= 0.01 {
            return 0.0;
        }
        t += max(d, threshold);
    }
    return visibility;
}

// Soft-shadow visibility over [t_min, t_max] through the mesh SDFs listed in the grid cell of
// `cell_point` (the ray origin, or the light for the stretch next to it).
fn mesh_sdf_soft_shadow(cell_point: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32, softness: f32, from_surface: bool) -> f32 {
    let cell = mesh_sdf_cell(cell_point);
    let inv_atlas = mesh_sdf_inv_atlas();
    var visibility = 1.0;
    for (var k = 0u; k < cell.y; k++) {
        let index = mesh_sdf_word(cell.x + k);
        if !mesh_sdf_near_ray(index, origin, dir, t_min, t_max) {
            continue;
        }
        let s = mesh_sdf_load(index);
        let qo = mesh_sdf_point(s, origin);
        let qd = mesh_sdf_vector(s, dir);
        // Grown by two voxels, so the penumbra fades out over the band around the box.
        let span = mesh_sdf_span(s, qo, qd, t_min, t_max, select(MESH_SDF_PENUMBRA_MARGIN, 0.0, s.heightfield.y != 0u));
        if span.x > span.y {
            continue;
        }
        let occlusion = 1.0 - mesh_sdf_shadow_instance(s, qo, qd, span, softness, from_surface && span.x <= t_min, inv_atlas);
        visibility = min(visibility, 1.0 - occlusion * mesh_sdf_coverage(index));
        if visibility <= 0.01 {
            return 0.0;
        }
    }
    return visibility;
}

// Distance-field soft shadow from a surface point towards a local light. Mesh SDFs cover the
// near-field distance at both ends - the receiver's surroundings and the light's own fixture,
// which the global field would blur into a blob around the light - and the global field the
// stretch between, if any. Where the near field doesn't reach, the global field alone is traced,
// stopping short of the light (light_shadow_distance).
//
// The penumbra cone is the light's angular radius as seen from the receiver; the blurrier global
// field gets a three times wider one.
fn local_light_shadow(origin: vec3<f32>, light_position: vec3<f32>, light_radius: f32) -> f32 {
    let to_light = light_position - origin;
    let distance = length(to_light);
    let dir = to_light / max(distance, 1e-6);
    let cone = max(light_radius / max(distance, 0.1), 0.02);
    if !mesh_sdf_covers(origin) || !mesh_sdf_covers(light_position) {
        return sdf_soft_shadow(origin, dir, light_shadow_distance(light_position, distance, light_radius), cone * 3.0);
    }
    let near = lumen.mesh_sdf_trace.x;
    let end = max(distance - light_radius, 0.0);
    var visibility = mesh_sdf_soft_shadow(origin, origin, dir, 0.0, min(end, near), cone, true);
    if distance > 2.0 * near && visibility > 0.01 {
        visibility = min(visibility, sdf_soft_shadow_span(origin, dir, near, distance - near, cone * 3.0));
    }
    if end > near && visibility > 0.01 {
        visibility = min(visibility, mesh_sdf_soft_shadow(light_position, origin, dir, max(near, distance - near), end, cone, false));
    }
    return visibility;
}

// Radiance leaving a near-field hit: the surface cache just off the surface. Objects too small
// for the global field have little surface-cache coverage; there the radiance is estimated
// from the radiance cache and the albedo the global field stores nearby.
fn near_field_radiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let level = sdf_finest_level(p);
    if level >= lumen.level_count {
        return vec3<f32>(0.0);
    }
    let lighting_voxel = lumen.sdf_levels[level].w * SDF_RES / LIGHT_RES;
    let cached = textureSampleLevel(voxel_lighting, lumen_sampler, sdf_uvw(p + n * lighting_voxel * 0.5, level, LIGHT_RES), 0.0);
    let albedo = abs(sdf_albedo(p, level));
    let estimate = albedo * radiance_cache_irradiance(p + n * 0.1, n, vec3<f32>(0.0)) / PI;
    return mix(estimate, cached.rgb / max(cached.a, 0.02), saturate(cached.a * 4.0));
}
