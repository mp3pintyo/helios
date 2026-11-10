// G-buffer resolve of tessellated pixels (included by gbuffer.wgsl, whose bindings it uses;
// requires displacement.wgsl, displacement_height.wgsl and tess_core.wgsl).
//
// The visibility buffer names the base triangle (tess.wgsl), the merged depth where the pixel ray
// hit its displaced micro-triangles, and `tess_lambda` the point of the base triangle that was
// displaced to that hit (its barycentrics, interpolated over the micro-triangle by the rasteriser).
// The pixel is shaded with the base triangle's attributes at that point and the normal of the
// displaced surface:
//  * a height: the interpolated normal tilted by the surface gradient of the displacement (finite
//    differences of the height a pixel apart, the height filtered as the tessellator filters it);
//  * a vector: the cross product of the displaced surface's tangents (finite differences of the
//    displaced position a pixel apart), which also holds where the surface folds over itself.
// Either way the shading is smooth across micro-triangles. An animated displacement is evaluated
// again at the previous frame's time: the point's motion vector.
// Payload of the visibility buffer: unchanged (cluster slot << 7 | triangle).

@group(0) @binding(67) var<uniform> tess: Tessellation;
@group(0) @binding(92) var tess_lambda: texture_2d<u32>;

fn tess_fetch(slot: u32) -> vec2<u32> {
    return visible_clusters[slot].xy;
}

fn tess_pixels_per_unit(item: u32, world: vec3<f32>) -> f32 {
    return tess.dice.y / max(distance(world, tess.eye.xyz), tess.dice.z);
}

struct DisplacedHit {
    lambda: vec3<f32>,     // barycentrics on the base triangle
    object: vec3<f32>,     // displaced object-space position
    previous: vec3<f32>,   // ... at the previous frame's time (animated displacement)
    world: vec3<f32>,
    normal: vec3<f32>,     // world-space normal of the displaced surface
}

// The base triangle's point at barycentrics `l`.
fn displaced_input(instance: Instance, objects: array<vec3<f32>, 3>, normals: array<vec3<f32>, 3>, uvs: array<vec2<f32>, 3>, l: vec3<f32>, time: f32) -> DisplacementInput {
    let object = (objects[0] * l.x) + (objects[1] * l.y) + (objects[2] * l.z);
    let normal = normalize((normals[0] * l.x) + (normals[1] * l.y) + (normals[2] * l.z));
    return displacement_input(instance, object, normal, (uvs[0] * l.x) + (uvs[1] * l.y) + (uvs[2] * l.z), time);
}

// `objects`, `normals`, `uvs`: the base corners (object space, unit normals); `worlds`: where they
// were rasterised (wind included). `hit`: the pixel's world position from the merged depth.
fn resolve_displaced(pixel: vec2<u32>, instance: Instance, objects: array<vec3<f32>, 3>, normals: array<vec3<f32>, 3>, uvs: array<vec2<f32>, 3>, worlds: array<vec3<f32>, 3>, hit: vec3<f32>) -> DisplacedHit {
    let e1 = worlds[1] - worlds[0];
    let e2 = worlds[2] - worlds[0];
    let material = instance.material;
    let scale = instance.max_scale;
    let l = tess_unpack_lambda(textureLoad(tess_lambda, pixel, 0).r);
    let p = displaced_input(instance, objects, normals, uvs, l, tess.time.x);
    let value = displacement_value(material, scale, p);
    var h: DisplacedHit;
    h.lambda = l;
    h.object = p.object + displacement_offset(instance, p, value);
    h.previous = h.object;
    if DISPLACEMENT_ANIMATED {
        let before = displaced_input(instance, objects, normals, uvs, l, tess.time.y);
        h.previous = before.object + displacement_offset(instance, before, displacement_value(material, scale, before));
    }
    h.world = hit;
    // The displacement a pixel apart along both edges.
    let footprint = max(distance(hit, tess.eye.xyz), tess.dice.z) / tess.dice.y;
    let d1 = footprint / max(length(e1), 1e-6);
    let d2 = footprint / max(length(e2), 1e-6);
    let p1 = displaced_input(instance, objects, normals, uvs, l + vec3<f32>(-d1, d1, 0.0), tess.time.x);
    let p2 = displaced_input(instance, objects, normals, uvs, l + vec3<f32>(-d2, 0.0, d2), tess.time.x);
    let v1 = displacement_value(material, scale, p1);
    let v2 = displacement_value(material, scale, p2);
    let n = p.world_normal;
    let face = cross(e1, e2);
    if value.is_vector {
        // Tangents of the displaced surface: the steps along the edges, taken in the plane of the
        // interpolated normal (so an undisplaced point keeps its smooth normal), plus how the
        // offset changes over them.
        let linear = mat3x3<f32>(instance.world[0].xyz, instance.world[1].xyz, instance.world[2].xyz);
        let s1 = e1 * d1;
        let s2 = e2 * d2;
        let t1 = (s1 - (n * dot(n, s1))) + (linear * (displacement_offset(instance, p1, v1) - (h.object - p.object)));
        let t2 = (s2 - (n * dot(n, s2))) + (linear * (displacement_offset(instance, p2, v2) - (h.object - p.object)));
        let c = cross(t1, t2) * select(-1.0, 1.0, dot(face, n) >= 0.0);
        h.normal = select(n, normalize(c), dot(c, c) > 1e-30);
        return h;
    }
    let inv = 1.0 / max(dot(face, face), 1e-30);
    // World gradients of the barycentrics λ1, λ2 in the triangle's plane.
    let grad1 = cross(e2, face) * inv;
    let grad2 = cross(face, e1) * inv;
    let gradient = ((grad1 * ((v1.height - value.height) / d1)) + (grad2 * ((v2.height - value.height) / d2))) * (tess.materials[material].params.x * scale);
    h.normal = normalize(n - (gradient - (n * dot(gradient, n))));
    return h;
}

// Debug view 25: the patch and micro-triangle the pixel lies in, replaying the split decisions of
// the main view's tessellation. Hue: tessellation depth (splits, then the dice level: blue = one
// micro-triangle per patch ... red = deep), brightness varies per micro-triangle.
fn tessellation_debug_color(item: u32, instance: Instance, lambda: vec3<f32>) -> vec3<f32> {
    let b = tess_base(item);
    var corners = vec3<u32>(tess_pack(vec3<u32>(TESS_ONE, 0u, 0u)), tess_pack(vec3<u32>(0u, TESS_ONE, 0u)), tess_pack(vec3<u32>(0u, 0u, TESS_ONE)));
    var p = tess_patch(item, b, instance, corners);
    var depth = 0u;
    loop {
        let mask = tess_split_mask(p);
        if mask == 0u || f32(depth) >= tess.dice.w {
            break;
        }
        var children = tess_children(item, p.corners, mask);
        var best = 0u;
        var best_inside = -1.0e30;
        for (var c = 0u; c < children.count; c++) {
            let local = tess_local(children.records[c].yzw, lambda);
            let inside = min(min(local.x, local.y), 1.0 - local.x - local.y);
            if inside > best_inside {
                best_inside = inside;
                best = c;
            }
        }
        corners = children.records[best].yzw;
        p = tess_patch(item, b, instance, corners);
        depth++;
    }
    let level = tess_dice(p).level;
    let f = f32(1u << level);
    let local = tess_local(corners, lambda) * f;
    let cell = vec2<u32>(max(floor(local), vec2<f32>(0.0)));
    let up = fract(local.x) + fract(local.y) < 1.0;
    let shade = 0.45 + 0.55 * hash_color(pcg_hash(cell.x + (cell.y * 64u) + (select(0u, 4096u, up)) + (pcg_hash(corners.x ^ corners.y) * 31u))).x;
    let heat = saturate((f32(depth) + (f32(level) / 3.0)) / 6.0);
    let hue = mix(mix(vec3<f32>(0.15, 0.3, 1.0), vec3<f32>(0.2, 0.95, 0.3), saturate(heat * 2.0)), vec3<f32>(1.0, 0.2, 0.1), saturate((heat * 2.0) - 1.0));
    return hue * shade;
}

// Coordinates (u, v) of barycentrics `lambda` in the patch with packed corners: lambda = c0 +
// (c1 - c0) u + (c2 - c0) v.
fn tess_local(packed: vec3<u32>, lambda: vec3<f32>) -> vec2<f32> {
    let c0 = vec3<f32>(tess_unpack(packed.x)) / f32(TESS_ONE);
    let a = (vec3<f32>(tess_unpack(packed.y)) / f32(TESS_ONE)) - c0;
    let b = (vec3<f32>(tess_unpack(packed.z)) / f32(TESS_ONE)) - c0;
    let d = lambda - c0;
    let aa = dot(a, a);
    let ab = dot(a, b);
    let bb = dot(b, b);
    let det = max((aa * bb) - (ab * ab), 1e-20);
    return vec2<f32>((bb * dot(a, d)) - (ab * dot(b, d)), (aa * dot(b, d)) - (ab * dot(a, d))) / det;
}
