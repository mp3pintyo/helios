// GPU particles (particles.rs; a Niagara-lite): every emitter owns a fixed range of slots in one
// particle buffer.
//
//   simulate : per slot. Dead slots claim this frame's spawns of their emitter (an atomic
//              counter against the count the CPU computed from rate and bursts) and are born in
//              the emitter's shape with a velocity in its cone; live particles age, fall with
//              gravity, slow with drag, swirl with curl noise (a divergence-free velocity field
//              from the curl of a noise potential: turbulence without sinks or sources) and
//              collide with the depth buffer: behind the visible surface and within a
//              thickness of it, they are pushed back onto it and bounce off its G-buffer normal
//              with restitution and friction, or stick (embers settling on the ground).
//   light    : per live particle, once per frame (UE's per-particle translucent lighting rather
//              than per pixel): the colour and opacity over life, the sun through the virtual
//              shadow map and cloud shadows with a Henyey-Greenstein phase (motes in a light
//              shaft glow towards the sun), Lumen's radiance cache, the clustered local lights
//              with their shadow maps, blackbody or coloured emission, then the fog in front of
//              the particle: the premultiplied radiance each sprite fragment scales.
//   vs / fs  : camera-facing quads or quads stretched along the velocity (sparks), cut by the
//              opaque depth, faded near it (soft particles), into weighted blended
//              order-independent transparency targets (McGuire and Bavoil 2013), split at the
//              nearest translucent surface like the volumes (oit.wgsl composites them).

#include "common.wgsl"
#include "noise.wgsl"
#include "blackbody.wgsl"
#include "media_common.wgsl"
#include "avboit_common.wgsl"
#include "avboit_read.wgsl"
#include "avboit_write.wgsl"

const MAX_EMITTERS: u32 = 16u;
const SHAPE_SPHERE: f32 = 0.0;
const SHAPE_BOX: f32 = 1.0;
const COLLISION_BOUNCE: f32 = 1.0;
const COLLISION_STICK: f32 = 2.0;
const STATE_DEAD: u32 = 0u;
const STATE_ALIVE: u32 = 1u;
const STATE_STUCK: u32 = 2u;
// Accumulated colour is stored divided by this (rgba16float headroom for bright sparks).
const OIT_COLOR_SCALE: f32 = 16.0;
const DEBUG_VOLUMES: u32 = 21u;

struct Particle {
    position: vec3<f32>,
    age: f32,
    velocity: vec3<f32>,
    lifetime: f32,
    seed: u32,
    emitter: u32,
    state: u32,
    scale: f32,
}

// What the draw needs, from `light`.
struct Visual {
    color: vec4<f32>,   // premultiplied, fogged radiance and opacity
    shape: vec4<f32>,   // x diameter (m), y length along the velocity (m), z sprite, w soft distance (m)
}

struct ParticleEmitter {
    position: vec4<f32>,    // xyz spawn centre, w shape (0 sphere, 1 box, 2 disc)
    extent: vec4<f32>,      // xyz box half extent / x radius, w random size spread
    direction: vec4<f32>,   // xyz cone axis, w cos(cone half angle)
    motion: vec4<f32>,      // speed min, max, lifetime min, max
    gravity: vec4<f32>,     // xyz m/s², w drag (per second)
    turbulence: vec4<f32>,  // x acceleration (m/s²), y frequency (per m), z time scale, w collision mode
    collision: vec4<f32>,   // restitution, friction, radius (m), stick speed (m/s)
    size: vec4<f32>,        // diameter at birth, at death (m), z stretch (s), w render mode (0 billboard, 1 stretched)
    colors: array<vec4<f32>, 4>, // linear colour and opacity at age 0, 1/3, 2/3, 1
    emission: vec4<f32>,    // x intensity, y blackbody kelvin at birth, z at death (0: colour emission), w lit (0 / 1)
    look: vec4<f32>,        // x phase anisotropy, y soft distance (m), z sprite (0 glow, 1 puff), w pre-warm
    range: vec4<u32>,       // first slot, capacity, spawns this frame, seed
}

struct ParticleSystem {
    timing: vec4<f32>,      // x time step (s), y time (s)
    counts: vec4<u32>,      // x emitters, y slots
    emitters: array<ParticleEmitter, MAX_EMITTERS>,
}

@group(0) @binding(110) var<uniform> system: ParticleSystem;
@group(0) @binding(111) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(112) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(113) var<storage, read_write> spawned: array<atomic<u32>>;
@group(0) @binding(114) var<storage, read_write> visuals: array<Visual>;
@group(0) @binding(115) var<storage, read> visuals_in: array<Visual>;
@group(0) @binding(116) var scene_depth: texture_2d<f32>;
@group(0) @binding(117) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(118) var translucent_depth: texture_2d<f32>;

fn emitter_of(slot: u32) -> u32 {
    for (var e = 0u; e < min(system.counts.x, MAX_EMITTERS); e++) {
        let r = system.emitters[e].range;
        if slot >= r.x && slot < r.x + r.y {
            return e;
        }
    }
    return MAX_EMITTERS;
}

// Uniform random numbers in [0, 1) from a hash chain.
fn next_random(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state >> 8u) / 16777216.0;
}

fn unit_vector(u: f32, v: f32) -> vec3<f32> {
    let z = u * 2.0 - 1.0;
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = 2.0 * PI * v;
    return vec3<f32>(r * cos(phi), z, r * sin(phi));
}

fn spawn(slot: u32, e: u32) -> Particle {
    let em = system.emitters[e];
    var rng = pcg_hash(slot ^ pcg_hash(u32(system.timing.y * 1000.0) + em.range.w * 7919u));
    var p: Particle;
    let shape = em.position.w;
    var offset = vec3<f32>(0.0);
    if shape == SHAPE_SPHERE {
        offset = unit_vector(next_random(&rng), next_random(&rng)) * em.extent.x * pow(next_random(&rng), 1.0 / 3.0);
    } else if shape == SHAPE_BOX {
        offset = (vec3<f32>(next_random(&rng), next_random(&rng), next_random(&rng)) * 2.0 - 1.0) * em.extent.xyz;
    } else {
        let r = em.extent.x * sqrt(next_random(&rng));
        let a = 2.0 * PI * next_random(&rng);
        offset = vec3<f32>(r * cos(a), 0.0, r * sin(a));
    }
    // Direction uniformly inside the cone around the axis.
    let cos_theta = mix(1.0, em.direction.w, next_random(&rng));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = 2.0 * PI * next_random(&rng);
    let axis = normalize(em.direction.xyz);
    let cone_local = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let dir = basis(axis) * cone_local;
    p.position = em.position.xyz + offset;
    p.velocity = dir * mix(em.motion.x, em.motion.y, next_random(&rng));
    p.age = 0.0;
    p.lifetime = mix(em.motion.z, em.motion.w, next_random(&rng));
    p.seed = rng;
    p.emitter = e;
    p.state = STATE_ALIVE;
    p.scale = 1.0 + em.extent.w * (next_random(&rng) * 2.0 - 1.0);
    if em.look.w > 0.5 {
        // Pre-warm (the emitter's first frame): born earlier, at a random age, moved along a
        // ballistic path unless they collide (they would pass through the ground; they fall
        // from the source instead).
        p.age = p.lifetime * next_random(&rng);
        if em.turbulence.w < COLLISION_BOUNCE {
            p.position += p.velocity * p.age + 0.5 * em.gravity.xyz * p.age * p.age;
        }
    }
    return p;
}

fn noise_potential(q: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(gradient_noise(q), gradient_noise(q + vec3<f32>(31.4, 47.2, 11.8)), gradient_noise(q + vec3<f32>(-23.1, 17.9, 61.3)));
}

// Curl of the noise potential at `q` (Bridson et al. 2007): divergence-free turbulence.
fn curl_noise(q: vec3<f32>) -> vec3<f32> {
    let e = 0.05;
    let dx = noise_potential(q + vec3<f32>(e, 0.0, 0.0)) - noise_potential(q - vec3<f32>(e, 0.0, 0.0));
    let dy = noise_potential(q + vec3<f32>(0.0, e, 0.0)) - noise_potential(q - vec3<f32>(0.0, e, 0.0));
    let dz = noise_potential(q + vec3<f32>(0.0, 0.0, e)) - noise_potential(q - vec3<f32>(0.0, 0.0, e));
    return vec3<f32>(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x) / (2.0 * e);
}

// Pushes a particle that went behind the visible surface back onto it and bounces or sticks.
fn collide(p: ptr<function, Particle>, em: ParticleEmitter) {
    let clip = frame.view_proj * vec4<f32>((*p).position, 1.0);
    if clip.w <= frame.camera_position.w {
        return;
    }
    let ndc = clip.xy / clip.w;
    if any(abs(ndc) >= vec2<f32>(1.0)) {
        return;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let pixel = vec2<u32>(uv * frame.viewport.xy);
    let depth = textureLoad(scene_depth, pixel, 0).r;
    if depth <= 0.0 {
        return;
    }
    let surface_depth = frame.camera_position.w / depth;
    let radius = em.collision.z;
    let thickness = 0.3 + length((*p).velocity) * system.timing.x * 2.0;
    if clip.w < surface_depth - radius || clip.w > surface_depth + thickness {
        return;
    }
    let n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    let surface = world_from_depth(uv, depth, frame.inv_view_proj);
    (*p).position = surface + n * radius;
    let vn = dot((*p).velocity, n);
    if vn < 0.0 {
        let tangent = (*p).velocity - vn * n;
        (*p).velocity = tangent * (1.0 - em.collision.y) - vn * em.collision.x * n;
    }
    if em.turbulence.w == COLLISION_STICK || length((*p).velocity) < em.collision.w {
        (*p).velocity = vec3<f32>(0.0);
        (*p).state = STATE_STUCK;
    }
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if slot >= system.counts.y {
        return;
    }
    let e = emitter_of(slot);
    if e >= MAX_EMITTERS {
        return;
    }
    let em = system.emitters[e];
    var p = particles[slot];
    if p.state == STATE_DEAD {
        if atomicLoad(&spawned[e]) < em.range.z && atomicAdd(&spawned[e], 1u) < em.range.z {
            particles[slot] = spawn(slot, e);
        }
        return;
    }
    let dt = system.timing.x;
    p.age += dt;
    if p.age >= p.lifetime {
        p.state = STATE_DEAD;
        particles[slot] = p;
        return;
    }
    if p.state == STATE_ALIVE {
        var v = p.velocity + em.gravity.xyz * dt;
        if em.turbulence.x > 0.0 {
            let q = p.position * em.turbulence.y + vec3<f32>(0.0, system.timing.y * em.turbulence.z, f32(e) * 13.7);
            v += curl_noise(q) * em.turbulence.x * dt;
        }
        v *= exp(-em.gravity.w * dt);
        p.velocity = v;
        p.position += v * dt;
        if em.turbulence.w >= COLLISION_BOUNCE {
            collide(&p, em);
        }
    }
    particles[slot] = p;
}

// Colour and opacity over life: linear between four keys.
fn gradient(em: ParticleEmitter, t: f32) -> vec4<f32> {
    let x = saturate(t) * 3.0;
    let i = min(u32(x), 2u);
    return mix(em.colors[i], em.colors[i + 1u], x - f32(i));
}

@compute @workgroup_size(64)
fn light(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if slot >= system.counts.y {
        return;
    }
    let p = particles_in[slot];
    var out: Visual;
    out.color = vec4<f32>(0.0);
    out.shape = vec4<f32>(0.0);
    let clip = frame.view_proj * vec4<f32>(p.position, 1.0);
    if p.state == STATE_DEAD || clip.w <= frame.camera_position.w {
        visuals[slot] = out;
        return;
    }
    let em = system.emitters[p.emitter];
    let t = p.age / p.lifetime;
    let c = gradient(em, t);
    let eye = frame.camera_position.xyz;
    let view_dir = normalize(p.position - eye);
    let uv = saturate(vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5);
    let size = mix(em.size.x, em.size.y, t) * p.scale;

    var radiance = vec3<f32>(0.0);
    if em.emission.w > 0.5 {
        let g = em.look.x;
        let sun = normalize(frame.sun_direction.xyz);
        // Through glass panes nearer the sun, in their colour.
        let sun_light = media_sun_light(p.position) * media_sun_visibility(p.position, 0xffffffffu) * media_sun_glass(p.position) * henyey_greenstein(dot(sun, view_dir), g);
        let nearby = media_clustered_lights(p.position, uv, clip.w, view_dir, g, size * size);
        radiance = c.rgb * (sun_light + media_ambient(p.position, 1.0) + nearby);
    }
    if em.emission.x > 0.0 {
        if em.emission.y > 0.0 {
            radiance += blackbody(mix(em.emission.y, em.emission.z, t)) * em.emission.x;
        } else {
            radiance += c.rgb * em.emission.x;
        }
    }
    let f = media_fog(p.position, uv, view_dir, clip.w);
    out.color = vec4<f32>(c.a * (radiance * f.transmittance + f.inscatter), c.a);
    out.color = select(out.color, vec4<f32>(0.0), (out.color != out.color) | (abs(out.color) > vec4<f32>(6e4)));
    out.shape = vec4<f32>(size, size + length(p.velocity) * em.size.z, em.look.z, em.look.y);
    if em.size.w < 0.5 {
        out.shape.y = 0.0;
    }
    visuals[slot] = out;
}

// ---- Drawing --------------------------------------------------------------------------------

struct ParticleVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) shape: vec4<f32>,
}

// Weighted blended OIT: accumulation (premultiplied colour x weight, opacity x weight) and
// revealage (product of 1 - opacity), for the parts in front of and behind the nearest
// translucent surface.
struct OitOutput {
    @location(0) front_accum: vec4<f32>,
    @location(1) front_reveal: vec4<f32>,
    @location(2) back_accum: vec4<f32>,
    @location(3) back_reveal: vec4<f32>,
}

@vertex
fn vs_particle(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) slot: u32) -> ParticleVertex {
    var out: ParticleVertex;
    let v = visuals_in[slot];
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0); // outside the clip volume
    out.corner = vec2<f32>(0.0);
    out.color = v.color;
    out.shape = v.shape;
    if v.color.a <= 0.0 {
        return out;
    }
    let p = particles_in[slot];
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vertex_index % 6u];
    let to_eye = normalize(frame.camera_position.xyz - p.position);
    var right = vec3<f32>(frame.view[0].x, frame.view[1].x, frame.view[2].x);
    var up = vec3<f32>(frame.view[0].y, frame.view[1].y, frame.view[2].y);
    var half_size = vec2<f32>(0.5 * v.shape.x);
    // Sparks: a quad along the screen-space velocity, as long as the distance covered.
    let along = p.velocity - to_eye * dot(p.velocity, to_eye);
    if v.shape.y > 0.0 && length(along) > 1e-3 {
        up = normalize(along);
        right = normalize(cross(up, to_eye));
        half_size.y = 0.5 * v.shape.y;
    }
    let world = p.position + right * (corner.x * half_size.x) + up * (corner.y * half_size.y);
    out.position = frame.view_proj * vec4<f32>(world, 1.0);
    out.corner = corner;
    return out;
}

// Depth weight of a fragment (McGuire and Bavoil's eq. 9, rescaled to metres): nearer layers
// dominate where many overlap.
fn oit_weight(alpha: f32, linear_depth: f32) -> f32 {
    let d = linear_depth / 20.0;
    return alpha * clamp(4.0 / (1e-3 + d * d * d), 0.05, 40.0);
}

// The sprite's coverage at a fragment (`pixel`: the internal pixel it covers, for the opaque
// depth): x opacity, y the sprite's shape factor; opacity 0 where it is hidden or negligible.
fn sprite_coverage(in: ParticleVertex, pixel: vec2<i32>) -> vec2<f32> {
    let scene = textureLoad(scene_depth, pixel, 0).r;
    if in.position.z < scene {
        return vec2<f32>(0.0);
    }
    let r2 = dot(in.corner, in.corner);
    var shape = exp(-4.0 * r2) * (1.0 - smoothstep(0.7, 1.0, r2));
    if in.shape.z > 0.5 {
        // A soft puff with a little structure.
        let n = gradient_noise(vec3<f32>(in.corner * 1.7, in.shape.x * 13.0));
        shape = saturate((1.0 - r2) * (1.0 - r2) * (0.75 + 0.5 * n));
    }
    let linear_depth = frame.camera_position.w / in.position.z;
    if in.shape.w > 0.0 && scene > 0.0 {
        shape *= saturate((frame.camera_position.w / scene - linear_depth) / in.shape.w);
    }
    let alpha = in.color.a * shape;
    if alpha < 1.0 / 512.0 {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(alpha, shape);
}

@fragment
fn fs_particle(in: ParticleVertex) -> OitOutput {
    let pixel = vec2<i32>(in.position.xy);
    let coverage = sprite_coverage(in, pixel);
    if coverage.x <= 0.0 {
        discard;
    }
    let alpha = coverage.x;
    let shape = coverage.y;
    let linear_depth = frame.camera_position.w / in.position.z;
    var out: OitOutput;
    out.front_accum = vec4<f32>(0.0);
    out.front_reveal = vec4<f32>(0.0);
    out.back_accum = vec4<f32>(0.0);
    out.back_reveal = vec4<f32>(0.0);
    if frame.debug_view == DEBUG_VOLUMES {
        out.front_accum = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    let w = oit_weight(alpha, linear_depth);
    let accum = vec4<f32>(in.color.rgb * shape * w / OIT_COLOR_SCALE, alpha * w);
    let reveal = vec4<f32>(alpha, 0.0, 0.0, 0.0);
    if in.position.z > textureLoad(translucent_depth, pixel, 0).r {
        out.front_accum = accum;
        out.front_reveal = reveal;
    } else {
        out.back_accum = accum;
        out.back_reveal = reveal;
    }
    return out;
}

// ---- Adaptive voxel-based OIT (avboit.rs) -----------------------------------------------------

// Full resolution: the premultiplied colour weighted by the transmittance in front of the
// fragment (grid), added; the background's transmittance multiplied by 1 - opacity (alpha: the
// particles' own share, for TSR's reactive mask). Accumulated alpha counts fragments (debug
// view 21). Behind the nearest refracting surface, only what lies between it and the fragment
// weighs (`avboit_weight`).
@fragment
fn fs_particle_avboit(in: ParticleVertex) -> OitOutput {
    let pixel = vec2<i32>(in.position.xy);
    let coverage = sprite_coverage(in, pixel);
    if coverage.x <= 0.0 {
        discard;
    }
    let alpha = coverage.x;
    let linear_depth = frame.camera_position.w / in.position.z;
    var out: OitOutput;
    out.front_accum = vec4<f32>(0.0);
    out.front_reveal = vec4<f32>(1.0);
    out.back_accum = vec4<f32>(0.0);
    out.back_reveal = vec4<f32>(1.0);
    let split = textureLoad(translucent_depth, pixel, 0).r;
    let front = in.position.z > split;
    let split_depth = select(frame.camera_position.w / max(split, 1e-9), 0.0, front);
    let w = avboit_weight(in.position.xy, linear_depth, vec3<f32>(1.0 - alpha), split_depth);
    var accum = vec4<f32>(in.color.rgb * coverage.y * w, 1.0);
    if frame.debug_view == DEBUG_VOLUMES {
        accum = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let transmittance = vec4<f32>(1.0 - alpha);
    if front {
        out.front_accum = accum;
        out.front_reveal = transmittance;
    } else {
        out.back_accum = accum;
        out.back_reveal = transmittance;
    }
    return out;
}

// Low resolution, one fragment per grid column (the column's centre pixel tests the opaque
// depth): widen the column's bounds.
@fragment
fn fs_particle_bounds(in: ParticleVertex) {
    let column = vec2<u32>(in.position.xy);
    let pixel = vec2<i32>(column * avboit.grid.w + avboit.grid.w / 2u);
    if sprite_coverage(in, pixel).x > 0.0 {
        avboit_extend(avboit_column_index(column), frame.camera_position.w / in.position.z);
    }
}

// Low resolution: add the sprite's optical depth to its cell.
@fragment
fn fs_particle_inject(in: ParticleVertex) {
    let column = vec2<u32>(in.position.xy);
    let pixel = vec2<i32>(column * avboit.grid.w + avboit.grid.w / 2u);
    let alpha = sprite_coverage(in, pixel).x;
    if alpha > 0.0 {
        avboit_add(avboit_column_index(column), frame.camera_position.w / in.position.z, vec3<f32>(1.0 - alpha));
    }
}
