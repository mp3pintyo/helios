// Shallow-water river (river.rs): virtual-pipes simulation steps, the surface texture for the
// shores and the surface mesh's vertices.

struct RiverParams {
    origin: vec2<f32>,
    cell: f32,
    dt: f32,
    size: vec2<u32>,
    first_vertex: u32,
    vertex_count: u32,
    gravity: f32,
    manning: f32,
    outlet_level: f32,
    source_count: u32,
    // xy: position (x, z), z: radius, w: inflow (m^3/s).
    sources: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> river: RiverParams;
// x: bed height, y: 1 for outlet cells (held at the outlet level), z: where the surface hides
// while dry (under the bed, or inside the rock of a wall).
@group(0) @binding(1) var<storage, read> terrain: array<vec4<f32>>;
// Outflow through the pipes to the -x, +x, -z and +z neighbours (m^3/s).
@group(0) @binding(2) var<storage, read_write> flux: array<vec4<f32>>;
// x: depth (m), yz: velocity (m/s, x and z), w: foam (0-1).
@group(0) @binding(3) var<storage, read> water_in: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> water_out: array<vec4<f32>>;
// The translucency pass's vertex buffer (TranslucentVertex: 14 floats).
@group(0) @binding(5) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(6) var surface_out: texture_storage_2d<r32float, write>;

const VERTEX_FLOATS: u32 = 14u;
// Below this depth a cell counts as dry.
const WET: f32 = 0.01;
const MAX_SPEED: f32 = 4.0;
// Foam fades over this many seconds.
const FOAM_LIFE: f32 = 5.0;
const NO_WATER: f32 = -1.0e4;
const PI: f32 = 3.14159265;

fn inside(c: vec2<i32>) -> bool {
    return all(c >= vec2<i32>(0)) && all(c < vec2<i32>(river.size));
}

fn index_of(c: vec2<i32>) -> u32 {
    return u32(c.y) * river.size.x + u32(c.x);
}

fn surface_level(c: vec2<i32>) -> f32 {
    let i = index_of(c);
    return terrain[i].x + water_in[i].x;
}

// The pipe's new outflow towards neighbour `n` (none beyond the grid).
fn pipe(previous: f32, level: f32, n: vec2<i32>) -> f32 {
    if !inside(n) {
        return 0.0;
    }
    return max(previous + river.dt * river.gravity * river.cell * (level - surface_level(n)), 0.0);
}

@compute @workgroup_size(8, 8)
fn flux_step(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= river.size) {
        return;
    }
    let c = vec2<i32>(id.xy);
    let i = index_of(c);
    let w = water_in[i];
    let level = terrain[i].x + w.x;
    // Bed friction (Manning): the flow slows by g n^2 |v| / d^(4/3) per second, plus a little
    // damping that keeps standing waves from ringing.
    let speed = length(w.yz);
    let friction = river.gravity * river.manning * river.manning * speed / pow(max(w.x, 0.03), 4.0 / 3.0);
    let f = flux[i] / (1.0 + river.dt * (friction + 0.15));
    var out = vec4<f32>(
        pipe(f.x, level, c + vec2<i32>(-1, 0)),
        pipe(f.y, level, c + vec2<i32>(1, 0)),
        pipe(f.z, level, c + vec2<i32>(0, -1)),
        pipe(f.w, level, c + vec2<i32>(0, 1)),
    );
    // A cell never gives more water than it holds.
    let total = (out.x + out.y + out.z + out.w) * river.dt;
    let volume = w.x * river.cell * river.cell;
    if total > volume {
        out *= volume / max(total, 1e-9);
    }
    flux[i] = out;
}

fn flux_at(c: vec2<i32>) -> vec4<f32> {
    if !inside(c) {
        return vec4<f32>(0.0);
    }
    return flux[index_of(c)];
}

// Foam of the previous state at cell coordinates `p` (bilinear, clamped to the grid).
fn foam_at(p: vec2<f32>) -> f32 {
    let q = clamp(p - 0.5, vec2<f32>(0.0), vec2<f32>(river.size) - 1.001);
    let base = vec2<i32>(floor(q));
    let f = q - floor(q);
    let a = mix(water_in[index_of(base)].w, water_in[index_of(base + vec2<i32>(1, 0))].w, f.x);
    let b = mix(water_in[index_of(base + vec2<i32>(0, 1))].w, water_in[index_of(base + vec2<i32>(1, 1))].w, f.x);
    return mix(a, b, f.y);
}

fn clamp_cell(c: vec2<i32>) -> vec2<i32> {
    return clamp(c, vec2<i32>(0), vec2<i32>(river.size) - 1);
}

@compute @workgroup_size(8, 8)
fn depth_step(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= river.size) {
        return;
    }
    let c = vec2<i32>(id.xy);
    let i = index_of(c);
    let w = water_in[i];
    let f = flux[i];
    let left = flux_at(c + vec2<i32>(-1, 0));
    let right = flux_at(c + vec2<i32>(1, 0));
    let back = flux_at(c + vec2<i32>(0, -1));
    let front = flux_at(c + vec2<i32>(0, 1));
    let inflow = left.y + right.x + back.w + front.z;
    let outflow = f.x + f.y + f.z + f.w;
    let area = river.cell * river.cell;
    var depth = w.x + river.dt * (inflow - outflow) / area;
    let center = river.origin + (vec2<f32>(c) + 0.5) * river.cell;
    for (var s = 0u; s < river.source_count; s++) {
        let source = river.sources[s];
        if distance(center, source.xy) < source.z {
            depth += river.dt * source.w / (PI * source.z * source.z);
        }
    }
    let bed = terrain[i];
    if bed.y > 0.5 {
        depth = max(river.outlet_level - bed.x, 0.0);
    }
    // The grid's border drains: water spilling that far leaves the simulated region instead of
    // piling up against its edge.
    if any(c == vec2<i32>(0)) || any(c == vec2<i32>(river.size) - 1) {
        depth = 0.0;
    }
    depth = max(depth, 0.0);
    // Velocity from the water passing through the cell, over its mean depth.
    let through = vec2<f32>(left.y - f.x + f.y - right.x, back.w - f.z + f.w - front.z) * 0.5;
    let mean_depth = 0.5 * (w.x + depth);
    var velocity = vec2<f32>(0.0);
    if mean_depth > WET {
        velocity = through / (river.cell * max(mean_depth, 0.05));
        let speed = length(velocity);
        if speed > MAX_SPEED {
            velocity *= MAX_SPEED / speed;
        }
    }
    // Foam: carried along the flow, fading; born where the flow is fast or the surface breaks
    // (steep around obstacles and drops).
    var foam = foam_at(vec2<f32>(c) + 0.5 - velocity * river.dt / river.cell) * exp(-river.dt / FOAM_LIFE);
    if depth > WET {
        let level = bed.x + depth;
        // Only slopes between wet cells (a bank is no breaking wave).
        let gx = slope_level(c + vec2<i32>(1, 0), level) - slope_level(c - vec2<i32>(1, 0), level);
        let gz = slope_level(c + vec2<i32>(0, 1), level) - slope_level(c - vec2<i32>(0, 1), level);
        let slope = min(length(vec2<f32>(gx, gz)) / (2.0 * river.cell), 1.0);
        let speed = length(velocity);
        foam += river.dt * (smoothstep(0.9, 2.0, speed) * 0.6 + smoothstep(0.12, 0.35, slope) * 3.0 * smoothstep(0.3, 0.8, speed));
    } else {
        foam = 0.0;
    }
    water_out[i] = vec4<f32>(depth, velocity, saturate(foam));
}

fn wet(c: vec2<i32>) -> bool {
    let i = index_of(c);
    return water_in[i].x > WET || terrain[i].y > 0.5;
}

// The water surface height for the shores: wet cells' own, dry cells within three cells of
// water the highest neighbouring water surface (the banks above the water line get the wet
// band), the rest none.
@compute @workgroup_size(8, 8)
fn surface_step(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= river.size) {
        return;
    }
    let c = vec2<i32>(id.xy);
    var level = NO_WATER;
    if wet(c) {
        level = surface_level(c);
    } else {
        for (var dz = -3; dz <= 3; dz++) {
            for (var dx = -3; dx <= 3; dx++) {
                let n = c + vec2<i32>(dx, dz);
                if inside(n) && wet(n) {
                    level = max(level, surface_level(n));
                }
            }
        }
    }
    textureStore(surface_out, c, vec4<f32>(level, 0.0, 0.0, 0.0));
}

// Height of the drawn surface at cell `c`: the water's where wet (outlet cells hide under the
// bed: the lake's own surface shows there), just under the bed where dry, so the surface meets
// the ground at the water line.
fn drawn_height(c: vec2<i32>) -> f32 {
    let i = index_of(c);
    let bed = terrain[i];
    if bed.y > 0.5 || water_in[i].x <= WET {
        return bed.z - 0.12;
    }
    return bed.x + water_in[i].x;
}

// The water level used for the surface's slope at `n`: its own where wet, `fallback` (the centre
// cell's) where dry, so the shoreline doesn't tilt the normals.
fn slope_level(n: vec2<i32>, fallback: f32) -> f32 {
    let m = clamp_cell(n);
    let i = index_of(m);
    if water_in[i].x <= WET || terrain[i].y > 0.5 {
        return fallback;
    }
    return terrain[i].x + water_in[i].x;
}

@compute @workgroup_size(64)
fn vertex_step(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= river.vertex_count {
        return;
    }
    let base = (river.first_vertex + id.x) * VERTEX_FLOATS;
    let xz = vec2<f32>(vertices[base], vertices[base + 2u]);
    let c = clamp_cell(vec2<i32>(floor((xz - river.origin) / river.cell)));
    let i = index_of(c);
    let w = water_in[i];
    let y = drawn_height(c);
    let own = select(y, terrain[i].x + w.x, w.x > WET);
    let hx = slope_level(c + vec2<i32>(1, 0), own) - slope_level(c - vec2<i32>(1, 0), own);
    let hz = slope_level(c + vec2<i32>(0, 1), own) - slope_level(c - vec2<i32>(0, 1), own);
    let normal = normalize(vec3<f32>(-hx, 2.0 * river.cell, -hz));
    vertices[base + 1u] = y;
    vertices[base + 3u] = normal.x;
    vertices[base + 4u] = normal.y;
    vertices[base + 5u] = normal.z;
    // Second UV set: the velocity; colour: foam, depth, speed and 0 in alpha (simulated water).
    vertices[base + 8u] = w.y;
    vertices[base + 9u] = w.z;
    vertices[base + 10u] = w.w;
    vertices[base + 11u] = w.x;
    vertices[base + 12u] = length(w.yz);
    vertices[base + 13u] = 0.0;
}
