// Eulerian smoke and fire simulation of a heterogeneous volume emitter (volumes.rs), after
// Stam 1999 ("Stable fluids"), Fedkiw et al. 2001 (vorticity confinement) and Selle et al. 2008
// (MacCormack advection), on a collocated grid of 3D textures:
//
//   velocity : rgb m/s
//   state    : x soot density, y temperature (0 ambient .. 1 hottest flame), z fuel
//
// One frame (dispatches in this order, see `VolumeEmitterGpu::record_simulation`):
//   curl       : vorticity w = curl v of the current velocity
//   forces     : buoyancy (hot gas rises, soot weighs it down), vorticity confinement (puts
//                back the small eddies numerical diffusion removes), the source (velocity,
//                fuel, soot and heat injected inside a flickering noise-modulated sphere), the
//                ambient wind and damping
//   divergence : of the forced velocity
//   pressure   : red-black SOR iterations of the Poisson equation (warm-started from the last
//                frame's pressure); solid floor, open sides and top
//   project    : subtract the pressure gradient: the velocity becomes divergence-free
//   advect     : semi-Lagrangian self-advection of the velocity, forward advection of the state
//   maccormack : back-advect the advected state and correct by half the round-trip error,
//                clamped to the departure point's neighbourhood (sharp, stable smoke detail);
//                then combustion (fuel burns into heat and soot), cooling and dissipation, a
//                fade at the open boundaries; sums the fire's emission for the light it drives
//
// `procedural` instead fills the state from animated noise (mist, distant or static volumes).

#include "common.wgsl"
#include "noise.wgsl"
#include "blackbody.wgsl"
#include "lights.wgsl"

struct VolumeSim {
    resolution: vec4<u32>,      // xyz cells
    cell: vec4<f32>,            // xyz cell size (m), w: time step (s)
    source: vec4<f32>,          // xyz source centre (cells), w: radius (cells)
    source_rates: vec4<f32>,    // x soot / s, y fuel / s, z temperature, w flicker (0..1)
    source_velocity: vec4<f32>, // xyz m/s, w: time (s)
    forces: vec4<f32>,          // x buoyancy (m/s² per unit temperature), y soot weight, z vorticity, w SOR weight
    dissipation: vec4<f32>,     // x soot, y cooling, z velocity (per second), w fuel burn rate (per second)
    combustion: vec4<f32>,      // x heat per fuel, y soot per fuel, z boundary fade (cells), w emission kelvin at T = 1
    wind: vec4<f32>,            // xyz ambient flow (m/s), w: rate the flow relaxes to it (per second)
    procedural: vec4<f32>,      // x density, y noise frequency (per m), z height falloff, w coverage
    drift: vec4<f32>,           // xyz noise offset (m), w: edge fade (fraction of the half extent)
    emission: vec4<f32>,        // x emission kelvin at T = 0, y: emitter index
}

@group(0) @binding(0) var<uniform> sim: VolumeSim;
@group(0) @binding(1) var velocity_in: texture_3d<f32>;
@group(0) @binding(3) var state_in: texture_3d<f32>;
@group(0) @binding(4) var state_aux: texture_3d<f32>;
@group(0) @binding(5) var curl_in: texture_3d<f32>;
@group(0) @binding(6) var linear_clamp: sampler;
@group(0) @binding(7) var velocity_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var state_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var curl_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(10) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(11) var<storage, read_write> divergence_out: array<f32>;
@group(0) @binding(12) var<storage, read_write> emission_sum: array<atomic<u32>>;
@group(0) @binding(13) var<storage, read> pressure_in: array<f32>;
@group(0) @binding(14) var<storage, read> divergence_in: array<f32>;

// Red (0) or black (1) cells of a pressure iteration.
override PARITY: u32 = 0u;

// Speed limit (m/s) and state limit that keep the fp16 fields finite whatever the parameters.
const MAX_SPEED: f32 = 20.0;
const MAX_STATE: f32 = 100.0;

const SEMI_LAGRANGIAN_SHARE: f32 = 0.5;

// Fixed-point scale of the emission sum (WebGPU has no float atomics).
const EMISSION_SCALE: f32 = 64.0;

fn res() -> vec3<i32> {
    return vec3<i32>(sim.resolution.xyz);
}

fn inside(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < res());
}

fn cell_index(c: vec3<i32>) -> u32 {
    let r = res();
    return u32(c.x + (c.y + c.z * r.y) * r.x);
}

fn velocity_at(c: vec3<i32>) -> vec3<f32> {
    return textureLoad(velocity_in, clamp(c, vec3<i32>(0), res() - 1), 0).xyz;
}

// Velocity with the boundary conditions: a solid floor (nothing flows through it), open sides
// and top (zero gradient).
fn velocity_bc(c: vec3<i32>) -> vec3<f32> {
    if c.y < 0 {
        return vec3<f32>(0.0);
    }
    return velocity_at(c);
}

// Pressure with the boundary conditions: 0 beyond the open sides and top, mirrored at the floor.
fn pressure_bc(c: vec3<i32>) -> f32 {
    var q = c;
    if q.y < 0 {
        q.y = 0;
    }
    if !inside(q) {
        return 0.0;
    }
    return pressure_in[cell_index(q)];
}

fn pressure_rw(c: vec3<i32>) -> f32 {
    var q = c;
    if q.y < 0 {
        q.y = 0;
    }
    if !inside(q) {
        return 0.0;
    }
    return pressure[cell_index(q)];
}

// `v` limited to MAX_SPEED; zero where it is not finite.
fn bounded_velocity(v: vec3<f32>) -> vec3<f32> {
    let speed = length(v);
    if !(speed < 1e30) {
        return vec3<f32>(0.0);
    }
    return v * min(1.0, MAX_SPEED / max(speed, 1e-6));
}

fn texel_uvw(c: vec3<i32>) -> vec3<f32> {
    return (vec3<f32>(c) + 0.5) / vec3<f32>(res());
}

// Displacement in texture coordinates of velocity `v` over the time step.
fn uvw_step(v: vec3<f32>) -> vec3<f32> {
    return v * sim.cell.w / (sim.cell.xyz * vec3<f32>(res()));
}

@compute @workgroup_size(4, 4, 4)
fn curl(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec3<i32>(gid);
    if !inside(c) {
        return;
    }
    let h2 = 2.0 * sim.cell.xyz;
    let dx = (velocity_at(c + vec3<i32>(1, 0, 0)) - velocity_at(c - vec3<i32>(1, 0, 0))) / h2.x;
    let dy = (velocity_at(c + vec3<i32>(0, 1, 0)) - velocity_at(c - vec3<i32>(0, 1, 0))) / h2.y;
    let dz = (velocity_at(c + vec3<i32>(0, 0, 1)) - velocity_at(c - vec3<i32>(0, 0, 1))) / h2.z;
    let w = vec3<f32>(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x);
    textureStore(curl_out, c, vec4<f32>(w, length(w)));
}

fn curl_magnitude(c: vec3<i32>) -> f32 {
    return textureLoad(curl_in, clamp(c, vec3<i32>(0), res() - 1), 0).w;
}

// Strength of the source at cell `c` (0..1): a soft sphere whose injection flickers with
// animated noise (a turbulent, licking flame instead of a steady jet).
fn source_mask(c: vec3<i32>) -> f32 {
    let p = vec3<f32>(c) + 0.5;
    let r = distance(p, sim.source.xyz) / max(sim.source.w, 1e-3);
    let shape = 1.0 - smoothstep(0.6, 1.0, r);
    if shape <= 0.0 {
        return 0.0;
    }
    let t = sim.source_velocity.w;
    let world = p * sim.cell.xyz;
    let n = gradient_noise(world * 3.0 + vec3<f32>(0.0, -t * 2.5, t * 0.7)) * 0.7 + gradient_noise(world * 7.0 + vec3<f32>(t * 1.3, -t * 4.0, 0.0)) * 0.3;
    let flicker = mix(1.0, saturate(0.55 + 1.6 * n), sim.source_rates.w);
    return shape * flicker;
}

@compute @workgroup_size(4, 4, 4)
fn forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec3<i32>(gid);
    if !inside(c) {
        return;
    }
    let dt = sim.cell.w;
    var v = textureLoad(velocity_in, c, 0).xyz;
    var s = textureLoad(state_in, c, 0);

    // Buoyancy (Boussinesq): temperature lifts, soot weighs down.
    v.y += dt * (sim.forces.x * s.y - sim.forces.y * s.x);

    // Vorticity confinement: push along N x w, N pointing towards stronger vorticity.
    let h2 = 2.0 * sim.cell.xyz;
    let eta = vec3<f32>(
        (curl_magnitude(c + vec3<i32>(1, 0, 0)) - curl_magnitude(c - vec3<i32>(1, 0, 0))) / h2.x,
        (curl_magnitude(c + vec3<i32>(0, 1, 0)) - curl_magnitude(c - vec3<i32>(0, 1, 0))) / h2.y,
        (curl_magnitude(c + vec3<i32>(0, 0, 1)) - curl_magnitude(c - vec3<i32>(0, 0, 1))) / h2.z,
    );
    let n = eta / (length(eta) + 1e-5);
    let w = textureLoad(curl_in, c, 0).xyz;
    let h = (sim.cell.x + sim.cell.y + sim.cell.z) / 3.0;
    v += dt * sim.forces.z * h * cross(n, w);

    // Source: drives the gas towards its velocity, adds fuel and soot, heats.
    let src = source_mask(c);
    if src > 0.0 {
        v = mix(v, sim.source_velocity.xyz, saturate(src * dt * 20.0));
        s.x += src * sim.source_rates.x * dt;
        s.z += src * sim.source_rates.y * dt;
        s.y = max(s.y, src * sim.source_rates.z);
    }

    // Ambient flow and damping.
    v += (sim.wind.xyz - v) * (1.0 - exp(-sim.wind.w * dt));
    v *= exp(-sim.dissipation.z * dt);
    textureStore(velocity_out, c, vec4<f32>(bounded_velocity(v), 0.0));
    textureStore(state_out, c, s);
}

@compute @workgroup_size(4, 4, 4)
fn divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec3<i32>(gid);
    if !inside(c) {
        return;
    }
    let h2 = 2.0 * sim.cell.xyz;
    let d = (velocity_bc(c + vec3<i32>(1, 0, 0)).x - velocity_bc(c - vec3<i32>(1, 0, 0)).x) / h2.x
        + (velocity_bc(c + vec3<i32>(0, 1, 0)).y - velocity_bc(c - vec3<i32>(0, 1, 0)).y) / h2.y
        + (velocity_bc(c + vec3<i32>(0, 0, 1)).z - velocity_bc(c - vec3<i32>(0, 0, 1)).z) / h2.z;
    divergence_out[cell_index(c)] = d;
}

// One half (PARITY) of a red-black successive over-relaxation sweep of
// laplacian(p) = divergence (p: pressure times the time step); neighbours of the other colour are final for this half.
@compute @workgroup_size(4, 4, 4)
fn pressure_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec3<i32>(i32(gid.x) * 2, i32(gid.y), i32(gid.z));
    // Cells of this colour along x: every other one, offset by the row's parity.
    let shift = (u32(c.y + c.z) + PARITY) & 1u;
    let cell = c + vec3<i32>(i32(shift), 0, 0);
    if !inside(cell) {
        return;
    }
    let ih2 = 1.0 / (sim.cell.xyz * sim.cell.xyz);
    let sum = (pressure_rw(cell + vec3<i32>(1, 0, 0)) + pressure_rw(cell - vec3<i32>(1, 0, 0))) * ih2.x
        + (pressure_rw(cell + vec3<i32>(0, 1, 0)) + pressure_rw(cell - vec3<i32>(0, 1, 0))) * ih2.y
        + (pressure_rw(cell + vec3<i32>(0, 0, 1)) + pressure_rw(cell - vec3<i32>(0, 0, 1))) * ih2.z;
    let index = cell_index(cell);
    let gauss_seidel = (sum - divergence_in[index]) / (2.0 * (ih2.x + ih2.y + ih2.z));
    pressure[index] = mix(pressure[index], gauss_seidel, sim.forces.w);
}

@compute @workgroup_size(4, 4, 4)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec3<i32>(gid);
    if !inside(c) {
        return;
    }
    let h2 = 2.0 * sim.cell.xyz;
    let gradient = vec3<f32>(
        (pressure_bc(c + vec3<i32>(1, 0, 0)) - pressure_bc(c - vec3<i32>(1, 0, 0))) / h2.x,
        (pressure_bc(c + vec3<i32>(0, 1, 0)) - pressure_bc(c - vec3<i32>(0, 1, 0))) / h2.y,
        (pressure_bc(c + vec3<i32>(0, 0, 1)) - pressure_bc(c - vec3<i32>(0, 0, 1))) / h2.z,
    );
    var v = textureLoad(velocity_in, c, 0).xyz - gradient;
    if c.y == 0 {
        v.y = max(v.y, 0.0);
    }
    textureStore(velocity_out, c, vec4<f32>(bounded_velocity(v), 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec3<i32>(gid);
    if !inside(c) {
        return;
    }
    let v = textureLoad(velocity_in, c, 0).xyz;
    let back = texel_uvw(c) - uvw_step(v);
    textureStore(velocity_out, c, textureSampleLevel(velocity_in, linear_clamp, back, 0.0));
    textureStore(state_out, c, textureSampleLevel(state_in, linear_clamp, back, 0.0));
}

var<workgroup> group_emission: atomic<u32>;

// state_in: the state before advection, state_aux: advected forward (advect).
@compute @workgroup_size(4, 4, 4)
fn maccormack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = vec3<i32>(gid);
    var emission = 0.0;
    if inside(c) {
        let dt = sim.cell.w;
        let v = textureLoad(velocity_in, c, 0).xyz;
        let uvw = texel_uvw(c);
        let forward = textureLoad(state_aux, c, 0);
        let round_trip = textureSampleLevel(state_aux, linear_clamp, uvw + uvw_step(v), 0.0);
        var s = forward + 0.5 * (textureLoad(state_in, c, 0) - round_trip);
        // Limiter: stay within the values the departure point was interpolated from.
        let q = (uvw - uvw_step(v)) * vec3<f32>(res()) - 0.5;
        let base = vec3<i32>(floor(q));
        var lo = vec4<f32>(1e9);
        var hi = vec4<f32>(-1e9);
        for (var k = 0; k < 8; k++) {
            let o = vec3<i32>(k & 1, (k >> 1u) & 1, (k >> 2u) & 1);
            let t = textureLoad(state_in, clamp(base + o, vec3<i32>(0), res() - 1), 0);
            lo = min(lo, t);
            hi = max(hi, t);
        }
        // Half of the plain semi-Lagrangian result keeps a little numerical viscosity: pure
        // MacCormack leaves voxel-sharp fronts that read as blocks where the smoke is opaque.
        s = mix(clamp(s, lo, hi), forward, SEMI_LAGRANGIAN_SHARE);

        // Combustion: fuel burns into heat and soot.
        let burn = s.z * (1.0 - exp(-sim.dissipation.w * dt));
        s.z -= burn;
        s.y = min(s.y + burn * sim.combustion.x, 1.5);
        s.x += burn * sim.combustion.y;
        s.x *= exp(-sim.dissipation.x * dt);
        s.y *= exp(-sim.dissipation.y * dt);

        // Fade towards the open sides and top, so gas leaving the box does not pile up there.
        let r = res();
        let edge = min(min(c.x, r.x - 1 - c.x), min(c.z, r.z - 1 - c.z));
        let distance_to_boundary = f32(min(edge, r.y - 1 - c.y));
        let fade = smoothstep(0.0, max(sim.combustion.z, 1.0), distance_to_boundary);
        s = s * vec4<f32>(fade, fade, fade, 1.0);
        // Finite and bounded (fp16 storage): a value that ever turned NaN would spread to its
        // neighbours through every filtered lookup.
        s = select(clamp(s, vec4<f32>(0.0), vec4<f32>(MAX_STATE)), vec4<f32>(0.0), s != s);
        textureStore(state_out, c, s);

        let kelvin = mix(sim.emission.x, sim.combustion.w, saturate(s.y));
        emission = luminance(blackbody(kelvin)) * smoothstep(0.02, 0.15, s.y);
    }
    // Sum of the emission for the light the fire drives: per workgroup, then one global add.
    atomicAdd(&group_emission, u32(min(emission, 4000.0) * EMISSION_SCALE));
    workgroupBarrier();
    if lid == 0u {
        let total = atomicLoad(&group_emission);
        if total > 0u {
            atomicAdd(&emission_sum[u32(sim.emission.y)], total);
        }
    }
}

@compute @workgroup_size(4, 4, 4)
fn procedural(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec3<i32>(gid);
    if !inside(c) {
        return;
    }
    let cell_centre = vec3<f32>(c) + 0.5;
    let world = cell_centre * sim.cell.xyz;
    let p = world * sim.procedural.y + sim.drift.xyz * sim.procedural.y;
    let n = gradient_noise(p) * 0.55 + gradient_noise(p * 2.3 + 7.1) * 0.3 + gradient_noise(p * 5.1 + 3.7) * 0.15;
    let coverage = sim.procedural.w;
    let shape = saturate((n * 0.5 + 0.5 - (1.0 - coverage)) / max(coverage, 1e-3));
    // Thins out with height and towards the box's sides (an ellipse in plan).
    let extent = vec3<f32>(res());
    let height = world.y;
    let plan = (cell_centre.xz / extent.xz) * 2.0 - 1.0;
    let fade = sim.drift.w;
    let side = 1.0 - smoothstep(1.0 - fade, 1.0, length(plan));
    let top = 1.0 - smoothstep(0.7, 1.0, cell_centre.y / extent.y);
    let density = sim.procedural.x * shape * exp(-height * sim.procedural.z) * side * top;
    textureStore(state_out, c, vec4<f32>(density, 0.0, 0.0, 0.0));
}

// ---- Fire light -------------------------------------------------------------------------------
// A fire drives a local light (`VolumeEmitter::light`): before the light culling, its colour is
// set to the scene's colour x intensity scaled by the fire's emission (summed by the last
// frame's `maccormack`) relative to its running average, so the light flickers with the flames
// the viewer sees, without a read-back.

struct Flicker {
    color: vec4<f32>,  // rgb: colour x intensity, w: source radius
    index: vec4<u32>,  // x: light, y: emitter
}

struct Flickers {
    count: vec4<u32>,
    lights: array<Flicker, 8>,
}

@group(0) @binding(20) var<uniform> flickers: Flickers;
@group(0) @binding(21) var<storage, read_write> lights_rw: array<GpuLight>;
@group(0) @binding(22) var<storage, read_write> flicker_state: array<vec4<f32>>;

@compute @workgroup_size(8)
fn flicker_lights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= flickers.count.x {
        return;
    }
    let f = flickers.lights[i];
    let emission = f32(atomicLoad(&emission_sum[f.index.y])) / EMISSION_SCALE;
    var average = flicker_state[i].x;
    average = select(mix(average, emission, 0.02), emission, average <= 0.0);
    flicker_state[i] = vec4<f32>(average, 0.0, 0.0, 0.0);
    let ratio = select(1.0, clamp(emission / average, 0.35, 1.8), average > 0.0);
    lights_rw[f.index.x].color = vec4<f32>(f.color.rgb * ratio, f.color.w);
}
