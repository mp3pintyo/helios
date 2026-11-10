//! GPU particles (a Niagara-lite): emitters simulated, lit and drawn entirely on the GPU, blended
//! order-independently (`oit.rs`). See `shaders/particles.wgsl` for the passes.
//!
//! Every [`ParticleEmitter`] of the scene owns a fixed range of slots (its `capacity`) in one
//! particle buffer. The CPU only turns spawn rates and bursts into a per-frame spawn count; dead
//! slots claim those spawns with an atomic counter, so there are no free lists, no compaction
//! and no read-back. Simulation (gravity, drag, curl-noise turbulence, collision with the depth
//! buffer: bounce or stick), lighting (once per particle per frame: sun through the virtual
//! shadow map and cloud shadows, Lumen's radiance cache, clustered local lights, emission, fog)
//! and drawing (one instanced draw of camera-facing or velocity-stretched quads, soft near the
//! opaque depth) each take one dispatch or draw for all emitters.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};
use wgpu::BindingResource::TextureView;
use wgpu::ShaderStages;

use crate::passes::shading::LightingInputs;
use crate::profiler::Recorder;
use crate::scene::Scene;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, Lazy, storage_buffer, uniform_buffer, workgroups};

/// Emitters per scene (the uniform's array).
const MAX_EMITTERS: usize = 16;
/// Bytes per particle (`Particle` in particles.wgsl) and per lit particle (`Visual`).
const PARTICLE_BYTES: u64 = 48;
const VISUAL_BYTES: u64 = 32;

/// Where particles are born, around [`ParticleEmitter::position`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpawnShape {
    Sphere { radius: f32 },
    Box { half_extent: Vec3 },
    /// A horizontal disc.
    Disc { radius: f32 },
}

/// What particles do when they reach the visible surface in the depth buffer (particles off
/// screen or behind other geometry don't collide).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collision {
    None,
    /// Bounce off the surface's normal: `restitution` of the normal speed is kept, `friction`
    /// of the tangential speed lost; slow particles come to rest.
    Bounce { restitution: f32, friction: f32 },
    /// Stop where they land.
    Stick,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParticleRender {
    /// Camera-facing square.
    Billboard,
    /// Stretched along the screen-space velocity by the distance covered in `stretch` seconds
    /// (sparks).
    VelocityStretched { stretch: f32 },
}

/// The sprite's opacity profile (procedural, no texture).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sprite {
    /// A soft round dot (sparks, embers, motes).
    Glow,
    /// A soft puff with some structure (smoke, spray).
    Puff,
}

/// Periodic bursts on top of the spawn rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Burst {
    pub count: u32,
    /// Seconds between bursts.
    pub period: f32,
}

/// A GPU particle emitter.
#[derive(Clone, Copy, Debug)]
pub struct ParticleEmitter {
    pub position: Vec3,
    /// Particles alive at most (the emitter's slots).
    pub capacity: u32,
    /// Particles born per second.
    pub rate: f32,
    pub burst: Option<Burst>,
    pub shape: SpawnShape,
    /// Lifetime range (seconds).
    pub lifetime: Vec2,
    /// Initial velocity: a direction, the half angle of a cone around it (radians) and a speed
    /// range (m/s).
    pub direction: Vec3,
    pub cone_angle: f32,
    pub speed: Vec2,
    /// m/s².
    pub gravity: Vec3,
    /// Velocity damping per second.
    pub drag: f32,
    /// Curl-noise turbulence: acceleration (m/s²), spatial frequency (per metre) and how fast
    /// the field evolves.
    pub turbulence: f32,
    pub turbulence_frequency: f32,
    pub turbulence_speed: f32,
    pub collision: Collision,
    /// Diameter at birth and at death (m), and a random ± spread of it (fraction).
    pub size: Vec2,
    pub size_spread: f32,
    /// Linear colour and opacity at age 0, 1/3, 2/3 and 1 of the lifetime (the colour is the
    /// albedo of lit particles).
    pub color: [Vec4; 4],
    /// Lit by the sun, the sky / GI and local lights.
    pub lit: bool,
    /// Emitted radiance: the colour times `emission`, or with `blackbody` (kelvin at birth and
    /// at death) a glowing body cooling over its life, times `emission`.
    pub emission: f32,
    pub blackbody: Option<Vec2>,
    pub render: ParticleRender,
    pub sprite: Sprite,
    /// Fade over this distance in front of the opaque surface behind (soft particles; 0: off).
    pub soft_distance: f32,
    /// Henyey–Greenstein anisotropy of the light lit particles scatter.
    pub anisotropy: f32,
}

impl ParticleEmitter {
    /// Hot sparks rising from a fire, stretched along their flight.
    pub fn sparks(position: Vec3) -> Self {
        Self {
            position,
            capacity: 1024,
            rate: 30.0,
            burst: Some(Burst { count: 24, period: 1.3 }),
            shape: SpawnShape::Sphere { radius: 0.25 },
            lifetime: Vec2::new(0.8, 2.2),
            direction: Vec3::Y,
            cone_angle: 0.45,
            speed: Vec2::new(1.5, 4.0),
            gravity: Vec3::new(0.0, -1.5, 0.0),
            drag: 0.6,
            turbulence: 6.0,
            turbulence_frequency: 1.2,
            turbulence_speed: 1.5,
            collision: Collision::Bounce { restitution: 0.3, friction: 0.4 },
            size: Vec2::new(0.012, 0.006),
            size_spread: 0.3,
            color: [Vec4::new(1.0, 1.0, 1.0, 1.0), Vec4::new(1.0, 1.0, 1.0, 1.0), Vec4::new(1.0, 1.0, 1.0, 0.8), Vec4::new(1.0, 1.0, 1.0, 0.0)],
            lit: false,
            emission: 1.0,
            blackbody: Some(Vec2::new(2050.0, 1250.0)),
            render: ParticleRender::VelocityStretched { stretch: 0.04 },
            sprite: Sprite::Glow,
            soft_distance: 0.0,
            anisotropy: 0.0,
        }
    }

    /// Glowing embers thrown out of a fire, settling on the ground and cooling there.
    pub fn embers(position: Vec3) -> Self {
        Self {
            position,
            capacity: 256,
            rate: 2.0,
            burst: Some(Burst { count: 10, period: 2.9 }),
            shape: SpawnShape::Sphere { radius: 0.25 },
            lifetime: Vec2::new(4.0, 7.0),
            direction: Vec3::Y,
            cone_angle: 0.9,
            speed: Vec2::new(1.5, 3.2),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            drag: 0.3,
            turbulence: 1.0,
            turbulence_frequency: 0.8,
            turbulence_speed: 1.0,
            collision: Collision::Stick,
            size: Vec2::new(0.03, 0.022),
            size_spread: 0.4,
            color: [Vec4::new(0.2, 0.08, 0.04, 1.0), Vec4::new(0.15, 0.06, 0.03, 1.0), Vec4::new(0.1, 0.05, 0.03, 1.0), Vec4::new(0.08, 0.05, 0.04, 0.0)],
            lit: true,
            emission: 0.6,
            blackbody: Some(Vec2::new(1650.0, 950.0)),
            render: ParticleRender::Billboard,
            sprite: Sprite::Glow,
            soft_distance: 0.0,
            anisotropy: 0.0,
        }
    }

    /// Dust motes drifting in a box of air around `centre`, visible where light shafts cross
    /// them (they scatter forwards).
    pub fn dust_motes(centre: Vec3, half_extent: Vec3, capacity: u32) -> Self {
        Self {
            position: centre,
            capacity,
            rate: capacity as f32 / 18.0,
            burst: None,
            shape: SpawnShape::Box { half_extent },
            lifetime: Vec2::new(14.0, 22.0),
            direction: Vec3::Y,
            cone_angle: std::f32::consts::PI,
            speed: Vec2::new(0.0, 0.04),
            gravity: Vec3::new(0.0, -0.01, 0.0),
            drag: 0.5,
            turbulence: 0.08,
            turbulence_frequency: 0.4,
            turbulence_speed: 0.15,
            collision: Collision::None,
            size: Vec2::new(0.025, 0.025),
            size_spread: 0.4,
            color: [Vec4::new(1.0, 0.95, 0.85, 0.0), Vec4::new(1.0, 0.95, 0.85, 0.85), Vec4::new(1.0, 0.95, 0.85, 0.85), Vec4::new(1.0, 0.95, 0.85, 0.0)],
            lit: true,
            emission: 0.0,
            blackbody: None,
            render: ParticleRender::Billboard,
            sprite: Sprite::Glow,
            soft_distance: 0.2,
            anisotropy: 0.55,
        }
    }

    fn mean_lifetime(&self) -> f32 {
        0.5 * (self.lifetime.x + self.lifetime.y)
    }
}

/// `ParticleEmitter` in particles.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct EmitterUniform {
    position: Vec4,
    extent: Vec4,
    direction: Vec4,
    motion: Vec4,
    gravity: Vec4,
    turbulence: Vec4,
    collision: Vec4,
    size: Vec4,
    colors: [Vec4; 4],
    emission: Vec4,
    look: Vec4,
    range: [u32; 4],
}

/// `ParticleSystem` in particles.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SystemUniform {
    timing: Vec4,
    counts: [u32; 4],
    emitters: [EmitterUniform; MAX_EMITTERS],
}

impl EmitterUniform {
    fn new(e: &ParticleEmitter, first: u32, spawns: u32, seed: u32, prewarm: bool) -> Self {
        let (shape, extent) = match e.shape {
            SpawnShape::Sphere { radius } => (0.0, Vec3::splat(radius)),
            SpawnShape::Box { half_extent } => (1.0, half_extent),
            SpawnShape::Disc { radius } => (2.0, Vec3::splat(radius)),
        };
        let (mode, restitution, friction) = match e.collision {
            Collision::None => (0.0, 0.0, 0.0),
            Collision::Bounce { restitution, friction } => (1.0, restitution, friction),
            Collision::Stick => (2.0, 0.0, 1.0),
        };
        let (render, stretch) = match e.render {
            ParticleRender::Billboard => (0.0, 0.0),
            ParticleRender::VelocityStretched { stretch } => (1.0, stretch),
        };
        let blackbody = e.blackbody.unwrap_or(Vec2::ZERO);
        Self {
            position: e.position.extend(shape),
            extent: extent.extend(e.size_spread),
            direction: e.direction.normalize_or(Vec3::Y).extend(e.cone_angle.clamp(0.0, std::f32::consts::PI).cos()),
            motion: Vec4::new(e.speed.x, e.speed.y, e.lifetime.x.max(0.01), e.lifetime.y.max(0.01)),
            gravity: e.gravity.extend(e.drag),
            turbulence: Vec4::new(e.turbulence, e.turbulence_frequency, e.turbulence_speed, mode),
            collision: Vec4::new(restitution, friction, 0.5 * e.size.x, 0.05),
            size: Vec4::new(e.size.x, e.size.y, stretch, render),
            colors: e.color,
            emission: Vec4::new(e.emission, blackbody.x, blackbody.y, e.lit as u32 as f32),
            look: Vec4::new(e.anisotropy, e.soft_distance, (e.sprite == Sprite::Puff) as u32 as f32, prewarm as u32 as f32),
            range: [first, e.capacity, spawns, seed],
        }
    }
}

/// Spawn bookkeeping of one emitter.
#[derive(Clone, Copy, Default)]
struct Spawner {
    /// Fractional particles carried over to the next frame.
    carry: f32,
    since_burst: f32,
}

impl Spawner {
    /// Whole particles to spawn over `dt`: the rate's share (the fraction carries over) plus a
    /// burst when its period has elapsed.
    fn spawns(&mut self, e: &ParticleEmitter, dt: f32) -> u32 {
        self.carry += e.rate * dt;
        let mut n = self.carry.floor();
        self.carry -= n;
        if let Some(burst) = e.burst.filter(|b| b.period > 0.0) {
            self.since_burst += dt;
            if self.since_burst >= burst.period {
                self.since_burst -= burst.period;
                n += burst.count as f32;
            }
        }
        n as u32
    }
}

pub(crate) struct ParticleSystem {
    simulate: Lazy<wgpu::ComputePipeline>,
    light: Lazy<wgpu::ComputePipeline>,
    draw: Lazy<wgpu::RenderPipeline>,
    simulate_layout: wgpu::BindGroupLayout,
    light_layout: wgpu::BindGroupLayout,
    draw_layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    /// Spawns claimed this frame, per emitter (atomic counters, zeroed every frame).
    spawned: wgpu::Buffer,
    particles: Option<wgpu::Buffer>,
    visuals: Option<wgpu::Buffer>,
    /// Capacities the buffers were laid out for (a change reallocates them: all particles die).
    layout_key: Vec<u32>,
    slots: u32,
    spawners: Vec<Spawner>,
    time: f32,
    frame: u32,
    active: bool,
}

impl ParticleSystem {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "particles.wgsl");
        let c = ShaderStages::COMPUTE;
        let simulate_layout = bind_group_layout(
            device,
            "particle simulation",
            c,
            &[
                (0, bind::uniform()),
                (110, bind::uniform()),
                (111, bind::storage(false)),
                (113, bind::storage(false)),
                (116, bind::texture(false)),
                (117, bind::texture(false)),
            ],
        );
        let mut light_entries = crate::media::lighting_entries();
        light_entries.extend([(110, bind::uniform()), (112, bind::storage(true)), (114, bind::storage(false))]);
        let light_layout = bind_group_layout(device, "particle lighting", c, &light_entries);
        let draw_layout = bind_group_layout(
            device,
            "particle draw",
            ShaderStages::VERTEX_FRAGMENT,
            &[(0, bind::uniform()), (112, bind::storage(true)), (115, bind::storage(true)), (116, bind::texture(false)), (118, bind::texture(false))],
        );
        Self {
            simulate: lazy_compute_pipeline(device, "particle simulation", &module, "simulate", &simulate_layout, &[]),
            light: lazy_compute_pipeline(device, "particle lighting", &module, "light", &light_layout, &[]),
            draw: {
                let (device, module, layout) = (device.clone(), module.clone(), draw_layout.clone());
                Lazy::new(move || crate::oit::particle_pipeline(&device, &module, &layout))
            },
            simulate_layout,
            light_layout,
            draw_layout,
            uniform: uniform_buffer(device, "particles", std::mem::size_of::<SystemUniform>() as u64),
            spawned: storage_buffer(device, "particle spawns", MAX_EMITTERS as u64 * 4, wgpu::BufferUsages::empty()),
            particles: None,
            visuals: None,
            layout_key: Vec::new(),
            slots: 0,
            spawners: Vec::new(),
            time: 0.0,
            frame: 0,
            active: false,
        }
    }

    /// Whether particles are simulated and drawn this frame.
    pub fn active(&self) -> bool {
        self.active
    }

    /// Lays out the emitters' slots, turns rates and bursts into this frame's spawn counts and
    /// uploads the uniform.
    /// Queues the pipelines for the browser's background compile if the scene has emitters.
    pub fn prefetch(&self, scene: &Scene) {
        if scene.particles.iter().take(MAX_EMITTERS).any(|e| e.capacity > 0) {
            self.simulate.prefetch();
            self.light.prefetch();
            self.draw.prefetch();
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, enabled: bool, delta_time: f32) {
        let emitters = &scene.particles[..scene.particles.len().min(MAX_EMITTERS)];
        self.active = enabled && emitters.iter().any(|e| e.capacity > 0);
        if !self.active {
            return;
        }
        let dt = delta_time.clamp(0.0, 0.1);
        self.time += dt;
        self.frame = self.frame.wrapping_add(1);
        let key: Vec<u32> = emitters.iter().map(|e| e.capacity).collect();
        let prewarm = key != self.layout_key;
        if prewarm {
            self.slots = key.iter().sum();
            let slots = self.slots.max(1) as u64;
            // Zero-initialised: every slot starts dead.
            self.particles = Some(storage_buffer(device, "particles", slots * PARTICLE_BYTES, wgpu::BufferUsages::empty()));
            self.visuals = Some(storage_buffer(device, "particle visuals", slots * VISUAL_BYTES, wgpu::BufferUsages::empty()));
            self.spawners = vec![Spawner::default(); emitters.len()];
            self.layout_key = key;
        }
        let mut uniform = SystemUniform { timing: Vec4::new(dt, self.time, 0.0, 0.0), counts: [emitters.len() as u32, self.slots, 0, 0], emitters: [EmitterUniform::default(); MAX_EMITTERS] };
        let mut first = 0;
        for (i, (e, spawner)) in emitters.iter().zip(&mut self.spawners).enumerate() {
            let spawns = if prewarm {
                // Start with the population the emitter sustains, at random ages.
                (e.rate * e.mean_lifetime()).min(e.capacity as f32) as u32
            } else {
                spawner.spawns(e, dt)
            };
            uniform.emitters[i] = EmitterUniform::new(e, first, spawns, self.frame.wrapping_mul(16) + i as u32, prewarm);
            first += e.capacity;
        }
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(&self.spawned, 0, &[0u8; MAX_EMITTERS * 4]);
    }

    /// Simulates and lights the particles (after the G-buffer: they collide with its depth).
    pub fn record_simulation(&self, rec: &mut Recorder, inputs: &LightingInputs) {
        let (Some(particles), Some(visuals)) = (&self.particles, &self.visuals) else { return };
        if !self.active {
            return;
        }
        let device = rec.device;
        let t = inputs.targets;
        let simulate = bind_group(
            device,
            "particle simulation",
            &self.simulate_layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (110, self.uniform.as_entire_binding()),
                (111, particles.as_entire_binding()),
                (113, self.spawned.as_entire_binding()),
                (116, TextureView(&t.depth)),
                (117, TextureView(&t.gbuffer_normal)),
            ],
        );
        let mut entries = crate::media::lighting_resources(inputs);
        entries.extend([(110, self.uniform.as_entire_binding()), (112, particles.as_entire_binding()), (114, visuals.as_entire_binding())]);
        let light = bind_group(device, "particle lighting", &self.light_layout, &entries);
        let groups = workgroups(self.slots, 64);
        let mut pass = rec.compute_pass("particle simulation");
        pass.set_pipeline(&self.simulate);
        pass.set_bind_group(0, &simulate, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        pass.set_pipeline(&self.light);
        pass.set_bind_group(0, &light, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Draws every slot (dead ones are culled in the vertex shader) into the bound OIT targets.
    pub fn draw(&self, device: &wgpu::Device, pass: &mut wgpu::RenderPass, inputs: &LightingInputs, translucent_depth: &wgpu::TextureView) {
        let (Some(particles), Some(visuals)) = (&self.particles, &self.visuals) else { return };
        if !self.active {
            return;
        }
        let group = bind_group(
            device,
            "particle draw",
            &self.draw_layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (112, particles.as_entire_binding()),
                (115, visuals.as_entire_binding()),
                (116, TextureView(&inputs.targets.depth)),
                (118, TextureView(translucent_depth)),
            ],
        );
        pass.set_pipeline(&self.draw);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..6, 0..self.slots);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_match_the_shader_layout() {
        assert_eq!(std::mem::size_of::<EmitterUniform>(), 240);
        assert_eq!(std::mem::size_of::<SystemUniform>(), 32 + 240 * MAX_EMITTERS);
    }

    #[test]
    fn spawn_rates_and_bursts_become_whole_particles() {
        let mut emitter = ParticleEmitter::sparks(Vec3::ZERO);
        emitter.rate = 25.0;
        emitter.burst = Some(Burst { count: 10, period: 0.5 });
        let mut spawner = Spawner::default();
        // One second at 60 fps: 25 from the rate (the fraction carried over) and two bursts.
        let total: u32 = (0..60).map(|_| spawner.spawns(&emitter, 1.0 / 60.0)).sum();
        assert!((44..=46).contains(&total), "{total}");
        let u = EmitterUniform::new(&emitter, 7, 3, 1, false);
        assert_eq!(u.range, [7, 1024, 3, 1]);
        assert!((u.direction.w - 0.45f32.cos()).abs() < 1e-6);
    }
}
