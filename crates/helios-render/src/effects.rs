//! Heterogeneous volumes and GPU particles in the frame: the renderer's few hooks into
//! [`crate::volumes`], [`crate::particles`] and [`crate::oit`].
//!
//! ```text
//! frame start      fire lights: scene light colours from last frame's simulated emission
//! after lighting   volume simulation + lighting caches · particle simulation + lighting
//!                  nearest translucent depth · volume march + temporal resolve · particles → OIT
//!                  composite of what lies behind the nearest translucent surface
//! translucency     (glass and water over that)
//! after it         composite of what lies in front · TSR reactive mask
//! ```
//!
//! Scenes without emitters (or with both switched off) record nothing and leave every other
//! pass as it was.

use glam::{Mat4, Vec3};

use crate::gpu_scene::GpuScene;
use crate::oit::{Bucket, Oit};
use crate::particles::ParticleSystem;
use crate::passes::shading::LightingInputs;
use crate::profiler::Recorder;
use crate::translucency::Translucency;
use crate::volumes::HeterogeneousVolumes;
use crate::Scene;

/// What [`Effects::update`] needs from the frame.
pub(crate) struct EffectsFrame<'a> {
    pub scene: &'a Scene,
    pub eye: Vec3,
    /// Unjittered.
    pub view_proj: &'a Mat4,
    pub volumes: bool,
    pub particles: bool,
    pub delta_time: f32,
    /// Internal resolution of the frame and the capacity of the internal targets.
    pub internal: (u32, u32),
    pub capacity: (u32, u32),
}

pub(crate) struct Effects {
    volumes: HeterogeneousVolumes,
    particles: ParticleSystem,
    oit: Oit,
    /// Anything is drawn this frame.
    active: bool,
}

impl Effects {
    pub fn new(device: &wgpu::Device) -> Self {
        Self { volumes: HeterogeneousVolumes::new(device), particles: ParticleSystem::new(device), oit: Oit::new(device), active: false }
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, f: &EffectsFrame) {
        self.volumes.update(device, queue, f.scene, f.eye, f.view_proj, f.volumes, f.delta_time, f.internal, f.capacity);
        self.particles.update(device, queue, f.scene, f.particles, f.delta_time);
        self.active = self.volumes.visible() || self.particles.active();
        if self.active {
            self.oit.update(device, queue, f.capacity, self.volumes.rect());
        }
    }

    /// Volumes or particles cover part of this frame (TSR reads the reactive mask).
    pub fn active(&self) -> bool {
        self.active
    }

    /// Before the light culling: fire-driven light colours.
    pub fn record_lights(&self, rec: &mut Recorder, lights: &crate::lights::LightCulling) {
        self.volumes.record_fire_lights(rec, lights);
    }

    /// After the deferred lighting: simulation, lighting, march, particle draw, and the
    /// composite of everything behind the nearest translucent surface.
    pub fn record_before_translucency(&self, rec: &mut Recorder, inputs: &LightingInputs, scene: &GpuScene, translucency: &Translucency) {
        self.volumes.record_simulation(rec, inputs);
        self.particles.record_simulation(rec, inputs);
        let Some(translucent_depth) = self.oit.translucent_depth().filter(|_| self.active) else { return };
        let behind = translucency.record_nearest_depth(rec, inputs.frame, scene, inputs.targets, translucent_depth);
        if !behind {
            self.oit.clear_translucent_depth(rec);
        }
        self.volumes.record(rec, inputs, translucent_depth);
        self.oit.record_particles(rec, inputs, &self.particles);
        if behind {
            self.oit.record_composite(rec, inputs, Bucket::Back, self.volumes.resolved());
        }
    }

    /// After the translucency pass: the composite of everything in front of it.
    pub fn record_after_translucency(&self, rec: &mut Recorder, inputs: &LightingInputs) {
        if self.active {
            self.oit.record_composite(rec, inputs, Bucket::Front, self.volumes.resolved());
        }
    }
}
