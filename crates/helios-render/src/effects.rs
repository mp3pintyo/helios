//! Heterogeneous volumes and GPU particles in the frame: the renderer's few hooks into
//! [`crate::volumes`], [`crate::particles`] and [`crate::oit`].
//!
//! ```text
//! frame start      fire lights: scene light colours from last frame's simulated emission
//! after lighting   volume simulation + lighting caches · particle simulation + lighting
//!                  nearest translucent (AVBOIT: refracting) depth
//!                  AVBOIT only: grid bounds (low-resolution surfaces + particles, dilated)
//!                    · grid transmittance (volumes, low-resolution surfaces + particles) · resolve
//!                  volume march + temporal resolve · particles → OIT
//!                  AVBOIT only: order-independent translucent surfaces → OIT
//!                  composite of what lies behind the nearest translucent surface
//! translucency     (glass and water over that; AVBOIT: the refracting surfaces only)
//! after it         composite of what lies in front · TSR reactive mask
//! ```
//!
//! Scenes without emitters (or with both switched off) and, with AVBOIT, without
//! order-independent translucent surfaces record none of these passes.

use glam::{Mat4, Vec3};

use crate::gpu_scene::GpuScene;
use crate::oit::{AvboitStage, Bucket, Oit};
use crate::particles::ParticleSystem;
use crate::passes::shading::LightingInputs;
use crate::profiler::Recorder;
use crate::textures::TextureLibrary;
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
    /// Composite with adaptive voxel-based OIT (`avboit.rs`).
    pub avboit: bool,
    /// The translucency pass has order-independent surfaces this frame (AVBOIT only).
    pub translucent_surfaces: bool,
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

    /// Queues the pipelines of the volumes and particles the scene has for the browser's
    /// background compile (with `avboit`, those of the AVBOIT path too).
    pub fn prefetch(&self, scene: &crate::Scene, avboit: bool) {
        self.volumes.prefetch(scene);
        self.particles.prefetch(scene);
        if avboit {
            self.volumes.prefetch_avboit(scene);
            self.particles.prefetch_avboit(scene);
            self.oit.prefetch_avboit();
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, f: &EffectsFrame) {
        self.volumes.update(device, queue, f.scene, f.eye, f.view_proj, f.volumes, f.delta_time, f.internal, f.capacity);
        self.particles.update(device, queue, f.scene, f.particles, f.delta_time);
        let media = self.volumes.visible() || self.particles.active();
        self.active = media || (f.avboit && f.translucent_surfaces);
        if self.active {
            self.oit.update(device, queue, f.capacity, f.internal, self.volumes.rect(), f.avboit);
        } else if self.oit.uses_avboit() {
            self.oit.disable_avboit(device, queue);
        }
    }

    /// Volumes, particles or (AVBOIT) order-independent surfaces cover part of this frame (TSR
    /// reads the reactive mask).
    pub fn active(&self) -> bool {
        self.active
    }

    /// AVBOIT: the nearest refracting surface's depth as a one-layer array (the translucency
    /// pass composites the fragments at it), while the effects run.
    pub fn nearest_refracting(&self) -> Option<&wgpu::TextureView> {
        self.oit.translucent_depth_array().filter(|_| self.active && self.oit.uses_avboit())
    }

    /// Before the light culling: fire-driven light colours.
    pub fn record_lights(&self, rec: &mut Recorder, lights: &crate::lights::LightCulling) {
        self.volumes.record_fire_lights(rec, lights);
    }

    /// After the deferred lighting: simulation, lighting, march, particle draw, and the
    /// composite of everything behind the nearest translucent surface. With AVBOIT also the grid
    /// and the order-independent translucent surfaces.
    pub fn record_before_translucency(&self, rec: &mut Recorder, inputs: &LightingInputs, scene: &GpuScene, translucency: &Translucency, textures: &TextureLibrary) {
        self.volumes.record_simulation(rec, inputs);
        self.particles.record_simulation(rec, inputs);
        let Some(translucent_depth) = self.oit.translucent_depth().filter(|_| self.active) else { return };
        let behind = translucency.record_nearest_depth(rec, inputs.frame, scene, inputs.targets, translucent_depth);
        if !behind {
            self.oit.clear_translucent_depth(rec);
        }
        if self.oit.uses_avboit() {
            self.record_avboit_grid(rec, inputs, scene, translucency, textures, translucent_depth);
        }
        let avboit = &self.oit.avboit;
        self.volumes.record(rec, inputs, translucent_depth, avboit);
        self.oit.record_particles(rec, inputs, &self.particles);
        if self.oit.uses_avboit() {
            let depth_array = self.oit.translucent_depth_array().expect("targets exist while active");
            for bucket in [Bucket::Front, Bucket::Back] {
                if bucket == Bucket::Back && !behind {
                    continue;
                }
                let Some(targets) = self.oit.bucket_targets(bucket) else { continue };
                translucency.record_order_independent(rec, inputs, scene, textures, avboit, depth_array, targets, bucket);
            }
        }
        if behind {
            self.oit.record_composite(rec, inputs, Bucket::Back, self.volumes.resolved());
        }
    }

    /// The AVBOIT grid: bounds, transmittance, resolve.
    fn record_avboit_grid(&self, rec: &mut Recorder, inputs: &LightingInputs, scene: &GpuScene, translucency: &Translucency, textures: &TextureLibrary, translucent_depth: &wgpu::TextureView) {
        let avboit = &self.oit.avboit;
        avboit.record_clear(rec);
        for stage in [AvboitStage::Bounds, AvboitStage::Transmittance] {
            if stage == AvboitStage::Transmittance {
                self.volumes.record_avboit(rec, inputs, translucent_depth, avboit);
            }
            let device = rec.device;
            let label = if stage == AvboitStage::Bounds { "avboit bounds" } else { "avboit transmittance" };
            let Some(mut pass) = avboit.low_resolution_pass(rec, label) else { return };
            translucency.draw_avboit(device, &mut pass, inputs, scene, textures, avboit, translucent_depth, stage);
            self.particles.draw_avboit(device, &mut pass, inputs, translucent_depth, avboit, stage);
            drop(pass);
            if stage == AvboitStage::Bounds {
                avboit.record_dilate(rec);
            }
        }
        avboit.record_resolve(rec);
    }

    /// After the translucency pass: the composite of everything in front of it.
    pub fn record_after_translucency(&self, rec: &mut Recorder, inputs: &LightingInputs) {
        if self.active {
            self.oit.record_composite(rec, inputs, Bucket::Front, self.volumes.resolved());
        }
    }
}
