//! The simulated tier of the dynamic wind (UE 5.9 Dynamic Wind's promoted instances; the motion
//! is `shaders/plant_sim.wgsl`).
//!
//! Plants share the parametric motion (`plant_wind.rs`) until a force reaches them: small plants
//! (at most [`SIM_BONES`] bones: grass, ferns, bushes) touched by the push capsule around the
//! camera, or passed by a gust burst's ring, are *promoted* to a simulation slot of their own —
//! at most [`SIM_SLOTS`] at a time — until their springs have settled. A slot holds a spring state per bone: its deviation
//! from the parametric rotation, pushed by a capsule that follows the camera (walking through a
//! fern parts its fronds, which spring back) and by the gust ring. Undisturbed, the deviation is
//! zero, so promotion and demotion never pop. A promoted instance's culling bounds grow by the
//! largest deviation the simulation allows: each bone deviates at most [`SIM_DEVIATION`] times
//! its parametric bound, so the procedural layers bounding it are scaled by 1 + that.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};

use crate::pipelines::Lazy;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, storage_buffer, uniform_buffer};

/// Simulation slots (promoted instances) at a time: UE's default budget.
pub const SIM_SLOTS: usize = 256;
/// Bones a promoted plant may have (one texture row of two texels a bone per slot).
pub const SIM_BONES: u32 = 512;
/// A bone's simulated deviation stays within this many times its parametric bound.
pub const SIM_DEVIATION: f32 = 3.0;
/// A promoted plant is demoted this long after the last force reached it (its springs have
/// settled: their decay time is a few tenths of a second).
const SETTLE_TIME: f32 = 2.0;
/// The push capsule: from the camera this far down, this thick (a walker's legs and body).
const CAPSULE_LENGTH: f32 = 2.2;
const CAPSULE_RADIUS: f32 = 0.45;
/// A gust burst: a ring travelling out at this speed, this wide, for this long, bending what it
/// passes by this many parametric bounds.
const GUST_SPEED: f32 = 7.0;
const GUST_WIDTH: f32 = 1.6;
const GUST_DURATION: f32 = 3.5;
const GUST_STRENGTH: f32 = 2.5;

/// `struct SimParams` in plant_sim.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    capsule: Vec4,
    time: Vec4,
    gust: Vec4,
    gust_shape: Vec4,
}

/// `struct SimSlot` in plant_sim.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Slot {
    instance: u32,
    bones: u32,
    fresh: u32,
    _pad: u32,
}

/// An instance that may be promoted: its index, bounds centre and bone count.
#[derive(Clone, Copy)]
pub(crate) struct Candidate {
    pub index: u32,
    pub center: Vec3,
    pub radius: f32,
    pub bones: u32,
}

pub struct PlantSim {
    layout: wgpu::BindGroupLayout,
    pipeline: Lazy<wgpu::ComputePipeline>,
    params: wgpu::Buffer,
    slots_buffer: wgpu::Buffer,
    state: wgpu::Buffer,
    candidates: Vec<Candidate>,
    /// Per slot: the promoted instance.
    slots: Vec<Option<Candidate>>,
    slot_of: HashMap<u32, usize>,
    fresh: Vec<bool>,
    /// When a force last reached each plant (promoted until [`SETTLE_TIME`] later).
    reached: HashMap<u32, f32>,
    /// A gust burst: centre and start time.
    gust: Option<(Vec3, f32)>,
    /// Turned off: nothing is promoted (`RenderSettings::wind_simulation`).
    pub enabled: bool,
}

impl PlantSim {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = bind_group_layout(
            device,
            "plant sim",
            wgpu::ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::storage(true)),
                (2, bind::storage(true)),
                (3, bind::storage(false)),
                (4, bind::storage_texture(wgpu::TextureFormat::Rgba32Float)),
                crate::wind::wind_entry(),
                crate::plant_wind::plant_entry(),
            ],
        );
        let module = create_module(device, "plant_sim.wgsl");
        Self {
            pipeline: lazy_compute_pipeline(device, "plant sim", &module, "simulate", &layout, &[]),
            layout,
            params: uniform_buffer(device, "plant sim params", std::mem::size_of::<Params>() as u64),
            slots_buffer: storage_buffer(device, "plant sim slots", (SIM_SLOTS * std::mem::size_of::<Slot>()) as u64, wgpu::BufferUsages::empty()),
            // Per (slot, bone): δ, its rate, the previous frame's δ (the pose pass reads them).
            state: storage_buffer(device, "plant sim state", SIM_SLOTS as u64 * SIM_BONES as u64 * 48, wgpu::BufferUsages::empty()),
            candidates: Vec::new(),
            slots: vec![None; SIM_SLOTS],
            slot_of: HashMap::new(),
            fresh: vec![false; SIM_SLOTS],
            reached: HashMap::new(),
            gust: None,
            enabled: true,
        }
    }

    /// The instances that may be promoted (after the instances were uploaded). Slots of
    /// instances no longer among them are freed.
    pub(crate) fn set_candidates(&mut self, candidates: Vec<Candidate>) {
        self.candidates = candidates;
        self.slots = vec![None; SIM_SLOTS];
        self.slot_of.clear();
        self.reached.clear();
    }

    /// The slot of instance `index`, if promoted.
    pub fn slot(&self, index: u32) -> Option<u32> {
        self.slot_of.get(&index).map(|&s| s as u32)
    }

    /// Starts a gust burst at `center` (world) at `time`.
    pub fn gust(&mut self, center: Vec3, time: f32) {
        self.gust = Some((center, time));
    }

    /// Promotes and demotes for this frame; returns the instances whose slot changed. A plant
    /// is promoted while a force reaches it — the push capsule around the camera touches its
    /// bounds, or the gust ring passes it — and stays promoted until its springs have settled
    /// ([`SETTLE_TIME`] later), so demotion never pops. Over the budget, the most recently
    /// reached nearest plants win.
    pub fn update(&mut self, camera: Vec3, time: f32) -> Vec<u32> {
        if self.gust.is_some_and(|(_, start)| time - start > GUST_DURATION) {
            self.gust = None;
        }
        let ring = self.gust.map(|(c, start)| (c, GUST_SPEED * (time - start).max(0.0)));
        let bottom = camera.y - CAPSULE_LENGTH;
        if self.enabled {
            for c in &self.candidates {
                let flat = Vec3::new(c.center.x - camera.x, 0.0, c.center.z - camera.z).length();
                let touched = flat - c.radius < CAPSULE_RADIUS + 0.2 && c.center.y - c.radius < camera.y && c.center.y + c.radius > bottom;
                let gusted = ring.is_some_and(|(g, r)| {
                    let d = Vec3::new(c.center.x - g.x, 0.0, c.center.z - g.z).length();
                    (d - r).abs() < c.radius + 2.0 * GUST_WIDTH
                });
                if touched || gusted {
                    self.reached.insert(c.index, time);
                }
            }
        }
        self.reached.retain(|_, t| time - *t < SETTLE_TIME);
        let by_index: HashMap<u32, Candidate> = self.candidates.iter().map(|c| (c.index, *c)).collect();
        let mut wanted: Vec<(f32, Candidate)> =
            self.reached.iter().filter_map(|(i, &t)| by_index.get(i).map(|c| ((time - t) + 0.1 * c.center.distance(camera), *c))).collect();
        wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
        wanted.truncate(SIM_SLOTS);
        let keep: HashMap<u32, Candidate> = wanted.iter().map(|w| (w.1.index, w.1)).collect();
        let mut changed = Vec::new();
        // Demote.
        for s in 0..SIM_SLOTS {
            if let Some(c) = self.slots[s] {
                if !keep.contains_key(&c.index) {
                    self.slots[s] = None;
                    self.slot_of.remove(&c.index);
                    changed.push(c.index);
                }
            }
            self.fresh[s] = false;
        }
        // Promote into free slots.
        let free_slots: Vec<usize> = (0..SIM_SLOTS).filter(|&s| self.slots[s].is_none()).collect();
        let mut free = free_slots.into_iter();
        for (_, c) in wanted {
            if self.slot_of.contains_key(&c.index) {
                continue;
            }
            let Some(s) = free.next() else { break };
            self.slots[s] = Some(c);
            self.slot_of.insert(c.index, s);
            self.fresh[s] = true;
            changed.push(c.index);
        }
        changed
    }

    /// Whether any instance is promoted (the pass runs).
    pub fn active(&self) -> bool {
        !self.slot_of.is_empty()
    }

    /// Writes this frame's parameters and slots.
    pub fn write(&self, queue: &wgpu::Queue, camera: Vec3, time: f32, dt: f32) {
        let (gust, start) = self.gust.unwrap_or((Vec3::ZERO, -1.0));
        let params = Params {
            capsule: camera.extend(CAPSULE_RADIUS),
            time: Vec4::new(time, dt, CAPSULE_LENGTH, SIM_DEVIATION),
            gust: gust.extend(start),
            gust_shape: Vec4::new(GUST_STRENGTH, GUST_SPEED, GUST_WIDTH, GUST_DURATION),
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let slots: Vec<Slot> = (0..SIM_SLOTS)
            .map(|s| match self.slots[s] {
                Some(c) => Slot { instance: c.index, bones: c.bones, fresh: self.fresh[s] as u32, _pad: 0 },
                None => Slot { instance: 0, bones: 0, fresh: 0, _pad: 0 },
            })
            .collect();
        queue.write_buffer(&self.slots_buffer, 0, bytemuck::cast_slice(&slots));
    }

    /// The spring states (per slot and bone: δ, its rate, the previous frame's δ).
    pub(crate) fn state(&self) -> &wgpu::Buffer {
        &self.state
    }

    /// Simulates every slot (one workgroup each; empty slots return at once) into the first
    /// [`SIM_SLOTS`] rows of the plant motion texture (`motion`, [`crate::plant_pose`]).
    pub(crate) fn record(&self, rec: &mut Recorder, instances: &wgpu::Buffer, wind: &wgpu::Buffer, bones: &wgpu::TextureView, motion: &wgpu::TextureView) {
        if !self.active() {
            return;
        }
        let used = self.slots.iter().rposition(Option::is_some).map_or(0, |s| s + 1) as u32;
        let group = bind_group(
            rec.device,
            "plant sim",
            &self.layout,
            &[
                (0, self.params.as_entire_binding()),
                (1, self.slots_buffer.as_entire_binding()),
                (2, instances.as_entire_binding()),
                (3, self.state.as_entire_binding()),
                (4, wgpu::BindingResource::TextureView(motion)),
                (crate::wind::BINDING_WIND, wind.as_entire_binding()),
                (crate::plant_wind::BINDING_PLANT_BONES, wgpu::BindingResource::TextureView(bones)),
            ],
        );
        let mut pass = rec.compute_pass("plant sim");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(used, 1, 1);
    }
}
