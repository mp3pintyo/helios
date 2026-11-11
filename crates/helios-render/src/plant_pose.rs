//! The dynamic wind's pose pass (`shaders/plant_pose.wgsl`): UE evaluates a plant's bones once
//! per instance and skins its vertices; Helios did walk each vertex's bone chain in every pass
//! that places geometry (hardware and software rasterisers, shadow pages, the G-buffer resolve,
//! voxels). Now, once per frame, every plant instance whose *ranked* bones move — the bones that
//! hang from another and are not leaves at the end of their chain (branches; [`crate::plant_wind::bone_links`])
//! — takes one pose entry per ranked bone from a budgeted pool ([`POSE_CAPACITY`]): the bone's
//! rotation and the rigid transform its ancestors' frames at its attachment compose to, for this
//! frame and the previous one. Vertices then pose with one lookup (`plant_wind.wgsl`).
//!
//! Distance level of detail: an instance takes entries only while its branches are within their
//! fade (or it is promoted by the simulation, or has ranked bones that never fade); beyond, every
//! ranked bone is at rest and vertices ride on the root's frame at their chain's attachment.
//! Leaves (most of a tree's bones) take no entries: a vertex evaluates its leaf's flutter itself.
//!
//! The entries share one `Rgba32Float` texture with the simulation's deviations (binding
//! [`BINDING_PLANT_MOTION`]: the simulation's rows first, `plant_sim.rs`), so the rasterisers
//! bind no more than before.

use crate::pipelines::Lazy;
use crate::plant_sim::SIM_SLOTS;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, storage_buffer, uniform_buffer};

/// Whether the browser's background compile starts with the rasterisers carrying the plant code
/// ([`expect_plant_bones`]).
static PLANTS_EXPECTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Which rasteriser set (`PLANTS` in plant_wind.wgsl) renderers created from now on compile in
/// the background from the start — in the browser, while the first scene is baked: with the
/// plant code for a scene with plant bones, without it otherwise (the default). The other set
/// is made when a scene first needs it. Only the first image's wait depends on it.
pub fn expect_plant_bones(expected: bool) {
    PLANTS_EXPECTED.store(expected, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn plant_bones_expected() -> bool {
    PLANTS_EXPECTED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Binding of the plant motion texture (simulated deviations, pose entries) in every shader that
/// positions geometry.
pub const BINDING_PLANT_MOTION: u32 = 171;
/// Pose entries in the pool (ranked bones of the instances near enough to move them).
pub const POSE_CAPACITY: u32 = 1 << 16;
/// Texels per entry: this frame's rotation vector, rotation (quaternion) and translation, the
/// previous frame's, then the bone's pivot and length and its axis (`PLANT_POSE_TEXELS` in
/// plant_wind.wgsl).
const POSE_TEXELS: u32 = 8;
/// Texels per row (`PLANT_TEXTURE_WIDTH`; also two per bone of a simulation slot).
const WIDTH: u32 = crate::plant_wind::PLANT_TEXTURE_WIDTH;

/// Layout entry of the plant motion texture.
pub fn motion_entry() -> (u32, wgpu::BindingType) {
    (BINDING_PLANT_MOTION, bind::texture(false))
}

/// The scene's plant motion texture for [`motion_entry`].
pub fn motion_resource(scene: &crate::gpu_scene::GpuScene) -> (u32, wgpu::BindingResource<'_>) {
    (BINDING_PLANT_MOTION, wgpu::BindingResource::TextureView(&scene.plant_pose.view))
}

/// `struct PoseParams` in plant_pose.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    plant_count: u32,
    part_first: u32,
    part_capacity: u32,
    capacity: u32,
}

pub struct PlantPose {
    layout: wgpu::BindGroupLayout,
    allocate: Lazy<wgpu::ComputePipeline>,
    evaluate: Lazy<wgpu::ComputePipeline>,
    params: wgpu::Buffer,
    plants: wgpu::Buffer,
    /// [0]: entries handed out this frame (read back for the statistics).
    pub(crate) counters: wgpu::Buffer,
    /// The evaluation's indirect dispatch, written by the allocation.
    args: wgpu::Buffer,
    /// Bound in its place to the evaluation (a buffer cannot be its indirect and storage buffer
    /// at once).
    spare: wgpu::Buffer,
    owners: wgpu::Buffer,
    /// Stands in for the assembly part counter in scenes without assemblies.
    no_parts: wgpu::Buffer,
    _texture: wgpu::Texture,
    /// The simulation's rows, then the pose entries.
    pub view: wgpu::TextureView,
    /// Scene instances with plant bones, part slots after them.
    plant_count: u32,
    part_capacity: u32,
}

impl PlantPose {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = bind_group_layout(
            device,
            "plant pose",
            wgpu::ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::storage(true)),
                (2, bind::storage(false)),
                (3, bind::storage(false)),
                (4, bind::storage(false)),
                (5, bind::storage(true)),
                (6, bind::storage(true)),
                (7, bind::storage_texture(wgpu::TextureFormat::Rgba32Float)),
                (8, bind::storage(false)),
                crate::wind::wind_entry(),
                crate::plant_wind::plant_entry(),
            ],
        );
        let module = create_module(device, "plant_pose.wgsl");
        let rows = SIM_SLOTS as u32 + (POSE_CAPACITY * POSE_TEXELS).div_ceil(WIDTH);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("plant motion"),
            size: wgpu::Extent3d { width: WIDTH, height: rows, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        Self {
            allocate: lazy_compute_pipeline(device, "plant pose allocate", &module, "allocate", &layout, &[]),
            evaluate: lazy_compute_pipeline(device, "plant pose evaluate", &module, "evaluate", &layout, &[]),
            layout,
            params: uniform_buffer(device, "plant pose params", std::mem::size_of::<Params>() as u64),
            plants: storage_buffer(device, "plant instances", 16, wgpu::BufferUsages::empty()),
            counters: storage_buffer(device, "plant pose counters", 16, wgpu::BufferUsages::COPY_SRC),
            args: storage_buffer(device, "plant pose dispatch", 16, wgpu::BufferUsages::INDIRECT),
            spare: storage_buffer(device, "plant pose spare", 16, wgpu::BufferUsages::empty()),
            owners: storage_buffer(device, "plant pose owners", POSE_CAPACITY as u64 * 4, wgpu::BufferUsages::empty()),
            no_parts: storage_buffer(device, "no assembly parts", 16, wgpu::BufferUsages::empty()),
            view: texture.create_view(&Default::default()),
            _texture: texture,
            plant_count: 0,
            part_capacity: 0,
        }
    }

    /// The scene's instances with plant bones (after the instances were uploaded) and where the
    /// assembly part slots are.
    pub(crate) fn set_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, plants: &[u32], part_first: u32, part_capacity: u32) {
        if self.plants.size() < (plants.len() as u64 * 4).max(16) {
            self.plants = storage_buffer(device, "plant instances", plants.len() as u64 * 4, wgpu::BufferUsages::empty());
        }
        if !plants.is_empty() {
            queue.write_buffer(&self.plants, 0, bytemuck::cast_slice(plants));
        }
        self.plant_count = plants.len() as u32;
        // Parts only matter where plants have bones.
        self.part_capacity = if plants.is_empty() { 0 } else { part_capacity };
        let params = Params { plant_count: self.plant_count, part_first, part_capacity: self.part_capacity, capacity: POSE_CAPACITY };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }

    /// Whether the scene has plants with bones (the pass runs while the wind blows).
    pub fn any(&self) -> bool {
        self.plant_count > 0
    }

    /// Hands out this frame's pose entries and evaluates them (after the assembly expansion and
    /// the simulation, before anything draws).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &self,
        rec: &mut Recorder,
        instances: &wgpu::Buffer,
        wind: &wgpu::Buffer,
        bones: &wgpu::TextureView,
        sim_state: &wgpu::Buffer,
        part_counter: Option<&wgpu::Buffer>,
    ) {
        if !self.any() {
            return;
        }
        let group = |args: &wgpu::Buffer| {
            bind_group(
                rec.device,
                "plant pose",
                &self.layout,
                &[
                    (0, self.params.as_entire_binding()),
                    (1, self.plants.as_entire_binding()),
                    (2, instances.as_entire_binding()),
                    (3, self.counters.as_entire_binding()),
                    (4, self.owners.as_entire_binding()),
                    (5, sim_state.as_entire_binding()),
                    (6, part_counter.unwrap_or(&self.no_parts).as_entire_binding()),
                    (7, wgpu::BindingResource::TextureView(&self.view)),
                    (8, args.as_entire_binding()),
                    (crate::wind::BINDING_WIND, wind.as_entire_binding()),
                    (crate::plant_wind::BINDING_PLANT_BONES, wgpu::BindingResource::TextureView(bones)),
                ],
            )
        };
        let (allocate, evaluate) = (group(&self.args), group(&self.spare));
        rec.encoder.clear_buffer(&self.counters, 0, None);
        rec.encoder.clear_buffer(&self.args, 0, None);
        let mut pass = rec.compute_pass("plant pose");
        pass.set_pipeline(&self.allocate);
        pass.set_bind_group(0, &allocate, &[]);
        pass.dispatch_workgroups((self.plant_count + self.part_capacity).div_ceil(64), 1, 1);
        pass.set_pipeline(&self.evaluate);
        pass.set_bind_group(0, &evaluate, &[]);
        pass.dispatch_workgroups_indirect(&self.args, 0);
    }
}
