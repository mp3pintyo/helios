//! Local lights: GPU upload and clustered (froxel) culling (see `shaders/lights.wgsl`).

use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use wgpu::ShaderStages;

use crate::profiler::Recorder;
use crate::scene::{LightKind, Scene};
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer, workgroups};

pub const MAX_LIGHTS: usize = 1024;
const CLUSTERS: u32 = 16 * 9 * 24;
const CLUSTER_WORDS: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuLight {
    position_range: Vec4,
    color: Vec4,
    direction: Vec4,
    params: Vec4,
}

pub struct LightCulling {
    pub lights: wgpu::Buffer,
    pub grid: wgpu::Buffer,
    pub count: u32,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl LightCulling {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "light_cull.wgsl");
        let layout = bind_group_layout(
            device,
            "light cull",
            ShaderStages::COMPUTE,
            &[(0, bind::uniform()), (1, bind::storage(true)), (2, bind::storage(false))],
        );
        Self {
            lights: storage_buffer(device, "lights", (MAX_LIGHTS * std::mem::size_of::<GpuLight>()) as u64, wgpu::BufferUsages::empty()),
            grid: storage_buffer(device, "light grid", (CLUSTERS * CLUSTER_WORDS) as u64 * 4, wgpu::BufferUsages::empty()),
            count: 0,
            pipeline: compute_pipeline(device, "light cull", &module, "cull_lights", &layout, &[]),
            layout,
        }
    }

    /// Uploads the scene's lights (cheap; lights may animate every frame) with their virtual
    /// shadow map slots (`u32::MAX`: none, the light gets distance-field shadows).
    pub fn upload(&mut self, queue: &wgpu::Queue, scene: &Scene, shadow_slots: &[u32]) {
        let lights: Vec<GpuLight> = scene
            .lights()
            .iter()
            .take(MAX_LIGHTS)
            .enumerate()
            .map(|(i, l)| {
                let (direction, cos_outer, cos_inner, kind) = match l.kind {
                    LightKind::Point => (glam::Vec3::NEG_Y, -1.0, -1.0, 0.0),
                    LightKind::Spot { direction, inner, outer } => (direction.normalize(), outer.cos(), inner.cos(), 1.0),
                };
                GpuLight {
                    position_range: l.position.extend(l.range),
                    color: (l.color * l.intensity).extend(l.source_radius),
                    direction: direction.extend(cos_outer),
                    params: Vec4::new(cos_inner, kind, l.cast_shadows as u32 as f32, shadow_slots.get(i).map_or(0.0, |&s| s.wrapping_add(1) as f32)),
                }
            })
            .collect();
        self.count = lights.len() as u32;
        if !lights.is_empty() {
            queue.write_buffer(&self.lights, 0, bytemuck::cast_slice(&lights));
        }
    }

    pub fn record(&self, rec: &mut Recorder, frame: &wgpu::Buffer) {
        if self.count == 0 {
            return;
        }
        let group = bind_group(
            rec.device,
            "light cull",
            &self.layout,
            &[(0, frame.as_entire_binding()), (1, self.lights.as_entire_binding()), (2, self.grid.as_entire_binding())],
        );
        let mut pass = rec.compute_pass("light culling");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(CLUSTERS, 64), 1, 1);
    }
}
