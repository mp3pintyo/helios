//! Assemblies on the GPU (Nanite assemblies; see [`helios_geometry::assembly`] and
//! `shaders/assembly.wgsl`): the tables that describe every assembly's parts, the assembly
//! instances, and the expansion pass that turns assemblies close enough to need them into part
//! instances in the tail of the instance buffer before anything is culled.
//!
//! Part instances are resolved every frame by the GPU, so memory holds each distinct part mesh
//! once and the CPU never iterates parts; their persistent visibility bits (two-pass occlusion)
//! are allocated per (assembly instance, part) after the scene's, so they stay valid whichever
//! slot a part takes. Scenes without assemblies allocate nothing and run no pass.
//!
//! Culling is hierarchical: every assembly instance has one more instance record, between the
//! scene's instances and the part slots, standing for all its parts (`INSTANCE_PART_LIST`). Views
//! test that record instead of each part, and its "pages" are the (part, page) pairs of its
//! parts, culled 64 to a workgroup (cull.wgsl).

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};
use helios_geometry::assembly::{parts_bounds, transition_error};

use crate::gpu_scene::MeshRange;
use crate::passes::culling::CullView;
use crate::profiler::Recorder;
use crate::scene::Scene;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer, storage_buffer_init, uniform_buffer};

/// Part instances a frame may expand (instance records after the scene's and the assembly
/// instances' culling records: 240 bytes each).
pub const PART_CAPACITY: u32 = 1 << 15;

/// `struct AssemblyParams` in assembly.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    static_count: u32,
    part_capacity: u32,
    instance_count: u32,
    threshold_scale: f32,
}

/// `struct Assembly` in assembly.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuAssembly {
    first_part: u32,
    part_count: u32,
    visibility_bits: u32,
    transition_error: f32,
    lod_sphere: Vec4,
    /// The most pages of any part: the culling record's (part, page) lanes per part.
    page_stride: u32,
    /// The largest scale of any part's transform (bounds the parts' instance scales).
    part_scale: f32,
    _pad: [u32; 2],
}

/// `struct AssemblyPart` in assembly.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPart {
    transform: Mat4,
    page_offset: u32,
    page_count: u32,
    cluster_offset: u32,
    cluster_count: u32,
    visibility: u32,
    _pad: [u32; 3],
}

/// `struct AssemblyInstance` in assembly.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuAssemblyInstance {
    instance: u32,
    assembly: u32,
    visibility_base: u32,
    _pad: u32,
}

pub struct Assemblies {
    layout: wgpu::BindGroupLayout,
    expand: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    assemblies: wgpu::Buffer,
    parts: wgpu::Buffer,
    instances: wgpu::Buffer,
    /// [0]: part slots handed out.
    counter: wgpu::Buffer,
    /// [0]: the scene's instance count, [1 + slot]: the scene instance owning part slot `slot`
    /// (`part_owners` in assembly.wgsl; read by the editor's picking and selection outline).
    owners: wgpu::Buffer,
    /// Per mesh: its assembly, if it is an assembly's base.
    assembly_of_mesh: Vec<Option<u32>>,
    /// Per assembly: its visibility bits per instance.
    visibility_bits: Vec<u32>,
    /// Per assembly: page batches its culling record takes in a view when expanded.
    batches: Vec<u32>,
    /// Page batches all assembly instances' records take in a view when all are expanded.
    batch_demand: u32,
    params: Params,
    /// Multiplier on the expansion's LOD threshold that keeps the parts requested within the
    /// slots (see [`Assemblies::update_budget`]).
    budget_scale: f32,
}

impl Assemblies {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "assembly.wgsl");
        let mut entries = vec![(0, bind::uniform()), (1, bind::uniform()), (2, bind::storage(false))];
        entries.extend((3..=5).map(|b| (b, bind::storage(true))));
        entries.extend([(6, bind::storage(false)), (7, bind::storage(false))]);
        let layout = bind_group_layout(device, "assemblies", wgpu::ShaderStages::COMPUTE, &entries);
        let empty = |label| storage_buffer(device, label, 16, wgpu::BufferUsages::empty());
        Self {
            expand: compute_pipeline(device, "assembly expand", &module, "expand", &layout, &[]),
            layout,
            params_buffer: uniform_buffer(device, "assembly params", std::mem::size_of::<Params>() as u64),
            assemblies: empty("assemblies"),
            parts: empty("assembly parts"),
            instances: empty("assembly instances"),
            // COPY_SRC: the part count is read back for statistics.
            counter: storage_buffer(device, "assembly part counter", 16, wgpu::BufferUsages::COPY_SRC),
            owners: no_owners(device),
            assembly_of_mesh: Vec::new(),
            visibility_bits: Vec::new(),
            batches: Vec::new(),
            batch_demand: 0,
            params: Params { static_count: 0, part_capacity: 0, instance_count: 0, threshold_scale: 1.0 },
            budget_scale: 1.0,
        }
    }

    /// Whether the scene has assemblies (part slots exist and the expansion runs).
    pub fn any(&self) -> bool {
        !self.visibility_bits.is_empty()
    }

    /// Part instance slots after the culling records.
    pub fn part_capacity(&self) -> u32 {
        if self.any() {
            PART_CAPACITY
        } else {
            0
        }
    }

    /// Culling records after the scene's instances: one per assembly instance.
    pub fn record_count(&self) -> u32 {
        self.params.instance_count
    }

    /// Page batches the culling records take in a view when every assembly instance is
    /// expanded (one per 64 (part, page) lanes).
    pub fn batch_demand(&self) -> u32 {
        self.batch_demand
    }

    /// Whether mesh `mesh` is an assembly's base.
    pub fn is_assembly(&self, mesh: u32) -> bool {
        self.assembly_of_mesh.get(mesh as usize).is_some_and(Option::is_some)
    }

    /// Uploads the assemblies' part tables (with the meshes' global ranges).
    pub fn upload_geometry(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, meshes: &[MeshRange]) {
        self.assembly_of_mesh = vec![None; scene.meshes.len()];
        self.visibility_bits.clear();
        self.batches.clear();
        let mut assemblies = Vec::new();
        let mut parts = Vec::new();
        for (base, list) in &scene.assemblies {
            let mut visibility = 0;
            let first_part = parts.len() as u32;
            let mut page_stride = 1;
            let mut part_scale = 0.0f32;
            for part in list {
                let range = meshes[part.mesh.0 as usize];
                page_stride = page_stride.max(range.page_count);
                part_scale = part_scale.max(helios_geometry::assembly::max_scale(&part.transform));
                parts.push(GpuPart {
                    transform: part.transform,
                    page_offset: range.page_offset,
                    page_count: range.page_count,
                    cluster_offset: range.cluster_offset,
                    cluster_count: range.cluster_count,
                    visibility,
                    _pad: [0; 3],
                });
                visibility += range.cluster_count;
            }
            let geometry: Vec<(&helios_geometry::VirtualMesh, Mat4)> = list.iter().map(|p| (&scene.meshes[p.mesh.0 as usize], p.transform)).collect();
            assemblies.push(GpuAssembly {
                first_part,
                part_count: list.len() as u32,
                visibility_bits: visibility,
                transition_error: transition_error(&scene.meshes[base.0 as usize]),
                lod_sphere: parts_bounds(&geometry).to_vec4(),
                page_stride,
                part_scale,
                _pad: [0; 2],
            });
            self.assembly_of_mesh[base.0 as usize] = Some(self.visibility_bits.len() as u32);
            self.visibility_bits.push(visibility);
            self.batches.push(record_batches(list.len() as u32, page_stride));
        }
        if !assemblies.is_empty() {
            let errors: Vec<String> = assemblies.iter().map(|a| format!("{:.3}", a.transition_error)).collect();
            log::info!("uploading {} assemblies with {} parts (parts drawn below errors of {} m)", assemblies.len(), parts.len(), errors.join(", "));
        }
        self.assemblies = storage_buffer_init(device, queue, "assemblies", bytemuck::cast_slice(&assemblies));
        self.parts = storage_buffer_init(device, queue, "assembly parts", bytemuck::cast_slice(&parts));
    }

    /// Uploads the assembly instances among the scene's `instances`, whose visibility bits end
    /// at `visibility_bits`; returns the end of the bits including the parts'.
    pub fn upload_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, visibility_bits: u32) -> u32 {
        let mut bits = visibility_bits;
        let mut records = Vec::new();
        self.batch_demand = 0;
        for (i, instance) in scene.instances.iter().enumerate() {
            if let Some(assembly) = self.assembly_of_mesh.get(instance.mesh.0 as usize).copied().flatten() {
                records.push(GpuAssemblyInstance { instance: i as u32, assembly, visibility_base: bits, _pad: 0 });
                bits += self.visibility_bits[assembly as usize];
                self.batch_demand += self.batches[assembly as usize];
            }
        }
        if !records.is_empty() {
            log::info!("{} assembly instances: culling records take up to {} page batches per view", records.len(), self.batch_demand);
        }
        self.params.static_count = scene.instances.len() as u32;
        self.params.part_capacity = self.part_capacity();
        self.params.instance_count = records.len() as u32;
        self.instances = storage_buffer_init(device, queue, "assembly instances", bytemuck::cast_slice(&records));
        self.owners = if records.is_empty() {
            no_owners(device)
        } else {
            // Part slots start after the culling records.
            let mut owners = vec![u32::MAX; 1 + PART_CAPACITY as usize];
            owners[0] = self.params.static_count + self.params.instance_count;
            crate::util::storage_buffer_init(device, queue, "assembly part owners", bytemuck::cast_slice(&owners))
        };
        bits
    }

    /// The part owner table (`part_owner.wgsl`).
    pub fn owners(&self) -> &wgpu::Buffer {
        &self.owners
    }

    /// Adapts the expansion to the parts the last read-back frame requested: when they near the
    /// slots, assemblies would keep their bases in whatever order the GPU happened to expand them,
    /// differently every frame (flicker). Instead the expansion's LOD threshold is raised just
    /// enough (part counts fall with its square, as with the renderer's LOD budget) and relaxed
    /// again with hysteresis, so the nearest assemblies are the ones drawn as parts.
    pub fn update_budget(&mut self, requested: u32) {
        let fill = requested as f32 / PART_CAPACITY as f32;
        if fill > 0.9 {
            self.budget_scale = (self.budget_scale * (fill / 0.8).sqrt().clamp(1.05, 2.0)).min(64.0);
        } else if fill < 0.6 && self.budget_scale > 1.0 {
            self.budget_scale = (self.budget_scale * 0.97).max(1.0);
        }
    }

    /// Writes this frame's expansion parameters.
    pub fn write_params(&self, queue: &wgpu::Queue) {
        if self.params.instance_count > 0 {
            queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&Params { threshold_scale: self.budget_scale, ..self.params }));
        }
    }

    /// Expands the assemblies the main `view` needs as parts into `instances`' part slots and
    /// writes their culling records.
    pub fn record(&self, rec: &mut Recorder, view: &CullView, instances: &wgpu::Buffer) {
        if self.params.instance_count == 0 {
            return;
        }
        let group = bind_group(
            rec.device,
            "assemblies",
            &self.layout,
            &[
                (0, view.uniform.as_entire_binding()),
                (1, self.params_buffer.as_entire_binding()),
                (2, instances.as_entire_binding()),
                (3, self.instances.as_entire_binding()),
                (4, self.assemblies.as_entire_binding()),
                (5, self.parts.as_entire_binding()),
                (6, self.counter.as_entire_binding()),
                (7, self.owners.as_entire_binding()),
            ],
        );
        rec.encoder.clear_buffer(&self.counter, 0, Some(4));
        let mut pass = rec.compute_pass("assembly expansion");
        pass.set_bind_group(0, &group, &[]);
        pass.set_pipeline(&self.expand);
        // One workgroup per assembly instance, 2D beyond the dispatch limit.
        let groups = self.params.instance_count;
        pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
    }

    /// The counter whose first word is the part slots requested this frame (statistics), if the
    /// scene has assemblies.
    pub fn part_counter(&self) -> Option<&wgpu::Buffer> {
        (self.params.instance_count > 0).then_some(&self.counter)
    }
}

/// Page batches (64 lanes each) of an expanded assembly's culling record: `page_stride` lanes
/// per part.
fn record_batches(parts: u32, page_stride: u32) -> u32 {
    (parts * page_stride).div_ceil(64)
}

/// The (part, page of the part) a culling record's lane stands for (cull.wgsl's `cull_pages`);
/// lanes past a part's page count are idle.
#[cfg(test)]
fn record_lane(lane: u32, page_stride: u32) -> (u32, u32) {
    (lane / page_stride, lane % page_stride)
}

/// The owner table of a scene without assemblies: no instance is a part.
fn no_owners(device: &wgpu::Device) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("assembly part owners"),
        contents: bytemuck::bytes_of(&u32::MAX),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_match_the_shader_layouts() {
        assert_eq!(std::mem::size_of::<Params>(), 16);
        assert_eq!(std::mem::size_of::<GpuAssembly>(), 48);
        assert_eq!(std::mem::size_of::<GpuPart>(), 96);
        assert_eq!(std::mem::size_of::<GpuAssemblyInstance>(), 16);
    }

    /// A culling record's batches visit every page of every part exactly once, whatever the
    /// parts' page counts (the stride is the largest).
    #[test]
    fn record_lanes_cover_every_part_page_once() {
        for page_counts in [vec![6; 176], vec![1, 7, 3, 64, 2], vec![65, 1], vec![1]] {
            let stride = *page_counts.iter().max().unwrap();
            let parts = page_counts.len() as u32;
            let mut seen = vec![vec![0; stride as usize]; page_counts.len()];
            let batches = record_batches(parts, stride);
            for b in 0..batches {
                // cull.wgsl: batch (start = 64 b, count = min(64, lanes - start)), lane start + lid.
                let count = 64.min(parts * stride - b * 64);
                for lid in 0..count {
                    let (part, page) = record_lane(b * 64 + lid, stride);
                    if page < page_counts[part as usize] {
                        seen[part as usize][page as usize] += 1;
                    }
                }
            }
            for (part, pages) in page_counts.iter().enumerate() {
                assert!(seen[part][..*pages as usize].iter().all(|&n| n == 1), "{page_counts:?}");
            }
        }
        // The forest's conifers: 176 sprays of 6 pages fill 17 batches instead of 176.
        assert_eq!(record_batches(176, 6), 17);
    }
}
