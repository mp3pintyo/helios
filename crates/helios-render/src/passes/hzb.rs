//! Hierarchical-Z pyramid construction from the early-pass depth buffer.

use wgpu::{BindingType, BufferBindingType, ShaderStages, TextureFormat};

use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, workgroups};

/// Uniform offset alignment: one [`HzbLevel`] per mip at this stride (dynamic offsets).
const LEVEL_STRIDE: u64 = 256;
/// More mips than any target has (a 2^16-texel side has 17).
const MAX_LEVELS: u64 = 17;

/// The part of a mip the internal resolution covers (`HzbLevel` in hzb.wgsl).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HzbLevel {
    source_limit: [u32; 2],
    size: [u32; 2],
}

pub struct Hzb {
    from_depth: wgpu::ComputePipeline,
    downsample: wgpu::ComputePipeline,
    depth_layout: wgpu::BindGroupLayout,
    down_layout: wgpu::BindGroupLayout,
    levels: wgpu::Buffer,
}

impl Hzb {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "hzb.wgsl");
        let level = BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None };
        let depth_layout = bind_group_layout(
            device,
            "hzb from depth",
            ShaderStages::COMPUTE,
            &[(0, bind::texture(false)), (2, bind::storage_texture(TextureFormat::R32Float)), (3, level)],
        );
        let down_layout = bind_group_layout(
            device,
            "hzb downsample",
            ShaderStages::COMPUTE,
            &[(1, bind::texture(false)), (2, bind::storage_texture(TextureFormat::R32Float)), (3, level)],
        );
        Self {
            from_depth: compute_pipeline(device, "hzb from depth", &module, "from_depth", &depth_layout, &[]),
            downsample: compute_pipeline(device, "hzb downsample", &module, "downsample", &down_layout, &[]),
            depth_layout,
            down_layout,
            levels: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hzb levels"),
                size: LEVEL_STRIDE * MAX_LEVELS,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    /// Covered size of each mip for the current internal resolution, starting with the depth
    /// buffer (texel (x, y) of mip L covers pixels [x, x + 1) · 2^(L+1)).
    fn covered_sizes(targets: &FrameTargets) -> impl Iterator<Item = (u32, u32)> {
        std::iter::successors(Some((targets.width, targets.height)), |&(w, h)| Some((w.div_ceil(2).max(1), h.div_ceil(2).max(1))))
            .take(targets.hzb_mips.len() + 1)
    }

    /// Writes the covered region of every mip for this frame's internal resolution.
    pub fn update(&self, queue: &wgpu::Queue, targets: &FrameTargets) {
        let sizes: Vec<(u32, u32)> = Self::covered_sizes(targets).collect();
        let mut data = vec![0u8; (LEVEL_STRIDE * MAX_LEVELS) as usize];
        for (mip, pair) in sizes.windows(2).enumerate().take(MAX_LEVELS as usize) {
            let (source, size) = (pair[0], pair[1]);
            let level = HzbLevel { source_limit: [source.0 - 1, source.1 - 1], size: [size.0, size.1] };
            let offset = mip * LEVEL_STRIDE as usize;
            data[offset..offset + std::mem::size_of::<HzbLevel>()].copy_from_slice(bytemuck::bytes_of(&level));
        }
        queue.write_buffer(&self.levels, 0, &data[..sizes.len().saturating_sub(1) * LEVEL_STRIDE as usize]);
    }

    pub fn record(&self, rec: &mut Recorder, targets: &FrameTargets) {
        let device = rec.device;
        let mut pass = rec.compute_pass("hzb");
        let level = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.levels,
            offset: 0,
            size: wgpu::BufferSize::new(std::mem::size_of::<HzbLevel>() as u64),
        });
        let sizes: Vec<(u32, u32)> = Self::covered_sizes(targets).skip(1).collect();
        for (mip, destination) in targets.hzb_mips.iter().enumerate() {
            let group = if mip == 0 {
                pass.set_pipeline(&self.from_depth);
                bind_group(
                    device,
                    "hzb",
                    &self.depth_layout,
                    &[
                        (0, wgpu::BindingResource::TextureView(&targets.depth)),
                        (2, wgpu::BindingResource::TextureView(destination)),
                        (3, level.clone()),
                    ],
                )
            } else {
                pass.set_pipeline(&self.downsample);
                bind_group(
                    device,
                    "hzb",
                    &self.down_layout,
                    &[
                        (1, wgpu::BindingResource::TextureView(&targets.hzb_mips[mip - 1])),
                        (2, wgpu::BindingResource::TextureView(destination)),
                        (3, level.clone()),
                    ],
                )
            };
            pass.set_bind_group(0, &group, &[(mip as u64 * LEVEL_STRIDE) as u32]);
            let (w, h) = sizes[mip];
            pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
        }
    }
}
