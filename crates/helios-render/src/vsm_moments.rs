//! Prefiltered distant sun shadows (round 8; UE 5.8's experimental "Prefiltered Distant" virtual
//! shadow maps): distant clipmap levels are rendered at a lower resolution
//! ([`crate::RenderSettings::shadow_distant_bias`]: receivers that would sample a distant level
//! take a coarser one) and turned into a prefiltered **moment shadow map**, which the lighting
//! pass filters linearly at the receiver's footprint (`shaders/vsm_moments.wgsl`,
//! `shaders/vsm_prefiltered.wgsl`).
//!
//! The moment atlas mirrors the depth atlas' tile layout at half its resolution: every physical
//! page has a 64² tile of four depth moments (rgba16float, each texel the mean of 2×2 depth
//! texels), mips 1 and 2 of it for wider filters, and in mip 6 (one texel per tile) the page's depth
//! range, to which its moments are normalised (half precision would otherwise blur contact
//! shadows by metres over the scene's depth range). Only distant pages rendered this frame are
//! converted, after the page rasters; cached pages keep their moments like their depth.

use wgpu::BindingResource::TextureView;
use wgpu::{ShaderStages, TextureFormat};

use crate::pipelines::Lazy;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, mip_view};

const FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// Moment tile side per page (half the depth page's 128 texels).
pub(crate) const MOMENT_TILE: u32 = 64;
/// Mips 0..=6: the filter mips (0..=2) and the per-tile range (6: one texel per tile).
const MIPS: u32 = 7;
const RANGE_MIP: u32 = 6;

/// The moment atlas and the pass that fills it.
pub struct VsmMoments {
    layout: wgpu::BindGroupLayout,
    build: Lazy<wgpu::ComputePipeline>,
    tiles_per_row: u32,
    /// Created when prefiltering is first switched on (43 MiB).
    atlas: Option<MomentAtlas>,
}

struct MomentAtlas {
    /// All mips, for sampling.
    view: wgpu::TextureView,
    /// Mips 0, 1, 2 and the range mip, for the build pass.
    mips: [wgpu::TextureView; 4],
}

/// What the build pass reads: the VSM uniform, the page table, the physical pages' owners and the
/// depth atlas.
pub struct MomentInputs<'a> {
    pub vsm: &'a wgpu::Buffer,
    pub entries: &'a wgpu::Buffer,
    pub owner: &'a wgpu::Buffer,
    pub depth: &'a wgpu::TextureView,
    pub physical_count: u32,
}

impl VsmMoments {
    pub fn new(device: &wgpu::Device, tiles_per_row: u32) -> Self {
        let out = || bind::storage_texture(FORMAT);
        let layout = bind_group_layout(
            device,
            "vsm moments",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::storage(true)),
                (2, bind::storage(true)),
                (3, bind::texture_depth()),
                (4, out()),
                (5, out()),
                (6, out()),
                (7, out()),
            ],
        );
        let module = create_module(device, "vsm_moments.wgsl");
        Self { build: compute_pipeline(device, "vsm moments", &module, "build", &layout, &[]), layout, tiles_per_row, atlas: None }
    }

    /// Creates the atlas (the first time prefiltering is on).
    pub fn ensure(&mut self, device: &wgpu::Device) {
        if self.atlas.is_some() {
            return;
        }
        let size = self.tiles_per_row * MOMENT_TILE;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vsm moments"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mips = [0, 1, 2, RANGE_MIP].map(|mip| mip_view(&texture, mip));
        self.atlas = Some(MomentAtlas { view: texture.create_view(&Default::default()), mips });
    }

    /// The moment atlas (all mips), once created.
    pub fn view(&self) -> Option<&wgpu::TextureView> {
        self.atlas.as_ref().map(|a| &a.view)
    }

    /// Converts the distant pages rendered this frame (record after the page rasters).
    pub fn record(&self, rec: &mut Recorder, inputs: &MomentInputs) {
        let Some(atlas) = &self.atlas else { return };
        let group = bind_group(
            rec.device,
            "vsm moments",
            &self.layout,
            &[
                (0, inputs.vsm.as_entire_binding()),
                (1, inputs.entries.as_entire_binding()),
                (2, inputs.owner.as_entire_binding()),
                (3, TextureView(inputs.depth)),
                (4, TextureView(&atlas.mips[0])),
                (5, TextureView(&atlas.mips[1])),
                (6, TextureView(&atlas.mips[2])),
                (7, TextureView(&atlas.mips[3])),
            ],
        );
        let mut pass = rec.compute_pass("vsm moments");
        pass.set_pipeline(&self.build);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(inputs.physical_count, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_mip_has_one_texel_per_tile() {
        assert_eq!(MOMENT_TILE >> RANGE_MIP, 1);
        assert!(RANGE_MIP < MIPS);
    }
}
