//! Resolution-dependent render targets.
//!
//! Two resolutions meet here. The scene renders at the *internal* resolution (a screen
//! percentage of the output, see `resolution.rs`) into targets allocated at `capacity`, the
//! largest internal size the settings allow: each frame occupies their top-left
//! `width × height` texels, so dynamic resolution changes reallocate nothing. Temporal super
//! resolution reconstructs the *output* resolution into `history`, which post-processing,
//! bloom and tone mapping work on.

use wgpu::{TextureFormat as F, TextureUsages as U};

use crate::util::{mip_count, mip_view, texture, TextureSpec};

pub const DEPTH_FORMAT: F = F::Depth32Float;
pub const VISIBILITY_FORMAT: F = F::R32Uint;
pub const HDR_FORMAT: F = F::Rgba16Float;
/// TSR per-pixel history state: dilated motion (output pixels) and the thin-feature lock.
pub const HISTORY_MOTION_FORMAT: F = F::Rgba16Float;
pub const BLOOM_LEVELS: u32 = 6;

pub struct FrameTargets {
    /// Internal resolution of the current frame (at most `capacity`); passes dispatch over it.
    pub width: u32,
    pub height: u32,
    /// Size the internal-resolution targets are allocated at.
    pub capacity: (u32, u32),
    /// Output resolution (TSR history, post-processing, bloom).
    pub output: (u32, u32),
    /// Hardware raster attachments.
    pub hw_depth: wgpu::TextureView,
    pub hw_visibility: wgpu::TextureView,
    /// Software raster target: depth bits then payload, one u32 per pixel each (rows of the
    /// current internal width).
    pub sw_target: wgpu::Buffer,
    /// Merged (hardware + software) reverse-Z depth and visibility used by all later passes.
    pub depth: wgpu::TextureView,
    pub visibility: wgpu::TextureView,
    /// Full HZB mip chain (sampled) and one view per mip (storage).
    pub hzb: wgpu::TextureView,
    pub hzb_mips: Vec<wgpu::TextureView>,
    /// Per-resolve-slot tile lists of the material classification.
    pub material_tiles: wgpu::Buffer,
    pub gbuffer_albedo: wgpu::TextureView,
    pub gbuffer_normal: wgpu::TextureView,
    pub gbuffer_motion: wgpu::TextureView,
    pub hdr: wgpu::TextureView,
    /// TSR output ping-pong at output resolution (rgb; a: accumulated sample weight); also the
    /// previous frame's radiance for Lumen's screen traces.
    pub history: [wgpu::TextureView; 2],
    /// TSR: the motion each history pixel was reprojected with (disocclusion test) and its
    /// thin-feature lock.
    pub history_motion: [wgpu::TextureView; 2],
    pub gi_raw: wgpu::TextureView,
    pub gi_history: [wgpu::TextureView; 2],
    /// Per pixel, the visibility (unorm8 x 4) of its four strongest lights with virtual shadow
    /// maps (lighting.wgsl, local_shadows).
    pub local_shadow_mask: wgpu::Buffer,
    pub bloom_down: Vec<wgpu::TextureView>,
    pub bloom_up: Vec<wgpu::TextureView>,
}

impl FrameTargets {
    /// Internal targets of `capacity` (the largest internal resolution) and output targets of
    /// `output`. The internal size starts at the capacity.
    pub fn new(device: &wgpu::Device, capacity: (u32, u32), output: (u32, u32)) -> Self {
        let (width, height) = (capacity.0.max(1), capacity.1.max(1));
        let output = (output.0.max(1), output.1.max(1));
        let make = |label: &str, w: u32, h: u32, mips: u32, format: F, usage: U| {
            texture(device, TextureSpec { label, width: w, height: h, layers: 1, mips, format, usage })
        };
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let storage = U::STORAGE_BINDING | U::TEXTURE_BINDING;

        let hw_depth = make("hw depth", width, height, 1, DEPTH_FORMAT, U::RENDER_ATTACHMENT | U::TEXTURE_BINDING);
        let hw_visibility = make("hw visibility", width, height, 1, VISIBILITY_FORMAT, U::RENDER_ATTACHMENT | U::TEXTURE_BINDING);
        let depth = make("depth", width, height, 1, F::R32Float, storage);
        let visibility = make("visibility", width, height, 1, VISIBILITY_FORMAT, storage);
        let sw_target = crate::util::storage_buffer(device, "sw raster target", width as u64 * height as u64 * 8, wgpu::BufferUsages::empty());

        let (hzb_w, hzb_h) = (width.div_ceil(2), height.div_ceil(2));
        let hzb_levels = mip_count(hzb_w, hzb_h);
        let hzb = make("hzb", hzb_w, hzb_h, hzb_levels, F::R32Float, storage);

        let (half_w, half_h) = (width.div_ceil(2), height.div_ceil(2));
        let (out_half_w, out_half_h) = (output.0.div_ceil(2), output.1.div_ceil(2));
        let bloom_levels = BLOOM_LEVELS.min(mip_count(out_half_w, out_half_h));
        let bloom_down = make("bloom down", out_half_w, out_half_h, bloom_levels, HDR_FORMAT, storage);
        let bloom_up = make("bloom up", out_half_w, out_half_h, bloom_levels, HDR_FORMAT, storage);

        let history = [0, 1].map(|i| view(&make(&format!("history {i}"), output.0, output.1, 1, HDR_FORMAT, storage)));
        let history_motion = [0, 1].map(|i| view(&make(&format!("history motion {i}"), output.0, output.1, 1, HISTORY_MOTION_FORMAT, storage)));
        let gi_history = [0, 1].map(|i| view(&make(&format!("gi history {i}"), half_w, half_h, 1, HDR_FORMAT, storage)));

        Self {
            width,
            height,
            capacity: (width, height),
            output,
            hw_depth: view(&hw_depth),
            hw_visibility: view(&hw_visibility),
            sw_target,
            depth: view(&depth),
            visibility: view(&visibility),
            hzb: view(&hzb),
            hzb_mips: (0..hzb_levels).map(|m| mip_view(&hzb, m)).collect(),
            material_tiles: crate::util::storage_buffer(
                device,
                "material tiles",
                // Counts and dispatch arguments, then a tile list per resolve slot (classify.wgsl).
                (80 + 16 * (width.div_ceil(8) * height.div_ceil(8)) as u64) * 4,
                wgpu::BufferUsages::INDIRECT,
            ),
            gbuffer_albedo: view(&make("gbuffer albedo", width, height, 1, F::Rgba8Unorm, storage)),
            gbuffer_normal: view(&make("gbuffer normal", width, height, 1, F::Rgba16Float, storage)),
            // The translucency pass renders into the HDR image and the motion vectors, and copies
            // the HDR image (the opaque scene it refracts).
            gbuffer_motion: view(&make("gbuffer motion", width, height, 1, F::Rgba16Float, storage | U::RENDER_ATTACHMENT)),
            hdr: view(&make("hdr", width, height, 1, HDR_FORMAT, storage | U::RENDER_ATTACHMENT | U::COPY_SRC)),
            history,
            history_motion,
            gi_raw: view(&make("gi raw", half_w, half_h, 1, HDR_FORMAT, storage)),
            gi_history,
            local_shadow_mask: crate::util::storage_buffer(device, "local shadow mask", width as u64 * height as u64 * 4, wgpu::BufferUsages::empty()),
            bloom_down: (0..bloom_levels).map(|m| mip_view(&bloom_down, m)).collect(),
            bloom_up: (0..bloom_levels).map(|m| mip_view(&bloom_up, m)).collect(),
        }
    }

    /// Sets the internal resolution of the next frame (clamped to the capacity).
    pub fn set_internal_size(&mut self, (width, height): (u32, u32)) {
        self.width = width.clamp(1, self.capacity.0);
        self.height = height.clamp(1, self.capacity.1);
    }

    pub fn hzb_mip_count(&self) -> u32 {
        self.hzb_mips.len() as u32
    }
}
