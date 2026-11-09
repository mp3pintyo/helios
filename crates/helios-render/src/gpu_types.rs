//! `#[repr(C)]` mirrors of the structs declared in `shaders/common.wgsl`.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuInstance {
    pub world: Mat4,
    pub prev_world: Mat4,
    pub normal_matrix: [Vec4; 3],
    /// World-space bounding sphere.
    pub bounds: Vec4,
    /// First global page of the instance's mesh.
    pub page_offset: u32,
    pub page_count: u32,
    /// First global cluster of the instance's mesh.
    pub cluster_offset: u32,
    pub cluster_count: u32,
    pub material: u32,
    /// First bit of this instance's clusters in the persistent visibility bitset.
    pub visibility_offset: u32,
    pub max_scale: f32,
    pub flags: u32,
    /// The material's [`crate::WindResponse`] (bend, sway, flutter, frequency); zero = static.
    pub wind: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMaterial {
    /// rgb: linear base colour, a: opacity (translucent materials).
    pub base_color: Vec4,
    /// xyz: emitted colour (independent of the base colour), w: emission as a multiple of it.
    pub emissive: Vec4,
    pub roughness: f32,
    pub metallic: f32,
    pub kind: u32,
    /// [`material_flags`].
    pub flags: u32,
    /// Texture array layer, or `u32::MAX`.
    pub texture: u32,
    pub uv_scale: f32,
    pub normal_strength: f32,
    pub mapping: u32,
    pub detail_scale: f32,
    pub detail_strength: f32,
    /// Terrain layer textures, one byte each (255 = none).
    pub layers: u32,
    /// Emission texture layer, or `u32::MAX`.
    pub emissive_texture: u32,
    /// Texture coordinates of `texture` (with `uv_scale` applied), `emissive_texture` and
    /// `occlusion_texture`, in that order.
    pub uv: [GpuUvTransform; 3],
    /// Ambient occlusion layer (its albedo alpha), or `u32::MAX`.
    pub occlusion_texture: u32,
    pub occlusion_strength: f32,
    pub _pad: [u32; 2],
}

/// A [`crate::UvTransform`]: `uv' = (dot(u, (uv, 1)), dot(v, (uv, 1)))` of the mesh UV set
/// `set`, then wrapped per axis (`wrap` = u mode | v mode << 4, [`crate::Wrap`] values).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuUvTransform {
    pub u: [f32; 3],
    pub uv_set: u32,
    pub v: [f32; 3],
    pub wrap: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CullViewUniform {
    pub view_proj: Mat4,
    pub view: Mat4,
    pub planes: [Vec4; 6],
    /// xyz: eye position, w: near plane.
    pub position: Vec4,
    pub viewport: Vec4,
    pub lod_scale: f32,
    pub lod_threshold: f32,
    pub orthographic: u32,
    pub instance_count: u32,
    pub work_capacity: u32,
    pub cluster_capacity: u32,
    pub hzb_mips: u32,
    pub frame_index: u32,
    /// Clusters whose average triangle is smaller than this (pixels) use the software rasteriser.
    pub sw_raster_threshold: f32,
    /// 0 = main view, 1 = shadow / virtual shadow map view.
    pub kind: u32,
    pub batch_capacity: u32,
    /// Capacity of the main view's alpha-masked hardware draw list (0 for shadow views).
    pub masked_capacity: u32,
    /// Virtual shadow map views: (level, page world size, absolute origin page x, y).
    pub vsm: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameUniform {
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub unjittered_view_proj: Mat4,
    pub prev_unjittered_view_proj: Mat4,
    pub view: Mat4,
    pub camera_position: Vec4,
    pub viewport: Vec4,
    pub jitter: Vec4,
    pub sun_direction: Vec4,
    pub sun_illuminance: Vec4,
    pub ground_albedo: Vec4,
    pub frame_index: u32,
    pub debug_view: u32,
    pub flags: u32,
    pub light_count: u32,
    pub exposure_compensation: f32,
    pub gi_intensity: f32,
    pub bloom_intensity: f32,
    pub time: f32,
    /// x: delta time, y: aerial perspective scale, z: GI ray length.
    pub misc: Vec4,
    /// The previous frame's internal viewport (width, height, 1 / width, 1 / height): histories
    /// at internal resolution were written into that corner of their targets.
    pub prev_viewport: Vec4,
    /// x, y: internal pixels per output pixel (the TSR upscale ratio, also the texture gradient
    /// scale that gives the negative mip bias); z, w: output width and height.
    pub upscale: Vec4,
}

/// Bit flags in [`GpuInstance::flags`].
pub mod instance_flags {
    pub const CAST_SHADOW: u32 = 1;
    /// The material is alpha-tested: culling bins hardware clusters into the masked draw.
    pub const ALPHA_MASKED: u32 = 2;
    /// The material is translucent: the main view's culling skips the instance (it is drawn by
    /// the forward translucency pass and must not occlude through the visibility buffer / HZB).
    pub const TRANSLUCENT: u32 = 4;
    /// Hidden (deleted in an editor): culled for every view.
    pub const HIDDEN: u32 = 8;
    /// [`crate::Mobility::Movable`]: a dynamic shadow caster, kept out of the shadow pages'
    /// static layers (`dynamic_caster` in wind.wgsl).
    pub const MOVABLE: u32 = 16;
    // Bits 12..32: the skin palette index (`skin::INSTANCE_SKIN_SHIFT`).
}

/// Bit flags in [`GpuMaterial::flags`].
pub mod material_flags {
    /// The texture is made from images: its data `w` channel holds metallic (procedural
    /// textures store height there) and texture values multiply the material's factors.
    pub const IMAGE_TEXTURE: u32 = 1;
    /// Bits 8..16: alpha cutoff (unorm8) of an alpha-tested material.
    pub const ALPHA_CUTOFF_SHIFT: u32 = 8;
    /// Bits 16..32: alpha mask layer + 1 (0: opaque).
    pub const ALPHA_MASK_SHIFT: u32 = 16;
    /// The ambient occlusion is the alpha of `texture` itself (same layer and coordinates), so
    /// it comes with the texture's sample.
    pub const OCCLUSION_IN_TEXTURE: u32 = 2;
}

/// Bit flags in [`FrameUniform::flags`].
pub mod flags {
    pub const SHADOWS: u32 = 1;
    pub const GI: u32 = 2;
    pub const TAA: u32 = 4;
    pub const AUTO_EXPOSURE: u32 = 8;
    pub const FOG: u32 = 16;
    pub const HISTORY_VALID: u32 = 32;
    /// The camera is inside a water body (exposure metering keeps Snell's window).
    pub const UNDERWATER: u32 = 64;
}

const _: () = {
    assert!(std::mem::size_of::<GpuInstance>() == 240);
    assert!(std::mem::size_of::<GpuMaterial>() == 192);
    assert!(std::mem::size_of::<CullViewUniform>() == 320);
    assert!(std::mem::size_of::<FrameUniform>() == 496);
    assert!(std::mem::size_of::<helios_geometry::GpuCluster>() == 80);
    assert!(std::mem::size_of::<helios_geometry::GpuPage>() == 64);
};
