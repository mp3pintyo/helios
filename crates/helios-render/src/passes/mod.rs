//! Render passes. Each pass owns its pipelines and layouts and creates its (cheap) bind
//! groups at record time, so passes never hold stale references to resized targets or
//! re-uploaded scene buffers.

pub mod culling;
pub mod hzb;
pub mod local_exposure;
pub mod material;
pub mod post;
pub mod post_fx;
pub mod shading;
pub mod sky;
pub mod tsr;

/// Samplers shared by several passes.
pub struct Samplers {
    pub linear: wgpu::Sampler,
    pub shadow: wgpu::Sampler,
    /// Repeating, trilinear, 16x anisotropic: material textures.
    pub material: wgpu::Sampler,
}

impl Samplers {
    pub fn new(device: &wgpu::Device) -> Self {
        let clamp = wgpu::AddressMode::ClampToEdge;
        Self {
            linear: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("linear clamp"),
                address_mode_u: clamp,
                address_mode_v: clamp,
                address_mode_w: clamp,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            material: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("material"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: 16,
                ..Default::default()
            }),
            shadow: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("shadow comparison"),
                address_mode_u: clamp,
                address_mode_v: clamp,
                address_mode_w: clamp,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                // Reverse-Z: a receiver is lit when it is at least as close to the light.
                compare: Some(wgpu::CompareFunction::GreaterEqual),
                ..Default::default()
            }),
        }
    }
}
