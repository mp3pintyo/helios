//! Volumetric fog: participating media in a camera-aligned froxel volume, lit by the sun (with
//! virtual shadow map visibility, so light shafts fall through windows and canopies), local
//! lights and Lumen's radiance cache. See `shaders/fog.wgsl` for the technique.

use glam::Vec4;
use wgpu::{ShaderStages, TextureFormat, TextureUsages};

use crate::profiler::Recorder;
use crate::scene::FogVolume;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, uniform_buffer, workgroups};

/// Screen pixels per froxel column (horizontally and vertically).
const TILE: u32 = 16;
const SLICES: u32 = 64;
const FORMAT: TextureFormat = TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FogUniform {
    media: Vec4,
    albedo: Vec4,
    height: Vec4,
    wind: Vec4,
    range: Vec4,
    enabled: u32,
    _pad: [u32; 3],
}

struct Volumes {
    size: (u32, u32),
    /// Ping-pong scattering volumes (this frame's result is next frame's history).
    scattering: [wgpu::TextureView; 2],
    integrated: wgpu::TextureView,
}

impl Volumes {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let size = (width.div_ceil(TILE).max(1), height.div_ceil(TILE).max(1));
        let volume = |label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: SLICES },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: FORMAT,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Self { size, scattering: [volume("fog scattering 0"), volume("fog scattering 1")], integrated: volume("fog integrated") }
    }
}

/// What fog injection reads from the rest of the frame.
pub struct FogInputs<'a> {
    pub frame: &'a wgpu::Buffer,
    pub vsm: &'a crate::vsm::VirtualShadowMap,
    pub sky: &'a crate::passes::sky::Sky,
    pub lights: &'a crate::lights::LightCulling,
    pub clouds: &'a crate::clouds::Clouds,
    pub lumen: &'a crate::lumen::Lumen,
    pub samplers: &'a crate::passes::Samplers,
}

pub struct VolumetricFog {
    pub uniform: wgpu::Buffer,
    enabled: bool,
    volumes: Option<Volumes>,
    inject: wgpu::ComputePipeline,
    integrate: wgpu::ComputePipeline,
    inject_layout: wgpu::BindGroupLayout,
    integrate_layout: wgpu::BindGroupLayout,
    /// A 1x1x1 empty volume bound to the lighting while fog is off.
    empty: wgpu::TextureView,
}

impl VolumetricFog {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "fog.wgsl");
        let c = ShaderStages::COMPUTE;
        let inject_layout = bind_group_layout(
            device,
            "fog inject",
            c,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::uniform()),
                (3, bind::storage(true)),
                (4, bind::texture_depth()),
                (5, bind::sampler_comparison()),
                (6, bind::storage(true)),
                (7, bind::storage(true)),
                (8, bind::storage(true)),
                (9, bind::uniform()),
                (10, bind::texture(true)),
                (11, bind::sampler()),
                (12, bind::texture(true)),
                (13, bind::uniform()),
                (14, bind::texture_3d(true)),
                (15, bind::texture_3d(true)),
                (16, bind::storage(true)),
                (17, bind::sampler()),
                (18, bind::texture_3d(true)),
                (19, bind::storage_texture_3d(FORMAT)),
                (60, bind::uniform()),
            ],
        );
        let integrate_layout = bind_group_layout(
            device,
            "fog integrate",
            c,
            &[(0, bind::uniform()), (1, bind::uniform()), (20, bind::texture_3d(false)), (21, bind::storage_texture_3d(FORMAT))],
        );
        let empty = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("fog off"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: FORMAT,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        Self {
            uniform: uniform_buffer(device, "fog", std::mem::size_of::<FogUniform>() as u64),
            enabled: false,
            volumes: None,
            inject: compute_pipeline(device, "fog inject", &module, "inject", &inject_layout, &[]),
            integrate: compute_pipeline(device, "fog integrate", &module, "integrate", &integrate_layout, &[]),
            inject_layout,
            integrate_layout,
            empty,
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, fog: &FogVolume, enabled: bool, time: f32, size: (u32, u32)) {
        self.enabled = enabled && fog.density > 0.0;
        let volume_size = (size.0.div_ceil(TILE).max(1), size.1.div_ceil(TILE).max(1));
        if self.enabled && self.volumes.as_ref().is_none_or(|v| v.size != volume_size) {
            self.volumes = Some(Volumes::new(device, size.0, size.1));
        }
        let drift = fog.wind * time * 0.06;
        let uniform = FogUniform {
            media: Vec4::new(fog.density, fog.anisotropy, fog.ambient, fog.noise),
            albedo: fog.albedo.extend(1.0),
            height: Vec4::new(fog.base_height, fog.height_falloff, 0.0, 0.0),
            wind: drift.extend(0.0),
            range: Vec4::new(0.5, fog.distance.max(1.0), 0.0, 0.0),
            enabled: self.enabled as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The integrated volume read by the lighting (an empty volume while fog is off).
    pub fn integrated(&self) -> &wgpu::TextureView {
        match (&self.volumes, self.enabled) {
            (Some(v), true) => &v.integrated,
            _ => &self.empty,
        }
    }

    pub fn record(&self, rec: &mut Recorder, inputs: &FogInputs, current: usize) {
        use wgpu::BindingResource::{Sampler, TextureView};
        let Some(v) = self.volumes.as_ref().filter(|_| self.enabled) else { return };
        let device = rec.device;
        let (sdf, voxel_lighting) = inputs.lumen.debug_views();
        let inject = bind_group(
            device,
            "fog inject",
            &self.inject_layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, self.uniform.as_entire_binding()),
                (2, inputs.vsm.uniform.as_entire_binding()),
                (3, inputs.vsm.cache.entries.as_entire_binding()),
                (4, TextureView(&inputs.vsm.atlas)),
                (5, Sampler(&inputs.samplers.shadow)),
                (6, inputs.sky.sh.as_entire_binding()),
                (7, inputs.lights.lights.as_entire_binding()),
                (8, inputs.lights.grid.as_entire_binding()),
                (9, inputs.clouds.uniform.as_entire_binding()),
                (10, TextureView(&inputs.clouds.shadow)),
                (11, Sampler(&inputs.samplers.linear)),
                (12, TextureView(&inputs.sky.transmittance)),
                (13, inputs.lumen.uniform().as_entire_binding()),
                (14, TextureView(sdf)),
                (15, TextureView(voxel_lighting)),
                (16, inputs.lumen.radiance_probes().as_entire_binding()),
                (17, Sampler(&inputs.samplers.linear)),
                (18, TextureView(&v.scattering[1 - current])),
                (19, TextureView(&v.scattering[current])),
                (60, inputs.vsm.local.uniform.as_entire_binding()),
            ],
        );
        let integrate = bind_group(
            device,
            "fog integrate",
            &self.integrate_layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, self.uniform.as_entire_binding()),
                (20, TextureView(&v.scattering[current])),
                (21, TextureView(&v.integrated)),
            ],
        );
        let mut pass = rec.compute_pass("volumetric fog");
        pass.set_pipeline(&self.inject);
        pass.set_bind_group(0, &inject, &[]);
        pass.dispatch_workgroups(workgroups(v.size.0, 4), workgroups(v.size.1, 4), workgroups(SLICES, 4));
        pass.set_pipeline(&self.integrate);
        pass.set_bind_group(0, &integrate, &[]);
        pass.dispatch_workgroups(workgroups(v.size.0, 8), workgroups(v.size.1, 8), 1);
    }
}
