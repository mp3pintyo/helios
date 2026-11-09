//! Volumetric clouds: a ray-marched cumulus layer around the planet, lit by the sun (with
//! self-shadowing and multiple scattering) and the sky, plus a cloud shadow map for the ground.
//! See `shaders/clouds.wgsl` for the technique.
//!
//! The march runs at quarter resolution (half width, half height) only where the depth buffer
//! shows sky, and a temporal pass accumulates its jittered results; interiors cost nothing.

use glam::{Vec2, Vec3, Vec4};
use wgpu::{ShaderStages, TextureFormat, TextureUsages};

use crate::profiler::Recorder;
use crate::scene::CloudLayer;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{
    bind, bind_group, bind_group_layout, compute_pipeline, texture, uniform_buffer, workgroups,
    TextureSpec,
};

const SHAPE_SIZE: u32 = 128;
const DETAIL_SIZE: u32 = 32;
const SHADOW_SIZE: u32 = 512;
/// Ground area covered by the cloud shadow map (metres).
const SHADOW_EXTENT: f32 = 16_000.0;
/// Output resolution divisor per axis.
const DOWNSCALE: u32 = 2;
const COLOR: TextureFormat = TextureFormat::Rgba16Float;
const DISTANCE: TextureFormat = TextureFormat::R32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CloudUniform {
    coverage: f32,
    density: f32,
    base: f32,
    thickness: f32,
    wind: Vec2,
    shape_scale: f32,
    detail_scale: f32,
    weather_scale: f32,
    shadow_strength: f32,
    shadow_extent: f32,
    enabled: u32,
    shadow_center: Vec4,
}

struct Targets {
    size: (u32, u32),
    raw: wgpu::TextureView,
    raw_distance: wgpu::TextureView,
    /// Ping-pong history (indexed by the renderer's frame parity).
    resolved: [wgpu::TextureView; 2],
    distance: [wgpu::TextureView; 2],
}

impl Targets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let size = (
            width.div_ceil(DOWNSCALE).max(1),
            height.div_ceil(DOWNSCALE).max(1),
        );
        let view = |label, format| {
            texture(
                device,
                TextureSpec {
                    label,
                    width: size.0,
                    height: size.1,
                    layers: 1,
                    mips: 1,
                    format,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                },
            )
            .create_view(&Default::default())
        };
        Self {
            size,
            raw: view("clouds raw", COLOR),
            raw_distance: view("clouds raw distance", DISTANCE),
            resolved: [view("clouds 0", COLOR), view("clouds 1", COLOR)],
            distance: [
                view("clouds distance 0", DISTANCE),
                view("clouds distance 1", DISTANCE),
            ],
        }
    }
}

pub struct Clouds {
    pub uniform: wgpu::Buffer,
    shape: wgpu::TextureView,
    detail: wgpu::TextureView,
    /// Ground transmittance of sunlight through the clouds (see `shadow` in clouds.wgsl).
    pub shadow: wgpu::TextureView,
    sampler: wgpu::Sampler,
    noise_built: bool,
    enabled: bool,
    targets: Option<Targets>,
    /// Texels of the targets this frame's internal resolution covers.
    extent: (u32, u32),
    build_shape: wgpu::ComputePipeline,
    build_detail: wgpu::ComputePipeline,
    march: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
    shadow_pipeline: wgpu::ComputePipeline,
    layouts: Layouts,
}

struct Layouts {
    noise: wgpu::BindGroupLayout,
    march: wgpu::BindGroupLayout,
    temporal: wgpu::BindGroupLayout,
    shadow: wgpu::BindGroupLayout,
}

/// What the cloud pass reads from the rest of the frame.
pub struct CloudInputs<'a> {
    pub frame: &'a wgpu::Buffer,
    pub sky: &'a crate::passes::sky::Sky,
    pub lut_sampler: &'a wgpu::Sampler,
    pub depth: &'a wgpu::TextureView,
}

impl Clouds {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "clouds.wgsl");
        let c = ShaderStages::COMPUTE;
        let noise_texture = |label, size| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: size,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let common = [
            (0, bind::uniform()),
            (1, bind::uniform()),
            (2, bind::texture_3d(true)),
            (3, bind::texture_3d(true)),
            (4, bind::sampler()),
        ];
        let with = |extra: &[(u32, wgpu::BindingType)]| -> Vec<(u32, wgpu::BindingType)> {
            common.iter().chain(extra).copied().collect()
        };
        let layouts = Layouts {
            noise: bind_group_layout(
                device,
                "cloud noise",
                c,
                &[
                    (20, bind::storage_texture_3d(TextureFormat::Rgba8Unorm)),
                    (21, bind::storage_texture_3d(TextureFormat::Rgba8Unorm)),
                ],
            ),
            march: bind_group_layout(
                device,
                "cloud march",
                c,
                &with(&[
                    (5, bind::texture(true)),
                    (6, bind::sampler()),
                    (7, bind::storage(true)),
                    (8, bind::texture(false)),
                    (9, bind::storage_texture(COLOR)),
                    (10, bind::storage_texture(DISTANCE)),
                ]),
            ),
            temporal: bind_group_layout(
                device,
                "cloud temporal",
                c,
                &[
                    (0, bind::uniform()),
                    (11, bind::texture(false)),
                    (12, bind::texture(false)),
                    (13, bind::texture(false)),
                    (14, bind::texture(false)),
                    (15, bind::storage_texture(COLOR)),
                    (16, bind::storage_texture(DISTANCE)),
                ],
            ),
            shadow: bind_group_layout(
                device,
                "cloud shadow",
                c,
                &with(&[(17, bind::storage_texture(COLOR))]),
            ),
        };
        let shadow = texture(
            device,
            TextureSpec {
                label: "cloud shadow",
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                layers: 1,
                mips: 1,
                format: COLOR,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            },
        )
        .create_view(&Default::default());
        let repeat = wgpu::AddressMode::Repeat;
        Self {
            uniform: uniform_buffer(device, "clouds", std::mem::size_of::<CloudUniform>() as u64),
            shape: noise_texture("cloud shape noise", SHAPE_SIZE),
            detail: noise_texture("cloud detail noise", DETAIL_SIZE),
            shadow,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("cloud noise"),
                address_mode_u: repeat,
                address_mode_v: repeat,
                address_mode_w: repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            noise_built: false,
            enabled: false,
            targets: None,
            extent: (1, 1),
            build_shape: compute_pipeline(
                device,
                "cloud shape noise",
                &module,
                "build_shape",
                &layouts.noise,
                &[],
            ),
            build_detail: compute_pipeline(
                device,
                "cloud detail noise",
                &module,
                "build_detail",
                &layouts.noise,
                &[],
            ),
            march: compute_pipeline(device, "cloud march", &module, "march", &layouts.march, &[]),
            temporal: compute_pipeline(
                device,
                "cloud temporal",
                &module,
                "temporal",
                &layouts.temporal,
                &[],
            ),
            shadow_pipeline: compute_pipeline(
                device,
                "cloud shadow",
                &module,
                "shadow",
                &layouts.shadow,
                &[],
            ),
            layouts,
        }
    }

    /// Writes this frame's parameters (`None` turns clouds off). `time` animates the wind.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layer: Option<&CloudLayer>, camera: Vec3, time: f32, frame: &FrameTargets) {
        let layer = layer.copied().unwrap_or_default();
        self.enabled = layer.coverage > 0.0;
        // Allocated for the largest internal resolution; a frame marches the corner it covers.
        let (capacity, size) = (frame.capacity, (frame.width, frame.height));
        let target_size = (capacity.0.div_ceil(DOWNSCALE).max(1), capacity.1.div_ceil(DOWNSCALE).max(1));
        if self.targets.as_ref().is_none_or(|t| t.size != target_size) {
            self.targets = Some(Targets::new(device, capacity.0, capacity.1));
        }
        self.extent = (size.0.div_ceil(DOWNSCALE).max(1), size.1.div_ceil(DOWNSCALE).max(1));
        // Snap the shadow map to whole texels so its content does not swim as the camera moves.
        let texel = SHADOW_EXTENT / SHADOW_SIZE as f32;
        let center = (Vec2::new(camera.x, camera.z) / texel).round() * texel;
        let km = 0.001;
        let uniform = CloudUniform {
            coverage: layer.coverage,
            density: layer.density,
            base: layer.base_altitude * km,
            thickness: layer.thickness * km,
            wind: layer.wind * time * km,
            shape_scale: 1.0 / 3.5,
            detail_scale: 1.0 / 0.6,
            weather_scale: 1.0 / 28.0,
            shadow_strength: layer.shadow_strength,
            shadow_extent: SHADOW_EXTENT,
            enabled: self.enabled as u32,
            shadow_center: Vec4::new(center.x, center.y, 0.0, 0.0),
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// The resolved clouds of the frame with parity `current` (rgb inscattered radiance,
    /// a transmittance), at quarter resolution.
    pub fn output(&self, current: usize) -> &wgpu::TextureView {
        &self
            .targets
            .as_ref()
            .expect("Clouds::update before output")
            .resolved[current]
    }

    pub fn record(&mut self, rec: &mut Recorder, inputs: &CloudInputs, current: usize) {
        use wgpu::BindingResource::{Sampler, TextureView};
        let device = rec.device;
        if !self.noise_built {
            let group = bind_group(
                device,
                "cloud noise",
                &self.layouts.noise,
                &[
                    (20, TextureView(&self.shape)),
                    (21, TextureView(&self.detail)),
                ],
            );
            let mut pass = rec.compute_pass("cloud noise");
            pass.set_bind_group(0, &group, &[]);
            pass.set_pipeline(&self.build_shape);
            let n = workgroups(SHAPE_SIZE, 4);
            pass.dispatch_workgroups(n, n, n);
            pass.set_pipeline(&self.build_detail);
            let n = workgroups(DETAIL_SIZE, 4);
            pass.dispatch_workgroups(n, n, n);
            self.noise_built = true;
        }
        let Some(t) = &self.targets else { return };
        let common = vec![
            (0, inputs.frame.as_entire_binding()),
            (1, self.uniform.as_entire_binding()),
            (2, TextureView(&self.shape)),
            (3, TextureView(&self.detail)),
            (4, Sampler(&self.sampler)),
        ];
        let march = bind_group(
            device,
            "cloud march",
            &self.layouts.march,
            &[
                common.as_slice(),
                &[
                    (5, TextureView(&inputs.sky.transmittance)),
                    (6, Sampler(inputs.lut_sampler)),
                    (7, inputs.sky.sh.as_entire_binding()),
                    (8, TextureView(inputs.depth)),
                    (9, TextureView(&t.raw)),
                    (10, TextureView(&t.raw_distance)),
                ],
            ]
            .concat(),
        );
        let previous = 1 - current;
        let temporal = bind_group(
            device,
            "cloud temporal",
            &self.layouts.temporal,
            &[
                (0, inputs.frame.as_entire_binding()),
                (11, TextureView(&t.raw)),
                (12, TextureView(&t.raw_distance)),
                (13, TextureView(&t.resolved[previous])),
                (14, TextureView(&t.distance[previous])),
                (15, TextureView(&t.resolved[current])),
                (16, TextureView(&t.distance[current])),
            ],
        );
        let shadow = bind_group(
            device,
            "cloud shadow",
            &self.layouts.shadow,
            &[common.as_slice(), &[(17, TextureView(&self.shadow))]].concat(),
        );

        let mut pass = rec.compute_pass("clouds");
        pass.set_pipeline(&self.march);
        pass.set_bind_group(0, &march, &[]);
        pass.dispatch_workgroups(workgroups(self.extent.0, 8), workgroups(self.extent.1, 8), 1);
        pass.set_pipeline(&self.temporal);
        pass.set_bind_group(0, &temporal, &[]);
        pass.dispatch_workgroups(workgroups(self.extent.0, 8), workgroups(self.extent.1, 8), 1);
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &shadow, &[]);
        pass.dispatch_workgroups(workgroups(SHADOW_SIZE, 8), workgroups(SHADOW_SIZE, 8), 1);
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }
}
