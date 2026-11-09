//! Sky atmosphere LUTs and sky irradiance.

use glam::Vec3;
use wgpu::{ShaderStages, TextureFormat, TextureUsages};

use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer, texture, workgroups, TextureSpec};

const LUT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
const MULTISCATTER_SIZE: u32 = 32;
const SKY_VIEW_SIZE: (u32, u32) = (192, 108);

pub struct Sky {
    pub transmittance: wgpu::TextureView,
    pub multiscatter: wgpu::TextureView,
    pub sky_view: wgpu::TextureView,
    /// Nine RGB SH coefficients of sky radiance per unit sun illuminance.
    pub sh: wgpu::Buffer,
    transmittance_pipeline: wgpu::ComputePipeline,
    multiscatter_pipeline: wgpu::ComputePipeline,
    sky_view_pipeline: wgpu::ComputePipeline,
    sh_pipeline: wgpu::ComputePipeline,
    layouts: [wgpu::BindGroupLayout; 4],
    /// Ground albedo the static LUTs were built for.
    built_for: Option<Vec3>,
}

impl Sky {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "sky.wgsl");
        let usage = TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
        let lut = |label, width, height| {
            texture(device, TextureSpec { label, width, height, layers: 1, mips: 1, format: LUT_FORMAT, usage })
                .create_view(&Default::default())
        };

        let c = ShaderStages::COMPUTE;
        let layouts = [
            bind_group_layout(device, "transmittance lut", c, &[(0, bind::uniform()), (1, bind::storage_texture(LUT_FORMAT))]),
            bind_group_layout(
                device,
                "multiscatter lut",
                c,
                &[(0, bind::uniform()), (2, bind::texture(true)), (3, bind::storage_texture(LUT_FORMAT)), (7, bind::sampler())],
            ),
            bind_group_layout(
                device,
                "sky view lut",
                c,
                &[
                    (0, bind::uniform()),
                    (2, bind::texture(true)),
                    (4, bind::texture(true)),
                    (5, bind::storage_texture(LUT_FORMAT)),
                    (7, bind::sampler()),
                ],
            ),
            bind_group_layout(
                device,
                "sky sh",
                c,
                &[
                    (0, bind::uniform()),
                    (2, bind::texture(true)),
                    (6, bind::texture(true)),
                    (7, bind::sampler()),
                    (8, bind::storage(false)),
                ],
            ),
        ];

        Self {
            transmittance: lut("transmittance lut", TRANSMITTANCE_SIZE.0, TRANSMITTANCE_SIZE.1),
            multiscatter: lut("multiscatter lut", MULTISCATTER_SIZE, MULTISCATTER_SIZE),
            sky_view: lut("sky view lut", SKY_VIEW_SIZE.0, SKY_VIEW_SIZE.1),
            sh: storage_buffer(device, "sky sh", 9 * 16, wgpu::BufferUsages::empty()),
            transmittance_pipeline: compute_pipeline(device, "transmittance lut", &module, "build_transmittance", &layouts[0], &[]),
            multiscatter_pipeline: compute_pipeline(device, "multiscatter lut", &module, "build_multiscatter", &layouts[1], &[]),
            sky_view_pipeline: compute_pipeline(device, "sky view lut", &module, "build_sky_view", &layouts[2], &[]),
            sh_pipeline: compute_pipeline(device, "sky sh", &module, "project_sky_sh", &layouts[3], &[]),
            layouts,
            built_for: None,
        }
    }

    pub fn record(
        &mut self,
        rec: &mut Recorder,
        frame: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        ground_albedo: Vec3,
    ) {
        let device = rec.device;
        use wgpu::BindingResource::{Sampler, TextureView};
        let frame = frame.as_entire_binding();
        let mut pass = rec.compute_pass("sky");

        // Transmittance and multiple scattering depend only on the atmosphere and ground.
        if self.built_for != Some(ground_albedo) {
            let group = bind_group(device, "transmittance", &self.layouts[0], &[(0, frame.clone()), (1, TextureView(&self.transmittance))]);
            pass.set_pipeline(&self.transmittance_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(workgroups(TRANSMITTANCE_SIZE.0, 8), workgroups(TRANSMITTANCE_SIZE.1, 8), 1);

            let group = bind_group(
                device,
                "multiscatter",
                &self.layouts[1],
                &[(0, frame.clone()), (2, TextureView(&self.transmittance)), (3, TextureView(&self.multiscatter)), (7, Sampler(sampler))],
            );
            pass.set_pipeline(&self.multiscatter_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(MULTISCATTER_SIZE, MULTISCATTER_SIZE, 1);
            self.built_for = Some(ground_albedo);
        }

        // The sky view and its irradiance follow the sun and camera every frame.
        let group = bind_group(
            device,
            "sky view",
            &self.layouts[2],
            &[
                (0, frame.clone()),
                (2, TextureView(&self.transmittance)),
                (4, TextureView(&self.multiscatter)),
                (5, TextureView(&self.sky_view)),
                (7, Sampler(sampler)),
            ],
        );
        pass.set_pipeline(&self.sky_view_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(SKY_VIEW_SIZE.0, 8), workgroups(SKY_VIEW_SIZE.1, 8), 1);

        let group = bind_group(
            device,
            "sky sh",
            &self.layouts[3],
            &[
                (0, frame),
                (2, TextureView(&self.transmittance)),
                (6, TextureView(&self.sky_view)),
                (7, Sampler(sampler)),
                (8, self.sh.as_entire_binding()),
            ],
        );
        pass.set_pipeline(&self.sh_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
