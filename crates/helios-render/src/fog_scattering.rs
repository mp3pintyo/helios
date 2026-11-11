//! Fog screen space scattering (UE 5.8's experimental Fog Screen Space Scattering): spreads the
//! light the volumetric fog scatters slightly out of each pixel's view ray over the pixel's
//! neighbourhood, so lamps, candles and lit windows glow through fog, smoke and dust, more where
//! the view ray crosses more fog. A post step at output resolution after TSR: it reads the fog's
//! integrated volume (in-scattering and transmittance per pixel) and leaves fog.wgsl alone. See
//! `shaders/fog_scattering.wgsl` for the model.

use std::f32::consts::PI;

use glam::Vec4;
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat, TextureUsages};

use crate::fog::VolumetricFog;
use crate::passes::Samplers;
use crate::pipelines::Lazy;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, texture, uniform_buffer, workgroups, TextureSpec};

const HDR: TextureFormat = TextureFormat::Rgba16Float;
/// Pyramid levels from half resolution down (1/2 … 1/256 of the output).
const MAX_LEVELS: usize = 8;
/// Radius of a level's blur in its own texels (13-tap downsample and tent upsample together).
const LEVEL_RADIUS: f32 = 1.5;
/// Light scattered at angle θ somewhere between the camera and its source appears, on average,
/// at about half that angle from the source (scattering near the source shows it at θ, near the
/// camera at almost 0).
const APPARENT_ANGLE: f32 = 0.5;

/// Fog screen space scattering settings ([`crate::RenderSettings::fog_scattering`]).
#[derive(Clone, Copy, Debug)]
pub struct FogScattering {
    /// Scale on the scattered light that is spread: 1 = the share of the fog's phase function
    /// that falls within the blur's reach (physically based), more stands in for the wider
    /// spread of many scattering orders, which the model keeps at the single-scattering angles;
    /// it never exceeds all the light the fog scattered.
    pub intensity: f32,
    /// Henyey–Greenstein anisotropy of the light the step spreads (the shape of the glow).
    /// Droplets of fog and mist and grains of dust scatter far more forward (0.8–0.9, with a
    /// diffraction peak HG does not even have) than the gentle phase a fog volume is usually
    /// given for its single scattering (`FogVolume::anisotropy`, 0.6–0.7 in the demo scenes,
    /// so sun shafts stay visible from the side); the fog volume's own value is used when it
    /// is larger.
    pub anisotropy: f32,
}

impl Default for FogScattering {
    fn default() -> Self {
        Self { intensity: 1.0, anisotropy: 0.85 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    params: Vec4,
}

struct Targets {
    size: (u32, u32),
    down: Vec<wgpu::TextureView>,
    up: Vec<wgpu::TextureView>,
    output: wgpu::TextureView,
}

pub struct FogScatteringPass {
    layout: wgpu::BindGroupLayout,
    prefilter: Lazy<wgpu::ComputePipeline>,
    downsample: Lazy<wgpu::ComputePipeline>,
    upsample: Lazy<wgpu::ComputePipeline>,
    composite: Lazy<wgpu::ComputePipeline>,
    uniform: wgpu::Buffer,
    /// Per pyramid level: its weight in the summed kernel and whether it is the coarsest.
    level_uniforms: Vec<wgpu::Buffer>,
    targets: Option<Targets>,
    /// A 1x1 texture bound where a dispatch reads no level.
    unused: wgpu::TextureView,
    enabled: bool,
}

/// Fraction of the light a Henyey–Greenstein phase function with anisotropy `g` scatters
/// within `theta` of the forward direction.
pub(crate) fn forward_fraction(g: f32, theta: f32) -> f32 {
    let mu = theta.clamp(0.0, PI).cos();
    let g = g.clamp(-0.95, 0.95);
    if g.abs() < 1e-3 {
        return (1.0 - mu) * 0.5;
    }
    let g2 = g * g;
    ((1.0 - g2) / (2.0 * g) * (1.0 / (1.0 - g) - 1.0 / (1.0 + g2 - 2.0 * g * mu).sqrt())).clamp(0.0, 1.0)
}

/// Pyramid levels for an output size (the coarsest at least 2 texels across).
fn level_count((width, height): (u32, u32)) -> usize {
    let smallest = (width / 2).min(height / 2);
    (1..=MAX_LEVELS).take_while(|&n| smallest >> (n - 1) >= 2).last().unwrap_or(1)
}

/// The weight of each level in the summed kernel (the phase function's scattered light between
/// its angular radius and the previous level's, normalised), and the share F of the scattered
/// light that the widest level reaches.
pub(crate) fn level_weights(anisotropy: f32, fov_y: f32, height: u32, levels: usize) -> (Vec<f32>, f32) {
    let pixel_angle = 2.0 * (fov_y * 0.5).tan() / height.max(1) as f32;
    let reach = |level: usize| {
        let radius = LEVEL_RADIUS * (2u32 << level) as f32;
        forward_fraction(anisotropy, (radius * pixel_angle).atan() / APPARENT_ANGLE)
    };
    let total = reach(levels - 1).max(1e-6);
    let mut previous = 0.0;
    let weights = (0..levels)
        .map(|level| {
            let f = reach(level);
            let w = (f - previous) / total;
            previous = f;
            w
        })
        .collect();
    (weights, total)
}

impl FogScatteringPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "fog_scattering.wgsl");
        let layout = bind_group_layout(
            device,
            "fog scattering",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::uniform()),
                (3, bind::uniform()),
                (4, bind::texture(false)),
                (5, bind::texture(false)),
                (6, bind::texture_3d(true)),
                (7, bind::sampler()),
                (8, bind::texture(true)),
                (9, bind::texture(true)),
                (10, bind::storage_texture(HDR)),
            ],
        );
        let pipeline = |entry| compute_pipeline(device, "fog scattering", &module, entry, &layout, &[]);
        let unused = texture(device, TextureSpec { label: "fog scattering unused", width: 1, height: 1, layers: 1, mips: 1, format: HDR, usage: TextureUsages::TEXTURE_BINDING })
            .create_view(&Default::default());
        Self {
            prefilter: pipeline("prefilter"),
            downsample: pipeline("downsample"),
            upsample: pipeline("upsample"),
            composite: pipeline("composite"),
            layout,
            uniform: uniform_buffer(device, "fog scattering", std::mem::size_of::<Uniform>() as u64),
            level_uniforms: (0..MAX_LEVELS).map(|_| uniform_buffer(device, "fog scattering level", 16)).collect(),
            targets: None,
            unused,
            enabled: false,
        }
    }

    /// Writes this frame's parameters: on when `settings` is given and the fog is on;
    /// `fog_anisotropy` is the fog volume's phase function's, `size` the output resolution.
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, settings: Option<&FogScattering>, fog_enabled: bool, fog_anisotropy: f32, fov_y: f32, size: (u32, u32)) {
        self.enabled = fog_enabled && settings.is_some_and(|s| s.intensity > 0.0);
        let Some(settings) = settings.filter(|_| self.enabled) else { return };
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            self.targets = Some(Self::create_targets(device, size));
        }
        let levels = level_count(size);
        let (weights, reach) = level_weights(settings.anisotropy.max(fog_anisotropy), fov_y, size.1, levels);
        let uniform = Uniform { params: Vec4::new(reach, settings.intensity, 0.0, 0.0) };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        for (level, weight) in weights.iter().enumerate() {
            let coarsest = if level + 1 == levels { 1.0 } else { 0.0 };
            queue.write_buffer(&self.level_uniforms[level], 0, bytemuck::bytes_of(&Vec4::new(*weight, coarsest, 0.0, 0.0)));
        }
    }

    fn create_targets(device: &wgpu::Device, size: (u32, u32)) -> Targets {
        let usage = TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
        let view = |label: &str, width: u32, height: u32| {
            texture(device, TextureSpec { label, width, height, layers: 1, mips: 1, format: HDR, usage }).create_view(&Default::default())
        };
        let levels = level_count(size);
        let level_size = |level: usize| ((size.0 / 2 >> level).max(1), (size.1 / 2 >> level).max(1));
        Targets {
            size,
            down: (0..levels).map(|l| view("fog scattering down", level_size(l).0, level_size(l).1)).collect(),
            up: (0..levels).map(|l| view("fog scattering up", level_size(l).0, level_size(l).1)).collect(),
            output: view("fog scattering", size.0, size.1),
        }
    }

    /// Adds the spread light to `input` (the TSR result) and returns the image to continue
    /// with (`input` itself while off).
    pub fn record<'a>(&'a self, rec: &mut Recorder, frame: &wgpu::Buffer, fog: &VolumetricFog, targets: &FrameTargets, samplers: &Samplers, input: &'a wgpu::TextureView) -> &'a wgpu::TextureView {
        let Some(t) = self.targets.as_ref().filter(|_| self.enabled && fog.enabled()) else { return input };
        let device = rec.device;
        let group = |level: usize, source: &wgpu::TextureView, lower: &wgpu::TextureView, destination: &wgpu::TextureView| {
            bind_group(
                device,
                "fog scattering",
                &self.layout,
                &[
                    (0, frame.as_entire_binding()),
                    (1, fog.uniform.as_entire_binding()),
                    (2, self.uniform.as_entire_binding()),
                    (3, self.level_uniforms[level].as_entire_binding()),
                    (4, TextureView(input)),
                    (5, TextureView(&targets.depth)),
                    (6, TextureView(fog.integrated())),
                    (7, Sampler(&samplers.linear)),
                    (8, TextureView(source)),
                    (9, TextureView(lower)),
                    (10, TextureView(destination)),
                ],
            )
        };
        let levels = t.down.len();
        let size = |view_level: usize| ((t.size.0 / 2 >> view_level).max(1), (t.size.1 / 2 >> view_level).max(1));
        let mut pass = rec.compute_pass("fog scattering");
        let dispatch = |pass: &mut wgpu::ComputePass, (w, h): (u32, u32)| pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);

        pass.set_pipeline(&self.prefilter);
        pass.set_bind_group(0, &group(0, &self.unused, &self.unused, &t.down[0]), &[]);
        dispatch(&mut pass, size(0));
        pass.set_pipeline(&self.downsample);
        for level in 1..levels {
            pass.set_bind_group(0, &group(level, &t.down[level - 1], &self.unused, &t.down[level]), &[]);
            dispatch(&mut pass, size(level));
        }
        pass.set_pipeline(&self.upsample);
        for level in (0..levels).rev() {
            let lower = if level + 1 < levels { &t.up[level + 1] } else { &self.unused };
            pass.set_bind_group(0, &group(level, &t.down[level], lower, &t.up[level]), &[]);
            dispatch(&mut pass, size(level));
        }
        pass.set_pipeline(&self.composite);
        pass.set_bind_group(0, &group(0, &t.up[0], &self.unused, &t.output), &[]);
        dispatch(&mut pass, t.size);
        &t.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_fraction_spans_the_sphere() {
        for g in [-0.5, 0.0, 0.3, 0.6, 0.9] {
            assert!(forward_fraction(g, 0.0).abs() < 1e-5, "g {g}");
            assert!((forward_fraction(g, PI) - 1.0).abs() < 1e-4, "g {g}");
        }
        // Forward scattering keeps more light near the forward direction.
        assert!(forward_fraction(0.8, 0.3) > forward_fraction(0.4, 0.3));
    }

    #[test]
    fn level_weights_are_normalised() {
        let levels = level_count((1600, 900));
        assert_eq!(levels, 8);
        let (weights, reach) = level_weights(0.7, 60f32.to_radians(), 900, levels);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(weights.iter().all(|&w| w >= 0.0));
        assert!(reach > 0.0 && reach < 1.0);
    }
}
