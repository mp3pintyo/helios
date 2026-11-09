//! Order-independent compositing of particles and heterogeneous volumes into the lit scene.
//!
//! Particles are blended with **weighted blended order-independent transparency** (McGuire and
//! Bavoil 2013): every fragment adds its premultiplied colour and opacity, weighted by depth,
//! to an accumulation target and multiplies a revealage target by `1 − opacity`; one full-screen
//! pass turns them into the average colour times the total coverage. No sorting, any number of
//! overlapping layers (dust in a light shaft, a shower of sparks), at the price of an
//! approximate order between layers of similar depth.
//!
//! **Order with the translucency pass.** Everything is split at the depth of the nearest
//! translucent surface (glass, water; drawn depth-only by [`crate::translucency`] when any is
//! visible): particles write to a *front* or a *back* pair of targets, the volume march splits
//! its rays there. The back bucket is composited before the translucency pass, so glass tints
//! and water refracts the smoke and sparks behind it; the front bucket after it, over the glass.
//! Within a bucket the particles go over the volume, the volume over the scene.
//!
//! **TSR.** The front composite writes a reactive mask into the motion target's z (unused by
//! then; UE's "responsive AA" for translucency), which TSR reads when `flags::REACTIVE` is set:
//! covered pixels trust their history less, so sparks and flames don't leave trails.

use bytemuck::{Pod, Zeroable};
use wgpu::BindingResource::TextureView;
use wgpu::{ShaderStages, TextureFormat};

use crate::passes::shading::LightingInputs;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::{DEPTH_FORMAT, HDR_FORMAT};
use crate::util::{bind, bind_group, bind_group_layout, uniform_buffer};

const ACCUM_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
const REVEAL_FORMAT: TextureFormat = TextureFormat::R16Float;

/// Which side of the nearest translucent surface a composite handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bucket {
    Front = 0,
    Back = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeUniform {
    volume_rect: [u32; 4],
}

/// OIT accumulation and revealage, front and back, and the nearest translucent depth.
struct Targets {
    size: (u32, u32),
    accum: [wgpu::TextureView; 2],
    reveal: [wgpu::TextureView; 2],
    translucent_depth: wgpu::TextureView,
}

impl Targets {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let make = |label: &str, format| {
            crate::util::texture(
                device,
                crate::util::TextureSpec {
                    label,
                    width: size.0,
                    height: size.1,
                    layers: 1,
                    mips: 1,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                },
            )
            .create_view(&Default::default())
        };
        Self {
            size,
            accum: [make("oit accum front", ACCUM_FORMAT), make("oit accum back", ACCUM_FORMAT)],
            reveal: [make("oit reveal front", REVEAL_FORMAT), make("oit reveal back", REVEAL_FORMAT)],
            translucent_depth: make("nearest translucent depth", DEPTH_FORMAT),
        }
    }
}

pub(crate) struct Oit {
    layout: wgpu::BindGroupLayout,
    back: wgpu::RenderPipeline,
    front: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    targets: Option<Targets>,
}

impl Oit {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "oit.wgsl");
        let f = ShaderStages::FRAGMENT;
        let layout = bind_group_layout(
            device,
            "oit composite",
            f,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::texture(false)),
                (3, bind::texture(false)),
                (4, bind::texture(false)),
                (5, bind::texture(false)),
                (6, bind::texture(false)),
                (7, bind::texture(false)),
            ],
        );
        let over = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add };
        let color = Some(wgpu::ColorTargetState { format: HDR_FORMAT, blend: Some(wgpu::BlendState { color: over, alpha: over }), write_mask: wgpu::ColorWrites::COLOR });
        let reactive = Some(wgpu::ColorTargetState { format: TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::BLUE });
        let pipeline = |entry: &str, targets: &[Option<wgpu::ColorTargetState>]| {
            let pipeline_layout = crate::util::pipeline_layout(device, "oit composite", &[&layout]);
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("oit composite"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some(entry), compilation_options: Default::default(), targets }),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            back: pipeline("fs_back", &[color.clone()]),
            front: pipeline("fs_front", &[color, reactive]),
            layout,
            uniform: uniform_buffer(device, "oit composite", std::mem::size_of::<CompositeUniform>() as u64),
            targets: None,
        }
    }

    /// Allocates the targets for the internal resolution's capacity and records where the
    /// volume textures are valid this frame.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, capacity: (u32, u32), volume_rect: [u32; 4]) {
        if self.targets.as_ref().is_none_or(|t| t.size != capacity) {
            self.targets = Some(Targets::new(device, capacity));
        }
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&CompositeUniform { volume_rect }));
    }

    /// The nearest translucent surface's depth (reverse-Z; 0 where there is none).
    pub fn translucent_depth(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.translucent_depth)
    }

    /// Clears the nearest translucent depth to "none" (no translucent surface is drawn).
    pub fn clear_translucent_depth(&self, rec: &mut Recorder) {
        let Some(t) = &self.targets else { return };
        let _ = rec.render_pass(
            "oit translucent depth clear",
            &[],
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: &t.translucent_depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
        );
    }

    /// Clears the accumulation targets and draws the particles into them.
    pub fn record_particles(&self, rec: &mut Recorder, inputs: &LightingInputs, particles: &crate::particles::ParticleSystem) {
        let Some(t) = &self.targets else { return };
        let device = rec.device;
        let attachment = |view, clear: f64| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: clear, g: clear, b: clear, a: clear }), store: wgpu::StoreOp::Store },
            })
        };
        let attachments = [attachment(&t.accum[0], 0.0), attachment(&t.reveal[0], 1.0), attachment(&t.accum[1], 0.0), attachment(&t.reveal[1], 1.0)];
        let mut pass = rec.render_pass("particles (oit)", &attachments, None);
        pass.set_viewport(0.0, 0.0, inputs.targets.width as f32, inputs.targets.height as f32, 0.0, 1.0);
        particles.draw(device, &mut pass, inputs, &t.translucent_depth);
    }

    /// Composites one bucket's particles and volume part over the HDR image (the front bucket
    /// also writes TSR's reactive mask). `volume`: the resolved volume parts, front and back.
    pub fn record_composite(&self, rec: &mut Recorder, inputs: &LightingInputs, bucket: Bucket, volume: Option<&[wgpu::TextureView; 2]>) {
        let Some(t) = &self.targets else { return };
        let (this, other) = (bucket as usize, 1 - bucket as usize);
        // Without volumes their rectangle is empty and the textures are never read.
        let volume_view = |i: usize| volume.map_or(&t.accum[i], |v| &v[i]);
        let group = bind_group(
            rec.device,
            "oit composite",
            &self.layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, self.uniform.as_entire_binding()),
                (2, TextureView(&t.accum[this])),
                (3, TextureView(&t.reveal[this])),
                (4, TextureView(volume_view(this))),
                (5, TextureView(&t.reveal[other])),
                (6, TextureView(volume_view(other))),
                (7, TextureView(&inputs.targets.gbuffer_albedo)),
            ],
        );
        let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
        let target = |view| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: load });
        let (label, pipeline, attachments) = match bucket {
            Bucket::Back => ("oit composite (behind translucency)", &self.back, vec![target(&inputs.targets.hdr)]),
            Bucket::Front => ("oit composite", &self.front, vec![target(&inputs.targets.hdr), target(&inputs.targets.gbuffer_motion)]),
        };
        let mut pass = rec.render_pass(label, &attachments, None);
        pass.set_viewport(0.0, 0.0, inputs.targets.width as f32, inputs.targets.height as f32, 0.0, 1.0);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The particles' draw into the four OIT targets: accumulation adds, revealage multiplies by
/// `1 − opacity` (`fs_particle` in particles.wgsl).
pub(crate) fn particle_pipeline(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let add = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
    let reveal = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::OneMinusSrc, operation: wgpu::BlendOperation::Add };
    let accum_target = Some(wgpu::ColorTargetState { format: ACCUM_FORMAT, blend: Some(wgpu::BlendState { color: add, alpha: add }), write_mask: wgpu::ColorWrites::ALL });
    let reveal_target = Some(wgpu::ColorTargetState { format: REVEAL_FORMAT, blend: Some(wgpu::BlendState { color: reveal, alpha: reveal }), write_mask: wgpu::ColorWrites::RED });
    let targets = [accum_target.clone(), reveal_target.clone(), accum_target, reveal_target];
    let pipeline_layout = crate::util::pipeline_layout(device, "particle draw", &[layout]);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("particle draw"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState { module, entry_point: Some("vs_particle"), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState { module, entry_point: Some("fs_particle"), compilation_options: Default::default(), targets: &targets }),
        multiview_mask: None,
        cache: None,
    })
}
