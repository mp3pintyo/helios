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
//! **Adaptive voxel-based OIT** (`avboit.rs`, the default where the device samples 17 textures
//! per stage): the same targets and buckets, but every fragment is weighted by the measured
//! transmittance in front of it instead of a depth guess, the translucent surfaces that don't
//! refract (`translucency.rs`) join the particles and the volume march in it, and the targets
//! hold weighted colour and coloured transmittance: the composite multiplies the background by
//! the transmittance, then adds the colour. The split is the nearest *refracting* surface.
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
use crate::util::{bind, bind_group, bind_group_layout, uniform_buffer, Lazy};

const ACCUM_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// Revealage (weighted blended: red) or coloured transmittance (AVBOIT: rgb, the particles' own
/// in alpha).
const REVEAL_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Which AVBOIT draw a particle or surface draw records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AvboitStage {
    /// Low resolution: widen the grid columns' bounds.
    Bounds,
    /// Low resolution: add optical depth to the cells.
    Transmittance,
    /// Full resolution: the weighted draw into the OIT targets.
    Shade,
}

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
    /// The same as a one-layer array (the translucency pass's `layers` binding).
    translucent_depth_array: wgpu::TextureView,
}

impl Targets {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let texture = |label: &str, format| {
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
        };
        let make = |label: &str, format| texture(label, format).create_view(&Default::default());
        let depth = texture("nearest translucent depth", DEPTH_FORMAT);
        Self {
            size,
            accum: [make("oit accum front", ACCUM_FORMAT), make("oit accum back", ACCUM_FORMAT)],
            reveal: [make("oit reveal front", REVEAL_FORMAT), make("oit reveal back", REVEAL_FORMAT)],
            translucent_depth: depth.create_view(&Default::default()),
            translucent_depth_array: depth.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() }),
        }
    }
}

pub(crate) struct Oit {
    layout: wgpu::BindGroupLayout,
    back: Lazy<wgpu::RenderPipeline>,
    front: Lazy<wgpu::RenderPipeline>,
    /// AVBOIT composites: background × transmittance, then the light of the back or front
    /// bucket added.
    avboit_modulate: [Lazy<wgpu::RenderPipeline>; 2],
    avboit_back: Lazy<wgpu::RenderPipeline>,
    avboit_front: Lazy<wgpu::RenderPipeline>,
    uniform: wgpu::Buffer,
    targets: Option<Targets>,
    /// The transmittance grid (allocated while AVBOIT is used).
    pub avboit: crate::avboit::Avboit,
    /// AVBOIT composites this frame.
    use_avboit: bool,
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
        let modulate = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::Src, operation: wgpu::BlendOperation::Add };
        let add = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
        let blended = |c: wgpu::BlendComponent| Some(wgpu::ColorTargetState { format: HDR_FORMAT, blend: Some(wgpu::BlendState { color: c, alpha: c }), write_mask: wgpu::ColorWrites::COLOR });
        let pipeline = |entry: &'static str, targets: Vec<Option<wgpu::ColorTargetState>>, background: bool| {
            let (device, module, layout) = (device.clone(), module.clone(), layout.clone());
            let make = move || {
                let pipeline_layout = crate::util::pipeline_layout(&device, "oit composite", &[&layout]);
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("oit composite"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some(entry), compilation_options: Default::default(), targets: &targets }),
                    multiview_mask: None,
                    cache: None,
                })
            };
            if background {
                Lazy::background(make)
            } else {
                Lazy::new(make)
            }
        };
        Self {
            back: pipeline("fs_back", vec![color.clone()], true),
            front: pipeline("fs_front", vec![color, reactive.clone()], true),
            // By bucket: the front pass also holds the motion target (not written).
            avboit_modulate: [
                pipeline("fs_avboit_modulate", vec![blended(modulate), Some(wgpu::ColorTargetState { format: TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::empty() })], false),
                pipeline("fs_avboit_modulate", vec![blended(modulate)], false),
            ],
            avboit_back: pipeline("fs_avboit_back", vec![blended(add)], false),
            avboit_front: pipeline("fs_avboit_front", vec![blended(add), reactive], false),
            layout,
            uniform: uniform_buffer(device, "oit composite", std::mem::size_of::<CompositeUniform>() as u64),
            targets: None,
            avboit: crate::avboit::Avboit::new(device),
            use_avboit: false,
        }
    }

    /// Queues the AVBOIT composites and resolve for the browser's background compile.
    pub fn prefetch_avboit(&self) {
        self.avboit_modulate.iter().for_each(Lazy::prefetch);
        self.avboit_back.prefetch();
        self.avboit_front.prefetch();
        self.avboit.prefetch();
    }

    /// Allocates the targets for the internal resolution's capacity and records where the
    /// volume textures are valid this frame.
    /// `avboit`: composite with adaptive voxel-based OIT this frame (the grid over `internal`).
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, capacity: (u32, u32), internal: (u32, u32), volume_rect: [u32; 4], avboit: bool) {
        if self.targets.as_ref().is_none_or(|t| t.size != capacity) {
            self.targets = Some(Targets::new(device, capacity));
        }
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&CompositeUniform { volume_rect }));
        self.use_avboit = avboit;
        self.avboit.update(device, queue, avboit, internal, capacity);
    }

    /// Turns the AVBOIT grid off (nothing is composited this frame).
    pub fn disable_avboit(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.use_avboit = false;
        self.avboit.update(device, queue, false, (0, 0), (0, 0));
    }

    /// Whether this frame composites with AVBOIT.
    pub fn uses_avboit(&self) -> bool {
        self.use_avboit
    }

    /// The nearest refracting depth as a one-layer array (AVBOIT: the translucency pass's layer
    /// ranges read it).
    pub fn translucent_depth_array(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.translucent_depth_array)
    }

    /// One bucket's OIT targets (accumulation, transmittance), for the translucency pass's
    /// order-independent surfaces.
    pub fn bucket_targets(&self, bucket: Bucket) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        self.targets.as_ref().map(|t| (&t.accum[bucket as usize], &t.reveal[bucket as usize]))
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
        if self.use_avboit {
            particles.draw_avboit(device, &mut pass, inputs, &t.translucent_depth, &self.avboit, AvboitStage::Shade);
        } else {
            particles.draw(device, &mut pass, inputs, &t.translucent_depth);
        }
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
        let (label, pipeline, attachments) = match (bucket, self.use_avboit) {
            (Bucket::Back, false) => ("oit composite (behind translucency)", &self.back, vec![target(&inputs.targets.hdr)]),
            (Bucket::Front, false) => ("oit composite", &self.front, vec![target(&inputs.targets.hdr), target(&inputs.targets.gbuffer_motion)]),
            (Bucket::Back, true) => ("avboit composite (behind translucency)", &self.avboit_back, vec![target(&inputs.targets.hdr)]),
            (Bucket::Front, true) => ("avboit composite", &self.avboit_front, vec![target(&inputs.targets.hdr), target(&inputs.targets.gbuffer_motion)]),
        };
        let mut pass = rec.render_pass(label, &attachments, None);
        pass.set_viewport(0.0, 0.0, inputs.targets.width as f32, inputs.targets.height as f32, 0.0, 1.0);
        pass.set_bind_group(0, &group, &[]);
        if self.use_avboit {
            // The background through everything in this bucket, then the bucket's own light.
            pass.set_pipeline(&self.avboit_modulate[bucket as usize]);
            pass.draw(0..3, 0..1);
        }
        pass.set_pipeline(pipeline);
        pass.draw(0..3, 0..1);
    }
}

/// The particles' AVBOIT draw into the four OIT targets: weighted colour adds, transmittance
/// multiplies (`fs_particle_avboit` in particles.wgsl).
pub(crate) fn particle_avboit_pipeline(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let pipeline_layout = crate::util::pipeline_layout(device, "particle draw (avboit)", &[layout]);
    let targets = avboit_targets(true);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("particle draw (avboit)"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState { module, entry_point: Some("vs_particle"), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState { module, entry_point: Some("fs_particle_avboit"), compilation_options: Default::default(), targets: &targets }),
        multiview_mask: None,
        cache: None,
    })
}

/// AVBOIT target states of one bucket (weighted colour adds, transmittance multiplies), of both
/// with `both` (front, then back).
pub(crate) fn avboit_targets(both: bool) -> Vec<Option<wgpu::ColorTargetState>> {
    let add = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
    let multiply = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::Src, operation: wgpu::BlendOperation::Add };
    let accum = Some(wgpu::ColorTargetState { format: ACCUM_FORMAT, blend: Some(wgpu::BlendState { color: add, alpha: add }), write_mask: wgpu::ColorWrites::ALL });
    let transmittance = Some(wgpu::ColorTargetState { format: REVEAL_FORMAT, blend: Some(wgpu::BlendState { color: multiply, alpha: multiply }), write_mask: wgpu::ColorWrites::ALL });
    let mut targets = vec![accum.clone(), transmittance.clone()];
    if both {
        targets.extend([accum, transmittance]);
    }
    targets
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
