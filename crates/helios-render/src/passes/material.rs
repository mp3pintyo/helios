//! Material classification and per-shading-model G-buffer resolve.
//!
//! Pixels are binned into 8×8 tiles per shading model, then every model is resolved by its own
//! pipeline (specialised with the `MATERIAL_KIND` override constant), dispatched indirectly over
//! just its tiles. This keeps each material shader small and coherent instead of one divergent
//! über-shader — the WebGPU counterpart of Nanite's material classification passes.
//!
//! Each model has a second permutation (`UV_EXTENSIONS`) with the rarer texture coordinate
//! features — the second UV set, wrap modes other than repeat, occlusion maps in a layer of their
//! own — compiled on first use, only for scenes whose materials need them
//! ([`GpuScene::uv_extensions`]): their code and registers would slow down every pixel otherwise.
//! Likewise, skinning (posing corners, previous positions for motion vectors) is a `SKINNING`
//! permutation used only while the scene has skinned instances: in the forest it would take the
//! resolve from 0.39 to 0.74 ms although no pixel is skinned.

use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat};

use super::Samplers;
use crate::gpu_scene::GpuScene;
use crate::passes::culling::CullView;
use crate::profiler::Recorder;
use crate::rvt::RuntimeVirtualTexture;
use crate::shader::create_module;
use crate::svt::StreamingVirtualTexture;
use crate::targets::FrameTargets;
use crate::textures::TextureLibrary;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, workgroups};
use crate::water::BINDING_WATER;
use crate::skin::{skin_entry, skin_resource};
use crate::wind::{wind_entry, wind_resource};

/// The merged visibility depth (voxel_resolve.wgsl finds the voxel a pixel hit from it).
const BINDING_DEPTH: u32 = 91;

/// Shading models with a resolve pipeline (see `MATERIAL_*` in material.wgsl); 7 is the sky.
/// Glass (5) is translucent: it never reaches the visibility buffer (see translucency.rs).
const KINDS: [u32; 6] = [0, 1, 2, 3, 4, 7];

pub struct MaterialPass {
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    classify_layout: wgpu::BindGroupLayout,
    classify: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    /// Resolve pipelines per shading model and permutation ([`permutation`]), compiled when first
    /// needed.
    resolve: [Option<Vec<(u32, wgpu::ComputePipeline)>>; 4],
    /// The permutation in use.
    permutation: usize,
}

/// Index of a resolve permutation: `UV_EXTENSIONS` in bit 0, `SKINNING` in bit 1.
fn permutation(uv_extensions: bool, skinning: bool) -> usize {
    uv_extensions as usize | ((skinning as usize) << 1)
}

impl MaterialPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "gbuffer.wgsl");
        let layout = bind_group_layout(
            device,
            "material",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::texture_uint()),
                (3, bind::storage(true)),
                (4, bind::storage(true)),
                (5, bind::storage(true)),
                (6, bind::storage(true)),
                (7, bind::storage(true)),
                (8, bind::storage(true)),
                (9, bind::texture_array(true)),
                (10, bind::texture_array(true)),
                (11, bind::sampler()),
                (12, bind::uniform()),
                (13, bind::storage(true)),
                (14, bind::storage(false)),
                (15, bind::texture(true)),
                (16, bind::texture(true)),
                (17, bind::sampler()),
                (18, bind::storage_texture(TextureFormat::Rgba8Unorm)),
                (19, bind::storage_texture(TextureFormat::Rgba16Float)),
                (20, bind::storage_texture(TextureFormat::Rgba16Float)),
                (45, bind::uniform()),
                (46, bind::texture_uint()),
                (47, bind::texture(true)),
                (48, bind::texture(true)),
                (49, bind::storage_texture(TextureFormat::R32Uint)),
                (80, bind::texture(true)),
                wind_entry(),
                skin_entry(),
                (BINDING_WATER, bind::uniform()),
                (BINDING_DEPTH, bind::texture(false)),
            ],
        );
        let resolve = [Some(resolve_pipelines(device, &module, &layout, 0)), None, None, None];
        let classify_module = create_module(device, "classify.wgsl");
        let classify_layout = bind_group_layout(
            device,
            "material classify",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::texture_uint()),
                (3, bind::storage(true)),
                (4, bind::storage(true)),
                (5, bind::storage(false)),
            ],
        );
        Self {
            classify: compute_pipeline(device, "material classify", &classify_module, "classify", &classify_layout, &[]),
            prepare: compute_pipeline(device, "material tiles", &classify_module, "prepare_tiles", &classify_layout, &[]),
            module,
            resolve,
            layout,
            classify_layout,
            permutation: 0,
        }
    }

    /// Selects the resolve permutation the scene's materials and instances need, compiling it on
    /// first use.
    pub fn sync(&mut self, device: &wgpu::Device, uv_extensions: bool, skinning: bool) {
        self.permutation = permutation(uv_extensions, skinning);
        let slot = &mut self.resolve[self.permutation];
        if slot.is_none() {
            *slot = Some(resolve_pipelines(device, &self.module, &self.layout, self.permutation));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        rec: &mut Recorder,
        frame: &wgpu::Buffer,
        view: &CullView,
        scene: &GpuScene,
        targets: &FrameTargets,
        textures: &TextureLibrary,
        rvt: &RuntimeVirtualTexture,
        svt: &StreamingVirtualTexture,
        water: &wgpu::Buffer,
        samplers: &Samplers,
    ) {
        let group = bind_group(
            rec.device,
            "material",
            &self.layout,
            &[
                (0, frame.as_entire_binding()),
                (1, scene.materials.as_entire_binding()),
                (2, TextureView(&targets.visibility)),
                (3, view.visible_clusters.as_entire_binding()),
                (4, scene.instances.as_entire_binding()),
                (5, scene.clusters.as_entire_binding()),
                (6, scene.page_table.as_entire_binding()),
                (7, scene.geometry.as_entire_binding()),
                (8, targets.material_tiles.as_entire_binding()),
                (9, TextureView(&textures.albedo)),
                (10, TextureView(&textures.data)),
                (11, Sampler(&samplers.material)),
                (12, rvt.params.as_entire_binding()),
                (13, rvt.cache.entries.as_entire_binding()),
                (14, rvt.cache.requests.as_entire_binding()),
                (15, TextureView(&rvt.albedo)),
                (16, TextureView(&rvt.detail)),
                (17, Sampler(&samplers.linear)),
                (18, TextureView(&targets.gbuffer_albedo)),
                (19, TextureView(&targets.gbuffer_normal)),
                (20, TextureView(&targets.gbuffer_motion)),
                (45, svt.params.as_entire_binding()),
                (46, TextureView(&svt.page_table)),
                (47, TextureView(&svt.albedo)),
                (48, TextureView(&svt.normal)),
                (49, TextureView(svt.feedback_view())),
                (80, TextureView(&svt.material)),
                wind_resource(scene),
                skin_resource(scene),
                (BINDING_WATER, water.as_entire_binding()),
                (BINDING_DEPTH, TextureView(&targets.depth)),
            ],
        );
        let classify_group = bind_group(
            rec.device,
            "material classify",
            &self.classify_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, scene.materials.as_entire_binding()),
                (2, TextureView(&targets.visibility)),
                (3, view.visible_clusters.as_entire_binding()),
                (4, scene.instances.as_entire_binding()),
                (5, targets.material_tiles.as_entire_binding()),
            ],
        );
        rec.encoder.clear_buffer(&targets.material_tiles, 0, Some(64));
        let mut pass = rec.compute_pass("material classify + resolve");
        pass.set_bind_group(0, &classify_group, &[]);
        pass.set_pipeline(&self.classify);
        pass.dispatch_workgroups(workgroups(targets.width, 8), workgroups(targets.height, 8), 1);
        pass.set_pipeline(&self.prepare);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_bind_group(0, &group, &[]);
        for (kind, pipeline) in self.resolve[self.permutation].iter().flatten() {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups_indirect(&targets.material_tiles, (8 + kind * 4) as u64 * 4);
        }
    }
}

/// One resolve pipeline per shading model.
fn resolve_pipelines(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout, permutation: usize) -> Vec<(u32, wgpu::ComputePipeline)> {
    KINDS
        .iter()
        .map(|&kind| {
            let flag = |bit: usize| ((permutation >> bit) & 1) as f64;
            let constants = [("MATERIAL_KIND", kind as f64), ("UV_EXTENSIONS", flag(0)), ("SKINNING", flag(1))];
            (kind, compute_pipeline(device, "material resolve", module, "resolve", layout, &constants))
        })
        .collect()
}
