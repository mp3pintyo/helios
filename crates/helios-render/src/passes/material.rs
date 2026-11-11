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
//! resolve from 0.39 to 0.74 ms although no pixel is skinned. The displaced-surface resolve of
//! tessellated pixels is a `TESSELLATION` permutation, for scenes with displaced instances, and
//! the debug views' recolouring a `DEBUG_VIEWS` one, for debug views only: kept in the lit
//! view's pipelines, the inputs of a dozen debug colourings stay alive across the material
//! evaluation, and the resolve ran 1.4x slower (cathedral at 1440p: 1.01 ms instead of 0.73).
//!
//! Materials with a Substrate slab or a material graph get **extended pipelines** in resolve
//! slots 8–23 (`substrate_gpu.rs` assigns one per distinct shading model and graph): the
//! resolve composed with `resolve_extended.wgsl` and the graph's generated code, bound to a
//! second group (slab table, graph parameters, slab texels). Classification bins their tiles
//! apart, so the plain pipelines stay exactly as they are. Modules are cached by graph hash and
//! pipelines by slot, model, graph, permutation and the Substrate switch: editing a graph
//! compiles one module; changing a parameter compiles nothing.
//!
//! Only the shading models the scene's materials use get a pipeline (and a dispatch: the others'
//! tile lists are empty), each a compile of the 24 KB resolve of 0.6–1.1 s in the browser. In the
//! browser the pipelines are compiled in the background ([`crate::pipelines`]); the
//! `TESSELLATION` permutation a scene with displaced surfaces will need is queued while the first
//! frames still resolve the undisplaced surfaces.

use std::collections::HashMap;

use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat};

use super::Samplers;
use crate::gpu_scene::GpuScene;
use crate::passes::culling::CullView;
use crate::profiler::Recorder;
use crate::rvt::RuntimeVirtualTexture;
use crate::shader::{create_module, create_module_from, extended_resolve};
use crate::substrate_gpu::{ExtendedMaterials, ExtendedSlot};
use crate::svt::StreamingVirtualTexture;
use crate::targets::FrameTargets;
use crate::textures::TextureLibrary;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, compute_pipeline_with, workgroups};
use crate::river::BINDING_RIVER_SURFACE;
use crate::water::BINDING_WATER;
use crate::plant_sim::{sim_entry, sim_resource};
use crate::plant_wind::{plant_entry, plant_resource};
use crate::skin::{skin_entry, skin_resource};
use crate::tessellation::{tessellation_entry, tessellation_resource};
use crate::wind::{wind_entry, wind_resource};
use crate::pipelines::{in_group, Group, Lazy};

/// The merged visibility depth (voxel_resolve.wgsl finds the voxel a pixel hit from it).
const BINDING_DEPTH: u32 = 91;
/// Tessellated pixels' barycentrics on their base triangle (tess_resolve.wgsl).
const BINDING_TESS_LAMBDA: u32 = 92;

/// Shading models with a resolve pipeline (see `MATERIAL_*` in material.wgsl); 7 is the sky.
/// Glass (5) is translucent: it never reaches the visibility buffer (see translucency.rs).
const KINDS: [u32; 6] = [0, 1, 2, 3, 4, 7];
/// The sky's slot; slots below `PLAIN_SLOTS` are the shading models'.
pub const SKY_SLOT: u32 = 7;
pub const PLAIN_SLOTS: u32 = 8;
/// Extended pipelines a frame can use (slots 8–23; classify.wgsl's tile buffer layout).
pub const EXTENDED_SLOTS: u32 = 16;
/// Resolve slots in all: the tile buffer's counts.
pub const RESOLVE_SLOTS: u32 = PLAIN_SLOTS + EXTENDED_SLOTS;
/// Word offset of the per-slot indirect dispatch arguments in the tile buffer.
const TILE_ARGS: u64 = RESOLVE_SLOTS as u64;

/// An extended pipeline: slot, shading model, graph hash (0: none), permutation, Substrate on.
type ExtendedKey = (u32, u32, u64, usize, bool);

pub struct MaterialPass {
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    classify_layout: wgpu::BindGroupLayout,
    classify: Lazy<wgpu::ComputePipeline>,
    prepare: Lazy<wgpu::ComputePipeline>,
    /// Resolve pipelines by permutation ([`permutation`]) and shading model, created when first
    /// needed.
    resolve: HashMap<(usize, u32), Lazy<wgpu::ComputePipeline>>,
    /// The permutation in use.
    permutation: usize,
    /// The shading models the scene's materials use (the sky's always), in [`KINDS`] order.
    kinds: Vec<u32>,
    /// Extended resolve modules by graph hash (0: slab materials without a graph) and pipelines.
    extended_modules: HashMap<u64, wgpu::ShaderModule>,
    extended_pipelines: HashMap<ExtendedKey, Lazy<wgpu::ComputePipeline>>,
    /// The extended pipelines of this frame, by slot.
    extended: Vec<(u32, ExtendedKey)>,
}

/// Index of a resolve permutation: `UV_EXTENSIONS` in bit 0, `SKINNING` in bit 1, `TESSELLATION`
/// in bit 2, `DEBUG_VIEWS` in bit 3.
fn permutation(uv_extensions: bool, skinning: bool, tessellation: bool, debug_views: bool) -> usize {
    uv_extensions as usize | ((skinning as usize) << 1) | ((tessellation as usize) << 2) | ((debug_views as usize) << 3)
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
                plant_entry(),
                sim_entry(),
                (BINDING_WATER, bind::uniform()),
                (BINDING_RIVER_SURFACE, bind::texture(false)),
                (BINDING_DEPTH, bind::texture(false)),
                tessellation_entry(),
                (BINDING_TESS_LAMBDA, bind::texture_uint()),
            ],
        );
        // None compiled yet: the first frame's `sync` compiles the permutation the scene needs
        // (the valley's tessellation one, say, never the plain one) for the models it uses.
        let resolve = HashMap::new();
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
            kinds: Vec::new(),
            extended_modules: HashMap::new(),
            extended_pipelines: HashMap::new(),
            extended: Vec::new(),
        }
    }

    /// Selects the resolve permutation the scene's materials and instances (and a debug view)
    /// need and the extended pipelines of its slab and graph materials (in the same permutation),
    /// creating them on first use, for the shading models in `kinds` (bit per model, from
    /// [`GpuScene::resolve_kinds`]). `tessellation_wanted`: the scene's displaced surfaces will
    /// be tessellated once the browser has compiled tessellation ([`Group::Tessellation`]); that
    /// permutation is created (queued) now.
    #[allow(clippy::too_many_arguments)]
    pub fn sync(
        &mut self,
        device: &wgpu::Device,
        uv_extensions: bool,
        skinning: bool,
        tessellation: bool,
        tessellation_wanted: bool,
        debug_views: bool,
        kinds: u32,
        extended: &ExtendedMaterials,
        substrate: bool,
    ) {
        self.permutation = permutation(uv_extensions, skinning, tessellation, debug_views);
        self.kinds = KINDS.iter().copied().filter(|&k| k == SKY_SLOT || kinds & (1 << k) != 0).collect();
        let plain = self.kinds.clone();
        self.create_resolve(device, self.permutation, &plain);
        self.extended.clear();
        for s in &extended.slots {
            let key = self.create_extended(device, extended, s, self.permutation, substrate);
            self.extended.push((s.slot, key));
        }
        if tessellation_wanted && !tessellation {
            let wanted = permutation(uv_extensions, skinning, true, debug_views);
            in_group(Group::Tessellation, || {
                self.create_resolve(device, wanted, &plain);
                for s in &extended.slots {
                    self.create_extended(device, extended, s, wanted, substrate);
                }
            });
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
        river_surface: &wgpu::TextureView,
        samplers: &Samplers,
        extended: Option<&wgpu::BindGroup>,
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
                plant_resource(scene),
                sim_resource(scene),
                (BINDING_WATER, water.as_entire_binding()),
                (BINDING_RIVER_SURFACE, TextureView(river_surface)),
                (BINDING_DEPTH, TextureView(&targets.depth)),
                tessellation_resource(scene),
                (BINDING_TESS_LAMBDA, TextureView(targets.tess_lambda())),
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
        rec.encoder.clear_buffer(&targets.material_tiles, 0, Some(RESOLVE_SLOTS as u64 * 4));
        let mut pass = rec.compute_pass("material classify + resolve");
        pass.set_bind_group(0, &classify_group, &[]);
        pass.set_pipeline(&self.classify);
        pass.dispatch_workgroups(workgroups(targets.width, 8), workgroups(targets.height, 8), 1);
        pass.set_pipeline(&self.prepare);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_bind_group(0, &group, &[]);
        let args = |slot: u32| (TILE_ARGS + slot as u64 * 4) * 4;
        for &kind in &self.kinds {
            pass.set_pipeline(&self.resolve[&(self.permutation, kind)]);
            pass.dispatch_workgroups_indirect(&targets.material_tiles, args(kind));
        }
        if let Some(group) = extended {
            pass.set_bind_group(1, group, &[]);
            for (slot, key) in &self.extended {
                pass.set_pipeline(&self.extended_pipelines[key]);
                pass.dispatch_workgroups_indirect(&targets.material_tiles, args(*slot));
            }
        }
    }
}

impl MaterialPass {
    /// Creates the extended pipeline of slot `s` in `permutation` (if new); returns its key.
    fn create_extended(&mut self, device: &wgpu::Device, extended: &ExtendedMaterials, s: &ExtendedSlot, permutation: usize, substrate: bool) -> ExtendedKey {
        let hash = s.graph.as_ref().map_or(0, |g| g.0);
        let key = (s.slot, s.kind, hash, permutation, substrate);
        if !self.extended_pipelines.contains_key(&key) {
            let module = self.extended_modules.entry(hash).or_insert_with(|| {
                let code = s.graph.as_ref().map(|g| g.1.as_str());
                create_module_from(device, if code.is_some() { "graph material resolve" } else { "slab material resolve" }, extended_resolve(code, s.displaces))
            });
            let flag = |bit: usize| ((permutation >> bit) & 1) as f64;
            let constants = [
                ("MATERIAL_KIND", s.kind as f64),
                ("RESOLVE_SLOT", s.slot as f64),
                ("UV_EXTENSIONS", flag(0)),
                ("SKINNING", flag(1)),
                ("TESSELLATION", flag(2)),
                ("DEBUG_VIEWS", flag(3)),
                ("SUBSTRATE", substrate as u32 as f64),
            ];
            let layouts = [&self.layout, &extended.resolve_layout];
            let pipeline = compute_pipeline_with(device, "extended material resolve", module, "resolve", &layouts, &constants);
            self.extended_pipelines.insert(key, pipeline);
        }
        key
    }

    /// Queues the resolve pipelines a scene about to be switched to will need — `permutation`'s
    /// flags, the shading models in `kinds` and the extended `slots` — without changing what the
    /// current scene's frames resolve with ([`crate::Renderer::prefetch_next_scene`]).
    #[allow(clippy::too_many_arguments)]
    pub fn prefetch(
        &mut self,
        device: &wgpu::Device,
        uv_extensions: bool,
        skinning: bool,
        tessellation: bool,
        debug_views: bool,
        kinds: u32,
        extended: &ExtendedMaterials,
        slots: &[ExtendedSlot],
        substrate: bool,
    ) {
        let permutation = permutation(uv_extensions, skinning, tessellation, debug_views);
        let plain: Vec<u32> = KINDS.iter().copied().filter(|&k| k == SKY_SLOT || kinds & (1 << k) != 0).collect();
        self.create_resolve(device, permutation, &plain);
        for s in slots {
            self.create_extended(device, extended, s, permutation, substrate);
        }
    }

    /// Creates the resolve pipelines of `permutation` for the shading models `kinds`.
    fn create_resolve(&mut self, device: &wgpu::Device, permutation: usize, kinds: &[u32]) {
        for &kind in kinds {
            self.resolve.entry((permutation, kind)).or_insert_with(|| {
                let flag = |bit: usize| ((permutation >> bit) & 1) as f64;
                let constants = [("MATERIAL_KIND", kind as f64), ("RESOLVE_SLOT", kind as f64), ("UV_EXTENSIONS", flag(0)), ("SKINNING", flag(1)), ("TESSELLATION", flag(2)), ("DEBUG_VIEWS", flag(3))];
                compute_pipeline(device, "material resolve", &self.module, "resolve", &self.layout, &constants)
            });
        }
    }
}
