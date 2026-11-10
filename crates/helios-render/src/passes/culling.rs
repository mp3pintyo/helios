//! GPU-driven culling, LOD selection and visibility-buffer rasterisation (hardware + software,
//! each with an alpha-tested path for masked materials).

use wgpu::BindingResource::TextureView;
use wgpu::{BufferUsages, ShaderStages, TextureFormat};

use crate::gpu_scene::GpuScene;
use crate::profiler::Recorder;
use crate::scene::Scene;
use crate::shader::create_module;
use crate::stats::STATE_SIZE;
use crate::targets::{FrameTargets, DEPTH_FORMAT, VISIBILITY_FORMAT};
use crate::textures::TextureLibrary;
use crate::util::{self, bind, bind_group, bind_group_layout, compute_pipeline, lazy_compute_pipeline, storage_buffer, uniform_buffer, Lazy};
use crate::skin::{skin_entry, skin_resource};
use crate::tessellation::{tessellation_entry, tessellation_resource, Tessellation};
use crate::wind::{wind_entry, wind_resource};

/// Byte offsets into a view's indirect argument buffer (see cull_args.wgsl).
const ARGS_PAGES: u64 = 0;
const ARGS_CLUSTERS: u64 = 16;
const ARGS_EARLY_DRAW: u64 = 32;
const ARGS_LATE_DRAW: u64 = 48;
pub const ARGS_EARLY_SW: u64 = 64;
const ARGS_LATE_SW: u64 = 80;
const ARGS_EARLY_MASKED: u64 = 96;
const ARGS_LATE_MASKED: u64 = 112;
const ARGS_SIZE: u64 = 128;

/// Bindings of the alpha test (`alpha_mask.wgsl`) in every rasteriser.
pub const BINDING_MATERIALS: u32 = 40;
pub const BINDING_ALPHA_MASKS: u32 = 41;
pub const BINDING_ALPHA_SAMPLER: u32 = 42;

/// Layout entries of the alpha test bindings.
pub fn alpha_mask_entries() -> [(u32, wgpu::BindingType); 3] {
    [(BINDING_MATERIALS, bind::uniform()), (BINDING_ALPHA_MASKS, bind::texture_array(true)), (BINDING_ALPHA_SAMPLER, bind::sampler())]
}

/// Resources of the alpha test bindings.
pub fn alpha_mask_resources(scene: &GpuScene) -> [(u32, wgpu::BindingResource<'_>); 3] {
    [
        (BINDING_MATERIALS, scene.materials.as_entire_binding()),
        (BINDING_ALPHA_MASKS, TextureView(&scene.masks.view)),
        (BINDING_ALPHA_SAMPLER, wgpu::BindingResource::Sampler(&scene.masks.sampler)),
    ]
}

/// Occlusion strategy of one culling pass; must match `CULL_MODE` in cull.wgsl.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullMode {
    /// No occlusion culling (shadow views, or occlusion disabled).
    Single = 0,
    /// Draw clusters that were visible last frame.
    Early = 1,
    /// Test the rest against the HZB of the early pass.
    Late = 2,
}

/// Capacities of one culling view's GPU work lists.
#[derive(Clone, Copy, Debug)]
pub struct CullCapacity {
    pub batches: u32,
    pub pages: u32,
    pub clusters: u32,
    /// Alpha-masked hardware clusters (main view; 0 for views that alpha-test per draw).
    pub masked: u32,
}

impl CullCapacity {
    /// Size of the work buffer: page batches, page work items, then up to two cluster
    /// candidates per page item slot (small pages, cull.wgsl), two per 16 bytes.
    fn work_bytes(&self) -> u64 {
        (self.batches + 2 * self.pages) as u64 * 16
    }
}

/// GPU buffers owned by one culling view (main camera or a shadow view).
pub struct CullView {
    pub uniform: wgpu::Buffer,
    /// Page batches, page work items, then cluster candidates.
    pub work: wgpu::Buffer,
    pub visible_clusters: wgpu::Buffer,
    pub state: wgpu::Buffer,
    pub args: wgpu::Buffer,
    pub capacity: CullCapacity,
}

impl CullView {
    pub fn new(device: &wgpu::Device, label: &str, capacity: CullCapacity) -> Self {
        Self {
            uniform: uniform_buffer(device, label, std::mem::size_of::<crate::gpu_types::CullViewUniform>() as u64),
            work: storage_buffer(device, label, capacity.work_bytes(), BufferUsages::empty()),
            // Entries, then the masked draw list (4 slots per entry-sized vec4).
            visible_clusters: storage_buffer(device, label, (capacity.clusters + capacity.masked.div_ceil(4)) as u64 * 16, BufferUsages::empty()),
            state: storage_buffer(device, label, STATE_SIZE, BufferUsages::COPY_SRC),
            args: storage_buffer(device, label, ARGS_SIZE, BufferUsages::INDIRECT),
            capacity,
        }
    }

    /// Makes room for at least `batches` page batches (an instance with up to 64 pages takes
    /// one, an expanded assembly's culling record one per 64 of its parts' pages).
    pub fn reserve_batches(&mut self, device: &wgpu::Device, batches: u32) {
        if batches > self.capacity.batches {
            self.capacity.batches = batches;
            self.work = storage_buffer(device, "culling work", self.capacity.work_bytes(), BufferUsages::empty());
        }
    }

    /// Uses `source`'s work buffer (and capacities). Views culled one after another may share
    /// it: a view's work lists are dead once its clusters are culled (the shadow clipmap levels).
    pub fn share_work(&mut self, source: &CullView) {
        self.work = source.work.clone();
        self.capacity = source.capacity;
    }
}

pub struct Culling {
    cull_layout: wgpu::BindGroupLayout,
    args_layout: wgpu::BindGroupLayout,
    raster_layout: wgpu::BindGroupLayout,
    sw_layout: wgpu::BindGroupLayout,
    merge_layout: wgpu::BindGroupLayout,
    /// The culling passes for scenes without and with skinned instances (`SKINNING` in
    /// dynamic_caster.wgsl), each compiled when a scene first needs it, as the alpha-masked and
    /// voxel rasterisers below ([`Lazy`]): scenes without that content never pay their compiles.
    passes: [Lazy<CullPasses>; 2],
    prepare_pages: wgpu::ComputePipeline,
    prepare_clusters: wgpu::ComputePipeline,
    prepare_early_draw: wgpu::ComputePipeline,
    prepare_late_draw: wgpu::ComputePipeline,
    raster_early: wgpu::RenderPipeline,
    raster_late: wgpu::RenderPipeline,
    /// Alpha-masked hardware clusters ("programmable raster": the fragment shader discards).
    raster_masked_early: Lazy<wgpu::RenderPipeline>,
    raster_masked_late: Lazy<wgpu::RenderPipeline>,
    /// [early depth, early payload, late depth, late payload]
    sw_raster: [wgpu::ComputePipeline; 4],
    /// The same for alpha-masked clusters (each specialisation skips the other's clusters).
    sw_raster_masked: [Lazy<wgpu::ComputePipeline>; 4],
    /// The same for voxel clusters (voxel_raster.wgsl).
    voxel_raster: [Lazy<wgpu::ComputePipeline>; 4],
    merge: wgpu::ComputePipeline,
    dummy_hzb: wgpu::TextureView,
    /// Nanite tessellation of displaced clusters, created when a scene first needs it.
    pub(crate) tessellation: Option<Tessellation>,
}

/// The pipelines of cull.wgsl's passes, specialised for one `SKINNING` value.
struct CullPasses {
    instances: wgpu::ComputePipeline,
    pages: wgpu::ComputePipeline,
    clusters: [wgpu::ComputePipeline; 3],
}

impl CullPasses {
    fn new(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout, skinning: bool) -> Self {
        let skinning = ("SKINNING", if skinning { 1.0 } else { 0.0 });
        let cluster_pipeline = |mode: CullMode| {
            compute_pipeline(device, "cull clusters", module, "cull_clusters", layout, &[("CULL_MODE", mode as u32 as f64), skinning])
        };
        Self {
            instances: compute_pipeline(device, "cull instances", module, "cull_instances", layout, &[skinning]),
            pages: compute_pipeline(device, "cull pages", module, "cull_pages", layout, &[skinning]),
            clusters: [cluster_pipeline(CullMode::Single), cluster_pipeline(CullMode::Early), cluster_pipeline(CullMode::Late)],
        }
    }
}

impl Culling {
    pub fn new(device: &wgpu::Device) -> Self {
        let cull_module = create_module(device, "cull.wgsl");
        let args_module = create_module(device, "cull_args.wgsl");
        let raster_module = create_module(device, "raster.wgsl");
        let sw_module = create_module(device, "sw_raster.wgsl");
        let merge_module = create_module(device, "visbuffer.wgsl");
        let c = ShaderStages::COMPUTE;

        let mut cull_entries = vec![(0, bind::uniform())];
        cull_entries.extend((1..=4).map(|b| (b, bind::storage(true))));
        cull_entries.extend((5..=8).map(|b| (b, bind::storage(false))));
        cull_entries.push((9, bind::texture(false)));
        cull_entries.push(wind_entry());
        cull_entries.push(skin_entry());
        cull_entries.push(tessellation_entry());
        let cull_layout = bind_group_layout(device, "cull", c, &cull_entries);
        let args_layout =
            bind_group_layout(device, "cull args", c, &[(0, bind::uniform()), (1, bind::storage(false)), (2, bind::storage(false))]);
        let mut raster_entries = vec![(0, bind::uniform())];
        raster_entries.extend((1..=6).map(|b| (b, bind::storage(true))));
        raster_entries.extend(alpha_mask_entries());
        raster_entries.push(wind_entry());
        raster_entries.push(skin_entry());
        let raster_layout = bind_group_layout(device, "raster", ShaderStages::VERTEX_FRAGMENT, &raster_entries);
        let mut sw_entries = raster_entries.clone();
        sw_entries.push((7, bind::storage(false)));
        let sw_layout = bind_group_layout(device, "sw raster", c, &sw_entries);
        let merge_layout = bind_group_layout(
            device,
            "visibility merge",
            c,
            &[
                (0, bind::texture_depth()),
                (1, bind::texture_uint()),
                (2, bind::storage(true)),
                (3, bind::storage_texture(TextureFormat::R32Uint)),
                (4, bind::storage_texture(TextureFormat::R32Float)),
                (5, bind::uniform()),
            ],
        );

        let cull_passes = |skinning: bool| {
            let (device, module, layout) = (device.clone(), cull_module.clone(), cull_layout.clone());
            Lazy::new(move || CullPasses::new(&device, &module, &layout, skinning))
        };
        let args_pipeline = |entry: &str| compute_pipeline(device, entry, &args_module, entry, &args_layout, &[]);
        let sw_pipeline = |pass: u32, late: bool, masked: bool| {
            lazy_compute_pipeline(
                device,
                if masked { "sw raster (alpha-masked)" } else { "sw raster" },
                &sw_module,
                "rasterize",
                &sw_layout,
                &[("PASS", pass as f64), ("LATE", if late { 1.0 } else { 0.0 }), ("MASKED", if masked { 1.0 } else { 0.0 })],
            )
        };
        let sw_pipelines = |masked| [sw_pipeline(0, false, masked), sw_pipeline(1, false, masked), sw_pipeline(0, true, masked), sw_pipeline(1, true, masked)];
        let voxel_module = create_module(device, "voxel_raster.wgsl");
        let voxel_pipeline = |pass: u32, late: bool| {
            let constants = [("PASS", pass as f64), ("LATE", if late { 1.0 } else { 0.0 })];
            lazy_compute_pipeline(device, "voxel raster", &voxel_module, "rasterize", &sw_layout, &constants)
        };
        let masked_raster_pipeline = |late: bool| {
            let (device, module, layout) = (device.clone(), raster_module.clone(), raster_layout.clone());
            Lazy::new(move || raster_pipeline(&device, &module, &layout, late, true))
        };

        let dummy_hzb = util::texture(
            device,
            util::TextureSpec {
                label: "dummy hzb",
                width: 1,
                height: 1,
                layers: 1,
                mips: 1,
                format: TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
            },
        )
        .create_view(&Default::default());

        Self {
            passes: [cull_passes(false), cull_passes(true)],
            prepare_pages: args_pipeline("prepare_page_dispatch"),
            prepare_clusters: args_pipeline("prepare_cluster_dispatch"),
            prepare_early_draw: args_pipeline("prepare_early_draw"),
            prepare_late_draw: args_pipeline("prepare_late_draw"),
            raster_early: raster_pipeline(device, &raster_module, &raster_layout, false, false),
            raster_late: raster_pipeline(device, &raster_module, &raster_layout, true, false),
            raster_masked_early: masked_raster_pipeline(false),
            raster_masked_late: masked_raster_pipeline(true),
            // Every scene has opaque clusters: compiled now.
            sw_raster: sw_pipelines(false).map(|p| (*p).clone()),
            sw_raster_masked: sw_pipelines(true),
            voxel_raster: [voxel_pipeline(0, false), voxel_pipeline(1, false), voxel_pipeline(0, true), voxel_pipeline(1, true)],
            merge: compute_pipeline(device, "visibility merge", &merge_module, "merge", &merge_layout, &[]),
            dummy_hzb,
            tessellation: None,
            cull_layout,
            args_layout,
            raster_layout,
            sw_layout,
            merge_layout,
        }
    }

    /// Records culling for `view`.
    ///
    /// * `Single` / `Early`: resets counters, culls instances → pages → clusters and prepares
    ///   the (early) draws.
    /// * `Late`: re-culls the same page work items against `hzb` and prepares the late draws.
    pub fn cull(&self, rec: &mut Recorder, view: &CullView, scene: &GpuScene, mode: CullMode, hzb: Option<&wgpu::TextureView>) {
        self.cull_impl(rec, view, scene, mode, hzb, &scene.atomics);
    }

    /// Culling with a custom atomics buffer (virtual shadow map levels bind their dirty-page
    /// rectangles there instead of streaming feedback and visibility bits).
    pub fn cull_with_atomics(&self, rec: &mut Recorder, view: &CullView, scene: &GpuScene, mode: CullMode, atomics: &wgpu::Buffer) {
        self.cull_impl(rec, view, scene, mode, None, atomics);
    }

    fn cull_impl(
        &self,
        rec: &mut Recorder,
        view: &CullView,
        scene: &GpuScene,
        mode: CullMode,
        hzb: Option<&wgpu::TextureView>,
        atomics: &wgpu::Buffer,
    ) {
        let device = rec.device;
        let passes = &self.passes[scene.skins.skinned_instances as usize];
        let cull_group = bind_group(
            device,
            "cull",
            &self.cull_layout,
            &[
                (0, view.uniform.as_entire_binding()),
                (1, scene.instances.as_entire_binding()),
                (2, scene.pages.as_entire_binding()),
                (3, scene.clusters.as_entire_binding()),
                (4, scene.page_table.as_entire_binding()),
                (5, view.work.as_entire_binding()),
                (6, view.visible_clusters.as_entire_binding()),
                (7, view.state.as_entire_binding()),
                (8, atomics.as_entire_binding()),
                (9, TextureView(hzb.unwrap_or(&self.dummy_hzb))),
                wind_resource(scene),
                skin_resource(scene),
                tessellation_resource(scene),
            ],
        );
        let args_group = bind_group(
            device,
            "cull args",
            &self.args_layout,
            &[(0, view.uniform.as_entire_binding()), (1, view.state.as_entire_binding()), (2, view.args.as_entire_binding())],
        );

        if mode != CullMode::Late {
            rec.encoder.clear_buffer(&view.state, 0, None);
        }
        let mut pass = rec.compute_pass(if mode == CullMode::Late { "cull (late)" } else { "cull" });
        if mode != CullMode::Late {
            pass.set_bind_group(0, &cull_group, &[]);
            pass.set_pipeline(&passes.instances);
            // The scene's instances and the assemblies' culling records.
            pass.dispatch_workgroups(util::workgroups(scene.cull_instance_count(), 64), 1, 1);

            pass.set_pipeline(&self.prepare_pages);
            pass.set_bind_group(0, &args_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&passes.pages);
            pass.set_bind_group(0, &cull_group, &[]);
            pass.dispatch_workgroups_indirect(&view.args, ARGS_PAGES);

            pass.set_pipeline(&self.prepare_clusters);
            pass.set_bind_group(0, &args_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        pass.set_pipeline(&passes.clusters[mode as usize]);
        pass.set_bind_group(0, &cull_group, &[]);
        pass.dispatch_workgroups_indirect(&view.args, ARGS_CLUSTERS);

        pass.set_pipeline(if mode == CullMode::Late { &self.prepare_late_draw } else { &self.prepare_early_draw });
        pass.set_bind_group(0, &args_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn raster_entries<'a>(view: &'a CullView, scene: &'a GpuScene) -> Vec<(u32, wgpu::BindingResource<'a>)> {
        let mut entries = vec![
            (0, view.uniform.as_entire_binding()),
            (1, view.visible_clusters.as_entire_binding()),
            (2, view.state.as_entire_binding()),
            (3, scene.instances.as_entire_binding()),
            (4, scene.clusters.as_entire_binding()),
            (5, scene.page_table.as_entire_binding()),
            (6, scene.geometry.as_entire_binding()),
        ];
        entries.extend(alpha_mask_resources(scene));
        entries.push(wind_resource(scene));
        entries.push(skin_resource(scene));
        entries
    }

    /// Creates the tessellator the first time a scene has displaced instances, points it at the
    /// material textures and graph parameters heights are read from, and selects its dicing
    /// permutation for the scene's displacing graphs.
    pub fn prepare_tessellation(
        &mut self,
        device: &wgpu::Device,
        scene: &Scene,
        gpu: &GpuScene,
        textures: &TextureLibrary,
        sampler: &wgpu::Sampler,
        graph_parameters: &wgpu::Buffer,
    ) {
        if gpu.has_displacement() {
            let tessellation = self.tessellation.get_or_insert_with(|| Tessellation::new(device));
            tessellation.set_textures(&textures.albedo, &textures.data, sampler, graph_parameters);
            tessellation.sync_graphs(device, scene);
        }
    }

    /// The tessellator, if the scene has displaced instances.
    pub fn tessellation(&self, scene: &GpuScene) -> Option<&Tessellation> {
        self.tessellation.as_ref().filter(|_| scene.has_displacement())
    }

    /// Rasterises the clusters selected by the early (`late = false`) or late pass: hardware
    /// clusters into the raster targets, software clusters into the atomic buffer, then merges
    /// both into the final visibility buffer and depth. The early pass clears everything.
    pub fn raster_visibility(&self, rec: &mut Recorder, view: &CullView, scene: &GpuScene, targets: &FrameTargets, late: bool) {
        let device = rec.device;
        let group = bind_group(device, "raster", &self.raster_layout, &Self::raster_entries(view, scene));
        let mut sw_entries = Self::raster_entries(view, scene);
        sw_entries.push((7, targets.sw_target.as_entire_binding()));
        let sw_group = bind_group(device, "sw raster", &self.sw_layout, &sw_entries);
        let tessellation = self.tessellation(scene);
        let tess_groups = tessellation.map(|t| t.main_groups(device, view, scene, &targets.sw_target, targets.tess_lambda()));

        if !late {
            // Only the depth half needs clearing; payloads are read where depth matched.
            rec.encoder.clear_buffer(&targets.sw_target, 0, Some(targets.width as u64 * targets.height as u64 * 4));
        }
        let (color_load, depth_load) = if late {
            (wgpu::LoadOp::Load, wgpu::LoadOp::Load)
        } else {
            (wgpu::LoadOp::Clear(wgpu::Color { r: u32::MAX as f64, g: 0.0, b: 0.0, a: 0.0 }), wgpu::LoadOp::Clear(0.0))
        };
        {
            let mut pass = rec.render_pass(
                if late { "raster hw (late)" } else { "raster hw" },
                &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.hw_visibility,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: color_load, store: wgpu::StoreOp::Store },
                })],
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.hw_depth,
                    depth_ops: Some(wgpu::Operations { load: depth_load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
            );
            // The internal resolution's corner of the targets (see `targets.rs`).
            pass.set_viewport(0.0, 0.0, targets.width as f32, targets.height as f32, 0.0, 1.0);
            pass.set_pipeline(if late { &self.raster_late } else { &self.raster_early });
            pass.set_bind_group(0, &group, &[]);
            pass.draw_indirect(&view.args, if late { ARGS_LATE_DRAW } else { ARGS_EARLY_DRAW });
            // After the opaque clusters, so their depth rejects hidden masked fragments early.
            if scene.masks.any() {
                pass.set_pipeline(if late { &self.raster_masked_late } else { &self.raster_masked_early });
                pass.draw_indirect(&view.args, if late { ARGS_LATE_MASKED } else { ARGS_EARLY_MASKED });
            }
        }

        let merge_group = bind_group(
            device,
            "visibility merge",
            &self.merge_layout,
            &[
                (0, TextureView(&targets.hw_depth)),
                (1, TextureView(&targets.hw_visibility)),
                (2, targets.sw_target.as_entire_binding()),
                (3, TextureView(&targets.visibility)),
                (4, TextureView(&targets.depth)),
                (5, view.uniform.as_entire_binding()),
            ],
        );
        if let Some(t) = tessellation {
            t.clear_main(&mut rec.encoder, !late);
        }
        let mut pass = rec.compute_pass(if late { "raster sw + merge (late)" } else { "raster sw + merge" });
        let base = if late { 2 } else { 0 };
        let args = if late { ARGS_LATE_SW } else { ARGS_EARLY_SW };
        // Displaced clusters' patches are split and diced before either raster pass.
        if let (Some(t), Some(groups)) = (tessellation, &tess_groups) {
            t.generate_main(&mut pass, groups, view, args, late);
        }
        // The depth pass of all clusters (opaque, alpha-masked, voxels and tessellated), then the
        // payload pass.
        for stage in [base, base + 1] {
            pass.set_bind_group(0, &sw_group, &[]);
            pass.set_pipeline(&self.sw_raster[stage]);
            pass.dispatch_workgroups_indirect(&view.args, args);
            if scene.masks.any() {
                pass.set_pipeline(&self.sw_raster_masked[stage]);
                pass.dispatch_workgroups_indirect(&view.args, args);
            }
            if scene.has_voxels() {
                pass.set_pipeline(&self.voxel_raster[stage]);
                pass.dispatch_workgroups_indirect(&view.args, args);
            }
            if let (Some(t), Some(groups)) = (tessellation, &tess_groups) {
                t.raster_main(&mut pass, groups, stage - base);
            }
        }
        pass.set_pipeline(&self.merge);
        pass.set_bind_group(0, &merge_group, &[]);
        pass.dispatch_workgroups(util::workgroups(targets.width, 8), util::workgroups(targets.height, 8), 1);
    }
}

fn raster_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    late: bool,
    masked: bool,
) -> wgpu::RenderPipeline {
    let layout = util::pipeline_layout(device, "raster", &[layout]);
    let constants = [("LATE", if late { 1.0 } else { 0.0 })];
    let targets = [Some(wgpu::ColorTargetState { format: VISIBILITY_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(if masked { "visibility raster (alpha-masked)" } else { "visibility raster" }),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(if masked { "vs_masked" } else { "vs_main" }),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &constants, ..Default::default() },
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(if masked { "fs_masked" } else { "fs_main" }),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}
