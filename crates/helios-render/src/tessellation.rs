//! Nanite tessellation (UE 5.4+): materials displace their surfaces at render time
//! ([`crate::Displacement`]), and the triangles of displaced clusters are tessellated on the GPU
//! into micro-triangles about [`crate::RenderSettings::dice_rate`] pixels long.
//!
//! Per culling pass of the main view (early and late) and once for the shadow pages:
//!
//! ```text
//! emit (one workgroup per software cluster / page item) → split passes → dice args
//!   → dice + raster per grid level (main view: depth pass, then payload pass)
//! ```
//!
//! Culling sends every triangle cluster of a displaced instance to the software list, with its
//! bounds padded by the displacement bound (`displacement.wgsl`). Each triangle is a patch (in
//! fixed-point barycentrics); per edge, a tessellation factor is the undisplaced edge length
//! projected to pixels as if it faced the camera, over the dice rate. Patches with factors above 8
//! are split at the midpoints of those edges (red-green: the split depends on the edge only),
//! through a global queue processed by a fixed number of passes (WebGPU has no forward-progress
//! guarantee for persistent threads; the count adapts to the deepest split seen), the rest are
//! diced on a 2^k grid with geomorphed vertices, displaced along the interpolated normal and
//! rasterised by the software rasteriser's rules (`tess_core.wgsl`, `tess_queue.wgsl`,
//! `tess.wgsl`, `vsm_tess.wgsl`). The visibility payload stays the base triangle's: the G-buffer
//! resolve recovers the displaced point and normal from the depth (`tess_resolve.wgsl`).
//!
//! The height is a texture channel, or a material graph's Displacement output: the dicing
//! pipelines then run a permutation with the scene's displacing graphs ([`graphs`]).

use glam::{Vec3, Vec4};
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ComputePass, ShaderStages};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::gpu_scene::{GpuScene, MAX_MATERIALS};
use crate::material_graphs::MaterialGraphHandle;
use crate::passes::culling::CullView;
use crate::profiler::Recorder;
use crate::scene::{Material, Scene, TextureHandle};
use crate::shader::{compose_variant, create_module, create_module_from};
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer};
use crate::wind::{wind_entry, wind_resource};

/// Binding of the tessellation uniform (`Tessellation` in displacement.wgsl).
pub const BINDING_TESSELLATION: u32 = 67;

/// Layout entry of the tessellation uniform.
pub fn tessellation_entry() -> (u32, wgpu::BindingType) {
    (BINDING_TESSELLATION, bind::uniform())
}

/// The scene's tessellation uniform for [`tessellation_entry`].
pub fn tessellation_resource(scene: &GpuScene) -> (u32, wgpu::BindingResource<'_>) {
    (BINDING_TESSELLATION, scene.tessellation.as_entire_binding())
}

/// Per material (`TessMaterial` in displacement.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TessMaterial {
    /// Magnitude, centre, bound, mesh UV density.
    pub params: Vec4,
    /// Texture layer, channel; graph id ([`graphs::graph_id`], 0: none) and parameter offset,
    /// written by [`write_graph_source`].
    pub source: [u32; 4],
}

/// The per-frame header of the tessellation uniform (`Tessellation` in displacement.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TessView {
    /// Dice rate (pixels), main view pixels per world unit at distance 1, near plane, split passes.
    pub dice: Vec4,
    pub eye: Vec4,
}

/// Size of the tessellation uniform: the header, then [`TessMaterial`] per material.
pub const UNIFORM_SIZE: u64 = (std::mem::size_of::<TessView>() + MAX_MATERIALS * std::mem::size_of::<TessMaterial>()) as u64;

/// Where a displaced material's height comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplacementSource {
    /// A channel of this texture layer ([`crate::Displacement::channel`]).
    Texture(TextureHandle),
    /// The Displacement output of the material's graph.
    Graph(MaterialGraphHandle),
}

/// Whether a material displaces and what its height is: opaque and not alpha-tested (UE allows
/// both; masked displaced materials would need the alpha test in the dicer), with a magnitude,
/// and a graph whose Displacement output is connected (and compiles) or else a texture.
pub fn displacement_source(scene: &Scene, material: &Material) -> Option<DisplacementSource> {
    let d = material.displacement?;
    let opaque = !material.kind.is_translucent() && material.alpha_cutoff.is_none();
    if !opaque || d.magnitude == 0.0 {
        return None;
    }
    match material.graph {
        Some(g) if scene.compiled_material_graph(g).is_some_and(|c| c.displacement.is_some()) => Some(DisplacementSource::Graph(g)),
        _ => Some(DisplacementSource::Texture(d.texture.or(material.texture)?)),
    }
}

/// TessMaterial::source.x of a material without a height layer (`DISPLACEMENT_NO_LAYER`).
const NO_LAYER: u32 = u32::MAX;

/// The material table of the tessellation uniform (graph sources are patched in afterwards by
/// [`write_graph_source`], which knows the parameter offsets).
pub fn material_table(scene: &Scene) -> Vec<TessMaterial> {
    let mut table: Vec<TessMaterial> = scene
        .materials
        .iter()
        .map(|m| match (m.displacement, displacement_source(scene, m)) {
            (Some(d), Some(source)) => TessMaterial {
                params: Vec4::new(d.magnitude, d.center, d.bound(), d.uv_density),
                source: match source {
                    DisplacementSource::Texture(layer) => [layer.0, d.channel.min(3), 0, 0],
                    DisplacementSource::Graph(_) => [NO_LAYER, 0, 0, 0],
                },
            },
            _ => TessMaterial::default(),
        })
        .collect();
    table.resize(MAX_MATERIALS, TessMaterial::default());
    table.truncate(MAX_MATERIALS);
    table
}

/// Points material `index`'s height at graph `id` with its parameters at `offset` in the graph
/// parameter block (`TessMaterial::source.zw`; id 0: none).
pub fn write_graph_source(queue: &wgpu::Queue, scene: &GpuScene, index: usize, id: u32, offset: u32) {
    let at = std::mem::size_of::<TessView>() + index * std::mem::size_of::<TessMaterial>() + std::mem::offset_of!(TessMaterial, source) + 8;
    queue.write_buffer(&scene.tessellation, at as u64, bytemuck::cast_slice(&[id, offset]));
}

/// Patches per split queue and per dice list (16 bytes each: 32 MiB in all).
const QUEUE_CAPACITY: u32 = 1 << 19;
const DICE_CAPACITY: u32 = 1 << 18;
/// u32 words of a view's state buffer (see tess_queue.wgsl): per culling pass, then frame totals.
const STATE_WORDS: u64 = 16;
const PASS_WORDS: u64 = 12;
/// A view's indirect dispatches: the split pass, then the dicing of each grid level (bytes).
const DISPATCH_BYTES: u64 = 20 * 4;
const DICE_DISPATCH: u64 = 4 * 4;
/// Split passes: their number follows the deepest split of recent frames within these bounds.
const MIN_PASSES: u32 = 1;
const MAX_PASSES: u32 = 14;
const INITIAL_PASSES: u32 = 6;

/// Tessellation statistics of a recent frame (both culling passes of the main view, and the
/// shadow pages).
#[derive(Clone, Copy, Debug, Default)]
pub struct TessellationStats {
    /// Patches routed (culled or not) and micro-triangles diced, main view and shadow pages.
    pub patches: u32,
    pub micro_triangles: u32,
    pub shadow_patches: u32,
    pub shadow_micro_triangles: u32,
    /// Patches lost to full lists (the dice rate is then raised).
    pub overflow: u32,
    /// Split passes per view and the factor on the dice rate that keeps the lists from
    /// overflowing (1 = none).
    pub split_passes: u32,
    pub dice_scale: f32,
}

/// Per grid level, per raster pass (the shadow view has one).
type DicePipelines = Vec<Vec<wgpu::ComputePipeline>>;

struct ViewPipelines {
    emit: Vec<wgpu::ComputePipeline>,
    split: wgpu::ComputePipeline,
    prepare_split: wgpu::ComputePipeline,
    prepare_dice: wgpu::ComputePipeline,
    dice: DicePipelines,
}

/// The dicing pipelines of a displacing-graph permutation ([`graphs`]).
struct GraphPermutation {
    main: DicePipelines,
    shadow: DicePipelines,
}

/// Permutations kept compiled (editing a graph makes new ones).
const MAX_PERMUTATIONS: usize = 8;

impl ViewPipelines {
    /// Records emit, the split passes and the dice arguments (`groups`: the queue's and the
    /// one-thread passes' bind groups).
    fn generate(&self, pass: &mut ComputePass, groups: [&wgpu::BindGroup; 2], emit: usize, args: (&wgpu::Buffer, u64), dispatch: &wgpu::Buffer, passes: u32) {
        let [queue, prepare] = groups;
        pass.set_bind_group(0, queue, &[]);
        pass.set_pipeline(&self.emit[emit]);
        pass.dispatch_workgroups_indirect(args.0, args.1);
        for _ in 0..passes {
            pass.set_bind_group(0, prepare, &[]);
            pass.set_pipeline(&self.prepare_split);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_bind_group(0, queue, &[]);
            pass.set_pipeline(&self.split);
            pass.dispatch_workgroups_indirect(dispatch, 0);
        }
        pass.set_bind_group(0, prepare, &[]);
        pass.set_pipeline(&self.prepare_dice);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Records the dicing and rasterisation of every grid level (`dice`: this view's pipelines or
    /// a permutation's).
    fn raster(pass: &mut ComputePass, dice: &DicePipelines, group: &wgpu::BindGroup, stage: usize, dispatch: &wgpu::Buffer) {
        pass.set_bind_group(0, group, &[]);
        for (level, pipelines) in dice.iter().enumerate() {
            pass.set_pipeline(&pipelines[stage]);
            pass.dispatch_workgroups_indirect(dispatch, DICE_DISPATCH + level as u64 * 16);
        }
    }
}

/// The main view's bind groups for one frame.
pub struct MainGroups {
    generate: wgpu::BindGroup,
    prepare: wgpu::BindGroup,
    dice: wgpu::BindGroup,
}

/// Inputs of the shadow page tessellation.
pub struct ShadowInputs<'a> {
    pub vsm: &'a wgpu::Buffer,
    pub draws: &'a wgpu::Buffer,
    pub sw_depth: &'a wgpu::Buffer,
    pub local: &'a wgpu::Buffer,
    /// The page items' software dispatch (`draws` at this offset).
    pub dispatch_offset: u64,
}

pub struct Tessellation {
    patches: wgpu::Buffer,
    main_state: wgpu::Buffer,
    shadow_state: wgpu::Buffer,
    main_dispatch: wgpu::Buffer,
    shadow_dispatch: wgpu::Buffer,
    prepare_layout: wgpu::BindGroupLayout,
    main_generate_layout: wgpu::BindGroupLayout,
    main_dice_layout: wgpu::BindGroupLayout,
    shadow_layout: wgpu::BindGroupLayout,
    main: ViewPipelines,
    shadow: ViewPipelines,
    /// Dicing permutations by the hash of their displacing graphs, the one in use, and the scene
    /// versions it was chosen at.
    permutations: HashMap<u64, GraphPermutation>,
    permutation: Option<u64>,
    permutation_versions: Option<(u64, u64)>,
    /// The material layers (heights; graph code samples colours too), their sampler and the
    /// graph parameter block (updated every frame).
    textures: Option<GraphInputs>,
    readback: wgpu::Buffer,
    readback_status: Arc<AtomicU32>,
    passes: u32,
    dice_scale: f32,
    stats: TessellationStats,
}

/// Override constants of every tessellation pipeline.
const CAPACITIES: [(&str, f64); 2] = [("QUEUE_CAPACITY", QUEUE_CAPACITY as f64), ("DICE_CAPACITY", DICE_CAPACITY as f64)];

/// What the dicing passes read besides the scene: material texture arrays, their sampler and the
/// graph parameter block.
struct GraphInputs {
    albedo: wgpu::TextureView,
    data: wgpu::TextureView,
    sampler: wgpu::Sampler,
    parameters: wgpu::Buffer,
}

fn main_dice(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout) -> DicePipelines {
    (0..4)
        .map(|level| {
            (0..2)
                .map(|stage| {
                    let constants = [CAPACITIES[0], CAPACITIES[1], ("LEVEL", level as f64), ("PASS", stage as f64)];
                    compute_pipeline(device, "tessellation dice", module, "dice", layout, &constants)
                })
                .collect()
        })
        .collect()
}

fn shadow_dice(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout) -> DicePipelines {
    let bias = ("SLOPE_BIAS", crate::vsm::SLOPE_BIAS as f64);
    (0..4)
        .map(|level| vec![compute_pipeline(device, "vsm tessellation dice", module, "dice", layout, &[CAPACITIES[0], CAPACITIES[1], ("LEVEL", level as f64), bias])])
        .collect()
}

const IDLE: u32 = 0;
const MAPPING: u32 = 1;
const READY: u32 = 2;

impl Tessellation {
    pub fn new(device: &wgpu::Device) -> Self {
        let c = ShaderStages::COMPUTE;
        let read = |b: u32| (b, bind::storage(true));
        let common = [read(3), read(4), read(5), read(6), (8, bind::storage(false)), (9, bind::storage(false)), wind_entry(), tessellation_entry()];
        let textures = [
            (40, bind::uniform()),
            (43, bind::texture_array(true)),
            (44, bind::sampler()),
            (graphs::BINDING_ALBEDO, bind::texture_array(true)),
            (graphs::BINDING_GRAPH_PARAMETERS, bind::uniform()),
        ];
        let mut entries = vec![(0, bind::uniform()), read(1), read(2)];
        entries.extend(common);
        let main_generate_layout = bind_group_layout(device, "tessellation", c, &entries);
        let mut entries = vec![(0, bind::uniform()), read(1), (7, bind::storage(false))];
        entries.extend(common);
        entries.extend(textures);
        let main_dice_layout = bind_group_layout(device, "tessellation dice", c, &entries);
        let mut entries = vec![(0, bind::uniform()), read(1), (7, bind::storage(false)), (60, bind::uniform())];
        entries.extend(common);
        entries.extend(textures);
        let shadow_layout = bind_group_layout(device, "tessellation (shadows)", c, &entries);
        let prepare_layout = bind_group_layout(device, "tessellation dispatch", c, &[(9, bind::storage(false)), (10, bind::storage(false))]);

        let pipelines = |module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout, label: &str, entry: &str, extra: &[(&str, f64)]| {
            let constants: Vec<(&str, f64)> = CAPACITIES.iter().copied().chain(extra.iter().copied()).collect();
            compute_pipeline(device, label, module, entry, layout, &constants)
        };
        let module = create_module(device, "tess.wgsl");
        let main = ViewPipelines {
            emit: [0.0, 1.0].map(|late| pipelines(&module, &main_generate_layout, "tessellation emit", "emit", &[("LATE", late)])).into(),
            split: pipelines(&module, &main_generate_layout, "tessellation split", "split", &[]),
            prepare_split: pipelines(&module, &prepare_layout, "tessellation split args", "prepare_split", &[]),
            prepare_dice: pipelines(&module, &prepare_layout, "tessellation dice args", "prepare_dice", &[]),
            dice: main_dice(device, &module, &main_dice_layout),
        };
        let module = create_module(device, "vsm_tess.wgsl");
        let shadow = ViewPipelines {
            emit: vec![pipelines(&module, &shadow_layout, "vsm tessellation emit", "emit", &[])],
            split: pipelines(&module, &shadow_layout, "vsm tessellation split", "split", &[]),
            prepare_split: pipelines(&module, &prepare_layout, "vsm tessellation split args", "prepare_split", &[]),
            prepare_dice: pipelines(&module, &prepare_layout, "vsm tessellation dice args", "prepare_dice", &[]),
            dice: shadow_dice(device, &module, &shadow_layout),
        };
        let state = |label| storage_buffer(device, label, STATE_WORDS * 4, wgpu::BufferUsages::COPY_SRC);
        let dispatch = |label| storage_buffer(device, label, DISPATCH_BYTES, wgpu::BufferUsages::INDIRECT);
        Self {
            patches: storage_buffer(device, "tessellation patches", (2 * QUEUE_CAPACITY as u64 + 4 * DICE_CAPACITY as u64) * 16, wgpu::BufferUsages::empty()),
            main_state: state("tessellation state"),
            shadow_state: state("tessellation state (shadows)"),
            main_dispatch: dispatch("tessellation dispatch"),
            shadow_dispatch: dispatch("tessellation dispatch (shadows)"),
            prepare_layout,
            main_generate_layout,
            main_dice_layout,
            shadow_layout,
            main,
            shadow,
            permutations: HashMap::new(),
            permutation: None,
            permutation_versions: None,
            textures: None,
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tessellation stats"),
                size: 32,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            readback_status: Arc::new(AtomicU32::new(IDLE)),
            passes: INITIAL_PASSES,
            dice_scale: 1.0,
            stats: TessellationStats::default(),
        }
    }

    /// The material texture arrays and sampler heights are read from, and the graph parameter
    /// block of displacing graphs.
    pub fn set_textures(&mut self, albedo: &wgpu::TextureView, data: &wgpu::TextureView, sampler: &wgpu::Sampler, parameters: &wgpu::Buffer) {
        self.textures = Some(GraphInputs { albedo: albedo.clone(), data: data.clone(), sampler: sampler.clone(), parameters: parameters.clone() });
    }

    /// Selects the dicing permutation of the scene's displacing graphs, compiling it on first use
    /// (none: the base pipelines).
    pub fn sync_graphs(&mut self, device: &wgpu::Device, scene: &Scene) {
        let versions = (scene.material_version, scene.graphs.version);
        if self.permutation_versions == Some(versions) {
            return;
        }
        self.permutation_versions = Some(versions);
        let set = graphs::displacing_graphs(scene);
        if set.is_empty() {
            self.permutation = None;
            return;
        }
        let key = graphs::set_hash(&set);
        if !self.permutations.contains_key(&key) {
            if self.permutations.len() >= MAX_PERMUTATIONS {
                self.permutations.clear();
            }
            let code = graphs::scene_dispatch_module(scene, &set);
            let module = |name: &str| create_module_from(device, "tessellation (displacing graphs)", compose_variant(name, &[("displacement_graphs_off.wgsl", &code)]));
            let permutation = GraphPermutation {
                main: main_dice(device, &module("tess.wgsl"), &self.main_dice_layout),
                shadow: shadow_dice(device, &module("vsm_tess.wgsl"), &self.shadow_layout),
            };
            self.permutations.insert(key, permutation);
        }
        self.permutation = Some(key);
    }

    fn permutation(&self) -> Option<&GraphPermutation> {
        self.permutation.and_then(|key| self.permutations.get(&key))
    }

    /// The tessellation uniform's header for this frame: dice rate (raised while lists overflow),
    /// the main view's pixel scale and near plane, the split pass count, the camera.
    pub fn view(&self, dice_rate: f32, pixels_per_unit: f32, near: f32, eye: Vec3) -> TessView {
        TessView { dice: Vec4::new(dice_rate.max(0.25) * self.dice_scale, pixels_per_unit, near, self.passes as f32), eye: eye.extend(0.0) }
    }

    pub fn stats(&self) -> TessellationStats {
        self.stats
    }

    fn texture_entries(&self) -> [(u32, wgpu::BindingResource<'_>); 4] {
        let t = self.textures.as_ref().expect("tessellation textures are set before recording");
        [
            (43, TextureView(&t.data)),
            (44, Sampler(&t.sampler)),
            (graphs::BINDING_ALBEDO, TextureView(&t.albedo)),
            (graphs::BINDING_GRAPH_PARAMETERS, t.parameters.as_entire_binding()),
        ]
    }

    /// Bind groups of the main view's passes.
    pub fn main_groups(&self, device: &wgpu::Device, view: &CullView, scene: &GpuScene, sw_target: &wgpu::Buffer) -> MainGroups {
        let common = || {
            [
                (3, scene.instances.as_entire_binding()),
                (4, scene.clusters.as_entire_binding()),
                (5, scene.page_table.as_entire_binding()),
                (6, scene.geometry.as_entire_binding()),
                (8, self.patches.as_entire_binding()),
                (9, self.main_state.as_entire_binding()),
                wind_resource(scene),
                tessellation_resource(scene),
            ]
        };
        let mut entries = vec![(0, view.uniform.as_entire_binding()), (1, view.visible_clusters.as_entire_binding()), (2, view.state.as_entire_binding())];
        entries.extend(common());
        let generate = bind_group(device, "tessellation", &self.main_generate_layout, &entries);
        let mut entries = vec![(0, view.uniform.as_entire_binding()), (1, view.visible_clusters.as_entire_binding()), (7, sw_target.as_entire_binding())];
        entries.extend(common());
        entries.push((40, scene.materials.as_entire_binding()));
        entries.extend(self.texture_entries());
        MainGroups {
            generate,
            prepare: self.prepare_group(device, &self.main_state, &self.main_dispatch),
            dice: bind_group(device, "tessellation dice", &self.main_dice_layout, &entries),
        }
    }

    fn prepare_group(&self, device: &wgpu::Device, state: &wgpu::Buffer, dispatch: &wgpu::Buffer) -> wgpu::BindGroup {
        bind_group(device, "tessellation dispatch", &self.prepare_layout, &[(9, state.as_entire_binding()), (10, dispatch.as_entire_binding())])
    }

    /// Resets the main view's lists before a culling pass (`first`: the frame's first, which also
    /// resets its totals).
    pub fn clear_main(&self, encoder: &mut wgpu::CommandEncoder, first: bool) {
        encoder.clear_buffer(&self.main_state, 0, Some(if first { STATE_WORDS } else { PASS_WORDS } * 4));
    }

    /// Emits, splits and prepares the dicing of a culling pass's tessellated clusters, whose
    /// software list `view` dispatches at `args_offset`.
    pub fn generate_main(&self, pass: &mut ComputePass, groups: &MainGroups, view: &CullView, args_offset: u64, late: bool) {
        self.main.generate(pass, [&groups.generate, &groups.prepare], late as usize, (&view.args, args_offset), &self.main_dispatch, self.passes);
    }

    /// Dices and rasterises the main view's patches: `stage` 0 the depth, 1 the payload.
    pub fn raster_main(&self, pass: &mut ComputePass, groups: &MainGroups, stage: usize) {
        let dice = self.permutation().map_or(&self.main.dice, |p| &p.main);
        ViewPipelines::raster(pass, dice, &groups.dice, stage, &self.main_dispatch);
    }

    /// Tessellates and rasterises the shadow page items of displaced clusters into the atlas'
    /// u32 mirror (a compute pass of its own, before the page clear copies it).
    pub fn record_shadows(&self, rec: &mut Recorder, scene: &GpuScene, inputs: &ShadowInputs) {
        let mut entries = vec![
            (0, inputs.vsm.as_entire_binding()),
            (1, inputs.draws.as_entire_binding()),
            (3, scene.instances.as_entire_binding()),
            (4, scene.clusters.as_entire_binding()),
            (5, scene.page_table.as_entire_binding()),
            (6, scene.geometry.as_entire_binding()),
            (7, inputs.sw_depth.as_entire_binding()),
            (8, self.patches.as_entire_binding()),
            (9, self.shadow_state.as_entire_binding()),
            (40, scene.materials.as_entire_binding()),
            (60, inputs.local.as_entire_binding()),
            wind_resource(scene),
            tessellation_resource(scene),
        ];
        entries.extend(self.texture_entries());
        let group = bind_group(rec.device, "tessellation (shadows)", &self.shadow_layout, &entries);
        let prepare = self.prepare_group(rec.device, &self.shadow_state, &self.shadow_dispatch);
        rec.encoder.clear_buffer(&self.shadow_state, 0, None);
        let mut pass = rec.compute_pass("vsm tessellation");
        self.shadow.generate(&mut pass, [&group, &prepare], 0, (inputs.draws, inputs.dispatch_offset), &self.shadow_dispatch, self.passes);
        let dice = self.permutation().map_or(&self.shadow.dice, |p| &p.shadow);
        ViewPipelines::raster(&mut pass, dice, &group, 0, &self.shadow_dispatch);
    }

    /// Copies the frame's totals for the statistics, unless a read-back is in flight; call
    /// [`Tessellation::after_submit`] after submitting.
    pub fn record_readback(&self, encoder: &mut wgpu::CommandEncoder) -> bool {
        if self.readback_status.load(Ordering::Acquire) != IDLE {
            return false;
        }
        encoder.copy_buffer_to_buffer(&self.main_state, PASS_WORDS * 4, &self.readback, 0, Some(16));
        encoder.copy_buffer_to_buffer(&self.shadow_state, PASS_WORDS * 4, &self.readback, 16, Some(16));
        true
    }

    pub fn after_submit(&self, recorded: bool) {
        if !recorded {
            return;
        }
        self.readback_status.store(MAPPING, Ordering::Release);
        let status = self.readback_status.clone();
        self.readback.map_async(wgpu::MapMode::Read, .., move |result| {
            status.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
        });
    }

    /// Takes a finished read-back: statistics, and the feedback that sizes the next frames' work
    /// (split passes follow the deepest split; the dice rate rises while lists overflow).
    pub fn poll(&mut self) {
        if self.readback_status.load(Ordering::Acquire) != READY {
            return;
        }
        if let Ok(view) = self.readback.get_mapped_range(..) {
            let w: &[u32] = bytemuck::cast_slice(&view);
            // [patches, micro-triangles, lost, deepest splitting pass] per view.
            let deepest = w[3].max(w[7]);
            self.passes = next_pass_count(self.passes, deepest);
            let overflow = w[2] + w[6];
            self.dice_scale = if overflow > 0 { (self.dice_scale * 1.25).min(8.0) } else { (self.dice_scale * 0.98).max(1.0) };
            self.stats = TessellationStats {
                patches: w[0],
                micro_triangles: w[1],
                shadow_patches: w[4],
                shadow_micro_triangles: w[5],
                overflow,
                split_passes: self.passes,
                dice_scale: self.dice_scale,
            };
        }
        self.readback.unmap();
        self.readback_status.store(IDLE, Ordering::Release);
    }
}

/// Split passes for the next frames given the current count and the deepest pass that split
/// patches: one more while the last pass still split (its children were diced coarser than
/// asked), one fewer once two passes were idle.
fn next_pass_count(passes: u32, deepest: u32) -> u32 {
    if deepest >= passes {
        (passes + 1).min(MAX_PASSES)
    } else if deepest + 1 < passes {
        (passes - 1).max(MIN_PASSES)
    } else {
        passes
    }
}

pub mod graphs;

#[cfg(test)]
mod crack_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_count_follows_the_deepest_split() {
        assert_eq!(next_pass_count(6, 6), 7);
        assert_eq!(next_pass_count(6, 5), 6);
        assert_eq!(next_pass_count(6, 2), 5);
        assert_eq!(next_pass_count(MAX_PASSES, MAX_PASSES), MAX_PASSES);
        assert_eq!(next_pass_count(MIN_PASSES, 0), MIN_PASSES);
    }

    #[test]
    fn uniform_matches_the_shader_layout() {
        // vec4 dice, vec4 eye, then 256 × (vec4 params, vec4<u32> source).
        assert_eq!(std::mem::size_of::<TessView>(), 32);
        assert_eq!(std::mem::size_of::<TessMaterial>(), 32);
        assert_eq!(UNIFORM_SIZE, 32 + 256 * 32);
        let source = include_str!("shaders/tess_queue.wgsl");
        assert!(source.contains(&format!("const STATE_TOTAL_PATCHES: u32 = {PASS_WORDS}u;")));
    }
}
