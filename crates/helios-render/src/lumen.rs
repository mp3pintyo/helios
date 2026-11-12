//! Lumen-style dynamic global illumination and reflections (software ray tracing).
//!
//! * **Global distance field** — 4 clipmap levels of 128³ (0.25 m … 2 m voxels) around the
//!   camera, composited on the GPU from every instance's mesh SDF (heightfields for terrain).
//!   A level is rebuilt when the camera crosses a brick boundary or the scene changes; moved
//!   instances only recompose the bricks under them (`lumen_dynamic.rs`).
//! * **Surface cache** — a 64³-per-level voxel lighting volume: each voxel near a surface is
//!   lit by the sun (distance-field soft shadows), the sky (distance-field occlusion), emissive
//!   surfaces and one bounce from the radiance cache; refreshed a quarter per frame.
//! * **Radiance cache** — 2 levels of 24³ world probes (L1 SH) traced through the distance
//!   field; they feed the surface cache back, so light bounces accumulate over frames.
//! * **Final gather** — screen probes (one per 16×16 tile, 64 rays each, screen trace →
//!   distance field → surface cache → sky), plus **adaptive probes** on the surfaces of 8×8
//!   cells that no uniform probe around them lies on (thin and depth-discontinuous geometry),
//!   filtered and projected to SH, then interpolated per pixel; **reflections** trace GGX rays
//!   the same way.
//! * **Near field** (`near_field.rs`) — the first metres of every gather, reflection and
//!   local-light shadow ray trace the mesh SDFs of nearby instances, which resolve objects far
//!   smaller than a global voxel; the global field takes over beyond.
//! * **Mesh cards** (`cards.rs`) — the surface cache proper: per-instance cards lit per texel;
//!   gather and reflection hits read them where they match the hit surface and fall back to the
//!   voxel lighting elsewhere. The global field stores the closest instance per voxel, so its
//!   hits know which instance's cards to read.
//! * **Displaced surfaces** (`lumen_displaced.rs`) — instances whose material displaces their
//!   mesh (Nanite tessellation) are represented by a distance field and cards baked from the
//!   displaced surface, so all of the above follow the displacement.
//! * **World cache** (`world_cache.rs`) — optional (UE 5.9's Lumen World Cache): sparse
//!   world-space cells filled by paths traced from the screen; ray hits read them behind the
//!   cards in place of the voxel lighting, or in place of the cards too.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{BindingType, ShaderStages, TextureFormat as F, TextureUsages as U};

use crate::cards::{CaptureInputs, CardCache, CardUniform};
use crate::lumen_displaced::{DisplaceInputs, DisplacedSurfaces, Surface};
use crate::lumen_dynamic::{ComposeParams, DynamicField, DynamicFieldStats, FieldBindings, LevelPlacement, SdfInstance};
use crate::lumen_light_grid::LumenLightGrid;
use crate::motion::SceneChanges;
use crate::near_field::{NearField, NearFieldUniform};
use crate::profiler::Recorder;
use crate::scene::{Instance, Scene};
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::textures::TextureLibrary;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer, storage_buffer_init, uniform_buffer, workgroups};
use crate::world_cache::{WorldCache, WorldCacheInputs, WorldCacheMode};
use crate::pipelines::{in_group, Group, Lazy};

pub const SDF_LEVELS: usize = 4;
const SDF_RES: u32 = 128;
const LIGHT_RES: u32 = 64;
const FIRST_VOXEL: f32 = 0.25;
const BRICK: u32 = 8;
const PROBE_RES: u32 = 24;
const PROBE_SPACING: [f32; 2] = [2.0, 8.0];
const PROBE_UPDATE_FRACTION: u32 = 8;
const PROBE_TILE: u32 = 16;
const PROBE_STRIDE: u64 = 11 * 16;
/// Adaptive screen probes are placed per cell of this many pixels (`ADAPTIVE_CELL` in
/// lumen_gather.wgsl).
const ADAPTIVE_CELL: u32 = 8;
/// Words before the per-cell table of the adaptive probe buffer: indirect dispatch size, probes
/// placed, probes requested (`ADAPTIVE_HEADER` in lumen_gather.wgsl).
const ADAPTIVE_HEADER_WORDS: u64 = 8;
/// Two-sided (open-mesh) distance fields are a shell half a voxel thick. Beyond this many metres
/// the shell reads as solid well above the surface - lights near it sit inside the "geometry" and
/// are shadowed - which is almost always a terrain or ground sheet baked as a coarse volume.
const MAX_SHELL: f32 = 0.5;
/// Alpha-masked instances whose surface is less solid than this are left out of Lumen.
const MIN_COVERAGE: f32 = 0.02;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LumenUniform {
    sdf_levels: [Vec4; SDF_LEVELS],
    probe_levels: [Vec4; 2],
    level_count: u32,
    frame: u32,
    sdf_enabled: u32,
    max_trace_distance: f32,
    sky_intensity: f32,
    probe_update_fraction: u32,
    voxel_update_phase: u32,
    /// [`WorldCacheMode::gpu`].
    world_cache: u32,
    near_field: NearFieldUniform,
    cards: CardUniform,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MeshSdfDesc {
    origin: Vec4,
    res: [u32; 4],
    params: Vec4,
}

struct Level {
    origin: Vec3,
    voxel: f32,
    /// Needs a rebuild (moved or scene changed).
    dirty: bool,
    /// Rebuild scheduled for this frame.
    compose_now: bool,
    params: wgpu::Buffer,
    candidates: wgpu::Buffer,
    candidate_count: u32,
}

/// Renderer resources the mesh-card capture and lighting read besides Lumen's own: the sun's
/// virtual shadow map, the materials (captured per texel; average albedo and emission until
/// then) with their graphs' slab table and parameters, and the queue (capture uploads).
pub struct CardInputs<'a> {
    pub vsm: &'a crate::vsm::VirtualShadowMap,
    /// Sunlight through glass (coloured patches bounce coloured light).
    pub glass: &'a crate::glass_shadow::GlassShadow,
    pub textures: &'a TextureLibrary,
    pub materials: &'a wgpu::Buffer,
    pub extended: &'a crate::substrate_gpu::ExtendedMaterials,
    pub queue: &'a wgpu::Queue,
    /// The volumetric fog's uniform (height fog on reflection hits).
    pub fog: &'a wgpu::Buffer,
    pub options: LumenOptions,
}

/// Lumen's switches from [`crate::RenderSettings`].
#[derive(Clone, Copy, Debug)]
pub struct LumenOptions {
    /// Screen traces pass hits on pixels last frame did not light (UE 5.8's
    /// `ScreenTraces.HZBTraversal.SkipUnlitHits`).
    pub skip_unlit_hits: bool,
    /// Height fog between a reflecting surface and its world-traced hit (`r.Lumen.HeightFog`).
    pub reflection_fog: bool,
    /// Trace reflections per pixel (off: Lumen Lite's screen-space reflections write them).
    pub traced_reflections: bool,
    /// The surface caches importance-sample local lights from the light grid (off: every
    /// light in range; `lumen_light_grid.rs`).
    pub light_sampling: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GatherOptions {
    skip_unlit_hits: u32,
    reflection_fog: u32,
    _pad: [u32; 2],
}

/// Per-entry-point pipeline with its own (exact) bind group layout.
pub(crate) struct Stage {
    layout: wgpu::BindGroupLayout,
    pub pipeline: Lazy<wgpu::ComputePipeline>,
}

impl Stage {
    pub fn new(device: &wgpu::Device, module: &wgpu::ShaderModule, entry: &str, bindings: &[(u32, BindingType)]) -> Self {
        let layout = bind_group_layout(device, entry, ShaderStages::COMPUTE, bindings);
        Self { pipeline: compute_pipeline(device, entry, module, entry, &layout, &[]), layout }
    }

    /// A debug view's stage: compiled when the view is first shown, not with the feature (three
    /// of them compiled on every visit: 0.25 s in Firefox's DXC).
    pub fn debug(device: &wgpu::Device, module: &wgpu::ShaderModule, entry: &str, bindings: &[(u32, BindingType)]) -> Self {
        let layout = bind_group_layout(device, entry, ShaderStages::COMPUTE, bindings);
        Self { pipeline: crate::util::lazy_compute_pipeline(device, entry, module, entry, &layout, &[]), layout }
    }

    pub fn group(&self, device: &wgpu::Device, entries: &[(u32, wgpu::BindingResource)]) -> wgpu::BindGroup {
        bind_group(device, "lumen", &self.layout, entries)
    }
}

/// Resolution-dependent final-gather resources.
struct GatherTargets {
    size: (u32, u32),
    /// Uniform screen probes, then up to [`adaptive_rows`] rows of adaptive ones.
    probes: wgpu::Buffer,
    /// Indirect dispatch size of the probe passes and adaptive probe count, then per
    /// [`ADAPTIVE_CELL`]² cell its adaptive probes (see `adaptive_alloc` in lumen_gather.wgsl).
    adaptive: wgpu::Buffer,
    atlas: wgpu::TextureView,
    reflection_raw: wgpu::TextureView,
    reflection: [wgpu::TextureView; 2],
}

/// The bindings the screen traces and reflections read besides the tracing ones: the fog, the
/// sun's transmittance and the gather options (lumen_gather.wgsl 150-152).
const REFLECT_BINDINGS: [u32; 3] = [160, 161, 162];

pub struct Lumen {
    uniform: wgpu::Buffer,
    /// `GatherOptions` of lumen_gather.wgsl.
    gather_options: wgpu::Buffer,
    pub global_sdf: wgpu::TextureView,
    global_sdf_storage: wgpu::TextureView,
    /// Closest instance per global-SDF voxel (16 bits, two voxels along x per texel).
    instance_ids: wgpu::TextureView,
    voxel_lighting: [wgpu::TextureView; 2],
    radiance_probes: wgpu::Buffer,
    sdf_instances: wgpu::Buffer,
    sdf_descs: wgpu::Buffer,
    sdf_words: wgpu::Buffer,
    bricks: wgpu::Buffer,
    near_field: NearField,
    cards: CardCache,
    /// Local-light candidates of the surface caches (stochastic light sampling).
    light_grid: LumenLightGrid,
    /// Incremental updates for moved instances.
    dynamic: DynamicField,
    levels: Vec<Level>,
    /// Displaced variants of meshes, which displaced instances are represented by.
    displaced: DisplacedSurfaces,
    /// Per distance-field instance: its field's mesh-space box and its world box.
    instance_aabbs: Vec<helios_geometry::Aabb>,
    instance_bounds: Vec<(Vec3, Vec3)>,
    /// Per distance-field instance: whether it is composed into the global field (alpha-masked
    /// instances are not, see `upload_scene`).
    in_global_field: Vec<bool>,
    mask_coverage: MaskCoverage,
    has_sdf: bool,
    versions: (u64, u64, u64, u64),
    probe_origins: [Vec3; 2],
    frame: u32,
    /// Which of `voxel_lighting` holds the latest relighting (see `set_surface_interval`).
    voxels_latest: usize,
    /// Frames between relightings of the voxel surface cache (`lumen_surface_interval`).
    surface_interval: u32,
    gather: Option<GatherTargets>,
    cull_bricks: Stage,
    compose: Stage,
    update_voxels: Stage,
    update_probes: Stage,
    place_probes: Stage,
    request_adaptive: Stage,
    place_adaptive: Stage,
    finish_adaptive: Stage,
    trace_probes: Stage,
    probe_sh: Stage,
    integrate: Stage,
    reflect_trace: Stage,
    reflect_temporal: Stage,
    light_cards: Stage,
    debug_cards: Stage,
    /// UE 5.9's world cache (`world_cache.rs`), allocated when first enabled.
    world_cache: WorldCache,
}

fn texture_3d(device: &wgpu::Device, label: &str, size: (u32, u32, u32), format: F) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: size.2 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: U::TEXTURE_BINDING | U::STORAGE_BINDING,
        view_formats: &[],
    })
}

impl Lumen {
    /// Queues the card capture pipelines of a scene about to be switched to (`active`: the graphs
    /// its resolve runs).
    pub fn prefetch_captures(&mut self, device: &wgpu::Device, scene: &Scene, active: &std::collections::HashSet<u64>) {
        self.cards.prefetch_captures(device, scene, active);
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let levels = SDF_LEVELS as u32;
        let sdf = texture_3d(device, "global sdf", (SDF_RES, SDF_RES, SDF_RES * levels), F::Rgba8Snorm);
        let lighting = [0, 1].map(|i| {
            texture_3d(device, &format!("voxel lighting {i}"), (LIGHT_RES, LIGHT_RES, LIGHT_RES * levels), F::Rgba16Float)
                .create_view(&Default::default())
        });
        let probe_count = (PROBE_RES * PROBE_RES * PROBE_RES * 2) as u64;

        let sdf_module = create_module(device, "lumen_sdf.wgsl");
        let light_module = create_module(device, "lumen_light.wgsl");
        let gather_module = create_module(device, "lumen_gather.wgsl");
        let card_module = create_module(device, "lumen_card_light.wgsl");
        let ids = texture_3d(device, "sdf instance ids", (SDF_RES / 2, SDF_RES, SDF_RES * levels), F::R32Uint);
        let u = bind::uniform;
        let tex = |f: bool| bind::texture(f);
        let tex3 = |f: bool| bind::texture_3d(f);
        let ro = || bind::storage(true);
        let rw = || bind::storage(false);
        let hdr = || bind::storage_texture(F::Rgba16Float);
        // The gather passes' mesh-card bindings: data, pages, capture, lighting, instance ids.
        let card_bindings = [(70, ro()), (71, ro()), (72, bind::texture_uint()), (73, tex(false)), (74, bind::texture_3d_uint())];
        // ... and the world cache's cells (lumen_world_cache_common.wgsl), read at every hit.
        let card_bindings = [&card_bindings[..], &[(150, bind::texture_uint())]].concat();
        let with_cards = |bindings: &[(u32, BindingType)]| [bindings, &card_bindings].concat();

        let level_list = (0..SDF_LEVELS)
            .map(|i| Level {
                origin: Vec3::ZERO,
                voxel: FIRST_VOXEL * (1 << i) as f32,
                dirty: true,
                compose_now: false,
                params: uniform_buffer(device, "sdf compose", std::mem::size_of::<ComposeParams>() as u64),
                candidates: storage_buffer(device, "sdf candidates", 16, wgpu::BufferUsages::empty()),
                candidate_count: 0,
            })
            .collect();

        Self {
            uniform: uniform_buffer(device, "lumen", std::mem::size_of::<LumenUniform>() as u64),
            gather_options: uniform_buffer(device, "lumen gather options", std::mem::size_of::<GatherOptions>() as u64),
            global_sdf: sdf.create_view(&Default::default()),
            global_sdf_storage: sdf.create_view(&Default::default()),
            instance_ids: ids.create_view(&Default::default()),
            voxel_lighting: lighting,
            radiance_probes: storage_buffer(device, "radiance cache", probe_count * 64, wgpu::BufferUsages::empty()),
            sdf_instances: storage_buffer(device, "sdf instances", 16, wgpu::BufferUsages::empty()),
            sdf_descs: storage_buffer(device, "sdf descs", 16, wgpu::BufferUsages::empty()),
            sdf_words: storage_buffer(device, "sdf words", 16, wgpu::BufferUsages::empty()),
            bricks: storage_buffer(device, "sdf bricks", 4096 * 64 * 4, wgpu::BufferUsages::empty()),
            near_field: NearField::new(device),
            cards: CardCache::new(device),
            light_grid: LumenLightGrid::new(device),
            // The global distance field is built (and its shadows traced) whether GI is on or not:
            // compiled with the core set in the browser.
            dynamic: in_group(Group::Core, || DynamicField::new(device, &sdf_module, SDF_LEVELS)),
            levels: level_list,
            displaced: DisplacedSurfaces::new(),
            instance_aabbs: Vec::new(),
            instance_bounds: Vec::new(),
            in_global_field: Vec::new(),
            mask_coverage: MaskCoverage::default(),
            has_sdf: false,
            versions: (u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            probe_origins: [Vec3::splat(f32::MAX); 2],
            frame: 0,
            voxels_latest: 0,
            surface_interval: 1,
            gather: None,
            cull_bricks: in_group(Group::Core, || Stage::new(device, &sdf_module, "cull_bricks", &[(0, u()), (1, ro()), (4, ro()), (5, rw())])),
            compose: in_group(Group::Core, || Stage::new(
                device,
                &sdf_module,
                "compose",
                &[
                    (0, u()),
                    (1, ro()),
                    (2, ro()),
                    (3, ro()),
                    (5, rw()),
                    (6, u()),
                    (7, bind::texture_array(true)),
                    (8, bind::sampler()),
                    (9, bind::storage_texture_3d(F::Rgba8Snorm)),
                    (12, ro()),
                    (70, bind::storage_texture_3d(F::R32Uint)),
                ],
            )),
            update_voxels: Stage::new(
                device,
                &light_module,
                "update_voxels",
                &[
                    (0, u()),
                    (1, u()),
                    (2, tex3(true)),
                    (3, tex3(true)),
                    (4, bind::sampler()),
                    (5, bind::storage_texture_3d(F::Rgba16Float)),
                    (6, rw()),
                    (7, ro()),
                    (8, tex(true)),
                    (10, bind::sampler()),
                    (11, ro()),
                    (163, u()),
                    (164, ro()),
                ],
            ),
            update_probes: Stage::new(
                device,
                &light_module,
                "update_probes",
                &with_cards(&[(0, u()), (1, u()), (2, tex3(true)), (3, tex3(true)), (4, bind::sampler()), (6, rw()), (9, tex(true)), (10, bind::sampler())]),
            ),
            place_probes: Stage::new(device, &gather_module, "place_probes", &[(0, u()), (2, tex(false)), (3, tex(false)), (10, rw()), (85, rw())]),
            request_adaptive: Stage::new(device, &gather_module, "request_adaptive", &[(0, u()), (2, tex(false)), (3, tex(false)), (10, rw()), (85, rw())]),
            place_adaptive: Stage::new(device, &gather_module, "place_adaptive", &[(0, u()), (2, tex(false)), (3, tex(false)), (10, rw()), (85, rw())]),
            finish_adaptive: Stage::new(device, &gather_module, "finish_adaptive", &[(0, u()), (10, rw()), (85, rw())]),
            trace_probes: Stage::new(
                device,
                &gather_module,
                "trace_probes",
                &with_cards(&[
                    (0, u()),
                    (1, u()),
                    (2, tex(false)),
                    (4, tex(false)),
                    (5, tex(true)),
                    (6, tex(true)),
                    (7, tex3(true)),
                    (8, tex3(true)),
                    (9, bind::sampler()),
                    (10, rw()),
                    (11, ro()),
                    (12, hdr()),
                    (18, ro()),
                    (19, ro()),
                    (20, tex3(true)),
                    (REFLECT_BINDINGS[0], u()),
                    (REFLECT_BINDINGS[1], tex(true)),
                    (REFLECT_BINDINGS[2], u()),
                ]),
            ),
            probe_sh: Stage::new(device, &gather_module, "probe_sh", &[(0, u()), (10, rw()), (13, tex(false)), (86, ro())]),
            integrate: Stage::new(
                device,
                &gather_module,
                "integrate",
                &[(0, u()), (1, u()), (2, tex(false)), (3, tex(false)), (10, rw()), (11, ro()), (14, hdr()), (86, ro())],
            ),
            reflect_trace: Stage::new(
                device,
                &gather_module,
                "reflect_trace",
                &with_cards(&[
                    (0, u()),
                    (1, u()),
                    (2, tex(false)),
                    (3, tex(false)),
                    (4, tex(false)),
                    (5, tex(true)),
                    (6, tex(true)),
                    (7, tex3(true)),
                    (8, tex3(true)),
                    (9, bind::sampler()),
                    (11, ro()),
                    (15, hdr()),
                    (18, ro()),
                    (19, ro()),
                    (20, tex3(true)),
                    (REFLECT_BINDINGS[0], u()),
                    (REFLECT_BINDINGS[1], tex(true)),
                    (REFLECT_BINDINGS[2], u()),
                ]),
            ),
            reflect_temporal: Stage::new(
                device,
                &gather_module,
                "reflect_temporal",
                &[(0, u()), (4, tex(false)), (9, bind::sampler()), (15, hdr()), (16, tex(false)), (17, tex(true)), (18, ro())],
            ),
            light_cards: Stage::new(
                device,
                &card_module,
                "light_cards",
                &[
                    (0, u()),
                    (1, u()),
                    (2, tex3(true)),
                    (4, bind::sampler()),
                    (5, ro()),
                    (6, ro()),
                    (7, tex(true)),
                    (8, ro()),
                    (9, ro()),
                    (11, bind::texture_uint()),
                    (13, hdr()),
                    (14, ro()),
                    (15, u()),
                    (16, ro()),
                    (17, bind::texture_depth()),
                    (18, bind::sampler_comparison()),
                    (19, ro()),
                    (20, tex3(true)),
                    (21, u()),
                    (22, bind::texture_array(true)),
                    (23, bind::sampler()),
                    (28, tex(false)),
                    (90, u()),
                    (91, tex(true)),
                    (163, u()),
                    (164, rw()),
                ],
            ),
            debug_cards: Stage::debug(
                device,
                &card_module,
                "debug_cards",
                &[
                    (0, u()),
                    (1, u()),
                    (2, tex3(true)),
                    (3, tex3(true)),
                    (4, bind::sampler()),
                    (9, ro()),
                    (10, ro()),
                    (11, bind::texture_uint()),
                    (12, tex(false)),
                    (24, bind::texture_3d_uint()),
                    (25, tex(false)),
                    (26, tex(false)),
                    (27, hdr()),
                ],
            ),
            world_cache: WorldCache::new(device),
        }
    }

    /// Where ray hits read the world cache (off: cards and voxel lighting only).
    pub fn set_world_cache(&mut self, mode: WorldCacheMode) {
        self.world_cache.mode = mode;
    }

    /// Statistics of the world cache (natively with `HELIOS_WORLD_CACHE_STATS`).
    pub fn world_cache_stats(&self) -> crate::world_cache::WorldCacheStats {
        self.world_cache.latest
    }

    /// Current reflection result (half resolution, rgb radiance, a = validity).
    pub fn reflection(&self, current: usize) -> Option<&wgpu::TextureView> {
        self.gather.as_ref().map(|g| &g.reflection[current])
    }

    /// Uploads mesh SDFs and instances when the scene changes (moved instances incrementally,
    /// see `lumen_dynamic.rs`) or a displaced variant is ready (`tessellation`: displaced
    /// materials displace, see `lumen_displaced.rs`), places the clipmaps around the camera and
    /// writes the uniforms.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        changes: &SceneChanges,
        camera_position: Vec3,
        frame: u32,
        targets: &FrameTargets,
        tessellation: bool,
    ) {
        self.frame = frame;
        let versions = (scene.geometry_version, scene.instance_version, scene.material_version, scene.texture_version);
        let displaced = self.displaced.sync(scene, tessellation);
        if versions != self.versions || changes.structural || displaced {
            self.versions = versions;
            self.world_cache.invalidate();
            self.displaced.assign(scene);
            // Uploading borrows Lumen mutably and the variants immutably.
            let variants = std::mem::replace(&mut self.displaced, DisplacedSurfaces::new());
            let surfaces = variants.surfaces(scene);
            let surface_of = variants.surface_of();
            let (sdf_instances, coverage) = self.upload_scene(device, queue, scene, &surfaces, surface_of);
            // Every scene instance's index among the distance-field instances (= card instance).
            let mut card_instances = vec![u32::MAX; scene.instances.len()];
            for (k, &i) in sdf_instances.iter().enumerate() {
                card_instances[i] = k as u32;
            }
            let fields: Vec<_> = surfaces.iter().map(Surface::sdf).collect();
            self.near_field.upload(device, queue, scene, &fields, surface_of, &card_instances, &coverage);
            self.cards.upload(device, queue, scene, &surfaces, surface_of, &sdf_instances, &self.instance_bounds);
            // The field's voxels take their card pair's captured mean (once captured).
            for (record, pair) in self.dynamic.records.iter_mut().zip(self.cards.instance_pairs()) {
                record.pair = pair;
            }
            queue.write_buffer(&self.sdf_instances, 0, bytemuck::cast_slice(&self.dynamic.records));
            drop(surfaces);
            self.displaced = variants;
            for level in &mut self.levels {
                level.dirty = true;
            }
        } else if !changes.moved.is_empty() {
            let placements: Vec<LevelPlacement> = self.placements();
            let moved = self.dynamic.apply(queue, scene, changes, &self.sdf_instances, &self.instance_aabbs, &mut self.instance_bounds, &self.in_global_field, &placements);
            let fields: Vec<_> = self.displaced.surfaces(scene).iter().map(Surface::sdf).collect();
            self.near_field.move_instances(queue, scene, &fields, moved.iter().map(|&(i, _)| i));
            for (i, k) in moved {
                self.cards.move_instance(queue, k, scene.instances[i].transform, self.instance_bounds[k]);
            }
        }
        self.cards.sync_materials(scene);
        // Captures finished since: the field's voxels take the captured means (a full recompose,
        // as after any scene change; once after loading, and after material edits).
        if self.cards.take_albedo_changed() {
            for level in &mut self.levels {
                level.dirty = true;
            }
        }

        // Clipmap placement: snap to whole bricks so a moving camera rebuilds rarely.
        for level in &mut self.levels {
            let snap = level.voxel * BRICK as f32;
            let origin = (camera_position / snap).floor() * snap - Vec3::splat(level.voxel * SDF_RES as f32 * 0.5);
            if origin != level.origin {
                level.origin = origin;
                level.dirty = true;
            }
        }
        // Rebuild at most one moved level per frame (all of them after a scene change).
        let all = self.levels.iter().all(|l| l.dirty);
        let mut budget = if all { SDF_LEVELS } else { 1 };
        for (i, level) in self.levels.iter_mut().enumerate() {
            if !level.dirty || level.compose_now || budget == 0 {
                continue;
            }
            budget -= 1;
            let band = level.voxel * 4.0;
            let lo = level.origin - band;
            let hi = level.origin + Vec3::splat(level.voxel * SDF_RES as f32) + band;
            let candidates: Vec<u32> = self
                .instance_bounds
                .iter()
                .zip(&self.in_global_field)
                .enumerate()
                .filter(|(_, ((a, b), &global))| global && a.cmple(hi).all() && b.cmpge(lo).all())
                .map(|(k, _)| k as u32)
                .collect();
            level.candidate_count = candidates.len() as u32;
            level.candidates = storage_buffer_init(device, queue, "sdf candidates", bytemuck::cast_slice(&candidates));
            let params = ComposeParams {
                origin: level.origin.extend(level.voxel),
                level: i as u32,
                candidate_count: level.candidate_count,
                level_count: SDF_LEVELS as u32,
                brick_count: 0,
            };
            queue.write_buffer(&level.params, 0, bytemuck::bytes_of(&params));
            level.compose_now = true;
        }
        let placements = self.placements();
        self.dynamic.stats.levels = self.levels.iter().filter(|l| l.compose_now).count() as u32;
        self.dynamic.schedule(queue, device, &placements, &self.instance_bounds, &self.in_global_field);

        self.light_grid.update(camera_position, frame);
        let mut probe_levels = [Vec4::ZERO; 2];
        for (i, spacing) in PROBE_SPACING.iter().enumerate() {
            let origin = (camera_position / *spacing).floor() * *spacing - Vec3::splat(spacing * (PROBE_RES / 2) as f32);
            self.probe_origins[i] = origin;
            probe_levels[i] = origin.extend(*spacing);
        }
        let mut sdf_levels = [Vec4::ZERO; SDF_LEVELS];
        for (i, level) in self.levels.iter().enumerate() {
            sdf_levels[i] = level.origin.extend(level.voxel);
        }
        let uniform = LumenUniform {
            sdf_levels,
            probe_levels,
            level_count: SDF_LEVELS as u32,
            frame,
            sdf_enabled: self.has_sdf as u32,
            max_trace_distance: 120.0,
            sky_intensity: 1.0,
            probe_update_fraction: self.probe_update_fraction(),
            voxel_update_phase: frame % 4,
            world_cache: if self.has_sdf { self.world_cache.mode.gpu() } else { 0 },
            near_field: self.near_field.update(device, queue, camera_position),
            cards: self.cards.update(device, queue, camera_position, frame),
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));

        // Sized for the largest internal resolution; each frame uses the corner it covers.
        let size = targets.capacity;
        if self.gather.as_ref().is_none_or(|g| g.size != size) {
            if let Some(old) = &self.gather {
                old.retire();
            }
            self.gather = Some(GatherTargets::new(device, size));
        }
        self.world_cache.prepare(device, camera_position, (targets.width, targets.height));
    }

    /// Relights the voxel surface cache every `frames` frames (1–8; `lumen_surface_interval`)
    /// and refreshes the radiance probes as much slower: both cost the same at any resolution.
    pub fn set_surface_interval(&mut self, frames: u32) {
        self.surface_interval = frames.clamp(1, 8);
    }

    /// Radiance probes refreshed per frame: one in `PROBE_UPDATE_FRACTION × surface_interval`
    /// (the probe count, 24³ × 2, divides by 8 × 1…8).
    fn probe_update_fraction(&self) -> u32 {
        PROBE_UPDATE_FRACTION * self.surface_interval
    }

    /// The clipmap levels as the incremental updates see them.
    fn placements(&self) -> Vec<LevelPlacement> {
        self.levels.iter().map(|l| LevelPlacement { origin: l.origin, voxel: l.voxel, rebuilding: l.dirty || l.compose_now }).collect()
    }

    /// Statistics of the incremental distance-field updates in the latest frame.
    pub fn dynamic_stats(&self) -> DynamicFieldStats {
        self.dynamic.stats
    }

    /// Per level and brick of the global field: whether it was recomposed this frame (for the
    /// invalidation debug view).
    pub fn brick_stamps(&self) -> &wgpu::Buffer {
        &self.dynamic.stamps
    }

    /// Uploads the distance-field instances; returns their scene instance indices and every scene
    /// instance's coverage (the solid fraction of its surface, 1 unless alpha-masked).
    ///
    /// Alpha-masked instances (foliage cards, chains, grilles) get mesh cards and take part in
    /// the near field, where traces treat them as partially transparent by their coverage
    /// (`lumen_mesh_sdf.wgsl`), but they stay out of the global distance field: its voxels
    /// (0.25 m and up) are far coarser than leaves and chain links, which would merge into
    /// solid blobs that block all light, and it has no channel left to hold a coverage.
    ///
    /// `surfaces` are the meshes' distance fields, then the displaced variants'; `surface_of`
    /// maps every scene instance to its surface.
    fn upload_scene(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, surfaces: &[Surface], surface_of: &[usize]) -> (Vec<usize>, Vec<f32>) {
        let mut descs: Vec<MeshSdfDesc> = Vec::new();
        let mut words: Vec<u32> = Vec::new();
        let mut mesh_desc: Vec<Option<u32>> = Vec::new();
        for surface in surfaces {
            let Some(sdf) = surface.sdf() else {
                mesh_desc.push(None);
                continue;
            };
            mesh_desc.push(Some(descs.len() as u32));
            let first = words.len() as u32;
            words.extend_from_slice(&sdf.words);
            descs.push(MeshSdfDesc {
                origin: sdf.origin.extend(sdf.voxel_size),
                res: [sdf.resolution[0], sdf.resolution[1], sdf.resolution[2], first],
                params: Vec4::new(sdf.band, sdf.kind as u32 as f32, sdf.height_range[0], sdf.height_range[1]),
            });
        }
        let mut instances: Vec<SdfInstance> = Vec::new();
        let mut indices = Vec::new();
        let mut warned = vec![false; surfaces.len()];
        let mut coverage = vec![1.0; scene.instances.len()];
        self.instance_aabbs.clear();
        self.instance_bounds.clear();
        self.in_global_field.clear();
        for (index, instance) in scene.instances.iter().enumerate().filter(|(_, i)| i.visible) {
            let surface = &surfaces[surface_of[index]];
            let Some(desc) = mesh_desc[surface_of[index]] else { continue };
            coverage[index] = self.mask_coverage.of(scene, instance);
            if coverage[index] < MIN_COVERAGE {
                continue;
            }
            let world = instance.transform;
            let linear = glam::Mat3::from_mat4(world);
            let max_scale = linear.x_axis.length().max(linear.y_axis.length()).max(linear.z_axis.length());
            if let Some(sdf) = surface.sdf().filter(|s| s.two_sided) {
                let shell = 0.5 * sdf.voxel_size * max_scale;
                if shell > MAX_SHELL && !std::mem::replace(&mut warned[surface_of[index]], true) {
                    log::warn!(
                        "lumen: open mesh {} has a {shell:.1} m thick distance-field shell, which blocks nearby light; \
                         bake terrain and ground sheets with SdfMode::Heightfield",
                        instance.mesh.0
                    );
                }
            }
            let mut record = SdfInstance {
                material: instance.material.0,
                mesh: desc,
                has_cards: surface.sdf().is_some_and(|s| s.kind == helios_geometry::SdfKind::Volume) as u32,
                ..Zeroable::zeroed()
            };
            let aabb = surface.aabb();
            self.instance_bounds.push(record.place(&aabb, world));
            self.instance_aabbs.push(aabb);
            instances.push(record);
            self.in_global_field.push(coverage[index] >= 1.0);
            indices.push(index);
        }
        self.has_sdf = !instances.is_empty();
        for old in [&self.sdf_instances, &self.sdf_descs, &self.sdf_words] {
            crate::gpu_memory::retire_buffer(old);
        }
        self.sdf_instances = storage_buffer_init(device, queue, "sdf instances", bytemuck::cast_slice(&instances));
        self.sdf_descs = storage_buffer_init(device, queue, "sdf descs", bytemuck::cast_slice(&descs));
        self.sdf_words = storage_buffer_init(device, queue, "sdf words", bytemuck::cast_slice(&words));
        log::info!(
            "lumen: {} SDF instances ({} alpha-masked), {} mesh SDFs ({:.1} MiB)",
            instances.len(),
            self.in_global_field.iter().filter(|&&g| !g).count(),
            descs.len(),
            words.len() as f64 * 4.0 / 1048576.0
        );
        self.dynamic.reset(instances, &indices, scene.instances.len());
        (indices, coverage)
    }

    /// Records the GPU displacement of displaced variants still to bake (nothing in scenes without
    /// displacement); call [`Lumen::after_submit`] once the frame is submitted.
    pub fn record_displacement(&mut self, rec: &mut Recorder, queue: &wgpu::Queue, scene: &Scene, inputs: &DisplaceInputs) {
        self.displaced.record(rec, queue, scene, inputs);
    }

    pub fn after_submit(&mut self) {
        self.displaced.after_submit();
        self.world_cache.after_submit();
    }

    /// Copies the world cache's statistics for read-back (end of the frame).
    pub fn record_stats(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.world_cache.record_stats(encoder);
    }

    /// Composes the clipmap levels of the global distance field scheduled by `update`. Recorded
    /// every frame, GI on or off: local-light soft shadows trace the field too, and with GI off
    /// they would otherwise read an empty (or stale) field and shadow everything.
    pub fn record_distance_field(&mut self, rec: &mut Recorder, textures: &TextureLibrary, materials: &wgpu::Buffer, samplers: &crate::passes::Samplers) {
        let device = rec.device;
        if self.has_sdf && self.levels.iter().any(|l| l.compose_now) {
            let mut pass = rec.compute_pass("lumen sdf");
            for level in self.levels.iter_mut().filter(|l| l.compose_now) {
                let cull = self.cull_bricks.group(
                    device,
                    &[
                        (0, level.params.as_entire_binding()),
                        (1, self.sdf_instances.as_entire_binding()),
                        (4, level.candidates.as_entire_binding()),
                        (5, self.bricks.as_entire_binding()),
                    ],
                );
                pass.set_pipeline(&self.cull_bricks.pipeline);
                pass.set_bind_group(0, &cull, &[]);
                pass.dispatch_workgroups(4096 / 64, 1, 1);
                let compose = self.compose.group(
                    device,
                    &[
                        (0, level.params.as_entire_binding()),
                        (1, self.sdf_instances.as_entire_binding()),
                        (2, self.sdf_descs.as_entire_binding()),
                        (3, self.sdf_words.as_entire_binding()),
                        (5, self.bricks.as_entire_binding()),
                        (6, materials.as_entire_binding()),
                        (7, TextureView(&textures.albedo)),
                        (8, Sampler(&samplers.material)),
                        (9, TextureView(&self.global_sdf_storage)),
                        (12, self.cards.pair_sums.as_entire_binding()),
                        (70, TextureView(&self.instance_ids)),
                    ],
                );
                pass.set_pipeline(&self.compose.pipeline);
                pass.set_bind_group(0, &compose, &[]);
                // One thread per pair of voxels along x.
                let groups = SDF_RES / 4;
                pass.dispatch_workgroups(groups / 2, groups, groups);
                level.dirty = false;
                level.compose_now = false;
            }
        }
        if self.has_sdf {
            let field = FieldBindings {
                sdf_instances: &self.sdf_instances,
                descs: &self.sdf_descs,
                words: &self.sdf_words,
                materials,
                albedo: &textures.albedo,
                sampler: &samplers.material,
                sdf: &self.global_sdf_storage,
                instance_ids: &self.instance_ids,
                pair_sums: &self.cards.pair_sums,
            };
            self.dynamic.record(rec, &field);
        }
    }

    /// Records surface cache (cards and voxels) and radiance cache updates, the screen-probe
    /// gather (into `targets.gi_raw`; skipped when Lumen Lite gathers instead) and reflections.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        rec: &mut Recorder,
        frame: &wgpu::Buffer,
        targets: &FrameTargets,
        sky: &crate::passes::sky::Sky,
        samplers: &crate::passes::Samplers,
        previous_radiance: &wgpu::TextureView,
        current: usize,
        lights: &crate::lights::LightCulling,
        exposure: &wgpu::Buffer,
        surfaces: &CardInputs,
        screen_probes: bool,
    ) {
        let device = rec.device;
        let Some(gather) = &self.gather else { return };
        let options = surfaces.options;
        let gather_options = GatherOptions { skip_unlit_hits: options.skip_unlit_hits as u32, reflection_fog: options.reflection_fog as u32, _pad: [0; 2] };
        surfaces.queue.write_buffer(&self.gather_options, 0, bytemuck::bytes_of(&gather_options));

        // ---- Local-light candidates of the surface caches ----------------------------------------
        if self.has_sdf {
            self.light_grid.record(rec, surfaces.queue, frame, lights, options.light_sampling);
        }

        // ---- Mesh cards: capture pending materials, light this frame's pages ---------------------
        if self.has_sdf {
            let inputs = CaptureInputs {
                frame,
                materials: surfaces.materials,
                textures: surfaces.textures,
                sampler: &samplers.material,
                extended: surfaces.extended,
            };
            self.cards.record_capture(rec, surfaces.queue, &inputs);
        }
        // The world cache replacing the cards: their lighting is not updated (UE 5.9).
        if self.has_sdf && self.cards.update_count() > 0 && self.world_cache.mode != WorldCacheMode::Replace {
            let group = self.light_cards.group(
                device,
                &[
                    (0, frame.as_entire_binding()),
                    (1, self.uniform.as_entire_binding()),
                    (2, TextureView(&self.global_sdf)),
                    (4, Sampler(&samplers.linear)),
                    (5, self.radiance_probes.as_entire_binding()),
                    (6, sky.sh.as_entire_binding()),
                    (7, TextureView(&sky.transmittance)),
                    (8, lights.lights.as_entire_binding()),
                    (9, self.cards.data.as_entire_binding()),
                    (11, TextureView(&self.cards.capture)),
                    (13, TextureView(&self.cards.lighting)),
                    (14, self.cards.updates.as_entire_binding()),
                    (15, surfaces.vsm.uniform.as_entire_binding()),
                    (16, surfaces.vsm.cache.entries.as_entire_binding()),
                    (17, TextureView(&surfaces.vsm.atlas)),
                    (18, Sampler(&samplers.shadow)),
                    (19, self.near_field.data.as_entire_binding()),
                    (20, TextureView(&self.near_field.atlas)),
                    (21, surfaces.materials.as_entire_binding()),
                    (22, TextureView(&surfaces.textures.albedo)),
                    (23, Sampler(&samplers.material)),
                    (28, TextureView(&self.cards.albedo)),
                    (90, surfaces.glass.uniform.as_entire_binding()),
                    (91, TextureView(surfaces.glass.view())),
                    (163, self.light_grid.uniform.as_entire_binding()),
                    (164, self.light_grid.words.as_entire_binding()),
                ],
            );
            let mut pass = rec.compute_pass("lumen cards");
            pass.set_pipeline(&self.light_cards.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.cards.update_count(), 1, 1);
        }

        // ---- Surface cache (voxel lighting, ping-pong) and radiance cache ------------------------
        // Relit every `surface_interval` frames; between relightings the radiance cache and the
        // gather read the latest one.
        let relight = self.frame.is_multiple_of(self.surface_interval);
        let read = self.voxels_latest;
        let write = if relight { 1 - read } else { read };
        {
            // Only when relighting: otherwise `read` and `write` are the same texture.
            let voxels = relight.then(|| {
                self.update_voxels.group(
                    device,
                    &[
                        (0, frame.as_entire_binding()),
                        (1, self.uniform.as_entire_binding()),
                        (2, TextureView(&self.global_sdf)),
                        (3, TextureView(&self.voxel_lighting[read])),
                        (4, Sampler(&samplers.linear)),
                        (5, TextureView(&self.voxel_lighting[write])),
                        (6, self.radiance_probes.as_entire_binding()),
                        (7, sky.sh.as_entire_binding()),
                        (8, TextureView(&sky.transmittance)),
                        (10, Sampler(&samplers.linear)),
                        (11, lights.lights.as_entire_binding()),
                        (163, self.light_grid.uniform.as_entire_binding()),
                        (164, self.light_grid.words.as_entire_binding()),
                    ],
                )
            });
            let probes = self.update_probes.group(
                device,
                &[
                    (0, frame.as_entire_binding()),
                    (1, self.uniform.as_entire_binding()),
                    (2, TextureView(&self.global_sdf)),
                    (3, TextureView(&self.voxel_lighting[write])),
                    (4, Sampler(&samplers.linear)),
                    (6, self.radiance_probes.as_entire_binding()),
                    (9, TextureView(&sky.sky_view)),
                    (10, Sampler(&samplers.linear)),
                    (70, self.cards.data.as_entire_binding()),
                    (71, self.cards.pages.as_entire_binding()),
                    (72, TextureView(&self.cards.capture)),
                    (73, TextureView(&self.cards.lighting)),
                    (74, TextureView(&self.instance_ids)),
                    (150, TextureView(self.world_cache.cells())),
                ],
            );
            let mut pass = rec.compute_pass("lumen surface + radiance cache");
            if self.has_sdf {
                if let Some(voxels) = &voxels {
                    pass.set_pipeline(&self.update_voxels.pipeline);
                    pass.set_bind_group(0, voxels, &[]);
                    pass.dispatch_workgroups(LIGHT_RES / 4, LIGHT_RES / 4, LIGHT_RES * SDF_LEVELS as u32 / 4);
                }
                pass.set_pipeline(&self.update_probes.pipeline);
                pass.set_bind_group(0, &probes, &[]);
                // One workgroup (a thread per ray) per refreshed probe.
                pass.dispatch_workgroups(PROBE_RES * PROBE_RES * PROBE_RES * 2 / self.probe_update_fraction(), 1, 1);
            }
        }
        self.voxels_latest = write;
        let lighting = &self.voxel_lighting[write];

        // ---- World cache: paths from the screen into its cells ------------------------------------
        if self.has_sdf {
            let inputs = WorldCacheInputs {
                frame,
                lumen: &self.uniform,
                global_sdf: &self.global_sdf,
                voxel_lighting: lighting,
                linear: &samplers.linear,
                shadow_sampler: &samplers.shadow,
                radiance_probes: &self.radiance_probes,
                transmittance: &sky.transmittance,
                sky_view: &sky.sky_view,
                lights: &lights.lights,
                exposure,
                vsm: surfaces.vsm,
                glass: surfaces.glass,
                near_field: (&self.near_field.data, &self.near_field.atlas),
                targets,
                card_data: &self.cards.data,
                card_pages: &self.cards.pages,
                card_capture: &self.cards.capture,
                card_lighting: &self.cards.lighting,
                card_albedo: &self.cards.albedo,
                instance_ids: &self.instance_ids,
            };
            self.world_cache.record(rec, &inputs);
            if self.world_cache.reference_active() && screen_probes {
                self.world_cache.record_reference(rec, &inputs);
            }
        }
        // The reference path tracer (comparisons) writes the GI in place of the screen probes.
        let screen_probes = screen_probes && !(self.has_sdf && self.world_cache.reference_active());
        let cells = self.world_cache.cells();

        // ---- Screen probes ----------------------------------------------------------------------
        let grid = (targets.width.div_ceil(PROBE_TILE), targets.height.div_ceil(PROBE_TILE));
        let common = |stage: &Stage, extra: Vec<(u32, wgpu::BindingResource)>| {
            let mut entries: Vec<(u32, wgpu::BindingResource)> = vec![(0, frame.as_entire_binding())];
            entries.extend(extra);
            stage.group(device, &entries)
        };
        let placing = |stage: &Stage| {
            common(
                stage,
                vec![
                    (2, TextureView(&targets.depth)),
                    (3, TextureView(&targets.gbuffer_normal)),
                    (10, gather.probes.as_entire_binding()),
                    (85, gather.adaptive.as_entire_binding()),
                ],
            )
        };
        let place = placing(&self.place_probes);
        let request_adaptive = placing(&self.request_adaptive);
        let place_adaptive = placing(&self.place_adaptive);
        let finish_adaptive =
            common(&self.finish_adaptive, vec![(10, gather.probes.as_entire_binding()), (85, gather.adaptive.as_entire_binding())]);
        let tracing = |stage: &Stage, out: (u32, &wgpu::TextureView)| {
            let mut entries = vec![
                (1, self.uniform.as_entire_binding()),
                (2, TextureView(&targets.depth)),
                (4, TextureView(&targets.gbuffer_motion)),
                (5, TextureView(previous_radiance)),
                (6, TextureView(&sky.sky_view)),
                (7, TextureView(&self.global_sdf)),
                (8, TextureView(lighting)),
                (9, Sampler(&samplers.linear)),
                (11, self.radiance_probes.as_entire_binding()),
                (18, exposure.as_entire_binding()),
                (19, self.near_field.data.as_entire_binding()),
                (20, TextureView(&self.near_field.atlas)),
                (70, self.cards.data.as_entire_binding()),
                (71, self.cards.pages.as_entire_binding()),
                (72, TextureView(&self.cards.capture)),
                (73, TextureView(&self.cards.lighting)),
                (74, TextureView(&self.instance_ids)),
                (150, TextureView(cells)),
                (REFLECT_BINDINGS[0], surfaces.fog.as_entire_binding()),
                (REFLECT_BINDINGS[1], TextureView(&sky.transmittance)),
                (REFLECT_BINDINGS[2], self.gather_options.as_entire_binding()),
                (out.0, TextureView(out.1)),
            ];
            if out.0 == 12 {
                entries.push((10, gather.probes.as_entire_binding()));
            } else {
                entries.push((3, TextureView(&targets.gbuffer_normal)));
            }
            common(stage, entries)
        };
        let trace = tracing(&self.trace_probes, (12, &gather.atlas));
        let sh = common(
            &self.probe_sh,
            vec![(10, gather.probes.as_entire_binding()), (13, TextureView(&gather.atlas)), (86, gather.adaptive.as_entire_binding())],
        );
        let integrate = common(
            &self.integrate,
            vec![
                (2, TextureView(&targets.depth)),
                (3, TextureView(&targets.gbuffer_normal)),
                (1, self.uniform.as_entire_binding()),
                (10, gather.probes.as_entire_binding()),
                (11, self.radiance_probes.as_entire_binding()),
                (14, TextureView(&targets.gi_raw)),
                (86, gather.adaptive.as_entire_binding()),
            ],
        );
        let reflect = tracing(&self.reflect_trace, (15, &gather.reflection_raw));
        let half = (targets.width.div_ceil(2), targets.height.div_ceil(2));
        if screen_probes {
            let mut pass = rec.compute_pass("lumen final gather");
            pass.set_pipeline(&self.place_probes.pipeline);
            pass.set_bind_group(0, &place, &[]);
            pass.dispatch_workgroups(workgroups(grid.0, 8), workgroups(grid.1, 8), 1);
            // Adaptive probes where the uniform ones miss a surface; the probe passes then run over
            // the uniform grid plus the tile rows the adaptive probes filled (indirect).
            // The cells of the internal resolution (the buffers are sized for the capacity).
            let cells = (targets.width.div_ceil(ADAPTIVE_CELL), targets.height.div_ceil(ADAPTIVE_CELL));
            for (stage, group) in [(&self.request_adaptive, &request_adaptive), (&self.place_adaptive, &place_adaptive)] {
                pass.set_pipeline(&stage.pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(workgroups(cells.0, 8), workgroups(cells.1, 8), 1);
            }
            pass.set_pipeline(&self.finish_adaptive.pipeline);
            pass.set_bind_group(0, &finish_adaptive, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.trace_probes.pipeline);
            pass.set_bind_group(0, &trace, &[]);
            pass.dispatch_workgroups_indirect(&gather.adaptive, 0);
            pass.set_pipeline(&self.probe_sh.pipeline);
            pass.set_bind_group(0, &sh, &[]);
            pass.dispatch_workgroups_indirect(&gather.adaptive, 0);
            pass.set_pipeline(&self.integrate.pipeline);
            pass.set_bind_group(0, &integrate, &[]);
            pass.dispatch_workgroups(workgroups(half.0, 8), workgroups(half.1, 8), 1);
        }
        if options.traced_reflections {
            let temporal = self.reflection_temporal_group(device, frame, targets, samplers, current, exposure);
            let mut pass = rec.compute_pass("lumen reflections");
            pass.set_pipeline(&self.reflect_trace.pipeline);
            pass.set_bind_group(0, &reflect, &[]);
            pass.dispatch_workgroups(workgroups(half.0, 8), workgroups(half.1, 8), 1);
            pass.set_pipeline(&self.reflect_temporal.pipeline);
            pass.set_bind_group(0, &temporal, &[]);
            pass.dispatch_workgroups(workgroups(half.0, 8), workgroups(half.1, 8), 1);
        }
    }

    fn reflection_temporal_group(&self, device: &wgpu::Device, frame: &wgpu::Buffer, targets: &FrameTargets, samplers: &crate::passes::Samplers, current: usize, exposure: &wgpu::Buffer) -> wgpu::BindGroup {
        let gather = self.gather.as_ref().expect("gather targets");
        self.reflect_temporal.group(
            device,
            &[
                (0, frame.as_entire_binding()),
                (4, TextureView(&targets.gbuffer_motion)),
                (9, Sampler(&samplers.linear)),
                (15, TextureView(&gather.reflection[current])),
                (16, TextureView(&gather.reflection_raw)),
                (17, TextureView(&gather.reflection[1 - current])),
                (18, exposure.as_entire_binding()),
            ],
        )
    }

    /// This frame's raw reflections (half resolution), which Lumen Lite's screen-space
    /// reflections write when Lumen does not trace them.
    pub(crate) fn reflection_raw(&self) -> Option<&wgpu::TextureView> {
        self.gather.as_ref().map(|g| &g.reflection_raw)
    }

    /// Spatial filter and temporal accumulation of the raw reflections into
    /// [`Lumen::reflection`]`(current)` (after Lumen Lite wrote them).
    pub(crate) fn record_reflection_temporal(&self, rec: &mut Recorder, frame: &wgpu::Buffer, targets: &FrameTargets, samplers: &crate::passes::Samplers, current: usize, exposure: &wgpu::Buffer) {
        if self.gather.is_none() {
            return;
        }
        let group = self.reflection_temporal_group(rec.device, frame, targets, samplers, current, exposure);
        let half = (targets.width.div_ceil(2), targets.height.div_ceil(2));
        let mut pass = rec.compute_pass("lumen reflections");
        pass.set_pipeline(&self.reflect_temporal.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(half.0, 8), workgroups(half.1, 8), 1);
    }

    /// Draws the card lighting each pixel's surface reads into `targets.hdr`
    /// ([`crate::DebugView::SurfaceCards`]; purple-tinted voxel lighting where no card matches).
    pub fn record_debug(&self, rec: &mut Recorder, frame: &wgpu::Buffer, targets: &FrameTargets, samplers: &crate::passes::Samplers) {
        let device = rec.device;
        let group = self.debug_cards.group(
            device,
            &[
                (0, frame.as_entire_binding()),
                (1, self.uniform.as_entire_binding()),
                (2, TextureView(&self.global_sdf)),
                (3, TextureView(self.debug_views().1)),
                (4, Sampler(&samplers.linear)),
                (9, self.cards.data.as_entire_binding()),
                (10, self.cards.pages.as_entire_binding()),
                (11, TextureView(&self.cards.capture)),
                (12, TextureView(&self.cards.lighting)),
                (24, TextureView(&self.instance_ids)),
                (25, TextureView(&targets.depth)),
                (26, TextureView(&targets.gbuffer_normal)),
                (27, TextureView(&targets.hdr)),
            ],
        );
        let mut pass = rec.compute_pass("lumen cards debug");
        pass.set_pipeline(&self.debug_cards.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(targets.width, 8), workgroups(targets.height, 8), 1);
    }

    /// Views for debug visualisation: (global SDF, current voxel lighting).
    pub fn debug_views(&self) -> (&wgpu::TextureView, &wgpu::TextureView) {
        (&self.global_sdf, &self.voxel_lighting[self.voxels_latest])
    }

    pub fn radiance_probes(&self) -> &wgpu::Buffer {
        &self.radiance_probes
    }

    pub fn uniform(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    /// Near-field bindings: (mesh SDF records + culling grid, mesh SDF atlas).
    pub fn near_field(&self) -> (&wgpu::Buffer, &wgpu::TextureView) {
        (&self.near_field.data, &self.near_field.atlas)
    }

    /// What `trace_world_radiance` (lumen_trace.wgsl) reads, for Lumen Lite's probe traces.
    pub(crate) fn trace_resources(&self) -> TraceResources<'_> {
        TraceResources {
            uniform: &self.uniform,
            global_sdf: &self.global_sdf,
            voxel_lighting: self.debug_views().1,
            radiance_probes: &self.radiance_probes,
            near_field: (&self.near_field.data, &self.near_field.atlas),
            card_data: &self.cards.data,
            card_pages: &self.cards.pages,
            card_capture: &self.cards.capture,
            card_lighting: &self.cards.lighting,
            instance_ids: &self.instance_ids,
            world_cache: self.world_cache.cells(),
        }
    }

    /// Draws the world cache's radiance at every pixel ([`crate::DebugView::WorldCache`]).
    #[allow(clippy::too_many_arguments)]
    pub fn record_world_cache_debug(
        &self,
        rec: &mut Recorder,
        frame: &wgpu::Buffer,
        targets: &FrameTargets,
        sky: &crate::passes::sky::Sky,
        samplers: &crate::passes::Samplers,
        lights: &crate::lights::LightCulling,
        exposure: &wgpu::Buffer,
        surfaces: &CardInputs,
    ) {
        let lighting = &self.voxel_lighting[self.voxels_latest];
        let inputs = WorldCacheInputs {
            frame,
            lumen: &self.uniform,
            global_sdf: &self.global_sdf,
            voxel_lighting: lighting,
            linear: &samplers.linear,
            shadow_sampler: &samplers.shadow,
            radiance_probes: &self.radiance_probes,
            transmittance: &sky.transmittance,
            sky_view: &sky.sky_view,
            lights: &lights.lights,
            exposure,
            vsm: surfaces.vsm,
            glass: surfaces.glass,
            near_field: (&self.near_field.data, &self.near_field.atlas),
            targets,
            card_data: &self.cards.data,
            card_pages: &self.cards.pages,
            card_capture: &self.cards.capture,
            card_lighting: &self.cards.lighting,
            card_albedo: &self.cards.albedo,
            instance_ids: &self.instance_ids,
        };
        self.world_cache.record_debug(rec, &inputs);
    }
}

/// Lumen's tracing resources: the uniform, the global distance field, this frame's voxel
/// lighting, the radiance cache, the near field and the mesh cards.
pub(crate) struct TraceResources<'a> {
    pub uniform: &'a wgpu::Buffer,
    pub global_sdf: &'a wgpu::TextureView,
    pub voxel_lighting: &'a wgpu::TextureView,
    pub radiance_probes: &'a wgpu::Buffer,
    pub near_field: (&'a wgpu::Buffer, &'a wgpu::TextureView),
    pub card_data: &'a wgpu::Buffer,
    pub card_pages: &'a wgpu::Buffer,
    pub card_capture: &'a wgpu::TextureView,
    pub card_lighting: &'a wgpu::TextureView,
    pub instance_ids: &'a wgpu::TextureView,
    /// The world cache's cells (binding 150).
    pub world_cache: &'a wgpu::TextureView,
}

/// Coverage of alpha-masked (mesh, material) pairs: the fraction of the surface that passes the
/// alpha test ([`crate::alpha_mask::surface_coverage`]), computed once per geometry, material
/// and texture version from the mesh's finest in-memory level.
#[derive(Default)]
struct MaskCoverage {
    versions: (u64, u64, u64),
    cache: HashMap<(u32, u32), f32>,
}

impl MaskCoverage {
    /// The solid fraction of `instance`'s surface: 1 unless its material is alpha-masked.
    fn of(&mut self, scene: &Scene, instance: &Instance) -> f32 {
        let versions = (scene.geometry_version, scene.material_version, scene.texture_version);
        if versions != self.versions {
            self.versions = versions;
            self.cache.clear();
        }
        let material = &scene.materials[instance.material.0 as usize];
        if material.alpha_cutoff.is_none() {
            return 1.0;
        }
        *self.cache.entry((instance.mesh.0, instance.material.0)).or_insert_with(|| {
            let (vertices, indices) = crate::translucency::decode_finest(&scene.meshes[instance.mesh.0 as usize]);
            let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
            // The base colour's texture coordinates (UV set and transform) select the mask texels.
            let uvs: Vec<[f32; 2]> = vertices.iter().map(|v| material.uv.apply(v.uv.into(), v.uv1.into()).into()).collect();
            let coverage = crate::alpha_mask::surface_coverage(scene, material, &positions, &uvs, &indices).unwrap_or(1.0);
            log::debug!("lumen: alpha-masked mesh {} with material {}: {:.0} % solid", instance.mesh.0, instance.material.0, coverage * 100.0);
            coverage
        })
    }
}

/// Tile rows of the probe atlas (and probe buffer) for adaptive probes below a uniform grid
/// `grid_rows` tiles high: at most half as many adaptive probes as uniform ones, as in UE5
/// (`adaptive_capacity` in lumen_gather.wgsl).
fn adaptive_rows(grid_rows: u32) -> u32 {
    grid_rows.div_ceil(2)
}

impl GatherTargets {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let grid = (size.0.div_ceil(PROBE_TILE), size.1.div_ceil(PROBE_TILE));
        let rows = grid.1 + adaptive_rows(grid.1);
        let cells = size.0.div_ceil(ADAPTIVE_CELL) as u64 * size.1.div_ceil(ADAPTIVE_CELL) as u64;
        let make = |label: &str, w: u32, h: u32| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: F::Rgba16Float,
                    usage: U::TEXTURE_BINDING | U::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let half = (size.0.div_ceil(2), size.1.div_ceil(2));
        Self {
            size,
            probes: storage_buffer(device, "screen probes", (grid.0 * rows) as u64 * PROBE_STRIDE, wgpu::BufferUsages::empty()),
            adaptive: storage_buffer(device, "adaptive screen probes", (ADAPTIVE_HEADER_WORDS + cells) * 4, wgpu::BufferUsages::INDIRECT),
            atlas: make("screen probe atlas", grid.0 * 8, rows * 8),
            reflection_raw: make("reflection raw", half.0, half.1),
            reflection: [make("reflection 0", half.0, half.1), make("reflection 1", half.0, half.1)],
        }
    }

    /// Hands the targets to `gpu_memory::destroy_retired` (they are being replaced).
    fn retire(&self) {
        crate::gpu_memory::retire_buffer(&self.probes);
        crate::gpu_memory::retire_buffer(&self.adaptive);
        for view in [&self.atlas, &self.reflection_raw].into_iter().chain(&self.reflection) {
            crate::gpu_memory::retire_view(view);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_probes_fit_their_buffers_and_dispatches() {
        for (w, h) in [(1u32, 1u32), (17, 9), (1600, 900), (2560, 1440), (3840, 2160), (8192, 8192)] {
            let grid = (w.div_ceil(PROBE_TILE), h.div_ceil(PROBE_TILE));
            let rows = adaptive_rows(grid.1);
            // `adaptive_capacity` in lumen_gather.wgsl: whole rows, at most half the uniform
            // probes rounded up to a row.
            let capacity = grid.1.div_ceil(2) * grid.0;
            assert_eq!(rows * grid.0, capacity);
            assert!(capacity <= grid.0 * grid.1 / 2 + grid.0);
            // The indirect dispatch (one workgroup per probe) stays within WebGPU's limit, and a
            // cell entry (`first probe << 2 | count`) holds every adaptive probe index.
            assert!(grid.1 + rows <= 65535);
            assert!(u64::from(capacity) << 2 < 1 << 32);
        }
    }
}
