//! Virtual shadow maps for the sun (see `shaders/vsm_common.wgsl`).
//!
//! A clipmap of 16k² virtual shadow maps is paged into a 4k² physical depth atlas. Each frame:
//!
//! 1. **mark** — every visible pixel requests the page it will sample,
//! 2. **cache** — the generic page cache keeps pages rendered in earlier frames (they stay valid
//!    while the sun and the scene are static) and allocates new ones,
//! 3. **render** — only dirty pages are rendered: each level's clusters are culled
//!    (restricted to that level's dirty rectangle) with an orthographic LOD metric, expanded to
//!    (cluster, page) pairs and rasterised into their physical tiles. As in the main view,
//!    clusters whose triangles are tiny in that level's texels take a compute rasteriser: it
//!    `atomicMax`es depth into a u32 mirror of the atlas, the page clear copies that into the
//!    depth atlas, and the hardware clusters are then drawn on top.
//!
//! With a static sun almost nothing is re-rendered from frame to frame — the main win of
//! virtual shadow maps besides their resolution. Geometry streaming, wind-animated vegetation
//! and moved instances invalidate only the cached pages under the instances concerned
//! ([`PageInvalidation`], `vsm_motion.rs`).
//!
//! **Static and dynamic casters** (UE5's "static separate" page caching). Casters the wind moves
//! this frame and `Mobility::Movable` instances are *dynamic*, all others *static*. A page that must be refreshed only because
//! dynamic casters moved keeps the depth of its static casters in a *static layer* — the page's
//! tile of the software raster's u32 mirror, which is otherwise only scratch space between the
//! software raster and the page clear, so the static layers cost no memory of their own. Such a
//! page is cleared to its static layer (merged with its software-rasterised dynamic casters, which
//! go to a scratch tile after the atlas' tiles) and only its dynamic casters are culled, expanded
//! and drawn: the cost of the wind scales with the moving geometry. A page gets its static layer
//! the first time it needs such a refresh: it is rendered in full once more, its static casters
//! first, whose depth is then copied into the mirror (`shaders/vsm_static.wgsl`), then its dynamic
//! ones. Anything that changes static casters (the sun or a light moving, scene changes,
//! streaming or moving static instances, the wind being switched on or off) frees the page, so it is
//! rendered in full again; a caster fading into or out of the wind with the camera's distance
//! only makes the static layers under it stale. The sampled atlas always holds the merged result.
//!
//! Local lights' shadow maps (`vsm_local.rs`) are pages of the same cache and atlas: their
//! requests join the marking, their dirty pages the same culling, expansion and raster passes,
//! with one more culling view for all lights.

use bytemuck::{Pod, Zeroable};
use glam::camera::rh::{proj::directx, view::look_to_mat4};
use glam::{Mat4, Vec3, Vec4};
use wgpu::BindingResource::TextureView;
use wgpu::{ShaderStages, TextureFormat};

use crate::camera::frustum_planes;
use crate::gpu_scene::GpuScene;
use crate::gpu_types::CullViewUniform;
use crate::lights::LightCulling;
use crate::page_cache::{dirty_list_start, CacheClient, PageCache};
use crate::passes::culling::{alpha_mask_entries, alpha_mask_resources, CullCapacity, CullMode, CullView, Culling};
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::tessellation::{tessellation_entry, tessellation_resource, ShadowInputs};
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, pipeline_layout, storage_buffer, uniform_buffer, workgroups};
use crate::vsm_local::{self, LocalCull, LocalFrame, LocalShadows};
use crate::skin::{skin_entry, skin_resource};
use crate::vsm_motion::{self, MoveFootprints, ShadowMove};
use crate::wind::{wind_entry, wind_resource};
use crate::Camera;

pub const MAX_LEVELS: usize = 16;
const PAGE_TEXELS: u32 = 128;
const LEVEL_PAGES: u32 = 128;
/// The physical atlas is TILES_PER_ROW² pages, shared by the sun and local lights: 1024 pages
/// of 128² texels, 64 MiB of depth (plus as much for the software raster's u32 mirror, which
/// also holds the pages' static layers).
const TILES_PER_ROW: u32 = 32;
pub const PHYSICAL_PAGES: u32 = TILES_PER_ROW * TILES_PER_ROW;
/// Tiles of the mirror after the atlas' tiles: scratch space for the software-rasterised dynamic
/// casters of pages refreshed over their static layer (16 MiB; whole rows of tiles). Pages beyond
/// it in a frame draw those casters with the hardware rasteriser.
const SCRATCH_TILES: u32 = 256;
/// Scratch tiles fill whole rows of tiles; a draw stores their index (physical page count +
/// scratch) in its 16-bit physical page field; the mirror stays within WebGPU's largest storage
/// binding (128 MiB).
const MIRROR_TEXELS: u64 = ((TILES_PER_ROW + SCRATCH_TILES / TILES_PER_ROW) * PAGE_TEXELS) as u64 * (TILES_PER_ROW * PAGE_TEXELS) as u64;
const _: () = assert!(SCRATCH_TILES.is_multiple_of(TILES_PER_ROW) && PHYSICAL_PAGES + SCRATCH_TILES <= 1 << 16);
const _: () = assert!(MIRROR_TEXELS * 4 <= 128 << 20);
/// Info buffer layout (vsm.wgsl, word offsets in vsm_common.wgsl): per sun level 8 words, then
/// the page clear arguments and counters, two bit masks over the clipmap page table entries (all
/// dirty pages, pages rendered in full), per local light slot LOCAL_SLOT_WORDS words
/// (LOCAL_INFO_SLOTS), per local face mip 4 words (LOCAL_INFO_RECTS) and three words per
/// physical page.
const INFO_ARGS: u32 = 128;
/// Word offset of the dirty page bit masks and the size of each: a bit per clipmap page table
/// entry.
const INFO_DIRTY_MASK: u64 = 136;
const DIRTY_MASK_WORDS: u64 = (MAX_LEVELS as u64 * LEVEL_PAGES as u64 * LEVEL_PAGES as u64) / 32;
/// Word offsets of the local lights' data: also the local culling view's `vsm.x` (cull.wgsl).
const INFO_LOCAL_SLOTS: u32 = (INFO_DIRTY_MASK + 2 * DIRTY_MASK_WORDS) as u32;
const LOCAL_SLOT_WORDS: u32 = 32;
const INFO_LOCAL_RECTS: u32 = INFO_LOCAL_SLOTS + vsm_local::LOCAL_SLOTS as u32 * LOCAL_SLOT_WORDS;
/// Per physical page: its kind this frame, the entry its static layer belongs to (persistent,
/// initially none) and its pending refresh flag (persistent).
const INFO_PAGE_STATE: u32 = INFO_LOCAL_RECTS + vsm_local::FACE_MIPS * 4;
const INFO_PAGE_STATIC: u32 = INFO_PAGE_STATE + PHYSICAL_PAGES;
const INFO_WORDS: u64 = (INFO_PAGE_STATIC + 2 * PHYSICAL_PAGES) as u64;
const _: () = assert!(INFO_LOCAL_SLOTS == 16520 && INFO_LOCAL_RECTS == 17544 && INFO_PAGE_STATE == 22152 && INFO_WORDS == 25224);
const _: () = assert!(vsm_local::LOCAL_LEVEL == MAX_LEVELS as u32);
/// Items of the info buffer reset each frame (info_begin in vsm.wgsl).
const INFO_ITEMS: u32 = MAX_LEVELS as u32 + vsm_local::LOCAL_SLOTS as u32 + vsm_local::FACE_MIPS;
/// World size of a level-0 texel (level i texels are 2^i times larger).
const FIRST_TEXEL: f32 = 1.0 / 1024.0;
pub const DRAW_CAPACITY: u32 = 1 << 20;
/// Byte offset of the software raster dispatch arguments in the draws buffer.
const SW_DISPATCH_OFFSET: u64 = 8 * 4;
/// Words of a draw list's header (vsm_expand.wgsl).
const DRAW_HEADER_WORDS: u64 = 16;
/// The second draw list (dynamic casters of split pages, drawn after their static layer was
/// captured) follows the first at a storage-binding-aligned byte offset.
const SPLIT_DRAWS_OFFSET: u64 = ((DRAW_HEADER_WORDS + DRAW_CAPACITY as u64 * 4) * 4).next_multiple_of(256);
const SPLIT_DRAW_CAPACITY: u32 = 1 << 18;
const SPLIT_DRAWS_SIZE: u64 = (DRAW_HEADER_WORDS + SPLIT_DRAW_CAPACITY as u64 * 4) * 4;
const RETAIN_FRAMES: u32 = 60;
/// Shadow draws alpha-test per (cluster, page) draw: no masked list.
const LEVEL_CAPACITY: CullCapacity = CullCapacity { batches: 1 << 14, pages: 1 << 16, clusters: 1 << 17, masked: 0 };
/// The local lights' view covers the dirty pages of all of them (every light's after a scene
/// change).
const LOCAL_CAPACITY: CullCapacity = CullCapacity { batches: 1 << 15, pages: 1 << 17, clusters: 1 << 18, masked: 0 };
/// Shadow pages send clusters with up to twice the main view's triangle size (in texels) to the
/// software rasteriser: it writes depth only (no payload pass) and clips exactly to the page,
/// while each hardware page draw re-transforms the whole cluster (measured: forest VSM -15%,
/// valley neutral, versus the main view's threshold).
const SW_THRESHOLD_SCALE: f32 = 2.0;
/// Slope-scaled depth bias against shadow acne, in multiples of the depth slope per texel
/// (pushed away from the light). Shared by the hardware pipeline and the software rasteriser.
pub(crate) const SLOPE_BIAS: f32 = 1.5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VsmUniform {
    light_view: Mat4,
    level_view_proj: [Mat4; MAX_LEVELS],
    level_origin: [[i32; 4]; MAX_LEVELS],
    level_info: [Vec4; MAX_LEVELS],
    level_count: u32,
    page_texels: u32,
    level_pages: u32,
    tiles_per_row: u32,
    atlas_size: f32,
    first_texel: f32,
    lod_bias: f32,
    frame: u32,
    pixel_angle: f32,
    physical_count: u32,
    enabled: u32,
    draw_capacity: u32,
    eye: Vec4,
}

const _: () = assert!(std::mem::size_of::<VsmUniform>() == 64 + 64 * 16 + 16 * 16 * 2 + 48 + 16);

/// Per-frame inputs to [`VirtualShadowMap::update`].
pub struct VsmFrame<'a> {
    pub camera: &'a Camera,
    pub viewport_height: f32,
    pub sun_direction: Vec3,
    pub scene_min: Vec3,
    pub scene_max: Vec3,
    pub instance_count: u32,
    pub frame: u32,
    pub levels: u32,
    pub lod_threshold: f32,
    pub resolution_bias: f32,
    /// Resolution bias of local lights' shadow maps (in mips).
    pub local_resolution_bias: f32,
    /// Software-rasterise clusters whose triangles are smaller than this in texels (0 = off).
    pub sw_raster_threshold: f32,
    pub enabled: bool,
    /// Something other than the sun direction invalidated cached pages (geometry, instances).
    pub scene_changed: bool,
    /// Streaming loaded or evicted geometry pages: only cached pages under the instances of
    /// the affected meshes are re-rendered.
    pub geometry_streamed: bool,
    /// The wind or skeletal animation moves shadow casters: cached pages under them re-render
    /// these (dynamic) casters as they move.
    pub wind: bool,
    /// Shadow casters moved this frame (only the pages under them are re-rendered).
    pub moves: &'a [ShadowMove],
    /// Some instance moved or changed mobility (casters may change layer).
    pub instances_changed: bool,
    /// View depth up to which volumetric fog samples the shadow map (0 = no volumetrics):
    /// coarse pages are requested throughout that part of the frustum.
    pub volume_distance: f32,
    /// `CullViewUniform::flags` of the shadow views (voxel LODs as in the main view).
    pub cull_flags: u32,
}

pub struct VirtualShadowMap {
    pub cache: PageCache,
    pub uniform: wgpu::Buffer,
    pub atlas: wgpu::TextureView,
    info: wgpu::Buffer,
    /// Hardware page draws from the front, software raster items from the back, then the
    /// hardware draws of dynamic casters into split pages (vsm_expand.wgsl).
    draws: wgpu::Buffer,
    /// Software-rasterised depth bits per atlas texel, then SCRATCH_TILES scratch tiles. Holds
    /// the static layer of pages that have one (`INFO_PAGE_STATIC` in vsm_common.wgsl); zero
    /// elsewhere except between the software raster and the page clear.
    sw_depth: wgpu::Buffer,
    levels: Vec<CullView>,
    level_count: u32,
    mark_layout: wgpu::BindGroupLayout,
    expand_layout: wgpu::BindGroupLayout,
    draw_layout: wgpu::BindGroupLayout,
    sw_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    mark: wgpu::ComputePipeline,
    mark_volume: wgpu::ComputePipeline,
    mark_local: wgpu::ComputePipeline,
    invalidate_local: wgpu::ComputePipeline,
    /// Volumetric fog samples the shadow map this frame.
    volume_marking: bool,
    info_begin: wgpu::ComputePipeline,
    dirty_info: wgpu::ComputePipeline,
    refresh: wgpu::ComputePipeline,
    expand: wgpu::ComputePipeline,
    expand_local: wgpu::ComputePipeline,
    /// Local lights' shadow maps (pages of this cache and atlas).
    pub local: LocalShadows,
    draw_args: wgpu::ComputePipeline,
    raster_sw: wgpu::ComputePipeline,
    /// The software rasteriser specialised for alpha-masked clusters.
    raster_sw_masked: wgpu::ComputePipeline,
    /// Voxel clusters' page items (vsm_voxel_raster.wgsl).
    raster_voxels: wgpu::ComputePipeline,
    resolve: wgpu::RenderPipeline,
    /// Hardware page raster without and with skinning (`SKINNING` in vsm_page.wgsl): scenes
    /// without skinned instances use the leaner one.
    raster: [wgpu::RenderPipeline; 2],
    static_layout: wgpu::BindGroupLayout,
    clear_static: wgpu::ComputePipeline,
    capture_static: wgpu::ComputePipeline,
    invalidation: PageInvalidation,
    /// Footprints of moved instances, with their recent speeds.
    moves: MoveFootprints,
    /// Geometry residency changed since the last recorded frame.
    residency_changed: bool,
    /// The wind moves shadow casters this frame.
    wind: bool,
    /// Instances moved, stopped or changed mobility this frame: their footprints are in the list
    /// already.
    instances_changed: bool,
    /// The wind moved shadow casters when the invalidation last ran: when it stops, the casters
    /// it moved return to the static layers once more.
    wind_collected: bool,
    last_sun: Vec3,
    enabled: bool,
}

impl VirtualShadowMap {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let physical = PHYSICAL_PAGES;
        // Clipmap levels, then the local lights' pages.
        let level_entries = MAX_LEVELS as u32 * LEVEL_PAGES * LEVEL_PAGES;
        let entries = level_entries + vsm_local::ENTRIES;
        let cache = PageCache::new(device, "vsm cache", CacheClient::Clipmap, entries, physical);
        cache.clear(queue);
        let atlas_size = TILES_PER_ROW * PAGE_TEXELS;
        let atlas = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("vsm physical pages"),
                size: wgpu::Extent3d { width: atlas_size, height: atlas_size, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());

        let c = ShaderStages::COMPUTE;
        let dirty_start = [("DIRTY_START", dirty_list_start(physical) as f64)];
        let vsm_constants = [dirty_start[0], ("SCRATCH_TILES", SCRATCH_TILES as f64)];
        let split_constants = [("SPLIT_DRAWS", (SPLIT_DRAWS_OFFSET / 4) as f64), ("SPLIT_CAPACITY", SPLIT_DRAW_CAPACITY as f64)];
        let vsm_module = create_module(device, "vsm.wgsl");
        let mark_layout = bind_group_layout(
            device,
            "vsm mark",
            c,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::texture(false)),
                (3, bind::storage(false)),
                (4, bind::storage(false)),
                (5, bind::storage(false)),
                // The page cache lists (the refresh appends to its dirty list).
                (6, bind::storage(false)),
                // Local lights: their uniform, the lights and their clusters, receiver normals,
                // page owners (invalidation).
                (60, bind::uniform()),
                (61, bind::storage(true)),
                (62, bind::storage(true)),
                (63, bind::texture(false)),
                (64, bind::storage(false)),
            ],
        );
        let expand_module = create_module(device, "vsm_expand.wgsl");
        let mut expand_entries = vec![(0, bind::uniform()), (1, bind::uniform())];
        expand_entries.extend((2..=6).map(|b| (b, bind::storage(true))));
        expand_entries.push((7, bind::storage(false)));
        expand_entries.push((8, bind::storage(true)));
        expand_entries.push(wind_entry());
        expand_entries.push(skin_entry());
        expand_entries.push(tessellation_entry());
        expand_entries.push((60, bind::uniform()));
        let expand_layout = bind_group_layout(device, "vsm expand", c, &expand_entries);
        let draw_module = create_module(device, "vsm_draw.wgsl");
        let mut draw_entries = vec![(0, bind::uniform())];
        draw_entries.extend((1..=7).map(|b| (b, bind::storage(true))));
        draw_entries.extend(alpha_mask_entries());
        draw_entries.push(wind_entry());
        draw_entries.push(skin_entry());
        draw_entries.push((60, bind::uniform()));
        let draw_layout = bind_group_layout(device, "vsm draw", ShaderStages::VERTEX_FRAGMENT, &draw_entries);
        let mut sw_entries = draw_entries[..6].to_vec();
        sw_entries.push((6, bind::storage(false)));
        sw_entries.extend(alpha_mask_entries());
        sw_entries.push(wind_entry());
        sw_entries.push(skin_entry());
        sw_entries.push((60, bind::uniform()));
        let sw_layout = bind_group_layout(device, "vsm raster sw", c, &sw_entries);
        // Page quads are placed in the vertex stage; only the fragment stage may write storage.
        let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None };
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vsm resolve"),
            entries: &[
                entry(0, ShaderStages::VERTEX_FRAGMENT, bind::uniform()),
                entry(6, ShaderStages::VERTEX, bind::storage(true)),
                entry(7, ShaderStages::VERTEX, bind::storage(true)),
                entry(8, ShaderStages::FRAGMENT, bind::storage(false)),
                entry(9, ShaderStages::VERTEX, bind::storage(true)),
            ],
        });
        let static_module = create_module(device, "vsm_static.wgsl");
        let static_layout = bind_group_layout(
            device,
            "vsm static layers",
            c,
            &[
                (0, bind::uniform()),
                (1, bind::storage(true)),
                (2, bind::storage(true)),
                (3, bind::storage(true)),
                (4, bind::storage(false)),
                (5, bind::texture_depth()),
            ],
        );

        let layout = pipeline_layout(device, "vsm draw", &[&draw_layout]);
        let resolve_pipeline_layout = pipeline_layout(device, "vsm resolve", &[&resolve_layout]);
        let depth_state = |compare, bias: wgpu::DepthBiasState| wgpu::DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(compare),
            stencil: Default::default(),
            bias,
        };
        // Clip page draws to their tile where the device can (see vsm_draw_clipped.wgsl).
        let clipped = device.features().contains(wgpu::Features::CLIP_DISTANCES);
        let clipped_module = clipped.then(|| create_module(device, "vsm_draw_clipped.wgsl"));
        let render = |label: &str,
                      layout: &wgpu::PipelineLayout,
                      module: &wgpu::ShaderModule,
                      (vs, constants): (&str, &[(&str, f64)]),
                      fs: &str,
                      depth: wgpu::DepthStencilState,
                      cull: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs),
                    compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState { cull_mode: cull, ..Default::default() },
                depth_stencil: Some(depth),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        // Reverse-Z: push stored depth away from the light by a slope-scaled amount against acne.
        let bias = wgpu::DepthBiasState { constant: 0, slope_scale: -SLOPE_BIAS, clamp: 0.0 };
        let raster_module = create_module(device, "vsm_raster.wgsl");
        let info = storage_buffer(device, "vsm info", INFO_WORDS * 4, wgpu::BufferUsages::INDIRECT);
        // No page has a static layer yet.
        queue.write_buffer(&info, INFO_PAGE_STATIC as u64 * 4, &vec![0xffu8; PHYSICAL_PAGES as usize * 4]);
        let vsm_pipeline = |label: &str, entry: &str| compute_pipeline(device, label, &vsm_module, entry, &mark_layout, &vsm_constants);
        let expand_pipeline = |label: &str, entry: &str| compute_pipeline(device, label, &expand_module, entry, &expand_layout, &split_constants);
        let static_pipeline = |entry: &str| compute_pipeline(device, "vsm static layers", &static_module, entry, &static_layout, &dirty_start);
        Self {
            mark: vsm_pipeline("vsm mark", "mark"),
            mark_volume: vsm_pipeline("vsm mark volume", "mark_volume"),
            mark_local: vsm_pipeline("vsm mark local", "mark_local"),
            invalidate_local: vsm_pipeline("vsm invalidate local", "invalidate_local"),
            volume_marking: false,
            info_begin: vsm_pipeline("vsm info", "info_begin"),
            dirty_info: vsm_pipeline("vsm info", "dirty_info"),
            refresh: vsm_pipeline("vsm refresh", "refresh"),
            expand: expand_pipeline("vsm expand", "expand"),
            expand_local: expand_pipeline("vsm expand local", "expand_local"),
            local: LocalShadows::new(device, level_entries, INFO_LOCAL_SLOTS, LOCAL_CAPACITY),
            draw_args: expand_pipeline("vsm draw args", "draw_args"),
            clear_static: static_pipeline("clear_static"),
            capture_static: static_pipeline("capture_static"),
            static_layout,
            raster_sw: compute_pipeline(
                device,
                "vsm raster sw",
                &raster_module,
                "rasterize",
                &sw_layout,
                &[("SLOPE_BIAS", SLOPE_BIAS as f64)],
            ),
            raster_sw_masked: compute_pipeline(
                device,
                "vsm raster sw (alpha-masked)",
                &raster_module,
                "rasterize",
                &sw_layout,
                &[("SLOPE_BIAS", SLOPE_BIAS as f64), ("MASKED", 1.0)],
            ),
            raster_voxels: compute_pipeline(
                device,
                "vsm raster voxels",
                &create_module(device, "vsm_voxel_raster.wgsl"),
                "rasterize",
                &sw_layout,
                &[("SLOPE_BIAS", SLOPE_BIAS as f64)],
            ),
            resolve: render(
                "vsm resolve",
                &resolve_pipeline_layout,
                &draw_module,
                ("vs_clear", &dirty_start),
                "fs_resolve",
                depth_state(wgpu::CompareFunction::Always, Default::default()),
                None,
            ),
            raster: [0.0, 1.0].map(|skinning| {
                render(
                    "vsm raster",
                    &layout,
                    clipped_module.as_ref().unwrap_or(&draw_module),
                    (if clipped { "vs_raster_clipped" } else { "vs_raster" }, &[dirty_start[0], ("SKINNING", skinning)]),
                    "fs_raster",
                    depth_state(wgpu::CompareFunction::Greater, bias),
                    Some(wgpu::Face::Back),
                )
            }),
            uniform: uniform_buffer(device, "vsm", std::mem::size_of::<VsmUniform>() as u64),
            info,
            // COPY_SRC: the header is read back for statistics.
            draws: storage_buffer(device, "vsm draws", SPLIT_DRAWS_OFFSET + SPLIT_DRAWS_SIZE, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC),
            sw_depth: storage_buffer(device, "vsm software depth", MIRROR_TEXELS * 4, wgpu::BufferUsages::empty()),
            levels: (0..MAX_LEVELS).map(|i| CullView::new(device, &format!("vsm level {i}"), LEVEL_CAPACITY)).collect(),
            level_count: 0,
            mark_layout,
            expand_layout,
            draw_layout,
            sw_layout,
            resolve_layout,
            invalidation: PageInvalidation::new(device),
            moves: MoveFootprints::default(),
            cache,
            atlas,
            last_sun: Vec3::ZERO,
            residency_changed: true,
            wind: false,
            instances_changed: false,
            wind_collected: false,
            enabled: false,
        }
    }

    /// Places the clipmap around the camera and writes all uniforms.
    pub fn update(&mut self, queue: &wgpu::Queue, f: &VsmFrame) {
        let levels = f.levels.clamp(1, MAX_LEVELS as u32);
        self.level_count = levels;
        self.enabled = f.enabled;
        self.volume_marking = f.volume_distance > 0.0;
        self.residency_changed |= f.geometry_streamed;
        self.wind = f.wind;
        let sun = f.sun_direction.normalize();
        let up = if sun.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        let light_view = look_to_mat4(Vec3::ZERO, -sun, up);

        // Depth range: the whole scene along the light direction (fixed, so pages stay valid).
        let mut z_min = f32::MAX;
        let mut z_max = f32::MIN;
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 { f.scene_min.x } else { f.scene_max.x },
                if i & 2 == 0 { f.scene_min.y } else { f.scene_max.y },
                if i & 4 == 0 { f.scene_min.z } else { f.scene_max.z },
            );
            let z = light_view.transform_point3(corner).z;
            z_min = z_min.min(z);
            z_max = z_max.max(z);
        }
        let (near, far) = (-z_max - 1.0, -z_min + 1.0);

        let eye_ls = light_view.transform_point3(f.camera.position);
        let mut u = VsmUniform {
            light_view,
            level_view_proj: [Mat4::IDENTITY; MAX_LEVELS],
            level_origin: [[0; 4]; MAX_LEVELS],
            level_info: [Vec4::ZERO; MAX_LEVELS],
            level_count: levels,
            page_texels: PAGE_TEXELS,
            level_pages: LEVEL_PAGES,
            tiles_per_row: TILES_PER_ROW,
            atlas_size: (TILES_PER_ROW * PAGE_TEXELS) as f32,
            first_texel: FIRST_TEXEL,
            lod_bias: f.resolution_bias,
            frame: f.frame,
            pixel_angle: 2.0 * (f.camera.fov_y * 0.5).tan() / f.viewport_height,
            physical_count: TILES_PER_ROW * TILES_PER_ROW,
            enabled: f.enabled as u32,
            draw_capacity: DRAW_CAPACITY,
            eye: f.camera.position.extend(f.volume_distance),
        };
        if f.scene_changed {
            self.moves.reset();
        }
        let footprints = self.moves.update(f.moves, &light_view);
        self.instances_changed = f.instances_changed || !footprints.is_empty();
        if self.instances_changed {
            self.invalidation.write_moves(queue, &footprints);
        }
        let mut cache = self.cache.default_params(f.frame, RETAIN_FRAMES, 1);
        cache.level_pages = LEVEL_PAGES;
        cache.level_count = MAX_LEVELS as u32;
        let sun_moved = (sun - self.last_sun).length() > 1e-5;
        cache.invalidate = (sun_moved || f.scene_changed) as u32;
        self.last_sun = sun;

        for level in 0..levels as usize {
            let texel = FIRST_TEXEL * (1u32 << level) as f32;
            let page_world = texel * PAGE_TEXELS as f32;
            let extent = page_world * LEVEL_PAGES as f32;
            // Clipmap origin in absolute pages (v axis = light -Y), centred on the camera.
            let center = glam::Vec2::new(eye_ls.x, -eye_ls.y) / page_world;
            let origin = center.floor().as_ivec2() - glam::IVec2::splat(LEVEL_PAGES as i32 / 2);
            let x0 = origin.x as f32 * page_world;
            let v0 = origin.y as f32 * page_world;
            // Reverse-Z orthographic: pass (far, near).
            let proj = directx::orthographic(x0, x0 + extent, -(v0 + extent), -v0, far, near);
            let view_proj = proj * light_view;
            u.level_view_proj[level] = view_proj;
            u.level_origin[level] = [origin.x, origin.y, 0, 0];
            u.level_info[level] = Vec4::new(page_world, texel, extent, 0.0);
            cache.origins[level] = [origin.x, origin.y, 0, 0];

            let view = &self.levels[level];
            let res = (LEVEL_PAGES * PAGE_TEXELS) as f32;
            let uniform = CullViewUniform {
                view_proj,
                view: light_view,
                planes: frustum_planes(&view_proj),
                position: f.camera.position.extend(0.0),
                viewport: Vec4::new(res, res, 1.0 / res, 1.0 / res),
                lod_scale: 1.0 / texel,
                lod_threshold: f.lod_threshold,
                orthographic: 1,
                instance_count: f.instance_count,
                work_capacity: view.capacity.pages,
                cluster_capacity: view.capacity.clusters,
                hzb_mips: 1,
                frame_index: f.frame,
                sw_raster_threshold: f.sw_raster_threshold * SW_THRESHOLD_SCALE,
                kind: 1,
                batch_capacity: view.capacity.batches,
                masked_capacity: 0,
                vsm: Vec4::new(level as f32, page_world, origin.x as f32, origin.y as f32),
                flags: f.cull_flags,
                _pad: [0; 3],
            };
            queue.write_buffer(&view.uniform, 0, bytemuck::bytes_of(&uniform));
        }
        // Levels beyond the active count must never be requested; park their origins far away.
        for level in levels as usize..MAX_LEVELS {
            cache.origins[level] = [i32::MIN / 4, i32::MIN / 4, 0, 0];
        }
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        self.cache.write_params(queue, &cache);
        let local = LocalCull {
            instance_count: f.instance_count,
            frame: f.frame,
            lod_threshold: f.lod_threshold,
            sw_raster_threshold: f.sw_raster_threshold * SW_THRESHOLD_SCALE,
            flags: f.cull_flags,
        };
        self.local.update(queue, f.local_resolution_bias, &local);
    }

    /// Makes room in every shadow culling view for `extra` more page batches: the part
    /// instances of assemblies take one each.
    pub fn reserve_batches(&mut self, device: &wgpu::Device, extra: u32) {
        for view in &mut self.levels {
            view.reserve_batches(device, LEVEL_CAPACITY.batches + extra);
        }
        self.local.reserve_batches(device, LOCAL_CAPACITY.batches, extra);
    }

    /// Chooses the local lights that get shadow maps this frame (before the lights are
    /// uploaded: they carry their slots).
    pub fn assign_local_lights(&mut self, f: &LocalFrame) {
        self.local.assign(f);
    }

    /// Records marking, allocation and page rendering. `frame` is the frame uniform buffer; the
    /// targets provide the merged scene depth and the receivers' normals, `lights` the local
    /// lights and their clusters.
    pub fn record(&mut self, rec: &mut Recorder, culling: &Culling, scene: &GpuScene, frame: &wgpu::Buffer, targets: &FrameTargets, lights: &LightCulling) {
        if !self.enabled {
            return;
        }
        let device = rec.device;
        let mark_group = bind_group(
            device,
            "vsm mark",
            &self.mark_layout,
            &[
                (0, self.uniform.as_entire_binding()),
                (1, frame.as_entire_binding()),
                (2, TextureView(&targets.depth)),
                (3, self.cache.requests.as_entire_binding()),
                (4, self.cache.entries.as_entire_binding()),
                (5, self.info.as_entire_binding()),
                (6, self.cache.lists.as_entire_binding()),
                (60, self.local.uniform.as_entire_binding()),
                (61, lights.lights.as_entire_binding()),
                (62, lights.grid.as_entire_binding()),
                (63, TextureView(&targets.gbuffer_normal)),
                (64, self.cache.owner.as_entire_binding()),
            ],
        );
        let local = self.local.view().is_some();
        {
            let mut pass = rec.compute_pass("vsm mark");
            pass.set_bind_group(0, &mark_group, &[]);
            pass.set_pipeline(&self.mark);
            pass.dispatch_workgroups(workgroups(targets.width, 8), workgroups(targets.height, 8), 1);
            if self.volume_marking {
                pass.set_pipeline(&self.mark_volume);
                pass.dispatch_workgroups(32 / 4, 18_u32.div_ceil(4), 32 / 4);
            }
        }
        if local {
            let mut pass = rec.compute_pass("vsm mark local");
            pass.set_bind_group(0, &mark_group, &[]);
            pass.set_pipeline(&self.mark_local);
            // One thread per 2x2 pixel quad.
            pass.dispatch_workgroups(workgroups(targets.width, 16), workgroups(targets.height, 16), 1);
            if self.local.any_stale() {
                pass.set_pipeline(&self.invalidate_local);
                pass.dispatch_workgroups(workgroups(self.cache.physical_count, 64), 1, 1);
            }
        }
        let residency_changed = std::mem::take(&mut self.residency_changed);
        if residency_changed || self.wind || self.wind_collected || self.instances_changed {
            let inputs = InvalidationInputs { uniform: &self.uniform, local: &self.local.uniform, cache: &self.cache, info: &self.info };
            let switched = self.wind != self.wind_collected;
            self.invalidation.record(rec, &inputs, scene, residency_changed, switched, self.instances_changed);
            self.wind_collected = self.wind;
        }
        self.cache.record(rec);
        rec.encoder.clear_buffer(&self.info, INFO_DIRTY_MASK * 4, Some(2 * DIRTY_MASK_WORDS * 4));
        {
            // The page cache's dirty pages, then the pages to refresh appended to them.
            let mut pass = rec.compute_pass("vsm dirty info");
            pass.set_bind_group(0, &mark_group, &[]);
            pass.set_pipeline(&self.info_begin);
            pass.dispatch_workgroups(workgroups(INFO_ITEMS, 64), 1, 1);
            pass.set_pipeline(&self.dirty_info);
            pass.dispatch_workgroups(workgroups(self.cache.physical_count, 64), 1, 1);
            pass.set_pipeline(&self.refresh);
            pass.dispatch_workgroups(workgroups(self.cache.physical_count, 64), 1, 1);
        }

        // The page draws, and those of dynamic casters into split pages.
        let draw_group = |draws: wgpu::BindingResource| {
            bind_group(
                device,
                "vsm draw",
                &self.draw_layout,
                &[
                    (0, self.uniform.as_entire_binding()),
                    (1, draws),
                    (2, scene.instances.as_entire_binding()),
                    (3, scene.clusters.as_entire_binding()),
                    (4, scene.page_table.as_entire_binding()),
                    (5, scene.geometry.as_entire_binding()),
                    (6, self.cache.lists.as_entire_binding()),
                    (7, self.cache.entries.as_entire_binding()),
                    (60, self.local.uniform.as_entire_binding()),
                ]
                .into_iter()
                .chain(alpha_mask_resources(scene))
                .chain([wind_resource(scene), skin_resource(scene)])
                .collect::<Vec<_>>(),
            )
        };
        let split_draws = wgpu::BufferBinding { buffer: &self.draws, offset: SPLIT_DRAWS_OFFSET, size: wgpu::BufferSize::new(SPLIT_DRAWS_SIZE) };
        let draw_groups = [draw_group(self.draws.as_entire_binding()), draw_group(wgpu::BindingResource::Buffer(split_draws))];
        let expand_group = |view: &CullView| {
            bind_group(
                device,
                "vsm expand",
                &self.expand_layout,
                &[
                    (0, self.uniform.as_entire_binding()),
                    (1, view.uniform.as_entire_binding()),
                    (2, view.visible_clusters.as_entire_binding()),
                    (3, view.state.as_entire_binding()),
                    (4, scene.instances.as_entire_binding()),
                    (5, scene.clusters.as_entire_binding()),
                    (6, self.cache.entries.as_entire_binding()),
                    (7, self.draws.as_entire_binding()),
                    (8, self.info.as_entire_binding()),
                    wind_resource(scene),
                    skin_resource(scene),
                    tessellation_resource(scene),
                    (60, self.local.uniform.as_entire_binding()),
                ],
            )
        };
        // Cull each level against its dirty rectangle, then expand to page draws; then each
        // local light against the box around its dirty pages.
        rec.encoder.clear_buffer(&self.draws, 0, Some(DRAW_HEADER_WORDS * 4));
        rec.encoder.clear_buffer(&self.draws, SPLIT_DRAWS_OFFSET, Some(DRAW_HEADER_WORDS * 4));
        let levels = self.levels[..self.level_count as usize].iter().map(|view| (view, &self.expand));
        for (view, expand) in levels.chain(self.local.view().map(|view| (view, &self.expand_local))) {
            culling.cull_with_atomics(rec, view, scene, CullMode::Single, &self.info);
            let group = expand_group(view);
            let mut pass = rec.compute_pass("vsm expand");
            pass.set_bind_group(0, &group, &[]);
            pass.set_pipeline(expand);
            pass.dispatch_workgroups_indirect(&view.args, crate::passes::culling::ARGS_EARLY_SW);
        }
        let group = expand_group(&self.levels[0]);
        {
            let mut pass = rec.compute_pass("vsm expand");
            pass.set_bind_group(0, &group, &[]);
            pass.set_pipeline(&self.draw_args);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Mirror tiles that still hold an older static layer are cleared, then software clusters
        // are rasterised into the u32 mirror (dynamic casters of pages refreshed over their
        // static layer into scratch tiles), then the page clear writes their depth (or far) into
        // every dirty page, merged with its static layer, then hardware clusters are
        // depth-tested against it. Split pages capture their static layer before their dynamic
        // casters are drawn.
        let static_group = bind_group(
            device,
            "vsm static layers",
            &self.static_layout,
            &[
                (0, self.uniform.as_entire_binding()),
                (1, self.cache.lists.as_entire_binding()),
                (2, self.cache.entries.as_entire_binding()),
                (3, self.info.as_entire_binding()),
                (4, self.sw_depth.as_entire_binding()),
                (5, TextureView(&self.atlas)),
            ],
        );
        {
            let mut pass = rec.compute_pass("vsm static layers");
            pass.set_bind_group(0, &static_group, &[]);
            pass.set_pipeline(&self.clear_static);
            pass.dispatch_workgroups(self.cache.physical_count, 1, 1);
        }
        let sw_group = bind_group(
            device,
            "vsm raster sw",
            &self.sw_layout,
            &[
                (0, self.uniform.as_entire_binding()),
                (1, self.draws.as_entire_binding()),
                (2, scene.instances.as_entire_binding()),
                (3, scene.clusters.as_entire_binding()),
                (4, scene.page_table.as_entire_binding()),
                (5, scene.geometry.as_entire_binding()),
                (6, self.sw_depth.as_entire_binding()),
                (60, self.local.uniform.as_entire_binding()),
            ]
            .into_iter()
            .chain(alpha_mask_resources(scene))
            .chain([wind_resource(scene), skin_resource(scene)])
            .collect::<Vec<_>>(),
        );
        {
            let mut pass = rec.compute_pass("vsm raster sw");
            pass.set_bind_group(0, &sw_group, &[]);
            pass.set_pipeline(&self.raster_sw);
            pass.dispatch_workgroups_indirect(&self.draws, SW_DISPATCH_OFFSET);
            if scene.masks.any() {
                pass.set_pipeline(&self.raster_sw_masked);
                pass.dispatch_workgroups_indirect(&self.draws, SW_DISPATCH_OFFSET);
            }
            if scene.has_voxels() {
                pass.set_pipeline(&self.raster_voxels);
                pass.dispatch_workgroups_indirect(&self.draws, SW_DISPATCH_OFFSET);
            }
        }
        if let Some(tessellation) = culling.tessellation(scene) {
            let inputs = ShadowInputs {
                vsm: &self.uniform,
                draws: &self.draws,
                sw_depth: &self.sw_depth,
                local: &self.local.uniform,
                dispatch_offset: SW_DISPATCH_OFFSET,
            };
            tessellation.record_shadows(rec, scene, &inputs);
        }
        let resolve_group = bind_group(
            device,
            "vsm resolve",
            &self.resolve_layout,
            &[
                (0, self.uniform.as_entire_binding()),
                (6, self.cache.lists.as_entire_binding()),
                (7, self.cache.entries.as_entire_binding()),
                (8, self.sw_depth.as_entire_binding()),
                (9, self.info.as_entire_binding()),
            ],
        );
        {
            let mut pass = self.atlas_pass(rec, "vsm resolve");
            pass.set_pipeline(&self.resolve);
            pass.set_bind_group(0, &resolve_group, &[]);
            pass.draw_indirect(&self.info, INFO_ARGS as u64 * 4);
        }
        {
            let mut pass = self.atlas_pass(rec, "vsm raster");
            pass.set_pipeline(&self.raster[scene.skins.skinned_instances as usize]);
            pass.set_bind_group(0, &draw_groups[0], &[]);
            pass.draw_indirect(&self.draws, 0);
        }
        {
            let mut pass = rec.compute_pass("vsm static layers");
            pass.set_bind_group(0, &static_group, &[]);
            pass.set_pipeline(&self.capture_static);
            pass.dispatch_workgroups(self.cache.physical_count, 1, 1);
        }
        let mut pass = self.atlas_pass(rec, "vsm raster");
        pass.set_pipeline(&self.raster[scene.skins.skinned_instances as usize]);
        pass.set_bind_group(0, &draw_groups[1], &[]);
        pass.draw_indirect(&self.draws, SPLIT_DRAWS_OFFSET);
    }

    /// A render pass that keeps and updates the physical depth atlas.
    fn atlas_pass<'a>(&self, rec: &'a mut Recorder, label: &str) -> wgpu::RenderPass<'a> {
        rec.render_pass(
            label,
            &[],
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.atlas,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
        )
    }

    /// The (cluster, page) draws buffer, whose header counts the frame's draws (statistics).
    pub fn draws(&self) -> &wgpu::Buffer {
        &self.draws
    }

    /// Culling statistics buffers of the levels and local lights rendered this frame (for the
    /// HUD).
    pub fn level_states(&self) -> impl Iterator<Item = &wgpu::Buffer> {
        let local = self.local.view().filter(|_| self.enabled);
        self.levels[..self.level_count as usize].iter().chain(local).map(|v| &v.state)
    }
}

/// The buffers [`PageInvalidation`] works on.
struct InvalidationInputs<'a> {
    uniform: &'a wgpu::Buffer,
    local: &'a wgpu::Buffer,
    cache: &'a PageCache,
    info: &'a wgpu::Buffer,
}

/// Invalidates cached shadow pages under instances whose geometry changed (see
/// `shaders/vsm_invalidate.wgsl`): their streamed pages changed residency, the wind moves them
/// or they were moved ([`vsm_motion`]). Only those pages are re-rendered instead of the whole
/// cache, and only their dynamic casters where the static ones did not change.
struct PageInvalidation {
    layout: wgpu::BindGroupLayout,
    /// Indexed by [residency changed][wind switched on or off]: the first also collects
    /// instances whose streamed pages changed residency, the second frees the pages of casters
    /// that changed layer instead of dropping their static layer.
    collect: [[wgpu::ComputePipeline; 2]; 2],
    remember_residency: wgpu::ComputePipeline,
    invalidate_pages: wgpu::ComputePipeline,
    /// The page table as of the previous residency check (resized with the scene's).
    previous_page_table: wgpu::Buffer,
    /// Count + light-space footprints of the changed instances.
    footprints: wgpu::Buffer,
    /// Per instance: the wind reach of its last collected pose, 0 for static casters (resized
    /// with the scene's instances).
    dynamic_reach: wgpu::Buffer,
}

impl PageInvalidation {
    /// Footprints before the list overflows into a global invalidation: well above the forest's
    /// shadow casters within the wind's disable distance.
    const CAPACITY: u64 = 16383;
    /// `struct Footprints` in vsm_invalidate.wgsl: the count, then 16-byte aligned items of
    /// `struct Footprint` (vec4, vec2, 3 × f32, vec2 at 40: 48 bytes).
    const FOOTPRINTS_HEADER: u64 = 16;
    const FOOTPRINT_BYTES: u64 = 48;

    fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "vsm_invalidate.wgsl");
        let mut entries = vec![(0, bind::uniform()), (1, bind::storage(true)), (2, bind::storage(true))];
        entries.extend((3..=8).map(|b| (b, bind::storage(false))));
        entries.push(wind_entry());
        entries.push(skin_entry());
        entries.push((60, bind::uniform()));
        let layout = bind_group_layout(device, "vsm invalidate", ShaderStages::COMPUTE, &entries);
        let pipeline = |entry: &str, residency: bool, abrupt: bool| {
            let flag = |on: bool| if on { 1.0 } else { 0.0 };
            let constants = [("CAPACITY", Self::CAPACITY as f64), ("RESIDENCY", flag(residency)), ("ABRUPT", flag(abrupt))];
            compute_pipeline(device, "vsm invalidate", &module, entry, &layout, &constants)
        };
        Self {
            collect: [false, true].map(|residency| [false, true].map(|abrupt| pipeline("collect", residency, abrupt))),
            remember_residency: pipeline("remember_residency", false, false),
            invalidate_pages: pipeline("invalidate_pages", false, false),
            layout,
            previous_page_table: storage_buffer(device, "vsm previous page table", 16, wgpu::BufferUsages::empty()),
            footprints: storage_buffer(
                device,
                "vsm invalidated footprints",
                Self::FOOTPRINTS_HEADER + Self::CAPACITY * Self::FOOTPRINT_BYTES,
                wgpu::BufferUsages::empty(),
            ),
            dynamic_reach: storage_buffer(device, "vsm dynamic casters", 16, wgpu::BufferUsages::empty()),
        }
    }

    /// Starts the frame's footprint list with the moves' (`record` then appends to it).
    fn write_moves(&self, queue: &wgpu::Queue, footprints: &[vsm_motion::Footprint]) {
        // Beyond the capacity the count alone overflows the list: every page is freed.
        let count = footprints.len().min(Self::CAPACITY as usize + 1);
        let mut header = [0u32; Self::FOOTPRINTS_HEADER as usize / 4];
        header[0] = count as u32;
        queue.write_buffer(&self.footprints, 0, bytemuck::cast_slice(&header));
        let items = &footprints[..count.min(Self::CAPACITY as usize)];
        if !items.is_empty() {
            queue.write_buffer(&self.footprints, Self::FOOTPRINTS_HEADER, bytemuck::cast_slice(items));
        }
    }

    /// Records the passes; must precede the page cache update. `residency_changed`: streaming
    /// loaded or evicted pages since the last call (otherwise only the wind and moves
    /// invalidate); `wind_switched`: the wind was switched on or off since the last call;
    /// `moves_written`: the list starts with this frame's moves (`write_moves`).
    fn record(
        &mut self,
        rec: &mut Recorder,
        inputs: &InvalidationInputs,
        scene: &GpuScene,
        residency_changed: bool,
        wind_switched: bool,
        moves_written: bool,
    ) {
        let device = rec.device;
        if self.previous_page_table.size() != scene.page_table.size() {
            // New geometry also invalidates every page globally, so the contents do not matter.
            let size = scene.page_table.size();
            self.previous_page_table = storage_buffer(device, "vsm previous page table", size, wgpu::BufferUsages::empty());
        }
        let reach_size = (scene.instance_count as u64 * 4).max(16).next_multiple_of(16);
        if self.dynamic_reach.size() != reach_size {
            // New instances invalidate every page globally: all start as static casters.
            self.dynamic_reach = storage_buffer(device, "vsm dynamic casters", reach_size, wgpu::BufferUsages::empty());
        }
        let cache = inputs.cache;
        let group = bind_group(
            device,
            "vsm invalidate",
            &self.layout,
            &[
                (0, inputs.uniform.as_entire_binding()),
                (1, scene.instances.as_entire_binding()),
                (2, scene.page_table.as_entire_binding()),
                (3, self.previous_page_table.as_entire_binding()),
                (4, self.footprints.as_entire_binding()),
                (5, cache.owner.as_entire_binding()),
                (6, cache.entries.as_entire_binding()),
                (7, self.dynamic_reach.as_entire_binding()),
                (8, inputs.info.as_entire_binding()),
                wind_resource(scene),
                skin_resource(scene),
                (60, inputs.local.as_entire_binding()),
            ],
        );
        if !moves_written {
            rec.encoder.clear_buffer(&self.footprints, 0, Some(Self::FOOTPRINTS_HEADER));
        }
        let mut pass = rec.compute_pass("vsm invalidate");
        pass.set_bind_group(0, &group, &[]);
        pass.set_pipeline(&self.collect[residency_changed as usize][wind_switched as usize]);
        pass.dispatch_workgroups(workgroups(scene.instance_count, 64), 1, 1);
        if residency_changed {
            pass.set_pipeline(&self.remember_residency);
            pass.dispatch_workgroups(workgroups((scene.page_table.size() / 4) as u32, 64), 1, 1);
        }
        pass.set_pipeline(&self.invalidate_pages);
        pass.dispatch_workgroups(cache.physical_count, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value of `const NAME: u32 = ...u;` in a shader (with its includes).
    fn shader_constant(shader: &str, name: &str) -> u32 {
        let source = crate::shader::compose(shader);
        let prefix = format!("const {name}: u32 = ");
        let line = source.lines().find_map(|l| l.trim().strip_prefix(prefix.as_str())).unwrap_or_else(|| panic!("{name} not in {shader}"));
        let value = line.trim_end_matches(';').trim_end_matches('u');
        match value.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).unwrap(),
            None => value.parse().unwrap(),
        }
    }

    #[test]
    fn info_layout_matches_the_shaders() {
        let words = [
            ("INFO_ARGS", INFO_ARGS),
            ("INFO_DIRTY_MASK", INFO_DIRTY_MASK as u32),
            ("INFO_FULL_MASK", (INFO_DIRTY_MASK + DIRTY_MASK_WORDS) as u32),
            ("INFO_PAGE_STATE", INFO_PAGE_STATE),
            ("INFO_PAGE_STATIC", INFO_PAGE_STATIC),
            ("INFO_PAGE_REFRESH", INFO_PAGE_STATIC + PHYSICAL_PAGES),
            ("LOCAL_INFO_SLOTS", INFO_LOCAL_SLOTS),
            ("LOCAL_INFO_SLOT_WORDS", LOCAL_SLOT_WORDS),
            ("LOCAL_INFO_RECTS", INFO_LOCAL_RECTS),
        ];
        for (name, value) in words {
            assert_eq!(shader_constant("vsm.wgsl", name), value, "{name}");
        }
        // The culling reads the masks and the local lights' boxes with its own constants.
        assert_eq!(shader_constant("cull.wgsl", "VSM_DIRTY_MASK"), INFO_DIRTY_MASK as u32);
        assert_eq!(shader_constant("cull.wgsl", "VSM_FULL_MASK"), (INFO_DIRTY_MASK + DIRTY_MASK_WORDS) as u32);
        assert_eq!(shader_constant("cull.wgsl", "LOCAL_SLOT_WORDS"), LOCAL_SLOT_WORDS);
    }

    #[test]
    fn scratch_tiles_fit_the_page_state() {
        // The scratch index is stored in the page state's upper 24 bits, NO_SCRATCH meaning none.
        assert!(SCRATCH_TILES < shader_constant("vsm.wgsl", "NO_SCRATCH"));
    }
}
