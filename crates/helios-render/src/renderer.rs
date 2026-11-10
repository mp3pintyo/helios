//! The frame graph: owns every pass and records one frame.
//!
//! ```text
//! sky LUTs ─┐
//! cull (early) → raster (early) → HZB → cull (late) → raster (late)      [visibility buffer]
//! cull + raster × cascades                                              [shadow maps]
//! G-buffer resolve → GI (trace + temporal) → lighting → translucency → TSR → DOF + motion blur → exposure
//!     → local exposure → bloom → tonemap
//! ```
//!
//! Everything up to the translucency renders at the internal resolution (the screen
//! percentage, see `resolution.rs`); temporal super resolution reconstructs the output
//! resolution, at which the camera effects, bloom and tone mapping run.

use glam::{Mat4, Vec2, Vec4};

use crate::camera::frustum_planes;
use crate::gpu_scene::GpuScene;
use crate::gpu_types::{flags, view_flags, CullViewUniform, FrameUniform};
use crate::passes::culling::{CullCapacity, CullMode, CullView, Culling};
use crate::profiler::{GpuProfiler, Recorder};
use crate::streaming::StreamingStats;
use crate::passes::hzb::Hzb;
use crate::passes::local_exposure::{LocalExposure, LocalExposurePass};
use crate::passes::post::{BloomPass, ExposurePass, TonemapPass};
use crate::passes::tsr::TsrPass;
use crate::passes::post_fx::{DepthOfField, PostFxPass};
use crate::passes::material::MaterialPass;
use crate::passes::shading::{GiPass, LightingInputs, LightingPass};
use crate::clouds::{CloudInputs, Clouds};
use crate::effects::{Effects, EffectsFrame};
use crate::fog::{FogInputs, VolumetricFog};
use crate::lights::{LightCulling, MAX_LIGHTS};
use crate::lumen::{CardInputs, Lumen};
use crate::picking::{PickResult, Picker};
use crate::lumen_lite::{GiMethod, LiteFallback, LiteInputs, LumenLite};
use crate::motion::MotionTracker;
use crate::megalights::{MegaLights, MegaLightsInputs};
use crate::resolution::{self, DynamicResolution, ResolutionController};
use crate::rvt::RuntimeVirtualTexture;
use crate::scene::InstanceHandle;
use crate::selection::SelectionOutline;
use crate::textures::TextureLibrary;
use crate::translucency::Translucency;
use crate::passes::sky::Sky;
use crate::passes::Samplers;
use crate::stats::{RenderStats, StatsReadback};
use crate::substrate_gpu::ExtendedMaterials;
use crate::subsurface::SubsurfacePass;
use crate::svt::{StreamingVirtualTexture, TextureStreamingStats, TileFormat};
use crate::targets::FrameTargets;
use crate::util::uniform_buffer;
use crate::vsm::{VirtualShadowMap, VsmFrame, MAX_LEVELS};
use crate::vsm_local::LocalFrame;
use crate::wind::WindUniform;
use crate::{Camera, Scene};

/// Visualisation modes, mirroring the `DEBUG_*` constants in common.wgsl.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DebugView {
    #[default]
    Lit = 0,
    Clusters = 1,
    Triangles = 2,
    LodLevel = 3,
    Instances = 4,
    Normals = 5,
    Albedo = 6,
    IndirectLight = 7,
    /// Virtual shadow map clipmap level and page grid.
    ShadowPages = 8,
    /// Software-rasterised clusters in red, hardware in green, alpha-masked hardware in blue.
    RasterPath = 9,
    /// Shading model per pixel (material classification).
    Materials = 10,
    /// Lumen surface cache (voxel lighting) at each pixel.
    SurfaceCache = 11,
    /// Ray-marched global distance field.
    DistanceField = 12,
    /// Indirect specular only (Lumen reflections, rough-surface fallback, sky).
    Reflections = 13,
    /// Lumen mesh-card surface cache at each pixel (purple: no card, voxel lighting).
    SurfaceCards = 14,
    /// Virtual shadow map pages of the local light that contributes most at each pixel: mip
    /// and page grid, shaded by its shadow.
    LocalShadowPages = 15,
    /// What moving instances invalidated this frame: movable instances tinted, shadow pages
    /// re-rendered and distance-field bricks recomposed (see `dynamic_debug.rs`).
    Invalidation = 16,
    /// Joint weights of skinned meshes: each joint's colour, blended by its weight (grey:
    /// unskinned).
    SkinWeights = 17,
    /// MegaLights: the light of each sample (one colour per light, dark where the shadow ray
    /// was blocked), averaged over frames by TSR.
    MegaLightsSamples = 18,
    /// MegaLights' local lighting: this frame's samples (left) and denoised (right).
    MegaLightsDenoising = 19,
    /// Lumen Lite's irradiance probes (spheres shaded with their irradiance; sleeping: blue,
    /// buried in geometry: red) over the irradiance they give each pixel. Runs Lumen Lite.
    LumenLiteProbes = 20,
    /// Particle overdraw (red to yellow) and heterogeneous volume march steps (blue to cyan).
    VolumesParticles = 21,
    /// Voxel clusters in colour (one per cluster), triangle clusters grey.
    Voxels = 22,
    /// Assembly part instances in colour (one per part), everything else grey.
    AssemblyParts = 23,
    /// Substrate slab features per pixel: clear coat blue, sheen magenta, subsurface red,
    /// transmission green, anisotropy yellow, specular / IOR cyan, unlit white (blended when
    /// several), brighter with more layers, striped for graph materials; plain materials grey.
    MaterialFeatures = 24,
    /// Nanite tessellation: displaced surfaces coloured by tessellation depth (blue: one
    /// micro-triangle per patch, through green, to red: deeply split), shaded per micro-triangle;
    /// the rest grey.
    Tessellation = 25,
}

impl DebugView {
    pub const ALL: [DebugView; 26] = [
        Self::Lit,
        Self::Clusters,
        Self::Triangles,
        Self::LodLevel,
        Self::Instances,
        Self::Normals,
        Self::Albedo,
        Self::IndirectLight,
        Self::ShadowPages,
        Self::RasterPath,
        Self::Materials,
        Self::SurfaceCache,
        Self::DistanceField,
        Self::Reflections,
        Self::SurfaceCards,
        Self::LocalShadowPages,
        Self::Invalidation,
        Self::SkinWeights,
        Self::MegaLightsSamples,
        Self::MegaLightsDenoising,
        Self::LumenLiteProbes,
        Self::VolumesParticles,
        Self::Voxels,
        Self::AssemblyParts,
        Self::MaterialFeatures,
        Self::Tessellation,
    ];

    /// The view with index `index` (its `--view` number), if any.
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|&v| v as usize == index)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Lit => "Lit",
            Self::Clusters => "Clusters",
            Self::Triangles => "Triangles",
            Self::LodLevel => "LOD level",
            Self::Instances => "Instances",
            Self::Normals => "Normals",
            Self::Albedo => "Albedo",
            Self::IndirectLight => "Indirect light",
            Self::ShadowPages => "Shadow pages (VSM levels)",
            Self::RasterPath => "Raster path (red = software)",
            Self::Materials => "Shading models",
            Self::SurfaceCache => "Lumen surface cache",
            Self::DistanceField => "Global distance field",
            Self::Reflections => "Reflections",
            Self::SurfaceCards => "Lumen surface cache (cards)",
            Self::LocalShadowPages => "Local light shadow pages (mips)",
            Self::Invalidation => "Invalidation (movable, re-rendered shadow pages, recomposed SDF bricks)",
            Self::SkinWeights => "Skin weights (joints)",
            Self::MegaLightsSamples => "MegaLights samples (light per sample, dark = blocked)",
            Self::MegaLightsDenoising => "MegaLights lighting: samples | denoised",
            Self::LumenLiteProbes => "Lumen Lite probes",
            Self::VolumesParticles => "Particle overdraw / volume march steps",
            Self::Voxels => "Voxel clusters (triangles grey)",
            Self::AssemblyParts => "Assembly parts",
            Self::MaterialFeatures => "Material features (Substrate slabs)",
            Self::Tessellation => "Tessellation (depth, micro-triangles)",
        }
    }

    pub fn next(self) -> Self {
        let position = Self::ALL.iter().position(|&v| v == self).unwrap_or(0);
        Self::ALL[(position + 1) % Self::ALL.len()]
    }
}

/// Runtime quality and feature switches.
#[derive(Clone, Copy, Debug)]
pub struct RenderSettings {
    /// Maximum allowed geometric error in pixels for the main view (Nanite uses ~1 px).
    pub lod_threshold: f32,
    pub shadow_lod_threshold: f32,
    pub debug_view: DebugView,
    /// Two-pass HZB occlusion culling for the main view.
    pub occlusion_culling: bool,
    pub shadows: bool,
    /// Virtual shadow maps for the most important shadow-casting local lights (others, or all
    /// when off, get distance-field shadows).
    pub local_light_shadow_maps: bool,
    /// Also give *moving* local lights virtual shadow maps (with slots to spare). A moving
    /// light's map is stale every frame, so it is re-rendered in full every frame: the
    /// cathedral's eleven swinging chandelier lights cost 3.5 ms of page rendering at 1440p.
    /// Off (default), they keep distance-field shadows while they move, like the lights beyond
    /// the shadow-map budget (MegaLights traces them in any case).
    pub moving_light_shadow_maps: bool,
    /// MegaLights: stochastic, denoised direct lighting in which every local light casts soft
    /// shadows, at a cost that barely grows with the light count (`megalights.rs`). Off: the
    /// clustered light loop (shadow maps for the 32 most important lights, distance-field shadows
    /// for four more per pixel).
    pub mega_lights: bool,
    pub global_illumination: bool,
    /// Final gather of the diffuse GI: Lumen's screen probes or Lumen Lite's irradiance field.
    pub gi_method: GiMethod,
    /// Switch to Lumen Lite when dynamic resolution stays at its minimum (see [`LiteFallback`]).
    pub gi_lite_fallback: bool,
    /// Temporal accumulation: sub-pixel jitter and the TSR history (off: TSR only upscales
    /// the current frame spatially).
    pub taa: bool,
    pub auto_exposure: bool,
    /// Exposure compensation in EV stops.
    pub exposure_compensation: f32,
    /// Compress the contrast between large bright and dark regions around the global exposure
    /// while keeping their detail (`None` = global exposure only).
    pub local_exposure: Option<LocalExposure>,
    pub gi_intensity: f32,
    /// World-space length of GI screen traces.
    pub gi_ray_length: f32,
    pub bloom_intensity: f32,
    pub aerial_perspective: bool,
    /// Multiplier on physical aerial perspective density (artistic haze).
    pub aerial_perspective_scale: f32,
    /// Virtual shadow map clipmap levels (level 0 texels are ~1 mm, each level doubles).
    pub shadow_levels: u32,
    /// Shift of the shadow page resolution in levels (positive = coarser, faster).
    pub shadow_resolution_bias: f32,
    /// The same for local lights' shadow maps, on top of `shadow_resolution_bias`: their
    /// shadows are soft and many lights share the page pool, so texels a little larger than
    /// the screen footprint by default (each level halves the resolution and quarters pages).
    pub local_shadow_resolution_bias: f32,
    /// Software-rasterise clusters whose average triangle is below this size in pixels
    /// (0 disables the software rasteriser).
    pub sw_raster_threshold: f32,
    /// Cache the landscape material in a runtime virtual texture.
    pub virtual_texturing: bool,
    /// Stream image textures at their native resolution (otherwise they are sampled from the
    /// 512² texture array layers only).
    pub texture_streaming: bool,
    /// Volumetric clouds (when the scene has a cloud layer).
    pub clouds: bool,
    /// Volumetric fog (when the scene has a fog volume).
    pub volumetric_fog: bool,
    /// Forward translucency pass (glass, water, blended materials). Off: translucent surfaces
    /// are not drawn at all.
    pub translucency: bool,
    /// Composite overlapping translucent surfaces in per-pixel depth order (depth peeling where
    /// they overlap on screen, see `translucency.rs`). Off: in per-instance order only.
    pub translucency_layers: bool,
    /// Fraction of a frame the virtual shutter stays open for motion blur (0.5 = 180° shutter,
    /// 0 = off).
    pub motion_blur: f32,
    /// Physically based depth of field (`None` = everything in focus).
    pub depth_of_field: Option<DepthOfField>,
    /// Animate vegetation with the scene's wind ([`crate::Scene::wind`]).
    pub wind: bool,
    /// Heterogeneous volumes ([`crate::Scene::volumes`]: smoke, fire, mist).
    pub volumes: bool,
    /// GPU particles ([`crate::Scene::particles`]).
    pub particles: bool,
    /// Voxel LODs of aggregate meshes (foliage): voxel clusters replace the simplified
    /// triangles they beat (off: the triangle-only cut of the same DAG).
    pub voxel_lods: bool,
    /// Nanite tessellation of displaced materials ([`crate::Displacement`]); off: their
    /// surfaces render undisplaced.
    pub tessellation: bool,
    /// Target length of micro-triangle edges in pixels (UE's dice rate, 2 by default): smaller is
    /// finer and costlier.
    pub dice_rate: f32,
    /// Internal render resolution in percent of the output (UE's `r.ScreenPercentage`,
    /// 25–100); temporal super resolution reconstructs the output. Ignored while dynamic
    /// resolution is on.
    pub screen_percentage: f32,
    /// Choose the screen percentage each frame to stay within a GPU time budget.
    pub dynamic_resolution: Option<DynamicResolution>,
    /// Apply instance moves incrementally: only the moved instances' GPU records, the shadow
    /// pages and distance-field bricks under their old and new bounds, their cards (see
    /// `motion.rs`). Off: every move rebuilds the instance data and invalidates every cached
    /// shadow page, as structural changes do (for comparisons).
    pub per_instance_invalidation: bool,
    /// Substrate slabs (clear coat, sheen, subsurface, transmission, IOR / specular colour,
    /// anisotropy, unlit). Off: every material renders as its plain metallic-roughness base;
    /// material graphs still run (`substrate.rs`).
    pub substrate: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            lod_threshold: 1.0,
            shadow_lod_threshold: 1.5,
            debug_view: DebugView::Lit,
            occlusion_culling: true,
            shadows: true,
            local_light_shadow_maps: true,
            moving_light_shadow_maps: false,
            mega_lights: false,
            global_illumination: true,
            gi_method: GiMethod::ScreenProbes,
            gi_lite_fallback: false,
            taa: true,
            auto_exposure: true,
            exposure_compensation: 0.0,
            local_exposure: Some(LocalExposure::default()),
            gi_intensity: 1.0,
            gi_ray_length: 6.0,
            bloom_intensity: 0.04,
            aerial_perspective: true,
            aerial_perspective_scale: 4.0,
            shadow_levels: 12,
            shadow_resolution_bias: 0.0,
            local_shadow_resolution_bias: 0.5,
            sw_raster_threshold: 2.5,
            virtual_texturing: true,
            texture_streaming: true,
            clouds: true,
            volumetric_fog: true,
            translucency: true,
            translucency_layers: true,
            motion_blur: 0.5,
            depth_of_field: None,
            wind: true,
            volumes: true,
            particles: true,
            voxel_lods: true,
            tessellation: true,
            dice_rate: 2.0,
            screen_percentage: 100.0,
            dynamic_resolution: None,
            per_instance_invalidation: true,
            substrate: true,
        }
    }
}

/// Capacities of the per-view GPU work lists.
const MAIN_CAPACITY: CullCapacity = CullCapacity { batches: 1 << 16, pages: 1 << 18, clusters: 1 << 20, masked: 1 << 18 };

/// Creation-time renderer configuration.
#[derive(Clone, Copy, Debug)]
pub struct RendererConfig {
    /// Size of the GPU pool that streamed cluster geometry lives in. Page payloads are
    /// compressed (about 13 bytes per vertex including its triangles): the largest demo scene,
    /// the cathedral at high detail, needs 80 MiB to be fully resident.
    pub geometry_pool_bytes: u64,
    /// Upper bound on the tile atlases of streamed image textures.
    pub texture_pool_bytes: u64,
    /// Block-compress the tile atlases where the device supports BC formats (3 instead of 8
    /// bytes per texel: 2.7× the tiles in the same pool).
    pub texture_compression: bool,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self { geometry_pool_bytes: 192 << 20, texture_pool_bytes: 64 << 20, texture_compression: true }
    }
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub settings: RenderSettings,
    frame_uniform: wgpu::Buffer,
    scene: GpuScene,
    targets: FrameTargets,
    samplers: Samplers,
    culling: Culling,
    main_view: CullView,
    vsm: VirtualShadowMap,
    hzb: Hzb,
    sky: Sky,
    material: MaterialPass,
    /// Substrate slabs and material graphs (`substrate_gpu.rs`) and the subsurface diffusion.
    extended: ExtendedMaterials,
    subsurface: SubsurfacePass,
    textures: TextureLibrary,
    rvt: RuntimeVirtualTexture,
    svt: StreamingVirtualTexture,
    lumen: Lumen,
    /// Created on first use (its probe atlases and buffers take 43 MiB).
    lumen_lite: Option<LumenLite>,
    lite_fallback: LiteFallback,
    lights: LightCulling,
    clouds: Clouds,
    fog: VolumetricFog,
    black: wgpu::TextureView,
    gi: GiPass,
    lighting: LightingPass,
    megalights: MegaLights,
    translucency: Translucency,
    effects: Effects,
    tsr: TsrPass,
    post_fx: PostFxPass,
    exposure: ExposurePass,
    local_exposure: LocalExposurePass,
    bloom: BloomPass,
    tonemap: TonemapPass,
    /// Editor tools: instance picking and the selection outline (idle unless used).
    picker: Picker,
    selection: SelectionOutline,
    stats: StatsReadback,
    profiler: GpuProfiler,
    /// This frame's instance moves.
    motion: MotionTracker,
    invalidation_debug: crate::dynamic_debug::InvalidationDebug,
    frame_index: u32,
    history_valid: bool,
    previous_view_proj: Option<Mat4>,
    /// Geometry or instances changed structurally this frame (invalidates cached shadow pages).
    scene_changed: bool,
    /// Voxel LODs and tessellation as the cached shadow pages were rendered.
    voxel_lods_cached: bool,
    tessellation_cached: bool,
    /// The displaced surfaces' shape (displacement table, displacing graphs and their
    /// parameters) the cached shadow pages were rendered with.
    displacement_cached: (u64, u64),
    time: f32,
    /// Multiplier on `lod_threshold` that keeps the main view's visible work inside the GPU
    /// list capacities (see [`Renderer::lod_budget_scale`]).
    lod_budget_scale: f32,
    /// Screen percentage of the current frame and the dynamic resolution controller.
    percentage: f32,
    resolution: ResolutionController,
    /// The previous frame's internal viewport (histories were written into that corner).
    previous_viewport: Vec4,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, output_format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        Self::with_config(device, queue, output_format, width, height, RendererConfig::default())
    }

    pub fn with_config(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        config: RendererConfig,
    ) -> Self {
        let limits = device.limits();
        let pool = config.geometry_pool_bytes.min(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        Self {
            device: device.clone(),
            queue: queue.clone(),
            settings: RenderSettings::default(),
            frame_uniform: uniform_buffer(device, "frame", std::mem::size_of::<FrameUniform>() as u64),
            scene: GpuScene::new(device, pool),
            targets: FrameTargets::new(device, (width, height), (width, height)),
            samplers: Samplers::new(device),
            culling: Culling::new(device),
            main_view: CullView::new(device, "main view", MAIN_CAPACITY),
            vsm: VirtualShadowMap::new(device, queue),
            hzb: Hzb::new(device),
            sky: Sky::new(device),
            material: MaterialPass::new(device),
            extended: ExtendedMaterials::new(device),
            subsurface: SubsurfacePass::new(device),
            textures: TextureLibrary::new(device, queue),
            rvt: RuntimeVirtualTexture::new(device, queue),
            svt: StreamingVirtualTexture::new(device, config.texture_pool_bytes, TileFormat::select(device.features(), config.texture_compression)),
            lumen: Lumen::new(device),
            lumen_lite: None,
            lite_fallback: LiteFallback::default(),
            lights: LightCulling::new(device),
            clouds: Clouds::new(device),
            fog: VolumetricFog::new(device),
            black: crate::util::texture(
                device,
                crate::util::TextureSpec {
                    label: "black",
                    width: 1,
                    height: 1,
                    layers: 1,
                    mips: 1,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                },
            )
            .create_view(&Default::default()),
            gi: GiPass::new(device),
            lighting: LightingPass::new(device),
            megalights: MegaLights::new(device, queue),
            translucency: Translucency::new(device),
            effects: Effects::new(device),
            tsr: TsrPass::new(device),
            post_fx: PostFxPass::new(device),
            exposure: ExposurePass::new(device),
            local_exposure: LocalExposurePass::new(device),
            bloom: BloomPass::new(device),
            tonemap: TonemapPass::new(device, output_format),
            picker: Picker::new(device),
            selection: SelectionOutline::new(device, output_format),
            stats: StatsReadback::new(device, 1 + MAX_LEVELS + 1, crate::vsm::PHYSICAL_PAGES, crate::vsm::DRAW_CAPACITY),
            profiler: GpuProfiler::new(device, queue),
            motion: MotionTracker::default(),
            invalidation_debug: crate::dynamic_debug::InvalidationDebug::new(device),
            frame_index: 0,
            history_valid: false,
            previous_view_proj: None,
            scene_changed: true,
            voxel_lods_cached: true,
            tessellation_cached: true,
            displacement_cached: (0, 0),
            time: 0.0,
            lod_budget_scale: 1.0,
            percentage: 100.0,
            resolution: ResolutionController::new(100.0),
            previous_viewport: viewport((width.max(1), height.max(1))),
        }
    }

    /// Sets the output resolution (the internal resolution follows from the screen percentage).
    pub fn resize(&mut self, width: u32, height: u32) {
        let output = (width.max(1), height.max(1));
        if output != self.targets.output {
            self.targets = FrameTargets::new(&self.device, self.capacity_for(output), output);
            self.history_valid = false;
        }
    }

    /// Output resolution.
    pub fn size(&self) -> (u32, u32) {
        self.targets.output
    }

    /// Internal (render) resolution of the latest frame.
    pub fn internal_size(&self) -> (u32, u32) {
        (self.targets.width, self.targets.height)
    }

    /// Screen percentage of the latest frame (the dynamic resolution's choice while it is on).
    pub fn screen_percentage(&self) -> f32 {
        self.percentage
    }

    /// Size of the internal targets for `output`: the largest internal resolution the
    /// settings allow, so that dynamic resolution renders into a corner of them instead of
    /// reallocating.
    fn capacity_for(&self, output: (u32, u32)) -> (u32, u32) {
        let s = &self.settings;
        let max = s.dynamic_resolution.map_or(s.screen_percentage, |d| d.max_percentage.max(d.min_percentage));
        resolution::internal_size(output, max)
    }

    /// Picks this frame's screen percentage (dynamic resolution: from the GPU time of recent
    /// frames) and sizes the internal viewport; reallocates the internal targets only when
    /// the largest allowed size changed.
    fn update_resolution(&mut self) {
        let output = self.targets.output;
        let capacity = self.capacity_for(output);
        if capacity != self.targets.capacity {
            self.targets = FrameTargets::new(&self.device, capacity, output);
            self.history_valid = false;
        }
        self.percentage = match &self.settings.dynamic_resolution {
            Some(dynamic) if self.profiler.supported() => self.resolution.update(self.profiler.latest_frame(), dynamic),
            Some(dynamic) => resolution::clamp_percentage(dynamic.max_percentage),
            None => {
                // Dynamic resolution starts from the fixed percentage when it is switched on.
                self.resolution = ResolutionController::new(self.settings.screen_percentage);
                resolution::clamp_percentage(self.settings.screen_percentage)
            }
        };
        self.targets.set_internal_size(resolution::internal_size(output, self.percentage));
        let s = &mut self.settings;
        if s.gi_lite_fallback && s.gi_method != GiMethod::LumenLite && self.lite_fallback.observe(self.percentage, s.dynamic_resolution.as_ref()) {
            log::info!("dynamic resolution stays at its minimum: switching the GI to Lumen Lite");
            s.gi_method = GiMethod::LumenLite;
        }
    }

    /// The final gather this frame: the probe view runs Lumen Lite whatever the method.
    fn uses_lumen_lite(&self) -> bool {
        self.settings.global_illumination && (self.settings.gi_method == GiMethod::LumenLite || self.settings.debug_view == DebugView::LumenLiteProbes)
    }

    /// GPU statistics from a recent frame.
    pub fn stats(&self) -> RenderStats {
        self.stats.latest
    }

    /// Geometry streaming statistics.
    pub fn streaming_stats(&self) -> StreamingStats {
        self.scene.streaming_stats()
    }

    /// Image texture streaming statistics.
    pub fn texture_streaming_stats(&self) -> TextureStreamingStats {
        self.svt.stats()
    }

    /// Smoothed GPU time per pass in milliseconds (empty without timestamp-query support).
    pub fn gpu_timings(&self) -> &[(String, f32)] {
        self.profiler.results()
    }

    pub fn gpu_time_ms(&self) -> Option<f32> {
        self.profiler.supported().then(|| self.profiler.total_ms())
    }

    /// Moving instances in the latest frame: instances moved, distance-field bricks recomposed
    /// and still queued.
    pub fn dynamic_stats(&self) -> DynamicStats {
        let field = self.lumen.dynamic_stats();
        DynamicStats {
            moved: self.motion.changes.moved.len() as u32,
            shadow_moves: self.scene.shadow_moves().len() as u32,
            bricks: field.bricks,
            queued_bricks: field.queued,
            sdf_levels: field.levels,
        }
    }

    /// Whether some mesh of the scene has voxel LODs ([`RenderSettings::voxel_lods`]).
    pub fn has_voxels(&self) -> bool {
        self.scene.has_voxels()
    }

    /// Nanite tessellation statistics of a recent frame (`None` without displaced instances).
    pub fn tessellation_stats(&self) -> Option<crate::TessellationStats> {
        self.culling.tessellation(&self.scene).map(|t| t.stats())
    }

    /// Local lights rendered into virtual shadow maps this frame.
    pub fn shadow_mapped_lights(&self) -> usize {
        self.vsm.local.active_count()
    }

    /// Factor by which the LOD threshold is currently raised because the visible geometry
    /// would not fit the culling lists (1.0 = no reduction).
    ///
    /// When a list overflows, the GPU drops arbitrary clusters, a different set every frame:
    /// geometry flickers. Instead the renderer coarsens the whole view just enough to fit
    /// (with hysteresis, so it does not oscillate) and relaxes again when there is room.
    pub fn lod_budget_scale(&self) -> f32 {
        self.lod_budget_scale
    }

    fn update_lod_budget(&mut self) {
        let st = self.stats.latest;
        let capacity = self.main_view.capacity;
        let fill = (st.clusters as f32 / capacity.clusters as f32).max(st.pages as f32 / capacity.pages as f32);
        if st.overflow > 0 || fill > 0.9 {
            // Projected error scales with 1/distance, so cluster counts fall roughly with the
            // square of the threshold: step proportionally to the excess.
            let step = (fill / 0.8).sqrt().clamp(1.1, 2.0);
            self.lod_budget_scale = (self.lod_budget_scale * step).min(64.0);
        } else if fill < 0.6 && self.lod_budget_scale > 1.0 {
            self.lod_budget_scale = (self.lod_budget_scale * 0.97).max(1.0);
        }
    }

    /// Asks which instance is under output pixel `pixel` (editor picking). The answer arrives
    /// asynchronously, a frame or two later, through [`Renderer::take_pick`] with the same `id`.
    pub fn request_pick(&mut self, id: u32, pixel: (u32, u32)) {
        self.picker.request(id, pixel);
    }

    /// The latest finished pick (once).
    pub fn take_pick(&mut self) -> Option<PickResult> {
        self.picker.take()
    }

    /// Instances outlined as the editor's selection (empty: no outline pass).
    pub fn set_selection(&mut self, selection: &[InstanceHandle]) {
        self.selection.set(selection);
    }

    /// Forgets temporal history (TSR, GI, exposure), e.g. after a camera cut.
    pub fn reset_history(&mut self) {
        self.history_valid = false;
    }

    /// Renders `scene` from `camera` into `output` (which must match the output format).
    pub fn render(&mut self, scene: &Scene, camera: &Camera, output: &wgpu::TextureView, delta_time: f32) {
        self.update_resolution();
        let frame = self.frame_index + 1;
        self.motion.incremental = self.settings.per_instance_invalidation;
        self.motion.update(scene);
        let changes = &self.motion.changes;
        self.scene.tessellation_enabled = self.settings.tessellation;
        let sync = self.scene.sync(&self.device, &self.queue, scene, changes, self.motion.previous(), frame);
        if sync.geometry_changed {
            self.history_valid = false;
        }
        // Assembly parts each take a page batch in every culling view.
        let parts = self.scene.assemblies.part_capacity();
        self.main_view.reserve_batches(&self.device, MAIN_CAPACITY.batches + parts);
        self.vsm.reserve_batches(&self.device, parts);
        // New geometry or instances invalidate every cached shadow page, and so does switching
        // voxel LODs (other casters). Moves and streaming only invalidate the pages under the
        // affected instances (done by the VSM on the GPU); growing scene bounds change the shadow
        // clipmap's depth range.
        let switched = self.settings.voxel_lods != self.voxel_lods_cached || self.settings.tessellation != self.tessellation_cached;
        self.scene_changed = changes.structural || sync.bounds_grew || switched;
        self.voxel_lods_cached = self.settings.voxel_lods;
        self.tessellation_cached = self.settings.tessellation;
        self.textures.sync(&self.device, &self.queue, scene);
        let substrate = self.settings.substrate;
        self.extended.sync(&self.device, &self.queue, scene, &self.scene, substrate, self.targets.capacity);
        self.culling.prepare_tessellation(&self.device, scene, &self.scene, &self.textures, &self.samplers.material, self.extended.parameter_buffer());
        self.targets.prepare_tessellation(&self.device, self.scene.has_displacement());
        // Editing a displacement (magnitude, a displacing graph or its parameters) reshapes
        // static surfaces: their cached shadow pages are stale.
        let displacement = (self.scene.displacement_table_hash, self.extended.displacement_key);
        if displacement != self.displacement_cached && self.scene.has_displacement() {
            self.scene_changed = true;
        }
        self.displacement_cached = displacement;
        let tessellation = self.scene.has_displacement();
        let debug_views = self.settings.debug_view != DebugView::Lit;
        self.material.sync(&self.device, self.scene.uv_extensions, self.scene.skins.skinned_instances, tessellation, debug_views, &self.extended, substrate);
        let slab_lighting = self.extended.lighting.then_some(&self.extended.lighting_layout);
        self.lighting.sync_substrate(&self.device, slab_lighting);
        self.megalights.sync_substrate(&self.device, self.extended.lighting.then_some(&self.extended.megalights_layout));
        self.rvt.sync(&self.device, &self.queue, scene, self.settings.virtual_texturing, frame);
        // Texture feedback is per rendered pixel (internal size); the froxel volume and the
        // screen targets of Lumen, clouds and translucency are sized for the capacity, so a
        // dynamic resolution change keeps them.
        let size = (self.targets.width, self.targets.height);
        let capacity = self.targets.capacity;
        self.svt.sync(&self.device, &self.queue, scene, self.settings.texture_streaming, frame, size);
        self.lumen.update(&self.device, &self.queue, scene, &self.motion.changes, camera.position, frame, &self.targets);
        if self.uses_lumen_lite() {
            let lite = self.lumen_lite.get_or_insert_with(|| LumenLite::new(&self.device));
            lite.update(&self.queue, scene, camera.position, frame);
        }
        let view_proj = camera.projection(self.targets.output.0 as f32 / self.targets.output.1 as f32) * camera.view_matrix();
        let lights = scene.lights();
        self.vsm.assign_local_lights(&LocalFrame {
            lights: &lights[..lights.len().min(MAX_LIGHTS)],
            eye: camera.position,
            planes: frustum_planes(&view_proj),
            enabled: self.settings.shadows && self.settings.local_light_shadow_maps,
            scene_changed: self.scene_changed,
            moving_shadow_maps: self.settings.moving_light_shadow_maps && !self.settings.mega_lights,
        });
        self.lights.upload(&self.queue, scene, self.vsm.local.light_slots());
        let clouds = self.settings.clouds.then_some(&scene.clouds);
        self.clouds.update(&self.device, &self.queue, clouds, camera.position, self.time + delta_time, &self.targets);
        let lit = matches!(self.settings.debug_view, DebugView::Lit);
        let shutter = if lit { self.settings.motion_blur } else { 0.0 };
        let dof = self.settings.depth_of_field.as_ref().filter(|_| lit);
        self.post_fx.update(&self.device, &self.queue, shutter, dof, camera.fov_y, self.targets.output);
        let local_exposure = self.settings.local_exposure.as_ref().filter(|_| lit);
        self.local_exposure.update(&self.device, &self.queue, local_exposure, self.settings.exposure_compensation, self.targets.output);
        self.tsr.update(&self.queue, self.targets.output);
        self.hzb.update(&self.queue, &self.targets);
        self.fog.update(&self.device, &self.queue, &scene.fog, self.settings.volumetric_fog, self.time + delta_time, capacity);
        // MegaLights' debug views sample even while the classic loop shades.
        let megalights_debug = matches!(self.settings.debug_view, DebugView::MegaLightsSamples | DebugView::MegaLightsDenoising);
        let megalights = (self.settings.mega_lights || megalights_debug) && self.lights.count > 0;
        self.megalights.update(&self.device, capacity, megalights);
        if self.stats.poll(self.scene.instance_count) {
            self.update_lod_budget();
            self.scene.assemblies.update_budget(self.stats.latest.assembly_parts_requested);
        }
        if let Some(tessellation) = &mut self.culling.tessellation {
            tessellation.poll();
        }
        // Volumes and particles are drawn over lit radiance and in their own debug view.
        let effects = matches!(self.settings.debug_view, DebugView::Lit | DebugView::VolumesParticles);
        self.effects.update(
            &self.device,
            &self.queue,
            &EffectsFrame {
                scene,
                eye: camera.position,
                view_proj: &view_proj,
                volumes: effects && self.settings.volumes,
                particles: effects && self.settings.particles,
                delta_time,
                internal: size,
                capacity,
            },
        );
        self.time += delta_time;

        let current = (self.frame_index & 1) as usize;
        let unjittered_view_proj = self.write_uniforms(scene, camera, delta_time);
        // Translucent surfaces are drawn over lit radiance only, not over debug views.
        let translucency = self.settings.translucency && self.settings.debug_view == DebugView::Lit;
        let layers = self.settings.translucency_layers;
        let changes = &self.motion.changes;
        self.translucency.update(&self.device, &self.queue, scene, changes, camera.position, &unjittered_view_proj, capacity, translucency, layers);

        let lumen_lite = self.lumen_lite.as_ref().filter(|_| self.uses_lumen_lite());
        let device = self.device.clone();
        let mut rec = Recorder::new(&device, &mut self.profiler, "frame");
        self.sky.record(&mut rec, &self.frame_uniform, &self.samplers.linear, scene.ground_albedo);
        self.effects.record_lights(&mut rec, &self.lights);
        self.lights.record(&mut rec, &self.frame_uniform);

        // Virtual geometry: assemblies the main view needs as parts get part instances (for
        // every view), then two-pass occlusion culling into the visibility buffer.
        self.scene.assemblies.record(&mut rec, &self.main_view, &self.scene.instances);
        let (culling, view, gpu_scene, targets) = (&self.culling, &self.main_view, &self.scene, &self.targets);
        if self.settings.occlusion_culling {
            culling.cull(&mut rec, view, gpu_scene, CullMode::Early, Some(&targets.hzb));
            culling.raster_visibility(&mut rec, view, gpu_scene, targets, false);
            self.hzb.record(&mut rec, targets);
            culling.cull(&mut rec, view, gpu_scene, CullMode::Late, Some(&targets.hzb));
            culling.raster_visibility(&mut rec, view, gpu_scene, targets, true);
        } else {
            culling.cull(&mut rec, view, gpu_scene, CullMode::Single, None);
            culling.raster_visibility(&mut rec, view, gpu_scene, targets, false);
        }

        // Shading.
        let resolve_group = self.extended.resolve_group(&device);
        self.material.record(
            &mut rec,
            &self.frame_uniform,
            &self.main_view,
            &self.scene,
            &self.targets,
            &self.textures,
            &self.rvt,
            &self.svt,
            self.translucency.water_uniform(),
            &self.samplers,
            resolve_group.as_ref(),
        );
        self.picker.record(&mut rec, &self.queue, &self.frame_uniform, &self.targets, &self.main_view.visible_clusters, self.scene.assemblies.owners());
        self.rvt.record(&mut rec, &self.textures, &self.samplers.material, self.settings.virtual_texturing);
        if self.settings.shadows {
            self.vsm.record(&mut rec, &self.culling, &self.scene, &self.frame_uniform, &self.targets, &self.lights);
        }
        self.lumen.record_distance_field(&mut rec, &self.textures, &self.scene.materials, &self.samplers);
        let lite_inputs = LiteInputs { frame: &self.frame_uniform, targets: &self.targets, sky: &self.sky, samplers: &self.samplers, exposure: &self.exposure.buffer };
        if self.settings.global_illumination {
            let previous_radiance = &self.targets.history[1 - current];
            self.lumen.record(
                &mut rec,
                &self.frame_uniform,
                &self.targets,
                &self.sky,
                &self.samplers,
                previous_radiance,
                current,
                &self.lights,
                &self.exposure.buffer,
                &CardInputs { vsm: &self.vsm, textures: &self.textures, materials: &self.scene.materials },
                lumen_lite.is_none(),
            );
            if let Some(lite) = lumen_lite {
                lite.record(&mut rec, &lite_inputs, &self.lumen);
            }
            self.gi.record(&mut rec, &self.frame_uniform, &self.targets, &self.samplers, current);
        }
        if self.clouds.enabled() {
            let inputs = CloudInputs { frame: &self.frame_uniform, sky: &self.sky, lut_sampler: &self.samplers.linear, depth: &self.targets.depth };
            self.clouds.record(&mut rec, &inputs, current);
        }
        if self.fog.enabled() {
            let inputs = FogInputs {
                frame: &self.frame_uniform,
                vsm: &self.vsm,
                sky: &self.sky,
                lights: &self.lights,
                clouds: &self.clouds,
                lumen: &self.lumen,
                samplers: &self.samplers,
            };
            self.fog.record(&mut rec, &inputs, current);
        }
        let reflection = if self.settings.global_illumination { self.lumen.reflection(current).unwrap_or(&self.black) } else { &self.black };
        let megalights_slabs = self.extended.megalights_group(&device);
        let megalights_inputs = MegaLightsInputs {
            frame: &self.frame_uniform,
            targets: &self.targets,
            lights: &self.lights,
            vsm: &self.vsm,
            lumen: &self.lumen,
            samplers: &self.samplers,
            substrate: megalights_slabs.as_ref(),
        };
        if megalights {
            self.megalights.record(&mut rec, &megalights_inputs, current);
        }
        let lighting_slabs = self.extended.lighting_group(&device);
        let lighting = LightingInputs {
            frame: &self.frame_uniform,
            targets: &self.targets,
            sky: &self.sky,
            samplers: &self.samplers,
            vsm: &self.vsm,
            gi: &self.targets.gi_history[current],
            reflection,
            lumen: &self.lumen,
            lights: &self.lights,
            clouds: &self.clouds,
            cloud_texture: self.clouds.output(current),
            fog: &self.fog,
            megalights: self.megalights.output(current).filter(|_| self.settings.mega_lights),
            black: &self.black,
            substrate: lighting_slabs.as_ref(),
        };
        self.lighting.record(&mut rec, &lighting);
        if self.extended.subsurface && lit {
            self.subsurface.record(&mut rec, &self.frame_uniform, &self.targets, &self.extended);
        }
        if megalights && megalights_debug {
            self.megalights.record_debug(&mut rec, &megalights_inputs, current);
        }
        if self.settings.debug_view == DebugView::SurfaceCards {
            self.lumen.record_debug(&mut rec, &self.frame_uniform, &self.targets, &self.samplers);
        }
        if let Some(lite) = lumen_lite.filter(|_| self.settings.debug_view == DebugView::LumenLiteProbes) {
            lite.record_debug(&mut rec, &lite_inputs, &self.targets.gi_history[current]);
        }
        if self.settings.debug_view == DebugView::Invalidation {
            self.invalidation_debug.record(&mut rec, &self.frame_uniform, &self.targets, &self.vsm, &self.lumen, &self.samplers);
        }
        self.effects.record_before_translucency(&mut rec, &lighting, &self.scene, &self.translucency);
        self.translucency.record(&mut rec, &lighting, &self.scene, &self.textures);
        self.effects.record_after_translucency(&mut rec, &lighting);

        // Post-processing, at output resolution from here on.
        self.tsr.record(&mut rec, &self.frame_uniform, &self.targets, &self.samplers, &self.exposure.buffer, current);
        // Camera effects read the TSR result (which stays untouched as next frame's history).
        let resolved = self.post_fx.record(&mut rec, &self.frame_uniform, &self.targets, &self.samplers, &self.targets.history[current]);
        self.exposure.record(&mut rec, &self.frame_uniform, resolved, &self.samplers);
        // Metering sees the scene as it is; bloom and tone mapping the locally exposed image (a
        // highlight brought down glows less).
        let resolved = self.local_exposure.record(&mut rec, &self.exposure.buffer, &self.samplers, resolved);
        self.bloom.record(&mut rec, resolved, &self.targets, &self.samplers);
        let bloom = if self.targets.bloom_up.len() > 1 { &self.targets.bloom_up[0] } else { &self.targets.bloom_down[0] };
        self.tonemap.record(&mut rec, &self.frame_uniform, resolved, bloom, &self.exposure.buffer, &self.samplers, output);
        let owners = self.scene.assemblies.owners();
        self.selection.record(&mut rec, &self.queue, &self.frame_uniform, &self.targets, &self.main_view.visible_clusters, owners, output);

        let mut states = vec![&self.main_view.state];
        states.extend(self.vsm.level_states());
        let assembly_parts = self.scene.assemblies.cull_dispatch().map(|(counter, _)| counter);
        let read_stats = self.stats.record(&mut rec.encoder, &states, [&self.vsm.cache.lists, self.vsm.draws()], assembly_parts);
        let read_feedback = self.scene.record_feedback(&mut rec.encoder, frame);
        let read_texture_feedback = self.svt.record_feedback(&mut rec.encoder, frame);
        let read_tessellation = self.culling.tessellation(&self.scene).is_some_and(|t| t.record_readback(&mut rec.encoder));

        let (commands, map_profiler) = rec.finish();
        self.queue.submit([commands]);
        self.profiler.after_submit(map_profiler);
        self.picker.after_submit();
        if read_stats {
            self.stats.map();
        }
        if read_feedback {
            self.scene.map_feedback();
        }
        if read_texture_feedback {
            self.svt.map_feedback();
        }
        if let Some(tessellation) = self.culling.tessellation(&self.scene) {
            tessellation.after_submit(read_tessellation);
        }
        let _ = self.device.poll(wgpu::PollType::Poll);

        self.previous_view_proj = Some(unjittered_view_proj);
        self.previous_viewport = viewport(size);
        self.frame_index = self.frame_index.wrapping_add(1);
        self.history_valid = true;
    }

    /// Fills the frame and culling-view uniforms. Returns the unjittered view-projection.
    fn write_uniforms(&mut self, scene: &Scene, camera: &Camera, delta_time: f32) -> Mat4 {
        let s = &self.settings;
        let (w, h) = (self.targets.width as f32, self.targets.height as f32);
        let (output_w, output_h) = (self.targets.output.0 as f32, self.targets.output.1 as f32);
        // Internal pixels per output pixel. The image always spans the output's aspect ratio
        // (internal pixels may be a hair non-square from rounding).
        let ratio = Vec2::new(w / output_w, h / output_h);
        let view = camera.view_matrix();
        let projection = camera.projection(output_w / output_h);
        let unjittered = projection * view;

        // Sub-pixel jitter (Halton 2, 3) gives TSR new sample positions every frame, over a
        // cycle long enough for every output pixel to be sampled (see `jitter_phase_count`).
        let jitter = if s.taa && matches!(s.debug_view, DebugView::Lit | DebugView::IndirectLight | DebugView::Reflections) {
            resolution::jitter_offset(self.frame_index, ratio.x.min(ratio.y)) * 2.0 / Vec2::new(w, h)
        } else {
            Vec2::ZERO
        };
        let view_proj = Mat4::from_translation(jitter.extend(0.0)) * projection * view;

        let sun = scene.sun.direction.normalize();

        let mut frame_flags = 0;
        for (enabled, flag) in [
            (s.shadows, flags::SHADOWS),
            (s.global_illumination, flags::GI),
            (s.taa, flags::TAA),
            (s.auto_exposure, flags::AUTO_EXPOSURE),
            (s.aerial_perspective, flags::FOG),
            (self.history_valid, flags::HISTORY_VALID),
            // Last frame's: the uniforms are written before translucency finds the camera's water
            // body this frame (exposure adapts over several frames anyway).
            (self.translucency.camera_in_water(), flags::UNDERWATER),
            (s.mega_lights && self.lights.count > 0, flags::MEGALIGHTS),
            (self.effects.active(), flags::REACTIVE),
        ] {
            if enabled {
                frame_flags |= flag;
            }
        }

        let frame = FrameUniform {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            unjittered_view_proj: unjittered,
            prev_unjittered_view_proj: self.previous_view_proj.filter(|_| self.history_valid).unwrap_or(unjittered),
            view,
            camera_position: camera.position.extend(camera.near),
            viewport: Vec4::new(w, h, 1.0 / w, 1.0 / h),
            jitter: jitter.extend(0.0).extend(0.0),
            sun_direction: sun.extend(scene.sun.angular_radius),
            sun_illuminance: scene.sun.illuminance.extend(1.0),
            ground_albedo: scene.ground_albedo.extend(0.0),
            frame_index: self.frame_index,
            debug_view: s.debug_view as u32,
            flags: frame_flags,
            light_count: self.lights.count,
            exposure_compensation: s.exposure_compensation,
            gi_intensity: s.gi_intensity,
            bloom_intensity: s.bloom_intensity,
            time: self.time,
            misc: Vec4::new(delta_time, s.aerial_perspective_scale, s.gi_ray_length, scene.stars),
            prev_viewport: if self.history_valid { self.previous_viewport } else { viewport(self.internal_size()) },
            upscale: Vec4::new(ratio.x, ratio.y, output_w, output_h),
        };
        self.queue.write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(&frame));

        let instance_count = self.scene.cull_instance_count();
        let main = CullViewUniform {
            view_proj,
            view,
            planes: frustum_planes(&view_proj),
            position: camera.position.extend(camera.near),
            viewport: Vec4::new(w, h, 1.0 / w, 1.0 / h),
            // Nanite's error threshold is in output pixels, as UE's TSR-based mesh LOD bias
            // does: TSR resolves sub-internal-pixel geometry over the jitter cycle, and the
            // geometry (and its streaming) stays the same whatever the dynamic resolution
            // picks. The scale stays in internal pixels for the software raster heuristic and
            // the HZB, which are about the pixels actually rasterised.
            lod_scale: camera.lod_scale(h),
            lod_threshold: s.lod_threshold * self.lod_budget_scale * ratio.y,
            orthographic: 0,
            instance_count,
            work_capacity: self.main_view.capacity.pages,
            cluster_capacity: self.main_view.capacity.clusters,
            hzb_mips: self.targets.hzb_mip_count(),
            frame_index: self.frame_index + 1,
            sw_raster_threshold: s.sw_raster_threshold,
            kind: 0,
            batch_capacity: self.main_view.capacity.batches,
            masked_capacity: self.main_view.capacity.masked,
            vsm: Vec4::ZERO,
            flags: self.cull_flags(),
            _pad: [0; 3],
        };
        self.queue.write_buffer(&self.main_view.uniform, 0, bytemuck::bytes_of(&main));
        if let Some(tessellation) = self.culling.tessellation(&self.scene) {
            // Dice rate in internal pixels, like the software raster heuristic.
            let view = tessellation.view(s.dice_rate, camera.lod_scale(h), camera.near, camera.position, [self.time, self.time - delta_time]);
            self.scene.write_tessellation_view(&self.queue, &view);
        }
        self.scene.assemblies.write_params(&self.queue);

        // Wind: positions at this and the previous frame's time (motion vectors). Its "too small
        // to see" distance is judged in output pixels, like the LOD.
        let pixel_angle = 1.0 / camera.lod_scale(output_h);
        let wind = WindUniform::new(&scene.wind, if s.wind { 1.0 } else { 0.0 }, self.time, self.time - delta_time, camera.position, pixel_angle);
        self.scene.write_wind(&self.queue, &wind);
        // Starting or stopping the wind moves vegetation between the shadow pages' static and
        // dynamic layers: the VSM re-renders the pages under it (per instance).
        // Animated skeletal meshes are dynamic casters too (see skin.rs).
        let wind_moves_shadows = (wind.is_active() && self.scene.wind_moves_shadows()) || self.scene.skins.moving_shadows;

        let (scene_min, scene_max) = self.scene.bounds();
        self.vsm.update(
            &self.queue,
            &VsmFrame {
                camera,
                // Shadow page resolution follows the internal resolution's upper bound: with a
                // fixed screen percentage shadow detail and cost scale with it (as UE's view
                // rect based page selection), while dynamic resolution leaves the page set alone
                // instead of invalidating cached pages and changing shadow sharpness whenever
                // the percentage moves.
                viewport_height: self.targets.capacity.1 as f32,
                sun_direction: sun,
                scene_min,
                scene_max,
                instance_count,
                frame: self.frame_index + 1,
                levels: s.shadow_levels,
                lod_threshold: s.shadow_lod_threshold,
                resolution_bias: s.shadow_resolution_bias,
                local_resolution_bias: s.shadow_resolution_bias + s.local_shadow_resolution_bias,
                sw_raster_threshold: s.sw_raster_threshold,
                enabled: s.shadows,
                scene_changed: self.scene_changed,
                geometry_streamed: self.scene.streaming_stats().changed(),
                wind: wind_moves_shadows,
                animated_displacement: self.scene.animated_displacement_shadows(),
                moves: self.scene.shadow_moves(),
                instances_changed: !self.motion.changes.moved.is_empty(),
                volume_distance: if self.fog.enabled() { scene.fog.distance } else { 0.0 },
                cull_flags: self.cull_flags(),
            },
        );
        unjittered
    }

    /// `CullViewUniform::flags` of every culling view.
    fn cull_flags(&self) -> u32 {
        if self.settings.voxel_lods {
            view_flags::VOXELS
        } else {
            0
        }
    }
}

/// Moving instances in a frame ([`Renderer::dynamic_stats`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicStats {
    /// Instances moved (or given another mobility).
    pub moved: u32,
    /// Shadow casters among them whose bounds moved (their shadow pages are invalidated).
    pub shadow_moves: u32,
    /// Global distance field bricks recomposed, and still queued for the next frames.
    pub bricks: u32,
    pub queued_bricks: u32,
    /// Global distance field clipmap levels rebuilt in full (camera movement, structural
    /// changes; with [`RenderSettings::per_instance_invalidation`] off, every move).
    pub sdf_levels: u32,
}

/// A viewport uniform (width, height, 1 / width, 1 / height).
fn viewport((width, height): (u32, u32)) -> Vec4 {
    let (w, h) = (width as f32, height as f32);
    Vec4::new(w, h, 1.0 / w, 1.0 / h)
}
