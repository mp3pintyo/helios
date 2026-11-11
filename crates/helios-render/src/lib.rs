//! # helios-render
//!
//! A WebGPU (wgpu) renderer built around the architecture of Unreal Engine 5:
//!
//! * **Virtual geometry** – cluster DAGs from [`helios_geometry`], streamed page by page into a
//!   GPU pool on demand; GPU-driven instance → page → cluster culling with a per-cluster LOD cut
//!   and two-pass HZB occlusion; tiny triangles go to a compute **software rasteriser**, the rest
//!   to one indirect hardware draw, both into a **visibility buffer**. Foliage gets **voxel LODs**
//!   and instanced **assemblies** of parts resolved during culling (Nanite Foliage).
//! * **Materials** – GPU-synthesised texture arrays, **streaming virtual textures** for imported
//!   images at native resolution, per-shading-model tile classification and resolve, and a
//!   **runtime virtual texture** for the landscape.
//! * **Shadows** – **virtual shadow maps**: a paged, cached 16k² clipmap for the sun, and cube
//!   (point) or perspective (spot) virtual shadow maps for the most important local lights.
//! * **Wind** – vegetation animated by World Position Offset in every rasteriser, with padded
//!   culling bounds and per-page shadow invalidation.
//! * **Tessellation** – materials displace their surfaces at render time ([`Displacement`]):
//!   displaced clusters are split and diced on the GPU into pixel-sized micro-triangles, crack-free
//!   and geomorphed, in the main view and the shadow pages (Nanite tessellation).
//! * **Skeletal meshes** – virtual geometry deformed by joint palettes (linear blend skinning in
//!   every rasteriser and the G-buffer resolve), with conservative posed culling bounds and
//!   dynamic shadow casting ([`Scene::add_skin`]).
//! * **Global illumination** – Lumen-style software ray tracing: global distance field, voxel
//!   surface cache, radiance cache, screen-probe final gather (or Lumen Lite's world-space
//!   irradiance field on weak GPUs) and traced reflections.
//! * **Lights** – clustered point/spot lights with virtual shadow maps (distance-field soft
//!   shadows beyond the shadow-map budget); Hillaire sky.
//! * **Clouds** – ray-marched volumetric cumulus with multiple scattering and ground shadows.
//! * **Volumetric fog** – froxel-based lit height fog with shadowed light shafts.
//! * **Translucency** – a forward pass for glass, water and alpha-blended surfaces after the
//!   deferred lighting: refraction of the lit scene, absorption, screen-space and distance-field
//!   reflections, fog. Sunlight through glass takes the panes' colour (a map of the panes seen
//!   from the sun).
//! * **Volumes and particles** – simulated smoke and fire or procedural mist in ray-marched
//!   voxel boxes with a lighting cache; GPU particle emitters; both blended order-independently
//!   around the translucent surfaces.
//! * **Temporal super resolution** – the scene renders at a screen percentage of the output
//!   (fixed, or picked per frame for a GPU time budget) and TSR reconstructs the output.
//! * **Post** – depth of field, motion blur, eye adaptation, bloom and ACES filmic tone mapping;
//!   per-pass GPU timings.
//!
//! Entry points: build a [`Scene`], create a [`Renderer`], call [`Renderer::render`].

mod alpha_mask;
mod assembly;
mod avboit;
mod camera;
mod cards;
mod clouds;
mod fog;
mod fog_scattering;
mod photo;
mod context;
mod dynamic_debug;
mod vsm_overview;
mod vsm_moments;
mod volume_shadow;
mod effects;
mod glass_light;
mod glass_shadow;
mod gpu_scene;
mod gpu_types;
mod lights;
mod lumen;
mod lumen_displaced;
mod lumen_dynamic;
mod lumen_light_grid;
mod lumen_lite;
mod lumen_offload;
mod material_graphs;
mod media;
mod megalights;
mod megalights_volume;
mod motion;
mod near_field;
mod oit;
mod page_cache;
mod page_decode;
pub mod page_source;
mod particles;
mod passes;
mod picking;
pub mod pipelines;
mod profiler;
mod renderer;
mod resolution;
mod rvt;
mod scene;
mod scene_edit;
mod selection;
mod shader;
pub use shader::{allow_half_precision, half_precision};
mod skin;
mod stats;
mod streaming;
mod substrate;
mod substrate_gpu;
mod subsurface;
mod svt;
mod targets;
mod tessellation;
mod textures;
mod translucency;
mod util;
mod volumes;
mod vsm;
mod vsm_anchor;
mod vsm_local;
mod vsm_motion;
mod water;
mod wind;
mod world_cache;

pub use camera::Camera;
pub use context::{GpuContext, ContextError};
pub use fog_scattering::FogScattering;
pub use photo::PhotoProgress;
pub use passes::local_exposure::LocalExposure;
pub use passes::post_fx::DepthOfField;
pub use picking::PickResult;
pub use renderer::{DebugView, DynamicStats, RenderSettings, Renderer, RendererConfig, DEFAULT_SHADOW_DISTANT_LEVEL, DEFAULT_SHADOW_REFRESH_BUDGET};
pub use lumen_lite::{GiMethod, LiteFallback};
pub use world_cache::{WorldCacheMode, WorldCacheStats};
pub use lumen_offload::{run_bake_task, set_bake_offload, BakeOffload, BakePoll};
pub use resolution::{DynamicResolution, MAX_SCREEN_PERCENTAGE, MIN_SCREEN_PERCENTAGE};
pub use scene::{
    AssemblyPart, CloudLayer, Displacement, FogVolume, Image, ImageTexture, InstanceHandle, Light, LightHandle, LightKind, Material, MaterialHandle, MaterialKind,
    MeshHandle, Mobility, Scene, Sun, TextureDesc, TextureHandle, TextureMapping, TexturePattern, TextureSource,
    UvTransform, Water, Wrap,
};
pub use page_source::{PageLocation, PageRead, PageSource, ReadCompletion};
pub use particles::{Burst, Collision, ParticleEmitter, ParticleRender, SpawnShape, Sprite};
pub use volumes::{FluidSimulation, ProceduralVolume, VolumeEmitter, VolumeSource};
pub use material_graphs::{MaterialGraphHandle, ParameterValue};
pub use substrate::{features as slab_features, Slab};
/// Material graphs: data model, validation, WGSL code generation, presets.
pub use helios_material as material_graph;
pub use scene_edit::transformed_bounds;
pub use skin::SkinHandle;
pub use stats::RenderStats;
pub use streaming::StreamingStats;
pub use svt::TextureStreamingStats;
pub use tessellation::TessellationStats;
pub use vsm_anchor::{AnchorStats, MovingLightShadows};
pub use vsm_local::MovingLightStats;
pub use wind::{Wind, WindResponse};

pub use helios_geometry as geometry;
pub use wgpu;
