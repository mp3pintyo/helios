//! # helios-render
//!
//! A WebGPU (wgpu) renderer built around the architecture of Unreal Engine 5:
//!
//! * **Virtual geometry** – cluster DAGs from [`helios_geometry`], streamed page by page into a
//!   GPU pool on demand; GPU-driven instance → page → cluster culling with a per-cluster LOD cut
//!   and two-pass HZB occlusion; tiny triangles go to a compute **software rasteriser**, the rest
//!   to one indirect hardware draw, both into a **visibility buffer**.
//! * **Materials** – GPU-synthesised texture arrays, **streaming virtual textures** for imported
//!   images at native resolution, per-shading-model tile classification and resolve, and a
//!   **runtime virtual texture** for the landscape.
//! * **Shadows** – **virtual shadow maps**: a paged, cached 16k² clipmap for the sun, and cube
//!   (point) or perspective (spot) virtual shadow maps for the most important local lights.
//! * **Wind** – vegetation animated by World Position Offset in every rasteriser, with padded
//!   culling bounds and per-page shadow invalidation.
//! * **Global illumination** – Lumen-style software ray tracing: global distance field, voxel
//!   surface cache, radiance cache, screen-probe final gather and traced reflections.
//! * **Lights** – clustered point/spot lights with virtual shadow maps (distance-field soft
//!   shadows beyond the shadow-map budget); Hillaire sky.
//! * **Clouds** – ray-marched volumetric cumulus with multiple scattering and ground shadows.
//! * **Volumetric fog** – froxel-based lit height fog with shadowed light shafts.
//! * **Translucency** – a forward pass for glass, water and alpha-blended surfaces after the
//!   deferred lighting: refraction of the lit scene, absorption, screen-space and distance-field
//!   reflections, fog.
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
mod camera;
mod cards;
mod clouds;
mod fog;
mod context;
mod effects;
mod gpu_scene;
mod gpu_types;
mod lights;
mod lumen;
mod media;
mod near_field;
mod oit;
mod page_cache;
pub mod page_source;
mod particles;
mod passes;
mod profiler;
mod renderer;
mod resolution;
mod rvt;
mod scene;
mod shader;
mod stats;
mod streaming;
mod svt;
mod targets;
mod textures;
mod translucency;
mod util;
mod volumes;
mod vsm;
mod vsm_local;
mod water;
mod wind;

pub use camera::Camera;
pub use context::{GpuContext, ContextError};
pub use passes::local_exposure::LocalExposure;
pub use passes::post_fx::DepthOfField;
pub use renderer::{DebugView, RenderSettings, Renderer, RendererConfig};
pub use resolution::{DynamicResolution, MAX_SCREEN_PERCENTAGE, MIN_SCREEN_PERCENTAGE};
pub use scene::{
    CloudLayer, FogVolume, Image, ImageTexture, InstanceHandle, Light, LightHandle, LightKind, Material, MaterialHandle, MaterialKind,
    MeshHandle, Scene, Sun, TextureDesc, TextureHandle, TextureMapping, TexturePattern, TextureSource,
    UvTransform, Water, Wrap,
};
pub use page_source::{PageLocation, PageRead, PageSource, ReadCompletion};
pub use particles::{Burst, Collision, ParticleEmitter, ParticleRender, SpawnShape, Sprite};
pub use volumes::{FluidSimulation, ProceduralVolume, VolumeEmitter, VolumeSource};
pub use stats::RenderStats;
pub use streaming::StreamingStats;
pub use svt::TextureStreamingStats;
pub use wind::{Wind, WindResponse};

pub use helios_geometry as geometry;
pub use wgpu;
