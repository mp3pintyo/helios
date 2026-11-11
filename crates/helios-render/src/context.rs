//! Adapter and device creation with the limits the renderer needs.

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

/// The most sampled textures per shader stage the device asks for (see [`GpuContext::new`]).
static SAMPLED_TEXTURES: AtomicU32 = AtomicU32::new(24);

/// Sampled textures per shader stage the renderer needs: the lighting pass binds 18 (the G-buffer,
/// shadows, sky, GI, clouds, fog, MegaLights' result, sunlight and local light through glass).
/// More enable optional variants: 19 the Substrate slab lighting and AVBOIT (otherwise weighted
/// blended OIT). WebGPU guarantees only 16; desktop adapters offer 48 or more.
pub const MIN_SAMPLED_TEXTURES: u32 = 18;

/// Owns the wgpu instance, adapter, device and queue.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

#[derive(Debug)]
pub enum ContextError {
    NoAdapter(String),
    Device(String),
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(e) => write!(f, "no suitable GPU adapter (WebGPU required): {e}"),
            Self::Device(e) => write!(f, "failed to create GPU device: {e}"),
        }
    }
}

impl std::error::Error for ContextError {}

impl GpuContext {
    /// Asks devices created from now on for at most `count` sampled textures per shader stage
    /// (default 24; at least WebGPU's default 16), as if the adapter offered no more: for testing
    /// the fallbacks of the passes that bind more than 16 (`--max-sampled-textures 16`,
    /// `?max-sampled-textures=16`).
    pub fn limit_sampled_textures(count: u32) {
        SAMPLED_TEXTURES.store(count.clamp(16, 24), Ordering::Relaxed);
    }

    pub fn create_instance() -> wgpu::Instance {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        if cfg!(target_arch = "wasm32") {
            desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        }
        wgpu::Instance::new(desc)
    }

    /// Requests a high-performance adapter (compatible with `surface`, if given) and a device.
    pub async fn new(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Self, ContextError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| ContextError::NoAdapter(e.to_string()))?;

        let available = adapter.limits();
        let sampled = available.max_sampled_textures_per_shader_stage.min(SAMPLED_TEXTURES.load(Ordering::Relaxed));
        if sampled < MIN_SAMPLED_TEXTURES {
            return Err(ContextError::Device(format!(
                "the renderer samples {MIN_SAMPLED_TEXTURES} textures per shader stage, this GPU offers {sampled} (maxSampledTexturesPerShaderStage)"
            )));
        }
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: available.max_storage_buffer_binding_size,
            max_buffer_size: available.max_buffer_size,
            max_storage_textures_per_shader_stage: available.max_storage_textures_per_shader_stage.min(8),
            // The lighting pass's Substrate variant samples a 17th texture, the translucency
            // shading 18 and AVBOIT 19 (desktop adapters offer 48 or more: Chromium 48, Firefox
            // 64 on an RTX 3090). With fewer those passes take a variant that fits (see
            // substrate_gpu.rs, translucency.rs).
            max_sampled_textures_per_shader_stage: sampled,
            max_texture_dimension_2d: available.max_texture_dimension_2d.min(16384),
            ..wgpu::Limits::default()
        };
        // Optional: GPU pass timings; BC formats for the streamed texture tiles (desktop GPUs;
        // without them the tiles stay uncompressed); clip distances, which let shadow page draws
        // clip to their tile (vsm_draw_clipped.wgsl).
        // f16 shaders: passes written in half.wgsl's `hf` types compute in half precision.
        let half = if crate::shader::half_precision_allowed() { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() };
        let required_features = adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::CLIP_DISTANCES | half);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("helios device"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|e| ContextError::Device(e.to_string()))?;

        let info = adapter.get_info();
        let half = if device.features().contains(wgpu::Features::SHADER_F16) { ", f16 shaders" } else { "" };
        log::info!("GPU: {} ({:?}, {:?}{half})", info.name, info.device_type, info.backend);
        Ok(Self { instance, adapter, device, queue })
    }
}
