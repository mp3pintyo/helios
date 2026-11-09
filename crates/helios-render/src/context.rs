//! Adapter and device creation with the limits the renderer needs.

use std::fmt;

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
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: available.max_storage_buffer_binding_size,
            max_buffer_size: available.max_buffer_size,
            max_storage_textures_per_shader_stage: available.max_storage_textures_per_shader_stage.min(8),
            max_texture_dimension_2d: available.max_texture_dimension_2d.min(16384),
            ..wgpu::Limits::default()
        };
        // Optional: GPU pass timings; BC formats for the streamed texture tiles (desktop GPUs;
        // without them the tiles stay uncompressed); clip distances, which let shadow page draws
        // clip to their tile (vsm_draw_clipped.wgsl).
        let required_features = adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::CLIP_DISTANCES);
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
        log::info!("GPU: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
        Ok(Self { instance, adapter, device, queue })
    }
}
