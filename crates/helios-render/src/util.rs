//! Small, explicit helpers that keep pass code focused on *what* it binds, not boilerplate.

use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType, ShaderStages,
    StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

/// Binding type constructors.
pub mod bind {
    use super::*;

    pub fn uniform() -> BindingType {
        BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }
    }
    pub fn storage(read_only: bool) -> BindingType {
        BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        }
    }
    pub fn texture(filterable: bool) -> BindingType {
        BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        }
    }
    pub fn texture_array(filterable: bool) -> BindingType {
        BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable },
            view_dimension: TextureViewDimension::D2Array,
            multisampled: false,
        }
    }
    pub fn texture_3d(filterable: bool) -> BindingType {
        BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable },
            view_dimension: TextureViewDimension::D3,
            multisampled: false,
        }
    }
    pub fn storage_texture_3d(format: TextureFormat) -> BindingType {
        BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format, view_dimension: TextureViewDimension::D3 }
    }
    pub fn texture_uint() -> BindingType {
        BindingType::Texture {
            sample_type: TextureSampleType::Uint,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        }
    }
    pub fn texture_3d_uint() -> BindingType {
        BindingType::Texture {
            sample_type: TextureSampleType::Uint,
            view_dimension: TextureViewDimension::D3,
            multisampled: false,
        }
    }
    pub fn texture_depth() -> BindingType {
        BindingType::Texture {
            sample_type: TextureSampleType::Depth,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        }
    }
    pub fn storage_texture(format: TextureFormat) -> BindingType {
        BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: TextureViewDimension::D2,
        }
    }
    pub fn sampler() -> BindingType {
        BindingType::Sampler(SamplerBindingType::Filtering)
    }
    pub fn sampler_comparison() -> BindingType {
        BindingType::Sampler(SamplerBindingType::Comparison)
    }
}

pub fn bind_group_layout(
    device: &wgpu::Device,
    label: &str,
    visibility: ShaderStages,
    entries: &[(u32, BindingType)],
) -> wgpu::BindGroupLayout {
    let entries: Vec<BindGroupLayoutEntry> = entries
        .iter()
        .map(|&(binding, ty)| BindGroupLayoutEntry { binding, visibility, ty, count: None })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &entries })
}

pub fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    entries: &[(u32, wgpu::BindingResource)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
}

pub fn pipeline_layout(device: &wgpu::Device, label: &str, layouts: &[&wgpu::BindGroupLayout]) -> wgpu::PipelineLayout {
    let layouts: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    })
}

pub fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layout: &wgpu::BindGroupLayout,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    compute_pipeline_with(device, label, module, entry_point, &[layout], constants)
}

/// A compute pipeline over several bind groups (group `i` = `layouts[i]`).
pub fn compute_pipeline_with(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layouts: &[&wgpu::BindGroupLayout],
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    let layout = pipeline_layout(device, label, layouts);
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    })
}

pub struct TextureSpec<'a> {
    pub label: &'a str,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub mips: u32,
    pub format: TextureFormat,
    pub usage: wgpu::TextureUsages,
}

pub fn texture(device: &wgpu::Device, spec: TextureSpec) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(spec.label),
        size: wgpu::Extent3d { width: spec.width.max(1), height: spec.height.max(1), depth_or_array_layers: spec.layers },
        mip_level_count: spec.mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: spec.format,
        usage: spec.usage,
        view_formats: &[],
    })
}

/// A view of a single mip level (and optionally a single layer).
pub fn mip_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        base_mip_level: mip,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

pub fn storage_buffer(device: &wgpu::Device, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

/// Creates a storage buffer initialised with `data` (padded to at least 16 bytes).
pub fn storage_buffer_init(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, data: &[u8]) -> wgpu::Buffer {
    let buffer = storage_buffer(device, label, data.len() as u64, wgpu::BufferUsages::empty());
    if !data.is_empty() {
        queue.write_buffer(&buffer, 0, data);
    }
    buffer
}

pub fn uniform_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn workgroups(size: u32, group: u32) -> u32 {
    size.div_ceil(group)
}

/// Mip count for a full chain down to 1x1.
pub fn mip_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

#[cfg(not(target_arch = "wasm32"))]
type Make<T> = Box<dyn Fn() -> T + Send + Sync>;
#[cfg(target_arch = "wasm32")]
type Make<T> = Box<dyn Fn() -> T>;

/// A value made on first use, for pipelines a frame may never need: permutations for content a
/// scene may not have (skinned instances, alpha-masked or voxel clusters, a translucent shading
/// model). Every pipeline is a shader compile, and in the browser those queue up in the GPU
/// process before the first frame: 0.1–1.5 s each in Firefox, which compiles one at a time.
/// Dereferences to the value, so `pass.set_pipeline(&self.pipeline)` reads as before.
pub struct Lazy<T> {
    value: std::sync::OnceLock<T>,
    make: Make<T>,
}

impl<T> Lazy<T> {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(make: impl Fn() -> T + Send + Sync + 'static) -> Self {
        Self { value: std::sync::OnceLock::new(), make: Box::new(make) }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new(make: impl Fn() -> T + 'static) -> Self {
        Self { value: std::sync::OnceLock::new(), make: Box::new(make) }
    }
}

impl<T> std::ops::Deref for Lazy<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value.get_or_init(|| (self.make)())
    }
}

/// [`compute_pipeline`], compiled on first use.
pub fn lazy_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layout: &wgpu::BindGroupLayout,
    constants: &[(&str, f64)],
) -> Lazy<wgpu::ComputePipeline> {
    let (device, module, layout) = (device.clone(), module.clone(), layout.clone());
    let (label, entry_point) = (label.to_string(), entry_point.to_string());
    let constants: Vec<(String, f64)> = constants.iter().map(|&(name, value)| (name.to_string(), value)).collect();
    Lazy::new(move || {
        let constants: Vec<(&str, f64)> = constants.iter().map(|(name, value)| (name.as_str(), *value)).collect();
        compute_pipeline(&device, &label, &module, &entry_point, &layout, &constants)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn lazy_makes_its_value_once_on_first_use() {
        let made = Arc::new(AtomicU32::new(0));
        let counter = made.clone();
        let lazy = Lazy::new(move || {
            counter.fetch_add(1, Ordering::Relaxed);
            7
        });
        assert_eq!(made.load(Ordering::Relaxed), 0);
        assert_eq!(*lazy + *lazy, 14);
        assert_eq!(made.load(Ordering::Relaxed), 1);
    }
}
