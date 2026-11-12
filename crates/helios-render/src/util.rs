//! Small, explicit helpers that keep pass code focused on *what* it binds, not boilerplate.

pub use crate::pipelines::{Lazy, MakeFn};
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

/// A bind group over `entries`. Passes ask for their bind groups every frame (≈ 150 a frame in
/// the valley), almost always over the same resources, so the group is reused when the layout
/// and every resource match one asked for within the last [`BIND_GROUP_KEEP_FRAMES`] frames.
/// Firefox (D3D12) frees a dropped bind group's descriptors only once the page's garbage
/// collector finalizes the JS object, and new groups every frame filled the 1 M-descriptor heap
/// in ≈ 800 frames ("Not enough memory left", then a lost device).
pub fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    entries: &[(u32, wgpu::BindingResource)],
) -> wgpu::BindGroup {
    let key = bind_group_cache::key(layout, entries);
    if let Some(group) = key.as_ref().and_then(bind_group_cache::get) {
        return group;
    }
    let descriptor_entries: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &descriptor_entries });
    if let Some(key) = key {
        bind_group_cache::insert(key, group.clone());
    }
    group
}

/// Frames a cached bind group survives without being asked for (it keeps its resources alive).
pub const BIND_GROUP_KEEP_FRAMES: u64 = 120;

/// Ends a frame for the bind group cache: drops the groups no frame asked for lately.
pub fn bind_group_cache_end_frame() {
    bind_group_cache::end_frame();
}

mod bind_group_cache {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::num::NonZeroU64;

    #[derive(Clone, PartialEq, Eq, Hash)]
    pub(super) enum Resource {
        Buffer(wgpu::Buffer, u64, Option<NonZeroU64>),
        Buffers(Vec<(wgpu::Buffer, u64, Option<NonZeroU64>)>),
        Sampler(wgpu::Sampler),
        Samplers(Vec<wgpu::Sampler>),
        View(wgpu::TextureView),
        Views(Vec<wgpu::TextureView>),
    }

    #[derive(Clone, PartialEq, Eq, Hash)]
    pub(super) struct Key {
        layout: wgpu::BindGroupLayout,
        entries: Vec<(u32, Resource)>,
    }

    struct Cache {
        frame: u64,
        groups: HashMap<Key, (wgpu::BindGroup, u64)>,
    }

    thread_local! {
        static CACHE: RefCell<Cache> = RefCell::new(Cache { frame: 0, groups: HashMap::new() });
    }

    fn buffer(b: &wgpu::BufferBinding) -> (wgpu::Buffer, u64, Option<NonZeroU64>) {
        (b.buffer.clone(), b.offset, b.size)
    }

    /// The cache key, or `None` for resources it doesn't track (acceleration structures, …).
    pub(super) fn key(layout: &wgpu::BindGroupLayout, entries: &[(u32, wgpu::BindingResource)]) -> Option<Key> {
        use wgpu::BindingResource as R;
        let entries = entries
            .iter()
            .map(|(binding, resource)| {
                let resource = match resource {
                    R::Buffer(b) => Resource::Buffer(b.buffer.clone(), b.offset, b.size),
                    R::BufferArray(bs) => Resource::Buffers(bs.iter().map(buffer).collect()),
                    R::Sampler(s) => Resource::Sampler((*s).clone()),
                    R::SamplerArray(ss) => Resource::Samplers(ss.iter().map(|s| (*s).clone()).collect()),
                    R::TextureView(v) => Resource::View((*v).clone()),
                    R::TextureViewArray(vs) => Resource::Views(vs.iter().map(|v| (*v).clone()).collect()),
                    _ => return None,
                };
                Some((*binding, resource))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Key { layout: layout.clone(), entries })
    }

    pub(super) fn get(key: &Key) -> Option<wgpu::BindGroup> {
        CACHE.with_borrow_mut(|cache| {
            let frame = cache.frame;
            cache.groups.get_mut(key).map(|(group, used)| {
                *used = frame;
                group.clone()
            })
        })
    }

    pub(super) fn insert(key: Key, group: wgpu::BindGroup) {
        CACHE.with_borrow_mut(|cache| {
            let frame = cache.frame;
            cache.groups.insert(key, (group, frame));
        });
    }

    pub(super) fn end_frame() {
        CACHE.with_borrow_mut(|cache| {
            cache.frame += 1;
            if cache.frame % 30 == 0 {
                let oldest = cache.frame.saturating_sub(super::BIND_GROUP_KEEP_FRAMES);
                cache.groups.retain(|_, (_, used)| *used >= oldest);
            }
        });
    }
}

pub fn pipeline_layout(device: &wgpu::Device, label: &str, layouts: &[&wgpu::BindGroupLayout]) -> wgpu::PipelineLayout {
    let layouts: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    })
}

/// A compute pipeline over one bind group, made in the background in the browser (see
/// [`crate::pipelines`]): compiled with the rest of its [`crate::pipelines::Group`].
pub fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layout: &wgpu::BindGroupLayout,
    constants: &[(&str, f64)],
) -> Lazy<wgpu::ComputePipeline> {
    compute_pipeline_with(device, label, module, entry_point, &[layout], constants)
}

/// A compute pipeline over several bind groups (group `i` = `layouts[i]`), made in the
/// background in the browser.
pub fn compute_pipeline_with(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layouts: &[&wgpu::BindGroupLayout],
    constants: &[(&str, f64)],
) -> Lazy<wgpu::ComputePipeline> {
    Lazy::background(compute_pipeline_maker(device, label, module, entry_point, layouts, constants))
}

/// [`compute_pipeline`], compiled when a frame first uses it (or once the scene is known to need
/// it: [`Lazy::prefetch`]).
pub fn lazy_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layout: &wgpu::BindGroupLayout,
    constants: &[(&str, f64)],
) -> Lazy<wgpu::ComputePipeline> {
    Lazy::new(compute_pipeline_maker(device, label, module, entry_point, &[layout], constants))
}

/// [`lazy_compute_pipeline`] over several bind groups (group `i` = `layouts[i]`).
pub fn lazy_compute_pipeline_with(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layouts: &[&wgpu::BindGroupLayout],
    constants: &[(&str, f64)],
) -> Lazy<wgpu::ComputePipeline> {
    Lazy::new(compute_pipeline_maker(device, label, module, entry_point, layouts, constants))
}

/// The constructor of a compute pipeline: owns what the descriptor refers to, so it can run
/// later (and twice in the browser: once to warm the compile, once to make the pipeline).
fn compute_pipeline_maker(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layouts: &[&wgpu::BindGroupLayout],
    constants: &[(&str, f64)],
) -> impl MakeFn<wgpu::ComputePipeline> {
    let (device, module) = (device.clone(), module.clone());
    let layouts: Vec<wgpu::BindGroupLayout> = layouts.iter().map(|&l| l.clone()).collect();
    let (label, entry_point) = (label.to_string(), entry_point.to_string());
    let constants: Vec<(String, f64)> = constants.iter().map(|&(name, value)| (name.to_string(), value)).collect();
    move || {
        let constants: Vec<(&str, f64)> = constants.iter().map(|(name, value)| (name.as_str(), *value)).collect();
        let layouts: Vec<&wgpu::BindGroupLayout> = layouts.iter().collect();
        let layout = pipeline_layout(&device, &label, &layouts);
        timed(&label, || {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(&entry_point),
                compilation_options: wgpu::PipelineCompilationOptions { constants: &constants, zero_initialize_workgroup_memory: true },
                cache: None,
            })
        })
    }
}

/// Creates a render pipeline through [`timed`] (labelled by its descriptor).
pub fn render_pipeline(device: &wgpu::Device, descriptor: &wgpu::RenderPipelineDescriptor) -> wgpu::RenderPipeline {
    timed(descriptor.label.unwrap_or("render pipeline"), || device.create_render_pipeline(descriptor))
}

/// Runs `make` (a pipeline's creation). Natively with `HELIOS_PIPELINE_TIMES` set it logs how long
/// it took: the shader compile. With `WGPU_BACKEND=dx12`, `WGPU_DX12_COMPILER=dxc` and Firefox's
/// `dxcompiler.dll` first on the `PATH` that is the compile a Firefox visitor waits for (Firefox
/// translates WGSL with naga to HLSL and compiles it with DXC, one pipeline at a time), measured
/// one by one without a browser.
pub fn timed<T>(label: &str, make: impl FnOnce() -> T) -> T {
    crate::pipelines::note_created(label);
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| std::env::var_os("HELIOS_PIPELINE_TIMES").is_some()) {
            let start = std::time::Instant::now();
            let value = make();
            log::info!("pipeline time {:.1} ms: {label}", start.elapsed().as_secs_f64() * 1000.0);
            return value;
        }
    }
    make()
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
