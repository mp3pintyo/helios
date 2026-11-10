//! Runtime virtual texture for the landscape (see `shaders/rvt.wgsl`).
//!
//! The terrain material (layer textures blended by slope, height and noise) is rendered into
//! 128² pages of a 128k² virtual texture on demand. Pixels request the page at the mip their
//! footprint needs; the generic [`PageCache`] maps requests to a 24×24-page physical atlas,
//! keeps pages across frames and evicts the least recently used.

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use helios_geometry::SdfKind;
use wgpu::{ShaderStages, TextureFormat as F, TextureUsages as U};

use crate::page_cache::{dirty_list_start, CacheClient, PageCache, DIRTY_ARGS_OFFSET};
use crate::profiler::Recorder;
use crate::scene::{MaterialKind, Scene};
use crate::shader::create_module;
use crate::textures::TextureLibrary;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, uniform_buffer};
use crate::pipelines::Lazy;

const VIRTUAL_PAGES: u32 = 1024;
const PAGE_TEXELS: u32 = 128;
const BORDER: u32 = 4;
const TILES_PER_ROW: u32 = 24;
const RETAIN_FRAMES: u32 = 240;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RvtParams {
    world_min: [f32; 2],
    world_size: f32,
    enabled: u32,
    virtual_pages: u32,
    mips: u32,
    page_texels: u32,
    border: u32,
    tiles_per_row: u32,
    atlas_size: f32,
    frame: u32,
    layers: u32,
    mip_offsets: [[u32; 4]; 4],
    hf_origin: [f32; 2],
    hf_cell: f32,
    hf_scale: f32,
    hf_translation: [f32; 3],
    lod_bias: f32,
    hf_resolution: [u32; 2],
    _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<RvtParams>() == 160);

pub struct RuntimeVirtualTexture {
    pub cache: PageCache,
    pub params: wgpu::Buffer,
    pub albedo: wgpu::TextureView,
    pub detail: wgpu::TextureView,
    albedo_storage: wgpu::TextureView,
    detail_storage: wgpu::TextureView,
    heightfield: wgpu::TextureView,
    layout: wgpu::BindGroupLayout,
    render: Lazy<wgpu::ComputePipeline>,
    state: RvtParams,
    /// Scene versions (geometry, material, texture) the setup was derived from.
    versions: (u64, u64, u64),
}

impl RuntimeVirtualTexture {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let mips = VIRTUAL_PAGES.trailing_zeros() + 1;
        let mut mip_offsets = [[0u32; 4]; 4];
        let mut entries = 0u32;
        for m in 0..mips {
            mip_offsets[(m / 4) as usize][(m % 4) as usize] = entries;
            let pages = (VIRTUAL_PAGES >> m).max(1);
            entries += pages * pages;
        }
        let physical = TILES_PER_ROW * TILES_PER_ROW;
        let cache = PageCache::new(device, "rvt cache", CacheClient::Static, entries, physical);
        cache.clear(queue);

        let span = PAGE_TEXELS + 2 * BORDER;
        let atlas_size = TILES_PER_ROW * span;
        let atlas = |label: &str, srgb: bool| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: atlas_size, height: atlas_size, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: F::Rgba8Unorm,
                usage: U::TEXTURE_BINDING | U::STORAGE_BINDING,
                view_formats: if srgb { &[F::Rgba8UnormSrgb] } else { &[] },
            })
        };
        let albedo = atlas("rvt albedo", true);
        let detail = atlas("rvt detail", false);

        let module = create_module(device, "rvt_render.wgsl");
        let layout = bind_group_layout(
            device,
            "rvt render",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::storage(true)),
                (2, bind::storage(true)),
                (3, bind::texture(false)),
                (4, bind::texture_array(true)),
                (5, bind::texture_array(true)),
                (6, bind::sampler()),
                (7, bind::storage_texture(F::Rgba8Unorm)),
                (8, bind::storage_texture(F::Rgba8Unorm)),
            ],
        );
        let render = compute_pipeline(
            device,
            "rvt render",
            &module,
            "render_pages",
            &layout,
            &[("DIRTY_START", dirty_list_start(physical) as f64)],
        );
        let state = RvtParams {
            world_min: [0.0; 2],
            world_size: 1.0,
            enabled: 0,
            virtual_pages: VIRTUAL_PAGES,
            mips,
            page_texels: PAGE_TEXELS,
            border: BORDER,
            tiles_per_row: TILES_PER_ROW,
            atlas_size: atlas_size as f32,
            frame: 0,
            layers: u32::MAX,
            mip_offsets,
            hf_origin: [0.0; 2],
            hf_cell: 1.0,
            hf_scale: 1.0,
            hf_translation: [0.0; 3],
            lod_bias: 0.0,
            hf_resolution: [1, 1],
            _pad: [0; 2],
        };
        Self {
            cache,
            params: uniform_buffer(device, "rvt params", std::mem::size_of::<RvtParams>() as u64),
            albedo: albedo.create_view(&wgpu::TextureViewDescriptor {
                format: Some(F::Rgba8UnormSrgb),
                usage: Some(U::TEXTURE_BINDING),
                ..Default::default()
            }),
            detail: detail.create_view(&Default::default()),
            albedo_storage: albedo.create_view(&Default::default()),
            detail_storage: detail.create_view(&Default::default()),
            heightfield: heightfield_texture(device, queue, &[0.0], [1, 1]),
            layout,
            render,
            state,
            versions: (u64::MAX, u64::MAX, u64::MAX),
        }
    }

    pub fn enabled(&self) -> bool {
        self.state.enabled != 0
    }

    /// Finds the landscape (first `Terrain` instance whose mesh has a heightfield) and sets the
    /// virtual texture up over its extent. Flushes the cache when anything relevant changed.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, enabled: bool, frame: u32) {
        let versions = (scene.geometry_version, scene.material_version, scene.texture_version);
        if versions != self.versions {
            self.versions = versions;
            self.state.enabled = 0;
            for instance in &scene.instances {
                let material = &scene.materials[instance.material.0 as usize];
                let mesh = &scene.meshes[instance.mesh.0 as usize];
                let Some(sdf) = mesh.sdf.as_ref().filter(|s| s.kind == SdfKind::Heightfield) else { continue };
                if material.kind != MaterialKind::Terrain {
                    continue;
                }
                let (scale, _, translation) = instance.transform.to_scale_rotation_translation();
                let min = translation + mesh.aabb.min * scale;
                let max = translation + mesh.aabb.max * scale;
                let size = (max.x - min.x).max(max.z - min.z);
                let heights: Vec<f32> = (0..sdf.resolution[2])
                    .flat_map(|z| (0..sdf.resolution[0]).map(move |x| (x, z)))
                    .map(|(x, z)| sdf.height(x, z))
                    .collect();
                self.heightfield = heightfield_texture(device, queue, &heights, [sdf.resolution[0], sdf.resolution[2]]);
                let layer = |t: Option<crate::scene::TextureHandle>| t.map_or(255, |t| t.0.min(254));
                self.state.layers = material.layers.iter().enumerate().fold(0, |acc, (i, &t)| acc | layer(t) << (i * 8));
                self.state.world_min = Vec2::new(min.x, min.z).into();
                self.state.world_size = size;
                self.state.hf_origin = [sdf.origin.x, sdf.origin.z];
                self.state.hf_cell = sdf.voxel_size;
                self.state.hf_scale = scale.x;
                self.state.hf_translation = translation.into();
                self.state.hf_resolution = [sdf.resolution[0], sdf.resolution[2]];
                self.state.enabled = 1;
                break;
            }
            self.cache.clear(queue);
        }
        let mut state = self.state;
        state.enabled &= enabled as u32;
        state.frame = frame;
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&state));
        let mut cache = self.cache.default_params(frame, RETAIN_FRAMES, groups_per_page());
        let top = self.state.mip_offsets[((self.state.mips - 1) / 4) as usize][((self.state.mips - 1) % 4) as usize];
        cache.pinned = top;
        self.cache.write_params(queue, &cache);
    }

    /// Allocates pages requested during the G-buffer pass and renders the new ones.
    pub fn record(&self, rec: &mut Recorder, textures: &TextureLibrary, material_sampler: &wgpu::Sampler, active: bool) {
        if !self.enabled() || !active {
            return;
        }
        self.cache.record(rec);
        use wgpu::BindingResource::{Sampler, TextureView};
        let group = bind_group(
            rec.device,
            "rvt render",
            &self.layout,
            &[
                (0, self.params.as_entire_binding()),
                (1, self.cache.entries.as_entire_binding()),
                (2, self.cache.lists.as_entire_binding()),
                (3, TextureView(&self.heightfield)),
                (4, TextureView(&textures.albedo)),
                (5, TextureView(&textures.data)),
                (6, Sampler(material_sampler)),
                (7, TextureView(&self.albedo_storage)),
                (8, TextureView(&self.detail_storage)),
            ],
        );
        let mut pass = rec.compute_pass("rvt pages");
        pass.set_pipeline(&self.render);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups_indirect(&self.cache.lists, DIRTY_ARGS_OFFSET);
    }
}

fn groups_per_page() -> u32 {
    let tiles = (PAGE_TEXELS + 2 * BORDER) / 8;
    tiles * tiles
}

fn heightfield_texture(device: &wgpu::Device, queue: &wgpu::Queue, heights: &[f32], size: [u32; 2]) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rvt heightfield"),
        size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: F::R32Float,
        usage: U::TEXTURE_BINDING | U::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(heights),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size[0] * 4), rows_per_image: Some(size[1]) },
        wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
    );
    texture.create_view(&Default::default())
}
