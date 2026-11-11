//! Material texture arrays, synthesised on the GPU from procedural recipes or resampled from
//! images.
//!
//! WebGPU has no bindless textures, so every material texture is a layer of two shared
//! 2D arrays (one bind slot each, any number of materials):
//!
//! * `albedo` — RGBA8 sRGB colour (alpha: coverage, 1 for procedural textures),
//! * `data`   — tangent-space normal XY, roughness, and height (procedural) or metallic
//!   (image layers; see `MATERIAL_FLAG_IMAGE_TEXTURE`).
//!
//! Layers are generated once when the scene's texture set changes, including a full mip chain
//! (albedo averaged in linear light, normal variance folded into roughness). Image layers are
//! area-resampled to the layer size first, so a material's base colour, normal and
//! metallic-roughness images (which share UVs) occupy a single layer; an occlusion image takes
//! the albedo alpha (see [`ImageTexture`]).

use bytemuck::{Pod, Zeroable};
use wgpu::{ShaderStages, TextureFormat as F, TextureUsages as U};

use crate::scene::{Image, ImageTexture, Scene, TextureDesc, TexturePattern, TextureSource};
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, lazy_compute_pipeline, mip_count, uniform_buffer, workgroups};
use crate::pipelines::Lazy;

pub const TEXTURE_SIZE: u32 = 512;
/// [`TexturePattern`] variants (`PATTERN_*` in texgen.wgsl).
const PATTERNS: u32 = TexturePattern::Cliff as u32 + 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Recipe {
    pattern: u32,
    seed: u32,
    scale: f32,
    layer: u32,
    color_a: [f32; 4],
    color_b: [f32; 4],
    params: [f32; 4],
}

pub struct TextureLibrary {
    pub albedo: wgpu::TextureView,
    pub data: wgpu::TextureView,
    pub layer_count: u32,
    version: u64,
    /// `generate` specialised per [`TexturePattern`] (`PATTERN`), each compiled when a scene
    /// first uses the pattern (in the browser in the background, before the arrays are built).
    generate: Vec<Lazy<wgpu::ComputePipeline>>,
    normals: Lazy<wgpu::ComputePipeline>,
    pack: Lazy<wgpu::ComputePipeline>,
    /// `pack` with the occlusion image in the albedo alpha.
    pack_occlusion: Lazy<wgpu::ComputePipeline>,
    down_albedo: Lazy<wgpu::ComputePipeline>,
    down_data: Lazy<wgpu::ComputePipeline>,
    layouts: [wgpu::BindGroupLayout; 5],
    image_sampler: wgpu::Sampler,
    /// 1×1 stand-ins for missing images: white base colour, flat normal, roughness = metallic = 1,
    /// no occlusion.
    neutral: [wgpu::Texture; 4],
}

/// The four images of an image layer, in `pack_images` binding order.
const IMAGE_SLOTS: [&str; 4] = ["base colour", "normal", "metallic-roughness", "occlusion"];

impl TextureLibrary {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let module = create_module(device, "texgen.wgsl");
        let c = ShaderStages::COMPUTE;
        let layouts = [
            bind_group_layout(
                device,
                "texgen",
                c,
                &[(0, bind::uniform()), (1, bind::storage_texture(F::Rgba8Unorm)), (2, bind::storage_texture(F::Rg32Float))],
            ),
            bind_group_layout(device, "texgen normals", c, &[(0, bind::uniform()), (3, bind::texture(false)), (4, bind::storage_texture(F::Rgba8Unorm))]),
            bind_group_layout(device, "texgen mip albedo", c, &[(1, bind::storage_texture(F::Rgba8Unorm)), (5, bind::texture(false))]),
            bind_group_layout(device, "texgen mip data", c, &[(4, bind::storage_texture(F::Rgba8Unorm)), (5, bind::texture(false))]),
            bind_group_layout(
                device,
                "texgen pack images",
                c,
                &[
                    (1, bind::storage_texture(F::Rgba8Unorm)),
                    (4, bind::storage_texture(F::Rgba8Unorm)),
                    (6, bind::texture(true)),
                    (7, bind::texture(true)),
                    (8, bind::texture(true)),
                    (9, bind::sampler()),
                    (10, bind::texture(true)),
                ],
            ),
        ];
        let neutral_pixel = |label, rgba: [u8; 4]| {
            upload_image(device, queue, &Image { width: 1, height: 1, pixels: rgba.to_vec() }, label).expect("1x1 texture")
        };
        let mut library = Self {
            generate: (0..PATTERNS).map(|p| lazy_compute_pipeline(device, "texgen", &module, "generate", &layouts[0], &[("PATTERN", p as f64)])).collect(),
            normals: compute_pipeline(device, "texgen normals", &module, "normals", &layouts[1], &[]),
            pack: compute_pipeline(device, "texgen pack images", &module, "pack_images", &layouts[4], &[]),
            pack_occlusion: compute_pipeline(device, "texgen pack images", &module, "pack_images", &layouts[4], &[("PACK_OCCLUSION", 1.0)]),
            down_albedo: compute_pipeline(device, "texgen mip albedo", &module, "downsample_albedo", &layouts[2], &[]),
            down_data: compute_pipeline(device, "texgen mip data", &module, "downsample_data", &layouts[3], &[]),
            layouts,
            image_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("texgen images"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            neutral: [
                neutral_pixel("neutral base colour", [255; 4]),
                neutral_pixel("neutral normal", [128, 128, 255, 255]),
                neutral_pixel("neutral metallic-roughness", [255; 4]),
                neutral_pixel("neutral occlusion", [255; 4]),
            ],
            albedo: dummy(device),
            data: dummy(device),
            layer_count: 0,
            version: u64::MAX,
        };
        library.build(device, queue, &[]);
        library
    }

    /// Queues the pattern pipelines of `scene`'s procedural textures (a scene about to be
    /// switched to); builds nothing.
    pub fn prefetch(&self, scene: &Scene) {
        for source in &scene.textures {
            if let TextureSource::Procedural(desc) = source {
                self.generate[desc.pattern as usize].prefetch();
            }
        }
    }

    /// Regenerates the arrays if the scene's textures changed. Returns true if it did.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) -> bool {
        if self.version == scene.texture_version {
            return false;
        }
        // In the browser the patterns' pipelines compile in the background first; the arrays are
        // built once they are in (the first frame waits for them).
        let mut compiling = false;
        for source in &scene.textures {
            if let TextureSource::Procedural(desc) = source {
                let pipeline = &self.generate[desc.pattern as usize];
                pipeline.prefetch();
                compiling |= crate::pipelines::deferring() && !pipeline.is_made();
            }
        }
        if compiling {
            return false;
        }
        self.version = scene.texture_version;
        self.build(device, queue, &scene.textures);
        true
    }

    fn build(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, textures: &[TextureSource]) {
        let max_layers = device.limits().max_texture_array_layers as usize;
        if textures.len() > max_layers {
            log::warn!("{} textures exceed the device limit of {max_layers} array layers; the rest are not generated", textures.len());
        }
        let textures = &textures[..textures.len().min(max_layers)];
        let layers = textures.len().max(1) as u32;
        let mips = mip_count(TEXTURE_SIZE, TEXTURE_SIZE);
        let make = |label: &str, srgb_view: bool| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: TEXTURE_SIZE, height: TEXTURE_SIZE, depth_or_array_layers: layers },
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: F::Rgba8Unorm,
                usage: U::TEXTURE_BINDING | U::STORAGE_BINDING,
                view_formats: if srgb_view { &[F::Rgba8UnormSrgb] } else { &[] },
            })
        };
        let albedo = make("material albedo", true);
        let data = make("material data", false);
        let height = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texgen height"),
            size: wgpu::Extent3d { width: TEXTURE_SIZE, height: TEXTURE_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: F::Rg32Float,
            usage: U::TEXTURE_BINDING | U::STORAGE_BINDING,
            view_formats: &[],
        });
        let height_view = height.create_view(&Default::default());
        let sub = |t: &wgpu::Texture, layer: u32, mip: u32| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };

        for (layer, source) in textures.iter().enumerate() {
            let layer = layer as u32;
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("texgen") });
            {
                use wgpu::BindingResource::TextureView;
                let groups = workgroups(TEXTURE_SIZE, 8);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("texgen"), timestamp_writes: None });
                let albedo0 = sub(&albedo, layer, 0);
                let data0 = sub(&data, layer, 0);
                match source {
                    TextureSource::Procedural(desc) => {
                        let uniform = uniform_buffer(device, "texgen recipe", std::mem::size_of::<Recipe>() as u64);
                        queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&recipe(desc, layer)));
                        let g = bind_group(
                            device,
                            "texgen",
                            &self.layouts[0],
                            &[(0, uniform.as_entire_binding()), (1, TextureView(&albedo0)), (2, TextureView(&height_view))],
                        );
                        pass.set_pipeline(&self.generate[desc.pattern as usize]);
                        pass.set_bind_group(0, &g, &[]);
                        pass.dispatch_workgroups(groups, groups, 1);
                        let g = bind_group(
                            device,
                            "texgen normals",
                            &self.layouts[1],
                            &[(0, uniform.as_entire_binding()), (3, TextureView(&height_view)), (4, TextureView(&data0))],
                        );
                        pass.set_pipeline(&self.normals);
                        pass.set_bind_group(0, &g, &[]);
                        pass.dispatch_workgroups(groups, groups, 1);
                    }
                    TextureSource::Images(images) => {
                        let views = self.image_views(device, queue, images);
                        let g = bind_group(
                            device,
                            "texgen pack images",
                            &self.layouts[4],
                            &[
                                (1, TextureView(&albedo0)),
                                (4, TextureView(&data0)),
                                (6, TextureView(&views[0])),
                                (7, TextureView(&views[1])),
                                (8, TextureView(&views[2])),
                                (9, wgpu::BindingResource::Sampler(&self.image_sampler)),
                                (10, TextureView(&views[3])),
                            ],
                        );
                        pass.set_pipeline(if images.occlusion.is_some() { &self.pack_occlusion } else { &self.pack });
                        pass.set_bind_group(0, &g, &[]);
                        pass.dispatch_workgroups(groups, groups, 1);
                    }
                }
                for mip in 1..mips {
                    let size = (TEXTURE_SIZE >> mip).max(1);
                    let groups = workgroups(size, 8);
                    for (pipeline, layout, texture) in [(&self.down_albedo, &self.layouts[2], &albedo), (&self.down_data, &self.layouts[3], &data)] {
                        let source = sub(texture, layer, mip - 1);
                        let target = sub(texture, layer, mip);
                        let binding = if std::ptr::eq(texture, &albedo) { 1 } else { 4 };
                        let g = bind_group(device, "texgen mip", layout, &[(binding, TextureView(&target)), (5, TextureView(&source))]);
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, &g, &[]);
                        pass.dispatch_workgroups(groups, groups, 1);
                    }
                }
            }
            queue.submit([encoder.finish()]);
        }
        if !textures.is_empty() {
            let images = textures.iter().filter(|t| matches!(t, TextureSource::Images(_))).count();
            log::info!("generated {} material textures ({images} from images; {}² with mips)", textures.len(), TEXTURE_SIZE);
        }

        self.albedo = albedo.create_view(&wgpu::TextureViewDescriptor {
            format: Some(F::Rgba8UnormSrgb),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(U::TEXTURE_BINDING),
            ..Default::default()
        });
        self.data = data.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        self.layer_count = layers;
    }

    /// Uploads an image layer's sources (neutral stand-ins for missing or unusable ones) and
    /// returns their views; the base colour view decodes sRGB so resampling averages linear light.
    fn image_views(&self, device: &wgpu::Device, queue: &wgpu::Queue, images: &ImageTexture) -> [wgpu::TextureView; 4] {
        let sources = [&images.base_color, &images.normal, &images.metallic_roughness, &images.occlusion];
        std::array::from_fn(|slot| {
            let uploaded = sources[slot].as_deref().and_then(|image| upload_image(device, queue, image, IMAGE_SLOTS[slot]));
            let texture = uploaded.as_ref().unwrap_or(&self.neutral[slot]);
            let format = if slot == 0 { F::Rgba8UnormSrgb } else { F::Rgba8Unorm };
            texture.create_view(&wgpu::TextureViewDescriptor { format: Some(format), ..Default::default() })
        })
    }
}

fn recipe(desc: &TextureDesc, layer: u32) -> Recipe {
    Recipe {
        pattern: desc.pattern as u32,
        seed: desc.seed,
        scale: desc.scale,
        layer,
        color_a: desc.color_a.extend(desc.roughness[0]).into(),
        color_b: desc.color_b.extend(desc.roughness[1]).into(),
        params: [desc.param, 0.0, 0.0, desc.bump],
    }
}

/// Uploads an RGBA8 image (viewable as linear or sRGB), or `None` if the device cannot hold it.
fn upload_image(device: &wgpu::Device, queue: &wgpu::Queue, image: &Image, label: &str) -> Option<wgpu::Texture> {
    let max = device.limits().max_texture_dimension_2d;
    let (width, height) = (image.width, image.height);
    if width == 0 || height == 0 || width > max || height > max || image.pixels.len() != (width * height * 4) as usize {
        log::warn!("ignoring {label} image of {width}x{height} ({} bytes; device limit {max}²)", image.pixels.len());
        return None;
    }
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: F::Rgba8Unorm,
        usage: U::TEXTURE_BINDING | U::COPY_DST,
        view_formats: &[F::Rgba8UnormSrgb],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &image.pixels,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height) },
        size,
    );
    Some(texture)
}

fn dummy(device: &wgpu::Device) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy array"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: F::Rgba8Unorm,
            usage: U::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
}
