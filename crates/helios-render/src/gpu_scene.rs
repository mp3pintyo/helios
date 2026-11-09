//! GPU mirror of a [`Scene`]: always-resident cluster/page records of every mesh, the
//! streamed geometry pool with its page table, instances, materials with their alpha masks and
//! the persistent cluster visibility bitset used by two-pass occlusion culling.

use glam::{Affine2, Mat3, Mat4, Vec2, Vec4};
use helios_geometry::{GpuCluster, GpuPage, INVALID_PAGE};

use crate::alpha_mask::{AlphaMasks, MaterialMask};
use crate::gpu_types::{instance_flags, material_flags, GpuInstance, GpuMaterial, GpuUvTransform};
use crate::scene::{Instance, Material, Scene, TextureHandle, TextureSource, UvTransform};
use crate::skin::{instance_skin_flags, SkinPalettes};
use crate::streaming::{StreamingStats, Streamer};
use crate::util::{storage_buffer, storage_buffer_init};
use crate::wind::WindUniform;

#[derive(Clone, Copy, Debug, Default)]
pub struct MeshRange {
    pub page_offset: u32,
    pub page_count: u32,
    pub cluster_offset: u32,
    pub cluster_count: u32,
    pub bounds: Vec4,
}

/// Materials live in a uniform buffer (keeps material passes within 8 storage bindings).
pub const MAX_MATERIALS: usize = 256;

pub struct GpuScene {
    pub clusters: wgpu::Buffer,
    pub pages: wgpu::Buffer,
    /// Global page -> geometry pool word offset (or `NOT_RESIDENT`).
    pub page_table: wgpu::Buffer,
    /// Streamed vertex and triangle data of resident pages.
    pub geometry: wgpu::Buffer,
    /// `[0, page_count)`: per page, the last frame in which a resident parent wanted to refine
    /// into it (streaming feedback). After that: one bit per (instance, cluster), "visible last
    /// frame" (occlusion culling). One buffer keeps the culling pass within 8 storage bindings.
    pub atomics: wgpu::Buffer,
    pub instances: wgpu::Buffer,
    pub materials: wgpu::Buffer,
    /// Some material samples a texture with the second UV set, a wrap mode other than repeat or
    /// an occlusion layer of its own: the G-buffer resolve needs its `UV_EXTENSIONS` permutation.
    pub uv_extensions: bool,
    /// [`WindUniform`], written every frame ([`GpuScene::write_wind`]).
    pub wind: wgpu::Buffer,
    /// Alpha masks of alpha-tested materials, sampled by every rasteriser.
    pub masks: AlphaMasks,
    /// Joint palettes of skinned instances, written every frame.
    pub skins: SkinPalettes,
    pub instance_count: u32,
    pub page_count: u32,
    pub meshes: Vec<MeshRange>,
    /// World bounds of all instances.
    bounds: (glam::Vec3, glam::Vec3),
    streamer: Streamer,
    previous_world: Vec<Mat4>,
    instances_moving: bool,
    /// Some shadow-casting instance has a wind response.
    wind_shadows: bool,
    /// Scene versions last uploaded: geometry, materials, instances, textures.
    versions: (u64, u64, u64, u64),
}

impl GpuScene {
    pub fn new(device: &wgpu::Device, pool_bytes: u64) -> Self {
        let empty = |label| storage_buffer(device, label, 16, wgpu::BufferUsages::empty());
        let pool_words = (pool_bytes / 4).min(u32::MAX as u64 - 1) as u32;
        Self {
            clusters: empty("clusters"),
            pages: empty("pages"),
            page_table: empty("page table"),
            geometry: storage_buffer(device, "geometry pool", pool_words as u64 * 4, wgpu::BufferUsages::empty()),
            atomics: storage_buffer(device, "culling atomics", 16, wgpu::BufferUsages::COPY_SRC),
            instances: empty("instances"),
            materials: crate::util::uniform_buffer(device, "materials", (MAX_MATERIALS * std::mem::size_of::<GpuMaterial>()) as u64),
            uv_extensions: false,
            wind: crate::util::uniform_buffer(device, "wind", std::mem::size_of::<WindUniform>() as u64),
            masks: AlphaMasks::new(device),
            skins: SkinPalettes::new(device),
            instance_count: 0,
            page_count: 0,
            meshes: Vec::new(),
            bounds: (glam::Vec3::ZERO, glam::Vec3::ZERO),
            streamer: Streamer::new(pool_words),
            previous_world: Vec::new(),
            instances_moving: false,
            wind_shadows: false,
            versions: (u64::MAX, u64::MAX, u64::MAX, u64::MAX),
        }
    }

    pub fn bounds(&self) -> (glam::Vec3, glam::Vec3) {
        self.bounds
    }

    pub fn streaming_stats(&self) -> StreamingStats {
        self.streamer.stats()
    }

    /// Whether the wind moves any shadow-casting instance (whose cached shadow pages it then
    /// invalidates).
    pub fn wind_moves_shadows(&self) -> bool {
        self.wind_shadows
    }

    pub fn write_wind(&self, queue: &wgpu::Queue, wind: &WindUniform) {
        queue.write_buffer(&self.wind, 0, bytemuck::bytes_of(wind));
    }

    /// Uploads whatever changed since the previous call and streams pages. Returns true if
    /// geometry was rebuilt.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, frame: u32) -> bool {
        let geometry_changed = self.versions.0 != scene.geometry_version;
        if geometry_changed {
            self.upload_geometry(device, queue, scene);
        }
        // Masks depend on materials and their textures; instances carry their material's
        // alpha-masked flag for culling.
        let materials_changed = geometry_changed || self.versions.1 != scene.material_version || self.versions.3 != scene.texture_version;
        if materials_changed {
            self.masks.sync(device, queue, scene);
            self.upload_materials(queue, scene);
        }
        let instances_changed = self.versions.2 != scene.instance_version;
        if materials_changed || instances_changed || self.instances_moving {
            self.upload_instances(device, queue, scene, geometry_changed);
        }
        self.versions = (scene.geometry_version, scene.material_version, scene.instance_version, scene.texture_version);
        self.skins.sync(device, queue, scene);

        self.streamer.update(queue, &self.geometry, &scene.meshes, frame);
        self.streamer.upload_page_table(queue, &self.page_table);
        geometry_changed
    }

    /// Records the feedback read-back (call at the end of the frame's command encoding).
    pub fn record_feedback(&mut self, encoder: &mut wgpu::CommandEncoder, frame: u32) -> bool {
        self.streamer.record_feedback(encoder, &self.atomics, frame)
    }

    pub fn map_feedback(&self) {
        self.streamer.map_feedback();
    }

    fn upload_geometry(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) {
        let mut clusters: Vec<GpuCluster> = Vec::new();
        let mut pages: Vec<GpuPage> = Vec::new();
        self.meshes.clear();

        for mesh in &scene.meshes {
            let page_base = pages.len() as u32;
            let cluster_base = clusters.len() as u32;
            let global = |p: u32| if p == INVALID_PAGE { INVALID_PAGE } else { p + page_base };
            clusters.extend(mesh.clusters.iter().map(|c| GpuCluster { page: global(c.page), child_page: global(c.child_page), ..*c }));
            pages.extend(mesh.pages.iter().map(|p| GpuPage { cluster_offset: p.cluster_offset + cluster_base, ..*p }));
            self.meshes.push(MeshRange {
                page_offset: page_base,
                page_count: mesh.pages.len() as u32,
                cluster_offset: cluster_base,
                cluster_count: mesh.clusters.len() as u32,
                bounds: mesh.bounds.to_vec4(),
            });
        }

        let payload: usize = scene.meshes.iter().map(|m| m.payload_bytes()).sum();
        let streamed = scene.page_locations.iter().filter(|l| l.is_some()).count();
        log::info!(
            "uploading geometry: {} meshes ({streamed} streamed from page sources), {} pages, {} clusters, {:.1} MiB streamable ({:.1} MiB pool)",
            scene.meshes.len(),
            pages.len(),
            clusters.len(),
            payload as f64 / 1048576.0,
            self.geometry.size() as f64 / 1048576.0
        );

        self.page_count = pages.len() as u32;
        self.clusters = storage_buffer_init(device, queue, "clusters", bytemuck::cast_slice(&clusters));
        self.pages = storage_buffer_init(device, queue, "pages", bytemuck::cast_slice(&pages));
        self.page_table = storage_buffer(device, "page table", pages.len() as u64 * 4, wgpu::BufferUsages::empty());
        if !self.streamer.reset(device, queue, &self.geometry, &scene.meshes, &scene.page_locations) {
            log::error!("geometry pool too small for the root pages; increase RendererConfig::geometry_pool_bytes");
        }
    }

    fn upload_materials(&mut self, queue: &wgpu::Queue, scene: &Scene) {
        let mut materials: Vec<GpuMaterial> =
            scene.materials.iter().enumerate().map(|(i, m)| gpu_material(scene, m, self.masks.material(i))).collect();
        if materials.len() > MAX_MATERIALS {
            log::warn!("{} materials exceed the limit of {MAX_MATERIALS}; instances of the rest render with the wrong material", materials.len());
        }
        materials.truncate(MAX_MATERIALS);
        self.uv_extensions = materials.iter().any(needs_uv_extensions);
        materials.resize(MAX_MATERIALS, bytemuck::Zeroable::zeroed());
        queue.write_buffer(&self.materials, 0, bytemuck::cast_slice(&materials));
    }

    fn upload_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, layout_changed: bool) {
        let count = scene.instances.len();
        if self.previous_world.len() != count {
            self.previous_world = scene.instances.iter().map(|i| i.transform).collect();
        }

        // Visibility bits start after the page feedback words.
        let mut visibility_bits = self.page_count * 32;
        let mut moving = false;
        let mut wind_shadows = false;
        let instances: Vec<GpuInstance> = scene
            .instances
            .iter()
            .zip(&self.previous_world)
            .map(|(instance, &prev_world)| {
                let mesh = self.meshes[instance.mesh.0 as usize];
                let world = instance.transform;
                moving |= world != prev_world;
                let linear = Mat3::from_mat4(world);
                let max_scale = linear.x_axis.length().max(linear.y_axis.length()).max(linear.z_axis.length());
                // Inverse transpose, rescaled so that uniform scale leaves normals unit length.
                let normal = linear.inverse().transpose() * max_scale;
                let center = world.transform_point3(mesh.bounds.truncate());
                let wind = instance_wind(scene, instance);
                wind_shadows |= instance.cast_shadow && wind != Vec4::ZERO;
                let gpu = GpuInstance {
                    world,
                    prev_world,
                    normal_matrix: [normal.x_axis.extend(0.0), normal.y_axis.extend(0.0), normal.z_axis.extend(0.0)],
                    bounds: center.extend(mesh.bounds.w * max_scale),
                    page_offset: mesh.page_offset,
                    page_count: mesh.page_count,
                    cluster_offset: mesh.cluster_offset,
                    cluster_count: mesh.cluster_count,
                    material: instance.material.0,
                    visibility_offset: visibility_bits,
                    max_scale,
                    flags: gpu_instance_flags(scene, instance, self.masks.material(instance.material.0 as usize).is_some())
                        | instance_skin_flags(scene, instance),
                    wind,
                };
                visibility_bits += mesh.cluster_count;
                gpu
            })
            .collect();

        let (mut lo, mut hi) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
        for i in &instances {
            lo = lo.min(i.bounds.truncate() - i.bounds.w);
            hi = hi.max(i.bounds.truncate() + i.bounds.w);
        }
        self.bounds = if instances.is_empty() { (glam::Vec3::ZERO, glam::Vec3::ONE) } else { (lo, hi) };
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        if self.instances.size() < bytes.len().max(16) as u64 || self.instance_count as usize != count {
            self.instances = storage_buffer(device, "instances", bytes.len() as u64, wgpu::BufferUsages::empty());
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.instances, 0, bytes);
        }
        if layout_changed || self.instance_count as usize != count {
            // Instance layout changed: start occlusion culling from scratch.
            let words = (visibility_bits as u64).div_ceil(32);
            self.atomics = storage_buffer(device, "culling atomics", words * 4, wgpu::BufferUsages::COPY_SRC);
        }

        self.instance_count = count as u32;
        self.instances_moving = moving;
        self.wind_shadows = wind_shadows;
        for (prev, instance) in self.previous_world.iter_mut().zip(&scene.instances) {
            *prev = instance.transform;
        }
    }
}

fn gpu_instance_flags(scene: &Scene, instance: &Instance, alpha_masked: bool) -> u32 {
    if !instance.visible {
        // No shadow flag either: shadow invalidation passes skip it like the culling does.
        return instance_flags::HIDDEN;
    }
    let mut flags = if instance.cast_shadow { instance_flags::CAST_SHADOW } else { 0 };
    if alpha_masked {
        flags |= instance_flags::ALPHA_MASKED;
    }
    if scene.materials.get(instance.material.0 as usize).is_some_and(|m| m.kind.is_translucent()) {
        flags |= instance_flags::TRANSLUCENT;
    }
    flags
}

/// The instance's wind response; translucent instances are drawn by the forward pass, which
/// doesn't animate them, so they stay static everywhere.
fn instance_wind(scene: &Scene, instance: &Instance) -> Vec4 {
    match scene.materials.get(instance.material.0 as usize) {
        Some(m) if !m.kind.is_translucent() && !m.wind.is_static() => m.wind.to_vec4(),
        _ => Vec4::ZERO,
    }
}

fn gpu_material(scene: &Scene, m: &Material, mask: Option<MaterialMask>) -> GpuMaterial {
    let layer = |t: Option<TextureHandle>| t.map_or(255, |t| t.0.min(254));
    let from_images = m.texture.is_some_and(|t| matches!(scene.textures.get(t.0 as usize), Some(TextureSource::Images(_))));
    let alpha_test = mask.map_or(0, |mask| {
        (mask.layer + 1) << material_flags::ALPHA_MASK_SHIFT | (mask.cutoff as u32) << material_flags::ALPHA_CUTOFF_SHIFT
    });
    let has_occlusion = |t: TextureHandle| matches!(scene.textures.get(t.0 as usize), Some(TextureSource::Images(i)) if i.occlusion.is_some());
    let occlusion = m.occlusion_texture.filter(|&t| has_occlusion(t) && m.occlusion_strength > 0.0);
    let occlusion_in_texture = occlusion.is_some() && occlusion == m.texture && m.occlusion_uv == m.uv;
    // The material's tiling applies to every mesh-UV mapping.
    let tiled = |t: &UvTransform| gpu_uv_transform(&UvTransform { matrix: Affine2::from_scale(Vec2::splat(m.uv_scale)) * t.matrix, ..*t });
    GpuMaterial {
        base_color: m.base_color.extend(m.opacity),
        emissive: m.emissive_color.extend(m.emissive),
        roughness: m.roughness,
        metallic: m.metallic,
        kind: m.kind as u32,
        flags: if from_images { material_flags::IMAGE_TEXTURE } else { 0 }
            | if occlusion_in_texture { material_flags::OCCLUSION_IN_TEXTURE } else { 0 }
            | alpha_test,
        texture: m.texture.map_or(u32::MAX, |t| t.0),
        uv_scale: m.uv_scale,
        normal_strength: m.normal_strength,
        mapping: m.mapping as u32,
        detail_scale: m.detail_scale,
        detail_strength: m.detail_strength,
        layers: m.layers.iter().enumerate().fold(0, |acc, (i, &t)| acc | layer(t) << (i * 8)),
        emissive_texture: m.emissive_texture.map_or(u32::MAX, |t| t.0),
        uv: [tiled(&m.uv), tiled(&m.emissive_uv), tiled(&m.occlusion_uv)],
        occlusion_texture: occlusion.map_or(u32::MAX, |t| t.0),
        occlusion_strength: m.occlusion_strength.clamp(0.0, 1.0),
        _pad: [0; 2],
    }
}

/// True if the material uses a texture coordinate feature of the resolve's `UV_EXTENSIONS`
/// permutation (see `passes/material.rs`). Texture transforms themselves are always applied.
fn needs_uv_extensions(m: &GpuMaterial) -> bool {
    let separate_occlusion = m.occlusion_texture != u32::MAX && m.flags & material_flags::OCCLUSION_IN_TEXTURE == 0;
    separate_occlusion || m.uv.iter().any(|t| t.uv_set != 0 || t.wrap != 0)
}

fn gpu_uv_transform(t: &UvTransform) -> GpuUvTransform {
    let (m, o) = (t.matrix.matrix2, t.matrix.translation);
    GpuUvTransform {
        u: [m.x_axis.x, m.y_axis.x, o.x],
        uv_set: t.set.min(1),
        v: [m.x_axis.y, m.y_axis.y, o.y],
        wrap: t.wrap[0] as u32 | (t.wrap[1] as u32) << 4,
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::scene::{ImageTexture, TextureDesc, TexturePattern};

    #[test]
    fn rare_texture_coordinates_select_the_extended_resolve() {
        use crate::scene::{UvTransform, Wrap};
        let mut scene = Scene::new();
        let orm = Some(std::sync::Arc::new(crate::scene::Image { width: 1, height: 1, pixels: vec![255; 4] }));
        let layer = scene.add_image_texture(ImageTexture { metallic_roughness: orm.clone(), occlusion: orm, ..Default::default() });
        let needs = |m: Material| needs_uv_extensions(&gpu_material(&scene, &m, None));
        let textured = Material { texture: Some(layer), ..Default::default() };
        // Transforms and occlusion in the material's own texture are part of the common path.
        let moved = UvTransform { matrix: glam::Affine2::from_translation(glam::Vec2::X), ..UvTransform::IDENTITY };
        assert!(!needs(Material { uv: moved, occlusion_texture: Some(layer), occlusion_uv: moved, ..textured }));
        assert!(!needs(Material::default()));
        // The second UV set, clamping and an occlusion layer sampled with other coordinates are not.
        assert!(needs(Material { uv: UvTransform { set: 1, ..UvTransform::IDENTITY }, ..textured }));
        assert!(needs(Material { emissive_uv: UvTransform { wrap: [Wrap::ClampToEdge, Wrap::Repeat], ..UvTransform::IDENTITY }, ..textured }));
        assert!(needs(Material { occlusion_texture: Some(layer), occlusion_uv: moved, ..textured }));
    }

    #[test]
    fn image_textures_flag_their_materials() {
        let mut scene = Scene::new();
        let procedural = scene.add_texture(TextureDesc::new(TexturePattern::Rock, Vec3::ONE, Vec3::ONE));
        let images = scene.add_image_texture(ImageTexture::default());
        let gpu = |texture| gpu_material(&scene, &Material { texture: Some(texture), ..Default::default() }, None);
        assert_eq!(gpu(procedural).flags & material_flags::IMAGE_TEXTURE, 0);
        assert_eq!(gpu(images).flags & material_flags::IMAGE_TEXTURE, material_flags::IMAGE_TEXTURE);

        let glowing = Material { emissive_color: Vec3::new(1.0, 0.5, 0.0), emissive_texture: Some(images), emissive: 2.0, ..Default::default() };
        let g = gpu_material(&scene, &glowing, None);
        assert_eq!(g.emissive, glam::Vec4::new(1.0, 0.5, 0.0, 2.0));
        assert_eq!((g.texture, g.emissive_texture, g.layers), (u32::MAX, images.0, u32::MAX));
    }

    #[test]
    fn alpha_tests_pack_above_the_material_flags() {
        let mut scene = Scene::new();
        let images = scene.add_image_texture(ImageTexture::default());
        let material = Material { texture: Some(images), alpha_cutoff: Some(0.5), ..Default::default() };
        let g = gpu_material(&scene, &material, Some(MaterialMask { layer: 2, cutoff: 128 }));
        assert_eq!(g.flags, material_flags::IMAGE_TEXTURE | 3 << 16 | 128 << 8);
        assert_eq!(gpu_material(&scene, &material, None).flags, material_flags::IMAGE_TEXTURE);
    }

    #[test]
    fn translucent_materials_flag_their_instances() {
        use crate::scene::MaterialKind;
        let mut scene = Scene::new();
        let mesh = scene.add_mesh(helios_geometry::VirtualMesh::default());
        let opaque = scene.add_material(Material::default());
        let water = scene.add_material(Material { kind: MaterialKind::Water, ..Default::default() });
        let glass = scene.add_material(Material { kind: MaterialKind::Glass, opacity: 0.5, ..Default::default() });
        let rock = scene.add_instance(mesh, opaque, Mat4::IDENTITY);
        let lake = scene.add_instance(mesh, water, Mat4::IDENTITY);
        let pane = scene.add_instance(mesh, glass, Mat4::IDENTITY);
        scene.set_cast_shadow(lake, false);
        let flags = |h: crate::scene::InstanceHandle, masked| gpu_instance_flags(&scene, &scene.instances[h.0 as usize], masked);
        assert_eq!(flags(rock, false), instance_flags::CAST_SHADOW);
        assert_eq!(flags(rock, true), instance_flags::CAST_SHADOW | instance_flags::ALPHA_MASKED);
        assert_eq!(flags(lake, false), instance_flags::TRANSLUCENT);
        assert_eq!(flags(pane, false), instance_flags::CAST_SHADOW | instance_flags::TRANSLUCENT);
        assert_eq!(gpu_material(&scene, &scene.materials[glass.0 as usize], None).base_color.w, 0.5);
    }

    #[test]
    fn instances_take_their_material_wind_response_unless_translucent() {
        use crate::scene::MaterialKind;
        use crate::wind::WindResponse;
        let mut scene = Scene::new();
        let mesh = scene.add_mesh(helios_geometry::VirtualMesh::default());
        let rock = scene.add_material(Material::default());
        let grass = scene.add_material(Material { kind: MaterialKind::Foliage, wind: WindResponse::GRASS, ..Default::default() });
        let glass = scene.add_material(Material { kind: MaterialKind::Glass, wind: WindResponse::GRASS, ..Default::default() });
        let handles = [rock, grass, glass].map(|m| scene.add_instance(mesh, m, Mat4::IDENTITY));
        let wind = |i: usize| instance_wind(&scene, &scene.instances[handles[i].0 as usize]);
        assert_eq!(wind(0), Vec4::ZERO);
        assert_eq!(wind(1), WindResponse::GRASS.to_vec4());
        assert_eq!(wind(2), Vec4::ZERO);
    }
}
