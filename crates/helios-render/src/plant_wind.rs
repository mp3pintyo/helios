//! Dynamic plant wind (UE 5.9's Dynamic Wind for Nanite foliage, its parametric `Mode 0`): the
//! bones of procedural plants ([`helios_geometry::plant_bones`]) bent by the wind, posed once
//! per instance and frame by the pose pass ([`crate::plant_pose`]) and per vertex by every pass
//! that positions geometry (`shaders/plant_wind.wgsl`).
//!
//! The bone tables live in one `Rgba32Float` texture (binding [`BINDING_PLANT_BONES`], read with
//! `textureLoad`, so no shader stage gains a storage buffer): per mesh with bones its table (a
//! header texel, four texels a bone, the ranked bones' list: [`table_texels`]), per assembly part
//! its own table (the part's leaves, then the chain of plant bones it hangs from, in the part's
//! space), and per (mesh, material) pair a header texel with
//! the material's [`crate::WindResponse`]. An instance finds them through
//! [`crate::gpu_types::GpuInstance::plant`]; its `wind` then holds the procedural layers whose
//! culling and shadow-invalidation bounds hold the bones' motion
//! ([`helios_geometry::BoneBounds::equivalent`]), so culling, page expansion and invalidation
//! are unchanged. `RenderSettings::wind_bones` off draws the same plants with the procedural wind.

use std::collections::HashMap;

use glam::{Mat4, Vec4};
use helios_geometry::PlantBone;

use crate::scene::{Instance, Scene};

/// Binding of the plant bone texture in every shader that positions geometry.
pub const BINDING_PLANT_BONES: u32 = 170;
/// Texels per row (`PLANT_TEXTURE_WIDTH` in plant_wind.wgsl).
pub const PLANT_TEXTURE_WIDTH: u32 = 1024;

/// Layout entry of the plant bone texture.
pub fn plant_entry() -> (u32, wgpu::BindingType) {
    (BINDING_PLANT_BONES, crate::util::bind::texture(false))
}

/// The scene's plant bone texture for [`plant_entry`].
pub fn plant_resource(scene: &crate::gpu_scene::GpuScene) -> (u32, wgpu::BindingResource<'_>) {
    (BINDING_PLANT_BONES, wgpu::BindingResource::TextureView(&scene.plants.view))
}

/// Texels per bone (`PLANT_BONE_TEXELS` in plant_motion.wgsl).
pub const BONE_TEXELS: usize = 4;

/// How a bone hangs in its plant for the pose pass and the per-vertex pose
/// (`plant_motion.wgsl`, `plant_pose.wgsl`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoneLink {
    /// Its pose entry (ranked: it hangs from another bone and is not a leaf at the end of its
    /// chain), if any.
    pub rank: Option<u32>,
    /// A leaf (flutter) at the end of its chain: its vertices evaluate its arc themselves and
    /// ride on the parent, its *carrier*, at its attachment (else the bone is its own carrier).
    pub leaf: bool,
    /// The carrier's pose entry.
    pub carrier_rank: Option<u32>,
    /// The root its chain hangs from and where on it (the attachment of the bone just below the
    /// root, the leaf itself where its parent is the root; −1 for a root): a leaf on a root and
    /// instances whose ranked bones are all at rest ride on it there.
    pub root: u16,
    pub root_attach: f32,
}

/// The links of a table's bones, the ranked bones in rank order and the classes among them (bit
/// per [`helios_geometry::BoneClass`] index).
pub fn bone_links(bones: &[PlantBone]) -> (Vec<BoneLink>, Vec<u16>, u32) {
    let mut children = vec![0u32; bones.len()];
    for b in bones {
        if let Some(p) = b.parent {
            children[p as usize] += 1;
        }
    }
    let leaf = |i: usize| bones[i].class == helios_geometry::BoneClass::Flutter && bones[i].parent.is_some() && children[i] == 0;
    let mut ranked = Vec::new();
    let mut classes = 0u32;
    let mut rank = vec![None; bones.len()];
    for (i, b) in bones.iter().enumerate() {
        if b.parent.is_some() && !leaf(i) {
            rank[i] = Some(ranked.len() as u32);
            ranked.push(i as u16);
            classes |= 1 << b.class.index();
        }
    }
    let links = (0..bones.len())
        .map(|i| {
            let carrier = if leaf(i) { bones[i].parent.unwrap() as usize } else { i };
            // Up the chain to the bone that hangs from the root (a leaf's own attachment where
            // its parent is the root).
            let (mut x, mut root, mut root_attach) = (i, i as u16, -1.0);
            let mut guard = 0;
            while let Some(p) = bones[x].parent {
                root = p;
                root_attach = bones[x].attach;
                x = p as usize;
                guard += 1;
                assert!(guard <= bones.len(), "plant bone cycle");
            }
            BoneLink { rank: rank[i], leaf: leaf(i), carrier_rank: rank[carrier], root, root_attach }
        })
        .collect();
    (links, ranked, classes)
}

/// The four texels of a bone (`plant_motion.wgsl`): pivot and length; axis and class + phase;
/// amplitude, frequency, attachment and parent (−1: none); its carrier's pose entry (−1: none),
/// root, attachment on the root, leaf flag. Everything a vertex needs of its bone is in them, so
/// it fetches them at once.
pub fn bone_texels(b: &PlantBone, link: &BoneLink) -> [Vec4; BONE_TEXELS] {
    [
        b.pivot.extend(b.length.max(1e-6)),
        b.axis.extend(b.class.index() as f32 + b.phase.clamp(0.0, 0.999)),
        Vec4::new(b.amplitude, b.frequency, b.attach, b.parent.map_or(-1.0, f32::from)),
        Vec4::new(link.carrier_rank.map_or(-1.0, |r| r as f32), f32::from(link.root), link.root_attach, if link.leaf { 1.0 } else { 0.0 }),
    ]
}

/// A table's texels: its header (bones, ranked bones, their classes), the bones, then the ranked
/// bones' indices four a texel. Instances point at the first bone (the header's texel + 1).
pub fn table_texels(bones: &[PlantBone]) -> Vec<Vec4> {
    let (links, ranked, classes) = bone_links(bones);
    let mut texels = vec![Vec4::new(bones.len() as f32, ranked.len() as f32, classes as f32, 0.0)];
    texels.extend(bones.iter().zip(&links).flat_map(|(b, l)| bone_texels(b, l)));
    texels.extend(ranked.chunks(4).map(|c| Vec4::from_array(std::array::from_fn(|k| c.get(k).map_or(0.0, |&b| f32::from(b))))));
    texels
}

pub struct PlantTables {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    /// Texels of the geometry's tables (meshes and parts), then the instances' headers.
    texels: Vec<Vec4>,
    geometry_texels: usize,
    /// Per mesh: the first texel of its bone table.
    mesh_tables: Vec<Option<u32>>,
    /// Per (mesh, material): its header texel.
    headers: HashMap<(u32, u32), u32>,
}

impl PlantTables {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = Self::texture(device, 1);
        let view = texture.create_view(&Default::default());
        Self { texture, view, texels: Vec::new(), geometry_texels: 0, mesh_tables: Vec::new(), headers: HashMap::new() }
    }

    fn texture(device: &wgpu::Device, rows: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("plant bones"),
            size: wgpu::Extent3d { width: PLANT_TEXTURE_WIDTH, height: rows.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// Lays out the bone tables of the scene's meshes and of every assembly part; returns per
    /// assembly (in `scene.assemblies` order) the first texel of each part's table (0: none).
    pub fn upload_geometry(&mut self, scene: &Scene) -> Vec<Vec<u32>> {
        self.texels.clear();
        // Texel 0 stays unused: 0 means "no table" in the part records.
        self.texels.push(Vec4::ZERO);
        self.mesh_tables = scene
            .meshes
            .iter()
            .map(|mesh| {
                let bones = mesh.bones.as_ref().filter(|b| !b.bones.is_empty())?;
                let first = self.texels.len() as u32 + 1;
                self.texels.extend(table_texels(&bones.bones));
                Some(first)
            })
            .collect();
        let parts = scene
            .assemblies
            .iter()
            .map(|(base, list)| {
                let plant = scene.meshes[base.0 as usize].bones.as_ref().map_or(&[][..], |b| &b.bones[..]);
                list.iter()
                    .map(|part| {
                        let own = scene.meshes[part.mesh.0 as usize].bones.as_ref().map_or(&[][..], |b| &b.bones[..]);
                        let table = helios_geometry::plant_bones::part_bone_table(own, plant, part.bone, &part.transform);
                        if table.is_empty() {
                            return 0;
                        }
                        let first = self.texels.len() as u32 + 1;
                        self.texels.extend(table_texels(&table));
                        first
                    })
                    .collect()
            })
            .collect();
        self.geometry_texels = self.texels.len();
        self.headers.clear();
        if self.geometry_texels > 1 {
            log::info!("plant bones: {} texels ({:.2} MiB)", self.geometry_texels, self.geometry_texels as f64 * 16.0 / 1048576.0);
        }
        parts
    }

    /// The instance's plant record ([`crate::gpu_types::GpuInstance::plant`]) and wind (the
    /// procedural layers bounding its bones' motion); `None` for instances without bones (or
    /// with the bones off), which keep the procedural wind.
    pub fn instance(&mut self, scene: &Scene, instance: &Instance, response: Vec4, enabled: bool) -> Option<([u32; 4], Vec4)> {
        if !enabled || response == Vec4::ZERO {
            return None;
        }
        let table = (*self.mesh_tables.get(instance.mesh.0 as usize)?)?;
        let bones = scene.meshes[instance.mesh.0 as usize].bones.as_ref()?;
        let key = (instance.mesh.0, instance.material.0);
        let header = *self.headers.entry(key).or_insert_with(|| {
            self.texels.push(response);
            self.texels.len() as u32 - 1
        });
        let (lo, hi) = scale_range(&instance.transform);
        Some(([header + 1, table, 0, 0], equivalent_wind(&bones.bounds, response, lo, hi)))
    }

    /// Uploads the tables and headers (after the instances took their headers).
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let rows = (self.texels.len() as u32).div_ceil(PLANT_TEXTURE_WIDTH).max(1);
        if self.texture.height() < rows {
            self.texture = Self::texture(device, rows);
            self.view = self.texture.create_view(&Default::default());
        }
        let mut data = self.texels.clone();
        data.resize((rows * PLANT_TEXTURE_WIDTH) as usize, Vec4::ZERO);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(PLANT_TEXTURE_WIDTH * 16), rows_per_image: Some(rows) },
            wgpu::Extent3d { width: PLANT_TEXTURE_WIDTH, height: rows, depth_or_array_layers: 1 },
        );
    }

    /// Drops the instances' headers (before the instances are uploaded again).
    pub fn reset_headers(&mut self) {
        self.texels.truncate(self.geometry_texels);
        self.headers.clear();
    }
}

/// Smallest and largest scale of an affine transform's axes.
fn scale_range(transform: &Mat4) -> (f32, f32) {
    let m = glam::Mat3::from_mat4(*transform);
    let s = [m.x_axis.length(), m.y_axis.length(), m.z_axis.length()];
    (s.iter().copied().fold(f32::MAX, f32::min), s.iter().copied().fold(0.0, f32::max))
}

/// [`helios_geometry::BoneBounds::equivalent`] for a transform scaling by `lo..=hi`: world
/// heights and axis distances are at least `lo` times the object's, displacements at most `hi`
/// times.
pub fn equivalent_wind(bounds: &helios_geometry::BoneBounds, response: Vec4, lo: f32, hi: f32) -> Vec4 {
    let lo = lo.max(1e-6);
    let e = bounds.equivalent(response, 1.0);
    // α·h_o²·hi ≤ α·hi/lo²·h_w²; β·ρ_o·hi ≤ β·hi/lo·ρ_w; κ·hi.
    Vec4::new(e.x * hi / (lo * lo), e.y * hi / lo, e.z * hi, e.w)
}
