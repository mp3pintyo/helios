//! Lumen near field: per-mesh distance fields traced directly for the first metres of a ray.
//!
//! The global distance field is a clipmap of 0.25 m voxels at best, so anything smaller vanishes
//! from it or turns into a voxel-sized blob (a candle shadowing its own flame, chair legs
//! occluding like boxes). Like UE5 Lumen, rays therefore first sphere-trace the *mesh* SDFs of
//! the instances around their origin and continue in the global field beyond
//! [`NEAR_FIELD_DISTANCE`] (`lumen_mesh_sdf.wgsl`). This module prepares the GPU data:
//!
//! * **Atlas** — every volume mesh SDF packed into one r8snorm 3D texture, so a tracing step is a
//!   single hardware-filtered fetch instead of eight buffer reads.
//! * **Instance records** — per shadow-casting instance with an SDF: the world → field-space
//!   transform (atlas texels for volumes, cells for heightfields), the field box, the world
//!   distance per field unit, for heightfields where their heights live, the instance's index in
//!   the mesh-card surface cache (so a hit can be shaded from its cards) and its coverage: the
//!   solid fraction of an alpha-masked surface, which traces treat as partially transparent.
//! * **Culling grid** — a camera-centred grid of 2 m cells; each cell lists the instances within
//!   reach of it, so a ray only tests the list of the cell it starts in. Cells are world-aligned
//!   and the grid moves in steps of several cells: a cell's list never depends on the camera, and
//!   it is rebuilt only every few metres of camera movement.
//!
//! Records, heightfield heights and the grid share one storage buffer — a single binding in the
//! passes that trace (WebGPU allows 8 storage buffers per stage, and the lighting pass is close).

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat3, Mat4, Vec3, Vec4};
use helios_geometry::{MeshSdf, SdfKind};
use wgpu::{TextureFormat as F, TextureUsages as U};

use crate::scene::Scene;
use crate::util::storage_buffer;

/// How far rays trace mesh SDFs before continuing in the global distance field (UE5 Lumen's
/// default mesh-SDF trace distance is 1.8 m).
pub const NEAR_FIELD_DISTANCE: f32 = 2.0;
const CELL_SIZE: f32 = 2.0;
const GRID_DIMS: IVec3 = IVec3::new(32, 16, 32);
/// The grid moves in steps of this many cells.
const GRID_SNAP: i32 = 4;
/// A cell lists the instances within this distance of it: the near-field distance plus the
/// offsets rays start at (a few centimetres off their surface).
const CELL_REACH: f32 = NEAR_FIELD_DISTANCE + 0.25;
/// Per-cell cap (the closest instances are kept); the count is stored in 8 bits.
const MAX_CELL_INSTANCES: usize = 64;
/// u32 words per instance record (8 vec4, see `mesh_sdf_load` in lumen_mesh_sdf.wgsl).
const RECORD_WORDS: usize = 32;
/// Volume fields are traced for soft shadows this many voxels beyond their box, where the
/// penumbra fades out (`MESH_SDF_PENUMBRA_MARGIN` in lumen_mesh_sdf.wgsl).
const PENUMBRA_MARGIN: f32 = 2.0;
const ATLAS_SIDE: u32 = 256;

/// Near-field part of the Lumen uniform (`mesh_sdf_*` in lumen_types.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct NearFieldUniform {
    /// Grid min corner, w: cell size (0 disables the near field).
    grid: Vec4,
    /// Cells per axis, w: first vec4 of the cell table in the data buffer.
    dims: [u32; 4],
    /// x: near-field distance.
    trace: Vec4,
}

/// Where each volume SDF lives in the atlas.
#[derive(Debug)]
struct AtlasLayout {
    offsets: Vec<[u32; 3]>,
    size: [u32; 3],
}

/// Packs boxes into an atlas `side` texels wide and high, in slabs along z. Boxes are sorted
/// deepest first; within a slab they fill rows (shelves) along x, a slab is as deep as its
/// deepest box.
fn pack_atlas(sizes: &[[u32; 3]], side: u32) -> AtlasLayout {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(sizes[i][2]), std::cmp::Reverse(sizes[i][1]), i));
    let mut offsets = vec![[0; 3]; sizes.len()];
    let mut size = [1, 1, 1];
    let (mut x, mut y, mut z) = (0, 0, 0);
    let (mut shelf, mut slab) = (0, 0);
    for i in order {
        let [w, h, d] = sizes[i];
        assert!(w <= side && h <= side, "mesh SDF {w}x{h} exceeds the {side}² atlas");
        if x + w > side {
            (x, y, shelf) = (0, y + shelf, 0);
        }
        if y + h > side {
            (x, y, z, shelf, slab) = (0, 0, z + slab, 0, 0);
        }
        offsets[i] = [x, y, z];
        size = [size[0].max(x + w), size[1].max(y + h), size[2].max(z + d)];
        x += w;
        shelf = shelf.max(h);
        slab = slab.max(d);
    }
    AtlasLayout { offsets, size }
}

/// One near-field instance: scene instance and mesh index, transform, surface-cache index and
/// coverage (the solid fraction of its surface: 1 unless alpha-masked).
#[derive(Clone, Copy, Debug)]
struct NearInstance {
    index: usize,
    mesh: usize,
    world: Mat4,
    card: u32,
    coverage: f32,
}

/// The record of one instance and its world-space bounds.
fn instance_record(sdf: &MeshSdf, instance: &NearInstance, atlas_offset: [u32; 3], heightfield_word: u32) -> ([u32; RECORD_WORDS], (Vec3, Vec3)) {
    let world = instance.world;
    let linear = Mat3::from_mat4(world);
    // Field distances times the smallest scale bound world distances from below, so sphere
    // tracing stays conservative under non-uniform scale.
    let min_scale = linear.x_axis.length().min(linear.y_axis.length()).min(linear.z_axis.length());
    let voxel = sdf.voxel_size;
    let res = Vec3::from(sdf.resolution.map(|r| r as f32));
    let to_cells = Mat4::from_scale(Vec3::splat(1.0 / voxel)) * Mat4::from_translation(-sdf.origin);
    let (to_field, lo, hi, scale, heightfield) = match sdf.kind {
        SdfKind::Volume => {
            // Voxel centres are texel centres of the atlas.
            let offset = Vec3::from(atlas_offset.map(|o| o as f32)) + 0.5;
            (Mat4::from_translation(offset) * to_cells, offset, offset + res - 1.0, sdf.band * min_scale, [0; 4])
        }
        SdfKind::Heightfield => {
            let [low, high] = sdf.height_range.map(|h| (h - sdf.origin.y) / voxel);
            assert!(sdf.resolution[0] < 1 << 16 && sdf.resolution[2] < 1 << 16);
            let packed_res = sdf.resolution[0] | (sdf.resolution[2] << 16);
            (
                to_cells,
                Vec3::new(0.0, low - 2.0, 0.0),
                Vec3::new(res.x - 1.0, high + 2.0, res.z - 1.0),
                voxel * min_scale,
                [heightfield_word, packed_res, low.to_bits(), high.to_bits()],
            )
        }
    };
    let world_to_field = to_field * world.inverse();

    // World bounds (AABB for the culling grid, sphere for a cheap per-ray rejection) of the box
    // the shaders trace, penumbra margin included.
    let margin = if sdf.kind == SdfKind::Volume { PENUMBRA_MARGIN } else { 0.0 };
    let field_to_world = world_to_field.inverse();
    let corners: Vec<Vec3> = (0..8)
        .map(|k| {
            let corner = Vec3::new(
                if k & 1 == 0 { lo.x - margin } else { hi.x + margin },
                if k & 2 == 0 { lo.y - margin } else { hi.y + margin },
                if k & 4 == 0 { lo.z - margin } else { hi.z + margin },
            );
            field_to_world.transform_point3(corner)
        })
        .collect();
    let (min, max) = corners.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
    let centre = field_to_world.transform_point3((lo + hi) * 0.5);
    let radius = corners.iter().map(|p| p.distance(centre)).fold(0.0, f32::max);

    let mut record = [0u32; RECORD_WORDS];
    record[0..4].copy_from_slice(&centre.extend(radius).to_array().map(f32::to_bits));
    for row in 0..3 {
        let r = world_to_field.row(row);
        record[4 + row * 4..8 + row * 4].copy_from_slice(&r.to_array().map(f32::to_bits));
    }
    record[16..20].copy_from_slice(&lo.extend(scale).to_array().map(f32::to_bits));
    record[20..24].copy_from_slice(&hi.extend(voxel * min_scale).to_array().map(f32::to_bits));
    record[24..28].copy_from_slice(&heightfield);
    record[28] = instance.card;
    record[29] = instance.coverage.to_bits();
    (record, (min, max))
}

/// CPU-side near-field scene: records + heightfield heights, the atlas and instance bounds.
struct NearFieldData {
    /// Records, then heightfield heights (2 per word); padded to whole vec4s.
    words: Vec<u32>,
    atlas: AtlasLayout,
    atlas_voxels: Vec<u8>,
    /// Per record: its instance, where its field lives (atlas offset, heightfield word) and its
    /// bounds.
    instances: Vec<NearInstance>,
    placements: Vec<([u32; 3], u32)>,
    bounds: Vec<(Vec3, Vec3)>,
}

/// Builds the near-field data for `instances` of `meshes`; instances whose mesh has no SDF are
/// skipped.
fn build_data(meshes: &[Option<&MeshSdf>], instances: &[NearInstance]) -> NearFieldData {
    let instances: Vec<NearInstance> = instances.iter().copied().filter(|i| meshes[i.mesh].is_some()).collect();
    let mut used = vec![false; meshes.len()];
    for i in &instances {
        used[i.mesh] = true;
    }

    // Atlas slots for volumes, heightfield heights after the records.
    let volumes: Vec<usize> = (0..meshes.len()).filter(|&m| used[m] && meshes[m].is_some_and(|s| s.kind == SdfKind::Volume)).collect();
    let atlas = pack_atlas(&volumes.iter().map(|&m| meshes[m].unwrap().resolution).collect::<Vec<_>>(), ATLAS_SIDE);
    let mut atlas_slot = vec![[0; 3]; meshes.len()];
    let [aw, ah, ad] = atlas.size;
    let mut atlas_voxels = vec![0u8; (aw * ah * ad) as usize];
    for (slot, &m) in volumes.iter().enumerate() {
        let sdf = meshes[m].unwrap();
        let [ox, oy, oz] = atlas.offsets[slot];
        let [rx, ry, rz] = sdf.resolution;
        for z in 0..rz {
            for y in 0..ry {
                for x in 0..rx {
                    let i = (x + rx * (y + ry * z)) as usize;
                    let byte = (sdf.words[i / 4] >> ((i % 4) * 8)) as u8;
                    atlas_voxels[(ox + x + aw * (oy + y + ah * (oz + z))) as usize] = byte;
                }
            }
        }
        atlas_slot[m] = atlas.offsets[slot];
    }

    let mut words = vec![0u32; instances.len() * RECORD_WORDS];
    let mut heightfield_word = vec![0u32; meshes.len()];
    for m in (0..meshes.len()).filter(|&m| used[m]) {
        let sdf = meshes[m].unwrap();
        if sdf.kind == SdfKind::Heightfield {
            heightfield_word[m] = words.len() as u32;
            words.extend_from_slice(&sdf.words);
        }
    }
    words.resize(words.len().next_multiple_of(4), 0);

    let mut bounds = Vec::with_capacity(instances.len());
    let mut placements = Vec::with_capacity(instances.len());
    for (i, instance) in instances.iter().enumerate() {
        let m = instance.mesh;
        let (record, b) = instance_record(meshes[m].unwrap(), instance, atlas_slot[m], heightfield_word[m]);
        words[i * RECORD_WORDS..(i + 1) * RECORD_WORDS].copy_from_slice(&record);
        bounds.push(b);
        placements.push((atlas_slot[m], heightfield_word[m]));
    }
    NearFieldData { words, atlas, atlas_voxels, instances, placements, bounds }
}

/// The grid cells (inclusive range, clamped to the grid) whose lists include an instance with
/// world bounds `(lo, hi)` (see `build_grid`).
fn cell_range((lo, hi): (Vec3, Vec3), origin: IVec3) -> (IVec3, IVec3) {
    let base = origin.as_vec3() * CELL_SIZE;
    let a = ((lo - CELL_REACH - base) / CELL_SIZE).floor().as_ivec3().max(IVec3::ZERO);
    let b = ((hi + CELL_REACH - base) / CELL_SIZE).floor().as_ivec3().min(GRID_DIMS - 1);
    (a, b)
}

/// Grid origin (in cells) for a camera position, snapped so the grid moves every GRID_SNAP cells.
fn grid_origin(camera: Vec3) -> IVec3 {
    let cell = (camera / CELL_SIZE).floor().as_ivec3();
    let snapped = IVec3::new(cell.x.div_euclid(GRID_SNAP), cell.y.div_euclid(GRID_SNAP), cell.z.div_euclid(GRID_SNAP)) * GRID_SNAP;
    snapped - GRID_DIMS / 2
}

/// The culling grid: one word per cell (`list start << 8 | count`, the start counted in words
/// from the beginning of the table), followed by the lists of instance indices.
fn build_grid(bounds: &[(Vec3, Vec3)], origin: IVec3) -> Vec<u32> {
    let dims = GRID_DIMS;
    let cell_count = (dims.x * dims.y * dims.z) as usize;
    let base = origin.as_vec3() * CELL_SIZE;
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); cell_count];
    for (i, &bounds) in bounds.iter().enumerate() {
        let (a, b) = cell_range(bounds, origin);
        for z in a.z..=b.z {
            for y in a.y..=b.y {
                for x in a.x..=b.x {
                    lists[(x + dims.x * (y + dims.y * z)) as usize].push(i as u32);
                }
            }
        }
    }
    let mut table = vec![0u32; cell_count];
    for (cell, list) in lists.iter_mut().enumerate() {
        if list.len() > MAX_CELL_INSTANCES {
            let c = IVec3::new(cell as i32 % dims.x, (cell as i32 / dims.x) % dims.y, cell as i32 / (dims.x * dims.y));
            let centre = base + (c.as_vec3() + 0.5) * CELL_SIZE;
            let gap = |i: u32| {
                let (lo, hi) = bounds[i as usize];
                (lo - centre).max(centre - hi).max(Vec3::ZERO).length_squared()
            };
            list.sort_by(|&a, &b| gap(a).total_cmp(&gap(b)).then(a.cmp(&b)));
            list.truncate(MAX_CELL_INSTANCES);
            list.sort_unstable();
        }
        let start = table.len() as u32;
        assert!(start < 1 << 24, "near-field grid lists overflow");
        table[cell] = (start << 8) | list.len() as u32;
        table.extend_from_slice(list);
    }
    table
}

/// GPU resources of the near field.
pub struct NearField {
    /// Records, heightfield heights, then the culling grid (`mesh_sdf_data` in the shaders).
    pub data: wgpu::Buffer,
    /// Volume SDFs (`mesh_sdf_atlas`).
    pub atlas: wgpu::TextureView,
    /// The scene part of `data` (records + heights).
    scene_words: Vec<u32>,
    instances: Vec<NearInstance>,
    placements: Vec<([u32; 3], u32)>,
    bounds: Vec<(Vec3, Vec3)>,
    /// Per scene instance: its record (`u32::MAX`: none).
    record_of: Vec<u32>,
    /// u32 words `data` holds.
    capacity: usize,
    grid: Option<IVec3>,
}

fn atlas_texture(device: &wgpu::Device, size: [u32; 3]) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mesh sdf atlas"),
        size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: size[2] },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: F::R8Snorm,
        usage: U::TEXTURE_BINDING | U::COPY_DST,
        view_formats: &[],
    })
}

impl NearField {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            data: storage_buffer(device, "mesh sdf data", 16, wgpu::BufferUsages::empty()),
            atlas: atlas_texture(device, [1, 1, 1]).create_view(&Default::default()),
            scene_words: Vec::new(),
            instances: Vec::new(),
            placements: Vec::new(),
            bounds: Vec::new(),
            record_of: Vec::new(),
            capacity: 4,
            grid: None,
        }
    }

    /// Rebuilds records and atlas for a changed scene. Instances that cast no shadow (flames)
    /// are left out: they must not occlude the light they represent. `card_instances` holds
    /// every scene instance's index in the mesh-card surface cache (`u32::MAX`: none, which also
    /// leaves out alpha-masked instances too sparse for Lumen), `coverage` the solid fraction of
    /// its surface.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, card_instances: &[u32], coverage: &[f32]) {
        let meshes: Vec<Option<&MeshSdf>> = scene.meshes.iter().map(|m| m.sdf.as_ref()).collect();
        let instances: Vec<NearInstance> = scene
            .instances
            .iter()
            .zip(card_instances.iter().zip(coverage))
            .enumerate()
            .filter(|(_, (i, (&card, _)))| i.cast_shadow && card != u32::MAX)
            .map(|(index, (i, (&card, &coverage)))| NearInstance { index, mesh: i.mesh.0 as usize, world: i.transform, card, coverage })
            .collect();
        let data = build_data(&meshes, &instances);
        let size = data.atlas.size;
        let atlas = atlas_texture(device, size);
        queue.write_texture(
            atlas.as_image_copy(),
            &data.atlas_voxels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size[0]), rows_per_image: Some(size[1]) },
            atlas.size(),
        );
        self.atlas = atlas.create_view(&Default::default());
        log::info!(
            "lumen near field: {} mesh SDF instances, atlas {}x{}x{}",
            data.bounds.len(),
            data.atlas.size[0],
            data.atlas.size[1],
            data.atlas.size[2]
        );
        self.record_of = vec![u32::MAX; scene.instances.len()];
        for (r, instance) in data.instances.iter().enumerate() {
            self.record_of[instance.index] = r as u32;
        }
        self.scene_words = data.words;
        self.instances = data.instances;
        self.placements = data.placements;
        self.bounds = data.bounds;
        self.capacity = 0;
        self.grid = None;
    }

    /// Re-places the records of moved scene instances `moved` (their atlas slots stay); the
    /// culling grid is rebuilt only if one crossed into other cells.
    pub fn move_instances(&mut self, queue: &wgpu::Queue, scene: &Scene, moved: impl Iterator<Item = usize>) {
        for i in moved {
            let Some(&r) = self.record_of.get(i).filter(|&&r| r != u32::MAX) else { continue };
            let r = r as usize;
            let instance = &mut self.instances[r];
            instance.world = scene.instances[i].transform;
            let Some(sdf) = scene.meshes[instance.mesh].sdf.as_ref() else { continue };
            let (atlas_offset, heightfield_word) = self.placements[r];
            let (record, bounds) = instance_record(sdf, instance, atlas_offset, heightfield_word);
            let range = r * RECORD_WORDS..(r + 1) * RECORD_WORDS;
            self.scene_words[range.clone()].copy_from_slice(&record);
            if self.capacity > 0 {
                queue.write_buffer(&self.data, range.start as u64 * 4, bytemuck::cast_slice(&record));
            }
            if let Some(origin) = self.grid {
                if cell_range(bounds, origin) != cell_range(self.bounds[r], origin) {
                    self.grid = None;
                }
            }
            self.bounds[r] = bounds;
        }
    }

    /// Moves the culling grid with the camera (rebuilding its lists when it moved) and returns
    /// the uniform block.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, camera: Vec3) -> NearFieldUniform {
        if self.bounds.is_empty() {
            return NearFieldUniform::default();
        }
        let origin = grid_origin(camera);
        if self.grid != Some(origin) {
            self.grid = Some(origin);
            let grid = build_grid(&self.bounds, origin);
            let cells = (GRID_DIMS.x * GRID_DIMS.y * GRID_DIMS.z) as usize;
            log::debug!(
                "lumen near field: grid at {origin}, {} list entries, longest list {}",
                grid.len() - cells,
                grid[..cells].iter().map(|w| w & 0xff).max().unwrap_or(0)
            );
            let needed = self.scene_words.len() + grid.len();
            if needed > self.capacity {
                self.capacity = needed + needed / 4;
                self.data = storage_buffer(device, "mesh sdf data", self.capacity as u64 * 4, wgpu::BufferUsages::empty());
                queue.write_buffer(&self.data, 0, bytemuck::cast_slice(&self.scene_words));
            }
            queue.write_buffer(&self.data, self.scene_words.len() as u64 * 4, bytemuck::cast_slice(&grid));
        }
        NearFieldUniform {
            grid: (origin.as_vec3() * CELL_SIZE).extend(CELL_SIZE),
            dims: [GRID_DIMS.x as u32, GRID_DIMS.y as u32, GRID_DIMS.z as u32, self.scene_words.len() as u32 / 4],
            trace: Vec4::new(NEAR_FIELD_DISTANCE, 0.0, 0.0, 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helios_geometry::primitives;

    fn soup(mesh: &helios_geometry::Mesh) -> Vec<[Vec3; 3]> {
        mesh.triangles().map(|t| t.map(|i| mesh.vertices[i as usize].position())).collect()
    }

    #[test]
    fn atlas_boxes_do_not_overlap() {
        let sizes: Vec<[u32; 3]> = (0..40u32).map(|i| [8 + (i * 37) % 120, 4 + (i * 53) % 100, 6 + (i * 29) % 90]).collect();
        let layout = pack_atlas(&sizes, 128);
        for (i, (a, sa)) in layout.offsets.iter().zip(&sizes).enumerate() {
            for k in 0..3 {
                assert!(a[k] + sa[k] <= layout.size[k]);
            }
            for (b, sb) in layout.offsets.iter().zip(&sizes).skip(i + 1) {
                let disjoint = (0..3).any(|k| a[k] + sa[k] <= b[k] || b[k] + sb[k] <= a[k]);
                assert!(disjoint, "{a:?} {sa:?} overlaps {b:?} {sb:?}");
            }
        }
        assert!(layout.size[0] <= 128 && layout.size[1] <= 128);
    }

    /// Evaluates a record the way lumen_mesh_sdf.wgsl does (nearest atlas voxel).
    fn record_distance(data: &NearFieldData, index: usize, p: Vec3) -> f32 {
        let r: Vec<f32> = data.words[index * RECORD_WORDS..(index + 1) * RECORD_WORDS].iter().map(|w| f32::from_bits(*w)).collect();
        let sphere = Vec4::from_slice(&r[0..4]);
        assert!(p.distance(sphere.truncate()) <= sphere.w || record_outside(data, index, p));
        let q = Vec3::new(
            Vec4::from_slice(&r[4..8]).dot(p.extend(1.0)),
            Vec4::from_slice(&r[8..12]).dot(p.extend(1.0)),
            Vec4::from_slice(&r[12..16]).dot(p.extend(1.0)),
        );
        let (lo, hi) = (Vec3::from_slice(&r[16..19]), Vec3::from_slice(&r[20..23]));
        let t = q.clamp(lo, hi).floor().as_uvec3();
        let [w, h, _] = data.atlas.size;
        let byte = data.atlas_voxels[(t.x + w * (t.y + h * t.z)) as usize] as i8;
        byte as f32 / 127.0 * r[19]
    }

    /// Whether `p` lies outside the instance's world AABB (and so may lie outside its sphere).
    fn record_outside(data: &NearFieldData, index: usize, p: Vec3) -> bool {
        let (lo, hi) = data.bounds[index];
        p.cmplt(lo).any() || p.cmpgt(hi).any()
    }

    #[test]
    fn records_map_world_points_into_the_atlas() {
        let sphere = MeshSdf::bake_volume(&soup(&primitives::icosphere(3)), 24).unwrap();
        let cylinder = MeshSdf::bake_volume(&soup(&primitives::cylinder(16, 1, 1.0, 2.0, true)), 16).unwrap();
        let meshes = [Some(&cylinder), None, Some(&sphere)];
        // Non-uniformly scaled, rotated and translated sphere; the mesh without SDF is skipped.
        let world = Mat4::from_translation(Vec3::new(10.0, 2.0, -3.0)) * Mat4::from_rotation_y(0.7) * Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0));
        let instance = |mesh, world, card, coverage| NearInstance { index: mesh, mesh, world, card, coverage };
        let data = build_data(&meshes, &[instance(0, Mat4::IDENTITY, 7, 1.0), instance(1, Mat4::IDENTITY, 8, 1.0), instance(2, world, 9, 0.25)]);
        assert_eq!(data.bounds.len(), 2);
        assert_eq!((data.words[28], data.words[RECORD_WORDS + 28]), (7, 9), "surface-cache instances");
        assert_eq!((f32::from_bits(data.words[29]), f32::from_bits(data.words[RECORD_WORDS + 29])), (1.0, 0.25), "coverage");
        let centre = Vec3::new(10.0, 2.0, -3.0);
        assert!(record_distance(&data, 1, centre) < -0.5 * sphere.band, "inside the sphere");
        // Above the centre, along the unscaled y axis: 0.15 outside the unit sphere.
        let above = record_distance(&data, 1, centre + Vec3::Y * 1.15);
        assert!((above - 0.15).abs() < 0.08, "{above}");
        let (lo, hi) = data.bounds[1];
        assert!(lo.x < 8.5 && hi.x > 11.5 && lo.y < 1.0 && hi.y > 3.0, "{lo} {hi}");
        // The cylinder keeps its own atlas slot.
        assert!(record_distance(&data, 0, Vec3::Y) < 0.0);
        assert!(record_distance(&data, 0, Vec3::splat(3.0)) > 0.0);
    }

    #[test]
    fn grid_lists_every_instance_in_reach() {
        let mut bounds = Vec::new();
        for i in 0..200 {
            let p = Vec3::new((i * 7 % 50) as f32 - 25.0, (i * 3 % 11) as f32 - 5.0, (i * 13 % 47) as f32 - 23.0);
            bounds.push((p, p + Vec3::splat(0.3 + (i % 5) as f32)));
        }
        let origin = grid_origin(Vec3::new(1.0, 0.5, -2.0));
        let table = build_grid(&bounds, origin);
        let base = origin.as_vec3() * CELL_SIZE;
        for cell in [0usize, 77, 4000, 8191, 9000, 16383] {
            let dims = GRID_DIMS;
            let c = IVec3::new(cell as i32 % dims.x, (cell as i32 / dims.x) % dims.y, cell as i32 / (dims.x * dims.y));
            let lo = base + c.as_vec3() * CELL_SIZE - CELL_REACH;
            let hi = lo + CELL_SIZE + 2.0 * CELL_REACH;
            let expected: Vec<u32> = (0..bounds.len() as u32).filter(|&i| bounds[i as usize].0.cmple(hi).all() && bounds[i as usize].1.cmpge(lo).all()).collect();
            let word = table[cell];
            let start = (word >> 8) as usize;
            let list = &table[start..start + (word & 0xff) as usize];
            if expected.len() <= MAX_CELL_INSTANCES {
                assert_eq!(list, &expected[..], "cell {cell}");
            } else {
                assert_eq!(list.len(), MAX_CELL_INSTANCES);
            }
        }
    }

    #[test]
    fn grid_moves_in_snapped_steps() {
        let a = grid_origin(Vec3::new(0.1, 0.0, 0.1));
        assert_eq!(a, grid_origin(Vec3::new(CELL_SIZE * GRID_SNAP as f32 - 0.1, 1.0, 3.0)));
        assert_ne!(a, grid_origin(Vec3::new(CELL_SIZE * GRID_SNAP as f32 + 0.1, 0.0, 0.0)));
        // The camera stays well inside the grid.
        let camera = Vec3::new(-37.3, 12.0, 101.0);
        let local = camera / CELL_SIZE - grid_origin(camera).as_vec3();
        assert!(local.cmpge(Vec3::splat(4.0)).all() && local.cmple(GRID_DIMS.as_vec3() - 4.0).all(), "{local}");
    }
}
