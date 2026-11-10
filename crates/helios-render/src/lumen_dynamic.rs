//! Lumen for moving instances: incremental updates of the global distance field, the near field
//! and the mesh cards (UE5 keeps movable meshes in its distance fields and updates them).
//!
//! A move does not rebuild Lumen's scene. Instead:
//!
//! * **Global distance field** — the instance's record (inverse transform, world box) is
//!   rewritten in place. Once any point of it moved [`RECOMPOSE_VOXELS`] of a level's voxels
//!   from where that level last composed it, the level's 8³-voxel bricks under its old and new
//!   box (plus the field's narrow band) are queued for recomposition, along with the per-voxel
//!   closest-instance ids the card lookups use: fine levels follow a swinging chandelier every
//!   few frames, coarse ones rarely. At most [`BRICK_BUDGET`] queued bricks
//!   are recomposed per frame (`cull_listed` / `compose_listed` in lumen_sdf.wgsl), finest level
//!   first; the rest follow in the next frames. A level that is rebuilt anyway (the camera
//!   crossed a brick boundary) drops its queue. Movable instances stay in the field, as in UE5:
//!   a swinging chandelier or a drifting raft occludes and bounces light where it is. Fast movers
//!   only cost more bricks per frame; beyond the budget their fields lag a few frames behind.
//! * **Near field** — the instance's record is rewritten; the culling grid is rebuilt only if
//!   its box crossed into other grid cells.
//! * **Mesh cards** — the card instance's transform is rewritten (cards move with it; nothing is
//!   recaptured) and its pages are relit first in the next card update, so lighting follows.
//! * **Voxel surface cache and radiance cache** — not invalidated: both are refreshed
//!   continuously (a quarter of the voxels and an eighth of the probes per frame) from the
//!   updated field, so they follow within a few frames.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

use crate::motion::SceneChanges;
use crate::profiler::Recorder;
use crate::scene::Scene;
use crate::util::{storage_buffer, uniform_buffer};

/// Bricks recomposed per frame at most (a whole level is 16³ = 4096).
pub(crate) const BRICK_BUDGET: usize = 1024;
/// Bricks per level side (`BRICKS_PER_SIDE` in lumen_sdf.wgsl).
const BRICKS_PER_SIDE: i32 = 16;
const LEVEL_BRICKS: usize = (BRICKS_PER_SIDE * BRICKS_PER_SIDE * BRICKS_PER_SIDE) as usize;
/// Voxels per brick side.
const BRICK: f32 = 8.0;
/// Words per brick table (`BRICK_WORDS`).
const BRICK_WORDS: u64 = 64;
/// Voxels of the field's narrow band (`SDF_BAND_VOXELS` in lumen_types.wgsl).
const BAND_VOXELS: f32 = 4.0;
/// A level recomposes the bricks under a moving instance once any point of it moved this many
/// of the level's voxels since it was last composed there: fine levels follow a swinging
/// chandelier every few frames, coarse ones rarely (a quarter voxel is well below what the
/// field's traces resolve).
const RECOMPOSE_VOXELS: f32 = 0.25;
const MAX_LEVELS: usize = crate::lumen::SDF_LEVELS;

/// `ComposeParams` in lumen_sdf.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ComposeParams {
    pub origin: Vec4,
    pub level: u32,
    pub candidate_count: u32,
    pub level_count: u32,
    /// Listed bricks (incremental updates only).
    pub brick_count: u32,
}

/// `SdfInstance` in lumen_sdf.wgsl: a distance-field instance of the global field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct SdfInstance {
    pub inv_world: Mat4,
    /// w: the smallest axis scale.
    pub world_min: Vec4,
    pub world_max: Vec4,
    pub material: u32,
    pub mesh: u32,
    /// 1 if the instance has mesh cards (a volume field): candidates for the instance index.
    pub has_cards: u32,
    /// 1 + the instance's card pair whose captured material mean the field stores (0: the
    /// material's average; see `CardCache::instance_pairs`).
    pub pair: u32,
}

impl SdfInstance {
    /// Places the record at `world`; returns its world box.
    pub fn place(&mut self, aabb: &helios_geometry::Aabb, world: Mat4) -> (Vec3, Vec3) {
        let (lo, hi) = world_box(aabb, world);
        let linear = glam::Mat3::from_mat4(world);
        let min_scale = linear.x_axis.length().min(linear.y_axis.length()).min(linear.z_axis.length());
        self.inv_world = world.inverse();
        self.world_min = lo.extend(min_scale);
        self.world_max = hi.extend(0.0);
        (lo, hi)
    }
}

/// World-space bounding box of `aabb` transformed by `world`.
pub(crate) fn world_box(aabb: &helios_geometry::Aabb, world: Mat4) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for k in 0..8 {
        let c = Vec3::new(
            if k & 1 == 0 { aabb.min.x } else { aabb.max.x },
            if k & 2 == 0 { aabb.min.y } else { aabb.max.y },
            if k & 4 == 0 { aabb.min.z } else { aabb.max.z },
        );
        let p = world.transform_point3(c);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    (lo, hi)
}

/// The largest distance any point of `aabb` moves between transforms `a` and `b` (the
/// displacement is affine, so its largest length over the box is at a corner).
fn displacement(aabb: &helios_geometry::Aabb, a: Mat4, b: Mat4) -> f32 {
    (0..8)
        .map(|k| {
            let c = Vec3::new(
                if k & 1 == 0 { aabb.min.x } else { aabb.max.x },
                if k & 2 == 0 { aabb.min.y } else { aabb.max.y },
                if k & 4 == 0 { aabb.min.z } else { aabb.max.z },
            );
            a.transform_point3(c).distance(b.transform_point3(c))
        })
        .fold(0.0, f32::max)
}

/// A clipmap level as the incremental updates see it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LevelPlacement {
    pub origin: Vec3,
    pub voxel: f32,
    /// The whole level is rebuilt (now or soon): its brick queue is moot.
    pub rebuilding: bool,
}

/// Bricks of a level touched by the world box `(lo, hi)` (with the band), as index ranges per
/// axis; `None` if it misses the level.
fn brick_range(level: &LevelPlacement, (lo, hi): (Vec3, Vec3)) -> Option<(glam::IVec3, glam::IVec3)> {
    let band = level.voxel * BAND_VOXELS;
    let size = level.voxel * BRICK;
    let a = ((lo - band - level.origin) / size).floor().as_ivec3().max(glam::IVec3::ZERO);
    let b = ((hi + band - level.origin) / size).floor().as_ivec3().min(glam::IVec3::splat(BRICKS_PER_SIDE - 1));
    a.cmple(b).all().then_some((a, b))
}

/// Per level: queued bricks and the GPU lists of this frame's batch.
struct LevelQueue {
    queued: Vec<u32>,
    /// One bit per brick: queued already.
    bits: Vec<u64>,
    params: wgpu::Buffer,
    list: wgpu::Buffer,
    tables: wgpu::Buffer,
    candidates: wgpu::Buffer,
    /// Bricks in this frame's batch.
    batch: u32,
}

impl LevelQueue {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            queued: Vec::new(),
            bits: vec![0; LEVEL_BRICKS / 64],
            params: uniform_buffer(device, "sdf brick params", std::mem::size_of::<ComposeParams>() as u64),
            list: storage_buffer(device, "sdf brick list", BRICK_BUDGET as u64 * 4, wgpu::BufferUsages::empty()),
            tables: storage_buffer(device, "sdf brick tables", BRICK_BUDGET as u64 * BRICK_WORDS * 4, wgpu::BufferUsages::empty()),
            candidates: storage_buffer(device, "sdf brick candidates", 16, wgpu::BufferUsages::empty()),
            batch: 0,
        }
    }

    fn push(&mut self, brick: u32) {
        let (word, bit) = (brick as usize / 64, 1u64 << (brick % 64));
        if self.bits[word] & bit == 0 {
            self.bits[word] |= bit;
            self.queued.push(brick);
        }
    }

    fn clear(&mut self) {
        self.queued.clear();
        self.bits.fill(0);
    }
}

/// The global field's resources the brick passes bind besides their own.
pub(crate) struct FieldBindings<'a> {
    pub sdf_instances: &'a wgpu::Buffer,
    pub descs: &'a wgpu::Buffer,
    pub words: &'a wgpu::Buffer,
    pub materials: &'a wgpu::Buffer,
    pub albedo: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub sdf: &'a wgpu::TextureView,
    pub instance_ids: &'a wgpu::TextureView,
    pub pair_sums: &'a wgpu::Buffer,
}

/// Counts of the latest frame (HUD, measurements).
#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicFieldStats {
    /// Distance-field instances moved this frame.
    pub moved: u32,
    /// Global distance field bricks recomposed this frame, and still queued after it.
    pub bricks: u32,
    pub queued: u32,
    /// Clipmap levels rebuilt in full this frame (camera movement, structural changes).
    pub levels: u32,
}

/// Incremental distance-field updates (see the module docs).
pub(crate) struct DynamicField {
    /// The distance-field instance records (CPU copy of `sdf_instances`).
    pub records: Vec<SdfInstance>,
    /// Per scene instance: its distance-field instance (`u32::MAX`: none).
    pub index: Vec<u32>,
    levels: Vec<LevelQueue>,
    /// Per moved distance-field instance in the global field: per level, the transform it was
    /// last composed with.
    composed: HashMap<usize, [Mat4; MAX_LEVELS]>,
    /// Per level and brick: recomposed this frame (debug view).
    pub stamps: wgpu::Buffer,
    stamped: bool,
    cull: crate::lumen::Stage,
    compose: crate::lumen::Stage,
    pub stats: DynamicFieldStats,
}

impl DynamicField {
    pub fn new(device: &wgpu::Device, module: &wgpu::ShaderModule, levels: usize) -> Self {
        use crate::util::bind;
        let (u, ro, rw) = (bind::uniform, || bind::storage(true), || bind::storage(false));
        Self {
            records: Vec::new(),
            index: Vec::new(),
            levels: (0..levels).map(|_| LevelQueue::new(device)).collect(),
            composed: HashMap::new(),
            stamps: storage_buffer(device, "sdf brick stamps", (levels * LEVEL_BRICKS * 4) as u64, wgpu::BufferUsages::empty()),
            stamped: true,
            cull: crate::lumen::Stage::new(device, module, "cull_listed", &[(0, u()), (1, ro()), (4, ro()), (5, rw()), (10, ro())]),
            compose: crate::lumen::Stage::new(
                device,
                module,
                "compose_listed",
                &[
                    (0, u()),
                    (1, ro()),
                    (2, ro()),
                    (3, ro()),
                    (5, rw()),
                    (6, u()),
                    (7, bind::texture_array(true)),
                    (8, bind::sampler()),
                    (9, bind::storage_texture_3d(wgpu::TextureFormat::Rgba8Snorm)),
                    (10, ro()),
                    (11, rw()),
                    (12, ro()),
                    (70, bind::storage_texture_3d(wgpu::TextureFormat::R32Uint)),
                ],
            ),
            stats: DynamicFieldStats::default(),
        }
    }

    /// Takes the records of a rebuilt scene (`indices`: their scene instances); drops every queue.
    pub fn reset(&mut self, records: Vec<SdfInstance>, indices: &[usize], instance_count: usize) {
        self.records = records;
        self.index = vec![u32::MAX; instance_count];
        for (k, &i) in indices.iter().enumerate() {
            self.index[i] = k as u32;
        }
        for level in &mut self.levels {
            level.clear();
        }
        self.composed.clear();
    }

    /// Moves the distance-field instances `changes` moved: rewrites their records (written to
    /// `sdf_instances`) and `bounds` (from their fields' mesh-space boxes `aabbs`), and queues the bricks under their old and new boxes in the
    /// levels of the global field they belong to (`in_global`), once they moved far enough from
    /// where a level last composed them. Returns the moved instances' (scene index,
    /// distance-field index).
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        queue: &wgpu::Queue,
        scene: &Scene,
        changes: &SceneChanges,
        sdf_instances: &wgpu::Buffer,
        aabbs: &[helios_geometry::Aabb],
        bounds: &mut [(Vec3, Vec3)],
        in_global: &[bool],
        levels: &[LevelPlacement],
    ) -> Vec<(usize, usize)> {
        let mut moved = Vec::new();
        for m in changes.motions(scene) {
            let i = m.index as usize;
            let Some(&k) = self.index.get(i).filter(|&&k| k != u32::MAX) else { continue };
            let k = k as usize;
            let instance = &scene.instances[i];
            let aabb = &aabbs[k];
            bounds[k] = self.records[k].place(aabb, instance.transform);
            let size = std::mem::size_of::<SdfInstance>();
            queue.write_buffer(sdf_instances, (k * size) as u64, bytemuck::bytes_of(&self.records[k]));
            moved.push((i, k));
            if !in_global[k] {
                continue;
            }
            let composed = self.composed.entry(k).or_insert([m.previous; MAX_LEVELS]);
            for ((queue, level), pose) in self.levels.iter_mut().zip(levels).zip(composed.iter_mut()) {
                if level.rebuilding {
                    *pose = instance.transform;
                    continue;
                }
                // The level holds the instance where it was last composed: recompose once it
                // moved a fraction of a voxel from there.
                if displacement(aabb, *pose, instance.transform) < level.voxel * RECOMPOSE_VOXELS {
                    continue;
                }
                let old = world_box(aabb, std::mem::replace(pose, instance.transform));
                for (a, b) in [old, bounds[k]].into_iter().filter_map(|b| brick_range(level, b)) {
                    for z in a.z..=b.z {
                        for y in a.y..=b.y {
                            for x in a.x..=b.x {
                                queue.push((x + BRICKS_PER_SIDE * (y + BRICKS_PER_SIDE * z)) as u32);
                            }
                        }
                    }
                }
            }
        }
        self.stats.moved = moved.len() as u32;
        moved
    }

    /// Picks this frame's bricks (finest level first, within [`BRICK_BUDGET`]) and uploads their
    /// lists, with the global-field instances near them as candidates. Levels being rebuilt drop
    /// their queues.
    pub fn schedule(&mut self, queue: &wgpu::Queue, device: &wgpu::Device, levels: &[LevelPlacement], bounds: &[(Vec3, Vec3)], in_global: &[bool]) {
        let mut budget = BRICK_BUDGET;
        let mut total = 0;
        for (index, (q, level)) in self.levels.iter_mut().zip(levels).enumerate() {
            q.batch = 0;
            if level.rebuilding {
                q.clear();
                continue;
            }
            if q.queued.is_empty() || budget == 0 {
                continue;
            }
            let take = q.queued.len().min(budget);
            budget -= take;
            let batch: Vec<u32> = q.queued.drain(..take).collect();
            for &b in &batch {
                q.bits[b as usize / 64] &= !(1u64 << (b % 64));
            }
            // Candidates: the global-field instances touching the batch's box (band included).
            let size = level.voxel * BRICK;
            let band = Vec3::splat(level.voxel * BAND_VOXELS);
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for &b in &batch {
                let b = b as i32;
                let coords = glam::IVec3::new(b % BRICKS_PER_SIDE, (b / BRICKS_PER_SIDE) % BRICKS_PER_SIDE, b / (BRICKS_PER_SIDE * BRICKS_PER_SIDE));
                lo = lo.min(level.origin + coords.as_vec3() * size);
                hi = hi.max(level.origin + (coords + 1).as_vec3() * size);
            }
            let (lo, hi) = (lo - band, hi + band);
            let candidates: Vec<u32> = bounds
                .iter()
                .zip(in_global)
                .enumerate()
                .filter(|(_, ((a, b), &global))| global && a.cmple(hi).all() && b.cmpge(lo).all())
                .map(|(k, _)| k as u32)
                .collect();
            let bytes = (candidates.len() as u64 * 4).max(16);
            if q.candidates.size() < bytes {
                q.candidates = storage_buffer(device, "sdf brick candidates", bytes.next_power_of_two(), wgpu::BufferUsages::empty());
            }
            queue.write_buffer(&q.candidates, 0, bytemuck::cast_slice(&candidates));
            queue.write_buffer(&q.list, 0, bytemuck::cast_slice(&batch));
            let params = ComposeParams {
                origin: level.origin.extend(level.voxel),
                level: index as u32,
                candidate_count: candidates.len() as u32,
                level_count: levels.len() as u32,
                brick_count: batch.len() as u32,
            };
            queue.write_buffer(&q.params, 0, bytemuck::bytes_of(&params));
            q.batch = batch.len() as u32;
            total += batch.len();
        }
        self.stats.bricks = total as u32;
        self.stats.queued = self.levels.iter().map(|q| q.queued.len() as u32).sum();
    }

    /// Records this frame's brick recomposition (after any full level rebuilds).
    pub fn record(&mut self, rec: &mut Recorder, field: &FieldBindings) {
        let stamp = self.levels.iter().any(|q| q.batch > 0);
        if std::mem::replace(&mut self.stamped, stamp) {
            rec.encoder.clear_buffer(&self.stamps, 0, None);
        }
        if !stamp {
            return;
        }
        let device = rec.device;
        let mut pass = rec.compute_pass("lumen sdf bricks");
        for q in self.levels.iter().filter(|q| q.batch > 0) {
            let cull = self.cull.group(
                device,
                &[
                    (0, q.params.as_entire_binding()),
                    (1, field.sdf_instances.as_entire_binding()),
                    (4, q.candidates.as_entire_binding()),
                    (5, q.tables.as_entire_binding()),
                    (10, q.list.as_entire_binding()),
                ],
            );
            pass.set_pipeline(&self.cull.pipeline);
            pass.set_bind_group(0, &cull, &[]);
            pass.dispatch_workgroups(crate::util::workgroups(q.batch, 64), 1, 1);
            let compose = self.compose.group(
                device,
                &[
                    (0, q.params.as_entire_binding()),
                    (1, field.sdf_instances.as_entire_binding()),
                    (2, field.descs.as_entire_binding()),
                    (3, field.words.as_entire_binding()),
                    (5, q.tables.as_entire_binding()),
                    (6, field.materials.as_entire_binding()),
                    (7, wgpu::BindingResource::TextureView(field.albedo)),
                    (8, wgpu::BindingResource::Sampler(field.sampler)),
                    (9, wgpu::BindingResource::TextureView(field.sdf)),
                    (10, q.list.as_entire_binding()),
                    (11, self.stamps.as_entire_binding()),
                    (12, field.pair_sums.as_entire_binding()),
                    (70, wgpu::BindingResource::TextureView(field.instance_ids)),
                ],
            );
            pass.set_pipeline(&self.compose.pipeline);
            pass.set_bind_group(0, &compose, &[]);
            pass.dispatch_workgroups(q.batch, 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brick_ranges_cover_the_band_and_clip_to_the_level() {
        let level = LevelPlacement { origin: Vec3::ZERO, voxel: 0.25, rebuilding: false };
        // Bricks are 2 m, the band 1 m: a box at (4.2..4.8) touches bricks 1..2 along each axis.
        let (a, b) = brick_range(&level, (Vec3::splat(4.2), Vec3::splat(4.8))).unwrap();
        assert_eq!((a, b), (glam::IVec3::splat(1), glam::IVec3::splat(2)));
        // Clipped at the level's edge, missed entirely outside it.
        let (a, b) = brick_range(&level, (Vec3::splat(-5.0), Vec3::splat(0.5))).unwrap();
        assert_eq!((a, b), (glam::IVec3::ZERO, glam::IVec3::ZERO));
        assert!(brick_range(&level, (Vec3::splat(-9.0), Vec3::splat(-2.0))).is_none());
        assert!(brick_range(&level, (Vec3::splat(40.0), Vec3::splat(41.0))).is_none());
    }

    #[test]
    fn displacement_bounds_every_point_of_the_box() {
        let aabb = helios_geometry::Aabb { min: Vec3::new(-1.0, 0.0, -1.0), max: Vec3::new(1.0, 12.0, 1.0) };
        let shift = Mat4::from_translation(Vec3::new(0.3, 0.0, 0.4));
        assert!((displacement(&aabb, Mat4::IDENTITY, shift) - 0.5).abs() < 1e-5);
        // A pendulum swing about the top: the bottom corners move the most.
        let pivot = Vec3::new(0.0, 12.0, 0.0);
        let swing = Mat4::from_translation(pivot) * Mat4::from_rotation_z(0.05) * Mat4::from_translation(-pivot);
        let bottom = Vec3::new(1.0, 0.0, 1.0);
        let d = displacement(&aabb, Mat4::IDENTITY, swing);
        assert!(d >= swing.transform_point3(bottom).distance(bottom) - 1e-5 && d < 12.1 * 0.05);
        assert_eq!(displacement(&aabb, swing, swing), 0.0);
    }

    #[test]
    fn records_follow_their_instance() {
        let aabb = helios_geometry::Aabb { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) };
        let mut record = SdfInstance::zeroed();
        let world = Mat4::from_scale_rotation_translation(Vec3::new(2.0, 1.0, 1.0), glam::Quat::from_rotation_y(0.5), Vec3::new(5.0, 0.0, 0.0));
        let (lo, hi) = record.place(&aabb, world);
        assert_eq!((record.world_min.truncate(), record.world_max.truncate()), (lo, hi));
        assert!((record.world_min.w - 1.0).abs() < 1e-5, "smallest scale");
        let p = Vec3::new(5.3, 0.2, -0.1);
        assert!(record.inv_world.transform_point3(world.transform_point3(p)).distance(p) < 1e-4);
        assert!(lo.cmple(Vec3::new(5.0, -1.0, 0.0)).all() && hi.cmpge(Vec3::new(5.0, 1.0, 0.0)).all());
    }
}
