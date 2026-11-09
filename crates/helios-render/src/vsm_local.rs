//! Virtual shadow maps for local lights (see `shaders/vsm_local_common.wgsl`).
//!
//! As in UE5, point and spot lights are rendered into virtual shadow maps too: a point light
//! gets six cube faces, a spot light one perspective face, each a 4096² virtual shadow map with
//! a mip chain. Pixels request the pages of the mip matching their screen footprint, for every
//! light of their light cluster that has a shadow map.
//!
//! The pages are entries of the sun's page cache placed after its clipmap levels and live in the
//! same physical atlas. Sharing the pool means one allocation, one dirty list and one page
//! renderer for all lights — the sun's culling, (cluster, page) expansion and hardware + software
//! page rasterisers draw local pages too, only with a perspective projection per face — and
//! physical pages go wherever they are needed, between the sun and the lights.
//!
//! Each frame up to [`LOCAL_SLOTS`] shadow-casting lights are given a *slot*, prioritised by
//! their influence on the screen; the others keep distance-field shadows. A light keeps its slot
//! (and its cached pages) while it stays selected; its pages are dropped when it moves or
//! changes shape, when the scene changes, or when the slot passes to another light.
//!
//! One culling view covers every light, face and mip, so the cost of a frame without new pages
//! does not grow with the number of lights. It keeps clusters that overlap a world box around
//! some light's dirty pages and selects LOD for the most demanding light's finest dirty mip
//! (`cull.wgsl`); the expansion then distributes each cluster to the dirty pages of the lights it
//! overlaps (`vsm_expand.wgsl`).

use bytemuck::{Pod, Zeroable};
use glam::camera::rh::{proj::directx, view::look_to_mat4};
use glam::{Mat4, Vec3, Vec4};

use crate::gpu_types::CullViewUniform;
use crate::passes::culling::{CullCapacity, CullView};
use crate::scene::{Light, LightKind};
use crate::util::uniform_buffer;

/// Lights with a virtual shadow map per frame.
pub const LOCAL_SLOTS: usize = 32;
const FACES: usize = 6;
/// Mips per face: 32² pages at mip 0 down to one page.
const MIPS: u32 = 6;
/// Page cache entries of one face (all mips) and of all slots.
const FACE_ENTRIES: u32 = 1365;
pub const ENTRIES: u32 = LOCAL_SLOTS as u32 * FACES as u32 * FACE_ENTRIES;
/// Face mips of all slots (each has a dirty page rectangle in the VSM info buffer).
pub const FACE_MIPS: u32 = LOCAL_SLOTS as u32 * FACES as u32 * MIPS;
/// Draw levels from here on are local light pages (`LOCAL_LEVEL` in the shaders).
pub const LOCAL_LEVEL: u32 = 16;
/// World units of the fixed-point dirty-page boxes the culling reads (vsm.wgsl).
const BOX_UNIT: f32 = 1.0 / 32.0;
/// Spot lights up to this cone half-angle get one perspective face; wider ones a cube.
const MAX_SPOT_HALF_ANGLE: f32 = 1.05;
/// Margin around the spot cone, so filter taps at its rim stay inside the face.
const SPOT_MARGIN: f32 = 0.05;
/// Lights whose source is larger than this cast soft shadows that need less resolution: each
/// doubling of the radius coarsens their mips by one.
const SHARP_SOURCE_RADIUS: f32 = 0.25;
/// A light that holds a slot keeps it against newcomers up to this much more influential, so
/// slots (and their cached pages) do not flip between lights of similar importance.
const KEEP_BONUS: f32 = 1.5;
/// A light that moved within this many frames counts as moving (see [`SlotAllocator::assign`]).
const MOVING_FRAMES: u32 = 30;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuSlot {
    view_proj: [Mat4; FACES],
    inv_view_proj: [Mat4; FACES],
    position_range: Vec4,
    /// Face count (0 unused), near plane, focal length, mip bias.
    params: Vec4,
    /// x: cached pages are stale.
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LocalVsmUniform {
    slots: [GpuSlot; LOCAL_SLOTS],
    entry_base: u32,
    enabled: u32,
    lod_bias: f32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<GpuSlot>() == 816);
const _: () = assert!(std::mem::size_of::<LocalVsmUniform>() == 816 * LOCAL_SLOTS + 16);

/// Influence of a light on the screen, for choosing which lights get shadow maps: the share of
/// the view its range sphere covers (1 with the camera inside it) times its brightness. Zero for
/// lights that cast no shadows or whose sphere is outside the view frustum (`planes` point inside).
pub fn screen_influence(light: &Light, eye: Vec3, planes: &[Vec4; 6]) -> f32 {
    if !light.cast_shadows || light.range <= 0.0 {
        return 0.0;
    }
    if planes.iter().any(|p| p.truncate().dot(light.position) + p.w < -light.range) {
        return 0.0;
    }
    let distance = light.position.distance(eye).max(light.range);
    let coverage = (light.range / distance).powi(2);
    let brightness = light.color.dot(Vec3::new(0.2126, 0.7152, 0.0722)) * light.intensity;
    coverage * brightness.max(1e-6)
}

/// The parts of a light its shadow map depends on (not its colour or intensity, which may
/// flicker without invalidating anything).
fn same_shadow(a: &Light, b: &Light) -> bool {
    let kind = match (a.kind, b.kind) {
        (LightKind::Point, LightKind::Point) => true,
        (LightKind::Spot { direction: da, outer: oa, .. }, LightKind::Spot { direction: db, outer: ob, .. }) => da == db && oa == ob,
        _ => false,
    };
    kind && a.position == b.position && a.range == b.range && a.source_radius == b.source_radius
}

/// Slot assignment of one frame.
#[derive(Debug, Default, PartialEq)]
pub struct Assignment {
    /// Per light: its slot, or `u32::MAX`.
    pub light_slots: Vec<u32>,
    /// Slots whose cached pages must be dropped before this frame's requests.
    pub stale: [bool; LOCAL_SLOTS],
}

/// Keeps lights in shadow-map slots across frames (pure CPU logic).
#[derive(Debug, Default)]
pub struct SlotAllocator {
    /// Light index and the light as it was when its pages were rendered.
    slots: [Option<(usize, Light)>; LOCAL_SLOTS],
}

impl SlotAllocator {
    /// Gives slots to the most influential lights (`influence[i] > 0` wants one), static ones
    /// before `moving` ones: a moving light's cached pages are stale every frame, so its shadow
    /// map costs a full re-render per frame (hundreds of pages for a cube) where a static light's
    /// costs nothing once cached. Moving lights get the slots left over (with few lights, all of
    /// them); without one they keep their traced shadows (MegaLights, or distance fields in the
    /// clustered loop). Lights keep their slot while selected; `invalidate_all` drops every
    /// cached page (scene change).
    pub fn assign(&mut self, lights: &[Light], influence: &[f32], moving: &[bool], invalidate_all: bool) -> Assignment {
        let held = |i: usize| self.slots.iter().flatten().any(|&(light, _)| light == i);
        let mut candidates: Vec<(usize, f32)> = influence
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v > 0.0)
            .map(|(i, &v)| (i, if held(i) { v * KEEP_BONUS } else { v }))
            .collect();
        let moves = |i: usize| moving.get(i).copied().unwrap_or(false);
        candidates.sort_by(|a, b| moves(a.0).cmp(&moves(b.0)).then(b.1.total_cmp(&a.1)).then(a.0.cmp(&b.0)));
        candidates.truncate(LOCAL_SLOTS);
        let selected = |i: usize| candidates.iter().any(|&(c, _)| c == i);

        let mut out = Assignment { light_slots: vec![u32::MAX; lights.len()], stale: [false; LOCAL_SLOTS] };
        for (s, slot) in self.slots.iter_mut().enumerate() {
            match slot {
                Some((i, light)) if *i < lights.len() && selected(*i) => {
                    if invalidate_all || !same_shadow(light, &lights[*i]) {
                        out.stale[s] = true;
                        *light = lights[*i];
                    }
                    out.light_slots[*i] = s as u32;
                }
                Some(_) => *slot = None,
                None => {}
            }
        }
        for &(i, _) in &candidates {
            if out.light_slots[i] != u32::MAX {
                continue;
            }
            let free = self.slots.iter().position(Option::is_none).expect("at most LOCAL_SLOTS candidates");
            self.slots[free] = Some((i, lights[i]));
            out.light_slots[i] = free as u32;
            out.stale[free] = true;
        }
        out
    }

    /// Slots in use and their lights.
    pub fn occupied(&self) -> impl Iterator<Item = (usize, &Light)> {
        self.slots.iter().enumerate().filter_map(|(s, slot)| slot.as_ref().map(|(_, light)| (s, light)))
    }
}

/// Which lights moved recently (light positions over the frames).
#[derive(Debug, Default)]
struct LightMotion {
    /// Per light: its position last frame and the frames since it last moved.
    last: Vec<(Vec3, u32)>,
}

impl LightMotion {
    /// Per light of this frame: whether it moved in the last [`MOVING_FRAMES`] frames. A new
    /// light list (another scene) starts at rest.
    fn update(&mut self, lights: &[Light]) -> Vec<bool> {
        if self.last.len() != lights.len() {
            self.last = lights.iter().map(|l| (l.position, MOVING_FRAMES)).collect();
        }
        self.last
            .iter_mut()
            .zip(lights)
            .map(|((position, since), light)| {
                *since = if *position == light.position { since.saturating_add(1) } else { 0 };
                *position = light.position;
                *since < MOVING_FRAMES
            })
            .collect()
    }
}

/// Projections of a light's faces: (view-projections, near plane, focal length).
fn face_projections(light: &Light) -> (Vec<Mat4>, f32, f32) {
    let near = (light.source_radius * 0.5).clamp(0.02, 0.25);
    match light.kind {
        LightKind::Spot { direction, outer, .. } if outer <= MAX_SPOT_HALF_ANGLE => {
            let half = outer + SPOT_MARGIN;
            let up = if direction.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
            let proj = directx::perspective_infinite_reverse(2.0 * half, 1.0, near);
            (vec![proj * look_to_mat4(light.position, direction, up)], near, 1.0 / half.tan())
        }
        _ => {
            let proj = directx::perspective_infinite_reverse(std::f32::consts::FRAC_PI_2, 1.0, near);
            // Face order +X, -X, +Y, -Y, +Z, -Z (local_face in vsm_local_common.wgsl).
            let faces = [
                (Vec3::X, Vec3::Y),
                (Vec3::NEG_X, Vec3::Y),
                (Vec3::Y, Vec3::Z),
                (Vec3::NEG_Y, Vec3::Z),
                (Vec3::Z, Vec3::Y),
                (Vec3::NEG_Z, Vec3::Y),
            ];
            (faces.iter().map(|&(dir, up)| proj * look_to_mat4(light.position, dir, up)).collect(), near, 1.0)
        }
    }
}

/// Per-frame inputs of [`LocalShadows::update`].
pub struct LocalFrame<'a> {
    pub lights: &'a [Light],
    pub eye: Vec3,
    /// Frustum planes of the camera (pointing inside).
    pub planes: [Vec4; 6],
    pub enabled: bool,
    /// Geometry or instances changed: every cached local page is stale.
    pub scene_changed: bool,
}

/// Per-frame culling parameters of the local lights' view.
pub struct LocalCull {
    pub instance_count: u32,
    pub frame: u32,
    pub lod_threshold: f32,
    pub sw_raster_threshold: f32,
}

pub struct LocalShadows {
    pub uniform: wgpu::Buffer,
    allocator: SlotAllocator,
    motion: LightMotion,
    gpu: Box<LocalVsmUniform>,
    /// The culling view of all local lights.
    view: CullView,
    /// Per light of the scene: its slot or `u32::MAX` (uploaded with the lights).
    light_slots: Vec<u32>,
    active: usize,
    any_stale: bool,
    info_base: u32,
}

impl LocalShadows {
    /// `entry_base`: the page cache entry of slot 0's first page; `info_base`: the word of the
    /// VSM info buffer where the slots' dirty-page data starts (read by the culling).
    pub fn new(device: &wgpu::Device, entry_base: u32, info_base: u32, capacity: CullCapacity) -> Self {
        let mut gpu: Box<LocalVsmUniform> = bytemuck::allocation::zeroed_box();
        gpu.entry_base = entry_base;
        Self {
            uniform: uniform_buffer(device, "local vsm", std::mem::size_of::<LocalVsmUniform>() as u64),
            allocator: SlotAllocator::default(),
            motion: LightMotion::default(),
            gpu,
            view: CullView::new(device, "local vsm", capacity),
            light_slots: Vec::new(),
            active: 0,
            any_stale: false,
            info_base,
        }
    }

    /// Chooses this frame's shadowed lights. Call before uploading the lights.
    pub fn assign(&mut self, f: &LocalFrame) {
        let influence: Vec<f32> =
            f.lights.iter().map(|l| if f.enabled { screen_influence(l, f.eye, &f.planes) } else { 0.0 }).collect();
        let moving = self.motion.update(f.lights);
        let assignment = self.allocator.assign(f.lights, &influence, &moving, f.scene_changed);
        self.light_slots = assignment.light_slots;
        self.any_stale = assignment.stale.iter().any(|&s| s);
        let gpu = &mut *self.gpu;
        gpu.enabled = f.enabled as u32;
        for (s, slot) in gpu.slots.iter_mut().enumerate() {
            *slot = GpuSlot::zeroed();
            slot.flags[0] = assignment.stale[s] as u32;
        }
        self.active = 0;
        for (s, light) in self.allocator.occupied() {
            let (faces, near, focal) = face_projections(light);
            let slot = &mut gpu.slots[s];
            for (i, view_proj) in faces.iter().enumerate() {
                slot.view_proj[i] = *view_proj;
                slot.inv_view_proj[i] = view_proj.inverse();
            }
            let mip_bias = (light.source_radius / SHARP_SOURCE_RADIUS).log2().max(0.0);
            slot.position_range = light.position.extend(light.range);
            slot.params = Vec4::new(faces.len() as f32, near, focal, mip_bias);
            self.active += 1;
        }
    }

    /// Per light of the scene: its shadow map slot, or `u32::MAX`.
    pub fn light_slots(&self) -> &[u32] {
        &self.light_slots
    }

    /// Writes the uniforms of the lights and of their culling view.
    pub fn update(&self, queue: &wgpu::Queue, lod_bias: f32, cull: &LocalCull) {
        let mut gpu = *self.gpu;
        gpu.lod_bias = lod_bias;
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&gpu));
        // Light positions, LOD scales and dirty-page boxes come from the info buffer
        // (cull.wgsl); the frustum test passes everything.
        let uniform = CullViewUniform {
            view_proj: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            planes: [Vec4::W; 6],
            position: Vec4::ZERO,
            viewport: Vec4::ONE,
            lod_scale: 0.0,
            lod_threshold: cull.lod_threshold,
            orthographic: 0,
            instance_count: cull.instance_count,
            work_capacity: self.view.capacity.pages,
            cluster_capacity: self.view.capacity.clusters,
            hzb_mips: 1,
            frame_index: cull.frame,
            sw_raster_threshold: cull.sw_raster_threshold,
            kind: 1,
            batch_capacity: self.view.capacity.batches,
            masked_capacity: 0,
            vsm: Vec4::new(self.info_base as f32, BOX_UNIT, LOCAL_SLOTS as f32, 0.0),
        };
        queue.write_buffer(&self.view.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// The culling view of the local lights, if any has a shadow map this frame.
    pub fn view(&self) -> Option<&CullView> {
        (self.active > 0).then_some(&self.view)
    }

    /// Lights with a shadow map this frame.
    pub fn active_count(&self) -> usize {
        self.active
    }

    /// Some slot's cached pages must be dropped this frame.
    pub fn any_stale(&self) -> bool {
        self.any_stale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lights(n: usize) -> Vec<Light> {
        (0..n).map(|i| Light::point(Vec3::new(i as f32, 1.0, 0.0), Vec3::ONE, 1.0, 5.0)).collect()
    }

    #[test]
    fn keeps_slots_of_static_lights() {
        let lights = lights(4);
        let mut a = SlotAllocator::default();
        let first = a.assign(&lights, &[1.0, 2.0, 0.0, 3.0], &[], false);
        assert_eq!(first.light_slots[2], u32::MAX);
        assert_eq!(first.stale.iter().filter(|&&s| s).count(), 3);
        let second = a.assign(&lights, &[1.0, 2.0, 0.0, 3.0], &[], false);
        assert_eq!(second.light_slots, first.light_slots);
        assert!(second.stale.iter().all(|&s| !s), "static lights keep their pages");
    }

    #[test]
    fn moving_or_changed_lights_invalidate_only_their_slot() {
        let mut lights = lights(3);
        let mut a = SlotAllocator::default();
        let first = a.assign(&lights, &[1.0; 3], &[], false);
        lights[1].position.y += 0.5;
        lights[2].intensity = 7.0; // brightness does not change the shadow
        let second = a.assign(&lights, &[1.0; 3], &[], false);
        let stale: Vec<usize> = (0..LOCAL_SLOTS).filter(|&s| second.stale[s]).collect();
        assert_eq!(stale, vec![first.light_slots[1] as usize]);
        let third = a.assign(&lights, &[1.0; 3], &[], true);
        assert_eq!(third.stale.iter().filter(|&&s| s).count(), 3, "scene changes drop every page");
    }

    #[test]
    fn budget_prefers_influential_lights_with_hysteresis() {
        let lights = lights(LOCAL_SLOTS + 2);
        let mut influence: Vec<f32> = (0..lights.len()).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut a = SlotAllocator::default();
        let first = a.assign(&lights, &influence, &[], false);
        assert_eq!(first.light_slots[0], u32::MAX);
        assert_eq!(first.light_slots[1], u32::MAX);
        assert_eq!(first.light_slots.iter().filter(|&&s| s != u32::MAX).count(), LOCAL_SLOTS);
        // Slightly more influential than a holder: not enough to take its slot.
        influence[0] = 1.2;
        let second = a.assign(&lights, &influence, &[], false);
        assert_eq!(second.light_slots[0], u32::MAX);
        // Much more influential: takes the slot of the least influential holder (light 2).
        influence[0] = 10.0;
        let third = a.assign(&lights, &influence, &[], false);
        assert_eq!(third.light_slots[0], first.light_slots[2]);
        assert_eq!(third.light_slots[2], u32::MAX);
        assert!(third.stale[third.light_slots[0] as usize]);
        assert_eq!(third.stale.iter().filter(|&&s| s).count(), 1);
    }

    #[test]
    fn influence_ignores_lights_outside_the_view_or_without_shadows() {
        // A frustum that only keeps x >= 0.
        let mut planes = [Vec4::new(0.0, 0.0, 0.0, 1.0); 6];
        planes[0] = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let mut light = Light::point(Vec3::new(-10.0, 0.0, 0.0), Vec3::ONE, 1.0, 5.0);
        assert_eq!(screen_influence(&light, Vec3::ZERO, &planes), 0.0);
        light.position.x = -3.0; // its range reaches into the view
        assert!(screen_influence(&light, Vec3::ZERO, &planes) > 0.0);
        light.cast_shadows = false;
        assert_eq!(screen_influence(&light, Vec3::ZERO, &planes), 0.0);
    }

    #[test]
    fn moving_lights_get_the_slots_left_over() {
        let lights = lights(LOCAL_SLOTS + 1);
        // The most influential light moves: every static one is served first.
        let influence: Vec<f32> = (0..lights.len()).map(|i| if i == 0 { 100.0 } else { 1.0 }).collect();
        let mut moving = vec![false; lights.len()];
        moving[0] = true;
        let mut a = SlotAllocator::default();
        let crowded = a.assign(&lights, &influence, &moving, false);
        assert_eq!(crowded.light_slots[0], u32::MAX);
        // With a slot to spare it gets one.
        let mut a = SlotAllocator::default();
        let spare = a.assign(&lights[..LOCAL_SLOTS], &influence[..LOCAL_SLOTS], &moving, false);
        assert_ne!(spare.light_slots[0], u32::MAX);
    }

    #[test]
    fn lights_count_as_moving_until_they_rest_a_while() {
        let mut lights = lights(2);
        let mut motion = LightMotion::default();
        assert_eq!(motion.update(&lights), vec![false, false], "a new scene starts at rest");
        lights[1].position.x += 0.1;
        assert_eq!(motion.update(&lights), vec![false, true]);
        for _ in 1..MOVING_FRAMES {
            assert_eq!(motion.update(&lights), vec![false, true]);
        }
        assert_eq!(motion.update(&lights), vec![false, false], "at rest again");
    }

    #[test]
    fn face_entries_hold_every_mip() {
        // 32² pages at mip 0 down to one page (LOCAL_FACE_ENTRIES and local_mip_offset in
        // vsm_local_common.wgsl).
        let pages: u32 = (0..MIPS).map(|mip| (32u32 >> mip).pow(2)).sum();
        assert_eq!(pages, FACE_ENTRIES);
        assert_eq!(32 >> (MIPS - 1), 1);
    }

    #[test]
    fn cube_faces_cover_their_axis() {
        let light = Light::point(Vec3::new(1.0, 2.0, 3.0), Vec3::ONE, 1.0, 10.0);
        let (faces, near, _) = face_projections(&light);
        let axes = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
        for (face, axis) in faces.iter().zip(axes) {
            let clip = *face * (light.position + axis * 2.0 + axis.any_orthonormal_vector() * 1.5).extend(1.0);
            let ndc = clip.truncate() / clip.w;
            assert!(clip.w > 0.0 && ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0, "{axis}: {ndc}");
            // Infinite reverse-Z: depth = near / w.
            assert!((ndc.z - near / 2.0).abs() < 1e-5);
        }
    }
}
