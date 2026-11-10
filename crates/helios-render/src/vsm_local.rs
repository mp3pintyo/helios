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
//! (and its cached pages) while it stays selected; its pages are dropped when its shadow map's
//! shape changes, when the scene changes, or when the slot passes to another light, and a face's
//! pages when the face's origin moves. A light that moves keeps each face's origin while it
//! stays close to it, and the faces are re-rendered within a budget of pages (`vsm_anchor.rs`):
//! the GPU counts the pages the view requests of every face, which this module reads back as
//! their cost.
//!
//! One culling view covers every light, face and mip, so the cost of a frame without new pages
//! does not grow with the number of lights. It keeps clusters that overlap a world box around
//! some light's dirty pages and selects LOD for the most demanding light's finest dirty mip
//! (`cull.wgsl`); the expansion then distributes each cluster to the dirty pages of the lights it
//! overlaps (`vsm_expand.wgsl`).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::camera::rh::{proj::directx, view::look_to_mat4};
use glam::{Mat4, Vec3, Vec4};

use crate::gpu_types::CullViewUniform;
use crate::passes::culling::{CullCapacity, CullView};
use crate::scene::{Light, LightKind};
use crate::util::uniform_buffer;
use crate::vsm_anchor::{self, AnchorSettings, AnchorStats, Anchors, LightState, MovingLightShadows, Origins};

/// Lights with a virtual shadow map per frame.
pub const LOCAL_SLOTS: usize = 32;
const FACES: usize = vsm_anchor::FACES;
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
/// A moving light's shadow map is this many mips coarser (UE's
/// `r.Shadow.Virtual.ResolutionLodBiasLocalMoving`): what it renders, per frame for the casters
/// moving with it and in full when its origin moves, covers a quarter of the pages.
const MOVING_MIP_BIAS: f32 = 1.0;
/// A light that holds a slot keeps it against newcomers up to this much more influential, so
/// slots (and their cached pages) do not flip between lights of similar importance.
const KEEP_BONUS: f32 = 1.5;
/// A moving light competes for a slot with this fraction of its influence (see
/// [`SlotAllocator::assign`]).
const MOVING_WEIGHT: f32 = 0.5;
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
    /// x: faces whose cached pages are stale (bit per face), y: 1 if some face is held at an
    /// origin away from the light.
    flags: [u32; 4],
    /// Per face, xyz: the face's origin minus the light's position (`position_range`): zero
    /// unless the light moves (`vsm_anchor.rs`).
    face_offsets: [Vec4; FACES],
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

const _: () = assert!(std::mem::size_of::<GpuSlot>() == 912);
const _: () = assert!(std::mem::size_of::<LocalVsmUniform>() == 912 * LOCAL_SLOTS + 16);

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

/// The parts of a light its shadow map's shape depends on besides its faces' origins (not its
/// colour or intensity, which may flicker without invalidating anything).
fn same_shape(a: &Light, b: &Light) -> bool {
    let kind = match (a.kind, b.kind) {
        (LightKind::Point, LightKind::Point) => true,
        (LightKind::Spot { direction: da, outer: oa, .. }, LightKind::Spot { direction: db, outer: ob, .. }) => da == db && oa == ob,
        _ => false,
    };
    kind && a.range == b.range && a.source_radius == b.source_radius
}

/// Faces of a light's shadow map: one perspective face for a spot light up to
/// [`MAX_SPOT_HALF_ANGLE`], else a cube.
pub(crate) fn face_count(light: &Light) -> usize {
    match light.kind {
        LightKind::Spot { outer, .. } if outer <= MAX_SPOT_HALF_ANGLE => 1,
        _ => FACES,
    }
}

/// All faces of a slot.
const ALL_FACES: u8 = (1 << FACES) - 1;

/// Slot assignment of one frame.
#[derive(Debug, Default, PartialEq)]
pub struct Assignment {
    /// Per light: its slot, or `u32::MAX`.
    pub light_slots: Vec<u32>,
    /// Per slot: the faces (bit per face) whose cached pages must be dropped before this
    /// frame's requests.
    pub stale: [u8; LOCAL_SLOTS],
}

/// A light in a slot: its index, the light as its shadow map renders it and its faces' origins.
type Held = (usize, Light, Origins);

/// Keeps lights in shadow-map slots across frames (pure CPU logic).
#[derive(Debug, Default)]
pub struct SlotAllocator {
    slots: [Option<Held>; LOCAL_SLOTS],
}

impl SlotAllocator {
    /// Gives slots to the most influential lights (`influence[i] > 0` wants one), `moving` ones
    /// counted at [`MOVING_WEIGHT`] of their influence: a static light's map costs nothing once
    /// cached, a moving light's costs re-renders — within the page budget, which has already
    /// turned down the moving lights it cannot follow (no origins, influence 0: MegaLights and
    /// distance fields trace them), so a moving light near the camera may still take the slot of
    /// a faint static one far off. Lights without a slot keep their traced shadows (MegaLights,
    /// or distance fields in the clustered loop). Lights keep their slot while selected; a face
    /// whose origin (`origins` per light; none: at the light) moved drops its pages,
    /// `invalidate_all` every cached page (scene change).
    pub fn assign(&mut self, lights: &[Light], origins: &[Option<Origins>], influence: &[f32], moving: &[bool], invalidate_all: bool) -> Assignment {
        let held = |i: usize| self.slots.iter().flatten().any(|&(light, _, _)| light == i);
        let mut candidates: Vec<(usize, f32)> = influence
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v > 0.0)
            .map(|(i, &v)| (i, v * if held(i) { KEEP_BONUS } else { 1.0 } * if moving.get(i).copied().unwrap_or(false) { MOVING_WEIGHT } else { 1.0 }))
            .collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        candidates.truncate(LOCAL_SLOTS);
        let selected = |i: usize| candidates.iter().any(|&(c, _)| c == i);

        let origin = |i: usize| origins.get(i).copied().flatten().unwrap_or([lights[i].position; FACES]);
        let mut out = Assignment { light_slots: vec![u32::MAX; lights.len()], stale: [0; LOCAL_SLOTS] };
        for (s, slot) in self.slots.iter_mut().enumerate() {
            match slot {
                Some((i, light, faces)) if *i < lights.len() && selected(*i) => {
                    let now = origin(*i);
                    out.stale[s] = if invalidate_all || !same_shape(light, &lights[*i]) {
                        ALL_FACES
                    } else {
                        (0..FACES).filter(|&f| faces[f] != now[f]).fold(0, |mask, f| mask | (1 << f))
                    };
                    *light = lights[*i];
                    *faces = now;
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
            self.slots[free] = Some((i, lights[i], origin(i)));
            out.light_slots[i] = free as u32;
            out.stale[free] = ALL_FACES;
        }
        out
    }

    /// Slots in use: the slot, its light's index, the light and its faces' origins.
    pub fn occupied(&self) -> impl Iterator<Item = (usize, usize, &Light, &Origins)> {
        self.slots.iter().enumerate().filter_map(|(s, slot)| slot.as_ref().map(|(index, light, origins)| (s, *index, light, origins)))
    }
}

/// Which lights moved recently (light positions over the frames).
#[derive(Debug, Default)]
struct LightMotion {
    /// Per light: its position last frame and the frames since it last moved.
    last: Vec<(Vec3, u32)>,
}

impl LightMotion {
    /// Per light of this frame: whether it moved in the last [`MOVING_FRAMES`] frames, and its
    /// move since the last frame. A new light list (another scene) starts at rest.
    fn update(&mut self, lights: &[Light]) -> Vec<(bool, Vec3)> {
        if self.last.len() != lights.len() {
            self.last = lights.iter().map(|l| (l.position, MOVING_FRAMES)).collect();
        }
        self.last
            .iter_mut()
            .zip(lights)
            .map(|((position, since), light)| {
                let step = light.position - *position;
                *since = if step == Vec3::ZERO { since.saturating_add(1) } else { 0 };
                *position = light.position;
                (*since < MOVING_FRAMES, step)
            })
            .collect()
    }
}

/// Projections of a light's faces, each from its origin: (view-projections, near plane, focal
/// length).
fn face_projections(light: &Light, origins: &Origins) -> (Vec<Mat4>, f32, f32) {
    let near = (light.source_radius * 0.5).clamp(0.02, 0.25);
    match light.kind {
        LightKind::Spot { direction, outer, .. } if outer <= MAX_SPOT_HALF_ANGLE => {
            let half = outer + SPOT_MARGIN;
            let up = if direction.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
            let proj = directx::perspective_infinite_reverse(2.0 * half, 1.0, near);
            (vec![proj * look_to_mat4(origins[0], direction, up)], near, 1.0 / half.tan())
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
            (faces.iter().zip(origins).map(|(&(dir, up), &origin)| proj * look_to_mat4(origin, dir, up)).collect(), near, 1.0)
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
    /// How moving lights are shadowed (`RenderSettings::moving_light_shadows`; distance fields
    /// under MegaLights, which traces them). They take the slots static lights leave over.
    pub moving: MovingLightShadows,
    /// Pages of moving lights' faces rendered from new origins per frame
    /// (`RenderSettings::moving_light_pages`).
    pub moving_pages: f32,
}

/// Per-frame culling parameters of the local lights' view.
pub struct LocalCull {
    pub instance_count: u32,
    pub frame: u32,
    pub lod_threshold: f32,
    pub sw_raster_threshold: f32,
    /// `CullViewUniform::flags`.
    pub flags: u32,
}

/// Words of a slot's data in the VSM info buffer (`LOCAL_INFO_SLOT_WORDS` in
/// vsm_local_common.wgsl); from word [`FEEDBACK_REQUESTED`] per face the pages the view requested
/// this frame, from [`FEEDBACK_FULL`] those rendered in full.
pub const SLOT_WORDS: u32 = 32;
const FEEDBACK_REQUESTED: usize = 18;
const FEEDBACK_FULL: usize = 24;
const FEEDBACK_BYTES: u64 = LOCAL_SLOTS as u64 * SLOT_WORDS as u64 * 4;

const IDLE: u32 = 0;
const MAPPING: u32 = 1;
const READY: u32 = 2;

/// Per-face page counts read back from the GPU (a few frames late): what re-anchoring a face
/// costs.
struct Feedback {
    buffer: wgpu::Buffer,
    status: Arc<AtomicU32>,
    /// A copy was recorded this frame.
    recorded: bool,
    /// The light in each slot when the copy in flight was recorded, and whether it moved.
    lights: [usize; LOCAL_SLOTS],
    moving: [bool; LOCAL_SLOTS],
    /// Per light of the scene: the pages its faces cost, as last counted.
    pages: Vec<[f32; FACES]>,
    /// Local light pages rendered in full in the latest read-back frame, and of them those of
    /// moving lights.
    full: u32,
    moving_full: u32,
}

/// Moving lights in the latest frame (the HUD's `moving lights` line).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MovingLightStats {
    pub anchors: AnchorStats,
    /// Local light pages rendered in full in a recent frame, and of them those of moving lights
    /// (re-anchored faces, and pages their faces newly need).
    pub full_pages: u32,
    pub moving_full_pages: u32,
}

pub struct LocalShadows {
    pub uniform: wgpu::Buffer,
    allocator: SlotAllocator,
    motion: LightMotion,
    anchors: Anchors,
    gpu: Box<LocalVsmUniform>,
    /// The culling view of all local lights.
    view: CullView,
    /// Per light of the scene: its slot or `u32::MAX` (uploaded with the lights).
    light_slots: Vec<u32>,
    active: usize,
    /// Slots whose light is away from some face's origin.
    held: usize,
    /// Per light of the scene: it moved recently.
    moving: Vec<bool>,
    any_stale: bool,
    info_base: u32,
    feedback: Feedback,
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
            anchors: Anchors::default(),
            gpu,
            view: CullView::new(device, "local vsm", capacity),
            light_slots: Vec::new(),
            active: 0,
            held: 0,
            moving: Vec::new(),
            any_stale: false,
            info_base,
            feedback: Feedback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("local vsm feedback"),
                    size: FEEDBACK_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                status: Arc::new(AtomicU32::new(IDLE)),
                recorded: false,
                lights: [usize::MAX; LOCAL_SLOTS],
                moving: [false; LOCAL_SLOTS],
                pages: Vec::new(),
                full: 0,
                moving_full: 0,
            },
        }
    }

    /// Makes room for `extra` more page batches in the lights' culling view (assembly parts).
    pub fn reserve_batches(&mut self, device: &wgpu::Device, base: u32, extra: u32) {
        self.view.reserve_batches(device, base + extra);
    }

    /// Chooses this frame's shadowed lights. Call before uploading the lights.
    pub fn assign(&mut self, f: &LocalFrame) {
        self.poll_feedback();
        let motion = self.motion.update(f.lights);
        let mut influence: Vec<f32> = f.lights.iter().map(|l| if f.enabled { screen_influence(l, f.eye, &f.planes) } else { 0.0 }).collect();
        // Each face of a light's shadow map is rendered from its origin, which a moving light
        // leaves behind (vsm_anchor.rs). Lights without origins get no slot.
        let pages = &mut self.feedback.pages;
        if pages.len() != f.lights.len() {
            *pages = vec![[vsm_anchor::DEFAULT_FACE_PAGES; FACES]; f.lights.len()];
        }
        let states: Vec<LightState> = (f.lights.iter().zip(&motion).zip(&influence).zip(pages.iter()))
            .map(|(((l, &(moving, step)), &influence), &pages)| LightState {
                position: l.position,
                moving,
                step,
                tolerance: vsm_anchor::tolerance(l),
                influence,
                faces: face_count(l),
                pages,
            })
            .collect();
        let origins = self.anchors.update(&states, AnchorSettings { mode: f.moving, budget: f.moving_pages });
        for (influence, origin) in influence.iter_mut().zip(&origins) {
            if origin.is_none() {
                *influence = 0.0;
            }
        }
        let moving: Vec<bool> = motion.iter().map(|m| m.0).collect();
        let assignment = self.allocator.assign(f.lights, &origins, &influence, &moving, f.scene_changed);
        self.moving = moving.clone();
        self.anchors.keep(|i| assignment.light_slots[i] != u32::MAX);
        self.light_slots = assignment.light_slots;
        self.any_stale = assignment.stale.iter().any(|&s| s != 0);
        let gpu = &mut *self.gpu;
        gpu.enabled = f.enabled as u32;
        for (s, slot) in gpu.slots.iter_mut().enumerate() {
            *slot = GpuSlot::zeroed();
            slot.flags[0] = assignment.stale[s] as u32;
        }
        self.active = 0;
        self.held = 0;
        for (s, index, light, origins) in self.allocator.occupied() {
            let (faces, near, focal) = face_projections(light, origins);
            let slot = &mut gpu.slots[s];
            for (i, view_proj) in faces.iter().enumerate() {
                slot.view_proj[i] = *view_proj;
                slot.inv_view_proj[i] = view_proj.inverse();
                slot.face_offsets[i] = (origins[i] - light.position).extend(0.0);
            }
            let cached_moving = moving[index] && f.moving == MovingLightShadows::Cached;
            let mip_bias = (light.source_radius / SHARP_SOURCE_RADIUS).log2().max(0.0) + if cached_moving { MOVING_MIP_BIAS } else { 0.0 };
            slot.position_range = light.position.extend(light.range);
            slot.params = Vec4::new(faces.len() as f32, near, focal, mip_bias);
            let held = slot.face_offsets.iter().any(|&o| o != Vec4::ZERO);
            slot.flags[1] = held as u32;
            self.held += held as usize;
            self.active += 1;
        }
    }

    /// Copies this frame's per-face page counts (the slots' words of the VSM `info` buffer, from
    /// `info_base`) for reading back, if the last read-back is done. Record after the pass that
    /// counts them (`dirty_info`).
    pub fn record_feedback(&mut self, encoder: &mut wgpu::CommandEncoder, info: &wgpu::Buffer) {
        let fb = &mut self.feedback;
        fb.recorded = fb.status.load(Ordering::Acquire) == IDLE;
        if !fb.recorded {
            return;
        }
        encoder.copy_buffer_to_buffer(info, self.info_base as u64 * 4, &fb.buffer, 0, Some(FEEDBACK_BYTES));
        fb.lights = [usize::MAX; LOCAL_SLOTS];
        for (s, index, _, _) in self.allocator.occupied() {
            fb.lights[s] = index;
            fb.moving[s] = self.moving.get(index).copied().unwrap_or(false);
        }
    }

    /// Call after submitting the frame: maps the feedback copied in it.
    pub fn after_submit(&mut self) {
        let fb = &mut self.feedback;
        if !std::mem::take(&mut fb.recorded) {
            return;
        }
        fb.status.store(MAPPING, Ordering::Release);
        let status = fb.status.clone();
        fb.buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            status.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
        });
    }

    /// Takes a finished read-back: each slot's light's face costs.
    fn poll_feedback(&mut self) {
        let fb = &mut self.feedback;
        if fb.status.load(Ordering::Acquire) != READY {
            return;
        }
        if let Ok(view) = fb.buffer.get_mapped_range(..) {
            let words: &[u32] = bytemuck::cast_slice(&view);
            fb.full = 0;
            fb.moving_full = 0;
            for (s, &light) in fb.lights.iter().enumerate() {
                let slot = &words[s * SLOT_WORDS as usize..(s + 1) * SLOT_WORDS as usize];
                let full: u32 = slot[FEEDBACK_FULL..FEEDBACK_FULL + FACES].iter().sum();
                fb.full += full;
                if fb.moving[s] {
                    fb.moving_full += full;
                }
                if let Some(pages) = fb.pages.get_mut(light) {
                    for (face, p) in pages.iter_mut().enumerate() {
                        *p = slot[FEEDBACK_REQUESTED + face] as f32;
                    }
                }
            }
        }
        fb.buffer.unmap();
        fb.status.store(IDLE, Ordering::Release);
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
            flags: cull.flags,
            _pad: [0; 3],
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

    /// Lights whose shadow map is held at an origin they moved away from.
    pub fn held_count(&self) -> usize {
        self.held
    }

    /// What the moving lights' shadow maps did in a recent frame.
    pub fn moving_stats(&self) -> MovingLightStats {
        MovingLightStats { anchors: self.anchors.stats, full_pages: self.feedback.full, moving_full_pages: self.feedback.moving_full }
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
        let first = a.assign(&lights, &[], &[1.0, 2.0, 0.0, 3.0], &[], false);
        assert_eq!(first.light_slots[2], u32::MAX);
        assert_eq!(first.stale.iter().filter(|&&s| s == ALL_FACES).count(), 3);
        let second = a.assign(&lights, &[], &[1.0, 2.0, 0.0, 3.0], &[], false);
        assert_eq!(second.light_slots, first.light_slots);
        assert!(second.stale.iter().all(|&s| s == 0), "static lights keep their pages");
    }

    #[test]
    fn moving_or_changed_lights_invalidate_only_their_slot() {
        let mut lights = lights(3);
        let mut a = SlotAllocator::default();
        let first = a.assign(&lights, &[], &[1.0; 3], &[], false);
        lights[1].position.y += 0.5;
        lights[2].intensity = 7.0; // brightness does not change the shadow
        let second = a.assign(&lights, &[], &[1.0; 3], &[], false);
        let stale: Vec<usize> = (0..LOCAL_SLOTS).filter(|&s| second.stale[s] != 0).collect();
        assert_eq!(stale, vec![first.light_slots[1] as usize]);
        assert_eq!(second.stale[stale[0]], ALL_FACES);
        // A face held at an origin: moving it drops only that face's pages; the light's own
        // position does not matter.
        let mut origins = vec![None; 3];
        let mut faces = [lights[0].position; FACES];
        faces[3] += Vec3::X * 0.01;
        origins[0] = Some(faces);
        lights[0].position.x += 0.02;
        let held = a.assign(&lights, &origins, &[1.0; 3], &[], false);
        assert_eq!(held.stale[first.light_slots[0] as usize], 1 << 3);
        assert_eq!(held.stale.iter().filter(|&&s| s != 0).count(), 1);
        let third = a.assign(&lights, &origins, &[1.0; 3], &[], true);
        assert_eq!(third.stale.iter().filter(|&&s| s == ALL_FACES).count(), 3, "scene changes drop every page");
    }

    #[test]
    fn budget_prefers_influential_lights_with_hysteresis() {
        let lights = lights(LOCAL_SLOTS + 2);
        let mut influence: Vec<f32> = (0..lights.len()).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut a = SlotAllocator::default();
        let first = a.assign(&lights, &[], &influence, &[], false);
        assert_eq!(first.light_slots[0], u32::MAX);
        assert_eq!(first.light_slots[1], u32::MAX);
        assert_eq!(first.light_slots.iter().filter(|&&s| s != u32::MAX).count(), LOCAL_SLOTS);
        // Slightly more influential than a holder: not enough to take its slot.
        influence[0] = 1.2;
        let second = a.assign(&lights, &[], &influence, &[], false);
        assert_eq!(second.light_slots[0], u32::MAX);
        // Much more influential: takes the slot of the least influential holder (light 2).
        influence[0] = 10.0;
        let third = a.assign(&lights, &[], &influence, &[], false);
        assert_eq!(third.light_slots[0], first.light_slots[2]);
        assert_eq!(third.light_slots[2], u32::MAX);
        assert_eq!(third.stale[third.light_slots[0] as usize], ALL_FACES);
        assert_eq!(third.stale.iter().filter(|&&s| s != 0).count(), 1);
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
    fn moving_lights_count_at_half_their_influence() {
        let lights = lights(LOCAL_SLOTS + 1);
        // A moving light a little more influential than the static ones loses...
        let mut influence: Vec<f32> = (0..lights.len()).map(|i| if i == 0 { 1.5 } else { 1.0 }).collect();
        let mut moving = vec![false; lights.len()];
        moving[0] = true;
        let mut a = SlotAllocator::default();
        let crowded = a.assign(&lights, &[], &influence, &moving, false);
        assert_eq!(crowded.light_slots[0], u32::MAX);
        // ... a much more influential one (a lantern near the camera) takes a faint static
        // light's slot.
        influence[0] = 4.0;
        let near = a.assign(&lights, &[], &influence, &moving, false);
        assert_ne!(near.light_slots[0], u32::MAX);
        assert_eq!(near.light_slots.iter().filter(|&&s| s == u32::MAX).count(), 1);
        // With a slot to spare it gets one.
        let mut a = SlotAllocator::default();
        let spare = a.assign(&lights[..LOCAL_SLOTS], &[], &influence[..LOCAL_SLOTS], &moving, false);
        assert_ne!(spare.light_slots[0], u32::MAX);
    }

    #[test]
    fn lights_count_as_moving_until_they_rest_a_while() {
        let mut lights = lights(2);
        let mut motion = LightMotion::default();
        let moving = |motion: &mut LightMotion, lights: &[Light]| motion.update(lights).iter().map(|m| m.0).collect::<Vec<_>>();
        assert_eq!(moving(&mut motion, &lights), vec![false, false], "a new scene starts at rest");
        lights[1].position.x += 0.5;
        assert_eq!(motion.update(&lights)[1], (true, Vec3::X * 0.5));
        for _ in 1..MOVING_FRAMES {
            assert_eq!(moving(&mut motion, &lights), vec![false, true]);
        }
        assert_eq!(moving(&mut motion, &lights), vec![false, false], "at rest again");
    }

    #[test]
    fn face_entries_hold_every_mip() {
        // 32² pages at mip 0 down to one page (LOCAL_FACE_ENTRIES and local_mip_offset in
        // vsm_local_common.wgsl).
        let pages: u32 = (0..MIPS).map(|mip| (32u32 >> mip).pow(2)).sum();
        assert_eq!(pages, FACE_ENTRIES);
        assert_eq!(32 >> (MIPS - 1), 1);
    }

    /// `local_sphere_rect` of vsm_local_common.wgsl: (valid, lo, hi, inside) in clip x / w.
    fn sphere_rect(view_proj: &Mat4, near: f32, center: Vec3, radius: f32) -> (bool, glam::Vec2, glam::Vec2, f32) {
        let clip = *view_proj * center.extend(1.0);
        let front = clip.w - radius;
        let mut valid = clip.w + radius > near;
        let (mut lo, mut hi) = (glam::Vec2::splat(-1.0), glam::Vec2::splat(1.0));
        let (a, b) = (clip.truncate().truncate() - radius, clip.truncate().truncate() + radius);
        let back = clip.w + radius;
        if front > near {
            lo = (a / front).min(a / back).max(lo);
            hi = (b / front).max(b / back).min(hi);
        } else if valid {
            lo = glam::Vec2::select(a.cmpgt(glam::Vec2::ZERO), (a / back).max(lo), lo);
            hi = glam::Vec2::select(b.cmplt(glam::Vec2::ZERO), (b / back).min(hi), hi);
        }
        valid = valid && hi.cmpge(lo).all();
        (valid, lo, hi, front.max(near).max(a.x.max(-b.x)).max(a.y.max(-b.y)))
    }

    #[test]
    fn sphere_rects_bound_what_a_face_sees() {
        // Spheres all around a light, many across the near planes of its faces: every point of a
        // sphere that a face sees lies in the sphere's rectangle, no nearer than its bound.
        let light = Light::point(Vec3::new(1.0, 2.0, 3.0), Vec3::ONE, 1.0, 10.0);
        let (faces, near, _) = face_projections(&light, &[light.position; FACES]);
        let mut state = 0x2545_f491u32;
        let mut random = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        let (mut culled, mut seen) = (0, 0);
        for _ in 0..400 {
            let center = light.position + Vec3::new(random(), random(), random()) * 6.0;
            let radius = 0.05 + (random() + 1.0) * 1.5;
            for face in &faces {
                let (valid, lo, hi, inside) = sphere_rect(face, near, center, radius);
                culled += !valid as usize;
                for _ in 0..60 {
                    let p = center + Vec3::new(random(), random(), random()).normalize_or_zero() * radius * random().abs();
                    let clip = *face * p.extend(1.0);
                    let ndc = clip.truncate().truncate() / clip.w;
                    if clip.w <= near || ndc.abs().max_element() > 1.0 {
                        continue;
                    }
                    seen += 1;
                    assert!(valid, "{center} {radius}: a visible point in a culled sphere");
                    assert!(ndc.cmpge(lo - 1e-4).all() && ndc.cmple(hi + 1e-4).all(), "{ndc} outside {lo}..{hi}");
                    assert!(clip.w >= inside - 1e-4, "depth {} below the bound {inside}", clip.w);
                }
            }
        }
        // The test means something: faces see points, and most spheres miss most faces — also
        // those reaching a face's near plane from the side (they used to cover the whole face).
        assert!(seen > 5000 && culled > 400 * 3, "{seen} {culled}");
        let beside = light.position + Vec3::new(0.1, 4.0, 0.0);
        assert!(!sphere_rect(&faces[0], near, beside, 1.0).0, "beside the +X face, across its near plane");
        assert!(sphere_rect(&faces[2], near, beside, 1.0).3 > 2.9, "in the +Y face, 3 m out");
    }

    #[test]
    fn cube_faces_cover_their_axis() {
        let light = Light::point(Vec3::new(1.0, 2.0, 3.0), Vec3::ONE, 1.0, 10.0);
        let (faces, near, _) = face_projections(&light, &[light.position; FACES]);
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
