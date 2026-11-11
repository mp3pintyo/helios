//! Lumen world cache (UE 5.9's Lumen World Cache, `r.Lumen.WorldCache`): a world-space lighting
//! cache in sparse hashed cells, filled by paths traced from the screen, read where Lumen's rays
//! hit a surface (see `lumen_world_cache_common.wgsl` and `lumen_world_cache.wgsl`).
//!
//! * **Cells**: a hash table of [`CAPACITY`] slots (linear probing) keyed by a cell of a
//!   camera-distance-dependent grid (20 cm within 8 m, doubling with each doubling of the
//!   distance) and the dominant axis and sign of the surface normal, so the two sides of a thin
//!   wall never share a cell. A cell holds the irradiance its surface receives (running mean of
//!   up to 16 frames), its albedo and emission, and whether the albedo was seen on screen.
//! * **Paths**: each frame one random pixel per 8×8 tile starts a path: the pixel's G-buffer
//!   surface, up to two hits of cosine rays through the mesh SDFs and the global distance
//!   field, and a third ray shaded by the caches (the world cache itself: multi-bounce over
//!   frames). Every vertex's irradiance (direct sun and local lights + the path's incoming
//!   light) is added to its cell.
//! * **Material at a hit** (no hit shaders): the albedo a cell learnt from the pixels on screen
//!   is exact for every surface ever seen (textured, skinned, displaced, wind-blown); hits on
//!   cells never seen take the mesh cards' captured texel, else the global field's mean albedo
//!   of the instance's material.
//! * **Readers**: the screen probes', reflections', Lumen Lite's and the radiance cache's ray
//!   hits read the cell of the surface they hit: [`WorldCacheMode::Fallback`] behind the mesh
//!   cards (in place of the voxel lighting, which leaks through thin walls),
//!   [`WorldCacheMode::Replace`] in place of the cards too (UE's behaviour: the card lighting
//!   is not updated).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{BindingType, TextureFormat as F};

use crate::lumen::Stage;
use crate::profiler::Recorder;
use crate::util::{bind, storage_buffer, texture, workgroups, TextureSpec};

/// Hash slots (`WC_CAPACITY` in lumen_world_cache_common.wgsl).
pub const CAPACITY: u32 = 1 << 18;
/// Slots per row of the resolved cell texture (`WC_SLOTS_X`).
const SLOTS_X: u32 = 512;
/// Words per slot in the record buffer (`WC_RECORD` in lumen_world_cache.wgsl).
const RECORD_WORDS: u64 = 16;
/// Counters of the statistics buffer: live cells, cells sampled this frame, debug-view pixels
/// with a cell, debug-view pixels.
const STAT_WORDS: u64 = 4;

/// Where Lumen's ray hits read the world cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WorldCacheMode {
    /// Mesh cards, voxel lighting where no card matches (round 7).
    #[default]
    Off,
    /// Mesh cards, the world cache where no card matches, the voxel lighting where neither has
    /// the surface.
    Fallback,
    /// The world cache, the voxel lighting where it has no cell; the card lighting is not
    /// updated (UE 5.9: the world cache replaces the surface cache for the view).
    Replace,
}

impl WorldCacheMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fallback => "behind cards",
            Self::Replace => "replaces cards",
        }
    }

    /// Off → behind the cards → in place of the cards → off.
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Fallback,
            Self::Fallback => Self::Replace,
            Self::Replace => Self::Off,
        }
    }

    /// `off`, `on` / `fallback` / `cards`, `replace` / `only`.
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "off" | "0" | "false" => Some(Self::Off),
            "on" | "1" | "true" | "fallback" | "cards" => Some(Self::Fallback),
            "replace" | "only" | "2" => Some(Self::Replace),
            _ => None,
        }
    }

    /// `lumen.world_cache` in the shaders.
    pub(crate) fn gpu(self) -> u32 {
        self as u32
    }
}

/// Statistics of the latest read-back (natively with `HELIOS_WORLD_CACHE_STATS`).
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldCacheStats {
    pub live_cells: u32,
    pub sampled_cells: u32,
    /// Fraction of the pixels whose surface has a cell (debug view only).
    pub screen_coverage: Option<f32>,
}

/// What the passes bind besides the cache's own buffers.
pub(crate) struct WorldCacheInputs<'a> {
    pub frame: &'a wgpu::Buffer,
    pub lumen: &'a wgpu::Buffer,
    pub global_sdf: &'a wgpu::TextureView,
    pub voxel_lighting: &'a wgpu::TextureView,
    pub linear: &'a wgpu::Sampler,
    pub shadow_sampler: &'a wgpu::Sampler,
    pub radiance_probes: &'a wgpu::Buffer,
    pub transmittance: &'a wgpu::TextureView,
    pub sky_view: &'a wgpu::TextureView,
    pub lights: &'a wgpu::Buffer,
    pub exposure: &'a wgpu::Buffer,
    pub vsm: &'a crate::vsm::VirtualShadowMap,
    pub glass: &'a crate::glass_shadow::GlassShadow,
    pub near_field: (&'a wgpu::Buffer, &'a wgpu::TextureView),
    pub targets: &'a crate::targets::FrameTargets,
    pub card_data: &'a wgpu::Buffer,
    pub card_pages: &'a wgpu::Buffer,
    pub card_capture: &'a wgpu::TextureView,
    pub card_lighting: &'a wgpu::TextureView,
    pub card_albedo: &'a wgpu::TextureView,
    pub instance_ids: &'a wgpu::TextureView,
}

struct Resources {
    /// Per slot [`RECORD_WORDS`] words: key, this frame's sums, last visit, resolved cell.
    records: wgpu::Buffer,
    /// Resolved cells, one rgba32uint texel per slot (what Lumen's traces read).
    cells: wgpu::TextureView,
    stats: wgpu::Buffer,
    readback: wgpu::Buffer,
}

struct Reference {
    stage: Stage,
    sums: Option<(wgpu::Buffer, glam::Vec3, (u32, u32))>,
    clear: bool,
}

const IDLE: u32 = 0;
const MAPPING: u32 = 1;
const READY: u32 = 2;

pub(crate) struct WorldCache {
    pub mode: WorldCacheMode,
    resources: Option<Resources>,
    /// 1×1 empty cell texture, bound while the cache is not allocated.
    placeholder: wgpu::TextureView,
    /// The scene changed: clear every cell before the next update.
    clear: bool,
    trace: Stage,
    resolve: Stage,
    debug: Stage,
    /// Brute-force path-traced GI in place of the final gather, for comparisons
    /// (`HELIOS_WORLD_CACHE_REFERENCE`, native only): sums per half-resolution pixel, and the
    /// camera position and size they were accumulated for.
    reference: Option<Reference>,
    /// Statistics read-back (only with `HELIOS_WORLD_CACHE_STATS`).
    stats_enabled: bool,
    status: Arc<AtomicU32>,
    recorded_stats: bool,
    pub latest: WorldCacheStats,
}

fn cell_texture(device: &wgpu::Device, label: &str, width: u32, height: u32) -> wgpu::TextureView {
    texture(
        device,
        TextureSpec {
            label,
            width,
            height,
            layers: 1,
            mips: 1,
            format: F::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        },
    )
    .create_view(&Default::default())
}

impl WorldCache {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = crate::shader::create_module(device, "lumen_world_cache.wgsl");
        let u = bind::uniform;
        let ro = || bind::storage(true);
        let rw = || bind::storage(false);
        let tex = bind::texture;
        let tex3 = bind::texture_3d;
        let stage = |entry: &str, bindings: &[(u32, BindingType)]| Stage::new(device, &module, entry, bindings);
        Self {
            mode: WorldCacheMode::Off,
            resources: None,
            placeholder: cell_texture(device, "world cache placeholder", 1, 1),
            clear: false,
            trace: stage(
                "trace_paths",
                &[
                    (0, u()),
                    (1, u()),
                    (2, tex3(true)),
                    (3, tex3(true)),
                    (4, bind::sampler()),
                    (5, ro()),
                    (6, tex(true)),
                    (7, ro()),
                    (8, tex(true)),
                    (9, ro()),
                    (15, u()),
                    (16, ro()),
                    (17, bind::texture_depth()),
                    (18, bind::sampler_comparison()),
                    (19, ro()),
                    (20, tex3(true)),
                    (25, tex(false)),
                    (26, tex(false)),
                    (27, tex(false)),
                    (28, tex(false)),
                    (29, tex(false)),
                    (70, ro()),
                    (71, ro()),
                    (72, bind::texture_uint()),
                    (73, tex(false)),
                    (74, bind::texture_3d_uint()),
                    (90, u()),
                    (91, tex(true)),
                    (150, bind::texture_uint()),
                    (151, rw()),
                ],
            ),
            resolve: stage("resolve", &[(1, u()), (9, ro()), (151, rw()), (152, bind::storage_texture(F::Rgba32Uint)), (154, rw())]),
            debug: stage(
                "debug",
                &[
                    (0, u()),
                    (1, u()),
                    (2, tex3(true)),
                    (3, tex3(true)),
                    (4, bind::sampler()),
                    (25, tex(false)),
                    (26, tex(false)),
                    (150, bind::texture_uint()),
                    (153, bind::storage_texture(F::Rgba16Float)),
                    (154, rw()),
                ],
            ),
            reference: (cfg!(not(target_arch = "wasm32")) && std::env::var_os("HELIOS_WORLD_CACHE_REFERENCE").is_some()).then(|| Reference {
                stage: stage(
                    "reference",
                    &[
                        (0, u()),
                        (1, u()),
                        (2, tex3(true)),
                        (3, tex3(true)),
                        (4, bind::sampler()),
                        (5, ro()),
                        (6, tex(true)),
                        (7, ro()),
                        (8, tex(true)),
                        (9, ro()),
                        (15, u()),
                        (16, ro()),
                        (17, bind::texture_depth()),
                        (18, bind::sampler_comparison()),
                        (19, ro()),
                        (20, tex3(true)),
                        (25, tex(false)),
                        (26, tex(false)),
                        (29, tex(false)),
                        (70, ro()),
                        (71, ro()),
                        (72, bind::texture_uint()),
                        (73, tex(false)),
                        (74, bind::texture_3d_uint()),
                        (90, u()),
                        (91, tex(true)),
                        (150, bind::texture_uint()),
                        (155, rw()),
                        (156, bind::storage_texture(F::Rgba16Float)),
                    ],
                ),
                sums: None,
                clear: false,
            }),
            stats_enabled: cfg!(not(target_arch = "wasm32")) && std::env::var_os("HELIOS_WORLD_CACHE_STATS").is_some(),
            status: Arc::new(AtomicU32::new(IDLE)),
            recorded_stats: false,
            latest: WorldCacheStats::default(),
        }
    }

    /// The cell texture Lumen's traces bind (binding 150).
    pub fn cells(&self) -> &wgpu::TextureView {
        self.resources.as_ref().map_or(&self.placeholder, |r| &r.cells)
    }

    /// Every cell is dropped before the next update (scene change).
    pub fn invalidate(&mut self) {
        self.clear = true;
    }

    /// The reference path tracer replaces the final gather (`HELIOS_WORLD_CACHE_REFERENCE`).
    pub fn reference_active(&self) -> bool {
        self.reference.is_some()
    }

    /// Allocates the cache on first use (and the reference's sums, restarted when the camera
    /// moves or the internal size changes).
    pub fn prepare(&mut self, device: &wgpu::Device, camera: glam::Vec3, size: (u32, u32)) {
        if let Some(reference) = &mut self.reference {
            if reference.sums.as_ref().is_none_or(|s| s.2 != size) {
                let half = (size.0.div_ceil(2), size.1.div_ceil(2));
                let buffer = storage_buffer(device, "world cache reference", half.0 as u64 * half.1 as u64 * 32, wgpu::BufferUsages::empty());
                reference.sums = Some((buffer, camera, size));
            } else if let Some(sums) = &mut reference.sums {
                if sums.1 != camera {
                    sums.1 = camera;
                    reference.clear = true;
                }
            }
        }
        if self.mode == WorldCacheMode::Off || self.resources.is_some() {
            return;
        }
        let records = storage_buffer(device, "world cache records", CAPACITY as u64 * RECORD_WORDS * 4, wgpu::BufferUsages::empty());
        let stats = storage_buffer(device, "world cache stats", STAT_WORDS * 4, wgpu::BufferUsages::COPY_SRC);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world cache stats readback"),
            size: STAT_WORDS * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cells = cell_texture(device, "world cache cells", SLOTS_X, CAPACITY / SLOTS_X);
        self.resources = Some(Resources { records, cells, stats, readback });
        // Fresh buffers are zeroed: nothing to clear.
        self.clear = false;
    }

    /// Records the path tracing and the resolve (after the voxel lighting and the radiance cache
    /// are updated, before the final gather reads the cells).
    pub fn record(&mut self, rec: &mut Recorder, inputs: &WorldCacheInputs) {
        if self.mode == WorldCacheMode::Off {
            return;
        }
        let Some(res) = &self.resources else { return };
        let device = rec.device;
        if self.clear {
            self.clear = false;
            rec.encoder.clear_buffer(&res.records, 0, None);
        }
        let stats = self.stats_enabled && self.status.load(Ordering::Acquire) == IDLE;
        if stats {
            rec.encoder.clear_buffer(&res.stats, 0, None);
        }
        let targets = inputs.targets;
        let trace = self.trace.group(
            device,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, inputs.lumen.as_entire_binding()),
                (2, TextureView(inputs.global_sdf)),
                (3, TextureView(inputs.voxel_lighting)),
                (4, Sampler(inputs.linear)),
                (5, inputs.radiance_probes.as_entire_binding()),
                (6, TextureView(inputs.transmittance)),
                (7, inputs.lights.as_entire_binding()),
                (8, TextureView(inputs.sky_view)),
                (9, inputs.exposure.as_entire_binding()),
                (15, inputs.vsm.uniform.as_entire_binding()),
                (16, inputs.vsm.cache.entries.as_entire_binding()),
                (17, TextureView(&inputs.vsm.atlas)),
                (18, Sampler(inputs.shadow_sampler)),
                (19, inputs.near_field.0.as_entire_binding()),
                (20, TextureView(inputs.near_field.1)),
                (25, TextureView(&targets.depth)),
                (26, TextureView(&targets.gbuffer_normal)),
                (27, TextureView(&targets.gbuffer_albedo)),
                (28, TextureView(&targets.gbuffer_motion)),
                (29, TextureView(inputs.card_albedo)),
                (70, inputs.card_data.as_entire_binding()),
                (71, inputs.card_pages.as_entire_binding()),
                (72, TextureView(inputs.card_capture)),
                (73, TextureView(inputs.card_lighting)),
                (74, TextureView(inputs.instance_ids)),
                (90, inputs.glass.uniform.as_entire_binding()),
                (91, TextureView(inputs.glass.view())),
                (150, TextureView(&res.cells)),
                (151, res.records.as_entire_binding()),
            ],
        );
        let resolve = self.resolve.group(
            device,
            &[
                (1, inputs.lumen.as_entire_binding()),
                (9, inputs.exposure.as_entire_binding()),
                (151, res.records.as_entire_binding()),
                (152, TextureView(&res.cells)),
                (154, res.stats.as_entire_binding()),
            ],
        );
        {
            let mut pass = rec.compute_pass("lumen world cache");
            pass.set_pipeline(&self.trace.pipeline);
            pass.set_bind_group(0, &trace, &[]);
            let tiles = (targets.width.div_ceil(8), targets.height.div_ceil(8));
            pass.dispatch_workgroups(workgroups(tiles.0, 8), workgroups(tiles.1, 8), 1);
            pass.set_pipeline(&self.resolve.pipeline);
            pass.set_bind_group(0, &resolve, &[]);
            pass.dispatch_workgroups(CAPACITY / 64, 1, 1);
        }
        self.recorded_stats = stats;
    }

    /// Path-traces the reference GI into `targets.gi_raw` (in place of the final gather).
    pub fn record_reference(&mut self, rec: &mut Recorder, inputs: &WorldCacheInputs) {
        let Some(reference) = &mut self.reference else { return };
        let Some((sums, _, _)) = &reference.sums else { return };
        if reference.clear {
            reference.clear = false;
            rec.encoder.clear_buffer(sums, 0, None);
        }
        let device = rec.device;
        let targets = inputs.targets;
        let cells = self.resources.as_ref().map_or(&self.placeholder, |r| &r.cells);
        let group = reference.stage.group(
            device,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, inputs.lumen.as_entire_binding()),
                (2, TextureView(inputs.global_sdf)),
                (3, TextureView(inputs.voxel_lighting)),
                (4, Sampler(inputs.linear)),
                (5, inputs.radiance_probes.as_entire_binding()),
                (6, TextureView(inputs.transmittance)),
                (7, inputs.lights.as_entire_binding()),
                (8, TextureView(inputs.sky_view)),
                (9, inputs.exposure.as_entire_binding()),
                (15, inputs.vsm.uniform.as_entire_binding()),
                (16, inputs.vsm.cache.entries.as_entire_binding()),
                (17, TextureView(&inputs.vsm.atlas)),
                (18, Sampler(inputs.shadow_sampler)),
                (19, inputs.near_field.0.as_entire_binding()),
                (20, TextureView(inputs.near_field.1)),
                (25, TextureView(&targets.depth)),
                (26, TextureView(&targets.gbuffer_normal)),
                (29, TextureView(inputs.card_albedo)),
                (70, inputs.card_data.as_entire_binding()),
                (71, inputs.card_pages.as_entire_binding()),
                (72, TextureView(inputs.card_capture)),
                (73, TextureView(inputs.card_lighting)),
                (74, TextureView(inputs.instance_ids)),
                (90, inputs.glass.uniform.as_entire_binding()),
                (91, TextureView(inputs.glass.view())),
                (150, TextureView(cells)),
                (155, sums.as_entire_binding()),
                (156, TextureView(&targets.gi_raw)),
            ],
        );
        let mut pass = rec.compute_pass("lumen world cache reference");
        pass.set_pipeline(&reference.stage.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(targets.width.div_ceil(2), 8), workgroups(targets.height.div_ceil(2), 8), 1);
    }

    /// Draws the cache's radiance at every pixel into `targets.hdr` ([`crate::DebugView::WorldCache`];
    /// purple-tinted voxel lighting where no cell holds the surface).
    pub fn record_debug(&self, rec: &mut Recorder, inputs: &WorldCacheInputs) {
        let Some(res) = &self.resources else { return };
        let device = rec.device;
        let targets = inputs.targets;
        let group = self.debug.group(
            device,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, inputs.lumen.as_entire_binding()),
                (2, TextureView(inputs.global_sdf)),
                (3, TextureView(inputs.voxel_lighting)),
                (4, Sampler(inputs.linear)),
                (25, TextureView(&targets.depth)),
                (26, TextureView(&targets.gbuffer_normal)),
                (150, TextureView(&res.cells)),
                (153, TextureView(&targets.hdr)),
                (154, res.stats.as_entire_binding()),
            ],
        );
        let mut pass = rec.compute_pass("lumen world cache debug");
        pass.set_pipeline(&self.debug.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(targets.width, 8), workgroups(targets.height, 8), 1);
    }

    /// Copies the statistics for read-back (end of the frame).
    pub fn record_stats(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.recorded_stats {
            return;
        }
        if let Some(res) = &self.resources {
            encoder.copy_buffer_to_buffer(&res.stats, 0, &res.readback, 0, Some(STAT_WORDS * 4));
        } else {
            self.recorded_stats = false;
        }
    }

    pub fn after_submit(&mut self) {
        if let Some(res) = &self.resources {
            if self.status.load(Ordering::Acquire) == READY {
                if let Ok(view) = res.readback.get_mapped_range(..) {
                    let words: &[u32] = bytemuck::cast_slice(&view);
                    self.latest = WorldCacheStats {
                        live_cells: words[0],
                        sampled_cells: words[1],
                        screen_coverage: (words[3] > 0).then(|| words[2] as f32 / words[3] as f32),
                    };
                }
                res.readback.unmap();
                self.status.store(IDLE, Ordering::Release);
                log::info!(
                    "world cache: {} live cells ({:.1} % of {CAPACITY}), {} sampled this frame{}",
                    self.latest.live_cells,
                    self.latest.live_cells as f32 * 100.0 / CAPACITY as f32,
                    self.latest.sampled_cells,
                    self.latest.screen_coverage.map_or(String::new(), |c| format!(", {:.1} % of the pixels have a cell", c * 100.0)),
                );
            }
            if self.recorded_stats {
                self.recorded_stats = false;
                self.status.store(MAPPING, Ordering::Release);
                let status = self.status.clone();
                res.readback.map_async(wgpu::MapMode::Read, .., move |result| {
                    status.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The capacity and layout constants agree with the shaders'.
    #[test]
    fn constants_match_the_shaders() {
        let common = include_str!("shaders/lumen_world_cache_common.wgsl");
        let update = include_str!("shaders/lumen_world_cache.wgsl");
        assert!(common.contains(&format!("const WC_CAPACITY: u32 = {CAPACITY}u;")));
        assert!(common.contains(&format!("const WC_SLOTS_X: u32 = {SLOTS_X}u;")));
        assert!(update.contains(&format!("const WC_RECORD: u32 = {RECORD_WORDS}u;")));
        assert!(CAPACITY.is_power_of_two() && CAPACITY % SLOTS_X == 0 && (CAPACITY / 64) <= 65535);
    }

    #[test]
    fn modes_cycle_and_parse() {
        let mut mode = WorldCacheMode::Off;
        for _ in 0..3 {
            assert_eq!(WorldCacheMode::parse(&format!("{}", mode.gpu())), Some(mode));
            mode = mode.next();
        }
        assert_eq!(mode, WorldCacheMode::Off);
        assert_eq!(WorldCacheMode::parse("on"), Some(WorldCacheMode::Fallback));
        assert_eq!(WorldCacheMode::parse("replace"), Some(WorldCacheMode::Replace));
    }
}
