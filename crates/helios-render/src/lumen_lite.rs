//! Lumen Lite: an irradiance-field final gather for weak GPUs (UE 5.8's
//! `r.Lumen.FinalGatherMethod 0`), selected with [`GiMethod::LumenLite`].
//!
//! Lumen's screen probes trace 64 rays from every 16×16 pixel tile (plus adaptive probes) every
//! frame, so their cost grows with the resolution. Lumen Lite gathers from world-space probes
//! instead (as DDGI does), which change slowly and are shared by every pixel near them:
//!
//! * **Clipmap** — [`LEVELS`] levels of [`GRID`]³ probes around the camera, the spacing doubling
//!   from [`FIRST_SPACING`] (±8 m … ±64 m). A level's window moves by whole spacings, with
//!   hysteresis; probe slots are addressed toroidally (world coordinate modulo [`GRID`]), so a
//!   move keeps every probe still inside the window and only the slab that entered is new.
//! * **Per-probe data** — an octahedral irradiance map (6×6 texels: cosine-weighted mean radiance
//!   and sky visibility) and a map of ray-distance moments (14×14: mean and mean square, for
//!   Chebyshev visibility as in DDGI), each with a one-texel border in its atlas so it can be
//!   sampled bilinearly. 43 MiB in all (atlases 40 MiB, ray and scratch buffers 3 MiB),
//!   allocated on first use.
//! * **Update budget** — [`PROBES_PER_FRAME`] probes a frame, picked on the GPU: probes that just
//!   entered first (finest levels first), then by age, four times as often on screen, an eighth
//!   as often where the distance field finds no surface around them (no pixel interpolates them,
//!   but foliage is not in the global field) and rarely when buried. A score histogram finds
//!   the threshold that fills the budget, so no CPU readback is involved. With
//!   [`crate::RenderSettings::lumen_lite_adaptive_rate`] the score also follows each probe's
//!   *variability* (how much its refreshes change it): settled probes at half the rate, changing
//!   ones up to four times, and none before a minimum weighted age ([`MIN_SCORE`]), so a steady
//!   scene refreshes fewer probes than the budget.
//! * **Tracing** — [`RAYS_PER_PROBE`] rays per probe through Lumen's own tracing (near-field
//!   mesh SDFs, global distance field, mesh-card and voxel surface caches, sky; the WGSL is
//!   shared through lumen_trace.wgsl), blended into the probe with hysteresis. Before tracing, the
//!   global field relocates a probe off a surface it sits on or in, or disables it.
//! * **Shading** — per half-resolution pixel, the 8 probes around a shading point biased along the
//!   normal and towards the eye, weighted by trilinear, backface and Chebyshev terms, blended
//!   into the next level at a window's edge and into Lumen's radiance cache beyond the last; the
//!   result replaces the screen-probe gather in `gi_raw` (same format), so GI denoising, sky
//!   visibility (specular occlusion), material AO and rough reflections — which reuse the
//!   gathered irradiance in the lighting pass — work unchanged.
//! * **Reflections** (UE 5.8's Lumen Lite: reflections fall back to SSR) — glossy reflections
//!   are traced in screen space over the closest-depth HZB (`hzb_trace.wgsl`), skipping pixels
//!   last frame did not light; where they miss, the irradiance field at the surface looked up in
//!   the reflected direction (its sky part sharpened to the sky itself), with height fog. They
//!   replace Lumen's per-pixel world traces in its reflection buffer, whose temporal filter and
//!   the lighting pass use them unchanged ([`crate::RenderSettings::lumen_lite_ssr`]).

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, IVec4, Vec3, Vec4};
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{BindingType, TextureFormat as F};

use crate::lumen::{Lumen, Stage, TraceResources};
use crate::passes::hzb::{ClosestHzb, Hzb};
use crate::passes::sky::Sky;
use crate::passes::Samplers;
use crate::profiler::Recorder;
use crate::resolution::DynamicResolution;
use crate::scene::Scene;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{bind, storage_buffer, uniform_buffer, workgroups};

/// Clipmap levels (`LITE_LEVELS` in lumen_lite.wgsl).
pub const LEVELS: usize = 4;
/// Probes per axis and level (`GRID`).
pub const GRID: u32 = 16;
/// Probe spacing of the finest level in metres; each level doubles it.
pub const FIRST_SPACING: f32 = 1.0;
/// Rays traced per probe update (`RAYS`).
pub const RAYS_PER_PROBE: u32 = 128;
/// Probes updated per frame (UE: "about 100+"; ours have fewer rays each).
pub const PROBES_PER_FRAME: u32 = 512;
/// Weight of a probe's previous irradiance and moments when it is refreshed. Probes are
/// refreshed every few dozen frames, not every frame as in DDGI, so the hysteresis is lower
/// than DDGI's 0.97 (large changes halve it, see `update_field`).
const HYSTERESIS: f32 = 0.85;
/// With the variability-driven rate, the weighted age (frames × on-screen, kind and variability
/// factors) a probe must reach before it is refreshed again (`min_score`): a settled probe on
/// screen waits 12 frames, off screen 48; a steady scene then refreshes ~300 probes a frame
/// instead of the whole budget (cathedral: 295–496, 0.28 → 0.24 ms), with the same stability.
const MIN_SCORE: f32 = 24.0;
const IRRADIANCE_TEXELS: u32 = 6;
const MOMENT_TEXELS: u32 = 14;
const PROBES: u32 = GRID * GRID * GRID * LEVELS as u32;
/// Words of the schedule buffer before the update list (`SCHEDULE_LIST`): indirect dispatch,
/// selection state and the 64-bucket score histogram.
const SCHEDULE_LIST: u32 = 72;
/// Bytes per probe of `probe_state` (`ProbeState`).
const PROBE_STATE_BYTES: u64 = 32;
/// The camera may stray this many cells from a window's centre before the window moves (half
/// a cell plus hysteresis: a camera hovering at a cell boundary doesn't re-trace slabs).
const RECENTRE_CELLS: f32 = 0.75;

/// The final gather of Lumen's diffuse GI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GiMethod {
    /// Screen probes with adaptive placement (UE5 Lumen's default final gather).
    #[default]
    ScreenProbes,
    /// World-space irradiance field (UE 5.8's Lumen Lite): much cheaper, softer, independent of
    /// the resolution; for integrated and mobile GPUs.
    LumenLite,
}

impl GiMethod {
    pub fn name(self) -> &'static str {
        match self {
            Self::ScreenProbes => "Lumen",
            Self::LumenLite => "Lumen Lite",
        }
    }

    pub fn toggled(self) -> Self {
        match self {
            Self::ScreenProbes => Self::LumenLite,
            Self::LumenLite => Self::ScreenProbes,
        }
    }

    /// `lumen` (screen probes) or `lite`.
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "lumen" | "screen" | "screen-probes" => Some(Self::ScreenProbes),
            "lite" | "lumen-lite" => Some(Self::LumenLite),
            _ => None,
        }
    }

    /// The method to start with on an adapter: Lumen Lite on integrated and software adapters.
    /// (Browsers don't report the adapter type; there [`LiteFallback`] decides from the frame
    /// times.)
    pub fn for_adapter(info: &wgpu::AdapterInfo) -> Self {
        match info.device_type {
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => Self::LumenLite,
            _ => Self::ScreenProbes,
        }
    }
}

/// Switches to Lumen Lite when dynamic resolution has been stuck at its minimum screen
/// percentage for [`LiteFallback::FRAMES`] frames in a row: the GPU cannot hold the budget even
/// at the lowest resolution, and the screen-probe gather is the largest cost that does not
/// shrink with it (enabled by [`crate::RenderSettings::gi_lite_fallback`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct LiteFallback {
    frames_at_minimum: u32,
}

impl LiteFallback {
    pub const FRAMES: u32 = 180;

    /// Observes a frame's screen percentage; true once the switch should happen.
    pub fn observe(&mut self, percentage: f32, dynamic: Option<&DynamicResolution>) -> bool {
        let at_minimum = dynamic.is_some_and(|d| percentage <= crate::resolution::clamp_percentage(d.min_percentage) + 0.5);
        self.frames_at_minimum = if at_minimum { self.frames_at_minimum + 1 } else { 0 };
        self.frames_at_minimum >= Self::FRAMES
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LiteUniform {
    levels: [Vec4; LEVELS],
    cells: [IVec4; LEVELS],
    frame: u32,
    budget: u32,
    reset: u32,
    hysteresis: f32,
    skip_unlit_hits: u32,
    reflection_fog: u32,
    adaptive_rate: u32,
    min_score: f32,
}

/// Renderer resources besides Lumen's that the irradiance field reads.
pub struct LiteInputs<'a> {
    pub frame: &'a wgpu::Buffer,
    pub targets: &'a FrameTargets,
    pub sky: &'a Sky,
    pub samplers: &'a Samplers,
    pub exposure: &'a wgpu::Buffer,
    /// Builds the closest-depth HZB of the reflections' screen traces.
    pub hzb: &'a Hzb,
    /// Last frame's lit image (the TSR history), which screen-trace hits read.
    pub previous_radiance: &'a wgpu::TextureView,
    /// The volumetric fog's uniform (height fog on reflection hits).
    pub fog: &'a wgpu::Buffer,
    /// Frame parity (Lumen's reflection history).
    pub current: usize,
    /// Screen-space reflections into Lumen's reflection buffer (Lumen traces none).
    pub reflections: bool,
}

pub struct LumenLite {
    uniform: wgpu::Buffer,
    /// Per probe slot: world coordinate, last refresh, relocation, kind (`ProbeState`).
    state: wgpu::Buffer,
    /// Indirect dispatch, selection state, score histogram, update list.
    schedule: wgpu::Buffer,
    /// Per listed probe and ray: radiance, sky, distance.
    rays: wgpu::Buffer,
    /// Per listed probe: its new interior texels, between `update_field` and `store_field`.
    scratch: wgpu::Buffer,
    irradiance: wgpu::TextureView,
    moments: wgpu::TextureView,
    /// Each level's window: its first probe coordinate.
    windows: [Option<IVec3>; LEVELS],
    geometry_version: Option<u64>,
    schedule_probes: Stage,
    select_probes: Stage,
    list_probes: Stage,
    trace: Stage,
    update_probes: Stage,
    store: Stage,
    integrate: Stage,
    reflect: Stage,
    debug: Stage,
    /// Closest-depth pyramid of the reflections' screen traces and the targets' capacity it
    /// was made for.
    hzb: Option<((u32, u32), ClosestHzb)>,
}

fn atlas(device: &wgpu::Device, label: &str, texels: u32) -> wgpu::TextureView {
    let tile = texels + 2;
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: GRID * GRID * tile, height: GRID * LEVELS as u32 * tile, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: F::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// The probe spacing of `level`.
fn spacing(level: usize) -> f32 {
    FIRST_SPACING * (1u32 << level) as f32
}

/// A level's window for a camera at `camera` (probe units): the current one while the camera
/// stays within [`RECENTRE_CELLS`] of its centre, else centred on the camera's nearest probe.
fn window(current: Option<IVec3>, camera: Vec3) -> IVec3 {
    let half = (GRID / 2) as i32;
    match current {
        Some(first) if (camera - (first + half).as_vec3()).abs().max_element() <= RECENTRE_CELLS => first,
        _ => camera.round().as_ivec3() - half,
    }
}

impl LumenLite {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "lumen_lite.wgsl");
        let u = bind::uniform;
        let ro = || bind::storage(true);
        let rw = || bind::storage(false);
        let tex = bind::texture;
        let tex3 = bind::texture_3d;
        let out = || bind::storage_texture(F::Rgba16Float);
        let stage = |entry: &str, bindings: &[(u32, BindingType)]| Stage::new(device, &module, entry, bindings);
        let schedule_list = || [(0, u()), (2, u()), (20, rw()), (21, rw())];
        let sampled_atlases = [(8, bind::sampler()), (25, tex(true)), (26, tex(true))];
        Self {
            uniform: uniform_buffer(device, "lumen lite", std::mem::size_of::<LiteUniform>() as u64),
            state: storage_buffer(device, "lumen lite probes", PROBES as u64 * PROBE_STATE_BYTES, wgpu::BufferUsages::empty()),
            schedule: storage_buffer(device, "lumen lite schedule", (SCHEDULE_LIST + PROBES_PER_FRAME) as u64 * 4, wgpu::BufferUsages::INDIRECT),
            rays: storage_buffer(device, "lumen lite rays", (PROBES_PER_FRAME * RAYS_PER_PROBE) as u64 * 16, wgpu::BufferUsages::empty()),
            scratch: storage_buffer(
                device,
                "lumen lite scratch",
                (PROBES_PER_FRAME * (IRRADIANCE_TEXELS * IRRADIANCE_TEXELS + MOMENT_TEXELS * MOMENT_TEXELS)) as u64 * 16,
                wgpu::BufferUsages::empty(),
            ),
            irradiance: atlas(device, "lumen lite irradiance", IRRADIANCE_TEXELS),
            moments: atlas(device, "lumen lite moments", MOMENT_TEXELS),
            windows: [None; LEVELS],
            geometry_version: None,
            schedule_probes: stage("schedule", &schedule_list()),
            select_probes: stage("select_probes", &[(2, u()), (21, rw())]),
            list_probes: stage("list_probes", &schedule_list()),
            trace: stage(
                "trace_field",
                &[
                    (0, u()),
                    (1, u()),
                    (2, u()),
                    (5, tex(true)),
                    (6, tex3(true)),
                    (7, tex3(true)),
                    (8, bind::sampler()),
                    (9, ro()),
                    (10, ro()),
                    (11, ro()),
                    (12, tex3(true)),
                    (20, rw()),
                    (22, ro()),
                    (23, rw()),
                    (70, ro()),
                    (71, ro()),
                    (72, bind::texture_uint()),
                    (73, tex(false)),
                    (74, bind::texture_3d_uint()),
                    (150, bind::texture_uint()),
                ],
            ),
            update_probes: stage("update_field", &[&[(2, u()), (20, rw()), (22, ro()), (23, rw()), (24, rw())], &sampled_atlases[1..]].concat()),
            store: stage("store_field", &[(2, u()), (20, rw()), (22, ro()), (24, rw()), (27, out()), (28, out())]),
            integrate: stage(
                "integrate_field",
                &[&[(0, u()), (1, u()), (2, u()), (3, tex(false)), (4, tex(false)), (9, ro()), (13, ro()), (20, rw()), (29, out())], &sampled_atlases[..]].concat(),
            ),
            reflect: stage(
                "reflect_field",
                &[
                    &[
                        (0, u()),
                        (1, u()),
                        (2, u()),
                        (3, tex(false)),
                        (4, tex(false)),
                        (5, tex(true)),
                        (6, tex3(true)),
                        (9, ro()),
                        (10, ro()),
                        (13, ro()),
                        (20, rw()),
                        (32, tex(false)),
                        (33, tex(true)),
                        (34, tex(false)),
                        (35, out()),
                        (36, tex(true)),
                        (37, u()),
                    ],
                    &sampled_atlases[..],
                ]
                .concat(),
            ),
            debug: stage(
                "debug_field",
                &[&[(0, u()), (2, u()), (3, tex(false)), (10, ro()), (20, rw()), (30, tex(false)), (31, out())], &sampled_atlases[..]].concat(),
            ),
            hzb: None,
        }
    }

    /// Moves the clipmap windows with the camera and writes the uniform. A scene with new
    /// geometry forgets every probe.
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, camera_position: Vec3, frame: u32, settings: &crate::RenderSettings, capacity: (u32, u32)) {
        if settings.lumen_lite_ssr && self.hzb.as_ref().is_none_or(|h| h.0 != capacity) {
            self.hzb = Some((capacity, ClosestHzb::new(device, capacity)));
        }
        let reset = self.geometry_version != Some(scene.geometry_version);
        self.geometry_version = Some(scene.geometry_version);
        let mut levels = [Vec4::ZERO; LEVELS];
        let mut cells = [IVec4::ZERO; LEVELS];
        for (level, current) in self.windows.iter_mut().enumerate() {
            let spacing = spacing(level);
            let first = window(*current, camera_position / spacing);
            *current = Some(first);
            levels[level] = (first.as_vec3() * spacing).extend(spacing);
            cells[level] = first.extend(0);
        }
        let uniform = LiteUniform {
            levels,
            cells,
            frame,
            budget: PROBES_PER_FRAME,
            reset: reset as u32,
            hysteresis: HYSTERESIS,
            skip_unlit_hits: settings.lumen_skip_unlit_hits as u32,
            reflection_fog: settings.lumen_reflection_fog as u32,
            adaptive_rate: settings.lumen_lite_adaptive_rate as u32,
            min_score: MIN_SCORE,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Updates this frame's probes and gathers the field into `targets.gi_raw` (after Lumen's
    /// surface caches are lit, in place of its screen probes); with `inputs.reflections`, traces
    /// the reflections into Lumen's reflection buffer and runs its temporal filter.
    pub fn record(&self, rec: &mut Recorder, inputs: &LiteInputs, lumen_ref: &Lumen) {
        let device = rec.device;
        let lumen: TraceResources = lumen_ref.trace_resources();
        let lumen_uniform = lumen.uniform;
        let frame = || (0, inputs.frame.as_entire_binding());
        let lite = || (2, self.uniform.as_entire_binding());
        let state = || (20, self.state.as_entire_binding());
        let schedule = || (21, self.schedule.as_entire_binding());
        let list = || (22, self.schedule.as_entire_binding());
        let sampler = || (8, Sampler(&inputs.samplers.linear));
        let atlases = || [(25, TextureView(&self.irradiance)), (26, TextureView(&self.moments))];

        let scheduling = |stage: &Stage| stage.group(device, &[frame(), lite(), state(), schedule()]);
        let schedule_group = scheduling(&self.schedule_probes);
        let select_group = self.select_probes.group(device, &[lite(), schedule()]);
        let list_group = scheduling(&self.list_probes);
        let trace_group = self.trace.group(
            device,
            &[
                frame(),
                (1, lumen_uniform.as_entire_binding()),
                lite(),
                (5, TextureView(&inputs.sky.sky_view)),
                (6, TextureView(lumen.global_sdf)),
                (7, TextureView(lumen.voxel_lighting)),
                sampler(),
                (9, lumen.radiance_probes.as_entire_binding()),
                (10, inputs.exposure.as_entire_binding()),
                (11, lumen.near_field.0.as_entire_binding()),
                (12, TextureView(lumen.near_field.1)),
                state(),
                list(),
                (23, self.rays.as_entire_binding()),
                (70, lumen.card_data.as_entire_binding()),
                (71, lumen.card_pages.as_entire_binding()),
                (72, TextureView(lumen.card_capture)),
                (73, TextureView(lumen.card_lighting)),
                (74, TextureView(lumen.instance_ids)),
                (150, TextureView(lumen.world_cache)),
            ],
        );
        let [irradiance, moments] = atlases();
        let update_group = self.update_probes.group(
            device,
            &[lite(), state(), list(), (23, self.rays.as_entire_binding()), (24, self.scratch.as_entire_binding()), irradiance, moments],
        );
        let store_group = self.store.group(
            device,
            &[
                lite(),
                state(),
                list(),
                (24, self.scratch.as_entire_binding()),
                (27, TextureView(&self.irradiance)),
                (28, TextureView(&self.moments)),
            ],
        );
        let [irradiance, moments] = atlases();
        let targets = inputs.targets;
        let integrate_group = self.integrate.group(
            device,
            &[
                frame(),
                (1, lumen_uniform.as_entire_binding()),
                lite(),
                (3, TextureView(&targets.depth)),
                (4, TextureView(&targets.gbuffer_normal)),
                sampler(),
                (9, lumen.radiance_probes.as_entire_binding()),
                (13, inputs.sky.sh.as_entire_binding()),
                state(),
                (29, TextureView(&targets.gi_raw)),
                irradiance,
                moments,
            ],
        );

        let probe_groups = workgroups(PROBES, 64);
        let mut pass = rec.compute_pass("lumen lite probes");
        pass.set_pipeline(&self.schedule_probes.pipeline);
        pass.set_bind_group(0, &schedule_group, &[]);
        pass.dispatch_workgroups(probe_groups, 1, 1);
        pass.set_pipeline(&self.select_probes.pipeline);
        pass.set_bind_group(0, &select_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.list_probes.pipeline);
        pass.set_bind_group(0, &list_group, &[]);
        pass.dispatch_workgroups(probe_groups, 1, 1);
        for (stage, group) in [(&self.trace, &trace_group), (&self.update_probes, &update_group), (&self.store, &store_group)] {
            pass.set_pipeline(&stage.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups_indirect(&self.schedule, 0);
        }
        drop(pass);
        let half = (targets.width.div_ceil(2), targets.height.div_ceil(2));
        let mut pass = rec.compute_pass("lumen lite gather");
        pass.set_pipeline(&self.integrate.pipeline);
        pass.set_bind_group(0, &integrate_group, &[]);
        pass.dispatch_workgroups(workgroups(half.0, 8), workgroups(half.1, 8), 1);
        drop(pass);

        let (Some(hzb), Some(reflection_raw)) = (self.hzb.as_ref().map(|h| &h.1).filter(|_| inputs.reflections), lumen_ref.reflection_raw()) else { return };
        inputs.hzb.record_closest(rec, targets, hzb);
        let [irradiance, moments] = atlases();
        let reflect_group = self.reflect.group(
            device,
            &[
                frame(),
                (1, lumen_uniform.as_entire_binding()),
                lite(),
                (3, TextureView(&targets.depth)),
                (4, TextureView(&targets.gbuffer_normal)),
                (5, TextureView(&inputs.sky.sky_view)),
                (6, TextureView(lumen.global_sdf)),
                sampler(),
                (9, lumen.radiance_probes.as_entire_binding()),
                (10, inputs.exposure.as_entire_binding()),
                (13, inputs.sky.sh.as_entire_binding()),
                state(),
                (32, TextureView(&targets.gbuffer_motion)),
                (33, TextureView(inputs.previous_radiance)),
                (34, TextureView(&hzb.view)),
                (35, TextureView(reflection_raw)),
                (36, TextureView(&inputs.sky.transmittance)),
                (37, inputs.fog.as_entire_binding()),
                irradiance,
                moments,
            ],
        );
        let mut pass = rec.compute_pass("lumen lite reflections");
        pass.set_pipeline(&self.reflect.pipeline);
        pass.set_bind_group(0, &reflect_group, &[]);
        pass.dispatch_workgroups(workgroups(half.0, 8), workgroups(half.1, 8), 1);
        drop(pass);
        lumen_ref.record_reflection_temporal(rec, inputs.frame, targets, inputs.samplers, inputs.current, inputs.exposure);
    }

    /// Draws the probes into `targets.hdr` over the gathered irradiance (`gi`: the denoised GI
    /// of this frame) ([`crate::DebugView::LumenLiteProbes`]).
    pub fn record_debug(&self, rec: &mut Recorder, inputs: &LiteInputs, gi: &wgpu::TextureView) {
        let targets = inputs.targets;
        let group = self.debug.group(
            rec.device,
            &[
                (0, inputs.frame.as_entire_binding()),
                (2, self.uniform.as_entire_binding()),
                (3, TextureView(&targets.depth)),
                (8, Sampler(&inputs.samplers.linear)),
                (10, inputs.exposure.as_entire_binding()),
                (20, self.state.as_entire_binding()),
                (25, TextureView(&self.irradiance)),
                (26, TextureView(&self.moments)),
                (30, TextureView(gi)),
                (31, TextureView(&targets.hdr)),
            ],
        );
        let mut pass = rec.compute_pass("lumen lite debug");
        pass.set_pipeline(&self.debug.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(targets.width, 8), workgroups(targets.height, 8), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_scroll_by_whole_cells_with_hysteresis() {
        let half = (GRID / 2) as i32;
        let first = window(None, Vec3::new(0.2, 0.0, -0.4));
        assert_eq!(first, IVec3::splat(-half));
        // Small moves around the centre keep the window (and every probe in it).
        for p in [Vec3::new(0.7, 0.0, 0.0), Vec3::new(-0.74, 0.5, 0.74)] {
            assert_eq!(window(Some(first), p), first);
        }
        // Leaving the hysteresis band recentres on the nearest probe.
        assert_eq!(window(Some(first), Vec3::new(0.8, 0.0, 0.0)), IVec3::new(1 - half, -half, -half));
        assert_eq!(window(Some(first), Vec3::new(-3.6, 0.0, 0.0)), IVec3::new(-4 - half, -half, -half));
    }

    #[test]
    fn finer_windows_nest_inside_coarser_ones() {
        // Wherever the camera wanders, every point of a level's window (where its pixels sample
        // it) lies inside the next coarser window, so the blend at the edge has probes to go to.
        let mut windows = [None; LEVELS];
        let mut camera = Vec3::new(3.3, 1.7, -2.2);
        for step in 0..500 {
            camera += Vec3::new((step as f32 * 0.37).sin(), (step as f32 * 0.11).cos() * 0.3, 0.8) * 0.45;
            for (level, current) in windows.iter_mut().enumerate() {
                *current = Some(window(*current, camera / spacing(level)));
            }
            for level in 0..LEVELS - 1 {
                let (fine, coarse) = (spacing(level), spacing(level + 1));
                let lo = windows[level].unwrap().as_vec3() * fine;
                let hi = lo + Vec3::splat((GRID - 1) as f32 * fine);
                let clo = windows[level + 1].unwrap().as_vec3() * coarse;
                let chi = clo + Vec3::splat((GRID - 1) as f32 * coarse);
                assert!(lo.cmpge(clo).all() && hi.cmple(chi).all(), "level {level} leaves level {} at {camera}", level + 1);
            }
        }
    }

    #[test]
    fn fallback_needs_a_persistent_minimum() {
        let dynamic = DynamicResolution::with_budget(14.0);
        let mut fallback = LiteFallback::default();
        for _ in 0..LiteFallback::FRAMES - 1 {
            assert!(!fallback.observe(50.0, Some(&dynamic)));
        }
        // One frame above the minimum starts the count again.
        assert!(!fallback.observe(55.0, Some(&dynamic)));
        for _ in 0..LiteFallback::FRAMES - 1 {
            assert!(!fallback.observe(50.0, Some(&dynamic)));
        }
        assert!(fallback.observe(50.0, Some(&dynamic)));
        // Without dynamic resolution a low fixed percentage is a choice, not a symptom.
        let mut fixed = LiteFallback::default();
        assert!((0..2 * LiteFallback::FRAMES).all(|_| !fixed.observe(50.0, None)));
    }

    #[test]
    fn gi_method_parses_and_toggles() {
        assert_eq!(GiMethod::parse("lite"), Some(GiMethod::LumenLite));
        assert_eq!(GiMethod::parse("Lumen"), Some(GiMethod::ScreenProbes));
        assert_eq!(GiMethod::parse("rtgi"), None);
        assert_eq!(GiMethod::ScreenProbes.toggled().toggled(), GiMethod::ScreenProbes);
    }
}

