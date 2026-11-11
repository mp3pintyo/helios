//! Heterogeneous volumes: smoke, fire and mist in boxes of voxels (UE's Heterogeneous Volumes).
//!
//! A [`VolumeEmitter`] is a box with a grid of soot density, temperature and fuel. Its content
//! comes from a GPU fluid simulation ([`FluidSimulation`]: buoyancy, vorticity confinement,
//! a pressure projection, MacCormack advection, combustion; `shaders/volume_sim.wgsl`) or from
//! animated noise ([`ProceduralVolume`]: distant or static media such as mist), and is rendered
//! by `shaders/volume_render.wgsl`:
//!
//! * a **lighting cache** per emitter at half the grid resolution holds, per voxel, the sun's
//!   transmittance (a march through the volume towards the sun, times the virtual shadow map
//!   and cloud shadows) and the isotropically scattered ambient (Lumen's radiance cache with the
//!   sky occluded by the volume above) and local light — so the march needs no secondary rays;
//! * the **march** runs per pixel of the visible volumes' screen rectangle at the internal
//!   resolution, front to back through up to [`MAX_VISIBLE`] boxes ordered per pixel, stopping
//!   at the opaque depth; temperature emits blackbody radiance (fire). The ray is split at the
//!   nearest translucent surface: the part behind it is composited before the translucency pass
//!   (glass and water show it), the part in front after (`oit.rs`); both are fogged at their
//!   opacity-weighted depth by the froxel fog and aerial perspective;
//! * a **temporal resolve** reprojects the last result with the depth of the volume each pixel
//!   sees, clamps it to the current neighbourhood and blends: the per-frame jitter of the step
//!   offsets averages out without ghosting the moving flames.
//!
//! A fire can drive a scene light ([`VolumeEmitter::light`]): its colour follows the summed
//! emission of the simulation relative to its running average (on the GPU, a frame late, no
//! read-back), so the fire flickers on its surroundings.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, UVec3, Vec2, Vec3, Vec4};
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat};

use crate::passes::shading::LightingInputs;
use crate::profiler::Recorder;
use crate::scene::{LightHandle, Scene};
use crate::shader::create_module;
use crate::avboit::Avboit;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, storage_buffer, uniform_buffer, workgroups};
use crate::pipelines::Lazy;
use crate::volume_shadow::{Caster, VolumeShadows};

/// Volumes marched per frame (the nearest visible ones).
pub(crate) const MAX_VISIBLE: usize = 4;
/// Emitters simulated per frame (and fire lights).
const MAX_EMITTERS: usize = 8;
const FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// Emitters farther than this from the camera are not simulated (they keep their last state).
const SIMULATION_DISTANCE: f32 = 250.0;
/// Local lights a volume's lighting cache considers (the brightest whose range reaches it).
const MAX_CACHE_LIGHTS: usize = 8;

/// A box of participating medium: smoke, fire or mist.
#[derive(Clone, Copy, Debug)]
pub struct VolumeEmitter {
    /// World position of the centre of the box's base.
    pub position: Vec3,
    /// World extent of the box.
    pub size: Vec3,
    /// Voxels along each axis (cells should be roughly cubic).
    pub resolution: UVec3,
    pub source: VolumeSource,
    /// Single-scattering albedo (colour) of the smoke.
    pub albedo: Vec3,
    /// Extinction per metre at density 1.
    pub extinction: f32,
    /// Henyey–Greenstein phase anisotropy (0 isotropic, towards 1 forward scattering).
    pub anisotropy: f32,
    /// Scale of the blackbody emission of hot gas (0: smoke only).
    pub emission: f32,
    /// Temperatures (kelvin) that a normalised temperature of 0 and 1 emit at.
    pub emission_kelvin: Vec2,
    /// Scale of the ambient light the medium scatters.
    pub ambient: f32,
    /// A scene light that the fire drives: its colour × intensity is scaled by the simulated
    /// emission relative to its running average (it flickers with the flames).
    pub light: Option<LightHandle>,
}

/// Where a volume's content comes from.
#[derive(Clone, Copy, Debug)]
pub enum VolumeSource {
    Simulated(FluidSimulation),
    Procedural(ProceduralVolume),
}

/// A smoke / fire simulation (`shaders/volume_sim.wgsl`). Temperatures are normalised: 0 is
/// the ambient air, 1 the hottest flame.
#[derive(Clone, Copy, Debug)]
pub struct FluidSimulation {
    /// Centre of the source above `position`, and its radius (m).
    pub source_offset: Vec3,
    pub source_radius: f32,
    /// Velocity the source drives the gas towards (m/s).
    pub source_velocity: Vec3,
    /// Soot and fuel injected per second (density units) and the source's temperature.
    pub soot_rate: f32,
    pub fuel_rate: f32,
    pub temperature: f32,
    /// How strongly noise modulates the injection (0 steady, 1 a licking flame).
    pub flicker: f32,
    /// Upward acceleration per unit temperature (m/s²) and downward per unit soot.
    pub buoyancy: f32,
    pub soot_weight: f32,
    /// Vorticity confinement strength.
    pub vorticity: f32,
    /// Fuel burnt per second (fraction), and the heat and soot one unit of fuel releases.
    pub burn_rate: f32,
    pub heat: f32,
    pub soot_yield: f32,
    /// Per second: soot dissipation, cooling, velocity damping.
    pub soot_dissipation: f32,
    pub cooling: f32,
    pub velocity_damping: f32,
    /// Ambient air flow (m/s) and how fast the gas relaxes to it (per second).
    pub wind: Vec3,
    pub wind_coupling: f32,
    /// Red-black SOR sweeps of the pressure solve per frame.
    pub pressure_iterations: u32,
}

/// Density from animated noise: no simulation (mist, haze banks, distant smoke).
#[derive(Clone, Copy, Debug)]
pub struct ProceduralVolume {
    /// Peak density.
    pub density: f32,
    /// Noise frequency (per metre).
    pub frequency: f32,
    /// Exponential falloff of the density per metre above the base.
    pub height_falloff: f32,
    /// Fraction of the box the medium covers (0..1).
    pub coverage: f32,
    /// Fraction of the half extent over which it fades towards the sides.
    pub edge_fade: f32,
    /// Drift of the noise (m/s).
    pub wind: Vec3,
}

impl VolumeEmitter {
    /// A campfire: a flickering flame over a bed of logs with a column of smoke above it.
    pub fn campfire(position: Vec3) -> Self {
        Self {
            position,
            size: Vec3::new(1.6, 3.2, 1.6),
            resolution: UVec3::new(40, 80, 40),
            source: VolumeSource::Simulated(FluidSimulation {
                source_offset: Vec3::new(0.0, 0.18, 0.0),
                source_radius: 0.3,
                source_velocity: Vec3::new(0.0, 1.0, 0.0),
                soot_rate: 0.1,
                fuel_rate: 10.0,
                temperature: 0.7,
                flicker: 0.85,
                buoyancy: 9.0,
                soot_weight: 0.4,
                vorticity: 2.5,
                burn_rate: 3.0,
                heat: 0.4,
                soot_yield: 0.03,
                soot_dissipation: 0.3,
                cooling: 2.5,
                velocity_damping: 0.1,
                wind: Vec3::new(0.15, 0.0, 0.05),
                wind_coupling: 0.3,
                pressure_iterations: 16,
            }),
            albedo: Vec3::new(0.78, 0.76, 0.74),
            extinction: 6.0,
            anisotropy: 0.2,
            emission: 2.0,
            emission_kelvin: Vec2::new(700.0, 1850.0),
            ambient: 1.0,
            light: None,
        }
    }

    /// A thin, curling wisp of incense smoke.
    pub fn incense(position: Vec3) -> Self {
        Self {
            position,
            size: Vec3::new(0.8, 2.8, 0.8),
            resolution: UVec3::new(32, 112, 32),
            source: VolumeSource::Simulated(FluidSimulation {
                source_offset: Vec3::new(0.0, 0.06, 0.0),
                source_radius: 0.045,
                source_velocity: Vec3::new(0.0, 0.6, 0.0),
                soot_rate: 40.0,
                fuel_rate: 0.0,
                temperature: 0.4,
                flicker: 0.3,
                buoyancy: 10.0,
                soot_weight: 0.0,
                vorticity: 5.0,
                burn_rate: 0.0,
                heat: 0.0,
                soot_yield: 0.0,
                soot_dissipation: 0.1,
                cooling: 0.35,
                velocity_damping: 0.05,
                wind: Vec3::new(0.05, 0.0, 0.0),
                wind_coupling: 0.08,
                pressure_iterations: 12,
            }),
            albedo: Vec3::new(0.8, 0.82, 0.86),
            extinction: 7.0,
            anisotropy: 0.45,
            emission: 0.0,
            emission_kelvin: Vec2::new(700.0, 1950.0),
            ambient: 1.0,
            light: None,
        }
    }

    /// Low mist drifting over water or a valley floor: a box of `size` centred on `centre`'s
    /// x and z, its base at `centre.y`.
    pub fn mist(centre: Vec3, size: Vec3) -> Self {
        let cells = 0.35;
        Self {
            position: centre,
            size,
            resolution: (size / cells).round().as_uvec3().max(UVec3::splat(4)).min(UVec3::new(128, 32, 128)),
            source: VolumeSource::Procedural(ProceduralVolume {
                density: 1.0,
                frequency: 0.14,
                height_falloff: 0.9,
                coverage: 0.6,
                edge_fade: 0.35,
                wind: Vec3::new(0.35, 0.02, 0.12),
            }),
            albedo: Vec3::splat(0.96),
            extinction: 0.5,
            anisotropy: 0.55,
            emission: 0.0,
            emission_kelvin: Vec2::new(700.0, 1950.0),
            ambient: 1.0,
            light: None,
        }
    }

    fn origin(&self) -> Vec3 {
        self.position - Vec3::new(self.size.x, 0.0, self.size.z) * 0.5
    }

    fn cell(&self) -> Vec3 {
        self.size / self.resolution.max(UVec3::ONE).as_vec3()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimUniform {
    resolution: [u32; 4],
    cell: Vec4,
    source: Vec4,
    source_rates: Vec4,
    source_velocity: Vec4,
    forces: Vec4,
    dissipation: Vec4,
    combustion: Vec4,
    wind: Vec4,
    procedural: Vec4,
    drift: Vec4,
    emission: Vec4,
}

/// `VolumeShading` in volume_render.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct ShadingUniform {
    origin: Vec4,
    size: Vec4,
    albedo: Vec4,
    emission: Vec4,
    lights: [u32; 8],
    params: [u32; 4],
    /// x: the emitter's box in the volume shadows (its own shadow is the cache's march).
    shadow: [u32; 4],
}

/// `VolumeMarch` in volume_render.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MarchUniform {
    rect: [u32; 4],
    previous_rect: [u32; 4],
    count: [u32; 4],
    volumes: [ShadingUniform; MAX_VISIBLE],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct FlickerLight {
    color: Vec4,
    index: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FlickerUniform {
    count: [u32; 4],
    lights: [FlickerLight; MAX_EMITTERS],
}

fn volume_texture(device: &wgpu::Device, label: &str, size: UVec3) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.x, height: size.y, depth_or_array_layers: size.z },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// The GPU side of one emitter: simulation fields and the lighting cache.
struct EmitterGpu {
    resolution: UVec3,
    simulated: bool,
    /// Velocity: the frame's state (0), after forces (1), projected (2).
    velocity: [wgpu::TextureView; 3],
    /// Soot, temperature, fuel: the frame's state (0), after forces and sources (1), advected (2).
    state: [wgpu::TextureView; 3],
    curl: wgpu::TextureView,
    cache: wgpu::TextureView,
    cache_size: UVec3,
    pressure: wgpu::Buffer,
    divergence: wgpu::Buffer,
    sim: wgpu::Buffer,
    shading: wgpu::Buffer,
    /// Run the simulation (or noise) this frame; render it this frame.
    update: bool,
    visible: bool,
    iterations: u32,
}

impl EmitterGpu {
    fn new(device: &wgpu::Device, emitter: &VolumeEmitter) -> Self {
        let resolution = emitter.resolution.max(UVec3::splat(4));
        let simulated = matches!(emitter.source, VolumeSource::Simulated(_));
        // Procedural volumes only need their state texture.
        let fields = if simulated { resolution } else { UVec3::ONE };
        let cells = (fields.x * fields.y * fields.z) as u64;
        let cache_size = (resolution + 1) / 2;
        Self {
            resolution,
            simulated,
            velocity: [0, 1, 2].map(|i| volume_texture(device, &format!("volume velocity {i}"), fields)),
            state: [0, 1, 2].map(|i| volume_texture(device, &format!("volume state {i}"), if i == 0 { resolution } else { fields })),
            curl: volume_texture(device, "volume curl", fields),
            cache: volume_texture(device, "volume lighting cache", cache_size),
            cache_size,
            pressure: storage_buffer(device, "volume pressure", cells * 4, wgpu::BufferUsages::empty()),
            divergence: storage_buffer(device, "volume divergence", cells * 4, wgpu::BufferUsages::empty()),
            sim: uniform_buffer(device, "volume simulation", std::mem::size_of::<SimUniform>() as u64),
            shading: uniform_buffer(device, "volume shading", std::mem::size_of::<ShadingUniform>() as u64),
            update: false,
            visible: false,
            iterations: 0,
        }
    }
}

/// Screen-space rectangle in internal pixels: min xy, max xy (exclusive).
type Rect = [u32; 4];

/// The march output and its temporal history, at the internal resolution's capacity.
struct ScreenTargets {
    size: (u32, u32),
    /// This frame's march: in front of, behind the nearest translucent surface.
    raw: [wgpu::TextureView; 2],
    /// Resolved front and back, ping-pong by frame.
    history: [[wgpu::TextureView; 2]; 2],
}

impl ScreenTargets {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let make = |label: &str| {
            crate::util::texture(
                device,
                crate::util::TextureSpec {
                    label,
                    width: size.0,
                    height: size.1,
                    layers: 1,
                    mips: 1,
                    format: FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                },
            )
            .create_view(&Default::default())
        };
        Self {
            size,
            raw: [make("volume front"), make("volume back")],
            history: [[make("volume front 0"), make("volume back 0")], [make("volume front 1"), make("volume back 1")]],
        }
    }
}

pub(crate) struct HeterogeneousVolumes {
    curl: Lazy<wgpu::ComputePipeline>,
    forces: Lazy<wgpu::ComputePipeline>,
    divergence: Lazy<wgpu::ComputePipeline>,
    pressure: [Lazy<wgpu::ComputePipeline>; 2],
    project: Lazy<wgpu::ComputePipeline>,
    advect: Lazy<wgpu::ComputePipeline>,
    maccormack: Lazy<wgpu::ComputePipeline>,
    procedural: Lazy<wgpu::ComputePipeline>,
    flicker: Lazy<wgpu::ComputePipeline>,
    light_cache: Lazy<wgpu::ComputePipeline>,
    march: Lazy<wgpu::ComputePipeline>,
    /// The volumes' optical depth in the AVBOIT grid.
    avboit_inject: Lazy<wgpu::ComputePipeline>,
    resolve: Lazy<wgpu::ComputePipeline>,
    curl_layout: wgpu::BindGroupLayout,
    forces_layout: wgpu::BindGroupLayout,
    divergence_layout: wgpu::BindGroupLayout,
    pressure_layout: wgpu::BindGroupLayout,
    project_layout: wgpu::BindGroupLayout,
    advect_layout: wgpu::BindGroupLayout,
    maccormack_layout: wgpu::BindGroupLayout,
    procedural_layout: wgpu::BindGroupLayout,
    flicker_layout: wgpu::BindGroupLayout,
    cache_layout: wgpu::BindGroupLayout,
    march_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    emitters: Vec<EmitterGpu>,
    /// Per emitter: the fire's summed emission (fixed point), written by the simulation.
    emission: wgpu::Buffer,
    flicker_uniform: wgpu::Buffer,
    /// Running average of each fire light's emission.
    flicker_state: wgpu::Buffer,
    flicker_count: u32,
    march_uniform: wgpu::Buffer,
    /// Indices of the emitters marched this frame.
    visible: Vec<usize>,
    rect: Rect,
    previous_rect: Rect,
    targets: Option<ScreenTargets>,
    /// A 1³ volume bound for the unused march slots.
    empty: wgpu::TextureView,
    time: f32,
    frame: usize,
    /// The volumes' shadows on the scene, the fog, particles and each other.
    pub(crate) shadows: VolumeShadows,
}

impl HeterogeneousVolumes {
    pub fn new(device: &wgpu::Device) -> Self {
        let sim = create_module(device, "volume_sim.wgsl");
        let render = create_module(device, "volume_render.wgsl");
        let c = ShaderStages::COMPUTE;
        let layout = |label: &str, entries: &[(u32, wgpu::BindingType)]| bind_group_layout(device, label, c, entries);
        let tex = || bind::texture_3d(true);
        let out = || bind::storage_texture_3d(FORMAT);
        let curl_layout = layout("volume curl", &[(0, bind::uniform()), (1, tex()), (9, out())]);
        let forces_layout = layout("volume forces", &[(0, bind::uniform()), (1, tex()), (3, tex()), (5, tex()), (7, out()), (8, out())]);
        let divergence_layout = layout("volume divergence", &[(0, bind::uniform()), (1, tex()), (11, bind::storage(false))]);
        let pressure_layout = layout("volume pressure", &[(0, bind::uniform()), (10, bind::storage(false)), (14, bind::storage(true))]);
        let project_layout = layout("volume project", &[(0, bind::uniform()), (1, tex()), (7, out()), (13, bind::storage(true))]);
        let advect_layout = layout("volume advect", &[(0, bind::uniform()), (1, tex()), (3, tex()), (6, bind::sampler()), (7, out()), (8, out())]);
        let maccormack_layout =
            layout("volume maccormack", &[(0, bind::uniform()), (1, tex()), (3, tex()), (4, tex()), (6, bind::sampler()), (8, out()), (12, bind::storage(false))]);
        let procedural_layout = layout("volume procedural", &[(0, bind::uniform()), (8, out())]);
        let flicker_layout =
            layout("volume fire lights", &[(12, bind::storage(false)), (20, bind::uniform()), (21, bind::storage(false)), (22, bind::storage(false))]);
        let mut cache_entries = crate::media::lighting_entries();
        cache_entries.extend([(70, bind::uniform()), (71, tex()), (72, out())]);
        let cache_layout = layout("volume lighting cache", &cache_entries);
        let mut march_entries = crate::media::fog_entries();
        march_entries.extend((81..=88).map(|b| (b, tex())));
        march_entries.extend([
            (80, bind::uniform()),
            (89, bind::texture(false)),
            (140, bind::texture(false)),
            (141, bind::storage_texture(FORMAT)),
            (142, bind::storage_texture(FORMAT)),
        ]);
        // The AVBOIT grid: the march reads it, the volumes' bounds and injection write it.
        march_entries.extend(crate::avboit::Avboit::read_entries());
        march_entries.extend(crate::avboit::Avboit::write_entries().into_iter().filter(|(b, _)| *b != 120));
        let march_layout = layout("volume march", &march_entries);
        let resolve_layout = layout(
            "volume resolve",
            &[
                (0, bind::uniform()),
                (11, bind::sampler()),
                (80, bind::uniform()),
                (89, bind::texture(false)),
                (141, bind::storage_texture(FORMAT)),
                (142, bind::storage_texture(FORMAT)),
                (143, bind::texture(true)),
                (144, bind::texture(true)),
                (145, bind::texture(true)),
                (146, bind::texture(true)),
            ],
        );
        // Compiled when a scene first has volumes (in the browser in the background: `update`).
        let pipeline = |label: &str, module: &wgpu::ShaderModule, entry: &str, layout: &wgpu::BindGroupLayout| {
            lazy_compute_pipeline(device, label, module, entry, layout, &[])
        };
        let pressure = [0.0, 1.0].map(|parity| lazy_compute_pipeline(device, "volume pressure", &sim, "pressure_sweep", &pressure_layout, &[("PARITY", parity)]));
        Self {
            curl: pipeline("volume curl", &sim, "curl", &curl_layout),
            forces: pipeline("volume forces", &sim, "forces", &forces_layout),
            divergence: pipeline("volume divergence", &sim, "divergence", &divergence_layout),
            pressure,
            project: pipeline("volume project", &sim, "project", &project_layout),
            advect: pipeline("volume advect", &sim, "advect", &advect_layout),
            maccormack: pipeline("volume maccormack", &sim, "maccormack", &maccormack_layout),
            procedural: pipeline("volume procedural", &sim, "procedural", &procedural_layout),
            flicker: pipeline("volume fire lights", &sim, "flicker_lights", &flicker_layout),
            light_cache: pipeline("volume lighting cache", &render, "light_cache", &cache_layout),
            march: pipeline("volume march", &render, "march", &march_layout),
            avboit_inject: pipeline("volume avboit transmittance", &render, "avboit_volume_inject", &march_layout),
            resolve: pipeline("volume resolve", &render, "resolve", &resolve_layout),
            curl_layout,
            forces_layout,
            divergence_layout,
            pressure_layout,
            project_layout,
            advect_layout,
            maccormack_layout,
            procedural_layout,
            flicker_layout,
            cache_layout,
            march_layout,
            resolve_layout,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("volume linear clamp"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            emitters: Vec::new(),
            emission: storage_buffer(device, "volume emission", MAX_EMITTERS as u64 * 4, wgpu::BufferUsages::empty()),
            flicker_uniform: uniform_buffer(device, "volume fire lights", std::mem::size_of::<FlickerUniform>() as u64),
            flicker_state: storage_buffer(device, "volume fire light state", MAX_EMITTERS as u64 * 16, wgpu::BufferUsages::empty()),
            flicker_count: 0,
            march_uniform: uniform_buffer(device, "volume march", std::mem::size_of::<MarchUniform>() as u64),
            visible: Vec::new(),
            rect: [0; 4],
            previous_rect: [0; 4],
            targets: None,
            empty: volume_texture(device, "volume empty", UVec3::ONE),
            time: 0.0,
            frame: 0,
            shadows: VolumeShadows::new(device),
        }
    }

    /// Whether any volume is marched this frame.
    pub fn visible(&self) -> bool {
        !self.visible.is_empty()
    }

    /// The pixels the resolved volume textures hold this frame.
    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// The resolved front (0) and back (1) parts of this frame (after [`Self::record`]).
    pub fn resolved(&self) -> Option<&[wgpu::TextureView; 2]> {
        self.targets.as_ref().filter(|_| self.visible()).map(|t| &t.history[self.frame & 1])
    }

    /// Mirrors the scene's emitters, picks which to simulate and march, uploads their uniforms.
    /// `enabled` false skips everything (fire lights keep the scene's intensity).
    #[allow(clippy::too_many_arguments)]
    /// Queues the pipelines for the browser's background compile if the scene has volumes.
    pub fn prefetch(&self, scene: &Scene) {
        if !scene.volumes.is_empty() {
            for pipeline in [&self.curl, &self.forces, &self.divergence, &self.project, &self.advect, &self.maccormack, &self.procedural, &self.flicker, &self.light_cache, &self.march, &self.resolve] {
                pipeline.prefetch();
            }
            self.pressure.iter().for_each(Lazy::prefetch);
            self.shadows.prefetch();
        }
    }

    /// Queues the AVBOIT grid passes for the background compile if the scene has volumes.
    pub fn prefetch_avboit(&self, scene: &Scene) {
        if !scene.volumes.is_empty() {
            self.avboit_inject.prefetch();
        }
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        eye: Vec3,
        view_proj: &Mat4,
        enabled: bool,
        shadows: bool,
        delta_time: f32,
        internal: (u32, u32),
        capacity: (u32, u32),
    ) {
        self.frame += 1;
        self.previous_rect = self.rect;
        self.visible.clear();
        self.rect = [0; 4];
        self.flicker_count = 0;
        for gpu in &mut self.emitters {
            gpu.update = false;
            gpu.visible = false;
        }
        let emitters = if enabled { &scene.volumes[..scene.volumes.len().min(MAX_EMITTERS)] } else { &[] };
        if emitters.is_empty() {
            self.shadows.update(device, queue, &[], &[]);
            return;
        }
        let dt = delta_time.clamp(1.0 / 240.0, 1.0 / 30.0);
        self.time += dt;
        self.emitters.truncate(emitters.len());
        for (i, emitter) in emitters.iter().enumerate() {
            let simulated = matches!(emitter.source, VolumeSource::Simulated(_));
            let resolution = emitter.resolution.max(UVec3::splat(4));
            if self.emitters.get(i).is_none_or(|g| g.resolution != resolution || g.simulated != simulated) {
                let gpu = EmitterGpu::new(device, emitter);
                if i < self.emitters.len() {
                    self.emitters[i] = gpu;
                } else {
                    self.emitters.push(gpu);
                }
            }
        }

        // Visibility: frustum and distance; the nearest few are marched.
        let planes = crate::camera::frustum_planes(view_proj);
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for (i, emitter) in emitters.iter().enumerate() {
            let centre = emitter.origin() + emitter.size * 0.5;
            let radius = emitter.size.length() * 0.5;
            let inside = planes.iter().all(|p| p.truncate().dot(centre) + p.w >= -radius);
            let distance = (centre.distance(eye) - radius).max(0.0);
            let gpu = &mut self.emitters[i];
            gpu.update = distance < SIMULATION_DISTANCE || (inside && !gpu.simulated);
            if inside {
                candidates.push((i, distance));
            }
        }
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        let lights = scene.lights();
        let mut march = MarchUniform { rect: [0; 4], previous_rect: self.previous_rect, count: [0; 4], volumes: [ShadingUniform::default(); MAX_VISIBLE] };
        let mut rect: Option<Rect> = None;
        for &(i, _) in candidates.iter().take(MAX_VISIBLE) {
            let emitter = &emitters[i];
            let Some(r) = screen_rect(emitter.origin(), emitter.size, view_proj, internal) else { continue };
            rect = Some(match rect {
                Some(u) => [u[0].min(r[0]), u[1].min(r[1]), u[2].max(r[2]), u[3].max(r[3])],
                None => r,
            });
            let shading = shading_uniform(emitter, lights, i);
            march.volumes[self.visible.len()] = shading;
            queue.write_buffer(&self.emitters[i].shading, 0, bytemuck::bytes_of(&shading));
            self.emitters[i].visible = true;
            self.visible.push(i);
        }
        self.rect = rect.unwrap_or([0; 4]);
        march.rect = self.rect;
        march.count[0] = self.visible.len() as u32;
        queue.write_buffer(&self.march_uniform, 0, bytemuck::bytes_of(&march));

        // Shadow casters: the emitters whose grid is current.
        let voxels: Vec<UVec3> = self.emitters.iter().map(|g| g.cache_size).collect();
        let casters: Vec<Option<Caster>> = emitters
            .iter()
            .zip(&self.emitters)
            .map(|(emitter, gpu)| {
                (shadows && gpu.update).then(|| Caster {
                    origin: emitter.origin(),
                    size: emitter.size,
                    extinction: emitter.extinction,
                    light: emitter
                        .light
                        .and_then(|h| lights.get(h.0 as usize).filter(|_| (h.0 as usize) < crate::lights::MAX_LIGHTS).map(|l| (h.0 as u32, l.position, l.source_radius))),
                })
            })
            .collect();
        self.shadows.update(device, queue, &voxels, &casters);

        for (i, emitter) in emitters.iter().enumerate() {
            let gpu = &mut self.emitters[i];
            if gpu.update {
                let (uniform, iterations) = sim_uniform(emitter, i, dt, self.time);
                gpu.iterations = iterations;
                queue.write_buffer(&gpu.sim, 0, bytemuck::bytes_of(&uniform));
            }
        }

        // Fire lights.
        let mut flicker = FlickerUniform { count: [0; 4], lights: [FlickerLight::default(); MAX_EMITTERS] };
        for (i, emitter) in emitters.iter().enumerate() {
            let Some(handle) = emitter.light else { continue };
            let index = handle.0 as usize;
            let (Some(light), true) = (lights.get(index), index < crate::lights::MAX_LIGHTS) else { continue };
            if !self.emitters[i].simulated {
                continue;
            }
            let slot = &mut flicker.lights[self.flicker_count as usize];
            slot.color = (light.color * light.intensity).extend(light.source_radius);
            slot.index = [index as u32, i as u32, 0, 0];
            self.flicker_count += 1;
        }
        flicker.count[0] = self.flicker_count;
        queue.write_buffer(&self.flicker_uniform, 0, bytemuck::bytes_of(&flicker));

        if self.visible() && self.targets.as_ref().is_none_or(|t| t.size != capacity) {
            self.targets = Some(ScreenTargets::new(device, capacity));
        }
    }

    /// Sets the fire lights' colours from the simulated emission. Record before the light
    /// culling (after the lights are uploaded).
    pub fn record_fire_lights(&self, rec: &mut Recorder, lights: &crate::lights::LightCulling) {
        if self.flicker_count == 0 || lights.count == 0 {
            return;
        }
        let group = bind_group(
            rec.device,
            "volume fire lights",
            &self.flicker_layout,
            &[
                (12, self.emission.as_entire_binding()),
                (20, self.flicker_uniform.as_entire_binding()),
                (21, lights.lights.as_entire_binding()),
                (22, self.flicker_state.as_entire_binding()),
            ],
        );
        let mut pass = rec.compute_pass("volume fire lights");
        pass.set_pipeline(&self.flicker);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Simulates the emitters and lights the visible ones.
    pub fn record_simulation(&self, rec: &mut Recorder, inputs: &LightingInputs) {
        if !self.emitters.iter().any(|e| e.update || e.visible) {
            return;
        }
        rec.encoder.clear_buffer(&self.emission, 0, None);
        let device = rec.device;
        let mut groups = Vec::new();
        for gpu in self.emitters.iter().filter(|e| e.update) {
            groups.push(self.simulation_groups(device, gpu));
        }
        {
            let mut pass = rec.compute_pass("volume simulation");
            for (gpu, groups) in self.emitters.iter().filter(|e| e.update).zip(&groups) {
                let r = gpu.resolution;
                let (x, y, z) = (workgroups(r.x, 4), workgroups(r.y, 4), workgroups(r.z, 4));
                let mut dispatch = |pipeline: &wgpu::ComputePipeline, group: &wgpu::BindGroup, size: (u32, u32, u32)| {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(size.0, size.1, size.2);
                };
                match groups {
                    SimulationGroups::Procedural(group) => dispatch(&self.procedural, group, (x, y, z)),
                    SimulationGroups::Fluid { curl, forces, divergence, pressure, project, advect, maccormack } => {
                        dispatch(&self.curl, curl, (x, y, z));
                        dispatch(&self.forces, forces, (x, y, z));
                        dispatch(&self.divergence, divergence, (x, y, z));
                        for _ in 0..gpu.iterations {
                            for parity in &self.pressure {
                                dispatch(parity, pressure, (workgroups(r.x.div_ceil(2), 4), y, z));
                            }
                        }
                        dispatch(&self.project, project, (x, y, z));
                        dispatch(&self.advect, advect, (x, y, z));
                        dispatch(&self.maccormack, maccormack, (x, y, z));
                    }
                }
            }
        }
        let states: Vec<&wgpu::TextureView> = self.emitters.iter().map(|g| &g.state[0]).collect();
        self.shadows.record(rec, inputs.frame, &self.sampler, &states);
        if !self.visible() {
            return;
        }
        let mut entries = crate::media::lighting_resources(inputs);
        let base = entries.len();
        let mut groups = Vec::new();
        for &i in &self.visible {
            let gpu = &self.emitters[i];
            entries.truncate(base);
            entries.extend([(70, gpu.shading.as_entire_binding()), (71, TextureView(&gpu.state[0])), (72, TextureView(&gpu.cache))]);
            groups.push(bind_group(device, "volume lighting cache", &self.cache_layout, &entries));
        }
        let mut pass = rec.compute_pass("volume lighting cache");
        pass.set_pipeline(&self.light_cache);
        for (&i, group) in self.visible.iter().zip(&groups) {
            let s = self.emitters[i].cache_size;
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(workgroups(s.x, 4), workgroups(s.y, 4), workgroups(s.z, 4));
        }
    }

    fn simulation_groups(&self, device: &wgpu::Device, gpu: &EmitterGpu) -> SimulationGroups {
        let sim = (0, gpu.sim.as_entire_binding());
        fn t(b: u32, view: &wgpu::TextureView) -> (u32, wgpu::BindingResource<'_>) {
            (b, TextureView(view))
        }
        let sampler = (6, Sampler(&self.sampler));
        if !gpu.simulated {
            return SimulationGroups::Procedural(bind_group(device, "volume procedural", &self.procedural_layout, &[sim, t(8, &gpu.state[0])]));
        }
        let [v0, v1, v2] = &gpu.velocity;
        let [s0, s1, s2] = &gpu.state;
        SimulationGroups::Fluid {
            curl: bind_group(device, "volume curl", &self.curl_layout, &[sim.clone(), t(1, v0), t(9, &gpu.curl)]),
            forces: bind_group(device, "volume forces", &self.forces_layout, &[sim.clone(), t(1, v0), t(3, s0), t(5, &gpu.curl), t(7, v1), t(8, s1)]),
            divergence: bind_group(device, "volume divergence", &self.divergence_layout, &[sim.clone(), t(1, v1), (11, gpu.divergence.as_entire_binding())]),
            pressure: bind_group(
                device,
                "volume pressure",
                &self.pressure_layout,
                &[sim.clone(), (10, gpu.pressure.as_entire_binding()), (14, gpu.divergence.as_entire_binding())],
            ),
            project: bind_group(device, "volume project", &self.project_layout, &[sim.clone(), t(1, v1), t(7, v2), (13, gpu.pressure.as_entire_binding())]),
            advect: bind_group(device, "volume advect", &self.advect_layout, &[sim.clone(), t(1, v2), t(3, s1), sampler.clone(), t(7, v0), t(8, s2)]),
            maccormack: bind_group(
                device,
                "volume maccormack",
                &self.maccormack_layout,
                &[sim, t(1, v2), t(3, s1), t(4, s2), sampler, t(8, s0), (12, self.emission.as_entire_binding())],
            ),
        }
    }

    /// Marches the visible volumes into this frame's front and back parts and resolves them
    /// temporally. `translucent_depth` is the nearest translucent surface's depth (0: none).
    pub fn record(&self, rec: &mut Recorder, inputs: &LightingInputs, translucent_depth: &wgpu::TextureView, avboit: &Avboit) {
        let Some(targets) = self.targets.as_ref().filter(|_| self.visible()) else { return };
        let device = rec.device;
        let march = self.march_group(device, inputs, translucent_depth, avboit, targets);
        let (current, previous) = (&targets.history[self.frame & 1], &targets.history[1 - (self.frame & 1)]);
        let resolve = bind_group(
            device,
            "volume resolve",
            &self.resolve_layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (11, Sampler(&inputs.samplers.linear)),
                (80, self.march_uniform.as_entire_binding()),
                (89, TextureView(&inputs.targets.depth)),
                (141, TextureView(&current[0])),
                (142, TextureView(&current[1])),
                (143, TextureView(&targets.raw[0])),
                (144, TextureView(&targets.raw[1])),
                (145, TextureView(&previous[0])),
                (146, TextureView(&previous[1])),
            ],
        );
        let (w, h) = (self.rect[2] - self.rect[0], self.rect[3] - self.rect[1]);
        let mut pass = rec.compute_pass("volume march");
        pass.set_pipeline(&self.march);
        pass.set_bind_group(0, &march, &[]);
        pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, &resolve, &[]);
        pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
    }

    /// The visible volumes' optical depth in the AVBOIT grid's columns over their rectangle
    /// (after the surfaces and particles bounded the columns: volumes don't).
    pub fn record_avboit(&self, rec: &mut Recorder, inputs: &LightingInputs, translucent_depth: &wgpu::TextureView, avboit: &Avboit) {
        let Some(targets) = self.targets.as_ref().filter(|_| self.visible() && avboit.enabled()) else { return };
        let march = self.march_group(rec.device, inputs, translucent_depth, avboit, targets);
        let (label, pipeline) = ("volume avboit transmittance", &self.avboit_inject);
        let tile = avboit.tile();
        let (x0, y0) = (self.rect[0] / tile, self.rect[1] / tile);
        let (x1, y1) = (self.rect[2].div_ceil(tile), self.rect[3].div_ceil(tile));
        let mut pass = rec.compute_pass(label);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &march, &[]);
        pass.dispatch_workgroups(workgroups(x1 - x0, 8), workgroups(y1 - y0, 8), 1);
    }

    fn march_group(&self, device: &wgpu::Device, inputs: &LightingInputs, translucent_depth: &wgpu::TextureView, avboit: &Avboit, targets: &ScreenTargets) -> wgpu::BindGroup {
        let mut entries = crate::media::fog_resources(inputs);
        let slot = |k: usize| self.visible.get(k).map(|&i| &self.emitters[i]);
        for k in 0..MAX_VISIBLE {
            let (state, cache) = slot(k).map_or((&self.empty, &self.empty), |g| (&g.state[0], &g.cache));
            entries.push((81 + k as u32, TextureView(state)));
            entries.push((85 + k as u32, TextureView(cache)));
        }
        entries.extend([
            (80, self.march_uniform.as_entire_binding()),
            (89, TextureView(&inputs.targets.depth)),
            (140, TextureView(translucent_depth)),
            (141, TextureView(&targets.raw[0])),
            (142, TextureView(&targets.raw[1])),
        ]);
        entries.extend(avboit.read_resources());
        entries.extend(avboit.write_resources().into_iter().filter(|(b, _)| *b != 120));
        bind_group(device, "volume march", &self.march_layout, &entries)
    }
}

enum SimulationGroups {
    Procedural(wgpu::BindGroup),
    Fluid {
        curl: wgpu::BindGroup,
        forces: wgpu::BindGroup,
        divergence: wgpu::BindGroup,
        pressure: wgpu::BindGroup,
        project: wgpu::BindGroup,
        advect: wgpu::BindGroup,
        maccormack: wgpu::BindGroup,
    },
}

/// The emitter's simulation uniform and pressure iterations.
fn sim_uniform(emitter: &VolumeEmitter, index: usize, dt: f32, time: f32) -> (SimUniform, u32) {
    let r = emitter.resolution.max(UVec3::splat(4));
    let cell = emitter.cell();
    let mut u = SimUniform {
        resolution: [r.x, r.y, r.z, 0],
        cell: cell.extend(dt),
        source: Vec4::ZERO,
        source_rates: Vec4::ZERO,
        source_velocity: Vec4::new(0.0, 0.0, 0.0, time),
        forces: Vec4::new(0.0, 0.0, 0.0, 1.7),
        dissipation: Vec4::ZERO,
        combustion: Vec4::new(0.0, 0.0, 3.0, emitter.emission_kelvin.y),
        wind: Vec4::ZERO,
        procedural: Vec4::ZERO,
        drift: Vec4::ZERO,
        emission: Vec4::new(emitter.emission_kelvin.x, index as f32, 0.0, 0.0),
    };
    match emitter.source {
        VolumeSource::Simulated(s) => {
            let mean_cell = (cell.x + cell.y + cell.z) / 3.0;
            let centre = (Vec3::new(emitter.size.x * 0.5, 0.0, emitter.size.z * 0.5) + s.source_offset) / cell;
            u.source = centre.extend(s.source_radius / mean_cell);
            u.source_rates = Vec4::new(s.soot_rate, s.fuel_rate, s.temperature, s.flicker);
            u.source_velocity = s.source_velocity.extend(time);
            u.forces = Vec4::new(s.buoyancy, s.soot_weight, s.vorticity, 1.7);
            u.dissipation = Vec4::new(s.soot_dissipation, s.cooling, s.velocity_damping, s.burn_rate);
            u.combustion.x = s.heat;
            u.combustion.y = s.soot_yield;
            u.wind = s.wind.extend(s.wind_coupling);
            (u, s.pressure_iterations)
        }
        VolumeSource::Procedural(p) => {
            u.procedural = Vec4::new(p.density, p.frequency, p.height_falloff, p.coverage);
            u.drift = (p.wind * time).extend(p.edge_fade.clamp(0.01, 1.0));
            (u, 0)
        }
    }
}

/// The emitter's look, with the brightest local lights whose range reaches its box.
fn shading_uniform(emitter: &VolumeEmitter, lights: &[crate::scene::Light], index: usize) -> ShadingUniform {
    let origin = emitter.origin();
    let (lo, hi) = (origin, origin + emitter.size);
    let mut reaching: Vec<(u32, f32)> = lights
        .iter()
        .enumerate()
        .take(crate::lights::MAX_LIGHTS)
        .filter(|(_, l)| l.position.clamp(lo, hi).distance(l.position) < l.range)
        .map(|(i, l)| (i as u32, l.intensity * l.color.max_element() / (1.0 + l.position.clamp(lo, hi).distance_squared(l.position))))
        .collect();
    reaching.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut indices = [0u32; 8];
    for (slot, (i, _)) in indices.iter_mut().zip(&reaching) {
        *slot = *i;
    }
    let cell = emitter.cell();
    ShadingUniform {
        origin: origin.extend(emitter.extinction),
        size: emitter.size.extend(emitter.anisotropy),
        albedo: emitter.albedo.extend(emitter.emission),
        emission: Vec4::new(emitter.emission_kelvin.x, emitter.emission_kelvin.y, 0.75 * cell.min_element(), emitter.ambient),
        lights: indices,
        params: [reaching.len().min(MAX_CACHE_LIGHTS) as u32, emitter.resolution.x, emitter.resolution.y, emitter.resolution.z],
        shadow: [index as u32, 0, 0, 0],
    }
}

/// The box's projection in internal pixels (2 pixels of margin), the whole view when it
/// reaches behind the camera; `None` when it is off screen.
fn screen_rect(origin: Vec3, size: Vec3, view_proj: &Mat4, (width, height): (u32, u32)) -> Option<Rect> {
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    let mut behind = 0;
    for i in 0..8 {
        let corner = origin + size * Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32);
        let clip = *view_proj * corner.extend(1.0);
        if clip.w <= 1e-3 {
            behind += 1;
            continue;
        }
        let ndc = clip.truncate().truncate() / clip.w;
        lo = lo.min(ndc);
        hi = hi.max(ndc);
    }
    match behind {
        0 => {}
        8 => return None,
        _ => return Some([0, 0, width, height]),
    }
    let to_pixel = |ndc: Vec2| Vec2::new((ndc.x * 0.5 + 0.5) * width as f32, (0.5 - ndc.y * 0.5) * height as f32);
    let (a, b) = (to_pixel(Vec2::new(lo.x, hi.y)), to_pixel(Vec2::new(hi.x, lo.y)));
    let min = (a - 2.0).max(Vec2::ZERO);
    let max = (b + 2.0).min(Vec2::new(width as f32, height as f32));
    (min.x < max.x && min.y < max.y).then(|| [min.x as u32, min.y as u32, max.x.ceil() as u32, max.y.ceil() as u32])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_rects_bound_the_projected_box() {
        let camera = crate::Camera { position: Vec3::ZERO, ..Default::default() };
        let view_proj = camera.projection(16.0 / 9.0) * camera.view_matrix();
        let size = (1600, 900);
        // A box straight ahead covers the centre of the view.
        let r = screen_rect(Vec3::new(-1.0, -1.0, -11.0), Vec3::splat(2.0), &view_proj, size).unwrap();
        assert!(r[0] < 800 && r[2] > 800 && r[1] < 450 && r[3] > 450);
        assert!(r[2] - r[0] < 400);
        // Behind the camera: off screen; around it: the whole view.
        assert_eq!(screen_rect(Vec3::new(-1.0, -1.0, 9.0), Vec3::splat(2.0), &view_proj, size), None);
        assert_eq!(screen_rect(Vec3::splat(-1.0), Vec3::splat(2.0), &view_proj, size), Some([0, 0, 1600, 900]));
    }

    #[test]
    fn cache_lights_are_those_reaching_the_box() {
        let fire = VolumeEmitter::campfire(Vec3::ZERO);
        let near = crate::scene::Light::point(Vec3::new(0.0, 0.5, 0.0), Vec3::ONE, 2.0, 8.0);
        let far = crate::scene::Light::point(Vec3::new(30.0, 0.5, 0.0), Vec3::ONE, 2.0, 8.0);
        let shading = shading_uniform(&fire, &[far, near], 0);
        assert_eq!(shading.params[0], 1);
        assert_eq!(shading.lights[0], 1);
    }

    #[test]
    fn mist_resolution_follows_its_size() {
        let mist = VolumeEmitter::mist(Vec3::ZERO, Vec3::new(35.0, 3.5, 7.0));
        assert_eq!(mist.resolution, UVec3::new(100, 10, 20));
        let cell = mist.cell();
        assert!((cell - Vec3::splat(0.35)).abs().max_element() < 0.01);
    }
}
