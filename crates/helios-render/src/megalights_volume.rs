//! MegaLights for volumetric fog and translucent surfaces (UE 5.8's MegaLights volume).
//!
//! With MegaLights on, opaque surfaces see every local light shadowed, but the fog injected the
//! clustered lights shadowed only by the ≤ 32 shadow-mapped ones and translucent surfaces looped
//! over them all unshadowed (one distance-field trace for the strongest): with hundreds of
//! lights, light leaked through walls into the fog and onto glass. As in UE, one froxel volume
//! of MegaLights lighting serves both (`megalights_volume.wgsl`): per froxel of the fog's grid
//! (16-pixel columns, 64 exponential slices), one light is sampled in proportion to its
//! unshadowed radiance at a point jittered through the froxel and traced with MegaLights'
//! shadow rays (`megalights_rays.wgsl`: shadow-map ray marching or distance fields), and the
//! estimates are accumulated over frames with reprojection. The volume holds the in-scattered
//! light towards the camera (fog.wgsl adds it instead of its light loop) and the incident light
//! as SH bands 0 and 1 with one colour (translucency.wgsl shades a directional and an ambient
//! part with it instead of its loop). The textures (6 × 8 bytes per froxel: 17.5 MB at
//! 1600×900) exist only while the mode is on.

use glam::Vec4;
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat, TextureUsages};

use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, uniform_buffer, workgroups, Lazy};

/// The fog's grid (`fog.rs`): screen pixels per froxel column and slices.
const TILE: u32 = 16;
const SLICES: u32 = 64;
const FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// Weight of this frame's estimate against the reprojected history.
const HISTORY_BLEND: f32 = 0.1;
/// Depth range of the slices while the scene has no fog (with fog: the fog's, so both volumes
/// share their froxels).
const NEAR: f32 = 0.5;
const FAR: f32 = 80.0;

/// What the volume reads of the frame.
pub struct MegaLightsVolumeInputs<'a> {
    pub frame: &'a wgpu::Buffer,
    pub fog: &'a crate::fog::VolumetricFog,
    pub lights: &'a crate::lights::LightCulling,
    pub vsm: &'a crate::vsm::VirtualShadowMap,
    pub lumen: &'a crate::lumen::Lumen,
    pub samplers: &'a crate::passes::Samplers,
}

struct Volumes {
    size: (u32, u32),
    /// Per frame parity: fog light, sh0, sh1 (the other parity is the history).
    sets: [[wgpu::TextureView; 3]; 2],
}

pub struct MegaLightsVolume {
    layout: wgpu::BindGroupLayout,
    inject: Lazy<wgpu::ComputePipeline>,
    uniform: wgpu::Buffer,
    volumes: Option<Volumes>,
    /// A 1×1×1 volume bound while the mode is off (consumers test the depth of what they get).
    empty: wgpu::TextureView,
}

impl MegaLightsVolume {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "megalights_volume.wgsl");
        let tex = bind::texture_3d(true);
        let out = bind::storage_texture_3d(FORMAT);
        let layout = bind_group_layout(
            device,
            "megalights volume",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::uniform()),
                (3, bind::storage(true)),
                (4, bind::storage(true)),
                (5, bind::uniform()),
                (6, tex),
                (7, bind::sampler()),
                (8, bind::storage(true)),
                (9, tex),
                (10, bind::uniform()),
                (11, bind::storage(true)),
                (12, bind::texture_depth()),
                (13, bind::uniform()),
                (14, tex),
                (15, tex),
                (16, tex),
                (17, out),
                (18, out),
                (19, out),
            ],
        );
        let empty = volume(device, "megalights volume off", (1, 1, 1), TextureUsages::TEXTURE_BINDING);
        Self {
            inject: lazy_compute_pipeline(device, "megalights volume", &module, "inject", &layout, &[]),
            layout,
            uniform: uniform_buffer(device, "megalights volume", 16),
            volumes: None,
            empty,
        }
    }

    /// Allocates the volumes for `capacity` while the mode runs (`enabled`), frees them while it
    /// does not. `fog_range`: the fog's (near, far) while the scene has fog, so the froxels match.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, capacity: (u32, u32), enabled: bool, fog_range: Option<(f32, f32)>) {
        let size = (capacity.0.div_ceil(TILE).max(1), capacity.1.div_ceil(TILE).max(1));
        if !enabled {
            self.volumes = None;
        } else if self.volumes.as_ref().is_none_or(|v| v.size != size) {
            let usage = TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
            let extent = (size.0, size.1, SLICES);
            self.volumes = Some(Volumes {
                size,
                sets: [0, 1].map(|i| {
                    [
                        volume(device, &format!("megalights volume fog {i}"), extent, usage),
                        volume(device, &format!("megalights volume sh0 {i}"), extent, usage),
                        volume(device, &format!("megalights volume sh1 {i}"), extent, usage),
                    ]
                }),
            });
        }
        let (near, far) = fog_range.unwrap_or((NEAR, FAR));
        let on = self.volumes.is_some() as u32 as f32;
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&Vec4::new(near, far, HISTORY_BLEND, on)));
    }

    /// The volume's mapping (`MlVolume` in megalights_volume_common.wgsl; w = 0 while off).
    pub fn uniform(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    /// In-scattered light for the fog of the frame of parity `current` (1×1×1 while off).
    pub fn fog_light(&self, current: usize) -> &wgpu::TextureView {
        self.volumes.as_ref().map_or(&self.empty, |v| &v.sets[current][0])
    }

    /// The incident light (sh0, sh1) for translucent surfaces (1×1×1 while off).
    pub fn incident(&self, current: usize) -> [&wgpu::TextureView; 2] {
        self.volumes.as_ref().map_or([&self.empty, &self.empty], |v| [&v.sets[current][1], &v.sets[current][2]])
    }

    pub fn record(&self, rec: &mut Recorder, inputs: &MegaLightsVolumeInputs, current: usize) {
        let Some(v) = &self.volumes else { return };
        let (global_sdf, _) = inputs.lumen.debug_views();
        let (mesh_sdf_data, mesh_sdf_atlas) = inputs.lumen.near_field();
        let (out, history) = (&v.sets[current], &v.sets[1 - current]);
        let group = bind_group(
            rec.device,
            "megalights volume",
            &self.layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, self.uniform.as_entire_binding()),
                (2, inputs.fog.uniform.as_entire_binding()),
                (3, inputs.lights.lights.as_entire_binding()),
                (4, inputs.lights.grid.as_entire_binding()),
                (5, inputs.lumen.uniform().as_entire_binding()),
                (6, TextureView(global_sdf)),
                (7, Sampler(&inputs.samplers.linear)),
                (8, mesh_sdf_data.as_entire_binding()),
                (9, TextureView(mesh_sdf_atlas)),
                (10, inputs.vsm.uniform.as_entire_binding()),
                (11, inputs.vsm.cache.entries.as_entire_binding()),
                (12, TextureView(&inputs.vsm.atlas)),
                (13, inputs.vsm.local.uniform.as_entire_binding()),
                (14, TextureView(&history[0])),
                (15, TextureView(&history[1])),
                (16, TextureView(&history[2])),
                (17, TextureView(&out[0])),
                (18, TextureView(&out[1])),
                (19, TextureView(&out[2])),
            ],
        );
        let mut pass = rec.compute_pass("megalights volume");
        pass.set_pipeline(&self.inject);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(v.size.0, 4), workgroups(v.size.1, 4), workgroups(SLICES, 4));
    }
}

fn volume(device: &wgpu::Device, label: &str, size: (u32, u32, u32), usage: TextureUsages) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: size.2 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: FORMAT,
            usage,
            view_formats: &[],
        })
        .create_view(&Default::default())
}
