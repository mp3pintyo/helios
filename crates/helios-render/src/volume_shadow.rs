//! Shadows of heterogeneous volumes (round 8; UE's heterogeneous volumes cast shadows through
//! an adaptive volumetric shadow map). Every emitter that is simulated this frame gets a region of
//! one shared 3D **transmittance atlas** (rgba8unorm, at its lighting cache's resolution, half the
//! simulation grid): per voxel the transmittance of the medium from the voxel to the box's
//! boundary towards the sun (r) and towards the fire light the emitter drives (g), marched through
//! the density grid (`shaders/volume_shadow.wgsl`). A receiver anywhere — inside, below or beside
//! the box — finds the point where its ray towards the sun enters the box (for the driven light:
//! where the segment from the light leaves the box towards it, or the receiver itself inside it)
//! and looks the transmittance up there (`shaders/volume_shadow_sample.wgsl`): one trilinear tap
//! per box its ray crosses. The deferred lighting (sun and the driven light), the volumetric fog,
//! the particles and the volumes' lighting caches (the other emitters' shadows, the driven light's
//! self-shadowing) sample it. The atlas is built after the simulation, so the opaque lighting and
//! the fog, which run earlier in the frame, see last frame's — a frame late, as the fire's light.
//!
//! The driven light sits inside its fire: the hot, emitting gas is its source, not an occluder,
//! so the light's march skips the medium in proportion to its temperature, and within the light's
//! source radius.

use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec3, Vec4};
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat};

use crate::pipelines::Lazy;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, uniform_buffer, workgroups};

/// Emitters with a region in the atlas (as many as are simulated).
pub(crate) const MAX_SHADOW_VOLUMES: usize = 8;
const FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
/// `VolumeShadows::count.y` when no shadow box drives a light (receivers skip the light test).
const NO_LIGHT: u32 = u32::MAX;

/// `VolumeShadowBox` in volume_shadow_sample.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct ShadowBox {
    /// xyz box minimum, w: the driven light's index (f32 bits; `NO_LIGHT`: none).
    origin: Vec4,
    /// xyz extent, w: 1 while the box casts shadows this frame.
    size: Vec4,
    /// xyz: the region's first voxel in the atlas.
    offset: Vec4,
    /// xyz: the region's voxels.
    voxels: Vec4,
}

/// `VolumeShadows` in volume_shadow_sample.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowUniform {
    /// x: boxes, y: 1 if any casting box drives a light.
    count: [u32; 4],
    /// xyz: the atlas' voxels.
    atlas: Vec4,
    boxes: [ShadowBox; MAX_SHADOW_VOLUMES],
}

/// `ShadowBuild` in volume_shadow.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct BuildUniform {
    /// xyz box minimum, w: extinction per metre at density 1.
    origin: Vec4,
    size: Vec4,
    /// xyz: the region's first voxel, w: 1 if the emitter drives a light.
    offset: [u32; 4],
    /// xyz: the region's voxels.
    voxels: [u32; 4],
    /// xyz: the driven light's position, w: its source radius.
    light: Vec4,
}

/// One emitter as a shadow caster this frame.
pub(crate) struct Caster {
    pub origin: Vec3,
    pub size: Vec3,
    pub extinction: f32,
    /// The driven light: its index, position and source radius.
    pub light: Option<(u32, Vec3, f32)>,
}

pub struct VolumeShadows {
    layout: wgpu::BindGroupLayout,
    build: Lazy<wgpu::ComputePipeline>,
    pub uniform: wgpu::Buffer,
    builds: Vec<wgpu::Buffer>,
    atlas: wgpu::TextureView,
    /// Per emitter its region (first voxel, voxels).
    regions: Vec<(UVec3, UVec3)>,
    /// Per emitter: its region holds a build (it was built in an earlier frame).
    ready: Vec<bool>,
    /// Emitters built this frame.
    building: Vec<bool>,
}

impl VolumeShadows {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = bind_group_layout(
            device,
            "volume shadows",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::texture_3d(true)),
                (3, bind::sampler()),
                (4, bind::storage_texture_3d(FORMAT)),
            ],
        );
        let module = create_module(device, "volume_shadow.wgsl");
        let uniform = uniform_buffer(device, "volume shadows", std::mem::size_of::<ShadowUniform>() as u64);
        let builds = (0..MAX_SHADOW_VOLUMES).map(|_| uniform_buffer(device, "volume shadow build", std::mem::size_of::<BuildUniform>() as u64)).collect();
        Self {
            build: lazy_compute_pipeline(device, "volume shadows", &module, "build", &layout, &[]),
            layout,
            uniform,
            builds,
            atlas: atlas(device, UVec3::ONE),
            regions: Vec::new(),
            ready: Vec::new(),
            building: Vec::new(),
        }
    }

    pub fn prefetch(&self) {
        self.build.prefetch();
    }

    /// The transmittance atlas (sampled by the receivers).
    pub fn atlas(&self) -> &wgpu::TextureView {
        &self.atlas
    }

    /// Lays the emitters' regions out side by side (`voxels`: each emitter's lighting cache size)
    /// and uploads which of them cast shadows this frame (`casters`, by emitter; `None`: not
    /// simulated this frame or shadows off).
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, voxels: &[UVec3], casters: &[Option<Caster>]) {
        let voxels = &voxels[..voxels.len().min(MAX_SHADOW_VOLUMES)];
        if self.regions.len() != voxels.len() || self.regions.iter().zip(voxels).any(|(r, v)| r.1 != *v) {
            let mut x = 0;
            self.regions = voxels.iter().map(|&v| {
                let region = (UVec3::new(x, 0, 0), v);
                x += v.x;
                (region.0, region.1)
            }).collect();
            let size = voxels.iter().fold(UVec3::ONE, |a, v| UVec3::new(x.max(1), a.y.max(v.y), a.z.max(v.z)));
            self.atlas = atlas(device, size);
            self.ready = vec![false; voxels.len()];
        }
        let size = self.regions.iter().fold(UVec3::ONE, |a, (o, v)| a.max(*o + *v));
        let mut uniform = ShadowUniform { count: [self.regions.len() as u32, 0, 0, 0], atlas: size.as_vec3().extend(0.0), boxes: [ShadowBox::default(); MAX_SHADOW_VOLUMES] };
        self.building = vec![false; self.regions.len()];
        for (i, &(offset, voxels)) in self.regions.iter().enumerate() {
            let Some(caster) = casters.get(i).and_then(|c| c.as_ref()) else { continue };
            let light = caster.light.map_or(NO_LIGHT, |l| l.0);
            uniform.boxes[i] = ShadowBox {
                origin: caster.origin.extend(f32::from_bits(light)),
                size: caster.size.extend(if self.ready[i] { 1.0 } else { 0.0 }),
                offset: offset.as_vec3().extend(0.0),
                voxels: voxels.as_vec3().extend(0.0),
            };
            if self.ready[i] && caster.light.is_some() {
                uniform.count[1] = 1;
            }
            let (position, radius) = caster.light.map_or((Vec3::ZERO, 0.0), |l| (l.1, l.2));
            let build = BuildUniform {
                origin: caster.origin.extend(caster.extinction),
                size: caster.size.extend(0.0),
                offset: [offset.x, offset.y, offset.z, caster.light.is_some() as u32],
                voxels: [voxels.x, voxels.y, voxels.z, 0],
                light: position.extend(radius),
            };
            queue.write_buffer(&self.builds[i], 0, bytemuck::bytes_of(&build));
            self.building[i] = true;
            // Receivers sample it from the next frame on (this frame's lighting runs before the
            // build).
            self.ready[i] = true;
        }
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Builds the regions of this frame's casters (`states[i]`: emitter i's density grid).
    pub fn record(&self, rec: &mut Recorder, frame: &wgpu::Buffer, sampler: &wgpu::Sampler, states: &[&wgpu::TextureView]) {
        if !self.building.iter().any(|&b| b) {
            return;
        }
        let groups: Vec<(usize, wgpu::BindGroup)> = self
            .building
            .iter()
            .enumerate()
            .filter(|(i, &b)| b && *i < states.len())
            .map(|(i, _)| {
                let group = bind_group(
                    rec.device,
                    "volume shadows",
                    &self.layout,
                    &[
                        (0, frame.as_entire_binding()),
                        (1, self.builds[i].as_entire_binding()),
                        (2, TextureView(states[i])),
                        (3, Sampler(sampler)),
                        (4, TextureView(&self.atlas)),
                    ],
                );
                (i, group)
            })
            .collect();
        let mut pass = rec.compute_pass("volume shadows");
        pass.set_pipeline(&self.build);
        for (i, group) in &groups {
            let v = self.regions[*i].1;
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(workgroups(v.x, 4), workgroups(v.y, 4), workgroups(v.z, 4));
        }
    }
}

fn atlas(device: &wgpu::Device, size: UVec3) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("volume shadow atlas"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_match_the_shaders() {
        assert_eq!(std::mem::size_of::<ShadowBox>(), 64);
        assert_eq!(std::mem::size_of::<ShadowUniform>(), 32 + 64 * MAX_SHADOW_VOLUMES);
        assert_eq!(std::mem::size_of::<BuildUniform>(), 80);
        let sample = crate::shader::source("volume_shadow_sample.wgsl");
        assert!(sample.contains(&format!("array<VolumeShadowBox, {MAX_SHADOW_VOLUMES}>")));
    }
}
