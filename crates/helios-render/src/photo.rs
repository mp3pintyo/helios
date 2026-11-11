//! Photo mode: a still accumulated from many frames, as UE's Movie Render Graph accumulates
//! temporal and spatial samples (and its Accumulation DOF lens samples) instead of relying on
//! temporal anti-aliasing and the post-process depth of field.
//!
//! While the camera stays where the photo started, every frame renders with
//!
//! * a sub-pixel jitter from the Halton (2, 3) sequence (no cycle: every sample is new), and
//! * with depth of field on, a lens sample: the camera moves to a point of the aperture disc
//!   (Halton (5, 7) mapped concentrically onto it; diameter focal length / f-number, as the
//!   real-time depth of field's thin lens) and the projection is sheared so the focus plane
//!   stays put — what is in focus lands on the same pixel through every part of the lens, what
//!   is not is spread over its circle of confusion. The focus distance is the setting's, or the
//!   autofocus distance read back from the GPU when the photo starts.
//!
//! TSR's photo permutation reconstructs each frame on its own (its Gaussian filter over this
//! frame's samples, with their weight); `photo.wgsl` sums them in an `rgba32float` target and
//! the mean goes on through exposure, bloom and tone mapping. The post-process depth of field
//! and motion blur are off, the scene's clock stands still (no wind, no drift, exposure held),
//! and Lumen, MegaLights and the fog keep converging over the frames, which average their
//! remaining noise too. The image stays on screen once complete until the camera moves or the
//! photo is stopped.

use glam::{Vec2, Vec4};
use wgpu::BindingResource::TextureView;
use wgpu::{ShaderStages, TextureFormat, TextureUsages};

use crate::camera::{halton, Camera};
use crate::passes::post_fx::DepthOfField;
use crate::pipelines::Lazy;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, texture, uniform_buffer, workgroups, TextureSpec};

const SUM: TextureFormat = TextureFormat::Rgba32Float;
const HDR: TextureFormat = TextureFormat::Rgba16Float;

/// How far a photo has come ([`crate::Renderer::photo_progress`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhotoProgress {
    /// Frames summed so far.
    pub accumulated: u32,
    /// Frames asked for.
    pub samples: u32,
    /// Waiting for the autofocus distance (the first frames of a photo with autofocus).
    pub focusing: bool,
}

impl PhotoProgress {
    pub fn done(&self) -> bool {
        self.accumulated >= self.samples
    }
}

/// This frame's photo sample.
#[derive(Clone, Copy, Debug)]
pub struct PhotoSample {
    /// The camera moved to its lens sample.
    pub camera: Camera,
    /// Clip-space shift that keeps the focus plane in place (part of the projection).
    pub lens_shift: Vec2,
    /// Sub-pixel jitter in internal pixels, each axis in [-0.5, 0.5).
    pub jitter: Vec2,
    /// The first sample of the photo (starts a new sum).
    pub first: bool,
}

#[derive(Clone, Copy, Debug)]
enum Focus {
    /// No depth of field: a pinhole, only sub-pixel samples.
    Pinhole,
    /// Waiting for the autofocus read-back.
    Waiting,
    /// Aperture radius and focus distance in metres.
    Lens { radius: f32, distance: f32 },
}

struct Photo {
    samples: u32,
    accumulated: u32,
    /// The camera the photo is taken from (`None` until the first frame).
    camera: Option<Camera>,
    focus: Focus,
}

struct Targets {
    size: (u32, u32),
    sums: [wgpu::TextureView; 2],
    mean: wgpu::TextureView,
}

pub struct PhotoMode {
    photo: Option<Photo>,
    pipeline: Lazy<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    targets: Option<Targets>,
}

/// A point of the unit disc from a point of the unit square (Shirley–Chiu concentric map).
fn concentric_disc(u: Vec2) -> Vec2 {
    let p = u * 2.0 - 1.0;
    if p == Vec2::ZERO {
        return Vec2::ZERO;
    }
    let (r, theta) = if p.x.abs() > p.y.abs() {
        (p.x, std::f32::consts::FRAC_PI_4 * (p.y / p.x))
    } else {
        (p.y, std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_4 * (p.x / p.y))
    };
    Vec2::new(theta.cos(), theta.sin()) * r
}

/// `camera` moved to the point `lens` (metres, camera right / up) of its aperture, and the
/// clip-space shift that keeps the plane at view depth `distance` where it was.
fn lens_camera(camera: &Camera, lens: Vec2, distance: f32, aspect: f32) -> (Camera, Vec2) {
    let right = camera.right();
    let up = right.cross(camera.forward());
    let mut moved = *camera;
    moved.position += right * lens.x + up * lens.y;
    // In view space a point on the focus plane moves by -lens; its clip x, y by the projection's
    // scale times that over its depth: shift them back.
    let scale_y = 1.0 / (camera.fov_y * 0.5).tan();
    let shift = Vec2::new(scale_y / aspect, scale_y) * lens / distance.max(camera.near * 2.0);
    (moved, shift)
}

/// The aperture radius in metres of a depth of field setting at a vertical field of view
/// (the real-time depth of field's thin lens: focal length from the sensor height).
fn aperture_radius(dof: &DepthOfField, fov_y: f32) -> f32 {
    let focal_length = dof.sensor_height * 0.5 / (fov_y * 0.5).tan();
    focal_length / dof.f_stop.max(0.5) * 0.5
}

impl PhotoMode {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "photo.wgsl");
        let layout = bind_group_layout(
            device,
            "photo",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::texture(false)),
                (2, bind::texture(false)),
                (3, bind::storage_texture(SUM)),
                (4, bind::storage_texture(HDR)),
            ],
        );
        Self {
            pipeline: lazy_compute_pipeline(device, "photo", &module, "accumulate", &layout, &[]),
            layout,
            uniform: uniform_buffer(device, "photo", 16),
            photo: None,
            targets: None,
        }
    }

    /// Starts a photo of `samples` frames from the camera of the next frame; `dof` is the depth
    /// of field setting (`None`: sub-pixel samples only).
    pub fn start(&mut self, samples: u32, dof: Option<&DepthOfField>) {
        let focus = match dof {
            None => Focus::Pinhole,
            // The radius needs the field of view: filled in on the first frame.
            Some(d) => match d.focus_distance {
                Some(distance) => Focus::Lens { radius: -1.0, distance },
                None => Focus::Waiting,
            },
        };
        self.photo = Some(Photo { samples: samples.max(1), accumulated: 0, camera: None, focus });
    }

    pub fn stop(&mut self) {
        self.photo = None;
    }

    pub fn progress(&self) -> Option<PhotoProgress> {
        self.photo.as_ref().map(|p| PhotoProgress { accumulated: p.accumulated, samples: p.samples, focusing: matches!(p.focus, Focus::Waiting) })
    }

    /// Whether the finished photo is on screen.
    pub fn complete(&self) -> bool {
        self.photo.as_ref().is_some_and(|p| p.accumulated >= p.samples) && self.targets.is_some()
    }

    /// The frame's sample, or `None` for an ordinary frame (no photo, or waiting for focus).
    /// A camera that moved ends the photo. `focus` is the autofocus read-back, if one arrived
    /// (the renderer asks for one while [`PhotoProgress::focusing`]).
    pub fn begin_frame(&mut self, camera: &Camera, dof: Option<&DepthOfField>, focus: Option<f32>, output_aspect: f32) -> Option<PhotoSample> {
        let photo = self.photo.as_mut()?;
        let start = *photo.camera.get_or_insert(*camera);
        let moved = start.position.distance(camera.position) > 1e-5
            || (start.yaw - camera.yaw).abs() > 1e-6
            || (start.pitch - camera.pitch).abs() > 1e-6
            || (start.fov_y - camera.fov_y).abs() > 1e-6;
        if moved {
            self.photo = None;
            return None;
        }
        match (photo.focus, dof) {
            (Focus::Waiting, Some(d)) => match focus {
                Some(distance) => photo.focus = Focus::Lens { radius: aperture_radius(d, camera.fov_y), distance },
                None => return None,
            },
            (Focus::Lens { radius, distance }, Some(d)) if radius < 0.0 => {
                photo.focus = Focus::Lens { radius: aperture_radius(d, camera.fov_y), distance }
            }
            // Depth of field switched off meanwhile: a pinhole.
            (_, None) => photo.focus = Focus::Pinhole,
            _ => {}
        }
        if photo.accumulated >= photo.samples {
            return None;
        }
        let i = photo.accumulated + 1;
        let jitter = Vec2::new(halton(i, 2) - 0.5, halton(i, 3) - 0.5);
        let mut sample = PhotoSample { camera: *camera, lens_shift: Vec2::ZERO, jitter, first: photo.accumulated == 0 };
        if let Focus::Lens { radius, distance } = photo.focus {
            let lens = concentric_disc(Vec2::new(halton(i, 5), halton(i, 7))) * radius;
            (sample.camera, sample.lens_shift) = lens_camera(camera, lens, distance, output_aspect);
        }
        Some(sample)
    }

    /// Adds this frame's TSR result (`color`: reconstruction and sample weight) to the sum and
    /// returns the mean so far.
    pub fn accumulate<'a>(&'a mut self, device: &wgpu::Device, queue: &wgpu::Queue, rec: &mut Recorder, color: &wgpu::TextureView, size: (u32, u32), first: bool) -> &'a wgpu::TextureView {
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            let view = |label: &str, format| {
                texture(device, TextureSpec { label, width: size.0, height: size.1, layers: 1, mips: 1, format, usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING })
                    .create_view(&Default::default())
            };
            self.targets = Some(Targets { size, sums: [view("photo sum 0", SUM), view("photo sum 1", SUM)], mean: view("photo", HDR) });
        }
        let photo = self.photo.as_mut().expect("accumulate without a photo");
        let t = self.targets.as_ref().expect("photo targets");
        let index = (photo.accumulated & 1) as usize;
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&Vec4::new(if first { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0)));
        let group = bind_group(
            device,
            "photo",
            &self.layout,
            &[
                (0, self.uniform.as_entire_binding()),
                (1, TextureView(color)),
                (2, TextureView(&t.sums[1 - index])),
                (3, TextureView(&t.sums[index])),
                (4, TextureView(&t.mean)),
            ],
        );
        let mut pass = rec.compute_pass("photo");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(size.0, 8), workgroups(size.1, 8), 1);
        photo.accumulated += 1;
        &t.mean
    }

    /// The finished photo (while [`PhotoMode::complete`]).
    pub fn mean(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().filter(|_| self.complete()).map(|t| &t.mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    #[test]
    fn concentric_disc_stays_inside_the_disc() {
        for i in 1..200 {
            let p = concentric_disc(Vec2::new(halton(i, 5), halton(i, 7)));
            assert!(p.length() <= 1.0 + 1e-5);
        }
        assert!((concentric_disc(Vec2::new(1.0, 0.5)) - Vec2::new(1.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn lens_samples_keep_the_focus_plane_and_blur_the_rest() {
        let camera = Camera { position: Vec3::new(1.0, 2.0, 3.0), yaw: 0.3, pitch: -0.1, ..Default::default() };
        let aspect = 16.0 / 9.0;
        let distance = 5.0;
        let project = |c: &Camera, shift: Vec2, p: Vec3| {
            let view_proj = Mat4::from_translation(shift.extend(0.0)) * c.projection(aspect) * c.view_matrix();
            let clip = view_proj * p.extend(1.0);
            Vec2::new(clip.x, clip.y) / clip.w
        };
        let at = |depth: f32| camera.position + camera.forward() * depth + camera.right() * 0.4;
        for lens in [Vec2::new(0.02, 0.0), Vec2::new(-0.01, 0.015)] {
            let (moved, shift) = lens_camera(&camera, lens, distance, aspect);
            assert!((moved.position - camera.position).length() > 0.01);
            // In focus: the same place through every part of the lens.
            let d = project(&moved, shift, at(distance)) - project(&camera, Vec2::ZERO, at(distance));
            assert!(d.length() < 1e-4, "focus plane moved by {d}");
            // Out of focus: spread.
            let d = project(&moved, shift, at(1.5)) - project(&camera, Vec2::ZERO, at(1.5));
            assert!(d.length() > 1e-3);
        }
    }
}
