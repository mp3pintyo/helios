//! Camera effects after TSR, at output resolution: physically based depth of field and motion
//! blur (see `shaders/post_fx.wgsl`).

use glam::Vec4;
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat, TextureUsages};

use super::Samplers;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer, texture, uniform_buffer, workgroups, TextureSpec};

const HDR: TextureFormat = TextureFormat::Rgba16Float;
const TILE: u32 = 16;

/// A thin-lens camera for depth of field.
#[derive(Clone, Copy, Debug)]
pub struct DepthOfField {
    /// Aperture as an f-number (f/1.4 is very shallow, f/8 nearly everything sharp).
    pub f_stop: f32,
    /// Focus distance in metres; `None` focuses on whatever is at the centre of the screen.
    pub focus_distance: Option<f32>,
    /// Sensor height in metres (0.024 = full-frame 35 mm). With the vertical field of view it
    /// defines the focal length, hence how shallow the depth of field is.
    pub sensor_height: f32,
}

impl Default for DepthOfField {
    fn default() -> Self {
        Self { f_stop: 1.8, focus_distance: None, sensor_height: 0.024 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostFxUniform {
    params: Vec4,
    focus: Vec4,
    /// Output width, height, 1 / width, 1 / height.
    output: Vec4,
}

struct Targets {
    size: (u32, u32),
    tiles: [wgpu::TextureView; 2],
    color: [wgpu::TextureView; 2],
}

pub struct PostFxPass {
    layout: wgpu::BindGroupLayout,
    focus: wgpu::ComputePipeline,
    tiles: wgpu::ComputePipeline,
    dilate: wgpu::ComputePipeline,
    dof: wgpu::ComputePipeline,
    motion: wgpu::ComputePipeline,
    uniform: wgpu::Buffer,
    focus_state: wgpu::Buffer,
    targets: Option<Targets>,
    motion_blur: bool,
    depth_of_field: bool,
}

impl PostFxPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "post_fx.wgsl");
        let layout = bind_group_layout(
            device,
            "post fx",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::texture(false)),
                (3, bind::texture(false)),
                (4, bind::texture(false)),
                (5, bind::texture(false)),
                (6, bind::storage_texture(HDR)),
                (7, bind::storage_texture(HDR)),
                (8, bind::storage(false)),
                (9, bind::sampler()),
            ],
        );
        let pipeline = |entry| compute_pipeline(device, "post fx", &module, entry, &layout, &[]);
        Self {
            focus: pipeline("focus"),
            tiles: pipeline("tiles"),
            dilate: pipeline("dilate"),
            dof: pipeline("dof"),
            motion: pipeline("motion"),
            layout,
            uniform: uniform_buffer(device, "post fx", std::mem::size_of::<PostFxUniform>() as u64),
            focus_state: storage_buffer(device, "focus", 16, wgpu::BufferUsages::empty()),
            targets: None,
            motion_blur: false,
            depth_of_field: false,
        }
    }

    /// Writes this frame's parameters. `shutter` is the fraction of the frame the shutter is open
    /// (0.5 = a 180° shutter, 0 disables motion blur); `size` is the output resolution.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, shutter: f32, dof: Option<&DepthOfField>, fov_y: f32, size: (u32, u32)) {
        self.motion_blur = shutter > 0.0;
        self.depth_of_field = dof.is_some();
        if (self.motion_blur || self.depth_of_field) && self.targets.as_ref().is_none_or(|t| t.size != size) {
            self.targets = Some(Self::create_targets(device, size));
        }
        let height = size.1 as f32;
        let (aperture, focal_length, focus) = match dof {
            Some(d) => {
                let focal_length = d.sensor_height * 0.5 / (fov_y * 0.5).tan();
                // Aperture diameter, expressed so that the shader's CoC comes out in pixels.
                let aperture_px = focal_length / d.f_stop.max(0.5) / d.sensor_height * height;
                (aperture_px, focal_length, d.focus_distance.unwrap_or(0.0))
            }
            None => (0.0, 0.05, 0.0),
        };
        let uniform = PostFxUniform {
            params: Vec4::new(shutter, (0.05 * height).max(4.0), aperture, focal_length),
            focus: Vec4::new(focus, (0.02 * height).max(4.0), 0.0, 0.0),
            output: Vec4::new(size.0 as f32, height, 1.0 / size.0 as f32, 1.0 / height),
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    fn create_targets(device: &wgpu::Device, (width, height): (u32, u32)) -> Targets {
        let view = |label, width, height| {
            texture(
                device,
                TextureSpec { label, width, height, layers: 1, mips: 1, format: HDR, usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING },
            )
            .create_view(&Default::default())
        };
        let (tw, th) = (width.div_ceil(TILE), height.div_ceil(TILE));
        Targets {
            size: (width, height),
            tiles: [view("post fx tiles", tw, th), view("post fx tiles dilated", tw, th)],
            color: [view("post fx 0", width, height), view("post fx 1", width, height)],
        }
    }

    /// Applies depth of field then motion blur to `input` and returns the image to continue
    /// with (`input` itself when both are off).
    pub fn record<'a>(&'a self, rec: &mut Recorder, frame: &wgpu::Buffer, targets: &FrameTargets, samplers: &Samplers, input: &'a wgpu::TextureView) -> &'a wgpu::TextureView {
        let Some(t) = self.targets.as_ref().filter(|_| self.motion_blur || self.depth_of_field) else { return input };
        let device = rec.device;
        let group = |color_in: &wgpu::TextureView, tiles_in: &wgpu::TextureView, tiles_out: &wgpu::TextureView, color_out: &wgpu::TextureView| {
            bind_group(
                device,
                "post fx",
                &self.layout,
                &[
                    (0, frame.as_entire_binding()),
                    (1, self.uniform.as_entire_binding()),
                    (2, TextureView(color_in)),
                    (3, TextureView(&targets.depth)),
                    (4, TextureView(&targets.gbuffer_motion)),
                    (5, TextureView(tiles_in)),
                    (6, TextureView(tiles_out)),
                    (7, TextureView(color_out)),
                    (8, self.focus_state.as_entire_binding()),
                    (9, Sampler(&samplers.linear)),
                ],
            )
        };
        let (w, h) = t.size;
        let tiles = (w.div_ceil(TILE), h.div_ceil(TILE));
        // Tile passes write tiles[0] then tiles[1]; colour passes read tiles[1].
        let classify = group(input, &t.tiles[1], &t.tiles[0], &t.color[0]);
        let dilate = group(input, &t.tiles[0], &t.tiles[1], &t.color[0]);
        let mut pass = rec.compute_pass("dof + motion blur");
        pass.set_bind_group(0, &classify, &[]);
        pass.set_pipeline(&self.focus);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.tiles);
        pass.dispatch_workgroups(workgroups(tiles.0, 8), workgroups(tiles.1, 8), 1);
        pass.set_bind_group(0, &dilate, &[]);
        pass.set_pipeline(&self.dilate);
        pass.dispatch_workgroups(workgroups(tiles.0, 8), workgroups(tiles.1, 8), 1);

        let mut current = input;
        let mut next = 0;
        if self.depth_of_field {
            let g = group(current, &t.tiles[1], &t.tiles[0], &t.color[next]);
            pass.set_bind_group(0, &g, &[]);
            pass.set_pipeline(&self.dof);
            pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
            current = &t.color[next];
            next = 1;
        }
        if self.motion_blur {
            let g = group(current, &t.tiles[1], &t.tiles[0], &t.color[next]);
            pass.set_bind_group(0, &g, &[]);
            pass.set_pipeline(&self.motion);
            pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
            current = &t.color[next];
        }
        current
    }
}
