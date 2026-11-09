//! Local exposure (UE5's "Local Exposure"), after TSR and the camera effects, before bloom and
//! tone mapping, at output resolution (see `shaders/local_exposure.wgsl`).
//!
//! A single exposure cannot hold a scene whose bright and dark regions are many stops apart
//! (Snell's window seen from under water, sunlit windows over dark aisles, a dawn sky over a
//! shaded valley). Local exposure splits each pixel's log luminance into a smooth, edge-aware
//! *base* — the average of its neighbours of similar brightness, looked up in a bilateral grid —
//! and the *detail* on top of it, then compresses only the base's distance from middle grey:
//! bright regions come down, dark ones up, while texture and edges keep their full contrast.

use glam::Vec4;
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat, TextureUsages};

use super::Samplers;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, texture, uniform_buffer, workgroups, TextureSpec};

const HDR: TextureFormat = TextureFormat::Rgba16Float;
/// Log-luminance bins of the bilateral grid (`BINS` in local_exposure.wgsl).
pub const BINS: u32 = 32;
/// The grid's bins span middle grey ± this many stops; brighter or darker pixels share the
/// outermost bins (their base is clamped to that range too).
pub const RANGE_STOPS: f32 = 10.0;
/// Rows of grid cells over the output height: a cell is about 1/45 of the screen (32 px at
/// 1440p, 20 px at 900p), so the result does not depend on the output resolution.
const GRID_ROWS: u32 = 45;

/// Local exposure settings (UE's `LocalExposure*` post-process settings). "Middle grey" is the
/// global exposure's metered scene average (before exposure compensation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalExposure {
    /// Contrast of regions brighter than middle grey relative to it (1 = untouched, 0.5 = their
    /// distance from middle grey in stops halves).
    pub highlight_contrast: f32,
    /// The same for regions darker than middle grey (below 1 lifts shadows).
    pub shadow_contrast: f32,
    /// How much of the base is the plain (not edge-aware) blurred luminance instead of the
    /// bilateral grid's: the bilateral base alone flattens large regions into an "HDR look";
    /// the wide blur keeps their relative brightness, at the cost of soft, broad halos.
    pub blurred_luminance_blend: f32,
}

impl Default for LocalExposure {
    fn default() -> Self {
        Self { highlight_contrast: 0.6, shadow_contrast: 0.85, blurred_luminance_blend: 0.4 }
    }
}

impl LocalExposure {
    /// Exposure change in stops for a pixel whose base log luminance is `base` stops from middle
    /// grey: the base moves to `base * contrast`, the detail on top of it stays.
    pub fn exposure_ev(&self, base: f32) -> f32 {
        let contrast = if base > 0.0 { self.highlight_contrast } else { self.shadow_contrast };
        base * (contrast - 1.0)
    }
}

/// Maps a log luminance relative to middle grey (in stops) to the grid's continuous bin
/// coordinate (bin `i` covers `[i, i + 1)`): `±RANGE_STOPS` fall on the centres of the outermost
/// bins, anything beyond is clamped to them.
pub fn bin_coordinate(stops: f32) -> f32 {
    let t = ((stops + RANGE_STOPS) / (2.0 * RANGE_STOPS)).clamp(0.0, 1.0);
    0.5 + t * (BINS - 1) as f32
}

/// [`bin_coordinate`] inside the range as `stops * scale + offset`: (scale, offset), what the
/// shader evaluates (and clamps).
fn bin_mapping() -> (f32, f32) {
    let offset = bin_coordinate(0.0);
    (bin_coordinate(1.0) - offset, offset)
}

/// Size of a grid cell in output pixels and the grid's cell count.
pub fn grid_layout((width, height): (u32, u32)) -> (u32, (u32, u32)) {
    let tile = height.div_ceil(GRID_ROWS).clamp(8, 64);
    (tile, (width.div_ceil(tile), height.div_ceil(tile)))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LocalExposureUniform {
    /// highlight contrast, shadow contrast, blurred luminance blend, exposure compensation (EV).
    params: Vec4,
    /// Cell size in pixels, its inverse, cells in x and y.
    grid: Vec4,
    /// [`bin_coordinate`] as scale and offset of the stops, and `RANGE_STOPS`.
    bins: Vec4,
}

struct Targets {
    output: (u32, u32),
    /// Bilateral grid ping-pong (cells × BINS): r = Σ weight · (log luminance + RANGE), g = Σ weight.
    grid: [wgpu::TextureView; 2],
    /// The same summed over all bins (cells × 1): the plain blurred log luminance.
    blurred: [wgpu::TextureView; 2],
    color: wgpu::TextureView,
}

pub struct LocalExposurePass {
    layout: wgpu::BindGroupLayout,
    build: wgpu::ComputePipeline,
    blur: [wgpu::ComputePipeline; 3],
    apply: wgpu::ComputePipeline,
    uniform: wgpu::Buffer,
    targets: Option<Targets>,
    enabled: bool,
}

impl LocalExposurePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "local_exposure.wgsl");
        let layout = bind_group_layout(
            device,
            "local exposure",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::storage(true)),
                (2, bind::texture(false)),
                (3, bind::texture_3d(true)),
                (4, bind::texture_3d(true)),
                (5, bind::storage_texture_3d(HDR)),
                (6, bind::storage_texture_3d(HDR)),
                (7, bind::storage_texture(HDR)),
                (8, bind::sampler()),
            ],
        );
        let pipeline = |label, entry, constants: &[(&str, f64)]| compute_pipeline(device, label, &module, entry, &layout, constants);
        Self {
            build: pipeline("local exposure build", "build", &[]),
            blur: [0.0, 1.0, 2.0].map(|axis| pipeline("local exposure blur", "blur", &[("AXIS", axis)])),
            apply: pipeline("local exposure apply", "apply", &[]),
            layout,
            uniform: uniform_buffer(device, "local exposure", std::mem::size_of::<LocalExposureUniform>() as u64),
            targets: None,
            enabled: false,
        }
    }

    /// Writes this frame's parameters (`None` disables the pass); `output` is the output
    /// resolution, `exposure_compensation` the global exposure's (middle grey sits that many
    /// stops away from the metered average).
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, settings: Option<&LocalExposure>, exposure_compensation: f32, output: (u32, u32)) {
        self.enabled = settings.is_some();
        let Some(s) = settings else { return };
        if self.targets.as_ref().is_none_or(|t| t.output != output) {
            self.targets = Some(Self::create_targets(device, output));
        }
        let (tile, (cells_x, cells_y)) = grid_layout(output);
        let uniform = LocalExposureUniform {
            params: Vec4::new(s.highlight_contrast, s.shadow_contrast, s.blurred_luminance_blend.clamp(0.0, 1.0), exposure_compensation),
            grid: Vec4::new(tile as f32, 1.0 / tile as f32, cells_x as f32, cells_y as f32),
            bins: {
                let (scale, offset) = bin_mapping();
                Vec4::new(scale, offset, RANGE_STOPS, 0.0)
            },
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    fn create_targets(device: &wgpu::Device, output: (u32, u32)) -> Targets {
        let (_, (cells_x, cells_y)) = grid_layout(output);
        let volume = |label, depth| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: cells_x, height: cells_y, depth_or_array_layers: depth },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: HDR,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let color = texture(
            device,
            TextureSpec { label: "local exposure", width: output.0, height: output.1, layers: 1, mips: 1, format: HDR, usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING },
        );
        Targets {
            output,
            grid: [volume("local exposure grid 0", BINS), volume("local exposure grid 1", BINS)],
            blurred: [volume("local exposure blurred 0", 1), volume("local exposure blurred 1", 1)],
            color: color.create_view(&Default::default()),
        }
    }

    /// Applies local exposure to `input` (the scene at output resolution) and returns the image
    /// to continue with (`input` itself when the pass is off). `exposure` is the global
    /// exposure buffer of this frame.
    pub fn record<'a>(&'a self, rec: &mut Recorder, exposure: &wgpu::Buffer, samplers: &Samplers, input: &'a wgpu::TextureView) -> &'a wgpu::TextureView {
        let Some(t) = self.targets.as_ref().filter(|_| self.enabled) else { return input };
        let device = rec.device;
        // Every dispatch reads one half of the ping-pong pairs and writes the other.
        let group = |grid_in: usize, blurred_in: usize| {
            bind_group(
                device,
                "local exposure",
                &self.layout,
                &[
                    (0, self.uniform.as_entire_binding()),
                    (1, exposure.as_entire_binding()),
                    (2, TextureView(input)),
                    (3, TextureView(&t.grid[grid_in])),
                    (4, TextureView(&t.blurred[blurred_in])),
                    (5, TextureView(&t.grid[1 - grid_in])),
                    (6, TextureView(&t.blurred[1 - blurred_in])),
                    (7, TextureView(&t.color)),
                    (8, Sampler(&samplers.linear)),
                ],
            )
        };
        let (_, (cells_x, cells_y)) = grid_layout(t.output);
        let mut pass = rec.compute_pass("local exposure");
        // Splat into grid[0] / blurred[0].
        pass.set_bind_group(0, &group(1, 1), &[]);
        pass.set_pipeline(&self.build);
        pass.dispatch_workgroups(cells_x, cells_y, 1);
        // Separable Gaussian: x and y (grid and blurred luminance), then z (grid only), ending
        // in grid[1] and blurred[0].
        for (axis, (grid_in, blurred_in)) in [(0, 0), (1, 1), (0, 0)].into_iter().enumerate() {
            pass.set_bind_group(0, &group(grid_in, blurred_in), &[]);
            pass.set_pipeline(&self.blur[axis]);
            pass.dispatch_workgroups(workgroups(cells_x, 4), workgroups(cells_y, 4), BINS / 4);
        }
        pass.set_bind_group(0, &group(1, 0), &[]);
        pass.set_pipeline(&self.apply);
        pass.dispatch_workgroups(workgroups(t.output.0, 8), workgroups(t.output.1, 8), 1);
        &t.color
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_contrast_changes_nothing() {
        let neutral = LocalExposure { highlight_contrast: 1.0, shadow_contrast: 1.0, blurred_luminance_blend: 0.3 };
        for base in [-8.0, -1.0, 0.0, 0.5, 6.0] {
            assert_eq!(neutral.exposure_ev(base), 0.0);
        }
    }

    #[test]
    fn highlights_come_down_and_shadows_up() {
        let s = LocalExposure { highlight_contrast: 0.5, shadow_contrast: 0.75, blurred_luminance_blend: 0.0 };
        // A region 4 stops over middle grey ends up 2 stops over it: 2 stops darker.
        assert_eq!(s.exposure_ev(4.0), -2.0);
        // One 4 stops under ends up 3 stops under: 1 stop brighter.
        assert_eq!(s.exposure_ev(-4.0), 1.0);
        // Continuous through middle grey, which never moves.
        assert_eq!(s.exposure_ev(0.0), 0.0);
        assert!(s.exposure_ev(1e-4).abs() < 1e-4 && s.exposure_ev(-1e-4).abs() < 1e-4);
        // Monotonic: compression never swaps the order of two regions' brightness.
        let mut previous = f32::NEG_INFINITY;
        for i in -100..=100 {
            let base = i as f32 * 0.1;
            let result = base + s.exposure_ev(base);
            assert!(result > previous);
            previous = result;
        }
    }

    #[test]
    fn bins_cover_the_range_around_middle_grey() {
        // Middle grey is the grid's centre; the range ends fall on the outermost bin centres.
        assert_eq!(bin_coordinate(0.0), BINS as f32 / 2.0);
        assert_eq!(bin_coordinate(-RANGE_STOPS), 0.5);
        assert_eq!(bin_coordinate(RANGE_STOPS), BINS as f32 - 0.5);
        // Out of range clamps to them.
        assert_eq!(bin_coordinate(-40.0), 0.5);
        assert_eq!(bin_coordinate(40.0), BINS as f32 - 0.5);
        // Bin centres are 2 · RANGE / (BINS - 1) stops apart.
        let width = 2.0 * RANGE_STOPS / (BINS - 1) as f32;
        assert!((bin_coordinate(1.0) - bin_coordinate(1.0 - width) - 1.0).abs() < 1e-5);
        // The shader's linear form (clamped to the outermost bin centres) is the same mapping.
        let (scale, offset) = bin_mapping();
        for i in -30..=30 {
            let stops = i as f32 * 0.5;
            let shader = (stops * scale + offset).clamp(0.5, BINS as f32 - 0.5);
            assert!((shader - bin_coordinate(stops)).abs() < 1e-4, "{stops}");
        }
    }

    #[test]
    fn grid_cells_follow_the_output_height() {
        assert_eq!(grid_layout((2560, 1440)), (32, (80, 45)));
        assert_eq!(grid_layout((1600, 900)), (20, (80, 45)));
        // Tiny and huge outputs stay within the cell size the build shader handles.
        assert_eq!(grid_layout((64, 32)).0, 8);
        assert_eq!(grid_layout((16384, 8192)).0, 64);
    }

    #[test]
    fn shader_bins_match() {
        let source = include_str!("../shaders/local_exposure.wgsl");
        assert!(source.contains(&format!("const BINS: u32 = {BINS}u;")));
    }
}
