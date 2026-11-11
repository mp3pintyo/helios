//! Adaptive voxel-based order-independent transparency (AVBOIT; UE 5.9's experimental
//! `r.OIT.Method=1`).
//!
//! Weighted blended OIT (`oit.rs`) guesses each fragment's share from its depth alone; it
//! saturates under many layers, and the order between a glass pane, the sparks behind it and the
//! smoke around them is lost. AVBOIT measures it instead:
//!
//! 1. **Bounds.** A camera-aligned grid of columns, one per `tile`² internal pixels (4² up to
//!    2048 pixels wide, 8² above), each with [`SLICES`] cells. A low-resolution draw of every
//!    order-independent translucent surface and particle records per column the nearest and
//!    farthest translucent view depth; a column's cells are spaced logarithmically between the
//!    bounds of its 3×3 neighbourhood — *adaptive*: they sit where translucency actually is, not
//!    over the whole view range, and neighbours space them alike (no streaks where a surface
//!    crosses columns). Volumes don't widen the bounds (smoke or a lake's mist would spread the
//!    cells over metres): their optical depth in front of the near bound is kept per column.
//! 2. **Transmittance.** The same low-resolution draw adds each fragment's optical depth,
//!    `-ln(transmittance)` per colour channel (coloured glass: rgb; particles and blended
//!    surfaces: grey) to the cell it falls in, with 32-bit atomics on 16-bit fixed-point fields
//!    (WebGPU has no rasteriser-ordered views; a sum is order independent); the volumes
//!    integrate their extinction over every cell (compute). Plus each cell's summed coverage.
//! 3. **Resolve** (`avboit.wgsl`): per column the exclusive product along the cells — the
//!    transmittance *in front of* each cell — into a 3D texture, with each cell's normalisation
//!    (its actual opacity over its fragments' summed coverage: one fragment keeps its full
//!    weight, several in one cell blend as their weighted average, as in UE).
//! 4. **Full-resolution draws** weight their premultiplied colour by the transmittance in front
//!    of their depth (bilinear between the four nearest columns) and *add* it, and multiply the
//!    background's transmittance by their own: order independent, coloured, any number of layers.
//!    The volume march weights each step by the surfaces in front of it (`avboit_surface`).
//!
//! The material contract (as UE's): a translucent surface composited this way contributes only
//! its *own* radiance (lighting, reflections, emission, fog in front) and its transmittance —
//! never a read of the scene colour behind it, which the composite applies once. Refraction and
//! blurred transmission read the background, so refracting surfaces keep the depth-peeled path
//! (`translucency.rs` decides which).
//!
//! Limits (as UE's): order within one cell is a weighted average; the low-resolution transmittance
//! shows its cells at pane edges (a few pixels of blend); sub-cell sparks may miss the
//! low-resolution draw (they barely attenuate anything).

use bytemuck::{Pod, Zeroable};
use wgpu::BindingResource::TextureView;
use wgpu::{BindingResource, BindingType, ShaderStages, TextureFormat};

use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, storage_buffer, uniform_buffer, workgroups, Lazy};

/// Cells per column.
pub(crate) const SLICES: u32 = 16;
/// The transmittance in front of each cell, the bounds in slice 0 (`avboit_front`).
const FRONT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// The surfaces' transmittance only (`avboit_surface`, read by the volume march).
const SURFACE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
/// The low-resolution draws' colour target (written by nothing: WebGPU render passes need an
/// attachment).
const DUMMY_FORMAT: TextureFormat = TextureFormat::R8Unorm;

/// `Avboit` in avboit_common.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct AvboitUniform {
    grid: [u32; 4],
}

/// Internal pixels per column side for targets of `capacity`.
pub(crate) fn tile_for(capacity: (u32, u32)) -> u32 {
    if capacity.0 > 2048 {
        8
    } else {
        4
    }
}

/// Columns of a grid over `size` internal pixels.
fn columns(size: (u32, u32), tile: u32) -> (u32, u32) {
    (size.0.div_ceil(tile).max(1), size.1.div_ceil(tile).max(1))
}

/// The grid at the internal resolution's capacity.
struct Grid {
    capacity: (u32, u32),
    tile: u32,
    /// Three words per cell (avboit_write.wgsl).
    cells: wgpu::Buffer,
    /// Two words per column.
    bounds: wgpu::Buffer,
    front: wgpu::TextureView,
    surface: wgpu::TextureView,
    dummy: wgpu::TextureView,
}

impl Grid {
    fn new(device: &wgpu::Device, capacity: (u32, u32)) -> Self {
        let tile = tile_for(capacity);
        let (x, y) = columns(capacity, tile);
        let cells = x as u64 * y as u64 * SLICES as u64;
        let volume = |label: &str, depth: u32, format: TextureFormat| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: x, height: y, depth_or_array_layers: depth },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let dummy = crate::util::texture(
            device,
            crate::util::TextureSpec { label: "avboit low-resolution target", width: x, height: y, layers: 1, mips: 1, format: DUMMY_FORMAT, usage: wgpu::TextureUsages::RENDER_ATTACHMENT },
        )
        .create_view(&Default::default());
        let column_count = x as u64 * y as u64;
        let bytes = cells * 12 + column_count * 16 + column_count * (SLICES as u64 + 2) * 8 + column_count * (SLICES as u64 + 1) * 4 + column_count;
        log::info!("avboit: {x}×{y} columns of {tile}² pixels, {SLICES} cells each ({:.1} MiB)", bytes as f64 / (1 << 20) as f64);
        Self {
            capacity,
            tile,
            cells: storage_buffer(device, "avboit cells", cells * 12, wgpu::BufferUsages::empty()),
            bounds: storage_buffer(device, "avboit bounds", column_count * 16, wgpu::BufferUsages::empty()),
            front: volume("avboit front transmittance", SLICES + 2, FRONT_FORMAT),
            surface: volume("avboit surface transmittance", SLICES + 1, SURFACE_FORMAT),
            dummy,
        }
    }
}

/// Stand-ins bound while the grid is off (the uniform's zero grid tells the shaders).
struct Placeholder {
    cells: wgpu::Buffer,
    bounds: wgpu::Buffer,
    front: wgpu::TextureView,
    surface: wgpu::TextureView,
}

pub(crate) struct Avboit {
    uniform: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    resolve: Lazy<wgpu::ComputePipeline>,
    dilate: Lazy<wgpu::ComputePipeline>,
    placeholder: Placeholder,
    grid: Option<Grid>,
    /// Columns this frame (0: off).
    current: AvboitUniform,
    written: Option<AvboitUniform>,
}

impl Avboit {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "avboit.wgsl");
        let layout = bind_group_layout(
            device,
            "avboit resolve",
            ShaderStages::COMPUTE,
            &[
                (120, bind::uniform()),
                (123, bind::storage(false)),
                (124, bind::storage(false)),
                (125, bind::storage_texture_3d(FRONT_FORMAT)),
                (126, bind::storage_texture_3d(SURFACE_FORMAT)),
            ],
        );
        let tiny = |label: &str, format: TextureFormat| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Self {
            uniform: uniform_buffer(device, "avboit", std::mem::size_of::<AvboitUniform>() as u64),
            resolve: lazy_compute_pipeline(device, "avboit resolve", &module, "resolve", &layout, &[]),
            dilate: lazy_compute_pipeline(device, "avboit bounds dilation", &module, "dilate", &layout, &[]),
            layout,
            placeholder: Placeholder {
                cells: storage_buffer(device, "avboit cells (off)", 16, wgpu::BufferUsages::empty()),
                bounds: storage_buffer(device, "avboit bounds (off)", 16, wgpu::BufferUsages::empty()),
                front: tiny("avboit front (off)", FRONT_FORMAT),
                surface: tiny("avboit surface (off)", SURFACE_FORMAT),
            },
            grid: None,
            current: AvboitUniform::default(),
            written: None,
        }
    }

    /// Queues the resolve for the browser's background compile.
    pub fn prefetch(&self) {
        self.resolve.prefetch();
        self.dilate.prefetch();
    }

    /// Turns the grid on over this frame's `internal` resolution (allocated for `capacity`) or
    /// off. Grown, never shrunk while on.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enabled: bool, internal: (u32, u32), capacity: (u32, u32)) {
        self.current = AvboitUniform::default();
        if enabled {
            if self.grid.as_ref().is_none_or(|g| g.capacity != capacity) {
                self.grid = Some(Grid::new(device, capacity));
            }
            let tile = self.grid.as_ref().map_or(4, |g| g.tile);
            let (x, y) = columns(internal, tile);
            self.current = AvboitUniform { grid: [x, y, SLICES, tile] };
        }
        if self.written != Some(self.current) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.current));
            self.written = Some(self.current);
        }
    }

    /// Whether the grid is built this frame.
    pub fn enabled(&self) -> bool {
        self.current.grid[0] > 0
    }

    /// Low-resolution viewport this frame (columns).
    pub fn columns(&self) -> (u32, u32) {
        (self.current.grid[0], self.current.grid[1])
    }

    pub fn tile(&self) -> u32 {
        self.current.grid[3]
    }

    fn live(&self) -> Option<&Grid> {
        self.grid.as_ref().filter(|_| self.enabled())
    }

    /// Layout entries of the readers (`avboit_read.wgsl`).
    pub fn read_entries() -> [(u32, BindingType); 3] {
        [(120, bind::uniform()), (121, bind::texture_3d(false)), (122, bind::texture_3d(false))]
    }

    pub fn read_resources(&self) -> [(u32, BindingResource<'_>); 3] {
        let (front, surface) = self.live().map_or((&self.placeholder.front, &self.placeholder.surface), |g| (&g.front, &g.surface));
        [(120, self.uniform.as_entire_binding()), (121, TextureView(front)), (122, TextureView(surface))]
    }

    /// Layout entries of the writers (`avboit_write.wgsl`).
    pub fn write_entries() -> [(u32, BindingType); 3] {
        [(120, bind::uniform()), (123, bind::storage(false)), (124, bind::storage(false))]
    }

    pub fn write_resources(&self) -> [(u32, BindingResource<'_>); 3] {
        let (cells, bounds) = self.live().map_or((&self.placeholder.cells, &self.placeholder.bounds), |g| (&g.cells, &g.bounds));
        [(120, self.uniform.as_entire_binding()), (123, cells.as_entire_binding()), (124, bounds.as_entire_binding())]
    }

    /// Zeroes the bounds and cells (first of the frame's grid passes).
    pub fn record_clear(&self, rec: &mut Recorder) {
        let Some(g) = self.live() else { return };
        let (x, y) = self.columns();
        rec.encoder.clear_buffer(&g.bounds, 0, Some(x as u64 * y as u64 * 16));
        rec.encoder.clear_buffer(&g.cells, 0, Some(x as u64 * y as u64 * SLICES as u64 * 12));
    }

    /// A render pass over the columns for the low-resolution draws (bounds, then transmittance).
    pub fn low_resolution_pass<'r>(&self, rec: &'r mut Recorder, label: &str) -> Option<wgpu::RenderPass<'r>> {
        let g = self.live()?;
        let attachment = Some(wgpu::RenderPassColorAttachment {
            view: &g.dummy,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Discard },
        });
        let mut pass = rec.render_pass(label, &[attachment], None);
        let (x, y) = self.columns();
        pass.set_viewport(0.0, 0.0, x as f32, y as f32, 0.0, 1.0);
        Some(pass)
    }

    /// After the bounds draw: fixes each column's cells from its neighbourhood's bounds.
    pub fn record_dilate(&self, rec: &mut Recorder) {
        self.dispatch(rec, "avboit bounds dilation", &self.dilate);
    }

    /// Turns the cells into the transmittance textures.
    pub fn record_resolve(&self, rec: &mut Recorder) {
        self.dispatch(rec, "avboit resolve", &self.resolve);
    }

    fn dispatch(&self, rec: &mut Recorder, label: &str, pipeline: &wgpu::ComputePipeline) {
        let Some(g) = self.live() else { return };
        let group = bind_group(
            rec.device,
            "avboit resolve",
            &self.layout,
            &[
                (120, self.uniform.as_entire_binding()),
                (123, g.cells.as_entire_binding()),
                (124, g.bounds.as_entire_binding()),
                (125, TextureView(&g.front)),
                (126, TextureView(&g.surface)),
            ],
        );
        let (x, y) = self.columns();
        let mut pass = rec.compute_pass(label);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(x, 8), workgroups(y, 8), 1);
    }
}

/// The colour target state of a low-resolution draw (nothing written).
pub(crate) fn low_resolution_target() -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState { format: DUMMY_FORMAT, blend: None, write_mask: wgpu::ColorWrites::empty() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_matches_the_shader_layout() {
        assert_eq!(std::mem::size_of::<AvboitUniform>(), 16);
    }

    #[test]
    fn columns_cover_the_view() {
        assert_eq!(tile_for((1600, 900)), 4);
        assert_eq!(tile_for((2560, 1440)), 8);
        assert_eq!(columns((1600, 900), 4), (400, 225));
        assert_eq!(columns((1601, 901), 4), (401, 226));
    }
}
