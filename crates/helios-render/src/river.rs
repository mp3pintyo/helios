//! Shallow-water rivers (UE 5.9's simulation-driven rivers): water that flows over a grid
//! following the terrain instead of a flat, normal-mapped sheet (`shaders/river_sim.wgsl`).
//!
//! * **Simulation.** A height-field shallow-water model, the "virtual pipes" scheme (O'Brien &
//!   Hodgins 1995, Mei et al. 2007) on the GPU: every cell holds a water depth and the flow
//!   through four pipes to its neighbours; each step accelerates the pipes' flow by the
//!   difference in water surface height, slows it by bed friction (Manning), scales it so a cell
//!   never gives more water than it holds, and moves the water. Velocity comes from the flow,
//!   foam is advected along it (semi-Lagrangian) and born where the flow is fast or the surface
//!   steep. The bed is the terrain the scene baked into [`River::bed`] — the mesh terrain's
//!   channel, rock walls as high cells, boulders standing in the stream as raised cells, so the
//!   current parts around them. [`RiverSource`]s pour water in; *outlet* cells (the lake the
//!   river runs into) hold their surface at [`River::outlet_level`], draining it. Fixed steps of
//!   [`STEP`] seconds, at most [`MAX_STEPS`] a frame; the first frame runs [`WARMUP_STEPS`] so a
//!   scene opens on a settled river.
//! * **Surface.** The river's surface instance is a grid mesh with one vertex per cell, drawn by
//!   the forward water shading like the lake. Every frame a compute pass rewrites its vertices
//!   in the translucency pass's vertex buffer: height (bed + depth, or just under the bed where
//!   dry, so the shoreline is the surface's intersection with the ground), normal from the water
//!   surface's slope, the velocity (second UV set) and foam and depth (vertex colour, alpha 0
//!   marking simulated water). The water shader advects its waves along the velocity (a
//!   two-phase flow map) and draws the simulated foam.
//! * **Shores.** The surface height, dilated a few cells onto the banks, goes into a texture the
//!   G-buffer resolve reads ([`BINDING_RIVER_SURFACE`]): the river's bed and banks get the lake
//!   bed's sediment and wet band, following the simulated water line.
//!
//! One river per scene (the first of [`Scene::rivers`]).

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use wgpu::TextureFormat;

use crate::pipelines::Lazy;
use crate::profiler::Recorder;
use crate::scene::{InstanceHandle, Scene};
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, storage_buffer, storage_buffer_init, uniform_buffer, workgroups};

/// Binding of the river surface texture in the G-buffer resolve (shores).
pub const BINDING_RIVER_SURFACE: u32 = 150;
/// Simulation step (seconds).
pub const STEP: f32 = 1.0 / 120.0;
/// Steps per frame at most (a slow frame lets the river run slow rather than explode).
pub const MAX_STEPS: u32 = 6;
/// Steps the first frame runs.
pub const WARMUP_STEPS: u32 = 1800;
pub const MAX_RIVER_SOURCES: usize = 4;
/// Floats per vertex of the translucency pass's vertex buffer (`TranslucentVertex`).
const VERTEX_FLOATS: u32 = 14;
const _: () = assert!(std::mem::size_of::<crate::translucency::TranslucentVertex>() == VERTEX_FLOATS as usize * 4);

/// Water poured into a river.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverSource {
    pub position: Vec2,
    pub radius: f32,
    /// Cubic metres per second.
    pub rate: f32,
}

/// A simulated river: a grid of cells over the world (x, z) with its bed and initial water.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct River {
    /// World (x, z) of the grid's corner, cell size and cells per axis.
    pub origin: Vec2,
    pub cell: f32,
    pub size: [u32; 2],
    /// Bed height per cell (x fastest): the ground, walls and obstacles.
    pub bed: Vec<f32>,
    /// Where the surface hides while a cell is dry (just under it): the bed, or for cells under
    /// a wall of rock (whose bed is the wall's height) a point inside the rock.
    pub hide: Vec<f32>,
    /// Initial water depth per cell.
    pub depth: Vec<f32>,
    /// Cells that drain into a body of water held at `outlet_level`.
    pub outlet: Vec<bool>,
    pub outlet_level: f32,
    pub sources: Vec<RiverSource>,
    /// Manning's roughness of the bed (0.03: a clean natural channel).
    pub manning: f32,
    /// The water instance drawing the surface (its mesh from [`River::surface_mesh`]).
    pub surface: Option<InstanceHandle>,
}

impl River {
    /// A grid of `size` cells from `origin`, its bed from `bed` at the cell centres, dry.
    pub fn new(origin: Vec2, cell: f32, size: [u32; 2], bed: impl Fn(Vec2) -> f32) -> Self {
        let cells = (size[0] * size[1]) as usize;
        let mut river = Self { origin, cell, size, manning: 0.03, ..Default::default() };
        river.bed = (0..cells).map(|i| bed(river.center(i))).collect();
        river.hide = river.bed.clone();
        river.depth = vec![0.0; cells];
        river.outlet = vec![false; cells];
        river
    }

    /// World (x, z) of cell `i`'s centre.
    pub fn center(&self, i: usize) -> Vec2 {
        let (x, z) = (i as u32 % self.size[0], i as u32 / self.size[0]);
        self.origin + (Vec2::new(x as f32, z as f32) + 0.5) * self.cell
    }

    /// The surface's resting mesh: a vertex at every cell centre at the initial water surface
    /// (just under the bed where dry), two triangles between four centres.
    pub fn surface_mesh(&self) -> helios_geometry::Mesh {
        let [w, h] = self.size;
        let vertices = (0..(w * h) as usize)
            .map(|i| {
                let c = self.center(i);
                let y = if self.depth[i] > 0.01 { self.bed[i] + self.depth[i] } else { self.hide[i].min(self.bed[i]) - 0.12 };
                helios_geometry::Vertex::new(Vec3::new(c.x, y, c.y), Vec3::Y, (c - self.origin) / self.cell)
            })
            .collect();
        let mut indices = Vec::with_capacity(((w - 1) * (h - 1) * 6) as usize);
        for z in 0..h - 1 {
            for x in 0..w - 1 {
                let i = z * w + x;
                indices.extend_from_slice(&[i, i + w, i + 1, i + 1, i + w, i + w + 1]);
            }
        }
        helios_geometry::Mesh::new(vertices, indices)
    }

    /// World bounds of everything the surface can reach.
    pub fn bounds(&self) -> helios_geometry::Aabb {
        let low = self.bed.iter().fold(f32::MAX, |a, &b| a.min(b)) - 0.2;
        let high = self.bed.iter().zip(&self.depth).zip(&self.hide).filter(|((b, _), h)| **h >= **b - 0.01).fold(f32::MIN, |a, ((b, d), _)| a.max(b + d)) + 1.0;
        let extent = Vec2::new(self.size[0] as f32, self.size[1] as f32) * self.cell;
        helios_geometry::Aabb { min: Vec3::new(self.origin.x, low, self.origin.y), max: Vec3::new(self.origin.x + extent.x, high, self.origin.y + extent.y) }
    }
}

/// Mirrors `RiverParams` in river_sim.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct RiverParams {
    origin: [f32; 2],
    cell: f32,
    dt: f32,
    size: [u32; 2],
    first_vertex: u32,
    vertex_count: u32,
    gravity: f32,
    manning: f32,
    outlet_level: f32,
    source_count: u32,
    sources: [[f32; 4]; MAX_RIVER_SOURCES],
}

/// GPU state of the scene's river.
struct RiverState {
    /// The river it was made for (rebuilt when the scene's river changes).
    river: River,
    terrain: wgpu::Buffer,
    flux: wgpu::Buffer,
    water: [wgpu::Buffer; 2],
    surface_view: wgpu::TextureView,
    /// Which `water` buffer holds the latest state.
    current: usize,
    /// Steps to run this frame.
    steps: u32,
    warm: bool,
}

/// The shallow-water simulation and surface update of the scene's river.
pub struct RiverSim {
    layout: wgpu::BindGroupLayout,
    flux: Lazy<wgpu::ComputePipeline>,
    depth: Lazy<wgpu::ComputePipeline>,
    surface: Lazy<wgpu::ComputePipeline>,
    vertices: Lazy<wgpu::ComputePipeline>,
    params: wgpu::Buffer,
    state: Option<RiverState>,
    /// A 1x1 surface texture for scenes without a river.
    none_view: wgpu::TextureView,
    dummy_vertices: wgpu::Buffer,
    accumulated: f32,
    /// Steps run since the river was made (for the HUD and measurements).
    pub steps_run: u64,
}

fn surface_texture(device: &wgpu::Device, size: [u32; 2]) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("river surface"),
        size: wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    })
}

impl RiverSim {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "river_sim.wgsl");
        let layout = bind_group_layout(
            device,
            "river",
            wgpu::ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::storage(true)),
                (2, bind::storage(false)),
                (3, bind::storage(true)),
                (4, bind::storage(false)),
                (5, bind::storage(false)),
                (6, bind::storage_texture(TextureFormat::R32Float)),
            ],
        );
        let pipeline = |entry: &str| lazy_compute_pipeline(device, &format!("river {entry}"), &module, entry, &layout, &[]);
        let none = surface_texture(device, [1, 1]);
        Self {
            flux: pipeline("flux_step"),
            depth: pipeline("depth_step"),
            surface: pipeline("surface_step"),
            vertices: pipeline("vertex_step"),
            layout,
            params: uniform_buffer(device, "river params", std::mem::size_of::<RiverParams>() as u64),
            state: None,
            none_view: none.create_view(&Default::default()),
            dummy_vertices: storage_buffer(device, "river no vertices", 64, wgpu::BufferUsages::empty()),
            accumulated: 0.0,
            steps_run: 0,
        }
    }

    /// The surface texture the G-buffer resolve reads (1x1 without a river).
    pub fn surface_view(&self) -> &wgpu::TextureView {
        self.state.as_ref().map_or(&self.none_view, |s| &s.surface_view)
    }

    /// Mirrors the scene's river (rebuilding the state when it changed) and decides how many
    /// steps this frame runs: `delta_time` worth while `running`, none while paused.
    /// `vertices`: where the translucency pass holds the surface mesh's vertices (first, count).
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, vertices: Option<(u32, u32)>, delta_time: f32, running: bool) {
        let Some(river) = scene.rivers.first().filter(|r| r.size[0] >= 2 && r.size[1] >= 2) else {
            self.state = None;
            return;
        };
        if self.state.as_ref().is_none_or(|s| s.river != *river) {
            let cells = (river.size[0] * river.size[1]) as usize;
            let terrain: Vec<[f32; 4]> = (0..cells).map(|i| [river.bed[i], river.outlet[i] as u32 as f32, river.hide.get(i).map_or(river.bed[i], |h| h.min(river.bed[i])), 0.0]).collect();
            let water: Vec<[f32; 4]> = (0..cells).map(|i| [river.depth[i], 0.0, 0.0, 0.0]).collect();
            let surface = surface_texture(device, river.size);
            self.state = Some(RiverState {
                river: river.clone(),
                terrain: storage_buffer_init(device, queue, "river terrain", bytemuck::cast_slice(&terrain)),
                flux: storage_buffer_init(device, queue, "river flux", &vec![0u8; cells * 16]),
                water: [0, 1].map(|_| storage_buffer_init(device, queue, "river water", bytemuck::cast_slice(&water))),
                surface_view: surface.create_view(&Default::default()),
                current: 0,
                steps: 0,
                warm: false,
            });
            self.accumulated = 0.0;
            self.steps_run = 0;
            // Made on first use natively; the browser compiles them in the background now.
            for pipeline in [&self.flux, &self.depth, &self.surface, &self.vertices] {
                pipeline.prefetch();
            }
            log::info!("river: {}x{} cells of {} m, {} sources", river.size[0], river.size[1], river.cell, river.sources.len());
        }
        let state = self.state.as_mut().expect("made above");
        state.steps = if !state.warm {
            state.warm = true;
            WARMUP_STEPS
        } else if running {
            self.accumulated = (self.accumulated + delta_time.max(0.0)).min(STEP * MAX_STEPS as f32);
            let steps = (self.accumulated / STEP) as u32;
            self.accumulated -= steps as f32 * STEP;
            steps
        } else {
            0
        };
        self.steps_run += state.steps as u64;
        let mut sources = [[0.0; 4]; MAX_RIVER_SOURCES];
        for (s, source) in sources.iter_mut().zip(&river.sources) {
            *s = [source.position.x, source.position.y, source.radius, source.rate];
        }
        let (first_vertex, vertex_count) = vertices.unwrap_or((0, 0));
        let params = RiverParams {
            origin: river.origin.into(),
            cell: river.cell,
            dt: STEP,
            size: river.size,
            first_vertex,
            vertex_count,
            gravity: 9.81,
            manning: river.manning,
            outlet_level: river.outlet_level,
            source_count: river.sources.len().min(MAX_RIVER_SOURCES) as u32,
            sources,
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }

    /// Runs this frame's steps, then writes the surface texture and (with `vertices`, the
    /// translucency pass's vertex buffer) the surface mesh.
    pub fn record(&mut self, rec: &mut Recorder, vertices: Option<&wgpu::Buffer>, vertex_count: u32) {
        let Some(state) = self.state.as_mut() else { return };
        let vertex_buffer = vertices.unwrap_or(&self.dummy_vertices);
        let group = |current: usize| {
            bind_group(
                rec.device,
                "river",
                &self.layout,
                &[
                    (0, self.params.as_entire_binding()),
                    (1, state.terrain.as_entire_binding()),
                    (2, state.flux.as_entire_binding()),
                    (3, state.water[current].as_entire_binding()),
                    (4, state.water[1 - current].as_entire_binding()),
                    (5, vertex_buffer.as_entire_binding()),
                    (6, wgpu::BindingResource::TextureView(&state.surface_view)),
                ],
            )
        };
        let groups = [group(0), group(1)];
        let (w, h) = (workgroups(state.river.size[0], 8), workgroups(state.river.size[1], 8));
        let mut pass = rec.compute_pass("river simulation");
        for _ in 0..state.steps {
            pass.set_bind_group(0, &groups[state.current], &[]);
            pass.set_pipeline(&self.flux);
            pass.dispatch_workgroups(w, h, 1);
            pass.set_pipeline(&self.depth);
            pass.dispatch_workgroups(w, h, 1);
            state.current = 1 - state.current;
        }
        pass.set_bind_group(0, &groups[state.current], &[]);
        pass.set_pipeline(&self.surface);
        pass.dispatch_workgroups(w, h, 1);
        if vertices.is_some() && vertex_count > 0 {
            pass.set_pipeline(&self.vertices);
            pass.dispatch_workgroups(workgroups(vertex_count, 64), 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_surface_mesh_has_a_vertex_per_cell_at_the_water() {
        let mut river = River::new(Vec2::new(-1.0, 2.0), 0.5, [4, 3], |p| p.x * 0.1);
        river.depth[5] = 0.4;
        let mesh = river.surface_mesh();
        assert_eq!(mesh.vertices.len(), 12);
        assert_eq!(mesh.indices.len(), 3 * 2 * 6);
        let v = mesh.vertices[5].position();
        let c = river.center(5);
        assert_eq!(c, Vec2::new(-1.0 + 0.75, 2.0 + 0.75));
        assert!((v.y - (c.x * 0.1 + 0.4)).abs() < 1e-6);
        assert!(mesh.vertices[0].position[1] < river.bed[0], "dry cells sink under the bed");
        assert_eq!(std::mem::size_of::<RiverParams>(), 112);
    }
}
