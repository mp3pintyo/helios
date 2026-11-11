//! Stochastic local-light sampling for Lumen's surface caches (UE's MegaLights idea applied to
//! the card and voxel lighting).
//!
//! The surface caches lit every card texel and voxel with every local light in range, each with
//! a global-distance-field shadow cone: with hundreds of candles that loop became the largest
//! light-count-dependent GI cost. Instead:
//!
//! * **Light grid** (`lumen_light_grid.wgsl`) — a camera-centred clipmap of
//!   [`LEVELS`] × [`CELLS`]³ world cells over the global distance field's extents (cells of 4,
//!   8, 16 and 32 m), rebuilt every frame: one workgroup per cell lists the lights whose range
//!   reaches it, all of them when at most [`ENTRIES`] do, otherwise [`ENTRIES`] drawn by
//!   systematic resampling in proportion to an upper bound of their contribution to the cell,
//!   each weighted by the inverse of its probability (a fresh offset every frame).
//! * **Sampling** (`lumen_light_sample.wgsl`) — a texel or voxel evaluates the unshadowed
//!   contribution of its cell's candidates, picks two in proportion to it and shadows only
//!   those (one cone each); with at most two contributing lights it shadows them all (exact).
//! * **Accumulation** — voxels already blend every refresh half into their previous value; card
//!   texels keep their sampled local light in this module's buffer (rgb9e5, one word per atlas
//!   texel, behind the grid) and blend half of each refresh into it, restarting when their page
//!   is reallocated or its instance moved (`FRESH_PAGE` in the update records).
//!
//! [`crate::RenderSettings::lumen_light_sampling`] off restores the exhaustive loop.

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};

use crate::pipelines::Lazy;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer, uniform_buffer};

/// Clipmap levels (`LIGHT_GRID_LEVELS` in lumen_light_grid_common.wgsl).
pub const LEVELS: usize = 4;
/// Cells per axis and level (`LIGHT_GRID_CELLS`).
pub const CELLS: u32 = 8;
/// Candidate entries per cell (`LIGHT_GRID_ENTRIES`).
pub const ENTRIES: u32 = 32;
/// Words per cell: count, exact flag, (light, weight) per entry (`LIGHT_GRID_CELL_WORDS`).
const CELL_WORDS: u32 = 2 + 2 * ENTRIES;
const GRID_CELLS: u32 = CELLS * CELLS * CELLS * LEVELS as u32;
/// Cell size of the finest level: the finest distance-field level's extent (0.25 m × 128) over
/// [`CELLS`]; each level doubles it.
const FIRST_CELL: f32 = 32.0 / CELLS as f32;
/// Texels of the mesh-card lighting atlas (`ATLAS_PAGES` × 8 per side, cards.rs).
const CARD_TEXELS: u64 = 2048 * 2048;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GridUniform {
    levels: [Vec4; LEVELS],
    frame: u32,
    sampling: u32,
    card_words: u32,
    _pad: u32,
}

/// The light grid and the cards' accumulated local light (bindings 163 and 164 of the build,
/// lumen_light.wgsl's `update_voxels` and lumen_card_light.wgsl's `light_cards`).
pub struct LumenLightGrid {
    pub uniform: wgpu::Buffer,
    pub words: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    build: Lazy<wgpu::ComputePipeline>,
    levels: [Vec4; LEVELS],
    frame: u32,
}

impl LumenLightGrid {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "lumen_light_grid.wgsl");
        let layout = bind_group_layout(
            device,
            "lumen light grid",
            wgpu::ShaderStages::COMPUTE,
            &[(0, bind::uniform()), (1, bind::storage(true)), (163, bind::uniform()), (164, bind::storage(false))],
        );
        let words = (GRID_CELLS * CELL_WORDS) as u64 + CARD_TEXELS;
        Self {
            uniform: uniform_buffer(device, "lumen light grid", std::mem::size_of::<GridUniform>() as u64),
            words: storage_buffer(device, "lumen light grid + card local light", words * 4, wgpu::BufferUsages::empty()),
            build: compute_pipeline(device, "lumen light grid", &module, "build", &layout, &[]),
            layout,
            levels: [Vec4::ZERO; LEVELS],
            frame: 0,
        }
    }

    /// Places the clipmap around the camera.
    pub fn update(&mut self, camera: Vec3, frame: u32) {
        for (i, level) in self.levels.iter_mut().enumerate() {
            let cell = FIRST_CELL * (1u32 << i) as f32;
            let first = ((camera / cell).floor() - Vec3::splat((CELLS / 2) as f32)) * cell;
            *level = first.extend(cell);
        }
        self.frame = frame;
    }

    /// Writes the uniform and builds the cells' candidate lists (while `sampling`, with lights).
    pub fn record(&self, rec: &mut Recorder, queue: &wgpu::Queue, frame: &wgpu::Buffer, lights: &crate::lights::LightCulling, sampling: bool) {
        let uniform = GridUniform { levels: self.levels, frame: self.frame, sampling: sampling as u32, card_words: GRID_CELLS * CELL_WORDS, _pad: 0 };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        if !sampling || lights.count == 0 {
            return;
        }
        let group = bind_group(
            rec.device,
            "lumen light grid",
            &self.layout,
            &[
                (0, frame.as_entire_binding()),
                (1, lights.lights.as_entire_binding()),
                (163, self.uniform.as_entire_binding()),
                (164, self.words.as_entire_binding()),
            ],
        );
        let mut pass = rec.compute_pass("lumen light grid");
        pass.set_pipeline(&self.build);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(GRID_CELLS, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_levels_cover_the_distance_field() {
        // Level i spans CELLS cells of FIRST_CELL · 2^i: the global distance field's level
        // extents (0.25 m voxels × 128 = 32 m, doubling), so every lit voxel has a cell.
        for i in 0..LEVELS {
            let extent = FIRST_CELL * (1u32 << i) as f32 * CELLS as f32;
            assert_eq!(extent, 0.25 * 128.0 * (1u32 << i) as f32);
        }
        assert_eq!(CELL_WORDS, 66);
    }
}
