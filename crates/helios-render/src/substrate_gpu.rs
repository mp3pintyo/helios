//! Extended materials on the GPU: Substrate slabs and material graphs.
//!
//! Every frame [`ExtendedMaterials::sync`] decides, from the scene's materials, which need an
//! extended resolve pipeline — a slab with features (while [`crate::RenderSettings::substrate`]
//! is on) or a graph that compiled — and gives each distinct (shading model, graph) pair one of
//! the resolve slots 8–23 (`passes/material.rs`): material classification then bins their tiles
//! apart, and plain materials keep the plain pipelines and cost. It writes
//!
//! * the **slab table** (one `GpuSlab` per material; uniform, 28 KiB),
//! * the **graph parameter block**: each graph material's parameter values (its instance
//!   overrides, else the graph's defaults), one `vec4` per parameter (uniform, 16 KiB),
//! * the materials' resolve slots, patched into the material uniform after `GpuScene` uploads it,
//! * the graph source of materials whose graph drives their displacement (graph id and parameter
//!   offset), patched into the tessellation uniform likewise: such a material displaces by its
//!   graph only while the graph has a resolve pipeline, so the tessellator and the resolve agree;
//!   with it the speed bound of an animated displacement for the instance's parameter values,
//!
//! and owns the G-buffer's slab texels (rgba32uint, allocated only while some material is
//! extended) and the subsurface light targets (only while some slab scatters beneath its
//! surface). The lighting passes switch to their slab variants only while some material has
//! slab features.

use std::collections::BTreeSet;

use wgpu::BindingResource::TextureView;
use wgpu::{ShaderStages, TextureFormat};

use crate::gpu_scene::{GpuScene, MAX_MATERIALS};
use crate::gpu_types::GpuMaterial;
use crate::material_graphs::ParameterValue;
use crate::passes::material::{EXTENDED_SLOTS, PLAIN_SLOTS, SKY_SLOT};
use crate::scene::{MaterialHandle, Scene};
use crate::substrate::{features, GpuSlab};
use crate::tessellation::graphs::graph_id;
use crate::tessellation::{displacement_source, displacement_speed, write_graph_motion, write_graph_source, DisplacementSource};
use crate::util::{bind, bind_group, bind_group_layout, texture, uniform_buffer, TextureSpec};
use helios_material::{output_bit, CompiledGraph, ParameterKind};

/// vec4 slots of the graph parameter block (`GRAPH_PARAMETER_SLOTS` in material_graph.wgsl).
pub const GRAPH_PARAMETER_SLOTS: usize = 1024;
/// An unset texture parameter samples the material's own texture (`GRAPH_MATERIAL_TEXTURE`).
const MATERIAL_TEXTURE: u32 = 0xffff_fffe;

pub const SUBSTRATE_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;
pub const SUBSURFACE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// An extended resolve pipeline in use: its slot, shading model and graph.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedSlot {
    pub slot: u32,
    pub kind: u32,
    /// Hash and code of the graph (`None`: slab materials without one).
    pub graph: Option<(u64, String)>,
    /// The graph's code drives its materials' displacement (`graph_displacement`).
    pub displaces: bool,
}

struct Targets {
    capacity: (u32, u32),
    substrate: wgpu::TextureView,
    /// Subsurface light (written by the lighting) and its horizontal diffusion.
    subsurface: Option<[wgpu::TextureView; 2]>,
}

pub struct ExtendedMaterials {
    slabs: wgpu::Buffer,
    params: wgpu::Buffer,
    /// Group 1 of the extended resolve pipelines, of the lighting's slab variant and of
    /// MegaLights' slab shading.
    pub resolve_layout: wgpu::BindGroupLayout,
    pub lighting_layout: wgpu::BindGroupLayout,
    pub megalights_layout: wgpu::BindGroupLayout,
    /// Bound where a subsurface target is not needed.
    dummy_storage: wgpu::TextureView,
    targets: Option<Targets>,
    pub slots: Vec<ExtendedSlot>,
    /// Some material has slab features: lighting runs its slab variant.
    pub lighting: bool,
    /// Some slab scatters beneath its surface: the subsurface pass runs.
    pub subsurface: bool,
    /// Materials whose resolve slot was patched (to restore when they turn plain).
    patched: BTreeSet<usize>,
    /// Materials whose displacement graph source was patched (to clear when they lose it).
    graph_displaced: BTreeSet<usize>,
    /// Hash of what graph-driven displacement depends on here — each such material's graph code
    /// and parameter values: a change reshapes the surfaces (cached shadow pages are stale).
    pub displacement_key: u64,
    /// Scene versions, material uploads and the setting the current state was built from.
    key: Option<(u64, u64, u64, u64, bool)>,
    /// The device can bind what the extended pipelines need (a fifth storage texture in the
    /// resolve, a 17th sampled texture in the lighting); otherwise every material renders plain.
    supported: bool,
    warned: bool,
}

impl ExtendedMaterials {
    pub fn new(device: &wgpu::Device) -> Self {
        let c = ShaderStages::COMPUTE;
        Self {
            slabs: uniform_buffer(device, "material slabs", (MAX_MATERIALS * std::mem::size_of::<GpuSlab>()) as u64),
            params: uniform_buffer(device, "graph parameters", (GRAPH_PARAMETER_SLOTS * 16) as u64),
            resolve_layout: bind_group_layout(
                device,
                "extended resolve",
                c,
                &[(0, bind::uniform()), (1, bind::uniform()), (2, bind::storage_texture(SUBSTRATE_FORMAT))],
            ),
            lighting_layout: bind_group_layout(
                device,
                "substrate lighting",
                c,
                &[(0, bind::uniform()), (1, bind::texture_uint()), (2, bind::storage_texture(SUBSURFACE_FORMAT))],
            ),
            megalights_layout: bind_group_layout(device, "substrate megalights", c, &[(0, bind::uniform()), (1, bind::texture_uint())]),
            dummy_storage: texture(
                device,
                TextureSpec { label: "subsurface (unused)", width: 1, height: 1, layers: 1, mips: 1, format: SUBSURFACE_FORMAT, usage: wgpu::TextureUsages::STORAGE_BINDING },
            )
            .create_view(&Default::default()),
            targets: None,
            slots: Vec::new(),
            lighting: false,
            subsurface: false,
            patched: BTreeSet::new(),
            graph_displaced: BTreeSet::new(),
            displacement_key: 0,
            key: None,
            // The slab lighting samples one texture more than the lighting pass.
            supported: device.limits().max_storage_textures_per_shader_stage >= 5 && device.limits().max_sampled_textures_per_shader_stage > crate::context::MIN_SAMPLED_TEXTURES,
            warned: false,
        }
    }

    /// Rebuilds the slab table, the parameter block and the slot assignment when materials,
    /// graphs, parameters or the setting changed, and sizes the targets for `capacity`.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, gpu: &GpuScene, enabled: bool, capacity: (u32, u32)) {
        let key = (scene.material_version, scene.graphs.version, scene.graphs.parameter_version, gpu.material_uploads, enabled);
        if self.key != Some(key) {
            self.key = Some(key);
            self.rebuild(queue, scene, gpu, enabled);
        }
        if self.slots.is_empty() {
            self.targets = None;
            return;
        }
        if self.targets.as_ref().is_none_or(|t| t.capacity != capacity || t.subsurface.is_some() != self.subsurface) {
            let make = |label: &str, format| {
                let usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
                texture(device, TextureSpec { label, width: capacity.0, height: capacity.1, layers: 1, mips: 1, format, usage }).create_view(&Default::default())
            };
            self.targets = Some(Targets {
                capacity,
                substrate: make("gbuffer substrate", SUBSTRATE_FORMAT),
                subsurface: self.subsurface.then(|| [make("subsurface light", SUBSURFACE_FORMAT), make("subsurface diffusion", SUBSURFACE_FORMAT)]),
            });
        }
    }

    fn rebuild(&mut self, queue: &wgpu::Queue, scene: &Scene, gpu: &GpuScene, enabled: bool) {
        let mut slabs = vec![GpuSlab::default(); MAX_MATERIALS];
        let mut params: Vec<[u32; 4]> = Vec::new();
        let mut slots: Vec<ExtendedSlot> = Vec::new();
        let mut assigned: Vec<(usize, u32)> = Vec::new();
        // (material, graph id, parameter offset) of graph-displaced materials.
        let mut graph_sources: Vec<(usize, u32, u32)> = Vec::new();
        let mut displacement_key = Fnv::default();
        let (mut lighting, mut subsurface) = (false, false);
        if !self.supported && scene.materials.iter().any(|m| m.graph.is_some() || m.slab.features() != 0) {
            self.warn("the device cannot bind the extended material pipelines: slabs and graphs are ignored");
        }
        for (i, m) in scene.materials.iter().enumerate().take(if self.supported { MAX_MATERIALS } else { 0 }) {
            if m.kind.is_translucent() {
                continue;
            }
            let mut graph = m.graph.and_then(|g| scene.compiled_material_graph(g));
            if graph.is_some_and(|g| params.len() + g.parameters.len() > GRAPH_PARAMETER_SLOTS) {
                let name = m.graph.map_or("", |g| scene.material_graph(g).name.as_str());
                self.warn(&format!("graph '{name}' does not fit the parameter block; its materials render without it"));
                graph = None;
            }
            let offset = params.len() as u32;
            if let Some(g) = graph {
                params.extend(g.parameters.iter().map(|p| parameter_value(p, scene.material_parameter(MaterialHandle(i as u32), &p.name))));
            }
            let own = if enabled { m.slab.features() } else { 0 };
            slabs[i] = GpuSlab::new(&m.slab, own, offset);
            let all = own | if enabled { graph.map_or(0, graph_features) } else { 0 };
            if graph.is_none() && own == 0 {
                continue;
            }
            let kind = (m.kind as u32).min(SKY_SLOT);
            let hash = graph.map(|g| g.hash);
            let slot = match slots.iter().position(|s| s.kind == kind && s.graph.as_ref().map(|g| g.0) == hash) {
                Some(slot) => slot,
                None if slots.len() < EXTENDED_SLOTS as usize => {
                    slots.push(ExtendedSlot {
                        slot: PLAIN_SLOTS + slots.len() as u32,
                        kind,
                        graph: graph.map(|g| (g.hash, g.wgsl.clone())),
                        displaces: graph.is_some_and(|g| g.displacement.is_some()),
                    });
                    slots.len() - 1
                }
                None => {
                    self.warn(&format!("more than {EXTENDED_SLOTS} distinct (shading model, graph) pairs: the rest render as plain materials"));
                    continue;
                }
            };
            assigned.push((i, slots[slot].slot));
            if let (Some(DisplacementSource::Graph(handle)), Some(g)) = (displacement_source(scene, m), graph) {
                graph_sources.push((i, graph_id(handle), offset));
                displacement_key.write(&(i as u64).to_le_bytes());
                displacement_key.write(&g.displacement_hash().unwrap_or(0).to_le_bytes());
                displacement_key.write(bytemuck::cast_slice(&params[offset as usize..]));
            }
            lighting |= all != 0;
            subsurface |= all & features::SUBSURFACE != 0;
        }
        queue.write_buffer(&self.slabs, 0, bytemuck::cast_slice(&slabs));
        if !params.is_empty() {
            queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&params));
        }
        // Resolve slots: extended materials point at theirs, materials that turned plain get
        // their shading model back (the upload wrote that; patching is idempotent).
        let offset = std::mem::offset_of!(GpuMaterial, resolve) as u64;
        let stride = std::mem::size_of::<GpuMaterial>() as u64;
        let now: BTreeSet<usize> = assigned.iter().map(|&(i, _)| i).collect();
        for &i in self.patched.difference(&now) {
            if let Some(m) = scene.materials.get(i) {
                queue.write_buffer(&gpu.materials, i as u64 * stride + offset, bytemuck::bytes_of(&(m.kind as u32).min(SKY_SLOT)));
            }
        }
        for &(i, slot) in &assigned {
            queue.write_buffer(&gpu.materials, i as u64 * stride + offset, bytemuck::bytes_of(&slot));
        }
        self.patched = now;
        let now: BTreeSet<usize> = graph_sources.iter().map(|s| s.0).collect();
        for &i in self.graph_displaced.difference(&now) {
            write_graph_source(queue, gpu, i, 0, 0);
            write_graph_motion(queue, gpu, i, Some(0.0));
        }
        // The speed bound follows the instance's parameters (a faster clock, a larger amplitude).
        for &(i, id, offset) in &graph_sources {
            write_graph_source(queue, gpu, i, id, offset);
            write_graph_motion(queue, gpu, i, displacement_speed(scene, MaterialHandle(i as u32)));
        }
        self.graph_displaced = now;
        self.displacement_key = displacement_key.0;
        self.slots = slots;
        self.lighting = lighting;
        self.subsurface = subsurface;
    }

    /// The graph parameter block (the tessellation's displacing graphs read it too).
    pub fn parameter_buffer(&self) -> &wgpu::Buffer {
        &self.params
    }

    /// The slab table (Lumen's card capture reads the graphs' parameter offsets from it).
    pub fn slab_buffer(&self) -> &wgpu::Buffer {
        &self.slabs
    }

    fn warn(&mut self, message: &str) {
        if !self.warned {
            log::warn!("{message}");
            self.warned = true;
        }
    }

    /// Group 1 of the extended resolve pipelines (`None` without extended materials).
    pub fn resolve_group(&self, device: &wgpu::Device) -> Option<wgpu::BindGroup> {
        let t = self.targets.as_ref()?;
        Some(bind_group(
            device,
            "extended resolve",
            &self.resolve_layout,
            &[(0, self.slabs.as_entire_binding()), (1, self.params.as_entire_binding()), (2, TextureView(&t.substrate))],
        ))
    }

    /// Group 1 of the lighting's slab variant (`None` while it is not needed).
    pub fn lighting_group(&self, device: &wgpu::Device) -> Option<wgpu::BindGroup> {
        let t = self.targets.as_ref().filter(|_| self.lighting)?;
        let subsurface = t.subsurface.as_ref().map_or(&self.dummy_storage, |s| &s[0]);
        Some(bind_group(
            device,
            "substrate lighting",
            &self.lighting_layout,
            &[(0, self.slabs.as_entire_binding()), (1, TextureView(&t.substrate)), (2, TextureView(subsurface))],
        ))
    }

    /// Group 1 of MegaLights' slab shading.
    pub fn megalights_group(&self, device: &wgpu::Device) -> Option<wgpu::BindGroup> {
        let t = self.targets.as_ref().filter(|_| self.lighting)?;
        Some(bind_group(device, "substrate megalights", &self.megalights_layout, &[(0, self.slabs.as_entire_binding()), (1, TextureView(&t.substrate))]))
    }

    /// What the subsurface pass reads and writes: the slab table, the slab texels, the
    /// subsurface light and its horizontal diffusion.
    pub fn subsurface_inputs(&self) -> Option<(&wgpu::Buffer, &wgpu::TextureView, &[wgpu::TextureView; 2])> {
        let t = self.targets.as_ref()?;
        Some((&self.slabs, &t.substrate, t.subsurface.as_ref()?))
    }
}

/// FNV-1a over written bytes.
struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self(0xcbf29ce484222325)
    }
}

impl Fnv {
    fn write(&mut self, bytes: &[u8]) {
        self.0 = bytes.iter().fold(self.0, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    }
}

/// Slab features a graph adds through its connected outputs.
fn graph_features(graph: &CompiledGraph) -> u32 {
    [
        ("Specular", features::SPECULAR),
        ("Clearcoat", features::CLEARCOAT),
        ("SheenColor", features::SHEEN),
        ("Subsurface", features::SUBSURFACE),
        ("Transmission", features::TRANSMISSION),
        ("Anisotropy", features::ANISOTROPY),
    ]
    .iter()
    .filter(|(pin, _)| graph.outputs & output_bit(pin) != 0)
    .fold(0, |acc, (_, bit)| acc | bit)
}

/// A parameter's slot in the block: the material's override if it has the parameter's kind,
/// else the graph's default (texture parameters: the material's own texture).
fn parameter_value(p: &helios_material::Parameter, value: Option<ParameterValue>) -> [u32; 4] {
    let bits = |v: [f32; 4]| v.map(f32::to_bits);
    match (p.kind, value) {
        (ParameterKind::Scalar, Some(ParameterValue::Scalar(x))) => bits([x, 0.0, 0.0, 0.0]),
        (ParameterKind::Vector, Some(ParameterValue::Vector(v))) => bits(v),
        (ParameterKind::Texture, Some(ParameterValue::Texture(t))) => [t.0, 0, 0, 0],
        (ParameterKind::Texture, _) => [MATERIAL_TEXTURE, 0, 0, 0],
        _ => bits(p.default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TextureHandle;

    #[test]
    fn parameters_take_overrides_of_their_kind() {
        let scalar = helios_material::Parameter { name: "a".into(), kind: ParameterKind::Scalar, default: [0.5, 0.0, 0.0, 0.0], range: [0.0, 1.0] };
        assert_eq!(parameter_value(&scalar, None), [0.5f32.to_bits(), 0, 0, 0]);
        assert_eq!(parameter_value(&scalar, Some(ParameterValue::Scalar(2.0))), [2.0f32.to_bits(), 0, 0, 0]);
        assert_eq!(parameter_value(&scalar, Some(ParameterValue::Vector([1.0; 4]))), [0.5f32.to_bits(), 0, 0, 0], "wrong kind: default");
        let texture = helios_material::Parameter { kind: ParameterKind::Texture, ..scalar };
        assert_eq!(parameter_value(&texture, None)[0], MATERIAL_TEXTURE);
        assert_eq!(parameter_value(&texture, Some(ParameterValue::Texture(TextureHandle(7))))[0], 7);
    }

    #[test]
    fn graphs_add_the_features_of_their_outputs() {
        let lava = helios_material::compile(&helios_material::presets::lava()).unwrap();
        assert_eq!(graph_features(&lava), 0);
        let water = helios_material::compile(&helios_material::presets::stylized_water()).unwrap();
        assert_eq!(graph_features(&water), features::SPECULAR);
    }
}
