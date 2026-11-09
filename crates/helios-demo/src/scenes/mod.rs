//! Demo scenes. Each scene is built procedurally (geometry, textures, materials, lights) or
//! from an imported glTF model, and baked into virtual meshes at startup.

pub mod candles;
pub mod cathedral;
pub mod forest;
pub mod model;
pub mod skeletal;
pub mod valley;

use helios_geometry::{build_virtual_mesh, BuildConfig, Mesh, VirtualMesh};
use helios_render::{Camera, MeshHandle, Scene};
use web_time::Instant;

use crate::mesh_store::{self, MeshStore};

/// Geometry density preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    /// Fast to bake (browser default).
    Medium,
    /// Millions of triangles per asset.
    High,
}

impl Detail {
    /// Picks `medium` or `high`.
    pub fn pick<T>(self, medium: T, high: T) -> T {
        match self {
            Detail::Medium => medium,
            Detail::High => high,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneKind {
    Valley,
    Cathedral,
    Forest,
    /// The imported glTF model (only available when one was loaded).
    Model,
}

impl SceneKind {
    pub const ALL: [SceneKind; 4] = [SceneKind::Valley, SceneKind::Cathedral, SceneKind::Forest, SceneKind::Model];

    pub fn name(self) -> &'static str {
        match self {
            SceneKind::Valley => "valley",
            SceneKind::Cathedral => "cathedral",
            SceneKind::Forest => "forest",
            SceneKind::Model => "model",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name().eq_ignore_ascii_case(name))
    }

    /// The next scene in the cycle; the model scene takes part only if a model is loaded.
    pub fn next(self, have_model: bool) -> Self {
        let next = Self::ALL[(self as usize + 1) % Self::ALL.len()];
        if next == SceneKind::Model && !have_model {
            next.next(have_model)
        } else {
            next
        }
    }
}

/// A built scene plus how to view it.
pub struct SceneSetup {
    pub scene: Scene,
    pub camera: Camera,
    /// Sun azimuth / elevation in radians and intensity.
    pub sun: (f32, f32, f32),
    /// Lowest allowed camera height at (x, z).
    pub floor: fn(f32, f32) -> f32,
    /// Skeletal animation of an imported model's skinned meshes.
    pub animation: Option<skeletal::SkeletalAnimation>,
    /// Props that move every frame.
    pub motion: crate::motion::DemoMotion,
}

/// Builds a scene. `SceneKind::Model` needs `model`; without one it falls back to the valley.
pub fn build(kind: SceneKind, detail: Detail, model: Option<&model::Model>) -> SceneSetup {
    build_with(kind, detail, model, MeshStore::Bake).0
}

/// Builds a scene whose procedural meshes come from `store` (baked, baked and recorded, or
/// streamed from pre-baked files); returns the store afterwards. Imported models always bake
/// their meshes (see [`mesh_store`]).
pub fn build_with(kind: SceneKind, detail: Detail, model: Option<&model::Model>, store: MeshStore) -> (SceneSetup, MeshStore) {
    let start = Instant::now();
    let (setup, store) = mesh_store::scoped(store, || match (kind, model) {
        (SceneKind::Valley, _) => valley::build(detail),
        (SceneKind::Cathedral, _) => cathedral::build(detail),
        (SceneKind::Forest, _) => forest::build(detail),
        (SceneKind::Model, Some(m)) => model::build(m),
        (SceneKind::Model, None) => {
            log::warn!("no model loaded (use --gltf FILE or ?gltf=URL); showing the valley");
            valley::build(detail)
        }
    });
    let mut setup = setup;
    name_materials(&mut setup.scene);
    log::info!(
        "scene '{}': {} instances, {:.1} M source triangles, {} lights, {:.1} MiB geometry in RAM (built in {:.1} s)",
        kind.name(),
        setup.scene.instance_count(),
        setup.scene.source_triangle_count() as f64 / 1e6,
        setup.scene.lights().len(),
        setup.scene.geometry_cpu_bytes() as f64 / 1048576.0,
        start.elapsed().as_secs_f64()
    );
    (setup, store)
}

/// Names a mesh `name`, or `name #2`, `name #3`… if the name is taken: scene files refer to
/// meshes by these names (stable as long as the builder adds meshes in the same order).
pub(crate) fn name_mesh(scene: &mut Scene, mesh: MeshHandle, name: &str) {
    let unique = (1..).map(|n| if n == 1 { name.to_string() } else { format!("{name} #{n}") }).find(|n| scene.find_mesh(n).is_none()).unwrap();
    scene.set_mesh_name(mesh, &unique);
}

/// Names the materials a builder left unnamed after their shading model, numbered per model in
/// build order (`rock 1`, `rock 2`, `glass 1`…).
fn name_materials(scene: &mut Scene) {
    let mut counts: std::collections::HashMap<String, u32> = Default::default();
    for i in 0..scene.material_count() as u32 {
        let handle = scene.material_handle(i).expect("index in range");
        let kind = format!("{:?}", scene.material(handle).kind).to_lowercase();
        let n = counts.entry(kind.clone()).or_insert(0);
        *n += 1;
        if scene.material_name(handle).is_empty() {
            let name = (*n..).map(|k| format!("{kind} {k}")).find(|candidate| scene.find_material(candidate).is_none()).unwrap();
            scene.set_material_name(handle, &name);
        }
    }
}

/// Bakes `mesh` and adds it to `scene` (or takes it from the active [`MeshStore`]).
pub(crate) fn add_baked(scene: &mut Scene, name: &str, mesh: &Mesh) -> MeshHandle {
    add_baked_with(scene, name, mesh, BuildConfig::default())
}

pub(crate) fn add_baked_with(scene: &mut Scene, name: &str, mesh: &Mesh, config: BuildConfig) -> MeshHandle {
    mesh_store::add(scene, name, || bake_with(name, mesh, config))
}

/// Adds the mesh `build` bakes (or the active store's) as `name`.
pub(crate) fn add_built(scene: &mut Scene, name: &str, build: impl FnOnce() -> VirtualMesh) -> MeshHandle {
    mesh_store::add(scene, name, || logged(name, build))
}

fn bake_with(name: &str, mesh: &Mesh, config: BuildConfig) -> VirtualMesh {
    logged(name, || build_virtual_mesh(mesh, &config))
}

/// Bakes with `build`, logging the result and the time it took.
fn logged(name: &str, build: impl FnOnce() -> VirtualMesh) -> VirtualMesh {
    let start = Instant::now();
    let vm = build();
    let voxels: u32 = vm.stats.voxels_per_level.iter().sum();
    log::info!(
        "baked {name}: {} triangles -> {} clusters, {} pages, {} levels{}, {:.2} MiB ({:.0} ms)",
        vm.triangle_count(),
        vm.clusters.len(),
        vm.pages.len(),
        vm.stats.level_count(),
        if voxels > 0 { format!(", {voxels} voxels") } else { String::new() },
        vm.cpu_bytes() as f64 / 1048576.0,
        start.elapsed().as_secs_f64() * 1000.0
    );
    vm
}

/// Maps `f` over `items` on all cores natively (work-stealing over an atomic index, so uneven
/// items such as meshes of very different sizes balance), sequentially on wasm.
pub(crate) fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(items.len());
        if threads > 1 {
            let next = AtomicUsize::new(0);
            let mut slots: Vec<Option<R>> = std::iter::repeat_with(|| None).take(items.len()).collect();
            std::thread::scope(|s| {
                let workers: Vec<_> = (0..threads)
                    .map(|_| {
                        s.spawn(|| {
                            let mut done = Vec::new();
                            loop {
                                let i = next.fetch_add(1, Ordering::Relaxed);
                                let Some(item) = items.get(i) else { break };
                                done.push((i, f(item)));
                            }
                            done
                        })
                    })
                    .collect();
                for worker in workers {
                    for (i, r) in worker.join().expect("worker panicked") {
                        slots[i] = Some(r);
                    }
                }
            });
            return slots.into_iter().map(|r| r.expect("every item mapped")).collect();
        }
    }
    items.iter().map(f).collect()
}

/// Deterministic pseudo-random sequence for scattering.
pub(crate) struct Rng(pub u32);

impl Rng {
    pub fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
