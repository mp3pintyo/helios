//! The scene editor's core, shared by the browser editor (`web_editor.rs`) and the native
//! `--scene-file` loader: a document of edits over a base scene ([`doc::SceneDoc`]), applied to
//! the live [`Scene`] and render settings, with undo / redo.
//!
//! **Document, then reconcile.** Every edit produces a new document; [`Editor::sync`] compares
//! it with the document the scene currently reflects and touches only what differs — an
//! actor's transform (and everything below it), a material's parameters, a PCG volume's
//! spawned instances. Undo and redo just swap documents (snapshots are small: they hold edits,
//! not the scene; generated PCG output lives in a cache keyed by the volume definition's hash).
//! Loading a scene file is the same operation from an empty document.
//!
//! **Actors** (`actors.rs`): the scene is a hierarchy of actors with components — mesh
//! instances, lights, volumes and particle emitters — addressed by stable keys in documents and
//! by live ids in the UI. Instance and light slots of removed components are hidden and reused,
//! and compacted away once the editor is idle (`Editor::compaction`).

pub mod actors;
pub mod content;
pub mod describe;
pub mod doc;
pub mod materials;
pub mod ops;
pub mod pcg;
pub mod settings;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use glam::{Mat4, Vec2, Vec3};
use helios_render::{InstanceHandle, Light, LightKind, Material, MaterialHandle, MeshHandle, Mobility, RenderSettings, Scene};

use self::actors::{Actor, BaseActor, BaseEmitters};
use self::content::{Content, PrefabPart};
use self::doc::*;
pub use self::materials::with_params;
use crate::scenes::outline::Outline;

/// What edits act on: the scene, the render settings and the demo's sun.
pub struct World<'a> {
    pub scene: &'a mut Scene,
    pub settings: &'a mut RenderSettings,
    /// Sun azimuth and elevation (radians) and intensity; the demo applies them every frame.
    pub sun: [&'a mut f32; 3],
    /// Ground height (the landscape PCG samples).
    pub floor: fn(f32, f32) -> f32,
}

/// Who an instance slot belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    /// Component `component` of live actor `actor`.
    Actor { actor: u32, component: u16 },
    /// Spawned by a PCG volume (its id).
    Pcg(u32),
    /// Hidden, waiting for reuse or compaction.
    Spare,
}

/// A volume's spawned instances and the output they show.
#[derive(Default)]
struct PcgState {
    hash: String,
    handles: Vec<InstanceHandle>,
    error: Option<String>,
}

pub struct Editor {
    pub doc: SceneDoc,
    applied: SceneDoc,
    undo: Vec<SceneDoc>,
    redo: Vec<SceneDoc>,
    /// The base scene's actors and their index by key.
    base_actors: Vec<BaseActor>,
    base_index: HashMap<String, usize>,
    /// Live actors by id, ids by key (never reused), the next id.
    actors: BTreeMap<u32, Actor>,
    ids: HashMap<String, u32>,
    next_live_id: u32,
    /// Per instance slot and per light slot: who has it.
    owners: Vec<Owner>,
    light_owners: Vec<Option<(u32, usize)>>,
    /// Free slots (hidden instances by mesh, disabled lights).
    spares: HashMap<MeshHandle, Vec<InstanceHandle>>,
    spare_lights: Vec<usize>,
    /// The base scene's emitters (the scene's lists are rebuilt from the live components).
    emitters: BaseEmitters,
    emitters_dirty: bool,
    /// Actors shown elsewhere than the document has them (a gizmo drag), and their world.
    preview_worlds: HashMap<u32, Mat4>,
    /// Instances made movable by a drag: their own mobility and the frames until it returns.
    interactive: HashMap<InstanceHandle, (Mobility, u32)>,
    /// Frames since the last edit or preview.
    idle_frames: u32,
    /// Compactions so far.
    pub compactions: u32,
    /// Drags move instances as they are (static ones re-render the shadow pages under them in
    /// full every frame): the behaviour before actors, for A/B measurements.
    pub static_drags: bool,
    /// What the selection outline shows.
    selected_actors: Vec<u32>,
    selected_components: Vec<(u32, String)>,
    /// Values before the first override, captured lazily (materials by name, settings by key).
    base_materials: HashMap<String, Material>,
    /// The base scene's graphs and material parameter values the document overrides.
    base_graphs: HashMap<String, helios_render::material_graph::MaterialGraph>,
    base_parameters: HashMap<String, Vec<(String, helios_render::ParameterValue)>>,
    base_settings: HashMap<String, serde_json::Value>,
    base_environment: Option<Environment>,
    pcg: BTreeMap<u32, PcgState>,
    /// Generated output by volume id (the latest per volume).
    pcg_cache: BTreeMap<u32, PcgOutputDef>,
    /// Imported glTF models' prefabs by import name, and where they came from (saved with the
    /// document; imports are not undone, their meshes stay in the scene).
    pub prefabs: BTreeMap<String, Vec<PrefabPart>>,
    pub imports: Vec<ImportDef>,
    pub content: Content,
    /// Problems met while applying the document (missing meshes, failed graphs).
    pub warnings: Vec<String>,
    /// Increments whenever the scene changes through the editor.
    pub revision: u64,
    /// Merge key of the latest commit: consecutive commits with the same key (a slider being
    /// dragged) form one undo step.
    last_merge: Option<String>,
}

const UNDO_LIMIT: usize = 200;

impl Editor {
    /// An editor over the scene as built (the base scene described by `base`, its actors by the
    /// builder's `outline`).
    pub fn new(scene: &Scene, base: BaseScene, outline: &Outline) -> Self {
        let doc = SceneDoc::new(base);
        let mut editor = Self {
            applied: doc.clone(),
            doc,
            undo: Vec::new(),
            redo: Vec::new(),
            base_actors: actors::base_actors(scene, outline),
            base_index: HashMap::new(),
            actors: BTreeMap::new(),
            ids: HashMap::new(),
            next_live_id: 1,
            owners: Vec::new(),
            light_owners: Vec::new(),
            spares: HashMap::new(),
            spare_lights: Vec::new(),
            emitters: BaseEmitters::default(),
            emitters_dirty: false,
            preview_worlds: HashMap::new(),
            interactive: HashMap::new(),
            idle_frames: 0,
            compactions: 0,
            static_drags: false,
            selected_actors: Vec::new(),
            selected_components: Vec::new(),
            base_materials: HashMap::new(),
            base_graphs: HashMap::new(),
            base_parameters: HashMap::new(),
            base_settings: HashMap::new(),
            base_environment: None,
            pcg: BTreeMap::new(),
            pcg_cache: BTreeMap::new(),
            prefabs: BTreeMap::new(),
            imports: Vec::new(),
            content: Content::default(),
            warnings: Vec::new(),
            revision: 0,
            last_merge: None,
        };
        editor.init_actors(scene);
        editor
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn owner(&self, instance: u32) -> Owner {
        self.owners.get(instance as usize).copied().unwrap_or(Owner::Spare)
    }

    /// The live actors of the base scene.
    pub fn base_actor_count(&self) -> usize {
        self.base_actors.len()
    }

    /// Free instance and light slots (waiting for reuse or compaction).
    pub fn free_slots(&self) -> (usize, usize) {
        (self.spares.values().map(Vec::len).sum(), self.spare_lights.len())
    }

    /// The live id of an added actor.
    pub fn added_actor(&self, id: u32) -> Option<u32> {
        self.actor_id(&actors::added_key(id))
    }

    pub fn pcg_instances(&self, id: u32) -> usize {
        self.pcg.get(&id).map_or(0, |s| s.handles.len())
    }

    pub fn pcg_error(&self, id: u32) -> Option<&str> {
        self.pcg.get(&id).and_then(|s| s.error.as_deref())
    }

    /// Makes `next` the document (one undo step) and applies it.
    pub fn commit(&mut self, next: SceneDoc, world: &mut World) {
        self.commit_merged(next, world, None);
    }

    /// As [`Editor::commit`]; a commit with the same `merge` key as the previous one joins its
    /// undo step instead of adding one.
    pub fn commit_merged(&mut self, next: SceneDoc, world: &mut World, merge: Option<String>) {
        if next == self.doc {
            return;
        }
        let joins = merge.is_some() && merge == self.last_merge && !self.undo.is_empty();
        let previous = std::mem::replace(&mut self.doc, next);
        if !joins {
            self.undo.push(previous);
            if self.undo.len() > UNDO_LIMIT {
                self.undo.remove(0);
            }
        }
        self.last_merge = merge;
        self.redo.clear();
        self.sync(world);
    }

    pub fn undo(&mut self, world: &mut World) -> bool {
        self.last_merge = None;
        let Some(previous) = self.undo.pop() else { return false };
        self.redo.push(std::mem::replace(&mut self.doc, previous));
        self.sync(world);
        true
    }

    pub fn redo(&mut self, world: &mut World) -> bool {
        self.last_merge = None;
        let Some(next) = self.redo.pop() else { return false };
        self.undo.push(std::mem::replace(&mut self.doc, next));
        self.sync(world);
        true
    }

    /// Replaces the document with a loaded one (no undo step): version 1 documents are migrated
    /// to actors first, and saved PCG output seeds the cache, so volumes show exactly what was
    /// saved. Its imports must have been imported already (see `DemoApp::import_model`).
    pub fn load(&mut self, mut doc: SceneDoc, world: &mut World) {
        self.warnings.clear();
        self.migrate(&mut doc, world.scene);
        let migration = std::mem::take(&mut self.warnings);
        doc.imports.clear();
        for output in std::mem::take(&mut doc.pcg_output) {
            self.pcg_cache.insert(output.id, output);
        }
        self.undo.clear();
        self.redo.clear();
        self.last_merge = None;
        self.doc = doc;
        self.sync(world);
        self.warnings.extend(migration);
    }

    /// The document to save: the edits plus the current PCG output and `camera`.
    pub fn save(&self, camera: CameraDef) -> SceneDoc {
        let mut doc = self.doc.clone();
        doc.camera = Some(camera);
        doc.imports = self.imports.clone();
        doc.pcg_output = doc.pcg.iter().filter_map(|v| self.pcg_cache.get(&v.id)).cloned().collect();
        doc
    }

    // ---- Reconciling the scene with the document ------------------------------------------------

    /// Applies every difference between the document and what the scene reflects.
    pub fn sync(&mut self, world: &mut World) {
        self.warnings.clear();
        self.sync_graphs(world);
        self.sync_materials(world);
        // A gesture's commits (gizmo drags, scrubbed fields) move instances as movable ones.
        let interactive = self.last_merge.is_some();
        self.sync_actors(world, interactive);
        self.sync_environment(world);
        self.sync_render(world);
        self.sync_pcg(world);
        self.applied = self.doc.clone();
        self.revision += 1;
        for warning in &self.warnings {
            log::warn!("scene document: {warning}");
        }
    }

    fn material_handle(&mut self, scene: &mut Scene, name: &str) -> Option<MaterialHandle> {
        let handle = self.content.material(scene, name);
        if handle.is_none() {
            self.warnings.push(format!("unknown material '{name}'"));
        }
        handle
    }

    fn mesh_handle(&mut self, scene: &mut Scene, name: &str) -> Option<MeshHandle> {
        let handle = self.content.mesh(scene, name);
        if handle.is_none() {
            self.warnings.push(format!("unknown mesh '{name}'"));
        }
        handle
    }

    fn sync_materials(&mut self, world: &mut World) {
        let names: BTreeSet<String> = self.doc.materials.keys().chain(self.applied.materials.keys()).cloned().collect();
        for name in names {
            if self.doc.materials.get(&name) == self.applied.materials.get(&name) {
                continue;
            }
            let Some(handle) = self.material_handle(world.scene, &name) else { continue };
            let base = *self.base_materials.entry(name.clone()).or_insert_with(|| *world.scene.material(handle));
            let params = self.doc.materials.get(&name).cloned().unwrap_or_default();
            self.apply_material(world.scene, &name, handle, base, &params);
        }
    }

    /// A free instance of `mesh` to reuse, or a new one (PCG output).
    fn instance_for(&mut self, scene: &mut Scene, mesh: MeshHandle, material: MaterialHandle, owner: Owner) -> InstanceHandle {
        let handle = match self.spares.get_mut(&mesh).and_then(Vec::pop) {
            Some(handle) => handle,
            None => scene.add_instance(mesh, material, Mat4::IDENTITY),
        };
        if !self.interactive.contains_key(&handle) {
            scene.set_mobility(handle, Mobility::Static);
        }
        let index = handle.index() as usize;
        if self.owners.len() <= index {
            self.owners.resize(index + 1, Owner::Spare);
        }
        self.owners[index] = owner;
        handle
    }

    fn place(&mut self, scene: &mut Scene, handle: InstanceHandle, material: MaterialHandle, transform: Mat4, cast_shadow: bool) {
        scene.set_transform(handle, transform);
        scene.set_instance_material(handle, material);
        if scene.casts_shadow(handle) != cast_shadow {
            scene.set_cast_shadow(handle, cast_shadow);
        }
        scene.set_visible(handle, true);
    }

    /// The environment as the scene has it now.
    pub fn current_environment(world: &World) -> Environment {
        let scene = &*world.scene;
        let (fog, clouds, wind) = (&scene.fog, &scene.clouds, &scene.wind);
        Environment {
            sun_azimuth_deg: world.sun[0].to_degrees(),
            sun_elevation_deg: world.sun[1].to_degrees(),
            sun_intensity: *world.sun[2],
            fog: FogDef {
                density: fog.density,
                height_falloff: fog.height_falloff,
                base_height: fog.base_height,
                albedo: fog.albedo.into(),
                anisotropy: fog.anisotropy,
                ambient: fog.ambient,
                noise: fog.noise,
                distance: fog.distance,
            },
            clouds: CloudsDef {
                coverage: clouds.coverage,
                density: clouds.density,
                base_altitude: clouds.base_altitude,
                thickness: clouds.thickness,
                wind: clouds.wind.into(),
                shadow_strength: clouds.shadow_strength,
            },
            wind: WindDef {
                direction: wind.direction.into(),
                strength: wind.strength,
                gust_frequency: wind.gust_frequency,
                gust_wavelength: wind.gust_wavelength,
                turbulence: wind.turbulence,
            },
        }
    }

    fn sync_environment(&mut self, world: &mut World) {
        if self.doc.environment == self.applied.environment {
            return;
        }
        let base = self.base_environment.get_or_insert_with(|| Self::current_environment(world)).clone();
        let env = self.doc.environment.clone().unwrap_or(base);
        *world.sun[0] = env.sun_azimuth_deg.to_radians();
        *world.sun[1] = env.sun_elevation_deg.to_radians();
        *world.sun[2] = env.sun_intensity;
        let scene = &mut *world.scene;
        let f = &env.fog;
        scene.fog = helios_render::FogVolume {
            density: f.density,
            height_falloff: f.height_falloff,
            base_height: f.base_height,
            albedo: f.albedo.into(),
            anisotropy: f.anisotropy,
            ambient: f.ambient,
            noise: f.noise,
            wind: scene.fog.wind,
            distance: f.distance,
        };
        let c = &env.clouds;
        scene.clouds = helios_render::CloudLayer {
            coverage: c.coverage,
            density: c.density,
            base_altitude: c.base_altitude,
            thickness: c.thickness,
            wind: Vec2::from(c.wind),
            shadow_strength: c.shadow_strength,
        };
        let w = &env.wind;
        scene.wind = helios_render::Wind {
            direction: Vec2::from(w.direction),
            strength: w.strength,
            gust_frequency: w.gust_frequency,
            gust_wavelength: w.gust_wavelength,
            turbulence: w.turbulence,
            ..scene.wind
        };
    }

    fn sync_render(&mut self, world: &mut World) {
        let keys: BTreeSet<String> = self.doc.render.keys().chain(self.applied.render.keys()).cloned().collect();
        for key in keys {
            let value = self.doc.render.get(&key);
            if value == self.applied.render.get(&key) {
                continue;
            }
            let Some(current) = settings::get(world.settings, &key) else {
                self.warnings.push(format!("unknown render setting '{key}'"));
                continue;
            };
            let base = self.base_settings.entry(key.clone()).or_insert(current).clone();
            let _ = settings::set(world.settings, &key, value.unwrap_or(&base));
        }
    }

    fn sync_pcg(&mut self, world: &mut World) {
        let wanted: BTreeMap<u32, PcgVolumeDef> = self.doc.pcg.iter().map(|v| (v.id, v.clone())).collect();
        let removed: Vec<u32> = self.pcg.keys().filter(|id| !wanted.contains_key(id)).copied().collect();
        for id in removed {
            let state = self.pcg.remove(&id).unwrap_or_default();
            for handle in state.handles {
                self.retire(world.scene, handle);
            }
        }
        for (id, def) in wanted {
            let hash = pcg::definition_hash(&def);
            if self.pcg.get(&id).is_some_and(|s| s.hash == hash) {
                continue;
            }
            let output = match self.pcg_cache.get(&id).filter(|o| o.hash == hash) {
                Some(output) => Ok(output.clone()),
                None => pcg::generate(&def, world.scene, world.floor, self.surface_instance(world.scene, &def.surface)),
            };
            let mut state = self.pcg.remove(&id).unwrap_or_default();
            state.hash = hash;
            match output {
                Ok(output) => {
                    state.error = None;
                    self.spawn(world.scene, id, &output, &mut state.handles);
                    self.pcg_cache.insert(id, output);
                }
                Err(e) => {
                    self.warnings.push(format!("PCG volume '{}': {e}", def.name));
                    state.error = Some(e);
                }
            }
            self.pcg.insert(id, state);
        }
    }

    /// Shows `output` with the volume's instances, reusing them per mesh.
    fn spawn(&mut self, scene: &mut Scene, volume: u32, output: &PcgOutputDef, handles: &mut Vec<InstanceHandle>) {
        let mut by_mesh: HashMap<MeshHandle, Vec<InstanceHandle>> = HashMap::new();
        for &h in handles.iter().rev() {
            by_mesh.entry(scene.instance_mesh(h)).or_default().push(h);
        }
        let parts: Vec<Option<(MeshHandle, MaterialHandle)>> = output
            .parts
            .iter()
            .map(|[mesh, material]| Some((self.mesh_handle(scene, mesh)?, self.material_handle(scene, material)?)))
            .collect();
        let mut used = Vec::with_capacity(output.instances.len());
        for (part, packed) in &output.instances {
            let Some((mesh, material)) = parts.get(*part as usize).copied().flatten() else { continue };
            let handle = match by_mesh.get_mut(&mesh).and_then(Vec::pop) {
                Some(h) => h,
                None => self.instance_for(scene, mesh, material, Owner::Pcg(volume)),
            };
            self.place(scene, handle, material, pcg::unpack(packed), true);
            used.push(handle);
        }
        for handle in by_mesh.into_values().flatten() {
            self.retire(scene, handle);
        }
        *handles = used;
    }

    // ---- Content ----------------------------------------------------------------------------

    /// The parts of a placeable item: an engine prefab, an imported model or a scene mesh
    /// (`mesh:NAME`, with the material its first instance uses).
    pub fn prefab(&self, scene: &Scene, item: &str) -> Option<Vec<PrefabPart>> {
        if let Some(parts) = content::prefab(item).or_else(|| self.prefabs.get(item).cloned()) {
            return Some(parts);
        }
        let mesh = scene.find_mesh(item.strip_prefix("mesh:")?)?;
        let material = scene
            .instance_handles()
            .find(|&h| scene.instance_mesh(h) == mesh)
            .map(|h| scene.material_name(scene.instance_material(h)).to_string())
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| "engine/default".into());
        Some(vec![PrefabPart::new(scene.mesh_name(mesh), &material)])
    }
}

pub fn light_def(l: &Light) -> LightDef {
    let (shape, direction, inner, outer) = match l.kind {
        LightKind::Point => (LightShape::Point, [0.0, -1.0, 0.0], 30.0, 40.0),
        LightKind::Spot { direction, inner, outer } => (LightShape::Spot, direction.into(), inner.to_degrees(), outer.to_degrees()),
    };
    LightDef {
        shape,
        position: l.position.into(),
        direction,
        color: l.color.into(),
        intensity: l.intensity,
        range: l.range,
        source_radius: l.source_radius,
        cast_shadows: l.cast_shadows,
        inner_deg: inner,
        outer_deg: outer,
        enabled: true,
    }
}

/// A light from its definition; a disabled light emits nothing, reaches nowhere and casts no
/// shadow (it keeps its slot).
pub fn light_from_def(d: &LightDef) -> Light {
    let kind = match d.shape {
        LightShape::Point => LightKind::Point,
        LightShape::Spot => LightKind::Spot {
            direction: Vec3::from(d.direction).normalize_or(Vec3::NEG_Y),
            inner: d.inner_deg.min(d.outer_deg).to_radians(),
            outer: d.outer_deg.clamp(1.0, 89.0).to_radians(),
        },
    };
    Light {
        kind,
        position: d.position.into(),
        color: d.color.into(),
        intensity: if d.enabled { d.intensity.max(0.0) } else { 0.0 },
        range: if d.enabled { d.range.max(0.01) } else { 0.001 },
        source_radius: d.source_radius.max(0.0),
        cast_shadows: d.enabled && d.cast_shadows,
    }
}

#[cfg(test)]
mod tests;
