//! Actors (UE's actors and components): named nodes with a transform relative to an optional
//! parent and components — mesh instances with their materials, lights, volumes and particle
//! emitters. A component's world transform is its actor's world transform times its own
//! relative one, and an actor's world transform is its parent's times its own, so moving an
//! actor moves its components and its children: instances through the per-instance path of the
//! dynamic world, lights and emitters with them.
//!
//! **Base actors** come from the scene builders' outline (`scenes::outline`): a tree is one actor
//! with trunk and foliage components, the campfire one with stones, logs, coals, light, fire
//! volume, sparks and embers. Whatever the builder left out becomes an actor of its own.
//!
//! **Live actors** are what the document asks for now: [`Editor::sync_actors`] resolves the
//! actors whose definition changed (base definition plus overrides, or an added actor), keeps
//! the instance and light slots of components that stay, takes slots for new ones from the free
//! lists (or adds them) and retires the slots of removed ones (hidden, onto the free lists),
//! then places the changed actors and everything below them.
//!
//! **Interactive moves** (gizmo previews and the commits of a gesture) make the moved instances
//! [`Mobility::Movable`] — dynamic shadow casters over the cached static shadow pages — and give
//! them their previous mobility back once the drag has settled ([`Editor::tick`]).
//!
//! **Compaction.** Retired slots wait on the free lists for the next component of the same mesh;
//! once the editor has been idle for a while with enough of them, [`Editor::compaction`] names
//! them for `Scene::compact`, and [`Editor::compacted`] maps every handle it holds through the
//! result. Documents never hold slots, so undo after a compaction just takes new slots.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use glam::{Mat4, Vec3};
use helios_render::{Compaction, InstanceHandle, LightHandle, LightKind, MaterialHandle, MeshHandle, Mobility, ParticleEmitter, Scene, VolumeEmitter};

use super::doc::*;
use super::{light_def, light_from_def, Editor, Owner, World};
use crate::scenes::outline::{position_key, Outline};

/// Frames a moved instance stays movable after its last move.
pub const SETTLE_FRAMES: u32 = 30;
/// Idle frames before free slots are compacted (and how many make it worthwhile).
const COMPACT_IDLE_FRAMES: u32 = 90;
const COMPACT_INSTANCES: usize = 32;
const COMPACT_LIGHTS: usize = 8;
/// After this long idle, any free slot is compacted.
const COMPACT_ANY_FRAMES: u32 = 1800;

/// The key of an added actor.
pub fn added_key(id: u32) -> String {
    format!("+{id}")
}

/// A component of the base scene.
#[derive(Clone, Debug)]
pub struct BaseComponent {
    pub name: String,
    /// Relative to the actor.
    pub local: Mat4,
    pub visible: bool,
    pub kind: BaseKind,
}

#[derive(Clone, Debug)]
pub enum BaseKind {
    /// The built scene's instance `instance`.
    Mesh { instance: u32, mesh: MeshHandle, material: MaterialHandle, cast_shadow: bool },
    /// The built scene's light `index` (position and direction relative to the actor).
    Light { index: u32, def: LightDef },
    /// The built scene's volume / particle emitter `index`.
    Volume { index: u32 },
    Particles { index: u32 },
}

/// An actor of the base scene.
#[derive(Clone, Debug)]
pub struct BaseActor {
    pub key: String,
    pub name: String,
    pub parent: Option<String>,
    /// Relative to the parent.
    pub local: Mat4,
    /// World transform as built.
    pub world: Mat4,
    pub folder: bool,
    pub components: Vec<BaseComponent>,
}

/// A live component.
#[derive(Clone, Debug)]
pub struct Component {
    pub name: String,
    pub local: Mat4,
    pub visible: bool,
    pub kind: Kind,
}

#[derive(Clone, Debug)]
pub enum Kind {
    Mesh { mesh: MeshHandle, material: MaterialHandle, cast_shadow: bool, slot: Option<InstanceHandle> },
    /// Position and direction relative to the actor; `slot` is the scene light index.
    Light { def: LightDef, slot: Option<usize> },
    Volume { index: u32 },
    Particles { index: u32 },
}

impl Kind {
    pub fn name(&self) -> &'static str {
        match self {
            Kind::Mesh { .. } => "mesh",
            Kind::Light { .. } => "light",
            Kind::Volume { .. } => "volume",
            Kind::Particles { .. } => "particles",
        }
    }
}

/// A live actor.
#[derive(Clone, Debug)]
pub struct Actor {
    pub key: String,
    pub name: String,
    pub parent_key: Option<String>,
    pub parent: Option<u32>,
    pub children: Vec<u32>,
    pub local: Mat4,
    pub world: Mat4,
    /// Its own visibility, and with its ancestors'.
    pub visible: bool,
    pub shown: bool,
    pub added: bool,
    /// A builder's folder (no components, no transform of its own).
    pub folder: bool,
    pub components: Vec<Component>,
}

/// The world-space light of a light component.
pub fn world_light(actor_world: Mat4, def: &LightDef, enabled: bool) -> LightDef {
    let mut d = def.clone();
    d.position = actor_world.transform_point3(Vec3::from(def.position)).into();
    d.direction = actor_world.transform_vector3(Vec3::from(def.direction)).normalize_or(Vec3::NEG_Y).into();
    d.enabled = def.enabled && enabled;
    d
}

/// A world-space light definition relative to `actor_world`.
pub fn local_light(actor_world: Mat4, def: &LightDef) -> LightDef {
    let inverse = actor_world.inverse();
    let mut d = def.clone();
    d.position = inverse.transform_point3(Vec3::from(def.position)).into();
    d.direction = inverse.transform_vector3(Vec3::from(def.direction)).normalize_or(Vec3::NEG_Y).into();
    d
}

fn unique(taken: &mut BTreeSet<String>, base: &str) -> String {
    let name = (1..).map(|n| if n == 1 { base.to_string() } else { format!("{base} #{n}") }).find(|n| !taken.contains(n)).expect("a free name");
    taken.insert(name.clone());
    name
}

/// The base scene's actors: the builder's outline, plus an actor for every instance, light and
/// emitter it left out.
pub fn base_actors(scene: &Scene, outline: &Outline) -> Vec<BaseActor> {
    let counts = (scene.instance_count(), scene.lights().len(), scene.volumes.len(), scene.particles.len());
    let mut claimed = (vec![false; counts.0], vec![false; counts.1], vec![false; counts.2], vec![false; counts.3]);
    let mut keys: BTreeSet<String> = outline.actors.iter().map(|a| a.key.clone()).collect();
    let mut actors: Vec<BaseActor> = Vec::with_capacity(outline.actors.len());
    let component = |taken: &mut BTreeSet<String>, name: &str, pivot_inverse: Mat4, world: Mat4, kind: BaseKind| BaseComponent {
        name: unique(taken, if name.is_empty() { "mesh" } else { name }),
        local: pivot_inverse * world,
        visible: true,
        kind,
    };
    let light_component = |taken: &mut BTreeSet<String>, index: u32, pivot: Mat4| {
        let light = scene.lights()[index as usize];
        let name = unique(taken, "light");
        let def = local_light(pivot, &light_def(&light));
        BaseComponent { name, local: Mat4::from_translation(Vec3::from(def.position)), visible: true, kind: BaseKind::Light { index, def } }
    };
    let mesh_kind = |i: u32| {
        let h = scene.instance_handle(i).expect("outlined instances exist");
        BaseKind::Mesh { instance: i, mesh: scene.instance_mesh(h), material: scene.instance_material(h), cast_shadow: scene.casts_shadow(h) }
    };
    for a in &outline.actors {
        let parent_world = a.parent.map_or(Mat4::IDENTITY, |p| outline.actors[p].pivot);
        let inverse = a.pivot.inverse();
        let mut taken = BTreeSet::new();
        let mut components = Vec::new();
        for &i in a.instances.iter().filter(|&&i| (i as usize) < counts.0) {
            claimed.0[i as usize] = true;
            let h = scene.instance_handle(i).expect("in range");
            components.push(component(&mut taken, scene.mesh_name(scene.instance_mesh(h)), inverse, scene.transform(h), mesh_kind(i)));
        }
        for &l in a.lights.iter().filter(|&&l| (l as usize) < counts.1) {
            claimed.1[l as usize] = true;
            components.push(light_component(&mut taken, l, a.pivot));
        }
        for &v in a.volumes.iter().filter(|&&v| (v as usize) < counts.2) {
            claimed.2[v as usize] = true;
            components.push(component(&mut taken, "volume", inverse, Mat4::from_translation(scene.volumes[v as usize].position), BaseKind::Volume { index: v }));
        }
        for &p in a.particles.iter().filter(|&&p| (p as usize) < counts.3) {
            claimed.3[p as usize] = true;
            components.push(component(&mut taken, "particles", inverse, Mat4::from_translation(scene.particles[p as usize].position), BaseKind::Particles { index: p }));
        }
        actors.push(BaseActor {
            key: a.key.clone(),
            name: a.name.clone(),
            parent: a.parent.map(|p| outline.actors[p].key.clone()),
            local: parent_world.inverse() * a.pivot,
            world: a.pivot,
            folder: a.folder,
            components,
        });
    }
    // Everything else: an actor each, at its own transform.
    let mut single = |name: &str, pivot: Mat4, make: &dyn Fn(&mut BTreeSet<String>, Mat4) -> BaseComponent| {
        let key = unique(&mut keys, &format!("{name}@{}", position_key(pivot.w_axis.truncate())));
        let mut taken = BTreeSet::new();
        let components = vec![make(&mut taken, pivot)];
        actors.push(BaseActor { key, name: name.to_string(), parent: None, local: pivot, world: pivot, folder: false, components });
    };
    for i in (0..claimed.0.len() as u32).filter(|&i| !claimed.0[i as usize]) {
        let h = scene.instance_handle(i).expect("in range");
        let mesh = scene.mesh_name(scene.instance_mesh(h));
        let name = if mesh.is_empty() { "Static mesh" } else { mesh };
        let t = scene.transform(h);
        single(name, t, &|taken, _| component(taken, name, Mat4::IDENTITY, Mat4::IDENTITY, mesh_kind(i)));
    }
    for l in (0..claimed.1.len() as u32).filter(|&l| !claimed.1[l as usize]) {
        let light = scene.lights()[l as usize];
        let name = if matches!(light.kind, LightKind::Spot { .. }) { "Spot light" } else { "Point light" };
        single(name, Mat4::from_translation(light.position), &|taken, pivot| light_component(taken, l, pivot));
    }
    for v in (0..claimed.2.len() as u32).filter(|&v| !claimed.2[v as usize]) {
        single("Volume", Mat4::from_translation(scene.volumes[v as usize].position), &|taken, _| component(taken, "volume", Mat4::IDENTITY, Mat4::IDENTITY, BaseKind::Volume { index: v }));
    }
    for p in (0..claimed.3.len() as u32).filter(|&p| !claimed.3[p as usize]) {
        single("Particles", Mat4::from_translation(scene.particles[p as usize].position), &|taken, _| {
            component(taken, "particles", Mat4::IDENTITY, Mat4::IDENTITY, BaseKind::Particles { index: p })
        });
    }
    actors
}

/// The emitters of the base scene and which light component a fire drives.
#[derive(Default)]
pub struct BaseEmitters {
    pub volumes: Vec<VolumeEmitter>,
    pub particles: Vec<ParticleEmitter>,
    /// Per volume: the (actor key, component name) of the light it drives.
    pub volume_lights: Vec<Option<(String, String)>>,
}

impl Editor {
    // ---- Lookup ------------------------------------------------------------------------------

    pub fn actor(&self, id: u32) -> Option<&Actor> {
        self.actors.get(&id)
    }

    /// Live actors by id (base actors in build order first, then added ones).
    pub fn actors(&self) -> impl Iterator<Item = (u32, &Actor)> {
        self.actors.iter().map(|(&id, a)| (id, a))
    }

    pub fn actor_id(&self, key: &str) -> Option<u32> {
        self.ids.get(key).copied().filter(|id| self.actors.contains_key(id))
    }

    /// The id of `key`, assigned on first use (ids are never reused, so the UI's references
    /// stay valid across undo).
    fn id_for(&mut self, key: &str) -> u32 {
        if let Some(&id) = self.ids.get(key) {
            return id;
        }
        let id = self.next_live_id;
        self.next_live_id += 1;
        self.ids.insert(key.to_string(), id);
        id
    }

    /// `id` and every actor below it, parents before children.
    pub fn subtree(&self, id: u32) -> Vec<u32> {
        let mut out = vec![id];
        let mut k = 0;
        while k < out.len() {
            if let Some(a) = self.actors.get(&out[k]) {
                out.extend(a.children.iter().copied());
            }
            k += 1;
        }
        out
    }

    /// Whether `ancestor` is `id` or above it.
    pub fn is_ancestor(&self, ancestor: u32, id: u32) -> bool {
        let mut at = Some(id);
        let mut steps = 0;
        while let Some(a) = at {
            if a == ancestor {
                return true;
            }
            at = self.actors.get(&a).and_then(|x| x.parent);
            steps += 1;
            if steps > 10_000 {
                return false;
            }
        }
        false
    }

    /// `ids` without those that have an ancestor in `ids` (moving, duplicating or deleting an
    /// actor takes its children along).
    pub fn top_level(&self, ids: &[u32]) -> Vec<u32> {
        let set: BTreeSet<u32> = ids.iter().copied().collect();
        let mut out = Vec::new();
        for &id in ids {
            let mut above = self.actors.get(&id).and_then(|a| a.parent);
            let mut covered = false;
            while let Some(p) = above {
                if set.contains(&p) {
                    covered = true;
                    break;
                }
                above = self.actors.get(&p).and_then(|a| a.parent);
            }
            if !covered && self.actors.contains_key(&id) && !out.contains(&id) {
                out.push(id);
            }
        }
        out
    }

    /// The world transform of the parent of actor `id` (identity at the root).
    pub fn parent_world(&self, id: u32) -> Mat4 {
        self.actors.get(&id).and_then(|a| a.parent).and_then(|p| self.actors.get(&p)).map_or(Mat4::IDENTITY, |p| p.world)
    }

    pub fn component_index(&self, actor: u32, name: &str) -> Option<usize> {
        self.actors.get(&actor)?.components.iter().position(|c| c.name == name)
    }

    /// The actor and component of instance `index` (scene index).
    pub fn instance_component(&self, index: u32) -> Option<(u32, usize)> {
        match self.owner(index) {
            Owner::Actor { actor, component } => Some((actor, component as usize)),
            _ => None,
        }
    }

    /// The actor and component of scene light `index`.
    pub fn light_component(&self, index: usize) -> Option<(u32, usize)> {
        self.light_owners.get(index).copied().flatten()
    }

    /// The instance slots of `id`'s mesh components and those of every actor below it.
    pub fn subtree_instances(&self, id: u32) -> Vec<InstanceHandle> {
        self.subtree(id)
            .into_iter()
            .filter_map(|a| self.actors.get(&a))
            .flat_map(|a| a.components.iter())
            .filter_map(|c| match c.kind {
                Kind::Mesh { slot, .. } => slot,
                _ => None,
            })
            .collect()
    }

    /// World bounds of an actor and everything below it (instances' bounds, lights' and
    /// emitters' positions); `None` for an empty folder.
    pub fn actor_bounds(&self, scene: &Scene, id: u32) -> Option<(Vec3, Vec3)> {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for a in self.subtree(id).into_iter().filter_map(|a| self.actors.get(&a)) {
            for c in &a.components {
                let (l, h) = match c.kind {
                    Kind::Mesh { slot: Some(h), .. } => scene.instance_bounds(h),
                    Kind::Mesh { slot: None, .. } => continue,
                    _ => {
                        let p = (a.world * c.local).w_axis.truncate();
                        (p - Vec3::splat(0.25), p + Vec3::splat(0.25))
                    }
                };
                lo = lo.min(l);
                hi = hi.max(h);
            }
        }
        (lo.x <= hi.x).then_some((lo, hi))
    }

    /// The instances the selection outline shows: every mesh component of the selected actors
    /// (with their children) and the selected components.
    pub fn selection_handles(&self) -> Vec<InstanceHandle> {
        let mut out: Vec<InstanceHandle> = self.selected_actors.iter().flat_map(|&a| self.subtree_instances(a)).collect();
        for (actor, name) in &self.selected_components {
            if let Some(Kind::Mesh { slot: Some(h), .. }) = self.component_index(*actor, name).map(|c| &self.actors[actor].components[c].kind) {
                out.push(*h);
            }
        }
        out.sort_by_key(|h| h.index());
        out.dedup();
        out
    }

    pub fn set_selection(&mut self, actors: Vec<u32>, components: Vec<(u32, String)>) {
        self.selected_actors = actors;
        self.selected_components = components;
    }

    // ---- Resolving the document ------------------------------------------------------------

    /// The actor the document asks for under `key` (slots not assigned yet), or `None`.
    fn resolve(&mut self, scene: &mut Scene, key: &str) -> Option<Actor> {
        if let Some(&b) = self.base_index.get(key) {
            let edit = self.doc.actors.get(key).cloned().unwrap_or_default();
            if edit.deleted {
                return None;
            }
            let base = self.base_actors[b].clone();
            let mut components = Vec::new();
            for c in &base.components {
                let e = edit.components.get(&c.name).cloned().unwrap_or_default();
                if e.deleted {
                    continue;
                }
                let kind = match &c.kind {
                    BaseKind::Mesh { mesh, material, cast_shadow, .. } => {
                        let material = match &e.material {
                            Some(name) => self.material_handle(scene, name).unwrap_or(*material),
                            None => *material,
                        };
                        Kind::Mesh { mesh: *mesh, material, cast_shadow: e.cast_shadow.unwrap_or(*cast_shadow), slot: None }
                    }
                    BaseKind::Light { def, .. } => Kind::Light { def: e.light.clone().unwrap_or_else(|| def.clone()), slot: None },

                    BaseKind::Volume { index } => Kind::Volume { index: *index },
                    BaseKind::Particles { index } => Kind::Particles { index: *index },
                };
                // A light's position is its definition's.
                let local = match &kind {
                    Kind::Light { def, .. } => Mat4::from_translation(Vec3::from(def.position)),
                    _ => e.transform.map_or(c.local, |t| t.to_mat4()),
                };
                components.push(Component { name: c.name.clone(), local, visible: e.visible.unwrap_or(c.visible), kind });
            }
            let parent_key = match &edit.parent {
                Some(p) if p.is_empty() => None,
                Some(p) => Some(p.clone()),
                None => base.parent.clone(),
            };
            return Some(Actor {
                key: key.to_string(),
                name: edit.name.clone().unwrap_or(base.name),
                parent_key,
                parent: None,
                children: Vec::new(),
                local: edit.transform.map_or(base.local, |t| t.to_mat4()),
                world: Mat4::IDENTITY,
                visible: edit.visible.unwrap_or(true),
                shown: true,
                added: false,
                folder: base.folder && edit.components.is_empty(),
                components,
            });
        }
        let id: u32 = key.strip_prefix('+')?.parse().ok()?;
        let def = self.doc.added_actors.iter().find(|a| a.id == id)?.clone();
        let mut components = Vec::new();
        for c in &def.components {
            let kind = match &c.kind {
                ComponentKindDef::Mesh { mesh, material, cast_shadow } => {
                    let (Some(mesh), Some(material)) = (self.mesh_handle(scene, mesh), self.material_handle(scene, material)) else { continue };
                    Kind::Mesh { mesh, material, cast_shadow: *cast_shadow, slot: None }
                }
                ComponentKindDef::Light { light } => Kind::Light { def: light.clone(), slot: None },
            };
            let local = match &kind {
                Kind::Light { def, .. } => Mat4::from_translation(Vec3::from(def.position)),
                _ => c.transform.to_mat4(),
            };
            components.push(Component { name: c.name.clone(), local, visible: c.visible, kind });
        }
        Some(Actor {
            key: key.to_string(),
            name: def.name.clone(),
            parent_key: def.parent.clone(),
            parent: None,
            children: Vec::new(),
            local: def.transform.to_mat4(),
            world: Mat4::IDENTITY,
            visible: def.visible,
            shown: true,
            added: true,
            folder: false,
            components,
        })
    }

    /// Keys whose actor definition differs between the document and what the scene shows.
    fn changed_actor_keys(&self) -> BTreeSet<String> {
        let mut keys: BTreeSet<String> = self.doc.actors.keys().chain(self.applied.actors.keys()).filter(|k| self.doc.actors.get(*k) != self.applied.actors.get(*k)).cloned().collect();
        let wanted: BTreeMap<u32, &AddedActor> = self.doc.added_actors.iter().map(|a| (a.id, a)).collect();
        let applied: BTreeMap<u32, &AddedActor> = self.applied.added_actors.iter().map(|a| (a.id, a)).collect();
        for id in wanted.keys().chain(applied.keys()) {
            if wanted.get(id) != applied.get(id) {
                keys.insert(added_key(*id));
            }
        }
        keys
    }

    /// Brings the live actors in line with the document (see the module notes); `interactive`:
    /// a gesture's commit (its instances move as movable ones).
    pub(crate) fn sync_actors(&mut self, world: &mut World, interactive: bool) {
        let keys = self.changed_actor_keys();
        if keys.is_empty() {
            return;
        }
        self.preview_worlds.clear();
        let mut changed = Vec::new();
        let mut relink = false;
        for key in &keys {
            let desired = self.resolve(world.scene, key);
            let id = self.id_for(key);
            let old = self.actors.remove(&id);
            let links = old.as_ref().map(|o| (o.parent, o.children.clone()));
            relink |= match (&old, &desired) {
                (Some(o), Some(d)) => o.parent_key != d.parent_key,
                (None, None) => false,
                _ => true,
            };
            match desired {
                Some(mut actor) => {
                    if let Some((parent, children)) = links {
                        actor.parent = parent;
                        actor.children = children;
                    }
                    let mut old_components = old.map(|o| o.components).unwrap_or_default();
                    for c in &mut actor.components {
                        let reuse = old_components.iter().position(|o| o.name == c.name && same_slot_kind(&o.kind, &c.kind));
                        if let Some(k) = reuse {
                            let o = old_components.remove(k);
                            match (&mut c.kind, o.kind) {
                                (Kind::Mesh { slot, .. }, Kind::Mesh { slot: old, .. }) => *slot = old,
                                (Kind::Light { slot, .. }, Kind::Light { slot: old, .. }) => *slot = old,
                                _ => {}
                            }
                        }
                    }
                    for o in old_components {
                        self.retire_component(world.scene, &o);
                    }
                    self.assign_slots(world.scene, id, &mut actor);
                    self.actors.insert(id, actor);
                    changed.push(id);
                }
                None => {
                    for c in old.iter().flat_map(|o| o.components.iter()) {
                        self.retire_component(world.scene, c);
                    }
                    self.selected_actors.retain(|&a| a != id);
                }
            }
        }
        // Unless parents changed or actors came or went, the links carried over stand.
        if relink {
            self.relink();
        }
        self.place_subtrees(world.scene, &changed, interactive);
        self.idle_frames = 0;
    }

    /// Recomputes every actor's parent and children from the parent keys (a parent that is
    /// missing or would close a cycle makes the actor a root).
    fn relink(&mut self) {
        let parents: Vec<(u32, Option<u32>)> = self.actors.iter().map(|(&id, a)| (id, a.parent_key.as_deref().and_then(|k| self.actor_id(k)).filter(|&p| p != id))).collect();
        for a in self.actors.values_mut() {
            a.children.clear();
            a.parent = None;
        }
        for &(id, parent) in &parents {
            self.actors.get_mut(&id).expect("listed").parent = parent;
        }
        // Break cycles (a corrupt file): walk up at most the actor count.
        let limit = self.actors.len();
        for &(id, _) in &parents {
            let mut at = self.actors[&id].parent;
            let mut steps = 0;
            while let Some(p) = at {
                if p == id || steps > limit {
                    self.warnings.push(format!("actor '{}' was its own ancestor: moved to the root", self.actors[&id].name));
                    self.actors.get_mut(&id).expect("listed").parent = None;
                    break;
                }
                at = self.actors.get(&p).and_then(|a| a.parent);
                steps += 1;
            }
        }
        let links: Vec<(u32, u32)> = self.actors.iter().filter_map(|(&id, a)| a.parent.map(|p| (p, id))).collect();
        for (p, c) in links {
            self.actors.get_mut(&p).expect("parent exists").children.push(c);
        }
        for (id, a) in &self.actors {
            if a.parent.is_none() {
                if let Some(k) = &a.parent_key {
                    if self.actor_id(k).is_none() {
                        self.warnings.push(format!("actor '{}' (#{id}): parent '{k}' does not exist", a.name));
                    }
                }
            }
        }
    }

    /// Gives components without a slot one (a free one of the same mesh, or a new one).
    fn assign_slots(&mut self, scene: &mut Scene, id: u32, actor: &mut Actor) {
        for (index, c) in actor.components.iter_mut().enumerate() {
            match &mut c.kind {
                Kind::Mesh { mesh, material, slot, .. } => {
                    let h = match *slot {
                        Some(h) => h,
                        None => {
                            let h = match self.spares.get_mut(mesh).and_then(Vec::pop) {
                                Some(h) => h,
                                None => scene.add_instance(*mesh, *material, Mat4::IDENTITY),
                            };
                            // A reused slot may have moved as a movable one before.
                            if !self.interactive.contains_key(&h) {
                                scene.set_mobility(h, Mobility::Static);
                            }
                            *slot = Some(h);
                            h
                        }
                    };
                    let i = h.index() as usize;
                    if self.owners.len() <= i {
                        self.owners.resize(i + 1, Owner::Spare);
                    }
                    self.owners[i] = Owner::Actor { actor: id, component: index as u16 };
                }
                Kind::Light { def, slot } => {
                    let i = match *slot {
                        Some(i) => i,
                        None => {
                            let light = light_from_def(def);
                            let i = self.spare_lights.pop().unwrap_or_else(|| {
                                scene.add_light(light);
                                scene.lights().len() - 1
                            });
                            *slot = Some(i);
                            i
                        }
                    };
                    if self.light_owners.len() <= i {
                        self.light_owners.resize(i + 1, None);
                    }
                    self.light_owners[i] = Some((id, index));
                }
                Kind::Volume { .. } | Kind::Particles { .. } => self.emitters_dirty = true,
            }
        }
    }

    /// Hides a component's slot and puts it on the free list.
    fn retire_component(&mut self, scene: &mut Scene, c: &Component) {
        match c.kind {
            Kind::Mesh { slot: Some(h), .. } => self.retire(scene, h),
            Kind::Light { slot: Some(i), .. } => {
                if let Some(light) = scene.lights_mut().get_mut(i) {
                    *light = light_from_def(&LightDef { enabled: false, ..light_def(light) });
                }
                if let Some(owner) = self.light_owners.get_mut(i) {
                    *owner = None;
                }
                self.spare_lights.push(i);
            }
            Kind::Volume { .. } | Kind::Particles { .. } => self.emitters_dirty = true,
            _ => {}
        }
    }

    pub(crate) fn retire(&mut self, scene: &mut Scene, handle: InstanceHandle) {
        scene.set_visible(handle, false);
        let index = handle.index() as usize;
        if let Some(owner) = self.owners.get_mut(index) {
            *owner = Owner::Spare;
        }
        self.spares.entry(scene.instance_mesh(handle)).or_default().push(handle);
    }

    /// Computes world transforms and visibility of `roots` and everything below them, and
    /// places their components.
    fn place_subtrees(&mut self, scene: &mut Scene, roots: &[u32], interactive: bool) {
        let mut done = BTreeSet::new();
        // Shallow first, so a parent's world is current before its children's.
        let mut order: Vec<(usize, u32)> = roots.iter().filter(|id| self.actors.contains_key(id)).map(|&id| (self.depth(id), id)).collect();
        order.sort();
        for (_, root) in order {
            for id in self.subtree(root) {
                if !done.insert(id) {
                    continue;
                }
                let (parent_world, parent_shown) = match self.actors[&id].parent.and_then(|p| self.actors.get(&p)) {
                    Some(p) => (p.world, p.shown),
                    None => (Mat4::IDENTITY, true),
                };
                let a = self.actors.get_mut(&id).expect("in the subtree");
                a.world = parent_world * a.local;
                a.shown = parent_shown && a.visible;
                let (world, shown) = (a.world, a.shown);
                self.place_components(scene, id, world, shown, interactive);
            }
        }
        if self.emitters_dirty {
            self.rebuild_emitters(scene);
        }
    }

    fn depth(&self, id: u32) -> usize {
        let mut d = 0;
        let mut at = self.actors.get(&id).and_then(|a| a.parent);
        while let Some(p) = at {
            d += 1;
            at = self.actors.get(&p).and_then(|a| a.parent);
            if d > self.actors.len() {
                break;
            }
        }
        d
    }

    /// Puts actor `id`'s components where an actor at `world` has them.
    fn place_components(&mut self, scene: &mut Scene, id: u32, world: Mat4, shown: bool, interactive: bool) {
        let count = self.actors[&id].components.len();
        for k in 0..count {
            let c = &self.actors[&id].components[k];
            let (local, visible) = (c.local, c.visible && shown);
            match c.kind.clone() {
                Kind::Mesh { slot: Some(h), material, cast_shadow, .. } => {
                    let transform = world * local;
                    if interactive && scene.transform(h) != transform {
                        self.mark_moving(scene, h);
                    }
                    scene.set_transform(h, transform);
                    scene.set_instance_material(h, material);
                    if scene.casts_shadow(h) != cast_shadow {
                        scene.set_cast_shadow(h, cast_shadow);
                    }
                    scene.set_visible(h, visible);
                }
                Kind::Light { def, slot: Some(i) } => {
                    if let Some(light) = scene.lights_mut().get_mut(i) {
                        *light = light_from_def(&world_light(world, &def, visible));
                    }
                }
                Kind::Volume { .. } | Kind::Particles { .. } => self.emitters_dirty = true,
                _ => {}
            }
        }
    }

    /// Rebuilds the scene's volume and particle emitters from the live components (in base
    /// order, so unchanged emitters keep their simulation state).
    fn rebuild_emitters(&mut self, scene: &mut Scene) {
        self.emitters_dirty = false;
        let mut volumes: BTreeMap<u32, VolumeEmitter> = BTreeMap::new();
        let mut particles: BTreeMap<u32, ParticleEmitter> = BTreeMap::new();
        for (id, a) in &self.actors {
            let world = self.preview_worlds.get(id).copied().unwrap_or(a.world);
            for c in a.components.iter().filter(|c| c.visible && a.shown) {
                let at = (world * c.local).w_axis.truncate();
                match c.kind {
                    Kind::Volume { index } => {
                        if let Some(mut e) = self.emitters.volumes.get(index as usize).copied() {
                            e.position = at;
                            e.light = self.emitters.volume_lights.get(index as usize).cloned().flatten().and_then(|(key, name)| {
                                let actor = self.actors.get(&self.actor_id(&key)?)?;
                                let shown = actor.shown;
                                match actor.components.iter().find(|c| c.name == name)?.kind {
                                    Kind::Light { slot: Some(i), .. } if shown => scene.light_handle(i as u32),
                                    _ => None,
                                }
                            });
                            volumes.insert(index, e);
                        }
                    }
                    Kind::Particles { index } => {
                        if let Some(mut e) = self.emitters.particles.get(index as usize).copied() {
                            e.position = at;
                            particles.insert(index, e);
                        }
                    }
                    _ => {}
                }
            }
        }
        scene.volumes = volumes.into_values().collect();
        scene.particles = particles.into_values().collect();
    }

    fn mark_moving(&mut self, scene: &mut Scene, h: InstanceHandle) {
        if self.static_drags {
            return;
        }
        let entry = self.interactive.entry(h).or_insert_with(|| (scene.mobility(h), 0));
        entry.1 = SETTLE_FRAMES;
        scene.set_mobility(h, Mobility::Movable);
    }

    // ---- Interactive moves --------------------------------------------------------------------

    /// Moves actors to new world transforms during a gizmo drag (no document change; the drag's
    /// end commits a transform edit, a cancelled drag calls [`Editor::revert_preview`]).
    /// Children follow; the moved instances become movable until the drag settles.
    pub fn preview_actors(&mut self, world: &mut World, items: &[(u32, Mat4)]) {
        let ids: Vec<u32> = items.iter().map(|i| i.0).collect();
        let top = self.top_level(&ids);
        for &(id, new_world) in items.iter().filter(|(id, _)| top.contains(id)) {
            let mut stack = vec![(id, new_world)];
            while let Some((a, w)) = stack.pop() {
                let Some(actor) = self.actors.get(&a) else { continue };
                let shown = actor.shown;
                for &c in &actor.children {
                    stack.push((c, w * self.actors[&c].local));
                }
                self.preview_worlds.insert(a, w);
                self.place_components(world.scene, a, w, shown, true);
            }
        }
        if self.emitters_dirty {
            self.rebuild_emitters(world.scene);
        }
        self.idle_frames = 0;
    }

    /// Moves components to new world transforms during a drag.
    pub fn preview_components(&mut self, world: &mut World, items: &[(u32, String, Mat4)]) {
        for (actor, name, w) in items {
            let Some(a) = self.actors.get(actor) else { continue };
            let Some(c) = a.components.iter().find(|c| &c.name == name) else { continue };
            let (visible, local, kind) = (a.shown && c.visible, c.local, c.kind.clone());
            self.preview_worlds.entry(*actor).or_insert(a.world);
            match kind {
                Kind::Mesh { slot: Some(h), .. } => {
                    self.mark_moving(world.scene, h);
                    world.scene.set_transform(h, *w);
                }
                Kind::Light { def, slot: Some(i) } => {
                    let d = world_light(*w * local.inverse(), &def, visible);
                    world.scene.lights_mut()[i] = light_from_def(&d);
                }
                _ => {}
            }
        }
        self.idle_frames = 0;
    }

    /// Puts previewed actors back where the document has them.
    pub fn revert_preview(&mut self, world: &mut World) {
        let ids: Vec<u32> = std::mem::take(&mut self.preview_worlds).into_keys().filter(|id| self.actors.contains_key(id)).collect();
        self.place_subtrees(world.scene, &ids, false);
        self.emitters_dirty = true;
        self.rebuild_emitters(world.scene);
    }

    /// Per frame: gives settled instances their mobility back and counts idle frames.
    pub fn tick(&mut self, scene: &mut Scene) {
        self.idle_frames = self.idle_frames.saturating_add(1);
        if self.interactive.is_empty() {
            return;
        }
        let mut settled = Vec::new();
        for (h, (_, left)) in self.interactive.iter_mut() {
            *left = left.saturating_sub(1);
            if *left == 0 {
                settled.push(*h);
            }
        }
        for h in settled {
            if let Some((mobility, _)) = self.interactive.remove(&h) {
                if (h.index() as usize) < scene.instance_count() {
                    scene.set_mobility(h, mobility);
                }
            }
        }
    }

    /// Instances currently movable because of a drag.
    pub fn moving_count(&self) -> usize {
        self.interactive.len()
    }

    // ---- Compaction -------------------------------------------------------------------------

    /// Free instance and light slots, if it is time to remove them (idle long enough, enough
    /// of them, or `force`).
    pub fn compaction(&self, scene: &Scene, force: bool) -> Option<(Vec<InstanceHandle>, Vec<LightHandle>)> {
        let instances: usize = self.spares.values().map(Vec::len).sum();
        let lights = self.spare_lights.len();
        if instances + lights == 0 || !self.interactive.is_empty() || !self.preview_worlds.is_empty() {
            return None;
        }
        let due = self.idle_frames >= COMPACT_IDLE_FRAMES && (instances >= COMPACT_INSTANCES || lights >= COMPACT_LIGHTS || self.idle_frames >= COMPACT_ANY_FRAMES);
        if !due && !force {
            return None;
        }
        let mut handles: Vec<InstanceHandle> = self.spares.values().flatten().copied().collect();
        handles.sort_by_key(|h| h.index());
        let lights = self.spare_lights.iter().filter_map(|&i| scene.light_handle(i as u32)).collect();
        Some((handles, lights))
    }

    /// Maps every held slot through `map` after `Scene::compact` removed the free ones.
    pub fn compacted(&mut self, scene: &mut Scene, map: &Compaction) {
        let instance = |h: InstanceHandle| map.instance(h);
        for a in self.actors.values_mut() {
            for c in &mut a.components {
                match &mut c.kind {
                    Kind::Mesh { slot, .. } => *slot = slot.and_then(instance),
                    Kind::Light { slot, .. } => *slot = slot.and_then(|i| map.light_index(i)),
                    _ => {}
                }
            }
        }
        for state in self.pcg.values_mut() {
            state.handles = state.handles.iter().filter_map(|&h| instance(h)).collect();
        }
        for list in self.spares.values_mut() {
            list.retain_mut(|h| match instance(*h) {
                Some(n) => {
                    *h = n;
                    true
                }
                None => false,
            });
        }
        self.spares.retain(|_, l| !l.is_empty());
        self.spare_lights = self.spare_lights.iter().filter_map(|&i| map.light_index(i)).collect();
        self.interactive = std::mem::take(&mut self.interactive).into_iter().filter_map(|(h, v)| Some((instance(h)?, v))).collect();
        self.rebuild_owners(scene);
        self.emitters_dirty = true;
        self.rebuild_emitters(scene);
        self.revision += 1;
        self.compactions += 1;
    }

    /// Owner tables from the live slots.
    fn rebuild_owners(&mut self, scene: &Scene) {
        self.owners = vec![Owner::Spare; scene.instance_count()];
        self.light_owners = vec![None; scene.lights().len()];
        for (&id, a) in &self.actors {
            for (k, c) in a.components.iter().enumerate() {
                match c.kind {
                    Kind::Mesh { slot: Some(h), .. } => self.owners[h.index() as usize] = Owner::Actor { actor: id, component: k as u16 },
                    Kind::Light { slot: Some(i), .. } => self.light_owners[i] = Some((id, k)),
                    _ => {}
                }
            }
        }
        for (&volume, state) in &self.pcg {
            for h in &state.handles {
                self.owners[h.index() as usize] = Owner::Pcg(volume);
            }
        }
    }

    // ---- Building the live actors -------------------------------------------------------------

    /// The live actors of the base scene as built (every slot the builder's).
    pub(crate) fn init_actors(&mut self, scene: &Scene) {
        self.base_index = self.base_actors.iter().enumerate().map(|(i, a)| (a.key.clone(), i)).collect();
        self.owners = vec![Owner::Spare; scene.instance_count()];
        self.light_owners = vec![None; scene.lights().len()];
        let base = self.base_actors.clone();
        for b in &base {
            let id = self.id_for(&b.key);
            let components = b
                .components
                .iter()
                .map(|c| Component {
                    name: c.name.clone(),
                    local: c.local,
                    visible: c.visible,
                    kind: match &c.kind {
                        BaseKind::Mesh { instance, mesh, material, cast_shadow } => {
                            Kind::Mesh { mesh: *mesh, material: *material, cast_shadow: *cast_shadow, slot: scene.instance_handle(*instance) }
                        }
                        BaseKind::Light { index, def } => Kind::Light { def: def.clone(), slot: Some(*index as usize) },
                        BaseKind::Volume { index } => Kind::Volume { index: *index },
                        BaseKind::Particles { index } => Kind::Particles { index: *index },
                    },
                })
                .collect();
            self.actors.insert(
                id,
                Actor {
                    key: b.key.clone(),
                    name: b.name.clone(),
                    parent_key: b.parent.clone(),
                    parent: None,
                    children: Vec::new(),
                    local: b.local,
                    world: b.world,
                    visible: true,
                    shown: true,
                    added: false,
                    folder: b.folder,
                    components,
                },
            );
        }
        self.relink();
        self.rebuild_owners(scene);
        if !base.is_empty() {
            let folders = base.iter().filter(|a| a.folder).count();
            let grouped = base.iter().filter(|a| a.components.len() > 1).count();
            log::info!("editor: {} actors ({folders} folders, {grouped} of several components) over {} instances", base.len(), scene.instance_count());
        }
        // Which light component each fire drives.
        let light_of: HashMap<u32, (String, String)> = base
            .iter()
            .flat_map(|a| a.components.iter().filter_map(move |c| if let BaseKind::Light { index, .. } = c.kind { Some((index, (a.key.clone(), c.name.clone()))) } else { None }))
            .collect();
        self.emitters = BaseEmitters {
            volumes: scene.volumes.clone(),
            particles: scene.particles.clone(),
            volume_lights: scene.volumes.iter().map(|v| v.light.and_then(|l| light_of.get(&l.index()).cloned())).collect(),
        };
    }

    // ---- Version 1 documents ------------------------------------------------------------------

    /// Turns a version 1 document (base instances and lights by build order, added instances and
    /// lights) into actor edits of this base scene.
    pub(crate) fn migrate(&mut self, doc: &mut SceneDoc, scene: &Scene) {
        if doc.version >= 2 {
            return;
        }
        let mut by_instance: HashMap<u32, (usize, usize)> = HashMap::new();
        let mut by_light: HashMap<u32, (usize, usize)> = HashMap::new();
        for (a, actor) in self.base_actors.iter().enumerate() {
            for (c, comp) in actor.components.iter().enumerate() {
                match comp.kind {
                    BaseKind::Mesh { instance, .. } => {
                        by_instance.insert(instance, (a, c));
                    }
                    BaseKind::Light { index, .. } => {
                        by_light.insert(index, (a, c));
                    }
                    _ => {}
                }
            }
        }
        let parent_world = |a: &BaseActor| a.world * a.local.inverse();
        for (index, edit) in std::mem::take(&mut doc.instances) {
            let Some(&(a, c)) = by_instance.get(&index) else {
                self.warnings.push(format!("instance {index} is not in the base scene"));
                continue;
            };
            let actor = &self.base_actors[a];
            let comp = &actor.components[c];
            let BaseKind::Mesh { mesh, .. } = comp.kind else { continue };
            let actual = scene.mesh_name(mesh);
            if !edit.mesh.is_empty() && edit.mesh != actual {
                self.warnings.push(format!("instance {index} is '{actual}', the file expected '{}' (scene changed since saving?)", edit.mesh));
                continue;
            }
            let single = actor.components.len() == 1;
            let entry = doc.actors.entry(actor.key.clone()).or_default();
            if let Some(t) = edit.transform {
                if single {
                    entry.transform = Some(Trs::from_mat4(&(parent_world(actor).inverse() * t.to_mat4() * comp.local.inverse())));
                } else {
                    entry.components.entry(comp.name.clone()).or_default().transform = Some(Trs::from_mat4(&(actor.world.inverse() * t.to_mat4())));
                }
            }
            if edit.material.is_some() {
                entry.components.entry(comp.name.clone()).or_default().material = edit.material.clone();
            }
            let hidden = edit.visible == Some(false);
            if single {
                entry.deleted |= edit.deleted;
                if edit.visible.is_some() {
                    entry.visible = edit.visible;
                }
            } else {
                let ce = entry.components.entry(comp.name.clone()).or_default();
                ce.deleted |= edit.deleted;
                if hidden || edit.visible.is_some() {
                    ce.visible = edit.visible;
                }
            }
        }
        for added in std::mem::take(&mut doc.added) {
            doc.added_actors.push(AddedActor {
                id: added.id,
                name: added.mesh.rsplit('/').next().unwrap_or(&added.mesh).to_string(),
                parent: None,
                transform: added.transform,
                visible: true,
                components: vec![ComponentDef {
                    name: added.mesh.rsplit('/').next().unwrap_or(&added.mesh).to_string(),
                    transform: Trs::IDENTITY,
                    visible: true,
                    kind: ComponentKindDef::Mesh { mesh: added.mesh, material: added.material, cast_shadow: added.cast_shadow },
                }],
            });
        }
        for (index, def) in std::mem::take(&mut doc.lights) {
            let Some(&(a, c)) = by_light.get(&index) else {
                self.warnings.push(format!("light {index} is not in the base scene"));
                continue;
            };
            let actor = &self.base_actors[a];
            let comp = &actor.components[c];
            let single = actor.components.len() == 1;
            let entry = doc.actors.entry(actor.key.clone()).or_default();
            if !def.enabled {
                if single {
                    entry.deleted = true;
                } else {
                    entry.components.entry(comp.name.clone()).or_default().deleted = true;
                }
                continue;
            }
            entry.components.entry(comp.name.clone()).or_default().light = Some(local_light(actor.world, &def));
        }
        for added in std::mem::take(&mut doc.added_lights) {
            if !added.light.enabled {
                continue;
            }
            let at = Vec3::from(added.light.position);
            let name = if added.light.shape == LightShape::Spot { "Spot light" } else { "Point light" };
            doc.added_actors.push(AddedActor {
                id: added.id,
                name: name.into(),
                parent: None,
                transform: Trs::from_mat4(&Mat4::from_translation(at)),
                visible: true,
                components: vec![ComponentDef { name: "light".into(), transform: Trs::IDENTITY, visible: true, kind: ComponentKindDef::Light { light: LightDef { position: [0.0; 3], ..added.light } } }],
            });
        }
        for volume in &mut doc.pcg {
            if let SurfaceDef::Instance { instance } = volume.surface {
                if let Some(&(a, c)) = by_instance.get(&instance) {
                    volume.surface = SurfaceDef::Component { actor: self.base_actors[a].key.clone(), component: self.base_actors[a].components[c].name.clone() };
                }
            }
        }
        doc.version = VERSION;
    }

    /// The instance a PCG surface samples.
    pub(crate) fn surface_instance(&self, scene: &Scene, surface: &SurfaceDef) -> Option<InstanceHandle> {
        match surface {
            SurfaceDef::Landscape => None,
            SurfaceDef::Instance { instance } => scene.instance_handle(*instance),
            SurfaceDef::Component { actor, component } => {
                let a = self.actors.get(&self.actor_id(actor)?)?;
                match a.components.iter().find(|c| &c.name == component)?.kind {
                    Kind::Mesh { slot, .. } => slot,
                    _ => None,
                }
            }
        }
    }
}

fn same_slot_kind(a: &Kind, b: &Kind) -> bool {
    match (a, b) {
        (Kind::Mesh { mesh: m1, .. }, Kind::Mesh { mesh: m2, .. }) => m1 == m2,
        (Kind::Light { .. }, Kind::Light { .. }) => true,
        (Kind::Volume { index: a }, Kind::Volume { index: b }) | (Kind::Particles { index: a }, Kind::Particles { index: b }) => a == b,
        _ => false,
    }
}
