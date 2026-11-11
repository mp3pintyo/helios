//! Edit operations: what the editor UI asks for, turned into a new document (one undo step).
//!
//! Actors are addressed by their live ids, components by actor id and component name, PCG
//! volumes by id; the older instance- and light-addressed operations (scene indices, as picking
//! returns them) map to the component or, for an actor of one component, to the actor. Each
//! operation edits the document's keys: base actor overrides by stable key, added actors by id.

use std::collections::HashMap;

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::actors::{added_key, local_light, Kind};
use super::doc::*;
use super::{pcg, Editor, Owner, World};

/// A component: its actor's live id and its name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComponentRef {
    pub actor: u32,
    pub component: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum EditOp {
    /// World transforms of actors (one per actor); an actor whose ancestor is listed too moves
    /// with it.
    ActorTransform { actors: Vec<u32>, transforms: Vec<Trs> },
    /// Transforms of actors relative to their parents (the Details panel).
    ActorLocal { actors: Vec<u32>, transforms: Vec<Trs> },
    ActorVisibility { actors: Vec<u32>, visible: bool },
    Rename { actor: u32, name: String },
    /// Attaches actors to `parent` (none: the world root), keeping where they are.
    Reparent {
        actors: Vec<u32>,
        #[serde(default)]
        parent: Option<u32>,
    },
    /// A new actor at the actors' centre with them as its children.
    Group {
        actors: Vec<u32>,
        #[serde(default)]
        name: Option<String>,
    },
    /// Moves the children of each actor up to its parent and deletes it if it has no components.
    Ungroup { actors: Vec<u32> },
    /// World transforms of components.
    ComponentTransform { components: Vec<ComponentRef>, transforms: Vec<Trs> },
    /// Transforms of components relative to their actors.
    ComponentLocal { components: Vec<ComponentRef>, transforms: Vec<Trs> },
    ComponentVisibility { components: Vec<ComponentRef>, visible: bool },
    /// A light component's light (world space).
    ComponentLight { component: ComponentRef, def: LightDef },
    /// Sets instance transforms (scene indices, one per instance).
    Transform { instances: Vec<u32>, transforms: Vec<Trs> },
    Visibility { instances: Vec<u32>, visible: bool },
    AssignMaterial {
        #[serde(default)]
        instances: Vec<u32>,
        #[serde(default)]
        components: Vec<ComponentRef>,
        /// Every mesh component of these actors.
        #[serde(default)]
        actors: Vec<u32>,
        material: String,
    },
    CastShadow {
        #[serde(default)]
        instances: Vec<u32>,
        #[serde(default)]
        components: Vec<ComponentRef>,
        cast_shadow: bool,
    },
    /// Deletes actors with everything below them, components, instances, lights (scene
    /// indices) and PCG volumes.
    Delete {
        #[serde(default)]
        actors: Vec<u32>,
        #[serde(default)]
        components: Vec<ComponentRef>,
        #[serde(default)]
        instances: Vec<u32>,
        #[serde(default)]
        lights: Vec<u32>,
        #[serde(default)]
        pcg: Vec<u32>,
    },
    /// Copies actors with everything below them (instances and lights: their actors) and PCG
    /// volumes, moved by `offset`.
    Duplicate {
        #[serde(default)]
        actors: Vec<u32>,
        #[serde(default)]
        instances: Vec<u32>,
        #[serde(default)]
        lights: Vec<u32>,
        #[serde(default)]
        pcg: Vec<u32>,
        offset: [f32; 3],
    },
    /// Places a content item (engine prefab, imported model, `mesh:NAME`) as a new actor.
    Place {
        item: String,
        transform: Trs,
        #[serde(default)]
        parent: Option<u32>,
    },
    Material { name: String, params: MaterialParams },
    /// Creates or replaces a material graph (by name).
    Graph { name: String, graph: helios_render::material_graph::MaterialGraph },
    /// A light by scene index (world space).
    Light { light: u32, def: LightDef },
    /// A new light actor (world space).
    AddLight { def: LightDef },
    Environment { env: Environment },
    Render { key: String, value: Value },
    AddPcg { preset: String, center: [f32; 3] },
    Pcg { def: PcgVolumeDef },
}

/// What an operation created (live actor ids, their instance and light slots, PCG volume
/// ids), so the UI can select it.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Created {
    pub actors: Vec<u32>,
    pub instances: Vec<u32>,
    pub lights: Vec<u32>,
    pub pcg: Vec<u32>,
}

enum ActorChange {
    /// Relative to the parent.
    Local(Mat4),
    Visible(bool),
    Name(String),
    /// A parent key (none: the root).
    Parent(Option<String>),
    Delete,
}

enum ComponentChange {
    /// Relative to the actor.
    Local(Mat4),
    Visible(bool),
    Material(String),
    CastShadow(bool),
    /// Relative to the actor.
    Light(LightDef),
    Delete,
}

impl Editor {
    /// Applies `op` as one undo step.
    pub fn apply(&mut self, op: EditOp, world: &mut World) -> Result<Created, String> {
        self.apply_merged(op, world, None)
    }

    /// Applies `op`, joining the previous undo step if it had the same `merge` key (continuous
    /// edits such as dragging a slider or the gizmo).
    pub fn apply_merged(&mut self, op: EditOp, world: &mut World, merge: Option<String>) -> Result<Created, String> {
        let mut next = self.doc.clone();
        // Added actors (document ids) to report.
        let mut new_actors: Vec<u32> = Vec::new();
        let mut created = Created::default();
        match op {
            EditOp::ActorTransform { actors, transforms } => {
                let top = self.top_level(&actors);
                for (id, trs) in actors.iter().zip(&transforms).filter(|(id, _)| top.contains(id)) {
                    let local = self.parent_world(*id).inverse() * trs.to_mat4();
                    self.change_actor(&mut next, *id, ActorChange::Local(local))?;
                }
            }
            EditOp::ActorLocal { actors, transforms } => {
                for (id, trs) in actors.iter().zip(&transforms) {
                    self.change_actor(&mut next, *id, ActorChange::Local(trs.to_mat4()))?;
                }
            }
            EditOp::ComponentLocal { components, transforms } => {
                for (c, trs) in components.iter().zip(&transforms) {
                    self.change_component(&mut next, c.actor, &c.component, ComponentChange::Local(trs.to_mat4()))?;
                }
            }
            EditOp::ActorVisibility { actors, visible } => {
                for id in actors {
                    self.change_actor(&mut next, id, ActorChange::Visible(visible))?;
                }
            }
            EditOp::Rename { actor, name } => {
                let name = name.trim();
                if name.is_empty() {
                    return Err("an actor needs a name".into());
                }
                self.change_actor(&mut next, actor, ActorChange::Name(name.into()))?;
            }
            EditOp::Reparent { actors, parent } => {
                let parent_world = match parent {
                    Some(p) => self.actor(p).ok_or_else(|| format!("no actor {p}"))?.world,
                    None => Mat4::IDENTITY,
                };
                let parent_key = parent.map(|p| self.actors[&p].key.clone());
                for id in self.top_level(&actors) {
                    if parent.is_some_and(|p| self.is_ancestor(id, p)) {
                        return Err(format!("'{}' cannot go below itself", self.actors[&id].name));
                    }
                    let world_now = self.actors[&id].world;
                    self.change_actor(&mut next, id, ActorChange::Parent(parent_key.clone()))?;
                    self.change_actor(&mut next, id, ActorChange::Local(parent_world.inverse() * world_now))?;
                }
            }
            EditOp::Group { actors, name } => {
                let top = self.top_level(&actors);
                if top.is_empty() {
                    return Err("nothing to group".into());
                }
                let parents: Vec<Option<u32>> = top.iter().map(|id| self.actors[id].parent).collect();
                let common = if parents.iter().all(|p| *p == parents[0]) { parents[0] } else { None };
                let center = top.iter().map(|id| self.actors[id].world.w_axis.truncate()).sum::<Vec3>() / top.len() as f32;
                let group_world = Mat4::from_translation(center);
                let parent_world = common.map_or(Mat4::IDENTITY, |p| self.actors[&p].world);
                let id = next.take_id();
                next.added_actors.push(AddedActor {
                    id,
                    name: name.unwrap_or_else(|| "Group".into()),
                    parent: common.map(|p| self.actors[&p].key.clone()),
                    transform: Trs::from_mat4(&(parent_world.inverse() * group_world)),
                    visible: true,
                    components: Vec::new(),
                });
                for a in top {
                    let world_now = self.actors[&a].world;
                    self.change_actor(&mut next, a, ActorChange::Parent(Some(added_key(id))))?;
                    self.change_actor(&mut next, a, ActorChange::Local(group_world.inverse() * world_now))?;
                }
                new_actors.push(id);
            }
            EditOp::Ungroup { actors } => {
                for id in actors {
                    let Some(a) = self.actor(id) else { continue };
                    let (parent_key, parent_world, children, empty) = (a.parent_key.clone().filter(|k| self.actor_id(k).is_some()), self.parent_world(id), a.children.clone(), a.components.is_empty());
                    for c in children {
                        let world_now = self.actors[&c].world;
                        self.change_actor(&mut next, c, ActorChange::Parent(parent_key.clone()))?;
                        self.change_actor(&mut next, c, ActorChange::Local(parent_world.inverse() * world_now))?;
                    }
                    if empty {
                        self.change_actor(&mut next, id, ActorChange::Delete)?;
                    }
                }
            }
            EditOp::ComponentTransform { components, transforms } => {
                for (c, trs) in components.iter().zip(&transforms) {
                    let actor_world = self.actor(c.actor).ok_or_else(|| format!("no actor {}", c.actor))?.world;
                    self.change_component(&mut next, c.actor, &c.component, ComponentChange::Local(actor_world.inverse() * trs.to_mat4()))?;
                }
            }
            EditOp::ComponentVisibility { components, visible } => {
                for c in &components {
                    self.change_component(&mut next, c.actor, &c.component, ComponentChange::Visible(visible))?;
                }
            }
            EditOp::ComponentLight { component, def } => {
                let actor_world = self.actor(component.actor).ok_or_else(|| format!("no actor {}", component.actor))?.world;
                self.change_component(&mut next, component.actor, &component.component, ComponentChange::Light(local_light(actor_world, &def)))?;
            }
            EditOp::Transform { instances, transforms } => {
                for (&i, trs) in instances.iter().zip(&transforms) {
                    let (actor, c) = self.instance_target(i)?;
                    let a = &self.actors[&actor];
                    if a.components.len() == 1 {
                        let local = self.parent_world(actor).inverse() * trs.to_mat4() * a.components[0].local.inverse();
                        self.change_actor(&mut next, actor, ActorChange::Local(local))?;
                    } else {
                        let name = a.components[c].name.clone();
                        self.change_component(&mut next, actor, &name, ComponentChange::Local(a.world.inverse() * trs.to_mat4()))?;
                    }
                }
            }
            EditOp::Visibility { instances, visible } => {
                for &i in &instances {
                    let (actor, c) = self.instance_target(i)?;
                    if self.actors[&actor].components.len() == 1 {
                        self.change_actor(&mut next, actor, ActorChange::Visible(visible))?;
                    } else {
                        let name = self.actors[&actor].components[c].name.clone();
                        self.change_component(&mut next, actor, &name, ComponentChange::Visible(visible))?;
                    }
                }
            }
            EditOp::AssignMaterial { instances, components, actors, material } => {
                for (actor, name) in self.component_targets(&instances, &components, &actors)? {
                    self.change_component(&mut next, actor, &name, ComponentChange::Material(material.clone()))?;
                }
            }
            EditOp::CastShadow { instances, components, cast_shadow } => {
                for (actor, name) in self.component_targets(&instances, &components, &[])? {
                    self.change_component(&mut next, actor, &name, ComponentChange::CastShadow(cast_shadow))?;
                }
            }
            EditOp::Delete { actors, components, instances, lights, pcg } => {
                let mut whole = actors.clone();
                let mut parts: Vec<(u32, String)> = components.iter().map(|c| (c.actor, c.component.clone())).collect();
                for &i in &instances {
                    match self.owner(i) {
                        Owner::Actor { actor, component } => self.whole_or_part(actor, component as usize, &mut whole, &mut parts),
                        Owner::Pcg(volume) => next.pcg.retain(|v| v.id != volume),
                        Owner::Spare => {}
                    }
                }
                for &l in &lights {
                    let (actor, c) = self.light_component(l as usize).ok_or_else(|| format!("light {l} was deleted"))?;
                    self.whole_or_part(actor, c, &mut whole, &mut parts);
                }
                for root in self.top_level(&whole) {
                    for id in self.subtree(root) {
                        self.change_actor(&mut next, id, ActorChange::Delete)?;
                    }
                }
                for (actor, name) in parts.into_iter().filter(|(a, _)| !whole.iter().any(|&w| self.is_ancestor(w, *a))) {
                    self.change_component(&mut next, actor, &name, ComponentChange::Delete)?;
                }
                next.pcg.retain(|v| !pcg.contains(&v.id));
            }
            EditOp::Duplicate { actors, instances, lights, pcg, offset } => {
                let mut sources = actors.clone();
                for &i in &instances {
                    sources.push(self.instance_target(i)?.0);
                }
                for &l in &lights {
                    sources.push(self.light_component(l as usize).ok_or_else(|| format!("light {l} was deleted"))?.0);
                }
                let shift = Mat4::from_translation(Vec3::from(offset));
                for root in self.top_level(&sources) {
                    new_actors.push(self.duplicate(&mut next, world, root, shift)?);
                }
                for &v in &pcg {
                    let mut def = next.pcg.iter().find(|d| d.id == v).cloned().ok_or_else(|| format!("no PCG volume {v}"))?;
                    def.id = next.take_id();
                    def.name = format!("{} copy", def.name);
                    def.volume.center = (Vec3::from(def.volume.center) + Vec3::from(offset)).into();
                    created.pcg.push(def.id);
                    next.pcg.push(def);
                }
            }
            EditOp::Place { item, transform, parent } => {
                let parts = self.prefab(world.scene, &item).ok_or_else(|| format!("unknown content item '{item}'"))?;
                let label = super::content::items().into_iter().find(|i| i.id == item).map(|i| i.label).unwrap_or_else(|| item.rsplit(['/', ':']).next().unwrap_or(&item).to_string());
                let mut taken = std::collections::BTreeSet::new();
                let components = parts
                    .into_iter()
                    .map(|part| {
                        let short = part.mesh.rsplit('/').next().unwrap_or(&part.mesh).to_string();
                        let name = (1..).map(|n| if n == 1 { short.clone() } else { format!("{short} #{n}") }).find(|n| !taken.contains(n)).expect("free");
                        taken.insert(name.clone());
                        ComponentDef { name, transform: Trs::from_mat4(&part.transform), visible: true, kind: ComponentKindDef::Mesh { mesh: part.mesh, material: part.material, cast_shadow: part.cast_shadow } }
                    })
                    .collect();
                let parent_world = parent.and_then(|p| self.actor(p)).map_or(Mat4::IDENTITY, |p| p.world);
                let id = next.take_id();
                next.added_actors.push(AddedActor {
                    id,
                    name: label,
                    parent: parent.and_then(|p| self.actor(p)).map(|p| p.key.clone()),
                    transform: Trs::from_mat4(&(parent_world.inverse() * transform.to_mat4())),
                    visible: true,
                    components,
                });
                new_actors.push(id);
            }
            EditOp::Material { name, params } => {
                next.materials.insert(name, params);
            }
            EditOp::Graph { name, graph } => {
                if name.trim().is_empty() {
                    return Err("a material graph needs a name".into());
                }
                next.graphs.insert(name, graph);
            }
            EditOp::Light { light, def } => {
                let (actor, c) = self.light_component(light as usize).ok_or_else(|| format!("light {light} was deleted"))?;
                let a = &self.actors[&actor];
                let (name, local) = (a.components[c].name.clone(), local_light(a.world, &def));
                self.change_component(&mut next, actor, &name, ComponentChange::Light(local))?;
            }
            EditOp::AddLight { def } => {
                let id = next.take_id();
                let name = if def.shape == LightShape::Spot { "Spot light" } else { "Point light" };
                next.added_actors.push(AddedActor {
                    id,
                    name: name.into(),
                    parent: None,
                    transform: Trs::from_mat4(&Mat4::from_translation(Vec3::from(def.position))),
                    visible: true,
                    components: vec![ComponentDef { name: "light".into(), transform: Trs::IDENTITY, visible: true, kind: ComponentKindDef::Light { light: LightDef { position: [0.0; 3], ..def } } }],
                });
                new_actors.push(id);
            }
            EditOp::Environment { env } => next.environment = Some(env),
            EditOp::Render { key, value } => {
                super::settings::get(world.settings, &key).ok_or_else(|| format!("unknown render setting '{key}'"))?;
                next.render.insert(key, value);
            }
            EditOp::AddPcg { preset, center } => {
                let id = next.take_id();
                let def = pcg::new_volume(id, &preset, Vec3::from(center), world.floor).ok_or_else(|| format!("unknown PCG preset '{preset}'"))?;
                next.pcg.push(def);
                created.pcg.push(id);
            }
            EditOp::Pcg { def } => {
                let slot = next.pcg.iter_mut().find(|d| d.id == def.id).ok_or_else(|| format!("no PCG volume {}", def.id))?;
                *slot = def;
            }
        }
        self.commit_merged(next, world, merge);
        created.actors = new_actors.iter().filter_map(|&id| self.added_actor(id)).collect();
        for &a in &created.actors {
            created.instances.extend(self.subtree_instances(a).iter().map(|h| h.index()));
            for id in self.subtree(a) {
                for c in &self.actors[&id].components {
                    if let Kind::Light { slot: Some(i), .. } = c.kind {
                        created.lights.push(i as u32);
                    }
                }
            }
        }
        Ok(created)
    }

    /// The actor and component of instance `i` (scene index).
    fn instance_target(&self, i: u32) -> Result<(u32, usize), String> {
        match self.owner(i) {
            Owner::Actor { actor, component } => Ok((actor, component as usize)),
            Owner::Pcg(_) => Err("instances spawned by a PCG volume are edited through the volume".into()),
            Owner::Spare => Err(format!("instance {i} was deleted")),
        }
    }

    /// Component `c` of `actor` goes, or the whole actor if it has no other component.
    fn whole_or_part(&self, actor: u32, c: usize, whole: &mut Vec<u32>, parts: &mut Vec<(u32, String)>) {
        let a = &self.actors[&actor];
        if a.components.len() == 1 {
            whole.push(actor);
        } else {
            parts.push((actor, a.components[c].name.clone()));
        }
    }

    /// (actor, component name) of instances, components and every mesh component of actors.
    fn component_targets(&self, instances: &[u32], components: &[ComponentRef], actors: &[u32]) -> Result<Vec<(u32, String)>, String> {
        let mut out: Vec<(u32, String)> = components.iter().map(|c| (c.actor, c.component.clone())).collect();
        for &i in instances {
            let (actor, c) = self.instance_target(i)?;
            out.push((actor, self.actors[&actor].components[c].name.clone()));
        }
        for &id in actors {
            let a = self.actor(id).ok_or_else(|| format!("no actor {id}"))?;
            out.extend(a.components.iter().filter(|c| matches!(c.kind, Kind::Mesh { .. })).map(|c| (id, c.name.clone())));
        }
        Ok(out)
    }

    /// Copies actor `root` and everything below it as added actors, its world transform moved
    /// by `shift`; returns the copy's document id.
    fn duplicate(&mut self, next: &mut SceneDoc, world: &World, root: u32, shift: Mat4) -> Result<u32, String> {
        let mut keys: HashMap<u32, String> = HashMap::new();
        let mut root_id = 0;
        let mut skipped = 0;
        for id in self.subtree(root) {
            let a = &self.actors[&id];
            let doc_id = next.take_id();
            keys.insert(id, added_key(doc_id));
            let (parent, local) = if id == root {
                (a.parent.map(|p| self.actors[&p].key.clone()), self.parent_world(id).inverse() * shift * a.world)
            } else {
                (a.parent.and_then(|p| keys.get(&p).cloned()), a.local)
            };
            let mut components = Vec::new();
            for c in &a.components {
                let kind = match &c.kind {
                    Kind::Mesh { mesh, material, cast_shadow, .. } => {
                        let (mesh, material) = (world.scene.mesh_name(*mesh).to_string(), world.scene.material_name(*material).to_string());
                        if mesh.is_empty() || material.is_empty() {
                            return Err(format!("'{}' has an unnamed mesh or material and cannot be duplicated", a.name));
                        }
                        ComponentKindDef::Mesh { mesh, material, cast_shadow: *cast_shadow }
                    }
                    Kind::Light { def, .. } => ComponentKindDef::Light { light: def.clone() },
                    Kind::Volume { .. } | Kind::Particles { .. } => {
                        skipped += 1;
                        continue;
                    }
                };
                components.push(ComponentDef { name: c.name.clone(), transform: Trs::from_mat4(&c.local), visible: c.visible, kind });
            }
            next.added_actors.push(AddedActor { id: doc_id, name: a.name.clone(), parent, transform: Trs::from_mat4(&local), visible: a.visible, components });
            if id == root {
                root_id = doc_id;
            }
        }
        if skipped > 0 {
            log::warn!("duplicate of '{}': {skipped} volume / particle component(s) are not copied", self.actors[&root].name);
        }
        Ok(root_id)
    }

    fn change_actor(&self, doc: &mut SceneDoc, id: u32, change: ActorChange) -> Result<(), String> {
        let a = self.actor(id).ok_or_else(|| format!("no actor {id}"))?;
        if a.added {
            let doc_id: u32 = a.key[1..].parse().map_err(|_| format!("bad actor key '{}'", a.key))?;
            if let ActorChange::Delete = change {
                doc.added_actors.retain(|x| x.id != doc_id);
                return Ok(());
            }
            // Deleted earlier in this operation (e.g. with its parent).
            let Some(def) = doc.added_actors.iter_mut().find(|x| x.id == doc_id) else { return Ok(()) };
            match change {
                ActorChange::Local(m) => def.transform = Trs::from_mat4(&m),
                ActorChange::Visible(v) => def.visible = v,
                ActorChange::Name(n) => def.name = n,
                ActorChange::Parent(p) => def.parent = p,
                ActorChange::Delete => {}
            }
        } else {
            let e = doc.actors.entry(a.key.clone()).or_default();
            match change {
                ActorChange::Local(m) => e.transform = Some(Trs::from_mat4(&m)),
                ActorChange::Visible(v) => e.visible = Some(v),
                ActorChange::Name(n) => e.name = Some(n),
                ActorChange::Parent(p) => e.parent = Some(p.unwrap_or_default()),
                ActorChange::Delete => e.deleted = true,
            }
        }
        Ok(())
    }

    fn change_component(&self, doc: &mut SceneDoc, actor: u32, name: &str, change: ComponentChange) -> Result<(), String> {
        let a = self.actor(actor).ok_or_else(|| format!("no actor {actor}"))?;
        let c = a.components.iter().find(|c| c.name == name).ok_or_else(|| format!("'{}' has no component '{name}'", a.name))?;
        // A light's position is its definition's.
        let change = match (change, &c.kind) {
            (ComponentChange::Local(m), Kind::Light { def, .. }) => ComponentChange::Light(LightDef { position: m.w_axis.truncate().into(), ..def.clone() }),
            (change, _) => change,
        };
        if a.added {
            let doc_id: u32 = a.key[1..].parse().map_err(|_| format!("bad actor key '{}'", a.key))?;
            let Some(def) = doc.added_actors.iter_mut().find(|x| x.id == doc_id) else { return Ok(()) };
            if let ComponentChange::Delete = change {
                def.components.retain(|c| c.name != name);
                return Ok(());
            }
            let Some(cd) = def.components.iter_mut().find(|c| c.name == name) else { return Ok(()) };
            match (change, &mut cd.kind) {
                (ComponentChange::Local(m), _) => cd.transform = Trs::from_mat4(&m),
                (ComponentChange::Visible(v), _) => cd.visible = v,
                (ComponentChange::Material(m), ComponentKindDef::Mesh { material, .. }) => *material = m,
                (ComponentChange::CastShadow(s), ComponentKindDef::Mesh { cast_shadow, .. }) => *cast_shadow = s,
                (ComponentChange::Light(l), ComponentKindDef::Light { light }) => *light = l,
                _ => return Err(format!("component '{name}' has no such property")),
            }
        } else {
            let e = doc.actors.entry(a.key.clone()).or_default().components.entry(name.to_string()).or_default();
            match change {
                ComponentChange::Local(m) => e.transform = Some(Trs::from_mat4(&m)),
                ComponentChange::Visible(v) => e.visible = Some(v),
                ComponentChange::Material(m) => e.material = Some(m),
                ComponentChange::CastShadow(s) => e.cast_shadow = Some(s),
                ComponentChange::Light(l) => e.light = Some(l),
                ComponentChange::Delete => e.deleted = true,
            }
        }
        Ok(())
    }
}
