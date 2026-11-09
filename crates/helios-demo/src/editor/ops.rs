//! Edit operations: what the editor UI asks for, turned into a new document (one undo step).
//!
//! Instances and lights are addressed by their scene indices (what picking and the outliner
//! show), PCG volumes by id; each operation maps them to the document's keys — base instance
//! index, added instance id, added light id.

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::doc::*;
use super::{light_def, pcg, Editor, Owner, World};

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum EditOp {
    /// Sets instance transforms (one per instance).
    Transform { instances: Vec<u32>, transforms: Vec<Trs> },
    Visibility { instances: Vec<u32>, visible: bool },
    AssignMaterial { instances: Vec<u32>, material: String },
    CastShadow { instances: Vec<u32>, cast_shadow: bool },
    Delete {
        #[serde(default)]
        instances: Vec<u32>,
        #[serde(default)]
        lights: Vec<u32>,
        #[serde(default)]
        pcg: Vec<u32>,
    },
    Duplicate {
        #[serde(default)]
        instances: Vec<u32>,
        #[serde(default)]
        lights: Vec<u32>,
        #[serde(default)]
        pcg: Vec<u32>,
        offset: [f32; 3],
    },
    /// Places a content item (engine prefab, imported model, `mesh:NAME`) at a transform.
    Place { item: String, transform: Trs },
    Material { name: String, params: MaterialParams },
    Light { light: u32, def: LightDef },
    AddLight { def: LightDef },
    Environment { env: Environment },
    Render { key: String, value: Value },
    AddPcg { preset: String, center: [f32; 3] },
    Pcg { def: PcgVolumeDef },
}

/// What an operation created (scene instance and light indices, PCG volume ids), so the UI
/// can select it.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Created {
    pub instances: Vec<u32>,
    pub lights: Vec<u32>,
    pub pcg: Vec<u32>,
}

impl Editor {
    /// Applies `op` as one undo step.
    pub fn apply(&mut self, op: EditOp, world: &mut World) -> Result<Created, String> {
        self.apply_merged(op, world, None)
    }

    /// Applies `op`, joining the previous undo step if it had the same `merge` key (continuous
    /// edits such as dragging a slider or a light).
    pub fn apply_merged(&mut self, op: EditOp, world: &mut World, merge: Option<String>) -> Result<Created, String> {
        let mut next = self.doc.clone();
        let mut new_instances = Vec::new();
        let mut new_lights = Vec::new();
        let mut created = Created::default();
        match op {
            EditOp::Transform { instances, transforms } => {
                for (&i, trs) in instances.iter().zip(&transforms) {
                    self.edit_instance(&mut next, world, i, |e| e.transform = Some(*trs), |a| a.transform = *trs)?;
                }
            }
            EditOp::Visibility { instances, visible } => {
                for &i in &instances {
                    self.edit_instance(&mut next, world, i, |e| e.visible = Some(visible), |_| {})?;
                }
            }
            EditOp::AssignMaterial { instances, material } => {
                for &i in &instances {
                    self.edit_instance(&mut next, world, i, |e| e.material = Some(material.clone()), |a| a.material = material.clone())?;
                }
            }
            EditOp::CastShadow { instances, cast_shadow } => {
                for &i in &instances {
                    match self.owner(i) {
                        Owner::Added(id) => next.added.iter_mut().filter(|a| a.id == id).for_each(|a| a.cast_shadow = cast_shadow),
                        _ => return Err("only added instances can change their shadow casting".into()),
                    }
                }
            }
            EditOp::Delete { instances, lights, pcg } => {
                for &i in &instances {
                    match self.owner(i) {
                        Owner::Base(_) => self.edit_instance(&mut next, world, i, |e| e.deleted = true, |_| {})?,
                        Owner::Added(id) => next.added.retain(|a| a.id != id),
                        Owner::Pcg(volume) => next.pcg.retain(|v| v.id != volume),
                        Owner::Spare => {}
                    }
                }
                for &l in &lights {
                    self.edit_light(&mut next, world, l, |d| d.enabled = false, true)?;
                }
                next.pcg.retain(|v| !pcg.contains(&v.id));
            }
            EditOp::Duplicate { instances, lights, pcg, offset } => {
                let shift = Mat4::from_translation(Vec3::from(offset));
                for &i in &instances {
                    let (mesh, material, transform, cast_shadow) = self.describe_instance(world, i)?;
                    let id = next.take_id();
                    next.added.push(AddedInstance { id, mesh, material, transform: Trs::from_mat4(&(shift * transform)), cast_shadow });
                    new_instances.push(id);
                }
                for &l in &lights {
                    let light = world.scene.lights().get(l as usize).ok_or_else(|| format!("no light {l}"))?;
                    let mut def = light_def(light);
                    def.position = (Vec3::from(def.position) + Vec3::from(offset)).into();
                    let id = next.take_id();
                    next.added_lights.push(AddedLight { id, light: def });
                    new_lights.push(id);
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
            EditOp::Place { item, transform } => {
                let parts = self.prefab(world.scene, &item).ok_or_else(|| format!("unknown content item '{item}'"))?;
                let placement = transform.to_mat4();
                for part in parts {
                    let id = next.take_id();
                    let transform = Trs::from_mat4(&(placement * part.transform));
                    next.added.push(AddedInstance { id, mesh: part.mesh, material: part.material, transform, cast_shadow: part.cast_shadow });
                    new_instances.push(id);
                }
            }
            EditOp::Material { name, params } => {
                next.materials.insert(name, params);
            }
            EditOp::Light { light, def } => self.edit_light(&mut next, world, light, |d| *d = def.clone(), false)?,
            EditOp::AddLight { def } => {
                let id = next.take_id();
                next.added_lights.push(AddedLight { id, light: def });
                new_lights.push(id);
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
        created.instances = new_instances.iter().filter_map(|&id| self.added_handle(id)).map(|h| h.index()).collect();
        created.lights = new_lights.iter().filter_map(|&id| self.added_light_index(id)).map(|i| i as u32).collect();
        Ok(created)
    }

    /// Edits instance `i` in `doc`: a base instance's overrides or an added instance.
    fn edit_instance(&self, doc: &mut SceneDoc, world: &World, i: u32, base: impl FnOnce(&mut InstanceEdit), added: impl FnOnce(&mut AddedInstance)) -> Result<(), String> {
        match self.owner(i) {
            Owner::Base(index) => {
                let mesh = world.scene.mesh_name(world.scene.instance_mesh(world.scene.instance_handle(i).ok_or("no such instance")?)).to_string();
                base(doc.instances.entry(index).or_insert_with(|| InstanceEdit { mesh, ..Default::default() }));
                Ok(())
            }
            Owner::Added(id) => {
                let entry = doc.added.iter_mut().find(|a| a.id == id).ok_or_else(|| format!("added instance {id} is gone"))?;
                added(entry);
                Ok(())
            }
            Owner::Pcg(_) => Err("instances spawned by a PCG volume are edited through the volume".into()),
            Owner::Spare => Err(format!("instance {i} was deleted")),
        }
    }

    /// Edits light `l` (scene index): a base light's definition or an added light; `remove`
    /// drops an added light instead.
    fn edit_light(&self, doc: &mut SceneDoc, world: &World, l: u32, edit: impl FnOnce(&mut LightDef), remove: bool) -> Result<(), String> {
        let light = world.scene.lights().get(l as usize).ok_or_else(|| format!("no light {l}"))?;
        if (l as usize) < self.base_light_count() {
            let def = doc.lights.entry(l).or_insert_with(|| light_def(light));
            edit(def);
            return Ok(());
        }
        let id = self.doc.added_lights.iter().map(|a| a.id).find(|&id| self.added_light_index(id) == Some(l as usize)).ok_or_else(|| format!("light {l} was deleted"))?;
        if remove {
            doc.added_lights.retain(|a| a.id != id);
        } else if let Some(entry) = doc.added_lights.iter_mut().find(|a| a.id == id) {
            edit(&mut entry.light);
        }
        Ok(())
    }

    /// Mesh name, material name, transform and shadow casting of an instance as the document
    /// has it (for duplicating it).
    fn describe_instance(&self, world: &World, i: u32) -> Result<(String, String, Mat4, bool), String> {
        let handle = world.scene.instance_handle(i).ok_or_else(|| format!("no instance {i}"))?;
        let mesh = world.scene.mesh_name(world.scene.instance_mesh(handle)).to_string();
        let material = world.scene.material_name(world.scene.instance_material(handle)).to_string();
        if mesh.is_empty() || material.is_empty() {
            return Err(format!("instance {i} has an unnamed mesh or material and cannot be duplicated"));
        }
        let transform = self.effective_transform(world.scene, i).ok_or("no transform")?;
        Ok((mesh, material, transform, world.scene.casts_shadow(handle)))
    }
}
