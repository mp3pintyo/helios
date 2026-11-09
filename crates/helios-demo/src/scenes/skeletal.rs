//! Skeletal animation of an imported model: an [`AnimationPlayer`] poses the glTF node hierarchy
//! every frame and the joint palettes of its skins go to the renderer's skinned instances.

use std::collections::HashMap;

use glam::{Mat4, Vec4};
use helios_asset::animation::{global_matrices, hierarchy_order, rest_pose};
use helios_asset::{AnimationClip, AnimationPlayer, Asset, Node, NodeTransform, Skin};
use helios_geometry::{Aabb, Sphere};
use helios_render::{Scene, SkinHandle};

/// Seconds a clip change blends from the old clip's pose to the new one.
const CROSSFADE: f32 = 0.4;

/// The renderer skin for each (glTF skin, mesh node transform) pair: glTF poses skinned vertices
/// with the joints alone, and the renderer's palette is relative to the node that places the mesh
/// (`Skin::joint_matrices`), so meshes sharing a skin share a palette only if their nodes agree.
#[derive(Clone, Debug, Default)]
pub struct SkinBindings {
    /// (asset skin, mesh node world transform) per renderer skin, in the scene's skin order.
    pub skins: Vec<(usize, Mat4)>,
    /// Asset instance → index into `skins`.
    pub instance_skin: HashMap<usize, usize>,
}

impl SkinBindings {
    pub fn of(asset: &Asset) -> Self {
        let mut bindings = Self::default();
        for (i, instance) in asset.instances.iter().enumerate() {
            let Some(skin) = instance.skin else { continue };
            let key = (skin, instance.transform);
            let index = match bindings.skins.iter().position(|&k| k == key) {
                Some(index) => index,
                None => {
                    bindings.skins.push(key);
                    bindings.skins.len() - 1
                }
            };
            bindings.instance_skin.insert(i, index);
        }
        bindings
    }

    /// Per renderer skin and joint, the rest-space sphere of the vertices the joint influences in
    /// every mesh bound to it (radius -1: none).
    pub fn joint_bounds(&self, asset: &Asset) -> Vec<Vec<Vec4>> {
        let mut bounds: Vec<Vec<Option<Sphere>>> = self.skins.iter().map(|&(skin, _)| vec![None; asset.skins[skin].joints.len()]).collect();
        for (&instance, &binding) in &self.instance_skin {
            let mesh = &asset.meshes[asset.instances[instance].mesh].mesh;
            let joints = bounds[binding].len();
            for (b, sphere) in bounds[binding].iter_mut().zip(helios_geometry::skin::joint_bounds(mesh, joints)) {
                *b = match (*b, sphere) {
                    (Some(a), Some(s)) => Some(a.merge(s)),
                    (a, s) => a.or(s),
                };
            }
        }
        bounds.iter().map(|b| helios_geometry::skin::joint_bounds_vec4(b)).collect()
    }
}

/// Plays a model's animation clips on its skinned instances.
pub struct SkeletalAnimation {
    nodes: Vec<Node>,
    order: Vec<usize>,
    rest: Vec<NodeTransform>,
    clips: Vec<AnimationClip>,
    skins: Vec<Skin>,
    /// Renderer skin, its asset skin and the mesh node transform it is relative to.
    bindings: Vec<(SkinHandle, usize, Mat4)>,
    pub player: AnimationPlayer,
}

impl SkeletalAnimation {
    /// The model's animation, playing its first clip (if it has any: otherwise the skins hold
    /// the node hierarchy's rest pose); `None` without skinned instances.
    pub fn new(asset: &Asset, bindings: Vec<(SkinHandle, usize, Mat4)>) -> Option<Self> {
        if bindings.is_empty() {
            return None;
        }
        let order = hierarchy_order(&asset.nodes);
        log::info!(
            "animation: {} clips ({}), {} skins, {} joints",
            asset.animations.len(),
            asset.animations.iter().map(|c| format!("{} {:.1} s", c.name, c.duration)).collect::<Vec<_>>().join(", "),
            bindings.len(),
            bindings.iter().map(|b| asset.skins[b.1].joints.len()).sum::<usize>()
        );
        Some(Self {
            rest: rest_pose(&asset.nodes),
            nodes: asset.nodes.clone(),
            order,
            clips: asset.animations.clone(),
            skins: asset.skins.clone(),
            bindings,
            player: AnimationPlayer::new(0),
        })
    }

    /// Advances the clock and poses every skin.
    pub fn update(&mut self, scene: &mut Scene, dt: f32) {
        self.player.advance(&self.clips, dt);
        let pose = self.player.pose(&self.clips, &self.rest);
        let globals = global_matrices(&self.nodes, &self.order, &pose);
        for &(handle, skin, bind) in &self.bindings {
            scene.set_skin_pose(handle, &self.skins[skin].joint_matrices(&globals, bind));
        }
    }

    /// The `Minus` key: the next clip (crossfading); after the last one the animation pauses, and
    /// the next press starts the first clip again.
    pub fn cycle(&mut self) {
        let next = self.player.clip() + 1;
        if self.clips.is_empty() {
            self.player.paused = !self.player.paused;
        } else if self.player.paused {
            self.player.paused = false;
            self.player.play(0, CROSSFADE);
        } else if next < self.clips.len() {
            self.player.play(next, CROSSFADE);
        } else {
            self.player.paused = true;
        }
    }

    /// One HUD line: the clip, its time, and whether it is paused.
    pub fn status(&self) -> String {
        let Some(clip) = self.clips.get(self.player.clip()) else { return "animation: rest pose (no clips)".into() };
        let state = if self.player.paused { " (paused)" } else { "" };
        format!("animation: {} ({}/{}) {:.2}/{:.2} s{state} · - next clip / pause", clip.name, self.player.clip() + 1, self.clips.len(), self.player.time(), clip.duration)
    }
}

/// World bounds of the asset with its skinned meshes posed at the start of its first clip (glTF
/// skinned vertices go where their joints put them, which need not be near the mesh node).
pub fn posed_bounds(asset: &Asset, bindings: &SkinBindings) -> Aabb {
    if bindings.skins.is_empty() {
        return asset.bounds();
    }
    let mut pose = rest_pose(&asset.nodes);
    if let Some(clip) = asset.animations.first() {
        clip.apply(0.0, &mut pose);
    }
    let globals = global_matrices(&asset.nodes, &hierarchy_order(&asset.nodes), &pose);
    let palettes: Vec<Vec<Mat4>> = bindings.skins.iter().map(|&(skin, bind)| asset.skins[skin].joint_matrices(&globals, bind)).collect();
    let mut points = Vec::new();
    for (i, instance) in asset.instances.iter().enumerate() {
        let mesh = &asset.meshes[instance.mesh].mesh;
        for (k, v) in mesh.vertices.iter().enumerate() {
            let p = match bindings.instance_skin.get(&i) {
                Some(&b) => helios_geometry::skin::skin_point(&mesh.skin[k], &palettes[b], v.position()),
                None => v.position(),
            };
            points.push(instance.transform.transform_point3(p));
        }
    }
    Aabb::from_points(points)
}
