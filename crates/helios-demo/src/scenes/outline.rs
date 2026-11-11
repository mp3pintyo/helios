//! The scene builders' outline: which instances, lights and emitters make one actor (a tree's
//! trunk and foliage, the campfire's stones, logs, coals, light and fire), in folders. The
//! editor turns it into its actors (`editor::actors`); whatever a builder leaves out becomes an
//! actor of its own.
//!
//! Builders call [`begin`] / [`end`] around the instances of one actor and [`folder`] /
//! [`end_folder`] around groups of actors. Outside [`scoped`] (tests building a scene directly)
//! every call is a no-op.
//!
//! **Keys.** Every actor gets a key that scene files address it by: its folder path, its name
//! and, for actors (not folders), its pivot's position in centimetres
//! (`Forest/Trees/Conifer@1250,-30,-4510`). Adding or removing other actors does not change a
//! key, unlike the build order the first scene files used; a builder that moves an object
//! changes its key, and edits of it are reported as not applying.

use std::cell::RefCell;

use glam::{Mat4, Vec3};
use helios_render::{LightHandle, Scene};

/// One actor (or folder) as a builder made it.
#[derive(Clone, Debug)]
pub struct OutlineActor {
    pub key: String,
    pub name: String,
    /// Index of the parent folder or actor in [`Outline::actors`].
    pub parent: Option<usize>,
    /// World transform of the actor's pivot (identity for folders).
    pub pivot: Mat4,
    pub folder: bool,
    pub instances: Vec<u32>,
    pub lights: Vec<u32>,
    pub volumes: Vec<u32>,
    pub particles: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Outline {
    pub actors: Vec<OutlineActor>,
}

/// An actor being built ([`begin`] → [`end`]).
#[derive(Clone, Copy, Debug)]
pub struct Token {
    index: usize,
    counts: [usize; 4],
}

#[derive(Default)]
struct Recorder {
    outline: Outline,
    /// Open folders (indices into the outline).
    folders: Vec<usize>,
}

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

fn with<R>(f: impl FnOnce(&mut Recorder) -> R) -> Option<R> {
    RECORDER.with(|r| r.borrow_mut().as_mut().map(f))
}

/// Runs a scene builder and returns its outline.
pub(crate) fn scoped<R>(build: impl FnOnce() -> R) -> (R, Outline) {
    let previous = RECORDER.with(|r| r.borrow_mut().replace(Recorder::default()));
    let result = build();
    let recorder = RECORDER.with(|r| std::mem::replace(&mut *r.borrow_mut(), previous)).unwrap_or_default();
    let mut outline = recorder.outline;
    assign_keys(&mut outline);
    (result, outline)
}

fn counts(scene: &Scene) -> [usize; 4] {
    [scene.instance_count(), scene.lights().len(), scene.volumes.len(), scene.particles.len()]
}

/// Opens a folder: actors begun until [`end_folder`] go into it.
pub(crate) fn folder(name: &str) {
    with(|r| {
        let parent = r.folders.last().copied();
        r.outline.actors.push(OutlineActor { key: String::new(), name: name.into(), parent, pivot: Mat4::IDENTITY, folder: true, instances: vec![], lights: vec![], volumes: vec![], particles: vec![] });
        r.folders.push(r.outline.actors.len() - 1);
    });
}

pub(crate) fn end_folder() {
    with(|r| r.folders.pop());
}

/// Starts an actor named `name` with its pivot at `pivot`: the instances, lights, volumes and
/// particle emitters added to the scene until [`end`] are its components.
pub(crate) fn begin(scene: &Scene, name: &str, pivot: Mat4) -> Option<Token> {
    with(|r| {
        let parent = r.folders.last().copied();
        r.outline.actors.push(OutlineActor { key: String::new(), name: name.into(), parent, pivot, folder: false, instances: vec![], lights: vec![], volumes: vec![], particles: vec![] });
        Token { index: r.outline.actors.len() - 1, counts: counts(scene) }
    })
}

/// Ends an actor started with [`begin`].
pub(crate) fn end(scene: &Scene, token: Option<Token>) {
    let Some(token) = token else { return };
    let now = counts(scene);
    with(|r| {
        let actor = &mut r.outline.actors[token.index];
        let range = |k: usize| (token.counts[k] as u32..now[k] as u32).collect::<Vec<u32>>();
        actor.instances.extend(range(0));
        actor.lights.extend(range(1));
        actor.volumes.extend(range(2));
        actor.particles.extend(range(3));
    });
}

/// Adds lights made later (after other actors) to the actor of `token`.
pub(crate) fn add_lights(token: Option<Token>, lights: &[LightHandle]) {
    let Some(token) = token else { return };
    with(|r| r.outline.actors[token.index].lights.extend(lights.iter().map(|l| l.index())));
}

/// [`begin`] + `build` + [`end`].
pub(crate) fn actor<R>(scene: &mut Scene, name: &str, pivot: Mat4, build: impl FnOnce(&mut Scene) -> R) -> R {
    let token = begin(scene, name, pivot);
    let result = build(scene);
    end(scene, token);
    result
}

/// A key segment for a position: centimetres, rounded.
pub fn position_key(p: Vec3) -> String {
    let cm = |v: f32| (v * 100.0).round() as i64;
    format!("{},{},{}", cm(p.x), cm(p.y), cm(p.z))
}

/// Keys from folder paths, names and pivots; duplicates get ` #2`, ` #3`…
fn assign_keys(outline: &mut Outline) {
    let mut taken = std::collections::HashSet::new();
    for i in 0..outline.actors.len() {
        let a = &outline.actors[i];
        let own = if a.folder { a.name.clone() } else { format!("{}@{}", a.name, position_key(a.pivot.w_axis.truncate())) };
        let base = match a.parent {
            Some(p) => format!("{}/{own}", outline.actors[p].key),
            None => own,
        };
        let key = (1..).map(|n| if n == 1 { base.clone() } else { format!("{base} #{n}") }).find(|k| !taken.contains(k)).expect("a free key");
        taken.insert(key.clone());
        outline.actors[i].key = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helios_geometry::VirtualMesh;
    use helios_render::{Light, Material};

    #[test]
    fn actors_collect_what_was_added_between_begin_and_end() {
        let ((), outline) = scoped(|| {
            let mut scene = Scene::new();
            let mesh = scene.add_mesh(VirtualMesh::default());
            let material = scene.add_material(Material::default());
            scene.add_instance(mesh, material, Mat4::IDENTITY);
            folder("Trees");
            for x in [1.0f32, 2.0] {
                let at = Mat4::from_translation(Vec3::X * x);
                actor(&mut scene, "Tree", at, |scene| {
                    scene.add_instance(mesh, material, at);
                    scene.add_instance(mesh, material, at);
                });
            }
            end_folder();
            let fire = begin(&scene, "Fire", Mat4::IDENTITY);
            scene.add_instance(mesh, material, Mat4::IDENTITY);
            end(&scene, fire);
            let light = scene.add_light(Light::point(Vec3::Y, Vec3::ONE, 1.0, 4.0));
            add_lights(fire, &[light]);
        });
        let a = &outline.actors;
        assert_eq!(a.len(), 4);
        assert_eq!((a[0].key.as_str(), a[0].folder), ("Trees", true));
        assert_eq!(a[1].key, "Trees/Tree@100,0,0");
        assert_eq!((a[1].parent, &a[1].instances), (Some(0), &vec![1, 2]));
        assert_eq!(a[2].instances, vec![3, 4]);
        assert_eq!((a[3].parent, &a[3].instances, &a[3].lights), (None, &vec![5], &vec![0]));
        // Outside a recording nothing is collected.
        assert!(begin(&Scene::new(), "x", Mat4::IDENTITY).is_none());
    }

    #[test]
    fn duplicate_keys_are_numbered() {
        let ((), outline) = scoped(|| {
            let scene = Scene::new();
            for _ in 0..3 {
                let t = begin(&scene, "Rock", Mat4::IDENTITY);
                end(&scene, t);
            }
        });
        let keys: Vec<_> = outline.actors.iter().map(|a| a.key.as_str()).collect();
        assert_eq!(keys, ["Rock@0,0,0", "Rock@0,0,0 #2", "Rock@0,0,0 #3"]);
    }
}
