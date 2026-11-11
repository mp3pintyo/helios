//! A temperate forest at golden hour: broadleaf and conifer trees with real leaf geometry,
//! bushes, ferns and grass over a leaf-litter forest floor, mossy boulders, fallen logs and
//! mushrooms. Thousands of instances; the canopy shadows come from virtual shadow maps and the
//! green bounce light under the canopy from Lumen. A gusty breeze sways the trees and ripples
//! through the grass and ferns. The foliage has voxel LODs (Nanite Foliage), so distant canopies
//! keep their density; the conifers are assemblies of shared needle sprays (their foliage bakes
//! in a quarter of the time and memory of whole-tree meshes).

use glam::{Mat4, Quat, Vec2, Vec3};
use helios_geometry::modeling::{lathe, merge, tube};
use helios_geometry::noise::fbm;
use helios_geometry::assembly::build_plant_assembly_base;
use helios_geometry::vegetation::{self, LeafShape, PlantAssembly, TreeParams};
use helios_geometry::{primitives, BuildConfig, Mesh, SdfMode, VirtualMesh, VoxelConfig};
use helios_render::{
    AssemblyPart, Camera, Light, Material, MaterialHandle, MaterialKind, MeshHandle, ParticleEmitter, Scene, Sun, TextureDesc, TextureMapping,
    TexturePattern, VolumeEmitter, Wind, WindResponse,
};

use super::outline;
use super::{add_baked_with, add_built, smoothstep, Detail, Rng, SceneSetup};
use crate::mesh_store;

const SIZE: f32 = 900.0;
/// Grid cells per side of the terrain mesh at medium and high detail.
const TERRAIN_RESOLUTION: (u32, u32) = (384, 768);
const SUN: (f32, f32, f32) = (2.5, 0.26, 9.0);

pub fn ground_height(x: f32, z: f32) -> f32 {
    let p = Vec3::new(x, 0.0, z) * 0.006;
    let hills = fbm(p, 5, 77) * 22.0 + fbm(p * 4.0, 3, 78) * 2.5;
    // A gentle hollow where the camera starts.
    let hollow = 1.0 - smoothstep(10.0, 60.0, Vec2::new(x, z).length());
    hills - hollow * 2.0
}

/// Height of the terrain *mesh* at (x, z): [`ground_height`] sampled at the vertices of its
/// `resolution`² grid (`primitives::terrain`, which the scene deforms onto `ground_height`) and
/// interpolated over the grid triangle that holds (x, z), exactly as the mesh is rasterised. On a
/// 1.2–2.3 m grid the mesh lies centimetres to decimetres below the analytic height on every
/// crest: plants placed at `ground_height` float there.
fn terrain_mesh_height(x: f32, z: f32, resolution: u32) -> f32 {
    let vertex = |i: i64, j: i64| {
        let uv = Vec2::new(i as f32, j as f32) / resolution as f32;
        let xz = (uv - 0.5) * SIZE;
        ground_height(xz.x, xz.y)
    };
    let cell = (Vec2::new(x, z) / SIZE + 0.5) * resolution as f32;
    let (i, j) = (cell.x.floor() as i64, cell.y.floor() as i64);
    let (fx, fz) = (cell.x - i as f32, cell.y - j as f32);
    // Each quad is split along the diagonal from (i + 1, j) to (i, j + 1).
    if fx + fz <= 1.0 {
        let h = vertex(i, j);
        h + fx * (vertex(i + 1, j) - h) + fz * (vertex(i, j + 1) - h)
    } else {
        let h = vertex(i + 1, j + 1);
        h + (1.0 - fx) * (vertex(i, j + 1) - h) + (1.0 - fz) * (vertex(i + 1, j) - h)
    }
}

/// Forest density in [0, 1] (clearings where low).
fn density(x: f32, z: f32) -> f32 {
    let n = fbm(Vec3::new(x, 0.0, z) * 0.012, 3, 91);
    let clearing = smoothstep(8.0, 22.0, Vec2::new(x - 6.0, z + 3.0).length());
    smoothstep(-0.25, 0.25, n) * clearing
}

struct Meshes<'a> {
    scene: &'a mut Scene,
}

impl Meshes<'_> {
    fn add(&mut self, name: &str, mesh: &Mesh, sdf: bool) -> MeshHandle {
        add_baked_with(self.scene, name, mesh, Self::config(sdf))
    }

    /// Aggregate geometry (leaves, needles, fronds, blades): voxel LODs at a distance.
    fn add_foliage(&mut self, name: &str, mesh: &Mesh, sdf: bool) -> MeshHandle {
        add_baked_with(self.scene, name, mesh, Self::foliage_config(sdf))
    }

    /// A tree's foliage as an assembly: its sprays are the parts, the base (with the distance
    /// field Lumen sees) is built from their roots.
    fn add_assembly(&mut self, name: &str, plant: &PlantAssembly) -> MeshHandle {
        // The base is built from the sprays' baked meshes: a scene-bake worker bakes them all.
        let (parts, base) = mesh_store::unit(|| {
            let parts: Vec<MeshHandle> = plant.sprays.iter().map(|spray| self.add_foliage(&format!("{name} spray"), spray, false)).collect();
            let meshes: Vec<VirtualMesh> = parts.iter().map(|&p| self.scene.mesh(p).clone()).collect();
            let cost = plant.sprays.iter().map(|s| s.triangle_count() as u64).sum();
            // The parts hang from the plant's bones (the base follows them too).
            let base = add_built(self.scene, name, cost, || {
                let geometry: Vec<(&VirtualMesh, &Mesh, Mat4, u16)> =
                    plant.placements.iter().zip(&plant.part_bones).map(|(&(i, transform), &bone)| (&meshes[i], &plant.sprays[i], transform, bone)).collect();
                build_plant_assembly_base(&geometry, &plant.wood.bones, &Self::foliage_config(true)).0
            });
            (parts, base)
        });
        let parts = plant.placements.iter().zip(&plant.part_bones).map(|(&(i, transform), &bone)| AssemblyPart { mesh: parts[i], transform, bone: Some(bone) }).collect();
        self.scene.set_assembly(base, parts);
        base
    }

    fn foliage_config(sdf: bool) -> BuildConfig {
        BuildConfig { voxels: Some(VoxelConfig::default()), ..Self::config(sdf) }
    }

    fn config(sdf: bool) -> BuildConfig {
        BuildConfig { sdf: if sdf { SdfMode::Volume { resolution: 40 } } else { SdfMode::None }, ..Default::default() }
    }
}

/// Places one scattered object at (x, z).
type Placer<'a> = dyn FnMut(&mut Scene, f32, f32, &mut Rng) + 'a;

struct Species {
    name: &'static str,
    wood: MeshHandle,
    foliage: MeshHandle,
    bark: MaterialHandle,
    leaves: MaterialHandle,
}

fn build_scene(detail: Detail) -> Scene {
    let mut scene = Scene::new();
    scene.sun = Sun::from_angles(SUN.0, SUN.1, SUN.2);
    scene.ground_albedo = Vec3::new(0.1, 0.12, 0.05);
    scene.clouds = helios_render::CloudLayer::cumulus(0.4);
    // Morning haze under the canopy, thicker in hollows, broken into banks by noise.
    scene.fog = helios_render::FogVolume {
        density: 0.01,
        height_falloff: 0.06,
        base_height: ground_height(0.0, 0.0) - 2.0,
        anisotropy: 0.65,
        noise: 0.7,
        distance: 180.0,
        ..Default::default()
    };
    scene.wind = Wind::breeze(1.0);

    // ---- Textures and materials -------------------------------------------------------------------
    let t = |scene: &mut Scene, d: TextureDesc| scene.add_texture(d);
    let litter = t(&mut scene, TextureDesc::new(TexturePattern::ForestFloor, Vec3::new(0.3, 0.17, 0.06), Vec3::new(0.09, 0.065, 0.04)).roughness(0.85, 0.9).bump(0.3).seed(71));
    let moss = t(&mut scene, TextureDesc::new(TexturePattern::Moss, Vec3::new(0.1, 0.15, 0.04), Vec3::new(0.07, 0.1, 0.03)).roughness(0.95, 0.9).scale(0.5).bump(0.2).seed(72));
    let dirt = t(&mut scene, TextureDesc::new(TexturePattern::Dirt, Vec3::new(0.17, 0.12, 0.08), Vec3::new(0.2, 0.18, 0.15)).roughness(0.95, 0.8).bump(0.25).seed(73));
    let rock = t(&mut scene, TextureDesc::new(TexturePattern::Rock, Vec3::new(0.3, 0.29, 0.27), Vec3::new(0.14, 0.14, 0.13)).roughness(0.75, 0.9).seed(74));
    let oak_bark = t(&mut scene, TextureDesc::new(TexturePattern::Bark, Vec3::new(0.24, 0.21, 0.17), Vec3::new(0.06, 0.05, 0.04)).roughness(0.9, 0.95).param(0.6).bump(1.5).seed(75));
    let pine_bark = t(&mut scene, TextureDesc::new(TexturePattern::Bark, Vec3::new(0.2, 0.12, 0.08), Vec3::new(0.06, 0.04, 0.03)).roughness(0.85, 0.95).param(0.2).scale(1.4).bump(1.8).seed(76));
    let leaf = t(&mut scene, TextureDesc::new(TexturePattern::Leaf, Vec3::new(0.13, 0.25, 0.04), Vec3::new(0.22, 0.28, 0.05)).roughness(0.55, 0.6).seed(77));
    let needle = t(&mut scene, TextureDesc::new(TexturePattern::Leaf, Vec3::new(0.035, 0.085, 0.03), Vec3::new(0.06, 0.12, 0.04)).scale(3.0).roughness(0.6, 0.6).seed(78));
    let blade = t(&mut scene, TextureDesc::new(TexturePattern::Leaf, Vec3::new(0.1, 0.18, 0.035), Vec3::new(0.16, 0.22, 0.05)).roughness(0.6, 0.6).seed(79));
    let fern_tex = t(&mut scene, TextureDesc::new(TexturePattern::Leaf, Vec3::new(0.1, 0.24, 0.04), Vec3::new(0.16, 0.3, 0.05)).roughness(0.55, 0.6).seed(80));

    let terrain_material = scene.add_material(Material { kind: MaterialKind::Terrain, base_color: Vec3::ONE, layers: [Some(litter), Some(moss), Some(rock), Some(dirt)], ..Default::default() });
    // Bark and leaves share bend and sway (WindResponse::WOOD / LEAVES), so trees and bushes move
    // as one; grass and ferns are more flexible.
    let bark = |scene: &mut Scene, tex| {
        scene.add_material(Material { normal_strength: 1.0, wind: WindResponse::WOOD, ..Material::textured(MaterialKind::Standard, tex, TextureMapping::MeshUv, 1.2) })
    };
    let foliage = |scene: &mut Scene, tex, tint: Vec3, wind: WindResponse| {
        scene.add_material(Material { base_color: tint, roughness: 0.6, wind, ..Material::textured(MaterialKind::Foliage, tex, TextureMapping::MeshUv, 1.0) })
    };
    let oak_bark_m = bark(&mut scene, oak_bark);
    let pine_bark_m = bark(&mut scene, pine_bark);
    let leaves_m = [Vec3::ONE, Vec3::new(1.15, 1.05, 0.7), Vec3::new(0.85, 1.0, 0.8)].map(|tint| foliage(&mut scene, leaf, tint, WindResponse::LEAVES));
    let needles_m = foliage(&mut scene, needle, Vec3::ONE, WindResponse::LEAVES);
    let grass_m = foliage(&mut scene, blade, Vec3::ONE, WindResponse::GRASS);
    let fern_m = foliage(&mut scene, fern_tex, Vec3::ONE, WindResponse::FERN);
    let mossy_rock = scene.add_material(Material {
        kind: MaterialKind::Rock,
        base_color: Vec3::new(0.3, 0.29, 0.27),
        roughness: 0.85,
        detail_scale: 5.0,
        detail_strength: 0.3,
        texture: Some(rock),
        mapping: TextureMapping::ObjectTriplanar,
        uv_scale: 0.5,
        layers: [Some(moss), None, None, None],
        ..Default::default()
    });
    let mossy_log = scene.add_material(Material {
        kind: MaterialKind::Rock,
        base_color: Vec3::new(0.32, 0.27, 0.21),
        roughness: 0.9,
        detail_scale: 3.0,
        detail_strength: 0.2,
        texture: Some(oak_bark),
        mapping: TextureMapping::ObjectTriplanar,
        uv_scale: 1.0,
        layers: [Some(moss), None, None, None],
        ..Default::default()
    });
    let mushroom_m = scene.add_material(Material::standard(Vec3::new(0.55, 0.08, 0.04), 0.4, 0.0));
    // Stable names for scene files.
    for (i, &leaves) in leaves_m.iter().enumerate() {
        scene.set_material_name(leaves, &format!("leaves {}", i + 1));
    }
    for (material, name) in [
        (terrain_material, "forest floor"),
        (oak_bark_m, "oak bark"),
        (pine_bark_m, "pine bark"),
        (needles_m, "needles"),
        (grass_m, "grass"),
        (fern_m, "fern"),
        (mossy_rock, "mossy rock"),
        (mossy_log, "mossy log"),
        (mushroom_m, "mushroom"),
    ] {
        scene.set_material_name(material, name);
    }

    // ---- Meshes --------------------------------------------------------------------------------------
    let mut m = Meshes { scene: &mut scene };
    let terrain_resolution = detail.pick(TERRAIN_RESOLUTION.0, TERRAIN_RESOLUTION.1);
    let terrain = add_baked_with(
        m.scene,
        "terrain",
        &{
            let mut t = primitives::terrain(SIZE, terrain_resolution, 1.0, 1);
            t.deform(|p| Vec3::new(p.x, ground_height(p.x, p.z), p.z));
            t
        },
        BuildConfig { sdf: SdfMode::Heightfield { resolution: 512 }, ..Default::default() },
    );

    let leaves = detail.pick(34, 56);
    let mut species: Vec<Species> = Vec::new();
    for (i, (height, radius, depth)) in [(18.0f32, 0.45f32, detail.pick(3u32, 4)), (14.0, 0.34, detail.pick(3, 4)), (9.0, 0.2, 3)].iter().enumerate() {
        let tree = vegetation::broadleaf_tree(&TreeParams {
            seed: 11 + i as u32 * 7,
            height: *height,
            trunk_radius: *radius,
            depth: *depth,
            children: 4,
            leaves_per_twig: if *depth == 4 { leaves / 3 } else { leaves },
            leaf: LeafShape { length: 0.3, width: 0.18, fold: 0.35 },
            sides: detail.pick(10, 16),
            gravitropism: 0.3 + i as f32 * 0.1,
        });
        species.push(Species {
            name: "Broadleaf tree",
            wood: m.add("broadleaf wood", &tree.wood, true),
            foliage: m.add_foliage("broadleaf foliage", &tree.foliage, true),
            bark: oak_bark_m,
            leaves: leaves_m[i],
        });
    }
    for (i, height) in [22.0f32, 15.0].iter().enumerate() {
        let tree = vegetation::conifer_assembly(41 + i as u32, *height, height * 0.018, detail.pick(34, 56), detail.pick(10, 14));
        species.push(Species {
            name: "Conifer",
            wood: m.add("conifer wood", &tree.wood, true),
            foliage: m.add_assembly("conifer needles", &tree),
            bark: pine_bark_m,
            leaves: needles_m,
        });
    }
    let bushes: Vec<(MeshHandle, MeshHandle)> = (0..2)
        .map(|i| {
            let b = vegetation::bush(61 + i, 1.4, detail.pick(10, 16));
            (m.add("bush wood", &b.wood, false), m.add_foliage("bush leaves", &b.foliage, true))
        })
        .collect();
    let ferns: Vec<MeshHandle> = (0..3).map(|i| m.add_foliage("fern", &vegetation::fern(81 + i, 9, 0.9, detail.pick(14, 22)), false)).collect();
    let grass: Vec<MeshHandle> = (0..2).map(|i| m.add_foliage("grass", &vegetation::grass_clump(91 + i, detail.pick(40, 80), 0.45), false)).collect();
    let rocks: Vec<MeshHandle> = (0..3).map(|i| m.add("boulder", &primitives::rock(101 + i * 5, detail.pick(5, 6)), true)).collect();
    let log = {
        let path: Vec<Vec3> = (0..=16).map(|i| Vec3::new(i as f32 * 0.6, (i as f32 * 0.7).sin() * 0.05, (i as f32 * 0.4).sin() * 0.15)).collect();
        let radii: Vec<f32> = (0..=16).map(|i| 0.38 - i as f32 * 0.008).collect();
        let mut log = tube(&path, &radii, detail.pick(16, 24), true);
        helios_geometry::modeling::displace(&mut log, 0.025, 4.0, 3, 5);
        m.add("fallen log", &log, true)
    };
    let mushrooms = {
        let cap = lathe(&[Vec2::new(0.0, 0.0), Vec2::new(0.012, 0.0), Vec2::new(0.012, 0.06), Vec2::new(0.05, 0.065), Vec2::new(0.045, 0.085), Vec2::new(0.0, 0.1)], 16, &|_, _| 1.0, true);
        let mut parts = Vec::new();
        let mut rng = Rng(5);
        for _ in 0..7 {
            let s = rng.range(0.6, 1.3);
            parts.push(cap.transformed(Mat4::from_scale_rotation_translation(Vec3::splat(s), Quat::from_rotation_z(rng.range(-0.2, 0.2)), Vec3::new(rng.range(-0.2, 0.2), 0.0, rng.range(-0.2, 0.2)))));
        }
        m.add("mushrooms", &merge(&parts), false)
    };

    // ---- Scatter ---------------------------------------------------------------------------------------
    outline::actor(&mut scene, "Landscape", Mat4::IDENTITY, |scene| scene.add_instance(terrain, terrain_material, Mat4::IDENTITY));
    let mut rng = Rng(0x1234_5678);
    // On the terrain mesh as rendered (not the analytic height it approximates).
    let place = |x: f32, z: f32, sink: f32, scale: Vec3, yaw: f32| {
        Mat4::from_scale_rotation_translation(scale, Quat::from_rotation_y(yaw), Vec3::new(x, terrain_mesh_height(x, z, terrain_resolution) - sink, z))
    };
    let mut placed_trees: Vec<Vec2> = Vec::new();
    let target = detail.pick(1500, 3200);
    let radius = detail.pick(260.0, 380.0);
    let mut attempts = 0;
    outline::folder("Trees");
    while placed_trees.len() < target && attempts < target * 20 {
        attempts += 1;
        let (x, z) = (rng.range(-radius, radius), rng.range(-radius, radius));
        if Vec2::new(x, z).length() > radius || rng.next() > density(x, z) {
            continue;
        }
        let p = Vec2::new(x, z);
        if placed_trees.iter().any(|q| q.distance_squared(p) < 16.0) {
            continue;
        }
        placed_trees.push(p);
        let s = &species[(rng.next() * species.len() as f32) as usize % species.len()];
        let scale = rng.range(0.8, 1.25);
        let xf = place(x, z, 0.3, Vec3::splat(scale), rng.range(0.0, 6.3));
        outline::actor(&mut scene, s.name, xf, |scene| {
            scene.add_instance(s.wood, s.bark, xf);
            scene.add_instance(s.foliage, s.leaves, xf);
        });
    }
    outline::end_folder();
    let mut scatter = |count: usize, range: f32, mut f: Box<Placer>| {
        let mut n = 0;
        let mut tries = 0;
        while n < count && tries < count * 10 {
            tries += 1;
            let (x, z) = (rng.range(-range, range), rng.range(-range, range));
            if Vec2::new(x, z).length() > range {
                continue;
            }
            f(&mut scene, x, z, &mut rng);
            n += 1;
        }
    };
    outline::folder("Bushes");
    scatter(detail.pick(500, 900), 160.0, Box::new(|scene, x, z, rng| {
        let (wood, leaves) = bushes[(rng.next() * 2.0) as usize % 2];
        let xf = place(x, z, 0.05, Vec3::splat(rng.range(0.7, 1.4)), rng.range(0.0, 6.3));
        outline::actor(scene, "Bush", xf, |scene| {
            scene.add_instance(wood, oak_bark_m, xf);
            scene.add_instance(leaves, leaves_m[2], xf);
        });
    }));
    outline::end_folder();
    outline::folder("Ferns");
    scatter(detail.pick(2500, 5000), 120.0, Box::new(|scene, x, z, rng| {
        let fern = ferns[(rng.next() * 3.0) as usize % 3];
        let xf = place(x, z, 0.02, Vec3::splat(rng.range(0.7, 1.5)), rng.range(0.0, 6.3));
        if !in_campfire(x, z) {
            outline::actor(scene, "Fern", xf, |scene| scene.add_instance(fern, fern_m, xf));
        }
    }));
    outline::end_folder();
    outline::folder("Grass");
    scatter(detail.pick(1500, 3000), 70.0, Box::new(|scene, x, z, rng| {
        let clump = grass[(rng.next() * 2.0) as usize % 2];
        let xf = place(x, z, 0.0, Vec3::splat(rng.range(0.8, 1.4)), rng.range(0.0, 6.3));
        if !in_campfire(x, z) {
            outline::actor(scene, "Grass clump", xf, |scene| scene.add_instance(clump, grass_m, xf));
        }
    }));
    outline::end_folder();
    outline::folder("Rocks");
    scatter(160, 250.0, Box::new(|scene, x, z, rng| {
        let rock = rocks[(rng.next() * 3.0) as usize % 3];
        let s = rng.range(0.5, 2.6);
        let xf = place(x, z, s * 0.25, Vec3::new(s * rng.range(0.8, 1.3), s * rng.range(0.6, 1.0), s * rng.range(0.8, 1.3)), rng.range(0.0, 6.3));
        outline::actor(scene, "Mossy boulder", xf, |scene| scene.add_instance(rock, mossy_rock, xf));
    }));
    outline::end_folder();
    outline::folder("Fallen logs");
    scatter(50, 150.0, Box::new(|scene, x, z, rng| {
        let yaw = rng.range(0.0, 6.3);
        let xf = place(x, z, 0.15, Vec3::splat(rng.range(0.8, 1.3)), yaw);
        let token = outline::begin(scene, "Fallen log", xf);
        scene.add_instance(log, mossy_log, xf);
        let side = xf.transform_point3(Vec3::new(rng.range(1.0, 7.0), 0.0, 0.5));
        scene.add_instance(mushrooms, mushroom_m, place(side.x, side.z, 0.0, Vec3::splat(rng.range(0.8, 1.6)), rng.range(0.0, 6.3)));
        outline::end(scene, token);
    }));
    outline::end_folder();

    // A camp lantern near the start: a warm local light that bounces under the canopy.
    let lantern = Vec3::new(9.0, ground_height(9.0, -4.0) + 0.6, -4.0);
    outline::actor(&mut scene, "Camp lantern", Mat4::from_translation(lantern), |scene| {
        scene.add_light(Light { source_radius: 0.1, ..Light::point(lantern, Vec3::new(1.0, 0.65, 0.3), 2.0, 12.0) })
    });
    // Built last, so pre-baked mesh libraries of the rest of the scene stay valid.
    campfire(&mut scene, CAMPFIRE.x, CAMPFIRE.y, terrain_resolution);
    scene
}

/// Where the campfire burns, in the clearing in front of the initial view.
const CAMPFIRE: Vec2 = Vec2::new(7.0, -1.0);

/// Scale of the campfire's soot (f32 bits; 1 by default): `--campfire-smoke 6` builds the forest
/// with a smokier fire (a damp log), whose shadow on the clearing shows plainly.
static CAMPFIRE_SMOKE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3f80_0000);

/// Scales the soot of the campfire of forests built from now on.
pub fn set_campfire_smoke(scale: f32) {
    CAMPFIRE_SMOKE.store(scale.max(0.0).to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// Ferns and grass don't grow in the fire ring.
fn in_campfire(x: f32, z: f32) -> bool {
    Vec2::new(x, z).distance(CAMPFIRE) < 1.4
}

/// A campfire: a ring of stones around a teepee of charred logs over glowing coals, a simulated
/// fire with its smoke column, sparks and embers, and a light the flames drive.
fn campfire(scene: &mut Scene, x: f32, z: f32, terrain_resolution: u32) {
    let ground = |x: f32, z: f32| terrain_mesh_height(x, z, terrain_resolution);
    let base = Vec3::new(x, ground(x, z), z);
    // One actor: stones, coals, logs, the light, the fire volume, sparks and embers.
    let token = outline::begin(scene, "Campfire", Mat4::from_translation(base));
    let stone_m = scene.add_material(Material { kind: MaterialKind::Rock, base_color: Vec3::new(0.3, 0.29, 0.27), roughness: 0.85, detail_scale: 8.0, detail_strength: 0.3, ..Default::default() });
    let charred = scene.add_material(Material::standard(Vec3::new(0.045, 0.035, 0.03), 0.9, 0.0));
    let coals_m = scene.add_material(Material { base_color: Vec3::new(1.0, 0.32, 0.07), emissive: 1.2, roughness: 0.9, ..Default::default() });
    let small = |sdf: bool| BuildConfig { sdf: if sdf { SdfMode::Volume { resolution: 24 } } else { SdfMode::None }, ..Default::default() };
    let stone = add_baked_with(scene, "campfire stone", &primitives::rock(131, 4), small(true));
    let coals = add_baked_with(scene, "campfire coals", &primitives::rock(137, 4), small(false));
    let path: Vec<Vec3> = (0..=6).map(|i| Vec3::new(0.0, i as f32 * 0.15, 0.0)).collect();
    let radii: Vec<f32> = (0..=6).map(|i| 0.06 - i as f32 * 0.004).collect();
    let mut log_mesh = tube(&path, &radii, 10, true);
    helios_geometry::modeling::displace(&mut log_mesh, 0.006, 12.0, 2, 9);
    let log = add_baked_with(scene, "campfire log", &log_mesh, small(false));

    let mut rng = Rng(0xF1E2);
    for i in 0..11 {
        let angle = i as f32 / 11.0 * std::f32::consts::TAU + rng.range(-0.1, 0.1);
        let r = 0.68 + rng.range(-0.04, 0.04);
        let (sx, sz) = (x + angle.cos() * r, z + angle.sin() * r);
        let s = rng.range(0.13, 0.19);
        let xf = Mat4::from_scale_rotation_translation(Vec3::new(s, s * 0.75, s), Quat::from_rotation_y(rng.range(0.0, 6.3)), Vec3::new(sx, ground(sx, sz) + s * 0.2, sz));
        scene.add_instance(stone, stone_m, xf);
    }
    scene.add_instance(coals, coals_m, Mat4::from_scale_rotation_translation(Vec3::new(0.42, 0.06, 0.42), Quat::IDENTITY, base));
    let apex = base + Vec3::new(0.0, 0.62, 0.0);
    for i in 0..6 {
        let angle = i as f32 / 6.0 * std::f32::consts::TAU + 0.3;
        let foot = base + Vec3::new(angle.cos(), 0.0, angle.sin()) * 0.5 + Vec3::Y * 0.03;
        let rotation = Quat::from_rotation_arc(Vec3::Y, (apex - foot).normalize());
        scene.add_instance(log, charred, Mat4::from_rotation_translation(rotation, foot));
    }

    // In the flames just above the teepee: the logs below it stripe the clearing with shadows, the
    // smoke above is lit without their shadow cone.
    let light = scene.add_light(Light { source_radius: 0.15, ..Light::point(base + Vec3::Y * 0.8, Vec3::new(1.0, 0.48, 0.18), 4.0, 18.0) });
    let mut fire = VolumeEmitter::campfire(base);
    if let helios_render::VolumeSource::Simulated(sim) = &mut fire.source {
        let smoke = f32::from_bits(CAMPFIRE_SMOKE.load(std::sync::atomic::Ordering::Relaxed));
        sim.soot_rate *= smoke;
        sim.soot_yield *= smoke;
    }
    scene.volumes.push(VolumeEmitter { light: Some(light), ..fire });
    scene.particles.push(ParticleEmitter::sparks(base + Vec3::Y * 0.3));
    scene.particles.push(ParticleEmitter::embers(base + Vec3::Y * 0.25));
    outline::end(scene, token);
}

fn initial_camera() -> Camera {
    let mut camera = Camera { position: Vec3::new(0.0, ground_height(0.0, 0.0) + 1.7, 0.0), fov_y: 60f32.to_radians(), ..Default::default() };
    camera.look_at(Vec3::new(30.0, ground_height(30.0, -8.0) + 4.0, -8.0));
    camera
}

/// The ground as rendered at each detail level (camera floor, PCG scattering).
fn floor_medium(x: f32, z: f32) -> f32 {
    terrain_mesh_height(x, z, TERRAIN_RESOLUTION.0)
}

fn floor_high(x: f32, z: f32) -> f32 {
    terrain_mesh_height(x, z, TERRAIN_RESOLUTION.1)
}

pub fn build(detail: Detail) -> SceneSetup {
    let floor: fn(f32, f32) -> f32 = detail.pick(floor_medium, floor_high);
    SceneSetup { scene: build_scene(detail), camera: initial_camera(), sun: SUN, floor, floor_below: None, animation: None, motion: Default::default(), outline: Default::default() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The placement height is the terrain mesh's: exact at its vertices, linear along its edges;
    /// the analytic height it replaces stands centimetres off it (above it on crests).
    #[test]
    fn plants_stand_on_the_terrain_mesh() {
        for resolution in [384u32, 768] {
            let step = SIZE / resolution as f32;
            let vertex = |i: i64, j: i64| Vec2::new(i as f32, j as f32) / resolution as f32 * SIZE - SIZE * 0.5;
            for (i, j) in [(192i64, 192i64), (200, 181), (230, 170)] {
                let p = vertex(i, j);
                assert!((terrain_mesh_height(p.x, p.y, resolution) - ground_height(p.x, p.y)).abs() < 1e-3);
                // The middle of the diagonal shared by a quad's two triangles.
                let (a, b) = (vertex(i + 1, j), vertex(i, j + 1));
                let m = (a + b) * 0.5;
                let expected = 0.5 * (ground_height(a.x, a.y) + ground_height(b.x, b.y));
                assert!((terrain_mesh_height(m.x, m.y, resolution) - expected).abs() < 1e-3);
            }
            // How far the analytic height floats above the mesh where grass and ferns grow.
            let mut rng = Rng(7);
            let (mut above, mut sum, mut n) = (0.0f32, 0.0f32, 0);
            for _ in 0..20000 {
                let (x, z) = (rng.range(-70.0, 70.0), rng.range(-70.0, 70.0));
                let d = ground_height(x, z) - terrain_mesh_height(x, z, resolution);
                above = above.max(d);
                sum += d.abs();
                n += 1;
            }
            eprintln!("grid {step:.2} m: analytic height above the mesh by up to {:.3} m, mean |difference| {:.4} m", above, sum / n as f32);
            assert!(above > 0.005, "the mesh approximates the analytic ground");
        }
    }
}