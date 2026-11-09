//! A temperate forest at golden hour: broadleaf and conifer trees with real leaf geometry,
//! bushes, ferns and grass over a leaf-litter forest floor, mossy boulders, fallen logs and
//! mushrooms. Thousands of instances; the canopy shadows come from virtual shadow maps and the
//! green bounce light under the canopy from Lumen. A gusty breeze sways the trees and ripples
//! through the grass and ferns.

use glam::{Mat4, Quat, Vec2, Vec3};
use helios_geometry::modeling::{lathe, merge, tube};
use helios_geometry::noise::fbm;
use helios_geometry::vegetation::{self, LeafShape, TreeParams};
use helios_geometry::{primitives, BuildConfig, Mesh, SdfMode};
use helios_render::{
    Camera, Light, Material, MaterialHandle, MaterialKind, MeshHandle, Scene, Sun, TextureDesc, TextureMapping, TexturePattern, Wind, WindResponse,
};

use super::{add_baked_with, smoothstep, Detail, Rng, SceneSetup};

const SIZE: f32 = 900.0;
const SUN: (f32, f32, f32) = (2.5, 0.26, 9.0);

pub fn ground_height(x: f32, z: f32) -> f32 {
    let p = Vec3::new(x, 0.0, z) * 0.006;
    let hills = fbm(p, 5, 77) * 22.0 + fbm(p * 4.0, 3, 78) * 2.5;
    // A gentle hollow where the camera starts.
    let hollow = 1.0 - smoothstep(10.0, 60.0, Vec2::new(x, z).length());
    hills - hollow * 2.0
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
        let config = BuildConfig { sdf: if sdf { SdfMode::Volume { resolution: 40 } } else { SdfMode::None }, ..Default::default() };
        add_baked_with(self.scene, name, mesh, config)
    }
}

/// Places one scattered object at (x, z).
type Placer<'a> = dyn FnMut(&mut Scene, f32, f32, &mut Rng) + 'a;

struct Species {
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
    let litter = t(&mut scene, TextureDesc::new(TexturePattern::ForestFloor, Vec3::new(0.3, 0.17, 0.06), Vec3::new(0.09, 0.065, 0.04)).roughness(0.85, 0.9).seed(71));
    let moss = t(&mut scene, TextureDesc::new(TexturePattern::Moss, Vec3::new(0.1, 0.15, 0.04), Vec3::new(0.07, 0.1, 0.03)).roughness(0.95, 0.9).scale(0.5).bump(0.6).seed(72));
    let dirt = t(&mut scene, TextureDesc::new(TexturePattern::Dirt, Vec3::new(0.17, 0.12, 0.08), Vec3::new(0.2, 0.18, 0.15)).roughness(0.95, 0.8).bump(0.6).seed(73));
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

    // ---- Meshes --------------------------------------------------------------------------------------
    let mut m = Meshes { scene: &mut scene };
    let terrain = add_baked_with(
        m.scene,
        "terrain",
        &{
            let mut t = primitives::terrain(SIZE, detail.pick(384, 768), 1.0, 1);
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
            wood: m.add("broadleaf wood", &tree.wood, true),
            foliage: m.add("broadleaf foliage", &tree.foliage, true),
            bark: oak_bark_m,
            leaves: leaves_m[i],
        });
    }
    for (i, height) in [22.0f32, 15.0].iter().enumerate() {
        let tree = vegetation::conifer(41 + i as u32, *height, height * 0.018, detail.pick(34, 56), detail.pick(10, 14));
        species.push(Species {
            wood: m.add("conifer wood", &tree.wood, true),
            foliage: m.add("conifer needles", &tree.foliage, true),
            bark: pine_bark_m,
            leaves: needles_m,
        });
    }
    let bushes: Vec<(MeshHandle, MeshHandle)> = (0..2)
        .map(|i| {
            let b = vegetation::bush(61 + i, 1.4, detail.pick(10, 16));
            (m.add("bush wood", &b.wood, false), m.add("bush leaves", &b.foliage, true))
        })
        .collect();
    let ferns: Vec<MeshHandle> = (0..3).map(|i| m.add("fern", &vegetation::fern(81 + i, 9, 0.9, detail.pick(14, 22)), false)).collect();
    let grass: Vec<MeshHandle> = (0..2).map(|i| m.add("grass", &vegetation::grass_clump(91 + i, detail.pick(40, 80), 0.45), false)).collect();
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
    scene.add_instance(terrain, terrain_material, Mat4::IDENTITY);
    let mut rng = Rng(0x1234_5678);
    let place = |x: f32, z: f32, sink: f32, scale: Vec3, yaw: f32| {
        Mat4::from_scale_rotation_translation(scale, Quat::from_rotation_y(yaw), Vec3::new(x, ground_height(x, z) - sink, z))
    };
    let mut placed_trees: Vec<Vec2> = Vec::new();
    let target = detail.pick(1500, 3200);
    let radius = detail.pick(260.0, 380.0);
    let mut attempts = 0;
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
        scene.add_instance(s.wood, s.bark, xf);
        scene.add_instance(s.foliage, s.leaves, xf);
    }
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
    scatter(detail.pick(500, 900), 160.0, Box::new(|scene, x, z, rng| {
        let (wood, leaves) = bushes[(rng.next() * 2.0) as usize % 2];
        let xf = place(x, z, 0.05, Vec3::splat(rng.range(0.7, 1.4)), rng.range(0.0, 6.3));
        scene.add_instance(wood, oak_bark_m, xf);
        scene.add_instance(leaves, leaves_m[2], xf);
    }));
    scatter(detail.pick(2500, 5000), 120.0, Box::new(|scene, x, z, rng| {
        let fern = ferns[(rng.next() * 3.0) as usize % 3];
        scene.add_instance(fern, fern_m, place(x, z, 0.02, Vec3::splat(rng.range(0.7, 1.5)), rng.range(0.0, 6.3)));
    }));
    scatter(detail.pick(1500, 3000), 70.0, Box::new(|scene, x, z, rng| {
        let clump = grass[(rng.next() * 2.0) as usize % 2];
        scene.add_instance(clump, grass_m, place(x, z, 0.0, Vec3::splat(rng.range(0.8, 1.4)), rng.range(0.0, 6.3)));
    }));
    scatter(160, 250.0, Box::new(|scene, x, z, rng| {
        let rock = rocks[(rng.next() * 3.0) as usize % 3];
        let s = rng.range(0.5, 2.6);
        let xf = place(x, z, s * 0.25, Vec3::new(s * rng.range(0.8, 1.3), s * rng.range(0.6, 1.0), s * rng.range(0.8, 1.3)), rng.range(0.0, 6.3));
        scene.add_instance(rock, mossy_rock, xf);
    }));
    scatter(50, 150.0, Box::new(|scene, x, z, rng| {
        let yaw = rng.range(0.0, 6.3);
        let xf = place(x, z, 0.15, Vec3::splat(rng.range(0.8, 1.3)), yaw);
        scene.add_instance(log, mossy_log, xf);
        let side = xf.transform_point3(Vec3::new(rng.range(1.0, 7.0), 0.0, 0.5));
        scene.add_instance(mushrooms, mushroom_m, place(side.x, side.z, 0.0, Vec3::splat(rng.range(0.8, 1.6)), rng.range(0.0, 6.3)));
    }));

    // A camp lantern near the start: a warm local light that bounces under the canopy.
    let lantern = Vec3::new(9.0, ground_height(9.0, -4.0) + 0.6, -4.0);
    scene.add_light(Light { source_radius: 0.1, ..Light::point(lantern, Vec3::new(1.0, 0.65, 0.3), 2.0, 12.0) });
    scene
}

fn initial_camera() -> Camera {
    let mut camera = Camera { position: Vec3::new(0.0, ground_height(0.0, 0.0) + 1.7, 0.0), fov_y: 60f32.to_radians(), ..Default::default() };
    camera.look_at(Vec3::new(30.0, ground_height(30.0, -8.0) + 4.0, -8.0));
    camera
}

pub fn build(detail: Detail) -> SceneSetup {
    SceneSetup { scene: build_scene(detail), camera: initial_camera(), sun: SUN, floor: ground_height, animation: None }
}
