//! The valley: a procedural mountain landscape, hundreds of high-density boulders, a metal
//! sculpture and a small lake — all rendered through the virtual geometry pipeline (the lake's
//! water through the forward translucency pass).

use glam::{Mat4, Quat, Vec2, Vec3};
use helios_geometry::modeling::cuboid;
use helios_geometry::{noise, primitives, BuildConfig, SdfMode};
use helios_render::{Camera, Material, MaterialKind, Scene, Sun, TextureDesc, TextureMapping, TexturePattern};

use super::{add_baked, add_baked_with, smoothstep, Detail, Rng, SceneSetup};
use crate::motion::{DemoMotion, Motion};

pub const TERRAIN_SIZE: f32 = 900.0;
pub const TERRAIN_HEIGHT: f32 = 110.0;
pub const TERRAIN_SEED: u32 = 3;
pub const SUN: (f32, f32, f32) = (0.9, 0.32, 10.0);

/// The lake fills the lowest trough of the valley floor, between the initial camera and the
/// sculpture.
const LAKE_CENTER: Vec2 = Vec2::new(-5.0, 22.0);
/// Semi-axes of the basin (the shore lies inside, where the basin dips below the water).
const LAKE_RADII: Vec2 = Vec2::new(14.0, 8.5);
const LAKE_ANGLE: f32 = 1.35;
const LAKE_DEPTH: f32 = 3.0;
/// Just below the lowest point of the (flat) valley floor, so water shows only where the basin
/// is carved, with low, natural shores.
const WATER_LEVEL: f32 = -2.2;

/// The landscape with the lake basin carved into it.
pub fn ground_height(x: f32, z: f32) -> f32 {
    primitives::terrain_height(Vec2::new(x, z), TERRAIN_SIZE, TERRAIN_HEIGHT, TERRAIN_SEED) - LAKE_DEPTH * lake_basin(x, z)
}

/// 1 in the middle of the lake basin, falling smoothly to 0 at its rim; the rim wanders with
/// noise so the shore line is irregular.
fn lake_basin(x: f32, z: f32) -> f32 {
    let (s, c) = LAKE_ANGLE.sin_cos();
    let d = Vec2::new(x, z) - LAKE_CENTER;
    let local = Vec2::new(c * d.x + s * d.y, -s * d.x + c * d.y) / LAKE_RADII;
    let wobble = 0.22 * noise::fbm(Vec3::new(x, 0.0, z) * 0.08, 3, 7);
    let r = local.length() * (1.0 + wobble);
    // Shallow shelves, then a gentle slope down to the middle.
    smoothstep(1.0, 0.15, r).powf(1.4)
}

/// The camera stays above the ground; it may dive into the lake (the renderer draws the
/// underwater view).
fn camera_floor(x: f32, z: f32) -> f32 {
    ground_height(x, z)
}

fn build_scene(detail: Detail) -> (Scene, DemoMotion) {
    let mut scene = Scene::new();
    let mut motion = DemoMotion::default();
    scene.sun = Sun::from_angles(SUN.0, SUN.1, SUN.2);
    scene.ground_albedo = Vec3::new(0.18, 0.17, 0.13);
    scene.clouds = helios_render::CloudLayer::cumulus(0.5);
    // A light haze lying in the valley floor.
    scene.fog = helios_render::FogVolume {
        density: 0.0015,
        height_falloff: 0.04,
        base_height: ground_height(0.0, 0.0) - 5.0,
        anisotropy: 0.6,
        noise: 0.6,
        distance: 400.0,
        ..Default::default()
    };
    scene.water = helios_render::Water { foam: 0.6, ..Default::default() };

    // ---- Textures (generated on the GPU) ------------------------------------------------------
    let grass = scene.add_texture(TextureDesc::new(TexturePattern::Grass, Vec3::new(0.16, 0.24, 0.06), Vec3::new(0.08, 0.12, 0.03)).roughness(0.92, 0.9).seed(11));
    let dirt = scene.add_texture(TextureDesc::new(TexturePattern::Dirt, Vec3::new(0.24, 0.17, 0.11), Vec3::new(0.22, 0.2, 0.18)).roughness(0.95, 0.8).bump(0.5).seed(12));
    let rock_tex = scene.add_texture(TextureDesc::new(TexturePattern::Rock, Vec3::new(0.3, 0.28, 0.25), Vec3::new(0.14, 0.13, 0.12)).roughness(0.75, 0.9).bump(1.4).seed(13));
    let snow = scene.add_texture(TextureDesc::new(TexturePattern::Snow, Vec3::new(0.86, 0.89, 0.93), Vec3::ONE).roughness(0.45, 0.5).bump(0.5).seed(14));
    let hammered = scene.add_texture(TextureDesc::new(TexturePattern::Hammered, Vec3::ONE, Vec3::ONE).roughness(0.18, 0.3).scale(1.5).bump(0.6).seed(15));

    // ---- Materials ----------------------------------------------------------------------------
    let terrain_material = scene.add_material(Material {
        kind: MaterialKind::Terrain,
        base_color: Vec3::ONE,
        roughness: 0.9,
        layers: [Some(grass), Some(dirt), Some(rock_tex), Some(snow)],
        ..Default::default()
    });
    let rock_materials = [Vec3::new(0.32, 0.29, 0.25), Vec3::new(0.26, 0.25, 0.24), Vec3::new(0.38, 0.33, 0.27)].map(|c| {
        scene.add_material(Material {
            kind: MaterialKind::Rock,
            base_color: c,
            roughness: 0.8,
            detail_scale: 7.0,
            detail_strength: 0.35,
            texture: Some(rock_tex),
            mapping: TextureMapping::ObjectTriplanar,
            uv_scale: 0.6,
            ..Default::default()
        })
    });
    let gold = scene.add_material(Material {
        kind: MaterialKind::Metal,
        base_color: Vec3::new(1.0, 0.76, 0.42),
        roughness: 0.22,
        metallic: 1.0,
        detail_scale: 3.0,
        texture: Some(hammered),
        mapping: TextureMapping::ObjectTriplanar,
        uv_scale: 1.2,
        normal_strength: 0.6,
        ..Default::default()
    });

    for (material, name) in [(terrain_material, "landscape"), (gold, "gold")] {
        scene.set_material_name(material, name);
    }

    // ---- Meshes -------------------------------------------------------------------------------
    let mut landscape = primitives::terrain(TERRAIN_SIZE, detail.pick(384, 768), TERRAIN_HEIGHT, TERRAIN_SEED);
    for v in &mut landscape.vertices {
        let [x, _, z] = v.position;
        v.position[1] = ground_height(x, z);
    }
    landscape.compute_normals();
    let terrain = add_baked_with(&mut scene, "terrain", &landscape, BuildConfig { sdf: SdfMode::Heightfield { resolution: 512 }, ..Default::default() });
    let rocks: Vec<_> = (0..3)
        .map(|seed| add_baked(&mut scene, "rock", &primitives::rock(seed * 7 + 1, detail.pick(6, 7))))
        .collect();
    let (segments, sides) = detail.pick((1024, 96), (2048, 192));
    let knot = add_baked(&mut scene, "sculpture", &primitives::torus_knot(2, 3, segments, sides, 0.33));

    // The lake: a flat sheet at the water level over the basin. It has no distance field (water
    // neither occludes Lumen's rays nor shades its surface cache) and casts no shadow.
    let extent = LAKE_RADII.max_element() * 2.6;
    let mut sheet = primitives::terrain(extent, 16, 0.0, 1);
    sheet.deform(|p| Vec3::new(p.x + LAKE_CENTER.x, WATER_LEVEL, p.z + LAKE_CENTER.y));
    let lake = add_baked_with(&mut scene, "lake", &sheet, BuildConfig { sdf: SdfMode::None, ..Default::default() });
    let water = scene.add_material(Material { kind: MaterialKind::Water, base_color: Vec3::ONE, roughness: 0.03, ..Default::default() });
    scene.set_material_name(water, "lake water");

    // ---- Instances ----------------------------------------------------------------------------
    scene.add_instance(terrain, terrain_material, Mat4::IDENTITY);
    let lake = scene.add_instance(lake, water, Mat4::IDENTITY);
    scene.set_cast_shadow(lake, false);

    // Hero arrangement: a plinth boulder carrying the sculpture.
    let plinth_height = ground_height(0.0, 0.0);
    scene.add_instance(
        rocks[0],
        rock_materials[1],
        Mat4::from_scale_rotation_translation(Vec3::new(5.0, 3.0, 5.0), Quat::from_rotation_y(0.4), Vec3::new(0.0, plinth_height + 0.6, 0.0)),
    );
    let sculpture_center = Vec3::new(0.0, plinth_height + 6.6, 0.0);
    let sculpture = scene.add_instance(knot, gold, Mat4::from_scale_rotation_translation(Vec3::splat(3.2), Quat::from_rotation_x(0.25), sculpture_center));
    // The sculpture slowly turns on its plinth.
    motion.add(&mut scene, Motion::Spin { center: sculpture_center, period: 45.0 }, &[sculpture], &[]);

    // Boulders in the lake, some breaking its surface.
    for (i, (x, z, size)) in [(-8.0, 15.5, 1.7), (-2.5, 26.0, 1.2), (-7.0, 31.0, 2.1), (-3.5, 19.0, 0.9), (-1.0, 12.5, 1.4)].into_iter().enumerate() {
        let shape = Vec3::new(1.3, 0.8, 1.1) * size;
        let at = Vec3::new(x, ground_height(x, z) + size * 0.25, z);
        scene.add_instance(rocks[i % 3], rock_materials[i % 3], Mat4::from_scale_rotation_translation(shape, Quat::from_rotation_y(i as f32 * 1.9), at));
    }

    // Scattered boulders: dense near the valley, sparse and huge on the slopes.
    let mut rng = Rng(0x9e37_79b9);
    let mut placed = 0;
    while placed < 600 {
        let radius = 12.0 + rng.next().powf(0.8) * TERRAIN_SIZE * 0.33;
        let angle = rng.range(0.0, std::f32::consts::TAU);
        let (x, z) = (radius * angle.cos(), radius * angle.sin());
        let scale = rng.range(0.4, 1.6) * (1.0 + radius / 120.0);
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, rng.range(0.0, 6.3), rng.range(-0.25, 0.25), rng.range(-0.25, 0.25));
        let stretch = Vec3::new(rng.range(0.8, 1.4), rng.range(0.6, 1.1), rng.range(0.8, 1.4)) * scale;
        let variant = (rng.next() * 3.0) as usize % 3;
        // Rest on the lowest point under the footprint, so that on a slope the boulder digs into
        // the hillside instead of hanging over the downhill side; skip cliffs where it would not
        // look seated either way.
        let footprint = stretch.x.max(stretch.z) * 0.8;
        let samples = [(0.0, 0.0), (footprint, 0.0), (-footprint, 0.0), (0.0, footprint), (0.0, -footprint)].map(|(dx, dz)| ground_height(x + dx, z + dz));
        let (low, high) = samples.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)));
        if high - low > footprint * 1.1 {
            continue;
        }
        let y = low + scale * 0.15;
        scene.add_instance(rocks[variant], rock_materials[variant], Mat4::from_scale_rotation_translation(stretch, rotation, Vec3::new(x, y, z)));
        placed += 1;
    }

    // (Added last, so pre-baked mesh libraries of the rest of the scene stay valid.)
    translucency_corner(&mut scene);
    raft(&mut scene, &mut motion, detail);
    (scene, motion)
}

/// A log raft drifting on the lake, in front of the initial view: it bobs, rolls and turns on a
/// slow loop (a movable instance with a distance field and a shadow).
fn raft(scene: &mut Scene, motion: &mut DemoMotion, detail: Detail) {
    let (length, radius) = (3.2, 0.14);
    let log = primitives::cylinder(detail.pick(16, 24), 4, radius, length, true);
    let along_x = Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2) * Mat4::from_translation(Vec3::new(0.0, -length * 0.5, 0.0));
    let mut parts: Vec<_> = (0..7).map(|i| log.transformed(Mat4::from_translation(Vec3::new(0.0, 0.0, (i as f32 - 3.0) * 2.05 * radius)) * along_x)).collect();
    for x in [-1.15, 1.15] {
        parts.push(cuboid(Vec3::new(x - 0.09, radius * 0.8, -1.05), Vec3::new(x + 0.09, radius * 0.8 + 0.12, 1.05)));
    }
    let mesh = add_baked(scene, "raft", &helios_geometry::modeling::merge(&parts));
    let wood = scene.add_texture(TextureDesc::new(TexturePattern::Wood, Vec3::new(0.3, 0.2, 0.12), Vec3::new(0.17, 0.11, 0.06)).roughness(0.8, 0.85).seed(16));
    let material = scene.add_material(Material::textured(MaterialKind::Standard, wood, TextureMapping::ObjectTriplanar, 0.8));
    let center = Vec3::new(-7.5, WATER_LEVEL - 0.04, 22.5);
    let instance = scene.add_instance(mesh, material, Mat4::from_rotation_translation(Quat::from_rotation_y(0.5), center));
    motion.add(scene, Motion::Float { center, drift: 1.2, period: 70.0 }, &[instance], &[]);
}

/// Layered translucency test content at the lake's north end, outside the initial view (see
/// `translucency.rs`): a clear glass pane standing in the water and, behind it, two tinted
/// panels crossing each other — glass seen through the water, water seen through the glass, and
/// intersecting layers, which only per-pixel ordering composites correctly. None of them has a
/// distance field or casts a shadow, so the rest of the scene renders exactly as without them.
fn translucency_corner(scene: &mut Scene) {
    let no_sdf = BuildConfig { sdf: SdfMode::None, ..Default::default() };
    let pane = add_baked_with(scene, "glass pane", &cuboid(Vec3::new(-1.2, -3.0, -0.02), Vec3::new(1.2, 1.0, 0.02)), no_sdf);
    let panel = add_baked_with(scene, "tinted panel", &cuboid(Vec3::new(-1.1, -2.0, -0.01), Vec3::new(1.1, 1.3, 0.01)), no_sdf);
    let glass = scene.add_material(Material { kind: MaterialKind::Glass, base_color: Vec3::new(0.82, 0.95, 0.9), roughness: 0.05, ..Default::default() });
    let tinted = |color| Material { kind: MaterialKind::Translucent, base_color: color, opacity: 0.5, roughness: 0.25, ..Default::default() };
    let red = scene.add_material(tinted(Vec3::new(0.8, 0.1, 0.06)));
    let blue = scene.add_material(tinted(Vec3::new(0.08, 0.25, 0.85)));
    let at = |x: f32, z: f32, angle: f32| Mat4::from_rotation_translation(Quat::from_rotation_y(angle), Vec3::new(x, WATER_LEVEL, z));
    for (mesh, material, transform) in [(pane, glass, at(-2.2, 28.4, 0.9)), (panel, red, at(-1.4, 30.6, 1.6)), (panel, blue, at(-1.4, 30.6, 0.2))] {
        let instance = scene.add_instance(mesh, material, transform);
        scene.set_cast_shadow(instance, false);
    }
}

fn initial_camera() -> Camera {
    let mut camera = Camera::default();
    let (x, z) = (-22.0, 30.0);
    camera.position = Vec3::new(x, ground_height(x, z) + 6.0, z);
    camera.look_at(Vec3::new(0.0, ground_height(0.0, 0.0) + 6.0, 0.0));
    camera.fov_y = 55f32.to_radians();
    camera.near = 0.05;
    camera
}

pub fn build(detail: Detail) -> SceneSetup {
    let (scene, motion) = build_scene(detail);
    SceneSetup { scene, camera: initial_camera(), sun: SUN, floor: camera_floor, animation: None, motion }
}
