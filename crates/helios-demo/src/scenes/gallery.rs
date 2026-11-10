//! The material gallery: specimens of the extended material model on plinths in a paved
//! courtyard at golden hour — Substrate slabs (clear-coat car paint, velvet, skin, candle wax,
//! marble, backlit leaves, brushed metal, IOR and specular colour, an unlit sign) and material
//! graphs (animated lava, wet stone with puddles, stylised water, and three beds displaced by
//! their graphs through Nanite tessellation: a brick pavement with carved mortar joints, a bed of
//! mushrooms — a vector displacement with overhangs — and a rippling cloth, whose displacement
//! is animated). `PageUp` switches the slabs off for an A/B against the plain metallic-roughness
//! base; debug view 24 shows each pixel's slab features, view 25 the tessellation.
//!
//! A low sun from the front left shows the skin's and the wax's soft terminators and the
//! coats' reflections; a lamp behind the bush shines through its leaves, a candle burns on the
//! wax and a warm spot light rakes the velvet and the car paint.

use glam::{Mat4, Quat, Vec3};
use helios_geometry::modeling::cuboid;
use helios_geometry::{primitives, vegetation, BuildConfig, Mesh, SdfMode};
use helios_render::material_graph::presets;
use helios_render::{
    Camera, Displacement, Light, Material, MaterialHandle, MaterialKind, MeshHandle, ParameterValue, Scene, Slab, Sun, TextureDesc, TextureMapping,
    TexturePattern,
};

use super::{add_baked, add_baked_with, Detail, SceneSetup};

pub const SUN: (f32, f32, f32) = (2.25, 0.42, 10.0);
/// Plinths stand in a row along x, 2.7 m apart.
const SPACING: f32 = 2.7;
const PLINTH_HEIGHT: f32 = 1.0;
/// Height of the brick terrace's undisplaced surface (its bricks' tops; mortar sinks below).
const BRICK_LEVEL: f32 = 0.03;
/// The beds beside the brick terrace are 3 m squares centred at x = ±BED_X, z = BED_Z.
const BED_X: f32 = 6.3;
const BED_Z: f32 = 9.0;
/// Height of the mushroom bed's ground and of the rippling cloth's rest surface.
const MOSS_LEVEL: f32 = 0.02;
const CLOTH_LEVEL: f32 = 0.16;

fn floor(_: f32, _: f32) -> f32 {
    0.0
}

struct Gallery<'a> {
    scene: &'a mut Scene,
    sphere: MeshHandle,
    plinth: MeshHandle,
    plinth_material: MaterialHandle,
}

impl Gallery<'_> {
    fn material(&mut self, name: &str, material: Material) -> MaterialHandle {
        let handle = self.scene.add_material(material);
        self.scene.set_material_name(handle, &format!("gallery/{name}"));
        handle
    }

    /// A plinth at row position `slot` (0 = leftmost); returns the centre of its top.
    fn plinth(&mut self, slot: f32) -> Vec3 {
        let x = (slot - 3.5) * SPACING;
        self.scene.add_instance(self.plinth, self.plinth_material, Mat4::from_translation(Vec3::new(x, 0.0, 0.0)));
        Vec3::new(x, PLINTH_HEIGHT, 0.0)
    }

    fn sphere(&mut self, material: MaterialHandle, center: Vec3, scale: Vec3) {
        self.scene.add_instance(self.sphere, material, Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, center));
    }
}

fn build_scene(detail: Detail) -> Scene {
    let mut scene = Scene::new();
    scene.sun = Sun::from_angles(SUN.0, SUN.1, SUN.2);
    scene.ground_albedo = Vec3::new(0.3, 0.28, 0.25);
    scene.clouds = helios_render::CloudLayer::cumulus(0.25);

    // ---- Textures -----------------------------------------------------------------------------
    let flagstone = scene.add_texture(TextureDesc::new(TexturePattern::Flagstone, Vec3::new(0.42, 0.4, 0.37), Vec3::new(0.3, 0.29, 0.27)).roughness(0.75, 0.9).seed(301));
    let marble = scene.add_texture(TextureDesc::new(TexturePattern::MarbleTiles, Vec3::new(0.82, 0.8, 0.76), Vec3::new(0.6, 0.58, 0.55)).roughness(0.3, 0.35).seed(302));
    let fabric = scene.add_texture(TextureDesc::new(TexturePattern::Fabric, Vec3::splat(0.9), Vec3::splat(0.65)).roughness(0.9, 0.95).scale(3.0).bump(0.4).seed(303));
    let leaf = scene.add_texture(TextureDesc::new(TexturePattern::Leaf, Vec3::new(0.13, 0.25, 0.04), Vec3::new(0.22, 0.28, 0.05)).roughness(0.55, 0.6).seed(304));
    let bark = scene.add_texture(TextureDesc::new(TexturePattern::Bark, Vec3::new(0.24, 0.21, 0.17), Vec3::new(0.06, 0.05, 0.04)).roughness(0.9, 0.95).bump(1.5).seed(305));
    let glass = scene.add_texture(TextureDesc::new(TexturePattern::StainedGlass, Vec3::new(0.9, 0.3, 0.1), Vec3::new(0.1, 0.3, 0.9)).seed(306));

    // ---- Meshes -------------------------------------------------------------------------------
    let volume = |r| BuildConfig { sdf: SdfMode::Volume { resolution: r }, ..Default::default() };
    let mut ground = primitives::terrain(400.0, 128, 0.0, 1);
    ground.deform(|p| Vec3::new(p.x, 0.0, p.z));
    let ground = add_baked_with(&mut scene, "gallery/courtyard", &ground, BuildConfig { sdf: SdfMode::Heightfield { resolution: 128 }, ..Default::default() });
    let mut puddles = primitives::terrain(24.0, 48, 0.0, 2);
    puddles.deform(|p| Vec3::new(p.x * 1.2, 0.005, p.z * 0.25 + 4.2));
    let puddles = add_baked_with(&mut scene, "gallery/terrace", &puddles, BuildConfig { sdf: SdfMode::None, ..Default::default() });
    let plinth = add_baked_with(&mut scene, "gallery/plinth", &cuboid(Vec3::new(-0.65, 0.0, -0.65), Vec3::new(0.65, PLINTH_HEIGHT, 0.65)), volume(24));
    let sphere = add_baked_with(&mut scene, "gallery/sphere", &primitives::icosphere(detail.pick(5, 6)), volume(32));
    let candle = add_baked_with(&mut scene, "gallery/candle", &primitives::cylinder(48, 8, 0.22, 0.9, true), volume(24));
    let pot = add_baked_with(&mut scene, "gallery/pot", &primitives::cylinder(96, 4, 0.3, 0.8, true), volume(24));
    let knot = add_baked(&mut scene, "gallery/knot", &primitives::torus_knot(2, 3, detail.pick(1024, 2048), detail.pick(64, 128), 0.18));
    let bush = vegetation::bush(161, 1.2, 10);
    let bush_wood = add_baked_with(&mut scene, "gallery/bush wood", &bush.wood, BuildConfig { sdf: SdfMode::None, ..Default::default() });
    let bush_leaves = add_baked_with(&mut scene, "gallery/bush leaves", &bush.foliage, BuildConfig { sdf: SdfMode::None, ..Default::default() });
    let sign = add_baked_with(&mut scene, "gallery/sign", &cuboid(Vec3::new(-1.6, 0.0, -0.04), Vec3::new(1.6, 1.0, 0.04)), volume(16));
    let ring: Mesh = primitives::cylinder(64, 1, 1.1, 0.25, false);
    let ring = add_baked_with(&mut scene, "gallery/basin", &ring, BuildConfig { sdf: SdfMode::None, ..Default::default() });
    // The brick terrace: 8 × 3 m in front of the wet stone, its bricks 3 cm above the courtyard
    // (the mortar joints sink 2.8 cm), framed by a kerb. Lumen sees it as a heightfield of 1.6 cm
    // cells, baked from its displaced surface (lumen_displaced.rs).
    let mut terrace = primitives::terrain(8.0, 32, 0.0, 4);
    terrace.deform(|p| Vec3::new(p.x, BRICK_LEVEL, p.z * 0.375 + 9.0));
    let terrace = add_baked_with(&mut scene, "gallery/brick terrace", &terrace, BuildConfig { sdf: SdfMode::Heightfield { resolution: 512 }, ..Default::default() });
    // The beds: flat 3 m squares (their graphs do the modelling). The cloth's UVs span it (its
    // graph pins the edges by them). Their distance fields (3.3 cm voxels) are baked per
    // material from the displaced surface: mushrooms on one, the cloth's mean surface on the other.
    let mut bed = primitives::terrain(3.0, 12, 0.0, 5);
    bed.deform(|p| Vec3::new(p.x, 0.0, p.z));
    let bed = add_baked_with(&mut scene, "gallery/bed", &bed, volume(96));
    let kerb = add_baked_with(&mut scene, "gallery/kerb", &cuboid(Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 1.0, 0.5)), volume(8));
    let mut pool = primitives::terrain(2.2, 24, 0.0, 3);
    pool.deform(|p| Vec3::new(p.x, 0.0, p.z));
    let pool = add_baked_with(&mut scene, "gallery/pool", &pool, BuildConfig { sdf: SdfMode::None, ..Default::default() });

    // ---- Materials and specimens -------------------------------------------------------------
    let stone = scene.add_material(Material::textured(MaterialKind::Standard, flagstone, TextureMapping::WorldTriplanar, 0.25));
    scene.set_material_name(stone, "gallery/flagstone");
    let plinth_material = scene.add_material(Material::textured(MaterialKind::Standard, marble, TextureMapping::ObjectTriplanar, 0.8));
    scene.set_material_name(plinth_material, "gallery/plinth");
    scene.add_instance(ground, stone, Mat4::IDENTITY);
    let mut g = Gallery { scene: &mut scene, sphere, plinth, plinth_material };

    // 0: car paint — red metallic flakes under a glossy clear coat.
    let top = g.plinth(0.0);
    let paint = g.material("car paint", Material { slab: Slab::clearcoat(1.0, 0.03), ..Material::standard(Vec3::new(0.55, 0.02, 0.03), 0.42, 0.7) });
    g.sphere(paint, top + Vec3::Y * 0.62, Vec3::new(0.95, 0.6, 0.62));
    g.scene.add_instance(knot, paint, Mat4::from_scale_rotation_translation(Vec3::splat(0.85), Quat::from_rotation_x(1.2), top + Vec3::new(0.0, 1.55, 0.0)));

    // 1: velvet — a cushion with a pink sheen over deep burgundy.
    let top = g.plinth(1.0);
    let velvet = g.material(
        "velvet",
        Material {
            slab: Slab::sheen(Vec3::new(0.95, 0.35, 0.5), 0.35),
            ..Material { base_color: Vec3::new(0.22, 0.015, 0.05), roughness: 0.9, ..Material::textured(MaterialKind::Standard, fabric, TextureMapping::ObjectTriplanar, 2.0) }
        },
    );
    g.sphere(velvet, top + Vec3::Y * 0.3, Vec3::new(0.6, 0.32, 0.6));
    g.scene.add_instance(knot, velvet, Mat4::from_scale_rotation_translation(Vec3::splat(0.7), Quat::from_rotation_z(0.4), top + Vec3::new(0.0, 1.25, 0.0)));

    // 2: skin — a head-sized sphere scattering red furthest.
    let top = g.plinth(2.0);
    let skin = g.material("skin", Material { slab: Slab::subsurface(1.0, Vec3::new(1.0, 0.38, 0.22), 0.012), ..Material::standard(Vec3::new(0.78, 0.5, 0.4), 0.45, 0.0) });
    g.sphere(skin, top + Vec3::Y * 0.55, Vec3::new(0.42, 0.55, 0.48));

    // 3: candle wax and a marble egg, the candle lit.
    let top = g.plinth(3.0);
    let wax = g.material("wax", Material { slab: Slab::subsurface(1.0, Vec3::new(1.0, 0.75, 0.45), 0.035), ..Material::standard(Vec3::new(0.92, 0.86, 0.7), 0.35, 0.0) });
    g.scene.add_instance(candle, wax, Mat4::from_translation(top + Vec3::new(-0.25, 0.0, 0.0)));
    let marble_slab = g.material(
        "marble",
        Material { slab: Slab::subsurface(0.7, Vec3::new(1.0, 0.92, 0.85), 0.012), ..Material::standard(Vec3::new(0.86, 0.85, 0.82), 0.22, 0.0) },
    );
    g.sphere(marble_slab, top + Vec3::new(0.32, 0.3, 0.1), Vec3::new(0.24, 0.3, 0.24));
    let flame = g.material("flame", Material { emissive: 40.0, ..Material::standard(Vec3::new(1.0, 0.62, 0.25), 0.5, 0.0) });
    g.sphere(flame, top + Vec3::new(-0.25, 0.95, 0.0), Vec3::new(0.025, 0.05, 0.025));
    g.scene.add_light(Light { source_radius: 0.02, ..Light::point(top + Vec3::new(-0.25, 1.0, 0.0), Vec3::new(1.0, 0.62, 0.3), 1.2, 4.0) });

    // 4: leaves — thin transmission lets the low sun through.
    let top = g.plinth(4.0);
    let leaves = g.material(
        "leaves",
        Material {
            slab: Slab::thin(0.7, Vec3::new(0.75, 1.0, 0.35)),
            roughness: 0.6,
            ..Material::textured(MaterialKind::Foliage, leaf, TextureMapping::MeshUv, 1.0)
        },
    );
    let wood = g.material("bush wood", Material::textured(MaterialKind::Standard, bark, TextureMapping::MeshUv, 1.2));
    for (mesh, material) in [(bush_wood, wood), (bush_leaves, leaves)] {
        g.scene.add_instance(mesh, material, Mat4::from_translation(top));
    }
    // A warm lamp behind the bush: its light comes through the leaves.
    g.scene.add_light(Light { source_radius: 0.1, ..Light::point(top + Vec3::new(0.15, 1.3, -0.9), Vec3::new(1.0, 0.75, 0.45), 14.0, 5.0) });

    // 5: brushed metal — highlights stretched along a pot's axis; dielectrics of different IOR and a
    // specular colour around it.
    let top = g.plinth(5.0);
    let brushed = g.material(
        "brushed metal",
        Material { slab: Slab { anisotropy: 0.8, anisotropy_rotation: std::f32::consts::FRAC_PI_2, ..Default::default() }, ..Material::standard(Vec3::new(0.91, 0.92, 0.93), 0.3, 1.0) },
    );
    g.scene.add_instance(pot, brushed, Mat4::from_translation(top + Vec3::new(0.0, 0.0, -0.2)));
    for (i, (name, slab)) in [
        ("ior 1.33", Slab { ior: 1.33, ..Default::default() }),
        ("ior 2.4", Slab { ior: 2.4, ..Default::default() }),
        ("specular colour", Slab { specular_color: Vec3::new(0.2, 1.0, 2.0), ior: 1.8, ..Default::default() }),
    ]
    .into_iter()
    .enumerate()
    {
        let m = g.material(name, Material { slab, ..Material::standard(Vec3::splat(0.02), 0.08, 0.0) });
        g.sphere(m, top + Vec3::new(-0.42 + 0.42 * i as f32, 0.16, 0.42), Vec3::splat(0.15));
    }

    // 6: graph — animated lava in a pool, and a lava egg.
    let top = g.plinth(6.0);
    let lava_graph = g.scene.add_material_graph(presets::lava());
    let lava = g.material("lava", Material { graph: Some(lava_graph), ..Material::standard(Vec3::splat(0.1), 0.8, 0.0) });
    g.sphere(lava, top + Vec3::Y * 0.5, Vec3::new(0.45, 0.5, 0.45));
    g.scene.add_instance(pool, lava, Mat4::from_translation(Vec3::new(top.x, 0.02, 3.0)));
    g.scene.add_instance(ring, plinth_material, Mat4::from_translation(Vec3::new(top.x, 0.0, 3.0)));

    // 7: graph — stylised water in a basin, a brighter instance of the same graph on the plinth.
    let top = g.plinth(7.0);
    let water_graph = g.scene.add_material_graph(presets::stylized_water());
    let water = g.material("stylised water", Material { graph: Some(water_graph), ..Material::standard(Vec3::splat(0.5), 0.1, 0.0) });
    g.scene.add_instance(pool, water, Mat4::from_translation(Vec3::new(top.x, 0.04, 3.0)));
    g.scene.add_instance(ring, plinth_material, Mat4::from_translation(Vec3::new(top.x, 0.0, 3.0)));
    let tropical = g.material("tropical water", Material { graph: Some(water_graph), ..Material::standard(Vec3::splat(0.5), 0.1, 0.0) });
    g.scene.set_material_parameter(tropical, "Shallow Color", ParameterValue::Vector([0.1, 0.75, 0.6, 1.0]));
    g.scene.set_material_parameter(tropical, "Deep Color", ParameterValue::Vector([0.0, 0.25, 0.35, 1.0]));
    g.sphere(tropical, top + Vec3::Y * 0.5, Vec3::splat(0.5));

    // Graph — wet stone with puddles: the terrace in front of the row (the same flagstone).
    let wet_graph = g.scene.add_material_graph(presets::wet_stone());
    let wet = g.material("wet stone", Material { graph: Some(wet_graph), ..Material::textured(MaterialKind::Standard, flagstone, TextureMapping::WorldTriplanar, 0.25) });
    g.scene.add_instance(puddles, wet, Mat4::IDENTITY);

    // Graph — a brick pavement whose Displacement output carves the mortar joints (Nanite
    // tessellation driven by the graph, not a texture): the terrace in front of the wet stone.
    let brick_graph = g.scene.add_material_graph(presets::brick_pavement());
    let bricks = g.material(
        "brick pavement",
        Material { graph: Some(brick_graph), displacement: Some(Displacement::height(0.035, 0.9)), ..Material::standard(Vec3::new(0.4, 0.16, 0.09), 0.85, 0.0) },
    );
    g.scene.add_instance(terrace, bricks, Mat4::IDENTITY);
    let kerb_material = g.material("kerb", Material::textured(MaterialKind::Standard, marble, TextureMapping::WorldTriplanar, 0.8));
    let kerb_height = BRICK_LEVEL + 0.015;
    for (center, size) in [
        (Vec3::new(0.0, 0.0, 7.45), Vec3::new(8.2, kerb_height, 0.1)),
        (Vec3::new(0.0, 0.0, 10.55), Vec3::new(8.2, kerb_height, 0.1)),
        (Vec3::new(-4.05, 0.0, 9.0), Vec3::new(0.1, kerb_height, 3.0)),
        (Vec3::new(4.05, 0.0, 9.0), Vec3::new(0.1, kerb_height, 3.0)),
    ] {
        g.scene.add_instance(kerb, kerb_material, Mat4::from_scale_rotation_translation(size, Quat::IDENTITY, center));
    }

    // Graph — mushrooms: a vector displacement (overhanging caps on stems) grows them out of a
    // flat bed left of the bricks.
    let mushroom_graph = g.scene.add_material_graph(presets::mushrooms());
    let mushrooms = g.material(
        "mushrooms",
        Material { graph: Some(mushroom_graph), displacement: Some(Displacement::height(presets::MUSHROOM_MAGNITUDE, 0.0)), ..Material::standard(Vec3::new(0.1, 0.15, 0.05), 0.9, 0.0) },
    );
    g.scene.add_instance(bed, mushrooms, Mat4::from_translation(Vec3::new(-BED_X, MOSS_LEVEL, BED_Z)));
    // Graph — a rippling cloth: an animated displacement (travelling waves) on a sheet stretched
    // over a frame right of the bricks; a dynamic shadow caster.
    let cloth_graph = g.scene.add_material_graph(presets::rippling_cloth());
    let cloth = g.material(
        "rippling cloth",
        Material { graph: Some(cloth_graph), displacement: Some(Displacement::height(0.16, 0.5)), ..Material::standard(Vec3::new(0.08, 0.2, 0.55), 0.42, 0.0) },
    );
    g.scene.add_instance(bed, cloth, Mat4::from_translation(Vec3::new(BED_X, CLOTH_LEVEL, BED_Z)));
    for (x, height) in [(-BED_X, MOSS_LEVEL + 0.03), (BED_X, CLOTH_LEVEL)] {
        for (center, size) in [
            (Vec3::new(x, 0.0, BED_Z - 1.55), Vec3::new(3.2, height, 0.1)),
            (Vec3::new(x, 0.0, BED_Z + 1.55), Vec3::new(3.2, height, 0.1)),
            (Vec3::new(x - 1.55, 0.0, BED_Z), Vec3::new(0.1, height, 3.0)),
            (Vec3::new(x + 1.55, 0.0, BED_Z), Vec3::new(0.1, height, 3.0)),
        ] {
            g.scene.add_instance(kerb, kerb_material, Mat4::from_scale_rotation_translation(size, Quat::IDENTITY, center));
        }
    }

    // An unlit sign behind the row (KHR_materials_unlit): the stained-glass pattern, unshaded.
    let unlit = g.material("unlit sign", Material { slab: Slab { unlit: true, ..Default::default() }, ..Material::textured(MaterialKind::Standard, glass, TextureMapping::MeshUv, 1.0) });
    g.scene.add_instance(sign, unlit, Mat4::from_translation(Vec3::new(0.0, 2.6, -3.5)));
    g.scene.add_instance(plinth, plinth_material, Mat4::from_scale_rotation_translation(Vec3::new(0.25, 2.6, 0.08), Quat::IDENTITY, Vec3::new(-1.4, 0.0, -3.52)));
    g.scene.add_instance(plinth, plinth_material, Mat4::from_scale_rotation_translation(Vec3::new(0.25, 2.6, 0.08), Quat::IDENTITY, Vec3::new(1.4, 0.0, -3.52)));

    // A warm spot light on the velvet and the car paint.
    let spot_from = Vec3::new(-7.5, 4.5, 4.0);
    g.scene.add_light(Light::spot(spot_from, Vec3::new(-6.4, 1.2, 0.0) - spot_from, Vec3::new(1.0, 0.85, 0.65), 18.0, 12.0, 0.5));
    scene
}

fn initial_camera() -> Camera {
    let mut camera = Camera { fov_y: 50f32.to_radians(), near: 0.05, ..Default::default() };
    camera.position = Vec3::new(0.0, 2.8, 12.5);
    camera.look_at(Vec3::new(0.0, 1.2, 0.0));
    camera
}

pub fn build(detail: Detail) -> SceneSetup {
    SceneSetup { scene: build_scene(detail), camera: initial_camera(), sun: SUN, floor, animation: None, motion: Default::default() }
}
