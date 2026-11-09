//! A Gothic cathedral: 11-bay nave with aisles, clustered piers, triforium and stained-glass
//! clerestory under quadripartite rib vaults, flying buttresses, a polygonal apse, a west
//! front with rose window and twin towers with spires — furnished with pews, an altar,
//! candelabra and chandeliers lit by candle lights, and wrought-iron screens around the choir
//! (alpha-tested cards).
//!
//! Every stone surface is densely tessellated and chisel-displaced, so the cathedral is
//! hundreds of millions of source triangles once instanced — the case virtual geometry exists for.

use std::sync::Arc;

use glam::{Mat4, Quat, Vec2, Vec3};
use helios_geometry::modeling::{self, circle, cuboid, extrude, lathe, merge, pointed_arch, subdivide_to, sweep, tube};
use helios_geometry::{primitives, BuildConfig, Mesh, SdfMode, Vertex};
use helios_render::{
    Camera, Image, ImageTexture, Light, Material, MaterialHandle, MaterialKind, MeshHandle, Scene, Sun, TextureDesc, TextureMapping,
    TexturePattern,
};

use super::{add_baked, add_baked_with, Detail, SceneSetup};

const BAY: f32 = 7.0;
const BAYS: usize = 11;
const LENGTH: f32 = BAY * BAYS as f32;
const NAVE_HALF: f32 = 7.0;
const AISLE_WALL: f32 = 13.5;
const WALL_T: f32 = 1.2;
const ARCADE_SPRING: f32 = 9.0;
const ARCADE_APEX: f32 = 13.5;
const WALL_TOP: f32 = 28.0;
const VAULT_RISE: f32 = 9.0;
const AISLE_SPRING: f32 = 10.0;
const AISLE_RISE: f32 = 3.2;
const SUN: (f32, f32, f32) = (1.15, 0.72, 10.0);

pub fn floor_height(x: f32, z: f32) -> f32 {
    terrain_height(x, z).max(if inside_footprint(x, z) { 0.0 } else { f32::MIN })
}

fn inside_footprint(x: f32, z: f32) -> bool {
    (-9.0..LENGTH + NAVE_HALF + 1.0).contains(&x) && z.abs() < AISLE_WALL + 4.0
}

/// Gently rolling ground around a flat precinct.
fn terrain_height(x: f32, z: f32) -> f32 {
    let p = Vec3::new(x, 0.0, z) * 0.012;
    let hills = helios_geometry::noise::fbm(p, 4, 21) * 10.0 + helios_geometry::noise::fbm(p * 3.0, 3, 22) * 2.0;
    let center = Vec2::new(x - LENGTH * 0.5, z);
    let flat = super::smoothstep(70.0, 160.0, center.length());
    -0.25 + hills.max(-3.0) * flat
}

struct Mats {
    ashlar: MaterialHandle,
    smooth: MaterialHandle,
    vault: MaterialHandle,
    floor: MaterialHandle,
    slate: MaterialHandle,
    wood: MaterialHandle,
    gold: MaterialHandle,
    iron: MaterialHandle,
    wax: MaterialHandle,
    flame: MaterialHandle,
    cloth: MaterialHandle,
    glass: [MaterialHandle; 3],
    plaza: MaterialHandle,
    terrain: MaterialHandle,
}

fn materials(scene: &mut Scene) -> Mats {
    let t = |scene: &mut Scene, d: TextureDesc| scene.add_texture(d);
    let ashlar = t(scene, TextureDesc::new(TexturePattern::Ashlar, Vec3::new(0.46, 0.41, 0.34), Vec3::new(0.36, 0.32, 0.27)).roughness(0.8, 0.9).bump(1.2).seed(31));
    let smooth = t(scene, TextureDesc::new(TexturePattern::Plaster, Vec3::new(0.5, 0.45, 0.38), Vec3::new(0.43, 0.39, 0.33)).roughness(0.7, 0.8).scale(2.0).bump(0.6).seed(32));
    let carved = t(scene, TextureDesc::new(TexturePattern::CarvedStone, Vec3::new(0.52, 0.47, 0.4), Vec3::new(0.4, 0.36, 0.31)).roughness(0.75, 0.85).bump(1.0).seed(33));
    let marble = t(scene, TextureDesc::new(TexturePattern::MarbleTiles, Vec3::new(0.78, 0.76, 0.72), Vec3::new(0.08, 0.075, 0.075)).roughness(0.25, 0.3).bump(0.4).seed(34));
    let slate = t(scene, TextureDesc::new(TexturePattern::Slate, Vec3::new(0.12, 0.13, 0.15), Vec3::new(0.18, 0.18, 0.2)).roughness(0.6, 0.7).seed(35));
    let wood = t(scene, TextureDesc::new(TexturePattern::Wood, Vec3::new(0.24, 0.13, 0.06), Vec3::new(0.13, 0.065, 0.03)).roughness(0.45, 0.6).seed(36));
    let hammered = t(scene, TextureDesc::new(TexturePattern::Hammered, Vec3::ONE, Vec3::ONE).roughness(0.2, 0.3).scale(2.0).bump(0.5).seed(37));
    let fabric = t(scene, TextureDesc::new(TexturePattern::Fabric, Vec3::new(0.32, 0.015, 0.02), Vec3::new(0.75, 0.55, 0.15)).roughness(0.85, 0.8).seed(38));
    let flagstone = t(scene, TextureDesc::new(TexturePattern::Flagstone, Vec3::new(0.42, 0.4, 0.37), Vec3::new(0.3, 0.29, 0.27)).roughness(0.85, 0.9).scale(1.0).seed(39));
    let glass = [40u32, 41, 42].map(|seed| {
        t(scene, TextureDesc::new(TexturePattern::StainedGlass, Vec3::ONE, Vec3::ONE).scale(if seed == 42 { 0.7 } else { 1.0 }).bump(0.4).seed(seed))
    });
    let grass = t(scene, TextureDesc::new(TexturePattern::Grass, Vec3::new(0.16, 0.24, 0.06), Vec3::new(0.08, 0.12, 0.03)).roughness(0.92, 0.9).seed(43));
    let dirt = t(scene, TextureDesc::new(TexturePattern::Dirt, Vec3::new(0.24, 0.17, 0.11), Vec3::new(0.22, 0.2, 0.18)).roughness(0.95, 0.8).bump(0.5).seed(44));
    let rock = t(scene, TextureDesc::new(TexturePattern::Rock, Vec3::new(0.3, 0.28, 0.25), Vec3::new(0.14, 0.13, 0.12)).roughness(0.75, 0.9).seed(45));

    let world = |kind, texture, scale: f32, base: Vec3| Material {
        kind,
        base_color: base,
        roughness: 0.5,
        texture: Some(texture),
        mapping: TextureMapping::WorldTriplanar,
        uv_scale: scale,
        ..Default::default()
    };
    let mut add = |m: Material| scene.add_material(m);
    Mats {
        ashlar: add(world(MaterialKind::Standard, ashlar, 1.0 / 3.0, Vec3::ONE)),
        smooth: add(world(MaterialKind::Standard, smooth, 0.25, Vec3::ONE)),
        vault: add(world(MaterialKind::Standard, carved, 0.2, Vec3::new(1.05, 1.02, 0.97))),
        floor: add(world(MaterialKind::Standard, marble, 0.5, Vec3::ONE)),
        slate: add(world(MaterialKind::Standard, slate, 0.25, Vec3::ONE)),
        wood: add(Material { normal_strength: 0.7, ..world(MaterialKind::Standard, wood, 0.5, Vec3::ONE) }),
        gold: add(Material {
            kind: MaterialKind::Metal,
            base_color: Vec3::new(1.0, 0.74, 0.32),
            roughness: 0.25,
            metallic: 1.0,
            detail_scale: 4.0,
            texture: Some(hammered),
            mapping: TextureMapping::ObjectTriplanar,
            uv_scale: 3.0,
            normal_strength: 0.4,
            ..Default::default()
        }),
        iron: add(Material { kind: MaterialKind::Metal, base_color: Vec3::splat(0.06), roughness: 0.55, metallic: 1.0, detail_scale: 6.0, ..Default::default() }),
        wax: add(Material::standard(Vec3::new(0.85, 0.8, 0.65), 0.4, 0.0)),
        flame: add(Material { base_color: Vec3::new(1.0, 0.55, 0.15), emissive: 60.0, roughness: 1.0, ..Default::default() }),
        cloth: add(Material { kind: MaterialKind::Standard, base_color: Vec3::ONE, texture: Some(fabric), mapping: TextureMapping::MeshUv, uv_scale: 0.5, roughness: 0.8, ..Default::default() }),
        glass: glass.map(|tex| {
            add(Material {
                kind: MaterialKind::Glass,
                base_color: Vec3::ONE,
                emissive: 1.0,
                texture: Some(tex),
                mapping: TextureMapping::MeshUv,
                uv_scale: 0.3,
                ..Default::default()
            })
        }),
        plaza: add(world(MaterialKind::Standard, flagstone, 0.25, Vec3::ONE)),
        terrain: add(Material { kind: MaterialKind::Terrain, base_color: Vec3::ONE, layers: [Some(grass), Some(dirt), Some(rock), Some(grass)], ..Default::default() }),
    }
}

// ---- Shapes -----------------------------------------------------------------------------------------

/// Lancet window opening (counter-clockwise polygon), centred on x = 0.
fn lancet(width: f32, sill: f32, spring: f32, apex: f32) -> Vec<Vec2> {
    let mut hole = vec![Vec2::new(-width * 0.5, sill), Vec2::new(width * 0.5, sill)];
    let arch = pointed_arch(width, apex - spring, 12);
    for p in arch.iter().rev() {
        let q = Vec2::new(p.x, p.y + spring);
        if hole.last().is_none_or(|l| l.distance(q) > 1e-3) {
            hole.push(q);
        }
    }
    hole.dedup_by(|a, b| a.distance(*b) < 1e-4);
    if hole.first().unwrap().distance(*hole.last().unwrap()) < 1e-4 {
        hole.pop();
    }
    hole
}

fn translate_poly(poly: &[Vec2], offset: Vec2) -> Vec<Vec2> {
    poly.iter().map(|p| *p + offset).collect()
}

/// Wall in the XY plane centred on z = 0 (extruded `thickness`).
fn wall(outline: &[Vec2], holes: &[Vec<Vec2>], thickness: f32) -> Mesh {
    extrude(outline, holes, thickness).transformed(Mat4::from_translation(Vec3::new(0.0, 0.0, -thickness * 0.5)))
}

/// Densifies a flat-faced mesh and adds subtle stone relief.
fn stone(mesh: &Mesh, edge: f32, relief: f32, seed: u32) -> Mesh {
    let mut m = subdivide_to(mesh, edge);
    if relief > 0.0 {
        modeling::displace(&mut m, relief, 3.0, 3, seed);
    }
    m
}

/// Glass pane filling `hole` (XY plane, thin).
fn glass_pane(hole: &[Vec2]) -> Mesh {
    extrude(hole, &[], 0.05).transformed(Mat4::from_translation(Vec3::new(0.0, 0.0, -0.025)))
}

/// Mullions and a ring of tracery for a lancet of `width` between `sill` and `apex`.
fn tracery(width: f32, sill: f32, spring: f32, apex: f32, detail: Detail) -> Mesh {
    let mut parts = Vec::new();
    let mullion = 0.16;
    for x in [-width / 6.0, width / 6.0] {
        parts.push(cuboid(Vec3::new(x - mullion * 0.5, sill, -0.2), Vec3::new(x + mullion * 0.5, spring + 0.2, 0.2)));
    }
    let ring_y = spring + (apex - spring) * 0.45;
    let ring_r = width * 0.22;
    let ring_path: Vec<Vec3> = (0..=40).map(|i| {
        let a = i as f32 / 40.0 * std::f32::consts::TAU;
        Vec3::new(a.cos() * ring_r, ring_y + a.sin() * ring_r, 0.0)
    }).collect();
    parts.push(sweep(&circle(0.08, detail.pick(8, 12)), &ring_path, Vec3::Z, &|_| 1.0, false));
    // Arch moulding around the opening.
    let arch: Vec<Vec3> = pointed_arch(width, apex - spring, 24).iter().map(|p| Vec3::new(p.x, p.y + spring, 0.0)).collect();
    parts.push(sweep(&circle(0.1, detail.pick(8, 12)), &arch, Vec3::Z, &|_| 1.0, true));
    merge(&parts)
}

/// Clustered pier with base, fluted shafts and a carved capital; plus the vaulting shaft that
/// rises on the nave side (+Z) to the vault springing.
fn pier(detail: Detail) -> Mesh {
    let mut profile = vec![Vec2::new(1.35, 0.0), Vec2::new(1.35, 0.45), Vec2::new(1.2, 0.5), Vec2::new(1.2, 0.7)];
    // Base mouldings: torus - scotia - torus.
    for i in 0..=12 {
        let a = i as f32 / 12.0 * std::f32::consts::PI;
        profile.push(Vec2::new(1.08 + 0.12 * a.sin(), 0.7 + 0.25 * i as f32 / 12.0));
    }
    profile.push(Vec2::new(0.98, 1.05));
    for i in 0..=8 {
        profile.push(Vec2::new(1.0 + 0.06 * (i as f32 / 8.0 * std::f32::consts::PI).sin(), 1.05 + 0.15 * i as f32 / 8.0));
    }
    // Shaft.
    let rows = detail.pick(40, 90);
    for i in 1..rows {
        profile.push(Vec2::new(0.95, 1.2 + (ARCADE_SPRING - 2.2) * i as f32 / rows as f32));
    }
    // Capital: bell with foliage, abacus.
    for i in 0..=16 {
        let t = i as f32 / 16.0;
        profile.push(Vec2::new(0.95 + 0.4 * t * t, ARCADE_SPRING - 1.0 + 0.8 * t));
    }
    profile.extend_from_slice(&[Vec2::new(1.45, ARCADE_SPRING - 0.2), Vec2::new(1.45, ARCADE_SPRING), Vec2::new(0.0, ARCADE_SPRING)]);
    let segments = detail.pick(128, 224);
    let clustered = |a: f32, y: f32| {
        // Eight engaged shafts around a core, merging into the capital.
        let lobes = (4.0 * a).cos().abs().powf(6.0);
        let in_shaft = super::smoothstep(1.15, 1.3, y) * (1.0 - super::smoothstep(ARCADE_SPRING - 1.0, ARCADE_SPRING - 0.4, y));
        let capital = super::smoothstep(ARCADE_SPRING - 1.0, ARCADE_SPRING - 0.3, y) * (1.0 - super::smoothstep(ARCADE_SPRING - 0.25, ARCADE_SPRING - 0.2, y));
        let leaves = helios_geometry::noise::ridged(Vec3::new(a.cos() * 3.0, y * 2.5, a.sin() * 3.0), 3, 5) * 0.12;
        1.0 + in_shaft * (0.18 * lobes - 0.03) + capital * leaves
    };
    let mut body = lathe(&profile, segments, &clustered, true);
    if detail == Detail::High {
        body = modeling::subdivide(&body, 1);
    }
    modeling::displace(&mut body, 0.004, 9.0, 3, 50);

    // Vaulting shaft on the nave face.
    let shaft_profile: Vec<Vec2> = (0..=detail.pick(30, 60)).map(|i| Vec2::new(0.24, ARCADE_SPRING + (WALL_TOP - ARCADE_SPRING) * i as f32 / detail.pick(30, 60) as f32)).collect();
    let shaft = lathe(&shaft_profile, detail.pick(24, 40), &|_, _| 1.0, true).transformed(Mat4::from_translation(Vec3::new(0.0, 0.0, 1.05)));
    let shaft_capital = lathe(
        &[Vec2::new(0.24, WALL_TOP - 0.6), Vec2::new(0.45, WALL_TOP - 0.15), Vec2::new(0.45, WALL_TOP), Vec2::new(0.0, WALL_TOP)],
        detail.pick(24, 40),
        &|a, _| 1.0 + 0.08 * (a * 6.0).sin(),
        true,
    )
    .transformed(Mat4::from_translation(Vec3::new(0.0, 0.0, 1.05)));
    merge(&[body, shaft, shaft_capital])
}

/// Nave wall bay (XY plane, x across the bay): arcade cut-out, triforium arcade, clerestory window.
fn nave_wall(detail: Detail) -> (Mesh, Vec<Vec2>, Vec<Vec<Vec2>>) {
    let half = BAY * 0.5;
    let span = 5.2;
    let mut outline = vec![Vec2::new(-half, ARCADE_SPRING), Vec2::new(-span * 0.5, ARCADE_SPRING)];
    for p in pointed_arch(span, ARCADE_APEX - 0.6 - ARCADE_SPRING, 16).iter().skip(1).take(31) {
        outline.push(Vec2::new(p.x, p.y + ARCADE_SPRING));
    }
    outline.extend_from_slice(&[Vec2::new(span * 0.5, ARCADE_SPRING), Vec2::new(half, ARCADE_SPRING), Vec2::new(half, WALL_TOP), Vec2::new(-half, WALL_TOP)]);
    outline.dedup_by(|a, b| a.distance(*b) < 1e-3);
    let mut holes = Vec::new();
    for x in [-1.9f32, 0.0, 1.9] {
        holes.push(translate_poly(&lancet(1.2, 14.3, 16.2, 17.2), Vec2::new(x, 0.0)));
    }
    let clerestory = lancet(4.0, 18.6, 24.6, WALL_TOP - 0.7);
    holes.push(clerestory.clone());
    let mesh = stone(&wall(&outline, &holes, 1.4), detail.pick(0.7, 0.35), 0.006, 51);
    (mesh, clerestory, holes)
}

fn arcade_moulding(detail: Detail) -> Mesh {
    let span = 5.2;
    let arch: Vec<Vec3> = pointed_arch(span, ARCADE_APEX - 0.6 - ARCADE_SPRING, detail.pick(32, 64))
        .iter()
        .map(|p| Vec3::new(p.x, p.y + ARCADE_SPRING, 0.0))
        .collect();
    // Rolled moulding: a stepped section with three rolls.
    let mut section = Vec::new();
    for i in 0..48 {
        let a = i as f32 / 48.0 * std::f32::consts::TAU;
        let lobes = 1.0 + 0.18 * (3.0 * a).cos().abs().powf(3.0);
        section.push(Vec2::new(a.cos() * 0.5 * lobes, a.sin() * 0.75 * lobes));
    }
    sweep(&section, &arch, Vec3::Z, &|_| 1.0, true)
}

/// One bay of the nave vault: shell plus transverse, diagonal and ridge ribs with a boss.
fn vault_bay(span: f32, rise: f32, spring: f32, detail: Detail) -> Mesh {
    let half = BAY * 0.5;
    let profile = pointed_arch(span, rise, detail.pick(24, 48));
    // Shell: band between the intrados and an offset extrados, extruded along the bay.
    let thickness = 0.35;
    let mut band: Vec<Vec2> = profile.iter().map(|p| Vec2::new(p.x, p.y + spring)).collect();
    let outer: Vec<Vec2> = profile.iter().rev().map(|p| Vec2::new(p.x, p.y + spring + thickness)).collect();
    band.extend(outer);
    let shell = extrude(&band, &[], BAY).transformed(Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2) * Mat4::from_translation(Vec3::new(0.0, 0.0, -half)));
    let shell = stone(&shell, detail.pick(0.8, 0.4), 0.0, 52);

    let rib_section: Vec<Vec2> = (0..detail.pick(12, 20)).map(|i| {
        let a = i as f32 / detail.pick(12, 20) as f32 * std::f32::consts::TAU;
        let lobes = 1.0 + 0.25 * (2.0 * a).cos().powi(4);
        Vec2::new(a.cos() * 0.16 * lobes, a.sin() * 0.24)
    }).collect();
    let height = |u: f32| {
        // Intrados height at offset u across the span (interpolated from the profile).
        let mut best = 0.0;
        for w in profile.windows(2) {
            if (w[0].x - u) * (w[1].x - u) <= 0.0 {
                let t = (u - w[0].x) / (w[1].x - w[0].x).max(1e-6);
                best = w[0].y + (w[1].y - w[0].y) * t;
            }
        }
        best + spring - 0.12
    };
    let steps = detail.pick(32, 64);
    let mut ribs = Vec::new();
    // Transverse rib at the bay start (the next bay supplies the end).
    let transverse: Vec<Vec3> = profile.iter().map(|p| Vec3::new(-half, p.y + spring - 0.12, p.x)).collect();
    ribs.push(sweep(&rib_section, &transverse, Vec3::X, &|_| 1.4, true));
    for dir in [1.0f32, -1.0] {
        let path: Vec<Vec3> = (0..=steps)
            .map(|i| {
                let t = i as f32 / steps as f32;
                let u = (t - 0.5) * span;
                Vec3::new((t - 0.5) * BAY * dir, height(u), u)
            })
            .collect();
        ribs.push(sweep(&rib_section, &path, Vec3::Y, &|_| 1.0, true));
    }
    let ridge = [Vec3::new(-half, height(0.0), 0.0), Vec3::new(half, height(0.0), 0.0)];
    ribs.push(sweep(&rib_section, &ridge, Vec3::Y, &|_| 0.8, true));
    let mut boss = primitives::rock(9, detail.pick(3, 4));
    boss.deform(|p| p * Vec3::new(0.45, 0.3, 0.45));
    ribs.push(boss.transformed(Mat4::from_translation(Vec3::new(0.0, height(0.0) - 0.1, 0.0))));
    let mut ribs = merge(&ribs);
    modeling::displace(&mut ribs, 0.003, 10.0, 2, 53);
    merge(&[shell, ribs])
}

fn aisle_wall(detail: Detail) -> (Mesh, Vec<Vec2>) {
    let half = BAY * 0.5;
    let outline = [Vec2::new(-half, 0.0), Vec2::new(half, 0.0), Vec2::new(half, AISLE_SPRING + AISLE_RISE + 0.5), Vec2::new(-half, AISLE_SPRING + AISLE_RISE + 0.5)];
    let window = lancet(3.2, 3.2, 9.2, 12.0);
    (stone(&wall(&outline, std::slice::from_ref(&window), WALL_T), detail.pick(0.7, 0.35), 0.006, 54), window)
}

/// Exterior buttress pier with a pinnacle, plus the flying arch to the clerestory wall.
/// Local frame: x along the nave, +z outward (pier at z = 0 is the aisle wall's outer face).
fn buttress(detail: Detail) -> Mesh {
    let mut parts = vec![cuboid(Vec3::new(-0.75, 0.0, 0.0), Vec3::new(0.75, 21.0, 3.2))];
    // Weathering set-offs.
    parts.push(cuboid(Vec3::new(-0.9, 0.0, 0.0), Vec3::new(0.9, 6.0, 3.6)));
    let pinnacle = lathe(
        &[Vec2::new(0.9, 0.0), Vec2::new(0.9, 1.5), Vec2::new(0.75, 1.7), Vec2::new(0.6, 2.0), Vec2::new(0.02, 7.0)],
        8,
        &|_, y| 1.0 + 0.12 * (y * 5.0).sin().max(0.0),
        true,
    );
    parts.push(pinnacle.transformed(Mat4::from_translation(Vec3::new(0.0, 21.0, 1.6))));
    // Flying buttress: an arc from the pier top to the nave clerestory.
    let reach = AISLE_WALL - NAVE_HALF + WALL_T;
    let path: Vec<Vec3> = (0..=detail.pick(24, 48))
        .map(|i| {
            let t = i as f32 / detail.pick(24, 48) as f32;
            let z = 1.0 - t * (reach + 1.0);
            let y = 19.5 + t * 6.5 + (t * std::f32::consts::PI).sin() * 1.6;
            Vec3::new(0.0, y, z)
        })
        .collect();
    let section = [Vec2::new(-0.45, -0.7), Vec2::new(0.45, -0.7), Vec2::new(0.45, 0.7), Vec2::new(-0.45, 0.7)];
    parts.push(sweep(&section, &path, Vec3::X, &|_| 1.0, true));
    stone(&merge(&parts), detail.pick(0.8, 0.4), 0.008, 55)
}

fn west_front(detail: Detail) -> (Mesh, Vec<Vec2>, Vec<Vec2>) {
    let half = AISLE_WALL + WALL_T * 0.5;
    let outline = [
        Vec2::new(-half, 0.0),
        Vec2::new(half, 0.0),
        Vec2::new(half, 16.0),
        Vec2::new(NAVE_HALF + 0.7, 18.0),
        Vec2::new(NAVE_HALF + 0.7, 36.0),
        Vec2::new(0.0, 45.0),
        Vec2::new(-NAVE_HALF - 0.7, 36.0),
        Vec2::new(-NAVE_HALF - 0.7, 18.0),
        Vec2::new(-half, 16.0),
    ];
    let portal = lancet(4.6, 0.0, 6.0, 9.5);
    let rose: Vec<Vec2> = circle(4.6, 64).iter().map(|p| *p + Vec2::new(0.0, 22.5)).collect();
    // The portal touches the ground: cut it as a notch in the outline instead of a hole.
    let mut notched = vec![outline[0], Vec2::new(-2.3, 0.0)];
    for p in portal.iter().skip(2).rev() {
        notched.push(*p);
    }
    notched.push(Vec2::new(2.3, 0.0));
    notched.extend_from_slice(&outline[1..]);
    notched.dedup_by(|a, b| a.distance(*b) < 1e-3);
    let side_windows = [translate_poly(&lancet(2.2, 4.0, 10.0, 12.5), Vec2::new(-10.5, 0.0)), translate_poly(&lancet(2.2, 4.0, 10.0, 12.5), Vec2::new(10.5, 0.0))];
    let mut holes = vec![rose.clone()];
    holes.extend(side_windows.iter().cloned());
    let mesh = stone(&wall(&notched, &holes, 2.0), detail.pick(0.7, 0.35), 0.006, 56);
    (mesh, rose, side_windows[0].clone())
}

fn rose_tracery(detail: Detail) -> Mesh {
    let mut parts = Vec::new();
    let sides = detail.pick(8, 12);
    let ring = |r: f32, thick: f32| {
        let path: Vec<Vec3> = (0..=96).map(|i| {
            let a = i as f32 / 96.0 * std::f32::consts::TAU;
            Vec3::new(a.cos() * r, a.sin() * r, 0.0)
        }).collect();
        sweep(&circle(thick, sides), &path, Vec3::Z, &|_| 1.0, false)
    };
    parts.push(ring(4.5, 0.18));
    parts.push(ring(1.4, 0.12));
    for i in 0..12 {
        let a = i as f32 / 12.0 * std::f32::consts::TAU;
        let d = Vec3::new(a.cos(), a.sin(), 0.0);
        parts.push(tube(&[d * 1.4, d * 4.4], &[0.09, 0.09], sides, true));
        let c = d * 3.0;
        let petal: Vec<Vec3> = (0..=48).map(|k| {
            let b = k as f32 / 48.0 * std::f32::consts::TAU;
            c + Vec3::new(b.cos(), b.sin(), 0.0) * 0.75
        }).collect();
        parts.push(sweep(&circle(0.07, sides), &petal, Vec3::Z, &|_| 1.0, false));
    }
    merge(&parts)
}

/// Square tower with belfry openings and an octagonal spire, centred on the origin.
fn tower(detail: Detail) -> Mesh {
    let w = 4.2;
    let height = 52.0;
    let mut parts = Vec::new();
    for k in 0..4 {
        let outline = [Vec2::new(-w, 0.0), Vec2::new(w, 0.0), Vec2::new(w, height), Vec2::new(-w, height)];
        let holes = vec![
            translate_poly(&lancet(1.4, 30.0, 34.0, 36.0), Vec2::new(-1.6, 0.0)),
            translate_poly(&lancet(1.4, 30.0, 34.0, 36.0), Vec2::new(1.6, 0.0)),
            lancet(2.6, 40.0, 47.0, 49.5),
        ];
        let face = wall(&outline, &holes, 1.0).transformed(Mat4::from_rotation_y(k as f32 * std::f32::consts::FRAC_PI_2) * Mat4::from_translation(Vec3::new(0.0, 0.0, w - 0.5)));
        parts.push(face);
    }
    // Corner buttresses.
    for (sx, sz) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let c = Vec3::new(sx * w, 0.0, sz * w);
        parts.push(cuboid(c - Vec3::new(0.8, 0.0, 0.8), c + Vec3::new(0.8, height + 1.0, 0.8)));
        let pinnacle = lathe(&[Vec2::new(0.8, 0.0), Vec2::new(0.6, 1.0), Vec2::new(0.02, 6.0)], 8, &|_, y| 1.0 + 0.15 * (y * 6.0).sin().max(0.0), true);
        parts.push(pinnacle.transformed(Mat4::from_translation(c + Vec3::Y * (height + 1.0))));
    }
    parts.push(cuboid(Vec3::new(-w - 0.4, height, -w - 0.4), Vec3::new(w + 0.4, height + 0.8, w + 0.4)));
    // Octagonal spire with crockets.
    let rows = detail.pick(60, 120);
    let spire_profile: Vec<Vec2> = (0..=rows).map(|i| {
        let t = i as f32 / rows as f32;
        Vec2::new(3.6 * (1.0 - t) + 0.05, height + 0.8 + t * 30.0)
    }).collect();
    let spire = lathe(&spire_profile, 8 * detail.pick(4, 8), &|a, y| {
        // Octagonal cross-section: apothem / cos(angle from the face centre).
        let local = (a % (std::f32::consts::PI / 4.0)) - std::f32::consts::PI / 8.0;
        let facet = (std::f32::consts::PI / 8.0).cos() / local.cos();
        let crocket = ((y * 1.2).fract() - 0.5).abs() * 2.0;
        facet * (1.0 + 0.08 * crocket.powf(6.0) * ((a * 4.0).cos().abs() > 0.97) as u32 as f32)
    }, true);
    parts.push(spire);
    stone(&merge(&parts), detail.pick(0.9, 0.45), 0.008, 57)
}

fn apse(detail: Detail) -> (Mesh, Vec<Vec2>, f32) {
    // Half of a 14-gon around the apse centre; each face is a wall with a tall lancet.
    let faces = 7;
    let width = 2.0 * NAVE_HALF * (std::f32::consts::PI / 14.0).sin() + 0.25;
    let outline = [Vec2::new(-width * 0.5, 0.0), Vec2::new(width * 0.5, 0.0), Vec2::new(width * 0.5, WALL_TOP), Vec2::new(-width * 0.5, WALL_TOP)];
    let window = lancet(1.8, 6.0, 23.0, 25.5);
    let face = stone(&wall(&outline, std::slice::from_ref(&window), WALL_T), detail.pick(0.7, 0.35), 0.006, 58);
    let mut parts = Vec::new();
    let apothem = NAVE_HALF * (std::f32::consts::PI / 14.0).cos();
    for i in 0..faces {
        let a = -std::f32::consts::FRAC_PI_2 + (i as f32 + 0.5) * std::f32::consts::PI / faces as f32;
        let rot = Quat::from_rotation_y(-a + std::f32::consts::FRAC_PI_2);
        parts.push(face.transformed(Mat4::from_rotation_translation(rot, Vec3::new(a.cos() * apothem, 0.0, a.sin() * apothem))));
    }
    // Half dome (lathe, keep the eastern half).
    let profile: Vec<Vec2> = pointed_arch(2.0 * NAVE_HALF, VAULT_RISE, detail.pick(24, 48))
        .iter()
        .filter(|p| p.x <= 0.0)
        .map(|p| Vec2::new(-p.x, p.y + WALL_TOP))
        .collect();
    let dome = lathe(&profile, detail.pick(64, 128), &|_, _| 1.0, false);
    let keep: Vec<u32> = dome
        .triangles()
        .filter(|t| t.iter().map(|&i| dome.vertices[i as usize].position[0]).sum::<f32>() >= -1e-3)
        .flatten()
        .collect();
    let mut half_dome = Mesh::new(dome.vertices.clone(), keep);
    // Faces point outward; the dome is seen from inside: flip.
    for t in half_dome.indices.chunks_exact_mut(3) {
        t.swap(1, 2);
    }
    half_dome.compute_normals();
    parts.push(stone(&half_dome, detail.pick(0.8, 0.4), 0.0, 59));
    (merge(&parts), window, width)
}

fn roof(detail: Detail) -> Mesh {
    let half = BAY * 0.5;
    let ridge = WALL_TOP + VAULT_RISE + 9.0;
    let eave = WALL_TOP + 1.0;
    let slab = |z0: f32, y0: f32, z1: f32, y1: f32| {
        let d = Vec2::new(z1 - z0, y1 - y0);
        let n = Vec2::new(-d.y, d.x).normalize() * 0.35;
        let poly = [Vec2::new(z0, y0), Vec2::new(z1, y1), Vec2::new(z1, y1) + n, Vec2::new(z0, y0) + n];
        let poly: Vec<Vec2> = if (poly[1] - poly[0]).perp_dot(poly[2] - poly[1]) < 0.0 { poly.iter().rev().copied().collect() } else { poly.to_vec() };
        extrude(&poly, &[], BAY).transformed(Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2) * Mat4::from_translation(Vec3::new(0.0, 0.0, -half)))
    };
    let parts = [
        slab(NAVE_HALF + 1.6, eave, 0.0, ridge),
        slab(0.0, ridge, -NAVE_HALF - 1.6, eave),
        slab(AISLE_WALL + 1.2, AISLE_SPRING + AISLE_RISE + 0.6, NAVE_HALF + 0.8, 18.5),
        slab(-NAVE_HALF - 0.8, 18.5, -AISLE_WALL - 1.2, AISLE_SPRING + AISLE_RISE + 0.6),
    ];
    stone(&merge(&parts), detail.pick(1.0, 0.5), 0.0, 60)
}

fn pew() -> Mesh {
    let length = 4.2;
    let mut parts = vec![
        cuboid(Vec3::new(-length * 0.5, 0.42, -0.25), Vec3::new(length * 0.5, 0.48, 0.2)),
        cuboid(Vec3::new(-length * 0.5, 0.48, 0.17), Vec3::new(length * 0.5, 0.95, 0.23)),
        cuboid(Vec3::new(-length * 0.5, 0.1, -0.5), Vec3::new(length * 0.5, 0.16, -0.38)),
    ];
    let end = [Vec2::new(-0.3, 0.0), Vec2::new(0.3, 0.0), Vec2::new(0.3, 1.0), Vec2::new(0.1, 1.12), Vec2::new(-0.15, 1.05), Vec2::new(-0.3, 0.9)];
    for x in [-length * 0.5, length * 0.5 - 0.07] {
        parts.push(extrude(&end, &[], 0.07).transformed(Mat4::from_translation(Vec3::new(x, 0.0, 0.0)) * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2)));
    }
    merge(&parts)
}

fn candle(height: f32) -> (Mesh, Mesh) {
    let wax = lathe(&[Vec2::new(0.035, 0.0), Vec2::new(0.035, height), Vec2::new(0.0, height)], 16, &|_, _| 1.0, true);
    let flame = lathe(
        &[Vec2::new(0.0, height + 0.01), Vec2::new(0.012, height + 0.03), Vec2::new(0.016, height + 0.05), Vec2::new(0.008, height + 0.08), Vec2::new(0.0, height + 0.1)],
        12,
        &|_, _| 1.0,
        false,
    );
    (wax, flame)
}

/// Floor candelabrum with three candles (iron stand, wax, flames).
fn candelabrum(detail: Detail) -> (Mesh, Mesh, Mesh, [Vec3; 3]) {
    let stand = lathe(
        &[Vec2::new(0.25, 0.0), Vec2::new(0.22, 0.05), Vec2::new(0.06, 0.15), Vec2::new(0.04, 1.6), Vec2::new(0.07, 1.7), Vec2::new(0.04, 1.75), Vec2::new(0.0, 1.75)],
        detail.pick(16, 32),
        &|a, y| 1.0 + if y < 0.2 { 0.2 * (a * 3.0).cos().max(0.0) } else { 0.0 },
        true,
    );
    let arm: Vec<Vec3> = (0..=16).map(|i| {
        let t = i as f32 / 16.0 * std::f32::consts::PI;
        Vec3::new(t.cos() * 0.35, 1.75 + t.sin() * 0.12, 0.0)
    }).collect();
    let arms = tube(&arm, &[0.02; 17], 8, true);
    let mut waxes = Vec::new();
    let mut flames = Vec::new();
    let tips = [Vec3::new(-0.35, 1.75, 0.0), Vec3::new(0.0, 1.87, 0.0), Vec3::new(0.35, 1.75, 0.0)];
    for tip in tips {
        let (w, f) = candle(0.3);
        waxes.push(w.transformed(Mat4::from_translation(tip)));
        flames.push(f.transformed(Mat4::from_translation(tip)));
    }
    let lights = tips.map(|t| t + Vec3::Y * 0.36);
    (merge(&[stand, arms]), merge(&waxes), merge(&flames), lights)
}

/// Iron ring chandelier with 8 candles, hanging from a chain.
fn chandelier(detail: Detail) -> (Mesh, Mesh, Mesh) {
    let r = 1.4;
    let ring_path: Vec<Vec3> = (0..=64).map(|i| {
        let a = i as f32 / 64.0 * std::f32::consts::TAU;
        Vec3::new(a.cos() * r, 0.0, a.sin() * r)
    }).collect();
    let mut iron = vec![sweep(&circle(0.04, detail.pick(8, 12)), &ring_path, Vec3::Y, &|_| 1.0, false)];
    for i in 0..4 {
        let a = i as f32 / 4.0 * std::f32::consts::TAU;
        iron.push(tube(&[Vec3::new(a.cos() * r, 0.0, a.sin() * r), Vec3::new(0.0, 2.2, 0.0)], &[0.012, 0.012], 6, true));
    }
    iron.push(tube(&[Vec3::new(0.0, 2.2, 0.0), Vec3::new(0.0, 12.0, 0.0)], &[0.025, 0.025], 6, true));
    let mut waxes = Vec::new();
    let mut flames = Vec::new();
    for i in 0..8 {
        let a = (i as f32 + 0.5) / 8.0 * std::f32::consts::TAU;
        let p = Vec3::new(a.cos() * r, 0.03, a.sin() * r);
        let (w, f) = candle(0.25);
        waxes.push(w.transformed(Mat4::from_translation(p)));
        flames.push(f.transformed(Mat4::from_translation(p)));
    }
    (merge(&iron), merge(&waxes), merge(&flames))
}

fn altar(detail: Detail) -> (Mesh, Mesh, Mesh) {
    let stone_block = stone(
        &merge(&[
            cuboid(Vec3::new(-0.9, 0.0, -2.0), Vec3::new(0.9, 1.05, 2.0)),
            cuboid(Vec3::new(-1.1, 1.05, -2.2), Vec3::new(1.1, 1.15, 2.2)),
            cuboid(Vec3::new(-2.5, -0.01, -3.5), Vec3::new(2.5, 0.3, 3.5)),
        ]),
        detail.pick(0.4, 0.2),
        0.004,
        61,
    );
    let cloth = cuboid(Vec3::new(-0.95, 0.45, -2.05), Vec3::new(-0.88, 1.16, 2.05));
    let mut gold = vec![
        cuboid(Vec3::new(-0.05, 1.15, -0.06), Vec3::new(0.05, 2.6, 0.06)),
        cuboid(Vec3::new(-0.05, 2.1, -0.45), Vec3::new(0.05, 2.2, 0.45)),
    ];
    for z in [-1.6f32, 1.6] {
        let stick = lathe(
            &[Vec2::new(0.12, 1.15), Vec2::new(0.1, 1.2), Vec2::new(0.03, 1.3), Vec2::new(0.025, 1.7), Vec2::new(0.06, 1.75), Vec2::new(0.0, 1.76)],
            detail.pick(16, 32),
            &|_, _| 1.0,
            true,
        );
        gold.push(stick.transformed(Mat4::from_translation(Vec3::new(0.0, 0.0, z))));
    }
    (stone_block, cloth, merge(&gold))
}

// ---- Wrought iron ------------------------------------------------------------------------------------------

/// One repeat of the choir screen pattern, in metres (width, height).
const SCREEN_TILE: Vec2 = Vec2::new(0.6, 3.0);
const SCREEN_TEXELS: u32 = 512;

/// Whether the choir screen has iron at `p` (metres within one tile, y up from the floor): a
/// post at the tile edges, balusters with oval collars between rails, a frieze of rings under
/// the top rail, a diamond lattice at the bottom and spear-headed cresting.
fn screen_iron(p: Vec2) -> bool {
    let near = |d: f32, half_width: f32| d.abs() < half_width;
    let (x, y) = (p.x, p.y);
    let baluster = ((x - 0.075) / 0.15).round() * 0.15 + 0.075;
    let dx = x - baluster;
    let post = (near(x, 0.016) || near(x - SCREEN_TILE.x, 0.016)) && y < 2.9;
    let knob = p.distance(Vec2::new(0.0, 2.92)) < 0.035 || p.distance(Vec2::new(SCREEN_TILE.x, 2.92)) < 0.035;
    let rail = [0.06f32, 0.4, 2.4, 2.72].iter().any(|&h| near(y - h, 0.018)) && y > 0.0;
    let middle = (0.4..=2.4).contains(&y);
    let bar = middle && near(dx, 0.011);
    let collar = middle && near((dx / 0.032).powi(2) + ((y - 1.4) / 0.07).powi(2) - 1.0, 0.35);
    let ring = near(p.distance(Vec2::new(if x < 0.3 { 0.15 } else { 0.45 }, 2.56)) - 0.12, 0.012);
    let diagonal = |t: f32| near((t / 0.15 - (t / 0.15).round()) * 0.15, 0.008);
    let lattice = (0.06..=0.4).contains(&y) && (diagonal(x + y) || diagonal(x - y));
    let spear = (y > 2.72 && y < 2.86 && near(dx, 0.009)) || dx.abs() / 0.03 + (y - 2.9).abs() / 0.07 < 1.0;
    post || knob || rail || bar || collar || ring || lattice || spear
}

/// The choir screen pattern as an RGBA image (rows from the top): blackened iron where
/// `screen_iron` holds, with alpha = supersampled coverage for the alpha test.
fn choir_screen_image() -> Image {
    const SAMPLES: u32 = 3;
    let n = SCREEN_TEXELS;
    let mut pixels = Vec::with_capacity((n * n * 4) as usize);
    for j in 0..n {
        for i in 0..n {
            let mut covered = 0;
            for k in 0..SAMPLES * SAMPLES {
                let u = (i as f32 + ((k % SAMPLES) as f32 + 0.5) / SAMPLES as f32) / n as f32;
                let v = (j as f32 + ((k / SAMPLES) as f32 + 0.5) / SAMPLES as f32) / n as f32;
                covered += screen_iron(Vec2::new(u, 1.0 - v) * SCREEN_TILE) as u32;
            }
            // Transparent texels keep the iron colour, so filtering leaves no fringe.
            pixels.extend_from_slice(&[70, 66, 62, (covered * 255 / (SAMPLES * SAMPLES)) as u8]);
        }
    }
    Image { width: n, height: n, pixels }
}

/// A two-sided card of `width` in the XY plane, centred on x = 0, `SCREEN_TILE.y` high, whose
/// UVs repeat the screen pattern across it (v down, the image convention).
fn screen_panel(width: f32) -> Mesh {
    let (half, height) = (width * 0.5, SCREEN_TILE.y);
    let corners = [Vec2::new(-half, 0.0), Vec2::new(half, 0.0), Vec2::new(half, height), Vec2::new(-half, height)];
    let mut vertices = Vec::new();
    for normal in [Vec3::Z, -Vec3::Z] {
        vertices.extend(corners.map(|c| Vertex::new(c.extend(0.0), normal, Vec2::new((c.x + half) / SCREEN_TILE.x, 1.0 - c.y / height))));
    }
    Mesh::new(vertices, vec![0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6])
}

// ---- Assembly ---------------------------------------------------------------------------------------------

struct Builder<'a> {
    scene: &'a mut Scene,
}

impl Builder<'_> {
    fn mesh(&mut self, name: &str, mesh: &Mesh) -> MeshHandle {
        add_baked(self.scene, name, mesh)
    }

    fn place(&mut self, mesh: MeshHandle, material: MaterialHandle, m: Mat4) {
        self.scene.add_instance(mesh, material, m);
    }

    fn glass(&mut self, mesh: MeshHandle, material: MaterialHandle, m: Mat4) {
        let h = self.scene.add_instance(mesh, material, m);
        self.scene.set_cast_shadow(h, false);
    }
}

fn build_scene(detail: Detail) -> Scene {
    let mut scene = Scene::new();
    scene.sun = Sun::from_angles(SUN.0, SUN.1, SUN.2);
    scene.ground_albedo = Vec3::new(0.2, 0.2, 0.16);
    scene.clouds = helios_render::CloudLayer::cumulus(0.35);
    // Dusty air: sunlight through the clerestory and the candles become visible shafts and halos.
    scene.fog = helios_render::FogVolume {
        density: 0.012,
        height_falloff: 0.015,
        anisotropy: 0.7,
        noise: 0.3,
        albedo: Vec3::new(1.0, 0.97, 0.92),
        distance: 140.0,
        ..Default::default()
    };
    let m = materials(&mut scene);
    let mut b = Builder { scene: &mut scene };
    let half = BAY * 0.5;
    let at = |x: f32, z: f32| Mat4::from_translation(Vec3::new(x, 0.0, z));
    let flip = Mat4::from_rotation_y(std::f32::consts::PI);

    // Ground: rolling terrain (heightfield SDF) plus the paved precinct.
    let terrain = {
        let size = 1800.0;
        let res = detail.pick(384, 768);
        let mut t = primitives::terrain(size, res, 1.0, 1);
        t.deform(|p| Vec3::new(p.x + LENGTH * 0.5, terrain_height(p.x + LENGTH * 0.5, p.z), p.z));
        t
    };
    let terrain = add_baked_with(b.scene, "terrain", &terrain, BuildConfig { sdf: SdfMode::Heightfield { resolution: 512 }, ..Default::default() });
    b.place(terrain, m.terrain, Mat4::IDENTITY);
    // Large flat slabs: a heightfield field keeps them exact (a volume's coarse voxels would
    // either lose the thin slab or inflate it by half a voxel, swallowing the pews).
    let heightfield = |resolution| BuildConfig { sdf: SdfMode::Heightfield { resolution }, ..Default::default() };
    let plaza = add_baked_with(b.scene, "plaza", &stone(&cuboid(Vec3::new(-40.0, -0.6, -40.0), Vec3::new(LENGTH + 30.0, -0.05, 40.0)), detail.pick(4.0, 2.0), 0.0, 62), heightfield(256));
    b.place(plaza, m.plaza, Mat4::IDENTITY);
    let floor = add_baked_with(b.scene, "floor", &cuboid(Vec3::new(-1.0, -0.4, -AISLE_WALL), Vec3::new(LENGTH + NAVE_HALF, 0.0, AISLE_WALL)), heightfield(256));
    b.place(floor, m.floor, Mat4::IDENTITY);

    // Piers, arcades and nave walls with clerestory glass and tracery.
    let pier = b.mesh("pier", &pier(detail));
    let arcade = b.mesh("arcade moulding", &arcade_moulding(detail));
    let (wall_mesh, clerestory, triforium_holes) = nave_wall(detail);
    let nave_wall = b.mesh("nave wall bay", &wall_mesh);
    let clerestory_glass = b.mesh("clerestory glass", &glass_pane(&clerestory));
    let clerestory_tracery = b.mesh("clerestory tracery", &stone(&tracery(4.0, 18.6, 24.6, WALL_TOP - 0.7, detail), detail.pick(0.5, 0.25), 0.003, 63));
    let triforium_back = b.mesh("triforium back", &cuboid(Vec3::new(-half, 13.6, 1.0), Vec3::new(half, 18.0, 1.6)));
    let _ = triforium_holes;
    for i in 0..=BAYS {
        let x = i as f32 * BAY;
        for (z, rot) in [(NAVE_HALF, Mat4::IDENTITY), (-NAVE_HALF, flip)] {
            // Piers face the nave with their vaulting shaft (+z local -> towards the axis).
            let face_in = if z > 0.0 { flip } else { Mat4::IDENTITY };
            b.place(pier, m.smooth, at(x, z) * face_in);
            if i < BAYS {
                let cx = x + half;
                b.place(nave_wall, m.ashlar, at(cx, z) * rot);
                b.place(arcade, m.smooth, at(cx, z) * rot);
                b.glass(clerestory_glass, m.glass[i % 2], at(cx, z) * rot);
                b.place(clerestory_tracery, m.smooth, at(cx, z) * rot);
                b.place(triforium_back, m.ashlar, at(cx, z) * rot);
            }
        }
    }

    // Vaults.
    let nave_vault = b.mesh("nave vault bay", &vault_bay(2.0 * NAVE_HALF, VAULT_RISE, WALL_TOP, detail));
    let aisle_vault = b.mesh("aisle vault bay", &vault_bay(AISLE_WALL - NAVE_HALF, AISLE_RISE, AISLE_SPRING, detail));
    let aisle_center = (AISLE_WALL + NAVE_HALF) * 0.5;
    for i in 0..BAYS {
        let cx = i as f32 * BAY + half;
        b.place(nave_vault, m.vault, at(cx, 0.0));
        b.place(aisle_vault, m.vault, at(cx, aisle_center));
        b.place(aisle_vault, m.vault, at(cx, -aisle_center));
    }

    // Aisle walls, windows, buttresses and roofs.
    let (aisle_mesh, aisle_window) = aisle_wall(detail);
    let aisle_wall = b.mesh("aisle wall bay", &aisle_mesh);
    let aisle_glass = b.mesh("aisle glass", &glass_pane(&aisle_window));
    let aisle_tracery = b.mesh("aisle tracery", &stone(&tracery(3.2, 3.2, 9.2, 12.0, detail), detail.pick(0.5, 0.25), 0.003, 64));
    let buttress = b.mesh("buttress", &buttress(detail));
    let roof = b.mesh("roof bay", &roof(detail));
    for i in 0..BAYS {
        let cx = i as f32 * BAY + half;
        for (z, rot) in [(AISLE_WALL, Mat4::IDENTITY), (-AISLE_WALL, flip)] {
            b.place(aisle_wall, m.ashlar, at(cx, z) * rot);
            b.glass(aisle_glass, m.glass[(i + 1) % 2], at(cx, z) * rot);
            b.place(aisle_tracery, m.smooth, at(cx, z) * rot);
        }
        b.place(roof, m.slate, at(cx, 0.0));
    }
    for i in 0..=BAYS {
        let x = i as f32 * BAY;
        b.place(buttress, m.ashlar, at(x, AISLE_WALL + WALL_T * 0.5));
        b.place(buttress, m.ashlar, at(x, -AISLE_WALL - WALL_T * 0.5) * flip);
    }

    // West front with portal and rose window; twin towers.
    let (front_mesh, rose, side_window) = west_front(detail);
    let front = b.mesh("west front", &front_mesh);
    let rose_glass = b.mesh("rose glass", &glass_pane(&rose));
    let rose_tracery = b.mesh("rose tracery", &stone(&rose_tracery(detail).transformed(Mat4::from_translation(Vec3::new(0.0, 22.5, 0.0))), detail.pick(0.5, 0.25), 0.003, 65));
    let west_glass = b.mesh("west glass", &glass_pane(&side_window));
    let face_west = Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_2);
    b.place(front, m.ashlar, at(-0.6, 0.0) * face_west);
    b.glass(rose_glass, m.glass[2], at(-0.6, 0.0) * face_west);
    b.place(rose_tracery, m.smooth, at(-0.6, 0.0) * face_west);
    b.glass(west_glass, m.glass[1], at(-0.6, 0.0) * face_west);
    b.glass(west_glass, m.glass[0], at(-0.6, 0.0) * face_west * Mat4::from_translation(Vec3::new(21.0, 0.0, 0.0)));
    let tower = b.mesh("tower", &tower(detail));
    b.place(tower, m.ashlar, at(-5.4, AISLE_WALL - 2.3));
    b.place(tower, m.ashlar, at(-5.4, -AISLE_WALL + 2.3));

    // Apse with tall windows and a half dome.
    let (apse_mesh, apse_window, face_width) = apse(detail);
    let apse = b.mesh("apse", &apse_mesh);
    b.place(apse, m.ashlar, at(LENGTH, 0.0));
    let apse_glass = b.mesh("apse glass", &glass_pane(&apse_window));
    let apothem = NAVE_HALF * (std::f32::consts::PI / 14.0).cos();
    for k in 0..7 {
        let a = -std::f32::consts::FRAC_PI_2 + (k as f32 + 0.5) * std::f32::consts::PI / 7.0;
        let rot = Quat::from_rotation_y(-a + std::f32::consts::FRAC_PI_2);
        let m4 = Mat4::from_rotation_translation(rot, Vec3::new(LENGTH + a.cos() * apothem, 0.0, a.sin() * apothem));
        b.glass(apse_glass, m.glass[k % 3], m4);
    }
    let _ = face_width;
    // Aisle east walls (closing the aisles beside the apse).
    let east = b.mesh("aisle east wall", &stone(&cuboid(Vec3::new(-0.6, 0.0, NAVE_HALF + 0.7), Vec3::new(0.6, AISLE_SPRING + AISLE_RISE + 0.5, AISLE_WALL + 0.6)), detail.pick(0.7, 0.35), 0.006, 66));
    b.place(east, m.ashlar, at(LENGTH, 0.0));
    b.place(east, m.ashlar, at(LENGTH, 0.0) * Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0)));

    // Furnishings: pews, altar, candelabra, chandeliers — with candle lights.
    let pew = b.mesh("pew", &pew());
    for row in 0..30 {
        let x = 9.0 + row as f32 * 1.15;
        b.place(pew, m.wood, at(x, -3.4) * Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_2));
        b.place(pew, m.wood, at(x, 3.4) * Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_2));
    }
    let (altar_stone, altar_cloth, altar_gold) = altar(detail);
    let altar_x = LENGTH - 6.0;
    let altar_mesh = b.mesh("altar", &altar_stone);
    b.place(altar_mesh, m.smooth, at(altar_x, 0.0));
    let cloth = b.mesh("altar cloth", &altar_cloth);
    b.place(cloth, m.cloth, at(altar_x, 0.0));
    let gold = b.mesh("altar gold", &altar_gold);
    b.place(gold, m.gold, at(altar_x, 0.0));
    let (stand, waxes, flames, flame_offsets) = candelabrum(detail);
    let stand = b.mesh("candelabrum", &stand);
    let wax = b.mesh("candles", &waxes);
    let flame = b.mesh("flames", &flames);
    let warm = Vec3::new(1.0, 0.62, 0.3);
    let mut lights = Vec::new();
    for i in 1..BAYS {
        let x = i as f32 * BAY;
        for z in [NAVE_HALF - 1.9, -NAVE_HALF + 1.9] {
            let t = at(x, z) * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
            b.place(stand, m.iron, t);
            b.place(wax, m.wax, t);
            let h = b.scene.add_instance(flame, m.flame, t);
            b.scene.set_cast_shadow(h, false);
            lights.push(t.transform_point3(flame_offsets[1]));
        }
    }
    for z in [-1.6f32, 1.6] {
        let (w, f) = candle(0.4);
        let wm = b.mesh("altar candle", &w);
        let fm = b.mesh("altar flame", &f);
        let t = at(altar_x, z) * Mat4::from_translation(Vec3::Y * 1.76);
        b.place(wm, m.wax, t);
        let h = b.scene.add_instance(fm, m.flame, t);
        b.scene.set_cast_shadow(h, false);
        lights.push(t.transform_point3(Vec3::Y * 0.46));
    }
    let (ring, ring_wax, ring_flames) = chandelier(detail);
    let ring = b.mesh("chandelier", &ring);
    let ring_wax = b.mesh("chandelier candles", &ring_wax);
    let ring_flames = b.mesh("chandelier flames", &ring_flames);
    for i in 0..BAYS {
        let t = at(i as f32 * BAY + half, 0.0) * Mat4::from_translation(Vec3::Y * 7.5);
        b.place(ring, m.iron, t);
        b.place(ring_wax, m.wax, t);
        let h = b.scene.add_instance(ring_flames, m.flame, t);
        b.scene.set_cast_shadow(h, false);
        b.scene.add_light(Light { source_radius: 1.4, ..Light::point(t.transform_point3(Vec3::Y * 0.35), warm, 3.0, 16.0) });
    }
    for p in lights {
        b.scene.add_light(Light { source_radius: 0.15, ..Light::point(p, warm, 0.6, 9.0) });
    }

    // Wrought-iron screens in the choir arcades: alpha-tested cards that cast lattice shadows.
    // (Built last, so pre-baked mesh libraries of the rest of the scene stay valid.)
    let screen_texture = b.scene.add_image_texture(ImageTexture { base_color: Some(Arc::new(choir_screen_image())), ..Default::default() });
    let screen_material = b.scene.add_material(Material {
        base_color: Vec3::ONE,
        roughness: 0.55,
        metallic: 1.0,
        texture: Some(screen_texture),
        mapping: TextureMapping::MeshUv,
        alpha_cutoff: Some(0.5),
        ..Default::default()
    });
    let screen = b.mesh("choir screen", &screen_panel(8.0 * SCREEN_TILE.x));
    for i in BAYS - 3..BAYS {
        for z in [NAVE_HALF, -NAVE_HALF] {
            b.place(screen, screen_material, at(i as f32 * BAY + half, z));
        }
    }

    // Free-standing glass screens at the west end of the north aisle, behind the initial view: a
    // stained-glass one across the aisle and two tinted clear panes crossing each other in front
    // of it, all seen against the west window — up to four glass layers deep, which only
    // per-pixel ordering composites correctly (see translucency.rs). No distance fields: Lumen's
    // lighting stays as it was.
    let no_sdf = BuildConfig { sdf: SdfMode::None, ..Default::default() };
    let large = add_baked_with(b.scene, "glass screen", &glass_pane(&lancet(3.6, 0.3, 3.2, 4.8)), no_sdf);
    let small = add_baked_with(b.scene, "small glass screen", &glass_pane(&lancet(2.2, 0.3, 2.2, 3.2)), no_sdf);
    let across = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let aisle = (AISLE_WALL + NAVE_HALF) * 0.5;
    b.glass(large, m.glass[2], at(2.5, aisle) * across);
    for (angle, tint) in [(0.6, Vec3::new(1.0, 0.72, 0.3)), (-0.6, Vec3::new(0.4, 0.62, 1.0))] {
        let clear = b.scene.add_material(Material { kind: MaterialKind::Glass, base_color: tint, roughness: 0.05, ..Default::default() });
        b.glass(small, clear, at(6.0, aisle + 0.9) * across * Mat4::from_rotation_y(angle));
    }
    scene
}

fn initial_camera() -> Camera {
    let mut camera = Camera { position: Vec3::new(5.0, 1.7, 0.0), fov_y: 62f32.to_radians(), near: 0.05, ..Default::default() };
    camera.look_at(Vec3::new(LENGTH, 9.0, 0.0));
    camera
}

pub fn build(detail: Detail) -> SceneSetup {
    SceneSetup { scene: build_scene(detail), camera: initial_camera(), sun: SUN, floor: floor_height, animation: None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choir_screen_is_mostly_open_and_tiles() {
        let image = choir_screen_image();
        let alpha = |i: u32, j: u32| image.pixels[((j * image.width + i) * 4 + 3) as usize];
        let n = image.width;
        let iron = (0..n * n).filter(|&k| alpha(k % n, k / n) >= 128).count() as f32 / (n * n) as f32;
        assert!((0.1..0.4).contains(&iron), "iron covers {iron}");
        // The left and right columns continue each other (the post straddles the tile edge).
        let matching = (0..n).filter(|&j| (alpha(0, j) >= 128) == (alpha(n - 1, j) >= 128)).count();
        assert!(matching as f32 > 0.95 * n as f32);
    }
}
