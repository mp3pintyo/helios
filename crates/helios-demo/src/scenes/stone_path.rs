//! Nanite tessellation in the valley: a cobblestone path and displaced boulders beside it, behind
//! the initial view (so the valley's default view renders as before). Their materials displace the
//! surface at render time ([`Displacement`]): the path's mesh is a smooth strip whose cobbles come
//! from the cobblestone texture's height, the boulders' crags from the rock texture's. One boulder
//! turns slowly on its plinth (a movable displaced instance). `PageDown` toggles tessellation.

use glam::{Mat4, Quat, Vec2, Vec3};
use helios_geometry::{primitives, BuildConfig, Mesh, SdfMode, Vertex};
use helios_render::{Displacement, Material, MaterialKind, Scene, TextureDesc, TextureMapping, TexturePattern};

use super::{add_baked, add_baked_with, Detail};
use crate::motion::{DemoMotion, Motion};

/// The path's centre line (a quadratic Bézier in x, z), its width and how far its surface lies
/// above the terrain.
const PATH: [Vec2; 3] = [Vec2::new(-27.0, 41.0), Vec2::new(-33.0, 55.0), Vec2::new(-46.0, 63.0)];
const PATH_WIDTH: f32 = 3.2;
const PATH_LIFT: f32 = 0.06;
/// Cobblestone texture repeats per metre (8 setts per repeat: ~19 cm setts).
const COBBLE_SCALE: f32 = 0.65;

fn bezier(t: f32) -> Vec2 {
    let u = 1.0 - t;
    PATH[0] * (u * u) + PATH[1] * (2.0 * u * t) + PATH[2] * (t * t)
}

/// A strip following the ground along the path: UVs in metres (u across, v along), skirts at
/// both sides reaching into the ground so the displaced edge never floats.
fn path_mesh(ground: fn(f32, f32) -> f32, detail: Detail) -> Mesh {
    let (along, across) = detail.pick((96, 12), (160, 16));
    let length: f32 = (0..64).map(|i| bezier(i as f32 / 64.0).distance(bezier((i + 1) as f32 / 64.0))).sum();
    let mut vertices = Vec::new();
    // Columns: skirt, the surface's across + 1 columns, skirt.
    let columns = across + 3;
    for i in 0..=along {
        let t = i as f32 / along as f32;
        let center = bezier(t);
        let tangent = (bezier((t + 0.01).min(1.0)) - bezier((t - 0.01).max(0.0))).normalize();
        let side = Vec2::new(-tangent.y, tangent.x);
        for c in 0..columns {
            let k = c.clamp(1, across + 1) - 1;
            let offset = (k as f32 / across as f32 - 0.5) * PATH_WIDTH;
            let p = center + side * offset;
            let skirt = c == 0 || c == columns - 1;
            let y = ground(p.x, p.y) + if skirt { -0.2 } else { PATH_LIFT };
            vertices.push(Vertex::new(Vec3::new(p.x, y, p.y), Vec3::Y, Vec2::new(offset + PATH_WIDTH * 0.5, t * length)));
        }
    }
    let mut indices = Vec::new();
    for i in 0..along {
        for c in 0..columns - 1 {
            let a = i * columns + c;
            let b = a + columns;
            indices.extend([a, a + 1, b + 1, a, b + 1, b]);
        }
    }
    let mut mesh = Mesh::new(vertices, indices);
    mesh.compute_normals();
    // Wind the strip upwards whichever way the centre line turns.
    if mesh.vertices[1].normal[1] < 0.0 {
        for t in mesh.indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
        mesh.compute_normals();
    }
    mesh
}

pub fn add(scene: &mut Scene, motion: &mut DemoMotion, detail: Detail, ground: fn(f32, f32) -> f32) {
    let cobble = scene.add_texture(
        TextureDesc::new(TexturePattern::Cobblestone, Vec3::new(0.42, 0.4, 0.37), Vec3::new(0.3, 0.27, 0.24)).roughness(0.7, 0.85).param(0.4).seed(41),
    );
    let crags = scene.add_texture(TextureDesc::new(TexturePattern::Rock, Vec3::new(0.34, 0.31, 0.27), Vec3::new(0.16, 0.15, 0.13)).roughness(0.75, 0.9).bump(1.2).seed(42));

    // The path: smooth geometry, cobbles from the texture's height (stones rise ~5 cm out of
    // joints ~3 cm deep). The normal map keeps the finer relief the dice rate leaves out.
    let mesh = path_mesh(ground, detail);
    let path = scene.add_material(Material {
        roughness: 0.75,
        normal_strength: 0.5,
        displacement: Some(Displacement { uv_density: Displacement::uv_density_of(&mesh), ..Displacement::height(0.085, 0.35) }),
        ..Material::textured(MaterialKind::Standard, cobble, TextureMapping::MeshUv, COBBLE_SCALE)
    });
    scene.set_material_name(path, "cobblestone path");
    let path_mesh = add_baked_with(scene, "cobblestone path", &mesh, BuildConfig { sdf: SdfMode::None, ..Default::default() });
    scene.add_instance(path_mesh, path, Mat4::IDENTITY);

    // Boulders whose crags are displaced from the rock texture (object triplanar), plus one
    // turning slowly on its plinth.
    let rock = scene.add_material(Material {
        kind: MaterialKind::Rock,
        base_color: Vec3::new(0.34, 0.31, 0.27),
        roughness: 0.8,
        detail_scale: 7.0,
        detail_strength: 0.2,
        displacement: Some(Displacement::height(0.14, 0.5)),
        ..Material::textured(MaterialKind::Rock, crags, TextureMapping::ObjectTriplanar, 0.5)
    });
    scene.set_material_name(rock, "displaced rock");
    let boulders: Vec<_> = (0..2).map(|seed| add_baked(scene, "displaced boulder", &primitives::rock(seed * 5 + 31, detail.pick(4, 5)))).collect();
    let mut place = |at: Vec2, size: Vec3, turn: f32, mesh| {
        let position = Vec3::new(at.x, ground(at.x, at.y) + size.y * 0.25, at.y);
        scene.add_instance(mesh, rock, Mat4::from_scale_rotation_translation(size, Quat::from_rotation_y(turn), position))
    };
    for (i, (t, offset, size)) in [(0.18, 3.6, Vec3::new(2.4, 1.6, 2.0)), (0.47, -3.4, Vec3::new(1.6, 1.2, 1.9)), (0.8, 3.8, Vec3::new(3.0, 2.0, 2.4))].into_iter().enumerate() {
        let tangent = (bezier(t + 0.01) - bezier(t - 0.01)).normalize();
        place(bezier(t) + Vec2::new(-tangent.y, tangent.x) * offset, size, i as f32 * 1.7, boulders[i % 2]);
    }
    let at = bezier(0.62) + Vec2::new(-4.2, -1.0);
    let turning = place(at, Vec3::new(1.3, 1.0, 1.3), 0.0, boulders[1]);
    let center = Vec3::new(at.x, ground(at.x, at.y), at.y);
    motion.add(scene, Motion::Spin { center, period: 60.0 }, &[turning], &[]);
}
