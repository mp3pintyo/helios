//! The candle-lit cathedral (`--lights N`, `?lights=N`): MegaLights' showcase.
//!
//! By default the cathedral has 33 lights, one per candelabrum (three candles), chandelier (eight)
//! and altar candle, which the classic path can almost all give shadow maps. With a light count
//! set, every candle flame becomes a light of its own — 150 in the furnishings — and votive racks
//! (three tiers of tea lights against the aisle walls of every bay, 858 candles in all) hold the
//! rest, up to 1008 shadow-casting lights. The first `N` flames are lit: the furnishings first,
//! then the racks' candles spread over all racks. The racks' meshes are added after every other
//! mesh, so mesh libraries baked from the default cathedral stay valid.

use std::sync::atomic::{AtomicUsize, Ordering};

use glam::{Mat4, Vec3};
use helios_geometry::modeling::{cuboid, lathe, merge};
use helios_geometry::Mesh;
use helios_render::{Light, LightHandle, MaterialHandle, Scene};

use super::add_baked;

/// Requested light count (0: the default 33 lights).
static CANDLE_LIGHTS: AtomicUsize = AtomicUsize::new(0);

/// Lights the cathedral built from now on with `count` individual candles (0: the default 33
/// lights, one per fixture).
pub fn set_candle_lights(count: usize) {
    CANDLE_LIGHTS.store(count, Ordering::Relaxed);
}

/// The requested light count, if the candle-lit variant is on.
pub(crate) fn candle_lights() -> Option<usize> {
    Some(CANDLE_LIGHTS.load(Ordering::Relaxed)).filter(|&n| n > 0)
}

/// What a flame belongs to (its light's brightness and reach).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fixture {
    Candelabrum,
    Chandelier,
    Altar,
    Votive,
}

/// Racks: tiers, candles per tier and their spacing (metres).
const TIERS: usize = 3;
const PER_TIER: usize = 13;
const SPACING: f32 = 0.12;
/// Height of a tea light's wax.
const TEA_LIGHT: f32 = 0.06;

fn tier_origin(tier: usize) -> Vec3 {
    Vec3::new(0.0, 0.75 + 0.15 * tier as f32, 0.17 * tier as f32)
}

/// A tea light: low wax cylinder and a small flame (`candle` in cathedral.rs, shorter).
fn tea_light() -> (Mesh, Mesh) {
    let wax = lathe(&[glam::Vec2::new(0.03, 0.0), glam::Vec2::new(0.03, TEA_LIGHT), glam::Vec2::new(0.0, TEA_LIGHT)], 12, &|_, _| 1.0, true);
    let flame = lathe(
        &[
            glam::Vec2::new(0.0, TEA_LIGHT + 0.005),
            glam::Vec2::new(0.009, TEA_LIGHT + 0.02),
            glam::Vec2::new(0.012, TEA_LIGHT + 0.035),
            glam::Vec2::new(0.006, TEA_LIGHT + 0.055),
            glam::Vec2::new(0.0, TEA_LIGHT + 0.07),
        ],
        10,
        &|_, _| 1.0,
        false,
    );
    (wax, flame)
}

/// A votive rack in its local frame (candle rows along x, the wall behind it at +z): an iron
/// stand with three stepped tiers. Returns the stand, the wax, the flames and the flame centres.
fn votive_rack() -> (Mesh, Mesh, Mesh, Vec<Vec3>) {
    let half = 0.5 * SPACING * PER_TIER as f32;
    let mut iron = Vec::new();
    for tier in 0..TIERS {
        let o = tier_origin(tier);
        iron.push(cuboid(Vec3::new(-half, o.y - 0.03, o.z - 0.08), Vec3::new(half, o.y, o.z + 0.08)));
    }
    let back = tier_origin(TIERS - 1).z + 0.08;
    for x in [-half, half - 0.04] {
        iron.push(cuboid(Vec3::new(x, 0.0, -0.08), Vec3::new(x + 0.04, tier_origin(0).y, -0.04)));
        iron.push(cuboid(Vec3::new(x, 0.0, back - 0.04), Vec3::new(x + 0.04, tier_origin(TIERS - 1).y, back)));
    }
    let (wax, flame) = tea_light();
    let mut waxes = Vec::new();
    let mut flames = Vec::new();
    let mut centres = Vec::new();
    for tier in 0..TIERS {
        for i in 0..PER_TIER {
            let p = tier_origin(tier) + Vec3::X * ((i as f32 + 0.5) * SPACING - half);
            waxes.push(wax.transformed(Mat4::from_translation(p)));
            flames.push(flame.transformed(Mat4::from_translation(p)));
            centres.push(p + Vec3::Y * (TEA_LIGHT + 0.03));
        }
    }
    (merge(&iron), merge(&waxes), merge(&flames), centres)
}

/// Places a votive rack at each of `placements` and returns their flames, ordered so that any
/// prefix spreads over all racks (candle `k` of every rack before candle `k + 1`).
pub(crate) fn place_votive_racks(scene: &mut Scene, placements: &[Mat4], iron: MaterialHandle, wax: MaterialHandle, flame: MaterialHandle) -> Vec<(Vec3, Fixture)> {
    let (stand, waxes, flames, centres) = votive_rack();
    let stand = add_baked(scene, "votive rack", &stand);
    let waxes = add_baked(scene, "votive candles", &waxes);
    let flames = add_baked(scene, "votive flames", &flames);
    for &t in placements {
        scene.add_instance(stand, iron, t);
        scene.add_instance(waxes, wax, t);
        let h = scene.add_instance(flames, flame, t);
        scene.set_cast_shadow(h, false);
    }
    centres.iter().flat_map(|&c| placements.iter().map(move |t| (t.transform_point3(c), Fixture::Votive))).collect()
}

/// Adds a light for each of the first `count` flames; returns them in the flames' order.
pub(crate) fn light_flames(scene: &mut Scene, flames: &[(Vec3, Fixture)], count: usize) -> Vec<LightHandle> {
    let warm = Vec3::new(1.0, 0.62, 0.3);
    let mut lights = Vec::with_capacity(count.min(flames.len()));
    for (i, &(position, fixture)) in flames.iter().take(count).enumerate() {
        // Each flame burns a little differently.
        let tint = 0.9 + 0.2 * super::Rng(0x9e37_79b9 ^ (i as u32).wrapping_mul(0x85eb_ca6b)).next();
        let (intensity, range, radius) = match fixture {
            Fixture::Candelabrum | Fixture::Altar => (0.25, 8.0, 0.02),
            Fixture::Chandelier => (0.4, 12.0, 0.02),
            Fixture::Votive => (0.06, 4.0, 0.012),
        };
        lights.push(scene.add_light(Light { source_radius: radius, ..Light::point(position, warm * tint, intensity, range) }));
    }
    lights
}
