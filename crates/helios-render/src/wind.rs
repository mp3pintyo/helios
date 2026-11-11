//! Wind animation of vegetation: UE5's foliage World Position Offset, as Nanite supports it (the
//! displacement itself is `shaders/wind.wgsl`).
//!
//! [`Wind`] is the scene's air flow; a material's [`WindResponse`] says how its surfaces follow it
//! (zero for rocks and buildings). Every vertex moves by a pure function of its rest position, its
//! instance's transform and the time, so every geometry consumer — hardware and software
//! visibility rasterisers, hardware and software shadow page rasterisers, the G-buffer resolve
//! that re-fetches triangles from the visibility buffer — computes the same position, and two
//! clusters or LOD levels sharing a vertex move it identically: no cracks. Motion vectors use the
//! previous frame's offset, so TAA and motion blur follow the motion.
//!
//! Animated geometry must never be culled while visible, so instance, page and cluster bounding
//! spheres are padded by an upper bound of the displacement (frustum, HZB occlusion and shadow
//! page culling alike). Cached shadow pages assume static geometry: the pages under moving
//! instances re-render these instances over a cached layer of their static casters as they move
//! (`shaders/vsm_invalidate.wgsl`, `vsm.rs`), the others stay cached.
//! Beyond [`Wind::disable_distance`] nothing moves (UE's "WPO disable distance"), which bounds that
//! cost; instances fade out of the wind before it, so they don't pop. Lumen's distance fields and
//! surface cache stay static: the motion is far below their voxel size.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

/// An instance stops moving where the bound of its displacement would cover fewer pixels than
/// this (its typical motion is a fraction of the bound): distant grass is still, trees still move.
pub const WIND_MIN_PIXELS: f32 = 4.0;

/// Binding of the [`WindUniform`] in every shader that positions or culls geometry.
pub const BINDING_WIND: u32 = 65;

/// Layout entry of the wind uniform.
pub fn wind_entry() -> (u32, wgpu::BindingType) {
    (BINDING_WIND, crate::util::bind::uniform())
}

/// The scene's wind uniform for [`wind_entry`].
pub fn wind_resource(scene: &crate::gpu_scene::GpuScene) -> (u32, wgpu::BindingResource<'_>) {
    (BINDING_WIND, scene.wind.as_entire_binding())
}

/// The scene's wind ([`crate::Scene::wind`]). The default is calm: nothing moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wind {
    /// Direction the wind blows towards (x, z); normalised on upload.
    pub direction: Vec2,
    /// 0 = calm, 1 = a fresh breeze (the [`WindResponse`] presets are tuned for it). The
    /// displacement scales linearly.
    pub strength: f32,
    /// Gusts per second passing a point.
    pub gust_frequency: f32,
    /// Distance between gust fronts in metres: gusts roll through the vegetation along the wind.
    pub gust_wavelength: f32,
    /// 0..1: how much plants sway around their lean (0 = they only lean with the gusts).
    pub turbulence: f32,
    /// Distance from the camera beyond which instances don't move (UE's WPO disable distance):
    /// their shadow pages stay cached. Small plants stop closer, where their motion would cover
    /// less than [`WIND_MIN_PIXELS`]; instances fade out over the last quarter of their distance.
    pub disable_distance: f32,
}

impl Default for Wind {
    fn default() -> Self {
        Self { strength: 0.0, ..Self::breeze(0.0) }
    }
}

impl Wind {
    /// A gusty breeze of the given strength blowing towards +x, slightly towards +z.
    pub fn breeze(strength: f32) -> Self {
        Self {
            direction: Vec2::new(1.0, 0.35),
            strength,
            gust_frequency: 0.12,
            gust_wavelength: 35.0,
            turbulence: 0.5,
            disable_distance: 80.0,
        }
    }
}

/// How a material's surfaces follow the wind ([`crate::Material::wind`]); all zero = static.
///
/// The displacement layers UE's hierarchical foliage wind: the whole plant *bends* about its
/// instance origin (growing with the square of the height above it, like a cantilever), branches
/// *sway* (growing with the horizontal distance from the plant's axis) and leaves *flutter*. A
/// plant made of several instances — wood and foliage sharing one transform — moves as one when
/// their materials share `bend`, `sway` and `frequency`; `flutter` may differ (bark doesn't
/// flutter). Translucent materials ignore it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindResponse {
    /// Horizontal displacement per unit strength of a point 1 m above the origin, in metres
    /// (× height²).
    pub bend: f32,
    /// Displacement per unit strength and metre of horizontal distance from the plant's axis.
    pub sway: f32,
    /// Leaf flutter amplitude per unit strength, in metres.
    pub flutter: f32,
    /// Natural sway frequency of the plant in Hz; branches sway 2.3× faster.
    pub frequency: f32,
}

impl WindResponse {
    pub const NONE: Self = Self { bend: 0.0, sway: 0.0, flutter: 0.0, frequency: 0.0 };
    /// Trunks and branches of trees and bushes: an 18 m crown leans ~0.3 m in a gust.
    pub const WOOD: Self = Self { bend: 0.0009, sway: 0.012, flutter: 0.0, frequency: 0.3 };
    /// Leaves and needles on [`WindResponse::WOOD`]: the same bend and sway, plus flutter.
    pub const LEAVES: Self = Self { flutter: 0.01, ..Self::WOOD };
    /// Grass blades: flexible, quick, a little flutter.
    pub const GRASS: Self = Self { bend: 0.22, sway: 0.0, flutter: 0.003, frequency: 0.9 };
    /// Fern fronds: bend with the plant, the frond tips bob.
    pub const FERN: Self = Self { bend: 0.1, sway: 0.02, flutter: 0.004, frequency: 0.6 };

    pub fn is_static(&self) -> bool {
        self.bend == 0.0 && self.sway == 0.0 && self.flutter == 0.0
    }

    /// GPU layout ([`crate::gpu_types::GpuInstance::wind`]).
    pub(crate) fn to_vec4(self) -> Vec4 {
        Vec4::new(self.bend, self.sway, self.flutter, self.frequency)
    }
}

/// Mirrors `struct Wind` in wind.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WindUniform {
    /// xz: unit direction the wind blows towards, w: strength (0: calm or disabled).
    pub direction: Vec4,
    /// x: time, y: previous frame's time (seconds), z: gust frequency, w: turbulence.
    pub time: Vec4,
    /// xyz: camera position, w: disable distance.
    pub camera: Vec4,
    /// x: distance per metre of displacement where the motion becomes too small to see,
    /// y: gust wavelength.
    pub params: Vec4,
}

const _: () = assert!(std::mem::size_of::<WindUniform>() == 64);

impl WindUniform {
    /// The wind at `time` (and `previous_time`, for motion vectors) seen from `camera`, whose
    /// pixels subtend `pixel_angle` radians, with the scene's strength scaled by `strength` (0
    /// disables it).
    pub fn new(wind: &Wind, strength: f32, time: f32, previous_time: f32, camera: Vec3, pixel_angle: f32) -> Self {
        let direction = wind.direction.try_normalize().unwrap_or(Vec2::X);
        let distance = wind.disable_distance.max(0.0);
        Self {
            direction: Vec4::new(direction.x, 0.0, direction.y, (wind.strength * strength).max(0.0)),
            time: Vec4::new(time, previous_time, wind.gust_frequency.max(0.0), wind.turbulence.clamp(0.0, 1.0)),
            camera: camera.extend(distance),
            params: Vec4::new(1.0 / (pixel_angle.max(1e-6) * WIND_MIN_PIXELS), wind.gust_wavelength.max(1.0), 0.0, 0.0),
        }
    }

    pub fn is_active(&self) -> bool {
        self.direction.w > 0.0 && self.camera.w > 0.0
    }
}

/// CPU mirror of wind.wgsl, so tests can check the culling bound against the displacement.
#[cfg(test)]
pub(crate) mod mirror {
    use glam::{Mat4, Vec3, Vec4};

    use super::WindUniform;

    const TAU: f32 = std::f32::consts::TAU;
    const FLUTTER_HZ: f32 = 2.6;
    const BEND_BOUND: f32 = 1.05;
    const FLUTTER_BOUND: f32 = 1.7321;

    fn gust(w: &WindUniform, origin: Vec3, time: f32) -> f32 {
        let travel = (origin.x * w.direction.x + origin.z * w.direction.z) / w.params.y;
        let p = TAU * (w.time.z * time - travel);
        (0.5 + 0.3 * p.sin() + 0.2 * (2.37 * p + 1.9).sin()).clamp(0.0, 1.0)
    }

    pub fn strength(w: &WindUniform, world: &Mat4, response: Vec4, bounds: Vec4) -> f32 {
        let s = w.direction.w;
        if s <= 0.0 || response.truncate() == Vec3::ZERO {
            return 0.0;
        }
        let limit = w.camera.w.min(reach(world, response, s, bounds) * w.params.x);
        let d = world.w_axis.truncate().distance(w.camera.truncate());
        let fade = (4.0 * (limit - d) / limit.max(1e-3)).clamp(0.0, 1.0);
        s * fade * fade * (3.0 - 2.0 * fade)
    }

    /// `wind_offset` with a fixed plant phase (the GPU hashes the origin's bits).
    pub fn offset(w: &WindUniform, world: &Mat4, response: Vec4, strength: f32, rest: Vec3, time: f32, phase: f32) -> Vec3 {
        let origin = world.w_axis.truncate();
        let rel = rest - origin;
        let height = rel.y.max(0.0);
        let reach = Vec3::new(rel.x, 0.0, rel.z).length();
        let dir = Vec3::new(w.direction.x, 0.0, w.direction.z);
        let side = Vec3::new(-dir.z, 0.0, dir.x);
        let g = gust(w, origin, time);
        let turbulence = w.time.w;
        let plant = TAU * response.w * time + phase;
        let lean = g * (1.0 - 0.5 * turbulence) + 0.5 * turbulence * plant.sin();
        let lateral = 0.3 * turbulence * (0.71 * plant + 1.3).sin();
        let mut offset = (dir * lean + side * lateral) * (response.x * height * height);
        let branch = (2.3 * plant + rel.dot(Vec3::new(0.9, 0.7, 0.8))).sin();
        offset += Vec3::new(0.6 * dir.x, 0.8, 0.6 * dir.z) * (response.y * reach * (0.4 + 0.6 * g) * branch);
        let t = TAU * FLUTTER_HZ * time;
        let q = rel * 5.0;
        let flutter = Vec3::new((t + q.x + q.y).sin(), (1.31 * t + q.y + q.z + 1.7).sin(), (1.17 * t + q.z - q.x + 3.1).sin());
        offset += flutter * (response.z * (0.35 + 0.65 * g));
        offset * strength
    }

    pub fn speed(w: &WindUniform, world: &Mat4, response: Vec4, strength: f32, sphere: Vec4) -> f32 {
        let origin = world.w_axis.truncate();
        let height = (sphere.y + sphere.w - origin.y).max(0.0);
        let reach = Vec3::new(sphere.x - origin.x, 0.0, sphere.z - origin.z).length() + sphere.w;
        let turbulence = w.time.w;
        let gust_rate = TAU * w.time.z * 0.774;
        let plant_rate = TAU * response.w;
        let bend = response.x * height * height * (gust_rate * (1.0 - 0.5 * turbulence) + plant_rate * turbulence * 0.713);
        let sway = response.y * reach * (0.6 * gust_rate + 2.3 * plant_rate);
        let flutter = response.z * (1.126 * gust_rate + 2.0235 * TAU * FLUTTER_HZ);
        strength * (bend + sway + flutter)
    }

    pub fn reach(world: &Mat4, response: Vec4, strength: f32, sphere: Vec4) -> f32 {
        let origin = world.w_axis.truncate();
        let height = (sphere.y + sphere.w - origin.y).max(0.0);
        let reach = Vec3::new(sphere.x - origin.x, 0.0, sphere.z - origin.z).length() + sphere.w;
        strength * (BEND_BOUND * response.x * height * height + response.y * reach + FLUTTER_BOUND * response.z)
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use super::mirror::{offset, reach, speed, strength};
    use super::*;

    /// 1600x900 at a 60° field of view.
    const PIXEL_ANGLE: f32 = 1.2830e-3;

    fn uniform(wind: &Wind, time: f32) -> WindUniform {
        WindUniform::new(wind, 1.0, time, time - 1.0 / 60.0, Vec3::ZERO, PIXEL_ANGLE)
    }

    #[test]
    fn displacement_stays_within_the_culling_bound() {
        let wind = Wind { turbulence: 1.0, strength: 1.7, ..Wind::breeze(1.0) };
        let mut rng = helios_geometry::vegetation::Rng::new(7);
        for response in [WindResponse::WOOD, WindResponse::LEAVES, WindResponse::GRASS, WindResponse::FERN] {
            let r = response.to_vec4();
            for _ in 0..200 {
                let world = Mat4::from_scale_rotation_translation(
                    Vec3::splat(rng.range(0.5, 2.0)),
                    Quat::from_rotation_y(rng.range(0.0, 6.3)),
                    Vec3::new(rng.range(-50.0, 50.0), rng.range(-5.0, 5.0), rng.range(-50.0, 50.0)),
                );
                let sphere = (world.transform_point3(Vec3::new(0.0, rng.range(0.0, 10.0), 0.0))).extend(rng.range(0.1, 12.0));
                let time = rng.range(0.0, 200.0);
                let w = uniform(&wind, time);
                let s = strength(&w, &world, r, sphere);
                let bound = reach(&world, r, s, sphere);
                let max_speed = speed(&w, &world, r, s, sphere);
                for _ in 0..20 {
                    let rest = sphere.truncate() + rng.unit() * sphere.w * rng.float();
                    let phase = rng.range(0.0, std::f32::consts::TAU);
                    let d = offset(&w, &world, r, s, rest, time, phase).length();
                    assert!(d <= bound + 1e-5, "{response:?}: |offset| {d} exceeds the bound {bound}");
                    // Mean speed over a short interval, which the bound on the derivative bounds.
                    let h = 1e-2;
                    let ahead = offset(&w, &world, r, s, rest, time + h, phase);
                    let behind = offset(&w, &world, r, s, rest, time - h, phase);
                    let v = (ahead - behind).length() / (2.0 * h);
                    assert!(v <= max_speed * 1.01 + 1e-3, "{response:?}: speed {v} exceeds the bound {max_speed}");
                }
            }
        }
    }

    #[test]
    fn static_materials_and_calm_air_do_not_move() {
        let w = uniform(&Wind::breeze(1.0), 12.0);
        let sphere = Vec4::new(0.0, 0.3, 0.0, 0.5);
        assert_eq!(strength(&w, &Mat4::IDENTITY, WindResponse::NONE.to_vec4(), sphere), 0.0);
        let calm = uniform(&Wind::default(), 12.0);
        assert!(!calm.is_active());
        assert_eq!(strength(&calm, &Mat4::IDENTITY, WindResponse::GRASS.to_vec4(), sphere), 0.0);
        assert!(WindResponse::NONE.is_static() && !WindResponse::WOOD.is_static());
        assert!(WindResponse::default().is_static());
    }

    #[test]
    fn plants_fade_out_of_the_wind_before_their_disable_distance() {
        let wind = Wind::breeze(1.0);
        let w = uniform(&wind, 3.0);
        // A tree stops at the scene's disable distance, a grass clump where its motion gets too
        // small to see.
        let tree = (WindResponse::LEAVES.to_vec4(), Vec4::new(0.0, 9.0, 0.0, 10.0));
        let grass = (WindResponse::GRASS.to_vec4(), Vec4::new(0.0, 0.3, 0.0, 0.5));
        let at = |(r, sphere): (Vec4, Vec4), d: f32| {
            let offset = Vec3::new(d, 0.0, 0.0);
            strength(&w, &Mat4::from_translation(offset), r, sphere + offset.extend(0.0))
        };
        let grass_limit = reach(&Mat4::IDENTITY, grass.0, 1.0, grass.1) * w.params.x;
        assert!(grass_limit > 15.0 && grass_limit < 0.6 * wind.disable_distance, "{grass_limit}");
        for (plant, limit) in [(tree, wind.disable_distance), (grass, grass_limit)] {
            assert_eq!(at(plant, 0.5 * limit), 1.0);
            assert_eq!(at(plant, limit), 0.0);
            assert_eq!(at(plant, 2.0 * limit), 0.0);
            // Monotonic and continuous through the fade: no popping.
            let mut previous = 1.0;
            for i in 0..=200 {
                let s = at(plant, limit * (0.7 + 0.3 * i as f32 / 200.0));
                assert!(s <= previous && previous - s < 0.02, "{previous} -> {s}");
                previous = s;
            }
        }
    }

    #[test]
    fn the_base_of_a_plant_stays_put_and_its_crown_moves_with_the_wind() {
        let wind = Wind { turbulence: 0.0, ..Wind::breeze(1.0) };
        let world = Mat4::from_translation(Vec3::new(3.0, 1.0, -2.0));
        let r = WindResponse::GRASS.to_vec4();
        let mut downwind = 0.0;
        for i in 0..100 {
            let time = i as f32 * 0.37;
            let w = uniform(&wind, time);
            let base = offset(&w, &world, r, 1.0, Vec3::new(3.0, 1.0, -2.0), time, 0.4);
            assert!(base.length() <= 1.7321 * r.z + 1e-6, "{base}");
            let tip = offset(&w, &world, r, 1.0, Vec3::new(3.0, 1.5, -2.0), time, 0.4);
            downwind += tip.dot(Vec3::new(w.direction.x, 0.0, w.direction.z));
        }
        // Without turbulence the bend only leans downwind.
        assert!(downwind > 0.5, "{downwind}");
    }
}

/// CPU mirror of plant_wind.wgsl: checks that plants' bone motion stays within the procedural
/// bounds their instances carry (`plant_wind.rs`), and that roots hold.
#[cfg(test)]
mod bone_tests {
    use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
    use helios_geometry::vegetation::{self, Rng};
    use helios_geometry::{BoneClass, Mesh, PlantBone, PlantBones};

    use super::mirror::{reach, speed, strength};
    use super::*;

    const TAU: f32 = std::f32::consts::TAU;

    struct Air {
        wind: Vec3,
        side: Vec3,
        up: Vec3,
        gust: f32,
        turbulence: f32,
        plant: f32,
        flutter: f32,
        response: Vec3,
    }

    fn air(w: &WindUniform, world: &Mat4, response: Vec4, strength: f32, time: f32, phase: f32) -> Air {
        let origin = world.w_axis.truncate();
        let to_object = Mat3::from_mat4(*world).inverse();
        let wind = (to_object * Vec3::new(w.direction.x, 0.0, w.direction.z)).normalize();
        let up = (to_object * Vec3::Y).normalize();
        let travel = (origin.x * w.direction.x + origin.z * w.direction.z) / w.params.y;
        let p = TAU * (w.time.z * time - travel);
        let gust = (0.5 + 0.3 * p.sin() + 0.2 * (2.37 * p + 1.9).sin()).clamp(0.0, 1.0);
        Air {
            wind,
            side: wind.cross(up),
            up,
            gust,
            turbulence: w.time.w,
            plant: TAU * response.w * time + phase,
            flutter: TAU * 2.6 * time,
            response: response.truncate() * strength,
        }
    }

    fn rotation(a: &Air, b: &PlantBone) -> Vec3 {
        let phase = b.phase * TAU;
        match b.class {
            BoneClass::Bend => {
                let p = a.plant * b.frequency + phase;
                let lean = a.gust * (1.0 - 0.5 * a.turbulence) + 0.5 * a.turbulence * p.sin();
                let lateral = 0.3 * a.turbulence * (0.71 * p + 1.3).sin();
                b.axis.cross(a.wind * lean + a.side * lateral) * (b.amplitude * a.response.x)
            }
            BoneClass::Sway => {
                let s = (2.3 * b.frequency * a.plant + phase).sin();
                let force = a.wind * (0.3 * a.gust) + (a.wind * 0.6 + a.up * 0.8) * (0.7 * (0.4 + 0.6 * a.gust) * s);
                b.axis.cross(force) * (b.amplitude * a.response.y)
            }
            BoneClass::Flutter => {
                let t = a.flutter * b.frequency + phase;
                let across = b.axis.cross(a.up);
                let e1 = if across.length_squared() > 1e-6 { across.normalize() } else { a.side };
                let e2 = b.axis.cross(e1);
                (e1 * t.sin() + b.axis * (1.31 * t + 1.7).sin() + e2 * (1.17 * t + 3.1).sin()) * (b.amplitude * a.response.z * (0.35 + 0.65 * a.gust))
            }
        }
    }

    /// `plant_pose`: the arc of the own bone, then the parents' frames.
    fn pose(bones: &[PlantBone], a: &Air, bone: u16, p: Vec3) -> Vec3 {
        let mut q = p;
        let mut b = bone as usize;
        let mut s = bones[b].project(p);
        loop {
            let bone = &bones[b];
            let omega = rotation(a, bone);
            let w = omega.length();
            let theta = w * s / bone.length;
            if theta > 1e-7 {
                let k = omega / w;
                let a_par = k * bone.axis.dot(k);
                let a_perp = bone.axis - a_par;
                let (sinc, cosc) = (theta.sin() / theta, 2.0 * (0.5 * theta).sin().powi(2) / theta);
                let arc = bone.pivot + (a_par + a_perp * sinc + k.cross(a_perp) * cosc) * s;
                q = arc + Quat::from_axis_angle(k, theta) * (q - (bone.pivot + bone.axis * s));
            }
            match bone.parent {
                Some(parent) => {
                    s = bone.attach;
                    b = parent as usize;
                }
                None => return q,
            }
        }
    }

    fn plants() -> Vec<(Mesh, WindResponse)> {
        let tree = vegetation::broadleaf_tree(&vegetation::TreeParams { depth: 3, leaves_per_twig: 6, height: 12.0, ..Default::default() });
        let bush = vegetation::bush(61, 1.4, 8);
        let conifer = vegetation::conifer(3, 12.0, 0.25, 4, 8);
        vec![
            (vegetation::grass_clump(91, 40, 0.45), WindResponse::GRASS),
            (vegetation::fern(81, 9, 0.9, 10), WindResponse::FERN),
            (tree.wood, WindResponse::WOOD),
            (tree.foliage, WindResponse::LEAVES),
            (bush.foliage, WindResponse::LEAVES),
            (conifer.foliage, WindResponse::LEAVES),
        ]
    }

    /// Every vertex's bone displacement and speed stay within the procedural bounds of the
    /// equivalent wind (`plant_wind::equivalent_wind`), at strengths up to 2, any scale and time.
    #[test]
    fn bone_motion_stays_within_the_culling_bounds() {
        let mut rng = Rng::new(17);
        for (mesh, response) in plants() {
            let bones = PlantBones::of_mesh(&mesh, mesh.bones.clone());
            let r = response.to_vec4();
            for trial in 0..24 {
                let turbulence = if trial % 2 == 0 { 1.0 } else { 0.0 };
                let wind = Wind { turbulence, strength: rng.range(0.3, 2.0), ..Wind::breeze(1.0) };
                let scale = rng.range(0.6, 1.6);
                let world = Mat4::from_scale_rotation_translation(
                    Vec3::splat(scale),
                    Quat::from_rotation_y(rng.range(0.0, 6.3)),
                    Vec3::new(rng.range(-30.0, 30.0), rng.range(-2.0, 2.0), rng.range(-30.0, 30.0)),
                );
                let time = rng.range(0.0, 300.0);
                let w = WindUniform::new(&wind, 1.0, time, time - 1.0 / 60.0, Vec3::ZERO, 1e-6);
                let equivalent = crate::plant_wind::equivalent_wind(&bones.bounds, r, scale, scale);
                let s = strength(&w, &world, equivalent, world.w_axis.truncate().extend(30.0));
                let phase = rng.range(0.0, TAU);
                let now = air(&w, &world, r, s, time, phase);
                let h = 2e-3;
                let ahead = air(&w, &world, r, s, time + h, phase);
                let behind = air(&w, &world, r, s, time - h, phase);
                for i in (0..mesh.vertices.len()).step_by(7) {
                    let p = mesh.vertices[i].position();
                    let bone = mesh.skin_of(i).dominant_joint();
                    let rest = world.transform_point3(p);
                    let sphere = rest.extend(1e-3);
                    let moved = world.transform_point3(pose(&mesh.bones, &now, bone, p));
                    let bound = reach(&world, equivalent, s, sphere);
                    let d = moved.distance(rest);
                    assert!(d <= bound * 1.001 + 1e-5, "{response:?} vertex {i} at {p}: moved {d} > bound {bound}");
                    let v = world.transform_vector3(pose(&mesh.bones, &ahead, bone, p) - pose(&mesh.bones, &behind, bone, p)).length() / (2.0 * h);
                    let max_speed = speed(&w, &world, equivalent, s, sphere);
                    assert!(v <= max_speed * 1.01 + 1e-4, "{response:?} vertex {i} at {p}: speed {v} > bound {max_speed}");
                }
            }
        }
    }

    /// Bones keep lengths (rotations and arcs) and hold the roots: a grass blade's root does not
    /// move, its tip does.
    #[test]
    fn roots_hold_and_blades_bend() {
        let grass = vegetation::grass_clump(91, 40, 0.45);
        let wind = Wind { turbulence: 0.0, ..Wind::breeze(1.0) };
        let w = WindUniform::new(&wind, 1.0, 5.0, 5.0 - 1.0 / 60.0, Vec3::ZERO, 1e-6);
        let a = air(&w, &Mat4::IDENTITY, WindResponse::GRASS.to_vec4(), 1.0, 5.0, 0.3);
        let mut tip_moved = 0.0f32;
        for (i, v) in grass.vertices.iter().enumerate() {
            let p = v.position();
            let bone = grass.skin_of(i).dominant_joint();
            let q = pose(&grass.bones, &a, bone, p);
            let b = &grass.bones[bone as usize];
            if p.distance(b.pivot) < 1e-3 {
                assert!(q.distance(p) < 1e-5, "a root moved {}", q.distance(p));
            }
            tip_moved = tip_moved.max(q.distance(p));
        }
        assert!(tip_moved > 0.01, "blades bend: {tip_moved}");
        // Nothing stretches: every edge keeps its length (an arc bends, it does not extend).
        let posed: Vec<Vec3> = grass.vertices.iter().enumerate().map(|(i, v)| pose(&grass.bones, &a, grass.skin_of(i).dominant_joint(), v.position())).collect();
        for t in grass.indices.chunks_exact(3) {
            for (x, y) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let rest = grass.vertices[x as usize].position().distance(grass.vertices[y as usize].position());
                let now = posed[x as usize].distance(posed[y as usize]);
                assert!((now - rest).abs() <= 0.03 * rest + 1e-5, "edge {rest} -> {now}");
            }
        }
    }
}

