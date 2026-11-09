//! Skeletal meshes on virtual geometry (UE 5.5's Nanite skinned meshes); the deformation itself
//! is `shaders/skinning.wgsl`.
//!
//! A [`SkinHandle`] is a joint palette that instances are bound to ([`Scene::set_instance_skin`]);
//! the application poses it every frame ([`Scene::set_skin_pose`], e.g. from an animation
//! player). Each matrix maps a rest (bind pose) vertex of the meshes bound to the skin to its
//! posed position in the same object space; the instance transform then places it. Meshes keep
//! their joint influences in the compressed clusters (`helios_geometry::encoding`), so skinning
//! — linear blend of the influencing joints' matrices — is a pure function of the rest vertex,
//! its quantised weights and the palette, evaluated identically by all four rasterisers
//! (visibility buffer and shadow pages, hardware and software) and by the G-buffer resolve,
//! exactly like the wind's offset: clusters and LOD levels sharing a vertex move it to the same
//! bits, so cuts stay watertight under animation. Motion vectors read the previous frame's palette.
//!
//! **Culling bounds.** The rest-pose cluster, page and instance spheres no longer hold the
//! geometry once it moves. Per frame and skin, the CPU derives two conservative bounds from
//! per-joint rest spheres (the vertices each joint influences, [`Scene::add_skin`]): a linearly
//! blended vertex lies in the convex hull of its joints' transforms of it, so
//! * the *posed sphere* — enclosing every joint's posed rest sphere — contains the whole posed
//!   mesh, and
//! * the *reach* — the largest distance any joint's transform moves a point of its rest sphere,
//!   `|P c − c| + ‖A − I‖₂ r` — bounds every vertex's displacement.
//!
//! Every visibility test (frustum, HZB, shadow pages) uses the rest sphere padded by the reach, or
//! the posed sphere where that is smaller (large motion, e.g. a figure turning around); LOD keeps
//! the rest-pose spheres, which preserves the monotonic errors and bounds of the DAG (crack-free
//! cuts) — the pose changes where geometry is, not how detailed it must be.
//!
//! **Shadows.** A skin that moved this frame makes its instances dynamic casters in the virtual
//! shadow map cache, like the wind: cached pages under them re-render their dynamic casters over
//! the cached static layer (`vsm_invalidate.wgsl`) once they may have moved 2 texels, bounded per
//! frame by the joints' motion `|ΔP c| + ‖ΔA‖₂ r`. The footprint is the rest sphere padded by the
//! largest reach seen so far, which covers every pose a cached page may still show.
//!
//! **Lumen.** As in UE, skinned meshes get no distance field or mesh cards (the application bakes
//! them without one); screen traces still see them.

use bytemuck::Zeroable;
use glam::{Mat3, Mat4, Vec3, Vec4};

use crate::scene::{Instance, Scene};

/// Binding of the skin palette texture in every shader that positions or culls geometry.
pub const BINDING_SKIN: u32 = 66;
/// Texels per palette row (`SKIN_TEXTURE_WIDTH` in skinning.wgsl).
const WIDTH: u32 = 1024;
/// Texels of a skin's header: the posed sphere, then (reach, largest reach, motion, -).
const HEADER_TEXELS: u32 = 2;
/// Texels per joint: the current matrix's three rows, then the previous frame's.
const JOINT_TEXELS: u32 = 6;
/// [`crate::gpu_types::GpuInstance::flags`] bits from here up hold the instance's palette (its
/// first texel + 1; 0: not skinned).
pub const INSTANCE_SKIN_SHIFT: u32 = 12;
const MAX_PALETTE_TEXELS: u32 = (1 << (32 - INSTANCE_SKIN_SHIFT)) - 1;

/// A joint palette that skinned instances follow ([`Scene::add_skin`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SkinHandle(pub(crate) u32);

/// A skin: per-joint rest bounds and the current pose.
#[derive(Clone, Debug)]
pub(crate) struct Skin {
    /// Per joint, the rest-space sphere (centre, radius) of the vertices it influences in any
    /// mesh bound to the skin; a negative radius for joints without vertices.
    joint_bounds: Vec<Vec4>,
    /// Rest → posed matrices in the instances' object space.
    pose: Vec<Mat4>,
}

/// Layout entry of the palette texture (read with `textureLoad`, unfilterable).
pub fn skin_entry() -> (u32, wgpu::BindingType) {
    (BINDING_SKIN, crate::util::bind::texture(false))
}

/// The scene's palette texture for [`skin_entry`].
pub fn skin_resource(scene: &crate::gpu_scene::GpuScene) -> (u32, wgpu::BindingResource<'_>) {
    (BINDING_SKIN, wgpu::BindingResource::TextureView(&scene.skins.view))
}

impl Scene {
    /// Adds a joint palette for meshes with skin weights; `joint_bounds[j]` is the rest-space
    /// bounding sphere (centre, radius) of every vertex joint `j` influences in the meshes that
    /// will be bound to it, a negative radius if none (see `helios_geometry::skin::joint_bounds`).
    /// It starts in the rest pose (identity matrices).
    pub fn add_skin(&mut self, joint_bounds: Vec<Vec4>) -> SkinHandle {
        let pose = vec![Mat4::IDENTITY; joint_bounds.len()];
        self.skins.push(Skin { joint_bounds, pose });
        SkinHandle(self.skins.len() as u32 - 1)
    }

    /// Poses a skin: one matrix per joint, mapping rest positions of the bound meshes to posed
    /// ones in their object space (glTF: `inverse(mesh node) · joint · inverse bind matrix`).
    /// Call it whenever the pose changes, typically every frame; the renderer keeps the previous
    /// frame's for motion vectors.
    pub fn set_skin_pose(&mut self, skin: SkinHandle, pose: &[Mat4]) {
        let skin = &mut self.skins[skin.0 as usize];
        assert_eq!(pose.len(), skin.pose.len(), "one matrix per joint");
        skin.pose.copy_from_slice(pose);
    }

    pub fn skin_joint_count(&self, skin: SkinHandle) -> usize {
        self.skins[skin.0 as usize].pose.len()
    }

    /// Binds an instance (whose mesh has skin weights) to a skin, or unbinds it.
    pub fn set_instance_skin(&mut self, instance: crate::InstanceHandle, skin: Option<SkinHandle>) {
        self.instances[instance.0 as usize].skin = skin;
        self.instance_version = crate::scene::next_version();
    }
}

/// First palette texel of every skin, and the total.
fn layout(skins: &[Skin]) -> (Vec<u32>, u32) {
    let mut next = 0;
    let bases = skins
        .iter()
        .map(|s| {
            let base = next;
            next += HEADER_TEXELS + JOINT_TEXELS * s.pose.len() as u32;
            base
        })
        .collect();
    (bases, next)
}

/// The skin bits of an instance's [`crate::gpu_types::GpuInstance::flags`].
pub(crate) fn instance_skin_flags(scene: &Scene, instance: &Instance) -> u32 {
    let Some(skin) = instance.skin else { return 0 };
    let base = layout(&scene.skins[..skin.0 as usize]).1;
    if base >= MAX_PALETTE_TEXELS {
        log::error!("skin palettes exceed {MAX_PALETTE_TEXELS} texels; instance drawn unskinned");
        return 0;
    }
    (base + 1) << INSTANCE_SKIN_SHIFT
}

/// Upper bound of the spectral norm (largest stretch) of `m`: the square root of the largest
/// eigenvalue of `mᵀm`, solved in closed form, with a small margin for rounding.
fn spectral_norm(m: Mat3) -> f32 {
    let a = (m.transpose() * m).to_cols_array_2d().map(|c| c.map(|v| v as f64));
    let p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    let p2 = (a[0][0] - q).powi(2) + (a[1][1] - q).powi(2) + (a[2][2] - q).powi(2) + 2.0 * p1;
    let largest = if p2 <= 1e-30 {
        q
    } else {
        // Smith's trigonometric solution for symmetric 3x3 matrices.
        let p = (p2 / 6.0).sqrt();
        let b = |i: usize, j: usize| (a[i][j] - if i == j { q } else { 0.0 }) / p;
        let det = b(0, 0) * (b(1, 1) * b(2, 2) - b(1, 2) * b(2, 1)) - b(0, 1) * (b(1, 0) * b(2, 2) - b(1, 2) * b(2, 0))
            + b(0, 2) * (b(1, 0) * b(2, 1) - b(1, 1) * b(2, 0));
        q + 2.0 * p * ((det * 0.5).clamp(-1.0, 1.0).acos() / 3.0).cos()
    };
    (largest.max(0.0).sqrt() * 1.0001 + 1e-7) as f32
}

/// Bounds of a pose ([`Skin::joint_bounds`] under `pose`, both in object space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PoseBounds {
    /// Encloses the whole posed mesh: every joint's posed rest sphere.
    pub sphere: Vec4,
    /// Largest displacement of any vertex from its rest position.
    pub reach: f32,
}

pub(crate) fn pose_bounds(joint_bounds: &[Vec4], pose: &[Mat4]) -> PoseBounds {
    let mut spheres = Vec::with_capacity(joint_bounds.len());
    let mut reach = 0.0f32;
    for (bounds, m) in joint_bounds.iter().zip(pose).filter(|(b, _)| b.w >= 0.0) {
        let (c, r) = (bounds.truncate(), bounds.w);
        let linear = Mat3::from_mat4(*m);
        let posed = m.transform_point3(c);
        spheres.push((posed, spectral_norm(linear) * r));
        reach = reach.max(posed.distance(c) + spectral_norm(linear - Mat3::IDENTITY) * r);
    }
    if spheres.is_empty() {
        return PoseBounds { sphere: Vec4::new(0.0, 0.0, 0.0, 0.0), reach: 0.0 };
    }
    let (lo, hi) = spheres.iter().fold((Vec3::MAX, Vec3::MIN), |(lo, hi), &(c, r)| (lo.min(c - r), hi.max(c + r)));
    let center = (lo + hi) * 0.5;
    let radius = spheres.iter().map(|&(c, r)| center.distance(c) + r).fold(0.0f32, f32::max);
    PoseBounds { sphere: center.extend(radius * 1.0001), reach: reach * 1.0001 }
}

/// Largest distance any vertex moved from pose `previous` to `pose`.
pub(crate) fn pose_motion(joint_bounds: &[Vec4], pose: &[Mat4], previous: &[Mat4]) -> f32 {
    let mut motion = 0.0f32;
    for ((bounds, m), p) in joint_bounds.iter().zip(pose).zip(previous).filter(|((b, _), _)| b.w >= 0.0) {
        if m == p {
            continue;
        }
        let c = bounds.truncate();
        motion = motion.max(m.transform_point3(c).distance(p.transform_point3(c)) + spectral_norm(Mat3::from_mat4(*m) - Mat3::from_mat4(*p)) * bounds.w);
    }
    motion * 1.0001
}

/// The rows of an affine matrix as `skinning.wgsl` reads them.
fn matrix_rows(m: &Mat4) -> [Vec4; 3] {
    let t = m.transpose();
    [t.x_axis, t.y_axis, t.z_axis]
}

/// GPU palettes of a scene's skins: one `Rgba32Float` texture, [`WIDTH`] texels per row. Per
/// skin, from its first texel: the posed object-space sphere, then (reach, largest reach so far,
/// motion since the previous frame, 0), then per joint the three rows of its current matrix and
/// of the previous frame's (motion vectors). Rewritten every frame (a few KiB per character).
pub(crate) struct SkinPalettes {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    /// The scene and skin layout the state below belongs to.
    signature: (u64, Vec<usize>),
    /// Per skin: the pose uploaded last frame, and the largest reach since the scene was loaded.
    previous: Vec<Vec<Mat4>>,
    reach_max: Vec<f32>,
    /// A skin bound to a shadow-casting instance moved this frame.
    pub moving_shadows: bool,
    /// Some instance is bound to a skin (the G-buffer resolve needs its skinning permutation).
    pub skinned_instances: bool,
    staging: Vec<Vec4>,
}

impl SkinPalettes {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = Self::texture(device, 1);
        Self {
            view: texture.create_view(&Default::default()),
            texture,
            signature: (u64::MAX, Vec::new()),
            previous: Vec::new(),
            reach_max: Vec::new(),
            moving_shadows: false,
            skinned_instances: false,
            staging: Vec::new(),
        }
    }

    fn texture(device: &wgpu::Device, rows: u32) -> wgpu::Texture {
        crate::util::texture(
            device,
            crate::util::TextureSpec {
                label: "skin palettes",
                width: WIDTH,
                height: rows,
                layers: 1,
                mips: 1,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            },
        )
    }

    /// Uploads this frame's palettes (scenes without skins upload nothing).
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) {
        let signature = (scene.geometry_version, scene.skins.iter().map(|s| s.pose.len()).collect::<Vec<_>>());
        if signature != self.signature {
            // A new scene (or skin): no motion from whatever was shown before.
            self.previous = scene.skins.iter().map(|s| s.pose.clone()).collect();
            self.reach_max = vec![0.0; scene.skins.len()];
            self.signature = signature;
        }
        self.moving_shadows = false;
        self.skinned_instances = scene.instances.iter().any(|i| i.skin.is_some());
        if scene.skins.is_empty() {
            return;
        }
        let (_, texels) = layout(&scene.skins);
        let rows = texels.div_ceil(WIDTH);
        if rows > self.texture.height() {
            self.texture = Self::texture(device, rows.next_power_of_two());
            self.view = self.texture.create_view(&Default::default());
        }
        self.staging.clear();
        let mut moving = vec![false; scene.skins.len()];
        for (i, skin) in scene.skins.iter().enumerate() {
            let bounds = pose_bounds(&skin.joint_bounds, &skin.pose);
            let motion = pose_motion(&skin.joint_bounds, &skin.pose, &self.previous[i]);
            self.reach_max[i] = self.reach_max[i].max(bounds.reach);
            moving[i] = motion > 0.0;
            self.staging.push(bounds.sphere);
            self.staging.push(Vec4::new(bounds.reach, self.reach_max[i], motion, 0.0));
            for (current, previous) in skin.pose.iter().zip(&self.previous[i]) {
                self.staging.extend(matrix_rows(current));
                self.staging.extend(matrix_rows(previous));
            }
            self.previous[i].copy_from_slice(&skin.pose);
        }
        self.staging.resize((rows * WIDTH) as usize, Vec4::zeroed());
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&self.staging),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(WIDTH * 16), rows_per_image: Some(rows) },
            wgpu::Extent3d { width: WIDTH, height: rows, depth_or_array_layers: 1 },
        );
        self.moving_shadows = scene.instances.iter().any(|i| i.cast_shadow && i.skin.is_some_and(|s| moving[s.0 as usize]));
    }
}

#[cfg(test)]
mod tests {
    use glam::Quat;
    use helios_geometry::skin::skin_point;
    use helios_geometry::SkinWeights;

    use super::*;

    fn rng(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        (*seed >> 8) as f32 / (1u32 << 24) as f32
    }

    #[test]
    fn spectral_norms_bound_every_stretch() {
        let mut seed = 99;
        for i in 0..500 {
            let m = if i % 3 == 0 {
                Mat3::from_quat(Quat::from_euler(glam::EulerRot::XYZ, rng(&mut seed) * 6.0, rng(&mut seed) * 6.0, rng(&mut seed) * 6.0)) * (0.5 + rng(&mut seed))
            } else {
                Mat3::from_cols_array(&std::array::from_fn(|_| rng(&mut seed) * 2.0 - 1.0))
            };
            let norm = spectral_norm(m);
            let mut largest = 0.0f32;
            for _ in 0..400 {
                let v = Vec3::new(rng(&mut seed) - 0.5, rng(&mut seed) - 0.5, rng(&mut seed) - 0.5).normalize_or(Vec3::X);
                largest = largest.max((m * v).length());
            }
            assert!(largest <= norm, "{largest} > {norm} for {m}");
            assert!(norm <= largest * 1.05 + 1e-4, "loose: {norm} vs {largest}");
        }
        assert!((spectral_norm(Mat3::IDENTITY) - 1.0).abs() < 1e-3);
        assert!(spectral_norm(Mat3::ZERO) < 1e-6);
    }

    /// The posed sphere holds every skinned vertex, the reach bounds every displacement and the
    /// motion every vertex's travel between two poses — for random weights and joint
    /// transforms, scale included.
    #[test]
    fn pose_bounds_hold_every_skinned_vertex() {
        let mut seed = 7;
        for _ in 0..40 {
            let joints = 5;
            let vertices: Vec<(Vec3, SkinWeights)> = (0..300)
                .map(|_| {
                    let p = Vec3::new(rng(&mut seed), rng(&mut seed) * 2.0, rng(&mut seed)) - 0.5;
                    let j = [0, 1, 2, 3].map(|_| (rng(&mut seed) * joints as f32) as u16 % joints as u16);
                    let w = [1.0, rng(&mut seed), rng(&mut seed) * 0.5, rng(&mut seed) * 0.2];
                    (p, SkinWeights::new(j, w))
                })
                .collect();
            let mut mesh = helios_geometry::Mesh::default();
            mesh.vertices = vertices.iter().map(|(p, _)| helios_geometry::Vertex::new(*p, Vec3::Y, glam::Vec2::ZERO)).collect();
            mesh.skin = vertices.iter().map(|v| v.1).collect();
            let joint_bounds = helios_geometry::skin::joint_bounds_vec4(&helios_geometry::skin::joint_bounds(&mesh, joints));
            let mut random_pose = || -> Vec<Mat4> {
                (0..joints)
                    .map(|_| {
                        let rotation = Quat::from_scaled_axis(Vec3::new(rng(&mut seed) - 0.5, rng(&mut seed) - 0.5, rng(&mut seed) - 0.5) * 3.0);
                        let translation = Vec3::new(rng(&mut seed) - 0.5, rng(&mut seed) - 0.5, rng(&mut seed) - 0.5);
                        Mat4::from_scale_rotation_translation(Vec3::new(0.8 + rng(&mut seed) * 0.4, 1.0, 1.0), rotation, translation)
                    })
                    .collect()
            };
            let (a, b) = (random_pose(), random_pose());
            let bounds = pose_bounds(&joint_bounds, &a);
            let motion = pose_motion(&joint_bounds, &a, &b);
            assert_eq!(pose_motion(&joint_bounds, &a, &a), 0.0);
            for (rest, skin) in &vertices {
                let posed = skin_point(skin, &a, *rest);
                assert!(posed.distance(bounds.sphere.truncate()) <= bounds.sphere.w + 1e-4, "outside the posed sphere");
                assert!(posed.distance(*rest) <= bounds.reach + 1e-4, "{} > reach {}", posed.distance(*rest), bounds.reach);
                assert!(posed.distance(skin_point(skin, &b, *rest)) <= motion + 1e-4, "moved further than the motion bound");
            }
        }
        // The rest pose moves nothing.
        let rest = pose_bounds(&[Vec4::new(1.0, 2.0, 3.0, 0.5)], &[Mat4::IDENTITY]);
        assert!(rest.reach < 1e-4 && (rest.sphere - Vec4::new(1.0, 2.0, 3.0, 0.5)).abs().max_element() < 1e-3);
    }

    #[test]
    fn palettes_lay_out_skin_after_skin() {
        let mut scene = Scene::new();
        let a = scene.add_skin(vec![Vec4::ZERO; 3]);
        let b = scene.add_skin(vec![Vec4::ZERO; 2]);
        assert_eq!(layout(&scene.skins), (vec![0, HEADER_TEXELS + 3 * JOINT_TEXELS], 2 * HEADER_TEXELS + 5 * JOINT_TEXELS));
        let mesh = scene.add_mesh(helios_geometry::VirtualMesh::default());
        let material = scene.add_material(crate::Material::default());
        let first = scene.add_instance(mesh, material, Mat4::IDENTITY);
        let second = scene.add_instance(mesh, material, Mat4::IDENTITY);
        scene.set_instance_skin(second, Some(b));
        let flags = |scene: &Scene, h: crate::InstanceHandle| instance_skin_flags(scene, &scene.instances[h.0 as usize]);
        assert_eq!(flags(&scene, first), 0);
        assert_eq!(flags(&scene, second) >> INSTANCE_SKIN_SHIFT, HEADER_TEXELS + 3 * JOINT_TEXELS + 1);
        scene.set_instance_skin(first, Some(a));
        assert_eq!(flags(&scene, first) >> INSTANCE_SKIN_SHIFT, 1);
        let translated = Mat4::from_translation(Vec3::X);
        assert_eq!(matrix_rows(&translated)[0], Vec4::new(1.0, 0.0, 0.0, 1.0));
    }
}
