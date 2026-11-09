//! A scene built from an imported glTF model (`--gltf file`, `?gltf=URL` in the browser).
//!
//! Every unique mesh is baked into virtual geometry once (in parallel natively) and placed by
//! the file's node instances. The model is moved so its bounding box is centred on x = z = 0
//! with its bottom at y = 0, standing on an optional procedural ground; the camera frames its
//! bounds. Skinned meshes follow joint palettes posed by the file's animations
//! ([`super::skeletal`]); like UE's skinned meshes they get no distance field (Lumen's screen
//! traces still see them).

use std::collections::HashMap;
use std::sync::Arc;

use glam::{Mat4, Vec3};
use helios_asset::{AlphaMode, Asset, TextureRef, WrapMode};
use helios_geometry::{primitives, BuildConfig, Mesh, SdfMode, Vertex, VirtualMesh};
use helios_render::{
    Camera, Image, ImageTexture, Material, MaterialKind, Scene, Sun, TextureDesc, TextureHandle, TextureMapping, TexturePattern, UvTransform, Wrap,
};
use web_time::Instant;

use super::skeletal::{posed_bounds, SkeletalAnimation, SkinBindings};
use super::SceneSetup;

const SUN: (f32, f32, f32) = (0.9, 0.95, 10.0);

/// Target world-space voxel size of mesh distance fields. Lumen's finest global distance field
/// level has 0.25 m voxels, so finer mesh fields gain little.
const SDF_VOXEL: f32 = 0.12;
const SDF_RESOLUTION: (u32, u32) = (24, 128);

/// An imported model kept for rebuilding its scene (scene cycling re-bakes, as for the
/// procedural scenes). Images are shared with the scenes built from it.
pub struct Model {
    pub name: String,
    /// Put a ground plane under the model.
    pub ground: bool,
    asset: Asset,
    images: Vec<Arc<Image>>,
}

impl Model {
    pub fn new(name: impl Into<String>, mut asset: Asset) -> Self {
        let images = std::mem::take(&mut asset.images)
            .into_iter()
            .map(|i| Arc::new(Image { width: i.width, height: i.height, pixels: i.pixels }))
            .collect();
        Self { name: name.into(), ground: true, asset, images }
    }
}

/// The renderer's alpha cutoff for a glTF material, against the base colour *texture's* alpha
/// (glTF tests texture alpha × factor alpha, so the factor divides the cutoff). `BLEND` is not
/// alpha-tested: it is drawn by the forward translucency pass (see `add_materials`).
fn alpha_cutoff(material: &helios_asset::Material) -> Option<f32> {
    let AlphaMode::Mask { cutoff } = material.alpha_mode else { return None };
    Some(cutoff / material.base_color.w.max(1e-3))
}

/// Alpha-tested materials whose factor alpha is so low that no texel passes are not drawn.
/// (Without a base colour texture the test passes everywhere or nowhere; the renderer applies
/// the alpha test only with a texture.)
fn fully_cut_out(material: &helios_asset::Material) -> bool {
    alpha_cutoff(material).is_some_and(|c| c > 1.0)
}

pub fn build(model: &Model) -> SceneSetup {
    let asset = &model.asset;
    let mut scene = Scene::new();
    scene.sun = Sun::from_angles(SUN.0, SUN.1, SUN.2);

    let skins = SkinBindings::of(asset);
    let bounds = posed_bounds(asset, &skins);
    let size = bounds.extent();
    let placement = Mat4::from_translation(Vec3::new(-bounds.center().x, -bounds.min.y, -bounds.center().z));
    log::info!(
        "model '{}': {} meshes, {} instances, {} materials, {} images, {:.2} M triangles, {:.1} x {:.1} x {:.1} m",
        model.name,
        asset.meshes.len(),
        asset.instances.len(),
        asset.materials.len(),
        model.images.len(),
        asset.triangle_count() as f64 / 1e6,
        size.x,
        size.y,
        size.z
    );

    let materials = add_materials(&mut scene, model);
    let skipped: Vec<bool> = asset.materials.iter().map(fully_cut_out).collect();
    let blended = |material: usize| asset.materials[material].alpha_mode == AlphaMode::Blend;

    // Bake every mesh some visible instance uses, with a distance field sized to its world extent
    // (none for meshes only drawn with blended materials: translucent surfaces neither occlude
    // nor reflect in Lumen).
    let mut world_scale = vec![0.0f32; asset.meshes.len()];
    let mut double_sided = vec![false; asset.meshes.len()];
    let mut opaque = vec![false; asset.meshes.len()];
    let mut skinned = vec![false; asset.meshes.len()];
    for (i, instance) in asset.instances.iter().enumerate() {
        skinned[instance.mesh] |= skins.instance_skin.contains_key(&i);
    }
    for instance in asset.instances.iter().filter(|i| !skipped[i.material]) {
        let m = instance.transform;
        let scale = m.x_axis.truncate().length().max(m.y_axis.truncate().length()).max(m.z_axis.truncate().length());
        world_scale[instance.mesh] = world_scale[instance.mesh].max(scale);
        double_sided[instance.mesh] |= asset.materials[instance.material].double_sided;
        opaque[instance.mesh] |= !blended(instance.material);
    }
    let jobs: Vec<usize> = (0..asset.meshes.len()).filter(|&i| world_scale[i] > 0.0).collect();
    let start = Instant::now();
    let baked = super::par_map(&jobs, |&i| {
        let source = &asset.meshes[i];
        let mesh = if double_sided[i] { with_back_faces(&source.mesh) } else { source.mesh.clone() };
        let extent = source.mesh.aabb().extent().max_element() * world_scale[i];
        let resolution = ((extent / SDF_VOXEL).ceil() as u32).clamp(SDF_RESOLUTION.0, SDF_RESOLUTION.1);
        let sdf = if opaque[i] && !skinned[i] { SdfMode::Volume { resolution } } else { SdfMode::None };
        super::bake_with(&source.name, &mesh, BuildConfig { sdf, ..Default::default() })
    });
    let voxels: usize = baked.iter().filter_map(|vm: &VirtualMesh| vm.sdf.as_ref()).map(|s| s.voxel_count()).sum();
    log::info!("baked {} meshes in {:.1} s ({:.1} M distance field voxels)", jobs.len(), start.elapsed().as_secs_f64(), voxels as f64 / 1e6);

    let mut handles = HashMap::new();
    for (&i, vm) in jobs.iter().zip(baked) {
        handles.insert(i, scene.add_mesh(vm));
    }
    let skin_handles: Vec<_> = skins.joint_bounds(asset).into_iter().map(|bounds| scene.add_skin(bounds)).collect();
    for (i, instance) in asset.instances.iter().enumerate().filter(|(_, i)| !skipped[i.material]) {
        let handle = scene.add_instance(handles[&instance.mesh], materials[instance.material], placement * instance.transform);
        // Blended surfaces let light through.
        scene.set_cast_shadow(handle, !blended(instance.material));
        if let Some(&skin) = skins.instance_skin.get(&i) {
            scene.set_instance_skin(handle, Some(skin_handles[skin]));
        }
    }
    let bindings = skins.skins.iter().zip(&skin_handles).map(|(&(skin, bind), &handle)| (handle, skin, bind)).collect();
    let mut animation = SkeletalAnimation::new(asset, bindings);
    if let Some(animation) = &mut animation {
        animation.update(&mut scene, 0.0);
    }

    if model.ground {
        add_ground(&mut scene, size.x.max(size.z));
    }
    scene.ground_albedo = Vec3::splat(0.25);
    scene.clouds = helios_render::CloudLayer::cumulus(0.35);
    SceneSetup { scene, camera: frame_camera(size), sun: SUN, floor: |_, _| 0.0, animation }
}

/// The renderer's texture coordinates for a glTF texture reference.
fn uv_transform(texture: Option<&TextureRef>) -> UvTransform {
    let wrap = |mode: WrapMode| match mode {
        WrapMode::Repeat => Wrap::Repeat,
        WrapMode::MirroredRepeat => Wrap::MirroredRepeat,
        WrapMode::ClampToEdge => Wrap::ClampToEdge,
    };
    texture.map_or(UvTransform::IDENTITY, |t| UvTransform { set: t.tex_coord, matrix: t.transform.matrix(), wrap: t.wrap.map(wrap) })
}

/// The images of a material's texture layers: the main layer (base colour, normal,
/// metallic-roughness and — when it can share them — occlusion images, which must all sample
/// with the same coordinates), an occlusion layer if the occlusion could not join it, and an
/// emission layer. Keys are image indices: base colour, normal, metallic-roughness, occlusion.
struct LayerKeys {
    main: Option<[Option<usize>; 4]>,
    /// The texture reference whose coordinates the main layer uses.
    main_ref: Option<TextureRef>,
    occlusion_in_main: bool,
    occlusion: Option<[Option<usize>; 4]>,
    emission: Option<[Option<usize>; 4]>,
}

impl LayerKeys {
    fn of(m: &helios_asset::Material) -> Self {
        let slots = [m.base_color_texture, m.normal_texture, m.metallic_roughness_texture];
        let main_ref = slots.iter().flatten().next().copied();
        for slot in slots.iter().flatten().filter(|t| !main_ref.is_some_and(|r| r.same_coordinates(t))) {
            // One layer is sampled once per pixel: its images share coordinates.
            log::warn!("material '{}': image {} uses other texture coordinates than the material's first texture; sampled with those", m.name, slot.image);
        }
        // Occlusion takes the main layer's albedo alpha unless that alpha is the coverage of a
        // blended material or the coordinates differ; then it gets a layer of its own (white
        // base colour, occlusion in the alpha).
        let occlusion_in_main = m.alpha_mode != AlphaMode::Blend && m.occlusion_texture.is_some_and(|o| main_ref.is_some_and(|r| r.same_coordinates(&o)));
        let image = |t: Option<TextureRef>| t.map(|t| t.image);
        let main = [image(m.base_color_texture), image(m.normal_texture), image(m.metallic_roughness_texture), image(m.occlusion_texture).filter(|_| occlusion_in_main)];
        Self {
            main: main.iter().any(Option::is_some).then_some(main),
            main_ref,
            occlusion_in_main,
            occlusion: m.occlusion_texture.filter(|_| !occlusion_in_main).map(|o| [None, None, None, Some(o.image)]),
            emission: m.emissive_texture.map(|e| [Some(e.image), None, None, None]),
        }
    }
}

/// glTF materials → renderer materials. Images that sample with the same coordinates are packed
/// into one texture layer ([`LayerKeys`]); layers are shared between materials that use the same
/// images (with any transforms: coordinates are per material).
fn add_materials(scene: &mut Scene, model: &Model) -> Vec<helios_render::MaterialHandle> {
    let image = |i: Option<usize>| i.and_then(|i| model.images.get(i)).cloned();
    let mut layers: HashMap<[Option<usize>; 4], TextureHandle> = HashMap::new();
    let mut layer = |scene: &mut Scene, key: [Option<usize>; 4]| {
        *layers.entry(key).or_insert_with(|| {
            scene.add_image_texture(ImageTexture {
                base_color: image(key[0]),
                normal: image(key[1]),
                metallic_roughness: image(key[2]),
                occlusion: image(key[3]),
            })
        })
    };
    model
        .asset
        .materials
        .iter()
        .map(|m| {
            let keys = LayerKeys::of(m);
            let texture = keys.main.map(|key| layer(scene, key));
            let emissive_texture = keys.emission.map(|key| layer(scene, key));
            let occlusion_texture = if keys.occlusion_in_main { texture } else { keys.occlusion.map(|key| layer(scene, key)) };
            // BLEND is opt-in translucency: the forward translucency pass, alpha-blended by the
            // base colour alpha (factor × texture). OPAQUE and MASK stay in the visibility buffer.
            let kind = if m.alpha_mode == AlphaMode::Blend { MaterialKind::Translucent } else { MaterialKind::Standard };
            scene.add_material(Material {
                kind,
                base_color: m.base_color.truncate(),
                opacity: m.base_color.w,
                roughness: m.roughness,
                metallic: m.metallic,
                texture,
                mapping: TextureMapping::MeshUv,
                uv: uv_transform(keys.main_ref.as_ref()),
                normal_strength: m.normal_scale,
                emissive_color: m.emissive,
                emissive_texture,
                emissive_uv: uv_transform(m.emissive_texture.as_ref()),
                occlusion_texture,
                occlusion_uv: uv_transform(m.occlusion_texture.as_ref()),
                occlusion_strength: m.occlusion_strength,
                alpha_cutoff: alpha_cutoff(m),
                ..Default::default()
            })
        })
        .collect()
}

/// Appends reversed copies of all triangles (normals flipped) so double-sided materials render
/// from behind: the rasteriser culls back faces.
fn with_back_faces(mesh: &Mesh) -> Mesh {
    let mut vertices = mesh.vertices.clone();
    vertices.extend(mesh.vertices.iter().map(|v| Vertex { normal: (-v.normal()).into(), ..*v }));
    let offset = mesh.vertices.len() as u32;
    let mut indices = mesh.indices.clone();
    indices.extend(mesh.triangles().flat_map(|[a, b, c]| [a + offset, c + offset, b + offset]));
    Mesh { skin: twice(&mesh.skin), ..Mesh::with_attributes(vertices, indices, twice(&mesh.uv1), twice(&mesh.colors)) }
}

/// The optional attribute `values` for the front and the back copy of every vertex.
fn twice<T: Copy>(values: &[T]) -> Vec<T> {
    values.iter().chain(values).copied().collect()
}

/// A paved ground plane a little below the model (its own floor, if any, stays visible).
fn add_ground(scene: &mut Scene, model_size: f32) {
    let size = (model_size * 6.0).max(60.0);
    let mut plane = primitives::terrain(size, 64, 0.0, 1);
    plane.deform(|p| Vec3::new(p.x, -0.01, p.z));
    let mesh = scene.add_mesh(super::bake_with("ground", &plane, BuildConfig { sdf: SdfMode::Heightfield { resolution: 128 }, ..Default::default() }));
    let stone = scene.add_texture(TextureDesc::new(TexturePattern::Flagstone, Vec3::new(0.42, 0.4, 0.37), Vec3::new(0.3, 0.29, 0.27)).roughness(0.8, 0.9));
    let material = scene.add_material(Material::textured(MaterialKind::Standard, stone, TextureMapping::WorldTriplanar, 0.25));
    scene.add_instance(mesh, material, Mat4::IDENTITY);
}

/// Looks at the model's centre from the front-right and above, far enough to see all of it.
fn frame_camera(size: Vec3) -> Camera {
    let mut camera = Camera { fov_y: 50f32.to_radians(), ..Default::default() };
    let target = Vec3::new(0.0, size.y * 0.4, 0.0);
    let radius = size.length() * 0.5;
    let distance = radius / (camera.fov_y * 0.5).tan() * 1.05;
    camera.position = target + Vec3::new(0.55, 0.45, 0.7).normalize() * distance;
    camera.near = (radius * 1e-3).clamp(0.005, 0.1);
    camera.look_at(target);
    camera
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn back_faces_reverse_winding_and_normals() {
        let mesh = Mesh::new(
            vec![
                Vertex::new(Vec3::ZERO, Vec3::Z, glam::Vec2::ZERO),
                Vertex::new(Vec3::X, Vec3::Z, glam::Vec2::X),
                Vertex::new(Vec3::Y, Vec3::Z, glam::Vec2::Y),
            ],
            vec![0, 1, 2],
        );
        let two_sided = with_back_faces(&mesh);
        assert_eq!(two_sided.triangle_count(), 2);
        let [a, b, c] = [3, 4, 5].map(|k| two_sided.vertices[two_sided.indices[k] as usize]);
        let face = (b.position() - a.position()).cross(c.position() - a.position());
        assert!(face.z < 0.0 && a.normal() == -Vec3::Z);
    }

    #[test]
    fn alpha_cutoffs_apply_to_texture_alpha() {
        let masked = helios_asset::Material { base_color_texture: Some(TextureRef::new(0)), alpha_mode: AlphaMode::Mask { cutoff: 0.3 }, ..Default::default() };
        assert_eq!(alpha_cutoff(&masked), Some(0.3));
        let faded = helios_asset::Material { base_color: glam::Vec4::new(1.0, 1.0, 1.0, 0.5), ..masked.clone() };
        assert_eq!(alpha_cutoff(&faded), Some(0.6));
        assert_eq!(alpha_cutoff(&helios_asset::Material { alpha_mode: AlphaMode::Blend, ..masked.clone() }), None);
        assert_eq!(alpha_cutoff(&helios_asset::Material { alpha_mode: AlphaMode::Opaque, ..masked.clone() }), None);

        let untextured = helios_asset::Material { base_color_texture: None, ..faded };
        assert!(!fully_cut_out(&masked) && !fully_cut_out(&untextured));
        assert!(fully_cut_out(&helios_asset::Material { base_color: glam::Vec4::new(1.0, 1.0, 1.0, 0.2), ..untextured }));
    }

    #[test]
    fn occlusion_shares_the_main_layer_when_it_can() {
        let orm = TextureRef::new(3);
        let base = helios_asset::Material {
            base_color_texture: Some(TextureRef::new(1)),
            metallic_roughness_texture: Some(orm),
            occlusion_texture: Some(orm),
            emissive_texture: Some(TextureRef::new(4)),
            ..Default::default()
        };
        // Same coordinates: the occlusion is the main layer's albedo alpha.
        let keys = LayerKeys::of(&base);
        assert!(keys.occlusion_in_main && keys.occlusion.is_none());
        assert_eq!(keys.main, Some([Some(1), None, Some(3), Some(3)]));
        assert_eq!(keys.emission, Some([Some(4), None, None, None]));

        // A second UV set (lightmap-style AO) or a blended material: a layer of its own.
        let second_set = helios_asset::Material { occlusion_texture: Some(TextureRef { tex_coord: 1, ..orm }), ..base.clone() };
        let blended = helios_asset::Material { alpha_mode: AlphaMode::Blend, ..base.clone() };
        for m in [second_set, blended] {
            let keys = LayerKeys::of(&m);
            assert!(!keys.occlusion_in_main);
            assert_eq!(keys.main, Some([Some(1), None, Some(3), None]));
            assert_eq!(keys.occlusion, Some([None, None, None, Some(3)]));
        }

        // Occlusion alone: no main layer.
        let only = helios_asset::Material { occlusion_texture: Some(orm), ..Default::default() };
        let keys = LayerKeys::of(&only);
        assert_eq!((keys.main, keys.occlusion), (None, Some([None, None, None, Some(3)])));
    }

    #[test]
    fn texture_references_become_uv_transforms() {
        let t = TextureRef {
            tex_coord: 1,
            transform: helios_asset::TextureTransform { offset: glam::Vec2::new(0.5, 0.0), rotation: 0.0, scale: glam::Vec2::splat(2.0) },
            wrap: [WrapMode::ClampToEdge, WrapMode::MirroredRepeat],
            ..TextureRef::new(0)
        };
        let uv = uv_transform(Some(&t));
        assert_eq!((uv.set, uv.wrap), (1, [Wrap::ClampToEdge, Wrap::MirroredRepeat]));
        assert_eq!(uv.apply(glam::Vec2::ZERO, glam::Vec2::new(0.25, 1.0)), glam::Vec2::new(1.0, 2.0));
        assert_eq!(uv_transform(None), UvTransform::IDENTITY);
    }

    #[test]
    fn camera_sees_the_model() {
        let camera = frame_camera(Vec3::new(4.0, 2.0, 3.0));
        let to_center = (Vec3::new(0.0, 0.8, 0.0) - camera.position).normalize();
        assert!(camera.forward().dot(to_center) > 0.999);
        assert!(camera.position.length() > 2.5);
    }
}
