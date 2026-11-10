//! CPU-side scene description. The renderer mirrors it to the GPU, re-uploading only what
//! changed (tracked with version counters).

use std::sync::Arc;

use glam::{Affine2, Mat4, Vec2, Vec3};
use helios_geometry::VirtualMesh;

use crate::page_source::{PageLocation, PageSource};
use crate::skin::{Skin, SkinHandle};
use crate::wind::{Wind, WindResponse};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub(crate) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialHandle(pub(crate) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceHandle(pub(crate) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub(crate) u32);

/// Shading model, selecting the material code path (see `shaders/material.wgsl`).
///
/// Opaque models are rasterised into the visibility buffer; pixels are classified by shading
/// model and each model is resolved by its own specialised pipeline. Translucent models
/// ([`MaterialKind::is_translucent`]) never enter the visibility buffer, the HZB or the main
/// view's culling lists: they are drawn by the forward translucency pass after the deferred
/// lighting (see `translucency.rs`). Whether they cast shadows or contribute to Lumen is up to
/// the instance ([`Scene::set_cast_shadow`]) and the mesh (a distance field or not).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum MaterialKind {
    /// Textured (or constant) metal/roughness PBR.
    #[default]
    Standard = 0,
    /// Procedural rock with optional triplanar texture and moss on upward faces.
    Rock = 1,
    /// Landscape: texture layers blended by slope/height, cached in a runtime virtual texture.
    Terrain = 2,
    /// Brushed / hammered metal.
    Metal = 3,
    /// Two-sided foliage with light transmission.
    Foliage = 4,
    /// Translucent, refracting glass tinted by the base colour × texture. A stained-glass
    /// texture's roughness channel (< 0.2) separates glass from the opaque lead came, and
    /// textured glass diffuses most of the light it transmits; untextured glass is a clear pane,
    /// its view through blurred by the roughness. Its mesh distance field still feeds Lumen,
    /// where the pane emits the daylight it transmits (that is how stained glass colours an
    /// interior).
    Glass = 5,
    /// Translucent water ([`Scene::water`] sets its look): animated waves, refraction,
    /// depth-based absorption and scattering, shoreline foam, reflections. The base colour
    /// tints the scattered light, roughness is the surface roughness.
    Water = 8,
    /// Alpha-blended metal/roughness PBR (glTF `BLEND`): coverage is [`Material::opacity`] ×
    /// the base-colour texture's alpha.
    Translucent = 9,
}

impl MaterialKind {
    /// Rendered by the forward translucency pass instead of the visibility buffer.
    pub fn is_translucent(self) -> bool {
        matches!(self, Self::Glass | Self::Water | Self::Translucent)
    }
}

/// How a material maps its texture onto geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum TextureMapping {
    /// The mesh's UV coordinates.
    #[default]
    MeshUv = 0,
    /// Triplanar projection in world space (no UVs needed, continuous across instances).
    WorldTriplanar = 1,
    /// Triplanar projection in object space (moves with the instance).
    ObjectTriplanar = 2,
}

/// A sampler's wrap mode along one texture axis, applied to the coordinates in the shader (the
/// texture arrays and tile atlases are sampled with repeat addressing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Wrap {
    #[default]
    Repeat = 0,
    MirroredRepeat = 1,
    ClampToEdge = 2,
}

/// How a texture's coordinates follow from the mesh's (glTF texture info and
/// `KHR_texture_transform`): pick a UV set, transform it, wrap. Applies to
/// [`TextureMapping::MeshUv`] only; the material's `uv_scale` tiles the result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvTransform {
    /// Mesh UV set: 0 or 1 ([`helios_geometry::Mesh::uv1`]; meshes without one use set 0).
    pub set: u32,
    pub matrix: Affine2,
    /// Along u and v.
    pub wrap: [Wrap; 2],
}

impl UvTransform {
    /// The first UV set, untransformed, repeating.
    pub const IDENTITY: Self = Self { set: 0, matrix: Affine2::IDENTITY, wrap: [Wrap::Repeat; 2] };

    /// Texture coordinates of a mesh point with UVs `uv` (set 0) and `uv1` (set 1), before
    /// wrapping (sampling repeats; [`Wrap`] is applied by the shaders).
    pub fn apply(&self, uv: Vec2, uv1: Vec2) -> Vec2 {
        self.matrix.transform_point2(if self.set == 1 { uv1 } else { uv })
    }
}

impl Default for UvTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Procedural texture generator (evaluated on the GPU into the material texture arrays).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TexturePattern {
    /// Cut stone blocks in running bond with mortar.
    Ashlar = 0,
    /// Large floor slabs.
    Flagstone = 1,
    /// Checkerboard marble tiles with veins.
    MarbleTiles = 2,
    /// Wooden planks with grain.
    Wood = 3,
    /// Overlapping roof slates.
    Slate = 4,
    /// Lead-came stained glass (roughness channel < 0.2 marks glass).
    StainedGlass = 5,
    /// Woven fabric with border bands.
    Fabric = 6,
    /// Furrowed tree bark (`param` = moss amount).
    Bark = 7,
    /// Leaf blade with veins (UV: midrib along v).
    Leaf = 8,
    Moss = 9,
    /// Leaf litter, soil and twigs.
    ForestFloor = 10,
    Grass = 11,
    /// Soil with pebbles.
    Dirt = 12,
    /// Ridged, cracked rock.
    Rock = 13,
    Snow = 14,
    /// Hammered metal dents.
    Hammered = 15,
    Plaster = 16,
    /// Ashlar with carved reliefs.
    CarvedStone = 17,
    /// Rounded setts in sandy joints (`param`: joint width); its height suits displacement.
    Cobblestone = 18,
}

/// A procedural texture recipe.
#[derive(Clone, Copy, Debug)]
pub struct TextureDesc {
    pub pattern: TexturePattern,
    /// Primary and secondary colour (linear).
    pub color_a: Vec3,
    pub color_b: Vec3,
    /// Roughness associated with `color_a` / `color_b`.
    pub roughness: [f32; 2],
    /// Pattern density multiplier.
    pub scale: f32,
    /// Height-to-normal strength.
    pub bump: f32,
    /// Pattern-specific parameter.
    pub param: f32,
    pub seed: u32,
}

impl TextureDesc {
    pub fn new(pattern: TexturePattern, color_a: Vec3, color_b: Vec3) -> Self {
        Self { pattern, color_a, color_b, roughness: [0.7, 0.8], scale: 1.0, bump: 1.0, param: 0.5, seed: 1 }
    }

    pub fn roughness(mut self, a: f32, b: f32) -> Self {
        self.roughness = [a, b];
        self
    }

    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn bump(mut self, bump: f32) -> Self {
        self.bump = bump;
        self
    }

    pub fn param(mut self, param: f32) -> Self {
        self.param = param;
        self
    }

    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }
}

/// A decoded 8-bit RGBA image, rows from top to bottom (texel (0, 0) is at UV (0, 0), the
/// glTF convention). Any size: it is resampled to the texture layer size on the GPU.
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes.
    pub pixels: Vec<u8>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Image({}x{})", self.width, self.height)
    }
}

/// Authored (e.g. glTF) images packed into one texture layer. Missing images take neutral
/// values: white base colour, flat normal, roughness and metallic 1 (so the material's factors
/// apply unchanged, as in glTF).
///
/// A material using such a layer follows glTF metallic-roughness semantics: texture roughness
/// and metallic *multiply* [`Material::roughness`] / [`Material::metallic`].
///
/// The layer's channels: albedo = (base colour rgb, alpha or occlusion), data = (normal xy,
/// roughness, metallic). Ambient occlusion takes the albedo alpha because opaque materials
/// never read it (alpha tests use their own mask array built from the base colour image): it
/// costs no memory, no extra atlas and no extra texture fetch — the BC7 atlas and the sRGB
/// array keep alpha linear and at full precision.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImageTexture {
    /// sRGB colour. Alpha is the coverage tested by alpha-masked materials
    /// ([`Material::alpha_cutoff`]); it does not affect shading.
    pub base_color: Option<Arc<Image>>,
    /// Tangent-space normal map, OpenGL convention (+Y towards the top of the image).
    pub normal: Option<Arc<Image>>,
    /// Linear; roughness in G, metallic in B (glTF `metallicRoughnessTexture`).
    pub metallic_roughness: Option<Arc<Image>>,
    /// Linear ambient occlusion in R (glTF `occlusionTexture`; often the metallic-roughness
    /// image itself). Replaces the base colour's alpha in the layer, so the layer must not be
    /// the coverage of a [`MaterialKind::Translucent`] material.
    pub occlusion: Option<Arc<Image>>,
}

/// The content of one material texture layer.
#[derive(Clone, Debug)]
pub enum TextureSource {
    /// Synthesised on the GPU from a recipe.
    Procedural(TextureDesc),
    /// Resampled from images.
    Images(ImageTexture),
}

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub kind: MaterialKind,
    /// Linear base colour (albedo, or F0 for metals); multiplies the texture.
    pub base_color: Vec3,
    /// Coverage of [`MaterialKind::Translucent`] surfaces (glTF base colour alpha); 1 = opaque.
    pub opacity: f32,
    pub roughness: f32,
    pub metallic: f32,
    /// Emitted radiance as a multiple of the base colour.
    pub emissive: f32,
    /// Emitted linear radiance independent of the base colour (glTF `emissiveFactor`),
    /// multiplied by `emissive_texture` if present. Standard and Foliage models only.
    pub emissive_color: Vec3,
    /// Emission colour texture, sampled with the mesh UVs (an image layer's base colour).
    pub emissive_texture: Option<TextureHandle>,
    /// Texture coordinates of `emissive_texture`.
    pub emissive_uv: UvTransform,
    /// Ambient occlusion: an image layer with an [`ImageTexture::occlusion`] (possibly
    /// `texture` itself). It darkens indirect light only — Lumen's diffuse and specular GI and
    /// the sky — never direct light, as UE's material ambient occlusion. Standard and Foliage
    /// models.
    pub occlusion_texture: Option<TextureHandle>,
    /// Texture coordinates of `occlusion_texture`.
    pub occlusion_uv: UvTransform,
    /// 0: no occlusion, 1: the texture's (glTF `strength`).
    pub occlusion_strength: f32,
    /// Frequency of procedural detail (per object-space unit).
    pub detail_scale: f32,
    /// Strength of procedural bump detail.
    pub detail_strength: f32,
    pub texture: Option<TextureHandle>,
    pub mapping: TextureMapping,
    /// Texture coordinates of `texture` with [`TextureMapping::MeshUv`] (base colour, normal,
    /// roughness and metallic, and the alpha test).
    pub uv: UvTransform,
    /// Texture repeats per UV unit (or per world unit for triplanar mapping).
    pub uv_scale: f32,
    pub normal_strength: f32,
    /// Terrain: grass, dirt, rock and snow layer textures.
    pub layers: [Option<TextureHandle>; 4],
    /// Alpha test (glTF `alphaMode: MASK`): surfaces are cut out where the alpha of the
    /// texture's base colour image is below this value, in the visibility buffer and in shadow
    /// maps. Needs an image texture ([`ImageTexture::base_color`]) mapped with
    /// [`TextureMapping::MeshUv`]; otherwise the material renders opaque. `None`: opaque.
    pub alpha_cutoff: Option<f32>,
    /// How surfaces move in [`Scene::wind`] (World Position Offset); static by default.
    pub wind: WindResponse,
    /// The Substrate slab: clear coat, sheen, subsurface, transmission, IOR / specular colour,
    /// anisotropy, unlit (opaque models; none by default).
    pub slab: crate::substrate::Slab,
    /// A material graph that evaluates the surface instead of the shading model's code (opaque
    /// models); its parameters are overridden per material with
    /// [`Scene::set_material_parameter`].
    pub graph: Option<crate::material_graphs::MaterialGraphHandle>,
    /// Moves the surface along its normal at render time (Nanite tessellation): opaque materials
    /// only. `None`: undisplaced.
    pub displacement: Option<Displacement>,
}

/// Displacement of a material's surface along its normal, applied at render time by Nanite
/// tessellation (UE 5.4+, see `tessellation.rs`): triangles of the meshes using it are diced into
/// micro-triangles about [`crate::RenderSettings::dice_rate`] pixels long and every vertex moves by
/// `magnitude * (height - center)` object-space units. The height is a channel of a texture
/// layer, sampled with the material's [`TextureMapping`] (mesh UVs through [`Material::uv`], or
/// triplanar). Its bound, `magnitude * max(center, 1 - center)`, pads the culling bounds, so a
/// tight `center` and `magnitude` keep culling efficient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Displacement {
    /// The layer holding the height; `None`: the material's own [`Material::texture`].
    pub texture: Option<TextureHandle>,
    /// Channel of the layer's data texel: 3 is a procedural texture's height ([`TexturePattern`]);
    /// image layers hold normal x, y, roughness and metallic there (0..=3).
    pub channel: u32,
    /// Object-space distance between heights 0 and 1.
    pub magnitude: f32,
    /// Height that stays in place (0: displacement outwards only, 0.5: both ways).
    pub center: f32,
    /// Mesh UV units per object-space unit of the meshes using it, for [`TextureMapping::MeshUv`]:
    /// picks the height's mip for the size of a dice segment (UE uses the mesh's UV density the
    /// same way). [`Displacement::uv_density_of`] measures it.
    pub uv_density: f32,
}

impl Displacement {
    /// Displacement by the height of the material's own (procedural) texture.
    pub fn height(magnitude: f32, center: f32) -> Self {
        Self { texture: None, channel: 3, magnitude, center, uv_density: 1.0 }
    }

    /// Largest object-space distance a point moves (culling bounds grow by it).
    pub fn bound(&self) -> f32 {
        self.magnitude.abs() * self.center.max(1.0 - self.center)
    }

    /// Area-weighted mean UV units per object-space unit of `mesh`'s triangles (the square root of
    /// the UV area over the surface area).
    pub fn uv_density_of(mesh: &helios_geometry::Mesh) -> f32 {
        let (mut uv_area, mut area) = (0.0f64, 0.0f64);
        for t in mesh.indices.chunks_exact(3) {
            let v = [t[0], t[1], t[2]].map(|i| &mesh.vertices[i as usize]);
            let p = v.map(|v| Vec3::from(v.position));
            let uv = v.map(|v| Vec2::new(v.u, v.v));
            area += (p[1] - p[0]).cross(p[2] - p[0]).length() as f64;
            uv_area += (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs() as f64;
        }
        if area > 0.0 {
            (uv_area / area).sqrt() as f32
        } else {
            1.0
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            kind: MaterialKind::Standard,
            base_color: Vec3::splat(0.5),
            opacity: 1.0,
            roughness: 0.5,
            metallic: 0.0,
            emissive: 0.0,
            emissive_color: Vec3::ZERO,
            emissive_texture: None,
            emissive_uv: UvTransform::IDENTITY,
            occlusion_texture: None,
            occlusion_uv: UvTransform::IDENTITY,
            occlusion_strength: 1.0,
            detail_scale: 1.0,
            detail_strength: 0.0,
            texture: None,
            mapping: TextureMapping::MeshUv,
            uv: UvTransform::IDENTITY,
            uv_scale: 1.0,
            normal_strength: 1.0,
            layers: [None; 4],
            alpha_cutoff: None,
            wind: WindResponse::NONE,
            slab: Default::default(),
            graph: None,
            displacement: None,
        }
    }
}

impl Material {
    pub fn standard(base_color: Vec3, roughness: f32, metallic: f32) -> Self {
        Self { base_color, roughness, metallic, ..Default::default() }
    }

    /// A textured material (white base colour, so the texture shows as authored).
    pub fn textured(kind: MaterialKind, texture: TextureHandle, mapping: TextureMapping, uv_scale: f32) -> Self {
        Self { kind, base_color: Vec3::ONE, texture: Some(texture), mapping, uv_scale, ..Default::default() }
    }
}

/// Directional sun light. Its colour is filtered by the simulated atmosphere.
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    /// Unit vector pointing *towards* the sun.
    pub direction: Vec3,
    /// Colour × intensity of sunlight outside the atmosphere.
    pub illuminance: Vec3,
    /// Angular radius of the solar disk in radians.
    pub angular_radius: f32,
}

impl Sun {
    /// Sun from azimuth (around +Y, from +X towards +Z) and elevation above the horizon, in radians.
    pub fn from_angles(azimuth: f32, elevation: f32, intensity: f32) -> Self {
        let (se, ce) = elevation.sin_cos();
        let (sa, ca) = azimuth.sin_cos();
        Self {
            direction: Vec3::new(ce * ca, se, ce * sa).normalize(),
            illuminance: Vec3::new(1.0, 0.98, 0.95) * intensity,
            angular_radius: 0.5f32.to_radians(),
        }
    }
}

impl Sun {
    /// The full moon as the directional light: sunlight reflected by the moon, bluish to the
    /// dark-adapted eye. `sun_intensity` is the daylight intensity of the scene; moonlight is
    /// brightened well above its physical 1/400 000 so night scenes stay readable after
    /// exposure ("day for night", as films do).
    pub fn moon(azimuth: f32, elevation: f32, sun_intensity: f32) -> Self {
        let mut moon = Self::from_angles(azimuth, elevation, sun_intensity * MOONLIGHT_RATIO);
        moon.illuminance *= Vec3::new(0.72, 0.82, 1.0);
        moon.angular_radius = 0.26f32.to_radians();
        moon
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Moonlight relative to sunlight (artistic; see [`Sun::moon`]).
const MOONLIGHT_RATIO: f32 = 1.0e-3;
/// Brightest star radiance relative to the sun intensity.
const STAR_RATIO: f32 = 2.0e-3;

impl Default for Sun {
    fn default() -> Self {
        Self::from_angles(0.6, 0.45, 10.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LightHandle(pub(crate) u32);

/// Shape of a local light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightKind {
    Point,
    /// Cone with full intensity inside `inner` and none outside `outer` (half-angles, radians).
    Spot { direction: Vec3, inner: f32, outer: f32 },
}

/// A local (punctual) light. Lights are culled into a clustered (froxel) grid, shadowed by
/// cone-tracing the global distance field and injected into Lumen's surface cache, so they
/// also bounce.
#[derive(Clone, Copy, Debug)]
pub struct Light {
    pub kind: LightKind,
    pub position: Vec3,
    /// Linear colour.
    pub color: Vec3,
    /// Luminous intensity in candela-like units (radiance scale at 1 m).
    pub intensity: f32,
    /// Influence radius; attenuation reaches zero here.
    pub range: f32,
    /// Size of the emitter, for soft shadows.
    pub source_radius: f32,
    pub cast_shadows: bool,
}

impl Light {
    pub fn point(position: Vec3, color: Vec3, intensity: f32, range: f32) -> Self {
        Self { kind: LightKind::Point, position, color, intensity, range, source_radius: 0.05, cast_shadows: true }
    }

    pub fn spot(position: Vec3, direction: Vec3, color: Vec3, intensity: f32, range: f32, outer: f32) -> Self {
        Self {
            kind: LightKind::Spot { direction: direction.normalize(), inner: outer * 0.7, outer },
            position,
            color,
            intensity,
            range,
            source_radius: 0.05,
            cast_shadows: true,
        }
    }
}

/// Whether an instance may move, as UE's component mobility.
///
/// Every instance can be moved with [`Scene::set_transform`]; the renderer always updates only
/// what the move touched (the instance's GPU record, the shadow pages and distance-field bricks
/// under its old and new bounds, its mesh cards). Mobility decides *how* the cached shadows treat
/// it: a [`Mobility::Static`] instance is baked into the static layer of the shadow pages it
/// covers, so moving it re-renders those pages in full (a one-off edit); a [`Mobility::Movable`]
/// instance is a *dynamic caster* that stays out of the static layers, so its moves only redraw
/// the dynamic casters of the pages under it, over their cached static layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mobility {
    /// Placed once (it may still be moved occasionally, e.g. by an editor).
    #[default]
    Static,
    /// Moves often (animated props, physics, vehicles).
    Movable,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Instance {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub transform: Mat4,
    pub cast_shadow: bool,
    /// Hidden instances (deleted in an editor) stay in the list, so handles never shift, but
    /// every pass skips them (see `scene_edit.rs`).
    pub visible: bool,
    /// The joint palette deforming a skinned mesh ([`Scene::set_instance_skin`]).
    pub skin: Option<SkinHandle>,
    pub mobility: Mobility,
    /// Scene version of the last change of `transform` or `mobility` (see
    /// [`Scene::changed_instances`]).
    pub version: u64,
}

/// A layer of volumetric cumulus clouds (rendered by `clouds.rs`).
#[derive(Clone, Copy, Debug)]
pub struct CloudLayer {
    /// Fraction of the sky the clouds may cover, 0 (none: the pass is skipped) to 1 (overcast).
    pub coverage: f32,
    /// Extinction per kilometre inside a fully dense cloud.
    pub density: f32,
    /// Height of the cloud base above the ground, in metres.
    pub base_altitude: f32,
    /// Thickness of the cloud layer, in metres.
    pub thickness: f32,
    /// Wind velocity (x, z) in metres per second.
    pub wind: glam::Vec2,
    /// How dark cloud shadows on the ground get (0 = no cloud shadows, 1 = physical).
    pub shadow_strength: f32,
}

impl Default for CloudLayer {
    /// No clouds.
    fn default() -> Self {
        Self { coverage: 0.0, ..Self::cumulus(0.0) }
    }
}

impl CloudLayer {
    /// Fair-weather cumulus with the given coverage.
    pub fn cumulus(coverage: f32) -> Self {
        Self {
            coverage,
            density: 20.0,
            base_altitude: 1500.0,
            thickness: 1800.0,
            wind: glam::Vec2::new(14.0, 5.0),
            shadow_strength: 0.8,
        }
    }
}

/// Height fog rendered as a lit participating medium (rendered by `fog.rs`): light shafts,
/// glowing haze around lights, depth. `density` 0 disables it.
#[derive(Clone, Copy, Debug)]
pub struct FogVolume {
    /// Extinction per metre at `base_height` (0.002 is a light haze, 0.03 a thick fog).
    pub density: f32,
    /// Exponential falloff of the density per metre above `base_height`.
    pub height_falloff: f32,
    pub base_height: f32,
    /// Scattering colour (1 = white, non-absorbing).
    pub albedo: Vec3,
    /// Henyey-Greenstein anisotropy: 0 isotropic, towards 1 strongly forward scattering
    /// (bright towards the sun, visible shafts).
    pub anisotropy: f32,
    /// Scale of the ambient (bounce / sky) light scattered by the fog.
    pub ambient: f32,
    /// 0 for uniform fog, up to 1 for strongly varying banks and wisps.
    pub noise: f32,
    /// Drift of the noise in metres per second.
    pub wind: Vec3,
    /// How far the fog volume reaches (metres); beyond it only aerial perspective remains.
    pub distance: f32,
}

impl Default for FogVolume {
    /// No volumetric fog.
    fn default() -> Self {
        Self {
            density: 0.0,
            height_falloff: 0.05,
            base_height: 0.0,
            albedo: Vec3::ONE,
            anisotropy: 0.6,
            ambient: 1.0,
            noise: 0.5,
            wind: Vec3::new(0.6, 0.0, 0.25),
            distance: 160.0,
        }
    }
}

/// The look of the scene's water surfaces ([`MaterialKind::Water`]) and of what lies under them.
/// Light under the surface follows a single-scattering medium: the bottom fades with
/// `exp(-(absorption + scattering) · distance)` and is replaced by sun and sky light scattered
/// back out of the water. Every water instance is a flat water body: the landscape under it turns
/// into a wet lake bed, the sun casts caustics on it, and a camera below the surface sees the
/// underwater view.
#[derive(Clone, Copy, Debug)]
pub struct Water {
    /// Absorption per metre (Beer–Lambert). Red is absorbed fastest, which tints deep water.
    pub absorption: Vec3,
    /// Scattering per metre by suspended particles (0 = perfectly clear water).
    pub scattering: Vec3,
    /// Wavelength of the longest wave in metres; shorter ones follow geometrically.
    pub wavelength: f32,
    /// Wave steepness: 0 is a mirror, 0.5 a choppy surface.
    pub steepness: f32,
    /// Direction the waves travel in (x, z).
    pub wind_direction: glam::Vec2,
    /// Amount of foam along shores and around anything breaking the surface (0 = none).
    pub foam: f32,
    /// Strength of the sun's caustics on the bed: 1 is the light focused by the waves, 0 none.
    pub caustics: f32,
}

impl Default for Water {
    /// A clear mountain lake: close to pure water's absorption (red ≈ 0.4 /m, green ≈ 0.05 /m,
    /// blue a little above pure water's ≈ 0.02 /m from traces of dissolved organic matter) and
    /// light scattering by fine particles: about 15 m of visibility, blue-green in the deep. More
    /// scattering than this washes the bed out into a uniform teal haze in shade, where the
    /// in-scattered sky light rivals the light reflected by the bed.
    fn default() -> Self {
        Self {
            absorption: Vec3::new(0.38, 0.05, 0.035),
            scattering: Vec3::new(0.015, 0.018, 0.02),
            wavelength: 3.0,
            steepness: 0.22,
            wind_direction: glam::Vec2::new(1.0, 0.35),
            foam: 1.0,
            caustics: 1.0,
        }
    }
}

/// A part of an assembly: a mesh placed in the assembly's space (see [`Scene::set_assembly`]).
#[derive(Clone, Copy, Debug)]
pub struct AssemblyPart {
    pub mesh: MeshHandle,
    pub transform: Mat4,
}

#[derive(Default)]
pub struct Scene {
    pub(crate) meshes: Vec<VirtualMesh>,
    /// Meshes that are assembly bases, with their parts.
    pub(crate) assemblies: Vec<(MeshHandle, Vec<AssemblyPart>)>,
    /// Where each mesh's non-resident page payloads stream from (`None`: all in memory).
    pub(crate) page_locations: Vec<Option<PageLocation>>,
    pub(crate) materials: Vec<Material>,
    pub(crate) textures: Vec<TextureSource>,
    pub(crate) lights: Vec<Light>,
    pub(crate) instances: Vec<Instance>,
    /// Joint palettes of skinned instances (`skin.rs`).
    pub(crate) skins: Vec<Skin>,
    pub sun: Sun,
    /// Average albedo of the ground, used for atmospheric multiple scattering and sky bounce.
    pub ground_albedo: Vec3,
    pub clouds: CloudLayer,
    /// Radiance of the brightest stars of the night sky (0 = no stars; see
    /// [`Scene::set_time_of_day`]).
    pub stars: f32,
    pub fog: FogVolume,
    pub water: Water,
    /// Air flow that animates materials with a [`Material::wind`] response (calm by default).
    pub wind: Wind,
    /// Editor metadata: stable mesh and material names (see `scene_edit.rs`).
    pub(crate) names: crate::scene_edit::SceneNames,
    /// Heterogeneous volumes: simulated smoke and fire, procedural mist (`volumes.rs`).
    pub volumes: Vec<crate::volumes::VolumeEmitter>,
    /// GPU particle emitters (`particles.rs`).
    pub particles: Vec<crate::particles::ParticleEmitter>,
    /// Material graphs and material instances' parameters (`material_graphs.rs`).
    pub(crate) graphs: crate::material_graphs::GraphStore,
    pub(crate) geometry_version: u64,
    pub(crate) material_version: u64,
    /// Structural instance changes (instances added, shadow casting toggled): every consumer
    /// rebuilds its instance data. Moves don't change it ([`Scene::set_transform`]).
    pub(crate) instance_version: u64,
    pub(crate) texture_version: u64,
    /// The latest [`Instance::version`]: some instance moved or changed mobility after it.
    pub(crate) transform_version: u64,
}

/// Versions come from one process-wide counter, so two different scenes never share a version
/// and a renderer that switches scenes always notices the change.
pub(crate) fn next_version() -> u64 {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl Scene {
    pub fn new() -> Self {
        let version = next_version();
        Self {
            ground_albedo: Vec3::splat(0.3),
            geometry_version: version,
            material_version: version,
            instance_version: version,
            texture_version: version,
            transform_version: version,
            ..Default::default()
        }
    }

    /// Places the sun at `azimuth` / `elevation` (radians). Down to about 11° below the horizon
    /// the sun stays the directional light (the sky glows with twilight); further down the moon
    /// rises opposite it as at full moon, fading in, and stars come out.
    pub fn set_time_of_day(&mut self, azimuth: f32, elevation: f32, sun_intensity: f32) {
        const NIGHT_START: f32 = -0.19;
        const NIGHT_FULL: f32 = -0.35;
        if elevation > NIGHT_START {
            self.sun = Sun::from_angles(azimuth, elevation, sun_intensity);
            // Stars appear in the deepening twilight before the moon takes over.
            self.stars = sun_intensity * STAR_RATIO * smoothstep(-0.08, NIGHT_START, elevation) * 0.5;
            return;
        }
        let night = smoothstep(NIGHT_START, NIGHT_FULL, elevation);
        let moon_elevation = (-elevation * 0.9).clamp(0.15, 1.2);
        self.sun = Sun::moon(azimuth + std::f32::consts::PI, moon_elevation, sun_intensity * (0.02 + 0.98 * night));
        self.stars = sun_intensity * STAR_RATIO * (0.5 + 0.5 * night);
    }

    /// 0 in daylight, rising to 1 in full night (for exposure and lighting choices).
    pub fn night_factor(elevation: f32) -> f32 {
        smoothstep(0.0, -0.3, elevation)
    }

    pub fn add_mesh(&mut self, mesh: VirtualMesh) -> MeshHandle {
        self.push_mesh(mesh, None)
    }

    /// Adds a mesh whose page payloads are not (all) in memory, typically one loaded with
    /// [`VirtualMesh::from_header_bytes`] plus its root pages. The others are read on demand
    /// from file `file` of `source`, which holds the mesh's `.hvm`.
    pub fn add_streamed_mesh(&mut self, mesh: VirtualMesh, source: Arc<dyn PageSource>, file: u32) -> MeshHandle {
        self.push_mesh(mesh, Some(PageLocation { source, file }))
    }

    fn push_mesh(&mut self, mesh: VirtualMesh, location: Option<PageLocation>) -> MeshHandle {
        self.meshes.push(mesh);
        self.page_locations.push(location);
        self.geometry_version = next_version();
        MeshHandle(self.meshes.len() as u32 - 1)
    }

    pub fn add_light(&mut self, light: Light) -> LightHandle {
        self.lights.push(light);
        LightHandle(self.lights.len() as u32 - 1)
    }

    pub fn light_mut(&mut self, handle: LightHandle) -> &mut Light {
        &mut self.lights[handle.0 as usize]
    }

    pub fn lights(&self) -> &[Light] {
        &self.lights
    }

    /// All lights, e.g. to animate them (a light that moves re-renders its shadow map).
    pub fn lights_mut(&mut self) -> &mut [Light] {
        &mut self.lights
    }

    /// Adds a procedural texture (generated on the GPU into the material texture arrays).
    pub fn add_texture(&mut self, texture: TextureDesc) -> TextureHandle {
        self.add_texture_source(TextureSource::Procedural(texture))
    }

    /// Adds a texture layer made from images (resampled and packed on the GPU).
    pub fn add_image_texture(&mut self, texture: ImageTexture) -> TextureHandle {
        self.add_texture_source(TextureSource::Images(texture))
    }

    pub fn add_texture_source(&mut self, source: TextureSource) -> TextureHandle {
        self.textures.push(source);
        self.texture_version = next_version();
        TextureHandle(self.textures.len() as u32 - 1)
    }

    pub fn add_material(&mut self, material: Material) -> MaterialHandle {
        self.materials.push(material);
        self.material_version = next_version();
        MaterialHandle(self.materials.len() as u32 - 1)
    }

    pub fn set_material(&mut self, handle: MaterialHandle, material: Material) {
        self.materials[handle.0 as usize] = material;
        self.material_version = next_version();
    }

    /// Adds a [`Mobility::Static`] instance. Adding instances is a structural change: the
    /// renderer rebuilds its instance data (GPU records, distance-field clipmaps, mesh cards,
    /// cached shadow pages) once, unlike moves.
    pub fn add_instance(&mut self, mesh: MeshHandle, material: MaterialHandle, transform: Mat4) -> InstanceHandle {
        // Not a move: a new instance is part of the structural change.
        self.instances.push(Instance {
            mesh,
            material,
            transform,
            cast_shadow: true,
            visible: true,
            skin: None,
            mobility: Mobility::Static,
            version: 0,
        });
        self.instance_version = next_version();
        InstanceHandle(self.instances.len() as u32 - 1)
    }

    /// Excludes an instance from shadow casting (e.g. glass panes that should let light through,
    /// water surfaces).
    pub fn set_cast_shadow(&mut self, handle: InstanceHandle, cast_shadow: bool) {
        self.instances[handle.0 as usize].cast_shadow = cast_shadow;
        self.instance_version = next_version();
    }

    /// Moves an instance. Only what the move touches is updated (see [`Mobility`]); temporal
    /// history is kept and motion vectors follow the instance.
    pub fn set_transform(&mut self, handle: InstanceHandle, transform: Mat4) {
        let instance = &mut self.instances[handle.0 as usize];
        if instance.transform != transform {
            instance.transform = transform;
            self.touch(handle);
        }
    }

    pub fn transform(&self, handle: InstanceHandle) -> Mat4 {
        self.instances[handle.0 as usize].transform
    }

    /// Declares whether an instance moves often (see [`Mobility`]).
    pub fn set_mobility(&mut self, handle: InstanceHandle, mobility: Mobility) {
        let instance = &mut self.instances[handle.0 as usize];
        if instance.mobility != mobility {
            instance.mobility = mobility;
            self.touch(handle);
        }
    }

    pub fn mobility(&self, handle: InstanceHandle) -> Mobility {
        self.instances[handle.0 as usize].mobility
    }

    fn touch(&mut self, handle: InstanceHandle) {
        let version = next_version();
        self.instances[handle.0 as usize].version = version;
        self.transform_version = version;
    }

    /// Indices of the instances moved (or given another mobility) after scene version `since`,
    /// in index order (nothing is scanned unless some instance changed since).
    pub(crate) fn changed_instances(&self, since: u64) -> impl Iterator<Item = usize> + '_ {
        let any = self.transform_version > since;
        self.instances.iter().enumerate().filter(move |(_, i)| any && i.version > since).map(|(index, _)| index)
    }

    pub fn mesh(&self, handle: MeshHandle) -> &VirtualMesh {
        &self.meshes[handle.0 as usize]
    }

    /// Makes `base` an assembly (Nanite assemblies; see [`helios_geometry::assembly`]): its
    /// instances draw `parts` — meshes placed by transforms in `base`'s space, in the instance's
    /// material — where the base's finest level would be too coarse, and `base` (built with
    /// [`helios_geometry::assembly::build_assembly_base`] from those parts) beyond. The parts'
    /// transforms are resolved on the GPU every frame, so an assembly of many shared parts costs
    /// memory and baking for its distinct parts and its base only.
    pub fn set_assembly(&mut self, base: MeshHandle, parts: Vec<AssemblyPart>) {
        self.assemblies.retain(|(mesh, _)| *mesh != base);
        self.assemblies.push((base, parts));
        self.geometry_version = next_version();
    }

    /// CPU memory held by the meshes (records, in-memory payloads, distance fields).
    pub fn geometry_cpu_bytes(&self) -> usize {
        self.meshes.iter().map(VirtualMesh::cpu_bytes).sum()
    }

    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Sum of full-detail triangles over all instances (the "virtual" triangle count; an
    /// assembly's are its parts').
    pub fn source_triangle_count(&self) -> u64 {
        let triangles = |mesh: MeshHandle| match self.assemblies.iter().find(|(base, _)| *base == mesh) {
            Some((_, parts)) => parts.iter().map(|p| self.meshes[p.mesh.0 as usize].triangle_count() as u64).sum(),
            None => self.meshes[mesh.0 as usize].triangle_count() as u64,
        };
        self.instances.iter().map(|i| triangles(i.mesh)).sum()
    }
}
