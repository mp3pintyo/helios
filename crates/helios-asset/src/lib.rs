//! # helios-asset
//!
//! Imports glTF 2.0 content into a renderer-neutral [`Asset`]:
//!
//! * **meshes** — one [`helios_geometry::Mesh`] per glTF primitive (positions, normals, both UV
//!   sets, vertex colours; flat normals are generated when a primitive has none, as the
//!   specification requires). Attributes may use any component type of glTF core or
//!   `KHR_mesh_quantization` (8/16-bit integers, normalized or not),
//! * **instances** — `(mesh, material, world transform)` for every primitive reached through
//!   the node hierarchy. Transforms are *not* baked into vertices, so a mesh that the file
//!   instances many times is baked into virtual geometry once,
//! * **materials** — the metallic-roughness PBR model (factors; texture references with their
//!   UV set, `KHR_texture_transform` and sampler wrap modes; normal scale, occlusion strength,
//!   emission with `KHR_materials_emissive_strength`, alpha mode),
//! * **images** — decoded RGBA8 (PNG and JPEG, the formats glTF core allows).
//!
//! Loading has two phases so it works the same natively and in the browser, where external
//! files can only be fetched asynchronously: [`GltfDocument::parse`] reads the JSON (or `.glb`
//! container) and lists the [external URIs](GltfDocument::external_uris) it needs; the caller
//! fetches them however it can and passes the bytes to [`GltfDocument::import`].
//! [`load_gltf`] handles self-contained files (`.glb`, embedded data URIs) in one call and
//! [`load_gltf_file`] (native only) resolves external files relative to the document.

mod accessor;
mod import;
mod uri;

use std::collections::HashMap;
use std::fmt;

use glam::{Affine2, Mat4, Vec2, Vec3, Vec4};
use helios_geometry::{Aabb, Mesh};

/// Decoded 8-bit RGBA image, rows from top to bottom (glTF's UV origin is the top-left
/// corner, so UVs index it directly).
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes. Colour images are sRGB-encoded, data images linear.
    pub pixels: Vec<u8>,
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Image({}x{})", self.width, self.height)
    }
}

/// How a material's base-colour alpha is interpreted.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AlphaMode {
    #[default]
    Opaque,
    /// Fully transparent below `cutoff`, opaque above (foliage cards, chains, curtains).
    Mask { cutoff: f32 },
    /// Alpha blending.
    Blend,
}

/// `KHR_texture_transform`: texture coordinates are scaled, then rotated, then offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureTransform {
    pub offset: Vec2,
    /// Radians: the UVs turn counter-clockwise as seen on the image (whose v axis points down),
    /// so the image turns clockwise on the surface.
    pub rotation: f32,
    pub scale: Vec2,
}

impl TextureTransform {
    pub const IDENTITY: Self = Self { offset: Vec2::ZERO, rotation: 0.0, scale: Vec2::ONE };

    /// The affine map `uv -> T * R * S * uv` of the extension, with the rotation matrix rows
    /// `(cos, sin)` and `(-sin, cos)` as implementations (three.js, the glTF Sample Viewer) and
    /// the TextureTransformTest sample apply it. (The specification's GLSL, read column-major,
    /// would turn the other way: the sample marks that direction with a red marker.)
    pub fn matrix(&self) -> Affine2 {
        Affine2::from_scale_angle_translation(self.scale, -self.rotation, self.offset)
    }
}

impl Default for TextureTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A sampler's wrap mode along one texture axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WrapMode {
    #[default]
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

/// A material's use of a texture: which image, through which coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureRef {
    /// Index into [`Asset::images`].
    pub image: usize,
    /// `TEXCOORD_n` set: 0 or 1 (the transform's `texCoord` override applied; higher sets are
    /// not imported and fall back to 0).
    pub tex_coord: u32,
    pub transform: TextureTransform,
    /// The sampler's wrap modes along u and v.
    pub wrap: [WrapMode; 2],
}

impl TextureRef {
    /// Image `image` with the first UV set, untransformed and repeating.
    pub fn new(image: usize) -> Self {
        Self { image, tex_coord: 0, transform: TextureTransform::IDENTITY, wrap: [WrapMode::Repeat; 2] }
    }

    /// True if both sample with the same coordinates (so their images can share a layer).
    pub fn same_coordinates(&self, other: &TextureRef) -> bool {
        (self.tex_coord, self.transform, self.wrap) == (other.tex_coord, other.transform, other.wrap)
    }
}

/// A metallic-roughness PBR material. Texture values multiply the corresponding factors.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub name: String,
    /// Linear RGBA.
    pub base_color: Vec4,
    /// sRGB colour, alpha in A.
    pub base_color_texture: Option<TextureRef>,
    pub metallic: f32,
    pub roughness: f32,
    /// Linear: roughness in G, metallic in B.
    pub metallic_roughness_texture: Option<TextureRef>,
    /// Tangent-space normals, OpenGL convention (+Y towards the top of the image).
    pub normal_texture: Option<TextureRef>,
    pub normal_scale: f32,
    /// Linear ambient occlusion in R (often the R of the metallic-roughness image: "ORM").
    pub occlusion_texture: Option<TextureRef>,
    /// 0: no occlusion, 1: the texture's full occlusion.
    pub occlusion_strength: f32,
    /// Linear emitted radiance (`KHR_materials_emissive_strength` already applied).
    pub emissive: Vec3,
    /// sRGB.
    pub emissive_texture: Option<TextureRef>,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
}

impl Default for Material {
    /// The glTF default material (used by primitives without one).
    fn default() -> Self {
        Self {
            name: "default".into(),
            base_color: Vec4::ONE,
            base_color_texture: None,
            metallic: 1.0,
            roughness: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive: Vec3::ZERO,
            emissive_texture: None,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
        }
    }
}

/// A unique mesh (one glTF primitive, possibly with reversed winding for mirrored instances).
#[derive(Clone, Debug)]
pub struct NamedMesh {
    pub name: String,
    pub mesh: Mesh,
}

/// One placement of a mesh in the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// Index into [`Asset::meshes`].
    pub mesh: usize,
    /// Index into [`Asset::materials`].
    pub material: usize,
    /// Object to world, the product of the node hierarchy's transforms.
    pub transform: Mat4,
}

/// An imported scene.
#[derive(Clone, Debug, Default)]
pub struct Asset {
    pub meshes: Vec<NamedMesh>,
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
    pub instances: Vec<Instance>,
}

impl Asset {
    /// World-space bounds of all instances (transformed mesh boxes).
    pub fn bounds(&self) -> Aabb {
        let boxes: Vec<Aabb> = self.meshes.iter().map(|m| m.mesh.aabb()).collect();
        self.instances.iter().fold(Aabb::EMPTY, |acc, instance| {
            let b = boxes[instance.mesh];
            let corners = (0..8).map(|i| {
                let pick = |bit: usize, lo: f32, hi: f32| if i & bit != 0 { hi } else { lo };
                instance.transform.transform_point3(Vec3::new(pick(1, b.min.x, b.max.x), pick(2, b.min.y, b.max.y), pick(4, b.min.z, b.max.z)))
            });
            let t = Aabb::from_points(corners);
            Aabb { min: acc.min.min(t.min), max: acc.max.max(t.max) }
        })
    }

    pub fn triangle_count(&self) -> usize {
        self.instances.iter().map(|i| self.meshes[i.mesh].mesh.triangle_count()).sum()
    }
}

#[derive(Debug)]
pub enum AssetError {
    /// Malformed or invalid glTF, or one that requires an unsupported extension.
    Gltf(gltf::Error),
    /// An external buffer or image the caller did not supply.
    MissingResource(String),
    /// A resource that could not be read or decoded.
    Resource { uri: String, message: String },
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetError::Gltf(e) => write!(f, "invalid glTF: {e}"),
            AssetError::MissingResource(uri) => write!(f, "missing external resource '{uri}'"),
            AssetError::Resource { uri, message } => write!(f, "cannot load '{uri}': {message}"),
        }
    }
}

impl std::error::Error for AssetError {}

impl From<gltf::Error> for AssetError {
    fn from(e: gltf::Error) -> Self {
        AssetError::Gltf(e)
    }
}

/// A parsed glTF document (`.gltf` JSON or `.glb`) whose external resources are not loaded yet.
pub struct GltfDocument {
    document: gltf::Document,
    blob: Option<Vec<u8>>,
}

impl GltfDocument {
    pub fn parse(bytes: &[u8]) -> Result<Self, AssetError> {
        let gltf = gltf::Gltf::from_slice_without_validation(bytes)?;
        import::validate(&gltf.document)?;
        Ok(Self { document: gltf.document, blob: gltf.blob })
    }

    /// External buffer and image URIs (as written in the document, i.e. relative and
    /// percent-encoded; data URIs excluded), each once, buffers first.
    pub fn external_uris(&self) -> Vec<String> {
        let buffers = self.document.buffers().filter_map(|b| match b.source() {
            gltf::buffer::Source::Uri(uri) => Some(uri),
            gltf::buffer::Source::Bin => None,
        });
        let images = self.document.images().filter_map(|i| match i.source() {
            gltf::image::Source::Uri { uri, .. } => Some(uri),
            gltf::image::Source::View { .. } => None,
        });
        let mut uris: Vec<String> = Vec::new();
        for uri in buffers.chain(images).filter(|u| !uri::is_data_uri(u)) {
            if !uris.iter().any(|u| u == uri) {
                uris.push(uri.to_string());
            }
        }
        uris
    }

    /// Builds the asset. `resources` maps every URI of [`GltfDocument::external_uris`] to its
    /// bytes.
    pub fn import(self, resources: &HashMap<String, Vec<u8>>) -> Result<Asset, AssetError> {
        import::import(&self.document, self.blob, resources)
    }
}

/// Loads a self-contained glTF (`.glb`, or `.gltf` with data URIs only).
pub fn load_gltf(bytes: &[u8]) -> Result<Asset, AssetError> {
    GltfDocument::parse(bytes)?.import(&HashMap::new())
}

/// Loads a `.gltf` or `.glb` file, reading external buffers and images relative to it.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_gltf_file(path: impl AsRef<std::path::Path>) -> Result<Asset, AssetError> {
    let path = path.as_ref();
    let read = |p: &std::path::Path, uri: &str| std::fs::read(p).map_err(|e| AssetError::Resource { uri: uri.to_string(), message: e.to_string() });
    let document = GltfDocument::parse(&read(path, &path.display().to_string())?)?;
    let base = path.parent().unwrap_or(std::path::Path::new("."));
    let mut resources = HashMap::new();
    for uri in document.external_uris() {
        let bytes = read(&base.join(uri::percent_decode(&uri)), &uri)?;
        resources.insert(uri, bytes);
    }
    document.import(&resources)
}
