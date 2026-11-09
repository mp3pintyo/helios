//! glTF document → [`Asset`].

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use glam::{Mat4, Vec2, Vec3, Vec4};
use helios_geometry::{Mesh, Vertex};

use crate::accessor::read_floats;
use crate::uri::{decode_data_uri, is_data_uri};
use crate::{AlphaMode, Asset, AssetError, Image, Instance, Material, NamedMesh, TextureRef, TextureTransform, WrapMode};

/// Extensions the importer interprets; others are ignored (with a warning).
const UNDERSTOOD_EXTENSIONS: &[&str] = &["KHR_materials_emissive_strength", "KHR_texture_transform", "KHR_mesh_quantization"];

/// The `gltf` crate's validation, except that a document may require extensions the importer
/// implements itself: `KHR_mesh_quantization` (attributes are decoded by [`read_floats`], not
/// by the crate's readers, which assume the core formats). Other unsupported required
/// extensions are still rejected.
pub fn validate(document: &gltf::Document) -> Result<(), AssetError> {
    use gltf::json::validation::Validate;
    let root = document.as_json();
    let mut errors = Vec::new();
    root.validate(root, gltf::json::Path::new, &mut |path, error| errors.push((path(), error)));
    errors.retain(|(path, _)| {
        !root.extensions_required.iter().enumerate().any(|(i, ext)| {
            UNDERSTOOD_EXTENSIONS.contains(&ext.as_str()) && path.as_str() == format!("extensionsRequired[{i}] = \"{ext}\"")
        })
    });
    if errors.is_empty() {
        Ok(())
    } else {
        Err(AssetError::Gltf(gltf::Error::Validation(errors)))
    }
}

pub fn import(document: &gltf::Document, blob: Option<Vec<u8>>, resources: &HashMap<String, Vec<u8>>) -> Result<Asset, AssetError> {
    for ext in document.extensions_used() {
        if !UNDERSTOOD_EXTENSIONS.contains(&ext) {
            log::warn!("glTF extension {ext} is ignored");
        }
    }
    let buffers = load_buffers(document, blob, resources)?;
    let images = decode_images(document, &buffers, resources)?;

    // Every primitive becomes a mesh (built in parallel); only instanced ones are kept.
    let primitives: Vec<gltf::Primitive> = document.meshes().flat_map(|m| m.primitives()).collect();
    let mut instancer = Instancer {
        primitives: par_map(&primitives, |p| read_primitive(p, &buffers)),
        first_primitive: document
            .meshes()
            .scan(0, |next, m| {
                let first = *next;
                *next += m.primitives().len();
                Some(first)
            })
            .collect(),
        materials: document.materials().map(|m| read_material(&m)).collect(),
        default_material: None,
        mesh_index: HashMap::new(),
        asset: Asset { images, ..Default::default() },
    };
    match document.default_scene().or_else(|| document.scenes().next()) {
        Some(scene) => scene.nodes().for_each(|root| instancer.visit(&root, Mat4::IDENTITY)),
        None => {
            // No scene: every node that is nobody's child is a root.
            let children: HashSet<usize> = document.nodes().flat_map(|n| n.children().map(|c| c.index())).collect();
            document.nodes().filter(|n| !children.contains(&n.index())).for_each(|root| instancer.visit(&root, Mat4::IDENTITY));
        }
    }
    let mut asset = instancer.asset;
    asset.materials = instancer.materials;
    Ok(asset)
}

/// Bytes of an external or `data:` URI.
fn resource<'a>(uri: &str, resources: &'a HashMap<String, Vec<u8>>) -> Result<Cow<'a, [u8]>, AssetError> {
    if is_data_uri(uri) {
        let bytes = decode_data_uri(uri).ok_or_else(|| AssetError::Resource { uri: "data URI".into(), message: "not base64".into() })?;
        return Ok(Cow::Owned(bytes));
    }
    resources.get(uri).map(|b| Cow::Borrowed(b.as_slice())).ok_or_else(|| AssetError::MissingResource(uri.to_string()))
}

fn load_buffers<'a>(
    document: &gltf::Document,
    mut blob: Option<Vec<u8>>,
    resources: &'a HashMap<String, Vec<u8>>,
) -> Result<Vec<Cow<'a, [u8]>>, AssetError> {
    document
        .buffers()
        .map(|buffer| {
            let data = match buffer.source() {
                gltf::buffer::Source::Bin => Cow::Owned(blob.take().ok_or_else(|| AssetError::MissingResource("GLB binary chunk".into()))?),
                gltf::buffer::Source::Uri(uri) => resource(uri, resources)?,
            };
            if data.len() < buffer.length() {
                return Err(AssetError::Resource { uri: format!("buffer {}", buffer.index()), message: "shorter than declared".into() });
            }
            Ok(data)
        })
        .collect()
}

/// Decodes every image in parallel (the slow part of loading).
fn decode_images(document: &gltf::Document, buffers: &[Cow<[u8]>], resources: &HashMap<String, Vec<u8>>) -> Result<Vec<Image>, AssetError> {
    let encoded = document
        .images()
        .map(|image| {
            let label = format!("image {}", image.index());
            match image.source() {
                gltf::image::Source::View { view, .. } => {
                    let range = view.offset()..view.offset() + view.length();
                    let bytes = buffers[view.buffer().index()]
                        .get(range)
                        .ok_or_else(|| AssetError::Resource { uri: label.clone(), message: "buffer view out of range".into() })?;
                    Ok((label, Cow::Borrowed(bytes)))
                }
                gltf::image::Source::Uri { uri, .. } => Ok((if is_data_uri(uri) { label } else { uri.to_string() }, resource(uri, resources)?)),
            }
        })
        .collect::<Result<Vec<(String, Cow<[u8]>)>, AssetError>>()?;
    par_map(&encoded, |(name, bytes)| decode_image(name, bytes)).into_iter().collect()
}

/// Walks the node hierarchy, turning each (node, primitive) into an instance.
struct Instancer {
    /// Meshes of all primitives in document order (`None`: not triangles).
    primitives: Vec<Option<Mesh>>,
    /// Index of each glTF mesh's first primitive in `primitives`.
    first_primitive: Vec<usize>,
    materials: Vec<Material>,
    default_material: Option<usize>,
    /// (primitive, mirrored) -> index in `asset.meshes`.
    mesh_index: HashMap<(usize, bool), usize>,
    asset: Asset,
}

impl Instancer {
    fn visit(&mut self, node: &gltf::Node, parent: Mat4) {
        let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            for (p, primitive) in mesh.primitives().enumerate() {
                let flat = self.first_primitive[mesh.index()] + p;
                if let Some(index) = self.mesh(flat, &mesh, p, world) {
                    let material = self.material(&primitive);
                    self.asset.instances.push(Instance { mesh: index, material, transform: world });
                }
            }
        }
        for child in node.children() {
            self.visit(&child, world);
        }
    }

    /// The asset mesh for a primitive, created on first use. A mirroring transform flips the
    /// winding and the renderer culls back faces, so mirrored instances get a copy of the mesh
    /// with reversed triangles.
    fn mesh(&mut self, flat: usize, mesh: &gltf::Mesh, primitive: usize, world: Mat4) -> Option<usize> {
        let source = self.primitives[flat].as_ref()?;
        let mirrored = world.determinant() < 0.0;
        let meshes = &mut self.asset.meshes;
        Some(*self.mesh_index.entry((flat, mirrored)).or_insert_with(|| {
            let mut name = mesh.name().map_or_else(|| format!("mesh {}", mesh.index()), str::to_string);
            if mesh.primitives().len() > 1 {
                name = format!("{name} #{primitive}");
            }
            let mut mesh = source.clone();
            if mirrored {
                mesh.indices.chunks_exact_mut(3).for_each(|t| t.swap(1, 2));
                name.push_str(" (mirrored)");
            }
            meshes.push(NamedMesh { name, mesh });
            meshes.len() - 1
        }))
    }

    /// The primitive's material; primitives without one share the glTF default material.
    fn material(&mut self, primitive: &gltf::Primitive) -> usize {
        match primitive.material().index() {
            Some(i) => i,
            None => *self.default_material.get_or_insert_with(|| {
                self.materials.push(Material::default());
                self.materials.len() - 1
            }),
        }
    }
}

fn decode_image(name: &str, bytes: &[u8]) -> Result<Image, AssetError> {
    let decoded = image::load_from_memory(bytes).map_err(|e| AssetError::Resource { uri: name.to_string(), message: e.to_string() })?;
    let rgba = decoded.into_rgba8();
    Ok(Image { width: rgba.width(), height: rgba.height(), pixels: rgba.into_raw() })
}

/// The primitive's attribute `semantic` as `N`-component vectors (missing components 1, so
/// RGB colours get an opaque alpha); `None` if absent, malformed or of the wrong length.
fn attribute<const N: usize>(primitive: &gltf::Primitive, semantic: gltf::Semantic, buffers: &[Cow<[u8]>], count: Option<usize>) -> Option<Vec<[f32; N]>> {
    let accessor = primitive.get(&semantic)?;
    let components = accessor.dimensions().multiplicity();
    let Some(values) = read_floats(&accessor, buffers).filter(|_| count.is_none_or(|c| c == accessor.count())) else {
        log::warn!("ignoring malformed {semantic:?} attribute");
        return None;
    };
    Some(values.chunks_exact(components).map(|v| std::array::from_fn(|i| v.get(i).copied().unwrap_or(1.0))).collect())
}

/// Reads one primitive as an indexed triangle mesh, or `None` for points and lines.
fn read_primitive(primitive: &gltf::Primitive, buffers: &[Cow<[u8]>]) -> Option<Mesh> {
    use gltf::mesh::Mode;
    use gltf::Semantic;
    let reader = primitive.reader(|b| buffers.get(b.index()).map(|d| &d[..]));
    let positions: Vec<Vec3> = attribute::<3>(primitive, Semantic::Positions, buffers, None)?.into_iter().map(Vec3::from).collect();
    let count = Some(positions.len());
    let indices: Vec<u32> = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };
    if indices.iter().any(|&i| i as usize >= positions.len()) {
        log::warn!("skipping primitive with out-of-range indices");
        return None;
    }
    let indices = match primitive.mode() {
        Mode::Triangles => indices[..indices.len() / 3 * 3].to_vec(),
        Mode::TriangleStrip => strip_to_list(&indices),
        Mode::TriangleFan => fan_to_list(&indices),
        mode => {
            log::warn!("skipping {mode:?} primitive (only triangles are rendered)");
            return None;
        }
    };
    let uv_set = |set| attribute::<2>(primitive, Semantic::TexCoords(set), buffers, count).map(|uvs| uvs.into_iter().map(Vec2::from).collect::<Vec<_>>());
    let uvs = uv_set(0).unwrap_or_else(|| vec![Vec2::ZERO; positions.len()]);
    let uv1 = uv_set(1).unwrap_or_default();
    let colors: Vec<Vec4> = attribute::<4>(primitive, Semantic::Colors(0), buffers, count).map_or_else(Vec::new, |c| c.into_iter().map(Vec4::from).collect());
    let mut mesh = match attribute::<3>(primitive, Semantic::Normals, buffers, count) {
        Some(normals) => {
            let vertices =
                positions.iter().zip(normals).zip(&uvs).map(|((&p, n), &uv)| Vertex::new(p, Vec3::from(n).normalize_or(Vec3::Y), uv)).collect();
            Mesh::with_attributes(vertices, indices, uv1, colors)
        }
        None => flat_shaded(&positions, &uvs, &uv1, &colors, &indices),
    };
    // Exporters often split vertices needlessly; welding lets the simplifier see the topology.
    mesh.weld();
    Some(mesh)
}

/// glTF requires flat normals for primitives without them: one wedge per triangle corner.
/// `uv1` and `colors` are empty or one per position.
fn flat_shaded(positions: &[Vec3], uvs: &[Vec2], uv1: &[Vec2], colors: &[Vec4], indices: &[u32]) -> Mesh {
    let mut vertices = Vec::with_capacity(indices.len());
    for t in indices.chunks_exact(3) {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| i as usize);
        let normal = (positions[b] - positions[a]).cross(positions[c] - positions[a]).normalize_or(Vec3::Y);
        vertices.extend([a, b, c].map(|i| Vertex::new(positions[i], normal, uvs[i])));
    }
    let corners = |values: &[Vec2]| if values.is_empty() { Vec::new() } else { indices.iter().map(|&i| values[i as usize]).collect() };
    let colors = if colors.is_empty() { Vec::new() } else { indices.iter().map(|&i| colors[i as usize]).collect() };
    Mesh::with_attributes(vertices, (0..indices.len() as u32).collect(), corners(uv1), colors)
}

/// Triangle strip → list, alternating the winding of odd triangles back to the first's.
fn strip_to_list(strip: &[u32]) -> Vec<u32> {
    strip
        .windows(3)
        .enumerate()
        .flat_map(|(i, w)| if i % 2 == 0 { [w[0], w[1], w[2]] } else { [w[1], w[0], w[2]] })
        .collect()
}

fn fan_to_list(fan: &[u32]) -> Vec<u32> {
    if fan.len() < 3 {
        return Vec::new();
    }
    fan[1..].windows(2).flat_map(|w| [fan[0], w[0], w[1]]).collect()
}

/// A texture reference: the image, the UV set (`KHR_texture_transform` may override it), the
/// transform and the sampler's wrap modes.
fn texture_ref(texture: gltf::Texture, tex_coord: u32, transform: Option<(TextureTransform, Option<u32>)>) -> TextureRef {
    let (transform, override_set) = transform.unwrap_or((TextureTransform::IDENTITY, None));
    let mut set = override_set.unwrap_or(tex_coord);
    if set > 1 {
        log::warn!("texture coordinate set {set} is not supported; using set 0");
        set = 0;
    }
    let wrap = |mode: gltf::texture::WrappingMode| match mode {
        gltf::texture::WrappingMode::Repeat => WrapMode::Repeat,
        gltf::texture::WrappingMode::MirroredRepeat => WrapMode::MirroredRepeat,
        gltf::texture::WrappingMode::ClampToEdge => WrapMode::ClampToEdge,
    };
    let sampler = texture.sampler();
    TextureRef { image: texture.source().index(), tex_coord: set, transform, wrap: [wrap(sampler.wrap_s()), wrap(sampler.wrap_t())] }
}

/// `KHR_texture_transform` from its JSON (normal and occlusion textures expose only that):
/// the transform and the optional `texCoord` override.
fn transform_from_json(value: Option<&gltf::json::Value>) -> Option<(TextureTransform, Option<u32>)> {
    let value = value?;
    let pair = |key: &str, default: Vec2| {
        value.get(key).and_then(|v| v.as_array()).filter(|a| a.len() == 2).map_or(default, |a| {
            Vec2::new(a[0].as_f64().unwrap_or(default.x as f64) as f32, a[1].as_f64().unwrap_or(default.y as f64) as f32)
        })
    };
    let transform = TextureTransform {
        offset: pair("offset", Vec2::ZERO),
        rotation: value.get("rotation").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        scale: pair("scale", Vec2::ONE),
    };
    Some((transform, value.get("texCoord").and_then(|v| v.as_u64()).map(|s| s as u32)))
}

fn read_material(m: &gltf::Material) -> Material {
    let info = |info: Option<gltf::texture::Info>| {
        info.map(|info| {
            let transform = info.texture_transform().map(|t| {
                let transform = TextureTransform { offset: Vec2::from(t.offset()), rotation: t.rotation(), scale: Vec2::from(t.scale()) };
                (transform, t.tex_coord())
            });
            texture_ref(info.texture(), info.tex_coord(), transform)
        })
    };
    const TRANSFORM: &str = "KHR_texture_transform";
    let pbr = m.pbr_metallic_roughness();
    let normal = m.normal_texture();
    let occlusion = m.occlusion_texture();
    Material {
        name: m.name().map_or_else(|| format!("material {}", m.index().unwrap_or(0)), str::to_string),
        base_color: Vec4::from(pbr.base_color_factor()),
        base_color_texture: info(pbr.base_color_texture()),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        metallic_roughness_texture: info(pbr.metallic_roughness_texture()),
        normal_texture: normal.as_ref().map(|n| texture_ref(n.texture(), n.tex_coord(), transform_from_json(n.extension_value(TRANSFORM)))),
        normal_scale: normal.as_ref().map_or(1.0, |n| n.scale()),
        occlusion_texture: occlusion.as_ref().map(|o| texture_ref(o.texture(), o.tex_coord(), transform_from_json(o.extension_value(TRANSFORM)))),
        occlusion_strength: occlusion.as_ref().map_or(1.0, |o| o.strength()),
        emissive: Vec3::from(m.emissive_factor()) * m.emissive_strength().unwrap_or(1.0),
        emissive_texture: info(m.emissive_texture()),
        alpha_mode: match m.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask { cutoff: m.alpha_cutoff().unwrap_or(0.5) },
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        double_sided: m.double_sided(),
    }
}

/// Maps `f` over `items`, natively on all cores (work-stealing over an atomic index, so
/// uneven items balance), sequentially on wasm.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(items.len());
        if threads > 1 {
            let next = AtomicUsize::new(0);
            let mut slots: Vec<Option<R>> = std::iter::repeat_with(|| None).take(items.len()).collect();
            std::thread::scope(|s| {
                let workers: Vec<_> = (0..threads)
                    .map(|_| {
                        s.spawn(|| {
                            let mut done = Vec::new();
                            loop {
                                let i = next.fetch_add(1, Ordering::Relaxed);
                                let Some(item) = items.get(i) else { break };
                                done.push((i, f(item)));
                            }
                            done
                        })
                    })
                    .collect();
                for worker in workers {
                    for (i, r) in worker.join().expect("worker panicked") {
                        slots[i] = Some(r);
                    }
                }
            });
            return slots.into_iter().map(|r| r.expect("every item mapped")).collect();
        }
    }
    items.iter().map(f).collect()
}

#[cfg(test)]
mod tests;
