use std::collections::HashMap;

use base64::Engine;
use glam::{Mat4, Vec3};

use super::{fan_to_list, strip_to_list};
use crate::{load_gltf, AlphaMode, AssetError, GltfDocument, TextureRef, TextureTransform, WrapMode};

/// A unit quad in the XY plane (counter-clockwise seen from +Z): positions, UVs, u16 indices.
fn quad_buffer() -> Vec<u8> {
    let positions: [[f32; 3]; 4] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    let uvs: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    let mut bytes = Vec::new();
    positions.iter().flatten().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
    uvs.iter().flatten().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
    indices.iter().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    bytes
}

fn png_2x2() -> Vec<u8> {
    let pixels = vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 128];
    let image = image::RgbaImage::from_raw(2, 2, pixels).unwrap();
    let mut out = std::io::Cursor::new(Vec::new());
    image.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn data_uri(mime: &str, bytes: &[u8]) -> String {
    format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes))
}

/// A document with a node hierarchy, a mirrored node, a material-less primitive and one
/// textured material. `buffer` / `image` are the JSON of the buffer's and image's source.
fn document(buffer: &str, image: &str) -> String {
    let len = quad_buffer().len();
    format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "extensionsUsed": ["KHR_materials_emissive_strength"],
  "buffers": [{{ {buffer} "byteLength": {len} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": 48 }},
    {{ "buffer": 0, "byteOffset": 48, "byteLength": 32 }},
    {{ "buffer": 0, "byteOffset": 80, "byteLength": 12 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0] }},
    {{ "bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2" }},
    {{ "bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR" }}
  ],
  "images": [{{ {image} }}],
  "textures": [{{ "source": 0 }}],
  "materials": [{{
    "name": "painted",
    "pbrMetallicRoughness": {{ "baseColorFactor": [0.5, 0.25, 1, 1], "baseColorTexture": {{ "index": 0 }}, "metallicFactor": 0.25, "roughnessFactor": 0.75 }},
    "normalTexture": {{ "index": 0, "scale": 0.5 }},
    "emissiveFactor": [1, 0.5, 0],
    "extensions": {{ "KHR_materials_emissive_strength": {{ "emissiveStrength": 2 }} }},
    "alphaMode": "MASK", "alphaCutoff": 0.3, "doubleSided": true
  }}],
  "meshes": [
    {{ "name": "quad", "primitives": [{{ "attributes": {{ "POSITION": 0, "TEXCOORD_0": 1 }}, "indices": 2, "material": 0 }}] }},
    {{ "primitives": [{{ "attributes": {{ "POSITION": 0 }}, "indices": 2 }}] }}
  ],
  "nodes": [
    {{ "translation": [1, 0, 0], "children": [1] }},
    {{ "mesh": 0, "scale": [2, 2, 2] }},
    {{ "mesh": 0, "scale": [-1, 1, 1] }},
    {{ "mesh": 1 }},
    {{ "mesh": 1, "translation": [0, 9, 0] }}
  ],
  "scenes": [{{ "nodes": [0, 2, 3] }}],
  "scene": 0
}}"#
    )
}

fn embedded() -> String {
    let buffer = format!(r#""uri": "{}","#, data_uri("application/octet-stream", &quad_buffer()));
    let image = format!(r#""uri": "{}""#, data_uri("image/png", &png_2x2()));
    document(&buffer, &image)
}

#[test]
fn imports_hierarchy_meshes_and_materials() {
    let asset = load_gltf(embedded().as_bytes()).unwrap();

    // Node 4 is not in the scene; node 2 mirrors, so mesh 0 also exists with reversed winding.
    assert_eq!(asset.instances.len(), 3);
    assert_eq!(asset.meshes.len(), 3);
    let child = asset.instances.iter().find(|i| i.transform.x_axis.x == 2.0).unwrap();
    assert_eq!(child.transform, Mat4::from_translation(Vec3::X) * Mat4::from_scale(Vec3::splat(2.0)));
    assert_eq!(asset.meshes[child.mesh].name, "quad");

    // No normals in the file: flat normals, facing +Z for this counter-clockwise quad.
    let quad = &asset.meshes[child.mesh].mesh;
    assert_eq!(quad.triangle_count(), 2);
    assert!(quad.vertices.iter().all(|v| v.normal() == Vec3::Z));
    assert_eq!(quad.vertices[0].uv(), glam::Vec2::new(0.0, 1.0));

    let mirrored = asset.instances.iter().find(|i| i.transform.determinant() < 0.0).unwrap();
    let flipped = &asset.meshes[mirrored.mesh];
    assert!(flipped.name.ends_with("(mirrored)"));
    let [a, b, c] = [0, 1, 2].map(|k| mirrored.transform.transform_point3(flipped.mesh.vertices[flipped.mesh.indices[k] as usize].position()));
    let world_normal = (b - a).cross(c - a);
    assert!(world_normal.z > 0.0, "mirrored instance must still face +Z after the transform");

    // The primitive without a material gets the glTF default material.
    assert_eq!(asset.materials.len(), 2);
    let plain = asset.instances.iter().find(|i| i.material == 1).unwrap();
    assert_eq!(asset.materials[plain.material].metallic, 1.0);
    assert!(asset.meshes[plain.mesh].mesh.vertices.iter().all(|v| v.uv() == glam::Vec2::ZERO));

    let m = &asset.materials[0];
    assert_eq!(m.name, "painted");
    assert_eq!(m.base_color, glam::Vec4::new(0.5, 0.25, 1.0, 1.0));
    assert_eq!((m.metallic, m.roughness), (0.25, 0.75));
    assert_eq!(m.base_color_texture, Some(TextureRef::new(0)));
    assert_eq!((m.normal_texture, m.normal_scale), (Some(TextureRef::new(0)), 0.5));
    assert_eq!(m.emissive, Vec3::new(2.0, 1.0, 0.0));
    assert_eq!(m.alpha_mode, AlphaMode::Mask { cutoff: 0.3 });
    assert!(m.double_sided);

    assert_eq!(asset.images.len(), 1);
    let image = &asset.images[0];
    assert_eq!((image.width, image.height), (2, 2));
    assert_eq!(&image.pixels[..4], &[255, 0, 0, 255]);
    assert_eq!(image.pixels[15], 128);

    let bounds = asset.bounds();
    assert_eq!(bounds.min, Vec3::new(-1.0, 0.0, 0.0));
    assert_eq!(bounds.max, Vec3::new(3.0, 2.0, 0.0));
    assert_eq!(asset.triangle_count(), 6);
}

#[test]
fn glb_container() {
    let image_bytes = png_2x2();
    let mut bin = quad_buffer();
    let image_offset = bin.len();
    bin.extend_from_slice(&image_bytes);
    bin.resize(bin.len().next_multiple_of(4), 0);
    // The image lives in a fourth buffer view of the binary chunk.
    let json = document("", r#""bufferView": 3, "mimeType": "image/png""#)
        .replace(&format!(r#""byteLength": {} }}],"#, quad_buffer().len()), &format!(r#""byteLength": {} }}],"#, bin.len()))
        .replace(
            r#"{ "buffer": 0, "byteOffset": 80, "byteLength": 12 }"#,
            &format!(r#"{{ "buffer": 0, "byteOffset": 80, "byteLength": 12 }}, {{ "buffer": 0, "byteOffset": {image_offset}, "byteLength": {} }}"#, image_bytes.len()),
        );
    let mut json = json.into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);

    let document = GltfDocument::parse(&glb).unwrap();
    assert!(document.external_uris().is_empty());
    let asset = document.import(&HashMap::new()).unwrap();
    assert_eq!(asset.instances.len(), 3);
    assert_eq!(asset.images[0].pixels[4..8], [0, 255, 0, 255]);
}

#[test]
fn external_resources() {
    let json = document(r#""uri": "quad.bin","#, r#""uri": "my%20texture.png""#);
    let document = GltfDocument::parse(json.as_bytes()).unwrap();
    assert_eq!(document.external_uris(), ["quad.bin", "my%20texture.png"]);
    let missing = GltfDocument::parse(json.as_bytes()).unwrap().import(&HashMap::new());
    assert!(matches!(missing, Err(AssetError::MissingResource(uri)) if uri == "quad.bin"));

    let resources = HashMap::from([("quad.bin".to_string(), quad_buffer()), ("my%20texture.png".to_string(), png_2x2())]);
    let asset = document.import(&resources).unwrap();
    assert_eq!(asset.meshes.len(), 3);
    assert_eq!(asset.images[0].width, 2);
}

#[test]
fn reads_pbr_extensions() {
    let extensions = r#""extensions": {
        "KHR_materials_emissive_strength": { "emissiveStrength": 2 },
        "KHR_materials_clearcoat": { "clearcoatFactor": 1, "clearcoatRoughnessFactor": 0.2,
            "clearcoatNormalTexture": { "index": 0, "scale": 0.5, "texCoord": 1 } },
        "KHR_materials_sheen": { "sheenColorFactor": [0.5, 0.25, 0], "sheenRoughnessFactor": 0.4, "sheenRoughnessTexture": { "index": 0 } },
        "KHR_materials_ior": { "ior": 1.33 },
        "KHR_materials_specular": { "specularFactor": 0.5, "specularColorFactor": [1, 0.5, 0.25] },
        "KHR_materials_transmission": { "transmissionFactor": 0.75 },
        "KHR_materials_volume": { "thicknessFactor": 0.1, "attenuationDistance": 2, "attenuationColor": [1, 0.5, 0.5] },
        "KHR_materials_unlit": {},
        "KHR_materials_anisotropy": { "anisotropyStrength": 0.6, "anisotropyRotation": 1.5 } }"#;
    let json = embedded()
        .replace(r#""extensions": { "KHR_materials_emissive_strength": { "emissiveStrength": 2 } }"#, extensions)
        .replace(r#""extensionsUsed""#, r#""extensionsRequired": ["KHR_materials_clearcoat", "KHR_materials_unlit"], "extensionsUsed""#);
    let asset = load_gltf(json.as_bytes()).unwrap();
    let e = &asset.materials[0].extensions;
    assert_eq!((e.clearcoat, e.clearcoat_roughness, e.clearcoat_normal_scale), (1.0, 0.2, 0.5));
    assert_eq!(e.clearcoat_normal_texture.map(|t| (t.image, t.tex_coord)), Some((0, 1)));
    assert_eq!((e.sheen_color, e.sheen_roughness), (Vec3::new(0.5, 0.25, 0.0), 0.4));
    assert!(e.sheen_roughness_texture.is_some() && e.sheen_color_texture.is_none());
    assert_eq!((e.ior, e.specular, e.specular_color), (1.33, 0.5, Vec3::new(1.0, 0.5, 0.25)));
    assert_eq!((e.transmission, e.thickness, e.attenuation_distance), (0.75, 0.1, 2.0));
    assert_eq!((e.unlit, e.anisotropy, e.anisotropy_rotation), (true, 0.6, 1.5));
    // Without the extensions: the defaults.
    let plain = load_gltf(embedded().as_bytes()).unwrap();
    assert_eq!(plain.materials[0].extensions, crate::MaterialExtensions::default());
}

#[test]
fn rejects_unsupported_required_extensions() {
    let json = embedded().replace(r#""extensionsUsed""#, r#""extensionsRequired": ["KHR_draco_mesh_compression"], "extensionsUsed""#);
    let result = GltfDocument::parse(json.as_bytes());
    assert!(matches!(result, Err(AssetError::Gltf(_))), "{:?}", result.err());
}

/// A `KHR_mesh_quantization` quad: i16 positions interleaved with i8 normalized normals
/// (stride 12), u16 unnormalized TEXCOORD_0, u8 normalized TEXCOORD_1 and RGB COLOR_0, u8
/// indices; one material exercising texture transforms, texCoord overrides, wrap modes and
/// occlusion.
fn quantized_document() -> String {
    let mut bytes = Vec::new();
    let corners: [[i16; 3]; 4] = [[0, 0, 0], [1000, 0, 0], [1000, 1000, 0], [0, 1000, 0]];
    for p in corners {
        p.iter().for_each(|c| bytes.extend_from_slice(&c.to_le_bytes()));
        bytes.extend_from_slice(&[0, 0, 0, 0, 127, 0]); // pad, normal (0, 0, 1) as i8, pad
    }
    for uv in [[0u16, 4096], [4096, 4096], [4096, 0], [0, 0]] {
        uv.iter().for_each(|c| bytes.extend_from_slice(&c.to_le_bytes()));
    }
    bytes.extend_from_slice(&[0, 0, 255, 0, 255, 255, 0, 255]); // TEXCOORD_1, u8 normalized
    bytes.extend_from_slice(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]); // COLOR_0, RGB u8
    bytes.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    let len = bytes.len();
    let data = data_uri("application/octet-stream", &bytes);
    let image = data_uri("image/png", &png_2x2());
    let transform = r#""KHR_texture_transform": { "offset": [0.5, 0.25], "rotation": 1.5, "scale": [2, 3], "texCoord": 1 }"#;
    format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "extensionsUsed": ["KHR_mesh_quantization", "KHR_texture_transform"],
  "extensionsRequired": ["KHR_mesh_quantization"],
  "buffers": [{{ "uri": "{data}", "byteLength": {len} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": 48, "byteStride": 12 }},
    {{ "buffer": 0, "byteOffset": 48, "byteLength": 16 }},
    {{ "buffer": 0, "byteOffset": 64, "byteLength": 8 }},
    {{ "buffer": 0, "byteOffset": 72, "byteLength": 12 }},
    {{ "buffer": 0, "byteOffset": 84, "byteLength": 6 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5122, "count": 4, "type": "VEC3", "min": [0, 0, 0], "max": [1000, 1000, 0] }},
    {{ "bufferView": 0, "byteOffset": 10, "componentType": 5120, "normalized": true, "count": 4, "type": "VEC3" }},
    {{ "bufferView": 1, "componentType": 5123, "count": 4, "type": "VEC2" }},
    {{ "bufferView": 2, "componentType": 5121, "normalized": true, "count": 4, "type": "VEC2" }},
    {{ "bufferView": 3, "componentType": 5121, "normalized": true, "count": 4, "type": "VEC3" }},
    {{ "bufferView": 4, "componentType": 5121, "count": 6, "type": "SCALAR" }}
  ],
  "images": [{{ "uri": "{image}" }}],
  "samplers": [{{ "wrapS": 33071, "wrapT": 33648 }}],
  "textures": [{{ "source": 0 }}, {{ "source": 0, "sampler": 0 }}],
  "materials": [{{
    "pbrMetallicRoughness": {{ "baseColorTexture": {{ "index": 1, "extensions": {{ {transform} }} }} }},
    "normalTexture": {{ "index": 0, "extensions": {{ {transform} }} }},
    "occlusionTexture": {{ "index": 0, "texCoord": 1, "strength": 0.5 }}
  }}],
  "meshes": [{{ "primitives": [{{ "attributes": {{ "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "TEXCOORD_1": 3, "COLOR_0": 4 }}, "indices": 5, "material": 0 }}] }}],
  "nodes": [{{ "mesh": 0, "scale": [0.001, 0.001, 0.001] }}],
  "scenes": [{{ "nodes": [0] }}]
}}"#
    )
}

#[test]
fn quantized_attributes_and_optional_sets() {
    let asset = load_gltf(quantized_document().as_bytes()).unwrap();
    let mesh = &asset.meshes[0].mesh;
    assert_eq!(mesh.triangle_count(), 2);
    // Unnormalized integers keep their values: the node's scale dequantises positions.
    assert_eq!(mesh.vertices[2].position(), Vec3::new(1000.0, 1000.0, 0.0));
    assert_eq!(asset.bounds().max, Vec3::new(1.0, 1.0, 0.0));
    assert!(mesh.vertices.iter().all(|v| v.normal() == Vec3::Z));
    assert_eq!(mesh.vertices[1].uv(), glam::Vec2::new(4096.0, 4096.0));
    assert_eq!((mesh.uv1.len(), mesh.colors.len()), (4, 4));
    assert_eq!(mesh.uv1[1], glam::Vec2::new(1.0, 0.0));
    // RGB colours get an opaque alpha.
    assert_eq!(mesh.colors[0], glam::Vec4::new(1.0, 0.0, 0.0, 1.0));
    assert_eq!(mesh.colors[3], glam::Vec4::ONE);
}

#[test]
fn texture_references_carry_sets_transforms_and_wrap_modes() {
    let asset = load_gltf(quantized_document().as_bytes()).unwrap();
    let m = &asset.materials[0];
    let expected = TextureTransform { offset: glam::Vec2::new(0.5, 0.25), rotation: 1.5, scale: glam::Vec2::new(2.0, 3.0) };
    let base = m.base_color_texture.unwrap();
    assert_eq!((base.tex_coord, base.transform, base.wrap), (1, expected, [WrapMode::ClampToEdge, WrapMode::MirroredRepeat]));
    // Normal and occlusion textures read the extension from their JSON.
    let normal = m.normal_texture.unwrap();
    assert_eq!((normal.tex_coord, normal.transform, normal.wrap), (1, expected, [WrapMode::Repeat; 2]));
    assert!(!normal.same_coordinates(&base));
    let occlusion = m.occlusion_texture.unwrap();
    assert_eq!((occlusion.tex_coord, occlusion.transform, m.occlusion_strength), (1, TextureTransform::IDENTITY, 0.5));
}

/// The transform is `T * R * S` with the rotation rows `(cos, sin)`, `(-sin, cos)`, as
/// three.js' `Texture.rotation` and the TextureTransformTest sample's green markers expect.
#[test]
fn texture_transform_matches_the_reference_implementations() {
    let t = TextureTransform { offset: glam::Vec2::new(0.25, -0.5), rotation: 0.7, scale: glam::Vec2::new(2.0, 0.5) };
    let (s, c) = t.rotation.sin_cos();
    let translation = glam::Mat3::from_cols(Vec3::X, Vec3::Y, Vec3::new(t.offset.x, t.offset.y, 1.0));
    let rotation = glam::Mat3::from_cols(Vec3::new(c, -s, 0.0), Vec3::new(s, c, 0.0), Vec3::Z);
    let scale = glam::Mat3::from_diagonal(Vec3::new(t.scale.x, t.scale.y, 1.0));
    let reference = translation * rotation * scale;
    for uv in [glam::Vec2::ZERO, glam::Vec2::X, glam::Vec2::new(0.3, 0.9), glam::Vec2::new(-2.0, 5.0)] {
        let expected = (reference * uv.extend(1.0)).truncate();
        assert!((t.matrix().transform_point2(uv) - expected).abs().max_element() < 1e-6, "{uv}");
    }
    // A quarter turn maps the u axis onto -v: counter-clockwise on the image, whose v points down.
    let quarter = TextureTransform { rotation: std::f32::consts::FRAC_PI_2, ..TextureTransform::IDENTITY };
    assert!((quarter.matrix().transform_point2(glam::Vec2::X) + glam::Vec2::Y).length() < 1e-6);
    assert_eq!(TextureTransform::IDENTITY.matrix(), glam::Affine2::IDENTITY);
}

#[test]
fn strips_and_fans() {
    // Strip 0-1-2-3: triangles (0,1,2) and (2,1,3), both with the first triangle's winding.
    assert_eq!(strip_to_list(&[0, 1, 2, 3]), [0, 1, 2, 2, 1, 3]);
    assert_eq!(fan_to_list(&[0, 1, 2, 3]), [0, 1, 2, 0, 2, 3]);
    assert!(fan_to_list(&[0, 1]).is_empty());
    assert!(strip_to_list(&[0, 1]).is_empty());
}

/// A two-bone strip in the style of the SimpleSkin sample: four vertices, the lower two bound to
/// joint 0, the upper two shared by both joints (u8 joints, normalized u8 weights that do not
/// quite sum to one), inverse bind matrices, and a clip rotating the upper joint 90° about Z.
fn skinned_document() -> String {
    let mut bytes = Vec::new();
    let positions: [[f32; 3]; 4] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]];
    positions.iter().flatten().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes())); // 0..48
    bytes.extend_from_slice(&[0, 1, 2, 0, 1, 2, 2, 0, 0, 0, 1, 0, 1, 0, 1, 0]); // joints 48..64
    bytes.extend_from_slice(&[255, 0, 0, 0, 255, 0, 0, 0, 100, 100, 50, 0, 120, 100, 30, 0]); // weights 64..80
    for m in [Mat4::IDENTITY, Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0))] {
        m.to_cols_array().iter().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes())); // 80..208
    }
    [0.0f32, 1.0].iter().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes())); // times 208..216
    let quarter = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    [glam::Quat::IDENTITY, quarter].iter().flat_map(|q| q.to_array()).for_each(|v| bytes.extend_from_slice(&v.to_le_bytes())); // 216..248
    [0u16, 1, 2, 1, 3, 2].iter().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes())); // 248..260
    let uri = data_uri("application/octet-stream", &bytes);
    format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "buffers": [{{ "uri": "{uri}", "byteLength": {} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": 48 }},
    {{ "buffer": 0, "byteOffset": 48, "byteLength": 16 }},
    {{ "buffer": 0, "byteOffset": 64, "byteLength": 16 }},
    {{ "buffer": 0, "byteOffset": 80, "byteLength": 128 }},
    {{ "buffer": 0, "byteOffset": 208, "byteLength": 8 }},
    {{ "buffer": 0, "byteOffset": 216, "byteLength": 32 }},
    {{ "buffer": 0, "byteOffset": 248, "byteLength": 12 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0, 0, 0], "max": [1, 2, 0] }},
    {{ "bufferView": 1, "componentType": 5121, "count": 4, "type": "VEC4" }},
    {{ "bufferView": 2, "componentType": 5121, "normalized": true, "count": 4, "type": "VEC4" }},
    {{ "bufferView": 3, "componentType": 5126, "count": 2, "type": "MAT4" }},
    {{ "bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1] }},
    {{ "bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC4" }},
    {{ "bufferView": 6, "componentType": 5123, "count": 6, "type": "SCALAR" }}
  ],
  "meshes": [{{ "primitives": [{{ "attributes": {{ "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2 }}, "indices": 6 }}] }}],
  "skins": [{{ "joints": [1, 2], "inverseBindMatrices": 3, "skeleton": 1 }}],
  "animations": [{{
    "name": "bend",
    "samplers": [{{ "input": 4, "output": 5, "interpolation": "LINEAR" }}],
    "channels": [{{ "sampler": 0, "target": {{ "node": 2, "path": "rotation" }} }}]
  }}],
  "nodes": [
    {{ "mesh": 0, "skin": 0, "translation": [5, 0, 0] }},
    {{ "children": [2] }},
    {{ "translation": [0, 1, 0] }}
  ],
  "scenes": [{{ "nodes": [0, 1] }}]
}}"#,
        bytes.len()
    )
}

#[test]
fn skins_joint_weights_and_animations() {
    use helios_geometry::skin::skin_point;
    use helios_geometry::SkinWeights;
    let asset = load_gltf(skinned_document().as_bytes()).unwrap();
    assert_eq!(asset.nodes.len(), 3);
    assert_eq!(asset.nodes[2].parent, Some(1));
    let skin = &asset.skins[0];
    assert_eq!((skin.joints.clone(), skin.skeleton), (vec![1, 2], Some(1)));
    assert_eq!(skin.inverse_bind_matrices[1], Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)));
    let instance = asset.instances[0];
    assert_eq!((instance.skin, instance.node), (Some(0), 0));

    // Weights are renormalised (100 + 100 + 50 of 255 over joints 0, 0, 1: merged) and
    // quantised; slots with zero weight are dropped.
    let mesh = &asset.meshes[instance.mesh].mesh;
    assert_eq!(mesh.skin.len(), mesh.vertices.len());
    let skin_at = |p: Vec3| *mesh.vertices.iter().zip(&mesh.skin).find(|(v, _)| v.position() == p).unwrap().1;
    assert_eq!(skin_at(Vec3::ZERO), SkinWeights::rigid(0));
    let top = skin_at(Vec3::new(0.0, 2.0, 0.0));
    assert_eq!((top.joints[..2].to_vec(), top.weights[..2].to_vec()), (vec![0, 1], vec![204, 51]));
    let shared = skin_at(Vec3::new(1.0, 2.0, 0.0));
    assert_eq!((shared.influences(), shared.weights.iter().map(|&w| w as u32).sum::<u32>()), (2, 255));

    let clip = &asset.animations[0];
    assert_eq!((clip.name.as_str(), clip.duration, clip.channels.len()), ("bend", 1.0, 1));
    // At t = 1 the upper joint has turned 90° about its pivot (0, 1): a vertex fully bound to it
    // at (1, 2) swings to (-1, 2); vertices of the lower joint stay. The palette is relative to
    // the mesh node (bind), whose own transform glTF ignores for skinned meshes.
    let order = crate::animation::hierarchy_order(&asset.nodes);
    let mut pose = crate::animation::rest_pose(&asset.nodes);
    clip.apply(1.0, &mut pose);
    let globals = crate::animation::global_matrices(&asset.nodes, &order, &pose);
    let palette = skin.joint_matrices(&globals, instance.transform);
    let swung = instance.transform.transform_point3(skin_point(&SkinWeights::rigid(1), &palette, Vec3::new(1.0, 2.0, 0.0)));
    assert!((swung - Vec3::new(-1.0, 2.0, 0.0)).length() < 1e-5, "{swung}");
    let lower = instance.transform.transform_point3(skin_point(&SkinWeights::rigid(0), &palette, Vec3::X));
    assert!((lower - Vec3::X).length() < 1e-5, "{lower}");
}
