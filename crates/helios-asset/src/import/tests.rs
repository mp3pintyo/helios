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
