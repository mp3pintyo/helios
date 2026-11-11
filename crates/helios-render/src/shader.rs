//! Embedded WGSL sources with a minimal `#include "file.wgsl"` preprocessor.
//!
//! Shaders that include `half.wgsl` get its f16 variant (`half_f16.wgsl`) on devices with f16
//! shaders ([`half_precision`]): one variant per device, chosen when the module is made.

use std::sync::atomic::{AtomicBool, Ordering};

const SOURCES: &[(&str, &str)] = &[
    ("alpha_mask.wgsl", include_str!("shaders/alpha_mask.wgsl")),
    ("assembly.wgsl", include_str!("shaders/assembly.wgsl")),
    ("atmosphere.wgsl", include_str!("shaders/atmosphere.wgsl")),
    ("avboit.wgsl", include_str!("shaders/avboit.wgsl")),
    ("avboit_common.wgsl", include_str!("shaders/avboit_common.wgsl")),
    ("avboit_read.wgsl", include_str!("shaders/avboit_read.wgsl")),
    ("avboit_write.wgsl", include_str!("shaders/avboit_write.wgsl")),
    ("blackbody.wgsl", include_str!("shaders/blackbody.wgsl")),
    ("bloom.wgsl", include_str!("shaders/bloom.wgsl")),
    ("brdf.wgsl", include_str!("shaders/brdf.wgsl")),
    ("classify.wgsl", include_str!("shaders/classify.wgsl")),
    ("clouds.wgsl", include_str!("shaders/clouds.wgsl")),
    ("clouds_common.wgsl", include_str!("shaders/clouds_common.wgsl")),
    ("common.wgsl", include_str!("shaders/common.wgsl")),
    ("cull.wgsl", include_str!("shaders/cull.wgsl")),
    ("cull_args.wgsl", include_str!("shaders/cull_args.wgsl")),
    ("displacement.wgsl", include_str!("shaders/displacement.wgsl")),
    ("displacement_graph.wgsl", include_str!("shaders/displacement_graph.wgsl")),
    ("displacement_graph_resolve.wgsl", include_str!("shaders/displacement_graph_resolve.wgsl")),
    ("displacement_graphs_off.wgsl", include_str!("shaders/displacement_graphs_off.wgsl")),
    ("displacement_height.wgsl", include_str!("shaders/displacement_height.wgsl")),
    ("dynamic_caster.wgsl", include_str!("shaders/dynamic_caster.wgsl")),
    ("dynamic_debug.wgsl", include_str!("shaders/dynamic_debug.wgsl")),
    ("vsm_overview.wgsl", include_str!("shaders/vsm_overview.wgsl")),
    ("exposure.wgsl", include_str!("shaders/exposure.wgsl")),
    ("fog.wgsl", include_str!("shaders/fog.wgsl")),
    ("fog_common.wgsl", include_str!("shaders/fog_common.wgsl")),
    ("fog_scattering.wgsl", include_str!("shaders/fog_scattering.wgsl")),
    ("gbuffer.wgsl", include_str!("shaders/gbuffer.wgsl")),
    ("geometry.wgsl", include_str!("shaders/geometry.wgsl")),
    ("gi.wgsl", include_str!("shaders/gi.wgsl")),
    ("glass_light.wgsl", include_str!("shaders/glass_light.wgsl")),
    ("glass_shadow.wgsl", include_str!("shaders/glass_shadow.wgsl")),
    ("half.wgsl", include_str!("shaders/half.wgsl")),
    ("half_f16.wgsl", include_str!("shaders/half_f16.wgsl")),
    ("hzb.wgsl", include_str!("shaders/hzb.wgsl")),
    ("hzb_trace.wgsl", include_str!("shaders/hzb_trace.wgsl")),
    ("light_cull.wgsl", include_str!("shaders/light_cull.wgsl")),
    ("lighting.wgsl", include_str!("shaders/lighting.wgsl")),
    ("local_exposure.wgsl", include_str!("shaders/local_exposure.wgsl")),
    ("lights.wgsl", include_str!("shaders/lights.wgsl")),
    ("lumen_card_light.wgsl", include_str!("shaders/lumen_card_light.wgsl")),
    ("lumen_cards.wgsl", include_str!("shaders/lumen_cards.wgsl")),
    ("lumen_card_capture.wgsl", include_str!("shaders/lumen_card_capture.wgsl")),
    ("lumen_common.wgsl", include_str!("shaders/lumen_common.wgsl")),
    ("lumen_displace.wgsl", include_str!("shaders/lumen_displace.wgsl")),
    ("lumen_gather.wgsl", include_str!("shaders/lumen_gather.wgsl")),
    ("lumen_light.wgsl", include_str!("shaders/lumen_light.wgsl")),
    ("lumen_light_grid.wgsl", include_str!("shaders/lumen_light_grid.wgsl")),
    ("lumen_light_grid_common.wgsl", include_str!("shaders/lumen_light_grid_common.wgsl")),
    ("lumen_light_sample.wgsl", include_str!("shaders/lumen_light_sample.wgsl")),
    ("lumen_lite.wgsl", include_str!("shaders/lumen_lite.wgsl")),
    ("lumen_material.wgsl", include_str!("shaders/lumen_material.wgsl")),
    ("lumen_mesh_sdf.wgsl", include_str!("shaders/lumen_mesh_sdf.wgsl")),
    ("lumen_reflect.wgsl", include_str!("shaders/lumen_reflect.wgsl")),
    ("lumen_sdf.wgsl", include_str!("shaders/lumen_sdf.wgsl")),
    ("lumen_trace.wgsl", include_str!("shaders/lumen_trace.wgsl")),
    ("lumen_types.wgsl", include_str!("shaders/lumen_types.wgsl")),
    ("lumen_world_cache.wgsl", include_str!("shaders/lumen_world_cache.wgsl")),
    ("lumen_world_cache_common.wgsl", include_str!("shaders/lumen_world_cache_common.wgsl")),
    ("material.wgsl", include_str!("shaders/material.wgsl")),
    ("material_graph.wgsl", include_str!("shaders/material_graph.wgsl")),
    ("material_graph_off.wgsl", include_str!("shaders/material_graph_off.wgsl")),
    ("media_common.wgsl", include_str!("shaders/media_common.wgsl")),
    ("megalights_common.wgsl", include_str!("shaders/megalights_common.wgsl")),
    ("megalights_denoise.wgsl", include_str!("shaders/megalights_denoise.wgsl")),
    ("megalights_sample.wgsl", include_str!("shaders/megalights_sample.wgsl")),
    ("megalights_shade.wgsl", include_str!("shaders/megalights_shade.wgsl")),
    ("megalights_surface.wgsl", include_str!("shaders/megalights_surface.wgsl")),
    ("megalights_rays.wgsl", include_str!("shaders/megalights_rays.wgsl")),
    ("megalights_volume.wgsl", include_str!("shaders/megalights_volume.wgsl")),
    ("megalights_volume_common.wgsl", include_str!("shaders/megalights_volume_common.wgsl")),
    ("noise.wgsl", include_str!("shaders/noise.wgsl")),
    ("oit.wgsl", include_str!("shaders/oit.wgsl")),
    ("page_cache.wgsl", include_str!("shaders/page_cache.wgsl")),
    ("particles.wgsl", include_str!("shaders/particles.wgsl")),
    ("part_owner.wgsl", include_str!("shaders/part_owner.wgsl")),
    ("pick.wgsl", include_str!("shaders/pick.wgsl")),
    ("photo.wgsl", include_str!("shaders/photo.wgsl")),
    ("post_fx.wgsl", include_str!("shaders/post_fx.wgsl")),
    ("raster.wgsl", include_str!("shaders/raster.wgsl")),
    ("resolve_extended.wgsl", include_str!("shaders/resolve_extended.wgsl")),
    ("resolve_extended_off.wgsl", include_str!("shaders/resolve_extended_off.wgsl")),
    ("rvt.wgsl", include_str!("shaders/rvt.wgsl")),
    ("rvt_render.wgsl", include_str!("shaders/rvt_render.wgsl")),
    ("selection.wgsl", include_str!("shaders/selection.wgsl")),
    ("sh.wgsl", include_str!("shaders/sh.wgsl")),
    ("short_range_ao.wgsl", include_str!("shaders/short_range_ao.wgsl")),
    ("shore.wgsl", include_str!("shaders/shore.wgsl")),
    ("skin_palette.wgsl", include_str!("shaders/skin_palette.wgsl")),
    ("skinning.wgsl", include_str!("shaders/skinning.wgsl")),
    ("sky.wgsl", include_str!("shaders/sky.wgsl")),
    ("substrate.wgsl", include_str!("shaders/substrate.wgsl")),
    ("substrate_read.wgsl", include_str!("shaders/substrate_read.wgsl")),
    ("substrate_read_off.wgsl", include_str!("shaders/substrate_read_off.wgsl")),
    ("subsurface.wgsl", include_str!("shaders/subsurface.wgsl")),
    ("subsurface_store.wgsl", include_str!("shaders/subsurface_store.wgsl")),
    ("subsurface_store_off.wgsl", include_str!("shaders/subsurface_store_off.wgsl")),
    ("sw_raster.wgsl", include_str!("shaders/sw_raster.wgsl")),
    ("svt.wgsl", include_str!("shaders/svt.wgsl")),
    ("svt_off.wgsl", include_str!("shaders/svt_off.wgsl")),
    ("stars.wgsl", include_str!("shaders/stars.wgsl")),
    ("tess.wgsl", include_str!("shaders/tess.wgsl")),
    ("tess_core.wgsl", include_str!("shaders/tess_core.wgsl")),
    ("tess_queue.wgsl", include_str!("shaders/tess_queue.wgsl")),
    ("tess_resolve.wgsl", include_str!("shaders/tess_resolve.wgsl")),
    ("texgen.wgsl", include_str!("shaders/texgen.wgsl")),
    ("tonemap.wgsl", include_str!("shaders/tonemap.wgsl")),
    ("translucency.wgsl", include_str!("shaders/translucency.wgsl")),
    ("tsr.wgsl", include_str!("shaders/tsr.wgsl")),
    ("visbuffer.wgsl", include_str!("shaders/visbuffer.wgsl")),
    ("volume_render.wgsl", include_str!("shaders/volume_render.wgsl")),
    ("volume_sim.wgsl", include_str!("shaders/volume_sim.wgsl")),
    ("voxel.wgsl", include_str!("shaders/voxel.wgsl")),
    ("voxel_raster.wgsl", include_str!("shaders/voxel_raster.wgsl")),
    ("voxel_resolve.wgsl", include_str!("shaders/voxel_resolve.wgsl")),
    ("vsm_voxel_raster.wgsl", include_str!("shaders/vsm_voxel_raster.wgsl")),
    ("vsm.wgsl", include_str!("shaders/vsm.wgsl")),
    ("vsm_common.wgsl", include_str!("shaders/vsm_common.wgsl")),
    ("vsm_draw.wgsl", include_str!("shaders/vsm_draw.wgsl")),
    ("vsm_draw_clipped.wgsl", include_str!("shaders/vsm_draw_clipped.wgsl")),
    ("vsm_expand.wgsl", include_str!("shaders/vsm_expand.wgsl")),
    ("vsm_invalidate.wgsl", include_str!("shaders/vsm_invalidate.wgsl")),
    ("vsm_local_common.wgsl", include_str!("shaders/vsm_local_common.wgsl")),
    ("vsm_local_sample.wgsl", include_str!("shaders/vsm_local_sample.wgsl")),
    ("vsm_page.wgsl", include_str!("shaders/vsm_page.wgsl")),
    ("vsm_raster.wgsl", include_str!("shaders/vsm_raster.wgsl")),
    ("vsm_sample.wgsl", include_str!("shaders/vsm_sample.wgsl")),
    ("vsm_moments.wgsl", include_str!("shaders/vsm_moments.wgsl")),
    ("vsm_prefiltered.wgsl", include_str!("shaders/vsm_prefiltered.wgsl")),
    ("vsm_static.wgsl", include_str!("shaders/vsm_static.wgsl")),
    ("vsm_tess.wgsl", include_str!("shaders/vsm_tess.wgsl")),
    ("volume_shadow.wgsl", include_str!("shaders/volume_shadow.wgsl")),
    ("volume_shadow_sample.wgsl", include_str!("shaders/volume_shadow_sample.wgsl")),
    ("water.wgsl", include_str!("shaders/water.wgsl")),
    ("wind.wgsl", include_str!("shaders/wind.wgsl")),
];

pub fn source(name: &str) -> &'static str {
    SOURCES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
        .unwrap_or_else(|| panic!("unknown shader `{name}`"))
}

/// Resolves includes (each file is included at most once) and returns the final WGSL.
pub fn compose(name: &str) -> String {
    compose_variant(name, &[])
}

/// As [`compose`], with the included files of `replacements` (file name → WGSL, whose own
/// includes are resolved too) substituted. Variants swap a default `*_off.wgsl` file for the
/// code of an optional feature, or for generated code (material graphs).
pub fn compose_variant(name: &str, replacements: &[(&str, &str)]) -> String {
    fn expand(name: &str, replacements: &[(&str, &str)], included: &mut Vec<String>, out: &mut String) {
        if included.iter().any(|n| n == name) {
            return;
        }
        included.push(name.to_owned());
        let text = replacements.iter().find(|(n, _)| *n == name).map_or_else(|| source(name), |(_, s)| *s);
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("#include") {
                let target = rest.trim().trim_matches('"');
                expand(target, replacements, included, out);
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    let mut out = String::new();
    expand(name, replacements, &mut Vec::new(), &mut out);
    out
}

/// Whether devices may use f16 shaders (`--no-shader-f16`, `?f16=0` turn it off).
static HALF_ALLOWED: AtomicBool = AtomicBool::new(true);

/// Lets devices created from now on use f16 shaders where the adapter offers them (the default),
/// or not (every `hf` is f32, as on a device without them).
pub fn allow_half_precision(allowed: bool) {
    HALF_ALLOWED.store(allowed, Ordering::Relaxed);
}

pub(crate) fn half_precision_allowed() -> bool {
    HALF_ALLOWED.load(Ordering::Relaxed)
}

/// Whether `device`'s shaders compute `hf` values (half.wgsl) in f16.
pub fn half_precision(device: &wgpu::Device) -> bool {
    device.features().contains(wgpu::Features::SHADER_F16)
}

/// [`compose`] with half.wgsl's f16 variant (`enable` directives moved to the top).
pub fn compose_half(name: &str) -> String {
    let source = compose_variant(name, &[("half.wgsl", source("half_f16.wgsl"))]);
    let (enables, rest): (Vec<&str>, Vec<&str>) = source.lines().partition(|l| l.trim_start().starts_with("enable "));
    let mut out = enables.join("\n");
    out.push('\n');
    out.push_str(&rest.join("\n"));
    out
}

pub fn create_module(device: &wgpu::Device, name: &str) -> wgpu::ShaderModule {
    let source = if half_precision(device) && source(name).contains("#include \"half.wgsl\"") { compose_half(name) } else { compose(name) };
    create_module_from(device, name, source)
}

pub fn create_module_from(device: &wgpu::Device, label: &str, wgsl: String) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(wgsl.into()) })
}

/// Replacements giving the deferred lighting and MegaLights' shading Substrate slabs (see
/// substrate_read.wgsl, subsurface_store.wgsl).
pub fn substrate_lighting() -> [(&'static str, &'static str); 2] {
    [("substrate_read_off.wgsl", source("substrate_read.wgsl")), ("subsurface_store_off.wgsl", source("subsurface_store.wgsl"))]
}

/// The extended G-buffer resolve (resolve_extended.wgsl) with a material graph's code (`None`:
/// slab materials without a graph). `displaces`: the code has `graph_displacement`, which then
/// gives the graph's materials their displaced points' height (displacement_graph_resolve.wgsl).
pub fn extended_resolve(graph: Option<&str>, displaces: bool) -> String {
    let graph = graph.map(|code| format!("const HAS_GRAPH: bool = true;\n{code}"));
    let mut replacements = vec![("resolve_extended_off.wgsl", source("resolve_extended.wgsl"))];
    if let Some(code) = &graph {
        replacements.push(("material_graph_off.wgsl", code.as_str()));
        if displaces {
            replacements.push(("displacement_graphs_off.wgsl", source("displacement_graph_resolve.wgsl")));
        }
    }
    compose_variant("gbuffer.wgsl", &replacements)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses and validates every shader (with includes resolved), so WGSL errors show up in
    /// `cargo test` with precise locations instead of at pipeline creation.
    #[test]
    fn all_shaders_validate() {
        let mut failures = Vec::new();
        for (name, raw) in SOURCES {
            // Include-only files are validated as part of the shaders that include them.
            if !raw.contains("@compute") && !raw.contains("@vertex") {
                continue;
            }
            let source = compose(name);
            let module = match naga::front::wgsl::parse_str(&source) {
                Ok(m) => m,
                Err(e) => {
                    failures.push(format!("{name}: {}", e.emit_to_string(&source)));
                    continue;
                }
            };
            let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
            if let Err(e) = validator.validate(&module) {
                failures.push(format!("{name}: {}", e.emit_to_string(&source)));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    fn validate(label: &str, source: &str) -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(source).map_err(|e| format!("{label}: {}", e.emit_to_string(source)))?;
        let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
        validator.validate(&module).map_err(|e| format!("{label}: {}", e.emit_to_string(source)))?;
        let problems: Vec<String> = source.lines().filter_map(unparenthesised_bitwise).collect();
        if problems.is_empty() {
            Ok(())
        } else {
            Err(format!("{label}: {}", problems.join("; ")))
        }
    }

    /// The f16 variants of the shaders that compute in half precision (none in this round:
    /// TSR and MegaLights' denoiser gained nothing, docs/round8/platform.md), and half.wgsl in
    /// both variants.
    #[test]
    fn half_precision_variants_validate() {
        let half: Vec<&str> = SOURCES.iter().filter(|(_, s)| s.contains("#include \"half.wgsl\"")).map(|(n, _)| *n).collect();
        let mut failures: Vec<String> = half.iter().filter_map(|name| validate(name, &compose_half(name)).err()).collect();
        let kernel = "@compute @workgroup_size(1) fn main() { let c = hf3(0.25) * hf(2.0); let x = f32(c.x) + HF_MAX; }";
        for (variant, file) in [("f32", "half.wgsl"), ("f16", "half_f16.wgsl")] {
            failures.extend(validate(variant, &format!("{}
{kernel}", source(file))).err());
        }
        assert!(failures.is_empty(), "{}", failures.join("
"));
    }

    /// The Substrate variants of the lighting passes and the extended resolve.
    #[test]
    fn substrate_variants_validate() {
        let mut failures = Vec::new();
        for name in ["lighting.wgsl", "megalights_shade.wgsl"] {
            failures.extend(validate(name, &compose_variant(name, &substrate_lighting())).err());
        }
        failures.extend(validate("extended resolve", &extended_resolve(None, false)).err());
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// Every preset graph, and a graph per node type on a texture coordinate path (so its
    /// derivative code is generated too).
    fn test_graphs() -> Vec<helios_material::MaterialGraph> {
        use helios_material::{presets, GraphBuilder, Op};
        let mut graphs: Vec<_> = presets::PRESETS.iter().map(|(id, _)| presets::preset(id).unwrap()).collect();
        for op in Op::palette().into_iter().filter(|op| *op != Op::Output) {
            let output_pin = helios_material::spec(&op).outputs[0];
            let mut g = GraphBuilder::new(op.type_name());
            let node = g.node(op, [0.0, 0.0]);
            let length = g.node(Op::Length, [0.0, 0.0]);
            let scale = g.node(Op::Multiply, [0.0, 0.0]);
            let uv = g.node(Op::TexCoord { set: 1, tiling: [2.0, 2.0] }, [0.0, 0.0]);
            let shifted = g.node(Op::Add, [0.0, 0.0]);
            let sample = g.node(Op::TextureSample { parameter: "Albedo".into() }, [0.0, 0.0]);
            let output = g.node(Op::Output, [0.0, 0.0]);
            g.link(node, output_pin, length, "In")
                .link(length, "Out", scale, "A")
                .value(scale, "B", 0.01)
                .link(uv, "Out", shifted, "A")
                .link(scale, "Out", shifted, "B")
                .link(shifted, "Out", sample, "UV")
                .link(sample, "RGB", output, "BaseColor")
                .link(length, "Out", output, "Roughness")
                .link(sample, "Normal", output, "Normal")
                .link(scale, "Out", output, "Clearcoat")
                .link(sample, "R", output, "Subsurface");
            graphs.push(g.build());
        }
        graphs
    }

    /// Every test graph compiles into a valid resolve pipeline.
    #[test]
    fn material_graph_code_validates() {
        let mut failures = Vec::new();
        for graph in &test_graphs() {
            match helios_material::compile(graph) {
                Ok(compiled) => failures.extend(validate(&graph.name, &extended_resolve(Some(&compiled.wgsl), compiled.displacement.is_some())).err()),
                Err(errors) => failures.push(format!("{}: {errors:?}", graph.name)),
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// Lumen's card material capture validates with every test graph's surface code (the
    /// displacement function left out) and without a graph.
    #[test]
    fn card_capture_variants_validate() {
        let mut failures: Vec<String> = validate("card capture", &crate::cards::capture_shader(None)).err().into_iter().collect();
        for graph in &test_graphs() {
            let compiled = helios_material::compile(graph).unwrap();
            let code = crate::cards::graph_surface_code(&compiled.wgsl);
            assert!(!code.contains("fn graph_displacement"), "{}", graph.name);
            failures.extend(validate(&graph.name, &crate::cards::capture_shader(Some(code))).err());
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// The dicing permutation of displacing graphs (tessellation/graphs.rs) validates in both
    /// tessellation passes, with every node type the displacement context allows on a path to a
    /// texture coordinate (so its filtering code is generated too), as a height and as a vector;
    /// the others are rejected. The resolve of each such graph validates too (its normal and, for
    /// the animated ones, its previous position).
    #[test]
    fn displacement_graph_permutations_validate() {
        use helios_material::{compile, presets, GraphBuilder, Op, DISPLACEMENT_PIN, VECTOR_DISPLACEMENT_PIN};
        let mut graphs: Vec<_> = presets::PRESETS.iter().map(|(id, _)| presets::preset(id).unwrap()).filter(|g| compile(g).is_ok_and(|c| c.displacement.is_some())).collect();
        assert!(graphs.len() >= 3, "the height, vector and animated presets");
        for pin in [DISPLACEMENT_PIN, VECTOR_DISPLACEMENT_PIN] {
            for op in Op::palette().into_iter().filter(|op| *op != Op::Output) {
                let output_pin = helios_material::spec(&op).outputs[0];
                let mut g = GraphBuilder::new(&format!("{} ({pin})", op.type_name()));
                let node = g.node(op, [0.0, 0.0]);
                let length = g.node(Op::Length, [0.0, 0.0]);
                let scale = g.node(Op::Multiply, [0.0, 0.0]);
                let uv = g.node(Op::TexCoord { set: 0, tiling: [2.0, 2.0] }, [0.0, 0.0]);
                let shifted = g.node(Op::Add, [0.0, 0.0]);
                let sample = g.node(Op::TextureSample { parameter: "Height".into() }, [0.0, 0.0]);
                let height = g.node(Op::Add, [0.0, 0.0]);
                let normal = g.node(Op::VertexNormal, [0.0, 0.0]);
                let along = g.node(Op::Multiply, [0.0, 0.0]);
                let output = g.node(Op::Output, [0.0, 0.0]);
                g.link(node, output_pin, length, "In")
                    .link(length, "Out", scale, "A")
                    .value(scale, "B", 0.01)
                    .link(uv, "Out", shifted, "A")
                    .link(scale, "Out", shifted, "B")
                    .link(shifted, "Out", sample, "UV")
                    .link(sample, "Height", height, "A")
                    .link(scale, "Out", height, "B")
                    .link(normal, "Out", along, "A")
                    .link(height, "Out", along, "B");
                if pin == DISPLACEMENT_PIN {
                    g.link(height, "Out", output, pin);
                } else {
                    g.link(along, "Out", output, pin);
                }
                graphs.push(g.build());
            }
        }
        let (mut functions, mut rejected, mut animated, mut failures) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, graph) in graphs.iter().enumerate() {
            let id = i as u32 + 1;
            match compile(graph) {
                Ok(compiled) => {
                    functions.push((id, compiled.displacement_kind, compiled.displacement_function(&format!("displacement_graph_{id}")).unwrap()));
                    if compiled.displacement_animated {
                        animated.push(graph.name.clone());
                    }
                    failures.extend(validate(&graph.name, &extended_resolve(Some(&compiled.wgsl), true)).err());
                }
                Err(errors) => rejected.push((graph.name.clone(), errors[0].message.clone())),
            }
        }
        let names: Vec<&str> = rejected.iter().map(|r| r.0.as_str()).collect();
        let expected = ["VertexColor", "CameraVector", "Fresnel"];
        let both: Vec<String> = [DISPLACEMENT_PIN, VECTOR_DISPLACEMENT_PIN].iter().flat_map(|pin| expected.map(|n| format!("{n} ({pin})"))).collect();
        assert_eq!(names, both, "{rejected:?}");
        for pin in [DISPLACEMENT_PIN, VECTOR_DISPLACEMENT_PIN] {
            for node in ["Time", "Panner"] {
                assert!(animated.contains(&format!("{node} ({pin})")), "{animated:?}");
            }
        }
        // Motion vectors: only an animated graph's resolve evaluates the previous frame's surface,
        // at the previous frame's time.
        let resolve = |graph| extended_resolve(Some(&compile(&graph).unwrap().wgsl), true);
        let (still, moving) = (resolve(presets::mushrooms()), resolve(presets::rippling_cloth()));
        assert!(still.contains("const GRAPH_DISPLACEMENT_ANIMATED: bool = false;") && moving.contains("const GRAPH_DISPLACEMENT_ANIMATED: bool = true;"));
        assert!(moving.contains("if DISPLACEMENT_ANIMATED {\n        let before = displaced_input(instance, objects, normals, uvs, l, tess.time.y);"));
        let code = crate::tessellation::graphs::dispatch_module(&functions);
        assert!(code.contains("displacement_from_height(displacement_graph_") && code.contains("displacement_from_vector(displacement_graph_"));
        for name in ["tess.wgsl", "vsm_tess.wgsl", "lumen_displace.wgsl"] {
            failures.extend(validate(name, &compose_variant(name, &[("displacement_graphs_off.wgsl", &code)])).err());
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// WGSL's grammar gives bitwise (`& | ^`) and shift (`<< >>`) operators no precedence
    /// relative to other binary operators: `a * b ^ c`, `a + b << c` or `a & b == c` must be
    /// parenthesised. naga accepts them anyway; Chrome's Tint rejects the whole shader, which then
    /// only shows up as a broken pipeline in the browser. This catches them in `cargo test`.
    #[test]
    fn bitwise_operators_are_parenthesised() {
        let mut failures = Vec::new();
        for (name, raw) in SOURCES {
            for (number, line) in raw.lines().enumerate() {
                if let Some(problem) = unparenthesised_bitwise(line) {
                    failures.push(format!("{name}:{}: {problem}: {}", number + 1, line.trim()));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn strict_bitwise_check_examples() {
        assert!(unparenthesised_bitwise("let h = a * 3u ^ b;").is_some());
        assert!(unparenthesised_bitwise("let h = a + b << 2u;").is_some());
        assert!(unparenthesised_bitwise("if x & 1u == 0u {").is_some());
        assert!(unparenthesised_bitwise("let h = a & b | c;").is_some());
        assert!(unparenthesised_bitwise("let h = (a * 3u) ^ b;").is_none());
        assert!(unparenthesised_bitwise("let h = a & b & c;").is_none());
        assert!(unparenthesised_bitwise("if (x & 1u) == 0u && y > -2 {").is_none());
        assert!(unparenthesised_bitwise("let v: vec2<u32> = vec2<u32>(a >> 1u, b);").is_none());
        assert!(unparenthesised_bitwise("var<storage, read> x: array<vec4<f32>>;").is_none());
        assert!(unparenthesised_bitwise("atomicAdd(&state.count, 1u);").is_none());
        assert!(unparenthesised_bitwise("let t = -a * b + c; // a ^ b * c").is_none());
    }

    /// Returns a description of the first unparenthesised mix of a bitwise or shift operator
    /// with another binary operator on `line`, if any.
    fn unparenthesised_bitwise(line: &str) -> Option<String> {
        let code = line.split("//").next().unwrap_or("");
        let chars: Vec<char> = code.chars().collect();
        // Per nesting level: the binary operators seen in the current segment.
        let mut stack: Vec<Vec<String>> = vec![Vec::new()];
        let mut template_depth = 0usize;
        let mut previous_is_operand = false;
        let mut previous_ident = String::new();
        let mut i = 0;
        let check = |ops: &Vec<String>| -> Option<String> {
            let bitwise: Vec<&String> = ops.iter().filter(|o| matches!(o.as_str(), "&" | "|" | "^" | "<<" | ">>")).collect();
            if bitwise.is_empty() {
                return None;
            }
            let first = bitwise[0];
            let shifts = matches!(first.as_str(), "<<" | ">>");
            let mixed = ops.iter().any(|o| o != first || (shifts && ops.len() > 1));
            mixed.then(|| format!("`{}` mixed with other operators without parentheses", ops.join(" ")))
        };
        while i < chars.len() {
            let c = chars[i];
            let next = chars.get(i + 1).copied().unwrap_or(' ');
            if c.is_alphanumeric() || c == '_' || c == '.' {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '.') {
                    i += 1;
                }
                previous_ident = chars[start..i].iter().collect();
                previous_is_operand = true;
                continue;
            }
            let templated = |name: &str| {
                name.starts_with("vec") || name.starts_with("mat") || name.starts_with("texture") || matches!(name, "array" | "ptr" | "atomic" | "bitcast" | "var" | "sampler")
            };
            if c == '<' && next != '<' && next != '=' && previous_is_operand && templated(&previous_ident) {
                template_depth += 1;
                previous_is_operand = false;
                i += 1;
                continue;
            }
            if template_depth > 0 && c == '>' {
                template_depth -= 1;
                previous_is_operand = true;
                previous_ident.clear();
                i += 1;
                continue;
            }
            if template_depth > 0 {
                i += 1;
                continue;
            }
            match c {
                '(' | '[' | '{' => {
                    stack.push(Vec::new());
                    previous_is_operand = false;
                }
                ')' | ']' | '}' => {
                    if let Some(ops) = stack.pop() {
                        if let Some(problem) = check(&ops) {
                            return Some(problem);
                        }
                    }
                    if stack.is_empty() {
                        stack.push(Vec::new());
                    }
                    previous_is_operand = true;
                }
                ',' | ';' | ':' => {
                    if let Some(problem) = check(stack.last().unwrap()) {
                        return Some(problem);
                    }
                    stack.last_mut().unwrap().clear();
                    previous_is_operand = false;
                }
                ' ' | '\t' => {}
                _ => {
                    // Operators: the longest match first.
                    let two: String = [c, next].iter().collect();
                    let op = if matches!(two.as_str(), "<<" | ">>" | "<=" | ">=" | "==" | "!=" | "&&" | "||" | "+=" | "-=" | "*=" | "/=" | "|=" | "&=" | "^=" | "->") {
                        i += 1;
                        two
                    } else {
                        c.to_string()
                    };
                    match op.as_str() {
                        // Logical operators and assignments separate the expressions they join.
                        "&&" | "||" | "=" | "+=" | "-=" | "*=" | "/=" | "|=" | "&=" | "^=" | "->" => {
                            if let Some(problem) = check(stack.last().unwrap()) {
                                return Some(problem);
                            }
                            stack.last_mut().unwrap().clear();
                        }
                        _ if previous_is_operand => stack.last_mut().unwrap().push(op),
                        _ => {} // unary - ! ~ & *
                    }
                    previous_is_operand = false;
                }
            }
            i += 1;
        }
        stack.iter().find_map(check)
    }
}
