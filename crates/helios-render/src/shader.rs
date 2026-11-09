//! Embedded WGSL sources with a minimal `#include "file.wgsl"` preprocessor.

const SOURCES: &[(&str, &str)] = &[
    ("alpha_mask.wgsl", include_str!("shaders/alpha_mask.wgsl")),
    ("atmosphere.wgsl", include_str!("shaders/atmosphere.wgsl")),
    ("bloom.wgsl", include_str!("shaders/bloom.wgsl")),
    ("classify.wgsl", include_str!("shaders/classify.wgsl")),
    ("clouds.wgsl", include_str!("shaders/clouds.wgsl")),
    ("clouds_common.wgsl", include_str!("shaders/clouds_common.wgsl")),
    ("common.wgsl", include_str!("shaders/common.wgsl")),
    ("cull.wgsl", include_str!("shaders/cull.wgsl")),
    ("cull_args.wgsl", include_str!("shaders/cull_args.wgsl")),
    ("exposure.wgsl", include_str!("shaders/exposure.wgsl")),
    ("fog.wgsl", include_str!("shaders/fog.wgsl")),
    ("fog_common.wgsl", include_str!("shaders/fog_common.wgsl")),
    ("gbuffer.wgsl", include_str!("shaders/gbuffer.wgsl")),
    ("geometry.wgsl", include_str!("shaders/geometry.wgsl")),
    ("gi.wgsl", include_str!("shaders/gi.wgsl")),
    ("hzb.wgsl", include_str!("shaders/hzb.wgsl")),
    ("light_cull.wgsl", include_str!("shaders/light_cull.wgsl")),
    ("lighting.wgsl", include_str!("shaders/lighting.wgsl")),
    ("local_exposure.wgsl", include_str!("shaders/local_exposure.wgsl")),
    ("lights.wgsl", include_str!("shaders/lights.wgsl")),
    ("lumen_card_light.wgsl", include_str!("shaders/lumen_card_light.wgsl")),
    ("lumen_cards.wgsl", include_str!("shaders/lumen_cards.wgsl")),
    ("lumen_common.wgsl", include_str!("shaders/lumen_common.wgsl")),
    ("lumen_gather.wgsl", include_str!("shaders/lumen_gather.wgsl")),
    ("lumen_light.wgsl", include_str!("shaders/lumen_light.wgsl")),
    ("lumen_lite.wgsl", include_str!("shaders/lumen_lite.wgsl")),
    ("lumen_material.wgsl", include_str!("shaders/lumen_material.wgsl")),
    ("lumen_mesh_sdf.wgsl", include_str!("shaders/lumen_mesh_sdf.wgsl")),
    ("lumen_sdf.wgsl", include_str!("shaders/lumen_sdf.wgsl")),
    ("lumen_trace.wgsl", include_str!("shaders/lumen_trace.wgsl")),
    ("lumen_types.wgsl", include_str!("shaders/lumen_types.wgsl")),
    ("material.wgsl", include_str!("shaders/material.wgsl")),
    ("noise.wgsl", include_str!("shaders/noise.wgsl")),
    ("page_cache.wgsl", include_str!("shaders/page_cache.wgsl")),
    ("pick.wgsl", include_str!("shaders/pick.wgsl")),
    ("post_fx.wgsl", include_str!("shaders/post_fx.wgsl")),
    ("raster.wgsl", include_str!("shaders/raster.wgsl")),
    ("rvt.wgsl", include_str!("shaders/rvt.wgsl")),
    ("rvt_render.wgsl", include_str!("shaders/rvt_render.wgsl")),
    ("selection.wgsl", include_str!("shaders/selection.wgsl")),
    ("sh.wgsl", include_str!("shaders/sh.wgsl")),
    ("shore.wgsl", include_str!("shaders/shore.wgsl")),
    ("skin_palette.wgsl", include_str!("shaders/skin_palette.wgsl")),
    ("skinning.wgsl", include_str!("shaders/skinning.wgsl")),
    ("sky.wgsl", include_str!("shaders/sky.wgsl")),
    ("sw_raster.wgsl", include_str!("shaders/sw_raster.wgsl")),
    ("svt.wgsl", include_str!("shaders/svt.wgsl")),
    ("svt_off.wgsl", include_str!("shaders/svt_off.wgsl")),
    ("stars.wgsl", include_str!("shaders/stars.wgsl")),
    ("texgen.wgsl", include_str!("shaders/texgen.wgsl")),
    ("tonemap.wgsl", include_str!("shaders/tonemap.wgsl")),
    ("translucency.wgsl", include_str!("shaders/translucency.wgsl")),
    ("tsr.wgsl", include_str!("shaders/tsr.wgsl")),
    ("visbuffer.wgsl", include_str!("shaders/visbuffer.wgsl")),
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
    ("vsm_static.wgsl", include_str!("shaders/vsm_static.wgsl")),
    ("water.wgsl", include_str!("shaders/water.wgsl")),
    ("wind.wgsl", include_str!("shaders/wind.wgsl")),
];

fn source(name: &str) -> &'static str {
    SOURCES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
        .unwrap_or_else(|| panic!("unknown shader `{name}`"))
}

/// Resolves includes (each file is included at most once) and returns the final WGSL.
pub fn compose(name: &str) -> String {
    fn expand(name: &str, included: &mut Vec<String>, out: &mut String) {
        if included.iter().any(|n| n == name) {
            return;
        }
        included.push(name.to_owned());
        for line in source(name).lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("#include") {
                let target = rest.trim().trim_matches('"');
                expand(target, included, out);
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    let mut out = String::new();
    expand(name, &mut Vec::new(), &mut out);
    out
}

pub fn create_module(device: &wgpu::Device, name: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(compose(name).into()),
    })
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
