// Virtual shadow map sampling. Requires vsm_common.wgsl and these bindings:
//   vsm: Vsm (uniform), vsm_entries: array<vec4<u32>>, vsm_atlas: texture_depth_2d,
//   shadow_sampler: sampler_comparison (GreaterEqual, reverse-Z).

struct VsmLookup {
    found: bool,
    uv: vec2<f32>,
    depth: f32,
    level: u32,
}

// Finds a resident page for `world`, starting at `level` and falling back to coarser levels.
fn vsm_lookup(world: vec3<f32>, level: u32) -> VsmLookup {
    var r: VsmLookup;
    r.found = false;
    for (var l = level; l < vsm.level_count; l++) {
        let a = vsm_address(world, l);
        if !a.valid {
            continue;
        }
        let e = vsm_entries[a.entry];
        if e.x == INVALID_ID {
            continue;
        }
        let n = f32(vsm.page_texels);
        // Keep the bilinear footprint inside the tile.
        let texel = clamp(a.texel, vec2<f32>(0.5), vec2<f32>(n - 0.5));
        r.found = true;
        r.uv = (vsm_tile(e.x) + texel) / vsm.atlas_size;
        r.depth = a.depth;
        r.level = l;
        return r;
    }
    return r;
}

fn vsm_vogel(index: u32, count: u32, phi: f32) -> vec2<f32> {
    let r = sqrt((f32(index) + 0.5) / f32(count));
    let theta = f32(index) * 2.4 + phi;
    return r * vec2<f32>(cos(theta), sin(theta));
}

// Filtered sun visibility at `world` with receiver normal `n`.
fn vsm_shadow(world: vec3<f32>, n: vec3<f32>, sun: vec3<f32>, noise: f32) -> f32 {
    if vsm.enabled == 0u {
        return 1.0;
    }
    let level = vsm_level(world);
    let texel = vsm.level_info[level].y;
    let n_dot_l = saturate(dot(n, sun));
    let p = world + n * texel * (1.0 + 2.5 * (1.0 - n_dot_l)) + sun * texel * 1.5;
    // Light-space axes in world space (rows of the light view matrix).
    let dx = vec3<f32>(vsm.light_view[0].x, vsm.light_view[1].x, vsm.light_view[2].x);
    let dy = vec3<f32>(vsm.light_view[0].y, vsm.light_view[1].y, vsm.light_view[2].y);
    let taps = 12u;
    let radius = 1.6 * texel;
    var lit = 0.0;
    var count = 0.0;
    for (var i = 0u; i < taps; i++) {
        let o = vsm_vogel(i, taps, noise * 6.2831853) * radius;
        let q = p + dx * o.x - dy * o.y;
        let l = vsm_lookup(q, level);
        if l.found {
            lit += textureSampleCompareLevel(vsm_atlas, shadow_sampler, l.uv, l.depth);
            count += 1.0;
        }
    }
    return select(1.0, lit / count, count > 0.0);
}

fn vsm_debug_color(world: vec3<f32>) -> vec3<f32> {
    let l = vsm_lookup(world, vsm_level(world));
    if !l.found {
        return vec3<f32>(1.0, 0.0, 1.0);
    }
    let palette = array<vec3<f32>, 8>(
        vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.95, 0.55, 0.15), vec3<f32>(0.95, 0.9, 0.2), vec3<f32>(0.35, 0.85, 0.25),
        vec3<f32>(0.2, 0.8, 0.8), vec3<f32>(0.25, 0.4, 0.95), vec3<f32>(0.6, 0.3, 0.9), vec3<f32>(0.9, 0.35, 0.75),
    );
    var p = palette;
    // Page grid lines.
    let a = vsm_address(world, l.level);
    let edge = min(min(a.texel.x, f32(vsm.page_texels) - a.texel.x), min(a.texel.y, f32(vsm.page_texels) - a.texel.y));
    return p[l.level % 8u] * select(1.0, 0.35, edge < 1.5);
}
