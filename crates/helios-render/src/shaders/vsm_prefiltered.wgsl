// Prefiltered distant sun shadows (vsm_moments.rs, vsm_moments.wgsl; UE 5.8's "Prefiltered
// Distant" virtual shadow maps). Requires vsm_common.wgsl, vsm_sample.wgsl, `lut_sampler` (linear
// clamp) and the moment atlas at binding 150.
//
// Receivers on distant clipmap levels (vsm.refresh.z and beyond, rendered at a lower resolution:
// vsm_level) filter the moment atlas instead of comparing depths: four taps around the receiver,
// each a bilinear lookup of the moments at the mip that matches the receiver's footprint (blended
// between two mips), turned into a visibility by the Hamburger 4MSM. The taps lie on the
// receiver's plane, and the receiver is lifted towards the sun by how far its own surface's depth
// spreads over a tap (its slope times the tap's width), so filtered surfaces do not shadow
// themselves.

@group(0) @binding(150) var vsm_moments: texture_2d<f32>;

// Mips of the moment atlas a lookup may use (vsm_moments.wgsl writes 0..2).
const MOMENT_LODS: f32 = 2.0;
const MOMENT_TILE: f32 = 64.0;
// The moment atlas' mip that holds each tile's depth range.
const MOMENT_RANGE_MIP: i32 = 6;
const MOMENT_TAPS: u32 = 4u;

struct MomentPage {
    found: bool,
    physical: u32,
    texel: vec2<f32>,   // in the page, [0, page_texels)
    depth: f32,
}

// The resident page for `world` at `level` or a coarser one (as vsm_lookup).
fn vsm_moment_lookup(world: vec3<f32>, level: u32) -> MomentPage {
    var r: MomentPage;
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
        r.found = true;
        r.physical = e.x;
        r.texel = a.texel;
        r.depth = a.depth;
        return r;
    }
    return r;
}

// The moments around `texel` of the page in tile `physical`, filtered at mip `lod` (blended
// between its two nearest mips; the sampler's mip filter is nearest).
fn vsm_moments_at(physical: u32, texel: vec2<f32>, lod: f32) -> vec4<f32> {
    let tile = vec2<f32>(f32(physical % vsm.tiles_per_row), f32(physical / vsm.tiles_per_row));
    let first = floor(lod);
    let blend = lod - first;
    var m = vec4<f32>(0.0);
    for (var k = 0u; k < 2u; k++) {
        let w = select(1.0 - blend, blend, k == 1u);
        if w <= 0.0 {
            continue;
        }
        let mip = first + f32(k);
        let size = MOMENT_TILE / exp2(mip);
        // Keep the bilinear footprint inside the tile.
        let pos = clamp(texel * (size / f32(vsm.page_texels)), vec2<f32>(0.5), vec2<f32>(size - 0.5));
        let uv = (tile * size + pos) / (f32(vsm.tiles_per_row) * size);
        m += textureSampleLevel(vsm_moments, lut_sampler, uv, mip) * w;
    }
    return m;
}

// Shadow intensity (0 lit, 1 shadowed) of a receiver at normalised depth `z` from filtered
// moments `moments` (Hamburger 4MSM, Peters and Klein 2015, algorithm 3).
fn msm_hamburger(moments: vec4<f32>, z: f32) -> f32 {
    let b = mix(moments, vec4<f32>(0.0, 0.628, 0.0, 0.628), vsm.prefilter.z);
    // Cholesky factorisation L D L^T of the Hankel matrix ((1, b1, b2), (b1, b2, b3), (b2, b3, b4)),
    // then (1, z, z²) solved for c.
    let d22 = max(b.y - b.x * b.x, 1e-7);
    let l32 = (b.z - b.x * b.y) / d22;
    let d33 = max(b.w - b.y * b.y - l32 * l32 * d22, 1e-9);
    var c1 = 1.0;
    var c2 = z - b.x;
    var c3 = z * z - b.y - l32 * c2;
    c2 /= d22;
    c3 /= d33;
    c2 -= l32 * c3;
    c1 -= b.x * c2 + b.y * c3;
    // The roots z2 <= z3 of c1 + c2 x + c3 x².
    let p = c2 / c3;
    let q = c1 / c3;
    let r = sqrt(max(p * p * 0.25 - q, 0.0));
    let z2 = -0.5 * p - r;
    let z3 = -0.5 * p + r;
    if z <= z2 {
        return 0.0;
    }
    if z <= z3 {
        return saturate((z * z3 - b.x * (z + z3) + b.y) / ((z3 - z2) * (z - z2)));
    }
    return saturate(1.0 - (z2 * z3 - b.x * (z2 + z3) + b.y) / ((z - z2) * (z - z3)));
}

// Sun visibility at `world` (receiver normal `n`): prefiltered on distant levels, the regular
// filtered comparison (vsm_shadow) elsewhere.
fn vsm_shadow_prefiltered(world: vec3<f32>, n: vec3<f32>, sun: vec3<f32>, noise: f32) -> f32 {
    if vsm.enabled == 0u {
        return 1.0;
    }
    let level = vsm_level(world);
    if vsm.prefilter.y == 0.0 || level < vsm.refresh.z {
        return vsm_shadow(world, n, sun, noise);
    }
    let texel = vsm.level_info[level].y;
    let moment_texel = 2.0 * texel;
    let n_dot_l = saturate(dot(n, sun));
    // The receiver's footprint on its surface (stretched at grazing view angles) sets the filter.
    let to_eye = vsm.eye.xyz - world;
    let distance_to_eye = length(to_eye);
    let n_dot_v = abs(dot(n, to_eye)) / max(distance_to_eye, 1e-4);
    let footprint = distance_to_eye * vsm.pixel_angle / max(n_dot_v, 0.25);
    let width = max(moment_texel, footprint);
    let lod = clamp(log2(width / moment_texel), 0.0, MOMENT_LODS);
    let tap_width = moment_texel * exp2(lod);
    // How far the receiver's own depth spreads over half a tap: its slope to the sun (up to 45
    // degrees: steeper receivers are barely lit, and foliage normals say little about the
    // surface) times half the tap.
    let slope = min(sqrt(max(1.0 - n_dot_l * n_dot_l, 0.0)) / max(n_dot_l, 1e-3), 1.0);
    let p = world + n * texel * (1.0 + 2.5 * (1.0 - n_dot_l)) + sun * (texel * 1.5 + slope * 0.5 * tap_width);
    let dx = vec3<f32>(vsm.light_view[0].x, vsm.light_view[1].x, vsm.light_view[2].x);
    let dy = vec3<f32>(vsm.light_view[0].y, vsm.light_view[1].y, vsm.light_view[2].y);
    let radius = 0.75 * width;
    let bleed = vsm.prefilter.w;
    var lit = 0.0;
    var count = 0.0;
    for (var i = 0u; i < MOMENT_TAPS; i++) {
        let o = vsm_vogel(i, MOMENT_TAPS, noise * 6.2831853) * radius;
        let t = dx * o.x - dy * o.y;
        // On the receiver's plane.
        let h = clamp(-dot(n, t) / max(n_dot_l, 0.25), -radius, radius);
        let page = vsm_moment_lookup(p + t + sun * h, level);
        if !page.found {
            continue;
        }
        let tile = vec2<i32>(i32(page.physical % vsm.tiles_per_row), i32(page.physical / vsm.tiles_per_row));
        let range = textureLoad(vsm_moments, tile, MOMENT_RANGE_MIP).xy;
        let z = clamp(1.0 - 2.0 * (page.depth - range.x) / max(range.y - range.x, 1e-7), -1.0, 1.0);
        let shadow = msm_hamburger(vsm_moments_at(page.physical, page.texel, lod), z);
        // Light bleeding reduction: partial visibilities below `bleed` count as shadow.
        lit += saturate((1.0 - shadow - bleed) / (1.0 - bleed));
        count += 1.0;
    }
    return select(1.0, lit / count, count > 0.0);
}
