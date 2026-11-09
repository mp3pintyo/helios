// Local light virtual shadow map sampling. Requires vsm_common.wgsl, vsm_sample.wgsl,
// vsm_local_common.wgsl, the sun's sampling bindings (vsm, vsm_entries, vsm_atlas,
// shadow_sampler) and `local_vsm: LocalVsm`.

struct LocalLookup {
    found: bool,
    uv: vec2<f32>,
    depth: f32,
}

// Finds a resident page for `world` starting at `mip` and falling back to coarser mips (pixels
// request the coarsest page of every face they see). Not found outside a spot light's face.
fn local_lookup(slot: u32, world: vec3<f32>, mip: u32) -> LocalLookup {
    var r: LocalLookup;
    r.found = false;
    let face = local_face(slot, world);
    let clip = local_vsm.slots[slot].view_proj[face] * vec4<f32>(world, 1.0);
    let uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    if clip.w <= 0.0 || any(uv < vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
        return r;
    }
    let page = local_page(slot, face, uv, mip);
    r.found = page.found;
    r.uv = local_page_uv(page, uv);
    r.depth = clip.z / clip.w;
    return r;
}

// A page of a face found resident (`found`) at `mip` or coarser: its page coordinate and tile.
struct LocalPage {
    found: bool,
    mip: u32,
    page: vec2<f32>,
    tile: vec2<f32>,
}

// The resident page holding face position `uv`, from `mip` on towards coarser mips.
fn local_page(slot: u32, face: u32, uv: vec2<f32>, mip: u32) -> LocalPage {
    var p: LocalPage;
    p.found = false;
    for (var m = mip; m < LOCAL_MIPS; m++) {
        let pages = f32(local_mip_pages(m));
        let page = min(floor(uv * pages), vec2<f32>(pages - 1.0));
        let e = vsm_entries[local_entry(slot, face, m, vec2<u32>(page))];
        if e.x != INVALID_ID {
            p.found = true;
            p.mip = m;
            p.page = page;
            p.tile = vsm_tile(e.x);
            return p;
        }
    }
    return p;
}

// Atlas position of face position `uv` inside page `p`, kept inside the tile by a texel so that
// 2x2 footprints (bilinear comparisons, gathers) do not read the neighbouring tile.
fn local_page_uv(p: LocalPage, uv: vec2<f32>) -> vec2<f32> {
    let n = f32(vsm.page_texels);
    let texel = (uv * f32(local_mip_pages(p.mip)) - p.page) * n;
    return (p.tile + clamp(texel, vec2<f32>(1.0), vec2<f32>(n - 1.0))) / vsm.atlas_size;
}

// The filter disk of one receiver. Taps are offsets perpendicular to the light, moved along the
// light direction back onto the receiver's tangent plane so that they compare against the
// receiver's own depth there (receiver-plane depth bias, bounded at grazing angles). A tap's
// clip position is then affine in its offset: `center + dx * o.x + dy * o.y`, and taps inside
// the centre's page reuse its lookup - only taps across a page or face border look up again.
struct LocalFilter {
    slot: u32,
    face: u32,
    mip: u32,
    p: vec3<f32>,
    dx_world: vec3<f32>,
    dy_world: vec3<f32>,
    center: vec4<f32>,
    dx: vec4<f32>,
    dy: vec4<f32>,
    page: LocalPage,
}

fn local_filter(slot: u32, r: LocalReceiver, n: vec3<f32>) -> LocalFilter {
    var f: LocalFilter;
    f.slot = slot;
    f.face = local_face(slot, r.p);
    f.mip = r.mip;
    f.p = r.p;
    let n_dot_l = dot(n, r.l);
    let slope = 1.0 / select(min(n_dot_l, -0.3), max(n_dot_l, 0.3), n_dot_l >= 0.0);
    let t = local_tangent(r.l);
    let b = cross(r.l, t);
    f.dx_world = t - r.l * (dot(t, n) * slope);
    f.dy_world = b - r.l * (dot(b, n) * slope);
    let view_proj = local_vsm.slots[slot].view_proj[f.face];
    f.center = view_proj * vec4<f32>(r.p, 1.0);
    f.dx = view_proj * vec4<f32>(f.dx_world, 0.0);
    f.dy = view_proj * vec4<f32>(f.dy_world, 0.0);
    let uv = vec2<f32>(f.center.x, -f.center.y) / max(f.center.w, 1e-6) * 0.5 + 0.5;
    f.page = local_page(slot, f.face, saturate(uv), f.mip);
    return f;
}

// The atlas position and receiver depth of the tap at offset `o` (world units on the disk).
fn local_filter_tap(f: LocalFilter, o: vec2<f32>) -> LocalLookup {
    let clip = f.center + f.dx * o.x + f.dy * o.y;
    let uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    if clip.w <= 0.0 || any(uv < vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
        // On another cube face (or outside a spot light's): the general lookup.
        return local_lookup(f.slot, f.p + f.dx_world * o.x + f.dy_world * o.y, f.mip);
    }
    var page = f.page;
    if !page.found || any(floor(uv * f32(local_mip_pages(page.mip))) != page.page) {
        page = local_page(f.slot, f.face, uv, f.mip);
    }
    var s: LocalLookup;
    s.found = page.found;
    s.uv = local_page_uv(page, uv);
    s.depth = clip.z / clip.w;
    return s;
}

// Contact-hardening soft shadow (PCSS) of the light in `slot` at `world` with normal `n`: a
// blocker search over the light's source radius estimates the penumbra from the average blocker
// distance, which then sets the radius of `taps` filtered depth comparisons - crisp where an
// occluder touches the receiver (pew legs, candle holders), soft far from it.
fn local_vsm_shadow(slot: u32, world: vec3<f32>, n: vec3<f32>, source_radius: f32, noise: f32, taps: u32) -> f32 {
    let r = local_receiver(slot, world, n, 0u);
    let f = local_filter(slot, r, n);
    let near = local_vsm.slots[slot].params.y;
    let phi = noise * 6.2831853;
    let max_radius = LOCAL_MAX_FILTER_TEXELS * r.texel;

    // Blocker search: average face depth of the texels nearer to the light than the receiver.
    let search = clamp(source_radius, 2.0 * r.texel, max_radius);
    let search_taps = max(taps / 3u, 2u);
    var blocker_w = 0.0;
    var blockers = 0.0;
    for (var i = 0u; i < search_taps; i++) {
        let s = local_filter_tap(f, vsm_vogel(i, search_taps, phi) * search);
        if !s.found {
            continue;
        }
        // The 2x2 texels a bilinear comparison would read (loaded: WebGPU allows no filtering
        // sampler on depth textures, so no gather).
        let corner = vec2<i32>(floor(s.uv * vsm.atlas_size - 0.5));
        let receiver_w = near / max(s.depth, 1e-9) - 2.0 * r.texel;
        for (var k = 0; k < 4; k++) {
            let depth = textureLoad(vsm_atlas, corner + vec2<i32>(k & 1, k >> 1u), 0);
            let w = near / max(depth, 1e-9);
            if w < receiver_w {
                blocker_w += w;
                blockers += 1.0;
            }
        }
    }
    if blockers == 0.0 {
        return 1.0;
    }
    let blocker = blocker_w / blockers;
    let penumbra = clamp(source_radius * (r.w - blocker) / max(blocker, near), r.texel, max_radius);

    var lit = 0.0;
    var count = 0.0;
    for (var i = 0u; i < taps; i++) {
        let s = local_filter_tap(f, vsm_vogel(i, taps, phi + 1.3) * penumbra);
        if s.found {
            lit += textureSampleCompareLevel(vsm_atlas, shadow_sampler, s.uv, s.depth);
            count += 1.0;
        }
    }
    return select(1.0, lit / count, count > 0.0);
}

// One bilinear depth comparison, `mip_bias` mips coarser than the screen footprint asks for: for
// lights that contribute little to a pixel, and for volumetric fog (froxels span many pixels;
// `n` = 0: no normal offset).
fn local_vsm_visibility(slot: u32, world: vec3<f32>, n: vec3<f32>, mip_bias: u32) -> f32 {
    let r = local_receiver(slot, world, n, mip_bias);
    let s = local_lookup(slot, r.p, r.mip);
    if !s.found {
        return 1.0;
    }
    return textureSampleCompareLevel(vsm_atlas, shadow_sampler, s.uv, s.depth);
}

// Debug view: the mip a pixel samples (one colour per mip, darker where a coarser mip stands in
// for a missing page), page borders, magenta where nothing is resident.
fn local_vsm_debug_color(slot: u32, world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let r = local_receiver(slot, world, n, 0u);
    let face = local_face(slot, r.p);
    for (var m = r.mip; m < LOCAL_MIPS; m++) {
        let a = local_address(slot, face, r.p, m);
        if !a.valid {
            break;
        }
        if vsm_entries[a.entry].x == INVALID_ID {
            continue;
        }
        let palette = array<vec3<f32>, 6>(
            vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.95, 0.6, 0.15), vec3<f32>(0.9, 0.9, 0.2),
            vec3<f32>(0.3, 0.85, 0.3), vec3<f32>(0.2, 0.75, 0.9), vec3<f32>(0.45, 0.35, 0.95),
        );
        var p = palette;
        let edge = min(min(a.texel.x, f32(vsm.page_texels) - a.texel.x), min(a.texel.y, f32(vsm.page_texels) - a.texel.y));
        let fallback = select(1.0, 0.45, m != r.mip);
        return p[m] * fallback * select(1.0, 0.35, edge < 1.5);
    }
    return vec3<f32>(1.0, 0.0, 1.0);
}
