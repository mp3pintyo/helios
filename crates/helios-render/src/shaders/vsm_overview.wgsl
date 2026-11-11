// Virtual shadow map overview (DEBUG_SHADOW_OVERVIEW, vsm_overview.rs; UE 5.9's "Virtual Shadow
// Map Overview"): four views of the sun's shadow map side by side, each the frame at half size.
//   top left     : clipmap level of the page each pixel samples (palette by level) with the page
//                  grid, shaded by the sun's shadow; distant levels (refreshed at their own rate)
//                  hatched; magenta: no page resident at any level
//   top right    : the page's state this frame: rendered in full (red; new or invalidated), its
//                  dynamic casters only over its static layer (yellow), split (orange: in full,
//                  capturing its static layer), waiting for a refresh the budget deferred (blue,
//                  brighter the longer it has waited), a distant page waiting for its level's turn
//                  (cyan), cached (grey)
//   bottom left  : why the page was last invalidated, fading over INVALIDATION_FADE frames: moving
//                  dynamic casters (wind, skins, Movable instances: yellow), a caster changing layer
//                  (magenta), static casters changed (red: page freed), streaming (cyan); white:
//                  rendered in full this frame without a footprint (a new page, or its level's
//                  light moved)
//   bottom right : the physical page pool (32 x 32 tiles, sun levels in the level palette, local
//                  lights in a colour per light, dark when cached, bright when rendered this frame,
//                  a blue dot for a pending refresh) and bars: requests vs. the pool (red: requests
//                  without a page), pages kept but not requested, pages rendered this frame (the
//                  white tick: the refresh budget), refreshes deferred (blue), held (cyan) and
//                  forced past the budget (red), the pool split between the sun and local lights.
// Drawn over the lighting pass's output only when selected: no cost otherwise.

#include "common.wgsl"
#include "vsm_common.wgsl"
#include "vsm_sample.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(3) var<uniform> vsm: Vsm;
@group(0) @binding(4) var<storage, read> vsm_entries: array<vec4<u32>>;
@group(0) @binding(5) var vsm_atlas: texture_depth_2d;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;
// The VSM info buffer (vsm.wgsl), the page cache's owners and lists (page_cache.wgsl).
@group(0) @binding(7) var<storage, read> info: array<u32>;
@group(0) @binding(8) var<storage, read> owner: array<u32>;
@group(0) @binding(9) var<storage, read> cache_lists: array<u32>;
@group(0) @binding(10) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var gbuffer_albedo: texture_2d<f32>;

// The first local light page entry and a light's entries (vsm_local_common.wgsl).
override LOCAL_BASE: u32 = 262144u;
const SLOT_ENTRIES: u32 = 8190u;
// Page cache header words (page_cache.wgsl).
const H_DIRTY: u32 = 2u;
const H_OVERFLOW: u32 = 4u;
const H_REQUESTED: u32 = 5u;
const H_RETAINED: u32 = 6u;
// Invalidation bits (vsm_invalidate.wgsl).
const HIT_DYNAMIC: u32 = 1u;
const HIT_LAYER: u32 = 2u;
const HIT_STATIC: u32 = 4u;
const HIT_STREAMED: u32 = 8u;
const INVALIDATION_FADE: f32 = 60.0;

const BACKGROUND: vec3<f32> = vec3<f32>(0.03, 0.035, 0.045);

fn level_color(level: u32) -> vec3<f32> {
    let palette = array<vec3<f32>, 8>(
        vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.95, 0.55, 0.15), vec3<f32>(0.95, 0.9, 0.2), vec3<f32>(0.35, 0.85, 0.25),
        vec3<f32>(0.2, 0.8, 0.8), vec3<f32>(0.25, 0.4, 0.95), vec3<f32>(0.6, 0.3, 0.9), vec3<f32>(0.9, 0.35, 0.75),
    );
    var p = palette;
    return p[level % 8u];
}

// A distinct colour per local light slot (golden-ratio hues).
fn slot_color(slot: u32) -> vec3<f32> {
    let h = fract(f32(slot) * 0.618034 + 0.1);
    let k = (vec3<f32>(5.0, 3.0, 1.0) + (h * 6.0)) % vec3<f32>(6.0);
    return vec3<f32>(1.0) - (0.65 * clamp(min(k, vec3<f32>(4.0) - k), vec3<f32>(0.0), vec3<f32>(1.0)));
}

struct Receiver {
    valid: bool,
    world: vec3<f32>,
    shade: f32,
    albedo: vec3<f32>,
}

fn receiver(pixel: vec2<u32>) -> Receiver {
    var r: Receiver;
    let depth = textureLoad(depth_texture, pixel, 0).r;
    r.valid = depth > 0.0;
    if !r.valid {
        return r;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    r.world = world_from_depth(uv, depth, frame.inv_view_proj);
    let n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    let sun = normalize(frame.sun_direction.xyz);
    let n_dot_l = saturate(dot(n, sun));
    var shadow = 0.0;
    if n_dot_l > 0.0 {
        shadow = vsm_shadow(r.world, n, sun, interleaved_gradient_noise(vec2<f32>(pixel), frame.frame_index));
    }
    r.shade = 0.3 + (0.7 * n_dot_l * shadow);
    let albedo = textureLoad(gbuffer_albedo, pixel, 0).rgb;
    r.albedo = albedo * albedo;
    return r;
}

// The page a receiver samples at its own level (INVALID_ID if not resident there) and that level.
struct PageAt {
    entry: u32,
    physical: u32,
    level: u32,
    texel: vec2<f32>,
}

fn page_at(world: vec3<f32>) -> PageAt {
    var p: PageAt;
    p.level = vsm_level(world);
    let a = vsm_address(world, p.level);
    p.entry = select(INVALID_ID, a.entry, a.valid);
    p.physical = INVALID_ID;
    p.texel = a.texel;
    if a.valid {
        p.physical = vsm_entries[a.entry].x;
    }
    return p;
}

fn grid_line(texel: vec2<f32>) -> bool {
    let n = f32(vsm.page_texels);
    return min(min(texel.x, n - texel.x), min(texel.y, n - texel.y)) < 1.5;
}

fn levels_panel(pixel: vec2<u32>, r: Receiver) -> vec3<f32> {
    let l = vsm_lookup(r.world, vsm_level(r.world));
    if !l.found {
        return vec3<f32>(1.0, 0.0, 1.0);
    }
    var color = level_color(l.level) * r.shade;
    let a = vsm_address(r.world, l.level);
    if grid_line(a.texel) {
        color *= 0.35;
    }
    if l.level >= vsm.refresh.z && ((pixel.x + pixel.y) / 3u) % 4u == 0u {
        color = mix(color, vec3<f32>(1.0), 0.35);
    }
    return color;
}

fn state_panel(r: Receiver) -> vec3<f32> {
    let p = page_at(r.world);
    let base = mix(r.albedo, vec3<f32>(0.5), 0.6) * r.shade;
    if p.physical == INVALID_ID {
        return vec3<f32>(0.6, 0.0, 0.6) * r.shade;
    }
    var color = base * 0.6;
    let e = vsm_entries[p.entry];
    if e.w == vsm.frame {
        let kind = page_kind(info[INFO_PAGE_STATE + p.physical]);
        if kind == PAGE_DYNAMIC {
            color = vec3<f32>(1.0, 0.9, 0.15);
        } else if kind == PAGE_SPLIT {
            color = vec3<f32>(1.0, 0.5, 0.1);
        } else {
            color = vec3<f32>(1.0, 0.12, 0.08);
        }
        color *= 0.35 + (0.65 * r.shade);
    } else if info[INFO_PAGE_REFRESH + p.physical] != 0u {
        let waited = f32(vsm.frame - e.w) / f32(max(vsm.refresh.y, 1u));
        let held = p.level >= vsm.refresh.z && !vsm_level_turn(p.level);
        let hue = select(vec3<f32>(0.15, 0.3, 1.0), vec3<f32>(0.1, 0.9, 1.0), held);
        color = hue * (0.4 + (0.6 * saturate(waited))) * (0.4 + (0.6 * r.shade));
    }
    if grid_line(p.texel) {
        color *= 0.5;
    }
    return color;
}

fn reasons_panel(r: Receiver) -> vec3<f32> {
    let p = page_at(r.world);
    var color = mix(r.albedo, vec3<f32>(0.5), 0.6) * r.shade * 0.5;
    if p.physical == INVALID_ID {
        return color;
    }
    let reason = info[INFO_PAGE_REASON + p.physical];
    let age = f32(((vsm.frame & 0xffffffu) - (reason >> 8u)) & 0xffffffu);
    let full = vsm_entries[p.entry].w == vsm.frame && page_kind(info[INFO_PAGE_STATE + p.physical]) == PAGE_FULL;
    if full && ((reason & 0xffu) == 0u || age > 0.0) {
        // Rendered in full without a footprint this frame: new, or its level's light moved.
        color = vec3<f32>(0.9) * (0.4 + (0.6 * r.shade));
    } else if (reason & 0xffu) != 0u && age < INVALIDATION_FADE {
        var hue = vec3<f32>(1.0, 0.9, 0.15);
        if (reason & HIT_STATIC) != 0u {
            hue = vec3<f32>(1.0, 0.12, 0.08);
        } else if (reason & HIT_STREAMED) != 0u {
            hue = vec3<f32>(0.1, 0.9, 1.0);
        } else if (reason & HIT_LAYER) != 0u {
            hue = vec3<f32>(1.0, 0.25, 0.9);
        }
        color = mix(color, hue * (0.4 + (0.6 * r.shade)), 1.0 - (age / INVALIDATION_FADE));
    }
    if grid_line(p.texel) {
        color *= 0.5;
    }
    return color;
}

// A horizontal bar at `uv` (0..1 inside the bar's box): segments of `values` (cumulative, in
// units of `scale`) coloured `colors`, a tick every 1/8 of the scale.
fn bar(uv: vec2<f32>, values: vec4<f32>, colors: array<vec3<f32>, 4>, scale: f32, marker: f32) -> vec3<f32> {
    var c = colors;
    let x = uv.x * scale;
    var color = vec3<f32>(0.12);
    var start = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let end = start + values[i];
        if x >= start && x < end {
            color = c[i];
        }
        start = end;
    }
    let tick = fract(uv.x * 8.0);
    if (tick < 0.004 || tick > 0.996) && uv.y > 0.7 {
        color = vec3<f32>(0.6);
    }
    if marker > 0.0 && abs(x - marker) < scale * 0.004 {
        color = vec3<f32>(1.0);
    }
    return color;
}

fn pool_panel(q: vec2<f32>, size: vec2<f32>) -> vec3<f32> {
    let margin = size.y * 0.05;
    let grid = min(size.x * 0.55, size.y - (2.0 * margin));
    let g = (q - vec2<f32>(margin)) / grid;
    if all(g >= vec2<f32>(0.0)) && all(g < vec2<f32>(1.0)) {
        let cell = vec2<u32>(g * 32.0);
        let f = fract(g * 32.0);
        // Cell borders where cells are large enough to show them (not at low internal
        // resolutions, where they would alias).
        if grid >= 320.0 && any(f < vec2<f32>(0.08)) {
            return vec3<f32>(0.06);
        }
        let p = cell.y * 32u + cell.x;
        let entry = owner[p];
        if entry == INVALID_ID {
            return vec3<f32>(0.0);
        }
        var color = vec3<f32>(0.0);
        if entry < LOCAL_BASE {
            color = level_color(entry / (vsm.level_pages * vsm.level_pages));
        } else {
            color = slot_color((entry - LOCAL_BASE) / SLOT_ENTRIES);
        }
        let e = vsm_entries[entry];
        if e.w == vsm.frame {
            color = mix(color, vec3<f32>(1.0), 0.25);
        } else if e.z == vsm.frame {
            color *= 0.45;
        } else {
            color *= 0.12; // kept for reuse, not requested this frame
        }
        if info[INFO_PAGE_REFRESH + p] != 0u && length(f - vec2<f32>(0.5)) < 0.25 {
            color = vec3<f32>(0.2, 0.4, 1.0);
        }
        return color;
    }
    // Bars to the right of the grid.
    let x0 = grid + (2.0 * margin);
    let w = size.x - x0 - margin;
    let row = (q.y - margin) / ((size.y - (2.0 * margin)) / 5.0);
    if q.x < x0 || q.x >= x0 + w || row < 0.0 || row >= 5.0 || fract(row) > 0.7 {
        return BACKGROUND;
    }
    let uv = vec2<f32>((q.x - x0) / w, fract(row) / 0.7);
    let pool = f32(vsm.physical_count);
    let white = vec3<f32>(0.9);
    let red = vec3<f32>(1.0, 0.15, 0.1);
    let blue = vec3<f32>(0.2, 0.4, 1.0);
    let cyan = vec3<f32>(0.1, 0.9, 1.0);
    var color = BACKGROUND;
    switch u32(row) {
        case 0u: {
            // Requests (white: with a page, red: without), on a scale of twice the pool; the
            // marker is the pool.
            let requested = f32(cache_lists[H_REQUESTED]);
            let overflow = f32(cache_lists[H_OVERFLOW]);
            color = bar(uv, vec4<f32>(requested - overflow, overflow, 0.0, 0.0), array<vec3<f32>, 4>(white, red, red, red), 2.0 * pool, pool);
        }
        case 1u: {
            // Pages kept for reuse although not requested.
            color = bar(uv, vec4<f32>(f32(cache_lists[H_RETAINED]), 0.0, 0.0, 0.0), array<vec3<f32>, 4>(vec3<f32>(0.45), red, red, red), 2.0 * pool, pool);
        }
        case 2u: {
            // Pages rendered this frame on a scale of 256; the marker is the refresh budget.
            color = bar(uv, vec4<f32>(f32(cache_lists[H_DIRTY]), 0.0, 0.0, 0.0), array<vec3<f32>, 4>(vec3<f32>(1.0, 0.5, 0.1), red, red, red), 256.0, f32(vsm.refresh.x));
        }
        case 3u: {
            let deferred = f32(info[INFO_REFRESH_DEFERRED]);
            let held = f32(info[INFO_REFRESH_HELD]);
            let forced = f32(info[INFO_REFRESH_FORCED]);
            color = bar(uv, vec4<f32>(deferred, held, forced, 0.0), array<vec3<f32>, 4>(blue, cyan, red, red), 256.0, 0.0);
        }
        default: {
            // The pool's pages: sun (orange), local lights (purple), free (dark).
            var sun = 0.0;
            var lights = 0.0;
            for (var p = 0u; p < vsm.physical_count; p++) {
                let entry = owner[p];
                if entry != INVALID_ID {
                    if entry < LOCAL_BASE {
                        sun += 1.0;
                    } else {
                        lights += 1.0;
                    }
                }
            }
            color = bar(uv, vec4<f32>(sun, lights, 0.0, 0.0), array<vec3<f32>, 4>(vec3<f32>(0.95, 0.55, 0.15), vec3<f32>(0.6, 0.3, 0.9), red, red), pool, 0.0);
        }
    }
    return color;
}

@compute @workgroup_size(8, 8)
fn overview(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if any(gid.xy >= size) {
        return;
    }
    let half_size = (size + 1u) / 2u;
    let right = gid.x >= half_size.x;
    let lower = gid.y >= half_size.y;
    let q = gid.xy - (vec2<u32>(select(0u, half_size.x, right), select(0u, half_size.y, lower)));
    var color = BACKGROUND;
    // A dark seam between the panels.
    if q.x < 2u || q.y < 2u {
        textureStore(output, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    if right && lower {
        color = pool_panel(vec2<f32>(q), vec2<f32>(half_size));
    } else {
        let pixel = min(q * 2u, size - 1u);
        let r = receiver(pixel);
        if r.valid {
            if !right && !lower {
                color = levels_panel(pixel, r);
            } else if right {
                color = state_panel(r);
            } else {
                color = reasons_panel(r);
            }
        }
    }
    textureStore(output, gid.xy, vec4<f32>(color, 1.0));
}
