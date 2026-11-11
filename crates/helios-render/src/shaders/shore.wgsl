// Lake beds and shores: the landscape material where water bodies cover or border it, as UE's
// landscapes blend towards wet sediment under water bodies. Requires common.wgsl, noise.wgsl,
// material.wgsl and water.wgsl (with its `water` uniform).
//
// Applied per pixel by the G-buffer resolve after the landscape material (runtime virtual
// texture or direct layering), at the pixel's exact height: the virtual texture's heightfield is
// coarser than the mesh, so a mask baked into its pages would miss the real water line.
//
// Zones by height above the water level (each boundary wanders with noise):
//   above ~0.2 m        landscape unchanged
//   shore               sand and gravel, from just above the water down to ~0.6 m
//   deeper              fine, dark silt with scattered stones and patches of algae
//   wet band            everything below ~0.1 m above the water is soaked: darker and glossy
//                       above the water (under it a wet surface looks matte: the water film
//                       matches it)

// The water line: the landscape is replaced up to this height above the water.
const SHORE_HEIGHT: f32 = 0.18;
// Sand gives way to silt over this depth range.
const SILT_DEPTH: vec2<f32> = vec2<f32>(0.4, 1.1);
const SAND_COLOR: vec3<f32> = vec3<f32>(0.42, 0.37, 0.29);
const SILT_COLOR: vec3<f32> = vec3<f32>(0.19, 0.17, 0.12);
const ALGAE_COLOR: vec3<f32> = vec3<f32>(0.11, 0.14, 0.06);
// A river's bed: a coarse gravel matrix (its current carries the sand away) and the olive-brown
// film of algae and fine sediment (periphyton) stones grow under running water, thicker deeper.
const GRAVEL_COLOR: vec3<f32> = vec3<f32>(0.25, 0.22, 0.17);
const BIOFILM_COLOR: vec3<f32> = vec3<f32>(0.13, 0.13, 0.065);

struct Pebbles {
    // 0 between stones, up to 1 on top of one.
    cover: f32,
    // Height gradient (dimensionless) of the stone under the point.
    slope: vec2<f32>,
    color: vec3<f32>,
}

fn unit_hash(h: u32) -> f32 {
    return f32(h & 0xffffu) / 65535.0;
}

// Rounded, elongated, irregular stones on a jittered grid of `cell`-metre cells, a fraction
// `density` of the cells holding one; the highest dome under `xz` wins. Sizes run from many
// small to a few large, and every stone lies partly buried: it sinks by its own share plus
// `bury` (0-1 of its height, varying over the ground), so stones gather where the sediment is
// thin, vanish where it is thick and show irregular outlines where it varies across them. They
// fade out where a cell shrinks to a couple of pixels (`footprint`, metres), so they never alias.
// A stone reaches at most 0.32 cells beyond its own cell (centre within [0.25, 0.75], radius
// <= 0.57 cells), so the 2x2 cells towards the point's nearest corner hold every stone that can
// cover it.
fn pebbles(xz: vec2<f32>, cell: f32, density: f32, seed: u32, footprint: f32, bury: f32) -> Pebbles {
    var p: Pebbles;
    p.cover = 0.0;
    p.slope = vec2<f32>(0.0);
    p.color = vec3<f32>(0.3, 0.29, 0.27);
    let fade = 1.0 - smoothstep(0.2, 0.5, footprint / cell);
    if fade <= 0.0 {
        return p;
    }
    let q = xz / cell;
    let base = floor(q);
    let toward = select(vec2<i32>(-1), vec2<i32>(1), q - base >= vec2<f32>(0.5));
    var top = 0.0;
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let c = vec2<i32>(base) + vec2<i32>(i, j) * toward;
            let h = pcg_hash((bitcast<u32>(c.x) * 73856093u) ^ (bitcast<u32>(c.y) * 19349663u) ^ seed);
            if unit_hash(h) > density {
                continue;
            }
            let h2 = pcg_hash(h);
            let h3 = pcg_hash(h2);
            let center = vec2<f32>(c) + 0.25 + 0.5 * vec2<f32>(unit_hash(h2), unit_hash(h2 >> 16u));
            let radius = 0.18 + 0.3 * pow(unit_hash(h3), 1.5);
            let angle = 6.2831853 * unit_hash(h3 >> 16u);
            let axis = vec2<f32>(cos(angle), sin(angle));
            let d = q - center;
            // Rotate into the stone's frame and squash one axis: elongated pebbles.
            let stretch = 1.0 + 0.6 * unit_hash(h >> 16u);
            let local = vec2<f32>(dot(d, axis), dot(d, vec2<f32>(-axis.y, axis.x)) * stretch);
            // A lumpy outline: two angular harmonics on the radius.
            let around = atan2(local.y, local.x);
            let lumps = 1.0 + 0.12 * sin(3.0 * around + 6.2831853 * unit_hash(h3 >> 8u)) + 0.07 * sin(5.0 * around + 6.2831853 * unit_hash(h2 >> 8u));
            let r = radius * lumps;
            let t = dot(local, local) / (r * r);
            if t >= 1.0 {
                continue;
            }
            // Flattened dome: height 0.4 r sqrt(1 - t), sunk into the sediment.
            let dome = sqrt(1.0 - t) - saturate(0.35 * unit_hash(pcg_hash(h2 + 7u)) + bury);
            if dome <= top {
                continue;
            }
            top = dome;
            let g = vec2<f32>(local.x, local.y * stretch);
            let gradient_local = -0.4 * g / (r * max(dome, 0.25));
            p.slope = axis * gradient_local.x + vec2<f32>(-axis.y, axis.x) * gradient_local.y;
            let tone = unit_hash(pcg_hash(h3));
            let hue = mix(vec3<f32>(1.08, 0.97, 0.84), vec3<f32>(0.9, 0.97, 1.05), unit_hash(pcg_hash(h3 + 11u)));
            p.color = mix(vec3<f32>(0.17, 0.165, 0.15), vec3<f32>(0.42, 0.39, 0.33), tone) * hue * (0.8 + 0.2 * smoothstep(0.0, 0.5, dome));
        }
    }
    p.cover = smoothstep(0.0, 0.3, top) * fade;
    p.slope *= fade;
    return p;
}

// Water surface of the simulated river over `xz` (its banks a few cells from the water too), or
// very low (river.rs: `surface_step`). Needs `river_surface`.
fn river_level_at(xz: vec2<f32>) -> f32 {
    if water.river.w <= 0.0 {
        return -1.0e9;
    }
    let c = vec2<i32>(floor((xz - water.river.xy) * water.river.z));
    if any(c < vec2<i32>(0)) || any(c >= vec2<i32>(textureDimensions(river_surface))) {
        return -1.0e9;
    }
    return textureLoad(river_surface, c, 0).x;
}

// Wet albedo of a porous surface: water fills its pores and darkens it (Lagarde's wet surfaces).
fn wet_albedo(albedo: vec3<f32>, wetness: f32) -> vec3<f32> {
    return albedo * mix(1.0, 0.55, wetness);
}

fn shore_surface(landscape: Surface, p: vec3<f32>, macro_normal: vec3<f32>, footprint: f32) -> Surface {
    if water.misc.z <= 0.0 && water.river.w <= 0.0 {
        return landscape;
    }
    let lake = water_level_at(p.xz);
    let river = river_level_at(p.xz);
    let level = max(lake, river);
    let height = p.y - level;
    if height > SHORE_HEIGHT + 0.15 {
        return landscape;
    }
    let wobble = fbm(vec3<f32>(p.xz * 0.6, 3.7), 2u, footprint * 0.6);
    let h = height + 0.15 * wobble;
    var s = landscape;

    // Sediment: sand near the water line, silt in the depths, stones throughout; a river's bed
    // (where its level is the higher) is gravel and cobbles instead of sand.
    let stream = select(0.0, 1.0, river > lake);
    let detail = fbm(vec3<f32>(p.xz * 4.0, 1.3), 2u, footprint * 4.0);
    let silt = 1.0 - smoothstep(-SILT_DEPTH.y, -SILT_DEPTH.x, h + 0.25 * detail);
    let shallow = mix(SAND_COLOR * (0.9 + 0.25 * detail), GRAVEL_COLOR * (0.85 + 0.3 * detail), stream);
    var albedo = mix(shallow, SILT_COLOR * (0.85 + 0.3 * detail), silt);
    if silt > 0.0 {
        let algae = smoothstep(0.1, 0.45, fbm(vec3<f32>(p.xz * 0.35, 8.1), 2u, footprint * 0.35)) * silt;
        albedo = mix(albedo, ALGAE_COLOR, algae * 0.8);
    }
    var slope = vec2<f32>(0.0);
    // Sand ripples across the waves' travel direction, in the shallows only (not in a river's bed:
    // the lake's waves don't reach it, its current leaves gravel).
    let ripple_fade = (1.0 - smoothstep(0.02, 0.05, footprint)) * (1.0 - smoothstep(-0.15, 0.05, h)) * (1.0 - silt) * select(1.0, 0.0, river > lake);
    if ripple_fade > 0.0 {
        let along = dot(p.xz, water.waves.xy) * 52.0 + 2.0 * detail;
        slope += water.waves.xy * (0.25 * cos(along) * ripple_fade);
    }
    // Gravel collects in patches between bare sediment; stones sink in by varying amounts. A
    // river's current packs its bed with them (an armoured bed: cobbles lying on top).
    let gravel_bury = smoothstep(-0.2, 0.35, detail + 0.8 * wobble) * mix(1.0, 0.55, stream);
    let gravel = pebbles(p.xz, 0.05, mix(mix(0.6, 0.1, silt), 0.75, stream), 17u, footprint, gravel_bury);
    let stones = pebbles(p.xz, 0.22, mix(mix(0.14, 0.05, silt), 0.3, stream), 29u, footprint, mix(0.25, 0.1, stream) + 0.3 * detail);
    // Stones in the silt are coated with it.
    let coating = mix(1.0, 0.6, silt);
    albedo = mix(albedo, gravel.color * coating, gravel.cover);
    slope = mix(slope, gravel.slope, gravel.cover);
    albedo = mix(albedo, stones.color * coating, stones.cover);
    slope = mix(slope, stones.slope, stones.cover);
    // Under running water everything wears a thin film of algae and silt: browner and greener
    // the deeper (and in patches), so the bed darkens and greens with depth.
    let film = stream * smoothstep(0.0, 0.4, -h) * (0.45 + 0.35 * smoothstep(-0.2, 0.3, wobble + detail));
    albedo = mix(albedo, BIOFILM_COLOR * (0.8 + 0.4 * detail), film);

    // Water soaks the ground below the water line and a little above it.
    let wetness = 1.0 - smoothstep(0.01, 0.1, h);
    // Sediment settles on level ground only: steep faces (the step under the cave's mouth, a
    // channel wall) stay the rock they are, just wet - its pattern lies in the horizontal plane
    // and would smear down a wall in vertical streaks.
    let settles = smoothstep(0.35, 0.65, macro_normal.y);
    let sediment = (1.0 - smoothstep(SHORE_HEIGHT - 0.14, SHORE_HEIGHT, h + 0.06 * detail)) * settles;
    s.albedo = wet_albedo(mix(landscape.albedo, albedo, sediment), wetness);
    // Wet stones deepen in colour as well as darken (no air-filled pores scattering white).
    let grey = dot(s.albedo, vec3<f32>(0.3, 0.59, 0.11));
    s.albedo = max(mix(vec3<f32>(grey), s.albedo, 1.0 + 0.35 * wetness * stream), vec3<f32>(0.0));
    // Glossy where a water film covers the ground in air; under water the film is the lake.
    let underwater = 1.0 - smoothstep(-0.05, 0.0, height);
    let wet_roughness = mix(0.3, 0.75, underwater);
    s.roughness = mix(mix(landscape.roughness, 0.8, sediment), wet_roughness, wetness);
    let sediment_normal = normalize(macro_normal + vec3<f32>(-slope.x, 0.0, -slope.y));
    s.normal = normalize(mix(landscape.normal, sediment_normal, sediment));
    return s;
}
