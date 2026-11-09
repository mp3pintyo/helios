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
// `density` of the cells holding one; the highest dome under `xz` wins. They fade out where a cell
// shrinks to a couple of pixels (`footprint`, metres), so they never alias. A stone reaches at
// most 0.32 cells beyond its own cell (centre within [0.25, 0.75], radius <= 0.57 cells), so the
// 2x2 cells towards the point's nearest corner hold every stone that can cover it.
fn pebbles(xz: vec2<f32>, cell: f32, density: f32, seed: u32, footprint: f32) -> Pebbles {
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
            let radius = 0.28 + 0.2 * unit_hash(h3);
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
            // Flattened dome: height 0.4 r sqrt(1 - t).
            let dome = sqrt(1.0 - t);
            if dome <= top {
                continue;
            }
            top = dome;
            let g = vec2<f32>(local.x, local.y * stretch);
            let gradient_local = -0.4 * g / (r * max(dome, 0.25));
            p.slope = axis * gradient_local.x + vec2<f32>(-axis.y, axis.x) * gradient_local.y;
            let tone = unit_hash(pcg_hash(h3));
            p.color = mix(vec3<f32>(0.24, 0.23, 0.21), vec3<f32>(0.44, 0.4, 0.34), tone) * (0.8 + 0.2 * smoothstep(0.0, 0.5, dome));
        }
    }
    p.cover = smoothstep(0.0, 0.3, top) * fade;
    p.slope *= fade;
    return p;
}

// Wet albedo of a porous surface: water fills its pores and darkens it (Lagarde's wet surfaces).
fn wet_albedo(albedo: vec3<f32>, wetness: f32) -> vec3<f32> {
    return albedo * mix(1.0, 0.55, wetness);
}

fn shore_surface(landscape: Surface, p: vec3<f32>, macro_normal: vec3<f32>, footprint: f32) -> Surface {
    if water.misc.z <= 0.0 {
        return landscape;
    }
    let level = water_level_at(p.xz);
    let height = p.y - level;
    if height > SHORE_HEIGHT + 0.15 {
        return landscape;
    }
    let wobble = fbm(vec3<f32>(p.xz * 0.6, 3.7), 2u, footprint * 0.6);
    let h = height + 0.15 * wobble;
    var s = landscape;

    // Sediment: sand near the water line, silt in the depths, stones throughout.
    let detail = fbm(vec3<f32>(p.xz * 4.0, 1.3), 2u, footprint * 4.0);
    let silt = 1.0 - smoothstep(-SILT_DEPTH.y, -SILT_DEPTH.x, h + 0.25 * detail);
    var albedo = mix(SAND_COLOR * (0.9 + 0.25 * detail), SILT_COLOR * (0.85 + 0.3 * detail), silt);
    if silt > 0.0 {
        let algae = smoothstep(0.1, 0.45, fbm(vec3<f32>(p.xz * 0.35, 8.1), 2u, footprint * 0.35)) * silt;
        albedo = mix(albedo, ALGAE_COLOR, algae * 0.8);
    }
    var slope = vec2<f32>(0.0);
    // Sand ripples across the waves' travel direction, in the shallows only.
    let ripple_fade = (1.0 - smoothstep(0.02, 0.05, footprint)) * (1.0 - smoothstep(-0.15, 0.05, h)) * (1.0 - silt);
    if ripple_fade > 0.0 {
        let along = dot(p.xz, water.waves.xy) * 52.0 + 2.0 * detail;
        slope += water.waves.xy * (0.25 * cos(along) * ripple_fade);
    }
    let gravel = pebbles(p.xz, 0.05, mix(0.55, 0.08, silt), 17u, footprint);
    let stones = pebbles(p.xz, 0.22, mix(0.1, 0.04, silt), 29u, footprint);
    // Stones in the silt are coated with it.
    let coating = mix(1.0, 0.6, silt);
    albedo = mix(albedo, gravel.color * coating, gravel.cover);
    slope = mix(slope, gravel.slope, gravel.cover);
    albedo = mix(albedo, stones.color * coating, stones.cover);
    slope = mix(slope, stones.slope, stones.cover);

    // Water soaks the ground below the water line and a little above it.
    let wetness = 1.0 - smoothstep(0.01, 0.1, h);
    let sediment = 1.0 - smoothstep(SHORE_HEIGHT - 0.14, SHORE_HEIGHT, h + 0.06 * detail);
    s.albedo = wet_albedo(mix(landscape.albedo, albedo, sediment), wetness);
    // Glossy where a water film covers the ground in air; under water the film is the lake.
    let underwater = 1.0 - smoothstep(-0.05, 0.0, height);
    let wet_roughness = mix(0.3, 0.75, underwater);
    s.roughness = mix(mix(landscape.roughness, 0.8, sediment), wet_roughness, wetness);
    let sediment_normal = normalize(macro_normal + vec3<f32>(-slope.x, 0.0, -slope.y));
    s.normal = normalize(mix(landscape.normal, sediment_normal, sediment));
    return s;
}
