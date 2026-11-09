// The surface BRDF shared by every pass that shades direct light: GGX specular (Smith
// height-correlated visibility, Schlick Fresnel) over a Lambert diffuse lobe, plus Karis' split-sum
// approximation for image-based light; and the Substrate-lite slab built from the same lobes plus
// anisotropic GGX, a clear coat, Charlie sheen, wrapped subsurface diffuse and thin transmission.
// Requires common.wgsl (PI).
//
// Passes call these instead of keeping copies, so the classic deferred lighting and MegaLights
// (megalights_*.wgsl) shade identically and a change of material model lands everywhere at once.

#include "substrate.wgsl"

fn d_ggx(n_dot_h: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn v_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, a: f32) -> f32 {
    let a2 = a * a;
    let gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

fn f_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);
}

// Karis' analytic approximation of the split-sum environment BRDF.
fn env_brdf(f0: vec3<f32>, roughness: f32, n_dot_v: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// The BRDF towards a light in direction `l`, before N.L: `diffuse` is the Lambert lobe's weight
// without the diffuse colour (callers multiply it in, or keep lighting demodulated), `specular`
// the GGX lobe. `a` is the GGX alpha (roughness squared).
struct Brdf {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

fn evaluate_brdf(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, n_dot_v: f32, f0: vec3<f32>, a: f32) -> Brdf {
    let h = normalize(v + l);
    let fresnel = f_schlick(f0, saturate(dot(v, h)));
    let n_dot_l = max(saturate(dot(n, l)), 1e-4);
    var b: Brdf;
    b.specular = d_ggx(saturate(dot(n, h)), a) * v_smith_ggx_correlated(n_dot_v, n_dot_l, a) * fresnel;
    b.diffuse = (vec3<f32>(1.0) - fresnel) / PI;
    return b;
}

// ---- Substrate-lite slabs (substrate.wgsl) --------------------------------------------------------

// Schlick's Fresnel towards F90 (KHR_materials_specular lowers a dielectric's F90).
fn f_schlick_f90(f0: vec3<f32>, f90: f32, v_dot_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(f90) - f0) * pow(1.0 - v_dot_h, 5.0);
}

fn env_brdf_f90(f0: vec3<f32>, f90: f32, roughness: f32, n_dot_v: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + f90 * ab.y;
}

// Anisotropic GGX (Burley 2012) with alphas `at` along the tangent and `ab` along the bitangent.
fn d_ggx_anisotropic(n_dot_h: f32, t_dot_h: f32, b_dot_h: f32, at: f32, ab: f32) -> f32 {
    let d = vec3<f32>(ab * t_dot_h, at * b_dot_h, at * ab * n_dot_h);
    let w = at * ab / dot(d, d);
    return at * ab * w * w / PI;
}

// Its height-correlated Smith visibility (Heitz 2014).
fn v_smith_ggx_anisotropic(at: f32, ab: f32, t_dot_v: f32, b_dot_v: f32, n_dot_v: f32, t_dot_l: f32, b_dot_l: f32, n_dot_l: f32) -> f32 {
    let gv = n_dot_l * length(vec3<f32>(at * t_dot_v, ab * b_dot_v, n_dot_v));
    let gl = n_dot_v * length(vec3<f32>(at * t_dot_l, ab * b_dot_l, n_dot_l));
    return 0.5 / max(gv + gl, 1e-5);
}

// The "Charlie" sheen distribution (Estevez & Kulla 2017; glTF's KHR_materials_sheen) with
// Neubelt & Pettineo's visibility: retro-reflective fuzz on velvet and cloth.
fn d_charlie(n_dot_h: f32, roughness: f32) -> f32 {
    let inv_a = 1.0 / max(roughness * roughness, 1e-3);
    let sin2 = max(1.0 - n_dot_h * n_dot_h, 0.0078125);
    return (2.0 + inv_a) * pow(sin2, inv_a * 0.5) / (2.0 * PI);
}

fn v_neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
    return saturate(1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)));
}

// Directional albedo of the sheen lobe (a fit): what the layers beneath it lose.
fn sheen_albedo(n_dot_v: f32, roughness: f32) -> f32 {
    return saturate((0.05 + 0.35 * roughness) * (1.0 + 1.5 * pow(1.0 - n_dot_v, 3.0)));
}

fn max3(c: vec3<f32>) -> f32 {
    return max(c.r, max(c.g, c.b));
}

// Fresnel reflectance of a clear coat (F0 0.04: IOR 1.5).
fn coat_fresnel(cos_theta: f32) -> f32 {
    return 0.04 + 0.96 * pow(1.0 - cos_theta, 5.0);
}

// A pixel's slab prepared for shading: the base layer's frame and lobes, the coat and sheen on
// top and what they leave of the light reaching the viewer from below.
struct SlabShading {
    features: u32,
    n: vec3<f32>,
    v: vec3<f32>,
    n_dot_v: f32,
    f0: vec3<f32>,
    f90: f32,
    roughness: f32,
    a: f32,
    anisotropy: f32,
    t: vec3<f32>,
    b: vec3<f32>,
    at: f32,
    ab: f32,
    coat: f32,
    coat_n: vec3<f32>,
    coat_roughness: f32,
    coat_n_dot_v: f32,
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    subsurface: f32,
    wrap: f32,
    transmission: f32,
    transmission_color: vec3<f32>,
    // Energy the coat and the sheen let through towards the viewer.
    coat_view: f32,
    sheen_view: f32,
}

// `n` and `roughness` are the G-buffer's: the clear coat's where the slab has one.
fn shade_slab(s: Slab, c: MaterialSlab, n: vec3<f32>, roughness: f32, albedo: vec3<f32>, metallic: f32, v: vec3<f32>) -> SlabShading {
    var sh: SlabShading;
    sh.features = s.features;
    sh.v = v;
    sh.n = n;
    sh.roughness = roughness;
    sh.coat_view = 1.0;
    sh.sheen_view = 1.0;
    if has_feature(s, SLAB_CLEARCOAT) {
        sh.coat = s.clearcoat;
        sh.coat_n = n;
        sh.coat_roughness = roughness;
        sh.coat_n_dot_v = max(dot(n, v), 1e-4);
        sh.coat_view = 1.0 - sh.coat * coat_fresnel(sh.coat_n_dot_v);
        // The base normal may face away where the coat's doesn't (silhouettes).
        let base = s.base_normal;
        sh.n = normalize(base + v * max(0.0, 0.05 - dot(base, v)));
        sh.roughness = clamp(s.base_roughness, 0.045, 1.0);
    }
    sh.n_dot_v = max(dot(sh.n, v), 1e-4);
    sh.a = sh.roughness * sh.roughness;
    sh.f0 = mix(s.f0, albedo, metallic);
    sh.f90 = mix(c.f90, 1.0, metallic);
    if has_feature(s, SLAB_ANISOTROPY) && !has_feature(s, SLAB_CLEARCOAT) {
        let t = normalize(s.tangent - sh.n * dot(sh.n, s.tangent) + vec3<f32>(1e-6, 0.0, 0.0));
        sh.anisotropy = s.anisotropy;
        sh.t = t;
        sh.b = cross(sh.n, t);
        sh.at = max(sh.a * (1.0 + s.anisotropy), 0.002);
        sh.ab = max(sh.a * (1.0 - s.anisotropy), 0.002);
    } else {
        sh.features &= ~SLAB_ANISOTROPY;
    }
    if has_feature(s, SLAB_SHEEN) {
        sh.sheen_color = s.sheen_color;
        sh.sheen_roughness = max(s.sheen_roughness, 0.07);
        sh.sheen_view = 1.0 - max3(sh.sheen_color) * sheen_albedo(sh.n_dot_v, sh.sheen_roughness);
    }
    if has_feature(s, SLAB_SUBSURFACE) {
        sh.subsurface = s.weight;
        sh.wrap = 0.3 * s.weight;
    }
    if has_feature(s, SLAB_TRANSMISSION) {
        sh.transmission = select(s.weight, c.transmission, has_feature(s, SLAB_SUBSURFACE));
        sh.transmission_color = c.transmission_color;
    }
    return sh;
}

// Light from direction `l` on a slab, before the light's radiance: `diffuse` without the diffuse
// colour (callers multiply it in, split off the subsurface share, or keep it demodulated),
// `specular` with every specular lobe (base, sheen, coat) and `back` the thin transmission's
// share of light from behind (callers multiply by diffuse and transmission colours). N.L and the
// layers' energy are included.
struct SlabLighting {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    back: f32,
}

fn evaluate_slab(sh: SlabShading, l: vec3<f32>) -> SlabLighting {
    var out: SlabLighting;
    let h = normalize(sh.v + l);
    let v_dot_h = saturate(dot(sh.v, h));
    let d = dot(sh.n, l);
    let lit = saturate(d);
    let n_dot_l = max(lit, 1e-4);
    let n_dot_h = saturate(dot(sh.n, h));
    let fresnel = f_schlick_f90(sh.f0, sh.f90, v_dot_h);
    var lobe = 0.0;
    if (sh.features & SLAB_ANISOTROPY) != 0u {
        let d_aniso = d_ggx_anisotropic(n_dot_h, dot(sh.t, h), dot(sh.b, h), sh.at, sh.ab);
        lobe = d_aniso * v_smith_ggx_anisotropic(sh.at, sh.ab, dot(sh.t, sh.v), dot(sh.b, sh.v), sh.n_dot_v, dot(sh.t, l), dot(sh.b, l), n_dot_l);
    } else {
        lobe = d_ggx(n_dot_h, sh.a) * v_smith_ggx_correlated(sh.n_dot_v, n_dot_l, sh.a);
    }
    var specular = fresnel * (lobe * lit);
    // Subsurface scattering softens the terminator: wrapped, renormalised N.L.
    let wrapped = saturate((d + sh.wrap) / (1.0 + sh.wrap)) / (1.0 + sh.wrap);
    var diffuse = (vec3<f32>(1.0) - fresnel) * (wrapped / PI);
    out.back = sh.transmission * saturate(-d) / PI;
    if (sh.features & SLAB_SHEEN) != 0u {
        let sheen = sh.sheen_color * (d_charlie(n_dot_h, sh.sheen_roughness) * v_neubelt(sh.n_dot_v, n_dot_l) * lit);
        let under = sh.sheen_view * (1.0 - max3(sh.sheen_color) * sheen_albedo(n_dot_l, sh.sheen_roughness));
        specular = specular * under + sheen;
        diffuse *= under;
    }
    if (sh.features & SLAB_CLEARCOAT) != 0u {
        let coat_l = saturate(dot(sh.coat_n, l));
        let coat_a = sh.coat_roughness * sh.coat_roughness;
        let coat = d_ggx(saturate(dot(sh.coat_n, h)), coat_a) * v_smith_ggx_correlated(sh.coat_n_dot_v, max(coat_l, 1e-4), coat_a) * coat_fresnel(v_dot_h);
        let under = sh.coat_view * (1.0 - sh.coat * coat_fresnel(coat_l));
        specular = specular * under + vec3<f32>(coat * coat_l * sh.coat);
        diffuse *= under;
        out.back *= under;
    }
    out.diffuse = diffuse;
    out.specular = specular;
    return out;
}

// The normal image-based specular light of the base layer reflects about: anisotropic
// surfaces bend it towards the direction their highlights stretch along (Filament's bent
// reflection normal).
fn slab_reflection_normal(sh: SlabShading) -> vec3<f32> {
    if (sh.features & SLAB_ANISOTROPY) == 0u {
        return sh.n;
    }
    let direction = select(sh.t, sh.b, sh.anisotropy >= 0.0);
    let across = cross(direction, sh.v);
    let bent = cross(across, direction);
    return normalize(mix(sh.n, bent, abs(sh.anisotropy) * saturate(5.0 * sh.roughness)));
}

// Image-based specular light of a slab: `base_env` and `coat_env` are the prefiltered radiance
// along the base's and the coat's reflections, `irradiance` the incident diffuse radiance (the
// sheen's broad lobe reflects it).
fn slab_ambient_specular(sh: SlabShading, base_env: vec3<f32>, coat_env: vec3<f32>, irradiance: vec3<f32>) -> vec3<f32> {
    var specular = base_env * env_brdf_f90(sh.f0, sh.f90, sh.roughness, sh.n_dot_v) * sh.sheen_view;
    if (sh.features & SLAB_SHEEN) != 0u {
        specular += irradiance * sh.sheen_color * sheen_albedo(sh.n_dot_v, sh.sheen_roughness);
    }
    specular *= sh.coat_view;
    if (sh.features & SLAB_CLEARCOAT) != 0u {
        specular += coat_env * (env_brdf(vec3<f32>(0.04), sh.coat_roughness, sh.coat_n_dot_v) * sh.coat);
    }
    return specular;
}
