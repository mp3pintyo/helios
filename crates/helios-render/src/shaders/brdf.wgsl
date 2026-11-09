// The surface BRDF shared by every pass that shades direct light: GGX specular (Smith
// height-correlated visibility, Schlick Fresnel) over a Lambert diffuse lobe, plus Karis' split-sum
// approximation for image-based light. Requires common.wgsl (PI).
//
// Passes call these instead of keeping copies, so the classic deferred lighting and MegaLights
// (megalights_*.wgsl) shade identically and a change of material model lands everywhere at once.

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
