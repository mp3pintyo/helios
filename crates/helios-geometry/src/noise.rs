//! Deterministic 3D gradient noise and fractal helpers for procedural geometry.

use glam::Vec3;

#[inline]
fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

#[inline]
fn lattice_gradient(ix: i32, iy: i32, iz: i32, seed: u32) -> Vec3 {
    let h = hash((ix as u32).wrapping_mul(0x8da6_b343)
        ^ (iy as u32).wrapping_mul(0xd816_3841)
        ^ (iz as u32).wrapping_mul(0xcb1a_b31f)
        ^ seed.wrapping_mul(0x1656_67b1));
    // 12 cube-edge gradients, as in improved Perlin noise.
    const G: [[f32; 3]; 12] = [
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
    ];
    Vec3::from(G[(h % 12) as usize])
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Perlin-style gradient noise in roughly `[-1, 1]`.
pub fn gradient(p: Vec3, seed: u32) -> f32 {
    let base = p.floor();
    let f = p - base;
    let (ix, iy, iz) = (base.x as i32, base.y as i32, base.z as i32);
    let u = Vec3::new(fade(f.x), fade(f.y), fade(f.z));

    let corner = |dx: i32, dy: i32, dz: i32| {
        let g = lattice_gradient(ix + dx, iy + dy, iz + dz, seed);
        g.dot(f - Vec3::new(dx as f32, dy as f32, dz as f32))
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;

    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u.x);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u.x);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u.x);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u.x);
    lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z)
}

/// Fractal Brownian motion: `octaves` layers of noise, each at double frequency.
pub fn fbm(p: Vec3, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..octaves {
        sum += gradient(p * freq, seed.wrapping_add(o * 101)) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
    }
    sum / norm
}

/// Ridged multifractal noise in `[0, 1]` — sharp crests, good for rock and mountains.
pub fn ridged(p: Vec3, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm, mut weight) = (0.0, 1.0, 1.0, 0.0, 1.0);
    for o in 0..octaves {
        let n = 1.0 - gradient(p * freq, seed.wrapping_add(o * 131)).abs();
        let n = n * n * weight;
        weight = (n * 1.5).clamp(0.0, 1.0);
        sum += n * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.1;
    }
    sum / norm
}
