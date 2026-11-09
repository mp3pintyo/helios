//! 2D value noise over the world's XZ plane (fractal, in [0, 1]), for density patterns such as
//! meadow patches. Lattice values are hashes, so the pattern is fixed in the world.

use crate::rng::{hash_all, unit};

fn lattice(x: i32, z: i32, seed: u32) -> f32 {
    unit(hash_all(&[x as u32, z as u32, seed]))
}

fn value_noise(x: f32, z: f32, seed: u32) -> f32 {
    let (x0, z0) = (x.floor(), z.floor());
    let (fx, fz) = (x - x0, z - z0);
    let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
    let (i, k) = (x0 as i32, z0 as i32);
    let a = lattice(i, k, seed) + (lattice(i + 1, k, seed) - lattice(i, k, seed)) * sx;
    let b = lattice(i, k + 1, seed) + (lattice(i + 1, k + 1, seed) - lattice(i, k + 1, seed)) * sx;
    a + (b - a) * sz
}

/// Three octaves of value noise with features about `scale` metres across, in [0, 1].
pub fn fbm(x: f32, z: f32, scale: f32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 0.5;
    let mut frequency = 1.0 / scale.max(1e-3);
    for octave in 0..3 {
        sum += value_noise(x * frequency, z * frequency, seed.wrapping_add(octave)) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.03;
    }
    sum / 0.875
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_bounded_and_continuous() {
        for i in 0..2000 {
            let (x, z) = (i as f32 * 0.37 - 300.0, i as f32 * 0.11);
            let n = fbm(x, z, 10.0, 3);
            assert!((0.0..=1.0).contains(&n));
            assert!((fbm(x + 0.01, z, 10.0, 3) - n).abs() < 0.02);
        }
    }
}
