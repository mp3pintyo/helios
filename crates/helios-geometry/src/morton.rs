//! Morton (Z-order) codes, used to give clustering and grouping a spatially coherent seed order.

use glam::Vec3;

use crate::Aabb;

#[inline]
fn spread_bits(mut x: u32) -> u32 {
    x &= 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

/// 30-bit Morton code of `p` quantised inside `bounds`.
pub fn morton3(p: Vec3, bounds: &Aabb) -> u32 {
    let extent = (bounds.max - bounds.min).max(Vec3::splat(1e-12));
    let n = ((p - bounds.min) / extent).clamp(Vec3::ZERO, Vec3::ONE) * 1023.0;
    spread_bits(n.x as u32) | (spread_bits(n.y as u32) << 1) | (spread_bits(n.z as u32) << 2)
}

/// Indices `0..points.len()` sorted along a Z-order curve.
pub fn morton_order(points: &[Vec3]) -> Vec<u32> {
    let bounds = Aabb::from_points(points.iter().copied());
    let mut keyed: Vec<(u32, u32)> = points
        .iter()
        .enumerate()
        .map(|(i, &p)| (morton3(p, &bounds), i as u32))
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, i)| i).collect()
}
