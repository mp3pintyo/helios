//! Stateless hashing: every random number is a pure function of its inputs, so results never
//! depend on the order in which points are processed.

/// PCG-style integer hash (O'Neill's permuted congruential output step).
pub fn hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Hash of several words.
pub fn hash_all(words: &[u32]) -> u32 {
    words.iter().fold(0x9e37_79b9, |h, &w| hash(h ^ w))
}

/// Uniform in [0, 1) from a hash.
pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// The `stream`-th uniform number of a point (`seed`) in a node (`salt`).
pub fn random(seed: u32, salt: u32, stream: u32) -> f32 {
    unit(hash_all(&[seed, salt, stream]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_numbers_cover_the_interval_evenly() {
        let mut buckets = [0u32; 10];
        for i in 0..10_000 {
            buckets[(random(i, 7, 0) * 10.0) as usize] += 1;
        }
        assert!(buckets.iter().all(|&b| (850..1150).contains(&b)), "{buckets:?}");
        assert_ne!(random(1, 2, 0), random(1, 2, 1));
    }
}
