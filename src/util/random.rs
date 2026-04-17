//! Easel random number generator used by HMMER's search pipeline.
//!
//! HMMER creates pipeline/domain-definition RNGs with
//! `esl_randomness_CreateFast()`, the legacy Knuth LCG path in
//! `esl_random.c`, so this type preserves that stream for parity.

/// Legacy Easel `CreateFast` random number generator.
pub struct MersenneTwister {
    x: u32,
    pub seed: u32,
}

impl MersenneTwister {
    /// Create a new RNG with the same stream as Easel's esl_randomness_CreateFast().
    pub fn new(seed: u32) -> Self {
        let seed = if seed == 0 { 42 } else { seed };
        let mut x = esl_mix3(seed, 87_654_321, 12_345_678);
        if x == 0 {
            x = 42;
        }
        MersenneTwister { x, seed }
    }

    /// Generate a random u32 using Easel's `(a=69069, c=1)` LCG.
    pub fn next_u32(&mut self) -> u32 {
        self.x = self.x.wrapping_mul(69_069).wrapping_add(1);
        self.x
    }

    /// Generate a uniform random double in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / 4294967296.0 // 2^32
    }

    /// Generate a uniform random float in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// Sample from a discrete probability distribution.
    /// Returns index 0..n-1 according to probabilities p[0..n-1].
    pub fn sample_discrete(&mut self, p: &[f32]) -> usize {
        let r = self.next_f64() as f32;
        let mut cumsum = 0.0_f32;
        for (i, &pi) in p.iter().enumerate() {
            cumsum += pi;
            if r < cumsum {
                return i;
            }
        }
        p.len() - 1
    }

    /// Generate a random digital residue according to background frequencies.
    pub fn sample_residue(&mut self, bg_f: &[f32]) -> u8 {
        self.sample_discrete(bg_f) as u8
    }
}

fn esl_mix3(mut a: u32, mut b: u32, mut c: u32) -> u32 {
    a = a.wrapping_sub(b);
    a = a.wrapping_sub(c);
    a ^= c >> 13;
    b = b.wrapping_sub(c);
    b = b.wrapping_sub(a);
    b ^= a << 8;
    c = c.wrapping_sub(a);
    c = c.wrapping_sub(b);
    c ^= b >> 13;
    a = a.wrapping_sub(b);
    a = a.wrapping_sub(c);
    a ^= c >> 12;
    b = b.wrapping_sub(c);
    b = b.wrapping_sub(a);
    b ^= a << 16;
    c = c.wrapping_sub(a);
    c = c.wrapping_sub(b);
    c ^= b >> 5;
    a = a.wrapping_sub(b);
    a = a.wrapping_sub(c);
    a ^= c >> 3;
    b = b.wrapping_sub(c);
    b = b.wrapping_sub(a);
    b ^= a << 10;
    c = c.wrapping_sub(a);
    c = c.wrapping_sub(b);
    c ^= b >> 15;
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_rng_deterministic() {
        let mut rng1 = MersenneTwister::new(42);
        let mut rng2 = MersenneTwister::new(42);
        for _ in 0..1000 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_fast_rng_range() {
        let mut rng = MersenneTwister::new(42);
        for _ in 0..10000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_fast_rng_seed42_first_values() {
        // Verify first few values match Easel's esl_randomness_CreateFast(42).
        let mut rng = MersenneTwister::new(42);
        assert_eq!(rng.next_u32(), 432_788_820);
        assert_eq!(rng.next_u32(), 3_613_595_717);
        assert_eq!(rng.next_u32(), 2_598_039_618);
    }
}
