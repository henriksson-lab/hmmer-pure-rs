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
    /// Create a new fast LCG RNG seeded identically to `esl_randomness_CreateFast`.
    ///
    /// Mirrors Easel's deprecated fast path: seed=0 is replaced with 42, then
    /// the state is dispersed through `esl_mix3` and forced non-zero so
    /// downstream sequences exactly match the C reference.
    pub fn new(seed: u32) -> Self {
        let seed = if seed == 0 { 42 } else { seed };
        let mut x = esl_mix3(seed, 87_654_321, 12_345_678);
        if x == 0 {
            x = 42;
        }
        MersenneTwister { x, seed }
    }

    /// Advance the LCG one step and return a uniform u32 on [0, 2^32).
    ///
    /// Mirrors Easel's internal `knuth()` helper used by the fast RNG.
    pub fn next_u32(&mut self) -> u32 {
        self.x = self.x.wrapping_mul(69_069).wrapping_add(1);
        self.x
    }

    /// Uniform random deviate on [0.0, 1.0) as a double.
    ///
    /// Divides the next u32 by 2^32, exactly matching Easel's `esl_random()`.
    pub fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / 4294967296.0 // 2^32
    }

    /// Uniform random deviate on [0.0, 1.0) as a float, computed in f64 first.
    ///
    /// Casting to f32 only after sampling preserves the half-open range.
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// Sample an index from a normalized discrete distribution `p[0..n-1]`.
    ///
    /// Returns the index `i` selected with probability `p[i]`. Caller must
    /// pass a normalized (sum-to-one) vector; behavior on unnormalized input
    /// is the last index. Port of Easel `esl_rnd_FChoose` (simplified).
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

    /// Sample a digital residue (0..K-1) drawn from background frequencies.
    ///
    /// Thin wrapper around `sample_discrete` that returns a `u8` suitable
    /// for a digital sequence alphabet.
    pub fn sample_residue(&mut self, bg_f: &[f32]) -> u8 {
        self.sample_discrete(bg_f) as u8
    }

    /// Uniform random integer on `0..n`, matching Easel `esl_rnd_Roll()`.
    pub fn roll(&mut self, n: usize) -> usize {
        assert!(n > 0);
        let factor = u32::MAX / n as u32;
        loop {
            let u = self.next_u32() / factor;
            if (u as usize) < n {
                return u as usize;
            }
        }
    }
}

/// Bob Jenkins' 3-word mixing function used by Easel to disperse seeds.
///
/// Direct port of `esl_mix3` from Easel; the bit-rotations have no fitness
/// claims but produce well-decorrelated seeds for similar inputs.
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
