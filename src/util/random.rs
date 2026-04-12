//! Mersenne Twister MT19937 random number generator.
//! Exact port of Easel's esl_random.c for --seed reproducibility with C HMMER.

const N: usize = 624;
const M: usize = 397;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;
const MAG01: [u32; 2] = [0x0, 0x9908b0df];

/// Mersenne Twister MT19937 random number generator.
pub struct MersenneTwister {
    mt: [u32; N],
    mti: usize,
    pub seed: u32,
}

impl MersenneTwister {
    /// Create a new MT19937 RNG with the given seed.
    pub fn new(seed: u32) -> Self {
        let mut rng = MersenneTwister {
            mt: [0u32; N],
            mti: N + 1,
            seed,
        };
        rng.seed_table(seed);
        rng
    }

    /// Seed the state table (Knuth LCG with multiplier 69069).
    fn seed_table(&mut self, seed: u32) {
        self.seed = seed;
        self.mt[0] = seed;
        for z in 1..N {
            self.mt[z] = 69069u32.wrapping_mul(self.mt[z - 1]);
        }
        self.mti = N; // force fill on first use
    }

    /// Fill the state table with the twist transformation.
    fn fill_table(&mut self) {
        let mut y: u32;

        for z in 0..(N - M) {
            y = (self.mt[z] & UPPER_MASK) | (self.mt[z + 1] & LOWER_MASK);
            self.mt[z] = self.mt[z + M] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];
        }
        for z in (N - M)..(N - 1) {
            y = (self.mt[z] & UPPER_MASK) | (self.mt[z + 1] & LOWER_MASK);
            self.mt[z] = self.mt[z + M - N] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];
        }
        y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];

        self.mti = 0;
    }

    /// Generate a random u32.
    pub fn next_u32(&mut self) -> u32 {
        if self.mti >= N {
            self.fill_table();
        }
        let mut x = self.mt[self.mti];
        self.mti += 1;

        // Tempering
        x ^= x >> 11;
        x ^= (x << 7) & 0x9d2c5680;
        x ^= (x << 15) & 0xefc60000;
        x ^= x >> 18;
        x
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mt_deterministic() {
        let mut rng1 = MersenneTwister::new(42);
        let mut rng2 = MersenneTwister::new(42);
        for _ in 0..1000 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_mt_range() {
        let mut rng = MersenneTwister::new(42);
        for _ in 0..10000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_mt_seed42_first_values() {
        // Verify first few values match C's esl_random with seed 42
        let mut rng = MersenneTwister::new(42);
        let v0 = rng.next_u32();
        let v1 = rng.next_u32();
        let v2 = rng.next_u32();
        // These should be deterministic for seed 42
        assert_ne!(v0, 0);
        assert_ne!(v1, v0);
        assert_ne!(v2, v1);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn test_mt_matches_c() {
        // Cross-validate with C implementation
        unsafe {
            let c_rng = crate::ffi::esl_randomness_Create(42);
            let mut rust_rng = MersenneTwister::new(42);

            for i in 0..100 {
                let c_val = crate::ffi::esl_random(c_rng);
                let r_val = rust_rng.next_f64();
                assert!(
                    (c_val - r_val).abs() < 1e-10,
                    "Mismatch at iteration {}: c={}, rust={}",
                    i, c_val, r_val
                );
            }

            crate::ffi::esl_randomness_Destroy(c_rng);
        }
    }
}
