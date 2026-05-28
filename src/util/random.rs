//! Easel random number generator used by HMMER's search pipeline.
//!
//! HMMER creates pipeline/domain-definition RNGs with
//! `esl_randomness_CreateFast()`, the legacy Knuth LCG path in
//! `esl_random.c`, so this type preserves that stream for parity.

use std::time::{SystemTime, UNIX_EPOCH};

/// Legacy Easel `CreateFast` random number generator.
pub struct MersenneTwister {
    x: u32,
    pub seed: u32,
}

impl MersenneTwister {
    /// Create a new fast LCG RNG seeded identically to `esl_randomness_CreateFast`.
    ///
    /// Mirrors Easel's deprecated fast path: seed=0 requests a one-time
    /// arbitrary seed, then the state is dispersed through `esl_mix3` and
    /// forced non-zero so nonzero seeds exactly match the C reference.
    pub fn new(seed: u32) -> Self {
        let seed = resolve_seed(seed);
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
        // Mirror Easel `esl_rnd_FChoose` exactly. Computing in double
        // precision is important: casting `roll` to f32 would give a [0,1]
        // number instead of [0,1). Keep `roll` as f64, accumulate the
        // cumulative sum in f64, normalize by the f64 sum of the weights,
        // and compare `roll < sum / norm`.
        let roll = self.next_f64();
        let mut norm = 0.0_f64;
        for &pi in p.iter() {
            norm += pi as f64;
        }
        debug_assert!(norm > 0.99 && norm < 1.01);

        let mut sum = 0.0_f64;
        for (i, &pi) in p.iter().enumerate() {
            sum += pi as f64;
            if roll < sum / norm {
                return i;
            }
        }
        // C reaches `esl_fatal("unreached code...")` here; in practice the
        // loop always returns. Fall back to the last index to keep a total
        // function for callers.
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
    ///
    /// Mirrors C exactly: `factor = UINT32_MAX / n`, then rejection-sample
    /// `u = esl_random_uint32(r) / factor` until `u < n`. The rejection bound
    /// is the raw u32 comparison `u >= n` (with `n` treated as a u32), so the
    /// RNG stream is consumed identically to the reference.
    pub fn roll(&mut self, n: usize) -> usize {
        assert!(n > 0);
        let n = n as u32;
        let factor = u32::MAX / n;
        loop {
            let u = self.next_u32() / factor;
            // C: `while (u >= n)` -> accept on `u < n`.
            if u < n {
                return u as usize;
            }
        }
    }
}

/// Resolve Easel seed semantics: nonzero seeds are exact, zero means
/// "choose an arbitrary one-time seed".
pub fn resolve_seed(seed: u32) -> u32 {
    resolve_seed_with(seed, arbitrary_seed)
}

fn resolve_seed_with<F>(seed: u32, arbitrary: F) -> u32
where
    F: FnOnce() -> u32,
{
    if seed != 0 {
        return seed;
    }
    match arbitrary() {
        0 | 42 => 43,
        value => value,
    }
}

fn arbitrary_seed() -> u32 {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    let addr = (&time as *const u64 as usize) as u64;
    let mixed = time ^ pid.rotate_left(17) ^ addr.rotate_left(7);
    esl_mix3(mixed as u32, (mixed >> 32) as u32, pid as u32 ^ 0x9e37_79b9)
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

/// Easel's 64-bit Mersenne Twister (MT19937-64), a faithful port of the
/// `ESL_RAND64` object in `esl_rand64.c`. This is a distinct generator from the
/// legacy 32-bit [`MersenneTwister`] (`esl_randomness`); `consensus_by_sample`
/// sequence weighting requires this exact stream for bit-faithful sampling.
pub struct Rand64 {
    mt: [u64; 312],
    mti: usize,
}

impl Rand64 {
    /// `esl_rand64_Create` / `esl_rand64_Init` with a nonzero seed
    /// (esl_rand64.c:92,126; seed 0 is never used here).
    pub fn new(seed: u64) -> Self {
        let mut rng = Rand64 {
            mt: [0u64; 312],
            mti: 0,
        };
        // mt64_seed_table (esl_rand64.c:442).
        rng.mt[0] = seed;
        for z in 1..312 {
            rng.mt[z] = 6_364_136_223_846_793_005u64
                .wrapping_mul(rng.mt[z - 1] ^ (rng.mt[z - 1] >> 62))
                .wrapping_add(z as u64);
        }
        rng.fill_table();
        rng
    }

    /// mt64_fill_table (esl_rand64.c:455).
    fn fill_table(&mut self) {
        const MAG01: [u64; 2] = [0u64, 0xB502_6F5A_A966_19E9u64];
        const UPPER: u64 = 0xFFFF_FFFF_8000_0000u64;
        const LOWER: u64 = 0x7FFF_FFFFu64;
        for z in 0..156 {
            let x = (self.mt[z] & UPPER) | (self.mt[z + 1] & LOWER);
            self.mt[z] = self.mt[z + 156] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
        }
        for z in 156..311 {
            let x = (self.mt[z] & UPPER) | (self.mt[z + 1] & LOWER);
            self.mt[z] = self.mt[z - 156] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
        }
        let x = (self.mt[311] & UPPER) | (self.mt[0] & LOWER);
        self.mt[311] = self.mt[155] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
        self.mti = 0;
    }

    /// `esl_rand64`: tempered 64-bit deviate on [0, 2^64-1] (esl_rand64.c:174).
    pub fn next_u64(&mut self) -> u64 {
        if self.mti >= 312 {
            self.fill_table();
        }
        let mut x = self.mt[self.mti];
        self.mti += 1;
        x ^= (x >> 29) & 0x5555_5555_5555_5555u64;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000u64;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000u64;
        x ^= x >> 43;
        x
    }

    /// `esl_rand64_double`: uniform double on [0,1) (esl_rand64.c:230).
    pub fn double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `esl_rand64_double_open`: uniform double on (0,1) (esl_rand64.c:258).
    pub fn double_open(&mut self) -> f64 {
        ((self.next_u64() >> 12) as f64 + 0.5) * (1.0 / 4_503_599_627_370_496.0)
    }
}

/// `esl_rand64_Deal` (esl_rand64.c:359): sample `m` integers without
/// replacement from `0..n-1`, returned in ascending order, via Vitter's
/// sequential sampling "Method D". Direct port; bit-faithful to the C.
pub fn esl_rand64_deal(rng: &mut Rand64, m_in: i64, n_in: i64) -> Vec<i64> {
    let mut m = m_in;
    let mut n = n_in;
    let mut deal = vec![0i64; m_in as usize];
    let mut i: usize = 0;
    let mut j: i64 = -1;
    let mut s: i64;
    let mut qu1 = n - m + 1;
    let negalphainv: i64 = -13;
    let mut threshold = -negalphainv * m;
    let mut mreal = m as f64;
    let mut nreal = n as f64;
    let mut minv = 1.0 / m as f64;
    #[allow(unused_assignments)]
    let mut mmin1inv = 1.0 / (m - 1) as f64;
    let mut vprime = (minv * rng.double().ln()).exp();
    let mut qu1real = nreal - mreal + 1.0;

    while m > 1 && n > threshold {
        mmin1inv = 1.0 / (-1.0 + mreal);
        let mut x: f64;
        loop {
            loop {
                x = nreal * (-vprime + 1.0);
                s = x.floor() as i64;
                if s < qu1 {
                    break;
                }
                vprime = (minv * rng.double_open().ln()).exp();
            }
            let u = rng.double_open();
            let negsreal = -s as f64;
            let y1 = (mmin1inv * (u * nreal / qu1real).ln()).exp();
            vprime = y1 * (-x / nreal + 1.0) * (qu1real / (negsreal + qu1real));
            if vprime <= 1.0 {
                break;
            }

            let mut y2 = 1.0f64;
            let mut top = nreal - 1.0;
            let bottom;
            let limit;
            if n - 1 > s {
                bottom = nreal - mreal;
                limit = n - s;
            } else {
                bottom = nreal + negsreal - 1.0;
                limit = qu1;
            }

            let mut bottom = bottom;
            let mut t = n - 1;
            while t >= limit {
                y2 = (y2 * top) / bottom;
                top -= 1.0;
                bottom -= 1.0;
                t -= 1;
            }

            if nreal / (nreal - x) >= y1 * (mmin1inv * y2.ln()).exp() {
                vprime = (mmin1inv * rng.double_open().ln()).exp();
                break;
            }
            vprime = (minv * rng.double_open().ln()).exp();
        }
        j += s + 1;
        deal[i] = j;
        i += 1;
        n = n - s - 1;
        nreal = nreal + (-s as f64) - 1.0;
        m -= 1;
        mreal -= 1.0;
        minv = mmin1inv;
        qu1 -= s;
        qu1real += -s as f64;
        threshold += negalphainv;
    }

    if m > 1 {
        vitter_a(rng, m, n, j, &mut deal[i..]);
    } else {
        s = (n as f64 * vprime).floor() as i64;
        j += s + 1;
        deal[i] = j;
    }
    deal
}

/// `vitter_a` (esl_rand64.c:289): Vitter "Method A", finishing a sample in
/// progress (sampling the remaining `m` from `n`, last sampled index `j`).
fn vitter_a(rng: &mut Rand64, m_in: i64, n_in: i64, j_in: i64, deal: &mut [i64]) {
    let mut m = m_in;
    let mut j = j_in;
    let mut i: usize = 0;
    let mut s: i64;
    let mut top = (n_in - m_in) as f64;
    let mut nreal = n_in as f64;

    while m >= 2 {
        let u = rng.double_open();
        s = 0;
        let mut quot = top / nreal;
        while quot > u {
            s += 1;
            top -= 1.0;
            nreal -= 1.0;
            quot = (quot * top) / nreal;
        }
        j += s + 1;
        deal[i] = j;
        i += 1;
        nreal -= 1.0;
        m -= 1;
    }
    s = (nreal.round() * rng.double()).floor() as i64;
    j += s + 1;
    deal[i] = j;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rand64_stream_matches_c_seed42() {
        // Reference values from esl_rand64.c (MT19937-64) seeded with 42.
        let mut rng = Rand64::new(42);
        assert_eq!(rng.next_u64(), 13_930_160_852_258_120_406);
        assert_eq!(rng.next_u64(), 11_788_048_577_503_494_824);
        assert_eq!(rng.next_u64(), 13_874_630_024_467_741_450);
        assert_eq!(rng.next_u64(), 2_513_787_319_205_155_662);
        assert_eq!(rng.next_u64(), 16_662_371_453_428_439_381);
    }

    #[test]
    fn rand64_deal_matches_c() {
        let mut rng = Rand64::new(42);
        assert_eq!(
            esl_rand64_deal(&mut rng, 10, 100),
            vec![4, 7, 27, 28, 51, 56, 66, 78, 86, 87]
        );
        // Larger range exercises Method D's main loop before Method A finishes.
        let mut rng = Rand64::new(42);
        assert_eq!(
            esl_rand64_deal(&mut rng, 20, 1_000_000),
            vec![
                13943, 36911, 52030, 156903, 162245, 284384, 312154, 362418, 427644, 474566,
                661311, 684796, 699340, 718078, 726888, 729921, 748393, 807346, 958321, 961014
            ]
        );
    }

    #[test]
    fn test_fast_rng_deterministic() {
        let mut rng1 = MersenneTwister::new(42);
        let mut rng2 = MersenneTwister::new(42);
        for _ in 0..1000 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn seed_zero_resolves_to_arbitrary_nonzero_seed() {
        assert_eq!(resolve_seed_with(42, || 7), 42);
        assert_eq!(resolve_seed_with(0, || 7), 7);
        assert_eq!(resolve_seed_with(0, || 0), 43);
        assert_eq!(resolve_seed_with(0, || 42), 43);
    }

    #[test]
    fn test_fast_rng_range() {
        let mut rng = MersenneTwister::new(42);
        for _ in 0..10000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
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
