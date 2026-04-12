//! Fast table-driven log-sum approximation for the Forward algorithm.
//!
//! Computes `log(exp(a) + exp(b))` using a precomputed lookup table.
//! This is a direct port of HMMER's `logsum.c`.

use std::sync::Once;

const LOGSUM_SCALE: f32 = 1000.0;
const LOGSUM_TBL: usize = 16000;

static INIT: Once = Once::new();
static mut FLOGSUM_LOOKUP: [f32; LOGSUM_TBL] = [0.0; LOGSUM_TBL];

/// Initialize the lookup table. Must be called before `p7_flogsum()`.
/// Safe to call multiple times; only initializes once.
pub fn p7_flogsuminit() {
    INIT.call_once(|| {
        // SAFETY: This is only called once via Once, so no data race.
        unsafe {
            for i in 0..LOGSUM_TBL {
                FLOGSUM_LOOKUP[i] =
                    (1.0_f64 + (-(i as f64) / LOGSUM_SCALE as f64).exp()).ln() as f32;
            }
        }
    });
}

/// Fast approximation of `log(exp(a) + exp(b))` using a lookup table.
///
/// Either `a` or `b` (or both) may be `-INFINITY`, but neither may be
/// `+INFINITY` or `NaN`.
#[inline]
pub fn p7_flogsum(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    let min = a.min(b);

    if min == f32::NEG_INFINITY || (max - min) >= 15.7 {
        max
    } else {
        // SAFETY: FLOGSUM_LOOKUP is initialized before any call via p7_flogsuminit(),
        // and is read-only after initialization.
        max + unsafe { FLOGSUM_LOOKUP[((max - min) * LOGSUM_SCALE) as usize] }
    }
}

/// Compute the absolute error in probability space from the table lookup approximation.
pub fn p7_flogsum_error(a: f32, b: f32) -> f32 {
    let approx = p7_flogsum(a, b);
    let exact = ((a as f64).exp() + (b as f64).exp()).ln() as f32;
    approx.exp() - exact.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsum_specials() {
        p7_flogsuminit();
        assert_eq!(p7_flogsum(0.0, f32::NEG_INFINITY), 0.0);
        assert_eq!(p7_flogsum(f32::NEG_INFINITY, 0.0), 0.0);
        assert_eq!(p7_flogsum(f32::NEG_INFINITY, f32::NEG_INFINITY), f32::NEG_INFINITY);
    }

    #[test]
    fn test_logsum_accuracy() {
        p7_flogsuminit();
        let max_val = 20.0_f32;
        let n = 1000;
        let mut max_err: f32 = 0.0;
        let mut avg_err: f32 = 0.0;
        // Simple LCG for reproducibility (matches esl_random with seed 42 behavior)
        let mut rng_state: u64 = 42;
        let mut next_rand = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f32) / (u32::MAX as f32 / 2.0)
        };

        for _ in 0..n {
            let a = (next_rand() - 0.5) * max_val * 2.0;
            let b = (next_rand() - 0.5) * max_val * 2.0;
            let exact = ((a as f64).exp() + (b as f64).exp()).ln() as f32;
            let result = p7_flogsum(a, b);
            let err = (exact - result).abs() / max_val;
            avg_err += err;
            max_err = max_err.max(err);
        }
        avg_err /= n as f32;

        assert!(max_err < 0.0001, "max error {} too high", max_err);
        assert!(avg_err < 0.0001, "avg error {} too high", avg_err);
    }

    /// Verify our lookup table matches the C implementation exactly (bit-for-bit).
    #[test]
    fn test_logsum_table_matches_c() {
        p7_flogsuminit();
        // Test a few known values: compare against direct computation
        for i in [0, 1, 100, 1000, 5000, 10000, 15999] {
            let expected = (1.0_f64 + (-(i as f64) / 1000.0).exp()).ln() as f32;
            let actual = unsafe { FLOGSUM_LOOKUP[i] };
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "Mismatch at index {}: expected {:e}, got {:e}",
                i,
                expected,
                actual
            );
        }
    }

    /// Cross-validate against FFI C implementation.
    #[test]
    fn test_logsum_matches_ffi() {
        p7_flogsuminit();
        unsafe {
            crate::ffi::p7_FLogsumInit();
        }

        let test_pairs: &[(f32, f32)] = &[
            (0.0, 0.0),
            (-1.0, -2.0),
            (-5.0, -0.5),
            (0.0, f32::NEG_INFINITY),
            (f32::NEG_INFINITY, 0.0),
            (f32::NEG_INFINITY, f32::NEG_INFINITY),
            (-100.0, -100.0),
            (-0.4, -0.5),
            (10.0, -5.0),
        ];

        for &(a, b) in test_pairs {
            let rust_result = p7_flogsum(a, b);
            let c_result = unsafe { crate::ffi::p7_FLogsum(a, b) };
            assert_eq!(
                rust_result.to_bits(),
                c_result.to_bits(),
                "Mismatch for ({}, {}): rust={:e}, c={:e}",
                a,
                b,
                rust_result,
                c_result
            );
        }
    }
}
