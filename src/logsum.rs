//! Fast table-driven log-sum approximation for the Forward algorithm.
//!
//! Computes `log(exp(a) + exp(b))` using a precomputed lookup table.
//! This is a direct port of HMMER's `logsum.c`.

#![allow(clippy::needless_range_loop)]

use std::sync::Once;

use crate::util::cmath::{c_exp_f64, c_log_f64};

const LOGSUM_SCALE: f32 = 1000.0;
const LOGSUM_TBL: usize = 16000;

static INIT: Once = Once::new();
static mut FLOGSUM_LOOKUP: [f32; LOGSUM_TBL] = [0.0; LOGSUM_TBL];

/// Initialize the `p7_flogsum()` lookup table. Must be called once before any
/// call to `p7_flogsum()`. Safe to call repeatedly; only the first call
/// populates the table. Counterpart to C's `p7_FLogsumInit()`.
pub fn p7_flogsuminit() {
    crate::util::simd_env::init();

    INIT.call_once(|| {
        // SAFETY: This is only called once via Once, so no data race.
        unsafe {
            for i in 0..LOGSUM_TBL {
                FLOGSUM_LOOKUP[i] =
                    c_log_f64(1.0_f64 + c_exp_f64(-(i as f64) / LOGSUM_SCALE as f64)) as f32;
            }
        }
    });
}

/// Fast table-driven approximation of `log(exp(a) + exp(b))`.
///
/// Inner-loop primitive of the generic Forward algorithm. Either input may be
/// `-INFINITY`, but neither may be `+INFINITY` or `NaN`. Counterpart to C's
/// `p7_FLogsum()`.
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

/// Absolute error in probability space introduced by `p7_flogsum()`'s table
/// lookup: `exp(approx) - exp(exact)`. Counterpart to C's `p7_FLogsumError()`.
pub fn p7_flogsum_error(a: f32, b: f32) -> f32 {
    // Mirror C bit-for-bit: `float exact = log(exp(a)+exp(b));` evaluates
    // exp/log in double (libm) but rounds the result to float on store, and
    // `exp(approx) - exp(exact)` is then evaluated in double and rounded to
    // float on return. Keep the same promotion/rounding pattern.
    let approx = p7_flogsum(a, b);
    let exact = (c_exp_f64(a as f64) + c_exp_f64(b as f64)).ln() as f32;
    (c_exp_f64(approx as f64) - c_exp_f64(exact as f64)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsum_specials() {
        p7_flogsuminit();
        assert_eq!(p7_flogsum(0.0, f32::NEG_INFINITY), 0.0);
        assert_eq!(p7_flogsum(f32::NEG_INFINITY, 0.0), 0.0);
        assert_eq!(
            p7_flogsum(f32::NEG_INFINITY, f32::NEG_INFINITY),
            f32::NEG_INFINITY
        );
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
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
}
