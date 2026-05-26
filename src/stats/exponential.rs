//! Exponential distribution functions.
//! Direct port of Easel's esl_exponential.c.

use crate::util::cmath::c_exp_f64;

/// Threshold below which `1 - exp(-x) ~ x` is used to avoid cancellation.
/// Mirrors Easel's `eslSMALLX1` (`= 5e-9`, defined in `easel.h`).
const ESL_SMALLX1: f64 = 5e-9;

/// Survivor function P(X > x), i.e. 1 - CDF, the right-tail probability mass.
///
/// Given offset `mu` and decay parameter `lambda`. Returns 1.0 for x < mu;
/// otherwise exp(-lambda*(x-mu)). Port of Easel `esl_exp_surv`.
pub fn surv(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 1.0;
    }
    c_exp_f64(-lambda * (x - mu))
}

/// Log survivor function log P(X > x), i.e. log(1 - CDF).
///
/// Returns 0.0 for x < mu; otherwise -lambda*(x-mu). Port of `esl_exp_logsurv`.
pub fn logsurv(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 0.0;
    }
    -lambda * (x - mu)
}

/// Probability density function P(X = x) for the exponential.
///
/// Returns 0.0 for x < mu; otherwise lambda * exp(-lambda*(x-mu)).
/// Port of Easel `esl_exp_pdf`.
pub fn pdf(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 0.0;
    }
    lambda * c_exp_f64(-lambda * (x - mu))
}

/// Cumulative distribution function P(X <= x) for the exponential.
///
/// Returns 0.0 for x < mu. Uses the small-y approximation 1 - exp(-y) ~ y
/// when y = lambda*(x-mu) is tiny. Port of Easel `esl_exp_cdf`.
pub fn cdf(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 0.0;
    }
    let y = lambda * (x - mu);
    if y < ESL_SMALLX1 {
        y
    } else {
        1.0 - c_exp_f64(-y)
    }
}

/// Maximum likelihood fit of exponential parameters to complete data.
///
/// ML mu is the smallest sample; ML lambda is the reciprocal of the mean
/// of (x_i - mu). Returns `(mu, lambda)`. Port of `esl_exp_FitComplete`.
pub fn fit_complete(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mu = x.iter().copied().fold(f64::INFINITY, f64::min);
    let mean: f64 = x.iter().sum::<f64>() / n;
    let lambda = 1.0 / (mean - mu);
    (mu, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surv_basic() {
        assert!((surv(0.0, 0.0, 1.0) - 1.0).abs() < 1e-15);
        assert!((surv(1.0, 0.0, 1.0) - c_exp_f64(-1.0)).abs() < 1e-15);
        assert_eq!(surv(-1.0, 0.0, 1.0), 1.0);
    }

    #[test]
    fn test_logsurv_basic() {
        assert_eq!(logsurv(0.0, 0.0, 1.0), 0.0);
        assert!((logsurv(5.0, 0.0, 1.0) - (-5.0)).abs() < 1e-15);
    }
}
