//! Exponential distribution functions.
//! Direct port of Easel's esl_exponential.c.

/// Survivor function P(X > x).
pub fn surv(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 1.0;
    }
    (-lambda * (x - mu)).exp()
}

/// Log survivor function log P(X > x).
pub fn logsurv(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 0.0;
    }
    -lambda * (x - mu)
}

/// Probability density function P(X = x).
pub fn pdf(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 0.0;
    }
    lambda * (-lambda * (x - mu)).exp()
}

/// Cumulative distribution function P(X <= x).
pub fn cdf(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 0.0;
    }
    let y = lambda * (x - mu);
    if y < 5e-9 {
        y
    } else {
        1.0 - (-y).exp()
    }
}

/// Maximum likelihood fit of exponential parameters to complete data.
/// Returns `(mu, lambda)`.
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
        assert!((surv(1.0, 0.0, 1.0) - (-1.0_f64).exp()).abs() < 1e-15);
        assert_eq!(surv(-1.0, 0.0, 1.0), 1.0);
    }

    #[test]
    fn test_logsurv_basic() {
        assert_eq!(logsurv(0.0, 0.0, 1.0), 0.0);
        assert!((logsurv(5.0, 0.0, 1.0) - (-5.0)).abs() < 1e-15);
    }
}
