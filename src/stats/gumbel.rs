//! Gumbel (type I extreme value) distribution functions.
//! Direct port of Easel's esl_gumbel.c.

use crate::errors::{HmmerError, HmmerResult};

const SMALLX1: f64 = 5e-9;

/// Probability density function P(X=x).
pub fn pdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    lambda * (-y - (-y).exp()).exp()
}

/// Log probability density function log P(X=x).
pub fn logpdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    lambda.ln() - y - (-y).exp()
}

/// Cumulative distribution function P(X <= x).
pub fn cdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    (-(-y).exp()).exp()
}

/// Log cumulative distribution function log P(X <= x).
pub fn logcdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    -(-y).exp()
}

/// Survivor function P(X > x), i.e. 1 - CDF.
pub fn surv(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    let ey = -(-y).exp();
    if ey.abs() < SMALLX1 {
        -ey
    } else {
        1.0 - ey.exp()
    }
}

/// Log survivor function log P(X > x).
pub fn logsurv(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    let ey = -(-y).exp();

    if ey.abs() < SMALLX1 {
        -y
    } else if ey.exp().abs() < SMALLX1 {
        -ey.exp()
    } else {
        (1.0 - ey.exp()).ln()
    }
}

/// Inverse CDF: returns x such that CDF(x) = p.
pub fn invcdf(p: f64, mu: f64, lambda: f64) -> f64 {
    mu - ((-1.0 * p.ln()).ln() / lambda)
}

/// Inverse survivor: returns x such that P(X > x) = p.
pub fn invsurv(p: f64, mu: f64, lambda: f64) -> f64 {
    let log_part = if p < SMALLX1 {
        (p.powf(p) - 1.0) / p
    } else {
        (-1.0 * (1.0 - p).ln()).ln()
    };
    mu - (log_part / lambda)
}

/// Lawless equation 4.1.6 and its derivative for ML fitting.
fn lawless416(x: &[f64], lambda: f64) -> (f64, f64) {
    let n = x.len() as f64;
    let mut esum = 0.0_f64;
    let mut xesum = 0.0_f64;
    let mut xxesum = 0.0_f64;
    let mut xsum = 0.0_f64;

    for &xi in x {
        let e = (-lambda * xi).exp();
        xsum += xi;
        xesum += xi * e;
        xxesum += xi * xi * e;
        esum += e;
    }

    let f = (1.0 / lambda) - (xsum / n) + (xesum / esum);
    let df = (xesum / esum) * (xesum / esum) - (xxesum / esum) - (1.0 / (lambda * lambda));
    (f, df)
}

/// Maximum likelihood fit of Gumbel parameters to complete data.
///
/// Returns `(mu, lambda)` or error if the fit fails.
pub fn fit_complete(x: &[f64]) -> HmmerResult<(f64, f64)> {
    let n = x.len();
    if n <= 1 {
        return Err(HmmerError::InvalidArg(
            "Need more than 1 sample for Gumbel fit".to_string(),
        ));
    }

    // 1. Initial guess at lambda
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let variance: f64 = x.iter().map(|&xi| (xi - mean) * (xi - mean)).sum::<f64>() / (n - 1) as f64;
    let mut lambda = std::f64::consts::PI / (6.0 * variance).sqrt();

    // 2. Newton/Raphson to solve Lawless 4.1.6
    let tol = 1e-5;
    let mut converged = false;
    for _ in 0..100 {
        let (fx, dfx) = lawless416(x, lambda);
        if fx.abs() < tol {
            converged = true;
            break;
        }
        lambda -= fx / dfx;
        if lambda <= 0.0 {
            lambda = 0.001;
        }
    }

    // 2.5: Fallback to bisection if Newton/Raphson failed
    if !converged {
        let mut left = 0.0_f64;
        let mut right = std::f64::consts::PI / (6.0 * variance).sqrt();
        let (mut fx, _) = lawless416(x, right);
        while fx > 0.0 {
            right *= 2.0;
            if right > 1000.0 {
                return Err(HmmerError::NoResult);
            }
            let result = lawless416(x, right);
            fx = result.0;
        }

        converged = false;
        for _ in 0..100 {
            let mid = (left + right) / 2.0;
            let (fx, _) = lawless416(x, mid);
            if fx.abs() < tol {
                lambda = mid;
                converged = true;
                break;
            }
            if fx > 0.0 {
                left = mid;
            } else {
                right = mid;
            }
        }

        if !converged {
            return Err(HmmerError::NoResult);
        }
    }

    // 3. Substitute into Lawless 4.1.5 to find mu
    let esum: f64 = x.iter().map(|&xi| (-lambda * xi).exp()).sum();
    let mu = -(esum / n as f64).ln() / lambda;

    Ok((mu, lambda))
}

/// ML estimate of mu given known lambda (complete data).
pub fn fit_complete_loc(x: &[f64], lambda: f64) -> HmmerResult<f64> {
    let n = x.len();
    if n <= 1 {
        return Err(HmmerError::InvalidArg(
            "Need more than 1 sample for Gumbel fit".to_string(),
        ));
    }
    let esum: f64 = x.iter().map(|&xi| (-lambda * xi).exp()).sum();
    let mu = -(esum / n as f64).ln() / lambda;
    Ok(mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surv_basic() {
        // P(X > mu) for standard Gumbel should be 1 - e^{-1} ≈ 0.6321
        let p = surv(0.0, 0.0, 1.0);
        assert!((p - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_invsurv_basic() {
        let mu = 10.0;
        let lambda = 0.5;
        let p = surv(15.0, mu, lambda);
        let x = invsurv(p, mu, lambda);
        assert!((x - 15.0).abs() < 1e-6, "x={}, expected 15.0", x);
    }

    #[test]
    fn test_surv_matches_ffi() {
        let test_cases = [
            (0.0, 0.0, 1.0),
            (5.0, 1.0, 0.5),
            (-3.0, 0.0, 1.0),
            (100.0, 50.0, 0.1),
        ];
        for (x, mu, lambda) in test_cases {
            let rust_val = surv(x, mu, lambda);
            let c_val = unsafe { crate::ffi::esl_gumbel_surv(x, mu, lambda) };
            assert!(
                (rust_val - c_val).abs() < 1e-15,
                "surv({},{},{}) mismatch: rust={}, c={}",
                x, mu, lambda, rust_val, c_val
            );
        }
    }

    #[test]
    fn test_logsurv_matches_ffi() {
        let test_cases = [
            (0.0, 0.0, 1.0),
            (5.0, 1.0, 0.5),
            (100.0, 50.0, 0.1),
        ];
        for (x, mu, lambda) in test_cases {
            let rust_val = logsurv(x, mu, lambda);
            let c_val = unsafe { crate::ffi::esl_gumbel_logsurv(x, mu, lambda) };
            assert!(
                (rust_val - c_val).abs() < 1e-12,
                "logsurv({},{},{}) mismatch: rust={}, c={}",
                x, mu, lambda, rust_val, c_val
            );
        }
    }

    #[test]
    fn test_invsurv_matches_ffi() {
        let test_cases = [
            (0.5, 0.0, 1.0),
            (0.01, 10.0, 0.5),
            (1e-10, 0.0, 1.0),
        ];
        for (p, mu, lambda) in test_cases {
            let rust_val = invsurv(p, mu, lambda);
            let c_val = unsafe { crate::ffi::esl_gumbel_invsurv(p, mu, lambda) };
            assert!(
                (rust_val - c_val).abs() < 1e-10,
                "invsurv({},{},{}) mismatch: rust={}, c={}",
                p, mu, lambda, rust_val, c_val
            );
        }
    }
}
