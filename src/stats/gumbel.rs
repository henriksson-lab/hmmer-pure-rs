//! Gumbel (type I extreme value) distribution functions.
//! Direct port of Easel's esl_gumbel.c.

use crate::errors::{HmmerError, HmmerResult};

const SMALLX1: f64 = 5e-9;

/// Probability density at `x`: P(X = x) for a Gumbel with location `mu`, scale `lambda`.
///
/// Let y = lambda*(x-mu); returns lambda * exp(-y - exp(-y)).
/// Useful dynamic range is roughly -6.5 <= y <= 710 for f64.
/// Port of Easel `esl_gumbel_pdf`.
pub fn pdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    lambda * (-y - (-y).exp()).exp()
}

/// Log probability density at `x`: log P(X = x) for the Gumbel.
///
/// Equals log(lambda) - y - exp(-y) where y = lambda*(x-mu).
/// Port of Easel `esl_gumbel_logpdf`.
pub fn logpdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    lambda.ln() - y - (-y).exp()
}

/// Cumulative distribution P(X <= x) for the Gumbel.
///
/// Returns exp(-exp(-y)) with y = lambda*(x-mu).
/// Port of Easel `esl_gumbel_cdf`.
pub fn cdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    (-(-y).exp()).exp()
}

/// Log cumulative distribution log P(X <= x) for the Gumbel.
///
/// Equals -exp(-y) with y = lambda*(x-mu).
/// Port of Easel `esl_gumbel_logcdf`.
pub fn logcdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    -(-y).exp()
}

/// Right tail mass P(X > x) = 1 - CDF for the Gumbel.
///
/// Uses the 1 - e^x ~ -x approximation when e^-y is tiny to avoid cancellation.
/// Port of Easel `esl_gumbel_surv`.
pub fn surv(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    let ey = -(-y).exp();
    if ey.abs() < SMALLX1 {
        -ey
    } else {
        1.0 - ey.exp()
    }
}

/// Log survival log P(X > x) for the Gumbel.
///
/// Real calculation is log(1 - exp(-exp(-y))); two limiting approximations
/// are used at the small/large-y extremes for numerical stability.
/// Port of Easel `esl_gumbel_logsurv`.
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

/// Inverse CDF: return quantile x such that P(X <= x) = p.
///
/// Port of Easel `esl_gumbel_invcdf`.
pub fn invcdf(p: f64, mu: f64, lambda: f64) -> f64 {
    mu - ((-1.0 * p.ln()).ln() / lambda)
}

/// Inverse survivor: return quantile x at which the right tail mass equals `p`.
///
/// Uses log(1-p) ~ -p and log(p) ~ (p^p - 1)/p for small `p` to avoid
/// the inf at p < ~1e-15. Port of Easel `esl_gumbel_invsurv`.
pub fn invsurv(p: f64, mu: f64, lambda: f64) -> f64 {
    let log_part = if p < SMALLX1 {
        (p.powf(p) - 1.0) / p
    } else {
        (-1.0 * (1.0 - p).ln()).ln()
    };
    mu - (log_part / lambda)
}

/// Evaluate Lawless equation 4.1.6 and its derivative at `lambda`.
///
/// Returns `(f, df)` where the ML estimate of lambda is the root of `f`.
/// Used by Newton-Raphson and bisection inside `fit_complete`.
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

/// Maximum likelihood fit of Gumbel parameters (mu, lambda) to complete data.
///
/// Uses `Lawless82`: Newton-Raphson on equation 4.1.6 for lambda, then
/// 4.1.5 to recover mu. Falls back to bisection if Newton-Raphson stalls.
/// Needs ~1000+ samples for a reliable lambda estimate. Returns
/// `HmmerError::NoResult` if the search cannot bracket/converge.
/// Port of Easel `esl_gumbel_FitComplete`.
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

/// ML estimate of `mu` given a known (or fixed) `lambda` for complete data.
///
/// Straight simplification of `fit_complete`: substitute lambda directly
/// into Lawless 4.1.5. Errors if n <= 1. Port of `esl_gumbel_FitCompleteLoc`.
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
}
