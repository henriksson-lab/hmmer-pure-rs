//! One-dimensional root finding.
//! Port of Easel's esl_rootfinder.c (Newton/Raphson slice).
//!
//! Only the `ESL_ROOTFINDER` configuration reachable from
//! `esl_rootfinder_CreateFDF()` + `esl_root_NewtonRaphson()` is ported here,
//! because that is the only entry point HMMER's builder uses (via
//! `esl_scorematrix_ProbifyGivenBG()`, esl_scorematrix.c:1290-1291).
//!
//! Note: `src/eweight.rs` carries its own private port of
//! `esl_root_Bisection()` with a different tolerance set (abs_tol 0.01 from
//! eweight.c:82). It is deliberately left there; merging the two would change
//! no arithmetic and is not backed by a parity result.

/// `esl_rootfinder_CreateFDF()` defaults (esl_rootfinder.c:98-101).
/// Note these differ from `esl_rootfinder_Create()`'s 1e-12 (esl_rootfinder.c:66-67).
const ABS_TOLERANCE: f64 = 1e-15;
const REL_TOLERANCE: f64 = 1e-15;
const RESIDUAL_TOL: f64 = 0.0;
const MAX_ITER: u32 = 100;

/// Faithful port of Easel `esl_root_NewtonRaphson()` (esl_rootfinder.c:315) as
/// configured by `esl_rootfinder_CreateFDF()` (esl_rootfinder.c:101).
///
/// `fdf(x) -> (f(x), f'(x))` mirrors C's `(*R->fdf)(x, params, &fx, &dfx)`.
/// The loop shape is C's exactly: evaluate once before looping, then on each
/// iteration take the step, re-evaluate, and only then test convergence. In
/// particular the step is always taken at least once, and the convergence test
/// compares against `R->x` *after* the step (not before), which is what C's
/// `rel_tolerance * R->x` term reads.
///
/// Returns `Err` where C throws `eslENOHALT` ("failed to converge in Newton").
pub fn newton_raphson<F>(mut fdf: F, guess: f64) -> Result<f64, String>
where
    F: FnMut(f64) -> (f64, f64),
{
    let mut x = guess;
    let (mut fx, mut dfx) = fdf(x);

    let mut iter: u32 = 0;
    loop {
        iter += 1;
        if iter > MAX_ITER {
            return Err("failed to converge in Newton".to_string());
        }

        // Take a Newton/Raphson step.
        let x0 = x;
        x -= fx / dfx;
        let next = fdf(x);
        fx = next.0;
        dfx = next.1;

        // Test for convergence.
        if fx == 0.0 {
            break; // an exact root, lucky
        }
        if (x - x0).abs() < ABS_TOLERANCE + REL_TOLERANCE * x || fx.abs() < RESIDUAL_TOL {
            break;
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Easel's own `quadratic_fdf` (esl_rootfinder.c:364) with the
    /// `utest_Newton` parameters (esl_rootfinder.c:407-410):
    /// `(5x-1)(x+2) = 5x^2 + 9x - 2`, roots 0.2 and -2.
    fn quadratic(x: f64) -> (f64, f64) {
        (5.0 * x * x + 9.0 * x - 2.0, 10.0 * x + 9.0)
    }

    /// `utest_Newton` (esl_rootfinder.c:413-414): guess 1.0 finds 0.2, and C
    /// asserts to within `R->abs_tolerance` (1e-15).
    #[test]
    fn newton_finds_positive_root() {
        let x = newton_raphson(quadratic, 1.0).unwrap();
        assert!((x - 0.2).abs() <= ABS_TOLERANCE, "got {x}");
    }

    /// `utest_Newton` (esl_rootfinder.c:418-419): guess -3.0 finds -2.0.
    #[test]
    fn newton_finds_negative_root() {
        let x = newton_raphson(quadratic, -3.0).unwrap();
        assert!((x + 2.0).abs() <= ABS_TOLERANCE, "got {x}");
    }

    #[test]
    fn newton_reports_nonconvergence() {
        // f(x) = exp(x) has no real root; Newton walks off toward -inf and the
        // step test never fires, so C would raise eslENOHALT.
        let err = newton_raphson(|x: f64| (x.exp(), x.exp()), 0.0).unwrap_err();
        assert!(err.contains("converge"), "unexpected: {err}");
    }
}
