//! Effective sequence number estimation.
//! Port of eweight.c — entropy-based Neff computation.

use crate::bg::Bg;
use crate::hmm::Hmm;
use crate::prior::PriorStrategy;
use crate::util::cmath::{c_log_f64, ESL_CONST_LOG2R};

/// Target relative entropy per match position (bits).
/// (p7_config.h:61-63: `p7_ETARGET_{AMINO,DNA,OTHER}`)
const ETARGET_AMINO: f64 = 0.59;
const ETARGET_DNA: f64 = 0.62;
const ETARGET_OTHER: f64 = 1.0;
const ESIGMA_DEFAULT: f64 = 45.0;
const LOG2_INV: f64 = ESL_CONST_LOG2R;

/// Determine the effective sequence number Neff by entropy weighting.
///
/// Bisects on Neff so that the resulting parameterized HMM's mean match-state
/// relative entropy matches the target (in bits). Caller passes a count-based
/// HMM; the HMM itself is not modified except for `eff_nseq`.
/// Returns Neff in `[0, hmm.nseq]`. Counterpart to C's `p7_EntropyWeight()`.
pub fn entropy_weight(
    hmm: &mut Hmm,
    bg: &Bg,
    prior: PriorStrategy,
    target_re: Option<f64>,
    target_sigma: Option<f64>,
) -> f32 {
    let target = target_re.unwrap_or(default_re_target(hmm));
    let etarget = entropy_target(hmm, target, target_sigma.unwrap_or(ESIGMA_DEFAULT));

    let nseq = hmm.nseq as f64;
    if nseq <= 0.0 {
        return 1.0;
    }

    let fx_full = relative_entropy_fx(hmm, bg, prior, nseq, etarget);
    if fx_full <= 0.0 {
        hmm.eff_nseq = nseq as f32;
        return nseq as f32;
    }

    // esl_root_Bisection (esl_rootfinder.c:244) with abs_tol=0.01 (eweight.c:82),
    // rel_tol=1e-12, residual_tol=0, max_iter=100 (rootfinder Create defaults).
    let neff = bisection(0.0, nseq, 0.01, |x| {
        relative_entropy_fx(hmm, bg, prior, x, etarget)
    }) as f32;
    hmm.eff_nseq = neff;
    neff
}

/// Faithful port of Easel `esl_root_Bisection` (esl_rootfinder.c:244) plus
/// `esl_rootfinder_SetBrackets` (esl_rootfinder.c:153). Returns the last
/// evaluated midpoint `R->x`, testing convergence after each function
/// evaluation but before narrowing the bracket, exactly as C does.
/// `rel_tolerance = 1e-12`, `residual_tol = 0`, `max_iter = 100`.
fn bisection<F: FnMut(f64) -> f64>(mut xl: f64, mut xr: f64, abs_tol: f64, mut func: F) -> f64 {
    const REL_TOL: f64 = 1e-12;
    const MAX_ITER: u32 = 100;

    let mut fl = func(xl);
    let _fr = func(xr);

    let mut x = (xl + xr) / 2.0;
    let mut iter = 0u32;
    loop {
        iter += 1;
        if iter > MAX_ITER {
            break; // C raises eslENOHALT; we return the last midpoint instead.
        }
        x = (xl + xr) / 2.0;
        let fx = func(x);

        let xmag = if xl < 0.0 && xr > 0.0 { 0.0 } else { x };
        if fx == 0.0 {
            break;
        }
        if (xr - xl) < abs_tol + REL_TOL * xmag {
            break;
        }

        if fl > 0.0 {
            if fx > 0.0 {
                xl = x;
                fl = fx;
            } else {
                xr = x;
            }
        } else if fx < 0.0 {
            xl = x;
            fl = fx;
        } else {
            xr = x;
        }
    }
    x
}

/// Determine exponential column-count scaling for entropy weighting.
///
/// Counterpart to C's `p7_EntropyWeight_exp()` plus the subsequent
/// `p7_hmm_ScaleExponential()` call in the builder.
pub fn entropy_weight_exp(
    hmm: &mut Hmm,
    bg: &Bg,
    prior: PriorStrategy,
    target_re: Option<f64>,
    target_sigma: Option<f64>,
) -> f32 {
    let target = target_re.unwrap_or(default_re_target(hmm));
    let etarget = entropy_target(hmm, target, target_sigma.unwrap_or(ESIGMA_DEFAULT));

    let fx_full = relative_entropy_exp_fx(hmm, bg, prior, 1.0, etarget);
    let exp = if fx_full > 0.0 {
        // esl_root_Bisection over [0,1] with abs_tol=0.001 (eweight.c:164).
        bisection(0.0, 1.0, 0.001, |x| {
            relative_entropy_exp_fx(hmm, bg, prior, x, etarget)
        })
    } else {
        1.0
    };

    scale_counts_exponential(hmm, exp);
    let neff = mean_match_count(hmm) as f32;
    hmm.eff_nseq = neff;
    neff
}

/// Pick the default relative-entropy target by alphabet *type*, matching the
/// C builder's switch (p7_builder.c:99-104): amino -> 0.59, DNA/RNA -> 0.62,
/// anything else (custom/unknown) -> 1.0.
fn default_re_target(hmm: &Hmm) -> f64 {
    use crate::alphabet::AlphabetType;
    match hmm.abc_type {
        AlphabetType::Amino => ETARGET_AMINO,
        AlphabetType::Dna | AlphabetType::Rna => ETARGET_DNA,
        AlphabetType::Unknown => ETARGET_OTHER,
    }
}

/// Apply Karplus-style entropy-target floor: at least `esigma / M` bits/match.
fn entropy_target(hmm: &Hmm, re_target: f64, esigma: f64) -> f64 {
    let m = hmm.m as f64;
    let sigma_target = (esigma - LOG2_INV * c_log_f64(2.0 / (m * (m + 1.0)))) / m;
    re_target.max(sigma_target)
}

/// Evaluate `f(Neff) = mean_match_re(scaled HMM) - etarget`; root sought by bisection.
fn relative_entropy_fx(hmm: &Hmm, bg: &Bg, prior: PriorStrategy, neff: f64, etarget: f64) -> f64 {
    let mut trial = hmm.clone();
    let nseq = trial.nseq as f64;
    scale_counts(&mut trial, neff / nseq);
    crate::prior::apply_priors_with_strategy(&mut trial, prior);
    mean_match_relative_entropy(&trial, bg) - etarget
}

fn relative_entropy_exp_fx(
    hmm: &Hmm,
    bg: &Bg,
    prior: PriorStrategy,
    exp: f64,
    etarget: f64,
) -> f64 {
    let mut trial = hmm.clone();
    scale_counts_exponential(&mut trial, exp);
    crate::prior::apply_priors_with_strategy(&mut trial, prior);
    mean_match_relative_entropy(&trial, bg) - etarget
}

/// Multiply all transition and emission counts in `hmm` by `scale` in place.
/// Used to rescale a count-based HMM to its effective sequence number.
pub fn scale_counts(hmm: &mut Hmm, scale: f64) {
    for k in 0..=hmm.m {
        for t in &mut hmm.t[k] {
            *t *= scale as f32;
        }
        for x in 0..hmm.abc_k {
            hmm.mat[k][x] *= scale as f32;
            hmm.ins[k][x] *= scale as f32;
        }
    }
}

fn scale_counts_exponential(hmm: &mut Hmm, exp: f64) {
    for k in 1..=hmm.m {
        let count: f32 = hmm.mat[k][..hmm.abc_k].iter().sum();
        let scale = if count > 0.0 {
            let new_count = (count as f64).powf(exp) as f32;
            new_count / count
        } else {
            1.0_f32
        };
        for t in &mut hmm.t[k] {
            *t *= scale;
        }
        for x in 0..hmm.abc_k {
            hmm.mat[k][x] *= scale;
            hmm.ins[k][x] *= scale;
        }
    }
}

fn mean_match_count(hmm: &Hmm) -> f64 {
    if hmm.m == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for k in 1..=hmm.m {
        let row_sum: f32 = hmm.mat[k][..hmm.abc_k].iter().sum();
        total += row_sum as f64;
    }
    total / hmm.m as f64
}

/// Mean per-position KL divergence (bits) between match emissions and background.
///
/// Faithful to `p7_MeanMatchRelativeEntropy` (modelstats.c:80) +
/// `esl_vec_FRelEntropy` (esl_vectorops.c:1438): each match state's KL is
/// summed in **f32** using `log2`, with float-precision `p/q`; a state whose
/// background prob `q==0` while `p>0` yields +inf. Those f32 per-state values
/// are then summed into an f64 and divided by M.
fn mean_match_relative_entropy(hmm: &Hmm, bg: &Bg) -> f64 {
    let mut kl_sum = 0.0_f64;
    for k in 1..=hmm.m {
        // esl_vec_FRelEntropy: accumulate this state's KL in f32.
        let mut kl = 0.0_f32;
        for x in 0..hmm.abc_k {
            let p = hmm.mat[k][x];
            let q = bg.f[x];
            if p > 0.0 {
                if q == 0.0 {
                    kl = f32::INFINITY;
                    break;
                }
                // C: p/q in float, log2() in double, product in double, then
                // accumulated into the f32 kl (truncating each iteration).
                kl += (p as f64 * ((p / q) as f64).log2()) as f32;
            }
        }
        kl_sum += kl as f64;
    }
    kl_sum / hmm.m as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use std::path::Path;

    #[test]
    fn test_entropy_weight() {
        let mut hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .into_iter()
        .next()
        .unwrap();

        let abc = Alphabet::new(hmm.abc_type);
        let bg = Bg::new(&abc);

        let neff = entropy_weight(&mut hmm, &bg, PriorStrategy::Default, None, None);
        assert!(
            neff > 0.0 && neff <= hmm.nseq as f32,
            "Neff {} should be between 0 and nseq={}",
            neff,
            hmm.nseq
        );
    }
}
