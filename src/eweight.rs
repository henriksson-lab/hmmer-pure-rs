//! Effective sequence number estimation.
//! Port of eweight.c — entropy-based Neff computation.

use crate::bg::Bg;
use crate::hmm::Hmm;
use crate::util::cmath::{c_log_f64, ESL_CONST_LOG2R};

/// Target relative entropy per match position (bits).
const ETARGET_AMINO: f64 = 0.59;
const ETARGET_DNA: f64 = 0.62;
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
    target_re: Option<f64>,
    target_sigma: Option<f64>,
) -> f32 {
    let target = target_re.unwrap_or(default_re_target(hmm));
    let etarget = entropy_target(hmm, target, target_sigma.unwrap_or(ESIGMA_DEFAULT));

    let nseq = hmm.nseq as f64;
    if nseq <= 0.0 {
        return 1.0;
    }

    let fx_full = relative_entropy_fx(hmm, bg, nseq, etarget);
    if fx_full <= 0.0 {
        hmm.eff_nseq = nseq as f32;
        return nseq as f32;
    }

    let mut lo = 0.0_f64;
    let mut hi = nseq;
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        let fx = relative_entropy_fx(hmm, bg, mid, etarget);
        if fx > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo).abs() < 0.01 {
            break;
        }
    }

    let neff = ((lo + hi) / 2.0) as f32;
    hmm.eff_nseq = neff;
    neff
}

/// Determine exponential column-count scaling for entropy weighting.
///
/// Counterpart to C's `p7_EntropyWeight_exp()` plus the subsequent
/// `p7_hmm_ScaleExponential()` call in the builder.
pub fn entropy_weight_exp(
    hmm: &mut Hmm,
    bg: &Bg,
    target_re: Option<f64>,
    target_sigma: Option<f64>,
) -> f32 {
    let target = target_re.unwrap_or(default_re_target(hmm));
    let etarget = entropy_target(hmm, target, target_sigma.unwrap_or(ESIGMA_DEFAULT));

    let fx_full = relative_entropy_exp_fx(hmm, bg, 1.0, etarget);
    let exp = if fx_full > 0.0 {
        let mut lo = 0.0_f64;
        let mut hi = 1.0_f64;
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            let fx = relative_entropy_exp_fx(hmm, bg, mid, etarget);
            if fx > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
            if (hi - lo).abs() < 0.001 {
                break;
            }
        }
        (lo + hi) / 2.0
    } else {
        1.0
    };

    scale_counts_exponential(hmm, exp);
    let neff = mean_match_count(hmm) as f32;
    hmm.eff_nseq = neff;
    neff
}

/// Pick the default relative-entropy target by alphabet (amino vs nucleic).
fn default_re_target(hmm: &Hmm) -> f64 {
    if hmm.abc_k == 20 {
        ETARGET_AMINO
    } else {
        ETARGET_DNA
    }
}

/// Apply Karplus-style entropy-target floor: at least `esigma / M` bits/match.
fn entropy_target(hmm: &Hmm, re_target: f64, esigma: f64) -> f64 {
    let m = hmm.m as f64;
    let sigma_target = (esigma - LOG2_INV * c_log_f64(2.0 / (m * (m + 1.0)))) / m;
    re_target.max(sigma_target)
}

/// Evaluate `f(Neff) = mean_match_re(scaled HMM) - etarget`; root sought by bisection.
fn relative_entropy_fx(hmm: &Hmm, bg: &Bg, neff: f64, etarget: f64) -> f64 {
    let mut trial = hmm.clone();
    let nseq = trial.nseq as f64;
    scale_counts(&mut trial, neff / nseq);
    crate::prior::apply_priors(&mut trial);
    mean_match_relative_entropy(&trial, bg) - etarget
}

fn relative_entropy_exp_fx(hmm: &Hmm, bg: &Bg, exp: f64, etarget: f64) -> f64 {
    let mut trial = hmm.clone();
    scale_counts_exponential(&mut trial, exp);
    crate::prior::apply_priors(&mut trial);
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
            (count as f64).powf(exp) / count as f64
        } else {
            1.0
        };
        for t in &mut hmm.t[k] {
            *t *= scale as f32;
        }
        for x in 0..hmm.abc_k {
            hmm.mat[k][x] *= scale as f32;
            hmm.ins[k][x] *= scale as f32;
        }
    }
}

fn mean_match_count(hmm: &Hmm) -> f64 {
    if hmm.m == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for k in 1..=hmm.m {
        total += hmm.mat[k][..hmm.abc_k]
            .iter()
            .map(|&v| v as f64)
            .sum::<f64>();
    }
    total / hmm.m as f64
}

/// Mean per-position KL divergence (bits) between match emissions and background.
fn mean_match_relative_entropy(hmm: &Hmm, bg: &Bg) -> f64 {
    let mut kl = 0.0_f64;
    for k in 1..=hmm.m {
        for x in 0..hmm.abc_k {
            let p = hmm.mat[k][x] as f64;
            let q = bg.f[x] as f64;
            if p > 0.0 && q > 0.0 {
                kl += p * c_log_f64(p / q) * ESL_CONST_LOG2R;
            }
        }
    }
    kl / hmm.m as f64
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

        let neff = entropy_weight(&mut hmm, &bg, None, None);
        assert!(
            neff > 0.0 && neff <= hmm.nseq as f32,
            "Neff {} should be between 0 and nseq={}",
            neff,
            hmm.nseq
        );
    }
}
