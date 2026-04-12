//! Effective sequence number estimation.
//! Port of eweight.c — entropy-based Neff computation.

use crate::bg::Bg;
use crate::hmm::Hmm;

/// Target relative entropy per match position (bits).
const ETARGET_AMINO: f64 = 0.59;
const ETARGET_DNA: f64 = 0.62;

/// Compute the effective sequence number for an HMM using entropy weighting.
/// Adjusts the effective sequence count so that the mean match-state relative
/// entropy matches a target value.
///
/// Returns the effective sequence number (Neff).
pub fn entropy_weight(hmm: &mut Hmm, bg: &Bg, target_re: Option<f64>) -> f32 {
    let k = hmm.abc_k;
    let target = target_re.unwrap_or(if k == 20 { ETARGET_AMINO } else { ETARGET_DNA });

    let nseq = hmm.nseq as f64;
    if nseq <= 0.0 {
        return 1.0;
    }

    // Binary search for Neff that gives target relative entropy
    let mut lo = 0.0_f64;
    let mut hi = nseq;

    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if mid < 0.01 {
            break;
        }

        // Scale counts by mid/nseq and recompute relative entropy
        let re = mean_relative_entropy_at_neff(hmm, bg, mid, nseq);

        if re > target {
            // Too much information → need lower Neff (more smoothing)
            hi = mid;
        } else {
            // Too little information → need higher Neff (less smoothing)
            lo = mid;
        }

        if (re - target).abs() < 0.001 {
            break;
        }
    }

    let neff = ((lo + hi) / 2.0) as f32;
    hmm.eff_nseq = neff;
    neff
}

/// Compute mean match-state relative entropy at a given effective sequence number.
fn mean_relative_entropy_at_neff(hmm: &Hmm, bg: &Bg, neff: f64, nseq: f64) -> f64 {
    let k = hmm.abc_k;
    let scale = neff / nseq;
    let mut total_re = 0.0_f64;

    for node in 1..=hmm.m {
        let mut node_re = 0.0_f64;
        for x in 0..k {
            // Scale the emission probability by neff
            // p_scaled ∝ (count * scale + prior) / (total_scaled + prior_total)
            // Simplified: interpolate between observed and background
            let p_obs = hmm.mat[node][x] as f64;
            let p_bg = bg.f[x] as f64;
            let p = p_obs * scale + p_bg * (1.0 - scale);
            let p = p.max(1e-10);

            if p > 0.0 && p_bg > 0.0 {
                node_re += p * (p / p_bg).log2();
            }
        }
        total_re += node_re;
    }

    total_re / hmm.m as f64
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

        let neff = entropy_weight(&mut hmm, &bg, None);
        assert!(neff > 0.0 && neff <= hmm.nseq as f32,
            "Neff {} should be between 0 and nseq={}", neff, hmm.nseq);
    }
}
