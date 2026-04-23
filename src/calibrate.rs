//! E-value calibration by simulation.
//! Port of evalues.c p7_Calibrate().

use crate::alphabet::{Alphabet, Dsq, DSQ_SENTINEL};
use crate::bg::Bg;
use crate::dp::generic_fwdback::g_forward;
use crate::dp::generic_msv::g_msv;
use crate::dp::generic_viterbi::g_viterbi;
use crate::dp::gmx::Gmx;
use crate::hmm::*;
use crate::profile::*;
use crate::simd::oprofile::OProfile;
use crate::stats;

use crate::util::random::MersenneTwister;

/// Generate a random digital sequence of given length using MT RNG.
fn random_seq(rng: &mut MersenneTwister, l: usize, bg_f: &[f32]) -> Vec<Dsq> {
    let mut dsq = Vec::with_capacity(l + 2);
    dsq.push(DSQ_SENTINEL);
    for _ in 0..l {
        dsq.push(rng.sample_residue(bg_f));
    }
    dsq.push(DSQ_SENTINEL);
    dsq
}

/// Compute the lambda parameter for the model.
/// This is the expected entropy per match position.
pub fn p7_lambda(hmm: &Hmm, bg: &Bg) -> f32 {
    let k = hmm.abc_k;
    // Use f64 precision for entropy computation in bits (log2), matching C's double
    let mut h = 0.0_f64;

    for node in 1..=hmm.m {
        let mut node_h = 0.0_f64;
        for x in 0..k {
            let p = hmm.mat[node][x] as f64;
            let f = bg.f[x] as f64;
            if p > 0.0 && f > 0.0 {
                node_h += p * (p / f).log2();
            }
        }
        h += node_h;
    }
    h /= hmm.m as f64;

    // lambda with edge-correction: log(2) + 1.44 / (M * H)
    let log2 = std::f64::consts::LN_2;
    (log2 + 1.44 / (hmm.m as f64 * h.max(0.01))) as f32
}

/// Calibrate E-value parameters for an HMM.
/// Simulates random sequences and fits Gumbel/exponential distributions.
pub fn calibrate(hmm: &mut Hmm, abc: &Alphabet, bg: &Bg) {
    crate::logsum::p7_flogsuminit();

    let lambda = p7_lambda(hmm, bg);
    let mut rng = MersenneTwister::new(42);

    // MSV calibration
    let mmu = calibrate_msv(hmm, abc, bg, lambda, &mut rng);
    hmm.evparam[P7_MMU] = mmu;
    hmm.evparam[P7_MLAMBDA] = lambda;

    // Viterbi calibration
    let vmu = calibrate_viterbi(hmm, abc, bg, lambda, &mut rng);
    hmm.evparam[P7_VMU] = vmu;
    hmm.evparam[P7_VLAMBDA] = lambda;

    // Forward calibration
    let (ftau, flambda) = calibrate_forward(hmm, abc, bg, lambda, &mut rng);
    hmm.evparam[P7_FTAU] = ftau;
    hmm.evparam[P7_FLAMBDA] = flambda;

    hmm.flags |= P7H_STATS;
}

fn calibrate_msv(
    hmm: &Hmm,
    abc: &crate::alphabet::Alphabet,
    bg: &Bg,
    lambda: f32,
    rng: &mut MersenneTwister,
) -> f32 {
    let n = 200;
    let l = 200;
    let mut bg = bg.clone();
    bg.set_length(l);

    let mut gm = Profile::new(hmm.m, abc);
    profile_config(hmm, &bg, &mut gm, l as i32, P7_LOCAL);
    let om = OProfile::convert(&gm);

    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        let dsq = random_seq(rng, l, &bg.f);
        let sc;
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                sc = match unsafe { crate::simd::msv_filter::msv_filter(&dsq, l, &om) } {
                    crate::simd::msv_filter::MsvResult::Ok(s) => s,
                    crate::simd::msv_filter::MsvResult::Overflow => f32::INFINITY,
                };
            } else {
                let mut gx = Gmx::new(hmm.m, l);
                sc = g_msv(&dsq, l, &gm, &mut gx, 2.0);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut gx = Gmx::new(hmm.m, l);
            sc = g_msv(&dsq, l, &gm, &mut gx, 2.0);
        }
        let null_sc = bg.null_one(l);
        let bits = ((sc - null_sc) / std::f32::consts::LN_2) as f64;
        scores.push(bits);
    }

    // Filter out non-finite scores
    scores.retain(|s| s.is_finite());
    if scores.len() < 10 {
        return -3.0; // not enough data
    }

    let mu = stats::gumbel::fit_complete_loc(&scores, lambda as f64).unwrap_or(-3.0);
    mu as f32
}

fn calibrate_viterbi(
    hmm: &Hmm,
    abc: &crate::alphabet::Alphabet,
    bg: &Bg,
    lambda: f32,
    rng: &mut MersenneTwister,
) -> f32 {
    let n = 200;
    let l = 200;
    let mut bg = bg.clone();
    bg.set_length(l);

    let mut gm = Profile::new(hmm.m, abc);
    profile_config(hmm, &bg, &mut gm, l as i32, P7_LOCAL);
    let om = OProfile::convert(&gm);

    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        let dsq = random_seq(rng, l, &bg.f);
        let sc;
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                sc = match unsafe { crate::simd::vit_filter::viterbi_filter(&dsq, l, &om) } {
                    crate::simd::vit_filter::VitResult::Ok(s) => s,
                    crate::simd::vit_filter::VitResult::Overflow => f32::INFINITY,
                };
            } else {
                let mut gx = Gmx::new(hmm.m, l);
                sc = g_viterbi(&dsq, l, &gm, &mut gx);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut gx = Gmx::new(hmm.m, l);
            sc = g_viterbi(&dsq, l, &gm, &mut gx);
        }
        let null_sc = bg.null_one(l);
        let bits = ((sc - null_sc) / std::f32::consts::LN_2) as f64;
        scores.push(bits);
    }

    scores.retain(|s| s.is_finite());
    if scores.len() < 10 {
        return -3.0;
    }

    let mu = stats::gumbel::fit_complete_loc(&scores, lambda as f64).unwrap_or(-3.0);
    mu as f32
}

fn calibrate_forward(
    hmm: &Hmm,
    abc: &crate::alphabet::Alphabet,
    bg: &Bg,
    lambda: f32,
    rng: &mut MersenneTwister,
) -> (f32, f32) {
    let n = 200; // EfN
    let l = 100; // EfL
    let tailp = 0.04_f64; // Eft
    let mut bg = bg.clone();
    bg.set_length(l);

    let mut gm = Profile::new(hmm.m, abc);
    profile_config(hmm, &bg, &mut gm, l as i32, P7_LOCAL);
    let om = OProfile::convert(&gm);

    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        let dsq = random_seq(rng, l, &bg.f);
        // Use SIMD Forward when available for speed
        let sc;
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                sc = unsafe { crate::simd::fwd_filter::forward_parser(&dsq, l, &om) };
            } else {
                let mut gx = Gmx::new(hmm.m, l);
                sc = g_forward(&dsq, l, &gm, &mut gx);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut gx = Gmx::new(hmm.m, l);
            sc = g_forward(&dsq, l, &gm, &mut gx);
        }
        let null_sc = bg.null_one(l);
        let bits = ((sc - null_sc) / std::f32::consts::LN_2) as f64;
        scores.push(bits);
    }

    // Filter out non-finite scores
    scores.retain(|s| s.is_finite());
    if scores.len() < 10 {
        return (-3.0, lambda);
    }

    // Fit Gumbel to Forward scores
    let (gmu, glam) = stats::gumbel::fit_complete(&scores).unwrap_or((-3.0, lambda as f64));

    // C code: tau = esl_gumbel_invcdf(1.0-tailp, gmu, glam) + (log(tailp) / lambda)
    // First find x where Gumbel tail mass = tailp, then back up by log(tailp)/lambda
    // to set the origin of the exponential tail to 1.0 instead of tailp.
    let tau = stats::gumbel::invcdf(1.0 - tailp, gmu, glam) + (tailp.ln() / lambda as f64);

    let tau = if tau.is_finite() { tau as f32 } else { -3.0 };
    // C stores the HMM-derived lambda for all three: MLAMBDA = VLAMBDA = FLAMBDA = lambda
    (tau, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_calibrate() {
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

        // Save original params
        let orig_mmu = hmm.evparam[P7_MMU];

        // Re-calibrate
        calibrate(&mut hmm, &abc, &bg);

        // Params should be in reasonable range
        assert!(hmm.evparam[P7_MMU].is_finite());
        assert!(hmm.evparam[P7_MLAMBDA] > 0.0);
        assert!(hmm.evparam[P7_VMU].is_finite());
        assert!(hmm.evparam[P7_FTAU].is_finite());

        // Should be in the same ballpark as original (within ~5)
        assert!(
            (hmm.evparam[P7_MMU] - orig_mmu).abs() < 5.0,
            "Calibrated MSV mu={} vs original={}, too different",
            hmm.evparam[P7_MMU],
            orig_mmu
        );
    }
}

#[cfg(test)]
mod lambda_test {
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use std::path::Path;

    #[test]
    fn check_lambda_directly() {
        let hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        let abc = Alphabet::new(hmm.abc_type);
        let bg = Bg::new(&abc);

        // Compute H (mean relative entropy) in bits (log2) to match C
        let mut h = 0.0_f64;
        for node in 1..=hmm.m {
            for x in 0..hmm.abc_k {
                let p = hmm.mat[node][x] as f64;
                let f = bg.f[x] as f64;
                if p > 0.0 && f > 0.0 {
                    h += p * (p / f).log2();
                }
            }
        }
        h /= hmm.m as f64;
        let lambda = std::f64::consts::LN_2 + 1.44 / (hmm.m as f64 * h);
        eprintln!("H={:.6} bits, lambda={:.5} (C expects 0.72049)", h, lambda);

        // Lambda should be close to C's 0.72049
        assert!(
            (lambda - 0.72049).abs() < 0.001,
            "lambda={} too far from C's 0.72049",
            lambda
        );
    }
}
