//! E-value calibration by simulation.
//! Port of evalues.c p7_Calibrate().

use crate::alphabet::{Alphabet, Dsq, DSQ_SENTINEL};
use crate::bg::Bg;
use crate::hmm::*;
use crate::profile::*;
use crate::simd::oprofile::OProfile;
use crate::stats;

use crate::util::cmath::{c_log_f64, ESL_CONST_LOG2, ESL_CONST_LOG2R};
use crate::util::random::MersenneTwister;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationConfig {
    pub em_l: usize,
    pub em_n: usize,
    pub ev_l: usize,
    pub ev_n: usize,
    pub ef_l: usize,
    pub ef_n: usize,
    pub eft: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            em_l: 200,
            em_n: 200,
            ev_l: 200,
            ev_n: 200,
            ef_l: 100,
            ef_n: 200,
            eft: 0.04,
        }
    }
}

/// Sample an iid digital sequence of length `l` from background frequencies `bg_f`.
/// Sentinels are added at positions 0 and l+1 so the result is 1-based.
fn random_seq(rng: &mut MersenneTwister, l: usize, bg_f: &[f32]) -> Vec<Dsq> {
    let mut dsq = Vec::with_capacity(l + 2);
    dsq.push(DSQ_SENTINEL);
    for _ in 0..l {
        dsq.push(rng.sample_residue(bg_f));
    }
    dsq.push(DSQ_SENTINEL);
    dsq
}

fn simulated_bitscore(sc: f32, null_sc: f32) -> f64 {
    (sc as f64 - null_sc as f64) / ESL_CONST_LOG2
}

fn msv_overflow_maxsc(base_b: u8, scale_b: f32) -> f32 {
    (255.0 - base_b as f32) / scale_b
}

fn viterbi_overflow_maxsc(base_w: i16, scale_w: f32) -> f32 {
    (32767.0 - base_w as f32) / scale_w
}

/// Score a simulated sequence with the optimized (quantized) MSV filter, mirroring
/// C's `p7_MSVMu`, which always calls `p7_MSVFilter`. Dispatches to the SSE2 filter
/// on x86_64 and the NEON filter on aarch64 (both baseline-available); `maxsc` is the
/// quantized overflow cap `(255 - base_b)/scale_b`. There is intentionally no generic
/// DP fallback, so the byte-quantized scores match C on every supported target.
fn msv_filter_score(dsq: &[Dsq], l: usize, om: &OProfile, maxsc: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            is_x86_feature_detected!("sse2"),
            "calibration requires SSE2 (baseline on x86_64) to match C's quantized MSV filter"
        );
        match unsafe { crate::simd::msv_filter::msv_filter(dsq, l, om) } {
            crate::simd::msv_filter::MsvResult::Ok(s) => s,
            crate::simd::msv_filter::MsvResult::Overflow => maxsc,
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        match unsafe { crate::simd::neon_msv::neon_msv_filter(dsq, l, om) } {
            crate::simd::neon_msv::NeonMsvResult::Ok(s) => s,
            crate::simd::neon_msv::NeonMsvResult::Overflow => maxsc,
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (dsq, l, om, maxsc);
        compile_error!(
            "calibration has no optimized MSV filter for this target; C always uses p7_MSVFilter"
        );
    }
}

/// Score a simulated sequence with the optimized (quantized) Viterbi filter, mirroring
/// C's `p7_ViterbiMu`, which always calls `p7_ViterbiFilter`. `maxsc` is the quantized
/// overflow cap `(32767 - base_w)/scale_w`.
fn viterbi_filter_score(dsq: &[Dsq], l: usize, om: &OProfile, maxsc: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            is_x86_feature_detected!("sse2"),
            "calibration requires SSE2 (baseline on x86_64) to match C's quantized Viterbi filter"
        );
        match unsafe { crate::simd::vit_filter::viterbi_filter(dsq, l, om) } {
            crate::simd::vit_filter::VitResult::Ok(s) => s,
            crate::simd::vit_filter::VitResult::Overflow => maxsc,
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        match unsafe { crate::simd::neon_vit::neon_viterbi_filter(dsq, l, om) } {
            crate::simd::neon_vit::NeonVitResult::Ok(s) => s,
            crate::simd::neon_vit::NeonVitResult::Overflow => maxsc,
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (dsq, l, om, maxsc);
        compile_error!(
            "calibration has no optimized Viterbi filter for this target; C always uses p7_ViterbiFilter"
        );
    }
}

/// Score a simulated sequence with the optimized Forward parser, mirroring C's
/// `p7_Tau`, which always calls `p7_ForwardParser`.
fn forward_filter_score(dsq: &[Dsq], l: usize, om: &OProfile) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            is_x86_feature_detected!("sse2"),
            "calibration requires SSE2 (baseline on x86_64) to match C's Forward parser"
        );
        unsafe { crate::simd::fwd_filter::forward_parser(dsq, l, om) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { crate::simd::neon_fwd::neon_forward_parser(dsq, l, om) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (dsq, l, om);
        compile_error!(
            "calibration has no optimized Forward parser for this target; C always uses p7_ForwardParser"
        );
    }
}

/// Determine the length-corrected local lambda for an HMM.
///
/// The true lambda is `log(2)`; this returns the edge-corrected estimate
/// `log(2) + 1.44 / (M * H)`, where H is the mean match-state relative
/// entropy in bits. Used for both Viterbi Gumbel and Forward exponential
/// tails. Counterpart to C's `p7_Lambda()`.
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
                node_h += p * c_log_f64(p / f) * ESL_CONST_LOG2R;
            }
        }
        h += node_h;
    }
    h /= hmm.m as f64;

    // lambda with edge-correction: log(2) + 1.44 / (M * H)
    (ESL_CONST_LOG2 + 1.44 / (hmm.m as f64 * h)) as f32
}

/// Calibrate E-value parameters for an HMM by simulation.
///
/// Computes lambda analytically, then runs short Monte Carlo simulations to
/// fit MSV Gumbel mu, Viterbi Gumbel mu, and Forward exponential-tail tau.
/// Stores results in `hmm.evparam[]` and sets P7H_STATS.
/// Counterpart to C's `p7_Calibrate()`.
pub fn calibrate(hmm: &mut Hmm, abc: &Alphabet, bg: &Bg) {
    calibrate_with_seed(hmm, abc, bg, 42);
}

/// Calibrate E-value parameters with a caller-selected Easel fast RNG seed.
pub fn calibrate_with_seed(hmm: &mut Hmm, abc: &Alphabet, bg: &Bg, seed: u32) {
    calibrate_with_config(hmm, abc, bg, seed, CalibrationConfig::default());
}

/// Calibrate E-value parameters with caller-selected simulation lengths,
/// counts, Forward tail mass, and Easel fast RNG seed.
pub fn calibrate_with_config(
    hmm: &mut Hmm,
    abc: &Alphabet,
    bg: &Bg,
    seed: u32,
    config: CalibrationConfig,
) {
    crate::logsum::p7_flogsuminit();

    let lambda = p7_lambda(hmm, bg);
    let mut rng = MersenneTwister::new(seed);

    // MSV calibration
    let mmu = calibrate_msv(hmm, abc, bg, lambda, config.em_l, config.em_n, &mut rng);
    hmm.evparam[P7_MMU] = mmu;
    hmm.evparam[P7_MLAMBDA] = lambda;

    // Viterbi calibration
    let vmu = calibrate_viterbi(hmm, abc, bg, lambda, config.ev_l, config.ev_n, &mut rng);
    hmm.evparam[P7_VMU] = vmu;
    hmm.evparam[P7_VLAMBDA] = lambda;

    // Forward calibration
    let (ftau, flambda) = calibrate_forward(
        hmm,
        abc,
        bg,
        lambda,
        config.ef_l,
        config.ef_n,
        config.eft,
        &mut rng,
    );
    hmm.evparam[P7_FTAU] = ftau;
    hmm.evparam[P7_FLAMBDA] = flambda;

    hmm.flags |= P7H_STATS;
}

/// Estimate the MSV Gumbel location parameter mu by simulating random
/// sequences and ML-fitting their MSV bit scores at fixed `lambda`.
/// Counterpart to C's `p7_MSVMu()`.
fn calibrate_msv(
    hmm: &Hmm,
    abc: &crate::alphabet::Alphabet,
    bg: &Bg,
    lambda: f32,
    l: usize,
    n: usize,
    rng: &mut MersenneTwister,
) -> f32 {
    let mut bg = bg.clone();
    bg.set_length(l);

    let mut gm = Profile::new(hmm.m, abc);
    profile_config(hmm, &bg, &mut gm, l as i32, P7_LOCAL);
    let om = OProfile::convert(&gm);
    let maxsc = msv_overflow_maxsc(om.base_b, om.scale_b);

    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        // Match C p7_MSVMu loop order: xfIID, then NullOne, then MSVFilter.
        let dsq = random_seq(rng, l, &bg.f);
        let null_sc = bg.null_one(l);
        let sc = msv_filter_score(&dsq, l, &om, maxsc);
        let bits = simulated_bitscore(sc, null_sc);
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

/// Estimate the Viterbi Gumbel location parameter mu by simulation and ML
/// fitting at fixed `lambda`. Counterpart to C's `p7_ViterbiMu()`.
fn calibrate_viterbi(
    hmm: &Hmm,
    abc: &crate::alphabet::Alphabet,
    bg: &Bg,
    lambda: f32,
    l: usize,
    n: usize,
    rng: &mut MersenneTwister,
) -> f32 {
    let mut bg = bg.clone();
    bg.set_length(l);

    let mut gm = Profile::new(hmm.m, abc);
    profile_config(hmm, &bg, &mut gm, l as i32, P7_LOCAL);
    let om = OProfile::convert(&gm);
    let maxsc = viterbi_overflow_maxsc(om.base_w, om.scale_w);

    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        // Match C p7_ViterbiMu loop order: xfIID, then NullOne, then ViterbiFilter.
        let dsq = random_seq(rng, l, &bg.f);
        let null_sc = bg.null_one(l);
        let sc = viterbi_filter_score(&dsq, l, &om, maxsc);
        let bits = simulated_bitscore(sc, null_sc);
        scores.push(bits);
    }

    scores.retain(|s| s.is_finite());
    if scores.len() < 10 {
        return -3.0;
    }

    let mu = stats::gumbel::fit_complete_loc(&scores, lambda as f64).unwrap_or(-3.0);
    mu as f32
}

/// Estimate the Forward exponential-tail location `tau` by simulation:
/// fit a Gumbel to the simulated Forward scores, then back the origin off by
/// `log(tailp)/lambda` so the right tail of mass `tailp` has origin 1.0.
/// Returns `(tau, lambda)`. Counterpart to C's `p7_Tau()`.
fn calibrate_forward(
    hmm: &Hmm,
    abc: &crate::alphabet::Alphabet,
    bg: &Bg,
    lambda: f32,
    l: usize,
    n: usize,
    tailp: f64,
    rng: &mut MersenneTwister,
) -> (f32, f32) {
    let mut bg = bg.clone();
    bg.set_length(l);

    let mut gm = Profile::new(hmm.m, abc);
    profile_config(hmm, &bg, &mut gm, l as i32, P7_LOCAL);
    let om = OProfile::convert(&gm);

    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        // Match C p7_Tau loop order: xfIID, then ForwardParser, then NullOne.
        let dsq = random_seq(rng, l, &bg.f);
        let sc = forward_filter_score(&dsq, l, &om);
        let null_sc = bg.null_one(l);
        let bits = simulated_bitscore(sc, null_sc);
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
    let tau = stats::gumbel::invcdf(1.0 - tailp, gmu, glam) + (c_log_f64(tailp) / lambda as f64);

    let tau = if tau.is_finite() { tau as f32 } else { -3.0 };
    (tau, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn simulated_bitscore_uses_double_log2() {
        let sc = 12.75_f32;
        let null_sc = -3.125_f32;
        let expected = (sc as f64 - null_sc as f64) / ESL_CONST_LOG2;

        assert_eq!(
            simulated_bitscore(sc, null_sc).to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn simd_overflow_max_scores_match_c_calibration_limits() {
        assert_eq!(
            msv_overflow_maxsc(190, (3.0 / ESL_CONST_LOG2) as f32).to_bits(),
            ((255.0_f32 - 190.0_f32) / ((3.0 / ESL_CONST_LOG2) as f32)).to_bits()
        );
        assert_eq!(
            viterbi_overflow_maxsc(12000, (500.0 / ESL_CONST_LOG2) as f32).to_bits(),
            ((32767.0_f32 - 12000.0_f32) / ((500.0 / ESL_CONST_LOG2) as f32)).to_bits()
        );
    }

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
    use crate::util::cmath::{c_log_f64, ESL_CONST_LOG2, ESL_CONST_LOG2R};
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
                    h += p * c_log_f64(p / f) * ESL_CONST_LOG2R;
                }
            }
        }
        h /= hmm.m as f64;
        let lambda = ESL_CONST_LOG2 + 1.44 / (hmm.m as f64 * h);
        eprintln!("H={:.6} bits, lambda={:.5} (C expects 0.72049)", h, lambda);

        // Lambda should be close to C's 0.72049
        assert!(
            (lambda - 0.72049).abs() < 0.001,
            "lambda={} too far from C's 0.72049",
            lambda
        );
    }
}
