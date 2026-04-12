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

/// Simple LCG-based random number generator for reproducible calibration.
/// Produces uniform doubles in [0,1).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 42 } else { seed },
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Generate a random digital residue according to background frequencies.
    fn sample_residue(&mut self, bg_f: &[f32]) -> Dsq {
        let r = self.next_f64() as f32;
        let mut cumsum = 0.0_f32;
        for (i, &f) in bg_f.iter().enumerate() {
            cumsum += f;
            if r < cumsum {
                return i as Dsq;
            }
        }
        (bg_f.len() - 1) as Dsq
    }

    /// Generate a random digital sequence of given length.
    fn random_seq(&mut self, l: usize, bg_f: &[f32]) -> Vec<Dsq> {
        let mut dsq = Vec::with_capacity(l + 2);
        dsq.push(DSQ_SENTINEL);
        for _ in 0..l {
            dsq.push(self.sample_residue(bg_f));
        }
        dsq.push(DSQ_SENTINEL);
        dsq
    }
}

/// Compute the lambda parameter for the model.
/// This is the expected entropy per match position.
pub fn p7_lambda(hmm: &Hmm, bg: &Bg) -> f32 {
    let k = hmm.abc_k;
    let mut h = 0.0_f32; // mean relative entropy

    for node in 1..=hmm.m {
        let mut node_h = 0.0_f32;
        for x in 0..k {
            let p = hmm.mat[node][x];
            if p > 0.0 && bg.f[x] > 0.0 {
                node_h += p * (p / bg.f[x]).ln();
            }
        }
        h += node_h;
    }
    h /= hmm.m as f32;

    // lambda with edge-correction
    let log2 = 2.0_f32.ln();
    log2 + 1.44 / (hmm.m as f32 * h.max(0.01))
}

/// Calibrate E-value parameters for an HMM.
/// Simulates random sequences and fits Gumbel/exponential distributions.
pub fn calibrate(hmm: &mut Hmm, abc: &Alphabet, bg: &Bg) {
    crate::logsum::p7_flogsuminit();

    let lambda = p7_lambda(hmm, bg);
    let mut rng = Rng::new(42);

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

fn calibrate_msv(hmm: &Hmm, abc: &crate::alphabet::Alphabet, bg: &Bg, lambda: f32, rng: &mut Rng) -> f32 {
    let n = 200;
    let l = 200;

    let mut gm = Profile::new(hmm.m, abc);
    profile_config(hmm, bg, &mut gm, l as i32, P7_LOCAL);
    let om = OProfile::convert(&gm);

    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        let dsq = rng.random_seq(l, &bg.f);
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

fn calibrate_viterbi(hmm: &Hmm, abc: &crate::alphabet::Alphabet, bg: &Bg, lambda: f32, rng: &mut Rng) -> f32 {
    let n = 200;
    let l = 200;

    let mut gm = Profile::new(hmm.m, abc);
    profile_config(hmm, bg, &mut gm, l as i32, P7_LOCAL);
    let om = OProfile::convert(&gm);

    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        let dsq = rng.random_seq(l, &bg.f);
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
    rng: &mut Rng,
) -> (f32, f32) {
    let n = 200;  // EfN
    let l = 100;  // EfL
    let tailp = 0.04_f64; // Eft

    let mut gm = Profile::new(hmm.m, abc);
    profile_config(hmm, bg, &mut gm, l as i32, P7_LOCAL);
    let om = OProfile::convert(&gm);

    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        let dsq = rng.random_seq(l, &bg.f);
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

    // Convert to exponential tail tau
    let tau = stats::gumbel::invcdf(1.0 - tailp, gmu, glam)
        + (tailp.ln() / lambda as f64);

    let tau = if tau.is_finite() { tau as f32 } else { -3.0 };
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
