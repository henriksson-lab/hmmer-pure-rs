//! Null2 bias correction by expectation from posterior probabilities.
//! Port of generic_null2.c p7_GNull2_ByExpectation().
//!
//! The null2 correction compensates for composition bias by computing the
//! expected emission distribution from the posterior-weighted model, then
//! comparing it to the background model.

use crate::dp::gmx::*;
use crate::profile::Profile;
use crate::trace::{State, Trace};

/// Compute null2 odds ratios from posterior probabilities.
/// Returns null2[0..K-1] where null2[x] = f'(x) / f(x).
pub fn null2_by_expectation(gm: &Profile, pp: &Gmx, ienv: usize, jenv: usize) -> Vec<f32> {
    let m = gm.m;
    let k = gm.abc_k;
    let ld = jenv - ienv + 1;

    // Accumulate expected state usage over the envelope.
    let mut exp_m = vec![0.0_f32; m + 1];
    let mut exp_i = vec![0.0_f32; m + 1];
    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;

    for i in ienv..=jenv {
        for node in 1..=m {
            exp_m[node] += pp.mmx(i, node);
            exp_i[node] += pp.imx(i, node);
        }
        exp_n += pp.xmx(i, P7G_N);
        exp_j += pp.xmx(i, P7G_J);
        exp_c += pp.xmx(i, P7G_C);
    }

    // Convert expected counts to frequencies.
    let norm = 1.0 / ld as f32;
    for node in 1..=m {
        exp_m[node] *= norm;
        exp_i[node] *= norm;
    }
    exp_n *= norm;
    exp_c *= norm;
    exp_j *= norm;

    let xfactor = crate::logsum::p7_flogsum(
        crate::logsum::p7_flogsum(exp_n.ln(), exp_c.ln()),
        exp_j.ln(),
    );

    // C generic_null2.c forms the weighted emission odds in log-space,
    // then exponentiates. Keep the order and state ranges aligned with C.
    let mut null2 = vec![f32::NEG_INFINITY; k];
    for x in 0..k {
        for node in 1..m {
            null2[x] = crate::logsum::p7_flogsum(null2[x], exp_m[node].ln() + gm.msc(node, x));
            null2[x] = crate::logsum::p7_flogsum(null2[x], exp_i[node].ln() + gm.isc(node, x));
        }
        null2[x] = crate::logsum::p7_flogsum(null2[x], exp_m[m].ln() + gm.msc(m, x));
        null2[x] = crate::logsum::p7_flogsum(null2[x], xfactor);
        null2[x] = null2[x].exp();
    }

    null2
}

/// Calculate total null2 correction score for a domain envelope.
/// Returns the correction in nats (to be subtracted from domain score).
pub fn null2_score(null2: &[f32], dsq: &[u8], ienv: usize, jenv: usize) -> f32 {
    let mut score = 0.0_f32;
    for i in ienv..=jenv {
        let x = dsq[i] as usize;
        if x < null2.len() && null2[x] > 0.0 {
            score += null2[x].ln();
        }
    }
    score
}

/// Compute null2 odds ratios from a stochastic traceback segment.
/// This is the generic counterpart of C HMMER's p7_GNull2_ByTrace().
pub fn null2_by_trace(gm: &Profile, tr: &Trace, zstart: usize, zend: usize) -> Vec<f32> {
    let m = gm.m;
    let k = gm.abc_k;
    let mut exp_m = vec![0.0_f32; m + 1];
    let mut exp_i = vec![0.0_f32; m + 1];
    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;
    let mut ld = 0usize;

    for z in zstart..=zend.min(tr.n.saturating_sub(1)) {
        match tr.st[z] {
            State::M => {
                ld += 1;
                exp_m[tr.k[z]] += 1.0;
            }
            State::I => {
                ld += 1;
                exp_i[tr.k[z]] += 1.0;
            }
            State::N if z > 0 && tr.st[z - 1] == State::N => {
                ld += 1;
                exp_n += 1.0;
            }
            State::C if z > 0 && tr.st[z - 1] == State::C => {
                ld += 1;
                exp_c += 1.0;
            }
            State::J if z > 0 && tr.st[z - 1] == State::J => {
                ld += 1;
                exp_j += 1.0;
            }
            _ => {}
        }
    }

    if ld == 0 {
        return vec![1.0; k];
    }

    let norm = 1.0 / ld as f32;
    for node in 1..=m {
        exp_m[node] *= norm;
        exp_i[node] *= norm;
    }
    exp_n *= norm;
    exp_c *= norm;
    exp_j *= norm;
    let xfactor = exp_n + exp_c + exp_j;

    let mut null2 = vec![0.0_f32; k];
    for x in 0..k {
        for node in 1..m {
            null2[x] += exp_m[node] * gm.msc(node, x).exp();
            null2[x] += exp_i[node] * gm.isc(node, x).exp();
        }
        null2[x] += exp_m[m] * gm.msc(m, x).exp();
        null2[x] += xfactor;
    }

    null2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::dp::generic_backward::g_backward;
    use crate::dp::generic_decoding::g_decoding;
    use crate::dp::generic_fwdback::g_forward;
    use crate::dp::gmx::Gmx;
    use crate::profile::{profile_config, reconfig_unihit, P7_LOCAL};
    use std::path::Path;

    #[test]
    fn test_null2_produces_valid_odds_ratios() {
        crate::logsum::p7_flogsuminit();
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
        let mut gm = crate::profile::Profile::new(hmm.m, &abc);
        profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);
        reconfig_unihit(&mut gm, 20);

        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2;

        let mut gx_fwd = Gmx::new(gm.m, l);
        g_forward(&dsq, l, &gm, &mut gx_fwd);
        let mut gx_bck = Gmx::new(gm.m, l);
        g_backward(&dsq, l, &gm, &mut gx_bck);
        let mut pp = Gmx::new(gm.m, l);
        g_decoding(&gm, &gx_fwd, &gx_bck, &mut pp);

        let null2 = null2_by_expectation(&gm, &pp, 1, l);

        // Null2 odds ratios should be positive and near 1.0
        for x in 0..abc.k {
            assert!(
                null2[x] > 0.0,
                "null2[{}] should be positive, got {}",
                x,
                null2[x]
            );
            assert!(
                null2[x] < 10.0,
                "null2[{}] should be < 10, got {}",
                x,
                null2[x]
            );
        }

        // Correction should be finite
        let correction = null2_score(&null2, &dsq, 1, l);
        assert!(correction.is_finite(), "null2 correction should be finite");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_null2_matches_generic() {
        crate::logsum::p7_flogsuminit();
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
        let mut gm = crate::profile::Profile::new(hmm.m, &abc);
        profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);
        reconfig_unihit(&mut gm, 20);

        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2;

        // Generic null2
        let mut gx_fwd = Gmx::new(gm.m, l);
        g_forward(&dsq, l, &gm, &mut gx_fwd);
        let mut gx_bck = Gmx::new(gm.m, l);
        g_backward(&dsq, l, &gm, &mut gx_bck);
        let mut pp = Gmx::new(gm.m, l);
        g_decoding(&gm, &gx_fwd, &gx_bck, &mut pp);
        let gen_null2 = null2_by_expectation(&gm, &pp, 1, l);
        let gen_correction = null2_score(&gen_null2, &dsq, 1, l);

        // SIMD null2
        use crate::simd::oprofile::{OProfile, P7O_C, P7O_J, P7O_LOOP, P7O_N};
        use crate::simd::probmx::{p_null2_from_pmx, ProbMx};
        let om = OProfile::convert(&gm);
        let mut fwd_pmx = ProbMx::new_full(gm.m, l);
        let fwd_sc =
            unsafe { crate::simd::fwd_filter::forward_parser_pmx(&dsq, l, &om, &mut fwd_pmx) };
        let mut bck_pmx = ProbMx::new_full(gm.m, l);
        unsafe {
            crate::simd::bck_filter::backward_parser_pmx(&dsq, l, &om, fwd_sc, &mut bck_pmx);
        };
        let njc_loop = [
            om.xf[P7O_N][P7O_LOOP],
            om.xf[P7O_J][P7O_LOOP],
            om.xf[P7O_C][P7O_LOOP],
        ];
        let simd_correction = p_null2_from_pmx(
            &fwd_pmx, &bck_pmx, gm.m, gm.abc_k, &dsq, 1, l, &gm.rsc, njc_loop,
        );

        // Should be close (not exact due to log-space vs probability-space)
        let diff = (gen_correction - simd_correction).abs();
        eprintln!(
            "Generic null2: {:.6}, SIMD null2: {:.6}, diff: {:.6}",
            gen_correction, simd_correction, diff
        );
        assert!(
            diff < 1.0,
            "SIMD null2 ({:.4}) too far from generic ({:.4}), diff={:.4}",
            simd_correction,
            gen_correction,
            diff
        );
    }
}
