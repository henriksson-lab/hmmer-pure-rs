//! Null2 bias correction by expectation from posterior probabilities.
//! Port of generic_null2.c p7_GNull2_ByExpectation().
//!
//! The null2 correction compensates for composition bias by computing the
//! expected emission distribution from the posterior-weighted model, then
//! comparing it to the background model.

use crate::dp::gmx::*;
use crate::profile::Profile;
use crate::trace::{State, Trace};
use crate::util::cmath::{c_expf_to_f32, c_logf_to_f32};

/// Calculate the null2 model from posterior probabilities (expectation method).
///
/// Applied to envelopes in well-resolved, single-envelope regions. Computes the
/// null2 odds emission ratios `f'(x)/f(x)` for residues `0..K-1` from the
/// state-usage frequencies of the posterior matrix `pp` over envelope
/// `ienv..=jenv`. Returns the odds-ratio vector (caller can convert to a score
/// correction via `null2_score`). Counterpart of `p7_GNull2_ByExpectation`.
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

    // Convert expected counts to log frequencies (the log posterior weights),
    // exactly as C does: FLog(count) then FIncrement(-log((float)Ld)). C uses
    // logf for the per-element log and a double log() for the -log(Ld) term, so
    // mirror that: logf(count) + (float)(-log((double)Ld)).
    // (generic_null2.c:91-95)
    let neg_log_ld = -crate::util::cmath::c_log_f32_to_f32(ld as f32);
    for node in 1..=m {
        exp_m[node] = c_logf_to_f32(exp_m[node]) + neg_log_ld;
        exp_i[node] = c_logf_to_f32(exp_i[node]) + neg_log_ld;
    }
    exp_n = c_logf_to_f32(exp_n) + neg_log_ld;
    exp_c = c_logf_to_f32(exp_c) + neg_log_ld;
    exp_j = c_logf_to_f32(exp_j) + neg_log_ld;

    // xfactor = FLogsum(N, C, J) over the log frequencies (generic_null2.c:102-104).
    let xfactor = crate::logsum::p7_flogsum(
        crate::logsum::p7_flogsum(exp_n, exp_c),
        exp_j,
    );

    // C generic_null2.c forms the weighted emission odds in log-space,
    // then exponentiates. Keep the order and state ranges aligned with C.
    let mut null2 = vec![f32::NEG_INFINITY; k];
    for x in 0..k {
        for node in 1..m {
            null2[x] = crate::logsum::p7_flogsum(null2[x], exp_m[node] + gm.msc(node, x));
            null2[x] = crate::logsum::p7_flogsum(null2[x], exp_i[node] + gm.isc(node, x));
        }
        null2[x] = crate::logsum::p7_flogsum(null2[x], exp_m[m] + gm.msc(m, x));
        null2[x] = crate::logsum::p7_flogsum(null2[x], xfactor);
        null2[x] = c_expf_to_f32(null2[x]);
    }

    null2
}

/// Sum the per-residue null2 log-odds over an envelope to obtain a bias correction.
///
/// Returns the total correction in nats: `sum_{i=ienv..=jenv} ln(null2[dsq[i]])`.
/// The caller subtracts this from the raw domain score.
pub fn null2_score(null2: &[f32], dsq: &[u8], ienv: usize, jenv: usize) -> f32 {
    let mut score = 0.0_f32;
    for i in ienv..=jenv {
        let x = dsq[i] as usize;
        if x < null2.len() && null2[x] > 0.0 {
            score += c_logf_to_f32(null2[x]);
        }
    }
    score
}

/// Assign null2 odds ratios to an envelope by the trace-sampling method.
///
/// Given a stochastic traceback `tr`, computes null2 odds ratios `f'(x)/f(x)`
/// as state-usage-weighted emission probabilities, with usages tallied from
/// trace positions `zstart..=zend`. Target sequence is irrelevant; profile
/// configuration is irrelevant (only emission odds are used).
/// Counterpart of `p7_GNull2_ByTrace`.
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
            null2[x] += exp_m[node] * c_expf_to_f32(gm.msc(node, x));
            null2[x] += exp_i[node] * c_expf_to_f32(gm.isc(node, x));
        }
        null2[x] += exp_m[m] * c_expf_to_f32(gm.msc(m, x));
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
