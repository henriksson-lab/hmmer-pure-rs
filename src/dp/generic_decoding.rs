//! Generic posterior decoding from Forward+Backward matrices.
//! Direct port of generic_decoding.c p7_GDecoding().

#![allow(clippy::doc_lazy_continuation, clippy::needless_range_loop)]

use crate::dp::gmx::*;
use crate::profile::*;
use crate::util::cmath::{c_exp_to_f32, c_expf_to_f32};

/// Posterior decoding of residue assignments from filled Forward/Backward matrices.
///
/// For each residue `i = 1..=L` and state, computes the posterior probability that
/// the state emitted that residue (the sum over `M_{1..M}`, `I_{1..M-1}`, and the
/// N/C/J loop emissions equals 1.0 for every `i`). Writes results into `pp`;
/// row `i=0` is unused (left zero). Each row is renormalized to absorb log-sum
/// table approximation error. Returns the overall Forward score used for
/// normalization. Counterpart of `p7_GDecoding`.
pub fn g_decoding(gm: &Profile, fwd: &Gmx, bck: &Gmx, pp: &mut Gmx) -> f32 {
    let l = fwd.l;
    let m = gm.m;
    let overall_sc = fwd.xmx(l, P7G_C) + gm.xsc[P7P_C][P7P_MOVE];

    pp.m = m;
    pp.l = l;

    // Row 0: all zero
    pp.set_xmx(0, P7G_E, 0.0);
    pp.set_xmx(0, P7G_N, 0.0);
    pp.set_xmx(0, P7G_J, 0.0);
    pp.set_xmx(0, P7G_B, 0.0);
    pp.set_xmx(0, P7G_C, 0.0);
    for k in 0..=m {
        pp.set_mmx(0, k, 0.0);
        pp.set_imx(0, k, 0.0);
        pp.set_dmx(0, k, 0.0);
    }

    for i in 1..=l {
        let mut denom = 0.0_f32;

        pp.set_mmx(i, 0, 0.0);
        pp.set_imx(i, 0, 0.0);
        pp.set_dmx(i, 0, 0.0);

        for k in 1..m {
            let mm_pp = c_expf_to_f32(fwd.mmx(i, k) + bck.mmx(i, k) - overall_sc);
            pp.set_mmx(i, k, mm_pp);
            denom += mm_pp;

            let im_pp = c_expf_to_f32(fwd.imx(i, k) + bck.imx(i, k) - overall_sc);
            pp.set_imx(i, k, im_pp);
            denom += im_pp;

            pp.set_dmx(i, k, 0.0);
        }

        let mm_pp = c_expf_to_f32(fwd.mmx(i, m) + bck.mmx(i, m) - overall_sc);
        pp.set_mmx(i, m, mm_pp);
        denom += mm_pp;
        pp.set_imx(i, m, 0.0);
        pp.set_dmx(i, m, 0.0);

        // Special states
        pp.set_xmx(i, P7G_E, 0.0);

        let n_pp = c_expf_to_f32(
            fwd.xmx(i - 1, P7G_N) + bck.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_LOOP] - overall_sc,
        );
        pp.set_xmx(i, P7G_N, n_pp);

        let j_pp = c_expf_to_f32(
            fwd.xmx(i - 1, P7G_J) + bck.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_LOOP] - overall_sc,
        );
        pp.set_xmx(i, P7G_J, j_pp);

        pp.set_xmx(i, P7G_B, 0.0);

        let c_pp = c_expf_to_f32(
            fwd.xmx(i - 1, P7G_C) + bck.xmx(i, P7G_C) + gm.xsc[P7P_C][P7P_LOOP] - overall_sc,
        );
        pp.set_xmx(i, P7G_C, c_pp);

        denom += n_pp + j_pp + c_pp;

        // Normalize
        let denom_inv = 1.0 / denom;
        for k in 1..m {
            pp.set_mmx(i, k, pp.mmx(i, k) * denom_inv);
            pp.set_imx(i, k, pp.imx(i, k) * denom_inv);
        }
        pp.set_mmx(i, m, pp.mmx(i, m) * denom_inv);
        pp.set_xmx(i, P7G_N, pp.xmx(i, P7G_N) * denom_inv);
        pp.set_xmx(i, P7G_J, pp.xmx(i, P7G_J) * denom_inv);
        pp.set_xmx(i, P7G_C, pp.xmx(i, P7G_C) * denom_inv);
    }

    overall_sc
}

/// Compute per-residue domain occupancy from a posterior probability matrix.
///
/// Returns `mocc[0..=L]` where `mocc[i]` = P(residue `i` is emitted by the
/// core model) = sum of `MMX(i,k) + IMX(i,k)` over all `k`. Helper used by
/// downstream domain decoding/parsing.
pub fn domain_occupancy(pp: &Gmx) -> Vec<f32> {
    let l = pp.l;
    let m = pp.m;
    let mut mocc = vec![0.0_f32; l + 1];

    for i in 1..=l {
        let mut sum = 0.0_f32;
        for k in 1..=m {
            sum += pp.mmx(i, k) + pp.imx(i, k);
        }
        mocc[i] = sum;
    }

    mocc
}

/// Posterior decoding of domain location.
///
/// Computes the cumulative expected B- and E-state usage and the per-residue
/// core-model occupancy from filled Forward/Backward matrices. Returns
/// `(btot, etot, mocc)`, each of length `L+1` (0-indexed):
/// - `btot[i]` = expected number of domains started at or before position `i`
/// - `etot[i]` = expected number of domains ended at or before position `i`
/// - `mocc[i]` = P(residue `i` is in a domain) = `1 - P(N|J|C loop)`
/// Adapted for generic (log-space) DP matrices. Counterpart of `p7_GDomainDecoding`.
pub fn domain_decoding(gm: &Profile, fwd: &Gmx, bck: &Gmx) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let l = fwd.l;
    let overall_sc = fwd.xmx(l, P7G_C) + gm.xsc[P7P_C][P7P_MOVE];

    let mut btot = vec![0.0_f32; l + 1];
    let mut etot = vec![0.0_f32; l + 1];
    let mut mocc = vec![0.0_f32; l + 1];

    for i in 1..=l {
        // B-state posterior at position i-1 (B at i-1 leads to M_1 at i)
        // In generic log-space: exp(fwd_B[i-1] + bck_B[i-1] - overall_sc)
        let b_post =
            c_exp_to_f32((fwd.xmx(i - 1, P7G_B) + bck.xmx(i - 1, P7G_B) - overall_sc) as f64);
        btot[i] = btot[i - 1] + b_post;

        // E-state posterior at position i
        let e_post = c_exp_to_f32((fwd.xmx(i, P7G_E) + bck.xmx(i, P7G_E) - overall_sc) as f64);
        etot[i] = etot[i - 1] + e_post;

        // mocc = 1 - P(N loop) - P(J loop) - P(C loop)
        let n_post = c_expf_to_f32(
            fwd.xmx(i - 1, P7G_N) + bck.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_LOOP] - overall_sc,
        );
        let j_post = c_expf_to_f32(
            fwd.xmx(i - 1, P7G_J) + bck.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_LOOP] - overall_sc,
        );
        let c_post = c_expf_to_f32(
            fwd.xmx(i - 1, P7G_C) + bck.xmx(i, P7G_C) + gm.xsc[P7P_C][P7P_LOOP] - overall_sc,
        );
        mocc[i] = 1.0 - (n_post + j_post + c_post);
    }

    (btot, etot, mocc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::dp::generic_backward::g_backward;
    use crate::dp::generic_fwdback::g_forward;
    use std::path::Path;

    #[test]
    fn test_decoding_basic() {
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
        let mut gm = Profile::new(hmm.m, &abc);
        crate::profile::profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);

        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2;

        let mut gx_fwd = Gmx::new(gm.m, l);
        g_forward(&dsq, l, &gm, &mut gx_fwd);

        let mut gx_bck = Gmx::new(gm.m, l);
        g_backward(&dsq, l, &gm, &mut gx_bck);

        let mut pp = Gmx::new(gm.m, l);
        g_decoding(&gm, &gx_fwd, &gx_bck, &mut pp);

        // Posterior probabilities should be between 0 and 1
        let mocc = domain_occupancy(&pp);
        for i in 1..=l {
            assert!(
                mocc[i] >= 0.0 && mocc[i] <= 1.01,
                "mocc[{}] = {} should be in [0,1]",
                i,
                mocc[i]
            );
        }

        // For a perfectly matching sequence, most positions should have high occupancy
        let high_occ = mocc[1..=l].iter().filter(|&&p| p > 0.5).count();
        assert!(
            high_occ >= l / 2,
            "Expected most positions to have high occupancy, got {}/{}",
            high_occ,
            l
        );
    }
}
