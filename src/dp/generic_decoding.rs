//! Generic posterior decoding from Forward+Backward matrices.
//! Direct port of generic_decoding.c p7_GDecoding().

use crate::dp::gmx::*;
use crate::profile::*;

/// Compute posterior probabilities from Forward and Backward matrices.
/// Writes posterior probabilities into `pp`.
/// Returns the overall Forward score used for normalization.
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
            let mm_pp = (fwd.mmx(i, k) + bck.mmx(i, k) - overall_sc).exp();
            pp.set_mmx(i, k, mm_pp);
            denom += mm_pp;

            let im_pp = (fwd.imx(i, k) + bck.imx(i, k) - overall_sc).exp();
            pp.set_imx(i, k, im_pp);
            denom += im_pp;

            pp.set_dmx(i, k, 0.0);
        }

        let mm_pp = (fwd.mmx(i, m) + bck.mmx(i, m) - overall_sc).exp();
        pp.set_mmx(i, m, mm_pp);
        denom += mm_pp;
        pp.set_imx(i, m, 0.0);
        pp.set_dmx(i, m, 0.0);

        // Special states
        pp.set_xmx(i, P7G_E, 0.0);

        let n_pp = (fwd.xmx(i - 1, P7G_N) + bck.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_LOOP]
            - overall_sc)
            .exp();
        pp.set_xmx(i, P7G_N, n_pp);

        let j_pp = (fwd.xmx(i - 1, P7G_J) + bck.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_LOOP]
            - overall_sc)
            .exp();
        pp.set_xmx(i, P7G_J, j_pp);

        pp.set_xmx(i, P7G_B, 0.0);

        let c_pp = (fwd.xmx(i - 1, P7G_C) + bck.xmx(i, P7G_C) + gm.xsc[P7P_C][P7P_LOOP]
            - overall_sc)
            .exp();
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

/// Compute domain occupancy from posterior probability matrix.
/// Returns a vector `mocc[0..=L]` where `mocc[i]` = P(residue i is in a domain).
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
