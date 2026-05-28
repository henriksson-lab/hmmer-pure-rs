//! Generic (non-SIMD) Forward algorithm.
//! Direct port of generic_fwdback.c.

#![allow(clippy::needless_range_loop)]

use crate::alphabet::Dsq;
use crate::dp::gmx::*;
use crate::logsum::p7_flogsum;
use crate::profile::*;

/// The Forward algorithm: total log-likelihood summed over all alignments.
///
/// Standard Forward DP. Given digital sequence `dsq` (1-based, `dsq[1..=L]`),
/// profile `gm`, and DP matrix `gx`, fills `gx` and returns the Forward lod
/// score in nats. Caller subtracts a null model lod score to obtain a bit
/// score. The log-sum table is (re)initialized internally via
/// `p7_flogsuminit`. Counterpart of `p7_GForward`.
pub fn g_forward(dsq: &[Dsq], l: usize, gm: &Profile, gx: &mut Gmx) -> f32 {
    let m = gm.m;
    let esc: f32 = if gm.is_local() {
        0.0
    } else {
        f32::NEG_INFINITY
    };

    crate::logsum::p7_flogsuminit();

    // Initialization of row 0
    gx.set_xmx(0, P7G_N, 0.0);
    gx.set_xmx(0, P7G_B, gm.xsc[P7P_N][P7P_MOVE]);
    gx.set_xmx(0, P7G_E, f32::NEG_INFINITY);
    gx.set_xmx(0, P7G_C, f32::NEG_INFINITY);
    gx.set_xmx(0, P7G_J, f32::NEG_INFINITY);
    for k in 0..=m {
        gx.set_mmx(0, k, f32::NEG_INFINITY);
        gx.set_imx(0, k, f32::NEG_INFINITY);
        gx.set_dmx(0, k, f32::NEG_INFINITY);
    }

    // DP recursion
    for i in 1..=l {
        let xi = dsq[i] as usize;

        gx.set_mmx(i, 0, f32::NEG_INFINITY);
        gx.set_imx(i, 0, f32::NEG_INFINITY);
        gx.set_dmx(i, 0, f32::NEG_INFINITY);
        gx.set_xmx(i, P7G_E, f32::NEG_INFINITY);

        for k in 1..m {
            // Match state
            let sc = p7_flogsum(
                p7_flogsum(
                    gx.mmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_MM),
                    gx.imx(i - 1, k - 1) + gm.tsc(k - 1, P7P_IM),
                ),
                p7_flogsum(
                    gx.xmx(i - 1, P7G_B) + gm.tsc(k - 1, P7P_BM),
                    gx.dmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_DM),
                ),
            );
            gx.set_mmx(i, k, sc + gm.msc(k, xi));

            // Insert state
            let sc = p7_flogsum(
                gx.mmx(i - 1, k) + gm.tsc(k, P7P_MI),
                gx.imx(i - 1, k) + gm.tsc(k, P7P_II),
            );
            gx.set_imx(i, k, sc + gm.isc(k, xi));

            // Delete state
            let dsc = p7_flogsum(
                gx.mmx(i, k - 1) + gm.tsc(k - 1, P7P_MD),
                gx.dmx(i, k - 1) + gm.tsc(k - 1, P7P_DD),
            );
            gx.set_dmx(i, k, dsc);

            // E state update (Forward includes D->E)
            let e = p7_flogsum(
                p7_flogsum(gx.mmx(i, k) + esc, gx.dmx(i, k) + esc),
                gx.xmx(i, P7G_E),
            );
            gx.set_xmx(i, P7G_E, e);
        }

        // Unrolled match state M_M
        let sc = p7_flogsum(
            p7_flogsum(
                gx.mmx(i - 1, m - 1) + gm.tsc(m - 1, P7P_MM),
                gx.imx(i - 1, m - 1) + gm.tsc(m - 1, P7P_IM),
            ),
            p7_flogsum(
                gx.xmx(i - 1, P7G_B) + gm.tsc(m - 1, P7P_BM),
                gx.dmx(i - 1, m - 1) + gm.tsc(m - 1, P7P_DM),
            ),
        );
        gx.set_mmx(i, m, sc + gm.msc(m, xi));
        gx.set_imx(i, m, f32::NEG_INFINITY);

        // Unrolled delete state D_M
        let dsc = p7_flogsum(
            gx.mmx(i, m - 1) + gm.tsc(m - 1, P7P_MD),
            gx.dmx(i, m - 1) + gm.tsc(m - 1, P7P_DD),
        );
        gx.set_dmx(i, m, dsc);

        // E state update from M_M and D_M
        let e = p7_flogsum(p7_flogsum(gx.mmx(i, m), gx.dmx(i, m)), gx.xmx(i, P7G_E));
        gx.set_xmx(i, P7G_E, e);

        // J state
        let sc = p7_flogsum(
            gx.xmx(i - 1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP],
            gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_LOOP],
        );
        gx.set_xmx(i, P7G_J, sc);

        // C state
        let sc = p7_flogsum(
            gx.xmx(i - 1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP],
            gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_C, sc);

        // N state
        gx.set_xmx(i, P7G_N, gx.xmx(i - 1, P7G_N) + gm.xsc[P7P_N][P7P_LOOP]);

        // B state
        let sc = p7_flogsum(
            gx.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_MOVE],
            gx.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_B, sc);
    }

    gx.m = m;
    gx.l = l;

    // T state: C->T transition
    gx.xmx(l, P7G_C) + gm.xsc[P7P_C][P7P_MOVE]
}

/// The Hybrid algorithm: run Forward, then return the maximum Match-state
/// value in the Forward matrix. Counterpart of `p7_GHybrid`.
pub fn g_hybrid(dsq: &[Dsq], l: usize, gm: &Profile, gx: &mut Gmx) -> f32 {
    let _fwd_sc = g_forward(dsq, l, gm, gx);
    let mut hybrid = f32::NEG_INFINITY;
    for i in 1..=l {
        for k in 1..=gm.m {
            hybrid = hybrid.max(gx.mmx(i, k));
        }
    }
    hybrid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use std::path::Path;

    #[test]
    fn test_forward_basic() {
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

        let mut gx = Gmx::new(gm.m, l);
        let fwd_sc = g_forward(&dsq, l, &gm, &mut gx);

        // Forward score >= Viterbi score
        let mut gx2 = Gmx::new(gm.m, l);
        let vit_sc = crate::dp::generic_viterbi::g_viterbi(&dsq, l, &gm, &mut gx2);
        assert!(
            fwd_sc >= vit_sc - 0.01,
            "Forward {} should be >= Viterbi {}",
            fwd_sc,
            vit_sc
        );
    }

    #[test]
    fn test_hybrid_matches_max_forward_match_cell() {
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
        let mut gx = Gmx::new(gm.m, l);
        let hybrid = g_hybrid(&dsq, l, &gm, &mut gx);
        let mut expected = f32::NEG_INFINITY;
        for i in 1..=l {
            for k in 1..=gm.m {
                expected = expected.max(gx.mmx(i, k));
            }
        }
        assert_eq!(hybrid, expected);
        assert!(hybrid.is_finite());
    }
}
