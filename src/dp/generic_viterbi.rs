//! Generic (non-SIMD) Viterbi algorithm.
//! Direct port of generic_viterbi.c.

use crate::alphabet::Dsq;
use crate::dp::gmx::*;
use crate::profile::*;

/// The Viterbi algorithm: maximum-scoring single-path alignment.
///
/// Standard Viterbi DP. Given digital sequence `dsq` (1-based, `dsq[1..=L]`),
/// profile `gm`, and DP matrix `gx`, returns the Viterbi lod score in nats and
/// leaves the filled Viterbi matrix in `gx` (so the caller may then recover the
/// path via `g_trace`). Caller subtracts a null model lod score and converts to
/// bits. Counterpart of `p7_GViterbi`.
pub fn g_viterbi(dsq: &[Dsq], l: usize, gm: &Profile, gx: &mut Gmx) -> f32 {
    let m = gm.m;
    let esc: f32 = if gm.is_local() {
        0.0
    } else {
        f32::NEG_INFINITY
    };

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
            let mut sc = gx.mmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_MM);
            sc = sc.max(gx.imx(i - 1, k - 1) + gm.tsc(k - 1, P7P_IM));
            sc = sc.max(gx.dmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_DM));
            sc = sc.max(gx.xmx(i - 1, P7G_B) + gm.tsc(k - 1, P7P_BM));
            gx.set_mmx(i, k, sc + gm.msc(k, xi));

            // E state update
            let e = gx.xmx(i, P7G_E).max(gx.mmx(i, k) + esc);
            gx.set_xmx(i, P7G_E, e);

            // Insert state
            sc = gx.mmx(i - 1, k) + gm.tsc(k, P7P_MI);
            sc = sc.max(gx.imx(i - 1, k) + gm.tsc(k, P7P_II));
            gx.set_imx(i, k, sc + gm.isc(k, xi));

            // Delete state
            let dsc = gx.mmx(i, k - 1) + gm.tsc(k - 1, P7P_MD);
            let dsc = dsc.max(gx.dmx(i, k - 1) + gm.tsc(k - 1, P7P_DD));
            gx.set_dmx(i, k, dsc);
        }

        // Unrolled match state M_M
        let mut sc = gx.mmx(i - 1, m - 1) + gm.tsc(m - 1, P7P_MM);
        sc = sc.max(gx.imx(i - 1, m - 1) + gm.tsc(m - 1, P7P_IM));
        sc = sc.max(gx.dmx(i - 1, m - 1) + gm.tsc(m - 1, P7P_DM));
        sc = sc.max(gx.xmx(i - 1, P7G_B) + gm.tsc(m - 1, P7P_BM));
        gx.set_mmx(i, m, sc + gm.msc(m, xi));

        // Unrolled delete state D_M
        let dsc = gx.mmx(i, m - 1) + gm.tsc(m - 1, P7P_MD);
        let dsc = dsc.max(gx.dmx(i, m - 1) + gm.tsc(m - 1, P7P_DD));
        gx.set_dmx(i, m, dsc);

        // E state update from M_M and D_M
        let e = gx.xmx(i, P7G_E).max(gx.mmx(i, m));
        let e = e.max(gx.dmx(i, m));
        gx.set_xmx(i, P7G_E, e);

        // Special states
        // J state
        let sc = gx.xmx(i - 1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP];
        let sc = sc.max(gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_LOOP]);
        gx.set_xmx(i, P7G_J, sc);

        // C state
        let sc = gx.xmx(i - 1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP];
        let sc = sc.max(gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_MOVE]);
        gx.set_xmx(i, P7G_C, sc);

        // N state
        gx.set_xmx(i, P7G_N, gx.xmx(i - 1, P7G_N) + gm.xsc[P7P_N][P7P_LOOP]);

        // B state
        let sc = gx.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_MOVE];
        let sc = sc.max(gx.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_MOVE]);
        gx.set_xmx(i, P7G_B, sc);
    }

    gx.m = m;
    gx.l = l;

    // T state (not stored): C->T transition
    gx.xmx(l, P7G_C) + gm.xsc[P7P_C][P7P_MOVE]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use std::path::Path;

    fn setup() -> (crate::hmm::Hmm, Alphabet, Bg, Profile) {
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
        (hmm, abc, bg, gm)
    }

    #[test]
    fn test_viterbi_basic() {
        let (_, abc, _, gm) = setup();

        // Digitize a test sequence
        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2; // subtract sentinels

        let mut gx = Gmx::new(gm.m, l);
        let sc = g_viterbi(&dsq, l, &gm, &mut gx);

        // Score should be positive for a matching sequence
        assert!(
            sc > 0.0,
            "Viterbi score {} should be positive for matching seq",
            sc
        );
    }
}
