//! Generic (non-SIMD) Backward algorithm.
//! Direct port of generic_fwdback.c p7_GBackward().

use crate::alphabet::Dsq;
use crate::dp::gmx::*;
use crate::logsum::p7_flogsum;
use crate::profile::*;

/// Generic Backward algorithm.
/// Returns the Backward log-likelihood score in nats.
/// `dsq` is a 1-based digital sequence (dsq[1..=L]).
/// NOTE: Backward calculates the probability we can get *out* of cell (i,k),
/// exclusive of emitting residue x_i.
pub fn g_backward(dsq: &[Dsq], l: usize, gm: &Profile, gx: &mut Gmx) -> f32 {
    let m = gm.m;
    let esc: f32 = if gm.is_local() { 0.0 } else { f32::NEG_INFINITY };

    crate::logsum::p7_flogsuminit();

    // Initialize row L
    gx.set_xmx(l, P7G_J, f32::NEG_INFINITY);
    gx.set_xmx(l, P7G_B, f32::NEG_INFINITY);
    gx.set_xmx(l, P7G_N, f32::NEG_INFINITY);
    gx.set_xmx(l, P7G_C, gm.xsc[P7P_C][P7P_MOVE]); // C<-T
    gx.set_xmx(
        l,
        P7G_E,
        gx.xmx(l, P7G_C) + gm.xsc[P7P_E][P7P_MOVE],
    ); // E<-C

    gx.set_mmx(l, m, gx.xmx(l, P7G_E));
    gx.set_dmx(l, m, gx.xmx(l, P7G_E));
    gx.set_imx(l, m, f32::NEG_INFINITY);

    for k in (1..m).rev() {
        let mm = p7_flogsum(gx.xmx(l, P7G_E) + esc, gx.dmx(l, k + 1) + gm.tsc(k, P7P_MD));
        gx.set_mmx(l, k, mm);
        let dm = p7_flogsum(gx.xmx(l, P7G_E) + esc, gx.dmx(l, k + 1) + gm.tsc(k, P7P_DD));
        gx.set_dmx(l, k, dm);
        gx.set_imx(l, k, f32::NEG_INFINITY);
    }

    // Main recursion
    for i in (1..l).rev() {
        // rsc points to emissions for residue x_{i+1}
        let xi1 = dsq[i + 1] as usize;

        // B state
        let mut xb = gx.mmx(i + 1, 1) + gm.tsc(0, P7P_BM) + gm.msc(1, xi1);
        for k in 2..=m {
            xb = p7_flogsum(xb, gx.mmx(i + 1, k) + gm.tsc(k - 1, P7P_BM) + gm.msc(k, xi1));
        }
        gx.set_xmx(i, P7G_B, xb);

        // J state
        let xj = p7_flogsum(
            gx.xmx(i + 1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP],
            gx.xmx(i, P7G_B) + gm.xsc[P7P_J][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_J, xj);

        // C state
        gx.set_xmx(
            i,
            P7G_C,
            gx.xmx(i + 1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP],
        );

        // E state
        let xe = p7_flogsum(
            gx.xmx(i, P7G_J) + gm.xsc[P7P_E][P7P_LOOP],
            gx.xmx(i, P7G_C) + gm.xsc[P7P_E][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_E, xe);

        // N state
        let xn = p7_flogsum(
            gx.xmx(i + 1, P7G_N) + gm.xsc[P7P_N][P7P_LOOP],
            gx.xmx(i, P7G_B) + gm.xsc[P7P_N][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_N, xn);

        gx.set_mmx(i, m, gx.xmx(i, P7G_E));
        gx.set_dmx(i, m, gx.xmx(i, P7G_E));
        gx.set_imx(i, m, f32::NEG_INFINITY);

        for k in (1..m).rev() {
            // M state
            let mm = p7_flogsum(
                p7_flogsum(
                    gx.mmx(i + 1, k + 1) + gm.tsc(k, P7P_MM) + gm.msc(k + 1, xi1),
                    gx.imx(i + 1, k) + gm.tsc(k, P7P_MI) + gm.isc(k, xi1),
                ),
                p7_flogsum(
                    gx.xmx(i, P7G_E) + esc,
                    gx.dmx(i, k + 1) + gm.tsc(k, P7P_MD),
                ),
            );
            gx.set_mmx(i, k, mm);

            // I state
            let im = p7_flogsum(
                gx.mmx(i + 1, k + 1) + gm.tsc(k, P7P_IM) + gm.msc(k + 1, xi1),
                gx.imx(i + 1, k) + gm.tsc(k, P7P_II) + gm.isc(k, xi1),
            );
            gx.set_imx(i, k, im);

            // D state
            let dm = p7_flogsum(
                gx.mmx(i + 1, k + 1) + gm.tsc(k, P7P_DM) + gm.msc(k + 1, xi1),
                p7_flogsum(
                    gx.dmx(i, k + 1) + gm.tsc(k, P7P_DD),
                    gx.xmx(i, P7G_E) + esc,
                ),
            );
            gx.set_dmx(i, k, dm);
        }
    }

    // At i=0, only N,B states are reachable
    let xi1 = dsq[1] as usize;
    let mut xb = gx.mmx(1, 1) + gm.tsc(0, P7P_BM) + gm.msc(1, xi1);
    for k in 2..=m {
        xb = p7_flogsum(xb, gx.mmx(1, k) + gm.tsc(k - 1, P7P_BM) + gm.msc(k, xi1));
    }
    gx.set_xmx(0, P7G_B, xb);
    gx.set_xmx(0, P7G_J, f32::NEG_INFINITY);
    gx.set_xmx(0, P7G_C, f32::NEG_INFINITY);
    gx.set_xmx(0, P7G_E, f32::NEG_INFINITY);
    let xn = p7_flogsum(
        gx.xmx(1, P7G_N) + gm.xsc[P7P_N][P7P_LOOP],
        gx.xmx(0, P7G_B) + gm.xsc[P7P_N][P7P_MOVE],
    );
    gx.set_xmx(0, P7G_N, xn);
    for k in (1..=m).rev() {
        gx.set_mmx(0, k, f32::NEG_INFINITY);
        gx.set_imx(0, k, f32::NEG_INFINITY);
        gx.set_dmx(0, k, f32::NEG_INFINITY);
    }

    gx.m = m;
    gx.l = l;
    gx.xmx(0, P7G_N)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::dp::generic_fwdback::g_forward;
    use std::path::Path;

    #[test]
    fn test_backward_basic() {
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
        let fwd_sc = g_forward(&dsq, l, &gm, &mut gx_fwd);

        let mut gx_bck = Gmx::new(gm.m, l);
        let bck_sc = g_backward(&dsq, l, &gm, &mut gx_bck);

        // Forward and Backward should give the same overall score
        assert!(
            (fwd_sc - bck_sc).abs() < 0.1,
            "Forward {} and Backward {} should agree",
            fwd_sc,
            bck_sc
        );
    }
