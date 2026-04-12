//! Generic (non-SIMD) MSV (Multi-Segment Viterbi) algorithm.
//! Direct port of generic_msv.c.

use crate::alphabet::Dsq;
use crate::dp::gmx::*;
use crate::profile::Profile;

/// Generic MSV algorithm.
/// Returns the MSV score in nats.
/// `dsq` is a 1-based digital sequence (dsq[1..=L]).
/// `nu` is the expected number of hits (typically 2.0).
pub fn g_msv(dsq: &[Dsq], l: usize, gm: &Profile, gx: &mut Gmx, nu: f32) -> f32 {
    let m = gm.m;
    let tloop = (l as f32 / (l as f32 + 3.0)).ln();
    let tmove = (3.0_f32 / (l as f32 + 3.0)).ln();
    let tbmk = (2.0_f32 / (m as f32 * (m as f32 + 1.0))).ln();
    let tej = ((nu - 1.0) / nu).ln();
    let tec = (1.0_f32 / nu).ln();

    // Initialization of row 0
    gx.set_xmx(0, P7G_N, 0.0);
    gx.set_xmx(0, P7G_B, tmove);
    gx.set_xmx(0, P7G_E, f32::NEG_INFINITY);
    gx.set_xmx(0, P7G_C, f32::NEG_INFINITY);
    gx.set_xmx(0, P7G_J, f32::NEG_INFINITY);
    for k in 0..=m {
        gx.set_mmx(0, k, f32::NEG_INFINITY);
    }

    // DP recursion
    for i in 1..=l {
        let xi = dsq[i] as usize;

        gx.set_mmx(i, 0, f32::NEG_INFINITY);
        gx.set_xmx(i, P7G_E, f32::NEG_INFINITY);

        for k in 1..=m {
            // Match state: only from previous M or B
            let sc = gx.mmx(i - 1, k - 1).max(gx.xmx(i - 1, P7G_B) + tbmk);
            gx.set_mmx(i, k, gm.msc(k, xi) + sc);

            // E state update
            let e = gx.xmx(i, P7G_E).max(gx.mmx(i, k));
            gx.set_xmx(i, P7G_E, e);
        }

        // Special states
        let j = (gx.xmx(i - 1, P7G_J) + tloop).max(gx.xmx(i, P7G_E) + tej);
        gx.set_xmx(i, P7G_J, j);

        let c = (gx.xmx(i - 1, P7G_C) + tloop).max(gx.xmx(i, P7G_E) + tec);
        gx.set_xmx(i, P7G_C, c);

        gx.set_xmx(i, P7G_N, gx.xmx(i - 1, P7G_N) + tloop);

        let b = (gx.xmx(i, P7G_N) + tmove).max(gx.xmx(i, P7G_J) + tmove);
        gx.set_xmx(i, P7G_B, b);
    }

    gx.m = m;
    gx.l = l;

    gx.xmx(l, P7G_C) + tmove
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::profile::*;
    use std::path::Path;

    #[test]
    fn test_msv_basic() {
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
        profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);

        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2;

        let mut gx = Gmx::new(gm.m, l);
        let sc = g_msv(&dsq, l, &gm, &mut gx, 2.0);

        assert!(sc > 0.0, "MSV score {} should be positive", sc);
    }
