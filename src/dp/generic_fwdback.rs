//! Generic (non-SIMD) Forward algorithm.
//! Direct port of generic_fwdback.c.

use crate::alphabet::Dsq;
use crate::dp::gmx::*;
use crate::logsum::p7_flogsum;
use crate::profile::*;

/// Generic Forward algorithm.
/// Returns the Forward log-likelihood score in nats.
/// `dsq` is a 1-based digital sequence (dsq[1..=L]).
pub fn g_forward(dsq: &[Dsq], l: usize, gm: &Profile, gx: &mut Gmx) -> f32 {
    let m = gm.m;
    let esc: f32 = if gm.is_local() { 0.0 } else { f32::NEG_INFINITY };

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
        let e = p7_flogsum(
            p7_flogsum(gx.mmx(i, m), gx.dmx(i, m)),
            gx.xmx(i, P7G_E),
        );
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
    #[cfg(feature = "ffi")]
    fn test_forward_matches_ffi() {
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
        let rust_sc = g_forward(&dsq, l, &gm, &mut gx);

        unsafe {
            let c_abc = crate::ffi::esl_alphabet_Create(3);
            let c_bg = crate::ffi::p7_bg_Create(c_abc);
            let path = std::ffi::CString::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/hmmer/testsuite/20aa.hmm"
            ))
            .unwrap();
            let mut hfp: *mut crate::ffi::P7_HMMFILE = std::ptr::null_mut();
            let mut errbuf = [0i8; 256];
            crate::ffi::p7_hmmfile_Open(
                path.as_ptr(),
                std::ptr::null_mut(),
                &mut hfp,
                errbuf.as_mut_ptr(),
            );
            let mut c_abc2: *mut crate::ffi::ESL_ALPHABET = std::ptr::null_mut();
            let mut c_hmm: *mut crate::ffi::P7_HMM = std::ptr::null_mut();
            crate::ffi::p7_hmmfile_Read(hfp, &mut c_abc2, &mut c_hmm);
            let c_gm = crate::ffi::p7_profile_Create((*c_hmm).M, c_abc2);
            crate::ffi::p7_ProfileConfig(c_hmm, c_bg, c_gm, 20, P7_LOCAL);

            let seq = std::ffi::CString::new("ACDEFGHIKLMNPQRSTVWY").unwrap();
            let mut c_dsq: *mut u8 = std::ptr::null_mut();
            crate::ffi::esl_abc_CreateDsq(c_abc2, seq.as_ptr(), &mut c_dsq);

            crate::ffi::p7_FLogsumInit();
            let c_gx = crate::ffi::p7_gmx_Create((*c_hmm).M, l as i32);
            let mut c_sc: f32 = 0.0;
            crate::ffi::p7_GForward(c_dsq, l as i32, c_gm, c_gx, &mut c_sc);

            assert!(
                (rust_sc - c_sc).abs() < 0.1,
                "Forward score mismatch: rust={}, c={}",
                rust_sc,
                c_sc
            );

            crate::ffi::p7_gmx_Destroy(c_gx);
            crate::ffi::p7_profile_Destroy(c_gm);
            crate::ffi::p7_hmm_Destroy(c_hmm);
            crate::ffi::esl_alphabet_Destroy(c_abc2);
            crate::ffi::p7_hmmfile_Close(hfp);
            crate::ffi::p7_bg_Destroy(c_bg);
            crate::ffi::esl_alphabet_Destroy(c_abc);
            libc::free(c_dsq as *mut libc::c_void);
        }
    }
}
