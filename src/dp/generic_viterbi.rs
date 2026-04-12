//! Generic (non-SIMD) Viterbi algorithm.
//! Direct port of generic_viterbi.c.

use crate::alphabet::Dsq;
use crate::dp::gmx::*;
use crate::profile::*;

/// Generic Viterbi algorithm.
/// Returns the Viterbi score in nats.
/// `dsq` is a 1-based digital sequence (dsq[1..=L]).
pub fn g_viterbi(dsq: &[Dsq], l: usize, gm: &Profile, gx: &mut Gmx) -> f32 {
    let m = gm.m;
    let esc: f32 = if gm.is_local() { 0.0 } else { f32::NEG_INFINITY };

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
        assert!(sc > 0.0, "Viterbi score {} should be positive for matching seq", sc);
    }

    #[test]
    fn test_viterbi_matches_ffi() {
        let (_, abc, _, gm) = setup();
        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2;

        let mut gx = Gmx::new(gm.m, l);
        let rust_sc = g_viterbi(&dsq, l, &gm, &mut gx);

        // Compare with C implementation
        unsafe {
            let c_abc = crate::ffi::esl_alphabet_Create(3); // amino
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

            // Digitize sequence via C
            let seq = std::ffi::CString::new("ACDEFGHIKLMNPQRSTVWY").unwrap();
            let mut c_dsq: *mut u8 = std::ptr::null_mut();
            crate::ffi::esl_abc_CreateDsq(c_abc2, seq.as_ptr(), &mut c_dsq);

            let c_gx = crate::ffi::p7_gmx_Create((*c_hmm).M, l as i32);
            let mut c_sc: f32 = 0.0;
            crate::ffi::p7_GViterbi(c_dsq, l as i32, c_gm, c_gx, &mut c_sc);

            assert!(
                (rust_sc - c_sc).abs() < 0.1,
                "Viterbi score mismatch: rust={}, c={}",
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
