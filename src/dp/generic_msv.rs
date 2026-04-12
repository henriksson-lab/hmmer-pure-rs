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

    #[test]
    fn test_msv_matches_ffi() {
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
        let rust_sc = g_msv(&dsq, l, &gm, &mut gx, 2.0);

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

            let c_gx = crate::ffi::p7_gmx_Create((*c_hmm).M, l as i32);
            let mut c_sc: f32 = 0.0;
            crate::ffi::p7_GMSV(c_dsq, l as i32, c_gm, c_gx, 2.0, &mut c_sc);

            assert!(
                (rust_sc - c_sc).abs() < 0.1,
                "MSV score mismatch: rust={}, c={}",
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
