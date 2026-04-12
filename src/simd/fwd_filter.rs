//! SSE-optimized Forward parser (float precision, probability space).
//! Direct port of impl_sse/fwdback.c forward_engine() in parser mode.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;

/// SSE Forward parser. Returns Forward score in nats.
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser(dsq: &[Dsq], l: usize, om: &OProfile) -> f32 {
    let q_count = nqf(om.m);
    let nscells = 3; // M, D, I

    // One-row DP matrix (parser mode)
    let mut dp: Vec<__m128> = vec![_mm_setzero_ps(); q_count * nscells];

    let zerov = _mm_setzero_ps();

    // Initialize special states
    let mut xe: f32 = 0.0;
    let mut xn: f32 = 1.0;
    let mut xj: f32 = 0.0;
    let mut xb: f32 = om.xf[P7O_N][P7O_MOVE];
    let mut xc: f32 = 0.0;
    let mut totscale: f32 = 0.0;

    macro_rules! mmo { ($q:expr) => { dp[$q * nscells + 0] }; }
    macro_rules! dmo { ($q:expr) => { dp[$q * nscells + 1] }; }
    macro_rules! imo { ($q:expr) => { dp[$q * nscells + 2] }; }

    for i in 1..=l {
        let xi = dsq[i] as usize;
        if xi >= om.abc_kp {
            continue;
        }

        // Save previous row (parser: same memory, but we swap via registers)
        // In parser mode, dpp == dpc (same row). We use mpv/dpv/ipv registers.
        let rsc = &om.rfv[xi];
        let mut dcv = zerov;
        let mut xev = zerov;
        let xbv = _mm_set1_ps(xb);

        // Right-shift by 1 float (4 bytes), zero-fill
        let mut mpv = rightshift_float(mmo!(q_count - 1));
        let mut dpv = rightshift_float(dmo!(q_count - 1));
        let mut ipv = rightshift_float(imo!(q_count - 1));

        let mut tsc_idx = 0;

        for q in 0..q_count {
            // Match: B*tBM + M*tMM + I*tIM + D*tDM, then * emission
            let tbm = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tmm = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tim = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tdm = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;

            let mut sv = _mm_mul_ps(xbv, tbm);
            sv = _mm_add_ps(sv, _mm_mul_ps(mpv, tmm));
            sv = _mm_add_ps(sv, _mm_mul_ps(ipv, tim));
            sv = _mm_add_ps(sv, _mm_mul_ps(dpv, tdm));

            let rsc_v = _mm_loadu_ps(rsc[q].as_ptr());

            sv = _mm_mul_ps(sv, rsc_v);
            xev = _mm_add_ps(xev, sv);

            // Save previous before overwriting
            mpv = mmo!(q);
            dpv = dmo!(q);
            ipv = imo!(q);

            // Store M and D
            mmo!(q) = sv;
            dmo!(q) = dcv;

            // M->D partial for next q
            let tmd = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            dcv = _mm_mul_ps(sv, tmd);

            // I state: M*tMI + I*tII (emission odds ratio = 1.0, so no emission multiply)
            let tmi = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tii = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let isv = _mm_add_ps(
                _mm_mul_ps(mpv, tmi),
                _mm_mul_ps(ipv, tii),
            );
            imo!(q) = isv;
        }

        // DD paths: first mandatory pass
        dcv = rightshift_float(dcv);
        dmo!(0) = zerov;
        let dd_offset = 7 * q_count;
        for q in 0..q_count {
            dmo!(q) = _mm_add_ps(dcv, dmo!(q));
            let tdd = _mm_loadu_ps(om.tfv[dd_offset + q].as_ptr());
            dcv = _mm_mul_ps(dmo!(q), tdd);
        }

        // Up to 3 more DD passes
        if om.m < 100 {
            // Full serialization for small models
            for _ in 1..4 {
                dcv = rightshift_float(dcv);
                for q in 0..q_count {
                    dmo!(q) = _mm_add_ps(dcv, dmo!(q));
                    let tdd = _mm_loadu_ps(om.tfv[dd_offset + q].as_ptr());
                    dcv = _mm_mul_ps(dcv, tdd);
                }
            }
        } else {
            // Parallel with early termination for large models
            for _ in 1..4 {
                dcv = rightshift_float(dcv);
                let mut cv = zerov;
                for q in 0..q_count {
                    let sv = _mm_add_ps(dcv, dmo!(q));
                    cv = _mm_or_ps(cv, _mm_cmpgt_ps(sv, dmo!(q)));
                    dmo!(q) = sv;
                    let tdd = _mm_loadu_ps(om.tfv[dd_offset + q].as_ptr());
                    dcv = _mm_mul_ps(dcv, tdd);
                }
                if _mm_movemask_ps(cv) == 0 {
                    break;
                }
            }
        }

        // Add D's to xEv
        for q in 0..q_count {
            xev = _mm_add_ps(dmo!(q), xev);
        }

        // Horizontal sum of xEv -> xE
        xev = _mm_add_ps(xev, _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xev, xev));
        xev = _mm_add_ps(xev, _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xev, xev));
        _mm_store_ss(&mut xe, xev);

        // Special states (scalar, probability space)
        xn *= om.xf[P7O_N][P7O_LOOP];
        xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
        xb = xj * om.xf[P7O_J][P7O_MOVE] + xn * om.xf[P7O_N][P7O_MOVE];


        // Sparse rescaling when xE gets large
        if xe > 1.0e4 {
            let scale = 1.0 / xe;
            xn *= scale;
            xc *= scale;
            xj *= scale;
            xb *= scale;
            let scale_v = _mm_set1_ps(scale);
            for q in 0..q_count {
                mmo!(q) = _mm_mul_ps(mmo!(q), scale_v);
                dmo!(q) = _mm_mul_ps(dmo!(q), scale_v);
                imo!(q) = _mm_mul_ps(imo!(q), scale_v);
            }
            totscale += xe.ln();
            xe = 1.0;
        }
    }

    // Final score: totscale + log(xC * C->T)
    if xc.is_nan() || (l > 0 && xc == 0.0) || xc.is_infinite() {
        return f32::NEG_INFINITY; // error conditions
    }

    totscale + (xc * om.xf[P7O_C][P7O_MOVE]).ln()
}

/// Right-shift a __m128 by one float element, zero-filling from the left.
/// [a, b, c, d] -> [0, a, b, c]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn rightshift_float(v: __m128) -> __m128 {
    // Cast to integer, shift left by 4 bytes, cast back
    let vi = _mm_castps_si128(v);
    let shifted = _mm_slli_si128::<4>(vi);
    _mm_castsi128_ps(shifted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::profile::*;
    use std::path::Path;

    #[test]
    fn test_forward_parser_basic() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }
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
        profile_config(&hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        let dsq = abc.digitize(b"AAAAAAAAAAGGGGGGGGGG");
        let sc = unsafe { forward_parser(&dsq, 20, &om) };
        assert!(sc.is_finite(), "Forward score should be finite, got {}", sc);
    }

    #[test]
    fn test_forward_parser_matches_ffi() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }
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
        profile_config(&hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        let dsq = abc.digitize(b"AAAAAAAAAAGGGGGGGGGG");
        let l = dsq.len() - 2;
        let rust_sc = unsafe { forward_parser(&dsq, l, &om) };

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
            crate::ffi::p7_hmmfile_Open(path.as_ptr(), std::ptr::null_mut(), &mut hfp, errbuf.as_mut_ptr());
            let mut c_abc2: *mut crate::ffi::ESL_ALPHABET = std::ptr::null_mut();
            let mut c_hmm: *mut crate::ffi::P7_HMM = std::ptr::null_mut();
            crate::ffi::p7_hmmfile_Read(hfp, &mut c_abc2, &mut c_hmm);
            let c_gm = crate::ffi::p7_profile_Create((*c_hmm).M, c_abc2);
            crate::ffi::p7_ProfileConfig(c_hmm, c_bg, c_gm, 400, P7_LOCAL);
            let c_om = crate::ffi::p7_oprofile_Create((*c_hmm).M, c_abc2);
            crate::ffi::p7_oprofile_Convert(c_gm, c_om);

            let seq = std::ffi::CString::new("AAAAAAAAAAGGGGGGGGGG").unwrap();
            let mut c_dsq: *mut u8 = std::ptr::null_mut();
            crate::ffi::esl_abc_CreateDsq(c_abc2, seq.as_ptr(), &mut c_dsq);
            let c_ox = crate::ffi::p7_omx_Create((*c_hmm).M, 0, l as i32);
            let mut c_sc: f32 = 0.0;
            crate::ffi::p7_ForwardParser(c_dsq, l as i32, c_om, c_ox, &mut c_sc);

            assert!(
                (rust_sc - c_sc).abs() < 0.5,
                "Forward scores differ: rust={}, c={}",
                rust_sc,
                c_sc
            );

            crate::ffi::p7_omx_Destroy(c_ox);
            crate::ffi::p7_oprofile_Destroy(c_om);
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
