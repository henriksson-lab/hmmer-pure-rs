//! SSE2-optimized Viterbi filter (int16 precision).
//! Direct port of impl_sse/vitfilter.c p7_ViterbiFilter().

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;

/// Result of Viterbi filter.
pub enum VitResult {
    /// Sequence passed, Viterbi score in nats
    Ok(f32),
    /// Score overflowed
    Overflow,
}

/// SSE2-optimized Viterbi filter using int16 precision.
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn viterbi_filter(dsq: &[Dsq], l: usize, om: &OProfile) -> VitResult {
    let q_count = nqw(om.m);
    let nscells = 3; // M, D, I per q

    // DP row: dp[q*3+0]=M, dp[q*3+1]=D, dp[q*3+2]=I
    let mut dp: Vec<__m128i> = vec![_mm_set1_epi16(-32768); q_count * nscells];

    let neg_inf_v = {
        let v = _mm_set1_epi16(-32768);
        _mm_srli_si128::<14>(v) // only lowest 2 bytes = -32768, rest = 0
    };

    let mut xn: i16 = om.base_w;
    let mut xb: i16 = add_i16(xn, om.xw[P7O_N][P7O_MOVE]);
    let mut xj: i16 = -32768;
    let mut xc: i16 = -32768;

    macro_rules! mmx {
        ($q:expr) => {
            dp[$q * nscells + 0]
        };
    }
    macro_rules! dmx {
        ($q:expr) => {
            dp[$q * nscells + 1]
        };
    }
    macro_rules! imx {
        ($q:expr) => {
            dp[$q * nscells + 2]
        };
    }

    for i in 1..=l {
        let xi = dsq[i] as usize;
        if xi >= om.abc_kp {
            continue;
        }
        let rsc = &om.rwv[xi];

        let mut dcv = _mm_set1_epi16(-32768);
        let mut xev = _mm_set1_epi16(-32768);
        let mut dmaxv = _mm_set1_epi16(-32768);
        let xbv = _mm_set1_epi16(xb);

        // Right shift by 1 word (2 bytes), fill with -32768
        let mut mpv = _mm_slli_si128::<2>(mmx!(q_count - 1));
        mpv = _mm_or_si128(mpv, neg_inf_v);
        let mut dpv = _mm_slli_si128::<2>(dmx!(q_count - 1));
        dpv = _mm_or_si128(dpv, neg_inf_v);
        let mut ipv = _mm_slli_si128::<2>(imx!(q_count - 1));
        ipv = _mm_or_si128(ipv, neg_inf_v);

        let mut tsc_idx = 0;

        for q in 0..q_count {
            // Match state: max(B+tBM, M+tMM, I+tIM, D+tDM) + emission
            let tsc_bm = _mm_loadu_si128(om.twv[tsc_idx].as_ptr() as *const __m128i);
            tsc_idx += 1;
            let tsc_mm = _mm_loadu_si128(om.twv[tsc_idx].as_ptr() as *const __m128i);
            tsc_idx += 1;
            let tsc_im = _mm_loadu_si128(om.twv[tsc_idx].as_ptr() as *const __m128i);
            tsc_idx += 1;
            let tsc_dm = _mm_loadu_si128(om.twv[tsc_idx].as_ptr() as *const __m128i);
            tsc_idx += 1;

            let mut sv = _mm_adds_epi16(xbv, tsc_bm);
            sv = _mm_max_epi16(sv, _mm_adds_epi16(mpv, tsc_mm));
            sv = _mm_max_epi16(sv, _mm_adds_epi16(ipv, tsc_im));
            sv = _mm_max_epi16(sv, _mm_adds_epi16(dpv, tsc_dm));

            let rsc_v = _mm_loadu_si128(rsc[q].as_ptr() as *const __m128i);
            sv = _mm_adds_epi16(sv, rsc_v);
            xev = _mm_max_epi16(xev, sv);

            // Save previous values before overwriting
            mpv = mmx!(q);
            dpv = dmx!(q);
            ipv = imx!(q);

            // Store M and D
            mmx!(q) = sv;
            dmx!(q) = dcv;

            // Calculate next D partially (M->D only)
            let tsc_md = _mm_loadu_si128(om.twv[tsc_idx].as_ptr() as *const __m128i);
            tsc_idx += 1;
            dcv = _mm_adds_epi16(sv, tsc_md);
            dmaxv = _mm_max_epi16(dcv, dmaxv);

            // Calculate and store I
            let tsc_mi = _mm_loadu_si128(om.twv[tsc_idx].as_ptr() as *const __m128i);
            tsc_idx += 1;
            let tsc_ii = _mm_loadu_si128(om.twv[tsc_idx].as_ptr() as *const __m128i);
            tsc_idx += 1;
            let isv = _mm_max_epi16(_mm_adds_epi16(mpv, tsc_mi), _mm_adds_epi16(ipv, tsc_ii));
            imx!(q) = isv;
        }

        // Horizontal max of xEv
        let xe = hmax_epi16(xev);
        if xe >= 32767 {
            return VitResult::Overflow;
        }

        // Special states (scalar)
        xn = add_i16(xn, om.xw[P7O_N][P7O_LOOP]);
        xc = add_i16(xc, om.xw[P7O_C][P7O_LOOP]).max(add_i16(xe, om.xw[P7O_E][P7O_MOVE]));
        xj = add_i16(xj, om.xw[P7O_J][P7O_LOOP]).max(add_i16(xe, om.xw[P7O_E][P7O_LOOP]));
        xb = add_i16(xj, om.xw[P7O_J][P7O_MOVE]).max(add_i16(xn, om.xw[P7O_N][P7O_MOVE]));

        // Lazy F loop: check if DD paths need evaluation
        let dmax = hmax_epi16(dmaxv);
        if (dmax as i32) + (om.ddbound_w as i32) > (xb as i32) {
            // Must compute DD paths
            dcv = _mm_slli_si128::<2>(dcv);
            dcv = _mm_or_si128(dcv, neg_inf_v);
            let dd_offset = 7 * q_count;

            for q in 0..q_count {
                dmx!(q) = _mm_max_epi16(dcv, dmx!(q));
                let tsc_dd = _mm_loadu_si128(om.twv[dd_offset + q].as_ptr() as *const __m128i);
                dcv = _mm_adds_epi16(dmx!(q), tsc_dd);
            }

            // Up to 3 more passes for segment boundary improvements
            loop {
                dcv = _mm_slli_si128::<2>(dcv);
                dcv = _mm_or_si128(dcv, neg_inf_v);
                let mut broke = false;
                for q in 0..q_count {
                    if !any_gt_epi16(dcv, dmx!(q)) {
                        broke = true;
                        break;
                    }
                    dmx!(q) = _mm_max_epi16(dcv, dmx!(q));
                    let tsc_dd = _mm_loadu_si128(om.twv[dd_offset + q].as_ptr() as *const __m128i);
                    dcv = _mm_adds_epi16(dmx!(q), tsc_dd);
                }
                if broke {
                    break;
                }
            }
        } else {
            // Just store last M->D partial
            dcv = _mm_slli_si128::<2>(dcv);
            dmx!(0) = _mm_or_si128(dcv, neg_inf_v);
        }
    }

    // Final score: C->T
    if xc > -32768 {
        let mut sc = xc as f32 + om.xw[P7O_C][P7O_MOVE] as f32 - om.base_w as f32;
        sc /= om.scale_w;
        sc -= 3.0; // NN/CC/JJ approximation
        VitResult::Ok(sc)
    } else {
        VitResult::Ok(f32::NEG_INFINITY)
    }
}

/// Horizontal max of 8 int16 elements in an SSE2 vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hmax_epi16(a: __m128i) -> i16 {
    let a = _mm_max_epi16(
        a,
        _mm_shuffle_epi32::<{ super::shuffle_mask(1, 0, 3, 2) }>(a),
    );
    let a = _mm_max_epi16(
        a,
        _mm_shufflelo_epi16::<{ super::shuffle_mask(1, 0, 3, 2) }>(a),
    );
    let a = _mm_max_epi16(a, _mm_srli_epi32::<16>(a));
    _mm_cvtsi128_si32(a) as i16
}

/// Check if any element of a > b.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn any_gt_epi16(a: __m128i, b: __m128i) -> bool {
    _mm_movemask_epi8(_mm_cmpgt_epi16(a, b)) != 0
}

#[inline(always)]
fn add_i16(a: i16, b: i16) -> i16 {
    (a as i32 + b as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::profile::*;
    use std::path::Path;

    #[test]
    fn test_viterbi_filter_basic() {
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

        // Test with a non-matching sequence
        let dsq = abc.digitize(b"AAAAAAAAAAGGGGGGGGGG");
        let result = unsafe { viterbi_filter(&dsq, 20, &om) };
        match result {
            VitResult::Ok(sc) => {
                assert!(sc.is_finite(), "Vit score should be finite, got {}", sc);
            }
            VitResult::Overflow => {}
        }
    }
}
