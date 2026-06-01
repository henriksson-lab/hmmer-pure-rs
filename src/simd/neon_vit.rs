//! ARM NEON Viterbi filter (8x int16 vectors, same width as SSE2).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;

/// Result of the NEON Viterbi filter: either a finite score or a saturating overflow.
pub enum NeonVitResult {
    Ok(f32),
    Overflow,
}

/// NEON variant of the Viterbi filter (C: `p7_ViterbiFilter`), using 8x int16 vectors.
///
/// Calculates an approximation of the Viterbi score in nats for digital sequence `dsq`
/// of length `l` using optimized profile `om`. Score may overflow on extremely
/// high-scoring sequences but will not underflow. The model must be in a local
/// alignment mode (the only mode that guarantees the limited dynamic range needed for
/// reduced-precision signed-word arithmetic).
///
/// Striped SIMD Viterbi after Farrar (2007), in 16-bit signed-word precision, with the
/// same algorithm as the SSE2 reference but expressed in ARM NEON intrinsics.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_viterbi_filter(dsq: &[Dsq], l: usize, om: &OProfile) -> NeonVitResult {
    let q_count = nqw(om.m);
    let nscells = 3;
    let neg_inf = -32768i16;
    let mut dp: Vec<int16x8_t> = vec![vdupq_n_s16(neg_inf); q_count * nscells];

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

    let mut xn: i16 = om.base_w;
    let mut xb: i16 = add_i16(xn, om.xw[P7O_N][P7O_MOVE]);
    let mut xj: i16 = neg_inf;
    let mut xc: i16 = neg_inf;

    for i in 1..=l {
        // Match C/SSE: rwv is built for every valid digital code and every
        // dsq[1..=L] row advances unconditionally.
        let xi = dsq[i] as usize;
        let rsc = &om.rwv[xi];

        let mut dcv = vdupq_n_s16(neg_inf);
        let mut xev = vdupq_n_s16(neg_inf);
        let mut dmaxv = vdupq_n_s16(neg_inf);
        let xbv = vdupq_n_s16(xb);

        let mut mpv = vextq_s16(vdupq_n_s16(neg_inf), mmx!(q_count - 1), 7);
        let mut dpv = vextq_s16(vdupq_n_s16(neg_inf), dmx!(q_count - 1), 7);
        let mut ipv = vextq_s16(vdupq_n_s16(neg_inf), imx!(q_count - 1), 7);

        let mut tsc_idx = 0;
        for q in 0..q_count {
            let tsc_bm = vld1q_s16(om.twv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tsc_mm = vld1q_s16(om.twv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tsc_im = vld1q_s16(om.twv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tsc_dm = vld1q_s16(om.twv[tsc_idx].as_ptr());
            tsc_idx += 1;

            let mut sv = vqaddq_s16(xbv, tsc_bm);
            sv = vmaxq_s16(sv, vqaddq_s16(mpv, tsc_mm));
            sv = vmaxq_s16(sv, vqaddq_s16(ipv, tsc_im));
            sv = vmaxq_s16(sv, vqaddq_s16(dpv, tsc_dm));

            let rsc_v = vld1q_s16(rsc[q].as_ptr());
            sv = vqaddq_s16(sv, rsc_v);
            xev = vmaxq_s16(xev, sv);

            mpv = mmx!(q);
            dpv = dmx!(q);
            ipv = imx!(q);

            mmx!(q) = sv;
            dmx!(q) = dcv;

            let tsc_md = vld1q_s16(om.twv[tsc_idx].as_ptr());
            tsc_idx += 1;
            dcv = vqaddq_s16(sv, tsc_md);
            dmaxv = vmaxq_s16(dcv, dmaxv);

            let tsc_mi = vld1q_s16(om.twv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tsc_ii = vld1q_s16(om.twv[tsc_idx].as_ptr());
            tsc_idx += 1;
            imx!(q) = vmaxq_s16(vqaddq_s16(mpv, tsc_mi), vqaddq_s16(ipv, tsc_ii));
        }

        let xe = vmaxvq_s16(xev);
        if xe >= 32767 {
            return NeonVitResult::Overflow;
        }

        xn = add_i16(xn, om.xw[P7O_N][P7O_LOOP]);
        xc = add_i16(xc, om.xw[P7O_C][P7O_LOOP]).max(add_i16(xe, om.xw[P7O_E][P7O_MOVE]));
        xj = add_i16(xj, om.xw[P7O_J][P7O_LOOP]).max(add_i16(xe, om.xw[P7O_E][P7O_LOOP]));
        xb = add_i16(xj, om.xw[P7O_J][P7O_MOVE]).max(add_i16(xn, om.xw[P7O_N][P7O_MOVE]));

        let dmax = vmaxvq_s16(dmaxv);
        if (dmax as i32) + (om.ddbound_w as i32) > (xb as i32) {
            dcv = vextq_s16(vdupq_n_s16(neg_inf), dcv, 7);
            let dd_offset = 7 * q_count;
            for q in 0..q_count {
                dmx!(q) = vmaxq_s16(dcv, dmx!(q));
                let tdd = vld1q_s16(om.twv[dd_offset + q].as_ptr());
                dcv = vqaddq_s16(dmx!(q), tdd);
            }
            loop {
                dcv = vextq_s16(vdupq_n_s16(neg_inf), dcv, 7);
                let mut broke = false;
                for q in 0..q_count {
                    if !any_gt_s16(dcv, dmx!(q)) {
                        broke = true;
                        break;
                    }
                    dmx!(q) = vmaxq_s16(dcv, dmx!(q));
                    let tdd = vld1q_s16(om.twv[dd_offset + q].as_ptr());
                    dcv = vqaddq_s16(dmx!(q), tdd);
                }
                if broke {
                    break;
                }
            }
        } else {
            dcv = vextq_s16(vdupq_n_s16(neg_inf), dcv, 7);
            dmx!(0) = dcv;
        }
    }

    if xc > neg_inf {
        let mut sc = xc as f32 + om.xw[P7O_C][P7O_MOVE] as f32 - om.base_w as f32;
        sc /= om.scale_w;
        sc -= 3.0;
        NeonVitResult::Ok(sc)
    } else {
        NeonVitResult::Ok(f32::NEG_INFINITY)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn add_i16(a: i16, b: i16) -> i16 {
    (a as i32 + b as i32) as i16
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn any_gt_s16(a: int16x8_t, b: int16x8_t) -> bool {
    vmaxvq_u16(vcgtq_s16(a, b)) != 0
}

/// Non-aarch64 stub for [`neon_viterbi_filter`]; always returns `Ok(-inf)`.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_viterbi_filter(_dsq: &[Dsq], _l: usize, _om: &OProfile) -> NeonVitResult {
    NeonVitResult::Ok(f32::NEG_INFINITY)
}
