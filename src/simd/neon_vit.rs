//! ARM NEON Viterbi filter (8x int16 vectors, same width as SSE2).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;

pub enum NeonVitResult {
    Ok(f32),
    Overflow,
}

/// NEON Viterbi filter — same algorithm as SSE2 but with ARM intrinsics.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_viterbi_filter(dsq: &[Dsq], l: usize, om: &OProfile) -> NeonVitResult {
    let q_count = nqw(om.m);
    let nscells = 3;
    let neg_inf = -32768i16;
    let mut dp: Vec<int16x8_t> = vec![vdupq_n_s16(neg_inf); q_count * nscells];

    macro_rules! mmx { ($q:expr) => { dp[$q * nscells + 0] }; }
    macro_rules! dmx { ($q:expr) => { dp[$q * nscells + 1] }; }
    macro_rules! imx { ($q:expr) => { dp[$q * nscells + 2] }; }

    let mut xn: i16 = om.base_w;
    let mut xb: i16 = xn.saturating_add(om.xw[P7O_N][P7O_MOVE]);
    let mut xj: i16 = neg_inf;
    let mut xc: i16 = neg_inf;

    for i in 1..=l {
        let xi = dsq[i] as usize;
        if xi >= om.abc_kp { continue; }
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
            let tsc_bm = vld1q_s16(om.twv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tsc_mm = vld1q_s16(om.twv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tsc_im = vld1q_s16(om.twv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tsc_dm = vld1q_s16(om.twv[tsc_idx].as_ptr()); tsc_idx += 1;

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

            let tsc_md = vld1q_s16(om.twv[tsc_idx].as_ptr()); tsc_idx += 1;
            dcv = vqaddq_s16(sv, tsc_md);
            dmaxv = vmaxq_s16(dcv, dmaxv);

            let tsc_mi = vld1q_s16(om.twv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tsc_ii = vld1q_s16(om.twv[tsc_idx].as_ptr()); tsc_idx += 1;
            imx!(q) = vmaxq_s16(vqaddq_s16(mpv, tsc_mi), vqaddq_s16(ipv, tsc_ii));
        }

        let xe = vmaxvq_s16(xev);
        if xe >= 32767 { return NeonVitResult::Overflow; }

        xn = xn.saturating_add(om.xw[P7O_N][P7O_LOOP]);
        xc = (xc.saturating_add(om.xw[P7O_C][P7O_LOOP]))
            .max(xe.saturating_add(om.xw[P7O_E][P7O_MOVE]));
        xj = (xj.saturating_add(om.xw[P7O_J][P7O_LOOP]))
            .max(xe.saturating_add(om.xw[P7O_E][P7O_LOOP]));
        xb = (xj.saturating_add(om.xw[P7O_J][P7O_MOVE]))
            .max(xn.saturating_add(om.xw[P7O_N][P7O_MOVE]));

        let dmax = vmaxvq_s16(dmaxv);
        if (dmax as i32) + (om.ddbound_w as i32) > (xb as i32) {
            dcv = vextq_s16(vdupq_n_s16(neg_inf), dcv, 7);
            let dd_offset = 7 * q_count;
            for q in 0..q_count {
                dmx!(q) = vmaxq_s16(dcv, dmx!(q));
                let tdd = vld1q_s16(om.twv[dd_offset + q].as_ptr());
                dcv = vqaddq_s16(dmx!(q), tdd);
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

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_viterbi_filter(_dsq: &[Dsq], _l: usize, _om: &OProfile) -> NeonVitResult {
    NeonVitResult::Ok(f32::NEG_INFINITY)
}
