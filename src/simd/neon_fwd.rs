//! ARM NEON Forward parser (4x float vectors, same width as SSE2).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;

pub enum NeonFwdResult {
    Ok(f32),
    Error,
}

/// NEON Forward parser.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_forward_parser(dsq: &[Dsq], l: usize, om: &OProfile) -> f32 {
    let q_count = nqf(om.m);
    let nscells = 3;
    let mut dp: Vec<float32x4_t> = vec![vdupq_n_f32(0.0); q_count * nscells];
    let zerov = vdupq_n_f32(0.0);

    macro_rules! mmo { ($q:expr) => { dp[$q * nscells + 0] }; }
    macro_rules! dmo { ($q:expr) => { dp[$q * nscells + 1] }; }
    macro_rules! imo { ($q:expr) => { dp[$q * nscells + 2] }; }

    let mut xe: f32 = 0.0;
    let mut xn: f32 = 1.0;
    let mut xj: f32 = 0.0;
    let mut xb: f32 = om.xf[P7O_N][P7O_MOVE];
    let mut xc: f32 = 0.0;
    let mut totscale: f32 = 0.0;

    for i in 1..=l {
        let xi = dsq[i] as usize;
        if xi >= om.abc_kp { continue; }
        let rsc = &om.rfv[xi];

        let mut dcv = zerov;
        let mut xev = zerov;
        let xbv = vdupq_n_f32(xb);

        let mut mpv = vextq_f32(zerov, mmo!(q_count - 1), 3);
        let mut dpv = vextq_f32(zerov, dmo!(q_count - 1), 3);
        let mut ipv = vextq_f32(zerov, imo!(q_count - 1), 3);

        let mut tsc_idx = 0;
        for q in 0..q_count {
            let tbm = vld1q_f32(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tmm = vld1q_f32(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tim = vld1q_f32(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tdm = vld1q_f32(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;

            let mut sv = vmulq_f32(xbv, tbm);
            sv = vaddq_f32(sv, vmulq_f32(mpv, tmm));
            sv = vaddq_f32(sv, vmulq_f32(ipv, tim));
            sv = vaddq_f32(sv, vmulq_f32(dpv, tdm));
            let rsc_v = vld1q_f32(rsc[q].as_ptr());
            sv = vmulq_f32(sv, rsc_v);
            xev = vaddq_f32(xev, sv);

            mpv = mmo!(q);
            dpv = dmo!(q);
            ipv = imo!(q);

            mmo!(q) = sv;
            dmo!(q) = dcv;

            let tmd = vld1q_f32(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            dcv = vmulq_f32(sv, tmd);

            let tmi = vld1q_f32(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tii = vld1q_f32(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            imo!(q) = vaddq_f32(vmulq_f32(mpv, tmi), vmulq_f32(ipv, tii));
        }

        // DD paths
        dcv = vextq_f32(zerov, dcv, 3);
        dmo!(0) = zerov;
        let dd_offset = 7 * q_count;
        for q in 0..q_count {
            dmo!(q) = vaddq_f32(dcv, dmo!(q));
            let tdd = vld1q_f32(om.tfv[dd_offset + q].as_ptr());
            dcv = vmulq_f32(dmo!(q), tdd);
        }
        for _ in 1..4 {
            dcv = vextq_f32(zerov, dcv, 3);
            for q in 0..q_count {
                dmo!(q) = vaddq_f32(dcv, dmo!(q));
                let tdd = vld1q_f32(om.tfv[dd_offset + q].as_ptr());
                dcv = vmulq_f32(dcv, tdd);
            }
        }

        for q in 0..q_count { xev = vaddq_f32(dmo!(q), xev); }

        // Horizontal sum
        xe = vaddvq_f32(xev);

        xn *= om.xf[P7O_N][P7O_LOOP];
        xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
        xb = xj * om.xf[P7O_J][P7O_MOVE] + xn * om.xf[P7O_N][P7O_MOVE];

        if xe > 1.0e4 {
            let scale = 1.0 / xe;
            xn *= scale; xc *= scale; xj *= scale; xb *= scale;
            let sv = vdupq_n_f32(scale);
            for q in 0..q_count {
                mmo!(q) = vmulq_f32(mmo!(q), sv);
                dmo!(q) = vmulq_f32(dmo!(q), sv);
                imo!(q) = vmulq_f32(imo!(q), sv);
            }
            totscale += xe.ln();
            xe = 1.0;
        }
    }

    if xc.is_nan() || (l > 0 && xc == 0.0) || xc.is_infinite() {
        return f32::NEG_INFINITY;
    }
    totscale + (xc * om.xf[P7O_C][P7O_MOVE]).ln()
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_forward_parser(_dsq: &[Dsq], _l: usize, _om: &OProfile) -> f32 {
    f32::NEG_INFINITY
}
