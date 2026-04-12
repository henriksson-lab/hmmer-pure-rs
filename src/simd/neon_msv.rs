//! ARM NEON-optimized MSV filter (16x uint8 vectors).
//! For Apple Silicon and ARM servers.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::OProfile;

/// Result of NEON MSV filter.
pub enum NeonMsvResult {
    Ok(f32),
    Overflow,
}

/// NEON MSV filter using 16x uint8 vectors (same width as SSE2).
/// Uses the same OProfile byte data as SSE2.
///
/// # Safety
/// Requires NEON support (always available on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_msv_filter(dsq: &[Dsq], l: usize, om: &OProfile) -> NeonMsvResult {
    let q_count = crate::simd::oprofile::nqb(om.m);

    let mut dp: Vec<uint8x16_t> = vec![vdupq_n_u8(0); q_count];

    let biasv = vdupq_n_u8(om.bias_b);
    let basev = vdupq_n_u8(om.base_b);

    let tjbm = om.tjb_b.wrapping_add(om.tbm_b);
    let tjbmv = vdupq_n_u8(tjbm);
    let tecv = vdupq_n_u8(om.tec_b);

    let mut xjv = vdupq_n_u8(0);
    let mut xbv = vqsubq_u8(basev, tjbmv);

    for i in 1..=l {
        let xi = dsq[i] as usize;
        if xi >= om.abc_kp {
            continue;
        }
        let rsc = &om.rbv[xi];

        let mut xev = vdupq_n_u8(0);

        // Right shift by 1 byte
        let mut mpv = vextq_u8(vdupq_n_u8(0), dp[q_count - 1], 15);

        for q in 0..q_count {
            let mut sv = vmaxq_u8(mpv, xbv);
            sv = vqaddq_u8(sv, biasv);
            let rsc_v = vld1q_u8(rsc[q].as_ptr());
            sv = vqsubq_u8(sv, rsc_v);
            xev = vmaxq_u8(xev, sv);

            mpv = dp[q];
            dp[q] = sv;
        }

        // Overflow test
        let tempv = vqaddq_u8(xev, biasv);
        let cmpv = vceqq_u8(tempv, vdupq_n_u8(255));
        if vmaxvq_u8(cmpv) != 0 {
            return NeonMsvResult::Overflow;
        }

        // Horizontal max of xev
        let xe_max = vmaxvq_u8(xev);
        xev = vdupq_n_u8(xe_max);

        xev = vqsubq_u8(xev, tecv);
        xjv = vmaxq_u8(xjv, xev);
        xbv = vmaxq_u8(basev, xjv);
        xbv = vqsubq_u8(xbv, tjbmv);
    }

    let xj = vgetq_lane_u8(xjv, 0);

    let mut sc = (xj.wrapping_sub(om.tjb_b) as f32) - om.base_b as f32;
    sc /= om.scale_b;
    sc -= 3.0;

    NeonMsvResult::Ok(sc)
}

// On non-aarch64 targets, provide a stub
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_msv_filter(_dsq: &[Dsq], _l: usize, _om: &OProfile) -> NeonMsvResult {
    NeonMsvResult::Ok(f32::NEG_INFINITY) // NEON not available
}
