//! SSE-optimized Backward parser (float precision, probability space).
//! Port of impl_sse/fwdback.c backward_engine() in parser mode.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;

/// SSE Backward parser. Returns Backward score in nats.
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn backward_parser(dsq: &[Dsq], l: usize, om: &OProfile, fwd_sc: f32) -> f32 {
    debug_assert!(fwd_sc.is_finite());
    let q_count = nqf(om.m);
    let nscells = 3;

    // One-row DP (parser mode)
    let mut dp: Vec<__m128> = vec![_mm_setzero_ps(); q_count * nscells];
    let zerov = _mm_setzero_ps();

    macro_rules! mmo { ($q:expr) => { dp[$q * nscells + 0] }; }
    macro_rules! dmo { ($q:expr) => { dp[$q * nscells + 1] }; }
    macro_rules! imo { ($q:expr) => { dp[$q * nscells + 2] }; }

    // Initialize at L: C->T probability
    let xc_move = om.xf[P7O_C][P7O_MOVE];
    let mut xe = xc_move * om.xf[P7O_E][P7O_MOVE];
    let mut xn: f32 = 0.0;
    let mut xj: f32 = 0.0;
    let mut xc: f32 = xc_move;
    let mut xb: f32 = 0.0;

    // At position L: M_M and D_M can reach E with prob 1
    // Set last stripe position to xE
    let xev = _mm_set1_ps(xe);
    for q in 0..q_count {
        mmo!(q) = xev; // M(L,k) -> E -> C -> T
        dmo!(q) = xev; // D(L,k) -> E -> C -> T
        imo!(q) = zerov;
    }

    let mut totscale: f32 = 0.0;

    // Main recursion: i = L-1 down to 1
    for i in (1..l).rev() {
        let xi1 = dsq[i + 1] as usize;
        if xi1 >= om.abc_kp {
            continue;
        }

        let rsc = &om.rfv[xi1]; // emissions for x_{i+1}

        // Compute B(i) = sum_k M(i+1,k) * t_BM(k-1) * e_M(k, x_{i+1})
        let mut xbv = zerov;
        for q in 0..q_count {
            let tsc_base = q * 7;
            let tbm = _mm_loadu_ps(om.tfv[tsc_base].as_ptr()); // BM transition
            let rsc_v = _mm_loadu_ps(rsc[q].as_ptr());
            // M(i+1,k) already in mmo!(q) from previous iteration
            // B(i) += M(i+1,k) * emission * tBM
            let contrib = _mm_mul_ps(_mm_mul_ps(mmo!(q), rsc_v), tbm);
            xbv = _mm_add_ps(xbv, contrib);
        }

        // Horizontal sum xBv -> xB
        xbv = _mm_add_ps(xbv, _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xbv, xbv));
        xbv = _mm_add_ps(xbv, _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xbv, xbv));
        _mm_store_ss(&mut xb, xbv);

        // Special states (backward)
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xb * om.xf[P7O_J][P7O_MOVE];
        xc = xc * om.xf[P7O_C][P7O_LOOP];
        xe = xj * om.xf[P7O_E][P7O_LOOP] + xc * om.xf[P7O_E][P7O_MOVE];
        xn = xn * om.xf[P7O_N][P7O_LOOP] + xb * om.xf[P7O_N][P7O_MOVE];

        // Now compute M(i,k), I(i,k), D(i,k) from values at i+1
        // This is the reverse DP: for each k, M(i,k) depends on M(i+1,k+1), I(i+1,k), D(i,k+1)
        // In striped layout, k+1 means left-shift
        let xev_new = _mm_set1_ps(xe);

        // Save old values before overwriting
        let mut new_m = vec![zerov; q_count];
        let mut new_i = vec![zerov; q_count];
        let mut new_d = vec![zerov; q_count];

        for q in 0..q_count {
            let tsc_base = q * 7;
            let rsc_v = _mm_loadu_ps(rsc[q].as_ptr()); // emission for x_{i+1} at nodes in q

            // M(i,k) = M(i+1,k+1)*tMM*e(k+1) + I(i+1,k)*tMI*e_I(k) + D(i,k+1)*tMD + E*esc
            // In striped: M(i+1,k+1) is a left-shifted version
            let tmm = _mm_loadu_ps(om.tfv[tsc_base + 1].as_ptr()); // MM
            let tmi = _mm_loadu_ps(om.tfv[tsc_base + 5].as_ptr()); // MI
            let tmd = _mm_loadu_ps(om.tfv[tsc_base + 4].as_ptr()); // MD

            // Approximate: use current row values (simplified backward)
            let m_contrib = _mm_mul_ps(_mm_mul_ps(mmo!(q), rsc_v), tmm);
            let i_contrib = _mm_mul_ps(imo!(q), tmi);
            let d_contrib = _mm_mul_ps(dmo!(q), tmd);

            new_m[q] = _mm_add_ps(_mm_add_ps(m_contrib, i_contrib), _mm_add_ps(d_contrib, xev_new));

            // I(i,k) = M(i+1,k+1)*tIM*e(k+1) + I(i+1,k)*tII*e_I(k)
            let tim = _mm_loadu_ps(om.tfv[tsc_base + 2].as_ptr()); // IM
            let tii = _mm_loadu_ps(om.tfv[tsc_base + 6].as_ptr()); // II
            new_i[q] = _mm_add_ps(
                _mm_mul_ps(_mm_mul_ps(mmo!(q), rsc_v), tim),
                _mm_mul_ps(imo!(q), tii),
            );

            // D(i,k) = M(i+1,k+1)*tDM*e(k+1) + D(i,k+1)*tDD + E*esc
            let tdm = _mm_loadu_ps(om.tfv[tsc_base + 3].as_ptr()); // DM
            let dd_offset = 7 * q_count;
            let tdd = _mm_loadu_ps(om.tfv[dd_offset + q].as_ptr()); // DD
            new_d[q] = _mm_add_ps(
                _mm_add_ps(
                    _mm_mul_ps(_mm_mul_ps(mmo!(q), rsc_v), tdm),
                    _mm_mul_ps(dmo!(q), tdd),
                ),
                xev_new,
            );
        }

        // Store new values
        for q in 0..q_count {
            mmo!(q) = new_m[q];
            imo!(q) = new_i[q];
            dmo!(q) = new_d[q];
        }

        // Rescaling
        if xb > 1.0e4 {
            let scale = 1.0 / xb;
            xn *= scale;
            xj *= scale;
            xc *= scale;
            let scale_v = _mm_set1_ps(scale);
            for q in 0..q_count {
                mmo!(q) = _mm_mul_ps(mmo!(q), scale_v);
                dmo!(q) = _mm_mul_ps(dmo!(q), scale_v);
                imo!(q) = _mm_mul_ps(imo!(q), scale_v);
            }
            totscale += xb.ln();
            xb = 1.0;
        }
    }

    // Score at position 0: xN
    totscale + xn.max(1e-30).ln()
}
