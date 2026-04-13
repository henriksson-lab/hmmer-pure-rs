//! SSE-optimized Backward parser that also computes domain decoding arrays.
//! Combines backward_parser + domain_decoding into a single pass.
//!
//! Port of p7_BackwardParser() + p7_DomainDecoding() from C HMMER.
//! Posterior formula: P(state at i) = fwd[i]*bck[i]*exp(fwd_cs[i]+bck_cs[i]-fwd_score)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::simd::fwd_filter::FwdSpecials;
use crate::simd::oprofile::*;

/// Result of backward parser with domain decoding.
pub struct BckDecodingResult {
    pub bck_sc: f32,
    pub btot: Vec<f32>,
    pub etot: Vec<f32>,
    pub mocc: Vec<f32>,
}

/// Run Backward parser and compute domain decoding arrays.
///
/// Matches C's p7_DomainDecoding() posterior computation:
/// - btot[i] += fwd_B[i-1] * bck_B[i-1] * exp(fwd_cs[i-1] + bck_cs[i-1] - fwd_score)
/// - etot[i] += fwd_E[i]   * bck_E[i]   * exp(fwd_cs[i]   + bck_cs[i]   - fwd_score)
/// - mocc[i] = 1 - sum(fwd_S[i-1]*bck_S[i]*t_S_loop * exp(fwd_cs[i-1]+bck_cs[i]-fwd_score))
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn backward_parser_with_decoding(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    fwd_specials: &[FwdSpecials],
    fwd_sc: f32,
) -> BckDecodingResult {
    let q_count = nqf(om.m);
    let nscells = 3;

    let mut dp: Vec<__m128> = vec![_mm_setzero_ps(); q_count * nscells];
    let zerov = _mm_setzero_ps();

    macro_rules! mmo { ($q:expr) => { dp[$q * nscells + 0] }; }
    macro_rules! dmo { ($q:expr) => { dp[$q * nscells + 1] }; }
    macro_rules! imo { ($q:expr) => { dp[$q * nscells + 2] }; }

    // Initialize at L
    let xc_move = om.xf[P7O_C][P7O_MOVE];
    let mut xe = xc_move * om.xf[P7O_E][P7O_MOVE];
    let mut xn: f32 = 0.0;
    let mut xj: f32 = 0.0;
    let mut xc: f32 = xc_move;
    let mut xb: f32 = 0.0;

    let xev = _mm_set1_ps(xe);
    for q in 0..q_count {
        mmo!(q) = xev;
        dmo!(q) = xev;
        imo!(q) = zerov;
    }

    let mut bck_totscale: f32 = 0.0;

    // Domain decoding arrays
    let mut btot = vec![0.0_f32; l + 1];
    let mut etot = vec![0.0_f32; l + 1];
    let mut mocc = vec![0.0_f32; l + 1];

    // Save backward specials per position for domain decoding
    // We need bck values at position i BEFORE processing position i's DP
    // (i.e., the backward values looking from position i toward L)
    let mut bck_specials: Vec<(f32, f32, f32, f32, f32, f32)> = vec![(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); l + 1];
    // (xn, xj, xc, xb, xe, totscale)
    bck_specials[l] = (xn, xj, xc, xb, xe, bck_totscale);

    // Main recursion: i = L-1 down to 1
    for i in (1..l).rev() {
        let xi1 = dsq[i + 1] as usize;
        if xi1 >= om.abc_kp {
            bck_specials[i] = (xn, xj, xc, xb, xe, bck_totscale);
            continue;
        }

        let rsc = &om.rfv[xi1];

        // Compute B(i)
        let mut xbv = zerov;
        for q in 0..q_count {
            let tsc_base = q * 7;
            let tbm = _mm_loadu_ps(om.tfv[tsc_base].as_ptr());
            let rsc_v = _mm_loadu_ps(rsc[q].as_ptr());
            let contrib = _mm_mul_ps(_mm_mul_ps(mmo!(q), rsc_v), tbm);
            xbv = _mm_add_ps(xbv, contrib);
        }
        xbv = _mm_add_ps(xbv, _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xbv, xbv));
        xbv = _mm_add_ps(xbv, _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xbv, xbv));
        _mm_store_ss(&mut xb, xbv);

        // Special states
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xb * om.xf[P7O_J][P7O_MOVE];
        xc = xc * om.xf[P7O_C][P7O_LOOP];
        xe = xj * om.xf[P7O_E][P7O_LOOP] + xc * om.xf[P7O_E][P7O_MOVE];
        xn = xn * om.xf[P7O_N][P7O_LOOP] + xb * om.xf[P7O_N][P7O_MOVE];

        // DP cells
        let xev_new = _mm_set1_ps(xe);
        let mut new_m = vec![zerov; q_count];
        let mut new_i = vec![zerov; q_count];
        let mut new_d = vec![zerov; q_count];

        for q in 0..q_count {
            let tsc_base = q * 7;
            let rsc_v = _mm_loadu_ps(rsc[q].as_ptr());

            let tmm = _mm_loadu_ps(om.tfv[tsc_base + 1].as_ptr());
            let tmi = _mm_loadu_ps(om.tfv[tsc_base + 5].as_ptr());
            let tmd = _mm_loadu_ps(om.tfv[tsc_base + 4].as_ptr());
            let m_contrib = _mm_mul_ps(_mm_mul_ps(mmo!(q), rsc_v), tmm);
            let i_contrib = _mm_mul_ps(imo!(q), tmi);
            let d_contrib = _mm_mul_ps(dmo!(q), tmd);
            new_m[q] = _mm_add_ps(_mm_add_ps(m_contrib, i_contrib), _mm_add_ps(d_contrib, xev_new));

            let tim = _mm_loadu_ps(om.tfv[tsc_base + 2].as_ptr());
            let tii = _mm_loadu_ps(om.tfv[tsc_base + 6].as_ptr());
            new_i[q] = _mm_add_ps(
                _mm_mul_ps(_mm_mul_ps(mmo!(q), rsc_v), tim),
                _mm_mul_ps(imo!(q), tii),
            );

            let tdm = _mm_loadu_ps(om.tfv[tsc_base + 3].as_ptr());
            let dd_offset = 7 * q_count;
            let tdd = _mm_loadu_ps(om.tfv[dd_offset + q].as_ptr());
            new_d[q] = _mm_add_ps(
                _mm_add_ps(
                    _mm_mul_ps(_mm_mul_ps(mmo!(q), rsc_v), tdm),
                    _mm_mul_ps(dmo!(q), tdd),
                ),
                xev_new,
            );
        }

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
            bck_totscale += xb.ln();
            xb = 1.0;
        }

        bck_specials[i] = (xn, xj, xc, xb, xe, bck_totscale);
    }

    // Handle position 0 backward
    if l >= 1 {
        let xi1 = dsq[1] as usize;
        if xi1 < om.abc_kp {
            let rsc = &om.rfv[xi1];
            let mut xbv = zerov;
            for q in 0..q_count {
                let tsc_base = q * 7;
                let tbm = _mm_loadu_ps(om.tfv[tsc_base].as_ptr());
                let rsc_v = _mm_loadu_ps(rsc[q].as_ptr());
                let contrib = _mm_mul_ps(_mm_mul_ps(mmo!(q), rsc_v), tbm);
                xbv = _mm_add_ps(xbv, contrib);
            }
            xbv = _mm_add_ps(xbv, _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xbv, xbv));
            xbv = _mm_add_ps(xbv, _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xbv, xbv));
            _mm_store_ss(&mut xb, xbv);
        }
        xn = xn * om.xf[P7O_N][P7O_LOOP] + xb * om.xf[P7O_N][P7O_MOVE];
    }
    bck_specials[0] = (xn, xj, xc, xb, xe, bck_totscale);

    let bck_sc = bck_totscale + xn.max(1e-30).ln();

    // Domain decoding: compute btot, etot, mocc from saved forward + backward specials
    // Matching C's p7_DomainDecoding() index conventions:
    //   btot[i] += fwd_B[i-1] * bck_B[i-1] * exp(fwd_cs[i-1] + bck_cs[i-1] - fwd_score)
    //   etot[i] += fwd_E[i]   * bck_E[i]   * exp(fwd_cs[i]   + bck_cs[i]   - fwd_score)
    //   njcp    += fwd_S[i-1] * bck_S[i]   * t_S * exp(fwd_cs[i-1] + bck_cs[i] - fwd_score)
    for i in 1..=l {
        let fi_prev = &fwd_specials[i - 1]; // forward at row i-1
        let fi = &fwd_specials[i];           // forward at row i
        let bi_prev = &bck_specials[i - 1];   // backward at row i-1
        let bi = &bck_specials[i];             // backward at row i

        // B posterior at position i (uses fwd/bck at row i-1)
        let b_scale = (fi_prev.totscale + bi_prev.5 - fwd_sc).exp();
        let b_post = fi_prev.xb * bi_prev.3 * b_scale; // .3 = xb, .5 = totscale
        btot[i] = btot[i - 1] + b_post.max(0.0);

        // E posterior at position i (uses fwd/bck at row i)
        let e_scale = (fi.totscale + bi.5 - fwd_sc).exp();
        let e_post = fi.xe * bi.4 * e_scale; // .4 = xe
        etot[i] = etot[i - 1] + e_post.max(0.0);

        // NJC loop posteriors (fwd at row i-1, bck at row i)
        let njc_scale = (fi_prev.totscale + bi.5 - fwd_sc).exp();
        let n_loop = fi_prev.xn * bi.0 * om.xf[P7O_N][P7O_LOOP] * njc_scale;
        let j_loop = fi_prev.xj * bi.1 * om.xf[P7O_J][P7O_LOOP] * njc_scale;
        let c_loop = fi_prev.xc * bi.2 * om.xf[P7O_C][P7O_LOOP] * njc_scale;
        let njcp = n_loop + j_loop + c_loop;
        mocc[i] = (1.0 - njcp).clamp(0.0, 1.0);
    }

    BckDecodingResult {
        bck_sc,
        btot,
        etot,
        mocc,
    }
}
