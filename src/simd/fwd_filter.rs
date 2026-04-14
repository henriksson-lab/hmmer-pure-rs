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
    forward_parser_offset(dsq, 0, l, om)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser_offset(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
) -> f32 {
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

    macro_rules! mmo {
        ($q:expr) => {
            dp[$q * nscells + 0]
        };
    }
    macro_rules! dmo {
        ($q:expr) => {
            dp[$q * nscells + 1]
        };
    }
    macro_rules! imo {
        ($q:expr) => {
            dp[$q * nscells + 2]
        };
    }

    for i in 1..=l {
        let xi = dsq[dsq_offset + i] as usize;
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
            let tbm = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tmm = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tim = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tdm = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;

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
            let tmd = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            dcv = _mm_mul_ps(sv, tmd);

            // I state: M*tMI + I*tII (emission odds ratio = 1.0, so no emission multiply)
            let tmi = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tii = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let isv = _mm_add_ps(_mm_mul_ps(mpv, tmi), _mm_mul_ps(ipv, tii));
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
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xev, xev),
        );
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xev, xev),
        );
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

/// Forward parser that stores specials and cumulative f64 scale into a ProbMx.
/// Uses f64 totscale (matching old codebase) for correct domain decoding normalization.
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser_pmx(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
) -> f32 {
    forward_parser_pmx_offset(dsq, 0, l, om, pmx)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser_pmx_offset(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
) -> f32 {
    let mut dp: Vec<__m128> = Vec::new();
    forward_parser_pmx_offset_with_scratch(dsq, dsq_offset, l, om, pmx, &mut dp)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser_pmx_offset_with_scratch(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
    dp: &mut Vec<__m128>,
) -> f32 {
    if pmx.has_dp && canonical_run(dsq, dsq_offset, l, om.abc_kp) {
        return forward_parser_pmx_offset_direct(dsq, dsq_offset, l, om, pmx);
    }

    use super::probmx::*;
    let q_count = nqf(om.m);
    let nscells = 3;
    let zerov = _mm_setzero_ps();
    dp.resize(q_count * nscells, zerov);
    for v in dp.iter_mut() {
        *v = zerov;
    }
    let dp_ptr = dp.as_mut_ptr();
    let rfv_ptr = om.rfv.as_ptr();
    let tfv_ptr = om.tfv.as_ptr();

    let mut xe: f32 = 0.0;
    let mut xn: f32 = 1.0;
    let mut xj: f32 = 0.0;
    let mut xb: f32 = om.xf[P7O_N][P7O_MOVE];
    let mut xc: f32 = 0.0;
    let mut totscale: f64 = 0.0; // f64 precision for domain decoding
    let mut score_scale: f32 = 0.0; // C forward_engine-style score accumulation

    macro_rules! mmo {
        ($q:expr) => {
            *dp_ptr.add($q * nscells)
        };
    }
    macro_rules! dmo {
        ($q:expr) => {
            *dp_ptr.add($q * nscells + 1)
        };
    }
    macro_rules! imo {
        ($q:expr) => {
            *dp_ptr.add($q * nscells + 2)
        };
    }

    // Row 0
    pmx.set_xmx(0, PXE, 0.0);
    pmx.set_xmx(0, PXN, xn);
    pmx.set_xmx(0, PXJ, 0.0);
    pmx.set_xmx(0, PXB, xb);
    pmx.set_xmx(0, PXC, 0.0);
    pmx.scale[0] = 0.0;

    for i in 1..=l {
        let xi = dsq[dsq_offset + i] as usize;
        if xi >= om.abc_kp {
            if pmx.has_dp {
                pmx.zero_simd_row(i);
            }
            xn *= om.xf[P7O_N][P7O_LOOP];
            xe = 0.0;
            xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
            xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
            xb = xn * om.xf[P7O_N][P7O_MOVE] + xj * om.xf[P7O_J][P7O_MOVE];
            pmx.set_xmx(i, PXE, xe);
            pmx.set_xmx(i, PXN, xn);
            pmx.set_xmx(i, PXJ, xj);
            pmx.set_xmx(i, PXB, xb);
            pmx.set_xmx(i, PXC, xc);
            pmx.scale[i] = totscale;
            continue;
        }

        let rsc_ptr = (*rfv_ptr.add(xi)).as_ptr();
        let xbv = _mm_set1_ps(xb);
        let mut dcv = zerov;
        let mut xev = zerov;

        let mut mpv = rightshift_float(mmo!(q_count - 1));
        let mut dpv = rightshift_float(dmo!(q_count - 1));
        let mut ipv = rightshift_float(imo!(q_count - 1));

        for q_idx in 0..q_count {
            let tsc_base = q_idx * 7;
            let rsc_v = _mm_loadu_ps((*rsc_ptr.add(q_idx)).as_ptr());

            let tbm = _mm_loadu_ps((*tfv_ptr.add(tsc_base)).as_ptr());
            let tmm = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 1)).as_ptr());
            let tim = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 2)).as_ptr());
            let tdm = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 3)).as_ptr());
            let tmd = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 4)).as_ptr());
            let tmi = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 5)).as_ptr());
            let tii = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 6)).as_ptr());

            // M(i,k) = (B*BM + Mprev*MM + Iprev*IM + Dprev*DM) * emission
            let mut sv = _mm_mul_ps(xbv, tbm);
            sv = _mm_add_ps(sv, _mm_mul_ps(mpv, tmm));
            sv = _mm_add_ps(sv, _mm_mul_ps(ipv, tim));
            sv = _mm_add_ps(sv, _mm_mul_ps(dpv, tdm));
            sv = _mm_mul_ps(sv, rsc_v);
            xev = _mm_add_ps(xev, sv);

            mpv = mmo!(q_idx);
            dpv = dmo!(q_idx);
            ipv = imo!(q_idx);

            mmo!(q_idx) = sv;
            dmo!(q_idx) = dcv;

            dcv = _mm_mul_ps(sv, tmd);

            // I(i,k) = (Mprev*MI + Iprev*II)  (parser mode: insert emission = 1.0)
            imo!(q_idx) = _mm_add_ps(_mm_mul_ps(mpv, tmi), _mm_mul_ps(ipv, tii));
        }

        // D->D wing unfolding
        {
            let dd_offset = 7 * q_count;
            dcv = rightshift_float(dcv);
            dmo!(0) = zerov;
            for q_idx in 0..q_count {
                let tdd = _mm_loadu_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                dmo!(q_idx) = _mm_add_ps(dmo!(q_idx), dcv);
                dcv = _mm_mul_ps(dmo!(q_idx), tdd);
            }
            if om.m < 100 {
                for _ in 0..3 {
                    dcv = rightshift_float(dcv);
                    for q_idx in 0..q_count {
                        let tdd = _mm_loadu_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                        dmo!(q_idx) = _mm_add_ps(dmo!(q_idx), dcv);
                        dcv = _mm_mul_ps(dcv, tdd);
                    }
                }
            } else {
                for _ in 0..3 {
                    dcv = rightshift_float(dcv);
                    let mut cv = zerov;
                    for q_idx in 0..q_count {
                        let tdd = _mm_loadu_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                        let sv = _mm_add_ps(dmo!(q_idx), dcv);
                        cv = _mm_or_ps(cv, _mm_cmpgt_ps(sv, dmo!(q_idx)));
                        dmo!(q_idx) = sv;
                        dcv = _mm_mul_ps(dcv, tdd);
                    }
                    if _mm_movemask_ps(cv) == 0 {
                        break;
                    }
                }
            }
        }

        // E state = sum(M) + sum(D)
        for q_idx in 0..q_count {
            xev = _mm_add_ps(dmo!(q_idx), xev);
        }
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xev, xev),
        );
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xev, xev),
        );
        _mm_store_ss(&mut xe, xev);

        // Special states
        xn *= om.xf[P7O_N][P7O_LOOP];
        xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
        xb = xj * om.xf[P7O_J][P7O_MOVE] + xn * om.xf[P7O_N][P7O_MOVE];

        // Sparse rescaling
        if xe > 1.0e4 {
            let inv_xe = 1.0 / xe;
            let scalev = _mm_set1_ps(inv_xe);
            xn *= inv_xe;
            xc *= inv_xe;
            xj *= inv_xe;
            xb *= inv_xe;
            for q_idx in 0..(q_count * nscells) {
                let p = dp_ptr.add(q_idx);
                *p = _mm_mul_ps(*p, scalev);
            }
            totscale += (xe as f64).ln();
            score_scale += xe.ln();
            xe = 1.0;
        }

        // Store full DP row if requested (for posterior decoding / null2)
        if pmx.has_dp {
            pmx.write_simd_row(&dp, q_count, om.m, i);
        }

        pmx.set_xmx(i, PXE, xe);
        pmx.set_xmx(i, PXN, xn);
        pmx.set_xmx(i, PXJ, xj);
        pmx.set_xmx(i, PXB, xb);
        pmx.set_xmx(i, PXC, xc);
        pmx.scale[i] = totscale;
    }

    let score = if xc.is_nan() || (l > 0 && xc == 0.0) || xc.is_infinite() {
        f32::NEG_INFINITY
    } else {
        score_scale + (xc * om.xf[P7O_C][P7O_MOVE]).ln()
    };
    score
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn forward_parser_pmx_offset_direct(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
) -> f32 {
    use super::probmx::*;

    let q_count = nqf(om.m);
    let row_width = pmx.striped_row_width();
    let zerov = _mm_setzero_ps();
    let dsq_ptr = dsq.as_ptr().add(dsq_offset);
    let striped_ptr = pmx.striped_dp.as_mut_ptr();
    let xmx_ptr = pmx.xmx.as_mut_ptr();
    let scale_ptr = pmx.scale.as_mut_ptr();
    let rfv_ptr = om.rfv.as_ptr();
    let tfv_ptr = om.tfv.as_ptr();

    let mut xe: f32 = 0.0;
    let mut xn: f32 = 1.0;
    let mut xj: f32 = 0.0;
    let mut xb: f32 = om.xf[P7O_N][P7O_MOVE];
    let mut xc: f32 = 0.0;
    let mut totscale: f64 = 0.0;
    let mut score_scale: f32 = 0.0;

    #[inline(always)]
    unsafe fn load_cell(row: *const f32, q: usize, s: usize) -> __m128 {
        _mm_loadu_ps(row.add(q * 12 + s * 4))
    }

    #[inline(always)]
    unsafe fn store_cell(row: *mut f32, q: usize, s: usize, v: __m128) {
        _mm_storeu_ps(row.add(q * 12 + s * 4), v);
    }

    #[inline(always)]
    unsafe fn store_xmx(
        xmx: *mut f32,
        i: usize,
        xe: f32,
        xn: f32,
        xj: f32,
        xb: f32,
        xc: f32,
    ) {
        let row = xmx.add(i * 5);
        *row.add(PXE) = xe;
        *row.add(PXN) = xn;
        *row.add(PXJ) = xj;
        *row.add(PXB) = xb;
        *row.add(PXC) = xc;
    }

    store_xmx(xmx_ptr, 0, 0.0, xn, 0.0, xb, 0.0);
    *scale_ptr = 0.0;

    for i in 1..=l {
        let prev_row = striped_ptr.add((i - 1) * row_width) as *const f32;
        let curr_row = striped_ptr.add(i * row_width);
        let xi = *dsq_ptr.add(i) as usize;
        let rsc_ptr = (*rfv_ptr.add(xi)).as_ptr();
        let xbv = _mm_set1_ps(xb);
        let mut dcv = zerov;
        let mut xev = zerov;

        let mut mpv = rightshift_float(load_cell(prev_row, q_count - 1, 0));
        let mut dpv = rightshift_float(load_cell(prev_row, q_count - 1, 1));
        let mut ipv = rightshift_float(load_cell(prev_row, q_count - 1, 2));

        for q_idx in 0..q_count {
            let tsc_base = q_idx * 7;
            let rsc_v = _mm_loadu_ps((*rsc_ptr.add(q_idx)).as_ptr());

            let tbm = _mm_loadu_ps((*tfv_ptr.add(tsc_base)).as_ptr());
            let tmm = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 1)).as_ptr());
            let tim = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 2)).as_ptr());
            let tdm = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 3)).as_ptr());
            let tmd = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 4)).as_ptr());
            let tmi = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 5)).as_ptr());
            let tii = _mm_loadu_ps((*tfv_ptr.add(tsc_base + 6)).as_ptr());

            let mut sv = _mm_mul_ps(xbv, tbm);
            sv = _mm_add_ps(sv, _mm_mul_ps(mpv, tmm));
            sv = _mm_add_ps(sv, _mm_mul_ps(ipv, tim));
            sv = _mm_add_ps(sv, _mm_mul_ps(dpv, tdm));
            sv = _mm_mul_ps(sv, rsc_v);
            xev = _mm_add_ps(xev, sv);

            mpv = load_cell(prev_row, q_idx, 0);
            dpv = load_cell(prev_row, q_idx, 1);
            ipv = load_cell(prev_row, q_idx, 2);

            store_cell(curr_row, q_idx, 0, sv);
            store_cell(curr_row, q_idx, 1, dcv);

            dcv = _mm_mul_ps(sv, tmd);
            store_cell(
                curr_row,
                q_idx,
                2,
                _mm_add_ps(_mm_mul_ps(mpv, tmi), _mm_mul_ps(ipv, tii)),
            );
        }

        {
            let dd_offset = 7 * q_count;
            dcv = rightshift_float(dcv);
            store_cell(curr_row, 0, 1, zerov);
            for q_idx in 0..q_count {
                let tdd = _mm_loadu_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                let d = _mm_add_ps(load_cell(curr_row, q_idx, 1), dcv);
                store_cell(curr_row, q_idx, 1, d);
                dcv = _mm_mul_ps(d, tdd);
            }
            if om.m < 100 {
                for _ in 0..3 {
                    dcv = rightshift_float(dcv);
                    for q_idx in 0..q_count {
                        let tdd = _mm_loadu_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                        let d = _mm_add_ps(load_cell(curr_row, q_idx, 1), dcv);
                        store_cell(curr_row, q_idx, 1, d);
                        dcv = _mm_mul_ps(dcv, tdd);
                    }
                }
            } else {
                for _ in 0..3 {
                    dcv = rightshift_float(dcv);
                    let mut cv = zerov;
                    for q_idx in 0..q_count {
                        let tdd = _mm_loadu_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                        let old_d = load_cell(curr_row, q_idx, 1);
                        let d = _mm_add_ps(old_d, dcv);
                        cv = _mm_or_ps(cv, _mm_cmpgt_ps(d, old_d));
                        store_cell(curr_row, q_idx, 1, d);
                        dcv = _mm_mul_ps(dcv, tdd);
                    }
                    if _mm_movemask_ps(cv) == 0 {
                        break;
                    }
                }
            }
        }

        for q_idx in 0..q_count {
            xev = _mm_add_ps(load_cell(curr_row, q_idx, 1), xev);
        }
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xev, xev),
        );
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xev, xev),
        );
        _mm_store_ss(&mut xe, xev);

        xn *= om.xf[P7O_N][P7O_LOOP];
        xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
        xb = xj * om.xf[P7O_J][P7O_MOVE] + xn * om.xf[P7O_N][P7O_MOVE];

        if xe > 1.0e4 {
            let inv_xe = 1.0 / xe;
            let scalev = _mm_set1_ps(inv_xe);
            xn *= inv_xe;
            xc *= inv_xe;
            xj *= inv_xe;
            xb *= inv_xe;
            let mut off = 0;
            while off < row_width {
                let p = curr_row.add(off);
                _mm_storeu_ps(p, _mm_mul_ps(_mm_loadu_ps(p), scalev));
                off += 4;
            }
            totscale += (xe as f64).ln();
            score_scale += xe.ln();
            xe = 1.0;
        }

        store_xmx(xmx_ptr, i, xe, xn, xj, xb, xc);
        *scale_ptr.add(i) = totscale;
    }

    if xc.is_nan() || (l > 0 && xc == 0.0) || xc.is_infinite() {
        f32::NEG_INFINITY
    } else {
        score_scale + (xc * om.xf[P7O_C][P7O_MOVE]).ln()
    }
}

#[inline(always)]
fn canonical_run(dsq: &[Dsq], dsq_offset: usize, l: usize, abc_kp: usize) -> bool {
    if l == 0 {
        return false;
    }
    let Some(end) = dsq_offset.checked_add(l) else {
        return false;
    };
    if end >= dsq.len() {
        return false;
    }
    unsafe {
        let ptr = dsq.as_ptr().add(dsq_offset);
        for i in 1..=l {
            if *ptr.add(i) as usize >= abc_kp {
                return false;
            }
        }
    }
    true
}

/// Per-position Forward special states plus cumulative scale.
#[derive(Clone, Copy, Default)]
pub struct FwdSpecials {
    pub xn: f32,
    pub xj: f32,
    pub xc: f32,
    pub xb: f32,
    pub xe: f32,
    pub totscale: f32, // cumulative log-scale at this position
}

/// SSE Forward parser that also saves per-position special states for domain decoding.
/// Returns (score, specials_vec).
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser_with_specials(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
) -> (f32, Vec<FwdSpecials>) {
    let q_count = nqf(om.m);
    let nscells = 3;
    let mut dp: Vec<__m128> = vec![_mm_setzero_ps(); q_count * nscells];
    let zerov = _mm_setzero_ps();

    let mut xe: f32 = 0.0;
    let mut xn: f32 = 1.0;
    let mut xj: f32 = 0.0;
    let mut xb: f32 = om.xf[P7O_N][P7O_MOVE];
    let mut xc: f32 = 0.0;
    let mut totscale: f32 = 0.0;

    macro_rules! mmo {
        ($q:expr) => {
            dp[$q * nscells + 0]
        };
    }
    macro_rules! dmo {
        ($q:expr) => {
            dp[$q * nscells + 1]
        };
    }
    macro_rules! imo {
        ($q:expr) => {
            dp[$q * nscells + 2]
        };
    }

    let mut specials = vec![FwdSpecials::default(); l + 1];
    specials[0] = FwdSpecials {
        xn,
        xj,
        xc,
        xb,
        xe,
        totscale,
    };

    for i in 1..=l {
        let xi = dsq[i] as usize;
        if xi >= om.abc_kp {
            specials[i] = FwdSpecials {
                xn,
                xj,
                xc,
                xb,
                xe,
                totscale,
            };
            continue;
        }

        let rsc = &om.rfv[xi];
        let xbv = _mm_set1_ps(xb);
        let mut dcv = zerov;
        let mut xev = zerov;

        // Shift previous M row right by 1 float
        let mut mpv = rightshift_float(mmo!(q_count - 1));

        for q in 0..q_count {
            let tsc_base = q * 7;
            let rsc_v = _mm_loadu_ps(rsc[q].as_ptr());

            let tbm = _mm_loadu_ps(om.tfv[tsc_base].as_ptr());
            let tmm = _mm_loadu_ps(om.tfv[tsc_base + 1].as_ptr());
            let tim = _mm_loadu_ps(om.tfv[tsc_base + 2].as_ptr());
            let tdm = _mm_loadu_ps(om.tfv[tsc_base + 3].as_ptr());
            let tmd = _mm_loadu_ps(om.tfv[tsc_base + 4].as_ptr());
            let tmi = _mm_loadu_ps(om.tfv[tsc_base + 5].as_ptr());
            let tii = _mm_loadu_ps(om.tfv[tsc_base + 6].as_ptr());

            // M(i,k) = (M(i-1,k-1)*tMM + I(i-1,k-1)*tIM + D(i-1,k-1)*tDM + B*tBM) * e_M(k,xi)
            let sv = _mm_mul_ps(
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(mpv, tmm), _mm_mul_ps(imo!(q), tim)),
                    _mm_add_ps(_mm_mul_ps(dmo!(q), tdm), _mm_mul_ps(xbv, tbm)),
                ),
                rsc_v,
            );

            xev = _mm_add_ps(xev, sv);

            // I(i,k) = (M(i-1,k)*tMI + I(i-1,k)*tII) * e_I(k,xi)
            // Parser mode: insert emissions = 1.0, so no multiply
            imo!(q) = _mm_add_ps(_mm_mul_ps(mmo!(q), tmi), _mm_mul_ps(imo!(q), tii));

            mpv = mmo!(q);
            mmo!(q) = sv;

            // D(i,k) = M(i,k-1)*tMD + D(i,k-1)*tDD
            dcv = _mm_add_ps(
                _mm_mul_ps(sv, tmd),
                _mm_mul_ps(dcv, {
                    let dd_offset = 7 * q_count;
                    _mm_loadu_ps(om.tfv[dd_offset + q].as_ptr())
                }),
            );
            dmo!(q) = dcv;
        }

        // DD wing unfolding
        {
            let last_val = dmo!(q_count - 1);
            let last_f = {
                let mut tmp = [0.0f32; 4];
                _mm_storeu_ps(tmp.as_mut_ptr(), last_val);
                tmp[3]
            };
            if last_f > 0.0 {
                for _iter in 0..q_count {
                    dcv = rightshift_float(dcv);
                    for q in 0..q_count {
                        let dd_offset = 7 * q_count;
                        let tdd = _mm_loadu_ps(om.tfv[dd_offset + q].as_ptr());
                        dcv = _mm_mul_ps(dcv, tdd);
                        dmo!(q) = _mm_add_ps(dmo!(q), dcv);
                        xev = _mm_add_ps(xev, dcv);
                    }
                    let cv = _mm_cmpgt_ps(dcv, zerov);
                    if _mm_movemask_ps(cv) == 0 {
                        break;
                    }
                }
            }
        }

        // Add D's to xEv
        for q in 0..q_count {
            xev = _mm_add_ps(dmo!(q), xev);
        }

        // Horizontal sum
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xev, xev),
        );
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xev, xev),
        );
        _mm_store_ss(&mut xe, xev);

        // Special states
        xn *= om.xf[P7O_N][P7O_LOOP];
        xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
        xb = xj * om.xf[P7O_J][P7O_MOVE] + xn * om.xf[P7O_N][P7O_MOVE];

        // Rescaling
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

        specials[i] = FwdSpecials {
            xn,
            xj,
            xc,
            xb,
            xe,
            totscale,
        };
    }

    let score = if xc.is_nan() || (l > 0 && xc == 0.0) || xc.is_infinite() {
        f32::NEG_INFINITY
    } else {
        totscale + (xc * om.xf[P7O_C][P7O_MOVE]).ln()
    };

    (score, specials)
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
}
