//! SSE-optimized Backward parser (float precision, probability space).
//! Adapted from hmmer-pure-rs bck_engine() which closely follows C HMMER's
//! backward_engine() in impl_sse/fwdback.c.

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
    let mut pmx = super::probmx::ProbMx::new(l);
    backward_parser_pmx(dsq, l, om, fwd_sc, &mut pmx)
}

/// SSE Backward parser that stores per-position specials and scale into a ProbMx.
/// Adapted from the proven hmmer-pure-rs bck_engine() implementation.
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn backward_parser_pmx(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    _fwd_sc: f32,
    pmx: &mut super::probmx::ProbMx,
) -> f32 {
    use super::probmx::*;

    let q = (om.m + 3) / 4; // nqf
    let zerov = _mm_setzero_ps();

    // Two-row rolling buffer: dpp = previous (i+1), dpc = current (i)
    let row_len = q * 3; // M, D, I per stripe
    let mut dpp_buf: Vec<__m128> = vec![zerov; row_len];
    let mut dpc_buf: Vec<__m128> = vec![zerov; row_len];

    // Special state init at position L
    let c_move = om.xf[P7O_C][P7O_MOVE];
    let c_loop = om.xf[P7O_C][P7O_LOOP];
    let e_move = om.xf[P7O_E][P7O_MOVE]; // E->C
    let e_loop = om.xf[P7O_E][P7O_LOOP]; // E->J
    let j_move = om.xf[P7O_J][P7O_MOVE]; // J->B
    let j_loop = om.xf[P7O_J][P7O_LOOP];
    let n_move = om.xf[P7O_N][P7O_MOVE]; // N->B
    let n_loop = om.xf[P7O_N][P7O_LOOP];

    let mut x_c: f32 = c_move; // C->T at position L
    let mut x_e: f32 = x_c * e_move;
    let mut x_j: f32 = 0.0;
    let mut x_b: f32 = 0.0;
    let mut x_n: f32 = 0.0;
    let mut totscale: f64 = 0.0;

    // Initialize row L: M(L,k)->E->C->T and D(L,k)->E->C->T
    {
        let dp = dpp_buf.as_mut_ptr();
        let x_ev = _mm_set1_ps(x_e);
        for qi in 0..q {
            *dp.add(qi * 3) = x_ev;     // M
            *dp.add(qi * 3 + 1) = x_ev; // D
            *dp.add(qi * 3 + 2) = zerov; // I
        }

        // D->D wing unfolding at row L (right to left in striped layout)
        {
            let mut dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(
                _mm_castps_si128(*dp.add((q - 1) * 3 + 1)),
            ));
            for qi in (0..q).rev() {
                let tdd = load_tfv_dd(om, qi);
                dcv = _mm_mul_ps(dcv, tdd);
                let d_ptr = dp.add(qi * 3 + 1);
                *d_ptr = _mm_add_ps(*d_ptr, dcv);
                dcv = *d_ptr;
            }
            for _ in 0..3 {
                dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(dcv)));
                for qi in (0..q).rev() {
                    let tdd = load_tfv_dd(om, qi);
                    dcv = _mm_mul_ps(dcv, tdd);
                    let d_ptr = dp.add(qi * 3 + 1);
                    *d_ptr = _mm_add_ps(*d_ptr, dcv);
                }
            }
        }

        // M->D at row L
        {
            let mut dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(
                _mm_castps_si128(*dp.add(1)),
            ));
            for qi in (0..q).rev() {
                let tmd = load_tfv(om, qi, 4); // MD transition
                let m_ptr = dp.add(qi * 3);
                *m_ptr = _mm_add_ps(*m_ptr, _mm_mul_ps(dcv, tmd));
                dcv = *dp.add(qi * 3 + 1);
            }
        }
    }

    // Store row L specials
    pmx.set_xmx(l, PXE, x_e);
    pmx.set_xmx(l, PXN, 0.0);
    pmx.set_xmx(l, PXJ, 0.0);
    pmx.set_xmx(l, PXB, 0.0);
    pmx.set_xmx(l, PXC, x_c);
    pmx.scale[l] = 0.0;

    // Main recursion: i = L-1 down to 1
    for i in (1..l).rev() {
        let xi_next = dsq[i + 1] as usize;
        if xi_next >= om.abc_kp {
            // Non-canonical residue: copy previous row, update specials
            for v in dpc_buf.iter_mut() { *v = zerov; }
            x_c *= c_loop;
            x_j = x_b * j_move + x_j * j_loop;
            x_n = x_b * n_move + x_n * n_loop;
            x_e = x_c * e_move + x_j * e_loop;
            pmx.set_xmx(i, PXE, x_e);
            pmx.set_xmx(i, PXN, x_n);
            pmx.set_xmx(i, PXJ, x_j);
            pmx.set_xmx(i, PXB, x_b);
            pmx.set_xmx(i, PXC, x_c);
            pmx.scale[i] = totscale;
            std::mem::swap(&mut dpp_buf, &mut dpc_buf);
            continue;
        }

        let dpp = dpp_buf.as_ptr();
        let dpc = dpc_buf.as_mut_ptr();

        // Phase 1: Compute M(i,k) and I(i,k) from row i+1
        // In backward: M(i,k) needs M(i+1,k+1)*emission*tMM + I(i+1,k)*tMI
        // k+1 in striped layout = right-shift
        // Initialize mpv = M(i+1, last_stripe+1) * emission, right-shifted
        let mpv_init = _mm_mul_ps(
            *dpp,
            load_rfv(om, xi_next, 0),
        );
        let mut mpv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(mpv_init)));

        let mut x_bv = zerov;

        for qi in (0..q).rev() {
            let ipv = *dpp.add(qi * 3 + 2); // I(i+1, k)

            // B->M contribution to xB
            let tbm = load_tfv(om, qi, 0);
            x_bv = _mm_add_ps(x_bv, _mm_mul_ps(mpv, tbm));

            // M(i,k) = mpv*tMM + ipv*tIM (partial — E contribution added later)
            let tmm = load_tfv(om, qi, 1);
            let tim = load_tfv(om, qi, 2);
            *dpc.add(qi * 3) = _mm_add_ps(_mm_mul_ps(mpv, tmm), _mm_mul_ps(ipv, tim));

            // D(i,k) = mpv*tDM (partial — D->D and E added later)
            let tdm = load_tfv(om, qi, 3);
            *dpc.add(qi * 3 + 1) = _mm_mul_ps(mpv, tdm);

            // I(i,k) = mpv*tMI + ipv*tII
            let tmi = load_tfv(om, qi, 5);
            let tii = load_tfv(om, qi, 6);
            *dpc.add(qi * 3 + 2) = _mm_add_ps(
                _mm_mul_ps(mpv, tmi),
                _mm_mul_ps(ipv, tii),
            );

            // Next mpv: M(i+1,k) * emission(k, x_{i+1})
            mpv = _mm_mul_ps(
                *dpp.add(qi * 3),
                load_rfv(om, xi_next, qi),
            );
        }

        // Horizontal sum xBv -> xB
        x_bv = _mm_add_ps(x_bv, _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(x_bv, x_bv));
        x_bv = _mm_add_ps(x_bv, _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(x_bv, x_bv));
        _mm_store_ss(&mut x_b, x_bv);

        // Phase 2: Special states
        x_c *= c_loop;
        x_j = x_b * j_move + x_j * j_loop;
        x_n = x_b * n_move + x_n * n_loop;
        x_e = x_c * e_move + x_j * e_loop;
        let x_ev = _mm_set1_ps(x_e);

        // Phase 3: Add E->M,D paths + D->D wing unfolding
        {
            let mut dpv = _mm_add_ps(*dpc.add(1), x_ev);
            dpv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(dpv)));
            for qi in (0..q).rev() {
                let tdd = load_tfv_dd(om, qi);
                let dcv = _mm_mul_ps(dpv, tdd);
                let d_ptr = dpc.add(qi * 3 + 1);
                *d_ptr = _mm_add_ps(*d_ptr, _mm_add_ps(dcv, x_ev));
                dpv = *d_ptr;
                let m_ptr = dpc.add(qi * 3);
                *m_ptr = _mm_add_ps(*m_ptr, x_ev);
            }

            // 3 more D->D passes for convergence
            let mut dcv = dpv;
            for _ in 0..3 {
                dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(dcv)));
                for qi in (0..q).rev() {
                    let tdd = load_tfv_dd(om, qi);
                    dcv = _mm_mul_ps(dcv, tdd);
                    let d_ptr = dpc.add(qi * 3 + 1);
                    *d_ptr = _mm_add_ps(*d_ptr, dcv);
                }
            }
        }

        // Phase 4: M->D paths
        {
            let mut dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(
                _mm_castps_si128(*dpc.add(1)),
            ));
            for qi in (0..q).rev() {
                let tmd = load_tfv(om, qi, 4);
                let m_ptr = dpc.add(qi * 3);
                *m_ptr = _mm_add_ps(*m_ptr, _mm_mul_ps(dcv, tmd));
                dcv = *dpc.add(qi * 3 + 1);
            }
        }

        // Sparse rescaling
        if x_b > 1.0e4 {
            let inv_xb = 1.0 / x_b;
            let scalev = _mm_set1_ps(inv_xb);
            x_e *= inv_xb;
            x_n *= inv_xb;
            x_j *= inv_xb;
            x_c *= inv_xb;
            for qi in 0..row_len {
                let p = dpc.add(qi);
                *p = _mm_mul_ps(*p, scalev);
            }
            totscale += (1.0 / inv_xb as f64).ln();
            x_b = 1.0;
        }

        // Store specials
        pmx.set_xmx(i, PXE, x_e);
        pmx.set_xmx(i, PXN, x_n);
        pmx.set_xmx(i, PXJ, x_j);
        pmx.set_xmx(i, PXB, x_b);
        pmx.set_xmx(i, PXC, x_c);
        pmx.scale[i] = totscale;

        std::mem::swap(&mut dpp_buf, &mut dpc_buf);
    }

    // Row 0 termination
    {
        let dp = dpp_buf.as_ptr();
        let xi1 = dsq[1] as usize;
        if xi1 < om.abc_kp {
            let mut x_bv = zerov;
            for qi in 0..q {
                let tbm = load_tfv(om, qi, 0);
                let mpv = _mm_mul_ps(*dp.add(qi * 3), load_rfv(om, xi1, qi));
                x_bv = _mm_add_ps(x_bv, _mm_mul_ps(mpv, tbm));
            }
            x_bv = _mm_add_ps(x_bv, _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(x_bv, x_bv));
            x_bv = _mm_add_ps(x_bv, _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(x_bv, x_bv));
            _mm_store_ss(&mut x_b, x_bv);
        }
        x_n = x_b * n_move + x_n * n_loop;
    }

    pmx.set_xmx(0, PXE, 0.0);
    pmx.set_xmx(0, PXN, x_n);
    pmx.set_xmx(0, PXJ, 0.0);
    pmx.set_xmx(0, PXB, x_b);
    pmx.set_xmx(0, PXC, 0.0);
    pmx.scale[0] = totscale;

    (totscale + (x_n as f64).ln()) as f32
}

/// Load transition vector for stripe qi, transition index tidx (0=BM,1=MM,2=IM,3=DM,4=MD,5=MI,6=II)
#[inline(always)]
unsafe fn load_tfv(om: &OProfile, qi: usize, tidx: usize) -> __m128 {
    _mm_loadu_ps(om.tfv[qi * 7 + tidx].as_ptr())
}

/// Load D->D transition vector for stripe qi
#[inline(always)]
unsafe fn load_tfv_dd(om: &OProfile, qi: usize) -> __m128 {
    let q = (om.m + 3) / 4;
    _mm_loadu_ps(om.tfv[7 * q + qi].as_ptr())
}

/// Load float emission vector for residue x, stripe qi
#[inline(always)]
unsafe fn load_rfv(om: &OProfile, x: usize, qi: usize) -> __m128 {
    _mm_loadu_ps(om.rfv[x][qi].as_ptr())
}
