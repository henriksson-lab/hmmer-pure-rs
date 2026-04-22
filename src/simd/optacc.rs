//! SSE optimal-accuracy DP over striped posterior matrices.
//! Direct port of HMMER impl_sse/optacc.c p7_OptimalAccuracy() for the
//! coordinate-only domain path.

use std::arch::x86_64::*;

use crate::simd::oprofile::*;
use crate::simd::probmx::{ProbMx, PXB, PXC, PXE, PXJ, PXN};
use crate::trace::{State, Trace};

const NXCELLS: usize = 5;
const DP_CELLS_PER_K: usize = 3; // striped order: M, D, I
const OM_M: usize = 0;
const OM_D: usize = 1;
const OM_I: usize = 2;

#[inline(always)]
unsafe fn load_cell(row: *const f32, qi: usize, state: usize) -> __m128 {
    _mm_load_ps(row.add(qi * DP_CELLS_PER_K * 4 + state * 4))
}

#[inline(always)]
unsafe fn store_cell(row: *mut f32, qi: usize, state: usize, v: __m128) {
    _mm_store_ps(row.add(qi * DP_CELLS_PER_K * 4 + state * 4), v);
}

#[inline(always)]
unsafe fn rightshift_ps(v: __m128, fill: __m128) -> __m128 {
    let shifted = _mm_shuffle_ps::<{ crate::simd::shuffle_mask(2, 1, 0, 0) }>(v, v);
    _mm_move_ss(shifted, fill)
}

#[inline(always)]
unsafe fn hmax_ps(v: __m128) -> f32 {
    let h1 = _mm_max_ps(
        v,
        _mm_shuffle_ps::<{ crate::simd::shuffle_mask(0, 3, 2, 1) }>(v, v),
    );
    let h2 = _mm_max_ps(
        h1,
        _mm_shuffle_ps::<{ crate::simd::shuffle_mask(1, 0, 3, 2) }>(h1, h1),
    );
    let mut out = 0.0_f32;
    _mm_store_ss(&mut out, h2);
    out
}

/// Convert striped Forward/Backward probability matrices into a striped
/// posterior matrix equivalent to C's `p7_Decoding()` output.
///
/// # Safety
/// Requires SSE2 support and full striped matrices in `fwd`, `bck`, and `pp`.
#[target_feature(enable = "sse2")]
pub unsafe fn posterior_decoding_pmx(fwd: &ProbMx, bck: &ProbMx, om: &OProfile, pp: &mut ProbMx) {
    let l = fwd.l;
    let m = om.m;
    pp.resize_full(m, l);

    let q = fwd.q_count();
    let row_width = fwd.striped_row_width();
    let fdp = fwd.striped_dp.as_ptr().add(fwd.striped_dp_offset);
    let bdp = bck.striped_dp.as_ptr().add(bck.striped_dp_offset);
    let pdp = pp.striped_dp.as_mut_ptr().add(pp.striped_dp_offset);
    let fx = fwd.xmx.as_ptr();
    let bx = bck.xmx.as_ptr();
    let px = pp.xmx.as_mut_ptr();

    let inv_total = if *bx.add(PXN) > 0.0 {
        1.0 / *bx.add(PXN)
    } else {
        0.0
    };
    pp.m = m;
    pp.l = l;

    // Row 0 is zero posterior probability.
    for qi in 0..q {
        store_cell(pdp, qi, OM_M, _mm_setzero_ps());
        store_cell(pdp, qi, OM_D, _mm_setzero_ps());
        store_cell(pdp, qi, OM_I, _mm_setzero_ps());
    }
    for s in 0..NXCELLS {
        *px.add(s) = 0.0;
    }

    let mut scaleproduct = inv_total;
    for i in 1..=l {
        let dp_scale = scaleproduct * *fwd.row_scale.as_ptr().add(i);
        let scalev = _mm_set1_ps(dp_scale);
        let frow = fdp.add(i * row_width);
        let brow = bdp.add(i * row_width);
        let prow = pdp.add(i * row_width);

        for qi in 0..q {
            let fm = load_cell(frow, qi, OM_M);
            let bm = load_cell(brow, qi, OM_M);
            let pm = _mm_mul_ps(_mm_mul_ps(fm, bm), scalev);
            store_cell(prow, qi, OM_M, pm);

            // C's p7_Decoding stores D posteriors as zero; D is not a residue assignment.
            store_cell(prow, qi, OM_D, _mm_setzero_ps());

            let fi = load_cell(frow, qi, OM_I);
            let bi = load_cell(brow, qi, OM_I);
            let pi = _mm_mul_ps(_mm_mul_ps(fi, bi), scalev);
            store_cell(prow, qi, OM_I, pi);
        }

        // Node M has no insert state. The padded lane values are irrelevant to C
        // traceback but keeping I_M zero matches the generic matrix semantics.
        let m_q = (m - 1) % q;
        let m_lane = (m - 1) / q;
        let mut lanes = [0.0_f32; 4];
        _mm_storeu_ps(lanes.as_mut_ptr(), load_cell(prow, m_q, OM_I));
        lanes[m_lane] = 0.0;
        store_cell(prow, m_q, OM_I, _mm_loadu_ps(lanes.as_ptr()));

        let xrow = i * NXCELLS;
        *px.add(xrow + PXE) = 0.0;
        *px.add(xrow + PXN) = *fx.add((i - 1) * NXCELLS + PXN)
            * *bx.add(i * NXCELLS + PXN)
            * om.xf[P7O_N][P7O_LOOP]
            * scaleproduct;
        *px.add(xrow + PXJ) = *fx.add((i - 1) * NXCELLS + PXJ)
            * *bx.add(i * NXCELLS + PXJ)
            * om.xf[P7O_J][P7O_LOOP]
            * scaleproduct;
        *px.add(xrow + PXB) = 0.0;
        *px.add(xrow + PXC) = *fx.add((i - 1) * NXCELLS + PXC)
            * *bx.add(i * NXCELLS + PXC)
            * om.xf[P7O_C][P7O_LOOP]
            * scaleproduct;

        if bck.has_own_scales {
            scaleproduct *= *fwd.row_scale.as_ptr().add(i) / *bck.row_scale.as_ptr().add(i);
        }
    }
}

/// SSE optimal-accuracy DP fill over a striped posterior matrix.
///
/// # Safety
/// Requires SSE2 support and full striped matrices in `pp` and `ox`.
#[target_feature(enable = "sse2")]
pub unsafe fn optimal_accuracy_pmx(om: &OProfile, pp: &ProbMx, ox: &mut ProbMx) -> f32 {
    let m = om.m;
    let l = pp.l;
    ox.resize_full(m, l);

    let q = pp.q_count();
    let row_width = pp.striped_row_width();
    let ppdp = pp.striped_dp.as_ptr().add(pp.striped_dp_offset);
    let pdp_x = pp.xmx.as_ptr();
    let oxdp = ox.striped_dp.as_mut_ptr().add(ox.striped_dp_offset);
    let oxx = ox.xmx.as_mut_ptr();
    let tfv = om.tfv_a.as_ptr();
    let dd_tfv = tfv.add(7 * q);
    let zerov = _mm_setzero_ps();
    let infv = _mm_set1_ps(f32::NEG_INFINITY);

    for qi in 0..q {
        store_cell(oxdp, qi, OM_M, infv);
        store_cell(oxdp, qi, OM_D, infv);
        store_cell(oxdp, qi, OM_I, infv);
    }
    *oxx.add(PXE) = f32::NEG_INFINITY;
    *oxx.add(PXN) = 0.0;
    *oxx.add(PXJ) = f32::NEG_INFINITY;
    *oxx.add(PXB) = 0.0;
    *oxx.add(PXC) = f32::NEG_INFINITY;

    for i in 1..=l {
        let dpp = oxdp.add((i - 1) * row_width);
        let dpc = oxdp.add(i * row_width);
        let ppp = ppdp.add(i * row_width);
        let mut dcv = infv;
        let mut xev = infv;
        let xbv = _mm_set1_ps(*oxx.add((i - 1) * NXCELLS + PXB));

        let mut mpv = rightshift_ps(load_cell(dpp, q - 1, OM_M), infv);
        let mut dpv = rightshift_ps(load_cell(dpp, q - 1, OM_D), infv);
        let mut ipv = rightshift_ps(load_cell(dpp, q - 1, OM_I), infv);

        for qi in 0..q {
            let tp = tfv.add(qi * 7);
            let tbm = _mm_load_ps((*tp.add(P7O_BM)).as_ptr());
            let tmm = _mm_load_ps((*tp.add(P7O_MM)).as_ptr());
            let tim = _mm_load_ps((*tp.add(P7O_IM)).as_ptr());
            let tdm = _mm_load_ps((*tp.add(P7O_DM)).as_ptr());
            let tmd = _mm_load_ps((*tp.add(P7O_MD)).as_ptr());
            let tmi = _mm_load_ps((*tp.add(P7O_MI)).as_ptr());
            let tii = _mm_load_ps((*tp.add(P7O_II)).as_ptr());
            let pcell = ppp.add(qi * DP_CELLS_PER_K * 4);
            let dpp_cell = dpp.add(qi * DP_CELLS_PER_K * 4);
            let dpc_cell = dpc.add(qi * DP_CELLS_PER_K * 4);

            let mut sv = _mm_and_ps(_mm_cmpgt_ps(tbm, zerov), xbv);
            sv = _mm_max_ps(sv, _mm_and_ps(_mm_cmpgt_ps(tmm, zerov), mpv));
            sv = _mm_max_ps(sv, _mm_and_ps(_mm_cmpgt_ps(tim, zerov), ipv));
            sv = _mm_max_ps(sv, _mm_and_ps(_mm_cmpgt_ps(tdm, zerov), dpv));
            sv = _mm_add_ps(sv, _mm_load_ps(pcell.add(OM_M * 4)));
            xev = _mm_max_ps(xev, sv);

            mpv = _mm_load_ps(dpp_cell.add(OM_M * 4));
            dpv = _mm_load_ps(dpp_cell.add(OM_D * 4));
            ipv = _mm_load_ps(dpp_cell.add(OM_I * 4));

            _mm_store_ps(dpc_cell.add(OM_M * 4), sv);
            _mm_store_ps(dpc_cell.add(OM_D * 4), dcv);

            dcv = _mm_and_ps(_mm_cmpgt_ps(tmd, zerov), sv);

            sv = _mm_and_ps(_mm_cmpgt_ps(tmi, zerov), mpv);
            sv = _mm_max_ps(sv, _mm_and_ps(_mm_cmpgt_ps(tii, zerov), ipv));
            _mm_store_ps(
                dpc_cell.add(OM_I * 4),
                _mm_add_ps(sv, _mm_load_ps(pcell.add(OM_I * 4))),
            );
        }

        dcv = rightshift_ps(dcv, infv);
        for qi in 0..q {
            let dptr = dpc.add(qi * DP_CELLS_PER_K * 4 + OM_D * 4);
            let d = _mm_max_ps(dcv, _mm_load_ps(dptr));
            _mm_store_ps(dptr, d);
            let tdd = _mm_load_ps((*dd_tfv.add(qi)).as_ptr());
            dcv = _mm_and_ps(_mm_cmpgt_ps(tdd, zerov), d);
        }

        for _ in 1..4 {
            dcv = rightshift_ps(dcv, infv);
            for qi in 0..q {
                let dptr = dpc.add(qi * DP_CELLS_PER_K * 4 + OM_D * 4);
                let d = _mm_max_ps(dcv, _mm_load_ps(dptr));
                _mm_store_ps(dptr, d);
                let tdd = _mm_load_ps((*dd_tfv.add(qi)).as_ptr());
                dcv = _mm_and_ps(_mm_cmpgt_ps(tdd, zerov), dcv);
            }
        }

        for qi in 0..q {
            xev = _mm_max_ps(xev, load_cell(dpc, qi, OM_D));
        }

        let xrow = i * NXCELLS;
        let xprev = xrow - NXCELLS;
        *oxx.add(xrow + PXE) = hmax_ps(xev);

        let t1 = if om.xf[P7O_J][P7O_LOOP] == 0.0 {
            0.0
        } else {
            *oxx.add(xprev + PXJ) + *pdp_x.add(xrow + PXJ)
        };
        let t2 = if om.xf[P7O_E][P7O_LOOP] == 0.0 {
            0.0
        } else {
            *oxx.add(xrow + PXE)
        };
        *oxx.add(xrow + PXJ) = if t1 > t2 { t1 } else { t2 };

        let t1 = if om.xf[P7O_C][P7O_LOOP] == 0.0 {
            0.0
        } else {
            *oxx.add(xprev + PXC) + *pdp_x.add(xrow + PXC)
        };
        let t2 = if om.xf[P7O_E][P7O_MOVE] == 0.0 {
            0.0
        } else {
            *oxx.add(xrow + PXE)
        };
        *oxx.add(xrow + PXC) = if t1 > t2 { t1 } else { t2 };

        *oxx.add(xrow + PXN) = if om.xf[P7O_N][P7O_LOOP] == 0.0 {
            0.0
        } else {
            *oxx.add(xprev + PXN) + *pdp_x.add(xrow + PXN)
        };

        let t1 = if om.xf[P7O_N][P7O_MOVE] == 0.0 {
            0.0
        } else {
            *oxx.add(xrow + PXN)
        };
        let t2 = if om.xf[P7O_J][P7O_MOVE] == 0.0 {
            0.0
        } else {
            *oxx.add(xrow + PXJ)
        };
        *oxx.add(xrow + PXB) = if t1 > t2 { t1 } else { t2 };
    }

    *oxx.add(l * NXCELLS + PXC)
}

/// Trace through a striped optimal-accuracy DP matrix and return only the
/// coordinate span needed by domtblout. This follows `oa_trace_pmx()` state
/// selection but avoids materializing and reversing a full Trace.
///
/// # Safety
/// Requires SSE2 support and full striped matrices in `pp` and `ox`.
#[target_feature(enable = "sse2")]
pub unsafe fn oa_trace_coords_pmx(
    om: &OProfile,
    pp: &ProbMx,
    ox: &ProbMx,
) -> Option<(usize, usize, usize, usize)> {
    let mut i = ox.l;
    let mut k = 0usize;
    let mut sprv = State::C;
    let mut hmmfrom = usize::MAX;
    let mut hmmto = 0usize;
    let mut sqfrom = usize::MAX;
    let mut sqto = 0usize;

    while sprv != State::S {
        let scur = match sprv {
            State::M => {
                let s = select_m(om, ox, i, k);
                k = k.saturating_sub(1);
                i = i.saturating_sub(1);
                s
            }
            State::D => {
                let s = select_d(om, ox, i, k);
                k = k.saturating_sub(1);
                s
            }
            State::I => {
                let s = select_i(om, ox, i, k);
                i = i.saturating_sub(1);
                s
            }
            State::N => {
                if i == 0 {
                    State::S
                } else {
                    State::N
                }
            }
            State::C => select_c(om, pp, ox, i),
            State::J => select_j(om, pp, ox, i),
            State::E => select_e(om, ox, i, &mut k),
            State::B => select_b(om, ox, i),
            _ => State::S,
        };

        match scur {
            State::M => {
                if k > 0 {
                    hmmfrom = hmmfrom.min(k);
                    hmmto = hmmto.max(k);
                }
                if i > 0 {
                    sqfrom = sqfrom.min(i);
                    sqto = sqto.max(i);
                }
            }
            State::D => {
                if k > 0 {
                    hmmfrom = hmmfrom.min(k);
                    hmmto = hmmto.max(k);
                }
            }
            State::I => {
                if i > 0 {
                    sqfrom = sqfrom.min(i);
                    sqto = sqto.max(i);
                }
            }
            _ => {}
        }

        if matches!(scur, State::N | State::J | State::C) && scur == sprv {
            i = i.saturating_sub(1);
        }
        sprv = scur;
    }

    if hmmto > 0 || sqto > 0 {
        Some((
            if hmmfrom == usize::MAX { 0 } else { hmmfrom },
            hmmto,
            if sqfrom == usize::MAX { 0 } else { sqfrom },
            sqto,
        ))
    } else {
        None
    }
}

/// Trace through a striped optimal-accuracy DP matrix. This returns only state,
/// model, and sequence coordinates; posterior-probability annotation is not
/// needed for domtblout coordinate extraction.
///
/// # Safety
/// Requires SSE2 support and full striped matrices in `pp` and `ox`.
#[target_feature(enable = "sse2")]
pub unsafe fn oa_trace_pmx(om: &OProfile, pp: &ProbMx, ox: &ProbMx) -> Trace {
    let mut tr = Trace::new();
    let mut i = ox.l;
    let mut k = 0usize;
    let mut sprv = State::C;

    tr.append(State::T, 0, i);
    tr.append(State::C, 0, i);

    while sprv != State::S {
        let scur = match sprv {
            State::M => {
                let s = select_m(om, ox, i, k);
                k = k.saturating_sub(1);
                i = i.saturating_sub(1);
                s
            }
            State::D => {
                let s = select_d(om, ox, i, k);
                k = k.saturating_sub(1);
                s
            }
            State::I => {
                let s = select_i(om, ox, i, k);
                i = i.saturating_sub(1);
                s
            }
            State::N => {
                if i == 0 {
                    State::S
                } else {
                    State::N
                }
            }
            State::C => select_c(om, pp, ox, i),
            State::J => select_j(om, pp, ox, i),
            State::E => select_e(om, ox, i, &mut k),
            State::B => select_b(om, ox, i),
            _ => State::S,
        };

        tr.append(scur, k, i);
        if matches!(scur, State::N | State::J | State::C) && scur == sprv {
            i = i.saturating_sub(1);
        }
        sprv = scur;
        if tr.n > ox.l + om.m + 100 {
            break;
        }
    }

    tr.st.reverse();
    tr.k.reverse();
    tr.i.reverse();
    tr
}

#[inline(always)]
unsafe fn cell_lane(ox: &ProbMx, i: usize, q: usize, state: usize, lane: usize) -> f32 {
    let row = ox
        .striped_dp
        .as_ptr()
        .add(ox.striped_dp_offset + i * ox.striped_row_width());
    let mut lanes = [0.0_f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), load_cell(row, q, state));
    lanes[lane]
}

#[inline(always)]
unsafe fn tfv_lane(om: &OProfile, q: usize, tidx: usize, lane: usize) -> f32 {
    om.tfv[q * 7 + tidx][lane]
}

#[inline(always)]
unsafe fn tdd_lane(om: &OProfile, q: usize, lane: usize) -> f32 {
    let q_count = om.m.div_ceil(4);
    om.tfv[7 * q_count + q][lane]
}

unsafe fn select_m(om: &OProfile, ox: &ProbMx, i: usize, k: usize) -> State {
    if k == 0 || k > om.m || i == 0 {
        debug_assert!(k >= 1 && k <= om.m && i > 0);
        return State::B;
    }
    let q_count = ox.q_count();
    let q = (k - 1) % q_count;
    let lane = (k - 1) / q_count;
    let x_b = ox.xmx[(i - 1) * NXCELLS + PXB];

    let (mpv, dpv, ipv) = if q > 0 {
        (
            cell_lane(ox, i - 1, q - 1, OM_M, lane),
            cell_lane(ox, i - 1, q - 1, OM_D, lane),
            cell_lane(ox, i - 1, q - 1, OM_I, lane),
        )
    } else if lane == 0 {
        (0.0, 0.0, 0.0)
    } else {
        (
            cell_lane(ox, i - 1, q_count - 1, OM_M, lane - 1),
            cell_lane(ox, i - 1, q_count - 1, OM_D, lane - 1),
            cell_lane(ox, i - 1, q_count - 1, OM_I, lane - 1),
        )
    };

    let paths = [
        if tfv_lane(om, q, P7O_MM, lane) == 0.0 {
            f32::NEG_INFINITY
        } else {
            mpv
        },
        if tfv_lane(om, q, P7O_IM, lane) == 0.0 {
            f32::NEG_INFINITY
        } else {
            ipv
        },
        if tfv_lane(om, q, P7O_DM, lane) == 0.0 {
            f32::NEG_INFINITY
        } else {
            dpv
        },
        if tfv_lane(om, q, P7O_BM, lane) == 0.0 {
            f32::NEG_INFINITY
        } else {
            x_b
        },
    ];
    let states = [State::M, State::I, State::D, State::B];
    states[argmax_first(&paths)]
}

unsafe fn select_d(om: &OProfile, ox: &ProbMx, i: usize, k: usize) -> State {
    if k == 0 || k > om.m {
        debug_assert!(k >= 1 && k <= om.m);
        return State::M;
    }
    let q_count = ox.q_count();
    let q = (k - 1) % q_count;
    let lane = (k - 1) / q_count;
    let (mpv, dpv, tmd, tdd) = if q > 0 {
        (
            cell_lane(ox, i, q - 1, OM_M, lane),
            cell_lane(ox, i, q - 1, OM_D, lane),
            tfv_lane(om, q - 1, P7O_MD, lane),
            tdd_lane(om, q - 1, lane),
        )
    } else if lane == 0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        (
            cell_lane(ox, i, q_count - 1, OM_M, lane - 1),
            cell_lane(ox, i, q_count - 1, OM_D, lane - 1),
            tfv_lane(om, q_count - 1, P7O_MD, lane - 1),
            tdd_lane(om, q_count - 1, lane - 1),
        )
    };
    let md = if tmd == 0.0 { f32::NEG_INFINITY } else { mpv };
    let dd = if tdd == 0.0 { f32::NEG_INFINITY } else { dpv };
    if md >= dd {
        State::M
    } else {
        State::D
    }
}

unsafe fn select_i(om: &OProfile, ox: &ProbMx, i: usize, k: usize) -> State {
    if k == 0 || k > om.m || i == 0 {
        debug_assert!(k >= 1 && k <= om.m && i > 0);
        return State::M;
    }
    let q_count = ox.q_count();
    let q = (k - 1) % q_count;
    let lane = (k - 1) / q_count;
    let mpv = cell_lane(ox, i - 1, q, OM_M, lane);
    let ipv = cell_lane(ox, i - 1, q, OM_I, lane);
    let mi = if tfv_lane(om, q, P7O_MI, lane) == 0.0 {
        f32::NEG_INFINITY
    } else {
        mpv
    };
    let ii = if tfv_lane(om, q, P7O_II, lane) == 0.0 {
        f32::NEG_INFINITY
    } else {
        ipv
    };
    if mi >= ii {
        State::M
    } else {
        State::I
    }
}

#[inline]
fn select_c(om: &OProfile, pp: &ProbMx, ox: &ProbMx, i: usize) -> State {
    let c_loop = if om.xf[P7O_C][P7O_LOOP] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[(i - 1) * NXCELLS + PXC] + pp.xmx[i * NXCELLS + PXC]
    };
    let e_move = if om.xf[P7O_E][P7O_MOVE] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[i * NXCELLS + PXE]
    };
    if c_loop > e_move {
        State::C
    } else {
        State::E
    }
}

#[inline]
fn select_j(om: &OProfile, pp: &ProbMx, ox: &ProbMx, i: usize) -> State {
    let j_loop = if om.xf[P7O_J][P7O_LOOP] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[(i - 1) * NXCELLS + PXJ] + pp.xmx[i * NXCELLS + PXJ]
    };
    let e_loop = if om.xf[P7O_E][P7O_LOOP] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[i * NXCELLS + PXE]
    };
    if j_loop > e_loop {
        State::J
    } else {
        State::E
    }
}

unsafe fn select_e(om: &OProfile, ox: &ProbMx, i: usize, ret_k: &mut usize) -> State {
    // Mirror C hmmer/src/impl_sse/optacc.c:select_e iteration order: for each
    // q, check all 4 M lanes first (M ties beat D via `>=`), then all 4 D
    // lanes (D only wins on strict `>`). Interleaving M/D per lane changes
    // tie-break picks and can shift the chosen k.
    let q_count = ox.q_count();
    let mut max = f32::NEG_INFINITY;
    let mut smax = State::M;
    let mut kmax = 1usize;
    for q in 0..q_count {
        for lane in 0..4 {
            let k = lane * q_count + q + 1;
            if k > om.m {
                continue;
            }
            let m = cell_lane(ox, i, q, OM_M, lane);
            if m >= max {
                max = m;
                smax = State::M;
                kmax = k;
            }
        }
        for lane in 0..4 {
            let k = lane * q_count + q + 1;
            if k > om.m {
                continue;
            }
            let d = cell_lane(ox, i, q, OM_D, lane);
            if d > max {
                max = d;
                smax = State::D;
                kmax = k;
            }
        }
    }
    *ret_k = kmax;
    smax
}

#[inline]
fn select_b(om: &OProfile, ox: &ProbMx, i: usize) -> State {
    let n_move = if om.xf[P7O_N][P7O_MOVE] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[i * NXCELLS + PXN]
    };
    let j_move = if om.xf[P7O_J][P7O_MOVE] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[i * NXCELLS + PXJ]
    };
    if n_move > j_move {
        State::N
    } else {
        State::J
    }
}

#[inline]
fn argmax_first(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = values[0];
    for (idx, &val) in values.iter().enumerate().skip(1) {
        if val > best_val {
            best = idx;
            best_val = val;
        }
    }
    best
}
