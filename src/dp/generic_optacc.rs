//! Generic optimal accuracy alignment.
//! Port of generic_optacc.c — finds alignment maximizing expected correct positions.

use crate::dp::gmx::*;
use crate::profile::*;
use crate::trace::{State, Trace};

pub struct OptAccTDelta {
    td: Vec<[f32; P7P_NTRANS]>,
    x_n_loop: f32,
    x_n_move: f32,
    x_j_loop: f32,
    x_j_move: f32,
    x_c_loop: f32,
    x_e_loop: f32,
    x_e_move: f32,
    esc: f32,
}

impl OptAccTDelta {
    pub fn from_profile(gm: &Profile) -> Self {
        let mut td = vec![[0.0_f32; P7P_NTRANS]; gm.m + 1];
        for (k, row) in td.iter_mut().enumerate().take(gm.m + 1) {
            for (s, cell) in row.iter_mut().enumerate().take(P7P_NTRANS) {
                *cell = tdelta(gm.tsc(k, s));
            }
        }

        OptAccTDelta {
            td,
            x_n_loop: tdelta(gm.xsc[P7P_N][P7P_LOOP]),
            x_n_move: tdelta(gm.xsc[P7P_N][P7P_MOVE]),
            x_j_loop: tdelta(gm.xsc[P7P_J][P7P_LOOP]),
            x_j_move: tdelta(gm.xsc[P7P_J][P7P_MOVE]),
            x_c_loop: tdelta(gm.xsc[P7P_C][P7P_LOOP]),
            x_e_loop: tdelta(gm.xsc[P7P_E][P7P_LOOP]),
            x_e_move: tdelta(gm.xsc[P7P_E][P7P_MOVE]),
            esc: if gm.is_local() { 1.0 } else { 0.0 },
        }
    }
}

/// Run optimal accuracy DP using the posterior probability matrix.
/// Returns the OA score (expected number of correctly aligned residues).
pub fn g_optimal_accuracy(gm: &Profile, pp: &Gmx, ox: &mut Gmx) -> f32 {
    let deltas = OptAccTDelta::from_profile(gm);
    g_optimal_accuracy_with_deltas(gm, pp, ox, &deltas)
}

pub fn g_optimal_accuracy_with_deltas(
    gm: &Profile,
    pp: &Gmx,
    ox: &mut Gmx,
    deltas: &OptAccTDelta,
) -> f32 {
    let l = pp.l;
    let m = gm.m;
    let pp_w = pp.row_width();
    let ox_w = ox.row_width();
    let pp_stride = pp_w * P7G_NSCELLS;
    let ox_stride = ox_w * P7G_NSCELLS;
    let ppdp = pp.dp_mem.as_ptr();
    let ppx = pp.xmx.as_ptr();
    let oxdp = ox.dp_mem.as_mut_ptr();
    let oxx = ox.xmx.as_mut_ptr();
    let td = deltas.td.as_ptr() as *const f32;

    ox.m = m;
    ox.l = l;

    let x_n_loop = deltas.x_n_loop;
    let x_n_move = deltas.x_n_move;
    let x_j_loop = deltas.x_j_loop;
    let x_j_move = deltas.x_j_move;
    let x_c_loop = deltas.x_c_loop;
    let x_e_loop = deltas.x_e_loop;
    let x_e_move = deltas.x_e_move;

    // Initialize row 0
    let esc = deltas.esc;

    unsafe {
        *oxx.add(P7G_N) = 0.0;
        *oxx.add(P7G_B) = 0.0;
        *oxx.add(P7G_E) = f32::NEG_INFINITY;
        *oxx.add(P7G_C) = f32::NEG_INFINITY;
        *oxx.add(P7G_J) = f32::NEG_INFINITY;
        for k in 0..=m {
            let idx = k * P7G_NSCELLS;
            *oxdp.add(idx + P7G_M) = f32::NEG_INFINITY;
            *oxdp.add(idx + P7G_I) = f32::NEG_INFINITY;
            *oxdp.add(idx + P7G_D) = f32::NEG_INFINITY;
        }

        for i in 1..=l {
            let row = i * ox_stride;
            let prev = row - ox_stride;
            let pp_row = i * pp_stride;
            let xrow = i * P7G_NXCELLS;
            let xprev = xrow - P7G_NXCELLS;

            *oxdp.add(row + P7G_M) = f32::NEG_INFINITY;
            *oxdp.add(row + P7G_I) = f32::NEG_INFINITY;
            *oxdp.add(row + P7G_D) = f32::NEG_INFINITY;
            *oxx.add(xrow + P7G_E) = f32::NEG_INFINITY;

            for k in 1..m {
                let k3 = k * P7G_NSCELLS;
                let pk3 = (k - 1) * P7G_NSCELLS;
                let tprev = (k - 1) * P7P_NTRANS;
                let tcur = k * P7P_NTRANS;
                let pp_m = *ppdp.add(pp_row + k3 + P7G_M);

                let sc = cmax(
                    cmax(
                        *td.add(tprev + P7P_MM) * (*oxdp.add(prev + pk3 + P7G_M) + pp_m),
                        *td.add(tprev + P7P_IM) * (*oxdp.add(prev + pk3 + P7G_I) + pp_m),
                    ),
                    cmax(
                        *td.add(tprev + P7P_DM) * (*oxdp.add(prev + pk3 + P7G_D) + pp_m),
                        *td.add(tprev + P7P_BM) * (*oxx.add(xprev + P7G_B) + pp_m),
                    ),
                );
                *oxdp.add(row + k3 + P7G_M) = sc;

                *oxx.add(xrow + P7G_E) = cmax(*oxx.add(xrow + P7G_E), esc * sc);

                let pp_i = *ppdp.add(pp_row + k3 + P7G_I);
                let mi = *td.add(tcur + P7P_MI) * (*oxdp.add(prev + k3 + P7G_M) + pp_i);
                let ii = *td.add(tcur + P7P_II) * (*oxdp.add(prev + k3 + P7G_I) + pp_i);
                *oxdp.add(row + k3 + P7G_I) = cmax(mi, ii);

                let md = *td.add(tprev + P7P_MD) * *oxdp.add(row + pk3 + P7G_M);
                let dd = *td.add(tprev + P7P_DD) * *oxdp.add(row + pk3 + P7G_D);
                *oxdp.add(row + k3 + P7G_D) = cmax(md, dd);
            }

            let m3 = m * P7G_NSCELLS;
            let pm3 = (m - 1) * P7G_NSCELLS;
            let tm = (m - 1) * P7P_NTRANS;
            let pp_m = *ppdp.add(pp_row + m3 + P7G_M);
            let sc = cmax(
                cmax(
                    *td.add(tm + P7P_MM) * (*oxdp.add(prev + pm3 + P7G_M) + pp_m),
                    *td.add(tm + P7P_IM) * (*oxdp.add(prev + pm3 + P7G_I) + pp_m),
                ),
                cmax(
                    *td.add(tm + P7P_DM) * (*oxdp.add(prev + pm3 + P7G_D) + pp_m),
                    *td.add(tm + P7P_BM) * (*oxx.add(xprev + P7G_B) + pp_m),
                ),
            );
            *oxdp.add(row + m3 + P7G_M) = sc;
            *oxdp.add(row + m3 + P7G_I) = f32::NEG_INFINITY;

            let md = *td.add(tm + P7P_MD) * *oxdp.add(row + pm3 + P7G_M);
            let dd = *td.add(tm + P7P_DD) * *oxdp.add(row + pm3 + P7G_D);
            *oxdp.add(row + m3 + P7G_D) = cmax(md, dd);

            *oxx.add(xrow + P7G_E) = cmax(
                *oxx.add(xrow + P7G_E),
                cmax(*oxdp.add(row + m3 + P7G_M), *oxdp.add(row + m3 + P7G_D)),
            );

            let pp_n = *ppx.add(xrow + P7G_N);
            let pp_j = *ppx.add(xrow + P7G_J);
            let pp_c = *ppx.add(xrow + P7G_C);

            let j = cmax(
                x_j_loop * (*oxx.add(xprev + P7G_J) + pp_j),
                x_e_loop * *oxx.add(xrow + P7G_E),
            );
            *oxx.add(xrow + P7G_J) = j;

            let c = cmax(
                x_c_loop * (*oxx.add(xprev + P7G_C) + pp_c),
                x_e_move * *oxx.add(xrow + P7G_E),
            );
            *oxx.add(xrow + P7G_C) = c;

            let n = x_n_loop * (*oxx.add(xprev + P7G_N) + pp_n);
            *oxx.add(xrow + P7G_N) = n;

            let b = cmax(x_n_move * *oxx.add(xrow + P7G_N), x_j_move * *oxx.add(xrow + P7G_J));
            *oxx.add(xrow + P7G_B) = b;
        }

        *oxx.add(l * P7G_NXCELLS + P7G_C)
    }
}

/// Trace back an optimal-accuracy alignment from an OA DP matrix.
/// Port of generic_optacc.c p7_GOATrace().
pub fn g_oa_trace(gm: &Profile, pp: &Gmx, ox: &Gmx) -> Trace {
    let mut tr = Trace::new();
    let mut i = ox.l;
    let mut k = 0usize;
    let mut sprv = State::C;

    tr.append(State::T, 0, i);
    tr.append(State::C, 0, i);

    while sprv != State::S {
        let scur = match sprv {
            State::M => {
                let s = select_m(gm, ox, i, k);
                k = k.saturating_sub(1);
                i = i.saturating_sub(1);
                s
            }
            State::D => {
                let s = select_d(gm, ox, i, k);
                k = k.saturating_sub(1);
                s
            }
            State::I => {
                let s = select_i(gm, ox, i, k);
                i = i.saturating_sub(1);
                s
            }
            State::N => select_n(i),
            State::C => select_c(gm, pp, ox, i),
            State::J => select_j(gm, pp, ox, i),
            State::E => select_e(gm, ox, i, &mut k),
            State::B => select_b(gm, ox, i),
            _ => State::S,
        };

        tr.append(scur, k, i);

        if matches!(scur, State::N | State::J | State::C) && scur == sprv {
            i = i.saturating_sub(1);
        }
        sprv = scur;

        if tr.n > ox.l + gm.m + 100 {
            break;
        }
    }

    tr.st.reverse();
    tr.k.reverse();
    tr.i.reverse();
    tr
}

#[inline]
fn tdelta(tsc: f32) -> f32 {
    if tsc == f32::NEG_INFINITY {
        f32::MIN_POSITIVE
    } else {
        1.0
    }
}

#[inline]
fn cmax(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

#[inline]
fn select_m(gm: &Profile, ox: &Gmx, i: usize, k: usize) -> State {
    let paths = [
        tdelta(gm.tsc(k - 1, P7P_MM)) * ox.mmx(i - 1, k - 1),
        tdelta(gm.tsc(k - 1, P7P_IM)) * ox.imx(i - 1, k - 1),
        tdelta(gm.tsc(k - 1, P7P_DM)) * ox.dmx(i - 1, k - 1),
        tdelta(gm.tsc(k - 1, P7P_BM)) * ox.xmx(i - 1, P7G_B),
    ];
    let states = [State::M, State::I, State::D, State::B];
    states[argmax_first(&paths)]
}

#[inline]
fn select_d(gm: &Profile, ox: &Gmx, i: usize, k: usize) -> State {
    let md = tdelta(gm.tsc(k - 1, P7P_MD)) * ox.mmx(i, k - 1);
    let dd = tdelta(gm.tsc(k - 1, P7P_DD)) * ox.dmx(i, k - 1);
    if md >= dd {
        State::M
    } else {
        State::D
    }
}

#[inline]
fn select_i(gm: &Profile, ox: &Gmx, i: usize, k: usize) -> State {
    let mi = tdelta(gm.tsc(k, P7P_MI)) * ox.mmx(i - 1, k);
    let ii = tdelta(gm.tsc(k, P7P_II)) * ox.imx(i - 1, k);
    if mi >= ii {
        State::M
    } else {
        State::I
    }
}

#[inline]
fn select_n(i: usize) -> State {
    if i == 0 {
        State::S
    } else {
        State::N
    }
}

#[inline]
fn select_c(gm: &Profile, pp: &Gmx, ox: &Gmx, i: usize) -> State {
    let c_loop = tdelta(gm.xsc[P7P_C][P7P_LOOP]) * (ox.xmx(i - 1, P7G_C) + pp.xmx(i, P7G_C));
    let e_move = tdelta(gm.xsc[P7P_E][P7P_MOVE]) * ox.xmx(i, P7G_E);
    if c_loop > e_move {
        State::C
    } else {
        State::E
    }
}

#[inline]
fn select_j(gm: &Profile, pp: &Gmx, ox: &Gmx, i: usize) -> State {
    let j_loop = tdelta(gm.xsc[P7P_J][P7P_LOOP]) * (ox.xmx(i - 1, P7G_J) + pp.xmx(i, P7G_J));
    let e_loop = tdelta(gm.xsc[P7P_E][P7P_LOOP]) * ox.xmx(i, P7G_E);
    if j_loop > e_loop {
        State::J
    } else {
        State::E
    }
}

#[inline]
fn select_e(gm: &Profile, ox: &Gmx, i: usize, ret_k: &mut usize) -> State {
    if !gm.is_local() {
        *ret_k = gm.m;
        return if ox.mmx(i, gm.m) >= ox.dmx(i, gm.m) {
            State::M
        } else {
            State::D
        };
    }

    let mut max = f32::NEG_INFINITY;
    let mut smax = State::M;
    let mut kmax = 1usize;
    for k in 1..=gm.m {
        if ox.mmx(i, k) >= max {
            max = ox.mmx(i, k);
            smax = State::M;
            kmax = k;
        }
        if ox.dmx(i, k) > max {
            max = ox.dmx(i, k);
            smax = State::D;
            kmax = k;
        }
    }
    *ret_k = kmax;
    smax
}

#[inline]
fn select_b(gm: &Profile, ox: &Gmx, i: usize) -> State {
    let n_move = tdelta(gm.xsc[P7P_N][P7P_MOVE]) * ox.xmx(i, P7G_N);
    let j_move = tdelta(gm.xsc[P7P_J][P7P_MOVE]) * ox.xmx(i, P7G_J);
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

/// Get posterior probability for a position in the trace.
/// Used to annotate each alignment position with its confidence.
pub fn get_postprob(pp: &Gmx, state: u8, i: usize, k: usize) -> f32 {
    match state {
        1 => pp.mmx(i, k),      // M state
        2 => 0.0,               // D state (no emission)
        3 => pp.imx(i, k),      // I state
        5 => pp.xmx(i, P7G_N),  // N
        8 => pp.xmx(i, P7G_C),  // C
        10 => pp.xmx(i, P7G_J), // J
        _ => 0.0,
    }
}

/// Convert a posterior probability to a display character.
/// 0-9 for deciles, * for >= 0.95.
pub fn pp_to_char(pp: f32) -> char {
    if pp >= 0.95 {
        '*'
    } else {
        let d = (pp * 10.0) as u32;
        char::from_digit(d.min(9), 10).unwrap_or('0')
    }
}
