//! Generic stochastic traceback — sample alignment from Forward distribution.
//! Port of generic_stotrace.c p7_GStochasticTrace().

use crate::alphabet::Dsq;
use crate::dp::gmx::*;
use crate::profile::*;
use crate::trace::{State, Trace};
use crate::util::random::MersenneTwister;

#[cfg(target_arch = "x86_64")]
use crate::simd::oprofile::*;
#[cfg(target_arch = "x86_64")]
use crate::simd::probmx::{ProbMx, PXB, PXC, PXE, PXJ, PXN};

#[cfg(target_arch = "x86_64")]
pub fn stochastic_trace_pmx(
    rng: &mut MersenneTwister,
    l: usize,
    om: &OProfile,
    ox: &ProbMx,
) -> Trace {
    let mut tr = Trace::new();
    stochastic_trace_pmx_into(rng, l, om, ox, &mut tr);
    tr
}

#[cfg(target_arch = "x86_64")]
pub fn stochastic_trace_pmx_into(
    rng: &mut MersenneTwister,
    l: usize,
    om: &OProfile,
    ox: &ProbMx,
    tr: &mut Trace,
) {
    tr.clear();
    let mut i = l;
    let mut k = 0usize;
    tr.append(State::T, k, i);
    tr.append(State::C, k, i);
    let mut s0 = State::C;

    while s0 != State::S {
        let s1 = match s0 {
            State::M => {
                let next = select_m_pmx(rng, om, ox, i, k);
                k -= 1;
                i -= 1;
                next
            }
            State::D => {
                let next = select_d_pmx(rng, om, ox, i, k);
                k -= 1;
                next
            }
            State::I => {
                let next = select_i_pmx(rng, om, ox, i, k);
                i -= 1;
                next
            }
            State::N => {
                if i == 0 {
                    State::S
                } else {
                    State::N
                }
            }
            State::C => select_c_pmx(rng, om, ox, i),
            State::J => select_j_pmx(rng, om, ox, i),
            State::E => select_e_pmx(rng, om, ox, i, &mut k),
            State::B => select_b_pmx(rng, om, ox, i),
            _ => State::S,
        };
        tr.append(s1, k, i);
        if matches!(s1, State::N | State::J | State::C) && s1 == s0 {
            i -= 1;
        }
        s0 = s1;
    }

    tr.st.reverse();
    tr.k.reverse();
    tr.i.reverse();
}

#[cfg(target_arch = "x86_64")]
fn tfv(om: &OProfile, k: usize, t: usize) -> f32 {
    let q = (k - 1) % nqf(om.m);
    let lane = (k - 1) / nqf(om.m);
    om.tfv[7 * q + t][lane]
}

#[cfg(target_arch = "x86_64")]
fn tfv_dd(om: &OProfile, k: usize) -> f32 {
    let q = (k - 1) % nqf(om.m);
    let lane = (k - 1) / nqf(om.m);
    om.tfv[7 * nqf(om.m) + q][lane]
}

#[cfg(target_arch = "x86_64")]
fn select_m_pmx(
    rng: &mut MersenneTwister,
    om: &OProfile,
    ox: &ProbMx,
    i: usize,
    k: usize,
) -> State {
    let prev_k = k - 1;
    let bm = ox.xmx(i - 1, PXB) * tfv(om, k, P7O_BM);
    let mm = if prev_k > 0 {
        ox.mmx(i - 1, prev_k)
    } else {
        0.0
    } * tfv(om, k, P7O_MM);
    let im = if prev_k > 0 {
        ox.imx(i - 1, prev_k)
    } else {
        0.0
    } * tfv(om, k, P7O_IM);
    let dm = if prev_k > 0 {
        ox.dmx(i - 1, prev_k)
    } else {
        0.0
    } * tfv(om, k, P7O_DM);
    match choose_probs(rng, &[bm, mm, im, dm]) {
        0 => State::B,
        1 => State::M,
        2 => State::I,
        _ => State::D,
    }
}

#[cfg(target_arch = "x86_64")]
fn select_d_pmx(
    rng: &mut MersenneTwister,
    om: &OProfile,
    ox: &ProbMx,
    i: usize,
    k: usize,
) -> State {
    let prev_k = k - 1;
    let md = ox.mmx(i, prev_k) * tfv(om, prev_k, P7O_MD);
    let dd = ox.dmx(i, prev_k) * tfv_dd(om, prev_k);
    if choose_probs(rng, &[md, dd]) == 0 {
        State::M
    } else {
        State::D
    }
}

#[cfg(target_arch = "x86_64")]
fn select_i_pmx(
    rng: &mut MersenneTwister,
    om: &OProfile,
    ox: &ProbMx,
    i: usize,
    k: usize,
) -> State {
    let mi = ox.mmx(i - 1, k) * tfv(om, k, P7O_MI);
    let ii = ox.imx(i - 1, k) * tfv(om, k, P7O_II);
    if choose_probs(rng, &[mi, ii]) == 0 {
        State::M
    } else {
        State::I
    }
}

#[cfg(target_arch = "x86_64")]
fn select_c_pmx(rng: &mut MersenneTwister, om: &OProfile, ox: &ProbMx, i: usize) -> State {
    let c = ox.xmx(i - 1, PXC) * om.xf[P7O_C][P7O_LOOP];
    let e = ox.xmx(i, PXE) * om.xf[P7O_E][P7O_MOVE] * ox.row_scale[i];
    if choose_probs(rng, &[c, e]) == 0 {
        State::C
    } else {
        State::E
    }
}

#[cfg(target_arch = "x86_64")]
fn select_j_pmx(rng: &mut MersenneTwister, om: &OProfile, ox: &ProbMx, i: usize) -> State {
    let j = ox.xmx(i - 1, PXJ) * om.xf[P7O_J][P7O_LOOP];
    let e = ox.xmx(i, PXE) * om.xf[P7O_E][P7O_LOOP] * ox.row_scale[i];
    if choose_probs(rng, &[j, e]) == 0 {
        State::J
    } else {
        State::E
    }
}

#[cfg(target_arch = "x86_64")]
fn select_b_pmx(rng: &mut MersenneTwister, om: &OProfile, ox: &ProbMx, i: usize) -> State {
    let n = ox.xmx(i, PXN) * om.xf[P7O_N][P7O_MOVE];
    let j = ox.xmx(i, PXJ) * om.xf[P7O_J][P7O_MOVE];
    if choose_probs(rng, &[n, j]) == 0 {
        State::N
    } else {
        State::J
    }
}

#[cfg(target_arch = "x86_64")]
fn select_e_pmx(
    rng: &mut MersenneTwister,
    om: &OProfile,
    ox: &ProbMx,
    i: usize,
    ret_k: &mut usize,
) -> State {
    let q = nqf(om.m);
    let roll = rng.next_f64();
    let norm = (1.0 / ox.xmx(i, PXE) as f64) as f32;
    let mut sum = 0.0_f64;
    for qi in 0..q {
        for lane in 0..4 {
            let k = lane * q + qi + 1;
            sum += (ox.mmx(i, k) * norm) as f64;
            if roll < sum {
                *ret_k = k;
                return State::M;
            }
        }
        for lane in 0..4 {
            let k = lane * q + qi + 1;
            sum += (ox.dmx(i, k) * norm) as f64;
            if roll < sum {
                *ret_k = k;
                return State::D;
            }
        }
    }
    *ret_k = om.m;
    State::M
}

#[cfg(target_arch = "x86_64")]
fn choose_probs(rng: &mut MersenneTwister, probs: &[f32]) -> usize {
    let mut normed = probs.to_vec();
    f_norm(&mut normed);
    let roll = rng.next_f64();
    let mut cumsum = 0.0_f64;
    let total: f64 = normed.iter().map(|&p| p as f64).sum();
    for (idx, &p) in normed.iter().enumerate() {
        cumsum += p as f64;
        if roll < cumsum / total {
            return idx;
        }
    }
    probs.len() - 1
}

#[cfg(target_arch = "x86_64")]
fn f_norm(vec: &mut [f32]) {
    let mut sum = 0.0_f32;
    let mut c = 0.0_f32;
    for &v in vec.iter() {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    if sum != 0.0 {
        for v in vec {
            *v /= sum;
        }
    } else {
        let p = 1.0 / vec.len() as f32;
        for v in vec {
            *v = p;
        }
    }
}

/// Sample a stochastic traceback from a Forward DP matrix.
/// Returns a Trace sampled from the posterior distribution of alignments.
pub fn g_stochastic_trace(
    rng: &mut MersenneTwister,
    dsq: &[Dsq],
    l: usize,
    gm: &Profile,
    gx: &Gmx,
) -> Trace {
    let m = gm.m;
    let mut tr = Trace::new();

    // Start from T->C at position L
    tr.append(State::T, 0, 0);
    tr.append(State::C, 0, l);
    let mut i = l;
    let mut cur_state = State::C;

    loop {
        match cur_state {
            State::C => {
                if i == 0 {
                    break;
                }
                // C from C(i-1) or E(i)
                let c_sc = gx.xmx(i - 1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP];
                let e_sc = gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_MOVE];
                if sample_two(rng, c_sc, e_sc) == 0 {
                    tr.append(State::C, 0, i);
                    i -= 1;
                } else {
                    cur_state = State::E;
                    tr.append(State::E, 0, i);
                }
            }
            State::E => {
                // Local E connects from any M_k or any D_k for k=2..M.
                let mut scores = vec![f32::NEG_INFINITY; 2 * m + 1];
                for k in 1..=m {
                    scores[k] = gx.mmx(i, k);
                }
                for k in 2..=m {
                    scores[k + m] = gx.dmx(i, k);
                }
                let choice = sample_from(rng, &scores);
                if choice <= m {
                    let k = choice;
                    cur_state = State::M;
                    tr.append(State::M, k, i);
                } else {
                    let k = choice - m;
                    cur_state = State::D;
                    tr.append(State::D, k, i);
                }
            }
            State::M => {
                let k = *tr.k.last().unwrap();
                // Predecessor scores weighted by transition probabilities
                // (emission score is implicit in the Forward matrix values)
                debug_assert!(i <= dsq.len() - 2, "sequence position out of bounds");
                let bm = gx.xmx(i - 1, P7G_B) + gm.tsc(k - 1, P7P_BM);
                let mm = gx.mmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_MM);
                let im = gx.imx(i - 1, k - 1) + gm.tsc(k - 1, P7P_IM);
                let dm = gx.dmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_DM);
                let choice = sample_from(rng, &[bm, mm, im, dm]);
                i -= 1;
                match choice {
                    0 => {
                        cur_state = State::B;
                        tr.append(State::B, 0, i);
                    }
                    1 => {
                        cur_state = State::M;
                        tr.append(State::M, k - 1, i);
                    }
                    2 => {
                        cur_state = State::I;
                        tr.append(State::I, k - 1, i);
                    }
                    _ => {
                        cur_state = State::D;
                        tr.append(State::D, k - 1, i);
                    }
                }
            }
            State::D => {
                let k = *tr.k.last().unwrap();
                if k <= 1 {
                    cur_state = State::B;
                    tr.append(State::B, 0, i);
                    continue;
                }
                let md = gx.mmx(i, k - 1) + gm.tsc(k - 1, P7P_MD);
                let dd = gx.dmx(i, k - 1) + gm.tsc(k - 1, P7P_DD);
                if sample_two(rng, md, dd) == 0 {
                    cur_state = State::M;
                    tr.append(State::M, k - 1, i);
                } else {
                    cur_state = State::D;
                    tr.append(State::D, k - 1, i);
                }
            }
            State::I => {
                let k = *tr.k.last().unwrap();
                if i == 0 {
                    break;
                }
                i -= 1;
                let mi = gx.mmx(i, k) + gm.tsc(k, P7P_MI);
                let ii = gx.imx(i, k) + gm.tsc(k, P7P_II);
                if sample_two(rng, mi, ii) == 0 {
                    cur_state = State::M;
                    tr.append(State::M, k, i);
                } else {
                    cur_state = State::I;
                    tr.append(State::I, k, i);
                }
            }
            State::B => {
                let bn = gx.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_MOVE];
                let bj = gx.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_MOVE];
                if sample_two(rng, bn, bj) == 0 {
                    cur_state = State::N;
                    tr.append(State::N, 0, i);
                } else {
                    cur_state = State::J;
                    tr.append(State::J, 0, i);
                }
            }
            State::N => {
                if i == 0 {
                    tr.append(State::S, 0, 0);
                    break;
                }
                tr.append(State::N, 0, i);
                i -= 1;
            }
            State::J => {
                if i == 0 {
                    cur_state = State::E;
                    tr.append(State::E, 0, 0);
                    continue;
                }
                let jj = gx.xmx(i - 1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP];
                let je = gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_LOOP];
                if sample_two(rng, jj, je) == 0 {
                    tr.append(State::J, 0, i);
                    i -= 1;
                } else {
                    cur_state = State::E;
                    tr.append(State::E, 0, i);
                }
            }
            _ => break,
        }

        if tr.n > l + m + 100 {
            break;
        }
    }

    tr.st.reverse();
    tr.k.reverse();
    tr.i.reverse();
    tr
}

/// Sample from two options with log-space scores, returns 0 or 1.
fn sample_two(rng: &mut MersenneTwister, a: f32, b: f32) -> usize {
    let max = a.max(b);
    let pa = (a - max).exp();
    let pb = (b - max).exp();
    let total = pa + pb;
    if total <= 0.0 {
        return 0;
    }
    if rng.next_f32() * total < pa {
        0
    } else {
        1
    }
}

/// Sample from a vector of log-space scores, returns index.
fn sample_from(rng: &mut MersenneTwister, scores: &[f32]) -> usize {
    if scores.is_empty() {
        return 0;
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        return 0;
    }
    let r = rng.next_f32() * total;
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    scores.len() - 1
}
