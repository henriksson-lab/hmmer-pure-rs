//! Generic stochastic traceback — sample alignment from Forward distribution.
//! Port of generic_stotrace.c p7_GStochasticTrace().

use crate::alphabet::Dsq;
use crate::dp::gmx::*;
use crate::profile::*;
use crate::trace::{State, Trace};
use crate::util::random::MersenneTwister;

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
                    i -= 1;
                    tr.append(State::C, 0, i);
                } else {
                    cur_state = State::E;
                    tr.append(State::E, 0, i);
                }
            }
            State::E => {
                // E from any M_k or D_M (local mode)
                let mut scores = Vec::with_capacity(m + 1);
                for k in 1..=m {
                    scores.push(gx.mmx(i, k));
                }
                scores.push(gx.dmx(i, m));
                let choice = sample_from(rng, &scores);
                if choice < m {
                    let k = choice + 1;
                    cur_state = State::M;
                    tr.append(State::M, k, i);
                } else {
                    cur_state = State::D;
                    tr.append(State::D, m, 0);
                }
            }
            State::M => {
                let k = *tr.k.last().unwrap();
                if k == 0 || i == 0 {
                    cur_state = State::B;
                    tr.append(State::B, 0, i);
                    continue;
                }
                if k == 1 {
                    i -= 1;
                    cur_state = State::B;
                    tr.append(State::B, 0, i);
                    continue;
                }
                // Predecessor scores weighted by transition probabilities
                // (emission score is implicit in the Forward matrix values)
                debug_assert!(i <= dsq.len() - 2, "sequence position out of bounds");
                let mm = gx.mmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_MM);
                let im = gx.imx(i - 1, k - 1) + gm.tsc(k - 1, P7P_IM);
                let dm = gx.dmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_DM);
                let bm = gx.xmx(i - 1, P7G_B) + gm.tsc(k - 1, P7P_BM);
                let choice = sample_from(rng, &[mm, im, dm, bm]);
                i -= 1;
                match choice {
                    0 => {
                        cur_state = State::M;
                        tr.append(State::M, k - 1, i);
                    }
                    1 => {
                        cur_state = State::I;
                        tr.append(State::I, k - 1, i);
                    }
                    2 => {
                        cur_state = State::D;
                        tr.append(State::D, k - 1, 0);
                    }
                    _ => {
                        cur_state = State::B;
                        tr.append(State::B, 0, i);
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
                    tr.append(State::D, k - 1, 0);
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
                let nn = gx.xmx(i - 1, P7G_N) + gm.xsc[P7P_N][P7P_LOOP];
                let ns = 0.0; // S->N is implicit
                if sample_two(rng, nn, ns) == 0 {
                    i -= 1;
                    tr.append(State::N, 0, i);
                } else {
                    tr.append(State::S, 0, 0);
                    break;
                }
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
                    i -= 1;
                    tr.append(State::J, 0, i);
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
