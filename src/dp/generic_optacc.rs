//! Generic optimal accuracy alignment.
//! Port of generic_optacc.c — finds alignment maximizing expected correct positions.

use crate::dp::gmx::*;
use crate::profile::*;

/// Run optimal accuracy DP using the posterior probability matrix.
/// Returns the OA score (expected number of correctly aligned residues).
pub fn g_optimal_accuracy(gm: &Profile, pp: &Gmx, ox: &mut Gmx) -> f32 {
    let l = pp.l;
    let m = gm.m;

    ox.m = m;
    ox.l = l;

    // Initialize row 0
    ox.set_xmx(0, P7G_N, 0.0);
    ox.set_xmx(0, P7G_B, 0.0);
    ox.set_xmx(0, P7G_E, f32::NEG_INFINITY);
    ox.set_xmx(0, P7G_C, f32::NEG_INFINITY);
    ox.set_xmx(0, P7G_J, f32::NEG_INFINITY);
    for k in 0..=m {
        ox.set_mmx(0, k, f32::NEG_INFINITY);
        ox.set_imx(0, k, f32::NEG_INFINITY);
        ox.set_dmx(0, k, f32::NEG_INFINITY);
    }

    for i in 1..=l {
        ox.set_mmx(i, 0, f32::NEG_INFINITY);
        ox.set_imx(i, 0, f32::NEG_INFINITY);
        ox.set_dmx(i, 0, f32::NEG_INFINITY);
        ox.set_xmx(i, P7G_E, f32::NEG_INFINITY);

        let esc: f32 = if gm.is_local() { 0.0 } else { f32::NEG_INFINITY };

        for k in 1..=m {
            let pp_m = pp.mmx(i, k); // posterior prob of M(i,k)

            // Match: max over predecessors + pp
            let mut sc = f32::NEG_INFINITY;
            if k > 1 {
                let mm = if gm.tsc(k - 1, P7P_MM) > f32::NEG_INFINITY {
                    ox.mmx(i - 1, k - 1) + pp_m
                } else {
                    f32::NEG_INFINITY
                };
                let im = if gm.tsc(k - 1, P7P_IM) > f32::NEG_INFINITY {
                    ox.imx(i - 1, k - 1) + pp_m
                } else {
                    f32::NEG_INFINITY
                };
                let dm = if gm.tsc(k - 1, P7P_DM) > f32::NEG_INFINITY {
                    ox.dmx(i - 1, k - 1) + pp_m
                } else {
                    f32::NEG_INFINITY
                };
                sc = mm.max(im).max(dm);
            }
            // B->M entry
            let bm = if gm.tsc(k.saturating_sub(1), P7P_BM) > f32::NEG_INFINITY {
                ox.xmx(i - 1, P7G_B) + pp_m
            } else {
                f32::NEG_INFINITY
            };
            sc = sc.max(bm);
            ox.set_mmx(i, k, sc);

            // E state update
            let e = ox.xmx(i, P7G_E).max(ox.mmx(i, k) + esc);
            ox.set_xmx(i, P7G_E, e);

            // Insert: max over predecessors + pp_i
            if k < m {
                let pp_i = pp.imx(i, k);
                let mi = if gm.tsc(k, P7P_MI) > f32::NEG_INFINITY {
                    ox.mmx(i - 1, k) + pp_i
                } else {
                    f32::NEG_INFINITY
                };
                let ii = if gm.tsc(k, P7P_II) > f32::NEG_INFINITY {
                    ox.imx(i - 1, k) + pp_i
                } else {
                    f32::NEG_INFINITY
                };
                ox.set_imx(i, k, mi.max(ii));
            } else {
                ox.set_imx(i, k, f32::NEG_INFINITY);
            }

            // Delete: max over predecessors (no emission)
            if k > 1 {
                let md = if gm.tsc(k - 1, P7P_MD) > f32::NEG_INFINITY {
                    ox.mmx(i, k - 1)
                } else {
                    f32::NEG_INFINITY
                };
                let dd = if gm.tsc(k - 1, P7P_DD) > f32::NEG_INFINITY {
                    ox.dmx(i, k - 1)
                } else {
                    f32::NEG_INFINITY
                };
                ox.set_dmx(i, k, md.max(dd));
            } else {
                ox.set_dmx(i, k, f32::NEG_INFINITY);
            }
        }

        // E state from D_M
        let e = ox.xmx(i, P7G_E).max(ox.dmx(i, m));
        ox.set_xmx(i, P7G_E, e);

        // Special states
        let pp_n = pp.xmx(i, P7G_N);
        let pp_j = pp.xmx(i, P7G_J);
        let pp_c = pp.xmx(i, P7G_C);

        let j = (ox.xmx(i - 1, P7G_J) + pp_j).max(ox.xmx(i, P7G_E));
        ox.set_xmx(i, P7G_J, j);

        let c = (ox.xmx(i - 1, P7G_C) + pp_c).max(ox.xmx(i, P7G_E));
        ox.set_xmx(i, P7G_C, c);

        let n = ox.xmx(i - 1, P7G_N) + pp_n;
        ox.set_xmx(i, P7G_N, n);

        let b = ox.xmx(i, P7G_N).max(ox.xmx(i, P7G_J));
        ox.set_xmx(i, P7G_B, b);
    }

    ox.xmx(l, P7G_C)
}

/// Get posterior probability for a position in the trace.
/// Used to annotate each alignment position with its confidence.
pub fn get_postprob(pp: &Gmx, state: u8, i: usize, k: usize) -> f32 {
    match state {
        1 => pp.mmx(i, k), // M state
        2 => 0.0,          // D state (no emission)
        3 => pp.imx(i, k), // I state
        5 => pp.xmx(i, P7G_N), // N
        8 => pp.xmx(i, P7G_C), // C
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
