//! SSV filter for long target sequences (nhmmer/Infernal).
//! Port of p7_SSVFilter_longtarget() from impl_sse/msvfilter.c.
//!
//! Slides a window across a long DNA/RNA sequence, returning hit windows
//! where the SSV score exceeds a P-value threshold. Used by nhmmer and
//! Infernal for scanning genomes.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::bg::Bg;
use crate::simd::oprofile::OProfile;
use crate::stats;

/// A window hit from the SSV long-target filter.
#[derive(Debug, Clone)]
pub struct HmmWindow {
    /// Position in target sequence where the diagonal starts (1-based)
    pub n: usize,
    /// Position in the model where the diagonal ends
    pub k: usize,
    /// Length of the diagonal (model positions)
    pub length: usize,
    /// Score of the diagonal (nats)
    pub score: f32,
    /// Length of the target sequence
    pub target_len: usize,
    /// Whether this is the complement strand
    pub complement: bool,
}

/// Extract SSV emission scores from the oprofile into a flat array.
/// Returns ssv_scores[(M+1) * Kp] in de-striped layout: ssv_scores[k * Kp + x].
/// Port of p7_oprofile_GetSSVEmissionScoreArray().
pub fn get_ssv_score_array(om: &OProfile) -> Vec<u8> {
    let m = om.m;
    let kp = om.abc_kp;
    let nq = crate::simd::oprofile::nqb(m); // p7O_NQB, min 2
    let cell_cnt = (m + 1) * kp;
    let mut arr = vec![0u8; cell_cnt];

    for x in 0..kp {
        if x >= om.rbv.len() {
            break;
        }
        let mut k = 1;
        for q in 0..nq {
            if q >= om.rbv[x].len() {
                break;
            }
            let vec = &om.rbv[x][q];
            for z in 0..16 {
                let idx = kp * (k + z * nq) + x;
                if idx < cell_cnt && (k + z * nq) <= m {
                    arr[idx] = vec[z];
                }
            }
            k += 1;
        }
    }
    arr
}

/// SSV filter for long target sequences.
/// Port of p7_SSVFilter_longtarget().
///
/// Scans the full sequence `dsq[1..=L]` and returns a list of windows
/// where the SSV score exceeds the P-value threshold `p_thresh`.
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn ssv_filter_longtarget(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    _bg: &Bg,
    p_thresh: f64,
    max_length: i32,
) -> Vec<HmmWindow> {
    let m = om.m;
    let kp = om.abc_kp;
    // Match C p7O_NQB: `ESL_MAX(2, ((M-1)/16 + 1))`. For small M (<=16) C
    // uses 2 vectors, not 1. Using `(m+15)/16` gives 1 for M=6, which
    // produces a different striping layout than C and misses peaks on
    // tiny HMMs (e.g. ecori, M=6).
    let q_count = crate::simd::oprofile::nqb(m);

    // Get flat SSV score array for diagonal traceback
    let ssv_scores = get_ssv_score_array(om);

    // Compute score threshold from P-value.
    // Matches C: uses om->max_length for length config, om->tjb_b for thresholds.
    // C calls p7_bg_SetLength + p7_oprofile_ReconfigMSVLength + p7_bg_NullOne
    // before computing the threshold. We use om's existing tjb_b which should
    // already be configured for om->max_length.
    let inv_p = stats::gumbel::invsurv(
        p_thresh,
        om.evparam[crate::hmm::P7_MMU] as f64,
        om.evparam[crate::hmm::P7_MLAMBDA] as f64,
    );

    // Null score: p7_bg_NullOne computes L * log(p1) + log(1-p1) where p1 = L/(L+1).
    // C uses om->max_length which was set by ReconfigMSVLength before this call.
    // We use the max_length parameter which the caller should set to om->max_length.
    // Note: C computes nullsc as a float via p7_bg_NullOne, which internally uses
    // double log(). We mirror that by computing in f64 and casting the final
    // result to f32.
    let ml = if max_length > 0 { max_length as usize } else { m * 4 };
    let p1: f32 = ml as f32 / (ml as f32 + 1.0);
    let nullsc: f32 =
        (ml as f64 * (p1 as f64).ln() + (1.0_f64 - p1 as f64).ln()) as f32;

    // Use om->tjb_b directly (matching C line 324, 334, 385, 420)
    // C reconfigures om->tjb_b via ReconfigMSVLength before calling this function.
    // The tjb_b in om should already be set for max_length.
    let tjb_b = om.tjb_b;

    // Match C msvfilter.c:324 exactly:
    //   sc_thresh = (int) ceil(((nullsc + invP*eslCONST_LOG2 + 3.0) * om->scale_b)
    //                          + om->base_b + om->tec_b + om->tjb_b);
    // C declares `float invP` (msvfilter.c:312), so the value is first truncated
    // to f32 precision before being promoted back to double for the arithmetic.
    // Rust's invsurv returns f64; keeping full f64 precision shifts the ceil()
    // boundary for a few sequences and produces different SSV peak sets.
    let inv_p_f32 = inv_p as f32;
    let sc_thresh_d: f64 = (((nullsc as f64)
        + (inv_p_f32 as f64 * std::f64::consts::LN_2)
        + 3.0)
        * (om.scale_b as f64))
        + (om.base_b as f64)
        + (om.tec_b as f64)
        + (tjb_b as f64);
    let sc_thresh: u8 = if sc_thresh_d >= 255.0 {
        255
    } else if sc_thresh_d <= 0.0 {
        0
    } else {
        sc_thresh_d.ceil() as u8
    };

    let sc_thresh_v = _mm_set1_epi8((255u8.wrapping_sub(sc_thresh)) as i8);
    let bias_v = _mm_set1_epi8(om.bias_b as i8);
    let ceiling_v = _mm_cmpeq_epi8(bias_v, bias_v); // all 0xFF
    let base_v = _mm_set1_epi8(om.base_b as i8);
    // C line 334: tjbmv = set1(om->tjb_b + om->tbm_b)
    let tjbm_v = _mm_set1_epi8((tjb_b.wrapping_add(om.tbm_b)) as i8);
    let xb_v = _mm_subs_epu8(base_v, tjbm_v);

    // DP row: q_count vectors of 16 u8s
    let mut dp = vec![_mm_setzero_si128(); q_count];

    let mut windows = Vec::new();

    let mut i = 1usize;
    while i <= l {
        let x = dsq[i] as usize;
        if x >= om.rbv.len() {
            i += 1;
            continue;
        }

        let mut xe_v = _mm_setzero_si128();

        // Shift previous row right by 1 byte (diagonal propagation)
        let mut mpv = _mm_slli_si128(dp[q_count - 1], 1);

        for q in 0..q_count {
            // Load emission score for this residue at this vector position
            let rsc = if q < om.rbv[x].len() {
                let bytes = om.rbv[x][q];
                _mm_loadu_si128(bytes.as_ptr() as *const __m128i)
            } else {
                _mm_setzero_si128()
            };

            // sv = max(mpv, xBv): continue diagonal or start from B
            let mut sv = _mm_max_epu8(mpv, xb_v);
            // Add bias
            sv = _mm_adds_epu8(sv, bias_v);
            // Subtract emission score (in offset unsigned arithmetic)
            sv = _mm_subs_epu8(sv, rsc);
            // Track max
            xe_v = _mm_max_epu8(xe_v, sv);

            mpv = dp[q];
            dp[q] = sv;
        }

        // Test if threshold exceeded
        let temp_v = _mm_adds_epu8(xe_v, sc_thresh_v);
        let temp_v = _mm_cmpeq_epi8(temp_v, ceiling_v);
        let cmp = _mm_movemask_epi8(temp_v);



        if cmp != 0 {
            // Find which model state exceeded threshold.
            // rem_sc is int (signed) in C to allow traceback to subtract past 0.
            let mut end = 0usize;
            let mut rem_sc: i32 = -1;

            for q in 0..q_count {
                let bytes: [u8; 16] = std::mem::transmute(dp[q]);
                for z in 0..16 {
                    let model_pos = q + 1 + z * q_count;
                    if model_pos <= m && bytes[z] >= sc_thresh && (bytes[z] as i32) > rem_sc {
                        end = model_pos;
                        rem_sc = bytes[z] as i32;
                    }
                }
                // Reset DP row (will restart from xBv next iteration)
                dp[q] = _mm_setzero_si128();
            }

            if end == 0 {
                i += 1;
                continue;
            }

            // Trace back along diagonal to find start.
            // Matches C msvfilter.c:385 — rem_sc is signed int, can go negative.
            let mut start = end;
            let mut target_start = i;
            let mut target_end = i;
            let sc_val = rem_sc;
            // C: entry_cost = base_b - tjb_b - tbm_b (all uint8_t, subtraction in int)
            let entry_cost: i32 = om.base_b as i32 - tjb_b as i32 - om.tbm_b as i32;

            // Match C msvfilter.c:385 exactly: no bounds check on start or
            // target_start. C relies on `rem_sc <= entry_cost` firing before
            // the indices go out of bounds. We use saturating subtraction so
            // out-of-bounds access becomes a zero emission rather than a
            // panic; entering this branch is rare enough in practice that
            // the saturation should be neutral, but it matches C behavior
            // when it does occur. Previously this loop had `start > 1 &&
            // target_start > 1` guards that caused early termination on a
            // few MADE1 peaks, perturbing the pre-merge SSV window list.
            while rem_sc > entry_cost {
                if start == 0 || target_start == 0 {
                    break;
                }
                let score_idx = start * kp + dsq[target_start] as usize;
                let emission = if score_idx < ssv_scores.len() {
                    ssv_scores[score_idx]
                } else {
                    0
                };
                rem_sc -= om.bias_b as i32 - emission as i32;
                start -= 1;
                target_start -= 1;
            }
            start += 1;
            target_start += 1;

            // Extend forward along diagonal
            let mut k = end + 1;
            let mut n = target_end + 1;
            let mut max_end = target_end;
            let mut max_sc = sc_val as i32;
            let mut pos_since_max = 0;
            let mut running_sc = sc_val as i32;

            // Match C: `while (k<om->M && n<=L)` — k strictly less than M.
            while k < m && n <= l {
                let score_idx = k * kp + dsq[n] as usize;
                let emission = if score_idx < ssv_scores.len() {
                    ssv_scores[score_idx]
                } else {
                    0
                };
                running_sc += om.bias_b as i32 - emission as i32;

                if running_sc >= max_sc {
                    max_sc = running_sc;
                    max_end = n;
                    pos_since_max = 0;
                } else {
                    pos_since_max += 1;
                    if pos_since_max == 5 {
                        break;
                    }
                }
                k += 1;
                n += 1;
            }

            end += max_end - target_end;
            target_end = max_end;

            // Convert score to nats (matching C line 420-422)
            // ret_sc = ((float)(max_sc - om->tjb_b) - (float)om->base_b) / om->scale_b - 3.0
            let ret_sc = ((max_sc as f32 - tjb_b as f32) - om.base_b as f32) / om.scale_b - 3.0;

            windows.push(HmmWindow {
                n: target_start,
                k: end,
                length: end - start + 1,
                score: ret_sc,
                target_len: l,
                complement: false,
            });

            // Skip forward past the hit
            i = target_end + 1;
        } else {
            i += 1;
        }
    }

    windows
}

/// Extend and merge windows, expanding each by prefix/suffix lengths.
/// Port of p7_pli_ExtendAndMergeWindows().
/// For simplicity, uses a fixed extension of max_length * 0.2 on each side.
/// Compute per-model-position prefix_lengths and suffix_lengths for the
/// window extension heuristic. Port of `p7_hmm_ScoreDataComputeRest`
/// (hmmer/src/p7_scoredata.c:314). Returns (prefix_lengths, suffix_lengths)
/// each of size `M+1`, normalized prefix-sums that represent the fractional
/// contribution of model positions to the expected window length.
///
/// For each position k in 1..M:
///   raw[k] = 1 + floor(log(BETA/t_MI[k]) / log(t_II[k]))  (if t_MI[k] != 0)
///   raw[k] = 1                                             (otherwise)
/// Then `raw` is normalized by sum, suffix-summed, and prefix-summed.
pub fn compute_prefix_suffix_lengths(hmm: &crate::hmm::Hmm) -> (Vec<f32>, Vec<f32>) {
    // Deprecated: uses raw HMM transitions. Prefer `compute_prefix_suffix_lengths_from_om`
    // which mirrors C `p7_hmm_ScoreDataComputeRest` exactly by reading
    // transitions via the optimized profile's tfv array (= round-tripped
    // exp(log)).
    compute_prefix_suffix_lengths_hmm(hmm)
}

fn compute_prefix_suffix_lengths_hmm(hmm: &crate::hmm::Hmm) -> (Vec<f32>, Vec<f32>) {
    use crate::hmm::{II, MI};
    const BETA: f32 = 1.0e-7;
    let m = hmm.m;
    let mut prefix = vec![0.0_f32; m + 1];
    let mut suffix = vec![0.0_f32; m + 1];
    let mut sum = 0.0_f32;
    for k in 1..m {
        // hmm.t[k] is the transitions at node k.
        let t_mi = hmm.t[k][MI];
        let t_ii = hmm.t[k][II];
        prefix[k] = if t_mi == 0.0 {
            1.0
        } else {
            let ln_t_ii = (t_ii as f64).ln();
            if ln_t_ii == 0.0 {
                1.0
            } else {
                let v = (BETA as f64 / t_mi as f64).ln() / ln_t_ii;
                1.0 + v.floor() as f32
            }
        };
        sum += prefix[k];
    }
    prefix[0] = 0.0;
    prefix[m] = 0.0;
    if sum > 0.0 {
        for k in 1..m {
            prefix[k] /= sum;
        }
    }

    // suffix_lengths[M] = prefix_lengths[M-1]
    if m >= 1 {
        suffix[m] = prefix[m - 1];
    }
    // suffix[k] = suffix[k+1] + prefix[k-1] for k = M-1 down to 1
    for k in (1..m).rev() {
        suffix[k] = suffix[k + 1] + if k >= 1 { prefix[k - 1] } else { 0.0 };
    }
    // Prefix-sum into prefix: prefix[k] += prefix[k-1] for k in 2..M
    for k in 2..m {
        prefix[k] += prefix[k - 1];
    }
    (prefix, suffix)
}

/// Compute prefix/suffix lengths using the OPTIMIZED profile's transitions
/// (tfv array), matching C `p7_hmm_ScoreDataComputeRest` byte-for-byte.
///
/// C reads transitions via `p7_oprofile_GetFwdTransitionArray(om, p7O_MI, ...)`
/// which pulls from `om->tfv` (= exp of log-transition = round-tripped
/// probability). Reading from raw HMM transitions gives slightly different
/// floating-point results that propagate through extend_and_merge_windows,
/// changing SSV window lengths by a few residues.
#[cfg(target_arch = "x86_64")]
pub fn compute_prefix_suffix_lengths_from_om(
    om: &crate::simd::oprofile::OProfile,
) -> (Vec<f32>, Vec<f32>) {
    // C uses the double literal `p7_DEFAULT_WINDOW_BETA == 1e-7` directly in
    // `log(BETA / t_mis[k])` (hmmer/src/p7_scoredata.c:366). Storing BETA as
    // f32 first and then casting to f64 yields a slightly different value
    // than the native double 1e-7 and shifts the `(int)` truncation boundary
    // for a few model positions.
    const BETA: f64 = 1.0e-7;
    let m = om.m;
    let nq = (m + 3) / 4; // nqf
    let mut prefix = vec![0.0_f32; m + 1];
    let mut suffix = vec![0.0_f32; m + 1];

    // Extract MI (offset 5) and II (offset 6) transitions from tfv:
    // arr[i+1 + j*nq] = tfv[5 + 7*i].x[j]  for MI, similarly II.
    let mut t_mi = vec![0.0_f32; m + 1];
    let mut t_ii = vec![0.0_f32; m + 1];
    for i in 0..nq {
        let mi_vec = om.tfv[5 + 7 * i];
        let ii_vec = om.tfv[6 + 7 * i];
        for j in 0..4 {
            let node = i + 1 + j * nq;
            if node <= m {
                t_mi[node] = mi_vec[j];
                t_ii[node] = ii_vec[j];
            }
        }
    }

    let mut sum = 0.0_f32;
    for k in 1..m {
        prefix[k] = if t_mi[k] == 0.0 {
            1.0
        } else {
            let ln_t_ii = (t_ii[k] as f64).ln();
            if ln_t_ii == 0.0 {
                1.0
            } else {
                // C: `1 + (int)(log(BETA / t_mi[k]) / log(t_ii[k]))`.
                // C `(int)` truncates toward zero (not floor).
                let v = (BETA / t_mi[k] as f64).ln() / ln_t_ii;
                1.0 + (v as i32) as f32
            }
        };
        sum += prefix[k];
    }
    prefix[0] = 0.0;
    prefix[m] = 0.0;
    if sum > 0.0 {
        for k in 1..m {
            prefix[k] /= sum;
        }
    }

    if m >= 1 {
        suffix[m] = prefix[m - 1];
    }
    for k in (1..m).rev() {
        suffix[k] = suffix[k + 1] + if k >= 1 { prefix[k - 1] } else { 0.0 };
    }
    for k in 2..m {
        prefix[k] += prefix[k - 1];
    }
    (prefix, suffix)
}

/// Extend windows outward by a fraction of max_length and merge those whose
/// overlap exceeds `pct_overlap` of the shorter window. Mirrors C
/// p7_pli_ExtendAndMergeWindows (hmmer/src/p7_pipeline.c:451).
///
/// C uses per-k `data->prefix_lengths[k]` / `suffix_lengths[k]` from
/// P7_SCOREDATA for a variable extension; we approximate with a fixed 0.1
/// fraction of max_length (the `0.1 +` term in C's formula). The pct_overlap
/// argument defaults to 0.5 at the post-Vit stage in C.
pub fn extend_and_merge_windows(
    windows: &mut Vec<HmmWindow>,
    max_length: usize,
    target_len: usize,
) {
    extend_and_merge_windows_pct(windows, max_length, target_len, 0.5);
}

/// Full C-matching variant: per-window extension uses
/// `max_length * (0.1 + prefix_lengths[k - length + 1])` for the left side
/// and `max_length * (0.1 + suffix_lengths[k])` for the right.
/// Mirrors the per-k branches in p7_pli_ExtendAndMergeWindows:482-487.
pub fn extend_and_merge_windows_with_scoredata(
    windows: &mut Vec<HmmWindow>,
    max_length: usize,
    target_len: usize,
    pct_overlap: f32,
    prefix_lengths: &[f32],
    suffix_lengths: &[f32],
) {
    if windows.is_empty() {
        return;
    }

    // Match C p7_pli_ExtendAndMergeWindows (p7_pipeline.c:485-486): the
    // extension amount is computed in double precision because the `0.1`
    // literal is a C double. Using f32 here caused a small number of
    // boundary-case window extensions to differ from C by one residue.
    let base_frac = 0.1_f64;
    let ml = max_length as f64;
    for w in windows.iter_mut() {
        let k = w.k.min(prefix_lengths.len().saturating_sub(1));
        let prefix_k = (w.k.saturating_sub(w.length).saturating_add(1))
            .min(prefix_lengths.len().saturating_sub(1));
        let pre_ext_f = ml * (base_frac + prefix_lengths[prefix_k] as f64);
        let suf_ext_f = ml * (base_frac + suffix_lengths[k] as f64);
        // Match C: (int64_t)(n - double), (int64_t)(n + length + double) with
        // truncation toward zero.
        let start_f = (w.n as i64) as f64 - pre_ext_f;
        let start = if start_f < 1.0 { 1 } else { start_f as i64 as usize };
        let end_f = ((w.n + w.length) as i64) as f64 + suf_ext_f;
        let end_raw = if end_f < 1.0 { 1 } else { end_f as i64 as usize };
        let end = end_raw.min(target_len);
        w.length = end - start + 1;
        w.n = start;
    }
    merge_windows_impl(windows, pct_overlap);
}

fn merge_windows_impl(windows: &mut Vec<HmmWindow>, pct_overlap: f32) {
    if windows.len() <= 1 {
        return;
    }
    let mut merged = Vec::with_capacity(windows.len());
    merged.push(windows[0].clone());

    for i in 1..windows.len() {
        let prev = merged.last_mut().unwrap();
        let curr = &windows[i];

        let prev_end = prev.n + prev.length - 1;
        let curr_end = curr.n + curr.length - 1;

        let overlap_start = prev.n.max(curr.n);
        let overlap_end = prev_end.min(curr_end);
        let overlap_len = if overlap_end >= overlap_start {
            overlap_end - overlap_start + 1
        } else {
            0
        };
        let min_len = prev.length.min(curr.length);
        let pct = if min_len > 0 {
            overlap_len as f32 / min_len as f32
        } else {
            0.0
        };

        if prev.complement == curr.complement && pct > pct_overlap {
            let new_start = prev.n.min(curr.n);
            let new_end = prev_end.max(curr_end);
            prev.n = new_start;
            prev.length = new_end - new_start + 1;
            if curr.score > prev.score {
                prev.score = curr.score;
            }
        } else {
            merged.push(curr.clone());
        }
    }

    *windows = merged;
}

pub fn extend_and_merge_windows_pct(
    windows: &mut Vec<HmmWindow>,
    max_length: usize,
    target_len: usize,
    pct_overlap: f32,
) {
    if windows.is_empty() {
        return;
    }

    // Extend each window by ~0.2 * max_length on each side. Fallback path used
    // when per-k scoredata isn't available.
    let extension = (max_length as f32 * 0.2).ceil() as usize;
    for w in windows.iter_mut() {
        let start = if w.n > extension { w.n - extension } else { 1 };
        let end = (w.n + w.length + extension).min(target_len);
        w.length = end - start + 1;
        w.n = start;
    }

    merge_windows_impl(windows, pct_overlap);
}
