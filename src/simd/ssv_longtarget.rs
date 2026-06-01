//! SSV filter for long target sequences (nhmmer/Infernal).
//! Port of p7_SSVFilter_longtarget() from impl_sse/msvfilter.c.
#![allow(clippy::needless_range_loop)]
//!
//! Slides a window across a long DNA/RNA sequence, returning hit windows
//! where the SSV score exceeds a P-value threshold. Used by nhmmer and
//! Infernal for scanning genomes.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::alphabet::Dsq;
#[cfg(target_arch = "x86_64")]
use crate::bg::Bg;
use crate::simd::oprofile::OProfile;
#[cfg(target_arch = "x86_64")]
use crate::stats;
#[cfg(target_arch = "x86_64")]
use crate::util::cmath::ESL_CONST_LOG2;

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
    /// FM segment id (C `P7_HMM_WINDOW.id`): the database-sequence id of the
    /// segment this window's diagonal hit. Used as a merge guard so windows
    /// from different FM segments on the same strand are never merged
    /// (C `p7_pipeline.c:506`). Defaults to 0 for the FASTA / single-segment
    /// path where there is exactly one segment per searched sequence.
    pub id: i32,
    /// Position in the concatenated FM-index sequence at which the diagonal
    /// starts (C `P7_HMM_WINDOW.fm_n`). Carried through extend+merge so the
    /// FM segment-boundary trim pass (C `p7_pipeline.c:1812-1849`) can map a
    /// window back to its original FM segment, and doubling as the sentinel
    /// for whether this window lives in the forward concatenated-FM frame
    /// (see `needs_complement_extension_flip`). The Rust FASTA / per-segment-FM
    /// driver sets `fm_n = -1` ("RC-local frame, no flip"); a future faithful
    /// concatenated-FM path would set `fm_n >= 0`.
    pub fm_n: i64,
    /// Segment start in the concatenated FM text. Only meaningful when
    /// `fm_n >= 0`; used by the Rust FM handoff to remap an extended C-frame
    /// window back to the searched per-segment sequence before DSQ slicing.
    pub fm_start: i64,
    /// FM BWT length including terminal symbol. Only meaningful when
    /// `fm_n >= 0`; needed for C's complement coordinate transform.
    pub fm_bwt_len: i64,
}

impl HmmWindow {
    /// Returns whether this window is on the complement strand in the **forward
    /// concatenated-FM coordinate frame** — i.e. whether C's `p7_COMPLEMENT`
    /// extension flip (`p7_pipeline.c:470-481`) should be applied during
    /// extension. The Rust FASTA / per-segment-FM driver reverse-complements
    /// the target up front and produces complement windows already in a
    /// forward-style RC-local frame, where the non-complement extension formula
    /// is the correct one; in that frame `fm_n < 0` is used as the sentinel for
    /// "not in the concatenated-FM frame". A window built directly in the
    /// concatenated forward FM frame (a future faithful FM path) would set
    /// `fm_n >= 0` and `complement = true`, triggering the flip.
    fn needs_complement_extension_flip(&self) -> bool {
        self.complement && self.fm_n >= 0
    }
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
        let mut k = 1;
        for q in 0..nq {
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

/// Finds windows with SSV scores above some threshold (vewy vewy fast, in
/// limited precision). Port of `p7_SSVFilter_longtarget`
/// (hmmer/src/impl_sse/msvfilter.c:213, despite the C filename it is the SSV
/// long-target entry point).
///
/// Calculates an approximation of the SSV (single ungapped diagonal) score
/// for regions of `dsq[1..=l]` using the optimized profile `om`, and
/// captures the positions at which such regions exceed the score required to
/// be significant for the supplied P-value (usually p=0.02 for nhmmer).
/// Never passes through the J state — the SSV threshold is sufficient to
/// pass MSV for essentially all DNA models tested.
///
/// Above-threshold diagonals become `HmmWindow` entries with start/end
/// positions established by tracing the diagonal forward and backward through
/// the SSV emission scores.
///
/// # Safety
/// Requires SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn p7_ssv_filter_longtarget(
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
    let ml = if max_length > 0 {
        max_length as usize
    } else {
        m * 4
    };
    let p1: f32 = ml as f32 / (ml as f32 + 1.0);
    let nullsc: f32 = (ml as f64 * crate::util::cmath::c_log_f64(p1 as f64)
        + crate::util::cmath::c_log_f64(1.0_f64 - p1 as f64)) as f32;

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
    let sc_thresh_d: f64 = (((nullsc as f64) + (inv_p_f32 as f64 * ESL_CONST_LOG2) + 3.0)
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

        let mut xe_v = _mm_setzero_si128();

        // Shift previous row right by 1 byte (diagonal propagation)
        let mut mpv = _mm_slli_si128(dp[q_count - 1], 1);

        for q in 0..q_count {
            // Load emission score for this residue at this vector position
            let bytes = om.rbv[x][q];
            let rsc = _mm_loadu_si128(bytes.as_ptr() as *const __m128i);

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

            // Match C msvfilter.c:385: the score table is indexed directly.
            // The remaining zero-bound break avoids Rust underflow if an invalid
            // diagonal violates C's implicit "entry_cost fires first" invariant.
            // Previously this loop had `start > 1 && target_start > 1` guards
            // that caused early termination on a few MADE1 peaks, perturbing the
            // pre-merge SSV window list.
            while rem_sc > entry_cost {
                if start == 0 || target_start == 0 {
                    break;
                }
                let score_idx = start * kp + dsq[target_start] as usize;
                let emission = ssv_scores[score_idx];
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
            let mut max_sc = sc_val;
            let mut pos_since_max = 0;
            let mut running_sc = sc_val;

            // Match C: `while (k<om->M && n<=L)` — k strictly less than M.
            while k < m && n <= l {
                let score_idx = k * kp + dsq[n] as usize;
                let emission = ssv_scores[score_idx];
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
                // FASTA SSV scan: single-segment, RC-local frame. id=0 (one
                // segment), fm_n=-1 (no concatenated-FM frame -> no flip).
                id: 0,
                fm_n: -1,
                fm_start: -1,
                fm_bwt_len: 0,
            });

            // Skip forward past the hit
            i = target_end + 1;
        } else {
            i += 1;
        }
    }

    windows
}

/// Compute per-model-position prefix_lengths and suffix_lengths for the
/// window extension heuristic. Port of `p7_hmm_ScoreDataComputeRest`
/// (hmmer/src/p7_scoredata.c:314). Returns (prefix_lengths, suffix_lengths)
/// each of size `M+1`, normalized prefix-sums that represent the fractional
/// contribution of model positions to the expected window length.
///
/// For each position k in 1..M:
///   `raw[k]` = 1 + floor(log(BETA/`t_MI[k]`) / log(`t_II[k]`))  (if `t_MI[k]` != 0)
///   `raw[k]` = 1                                             (otherwise)
/// Then `raw` is normalized by sum, suffix-summed, and prefix-summed.
pub fn compute_prefix_suffix_lengths(hmm: &crate::hmm::Hmm) -> (Vec<f32>, Vec<f32>) {
    // Deprecated: uses raw HMM transitions. Prefer `compute_prefix_suffix_lengths_from_om`
    // which mirrors C `p7_hmm_ScoreDataComputeRest` exactly by reading
    // transitions via the optimized profile's tfv array (= round-tripped
    // exp(log)).
    compute_prefix_suffix_lengths_hmm(hmm)
}

/// Legacy variant of `p7_hmm_ScoreDataComputeRest` that reads transitions
/// straight from the raw HMM (`hmm.t[k][MI]`, `hmm.t[k][II]`). Kept for
/// callers that don't yet have an `OProfile`; prefer
/// `compute_prefix_suffix_lengths_from_om` for byte-exact C parity.
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
            let ln_t_ii = crate::util::cmath::c_log_f64(t_ii as f64);
            if ln_t_ii == 0.0 {
                1.0
            } else {
                let v = crate::util::cmath::c_log_f64(BETA as f64 / t_mi as f64) / ln_t_ii;
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
/// floating-point results that propagate through p7_pli_extend_and_merge_windows,
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
    let nq = crate::simd::oprofile::nqf(m);
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
            let ln_t_ii = crate::util::cmath::c_log_f64(t_ii[k] as f64);
            if ln_t_ii == 0.0 {
                1.0
            } else {
                // C: `1 + (int)(log(BETA / t_mi[k]) / log(t_ii[k]))`.
                // C `(int)` truncates toward zero (not floor).
                let v = crate::util::cmath::c_log_f64(BETA / t_mi[k] as f64) / ln_t_ii;
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

/// Full C-matching variant: per-window extension uses
/// `max_length * (0.1 + prefix_lengths[k - length + 1])` for the left side
/// and `max_length * (0.1 + suffix_lengths[k])` for the right.
/// Mirrors the per-k branches in p7_pli_ExtendAndMergeWindows:482-487.
pub fn p7_pli_extend_and_merge_windows(
    windows: &mut Vec<HmmWindow>,
    max_length: usize,
    _target_len: usize,
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
        debug_assert!(w.k < suffix_lengths.len());
        debug_assert!(w.length <= w.k + 1);
        debug_assert!(w.k + 1 - w.length < prefix_lengths.len());

        let k = w.k;
        let prefix_k = w.k + 1 - w.length;
        let pre_ext_f = ml * (base_frac + prefix_lengths[prefix_k] as f64);
        let suf_ext_f = ml * (base_frac + suffix_lengths[k] as f64);

        let (window_start, window_end): (i64, i64);
        let tlen = w.target_len as i64;
        if w.needs_complement_extension_flip() {
            // Port of C p7_pli_ExtendAndMergeWindows complement branch
            // (hmmer/src/p7_pipeline.c:470-481): flip n into target-relative
            // coordinates, extend with prefix/suffix SWAPPED, then flip the
            // bounds back. This is only used for windows in the forward
            // concatenated-FM frame (fm_n >= 0); the per-segment RC-local Rust
            // path uses the non-complement branch below.
            //
            // C:
            //   n = target_len - n + 1;
            //   window_start = MAX(1, n - length - max_length*(0.1+suffix[k]));
            //   window_end   = MIN(target_len, n + max_length*(0.1+prefix[k-length+1]));
            //   tmp = window_end;
            //   window_end   = target_len - window_start;
            //   window_start = target_len - tmp;
            //   n = target_len - n + 1;
            let n_flipped = tlen - w.n as i64 + 1;
            let ws = (n_flipped - w.length as i64) as f64 - suf_ext_f;
            let ws = if ws < 1.0 { 1.0 } else { ws };
            let we = (n_flipped as f64) + pre_ext_f;
            let we = if we > tlen as f64 { tlen as f64 } else { we };
            let start_flipped = ws as i64;
            let end_flipped = we as i64;
            // flip the bounds back (C does NOT add the commented-out +1)
            window_end = tlen - start_flipped;
            window_start = tlen - end_flipped;
        } else {
            // Match C non-complement branch (p7_pipeline.c:485-486) and the
            // (int64_t)(n - double) / (int64_t)(n + length + double)
            // truncation toward zero.
            let start_f = (w.n as i64) as f64 - pre_ext_f;
            window_start = if start_f < 1.0 { 1 } else { start_f as i64 };
            let end_f = ((w.n + w.length) as i64) as f64 + suf_ext_f;
            let end_raw = if end_f < 1.0 { 1 } else { end_f as i64 };
            window_end = end_raw.min(tlen);
        }

        let window_start = window_start.max(1);
        let window_end = window_end.min(tlen).max(window_start);
        // C p7_pipeline.c:489-492: length, then fm_n -= (n - window_start), n.
        w.length = (window_end - window_start + 1) as usize;
        w.fm_n -= w.n as i64 - window_start;
        w.n = window_start as usize;
    }
    merge_windows_impl(windows, pct_overlap);
}

/// Merge adjacent windows whose overlap exceeds `pct_overlap` of the
/// shorter window's length. Walks the (sorted) `windows` list once and
/// folds each entry into the previous merged entry when the same-strand
/// overlap fraction crosses the threshold; otherwise pushes a new entry.
/// Score is promoted to the max of the two.
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

        // C p7_pipeline.c:505-507 requires BOTH complementarity AND id (the FM
        // segment id) to match before merging, so windows from different FM
        // segments on the same strand are never merged. id defaults to 0 in the
        // FASTA / per-segment path (one segment), so this adds no behavior
        // change there.
        if prev.complement == curr.complement && prev.id == curr.id && pct > pct_overlap {
            let new_start = prev.n.min(curr.n);
            let new_end = prev_end.max(curr_end);
            // C p7_pipeline.c:514-516: fm_n -= (n - window_start) before n is moved.
            prev.fm_n -= prev.n as i64 - new_start as i64;
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

/// Split windows longer than `max_window_len` into overlapping sub-windows for
/// numerical stability in Forward. Faithful port of C
/// p7_pli_postSSV_LongTarget's split loop (hmmer/src/p7_pipeline.c:1620-1634):
///
/// ```c
/// if (window.length > max_window_len) {
///   new_n   = window.n;
///   new_len = window.length;
///   window.length = max_window_len;        // trim the head
///   do {
///     int shift = max_window_len - overlap_len;
///     new_n   += shift;
///     new_len -= shift;
///     p7_hmmwindow_new(.., ESL_MIN(max_window_len, new_len), ..);
///   } while (new_len > max_window_len);
/// }
/// ```
///
/// LOW-1 (audit 02-nhmmer-longtarget): the C structure is a `do/while`, so it
/// ALWAYS emits at least one tail window per oversized input window — including
/// a 0-length (or, via C `int32_t` underflow, negative-length) tail when
/// `new_len` lands exactly on a `shift` multiple. The previous Rust code broke
/// out on `new_len == 0` BEFORE pushing, dropping that final tail. We reproduce
/// C exactly with signed `i64` arithmetic; non-positive `ESL_MIN(max_window_len,
/// new_len)` becomes a 0-length window that downstream `win_len < hmm.m` guards
/// drop, matching C (where such a window contributes nothing in later passes).
pub fn split_long_windows(
    windows: &[HmmWindow],
    max_window_len: usize,
    overlap_len: usize,
) -> Vec<HmmWindow> {
    let mut out: Vec<HmmWindow> = Vec::with_capacity(windows.len());
    for w in windows {
        if w.length <= max_window_len {
            out.push(w.clone());
            continue;
        }
        let mut head = w.clone();
        head.length = max_window_len;
        out.push(head);
        let mut new_n = w.n as i64;
        let mut new_len = w.length as i64;
        let shift = (max_window_len - overlap_len) as i64;
        loop {
            new_n += shift;
            new_len -= shift;
            let chunk = (max_window_len as i64).min(new_len);
            let fm_n = if w.fm_n >= 0 {
                w.fm_n + new_n - w.n as i64
            } else {
                -1
            };
            out.push(HmmWindow {
                n: new_n.max(0) as usize,
                k: 0,
                length: chunk.max(0) as usize,
                score: 0.0,
                target_len: w.target_len,
                complement: w.complement,
                id: w.id,
                fm_n,
                fm_start: w.fm_start,
                fm_bwt_len: w.fm_bwt_len,
            });
            if new_len <= max_window_len as i64 {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(n: usize, length: usize, complement: bool, id: i32) -> HmmWindow {
        HmmWindow {
            n,
            k: length,
            length,
            score: 0.0,
            target_len: 1_000_000,
            complement,
            id,
            fm_n: -1,
            fm_start: -1,
            fm_bwt_len: 0,
        }
    }

    // LOW-1: C's do/while always emits at least one tail per oversized window.
    #[test]
    fn split_long_windows_emits_tail_at_exact_shift_multiple() {
        let max = 80000;
        let overlap = 40000; // shift = 40000
        let w = win(1, max + (max - overlap), false, 0); // 120000
        let out = split_long_windows(&[w], max, overlap);
        assert!(
            out.len() >= 2,
            "expected head plus a tail, got {}",
            out.len()
        );
        assert_eq!(out[0].length, max);
        // new_len: 120000 - 40000 = 80000 (<=max) => final tail length 80000.
        assert_eq!(out[1].length, 80000);
    }

    #[test]
    fn split_long_windows_walks_multiple_tails() {
        let max = 80000;
        let overlap = 40000; // shift 40000
        let w = win(1, 120001, false, 0);
        let out = split_long_windows(&[w], max, overlap);
        // head=80000; iter1 new_len=80001 (>max) push 80000; iter2 new_len=40001
        // (<=max) push 40001, stop.
        assert_eq!(out[0].length, 80000);
        assert_eq!(out[1].length, 80000);
        assert_eq!(out[2].length, 40001);
        assert_eq!(out.len(), 3);

        let w2 = win(1, max + 2 * 40000, false, 0); // 160000
        let out2 = split_long_windows(&[w2], max, overlap);
        // head 80000; iter1 new_len 120000 push 80000; iter2 new_len 80000 push
        // 80000 stop.
        assert_eq!(out2.len(), 3);
        assert!(out2.iter().all(|w| w.length <= max));
    }

    // MED-2: merges require BOTH complementarity AND id to match.
    #[test]
    fn merge_respects_id_guard() {
        let mut ws = vec![win(100, 100, false, 0), win(110, 100, false, 1)];
        merge_windows_impl(&mut ws, 0.0);
        assert_eq!(
            ws.len(),
            2,
            "windows from different FM segments must not merge"
        );

        let mut ws2 = vec![win(100, 100, false, 7), win(110, 100, false, 7)];
        merge_windows_impl(&mut ws2, 0.0);
        assert_eq!(ws2.len(), 1, "same-segment overlapping windows must merge");

        let mut ws3 = vec![win(100, 100, false, 3), win(110, 100, true, 3)];
        merge_windows_impl(&mut ws3, 0.0);
        assert_eq!(ws3.len(), 2, "opposite-strand windows must not merge");
    }

    // MED-2: the complement extension flip only fires in the forward
    // concatenated-FM frame (fm_n >= 0); RC-local windows (fm_n=-1) do not flip.
    #[test]
    fn complement_flip_gated_on_fm_frame() {
        let mut rc_local = win(10, 5, true, 0);
        rc_local.fm_n = -1;
        assert!(!rc_local.needs_complement_extension_flip());

        let mut fm_frame = win(10, 5, true, 0);
        fm_frame.fm_n = 0;
        assert!(fm_frame.needs_complement_extension_flip());

        let mut fwd = win(10, 5, false, 0);
        fwd.fm_n = 0;
        assert!(!fwd.needs_complement_extension_flip());
    }

    #[test]
    fn p7_pli_extend_uses_window_target_len_like_c() {
        let mut w = win(40, 5, false, 0);
        w.target_len = 50;
        let mut windows = vec![w];
        let prefix = vec![0.0; 8];
        let suffix = vec![0.0; 8];

        p7_pli_extend_and_merge_windows(&mut windows, 10, 10, 0.0, &prefix, &suffix);

        assert_eq!(windows[0].n, 39);
        assert_eq!(windows[0].length, 8);
    }
}
