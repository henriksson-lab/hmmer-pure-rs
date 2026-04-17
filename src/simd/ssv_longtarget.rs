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
    let nq = (m + 15) / 16; // p7O_NQB
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
    let q_count = (m + 15) / 16; // number of SIMD vectors (NQB)

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

    // Null score: p7_bg_NullOne computes L * log(p1) + log(1-p1) where p1 = L/(L+1)
    // C uses om->max_length which was set by ReconfigMSVLength before this call.
    // We use the max_length parameter which the caller should set to om->max_length.
    let ml = if max_length > 0 { max_length as usize } else { m * 4 };
    let p1: f32 = ml as f32 / (ml as f32 + 1.0);
    let nullsc: f32 = ml as f32 * p1.ln() + (1.0f32 - p1).ln();

    // Use om->tjb_b directly (matching C line 324, 334, 385, 420)
    // C reconfigures om->tjb_b via ReconfigMSVLength before calling this function.
    // The tjb_b in om should already be set for max_length.
    let tjb_b = om.tjb_b;

    let sc_thresh_f = ((nullsc + (inv_p as f32 * std::f32::consts::LN_2) + 3.0) * om.scale_b)
        + om.base_b as f32
        + om.tec_b as f32
        + tjb_b as f32;
    let sc_thresh: u8 = if sc_thresh_f >= 255.0 {
        255
    } else if sc_thresh_f <= 0.0 {
        0
    } else {
        sc_thresh_f.ceil() as u8
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
            // Find which model state exceeded threshold
            let mut end = 0usize;
            let mut rem_sc = 0u8;

            for q in 0..q_count {
                let bytes: [u8; 16] = std::mem::transmute(dp[q]);
                for z in 0..16 {
                    let model_pos = q + 1 + z * q_count;
                    if model_pos <= m && bytes[z] >= sc_thresh && bytes[z] > rem_sc {
                        end = model_pos;
                        rem_sc = bytes[z];
                    }
                }
                // Reset DP row (will restart from xBv next iteration)
                dp[q] = _mm_setzero_si128();
            }

            if end == 0 {
                i += 1;
                continue;
            }

            // Trace back along diagonal to find start
            let mut start = end;
            let mut target_start = i;
            let mut target_end = i;
            let sc_val = rem_sc;
            // C line 385: while (rem_sc > om->base_b - om->tjb_b - om->tbm_b)
            let entry_cost = om.base_b.wrapping_sub(tjb_b).wrapping_sub(om.tbm_b);

            while rem_sc > entry_cost && start > 1 && target_start > 1 {
                let score_idx = start * kp + dsq[target_start] as usize;
                let emission = if score_idx < ssv_scores.len() {
                    ssv_scores[score_idx]
                } else {
                    0
                };
                // rem_sc -= bias_b - emission (in uint8 arithmetic)
                let delta = om.bias_b.wrapping_sub(emission);
                if delta > rem_sc {
                    break;
                }
                rem_sc -= delta;
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

            while k <= m && n <= l {
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
pub fn extend_and_merge_windows(
    windows: &mut Vec<HmmWindow>,
    max_length: usize,
    target_len: usize,
) {
    if windows.is_empty() {
        return;
    }

    // Extend each window
    let extension = max_length / 5; // ~0.2 * max_length
    for w in windows.iter_mut() {
        let start = if w.n > extension { w.n - extension } else { 1 };
        let end = (w.n + w.length + extension).min(target_len);
        w.length = end - start + 1;
        w.n = start;
    }

    // Merge overlapping windows
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

        // Check overlap
        if curr.n <= prev_end + 1 && curr.complement == prev.complement {
            // Merge
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
