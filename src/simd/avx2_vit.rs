//! AVX2-optimized Viterbi filter (16x int16 vectors).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::profile::*;
use crate::simd::oprofile::*;

/// Number of AVX2 vectors for Viterbi: ceil(M/16), min 2.
pub fn nqw_avx2(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 16) + 1)
}

/// AVX2 OProfile extension for Viterbi word scores.
pub struct OProfileAvx2Vit {
    pub rwv: Vec<Vec<[i16; 16]>>,
    pub twv: Vec<[i16; 16]>,
    pub xw: [[i16; P7O_NXTRANS]; P7O_NXSTATES],
    pub scale_w: f32,
    pub base_w: i16,
    pub ddbound_w: i16,
    pub m: usize,
    pub abc_kp: usize,
}

impl OProfileAvx2Vit {
    /// Build AVX2 Viterbi profile by restriping from SSE2 OProfile.
    pub fn from_oprofile(om: &OProfile) -> Self {
        let m = om.m;
        let nq = nqw_avx2(m);
        let kp = om.abc_kp;
        let nq_sse = nqw(m);

        // Restripe word emission scores: 16 positions per vector
        let mut rwv = vec![vec![[0i16; 16]; nq]; kp];
        for x in 0..kp {
            for q in 0..nq {
                let mut tmp = [0i16; 16];
                for z in 0..16 {
                    let node = q + 1 + z * nq;
                    if node <= m {
                        let sse_q = (node - 1) % nq_sse;
                        let sse_z = (node - 1) / nq_sse;
                        if sse_z < 8 && sse_q < om.rwv[x].len() {
                            tmp[z] = om.rwv[x][sse_q][sse_z];
                        } else {
                            tmp[z] = -32768;
                        }
                    } else {
                        tmp[z] = -32768;
                    }
                }
                rwv[x][q] = tmp;
            }
        }

        // Restripe transition scores
        let mut twv = vec![[0i16; 16]; 8 * nq];
        let mut j = 0;
        for qi in 0..nq {
            let ki = qi + 1;
            let trans_specs: [(usize, usize, i16); 7] = [
                (P7P_BM, ki.wrapping_sub(1), 0),
                (P7P_MM, ki.wrapping_sub(1), 0),
                (P7P_IM, ki.wrapping_sub(1), 0),
                (P7P_DM, ki.wrapping_sub(1), 0),
                (P7P_MD, ki, 0),
                (P7P_MI, ki, 0),
                (P7P_II, ki, -1),
            ];
            for &(tg, kb, maxval) in &trans_specs {
                let mut tmp = [0i16; 16];
                for z in 0..16 {
                    let node = kb + z * nq;
                    let val = if node < m {
                        crate::simd::oprofile::wordify_pub(om.scale_w, om.tsc_at(node, tg))
                    } else {
                        -32768
                    };
                    tmp[z] = if val >= maxval { maxval } else { val };
                }
                twv[j] = tmp;
                j += 1;
            }
        }
        // DD at end
        for qi in 0..nq {
            let ki = qi + 1;
            let mut tmp = [0i16; 16];
            for z in 0..16 {
                let node = ki + z * nq;
                tmp[z] = if node < m {
                    crate::simd::oprofile::wordify_pub(om.scale_w, om.tsc_at(node, P7P_DD))
                } else {
                    -32768
                };
            }
            twv[j] = tmp;
            j += 1;
        }

        OProfileAvx2Vit {
            rwv,
            twv,
            xw: om.xw,
            scale_w: om.scale_w,
            base_w: om.base_w,
            ddbound_w: om.ddbound_w,
            m,
            abc_kp: kp,
        }
    }
}

pub enum Avx2VitResult {
    Ok(f32),
    Overflow,
}

/// AVX2 Viterbi filter using 16x int16 vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_viterbi_filter(dsq: &[Dsq], l: usize, om: &OProfileAvx2Vit) -> Avx2VitResult {
    let q_count = nqw_avx2(om.m);
    let nscells = 3;
    let mut dp: Vec<__m256i> = vec![_mm256_set1_epi16(-32768); q_count * nscells];

    macro_rules! mmx { ($q:expr) => { dp[$q * nscells + 0] }; }
    macro_rules! dmx { ($q:expr) => { dp[$q * nscells + 1] }; }
    macro_rules! imx { ($q:expr) => { dp[$q * nscells + 2] }; }

    let neg_inf_16 = _mm256_set1_epi16(-32768);
    // For cross-lane shift: need -32768 in lowest word only
    let neg_inf_v = _mm256_insert_epi16::<0>(
        _mm256_setzero_si256(), -32768
    );

    let mut xn: i16 = om.base_w;
    let mut xb: i16 = xn.saturating_add(om.xw[P7O_N][P7O_MOVE]);
    let mut xj: i16 = -32768;
    let mut xc: i16 = -32768;

    for i in 1..=l {
        let xi = dsq[i] as usize;
        if xi >= om.abc_kp { continue; }
        let rsc = &om.rwv[xi];

        let mut dcv = neg_inf_16;
        let mut xev = neg_inf_16;
        let mut dmaxv = neg_inf_16;
        let xbv = _mm256_set1_epi16(xb);

        // Shift by 1 word (2 bytes) — AVX2 slli_si256 is per-lane
        // Use permute + alignr for cross-lane shift
        let last_m = mmx!(q_count - 1);
        let last_d = dmx!(q_count - 1);
        let last_i = imx!(q_count - 1);
        let mut mpv = cross_lane_shift_epi16(last_m);
        let mut dpv = cross_lane_shift_epi16(last_d);
        let mut ipv = cross_lane_shift_epi16(last_i);

        let mut tsc_idx = 0;
        for q in 0..q_count {
            let tsc_bm = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i); tsc_idx += 1;
            let tsc_mm = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i); tsc_idx += 1;
            let tsc_im = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i); tsc_idx += 1;
            let tsc_dm = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i); tsc_idx += 1;

            let mut sv = _mm256_adds_epi16(xbv, tsc_bm);
            sv = _mm256_max_epi16(sv, _mm256_adds_epi16(mpv, tsc_mm));
            sv = _mm256_max_epi16(sv, _mm256_adds_epi16(ipv, tsc_im));
            sv = _mm256_max_epi16(sv, _mm256_adds_epi16(dpv, tsc_dm));

            let rsc_v = _mm256_loadu_si256(rsc[q].as_ptr() as *const __m256i);
            sv = _mm256_adds_epi16(sv, rsc_v);
            xev = _mm256_max_epi16(xev, sv);

            mpv = mmx!(q);
            dpv = dmx!(q);
            ipv = imx!(q);

            mmx!(q) = sv;
            dmx!(q) = dcv;

            let tsc_md = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i); tsc_idx += 1;
            dcv = _mm256_adds_epi16(sv, tsc_md);
            dmaxv = _mm256_max_epi16(dcv, dmaxv);

            let tsc_mi = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i); tsc_idx += 1;
            let tsc_ii = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i); tsc_idx += 1;
            imx!(q) = _mm256_max_epi16(
                _mm256_adds_epi16(mpv, tsc_mi),
                _mm256_adds_epi16(ipv, tsc_ii),
            );
        }

        // Horizontal max of xev (16x i16 → scalar)
        let xe = hmax_epi16_avx2(xev);
        if xe >= 32767 { return Avx2VitResult::Overflow; }

        xn = xn.saturating_add(om.xw[P7O_N][P7O_LOOP]);
        xc = (xc.saturating_add(om.xw[P7O_C][P7O_LOOP]))
            .max(xe.saturating_add(om.xw[P7O_E][P7O_MOVE]));
        xj = (xj.saturating_add(om.xw[P7O_J][P7O_LOOP]))
            .max(xe.saturating_add(om.xw[P7O_E][P7O_LOOP]));
        xb = (xj.saturating_add(om.xw[P7O_J][P7O_MOVE]))
            .max(xn.saturating_add(om.xw[P7O_N][P7O_MOVE]));

        // Lazy DD evaluation
        let dmax = hmax_epi16_avx2(dmaxv);
        if (dmax as i32) + (om.ddbound_w as i32) > (xb as i32) {
            dcv = cross_lane_shift_epi16(dcv);
            let dd_offset = 7 * q_count;
            for q in 0..q_count {
                dmx!(q) = _mm256_max_epi16(dcv, dmx!(q));
                let tdd = _mm256_loadu_si256(om.twv[dd_offset + q].as_ptr() as *const __m256i);
                dcv = _mm256_adds_epi16(dmx!(q), tdd);
            }
            loop {
                dcv = cross_lane_shift_epi16(dcv);
                let mut any = false;
                for q in 0..q_count {
                    let cmp = _mm256_cmpgt_epi16(dcv, dmx!(q));
                    if _mm256_movemask_epi8(cmp) != 0 {
                        any = true;
                        dmx!(q) = _mm256_max_epi16(dcv, dmx!(q));
                        let tdd = _mm256_loadu_si256(om.twv[dd_offset + q].as_ptr() as *const __m256i);
                        dcv = _mm256_adds_epi16(dmx!(q), tdd);
                    } else {
                        break;
                    }
                }
                if !any { break; }
            }
        } else {
            dcv = cross_lane_shift_epi16(dcv);
            dmx!(0) = _mm256_or_si256(dcv, neg_inf_v);
        }
    }

    if xc > -32768 {
        let mut sc = xc as f32 + om.xw[P7O_C][P7O_MOVE] as f32 - om.base_w as f32;
        sc /= om.scale_w;
        sc -= 3.0;
        Avx2VitResult::Ok(sc)
    } else {
        Avx2VitResult::Ok(f32::NEG_INFINITY)
    }
}

/// Cross-lane right-shift by 1 word (2 bytes) for AVX2.
/// [a0,a1,...,a15] → [-inf,a0,a1,...,a14]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cross_lane_shift_epi16(v: __m256i) -> __m256i {
    // alignr within each lane, then fix the cross-lane boundary
    let hi = _mm256_permute2x128_si256::<0x08>(v, v); // lane0 = zero, lane1 = old lane0
    let shifted = _mm256_alignr_epi8::<2>(v, hi); // shift right by 2 bytes across lanes
    // Set lowest word to -32768
    _mm256_insert_epi16::<0>(shifted, -32768)
}

/// Horizontal max of 16x i16 in AVX2 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hmax_epi16_avx2(v: __m256i) -> i16 {
    let hi = _mm256_extracti128_si256::<1>(v);
    let lo = _mm256_castsi256_si128(v);
    let max128 = _mm_max_epi16(hi, lo);
    let max128 = _mm_max_epi16(max128, _mm_shuffle_epi32::<0x4E>(max128));
    let max128 = _mm_max_epi16(max128, _mm_shufflelo_epi16::<0x4E>(max128));
    let max128 = _mm_max_epi16(max128, _mm_srli_epi32::<16>(max128));
    _mm_cvtsi128_si32(max128) as i16
}
