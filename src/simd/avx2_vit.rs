//! AVX2-optimized Viterbi filter (16x int16 vectors).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::profile::*;
use crate::simd::oprofile::*;

/// Number of AVX2 16-bit-word vectors needed to stripe a model of length M:
/// ceil(M/16), min 2.
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
    /// Build an AVX2 Viterbi profile by restriping the SSE2 `OProfile` word-score and
    /// transition tables into 16-way striped vectors.
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

/// Result of the AVX2 Viterbi filter: either a finite score or a saturating overflow.
pub enum Avx2VitResult {
    Ok(f32),
    Overflow,
}

/// AVX2 variant of the Viterbi filter (C: `p7_ViterbiFilter`), using 16x int16 vectors.
///
/// Calculates an approximation of the Viterbi score in nats for digital sequence `dsq`
/// of length `l` using optimized profile `om`. Score may overflow on extremely
/// high-scoring sequences but will not underflow. The model must be in a local
/// alignment mode (the only mode that guarantees the limited dynamic range needed for
/// reduced-precision signed-word arithmetic).
///
/// Striped SIMD Viterbi after Farrar (2007), in 16-bit signed-word precision.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_viterbi_filter(dsq: &[Dsq], l: usize, om: &OProfileAvx2Vit) -> Avx2VitResult {
    let q_count = nqw_avx2(om.m);
    let nscells = 3;
    let mut dp: Vec<__m256i> = vec![_mm256_set1_epi16(-32768); q_count * nscells];

    macro_rules! mmx {
        ($q:expr) => {
            dp[$q * nscells + 0]
        };
    }
    macro_rules! dmx {
        ($q:expr) => {
            dp[$q * nscells + 1]
        };
    }
    macro_rules! imx {
        ($q:expr) => {
            dp[$q * nscells + 2]
        };
    }

    let neg_inf_16 = _mm256_set1_epi16(-32768);
    // For cross-lane shift: need -32768 in lowest word only
    let neg_inf_v = _mm256_insert_epi16::<0>(_mm256_setzero_si256(), -32768);

    let mut xn: i16 = om.base_w;
    let mut xb: i16 = add_i16(xn, om.xw[P7O_N][P7O_MOVE]);
    let mut xj: i16 = -32768;
    let mut xc: i16 = -32768;

    for i in 1..=l {
        // C indexes `om->rwv[dsq[i]]` unconditionally (vitfilter.c:127): rwv is
        // filled for all Kp codes (from_oprofile builds it `for x in 0..kp`), and
        // every valid digital code is < Kp, so the row always exists and the
        // recurrence (plus per-row special-state updates) must advance.
        let xi = dsq[i] as usize;
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
            let tsc_bm = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i);
            tsc_idx += 1;
            let tsc_mm = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i);
            tsc_idx += 1;
            let tsc_im = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i);
            tsc_idx += 1;
            let tsc_dm = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i);
            tsc_idx += 1;

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

            let tsc_md = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i);
            tsc_idx += 1;
            dcv = _mm256_adds_epi16(sv, tsc_md);
            dmaxv = _mm256_max_epi16(dcv, dmaxv);

            let tsc_mi = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i);
            tsc_idx += 1;
            let tsc_ii = _mm256_loadu_si256(om.twv[tsc_idx].as_ptr() as *const __m256i);
            tsc_idx += 1;
            imx!(q) = _mm256_max_epi16(
                _mm256_adds_epi16(mpv, tsc_mi),
                _mm256_adds_epi16(ipv, tsc_ii),
            );
        }

        // Horizontal max of xev (16x i16 → scalar)
        let xe = hmax_epi16_avx2(xev);
        if xe >= 32767 {
            return Avx2VitResult::Overflow;
        }

        // Scalar special-state updates use wrapping int16 arithmetic to match the
        // SSE port (vit_filter.rs `add_i16`) and C (vitfilter.c:177-180), which add
        // int16+int16 promoted to int and truncate on store. Saturating add would
        // diverge at the -32768 "-inf" sentinel plus a negative loop cost.
        xn = add_i16(xn, om.xw[P7O_N][P7O_LOOP]);
        xc = add_i16(xc, om.xw[P7O_C][P7O_LOOP]).max(add_i16(xe, om.xw[P7O_E][P7O_MOVE]));
        xj = add_i16(xj, om.xw[P7O_J][P7O_LOOP]).max(add_i16(xe, om.xw[P7O_E][P7O_LOOP]));
        xb = add_i16(xj, om.xw[P7O_J][P7O_MOVE]).max(add_i16(xn, om.xw[P7O_N][P7O_MOVE]));

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
            // C `do { ... } while (q == Q)`: the outer loop repeats only when the
            // inner q-loop ran to completion (never broke early). Mirror the SSE
            // port's `broke` flag rather than an "any update happened" flag — those
            // differ when the inner loop breaks partway (some q updated but q != Q),
            // in which case C does NOT repeat. (vitfilter.c:215-225)
            loop {
                dcv = cross_lane_shift_epi16(dcv);
                let mut broke = false;
                for q in 0..q_count {
                    let cmp = _mm256_cmpgt_epi16(dcv, dmx!(q));
                    if _mm256_movemask_epi8(cmp) == 0 {
                        broke = true;
                        break;
                    }
                    dmx!(q) = _mm256_max_epi16(dcv, dmx!(q));
                    let tdd =
                        _mm256_loadu_si256(om.twv[dd_offset + q].as_ptr() as *const __m256i);
                    dcv = _mm256_adds_epi16(dmx!(q), tdd);
                }
                if broke {
                    break;
                }
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

/// Wrapping i16 addition done via i32 intermediates, matching the SSE port's
/// `add_i16` and C's int16+int16 scalar special-state arithmetic (wrap on store,
/// not saturate).
#[inline(always)]
fn add_i16(a: i16, b: i16) -> i16 {
    (a as i32 + b as i32) as i16
}

/// Cross-lane right-shift by 1 16-bit word for AVX2 (helper, Rust-only).
/// Transforms `[a0,a1,...,a15]` into `[-inf,a0,a1,...,a14]`; needed because
/// `_mm256_slli_si256` only shifts within 128-bit lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cross_lane_shift_epi16(v: __m256i) -> __m256i {
    // We want, for input [a0..a15], the result [a0..a14] shifted up by one word with
    // the previous lane's top word carried across the 128-bit boundary, i.e.
    // [_, a0, a1, ..., a14] (a15 drops). This mirrors the SSE `_mm_slli_si128(x, 2)`
    // one-word left shift (in little-endian word terms) applied across all 256 bits.
    //
    // `perm` places zero in lane0 and old-lane0 in lane1. `_mm256_alignr_epi8::<14>(a, b)`
    // per lane takes bytes 14..30 of (a_lane:b_lane) = `b_lane[14..16] ++ a_lane[0..14]`,
    // a 2-byte (1-word) up-shift carrying in the prior lane's top word. With a = v,
    // b = perm: lane0 -> [0, a0..a6], lane1 -> [a7, a8..a14] (a7 carried from lane0).
    let perm = _mm256_permute2x128_si256::<0x08>(v, v); // lane0 = zero, lane1 = old lane0
    let shifted = _mm256_alignr_epi8::<14>(v, perm);
    // Set lowest word to the -inf sentinel, matching the SSE `_mm_or_si128(x, negInfv)`.
    _mm256_insert_epi16::<0>(shifted, -32768)
}

/// Horizontal maximum of the 16 signed 16-bit words in an AVX2 register (helper).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use std::path::Path;

    /// Verifies `cross_lane_shift_epi16` is a true whole-vector word up-shift across
    /// the 128-bit lane boundary: `[a0..a15] -> [-32768, a0..a14]` (a15 dropped).
    #[test]
    fn test_avx2_vit_cross_lane_shift() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        unsafe {
            let v = _mm256_set_epi16(
                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            );
            let shifted = cross_lane_shift_epi16(v);
            let mut out = [0i16; 16];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, shifted);
            assert_eq!(
                out,
                [-32768, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            );
        }
    }

    /// Cross-checks the AVX2 Viterbi filter against the SSE2 reference within 1e-3.
    #[test]
    fn test_avx2_vit_matches_sse() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        let abc = Alphabet::new(hmm.abc_type);
        let bg = Bg::new(&abc);
        let mut gm = Profile::new(hmm.m, &abc);
        profile_config(&hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);
        let avx_om = OProfileAvx2Vit::from_oprofile(&om);
        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2;
        let sse = match unsafe { crate::simd::vit_filter::viterbi_filter(&dsq, l, &om) } {
            crate::simd::vit_filter::VitResult::Ok(sc) => sc,
            crate::simd::vit_filter::VitResult::Overflow => return,
        };
        let avx = match unsafe { avx2_viterbi_filter(&dsq, l, &avx_om) } {
            Avx2VitResult::Ok(sc) => sc,
            Avx2VitResult::Overflow => return,
        };
        assert!(
            (sse - avx).abs() < 1.0e-3,
            "SSE Viterbi {sse} and AVX2 Viterbi {avx} differ"
        );
    }
}
