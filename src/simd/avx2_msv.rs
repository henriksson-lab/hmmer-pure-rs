//! AVX2-optimized MSV filter (32x uint8 vectors).
//! Wider SIMD for ~2x speedup over SSE2 on supported CPUs.
#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::OProfile;

/// Number of AVX2 byte vectors needed to stripe a model of length M: ceil(M/32), min 2.
pub fn nqb_avx2(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 32) + 1)
}

/// AVX2 OProfile extension for MSV byte scores.
pub struct OProfileAvx2 {
    /// MSV byte scores: `rbv[residue][q][32 bytes]`
    pub rbv: Vec<Vec<[u8; 32]>>,
    pub tbm_b: u8,
    pub tec_b: u8,
    pub tjb_b: u8,
    pub scale_b: f32,
    pub base_b: u8,
    pub bias_b: u8,
    pub m: usize,
    pub abc_kp: usize,
}

impl OProfileAvx2 {
    /// Build an AVX2 MSV profile by restriping the SSE2 `OProfile` byte scores into
    /// 32-way striped vectors.
    pub fn from_oprofile(om: &OProfile) -> Self {
        let m = om.m;
        let nq = nqb_avx2(m);
        let kp = om.abc_kp;

        // Restripe: interleave 32 positions per vector instead of 16
        let mut rbv = vec![vec![[0u8; 32]; nq]; kp];
        for x in 0..kp {
            for q in 0..nq {
                let mut tmp = [0u8; 32];
                for z in 0..32 {
                    let node = q + 1 + z * nq;
                    if node <= m {
                        // Look up the byte score from SSE2 profile
                        let sse_q = (node - 1) % crate::simd::oprofile::nqb(m);
                        let sse_z = (node - 1) / crate::simd::oprofile::nqb(m);
                        if sse_z < 16 && sse_q < om.rbv[x].len() {
                            tmp[z] = om.rbv[x][sse_q][sse_z];
                        } else {
                            tmp[z] = 255;
                        }
                    } else {
                        tmp[z] = 255;
                    }
                }
                rbv[x][q] = tmp;
            }
        }

        OProfileAvx2 {
            rbv,
            tbm_b: om.tbm_b,
            tec_b: om.tec_b,
            tjb_b: om.tjb_b,
            scale_b: om.scale_b,
            base_b: om.base_b,
            bias_b: om.bias_b,
            m,
            abc_kp: kp,
        }
    }
}

/// Result of the AVX2 MSV filter: either a finite score or a saturating overflow.
pub enum Avx2MsvResult {
    Ok(f32),
    Overflow,
}

/// AVX2 variant of the MSV filter (C: `p7_MSVFilter`), using 32x uint8 vectors.
///
/// Calculates an approximation of the MSV score for digital sequence `dsq` of length
/// `l` using AVX2-striped profile `om`. Score may overflow on extremely high-scoring
/// sequences but will not underflow. The MSV filter assumes multihit local mode and
/// uses its own special state transition scores rather than the scores in the profile.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_msv_filter(dsq: &[Dsq], l: usize, om: &OProfileAvx2) -> Avx2MsvResult {
    let q_count = nqb_avx2(om.m);

    let mut dp: Vec<__m256i> = vec![_mm256_setzero_si256(); q_count];

    let biasv = _mm256_set1_epi8(om.bias_b as i8);
    let basev = _mm256_set1_epi8(om.base_b as i8);
    let ceilingv = _mm256_cmpeq_epi8(biasv, biasv); // all 0xFF

    let tjbm = om.tjb_b.wrapping_add(om.tbm_b);
    let tjbmv = _mm256_set1_epi8(tjbm as i8);
    let tecv = _mm256_set1_epi8(om.tec_b as i8);

    let mut xjv = _mm256_setzero_si256();
    let mut xbv = _mm256_subs_epu8(basev, tjbmv);

    for i in 1..=l {
        // C indexes `om->rbv[dsq[i]]` unconditionally for every residue 1..L
        // (msvfilter.c:134). rbv is filled for all Kp codes (from_oprofile builds
        // it `for x in 0..kp`), and every valid digital code is < Kp, so the row
        // always exists and the recurrence must advance.
        let xi = dsq[i] as usize;
        let rsc = &om.rbv[xi];

        let mut xev = _mm256_setzero_si256();

        // Full-width 1-byte left shift across the whole 256-bit register
        // (HMMER's "right shift": result[j] = v[j-1], result[0] = 0), matching the
        // SSE `_mm_slli_si128(dp[Q-1], 1)`. Plain `_mm256_slli_si256::<1>` shifts each
        // 128-bit lane independently and drops the byte that must carry from lane-0
        // byte 15 into lane-1 byte 16, so we emulate a full-width shift.
        //
        // `permute2x128::<0x08>` yields perm = [lane0 = 0, lane1 = v's lane0].
        // `alignr_epi8::<15>(v, perm)` then takes bytes 15..30 of (v_lane : perm_lane)
        // per lane, producing:
        //   lane0 = [0,   v0..v14]   (byte 0 sentinel = 0)
        //   lane1 = [v15, v16..v30]  (carry v15 crosses the lane boundary; v31 dropped)
        let last = dp[q_count - 1];
        let perm = _mm256_permute2x128_si256::<0x08>(last, last);
        let mut mpv = _mm256_alignr_epi8::<15>(last, perm);

        for q in 0..q_count {
            let mut sv = _mm256_max_epu8(mpv, xbv);
            sv = _mm256_adds_epu8(sv, biasv);
            let rsc_v = _mm256_loadu_si256(rsc[q].as_ptr() as *const __m256i);
            sv = _mm256_subs_epu8(sv, rsc_v);
            xev = _mm256_max_epu8(xev, sv);

            mpv = dp[q];
            dp[q] = sv;
        }

        // Overflow test
        let tempv = _mm256_adds_epu8(xev, biasv);
        let tempv = _mm256_cmpeq_epi8(tempv, ceilingv);
        let cmp = _mm256_movemask_epi8(tempv);
        if cmp != 0 {
            return Avx2MsvResult::Overflow;
        }

        // Horizontal max across all 32 bytes of xev
        // Reduce 256→128→64→32→16→8 bits
        let hi128 = _mm256_extracti128_si256::<1>(xev);
        let lo128 = _mm256_castsi256_si128(xev);
        let max128 = _mm_max_epu8(hi128, lo128);
        // Now reduce 128-bit (same as SSE2)
        let temp = _mm_shuffle_epi32::<{ super::shuffle_mask(2, 3, 0, 1) }>(max128);
        let max128 = _mm_max_epu8(max128, temp);
        let temp = _mm_shuffle_epi32::<{ super::shuffle_mask(0, 1, 2, 3) }>(max128);
        let max128 = _mm_max_epu8(max128, temp);
        let temp = _mm_shufflelo_epi16::<{ super::shuffle_mask(2, 3, 0, 1) }>(max128);
        let max128 = _mm_max_epu8(max128, temp);
        let temp = _mm_srli_si128::<1>(max128);
        let max128 = _mm_max_epu8(max128, temp);
        // Broadcast
        let xev_scalar = _mm_extract_epi16::<0>(max128) as u16 as u8;
        xev = _mm256_set1_epi8(xev_scalar as i8);

        xev = _mm256_subs_epu8(xev, tecv);
        xjv = _mm256_max_epu8(xjv, xev);
        xbv = _mm256_max_epu8(basev, xjv);
        xbv = _mm256_subs_epu8(xbv, tjbmv);
    }

    let xj = _mm_extract_epi16::<0>(_mm256_castsi256_si128(xjv)) as u16 as u8;

    let mut sc = (xj.wrapping_sub(om.tjb_b) as f32) - om.base_b as f32;
    sc /= om.scale_b;
    sc -= 3.0;

    Avx2MsvResult::Ok(sc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::profile::*;
    use std::path::Path;

    /// Verifies the in-loop cross-lane byte up-shift is a true whole-vector shift:
    /// `[v0..v31] -> [0, v0..v30]` (v31 dropped, v15 carried across the lane boundary).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_msv_cross_lane_shift() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        unsafe fn shift(v: __m256i) -> __m256i {
            let perm = _mm256_permute2x128_si256::<0x08>(v, v);
            _mm256_alignr_epi8::<15>(v, perm)
        }
        unsafe {
            let mut input = [0u8; 32];
            for (z, b) in input.iter_mut().enumerate() {
                *b = z as u8;
            }
            let v = _mm256_loadu_si256(input.as_ptr() as *const __m256i);
            let shifted = shift(v);
            let mut out = [0u8; 32];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, shifted);
            let mut expect = [0u8; 32];
            for z in 1..32 {
                expect[z] = (z - 1) as u8; // result[z] = v[z-1]; result[0] = 0
            }
            assert_eq!(out, expect);
        }
    }

    /// Cross-checks the AVX2 MSV filter against the SSE2 reference within 1e-3.
    #[test]
    fn test_avx2_msv_matches_sse() {
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
        let om = crate::simd::oprofile::OProfile::convert(&gm);
        let om_avx2 = OProfileAvx2::from_oprofile(&om);
        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2;
        // Compare the full MSV DP directly (bypass the SSV shortcut) so we exercise
        // the AVX2 striped recurrence and its cross-lane shift.
        let mut scratch = vec![unsafe { _mm_setzero_si128() }; 1];
        let sse = match unsafe {
            crate::simd::msv_filter::p7_msv_filter_dp_only_rust_helper(&dsq, l, &om, &mut scratch)
        } {
            crate::simd::msv_filter::MsvResult::Ok(sc) => sc,
            crate::simd::msv_filter::MsvResult::Overflow => return,
        };
        let avx = match unsafe { avx2_msv_filter(&dsq, l, &om_avx2) } {
            Avx2MsvResult::Ok(sc) => sc,
            Avx2MsvResult::Overflow => return,
        };
        assert!(
            (sse - avx).abs() < 1.0e-3,
            "SSE MSV {sse} and AVX2 MSV {avx} differ"
        );
    }

    /// Smoke test: AVX2 MSV filter returns a finite score (or overflow) on a small model.
    #[test]
    fn test_avx2_msv_basic() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available, skipping test");
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
        let om = crate::simd::oprofile::OProfile::convert(&gm);
        let om_avx2 = OProfileAvx2::from_oprofile(&om);

        let dsq = abc.digitize(b"AAAAAAAAAAGGGGGGGGGG");
        let result = unsafe { avx2_msv_filter(&dsq, 20, &om_avx2) };
        match result {
            Avx2MsvResult::Ok(sc) => assert!(sc.is_finite()),
            Avx2MsvResult::Overflow => {} // also acceptable
        }
    }
}
