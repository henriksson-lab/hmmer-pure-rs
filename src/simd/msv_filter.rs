//! SSE2-optimized MSV filter.
//! Direct port of impl_sse/msvfilter.c p7_MSVFilter().

use crate::alphabet::Dsq;
use crate::simd::oprofile::{nqb, OProfile};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Result of the MSV filter: either a finite score or a saturating overflow.
pub enum MsvResult {
    /// Sequence passed filter, score is returned
    Ok(f32),
    /// Score overflowed (extremely high-scoring hit)
    Overflow,
}

/// Calculates the MSV score, vewy vewy fast, in limited precision (C: `p7_MSVFilter`).
///
/// Computes an approximation of the MSV score for digital sequence `dsq` of length `l`
/// using optimized profile `om`. Returns the estimated MSV score in nats, or `Overflow`
/// for very high-scoring sequences (the score may overflow but will not underflow).
///
/// The model may be in any mode because only its match emission scores are used; the
/// MSV filter inherently assumes a multihit local mode and uses its own special state
/// transition scores rather than the scores in the profile.
///
/// # Safety
/// Requires SSE2 support. Caller must verify CPU support.
#[target_feature(enable = "sse2")]
pub unsafe fn msv_filter(dsq: &[Dsq], l: usize, om: &OProfile) -> MsvResult {
    let q_count = nqb(om.m);

    // Working DP row (one row of Q vectors)
    let mut dp: Vec<__m128i> = vec![_mm_setzero_si128(); q_count];
    msv_filter_with_scratch(dsq, l, om, &mut dp)
}

/// SSE2 MSV filter variant that reuses a caller-owned single DP row (C: `p7_MSVFilter`).
///
/// Identical math to [`msv_filter`] but skips the per-call allocation by reusing the
/// caller's `dp` vector. Keep tracehash builds on `msv_filter()` unless a trace proves
/// this allocation change is harmless for exact instrumentation-sensitive comparisons.
#[target_feature(enable = "sse2")]
pub unsafe fn msv_filter_with_scratch(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    dp: &mut Vec<__m128i>,
) -> MsvResult {
    // C p7_MSVFilter first tries the highly-optimized SSV filter; if it returns
    // a definitive result (eslOK score or eslERANGE overflow) we return it
    // directly. Only when SSV returns eslENORESULT do we run the full MSV DP.
    // (msvfilter.c:102-104)
    //
    // LANDMINE (see TODO.md "Known Bad Or Neutral Experiments"): many earlier
    // attempts to add a general SSV pre-filter into this hot MSV path were
    // REVERTED because carrying SSV code/`sbv` storage here perturbed downstream
    // exact trace hashes (score_domain_null2, define_domains_summary, pipeline
    // score probes) or were slower. `ssv_filter_q17` is the *accepted* narrow
    // shortcut: it is restricted to the reference shape (Q=17, band widths 8/9),
    // returns NoResult on any unsupported shape, and is tracehash-parity verified.
    // Do NOT broaden this into a generic SSV port without re-verifying trace parity.
    match super::ssv_filter::ssv_filter_q17(dsq, l, om) {
        super::ssv_filter::SsvResult::Ok(sc) => return MsvResult::Ok(sc),
        super::ssv_filter::SsvResult::Overflow => return MsvResult::Overflow,
        super::ssv_filter::SsvResult::NoResult => {}
    }

    msv_filter_dp_only(dsq, l, om, dp)
}

/// Runs only the striped MSV dynamic-programming recurrence (C: the body of
/// `p7_MSVFilter` after the `p7_SSVFilter` shortcut), without the SSV pre-filter.
///
/// This is the faithful MSV DP that C falls back to when SSV returns
/// `eslENORESULT`. It is split out so callers (and AVX2 cross-check tests) can
/// exercise the DP directly without the SSV short-circuit.
///
/// # Safety
/// Requires SSE2 support.
#[target_feature(enable = "sse2")]
pub unsafe fn msv_filter_dp_only(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    dp: &mut Vec<__m128i>,
) -> MsvResult {
    let q_count = nqb(om.m);
    let zerov = _mm_setzero_si128();
    dp.resize(q_count, zerov);
    for v in dp.iter_mut() {
        *v = zerov;
    }

    let biasv = _mm_set1_epi8(om.bias_b as i8);
    let basev = _mm_set1_epi8(om.base_b as i8);
    let ceilingv = _mm_cmpeq_epi8(biasv, biasv); // all 0xFF

    let tjbm = om.tjb_b.wrapping_add(om.tbm_b);
    let tjbmv = _mm_set1_epi8(tjbm as i8);
    let tecv = _mm_set1_epi8(om.tec_b as i8);

    let mut xjv = _mm_setzero_si128();
    let mut xbv = _mm_subs_epu8(basev, tjbmv);

    for i in 1..=l {
        // C indexes `om->rbv[dsq[i]]` unconditionally for every residue 1..L:
        // rbv is filled for all Kp codes (oprofile.rs builds it `for x in 0..kp`,
        // covering canonical + degenerate + gap/missing), and every valid digital
        // code is < Kp, so the row always exists and the recurrence must advance.
        // (msvfilter.c:134)
        let xi = dsq[i] as usize;
        let rsc_ptr = om.rbv.get_unchecked(xi).as_ptr();
        let dp_ptr = dp.as_mut_ptr();

        let mut xev = _mm_setzero_si128();

        // Right shift by 1 byte (shift in zero = -infinity in offset arithmetic)
        let mut mpv = _mm_slli_si128::<1>(*dp_ptr.add(q_count - 1));

        for q in 0..q_count {
            // Calculate new M(i,q)
            let mut sv = _mm_max_epu8(mpv, xbv);
            sv = _mm_adds_epu8(sv, biasv);
            // Load emission score and subtract (in offset arithmetic, subtraction = adding score)
            let rsc_v = _mm_loadu_si128((*rsc_ptr.add(q)).as_ptr() as *const __m128i);
            sv = _mm_subs_epu8(sv, rsc_v);
            xev = _mm_max_epu8(xev, sv);

            let cell = dp_ptr.add(q);
            mpv = *cell;
            *cell = sv;
        }

        // Test for overflow
        let tempv = _mm_adds_epu8(xev, biasv);
        let tempv = _mm_cmpeq_epi8(tempv, ceilingv);
        let cmp = _mm_movemask_epi8(tempv);
        if cmp != 0 {
            return MsvResult::Overflow;
        }

        // Horizontal max across the xEv vector
        let mut tempv = _mm_shuffle_epi32::<{ shuffle_mask(2, 3, 0, 1) }>(xev);
        xev = _mm_max_epu8(xev, tempv);
        tempv = _mm_shuffle_epi32::<{ shuffle_mask(0, 1, 2, 3) }>(xev);
        xev = _mm_max_epu8(xev, tempv);
        tempv = _mm_shufflelo_epi16::<{ shuffle_mask(2, 3, 0, 1) }>(xev);
        xev = _mm_max_epu8(xev, tempv);
        tempv = _mm_srli_si128::<1>(xev);
        xev = _mm_max_epu8(xev, tempv);
        // Broadcast the max to all positions
        xev = _mm_shuffle_epi32::<{ shuffle_mask(0, 0, 0, 0) }>(xev);

        // E->C transition
        xev = _mm_subs_epu8(xev, tecv);
        // J state update
        xjv = _mm_max_epu8(xjv, xev);
        // B state update
        xbv = _mm_max_epu8(basev, xjv);
        xbv = _mm_subs_epu8(xbv, tjbmv);
    }

    // Extract final J value
    let xj = _mm_extract_epi16::<0>(xjv) as u16 as u8;

    // Convert back to nats
    let mut sc = (xj.wrapping_sub(om.tjb_b) as f32) - om.base_b as f32;
    sc /= om.scale_b;
    sc -= 3.0; // approximate L * log(L/(L+3)) for NN,CC,JJ

    MsvResult::Ok(sc)
}

use super::shuffle_mask;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::profile::*;
    use std::path::Path;

    /// Smoke test: MSV filter returns a finite score (or overflow) on a tiny model.
    #[test]
    fn test_msv_filter_basic() {
        if !is_x86_feature_detected!("sse2") {
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
        profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);

        let om = OProfile::convert(&gm);
        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");

        let result = unsafe { msv_filter(&dsq, 20, &om) };
        match result {
            MsvResult::Ok(sc) => {
                // MSV score may be negative for short sequences
                assert!(sc.is_finite(), "MSV score {} should be finite", sc);
            }
            MsvResult::Overflow => {
                // Overflow is also acceptable for a perfect match
            }
        }
    }
}
