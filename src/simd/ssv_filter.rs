//! Narrow SSE2 SSV filter for the common Q=17 band shape.
//!
//! This mirrors HMMER's impl_sse/ssvfilter.c for the Pkinase reference shape,
//! where Q=ceil(M/16)=17 and C splits the scan into band widths 8 and 9.
#![allow(clippy::absurd_extreme_comparisons, clippy::never_loop)]

use crate::alphabet::Dsq;
use crate::simd::oprofile::{nqb, OProfile};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Result of the SSV filter.
pub enum SsvResult {
    /// SSV score in nats (after C->T and NN/CC/JJ approximation).
    Ok(f32),
    /// Score saturated 8-bit unsigned range; caller should fall through to MSV.
    Overflow,
    /// SSV cannot give a reliable answer (e.g. J state might have been used,
    /// or bias parameters violate the assumption tjb+tbm+tec+bias < 127).
    /// Mirrors C `eslENORESULT`.
    NoResult,
}

/// Single-cell SSV step: load 16 emission bytes for one stripe lane, subtract
/// from `sv` with unsigned saturation, fold into the running max `xev`, then
/// advance the score pointer (mirrors C's `STEP_SINGLE` macro).
#[cfg(target_arch = "x86_64")]
macro_rules! step_lane {
    ($sv:ident, $rsc:ident, $xev:ident) => {{
        let r = _mm_loadu_si128($rsc as *const __m128i);
        $sv = _mm_subs_epi8($sv, r);
        $xev = _mm_max_epu8($xev, $sv);
        $rsc = $rsc.add(1);
    }};
}

/// Stripe boundary handling: shift one lane left by one byte and OR in the
/// `beginv` (-128) initial value so the score doesn't leak across stripes.
/// Mirrors C's `CONVERT_STEP` macro.
#[cfg(target_arch = "x86_64")]
macro_rules! convert_lane {
    ($sv:ident, $beginv:ident) => {{
        $sv = _mm_slli_si128::<1>($sv);
        $sv = _mm_or_si128($sv, $beginv);
    }};
}

/// Narrow SSE2 SSV filter specialised to the Q=17 stripe shape.
///
/// Port of `p7_SSVFilter` (hmmer/src/impl_sse/ssvfilter.c:876) for the case
/// where `nqb(M)==17` (e.g. the Pkinase reference profile). The full C
/// implementation auto-dispatches over `MAX_BANDS` of widths 1..18; this
/// Rust variant only handles the Q=17 split (band-8 + band-9) and returns
/// `NoResult` for other shapes. The MSV filter removes the J state entirely
/// for speed (~1% of comparisons fall back to MSVFilter via `NoResult`).
///
/// Returns `Overflow` on saturating high-scoring sequences (caller treats as
/// a hit) and `NoResult` when the J state might have been used so the score
/// is unreliable (caller re-runs the full MSVFilter).
///
/// # Safety
/// Requires SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn ssv_filter_q17(dsq: &[Dsq], l: usize, om: &OProfile) -> SsvResult {
    let q = nqb(om.m);
    if q != 17 || om.sbv.is_empty() || l == 0 {
        return SsvResult::NoResult;
    }
    if om.tjb_b as u16 + om.tbm_b as u16 + om.tec_b as u16 + om.bias_b as u16 >= 127 {
        return SsvResult::NoResult;
    }

    let beginv = _mm_set1_epi8(-128i8);
    let mut xev = beginv;
    xev = calc_band_8(dsq, l, om, beginv, xev);
    xev = calc_band_9(dsq, l, om, beginv, xev);

    let mut xe = hmax_epu8(xev) as u16;
    if xe >= 255 - om.bias_b as u16 {
        if om.base_b.wrapping_sub(om.tjb_b).wrapping_sub(om.tbm_b) < 128 {
            return SsvResult::NoResult;
        }
        return SsvResult::Overflow;
    }

    xe = xe
        .wrapping_add(om.base_b as u16)
        .wrapping_sub(om.tjb_b as u16)
        .wrapping_sub(om.tbm_b as u16)
        .wrapping_sub(128);
    if xe >= 255 - om.bias_b as u16 {
        return SsvResult::Overflow;
    }

    let xj = xe.wrapping_sub(om.tec_b as u16);
    if xj > om.base_b as u16 {
        return SsvResult::NoResult;
    }

    let mut sc = (xj.wrapping_sub(om.tjb_b as u16) as f32) - om.base_b as f32;
    sc /= om.scale_b;
    sc -= 3.0;
    SsvResult::Ok(sc)
}

/// SSV inner loop for an 8-wide stripe band, processing 8 score lanes per
/// residue. Port of C's `calc_band_8` (ssvfilter.c:759). The body is fully
/// unrolled to keep the eight `sv` registers live in SSE2 XMMs; the three
/// loop phases (prologue, steady-state stripe rotation, epilogue) walk the
/// stripe layout so that stripe boundaries trigger a single `convert_lane!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unused_assignments, unused_comparisons)]
unsafe fn calc_band_8(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    beginv: __m128i,
    mut xev: __m128i,
) -> __m128i {
    let mut sv00 = beginv;
    let mut sv01 = beginv;
    let mut sv02 = beginv;
    let mut sv03 = beginv;
    let mut sv04 = beginv;
    let mut sv05 = beginv;
    let mut sv06 = beginv;
    let mut sv07 = beginv;
    let dsq_ptr = dsq.as_ptr().add(1);

    let mut i = 0usize;
    while i < l && i < 9 {
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(i);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        i += 1;
    }

    i = 9;
    loop {
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(9);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv07, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(10);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv06, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(11);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv05, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(12);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv04, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(13);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv03, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(14);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv02, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(15);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv01, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(16);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv00, beginv);
        i += 1;
        break;
    }

    let mut i2 = 17usize;
    while i2 < l.saturating_sub(17) {
        i = 0;
        while i < 9 {
            let x = *dsq_ptr.add(i2 + i) as usize;
            let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(i);
            step_lane!(sv00, rsc, xev);
            step_lane!(sv01, rsc, xev);
            step_lane!(sv02, rsc, xev);
            step_lane!(sv03, rsc, xev);
            step_lane!(sv04, rsc, xev);
            step_lane!(sv05, rsc, xev);
            step_lane!(sv06, rsc, xev);
            step_lane!(sv07, rsc, xev);
            i += 1;
        }
        i += i2;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(9);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv07, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(10);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv06, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(11);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv05, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(12);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv04, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(13);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv03, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(14);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv02, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(15);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv01, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(16);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv00, beginv);
        i += 1;
        i2 += 17;
    }

    i = 0;
    while i2 + i < l && i < 9 {
        let x = *dsq_ptr.add(i2 + i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(i);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        i += 1;
    }
    i += i2;
    loop {
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(9);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv07, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(10);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv06, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(11);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv05, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(12);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv04, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(13);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv03, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(14);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv02, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(15);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv01, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(16);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        convert_lane!(sv00, beginv);
        i += 1;
        break;
    }

    xev
}

/// SSV inner loop for a 9-wide stripe band (9 score lanes per residue). Port
/// of C's `calc_band_9` (ssvfilter.c:765). Same unrolled structure as
/// `calc_band_8` with one additional lane; used together with `calc_band_8`
/// to cover Q=17 striped DP (8 + 9 = 17).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unused_assignments, unused_comparisons)]
unsafe fn calc_band_9(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    beginv: __m128i,
    mut xev: __m128i,
) -> __m128i {
    let mut sv00 = beginv;
    let mut sv01 = beginv;
    let mut sv02 = beginv;
    let mut sv03 = beginv;
    let mut sv04 = beginv;
    let mut sv05 = beginv;
    let mut sv06 = beginv;
    let mut sv07 = beginv;
    let mut sv08 = beginv;
    let dsq_ptr = dsq.as_ptr().add(1);

    let mut i = 0usize;
    while i < l && i < 0 {
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(i + 8);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        i += 1;
    }

    i = 0;
    loop {
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(8);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv08, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(9);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv07, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(10);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv06, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(11);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv05, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(12);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv04, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(13);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv03, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(14);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv02, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(15);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv01, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(16);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv00, beginv);
        i += 1;
        break;
    }

    let mut i2 = 9usize;
    while i2 < l.saturating_sub(17) {
        i = 0;
        while i < 8 {
            let x = *dsq_ptr.add(i2 + i) as usize;
            let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(i);
            step_lane!(sv00, rsc, xev);
            step_lane!(sv01, rsc, xev);
            step_lane!(sv02, rsc, xev);
            step_lane!(sv03, rsc, xev);
            step_lane!(sv04, rsc, xev);
            step_lane!(sv05, rsc, xev);
            step_lane!(sv06, rsc, xev);
            step_lane!(sv07, rsc, xev);
            step_lane!(sv08, rsc, xev);
            i += 1;
        }
        i += i2;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(8);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv08, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(9);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv07, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(10);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv06, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(11);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv05, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(12);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv04, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(13);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv03, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(14);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv02, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(15);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv01, beginv);
        i += 1;
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(16);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv00, beginv);
        i += 1;
        i2 += 17;
    }

    i = 0;
    while i2 + i < l && i < 8 {
        let x = *dsq_ptr.add(i2 + i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(i);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        i += 1;
    }
    i += i2;
    loop {
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(8);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv08, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(9);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv07, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(10);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv06, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(11);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv05, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(12);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv04, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(13);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv03, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(14);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv02, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(15);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv01, beginv);
        i += 1;
        if i >= l {
            break;
        }
        let x = *dsq_ptr.add(i) as usize;
        let mut rsc = om.sbv.get_unchecked(x).as_ptr().add(16);
        step_lane!(sv00, rsc, xev);
        step_lane!(sv01, rsc, xev);
        step_lane!(sv02, rsc, xev);
        step_lane!(sv03, rsc, xev);
        step_lane!(sv04, rsc, xev);
        step_lane!(sv05, rsc, xev);
        step_lane!(sv06, rsc, xev);
        step_lane!(sv07, rsc, xev);
        step_lane!(sv08, rsc, xev);
        convert_lane!(sv00, beginv);
        i += 1;
        break;
    }

    xev
}

/// Horizontal max of 16 unsigned bytes in an SSE2 vector (4-step reduction).
/// Easel `esl_sse_hmax_epu8` equivalent.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hmax_epu8(mut v: __m128i) -> u8 {
    let t = _mm_shuffle_epi32::<{ super::shuffle_mask(2, 3, 0, 1) }>(v);
    v = _mm_max_epu8(v, t);
    let t = _mm_shuffle_epi32::<{ super::shuffle_mask(0, 1, 2, 3) }>(v);
    v = _mm_max_epu8(v, t);
    let t = _mm_shufflelo_epi16::<{ super::shuffle_mask(2, 3, 0, 1) }>(v);
    v = _mm_max_epu8(v, t);
    let t = _mm_srli_si128::<1>(v);
    v = _mm_max_epu8(v, t);
    _mm_extract_epi16::<0>(v) as u16 as u8
}
