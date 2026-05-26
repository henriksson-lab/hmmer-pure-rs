//! SSE-optimized Backward parser (float precision, probability space).
//! Adapted from hmmer-pure-rs bck_engine() which closely follows C HMMER's
//! backward_engine() in impl_sse/fwdback.c.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;
use crate::util::cmath::c_log_f64;

/// Tracehash helper: sum one DP state (M/D/I) across stripes and emit a quantized hash.
///
/// Iterates lanes [`q_start`, `q_end`) of the striped row, accumulating the per-lane
/// values for the chosen state, then dispatches to the correct tracehash label.
/// Used only under `feature = "tracehash"` to compare against the C reference.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_engine_sum_q1e5(
    name: &'static str,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q: usize,
    row: &[[f32; 4]],
    state: usize,
    q_start: usize,
    q_end: usize,
) {
    let mut sum = 0.0_f32;
    for qi in q_start.min(q)..q_end.min(q) {
        let v = row[qi * 3 + state];
        for (lane, val) in v.iter().enumerate() {
            let k = qi + 1 + lane * q;
            if k <= m {
                sum += *val;
            }
        }
    }

    let mut th = match name {
        "rowl_m" => tracehash::th_call!("simd_backward_engine_rowl_msum_q1e5"),
        "rowl_d" => tracehash::th_call!("simd_backward_engine_rowl_dsum_q1e5"),
        "rowl_i" => tracehash::th_call!("simd_backward_engine_rowl_isum_q1e5"),
        "rowlm1_m" => tracehash::th_call!("simd_backward_engine_rowlm1_msum_q1e5"),
        "rowlm1_d" => tracehash::th_call!("simd_backward_engine_rowlm1_dsum_q1e5"),
        "rowlm1_i" => tracehash::th_call!("simd_backward_engine_rowlm1_isum_q1e5"),
        "rowlm2_m" => tracehash::th_call!("simd_backward_engine_rowlm2_msum_q1e5"),
        "rowlm2_d" => tracehash::th_call!("simd_backward_engine_rowlm2_dsum_q1e5"),
        "rowlm2_i" => tracehash::th_call!("simd_backward_engine_rowlm2_isum_q1e5"),
        "rowlm4_m" => tracehash::th_call!("simd_backward_engine_rowlm4_msum_q1e5"),
        "rowlm4_d" => tracehash::th_call!("simd_backward_engine_rowlm4_dsum_q1e5"),
        "rowlm4_i" => tracehash::th_call!("simd_backward_engine_rowlm4_isum_q1e5"),
        "rowlm8_m" => tracehash::th_call!("simd_backward_engine_rowlm8_msum_q1e5"),
        "rowlm8_d" => tracehash::th_call!("simd_backward_engine_rowlm8_dsum_q1e5"),
        "rowlm8_i" => tracehash::th_call!("simd_backward_engine_rowlm8_isum_q1e5"),
        "rowlm16_m" => tracehash::th_call!("simd_backward_engine_rowlm16_msum_q1e5"),
        "rowlm16_d" => tracehash::th_call!("simd_backward_engine_rowlm16_dsum_q1e5"),
        "rowlm16_i" => tracehash::th_call!("simd_backward_engine_rowlm16_isum_q1e5"),
        "rowlm32_m" => tracehash::th_call!("simd_backward_engine_rowlm32_msum_q1e5"),
        "rowlm32_d" => tracehash::th_call!("simd_backward_engine_rowlm32_dsum_q1e5"),
        "rowlm32_i" => tracehash::th_call!("simd_backward_engine_rowlm32_isum_q1e5"),
        "rowlm64_m" => tracehash::th_call!("simd_backward_engine_rowlm64_msum_q1e5"),
        "rowlm64_d" => tracehash::th_call!("simd_backward_engine_rowlm64_dsum_q1e5"),
        "rowlm64_i" => tracehash::th_call!("simd_backward_engine_rowlm64_isum_q1e5"),
        "rowlm128_m" => tracehash::th_call!("simd_backward_engine_rowlm128_msum_q1e5"),
        "rowlm128_d" => tracehash::th_call!("simd_backward_engine_rowlm128_dsum_q1e5"),
        "rowlm128_i" => tracehash::th_call!("simd_backward_engine_rowlm128_isum_q1e5"),
        "row128_m" => tracehash::th_call!("simd_backward_engine_row128_msum_q1e5"),
        "row128_d" => tracehash::th_call!("simd_backward_engine_row128_dsum_q1e5"),
        "row128_i" => tracehash::th_call!("simd_backward_engine_row128_isum_q1e5"),
        "row64_m" => tracehash::th_call!("simd_backward_engine_row64_msum_q1e5"),
        "row64_d" => tracehash::th_call!("simd_backward_engine_row64_dsum_q1e5"),
        "row64_i" => tracehash::th_call!("simd_backward_engine_row64_isum_q1e5"),
        "row32_m" => tracehash::th_call!("simd_backward_engine_row32_msum_q1e5"),
        "row32_d" => tracehash::th_call!("simd_backward_engine_row32_dsum_q1e5"),
        "row32_i" => tracehash::th_call!("simd_backward_engine_row32_isum_q1e5"),
        "row16_m" => tracehash::th_call!("simd_backward_engine_row16_msum_q1e5"),
        "row16_d" => tracehash::th_call!("simd_backward_engine_row16_dsum_q1e5"),
        "row16_i" => tracehash::th_call!("simd_backward_engine_row16_isum_q1e5"),
        "row8_m" => tracehash::th_call!("simd_backward_engine_row8_msum_q1e5"),
        "row8_d" => tracehash::th_call!("simd_backward_engine_row8_dsum_q1e5"),
        "row8_i" => tracehash::th_call!("simd_backward_engine_row8_isum_q1e5"),
        "row4_m" => tracehash::th_call!("simd_backward_engine_row4_msum_q1e5"),
        "row4_d" => tracehash::th_call!("simd_backward_engine_row4_dsum_q1e5"),
        "row4_i" => tracehash::th_call!("simd_backward_engine_row4_isum_q1e5"),
        "row2_m" => tracehash::th_call!("simd_backward_engine_row2_msum_q1e5"),
        "row2_d" => tracehash::th_call!("simd_backward_engine_row2_dsum_q1e5"),
        "row2_i" => tracehash::th_call!("simd_backward_engine_row2_isum_q1e5"),
        "m" => tracehash::th_call!("simd_backward_engine_row1_msum_q1e5"),
        "d" => tracehash::th_call!("simd_backward_engine_row1_dsum_q1e5"),
        "i" => tracehash::th_call!("simd_backward_engine_row1_isum_q1e5"),
        "phase1_m" => tracehash::th_call!("simd_backward_engine_row1_phase1_msum_q1e5"),
        "phase1_d" => tracehash::th_call!("simd_backward_engine_row1_phase1_dsum_q1e5"),
        "phase1_i" => tracehash::th_call!("simd_backward_engine_row1_phase1_isum_q1e5"),
        "phase3_m" => tracehash::th_call!("simd_backward_engine_row1_phase3_msum_q1e5"),
        "phase3_d" => tracehash::th_call!("simd_backward_engine_row1_phase3_dsum_q1e5"),
        "phase4_d" => tracehash::th_call!("simd_backward_engine_row1_phase4_dsum_q1e5"),
        "phase5_m" => tracehash::th_call!("simd_backward_engine_row1_phase5_msum_q1e5"),
        "m_q0_8" => tracehash::th_call!("simd_backward_engine_row1_m_q0_8_sum_q1e5"),
        "m_q8_16" => tracehash::th_call!("simd_backward_engine_row1_m_q8_16_sum_q1e5"),
        "m_q16_32" => tracehash::th_call!("simd_backward_engine_row1_m_q16_32_sum_q1e5"),
        _ => tracehash::th_call!("simd_backward_engine_row1_m_q32_end_sum_q1e5"),
    };
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.output_f32_quant(sum, 1.0e-5);
    th.finish();
}

/// Tracehash helper: hash the fully populated M/D/I state sums for a labeled row.
///
/// Unpacks the SSE row into scalar lanes, then calls [`trace_engine_sum_q1e5`] three
/// times (once per state) with row-specific tracehash labels (`rowl`, `rowlmN`, `rowN`).
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_engine_row_final_q1e5(
    row_label: &'static str,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q: usize,
    dpc_buf: &[__m128],
) {
    let mut row = Vec::with_capacity(dpc_buf.len());
    for v in dpc_buf {
        let mut lanes = [0.0_f32; 4];
        _mm_storeu_ps(lanes.as_mut_ptr(), *v);
        row.push(lanes);
    }

    match row_label {
        "rowl" => {
            trace_engine_sum_q1e5("rowl_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("rowl_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5("rowl_i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
        "rowlm1" => {
            trace_engine_sum_q1e5("rowlm1_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("rowlm1_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5("rowlm1_i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
        "rowlm2" => {
            trace_engine_sum_q1e5("rowlm2_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("rowlm2_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5("rowlm2_i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
        "rowlm4" => {
            trace_engine_sum_q1e5("rowlm4_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("rowlm4_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5("rowlm4_i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
        "rowlm8" => {
            trace_engine_sum_q1e5("rowlm8_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("rowlm8_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5("rowlm8_i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
        "rowlm16" => {
            trace_engine_sum_q1e5("rowlm16_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("rowlm16_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5("rowlm16_i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
        "rowlm32" => {
            trace_engine_sum_q1e5("rowlm32_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("rowlm32_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5("rowlm32_i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
        "rowlm64" => {
            trace_engine_sum_q1e5("rowlm64_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("rowlm64_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5("rowlm64_i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
        "rowlm128" => {
            trace_engine_sum_q1e5("rowlm128_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("rowlm128_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5("rowlm128_i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
        label => {
            let m_name = match label {
                "row128" => "row128_m",
                "row64" => "row64_m",
                "row32" => "row32_m",
                "row16" => "row16_m",
                "row8" => "row8_m",
                "row4" => "row4_m",
                _ => "row2_m",
            };
            let d_name = match label {
                "row128" => "row128_d",
                "row64" => "row64_d",
                "row32" => "row32_d",
                "row16" => "row16_d",
                "row8" => "row8_d",
                "row4" => "row4_d",
                _ => "row2_d",
            };
            let i_name = match label {
                "row128" => "row128_i",
                "row64" => "row64_i",
                "row32" => "row32_i",
                "row16" => "row16_i",
                "row8" => "row8_i",
                "row4" => "row4_i",
                _ => "row2_i",
            };
            trace_engine_sum_q1e5(m_name, dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5(d_name, dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5(i_name, dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
    }
}

/// Tracehash helper: hash row 1 with extra coarse partitions over the model length.
///
/// Emits the standard M/D/I sums plus stripe-window sums (`q0_8`, `q8_16`, `q16_32`,
/// `q32_end`) so tracehash regressions can localize divergences along the profile.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_engine_row1_q1e5(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q: usize,
    dpc_buf: &[__m128],
) {
    let mut row = Vec::with_capacity(dpc_buf.len());
    for v in dpc_buf {
        let mut lanes = [0.0_f32; 4];
        _mm_storeu_ps(lanes.as_mut_ptr(), *v);
        row.push(lanes);
    }
    trace_engine_sum_q1e5("m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
    trace_engine_sum_q1e5("d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
    trace_engine_sum_q1e5("i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
    trace_engine_sum_q1e5("m_q0_8", dsq, dsq_offset, l, m, q, &row, 0, 0, 8);
    trace_engine_sum_q1e5("m_q8_16", dsq, dsq_offset, l, m, q, &row, 0, 8, 16);
    trace_engine_sum_q1e5("m_q16_32", dsq, dsq_offset, l, m, q, &row, 0, 16, 32);
    trace_engine_sum_q1e5("m_q32_end", dsq, dsq_offset, l, m, q, &row, 0, 32, q);
}

/// Tracehash helper: hash row 1 at a mid-recursion phase checkpoint.
///
/// Phases `phase1`/`phase3`/`phase4`/`phase5` correspond to the M/I-init, E->M+DD,
/// post-DD, and M->D stages inside the Backward inner loop.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_engine_row1_checkpoint_q1e5(
    label: &'static str,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q: usize,
    dpc_buf: &[__m128],
) {
    let mut row = Vec::with_capacity(dpc_buf.len());
    for v in dpc_buf {
        let mut lanes = [0.0_f32; 4];
        _mm_storeu_ps(lanes.as_mut_ptr(), *v);
        row.push(lanes);
    }

    match label {
        "phase1" => {
            trace_engine_sum_q1e5("phase1_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("phase1_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
            trace_engine_sum_q1e5("phase1_i", dsq, dsq_offset, l, m, q, &row, 2, 0, q);
        }
        "phase3" => {
            trace_engine_sum_q1e5("phase3_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
            trace_engine_sum_q1e5("phase3_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
        }
        "phase4" => {
            trace_engine_sum_q1e5("phase4_d", dsq, dsq_offset, l, m, q, &row, 1, 0, q);
        }
        _ => {
            trace_engine_sum_q1e5("phase5_m", dsq, dsq_offset, l, m, q, &row, 0, 0, q);
        }
    }
}

/// The Backward algorithm, linear-memory parsing version (SSE, probability space).
///
/// Variant of C `p7_BackwardParser` — same as `p7_Backward` except the full DP
/// matrix is not kept; only the special (BENCJ) state values per row are stored,
/// in $O(M+L)$ memory. These are sufficient for posterior decoding to locate
/// high-probability domain regions. Requires a previously filled Forward parser
/// matrix because the same sparse scale factors must be re-applied.
///
/// # Args
/// - `dsq` digital target sequence, indices 1..L
/// - `l` sequence length
/// - `om` optimized profile (must be in local alignment mode)
/// - `_fwd_sc` Forward score (unused; kept for API symmetry)
/// - `pmx` Backward probability matrix to fill (specials + optional DP)
///
/// # Returns
/// Backward score in nats.
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn backward_parser_pmx(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    _fwd_sc: f32,
    pmx: &mut super::probmx::ProbMx,
) -> f32 {
    backward_parser_pmx_offset(dsq, 0, l, om, _fwd_sc, pmx)
}

/// Backward parser over a `dsq` slice with an explicit base offset (variant of
/// C `p7_BackwardParser`).
///
/// Identical to [`backward_parser_pmx`] but indexes residues at `dsq[dsq_offset+1..]`,
/// letting callers process windows of a larger digital sequence without copying.
/// Allocates per-call scratch SSE buffers; see `_with_scratch` for reuse.
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn backward_parser_pmx_offset(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    _fwd_sc: f32,
    pmx: &mut super::probmx::ProbMx,
) -> f32 {
    let mut dpp_buf: Vec<__m128> = Vec::new();
    let mut dpc_buf: Vec<__m128> = Vec::new();
    backward_parser_pmx_offset_with_scratch(
        dsq,
        dsq_offset,
        l,
        om,
        _fwd_sc,
        pmx,
        None,
        &mut dpp_buf,
        &mut dpc_buf,
    )
}

/// Core Backward engine with caller-supplied scratch buffers (variant of C
/// `backward_engine`).
///
/// Mirrors the fused `backward_engine` in `impl_sse/fwdback.c` that backs both
/// `p7_Backward` and `p7_BackwardParser`. Two SSE row buffers (`dpp_buf`, `dpc_buf`)
/// roll backward from row L to row 0, computing M/D/I per stripe with D->D wing
/// unfolding and sparse rescaling. When `pmx.has_dp` is set and the sequence is
/// fully canonical, dispatches to `backward_parser_pmx_offset_direct` for the
/// in-place full-matrix path; otherwise fills only the special states. If
/// `fwd_row_scales` is `Some`, those scales are reused for cancellation against
/// a paired Forward; if `None`, Backward picks its own when xB grows large.
///
/// Returns the Backward score (nats).
///
/// # Safety
/// Requires SSE2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn backward_parser_pmx_offset_with_scratch(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    _fwd_sc: f32,
    pmx: &mut super::probmx::ProbMx,
    fwd_row_scales: Option<&[f32]>,
    dpp_buf: &mut Vec<__m128>,
    dpc_buf: &mut Vec<__m128>,
) -> f32 {
    if pmx.has_dp && canonical_run(dsq, dsq_offset, l, om.abc_kp) {
        return backward_parser_pmx_offset_direct(dsq, dsq_offset, l, om, pmx, fwd_row_scales);
    }

    use super::probmx::*;

    let q = (om.m + 3) / 4; // nqf
    let zerov = _mm_setzero_ps();

    // Two-row rolling buffer: dpp = previous (i+1), dpc = current (i)
    let row_len = q * 3; // M, D, I per stripe
    dpp_buf.resize(row_len, zerov);
    dpc_buf.resize(row_len, zerov);

    // Special state init at position L
    let c_move = om.xf[P7O_C][P7O_MOVE];
    let c_loop = om.xf[P7O_C][P7O_LOOP];
    let e_move = om.xf[P7O_E][P7O_MOVE]; // E->C
    let e_loop = om.xf[P7O_E][P7O_LOOP]; // E->J
    let j_move = om.xf[P7O_J][P7O_MOVE]; // J->B
    let j_loop = om.xf[P7O_J][P7O_LOOP];
    let n_move = om.xf[P7O_N][P7O_MOVE]; // N->B
    let n_loop = om.xf[P7O_N][P7O_LOOP];

    let mut x_c: f32 = c_move; // C->T at position L
    let mut x_e: f32 = x_c * e_move;
    let mut x_j: f32 = 0.0;
    let mut x_b: f32 = 0.0;
    let mut x_n: f32 = 0.0;
    let mut totscale: f64 = 0.0;
    let track_scales = cfg!(feature = "tracehash") || fwd_row_scales.is_none();
    let mut has_own_scales = fwd_row_scales.is_none();
    // Initialize row L: M(L,k)->E->C->T and D(L,k)->E->C->T
    {
        let dp = dpp_buf.as_mut_ptr();
        let x_ev = _mm_set1_ps(x_e);
        for qi in 0..q {
            *dp.add(qi * 3) = x_ev; // M
            *dp.add(qi * 3 + 1) = x_ev; // D
            *dp.add(qi * 3 + 2) = zerov; // I
        }

        // D->D wing unfolding at row L (right to left in striped layout)
        {
            let mut dpv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(
                *dp.add((q - 1) * 3 + 1),
            )));
            let mut dcv = zerov;
            for qi in (0..q).rev() {
                let tdd = load_tfv_dd(om, qi);
                dcv = _mm_mul_ps(dpv, tdd);
                let d_ptr = dp.add(qi * 3 + 1);
                *d_ptr = _mm_add_ps(*d_ptr, dcv);
                dpv = *d_ptr;
            }
            for _ in 0..3 {
                dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(dcv)));
                for qi in (0..q).rev() {
                    let tdd = load_tfv_dd(om, qi);
                    dcv = _mm_mul_ps(dcv, tdd);
                    let d_ptr = dp.add(qi * 3 + 1);
                    *d_ptr = _mm_add_ps(*d_ptr, dcv);
                }
            }
        }

        // M->D at row L
        {
            let mut dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(*dp.add(1))));
            for qi in (0..q).rev() {
                let tmd = load_tfv(om, qi, 4); // MD transition
                let m_ptr = dp.add(qi * 3);
                *m_ptr = _mm_add_ps(*m_ptr, _mm_mul_ps(dcv, tmd));
                dcv = *dp.add(qi * 3 + 1);
            }
        }
    }

    let row_scale_l = fwd_row_scales.map(|s| s[l]).unwrap_or(1.0);
    if row_scale_l > 1.0 {
        let inv = 1.0 / row_scale_l;
        let scalev = _mm_set1_ps(inv);
        x_e /= row_scale_l;
        x_n /= row_scale_l;
        x_j /= row_scale_l;
        x_b /= row_scale_l;
        x_c /= row_scale_l;
        for v in dpp_buf.iter_mut() {
            *v = _mm_mul_ps(*v, scalev);
        }
        if track_scales {
            totscale += c_log_f64(row_scale_l as f64);
        }
    }

    if pmx.has_dp {
        pmx.write_simd_row(&dpp_buf, q, om.m, l);
    }

    // Store row L specials
    pmx.set_xmx(l, PXE, x_e);
    pmx.set_xmx(l, PXN, 0.0);
    pmx.set_xmx(l, PXJ, 0.0);
    pmx.set_xmx(l, PXB, 0.0);
    pmx.set_xmx(l, PXC, x_c);
    pmx.scale[l] = if track_scales { totscale } else { 0.0 };
    pmx.row_scale[l] = row_scale_l;

    #[cfg(feature = "tracehash")]
    trace_engine_row_final_q1e5("rowl", dsq, dsq_offset, l, om.m, q, &dpp_buf);

    // Main recursion: i = L-1 down to 1
    for i in (1..l).rev() {
        let xi_next = dsq[dsq_offset + i + 1] as usize;
        if xi_next >= om.abc_kp {
            // Non-canonical residue: copy previous row, update specials
            for v in dpc_buf.iter_mut() {
                *v = zerov;
            }
            x_c *= c_loop;
            x_j = x_b * j_move + x_j * j_loop;
            x_n = x_b * n_move + x_n * n_loop;
            x_e = x_c * e_move + x_j * e_loop;
            pmx.set_xmx(i, PXE, x_e);
            pmx.set_xmx(i, PXN, x_n);
            pmx.set_xmx(i, PXJ, x_j);
            pmx.set_xmx(i, PXB, x_b);
            pmx.set_xmx(i, PXC, x_c);
            pmx.scale[i] = if track_scales { totscale } else { 0.0 };
            pmx.row_scale[i] = fwd_row_scales.map(|s| s[i]).unwrap_or(1.0);
            if pmx.has_dp {
                pmx.zero_simd_row(i);
            }
            std::mem::swap(dpp_buf, dpc_buf);
            continue;
        }

        let dpp = dpp_buf.as_ptr();
        let dpc = dpc_buf.as_mut_ptr();

        // Phase 1: Compute M(i,k) and I(i,k) from row i+1.
        // This follows C impl_sse/fwdback.c exactly: M(i+1,k+1)
        // contributions use left-shifted MM/IM/DM transition vectors.
        let mpv_init = _mm_mul_ps(*dpp, load_rfv(om, xi_next, 0));
        let mut mpv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(mpv_init)));
        let mut tmmv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(load_tfv(om, 0, 1))));
        let mut timv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(load_tfv(om, 0, 2))));
        let mut tdmv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(load_tfv(om, 0, 3))));

        let mut x_bv = zerov;

        for qi in (0..q).rev() {
            let ipv = *dpp.add(qi * 3 + 2); // I(i+1, k)

            // I(i,k) = I(i+1,k)*II + M(i+1,k+1)*e(x_{i+1})*IM
            let tii = load_tfv(om, qi, 6);
            *dpc.add(qi * 3 + 2) = _mm_add_ps(_mm_mul_ps(ipv, tii), _mm_mul_ps(mpv, timv));

            // D(i,k) = M(i+1,k+1)*e(x_{i+1})*DM, partial before D/E paths.
            *dpc.add(qi * 3 + 1) = _mm_mul_ps(mpv, tdmv);

            // M(i,k) = I(i+1,k)*MI + M(i+1,k+1)*e(x_{i+1})*MM, partial.
            let tmi = load_tfv(om, qi, 5);
            let mcv = _mm_add_ps(_mm_mul_ps(ipv, tmi), _mm_mul_ps(mpv, tmmv));

            // Next mpv: M(i+1,k) * emission(k, x_{i+1}). C reads this
            // before storing M(i,k), because its parser row is in-place.
            let next_mpv = _mm_mul_ps(*dpp.add(qi * 3), load_rfv(om, xi_next, qi));
            *dpc.add(qi * 3) = mcv;
            mpv = next_mpv;

            // B->M contribution to xB uses the newly obtained M(i+1,k) term.
            let tbm = load_tfv(om, qi, 0);
            x_bv = _mm_add_ps(x_bv, _mm_mul_ps(mpv, tbm));

            tdmv = load_tfv(om, qi, 3);
            timv = load_tfv(om, qi, 2);
            tmmv = load_tfv(om, qi, 1);
        }

        #[cfg(feature = "tracehash")]
        if i == 1 {
            trace_engine_row1_checkpoint_q1e5("phase1", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }

        // Horizontal sum xBv -> xB
        x_bv = _mm_add_ps(
            x_bv,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(x_bv, x_bv),
        );
        x_bv = _mm_add_ps(
            x_bv,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(x_bv, x_bv),
        );
        _mm_store_ss(&mut x_b, x_bv);

        // Phase 2: Special states
        x_c *= c_loop;
        x_j = x_b * j_move + x_j * j_loop;
        x_n = x_b * n_move + x_n * n_loop;
        x_e = x_c * e_move + x_j * e_loop;
        let x_ev = _mm_set1_ps(x_e);

        // Phase 3: Add E->M,D paths + D->D wing unfolding
        {
            let mut dpv = _mm_add_ps(*dpc.add(1), x_ev);
            dpv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(dpv)));
            let mut dcv = zerov;
            for qi in (0..q).rev() {
                let tdd = load_tfv_dd(om, qi);
                dcv = _mm_mul_ps(dpv, tdd);
                let d_ptr = dpc.add(qi * 3 + 1);
                *d_ptr = _mm_add_ps(*d_ptr, _mm_add_ps(dcv, x_ev));
                dpv = *d_ptr;
                let m_ptr = dpc.add(qi * 3);
                *m_ptr = _mm_add_ps(*m_ptr, x_ev);
            }

            #[cfg(feature = "tracehash")]
            if i == 1 {
                trace_engine_row1_checkpoint_q1e5("phase3", dsq, dsq_offset, l, om.m, q, &dpc_buf);
            }

            // 3 more D->D passes for convergence
            for _ in 0..3 {
                dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(dcv)));
                for qi in (0..q).rev() {
                    let tdd = load_tfv_dd(om, qi);
                    dcv = _mm_mul_ps(dcv, tdd);
                    let d_ptr = dpc.add(qi * 3 + 1);
                    *d_ptr = _mm_add_ps(*d_ptr, dcv);
                }
            }

            #[cfg(feature = "tracehash")]
            if i == 1 {
                trace_engine_row1_checkpoint_q1e5("phase4", dsq, dsq_offset, l, om.m, q, &dpc_buf);
            }
        }

        // Phase 4: M->D paths
        {
            let mut dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(*dpc.add(1))));
            for qi in (0..q).rev() {
                let tmd = load_tfv(om, qi, 4);
                let m_ptr = dpc.add(qi * 3);
                *m_ptr = _mm_add_ps(*m_ptr, _mm_mul_ps(dcv, tmd));
                dcv = *dpc.add(qi * 3 + 1);
            }

            #[cfg(feature = "tracehash")]
            if i == 1 {
                trace_engine_row1_checkpoint_q1e5("phase5", dsq, dsq_offset, l, om.m, q, &dpc_buf);
            }
        }

        // Sparse rescaling.
        if x_b > 1.0e16 {
            has_own_scales = true;
        }
        let row_scale = if has_own_scales {
            if x_b > 1.0e4 {
                x_b
            } else {
                1.0
            }
        } else {
            fwd_row_scales.unwrap()[i]
        };
        if row_scale > 1.0 {
            let inv_xb = 1.0 / row_scale;
            let scalev = _mm_set1_ps(inv_xb);
            x_e /= row_scale;
            x_n /= row_scale;
            x_j /= row_scale;
            x_c /= row_scale;
            for qi in 0..row_len {
                let p = dpc.add(qi);
                *p = _mm_mul_ps(*p, scalev);
            }
            if track_scales {
                totscale += c_log_f64(row_scale as f64);
            }
            if has_own_scales {
                x_b = 1.0;
            } else {
                x_b /= row_scale;
            }
        }

        // Store full DP row if requested
        if pmx.has_dp {
            pmx.write_simd_row(&dpc_buf, q, om.m, i);
        }

        // Store specials
        pmx.set_xmx(i, PXE, x_e);
        pmx.set_xmx(i, PXN, x_n);
        pmx.set_xmx(i, PXJ, x_j);
        pmx.set_xmx(i, PXB, x_b);
        pmx.set_xmx(i, PXC, x_c);
        pmx.scale[i] = if track_scales { totscale } else { 0.0 };
        pmx.row_scale[i] = row_scale;

        #[cfg(feature = "tracehash")]
        if i == 1 {
            trace_engine_row1_q1e5(dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i == 2 {
            trace_engine_row_final_q1e5("row2", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i + 1 == l {
            trace_engine_row_final_q1e5("rowlm1", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i + 2 == l {
            trace_engine_row_final_q1e5("rowlm2", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i + 4 == l {
            trace_engine_row_final_q1e5("rowlm4", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i + 8 == l {
            trace_engine_row_final_q1e5("rowlm8", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i + 16 == l {
            trace_engine_row_final_q1e5("rowlm16", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i + 32 == l {
            trace_engine_row_final_q1e5("rowlm32", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i + 64 == l {
            trace_engine_row_final_q1e5("rowlm64", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i + 128 == l {
            trace_engine_row_final_q1e5("rowlm128", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i == 4 {
            trace_engine_row_final_q1e5("row4", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i == 8 {
            trace_engine_row_final_q1e5("row8", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i == 16 {
            trace_engine_row_final_q1e5("row16", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i == 32 {
            trace_engine_row_final_q1e5("row32", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i == 64 {
            trace_engine_row_final_q1e5("row64", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }
        #[cfg(feature = "tracehash")]
        if i == 128 {
            trace_engine_row_final_q1e5("row128", dsq, dsq_offset, l, om.m, q, &dpc_buf);
        }

        std::mem::swap(dpp_buf, dpc_buf);
    }

    // Row 0 termination
    {
        let dp = dpp_buf.as_ptr();
        let xi1 = dsq[dsq_offset + 1] as usize;
        if xi1 < om.abc_kp {
            let mut x_bv = zerov;
            for qi in 0..q {
                let tbm = load_tfv(om, qi, 0);
                let mpv = _mm_mul_ps(*dp.add(qi * 3), load_rfv(om, xi1, qi));
                x_bv = _mm_add_ps(x_bv, _mm_mul_ps(mpv, tbm));
            }
            x_bv = _mm_add_ps(
                x_bv,
                _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(x_bv, x_bv),
            );
            x_bv = _mm_add_ps(
                x_bv,
                _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(x_bv, x_bv),
            );
            _mm_store_ss(&mut x_b, x_bv);
        }
        x_n = x_b * n_move + x_n * n_loop;
    }

    pmx.set_xmx(0, PXE, 0.0);
    pmx.set_xmx(0, PXN, x_n);
    pmx.set_xmx(0, PXJ, 0.0);
    pmx.set_xmx(0, PXB, x_b);
    pmx.set_xmx(0, PXC, 0.0);
    pmx.scale[0] = if track_scales { totscale } else { 0.0 };
    pmx.row_scale[0] = 1.0;
    pmx.has_own_scales = has_own_scales;

    (totscale + c_log_f64(x_n as f64)) as f32
}

/// Backward full-matrix fill assuming a canonical (no degenerate residues) sequence.
///
/// Variant of C `backward_engine`: the caller asserts `pmx.has_dp` and that all
/// residues are within the canonical alphabet, allowing the routine to skip the
/// degenerate-residue branch and write directly into the striped DP matrix via
/// `backward_parser_pmx_offset_direct`.
///
/// # Safety
/// Requires SSE2 support and `pmx.has_dp` must be true.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn backward_parser_pmx_offset_canonical(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
    fwd_row_scales: Option<&[f32]>,
) -> f32 {
    debug_assert!(pmx.has_dp);
    backward_parser_pmx_offset_direct(dsq, dsq_offset, l, om, pmx, fwd_row_scales)
}

/// Backward full-matrix fill, writing directly into the striped ProbMx (variant of
/// C `p7_Backward`).
///
/// Direct counterpart to `p7_Backward` (full $O(ML)$ memory/time): no rolling row
/// buffers — every row is computed in-place inside `pmx.striped_dp`. Caller must
/// pre-allocate `pmx.has_dp == true`. Per-row sparse rescaling matches Forward
/// when `fwd_row_scales` is `Some`, or is chosen locally when xB exceeds 1e4.
/// Assumes all residues are canonical; the dispatcher
/// [`backward_parser_pmx_offset_with_scratch`] checks this precondition.
///
/// Returns the Backward score (nats).
///
/// # Safety
/// Requires SSE2 support; `pmx` must have its striped DP storage allocated and
/// `dsq_offset+l` must be in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn backward_parser_pmx_offset_direct(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
    fwd_row_scales: Option<&[f32]>,
) -> f32 {
    use super::probmx::*;

    let q = (om.m + 3) / 4;
    let row_width = pmx.striped_row_width();
    let zerov = _mm_setzero_ps();
    let dsq_ptr = dsq.as_ptr().add(dsq_offset);
    let striped_ptr = pmx.striped_dp.as_mut_ptr().add(pmx.striped_dp_offset);
    let xmx_ptr = pmx.xmx.as_mut_ptr();
    let scale_ptr = pmx.scale.as_mut_ptr();
    let row_scale_ptr = pmx.row_scale.as_mut_ptr();
    let c_move = om.xf[P7O_C][P7O_MOVE];
    let c_loop = om.xf[P7O_C][P7O_LOOP];
    let e_move = om.xf[P7O_E][P7O_MOVE];
    let e_loop = om.xf[P7O_E][P7O_LOOP];
    let j_move = om.xf[P7O_J][P7O_MOVE];
    let j_loop = om.xf[P7O_J][P7O_LOOP];
    let n_move = om.xf[P7O_N][P7O_MOVE];
    let n_loop = om.xf[P7O_N][P7O_LOOP];

    let mut x_c: f32 = c_move;
    let mut x_e: f32 = x_c * e_move;
    let mut x_j: f32 = 0.0;
    let mut x_b: f32 = 0.0;
    let mut x_n: f32 = 0.0;
    let mut totscale: f64 = 0.0;
    let track_scales = cfg!(feature = "tracehash") || fwd_row_scales.is_none();
    let mut has_own_scales = fwd_row_scales.is_none();

    #[inline(always)]
    unsafe fn load_cell(row: *const f32, q: usize, s: usize) -> __m128 {
        _mm_load_ps(row.add(q * 12 + s * 4))
    }

    #[inline(always)]
    unsafe fn store_cell(row: *mut f32, q: usize, s: usize, v: __m128) {
        _mm_store_ps(row.add(q * 12 + s * 4), v);
    }

    #[inline(always)]
    unsafe fn store_xmx(xmx: *mut f32, i: usize, xe: f32, xn: f32, xj: f32, xb: f32, xc: f32) {
        let row = xmx.add(i * 5);
        *row.add(PXE) = xe;
        *row.add(PXN) = xn;
        *row.add(PXJ) = xj;
        *row.add(PXB) = xb;
        *row.add(PXC) = xc;
    }

    {
        let row_l = striped_ptr.add(l * row_width);
        let x_ev = _mm_set1_ps(x_e);
        for qi in 0..q {
            store_cell(row_l, qi, 0, x_ev);
            store_cell(row_l, qi, 1, x_ev);
            store_cell(row_l, qi, 2, zerov);
        }

        {
            let mut dpv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(load_cell(
                row_l,
                q - 1,
                1,
            ))));
            let mut dcv = zerov;
            for qi in (0..q).rev() {
                let tdd = load_tfv_dd(om, qi);
                dcv = _mm_mul_ps(dpv, tdd);
                let d = _mm_add_ps(load_cell(row_l, qi, 1), dcv);
                store_cell(row_l, qi, 1, d);
                dpv = d;
            }
            for _ in 0..3 {
                dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(dcv)));
                for qi in (0..q).rev() {
                    let tdd = load_tfv_dd(om, qi);
                    dcv = _mm_mul_ps(dcv, tdd);
                    let d = _mm_add_ps(load_cell(row_l, qi, 1), dcv);
                    store_cell(row_l, qi, 1, d);
                }
            }
        }

        {
            let mut dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(load_cell(
                row_l, 0, 1,
            ))));
            for qi in (0..q).rev() {
                let tmd = load_tfv(om, qi, 4);
                let m = _mm_add_ps(load_cell(row_l, qi, 0), _mm_mul_ps(dcv, tmd));
                store_cell(row_l, qi, 0, m);
                dcv = load_cell(row_l, qi, 1);
            }
        }
    }

    let row_scale_l = fwd_row_scales.map(|s| s[l]).unwrap_or(1.0);
    if row_scale_l > 1.0 {
        let inv = 1.0 / row_scale_l;
        let scalev = _mm_set1_ps(inv);
        x_e /= row_scale_l;
        x_n /= row_scale_l;
        x_j /= row_scale_l;
        x_b /= row_scale_l;
        x_c /= row_scale_l;
        let row_l = striped_ptr.add(l * row_width);
        let mut off = 0;
        while off < row_width {
            let p = row_l.add(off);
            _mm_store_ps(p, _mm_mul_ps(_mm_load_ps(p), scalev));
            off += 4;
        }
        if track_scales {
            totscale += c_log_f64(row_scale_l as f64);
        }
    }

    store_xmx(xmx_ptr, l, x_e, 0.0, 0.0, 0.0, x_c);
    *scale_ptr.add(l) = if track_scales { totscale } else { 0.0 };
    *row_scale_ptr.add(l) = row_scale_l;

    for i in (1..l).rev() {
        let dpp = striped_ptr.add((i + 1) * row_width) as *const f32;
        let dpc = striped_ptr.add(i * row_width);
        let xi_next = *dsq_ptr.add(i + 1) as usize;

        let mpv_init = _mm_mul_ps(load_cell(dpp, 0, 0), load_rfv(om, xi_next, 0));
        let mut mpv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(mpv_init)));
        let mut tmmv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(load_tfv(om, 0, 1))));
        let mut timv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(load_tfv(om, 0, 2))));
        let mut tdmv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(load_tfv(om, 0, 3))));

        let mut x_bv = zerov;

        for qi in (0..q).rev() {
            let ipv = load_cell(dpp, qi, 2);
            let tii = load_tfv(om, qi, 6);
            store_cell(
                dpc,
                qi,
                2,
                _mm_add_ps(_mm_mul_ps(ipv, tii), _mm_mul_ps(mpv, timv)),
            );
            store_cell(dpc, qi, 1, _mm_mul_ps(mpv, tdmv));

            let tmi = load_tfv(om, qi, 5);
            store_cell(
                dpc,
                qi,
                0,
                _mm_add_ps(_mm_mul_ps(ipv, tmi), _mm_mul_ps(mpv, tmmv)),
            );

            mpv = _mm_mul_ps(load_cell(dpp, qi, 0), load_rfv(om, xi_next, qi));
            let tbm = load_tfv(om, qi, 0);
            x_bv = _mm_add_ps(x_bv, _mm_mul_ps(mpv, tbm));

            tdmv = load_tfv(om, qi, 3);
            timv = load_tfv(om, qi, 2);
            tmmv = load_tfv(om, qi, 1);
        }

        x_bv = _mm_add_ps(
            x_bv,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(x_bv, x_bv),
        );
        x_bv = _mm_add_ps(
            x_bv,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(x_bv, x_bv),
        );
        _mm_store_ss(&mut x_b, x_bv);

        x_c *= c_loop;
        x_j = x_b * j_move + x_j * j_loop;
        x_n = x_b * n_move + x_n * n_loop;
        x_e = x_c * e_move + x_j * e_loop;
        let x_ev = _mm_set1_ps(x_e);

        {
            let mut dpv = _mm_add_ps(load_cell(dpc, 0, 1), x_ev);
            dpv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(dpv)));
            let mut dcv = zerov;
            for qi in (0..q).rev() {
                let tdd = load_tfv_dd(om, qi);
                dcv = _mm_mul_ps(dpv, tdd);
                let d = _mm_add_ps(load_cell(dpc, qi, 1), _mm_add_ps(dcv, x_ev));
                store_cell(dpc, qi, 1, d);
                dpv = d;
                let m = _mm_add_ps(load_cell(dpc, qi, 0), x_ev);
                store_cell(dpc, qi, 0, m);
            }

            for _ in 0..3 {
                dcv = _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(dcv)));
                for qi in (0..q).rev() {
                    let tdd = load_tfv_dd(om, qi);
                    dcv = _mm_mul_ps(dcv, tdd);
                    let d = _mm_add_ps(load_cell(dpc, qi, 1), dcv);
                    store_cell(dpc, qi, 1, d);
                }
            }
        }

        {
            let mut dcv =
                _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(load_cell(dpc, 0, 1))));
            for qi in (0..q).rev() {
                let tmd = load_tfv(om, qi, 4);
                let m = _mm_add_ps(load_cell(dpc, qi, 0), _mm_mul_ps(dcv, tmd));
                store_cell(dpc, qi, 0, m);
                dcv = load_cell(dpc, qi, 1);
            }
        }

        if x_b > 1.0e16 {
            has_own_scales = true;
        }
        let row_scale = if has_own_scales {
            if x_b > 1.0e4 {
                x_b
            } else {
                1.0
            }
        } else {
            fwd_row_scales.unwrap()[i]
        };
        if row_scale > 1.0 {
            let inv_xb = 1.0 / row_scale;
            let scalev = _mm_set1_ps(inv_xb);
            x_e /= row_scale;
            x_n /= row_scale;
            x_j /= row_scale;
            x_c /= row_scale;
            let mut off = 0;
            while off < row_width {
                let p = dpc.add(off);
                _mm_store_ps(p, _mm_mul_ps(_mm_load_ps(p), scalev));
                off += 4;
            }
            if track_scales {
                totscale += c_log_f64(row_scale as f64);
            }
            if has_own_scales {
                x_b = 1.0;
            } else {
                x_b /= row_scale;
            }
        }

        store_xmx(xmx_ptr, i, x_e, x_n, x_j, x_b, x_c);
        *scale_ptr.add(i) = if track_scales { totscale } else { 0.0 };
        *row_scale_ptr.add(i) = row_scale;
    }

    {
        let dp = striped_ptr.add(row_width) as *const f32;
        let xi1 = *dsq_ptr.add(1) as usize;
        let mut x_bv = zerov;
        for qi in 0..q {
            let tbm = load_tfv(om, qi, 0);
            let mpv = _mm_mul_ps(load_cell(dp, qi, 0), load_rfv(om, xi1, qi));
            x_bv = _mm_add_ps(x_bv, _mm_mul_ps(mpv, tbm));
        }
        x_bv = _mm_add_ps(
            x_bv,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(x_bv, x_bv),
        );
        x_bv = _mm_add_ps(
            x_bv,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(x_bv, x_bv),
        );
        _mm_store_ss(&mut x_b, x_bv);
        x_n = x_b * n_move + x_n * n_loop;
    }

    store_xmx(xmx_ptr, 0, 0.0, x_n, 0.0, x_b, 0.0);
    *scale_ptr = if track_scales { totscale } else { 0.0 };
    *row_scale_ptr = 1.0;
    pmx.has_own_scales = has_own_scales;

    (totscale + c_log_f64(x_n as f64)) as f32
}

/// Returns `true` iff every residue in `dsq[dsq_offset+1..=dsq_offset+l]` is a
/// canonical alphabet symbol (`< abc_kp`).
///
/// Used to decide whether the Backward engine can take the fast in-place path.
/// Returns `false` on empty windows or out-of-bounds spans.
#[inline(always)]
fn canonical_run(dsq: &[Dsq], dsq_offset: usize, l: usize, abc_kp: usize) -> bool {
    if l == 0 {
        return false;
    }
    let Some(end) = dsq_offset.checked_add(l) else {
        return false;
    };
    if end >= dsq.len() {
        return false;
    }
    unsafe {
        let ptr = dsq.as_ptr().add(dsq_offset);
        for i in 1..=l {
            if *ptr.add(i) as usize >= abc_kp {
                return false;
            }
        }
    }
    true
}

/// Load transition vector for stripe qi, transition index tidx (0=BM,1=MM,2=IM,3=DM,4=MD,5=MI,6=II)
#[inline(always)]
unsafe fn load_tfv(om: &OProfile, qi: usize, tidx: usize) -> __m128 {
    let ptr = om.tfv_a.as_ptr().add(qi * 7 + tidx);
    _mm_load_ps((*ptr).as_ptr())
}

/// Load D->D transition vector for stripe qi
#[inline(always)]
unsafe fn load_tfv_dd(om: &OProfile, qi: usize) -> __m128 {
    let q = (om.m + 3) / 4;
    let ptr = om.tfv_a.as_ptr().add(7 * q + qi);
    _mm_load_ps((*ptr).as_ptr())
}

/// Load float emission vector for residue x, stripe qi
#[inline(always)]
unsafe fn load_rfv(om: &OProfile, x: usize, qi: usize) -> __m128 {
    let residue = om.rfv_a.as_ptr().add(x);
    let ptr = (*residue).as_ptr().add(qi);
    _mm_load_ps((*ptr).as_ptr())
}
