//! SSE-optimized Forward parser (float precision, probability space).
//! Direct port of impl_sse/fwdback.c forward_engine() in parser mode.
#![allow(clippy::let_and_return, clippy::needless_borrow)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;
use crate::util::cmath::c_log_f64;

/// Tracehash helper: emits quantized per-state sums (M/D) of the striped DP row
/// for rows 18/19/20. Mirrors `trace_forward_engine_row_sums_q1e5` in fwdback.c.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_forward_engine_row_sums_q1e5(
    row: usize,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q_count: usize,
    dp: &[__m128],
) {
    let mut msum = 0.0_f32;
    let mut dsum = 0.0_f32;
    for qi in 0..q_count {
        let mut mlanes = [0.0_f32; 4];
        let mut dlanes = [0.0_f32; 4];
        _mm_storeu_ps(mlanes.as_mut_ptr(), dp[qi * 3]);
        _mm_storeu_ps(dlanes.as_mut_ptr(), dp[qi * 3 + 1]);
        for lane in 0..4 {
            let k = qi + 1 + lane * q_count;
            if k <= m {
                msum += mlanes[lane];
                dsum += dlanes[lane];
            }
        }
    }

    let mut th = match row {
        18 => tracehash::th_call!("simd_forward_engine_row18_msum_q1e5"),
        19 => tracehash::th_call!("simd_forward_engine_row19_msum_q1e5"),
        _ => tracehash::th_call!("simd_forward_engine_row20_msum_q1e5"),
    };
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.output_f32_quant(msum, 1.0e-5);
    th.finish();

    let mut th = match row {
        18 => tracehash::th_call!("simd_forward_engine_row18_dsum_q1e5"),
        19 => tracehash::th_call!("simd_forward_engine_row19_dsum_q1e5"),
        _ => tracehash::th_call!("simd_forward_engine_row20_dsum_q1e5"),
    };
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.output_f32_quant(dsum, 1.0e-5);
    th.finish();
}

/// Tracehash helper: quantized M/D/I sums for the row that triggers the 10th
/// sparse-rescaling event (debug probe used to localize divergences).
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_forward_engine_scale10_row_q1e5(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q_count: usize,
    dp: &[__m128],
) {
    let mut sums = [0.0_f32; 3];
    for qi in 0..q_count {
        for state in 0..3 {
            let mut lanes = [0.0_f32; 4];
            _mm_storeu_ps(lanes.as_mut_ptr(), dp[qi * 3 + state]);
            for (lane, value) in lanes.iter().enumerate() {
                let k = qi + 1 + lane * q_count;
                if k <= m {
                    sums[state] += *value;
                }
            }
        }
    }

    macro_rules! emit_sum {
        ($name:literal, $value:expr) => {{
            let mut th = tracehash::th_call!($name);
            th.input_usize(l);
            th.input_usize(m);
            th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
            th.output_f32_quant($value, 1.0e-5);
            th.finish();
        }};
    }
    emit_sum!("simd_forward_engine_scale10_msum_q1e5", sums[0]);
    emit_sum!("simd_forward_engine_scale10_dsum_q1e5", sums[1]);
    emit_sum!("simd_forward_engine_scale10_isum_q1e5", sums[2]);
}

/// Tracehash helper: emits quantized M/D/I sums for selected rows around the
/// scale-10 window, tagged with the row index.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_forward_engine_scale10_window_row_q1e5(
    row: usize,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q_count: usize,
    dp: &[__m128],
) {
    let mut sums = [0.0_f32; 3];
    for qi in 0..q_count {
        for state in 0..3 {
            let mut lanes = [0.0_f32; 4];
            _mm_storeu_ps(lanes.as_mut_ptr(), dp[qi * 3 + state]);
            for (lane, value) in lanes.iter().enumerate() {
                let k = qi + 1 + lane * q_count;
                if k <= m {
                    sums[state] += *value;
                }
            }
        }
    }

    let mut th = tracehash::th_call!("simd_forward_engine_scale10_window_row_q1e5");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.input_usize(row);
    th.output_f32_quant(sums[0], 1.0e-5);
    th.output_f32_quant(sums[1], 1.0e-5);
    th.output_f32_quant(sums[2], 1.0e-5);
    th.finish();
}

/// Tracehash helper: emits exact-bit FNV-1a hashes of M/D/I cells for
/// scale-10 window rows (bit-level divergence probe).
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_forward_engine_scale10_window_row_bits(
    row: usize,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q_count: usize,
    dp: &[__m128],
) {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hashes = [FNV_OFFSET; 3];
    for qi in 0..q_count {
        for state in 0..3 {
            let mut lanes = [0.0_f32; 4];
            _mm_storeu_ps(lanes.as_mut_ptr(), dp[qi * 3 + state]);
            for (lane, value) in lanes.iter().enumerate() {
                let k = qi + 1 + lane * q_count;
                if k <= m {
                    hashes[state] ^= value.to_bits() as u64;
                    hashes[state] = hashes[state].wrapping_mul(FNV_PRIME);
                }
            }
        }
    }

    macro_rules! emit_state {
        ($name:literal, $hash:expr) => {{
            let mut th = tracehash::th_call!($name);
            th.input_usize(l);
            th.input_usize(m);
            th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
            th.input_usize(row);
            th.output_u64($hash);
            th.finish();
        }};
    }
    emit_state!("simd_forward_engine_scale10_window_m_bits", hashes[0]);
    emit_state!("simd_forward_engine_scale10_window_d_bits", hashes[1]);
    emit_state!("simd_forward_engine_scale10_window_i_bits", hashes[2]);
}

/// Tracehash helper: bit-exact M/D/I hashes captured immediately after each
/// sparse-rescale event, tagged by event index and row.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_forward_engine_scale_event_row_bits(
    event: usize,
    row: usize,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q_count: usize,
    dp: &[__m128],
) {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hashes = [FNV_OFFSET; 3];
    for qi in 0..q_count {
        for state in 0..3 {
            let mut lanes = [0.0_f32; 4];
            _mm_storeu_ps(lanes.as_mut_ptr(), dp[qi * 3 + state]);
            for (lane, value) in lanes.iter().enumerate() {
                let k = qi + 1 + lane * q_count;
                if k <= m {
                    hashes[state] ^= value.to_bits() as u64;
                    hashes[state] = hashes[state].wrapping_mul(FNV_PRIME);
                }
            }
        }
    }

    let mut th = tracehash::th_call!("simd_forward_engine_scale_event_row_bits");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.input_usize(event);
    th.output_u64(row as u64);
    th.output_u64(hashes[0]);
    th.output_u64(hashes[1]);
    th.output_u64(hashes[2]);
    th.finish();
}

/// Tracehash helper: quantized M/D/I sums for the "main" vs. "dd" phase of the
/// scale-10 row, including per-state main-phase sums.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_forward_engine_scale10_phase_q1e5(
    name: &'static str,
    row: usize,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q_count: usize,
    dp: &[__m128],
) {
    let mut sums = [0.0_f32; 3];
    for qi in 0..q_count {
        for state in 0..3 {
            let mut lanes = [0.0_f32; 4];
            _mm_storeu_ps(lanes.as_mut_ptr(), dp[qi * 3 + state]);
            for (lane, value) in lanes.iter().enumerate() {
                let k = qi + 1 + lane * q_count;
                if k <= m {
                    sums[state] += *value;
                }
            }
        }
    }

    macro_rules! emit_phase {
        ($trace_name:literal) => {{
            let mut th = tracehash::th_call!($trace_name);
            th.input_usize(l);
            th.input_usize(m);
            th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
            th.input_usize(row);
            th.output_f32_quant(sums[0], 1.0e-5);
            th.output_f32_quant(sums[1], 1.0e-5);
            th.output_f32_quant(sums[2], 1.0e-5);
            th.finish();
        }};
    }

    match name {
        "main" => emit_phase!("simd_forward_engine_scale10_phase_main_q1e5"),
        _ => emit_phase!("simd_forward_engine_scale10_phase_dd_q1e5"),
    }

    if name == "main" {
        macro_rules! emit_main_sum {
            ($trace_name:literal, $value:expr) => {{
                let mut th = tracehash::th_call!($trace_name);
                th.input_usize(l);
                th.input_usize(m);
                th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
                th.input_usize(row);
                th.output_f32_quant($value, 1.0e-5);
                th.finish();
            }};
        }
        emit_main_sum!("simd_forward_engine_scale10_phase_main_msum_q1e5", sums[0]);
        emit_main_sum!("simd_forward_engine_scale10_phase_main_dsum_q1e5", sums[1]);
        emit_main_sum!("simd_forward_engine_scale10_phase_main_isum_q1e5", sums[2]);
    }
}

/// Tracehash helper: bucketed M-state sums (5 q-index buckets) for the
/// scale-10 main phase, isolating which stripes drive divergence.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_forward_engine_scale10_phase_main_m_buckets_q1e5(
    row: usize,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    q_count: usize,
    dp: &[__m128],
) {
    let mut buckets = [0.0_f32; 5];
    for qi in 0..q_count {
        let mut lanes = [0.0_f32; 4];
        _mm_storeu_ps(lanes.as_mut_ptr(), dp[qi * 3]);
        let bucket = if qi < 8 {
            0
        } else if qi < 16 {
            1
        } else if qi < 32 {
            2
        } else if qi < 64 {
            3
        } else {
            4
        };
        for (lane, value) in lanes.iter().enumerate() {
            let k = qi + 1 + lane * q_count;
            if k <= m {
                buckets[bucket] += *value;
            }
        }
    }

    let mut th = tracehash::th_call!("simd_forward_engine_scale10_phase_main_m_buckets_q1e5");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.input_usize(row);
    for value in buckets {
        th.output_f32_quant(value, 1.0e-5);
    }
    th.finish();
}

/// Tracehash helper: per-lane bit-exact hash of the xEv vector at the
/// scale-10 trigger row.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_forward_engine_scale10_xev_bits(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    xev: __m128,
) {
    let mut lanes = [0.0_f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), xev);
    let mut th = tracehash::th_call!("simd_forward_engine_scale10_xev_lanes_bits");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    for lane in lanes {
        th.output_u64(lane.to_bits() as u64);
    }
    th.finish();
}

/// Tracehash helper: bit-exact value of the scalar xE at the scale-10 row.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_forward_engine_scale10_xe_bits(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    xe: f32,
) {
    let mut th = tracehash::th_call!("simd_forward_engine_scale10_xe_bits");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.output_u64(xe.to_bits() as u64);
    th.finish();
}

/// Tracehash helper: per-row "special step" probe — emits xEv lanes (pre
/// horizontal-sum), xE, and the row scale to pinpoint per-row divergences.
/// Mirrors the C helper used inside forward_engine.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_forward_engine_step_bits(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    row: usize,
    xev_before_hsum: __m128,
    xe: f32,
    row_scale: f32,
) {
    let mut lanes = [0.0_f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), xev_before_hsum);
    let mut th = tracehash::th_call!("forward_special_step_bits");
    th.input_usize(l);
    th.input_usize(m);
    th.input_usize(row);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    for lane in lanes {
        th.output_f32(lane);
    }
    th.output_f32(xe);
    th.output_f32(row_scale);
    th.finish();
}

/// Tracehash helper: bit-exact dump of (xN, xJ, xB, xC) at the start of the
/// scale-10 trigger row, plus individual per-special probes.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_forward_engine_scale10_row_start_bits(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    row: usize,
    xn: f32,
    xj: f32,
    xb: f32,
    xc: f32,
) {
    let mut th = tracehash::th_call!("simd_forward_engine_scale10_row_start_bits");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.input_usize(row);
    th.output_u64(xn.to_bits() as u64);
    th.output_u64(xj.to_bits() as u64);
    th.output_u64(xb.to_bits() as u64);
    th.output_u64(xc.to_bits() as u64);
    th.finish();

    macro_rules! emit_special {
        ($name:literal, $value:expr) => {{
            let mut th = tracehash::th_call!($name);
            th.input_usize(l);
            th.input_usize(m);
            th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
            th.input_usize(row);
            th.output_u64($value.to_bits() as u64);
            th.finish();
        }};
    }
    emit_special!("simd_forward_engine_scale10_row_start_xn_bits", xn);
    emit_special!("simd_forward_engine_scale10_row_start_xj_bits", xj);
    emit_special!("simd_forward_engine_scale10_row_start_xb_bits", xb);
    emit_special!("simd_forward_engine_scale10_row_start_xc_bits", xc);
}

/// Tracehash helper: per-lane bit-exact hash of xEv at row 19 (canonical
/// divergence-localization row).
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
unsafe fn trace_forward_engine_row19_xev_bits(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    xev: __m128,
) {
    let mut lanes = [0.0_f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), xev);
    let mut th = tracehash::th_call!("simd_forward_engine_row19_xev_lanes_bits");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    for lane in lanes {
        th.output_u64(lane.to_bits() as u64);
    }
    th.finish();
}

/// Tracehash helper: bit-exact value of the scalar xE at row 19.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_forward_engine_row19_xe_bits(dsq: &[Dsq], dsq_offset: usize, l: usize, m: usize, xe: f32) {
    let mut th = tracehash::th_call!("simd_forward_engine_row19_xe_bits");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.output_u64(xe.to_bits() as u64);
    th.finish();
}

/// Tracehash helper: emits row, pre-scale xE bits, and quantized specials
/// (xN/xJ/xB/xC) for the very first sparse-rescaling event. Mirrors the
/// same-named C helper in fwdback.c.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_forward_engine_first_scale_q1e5(
    do_full: bool,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    row: usize,
    pre_xe: f32,
    xn: f32,
    xj: f32,
    xb: f32,
    xc: f32,
) {
    let mut th = tracehash::th_call!("simd_forward_engine_first_scale_q1e5");
    th.input_bool(do_full);
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.output_u64(row as u64);
    th.output_u64(pre_xe.to_bits() as u64);
    th.output_f32_quant(xn, 1.0e-5);
    th.output_f32_quant(xj, 1.0e-5);
    th.output_f32_quant(xb, 1.0e-5);
    th.output_f32_quant(xc, 1.0e-5);
    th.finish();

    let mut th = tracehash::th_call!("simd_forward_engine_first_scale_row");
    th.input_bool(do_full);
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.output_u64(row as u64);
    th.finish();

    let mut th = tracehash::th_call!("simd_forward_engine_first_scale_xe_bits");
    th.input_bool(do_full);
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.output_u64(pre_xe.to_bits() as u64);
    th.finish();

    let mut th = tracehash::th_call!("simd_forward_engine_first_scale_post_specials_q1e5");
    th.input_bool(do_full);
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.output_f32_quant(xn, 1.0e-5);
    th.output_f32_quant(xj, 1.0e-5);
    th.output_f32_quant(xb, 1.0e-5);
    th.output_f32_quant(xc, 1.0e-5);
    th.finish();
}

/// Tracehash helper: detailed per-event probe — emits the event index, row,
/// bit-exact pre-scale xE, and quantized post-scale specials.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_forward_engine_scale_event_detail_q1e5(
    event: usize,
    do_full: bool,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    row: usize,
    pre_xe: f32,
    xn: f32,
    xj: f32,
    xb: f32,
    xc: f32,
) {
    let mut th = tracehash::th_call!("simd_forward_engine_scale_event_detail");
    th.input_bool(do_full);
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.input_usize(event);
    th.output_u64(row as u64);
    th.output_u64(pre_xe.to_bits() as u64);
    th.output_f32_quant(xn, 1.0e-5);
    th.output_f32_quant(xj, 1.0e-5);
    th.output_f32_quant(xb, 1.0e-5);
    th.output_f32_quant(xc, 1.0e-5);
    th.finish();

    let mut th = tracehash::th_call!("simd_forward_engine_scale_event_row");
    th.input_bool(do_full);
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.input_usize(event);
    th.output_u64(row as u64);
    th.finish();

    let mut th = tracehash::th_call!("simd_forward_engine_scale_event_xe_bits");
    th.input_bool(do_full);
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
    th.input_usize(event);
    th.output_u64(pre_xe.to_bits() as u64);
    th.finish();
}

/// Tracehash helper: per-event trace dispatch — calls `first_scale` for event
/// 1 and emits per-event row/xE/specials probes for events 2..=5+.
#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_forward_engine_scale_event_q1e5(
    event: usize,
    do_full: bool,
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    m: usize,
    row: usize,
    pre_xe: f32,
    xn: f32,
    xj: f32,
    xb: f32,
    xc: f32,
) {
    if event == 1 {
        trace_forward_engine_first_scale_q1e5(
            do_full, dsq, dsq_offset, l, m, row, pre_xe, xn, xj, xb, xc,
        );
        return;
    }

    macro_rules! trace_split {
        ($row_name:literal, $xe_name:literal, $sp_name:literal) => {{
            let mut th = tracehash::th_call!($row_name);
            th.input_bool(do_full);
            th.input_usize(l);
            th.input_usize(m);
            th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
            th.output_u64(row as u64);
            th.finish();

            let mut th = tracehash::th_call!($xe_name);
            th.input_bool(do_full);
            th.input_usize(l);
            th.input_usize(m);
            th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
            th.output_u64(pre_xe.to_bits() as u64);
            th.finish();

            let mut th = tracehash::th_call!($sp_name);
            th.input_bool(do_full);
            th.input_usize(l);
            th.input_usize(m);
            th.input_bytes(&dsq[dsq_offset + 1..=dsq_offset + l]);
            th.output_f32_quant(xn, 1.0e-5);
            th.output_f32_quant(xj, 1.0e-5);
            th.output_f32_quant(xb, 1.0e-5);
            th.output_f32_quant(xc, 1.0e-5);
            th.finish();
        }};
    }

    match event {
        2 => trace_split!(
            "simd_forward_engine_scale2_row",
            "simd_forward_engine_scale2_xe_bits",
            "simd_forward_engine_scale2_post_specials_q1e5"
        ),
        3 => trace_split!(
            "simd_forward_engine_scale3_row",
            "simd_forward_engine_scale3_xe_bits",
            "simd_forward_engine_scale3_post_specials_q1e5"
        ),
        4 => trace_split!(
            "simd_forward_engine_scale4_row",
            "simd_forward_engine_scale4_xe_bits",
            "simd_forward_engine_scale4_post_specials_q1e5"
        ),
        _ => trace_split!(
            "simd_forward_engine_scale_last_row",
            "simd_forward_engine_scale_last_xe_bits",
            "simd_forward_engine_scale_last_post_specials_q1e5"
        ),
    }
}

/// SSE Forward algorithm in linear (parser) memory. Returns score in nats.
///
/// Port of `p7_Forward` / `p7_ForwardParser` (impl_sse/fwdback.c) restricted to
/// parser mode (no full DP matrix kept). Computes the Forward score over the
/// digital sequence `dsq[1..=l]` against optimized profile `om` using the
/// sparse-rescaling scheme to stay inside f32 dynamic range. Convenience
/// wrapper around `forward_parser_offset` with `dsq_offset = 0`.
///
/// Profile must be in local alignment mode (sparse rescaling is unsafe for
/// glocal/global).
///
/// # Safety
/// Requires SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser(dsq: &[Dsq], l: usize, om: &OProfile) -> f32 {
    forward_parser_offset(dsq, 0, l, om)
}

/// Forward parser score over `dsq[dsq_offset+1..=dsq_offset+l]`.
///
/// Variant of `p7_Forward` (impl_sse/fwdback.c) that accepts an explicit
/// `dsq_offset` so callers can score a window without copying the digital
/// sequence. Otherwise identical to [`forward_parser`]: striped SIMD MDI
/// recurrence, scalar BENCJ specials, sparse rescaling, and a final
/// `totscale + ln(xC * tCT)` return in nats.
///
/// # Safety
/// Requires SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser_offset(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
) -> f32 {
    let q_count = nqf(om.m);
    let nscells = 3; // M, D, I

    // One-row DP matrix (parser mode)
    let mut dp: Vec<__m128> = vec![_mm_setzero_ps(); q_count * nscells];

    let zerov = _mm_setzero_ps();

    // Initialize special states
    let mut xe: f32 = 0.0;
    let mut xn: f32 = 1.0;
    let mut xj: f32 = 0.0;
    let mut xb: f32 = om.xf[P7O_N][P7O_MOVE];
    let mut xc: f32 = 0.0;
    let mut totscale: f32 = 0.0;

    macro_rules! mmo {
        ($q:expr) => {
            dp[$q * nscells + 0]
        };
    }
    macro_rules! dmo {
        ($q:expr) => {
            dp[$q * nscells + 1]
        };
    }
    macro_rules! imo {
        ($q:expr) => {
            dp[$q * nscells + 2]
        };
    }

    for i in 1..=l {
        // C indexes `om->rfv[dsq[i]]` unconditionally (fwdback.c:952-1176): rfv is
        // filled for all Kp codes (oprofile.rs builds it `for x in 0..kp`,
        // including degenerate X/*/~), and every valid digital code is < Kp, so the
        // row always exists and the recurrence (with its special-state updates)
        // must advance. Matches the unconditional indexing in the MSV/Viterbi
        // filters and the pmx Forward variant.
        let xi = dsq[dsq_offset + i] as usize;

        // Save previous row (parser: same memory, but we swap via registers)
        // In parser mode, dpp == dpc (same row). We use mpv/dpv/ipv registers.
        let rsc = &om.rfv[xi];
        let mut dcv = zerov;
        let mut xev = zerov;
        let xbv = _mm_set1_ps(xb);

        // Right-shift by 1 float (4 bytes), zero-fill
        let mut mpv = rightshift_float(mmo!(q_count - 1));
        let mut dpv = rightshift_float(dmo!(q_count - 1));
        let mut ipv = rightshift_float(imo!(q_count - 1));

        let mut tsc_idx = 0;

        for q in 0..q_count {
            // Match: B*tBM + M*tMM + I*tIM + D*tDM, then * emission
            let tbm = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tmm = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tim = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tdm = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;

            let mut sv = _mm_mul_ps(xbv, tbm);
            sv = _mm_add_ps(sv, _mm_mul_ps(mpv, tmm));
            sv = _mm_add_ps(sv, _mm_mul_ps(ipv, tim));
            sv = _mm_add_ps(sv, _mm_mul_ps(dpv, tdm));

            let rsc_v = _mm_loadu_ps(rsc[q].as_ptr());

            sv = _mm_mul_ps(sv, rsc_v);
            xev = _mm_add_ps(xev, sv);

            // Save previous before overwriting
            mpv = mmo!(q);
            dpv = dmo!(q);
            ipv = imo!(q);

            // Store M and D
            mmo!(q) = sv;
            dmo!(q) = dcv;

            // M->D partial for next q
            let tmd = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            dcv = _mm_mul_ps(sv, tmd);

            // I state: M*tMI + I*tII (emission odds ratio = 1.0, so no emission multiply)
            let tmi = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let tii = _mm_loadu_ps(om.tfv[tsc_idx].as_ptr());
            tsc_idx += 1;
            let isv = _mm_add_ps(_mm_mul_ps(mpv, tmi), _mm_mul_ps(ipv, tii));
            imo!(q) = isv;
        }

        // DD paths: first mandatory pass
        dcv = rightshift_float(dcv);
        dmo!(0) = zerov;
        let dd_offset = 7 * q_count;
        for q in 0..q_count {
            dmo!(q) = _mm_add_ps(dcv, dmo!(q));
            let tdd = _mm_loadu_ps(om.tfv[dd_offset + q].as_ptr());
            dcv = _mm_mul_ps(dmo!(q), tdd);
        }

        // Up to 3 more DD passes
        if om.m < 100 {
            // Full serialization for small models
            for _ in 1..4 {
                dcv = rightshift_float(dcv);
                for q in 0..q_count {
                    dmo!(q) = _mm_add_ps(dcv, dmo!(q));
                    let tdd = _mm_loadu_ps(om.tfv[dd_offset + q].as_ptr());
                    dcv = _mm_mul_ps(dcv, tdd);
                }
            }
        } else {
            // Parallel with early termination for large models
            for _ in 1..4 {
                dcv = rightshift_float(dcv);
                let mut cv = zerov;
                for q in 0..q_count {
                    let sv = _mm_add_ps(dcv, dmo!(q));
                    cv = _mm_or_ps(cv, _mm_cmpgt_ps(sv, dmo!(q)));
                    dmo!(q) = sv;
                    let tdd = _mm_loadu_ps(om.tfv[dd_offset + q].as_ptr());
                    dcv = _mm_mul_ps(dcv, tdd);
                }
                if _mm_movemask_ps(cv) == 0 {
                    break;
                }
            }
        }

        // Add D's to xEv
        for q in 0..q_count {
            xev = _mm_add_ps(dmo!(q), xev);
        }

        // Horizontal sum of xEv -> xE
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xev, xev),
        );
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xev, xev),
        );
        _mm_store_ss(&mut xe, xev);

        // Special states (scalar, probability space)
        xn *= om.xf[P7O_N][P7O_LOOP];
        xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
        xb = xj * om.xf[P7O_J][P7O_MOVE] + xn * om.xf[P7O_N][P7O_MOVE];

        // Sparse rescaling when xE gets large
        if xe > 1.0e4 {
            let scale = 1.0 / xe;
            xn /= xe;
            xc /= xe;
            xj /= xe;
            xb /= xe;
            let scale_v = _mm_set1_ps(scale);
            for q in 0..q_count {
                mmo!(q) = _mm_mul_ps(mmo!(q), scale_v);
                dmo!(q) = _mm_mul_ps(dmo!(q), scale_v);
                imo!(q) = _mm_mul_ps(imo!(q), scale_v);
            }
            totscale += c_log_f64(xe as f64) as f32;
            xe = 1.0;
        }
    }

    // Final score: totscale + log(xC * C->T)
    if xc.is_nan() || (l > 0 && xc == 0.0) || xc.is_infinite() {
        return f32::NEG_INFINITY; // error conditions
    }

    totscale + c_log_f64((xc * om.xf[P7O_C][P7O_MOVE]) as f64) as f32
}

/// Forward parser writing specials (xE/N/J/B/C), cumulative f64 scale and
/// per-row scale factors into a `ProbMx`. Returns Forward score in nats.
///
/// Maps to C `p7_ForwardParser` (impl_sse/fwdback.c). Like the C engine, this
/// keeps the running totscale in f64 (then narrows on store) so that posterior
/// decoding normalization stays bit-equivalent to the original codebase. If
/// `pmx.has_dp` is set the full striped DP matrix is also written.
///
/// # Safety
/// Requires SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser_pmx(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
) -> f32 {
    forward_parser_pmx_offset(dsq, 0, l, om, pmx)
}

/// `p7_ForwardParser` variant that scores a window starting at `dsq_offset`.
///
/// Allocates a fresh per-call SIMD scratch buffer; for hot loops prefer
/// [`forward_parser_pmx_offset_with_scratch`] which lets callers amortize the
/// allocation. Otherwise semantically identical to [`forward_parser_pmx`].
///
/// # Safety
/// Requires SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser_pmx_offset(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
) -> f32 {
    let mut dp: Vec<__m128> = Vec::new();
    forward_parser_pmx_offset_with_scratch(dsq, dsq_offset, l, om, pmx, &mut dp)
}

/// Core Forward engine in parser mode with a caller-supplied SIMD scratch
/// buffer. Returns the Forward score in nats.
///
/// Variant of C `forward_engine` (static in impl_sse/fwdback.c) — the fused
/// engine that backs both `p7_Forward` (full matrix) and `p7_ForwardParser`
/// (linear memory). Implements the striped MDI recurrence, D->D wing
/// unrolling (with serial vs. parallel-with-early-termination branches), xE
/// horizontal sum, scalar BENCJ specials, sparse rescaling triggered by
/// `xE > 1e4`, and accumulation of `totscale` in f64. Writes specials and
/// row scales to `pmx`; if `pmx.has_dp` is set and the sequence is canonical
/// (no degenerate residues) it dispatches to the direct full-DP variant.
///
/// # Safety
/// Requires SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser_pmx_offset_with_scratch(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
    dp: &mut Vec<__m128>,
) -> f32 {
    const USE_DIRECT_FULL_DP_FORWARD: bool = true;
    if USE_DIRECT_FULL_DP_FORWARD && pmx.has_dp && canonical_run(dsq, dsq_offset, l, om.abc_kp) {
        return forward_parser_pmx_offset_direct(dsq, dsq_offset, l, om, pmx);
    }

    use super::probmx::*;
    let q_count = nqf(om.m);
    let nscells = 3;
    let zerov = _mm_setzero_ps();
    dp.resize(q_count * nscells, zerov);
    for v in dp.iter_mut() {
        *v = zerov;
    }
    let dp_ptr = dp.as_mut_ptr();
    let rfv_ptr = om.rfv_a.as_ptr();
    let tfv_ptr = om.tfv_a.as_ptr();

    let mut xe: f32 = 0.0;
    let mut xn: f32 = 1.0;
    let mut xj: f32 = 0.0;
    let mut xb: f32 = om.xf[P7O_N][P7O_MOVE];
    let mut xc: f32 = 0.0;
    let mut totscale: f32 = 0.0;
    #[cfg(feature = "tracehash")]
    let mut trace_scale_event_count = 0usize;
    #[cfg(feature = "tracehash")]
    let mut trace_last_scale_event = None;

    macro_rules! mmo {
        ($q:expr) => {
            *dp_ptr.add($q * nscells)
        };
    }
    macro_rules! dmo {
        ($q:expr) => {
            *dp_ptr.add($q * nscells + 1)
        };
    }
    macro_rules! imo {
        ($q:expr) => {
            *dp_ptr.add($q * nscells + 2)
        };
    }

    // Row 0
    pmx.set_xmx(0, PXE, 0.0);
    pmx.set_xmx(0, PXN, xn);
    pmx.set_xmx(0, PXJ, 0.0);
    pmx.set_xmx(0, PXB, xb);
    pmx.set_xmx(0, PXC, 0.0);
    pmx.scale[0] = 0.0;
    pmx.row_scale[0] = 1.0;

    for i in 1..=l {
        let xi = dsq[dsq_offset + i] as usize;
        // Defensive branch for out-of-range codes (>= Kp), which never occur in a
        // well-formed dsq (all valid codes, including degenerate X/*/~, are < Kp
        // and have filled rfv rows). C has no such branch and would read OOB here;
        // for valid input this is a no-op and the normal recurrence below runs.
        if xi >= om.abc_kp {
            if pmx.has_dp {
                pmx.zero_simd_row(i);
            }
            xn *= om.xf[P7O_N][P7O_LOOP];
            xe = 0.0;
            xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
            xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
            xb = xn * om.xf[P7O_N][P7O_MOVE] + xj * om.xf[P7O_J][P7O_MOVE];
            pmx.set_xmx(i, PXE, xe);
            pmx.set_xmx(i, PXN, xn);
            pmx.set_xmx(i, PXJ, xj);
            pmx.set_xmx(i, PXB, xb);
            pmx.set_xmx(i, PXC, xc);
            pmx.scale[i] = totscale as f64;
            pmx.row_scale[i] = 1.0;
            continue;
        }

        let rsc_ptr = (*rfv_ptr.add(xi)).as_ptr();
        let xbv = _mm_set1_ps(xb);
        let mut dcv = zerov;
        let mut xev = zerov;
        #[cfg(feature = "tracehash")]
        if (l == 1130 && i == 919) || (l == 465 && i == 291) {
            trace_forward_engine_scale10_row_start_bits(
                dsq, dsq_offset, l, om.m, i, xn, xj, xb, xc,
            );
        }

        let mut mpv = rightshift_float(mmo!(q_count - 1));
        let mut dpv = rightshift_float(dmo!(q_count - 1));
        let mut ipv = rightshift_float(imo!(q_count - 1));

        for q_idx in 0..q_count {
            let tsc_base = q_idx * 7;
            let rsc_v = _mm_load_ps((*rsc_ptr.add(q_idx)).as_ptr());

            let tbm = _mm_load_ps((*tfv_ptr.add(tsc_base)).as_ptr());
            let tmm = _mm_load_ps((*tfv_ptr.add(tsc_base + 1)).as_ptr());
            let tim = _mm_load_ps((*tfv_ptr.add(tsc_base + 2)).as_ptr());
            let tdm = _mm_load_ps((*tfv_ptr.add(tsc_base + 3)).as_ptr());
            let tmd = _mm_load_ps((*tfv_ptr.add(tsc_base + 4)).as_ptr());
            let tmi = _mm_load_ps((*tfv_ptr.add(tsc_base + 5)).as_ptr());
            let tii = _mm_load_ps((*tfv_ptr.add(tsc_base + 6)).as_ptr());

            // M(i,k) = (B*BM + Mprev*MM + Iprev*IM + Dprev*DM) * emission
            let mut sv = _mm_mul_ps(xbv, tbm);
            sv = _mm_add_ps(sv, _mm_mul_ps(mpv, tmm));
            sv = _mm_add_ps(sv, _mm_mul_ps(ipv, tim));
            sv = _mm_add_ps(sv, _mm_mul_ps(dpv, tdm));
            sv = _mm_mul_ps(sv, rsc_v);
            xev = _mm_add_ps(xev, sv);

            mpv = mmo!(q_idx);
            dpv = dmo!(q_idx);
            ipv = imo!(q_idx);

            mmo!(q_idx) = sv;
            dmo!(q_idx) = dcv;

            dcv = _mm_mul_ps(sv, tmd);

            // I(i,k) = (Mprev*MI + Iprev*II)  (parser mode: insert emission = 1.0)
            imo!(q_idx) = _mm_add_ps(_mm_mul_ps(mpv, tmi), _mm_mul_ps(ipv, tii));
        }

        #[cfg(feature = "tracehash")]
        if (l == 1130 && i == 919) || (l == 465 && i == 291) {
            trace_forward_engine_scale10_phase_q1e5(
                "main", i, dsq, dsq_offset, l, om.m, q_count, dp,
            );
            trace_forward_engine_scale10_phase_main_m_buckets_q1e5(
                i, dsq, dsq_offset, l, om.m, q_count, dp,
            );
        }

        // D->D wing unfolding
        {
            let dd_offset = 7 * q_count;
            dcv = rightshift_float(dcv);
            dmo!(0) = zerov;
            for q_idx in 0..q_count {
                let tdd = _mm_load_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                dmo!(q_idx) = _mm_add_ps(dcv, dmo!(q_idx));
                dcv = _mm_mul_ps(dmo!(q_idx), tdd);
            }
            if om.m < 100 {
                for _ in 0..3 {
                    dcv = rightshift_float(dcv);
                    for q_idx in 0..q_count {
                        let tdd = _mm_load_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                        dmo!(q_idx) = _mm_add_ps(dcv, dmo!(q_idx));
                        dcv = _mm_mul_ps(dcv, tdd);
                    }
                }
            } else {
                for _ in 0..3 {
                    dcv = rightshift_float(dcv);
                    let mut cv = zerov;
                    for q_idx in 0..q_count {
                        let tdd = _mm_load_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                        let sv = _mm_add_ps(dcv, dmo!(q_idx));
                        cv = _mm_or_ps(cv, _mm_cmpgt_ps(sv, dmo!(q_idx)));
                        dmo!(q_idx) = sv;
                        dcv = _mm_mul_ps(dcv, tdd);
                    }
                    if _mm_movemask_ps(cv) == 0 {
                        break;
                    }
                }
            }
        }

        #[cfg(feature = "tracehash")]
        if (l == 1130 && i == 919) || (l == 465 && i == 291) {
            trace_forward_engine_scale10_phase_q1e5("dd", i, dsq, dsq_offset, l, om.m, q_count, dp);
        }

        // E state = sum(M) + sum(D)
        for q_idx in 0..q_count {
            xev = _mm_add_ps(dmo!(q_idx), xev);
        }
        #[cfg(feature = "tracehash")]
        if i == 18 || i == 19 || i == 20 {
            trace_forward_engine_row_sums_q1e5(i, dsq, dsq_offset, l, om.m, q_count, dp);
        }
        #[cfg(feature = "tracehash")]
        if i == 19 {
            trace_forward_engine_row19_xev_bits(dsq, dsq_offset, l, om.m, xev);
        }
        #[cfg(feature = "tracehash")]
        if (l == 1130 && i == 919) || (l == 465 && i == 292) {
            trace_forward_engine_scale10_row_q1e5(dsq, dsq_offset, l, om.m, q_count, dp);
            trace_forward_engine_scale10_xev_bits(dsq, dsq_offset, l, om.m, xev);
        }
        #[cfg(feature = "tracehash")]
        if (l == 1130 && matches!(i, 847 | 860 | 880 | 900 | 918 | 919))
            || (l == 465 && matches!(i, 232 | 240 | 260 | 280 | 285 | 288 | 289 | 290 | 291 | 292))
        {
            trace_forward_engine_scale10_window_row_q1e5(i, dsq, dsq_offset, l, om.m, q_count, dp);
        }
        #[cfg(feature = "tracehash")]
        if (l == 1130 && matches!(i, 918 | 919))
            || (l == 465 && matches!(i, 280 | 285 | 288 | 289 | 290 | 291 | 292))
        {
            trace_forward_engine_scale10_window_row_bits(i, dsq, dsq_offset, l, om.m, q_count, dp);
        }
        #[cfg(feature = "tracehash")]
        let trace_xev_before_hsum = xev;
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xev, xev),
        );
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xev, xev),
        );
        _mm_store_ss(&mut xe, xev);
        #[cfg(feature = "tracehash")]
        if i == 19 {
            trace_forward_engine_row19_xe_bits(dsq, dsq_offset, l, om.m, xe);
        }
        #[cfg(feature = "tracehash")]
        if (l == 1130 && i == 919) || (l == 465 && i == 292) {
            trace_forward_engine_scale10_xe_bits(dsq, dsq_offset, l, om.m, xe);
        }

        // Special states
        xn *= om.xf[P7O_N][P7O_LOOP];
        xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
        xb = xj * om.xf[P7O_J][P7O_MOVE] + xn * om.xf[P7O_N][P7O_MOVE];

        // Sparse rescaling
        let row_scale = if xe > 1.0e4 {
            let row_scale = xe;
            let inv_xe = 1.0 / xe;
            let scalev = _mm_set1_ps(inv_xe);
            xn /= row_scale;
            xc /= row_scale;
            xj /= row_scale;
            xb /= row_scale;
            #[cfg(feature = "tracehash")]
            {
                trace_scale_event_count += 1;
                trace_forward_engine_scale_event_detail_q1e5(
                    trace_scale_event_count,
                    pmx.has_dp,
                    dsq,
                    dsq_offset,
                    l,
                    om.m,
                    i,
                    row_scale,
                    xn,
                    xj,
                    xb,
                    xc,
                );
                if trace_scale_event_count <= 4 {
                    trace_forward_engine_scale_event_q1e5(
                        trace_scale_event_count,
                        pmx.has_dp,
                        dsq,
                        dsq_offset,
                        l,
                        om.m,
                        i,
                        row_scale,
                        xn,
                        xj,
                        xb,
                        xc,
                    );
                }
                trace_last_scale_event = Some((i, row_scale, xn, xj, xb, xc));
            }
            for q_idx in 0..(q_count * nscells) {
                let p = dp_ptr.add(q_idx);
                *p = _mm_mul_ps(*p, scalev);
            }
            #[cfg(feature = "tracehash")]
            if matches!(l, 465 | 1130) && trace_scale_event_count <= 10 {
                trace_forward_engine_scale_event_row_bits(
                    trace_scale_event_count,
                    i,
                    dsq,
                    dsq_offset,
                    l,
                    om.m,
                    q_count,
                    dp,
                );
            }
            totscale = (totscale as f64 + c_log_f64(xe as f64)) as f32;
            xe = 1.0;
            row_scale
        } else {
            1.0
        };
        #[cfg(feature = "tracehash")]
        if i <= 8 || i == l {
            trace_forward_engine_step_bits(
                dsq,
                dsq_offset,
                l,
                om.m,
                i,
                trace_xev_before_hsum,
                xe,
                row_scale,
            );
        }

        // Store full DP row if requested (for posterior decoding / null2)
        if pmx.has_dp {
            pmx.write_simd_row(&dp, q_count, om.m, i);
        }

        pmx.set_xmx(i, PXE, xe);
        pmx.set_xmx(i, PXN, xn);
        pmx.set_xmx(i, PXJ, xj);
        pmx.set_xmx(i, PXB, xb);
        pmx.set_xmx(i, PXC, xc);
        pmx.scale[i] = totscale as f64;
        pmx.row_scale[i] = row_scale;
    }

    #[cfg(feature = "tracehash")]
    if let Some((row, pre_xe, last_xn, last_xj, last_xb, last_xc)) = trace_last_scale_event {
        trace_forward_engine_scale_event_q1e5(
            0, pmx.has_dp, dsq, dsq_offset, l, om.m, row, pre_xe, last_xn, last_xj, last_xb,
            last_xc,
        );
    }

    let score = if xc.is_nan() || (l > 0 && xc == 0.0) || xc.is_infinite() {
        f32::NEG_INFINITY
    } else {
        (totscale as f64 + c_log_f64((xc * om.xf[P7O_C][P7O_MOVE]) as f64)) as f32
    };
    score
}

/// Canonical-sequence fast path: enters the direct full-DP variant of
/// `forward_engine` unconditionally (caller guarantees no degenerate residues
/// and `pmx.has_dp`). Variant of `forward_engine` that skips the dispatch
/// check inside [`forward_parser_pmx_offset_with_scratch`].
///
/// # Safety
/// Requires SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn forward_parser_pmx_offset_canonical(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
) -> f32 {
    debug_assert!(pmx.has_dp);
    forward_parser_pmx_offset_direct(dsq, dsq_offset, l, om, pmx)
}

/// Direct full-DP forward engine: writes each row straight into the striped
/// posterior matrix in `pmx`, reading the previous row from the same buffer
/// (no per-call scratch). Variant of `p7_Forward` / `forward_engine` for the
/// canonical, full-matrix case used by posterior decoding and null2.
///
/// # Safety
/// Requires SSE2; caller guarantees `pmx.has_dp` and that `dsq[1..=l]`
/// contains no degenerate residues.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn forward_parser_pmx_offset_direct(
    dsq: &[Dsq],
    dsq_offset: usize,
    l: usize,
    om: &OProfile,
    pmx: &mut super::probmx::ProbMx,
) -> f32 {
    use super::probmx::*;

    let q_count = nqf(om.m);
    let row_width = pmx.striped_row_width();
    let zerov = _mm_setzero_ps();
    let dsq_ptr = dsq.as_ptr().add(dsq_offset);
    let striped_ptr = pmx.striped_dp.as_mut_ptr().add(pmx.striped_dp_offset);
    let xmx_ptr = pmx.xmx.as_mut_ptr();
    let scale_ptr = pmx.scale.as_mut_ptr();
    let row_scale_ptr = pmx.row_scale.as_mut_ptr();
    let rfv_ptr = om.rfv_a.as_ptr();
    let tfv_ptr = om.tfv_a.as_ptr();

    let mut xe: f32 = 0.0;
    let mut xn: f32 = 1.0;
    let mut xj: f32 = 0.0;
    let mut xb: f32 = om.xf[P7O_N][P7O_MOVE];
    let mut xc: f32 = 0.0;
    let mut totscale: f32 = 0.0;
    #[cfg(feature = "tracehash")]
    let mut trace_scale_event_count = 0usize;
    #[cfg(feature = "tracehash")]
    let mut trace_last_scale_event = None;

    /// Loads one striped MDI cell (`s` in 0=M, 1=D, 2=I) from a row pointer.
    #[inline(always)]
    unsafe fn load_cell(row: *const f32, q: usize, s: usize) -> __m128 {
        _mm_load_ps(row.add(q * 12 + s * 4))
    }

    /// Stores one striped MDI cell into the row pointer.
    #[inline(always)]
    unsafe fn store_cell(row: *mut f32, q: usize, s: usize, v: __m128) {
        _mm_store_ps(row.add(q * 12 + s * 4), v);
    }

    /// Writes the five special-state values (E/N/J/B/C) for row `i` into `xmx`.
    #[inline(always)]
    unsafe fn store_xmx(xmx: *mut f32, i: usize, xe: f32, xn: f32, xj: f32, xb: f32, xc: f32) {
        let row = xmx.add(i * 5);
        *row.add(PXE) = xe;
        *row.add(PXN) = xn;
        *row.add(PXJ) = xj;
        *row.add(PXB) = xb;
        *row.add(PXC) = xc;
    }

    store_xmx(xmx_ptr, 0, 0.0, xn, 0.0, xb, 0.0);
    *scale_ptr = 0.0;
    *row_scale_ptr = 1.0;

    for i in 1..=l {
        let prev_row = striped_ptr.add((i - 1) * row_width) as *const f32;
        let curr_row = striped_ptr.add(i * row_width);
        let xi = *dsq_ptr.add(i) as usize;
        let rsc_ptr = (*rfv_ptr.add(xi)).as_ptr();
        let xbv = _mm_set1_ps(xb);
        let mut dcv = zerov;
        let mut xev = zerov;

        let mut mpv = rightshift_float(load_cell(prev_row, q_count - 1, 0));
        let mut dpv = rightshift_float(load_cell(prev_row, q_count - 1, 1));
        let mut ipv = rightshift_float(load_cell(prev_row, q_count - 1, 2));

        for q_idx in 0..q_count {
            let tsc_base = q_idx * 7;
            let rsc_v = _mm_load_ps((*rsc_ptr.add(q_idx)).as_ptr());

            let tbm = _mm_load_ps((*tfv_ptr.add(tsc_base)).as_ptr());
            let tmm = _mm_load_ps((*tfv_ptr.add(tsc_base + 1)).as_ptr());
            let tim = _mm_load_ps((*tfv_ptr.add(tsc_base + 2)).as_ptr());
            let tdm = _mm_load_ps((*tfv_ptr.add(tsc_base + 3)).as_ptr());
            let tmd = _mm_load_ps((*tfv_ptr.add(tsc_base + 4)).as_ptr());
            let tmi = _mm_load_ps((*tfv_ptr.add(tsc_base + 5)).as_ptr());
            let tii = _mm_load_ps((*tfv_ptr.add(tsc_base + 6)).as_ptr());

            let mut sv = _mm_mul_ps(xbv, tbm);
            sv = _mm_add_ps(sv, _mm_mul_ps(mpv, tmm));
            sv = _mm_add_ps(sv, _mm_mul_ps(ipv, tim));
            sv = _mm_add_ps(sv, _mm_mul_ps(dpv, tdm));
            sv = _mm_mul_ps(sv, rsc_v);
            xev = _mm_add_ps(xev, sv);

            mpv = load_cell(prev_row, q_idx, 0);
            dpv = load_cell(prev_row, q_idx, 1);
            ipv = load_cell(prev_row, q_idx, 2);

            store_cell(curr_row, q_idx, 0, sv);
            store_cell(curr_row, q_idx, 1, dcv);

            dcv = _mm_mul_ps(sv, tmd);
            store_cell(
                curr_row,
                q_idx,
                2,
                _mm_add_ps(_mm_mul_ps(mpv, tmi), _mm_mul_ps(ipv, tii)),
            );
        }

        {
            let dd_offset = 7 * q_count;
            dcv = rightshift_float(dcv);
            store_cell(curr_row, 0, 1, zerov);
            for q_idx in 0..q_count {
                let tdd = _mm_load_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                let d = _mm_add_ps(dcv, load_cell(curr_row, q_idx, 1));
                store_cell(curr_row, q_idx, 1, d);
                dcv = _mm_mul_ps(d, tdd);
            }
            if om.m < 100 {
                for _ in 0..3 {
                    dcv = rightshift_float(dcv);
                    for q_idx in 0..q_count {
                        let tdd = _mm_load_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                        let d = _mm_add_ps(dcv, load_cell(curr_row, q_idx, 1));
                        store_cell(curr_row, q_idx, 1, d);
                        dcv = _mm_mul_ps(dcv, tdd);
                    }
                }
            } else {
                for _ in 0..3 {
                    dcv = rightshift_float(dcv);
                    let mut cv = zerov;
                    for q_idx in 0..q_count {
                        let tdd = _mm_load_ps((*tfv_ptr.add(dd_offset + q_idx)).as_ptr());
                        let old_d = load_cell(curr_row, q_idx, 1);
                        let d = _mm_add_ps(dcv, old_d);
                        cv = _mm_or_ps(cv, _mm_cmpgt_ps(d, old_d));
                        store_cell(curr_row, q_idx, 1, d);
                        dcv = _mm_mul_ps(dcv, tdd);
                    }
                    if _mm_movemask_ps(cv) == 0 {
                        break;
                    }
                }
            }
        }

        for q_idx in 0..q_count {
            xev = _mm_add_ps(load_cell(curr_row, q_idx, 1), xev);
        }
        #[cfg(feature = "tracehash")]
        if i == 18 || i == 19 || i == 20 {
            let row = std::slice::from_raw_parts(curr_row as *const __m128, q_count * 3);
            trace_forward_engine_row_sums_q1e5(i, dsq, dsq_offset, l, om.m, q_count, row);
        }
        #[cfg(feature = "tracehash")]
        if i == 19 {
            trace_forward_engine_row19_xev_bits(dsq, dsq_offset, l, om.m, xev);
        }
        #[cfg(feature = "tracehash")]
        let trace_xev_before_hsum = xev;
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xev, xev),
        );
        xev = _mm_add_ps(
            xev,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xev, xev),
        );
        _mm_store_ss(&mut xe, xev);
        #[cfg(feature = "tracehash")]
        if i == 19 {
            trace_forward_engine_row19_xe_bits(dsq, dsq_offset, l, om.m, xe);
        }

        xn *= om.xf[P7O_N][P7O_LOOP];
        xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
        xb = xj * om.xf[P7O_J][P7O_MOVE] + xn * om.xf[P7O_N][P7O_MOVE];

        let row_scale = if xe > 1.0e4 {
            let row_scale = xe;
            let inv_xe = 1.0 / xe;
            let scalev = _mm_set1_ps(inv_xe);
            xn /= row_scale;
            xc /= row_scale;
            xj /= row_scale;
            xb /= row_scale;
            #[cfg(feature = "tracehash")]
            {
                trace_scale_event_count += 1;
                trace_forward_engine_scale_event_detail_q1e5(
                    trace_scale_event_count,
                    true,
                    dsq,
                    dsq_offset,
                    l,
                    om.m,
                    i,
                    row_scale,
                    xn,
                    xj,
                    xb,
                    xc,
                );
                if trace_scale_event_count <= 4 {
                    trace_forward_engine_scale_event_q1e5(
                        trace_scale_event_count,
                        true,
                        dsq,
                        dsq_offset,
                        l,
                        om.m,
                        i,
                        row_scale,
                        xn,
                        xj,
                        xb,
                        xc,
                    );
                }
                trace_last_scale_event = Some((i, row_scale, xn, xj, xb, xc));
            }
            let mut off = 0;
            while off < row_width {
                let p = curr_row.add(off);
                _mm_store_ps(p, _mm_mul_ps(_mm_load_ps(p), scalev));
                off += 4;
            }
            totscale = (totscale as f64 + c_log_f64(xe as f64)) as f32;
            xe = 1.0;
            row_scale
        } else {
            1.0
        };

        store_xmx(xmx_ptr, i, xe, xn, xj, xb, xc);
        *scale_ptr.add(i) = totscale as f64;
        *row_scale_ptr.add(i) = row_scale;

        #[cfg(feature = "tracehash")]
        if i <= 8 || i == l {
            trace_forward_engine_step_bits(
                dsq,
                dsq_offset,
                l,
                om.m,
                i,
                trace_xev_before_hsum,
                xe,
                row_scale,
            );
        }
    }

    #[cfg(feature = "tracehash")]
    if let Some((row, pre_xe, last_xn, last_xj, last_xb, last_xc)) = trace_last_scale_event {
        trace_forward_engine_scale_event_q1e5(
            0, true, dsq, dsq_offset, l, om.m, row, pre_xe, last_xn, last_xj, last_xb, last_xc,
        );
    }

    if xc.is_nan() || (l > 0 && xc == 0.0) || xc.is_infinite() {
        f32::NEG_INFINITY
    } else {
        (totscale as f64 + c_log_f64((xc * om.xf[P7O_C][P7O_MOVE]) as f64)) as f32
    }
}

/// Returns true if `dsq[dsq_offset+1..=dsq_offset+l]` contains only canonical
/// residue codes (`< abc_kp`); used to gate entry to the direct full-DP
/// engine, which assumes no degenerate residues.
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

/// Right-shift a __m128 by one float element, zero-filling from the left.
/// [a, b, c, d] -> [0, a, b, c]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn rightshift_float(v: __m128) -> __m128 {
    // Cast to integer, shift left by 4 bytes, cast back
    let vi = _mm_castps_si128(v);
    let shifted = _mm_slli_si128::<4>(vi);
    _mm_castsi128_ps(shifted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::profile::*;
    use std::path::Path;

    /// Smoke test: build OProfile from the 20aa test HMM and run the SSE
    /// Forward parser on a short residue mix, asserting a finite score.
    #[test]
    fn test_forward_parser_basic() {
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
        profile_config(&hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        let dsq = abc.digitize(b"AAAAAAAAAAGGGGGGGGGG");
        let sc = unsafe { forward_parser(&dsq, 20, &om) };
        assert!(sc.is_finite(), "Forward score should be finite, got {}", sc);
    }
}
