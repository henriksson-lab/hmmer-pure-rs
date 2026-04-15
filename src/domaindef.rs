//! Domain definition using posterior probabilities.
//! Port of p7_domaindef.c p7_domaindef_ByPosteriorHeuristics().

use crate::alphabet::{Alphabet, Dsq};
use crate::bg::Bg;
use crate::dp::generic_backward::g_backward;
use crate::dp::generic_decoding::{domain_decoding, g_decoding};
use crate::dp::generic_fwdback::g_forward;
use crate::dp::generic_null2;
use crate::dp::generic_optacc::{g_oa_trace, g_optimal_accuracy_with_deltas, OptAccTDelta};
use crate::dp::generic_stotrace::g_stochastic_trace;
use crate::dp::gmx::*;
use crate::hmm::Hmm;
use crate::profile::*;
use crate::spensemble::{self, ClusterParams, SegmentPair};
use crate::tophits::{AliDisplay, Domain};
use crate::trace::{State, Trace};
use crate::util::random::MersenneTwister;

// Thresholds matching C's p7_domaindef defaults
const RT1: f32 = 0.25; // mocc threshold to trigger a domain region
const RT2: f32 = 0.10; // mocc threshold to end a domain region
const RT3: f32 = 0.20; // threshold for multi-domain region detection
const NSAMPLES: usize = 200; // stochastic tracebacks for clustering

#[derive(Debug, Clone, Copy, Default)]
pub struct DomainDefinitionStats {
    pub nregions: usize,
    pub nclustered: usize,
    pub noverlaps: usize,
    pub nenvelopes: usize,
}

#[cfg(target_arch = "x86_64")]
struct DomainSimdScratch {
    fwd_dp: Vec<std::arch::x86_64::__m128>,
    bck_prev: Vec<std::arch::x86_64::__m128>,
    bck_cur: Vec<std::arch::x86_64::__m128>,
    fwd_pmx: crate::simd::probmx::ProbMx,
    bck_pmx: crate::simd::probmx::ProbMx,
    null2: Vec<f32>,
    exp_m: Vec<f32>,
    exp_i: Vec<f32>,
    pp_gmx: Gmx,
    oa_gmx: Gmx,
}

#[cfg(target_arch = "x86_64")]
impl DomainSimdScratch {
    fn new() -> Self {
        Self {
            fwd_dp: Vec::new(),
            bck_prev: Vec::new(),
            bck_cur: Vec::new(),
            fwd_pmx: crate::simd::probmx::ProbMx::new_full(0, 0),
            bck_pmx: crate::simd::probmx::ProbMx::new_full(0, 0),
            null2: Vec::new(),
            exp_m: Vec::new(),
            exp_i: Vec::new(),
            pp_gmx: Gmx::new(0, 0),
            oa_gmx: Gmx::new(0, 0),
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
struct DomainSimdScratch;

#[cfg(not(target_arch = "x86_64"))]
impl DomainSimdScratch {
    fn new() -> Self {
        Self
    }
}

/// Region detection state machine matching C's p7_domaindef_ByPosteriorHeuristics().
/// Uses btot/etot/mocc arrays to find domain regions.
fn find_domain_regions(btot: &[f32], etot: &[f32], mocc: &[f32], l: usize) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut i: i64 = -1;
    let mut triggered = false;

    for j in 1..=l {
        if !triggered {
            // Looking for the START of a domain region.
            // Reset i when mocc (minus local B contribution) drops below rt2.
            if mocc[j] - (btot[j] - btot[j - 1]) < RT2 {
                i = j as i64;
            } else if i == -1 {
                i = j as i64;
            }
            // Trigger when mocc rises above rt1
            if mocc[j] >= RT1 {
                triggered = true;
            }
        } else if mocc[j] - (etot[j] - etot[j - 1]) < RT2 {
            // Found the END of a domain region: mocc dropped below rt2
            // (after subtracting local E contribution). Region is i..j.
            if i >= 1 {
                regions.push((i as usize, j));
            }
            i = -1;
            triggered = false;
        }
    }

    regions
}

/// Check if a region contains multiple domains.
/// Matches C's is_multidomain_region().
fn is_multidomain_region(btot: &[f32], etot: &[f32], i: usize, j: usize) -> bool {
    let mut max = -1.0_f32;
    for z in i..=j {
        let expected_n = (etot[z] - etot[i - 1]).min(btot[j] - btot[z - 1]);
        max = max.max(expected_n);
    }
    #[cfg(feature = "tracehash")]
    {
        let mut th = tracehash::th_call!("is_multidomain_region");
        th.input_usize(i);
        th.input_usize(j);
        th.input_f32(btot[j]);
        th.input_f32(etot[i - 1]);
        th.output_f32(max);
        th.output_u64((max >= RT3) as u64);
        th.finish();
    }
    max >= RT3
}

#[cfg(feature = "tracehash")]
fn trace_domain_region(l: usize, m: usize, ordinal: usize, i: usize, j: usize, is_multi: bool) {
    let mut th = tracehash::th_call!("domain_region");
    th.input_usize(l);
    th.input_usize(m);
    th.input_usize(ordinal);
    th.output_u64(i as u64);
    th.output_u64(j as u64);
    th.output_u64(is_multi as u64);
    th.finish();
}

#[cfg(feature = "tracehash")]
fn trace_domain_cluster_summary(
    l: usize,
    m: usize,
    i: usize,
    j: usize,
    segment_count: usize,
    envelope_count: usize,
) {
    let mut th = tracehash::th_call!("domain_cluster_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_usize(i);
    th.input_usize(j);
    th.output_u64(segment_count as u64);
    th.output_u64(envelope_count as u64);
    th.finish();
}

#[cfg(feature = "tracehash")]
fn trace_domain_envelope_candidate(
    l: usize,
    m: usize,
    region_i: usize,
    region_j: usize,
    ordinal: usize,
    ienv: usize,
    jenv: usize,
    clustered: bool,
) {
    let mut th = tracehash::th_call!("domain_envelope_candidate");
    th.input_usize(l);
    th.input_usize(m);
    th.input_usize(region_i);
    th.input_usize(region_j);
    th.input_usize(ordinal);
    th.input_bool(clustered);
    th.output_u64(ienv as u64);
    th.output_u64(jenv as u64);
    th.finish();
}

#[cfg(all(target_arch = "x86_64"))]
fn null2_by_trace_optimized(
    om: &crate::simd::oprofile::OProfile,
    tr: &Trace,
    zstart: usize,
    zend: usize,
) -> Vec<f32> {
    let q = crate::simd::oprofile::nqf(om.m);
    let mut exp_m = vec![0.0_f32; q * 4];
    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;
    let mut ld = 0usize;

    for z in zstart..=zend.min(tr.n.saturating_sub(1)) {
        if tr.i[z] == 0 {
            continue;
        }
        ld += 1;
        if tr.k[z] > 0 {
            let k = tr.k[z];
            let qi = (k - 1) % q;
            let lane = (k - 1) / q;
            exp_m[qi * 4 + lane] += 1.0;
        } else {
            match tr.st[z] {
                State::N => exp_n += 1.0,
                State::C => exp_c += 1.0,
                State::J => exp_j += 1.0,
                _ => {}
            }
        }
    }

    if ld == 0 {
        return vec![1.0; om.abc_k];
    }

    let norm = 1.0 / ld as f32;
    for val in &mut exp_m {
        *val *= norm;
    }
    exp_n *= norm;
    exp_c *= norm;
    exp_j *= norm;
    let xfactor = exp_n + exp_c + exp_j;

    let mut null2 = vec![0.0_f32; om.abc_k];
    for x in 0..om.abc_k {
        let mut lanes = [0.0_f32; 4];
        for qi in 0..q {
            let odds = om.rfv[x][qi];
            let base = qi * 4;
            for lane in 0..4 {
                lanes[lane] += exp_m[base + lane] * odds[lane];
            }
        }
        let h01 = lanes[0] + lanes[1];
        let h23 = lanes[2] + lanes[3];
        null2[x] = (h01 + h23) + xfactor;
    }
    null2
}

#[cfg(feature = "tracehash")]
fn trace_region_null2_segment(
    model_len: usize,
    region_i: usize,
    region_j: usize,
    trace_idx: usize,
    domain_idx: usize,
    sqfrom: usize,
    sqto: usize,
    tfrom: usize,
    tto: usize,
    tr: &Trace,
) {
    let mut th = tracehash::th_call!("region_null2_trace_segment");
    th.input_usize(model_len);
    th.input_usize(region_i);
    th.input_usize(region_j);
    th.input_usize(trace_idx);
    th.input_usize(domain_idx);
    th.output_u64(sqfrom as u64);
    th.output_u64(sqto as u64);
    th.output_u64(tfrom as u64);
    th.output_u64(tto as u64);
    for z in tfrom..=tto.min(tr.n.saturating_sub(1)) {
        th.output_u64(tr.st[z] as u8 as u64);
        th.output_u64(tr.k[z] as u64);
        th.output_u64(tr.i[z] as u64);
    }
    th.finish();
}

#[cfg(feature = "tracehash")]
fn trace_domain_decoding_summary(
    dsq: &[Dsq],
    l: usize,
    m: usize,
    btot: &[f32],
    etot: &[f32],
    mocc: &[f32],
) {
    let mut th = tracehash::th_call!("domain_decoding_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    th.output_f32(btot[l]);
    th.output_f32(etot[l]);
    for pos in 1..=l {
        th.output_f32(btot[pos]);
        th.output_f32(etot[pos]);
        th.output_f32(mocc[pos]);
    }
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_region_forward_summary(
    dsq: &[Dsq],
    seq_len: usize,
    model_len: usize,
    ri: usize,
    rj: usize,
    om: &crate::simd::oprofile::OProfile,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let region_len = rj - ri + 1;
    let q = crate::simd::oprofile::nqf(om.m);
    let mut th = tracehash::th_call!("region_forward_summary");
    th.input_usize(seq_len);
    th.input_usize(model_len);
    th.input_usize(ri);
    th.input_usize(rj);
    th.input_bytes(&dsq[ri..=rj]);

    for &row in &[0usize, 1, 2, 4, 8, 16, 32, 64, 128, region_len] {
        if row > region_len {
            continue;
        }
        th.output_u64(row as u64);
        th.output_f32(pmx.xmx(row, crate::simd::probmx::PXE));
        th.output_f32(pmx.xmx(row, crate::simd::probmx::PXN));
        th.output_f32(pmx.xmx(row, crate::simd::probmx::PXJ));
        th.output_f32(pmx.xmx(row, crate::simd::probmx::PXB));
        th.output_f32(pmx.xmx(row, crate::simd::probmx::PXC));
        th.output_f32(pmx.row_scale[row]);

        for valid_only in [true, false] {
            let mut msum = 0.0f32;
            let mut dsum = 0.0f32;
            let mut isum = 0.0f32;
            for qi in 0..q {
                for lane in 0..4 {
                    let k = lane * q + qi + 1;
                    if !valid_only || k <= om.m {
                        msum += pmx.mmx(row, k);
                        dsum += pmx.dmx(row, k);
                        isum += pmx.imx(row, k);
                    }
                }
            }
            th.output_f32(msum);
            th.output_f32(dsum);
            th.output_f32(isum);
        }
    }
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_forward_specials_summary(
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = tracehash::th_call!("simd_forward_specials_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    for pos in 0..=l {
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXE));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXN));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXJ));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXB));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXC));
        th.output_f32(pmx.row_scale[pos]);
    }
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_forward_states_summary(
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = tracehash::th_call!("simd_forward_states_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    for pos in 0..=l {
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXE));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXN));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXJ));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXB));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXC));
    }
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_forward_scales_summary(
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = tracehash::th_call!("simd_forward_scales_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    for pos in 0..=l {
        th.output_f32(pmx.row_scale[pos]);
    }
    th.finish();

    if l == 465 || l == 1130 {
        for pos in 0..=l {
            let mut th = tracehash::th_call!("simd_forward_scale_pos_bits");
            th.input_usize(l);
            th.input_usize(m);
            th.input_bytes(&dsq[1..=l]);
            th.input_usize(pos);
            th.output_u64(pmx.row_scale[pos].to_bits() as u64);
            th.finish();
        }
    }
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_forward_states_q1e5_summary(
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = tracehash::th_call!("simd_forward_states_q1e5_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    for pos in 0..=l {
        th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXE), 1.0e-5);
        th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXN), 1.0e-5);
        th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXJ), 1.0e-5);
        th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXB), 1.0e-5);
        th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXC), 1.0e-5);
    }
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_forward_anchor_q1e5(
    name: &'static str,
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pos: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = match name {
        "row0" => tracehash::th_call!("simd_forward_row0_q1e5"),
        "row1" => tracehash::th_call!("simd_forward_row1_q1e5"),
        "row2" => tracehash::th_call!("simd_forward_row2_q1e5"),
        "row4" => tracehash::th_call!("simd_forward_row4_q1e5"),
        "row8" => tracehash::th_call!("simd_forward_row8_q1e5"),
        "row16" => tracehash::th_call!("simd_forward_row16_q1e5"),
        "row17" => tracehash::th_call!("simd_forward_row17_q1e5"),
        "row18" => tracehash::th_call!("simd_forward_row18_q1e5"),
        "row19" => tracehash::th_call!("simd_forward_row19_q1e5"),
        "row20" => tracehash::th_call!("simd_forward_row20_q1e5"),
        "row24" => tracehash::th_call!("simd_forward_row24_q1e5"),
        "row28" => tracehash::th_call!("simd_forward_row28_q1e5"),
        "row32" => tracehash::th_call!("simd_forward_row32_q1e5"),
        "row64" => tracehash::th_call!("simd_forward_row64_q1e5"),
        "row128" => tracehash::th_call!("simd_forward_row128_q1e5"),
        "first_scale" => tracehash::th_call!("simd_forward_first_scale_row_q1e5"),
        _ => tracehash::th_call!("simd_forward_rowl_q1e5"),
    };
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXE), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXN), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXJ), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXB), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXC), 1.0e-5);
    th.output_f32_quant(pmx.row_scale[pos], 1.0e-5);
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_score_domain_forward_anchor_q1e5(
    name: &'static str,
    seq_len: usize,
    model_len: usize,
    env_len: usize,
    ienv: usize,
    jenv: usize,
    null2_is_done: bool,
    env_dsq: &[Dsq],
    pos: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = match name {
        "row0" => tracehash::th_call!("score_domain_forward_row0_q1e5"),
        "row1" => tracehash::th_call!("score_domain_forward_row1_q1e5"),
        "row2" => tracehash::th_call!("score_domain_forward_row2_q1e5"),
        "row4" => tracehash::th_call!("score_domain_forward_row4_q1e5"),
        "row8" => tracehash::th_call!("score_domain_forward_row8_q1e5"),
        "row9" => tracehash::th_call!("score_domain_forward_row9_q1e5"),
        "row10" => tracehash::th_call!("score_domain_forward_row10_q1e5"),
        "row11" => tracehash::th_call!("score_domain_forward_row11_q1e5"),
        "row12" => tracehash::th_call!("score_domain_forward_row12_q1e5"),
        "row13" => tracehash::th_call!("score_domain_forward_row13_q1e5"),
        "row14" => tracehash::th_call!("score_domain_forward_row14_q1e5"),
        "row15" => tracehash::th_call!("score_domain_forward_row15_q1e5"),
        "row16" => tracehash::th_call!("score_domain_forward_row16_q1e5"),
        "row17" => tracehash::th_call!("score_domain_forward_row17_q1e5"),
        "row32" => tracehash::th_call!("score_domain_forward_row32_q1e5"),
        "row64" => tracehash::th_call!("score_domain_forward_row64_q1e5"),
        _ => tracehash::th_call!("score_domain_forward_rowl_q1e5"),
    };
    th.input_usize(seq_len);
    th.input_usize(model_len);
    th.input_usize(env_len);
    th.input_usize(ienv);
    th.input_usize(jenv);
    th.input_bool(null2_is_done);
    th.input_bytes(env_dsq);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXE), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXN), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXJ), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXB), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXC), 1.0e-5);
    th.output_f32_quant(pmx.row_scale[pos], 1.0e-5);
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_score_domain_forward_anchors_q1e5(
    seq_len: usize,
    model_len: usize,
    env_len: usize,
    ienv: usize,
    jenv: usize,
    null2_is_done: bool,
    env_dsq: &[Dsq],
    om: &crate::simd::oprofile::OProfile,
    pmx: &crate::simd::probmx::ProbMx,
) {
    trace_score_domain_forward_anchor_q1e5(
        "row0",
        seq_len,
        model_len,
        env_len,
        ienv,
        jenv,
        null2_is_done,
        env_dsq,
        0,
        pmx,
    );
    for &(name, pos) in &[
        ("row1", 1usize),
        ("row2", 2),
        ("row4", 4),
        ("row8", 8),
        ("row9", 9),
        ("row10", 10),
        ("row11", 11),
        ("row12", 12),
        ("row13", 13),
        ("row14", 14),
        ("row15", 15),
        ("row16", 16),
        ("row17", 17),
        ("row32", 32),
        ("row64", 64),
    ] {
        if env_len >= pos {
            trace_score_domain_forward_anchor_q1e5(
                name,
                seq_len,
                model_len,
                env_len,
                ienv,
                jenv,
                null2_is_done,
                env_dsq,
                pos,
                pmx,
            );
        }
    }
    trace_score_domain_forward_anchor_q1e5(
        "rowl",
        seq_len,
        model_len,
        env_len,
        ienv,
        jenv,
        null2_is_done,
        env_dsq,
        env_len,
        pmx,
    );

    for &(row, e_name, n_name, j_name, b_name, c_name, scale_name) in &[
        (
            14usize,
            "score_domain_forward_row14_e_q1e5",
            "score_domain_forward_row14_n_q1e5",
            "score_domain_forward_row14_j_q1e5",
            "score_domain_forward_row14_b_q1e5",
            "score_domain_forward_row14_c_q1e5",
            "score_domain_forward_row14_scale_q1e5",
        ),
        (
            15usize,
            "score_domain_forward_row15_e_q1e5",
            "score_domain_forward_row15_n_q1e5",
            "score_domain_forward_row15_j_q1e5",
            "score_domain_forward_row15_b_q1e5",
            "score_domain_forward_row15_c_q1e5",
            "score_domain_forward_row15_scale_q1e5",
        ),
        (
            16usize,
            "score_domain_forward_row16_e_q1e5",
            "score_domain_forward_row16_n_q1e5",
            "score_domain_forward_row16_j_q1e5",
            "score_domain_forward_row16_b_q1e5",
            "score_domain_forward_row16_c_q1e5",
            "score_domain_forward_row16_scale_q1e5",
        ),
        (
            17usize,
            "score_domain_forward_row17_e_q1e5",
            "score_domain_forward_row17_n_q1e5",
            "score_domain_forward_row17_j_q1e5",
            "score_domain_forward_row17_b_q1e5",
            "score_domain_forward_row17_c_q1e5",
            "score_domain_forward_row17_scale_q1e5",
        ),
    ] {
        if env_len < row {
            continue;
        }
        let inputs = |th: &mut tracehash::Call| {
            th.input_usize(seq_len);
            th.input_usize(model_len);
            th.input_usize(env_len);
            th.input_usize(ienv);
            th.input_usize(jenv);
            th.input_bool(null2_is_done);
            th.input_bytes(env_dsq);
        };

        let mut th = match e_name {
            "score_domain_forward_row14_e_q1e5" => {
                tracehash::th_call!("score_domain_forward_row14_e_q1e5")
            }
            "score_domain_forward_row15_e_q1e5" => {
                tracehash::th_call!("score_domain_forward_row15_e_q1e5")
            }
            "score_domain_forward_row16_e_q1e5" => {
                tracehash::th_call!("score_domain_forward_row16_e_q1e5")
            }
            _ => tracehash::th_call!("score_domain_forward_row17_e_q1e5"),
        };
        inputs(&mut th);
        th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXE), 1.0e-5);
        th.finish();

        let mut th = match n_name {
            "score_domain_forward_row14_n_q1e5" => {
                tracehash::th_call!("score_domain_forward_row14_n_q1e5")
            }
            "score_domain_forward_row15_n_q1e5" => {
                tracehash::th_call!("score_domain_forward_row15_n_q1e5")
            }
            "score_domain_forward_row16_n_q1e5" => {
                tracehash::th_call!("score_domain_forward_row16_n_q1e5")
            }
            _ => tracehash::th_call!("score_domain_forward_row17_n_q1e5"),
        };
        inputs(&mut th);
        th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXN), 1.0e-5);
        th.finish();

        let mut th = match j_name {
            "score_domain_forward_row14_j_q1e5" => {
                tracehash::th_call!("score_domain_forward_row14_j_q1e5")
            }
            "score_domain_forward_row15_j_q1e5" => {
                tracehash::th_call!("score_domain_forward_row15_j_q1e5")
            }
            "score_domain_forward_row16_j_q1e5" => {
                tracehash::th_call!("score_domain_forward_row16_j_q1e5")
            }
            _ => tracehash::th_call!("score_domain_forward_row17_j_q1e5"),
        };
        inputs(&mut th);
        th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXJ), 1.0e-5);
        th.finish();

        let mut th = match b_name {
            "score_domain_forward_row14_b_q1e5" => {
                tracehash::th_call!("score_domain_forward_row14_b_q1e5")
            }
            "score_domain_forward_row15_b_q1e5" => {
                tracehash::th_call!("score_domain_forward_row15_b_q1e5")
            }
            "score_domain_forward_row16_b_q1e5" => {
                tracehash::th_call!("score_domain_forward_row16_b_q1e5")
            }
            _ => tracehash::th_call!("score_domain_forward_row17_b_q1e5"),
        };
        inputs(&mut th);
        th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXB), 1.0e-5);
        th.finish();

        let mut th = match c_name {
            "score_domain_forward_row14_c_q1e5" => {
                tracehash::th_call!("score_domain_forward_row14_c_q1e5")
            }
            "score_domain_forward_row15_c_q1e5" => {
                tracehash::th_call!("score_domain_forward_row15_c_q1e5")
            }
            "score_domain_forward_row16_c_q1e5" => {
                tracehash::th_call!("score_domain_forward_row16_c_q1e5")
            }
            _ => tracehash::th_call!("score_domain_forward_row17_c_q1e5"),
        };
        inputs(&mut th);
        th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXC), 1.0e-5);
        th.finish();

        let mut th = match scale_name {
            "score_domain_forward_row14_scale_q1e5" => {
                tracehash::th_call!("score_domain_forward_row14_scale_q1e5")
            }
            "score_domain_forward_row15_scale_q1e5" => {
                tracehash::th_call!("score_domain_forward_row15_scale_q1e5")
            }
            "score_domain_forward_row16_scale_q1e5" => {
                tracehash::th_call!("score_domain_forward_row16_scale_q1e5")
            }
            _ => tracehash::th_call!("score_domain_forward_row17_scale_q1e5"),
        };
        inputs(&mut th);
        th.output_f32_quant(pmx.row_scale[row], 1.0e-5);
        th.finish();

        let msum = pmx.striped_row_state_sum(row, 0);
        let dsum = pmx.striped_row_state_sum(row, 1);
        let mut th = match row {
            14 => tracehash::th_call!("score_domain_forward_row14_msum_q1e5"),
            15 => tracehash::th_call!("score_domain_forward_row15_msum_q1e5"),
            16 => tracehash::th_call!("score_domain_forward_row16_msum_q1e5"),
            _ => tracehash::th_call!("score_domain_forward_row17_msum_q1e5"),
        };
        inputs(&mut th);
        th.output_f32_quant(msum, 1.0e-5);
        th.finish();

        let mut th = match row {
            14 => tracehash::th_call!("score_domain_forward_row14_dsum_q1e5"),
            15 => tracehash::th_call!("score_domain_forward_row15_dsum_q1e5"),
            16 => tracehash::th_call!("score_domain_forward_row16_dsum_q1e5"),
            _ => tracehash::th_call!("score_domain_forward_row17_dsum_q1e5"),
        };
        inputs(&mut th);
        th.output_f32_quant(dsum, 1.0e-5);
        th.finish();

        let msum_all = pmx.striped_row_state_sum_all_lanes(row, 0);
        let dsum_all = pmx.striped_row_state_sum_all_lanes(row, 1);
        let mut th = match row {
            14 => tracehash::th_call!("score_domain_forward_row14_msum_all_q1e5"),
            15 => tracehash::th_call!("score_domain_forward_row15_msum_all_q1e5"),
            16 => tracehash::th_call!("score_domain_forward_row16_msum_all_q1e5"),
            _ => tracehash::th_call!("score_domain_forward_row17_msum_all_q1e5"),
        };
        inputs(&mut th);
        th.output_f32_quant(msum_all, 1.0e-5);
        th.finish();

        let mut th = match row {
            14 => tracehash::th_call!("score_domain_forward_row14_dsum_all_q1e5"),
            15 => tracehash::th_call!("score_domain_forward_row15_dsum_all_q1e5"),
            16 => tracehash::th_call!("score_domain_forward_row16_dsum_all_q1e5"),
            _ => tracehash::th_call!("score_domain_forward_row17_dsum_all_q1e5"),
        };
        inputs(&mut th);
        th.output_f32_quant(dsum_all, 1.0e-5);
        th.finish();

        if row == 17 {
            #[cfg(target_arch = "x86_64")]
            let (xev, h1, h2) = unsafe {
                use std::arch::x86_64::*;

                let mut xev_v = _mm_setzero_ps();
                for q in 0..pmx.q_count() {
                    let mv = pmx.striped_row_state_vector(row, 0, q);
                    xev_v = _mm_add_ps(_mm_loadu_ps(mv.as_ptr()), xev_v);
                }
                for q in 0..pmx.q_count() {
                    let dv = pmx.striped_row_state_vector(row, 1, q);
                    xev_v = _mm_add_ps(_mm_loadu_ps(dv.as_ptr()), xev_v);
                }
                let h1_v = _mm_add_ps(
                    xev_v,
                    _mm_shuffle_ps::<{ crate::simd::shuffle_mask(0, 3, 2, 1) }>(xev_v, xev_v),
                );
                let h2_v = _mm_add_ps(
                    h1_v,
                    _mm_shuffle_ps::<{ crate::simd::shuffle_mask(1, 0, 3, 2) }>(h1_v, h1_v),
                );
                let mut xev = [0.0_f32; 4];
                let mut h1 = [0.0_f32; 4];
                let mut h2 = [0.0_f32; 4];
                _mm_storeu_ps(xev.as_mut_ptr(), xev_v);
                _mm_storeu_ps(h1.as_mut_ptr(), h1_v);
                _mm_storeu_ps(h2.as_mut_ptr(), h2_v);
                (xev, h1, h2)
            };

            let mut th = tracehash::th_call!("score_domain_forward_row17_xev_vec_q1e5");
            inputs(&mut th);
            for lane in xev {
                th.output_f32_quant(lane, 1.0e-5);
            }
            th.finish();

            let mut th = tracehash::th_call!("score_domain_forward_row17_xev_h1_q1e5");
            inputs(&mut th);
            for lane in h1 {
                th.output_f32_quant(lane, 1.0e-5);
            }
            th.finish();

            let mut th = tracehash::th_call!("score_domain_forward_row17_xev_h2_lane0_q1e5");
            inputs(&mut th);
            th.output_f32_quant(h2[0], 1.0e-5);
            th.finish();

            let mut th = tracehash::th_call!("score_domain_forward_row17_xev_h2_lane0_bits");
            inputs(&mut th);
            th.output_u64(h2[0].to_bits() as u64);
            th.finish();

            let mut lane_m = [0.0_f32; 4];
            let mut lane_d = [0.0_f32; 4];
            for q in 0..pmx.q_count() {
                let mv = pmx.striped_row_state_vector(row, 0, q);
                let dv = pmx.striped_row_state_vector(row, 1, q);
                for lane in 0..4 {
                    lane_m[lane] += mv[lane];
                    lane_d[lane] += dv[lane];
                }
            }
            for lane in 0..4 {
                let mut th = match lane {
                    0 => tracehash::th_call!("score_domain_forward_row17_xev_lane0_q1e5"),
                    1 => tracehash::th_call!("score_domain_forward_row17_xev_lane1_q1e5"),
                    2 => tracehash::th_call!("score_domain_forward_row17_xev_lane2_q1e5"),
                    _ => tracehash::th_call!("score_domain_forward_row17_xev_lane3_q1e5"),
                };
                inputs(&mut th);
                th.output_f32_quant(xev[lane], 1.0e-5);
                th.finish();

                let mut th = match lane {
                    0 => tracehash::th_call!("score_domain_forward_row17_m_lane0_sum_q1e5"),
                    1 => tracehash::th_call!("score_domain_forward_row17_m_lane1_sum_q1e5"),
                    2 => tracehash::th_call!("score_domain_forward_row17_m_lane2_sum_q1e5"),
                    _ => tracehash::th_call!("score_domain_forward_row17_m_lane3_sum_q1e5"),
                };
                inputs(&mut th);
                th.output_f32_quant(lane_m[lane], 1.0e-5);
                th.finish();

                let mut th = match lane {
                    0 => tracehash::th_call!("score_domain_forward_row17_d_lane0_sum_q1e5"),
                    1 => tracehash::th_call!("score_domain_forward_row17_d_lane1_sum_q1e5"),
                    2 => tracehash::th_call!("score_domain_forward_row17_d_lane2_sum_q1e5"),
                    _ => tracehash::th_call!("score_domain_forward_row17_d_lane3_sum_q1e5"),
                };
                inputs(&mut th);
                th.output_f32_quant(lane_d[lane], 1.0e-5);
                th.finish();
            }

            for &(label, q_start, q_end) in &[
                ("q0_8", 0usize, 8usize),
                ("q8_16", 8, 16),
                ("q16_32", 16, 32),
                ("q32_64", 32, 64),
                ("q64_end", 64, pmx.q_count()),
            ] {
                let mut sum = 0.0_f32;
                for q in q_start..q_end.min(pmx.q_count()) {
                    sum += pmx.striped_row_state_vector(row, 0, q)[0];
                }
                let mut th = match label {
                    "q0_8" => {
                        tracehash::th_call!("score_domain_forward_row17_m_lane0_q0_8_sum_q1e5")
                    }
                    "q8_16" => {
                        tracehash::th_call!("score_domain_forward_row17_m_lane0_q8_16_sum_q1e5")
                    }
                    "q16_32" => {
                        tracehash::th_call!("score_domain_forward_row17_m_lane0_q16_32_sum_q1e5")
                    }
                    "q32_64" => {
                        tracehash::th_call!("score_domain_forward_row17_m_lane0_q32_64_sum_q1e5")
                    }
                    _ => {
                        tracehash::th_call!("score_domain_forward_row17_m_lane0_q64_end_sum_q1e5")
                    }
                };
                inputs(&mut th);
                th.output_f32_quant(sum, 1.0e-5);
                th.finish();
            }

            for q in 0..pmx.q_count() {
                let mut th = tracehash::th_call!("score_domain_forward_row17_m_lane0_q_bits");
                inputs(&mut th);
                th.input_usize(q);
                th.output_u64(pmx.striped_row_state_vector(row, 0, q)[0].to_bits() as u64);
                th.finish();
            }
        }

        if row == 15 && pmx.q_count() > 14 {
            use crate::simd::oprofile::{P7O_BM, P7O_DM, P7O_IM, P7O_MM};
            let q = 14usize;
            let lane = 0usize;
            let k = q + 1 + lane * pmx.q_count();
            let xi = env_dsq[row - 1] as usize;
            let prev_m = pmx.striped_row_state_vector(row - 1, 0, q - 1)[lane];
            let prev_d = pmx.striped_row_state_vector(row - 1, 1, q - 1)[lane];
            let prev_i = pmx.striped_row_state_vector(row - 1, 2, q - 1)[lane];
            let xb = pmx.xmx(row - 1, crate::simd::probmx::PXB);
            let tbm = om.tfv[q * 7 + P7O_BM][lane];
            let tmm = om.tfv[q * 7 + P7O_MM][lane];
            let tim = om.tfv[q * 7 + P7O_IM][lane];
            let tdm = om.tfv[q * 7 + P7O_DM][lane];
            let rsc = om.rfv[xi][q][lane];
            let entry = xb * tbm;
            let m_term = prev_m * tmm;
            let i_term = prev_i * tim;
            let d_term = prev_d * tdm;
            let pre = entry + m_term + i_term + d_term;
            let val = pre * rsc;

            let recur_inputs = |th: &mut tracehash::Call| {
                inputs(th);
                th.input_usize(q);
                th.input_usize(lane);
                th.input_usize(k);
                th.input_usize(xi);
            };

            for &(name, value) in &[
                ("entry", entry),
                ("prev_m", prev_m),
                ("tmm", tmm),
                ("m_term", m_term),
                ("prev_i", prev_i),
                ("tim", tim),
                ("i_term", i_term),
                ("prev_d", prev_d),
                ("tdm", tdm),
                ("d_term", d_term),
                ("rsc", rsc),
                ("pre", pre),
                ("val", val),
            ] {
                let mut th = match name {
                    "entry" => {
                        tracehash::th_call!("score_domain_forward_row15_m_k15_entry_q1e5")
                    }
                    "prev_m" => {
                        tracehash::th_call!("score_domain_forward_row15_m_k15_prev_m_q1e5")
                    }
                    "tmm" => tracehash::th_call!("score_domain_forward_row15_m_k15_tmm_q1e5"),
                    "m_term" => {
                        tracehash::th_call!("score_domain_forward_row15_m_k15_m_term_q1e5")
                    }
                    "prev_i" => {
                        tracehash::th_call!("score_domain_forward_row15_m_k15_prev_i_q1e5")
                    }
                    "tim" => tracehash::th_call!("score_domain_forward_row15_m_k15_tim_q1e5"),
                    "i_term" => {
                        tracehash::th_call!("score_domain_forward_row15_m_k15_i_term_q1e5")
                    }
                    "prev_d" => {
                        tracehash::th_call!("score_domain_forward_row15_m_k15_prev_d_q1e5")
                    }
                    "tdm" => tracehash::th_call!("score_domain_forward_row15_m_k15_tdm_q1e5"),
                    "d_term" => {
                        tracehash::th_call!("score_domain_forward_row15_m_k15_d_term_q1e5")
                    }
                    "rsc" => tracehash::th_call!("score_domain_forward_row15_m_k15_rsc_q1e5"),
                    "pre" => tracehash::th_call!("score_domain_forward_row15_m_k15_pre_q1e5"),
                    _ => tracehash::th_call!("score_domain_forward_row15_m_k15_val_q1e5"),
                };
                recur_inputs(&mut th);
                th.output_f32_quant(value, 1.0e-5);
                th.finish();
            }

            macro_rules! emit_recur_bits {
                ($trace_name:literal, $value:expr) => {{
                    let mut th = tracehash::th_call!($trace_name);
                    recur_inputs(&mut th);
                    th.output_u64(($value).to_bits() as u64);
                    th.finish();
                }};
            }
            emit_recur_bits!("score_domain_forward_row15_m_k15_entry_bits", entry);
            emit_recur_bits!("score_domain_forward_row15_m_k15_prev_m_bits", prev_m);
            emit_recur_bits!("score_domain_forward_row15_m_k15_tmm_bits", tmm);
            emit_recur_bits!("score_domain_forward_row15_m_k15_m_term_bits", m_term);
            emit_recur_bits!("score_domain_forward_row15_m_k15_prev_i_bits", prev_i);
            emit_recur_bits!("score_domain_forward_row15_m_k15_tim_bits", tim);
            emit_recur_bits!("score_domain_forward_row15_m_k15_i_term_bits", i_term);
            emit_recur_bits!("score_domain_forward_row15_m_k15_prev_d_bits", prev_d);
            emit_recur_bits!("score_domain_forward_row15_m_k15_tdm_bits", tdm);
            emit_recur_bits!("score_domain_forward_row15_m_k15_d_term_bits", d_term);
            emit_recur_bits!("score_domain_forward_row15_m_k15_rsc_bits", rsc);
            emit_recur_bits!("score_domain_forward_row15_m_k15_pre_bits", pre);
            emit_recur_bits!("score_domain_forward_row15_m_k15_val_bits", val);
        }

        if row == 15 {
            for &(name, state, qi) in &[
                ("m_q0", 0usize, 0usize),
                ("m_q1", 0, 1),
                ("m_q2", 0, 2),
                ("m_q4", 0, 4),
                ("m_q8", 0, 8),
                ("m_q9", 0, 9),
                ("m_q10", 0, 10),
                ("m_q11", 0, 11),
                ("m_q12", 0, 12),
                ("m_q13", 0, 13),
                ("m_q14", 0, 14),
                ("m_q15", 0, 15),
                ("m_q16", 0, 16),
                ("m_q32", 0, 32),
                ("m_q64", 0, 64),
                ("d_q0", 1, 0),
                ("d_q1", 1, 1),
                ("d_q2", 1, 2),
                ("d_q4", 1, 4),
                ("d_q8", 1, 8),
                ("d_q16", 1, 16),
                ("d_q32", 1, 32),
                ("d_q64", 1, 64),
            ] {
                if qi >= pmx.q_count() {
                    continue;
                }
                let mut th = match name {
                    "m_q0" => tracehash::th_call!("score_domain_forward_row15_m_q0_q1e5"),
                    "m_q1" => tracehash::th_call!("score_domain_forward_row15_m_q1_q1e5"),
                    "m_q2" => tracehash::th_call!("score_domain_forward_row15_m_q2_q1e5"),
                    "m_q4" => tracehash::th_call!("score_domain_forward_row15_m_q4_q1e5"),
                    "m_q8" => tracehash::th_call!("score_domain_forward_row15_m_q8_q1e5"),
                    "m_q9" => tracehash::th_call!("score_domain_forward_row15_m_q9_q1e5"),
                    "m_q10" => tracehash::th_call!("score_domain_forward_row15_m_q10_q1e5"),
                    "m_q11" => tracehash::th_call!("score_domain_forward_row15_m_q11_q1e5"),
                    "m_q12" => tracehash::th_call!("score_domain_forward_row15_m_q12_q1e5"),
                    "m_q13" => tracehash::th_call!("score_domain_forward_row15_m_q13_q1e5"),
                    "m_q14" => tracehash::th_call!("score_domain_forward_row15_m_q14_q1e5"),
                    "m_q15" => tracehash::th_call!("score_domain_forward_row15_m_q15_q1e5"),
                    "m_q16" => tracehash::th_call!("score_domain_forward_row15_m_q16_q1e5"),
                    "m_q32" => tracehash::th_call!("score_domain_forward_row15_m_q32_q1e5"),
                    "m_q64" => tracehash::th_call!("score_domain_forward_row15_m_q64_q1e5"),
                    "d_q0" => tracehash::th_call!("score_domain_forward_row15_d_q0_q1e5"),
                    "d_q1" => tracehash::th_call!("score_domain_forward_row15_d_q1_q1e5"),
                    "d_q2" => tracehash::th_call!("score_domain_forward_row15_d_q2_q1e5"),
                    "d_q4" => tracehash::th_call!("score_domain_forward_row15_d_q4_q1e5"),
                    "d_q8" => tracehash::th_call!("score_domain_forward_row15_d_q8_q1e5"),
                    "d_q16" => tracehash::th_call!("score_domain_forward_row15_d_q16_q1e5"),
                    "d_q32" => tracehash::th_call!("score_domain_forward_row15_d_q32_q1e5"),
                    _ => tracehash::th_call!("score_domain_forward_row15_d_q64_q1e5"),
                };
                inputs(&mut th);
                let lanes = pmx.striped_row_state_vector(row, state, qi);
                for lane in lanes {
                    th.output_f32_quant(lane, 1.0e-5);
                }
                th.finish();
            }

            if 14 < pmx.q_count() {
                let lanes = pmx.striped_row_state_vector(row, 0, 14);
                for &(lane_name, lane_idx) in
                    &[("lane0", 0usize), ("lane1", 1), ("lane2", 2), ("lane3", 3)]
                {
                    let mut th = match lane_name {
                        "lane0" => {
                            tracehash::th_call!("score_domain_forward_row15_m_q14_lane0_q1e5")
                        }
                        "lane1" => {
                            tracehash::th_call!("score_domain_forward_row15_m_q14_lane1_q1e5")
                        }
                        "lane2" => {
                            tracehash::th_call!("score_domain_forward_row15_m_q14_lane2_q1e5")
                        }
                        _ => tracehash::th_call!("score_domain_forward_row15_m_q14_lane3_q1e5"),
                    };
                    inputs(&mut th);
                    th.output_u64((14 + 1 + lane_idx * pmx.q_count()) as u64);
                    th.output_f32_quant(lanes[lane_idx], 1.0e-5);
                    th.finish();
                }
            }

            for &(name, state, q_start, q_end) in &[
                ("m_q0_8", 0usize, 0usize, 8usize),
                ("m_q8_16", 0, 8, 16),
                ("m_q16_32", 0, 16, 32),
                ("m_q32_64", 0, 32, 64),
                ("m_q64_end", 0, 64, usize::MAX),
                ("d_q0_8", 1, 0, 8),
                ("d_q8_16", 1, 8, 16),
                ("d_q16_32", 1, 16, 32),
                ("d_q32_64", 1, 32, 64),
                ("d_q64_end", 1, 64, usize::MAX),
            ] {
                let mut th = match name {
                    "m_q0_8" => {
                        tracehash::th_call!("score_domain_forward_row15_m_q0_8_sum_q1e5")
                    }
                    "m_q8_16" => {
                        tracehash::th_call!("score_domain_forward_row15_m_q8_16_sum_q1e5")
                    }
                    "m_q16_32" => {
                        tracehash::th_call!("score_domain_forward_row15_m_q16_32_sum_q1e5")
                    }
                    "m_q32_64" => {
                        tracehash::th_call!("score_domain_forward_row15_m_q32_64_sum_q1e5")
                    }
                    "m_q64_end" => {
                        tracehash::th_call!("score_domain_forward_row15_m_q64_end_sum_q1e5")
                    }
                    "d_q0_8" => {
                        tracehash::th_call!("score_domain_forward_row15_d_q0_8_sum_q1e5")
                    }
                    "d_q8_16" => {
                        tracehash::th_call!("score_domain_forward_row15_d_q8_16_sum_q1e5")
                    }
                    "d_q16_32" => {
                        tracehash::th_call!("score_domain_forward_row15_d_q16_32_sum_q1e5")
                    }
                    "d_q32_64" => {
                        tracehash::th_call!("score_domain_forward_row15_d_q32_64_sum_q1e5")
                    }
                    _ => tracehash::th_call!("score_domain_forward_row15_d_q64_end_sum_q1e5"),
                };
                inputs(&mut th);
                th.output_f32_quant(
                    pmx.striped_row_state_q_range_sum(row, state, q_start, q_end),
                    1.0e-5,
                );
                th.finish();
            }
        }
    }
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_forward_row_parts_q1e5(
    row: usize,
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let inputs = |th: &mut tracehash::Call| {
        th.input_usize(l);
        th.input_usize(m);
        th.input_bytes(&dsq[1..=l]);
    };

    let mut th = match row {
        17 => tracehash::th_call!("simd_forward_row17_e_q1e5"),
        _ => tracehash::th_call!("simd_forward_row19_e_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXE), 1.0e-5);
    th.finish();

    let mut th = match row {
        17 => tracehash::th_call!("simd_forward_row17_n_q1e5"),
        _ => tracehash::th_call!("simd_forward_row19_n_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXN), 1.0e-5);
    th.finish();

    let mut th = match row {
        17 => tracehash::th_call!("simd_forward_row17_j_q1e5"),
        _ => tracehash::th_call!("simd_forward_row19_j_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXJ), 1.0e-5);
    th.finish();

    let mut th = match row {
        17 => tracehash::th_call!("simd_forward_row17_b_q1e5"),
        _ => tracehash::th_call!("simd_forward_row19_b_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXB), 1.0e-5);
    th.finish();

    let mut th = match row {
        17 => tracehash::th_call!("simd_forward_row17_c_q1e5"),
        _ => tracehash::th_call!("simd_forward_row19_c_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXC), 1.0e-5);
    th.finish();

    let mut th = match row {
        17 => tracehash::th_call!("simd_forward_row17_scale_q1e5"),
        _ => tracehash::th_call!("simd_forward_row19_scale_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.row_scale[row], 1.0e-5);
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn first_forward_scale_row(l: usize, pmx: &crate::simd::probmx::ProbMx) -> Option<usize> {
    (0..=l).find(|&pos| pmx.row_scale[pos] > 1.0)
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_forward_scale_events_summary(
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut count = 0usize;
    let mut first = 0usize;
    let mut last = 0usize;
    let mut sum = 0.0_f32;
    for pos in 0..=l {
        if pmx.row_scale[pos] > 1.0 {
            if count == 0 {
                first = pos;
            }
            last = pos;
            count += 1;
            sum += pmx.row_scale[pos].ln();
        }
    }
    let mut th = tracehash::th_call!("simd_forward_scale_events_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    th.output_u64(count as u64);
    th.output_u64(first as u64);
    th.output_u64(last as u64);
    th.output_f32_quant(sum, 1.0e-5);
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_backward_specials_summary(
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = tracehash::th_call!("simd_backward_specials_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    for pos in 0..=l {
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXE));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXN));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXJ));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXB));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXC));
        th.output_f32(pmx.row_scale[pos]);
    }
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_backward_states_summary(
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = tracehash::th_call!("simd_backward_states_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    for pos in 0..=l {
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXE));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXN));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXJ));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXB));
        th.output_f32(pmx.xmx(pos, crate::simd::probmx::PXC));
    }
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_backward_scales_summary(
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = tracehash::th_call!("simd_backward_scales_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    for pos in 0..=l {
        th.output_f32(pmx.row_scale[pos]);
    }
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_backward_states_q1e5_summary(
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = tracehash::th_call!("simd_backward_states_q1e5_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    for pos in 0..=l {
        th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXE), 1.0e-5);
        th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXN), 1.0e-5);
        th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXJ), 1.0e-5);
        th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXB), 1.0e-5);
        th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXC), 1.0e-5);
    }
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_backward_anchor_q1e5(
    name: &'static str,
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pos: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let mut th = match name {
        "row0" => tracehash::th_call!("simd_backward_row0_q1e5"),
        "row1" => tracehash::th_call!("simd_backward_row1_q1e5"),
        "row2" => tracehash::th_call!("simd_backward_row2_q1e5"),
        "row4" => tracehash::th_call!("simd_backward_row4_q1e5"),
        "row8" => tracehash::th_call!("simd_backward_row8_q1e5"),
        "row16" => tracehash::th_call!("simd_backward_row16_q1e5"),
        "row32" => tracehash::th_call!("simd_backward_row32_q1e5"),
        "row64" => tracehash::th_call!("simd_backward_row64_q1e5"),
        "row128" => tracehash::th_call!("simd_backward_row128_q1e5"),
        "first_scale" => tracehash::th_call!("simd_backward_first_scale_row_q1e5"),
        _ => tracehash::th_call!("simd_backward_rowl_q1e5"),
    };
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXE), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXN), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXJ), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXB), 1.0e-5);
    th.output_f32_quant(pmx.xmx(pos, crate::simd::probmx::PXC), 1.0e-5);
    th.output_f32_quant(pmx.row_scale[pos], 1.0e-5);
    th.finish();
}

#[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
fn trace_pmx_backward_row_parts_q1e5(
    row: usize,
    dsq: &[Dsq],
    l: usize,
    m: usize,
    pmx: &crate::simd::probmx::ProbMx,
) {
    let inputs = |th: &mut tracehash::Call| {
        th.input_usize(l);
        th.input_usize(m);
        th.input_bytes(&dsq[1..=l]);
    };

    let mut th = match row {
        0 => tracehash::th_call!("simd_backward_row0_e_q1e5"),
        _ => tracehash::th_call!("simd_backward_row1_e_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXE), 1.0e-5);
    th.finish();

    let mut th = match row {
        0 => tracehash::th_call!("simd_backward_row0_n_q1e5"),
        _ => tracehash::th_call!("simd_backward_row1_n_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXN), 1.0e-5);
    th.finish();

    let mut th = match row {
        0 => tracehash::th_call!("simd_backward_row0_j_q1e5"),
        _ => tracehash::th_call!("simd_backward_row1_j_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXJ), 1.0e-5);
    th.finish();

    let mut th = match row {
        0 => tracehash::th_call!("simd_backward_row0_b_q1e5"),
        _ => tracehash::th_call!("simd_backward_row1_b_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXB), 1.0e-5);
    th.finish();

    let mut th = match row {
        0 => tracehash::th_call!("simd_backward_row0_c_q1e5"),
        _ => tracehash::th_call!("simd_backward_row1_c_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.xmx(row, crate::simd::probmx::PXC), 1.0e-5);
    th.finish();

    let mut th = match row {
        0 => tracehash::th_call!("simd_backward_row0_scale_q1e5"),
        _ => tracehash::th_call!("simd_backward_row1_scale_q1e5"),
    };
    inputs(&mut th);
    th.output_f32_quant(pmx.row_scale[row], 1.0e-5);
    th.finish();
}

fn add_null2_correction(
    null2_arr: &[f32],
    dsq: &[Dsq],
    ienv: usize,
    jenv: usize,
    mut n2sc: Option<&mut [f32]>,
    dom_correction: &mut f32,
) {
    let mut log_null2 = [0.0_f32; 256];
    for (x, val) in null2_arr.iter().enumerate() {
        log_null2[x] = val.ln();
    }
    for pos in ienv..=jenv {
        let x = dsq[pos] as usize;
        let sc = if x < null2_arr.len() {
            log_null2[x]
        } else {
            0.0
        };
        if let Some(ref mut n2sc) = n2sc {
            n2sc[pos] = sc;
        }
        *dom_correction += sc;
    }
}

/// Score a domain envelope: Forward for score, Viterbi for traceback,
/// posterior decoding for PP annotation and null2 bias.
/// `seq_len` is the full sequence length (for null model and length correction).
/// `bg_f` is the background frequencies for the alphabet.
fn score_domain_envelope(
    dsq: &[Dsq],
    seq_len: usize,
    gm: &Profile,
    env_gm: &Profile,
    env_om: Option<&crate::simd::oprofile::OProfile>,
    hmm: &Hmm,
    ienv: usize,
    jenv: usize,
    null_sc: f32,
    _bg_f: &[f32],
    mut n2sc: Option<&mut [f32]>,
    null2_is_done: bool,
    match_odds: Option<&[f32]>,
    optacc_deltas: &OptAccTDelta,
    mut simd_scratch: Option<&mut DomainSimdScratch>,
    make_alignment: bool,
    make_alignment_display: bool,
) -> Domain {
    debug_assert!(ienv >= 1 && jenv <= seq_len);
    debug_assert!(null_sc.is_finite());
    let env_len = jenv - ienv + 1;
    let l = seq_len; // full sequence length for length correction

    // Forward score for the envelope.
    let env_fwd_sc;
    let mut simd_null2_arr: Option<Vec<f32>> = None;
    let mut simd_null2_scratch: Option<&[f32]> = None;
    let mut env_pp_storage: Option<Gmx>;
    let mut env_pp_ptr: *mut Gmx = std::ptr::null_mut();
    if make_alignment {
        if let Some(ref mut scratch) = simd_scratch {
            scratch.pp_gmx.grow_to_zeroed(gm.m, env_len);
            env_pp_ptr = &mut scratch.pp_gmx;
        } else {
            env_pp_storage = Some(Gmx::new(gm.m, env_len));
            env_pp_ptr = env_pp_storage.as_mut().unwrap();
        }
    }

    #[cfg(target_arch = "x86_64")]
    let use_simd = is_x86_feature_detected!("sse2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_simd = false;
    let dsq_offset = ienv - 1;
    let mut sub_dsq = Vec::new();
    if !use_simd || make_alignment_display {
        sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
        sub_dsq.extend_from_slice(&dsq[ienv..=jenv]);
        sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
    }

    if use_simd {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::simd::oprofile::OProfile;
            let env_om_storage;
            let env_om = if let Some(om) = env_om {
                om
            } else {
                env_om_storage = OProfile::convert(env_gm);
                &env_om_storage
            };
            if make_alignment || !null2_is_done {
                use crate::simd::oprofile::{P7O_C, P7O_J, P7O_LOOP, P7O_N};
                use crate::simd::probmx::{
                    match_odds_from_rsc, p_decoding_to_gmx,
                    p_null2_odds_from_omx_expectation_reuse, p_null2_odds_from_pmx,
                    p_null2_odds_from_pmx_reuse, ProbMx,
                };

                if let Some(ref mut scratch) = simd_scratch {
                    scratch.fwd_pmx.resize_full(gm.m, env_len);
                    env_fwd_sc = unsafe {
                        crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                            dsq,
                            dsq_offset,
                            env_len,
                            &env_om,
                            &mut scratch.fwd_pmx,
                            &mut scratch.fwd_dp,
                        )
                    };
                    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
                    trace_score_domain_forward_anchors_q1e5(
                        seq_len,
                        gm.m,
                        env_len,
                        ienv,
                        jenv,
                        null2_is_done,
                        &dsq[ienv..=jenv],
                        env_om,
                        &scratch.fwd_pmx,
                    );
                    scratch.bck_pmx.resize_full(gm.m, env_len);
                    unsafe {
                        crate::simd::bck_filter::backward_parser_pmx_offset_with_scratch(
                            dsq,
                            dsq_offset,
                            env_len,
                            &env_om,
                            env_fwd_sc,
                            &mut scratch.bck_pmx,
                            Some(&scratch.fwd_pmx.row_scale),
                            &mut scratch.bck_prev,
                            &mut scratch.bck_cur,
                        );
                    };
                    let njc_loop = [
                        env_om.xf[P7O_N][P7O_LOOP],
                        env_om.xf[P7O_J][P7O_LOOP],
                        env_om.xf[P7O_C][P7O_LOOP],
                    ];
                    if make_alignment {
                        p_decoding_to_gmx(
                            &scratch.fwd_pmx,
                            &scratch.bck_pmx,
                            gm.m,
                            njc_loop,
                            unsafe { &mut *env_pp_ptr },
                        );
                    }
                    if !null2_is_done {
                        if make_alignment {
                            p_null2_odds_from_omx_expectation_reuse(
                                &scratch.fwd_pmx,
                                &scratch.bck_pmx,
                                gm.abc_k,
                                &env_om.rfv,
                                njc_loop,
                                &mut scratch.null2,
                                &mut scratch.exp_m,
                                &mut scratch.exp_i,
                            );
                            simd_null2_scratch = Some(&scratch.null2);
                        } else {
                            let local_match_odds;
                            let match_odds = if let Some(match_odds) = match_odds {
                                match_odds
                            } else {
                                local_match_odds = match_odds_from_rsc(&gm.rsc, gm.abc_k, gm.m);
                                &local_match_odds
                            };
                            p_null2_odds_from_pmx_reuse(
                                &scratch.fwd_pmx,
                                &scratch.bck_pmx,
                                gm.m,
                                gm.abc_k,
                                match_odds,
                                njc_loop,
                                &mut scratch.null2,
                                &mut scratch.exp_m,
                                &mut scratch.exp_i,
                            );
                            simd_null2_scratch = Some(&scratch.null2);
                        }
                    }
                } else {
                    let mut fwd_pmx = ProbMx::new_full(gm.m, env_len);
                    env_fwd_sc = unsafe {
                        crate::simd::fwd_filter::forward_parser_pmx_offset(
                            dsq,
                            dsq_offset,
                            env_len,
                            &env_om,
                            &mut fwd_pmx,
                        )
                    };
                    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
                    trace_score_domain_forward_anchors_q1e5(
                        seq_len,
                        gm.m,
                        env_len,
                        ienv,
                        jenv,
                        null2_is_done,
                        &dsq[ienv..=jenv],
                        env_om,
                        &fwd_pmx,
                    );
                    let mut bck_pmx = ProbMx::new_full(gm.m, env_len);
                    unsafe {
                        crate::simd::bck_filter::backward_parser_pmx_offset_with_fwd_scales(
                            dsq,
                            dsq_offset,
                            env_len,
                            &env_om,
                            env_fwd_sc,
                            &mut bck_pmx,
                            &fwd_pmx.row_scale,
                        );
                    }
                    let njc_loop = [
                        env_om.xf[P7O_N][P7O_LOOP],
                        env_om.xf[P7O_J][P7O_LOOP],
                        env_om.xf[P7O_C][P7O_LOOP],
                    ];
                    if make_alignment {
                        p_decoding_to_gmx(&fwd_pmx, &bck_pmx, gm.m, njc_loop, unsafe {
                            &mut *env_pp_ptr
                        });
                    }
                    if !null2_is_done {
                        let local_match_odds;
                        let match_odds = if let Some(match_odds) = match_odds {
                            match_odds
                        } else {
                            local_match_odds = match_odds_from_rsc(&gm.rsc, gm.abc_k, gm.m);
                            &local_match_odds
                        };
                        simd_null2_arr = Some(if make_alignment {
                            let mut null2 = Vec::new();
                            let mut exp_m = Vec::new();
                            let mut exp_i = Vec::new();
                            p_null2_odds_from_omx_expectation_reuse(
                                &fwd_pmx,
                                &bck_pmx,
                                gm.abc_k,
                                &env_om.rfv,
                                njc_loop,
                                &mut null2,
                                &mut exp_m,
                                &mut exp_i,
                            );
                            null2
                        } else {
                            p_null2_odds_from_pmx(
                                &fwd_pmx, &bck_pmx, gm.m, gm.abc_k, match_odds, njc_loop,
                            )
                        });
                    }
                }
            } else {
                env_fwd_sc = unsafe {
                    crate::simd::fwd_filter::forward_parser_offset(
                        dsq, dsq_offset, env_len, &env_om,
                    )
                };
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        unreachable!();
    } else {
        let mut gx = Gmx::new(gm.m, env_len);
        env_fwd_sc = g_forward(&sub_dsq, env_len, env_gm, &mut gx);
    }

    // Posterior decoding for PP annotation and null2. C HMMER fills n2sc[]
    // from these per-envelope posteriors for simple regions.
    if env_pp_ptr.is_null()
        && simd_null2_arr.is_none()
        && simd_null2_scratch.is_none()
        && !null2_is_done
    {
        env_pp_storage = Some(Gmx::new(gm.m, env_len));
        env_pp_ptr = env_pp_storage.as_mut().unwrap();
    }
    if !env_pp_ptr.is_null()
        && unsafe { (*env_pp_ptr).l == 0 }
        && (make_alignment
            || simd_null2_arr.is_none() && simd_null2_scratch.is_none() && !null2_is_done)
    {
        let mut gx_fwd = Gmx::new(gm.m, env_len);
        g_forward(&sub_dsq, env_len, env_gm, &mut gx_fwd);
        let mut gx_bck = Gmx::new(gm.m, env_len);
        g_backward(&sub_dsq, env_len, env_gm, &mut gx_bck);
        g_decoding(env_gm, &gx_fwd, &gx_bck, unsafe { &mut *env_pp_ptr });
    }
    let mut dom_correction = 0.0_f32;
    if null2_is_done {
        if let Some(ref n2sc) = n2sc {
            for pos in ienv..=jenv {
                dom_correction += n2sc[pos];
            }
        }
    } else {
        if let Some(null2_arr) = simd_null2_scratch {
            add_null2_correction(
                null2_arr,
                dsq,
                ienv,
                jenv,
                n2sc.as_deref_mut(),
                &mut dom_correction,
            );
        } else {
            let null2_arr = simd_null2_arr.unwrap_or_else(|| {
                generic_null2::null2_by_expectation(env_gm, unsafe { &*env_pp_ptr }, 1, env_len)
            });
            add_null2_correction(
                &null2_arr,
                dsq,
                ienv,
                jenv,
                n2sc.as_deref_mut(),
                &mut dom_correction,
            );
        }
    }

    let mut oasc = 0.0_f32;
    let ad = if make_alignment {
        let env_pp = unsafe { &*env_pp_ptr };
        if let Some(ref mut scratch) = simd_scratch {
            scratch.oa_gmx.grow_to_zeroed(gm.m, env_len);
            oasc =
                g_optimal_accuracy_with_deltas(env_gm, env_pp, &mut scratch.oa_gmx, optacc_deltas);
            let tr = g_oa_trace(env_gm, env_pp, &scratch.oa_gmx);
            if make_alignment_display {
                let abc = Alphabet::new(hmm.abc_type);
                crate::trace::alignment_display_with_pp(&tr, &sub_dsq, hmm, &abc, Some(env_pp)).map(
                    |mut ad| {
                        ad.sqfrom += ienv - 1;
                        ad.sqto += ienv - 1;
                        AliDisplay {
                            model: ad.model,
                            mline: ad.mline,
                            aseq: ad.aseq,
                            hmmfrom: ad.hmmfrom,
                            hmmto: ad.hmmto,
                            sqfrom: ad.sqfrom,
                            sqto: ad.sqto,
                            ppline: ad.ppline,
                        }
                    },
                )
            } else {
                crate::trace::alignment_coords(&tr).map(|(hmmfrom, hmmto, mut sqfrom, mut sqto)| {
                    if sqfrom > 0 {
                        sqfrom += ienv - 1;
                    }
                    if sqto > 0 {
                        sqto += ienv - 1;
                    }
                    AliDisplay {
                        model: String::new(),
                        mline: String::new(),
                        aseq: String::new(),
                        hmmfrom,
                        hmmto,
                        sqfrom,
                        sqto,
                        ppline: String::new(),
                    }
                })
            }
        } else {
            let mut gx_oa = Gmx::new(gm.m, env_len);
            oasc = g_optimal_accuracy_with_deltas(env_gm, env_pp, &mut gx_oa, optacc_deltas);
            let tr = g_oa_trace(env_gm, env_pp, &gx_oa);
            if make_alignment_display {
                let abc = Alphabet::new(hmm.abc_type);
                crate::trace::alignment_display_with_pp(&tr, &sub_dsq, hmm, &abc, Some(env_pp)).map(
                    |mut ad| {
                        ad.sqfrom += ienv - 1;
                        ad.sqto += ienv - 1;
                        AliDisplay {
                            model: ad.model,
                            mline: ad.mline,
                            aseq: ad.aseq,
                            hmmfrom: ad.hmmfrom,
                            hmmto: ad.hmmto,
                            sqfrom: ad.sqfrom,
                            sqto: ad.sqto,
                            ppline: ad.ppline,
                        }
                    },
                )
            } else {
                crate::trace::alignment_coords(&tr).map(|(hmmfrom, hmmto, mut sqfrom, mut sqto)| {
                    if sqfrom > 0 {
                        sqfrom += ienv - 1;
                    }
                    if sqto > 0 {
                        sqto += ienv - 1;
                    }
                    AliDisplay {
                        model: String::new(),
                        mline: String::new(),
                        aseq: String::new(),
                        hmmfrom,
                        hmmto,
                        sqfrom,
                        sqto,
                        ppline: String::new(),
                    }
                })
            }
        }
    } else {
        None
    };

    let (iali, jali) = if let Some(ref a) = ad {
        (a.sqfrom as i64, a.sqto as i64)
    } else {
        (ienv as i64, jenv as i64)
    };

    // Omega-weighted null2 bias.
    let omega = 1.0_f32 / 256.0;
    let dom_bias = crate::logsum::p7_flogsum(0.0, omega.ln() + dom_correction);

    // Domain bitscore matching C's rescore_isolated_domain():
    // C: bitscore = envsc + (L - Ld) * log(L / (L+3))
    //    bitscore = (bitscore - (nullsc + dombias)) / log(2)
    // The first line adds a correction for non-envelope residues under the null model.
    // nullsc is the full-sequence null model score.
    let length_correction = (l - env_len) as f32 * (l as f32 / (l as f32 + 3.0)).ln();
    let dom_bitscore =
        (env_fwd_sc + length_correction - (null_sc + dom_bias)) / std::f32::consts::LN_2;
    let tau = gm.evparam[crate::hmm::P7_FTAU] as f64;
    let lambda = gm.evparam[crate::hmm::P7_FLAMBDA] as f64;
    let dom_lnp = crate::stats::exponential::surv(dom_bitscore as f64, tau, lambda).ln();

    let domain = Domain {
        iali,
        jali,
        ienv: ienv as i64,
        jenv: jenv as i64,
        bitscore: dom_bitscore,
        lnp: dom_lnp,
        dombias: (dom_bias / std::f32::consts::LN_2).max(0.0),
        oasc,
        envsc: env_fwd_sc,
        domcorrection: dom_correction,
        is_reported: false,
        is_included: false,
        ad,
    };

    #[cfg(feature = "tracehash")]
    {
        let inputs = |th: &mut tracehash::Call| {
            th.input_usize(seq_len);
            th.input_usize(gm.m);
            th.input_usize(env_len);
            th.input_usize(ienv);
            th.input_usize(jenv);
            th.input_bool(null2_is_done);
            th.input_bytes(&dsq[ienv..=jenv]);
        };

        let mut th = tracehash::th_call!("score_domain_forward");
        inputs(&mut th);
        th.output_f32(domain.envsc);
        th.output_f32_quant(domain.envsc, 1.0e-5);
        th.finish();

        let mut th = tracehash::th_call!("score_domain_null2");
        inputs(&mut th);
        th.output_f32(domain.domcorrection);
        th.output_f32_quant(domain.domcorrection, 1.0e-5);
        th.finish();

        let mut th = tracehash::th_call!("score_domain_oa");
        inputs(&mut th);
        th.output_f32(domain.oasc);
        th.output_f32_quant(domain.oasc, 1.0e-5);
        th.output_i64(domain.iali);
        th.output_i64(domain.jali);
        th.finish();
    }

    domain
}

#[cfg(feature = "tracehash")]
fn trace_define_domains_summary(
    l: usize,
    m: usize,
    domains_len: usize,
    nexpected: f32,
    seq_bias: f32,
    stats: DomainDefinitionStats,
) {
    let mut th = tracehash::th_call!("define_domains_summary");
    th.input_usize(l);
    th.input_usize(m);
    th.output_u64(domains_len as u64);
    th.output_f32(nexpected);
    th.output_f32(seq_bias);
    th.output_u64(stats.nregions as u64);
    th.output_u64(stats.nclustered as u64);
    th.output_u64(stats.noverlaps as u64);
    th.output_u64(stats.nenvelopes as u64);
    th.finish();
}

/// Run domain definition on a sequence that passed Forward filter.
/// Port of p7_domaindef_ByPosteriorHeuristics().
/// Returns (domains, nexpected, seq_bias_nats, stats).
pub fn define_domains(
    dsq: &[Dsq],
    l: usize,
    gm: &Profile,
    om: Option<&crate::simd::oprofile::OProfile>,
    fwd_pmx_input: Option<&crate::simd::probmx::ProbMx>,
    fwd_sc_input: Option<f32>,
    hmm: &Hmm,
    bg: &Bg,
    null_sc: f32,
    seed: u32,
    make_alignment: bool,
    make_alignment_display: bool,
) -> (Vec<Domain>, f32, f32, DomainDefinitionStats) {
    crate::logsum::p7_flogsuminit();

    // Phase 1: Domain decoding (btot/etot/mocc) — SIMD when available
    let (btot, etot, mocc);

    #[cfg(target_arch = "x86_64")]
    let use_simd = is_x86_feature_detected!("sse2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_simd = false;
    let mut simd_scratch = DomainSimdScratch::new();
    if use_simd {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::simd::oprofile::*;
            use crate::simd::probmx::p_domain_decoding;

            let om_storage;
            let om = if let Some(om) = om {
                om
            } else {
                om_storage = OProfile::convert(gm);
                &om_storage
            };
            let mut fwd_pmx_storage = None;
            let (fwd_pmx_ref, fwd_sc) =
                if let (Some(fwd_pmx), Some(fwd_sc)) = (fwd_pmx_input, fwd_sc_input) {
                    (fwd_pmx, fwd_sc)
                } else {
                    let mut fwd_pmx = crate::simd::probmx::ProbMx::new(l);
                    let fwd_sc = unsafe {
                        crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                            dsq,
                            0,
                            l,
                            om,
                            &mut fwd_pmx,
                            &mut simd_scratch.fwd_dp,
                        )
                    };
                    let fwd_pmx_ref = fwd_pmx_storage.insert(fwd_pmx);
                    (&*fwd_pmx_ref, fwd_sc)
                };
            let mut bck_pmx = crate::simd::probmx::ProbMx::new(l);
            unsafe {
                crate::simd::bck_filter::backward_parser_pmx_offset_with_scratch(
                    dsq,
                    0,
                    l,
                    om,
                    fwd_sc,
                    &mut bck_pmx,
                    Some(&fwd_pmx_ref.row_scale),
                    &mut simd_scratch.bck_prev,
                    &mut simd_scratch.bck_cur,
                );
            };
            #[cfg(feature = "tracehash")]
            {
                trace_pmx_forward_specials_summary(dsq, l, gm.m, fwd_pmx_ref);
                trace_pmx_forward_states_summary(dsq, l, gm.m, fwd_pmx_ref);
                trace_pmx_forward_scales_summary(dsq, l, gm.m, fwd_pmx_ref);
                trace_pmx_forward_states_q1e5_summary(dsq, l, gm.m, fwd_pmx_ref);
                trace_pmx_forward_anchor_q1e5("row0", dsq, l, gm.m, 0, fwd_pmx_ref);
                if l >= 1 {
                    trace_pmx_forward_anchor_q1e5("row1", dsq, l, gm.m, 1, fwd_pmx_ref);
                }
                if l >= 2 {
                    trace_pmx_forward_anchor_q1e5("row2", dsq, l, gm.m, 2, fwd_pmx_ref);
                }
                if l >= 4 {
                    trace_pmx_forward_anchor_q1e5("row4", dsq, l, gm.m, 4, fwd_pmx_ref);
                }
                if l >= 8 {
                    trace_pmx_forward_anchor_q1e5("row8", dsq, l, gm.m, 8, fwd_pmx_ref);
                }
                if l >= 16 {
                    trace_pmx_forward_anchor_q1e5("row16", dsq, l, gm.m, 16, fwd_pmx_ref);
                }
                if l >= 17 {
                    trace_pmx_forward_anchor_q1e5("row17", dsq, l, gm.m, 17, fwd_pmx_ref);
                    trace_pmx_forward_row_parts_q1e5(17, dsq, l, gm.m, fwd_pmx_ref);
                }
                if l >= 18 {
                    trace_pmx_forward_anchor_q1e5("row18", dsq, l, gm.m, 18, fwd_pmx_ref);
                }
                if l >= 19 {
                    trace_pmx_forward_anchor_q1e5("row19", dsq, l, gm.m, 19, fwd_pmx_ref);
                    trace_pmx_forward_row_parts_q1e5(19, dsq, l, gm.m, fwd_pmx_ref);
                }
                if l >= 20 {
                    trace_pmx_forward_anchor_q1e5("row20", dsq, l, gm.m, 20, fwd_pmx_ref);
                }
                if l >= 24 {
                    trace_pmx_forward_anchor_q1e5("row24", dsq, l, gm.m, 24, fwd_pmx_ref);
                }
                if l >= 28 {
                    trace_pmx_forward_anchor_q1e5("row28", dsq, l, gm.m, 28, fwd_pmx_ref);
                }
                if l >= 32 {
                    trace_pmx_forward_anchor_q1e5("row32", dsq, l, gm.m, 32, fwd_pmx_ref);
                }
                if l >= 64 {
                    trace_pmx_forward_anchor_q1e5("row64", dsq, l, gm.m, 64, fwd_pmx_ref);
                }
                if l >= 128 {
                    trace_pmx_forward_anchor_q1e5("row128", dsq, l, gm.m, 128, fwd_pmx_ref);
                }
                if let Some(pos) = first_forward_scale_row(l, fwd_pmx_ref) {
                    trace_pmx_forward_anchor_q1e5("first_scale", dsq, l, gm.m, pos, fwd_pmx_ref);
                }
                trace_pmx_forward_anchor_q1e5("rowl", dsq, l, gm.m, l, fwd_pmx_ref);
                trace_pmx_forward_scale_events_summary(dsq, l, gm.m, fwd_pmx_ref);
                trace_pmx_backward_specials_summary(dsq, l, gm.m, &bck_pmx);
                trace_pmx_backward_states_summary(dsq, l, gm.m, &bck_pmx);
                trace_pmx_backward_scales_summary(dsq, l, gm.m, &bck_pmx);
                trace_pmx_backward_states_q1e5_summary(dsq, l, gm.m, &bck_pmx);
                trace_pmx_backward_anchor_q1e5("row0", dsq, l, gm.m, 0, &bck_pmx);
                trace_pmx_backward_row_parts_q1e5(0, dsq, l, gm.m, &bck_pmx);
                if l >= 1 {
                    trace_pmx_backward_anchor_q1e5("row1", dsq, l, gm.m, 1, &bck_pmx);
                    trace_pmx_backward_row_parts_q1e5(1, dsq, l, gm.m, &bck_pmx);
                }
                if l >= 2 {
                    trace_pmx_backward_anchor_q1e5("row2", dsq, l, gm.m, 2, &bck_pmx);
                }
                if l >= 4 {
                    trace_pmx_backward_anchor_q1e5("row4", dsq, l, gm.m, 4, &bck_pmx);
                }
                if l >= 8 {
                    trace_pmx_backward_anchor_q1e5("row8", dsq, l, gm.m, 8, &bck_pmx);
                }
                if l >= 16 {
                    trace_pmx_backward_anchor_q1e5("row16", dsq, l, gm.m, 16, &bck_pmx);
                }
                if l >= 32 {
                    trace_pmx_backward_anchor_q1e5("row32", dsq, l, gm.m, 32, &bck_pmx);
                }
                if l >= 64 {
                    trace_pmx_backward_anchor_q1e5("row64", dsq, l, gm.m, 64, &bck_pmx);
                }
                if l >= 128 {
                    trace_pmx_backward_anchor_q1e5("row128", dsq, l, gm.m, 128, &bck_pmx);
                }
                if let Some(pos) = first_forward_scale_row(l, fwd_pmx_ref) {
                    trace_pmx_backward_anchor_q1e5("first_scale", dsq, l, gm.m, pos, &bck_pmx);
                }
                trace_pmx_backward_anchor_q1e5("rowl", dsq, l, gm.m, l, &bck_pmx);
            }
            let njc_loop = [
                om.xf[P7O_N][P7O_LOOP],
                om.xf[P7O_J][P7O_LOOP],
                om.xf[P7O_C][P7O_LOOP],
            ];
            let r = p_domain_decoding(fwd_pmx_ref, &bck_pmx, l, njc_loop);
            btot = r.0;
            etot = r.1;
            mocc = r.2;
        }
        #[cfg(not(target_arch = "x86_64"))]
        unreachable!();
    } else {
        let mut gx_fwd = Gmx::new(gm.m, l);
        g_forward(dsq, l, gm, &mut gx_fwd);
        let mut gx_bck = Gmx::new(gm.m, l);
        g_backward(dsq, l, gm, &mut gx_bck);
        let r = domain_decoding(gm, &gx_fwd, &gx_bck);
        btot = r.0;
        etot = r.1;
        mocc = r.2;
    }

    #[cfg(feature = "tracehash")]
    trace_domain_decoding_summary(dsq, l, gm.m, &btot, &etot, &mocc);

    let nexpected = btot[l].max(0.01);

    // Region detection using C's state machine
    let regions = find_domain_regions(&btot, &etot, &mocc, l);
    let mut stats = DomainDefinitionStats {
        nregions: regions.len(),
        ..Default::default()
    };
    let mut n2sc = vec![0.0_f32; l + 1];
    let mut env_gm = gm.clone();
    reconfig_unihit(&mut env_gm, l as i32);
    let optacc_deltas = OptAccTDelta::from_profile(&env_gm);
    #[cfg(target_arch = "x86_64")]
    let simd_match_odds = if use_simd {
        Some(crate::simd::probmx::match_odds_from_rsc(
            &gm.rsc, gm.abc_k, gm.m,
        ))
    } else {
        None
    };
    #[cfg(not(target_arch = "x86_64"))]
    let simd_match_odds: Option<Vec<f32>> = None;
    #[cfg(target_arch = "x86_64")]
    let env_om = if use_simd {
        if let Some(om) = om {
            let mut env_om = om.clone();
            env_om.reconfig_unihit(l as i32);
            Some(env_om)
        } else {
            Some(crate::simd::oprofile::OProfile::convert(&env_gm))
        }
    } else {
        None
    };
    #[cfg(not(target_arch = "x86_64"))]
    let env_om: Option<crate::simd::oprofile::OProfile> = None;

    if regions.is_empty() {
        // No regions found — if nexpected > 0, return single domain covering sequence
        if nexpected >= 0.5 {
            let dom = score_domain_envelope(
                dsq,
                l,
                gm,
                &env_gm,
                env_om.as_ref(),
                hmm,
                1,
                l,
                null_sc,
                &bg.f,
                Some(&mut n2sc),
                false,
                simd_match_odds.as_deref(),
                &optacc_deltas,
                Some(&mut simd_scratch),
                make_alignment,
                make_alignment_display,
            );
            stats.nenvelopes += 1;
            let seq_bias = n2sc.iter().sum();
            #[cfg(feature = "tracehash")]
            trace_define_domains_summary(l, gm.m, 1, nexpected, seq_bias, stats);
            return (vec![dom], nexpected, seq_bias, stats);
        }
        #[cfg(feature = "tracehash")]
        trace_define_domains_summary(l, gm.m, 0, nexpected, 0.0, stats);
        return (Vec::new(), nexpected, 0.0, stats);
    }

    let mut domains = Vec::new();

    for (_region_idx, &(ri, rj)) in regions.iter().enumerate() {
        let is_multi = is_multidomain_region(&btot, &etot, ri, rj);
        #[cfg(feature = "tracehash")]
        trace_domain_region(l, gm.m, _region_idx, ri, rj, is_multi);
        if is_multi {
            stats.nclustered += 1;
            // Multi-domain region: resolve by stochastic traceback clustering
            // Run Forward on the region in multihit mode
            let region_len = rj - ri + 1;
            let mut region_gm = gm.clone();
            reconfig_multihit(&mut region_gm, l as i32);

            let mut sub_dsq = vec![crate::alphabet::DSQ_SENTINEL];
            sub_dsq.extend_from_slice(&dsq[ri..=rj]);
            sub_dsq.push(crate::alphabet::DSQ_SENTINEL);

            let mut region_fwd = Gmx::new(gm.m, region_len);
            g_forward(&sub_dsq, region_len, &region_gm, &mut region_fwd);

            #[cfg(target_arch = "x86_64")]
            let region_trace_pmx = if use_simd {
                if let Some(om) = om {
                    let mut region_om = om.clone();
                    region_om.reconfig_multihit(l as i32);
                    let mut region_pmx = crate::simd::probmx::ProbMx::new_full(gm.m, region_len);
                    unsafe {
                        crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                            dsq,
                            ri - 1,
                            region_len,
                            &region_om,
                            &mut region_pmx,
                            &mut simd_scratch.fwd_dp,
                        );
                    }
                    #[cfg(feature = "tracehash")]
                    trace_region_forward_summary(
                        dsq,
                        l,
                        gm.m,
                        ri,
                        rj,
                        &region_om,
                        &region_pmx,
                    );
                    Some((region_om, region_pmx))
                } else {
                    None
                }
            } else {
                None
            };

            let mut rng = MersenneTwister::new(seed);
            let mut segments = Vec::new();
            for pos in ri..=rj {
                n2sc[pos] = 0.0;
            }

            for trace_idx in 0..NSAMPLES {
                #[cfg(target_arch = "x86_64")]
                let tr = if let Some((ref region_om, ref region_pmx)) = region_trace_pmx {
                    crate::dp::generic_stotrace::stochastic_trace_pmx(
                        &mut rng,
                        region_len,
                        region_om,
                        region_pmx,
                    )
                } else {
                    g_stochastic_trace(&mut rng, &sub_dsq, region_len, &region_gm, &region_fwd)
                };
                #[cfg(not(target_arch = "x86_64"))]
                let tr =
                    g_stochastic_trace(&mut rng, &sub_dsq, region_len, &region_gm, &region_fwd);
                let mut in_domain = false;
                let mut seg_i = 0;
                let mut seg_k = 0;
                let mut seg_j = 0;
                let mut seg_m = 0;
                let mut zstart = 0usize;
                let mut trace_domains = Vec::new();

                for z in 0..tr.n {
                    match tr.st[z] {
                        State::B => {
                            in_domain = true;
                            seg_i = 0;
                            seg_k = 0;
                            zstart = z;
                        }
                        State::M if in_domain => {
                            if seg_i == 0 {
                                seg_i = tr.i[z];
                                seg_k = tr.k[z];
                            }
                            seg_j = tr.i[z];
                            seg_m = tr.k[z];
                        }
                        State::E if in_domain => {
                            if seg_i > 0 && seg_j > 0 {
                                let global_i = seg_i + ri - 1;
                                let global_j = seg_j + ri - 1;
                                segments.push(SegmentPair {
                                    i: global_i,
                                    j: global_j,
                                    k: seg_k,
                                    m: seg_m,
                                    trace_idx,
                                });
                                trace_domains.push((seg_i, seg_j, zstart, z));
                            }
                            in_domain = false;
                        }
                        _ => {}
                    }
                }

                let mut pos = 1usize;
                for (_domain_idx, &(sqfrom, sqto, tfrom, tto)) in trace_domains.iter().enumerate() {
                    #[cfg(feature = "tracehash")]
                    trace_region_null2_segment(
                        gm.m, ri, rj, trace_idx, _domain_idx, sqfrom, sqto, tfrom, tto, &tr,
                    );
                    #[cfg(target_arch = "x86_64")]
                    let null2 = if let Some((ref region_om, _)) = region_trace_pmx {
                        null2_by_trace_optimized(region_om, &tr, tfrom, tto)
                    } else {
                        generic_null2::null2_by_trace(&region_gm, &tr, tfrom, tto)
                    };
                    #[cfg(not(target_arch = "x86_64"))]
                    let null2 = generic_null2::null2_by_trace(&region_gm, &tr, tfrom, tto);
                    while pos <= sqfrom && pos <= region_len {
                        n2sc[ri + pos - 1] += 1.0;
                        pos += 1;
                    }
                    while pos <= sqto && pos <= region_len {
                        let x = dsq[ri + pos - 1] as usize;
                        n2sc[ri + pos - 1] += if x < null2.len() { null2[x] } else { 1.0 };
                        pos += 1;
                    }
                }
                while pos <= region_len {
                    n2sc[ri + pos - 1] += 1.0;
                    pos += 1;
                }
            }

            for pos in ri..=rj {
                n2sc[pos] = (n2sc[pos] / NSAMPLES as f32).ln();
            }

            if !segments.is_empty() {
                let params = ClusterParams::default();
                let envelopes = spensemble::cluster(&segments, NSAMPLES, &params);
                #[cfg(feature = "tracehash")]
                trace_domain_cluster_summary(l, gm.m, ri, rj, segments.len(), envelopes.len());
                let mut last_jenv = 0usize;
                for (_env_idx, env) in envelopes.iter().enumerate() {
                    let ienv = env.ienv.max(1).min(l);
                    let jenv = env.jenv.max(1).min(l);
                    if jenv >= ienv {
                        #[cfg(feature = "tracehash")]
                        trace_domain_envelope_candidate(
                            l, gm.m, ri, rj, _env_idx, ienv, jenv, true,
                        );
                        if ienv <= last_jenv {
                            stats.noverlaps += 1;
                        }
                        stats.nenvelopes += 1;
                        domains.push(score_domain_envelope(
                            dsq,
                            l,
                            gm,
                            &env_gm,
                            env_om.as_ref(),
                            hmm,
                            ienv,
                            jenv,
                            null_sc,
                            &bg.f,
                            Some(&mut n2sc),
                            true,
                            simd_match_odds.as_deref(),
                            &optacc_deltas,
                            Some(&mut simd_scratch),
                            make_alignment,
                            make_alignment_display,
                        ));
                        last_jenv = jenv;
                    }
                }
            } else {
                // Clustering failed — treat as single domain
                #[cfg(feature = "tracehash")]
                trace_domain_cluster_summary(l, gm.m, ri, rj, 0, 1);
                #[cfg(feature = "tracehash")]
                trace_domain_envelope_candidate(l, gm.m, ri, rj, 0, ri, rj, true);
                stats.nenvelopes += 1;
                domains.push(score_domain_envelope(
                    dsq,
                    l,
                    gm,
                    &env_gm,
                    env_om.as_ref(),
                    hmm,
                    ri,
                    rj,
                    null_sc,
                    &bg.f,
                    Some(&mut n2sc),
                    false,
                    simd_match_odds.as_deref(),
                    &optacc_deltas,
                    Some(&mut simd_scratch),
                    make_alignment,
                    make_alignment_display,
                ));
            }
        } else {
            // Single-domain region: the region IS the envelope
            #[cfg(feature = "tracehash")]
            trace_domain_envelope_candidate(l, gm.m, ri, rj, 0, ri, rj, false);
            stats.nenvelopes += 1;
            domains.push(score_domain_envelope(
                dsq,
                l,
                gm,
                &env_gm,
                env_om.as_ref(),
                hmm,
                ri,
                rj,
                null_sc,
                &bg.f,
                Some(&mut n2sc),
                false,
                simd_match_odds.as_deref(),
                &optacc_deltas,
                Some(&mut simd_scratch),
                make_alignment,
                make_alignment_display,
            ));
        }
    }

    // Sort domains by envelope start position
    domains.sort_by_key(|d| d.ienv);

    let seq_bias = n2sc.iter().sum();
    #[cfg(feature = "tracehash")]
    trace_define_domains_summary(l, gm.m, domains.len(), nexpected, seq_bias, stats);
    (domains, nexpected, seq_bias, stats)
}
