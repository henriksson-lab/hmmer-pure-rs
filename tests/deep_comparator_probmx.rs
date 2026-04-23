#![cfg(target_arch = "x86_64")]

use hmmer_pure_rs::{alphabet::Alphabet, bg::Bg, hmmfile, profile, simd};
use std::path::Path;

fn configured_profile(
    seq_len: usize,
) -> (Vec<hmmer_pure_rs::alphabet::Dsq>, simd::oprofile::OProfile) {
    let hmms = hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_data/Pkinase_pfam.hmm"
    )))
    .unwrap();
    let hmm = &hmms[0];
    let abc = Alphabet::new(hmm.abc_type);
    let bg = Bg::new(&abc);
    let mut gm = profile::Profile::new(hmm.m, &abc);
    profile::profile_config(hmm, &bg, &mut gm, seq_len as i32, profile::P7_LOCAL);

    let motif = b"MQLVETKGGTFGKVYKARDLKSEMEVAIKQIEHPNVVKLLGACTQGGPLYVLMEYAAKGSLRDLVRR";
    let mut seq = Vec::with_capacity(seq_len);
    while seq.len() < seq_len {
        let take = motif.len().min(seq_len - seq.len());
        seq.extend_from_slice(&motif[..take]);
    }
    let dsq = abc.digitize(&seq);
    (dsq, simd::oprofile::OProfile::convert(&gm))
}

fn assert_probmx_equal(label: &str, got: &simd::probmx::ProbMx, expected: &simd::probmx::ProbMx) {
    assert_f32_slice_exact(&format!("{label}.xmx"), &got.xmx, &expected.xmx);
    assert_f64_slice_exact(&format!("{label}.scale"), &got.scale, &expected.scale);
    assert_f32_slice_exact(
        &format!("{label}.row_scale"),
        &got.row_scale,
        &expected.row_scale,
    );
    assert_f32_slice_exact(
        &format!("{label}.striped_dp"),
        &got.striped_dp,
        &expected.striped_dp,
    );
}

fn assert_f32_exact(label: &str, got: f32, expected: f32) {
    assert!(
        got.to_bits() == expected.to_bits(),
        "{label} differed: got {got:?} ({:#010x}), expected {expected:?} ({:#010x})",
        got.to_bits(),
        expected.to_bits()
    );
}

fn assert_f32_slice_exact(label: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label} length differed: got {}, expected {}",
        got.len(),
        expected.len()
    );
    for (idx, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert_f32_exact(&format!("{label}[{idx}]"), g, e);
    }
}

fn assert_f64_slice_exact(label: &str, got: &[f64], expected: &[f64]) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label} length differed: got {}, expected {}",
        got.len(),
        expected.len()
    );
    for (idx, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            g.to_bits() == e.to_bits(),
            "{label}[{idx}] differed: got {g:?} ({:#018x}), expected {e:?} ({:#018x})",
            g.to_bits(),
            e.to_bits()
        );
    }
}

#[test]
fn resized_probmx_overwrites_poisoned_storage() {
    if !std::is_x86_feature_detected!("sse2") {
        return;
    }

    let seq_len = 257;
    let (dsq, om) = configured_profile(seq_len);
    let mut clean_fwd = simd::probmx::ProbMx::new_full(om.m, seq_len);
    let mut reused_fwd = simd::probmx::ProbMx::new_full(om.m, 7);
    reused_fwd.striped_dp.fill(12345.5);
    reused_fwd.resize_full(om.m, seq_len);

    let mut clean_dp = Vec::new();
    let mut reused_dp = Vec::new();
    let clean_sc = unsafe {
        simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
            &dsq,
            0,
            seq_len,
            &om,
            &mut clean_fwd,
            &mut clean_dp,
        )
    };
    let reused_sc = unsafe {
        simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
            &dsq,
            0,
            seq_len,
            &om,
            &mut reused_fwd,
            &mut reused_dp,
        )
    };

    assert_f32_exact("forward_score", reused_sc, clean_sc);
    assert_probmx_equal("forward", &reused_fwd, &clean_fwd);

    let mut clean_bck = simd::probmx::ProbMx::new_full(om.m, seq_len);
    let mut reused_bck = simd::probmx::ProbMx::new_full(om.m, 11);
    reused_bck.striped_dp.fill(-7777.25);
    reused_bck.resize_full(om.m, seq_len);

    let mut clean_prev = Vec::new();
    let mut clean_cur = Vec::new();
    let mut reused_prev = Vec::new();
    let mut reused_cur = Vec::new();
    let clean_bck_sc = unsafe {
        simd::bck_filter::backward_parser_pmx_offset_with_scratch(
            &dsq,
            0,
            seq_len,
            &om,
            clean_sc,
            &mut clean_bck,
            Some(&clean_fwd.row_scale),
            &mut clean_prev,
            &mut clean_cur,
        )
    };
    let reused_bck_sc = unsafe {
        simd::bck_filter::backward_parser_pmx_offset_with_scratch(
            &dsq,
            0,
            seq_len,
            &om,
            clean_sc,
            &mut reused_bck,
            Some(&clean_fwd.row_scale),
            &mut reused_prev,
            &mut reused_cur,
        )
    };

    assert_f32_exact("backward_score", reused_bck_sc, clean_bck_sc);
    assert_probmx_equal("backward", &reused_bck, &clean_bck);
}
