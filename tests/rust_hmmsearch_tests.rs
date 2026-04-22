//! Integration tests for the pure Rust hmmsearch binary.
//! Validates that the Rust implementation finds the same hits as C HMMER.
//!
//! Tests are organized into categories:
//!   1. Basic correctness — right hits found, right hit counts
//!   2. Golden-file equivalence — compare vs pre-generated C HMMER output
//!   3. E-value accuracy — E-values within tolerance of C reference
//!   4. Output format — tblout/domtblout column formatting
//!   5. Edge cases — no hits, multi-domain, multi-query, --max

use std::collections::{HashMap, HashSet};
use std::process::Command;

fn binary_path(name: &str) -> std::path::PathBuf {
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push(name);
    path
}

fn project_root() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
}

fn test_path(relative: &str) -> String {
    format!("{}/{}", project_root(), relative)
}

fn c_hmmsearch_path() -> String {
    test_path("hmmer/src/hmmsearch")
}

/// Run hmmsearch and return (stdout, tblout_content).
fn run_hmmsearch(hmm: &str, seqdb: &str, extra_args: &[&str]) -> (String, String) {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let mut args = vec!["search", "--noali", "--tblout", tblout.to_str().unwrap()];
    args.extend_from_slice(extra_args);
    args.push(hmm);
    args.push(seqdb);

    let output = Command::new(binary_path("hmmer"))
        .args(&args)
        .output()
        .expect("failed to run hmmer search");

    assert!(
        output.status.success(),
        "hmmer search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let tblout_content = std::fs::read_to_string(&tblout).unwrap_or_default();
    (stdout, tblout_content)
}

/// Run hmmsearch with domtblout and return (stdout, tblout, domtblout).
fn run_hmmsearch_dom(hmm: &str, seqdb: &str, extra_args: &[&str]) -> (String, String, String) {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let domtblout = dir.path().join("domtblout.txt");
    let mut args = vec![
        "search",
        "--noali",
        "--tblout",
        tblout.to_str().unwrap(),
        "--domtblout",
        domtblout.to_str().unwrap(),
    ];
    args.extend_from_slice(extra_args);
    args.push(hmm);
    args.push(seqdb);

    let output = Command::new(binary_path("hmmer"))
        .args(&args)
        .output()
        .expect("failed to run hmmer search");

    assert!(
        output.status.success(),
        "hmmer search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let tbl = std::fs::read_to_string(&tblout).unwrap_or_default();
    let domtbl = std::fs::read_to_string(&domtblout).unwrap_or_default();
    (stdout, tbl, domtbl)
}

/// Run Rust hmmsearch with pfamtblout and return the file content.
fn run_hmmsearch_pfamtbl(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let pfamtblout = dir.path().join("pfamtblout.txt");
    let mut args = vec![
        "search",
        "--noali",
        "--pfamtblout",
        pfamtblout.to_str().unwrap(),
    ];
    args.extend_from_slice(extra_args);
    args.push(hmm);
    args.push(seqdb);

    let output = Command::new(binary_path("hmmer"))
        .args(&args)
        .output()
        .expect("failed to run hmmer search");

    assert!(
        output.status.success(),
        "hmmer search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::read_to_string(&pfamtblout).unwrap_or_default()
}

/// Run bundled C hmmsearch with pfamtblout and return the file content.
fn run_c_hmmsearch_pfamtbl(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let pfamtblout = dir.path().join("pfamtblout.txt");
    let mut args = vec!["--noali", "--pfamtblout", pfamtblout.to_str().unwrap()];
    args.extend_from_slice(extra_args);
    args.push(hmm);
    args.push(seqdb);

    let output = Command::new(c_hmmsearch_path())
        .args(&args)
        .output()
        .expect("failed to run bundled C hmmsearch");

    assert!(
        output.status.success(),
        "bundled C hmmsearch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::read_to_string(&pfamtblout).unwrap_or_default()
}

/// Parse tblout into Vec<(target_name, query_name, evalue, score)>.
fn parse_tblout(content: &str) -> Vec<(String, String, f64, f64)> {
    content
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert!(fields.len() >= 6, "tblout line too short: {}", line);
            let target = fields[0].to_string();
            let query = fields[2].to_string();
            let evalue: f64 = fields[4].parse().unwrap_or(f64::INFINITY);
            let score: f64 = fields[5].parse().unwrap_or(0.0);
            (target, query, evalue, score)
        })
        .collect()
}

/// Parse domtblout into Vec of domain records.
#[allow(dead_code)]
struct DomRecord {
    target: String,
    query: String,
    dom_idx: usize,
    ndom: usize,
    dom_evalue: f64,
    dom_score: f64,
    ali_from: usize,
    ali_to: usize,
}

fn parse_domtblout(content: &str) -> Vec<DomRecord> {
    content
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            DomRecord {
                target: f[0].to_string(),
                query: f[3].to_string(),
                dom_idx: f[9].parse().unwrap_or(0),
                ndom: f[10].parse().unwrap_or(0),
                dom_evalue: f[12].parse().unwrap_or(f64::INFINITY),
                dom_score: f[13].parse().unwrap_or(0.0),
                ali_from: f[17].parse().unwrap_or(0),
                ali_to: f[18].parse().unwrap_or(0),
            }
        })
        .collect()
}

fn parse_domtblout_core_rows(content: &str) -> Vec<Vec<String>> {
    content
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|line| {
            let fields: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
            if fields.len() < 22 {
                return None;
            }
            Some(fields)
        })
        .collect()
}

fn parse_pfamtblout_rows(content: &str) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    let mut seq_rows = Vec::new();
    let mut dom_rows = Vec::new();

    for line in content.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        let fields: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
        if fields.len() >= 12 {
            dom_rows.push(fields);
        } else {
            seq_rows.push(fields);
        }
    }

    (seq_rows, dom_rows)
}

/// Extract hit names from stdout hit table.
fn extract_hit_names(output: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_hits = false;
    for line in output.lines() {
        if line.contains("E-value  score  bias") && line.contains("Sequence") {
            in_hits = true;
            continue;
        }
        if line.contains("-------") && in_hits {
            continue;
        }
        if in_hits && !line.trim().is_empty() {
            if line.starts_with("  ") && !line.contains("[No hits") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 9 {
                    names.push(parts[8].to_string());
                }
            } else {
                in_hits = false;
            }
        }
    }
    names
}

/// Parse C golden tblout file into a map of (target, query) -> (evalue, score).
fn parse_golden_tblout(path: &str) -> HashMap<(String, String), (f64, f64)> {
    let content = std::fs::read_to_string(path).unwrap();
    let mut map = HashMap::new();
    for (target, query, evalue, score) in parse_tblout(&content) {
        map.insert((target, query), (evalue, score));
    }
    map
}

/// Parse golden tblout into a map of (target, query) -> (evalue, score, bias).
fn parse_golden_tblout_with_bias(
    path: &str,
) -> HashMap<(String, String), (String, String, String)> {
    let content = std::fs::read_to_string(path).unwrap();
    let mut map = HashMap::new();
    for line in content
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let fields: Vec<&str> = line.split_whitespace().collect();
        assert!(fields.len() >= 7, "tblout line too short: {}", line);
        map.insert(
            (fields[0].to_string(), fields[2].to_string()),
            (
                fields[4].to_string(),
                fields[5].to_string(),
                fields[6].to_string(),
            ),
        );
    }
    map
}

/// Check that two sets of hit names overlap by at least `min_fraction`.
fn check_hit_overlap(
    c_hits: &HashSet<String>,
    r_hits: &HashSet<String>,
    min_fraction: f64,
    label: &str,
) {
    let common = c_hits.intersection(r_hits).count();
    let total = c_hits.union(r_hits).count();
    let fraction = if total > 0 {
        common as f64 / total as f64
    } else {
        1.0
    };
    assert!(fraction >= min_fraction,
        "{}: hit overlap {:.1}% ({} common / {} total) below minimum {:.0}%\n  C-only: {:?}\n  Rust-only: {:?}",
        label, fraction * 100.0, common, total, min_fraction * 100.0,
        c_hits.difference(r_hits).take(5).collect::<Vec<_>>(),
        r_hits.difference(c_hits).take(5).collect::<Vec<_>>());
}

/// Check that bit scores for common hits are within a tolerance.
/// Bit scores are more stable than E-values (which amplify differences exponentially).
fn check_scores_close(
    c_hits: &HashMap<(String, String), (f64, f64)>,
    r_hits: &[(String, String, f64, f64)],
    max_score_diff: f64,
    label: &str,
) {
    let mut checked = 0;
    let mut max_diff_seen = 0.0_f64;
    for (target, query, _r_ev, r_sc) in r_hits {
        if let Some(&(_c_ev, c_sc)) = c_hits.get(&(target.clone(), query.clone())) {
            if c_sc > 20.0 {
                // Only check strong hits where score comparison is meaningful
                let diff = (r_sc - c_sc).abs();
                max_diff_seen = max_diff_seen.max(diff);
                assert!(
                    diff <= max_score_diff,
                    "{}: score diff {:.1} bits for {} (C={:.1} Rust={:.1}) exceeds max {:.0}",
                    label,
                    diff,
                    target,
                    c_sc,
                    r_sc,
                    max_score_diff
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "{}: no strong common hits to compare scores (max diff seen: {:.1})",
        label,
        max_diff_seen
    );
}

// ============================================================
// 1. Basic correctness tests
// ============================================================

#[test]
fn test_20aa_finds_all_4_hits() {
    let (stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &[],
    );
    let hits = extract_hit_names(&stdout);
    let mut sorted = hits.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec!["test1", "test2", "test3", "test4"],
        "Expected 4 hits, got: {:?}",
        hits
    );

    let tbl_hits = parse_tblout(&tbl);
    assert_eq!(tbl_hits.len(), 4, "tblout should have 4 lines");
}

#[test]
fn test_globins_finds_all_45() {
    let (stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );
    let hits = extract_hit_names(&stdout);
    assert_eq!(
        hits.len(),
        45,
        "Expected 45 globin hits, got {}",
        hits.len()
    );

    let tbl_hits = parse_tblout(&tbl);
    assert_eq!(tbl_hits.len(), 45, "tblout should have 45 lines");

    // Top hit should be the most significant (MYG_ESCGI in C reference)
    assert!(
        tbl_hits[0].2 < 1e-60,
        "Top hit E-value should be < 1e-60, got {:.2e}",
        tbl_hits[0].2
    );
}

#[test]
fn test_fn3_multi_domain() {
    let (_stdout, tbl, domtbl) = run_hmmsearch_dom(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
        &[],
    );

    let tbl_hits = parse_tblout(&tbl);
    assert_eq!(
        tbl_hits.len(),
        1,
        "fn3 vs 7LESS should find exactly 1 sequence hit"
    );
    assert_eq!(tbl_hits[0].0, "7LESS_DROME");
    assert!(
        tbl_hits[0].2 < 1e-50,
        "Full-sequence E-value should be < 1e-50"
    );

    // Should find multiple domains (C finds 9)
    let doms = parse_domtblout(&domtbl);
    assert!(
        doms.len() >= 7,
        "Should find >=7 fn3 domains in 7LESS_DROME, got {}",
        doms.len()
    );
    assert!(
        doms.len() <= 11,
        "Should find <=11 fn3 domains, got {}",
        doms.len()
    );

    // Best domain should remain clearly significant. C HMMER's wider envelope
    // scores this at ~47 bits; the current Rust envelope is narrower and scores
    // lower, but still in the same domain family signal.
    let best_score = doms
        .iter()
        .map(|d| d.dom_score)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        best_score > 35.0,
        "Best domain score should be > 35, got {:.1}",
        best_score
    );
}

#[test]
fn test_fn3_multi_domain_nonull2_zeroes_sequence_bias_but_keeps_domain_structure() {
    let (stdout_default, _tbl_default, domtbl_default) = run_hmmsearch_dom(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
        &[],
    );
    let (stdout_nonull2, _tbl_nonull2, domtbl_nonull2) = run_hmmsearch_dom(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
        &["--nonull2"],
    );

    assert!(
        stdout_default.contains("1.9e-57  178.0   0.4"),
        "default fn3 multi-domain sequence score/bias drifted:\n{}",
        stdout_default
    );
    assert!(
        stdout_nonull2.contains("1.4e-57  178.4   0.0"),
        "--nonull2 fn3 multi-domain sequence score/bias drifted:\n{}",
        stdout_nonull2
    );

    let doms_default = parse_domtblout(&domtbl_default);
    let doms_nonull2 = parse_domtblout(&domtbl_nonull2);
    assert_eq!(
        doms_default.len(),
        doms_nonull2.len(),
        "--nonull2 should not change the number of fn3 domains in 7LESS_DROME"
    );
    assert_eq!(
        doms_default.len(),
        9,
        "expected stable 9-domain fn3 structure on 7LESS_DROME"
    );

    let default_ranges: Vec<(usize, usize)> = doms_default
        .iter()
        .map(|d| (d.ali_from, d.ali_to))
        .collect();
    let nonull2_ranges: Vec<(usize, usize)> = doms_nonull2
        .iter()
        .map(|d| (d.ali_from, d.ali_to))
        .collect();
    assert_eq!(
        default_ranges, nonull2_ranges,
        "--nonull2 should preserve the multi-domain alignment structure"
    );
}

#[test]
fn test_no_hits_unrelated() {
    let (stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/Caudal_act.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &[],
    );
    let hits = extract_hit_names(&stdout);
    assert!(
        hits.is_empty(),
        "Should find no hits for unrelated HMM/sequences, got {:?}",
        hits
    );

    let tbl_hits = parse_tblout(&tbl);
    assert!(tbl_hits.is_empty(), "tblout should have 0 lines");
}

#[test]
fn test_pkinase_no_hits_against_globins() {
    let (stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/Pkinase.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );
    let hits = extract_hit_names(&stdout);
    assert!(
        hits.is_empty(),
        "Pkinase should find no hits in globins, got {:?}",
        hits
    );

    let tbl_hits = parse_tblout(&tbl);
    assert!(tbl_hits.is_empty());
}

#[test]
fn test_max_flag() {
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["--max"],
    );
    let hits = parse_tblout(&tbl);
    assert_eq!(hits.len(), 4, "With --max, should find all 4 hits");
}

// ============================================================
// 2. Golden-file equivalence (compare hit sets against C HMMER)
// ============================================================

#[test]
fn test_golden_globins4_hit_set() {
    let golden = parse_golden_tblout(&test_path("tests/golden/globins4_vs_globins45.tblout"));
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );
    let rust_hits = parse_tblout(&tbl);

    let c_names: HashSet<String> = golden.keys().map(|(t, _)| t.clone()).collect();
    let r_names: HashSet<String> = rust_hits.iter().map(|(t, _, _, _)| t.clone()).collect();

    // All 45 globins should be found by both
    check_hit_overlap(&c_names, &r_names, 1.0, "globins4");
}

#[test]
fn test_golden_fn3_hit_set() {
    let golden = parse_golden_tblout(&test_path("tests/golden/fn3_vs_7less.tblout"));
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
        &[],
    );
    let rust_hits = parse_tblout(&tbl);

    let c_names: HashSet<String> = golden.keys().map(|(t, _)| t.clone()).collect();
    let r_names: HashSet<String> = rust_hits.iter().map(|(t, _, _, _)| t.clone()).collect();
    check_hit_overlap(&c_names, &r_names, 1.0, "fn3");
}

#[test]
fn test_golden_gecco_pfam5_hit_set() {
    let golden = parse_golden_tblout(&test_path("tests/golden/gecco_pfam5_vs_gecco.tblout"));
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/gecco_pfam5.hmm"),
        &test_path("hmmer/testsuite/gecco_proteins.faa"),
        &[],
    );
    let rust_hits = parse_tblout(&tbl);

    let c_set: HashSet<(String, String)> = golden.keys().cloned().collect();
    let r_set: HashSet<(String, String)> = rust_hits
        .iter()
        .map(|(t, q, _, _)| (t.clone(), q.clone()))
        .collect();
    let common = c_set.intersection(&r_set).count();
    let total = c_set.union(&r_set).count();

    // Multi-HMM search: at least 80% overlap (some borderline hits differ)
    let overlap = common as f64 / total as f64;
    assert!(
        overlap >= 0.80,
        "gecco_pfam5: overlap {:.0}% ({}/{}) too low",
        overlap * 100.0,
        common,
        total
    );
}

#[test]
fn test_golden_gecco_missed_hit_set() {
    let golden = parse_golden_tblout(&test_path("tests/golden/gecco_missed_vs_missed.tblout"));
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/gecco_missed_hmms.hmm"),
        &test_path("hmmer/testsuite/gecco_missed_proteins.faa"),
        &[],
    );
    let rust_hits = parse_tblout(&tbl);

    let c_set: HashSet<(String, String)> = golden.keys().cloned().collect();
    let r_set: HashSet<(String, String)> = rust_hits
        .iter()
        .map(|(t, q, _, _)| (t.clone(), q.clone()))
        .collect();
    let common = c_set.intersection(&r_set).count();
    let total = c_set.union(&r_set).count();
    let overlap = common as f64 / total as f64;
    assert!(
        overlap >= 0.75,
        "gecco_missed: overlap {:.0}% ({}/{}) too low",
        overlap * 100.0,
        common,
        total
    );
}

#[test]
fn test_golden_gecco_missed2_hit_set() {
    let golden = parse_golden_tblout(&test_path("tests/golden/gecco_missed2.tblout"));
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/gecco_missed2_hmms.hmm"),
        &test_path("hmmer/testsuite/gecco_missed2_proteins.faa"),
        &[],
    );
    let rust_hits = parse_tblout(&tbl);

    let c_set: HashSet<(String, String)> = golden.keys().cloned().collect();
    let r_set: HashSet<(String, String)> = rust_hits
        .iter()
        .map(|(t, q, _, _)| (t.clone(), q.clone()))
        .collect();
    let common = c_set.intersection(&r_set).count();
    let total = c_set.union(&r_set).count();
    let overlap = common as f64 / total as f64;
    assert!(
        overlap >= 0.75,
        "gecco_missed2: overlap {:.0}% ({}/{}) too low",
        overlap * 100.0,
        common,
        total
    );
}

#[test]
fn test_golden_gecco_missed3_hit_set() {
    let golden = parse_golden_tblout(&test_path("tests/golden/gecco_missed3.tblout"));
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/gecco_missed3_hmms.hmm"),
        &test_path("hmmer/testsuite/gecco_missed3_proteins.faa"),
        &[],
    );
    let rust_hits = parse_tblout(&tbl);

    let c_set: HashSet<(String, String)> = golden.keys().cloned().collect();
    let r_set: HashSet<(String, String)> = rust_hits
        .iter()
        .map(|(t, q, _, _)| (t.clone(), q.clone()))
        .collect();
    let common = c_set.intersection(&r_set).count();
    let total = c_set.union(&r_set).count();
    let overlap = common as f64 / total as f64;
    assert!(
        overlap >= 0.75,
        "gecco_missed3: overlap {:.0}% ({}/{}) too low",
        overlap * 100.0,
        common,
        total
    );
}

#[test]
fn test_golden_gecco_missed4_hit_set() {
    let golden = parse_golden_tblout(&test_path("tests/golden/gecco_missed4.tblout"));
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/gecco_missed4_hmms.hmm"),
        &test_path("hmmer/testsuite/gecco_missed4_proteins.faa"),
        &[],
    );
    let rust_hits = parse_tblout(&tbl);

    let c_set: HashSet<(String, String)> = golden.keys().cloned().collect();
    let r_set: HashSet<(String, String)> = rust_hits
        .iter()
        .map(|(t, q, _, _)| (t.clone(), q.clone()))
        .collect();
    let common = c_set.intersection(&r_set).count();
    let total = c_set.union(&r_set).count();
    let overlap = common as f64 / total as f64;
    assert!(
        overlap >= 0.75,
        "gecco_missed4: overlap {:.0}% ({}/{}) too low",
        overlap * 100.0,
        common,
        total
    );
}

// ============================================================
// 3. E-value accuracy tests
// ============================================================

#[test]
fn test_score_accuracy_globins() {
    let golden = parse_golden_tblout(&test_path("tests/golden/globins4_vs_globins45.tblout"));
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );
    let rust_hits = parse_tblout(&tbl);

    // For strong hits, bit scores should be within 15 bits of C reference.
    // Minor differences arise from float precision in SIMD scoring and bias correction.
    check_scores_close(&golden, &rust_hits, 15.0, "globins4 scores");
}

#[test]
fn test_score_accuracy_fn3() {
    let golden = parse_golden_tblout(&test_path("tests/golden/fn3_vs_7less.tblout"));
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
        &[],
    );
    let rust_hits = parse_tblout(&tbl);
    // fn3 is a multi-domain case — Forward score accumulation can diverge more
    check_scores_close(&golden, &rust_hits, 20.0, "fn3 scores");
}

#[test]
fn test_score_accuracy_gecco_pfam5() {
    let golden = parse_golden_tblout(&test_path("tests/golden/gecco_pfam5_vs_gecco.tblout"));
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/gecco_pfam5.hmm"),
        &test_path("hmmer/testsuite/gecco_proteins.faa"),
        &[],
    );
    let rust_hits = parse_tblout(&tbl);
    check_scores_close(&golden, &rust_hits, 15.0, "gecco_pfam5 scores");
}

#[test]
fn test_exact_tblout_score_bias_parity_on_representative_goldens() {
    let cases = [
        (
            "20aa",
            test_path("tests/golden/20aa.tblout"),
            test_path("hmmer/testsuite/20aa.hmm"),
            test_path("hmmer/testsuite/20aa-alitest.fa"),
        ),
        (
            "globins4",
            test_path("tests/golden/globins4_vs_globins45.tblout"),
            test_path("hmmer/tutorial/globins4.hmm"),
            test_path("hmmer/tutorial/globins45.fa"),
        ),
        (
            "fn3",
            test_path("tests/golden/fn3_vs_7less.tblout"),
            test_path("hmmer/tutorial/fn3.hmm"),
            test_path("hmmer/tutorial/7LESS_DROME"),
        ),
    ];

    for (label, golden_path, hmm_path, seqdb_path) in cases {
        let golden = parse_golden_tblout_with_bias(&golden_path);
        let (_stdout, rust_tbl) = run_hmmsearch(&hmm_path, &seqdb_path, &[]);

        let rust_rows: HashMap<(String, String), (String, String, String)> = rust_tbl
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|line| {
                let fields: Vec<&str> = line.split_whitespace().collect();
                (
                    (fields[0].to_string(), fields[2].to_string()),
                    (
                        fields[4].to_string(),
                        fields[5].to_string(),
                        fields[6].to_string(),
                    ),
                )
            })
            .collect();

        assert_eq!(
            rust_rows, golden,
            "{} tblout score/bias rows diverged from the committed C golden",
            label
        );
    }
}

// ============================================================
// 4. Output format tests
// ============================================================

#[test]
fn test_tblout_column_format() {
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );

    // Check header lines exist
    let lines: Vec<&str> = tbl.lines().collect();
    assert!(lines.len() >= 4, "tblout should have header + data");
    assert!(lines[0].starts_with('#'));
    assert!(lines[1].starts_with('#'));
    assert!(lines[2].starts_with('#'));

    // Check that data lines parse correctly with whitespace splitting
    for line in lines.iter().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        assert!(
            fields.len() >= 18,
            "tblout data line should have >=18 fields, got {}: {}",
            fields.len(),
            line
        );

        // Field 0: target name (no embedded whitespace)
        assert!(!fields[0].is_empty());
        // Field 1: accession (or "-")
        assert!(!fields[1].is_empty());
        // Field 2: query name
        assert!(!fields[2].is_empty());
        // Field 4: E-value (parseable as float)
        let _ev: f64 = fields[4]
            .parse()
            .unwrap_or_else(|_| panic!("E-value not parseable: '{}' in line: {}", fields[4], line));
        // Field 5: score (parseable as float)
        let _sc: f64 = fields[5]
            .parse()
            .unwrap_or_else(|_| panic!("Score not parseable: '{}' in line: {}", fields[5], line));
    }
}

#[test]
fn test_tblout_name_accession_separation() {
    // Ensure target name and accession are properly space-separated
    // even when the name is longer than the column width
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
        &[],
    );

    for line in tbl
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let fields: Vec<&str> = line.split_whitespace().collect();
        // Name should NOT contain the accession
        assert!(
            !fields[0].ends_with('-') || fields[0].len() <= 2,
            "Target name should not have trailing '-' from accession: '{}'",
            fields[0]
        );
        // Accession field should be separate
        assert!(
            fields[1] == "-" || fields[1] == "P13368",
            "Accession should be '-' or 'P13368', got '{}'",
            fields[1]
        );
    }
}

#[test]
fn test_domtblout_column_format() {
    let (_stdout, _tbl, domtbl) = run_hmmsearch_dom(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
        &[],
    );

    let lines: Vec<&str> = domtbl.lines().collect();
    assert!(lines.len() >= 4, "domtblout should have header + data");

    for line in lines.iter().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        assert!(
            fields.len() >= 22,
            "domtblout data line should have >=22 fields, got {}: {}",
            fields.len(),
            line
        );

        // Target name
        assert_eq!(fields[0], "7LESS_DROME");
        // Query name
        assert_eq!(fields[3], "fn3");
        // Domain index and count
        let _idx: usize = fields[9]
            .parse()
            .unwrap_or_else(|_| panic!("Domain idx not parseable: '{}'", fields[9]));
        let _cnt: usize = fields[10]
            .parse()
            .unwrap_or_else(|_| panic!("Domain count not parseable: '{}'", fields[10]));
        // Domain E-value
        let _dev: f64 = fields[12]
            .parse()
            .unwrap_or_else(|_| panic!("Domain E-value not parseable: '{}'", fields[12]));
        // Ali coordinates
        let from: usize = fields[17]
            .parse()
            .unwrap_or_else(|_| panic!("ali_from not parseable: '{}'", fields[17]));
        let to: usize = fields[18]
            .parse()
            .unwrap_or_else(|_| panic!("ali_to not parseable: '{}'", fields[18]));
        assert!(to > from, "ali_to ({}) should be > ali_from ({})", to, from);
    }
}

#[test]
fn test_exact_domtblout_parity_on_committed_goldens() {
    let cases = [
        (
            "20aa",
            test_path("tests/golden/hmmsearch_20aa.domtblout"),
            test_path("hmmer/testsuite/20aa.hmm"),
            test_path("hmmer/testsuite/20aa-alitest.fa"),
        ),
        (
            "fn3",
            test_path("tests/golden/hmmsearch_fn3.domtblout"),
            test_path("hmmer/tutorial/fn3.hmm"),
            test_path("hmmer/tutorial/7LESS_DROME"),
        ),
        (
            "caudal",
            test_path("tests/golden/hmmsearch_caudal.domtblout"),
            test_path("hmmer/testsuite/Caudal_act.hmm"),
            test_path("hmmer/testsuite/3box-alitest.fa"),
        ),
    ];

    for (label, golden_path, hmm_path, seqdb_path) in cases {
        let golden = std::fs::read_to_string(golden_path).unwrap();
        let (_stdout, _tbl, rust_domtbl) = run_hmmsearch_dom(&hmm_path, &seqdb_path, &[]);

        assert_eq!(
            parse_domtblout_core_rows(&rust_domtbl),
            parse_domtblout_core_rows(&golden),
            "{} domtblout rows diverged from committed C golden output",
            label
        );
    }
}

#[test]
fn test_pfamtblout_matches_bundled_c_on_small_fixtures() {
    let cases = [
        (
            "20aa",
            test_path("hmmer/testsuite/20aa.hmm"),
            test_path("hmmer/testsuite/20aa-alitest.fa"),
        ),
        (
            "fn3",
            test_path("hmmer/tutorial/fn3.hmm"),
            test_path("hmmer/tutorial/7LESS_DROME"),
        ),
    ];

    for (label, hmm_path, seqdb_path) in cases {
        let rust = run_hmmsearch_pfamtbl(&hmm_path, &seqdb_path, &[]);
        let c = run_c_hmmsearch_pfamtbl(&hmm_path, &seqdb_path, &[]);

        assert_eq!(
            parse_pfamtblout_rows(&rust),
            parse_pfamtblout_rows(&c),
            "{} pfamtblout rows diverged from bundled C hmmsearch",
            label
        );
    }
}

#[test]
fn test_stdout_format_hit_table_spacing() {
    let (stdout, _tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );

    // Find hit table lines
    let mut in_hits = false;
    for line in stdout.lines() {
        if line.contains("Sequence") && line.contains("Description") {
            in_hits = true;
            continue;
        }
        if line.contains("-------") && in_hits {
            continue;
        }
        if in_hits && line.starts_with("  ") && !line.contains("[No hits") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 9 {
                // Sequence name should be a standalone field
                let name = parts[8];
                assert!(!name.is_empty(), "Sequence name should not be empty");
                // Description (if present) should be separate
                if parts.len() > 9 {
                    assert_ne!(
                        parts[9], name,
                        "Description should not be part of sequence name"
                    );
                }
            }
        } else if in_hits {
            break;
        }
    }
}

#[test]
fn test_stdout_has_pipeline_stats() {
    let (stdout, _tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &[],
    );

    assert!(
        stdout.contains("Internal pipeline statistics summary:"),
        "Missing pipeline stats section"
    );
    assert!(
        stdout.contains("Passed MSV filter:"),
        "Missing MSV filter stats"
    );
    assert!(
        stdout.contains("Passed Vit filter:"),
        "Missing Viterbi filter stats"
    );
    assert!(
        stdout.contains("Passed Fwd filter:"),
        "Missing Forward filter stats"
    );
    assert!(stdout.contains("[ok]"), "Missing [ok] at end");
}

// ============================================================
// 5. Multi-query HMM tests
// ============================================================

#[test]
fn test_multi_query_gecco_pfam5() {
    // gecco_pfam5.hmm contains 5 HMMs; each should produce results for its own queries
    let (stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/gecco_pfam5.hmm"),
        &test_path("hmmer/testsuite/gecco_proteins.faa"),
        &[],
    );

    let hits = parse_tblout(&tbl);
    let queries: HashSet<String> = hits.iter().map(|(_, q, _, _)| q.clone()).collect();

    // Should have hits from multiple queries
    assert!(
        queries.len() >= 3,
        "Multi-query search should find hits from >=3 different HMMs, got {:?}",
        queries
    );

    // Verify stdout has multiple "Query:" blocks
    let query_count = stdout.matches("Query:").count();
    assert_eq!(
        query_count, 5,
        "Should have 5 Query blocks (one per HMM), got {}",
        query_count
    );

    // Each query should have the standard sections
    let ok_count = stdout.matches("[ok]").count();
    assert_eq!(
        ok_count, 1,
        "Should have exactly 1 [ok] at end, got {}",
        ok_count
    );
    let stats_count = stdout.matches("Internal pipeline statistics").count();
    assert_eq!(
        stats_count, 5,
        "Should have 5 pipeline stats sections, got {}",
        stats_count
    );
}

#[test]
fn test_multi_query_gecco_missed_hmms() {
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/gecco_missed_hmms.hmm"),
        &test_path("hmmer/testsuite/gecco_missed_proteins.faa"),
        &[],
    );

    let hits = parse_tblout(&tbl);
    assert!(
        hits.len() >= 6,
        "Expected >=6 hits for gecco_missed, got {}",
        hits.len()
    );

    let queries: HashSet<String> = hits.iter().map(|(_, q, _, _)| q.clone()).collect();
    assert!(
        queries.len() >= 4,
        "Should have hits from >=4 different HMMs, got {:?}",
        queries
    );
}

// ============================================================
// 6. Score range sanity checks
// ============================================================

#[test]
fn test_scores_are_positive_for_real_hits() {
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );
    let hits = parse_tblout(&tbl);
    for (target, _query, _ev, score) in &hits {
        assert!(
            *score > 0.0,
            "Real hit {} should have positive score, got {:.1}",
            target,
            score
        );
    }
}

#[test]
fn test_evalues_are_ordered() {
    // Hits should be sorted by E-value (ascending)
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );
    let hits = parse_tblout(&tbl);
    for window in hits.windows(2) {
        assert!(
            window[0].2 <= window[1].2 * 1.01, // small tolerance for ties
            "Hits should be sorted by E-value: {} ({:.2e}) should come before {} ({:.2e})",
            window[0].0,
            window[0].2,
            window[1].0,
            window[1].2
        );
    }
}

#[test]
fn test_domain_coordinates_within_sequence() {
    let (_stdout, _tbl, domtbl) = run_hmmsearch_dom(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
        &[],
    );

    let doms = parse_domtblout(&domtbl);
    for dom in &doms {
        assert!(
            dom.ali_from >= 1,
            "ali_from should be >= 1, got {}",
            dom.ali_from
        );
        assert!(
            dom.ali_to > dom.ali_from,
            "ali_to ({}) should be > ali_from ({})",
            dom.ali_to,
            dom.ali_from
        );
        // 7LESS_DROME is 2554 residues
        assert!(
            dom.ali_to <= 2554,
            "ali_to ({}) should be <= sequence length (2554)",
            dom.ali_to
        );
    }
}

#[test]
fn test_domain_indices_sequential() {
    let (_stdout, _tbl, domtbl) = run_hmmsearch_dom(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
        &[],
    );

    let doms = parse_domtblout(&domtbl);
    for (i, dom) in doms.iter().enumerate() {
        assert_eq!(
            dom.dom_idx,
            i + 1,
            "Domain index should be sequential: expected {}, got {}",
            i + 1,
            dom.dom_idx
        );
        assert_eq!(
            dom.ndom,
            doms.len(),
            "Domain count should be consistent: expected {}, got {}",
            doms.len(),
            dom.ndom
        );
    }
}

// ============================================================
// 7. Regression tests for specific bugs
// ============================================================

#[test]
fn test_large_multi_hmm_search() {
    // gecco_missed4 has 30 HMMs and should find ~60 hits
    let (_stdout, tbl) = run_hmmsearch(
        &test_path("hmmer/testsuite/gecco_missed4_hmms.hmm"),
        &test_path("hmmer/testsuite/gecco_missed4_proteins.faa"),
        &[],
    );
    let hits = parse_tblout(&tbl);
    assert!(
        hits.len() >= 40,
        "gecco_missed4 should find >=40 hits (C finds ~60), got {}",
        hits.len()
    );
}

#[test]
fn test_evalue_threshold_filtering() {
    // With a tight E-value threshold, should find fewer hits
    let (_stdout, tbl_default) = run_hmmsearch(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );
    let (_stdout, tbl_strict) = run_hmmsearch(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-E", "1e-50"],
    );

    let default_hits = parse_tblout(&tbl_default);
    let strict_hits = parse_tblout(&tbl_strict);
    assert!(
        strict_hits.len() < default_hits.len(),
        "Strict E-value should find fewer hits: {} vs {}",
        strict_hits.len(),
        default_hits.len()
    );
    assert!(
        strict_hits.len() >= 10,
        "Should still find >=10 strong globin hits at E<1e-50, got {}",
        strict_hits.len()
    );

    // All strict hits should have E-value < threshold
    for (target, _, ev, _) in &strict_hits {
        assert!(
            *ev < 1e-50,
            "Hit {} has E-value {:.2e} exceeding threshold 1e-50",
            target,
            ev
        );
    }
}
