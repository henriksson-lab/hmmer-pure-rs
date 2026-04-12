//! Integration tests for the pure Rust hmmsearch binary.
//! Validates that the Rust implementation finds the same hits as the C version.

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

/// Extract hit names from hmmsearch stdout output
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

#[test]
fn rust_hmmsearch_finds_20aa_hits() {
    let output = Command::new(binary_path("hmmsearch"))
        .args(&[
            &test_path("hmmer/testsuite/20aa.hmm"),
            &test_path("hmmer/testsuite/20aa-alitest.fa"),
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let hits = extract_hit_names(&stdout);
    let mut sorted_hits = hits.clone();
    sorted_hits.sort();
    assert_eq!(sorted_hits, vec!["test1", "test2", "test3", "test4"],
        "Expected 4 hits (any order), got: {:?}", hits);
}

#[test]
fn rust_hmmsearch_finds_globin_hits() {
    let output = Command::new(binary_path("hmmsearch"))
        .args(&[
            &test_path("hmmer/tutorial/globins4.hmm"),
            &test_path("hmmer/tutorial/globins45.fa"),
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let hits = extract_hit_names(&stdout);
    assert!(hits.len() >= 40, "Expected >=40 globin hits, got {}", hits.len());
    // Top hit should be a known globin
    assert!(hits[0].contains("MYG") || hits[0].contains("HB"),
        "Top hit should be a myoglobin or hemoglobin, got: {}", hits[0]);
}

#[test]
fn rust_hmmsearch_tblout_works() {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(binary_path("hmmsearch"))
        .args(&[
            "--tblout", tblout.to_str().unwrap(),
            &test_path("hmmer/testsuite/20aa.hmm"),
            &test_path("hmmer/testsuite/20aa-alitest.fa"),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let tblout_content = std::fs::read_to_string(&tblout).unwrap();
    let data_lines: Vec<&str> = tblout_content.lines()
        .filter(|l| !l.starts_with('#'))
        .collect();
    assert_eq!(data_lines.len(), 4, "Expected 4 hits in tblout");
}

#[test]
fn rust_hmmsearch_no_hits_for_unrelated() {
    let output = Command::new(binary_path("hmmsearch"))
        .args(&[
            &test_path("hmmer/testsuite/Caudal_act.hmm"),
            &test_path("hmmer/testsuite/20aa-alitest.fa"),
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("No hits detected") || extract_hit_names(&stdout).is_empty(),
        "Should find no hits for unrelated HMM/sequences");
}

#[test]
fn rust_hmmsearch_max_flag_works() {
    let output = Command::new(binary_path("hmmsearch"))
        .args(&[
            "--max",
            &test_path("hmmer/testsuite/20aa.hmm"),
            &test_path("hmmer/testsuite/20aa-alitest.fa"),
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let hits = extract_hit_names(&stdout);
    assert_eq!(hits.len(), 4, "With --max, should find all 4 hits");
}
