//! Golden-file integration tests.
//! Compare Rust binary output against reference outputs from the C HMMER binaries.
//! Lines that vary between runs (timing, file paths) are stripped before comparison.

use std::process::Command;

/// Get path to the built binary
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

/// Strip lines that vary between runs from HMMER output
fn strip_variable_lines(output: &str) -> String {
    let variable_prefixes = [
        "# hmmsearch",   // program name line varies by binary name
        "# CPU time:",
        "# Mc/sec:",
        "# Elapsed:",
        "# query HMM file:",
        "# target sequence database:",
        "# query sequence file:",
        "# per-seq hits tabular output:",
        "# per-dom hits tabular output:",
        "# output directed to file:",
        "# MSA of hits saved to file:",
        "# input query is asserted as:",
        // tblout/domtblout footer comments
        "# Program:",
        "# Version:",
        "# Pipeline mode:",
        "# Query file:",
        "# Target file:",
        "# Option settings:",
        "# Current dir:",
        "# Date:",
    ];
    output
        .lines()
        .filter(|line| {
            !variable_prefixes.iter().any(|prefix| line.starts_with(prefix))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Run a command with a temp file for tblout/domtblout, return combined stdout+stderr
fn run_hmmsearch(args: &[&str]) -> String {
    let path = binary_path("hmmsearch_ffi");
    let output = Command::new(&path)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", path.display(), e));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    format!("{}{}", stdout, stderr)
}

/// Run hmmsearch with --tblout to a temp file, return the tblout content
fn run_hmmsearch_tblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout_path = dir.path().join("tblout.txt");
    let path = binary_path("hmmsearch_ffi");
    let output = Command::new(&path)
        .args(&[
            "--tblout",
            tblout_path.to_str().unwrap(),
            hmm,
            seqdb,
        ])
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", path.display(), e));
    assert!(output.status.success(), "hmmsearch failed");
    std::fs::read_to_string(&tblout_path).unwrap()
}

/// Run hmmsearch with --domtblout to a temp file, return the domtblout content
fn run_hmmsearch_domtblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout_path = dir.path().join("domtblout.txt");
    let path = binary_path("hmmsearch_ffi");
    let output = Command::new(&path)
        .args(&[
            "--domtblout",
            domtblout_path.to_str().unwrap(),
            hmm,
            seqdb,
        ])
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", path.display(), e));
    assert!(output.status.success(), "hmmsearch failed");
    std::fs::read_to_string(&domtblout_path).unwrap()
}

/// Compare output against golden file, stripping variable lines
fn assert_golden(actual: &str, golden_path: &str) {
    let golden = std::fs::read_to_string(golden_path)
        .unwrap_or_else(|e| panic!("Failed to read golden file {}: {}", golden_path, e));
    let actual_stripped = strip_variable_lines(actual);
    let golden_stripped = strip_variable_lines(&golden);

    if actual_stripped != golden_stripped {
        let actual_lines: Vec<&str> = actual_stripped.lines().collect();
        let golden_lines: Vec<&str> = golden_stripped.lines().collect();
        for (i, (a, g)) in actual_lines.iter().zip(golden_lines.iter()).enumerate() {
            if a != g {
                panic!(
                    "Golden file mismatch at line {} in {}:\n  expected: {:?}\n  actual:   {:?}",
                    i + 1,
                    golden_path,
                    g,
                    a
                );
            }
        }
        if actual_lines.len() != golden_lines.len() {
            panic!(
                "Golden file line count mismatch in {}: expected {} lines, got {}",
                golden_path,
                golden_lines.len(),
                actual_lines.len()
            );
        }
    }
}

fn project_root() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
}

fn test_path(relative: &str) -> String {
    format!("{}/{}", project_root(), relative)
}

// ========== hmmsearch stdout tests ==========

#[test]
fn hmmsearch_20aa_stdout() {
    let output = run_hmmsearch(&[
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
    ]);
    assert_golden(&output, &test_path("tests/golden/hmmsearch_20aa.stdout"));
}

#[test]
fn hmmsearch_caudal_stdout() {
    let output = run_hmmsearch(&[
        &test_path("hmmer/testsuite/Caudal_act.hmm"),
        &test_path("hmmer/testsuite/3box-alitest.fa"),
    ]);
    assert_golden(&output, &test_path("tests/golden/hmmsearch_caudal.stdout"));
}

#[test]
fn hmmsearch_fn3_stdout() {
    let output = run_hmmsearch(&[
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
    ]);
    assert_golden(&output, &test_path("tests/golden/hmmsearch_fn3.stdout"));
}

// ========== hmmsearch tblout tests ==========

#[test]
fn hmmsearch_20aa_tblout() {
    let output = run_hmmsearch_tblout(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
    );
    assert_golden(&output, &test_path("tests/golden/hmmsearch_20aa.tblout"));
}

#[test]
fn hmmsearch_fn3_tblout() {
    let output = run_hmmsearch_tblout(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
    );
    assert_golden(&output, &test_path("tests/golden/hmmsearch_fn3.tblout"));
}

// ========== hmmsearch domtblout tests ==========

#[test]
fn hmmsearch_20aa_domtblout() {
    let output = run_hmmsearch_domtblout(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
    );
    assert_golden(&output, &test_path("tests/golden/hmmsearch_20aa.domtblout"));
}

#[test]
fn hmmsearch_fn3_domtblout() {
    let output = run_hmmsearch_domtblout(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
    );
    assert_golden(&output, &test_path("tests/golden/hmmsearch_fn3.domtblout"));
}
