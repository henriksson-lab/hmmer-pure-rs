//! Real-world regression tests that exercise end-to-end CLI behavior on
//! realistic fixtures instead of synthetic toy cases.

use std::collections::{BTreeMap, HashSet};
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

fn test_path(relative: &str) -> String {
    format!("{}/{}", env!("CARGO_MANIFEST_DIR"), relative)
}

fn run_hmmsearch_tblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "search",
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
            hmm,
            seqdb,
        ])
        .output()
        .expect("failed to run hmmer search");
    assert!(
        output.status.success(),
        "hmmer search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_nhmmer_tblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args(["nhmmer", "--tblout", tblout.to_str().unwrap(), hmm, seqdb])
        .output()
        .expect("failed to run hmmer nhmmer");
    assert!(
        output.status.success(),
        "hmmer nhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_nhmmer_dfamtblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let dfamtblout = dir.path().join("dfamtblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "nhmmer",
            "--dfamtblout",
            dfamtblout.to_str().unwrap(),
            hmm,
            seqdb,
        ])
        .output()
        .expect("failed to run hmmer nhmmer");
    assert!(
        output.status.success(),
        "hmmer nhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(dfamtblout).unwrap()
}

fn run_c_nhmmer_dfamtblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let dfamtblout = dir.path().join("dfamtblout.txt");
    let output = Command::new(test_path("hmmer/src/nhmmer"))
        .args(["--dfamtblout", dfamtblout.to_str().unwrap(), hmm, seqdb])
        .output()
        .expect("failed to run bundled C nhmmer");
    assert!(
        output.status.success(),
        "bundled C nhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(dfamtblout).unwrap()
}

fn run_c_nhmmer_tblout_with_args(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let mut args = extra_args.to_vec();
    args.extend_from_slice(&["--tblout", tblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(test_path("hmmer/src/nhmmer"))
        .args(&args)
        .output()
        .expect("failed to run bundled C nhmmer");
    assert!(
        output.status.success(),
        "bundled C nhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_c_nhmmer_with_args(hmm: &str, seqdb: &str, extra_args: &[&str]) -> (String, String) {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let mut args = extra_args.to_vec();
    args.extend_from_slice(&["--tblout", tblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(test_path("hmmer/src/nhmmer"))
        .args(&args)
        .output()
        .expect("failed to run bundled C nhmmer");
    assert!(
        output.status.success(),
        "bundled C nhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    (
        String::from_utf8(output.stdout).unwrap(),
        std::fs::read_to_string(tblout).unwrap(),
    )
}

fn run_nhmmer(hmm: &str, seqdb: &str, extra_args: &[&str]) -> (String, String) {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let mut args = vec!["nhmmer"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--tblout", tblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(binary_path("hmmer"))
        .args(&args)
        .output()
        .expect("failed to run hmmer nhmmer");
    assert!(
        output.status.success(),
        "hmmer nhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        std::fs::read_to_string(tblout).unwrap(),
    )
}

fn run_nhmmscan_tblout(hmmdb: &str, seqfile: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("models.hmm");
    std::fs::copy(hmmdb, &hmm_copy).unwrap();
    let press = Command::new(binary_path("hmmer"))
        .args(["press", "-f", hmm_copy.to_str().unwrap()])
        .output()
        .expect("failed to run hmmer press");
    assert!(
        press.status.success(),
        "hmmer press failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "nhmmscan",
            "--tblout",
            tblout.to_str().unwrap(),
            hmm_copy.to_str().unwrap(),
            seqfile,
        ])
        .output()
        .expect("failed to run hmmer nhmmscan");
    assert!(
        output.status.success(),
        "hmmer nhmmscan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_nhmmscan_dfamtblout(hmmdb: &str, seqfile: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("models.hmm");
    std::fs::copy(hmmdb, &hmm_copy).unwrap();
    let press = Command::new(binary_path("hmmer"))
        .args(["press", "-f", hmm_copy.to_str().unwrap()])
        .output()
        .expect("failed to run hmmer press");
    assert!(
        press.status.success(),
        "hmmer press failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );
    let dfamtblout = dir.path().join("dfamtblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "nhmmscan",
            "--dfamtblout",
            dfamtblout.to_str().unwrap(),
            hmm_copy.to_str().unwrap(),
            seqfile,
        ])
        .output()
        .expect("failed to run hmmer nhmmscan");
    assert!(
        output.status.success(),
        "hmmer nhmmscan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(dfamtblout).unwrap()
}

fn run_c_nhmmscan_tblout(hmmdb: &str, seqfile: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("models.hmm");
    std::fs::copy(hmmdb, &hmm_copy).unwrap();

    let press = Command::new(test_path("hmmer/src/hmmpress"))
        .args(["-f", hmm_copy.to_str().unwrap()])
        .output()
        .expect("failed to run bundled C hmmpress");
    assert!(
        press.status.success(),
        "bundled C hmmpress failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );

    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(test_path("hmmer/src/nhmmscan"))
        .args([
            "--tblout",
            tblout.to_str().unwrap(),
            hmm_copy.to_str().unwrap(),
            seqfile,
        ])
        .output()
        .expect("failed to run bundled C nhmmscan");
    assert!(
        output.status.success(),
        "bundled C nhmmscan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_c_nhmmscan_dfamtblout(hmmdb: &str, seqfile: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("models.hmm");
    std::fs::copy(hmmdb, &hmm_copy).unwrap();

    let press = Command::new(test_path("hmmer/src/hmmpress"))
        .args(["-f", hmm_copy.to_str().unwrap()])
        .output()
        .expect("failed to run bundled C hmmpress");
    assert!(
        press.status.success(),
        "bundled C hmmpress failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );

    let dfamtblout = dir.path().join("dfamtblout.txt");
    let output = Command::new(test_path("hmmer/src/nhmmscan"))
        .args([
            "--dfamtblout",
            dfamtblout.to_str().unwrap(),
            hmm_copy.to_str().unwrap(),
            seqfile,
        ])
        .output()
        .expect("failed to run bundled C nhmmscan");
    assert!(
        output.status.success(),
        "bundled C nhmmscan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(dfamtblout).unwrap()
}

fn run_hmmscan_tblout(hmmdb: &str, seqfile: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("models.hmm");
    std::fs::copy(hmmdb, &hmm_copy).unwrap();
    let press = Command::new(binary_path("hmmer"))
        .args(["press", "-f", hmm_copy.to_str().unwrap()])
        .output()
        .expect("failed to run hmmer press");
    assert!(
        press.status.success(),
        "hmmer press failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "scan",
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
            hmm_copy.to_str().unwrap(),
            seqfile,
        ])
        .output()
        .expect("failed to run hmmer scan");
    assert!(
        output.status.success(),
        "hmmer scan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_c_hmmscan_tblout(hmmdb: &str, seqfile: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("models.hmm");
    std::fs::copy(hmmdb, &hmm_copy).unwrap();
    let press = Command::new(test_path("hmmer/src/hmmpress"))
        .args(["-f", hmm_copy.to_str().unwrap()])
        .output()
        .expect("failed to run bundled C hmmpress");
    assert!(
        press.status.success(),
        "bundled C hmmpress failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(test_path("hmmer/src/hmmscan"))
        .args([
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
            hmm_copy.to_str().unwrap(),
            seqfile,
        ])
        .output()
        .expect("failed to run bundled C hmmscan");
    assert!(
        output.status.success(),
        "bundled C hmmscan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_hmmscan_domtblout(hmmdb: &str, seqfile: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("models.hmm");
    std::fs::copy(hmmdb, &hmm_copy).unwrap();
    let press = Command::new(binary_path("hmmer"))
        .args(["press", "-f", hmm_copy.to_str().unwrap()])
        .output()
        .expect("failed to run hmmer press");
    assert!(
        press.status.success(),
        "hmmer press failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );
    let domtblout = dir.path().join("domtblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "scan",
            "--noali",
            "--domtblout",
            domtblout.to_str().unwrap(),
            hmm_copy.to_str().unwrap(),
            seqfile,
        ])
        .output()
        .expect("failed to run hmmer scan");
    assert!(
        output.status.success(),
        "hmmer scan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(domtblout).unwrap()
}

fn run_c_hmmscan_domtblout(hmmdb: &str, seqfile: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("models.hmm");
    std::fs::copy(hmmdb, &hmm_copy).unwrap();
    let press = Command::new(test_path("hmmer/src/hmmpress"))
        .args(["-f", hmm_copy.to_str().unwrap()])
        .output()
        .expect("failed to run bundled C hmmpress");
    assert!(
        press.status.success(),
        "bundled C hmmpress failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );
    let domtblout = dir.path().join("domtblout.txt");
    let output = Command::new(test_path("hmmer/src/hmmscan"))
        .args([
            "--noali",
            "--domtblout",
            domtblout.to_str().unwrap(),
            hmm_copy.to_str().unwrap(),
            seqfile,
        ])
        .output()
        .expect("failed to run bundled C hmmscan");
    assert!(
        output.status.success(),
        "bundled C hmmscan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(domtblout).unwrap()
}

fn run_hmmscan_pfamtblout(hmmdb: &str, seqfile: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("models.hmm");
    std::fs::copy(hmmdb, &hmm_copy).unwrap();
    let press = Command::new(binary_path("hmmer"))
        .args(["press", "-f", hmm_copy.to_str().unwrap()])
        .output()
        .expect("failed to run hmmer press");
    assert!(
        press.status.success(),
        "hmmer press failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );
    let pfamtblout = dir.path().join("pfamtblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "scan",
            "--noali",
            "--pfamtblout",
            pfamtblout.to_str().unwrap(),
            hmm_copy.to_str().unwrap(),
            seqfile,
        ])
        .output()
        .expect("failed to run hmmer scan");
    assert!(
        output.status.success(),
        "hmmer scan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(pfamtblout).unwrap()
}

fn run_c_hmmscan_pfamtblout(hmmdb: &str, seqfile: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("models.hmm");
    std::fs::copy(hmmdb, &hmm_copy).unwrap();
    let press = Command::new(test_path("hmmer/src/hmmpress"))
        .args(["-f", hmm_copy.to_str().unwrap()])
        .output()
        .expect("failed to run bundled C hmmpress");
    assert!(
        press.status.success(),
        "bundled C hmmpress failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );
    let pfamtblout = dir.path().join("pfamtblout.txt");
    let output = Command::new(test_path("hmmer/src/hmmscan"))
        .args([
            "--noali",
            "--pfamtblout",
            pfamtblout.to_str().unwrap(),
            hmm_copy.to_str().unwrap(),
            seqfile,
        ])
        .output()
        .expect("failed to run bundled C hmmscan");
    assert!(
        output.status.success(),
        "bundled C hmmscan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(pfamtblout).unwrap()
}

fn run_nhmmer_stdout(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let mut args = vec!["nhmmer"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[hmm, seqdb]);
    let output = Command::new(binary_path("hmmer"))
        .args(&args)
        .output()
        .expect("failed to run hmmer nhmmer");
    assert!(
        output.status.success(),
        "hmmer nhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn run_hmmsearch_domtblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("domtblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "search",
            "--noali",
            "--domtblout",
            domtblout.to_str().unwrap(),
            hmm,
            seqdb,
        ])
        .output()
        .expect("failed to run hmmer search");
    assert!(
        output.status.success(),
        "hmmer search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(domtblout).unwrap()
}

fn run_c_hmmsearch_domtblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("domtblout.txt");
    let output = Command::new(test_path("hmmer/src/hmmsearch"))
        .args([
            "--noali",
            "--domtblout",
            domtblout.to_str().unwrap(),
            hmm,
            seqdb,
        ])
        .output()
        .expect("failed to run bundled C hmmsearch");
    assert!(
        output.status.success(),
        "bundled C hmmsearch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(domtblout).unwrap()
}

fn run_hmmsearch_pfamtblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let pfamtblout = dir.path().join("pfamtblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "search",
            "--noali",
            "--pfamtblout",
            pfamtblout.to_str().unwrap(),
            hmm,
            seqdb,
        ])
        .output()
        .expect("failed to run hmmer search");
    assert!(
        output.status.success(),
        "hmmer search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(pfamtblout).unwrap()
}

fn run_c_hmmsearch_pfamtblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let pfamtblout = dir.path().join("pfamtblout.txt");
    let output = Command::new(test_path("hmmer/src/hmmsearch"))
        .args([
            "--noali",
            "--pfamtblout",
            pfamtblout.to_str().unwrap(),
            hmm,
            seqdb,
        ])
        .output()
        .expect("failed to run bundled C hmmsearch");
    assert!(
        output.status.success(),
        "bundled C hmmsearch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(pfamtblout).unwrap()
}

fn parse_hmmsearch_rows(content: &str) -> Vec<(String, String, f64, f64)> {
    content
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 6 {
                return None;
            }
            Some((
                fields[0].to_string(),
                fields[2].to_string(),
                fields[4].parse().ok()?,
                fields[5].parse().ok()?,
            ))
        })
        .collect()
}

fn parse_hmmsearch_score_bias_rows(content: &str) -> Vec<(String, String, String, String, String)> {
    content
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 7 {
                return None;
            }
            Some((
                fields[0].to_string(),
                fields[2].to_string(),
                fields[4].to_string(),
                fields[5].to_string(),
                fields[6].to_string(),
            ))
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DomtblRow {
    target: String,
    query: String,
    dom_idx: usize,
    dom_count: usize,
    score: String,
    hmm_from: usize,
    hmm_to: usize,
    ali_from: usize,
    ali_to: usize,
    env_from: usize,
    env_to: usize,
}

fn parse_domtbl_rows(content: &str) -> Vec<DomtblRow> {
    content
        .lines()
        .filter_map(|line| {
            if line.starts_with('#') || line.trim().is_empty() {
                return None;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 21 {
                return None;
            }
            Some(DomtblRow {
                target: fields[0].to_string(),
                query: fields[3].to_string(),
                dom_idx: fields[9].parse().ok()?,
                dom_count: fields[10].parse().ok()?,
                score: fields[13].to_string(),
                hmm_from: fields[15].parse().ok()?,
                hmm_to: fields[16].parse().ok()?,
                ali_from: fields[17].parse().ok()?,
                ali_to: fields[18].parse().ok()?,
                env_from: fields[19].parse().ok()?,
                env_to: fields[20].parse().ok()?,
            })
        })
        .collect()
}

fn parse_pfamtbl_rows(content: &str) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct NhmmerRow {
    target: String,
    query: String,
    hmm_from: usize,
    hmm_to: usize,
    ali_from: usize,
    ali_to: usize,
    env_from: usize,
    env_to: usize,
    sq_len: usize,
    strand: String,
    evalue: String,
    score: String,
    bias: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DfamRow {
    target: String,
    acc: String,
    query: String,
    score: String,
    evalue: String,
    bias: String,
    hmm_from: usize,
    hmm_to: usize,
    strand: String,
    ali_from: usize,
    ali_to: usize,
    env_from: usize,
    env_to: usize,
    sq_len: usize,
}

fn parse_dfamtbl_rows(content: &str) -> Vec<DfamRow> {
    content
        .lines()
        .filter_map(|line| {
            if line.starts_with('#') || line.trim().is_empty() {
                return None;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 14 || (fields[8] != "+" && fields[8] != "-") {
                return None;
            }
            Some(DfamRow {
                target: fields[0].to_string(),
                acc: fields[1].to_string(),
                query: fields[2].to_string(),
                score: fields[3].to_string(),
                evalue: fields[4].to_string(),
                bias: fields[5].to_string(),
                hmm_from: fields[6].parse().ok()?,
                hmm_to: fields[7].parse().ok()?,
                strand: fields[8].to_string(),
                ali_from: fields[9].parse().ok()?,
                ali_to: fields[10].parse().ok()?,
                env_from: fields[11].parse().ok()?,
                env_to: fields[12].parse().ok()?,
                sq_len: fields[13].parse().ok()?,
            })
        })
        .collect()
}

fn parse_nhmmer_rows(content: &str) -> Vec<NhmmerRow> {
    content
        .lines()
        .filter_map(|line| {
            if line.starts_with('#') || line.trim().is_empty() {
                return None;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 16 || (fields[11] != "+" && fields[11] != "-") {
                return None;
            }
            Some(NhmmerRow {
                target: fields[0].to_string(),
                query: fields[2].to_string(),
                hmm_from: fields[4].parse().ok()?,
                hmm_to: fields[5].parse().ok()?,
                ali_from: fields[6].parse().ok()?,
                ali_to: fields[7].parse().ok()?,
                env_from: fields[8].parse().ok()?,
                env_to: fields[9].parse().ok()?,
                sq_len: fields[10].parse().ok()?,
                strand: fields[11].to_string(),
                evalue: fields[12].to_string(),
                score: fields[13].to_string(),
                bias: fields[14].to_string(),
            })
        })
        .collect()
}

fn query_hit_counts(rows: &[(String, String, f64, f64)]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, query, _, _) in rows {
        *counts.entry(query.clone()).or_insert(0) += 1;
    }
    counts
}

fn normalize_nhmmer_stdout(stdout: &str) -> Vec<String> {
    let root_prefix = format!("{}/", env!("CARGO_MANIFEST_DIR"));
    stdout
        .lines()
        .filter(|line| {
            !line.starts_with("# CPU time:")
                && !line.starts_with("# Mc/sec:")
                && !line.starts_with("# Current dir:")
                && !line.starts_with("# Date:")
                && !line.starts_with("# hits tabular output:")
        })
        .map(|line| line.replace(&root_prefix, ""))
        .collect()
}

fn normalize_nhmmer_tblout_with_fixture(content: &str, hmm: &str, target: &str) -> Vec<String> {
    let root_prefix = format!("{}/", env!("CARGO_MANIFEST_DIR"));
    let option_line = format!(
        "# Option settings: hmmer nhmmer --dna --tblout /tmp/TMPFILE {} {} ",
        hmm, target
    );
    content
        .lines()
        .filter(|line| !line.starts_with("# Current dir:") && !line.starts_with("# Date:"))
        .map(|line| {
            let line = line.replace(&root_prefix, "");
            if line.starts_with("# Option settings:") {
                option_line.clone()
            } else {
                line
            }
        })
        .collect()
}

const PFAM_FAMILIES: [&str; 18] = [
    "Globin",
    "Trypsin",
    "Ras",
    "GTP_EFTU",
    "Pkinase",
    "RRM_1",
    "AAA",
    "7tm_1",
    "ABC_tran",
    "Ank",
    "WD40",
    "ig",
    "zf_C2H2",
    "Homeodomain",
    "Pkinase_Tyr",
    "RVT_1",
    "adh_short",
    "Sugar_tr",
];

#[test]
fn test_nhmmer_made1_tblout_matches_golden_rows() {
    let golden = std::fs::read_to_string(test_path("tests/golden/nhmmer_made1.tblout")).unwrap();
    let rust = run_nhmmer_tblout(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
    );

    let golden_rows = parse_nhmmer_rows(&golden);
    let rust_rows = parse_nhmmer_rows(&rust);

    assert_eq!(
        rust_rows, golden_rows,
        "nhmmer MADE1 rows diverged from golden output"
    );
}

#[test]
fn test_nhmmer_made1_dfamtblout_is_written() {
    let golden = std::fs::read_to_string(test_path("tests/golden/nhmmer_made1.tblout")).unwrap();
    let rust = run_nhmmer_dfamtblout(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
    );

    assert!(rust.contains("# hit scores"));
    assert!(rust.contains("sq-len"));
    assert!(!rust.contains("modlen"));
    let dfam_rows = parse_dfamtbl_rows(&rust);
    let tblout_rows = parse_nhmmer_rows(&golden);
    assert_eq!(dfam_rows.len(), tblout_rows.len());
    for (dfam, tblout) in dfam_rows.iter().zip(tblout_rows.iter()) {
        assert_eq!(dfam.target, tblout.target);
        assert_eq!(dfam.acc, "DF0000629.2");
        assert_eq!(dfam.query, tblout.query);
        assert_eq!(dfam.score, tblout.score);
        assert_eq!(dfam.evalue, tblout.evalue);
        assert_eq!(dfam.bias, tblout.bias);
        assert_eq!(
            (dfam.hmm_from, dfam.hmm_to),
            (tblout.hmm_from, tblout.hmm_to)
        );
        assert_eq!(
            (dfam.ali_from, dfam.ali_to),
            (tblout.ali_from, tblout.ali_to)
        );
        assert_eq!(
            (dfam.env_from, dfam.env_to),
            (tblout.env_from, tblout.env_to)
        );
        assert_eq!(dfam.strand, tblout.strand);
        assert_eq!(dfam.sq_len, tblout.sq_len);
    }
}

#[test]
fn test_nhmmer_made1_dfamtblout_matches_bundled_c_rows() {
    let hmm = test_path("hmmer/tutorial/MADE1.hmm");
    let seqdb = test_path("hmmer/tutorial/dna_target.fa");
    let rust_rows = parse_dfamtbl_rows(&run_nhmmer_dfamtblout(&hmm, &seqdb));
    let c_rows = parse_dfamtbl_rows(&run_c_nhmmer_dfamtblout(&hmm, &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "nhmmer MADE1 dfamtblout rows diverged from bundled C output"
    );
}

#[test]
fn test_nhmmer_made1_preserves_strand_and_coordinate_conventions() {
    let rust = run_nhmmer_tblout(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
    );
    let rows = parse_nhmmer_rows(&rust);

    assert_eq!(rows.len(), 5, "MADE1 fixture should produce 5 nhmmer hits");

    let plus = rows.iter().filter(|row| row.strand == "+").count();
    let minus = rows.iter().filter(|row| row.strand == "-").count();
    assert_eq!(plus, 3, "expected 3 plus-strand hits");
    assert_eq!(minus, 2, "expected 2 minus-strand hits");

    for row in &rows {
        if row.strand == "+" {
            assert!(
                row.ali_from <= row.ali_to && row.env_from <= row.env_to,
                "plus-strand hit should use ascending coordinates: {:?}",
                row
            );
        } else {
            assert!(
                row.ali_from >= row.ali_to && row.env_from >= row.env_to,
                "minus-strand hit should use descending coordinates: {:?}",
                row
            );
        }
    }

    assert_eq!(rows[0].target, "humanchr1_frag");
    assert_eq!(rows[0].query, "MADE1");
    assert_eq!(rows[0].ali_from, 302390);
    assert_eq!(rows[0].ali_to, 302466);
    assert_eq!(rows[0].strand, "+");
}

#[test]
fn test_nhmmer_3box_preserves_c_longtarget_minimum_alignment_span() {
    let hmm = test_path("hmmer/testsuite/3box.hmm");
    let seqdb = test_path("hmmer/testsuite/3box-alitest.fa");
    let rust_rows = parse_nhmmer_rows(&run_nhmmer(&hmm, &seqdb, &["--dna", "--max", "-T=-100"]).1);
    let c_rows = parse_nhmmer_rows(&run_c_nhmmer_tblout_with_args(
        &hmm,
        &seqdb,
        &["--dna", "--max", "-T", "-100"],
    ));

    let span = |row: &NhmmerRow| row.ali_from.abs_diff(row.ali_to) + 1;
    let rust_min = rust_rows.iter().map(span).min();
    let c_min = c_rows.iter().map(span).min();

    assert_eq!(c_min, Some(8), "bundled C fixture should exercise span 8");
    assert_eq!(rust_min, Some(8), "Rust should preserve C's span 8 floor");
    assert!(
        rust_rows.iter().all(|row| span(row) >= 8),
        "Rust nhmmer emitted a long-target alignment shorter than C's span floor: {rust_rows:?}"
    );
}

#[test]
fn test_nhmmer_made1_stdout_matches_golden_after_normalization() {
    let golden = std::fs::read_to_string(test_path("tests/golden/nhmmer_made1.stdout")).unwrap();
    let stdout = run_nhmmer_stdout(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &[],
    );

    assert_eq!(
        normalize_nhmmer_stdout(&stdout),
        normalize_nhmmer_stdout(&golden),
        "nhmmer MADE1 stdout diverged from golden output after stripping volatile footer lines"
    );
}

#[test]
fn test_nhmmer_made1_watson_and_crick_split_hits_cleanly() {
    let (_both_stdout, both_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &[],
    );
    let (_watson_stdout, watson_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &["--watson"],
    );
    let (_crick_stdout, crick_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &["--crick"],
    );

    let both_rows = parse_nhmmer_rows(&both_tbl);
    let watson_rows = parse_nhmmer_rows(&watson_tbl);
    let crick_rows = parse_nhmmer_rows(&crick_tbl);

    assert_eq!(watson_rows.len(), 3, "expected 3 Watson-strand hits");
    assert_eq!(crick_rows.len(), 2, "expected 2 Crick-strand hits");
    assert!(watson_rows.iter().all(|row| row.strand == "+"));
    assert!(crick_rows.iter().all(|row| row.strand == "-"));

    let watson_set: HashSet<(usize, usize, String)> = watson_rows
        .iter()
        .map(|row| (row.ali_from, row.ali_to, row.strand.clone()))
        .collect();
    let crick_set: HashSet<(usize, usize, String)> = crick_rows
        .iter()
        .map(|row| (row.ali_from, row.ali_to, row.strand.clone()))
        .collect();
    let both_set: HashSet<(usize, usize, String)> = both_rows
        .iter()
        .map(|row| (row.ali_from, row.ali_to, row.strand.clone()))
        .collect();

    assert!(
        watson_set.is_disjoint(&crick_set),
        "strand-specific hit sets should not overlap"
    );
    assert_eq!(both_set.len(), watson_set.len() + crick_set.len());
    assert_eq!(
        both_set,
        watson_set.union(&crick_set).cloned().collect(),
        "combined nhmmer output should equal the Watson/Crick union"
    );
}

#[test]
fn test_nhmmer_z_option_scales_longtarget_evalues() {
    let (_z1_stdout, z1_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &["--watson", "-Z", "1"],
    );
    let (_z2_stdout, z2_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &["--watson", "-Z", "2"],
    );

    let z1_rows = parse_nhmmer_rows(&z1_tbl);
    let z2_rows = parse_nhmmer_rows(&z2_tbl);
    assert_eq!(z1_rows.len(), z2_rows.len());
    assert!(!z1_rows.is_empty());
    for (a, b) in z1_rows.iter().zip(&z2_rows) {
        assert_eq!(
            (&a.target, &a.query, a.ali_from, a.ali_to),
            (&b.target, &b.query, b.ali_from, b.ali_to)
        );
        let e1: f64 = a.evalue.parse().unwrap();
        let e2: f64 = b.evalue.parse().unwrap();
        let ratio = e2 / e1;
        assert!(
            (1.8..=2.2).contains(&ratio),
            "-Z did not scale nhmmer E-value about 2x: {e1} -> {e2} ({ratio})"
        );
    }
}

#[test]
fn test_nhmmer_keeps_hits_from_duplicate_target_names() {
    let dir = tempfile::tempdir().unwrap();
    let dup_targets = dir.path().join("duplicate-targets.fa");
    let target = std::fs::read_to_string(test_path("hmmer/tutorial/dna_target.fa")).unwrap();
    std::fs::write(&dup_targets, format!("{target}{target}")).unwrap();

    let (_single_stdout, single_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &[],
    );
    let (_dup_stdout, dup_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        dup_targets.to_str().unwrap(),
        &[],
    );

    let single_rows = parse_nhmmer_rows(&single_tbl);
    let dup_rows = parse_nhmmer_rows(&dup_tbl);
    assert_eq!(dup_rows.len(), single_rows.len() * 2);
}

#[test]
fn test_nhmmer_longtarget_window_length_override_is_honored() {
    let (_default_stdout, default_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &[],
    );
    let (_tiny_window_stdout, tiny_window_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &["--w_length", "4"],
    );

    assert_eq!(parse_nhmmer_rows(&default_tbl).len(), 5);
    assert!(
        parse_nhmmer_rows(&tiny_window_tbl).is_empty(),
        "a tiny --w_length override should be used by all long-target filter stages"
    );
}

#[test]
fn test_nhmmscan_made1_tblout_matches_bundled_c_rows() {
    let hmm = test_path("hmmer/tutorial/MADE1.hmm");
    let seqdb = test_path("hmmer/tutorial/dna_target.fa");
    let c_rows = parse_nhmmer_rows(&run_c_nhmmscan_tblout(&hmm, &seqdb));
    let rust_rows = parse_nhmmer_rows(&run_nhmmscan_tblout(&hmm, &seqdb));

    assert_eq!(rust_rows, c_rows);
    assert_eq!(rust_rows.len(), 5);
    assert_eq!(rust_rows.iter().filter(|row| row.strand == "+").count(), 3);
    assert_eq!(rust_rows.iter().filter(|row| row.strand == "-").count(), 2);
}

#[test]
fn test_nhmmscan_made1_dfamtblout_matches_bundled_c() {
    let hmm = test_path("hmmer/tutorial/MADE1.hmm");
    let seqdb = test_path("hmmer/tutorial/dna_target.fa");
    let c_dfam = run_c_nhmmscan_dfamtblout(&hmm, &seqdb);
    let rust_dfam = run_nhmmscan_dfamtblout(&hmm, &seqdb);

    assert_eq!(rust_dfam, c_dfam);
    assert!(rust_dfam.contains(" strand  ali-st  ali-en"));
    assert!(rust_dfam.contains("    -    302466  302390"));
}

#[test]
fn test_hmmscan_fn3_tblout_matches_bundled_c_rows() {
    let hmm = test_path("hmmer/tutorial/fn3.hmm");
    let seqdb = test_path("hmmer/tutorial/7LESS_DROME");
    let rust_rows = parse_hmmsearch_rows(&run_hmmscan_tblout(&hmm, &seqdb));
    let c_rows = parse_hmmsearch_rows(&run_c_hmmscan_tblout(&hmm, &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "hmmscan fn3 tblout rows diverged from bundled C output"
    );
}

#[test]
fn test_hmmscan_fn3_domtblout_matches_bundled_c_rows() {
    let hmm = test_path("hmmer/tutorial/fn3.hmm");
    let seqdb = test_path("hmmer/tutorial/7LESS_DROME");
    let rust_rows = parse_domtbl_rows(&run_hmmscan_domtblout(&hmm, &seqdb));
    let c_rows = parse_domtbl_rows(&run_c_hmmscan_domtblout(&hmm, &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "hmmscan fn3 domtblout rows diverged from bundled C output"
    );
}

#[test]
fn test_hmmscan_fn3_pfamtblout_matches_bundled_c_rows() {
    let hmm = test_path("hmmer/tutorial/fn3.hmm");
    let seqdb = test_path("hmmer/tutorial/7LESS_DROME");
    let rust_rows = parse_pfamtbl_rows(&run_hmmscan_pfamtblout(&hmm, &seqdb));
    let c_rows = parse_pfamtbl_rows(&run_c_hmmscan_pfamtblout(&hmm, &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "hmmscan fn3 pfamtblout rows diverged from bundled C output"
    );
}

#[test]
fn test_nhmmer_ecori_requires_explicit_dna_and_runs_cleanly() {
    let golden_stdout =
        std::fs::read_to_string(test_path("tests/golden/nhmmer_ecori.stdout")).unwrap();
    let golden_tbl =
        std::fs::read_to_string(test_path("tests/golden/nhmmer_ecori.tblout")).unwrap();
    let (stdout, tbl) = run_nhmmer(
        &test_path("hmmer/testsuite/ecori.hmm"),
        &test_path("hmmer/testsuite/ecori.fa"),
        &["--dna"],
    );
    let rows = parse_nhmmer_rows(&tbl);
    assert!(
        rows.is_empty(),
        "ecori fixture should not report hits at default thresholds"
    );
    assert!(stdout.contains("# input query is asserted as:      DNA"));
    assert!(stdout.contains("[No hits detected that satisfy reporting thresholds]"));
    assert!(stdout.contains("[ok]"));
    assert_eq!(
        normalize_nhmmer_stdout(&stdout),
        normalize_nhmmer_stdout(&golden_stdout),
        "nhmmer ecori stdout diverged from golden output"
    );
    assert_eq!(
        normalize_nhmmer_tblout_with_fixture(
            &tbl,
            "hmmer/testsuite/ecori.hmm",
            "hmmer/testsuite/ecori.fa",
        ),
        normalize_nhmmer_tblout_with_fixture(
            &golden_tbl,
            "hmmer/testsuite/ecori.hmm",
            "hmmer/testsuite/ecori.fa",
        ),
        "nhmmer ecori tblout diverged from golden output"
    );
}

#[test]
fn test_hmmsearch_max_uses_c_like_thresholds_and_domains() {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("domains.tbl");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "search",
            "--max",
            "--cpu",
            "1",
            "--noali",
            "--domtblout",
            domtblout.to_str().unwrap(),
            &test_path("test_data/mapali/20aa-rebuilt.hmm"),
            &test_path("test_data/gecco_cluster1_proteins.faa"),
        ])
        .output()
        .expect("failed to run hmmer search --max");
    assert!(
        output.status.success(),
        "hmmer search --max failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "Passed MSV filter:                         6  (1); expected 6.0 (1)",
        "Passed bias filter:                        6  (1); expected 6.0 (1)",
        "Passed Vit filter:                         6  (1); expected 6.0 (1)",
        "Passed Fwd filter:                         6  (1); expected 6.0 (1)",
        "Domain search space  (domZ):               4  [number of targets reported over threshold]",
    ] {
        assert!(
            stdout.contains(expected),
            "{expected:?} missing from:\n{stdout}"
        );
    }

    let rows: Vec<Vec<String>> = std::fs::read_to_string(domtblout)
        .unwrap()
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| line.split_whitespace().map(str::to_string).collect())
        .collect();
    assert_eq!(rows.len(), 5, "hmmsearch --max domain row count drifted");
    assert_eq!(rows[0][0], "CP157504.1_560");
    assert_eq!(rows[0][15].as_str(), "5");
    assert_eq!(rows[0][16].as_str(), "10");
    assert_eq!(rows[0][17].as_str(), "46");
    assert_eq!(rows[0][18].as_str(), "51");
    assert_eq!(rows[4][0], "CP157504.1_562");
}

#[test]
fn test_nhmmer_ecori_max_matches_c_no_hit_behavior() {
    let (stdout, tbl) = run_nhmmer(
        &test_path("test_data/mapali/ecori-rebuilt.hmm"),
        &test_path("test_data/mapali/ecori-query.fa"),
        &["--dna", "--max", "--cpu", "1", "--noali"],
    );

    assert!(
        parse_nhmmer_rows(&tbl).is_empty(),
        "nhmmer --max ecori fixture should report no hits"
    );
    for expected in [
        "Residues passing SSV filter:              12  (1); expected (0.3)",
        "Residues passing bias filter:             12  (1); expected (0.3)",
        "Residues passing Vit filter:              12  (1); expected (1)",
        "Residues passing Fwd filter:              12  (1); expected (1)",
        "Total number of hits:                      0  (0)",
    ] {
        assert!(
            stdout.contains(expected),
            "{expected:?} missing from:\n{stdout}"
        );
    }
}

#[test]
fn test_nhmmer_max_uses_longtarget_max_filter_thresholds() {
    let (stdout, _tbl) = run_nhmmer(
        &test_path("test_data/mapali/ecori-rebuilt.hmm"),
        &test_path("test_data/mapali/ecori-query.fa"),
        &["--dna", "--max", "--cpu", "1", "--noali"],
    );

    for expected in [
        "# Max sensitivity mode:            on [all heuristic filters off]",
        "Residues passing SSV filter:              12  (1); expected (0.3)",
        "Residues passing bias filter:             12  (1); expected (0.3)",
        "Residues passing Vit filter:              12  (1); expected (1)",
        "Residues passing Fwd filter:              12  (1); expected (1)",
    ] {
        assert!(
            stdout.contains(expected),
            "{expected:?} missing from:\n{stdout}"
        );
    }
}

#[test]
fn test_nhmmer_max_is_at_least_as_sensitive_as_default_on_tutorial() {
    let (_default_stdout, default_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &["--cpu", "1", "--noali"],
    );
    let (_max_stdout, max_tbl) = run_nhmmer(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
        &["--max", "--cpu", "1", "--noali"],
    );

    let default_rows = parse_nhmmer_rows(&default_tbl);
    let max_rows = parse_nhmmer_rows(&max_tbl);
    assert!(
        max_rows.len() >= default_rows.len(),
        "nhmmer --max should not report fewer tutorial hits than default"
    );
}

#[test]
fn test_nhmmer_3box_exact_parity_bundle() {
    let golden_stdout =
        std::fs::read_to_string(test_path("tests/golden/nhmmer_3box.stdout")).unwrap();
    let golden_tbl = std::fs::read_to_string(test_path("tests/golden/nhmmer_3box.tblout")).unwrap();
    let (stdout, tbl) = run_nhmmer(
        &test_path("hmmer/testsuite/3box.hmm"),
        &test_path("hmmer/testsuite/3box-alitest.fa"),
        &["--dna"],
    );

    let rows = parse_nhmmer_rows(&tbl);
    assert_eq!(rows.len(), 2, "3box fixture should report exactly two hits");
    assert!(rows.iter().all(|row| row.strand == "+"));
    assert_eq!(rows[0].target, "random");
    assert_eq!((rows[0].ali_from, rows[0].ali_to), (4141, 4158));
    assert_eq!((rows[1].ali_from, rows[1].ali_to), (7162, 7181));

    assert_eq!(
        normalize_nhmmer_stdout(&stdout),
        normalize_nhmmer_stdout(&golden_stdout),
        "nhmmer 3box stdout diverged from golden output"
    );
    assert_eq!(
        normalize_nhmmer_tblout_with_fixture(
            &tbl,
            "hmmer/testsuite/3box.hmm",
            "hmmer/testsuite/3box-alitest.fa",
        ),
        normalize_nhmmer_tblout_with_fixture(
            &golden_tbl,
            "hmmer/testsuite/3box.hmm",
            "hmmer/testsuite/3box-alitest.fa",
        ),
        "nhmmer 3box tblout diverged from golden output"
    );
}

#[test]
fn test_nhmmer_3box_dna_target_matches_c_block_window_ssv_counter() {
    // Bundled C scans long targets in overlapping blocks by default. The overlap
    // can duplicate SSV accounting even when reported hits and later counters
    // are unchanged, so Rust mirrors that visible stdout counter.
    let hmm = test_path("hmmer/testsuite/3box.hmm");
    let seqdb = test_path("hmmer/tutorial/dna_target.fa");
    let args = ["--dna", "--cpu", "1", "--noali"];
    let (stdout, tbl) = run_nhmmer(&hmm, &seqdb, &args);
    let (c_stdout, c_tbl) = run_c_nhmmer_with_args(&hmm, &seqdb, &args);

    let rows = parse_nhmmer_rows(&tbl);
    let c_rows = parse_nhmmer_rows(&c_tbl);
    assert_eq!(rows, c_rows, "3box dna_target rows should match bundled C");
    assert_eq!(
        rows.len(),
        2,
        "3box dna_target fixture should keep two hits"
    );
    assert_eq!(rows[0].target, "humanchr1_frag");
    assert_eq!((rows[0].ali_from, rows[0].ali_to), (178064, 178049));
    assert_eq!((rows[1].ali_from, rows[1].ali_to), (96791, 96776));
    assert!(
        c_stdout
            .contains("Residues passing SSV filter:           45479  (0.0689); expected (0.02)"),
        "bundled C 3box dna_target SSV counter changed:\n{c_stdout}"
    );
    assert!(
        stdout.contains("Residues passing SSV filter:           45479  (0.0689); expected (0.02)"),
        "Rust 3box dna_target SSV counter diverged from bundled C:\n{stdout}"
    );
    for expected in [
        "Residues passing bias filter:          21408  (0.0324); expected (0.02)",
        "Residues passing Vit filter:            3154  (0.00478); expected (0.003)",
        "Residues passing Fwd filter:             108  (0.000164); expected (3e-05)",
    ] {
        assert!(
            stdout.contains(expected),
            "{expected:?} missing from:\n{stdout}"
        );
        assert!(
            c_stdout.contains(expected),
            "{expected:?} missing from bundled C output:\n{c_stdout}"
        );
    }
}

#[test]
fn test_gecco_pfam5_real_world_query_hit_counts_match_golden() {
    let golden =
        std::fs::read_to_string(test_path("tests/golden/gecco_pfam5_vs_gecco.tblout")).unwrap();
    let rust = run_hmmsearch_tblout(
        &test_path("hmmer/testsuite/gecco_pfam5.hmm"),
        &test_path("hmmer/testsuite/gecco_proteins.faa"),
    );

    let golden_rows = parse_hmmsearch_rows(&golden);
    let rust_rows = parse_hmmsearch_rows(&rust);

    assert_eq!(
        rust_rows.len(),
        16,
        "gecco_pfam5 fixture should produce 16 hits"
    );
    assert_eq!(
        rust_rows.len(),
        golden_rows.len(),
        "gecco_pfam5 total hit count diverged"
    );
    assert_eq!(rust_rows, golden_rows, "gecco_pfam5 core hit rows diverged");
}

#[test]
fn test_gecco_missed_real_world_query_hit_counts_match_golden() {
    let golden =
        std::fs::read_to_string(test_path("tests/golden/gecco_missed_vs_missed.tblout")).unwrap();
    let rust = run_hmmsearch_tblout(
        &test_path("hmmer/testsuite/gecco_missed_hmms.hmm"),
        &test_path("hmmer/testsuite/gecco_missed_proteins.faa"),
    );

    let golden_rows = parse_hmmsearch_rows(&golden);
    let rust_rows = parse_hmmsearch_rows(&rust);

    assert_eq!(
        rust_rows.len(),
        golden_rows.len(),
        "gecco_missed total hit count diverged"
    );
    assert_eq!(
        rust_rows, golden_rows,
        "gecco_missed core hit rows diverged"
    );
}

#[test]
fn test_gecco_missed2_real_world_query_hit_counts_match_golden() {
    let golden = std::fs::read_to_string(test_path("tests/golden/gecco_missed2.tblout")).unwrap();
    let rust = run_hmmsearch_tblout(
        &test_path("hmmer/testsuite/gecco_missed2_hmms.hmm"),
        &test_path("hmmer/testsuite/gecco_missed2_proteins.faa"),
    );

    let golden_rows = parse_hmmsearch_rows(&golden);
    let rust_rows = parse_hmmsearch_rows(&rust);

    assert_eq!(
        rust_rows.len(),
        golden_rows.len(),
        "gecco_missed2 total hit count diverged"
    );
    assert_eq!(
        query_hit_counts(&rust_rows),
        query_hit_counts(&golden_rows),
        "gecco_missed2 per-query hit counts diverged"
    );
}

#[test]
fn test_gecco_missed3_real_world_query_hit_counts_match_golden() {
    let golden = std::fs::read_to_string(test_path("tests/golden/gecco_missed3.tblout")).unwrap();
    let rust = run_hmmsearch_tblout(
        &test_path("hmmer/testsuite/gecco_missed3_hmms.hmm"),
        &test_path("hmmer/testsuite/gecco_missed3_proteins.faa"),
    );

    let golden_rows = parse_hmmsearch_rows(&golden);
    let rust_rows = parse_hmmsearch_rows(&rust);

    assert_eq!(
        rust_rows.len(),
        golden_rows.len(),
        "gecco_missed3 total hit count diverged"
    );
    assert_eq!(
        query_hit_counts(&rust_rows),
        query_hit_counts(&golden_rows),
        "gecco_missed3 per-query hit counts diverged"
    );
}

#[test]
fn test_gecco_missed4_real_world_query_hit_counts_match_golden() {
    let golden = std::fs::read_to_string(test_path("tests/golden/gecco_missed4.tblout")).unwrap();
    let rust = run_hmmsearch_tblout(
        &test_path("hmmer/testsuite/gecco_missed4_hmms.hmm"),
        &test_path("hmmer/testsuite/gecco_missed4_proteins.faa"),
    );

    let golden_rows = parse_hmmsearch_rows(&golden);
    let rust_rows = parse_hmmsearch_rows(&rust);

    assert_eq!(
        rust_rows.len(),
        60,
        "gecco_missed4 fixture should produce 60 hits"
    );
    assert_eq!(
        rust_rows.len(),
        golden_rows.len(),
        "gecco_missed4 total hit count diverged"
    );
    assert_eq!(
        query_hit_counts(&rust_rows),
        query_hit_counts(&golden_rows),
        "gecco_missed4 per-query hit counts diverged"
    );

    let golden_queries: HashSet<String> =
        golden_rows.iter().map(|(_, q, _, _)| q.clone()).collect();
    let rust_queries: HashSet<String> = rust_rows.iter().map(|(_, q, _, _)| q.clone()).collect();
    assert_eq!(
        rust_queries, golden_queries,
        "gecco_missed4 query coverage diverged"
    );
}

#[test]
fn test_gecco_real_world_score_bias_rows_match_golden_exactly() {
    let cases = [
        (
            "gecco_pfam5",
            test_path("tests/golden/gecco_pfam5_vs_gecco.tblout"),
            test_path("hmmer/testsuite/gecco_pfam5.hmm"),
            test_path("hmmer/testsuite/gecco_proteins.faa"),
        ),
        (
            "gecco_missed",
            test_path("tests/golden/gecco_missed_vs_missed.tblout"),
            test_path("hmmer/testsuite/gecco_missed_hmms.hmm"),
            test_path("hmmer/testsuite/gecco_missed_proteins.faa"),
        ),
        (
            "gecco_missed2",
            test_path("tests/golden/gecco_missed2.tblout"),
            test_path("hmmer/testsuite/gecco_missed2_hmms.hmm"),
            test_path("hmmer/testsuite/gecco_missed2_proteins.faa"),
        ),
        (
            "gecco_missed3",
            test_path("tests/golden/gecco_missed3.tblout"),
            test_path("hmmer/testsuite/gecco_missed3_hmms.hmm"),
            test_path("hmmer/testsuite/gecco_missed3_proteins.faa"),
        ),
        (
            "gecco_missed4",
            test_path("tests/golden/gecco_missed4.tblout"),
            test_path("hmmer/testsuite/gecco_missed4_hmms.hmm"),
            test_path("hmmer/testsuite/gecco_missed4_proteins.faa"),
        ),
    ];

    for (label, golden_path, hmm_path, seqdb_path) in cases {
        let golden = std::fs::read_to_string(golden_path).unwrap();
        let rust = run_hmmsearch_tblout(&hmm_path, &seqdb_path);

        assert_eq!(
            parse_hmmsearch_score_bias_rows(&rust),
            parse_hmmsearch_score_bias_rows(&golden),
            "{} score/bias rows diverged from golden output",
            label
        );
    }
}

#[test]
fn test_gecco_pfam5_top_hits_match_golden_rows() {
    let golden =
        std::fs::read_to_string(test_path("tests/golden/gecco_pfam5_vs_gecco.tblout")).unwrap();
    let rust = run_hmmsearch_tblout(
        &test_path("hmmer/testsuite/gecco_pfam5.hmm"),
        &test_path("hmmer/testsuite/gecco_proteins.faa"),
    );

    let golden_rows = parse_hmmsearch_rows(&golden);
    let rust_rows = parse_hmmsearch_rows(&rust);

    assert_eq!(
        rust_rows
            .iter()
            .take(5)
            .map(|r| (&r.0, &r.1))
            .collect::<Vec<_>>(),
        golden_rows
            .iter()
            .take(5)
            .map(|r| (&r.0, &r.1))
            .collect::<Vec<_>>(),
        "gecco_pfam5 top five hit rows diverged from golden ordering"
    );
}

#[test]
fn test_representative_pfam_real_world_top_hits_match_golden() {
    let families = ["Globin", "Ras", "Trypsin", "RVT_1"];
    let seqdb = test_path("test_data/human_swissprot_2k.fasta");

    for family in families {
        let golden =
            std::fs::read_to_string(test_path(&format!("tests/golden/pfam_{}.tblout", family)))
                .unwrap();
        let rust = run_hmmsearch_tblout(
            &test_path(&format!("test_data/{}_pfam.hmm", family)),
            &seqdb,
        );

        let golden_rows = parse_hmmsearch_rows(&golden);
        let rust_rows = parse_hmmsearch_rows(&rust);

        assert!(
            !golden_rows.is_empty() && !rust_rows.is_empty(),
            "{} should produce real hits in the Swiss-Prot fixture",
            family
        );
        assert_eq!(
            rust_rows[0].0, golden_rows[0].0,
            "{} top hit target diverged from golden output",
            family
        );
        assert_eq!(
            rust_rows[0].1, golden_rows[0].1,
            "{} top hit query diverged from golden output",
            family
        );
    }
}

#[test]
fn test_gecco_pfam5_pfamtblout_matches_bundled_c_exactly() {
    let rust = run_hmmsearch_pfamtblout(
        &test_path("hmmer/testsuite/gecco_pfam5.hmm"),
        &test_path("hmmer/testsuite/gecco_proteins.faa"),
    );
    let c = run_c_hmmsearch_pfamtblout(
        &test_path("hmmer/testsuite/gecco_pfam5.hmm"),
        &test_path("hmmer/testsuite/gecco_proteins.faa"),
    );

    assert_eq!(
        parse_pfamtbl_rows(&rust),
        parse_pfamtbl_rows(&c),
        "gecco_pfam5 pfamtblout rows diverged from bundled C output"
    );
}

#[test]
fn test_fn3_domtblout_rows_match_golden_core_columns() {
    let golden = std::fs::read_to_string(test_path("tests/golden/fn3_vs_7less.domtblout")).unwrap();
    let rust = run_hmmsearch_domtblout(
        &test_path("hmmer/tutorial/fn3.hmm"),
        &test_path("hmmer/tutorial/7LESS_DROME"),
    );

    let golden_rows = parse_domtbl_rows(&golden);
    let rust_rows = parse_domtbl_rows(&rust);
    assert_eq!(
        rust_rows, golden_rows,
        "fn3 domtblout rows diverged from golden output"
    );
}

#[test]
fn test_gecco_pfam5_domtblout_query_counts_are_stable() {
    let hmm = test_path("hmmer/testsuite/gecco_pfam5.hmm");
    let seqdb = test_path("hmmer/testsuite/gecco_proteins.faa");
    let rust_rows = parse_domtbl_rows(&run_hmmsearch_domtblout(&hmm, &seqdb));
    let c_rows = parse_domtbl_rows(&run_c_hmmsearch_domtblout(&hmm, &seqdb));
    assert_eq!(rust_rows, c_rows);
}

#[test]
fn test_rrm1_domtblout_multi_domain_profile_is_stable() {
    let rows = parse_domtbl_rows(&run_hmmsearch_domtblout(
        &test_path("test_data/RRM_1_pfam.hmm"),
        &test_path("test_data/human_swissprot_2k.fasta"),
    ));
    assert_eq!(
        rows.len(),
        419,
        "RRM_1 domtblout total domain count changed"
    );

    let best = rows
        .iter()
        .max_by(|a, b| {
            a.score
                .parse::<f64>()
                .unwrap()
                .total_cmp(&b.score.parse::<f64>().unwrap())
        })
        .unwrap();
    assert_eq!(best.target, "sp|Q14011|CIRBP_HUMAN");
    assert_eq!(best.query, "RRM_1");
    assert_eq!(best.score, "91.2");
    assert_eq!((best.ali_from, best.ali_to), (8, 78));

    let pabp1_domains = rows
        .iter()
        .filter(|row| row.target == "sp|P11940|PABP1_HUMAN")
        .count();
    assert_eq!(
        pabp1_domains, 4,
        "PABP1_HUMAN should still carry four RRM_1 domains"
    );
}

#[test]
#[ignore = "slow parity sweep across all committed Pfam golden fixtures"]
fn test_pfam_top3_rows_match_golden_for_all_families() {
    let seqdb = test_path("test_data/human_swissprot_2k.fasta");
    for family in PFAM_FAMILIES {
        let golden =
            std::fs::read_to_string(test_path(&format!("tests/golden/pfam_{}.tblout", family)))
                .unwrap();
        let rust = run_hmmsearch_tblout(
            &test_path(&format!("test_data/{}_pfam.hmm", family)),
            &seqdb,
        );
        let golden_rows = parse_hmmsearch_rows(&golden);
        let rust_rows = parse_hmmsearch_rows(&rust);
        let n = golden_rows.len().min(rust_rows.len()).min(3);
        assert_eq!(
            rust_rows
                .iter()
                .take(n)
                .map(|r| (&r.0, &r.1))
                .collect::<Vec<_>>(),
            golden_rows
                .iter()
                .take(n)
                .map(|r| (&r.0, &r.1))
                .collect::<Vec<_>>(),
            "{} top-{} rows diverged from golden output",
            family,
            n
        );
    }
}

#[test]
#[ignore = "slow parity sweep across all committed Pfam golden fixtures"]
fn test_pfam_top3_score_bias_rows_match_golden_for_all_families() {
    let seqdb = test_path("test_data/human_swissprot_2k.fasta");
    for family in PFAM_FAMILIES {
        let golden =
            std::fs::read_to_string(test_path(&format!("tests/golden/pfam_{}.tblout", family)))
                .unwrap();
        let rust = run_hmmsearch_tblout(
            &test_path(&format!("test_data/{}_pfam.hmm", family)),
            &seqdb,
        );
        let golden_rows = parse_hmmsearch_score_bias_rows(&golden);
        let rust_rows = parse_hmmsearch_score_bias_rows(&rust);
        let n = golden_rows.len().min(rust_rows.len()).min(3);
        assert_eq!(
            rust_rows.iter().take(n).collect::<Vec<_>>(),
            golden_rows.iter().take(n).collect::<Vec<_>>(),
            "{} top-{} score/bias rows diverged from golden output",
            family,
            n
        );
    }
}

#[test]
#[ignore = "slow parity sweep across all committed Pfam golden fixtures"]
fn test_pfam_per_query_hit_counts_match_golden_for_all_families() {
    let seqdb = test_path("test_data/human_swissprot_2k.fasta");
    for family in PFAM_FAMILIES {
        let golden =
            std::fs::read_to_string(test_path(&format!("tests/golden/pfam_{}.tblout", family)))
                .unwrap();
        let rust = run_hmmsearch_tblout(
            &test_path(&format!("test_data/{}_pfam.hmm", family)),
            &seqdb,
        );
        assert_eq!(
            query_hit_counts(&parse_hmmsearch_rows(&rust)),
            query_hit_counts(&parse_hmmsearch_rows(&golden)),
            "{} per-query hit counts diverged from golden output",
            family
        );
    }
}
