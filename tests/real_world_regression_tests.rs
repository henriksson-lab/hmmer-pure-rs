//! Real-world regression tests that exercise end-to-end CLI behavior on
//! realistic fixtures instead of synthetic toy cases.

use std::collections::{BTreeMap, HashSet};
use std::io::Read;
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

fn run_c_hmmsearch_tblout(hmm: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(test_path("hmmer/src/hmmsearch"))
        .args(["--noali", "--tblout", tblout.to_str().unwrap(), hmm, seqdb])
        .output()
        .expect("failed to run bundled C hmmsearch");
    assert!(
        output.status.success(),
        "bundled C hmmsearch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_hmmsearch_tblout_with_args(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let mut args = vec!["search", "--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--tblout", tblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
        .output()
        .expect("failed to run hmmer search");
    assert!(
        output.status.success(),
        "hmmer search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_c_hmmsearch_tblout_with_args(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let mut args = vec!["--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--tblout", tblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(test_path("hmmer/src/hmmsearch"))
        .args(args)
        .output()
        .expect("failed to run bundled C hmmsearch");
    assert!(
        output.status.success(),
        "bundled C hmmsearch failed: {}",
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

fn run_c_nhmmer_tblout(hmm: &str, seqdb: &str) -> String {
    run_c_nhmmer_tblout_with_args(hmm, seqdb, &[])
}

fn run_nhmmer_dfamtblout(hmm: &str, seqdb: &str) -> String {
    run_nhmmer_dfamtblout_with_args(hmm, seqdb, &[])
}

fn run_nhmmer_dfamtblout_with_args(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let dfamtblout = dir.path().join("dfamtblout.txt");
    let mut args = vec!["nhmmer"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--dfamtblout", dfamtblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
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
    run_c_nhmmer_dfamtblout_with_args(hmm, seqdb, &[])
}

fn run_c_nhmmer_dfamtblout_with_args(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let dfamtblout = dir.path().join("dfamtblout.txt");
    let mut args = extra_args.to_vec();
    args.extend_from_slice(&["--dfamtblout", dfamtblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(test_path("hmmer/src/nhmmer"))
        .args(args)
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

fn build_rust_hmm(args: &[&str], output: &std::path::Path, msa: &str) {
    let mut cmd_args = vec!["build"];
    cmd_args.extend_from_slice(args);
    cmd_args.extend_from_slice(&[output.to_str().unwrap(), msa]);
    let output = Command::new(binary_path("hmmer"))
        .args(cmd_args)
        .output()
        .expect("failed to run hmmer build");
    assert!(
        output.status.success(),
        "hmmer build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn build_c_hmm(args: &[&str], output: &std::path::Path, msa: &str) {
    let mut cmd_args = args.to_vec();
    cmd_args.extend_from_slice(&[output.to_str().unwrap(), msa]);
    let output = Command::new(test_path("hmmer/src/hmmbuild"))
        .args(cmd_args)
        .output()
        .expect("failed to run bundled C hmmbuild");
    assert!(
        output.status.success(),
        "bundled C hmmbuild failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn hmm_without_date(path: &std::path::Path) -> String {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.starts_with("DATE  "))
        .collect::<Vec<_>>()
        .join("\n")
}

fn hmm_without_date_or_com(path: &std::path::Path) -> String {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.starts_with("DATE  "))
        .filter(|line| !line.starts_with("COM   "))
        .collect::<Vec<_>>()
        .join("\n")
}

fn assert_hmm_text_matches_with_float_tolerance(rust: &str, c: &str, tol: f64, context: &str) {
    let rust_tokens: Vec<&str> = rust.split_whitespace().collect();
    let c_tokens: Vec<&str> = c.split_whitespace().collect();
    assert_eq!(
        rust_tokens.len(),
        c_tokens.len(),
        "{context}: token counts differ"
    );
    for (idx, (rust_tok, c_tok)) in rust_tokens.iter().zip(c_tokens.iter()).enumerate() {
        if rust_tok == c_tok {
            continue;
        }
        match (rust_tok.parse::<f64>(), c_tok.parse::<f64>()) {
            (Ok(rust_val), Ok(c_val)) if (rust_val - c_val).abs() <= tol => {}
            _ => panic!("{context}: token {idx} differs: rust={rust_tok:?} c={c_tok:?}"),
        }
    }
}

fn first_stockholm_from_gzip(gz_path: &str, output: &std::path::Path) {
    let file = std::fs::File::open(gz_path).unwrap();
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut text = String::new();
    decoder.read_to_string(&mut text).unwrap();
    let mut first = String::new();
    for line in text.lines() {
        first.push_str(line);
        first.push('\n');
        if line.trim() == "//" {
            break;
        }
    }
    assert!(first.contains("//"));
    std::fs::write(output, first).unwrap();
}

fn write_first_fasta_records(path: &std::path::Path, source_rel: &str, n_records: usize) {
    let src = std::fs::read_to_string(test_path(source_rel)).unwrap();
    let mut out = String::new();
    let mut seen = 0usize;
    for line in src.lines() {
        if line.starts_with('>') {
            seen += 1;
        }
        if seen > n_records {
            break;
        }
        if seen > 0 {
            out.push_str(line);
            out.push('\n');
        }
    }
    assert_eq!(
        seen.min(n_records),
        n_records,
        "source FASTA {source_rel} contained fewer than {n_records} records"
    );
    std::fs::write(path, out).unwrap();
}

fn output_without_cpu_footer(output: &str) -> String {
    output
        .lines()
        .filter(|line| !line.starts_with("# CPU time:"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_hmmbuild_summary(output: &str) -> String {
    output
        .lines()
        .filter(|line| !line.starts_with("# CPU time:"))
        .filter(|line| !line.starts_with("# output HMM file:"))
        .filter(|line| !line.starts_with("# output alignment file:"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn make_rust_fm_database(seqdb: &str, output_path: &std::path::Path) {
    let output = Command::new(binary_path("hmmer"))
        .args(["makehmmerdb", seqdb, output_path.to_str().unwrap()])
        .output()
        .expect("failed to run hmmer makehmmerdb");
    assert!(
        output.status.success(),
        "hmmer makehmmerdb failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn make_c_fm_database(seqdb: &str, output_path: &std::path::Path) {
    let output = Command::new(test_path("hmmer/src/makehmmerdb"))
        .args([seqdb, output_path.to_str().unwrap()])
        .output()
        .expect("failed to run bundled C makehmmerdb");
    assert!(
        output.status.success(),
        "bundled C makehmmerdb failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_hmmer_stdout(args: &[&str]) -> String {
    let output = Command::new(binary_path("hmmer"))
        .args(args)
        .output()
        .expect("failed to run hmmer command");
    assert!(
        output.status.success(),
        "hmmer command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn run_c_tool_stdout(tool: &str, args: &[&str]) -> String {
    let output = Command::new(test_path(&format!("hmmer/src/{tool}")))
        .args(args)
        .output()
        .expect("failed to run bundled C command");
    assert!(
        output.status.success(),
        "bundled C {tool} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn press_rust_hmmdb(hmmdb: &std::path::Path) {
    let output = Command::new(binary_path("hmmer"))
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .expect("failed to run hmmer press");
    assert!(
        output.status.success(),
        "hmmer press failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn press_c_hmmdb(hmmdb: &std::path::Path) {
    let output = Command::new(test_path("hmmer/src/hmmpress"))
        .args(["-f", hmmdb.to_str().unwrap()])
        .output()
        .expect("failed to run bundled C hmmpress");
    assert!(
        output.status.success(),
        "bundled C hmmpress failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_phmmer_tblout(seqfile: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "phmmer",
            "--tblout",
            tblout.to_str().unwrap(),
            seqfile,
            seqdb,
        ])
        .output()
        .expect("failed to run hmmer phmmer");
    assert!(
        output.status.success(),
        "hmmer phmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_c_phmmer_tblout_real(seqfile: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(test_path("hmmer/src/phmmer"))
        .args(["--tblout", tblout.to_str().unwrap(), seqfile, seqdb])
        .output()
        .expect("failed to run bundled C phmmer");
    assert!(
        output.status.success(),
        "bundled C phmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_phmmer_tblout_with_args(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let mut args = vec!["phmmer", "--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--tblout", tblout.to_str().unwrap(), seqfile, seqdb]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
        .output()
        .expect("failed to run hmmer phmmer");
    assert!(
        output.status.success(),
        "hmmer phmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_c_phmmer_tblout_with_args(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let mut args = vec!["--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--tblout", tblout.to_str().unwrap(), seqfile, seqdb]);
    let output = Command::new(test_path("hmmer/src/phmmer"))
        .args(args)
        .output()
        .expect("failed to run bundled C phmmer");
    assert!(
        output.status.success(),
        "bundled C phmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_phmmer_domtblout_with_args(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("domtblout.txt");
    let mut args = vec!["phmmer"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--domtblout", domtblout.to_str().unwrap(), seqfile, seqdb]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
        .output()
        .expect("failed to run hmmer phmmer");
    assert!(
        output.status.success(),
        "hmmer phmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(domtblout).unwrap()
}

fn run_phmmer_pfamtblout_with_args(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let pfamtblout = dir.path().join("pfamtblout.txt");
    let mut args = vec!["phmmer"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--pfamtblout", pfamtblout.to_str().unwrap(), seqfile, seqdb]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
        .output()
        .expect("failed to run hmmer phmmer");
    assert!(
        output.status.success(),
        "hmmer phmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(pfamtblout).unwrap()
}

fn run_c_phmmer_pfamtblout_with_args(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let pfamtblout = dir.path().join("pfamtblout.txt");
    let mut args = extra_args.to_vec();
    args.extend_from_slice(&["--pfamtblout", pfamtblout.to_str().unwrap(), seqfile, seqdb]);
    let output = Command::new(test_path("hmmer/src/phmmer"))
        .args(args)
        .output()
        .expect("failed to run bundled C phmmer");
    assert!(
        output.status.success(),
        "bundled C phmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(pfamtblout).unwrap()
}

fn run_c_phmmer_domtblout_with_args(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("domtblout.txt");
    let mut args = extra_args.to_vec();
    args.extend_from_slice(&["--domtblout", domtblout.to_str().unwrap(), seqfile, seqdb]);
    let output = Command::new(test_path("hmmer/src/phmmer"))
        .args(args)
        .output()
        .expect("failed to run bundled C phmmer");
    assert!(
        output.status.success(),
        "bundled C phmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(domtblout).unwrap()
}

fn run_jackhmmer_tblout(seqfile: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "jackhmmer",
            "-N",
            "1",
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
            seqfile,
            seqdb,
        ])
        .output()
        .expect("failed to run hmmer jackhmmer");
    assert!(
        output.status.success(),
        "hmmer jackhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_c_jackhmmer_tblout(seqfile: &str, seqdb: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let output = Command::new(test_path("hmmer/src/jackhmmer"))
        .args([
            "-N",
            "1",
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
            seqfile,
            seqdb,
        ])
        .output()
        .expect("failed to run bundled C jackhmmer");
    assert!(
        output.status.success(),
        "bundled C jackhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_jackhmmer_tblout_with_args(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let mut args = vec!["jackhmmer"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--noali",
        "--tblout",
        tblout.to_str().unwrap(),
        seqfile,
        seqdb,
    ]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
        .output()
        .expect("failed to run hmmer jackhmmer");
    assert!(
        output.status.success(),
        "hmmer jackhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_c_jackhmmer_tblout_with_args(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("tblout.txt");
    let mut args = extra_args.to_vec();
    args.extend_from_slice(&[
        "--noali",
        "--tblout",
        tblout.to_str().unwrap(),
        seqfile,
        seqdb,
    ]);
    let output = Command::new(test_path("hmmer/src/jackhmmer"))
        .args(args)
        .output()
        .expect("failed to run bundled C jackhmmer");
    assert!(
        output.status.success(),
        "bundled C jackhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(tblout).unwrap()
}

fn run_jackhmmer_domtblout_with_args(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("domtblout.txt");
    let mut args = vec!["jackhmmer"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--noali",
        "--domtblout",
        domtblout.to_str().unwrap(),
        seqfile,
        seqdb,
    ]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
        .output()
        .expect("failed to run hmmer jackhmmer");
    assert!(
        output.status.success(),
        "hmmer jackhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(domtblout).unwrap()
}

fn run_c_jackhmmer_domtblout_with_args(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("domtblout.txt");
    let mut args = extra_args.to_vec();
    args.extend_from_slice(&[
        "--noali",
        "--domtblout",
        domtblout.to_str().unwrap(),
        seqfile,
        seqdb,
    ]);
    let output = Command::new(test_path("hmmer/src/jackhmmer"))
        .args(args)
        .output()
        .expect("failed to run bundled C jackhmmer");
    assert!(
        output.status.success(),
        "bundled C jackhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(domtblout).unwrap()
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
    run_nhmmscan_tblout_with_args(hmmdb, seqfile, &[])
}

fn run_nhmmscan_tblout_with_args(hmmdb: &str, seqfile: &str, extra_args: &[&str]) -> String {
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
    let mut args = vec!["nhmmscan"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--tblout",
        tblout.to_str().unwrap(),
        hmm_copy.to_str().unwrap(),
        seqfile,
    ]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
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
    run_nhmmscan_dfamtblout_with_args(hmmdb, seqfile, &[])
}

fn run_nhmmscan_dfamtblout_with_args(hmmdb: &str, seqfile: &str, extra_args: &[&str]) -> String {
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
    let mut args = vec!["nhmmscan"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--dfamtblout",
        dfamtblout.to_str().unwrap(),
        hmm_copy.to_str().unwrap(),
        seqfile,
    ]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
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
    run_c_nhmmscan_tblout_with_args(hmmdb, seqfile, &[])
}

fn run_c_nhmmscan_tblout_with_args(hmmdb: &str, seqfile: &str, extra_args: &[&str]) -> String {
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
    let mut args = extra_args.to_vec();
    args.extend_from_slice(&[
        "--tblout",
        tblout.to_str().unwrap(),
        hmm_copy.to_str().unwrap(),
        seqfile,
    ]);
    let output = Command::new(test_path("hmmer/src/nhmmscan"))
        .args(args)
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
    run_c_nhmmscan_dfamtblout_with_args(hmmdb, seqfile, &[])
}

fn run_c_nhmmscan_dfamtblout_with_args(hmmdb: &str, seqfile: &str, extra_args: &[&str]) -> String {
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
    let mut args = extra_args.to_vec();
    args.extend_from_slice(&[
        "--dfamtblout",
        dfamtblout.to_str().unwrap(),
        hmm_copy.to_str().unwrap(),
        seqfile,
    ]);
    let output = Command::new(test_path("hmmer/src/nhmmscan"))
        .args(args)
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
    run_hmmscan_tblout_with_args(hmmdb, seqfile, &[])
}

fn run_hmmscan_tblout_with_args(hmmdb: &str, seqfile: &str, extra_args: &[&str]) -> String {
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
    let mut args = vec!["scan", "--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--tblout",
        tblout.to_str().unwrap(),
        hmm_copy.to_str().unwrap(),
        seqfile,
    ]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
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
    run_c_hmmscan_tblout_with_args(hmmdb, seqfile, &[])
}

fn run_c_hmmscan_tblout_with_args(hmmdb: &str, seqfile: &str, extra_args: &[&str]) -> String {
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
    let mut args = vec!["--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--tblout",
        tblout.to_str().unwrap(),
        hmm_copy.to_str().unwrap(),
        seqfile,
    ]);
    let output = Command::new(test_path("hmmer/src/hmmscan"))
        .args(args)
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
    run_hmmscan_domtblout_with_args(hmmdb, seqfile, &[])
}

fn run_hmmscan_domtblout_with_args(hmmdb: &str, seqfile: &str, extra_args: &[&str]) -> String {
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
    let mut args = vec!["scan", "--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--domtblout",
        domtblout.to_str().unwrap(),
        hmm_copy.to_str().unwrap(),
        seqfile,
    ]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
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
    run_c_hmmscan_domtblout_with_args(hmmdb, seqfile, &[])
}

fn run_c_hmmscan_domtblout_with_args(hmmdb: &str, seqfile: &str, extra_args: &[&str]) -> String {
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
    let mut args = vec!["--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--domtblout",
        domtblout.to_str().unwrap(),
        hmm_copy.to_str().unwrap(),
        seqfile,
    ]);
    let output = Command::new(test_path("hmmer/src/hmmscan"))
        .args(args)
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
    run_hmmscan_pfamtblout_with_args(hmmdb, seqfile, &[])
}

fn run_hmmscan_pfamtblout_with_args(hmmdb: &str, seqfile: &str, extra_args: &[&str]) -> String {
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
    let mut args = vec!["scan", "--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--pfamtblout",
        pfamtblout.to_str().unwrap(),
        hmm_copy.to_str().unwrap(),
        seqfile,
    ]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
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
    run_c_hmmscan_pfamtblout_with_args(hmmdb, seqfile, &[])
}

fn run_c_hmmscan_pfamtblout_with_args(hmmdb: &str, seqfile: &str, extra_args: &[&str]) -> String {
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
    let mut args = vec!["--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--pfamtblout",
        pfamtblout.to_str().unwrap(),
        hmm_copy.to_str().unwrap(),
        seqfile,
    ]);
    let output = Command::new(test_path("hmmer/src/hmmscan"))
        .args(args)
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
    run_hmmsearch_domtblout_with_args(hmm, seqdb, &[])
}

fn run_hmmsearch_domtblout_with_args(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("domtblout.txt");
    let mut args = vec!["search", "--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--domtblout", domtblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
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
    run_c_hmmsearch_domtblout_with_args(hmm, seqdb, &[])
}

fn run_c_hmmsearch_domtblout_with_args(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("domtblout.txt");
    let mut args = vec!["--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--domtblout", domtblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(test_path("hmmer/src/hmmsearch"))
        .args(args)
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
    run_hmmsearch_pfamtblout_with_args(hmm, seqdb, &[])
}

fn run_hmmsearch_pfamtblout_with_args(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let pfamtblout = dir.path().join("pfamtblout.txt");
    let mut args = vec!["search", "--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--pfamtblout", pfamtblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(binary_path("hmmer"))
        .args(args)
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
    run_c_hmmsearch_pfamtblout_with_args(hmm, seqdb, &[])
}

fn run_c_hmmsearch_pfamtblout_with_args(hmm: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let pfamtblout = dir.path().join("pfamtblout.txt");
    let mut args = vec!["--noali"];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&["--pfamtblout", pfamtblout.to_str().unwrap(), hmm, seqdb]);
    let output = Command::new(test_path("hmmer/src/hmmsearch"))
        .args(args)
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
        "# Option settings: nhmmer --dna --tblout /tmp/TMPFILE {} {}",
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
fn test_nhmmer_nonull2_longtarget_tblout_matches_bundled_c_rows() {
    let hmm = test_path("hmmer/tutorial/MADE1.hmm");
    let seqdb = test_path("hmmer/tutorial/dna_target.fa");
    let args = ["--dna", "--nonull2", "--noali"];
    let rust_rows = parse_nhmmer_rows(&run_nhmmer(&hmm, &seqdb, &args).1);
    let c_rows = parse_nhmmer_rows(&run_c_nhmmer_tblout_with_args(&hmm, &seqdb, &args));

    assert!(!c_rows.is_empty(), "bundled C fixture should produce rows");
    assert_eq!(
        rust_rows, c_rows,
        "nhmmer --nonull2 long-target rows diverged from bundled C"
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
fn test_nhmmer_nhmmscan_suppress_runtime_footer_lines() {
    // The non-deterministic run-time footer (`# CPU time:` / `# Mc/sec:`) is
    // suppressed for determinism, matching hmmsearch/phmmer/hmmscan/jackhmmer.
    let hmm = test_path("hmmer/tutorial/MADE1.hmm");
    let target = test_path("hmmer/tutorial/dna_target.fa");

    let nhmmer_stdout = run_nhmmer_stdout(&hmm, &target, &[]);
    assert!(
        !nhmmer_stdout.contains("# CPU time:") && !nhmmer_stdout.contains("# Mc/sec:"),
        "nhmmer must not emit the run-time footer lines:\n{nhmmer_stdout}"
    );

    // nhmmscan over a pressed copy of the same model.
    let dir = tempfile::tempdir().unwrap();
    let ndb = dir.path().join("ndb.hmm");
    std::fs::copy(&hmm, &ndb).unwrap();
    let press = Command::new(binary_path("hmmer"))
        .args(["hmmpress", "-f", ndb.to_str().unwrap()])
        .output()
        .expect("failed to run hmmpress");
    assert!(
        press.status.success(),
        "hmmpress failed: {}",
        String::from_utf8_lossy(&press.stderr)
    );
    let scan = Command::new(binary_path("hmmer"))
        .args(["nhmmscan", ndb.to_str().unwrap(), &target])
        .output()
        .expect("failed to run nhmmscan");
    assert!(
        scan.status.success(),
        "nhmmscan failed: {}",
        String::from_utf8_lossy(&scan.stderr)
    );
    let scan_stdout = String::from_utf8_lossy(&scan.stdout);
    assert!(
        !scan_stdout.contains("# CPU time:") && !scan_stdout.contains("# Mc/sec:"),
        "nhmmscan must not emit the run-time footer lines:\n{scan_stdout}"
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
fn test_nhmmer_trna_ecoli_genome_fasta_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    let seqdb = test_path("external/new_real/dna/GCF_000005845.2_ASM584v2_genomic.fna");

    let rust_rows = parse_nhmmer_rows(&run_nhmmer_tblout(hmm.to_str().unwrap(), &seqdb));
    let c_rows = parse_nhmmer_rows(&run_c_nhmmer_tblout(hmm.to_str().unwrap(), &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "tRNA vs E. coli genome FASTA nhmmer rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 83);
}

#[test]
fn test_nhmmer_trna_ecoli_genome_gzip_watson_dfamtblout_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    let seqdb = test_path("external/new_real/dna/GCF_000005845.2_ASM584v2_genomic.fna.gz");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );

    let rust_rows = parse_dfamtbl_rows(&run_nhmmer_dfamtblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &["--watson"],
    ));
    let c_rows = parse_dfamtbl_rows(&run_c_nhmmer_dfamtblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &["--watson"],
    ));

    assert_eq!(
        rust_rows, c_rows,
        "tRNA vs gzipped E. coli genome nhmmer --watson dfamtblout rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 50);
}

#[test]
fn test_nhmmer_trna_ecoli_genome_fm_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    let rust_fm = dir.path().join("rust.ecoli.hmmerdb");
    let c_fm = dir.path().join("c.ecoli.hmmerdb");
    let seqdb = test_path("external/new_real/dna/GCF_000005845.2_ASM584v2_genomic.fna");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    make_rust_fm_database(&seqdb, &rust_fm);
    make_c_fm_database(&seqdb, &c_fm);

    let rust_rows = parse_nhmmer_rows(&run_nhmmer_tblout(
        hmm.to_str().unwrap(),
        rust_fm.to_str().unwrap(),
    ));
    let c_rows = parse_nhmmer_rows(&run_c_nhmmer_tblout(
        hmm.to_str().unwrap(),
        c_fm.to_str().unwrap(),
    ));

    assert_eq!(
        rust_rows, c_rows,
        "tRNA vs E. coli genome FM-index nhmmer rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 85);
}

#[test]
fn test_makehmmerdb_ecoli_slice_option_output_matches_bundled_c_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let rust_fm = dir.path().join("rust.ecoli250k.hmmerdb");
    let c_fm = dir.path().join("c.ecoli250k.hmmerdb");
    let seqdb = test_path("external/new_real/derived/GCF_000005845.2_ASM584v2_first250k.fna");

    let rust = Command::new(binary_path("hmmer"))
        .args([
            "makehmmerdb",
            "--informat",
            "fasta",
            "--bin_length",
            "128",
            "--sa_freq",
            "4",
            &seqdb,
            rust_fm.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer makehmmerdb with options");
    assert!(
        rust.status.success(),
        "hmmer makehmmerdb with options failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let c = Command::new(test_path("hmmer/src/makehmmerdb"))
        .args([
            "--informat",
            "fasta",
            "--bin_length",
            "128",
            "--sa_freq",
            "4",
            &seqdb,
            c_fm.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run bundled C makehmmerdb with options");
    assert!(
        c.status.success(),
        "bundled C makehmmerdb with options failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    assert_eq!(
        std::fs::read(rust_fm).unwrap(),
        std::fs::read(c_fm).unwrap(),
        "makehmmerdb --bin_length/--sa_freq output bytes diverged from bundled C"
    );
}

#[test]
fn test_makehmmerdb_ecoli_slice_fwd_only_cstream_matches_bundled_c_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let rust_fm = dir.path().join("rust.ecoli250k.fwd.hmmerdb");
    let c_fm = dir.path().join("c.ecoli250k.fwd.hmmerdb");
    let seqdb = test_path("external/new_real/derived/GCF_000005845.2_ASM584v2_first250k.fna");

    let rust = Command::new(binary_path("hmmer"))
        .args([
            "makehmmerdb",
            "--dna",
            "--bin_length",
            "128",
            "--sa_freq",
            "16",
            "--fwd_only",
            "--cstream",
            &seqdb,
            rust_fm.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer makehmmerdb --fwd_only --cstream");
    assert!(
        rust.status.success(),
        "hmmer makehmmerdb --fwd_only --cstream failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let c = Command::new(test_path("hmmer/src/makehmmerdb"))
        .args([
            "--dna",
            "--bin_length",
            "128",
            "--sa_freq",
            "16",
            "--fwd_only",
            &seqdb,
            c_fm.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run bundled C makehmmerdb --fwd_only");
    assert!(
        c.status.success(),
        "bundled C makehmmerdb --fwd_only failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    assert_eq!(
        std::fs::read(rust_fm).unwrap(),
        std::fs::read(c_fm).unwrap(),
        "makehmmerdb --fwd_only --cstream output bytes diverged from bundled C"
    );
}

#[test]
fn test_makehmmerdb_ecoli_genome_gzip_block_options_match_bundled_c_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let rust_fm = dir.path().join("rust.ecoli_genome.hmmerdb");
    let c_fm = dir.path().join("c.ecoli_genome.hmmerdb");
    let seqdb = test_path("external/new_real/dna/GCF_000005845.2_ASM584v2_genomic.fna.gz");

    let rust = Command::new(binary_path("hmmer"))
        .args([
            "makehmmerdb",
            "--dna",
            "--bin_length",
            "256",
            "--sa_freq",
            "4",
            "--block_size",
            "1",
            "--cstream",
            &seqdb,
            rust_fm.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer makehmmerdb on gzipped genome");
    assert!(
        rust.status.success(),
        "hmmer makehmmerdb gzipped genome failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let c = Command::new(test_path("hmmer/src/makehmmerdb"))
        .args([
            "--dna",
            "--bin_length",
            "256",
            "--sa_freq",
            "4",
            "--block_size",
            "1",
            &seqdb,
            c_fm.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run bundled C makehmmerdb on gzipped genome");
    assert!(
        c.status.success(),
        "bundled C makehmmerdb gzipped genome failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    assert_eq!(
        std::fs::read(rust_fm).unwrap(),
        std::fs::read(c_fm).unwrap(),
        "makehmmerdb gzipped-genome block option output bytes diverged from bundled C"
    );
}

#[test]
fn test_makehmmerdb_yeast_gzip_multirecord_matches_bundled_c_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let rust_fm = dir.path().join("rust.yeast.hmmerdb");
    let c_fm = dir.path().join("c.yeast.hmmerdb");
    let seqdb =
        test_path("external/realistic/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz");

    let rust = Command::new(binary_path("hmmer"))
        .args([
            "makehmmerdb",
            "--dna",
            "--bin_length",
            "32",
            "--sa_freq",
            "16",
            &seqdb,
            rust_fm.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer makehmmerdb on gzipped yeast FASTA");
    assert!(
        rust.status.success(),
        "hmmer makehmmerdb gzipped yeast failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let c = Command::new(test_path("hmmer/src/makehmmerdb"))
        .args([
            "--dna",
            "--bin_length",
            "32",
            "--sa_freq",
            "16",
            &seqdb,
            c_fm.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run bundled C makehmmerdb on gzipped yeast FASTA");
    assert!(
        c.status.success(),
        "bundled C makehmmerdb gzipped yeast failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    assert_eq!(
        std::fs::read(rust_fm).unwrap(),
        std::fs::read(c_fm).unwrap(),
        "makehmmerdb should index every record in gzipped multi-record FASTA like bundled C"
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
fn test_new_real_hatpase_metadata_and_serialization_tools_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let hmm = hmm.to_str().unwrap();

    assert_eq!(
        run_hmmer_stdout(&["stat", hmm]),
        run_c_tool_stdout("hmmstat", &[hmm])
    );
    assert_eq!(
        run_hmmer_stdout(&["convert", hmm]),
        run_c_tool_stdout("hmmconvert", &[hmm])
    );
}

#[test]
fn test_realistic_pfam_fetch_and_convert_modes_match_bundled_c() {
    let hmmdb = test_path("external/realistic/pfam/Pfam-A.first12.hmm");
    for key in ["14-3-3", "PF00244.27"] {
        assert_eq!(
            run_hmmer_stdout(&["fetch", &hmmdb, key]),
            run_c_tool_stdout("hmmfetch", &[&hmmdb, key]),
            "hmmfetch {key} from Pfam first12 diverged from bundled C"
        );
    }

    let hmm = test_path("test_data/mapali/20aa-rebuilt.hmm");
    assert_eq!(
        run_hmmer_stdout(&["convert", "-a", &hmm]),
        run_c_tool_stdout("hmmconvert", &["-a", &hmm])
    );
    assert_eq!(
        run_hmmer_stdout(&["convert", "-2", &hmm]),
        run_c_tool_stdout("hmmconvert", &["-2", &hmm])
    );
    assert_eq!(
        run_hmmer_stdout(&["convert", "--outfmt", "3/b", &hmm]),
        run_c_tool_stdout("hmmconvert", &["--outfmt", "3/b", &hmm])
    );
}

#[test]
fn test_realistic_pfam_hmmpress_sidecars_match_bundled_c_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_copy = dir.path().join("Pfam-A.first12.hmm");
    let hmmdb = test_path("external/realistic/pfam/Pfam-A.first12.hmm");
    std::fs::copy(&hmmdb, &hmm_copy).unwrap();

    press_rust_hmmdb(&hmm_copy);
    let rust_sidecars = ["h3m", "h3i", "h3f", "h3p"].map(|suffix| {
        let path = hmm_copy.with_extension(format!("hmm.{suffix}"));
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        (suffix, bytes)
    });
    press_c_hmmdb(&hmm_copy);

    for (suffix, rust_bytes) in rust_sidecars {
        let c_sidecar = hmm_copy.with_extension(format!("hmm.{suffix}"));
        let c_bytes = std::fs::read(&c_sidecar).unwrap();
        assert_eq!(
            rust_bytes, c_bytes,
            "hmmpress sidecar .{suffix} diverged from bundled C"
        );
    }
}

#[test]
fn test_realistic_dnaj_hmmbuild_and_hmmlogo_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let rust_hmm = dir.path().join("rust_DnaJ.hmm");
    let c_hmm = dir.path().join("c_DnaJ.hmm");
    let msa = test_path("external/realistic/pfam/PF00226_DnaJ.seed.sto");
    build_rust_hmm(&["--amino"], &rust_hmm, &msa);
    build_c_hmm(&["--amino"], &c_hmm, &msa);

    assert_eq!(hmm_without_date(&rust_hmm), hmm_without_date(&c_hmm));
    assert_eq!(
        run_hmmer_stdout(&["logo", rust_hmm.to_str().unwrap()]),
        run_c_tool_stdout("hmmlogo", &[rust_hmm.to_str().unwrap()])
    );
}

#[test]
fn test_realistic_dnaj_hmmbuild_name_summary_matches_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let rust_hmm = dir.path().join("rust_DnaJ_named.hmm");
    let c_hmm = dir.path().join("c_DnaJ_named.hmm");
    let msa = test_path("external/realistic/pfam/PF00226_DnaJ.seed.sto");

    let rust = Command::new(binary_path("hmmer"))
        .args([
            "build",
            "-n",
            "DnaJ_audit",
            rust_hmm.to_str().unwrap(),
            &msa,
        ])
        .output()
        .expect("failed to run hmmer build -n");
    assert!(
        rust.status.success(),
        "hmmer build -n failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );
    let c = Command::new(test_path("hmmer/src/hmmbuild"))
        .args(["-n", "DnaJ_audit", c_hmm.to_str().unwrap(), &msa])
        .output()
        .expect("failed to run bundled C hmmbuild -n");
    assert!(
        c.status.success(),
        "bundled C hmmbuild -n failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    let rust_stdout = String::from_utf8(rust.stdout).unwrap();
    assert!(rust_stdout.contains("# name (the single) HMM:            DnaJ_audit"));
    assert_eq!(
        normalize_hmmbuild_summary(&rust_stdout),
        normalize_hmmbuild_summary(&String::from_utf8(c.stdout).unwrap())
    );
    assert_eq!(hmm_without_date(&rust_hmm), hmm_without_date(&c_hmm));
}

#[test]
fn test_realistic_dnaj_gzip_hmm_readers_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("DnaJ.hmm");
    let gz_hmm = dir.path().join("DnaJ.hmm.gz");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/realistic/pfam/PF00226_DnaJ.seed.sto"),
    );
    {
        let input = std::fs::File::open(&hmm).unwrap();
        let output = std::fs::File::create(&gz_hmm).unwrap();
        let mut encoder = flate2::write::GzEncoder::new(output, flate2::Compression::default());
        std::io::copy(&mut std::io::BufReader::new(input), &mut encoder).unwrap();
        encoder.finish().unwrap();
    }
    let gz_hmm = gz_hmm.to_str().unwrap();

    assert_eq!(
        run_hmmer_stdout(&["fetch", gz_hmm, "DnaJ"]),
        run_c_tool_stdout("hmmfetch", &[gz_hmm, "DnaJ"])
    );
    assert_eq!(
        run_hmmer_stdout(&["stat", gz_hmm]),
        run_c_tool_stdout("hmmstat", &[gz_hmm])
    );
    assert_eq!(
        run_hmmer_stdout(&["convert", gz_hmm]),
        run_c_tool_stdout("hmmconvert", &[gz_hmm])
    );
}

#[test]
fn test_realistic_full_pfam_gzip_hmmstat_matches_bundled_c_for_edge_rounding_row() {
    let hmmdb = test_path("external/realistic/pfam/Pfam-A.hmm.gz");
    let rust = run_hmmer_stdout(&["stat", &hmmdb]);
    let c = run_c_tool_stdout("hmmstat", &[&hmmdb]);
    let find_row = |output: &str| {
        output
            .lines()
            .find(|line| line.contains("Zn2Cys6-like"))
            .unwrap()
            .to_string()
    };

    assert_eq!(find_row(&rust), find_row(&c));
    assert!(find_row(&rust).contains("  0.52   0.05"));
}

#[test]
fn test_realistic_dnaj_hmmsim_msv_matches_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("DnaJ.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/realistic/pfam/PF00226_DnaJ.seed.sto"),
    );
    let hmm = hmm.to_str().unwrap();

    assert_eq!(
        output_without_cpu_footer(&run_hmmer_stdout(&[
            "sim", "--seed", "42", "-N", "20", "--msv", hmm
        ])),
        output_without_cpu_footer(&run_c_tool_stdout(
            "hmmsim",
            &["--seed", "42", "-N", "20", "--msv", hmm]
        ))
    );
}

#[test]
fn test_realistic_dnaj_hmmsim_vit_fwd_hyb_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("DnaJ.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/realistic/pfam/PF00226_DnaJ.seed.sto"),
    );
    let hmm = hmm.to_str().unwrap();

    for mode in ["--vit", "--fwd", "--hyb"] {
        assert_eq!(
            output_without_cpu_footer(&run_hmmer_stdout(&[
                "sim", "--seed", "42", "-N", "1000", mode, hmm
            ])),
            output_without_cpu_footer(&run_c_tool_stdout(
                "hmmsim",
                &["--seed", "42", "-N", "1000", mode, hmm]
            )),
            "DnaJ hmmsim {mode} output diverged from bundled C"
        );
    }
}

#[test]
fn test_hatpase_hmmsim_forward_small_tail_mass_matches_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let hmm = hmm.to_str().unwrap();

    assert_eq!(
        output_without_cpu_footer(&run_hmmer_stdout(&[
            "sim", "--seed", "21", "-N", "120", "-L", "130", "--fwd", "--bgflat", hmm
        ])),
        output_without_cpu_footer(&run_c_tool_stdout(
            "hmmsim",
            &["--seed", "21", "-N", "120", "-L", "130", "--fwd", "--bgflat", hmm]
        )),
        "HATPase_c hmmsim --fwd --bgflat small-tail output diverged from bundled C"
    );
}

#[test]
fn test_dnaj_hmmsim_forward_bgcomp_no_lengthmodel_artifacts_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("DnaJ.hmm");
    let rust_out = dir.path().join("rust.out");
    let rust_p = dir.path().join("rust.p");
    let rust_e = dir.path().join("rust.e");
    let rust_f = dir.path().join("rust.f");
    let c_out = dir.path().join("c.out");
    let c_p = dir.path().join("c.p");
    let c_e = dir.path().join("c.e");
    let c_f = dir.path().join("c.f");
    build_rust_hmm(
        &[
            "--amino",
            "--wgsc",
            "--eclust",
            "--eid",
            "0.7",
            "--plaplace",
        ],
        &hmm,
        &test_path("external/realistic/pfam/PF00226_DnaJ.seed.sto"),
    );

    let common = [
        "--seed",
        "31",
        "--fwd",
        "--bgcomp",
        "--x-no-lengthmodel",
        "-N",
        "300",
        "-L",
        "85",
    ];
    let mut rust_args = vec!["sim"];
    rust_args.extend_from_slice(&common);
    rust_args.extend_from_slice(&[
        "--pfile",
        rust_p.to_str().unwrap(),
        "--efile",
        rust_e.to_str().unwrap(),
        "--ffile",
        rust_f.to_str().unwrap(),
        "-o",
        rust_out.to_str().unwrap(),
        hmm.to_str().unwrap(),
    ]);
    let rust = Command::new(binary_path("hmmer"))
        .args(rust_args)
        .output()
        .expect("failed to run hmmer sim");
    assert!(
        rust.status.success(),
        "hmmer sim failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let mut c_args = common.to_vec();
    c_args.extend_from_slice(&[
        "--pfile",
        c_p.to_str().unwrap(),
        "--efile",
        c_e.to_str().unwrap(),
        "--ffile",
        c_f.to_str().unwrap(),
        "-o",
        c_out.to_str().unwrap(),
        hmm.to_str().unwrap(),
    ]);
    let c = Command::new(test_path("hmmer/src/hmmsim"))
        .args(c_args)
        .output()
        .expect("failed to run bundled C hmmsim");
    assert!(
        c.status.success(),
        "bundled C hmmsim failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    assert_eq!(
        std::fs::read_to_string(&rust_out).unwrap(),
        std::fs::read_to_string(&c_out).unwrap(),
        "hmmsim --fwd --bgcomp --x-no-lengthmodel summary diverged from bundled C"
    );
    assert_eq!(
        std::fs::read_to_string(&rust_f).unwrap(),
        std::fs::read_to_string(&c_f).unwrap(),
        "Forward --ffile should be empty like bundled C"
    );
    assert_hmm_text_matches_with_float_tolerance(
        &std::fs::read_to_string(&rust_p).unwrap(),
        &std::fs::read_to_string(&c_p).unwrap(),
        1.0e-4,
        "hmmsim Forward pfile",
    );
    assert_hmm_text_matches_with_float_tolerance(
        &std::fs::read_to_string(&rust_e).unwrap(),
        &std::fs::read_to_string(&c_e).unwrap(),
        5.0e-3,
        "hmmsim Forward efile",
    );
}

#[test]
fn test_realistic_dnaj_hmmsearch_yeast_proteome_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("DnaJ.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/realistic/pfam/PF00226_DnaJ.seed.sto"),
    );
    let seqdb = test_path("external/realistic/protein/uniprot_UP000002311_yeast.fasta");

    let rust_rows = parse_hmmsearch_rows(&run_hmmsearch_tblout(hmm.to_str().unwrap(), &seqdb));
    let c_rows = parse_hmmsearch_rows(&run_c_hmmsearch_tblout(hmm.to_str().unwrap(), &seqdb));
    assert_eq!(
        rust_rows, c_rows,
        "DnaJ vs yeast proteome hmmsearch rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 22);

    let rust_domains = parse_domtbl_rows(&run_hmmsearch_domtblout(hmm.to_str().unwrap(), &seqdb));
    let c_domains = parse_domtbl_rows(&run_c_hmmsearch_domtblout(hmm.to_str().unwrap(), &seqdb));
    assert_eq!(
        rust_domains, c_domains,
        "DnaJ vs yeast proteome hmmsearch domain rows diverged from bundled C"
    );
    assert_eq!(rust_domains.len(), 32);
}

#[test]
fn test_realistic_dnaj_hmmsearch_max_yeast_proteome_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("DnaJ.hmm");
    let tblout = dir.path().join("rust.tblout");
    let c_tblout = dir.path().join("c.tblout");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/realistic/pfam/PF00226_DnaJ.seed.sto"),
    );
    let seqdb = test_path("external/realistic/protein/uniprot_UP000002311_yeast.fasta");

    let rust = Command::new(binary_path("hmmer"))
        .args([
            "search",
            "--max",
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
            hmm.to_str().unwrap(),
            &seqdb,
        ])
        .output()
        .expect("failed to run hmmer search --max");
    assert!(
        rust.status.success(),
        "hmmer search --max failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );
    let c = Command::new(test_path("hmmer/src/hmmsearch"))
        .args([
            "--max",
            "--noali",
            "--tblout",
            c_tblout.to_str().unwrap(),
            hmm.to_str().unwrap(),
            &seqdb,
        ])
        .output()
        .expect("failed to run bundled C hmmsearch --max");
    assert!(
        c.status.success(),
        "bundled C hmmsearch --max failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    let rust_rows = parse_hmmsearch_rows(&std::fs::read_to_string(tblout).unwrap());
    let c_rows = parse_hmmsearch_rows(&std::fs::read_to_string(c_tblout).unwrap());
    assert_eq!(
        rust_rows, c_rows,
        "DnaJ vs yeast proteome hmmsearch --max rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 73);
}

#[test]
fn test_realistic_dnaj_hmmscan_yeast_proteome_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("DnaJ.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/realistic/pfam/PF00226_DnaJ.seed.sto"),
    );
    let seqdb = test_path("external/realistic/protein/uniprot_UP000002311_yeast.fasta");

    let rust_rows = parse_hmmsearch_rows(&run_hmmscan_tblout(hmm.to_str().unwrap(), &seqdb));
    let c_rows = parse_hmmsearch_rows(&run_c_hmmscan_tblout(hmm.to_str().unwrap(), &seqdb));
    assert_eq!(
        rust_rows, c_rows,
        "DnaJ vs yeast proteome hmmscan rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 22);
}

#[test]
fn test_realistic_pfam_panel_yeast_proteome_matches_bundled_c_rows() {
    let hmmdb = test_path("external/realistic/pfam/Pfam-A.first12.hmm");
    let seqdb = test_path("external/realistic/protein/uniprot_UP000002311_yeast.fasta");

    let rust_search_rows = parse_hmmsearch_rows(&run_hmmsearch_tblout(&hmmdb, &seqdb));
    let c_search_rows = parse_hmmsearch_rows(&run_c_hmmsearch_tblout(&hmmdb, &seqdb));
    assert_eq!(
        rust_search_rows, c_search_rows,
        "Pfam first12 vs yeast proteome hmmsearch rows diverged from bundled C"
    );
    assert_eq!(rust_search_rows.len(), 14);

    let rust_search_domains = parse_domtbl_rows(&run_hmmsearch_domtblout(&hmmdb, &seqdb));
    let c_search_domains = parse_domtbl_rows(&run_c_hmmsearch_domtblout(&hmmdb, &seqdb));
    assert_eq!(
        rust_search_domains, c_search_domains,
        "Pfam first12 vs yeast proteome hmmsearch domain rows diverged from bundled C"
    );
    assert_eq!(rust_search_domains.len(), 16);

    let rust_scan_rows = parse_hmmsearch_rows(&run_hmmscan_tblout(&hmmdb, &seqdb));
    let c_scan_rows = parse_hmmsearch_rows(&run_c_hmmscan_tblout(&hmmdb, &seqdb));
    assert_eq!(
        rust_scan_rows, c_scan_rows,
        "Pfam first12 vs yeast proteome hmmscan rows diverged from bundled C"
    );
    assert_eq!(rust_scan_rows.len(), 14);

    let rust_scan_domains = parse_domtbl_rows(&run_hmmscan_domtblout(&hmmdb, &seqdb));
    let c_scan_domains = parse_domtbl_rows(&run_c_hmmscan_domtblout(&hmmdb, &seqdb));
    assert_eq!(
        rust_scan_domains, c_scan_domains,
        "Pfam first12 vs yeast proteome hmmscan domain rows diverged from bundled C"
    );
    assert_eq!(rust_scan_domains.len(), 16);
}

#[test]
fn test_realistic_pfam_panel_hmmsearch_acc_matches_bundled_c_rows() {
    let hmmdb = test_path("external/realistic/pfam/Pfam-A.first12.hmm");
    let seqdb = test_path("external/realistic/protein/uniprot_UP000002311_yeast.fasta");

    let rust_rows =
        parse_hmmsearch_rows(&run_hmmsearch_tblout_with_args(&hmmdb, &seqdb, &["--acc"]));
    let c_rows = parse_hmmsearch_rows(&run_c_hmmsearch_tblout_with_args(
        &hmmdb,
        &seqdb,
        &["--acc"],
    ));
    assert_eq!(
        rust_rows, c_rows,
        "Pfam first12 --acc hmmsearch rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 14);
}

#[test]
fn test_realistic_yeast_first_protein_phmmer_and_jackhmmer_match_bundled_c_rows() {
    let query = test_path("external/realistic/queries/yeast_first_protein.fa");
    let seqdb = test_path("external/realistic/protein/uniprot_UP000002311_yeast.fasta");

    let rust_phmmer = parse_hmmsearch_rows(&run_phmmer_tblout(&query, &seqdb));
    let c_phmmer = parse_hmmsearch_rows(&run_c_phmmer_tblout_real(&query, &seqdb));
    assert_eq!(
        rust_phmmer, c_phmmer,
        "yeast first protein phmmer rows diverged from bundled C"
    );
    assert_eq!(rust_phmmer.len(), 46);

    let rust_jackhmmer = parse_hmmsearch_rows(&run_jackhmmer_tblout(&query, &seqdb));
    let c_jackhmmer = parse_hmmsearch_rows(&run_c_jackhmmer_tblout(&query, &seqdb));
    assert_eq!(
        rust_jackhmmer, c_jackhmmer,
        "yeast first protein jackhmmer rows diverged from bundled C"
    );
    assert_eq!(rust_jackhmmer.len(), 46);
}

#[test]
fn test_realistic_dnak_phmmer_acc_domtblout_matches_bundled_c_rows() {
    let query = test_path("external/new_real/queries/sp_P0A6Y8_DNAK_ECOLI.fa");
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta");

    let rust_rows = parse_domtbl_rows(&run_phmmer_domtblout_with_args(&query, &seqdb, &["--acc"]));
    let c_rows = parse_domtbl_rows(&run_c_phmmer_domtblout_with_args(
        &query,
        &seqdb,
        &["--acc"],
    ));
    assert_eq!(
        rust_rows, c_rows,
        "DnaK --acc phmmer domtblout rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 8);
}

#[test]
fn test_realistic_trna_nhmmer_yeast_genome_tblouts_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/realistic/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    let seqdb =
        test_path("external/realistic/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa");

    let rust_rows = parse_nhmmer_rows(&run_nhmmer_tblout(hmm.to_str().unwrap(), &seqdb));
    let c_rows = parse_nhmmer_rows(&run_c_nhmmer_tblout(hmm.to_str().unwrap(), &seqdb));
    assert_eq!(
        rust_rows, c_rows,
        "tRNA vs yeast genome nhmmer rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 214);

    let rust_dfam = parse_dfamtbl_rows(&run_nhmmer_dfamtblout(hmm.to_str().unwrap(), &seqdb));
    let c_dfam = parse_dfamtbl_rows(&run_c_nhmmer_dfamtblout(hmm.to_str().unwrap(), &seqdb));
    assert_eq!(
        rust_dfam, c_dfam,
        "tRNA vs yeast genome nhmmer dfamtblout rows diverged from bundled C"
    );
    assert_eq!(rust_dfam.len(), 214);
}

#[test]
#[ignore = "slow full-yeast FM-index parity regression"]
fn test_realistic_trna_nhmmer_yeast_genome_fm_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    let rust_fm = dir.path().join("rust.yeast.hmmerdb");
    let c_fm = dir.path().join("c.yeast.hmmerdb");
    let seqdb =
        test_path("external/realistic/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/realistic/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    make_rust_fm_database(&seqdb, &rust_fm);
    make_c_fm_database(&seqdb, &c_fm);

    let rust_rows = parse_nhmmer_rows(&run_nhmmer_tblout(
        hmm.to_str().unwrap(),
        rust_fm.to_str().unwrap(),
    ));
    let c_rows = parse_nhmmer_rows(&run_c_nhmmer_tblout(
        hmm.to_str().unwrap(),
        c_fm.to_str().unwrap(),
    ));

    assert_eq!(
        rust_rows, c_rows,
        "tRNA vs yeast genome FM-index nhmmer rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 219);
}

#[test]
fn test_realistic_trna_nhmmscan_yeast_genome_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/realistic/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    let seqdb =
        test_path("external/realistic/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa");

    let rust_rows = parse_nhmmer_rows(&run_nhmmscan_tblout(hmm.to_str().unwrap(), &seqdb));
    let c_rows = parse_nhmmer_rows(&run_c_nhmmscan_tblout(hmm.to_str().unwrap(), &seqdb));
    assert_eq!(
        rust_rows, c_rows,
        "tRNA vs yeast genome nhmmscan rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 226);

    let rust_dfam = parse_dfamtbl_rows(&run_nhmmscan_dfamtblout(hmm.to_str().unwrap(), &seqdb));
    let c_dfam = parse_dfamtbl_rows(&run_c_nhmmscan_dfamtblout(hmm.to_str().unwrap(), &seqdb));
    assert_eq!(
        rust_dfam, c_dfam,
        "tRNA vs yeast genome nhmmscan dfamtblout rows diverged from bundled C"
    );
    assert_eq!(rust_dfam.len(), 226);
}

#[test]
fn test_realistic_trna_hmmpress_sidecars_match_bundled_c_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/realistic/rfam/RF00005_tRNA.dna.seed.sto"),
    );

    press_rust_hmmdb(&hmm);
    let rust_sidecars = ["h3m", "h3i", "h3f", "h3p"].map(|suffix| {
        let path = hmm.with_extension(format!("hmm.{suffix}"));
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        (suffix, bytes)
    });
    press_c_hmmdb(&hmm);

    for (suffix, rust_bytes) in rust_sidecars {
        let c_sidecar = hmm.with_extension(format!("hmm.{suffix}"));
        let c_bytes = std::fs::read(&c_sidecar).unwrap();
        assert_eq!(
            rust_bytes, c_bytes,
            "tRNA hmmpress sidecar .{suffix} diverged from bundled C"
        );
    }
}

#[test]
fn test_realistic_dnaj_emit_modes_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("DnaJ.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/realistic/pfam/PF00226_DnaJ.seed.sto"),
    );
    let hmm = hmm.to_str().unwrap();

    assert_eq!(
        run_hmmer_stdout(&["emit", "--seed", "7", "-c", hmm]),
        run_c_tool_stdout("hmmemit", &["--seed", "7", "-c", hmm])
    );
    assert_eq!(
        run_hmmer_stdout(&["emit", "--seed", "7", "-N", "3", hmm]),
        run_c_tool_stdout("hmmemit", &["--seed", "7", "-N", "3", hmm])
    );
}

#[test]
fn test_new_real_trna_hmmbuild_matches_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let rust_hmm = dir.path().join("rust_RF00005_tRNA.hmm");
    let c_hmm = dir.path().join("c_RF00005_tRNA.hmm");
    let msa = test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto");
    build_rust_hmm(&["--dna"], &rust_hmm, &msa);
    build_c_hmm(&["--dna"], &c_hmm, &msa);

    assert_eq!(hmm_without_date(&rust_hmm), hmm_without_date(&c_hmm));
}

#[test]
fn test_new_real_trna_hmmbuild_option_modes_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let msa = test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto");
    for (idx, opts) in [
        ["--dna", "--hand", "--wgiven", "--enone"].as_slice(),
        [
            "--dna",
            "--fast",
            "--symfrac",
            "0.7",
            "--wnone",
            "--eset",
            "12",
        ]
        .as_slice(),
    ]
    .iter()
    .enumerate()
    {
        let rust_hmm = dir.path().join(format!("rust_tRNA_opts_{idx}.hmm"));
        let c_hmm = dir.path().join(format!("c_tRNA_opts_{idx}.hmm"));
        build_rust_hmm(opts, &rust_hmm, &msa);
        build_c_hmm(opts, &c_hmm, &msa);

        assert_eq!(
            hmm_without_date(&rust_hmm),
            hmm_without_date(&c_hmm),
            "tRNA hmmbuild option set {idx} diverged from bundled C"
        );
    }
}

#[test]
fn test_hatpase_hmmbuild_entropy_weighted_options_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let rust_hmm = dir.path().join("rust_HATPase_weighted.hmm");
    let c_hmm = dir.path().join("c_HATPase_weighted.hmm");
    let msa = test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto");
    let opts = [
        "--amino",
        "--fast",
        "--symfrac",
        "0.58",
        "--wpb",
        "--eentexp",
        "--fragthresh",
        "0.45",
        "--maxinsertlen",
        "5",
        "--EmN",
        "25",
        "--EvN",
        "25",
        "--EfN",
        "25",
    ];
    build_rust_hmm(&opts, &rust_hmm, &msa);
    build_c_hmm(&opts, &c_hmm, &msa);

    assert_hmm_text_matches_with_float_tolerance(
        &hmm_without_date_or_com(&rust_hmm),
        &hmm_without_date_or_com(&c_hmm),
        1.1e-5,
        "HATPase hmmbuild weighted option path should serialize like bundled C",
    );
}

#[test]
fn test_hatpase_hmmbuild_resaved_msa_matches_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let rust_hmm = dir.path().join("rust_HATPase.hmm");
    let c_hmm = dir.path().join("c_HATPase.hmm");
    let rust_sto = dir.path().join("rust_resaved.sto");
    let c_sto = dir.path().join("c_resaved.sto");
    let msa = test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto");

    let rust = Command::new(binary_path("hmmer"))
        .args([
            "build",
            "--amino",
            "--wnone",
            "--enone",
            "-O",
            rust_sto.to_str().unwrap(),
            rust_hmm.to_str().unwrap(),
            &msa,
        ])
        .output()
        .expect("failed to run hmmer build -O");
    assert!(
        rust.status.success(),
        "hmmer build -O failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let c = Command::new(test_path("hmmer/src/hmmbuild"))
        .args([
            "--amino",
            "--wnone",
            "--enone",
            "-O",
            c_sto.to_str().unwrap(),
            c_hmm.to_str().unwrap(),
            &msa,
        ])
        .output()
        .expect("failed to run bundled C hmmbuild -O");
    assert!(
        c.status.success(),
        "bundled C hmmbuild -O failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    assert_eq!(hmm_without_date(&rust_hmm), hmm_without_date(&c_hmm));
    assert_eq!(
        std::fs::read_to_string(rust_sto).unwrap(),
        std::fs::read_to_string(c_sto).unwrap(),
        "hmmbuild -O resaved MSA should match bundled C"
    );
}

#[test]
fn test_response_reg_hmmbuild_resaved_blosum_weights_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let rust_hmm = dir.path().join("rust_Response_reg.hmm");
    let c_hmm = dir.path().join("c_Response_reg.hmm");
    let rust_sto = dir.path().join("rust_resaved.sto");
    let c_sto = dir.path().join("c_resaved.sto");
    let msa = test_path("external/new_real/pfam/PF00072_Response_reg.seed.sto");
    let opts = [
        "--amino",
        "--wblosum",
        "--wid",
        "0.50",
        "--eclust",
        "--eid",
        "0.55",
        "--plaplace",
        "--maxinsertlen",
        "5",
    ];

    let mut rust_args = vec!["build"];
    rust_args.extend_from_slice(&opts);
    rust_args.extend_from_slice(&[
        "-O",
        rust_sto.to_str().unwrap(),
        rust_hmm.to_str().unwrap(),
        &msa,
    ]);
    let rust = Command::new(binary_path("hmmer"))
        .args(rust_args)
        .output()
        .expect("failed to run hmmer build -O with BLOSUM weights");
    assert!(
        rust.status.success(),
        "hmmer build -O failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let mut c_args = opts.to_vec();
    c_args.extend_from_slice(&["-O", c_sto.to_str().unwrap(), c_hmm.to_str().unwrap(), &msa]);
    let c = Command::new(test_path("hmmer/src/hmmbuild"))
        .args(c_args)
        .output()
        .expect("failed to run bundled C hmmbuild -O with BLOSUM weights");
    assert!(
        c.status.success(),
        "bundled C hmmbuild -O failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    assert_eq!(
        std::fs::read_to_string(rust_sto).unwrap(),
        std::fs::read_to_string(c_sto).unwrap(),
        "hmmbuild -O should emit C-matching #=GS WT rows after BLOSUM weighting"
    );
}

#[test]
fn test_new_real_dnak_singlemx_hmmbuild_preserves_bundled_c_query_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let rust_hmm = dir.path().join("rust_DnaK_singlemx.hmm");
    let c_hmm = dir.path().join("c_DnaK_singlemx.hmm");
    let query = test_path("external/new_real/queries/sp_P0A6Y8_DNAK_ECOLI.fa");
    let opts = [
        "--amino",
        "--informat",
        "afa",
        "--singlemx",
        "--mx",
        "BLOSUM45",
        "--popen",
        "0.03",
        "--pextend",
        "0.2",
    ];
    build_rust_hmm(&opts, &rust_hmm, &query);
    build_c_hmm(&opts, &c_hmm, &query);

    let rust = hmm_without_date(&rust_hmm);
    let c = hmm_without_date(&c_hmm);
    for line in [
        "NAME  sp|P0A6Y8|DNAK_ECOLI",
        "COM   [1] [HMM created from a query sequence]",
    ] {
        assert!(
            rust.contains(line),
            "Rust singlemx HMM is missing C metadata line {line:?}:\n{rust}"
        );
        assert!(c.contains(line));
    }
}

#[test]
fn test_new_real_pfam_gz_first_seed_utility_modes_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let seed = dir.path().join("pfam_first_03009_C.sto");
    first_stockholm_from_gzip(&test_path("external/new_real/pfam/Pfam-A.seed.gz"), &seed);
    let rust_hmm = dir.path().join("rust_03009_C.hmm");
    let c_hmm = dir.path().join("c_03009_C.hmm");
    let opts = ["--amino", "--wnone", "--enone", "--symfrac", "0.35"];
    build_rust_hmm(&opts, &rust_hmm, seed.to_str().unwrap());
    build_c_hmm(&opts, &c_hmm, seed.to_str().unwrap());

    assert_eq!(hmm_without_date(&rust_hmm), hmm_without_date(&c_hmm));
    assert_eq!(
        run_hmmer_stdout(&["stat", rust_hmm.to_str().unwrap()]),
        run_c_tool_stdout("hmmstat", &[c_hmm.to_str().unwrap()])
    );
    assert_eq!(
        run_hmmer_stdout(&[
            "logo",
            "--height_score",
            "--no_indel",
            rust_hmm.to_str().unwrap()
        ]),
        run_c_tool_stdout(
            "hmmlogo",
            &["--height_score", "--no_indel", c_hmm.to_str().unwrap()]
        )
    );
    assert_eq!(
        run_hmmer_stdout(&[
            "emit",
            "-C",
            "--minl",
            "0.25",
            "--minu",
            "0.85",
            rust_hmm.to_str().unwrap()
        ]),
        run_c_tool_stdout(
            "hmmemit",
            &[
                "-C",
                "--minl",
                "0.25",
                "--minu",
                "0.85",
                c_hmm.to_str().unwrap()
            ]
        )
    );
}

#[test]
fn test_new_real_hatpase_press_and_fetch_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    let rust_db = dir.path().join("rust.PF02518_HATPase_c.hmm");
    let c_db = dir.path().join("c.PF02518_HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    std::fs::copy(&hmm, &rust_db).unwrap();
    std::fs::copy(&hmm, &c_db).unwrap();
    press_rust_hmmdb(&rust_db);
    press_c_hmmdb(&c_db);

    assert_eq!(
        run_hmmer_stdout(&["fetch", rust_db.to_str().unwrap(), "HATPase_c"]),
        run_c_tool_stdout("hmmfetch", &[c_db.to_str().unwrap(), "HATPase_c"])
    );
}

#[test]
fn test_new_real_hatpase_emit_and_align_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    let emitted = dir.path().join("emitted.fa");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let hmm = hmm.to_str().unwrap();
    let rust_emit = run_hmmer_stdout(&["emit", "--seed", "42", "-N", "5", hmm]);
    let c_emit = run_c_tool_stdout("hmmemit", &["--seed", "42", "-N", "5", hmm]);
    assert_eq!(rust_emit, c_emit);
    std::fs::write(&emitted, rust_emit).unwrap();

    assert_eq!(
        run_hmmer_stdout(&["align", hmm, emitted.to_str().unwrap()]),
        run_c_tool_stdout("hmmalign", &[hmm, emitted.to_str().unwrap()])
    );
}

#[test]
fn test_new_real_hatpase_hmmalign_output_formats_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    let emitted = dir.path().join("emitted.fa");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let hmm = hmm.to_str().unwrap();
    let emitted_sequences = run_hmmer_stdout(&["emit", "--seed", "23", "-N", "3", hmm]);
    std::fs::write(&emitted, emitted_sequences).unwrap();

    for format in ["afa", "a2m", "pfam", "clustal"] {
        assert_eq!(
            run_hmmer_stdout(&[
                "align",
                "--outformat",
                format,
                hmm,
                emitted.to_str().unwrap()
            ]),
            run_c_tool_stdout(
                "hmmalign",
                &["--outformat", format, hmm, emitted.to_str().unwrap()]
            ),
            "HATPase_c hmmalign --outformat {format} diverged from bundled C"
        );
    }
}

#[test]
fn test_response_reg_hmmalign_ecoli_first50_matches_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("Response_reg.hmm");
    let seqs = dir.path().join("ecoli_first50.fa");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF00072_Response_reg.seed.sto"),
    );
    write_first_fasta_records(
        &seqs,
        "external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta",
        50,
    );
    let hmm = hmm.to_str().unwrap();
    let seqs = seqs.to_str().unwrap();

    assert_eq!(
        run_hmmer_stdout(&["align", "--outformat", "afa", hmm, seqs]),
        run_c_tool_stdout("hmmalign", &["--outformat", "afa", hmm, seqs]),
        "Response_reg hmmalign AFA output should match bundled C on weak E. coli alignments"
    );
    assert_eq!(
        run_hmmer_stdout(&["align", "--trim", "--outformat", "pfam", hmm, seqs]),
        run_c_tool_stdout(
            "hmmalign",
            &["--trim", "--outformat", "pfam", hmm, seqs],
        ),
        "Response_reg hmmalign trimmed Pfam output should match bundled C on weak E. coli alignments"
    );
}

#[test]
fn test_hatpase_hmmalign_trim_mapali_matches_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    let emitted = dir.path().join("emitted.fa");
    let mapali = test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto");
    build_rust_hmm(&["--amino"], &hmm, &mapali);
    let hmm = hmm.to_str().unwrap();
    let emitted_sequences = run_hmmer_stdout(&["emit", "--seed", "29", "-N", "3", hmm]);
    std::fs::write(&emitted, emitted_sequences).unwrap();

    assert_eq!(
        run_hmmer_stdout(&[
            "align",
            "--trim",
            "--mapali",
            &mapali,
            hmm,
            emitted.to_str().unwrap(),
        ]),
        run_c_tool_stdout(
            "hmmalign",
            &[
                "--trim",
                "--mapali",
                &mapali,
                hmm,
                emitted.to_str().unwrap(),
            ],
        ),
        "hmmalign --trim --mapali should match bundled C insertion placement"
    );
}

#[test]
fn test_new_real_hatpase_emit_logo_convert_option_modes_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let hmm = hmm.to_str().unwrap();

    for args in [
        ["emit", "--seed", "9", "-a", "-N", "4", hmm].as_slice(),
        ["emit", "--seed", "9", "-p", "-L", "80", "-N", "4", hmm].as_slice(),
        ["emit", "-C", "--minl", "0.4", "--minu", "0.9", hmm].as_slice(),
        ["logo", "--height_score", hmm].as_slice(),
        ["logo", "--height_relent_all", "--no_indel", hmm].as_slice(),
        ["convert", "-a", hmm].as_slice(),
    ] {
        let c_tool = match args[0] {
            "emit" => "hmmemit",
            "logo" => "hmmlogo",
            "convert" => "hmmconvert",
            _ => unreachable!(),
        };
        assert_eq!(
            run_hmmer_stdout(args),
            run_c_tool_stdout(c_tool, &args[1..]),
            "HATPase_c {:?} output diverged from bundled C",
            args
        );
    }
}

#[test]
fn test_new_real_response_reg_build_search_scan_emit_sim_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let rust_hmm = dir.path().join("rust_Response_reg.hmm");
    let c_hmm = dir.path().join("c_Response_reg.hmm");
    let rust_fast_hmm = dir.path().join("rust_Response_reg_fast.hmm");
    let c_fast_hmm = dir.path().join("c_Response_reg_fast.hmm");
    let msa = test_path("external/new_real/pfam/PF00072_Response_reg.seed.sto");
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta");

    build_rust_hmm(&["--amino", "--wnone", "--enone"], &rust_hmm, &msa);
    build_c_hmm(&["--amino", "--wnone", "--enone"], &c_hmm, &msa);
    assert_eq!(
        hmm_without_date_or_com(&rust_hmm),
        hmm_without_date_or_com(&c_hmm)
    );

    build_rust_hmm(
        &["--amino", "--fast", "--symfrac", "0.62", "--wpb", "--enone"],
        &rust_fast_hmm,
        &msa,
    );
    build_c_hmm(
        &["--amino", "--fast", "--symfrac", "0.62", "--wpb", "--enone"],
        &c_fast_hmm,
        &msa,
    );
    assert_eq!(
        hmm_without_date_or_com(&rust_fast_hmm),
        hmm_without_date_or_com(&c_fast_hmm)
    );

    let hmm = rust_hmm.to_str().unwrap();
    let rust_search_rows = parse_hmmsearch_rows(&run_hmmsearch_tblout(hmm, &seqdb));
    let c_search_rows = parse_hmmsearch_rows(&run_c_hmmsearch_tblout(hmm, &seqdb));
    assert_eq!(
        rust_search_rows, c_search_rows,
        "Response_reg vs E. coli proteome hmmsearch rows diverged from bundled C"
    );
    assert_eq!(rust_search_rows.len(), 37);

    let rust_scan_rows = parse_hmmsearch_rows(&run_hmmscan_tblout(hmm, &seqdb));
    let c_scan_rows = parse_hmmsearch_rows(&run_c_hmmscan_tblout(hmm, &seqdb));
    assert_eq!(
        rust_scan_rows, c_scan_rows,
        "Response_reg vs E. coli proteome hmmscan rows diverged from bundled C"
    );
    assert_eq!(rust_scan_rows.len(), 37);

    assert_eq!(
        run_hmmer_stdout(&["emit", "--seed", "17", "-N", "4", hmm]),
        run_c_tool_stdout("hmmemit", &["--seed", "17", "-N", "4", hmm])
    );
    assert_eq!(
        output_without_cpu_footer(&run_hmmer_stdout(&[
            "sim", "--seed", "13", "-N", "25", "--vit", hmm
        ])),
        output_without_cpu_footer(&run_c_tool_stdout(
            "hmmsim",
            &["--seed", "13", "-N", "25", "--vit", hmm]
        ))
    );
}

#[test]
fn test_phmmer_dnak_ecoli_proteome_matches_bundled_c_rows() {
    let query = test_path("external/new_real/queries/sp_P0A6Y8_DNAK_ECOLI.fa");
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta.gz");
    let rust_rows = parse_hmmsearch_rows(&run_phmmer_tblout(&query, &seqdb));
    let c_rows = parse_hmmsearch_rows(&run_c_phmmer_tblout_real(&query, &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "DnaK vs E. coli proteome phmmer rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 6);
}

#[test]
fn test_new_real_dnak_phmmer_max_domtblout_gzip_matches_bundled_c_rows() {
    let query = test_path("external/new_real/queries/sp_P0A6Y8_DNAK_ECOLI.fa");
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta.gz");
    let rust_rows = parse_domtbl_rows(&run_phmmer_domtblout_with_args(
        &query,
        &seqdb,
        &["--max", "--acc"],
    ));
    let c_rows = parse_domtbl_rows(&run_c_phmmer_domtblout_with_args(
        &query,
        &seqdb,
        &["--max", "--acc"],
    ));

    assert_eq!(
        rust_rows, c_rows,
        "DnaK phmmer --max --acc domtblout rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 35);
}

#[test]
fn test_dnak_phmmer_alignment_pp_cons_matches_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let rust_sto = dir.path().join("rust.sto");
    let c_sto = dir.path().join("c.sto");
    let rust_tblout = dir.path().join("rust.tblout");
    let c_tblout = dir.path().join("c.tblout");
    let query = test_path("external/new_real/queries/sp_P0A6Y8_DNAK_ECOLI.fa");
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta");
    let args = [
        "--notextw",
        "--mx",
        "PAM30",
        "--popen",
        "0.04",
        "--pextend",
        "0.3",
        "-E",
        "1e-20",
        "--domE",
        "1e-20",
    ];

    let mut rust_args = vec!["phmmer"];
    rust_args.extend_from_slice(&args);
    rust_args.extend_from_slice(&[
        "-A",
        rust_sto.to_str().unwrap(),
        "--tblout",
        rust_tblout.to_str().unwrap(),
        &query,
        &seqdb,
    ]);
    let rust = Command::new(binary_path("hmmer"))
        .args(rust_args)
        .output()
        .expect("failed to run hmmer phmmer -A");
    assert!(
        rust.status.success(),
        "hmmer phmmer -A failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let mut c_args = args.to_vec();
    c_args.extend_from_slice(&[
        "-A",
        c_sto.to_str().unwrap(),
        "--tblout",
        c_tblout.to_str().unwrap(),
        &query,
        &seqdb,
    ]);
    let c = Command::new(test_path("hmmer/src/phmmer"))
        .args(c_args)
        .output()
        .expect("failed to run bundled C phmmer -A");
    assert!(
        c.status.success(),
        "bundled C phmmer -A failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    let rust_rows = parse_hmmsearch_rows(&std::fs::read_to_string(rust_tblout).unwrap());
    let c_rows = parse_hmmsearch_rows(&std::fs::read_to_string(c_tblout).unwrap());
    assert_eq!(rust_rows, c_rows);
    assert_eq!(rust_rows.len(), 3);
    assert_eq!(
        std::fs::read_to_string(rust_sto).unwrap(),
        std::fs::read_to_string(c_sto).unwrap(),
        "phmmer -A saved alignment should match bundled C, including PP_cons"
    );
}

#[test]
fn test_phmmer_dnak_ecoli_first500_alignment_wraps_like_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("ecoli_first500.fa");
    let rust_ali = dir.path().join("rust.sto");
    let c_ali = dir.path().join("c.sto");
    write_first_fasta_records(
        &seqdb,
        "external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta",
        500,
    );
    let query = test_path("external/new_real/queries/sp_P0A6Y8_DNAK_ECOLI.fa");

    let rust = Command::new(binary_path("hmmer"))
        .args([
            "phmmer",
            "--max",
            "-E",
            "100",
            "--incE",
            "100",
            "-A",
            rust_ali.to_str().unwrap(),
            &query,
            seqdb.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer phmmer -A");
    assert!(
        rust.status.success(),
        "hmmer phmmer -A failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let c = Command::new(test_path("hmmer/src/phmmer"))
        .args([
            "--max",
            "-E",
            "100",
            "--incE",
            "100",
            "-A",
            c_ali.to_str().unwrap(),
            &query,
            seqdb.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run bundled C phmmer -A");
    assert!(
        c.status.success(),
        "bundled C phmmer -A failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    assert_eq!(
        std::fs::read_to_string(rust_ali).unwrap(),
        std::fs::read_to_string(c_ali).unwrap(),
        "phmmer -A Stockholm block wrapping diverged from bundled C"
    );
}

#[test]
fn test_jackhmmer_dnak_ecoli_proteome_matches_bundled_c_rows() {
    let query = test_path("external/new_real/queries/sp_P0A6Y8_DNAK_ECOLI.fa");
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta");
    let rust_rows = parse_hmmsearch_rows(&run_jackhmmer_tblout(&query, &seqdb));
    let c_rows = parse_hmmsearch_rows(&run_c_jackhmmer_tblout(&query, &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "DnaK vs E. coli proteome jackhmmer rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 6);
}

#[test]
fn test_jackhmmer_dnak_ecoli_first800_chkhmm_matches_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("ecoli_first800.fa");
    let rust_prefix = dir.path().join("rust_jack_chk");
    let c_prefix = dir.path().join("c_jack_chk");
    write_first_fasta_records(
        &seqdb,
        "external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta",
        800,
    );
    let query = test_path("external/new_real/queries/sp_P0A6Y8_DNAK_ECOLI.fa");

    let rust = Command::new(binary_path("hmmer"))
        .args([
            "jackhmmer",
            "-N",
            "2",
            "--cpu",
            "1",
            "--noali",
            "--chkhmm",
            rust_prefix.to_str().unwrap(),
            &query,
            seqdb.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer jackhmmer --chkhmm");
    assert!(
        rust.status.success(),
        "hmmer jackhmmer --chkhmm failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let c = Command::new(test_path("hmmer/src/jackhmmer"))
        .args([
            "-N",
            "2",
            "--cpu",
            "1",
            "--noali",
            "--chkhmm",
            c_prefix.to_str().unwrap(),
            &query,
            seqdb.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run bundled C jackhmmer --chkhmm");
    assert!(
        c.status.success(),
        "bundled C jackhmmer --chkhmm failed: {}",
        String::from_utf8_lossy(&c.stderr)
    );

    for round in 1..=2 {
        let rust_hmm = dir.path().join(format!("rust_jack_chk-{round}.hmm"));
        let c_hmm = dir.path().join(format!("c_jack_chk-{round}.hmm"));
        assert_hmm_text_matches_with_float_tolerance(
            &hmm_without_date_or_com(&rust_hmm),
            &hmm_without_date_or_com(&c_hmm),
            1.0e-4,
            &format!("jackhmmer DnaK/E. coli round {round} checkpoint"),
        );
    }
}

#[test]
fn test_protein_medium_human_phmmer_and_jackhmmer_options_match_bundled_c_rows() {
    let query = test_path("external/protein_medium/queries/sp_O43739_CYH3_HUMAN.fa");
    let seqdb = test_path("external/protein_medium/uniprot_UP000005640_human.fasta");

    let rust_phmmer =
        parse_hmmsearch_rows(&run_phmmer_tblout_with_args(&query, &seqdb, &["--acc"]));
    let c_phmmer = parse_hmmsearch_rows(&run_c_phmmer_tblout_with_args(&query, &seqdb, &["--acc"]));
    assert_eq!(
        rust_phmmer, c_phmmer,
        "human medium phmmer --acc rows diverged from bundled C"
    );
    assert_eq!(rust_phmmer.len(), 86);

    let rust_domains = parse_domtbl_rows(&run_phmmer_domtblout_with_args(
        &query,
        &seqdb,
        &["--domE", "1e-20", "--incdomE", "1e-30"],
    ));
    let c_domains = parse_domtbl_rows(&run_c_phmmer_domtblout_with_args(
        &query,
        &seqdb,
        &["--domE", "1e-20", "--incdomE", "1e-30"],
    ));
    assert_eq!(
        rust_domains, c_domains,
        "human medium phmmer domain threshold rows diverged from bundled C"
    );
    assert_eq!(rust_domains.len(), 15);

    let rust_jackhmmer = parse_hmmsearch_rows(&run_jackhmmer_tblout_with_args(
        &query,
        &seqdb,
        &["-N", "2", "--acc"],
    ));
    let c_jackhmmer = parse_hmmsearch_rows(&run_c_jackhmmer_tblout_with_args(
        &query,
        &seqdb,
        &["-N", "2", "--acc"],
    ));
    assert_eq!(
        rust_jackhmmer, c_jackhmmer,
        "human medium jackhmmer -N 2 --acc rows diverged from bundled C"
    );
    assert_eq!(rust_jackhmmer.len(), 162);
}

#[test]
fn test_protein_medium_human_search_pfamtblout_and_matrix_options_match_bundled_c_rows() {
    let query = test_path("external/protein_medium/queries/sp_O43739_CYH3_HUMAN.fa");
    let seqdb = test_path("external/protein_medium/uniprot_UP000005640_human.fasta");
    let seqdb_gz = test_path("external/protein_medium/uniprot_UP000005640_human.fasta.gz");

    let rust_pfam = parse_pfamtbl_rows(&run_phmmer_pfamtblout_with_args(
        &query,
        &seqdb_gz,
        &["--acc", "--notextw", "-E", "1e-3", "--domE", "1e-3"],
    ));
    let c_pfam = parse_pfamtbl_rows(&run_c_phmmer_pfamtblout_with_args(
        &query,
        &seqdb_gz,
        &["--acc", "--notextw", "-E", "1e-3", "--domE", "1e-3"],
    ));
    assert_eq!(
        rust_pfam, c_pfam,
        "human medium phmmer pfamtblout option rows diverged from bundled C"
    );
    assert_eq!(rust_pfam.0.len() + rust_pfam.1.len(), 114);

    let rust_mx = parse_hmmsearch_rows(&run_phmmer_tblout_with_args(
        &query,
        &seqdb,
        &["--mx", "BLOSUM45", "-Z", "20000", "--domZ", "200"],
    ));
    let c_mx = parse_hmmsearch_rows(&run_c_phmmer_tblout_with_args(
        &query,
        &seqdb,
        &["--mx", "BLOSUM45", "-Z", "20000", "--domZ", "200"],
    ));
    assert_eq!(
        rust_mx, c_mx,
        "human medium phmmer matrix/Z option rows diverged from bundled C"
    );
    assert_eq!(rust_mx.len(), 79);

    let rust_jack_domains = parse_domtbl_rows(&run_jackhmmer_domtblout_with_args(
        &query,
        &seqdb,
        &[
            "-N",
            "2",
            "-E",
            "1e-3",
            "--incE",
            "1e-4",
            "--domE",
            "1e-3",
            "--incdomE",
            "1e-4",
        ],
    ));
    let c_jack_domains = parse_domtbl_rows(&run_c_jackhmmer_domtblout_with_args(
        &query,
        &seqdb,
        &[
            "-N",
            "2",
            "-E",
            "1e-3",
            "--incE",
            "1e-4",
            "--domE",
            "1e-3",
            "--incdomE",
            "1e-4",
        ],
    ));
    assert_eq!(
        rust_jack_domains, c_jack_domains,
        "human medium jackhmmer domain threshold rows diverged from bundled C"
    );
    assert_eq!(rust_jack_domains.len(), 152);
}

#[test]
fn test_new_real_hatpase_alimask_matches_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let rust_out = dir.path().join("rust.sto");
    let c_out = dir.path().join("c.sto");
    let msa = test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto");
    run_hmmer_stdout(&[
        "alimask",
        "--amino",
        "--modelrange",
        "2..10",
        &msa,
        rust_out.to_str().unwrap(),
    ]);
    run_c_tool_stdout(
        "alimask",
        &[
            "--amino",
            "--modelrange",
            "2..10",
            &msa,
            c_out.to_str().unwrap(),
        ],
    );

    assert_eq!(
        std::fs::read_to_string(rust_out).unwrap(),
        std::fs::read_to_string(c_out).unwrap()
    );
}

#[test]
fn test_new_real_alimask_pfam_outformat_and_fast_summary_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let rust_pfam = dir.path().join("rust.pfam");
    let c_pfam = dir.path().join("c.pfam");
    let hatpase = test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto");
    run_hmmer_stdout(&[
        "alimask",
        "--modelrange",
        "5-25,40-55",
        "--outformat",
        "Pfam",
        &hatpase,
        rust_pfam.to_str().unwrap(),
    ]);
    run_c_tool_stdout(
        "alimask",
        &[
            "--modelrange",
            "5-25,40-55",
            "--outformat",
            "Pfam",
            &hatpase,
            c_pfam.to_str().unwrap(),
        ],
    );
    assert_eq!(
        std::fs::read_to_string(&rust_pfam).unwrap(),
        std::fs::read_to_string(&c_pfam).unwrap(),
        "alimask --outformat Pfam post-MSA diverged from bundled C"
    );

    let rust_fast = dir.path().join("rust_fast.sto");
    let c_fast = dir.path().join("c_fast.sto");
    let trna = test_path("external/new_real/rfam/RF00005_tRNA.seed.sto");
    let rust_stdout = run_hmmer_stdout(&[
        "alimask",
        "--fast",
        "--wnone",
        "--symfrac",
        "0.7",
        "--alirange",
        "3-30",
        &trna,
        rust_fast.to_str().unwrap(),
    ]);
    let c_stdout = run_c_tool_stdout(
        "alimask",
        &[
            "--fast",
            "--wnone",
            "--symfrac",
            "0.7",
            "--alirange",
            "3-30",
            &trna,
            c_fast.to_str().unwrap(),
        ],
    );
    assert_eq!(
        std::fs::read_to_string(&rust_fast).unwrap(),
        std::fs::read_to_string(&c_fast).unwrap()
    );
    assert_eq!(
        normalize_hmmbuild_summary(&rust_stdout),
        normalize_hmmbuild_summary(&c_stdout),
        "alimask --fast summary diverged from bundled C"
    );
}

#[test]
fn test_new_real_trna_alimask_coordinate_modes_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let msa = test_path("external/new_real/rfam/RF00005_tRNA.seed.sto");
    let rust_masked = dir.path().join("rust_tRNA_masked.sto");
    let c_masked = dir.path().join("c_tRNA_masked.sto");

    run_hmmer_stdout(&[
        "alimask",
        "--rna",
        "--wgiven",
        "--alirange",
        "2..6,10..14",
        "--appendmask",
        &msa,
        rust_masked.to_str().unwrap(),
    ]);
    run_c_tool_stdout(
        "alimask",
        &[
            "--rna",
            "--wgiven",
            "--alirange",
            "2..6,10..14",
            "--appendmask",
            &msa,
            c_masked.to_str().unwrap(),
        ],
    );
    assert_eq!(
        std::fs::read_to_string(&rust_masked).unwrap(),
        std::fs::read_to_string(&c_masked).unwrap(),
        "tRNA alimask --appendmask output diverged from bundled C"
    );

    assert_eq!(
        run_hmmer_stdout(&[
            "alimask",
            "--rna",
            "--fast",
            "--symfrac",
            "0.60",
            "--model2ali",
            "1..5",
            &msa,
        ]),
        run_c_tool_stdout(
            "alimask",
            &[
                "--rna",
                "--fast",
                "--symfrac",
                "0.60",
                "--model2ali",
                "1..5",
                &msa,
            ],
        ),
        "tRNA alimask --model2ali output diverged from bundled C"
    );
    assert_eq!(
        run_hmmer_stdout(&[
            "alimask",
            "--rna",
            "--fast",
            "--symfrac",
            "0.60",
            "--ali2model",
            "3..12",
            &msa,
        ]),
        run_c_tool_stdout(
            "alimask",
            &[
                "--rna",
                "--fast",
                "--symfrac",
                "0.60",
                "--ali2model",
                "3..12",
                &msa,
            ],
        ),
        "tRNA alimask --ali2model output diverged from bundled C"
    );
}

#[test]
fn test_hmmsearch_hatpase_ecoli_proteome_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta");

    let rust_rows = parse_hmmsearch_rows(&run_hmmsearch_tblout(hmm.to_str().unwrap(), &seqdb));
    let c_rows = parse_hmmsearch_rows(&run_c_hmmsearch_tblout(hmm.to_str().unwrap(), &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "HATPase_c vs E. coli proteome hmmsearch rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 37);
}

#[test]
fn test_hmmsearch_hatpase_ecoli_proteome_gzip_target_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta.gz");

    let rust_rows = parse_hmmsearch_rows(&run_hmmsearch_tblout(hmm.to_str().unwrap(), &seqdb));
    let c_rows = parse_hmmsearch_rows(&run_c_hmmsearch_tblout(hmm.to_str().unwrap(), &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "HATPase_c vs gzipped E. coli proteome hmmsearch rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 37);
}

#[test]
fn test_hmmsearch_hatpase_ecoli_proteome_threshold_options_match_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta");
    let args = [
        "-E",
        "1e-5",
        "--incE",
        "1e-6",
        "--domE",
        "1e-5",
        "--incdomE",
        "1e-6",
    ];

    let rust_rows = parse_hmmsearch_rows(&run_hmmsearch_tblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &args,
    ));
    let c_rows = parse_hmmsearch_rows(&run_c_hmmsearch_tblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &args,
    ));

    assert_eq!(
        rust_rows, c_rows,
        "HATPase_c hmmsearch threshold-option rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 34);
}

#[test]
fn test_hmmsearch_hatpase_ecoli_proteome_domtblout_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta");

    let rust_rows = parse_domtbl_rows(&run_hmmsearch_domtblout(hmm.to_str().unwrap(), &seqdb));
    let c_rows = parse_domtbl_rows(&run_c_hmmsearch_domtblout(hmm.to_str().unwrap(), &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "HATPase_c vs E. coli proteome hmmsearch domtblout rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 41);
}

#[test]
fn test_new_real_hatpase_hmmsearch_human_nobias_nonull2_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let seqdb = test_path("external/protein_medium/uniprot_UP000005640_human.fasta");
    let args = ["--nobias", "--nonull2", "--domZ", "1234"];

    let rust_rows = parse_hmmsearch_rows(&run_hmmsearch_tblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &args,
    ));
    let c_rows = parse_hmmsearch_rows(&run_c_hmmsearch_tblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &args,
    ));
    assert_eq!(
        rust_rows, c_rows,
        "HATPase_c vs human hmmsearch --nobias --nonull2 rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 31);

    let rust_domains = parse_domtbl_rows(&run_hmmsearch_domtblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &args,
    ));
    let c_domains = parse_domtbl_rows(&run_c_hmmsearch_domtblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &args,
    ));
    assert_eq!(
        rust_domains, c_domains,
        "HATPase_c vs human hmmsearch --nobias --nonull2 domain rows diverged from bundled C"
    );
    assert_eq!(rust_domains.len(), 35);
}

#[test]
fn test_hmmsearch_hatpase_ecoli_proteome_pfamtblout_cutoffs_match_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta");
    let seqdb_gz = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta.gz");

    let rust_ga = parse_pfamtbl_rows(&run_hmmsearch_pfamtblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb_gz,
        &["--cut_ga"],
    ));
    let c_ga = parse_pfamtbl_rows(&run_c_hmmsearch_pfamtblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb_gz,
        &["--cut_ga"],
    ));
    assert_eq!(
        rust_ga, c_ga,
        "HATPase_c hmmsearch --cut_ga pfamtblout rows diverged from bundled C"
    );
    assert_eq!(rust_ga.0.len() + rust_ga.1.len(), 67);

    let rust_nc = parse_pfamtbl_rows(&run_hmmsearch_pfamtblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &["--cut_nc"],
    ));
    let c_nc = parse_pfamtbl_rows(&run_c_hmmsearch_pfamtblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &["--cut_nc"],
    ));
    assert_eq!(
        rust_nc, c_nc,
        "HATPase_c hmmsearch --cut_nc pfamtblout rows diverged from bundled C"
    );
    assert_eq!(rust_nc.0.len() + rust_nc.1.len(), 68);
}

#[test]
fn test_hmmscan_hatpase_ecoli_proteome_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("PF02518_HATPase_c.hmm");
    build_rust_hmm(
        &["--amino"],
        &hmm,
        &test_path("external/new_real/pfam/PF02518_HATPase_c.seed.sto"),
    );
    let seqdb = test_path("external/new_real/protein/uniprot_UP000000625_ecoli_k12.fasta");

    let rust_rows = parse_hmmsearch_rows(&run_hmmscan_tblout(hmm.to_str().unwrap(), &seqdb));
    let c_rows = parse_hmmsearch_rows(&run_c_hmmscan_tblout(hmm.to_str().unwrap(), &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "HATPase_c vs E. coli proteome hmmscan rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 37);
}

#[test]
fn test_hmmscan_pfam_panel_cutoff_pfamtblout_matches_bundled_c_rows() {
    let hmmdb = test_path("external/realistic/pfam/Pfam-A.first12.hmm");
    let seqdb = test_path("external/realistic/protein/uniprot_UP000002311_yeast.fasta.gz");
    let rust_rows = parse_pfamtbl_rows(&run_hmmscan_pfamtblout_with_args(
        &hmmdb,
        &seqdb,
        &["--cut_ga"],
    ));
    let c_rows = parse_pfamtbl_rows(&run_c_hmmscan_pfamtblout_with_args(
        &hmmdb,
        &seqdb,
        &["--cut_ga"],
    ));

    assert_eq!(
        rust_rows, c_rows,
        "Pfam first12 hmmscan --cut_ga pfamtblout rows diverged from bundled C"
    );
    assert_eq!(rust_rows.0.len() + rust_rows.1.len(), 18);
}

#[test]
fn test_hmmscan_pfam_panel_human_query_max_outputs_match_bundled_c_rows() {
    let hmmdb = test_path("external/realistic/pfam/Pfam-A.first12.hmm");
    let query = test_path("external/protein_medium/queries/sp_O43739_CYH3_HUMAN.fa");
    let args = ["--acc", "--max"];

    let rust_rows = parse_hmmsearch_rows(&run_hmmscan_tblout_with_args(&hmmdb, &query, &args));
    let c_rows = parse_hmmsearch_rows(&run_c_hmmscan_tblout_with_args(&hmmdb, &query, &args));
    assert_eq!(
        rust_rows, c_rows,
        "Pfam first12 vs human query hmmscan --acc --max rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 6);

    let rust_domains = parse_domtbl_rows(&run_hmmscan_domtblout_with_args(&hmmdb, &query, &args));
    let c_domains = parse_domtbl_rows(&run_c_hmmscan_domtblout_with_args(&hmmdb, &query, &args));
    assert_eq!(
        rust_domains, c_domains,
        "Pfam first12 vs human query hmmscan --acc --max domain rows diverged from bundled C"
    );
    assert_eq!(rust_domains.len(), 8);
}

#[test]
fn test_nhmmscan_trna_ecoli_genome_slice_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    let seqdb = test_path("external/new_real/derived/GCF_000005845.2_ASM584v2_first250k.fna");

    let rust_rows = parse_nhmmer_rows(&run_nhmmscan_tblout(hmm.to_str().unwrap(), &seqdb));
    let c_rows = parse_nhmmer_rows(&run_c_nhmmscan_tblout(hmm.to_str().unwrap(), &seqdb));

    assert_eq!(
        rust_rows, c_rows,
        "tRNA vs E. coli genome slice nhmmscan rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 4);
}

#[test]
fn test_nhmmscan_trna_ecoli_genome_gzip_max_dfamtblout_matches_bundled_c_rows() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    let seqdb = test_path("external/new_real/dna/GCF_000005845.2_ASM584v2_genomic.fna.gz");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );

    let rust_rows = parse_dfamtbl_rows(&run_nhmmscan_dfamtblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &["--acc", "--max"],
    ));
    let c_rows = parse_dfamtbl_rows(&run_c_nhmmscan_dfamtblout_with_args(
        hmm.to_str().unwrap(),
        &seqdb,
        &["--acc", "--max"],
    ));

    assert_eq!(
        rust_rows, c_rows,
        "tRNA vs gzipped E. coli genome nhmmscan --acc --max dfamtblout rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 86);
}

#[test]
fn test_nhmmscan_crick_pipeline_stats_use_c_residue_denominator() {
    let dir = tempfile::tempdir().unwrap();
    let rust_hmm = dir.path().join("rust.RF00005_tRNA.hmm");
    let c_hmm = dir.path().join("c.RF00005_tRNA.hmm");
    build_rust_hmm(
        &["--dna"],
        &rust_hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    std::fs::copy(&rust_hmm, &c_hmm).unwrap();
    press_rust_hmmdb(&rust_hmm);
    press_c_hmmdb(&c_hmm);

    let seqdb = test_path("external/new_real/derived/GCF_000005845.2_ASM584v2_first250k.fna");
    let rust_stdout = run_hmmer_stdout(&[
        "nhmmscan",
        "--crick",
        "--noali",
        "-E",
        "100",
        rust_hmm.to_str().unwrap(),
        &seqdb,
    ]);
    let c_stdout = run_c_tool_stdout(
        "nhmmscan",
        &[
            "--crick",
            "--noali",
            "-E",
            "100",
            c_hmm.to_str().unwrap(),
            &seqdb,
        ],
    );

    let query_stats = |stdout: &str| {
        stdout
            .lines()
            .find(|line| line.starts_with("Query sequence(s):"))
            .unwrap()
            .to_string()
    };
    assert_eq!(query_stats(&rust_stdout), query_stats(&c_stdout));
    assert!(
        rust_stdout
            .contains("Query sequence(s):                         1  (500000 residues searched)"),
        "nhmmscan --crick should report C's doubled residue denominator:\n{rust_stdout}"
    );
}

#[test]
fn test_nhmmer_fmindex_watson_seed_options_use_c_residue_denominator() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    let rust_fm = dir.path().join("rust.ecoli250k.hmmerdb");
    let c_fm = dir.path().join("c.ecoli250k.hmmerdb");
    let seqdb = test_path("external/new_real/derived/GCF_000005845.2_ASM584v2_first250k.fna");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    make_rust_fm_database(&seqdb, &rust_fm);
    make_c_fm_database(&seqdb, &c_fm);

    let args = [
        "--watson",
        "--seed_sc_thresh",
        "10",
        "--seed_sc_density",
        "0.5",
        "--seed_req_pos",
        "3",
        "--noali",
    ];
    let (rust_stdout, rust_tbl) =
        run_nhmmer(hmm.to_str().unwrap(), rust_fm.to_str().unwrap(), &args);
    let (c_stdout, c_tbl) =
        run_c_nhmmer_with_args(hmm.to_str().unwrap(), c_fm.to_str().unwrap(), &args);
    assert_eq!(
        parse_nhmmer_rows(&rust_tbl),
        parse_nhmmer_rows(&c_tbl),
        "nhmmer FM --watson seed-option tblout rows diverged from bundled C"
    );

    let target_stats = |stdout: &str| {
        stdout
            .lines()
            .find(|line| line.starts_with("Target sequences:"))
            .unwrap()
            .to_string()
    };
    assert_eq!(target_stats(&rust_stdout), target_stats(&c_stdout));
    assert!(
        rust_stdout
            .contains("Target sequences:                          1  (500000 residues searched)"),
        "FM-index nhmmer --watson should keep C's doubled residue denominator:\n{rust_stdout}"
    );
}

#[test]
fn test_nhmmer_fmindex_w_beta_rows_match_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    let rust_fm = dir.path().join("rust.ecoli250k.hmmerdb");
    let c_fm = dir.path().join("c.ecoli250k.hmmerdb");
    let seqdb = test_path("external/new_real/derived/GCF_000005845.2_ASM584v2_first250k.fna");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    make_rust_fm_database(&seqdb, &rust_fm);
    make_c_fm_database(&seqdb, &c_fm);

    let args = ["--watson", "--w_beta", "0.5", "--noali"];
    let (_rust_stdout, rust_tbl) =
        run_nhmmer(hmm.to_str().unwrap(), rust_fm.to_str().unwrap(), &args);
    let (_c_stdout, c_tbl) =
        run_c_nhmmer_with_args(hmm.to_str().unwrap(), c_fm.to_str().unwrap(), &args);
    assert_eq!(
        parse_nhmmer_rows(&rust_tbl),
        parse_nhmmer_rows(&c_tbl),
        "nhmmer FM --w_beta tblout rows diverged from bundled C"
    );

    let rust_dfam = parse_dfamtbl_rows(&run_nhmmer_dfamtblout_with_args(
        hmm.to_str().unwrap(),
        rust_fm.to_str().unwrap(),
        &args,
    ));
    let c_dfam = parse_dfamtbl_rows(&run_c_nhmmer_dfamtblout_with_args(
        hmm.to_str().unwrap(),
        c_fm.to_str().unwrap(),
        &args,
    ));
    assert_eq!(
        rust_dfam, c_dfam,
        "nhmmer FM --w_beta dfamtblout rows diverged from bundled C"
    );
}

#[test]
fn test_nhmmscan_w_beta_reports_option_but_uses_pressed_maxl_like_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let fasta = dir.path().join("yeast_chr_i_ii.fa");
    let rust_hmm = dir.path().join("rust.RF00005_tRNA.hmm");
    let c_hmm = dir.path().join("c.RF00005_tRNA.hmm");
    write_named_fasta_records(
        &fasta,
        "external/realistic/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa",
        &["I", "II"],
    );
    build_rust_hmm(
        &["--dna"],
        &rust_hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    std::fs::copy(&rust_hmm, &c_hmm).unwrap();
    press_rust_hmmdb(&rust_hmm);
    press_c_hmmdb(&c_hmm);

    let args = ["--crick", "--w_beta", "0.7", "--nonull2"];
    let rust_rows = parse_nhmmer_rows(&run_nhmmscan_tblout_with_args(
        rust_hmm.to_str().unwrap(),
        fasta.to_str().unwrap(),
        &args,
    ));
    let c_rows = parse_nhmmer_rows(&run_c_nhmmscan_tblout_with_args(
        c_hmm.to_str().unwrap(),
        fasta.to_str().unwrap(),
        &args,
    ));
    assert_eq!(
        rust_rows, c_rows,
        "nhmmscan --w_beta tblout rows diverged from bundled C"
    );

    let rust_dfam = parse_dfamtbl_rows(&run_nhmmscan_dfamtblout_with_args(
        rust_hmm.to_str().unwrap(),
        fasta.to_str().unwrap(),
        &args,
    ));
    let c_dfam = parse_dfamtbl_rows(&run_c_nhmmscan_dfamtblout_with_args(
        c_hmm.to_str().unwrap(),
        fasta.to_str().unwrap(),
        &args,
    ));
    assert_eq!(
        rust_dfam, c_dfam,
        "nhmmscan --w_beta dfamtblout rows diverged from bundled C"
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
        "Residues passing SSV filter:              12  (1); expected (0.02)",
        "Residues passing bias filter:             12  (1); expected (0.02)",
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
        "Residues passing SSV filter:              12  (1); expected (0.02)",
        "Residues passing bias filter:             12  (1); expected (0.02)",
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
#[ignore = "requires local human_swissprot_2k Pfam fixture"]
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
#[ignore = "requires local human_swissprot_2k Pfam fixture"]
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
#[ignore = "slow parity sweep across all local/generated Pfam golden fixtures"]
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
#[ignore = "slow parity sweep across all local/generated Pfam golden fixtures"]
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
#[ignore = "slow parity sweep across all local/generated Pfam golden fixtures"]
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

// ---------------------------------------------------------------------------
// FM-index (makehmmerdb) target-search parity vs bundled C nhmmer.
//
// These exercise the exact two-sweep SSV-over-FM kernel (src/simd/fm_ssv.rs)
// wired into the FM path. They build a makehmmerdb database from a DNA FASTA
// (byte-identical to C), then compare Rust vs C nhmmer hit sets. Low-threshold
// variants stress the weak diagonals the FM seeding/extension must reproduce.
// ---------------------------------------------------------------------------

fn build_fmdb(dir: &std::path::Path, fasta_rel: &str) -> String {
    let db = dir.join("target.hmmerdb");
    let out = Command::new(binary_path("hmmer"))
        .args([
            "makehmmerdb",
            "--dna",
            &test_path(fasta_rel),
            db.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run makehmmerdb");
    assert!(
        out.status.success(),
        "makehmmerdb failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    db.to_str().unwrap().to_string()
}

// Coordinate/strand key for hit-set comparison (robust to last-digit float
// formatting differences in the score/E-value columns).
fn fm_hit_keys(rows: &[NhmmerRow]) -> Vec<(String, usize, usize, String)> {
    let mut v: Vec<_> = rows
        .iter()
        .map(|r| (r.target.clone(), r.ali_from, r.ali_to, r.strand.clone()))
        .collect();
    v.sort();
    v
}

fn assert_fm_parity(hmm_rel: &str, fasta_rel: &str, extra: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    let db = build_fmdb(dir.path(), fasta_rel);
    let hmm = test_path(hmm_rel);
    let mut args = vec!["--cpu", "1", "--noali"];
    args.extend_from_slice(extra);
    let (_stdout, rust_tbl) = run_nhmmer(&hmm, &db, &args);
    let c_tbl = run_c_nhmmer_tblout_with_args(&hmm, &db, &args);
    let rust_keys = fm_hit_keys(&parse_nhmmer_rows(&rust_tbl));
    let c_keys = fm_hit_keys(&parse_nhmmer_rows(&c_tbl));
    assert_eq!(
        rust_keys,
        c_keys,
        "FM-index nhmmer hit set diverged from C for {hmm_rel} vs {fasta_rel} {extra:?}\n\
         Rust ({} hits): {rust_keys:?}\nC ({} hits): {c_keys:?}",
        rust_keys.len(),
        c_keys.len()
    );
}

#[test]
fn test_nhmmer_fmindex_made1_matches_c_hit_set() {
    assert_fm_parity(
        "hmmer/tutorial/MADE1.hmm",
        "hmmer/tutorial/dna_target.fa",
        &[],
    );
}

#[test]
fn test_nhmmer_fmindex_made1_low_threshold_matches_c() {
    // Lower reporting threshold pulls in weak diagonals on both strands, where
    // the FM seed-then-rescore vs exact-SSV difference would show up.
    assert_fm_parity(
        "hmmer/tutorial/MADE1.hmm",
        "hmmer/tutorial/dna_target.fa",
        &["-T", "0"],
    );
}

#[test]
fn test_nhmmer_fmindex_3box_matches_c_hit_set() {
    assert_fm_parity(
        "hmmer/testsuite/3box.hmm",
        "hmmer/testsuite/3box-alitest.fa",
        &[],
    );
}

#[test]
fn test_nhmmer_fmindex_3box_low_threshold_matches_c() {
    // Negative threshold in C/Easel's space-separated form; Rust now accepts it
    // too (`allow_hyphen_values` on the threshold args).
    assert_fm_parity(
        "hmmer/testsuite/3box.hmm",
        "hmmer/testsuite/3box-alitest.fa",
        &["-T", "-20"],
    );
}

#[test]
fn test_nhmmer_fmindex_ecori_matches_c_hit_set() {
    assert_fm_parity(
        "hmmer/testsuite/ecori.hmm",
        "hmmer/testsuite/3box-alitest.fa",
        &[],
    );
}

// Parse the four "Residues passing ... filter" counters (SSV, bias, Vit, Fwd)
// from an nhmmer stdout footer. Returns them in (ssv, bias, vit, fwd) order.
fn parse_filter_residue_counters(stdout: &str) -> (u64, u64, u64, u64) {
    let grab = |needle: &str| -> u64 {
        let line = stdout
            .lines()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("missing counter line {needle:?} in:\n{stdout}"));
        // "Residues passing SSV filter:           41383  (0.0627); expected (0.03)"
        line.split(':')
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|tok| tok.parse::<u64>().ok())
            .unwrap_or_else(|| panic!("could not parse counter from {line:?}"))
    };
    (
        grab("Residues passing SSV filter"),
        grab("Residues passing bias filter"),
        grab("Residues passing Vit filter"),
        grab("Residues passing Fwd filter"),
    )
}

// MED-1 regression: the FM-index path's "Residues passing ... filter" counters
// must match the bundled C nhmmer. Before the fix, the Rust seed-then-rescore
// FM window construction credited far fewer residues than C (e.g. SSV 23924 vs
// 41383 for MADE1). The fix extends FM seed diagonals with the C-faithful
// `FM_extendSeed` port and feeds the un-pre-merged extended-diagonal list to
// the same `p7_pli_extend_and_merge_windows(.., 0)` pass C uses.
//
// SSV and bias counters must match C exactly; Vit/Fwd are allowed a 1-residue
// slack for a single boundary tie in the post-Vit 0.5-overlap merge that does
// not affect the hit set (verified bit-identical to C).
fn assert_fm_counters_match_c(hmm_rel: &str, fasta_rel: &str) {
    let dir = tempfile::tempdir().unwrap();
    let db = build_fmdb(dir.path(), fasta_rel);
    let hmm = test_path(hmm_rel);
    let args = ["--cpu", "1", "--noali"];

    let (rust_stdout, _) = run_nhmmer(&hmm, &db, &args);
    let (c_stdout, _) = run_c_nhmmer_with_args(&hmm, &db, &args);

    let (r_ssv, r_bias, r_vit, r_fwd) = parse_filter_residue_counters(&rust_stdout);
    let (c_ssv, c_bias, c_vit, c_fwd) = parse_filter_residue_counters(&c_stdout);

    assert_eq!(
        r_ssv, c_ssv,
        "FM SSV residue counter diverged from C for {hmm_rel}: Rust {r_ssv} vs C {c_ssv}"
    );
    assert_eq!(
        r_bias, c_bias,
        "FM bias residue counter diverged from C for {hmm_rel}: Rust {r_bias} vs C {c_bias}"
    );
    assert!(
        (r_vit as i64 - c_vit as i64).abs() <= 1,
        "FM Vit residue counter diverged from C for {hmm_rel}: Rust {r_vit} vs C {c_vit}"
    );
    assert!(
        (r_fwd as i64 - c_fwd as i64).abs() <= 1,
        "FM Fwd residue counter diverged from C for {hmm_rel}: Rust {r_fwd} vs C {c_fwd}"
    );
}

#[test]
fn test_nhmmer_fmindex_made1_filter_residue_counters_match_c() {
    // C (verified): SSV/bias/Vit/Fwd = 41383/34723/3552/1942.
    assert_fm_counters_match_c("hmmer/tutorial/MADE1.hmm", "hmmer/tutorial/dna_target.fa");
}

#[test]
fn test_nhmmer_fmindex_3box_filter_residue_counters_match_c() {
    // C (verified): SSV/bias/Vit/Fwd = 677/677/166/94 (exact on all four).
    assert_fm_counters_match_c(
        "hmmer/testsuite/3box.hmm",
        "hmmer/testsuite/3box-alitest.fa",
    );
}

// MED-2 regression: a MULTI-SEGMENT FM database. The single-segment fixtures
// (made1/3box/ecori) map to one FM segment, so they never exercise the
// `id` merge guard, the p7_COMPLEMENT extension flip, or the cross-segment
// coordinate transform that MED-2/LOW-2 touch. This test builds an FM DB from
// a DNA FASTA split into many separate records (segments) and asserts the Rust
// and bundled-C nhmmer hit sets (target name + ali coords + strand) match
// across segments on BOTH strands, at default and low reporting thresholds.
fn write_multi_segment_dna_fasta(path: &std::path::Path, source_rel: &str, seg_len: usize) {
    let src = std::fs::read_to_string(test_path(source_rel)).unwrap();
    let mut seq = String::new();
    for line in src.lines() {
        if !line.starts_with('>') {
            seq.push_str(line.trim());
        }
    }
    let mut out = String::new();
    let mut k = 0;
    let mut i = 0;
    while i < seq.len() {
        let end = (i + seg_len).min(seq.len());
        out.push_str(&format!(">seg{k}\n"));
        let chunk = &seq[i..end];
        for j in (0..chunk.len()).step_by(60) {
            let line_end = (j + 60).min(chunk.len());
            out.push_str(&chunk[j..line_end]);
            out.push('\n');
        }
        i = end;
        k += 1;
    }
    std::fs::write(path, out).unwrap();
}

fn write_named_fasta_records(path: &std::path::Path, source_rel: &str, names: &[&str]) {
    let wanted: std::collections::BTreeSet<&str> = names.iter().copied().collect();
    let src = std::fs::read_to_string(test_path(source_rel)).unwrap();
    let mut out = String::new();
    let mut keep = false;
    for line in src.lines() {
        if let Some(rest) = line.strip_prefix('>') {
            let name = rest.split_whitespace().next().unwrap_or("");
            keep = wanted.contains(name);
        }
        if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    for name in names {
        assert!(
            out.contains(&format!(">{name}")),
            "source FASTA {source_rel} did not contain record {name}"
        );
    }
    std::fs::write(path, out).unwrap();
}

fn assert_multi_segment_fm_parity(hmm_rel: &str, source_rel: &str, seg_len: usize, extra: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    let fasta = dir.path().join("multi.fa");
    write_multi_segment_dna_fasta(&fasta, source_rel, seg_len);

    // makehmmerdb on the multi-segment FASTA (multiple FM segments).
    let db = dir.path().join("multi.hmmerdb");
    let out = Command::new(binary_path("hmmer"))
        .args([
            "makehmmerdb",
            "--dna",
            fasta.to_str().unwrap(),
            db.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run makehmmerdb");
    assert!(
        out.status.success(),
        "makehmmerdb failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let db = db.to_str().unwrap();
    let hmm = test_path(hmm_rel);

    let mut args = vec!["--cpu", "1", "--noali"];
    args.extend_from_slice(extra);
    let (_stdout, rust_tbl) = run_nhmmer(&hmm, db, &args);
    let c_tbl = run_c_nhmmer_tblout_with_args(&hmm, db, &args);
    let rust_keys = fm_hit_keys(&parse_nhmmer_rows(&rust_tbl));
    let c_keys = fm_hit_keys(&parse_nhmmer_rows(&c_tbl));
    assert_eq!(
        rust_keys, c_keys,
        "multi-segment FM nhmmer hit set diverged from C for {hmm_rel} (seg_len {seg_len}, {extra:?})\n\
         Rust ({} hits): {rust_keys:?}\nC ({} hits): {c_keys:?}",
        rust_keys.len(),
        c_keys.len()
    );
    // The multi-segment DB must actually span more than one segment of hits to
    // be a meaningful exercise of the cross-segment paths.
    let distinct_segments: std::collections::BTreeSet<_> =
        c_keys.iter().map(|(name, _, _, _)| name.clone()).collect();
    assert!(
        distinct_segments.len() >= 2,
        "expected hits across >=2 FM segments to exercise MED-2; got {distinct_segments:?}"
    );
}

#[test]
fn test_nhmmer_fmindex_multi_segment_matches_c_hit_set() {
    // 4 segments (~82.5kb each) from the tutorial DNA target. MADE1 hits land in
    // multiple segments on both strands.
    assert_multi_segment_fm_parity(
        "hmmer/tutorial/MADE1.hmm",
        "hmmer/tutorial/dna_target.fa",
        82500,
        &[],
    );
}

#[test]
fn test_nhmmer_fmindex_multi_segment_low_threshold_matches_c() {
    // Many small segments + low threshold: stresses the id merge guard and the
    // complement extension/coordinate paths near segment boundaries.
    assert_multi_segment_fm_parity(
        "hmmer/tutorial/MADE1.hmm",
        "hmmer/tutorial/dna_target.fa",
        10000,
        &["-T", "0"],
    );
}

#[test]
fn test_nhmmer_fmindex_yeast_chr_i_ii_crick_rows_match_c() {
    let dir = tempfile::tempdir().unwrap();
    let fasta = dir.path().join("yeast_chr_i_ii.fa");
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    let rust_fm = dir.path().join("rust.yeast_chr_i_ii.hmmerdb");
    let c_fm = dir.path().join("c.yeast_chr_i_ii.hmmerdb");
    write_named_fasta_records(
        &fasta,
        "external/realistic/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa",
        &["I", "II"],
    );
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    make_rust_fm_database(fasta.to_str().unwrap(), &rust_fm);
    make_c_fm_database(fasta.to_str().unwrap(), &c_fm);

    let rust_rows = parse_nhmmer_rows(&run_nhmmer_tblout(
        hmm.to_str().unwrap(),
        rust_fm.to_str().unwrap(),
    ));
    let c_rows = parse_nhmmer_rows(&run_c_nhmmer_tblout(
        hmm.to_str().unwrap(),
        c_fm.to_str().unwrap(),
    ));
    assert_eq!(
        rust_rows, c_rows,
        "tRNA vs yeast chr I+II FM-index nhmmer rows diverged from bundled C"
    );
    assert_eq!(rust_rows.len(), 13);
    assert!(
        rust_rows
            .iter()
            .any(|row| row.strand == "-" && row.target == "I"),
        "fixture must include a Crick hit on chr I"
    );
    assert!(
        rust_rows
            .iter()
            .any(|row| row.strand == "-" && row.target == "II"),
        "fixture must include a Crick hit on chr II"
    );
}

#[test]
fn test_nhmmer_fmindex_seed_drop_options_keep_c_row_count() {
    let dir = tempfile::tempdir().unwrap();
    let fasta = dir.path().join("yeast_chr_vii_viii.fa");
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    let rust_fm = dir.path().join("rust.yeast_chr_vii_viii.hmmerdb");
    let c_fm = dir.path().join("c.yeast_chr_vii_viii.hmmerdb");
    write_named_fasta_records(
        &fasta,
        "external/realistic/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa",
        &["VII", "VIII"],
    );
    build_rust_hmm(
        &["--dna", "--w_beta", "0.25", "--seed", "41"],
        &hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    make_rust_fm_database(fasta.to_str().unwrap(), &rust_fm);
    make_c_fm_database(fasta.to_str().unwrap(), &c_fm);

    let args = [
        "--crick",
        "--seed_max_depth",
        "12",
        "--seed_drop_max_len",
        "3",
        "--seed_drop_lim",
        "0.2",
        "--seed_consens_match",
        "9",
        "--seed_ssv_length",
        "80",
        "--noali",
    ];
    let (_rust_stdout, rust_tbl) =
        run_nhmmer(hmm.to_str().unwrap(), rust_fm.to_str().unwrap(), &args);
    let (_c_stdout, c_tbl) =
        run_c_nhmmer_with_args(hmm.to_str().unwrap(), c_fm.to_str().unwrap(), &args);
    let rust_rows = parse_nhmmer_rows(&rust_tbl);
    let c_rows = parse_nhmmer_rows(&c_tbl);

    assert_eq!(
        rust_rows.len(),
        c_rows.len(),
        "custom FM seed options should not leak extra Crick windows"
    );
    assert_eq!(rust_rows.len(), 17);
}

#[test]
fn test_nhmmer_accepts_space_separated_negative_threshold() {
    // C/Easel accepts `-T -20` (space-separated negative bit score); Rust's clap
    // parser must too (regression for allow_hyphen_values on the threshold args).
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("t.tbl");
    let output = Command::new(binary_path("hmmer"))
        .args([
            "nhmmer",
            "--noali",
            "-T",
            "-20",
            "--tblout",
            tblout.to_str().unwrap(),
            &test_path("hmmer/testsuite/3box.hmm"),
            &test_path("hmmer/testsuite/3box-alitest.fa"),
        ])
        .output()
        .expect("failed to run hmmer nhmmer");
    assert!(
        output.status.success(),
        "nhmmer -T -20 should be accepted: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_nhmmer_tformat_fasta_rejects_fmindex_like_bundled_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("RF00005_tRNA.hmm");
    let fm = dir.path().join("ecoli250k.hmmerdb");
    build_rust_hmm(
        &["--dna"],
        &hmm,
        &test_path("external/new_real/rfam/RF00005_tRNA.dna.seed.sto"),
    );
    make_rust_fm_database(
        &test_path("external/new_real/derived/GCF_000005845.2_ASM584v2_first250k.fna"),
        &fm,
    );

    let rust = Command::new(binary_path("hmmer"))
        .args([
            "nhmmer",
            "--tformat",
            "fasta",
            hmm.to_str().unwrap(),
            fm.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer nhmmer --tformat fasta on FM index");
    let c = Command::new(test_path("hmmer/src/nhmmer"))
        .args([
            "--tformat",
            "fasta",
            hmm.to_str().unwrap(),
            fm.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run bundled C nhmmer --tformat fasta on FM index");

    assert!(!rust.status.success());
    assert!(!c.status.success());
    let rust_stderr = String::from_utf8_lossy(&rust.stderr);
    let c_stderr = String::from_utf8_lossy(&c.stderr);
    assert!(c_stderr.contains("Unable to guess alphabet for target sequence database file"));
    assert!(
        rust_stderr.contains("Unable to guess alphabet for target sequence database file"),
        "Rust should reject FM index as FASTA like bundled C:\n{rust_stderr}"
    );
}

#[test]
fn test_nhmmer_f1_f2_f3_accept_hyphen_values() {
    // L1 (audit-20260527/02): the SSV/Vit/Fwd P-value threshold flags must carry
    // `allow_hyphen_values = true` so a value that begins with a hyphen reaches
    // the value parser instead of being rejected by clap as an unknown option
    // (same class as the `-T -20` fix). We assert the parser accepts the hyphen
    // token — i.e. the failure mode is NOT clap's "unexpected argument" usage
    // error. (A hyphen-leading scientific-notation value like `-1e-3` is a valid
    // f64 here; the point is purely that clap does not reject the token.)
    for flag in ["--F1", "--F2", "--F3"] {
        let dir = tempfile::tempdir().unwrap();
        let tblout = dir.path().join("t.tbl");
        let output = Command::new(binary_path("hmmer"))
            .args([
                "nhmmer",
                "--noali",
                flag,
                "-1e-3",
                "--tblout",
                tblout.to_str().unwrap(),
                &test_path("hmmer/testsuite/3box.hmm"),
                &test_path("hmmer/testsuite/3box-alitest.fa"),
            ])
            .output()
            .expect("failed to run hmmer nhmmer");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("unexpected argument")
                && !stderr.contains("error: a value is required")
                && !stderr.contains("found argument"),
            "nhmmer {flag} -1e-3 must parse the hyphen value (allow_hyphen_values); \
             clap rejected it:\n{stderr}"
        );
    }
}

/// Regression test for audit-20260527 finding #1: the nhmmer/nhmmscan `--tblout`
/// dash separator line must right-justify FIXED-LENGTH dash literals padded with
/// spaces (matching C p7_tophits.c:1626-1627 `%*s`), NOT fill the whole column
/// with dashes. They only agree when every column is at minimum width; the MADE1
/// fixture has query accession `DF0000629.2` (11 chars > 10), widening qaccw, so a
/// fill-char dash line would be longer than the header line and misalign the
/// columns. Assert the dash line aligns exactly with the header line.
#[test]
fn test_nhmmer_tblout_dash_line_aligns_with_header_wide_accession() {
    let out = run_nhmmer_tblout(
        &test_path("hmmer/tutorial/MADE1.hmm"),
        &test_path("hmmer/tutorial/dna_target.fa"),
    );
    let lines: Vec<&str> = out.lines().collect();
    let header = lines
        .iter()
        .find(|l| l.contains("target name") && l.contains("description of target"))
        .expect("tblout should have a column header line");
    let dash = lines
        .iter()
        .find(|l| l.starts_with("#---"))
        .expect("tblout should have a dash separator line");

    // The dash underline must be exactly as wide as the header it underlines.
    assert_eq!(
        header.len(),
        dash.len(),
        "dash line width must equal header width\nheader: {:?}\ndash:   {:?}",
        header,
        dash
    );

    // Each whitespace-separated dash token must be a run of dashes only and must
    // never exceed its C fixed-literal length (i.e. it is space-padded, not
    // dash-filled). qaccw widened to 11 must show a 10-dash token + leading space,
    // not an 11-dash token.
    let tokens: Vec<&str> = dash
        .trim_start_matches('#')
        .split(' ')
        .filter(|t| !t.is_empty())
        .collect();
    let max_literal = [19usize, 10, 20, 10, 7, 7, 7, 7, 7, 7, 7, 6, 9, 6, 5, 21];
    assert_eq!(
        tokens.len(),
        max_literal.len(),
        "unexpected dash token count: {:?}",
        tokens
    );
    for (tok, &maxlen) in tokens.iter().zip(max_literal.iter()) {
        assert!(
            tok.bytes().all(|b| b == b'-'),
            "dash token has non-dash char: {:?}",
            tok
        );
        assert!(
            tok.len() <= maxlen,
            "dash token {:?} (len {}) exceeds C fixed literal length {} (column was dash-filled instead of space-padded)",
            tok,
            tok.len(),
            maxlen
        );
    }
}
