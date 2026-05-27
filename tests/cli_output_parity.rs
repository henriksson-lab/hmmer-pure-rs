use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use hmmer_pure_rs::hmm::{P7H_CA, P7H_CONS, P7H_CS, P7H_MAP, P7H_MMASK, P7H_RF};
use hmmer_pure_rs::hmmfile_binary;

static HMMPGMD_TEST_LOCK: Mutex<()> = Mutex::new(());

fn hmmpgmd_test_guard() -> std::sync::MutexGuard<'static, ()> {
    HMMPGMD_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn hmmer() -> &'static str {
    env!("CARGO_BIN_EXE_hmmer")
}

fn project_root() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
}

fn c_hmmpgmd() -> String {
    format!("{}/hmmer/src/hmmpgmd", project_root())
}

fn c_hmmpress() -> String {
    format!("{}/hmmer/src/hmmpress", project_root())
}

fn c_hmmconvert() -> String {
    format!("{}/hmmer/src/hmmconvert", project_root())
}

fn c_alimask() -> String {
    format!("{}/hmmer/src/alimask", project_root())
}

fn c_hmmemit() -> String {
    format!("{}/hmmer/src/hmmemit", project_root())
}

fn custom_protein_score_matrix() -> String {
    let residues = "ACDEFGHIKLMNPQRSTVWY";
    let header = residues
        .chars()
        .map(|residue| residue.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let mut matrix = format!("  {header}\n");
    for row in residues.chars() {
        matrix.push(row);
        for col in residues.chars() {
            let score = if row == col { 5 } else { -4 };
            matrix.push_str(&format!(" {score:>3}"));
        }
        matrix.push('\n');
    }
    matrix
}

fn extract_hmm_stats(hmm: &str) -> Vec<&str> {
    hmm.lines()
        .filter(|line| line.starts_with("STATS "))
        .collect()
}

fn hmm_stat_values(hmm: &str, label: &str) -> (f32, f32) {
    let line = hmm
        .lines()
        .find(|line| line.starts_with(label))
        .unwrap_or_else(|| panic!("missing {label} in:\n{hmm}"));
    let fields: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(fields.len(), 5, "{line}");
    (
        fields[3].parse::<f32>().unwrap(),
        fields[4].parse::<f32>().unwrap(),
    )
}

fn stdout_without_cpu_header(stdout: &str) -> String {
    let mut normalized = stdout
        .lines()
        .filter(|line| {
            !line.starts_with("# number of worker threads:")
                && !line.starts_with("# multithread parallelization:")
        })
        .collect::<Vec<_>>()
        .join("\n");
    if stdout.ends_with('\n') {
        normalized.push('\n');
    }
    normalized
}

fn table_data_rows(table: &str) -> Vec<&str> {
    table
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .collect()
}

fn assert_auto_domz_zero(stdout: &str) {
    assert!(
        stdout.contains(
            "Domain search space  (domZ):               0  [number of targets reported over threshold]"
        ),
        "{stdout}"
    );
}

#[test]
fn protein_searches_do_not_clamp_auto_domz_on_no_hit_searches() {
    for args in [
        vec![
            "phmmer",
            "--noali",
            "-E",
            "1e-200",
            "hmmer/tutorial/HBB_HUMAN",
            "hmmer/tutorial/globins45.fa",
        ],
        vec![
            "jackhmmer",
            "--noali",
            "-E",
            "1e-200",
            "hmmer/tutorial/HBB_HUMAN",
            "hmmer/tutorial/globins45.fa",
        ],
    ] {
        let output = Command::new(hmmer()).args(args).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_auto_domz_zero(&String::from_utf8(output.stdout).unwrap());
    }

    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "scan",
            "--noali",
            "-E",
            "1e-200",
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_auto_domz_zero(&String::from_utf8(output.stdout).unwrap());
}

#[test]
fn hmmlogo_default_output_matches_c_shape_and_values() {
    let output = Command::new(hmmer())
        .args(["logo", "hmmer/testsuite/20aa.hmm"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.starts_with("max expected height = 6.45\nResidue heights\n"));
    assert!(
        stdout.contains("1:  1.327  0.016  0.023  0.025  0.013  0.053  0.010  0.027  0.024  0.031")
    );
    assert!(stdout.contains("Indel values\n1:  0.010  1.858  0.995\n"));
    assert!(!stdout.contains("# Logo data for:"));
    assert!(!stdout.contains("pos\tIC"));
}

#[test]
fn hmmlogo_no_indel_suppresses_indel_block() {
    let output = Command::new(hmmer())
        .args(["logo", "--no_indel", "hmmer/testsuite/20aa.hmm"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Residue heights\n"));
    assert!(!stdout.contains("Indel values\n"));
}

#[test]
fn hmmlogo_height_modes_have_c_style_sections() {
    let score = Command::new(hmmer())
        .args([
            "logo",
            "--height_score",
            "--no_indel",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        score.status.success(),
        "{}",
        String::from_utf8_lossy(&score.stderr)
    );
    let stdout = String::from_utf8(score.stdout).unwrap();
    assert!(stdout.starts_with("Residue heights\n"));
    assert!(!stdout.contains("max expected height"));
    assert!(!stdout.contains("Indel values\n"));

    let above_bg = Command::new(hmmer())
        .args([
            "logo",
            "--height_relent_abovebg",
            "--no_indel",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        above_bg.status.success(),
        "{}",
        String::from_utf8_lossy(&above_bg.stderr)
    );
    let stdout = String::from_utf8(above_bg.stdout).unwrap();
    assert!(stdout.starts_with("max expected height = "));
    assert!(stdout.contains("Residue heights\n"));
    assert!(!stdout.contains("Indel values\n"));
}

#[test]
fn hmmlogo_uses_only_first_hmm_like_c() {
    let dir = tempfile::tempdir().unwrap();
    let multi_hmm = dir.path().join("multi.hmm");
    let first = std::fs::read("hmmer/testsuite/20aa.hmm").unwrap();
    let second = std::fs::read("hmmer/tutorial/fn3.hmm").unwrap();
    let mut both = first;
    both.extend_from_slice(&second);
    std::fs::write(&multi_hmm, both).unwrap();

    let output = Command::new(hmmer())
        .args(["logo", multi_hmm.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.matches("Residue heights\n").count(), 1);
    assert!(stdout.contains("20:"));
    assert!(!stdout.contains("86:"));
}

#[test]
fn hmmsearch_tblout_has_c_style_header_options_and_footer() {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("hits.tbl");
    let pfamtblout = dir.path().join("hits.pfam.tbl");
    let output = Command::new(hmmer())
        .args([
            "search",
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
            "--pfamtblout",
            pfamtblout.to_str().unwrap(),
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(&format!(
        "# per-seq hits tabular output:     {}",
        tblout.display()
    )));
    assert!(stdout.contains(&format!(
        "# pfam-style tabular hit output:   {}",
        pfamtblout.display()
    )));
    assert!(stdout.contains("# show alignments in output:       no"));

    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert!(tbl.contains("# Program:         hmmsearch\n"));
    assert!(tbl.contains("# Pipeline mode:   SEARCH\n"));
    assert!(tbl.contains("# Query file:      hmmer/tutorial/fn3.hmm\n"));
    assert!(tbl.contains("# Target file:     hmmer/tutorial/7LESS_DROME\n"));
    assert!(tbl.contains("# Option settings: hmmer search --noali --tblout "));
    assert!(tbl.ends_with("# [ok]\n"));
}

#[test]
fn phmmer_reports_c_style_header_option_annotations() {
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--noali",
            "--popen",
            "0.03",
            "--pextend",
            "0.3",
            "-E",
            "1",
            "--domE",
            "2",
            "--incE",
            "0.5",
            "--incdomE",
            "0.6",
            "--F1",
            "0.1",
            "--F2",
            "0.2",
            "--F3",
            "0.3",
            "--nobias",
            "--nonull2",
            "-Z",
            "10",
            "--domZ",
            "5",
            "--seed",
            "42",
            "--EmL",
            "80",
            "--EmN",
            "30",
            "--EvL",
            "90",
            "--EvN",
            "40",
            "--EfL",
            "70",
            "--EfN",
            "50",
            "--Eft",
            "0.02",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "# gap open probability:            0.030000\n",
        "# gap extend probability:          0.300000\n",
        "# sequence reporting threshold:    E-value <= 1\n",
        "# domain reporting threshold:      E-value <= 2\n",
        "# sequence inclusion threshold:    E-value <= 0.5\n",
        "# domain inclusion threshold:      E-value <= 0.6\n",
        "# MSV filter P threshold:       <= 0.1\n",
        "# Vit filter P threshold:       <= 0.2\n",
        "# Fwd filter P threshold:       <= 0.3\n",
        "# biased composition HMM filter:   off\n",
        "# null2 bias corrections:          off\n",
        "# sequence search space set to:    10\n",
        "# domain search space set to:      5\n",
        "# random number seed set to:       42\n",
        "# seq length, MSV Gumbel mu fit:   80\n",
        "# seq number, MSV Gumbel mu fit:   30\n",
        "# seq length, Vit Gumbel mu fit:   90\n",
        "# seq number, Vit Gumbel mu fit:   40\n",
        "# seq length, Fwd exp tau fit:     70\n",
        "# seq number, Fwd exp tau fit:     50\n",
        "# tail mass for Fwd exp tau fit:   0.020000\n",
    ] {
        assert!(
            stdout.contains(expected),
            "{expected:?} missing from:\n{stdout}"
        );
    }
}

#[test]
fn phmmer_compact_short_options_are_reported_in_stdout_header() {
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--noali",
            "-E1e-3",
            "-Z123",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "# sequence reporting threshold:    E-value <= 0.001\n",
        "# sequence search space set to:    123\n",
    ] {
        assert!(
            stdout.contains(expected),
            "{expected:?} missing from:\n{stdout}"
        );
    }
}

#[test]
fn jackhmmer_reports_c_style_header_option_annotations_and_cpu_only_when_requested() {
    let dir = tempfile::tempdir().unwrap();
    let query_path = dir.path().join("query.fa");
    std::fs::write(&query_path, ">query\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let default_output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "-N",
            "1",
            "--noali",
            query_path.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .env_remove("HMMER_NCPU")
        .output()
        .unwrap();
    assert!(
        default_output.status.success(),
        "{}",
        String::from_utf8_lossy(&default_output.stderr)
    );
    let default_stdout = String::from_utf8(default_output.stdout).unwrap();
    assert!(!default_stdout.contains("# number of worker threads:"));

    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "-N",
            "1",
            "--noali",
            "--cpu",
            "1",
            "--popen",
            "0.03",
            "--pextend",
            "0.3",
            "-E",
            "1",
            "--domE",
            "2",
            "--incE",
            "0.5",
            "--incdomE",
            "0.6",
            "--F1",
            "0.1",
            "--F2",
            "0.2",
            "--F3",
            "0.3",
            "--nobias",
            "--nonull2",
            "-Z",
            "10",
            "--domZ",
            "5",
            "--seed",
            "42",
            "--EmL",
            "80",
            "--EmN",
            "30",
            "--EvL",
            "90",
            "--EvN",
            "40",
            "--EfL",
            "70",
            "--EfN",
            "50",
            "--Eft",
            "0.02",
            query_path.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "# gap open probability:            0.030000\n",
        "# gap extend probability:          0.300000\n",
        "# sequence reporting threshold:    E-value <= 1\n",
        "# domain reporting threshold:      E-value <= 2\n",
        "# sequence inclusion threshold:    E-value <= 0.5\n",
        "# domain inclusion threshold:      E-value <= 0.6\n",
        "# MSV filter P threshold:       <= 0.1\n",
        "# Vit filter P threshold:       <= 0.2\n",
        "# Fwd filter P threshold:       <= 0.3\n",
        "# biased composition HMM filter:   off\n",
        "# null2 bias corrections:          off\n",
        "# sequence search space set to:    10\n",
        "# domain search space set to:      5\n",
        "# random number seed set to:       42\n",
        "# number of worker threads:        1\n",
        "# seq length, MSV Gumbel mu fit:   80\n",
        "# seq number, MSV Gumbel mu fit:   30\n",
        "# seq length, Vit Gumbel mu fit:   90\n",
        "# seq number, Vit Gumbel mu fit:   40\n",
        "# seq length, Fwd exp tau fit:     70\n",
        "# seq number, Fwd exp tau fit:     50\n",
        "# tail mass for Fwd exp tau fit:   0.020000\n",
    ] {
        assert!(
            stdout.contains(expected),
            "{expected:?} missing from:\n{stdout}"
        );
    }
}

#[test]
fn hmmbuild_and_hmmsim_advertise_visible_stall_debug_option() {
    for (subcmd, help) in [
        (
            "build",
            "arrest after start: for attaching debugger to process",
        ),
        ("sim", "arrest after start: for debugging MPI under gdb"),
    ] {
        let output = Command::new(hmmer())
            .args([subcmd, "--help"])
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(
            stdout.contains("--stall"),
            "{subcmd} help omitted --stall:\n{stdout}"
        );
        assert!(
            stdout.contains(help),
            "{subcmd} help omitted C-style stall help:\n{stdout}"
        );
    }
}

#[test]
fn scan_help_describes_target_thresholds_not_sequence_thresholds() {
    for (subcmd, expected) in [
        (
            "scan",
            [
                "Report profiles <= this E-value threshold",
                "Report profiles >= this score threshold",
                "Include profiles <= this E-value threshold",
                "Include profiles >= this score threshold",
            ],
        ),
        (
            "nhmmscan",
            [
                "Report models <= this E-value threshold",
                "Report models >= this score threshold",
                "Include models <= this E-value threshold",
                "Include models >= this score threshold",
            ],
        ),
    ] {
        let output = Command::new(hmmer())
            .args([subcmd, "--help"])
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(
            !stdout.contains("Report sequences <= this E-value threshold"),
            "{subcmd} help still describes reporting threshold as sequences:\n{stdout}"
        );
        assert!(
            !stdout.contains("Include sequences <= this E-value threshold"),
            "{subcmd} help still describes inclusion threshold as sequences:\n{stdout}"
        );
        for help in expected {
            assert!(
                stdout.contains(help),
                "{subcmd} help omitted {help:?}:\n{stdout}"
            );
        }
    }
}

#[test]
fn hmmsearch_reports_threshold_and_search_space_options_in_header() {
    let output = Command::new(hmmer())
        .args([
            "search",
            "--noali",
            "-E",
            "1",
            "--domE",
            "2",
            "--incE",
            "0.5",
            "--incdomE",
            "0.6",
            "--F1",
            "0.1",
            "--nonull2",
            "-Z",
            "10",
            "--domZ",
            "5",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "# sequence reporting threshold:    E-value <= 1\n",
        "# domain reporting threshold:      E-value <= 2\n",
        "# sequence inclusion threshold:    E-value <= 0.5\n",
        "# domain inclusion threshold:      E-value <= 0.6\n",
        "# MSV filter P threshold:       <= 0.1\n",
        "# null2 bias corrections:          off\n",
        "# sequence search space set to:    10\n",
        "# domain search space set to:      5\n",
    ] {
        assert!(
            stdout.contains(expected),
            "{expected:?} missing from:\n{stdout}"
        );
    }
}

#[test]
fn hmmsearch_compact_short_options_are_reported_in_stdout_header() {
    let output = Command::new(hmmer())
        .args([
            "search",
            "--noali",
            "-E1e-3",
            "-Z123",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "# sequence reporting threshold:    E-value <= 0.001\n",
        "# sequence search space set to:    123\n",
    ] {
        assert!(
            stdout.contains(expected),
            "{expected:?} missing from:\n{stdout}"
        );
    }
}

#[test]
fn hmmsearch_parallel_workers_honor_sequence_reporting_thresholds() {
    for threshold_args in [["-E", "1e-200"], ["-T", "9999"]] {
        let output = Command::new(hmmer())
            .args(["search", "--cpu", "2", "--noali"])
            .args(threshold_args)
            .args(["hmmer/tutorial/fn3.hmm", "hmmer/tutorial/7LESS_DROME"])
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(
            stdout.contains("[No hits detected that satisfy reporting thresholds]"),
            "{stdout}"
        );
    }
}

/// Regression for the "bias columns clamp small negatives to 0.0" audit finding.
/// C HMMER prints the raw per-domain bias correction
/// (`dcl[d].dombias * eslCONST_LOG2R`, p7_tophits.c:1664) unclamped, which can be
/// slightly negative. `RVT_1_pfam.hmm --max` vs `human_swissprot_2k.fasta` produces
/// several negative domain-bias values; the Rust port must reproduce them byte-for-byte
/// rather than flooring at 0.0. Guards against re-introducing a `.max(0.0)` clamp on the
/// bias fed to the domtblout writer.
#[test]
fn hmmsearch_domtblout_preserves_negative_domain_bias_like_c() {
    let hmm = format!("{}/test_data/RVT_1_pfam.hmm", project_root());
    let db = format!("{}/test_data/human_swissprot_2k.fasta", project_root());
    if !std::path::Path::new(&hmm).exists() || !std::path::Path::new(&db).exists() {
        eprintln!("skipping: fixtures not present");
        return;
    }

    let rust_dt = std::env::temp_dir().join("rust_neg_bias_domtbl.txt");
    let rust = Command::new(hmmer())
        .args(["hmmsearch", "--max"])
        .arg("--domtblout")
        .arg(&rust_dt)
        .args(["-o", "/dev/null"])
        .args([&hmm, &db])
        .output()
        .unwrap();
    assert!(
        rust.status.success(),
        "{}",
        String::from_utf8_lossy(&rust.stderr)
    );
    let stdout = std::fs::read_to_string(&rust_dt).unwrap();

    // Collect (target, dom-bias) pairs for negative-bias domains. domtblout column 14
    // (1-based) — index 13 (0-based) — is the per-domain bias correction in bits.
    let neg: Vec<(String, String)> = stdout
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            if f.len() > 13 && f[13].parse::<f32>().map(|v| v < 0.0).unwrap_or(false) {
                Some((f[0].to_string(), f[13].to_string()))
            } else {
                None
            }
        })
        .collect();

    // These are the values C HMMER 3.4 emits on this fixture (verified against
    // `hmmer/src/hmmsearch`); they are genuinely negative, not clamped to 0.0.
    assert!(
        neg.iter().any(|(t, b)| t.contains("ZF69B_HUMAN") && b == "-1.3"),
        "expected ZF69B_HUMAN dom-bias -1.3, got {neg:?}"
    );
    assert!(
        neg.iter().any(|(t, b)| t.contains("DYN1_HUMAN") && b == "-2.6"),
        "expected DYN1_HUMAN dom-bias -2.6, got {neg:?}"
    );
    assert!(
        neg.iter().any(|(t, b)| t.contains("CACO1_HUMAN") && b == "-3.0"),
        "expected CACO1_HUMAN dom-bias -3.0, got {neg:?}"
    );

    // If the C binary is available, require byte-identical domtblout data rows so this
    // doubles as a true parity check (not merely a snapshot of hard-coded values).
    let c_bin = format!("{}/hmmer/src/hmmsearch", project_root());
    if std::path::Path::new(&c_bin).exists() {
        let c_dt = std::env::temp_dir().join("c_neg_bias_domtbl.txt");
        let c = Command::new(&c_bin)
            .args(["--max"])
            .arg("--domtblout")
            .arg(&c_dt)
            .args(["-o", "/dev/null"])
            .args([&hmm, &db])
            .output()
            .unwrap();
        assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
        let c_out = std::fs::read_to_string(&c_dt).unwrap();
        let mut c_rows: Vec<&str> = c_out.lines().filter(|l| !l.starts_with('#')).collect();
        let mut r_rows: Vec<&str> = stdout.lines().filter(|l| !l.starts_with('#')).collect();
        c_rows.sort_unstable();
        r_rows.sort_unstable();
        assert_eq!(c_rows, r_rows, "domtblout data rows must match C exactly");
    }
}

#[test]
fn search_commands_accept_sequence_stdin() {
    let seq = std::fs::read("hmmer/tutorial/7LESS_DROME").unwrap();
    let mut child = Command::new(hmmer())
        .args(["search", "--noali", "hmmer/tutorial/fn3.hmm", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(&seq).unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# target sequence database:        -\n"));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn phmmer_parallel_workers_honor_sequence_reporting_thresholds() {
    for threshold_args in [["-E", "1e-300"], ["-T", "9999"]] {
        let output = Command::new(hmmer())
            .args(["phmmer", "--cpu", "2", "--noali"])
            .args(threshold_args)
            .args(["hmmer/tutorial/HBB_HUMAN", "hmmer/tutorial/globins45.fa"])
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(
            stdout.contains("[No hits detected that satisfy reporting thresholds]"),
            "{stdout}"
        );
    }
}

#[test]
fn phmmer_advertises_all_requested_table_outputs_and_footers() {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("hits.tbl");
    let domtblout = dir.path().join("domains.tbl");
    let pfamtblout = dir.path().join("pfam.tbl");
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--tblout",
            tblout.to_str().unwrap(),
            "--domtblout",
            domtblout.to_str().unwrap(),
            "--pfamtblout",
            pfamtblout.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(&format!(
        "# per-seq hits tabular output:     {}",
        tblout.display()
    )));
    assert!(stdout.contains(&format!(
        "# per-dom hits tabular output:     {}",
        domtblout.display()
    )));
    assert!(stdout.contains(&format!(
        "# pfam-style tabular hit output:   {}",
        pfamtblout.display()
    )));

    for path in [&tblout, &domtblout, &pfamtblout] {
        let text = std::fs::read_to_string(path).unwrap();
        assert!(text.contains("# Program:         phmmer\n"));
        assert!(text.contains("# Pipeline mode:   SEARCH\n"));
        assert!(text.ends_with("# [ok]\n"));
    }
}

#[test]
fn phmmer_output_file_is_advertised_like_c() {
    let dir = tempfile::tempdir().unwrap();
    let output_path = dir.path().join("phmmer.out");
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "-o",
            output_path.to_str().unwrap(),
            "hmmer/tutorial/HBB_HUMAN",
            "hmmer/tutorial/globins45.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(&output_path).unwrap();
    assert!(text.contains(&format!(
        "# output directed to file:         {}",
        output_path.display()
    )));
}

#[test]
fn phmmer_noali_suppresses_alignment_blocks() {
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--noali",
            "-E",
            "1000",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# show alignments in output:       no\n"));
    assert!(stdout.contains("Domain annotation for each sequence:\n"));
    assert!(!stdout.contains("Domain annotation for each sequence (and alignments):"));
    assert!(!stdout.contains("Alignments for each domain:"));
}

#[test]
fn phmmer_acc_and_text_width_controls_are_reported() {
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--acc",
            "--textw",
            "140",
            "--noali",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# prefer accessions over names:    yes\n"));
    assert!(stdout.contains("# max ASCII text line length:      140\n"));

    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--notextw",
            "--noali",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# max ASCII text line length:      unlimited\n"));
}

#[test]
fn phmmer_accepts_explicit_default_substitution_matrix() {
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--mx",
            "BLOSUM62",
            "--noali",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# subst score matrix (built-in):   BLOSUM62\n"));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn phmmer_accepts_nondefault_builtin_substitution_matrix() {
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--mx",
            "PAM30",
            "--noali",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# subst score matrix (built-in):   PAM30\n"));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn phmmer_accepts_custom_substitution_matrix_file() {
    let dir = tempfile::tempdir().unwrap();
    let mxfile = dir.path().join("custom.mx");
    std::fs::write(&mxfile, custom_protein_score_matrix()).unwrap();

    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--mxfile",
            mxfile.to_str().unwrap(),
            "--noali",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(&format!(
        "# subst score matrix (file):       {}",
        mxfile.display()
    )));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn jackhmmer_accepts_explicit_default_substitution_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let query_path = dir.path().join("query.fa");
    std::fs::write(&query_path, ">query\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "--mx",
            "BLOSUM62",
            "-N",
            "1",
            "--noali",
            query_path.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# subst score matrix (built-in):   BLOSUM62\n"));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn jackhmmer_accepts_nondefault_builtin_substitution_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let query_path = dir.path().join("query.fa");
    std::fs::write(&query_path, ">query\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "--mx",
            "PAM30",
            "-N",
            "1",
            "--noali",
            query_path.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# subst score matrix (built-in):   PAM30\n"));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn jackhmmer_accepts_custom_substitution_matrix_file() {
    let dir = tempfile::tempdir().unwrap();
    let query_path = dir.path().join("query.fa");
    let mxfile = dir.path().join("custom.mx");
    std::fs::write(&query_path, ">query\nACDEFGHIKLMNPQRSTVWY\n").unwrap();
    std::fs::write(&mxfile, custom_protein_score_matrix()).unwrap();

    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "--mxfile",
            mxfile.to_str().unwrap(),
            "-N",
            "1",
            "--noali",
            query_path.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(&format!(
        "# subst score matrix (file):       {}",
        mxfile.display()
    )));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn phmmer_accepts_fasta_qformat_tformat_assertions() {
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--qformat",
            "fasta",
            "--tformat",
            "fasta",
            "--noali",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# query <seqfile> format asserted: fasta\n"));
    assert!(stdout.contains("# target <seqdb> format asserted:  fasta\n"));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn phmmer_preserves_qformat_tformat_header_spelling() {
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--qformat",
            "FASTA",
            "--tformat",
            "FASTA",
            "--noali",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# query <seqfile> format asserted: FASTA\n"));
    assert!(stdout.contains("# target <seqdb> format asserted:  FASTA\n"));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn phmmer_accepts_stockholm_qformat_tformat_assertions() {
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--qformat",
            "stockholm",
            "--tformat",
            "stockholm",
            "--noali",
            "hmmer/testsuite/20aa.sto",
            "hmmer/testsuite/20aa.sto",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# query <seqfile> format asserted: stockholm\n"));
    assert!(stdout.contains("# target <seqdb> format asserted:  stockholm\n"));
    assert!(stdout.contains("Query:       seq1  [L=20]"), "{stdout}");
}

#[test]
fn phmmer_writes_alignment_output_file() {
    let dir = tempfile::tempdir().unwrap();
    let ali = dir.path().join("phmmer_hits.sto");
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "-A",
            ali.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(&format!(
        "# MSA of hits saved to file:       {}",
        ali.display()
    )));
    // C phmmer echoes a per-query confirmation line carrying the MSA's
    // sequence count (phmmer.c:633).
    assert!(
        stdout.contains(&format!(
            "# Alignment of 4 hits satisfying inclusion thresholds saved to: {}",
            ali.display()
        )),
        "{stdout}"
    );
    let msa = std::fs::read_to_string(ali).unwrap();
    // Faithful to C `p7_tophits_Alignment` + `esl_msafile_Write`: single-space
    // GF ID, a phmmer AU author line, `name/from-to` names, `#=GS DE [subseq
    // from]` lines, and an RF row.
    assert!(msa.starts_with("# STOCKHOLM 1.0\n"));
    assert!(msa.contains("#=GF ID test1\n"), "{msa}");
    assert!(msa.contains("#=GF AU phmmer (HMMER 3.4)\n"), "{msa}");
    assert!(msa.contains("#=GS test1/1-20 DE [subseq from] test1"), "{msa}");
    assert!(msa.lines().any(|l| l.starts_with("#=GC RF")), "{msa}");
    assert!(msa.trim_end().ends_with("//"));
}

/// Regression for audit finding F1: phmmer `-A` must produce the real
/// `p7_tophits_Alignment` MSA, byte-identical to the bundled C phmmer (modulo
/// the `#=GC PP_cons` rounding row).
#[test]
fn phmmer_alignment_output_matches_c_binary() {
    let dir = tempfile::tempdir().unwrap();
    let c_ali = dir.path().join("c.sto");
    let r_ali = dir.path().join("r.sto");

    let c = Command::new(c_phmmer())
        .args([
            "-A",
            c_ali.to_str().unwrap(),
            "hmmer/tutorial/HBB_HUMAN",
            "hmmer/tutorial/globins45.fa",
        ])
        .current_dir(project_root())
        .output()
        .unwrap();
    assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));

    let r = Command::new(hmmer())
        .args([
            "phmmer",
            "-A",
            r_ali.to_str().unwrap(),
            "hmmer/tutorial/HBB_HUMAN",
            "hmmer/tutorial/globins45.fa",
        ])
        .current_dir(project_root())
        .output()
        .unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));

    let c_sto = std::fs::read_to_string(&c_ali).unwrap();
    let r_sto = std::fs::read_to_string(&r_ali).unwrap();
    assert_eq!(
        stockholm_structure_lines(&c_sto),
        stockholm_structure_lines(&r_sto),
        "phmmer -A diverged from C\n--- C ---\n{c_sto}\n--- R ---\n{r_sto}"
    );
}

#[test]
fn jackhmmer_output_file_is_advertised_like_c() {
    let dir = tempfile::tempdir().unwrap();
    let query_path = dir.path().join("query.fa");
    let output_path = dir.path().join("jackhmmer.out");
    std::fs::write(&query_path, ">query\nACDEFGHIKLMNPQRSTVWY\n").unwrap();
    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "-N",
            "1",
            "-o",
            output_path.to_str().unwrap(),
            query_path.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(&output_path).unwrap();
    assert!(text.contains(&format!(
        "# output directed to file:         {}",
        output_path.display()
    )));
}

#[test]
fn jackhmmer_output_controls_are_reported_and_noali_suppresses_blocks() {
    let dir = tempfile::tempdir().unwrap();
    let query_path = dir.path().join("query.fa");
    std::fs::write(&query_path, ">query\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "-N",
            "1",
            "--acc",
            "--textw",
            "140",
            "--noali",
            query_path.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# prefer accessions over names:    yes\n"));
    assert!(stdout.contains("# show alignments in output:       no\n"));
    assert!(stdout.contains("# max ASCII text line length:      140\n"));
    assert!(stdout.contains("Domain annotation for each sequence:\n"));
    assert!(!stdout.contains("Alignments for each domain:"));

    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "-N",
            "1",
            "--notextw",
            "--noali",
            query_path.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# max ASCII text line length:      unlimited\n"));
}

#[test]
fn jackhmmer_accepts_fasta_qformat_tformat_assertions() {
    let dir = tempfile::tempdir().unwrap();
    let query_path = dir.path().join("query.fa");
    std::fs::write(&query_path, ">query\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "-N",
            "1",
            "--qformat",
            "fasta",
            "--tformat",
            "fasta",
            "--noali",
            query_path.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# query <seqfile> format asserted: fasta\n"));
    assert!(stdout.contains("# target <seqdb> format asserted:  fasta\n"));
    assert!(stdout.contains("Scores for complete sequences"));
    assert!(stdout.contains("[ok]"));
}

#[test]
fn jackhmmer_accepts_stockholm_qformat_tformat_assertions() {
    let dir = tempfile::tempdir().unwrap();
    let query_path = dir.path().join("query.sto");
    let target_path = dir.path().join("targets.sto");
    std::fs::write(
        &query_path,
        "# STOCKHOLM 1.0\nquery ACDEFGHIKLMNPQRSTVWY\n//\n",
    )
    .unwrap();
    std::fs::write(
        &target_path,
        "# STOCKHOLM 1.0\ntarget ACDEFGHIKLMNPQRSTVWY\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "-N",
            "1",
            "--qformat",
            "stockholm",
            "--tformat",
            "stockholm",
            "--noali",
            query_path.to_str().unwrap(),
            target_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# query <seqfile> format asserted: stockholm\n"));
    assert!(stdout.contains("# target <seqdb> format asserted:  stockholm\n"));
    assert!(stdout.contains("Scores for complete sequences"));
    assert!(stdout.contains("[ok]"));
}

#[test]
fn hmmsearch_multi_query_separates_stdout_and_single_tblout_header() {
    let dir = tempfile::tempdir().unwrap();
    let multi_hmm = dir.path().join("multi.hmm");
    let tblout = dir.path().join("hits.tbl");
    let one = std::fs::read("hmmer/tutorial/fn3.hmm").unwrap();
    let mut both = one.clone();
    both.extend_from_slice(&one);
    std::fs::write(&multi_hmm, both).unwrap();

    let output = Command::new(hmmer())
        .args([
            "search",
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
            multi_hmm.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.matches("\n//\n").count(), 2);

    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert_eq!(tbl.matches("# target name").count(), 1);
}

fn c_hmmsearch() -> String {
    format!("{}/hmmer/src/hmmsearch", project_root())
}

fn c_phmmer() -> String {
    format!("{}/hmmer/src/phmmer", project_root())
}

/// Lines of a Stockholm `-A` file that should be byte-for-byte identical to C:
/// everything except the per-column `#=GC PP_cons` row, whose averaged
/// posterior-probability quantization can differ from C by one bucket in the
/// last position (a pre-existing DP-rounding difference, unrelated to MSA
/// formatting / `p7_tophits_Alignment`).
fn stockholm_structure_lines(sto: &str) -> Vec<&str> {
    sto.lines()
        .filter(|l| !l.starts_with("#=GC PP_cons"))
        .collect()
}

#[test]
fn hmmsearch_writes_alignment_output_file() {
    let dir = tempfile::tempdir().unwrap();
    let ali = dir.path().join("hits.sto");
    let output = Command::new(hmmer())
        .args([
            "search",
            "--noali",
            "-A",
            ali.to_str().unwrap(),
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(&format!(
        "# MSA of all hits saved to file:   {}",
        ali.display()
    )));
    // C hmmsearch echoes a per-search confirmation line carrying the MSA's
    // sequence count (hmmsearch.c:569).
    assert!(
        stdout.contains(&format!(
            "# Alignment of 7 hits satisfying inclusion thresholds saved to: {}",
            ali.display()
        )),
        "{stdout}"
    );
    let msa = std::fs::read_to_string(ali).unwrap();
    // Faithful to C `p7_tophits_Alignment` + `esl_msafile_Write`: single-space
    // GF ID, an AU author line, `name/from-to` sequence names, `#=GS DE
    // [subseq from]` lines, and a consensus RF row.
    assert!(msa.starts_with("# STOCKHOLM 1.0\n"));
    assert!(msa.contains("#=GF ID fn3\n"), "{msa}");
    assert!(msa.contains("#=GF AU hmmsearch (HMMER 3.4)\n"), "{msa}");
    assert!(msa.contains("7LESS_DROME/439-520"), "{msa}");
    assert!(
        msa.contains("DE [subseq from] RecName: Full=Protein sevenless;"),
        "{msa}"
    );
    assert!(msa.lines().any(|l| l.starts_with("#=GC RF")), "{msa}");
    assert!(msa.trim_end().ends_with("//"));
}

/// Regression for audit finding F1: hmmsearch `-A` must produce the real
/// `p7_tophits_Alignment` MSA, byte-identical to the bundled C hmmsearch
/// (modulo the `#=GC PP_cons` rounding row). Uses globins4/globins45, which
/// have no per-sequence accessions or multi-space descriptions, so the only
/// divergence is PP_cons.
#[test]
fn hmmsearch_alignment_output_matches_c_binary() {
    let dir = tempfile::tempdir().unwrap();
    let c_ali = dir.path().join("c.sto");
    let r_ali = dir.path().join("r.sto");

    let c = Command::new(c_hmmsearch())
        .args([
            "-A",
            c_ali.to_str().unwrap(),
            "hmmer/tutorial/globins4.hmm",
            "hmmer/tutorial/globins45.fa",
        ])
        .current_dir(project_root())
        .output()
        .unwrap();
    assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));

    let r = Command::new(hmmer())
        .args([
            "hmmsearch",
            "-A",
            r_ali.to_str().unwrap(),
            "hmmer/tutorial/globins4.hmm",
            "hmmer/tutorial/globins45.fa",
        ])
        .current_dir(project_root())
        .output()
        .unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));

    let c_sto = std::fs::read_to_string(&c_ali).unwrap();
    let r_sto = std::fs::read_to_string(&r_ali).unwrap();
    assert_eq!(
        stockholm_structure_lines(&c_sto),
        stockholm_structure_lines(&r_sto),
        "hmmsearch -A diverged from C\n--- C ---\n{c_sto}\n--- R ---\n{r_sto}"
    );
}

#[test]
fn hmmsearch_rejects_multi_query_sequence_stdin() {
    let dir = tempfile::tempdir().unwrap();
    let multi_hmm = dir.path().join("multi.hmm");
    let one = std::fs::read("hmmer/tutorial/fn3.hmm").unwrap();
    let mut both = one.clone();
    both.extend_from_slice(&one);
    std::fs::write(&multi_hmm, both).unwrap();

    let seq = std::fs::read("hmmer/tutorial/7LESS_DROME").unwrap();
    let mut child = Command::new(hmmer())
        .args(["search", multi_hmm.to_str().unwrap(), "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(&seq).unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("isn't rewindable"));
}

#[test]
fn hmmsearch_accepts_query_hmm_from_stdin() {
    let hmm = std::fs::read("hmmer/testsuite/20aa.hmm").unwrap();
    let output = run_with_stdin(
        &["search", "--noali", "-", "hmmer/testsuite/20aa-alitest.fa"],
        &hmm,
    );

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Query:       test"));
    assert!(stdout.contains("[ok]"));
}

#[test]
fn hmmsearch_cpu_zero_uses_serial_output_path() {
    let default = Command::new(hmmer())
        .args([
            "search",
            "--noali",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .env_remove("HMMER_NCPU")
        .output()
        .unwrap();
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );

    let cpu0 = Command::new(hmmer())
        .args([
            "search",
            "--cpu",
            "0",
            "--noali",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        cpu0.status.success(),
        "{}",
        String::from_utf8_lossy(&cpu0.stderr)
    );

    let cpu1 = Command::new(hmmer())
        .args([
            "search",
            "--cpu",
            "1",
            "--noali",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        cpu1.status.success(),
        "{}",
        String::from_utf8_lossy(&cpu1.stderr)
    );

    let default_stdout = String::from_utf8(default.stdout).unwrap();
    let cpu0_stdout = String::from_utf8(cpu0.stdout).unwrap();
    let cpu1_stdout = String::from_utf8(cpu1.stdout).unwrap();
    assert!(!default_stdout.contains("# number of worker threads:"));
    assert!(cpu0_stdout.contains("# number of worker threads:        0\n"));
    assert_eq!(
        stdout_without_cpu_header(&cpu0_stdout),
        stdout_without_cpu_header(&cpu1_stdout)
    );
}

#[test]
fn hmmscan_tblout_has_scan_footer_and_pfamtblout_has_search_footer() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );
    let tblout = dir.path().join("hits.tbl");
    let pfamtblout = dir.path().join("hits.pfam.tbl");
    let output = Command::new(hmmer())
        .args([
            "scan",
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
            "--pfamtblout",
            pfamtblout.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# Copyright (C) 2023 Howard Hughes Medical Institute.\n"));
    assert!(stdout.contains("# Freely distributed under the BSD open source license.\n"));
    assert!(stdout.contains("# query sequence file:             hmmer/tutorial/7LESS_DROME\n"));
    assert!(stdout.contains(&format!(
        "# target HMM database:             {}\n",
        hmmdb.display()
    )));
    assert!(stdout.contains(&format!(
        "# pfam-style tabular hit output:   {}\n",
        pfamtblout.display()
    )));
    assert!(stdout.contains("Accession:   P13368\n"));
    // Multi-line UniProt DE: Easel preserves the continuation line's leading
    // whitespace (esl_sqio_ascii.c strips only the 5-char "DE   " prefix +
    // trailing whitespace) and esl_sq_AppendDesc joins with a single space.
    // C therefore emits the wide inter-field spacing, not a single space.
    assert!(stdout
        .contains("Description: RecName: Full=Protein sevenless;          EC=2.7.10.1;\n"));
    assert!(stdout.contains("Internal pipeline statistics summary:\n"));
    assert!(stdout.contains("Domain annotation for each model:\n"));
    assert!(stdout.contains(">> fn3  Fibronectin type III domain\n"));
    assert!(stdout.contains("hmmfrom  hmm to    alifrom  ali to    envfrom  env to"));
    assert!(!stdout.contains("Alignments for each domain:"));
    assert!(stdout.contains("Target model(s):"));
    assert!(stdout.contains("Passed MSV filter:"));
    assert!(stdout.contains("Passed bias filter:"));
    assert!(stdout.contains("Passed Vit filter:"));
    assert!(stdout.contains("Passed Fwd filter:"));
    assert!(stdout.contains("Initial search space (Z):"));
    assert!(stdout.contains("Domain search space  (domZ):"));

    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert!(tbl.contains("# Program:         hmmscan\n"));
    assert!(tbl.contains("# Pipeline mode:   SCAN\n"));
    assert!(tbl.contains("# Query file:      hmmer/tutorial/7LESS_DROME\n"));
    assert!(tbl.contains(&format!("# Target file:     {}\n", hmmdb.display())));
    assert!(tbl.ends_with("# [ok]\n"));

    let pfam = std::fs::read_to_string(pfamtblout).unwrap();
    assert!(pfam.contains("# Program:         hmmscan\n"));
    assert!(pfam.contains("# Pipeline mode:   SEARCH\n"));
    assert!(pfam.ends_with("# [ok]\n"));
}

#[test]
fn hmmscan_accepts_fasta_qformat_assertion() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "scan",
            "--qformat",
            "fasta",
            "--noali",
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/globins45.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# input seqfile format asserted:   fasta\n"));
    assert!(stdout.contains("Scores for complete sequence"));
    assert!(stdout.contains("[ok]"));
}

#[test]
fn hmmscan_preserves_qformat_header_spelling() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "scan",
            "--qformat",
            "FASTA",
            "--noali",
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/globins45.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# input seqfile format asserted:   FASTA\n"));
    assert!(stdout.contains("Scores for complete sequence"));
    assert!(stdout.contains("[ok]"));
}

#[test]
fn hmmscan_accepts_stockholm_qformat_assertion() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("20aa.hmm");
    std::fs::copy("hmmer/testsuite/20aa.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "scan",
            "--qformat",
            "stockholm",
            "--noali",
            hmmdb.to_str().unwrap(),
            "hmmer/testsuite/20aa.sto",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# input seqfile format asserted:   stockholm\n"));
    assert!(stdout.contains("Scores for complete sequence"));
    assert!(stdout.contains("[ok]"));
}

#[test]
fn hmmscan_output_file_acc_noali_and_textw_work() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    let out = dir.path().join("scan.out");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "scan",
            "-o",
            out.to_str().unwrap(),
            "--acc",
            "--noali",
            "--cpu",
            "1",
            "--textw",
            "140",
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("# prefer accessions over names:    yes\n"));
    assert!(text.contains("# multithread parallelization:     1 workers\n"));
    assert!(text.contains("# max ASCII text line length:      140\n"));
    assert!(!text.contains("Alignments for each domain:"));
}

#[test]
fn hmmscan_cpu_zero_is_threading_off_and_matches_serial_output() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let cpu0 = Command::new(hmmer())
        .args([
            "scan",
            "--cpu",
            "0",
            "--noali",
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        cpu0.status.success(),
        "{}",
        String::from_utf8_lossy(&cpu0.stderr)
    );

    let cpu1 = Command::new(hmmer())
        .args([
            "scan",
            "--cpu",
            "1",
            "--noali",
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        cpu1.status.success(),
        "{}",
        String::from_utf8_lossy(&cpu1.stderr)
    );

    let cpu0_stdout = String::from_utf8(cpu0.stdout).unwrap();
    let cpu1_stdout = String::from_utf8(cpu1.stdout).unwrap();
    assert!(cpu0_stdout.contains("# multithread parallelization:     off\n"));
    assert_eq!(
        stdout_without_cpu_header(&cpu0_stdout),
        stdout_without_cpu_header(&cpu1_stdout)
    );
}

#[test]
fn hmmscan_search_space_overrides_are_reported_in_stdout_and_table_footers() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let tblout = dir.path().join("hits.tbl");
    let domtblout = dir.path().join("domains.tbl");
    let output = Command::new(hmmer())
        .args([
            "scan",
            "--noali",
            "-Z",
            "123",
            "--domZ",
            "5",
            "--tblout",
            tblout.to_str().unwrap(),
            "--domtblout",
            domtblout.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Initial search space (Z):                123  [as set by --Z on cmdline]")
    );
    assert!(stdout
        .contains("Domain search space  (domZ):               5  [as set by --domZ on cmdline]"));

    for path in [&tblout, &domtblout] {
        let text = std::fs::read_to_string(path).unwrap();
        assert!(text.contains("# Program:         hmmscan\n"));
        // Audit F4: the `hmmer` wrapper token is stripped and the subcommand
        // alias is normalized to the canonical `hmmscan`, matching phmmer/jackhmmer.
        assert!(
            text.contains("# Option settings: hmmscan --noali -Z 123 --domZ 5 "),
            "footer must strip wrapper and normalize to hmmscan: {text}"
        );
        assert!(text.ends_with("# [ok]\n"));
    }
}

#[test]
fn hmmscan_reports_profile_threshold_options_in_header() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    for (args, expected) in [
        (
            vec!["-E", "1", "--incE", "0.5"],
            vec![
                "# profile reporting threshold:     E-value <= 1\n",
                "# profile inclusion threshold:     E-value <= 0.5\n",
            ],
        ),
        (
            vec![
                "--domE",
                "2",
                "--incdomE",
                "0.6",
                "--F1",
                "0.1",
                "--F2",
                "0.2",
                "--F3",
                "0.3",
                "-Z",
                "10",
                "--domZ",
                "5",
                "--seed",
                "7",
            ],
            vec![
                "# domain reporting threshold:      E-value <= 2\n",
                "# domain inclusion threshold:      E-value <= 0.6\n",
                "# MSV filter P threshold:       <= 0.1\n",
                "# Vit filter P threshold:       <= 0.2\n",
                "# Fwd filter P threshold:       <= 0.3\n",
                "# sequence search space set to:    10\n",
                "# domain search space set to:      5\n",
                "# random number seed set to:       7\n",
            ],
        ),
        (
            vec!["-T", "42", "--incT", "13"],
            vec![
                "# profile reporting threshold:     score >= 42\n",
                "# profile inclusion threshold:     score >= 13\n",
            ],
        ),
        (
            vec!["--domT", "3", "--incdomT", "2"],
            vec![
                "# domain reporting threshold:      score >= 3\n",
                "# domain inclusion threshold:      score >= 2\n",
            ],
        ),
        (
            vec!["--max"],
            vec!["# Max sensitivity mode:            on [all heuristic filters off]\n"],
        ),
        (
            vec!["--nobias", "--nonull2"],
            vec![
                "# biased composition HMM filter:   off\n",
                "# null2 bias corrections:          off\n",
            ],
        ),
        (
            vec!["--seed", "42"],
            vec!["# random number seed set to:       42\n"],
        ),
        (
            vec!["--seed", "0"],
            vec!["# random number seed:              one-time arbitrary\n"],
        ),
    ] {
        let mut command_args = vec!["scan", "--noali"];
        command_args.extend(args);
        command_args.extend([hmmdb.to_str().unwrap(), "hmmer/tutorial/7LESS_DROME"]);
        let output = Command::new(hmmer()).args(&command_args).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        for line in expected {
            assert!(stdout.contains(line), "{line:?} missing from:\n{stdout}");
        }
        if command_args.windows(2).any(|w| w == ["--seed", "0"]) {
            assert!(!stdout.contains("# random number seed set to:       0\n"));
        }
    }
}

#[test]
fn hmmscan_default_stdout_includes_c_style_alignments() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "scan",
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Domain annotation for each model (and alignments):\n"));
    assert!(stdout.contains(">> fn3  Fibronectin type III domain\n"));
    assert_eq!(stdout.matches("  == domain ").count(), 9);
    assert!(stdout.contains("  Alignments for each domain:\n"));
    assert!(stdout.contains("  == domain 1  score: -1.3 bits;  conditional E-value: 0.17\n"));
    assert!(stdout.contains("  == domain 9  score: 12.8 bits;  conditional E-value: 6.6e-06\n"));
    assert!(stdout.contains("fn3    1"));
    assert!(stdout.contains("7LESS_DROME 1993"));
    assert!(stdout.contains("fn3   69"));
    assert!(stdout.contains("7LESS_DROME 2089"));
}

#[test]
fn nhmmscan_tblout_uses_model_length_and_dfamtblout_has_no_footer_like_c() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let tblout = dir.path().join("hits.tbl");
    let dfamtblout = dir.path().join("hits.dfam.tbl");
    let output = Command::new(hmmer())
        .args([
            "nhmmscan",
            "--tblout",
            tblout.to_str().unwrap(),
            "--dfamtblout",
            dfamtblout.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# nhmmscan :: search DNA sequence(s) against a DNA profile database\n")
    );
    assert!(stdout.contains("# Copyright (C) 2023 Howard Hughes Medical Institute.\n"));
    assert!(stdout.contains(&format!(
        "# hits output in Dfam format:      {}\n",
        dfamtblout.display()
    )));
    assert!(stdout.contains("Scores for complete hit:\n"));
    assert!(stdout.contains("    E-value  score  bias  Model     start    end  Description\n"));
    assert!(stdout.contains("  ------ inclusion threshold ------\n"));
    assert!(stdout.contains("Annotation for each hit  (and alignments):\n"));
    assert!(stdout
        .contains(">> MADE1  MADE1 (MAriner Derived Element 1), a TcMar-Mariner DNA transposon\n"));
    assert!(stdout.contains("    score  bias    Evalue   hmmfrom    hmm to"));
    assert!(stdout.contains(" !   38.8   7.4   9.6e-11"));
    assert!(stdout.contains("  Alignment:\n"));
    assert!(stdout.contains("  score: 38.8 bits\n"));
    assert!(stdout.contains("Internal pipeline statistics summary:\n"));
    assert!(stdout.contains("Residues passing SSV filter:"));
    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert!(tbl.contains(" modlen"));
    assert!(!tbl.contains(" sq len"));
    assert!(tbl.contains("# Program:         nhmmscan\n"));
    assert!(tbl.ends_with("# [ok]\n"));

    let dfam = std::fs::read_to_string(dfamtblout).unwrap();
    assert!(!dfam.contains("# Program:         nhmmscan\n"));
    assert!(!dfam.ends_with("# [ok]\n"));
}

#[test]
fn nhmmer_and_nhmmscan_bgfile_changes_dna_tblout_scores() {
    let dir = tempfile::tempdir().unwrap();
    let bgfile = dir.path().join("biased.bg");
    std::fs::write(&bgfile, "DNA\nA 0.7\nC 0.1\nG 0.1\nT 0.1\n").unwrap();

    let nhmmer_default = dir.path().join("nhmmer-default.tbl");
    let nhmmer_custom = dir.path().join("nhmmer-custom.tbl");
    for (extra, tblout) in [
        (Vec::<&str>::new(), nhmmer_default.as_path()),
        (
            vec!["--bgfile", bgfile.to_str().unwrap()],
            nhmmer_custom.as_path(),
        ),
    ] {
        let mut args = vec!["nhmmer", "--dna", "--noali", "--tblout"];
        args.push(tblout.to_str().unwrap());
        args.extend(extra);
        args.extend([
            "hmmer/testsuite/3box.hmm",
            "hmmer/testsuite/3box-alitest.fa",
        ]);
        let output = Command::new(hmmer()).args(args).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let nhmmer_default_table = std::fs::read_to_string(&nhmmer_default).unwrap();
    let nhmmer_custom_table = std::fs::read_to_string(&nhmmer_custom).unwrap();
    assert_ne!(
        table_data_rows(&nhmmer_default_table),
        table_data_rows(&nhmmer_custom_table)
    );

    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );
    let nhmmscan_default = dir.path().join("nhmmscan-default.tbl");
    let nhmmscan_custom = dir.path().join("nhmmscan-custom.tbl");
    for (extra, tblout) in [
        (Vec::<&str>::new(), nhmmscan_default.as_path()),
        (
            vec!["--bgfile", bgfile.to_str().unwrap()],
            nhmmscan_custom.as_path(),
        ),
    ] {
        let mut args = vec!["nhmmscan", "--noali", "--tblout"];
        args.push(tblout.to_str().unwrap());
        args.extend(extra);
        args.extend([hmmdb.to_str().unwrap(), "hmmer/tutorial/dna_target.fa"]);
        let output = Command::new(hmmer()).args(args).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let nhmmscan_default_table = std::fs::read_to_string(&nhmmscan_default).unwrap();
    let nhmmscan_custom_table = std::fs::read_to_string(&nhmmscan_custom).unwrap();
    assert_ne!(
        table_data_rows(&nhmmscan_default_table),
        table_data_rows(&nhmmscan_custom_table)
    );
}

#[test]
fn nhmmscan_accepts_fasta_qformat_assertion() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "nhmmscan",
            "--qformat",
            "fasta",
            "--noali",
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# input seqfile format asserted:   fasta\n"));
    assert!(stdout.contains("Scores for complete hit:"));
    assert!(stdout.contains("[ok]"));
}

#[test]
fn nhmmscan_accepts_genbank_qformat_assertion() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("ecori.hmm");
    let seq = dir.path().join("query.gb");
    std::fs::copy("hmmer/testsuite/ecori.hmm", &hmmdb).unwrap();
    std::fs::write(
        &seq,
        "LOCUS       query          12 bp    DNA     linear   01-JAN-2000\n\
DEFINITION  synthetic query.\n\
ORIGIN\n\
        1 acgtacgtac gt\n//\n",
    )
    .unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "nhmmscan",
            "--qformat",
            "genbank",
            "--noali",
            hmmdb.to_str().unwrap(),
            seq.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# input seqfile format asserted:   genbank\n"));
    assert!(stdout.contains("Scores for complete hit:"));
    assert!(stdout.contains("[ok]"));
}

#[test]
fn nhmmer_accepts_c_compat_query_matrix_block_and_hidden_options() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("same-length.fa");
    let mxfile = dir.path().join("custom-dna.mx");
    let bgfile = dir.path().join("custom-dna.bg");
    std::fs::write(&query, ">q1\nGAATTC\n>q2\nGAATTC\n").unwrap();
    std::fs::write(
        &mxfile,
        "  A C G T\nA 1 -1 -1 -1\nC -1 1 -1 -1\nG -1 -1 1 -1\nT -1 -1 -1 1\n",
    )
    .unwrap();
    std::fs::write(&bgfile, "DNA\nA 0.7\nC 0.1\nG 0.1\nT 0.1\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qsingle_seqs",
            "--singlemx",
            "--mxfile",
            mxfile.to_str().unwrap(),
            "--block_length",
            "50000",
            "--B1",
            "111",
            "--B2",
            "222",
            "--B3",
            "333",
            "--bgfile",
            bgfile.to_str().unwrap(),
            "--seed_max_depth",
            "16",
            "--seed_sc_thresh",
            "12",
            "--seed_sc_density",
            "0.5",
            "--seed_drop_max_len",
            "5",
            "--seed_drop_lim",
            "0.4",
            "--seed_req_pos",
            "6",
            "--seed_consens_match",
            "9",
            "--seed_ssv_length",
            "101",
            "--domZ",
            "1",
            "--dna",
            "--noali",
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# Use score matrix for 1-seq MSAs:  on\n"));
    assert!(stdout.contains(&format!(
        "# subst score matrix (file):       {}",
        mxfile.display()
    )));
    assert!(stdout.contains("# block length :                   50000\n"));
    assert!(stdout.contains("# query contains individual seqs:  on\n"));
    assert!(stdout.contains("# biased comp SSV window len:      111\n"));
    assert!(stdout.contains("# biased comp Viterbi window len:  222\n"));
    assert!(stdout.contains("# biased comp Forward window len:  333\n"));
    assert!(stdout.contains(&format!(
        "# file with custom bg probs:       {}\n",
        bgfile.display()
    )));
    assert!(stdout.contains("# FM Seed length:                  16\n"));
    assert!(stdout.contains("# FM score threshold (bits):       12\n"));
    assert!(stdout.contains("# FM score density (bits/pos):     0.5\n"));
    assert!(stdout.contains("# FM max neg-growth length:        5\n"));
    assert!(stdout.contains("# FM max run drop:                 0.4\n"));
    assert!(stdout.contains("# FM req positive run length:      6\n"));
    assert!(stdout.contains("# FM consec consensus match req:   9\n"));
    assert!(stdout.contains("# FM len used for Vit window:      101\n"));
    assert!(stdout.contains("Query:       q1  [M=6]"));
    assert!(stdout.contains("Query:       q2  [M=6]"));
}

#[test]
fn hmmsearch_and_phmmer_restrictdb_n_limit_target_count() {
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    let query = dir.path().join("query.fa");
    let seqdb_text =
        ">test1\nACDEFGHIKLMNPQRSTVWY\n>test2\nACDEFGHIKLMNPQRSTVWY\n>test3\nACDEFGHIKLMNPQRSTVWY\n";
    std::fs::write(&seqdb, seqdb_text).unwrap();
    std::fs::write(&query, ">query\nACDEFGHIKLMNPQRSTVWY\n").unwrap();
    let test2_offset = seqdb_text.find(">test2\n").unwrap() as u64;
    let ssi = dir.path().join("override.ssi");
    hmmer_pure_rs::ssi::write_hmm_ssi_records(
        &seqdb,
        &ssi,
        [
            ("test1".to_string(), None, 0),
            (
                "test2".to_string(),
                Some("alias2".to_string()),
                test2_offset,
            ),
            (
                "test3".to_string(),
                None,
                seqdb_text.find(">test3\n").unwrap() as u64,
            ),
        ],
        true,
    )
    .unwrap();
    let seqdb_arg = seqdb.to_str().unwrap();
    let query_arg = query.to_str().unwrap();
    let ssi_arg = ssi.to_str().unwrap();

    let cases: Vec<Vec<String>> = vec![
        vec![
            "search",
            "--cpu",
            "1",
            "--noali",
            "--restrictdb_stkey",
            "alias2",
            "--restrictdb_n",
            "1",
            "--ssifile",
            ssi_arg,
            "hmmer/testsuite/20aa.hmm",
            seqdb_arg,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        vec![
            "phmmer",
            "--cpu",
            "1",
            "--noali",
            "--restrictdb_stkey",
            "alias2",
            "--restrictdb_n",
            "1",
            "--ssifile",
            ssi_arg,
            query_arg,
            seqdb_arg,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    ];

    for args in cases {
        let output = Command::new(hmmer()).args(&args).output().unwrap();
        assert!(
            output.status.success(),
            "{} failed: {}",
            args[0],
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("# Restrict db to start at seq key: alias2\n"));
        assert!(stdout.contains("# Restrict db to # target seqs:    1\n"));
        assert!(stdout.contains(&format!(
            "# Override ssi file to:            {}\n",
            ssi.display()
        )));
        assert!(
            stdout.contains("Target sequences:                          1"),
            "{stdout}"
        );
        assert!(stdout.contains("test2"), "{stdout}");
        assert!(!stdout.contains("test1"), "{stdout}");
    }
}

#[test]
fn nhmmer_restricted_database_uses_ssi_offset_and_reports_header() {
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("dna-targets.fa");
    let seqdb_text = ">t1\nGAATTC\n>t2\nGAATTC\n>t3\nGAATTC\n";
    std::fs::write(&seqdb, seqdb_text).unwrap();
    let ssi = dir.path().join("override.ssi");
    hmmer_pure_rs::ssi::write_hmm_ssi_records(
        &seqdb,
        &ssi,
        [
            ("t1".to_string(), None, 0),
            (
                "t2".to_string(),
                Some("alias2".to_string()),
                seqdb_text.find(">t2\n").unwrap() as u64,
            ),
            (
                "t3".to_string(),
                None,
                seqdb_text.find(">t3\n").unwrap() as u64,
            ),
        ],
        true,
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--dna",
            "--cpu",
            "1",
            "--noali",
            "--restrictdb_stkey",
            "alias2",
            "--restrictdb_n",
            "1",
            "--ssifile",
            ssi.to_str().unwrap(),
            "test_data/mapali/ecori-rebuilt.hmm",
            seqdb.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# Restrict db to start at seq key: alias2\n"));
    assert!(stdout.contains("# Restrict db to # target seqs:    1\n"));
    assert!(stdout.contains(&format!(
        "# Override ssi file to:            {}\n",
        ssi.display()
    )));
    assert!(stdout.contains("Target sequences:                          1"));
}

#[test]
fn search_commands_use_hmmer_ncpu_env_default_without_explicit_cpu() {
    let dir = tempfile::tempdir().unwrap();
    let scan_hmm = dir.path().join("20aa.hmm");
    std::fs::copy("hmmer/testsuite/20aa.hmm", &scan_hmm).unwrap();
    let scan_press = Command::new(hmmer())
        .args(["press", "-f", scan_hmm.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        scan_press.status.success(),
        "{}",
        String::from_utf8_lossy(&scan_press.stderr)
    );

    let nscan_hmm = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &nscan_hmm).unwrap();
    let nscan_press = Command::new(hmmer())
        .args(["press", "-f", nscan_hmm.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        nscan_press.status.success(),
        "{}",
        String::from_utf8_lossy(&nscan_press.stderr)
    );

    let cases: Vec<(Vec<String>, &str)> = vec![
        (
            vec![
                "search".to_string(),
                "--noali".to_string(),
                "hmmer/testsuite/20aa.hmm".to_string(),
                "hmmer/testsuite/20aa-alitest.fa".to_string(),
            ],
            "# number of worker threads:        1\n",
        ),
        (
            vec![
                "phmmer".to_string(),
                "--noali".to_string(),
                "hmmer/testsuite/20aa-alitest.fa".to_string(),
                "hmmer/testsuite/20aa-alitest.fa".to_string(),
            ],
            "# number of worker threads:        1\n",
        ),
        (
            vec![
                "jackhmmer".to_string(),
                "-N".to_string(),
                "1".to_string(),
                "--noali".to_string(),
                "hmmer/testsuite/20aa-alitest.fa".to_string(),
                "hmmer/testsuite/20aa-alitest.fa".to_string(),
            ],
            "# number of worker threads:        1\n",
        ),
        (
            vec![
                "nhmmer".to_string(),
                "--dna".to_string(),
                "--noali".to_string(),
                "hmmer/testsuite/ecori.hmm".to_string(),
                "hmmer/testsuite/ecori.fa".to_string(),
            ],
            "# number of worker threads:        1\n",
        ),
        (
            vec![
                "scan".to_string(),
                "--noali".to_string(),
                scan_hmm.to_string_lossy().into_owned(),
                "hmmer/testsuite/20aa-alitest.fa".to_string(),
            ],
            "# multithread parallelization:     1 workers\n",
        ),
        (
            vec![
                "nhmmscan".to_string(),
                "--noali".to_string(),
                nscan_hmm.to_string_lossy().into_owned(),
                "hmmer/tutorial/dna_target.fa".to_string(),
            ],
            "# multithread parallelization:     1 workers\n",
        ),
    ];

    for (args, expected) in cases {
        let output = Command::new(hmmer())
            .env("HMMER_NCPU", "1")
            .args(&args)
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "{} failed: {}",
            args[0],
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains(expected), "{} stdout:\n{stdout}", args[0]);
    }
}

#[test]
fn nhmmscan_w_beta_recomputes_window_length_evalues() {
    fn first_tblout_evalue(path: &std::path::Path) -> f64 {
        let tbl = std::fs::read_to_string(path).unwrap();
        let line = tbl
            .lines()
            .find(|line| !line.starts_with('#') && !line.trim().is_empty())
            .unwrap_or_else(|| panic!("no tblout hit in:\n{tbl}"));
        line.split_whitespace().nth(12).unwrap().parse().unwrap()
    }

    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let original = hmmer_pure_rs::hmmfile::read_hmm_file_auto(std::path::Path::new(
        "hmmer/tutorial/MADE1.hmm",
    ))
    .unwrap()
    .remove(0);
    let beta = 0.5_f64;
    let expected_maxl = hmmer_pure_rs::builder::max_length_from_beta(&original, beta);
    assert_ne!(expected_maxl, original.max_length);

    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let default_tbl = dir.path().join("default.tbl");
    let default = Command::new(hmmer())
        .args([
            "nhmmscan",
            "--cpu",
            "1",
            "--tblout",
            default_tbl.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );

    let beta_tbl = dir.path().join("beta.tbl");
    let beta_out = Command::new(hmmer())
        .args([
            "nhmmscan",
            "--cpu",
            "1",
            "--w_beta",
            "0.5",
            "--tblout",
            beta_tbl.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(
        beta_out.status.success(),
        "{}",
        String::from_utf8_lossy(&beta_out.stderr)
    );
    let stdout = String::from_utf8(beta_out.stdout).unwrap();
    assert!(stdout.contains("# window length beta value:        0.5\n"));

    let default_evalue = first_tblout_evalue(&default_tbl);
    let beta_evalue = first_tblout_evalue(&beta_tbl);
    let expected_ratio = original.max_length as f64 / expected_maxl as f64;
    let observed_ratio = beta_evalue / default_evalue;
    assert!(
        (observed_ratio - expected_ratio).abs() < expected_ratio * 0.05,
        "observed ratio {observed_ratio} expected {expected_ratio}; default={default_evalue} beta={beta_evalue}"
    );
}

#[test]
fn nhmmscan_watson_and_crick_limit_tblout_strands() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let watson_tbl = dir.path().join("watson.tbl");
    let watson = Command::new(hmmer())
        .args([
            "nhmmscan",
            "--watson",
            "--tblout",
            watson_tbl.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(
        watson.status.success(),
        "{}",
        String::from_utf8_lossy(&watson.stderr)
    );
    let watson_tbl = std::fs::read_to_string(watson_tbl).unwrap();
    assert!(watson_tbl.contains(" strand   E-value"));
    assert!(watson_tbl.contains(" 302390  302466  302387  302466      80    +"));
    assert!(!watson_tbl.contains("      80    -"));

    let crick_tbl = dir.path().join("crick.tbl");
    let crick = Command::new(hmmer())
        .args([
            "nhmmscan",
            "--crick",
            "--tblout",
            crick_tbl.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(
        crick.status.success(),
        "{}",
        String::from_utf8_lossy(&crick.stderr)
    );
    let crick_tbl = std::fs::read_to_string(crick_tbl).unwrap();
    assert!(crick_tbl.contains(" 302466  302390  302466  302387      80    -"));
    assert!(!crick_tbl.contains("      80    +"));
}

#[test]
fn nhmmscan_default_cpu_is_serial_and_unreported() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "nhmmscan",
            "--noali",
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .env_remove("HMMER_NCPU")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.contains("# multithread parallelization:"));
}

#[test]
fn nhmmscan_z_override_is_reported_and_scales_tblout_evalues() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let tblout = dir.path().join("hits.tbl");
    let output = Command::new(hmmer())
        .args([
            "nhmmscan",
            "-Z",
            "123",
            "--tblout",
            tblout.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# sequence search space set to:    123\n"));
    assert!(
        stdout.contains("Query sequence(s):                         1  (660000 residues searched)")
    );

    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert!(tbl.contains("MADE1                DF0000629.2 humanchr1_frag"));
    assert!(tbl.contains("    1.2e-08   38.8   7.4"));
    assert!(tbl.contains("      80    -     1.2e-05   29.2   6.0"));
    assert!(!tbl.contains("      80    +         1.4    6.3   7.0"));
}

#[test]
fn nhmmscan_fractional_z_override_scales_evalues_without_clamping() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let tblout = dir.path().join("half.tbl");
    let output = Command::new(hmmer())
        .args([
            "nhmmscan",
            "-Z",
            "0.5",
            "--tblout",
            tblout.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# sequence search space set to:    0\n"));

    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert!(tbl.contains("    4.8e-11   38.8   7.4"));
    assert!(tbl.contains("      80    -       5e-08   29.2   6.0"));
    assert!(tbl.contains("      80    +         0.7    6.3   7.0"));
    assert!(!tbl.contains("    9.6e-11   38.8   7.4"));
}

#[test]
fn nhmmscan_output_file_acc_noali_textw_cpu_and_seed_work() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    let out = dir.path().join("nhmmscan.out");
    let tblout = dir.path().join("hits.tbl");
    let output = Command::new(hmmer())
        .args([
            "nhmmscan",
            "-o",
            out.to_str().unwrap(),
            "--acc",
            "--noali",
            "--textw",
            "120",
            "--cpu",
            "1",
            "--seed",
            "7",
            "--tblout",
            tblout.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());

    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains(&format!(
        "# output directed to file:         {}\n",
        out.display()
    )));
    assert!(text.contains("# prefer accessions over names:    yes\n"));
    assert!(text.contains("# show alignments in output:       no\n"));
    assert!(text.contains("# max ASCII text line length:      120\n"));
    assert!(text.contains("# multithread parallelization:     1 workers\n"));
    assert!(text.contains("# random number seed set to:       7\n"));
    assert!(!text.contains("  Alignment:\n"));

    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert!(tbl.contains("DF0000629.2"));
    assert!(tbl.contains("# Program:         nhmmscan\n"));
    assert!(tbl.ends_with("# [ok]\n"));
}

#[test]
fn nhmmscan_reports_profile_threshold_options_in_header() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    for (args, expected) in [
        (
            vec!["-E", "1", "--incE", "0.5"],
            vec![
                "# profile reporting threshold:     E-value <= 1\n",
                "# profile inclusion threshold:     E-value <= 0.5\n",
            ],
        ),
        (
            vec!["-T", "42", "--incT", "13"],
            vec![
                "# profile reporting threshold:     score >= 42\n",
                "# profile inclusion threshold:     score >= 13\n",
            ],
        ),
        (
            vec!["--F1", "0.1", "--F2", "0.2", "--F3", "0.3"],
            vec![
                "# MSV filter P threshold:       <= 0.1\n",
                "# Vit filter P threshold:       <= 0.2\n",
                "# Fwd filter P threshold:       <= 0.3\n",
            ],
        ),
        (
            vec!["--max"],
            vec!["# Max sensitivity mode:            on [all heuristic filters off]\n"],
        ),
        (
            vec!["--nobias", "--nonull2"],
            vec![
                "# biased composition HMM filter:   off\n",
                "# null2 bias corrections:          off\n",
            ],
        ),
        (
            vec!["--watson"],
            vec!["# search only top strand:          on\n"],
        ),
        (
            vec!["--crick"],
            vec!["# search only bottom strand:       on\n"],
        ),
        (
            vec!["--seed", "42"],
            vec!["# random number seed set to:       42\n"],
        ),
        (
            vec!["--seed", "0"],
            vec!["# random number seed:              one-time arbitrary\n"],
        ),
    ] {
        let mut command_args = vec!["nhmmscan", "--noali"];
        command_args.extend(args);
        command_args.extend([hmmdb.to_str().unwrap(), "hmmer/tutorial/dna_target.fa"]);
        let output = Command::new(hmmer()).args(&command_args).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        for line in expected {
            assert!(stdout.contains(line), "{line:?} missing from:\n{stdout}");
        }
        if command_args.windows(2).any(|w| w == ["--seed", "0"]) {
            assert!(!stdout.contains("# random number seed set to:       0\n"));
        }
    }
}

#[test]
fn scan_commands_reject_unpressed_hmm_databases_like_c() {
    let output = Command::new(hmmer())
        .args([
            "scan",
            "--noali",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("use hmmpress first"), "{stderr}");

    let output = Command::new(hmmer())
        .args([
            "nhmmscan",
            "hmmer/tutorial/MADE1.hmm",
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("use hmmpress first"), "{stderr}");
}

#[test]
fn hmmsearch_model_specific_cutoffs_accept_annotated_models_and_reject_missing_cutoffs() {
    for cutoff in ["--cut_ga", "--cut_tc", "--cut_nc"] {
        let output = Command::new(hmmer())
            .args([
                "search",
                cutoff,
                "--noali",
                "hmmer/tutorial/fn3.hmm",
                "hmmer/tutorial/7LESS_DROME",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{} failed: {}",
            cutoff,
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("Query:       fn3  [M=86]"));
        assert!(stdout.contains("7LESS_DROME"));
    }

    let output = Command::new(hmmer())
        .args([
            "search",
            "--cut_ga",
            "--noali",
            "hmmer/testsuite/20aa.hmm",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("GA cutoff not set in model for model test"));
}

#[test]
fn phmmer_hidden_model_specific_cutoffs_do_not_emit_stdout_header_lines() {
    for cutoff in ["--cut_ga", "--cut_tc", "--cut_nc"] {
        let output = Command::new(hmmer())
            .args([
                "phmmer",
                cutoff,
                "--noali",
                "hmmer/testsuite/20aa-alitest.fa",
                "hmmer/testsuite/20aa-alitest.fa",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{} failed: {}",
            cutoff,
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(!stdout.contains("# model-specific thresholding:"));
    }
}

#[test]
fn hmmscan_model_specific_cutoffs_accept_annotated_models_and_reject_missing_cutoffs() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    for cutoff in ["--cut_ga", "--cut_tc", "--cut_nc"] {
        let output = Command::new(hmmer())
            .args([
                "scan",
                cutoff,
                "--noali",
                hmmdb.to_str().unwrap(),
                "hmmer/tutorial/7LESS_DROME",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{} failed: {}",
            cutoff,
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("Query:       7LESS_DROME  [L=2554]"));
        assert!(stdout.contains("fn3"));
    }

    let missing_dir = tempfile::tempdir().unwrap();
    let missing_hmmdb = missing_dir.path().join("20aa.hmm");
    std::fs::copy("hmmer/testsuite/20aa.hmm", &missing_hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", missing_hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    for (cutoff, message) in [
        ("--cut_ga", "GA cutoff not set in model for model test"),
        ("--cut_tc", "TC cutoff not set in model for model test"),
        ("--cut_nc", "NC cutoff not set in model for model test"),
    ] {
        let output = Command::new(hmmer())
            .args([
                "scan",
                cutoff,
                "--noali",
                missing_hmmdb.to_str().unwrap(),
                "hmmer/testsuite/20aa-alitest.fa",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{cutoff} unexpectedly succeeded");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains(message), "{stderr}");
    }
}

#[test]
fn nhmmscan_model_specific_cutoffs_reject_missing_cutoffs() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("made1.hmm");
    std::fs::copy("hmmer/tutorial/MADE1.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );

    for (cutoff, message) in [
        ("--cut_ga", "GA cutoff not set in model for model MADE1"),
        ("--cut_tc", "TC cutoff not set in model for model MADE1"),
        ("--cut_nc", "NC cutoff not set in model for model MADE1"),
    ] {
        let output = Command::new(hmmer())
            .args([
                "nhmmscan",
                cutoff,
                hmmdb.to_str().unwrap(),
                "hmmer/tutorial/dna_target.fa",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{cutoff} unexpectedly succeeded");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains(message), "{stderr}");
    }
}

#[test]
fn nhmmer_output_file_acc_noali_textw_cpu_seed_and_tblout_work() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("nhmmer.out");
    let tblout = dir.path().join("hits.tbl");
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "-o",
            out.to_str().unwrap(),
            "--acc",
            "--noali",
            "--textw",
            "120",
            "--cpu",
            "1",
            "--seed",
            "7",
            "--tblout",
            tblout.to_str().unwrap(),
            "hmmer/tutorial/MADE1.hmm",
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());

    let text = std::fs::read_to_string(out).unwrap();
    assert!(text.contains("# hits tabular output:             "));
    assert!(text.contains("# number of worker threads:        1\n"));
    // C prints "Annotation for each hit %s:\n" with %s="" under --noali, leaving
    // the literal space after "hit" (verified against bundled C nhmmer).
    assert!(text.contains("Annotation for each hit :\n"));
    assert!(!text.contains("  Alignment:\n"));
    assert!(text.contains("DF0000629.2"));

    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert!(tbl.contains("# Program:         nhmmer\n"));
    assert!(tbl.contains("humanchr1_frag"));
    assert!(tbl.ends_with("# [ok]\n"));
}

#[test]
fn nhmmer_reports_used_search_options_in_header() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--dna",
            "--max",
            "--cpu",
            "1",
            "--noali",
            "--seed",
            "0",
            "--watson",
            "-Z",
            "2",
            "test_data/mapali/ecori-rebuilt.hmm",
            "test_data/mapali/ecori-query.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "# show alignments in output:       no\n",
        "# Max sensitivity mode:            on [all heuristic filters off]\n",
        "# input query is asserted as:      DNA\n",
        "# search only top strand:          on\n",
        "# database size is set to:         2.0 Mb\n",
        "# random number seed:              one-time arbitrary\n",
    ] {
        assert!(
            stdout.contains(expected),
            "{expected:?} missing from:\n{stdout}"
        );
    }
}

#[test]
fn nhmmer_writes_alignment_hmmout_and_aliscoresout_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let ali = dir.path().join("hits.sto");
    let hmmout = dir.path().join("query.hmm");
    let aliscores = dir.path().join("ali.scores");
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--cpu",
            "1",
            "-A",
            ali.to_str().unwrap(),
            "--hmmout",
            hmmout.to_str().unwrap(),
            "--aliscoresout",
            aliscores.to_str().unwrap(),
            "hmmer/tutorial/MADE1.hmm",
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(&format!(
        "# MSA of all hits saved to file:   {}",
        ali.display()
    )));
    assert!(stdout.contains(&format!(
        "# alignment scores output:         {}",
        aliscores.display()
    )));
    assert!(stdout.contains(&format!(
        "# hmm output:                      {}",
        hmmout.display()
    )));
    assert!(stdout.contains("# Alignment of "));

    let msa = std::fs::read_to_string(ali).unwrap();
    assert!(msa.starts_with("# STOCKHOLM 1.0\n"), "{msa}");
    assert!(msa.contains("#=GF ID MADE1"), "{msa}");
    assert!(msa.contains("#=GF AC DF0000629.2"), "{msa}");
    assert!(msa.contains("humanchr1_frag/"), "{msa}");
    assert!(msa.trim_end().ends_with("//"), "{msa}");

    let hmm = std::fs::read_to_string(hmmout).unwrap();
    assert!(hmm.starts_with("HMMER3/f "), "{hmm}");
    assert!(hmm.contains("NAME  MADE1\n"), "{hmm}");
    assert!(hmm.contains("ACC   DF0000629.2\n"), "{hmm}");
    assert!(hmm.trim_end().ends_with("//"), "{hmm}");

    let scores = std::fs::read_to_string(aliscores).unwrap();
    assert!(scores.contains("MADE1 humanchr1_frag "), "{scores}");
    assert_eq!(scores.lines().count(), 5, "{scores}");
    assert!(
        scores.starts_with("MADE1 humanchr1_frag 302390 302466 : 1.333 1.422 1.505 -1.090"),
        "{scores}"
    );
    assert!(scores.contains(" > > > -8.692 1.268 -7.397"), "{scores}");
    assert!(!scores.contains(" : 0.000"), "{scores}");
}

#[test]
fn nhmmer_w_beta_recomputes_hmmout_max_length() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("nhmmer.out");
    let hmmout = dir.path().join("query.hmm");
    let beta = 0.5_f64;
    let input_hmm = "hmmer/tutorial/MADE1.hmm";
    let original = hmmer_pure_rs::hmmfile::read_hmm_file_auto(std::path::Path::new(input_hmm))
        .unwrap()
        .remove(0);
    let expected_maxl = hmmer_pure_rs::builder::max_length_from_beta(&original, beta);
    assert_ne!(expected_maxl, original.max_length);

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--cpu",
            "1",
            "-o",
            out.to_str().unwrap(),
            "--hmmout",
            hmmout.to_str().unwrap(),
            "--w_beta",
            "0.5",
            input_hmm,
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let text = std::fs::read_to_string(out).unwrap();
    assert!(text.contains("# window length beta value:        0.5\n"));

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(
        saved.contains(&format!("MAXL  {expected_maxl}\n")),
        "{saved}"
    );
}

#[test]
fn nhmmer_watson_and_crick_limit_tblout_strands() {
    let dir = tempfile::tempdir().unwrap();

    let watson_tbl = dir.path().join("watson.tbl");
    let watson = Command::new(hmmer())
        .args([
            "nhmmer",
            "--watson",
            "--tblout",
            watson_tbl.to_str().unwrap(),
            "hmmer/tutorial/MADE1.hmm",
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(
        watson.status.success(),
        "{}",
        String::from_utf8_lossy(&watson.stderr)
    );
    let watson_tbl = std::fs::read_to_string(watson_tbl).unwrap();
    assert!(watson_tbl.contains(" strand   E-value"));
    assert!(watson_tbl.contains(" 302390  302466  302387  302466  330000    +"));
    assert!(!watson_tbl.contains(" 330000    -"));

    let crick_tbl = dir.path().join("crick.tbl");
    let crick = Command::new(hmmer())
        .args([
            "nhmmer",
            "--crick",
            "--tblout",
            crick_tbl.to_str().unwrap(),
            "hmmer/tutorial/MADE1.hmm",
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(
        crick.status.success(),
        "{}",
        String::from_utf8_lossy(&crick.stderr)
    );
    let crick_tbl = std::fs::read_to_string(crick_tbl).unwrap();
    assert!(crick_tbl.contains(" 302466  302390  302466  302387  330000    -"));
    assert!(!crick_tbl.contains(" 330000    +"));
}

#[test]
fn hmmpress_writes_sidecars_and_honors_force() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();

    let first = Command::new(hmmer())
        .args(["press", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    for suffix in [".h3m", ".h3f", ".h3p", ".h3i"] {
        assert!(
            std::path::PathBuf::from(format!("{}{}", hmmdb.to_string_lossy(), suffix)).exists()
        );
    }

    let second = Command::new(hmmer())
        .args(["press", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!second.status.success());
    let stderr = String::from_utf8(second.stderr).unwrap();
    assert!(stderr.contains("already exists"), "{stderr}");

    let forced = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        forced.status.success(),
        "{}",
        String::from_utf8_lossy(&forced.stderr)
    );
}

#[cfg(unix)]
#[test]
fn hmmpress_preserves_non_utf8_sidecar_path_bytes() {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};

    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir
        .path()
        .join(std::ffi::OsString::from_vec(b"fn3-\xff.hmm".to_vec()));
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmmdb).unwrap();

    let output = Command::new(hmmer())
        .arg("press")
        .arg(&hmmdb)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    for suffix in [".h3m", ".h3f", ".h3p", ".h3i"] {
        let sidecar = hmmer_pure_rs::ssi::path_with_appended_suffix(&hmmdb, suffix);
        assert!(sidecar.exists(), "missing sidecar {}", sidecar.display());
        assert!(sidecar.as_os_str().as_bytes().contains(&0xff));
    }
}

#[test]
fn hmmbuild_summary_uses_c_table_shape() {
    let outdir = std::path::Path::new(project_root()).join(".tmp/cli-output-parity");
    std::fs::create_dir_all(&outdir).unwrap();
    let hmm_out = outdir.join(format!("hmmbuild-{}.hmm", std::process::id()));

    let output = Command::new(hmmer())
        .args([
            "build",
            hmm_out.to_str().unwrap(),
            "hmmer/testsuite/20aa.sto",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# Copyright (C) 2023 Howard Hughes Medical Institute.\n"));
    assert!(stdout
        .contains("# idx name                  nseq  alen  mlen eff_nseq re/pos description\n"));
    assert!(stdout.contains("1     test                    10    20    20     1.96  2.634 "));
    assert!(!stdout.contains("# [ok]"));

    let _ = std::fs::remove_file(hmm_out);
}

#[test]
fn hmmconvert_binary_output_is_readable_hmmer3f() {
    let output = Command::new(hmmer())
        .args(["convert", "-b", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.stdout.starts_with(b"HMMER3/"));

    let mut cursor = std::io::Cursor::new(output.stdout);
    let hmm = hmmfile_binary::read_binary_hmm(&mut cursor)
        .unwrap()
        .expect("binary hmm record");
    assert_eq!(hmm.name, "fn3");
    assert_eq!(hmm.m, 86);
    assert!(hmmfile_binary::read_binary_hmm(&mut cursor)
        .unwrap()
        .is_none());
}

#[test]
fn hmmconvert_legacy_binary_outfmt_magic_matches_c_and_is_readable() {
    for fmt in ["3/a", "3/b", "3/c", "3/d", "3/e"] {
        let rust = Command::new(hmmer())
            .args(["convert", "-b", "--outfmt", fmt, "hmmer/tutorial/fn3.hmm"])
            .output()
            .unwrap();
        assert!(
            rust.status.success(),
            "{fmt}: {}",
            String::from_utf8_lossy(&rust.stderr)
        );
        assert!(rust.stdout.len() > 4, "{fmt}: missing binary payload");

        let c = Command::new(c_hmmconvert())
            .args(["-b", "--outfmt", fmt, "hmmer/tutorial/fn3.hmm"])
            .output()
            .unwrap();
        assert!(
            c.status.success(),
            "{fmt}: {}",
            String::from_utf8_lossy(&c.stderr)
        );
        assert_eq!(&rust.stdout[..4], &c.stdout[..4], "{fmt}: magic mismatch");

        let mut cursor = std::io::Cursor::new(&rust.stdout);
        let hmm = hmmfile_binary::read_binary_hmm(&mut cursor)
            .unwrap()
            .expect("binary hmm record");
        assert_eq!(hmm.name, "fn3");
        assert_eq!(hmm.m, 86);
        assert!(hmmfile_binary::read_binary_hmm(&mut cursor)
            .unwrap()
            .is_none());
    }
}

#[test]
fn hmmconvert_legacy_binary_3e_output_is_readable_by_c() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fn3-3e.h3m");
    let rust = Command::new(hmmer())
        .args(["convert", "-b", "--outfmt", "3/e", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    assert!(
        rust.status.success(),
        "{}",
        String::from_utf8_lossy(&rust.stderr)
    );
    std::fs::write(&path, &rust.stdout).unwrap();

    let c = Command::new(c_hmmconvert())
        .arg(path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
    let stdout = String::from_utf8(c.stdout).unwrap();
    assert!(stdout.starts_with("HMMER3/f"), "{stdout}");
    assert!(stdout.contains("NAME  fn3\n"), "{stdout}");
}

#[test]
fn hmmconvert_legacy_ascii_outfmt_headers_and_fields() {
    let mut outputs = std::collections::HashMap::new();
    for (fmt, header) in [
        ("3/a", "HMMER3/a"),
        ("3/b", "HMMER3/b"),
        ("3/c", "HMMER3/c"),
        ("3/d", "HMMER3/d"),
        ("3/e", "HMMER3/e"),
        ("3/f", "HMMER3/f"),
    ] {
        let output = Command::new(hmmer())
            .args(["convert", "--outfmt", fmt, "hmmer/tutorial/fn3.hmm"])
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.starts_with(header), "{fmt}: {stdout}");
        assert!(stdout.contains("NAME  fn3\n"));
        outputs.insert(fmt, stdout);
    }

    let text_3b = outputs.get("3/b").unwrap();
    assert!(!text_3b.contains("\nMAXL  "));
    assert!(!text_3b.contains("\nCONS  "));
    assert!(!text_3b.contains("\nMM    "));

    let text_3e = outputs.get("3/e").unwrap();
    assert!(text_3e.contains("\nCONS  "));
    assert!(!text_3e.contains("\nMM    "));

    let text_3f = outputs.get("3/f").unwrap();
    assert!(text_3f.contains("\nMM    "));
}

#[test]
fn hmmconvert_explicit_ascii_modes_match_default_output() {
    let default = Command::new(hmmer())
        .args(["convert", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    let ascii = Command::new(hmmer())
        .args(["convert", "-a", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    let outfmt = Command::new(hmmer())
        .args(["convert", "--outfmt", "3/f", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();

    for output in [&default, &ascii, &outfmt] {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.starts_with(b"HMMER3/f"));
        assert!(String::from_utf8_lossy(&output.stdout).contains("NAME  fn3\n"));
    }
    assert_eq!(default.stdout, ascii.stdout);
    assert_eq!(default.stdout, outfmt.stdout);
}

#[test]
fn binary_hmm_input_is_detected_by_magic_without_h3m_extension() {
    let dir = tempfile::tempdir().unwrap();
    let binary_path = dir.path().join("fn3.binary");
    let binary = Command::new(hmmer())
        .args(["convert", "-b", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    assert!(
        binary.status.success(),
        "{}",
        String::from_utf8_lossy(&binary.stderr)
    );
    std::fs::write(&binary_path, &binary.stdout).unwrap();

    let stat = Command::new(hmmer())
        .args(["stat", binary_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        stat.status.success(),
        "{}",
        String::from_utf8_lossy(&stat.stderr)
    );
    assert!(String::from_utf8(stat.stdout).unwrap().contains("fn3"));

    let convert = Command::new(hmmer())
        .args(["convert", binary_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        convert.status.success(),
        "{}",
        String::from_utf8_lossy(&convert.stderr)
    );
    assert!(convert.stdout.starts_with(b"HMMER3/f"));

    let convert_stdin = run_with_stdin(&["convert", "-"], &binary.stdout);
    assert!(
        convert_stdin.status.success(),
        "{}",
        String::from_utf8_lossy(&convert_stdin.stderr)
    );
    assert!(convert_stdin.stdout.starts_with(b"HMMER3/f"));

    let fetch = Command::new(hmmer())
        .args(["fetch", binary_path.to_str().unwrap(), "PF00041.13"])
        .output()
        .unwrap();
    assert!(
        fetch.status.success(),
        "{}",
        String::from_utf8_lossy(&fetch.stderr)
    );
    let stdout = String::from_utf8(fetch.stdout).unwrap();
    assert!(stdout.contains("NAME  fn3"));
    assert!(stdout.contains("ACC   PF00041.13"));

    let search = Command::new(hmmer())
        .args([
            "search",
            "--noali",
            binary_path.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        search.status.success(),
        "{}",
        String::from_utf8_lossy(&search.stderr)
    );
    assert!(String::from_utf8(search.stdout)
        .unwrap()
        .contains("Scores for complete sequences"));

    let align = Command::new(hmmer())
        .args([
            "align",
            binary_path.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        align.status.success(),
        "{}",
        String::from_utf8_lossy(&align.stderr)
    );
    assert!(String::from_utf8(align.stdout)
        .unwrap()
        .starts_with("# STOCKHOLM 1.0\n"));

    let press = Command::new(hmmer())
        .args(["press", "-f", binary_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );
    for suffix in [".h3m", ".h3f", ".h3p", ".h3i"] {
        assert!(
            std::path::PathBuf::from(format!("{}{}", binary_path.to_string_lossy(), suffix))
                .exists()
        );
    }
}

#[test]
fn hmmsearch_accepts_fasta_tformat_assertion() {
    let output = Command::new(hmmer())
        .args([
            "search",
            "--tformat",
            "fasta",
            "--noali",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/testsuite/proteins.faa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# target sequence database:        hmmer/testsuite/proteins.faa\n"));
    assert!(stdout.contains("# targ <seqfile> format asserted:  fasta\n"));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn hmmsearch_preserves_tformat_header_spelling() {
    let output = Command::new(hmmer())
        .args([
            "search",
            "--tformat",
            "FASTA",
            "--noali",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/testsuite/proteins.faa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# target sequence database:        hmmer/testsuite/proteins.faa\n"));
    assert!(stdout.contains("# targ <seqfile> format asserted:  FASTA\n"));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn hmmsearch_accepts_uniprot_tformat_assertion() {
    let output = Command::new(hmmer())
        .args([
            "search",
            "--tformat",
            "uniprot",
            "--noali",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/easel/formats/uniprot",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# targ <seqfile> format asserted:  uniprot\n"));
    assert!(stdout.contains("Scores for complete sequences"));
}

#[test]
fn nhmmer_accepts_fasta_tformat_assertion() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--tformat",
            "fasta",
            "--noali",
            "hmmer/tutorial/MADE1.hmm",
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# target sequence database:        hmmer/tutorial/dna_target.fa\n"));
    assert!(stdout.contains("# target format asserted:          fasta\n"));
    assert!(stdout.contains("Scores for complete hits"));
}

#[test]
fn nhmmer_accepts_makehmmerdb_fmindex_tformat_assertion() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("dna_target.hmmerdb");
    let make_db = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "hmmer/tutorial/dna_target.fa",
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        make_db.status.success(),
        "{}",
        String::from_utf8_lossy(&make_db.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--cpu",
            "1",
            "--noali",
            "--tformat",
            "fmindex",
            "hmmer/tutorial/MADE1.hmm",
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# target format asserted:          fmindex\n"));
    assert!(stdout.contains("Target sequences:                          1"));
    assert!(stdout.contains("humanchr1_frag"), "{stdout}");
}

#[test]
fn nhmmer_accepts_native_makehmmerdb_cstream_tformat_assertion() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("dna_target.fm");
    let make_db = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--cstream",
            "hmmer/tutorial/dna_target.fa",
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        make_db.status.success(),
        "{}",
        String::from_utf8_lossy(&make_db.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--cpu",
            "1",
            "--noali",
            "--tformat",
            "fmindex",
            "hmmer/tutorial/MADE1.hmm",
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# target format asserted:          fmindex\n"));
    assert!(stdout.contains("Target sequences:                          1"));
    assert!(stdout.contains("humanchr1_frag"), "{stdout}");
}

#[test]
fn nhmmer_autodetects_native_makehmmerdb_cstream_targets() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("dna_target.fm");
    let make_db = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--cstream",
            "hmmer/tutorial/dna_target.fa",
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        make_db.status.success(),
        "{}",
        String::from_utf8_lossy(&make_db.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--cpu",
            "1",
            "--noali",
            "hmmer/tutorial/MADE1.hmm",
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.contains("# target format asserted:          fmindex\n"));
    assert!(stdout.contains("Target sequences:                          1"));
    assert!(stdout.contains("humanchr1_frag"), "{stdout}");
}

#[test]
fn nhmmer_autodetects_makehmmerdb_fmindex_targets() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("dna_target.hmmerdb");
    let make_db = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "hmmer/tutorial/dna_target.fa",
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        make_db.status.success(),
        "{}",
        String::from_utf8_lossy(&make_db.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--cpu",
            "1",
            "--noali",
            "hmmer/tutorial/MADE1.hmm",
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.contains("# target format asserted:          fmindex\n"));
    assert!(stdout.contains("Target sequences:                          1"));
    assert!(stdout.contains("humanchr1_frag"), "{stdout}");
}

#[test]
fn nhmmer_dfamtblout_uses_c_header_and_query_accession() {
    let dir = tempfile::tempdir().unwrap();
    let tbl = dir.path().join("hits.tbl");
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--noali",
            "--dfamtblout",
            tbl.to_str().unwrap(),
            "hmmer/tutorial/MADE1.hmm",
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(tbl).unwrap();
    assert!(text.contains("sq-len"), "{text}");
    assert!(!text.contains("modlen"), "{text}");
    assert!(text.contains("DF0000629.2"), "{text}");
}

#[test]
fn nhmmer_tblout_footer_uses_normalized_option_settings() {
    let dir = tempfile::tempdir().unwrap();
    let tbl = dir.path().join("hits.tbl");
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--dna",
            "--tblout",
            tbl.to_str().unwrap(),
            "hmmer/tutorial/MADE1.hmm",
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(tbl).unwrap();
    assert!(
        text.contains("# Option settings: hmmer nhmmer --dna --tblout "),
        "{text}"
    );
    assert!(!text.contains("# Option settings: target/"), "{text}");
}

#[test]
fn nhmmer_reads_binary_hmm_by_magic_without_h3m_extension() {
    let dir = tempfile::tempdir().unwrap();
    let binary = dir.path().join("ecori_binary_noext");
    let converted = Command::new(hmmer())
        .args(["convert", "-b", "hmmer/testsuite/ecori.hmm"])
        .output()
        .unwrap();
    assert!(
        converted.status.success(),
        "{}",
        String::from_utf8_lossy(&converted.stderr)
    );
    std::fs::write(&binary, converted.stdout).unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--dna",
            "--noali",
            binary.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn nhmmer_builds_query_hmm_from_stockholm_msa() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.sto");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(
        &query,
        "# STOCKHOLM 1.0\n#=GF ID ecori_msa\nq1 GAATTC\nq2 GAATTC\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "stockholm",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Query:       ecori_msa  [M=6]"), "{stdout}");

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  ecori_msa\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_pfam_msa_alias() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.pfam");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(
        &query,
        "# STOCKHOLM 1.0\n#=GF ID ecori_pfam\nq1 GAATTC\nq2 GAATTC\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "pfam",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           pfam\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Query:       ecori_pfam  [M=6]"),
        "{stdout}"
    );

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  ecori_pfam\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_aligned_fasta_msa() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.afa");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(
        &query,
        ">ecori_a first row\nGAATTC\n>ecori_b second row\nGA-TTC\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "afa",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           afa\n"),
        "{stdout}"
    );
    assert!(stdout.contains("Query:       query  [M=6]"), "{stdout}");

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  query\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_a2m_msa_alias() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.a2m");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(&query, ">ecori_a\nGAATTC\n>ecori_b\nGAaATTC\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "a2m",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           a2m\n"),
        "{stdout}"
    );
    assert!(stdout.contains("Query:       query  [M=6]"), "{stdout}");

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  query\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
    assert!(saved.contains("LENG  6\n"), "{saved}");
    assert!(saved.contains("RF    yes\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_psiblast_msa_alias() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.psi");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(&query, "ecori_a  GAATTC\necori_b  GA-TTC\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "psiblast",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           psiblast\n"),
        "{stdout}"
    );
    assert!(stdout.contains("Query:       query  [M=6]"), "{stdout}");

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  query\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_clustallike_msa_alias() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.aln");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(
        &query,
        "CLUSTAL W multiple sequence alignment\n\necori_a  GAA\necori_b  GA-\n         ** \n\necori_a  TTC\necori_b  TTC\n         ***\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "clustallike",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           clustallike\n"),
        "{stdout}"
    );
    assert!(stdout.contains("Query:       query  [M=6]"), "{stdout}");

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  query\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_selex_msa_alias() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.slx");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(&query, "ecori_a  GAA TTC\necori_b  GA- TTC\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "selex",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           selex\n"),
        "{stdout}"
    );
    assert!(stdout.contains("Query:       query  [M=6]"), "{stdout}");

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  query\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_phylip_msa_alias() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.phy");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(
        &query,
        "2 6\necori_a GAA\necori_b GA-\necori_a TTC\necori_b TTC\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "phylip",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           phylip\n"),
        "{stdout}"
    );
    assert!(stdout.contains("Query:       query  [M=6]"), "{stdout}");

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  query\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_phylips_msa_alias() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.phys");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(&query, "2 6\necori_a GAA\nTTC\necori_b GA-\nTTC\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "phylips",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           phylips\n"),
        "{stdout}"
    );
    assert!(stdout.contains("Query:       query  [M=6]"), "{stdout}");

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  query\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_uniprot_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.dat");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(
        &query,
        "ID   ECORI_UNIPROT\nAC   ECO123;\nDE   EcoRI site query\nSQ   SEQUENCE 6 BP;\n     gaa ttc\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "uniprot",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           uniprot\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Query:       ECORI_UNIPROT  [M=6]"),
        "{stdout}"
    );

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  ECORI_UNIPROT\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_genbank_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.gb");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(
        &query,
        "LOCUS       ECORI_GENBANK 6 bp DNA\nDEFINITION  EcoRI site query\nACCESSION   ECO456\nORIGIN\n        1 gaattc\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "genbank",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           genbank\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Query:       ECORI_GENBANK  [M=6]"),
        "{stdout}"
    );

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  ECORI_GENBANK\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_embl_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.embl");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(
        &query,
        "ID   ECORI_EMBL; SV 1; linear; genomic DNA; STD; UNC; 6 BP.\nAC   ECO789;\nDE   EcoRI site query\nSQ   Sequence 6 BP;\n     gaa ttc\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "embl",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           embl\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Query:       ECORI_EMBL;  [M=6]"),
        "{stdout}"
    );

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  ECORI_EMBL;\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_ddbj_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.ddbj");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(
        &query,
        "LOCUS       ECORI_DDBJ 6 bp DNA\nDEFINITION  EcoRI site query\nACCESSION   ECO987\nORIGIN\n        1 gaattc\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "ddbj",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           ddbj\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Query:       ECORI_DDBJ  [M=6]"),
        "{stdout}"
    );

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  ECORI_DDBJ\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
}

#[test]
fn nhmmer_builds_query_hmm_from_fasta_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.fa");
    let hmmout = dir.path().join("query.hmm");
    std::fs::write(&query, ">ecori_seq\nGAATTC\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "fasta",
            "--dna",
            "--noali",
            "--hmmout",
            hmmout.to_str().unwrap(),
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# query format asserted:           fasta\n"),
        "{stdout}"
    );
    assert!(stdout.contains("Query:       ecori_seq  [M=6]"), "{stdout}");

    let saved = std::fs::read_to_string(hmmout).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("NAME  ecori_seq\n"), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("CONS  yes\n"), "{saved}");
    assert!(saved.contains("MAXL  "), "{saved}");
}

#[test]
fn nhmmer_autodetects_single_fasta_sequence_query() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("query.fa");
    std::fs::write(&query, ">ecori_auto\nGAATTC\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--dna",
            "--noali",
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Query:       ecori_auto  [M=6]"),
        "{stdout}"
    );
    assert!(stdout.contains("Scores for complete hits"), "{stdout}");
}

#[test]
fn makehmmerdb_accepts_c_style_output_positional() {
    let dir = tempfile::tempdir().unwrap();
    let seq = dir.path().join("seq.fa");
    let db = dir.path().join("seq.hmmerdb");
    std::fs::write(&seq, ">s1\nACGTACGT\n>s2\nTTTT\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--dna",
            seq.to_str().unwrap(),
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(db).unwrap();
    assert!(!bytes.starts_with(b"HMMERDB\0"));
    let stream = parse_makehmmerdb_native_c_stream(&bytes);
    assert_eq!(stream.version, 1);
    assert_eq!(stream.meta.block_count, 1);
    assert_eq!(stream.records.len(), 2);
}

#[test]
fn makehmmerdb_cstream_keeps_native_c_stream_top_level() {
    let dir = tempfile::tempdir().unwrap();
    let seq = dir.path().join("seq.fa");
    let db = dir.path().join("seq.fm");
    std::fs::write(&seq, ">s1 first description\nACNNRYT\n>s2\nTTTT\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--dna",
            "--cstream",
            "--fwd_only",
            seq.to_str().unwrap(),
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(db).unwrap();
    assert!(!bytes.starts_with(b"HMMERDB\0"));
    assert!(!bytes
        .windows(b"HMMERDB_C_STREAM\0".len())
        .any(|window| { window == b"HMMERDB_C_STREAM\0" }));

    let stream = parse_makehmmerdb_native_c_stream(&bytes);
    assert_eq!(stream.version, 1);
    assert!(stream.meta.fwd_only);
    assert_eq!(stream.meta.block_count, 1);
    assert_eq!(stream.meta.char_count, 11);
    assert_eq!(stream.meta.sequences[0].name, "s1");
    assert_eq!(stream.meta.sequences[0].desc, "first description");
    assert_eq!(stream.meta.ambiguities, vec![(2, 5)]);
    assert_eq!(stream.records.len(), 1);
    assert!(stream.records[0].has_text_and_sa);
    assert_eq!(stream.records[0].seq_cnt, 2);
    assert_eq!(stream.records[0].ambig_cnt, 1);
}

#[test]
fn makehmmerdb_container_preserves_metadata_and_ambiguity_ranges() {
    let dir = tempfile::tempdir().unwrap();
    let seq = dir.path().join("seq.fa");
    let db = dir.path().join("seq.hmmerdb");
    std::fs::write(
        &seq,
        ">s1 first description\nACNNRYT\n>s2 second description\nTTTT\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--dna",
            "--container",
            seq.to_str().unwrap(),
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(db).unwrap();
    let meta = parse_makehmmerdb_metadata(&bytes);
    let c_meta = parse_makehmmerdb_c_metadata(&bytes);
    let c_stream = parse_makehmmerdb_c_stream(&bytes);
    assert_eq!(meta.seq_count, 2);
    assert_eq!(meta.block_count, 1);
    assert_eq!(meta.ambig_count, 1);
    assert_eq!(meta.sequences[0].name, "s1");
    assert_eq!(meta.sequences[0].desc, "first description");
    assert_eq!(meta.sequences[0].fm_start, 0);
    assert_eq!(meta.sequences[0].length, 7);
    assert_eq!(meta.sequences[1].name, "s2");
    assert_eq!(meta.sequences[1].desc, "second description");
    assert_eq!(meta.ambiguities, vec![(2, 5)]);

    assert!(!c_meta.fwd_only);
    assert_eq!(c_meta.alph_type, 0);
    assert_eq!(c_meta.alph_size, 4);
    assert_eq!(c_meta.char_bits, 2);
    assert_eq!(c_meta.freq_sa, 8);
    assert_eq!(c_meta.freq_cnt_sb, 65_536);
    assert_eq!(c_meta.freq_cnt_b, 256);
    assert_eq!(c_meta.block_count, 1);
    assert_eq!(c_meta.char_count, 11);
    assert_eq!(c_meta.sequences[0].target_id, 0);
    assert_eq!(c_meta.sequences[0].target_start, 1);
    assert_eq!(c_meta.sequences[0].fm_start, 0);
    assert_eq!(c_meta.sequences[0].length, 7);
    assert_eq!(c_meta.sequences[0].name, "s1");
    assert_eq!(c_meta.sequences[0].acc, "");
    // C makehmmerdb sets source = name for the windowed (DNA/RNA) read path
    // (verified against hmmer/src/makehmmerdb on this exact input).
    assert_eq!(c_meta.sequences[0].source, "s1");
    assert_eq!(c_meta.sequences[0].desc, "first description");
    assert_eq!(c_meta.sequences[1].target_id, 1);
    assert_eq!(c_meta.sequences[1].target_start, 1);
    assert_eq!(c_meta.sequences[1].fm_start, 7);
    assert_eq!(c_meta.sequences[1].length, 4);
    assert_eq!(c_meta.sequences[1].name, "s2");
    assert_eq!(c_meta.sequences[1].desc, "second description");
    assert_eq!(c_meta.ambiguities, vec![(2, 5)]);

    assert_eq!(c_stream.version, 1);
    assert!(!c_stream.payload.starts_with(b"HMMERDB\0"));
    assert_eq!(c_stream.meta.fwd_only, c_meta.fwd_only);
    assert_eq!(c_stream.meta.block_count, c_meta.block_count);
    assert_eq!(c_stream.meta.char_count, c_meta.char_count);
    assert_eq!(c_stream.meta.sequences.len(), 2);
    assert_eq!(c_stream.records.len(), 2);
    assert!(c_stream.records[0].has_text_and_sa);
    assert!(!c_stream.records[1].has_text_and_sa);
    assert_eq!(c_stream.records[0].n, c_meta.char_count + 1);
    assert_eq!(c_stream.records[1].n, c_meta.char_count + 1);
    assert_eq!(c_stream.records[0].seq_cnt, 2);
    assert_eq!(c_stream.records[0].ambig_cnt, 1);
}

#[test]
fn makehmmerdb_fwd_only_suppresses_reverse_strand_index_records() {
    let dir = tempfile::tempdir().unwrap();
    let seq = dir.path().join("seq.fa");
    let default_db = dir.path().join("seq.default.hmmerdb");
    let fwd_only_db = dir.path().join("seq.fwd_only.hmmerdb");
    std::fs::write(&seq, ">s1\nACGTACGT\n>s2\nTTTT\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--dna",
            "--container",
            seq.to_str().unwrap(),
            default_db.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--dna",
            "--container",
            "--fwd_only",
            seq.to_str().unwrap(),
            fwd_only_db.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let default_bytes = std::fs::read(default_db).unwrap();
    assert!(default_bytes.starts_with(b"HMMERDB\0"));
    assert!(!parse_makehmmerdb_c_metadata(&default_bytes).fwd_only);
    let default_indexes = parse_makehmmerdb_indexes(&default_bytes);
    assert!(!default_indexes.fwd_only);
    assert_eq!(
        default_indexes
            .records
            .iter()
            .map(|record| record.kind)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(default_indexes
        .records
        .iter()
        .all(|record| record.block_id == 0));
    assert!(default_indexes
        .records
        .iter()
        .all(|record| record.text_len > 0));
    assert_eq!(
        default_indexes.records[0].text_len,
        default_indexes.records[1].text_len
    );

    let fwd_only_bytes = std::fs::read(fwd_only_db).unwrap();
    assert!(fwd_only_bytes.starts_with(b"HMMERDB\0"));
    assert!(parse_makehmmerdb_c_metadata(&fwd_only_bytes).fwd_only);
    let fwd_only_c_stream = parse_makehmmerdb_c_stream(&fwd_only_bytes);
    assert!(fwd_only_c_stream.meta.fwd_only);
    assert_eq!(fwd_only_c_stream.records.len(), 1);
    assert!(fwd_only_c_stream.records[0].has_text_and_sa);
    let fwd_only_indexes = parse_makehmmerdb_indexes(&fwd_only_bytes);
    assert!(fwd_only_indexes.fwd_only);
    assert_eq!(
        fwd_only_indexes
            .records
            .iter()
            .map(|record| record.kind)
            .collect::<Vec<_>>(),
        vec![0]
    );
}

#[test]
fn makehmmerdb_amino_writes_c_compatible_fm_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let seq = dir.path().join("protein.fa");
    let db = dir.path().join("protein.hmmerdb");
    std::fs::write(&seq, ">p1 small protein\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--amino",
            seq.to_str().unwrap(),
            db.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(db).unwrap();
    assert!(!bytes.starts_with(b"HMMERDB\0"));
    let stream = parse_makehmmerdb_native_c_stream(&bytes);
    assert!(stream.meta.fwd_only);
    assert_eq!(stream.meta.alph_type, 4);
    assert_eq!(stream.meta.alph_size, 26);
    assert_eq!(stream.meta.char_bits, 5);
    assert_eq!(stream.meta.char_count, 20);
    assert_eq!(stream.records.len(), 1);
    assert!(stream.records[0].has_text_and_sa);
    assert_eq!(stream.records[0].n, 21);
    assert_eq!(stream.records[0].seq_cnt, 1);
    assert_eq!(stream.records[0].ambig_cnt, 0);
}

#[test]
fn makehmmerdb_accepts_stdin_informat_and_tuning_flags() {
    let dir = tempfile::tempdir().unwrap();
    let stdin_db = dir.path().join("stdin.hmmerdb");
    let seq = b">s1\nACGTACGT\n>s2\nTTTT\n";
    let output = run_with_stdin(
        &[
            "makehmmerdb",
            "--dna",
            "--informat",
            "fasta",
            "-",
            stdin_db.to_str().unwrap(),
        ],
        seq,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!std::fs::read(stdin_db).unwrap().starts_with(b"HMMERDB\0"));

    let seq_path = dir.path().join("rna.fa");
    let tuned_db = dir.path().join("rna.hmmerdb");
    std::fs::write(&seq_path, ">r1\nACGUACGU\n").unwrap();
    let output = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--rna",
            "--bin_length",
            "128",
            "--block_size",
            "1",
            seq_path.to_str().unwrap(),
            tuned_db.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!std::fs::read(tuned_db).unwrap().starts_with(b"HMMERDB\0"));
}

#[derive(Debug)]
struct MakehmmerdbMetadata {
    seq_count: u64,
    block_count: u64,
    ambig_count: u64,
    sequences: Vec<MakehmmerdbSequenceMetadata>,
    ambiguities: Vec<(u64, u64)>,
}

#[derive(Debug)]
struct MakehmmerdbSequenceMetadata {
    name: String,
    desc: String,
    fm_start: u64,
    length: u64,
}

#[derive(Debug)]
struct MakehmmerdbIndexMetadata {
    fwd_only: bool,
    records: Vec<MakehmmerdbIndexRecord>,
}

#[derive(Debug)]
struct MakehmmerdbIndexRecord {
    block_id: u64,
    kind: u32,
    text_len: u64,
}

#[derive(Debug)]
struct MakehmmerdbCMetadata {
    fwd_only: bool,
    alph_type: u8,
    alph_size: u8,
    char_bits: u8,
    freq_sa: u32,
    freq_cnt_sb: u32,
    freq_cnt_b: u32,
    block_count: u16,
    char_count: u64,
    sequences: Vec<MakehmmerdbCSequenceMetadata>,
    ambiguities: Vec<(i32, i32)>,
}

#[derive(Debug)]
struct MakehmmerdbCStream {
    version: u32,
    payload: Vec<u8>,
    meta: MakehmmerdbCMetadata,
    records: Vec<MakehmmerdbCFmRecord>,
}

#[derive(Debug)]
struct MakehmmerdbCFmRecord {
    n: u64,
    has_text_and_sa: bool,
    seq_cnt: u32,
    ambig_cnt: u32,
}

#[derive(Debug)]
struct MakehmmerdbCSequenceMetadata {
    target_id: u32,
    target_start: u64,
    fm_start: u32,
    length: u32,
    name: String,
    acc: String,
    source: String,
    desc: String,
}

fn parse_makehmmerdb_indexes(bytes: &[u8]) -> MakehmmerdbIndexMetadata {
    let magic = b"HMMERDB_INDEXES\0";
    let pos = bytes
        .windows(magic.len())
        .position(|window| window == magic)
        .expect("index extension not found");
    let mut cursor = std::io::Cursor::new(&bytes[pos + magic.len()..]);

    assert_eq!(read_u32_le(&mut cursor), 1);
    let fwd_only = read_u32_le(&mut cursor) != 0;
    let record_count = read_u64_le(&mut cursor);

    let mut records = Vec::new();
    for _ in 0..record_count {
        let block_id = read_u64_le(&mut cursor);
        let _text_start = read_u64_le(&mut cursor);
        let text_len = read_u64_le(&mut cursor);
        let _seq_offset = read_u64_le(&mut cursor);
        let _seq_count = read_u64_le(&mut cursor);
        let _ambig_offset = read_u64_le(&mut cursor);
        let _ambig_count = read_u64_le(&mut cursor);
        let _overlap_bases = read_u64_le(&mut cursor);
        let kind = read_u32_le(&mut cursor);
        let bwt_len = read_u64_le(&mut cursor) as usize;
        let sa_len = read_u64_le(&mut cursor) as usize;
        let c_len = read_u64_le(&mut cursor) as usize;
        let mut skip = vec![0; bwt_len + sa_len * 4 + c_len * 8];
        cursor.read_exact(&mut skip).unwrap();
        records.push(MakehmmerdbIndexRecord {
            block_id,
            kind,
            text_len,
        });
    }

    MakehmmerdbIndexMetadata { fwd_only, records }
}

fn parse_makehmmerdb_c_metadata(bytes: &[u8]) -> MakehmmerdbCMetadata {
    let magic = b"HMMERDB_C_META\0";
    let pos = bytes
        .windows(magic.len())
        .position(|window| window == magic)
        .expect("C metadata extension not found");
    let mut cursor = std::io::Cursor::new(&bytes[pos + magic.len()..]);

    assert_eq!(read_u32_le(&mut cursor), 1);
    parse_makehmmerdb_c_metadata_payload(&mut cursor)
}

fn parse_makehmmerdb_c_stream(bytes: &[u8]) -> MakehmmerdbCStream {
    let magic = b"HMMERDB_C_STREAM\0";
    let pos = bytes
        .windows(magic.len())
        .position(|window| window == magic)
        .expect("C stream extension not found");
    let mut cursor = std::io::Cursor::new(&bytes[pos + magic.len()..]);

    let version = read_u32_le(&mut cursor);
    let payload_len = read_u64_le(&mut cursor) as usize;
    let mut payload = vec![0; payload_len];
    cursor.read_exact(&mut payload).unwrap();

    parse_makehmmerdb_c_stream_payload(version, payload)
}

fn parse_makehmmerdb_native_c_stream(bytes: &[u8]) -> MakehmmerdbCStream {
    parse_makehmmerdb_c_stream_payload(1, bytes.to_vec())
}

fn parse_makehmmerdb_c_stream_payload(version: u32, payload: Vec<u8>) -> MakehmmerdbCStream {
    let mut payload_cursor = std::io::Cursor::new(payload.as_slice());
    let meta = parse_makehmmerdb_c_metadata_payload(&mut payload_cursor);
    let record_passes = if meta.fwd_only { 1 } else { 2 };
    let mut records = Vec::new();
    for _ in 0..meta.block_count {
        for pass in 0..record_passes {
            records.push(parse_makehmmerdb_c_fm_record(
                &mut payload_cursor,
                &meta,
                pass == 0,
            ));
        }
    }
    assert_eq!(payload_cursor.position() as usize, payload.len());

    MakehmmerdbCStream {
        version,
        payload,
        meta,
        records,
    }
}

fn parse_makehmmerdb_c_metadata_payload<R: Read>(cursor: &mut R) -> MakehmmerdbCMetadata {
    let fwd_only = read_u8(cursor) != 0;
    let alph_type = read_u8(cursor);
    let alph_size = read_u8(cursor);
    let char_bits = read_u8(cursor);
    let freq_sa = read_u32_le(cursor);
    let freq_cnt_sb = read_u32_le(cursor);
    let freq_cnt_b = read_u32_le(cursor);
    let block_count = read_u16_le(cursor);
    let seq_count = read_u32_le(cursor);
    let ambig_count = read_u32_le(cursor);
    let char_count = read_u64_le(cursor);

    let mut sequences = Vec::new();
    for _ in 0..seq_count {
        let target_id = read_u32_le(cursor);
        let target_start = read_u64_le(cursor);
        let fm_start = read_u32_le(cursor);
        let length = read_u32_le(cursor);
        let name_len = read_u16_le(cursor) as usize;
        let acc_len = read_u16_le(cursor) as usize;
        let source_len = read_u16_le(cursor) as usize;
        let desc_len = read_u16_le(cursor) as usize;
        let name = read_c_string(cursor, name_len);
        let acc = read_c_string(cursor, acc_len);
        let source = read_c_string(cursor, source_len);
        let desc = read_c_string(cursor, desc_len);
        sequences.push(MakehmmerdbCSequenceMetadata {
            target_id,
            target_start,
            fm_start,
            length,
            name,
            acc,
            source,
            desc,
        });
    }

    let mut ambiguities = Vec::new();
    for _ in 0..ambig_count {
        ambiguities.push((read_i32_le(cursor), read_i32_le(cursor)));
    }

    MakehmmerdbCMetadata {
        fwd_only,
        alph_type,
        alph_size,
        char_bits,
        freq_sa,
        freq_cnt_sb,
        freq_cnt_b,
        block_count,
        char_count,
        sequences,
        ambiguities,
    }
}

fn parse_makehmmerdb_c_fm_record<R: Read>(
    cursor: &mut R,
    meta: &MakehmmerdbCMetadata,
    has_text_and_sa: bool,
) -> MakehmmerdbCFmRecord {
    let n = read_u64_le(cursor);
    let term_loc = read_u32_le(cursor);
    let _seq_offset = read_u32_le(cursor);
    let _ambig_offset = read_u32_le(cursor);
    let _overlap = read_u32_le(cursor);
    let seq_cnt = read_u32_le(cursor);
    let ambig_cnt = read_u32_le(cursor);

    let compressed_bytes = n.div_ceil((8 / meta.char_bits) as u64) as usize;
    let num_freq_cnts_b = 1 + n.div_ceil(meta.freq_cnt_b as u64) as usize;
    let num_freq_cnts_sb = 1 + n.div_ceil(meta.freq_cnt_sb as u64) as usize;
    let num_sa_samples = 1 + (n / meta.freq_sa as u64) as usize;

    assert!((term_loc as u64) < n);
    if has_text_and_sa {
        skip_bytes(cursor, compressed_bytes);
    }
    skip_bytes(cursor, compressed_bytes);
    if has_text_and_sa {
        skip_bytes(cursor, num_sa_samples * 4);
    }
    skip_bytes(cursor, num_freq_cnts_b * meta.alph_size as usize * 2);
    skip_bytes(cursor, num_freq_cnts_sb * meta.alph_size as usize * 4);

    MakehmmerdbCFmRecord {
        n,
        has_text_and_sa,
        seq_cnt,
        ambig_cnt,
    }
}

fn skip_bytes<R: Read>(reader: &mut R, len: usize) {
    let mut skip = vec![0; len];
    reader.read_exact(&mut skip).unwrap();
}

fn parse_makehmmerdb_metadata(bytes: &[u8]) -> MakehmmerdbMetadata {
    let magic = b"HMMERDB_META\0";
    let pos = bytes
        .windows(magic.len())
        .position(|window| window == magic)
        .expect("metadata extension not found");
    let mut cursor = std::io::Cursor::new(&bytes[pos + magic.len()..]);

    assert_eq!(read_u32_le(&mut cursor), 1);
    let _block_size_mb = read_u64_le(&mut cursor);
    let _overlap_bases = read_u64_le(&mut cursor);
    let seq_count = read_u64_le(&mut cursor);
    let block_count = read_u64_le(&mut cursor);
    let ambig_count = read_u64_le(&mut cursor);

    let mut sequences = Vec::new();
    for _ in 0..seq_count {
        let _target_id = read_u64_le(&mut cursor);
        let _target_start = read_u64_le(&mut cursor);
        let fm_start = read_u64_le(&mut cursor);
        let length = read_u64_le(&mut cursor);
        let _block_id = read_u64_le(&mut cursor);
        let _block_offset = read_u64_le(&mut cursor);
        let _overlap_bases = read_u64_le(&mut cursor);
        let name = read_string(&mut cursor);
        let _acc = read_string(&mut cursor);
        let desc = read_string(&mut cursor);
        sequences.push(MakehmmerdbSequenceMetadata {
            name,
            desc,
            fm_start,
            length,
        });
    }

    for _ in 0..block_count {
        for _ in 0..8 {
            let _ = read_u64_le(&mut cursor);
        }
    }

    let mut ambiguities = Vec::new();
    for _ in 0..ambig_count {
        ambiguities.push((read_u64_le(&mut cursor), read_u64_le(&mut cursor)));
    }

    MakehmmerdbMetadata {
        seq_count,
        block_count,
        ambig_count,
        sequences,
        ambiguities,
    }
}

fn read_u8<R: Read>(reader: &mut R) -> u8 {
    let mut buf = [0; 1];
    reader.read_exact(&mut buf).unwrap();
    buf[0]
}

fn read_u16_le<R: Read>(reader: &mut R) -> u16 {
    let mut buf = [0; 2];
    reader.read_exact(&mut buf).unwrap();
    u16::from_le_bytes(buf)
}

fn read_u32_le<R: Read>(reader: &mut R) -> u32 {
    let mut buf = [0; 4];
    reader.read_exact(&mut buf).unwrap();
    u32::from_le_bytes(buf)
}

fn read_i32_le<R: Read>(reader: &mut R) -> i32 {
    let mut buf = [0; 4];
    reader.read_exact(&mut buf).unwrap();
    i32::from_le_bytes(buf)
}

fn read_u64_le<R: Read>(reader: &mut R) -> u64 {
    let mut buf = [0; 8];
    reader.read_exact(&mut buf).unwrap();
    u64::from_le_bytes(buf)
}

fn read_string<R: Read>(reader: &mut R) -> String {
    let len = read_u32_le(reader) as usize;
    let mut buf = vec![0; len];
    reader.read_exact(&mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}

fn read_c_string<R: Read>(reader: &mut R, len: usize) -> String {
    let mut buf = vec![0; len + 1];
    reader.read_exact(&mut buf).unwrap();
    assert_eq!(buf.pop(), Some(0));
    String::from_utf8(buf).unwrap()
}

#[test]
fn hmmsim_is_deterministic_and_supports_c_scoring_option_names() {
    let out1 = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "3",
            "-L",
            "12",
            "--seed",
            "7",
            "--vit",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    let out2 = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "3",
            "-L",
            "12",
            "--seed",
            "7",
            "--vit",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();

    assert!(
        out1.status.success(),
        "{}",
        String::from_utf8_lossy(&out1.stderr)
    );
    assert!(
        out2.status.success(),
        "{}",
        String::from_utf8_lossy(&out2.stderr)
    );
    assert_eq!(out1.stdout, out2.stdout);
    let stdout = String::from_utf8(out1.stdout).unwrap();
    assert!(stdout.contains("# hmmsim: 3 random sequences of length 12 against fn3"));
    assert_eq!(
        stdout
            .lines()
            .filter(|line| line.split_whitespace().count() == 1)
            .count(),
        3
    );
}

#[test]
fn hmmsim_accepts_negative_seed_like_c() {
    // C hmmsim.c declares --seed as eslARG_INT with NO range, so a negative seed
    // is accepted (the C binary exits 0). Match that: the run must succeed.
    let output = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "2",
            "-L",
            "10",
            "--seed",
            "-5",
            "--vit",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "negative --seed should be accepted: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# hmmsim: 2 random sequences of length 10 against fn3"));
}

#[test]
fn hmmsim_supports_forward_hybrid_msv_aliases_and_output_file() {
    for mode in ["--fwd", "--forward", "--hyb", "--msv"] {
        let output = Command::new(hmmer())
            .args([
                "sim",
                "-N",
                "2",
                "-L",
                "10",
                "--seed",
                "11",
                mode,
                "hmmer/tutorial/fn3.hmm",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("# hmmsim: 2 random sequences of length 10 against fn3"));
        let score_rows = stdout
            .lines()
            .filter(|line| line.split_whitespace().count() == 1)
            .count();
        assert_eq!(score_rows, 2);
    }

    let fast = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "2",
            "-L",
            "10",
            "--seed",
            "11",
            "-v",
            "--fast",
            "--vit",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        fast.status.success(),
        "{}",
        String::from_utf8_lossy(&fast.stderr)
    );
    let stdout = String::from_utf8(fast.stdout).unwrap();
    assert!(stdout.contains("# hmmsim: 2 random sequences of length 10 against fn3"));
    assert_eq!(
        stdout
            .lines()
            .filter(|line| line.split_whitespace().count() == 1)
            .count(),
        2
    );

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim.out");
    let output = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "2",
            "-L",
            "10",
            "--seed",
            "11",
            "--vit",
            "-o",
            out.to_str().unwrap(),
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    let written = std::fs::read_to_string(out).unwrap();
    assert!(written.contains("# hmmsim: 2 random sequences of length 10 against fn3"));
}

#[test]
fn hmmsim_writes_statistical_output_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let pfile = dir.path().join("survival.xy");
    let efile = dir.path().join("expect.xy");
    let ffile = dir.path().join("filter.tsv");
    let xfile = dir.path().join("scores.bin");
    let afile = dir.path().join("align.tsv");

    let output = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "6",
            "-L",
            "12",
            "--seed",
            "23",
            "--vit",
            "--pfile",
            pfile.to_str().unwrap(),
            "--efile",
            efile.to_str().unwrap(),
            "--ffile",
            ffile.to_str().unwrap(),
            "--pthresh",
            "0.02",
            "--xfile",
            xfile.to_str().unwrap(),
            "-a",
            "--afile",
            afile.to_str().unwrap(),
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        stdout
            .lines()
            .filter(|line| line.split_whitespace().count() == 1)
            .count(),
        6
    );

    let ptext = std::fs::read_to_string(&pfile).unwrap();
    assert_eq!(ptext.lines().filter(|line| *line == "&").count(), 3);
    let first_pdata = ptext.lines().find(|line| *line != "&").unwrap();
    assert!(
        first_pdata.contains('\t'),
        "survival plot rows should use C xmgrace tab-separated fields: {first_pdata}"
    );
    assert!(ptext.ends_with("&\n"));

    let etext = std::fs::read_to_string(&efile).unwrap();
    assert!(etext.starts_with("# fn3\n1 "));
    assert_eq!(
        etext
            .lines()
            .filter(|line| !line.starts_with('#') && *line != "&")
            .count(),
        6
    );
    assert!(etext.ends_with("&\n"));

    let ftext = std::fs::read_to_string(&ffile).unwrap();
    let fields: Vec<_> = ftext.trim_end().split('\t').collect();
    assert_eq!(fields[0], "fn3");
    assert_eq!(fields.len(), 3);

    let xbytes = std::fs::read(&xfile).unwrap();
    assert_eq!(xbytes.len(), 6 * std::mem::size_of::<f64>());

    let atext = std::fs::read_to_string(&afile).unwrap();
    assert!(atext.starts_with("# fn3\n# alilen bitscore\n"));
    assert_eq!(
        atext.lines().filter(|line| !line.starts_with('#')).count(),
        6
    );
}

#[test]
fn hmmsim_wires_experimental_background_length_and_msv_nu_options() {
    let base_args = [
        "sim",
        "-N",
        "4",
        "-L",
        "14",
        "--seed",
        "19",
        "--vit",
        "hmmer/tutorial/fn3.hmm",
    ];
    let default = Command::new(hmmer()).args(base_args).output().unwrap();
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );

    for option in ["--bgflat", "--bgcomp", "--x-no-lengthmodel"] {
        let mut args = base_args.to_vec();
        args.insert(1, option);
        let output = Command::new(hmmer()).args(args).output().unwrap();
        assert!(
            output.status.success(),
            "{option}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_ne!(output.stdout, default.stdout, "{option} had no effect");
    }

    let msv_default = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "4",
            "-L",
            "14",
            "--seed",
            "19",
            "--msv",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    let msv_nu = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "4",
            "-L",
            "14",
            "--seed",
            "19",
            "--msv",
            "--nu",
            "3.5",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(msv_default.status.success());
    assert!(
        msv_nu.status.success(),
        "{}",
        String::from_utf8_lossy(&msv_nu.stderr)
    );
    assert_ne!(msv_nu.stdout, msv_default.stdout);

    let dir = tempfile::tempdir().unwrap();
    let default_pfile = dir.path().join("default.xy");
    let tuned_pfile = dir.path().join("tuned.xy");
    let default = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "4",
            "-L",
            "14",
            "--seed",
            "19",
            "--vit",
            "--pfile",
            default_pfile.to_str().unwrap(),
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    let tuned = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "4",
            "-L",
            "14",
            "--seed",
            "19",
            "--vit",
            "--EvL",
            "30",
            "--EvN",
            "20",
            "--pfile",
            tuned_pfile.to_str().unwrap(),
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        tuned.status.success(),
        "{}",
        String::from_utf8_lossy(&tuned.stderr)
    );
    assert_ne!(tuned.stdout, default.stdout);
    assert_ne!(
        std::fs::read_to_string(tuned_pfile).unwrap(),
        std::fs::read_to_string(default_pfile).unwrap()
    );
}

#[test]
fn hmmsim_forward_pfile_uses_tail_mass_controls() {
    let dir = tempfile::tempdir().unwrap();
    let default_pfile = dir.path().join("forward-default.xy");
    let wide_tail_pfile = dir.path().join("forward-wide-tail.xy");
    let default = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "30",
            "-L",
            "10",
            "--seed",
            "29",
            "--fwd",
            "--EfL",
            "10",
            "--EfN",
            "5",
            "--pfile",
            default_pfile.to_str().unwrap(),
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );

    let wide = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "30",
            "-L",
            "10",
            "--seed",
            "29",
            "--fwd",
            "--EfL",
            "10",
            "--EfN",
            "5",
            "--tmin",
            "0.50",
            "--tmax",
            "0.50",
            "--pfile",
            wide_tail_pfile.to_str().unwrap(),
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        wide.status.success(),
        "{}",
        String::from_utf8_lossy(&wide.stderr)
    );

    let default_ptext = std::fs::read_to_string(default_pfile).unwrap();
    let wide_tail_ptext = std::fs::read_to_string(wide_tail_pfile).unwrap();
    assert_eq!(default_ptext.lines().filter(|line| *line == "&").count(), 3);
    assert_eq!(
        wide_tail_ptext.lines().filter(|line| *line == "&").count(),
        3
    );
    assert_ne!(wide_tail_ptext, default_ptext);
}

#[test]
fn hmmsim_forward_prints_every_tail_mass_sweep_row() {
    let output = Command::new(hmmer())
        .args([
            "sim",
            "-N",
            "30",
            "-L",
            "10",
            "--seed",
            "29",
            "--fwd",
            "--EfL",
            "10",
            "--EfN",
            "5",
            "--tmin",
            "0.02",
            "--tmax",
            "0.08",
            "--tpoints",
            "3",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let rows: Vec<_> = stdout
        .lines()
        .filter(|line| line.starts_with("fn3 "))
        .collect();
    assert_eq!(rows.len(), 3, "{stdout}");
    for (row, tailp) in rows.iter().zip(["0.0200", "0.0400", "0.0800"]) {
        let fields: Vec<_> = row.split_whitespace().collect();
        assert_eq!(fields.len(), 10, "{row}");
        assert_eq!(fields[0], "fn3");
        assert_eq!(fields[1], tailp);
        for field in &fields[2..] {
            field.parse::<f64>().unwrap();
        }
    }
}

#[test]
fn hmmsim_viterbi_msv_and_hybrid_print_c_shaped_gumbel_summary_row() {
    for mode in ["--vit", "--msv", "--hyb"] {
        let output = Command::new(hmmer())
            .args([
                "sim",
                "-N",
                "30",
                "-L",
                "10",
                "--seed",
                "31",
                mode,
                "hmmer/tutorial/fn3.hmm",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8(output.stdout).unwrap();
        assert_eq!(
            stdout
                .lines()
                .filter(|line| line.split_whitespace().count() == 1)
                .count(),
            30,
            "{mode}: {stdout}"
        );
        let rows: Vec<_> = stdout
            .lines()
            .filter(|line| line.starts_with("fn3 "))
            .collect();
        assert_eq!(rows.len(), 1, "{mode}: {stdout}");
        let fields: Vec<_> = rows[0].split_whitespace().collect();
        assert_eq!(fields.len(), 12, "{mode}: {}", rows[0]);
        assert_eq!(fields[0], "fn3");
        assert_eq!(fields[1], "1.0000");
        for field in &fields[2..] {
            field.parse::<f64>().unwrap();
        }
    }
}

fn run_with_stdin(args: &[&str], stdin_bytes: &[u8]) -> std::process::Output {
    let mut child = Command::new(hmmer())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(stdin_bytes)
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn hmm_holding_commands_accept_hmmfile_from_stdin() {
    let hmm = std::fs::read("hmmer/tutorial/fn3.hmm").unwrap();

    let convert_file = Command::new(hmmer())
        .args(["convert", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    let convert_stdin = run_with_stdin(&["convert", "-"], &hmm);
    assert!(convert_file.status.success());
    assert!(convert_stdin.status.success());
    assert_eq!(convert_file.stdout, convert_stdin.stdout);

    let stat_file = Command::new(hmmer())
        .args(["stat", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    let stat_stdin = run_with_stdin(&["stat", "-"], &hmm);
    assert!(stat_file.status.success());
    assert!(stat_stdin.status.success());
    assert_eq!(stat_file.stdout, stat_stdin.stdout);

    let emit_file = Command::new(hmmer())
        .args(["emit", "--seed", "42", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    let emit_stdin = run_with_stdin(&["emit", "--seed", "42", "-"], &hmm);
    assert!(emit_file.status.success());
    assert!(emit_stdin.status.success());
    assert_eq!(emit_file.stdout, emit_stdin.stdout);

    let emit_seed_zero = Command::new(hmmer())
        .args(["emit", "--seed", "0", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    assert!(
        emit_seed_zero.status.success(),
        "{}",
        String::from_utf8_lossy(&emit_seed_zero.stderr)
    );
    assert_ne!(emit_seed_zero.stdout, emit_file.stdout);

    let logo_file = Command::new(hmmer())
        .args(["logo", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    let logo_stdin = run_with_stdin(&["logo", "-"], &hmm);
    assert!(logo_file.status.success());
    assert!(logo_stdin.status.success());
    assert_eq!(logo_file.stdout, logo_stdin.stdout);
}

#[test]
fn hmmemit_cli_modes_and_output_file_work() {
    let consensus = Command::new(hmmer())
        .args(["emit", "-c", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    assert!(
        consensus.status.success(),
        "{}",
        String::from_utf8_lossy(&consensus.stderr)
    );
    let stdout = String::from_utf8(consensus.stdout).unwrap();
    assert!(stdout.starts_with(">fn3-consensus\n"));

    let fancy = Command::new(hmmer())
        .args([
            "emit",
            "-C",
            "--minl",
            "0.4",
            "--minu",
            "0.8",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        fancy.status.success(),
        "{}",
        String::from_utf8_lossy(&fancy.stderr)
    );
    let stdout = String::from_utf8(fancy.stdout).unwrap();
    assert!(stdout.starts_with(">fn3-consensus\n"));

    let alignment = Command::new(hmmer())
        .args([
            "emit",
            "-a",
            "-N",
            "2",
            "--seed",
            "7",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        alignment.status.success(),
        "{}",
        String::from_utf8_lossy(&alignment.stderr)
    );
    let stdout = String::from_utf8(alignment.stdout).unwrap();
    assert!(stdout.starts_with("# STOCKHOLM 1.0\n"));
    assert!(stdout.contains("fn3-sample1"));
    assert!(stdout.contains("fn3-sample2"));

    // `-a` Stockholm output must be byte-identical to the C binary, including
    // interleaved block wrapping (width 200) and Easel insert rejustification.
    // A 374-residue model with -N 4 forces multiple wrapped blocks with inserts.
    if std::path::Path::new(&c_hmmemit()).exists() {
        for (seed, n, hmm) in [
            ("5", "4", "hmmer/testsuite/gecco_missed_hmms.hmm"),
            ("7", "8", "hmmer/testsuite/gecco_missed_hmms.hmm"),
            ("3", "10", "hmmer/testsuite/minipfam.hmm"),
            ("7", "2", "hmmer/tutorial/fn3.hmm"),
        ] {
            let c = Command::new(c_hmmemit())
                .args(["-a", "-N", n, "--seed", seed, hmm])
                .output()
                .unwrap();
            let r = Command::new(hmmer())
                .args(["emit", "-a", "-N", n, "--seed", seed, hmm])
                .output()
                .unwrap();
            assert!(c.status.success() && r.status.success());
            assert_eq!(
                String::from_utf8_lossy(&r.stdout),
                String::from_utf8_lossy(&c.stdout),
                "hmmemit -a output diverges from C for seed={seed} N={n} hmm={hmm}"
            );
        }
    }

    let profile = Command::new(hmmer())
        .args([
            "emit",
            "-p",
            "-L",
            "25",
            "--unilocal",
            "--seed",
            "7",
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        profile.status.success(),
        "{}",
        String::from_utf8_lossy(&profile.stderr)
    );
    let stdout = String::from_utf8(profile.stdout).unwrap();
    assert!(stdout.starts_with(">fn3-sample1\n"));

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("emit.fa");
    let output = Command::new(hmmer())
        .args([
            "emit",
            "-c",
            "-o",
            out.to_str().unwrap(),
            "hmmer/tutorial/fn3.hmm",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    let written = std::fs::read_to_string(out).unwrap();
    assert!(written.starts_with(">fn3-consensus\n"));
}

#[test]
fn hmmfetch_accepts_stdin_sources_and_multifetch_output_file() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("fetch.hmm");
    let hmm = std::fs::read("hmmer/testsuite/Caudal_act.hmm").unwrap();

    let file_fetch = Command::new(hmmer())
        .args(["fetch", "hmmer/testsuite/Caudal_act.hmm", "Caudal_act"])
        .output()
        .unwrap();
    let stdin_fetch = run_with_stdin(&["fetch", "-", "Caudal_act"], &hmm);
    assert!(file_fetch.status.success());
    assert!(stdin_fetch.status.success());
    assert_eq!(file_fetch.stdout, stdin_fetch.stdout);

    let key_fetch = run_with_stdin(
        &[
            "fetch",
            "-f",
            "-o",
            out.to_str().unwrap(),
            "hmmer/testsuite/Caudal_act.hmm",
            "-",
        ],
        b"Caudal_act\n",
    );
    assert!(
        key_fetch.status.success(),
        "{}",
        String::from_utf8_lossy(&key_fetch.stderr)
    );
    let stdout = String::from_utf8(key_fetch.stdout).unwrap();
    assert!(stdout.contains("Retrieved 1 HMMs."), "{stdout}");
    let fetched = std::fs::read_to_string(out).unwrap();
    assert!(fetched.contains("NAME  Caudal_act"));

    let stdout_fetch = run_with_stdin(
        &["fetch", "-f", "hmmer/testsuite/Caudal_act.hmm", "-"],
        b"Caudal_act\n",
    );
    assert!(
        stdout_fetch.status.success(),
        "{}",
        String::from_utf8_lossy(&stdout_fetch.stderr)
    );
    let stdout = String::from_utf8(stdout_fetch.stdout).unwrap();
    assert!(stdout.contains("NAME  Caudal_act"));
    assert!(!stdout.contains("Retrieved 1 HMMs."), "{stdout}");
}

#[test]
fn hmmfetch_keyfile_uses_file_order_without_index_and_ignores_missing_keys() {
    let dir = tempfile::tempdir().unwrap();
    let multi_hmm = dir.path().join("multi.hmm");
    let mut hmms = std::fs::read("hmmer/testsuite/Caudal_act.hmm").unwrap();
    hmms.extend_from_slice(&std::fs::read("hmmer/tutorial/fn3.hmm").unwrap());
    std::fs::write(&multi_hmm, hmms).unwrap();

    let reverse_keys = dir.path().join("reverse.keys");
    std::fs::write(&reverse_keys, "fn3\nCaudal_act\n").unwrap();
    let output = Command::new(hmmer())
        .args([
            "fetch",
            "-f",
            multi_hmm.to_str().unwrap(),
            reverse_keys.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let fn3_pos = stdout.find("NAME  fn3").unwrap();
    let caudal_pos = stdout.find("NAME  Caudal_act").unwrap();
    assert!(caudal_pos < fn3_pos, "{stdout}");

    let missing_keys = dir.path().join("missing.keys");
    std::fs::write(&missing_keys, "Caudal_act\nNO_SUCH_MODEL\n").unwrap();
    let output = Command::new(hmmer())
        .args([
            "fetch",
            "-f",
            multi_hmm.to_str().unwrap(),
            missing_keys.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("NAME  Caudal_act"), "{stdout}");
    assert!(!stdout.contains("NO_SUCH_MODEL"), "{stdout}");
    assert!(!stdout.contains("NAME  fn3"), "{stdout}");
}

#[test]
fn hmmfetch_indexes_accessions_and_output_key_files() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmm_path).unwrap();

    let index = Command::new(hmmer())
        .args(["fetch", "--index", hmm_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        index.status.success(),
        "{}",
        String::from_utf8_lossy(&index.stderr)
    );
    let stdout = String::from_utf8(index.stdout).unwrap();
    assert!(stdout.contains("SSI index written to file"));
    let ssi_path = std::path::PathBuf::from(format!("{}.ssi", hmm_path.to_string_lossy()));
    assert!(ssi_path.exists());

    let accession_fetch = Command::new(hmmer())
        .args(["fetch", hmm_path.to_str().unwrap(), "PF00041.13"])
        .output()
        .unwrap();
    assert!(
        accession_fetch.status.success(),
        "{}",
        String::from_utf8_lossy(&accession_fetch.stderr)
    );
    let stdout = String::from_utf8(accession_fetch.stdout).unwrap();
    assert!(stdout.contains("NAME  fn3"));
    assert!(stdout.contains("ACC   PF00041.13"));

    let output_dir = tempfile::tempdir().unwrap();
    let key_fetch = Command::new(hmmer())
        .current_dir(output_dir.path())
        .args(["fetch", "-O", hmm_path.to_str().unwrap(), "PF00041.13"])
        .output()
        .unwrap();
    assert!(
        key_fetch.status.success(),
        "{}",
        String::from_utf8_lossy(&key_fetch.stderr)
    );
    let keyed = output_dir.path().join("PF00041.13");
    assert!(keyed.exists());
    let fetched = std::fs::read_to_string(keyed).unwrap();
    assert!(fetched.contains("NAME  fn3"));
}

#[cfg(unix)]
#[test]
fn hmmfetch_index_preserves_non_utf8_ssi_path_bytes() {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};

    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir
        .path()
        .join(std::ffi::OsString::from_vec(b"fn3-\xff.hmm".to_vec()));
    std::fs::copy("hmmer/tutorial/fn3.hmm", &hmm_path).unwrap();

    let index = Command::new(hmmer())
        .arg("fetch")
        .arg("--index")
        .arg(&hmm_path)
        .output()
        .unwrap();
    assert!(
        index.status.success(),
        "{}",
        String::from_utf8_lossy(&index.stderr)
    );

    let ssi_path = hmmer_pure_rs::ssi::path_with_appended_suffix(&hmm_path, ".ssi");
    assert!(ssi_path.exists(), "missing SSI {}", ssi_path.display());
    assert!(ssi_path.as_os_str().as_bytes().contains(&0xff));
}

#[test]
fn hmmfetch_indexes_binary_h3m_and_fetches_by_accession() {
    let dir = tempfile::tempdir().unwrap();
    let h3m_path = dir.path().join("fn3.h3m");
    let binary = Command::new(hmmer())
        .args(["convert", "-b", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    assert!(
        binary.status.success(),
        "{}",
        String::from_utf8_lossy(&binary.stderr)
    );
    std::fs::write(&h3m_path, binary.stdout).unwrap();

    let index = Command::new(hmmer())
        .args(["fetch", "--index", h3m_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        index.status.success(),
        "{}",
        String::from_utf8_lossy(&index.stderr)
    );
    let ssi_path = std::path::PathBuf::from(format!("{}.ssi", h3m_path.to_string_lossy()));
    assert!(ssi_path.exists());

    let fetch = Command::new(hmmer())
        .args(["fetch", h3m_path.to_str().unwrap(), "PF00041.13"])
        .output()
        .unwrap();
    assert!(
        fetch.status.success(),
        "{}",
        String::from_utf8_lossy(&fetch.stderr)
    );
    let stdout = String::from_utf8(fetch.stdout).unwrap();
    assert!(stdout.contains("NAME  fn3"), "{stdout}");
    assert!(stdout.contains("ACC   PF00041.13"), "{stdout}");
}

#[test]
fn hmmfetch_uses_existing_ssi_for_random_access() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("multi.hmm");
    let mut hmms = std::fs::read("hmmer/testsuite/Caudal_act.hmm").unwrap();
    hmms.extend_from_slice(&std::fs::read("hmmer/tutorial/fn3.hmm").unwrap());
    std::fs::write(&hmm_path, hmms).unwrap();

    let index = Command::new(hmmer())
        .args(["fetch", "--index", hmm_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        index.status.success(),
        "{}",
        String::from_utf8_lossy(&index.stderr)
    );

    let mut with_bad_tail = std::fs::OpenOptions::new()
        .append(true)
        .open(&hmm_path)
        .unwrap();
    use std::io::Write as _;
    writeln!(with_bad_tail, "HMMER3/f\nNAME  bad\nLENG  1").unwrap();

    let by_name = Command::new(hmmer())
        .args(["fetch", hmm_path.to_str().unwrap(), "Caudal_act"])
        .output()
        .unwrap();
    assert!(
        by_name.status.success(),
        "{}",
        String::from_utf8_lossy(&by_name.stderr)
    );
    let stdout = String::from_utf8(by_name.stdout).unwrap();
    assert!(stdout.contains("NAME  Caudal_act"), "{stdout}");
    assert!(!stdout.contains("NAME  fn3"), "{stdout}");

    let by_acc = Command::new(hmmer())
        .args(["fetch", hmm_path.to_str().unwrap(), "PF00041.13"])
        .output()
        .unwrap();
    assert!(
        by_acc.status.success(),
        "{}",
        String::from_utf8_lossy(&by_acc.stderr)
    );
    let stdout = String::from_utf8(by_acc.stdout).unwrap();
    assert!(stdout.contains("NAME  fn3"), "{stdout}");
    assert!(stdout.contains("ACC   PF00041.13"), "{stdout}");
}

#[test]
fn hmmstat_20aa_stdout_matches_golden() {
    let output = Command::new(hmmer())
        .args(["stat", "hmmer/testsuite/20aa.hmm"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let golden = std::fs::read_to_string("tests/golden/hmmstat_20aa.stdout").unwrap();
    assert_eq!(stdout, golden);
}

#[test]
fn hmmalign_accepts_one_stdin_input() {
    let hmm = std::fs::read("hmmer/tutorial/fn3.hmm").unwrap();
    let seq = std::fs::read("hmmer/tutorial/7LESS_DROME").unwrap();

    let file_align = Command::new(hmmer())
        .args([
            "align",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    let hmm_stdin = run_with_stdin(&["align", "-", "hmmer/tutorial/7LESS_DROME"], &hmm);
    let seq_stdin = run_with_stdin(&["align", "hmmer/tutorial/fn3.hmm", "-"], &seq);
    assert!(file_align.status.success());
    assert!(
        hmm_stdin.status.success(),
        "{}",
        String::from_utf8_lossy(&hmm_stdin.stderr)
    );
    assert!(
        seq_stdin.status.success(),
        "{}",
        String::from_utf8_lossy(&seq_stdin.stderr)
    );
    assert_eq!(file_align.stdout, hmm_stdin.stdout);
    assert_eq!(file_align.stdout, seq_stdin.stdout);
}

#[test]
fn hmmalign_accepts_uniprot_informat_assertion() {
    let autodetect = Command::new(hmmer())
        .args([
            "align",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        autodetect.status.success(),
        "{}",
        String::from_utf8_lossy(&autodetect.stderr)
    );

    let asserted = Command::new(hmmer())
        .args([
            "align",
            "--informat",
            "uniprot",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        asserted.status.success(),
        "{}",
        String::from_utf8_lossy(&asserted.stderr)
    );
    assert_eq!(asserted.stdout, autodetect.stdout);
}

#[test]
fn hmmalign_accepts_stockholm_informat_assertion() {
    for informat in ["stockholm", "sto"] {
        let output = Command::new(hmmer())
            .args([
                "align",
                "--informat",
                informat,
                "hmmer/testsuite/20aa.hmm",
                "hmmer/testsuite/20aa.sto",
            ])
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "{informat} stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("# STOCKHOLM 1.0"), "{stdout}");
        assert!(stdout.contains("seq1"), "{stdout}");
    }
}

#[test]
fn hmmalign_supports_a2m_and_output_file() {
    let a2m = Command::new(hmmer())
        .args([
            "align",
            "--outformat",
            "A2M",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        a2m.status.success(),
        "{}",
        String::from_utf8_lossy(&a2m.stderr)
    );
    let stdout = String::from_utf8(a2m.stdout).unwrap();
    assert!(stdout.starts_with(">7LESS_DROME\n"));
    assert!(!stdout.contains("# STOCKHOLM"));

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("aligned.sto");
    let output = Command::new(hmmer())
        .args([
            "align",
            "-o",
            out.to_str().unwrap(),
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    let written = std::fs::read_to_string(out).unwrap();
    assert!(written.starts_with("# STOCKHOLM 1.0\n"));
    assert!(written.contains("7LESS_DROME"));
}

#[test]
fn hmmbuild_accepts_stockholm_stdin_with_informat() {
    for informat in ["stockholm", "sto"] {
        let dir = tempfile::tempdir().unwrap();
        let hmm_out = dir.path().join("stdin-build.hmm");
        let sto = std::fs::read("hmmer/testsuite/20aa.sto").unwrap();

        let output = run_with_stdin(
            &[
                "build",
                "--informat",
                informat,
                hmm_out.to_str().unwrap(),
                "-",
            ],
            &sto,
        );
        assert!(
            output.status.success(),
            "{informat} stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(std::fs::read_to_string(hmm_out)
            .unwrap()
            .contains("NAME  test"));
    }
}

#[test]
fn hmmbuild_builds_from_aligned_fasta_informat() {
    let dir = tempfile::tempdir().unwrap();
    let afa = dir.path().join("toy.afa");
    let hmm_out = dir.path().join("toy.hmm");
    std::fs::write(&afa, ">toy1 first row\nACDEFG\n>toy2 second row\nAC-EFG\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--informat",
            "afa",
            hmm_out.to_str().unwrap(),
            afa.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let saved = std::fs::read_to_string(hmm_out).unwrap();
    assert!(saved.contains("NAME  toy\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
}

#[test]
fn hmmbuild_builds_from_a2m_informat_with_consensus_insert_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let a2m = dir.path().join("toy.a2m");
    let hmm_out = dir.path().join("toy.hmm");
    let processed_msa = dir.path().join("toy.sto");
    std::fs::write(&a2m, ">toy1\nGAATTC\n>toy2\nGAaATTC\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--dna",
            "--informat",
            "a2m",
            "-O",
            processed_msa.to_str().unwrap(),
            hmm_out.to_str().unwrap(),
            a2m.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let saved = std::fs::read_to_string(hmm_out).unwrap();
    assert!(saved.contains("NAME  toy\n"), "{saved}");
    assert!(saved.contains("LENG  6\n"), "{saved}");
    assert!(saved.contains("RF    yes\n"), "{saved}");
    let processed = std::fs::read_to_string(processed_msa).unwrap();
    assert!(processed.contains("toy1 GA-ATTC\n"), "{processed}");
    assert!(processed.contains("toy2 GAaATTC\n"), "{processed}");
    assert!(processed.contains("#=GC RF xx.xxxx\n"), "{processed}");
}

#[test]
fn hmmbuild_builds_from_psiblast_informat() {
    let dir = tempfile::tempdir().unwrap();
    let psi = dir.path().join("toy.psi");
    let hmm_out = dir.path().join("toy.hmm");
    std::fs::write(
        &psi,
        "# tiny PSIBLAST alignment\nseq1  GAA-TTC\nseq2  GAAT-TC\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--dna",
            "--informat",
            "psiblast",
            hmm_out.to_str().unwrap(),
            psi.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let saved = std::fs::read_to_string(hmm_out).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NAME  toy\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
    assert!(saved.contains("LENG  7\n"), "{saved}");
}

#[test]
fn hmmbuild_builds_from_clustal_and_clustallike_informat() {
    for informat in ["clustal", "clustallike"] {
        let dir = tempfile::tempdir().unwrap();
        let aln = dir.path().join(format!("toy-{informat}.aln"));
        let hmm_out = dir.path().join(format!("toy-{informat}.hmm"));
        std::fs::write(
            &aln,
            "CLUSTAL W multiple sequence alignment\n\nseq1  GAA\nseq2  GA-\n      ** \n\nseq1  TTC\nseq2  TTC\n      ***\n",
        )
        .unwrap();

        let output = Command::new(hmmer())
            .args([
                "build",
                "--dna",
                "--informat",
                informat,
                hmm_out.to_str().unwrap(),
                aln.to_str().unwrap(),
            ])
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "{informat} stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("toy-"), "{stdout}");
        let saved = std::fs::read_to_string(hmm_out).unwrap();
        assert!(saved.starts_with("HMMER3/f "), "{saved}");
        assert!(saved.contains("ALPH  DNA\n"), "{saved}");
        assert!(saved.contains("NSEQ  2\n"), "{saved}");
        assert!(saved.contains("LENG  6\n"), "{saved}");
    }
}

#[test]
fn hmmbuild_builds_from_selex_informat() {
    let dir = tempfile::tempdir().unwrap();
    let selex = dir.path().join("toy.slx");
    let hmm_out = dir.path().join("toy.hmm");
    std::fs::write(
        &selex,
        "# tiny interleaved SELEX alignment\nseq1  GAA\nseq2  GA-\n\nseq1  TTC\nseq2  TTC\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--dna",
            "--informat",
            "selex",
            hmm_out.to_str().unwrap(),
            selex.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("toy"), "{stdout}");
    let saved = std::fs::read_to_string(hmm_out).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NAME  toy\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
    assert!(saved.contains("LENG  6\n"), "{saved}");
}

#[test]
fn hmmbuild_builds_from_interleaved_phylip_informat() {
    let dir = tempfile::tempdir().unwrap();
    let phylip = dir.path().join("toy.phy");
    let hmm_out = dir.path().join("toy.hmm");
    std::fs::write(&phylip, "2 6\ntoy1 GAA\ntoy2 GA-\nTTC\nTTC\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--dna",
            "--informat",
            "phylip",
            hmm_out.to_str().unwrap(),
            phylip.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let saved = std::fs::read_to_string(hmm_out).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NAME  toy\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
    assert!(saved.contains("LENG  6\n"), "{saved}");
}

#[test]
fn hmmbuild_builds_from_sequential_phylips_informat() {
    let dir = tempfile::tempdir().unwrap();
    let phylips = dir.path().join("toy.phys");
    let hmm_out = dir.path().join("toy.hmm");
    std::fs::write(&phylips, "2 6\ntoy1 GAA\nTTC\ntoy2 GA-\nTTC\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--dna",
            "--informat",
            "phylips",
            hmm_out.to_str().unwrap(),
            phylips.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let saved = std::fs::read_to_string(hmm_out).unwrap();
    assert!(saved.starts_with("HMMER3/f "), "{saved}");
    assert!(saved.contains("ALPH  DNA\n"), "{saved}");
    assert!(saved.contains("NAME  toy\n"), "{saved}");
    assert!(saved.contains("NSEQ  2\n"), "{saved}");
    assert!(saved.contains("LENG  6\n"), "{saved}");
}

#[test]
fn hmmbuild_name_symfrac_and_summary_output_work() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_out = dir.path().join("custom.hmm");
    let summary = dir.path().join("summary.txt");
    let processed_msa = dir.path().join("processed.sto");

    let output = Command::new(hmmer())
        .args([
            "build",
            "-n",
            "custom",
            "--symfrac",
            "0.7",
            "--fragthresh",
            "0.3",
            "-o",
            summary.to_str().unwrap(),
            "-O",
            processed_msa.to_str().unwrap(),
            hmm_out.to_str().unwrap(),
            "hmmer/testsuite/20aa.sto",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    let summary = std::fs::read_to_string(summary).unwrap();
    assert!(summary.contains("# output directed to file:"));
    assert!(summary.contains("# processed alignment resaved to:"));
    assert!(summary.contains("# sym fraction for model structure: 0.700\n"));
    assert!(summary.contains("# seq called frag if L <= x*alen:  0.300\n"));
    assert!(summary.contains("1     custom"));
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.contains("NAME  custom\n"));
    let processed = std::fs::read_to_string(processed_msa).unwrap();
    assert!(processed.starts_with("# STOCKHOLM 1.0\n"));
    assert!(processed.contains("#=GF ID test\n"));
    assert!(processed.contains("#=GC RF xxxxxxxxxxxxxxxxxxxx\n"));
    assert!(processed.trim_end().ends_with("//"));
}

#[test]
fn hmmbuild_processed_msa_uses_builder_architecture_not_input_rf() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.sto");
    let hmm_out = dir.path().join("built.hmm");
    let processed = dir.path().join("processed.sto");
    std::fs::write(
        &input,
        b"# STOCKHOLM 1.0\n#=GF ID arch\ns1 ACDEFGHIKLMN\ns2 ACDEFGHIKLMN\n#=GC RF ............\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--amino",
            "-O",
            processed.to_str().unwrap(),
            hmm_out.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let processed = std::fs::read_to_string(processed).unwrap();
    assert!(processed.contains("#=GC RF xxxxxxxxxxxx\n"), "{processed}");
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.contains("LENG  12\n"), "{hmm}");
}

#[test]
fn hmmbuild_accepts_explicit_default_weighting_options() {
    let dir = tempfile::tempdir().unwrap();
    for option in ["--wpb", "--eent", "--eentexp"] {
        let hmm_out = dir
            .path()
            .join(format!("{}.hmm", option.trim_start_matches("--")));
        let output = Command::new(hmmer())
            .args([
                "build",
                option,
                hmm_out.to_str().unwrap(),
                "hmmer/tutorial/globins4.sto",
            ])
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "{option}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let hmm = std::fs::read_to_string(hmm_out).unwrap();
        assert!(hmm.starts_with("HMMER3/f"));
        assert!(hmm.contains("LENG  149\n"));
    }
}

#[test]
fn hmmbuild_eentexp_reports_c_header_and_effn() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_out = dir.path().join("eentexp.hmm");
    let output = Command::new(hmmer())
        .args([
            "build",
            "--eentexp",
            hmm_out.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = String::from_utf8(output.stdout).unwrap();
    assert!(summary.contains(
        "# effective seq number scheme:      entropy weighting using exponent-based scaling\n"
    ));
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.contains("EFFN  "));
}

#[test]
fn hmmbuild_supports_no_relative_weighting_option() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_out = dir.path().join("wnone.hmm");
    let output = Command::new(hmmer())
        .args([
            "build",
            "--wnone",
            hmm_out.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = String::from_utf8(output.stdout).unwrap();
    assert!(summary.contains("# relative weighting scheme:        none\n"));
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.starts_with("HMMER3/f"));
    assert!(hmm.contains("LENG  149\n"));
}

#[test]
fn hmmbuild_supports_remaining_relative_weighting_options() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("weighted.sto");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID weighted\n#=GS s1 WT 0.25\n#=GS s2 WT 1.75\n#=GS s3 WT 1.00\ns1 ACDEFGHIKLM\ns2 ACDEYGHIKLM\ns3 ACDEYGHIKL-\n//\n",
    )
    .unwrap();

    for (option, expected_summary) in [
        ("--wgsc", "# relative weighting scheme:        G/S/C\n"),
        (
            "--wblosum",
            "# relative weighting scheme:        BLOSUM filter\n",
        ),
        ("--wgiven", "# relative weighting scheme:        given\n"),
    ] {
        let hmm_out = dir
            .path()
            .join(format!("{}.hmm", option.trim_start_matches("--")));
        let output = Command::new(hmmer())
            .args([
                "build",
                "--amino",
                option,
                hmm_out.to_str().unwrap(),
                sto.to_str().unwrap(),
            ])
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "{option}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let summary = String::from_utf8(output.stdout).unwrap();
        assert!(summary.contains(expected_summary), "{summary}");
        let hmm = std::fs::read_to_string(hmm_out).unwrap();
        assert!(hmm.starts_with("HMMER3/f"));
        assert!(hmm.contains("LENG  11\n"));
    }

    let hmm_out = dir.path().join("wblosum-wid.hmm");
    let output = Command::new(hmmer())
        .args([
            "build",
            "--amino",
            "--wblosum",
            "--wid",
            "0.80",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = String::from_utf8(output.stdout).unwrap();
    assert!(summary.contains("# frac id cutoff for BLOSUM wgts:   0.800000\n"));
}

#[test]
fn hmmbuild_supports_selectable_prior_options() {
    let dir = tempfile::tempdir().unwrap();
    let default_hmm = dir.path().join("default.hmm");
    let none_hmm = dir.path().join("none.hmm");
    let laplace_hmm = dir.path().join("laplace.hmm");

    let default = Command::new(hmmer())
        .args([
            "build",
            default_hmm.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );

    let none = Command::new(hmmer())
        .args([
            "build",
            "--pnone",
            none_hmm.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();
    assert!(
        none.status.success(),
        "{}",
        String::from_utf8_lossy(&none.stderr)
    );
    let none_summary = String::from_utf8(none.stdout).unwrap();
    assert!(none_summary.contains("# prior scheme:                     none\n"));

    let laplace = Command::new(hmmer())
        .args([
            "build",
            "--plaplace",
            laplace_hmm.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();
    assert!(
        laplace.status.success(),
        "{}",
        String::from_utf8_lossy(&laplace.stderr)
    );
    let laplace_summary = String::from_utf8(laplace.stdout).unwrap();
    assert!(laplace_summary.contains("# prior scheme:                     Laplace\n"));

    let default_hmm = std::fs::read_to_string(default_hmm).unwrap();
    let none_hmm = std::fs::read_to_string(none_hmm).unwrap();
    let laplace_hmm = std::fs::read_to_string(laplace_hmm).unwrap();
    assert!(none_hmm.starts_with("HMMER3/f"));
    assert!(laplace_hmm.starts_with("HMMER3/f"));
    assert_ne!(default_hmm, none_hmm);
    assert_ne!(default_hmm, laplace_hmm);
    assert_ne!(none_hmm, laplace_hmm);
}

#[test]
fn hmmbuild_supports_enone_and_seed_options() {
    let dir = tempfile::tempdir().unwrap();
    let enone_hmm = dir.path().join("enone.hmm");
    let enone = Command::new(hmmer())
        .args([
            "build",
            "--enone",
            enone_hmm.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();
    assert!(
        enone.status.success(),
        "{}",
        String::from_utf8_lossy(&enone.stderr)
    );
    let summary = String::from_utf8(enone.stdout).unwrap();
    assert!(summary.contains("# effective seq number scheme:      none\n"));
    assert!(summary.contains("     4.00 "));
    let hmm = std::fs::read_to_string(enone_hmm).unwrap();
    assert!(hmm.contains("EFFN  4.000000\n"));

    let seeded_hmm = dir.path().join("seeded.hmm");
    let seeded = Command::new(hmmer())
        .args([
            "build",
            "--seed",
            "42",
            seeded_hmm.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();
    assert!(
        seeded.status.success(),
        "{}",
        String::from_utf8_lossy(&seeded.stderr)
    );
    let summary = String::from_utf8(seeded.stdout).unwrap();
    assert!(summary.contains("# random number seed set to:        42\n"));
    assert!(std::fs::read_to_string(seeded_hmm)
        .unwrap()
        .starts_with("HMMER3/f"));
}

#[test]
fn hmmbuild_supports_fixed_effective_sequence_number_option() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_out = dir.path().join("eset.hmm");
    let output = Command::new(hmmer())
        .args([
            "build",
            "--eset",
            "2.5",
            hmm_out.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = String::from_utf8(output.stdout).unwrap();
    assert!(summary.contains("# effective seq number:             set to 2.500000\n"));
    assert!(summary.contains("     2.50 "));
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.contains("EFFN  2.500000\n"));
}

#[test]
fn hmmbuild_supports_cluster_effective_sequence_number_option() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("clustered.sto");
    let hmm_out = dir.path().join("eclust.hmm");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID clustered\ns1 ACDEFGHIKLM\ns2 ACDEFGHIKLM\ns3 ACDEYGHIKLM\ns4 YYYYYYYYYYY\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--amino",
            "--eclust",
            "--eid",
            "0.80",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = String::from_utf8(output.stdout).unwrap();
    assert!(summary.contains("# effective seq number scheme:      single linkage clusters\n"));
    assert!(summary.contains("# frac id cutoff for --eclust:      0.800000\n"));
    assert!(summary.contains("     2.00 "));
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.contains("EFFN  2.000000\n"));
}

#[test]
fn hmmbuild_supports_entropy_target_tuning_options() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_out = dir.path().join("entropy-tuned.hmm");
    let output = Command::new(hmmer())
        .args([
            "build",
            "--ere",
            "0.7",
            "--esigma",
            "50",
            hmm_out.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = String::from_utf8(output.stdout).unwrap();
    assert!(summary.contains("# minimum rel entropy target:       0.700000 bits\n"));
    assert!(summary.contains("# entropy target sigma parameter:   50.000000 bits\n"));
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.starts_with("HMMER3/f"));
    assert!(hmm.contains("EFFN  "));
}

#[test]
fn hmmbuild_supports_calibration_tuning_options() {
    let dir = tempfile::tempdir().unwrap();
    let tuned_hmm = dir.path().join("cal-tuned.hmm");
    let default_hmm = dir.path().join("cal-default.hmm");

    let tuned = Command::new(hmmer())
        .args([
            "build",
            "--EmL",
            "80",
            "--EmN",
            "30",
            "--EvL",
            "90",
            "--EvN",
            "30",
            "--EfL",
            "70",
            "--EfN",
            "30",
            "--Eft",
            "0.07",
            tuned_hmm.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();
    assert!(
        tuned.status.success(),
        "{}",
        String::from_utf8_lossy(&tuned.stderr)
    );
    let summary = String::from_utf8(tuned.stdout).unwrap();
    assert!(summary.contains("# seq length for MSV Gumbel mu fit: 80\n"));
    assert!(summary.contains("# seq number for MSV Gumbel mu fit: 30\n"));
    assert!(summary.contains("# seq length for Vit Gumbel mu fit: 90\n"));
    assert!(summary.contains("# seq number for Vit Gumbel mu fit: 30\n"));
    assert!(summary.contains("# seq length for Fwd exp tau fit:   70\n"));
    assert!(summary.contains("# seq number for Fwd exp tau fit:   30\n"));
    assert!(summary.contains("# tail mass for Fwd exp tau fit:    0.070000\n"));

    let default = Command::new(hmmer())
        .args([
            "build",
            default_hmm.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    let tuned_hmm = std::fs::read_to_string(tuned_hmm).unwrap();
    let default_hmm = std::fs::read_to_string(default_hmm).unwrap();
    assert!(tuned_hmm.contains("STATS LOCAL MSV"));
    assert!(tuned_hmm.contains("STATS LOCAL FORWARD"));
    let (msv_mu, msv_lambda) = hmm_stat_values(&tuned_hmm, "STATS LOCAL MSV");
    let (_, viterbi_lambda) = hmm_stat_values(&tuned_hmm, "STATS LOCAL VITERBI");
    let (forward_mu, forward_lambda) = hmm_stat_values(&tuned_hmm, "STATS LOCAL FORWARD");
    // HMMER's p7_Calibrate (hmmer/src/evalues.c) computes a single lambda via
    // p7_Lambda and reuses it for the MSV/Viterbi Gumbel fits AND the Forward
    // exponential tail fit, so all three STATS LOCAL lines share one lambda.
    // This matches the bundled C hmmbuild output byte-for-byte. The distinctness
    // of Forward calibration lives in its MU/tau, not its lambda.
    assert_eq!(
        forward_lambda.to_bits(),
        msv_lambda.to_bits(),
        "Forward, MSV and Viterbi share the single calibration lambda (p7_Calibrate/p7_Lambda)"
    );
    assert_eq!(
        forward_lambda.to_bits(),
        viterbi_lambda.to_bits(),
        "Forward, MSV and Viterbi share the single calibration lambda (p7_Calibrate/p7_Lambda)"
    );
    // Forward is still calibrated distinctly: its fitted MU differs from MSV's.
    assert_ne!(
        forward_mu.to_bits(),
        msv_mu.to_bits(),
        "Forward calibration should fit a distinct MU from the MSV Gumbel fit"
    );
    assert_ne!(
        extract_hmm_stats(&tuned_hmm),
        extract_hmm_stats(&default_hmm),
        "calibration tuning options should affect serialized STATS"
    );
}

#[test]
fn hmmbuild_accepts_window_and_maxinsert_options() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("dna-window.sto");
    let hmm_out = dir.path().join("dna-window.hmm");
    let default_hmm_out = dir.path().join("dna-window-default.hmm");
    let insert = "C".repeat(20);
    let insert_rf = ".".repeat(20);
    std::fs::write(
        &sto,
        format!(
            "# STOCKHOLM 1.0\n#=GF ID dna_window\n#=GC RF x{insert_rf}x\ns1 A{insert}G\ns2 A{insert}G\n//\n"
        ),
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--dna",
            "--hand",
            "--w_beta",
            "0.5",
            "--w_length",
            "12",
            "--maxinsertlen",
            "5",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = String::from_utf8(output.stdout).unwrap();
    assert!(summary.contains("# window length tail mass:          0.5 bits\n"));
    assert!(summary.contains("# window length :                   12\n"));
    assert!(summary.contains("# max insert length:                5\n"));

    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.contains("MAXL  12\n"), "{hmm}");

    let default = Command::new(hmmer())
        .args([
            "build",
            "--dna",
            "--hand",
            "--w_length",
            "12",
            default_hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    assert_ne!(hmm, std::fs::read_to_string(default_hmm_out).unwrap());
}

#[test]
fn hmmbuild_singlemx_builds_one_sequence_model_with_gap_options() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("single.sto");
    let hmm_out = dir.path().join("single.hmm");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID single_query\nq1 ACDEFGHIKL--MNPQRSTVWY\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--singlemx",
            "--popen",
            "0.03",
            "--pextend",
            "0.2",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = String::from_utf8(output.stdout).unwrap();
    assert!(summary.contains("# single sequence builder:          substitution matrix\n"));
    assert!(summary.contains("# substitution score matrix:        BLOSUM62\n"));
    assert!(summary.contains("# gap open probability:             0.030000\n"));
    assert!(summary.contains("# gap extend probability:           0.200000\n"));

    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.contains("NAME  single_query\n"));
    assert!(hmm.contains("LENG  20\n"));
    assert!(hmm.contains("NSEQ  1\n"));
    assert!(hmm.contains("EFFN  1.000000\n"));
    assert!(hmm.contains("CONS  yes\n"));
}

#[test]
fn hmmbuild_singlemx_supports_dna_and_rna_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let cases = [
        (
            "--dna",
            b"# STOCKHOLM 1.0\n#=GF ID dna_single\nq1 ACGTACGT--NN\n//\n".as_slice(),
            hmmer_pure_rs::alphabet::AlphabetType::Dna,
        ),
        (
            "--rna",
            b"# STOCKHOLM 1.0\n#=GF ID rna_single\nq1 ACGUACGU--NN\n//\n".as_slice(),
            hmmer_pure_rs::alphabet::AlphabetType::Rna,
        ),
    ];

    for (alphabet_flag, sto_text, expected_alphabet) in cases {
        let sto = dir.path().join(format!(
            "single-{}.sto",
            alphabet_flag.trim_start_matches("--")
        ));
        let hmm_out = dir.path().join(format!(
            "single-{}.hmm",
            alphabet_flag.trim_start_matches("--")
        ));
        std::fs::write(&sto, sto_text).unwrap();

        let output = Command::new(hmmer())
            .args([
                "build",
                alphabet_flag,
                "--singlemx",
                "--EmL",
                "20",
                "--EmN",
                "5",
                "--EvL",
                "20",
                "--EvN",
                "5",
                "--EfL",
                "20",
                "--EfN",
                "5",
                hmm_out.to_str().unwrap(),
                sto.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let summary = String::from_utf8(output.stdout).unwrap();
        assert!(summary.contains("# single sequence builder:          substitution matrix\n"));
        assert!(summary.contains("# substitution score matrix:        DNA1\n"));

        let hmm = hmmer_pure_rs::hmmfile::read_hmm_file_auto(&hmm_out)
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(hmm.abc_type, expected_alphabet);
        assert_eq!(hmm.m, 10);
        assert_eq!(hmm.nseq, 1);
        assert!((hmm.eff_nseq - 1.0).abs() < 1e-6);

        use hmmer_pure_rs::hmm::{DD, DM, II, IM, MD, MI, MM};
        let eps = 1e-4;
        assert!((hmm.t[0][MM] - 0.9375).abs() < eps);
        assert!((hmm.t[0][MI] - 0.03125).abs() < eps);
        assert!((hmm.t[0][MD] - 0.03125).abs() < eps);
        assert!((hmm.t[0][IM] - 0.25).abs() < eps);
        assert!((hmm.t[0][II] - 0.75).abs() < eps);
        assert!((hmm.t[0][DM] - 0.25).abs() < eps);
        assert!((hmm.t[0][DD] - 0.75).abs() < eps);
        assert!((hmm.t[hmm.m][MM] - 0.96875).abs() < eps);
        assert_eq!(hmm.t[hmm.m][MD], 0.0);
        assert_eq!(hmm.t[hmm.m][DM], 1.0);
        assert_eq!(hmm.t[hmm.m][DD], 0.0);
    }

    let sto = dir.path().join("single-dna-custom.sto");
    let hmm_out = dir.path().join("single-dna-custom.hmm");
    let mxfile = dir.path().join("dna-custom.mx");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID dna_custom\nq1 ACGTACGT\n//\n",
    )
    .unwrap();
    std::fs::write(
        &mxfile,
        b"A C G T\nA 5 -4 -4 -4\nC -4 5 -4 -4\nG -4 -4 5 -4\nT -4 -4 -4 5\n",
    )
    .unwrap();
    let custom = Command::new(hmmer())
        .args([
            "build",
            "--dna",
            "--singlemx",
            "--mxfile",
            mxfile.to_str().unwrap(),
            "--EmL",
            "20",
            "--EmN",
            "5",
            "--EvL",
            "20",
            "--EvN",
            "5",
            "--EfL",
            "20",
            "--EfN",
            "5",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        custom.status.success(),
        "{}",
        String::from_utf8_lossy(&custom.stderr)
    );
    let summary = String::from_utf8(custom.stdout).unwrap();
    assert!(summary.contains(&format!(
        "# substitution score matrix:        {}\n",
        mxfile.display()
    )));
}

#[test]
fn hmmbuild_applies_fixed_gap_options_after_normal_msa_builds() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("fixed-gaps.sto");
    let hmm_out = dir.path().join("fixed-gaps.hmm");
    let default_hmm_out = dir.path().join("fixed-gaps-default.hmm");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID fixed_gaps\ns1 ACDEFGHIKLMNPQRSTVWY\ns2 ACDEYGHIKLMNPQRSTVWY\n//\n",
    )
    .unwrap();
    let output = Command::new(hmmer())
        .args([
            "build",
            "--amino",
            "--popen",
            "0.03",
            "--pextend",
            "0.2",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = String::from_utf8(output.stdout).unwrap();
    assert!(summary.contains("# gap open probability:             0.030000\n"));
    assert!(summary.contains("# gap extend probability:           0.200000\n"));

    let hmm = hmmer_pure_rs::hmmfile::read_hmm_file_auto(&hmm_out)
        .unwrap()
        .pop()
        .unwrap();
    use hmmer_pure_rs::hmm::{DD, DM, II, IM, MD, MI, MM};
    let eps = 1e-4;
    for node in 0..hmm.m {
        assert!((hmm.t[node][MM] - 0.94).abs() < eps);
        assert!((hmm.t[node][MI] - 0.03).abs() < eps);
        assert!((hmm.t[node][MD] - 0.03).abs() < eps);
        assert!((hmm.t[node][IM] - 0.8).abs() < eps);
        assert!((hmm.t[node][II] - 0.2).abs() < eps);
        assert!((hmm.t[node][DM] - 0.8).abs() < eps);
        assert!((hmm.t[node][DD] - 0.2).abs() < eps);
    }
    assert!((hmm.t[hmm.m][MM] - 0.97).abs() < eps);
    assert_eq!(hmm.t[hmm.m][MD], 0.0);
    assert_eq!(hmm.t[hmm.m][DM], 1.0);
    assert_eq!(hmm.t[hmm.m][DD], 0.0);

    let default = Command::new(hmmer())
        .args([
            "build",
            "--amino",
            "--seed",
            "42",
            default_hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    let fixed_text = std::fs::read_to_string(hmm_out).unwrap();
    let default_text = std::fs::read_to_string(default_hmm_out).unwrap();
    assert_ne!(
        extract_hmm_stats(&fixed_text),
        extract_hmm_stats(&default_text),
        "fixed gap probabilities must recalibrate serialized STATS"
    );
}

#[test]
fn hmmbuild_singlemx_accepts_builtin_and_custom_score_matrices() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("single.sto");
    let pam_hmm = dir.path().join("pam30.hmm");
    let custom_hmm = dir.path().join("custom.hmm");
    let mxfile = dir.path().join("custom.mx");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID single\nquery ACDEFGHIKLMNPQRSTVWY\n//\n",
    )
    .unwrap();
    std::fs::write(&mxfile, custom_protein_score_matrix()).unwrap();

    let pam = Command::new(hmmer())
        .args([
            "build",
            "--amino",
            "--singlemx",
            "--mx",
            "PAM30",
            pam_hmm.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        pam.status.success(),
        "{}",
        String::from_utf8_lossy(&pam.stderr)
    );
    let pam_summary = String::from_utf8(pam.stdout).unwrap();
    assert!(pam_summary.contains("# substitution score matrix:        PAM30\n"));

    let custom = Command::new(hmmer())
        .args([
            "build",
            "--amino",
            "--singlemx",
            "--mxfile",
            mxfile.to_str().unwrap(),
            custom_hmm.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        custom.status.success(),
        "{}",
        String::from_utf8_lossy(&custom.stderr)
    );
    let custom_summary = String::from_utf8(custom.stdout).unwrap();
    assert!(custom_summary.contains(&format!(
        "# substitution score matrix:        {}\n",
        mxfile.display()
    )));
    assert_ne!(
        std::fs::read_to_string(pam_hmm).unwrap(),
        std::fs::read_to_string(custom_hmm).unwrap()
    );
}

#[test]
fn hmmbuild_accepts_cpu_option() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_out = dir.path().join("cpu.hmm");
    let output = Command::new(hmmer())
        .args([
            "build",
            "--cpu",
            "1",
            hmm_out.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = String::from_utf8(output.stdout).unwrap();
    assert!(summary.contains("# number of worker threads:         1\n"));
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.starts_with("HMMER3/f"));
    assert!(hmm.contains("LENG  149\n"));
}

#[test]
fn hmmbuild_preserves_stockholm_gf_accession_and_description() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("meta.sto");
    let hmm_out = dir.path().join("meta.hmm");
    std::fs::write(
        &sto,
        "# STOCKHOLM 1.0\n#=GF ID meta\n#=GF AC PF99999.1\n#=GF DE first line\n#=GF DE second line\ns1 ACDEFGHIKLMNPQRSTVWY\ns2 ACDEFGHIKLMNPQRSTVWY\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--amino",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.contains("ACC   PF99999.1\n"));
    assert!(hmm.contains("DESC  first line second line\n"));
}

#[test]
fn alimask_accepts_stockholm_stdin_with_informat() {
    for informat in ["stockholm", "sto"] {
        let dir = tempfile::tempdir().unwrap();
        let masked = dir.path().join("masked.sto");
        let sto = std::fs::read("hmmer/testsuite/20aa.sto").unwrap();

        let output = run_with_stdin(
            &[
                "alimask",
                "--informat",
                informat,
                "--alirange",
                "1..20",
                "-",
                masked.to_str().unwrap(),
            ],
            &sto,
        );
        assert!(
            output.status.success(),
            "{informat} stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let masked = std::fs::read_to_string(masked).unwrap();
        assert!(masked.contains("#=GC MM mmmmmmmmmmmmmmmmmmmm"));
    }
}

#[test]
fn alimask_alirange_writes_alignment_length_model_mask() {
    let dir = tempfile::tempdir().unwrap();
    let masked = dir.path().join("masked.sto");

    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--seed",
            "7",
            "--alirange",
            "2-4,7..7",
            "hmmer/testsuite/20aa.sto",
            masked.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# alimask :: append modelmask line to a multiple sequence alignment\n")
    );
    assert!(stdout.contains("# input alignment file:             hmmer/testsuite/20aa.sto\n"));
    assert!(stdout.contains("# alignment range:                  2-4,7..7\n"));
    assert!(stdout.contains("# random number seed set to:        7\n"));
    let masked = std::fs::read_to_string(masked).unwrap();
    assert!(masked.contains("#=GC MM .mmm..m............."));
}

#[test]
fn alimask_report_stdout_stderr_and_o_match_c_shape() {
    let dir = tempfile::tempdir().unwrap();
    let rust_masked = dir.path().join("rust-masked.sto");
    let c_masked = dir.path().join("c-masked.sto");

    let rust = Command::new(hmmer())
        .args([
            "alimask",
            "--alirange",
            "2-4",
            "hmmer/testsuite/20aa.sto",
            rust_masked.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        rust.status.success(),
        "{}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let c = Command::new(c_alimask())
        .args([
            "--alirange",
            "2-4",
            "hmmer/testsuite/20aa.sto",
            c_masked.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));

    assert!(rust.stderr.is_empty());
    assert!(c.stderr.is_empty());
    let rust_stdout = String::from_utf8(rust.stdout)
        .unwrap()
        .replace(rust_masked.to_str().unwrap(), "<postmsa>");
    let c_stdout = String::from_utf8(c.stdout)
        .unwrap()
        .replace(c_masked.to_str().unwrap(), "<postmsa>");
    assert_eq!(rust_stdout, c_stdout);

    let summary = dir.path().join("summary.txt");
    let masked = dir.path().join("masked-o.sto");
    let output = Command::new(hmmer())
        .args([
            "alimask",
            "-o",
            summary.to_str().unwrap(),
            "--alirange",
            "2-4",
            "hmmer/testsuite/20aa.sto",
            masked.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
    let summary = std::fs::read_to_string(summary).unwrap();
    assert!(summary.contains("# alignment range:                  2-4\n"));
    assert!(summary.contains("# output directed to file:          "));
}

#[test]
fn alimask_modelrange_maps_rf_model_columns_to_alignment_columns() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("rf.sto");
    let masked = dir.path().join("masked.sto");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID rfcase\ns1 ACDEFG\ns2 AC-EFG\n#=GC RF xx..xx\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--hand",
            "--modelrange",
            "2-3",
            sto.to_str().unwrap(),
            masked.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let masked = std::fs::read_to_string(masked).unwrap();
    assert!(masked.contains("#=GC RF xx..xx"));
    assert!(masked.contains("#=GC MM .mmmm."));
}

#[test]
fn alimask_model2ali_and_ali2model_print_coordinate_maps() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("rf.sto");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID rfcase\ns1 ACDEFG\ns2 AC-EFG\n#=GC RF xx..xx\n//\n",
    )
    .unwrap();

    let model2ali = Command::new(hmmer())
        .args([
            "alimask",
            "--hand",
            "--model2ali",
            "2..3",
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        model2ali.status.success(),
        "{}",
        String::from_utf8_lossy(&model2ali.stderr)
    );
    let stdout = String::from_utf8(model2ali.stdout).unwrap();
    assert!(stdout.contains("model coordinates     alignment coordinates\n"));
    assert!(stdout.contains("       2..3        ->        2..5       \n"));

    let ali2model = Command::new(hmmer())
        .args([
            "alimask",
            "--hand",
            "--ali2model",
            "3..4",
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        ali2model.status.success(),
        "{}",
        String::from_utf8_lossy(&ali2model.stderr)
    );
    let stdout = String::from_utf8(ali2model.stdout).unwrap();
    assert!(stdout.contains("alignment coordinates     model coordinates\n"));
    assert!(stdout.contains("          3..4        ->       -..-  (no map)\n"));
}

#[test]
fn alimask_model_maps_use_selected_relative_weighting() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("weighted.sto");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID weighted\ns1 AAAA\ns2 AAAA\ns3 CCC-\n//\n",
    )
    .unwrap();

    let pb = Command::new(hmmer())
        .args([
            "alimask",
            "--dna",
            "--symfrac",
            "0.56",
            "--ali2model",
            "4..4",
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        pb.status.success(),
        "{}",
        String::from_utf8_lossy(&pb.stderr)
    );
    let stdout = String::from_utf8(pb.stdout).unwrap();
    assert!(stdout.contains("          4..4        ->       -..-  (no map)\n"));

    let unweighted = Command::new(hmmer())
        .args([
            "alimask",
            "--dna",
            "--wnone",
            "--symfrac",
            "0.56",
            "--ali2model",
            "4..4",
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        unweighted.status.success(),
        "{}",
        String::from_utf8_lossy(&unweighted.stderr)
    );
    let stdout = String::from_utf8(unweighted.stdout).unwrap();
    assert!(stdout.contains("          4..4        ->        4..4       \n"));

    let explicit_pb = Command::new(hmmer())
        .args([
            "alimask",
            "--dna",
            "--wpb",
            "--symfrac",
            "0.56",
            "--ali2model",
            "4..4",
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        explicit_pb.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit_pb.stderr)
    );
    let stdout = String::from_utf8(explicit_pb.stdout).unwrap();
    assert!(stdout.contains("          4..4        ->       -..-  (no map)\n"));

    let masked = dir.path().join("weighted-mask.sto");
    let unweighted_modelrange = Command::new(hmmer())
        .args([
            "alimask",
            "--dna",
            "--wnone",
            "--symfrac",
            "0.56",
            "--modelrange",
            "4..4",
            sto.to_str().unwrap(),
            masked.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        unweighted_modelrange.status.success(),
        "{}",
        String::from_utf8_lossy(&unweighted_modelrange.stderr)
    );
    let masked = std::fs::read_to_string(masked).unwrap();
    assert!(masked.contains("#=GC MM ...m\n"));
}

#[test]
fn alimask_accepts_c_relative_weighting_options_for_model_maps() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("weighted.sto");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID weighted\n#=GS s1 WT 1.0\n#=GS s2 WT 1.0\n#=GS s3 WT 0.1\ns1 AAA-\ns2 AAA-\ns3 CCCC\n//\n",
    )
    .unwrap();

    let wgsc = Command::new(hmmer())
        .args([
            "alimask",
            "--dna",
            "--wgsc",
            "--ali2model",
            "1-4",
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        wgsc.status.success(),
        "{}",
        String::from_utf8_lossy(&wgsc.stderr)
    );
    let stdout = String::from_utf8(wgsc.stdout).unwrap();
    assert!(stdout.contains("# relative weighting scheme:        G/S/C\n"));
    assert!(stdout.contains("alignment coordinates     model coordinates\n"));

    let wnone = Command::new(hmmer())
        .args([
            "alimask",
            "--dna",
            "--wnone",
            "--symfrac",
            "0.2",
            "--ali2model",
            "4-4",
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        wnone.status.success(),
        "{}",
        String::from_utf8_lossy(&wnone.stderr)
    );
    let stdout = String::from_utf8(wnone.stdout).unwrap();
    assert!(stdout.contains("          4..4        ->        4..4       \n"));

    let wgiven = Command::new(hmmer())
        .args([
            "alimask",
            "--dna",
            "--wgiven",
            "--symfrac",
            "0.2",
            "--ali2model",
            "4-4",
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        wgiven.status.success(),
        "{}",
        String::from_utf8_lossy(&wgiven.stderr)
    );
    let stdout = String::from_utf8(wgiven.stdout).unwrap();
    assert!(stdout.contains("# relative weighting scheme:        given\n"));
    assert!(stdout.contains("          4..4        ->       -..-  (no map)\n"));

    let wblosum = Command::new(hmmer())
        .args([
            "alimask",
            "--dna",
            "--wblosum",
            "--wid",
            "0.8",
            "--symfrac",
            "0.4",
            "--ali2model",
            "4-4",
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        wblosum.status.success(),
        "{}",
        String::from_utf8_lossy(&wblosum.stderr)
    );
    let stdout = String::from_utf8(wblosum.stdout).unwrap();
    assert!(stdout.contains("# relative weighting scheme:        BLOSUM filter\n"));
    assert!(stdout.contains("# frac id cutoff for BLOSUM wgts:   0.800000\n"));
    assert!(stdout.contains("          4..4        ->        4..4       \n"));
}

#[test]
fn alimask_accepts_alphabet_assertion_and_fragthresh_for_coordinate_maps() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("fragments.sto");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID fragments\ns1 --A---\ns2 CCA---\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--amino",
            "--fragthresh",
            "0.5",
            "--symfrac",
            "0.75",
            "--ali2model",
            "1..1",
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("alignment coordinates     model coordinates\n"));
    assert!(stdout.contains("          1..1        ->        1..1       \n"));
}

#[test]
fn alimask_modelmask_writes_standalone_model_mask() {
    let dir = tempfile::tempdir().unwrap();
    let masked = dir.path().join("masked.sto");

    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--modelmask",
            "m.m.m.m.m.m.m.m.m.m.",
            "hmmer/testsuite/20aa.sto",
            masked.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let masked = std::fs::read_to_string(masked).unwrap();
    assert!(masked.contains("#=GC MM m.m.m.m.m.m.m.m.m.m."));
}

#[test]
fn alimask_appendmask_merges_alignment_range_and_model_mask() {
    let dir = tempfile::tempdir().unwrap();
    let masked = dir.path().join("masked.sto");

    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--appendmask",
            "--alirange",
            "2..3",
            "--modelmask",
            "m...................",
            "hmmer/testsuite/20aa.sto",
            masked.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let masked = std::fs::read_to_string(masked).unwrap();
    assert!(masked.contains("#=GC MM mmm................."));
}

#[test]
fn alimask_preserves_stockholm_metadata_while_replacing_mask() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("meta.sto");
    let masked = dir.path().join("masked.sto");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID meta\n#=GF AC PF00001.1\n#=GF DE first line\n#=GF XX unknown gf\n#=GS s1 DE parsed seq desc\n#=GS s1 OS unknown gs\ns1 ACDEFG\n#=GR s1 PP 999999\n#=GR s1 SA unknown\n#=GC RF xxxxxx\n#=GC PP_cons 888888\n#=GC SS_cons <<<<<<\n#=GC MM ......\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--alirange",
            "2..4",
            sto.to_str().unwrap(),
            masked.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let masked = std::fs::read_to_string(masked).unwrap();
    assert!(masked.contains("#=GF ID meta\n"));
    assert!(masked.contains("#=GF AC PF00001.1\n"));
    assert!(masked.contains("#=GF DE first line\n"));
    assert!(masked.contains("#=GF XX unknown gf\n"));
    assert!(masked.contains("#=GS s1 DE parsed seq desc\n"));
    assert!(masked.contains("#=GS s1 OS unknown gs\n"));
    assert!(masked.contains("#=GR s1 PP 999999\n"));
    assert!(masked.contains("#=GR s1 SA unknown\n"));
    assert!(masked.contains("#=GC RF xxxxxx\n"));
    assert!(masked.contains("#=GC PP_cons 888888\n"));
    assert!(masked.contains("#=GC SS_cons <<<<<<\n"));
    assert!(masked.contains("#=GC MM .mmm..\n"));
    assert!(!masked.contains("#=GC MM ......\n"));
}

#[test]
fn upstream_command_aliases_smoke_test() {
    let stat = Command::new(hmmer())
        .args(["hmmstat", "hmmer/testsuite/20aa.hmm"])
        .output()
        .unwrap();
    assert!(
        stat.status.success(),
        "{}",
        String::from_utf8_lossy(&stat.stderr)
    );
    assert!(String::from_utf8(stat.stdout)
        .unwrap()
        .contains("# hmmstat ::"));

    let emit = Command::new(hmmer())
        .args(["hmmemit", "-c", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    assert!(
        emit.status.success(),
        "{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    assert!(String::from_utf8(emit.stdout)
        .unwrap()
        .starts_with(">fn3-consensus\n"));

    let logo = Command::new(hmmer())
        .args(["hmmlogo", "--no_indel", "hmmer/tutorial/fn3.hmm"])
        .output()
        .unwrap();
    assert!(
        logo.status.success(),
        "{}",
        String::from_utf8_lossy(&logo.stderr)
    );
    assert!(String::from_utf8(logo.stdout)
        .unwrap()
        .contains("Residue heights\n"));

    let search = Command::new(hmmer())
        .args([
            "hmmsearch",
            "--noali",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();
    assert!(
        search.status.success(),
        "{}",
        String::from_utf8_lossy(&search.stderr)
    );
    assert!(String::from_utf8(search.stdout)
        .unwrap()
        .contains("Scores for complete sequences"));
}

fn hmmpgmd_stats_u64(payload: &[u8], field_offset: usize) -> u64 {
    u64::from_be_bytes(payload[field_offset..field_offset + 8].try_into().unwrap())
}

fn hmmpgmd_first_hit_name(payload: &[u8]) -> String {
    let nhits = hmmpgmd_stats_u64(payload, 90);
    assert!(nhits > 0, "expected at least one serialized hit");
    let first_offset = hmmpgmd_stats_u64(payload, 114) as usize;
    assert_eq!(first_offset, 0);
    let stats_size = 114 + nhits as usize * 8;
    let hit_start = stats_size + first_offset;
    let hit_size =
        u32::from_be_bytes(payload[hit_start..hit_start + 4].try_into().unwrap()) as usize;
    assert!(hit_size >= 110, "serialized P7_HIT shell is too short");
    assert!(
        hit_start + hit_size <= payload.len(),
        "serialized P7_HIT extends past payload"
    );
    let name_start = hit_start + 109;
    let name_len = payload[name_start..hit_start + hit_size]
        .iter()
        .position(|&b| b == 0)
        .unwrap();
    String::from_utf8(payload[name_start..name_start + name_len].to_vec()).unwrap()
}

fn hmmpgmd_hit_names(payload: &[u8]) -> Vec<String> {
    let nhits = hmmpgmd_stats_u64(payload, 90) as usize;
    let stats_size = 114 + nhits * 8;
    let mut names = Vec::with_capacity(nhits);
    for i in 0..nhits {
        let offset = hmmpgmd_stats_u64(payload, 114 + i * 8) as usize;
        let hit_start = stats_size + offset;
        let hit_size =
            u32::from_be_bytes(payload[hit_start..hit_start + 4].try_into().unwrap()) as usize;
        let name_start = hit_start + 109;
        let name_len = payload[name_start..hit_start + hit_size]
            .iter()
            .position(|&b| b == 0)
            .unwrap();
        names.push(String::from_utf8(payload[name_start..name_start + name_len].to_vec()).unwrap());
    }
    names
}

fn hmmpgmd_first_hit_domain_summary(payload: &[u8]) -> (i32, i64, i64, u32) {
    let nhits = hmmpgmd_stats_u64(payload, 90);
    assert!(nhits > 0, "expected at least one serialized hit");
    let first_offset = hmmpgmd_stats_u64(payload, 114) as usize;
    let stats_size = 114 + nhits as usize * 8;
    let hit_start = stats_size + first_offset;
    let hit_size =
        u32::from_be_bytes(payload[hit_start..hit_start + 4].try_into().unwrap()) as usize;
    let ndom = i32::from_be_bytes(payload[hit_start + 72..hit_start + 76].try_into().unwrap());
    assert!(ndom > 0, "expected serialized domain payloads");
    let domain_start = hit_start + hit_size;
    assert!(
        domain_start + 92 <= payload.len(),
        "serialized P7_DOMAIN shell extends past payload"
    );
    let domain_size =
        u32::from_be_bytes(payload[domain_start..domain_start + 4].try_into().unwrap());
    let ienv = i64::from_be_bytes(
        payload[domain_start + 4..domain_start + 12]
            .try_into()
            .unwrap(),
    );
    let jenv = i64::from_be_bytes(
        payload[domain_start + 12..domain_start + 20]
            .try_into()
            .unwrap(),
    );
    let ad_start = domain_start + domain_size as usize;
    assert!(
        ad_start + 45 <= payload.len(),
        "serialized P7_ALIDISPLAY shell extends past payload"
    );
    let alidisplay_size = u32::from_be_bytes(payload[ad_start..ad_start + 4].try_into().unwrap());
    assert!(
        ad_start + alidisplay_size as usize <= payload.len(),
        "serialized P7_ALIDISPLAY extends past payload"
    );
    (ndom, ienv, jenv, alidisplay_size)
}

fn hmmpgmd_first_hit_alidisplay_payload(payload: &[u8]) -> Vec<u8> {
    let nhits = hmmpgmd_stats_u64(payload, 90);
    assert!(nhits > 0, "expected at least one serialized hit");
    let first_offset = hmmpgmd_stats_u64(payload, 114) as usize;
    let stats_size = 114 + nhits as usize * 8;
    let hit_start = stats_size + first_offset;
    let hit_size =
        u32::from_be_bytes(payload[hit_start..hit_start + 4].try_into().unwrap()) as usize;
    let domain_start = hit_start + hit_size;
    let domain_size =
        u32::from_be_bytes(payload[domain_start..domain_start + 4].try_into().unwrap()) as usize;
    let ad_start = domain_start + domain_size;
    let ad_size = u32::from_be_bytes(payload[ad_start..ad_start + 4].try_into().unwrap()) as usize;
    payload[ad_start..ad_start + ad_size].to_vec()
}

fn hmmpgmd_first_hit_alidisplay_metadata(
    payload: &[u8],
) -> (i32, i64, String, String, String, String) {
    let nhits = hmmpgmd_stats_u64(payload, 90);
    assert!(nhits > 0, "expected at least one serialized hit");
    let first_offset = hmmpgmd_stats_u64(payload, 114) as usize;
    let stats_size = 114 + nhits as usize * 8;
    let hit_start = stats_size + first_offset;
    let hit_size =
        u32::from_be_bytes(payload[hit_start..hit_start + 4].try_into().unwrap()) as usize;
    let domain_start = hit_start + hit_size;
    let domain_size =
        u32::from_be_bytes(payload[domain_start..domain_start + 4].try_into().unwrap()) as usize;
    let ad_start = domain_start + domain_size;
    let ad_size = u32::from_be_bytes(payload[ad_start..ad_start + 4].try_into().unwrap()) as usize;
    let n = i32::from_be_bytes(payload[ad_start + 4..ad_start + 8].try_into().unwrap());
    let model_len = i32::from_be_bytes(payload[ad_start + 16..ad_start + 20].try_into().unwrap());
    let seq_len = i64::from_be_bytes(payload[ad_start + 36..ad_start + 44].try_into().unwrap());
    let presence = payload[ad_start + 44];
    let mut pos = ad_start + 45;
    let fixed = n as usize + 1;
    if presence & 0x01 != 0 {
        pos += fixed;
    }
    if presence & 0x02 != 0 {
        pos += fixed;
    }
    if presence & 0x04 != 0 {
        pos += fixed;
    }
    pos += fixed; // model
    pos += fixed; // mline
    if presence & 0x10 != 0 {
        pos += fixed;
    }
    if presence & 0x20 != 0 {
        pos += n as usize * 3 + 1;
    }
    if presence & 0x08 != 0 {
        pos += fixed;
    }
    assert!(
        pos < ad_start + ad_size,
        "missing serialized ALIDISPLAY metadata strings"
    );

    fn take_c_string(payload: &[u8], pos: &mut usize) -> String {
        let end = *pos
            + payload[*pos..]
                .iter()
                .position(|&b| b == 0)
                .expect("unterminated C string");
        let value = String::from_utf8(payload[*pos..end].to_vec()).unwrap();
        *pos = end + 1;
        value
    }

    let hmm_name = take_c_string(payload, &mut pos);
    let hmm_acc = take_c_string(payload, &mut pos);
    let _hmm_desc = take_c_string(payload, &mut pos);
    let seq_name = take_c_string(payload, &mut pos);
    let seq_acc = take_c_string(payload, &mut pos);
    let _seq_desc = take_c_string(payload, &mut pos);

    (model_len, seq_len, hmm_name, hmm_acc, seq_name, seq_acc)
}

fn hmmpgmd_first_hit_score(payload: &[u8]) -> f32 {
    let nhits = hmmpgmd_stats_u64(payload, 90);
    let first_offset = hmmpgmd_stats_u64(payload, 114) as usize;
    let stats_size = 114 + nhits as usize * 8;
    let hit_start = stats_size + first_offset;
    f32::from_bits(u32::from_be_bytes(
        payload[hit_start + 16..hit_start + 20].try_into().unwrap(),
    ))
}

fn hmmpgmd_first_hit_ndom(payload: &[u8]) -> i32 {
    let nhits = hmmpgmd_stats_u64(payload, 90);
    assert!(nhits > 0, "expected at least one serialized hit");
    let first_offset = hmmpgmd_stats_u64(payload, 114) as usize;
    let stats_size = 114 + nhits as usize * 8;
    let hit_start = stats_size + first_offset;
    i32::from_be_bytes(payload[hit_start + 72..hit_start + 76].try_into().unwrap())
}

fn hmmpgmd_first_hit_nreported(payload: &[u8]) -> i32 {
    let nhits = hmmpgmd_stats_u64(payload, 90);
    assert!(nhits > 0, "expected at least one serialized hit");
    let first_offset = hmmpgmd_stats_u64(payload, 114) as usize;
    let stats_size = 114 + nhits as usize * 8;
    let hit_start = stats_size + first_offset;
    i32::from_be_bytes(payload[hit_start + 80..hit_start + 84].try_into().unwrap())
}

fn hmmpgmd_query_rust_master_hmmdb(hmmdb: &str) -> Vec<u8> {
    let listener = bind_hmmpgmd_listener();
    let cport = listener.local_addr().unwrap().port();
    drop(listener);

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(worker_listener);

    let mut child = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--hmmdb",
            hmmdb,
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let payload = match hmmpgmd_send_hmmdb_sequence_query(cport) {
        Ok(payload) => payload,
        Err(err) => {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "Rust hmmpgmd did not return a framed HMMDB payload: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    let _ = child.kill();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success() || output.status.code().is_none(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    payload
}

fn hmmpgmd_query_c_master_worker_hmmdb(hmmdb: &std::path::Path) -> Vec<u8> {
    let cport = reserve_c_hmmpgmd_port(50000);
    let wport = reserve_c_hmmpgmd_port(cport.saturating_add(1));
    let mut master = Command::new(c_hmmpgmd())
        .args([
            "--master",
            "--hmmdb",
            hmmdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut worker = None;
    let payload = (|| -> std::io::Result<Vec<u8>> {
        let _ = connect_with_deadline(cport, Duration::from_secs(5))?;
        worker = Some(
            Command::new(c_hmmpgmd())
                .args([
                    "--worker",
                    "127.0.0.1",
                    "--wport",
                    &wport.to_string(),
                    "--cpu",
                    "1",
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        std::thread::sleep(Duration::from_secs(2));
        hmmpgmd_send_hmmdb_sequence_query(cport)
    })();

    if let Some(worker) = worker.as_mut() {
        let _ = worker.kill();
    }
    let _ = master.kill();
    let worker_output = worker.map(|worker| worker.wait_with_output().unwrap());
    let master_output = master.wait_with_output().unwrap();

    payload.unwrap_or_else(|err| {
        panic!(
            "C hmmpgmd did not return a framed HMMDB payload: {err}; master stdout={}; master stderr={}; worker stdout={}; worker stderr={}",
            String::from_utf8_lossy(&master_output.stdout),
            String::from_utf8_lossy(&master_output.stderr),
            worker_output
                .as_ref()
                .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
                .unwrap_or_default(),
            worker_output
                .as_ref()
                .map(|output| String::from_utf8_lossy(&output.stderr).into_owned())
                .unwrap_or_default()
        )
    })
}

fn hmmpgmd_send_hmmdb_sequence_query(port: u16) -> std::io::Result<Vec<u8>> {
    let mut stream = connect_with_deadline(port, Duration::from_secs(5))?;
    stream.write_all(b"@--hmmdb 1\n>query\nACDEFGHIKLMNPQRSTVWY\n//\n")?;
    let mut status = [0u8; 12];
    stream.read_exact(&mut status)?;
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0u8; msg_size as usize];
    stream.read_exact(&mut payload)?;
    if code == 0 {
        Ok(payload)
    } else {
        Err(std::io::Error::other(
            String::from_utf8_lossy(&payload).into_owned(),
        ))
    }
}

fn reserve_c_hmmpgmd_port(first_port: u16) -> u16 {
    for port in first_port..65535 {
        if let Ok(listener) = TcpListener::bind(("0.0.0.0", port)) {
            drop(listener);
            return port;
        }
    }
    panic!("no high TCP port available for C hmmpgmd");
}

/// Bind a listener on a port within hmmpgmd's valid `--cport`/`--wport` range
/// (49151 < n < 65536). Binding to ephemeral port 0 can land below 49152 on
/// Linux, which hmmpgmd now rejects to match the C option range. A
/// process-global cursor advances through the range so consecutive calls (and
/// parallel tests) get distinct ports rather than all reusing the first free
/// one.
fn bind_hmmpgmd_listener() -> TcpListener {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    const LO: u16 = 49152;
    const HI: u16 = 65535; // valid range is 49151 < n < 65536, so n <= 65534
    let span = (HI - LO) as usize;
    for _ in 0..span {
        let idx = NEXT.fetch_add(1, Ordering::Relaxed) % span;
        let port = LO + idx as u16;
        if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
            return listener;
        }
    }
    panic!("no valid hmmpgmd TCP port available");
}

#[test]
fn hmmpgmd_serves_hmm_hits_over_tcp() {
    let _guard = hmmpgmd_test_guard();
    let listener = bind_hmmpgmd_listener();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let mut child = Command::new(hmmer())
        .args([
            "pgmd",
            "--hmmdb",
            "hmmer/testsuite/20aa.hmm",
            "--port",
            &port.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match connect_with_deadline(port, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "hmmpgmd did not accept a connection: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    stream.write_all(b"ACDEFGHIKLMNPQRSTVWY\n").unwrap();
    stream.shutdown(std::net::Shutdown::Write).unwrap();
    let mut response = String::new();
    let read_result = stream.read_to_string(&mut response);

    let _ = child.kill();
    let output = child.wait_with_output().unwrap();
    if let Err(err) = read_result {
        assert!(
            !response.is_empty() && err.kind() == std::io::ErrorKind::ConnectionReset,
            "hmmpgmd read failed: {err}; response={response:?}; stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert!(
        output.status.success() || output.status.code().is_none(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(response.starts_with("HITS "), "{response}");
    assert!(response.contains("test\t"), "{response}");
    assert!(response.ends_with("//\n"), "{response}");
}

#[test]
fn hmmpgmd_serves_sequence_hits_over_tcp() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    std::fs::write(
        &seqdb,
        ">target1\nACDEFGHIKLMNPQRSTVWY\n>target2\nYYYYYYYYYYYYYYYYYYYY\n",
    )
    .unwrap();

    let listener = bind_hmmpgmd_listener();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let mut child = Command::new(hmmer())
        .args([
            "pgmd",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--port",
            &port.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match connect_with_deadline(port, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "hmmpgmd --seqdb did not accept a connection: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    stream.write_all(b"ACDEFGHIKLMNPQRSTVWY\n").unwrap();
    stream.shutdown(std::net::Shutdown::Write).unwrap();
    let mut response = String::new();
    let read_result = stream.read_to_string(&mut response);

    let _ = child.kill();
    let output = child.wait_with_output().unwrap();
    if let Err(err) = read_result {
        assert!(
            !response.is_empty() && err.kind() == std::io::ErrorKind::ConnectionReset,
            "hmmpgmd --seqdb read failed: {err}; response={response:?}; stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert!(
        output.status.success() || output.status.code().is_none(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(response.starts_with("HITS "), "{response}");
    assert!(response.contains("target1\t"), "{response}");
    assert!(response.ends_with("//\n"), "{response}");
}

#[test]
fn hmmpgmd_master_serves_c_framed_client_status() {
    let _guard = hmmpgmd_test_guard();
    let listener = bind_hmmpgmd_listener();
    let cport = listener.local_addr().unwrap().port();
    drop(listener);

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(worker_listener);

    let mut child = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--hmmdb",
            "hmmer/testsuite/20aa.hmm",
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match connect_with_deadline(cport, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "hmmpgmd --master did not accept a C-framed client: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    stream
        .write_all(b"@--hmmdb 1\n>query\nACDEFGHIKLMNPQRSTVWY\n//\n")
        .unwrap();
    let mut status = [0u8; 12];
    stream.read_exact(&mut status).unwrap();
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0u8; msg_size as usize];
    stream.read_exact(&mut payload).unwrap();

    let _ = child.kill();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success() || output.status.code().is_none(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert_eq!(code, 0);
    let nhits = hmmpgmd_stats_u64(&payload, 90);
    let nreported = hmmpgmd_stats_u64(&payload, 98);
    let nincluded = hmmpgmd_stats_u64(&payload, 106);
    assert_eq!(msg_size as usize, payload.len());
    assert_eq!(nhits, 1);
    assert_eq!(nreported, 1);
    assert!(nincluded <= nhits);
    assert_eq!(hmmpgmd_first_hit_name(&payload), "test");
    assert!(hmmpgmd_first_hit_score(&payload) > 0.0);
    let (ndom, ienv, jenv, alidisplay_size) = hmmpgmd_first_hit_domain_summary(&payload);
    assert!(ndom > 0);
    assert!(ienv >= 1);
    assert!(jenv >= ienv);
    assert!(alidisplay_size > 45);
    let (model_len, seq_len, hmm_name, hmm_acc, seq_name, seq_acc) =
        hmmpgmd_first_hit_alidisplay_metadata(&payload);
    assert_eq!(model_len, 20);
    assert_eq!(seq_len, 20);
    assert_eq!(hmm_name, "000000001");
    assert_eq!(hmm_acc, "");
    assert_eq!(seq_name, "query");
    assert_eq!(seq_acc, "");
}

#[test]
fn hmmpgmd_master_applies_listener_backlog_sizing_options() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    std::fs::write(&seqdb, ">target1\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let client_listener = bind_hmmpgmd_listener();
    let cport = client_listener.local_addr().unwrap().port();
    drop(client_listener);

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(worker_listener);

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
            "--ccncts",
            "7",
            "--wcncts",
            "11",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match connect_with_deadline(cport, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = master.kill();
            let output = master.wait_with_output().unwrap();
            panic!(
                "hmmpgmd --master did not accept control connection: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };
    stream.write_all(b"!shutdown\n//\n").unwrap();
    let mut status = [0u8; 12];
    stream.read_exact(&mut status).unwrap();
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0u8; msg_size as usize];
    stream.read_exact(&mut payload).unwrap();

    let master_status = wait_for_child(&mut master, Duration::from_secs(5));
    let output = master.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        master_status
            .map(|status| status.success())
            .unwrap_or(false),
        "{stderr}"
    );
    assert_eq!(u32::from_be_bytes(status[0..4].try_into().unwrap()), 0);
    assert_eq!(payload.len(), 122);
    assert!(
        stderr.contains(&format!(
            "Listening for client connections on 0.0.0.0:{cport} (backlog 7)"
        )),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!(
            "Listening for worker connections on 0.0.0.0:{wport} (backlog 11)"
        )),
        "{stderr}"
    );
}

#[test]
fn hmmpgmd_hmmdb_alidisplay_payload_matches_c_daemon_bytes() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let c_hmmdb = dir.path().join("20aa.hmm");
    std::fs::copy("hmmer/testsuite/20aa.hmm", &c_hmmdb).unwrap();
    let press = Command::new(c_hmmpress())
        .args(["-f", c_hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}{}",
        String::from_utf8_lossy(&press.stdout),
        String::from_utf8_lossy(&press.stderr)
    );

    let rust_payload = hmmpgmd_query_rust_master_hmmdb("hmmer/testsuite/20aa.hmm");
    let c_payload = hmmpgmd_query_c_master_worker_hmmdb(&c_hmmdb);
    assert_eq!(
        hmmpgmd_first_hit_alidisplay_payload(&rust_payload),
        hmmpgmd_first_hit_alidisplay_payload(&c_payload)
    );
}

#[test]
fn hmmpgmd_master_accepts_c_framed_hmm_query_for_seqdb() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    std::fs::write(&seqdb, ">target1\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let listener = bind_hmmpgmd_listener();
    let cport = listener.local_addr().unwrap().port();
    drop(listener);

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(worker_listener);

    let mut child = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match connect_with_deadline(cport, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "hmmpgmd --master did not accept a C-framed HMM query: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    let hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    stream.write_all(b"@--seqdb 1\n").unwrap();
    stream.write_all(hmm.as_bytes()).unwrap();
    if !hmm.ends_with('\n') {
        stream.write_all(b"\n").unwrap();
    }

    let mut status = [0u8; 12];
    stream.read_exact(&mut status).unwrap();
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0u8; msg_size as usize];
    stream.read_exact(&mut payload).unwrap();

    let _ = child.kill();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success() || output.status.code().is_none(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let nmodels = u64::from_be_bytes(payload[42..50].try_into().unwrap());
    let nseqs = u64::from_be_bytes(payload[50..58].try_into().unwrap());
    let nhits = hmmpgmd_stats_u64(&payload, 90);
    let nreported = hmmpgmd_stats_u64(&payload, 98);
    let nincluded = hmmpgmd_stats_u64(&payload, 106);
    assert_eq!(code, 0);
    assert_eq!(msg_size as usize, payload.len());
    assert_eq!(nmodels, 1);
    assert_eq!(nseqs, 1);
    assert_eq!(nhits, 1);
    assert_eq!(nreported, 1);
    assert!(nincluded <= nhits);
    assert_eq!(hmmpgmd_first_hit_name(&payload), "target1");
    assert!(hmmpgmd_first_hit_score(&payload) > 0.0);
    let (ndom, ienv, jenv, alidisplay_size) = hmmpgmd_first_hit_domain_summary(&payload);
    assert!(ndom > 0);
    assert!(ienv >= 1);
    assert!(jenv >= ienv);
    assert!(alidisplay_size > 45);
    let (model_len, seq_len, hmm_name, hmm_acc, seq_name, seq_acc) =
        hmmpgmd_first_hit_alidisplay_metadata(&payload);
    assert_eq!(model_len, 20);
    assert_eq!(seq_len, 20);
    assert_eq!(hmm_name, "test");
    assert_eq!(hmm_acc, "");
    assert_eq!(seq_name, "target1");
    assert_eq!(seq_acc, "");
}

#[test]
fn hmmpgmd_worker_accepts_master_init_frame() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    std::fs::write(&seqdb, ">target1\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let client_listener = bind_hmmpgmd_listener();
    let cport = client_listener.local_addr().unwrap().port();

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(client_listener);
    drop(worker_listener);

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    if let Err(err) = connect_with_deadline(cport, Duration::from_secs(5)) {
        let _ = master.kill();
        let output = master.wait_with_output().unwrap();
        panic!(
            "hmmpgmd master did not accept client probe on {cport}: {err}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let mut worker = Command::new(hmmer())
        .args([
            "pgmd",
            "--worker",
            "127.0.0.1",
            "--cpu",
            "3",
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    std::thread::sleep(Duration::from_millis(250));
    let _ = master.kill();
    let master_output = master.wait_with_output().unwrap();
    let worker_status = wait_for_child(&mut worker, Duration::from_secs(5));
    let worker_output = worker.wait_with_output().unwrap();

    assert!(
        worker_status
            .map(|status| status.success())
            .unwrap_or(false),
        "worker failed; worker stderr={}; master stderr={}",
        String::from_utf8_lossy(&worker_output.stderr),
        String::from_utf8_lossy(&master_output.stderr)
    );
}

#[test]
fn hmmpgmd_master_schedules_c_framed_search_to_worker() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let master_seqdb = dir.path().join("master-targets.fa");
    let worker_seqdb = dir.path().join("worker-targets.fa");
    std::fs::write(&master_seqdb, ">master_only\nYYYYYYYYYYYYYYYYYYYY\n").unwrap();
    std::fs::write(&worker_seqdb, ">target1\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let client_listener = bind_hmmpgmd_listener();
    let cport = client_listener.local_addr().unwrap().port();

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(client_listener);
    drop(worker_listener);

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            master_seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let _ = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();

    let mut worker = Command::new(hmmer())
        .args([
            "pgmd",
            "--worker",
            "127.0.0.1",
            "--wport",
            &wport.to_string(),
            "--seqdb",
            worker_seqdb.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let request = format!("@--seqdb 1\n{hmm}");
    let start = Instant::now();
    let mut payload = Vec::new();
    let mut code = 1;
    while start.elapsed() < Duration::from_secs(5) {
        let mut stream = connect_with_deadline(cport, Duration::from_secs(1)).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut status = [0u8; 12];
        stream.read_exact(&mut status).unwrap();
        code = u32::from_be_bytes(status[0..4].try_into().unwrap());
        let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
        payload = vec![0u8; msg_size as usize];
        stream.read_exact(&mut payload).unwrap();
        if code == 0
            && hmmpgmd_stats_u64(&payload, 90) == 1
            && hmmpgmd_first_hit_name(&payload) == "target1"
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }

    let mut shutdown_stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    shutdown_stream.write_all(b"!shutdown\n//\n").unwrap();
    let mut shutdown_status = [0u8; 12];
    shutdown_stream.read_exact(&mut shutdown_status).unwrap();
    let shutdown_code = u32::from_be_bytes(shutdown_status[0..4].try_into().unwrap());
    let shutdown_msg_size = u64::from_be_bytes(shutdown_status[4..12].try_into().unwrap());
    let mut shutdown_payload = vec![0u8; shutdown_msg_size as usize];
    shutdown_stream.read_exact(&mut shutdown_payload).unwrap();

    let master_status = wait_for_child(&mut master, Duration::from_secs(5));
    let master_output = master.wait_with_output().unwrap();
    let worker_status = wait_for_child(&mut worker, Duration::from_secs(5));
    let worker_output = worker.wait_with_output().unwrap();

    assert!(
        master_status
            .map(|status| status.success())
            .unwrap_or(false),
        "{}",
        String::from_utf8_lossy(&master_output.stderr)
    );
    assert!(
        worker_status
            .map(|status| status.success())
            .unwrap_or(false),
        "worker failed; worker stderr={}; master stderr={}",
        String::from_utf8_lossy(&worker_output.stderr),
        String::from_utf8_lossy(&master_output.stderr)
    );
    assert_eq!(shutdown_code, 0);
    assert_eq!(shutdown_msg_size as usize, 122);
    assert_eq!(shutdown_payload.len(), 122);
    assert_eq!(
        code,
        0,
        "master returned error payload: {}",
        String::from_utf8_lossy(&payload)
    );
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 1);
    assert_eq!(hmmpgmd_first_hit_name(&payload), "target1");
    let (model_len, seq_len, hmm_name, hmm_acc, seq_name, seq_acc) =
        hmmpgmd_first_hit_alidisplay_metadata(&payload);
    assert_eq!(model_len, 20);
    assert_eq!(seq_len, 20);
    assert_eq!(hmm_name, "test");
    assert_eq!(hmm_acc, "");
    assert_eq!(seq_name, "target1");
    assert_eq!(seq_acc, "");
}

#[test]
fn hmmpgmd_master_assigns_seqdb_shards_to_workers() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("master-targets.fa");
    std::fs::write(
        &seqdb,
        ">master_a\nACDEFGHIKLMNPQRSTVWY\n>master_b\nYYYYYYYYYYYYYYYYYYYY\n",
    )
    .unwrap();

    let cport = reserve_c_hmmpgmd_port(50000);
    let wport = reserve_c_hmmpgmd_port(cport.saturating_add(1));

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let _ = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();

    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (range_tx, range_rx) = std::sync::mpsc::channel();
    let mut fake_workers = Vec::new();
    for (name, sortkey, score, nincluded) in [
        ("shard_a", 0.20_f64, 11.0_f32, 0_i32),
        ("shard_b", 0.10_f64, 10.0_f32, 1_i32),
    ] {
        let ready_tx = ready_tx.clone();
        let range_tx = range_tx.clone();
        fake_workers.push(std::thread::spawn(move || {
            let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
            let mut header = [0u8; 12];
            stream.read_exact(&mut header).unwrap();
            let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
            assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
            let mut init_body = vec![0u8; init_len];
            stream.read_exact(&mut init_body).unwrap();
            assert!(init_body.len() > 88);
            assert_eq!(u32::from_ne_bytes(init_body[72..76].try_into().unwrap()), 1);
            assert_eq!(u32::from_ne_bytes(init_body[76..80].try_into().unwrap()), 2);
            assert!(std::str::from_utf8(&init_body[88..init_body.len() - 1])
                .unwrap()
                .ends_with("master-targets.fa"));
            write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
            stream.write_all(&[0u8; 96]).unwrap();
            stream.flush().unwrap();
            ready_tx.send(()).unwrap();

            stream.read_exact(&mut header).unwrap();
            let search_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
            assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10001);
            let mut search_body = vec![0u8; search_len];
            stream.read_exact(&mut search_body).unwrap();
            assert!(search_body.len() >= 28);
            assert_eq!(u32::from_ne_bytes(search_body[0..4].try_into().unwrap()), 0);
            assert_eq!(u32::from_ne_bytes(search_body[4..8].try_into().unwrap()), 0);
            assert_eq!(
                u32::from_ne_bytes(search_body[16..20].try_into().unwrap()),
                102
            );
            assert_eq!(
                u32::from_ne_bytes(search_body[20..24].try_into().unwrap()),
                20
            );
            let opts_len = u32::from_ne_bytes(search_body[24..28].try_into().unwrap()) as usize;
            let opts = &search_body[28..28 + opts_len];
            assert_eq!(opts.last().copied(), Some(0));
            let opts_text = String::from_utf8_lossy(&opts[..opts.len() - 1]);
            let range = opts_text
                .split_whitespace()
                .skip_while(|word| *word != "--seqdb_ranges")
                .nth(1)
                .unwrap()
                .to_string();
            let query = &search_body[28 + opts_len..];
            assert!(query.len() > 296);
            assert_eq!(i32::from_ne_bytes(query[0..4].try_into().unwrap()), 20);
            assert_ne!(
                usize::from_ne_bytes(query[32..40].try_into().unwrap()),
                0,
                "serialized C P7_HMM shell should retain a non-null name marker"
            );
            assert!(query[296..].windows(5).any(|window| window == b"test\0"));
            range_tx.send(range).unwrap();

            let payload = if name == "shard_a" {
                hmmpgmd_fake_c_worker_payload_with_stats(
                    name,
                    sortkey,
                    score,
                    1,
                    nincluded,
                    [2, 3, 4, 5],
                )
            } else {
                hmmpgmd_fake_worker_payload_with_filter_stats(
                    name,
                    sortkey,
                    score,
                    1,
                    nincluded,
                    [7, 11, 13, 17],
                )
            };
            stream.write_all(&0u32.to_be_bytes()).unwrap();
            stream
                .write_all(&(payload.len() as u64).to_be_bytes())
                .unwrap();
            stream.write_all(&payload).unwrap();
            stream.flush().unwrap();

            stream.read_exact(&mut header).unwrap();
            assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10004);
            write_hmmd_header(&mut stream, 0, 10004, 0).unwrap();
            stream.flush().unwrap();
        }));
    }

    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    std::thread::sleep(Duration::from_millis(250));

    let hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let request = format!("@--seqdb 1\n{hmm}");
    let mut stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut status = [0u8; 12];
    stream.read_exact(&mut status).unwrap();
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0u8; msg_size as usize];
    stream.read_exact(&mut payload).unwrap();

    let mut ranges = vec![
        range_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
        range_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
    ];
    ranges.sort();

    let mut shutdown_stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    shutdown_stream.write_all(b"!shutdown\n//\n").unwrap();
    let mut shutdown_status = [0u8; 12];
    shutdown_stream.read_exact(&mut shutdown_status).unwrap();
    let shutdown_code = u32::from_be_bytes(shutdown_status[0..4].try_into().unwrap());

    for worker in fake_workers {
        worker.join().unwrap();
    }
    let master_status = wait_for_child(&mut master, Duration::from_secs(5));
    let master_output = master.wait_with_output().unwrap();

    assert!(
        master_status
            .map(|status| status.success())
            .unwrap_or(false),
        "{}",
        String::from_utf8_lossy(&master_output.stderr)
    );
    assert_eq!(code, 0);
    assert_eq!(shutdown_code, 0);
    assert_eq!(ranges, ["1..1".to_string(), "2..2".to_string()]);
    assert_eq!(u64::from_be_bytes(payload[42..50].try_into().unwrap()), 1);
    assert_eq!(u64::from_be_bytes(payload[50..58].try_into().unwrap()), 2);
    assert_eq!(hmmpgmd_stats_u64(&payload, 58), 9);
    assert_eq!(hmmpgmd_stats_u64(&payload, 66), 14);
    assert_eq!(hmmpgmd_stats_u64(&payload, 74), 17);
    assert_eq!(hmmpgmd_stats_u64(&payload, 82), 22);
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 2);
    assert_eq!(hmmpgmd_stats_u64(&payload, 98), 2);
    assert_eq!(hmmpgmd_stats_u64(&payload, 106), 1);
    assert_eq!(
        hmmpgmd_hit_names(&payload),
        ["shard_b".to_string(), "shard_a".to_string()]
    );
}

#[test]
fn hmmpgmd_master_splits_requested_seqdb_range_across_workers() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("range-targets.fa");
    std::fs::write(
        &seqdb,
        ">target1\nACDEFGHIKLMNPQRSTVWY\n>target2\nACDEFGHIKLMNPQRSTVWY\n",
    )
    .unwrap();

    let cport = reserve_c_hmmpgmd_port(50000);
    let wport = reserve_c_hmmpgmd_port(cport.saturating_add(1));

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let _ = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();

    let mut workers = Vec::new();
    for _ in 0..2 {
        workers.push(
            Command::new(hmmer())
                .args([
                    "pgmd",
                    "--worker",
                    "127.0.0.1",
                    "--wport",
                    &wport.to_string(),
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }

    std::thread::sleep(Duration::from_millis(500));

    let hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let request = format!("@--seqdb 1 --seqdb_ranges 2..2\n{hmm}");
    let mut stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let (code, payload) = hmmpgmd_read_c_status_payload(&mut stream);

    let mut shutdown_stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    shutdown_stream.write_all(b"!shutdown\n//\n").unwrap();
    let mut shutdown_status = [0u8; 12];
    shutdown_stream.read_exact(&mut shutdown_status).unwrap();
    let shutdown_code = u32::from_be_bytes(shutdown_status[0..4].try_into().unwrap());

    let master_status = wait_for_child(&mut master, Duration::from_secs(5));
    let master_output = master.wait_with_output().unwrap();
    let worker_outputs = workers
        .into_iter()
        .map(|mut worker| {
            let status = wait_for_child(&mut worker, Duration::from_secs(5));
            (status, worker.wait_with_output().unwrap())
        })
        .collect::<Vec<_>>();

    assert!(
        master_status
            .map(|status| status.success())
            .unwrap_or(false),
        "{}",
        String::from_utf8_lossy(&master_output.stderr)
    );
    for (status, output) in worker_outputs {
        assert!(
            status.map(|status| status.success()).unwrap_or(false),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert_eq!(code, 0);
    assert_eq!(shutdown_code, 0);
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 1);
    assert_eq!(hmmpgmd_first_hit_name(&payload), "target2");
}

#[test]
fn hmmpgmd_master_falls_back_after_partial_worker_search_failure() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("partial-targets.fa");
    std::fs::write(&seqdb, ">fallback_target\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let cport = reserve_c_hmmpgmd_port(50000);
    let wport = reserve_c_hmmpgmd_port(cport.saturating_add(1));

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let _ = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();

    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let mut fake_workers = Vec::new();
    for succeeds in [true, false] {
        let ready_tx = ready_tx.clone();
        fake_workers.push(std::thread::spawn(move || {
            let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
            let mut header = [0u8; 12];
            stream.read_exact(&mut header).unwrap();
            let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
            assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
            let mut init_body = vec![0u8; init_len];
            stream.read_exact(&mut init_body).unwrap();
            write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
            stream.write_all(&[0u8; 96]).unwrap();
            stream.flush().unwrap();
            ready_tx.send(()).unwrap();

            stream.read_exact(&mut header).unwrap();
            let search_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
            assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10001);
            let mut search_body = vec![0u8; search_len];
            stream.read_exact(&mut search_body).unwrap();
            if succeeds {
                let payload = hmmpgmd_fake_worker_payload("worker_only_target");
                stream.write_all(&0u32.to_be_bytes()).unwrap();
                stream
                    .write_all(&(payload.len() as u64).to_be_bytes())
                    .unwrap();
                stream.write_all(&payload).unwrap();
                stream.flush().unwrap();
            }
        }));
    }

    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    std::thread::sleep(Duration::from_millis(250));

    let hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let request = format!("@--seqdb 1\n{hmm}");
    let mut stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let (code, payload) = hmmpgmd_read_c_status_payload(&mut stream);

    for worker in fake_workers {
        worker.join().unwrap();
    }

    let mut shutdown_stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    shutdown_stream.write_all(b"!shutdown\n//\n").unwrap();
    let mut shutdown_status = [0u8; 12];
    shutdown_stream.read_exact(&mut shutdown_status).unwrap();
    let shutdown_code = u32::from_be_bytes(shutdown_status[0..4].try_into().unwrap());

    let master_status = wait_for_child(&mut master, Duration::from_secs(5));
    let master_output = master.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&master_output.stderr);

    assert!(
        master_status
            .map(|status| status.success())
            .unwrap_or(false),
        "{stderr}"
    );
    assert_eq!(code, 0);
    assert_eq!(shutdown_code, 0);
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 1);
    assert_eq!(hmmpgmd_first_hit_name(&payload), "fallback_target");
    assert!(
        stderr.contains("Discarding incomplete worker search result: received 1 of 2 shards"),
        "{stderr}"
    );
}

#[test]
fn hmmpgmd_master_assigns_hmmdb_shards_with_binary_search_cmd() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let hmm_template = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let hmmdb = dir.path().join("models.hmm");
    std::fs::write(
        &hmmdb,
        format!(
            "{}{}",
            hmm_template.replacen("NAME  test", "NAME  model_a", 1),
            hmm_template.replacen("NAME  test", "NAME  model_b", 1)
        ),
    )
    .unwrap();

    let client_listener = bind_hmmpgmd_listener();
    let cport = client_listener.local_addr().unwrap().port();
    drop(client_listener);

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(worker_listener);

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--hmmdb",
            hmmdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let _ = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();

    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (slice_tx, slice_rx) = std::sync::mpsc::channel();
    let mut fake_workers = Vec::new();
    for name in ["model_shard_a", "model_shard_b"] {
        let ready_tx = ready_tx.clone();
        let slice_tx = slice_tx.clone();
        let hmmdb = hmmdb.clone();
        fake_workers.push(std::thread::spawn(move || {
            let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
            let mut header = [0u8; 12];
            stream.read_exact(&mut header).unwrap();
            let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
            assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
            let mut init_body = vec![0u8; init_len];
            stream.read_exact(&mut init_body).unwrap();
            assert!(init_body.len() > 88);
            assert_eq!(u32::from_ne_bytes(init_body[68..72].try_into().unwrap()), 0);
            assert_eq!(u32::from_ne_bytes(init_body[80..84].try_into().unwrap()), 1);
            assert_eq!(u32::from_ne_bytes(init_body[84..88].try_into().unwrap()), 2);
            assert_eq!(
                std::str::from_utf8(&init_body[88..init_body.len() - 1]).unwrap(),
                hmmdb.to_str().unwrap()
            );
            write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
            stream.write_all(&[0u8; 96]).unwrap();
            stream.flush().unwrap();
            ready_tx.send(()).unwrap();

            stream.read_exact(&mut header).unwrap();
            let search_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
            assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10002);
            let mut search_body = vec![0u8; search_len];
            stream.read_exact(&mut search_body).unwrap();
            assert!(search_body.len() >= 28);
            assert_eq!(u32::from_ne_bytes(search_body[0..4].try_into().unwrap()), 0);
            assert!(u32::from_ne_bytes(search_body[8..12].try_into().unwrap()) <= 1);
            assert_eq!(
                u32::from_ne_bytes(search_body[12..16].try_into().unwrap()),
                1
            );
            assert_eq!(
                u32::from_ne_bytes(search_body[16..20].try_into().unwrap()),
                101
            );
            assert_eq!(
                u32::from_ne_bytes(search_body[20..24].try_into().unwrap()),
                22
            );
            let opts_len = u32::from_ne_bytes(search_body[24..28].try_into().unwrap()) as usize;
            let opts = &search_body[28..28 + opts_len];
            assert_eq!(opts.last().copied(), Some(0));
            assert!(String::from_utf8_lossy(&opts[..opts.len() - 1]).contains("--hmmdb"));
            let query = &search_body[28 + opts_len..];
            assert!(query.starts_with(b"query\0\0"));
            assert_eq!(
                &query[7..29],
                &[255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 255]
            );
            assert_eq!(&query[29..], &[0u8; 60]);
            let inx = u32::from_ne_bytes(search_body[8..12].try_into().unwrap());
            slice_tx.send((inx, 1u32)).unwrap();

            let payload = hmmpgmd_fake_worker_payload(name);
            stream.write_all(&0u32.to_be_bytes()).unwrap();
            stream
                .write_all(&(payload.len() as u64).to_be_bytes())
                .unwrap();
            stream.write_all(&payload).unwrap();
            stream.flush().unwrap();

            stream.read_exact(&mut header).unwrap();
            assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10004);
            write_hmmd_header(&mut stream, 0, 10004, 0).unwrap();
            stream.flush().unwrap();
        }));
    }

    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    std::thread::sleep(Duration::from_millis(250));

    let request = "@--hmmdb 1\n>query\nACDEFGHIKLMNPQRSTVWY\n//\n";
    let mut stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut status = [0u8; 12];
    stream.read_exact(&mut status).unwrap();
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0u8; msg_size as usize];
    stream.read_exact(&mut payload).unwrap();

    let mut slices = vec![
        slice_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
        slice_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
    ];
    slices.sort();

    let mut shutdown_stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    shutdown_stream.write_all(b"!shutdown\n//\n").unwrap();
    let mut shutdown_status = [0u8; 12];
    shutdown_stream.read_exact(&mut shutdown_status).unwrap();
    let shutdown_code = u32::from_be_bytes(shutdown_status[0..4].try_into().unwrap());

    for worker in fake_workers {
        worker.join().unwrap();
    }
    let master_status = wait_for_child(&mut master, Duration::from_secs(5));
    let master_output = master.wait_with_output().unwrap();

    assert!(
        master_status
            .map(|status| status.success())
            .unwrap_or(false),
        "{}",
        String::from_utf8_lossy(&master_output.stderr)
    );
    assert_eq!(code, 0);
    assert_eq!(shutdown_code, 0);
    assert_eq!(slices, [(0, 1), (1, 1)]);
    assert_eq!(u64::from_be_bytes(payload[42..50].try_into().unwrap()), 2);
    assert_eq!(u64::from_be_bytes(payload[50..58].try_into().unwrap()), 1);
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 2);
    let mut hit_names = hmmpgmd_hit_names(&payload);
    hit_names.sort();
    assert_eq!(
        hit_names,
        ["model_shard_a".to_string(), "model_shard_b".to_string()]
    );
}

#[test]
fn hmmpgmd_ascii_seqdb_worker_request_carries_filter_and_null_flags() {
    let _guard = hmmpgmd_test_guard();
    for flag in ["--nobias", "--nonull2", "--max"] {
        let body = hmmpgmd_capture_seqdb_hmm_ascii_worker_request(flag);
        let text = String::from_utf8(body).unwrap();
        let options = text.lines().next().unwrap();
        let tokens = options
            .split_whitespace()
            .map(|word| word.strip_prefix('@').unwrap_or(word))
            .collect::<Vec<_>>();
        for expected in ["--seqdb", flag, "--seqdb_ranges"] {
            assert!(
                tokens.contains(&expected),
                "{expected} missing from ASCII worker request options: {options}"
            );
        }
        assert!(
            text.contains("\nHMMER3/"),
            "ASCII seqdb worker request should carry the HMM query after options"
        );
    }
}

#[test]
fn hmmpgmd_binary_hmmdb_worker_request_carries_filter_and_null_flags() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("20aa.hmm");
    std::fs::copy("hmmer/testsuite/20aa.hmm", &hmmdb).unwrap();
    let press = Command::new(c_hmmpress())
        .args(["-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}{}",
        String::from_utf8_lossy(&press.stdout),
        String::from_utf8_lossy(&press.stderr)
    );

    for flag in ["--nobias", "--nonull2", "--max"] {
        let option_suffix = format!(" {flag}");
        let rust_body = hmmpgmd_capture_hmmdb_sequence_worker_request(true, &hmmdb, &option_suffix);

        let opts_len = u32::from_ne_bytes(rust_body[24..28].try_into().unwrap()) as usize;
        let options = normalized_hmmpgmd_worker_options(&rust_body[28..28 + opts_len]);
        assert_eq!(options, ["--hmmdb", "1", flag]);
        assert_eq!(
            u32::from_ne_bytes(rust_body[16..20].try_into().unwrap()),
            101
        );
    }
}

#[test]
fn hmmpgmd_seqdb_hmm_worker_request_tail_matches_c_daemon_bytes() {
    let _guard = hmmpgmd_test_guard();
    let rust_body = hmmpgmd_capture_seqdb_hmm_worker_request(true, "");
    let c_body = hmmpgmd_capture_seqdb_hmm_worker_request(false, "");
    assert_eq!(
        normalized_hmmpgmd_worker_request_body(&rust_body, true),
        normalized_hmmpgmd_worker_request_body(&c_body, true),
        "full stable HMMD_SEARCH_CMD body differs after normalizing process-local pointer slots and C-only option spelling"
    );

    for offset in [0usize, 4, 8, 12, 16, 20] {
        assert_eq!(
            &rust_body[offset..offset + 4],
            &c_body[offset..offset + 4],
            "HMMD_SEARCH_CMD field at offset {offset} differs"
        );
    }
    assert_eq!(
        u32::from_ne_bytes(rust_body[16..20].try_into().unwrap()),
        102
    );

    let rust_opts_len = u32::from_ne_bytes(rust_body[24..28].try_into().unwrap()) as usize;
    let c_opts_len = u32::from_ne_bytes(c_body[24..28].try_into().unwrap()) as usize;
    assert_eq!(
        normalized_hmmpgmd_worker_options(&rust_body[28..28 + rust_opts_len]),
        normalized_hmmpgmd_worker_options(&c_body[28..28 + c_opts_len]),
        "stable HMMD_SEARCH_CMD options differ after normalizing C-only worker options"
    );
    assert_eq!(
        normalized_hmmpgmd_worker_options(&rust_body[28..28 + rust_opts_len]),
        vec!["--seqdb".to_string(), "1".to_string()]
    );

    let rust_query = &rust_body[28 + rust_opts_len..];
    let c_query = &c_body[28 + c_opts_len..];
    assert!(rust_query.len() > 296);
    let query_length = u32::from_ne_bytes(rust_body[20..24].try_into().unwrap()) as usize;
    let c_query_length = u32::from_ne_bytes(c_body[20..24].try_into().unwrap()) as usize;
    let rust_shell_m = i32::from_ne_bytes(rust_query[0..4].try_into().unwrap()) as usize;
    let c_shell_m = i32::from_ne_bytes(c_query[0..4].try_into().unwrap()) as usize;
    assert_eq!(
        query_length, rust_shell_m,
        "Rust HMMD_SEARCH_CMD query_length does not match serialized P7_HMM.M"
    );
    assert_eq!(
        c_query_length, c_shell_m,
        "C HMMD_SEARCH_CMD query_length does not match serialized P7_HMM.M"
    );
    let alphabet_len = 20usize;
    let transitions_len = (query_length + 1) * 7 * std::mem::size_of::<f32>();
    let emissions_len = (query_length + 1) * alphabet_len * std::mem::size_of::<f32>();
    let numeric_blocks_len = transitions_len + emissions_len + emissions_len;
    let flags = u32::from_ne_bytes(rust_query[288..292].try_into().unwrap());
    let c_flags = u32::from_ne_bytes(c_query[288..292].try_into().unwrap());
    assert_eq!(flags, c_flags, "serialized P7_HMM flags differ");
    assert_eq!(
        flags & (P7H_RF | P7H_MMASK | P7H_CONS | P7H_CS | P7H_CA | P7H_MAP),
        P7H_RF | P7H_CONS | P7H_MAP,
        "20aa.hmm should serialize exactly RF, CONS, and MAP optional tail blocks"
    );
    let expected_hmm_tail_len = b"test\0".len()
        + (query_length + 2)
        + (query_length + 2)
        + (query_length + 1) * std::mem::size_of::<i32>();
    let expected_query_bytes_len = 296 + numeric_blocks_len + expected_hmm_tail_len;
    let expected_command_padding_len = 60usize;
    assert_eq!(
        rust_body.len(),
        28 + rust_opts_len + expected_query_bytes_len + expected_command_padding_len,
        "Rust HMMD_SEARCH_CMD body length does not match fixed prefix/options/P7_HMM/padding lengths"
    );
    assert_eq!(
        c_body.len(),
        28 + c_opts_len + expected_query_bytes_len + expected_command_padding_len,
        "C HMMD_SEARCH_CMD body length does not match fixed prefix/options/P7_HMM/padding lengths"
    );
    let rust_arrays = &rust_query[296..296 + numeric_blocks_len];
    let c_arrays_start = c_query
        .windows(64)
        .position(|window| window == &rust_arrays[..64])
        .expect("C request does not contain the Rust transition-array prefix");
    assert_eq!(
        c_arrays_start, 296,
        "C request serialized P7_HMM numeric arrays at an unexpected shell offset"
    );
    assert_eq!(
        normalized_hmmpgmd_p7_hmm_shell(&rust_query[..296]),
        normalized_hmmpgmd_p7_hmm_shell(&c_query[..296]),
        "serialized P7_HMM shell differs from C after ignoring process-local pointer addresses"
    );
    assert_hmmpgmd_p7_hmm_shell_only_pointer_slots_differ(&rust_query[..296], &c_query[..296]);
    assert_hmmpgmd_rust_p7_hmm_pointer_layout_matches_serialized_blocks(
        &rust_query[..296],
        transitions_len,
        emissions_len,
        query_length,
    );
    let c_arrays = &c_query[c_arrays_start..c_arrays_start + numeric_blocks_len];
    if rust_arrays != c_arrays {
        let offset = rust_arrays
            .iter()
            .zip(c_arrays)
            .position(|(rust, c)| rust != c)
            .expect("array blocks differ in length");
        panic!(
            "serialized P7_HMM transition/match/insert numeric blocks after the pointer-bearing shell differ from C at numeric-block byte offset {offset}: rust={} c={}",
            rust_arrays[offset],
            c_arrays[offset]
        );
    }
    let rust_tail = &rust_query[296 + numeric_blocks_len..];
    let c_tail = &c_query[c_arrays_start + numeric_blocks_len..];
    assert_eq!(
        rust_tail.len(),
        expected_hmm_tail_len + expected_command_padding_len,
        "Rust serialized P7_HMM tail plus command padding has an unexpected length"
    );
    assert_eq!(
        c_tail.len(),
        expected_hmm_tail_len + expected_command_padding_len,
        "C serialized P7_HMM tail plus command padding has an unexpected length"
    );
    assert_eq!(
        rust_tail, c_tail,
        "serialized P7_HMM optional name/annotation/map tail and C command padding after numeric blocks differs from C"
    );
    assert!(rust_tail.starts_with(b"test\0"));
    assert!(rust_tail.ends_with(&[0u8; 60]));
}

#[test]
fn hmmpgmd_seqdb_hmm_worker_request_preserves_threshold_options() {
    let _guard = hmmpgmd_test_guard();
    let option_suffix =
        " -E 7 --domE 8 -T 9 --domT 10 --incE 11 --incdomE 12 --incT 13 --incdomT 14";
    let rust_body = hmmpgmd_capture_seqdb_hmm_worker_request(true, option_suffix);

    let rust_opts_len = u32::from_ne_bytes(rust_body[24..28].try_into().unwrap()) as usize;
    let expected = vec![
        "--seqdb".to_string(),
        "1".to_string(),
        "-E".to_string(),
        "7".to_string(),
        "--domE".to_string(),
        "8".to_string(),
        "-T".to_string(),
        "9".to_string(),
        "--domT".to_string(),
        "10".to_string(),
        "--incE".to_string(),
        "11".to_string(),
        "--incdomE".to_string(),
        "12".to_string(),
        "--incT".to_string(),
        "13".to_string(),
        "--incdomT".to_string(),
        "14".to_string(),
    ];
    assert_eq!(
        normalized_hmmpgmd_worker_options(&rust_body[28..28 + rust_opts_len]),
        expected
    );
}

#[test]
fn hmmpgmd_seqdb_hmm_worker_request_preserves_model_cutoff_options() {
    let _guard = hmmpgmd_test_guard();
    for cutoff in ["--cut_ga", "--cut_tc", "--cut_nc"] {
        let option_suffix = format!(" {cutoff}");
        let rust_body = hmmpgmd_capture_seqdb_hmm_worker_request(true, &option_suffix);
        let c_body = hmmpgmd_capture_seqdb_hmm_worker_request(false, &option_suffix);

        let rust_opts_len = u32::from_ne_bytes(rust_body[24..28].try_into().unwrap()) as usize;
        let c_opts_len = u32::from_ne_bytes(c_body[24..28].try_into().unwrap()) as usize;
        let expected = vec!["--seqdb".to_string(), "1".to_string(), cutoff.to_string()];
        assert_eq!(
            normalized_hmmpgmd_worker_options(&rust_body[28..28 + rust_opts_len]),
            expected
        );
        assert_eq!(
            normalized_hmmpgmd_worker_options(&rust_body[28..28 + rust_opts_len]),
            normalized_hmmpgmd_worker_options(&c_body[28..28 + c_opts_len])
        );
    }
}

#[test]
fn hmmpgmd_hmmdb_sequence_worker_request_matches_c_daemon_bytes() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("20aa.hmm");
    std::fs::copy("hmmer/testsuite/20aa.hmm", &hmmdb).unwrap();
    let press = Command::new(c_hmmpress())
        .args(["-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}{}",
        String::from_utf8_lossy(&press.stdout),
        String::from_utf8_lossy(&press.stderr)
    );

    let rust_body = hmmpgmd_capture_hmmdb_sequence_worker_request(true, &hmmdb, "");
    let c_body = hmmpgmd_capture_hmmdb_sequence_worker_request(false, &hmmdb, "");
    assert_eq!(
        normalized_hmmpgmd_worker_request_body(&rust_body, false),
        normalized_hmmpgmd_worker_request_body(&c_body, false),
        "full stable HMMDB sequence HMMD_SEARCH_CMD body differs after normalizing C-only option spelling"
    );

    for offset in [0usize, 4, 8, 12, 16, 20] {
        assert_eq!(
            &rust_body[offset..offset + 4],
            &c_body[offset..offset + 4],
            "HMMD_SEARCH_CMD field at offset {offset} differs"
        );
    }
    assert_eq!(
        u32::from_ne_bytes(rust_body[16..20].try_into().unwrap()),
        101
    );
    assert_eq!(
        u32::from_ne_bytes(rust_body[20..24].try_into().unwrap()),
        22
    );

    let rust_opts_len = u32::from_ne_bytes(rust_body[24..28].try_into().unwrap()) as usize;
    let c_opts_len = u32::from_ne_bytes(c_body[24..28].try_into().unwrap()) as usize;
    assert_eq!(
        normalized_hmmpgmd_worker_options(&rust_body[28..28 + rust_opts_len]),
        normalized_hmmpgmd_worker_options(&c_body[28..28 + c_opts_len]),
        "stable HMMD_SEARCH_CMD options differ after normalizing C-only worker options"
    );
    assert_eq!(
        normalized_hmmpgmd_worker_options(&rust_body[28..28 + rust_opts_len]),
        vec!["--hmmdb".to_string(), "1".to_string()]
    );

    let rust_query = &rust_body[28 + rust_opts_len..];
    let c_query = &c_body[28 + c_opts_len..];
    assert_eq!(
        rust_query, c_query,
        "serialized HMMDB sequence query body differs from C"
    );
    assert_eq!(
        &rust_query[..29],
        &[
            b'q', b'u', b'e', b'r', b'y', 0, 0, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 255,
        ]
    );
    assert_eq!(&rust_query[29..], &[0u8; 60]);
    assert_eq!(rust_body.len(), 28 + rust_opts_len + rust_query.len());
    assert_eq!(c_body.len(), 28 + c_opts_len + c_query.len());
}

#[test]
fn hmmpgmd_hmmdb_sequence_worker_request_preserves_threshold_options() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("20aa.hmm");
    std::fs::copy("hmmer/testsuite/20aa.hmm", &hmmdb).unwrap();
    let press = Command::new(c_hmmpress())
        .args(["-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}{}",
        String::from_utf8_lossy(&press.stdout),
        String::from_utf8_lossy(&press.stderr)
    );

    let option_suffix =
        " -E 7 --domE 8 -T 9 --domT 10 --incE 11 --incdomE 12 --incT 13 --incdomT 14";
    let rust_body = hmmpgmd_capture_hmmdb_sequence_worker_request(true, &hmmdb, option_suffix);

    let rust_opts_len = u32::from_ne_bytes(rust_body[24..28].try_into().unwrap()) as usize;
    let expected = vec![
        "--hmmdb".to_string(),
        "1".to_string(),
        "-E".to_string(),
        "7".to_string(),
        "--domE".to_string(),
        "8".to_string(),
        "-T".to_string(),
        "9".to_string(),
        "--domT".to_string(),
        "10".to_string(),
        "--incE".to_string(),
        "11".to_string(),
        "--incdomE".to_string(),
        "12".to_string(),
        "--incT".to_string(),
        "13".to_string(),
        "--incdomT".to_string(),
        "14".to_string(),
    ];
    assert_eq!(
        normalized_hmmpgmd_worker_options(&rust_body[28..28 + rust_opts_len]),
        expected
    );
}

#[test]
fn hmmpgmd_hmmdb_sequence_worker_request_preserves_model_cutoff_options() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("20aa.hmm");
    std::fs::copy("hmmer/testsuite/20aa.hmm", &hmmdb).unwrap();
    let press = Command::new(c_hmmpress())
        .args(["-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}{}",
        String::from_utf8_lossy(&press.stdout),
        String::from_utf8_lossy(&press.stderr)
    );

    for cutoff in ["--cut_ga", "--cut_tc", "--cut_nc"] {
        let option_suffix = format!(" {cutoff}");
        let rust_body = hmmpgmd_capture_hmmdb_sequence_worker_request(true, &hmmdb, &option_suffix);
        let c_body = hmmpgmd_capture_hmmdb_sequence_worker_request(false, &hmmdb, &option_suffix);

        let rust_opts_len = u32::from_ne_bytes(rust_body[24..28].try_into().unwrap()) as usize;
        let c_opts_len = u32::from_ne_bytes(c_body[24..28].try_into().unwrap()) as usize;
        let expected = vec!["--hmmdb".to_string(), "1".to_string(), cutoff.to_string()];
        assert_eq!(
            normalized_hmmpgmd_worker_options(&rust_body[28..28 + rust_opts_len]),
            expected
        );
        assert_eq!(
            normalized_hmmpgmd_worker_options(&rust_body[28..28 + rust_opts_len]),
            normalized_hmmpgmd_worker_options(&c_body[28..28 + c_opts_len])
        );
    }
}

#[test]
fn hmmpgmd_hmmdb_init_body_matches_c_daemon_bytes() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("20aa.hmm");
    std::fs::copy("hmmer/testsuite/20aa.hmm", &hmmdb).unwrap();
    let press = Command::new(c_hmmpress())
        .args(["-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}{}",
        String::from_utf8_lossy(&press.stdout),
        String::from_utf8_lossy(&press.stderr)
    );

    let rust_body = hmmpgmd_capture_hmmdb_init_body(true, &hmmdb);
    let c_body = hmmpgmd_capture_hmmdb_init_body(false, &hmmdb);
    assert_eq!(
        rust_body, c_body,
        "Rust master-emitted HMMD_CMD_INIT HMMDB body differs from C"
    );
    assert_eq!(rust_body.len(), 88 + hmmdb.to_string_lossy().len() + 1);
    assert_eq!(u32::from_ne_bytes(rust_body[68..72].try_into().unwrap()), 0);
    assert_eq!(u32::from_ne_bytes(rust_body[80..84].try_into().unwrap()), 1);
    assert_eq!(u32::from_ne_bytes(rust_body[84..88].try_into().unwrap()), 1);
    assert_eq!(
        std::str::from_utf8(&rust_body[88..rust_body.len() - 1]).unwrap(),
        hmmdb.to_str().unwrap()
    );
    assert_eq!(rust_body.last().copied(), Some(0));
}

#[test]
fn hmmpgmd_seqdb_init_body_matches_c_daemon_bytes() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    std::fs::write(
        &seqdb,
        "#40 2 1 2 2 parity\n>target1 1\nACDEFGHIKLMNPQRSTVWY\n>target2 1\nYYYYYYYYYYYYYYYYYYYY\n",
    )
    .unwrap();

    let rust_body = hmmpgmd_capture_seqdb_init_body(true, &seqdb);
    let c_body = hmmpgmd_capture_seqdb_init_body(false, &seqdb);
    assert_eq!(
        rust_body, c_body,
        "Rust master-emitted HMMD_CMD_INIT SEQDB body differs from C"
    );
    assert_eq!(rust_body.len(), 88 + seqdb.to_string_lossy().len() + 1);
    assert_eq!(&rust_body[..7], b"parity\0");
    assert_eq!(u32::from_ne_bytes(rust_body[64..68].try_into().unwrap()), 0);
    assert_eq!(u32::from_ne_bytes(rust_body[72..76].try_into().unwrap()), 1);
    assert_eq!(u32::from_ne_bytes(rust_body[76..80].try_into().unwrap()), 2);
    assert_eq!(
        std::str::from_utf8(&rust_body[88..rust_body.len() - 1]).unwrap(),
        seqdb.to_str().unwrap()
    );
    assert_eq!(rust_body.last().copied(), Some(0));
}

const HMMPGMD_P7_HMM_POINTER_SLOTS: [usize; 15] =
    [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 120, 128, 280];

fn normalized_hmmpgmd_p7_hmm_shell(shell: &[u8]) -> Vec<u8> {
    let mut normalized = shell.to_vec();
    for offset in HMMPGMD_P7_HMM_POINTER_SLOTS {
        normalized[offset..offset + std::mem::size_of::<usize>()].fill(0);
    }
    normalized
}

fn normalized_hmmpgmd_worker_request_body(body: &[u8], normalize_hmm_shell: bool) -> Vec<u8> {
    assert!(
        body.len() >= 28,
        "captured HMMD_SEARCH_CMD body is shorter than its fixed prefix"
    );
    let opts_len = u32::from_ne_bytes(body[24..28].try_into().unwrap()) as usize;
    assert!(
        body.len() >= 28 + opts_len,
        "captured HMMD_SEARCH_CMD body is shorter than its options block"
    );

    let opts = normalized_hmmpgmd_worker_options(&body[28..28 + opts_len]);
    let mut canonical_opts = opts.join(" ").into_bytes();
    canonical_opts.push(0);

    let mut normalized = Vec::with_capacity(body.len() - opts_len + canonical_opts.len());
    normalized.extend_from_slice(&body[..24]);
    normalized.extend_from_slice(&(canonical_opts.len() as u32).to_ne_bytes());
    normalized.extend_from_slice(&canonical_opts);

    let mut query = body[28 + opts_len..].to_vec();
    if normalize_hmm_shell {
        assert!(
            query.len() >= 296,
            "captured HMM query body is shorter than its P7_HMM shell"
        );
        let normalized_shell = normalized_hmmpgmd_p7_hmm_shell(&query[..296]);
        query[..296].copy_from_slice(&normalized_shell);
    }
    normalized.extend_from_slice(&query);
    normalized
}

fn assert_hmmpgmd_p7_hmm_shell_only_pointer_slots_differ(rust_shell: &[u8], c_shell: &[u8]) {
    assert_eq!(rust_shell.len(), 296);
    assert_eq!(c_shell.len(), 296);

    let ptr_width = std::mem::size_of::<usize>();
    for (offset, (rust, c)) in rust_shell.iter().zip(c_shell).enumerate() {
        if rust == c {
            continue;
        }
        assert!(
            HMMPGMD_P7_HMM_POINTER_SLOTS
                .iter()
                .any(|slot| (*slot..*slot + ptr_width).contains(&offset)),
            "serialized P7_HMM shell differs outside pointer slots at byte {offset}: rust={rust} c={c}"
        );
    }

    for offset in [8usize, 16, 24, 32, 56, 72, 120, 128, 280] {
        assert!(
            rust_shell[offset..offset + ptr_width]
                .iter()
                .any(|byte| *byte != 0),
            "Rust P7_HMM shell pointer slot at offset {offset} is unexpectedly NULL"
        );
        assert!(
            c_shell[offset..offset + ptr_width]
                .iter()
                .any(|byte| *byte != 0),
            "C P7_HMM shell pointer slot at offset {offset} is unexpectedly NULL"
        );
    }

    for offset in HMMPGMD_P7_HMM_POINTER_SLOTS {
        let rust_present = rust_shell[offset..offset + ptr_width]
            .iter()
            .any(|byte| *byte != 0);
        let c_present = c_shell[offset..offset + ptr_width]
            .iter()
            .any(|byte| *byte != 0);
        assert_eq!(
            rust_present, c_present,
            "serialized P7_HMM pointer-slot presence differs at offset {offset}"
        );
    }
}

fn assert_hmmpgmd_rust_p7_hmm_pointer_layout_matches_serialized_blocks(
    shell: &[u8],
    transitions_len: usize,
    emissions_len: usize,
    query_length: usize,
) {
    let t = hmmpgmd_shell_ptr(shell, 8);
    let mat = hmmpgmd_shell_ptr(shell, 16);
    let ins = hmmpgmd_shell_ptr(shell, 24);
    let name = hmmpgmd_shell_ptr(shell, 32);
    let rf = hmmpgmd_shell_ptr(shell, 56);
    let consensus = hmmpgmd_shell_ptr(shell, 72);
    let map = hmmpgmd_shell_ptr(shell, 128);

    for (label, ptr) in [
        ("transition", t),
        ("match emission", mat),
        ("insert emission", ins),
        ("name", name),
        ("RF annotation", rf),
        ("consensus annotation", consensus),
        ("MAP", map),
    ] {
        assert_ne!(ptr, 0, "Rust P7_HMM {label} pointer is NULL");
        assert_ne!(
            ptr, 1,
            "Rust P7_HMM {label} pointer is still the old marker byte value"
        );
    }

    assert_eq!(mat - t, transitions_len);
    assert_eq!(ins - mat, emissions_len);
    assert_eq!(name - ins, emissions_len);
    assert_eq!(rf - name, b"test\0".len());
    assert_eq!(consensus - rf, query_length + 2);
    assert_eq!(map - consensus, query_length + 2);
}

fn hmmpgmd_shell_ptr(shell: &[u8], offset: usize) -> usize {
    let width = std::mem::size_of::<usize>();
    let mut value = [0u8; std::mem::size_of::<usize>()];
    value.copy_from_slice(&shell[offset..offset + width]);
    usize::from_ne_bytes(value)
}

fn normalized_hmmpgmd_worker_options(options: &[u8]) -> Vec<String> {
    assert_eq!(
        options.last().copied(),
        Some(0),
        "HMMD_SEARCH_CMD options block is not NUL-terminated"
    );
    let text = std::str::from_utf8(&options[..options.len() - 1])
        .expect("HMMD_SEARCH_CMD options block is not UTF-8");
    let mut tokens = Vec::new();
    let mut words = text.split_whitespace();
    while let Some(word) = words.next() {
        if word == "hmmpgmd" {
            continue;
        }
        if word == "--cpu" {
            let _ = words.next();
            continue;
        }
        tokens.push(word.strip_prefix('@').unwrap_or(word).to_string());
    }
    tokens
}

fn hmmpgmd_capture_hmmdb_init_body(rust_master: bool, hmmdb: &std::path::Path) -> Vec<u8> {
    let cport = reserve_c_hmmpgmd_port(50000);
    let wport = reserve_c_hmmpgmd_port(cport.saturating_add(1));

    let mut master_cmd = if rust_master {
        let mut command = Command::new(hmmer());
        command.arg("pgmd");
        command
    } else {
        Command::new(c_hmmpgmd())
    };
    let mut master = master_cmd
        .args([
            "--master",
            "--hmmdb",
            hmmdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    if let Err(err) = connect_with_deadline(cport, Duration::from_secs(5)) {
        let _ = master.kill();
        let output = master.wait_with_output().unwrap();
        panic!(
            "hmmpgmd master did not accept client probe on {cport}: {err}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let (body_tx, body_rx) = std::sync::mpsc::channel();
    let fake_worker = std::thread::spawn(move || {
        let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
        let mut header = [0u8; 12];
        stream.read_exact(&mut header).unwrap();
        let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
        assert_eq!(u32::from_ne_bytes(header[8..12].try_into().unwrap()), 0);
        let mut init_body = vec![0u8; init_len];
        stream.read_exact(&mut init_body).unwrap();
        write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
        stream.write_all(&[0u8; 96]).unwrap();
        stream.flush().unwrap();
        body_tx.send(init_body).unwrap();
    });

    let body = body_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    fake_worker.join().unwrap();

    let _ = master.kill();
    let master_output = master.wait_with_output().unwrap();
    assert!(
        master_output.status.success() || master_output.status.code().is_none(),
        "master failed before INIT capture; stdout={}; stderr={}",
        String::from_utf8_lossy(&master_output.stdout),
        String::from_utf8_lossy(&master_output.stderr)
    );

    body
}

fn hmmpgmd_capture_seqdb_init_body(rust_master: bool, seqdb: &std::path::Path) -> Vec<u8> {
    let cport = reserve_c_hmmpgmd_port(50000);
    let wport = reserve_c_hmmpgmd_port(cport.saturating_add(1));

    let mut master_cmd = if rust_master {
        let mut command = Command::new(hmmer());
        command.arg("pgmd");
        command
    } else {
        Command::new(c_hmmpgmd())
    };
    let mut master = master_cmd
        .args([
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    if let Err(err) = connect_with_deadline(cport, Duration::from_secs(5)) {
        let _ = master.kill();
        let output = master.wait_with_output().unwrap();
        panic!(
            "hmmpgmd master did not accept client probe on {cport}: {err}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let (body_tx, body_rx) = std::sync::mpsc::channel();
    let fake_worker = std::thread::spawn(move || {
        let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
        let mut header = [0u8; 12];
        stream.read_exact(&mut header).unwrap();
        let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
        assert_eq!(u32::from_ne_bytes(header[8..12].try_into().unwrap()), 0);
        let mut init_body = vec![0u8; init_len];
        stream.read_exact(&mut init_body).unwrap();
        write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
        stream.write_all(&[0u8; 96]).unwrap();
        stream.flush().unwrap();
        body_tx.send(init_body).unwrap();
    });

    let body = body_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    fake_worker.join().unwrap();

    let _ = master.kill();
    let master_output = master.wait_with_output().unwrap();
    assert!(
        master_output.status.success() || master_output.status.code().is_none(),
        "master failed before INIT capture; stdout={}; stderr={}",
        String::from_utf8_lossy(&master_output.stdout),
        String::from_utf8_lossy(&master_output.stderr)
    );

    body
}

fn hmmpgmd_capture_seqdb_hmm_worker_request(rust_master: bool, option_suffix: &str) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    let seqdb_contents = if rust_master {
        ">worker_target\nACDEFGHIKLMNPQRSTVWY\n".to_string()
    } else {
        "#20 1 1 1 1 parity\n>worker_target 1\nACDEFGHIKLMNPQRSTVWY\n".to_string()
    };
    std::fs::write(&seqdb, seqdb_contents).unwrap();

    let cport = reserve_c_hmmpgmd_port(50000);
    let wport = reserve_c_hmmpgmd_port(cport.saturating_add(1));

    let mut master_cmd = if rust_master {
        let mut command = Command::new(hmmer());
        command.arg("pgmd");
        command
    } else {
        Command::new(c_hmmpgmd())
    };
    let mut master = master_cmd
        .args([
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (body_tx, body_rx) = std::sync::mpsc::channel();
    let fake_worker = std::thread::spawn(move || {
        let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
        let mut header = [0u8; 12];
        stream.read_exact(&mut header).unwrap();
        let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
        let mut init_body = vec![0u8; init_len];
        stream.read_exact(&mut init_body).unwrap();
        write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
        stream.write_all(&[0u8; 96]).unwrap();
        stream.flush().unwrap();
        ready_tx.send(()).unwrap();

        stream.read_exact(&mut header).unwrap();
        let search_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10001);
        assert_eq!(u32::from_ne_bytes(header[8..12].try_into().unwrap()), 0);
        let mut search_body = vec![0u8; search_len];
        stream.read_exact(&mut search_body).unwrap();
        assert_eq!(search_len, search_body.len());
        body_tx.send(search_body).unwrap();
    });

    if let Err(err) = connect_with_deadline(cport, Duration::from_secs(5)) {
        let _ = master.kill();
        let output = master.wait_with_output().unwrap();
        panic!(
            "hmmpgmd master did not accept client probe on {cport}: {err}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    std::thread::sleep(Duration::from_millis(250));

    let hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let request = format!("@--seqdb 1{option_suffix}\n{hmm}");
    let mut stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    stream.flush().unwrap();

    let body = body_rx.recv_timeout(Duration::from_secs(5)).unwrap();

    let _ = master.kill();
    fake_worker.join().unwrap();
    let master_output = master.wait_with_output().unwrap();
    assert!(
        master_output.status.success() || master_output.status.code().is_none(),
        "master failed before request capture; stdout={}; stderr={}",
        String::from_utf8_lossy(&master_output.stdout),
        String::from_utf8_lossy(&master_output.stderr)
    );

    body
}

fn hmmpgmd_capture_seqdb_hmm_ascii_worker_request(request_options: &str) -> Vec<u8> {
    let option_suffix = if request_options.is_empty() {
        " --seqdb_ranges 1..1".to_string()
    } else {
        format!(" {request_options} --seqdb_ranges 1..1")
    };
    hmmpgmd_capture_seqdb_hmm_worker_request(true, &option_suffix)
}

fn hmmpgmd_capture_hmmdb_sequence_worker_request(
    rust_master: bool,
    hmmdb: &std::path::Path,
    option_suffix: &str,
) -> Vec<u8> {
    let cport = reserve_c_hmmpgmd_port(50000);
    let wport = reserve_c_hmmpgmd_port(cport.saturating_add(1));

    let mut master_cmd = if rust_master {
        let mut command = Command::new(hmmer());
        command.arg("pgmd");
        command
    } else {
        Command::new(c_hmmpgmd())
    };
    let mut master = master_cmd
        .args([
            "--master",
            "--hmmdb",
            hmmdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (body_tx, body_rx) = std::sync::mpsc::channel();
    let fake_worker = std::thread::spawn(move || {
        let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
        let mut header = [0u8; 12];
        stream.read_exact(&mut header).unwrap();
        let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
        let mut init_body = vec![0u8; init_len];
        stream.read_exact(&mut init_body).unwrap();
        write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
        stream.write_all(&[0u8; 96]).unwrap();
        stream.flush().unwrap();
        ready_tx.send(()).unwrap();

        stream.read_exact(&mut header).unwrap();
        let search_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10002);
        assert_eq!(u32::from_ne_bytes(header[8..12].try_into().unwrap()), 0);
        let mut search_body = vec![0u8; search_len];
        stream.read_exact(&mut search_body).unwrap();
        assert_eq!(search_len, search_body.len());
        body_tx.send(search_body).unwrap();
    });

    if let Err(err) = connect_with_deadline(cport, Duration::from_secs(5)) {
        let _ = master.kill();
        let output = master.wait_with_output().unwrap();
        panic!(
            "hmmpgmd master did not accept client probe on {cport}: {err}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    std::thread::sleep(Duration::from_millis(250));

    let request = format!("@--hmmdb 1{option_suffix}\n>query\nACDEFGHIKLMNPQRSTVWY\n//\n");
    let mut stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    stream.flush().unwrap();

    let body = body_rx.recv_timeout(Duration::from_secs(5)).unwrap();

    let _ = master.kill();
    fake_worker.join().unwrap();
    let master_output = master.wait_with_output().unwrap();
    assert!(
        master_output.status.success() || master_output.status.code().is_none(),
        "master failed before request capture; stdout={}; stderr={}",
        String::from_utf8_lossy(&master_output.stdout),
        String::from_utf8_lossy(&master_output.stderr)
    );

    body
}

#[test]
fn hmmpgmd_master_drops_malformed_worker_payload_and_falls_back() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("master-targets.fa");
    std::fs::write(&seqdb, ">fallback_target\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let client_listener = bind_hmmpgmd_listener();
    let cport = client_listener.local_addr().unwrap().port();
    drop(client_listener);

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(worker_listener);

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let _ = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();

    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let fake_worker = std::thread::spawn(move || {
        let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
        let mut header = [0u8; 12];
        stream.read_exact(&mut header).unwrap();
        let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
        let mut init_body = vec![0u8; init_len];
        stream.read_exact(&mut init_body).unwrap();
        write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
        stream.write_all(&[0u8; 96]).unwrap();
        stream.flush().unwrap();
        ready_tx.send(()).unwrap();

        stream.read_exact(&mut header).unwrap();
        let search_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        let command = u32::from_ne_bytes(header[4..8].try_into().unwrap());
        let mut search_body = vec![0u8; search_len];
        stream.read_exact(&mut search_body).unwrap();
        assert_eq!(command, 10001);
        stream.write_all(&0u32.to_be_bytes()).unwrap();
        stream.write_all(&4u64.to_be_bytes()).unwrap();
        stream.write_all(b"bad!").unwrap();
        stream.flush().unwrap();
    });

    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    std::thread::sleep(Duration::from_millis(250));

    let hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let request = format!("@--seqdb 1\n{hmm}");
    let mut stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut status = [0u8; 12];
    stream.read_exact(&mut status).unwrap();
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0u8; msg_size as usize];
    stream.read_exact(&mut payload).unwrap();

    fake_worker.join().unwrap();

    let mut shutdown_stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    shutdown_stream.write_all(b"!shutdown\n//\n").unwrap();
    let mut shutdown_status = [0u8; 12];
    shutdown_stream.read_exact(&mut shutdown_status).unwrap();

    let master_status = wait_for_child(&mut master, Duration::from_secs(5));
    let master_output = master.wait_with_output().unwrap();

    assert!(
        master_status
            .map(|status| status.success())
            .unwrap_or(false),
        "{}",
        String::from_utf8_lossy(&master_output.stderr)
    );
    assert_eq!(code, 0);
    assert_eq!(msg_size as usize, payload.len());
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 1);
    assert_eq!(hmmpgmd_first_hit_name(&payload), "fallback_target");
    assert!(
        String::from_utf8_lossy(&master_output.stderr)
            .contains("Dropping failed worker connection: short worker search payload"),
        "{}",
        String::from_utf8_lossy(&master_output.stderr)
    );
}

#[test]
fn hmmpgmd_master_drops_worker_payload_with_malformed_alignment() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("master-targets.fa");
    std::fs::write(&seqdb, ">fallback_target\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let client_listener = bind_hmmpgmd_listener();
    let cport = client_listener.local_addr().unwrap().port();
    drop(client_listener);

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(worker_listener);

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let _ = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();

    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let fake_worker = std::thread::spawn(move || {
        let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
        let mut header = [0u8; 12];
        stream.read_exact(&mut header).unwrap();
        let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
        let mut init_body = vec![0u8; init_len];
        stream.read_exact(&mut init_body).unwrap();
        write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
        stream.write_all(&[0u8; 96]).unwrap();
        stream.flush().unwrap();
        ready_tx.send(()).unwrap();

        stream.read_exact(&mut header).unwrap();
        let search_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        let command = u32::from_ne_bytes(header[4..8].try_into().unwrap());
        let mut search_body = vec![0u8; search_len];
        stream.read_exact(&mut search_body).unwrap();
        assert_eq!(command, 10001);
        let payload = hmmpgmd_fake_worker_payload_with_malformed_alignment("bad_alignment");
        stream.write_all(&0u32.to_be_bytes()).unwrap();
        stream
            .write_all(&(payload.len() as u64).to_be_bytes())
            .unwrap();
        stream.write_all(&payload).unwrap();
        stream.flush().unwrap();
    });

    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    std::thread::sleep(Duration::from_millis(250));

    let hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let request = format!("@--seqdb 1\n{hmm}");
    let mut stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut status = [0u8; 12];
    stream.read_exact(&mut status).unwrap();
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0u8; msg_size as usize];
    stream.read_exact(&mut payload).unwrap();

    fake_worker.join().unwrap();

    let mut shutdown_stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    shutdown_stream.write_all(b"!shutdown\n//\n").unwrap();
    let mut shutdown_status = [0u8; 12];
    shutdown_stream.read_exact(&mut shutdown_status).unwrap();

    let master_status = wait_for_child(&mut master, Duration::from_secs(5));
    let master_output = master.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&master_output.stderr);

    assert!(
        master_status
            .map(|status| status.success())
            .unwrap_or(false),
        "{stderr}"
    );
    assert_eq!(code, 0);
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 1);
    assert_eq!(hmmpgmd_first_hit_name(&payload), "fallback_target");
    assert!(
        stderr.contains("Dropping failed worker connection: worker alignment"),
        "{stderr}"
    );
}

#[test]
fn hmmpgmd_master_recovers_from_unknown_control_before_shutdown() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    std::fs::write(&seqdb, ">target1\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let client_listener = bind_hmmpgmd_listener();
    let cport = client_listener.local_addr().unwrap().port();
    drop(client_listener);

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(worker_listener);

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match connect_with_deadline(cport, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = master.kill();
            let output = master.wait_with_output().unwrap();
            panic!(
                "hmmpgmd --master did not accept control connection: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    stream
        .write_all(b"!not-a-command\n//\n!shutdown\n//\n")
        .unwrap();

    let mut error_status = [0u8; 12];
    stream.read_exact(&mut error_status).unwrap();
    let error_code = u32::from_be_bytes(error_status[0..4].try_into().unwrap());
    let error_size = u64::from_be_bytes(error_status[4..12].try_into().unwrap());
    let mut error_payload = vec![0u8; error_size as usize];
    stream.read_exact(&mut error_payload).unwrap();

    let mut shutdown_status = [0u8; 12];
    stream.read_exact(&mut shutdown_status).unwrap();
    let shutdown_code = u32::from_be_bytes(shutdown_status[0..4].try_into().unwrap());
    let shutdown_size = u64::from_be_bytes(shutdown_status[4..12].try_into().unwrap());
    let mut shutdown_payload = vec![0u8; shutdown_size as usize];
    stream.read_exact(&mut shutdown_payload).unwrap();

    let master_status = wait_for_child(&mut master, Duration::from_secs(5));
    let master_output = master.wait_with_output().unwrap();

    assert!(
        master_status
            .map(|status| status.success())
            .unwrap_or(false),
        "{}",
        String::from_utf8_lossy(&master_output.stderr)
    );
    assert_eq!(error_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&error_payload),
        "Unknown server command"
    );
    assert_eq!(shutdown_code, 0);
    assert_eq!(shutdown_size as usize, 122);
    assert_eq!(shutdown_payload.len(), 122);
}

#[test]
fn hmmpgmd_master_uses_reconnected_worker_after_failed_search() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let master_seqdb = dir.path().join("master-targets.fa");
    std::fs::write(&master_seqdb, ">master_only\nYYYYYYYYYYYYYYYYYYYY\n").unwrap();

    let client_listener = bind_hmmpgmd_listener();
    let cport = client_listener.local_addr().unwrap().port();
    drop(client_listener);

    let worker_listener = bind_hmmpgmd_listener();
    let wport = worker_listener.local_addr().unwrap().port();
    drop(worker_listener);

    let mut master = Command::new(hmmer())
        .args([
            "pgmd",
            "--master",
            "--seqdb",
            master_seqdb.to_str().unwrap(),
            "--cport",
            &cport.to_string(),
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let _ = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();

    let (bad_ready_tx, bad_ready_rx) = std::sync::mpsc::channel();
    let (search_seen_tx, search_seen_rx) = std::sync::mpsc::channel();
    let (replacement_ready_tx, replacement_ready_rx) = std::sync::mpsc::channel();

    let bad_worker = std::thread::spawn(move || {
        let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
        let mut header = [0u8; 12];
        stream.read_exact(&mut header).unwrap();
        let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
        let mut init_body = vec![0u8; init_len];
        stream.read_exact(&mut init_body).unwrap();
        write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
        stream.write_all(&[0u8; 96]).unwrap();
        stream.flush().unwrap();
        bad_ready_tx.send(()).unwrap();

        stream.read_exact(&mut header).unwrap();
        let search_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        let command = u32::from_ne_bytes(header[4..8].try_into().unwrap());
        let mut search_body = vec![0u8; search_len];
        stream.read_exact(&mut search_body).unwrap();
        assert_eq!(command, 10001);
        search_seen_tx.send(()).unwrap();
        replacement_ready_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        std::thread::sleep(Duration::from_millis(100));

        stream.write_all(&0u32.to_be_bytes()).unwrap();
        stream.write_all(&4u64.to_be_bytes()).unwrap();
        stream.write_all(b"bad!").unwrap();
        stream.flush().unwrap();
    });

    bad_ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    std::thread::sleep(Duration::from_millis(250));

    let hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let request = format!("@--seqdb 1\n{hmm}");
    let client = std::thread::spawn(move || {
        let mut stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut status = [0u8; 12];
        stream.read_exact(&mut status).unwrap();
        let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
        let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
        let mut payload = vec![0u8; msg_size as usize];
        stream.read_exact(&mut payload).unwrap();
        (code, payload)
    });

    search_seen_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let replacement_worker = std::thread::spawn(move || {
        let mut stream = connect_with_deadline(wport, Duration::from_secs(5)).unwrap();
        let mut header = [0u8; 12];
        stream.read_exact(&mut header).unwrap();
        let init_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10003);
        let mut init_body = vec![0u8; init_len];
        stream.read_exact(&mut init_body).unwrap();
        write_hmmd_header(&mut stream, 96, 10003, 0).unwrap();
        stream.write_all(&[0u8; 96]).unwrap();
        stream.flush().unwrap();
        replacement_ready_tx.send(()).unwrap();

        stream.read_exact(&mut header).unwrap();
        let search_len = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
        let command = u32::from_ne_bytes(header[4..8].try_into().unwrap());
        let mut search_body = vec![0u8; search_len];
        stream.read_exact(&mut search_body).unwrap();
        assert_eq!(command, 10001);
        let payload = hmmpgmd_fake_worker_payload("replacement_target");
        stream.write_all(&0u32.to_be_bytes()).unwrap();
        stream
            .write_all(&(payload.len() as u64).to_be_bytes())
            .unwrap();
        stream.write_all(&payload).unwrap();
        stream.flush().unwrap();

        stream.read_exact(&mut header).unwrap();
        assert_eq!(u32::from_ne_bytes(header[4..8].try_into().unwrap()), 10004);
        write_hmmd_header(&mut stream, 0, 10004, 0).unwrap();
        stream.flush().unwrap();
    });

    let (code, payload) = client.join().unwrap();
    bad_worker.join().unwrap();

    let mut shutdown_stream = connect_with_deadline(cport, Duration::from_secs(5)).unwrap();
    shutdown_stream.write_all(b"!shutdown\n//\n").unwrap();
    let mut shutdown_status = [0u8; 12];
    shutdown_stream.read_exact(&mut shutdown_status).unwrap();

    replacement_worker.join().unwrap();
    let master_status = wait_for_child(&mut master, Duration::from_secs(5));
    let master_output = master.wait_with_output().unwrap();

    assert!(
        master_status
            .map(|status| status.success())
            .unwrap_or(false),
        "{}",
        String::from_utf8_lossy(&master_output.stderr)
    );
    assert_eq!(
        code,
        0,
        "master returned error payload: {}",
        String::from_utf8_lossy(&payload)
    );
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 1);
    assert_eq!(hmmpgmd_first_hit_name(&payload), "replacement_target");
    let stderr = String::from_utf8_lossy(&master_output.stderr);
    assert!(
        stderr.contains("Dropping failed worker connection: short worker search payload"),
        "{}",
        stderr
    );
}

fn hmmpgmd_fake_worker_payload(name: &str) -> Vec<u8> {
    hmmpgmd_fake_worker_payload_with_stats(name, 0.0, 1.0, 1)
}

fn hmmpgmd_fake_worker_payload_with_stats(
    name: &str,
    sortkey: f64,
    score: f32,
    nincluded: i32,
) -> Vec<u8> {
    hmmpgmd_fake_worker_payload_with_filter_stats(name, sortkey, score, 1, nincluded, [0; 4])
}

fn hmmpgmd_fake_worker_payload_with_filter_stats(
    name: &str,
    sortkey: f64,
    score: f32,
    nreported: u64,
    nincluded: i32,
    filter_stats: [u64; 4],
) -> Vec<u8> {
    let mut payload = vec![0u8; 122];
    payload[42..50].copy_from_slice(&1u64.to_be_bytes());
    payload[50..58].copy_from_slice(&1u64.to_be_bytes());
    payload[58..66].copy_from_slice(&filter_stats[0].to_be_bytes());
    payload[66..74].copy_from_slice(&filter_stats[1].to_be_bytes());
    payload[74..82].copy_from_slice(&filter_stats[2].to_be_bytes());
    payload[82..90].copy_from_slice(&filter_stats[3].to_be_bytes());
    payload[90..98].copy_from_slice(&1u64.to_be_bytes());
    payload[98..106].copy_from_slice(&nreported.to_be_bytes());
    let included_count = if nincluded > 0 { 1u64 } else { 0u64 };
    payload[106..114].copy_from_slice(&included_count.to_be_bytes());
    payload[114..122].copy_from_slice(&0u64.to_be_bytes());

    let mut hit = vec![0u8; 109 + name.len() + 1];
    let hit_len = hit.len() as u32;
    hit[0..4].copy_from_slice(&hit_len.to_be_bytes());
    hit[8..16].copy_from_slice(&sortkey.to_bits().to_be_bytes());
    hit[16..20].copy_from_slice(&score.to_bits().to_be_bytes());
    hit[80..84].copy_from_slice(&1i32.to_be_bytes());
    hit[84..88].copy_from_slice(&nincluded.to_be_bytes());
    hit[109..109 + name.len()].copy_from_slice(name.as_bytes());
    payload.extend_from_slice(&hit);
    payload
}

fn hmmpgmd_fake_c_worker_payload_with_stats(
    name: &str,
    sortkey: f64,
    score: f32,
    nreported: u64,
    nincluded: i32,
    filter_stats: [u64; 4],
) -> Vec<u8> {
    let mut payload = hmmpgmd_fake_worker_payload_with_filter_stats(
        name,
        sortkey,
        score,
        nreported,
        nincluded,
        filter_stats,
    );
    payload[114..122].copy_from_slice(&u64::MAX.to_be_bytes());
    payload
}

fn hmmpgmd_fake_worker_payload_with_malformed_alignment(name: &str) -> Vec<u8> {
    let mut payload = vec![0u8; 122];
    payload[42..50].copy_from_slice(&1u64.to_be_bytes());
    payload[50..58].copy_from_slice(&1u64.to_be_bytes());
    payload[90..98].copy_from_slice(&1u64.to_be_bytes());
    payload[98..106].copy_from_slice(&1u64.to_be_bytes());
    payload[106..114].copy_from_slice(&1u64.to_be_bytes());
    payload[114..122].copy_from_slice(&0u64.to_be_bytes());

    let mut hit = vec![0u8; 109 + name.len() + 1];
    let hit_len = hit.len() as u32;
    hit[0..4].copy_from_slice(&hit_len.to_be_bytes());
    hit[16..20].copy_from_slice(&1.0f32.to_bits().to_be_bytes());
    hit[72..76].copy_from_slice(&1i32.to_be_bytes());
    hit[109..109 + name.len()].copy_from_slice(name.as_bytes());
    payload.extend_from_slice(&hit);

    let mut domain = vec![0u8; 92];
    domain[0..4].copy_from_slice(&92u32.to_be_bytes());
    payload.extend_from_slice(&domain);

    let mut alidisplay = vec![0u8; 45];
    alidisplay[0..4].copy_from_slice(&45u32.to_be_bytes());
    alidisplay[44] = 0x10;
    payload.extend_from_slice(&alidisplay);
    payload
}

fn hmmpgmd_seqdb_init_body(seqdb: &std::path::Path, nseqs: u32) -> Vec<u8> {
    let mut body = vec![0u8; 88];
    body[64..68].copy_from_slice(&0u32.to_ne_bytes());
    body[72..76].copy_from_slice(&1u32.to_ne_bytes());
    body[76..80].copy_from_slice(&nseqs.to_ne_bytes());
    body.extend_from_slice(seqdb.to_string_lossy().as_bytes());
    body.push(0);
    body
}

fn hmmpgmd_hmmdb_init_body(hmmdb: &std::path::Path, nmodels: u32) -> Vec<u8> {
    let mut body = vec![0u8; 88];
    body[68..72].copy_from_slice(&0u32.to_ne_bytes());
    body[80..84].copy_from_slice(&1u32.to_ne_bytes());
    body[84..88].copy_from_slice(&nmodels.to_ne_bytes());
    body.extend_from_slice(hmmdb.to_string_lossy().as_bytes());
    body.push(0);
    body
}

fn hmmpgmd_worker_ascii_seqdb_search(
    stream: &mut TcpStream,
    hmm: &str,
    option_suffix: &str,
) -> (u32, Vec<u8>) {
    let request = format!("@--seqdb 1{option_suffix}\n{hmm}");
    write_hmmd_header(stream, request.len() as u32, 10001, 0).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    stream.flush().unwrap();

    let mut status = [0u8; 12];
    stream.read_exact(&mut status).unwrap();
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0u8; msg_size as usize];
    stream.read_exact(&mut payload).unwrap();
    assert_eq!(msg_size as usize, payload.len());
    (code, payload)
}

#[test]
fn hmmpgmd_worker_executes_seqdb_search_frame() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    std::fs::write(&seqdb, ">target1\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let listener = bind_hmmpgmd_listener();
    let wport = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();

    let mut worker = Command::new(hmmer())
        .args([
            "pgmd",
            "--worker",
            "127.0.0.1",
            "--cpu",
            "3",
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match accept_with_deadline(&listener, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = worker.kill();
            let output = worker.wait_with_output().unwrap();
            panic!(
                "hmmpgmd worker did not connect: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    let init_body = hmmpgmd_seqdb_init_body(&seqdb, 1);
    write_hmmd_header(&mut stream, init_body.len() as u32, 10003, 0).unwrap();
    stream.write_all(&init_body).unwrap();
    let mut init_header = [0u8; 12];
    stream.read_exact(&mut init_header).unwrap();
    assert_eq!(
        u32::from_ne_bytes(init_header[0..4].try_into().unwrap()),
        96
    );
    assert_eq!(
        u32::from_ne_bytes(init_header[4..8].try_into().unwrap()),
        10003
    );
    assert_eq!(
        u32::from_ne_bytes(init_header[8..12].try_into().unwrap()),
        0
    );
    let mut init_body = vec![0u8; 96];
    stream.read_exact(&mut init_body).unwrap();
    assert_eq!(init_body, [0u8; 96]);

    let hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let (code, payload) = hmmpgmd_worker_ascii_seqdb_search(&mut stream, &hmm, "");
    let (strict_t_code, strict_t_payload) =
        hmmpgmd_worker_ascii_seqdb_search(&mut stream, &hmm, " -T 9999");
    let (strict_e_code, strict_e_payload) =
        hmmpgmd_worker_ascii_seqdb_search(&mut stream, &hmm, " -E 0");
    let (strict_dom_e_code, strict_dom_e_payload) =
        hmmpgmd_worker_ascii_seqdb_search(&mut stream, &hmm, " --domE 0");
    let (strict_dom_t_code, strict_dom_t_payload) =
        hmmpgmd_worker_ascii_seqdb_search(&mut stream, &hmm, " --domT 9999");
    let (strict_inc_e_code, strict_inc_e_payload) =
        hmmpgmd_worker_ascii_seqdb_search(&mut stream, &hmm, " --incE 0");
    let (strict_inc_t_code, strict_inc_t_payload) =
        hmmpgmd_worker_ascii_seqdb_search(&mut stream, &hmm, " --incT 9999");
    let (filter_flags_code, filter_flags_payload) =
        hmmpgmd_worker_ascii_seqdb_search(&mut stream, &hmm, " --nobias --nonull2");
    let (max_code, max_payload) = hmmpgmd_worker_ascii_seqdb_search(&mut stream, &hmm, " --max");

    write_hmmd_header(&mut stream, 0, 10004, 0).unwrap();
    let mut shutdown_header = [0u8; 12];
    stream.read_exact(&mut shutdown_header).unwrap();
    let worker_status = wait_for_child(&mut worker, Duration::from_secs(5));
    let worker_output = worker.wait_with_output().unwrap();

    assert!(
        worker_status
            .map(|status| status.success())
            .unwrap_or(false),
        "worker failed; worker stderr={}",
        String::from_utf8_lossy(&worker_output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&worker_output.stderr).contains("Worker search CPU threads: 3"),
        "{}",
        String::from_utf8_lossy(&worker_output.stderr)
    );
    assert_eq!(code, 0);
    assert_eq!(u64::from_be_bytes(payload[42..50].try_into().unwrap()), 1);
    assert_eq!(u64::from_be_bytes(payload[50..58].try_into().unwrap()), 1);
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 1);
    assert_eq!(filter_flags_code, 0);
    assert_eq!(
        u64::from_be_bytes(filter_flags_payload[42..50].try_into().unwrap()),
        1
    );
    assert_eq!(
        u64::from_be_bytes(filter_flags_payload[50..58].try_into().unwrap()),
        1
    );
    assert_eq!(hmmpgmd_stats_u64(&filter_flags_payload, 90), 1);
    assert_eq!(hmmpgmd_first_hit_name(&filter_flags_payload), "target1");
    assert_eq!(max_code, 0);
    assert_eq!(
        u64::from_be_bytes(max_payload[42..50].try_into().unwrap()),
        1
    );
    assert_eq!(
        u64::from_be_bytes(max_payload[50..58].try_into().unwrap()),
        1
    );
    assert_eq!(hmmpgmd_stats_u64(&max_payload, 90), 1);
    assert_eq!(hmmpgmd_first_hit_name(&max_payload), "target1");
    for (strict_code, strict_payload) in [
        (strict_t_code, &strict_t_payload),
        (strict_e_code, &strict_e_payload),
    ] {
        assert_eq!(strict_code, 0);
        assert_eq!(
            u64::from_be_bytes(strict_payload[42..50].try_into().unwrap()),
            1
        );
        assert_eq!(
            u64::from_be_bytes(strict_payload[50..58].try_into().unwrap()),
            1
        );
        assert_eq!(hmmpgmd_stats_u64(strict_payload, 90), 0);
    }
    for (strict_code, strict_payload) in [
        (strict_dom_e_code, &strict_dom_e_payload),
        (strict_dom_t_code, &strict_dom_t_payload),
    ] {
        assert_eq!(strict_code, 0);
        assert_eq!(hmmpgmd_stats_u64(strict_payload, 90), 1);
        assert_eq!(hmmpgmd_first_hit_ndom(strict_payload), 1);
        assert_eq!(hmmpgmd_first_hit_nreported(strict_payload), 0);
    }
    for (strict_code, strict_payload) in [
        (strict_inc_e_code, &strict_inc_e_payload),
        (strict_inc_t_code, &strict_inc_t_payload),
    ] {
        assert_eq!(strict_code, 0);
        assert_eq!(hmmpgmd_stats_u64(strict_payload, 90), 1);
        assert_eq!(hmmpgmd_stats_u64(strict_payload, 98), 1);
        assert_eq!(hmmpgmd_stats_u64(strict_payload, 106), 0);
    }
    assert_eq!(
        u64::from_be_bytes(strict_t_payload[42..50].try_into().unwrap()),
        1
    );
    assert_eq!(
        u64::from_be_bytes(strict_t_payload[50..58].try_into().unwrap()),
        1
    );
    assert_eq!(hmmpgmd_first_hit_name(&payload), "target1");
    let (model_len, seq_len, hmm_name, hmm_acc, seq_name, seq_acc) =
        hmmpgmd_first_hit_alidisplay_metadata(&payload);
    assert_eq!(model_len, 20);
    assert_eq!(seq_len, 20);
    assert_eq!(hmm_name, "test");
    assert_eq!(hmm_acc, "");
    assert_eq!(seq_name, "target1");
    assert_eq!(seq_acc, "");
}

#[test]
fn hmmpgmd_worker_binary_seqdb_search_honors_slice() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    std::fs::write(
        &seqdb,
        ">target1\nACDEFGHIKLMNPQRSTVWY\n>target2\nACDEFGHIKLMNPQRSTVWY\n",
    )
    .unwrap();

    let listener = bind_hmmpgmd_listener();
    let wport = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();

    let mut worker = Command::new(hmmer())
        .args([
            "pgmd",
            "--worker",
            "127.0.0.1",
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match accept_with_deadline(&listener, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = worker.kill();
            let output = worker.wait_with_output().unwrap();
            panic!(
                "hmmpgmd worker did not connect: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    let init_body = hmmpgmd_seqdb_init_body(&seqdb, 2);
    write_hmmd_header(&mut stream, init_body.len() as u32, 10003, 0).unwrap();
    stream.write_all(&init_body).unwrap();
    let mut init_header = [0u8; 12];
    stream.read_exact(&mut init_header).unwrap();
    let mut init_body = vec![0u8; 96];
    stream.read_exact(&mut init_body).unwrap();

    let opts = b"@--seqdb 1";
    let hmm = std::fs::read("hmmer/testsuite/20aa.hmm").unwrap();
    let mut request = Vec::new();
    for value in [
        0u32,
        1u32,
        1u32,
        1u32,
        102u32,
        hmm.len() as u32,
        opts.len() as u32 + 1,
    ] {
        request.extend_from_slice(&value.to_ne_bytes());
    }
    request.extend_from_slice(opts);
    request.push(0);
    request.extend_from_slice(&hmm);
    write_hmmd_header(&mut stream, request.len() as u32, 10001, 0).unwrap();
    stream.write_all(&request).unwrap();
    stream.flush().unwrap();
    let (code, payload) = hmmpgmd_read_c_status_payload(&mut stream);

    write_hmmd_header(&mut stream, 0, 10004, 0).unwrap();
    let mut shutdown_header = [0u8; 12];
    stream.read_exact(&mut shutdown_header).unwrap();
    let worker_status = wait_for_child(&mut worker, Duration::from_secs(5));
    let worker_output = worker.wait_with_output().unwrap();

    assert!(
        worker_status
            .map(|status| status.success())
            .unwrap_or(false),
        "worker failed; worker stderr={}",
        String::from_utf8_lossy(&worker_output.stderr)
    );
    assert_eq!(
        code,
        0,
        "worker returned error payload: {}",
        String::from_utf8_lossy(&payload)
    );
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 1);
    assert_eq!(hmmpgmd_first_hit_name(&payload), "target2");
}

#[test]
fn hmmpgmd_worker_seqdb_hmm_model_cutoffs_error_when_absent_and_succeed_when_present() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let seqdb = dir.path().join("targets.fa");
    std::fs::write(&seqdb, ">target1\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let listener = bind_hmmpgmd_listener();
    let wport = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();

    let mut worker = Command::new(hmmer())
        .args([
            "pgmd",
            "--worker",
            "127.0.0.1",
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match accept_with_deadline(&listener, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = worker.kill();
            let output = worker.wait_with_output().unwrap();
            panic!(
                "hmmpgmd worker did not connect: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    let init_body = hmmpgmd_seqdb_init_body(&seqdb, 1);
    write_hmmd_header(&mut stream, init_body.len() as u32, 10003, 0).unwrap();
    stream.write_all(&init_body).unwrap();
    let mut init_header = [0u8; 12];
    stream.read_exact(&mut init_header).unwrap();
    let mut init_response = vec![0u8; 96];
    stream.read_exact(&mut init_response).unwrap();

    let missing_hmm = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    for (cutoff, message) in [
        ("--cut_ga", "GA cutoff not set in model"),
        ("--cut_tc", "TC cutoff not set in model"),
        ("--cut_nc", "NC cutoff not set in model"),
    ] {
        let request = format!("@--seqdb 1 {cutoff}\n{missing_hmm}");
        write_hmmd_header(&mut stream, request.len() as u32, 10001, 0).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        stream.flush().unwrap();
        let (code, payload) = hmmpgmd_read_c_status_payload(&mut stream);
        assert_eq!(code, 1, "{cutoff} unexpectedly succeeded");
        assert!(
            String::from_utf8_lossy(&payload).contains(message),
            "{}",
            String::from_utf8_lossy(&payload)
        );
    }

    let annotated_hmm = std::fs::read_to_string("hmmer/tutorial/fn3.hmm").unwrap();
    for cutoff in ["--cut_ga", "--cut_tc", "--cut_nc"] {
        let request = format!("@--seqdb 1 {cutoff}\n{annotated_hmm}");
        write_hmmd_header(&mut stream, request.len() as u32, 10001, 0).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        stream.flush().unwrap();
        let (code, payload) = hmmpgmd_read_c_status_payload(&mut stream);
        assert_eq!(
            code,
            0,
            "{cutoff} returned error payload: {}",
            String::from_utf8_lossy(&payload)
        );
    }

    write_hmmd_header(&mut stream, 0, 10004, 0).unwrap();
    let mut shutdown_header = [0u8; 12];
    stream.read_exact(&mut shutdown_header).unwrap();
    let worker_status = wait_for_child(&mut worker, Duration::from_secs(5));
    let worker_output = worker.wait_with_output().unwrap();

    assert!(
        worker_status
            .map(|status| status.success())
            .unwrap_or(false),
        "worker failed; worker stderr={}",
        String::from_utf8_lossy(&worker_output.stderr)
    );
}

#[test]
fn hmmpgmd_worker_executes_hmmdb_binary_scan_shard() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();
    let hmm_template = std::fs::read_to_string("hmmer/testsuite/20aa.hmm").unwrap();
    let hmmdb = dir.path().join("models.hmm");
    std::fs::write(
        &hmmdb,
        format!(
            "{}{}",
            hmm_template.replacen("NAME  test", "NAME  model_a", 1),
            hmm_template.replacen("NAME  test", "NAME  model_b", 1)
        ),
    )
    .unwrap();

    let listener = bind_hmmpgmd_listener();
    let wport = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();

    let mut worker = Command::new(hmmer())
        .args([
            "pgmd",
            "--worker",
            "127.0.0.1",
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match accept_with_deadline(&listener, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = worker.kill();
            let output = worker.wait_with_output().unwrap();
            panic!(
                "hmmpgmd worker did not connect: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    let init_body = hmmpgmd_hmmdb_init_body(&hmmdb, 2);
    write_hmmd_header(&mut stream, init_body.len() as u32, 10003, 0).unwrap();
    stream.write_all(&init_body).unwrap();
    let mut init_header = [0u8; 12];
    stream.read_exact(&mut init_header).unwrap();
    assert_eq!(
        u32::from_ne_bytes(init_header[0..4].try_into().unwrap()),
        96
    );
    assert_eq!(
        u32::from_ne_bytes(init_header[4..8].try_into().unwrap()),
        10003
    );
    assert_eq!(
        u32::from_ne_bytes(init_header[8..12].try_into().unwrap()),
        0
    );
    let mut init_body = vec![0u8; 96];
    stream.read_exact(&mut init_body).unwrap();
    assert_eq!(init_body, [0u8; 96]);

    let opts = b"@--hmmdb 1 --max";
    let query = [
        b"query\0\0".as_slice(),
        &[
            255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 255,
        ],
    ]
    .concat();
    let mut request = Vec::new();
    for value in [0u32, 2u32, 1u32, 1u32, 101u32, 22u32, opts.len() as u32 + 1] {
        request.extend_from_slice(&value.to_ne_bytes());
    }
    request.extend_from_slice(opts);
    request.push(0);
    request.extend_from_slice(&query);
    write_hmmd_header(&mut stream, request.len() as u32, 10002, 0).unwrap();
    stream.write_all(&request).unwrap();
    stream.flush().unwrap();

    let mut status = [0u8; 12];
    stream.read_exact(&mut status).unwrap();
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0u8; msg_size as usize];
    stream.read_exact(&mut payload).unwrap();

    write_hmmd_header(&mut stream, 0, 10004, 0).unwrap();
    let mut shutdown_header = [0u8; 12];
    stream.read_exact(&mut shutdown_header).unwrap();
    let worker_status = wait_for_child(&mut worker, Duration::from_secs(5));
    let worker_output = worker.wait_with_output().unwrap();

    assert!(
        worker_status
            .map(|status| status.success())
            .unwrap_or(false),
        "worker failed; worker stderr={}",
        String::from_utf8_lossy(&worker_output.stderr)
    );
    assert_eq!(code, 0);
    assert_eq!(msg_size as usize, payload.len());
    assert_eq!(u64::from_be_bytes(payload[42..50].try_into().unwrap()), 2);
    assert_eq!(u64::from_be_bytes(payload[50..58].try_into().unwrap()), 1);
    assert_eq!(hmmpgmd_stats_u64(&payload, 90), 1);
    assert_eq!(hmmpgmd_first_hit_name(&payload), "model_b");
}

#[test]
fn hmmpgmd_worker_hmmdb_binary_scan_model_cutoffs_error_when_absent_and_succeed_when_present() {
    let _guard = hmmpgmd_test_guard();
    let dir = tempfile::tempdir().unwrap();

    let missing_hmmdb = dir.path().join("missing-cutoffs.hmm");
    std::fs::copy("hmmer/testsuite/20aa.hmm", &missing_hmmdb).unwrap();
    hmmpgmd_assert_hmmdb_binary_cutoff_scan_errors(
        &missing_hmmdb,
        [
            ("--cut_ga", "GA cutoff not set in model"),
            ("--cut_tc", "TC cutoff not set in model"),
            ("--cut_nc", "NC cutoff not set in model"),
        ],
    );

    let annotated_hmmdb = dir.path().join("annotated-cutoffs.hmm");
    std::fs::copy("hmmer/tutorial/fn3.hmm", &annotated_hmmdb).unwrap();
    hmmpgmd_assert_hmmdb_binary_cutoff_scan_succeeds(
        &annotated_hmmdb,
        ["--cut_ga", "--cut_tc", "--cut_nc"],
    );
}

fn hmmpgmd_assert_hmmdb_binary_cutoff_scan_errors(
    hmmdb: &std::path::Path,
    cutoffs: [(&str, &str); 3],
) {
    let (worker, mut stream) = hmmpgmd_start_hmmdb_worker(hmmdb, 1);
    for (cutoff, message) in cutoffs {
        let request = hmmpgmd_hmmdb_sequence_scan_request(&format!(" {cutoff}"));
        write_hmmd_header(&mut stream, request.len() as u32, 10002, 0).unwrap();
        stream.write_all(&request).unwrap();
        stream.flush().unwrap();
        let (code, payload) = hmmpgmd_read_c_status_payload(&mut stream);
        assert_eq!(code, 1, "{cutoff} unexpectedly succeeded");
        assert!(
            String::from_utf8_lossy(&payload).contains(message),
            "{}",
            String::from_utf8_lossy(&payload)
        );
    }
    hmmpgmd_shutdown_worker(worker, stream);
}

fn hmmpgmd_assert_hmmdb_binary_cutoff_scan_succeeds(hmmdb: &std::path::Path, cutoffs: [&str; 3]) {
    let (worker, mut stream) = hmmpgmd_start_hmmdb_worker(hmmdb, 1);
    for cutoff in cutoffs {
        let request = hmmpgmd_hmmdb_sequence_scan_request(&format!(" {cutoff}"));
        write_hmmd_header(&mut stream, request.len() as u32, 10002, 0).unwrap();
        stream.write_all(&request).unwrap();
        stream.flush().unwrap();
        let (code, payload) = hmmpgmd_read_c_status_payload(&mut stream);
        assert_eq!(
            code,
            0,
            "{cutoff} returned error payload: {}",
            String::from_utf8_lossy(&payload)
        );
    }
    hmmpgmd_shutdown_worker(worker, stream);
}

fn hmmpgmd_start_hmmdb_worker(
    hmmdb: &std::path::Path,
    nmodels: u32,
) -> (std::process::Child, TcpStream) {
    let listener = bind_hmmpgmd_listener();
    let wport = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();

    let mut worker = Command::new(hmmer())
        .args([
            "pgmd",
            "--worker",
            "127.0.0.1",
            "--wport",
            &wport.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stream = match accept_with_deadline(&listener, Duration::from_secs(5)) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = worker.kill();
            let output = worker.wait_with_output().unwrap();
            panic!(
                "hmmpgmd worker did not connect: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };

    let init_body = hmmpgmd_hmmdb_init_body(hmmdb, nmodels);
    write_hmmd_header(&mut stream, init_body.len() as u32, 10003, 0).unwrap();
    stream.write_all(&init_body).unwrap();
    let mut init_header = [0u8; 12];
    stream.read_exact(&mut init_header).unwrap();
    assert_eq!(
        u32::from_ne_bytes(init_header[0..4].try_into().unwrap()),
        96
    );
    let mut init_response = vec![0u8; 96];
    stream.read_exact(&mut init_response).unwrap();
    (worker, stream)
}

fn hmmpgmd_hmmdb_sequence_scan_request(option_suffix: &str) -> Vec<u8> {
    let opts = format!("@--hmmdb 1{option_suffix}");
    let query = [
        b"query\0\0".as_slice(),
        &[
            255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 255,
        ],
    ]
    .concat();
    let mut request = Vec::new();
    for value in [0u32, 1u32, 0u32, 1u32, 101u32, 22u32, opts.len() as u32 + 1] {
        request.extend_from_slice(&value.to_ne_bytes());
    }
    request.extend_from_slice(opts.as_bytes());
    request.push(0);
    request.extend_from_slice(&query);
    request
}

fn hmmpgmd_shutdown_worker(mut worker: std::process::Child, mut stream: TcpStream) {
    write_hmmd_header(&mut stream, 0, 10004, 0).unwrap();
    let mut shutdown_header = [0u8; 12];
    stream.read_exact(&mut shutdown_header).unwrap();
    let worker_status = wait_for_child(&mut worker, Duration::from_secs(5));
    let worker_output = worker.wait_with_output().unwrap();

    assert!(
        worker_status
            .map(|status| status.success())
            .unwrap_or(false),
        "worker failed; worker stderr={}",
        String::from_utf8_lossy(&worker_output.stderr)
    );
}

fn connect_with_deadline(port: u16, timeout: Duration) -> std::io::Result<TcpStream> {
    let start = Instant::now();
    let addr = ("127.0.0.1", port);
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => return Ok(stream),
            Err(err) if start.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(25));
                if err.kind() == std::io::ErrorKind::ConnectionRefused {
                    continue;
                }
            }
            Err(err) => return Err(err),
        }
    }
}

fn wait_for_child(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Option<std::process::ExitStatus> {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return Some(status);
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            return None;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

fn accept_with_deadline(listener: &TcpListener, timeout: Duration) -> std::io::Result<TcpStream> {
    let start = Instant::now();
    loop {
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(err)
                if start.elapsed() < timeout && err.kind() == std::io::ErrorKind::WouldBlock =>
            {
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(err) => return Err(err),
        }
    }
}

fn write_hmmd_header(
    stream: &mut TcpStream,
    length: u32,
    command: u32,
    status: u32,
) -> std::io::Result<()> {
    stream.write_all(&length.to_ne_bytes())?;
    stream.write_all(&command.to_ne_bytes())?;
    stream.write_all(&status.to_ne_bytes())
}

fn hmmpgmd_read_c_status_payload(stream: &mut TcpStream) -> (u32, Vec<u8>) {
    let mut status = [0u8; 12];
    stream.read_exact(&mut status).unwrap();
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap()) as usize;
    let mut payload = vec![0u8; msg_size];
    stream.read_exact(&mut payload).unwrap();
    (code, payload)
}

#[test]
fn hmmbuild_hand_uses_rf_architecture() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("hand.sto");
    let hmm_out = dir.path().join("hand.hmm");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID handcase\ns1 ACDEFG\ns2 AC-EFG\n#=GC RF xx..xx\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--hand",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# model architecture construction:  hand-specified by RF annotation\n")
    );
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.contains("NAME  handcase\n"));
    assert!(hmm.contains("LENG  4\n"));
}

#[test]
fn hmmbuild_autodetects_dna_and_prints_w_column() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("dna.sto");
    let hmm_out = dir.path().join("dna.hmm");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID dnacase\ns1 ACGTACGTACGT\ns2 ACGTACGTACGT\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args(["build", hmm_out.to_str().unwrap(), sto.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("mlen     W eff_nseq"));
    let hmm = std::fs::read_to_string(hmm_out).unwrap();
    assert!(hmm.contains("ALPH  DNA\n"));
    assert!(hmm.contains("MAXL  "));
}

/// Audit F3 + F4 (04-scan-iterative-drivers): hmmscan on a model that has no
/// DESC must print the empty description after the name (">> globins4  " with a
/// trailing space and no "-"), matching C `p7_tophits_Domains`; and the tabular
/// footer's "Option settings" must strip the `hmmer` wrapper token to read
/// `hmmscan ...`, matching the phmmer/jackhmmer footers.
#[test]
fn hmmscan_empty_desc_has_no_dash_and_footer_strips_wrapper() {
    let dir = tempfile::tempdir().unwrap();
    let hmmdb = dir.path().join("globins4.hmm");
    std::fs::copy("hmmer/tutorial/globins4.hmm", &hmmdb).unwrap();
    let press = Command::new(hmmer())
        .args(["press", "-f", hmmdb.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        press.status.success(),
        "{}",
        String::from_utf8_lossy(&press.stderr)
    );
    let tblout = dir.path().join("hits.tbl");
    let output = Command::new(hmmer())
        .args([
            "scan",
            "--tblout",
            tblout.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/HBB_HUMAN",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();

    // F3: empty DESC => ">> globins4  " (name + two spaces, then nothing), not
    // ">> globins4  -".
    assert!(
        stdout.lines().any(|l| l == ">> globins4  "),
        "missing empty-desc >> line; got: {:?}",
        stdout.lines().find(|l| l.starts_with(">> globins4"))
    );
    assert!(
        !stdout.contains(">> globins4  -"),
        "empty desc must not render as a dash"
    );

    // F4: tabular footer must read "hmmscan ...", not "hmmer hmmscan ...".
    let tbl = std::fs::read_to_string(&tblout).unwrap();
    let footer = tbl
        .lines()
        .find(|l| l.starts_with("# Option settings:"))
        .expect("missing Option settings footer");
    assert!(
        footer.starts_with("# Option settings: hmmscan "),
        "footer must strip wrapper token: {footer}"
    );
    assert!(
        !footer.contains("hmmer hmmscan"),
        "footer must not keep the hmmer wrapper token: {footer}"
    );
}
