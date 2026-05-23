use std::io::Write;
use std::process::Command;
use std::process::Stdio;

use hmmer_pure_rs::hmmfile_binary;

fn hmmer() -> &'static str {
    env!("CARGO_BIN_EXE_hmmer")
}

fn project_root() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
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
    let output = Command::new(hmmer())
        .args([
            "search",
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
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
fn hmmscan_tblout_has_scan_footer_and_database_header() {
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
    let output = Command::new(hmmer())
        .args([
            "scan",
            "--noali",
            "--tblout",
            tblout.to_str().unwrap(),
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# query sequence file:             hmmer/tutorial/7LESS_DROME\n"));
    assert!(stdout.contains(&format!(
        "# target HMM database:             {}\n",
        hmmdb.display()
    )));

    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert!(tbl.contains("# Program:         hmmscan\n"));
    assert!(tbl.contains("# Pipeline mode:   SCAN\n"));
    assert!(tbl.contains("# Query file:      hmmer/tutorial/7LESS_DROME\n"));
    assert!(tbl.contains(&format!("# Target file:     {}\n", hmmdb.display())));
    assert!(tbl.ends_with("# [ok]\n"));
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
fn makehmmerdb_accepts_c_style_output_positional() {
    let dir = tempfile::tempdir().unwrap();
    let seq = dir.path().join("seq.fa");
    let db = dir.path().join("seq.hmmerdb");
    std::fs::write(&seq, ">s1\nACGTACGT\n>s2\nTTTT\n").unwrap();

    let output = Command::new(hmmer())
        .args(["makehmmerdb", seq.to_str().unwrap(), db.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(db).unwrap();
    assert!(bytes.starts_with(b"HMMERDB\0"));
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
        stdout.lines().filter(|line| !line.starts_with('#')).count(),
        3
    );
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
    let fetched = std::fs::read_to_string(out).unwrap();
    assert!(fetched.contains("NAME  Caudal_act"));
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
fn hmmbuild_accepts_stockholm_stdin_with_informat() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_out = dir.path().join("stdin-build.hmm");
    let sto = std::fs::read("hmmer/testsuite/20aa.sto").unwrap();

    let output = run_with_stdin(
        &[
            "build",
            "--informat",
            "stockholm",
            hmm_out.to_str().unwrap(),
            "-",
        ],
        &sto,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(std::fs::read_to_string(hmm_out)
        .unwrap()
        .contains("NAME  test"));
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
    let dir = tempfile::tempdir().unwrap();
    let masked = dir.path().join("masked.sto");
    let sto = std::fs::read("hmmer/testsuite/20aa.sto").unwrap();

    let output = run_with_stdin(
        &[
            "alimask",
            "--informat",
            "stockholm",
            "--alirange",
            "1..20",
            "-",
            masked.to_str().unwrap(),
        ],
        &sto,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let masked = std::fs::read_to_string(masked).unwrap();
    assert!(masked.contains("#=GC MM mmmmmmmmmmmmmmmmmmmm"));
}

#[test]
fn alimask_alirange_writes_alignment_length_model_mask() {
    let dir = tempfile::tempdir().unwrap();
    let masked = dir.path().join("masked.sto");

    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--alirange",
            "2..4,7..7",
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
    assert!(masked.contains("#=GC MM .mmm..m............."));
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
