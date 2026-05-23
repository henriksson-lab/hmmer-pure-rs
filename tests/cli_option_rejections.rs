use std::io::Write;
use std::process::Command;
use std::process::Stdio;

fn hmmer() -> &'static str {
    env!("CARGO_BIN_EXE_hmmer")
}

fn write_bad_sequence_pair(dir: &tempfile::TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
    let good = dir.path().join("good.fa");
    let bad = dir.path().join("bad.fa");
    std::fs::write(&good, ">q\nACDEFGHIKLMNPQRSTVWY\n").unwrap();
    std::fs::write(&bad, ">bad\nACD#EF\n").unwrap();
    (good, bad)
}

#[test]
fn hmmsearch_tformat_rejects_non_fasta_explicitly() {
    let output = Command::new(hmmer())
        .args([
            "search",
            "--tformat",
            "stockholm",
            "missing-query.hmm",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("hmmsearch --tformat=stockholm is not implemented"));
}

#[test]
fn hmmsearch_tformat_fasta_rejects_swissprot_input() {
    let output = Command::new(hmmer())
        .args([
            "search",
            "--tformat",
            "fasta",
            "--noali",
            "hmmer/tutorial/fn3.hmm",
            "hmmer/tutorial/7LESS_DROME",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("# targ <seqfile> format asserted:  fasta"),
        "{stdout}"
    );
    assert!(stdout.contains("Query:       fn3"), "{stdout}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("unrecognized sequence file record start"),
        "{stderr}"
    );
}

#[test]
fn hmmsearch_rejects_conflicting_model_cutoffs_before_io() {
    let output = Command::new(hmmer())
        .args([
            "search",
            "--cut_ga",
            "--cut_tc",
            "missing-query.hmm",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("cannot be used with"));
}

#[test]
fn hmmscan_rejects_conflicting_model_cutoffs_before_io() {
    let output = Command::new(hmmer())
        .args([
            "scan",
            "--cut_ga",
            "--cut_nc",
            "missing-models.hmm",
            "missing-queries.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("cannot be used with"));
}

#[test]
fn nhmmer_tformat_rejects_non_fasta_explicitly() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--tformat",
            "stockholm",
            "missing-query.hmm",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("nhmmer --tformat=stockholm is not implemented"));
}

#[test]
fn nhmmer_tformat_fasta_rejects_non_fasta_target_without_panic() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--tformat",
            "fasta",
            "--noali",
            "hmmer/testsuite/ecori.hmm",
            "hmmer/easel/formats/genbank",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Query:       ecori"), "{stdout}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("unrecognized sequence file record start"),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn fasta_headers_require_nonempty_names() {
    let dir = tempfile::tempdir().unwrap();
    let blank = dir.path().join("blank-name.fa");
    std::fs::write(&blank, ">\nACDEFGHIK\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "search",
            "--noali",
            "hmmer/testsuite/20aa.hmm",
            blank.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("no FASTA name found"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn nhmmer_rejects_conflicting_alphabet_assertions_before_io() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--dna",
            "--rna",
            "missing-query.hmm",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("options --dna and --rna are mutually exclusive"));
}

#[test]
fn nhmmer_rejects_invalid_window_length_before_io() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--w_length",
            "3",
            "missing-query.hmm",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Invalid window length value"));
}

#[test]
fn search_commands_reject_text_width_conflicts_before_io() {
    for subcmd in ["search", "nhmmer"] {
        let output = Command::new(hmmer())
            .args([
                subcmd,
                "--notextw",
                "--textw",
                "120",
                "missing-query.hmm",
                "missing-targets.fa",
            ])
            .output()
            .unwrap();

        assert!(
            !output.status.success(),
            "{subcmd} accepted --notextw --textw"
        );
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("cannot be used with"), "{stderr}");
    }
}

#[test]
fn commands_reject_too_narrow_text_width_before_io() {
    for subcmd in ["search", "scan", "nhmmer", "nhmmscan"] {
        let output = Command::new(hmmer())
            .args([
                subcmd,
                "--textw",
                "80",
                "missing-query.hmm",
                "missing-targets.fa",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success(), "{subcmd} accepted --textw 80");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("120"), "{stderr}");
    }
}

#[test]
fn nhmmer_rejects_conflicting_strand_options_before_io() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--watson",
            "--crick",
            "missing-query.hmm",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("cannot be used with"));
}

#[test]
fn search_commands_reject_threshold_conflicts_before_io() {
    for subcmd in [
        "search",
        "scan",
        "nhmmer",
        "nhmmscan",
        "phmmer",
        "jackhmmer",
    ] {
        let output = Command::new(hmmer())
            .args([
                subcmd,
                "-E",
                "1",
                "-T",
                "10",
                "missing-query.hmm",
                "missing-targets.fa",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success(), "{subcmd} accepted -E with -T");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("cannot be used with"), "{stderr}");
    }
}

#[test]
fn search_commands_reject_nonpositive_evalue_space_before_io() {
    for subcmd in [
        "search",
        "scan",
        "nhmmer",
        "nhmmscan",
        "phmmer",
        "jackhmmer",
    ] {
        let output = Command::new(hmmer())
            .args([subcmd, "-E", "0", "missing-query.hmm", "missing-targets.fa"])
            .output()
            .unwrap();

        assert!(!output.status.success(), "{subcmd} accepted -E 0");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("value must be > 0"), "{stderr}");
    }

    for subcmd in ["search", "scan", "phmmer", "jackhmmer"] {
        let output = Command::new(hmmer())
            .args([
                subcmd,
                "--domZ",
                "0",
                "missing-query.hmm",
                "missing-targets.fa",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success(), "{subcmd} accepted --domZ 0");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("value must be > 0"), "{stderr}");
    }
}

#[test]
fn search_commands_reject_filter_and_cutoff_conflicts_before_io() {
    for subcmd in [
        "search",
        "scan",
        "nhmmer",
        "nhmmscan",
        "phmmer",
        "jackhmmer",
    ] {
        let output = Command::new(hmmer())
            .args([
                subcmd,
                "--max",
                "--F1",
                "0.1",
                "missing-query.hmm",
                "missing-targets.fa",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success(), "{subcmd} accepted --max --F1");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("cannot be used with"), "{stderr}");
    }

    for subcmd in ["search", "scan", "nhmmer", "nhmmscan"] {
        let output = Command::new(hmmer())
            .args([
                subcmd,
                "--cut_ga",
                "-E",
                "1",
                "missing-query.hmm",
                "missing-targets.fa",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success(), "{subcmd} accepted --cut_ga -E");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("cannot be used with"), "{stderr}");
    }
}

#[test]
fn search_commands_fail_on_missing_requested_model_cutoffs() {
    let output = Command::new(hmmer())
        .args([
            "search",
            "--cut_ga",
            "hmmer/testsuite/20aa.hmm",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("GA cutoff not set"), "{stderr}");

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--cut_ga",
            "hmmer/tutorial/MADE1.hmm",
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("GA cutoff not set"), "{stderr}");
}

#[test]
fn phmmer_and_jackhmmer_reject_gap_probability_ranges_before_io() {
    for subcmd in ["phmmer", "jackhmmer"] {
        for (flag, value) in [("--popen", "0.5"), ("--pextend", "1")] {
            let output = Command::new(hmmer())
                .args([
                    subcmd,
                    flag,
                    value,
                    "missing-query.fa",
                    "missing-targets.fa",
                ])
                .output()
                .unwrap();

            assert!(!output.status.success(), "{subcmd} accepted {flag} {value}");
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(stderr.contains("must be"), "{stderr}");
        }
    }
}

#[test]
fn jackhmmer_rejects_zero_iterations_before_io() {
    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "-N",
            "0",
            "missing-query.fa",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("value must be > 0"));
}

#[test]
fn phmmer_rejects_ambiguous_or_unrewindable_stdin() {
    let output = run_with_stdin(&["phmmer", "-", "-"], b"");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("may be '-' but not both"), "{stderr}");

    let target = std::fs::read("hmmer/testsuite/20aa-alitest.fa").unwrap();
    let output = run_with_stdin(&["phmmer", "hmmer/testsuite/20aa-alitest.fa", "-"], &target);
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("isn't rewindable"), "{stderr}");
}

#[test]
fn jackhmmer_rejects_target_stdin() {
    let output = run_with_stdin(&["jackhmmer", "hmmer/testsuite/20aa-alitest.fa", "-"], b"");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("may not be '-'"), "{stderr}");
}

#[test]
fn jackhmmer_rejects_multi_query_input_until_supported() {
    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "-N",
            "1",
            "hmmer/testsuite/20aa-alitest.fa",
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("multi-query sequence input is not implemented"),
        "{stderr}"
    );
}

#[test]
fn hmmpgmd_seqdb_fails_explicitly() {
    let output = Command::new(hmmer())
        .args([
            "pgmd",
            "--hmmdb",
            "missing.hmm",
            "--seqdb",
            "missing.fa",
            "--port",
            "9",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("hmmpgmd --seqdb=missing.fa is not implemented"));
}

#[test]
fn hmmsim_rejects_conflicting_scoring_modes_before_io() {
    let output = Command::new(hmmer())
        .args(["sim", "--vit", "--fwd", "missing.hmm"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr
        .contains("hmmsim scoring options --vit, --fwd, --hyb, and --msv are mutually exclusive"));
}

#[test]
fn hmmsim_rejects_zero_sample_or_length_before_io() {
    let output = Command::new(hmmer())
        .args(["sim", "-N", "0", "missing.hmm"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("-N must be > 0"));

    let output = Command::new(hmmer())
        .args(["sim", "-L", "0", "missing.hmm"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("-L must be > 0"));
}

#[test]
fn makehmmerdb_rejects_invalid_index_options_before_io() {
    let output = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--bin_length",
            "33",
            "missing.fa",
            "out.hmmerdb",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--bin_length must be a power of 2"));

    let output = Command::new(hmmer())
        .args(["makehmmerdb", "--sa_freq", "0", "missing.fa", "out.hmmerdb"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--sa_freq must be a power of 2"));
}

#[test]
fn hmmconvert_rejects_conflicting_ascii_binary_before_io() {
    let output = Command::new(hmmer())
        .args(["convert", "-a", "-b", "missing.hmm"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("options -a and -b are mutually exclusive"));
}

#[test]
fn hmmconvert_outfmt_fails_explicitly_before_io() {
    let output = Command::new(hmmer())
        .args(["convert", "--outfmt", "3/b", "missing.hmm"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("hmmconvert --outfmt=3/b is not implemented"));
}

#[test]
fn hmmconvert_ascii_outfmt_fails_explicitly_before_io() {
    let output = Command::new(hmmer())
        .args(["convert", "--outfmt", "3/a", "missing.hmm"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("hmmconvert --outfmt=3/a is not implemented"));
}

#[test]
fn search_commands_reject_empty_sequence_input() {
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty.fa");
    std::fs::write(&empty, "").unwrap();

    let output = Command::new(hmmer())
        .args(["search", "hmmer/tutorial/fn3.hmm", empty.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("no sequences found"), "{stderr}");

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "hmmer/tutorial/MADE1.hmm",
            empty.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("no sequences found"), "{stderr}");
}

#[test]
fn nhmmscan_rejects_empty_sequence_input() {
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
    let empty = dir.path().join("empty.fa");
    std::fs::write(&empty, "").unwrap();

    let output = Command::new(hmmer())
        .args(["nhmmscan", hmmdb.to_str().unwrap(), empty.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("no sequences found"), "{stderr}");
}

#[test]
fn nhmmer_rejects_amino_hmm_without_panic() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--noali",
            "hmmer/testsuite/20aa.hmm",
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("Invalid alphabet type in query for nhmmer"),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn nhmmer_rna_assertion_rejects_dna_hmm() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--rna",
            "--noali",
            "hmmer/testsuite/ecori.hmm",
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("expected RNA query HMM"), "{stderr}");
}

#[test]
fn phmmer_rejects_empty_sequence_input() {
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty.fa");
    std::fs::write(&empty, "").unwrap();

    let output = Command::new(hmmer())
        .args([
            "phmmer",
            empty.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("no sequences found"), "{stderr}");

    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "hmmer/testsuite/20aa-alitest.fa",
            empty.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("no sequences found"), "{stderr}");
}

#[test]
fn sequence_commands_reject_malformed_records_without_panic() {
    let dir = tempfile::tempdir().unwrap();
    let (good, bad) = write_bad_sequence_pair(&dir);

    let cases: Vec<(&str, Vec<String>)> = vec![
        (
            "phmmer bad query",
            vec![
                "phmmer".to_string(),
                bad.to_string_lossy().into_owned(),
                good.to_string_lossy().into_owned(),
            ],
        ),
        (
            "phmmer bad target",
            vec![
                "phmmer".to_string(),
                good.to_string_lossy().into_owned(),
                bad.to_string_lossy().into_owned(),
            ],
        ),
        (
            "hmmalign bad sequence",
            vec![
                "align".to_string(),
                "hmmer/testsuite/20aa.hmm".to_string(),
                bad.to_string_lossy().into_owned(),
            ],
        ),
        (
            "jackhmmer bad query",
            vec![
                "jackhmmer".to_string(),
                "-N".to_string(),
                "1".to_string(),
                bad.to_string_lossy().into_owned(),
                good.to_string_lossy().into_owned(),
            ],
        ),
    ];

    for (label, args) in cases {
        let output = Command::new(hmmer()).args(args).output().unwrap();
        assert!(!output.status.success(), "{label} succeeded unexpectedly");
        let stdout = String::from_utf8(output.stdout).unwrap();
        if label == "phmmer bad target" {
            assert!(stdout.contains("Query:       q  [L=20]"), "{stdout}");
        }
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("Illegal symbol"), "{label}: {stderr}");
        assert!(!stderr.contains("panicked"), "{label}: {stderr}");
    }

    let hmmdb = dir.path().join("models.hmm");
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
        .args(["scan", hmmdb.to_str().unwrap(), bad.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Illegal symbol"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn hmmalign_empty_hmm_fails_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("empty.hmm");
    let seq = dir.path().join("seq.fa");
    std::fs::write(&hmm, "").unwrap();
    std::fs::write(&seq, ">s\nACDE\n").unwrap();

    let output = Command::new(hmmer())
        .args(["align", hmm.to_str().unwrap(), seq.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("no HMMs found"));
    assert!(!stderr.contains("panicked"));
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
fn hmmalign_rejects_double_stdin_before_reading() {
    let output = run_with_stdin(&["align", "-", "-"], b"");

    assert!(!output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Either <hmmfile> or <seqfile> may be '-'"));
    assert!(!stderr.contains("panicked"));
}

#[test]
fn hmmbuild_rejects_stdout_hmm_and_requires_informat_for_stdin_alignment() {
    let output = Command::new(hmmer())
        .args(["build", "-", "hmmer/testsuite/20aa.sto"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("cannot write <hmmfile_out> to stdout"));

    let output = run_with_stdin(&["build", "missing-output.hmm", "-"], b"");
    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Must specify --informat"));
}

#[test]
fn hmmfetch_rejects_invalid_stdin_combinations() {
    let output = run_with_stdin(&["fetch", "--index", "-"], b"");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Can't use - with --index"));

    let output = run_with_stdin(&["fetch", "-f", "-", "-"], b"");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Either <hmmfile> or <keyfile> can be - but not both"));
}

#[test]
fn alimask_requires_informat_for_stdin_alignment() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("masked.sto");
    let output = run_with_stdin(
        &["alimask", "--alirange", "1..2", "-", out.to_str().unwrap()],
        b"",
    );

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Must specify --informat"));
}

#[test]
fn alimask_rejects_invalid_ranges_and_mask_lengths() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("masked.sto");

    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--alirange",
            "21..22",
            "hmmer/testsuite/20aa.sto",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("exceeds alignment length"));

    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--alirange",
            "1..2",
            "--modelmask",
            "mmm",
            "hmmer/testsuite/20aa.sto",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("does not match alignment length"));
}

#[test]
fn alimask_rejects_missing_or_unsupported_modes_before_io() {
    let output = Command::new(hmmer())
        .args(["alimask", "missing.sto", "out.sto"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Must specify one of"));

    let output = Command::new(hmmer())
        .args(["alimask", "--model2ali", "1..3", "missing.sto"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("alimask --model2ali is not implemented"));

    let output = Command::new(hmmer())
        .args(["alimask", "--alirange", "1..3", "hmmer/testsuite/20aa.sto"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("requires <postmsafile>"));
}

#[test]
fn hmmbuild_rejects_hand_without_rf_and_invalid_architecture_options() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("no-rf.sto");
    let hmm_out = dir.path().join("out.hmm");
    std::fs::write(&sto, b"# STOCKHOLM 1.0\ns1 ACDE\ns2 ACDE\n//\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--hand",
            "--amino",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("RF line"));

    let output = Command::new(hmmer())
        .args([
            "build",
            "--hand",
            "--fast",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("mutually exclusive"));

    let output = Command::new(hmmer())
        .args([
            "build",
            "--symfrac",
            "1.5",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--symfrac must be between 0 and 1"));
}

#[test]
fn hmmalign_rejects_multi_hmm_input() {
    let dir = tempfile::tempdir().unwrap();
    let multi_hmm = dir.path().join("multi.hmm");
    let one = std::fs::read("hmmer/testsuite/20aa.hmm").unwrap();
    let mut both = one.clone();
    both.extend_from_slice(&one);
    std::fs::write(&multi_hmm, both).unwrap();

    let output = Command::new(hmmer())
        .args([
            "align",
            multi_hmm.to_str().unwrap(),
            "hmmer/testsuite/20aa-alitest.fa",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("does not contain just one HMM"));
}

#[test]
fn hmmbuild_rejects_name_for_alignment_database() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("multi.sto");
    let hmm_out = dir.path().join("out.hmm");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\n#=GF ID a\ns1 ACDE\n//\n# STOCKHOLM 1.0\n#=GF ID b\ns1 ACDE\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "-n",
            "forced",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("can't use -n with an alignment database"));
}

#[test]
fn hmmbuild_rejects_ambiguous_short_alphabet_without_assertion() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("short.sto");
    let hmm_out = dir.path().join("short.hmm");
    std::fs::write(&sto, b"# STOCKHOLM 1.0\ns1 ACGT\ns2 ACGT\n//\n").unwrap();

    let output = Command::new(hmmer())
        .args(["build", hmm_out.to_str().unwrap(), sto.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("please specify --amino, --dna, or --rna"));
}
