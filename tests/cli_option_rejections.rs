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
fn search_output_file_open_failures_use_c_style_messages() {
    let dir = tempfile::tempdir().unwrap();
    let unwritable = dir.path();
    let unwritable_s = unwritable.to_str().unwrap();

    for (args, message) in [
        (
            vec![
                "search",
                "--tblout",
                unwritable_s,
                "hmmer/testsuite/20aa.hmm",
                "hmmer/testsuite/20aa.fa",
            ],
            "Failed to open tabular per-seq output file",
        ),
        (
            vec![
                "search",
                "--domtblout",
                unwritable_s,
                "hmmer/testsuite/20aa.hmm",
                "hmmer/testsuite/20aa.fa",
            ],
            "Failed to open tabular per-dom output file",
        ),
        (
            vec![
                "search",
                "--pfamtblout",
                unwritable_s,
                "hmmer/testsuite/20aa.hmm",
                "hmmer/testsuite/20aa.fa",
            ],
            "Failed to open pfam-style tabular output file",
        ),
        (
            vec![
                "search",
                "-A",
                unwritable_s,
                "hmmer/testsuite/20aa.hmm",
                "hmmer/testsuite/20aa.fa",
            ],
            "Failed to open alignment file",
        ),
        (
            vec![
                "phmmer",
                "-A",
                unwritable_s,
                "hmmer/testsuite/20aa.fa",
                "hmmer/testsuite/20aa.fa",
            ],
            "Failed to open alignment output file",
        ),
        (
            vec![
                "jackhmmer",
                "--domtblout",
                unwritable_s,
                "hmmer/testsuite/20aa.fa",
                "hmmer/testsuite/20aa.fa",
            ],
            "Failed to open tabular per-dom output file",
        ),
    ] {
        let output = Command::new(hmmer()).args(args).output().unwrap();

        assert!(!output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains(message), "{stderr}");
        assert!(stderr.contains(unwritable_s), "{stderr}");
        assert!(stderr.contains("for writing"), "{stderr}");
    }
}

#[test]
fn hmmsearch_tformat_rejects_unknown_formats_explicitly() {
    let output = Command::new(hmmer())
        .args([
            "search",
            "--tformat",
            "selex",
            "missing-query.hmm",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("hmmsearch --tformat=selex is not a recognized input sequence file format")
    );
}

#[test]
fn phmmer_qformat_tformat_reject_unknown_formats_explicitly() {
    for (flag, message) in [
        (
            "--qformat",
            "phmmer --qformat=selex is not a recognized input sequence file format",
        ),
        (
            "--tformat",
            "phmmer --tformat=selex is not a recognized input sequence file format",
        ),
    ] {
        let output = Command::new(hmmer())
            .args([
                "phmmer",
                flag,
                "selex",
                "missing-query.fa",
                "missing-targets.fa",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success(), "phmmer accepted {flag} selex");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains(message), "{stderr}");
    }
}

#[test]
fn jackhmmer_qformat_tformat_reject_unknown_formats_explicitly() {
    for (flag, message) in [
        (
            "--qformat",
            "jackhmmer --qformat=selex is not a recognized input sequence file format",
        ),
        (
            "--tformat",
            "jackhmmer --tformat=selex is not a recognized input sequence file format",
        ),
    ] {
        let output = Command::new(hmmer())
            .args([
                "jackhmmer",
                flag,
                "selex",
                "missing-query.fa",
                "missing-targets.fa",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success(), "jackhmmer accepted {flag} selex");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains(message), "{stderr}");
    }
}

#[test]
fn jackhmmer_accepts_then_rejects_hidden_prohibited_c_options() {
    for (args, expected) in [
        (
            vec![
                "jackhmmer",
                "--cut_ga",
                "missing-query.fa",
                "missing-targets.fa",
            ],
            "Failed to parse command line: jackhmmer does not accept a --cut-ga option",
        ),
        (
            vec![
                "jackhmmer",
                "--cut_nc",
                "missing-query.fa",
                "missing-targets.fa",
            ],
            "Failed to parse command line: jackhmmer does not accept a --cut-nc option",
        ),
        (
            vec![
                "jackhmmer",
                "--cut_tc",
                "missing-query.fa",
                "missing-targets.fa",
            ],
            "Failed to parse command line: jackhmmer does not accept a --cut-tc option",
        ),
        (
            vec![
                "jackhmmer",
                "--fast",
                "missing-query.fa",
                "missing-targets.fa",
            ],
            "Failed to parse command line: jackhmmer does not accept a --fast option",
        ),
        (
            vec![
                "jackhmmer",
                "--hand",
                "missing-query.fa",
                "missing-targets.fa",
            ],
            "Failed to parse command line: jackhmmer does not accept a --hand option",
        ),
        (
            vec![
                "jackhmmer",
                "--symfrac",
                "0.7",
                "missing-query.fa",
                "missing-targets.fa",
            ],
            "Failed to parse command line: jackhmmer does not accept a --symfrac option",
        ),
        (
            vec![
                "jackhmmer",
                "--wgiven",
                "missing-query.fa",
                "missing-targets.fa",
            ],
            "Failed to parse command line: jackhmmer does not accept a --wgiven option",
        ),
    ] {
        let output = Command::new(hmmer()).args(args).output().unwrap();
        assert!(!output.status.success());
        let combined = format!(
            "{}{}",
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap()
        );
        assert!(
            combined.contains(expected),
            "{expected:?} missing from:\n{combined}"
        );
    }
}

#[test]
fn scan_qformat_rejects_unknown_formats_explicitly() {
    for (command, message) in [
        (
            "scan",
            "hmmscan --qformat=selex is not a recognized input sequence file format",
        ),
        (
            "nhmmscan",
            "nhmmscan --qformat=selex is not a recognized input sequence file format",
        ),
    ] {
        let output = Command::new(hmmer())
            .args([
                command,
                "--qformat",
                "selex",
                "missing-models.hmm",
                "missing-query.fa",
            ])
            .output()
            .unwrap();

        assert!(
            !output.status.success(),
            "{command} accepted --qformat selex"
        );
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains(message), "{stderr}");
    }
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
    assert!(stderr.contains(
        "nhmmer --tformat=stockholm is not implemented: supported target format assertions are fasta and fmindex"
    ));
}

#[test]
fn nhmmer_and_nhmmscan_reject_missing_bgfile() {
    let nhmmer = Command::new(hmmer())
        .args([
            "nhmmer",
            "--dna",
            "--bgfile",
            "missing-bg.txt",
            "hmmer/testsuite/3box.hmm",
            "hmmer/testsuite/3box-alitest.fa",
        ])
        .output()
        .unwrap();
    assert!(!nhmmer.status.success());
    assert!(String::from_utf8_lossy(&nhmmer.stderr).contains("couldn't open bg file"));

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
    let nhmmscan = Command::new(hmmer())
        .args([
            "nhmmscan",
            "--bgfile",
            "missing-bg.txt",
            hmmdb.to_str().unwrap(),
            "hmmer/tutorial/dna_target.fa",
        ])
        .output()
        .unwrap();
    assert!(!nhmmscan.status.success());
    assert!(String::from_utf8_lossy(&nhmmscan.stderr).contains("couldn't open bg file"));
}

#[test]
fn nhmmer_rejects_max_with_fmindex_targets() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("dna_target.fm");
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

    for args in [
        vec![
            "nhmmer",
            "--max",
            "--tformat",
            "fmindex",
            "hmmer/tutorial/MADE1.hmm",
            db.to_str().unwrap(),
        ],
        vec![
            "nhmmer",
            "--max",
            "hmmer/tutorial/MADE1.hmm",
            db.to_str().unwrap(),
        ],
    ] {
        let output = Command::new(hmmer()).args(args).output().unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains("--max flag is incompatible with the fmindex target type"),
            "{stderr}"
        );
    }
}

#[test]
fn nhmmer_qformat_rejects_unknown_query_format() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "bogus",
            "missing-query.fa",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("nhmmer --qformat=bogus is not implemented"),
        "{stderr}"
    );
}

#[test]
fn nhmmer_autodetect_rejects_ambiguous_same_length_fasta_query() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("ambiguous.fa");
    std::fs::write(&query, ">q1\nGAATTC\n>q2\nGAATTC\n").unwrap();

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

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains(
            "Query file type could be either aligned or unaligned; please specify (--qformat [afa|fasta])"
        ),
        "{stderr}"
    );
}

#[test]
fn nhmmer_qformat_afa_rejects_unequal_aligned_fasta_rows() {
    let dir = tempfile::tempdir().unwrap();
    let query = dir.path().join("bad.afa");
    std::fs::write(&query, ">q1\nGAATTC\n>q2\nGAATT\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--qformat",
            "afa",
            "--dna",
            "--noali",
            query.to_str().unwrap(),
            "hmmer/testsuite/ecori.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("aligned FASTA sequence q2 has aligned length 5, expected 6"),
        "{stderr}"
    );
}

#[test]
fn nhmmer_requires_qformat_for_non_hmm_query_stdin() {
    let output = run_with_stdin(
        &["nhmmer", "-", "hmmer/testsuite/ecori.fa"],
        b"# STOCKHOLM 1.0\n#=GF ID q\nq1 GAATTC\n//\n",
    );

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr
            .contains("Must specify query file format (--qformat) to read <query file> from stdin"),
        "{stderr}"
    );
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
    for subcmd in ["nhmmer", "nhmmscan"] {
        let output = Command::new(hmmer())
            .args([
                subcmd,
                "--w_length",
                "3",
                "missing-query.hmm",
                "missing-targets.fa",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success(), "{subcmd} accepted --w_length 3");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("Invalid window length value"), "{stderr}");
    }
}

#[test]
fn nhmmer_rejects_invalid_window_beta_before_io() {
    for subcmd in ["nhmmer", "nhmmscan"] {
        let output = Command::new(hmmer())
            .args([
                subcmd,
                "--w_beta=-0.1",
                "missing-query.hmm",
                "missing-targets.fa",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success(), "{subcmd} accepted --w_beta -0.1");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains("Invalid window-length beta value"),
            "{stderr}"
        );
    }
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
    for subcmd in ["search", "scan", "nhmmer", "nhmmscan", "phmmer"] {
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

    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--cut_ga",
            "missing-query.fa",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!stderr.contains("unexpected argument"), "{stderr}");
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
fn nhmmer_rejects_invalid_block_length_before_io() {
    let output = Command::new(hmmer())
        .args([
            "nhmmer",
            "--block_length",
            "49999",
            "missing-query.hmm",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("value must be >= 50000"), "{stderr}");
}

#[test]
fn nhmmer_hidden_bias_windows_preserve_c_conflicts() {
    for subcmd in ["nhmmer", "nhmmscan"] {
        for conflict in ["--max", "--nobias"] {
            let output = Command::new(hmmer())
                .args([
                    subcmd,
                    "--B1",
                    "111",
                    conflict,
                    "missing-query.hmm",
                    "missing-targets.fa",
                ])
                .output()
                .unwrap();

            assert!(
                !output.status.success(),
                "{subcmd} accepted --B1 with {conflict}"
            );
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(stderr.contains("cannot be used with"), "{stderr}");
        }
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
fn phmmer_rejects_unknown_substitution_matrix_before_io() {
    let output = Command::new(hmmer())
        .args([
            "phmmer",
            "--mx",
            "NO_SUCH_MATRIX",
            "missing-query.fa",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("unknown built-in protein score matrix NO_SUCH_MATRIX"),
        "{stderr}"
    );
}

#[test]
fn jackhmmer_rejects_unknown_substitution_matrix_before_io() {
    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "--mx",
            "NO_SUCH_MATRIX",
            "missing-query.fa",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("unknown built-in protein score matrix NO_SUCH_MATRIX"),
        "{stderr}"
    );
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
fn phmmer_and_jackhmmer_reject_too_narrow_text_width_before_io() {
    for subcmd in ["phmmer", "jackhmmer"] {
        let output = Command::new(hmmer())
            .args([
                subcmd,
                "--textw",
                "119",
                "missing-query.fa",
                "missing-target.fa",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{subcmd} accepted --textw 119");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("--textw must be >= 120"), "{stderr}");
    }
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
fn hmmpgmd_rejects_conflicting_served_databases() {
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
    assert!(stderr.contains("cannot be used with"), "{stderr}");
}

#[test]
fn hmmpgmd_requires_served_database() {
    let output = Command::new(hmmer())
        .args(["pgmd", "--port", "9"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("hmmpgmd requires --hmmdb or --seqdb"),
        "{stderr}"
    );
}

#[test]
fn hmmpgmd_rejects_master_worker_conflict() {
    let output = Command::new(hmmer())
        .args(["pgmd", "--master", "--worker", "127.0.0.1"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("cannot be used with"), "{stderr}");
}

#[test]
fn hmmpgmd_worker_rejects_conflicting_served_database_options() {
    let output = Command::new(hmmer())
        .args([
            "pgmd",
            "--worker",
            "127.0.0.1",
            "--hmmdb",
            "missing.hmm",
            "--seqdb",
            "missing.fa",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("cannot be used with"), "{stderr}");
}

#[test]
fn hmmpgmd_worker_accepts_c_sizing_options_before_socket_failure() {
    let output = Command::new(hmmer())
        .args([
            "pgmd",
            "--worker",
            "127.0.0.1",
            "--cpu",
            "1",
            "--wport",
            "9",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!stderr.contains("unexpected argument"), "{stderr}");
    assert!(
        stderr.contains("Cannot connect to master worker port"),
        "{stderr}"
    );
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
fn hmmsim_rejects_invalid_nu_before_io() {
    let output = Command::new(hmmer())
        .args(["sim", "--nu", "3.0", "missing.hmm"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--nu is only supported with --msv"));

    let output = Command::new(hmmer())
        .args(["sim", "--msv", "--nu", "1.0", "missing.hmm"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--nu must be finite and > 1.0"));

    let output = Command::new(hmmer())
        .args(["sim", "--msv", "--fast", "--nu", "3.0", "missing.hmm"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--nu cannot be used with --fast"));
}

#[test]
fn hmmsim_rejects_invalid_pthresh_and_missing_required_flags_before_io() {
    let output = Command::new(hmmer())
        .args(["sim", "--afile", "align.tsv", "missing.hmm"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--afile requires -a"));

    let output = Command::new(hmmer())
        .args(["sim", "--pthresh", "0.02", "missing.hmm"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--pthresh requires --ffile"));

    let output = Command::new(hmmer())
        .args([
            "sim",
            "--ffile",
            "filter.tsv",
            "--pthresh",
            "1.5",
            "missing.hmm",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--pthresh must be finite and between 0.0 and 1.0"));

    let output = Command::new(hmmer())
        .args(["sim", "-a", "--fwd", "missing.hmm"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("-a requires Viterbi scoring"));
}

#[test]
fn hmmsim_rejects_invalid_calibration_options_before_io() {
    for option in [
        "--EmL",
        "--EmN",
        "--EvL",
        "--EvN",
        "--EfL",
        "--EfN",
        "--tpoints",
    ] {
        let output = Command::new(hmmer())
            .args(["sim", option, "0", "missing.hmm"])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{option}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("value must be > 0"), "{option}: {stderr}");
    }

    for option in ["--tmin", "--tmax"] {
        let output = Command::new(hmmer())
            .args(["sim", option, "0", "missing.hmm"])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{option}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains("value must be finite and > 0"),
            "{option}: {stderr}"
        );
    }

    for bad in ["0", "1"] {
        let output = Command::new(hmmer())
            .args(["sim", "--Eft", bad, "missing.hmm"])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{bad}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains("value must be > 0 and < 1"),
            "{bad}: {stderr}"
        );
    }
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

    let output = Command::new(hmmer())
        .args([
            "makehmmerdb",
            "--cstream",
            "--container",
            "missing.fa",
            "out.hmmerdb",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--cstream and --container are mutually exclusive"));
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
fn hmmconvert_unknown_outfmt_fails_explicitly_before_io() {
    let output = Command::new(hmmer())
        .args(["convert", "--outfmt", "4/a", "missing.hmm"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("No such 3.x output format code 4/a"));
}

#[test]
fn hmmconvert_hmmer2_output_is_explicitly_unsupported_before_io() {
    let output = Command::new(hmmer())
        .args(["convert", "-2", "missing.hmm"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("HMMER2 ASCII output is intentionally unsupported"));
}

#[test]
fn hmmer2_ascii_input_gap_is_explicitly_documented() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("legacy.hmm");
    std::fs::write(&hmm, "HMMER2.0\nNAME  legacy\n//\n").unwrap();

    let output = Command::new(hmmer())
        .args(["search", hmm.to_str().unwrap(), "missing.fa"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("HMMER2 ASCII input is intentionally unsupported"),
        "{stderr}"
    );
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
fn hmmfetch_index_rejects_records_rejected_by_full_parser() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("incomplete.hmm");
    std::fs::write(&hmm, b"HMMER3/f\nNAME  bad\nLENG  1\n//\n").unwrap();

    let output = Command::new(hmmer())
        .args(["fetch", "--index", hmm.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Error creating SSI index"), "{stderr}");
    assert!(!hmmer_pure_rs::ssi::path_with_appended_suffix(&hmm, ".ssi").exists());
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

    let no_rf = dir.path().join("no-rf.sto");
    std::fs::write(&no_rf, b"# STOCKHOLM 1.0\ns1 ACDE\ns2 ACDE\n//\n").unwrap();
    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--hand",
            "--modelrange",
            "1..2",
            no_rf.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("required for --hand"));
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
        .args(["alimask", "--alirange", "1..3", "hmmer/testsuite/20aa.sto"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("requires <postmsafile>"));
}

#[test]
fn alimask_rejects_conflicting_weighting_options() {
    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--wpb",
            "--wnone",
            "--ali2model",
            "1..1",
            "hmmer/testsuite/20aa.sto",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--wpb"));
    assert!(stderr.contains("--wnone"));

    let output = Command::new(hmmer())
        .args([
            "alimask",
            "--wid",
            "0.8",
            "--ali2model",
            "1..1",
            "hmmer/testsuite/20aa.sto",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--wid"));
    assert!(stderr.contains("--wblosum"));
}

#[test]
fn hmmbuild_rejects_conflicting_weighting_options_and_stray_wid() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("input.sto");
    let hmm_out = dir.path().join("out.hmm");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\ns1 ACDEFGHIKLM\ns2 ACDEYGHIKLM\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--wpb",
            "--wgsc",
            "--amino",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--wpb"));
    assert!(stderr.contains("--wgsc"));

    let output = Command::new(hmmer())
        .args([
            "build",
            "--wid",
            "0.8",
            "--amino",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--wid"));
    assert!(stderr.contains("--wblosum"));
}

#[test]
fn hmmbuild_rejects_conflicting_effective_number_options_and_stray_eid() {
    let dir = tempfile::tempdir().unwrap();
    let sto = dir.path().join("input.sto");
    let hmm_out = dir.path().join("out.hmm");
    std::fs::write(
        &sto,
        b"# STOCKHOLM 1.0\ns1 ACDEFGHIKLM\ns2 ACDEYGHIKLM\n//\n",
    )
    .unwrap();

    let output = Command::new(hmmer())
        .args([
            "build",
            "--eent",
            "--eclust",
            "--amino",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--eent"));
    assert!(stderr.contains("--eclust"));

    let output = Command::new(hmmer())
        .args([
            "build",
            "--eent",
            "--eentexp",
            "--amino",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--eent"), "{stderr}");
    assert!(stderr.contains("--eentexp"), "{stderr}");

    let output = Command::new(hmmer())
        .args([
            "jackhmmer",
            "--eent",
            "--eentexp",
            "missing-query.fa",
            "missing-targets.fa",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--eent"), "{stderr}");
    assert!(stderr.contains("--eentexp"), "{stderr}");

    let output = Command::new(hmmer())
        .args([
            "build",
            "--eid",
            "0.8",
            "--amino",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--eid"));
    assert!(stderr.contains("--eclust"));
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

    let output = Command::new(hmmer())
        .args([
            "build",
            "--fragthresh",
            "1.5",
            hmm_out.to_str().unwrap(),
            sto.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--fragthresh must be between 0 and 1"));
}

#[test]
fn hmmbuild_rejects_conflicting_prior_options() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_out = dir.path().join("out.hmm");
    let output = Command::new(hmmer())
        .args([
            "build",
            "--pnone",
            "--plaplace",
            hmm_out.to_str().unwrap(),
            "hmmer/tutorial/globins4.sto",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--pnone"));
    assert!(stderr.contains("--plaplace"));
}

#[test]
fn hmmbuild_rejects_invalid_calibration_options_before_io() {
    for option in ["--EmL", "--EmN", "--EvL", "--EvN", "--EfL", "--EfN"] {
        let output = Command::new(hmmer())
            .args([
                "build",
                option,
                "0",
                "missing-output.hmm",
                "missing-input.sto",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{option}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("value must be > 0"), "{option}: {stderr}");
    }

    for bad in ["0", "1"] {
        let output = Command::new(hmmer())
            .args([
                "build",
                "--Eft",
                bad,
                "missing-output.hmm",
                "missing-input.sto",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{bad}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains("value must be > 0 and < 1"),
            "{bad}: {stderr}"
        );
    }
}

#[test]
fn hmmbuild_rejects_invalid_window_and_maxinsert_options_before_io() {
    let output = Command::new(hmmer())
        .args([
            "build",
            "--w_length",
            "3",
            "missing-output.hmm",
            "missing-input.sto",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Invalid window length value"), "{stderr}");

    let output = Command::new(hmmer())
        .args([
            "build",
            "--w_beta=-0.1",
            "missing-output.hmm",
            "missing-input.sto",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("Invalid window-length beta value"),
        "{stderr}"
    );

    let output = Command::new(hmmer())
        .args([
            "build",
            "--maxinsertlen",
            "4",
            "missing-output.hmm",
            "missing-input.sto",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("value must be >= 5"), "{stderr}");
}

#[test]
fn hmmbuild_rejects_invalid_singlemx_options_before_io() {
    let output = Command::new(hmmer())
        .args([
            "build",
            "--mx",
            "UNKNOWN",
            "--singlemx",
            "missing-output.hmm",
            "missing-input.sto",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("unknown built-in protein score matrix"),
        "{stderr}"
    );

    let output = Command::new(hmmer())
        .args([
            "build",
            "--mxfile",
            "missing-custom.mx",
            "--singlemx",
            "missing-output.hmm",
            "missing-input.sto",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("failed to read score matrix file missing-custom.mx"),
        "{stderr}"
    );

    for option in ["--mx", "--mxfile"] {
        let mut args = vec!["build", option];
        args.push(if option == "--mxfile" {
            "custom.mx"
        } else {
            "PAM30"
        });
        args.extend(["missing-output.hmm", "missing-input.sto"]);
        let output = Command::new(hmmer()).args(args).output().unwrap();
        assert!(!output.status.success(), "{option}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("currently require --singlemx"), "{stderr}");
    }
}

#[test]
fn hmmbuild_rejects_mx_and_mxfile_together() {
    let output = Command::new(hmmer())
        .args([
            "build",
            "--singlemx",
            "--mx",
            "BLOSUM62",
            "--mxfile",
            "custom.mx",
            "missing-output.hmm",
            "missing-input.sto",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("cannot be used with") || stderr.contains("conflicts with"),
        "{stderr}"
    );
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

#[test]
fn hmmbuild_rejects_illegal_stockholm_alignment_symbols() {
    let dir = tempfile::tempdir().unwrap();
    let msa = dir.path().join("bad.sto");
    let hmm = dir.path().join("out.hmm");
    std::fs::write(&msa, "# STOCKHOLM 1.0\ns1 AC#D\n//\n").unwrap();

    let output = Command::new(hmmer())
        .args([
            "hmmbuild",
            "--amino",
            hmm.to_str().unwrap(),
            msa.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Stockholm sequence s1 contains illegal symbol '#'"));
    assert!(!hmm.exists());
}
