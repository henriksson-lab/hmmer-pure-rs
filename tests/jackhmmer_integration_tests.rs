use std::io::Write;
use std::path::PathBuf;
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

fn run_jackhmmer(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let output = Command::new(binary_path("hmmer"))
        .arg("jackhmmer")
        .args(extra_args)
        .args([seqfile, seqdb])
        .output()
        .expect("failed to run hmmer jackhmmer");

    assert!(
        output.status.success(),
        "hmmer jackhmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout).unwrap()
}

fn run_jackhmmer_with_tblout(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> (String, String) {
    let mut tblout = std::env::temp_dir();
    tblout.push(format!(
        "jackhmmer-test-{}-{}.tblout",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    let output = Command::new(binary_path("hmmer"))
        .arg("jackhmmer")
        .args(extra_args)
        .arg("--tblout")
        .arg(&tblout)
        .args([seqfile, seqdb])
        .output()
        .expect("failed to run hmmer jackhmmer --tblout");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let tbl = std::fs::read_to_string(&tblout).expect("failed to read jackhmmer tblout");
    let _ = std::fs::remove_file(&tblout);

    assert!(
        output.status.success(),
        "hmmer jackhmmer failed: {}",
        stderr
    );
    (stdout, tbl)
}

fn run_jackhmmer_with_domtblout(
    seqfile: &str,
    seqdb: &str,
    extra_args: &[&str],
) -> (String, String) {
    let mut domtblout = std::env::temp_dir();
    domtblout.push(format!(
        "jackhmmer-test-{}-{}.domtblout",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    let output = Command::new(binary_path("hmmer"))
        .arg("jackhmmer")
        .args(extra_args)
        .arg("--domtblout")
        .arg(&domtblout)
        .args([seqfile, seqdb])
        .output()
        .expect("failed to run hmmer jackhmmer --domtblout");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let domtbl = std::fs::read_to_string(&domtblout).expect("failed to read jackhmmer domtblout");
    let _ = std::fs::remove_file(&domtblout);

    assert!(
        output.status.success(),
        "hmmer jackhmmer failed: {}",
        stderr
    );
    (stdout, domtbl)
}

fn run_jackhmmer_with_tblout_and_domtblout(
    seqfile: &str,
    seqdb: &str,
    extra_args: &[&str],
) -> (String, String, String) {
    let mut tblout = std::env::temp_dir();
    tblout.push(format!(
        "jackhmmer-test-{}-{}.tblout",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut domtblout = std::env::temp_dir();
    domtblout.push(format!(
        "jackhmmer-test-{}-{}.domtblout",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    let output = Command::new(binary_path("hmmer"))
        .arg("jackhmmer")
        .args(extra_args)
        .arg("--tblout")
        .arg(&tblout)
        .arg("--domtblout")
        .arg(&domtblout)
        .args([seqfile, seqdb])
        .output()
        .expect("failed to run hmmer jackhmmer with both tabular outputs");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let tbl = std::fs::read_to_string(&tblout).expect("failed to read jackhmmer tblout");
    let domtbl = std::fs::read_to_string(&domtblout).expect("failed to read jackhmmer domtblout");
    let _ = std::fs::remove_file(&tblout);
    let _ = std::fs::remove_file(&domtblout);

    assert!(
        output.status.success(),
        "hmmer jackhmmer failed: {}",
        stderr
    );
    (stdout, tbl, domtbl)
}

fn run_jackhmmer_with_chkhmm(
    binary: &std::path::Path,
    binary_needs_subcommand: bool,
    seqfile: &str,
    seqdb: &str,
    extra_args: &[&str],
) -> (String, Vec<String>) {
    let (stdout, hmms, _tblout, _domtblout) = run_jackhmmer_with_chkhmm_and_optional_tables(
        binary,
        binary_needs_subcommand,
        seqfile,
        seqdb,
        extra_args,
        false,
    );
    (stdout, hmms)
}

fn run_jackhmmer_with_chkhmm_and_tables(
    binary: &std::path::Path,
    binary_needs_subcommand: bool,
    seqfile: &str,
    seqdb: &str,
    extra_args: &[&str],
) -> (String, Vec<String>, String, String) {
    let (stdout, hmms, tblout, domtblout) = run_jackhmmer_with_chkhmm_and_optional_tables(
        binary,
        binary_needs_subcommand,
        seqfile,
        seqdb,
        extra_args,
        true,
    );
    (
        stdout,
        hmms,
        tblout.expect("missing tblout"),
        domtblout.expect("missing domtblout"),
    )
}

fn run_jackhmmer_with_chkhmm_and_optional_tables(
    binary: &std::path::Path,
    binary_needs_subcommand: bool,
    seqfile: &str,
    seqdb: &str,
    extra_args: &[&str],
    write_tables: bool,
) -> (String, Vec<String>, Option<String>, Option<String>) {
    let prefix = unique_prefix("chkhmm", "prefix");
    let tblout = unique_prefix("chkhmm-tblout", "tblout");
    let domtblout = unique_prefix("chkhmm-domtblout", "domtblout");

    let mut cmd = Command::new(binary);
    if binary_needs_subcommand {
        cmd.arg("jackhmmer");
    }
    cmd.args(extra_args).arg("--chkhmm").arg(&prefix);
    if write_tables {
        cmd.arg("--tblout")
            .arg(&tblout)
            .arg("--domtblout")
            .arg(&domtblout);
    }
    let output = cmd
        .args([seqfile, seqdb])
        .output()
        .expect("failed to run jackhmmer --chkhmm");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "jackhmmer --chkhmm failed: {}",
        stderr
    );

    let mut hmms = Vec::new();
    for round in 1..=9 {
        let path = format!("{}-{}.hmm", prefix.display(), round);
        match std::fs::read_to_string(&path) {
            Ok(hmm) => {
                hmms.push(hmm);
                let _ = std::fs::remove_file(path);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => break,
            Err(err) => panic!("failed to read HMM checkpoint {}: {}", path, err),
        }
    }

    assert!(!hmms.is_empty(), "expected at least one HMM checkpoint");
    let tables = if write_tables {
        let tbl = std::fs::read_to_string(&tblout).expect("failed to read jackhmmer tblout");
        let domtbl =
            std::fs::read_to_string(&domtblout).expect("failed to read jackhmmer domtblout");
        let _ = std::fs::remove_file(&tblout);
        let _ = std::fs::remove_file(&domtblout);
        (Some(tbl), Some(domtbl))
    } else {
        let _ = std::fs::remove_file(&tblout);
        let _ = std::fs::remove_file(&domtblout);
        (None, None)
    };
    (stdout, hmms, tables.0, tables.1)
}

fn run_jackhmmer_with_chkali(
    binary: &std::path::Path,
    binary_needs_subcommand: bool,
    seqfile: &str,
    seqdb: &str,
    extra_args: &[&str],
) -> (String, Vec<String>) {
    let prefix = unique_prefix("chkali", "prefix");

    let mut cmd = Command::new(binary);
    if binary_needs_subcommand {
        cmd.arg("jackhmmer");
    }
    let output = cmd
        .args(extra_args)
        .arg("--chkali")
        .arg(&prefix)
        .args([seqfile, seqdb])
        .output()
        .expect("failed to run jackhmmer --chkali");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "jackhmmer --chkali failed: {}",
        stderr
    );

    let mut msas = Vec::new();
    for round in 1..=9 {
        let path = format!("{}-{}.sto", prefix.display(), round);
        match std::fs::read_to_string(&path) {
            Ok(msa) => {
                msas.push(msa);
                let _ = std::fs::remove_file(path);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => break,
            Err(err) => panic!("failed to read MSA checkpoint {}: {}", path, err),
        }
    }

    assert!(!msas.is_empty(), "expected at least one MSA checkpoint");
    (stdout, msas)
}

fn round_block<'a>(stdout: &'a str, round: usize) -> &'a str {
    if round == 1 {
        let start = stdout.find("Scores for complete sequences").unwrap();
        let rest = &stdout[start..];
        if let Some(next) = rest.find("@@ Round:                  2") {
            &rest[..next]
        } else {
            rest
        }
    } else {
        let marker = format!("@@ Round:                  {}", round);
        let start = stdout.find(&marker).unwrap();
        let rest = &stdout[start..];
        if let Some(next) = rest[marker.len()..].find("@@ Round:                  ") {
            &rest[..marker.len() + next]
        } else {
            rest
        }
    }
}

fn top_hit_rows(block: &str, n: usize) -> Vec<(String, String, String)> {
    let mut rows = Vec::new();
    let mut in_hits = false;
    for line in block.lines() {
        if line.contains("E-value  score  bias") && line.contains("Sequence") {
            in_hits = true;
            continue;
        }
        if !in_hits {
            continue;
        }
        if line.trim().is_empty() {
            break;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let offset = usize::from(matches!(fields.first(), Some(&"+") | Some(&"-")));
        if fields.len() >= offset + 9
            && (fields[offset].chars().next().unwrap().is_ascii_digit()
                || fields[offset].contains('e'))
        {
            rows.push((
                fields[offset].to_string(),
                fields[offset + 1].to_string(),
                fields[offset + 8].to_string(),
            ));
            if rows.len() == n {
                break;
            }
        }
    }
    rows
}

fn tblout_rows(tblout: &str, n: usize) -> Vec<(String, String, String)> {
    tblout
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .take(n)
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            (
                fields[0].to_string(),
                fields[4].to_string(),
                fields[5].to_string(),
            )
        })
        .collect()
}

fn domtblout_rows(domtblout: &str, n: usize) -> Vec<Vec<String>> {
    domtblout
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .take(n)
        .map(|line| line.split_whitespace().map(|s| s.to_string()).collect())
        .collect()
}

fn normalized_data_rows(table: &str) -> Vec<String> {
    table
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

fn query_lines(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter(|line| line.starts_with("Query:"))
        .map(|line| line.to_string())
        .collect()
}

fn unique_prefix(stem: &str, extless_suffix: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "jackhmmer-test-{}-{}-{}",
        stem,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    path.set_extension(extless_suffix);
    path.with_extension("")
}

fn extract_fasta_record(src: &str, id_prefix: &str) -> PathBuf {
    let fasta = std::fs::read_to_string(src).expect("failed to read FASTA fixture");
    let mut out = String::new();
    let mut keep = false;
    for line in fasta.lines() {
        if let Some(header) = line.strip_prefix('>') {
            keep = header.starts_with(id_prefix);
            if keep {
                out.push('>');
                out.push_str(header);
                out.push('\n');
            }
        } else if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    assert!(
        !out.is_empty(),
        "failed to find FASTA record starting with {id_prefix}"
    );

    let mut path = std::env::temp_dir();
    path.push(format!(
        "jackhmmer-query-{}-{}.fa",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut file = std::fs::File::create(&path).expect("failed to create temp FASTA query");
    file.write_all(out.as_bytes())
        .expect("failed to write temp FASTA query");
    path
}

fn write_two_record_20aa_query() -> PathBuf {
    let path = unique_prefix("two-query-20aa", "fa");
    let mut file = std::fs::File::create(&path).expect("failed to create temp FASTA query");
    file.write_all(b">test1\nACDEFGHIKLMNPQRSTVWY\n>test2\nXXXXACDEFGHIMNXXXPQRSTVWY\n")
        .expect("failed to write temp FASTA query");
    path
}

fn hmm_header_value<'a>(hmm: &'a str, key: &str) -> &'a str {
    hmm.lines()
        .find_map(|line| line.strip_prefix(key).map(str::trim))
        .unwrap_or_else(|| panic!("missing HMM header field {}", key))
}

fn hmm_stats_lines(hmm: &str) -> Vec<String> {
    hmm.lines()
        .filter(|line| line.starts_with("STATS LOCAL "))
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

fn assert_hmm_float_field_close(rust_hmm: &str, c_hmm: &str, key: &str, tol: f64) {
    let rust = hmm_header_value(rust_hmm, key)
        .parse::<f64>()
        .unwrap_or_else(|err| panic!("failed to parse Rust HMM field {}: {}", key, err));
    let c = hmm_header_value(c_hmm, key)
        .parse::<f64>()
        .unwrap_or_else(|err| panic!("failed to parse C HMM field {}: {}", key, err));
    assert!(
        (rust - c).abs() <= tol,
        "round-2 HMM header field {} drifted: rust={} c={} tol={}",
        key,
        rust,
        c,
        tol
    );
}

fn assert_round2_checkpoint_and_table_parity(
    rust_hmms: &[String],
    c_hmms: &[String],
    rust_tblout: &str,
    c_tblout: &str,
    rust_domtblout: &str,
    c_domtblout: &str,
) {
    assert_eq!(rust_hmms.len(), 2);
    assert_eq!(c_hmms.len(), 2);

    let rust_round2 = &rust_hmms[1];
    let c_round2 = &c_hmms[1];
    for key in [
        "NAME", "LENG", "ALPH", "RF", "CONS", "CS", "MAP", "NSEQ", "CKSUM",
    ] {
        assert_eq!(
            hmm_header_value(rust_round2, key),
            hmm_header_value(c_round2, key),
            "round-2 HMM header field {} drifted",
            key
        );
    }
    assert_hmm_float_field_close(rust_round2, c_round2, "EFFN", 1.0e-6);
    assert_eq!(hmm_stats_lines(rust_round2), hmm_stats_lines(c_round2));
    assert_eq!(
        normalized_hmm_for_exact_parity(rust_round2),
        normalized_hmm_for_exact_parity(c_round2),
        "round-2 HMM body drifted"
    );

    assert_eq!(
        normalized_data_rows(rust_tblout),
        normalized_data_rows(c_tblout),
        "round-2 tblout rows drifted"
    );
    assert_eq!(
        normalized_data_rows(rust_domtblout),
        normalized_data_rows(c_domtblout),
        "round-2 domtblout rows drifted"
    );
}

fn normalized_hmm_for_exact_parity(hmm: &str) -> Vec<String> {
    hmm.lines()
        .filter(|line| {
            !line.starts_with("HMMER3/") && !line.starts_with("DATE  ") && !line.trim().is_empty()
        })
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

fn normalized_stockholm_for_exact_parity(sto: &str) -> Vec<String> {
    sto.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

#[test]
fn jackhmmer_round1_20aa_matches_current_single_sequence_baseline() {
    let query = extract_fasta_record(&test_path("hmmer/testsuite/20aa-alitest.fa"), "test1");
    let stdout = run_jackhmmer(
        query.to_str().unwrap(),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["-N", "1"],
    );
    let _ = std::fs::remove_file(query);
    let round1 = round_block(&stdout, 1);

    assert!(stdout.contains("[ok]"));
    assert!(stdout.contains("Query:       test1  [L=20]"));
    assert_eq!(
        top_hit_rows(round1, 4),
        vec![
            ("2e-16".to_string(), "44.3".to_string(), "test1".to_string()),
            (
                "5.1e-16".to_string(),
                "43.2".to_string(),
                "test4".to_string()
            ),
            ("5e-11".to_string(), "28.8".to_string(), "test2".to_string()),
            (
                "1.8e-10".to_string(),
                "27.2".to_string(),
                "test3".to_string()
            ),
        ],
        "jackhmmer round-1 20aa hits changed"
    );
}

#[test]
fn jackhmmer_round1_accepts_seed_and_calibration_tuning() {
    let stdout = run_jackhmmer(
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &[
            "-N", "1", "--seed", "7", "--EmN", "20", "--EvN", "20", "--EfN", "20",
        ],
    );

    assert!(stdout.contains("# random number seed set to:       7\n"));
    assert!(stdout.contains("# seq number for MSV Gumbel mu fit: 20\n"));
    assert!(stdout.contains("# seq number for Vit Gumbel mu fit: 20\n"));
    assert!(stdout.contains("# seq number for Fwd exp tau fit:   20\n"));
}

#[test]
fn jackhmmer_score_reporting_threshold_headers_match_c_wording() {
    let stdout = run_jackhmmer(
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["-N", "1", "--noali", "-T", "25", "--domT", "3"],
    );

    assert!(stdout.contains("# sequence reporting threshold:    score <= 25\n"));
    assert!(stdout.contains("# domain reporting threshold:      score <= 3\n"));
}

#[test]
fn jackhmmer_round1_two_query_fasta_matches_c_query_blocks() {
    let query = write_two_record_20aa_query();
    let target = test_path("hmmer/testsuite/20aa-alitest.fa");
    let rust_stdout = run_jackhmmer(query.to_str().unwrap(), &target, &["-N", "1"]);
    let c_output = Command::new(test_path("hmmer/src/jackhmmer"))
        .args(["-N", "1", query.to_str().unwrap(), &target])
        .output()
        .expect("failed to run bundled C jackhmmer");
    assert!(
        c_output.status.success(),
        "bundled C jackhmmer failed: {}",
        String::from_utf8_lossy(&c_output.stderr)
    );
    let c_stdout = String::from_utf8(c_output.stdout).unwrap();

    let rust_queries = query_lines(&rust_stdout);
    let c_queries = query_lines(&c_stdout);
    assert_eq!(rust_queries, c_queries);
    assert_eq!(rust_queries.len(), 2);
    assert_eq!(
        rust_stdout.lines().filter(|line| *line == "//").count(),
        c_stdout.lines().filter(|line| *line == "//").count()
    );
    let second_query_pos = rust_stdout
        .match_indices("Query:")
        .nth(1)
        .expect("missing second query block")
        .0;
    assert!(
        rust_stdout[..second_query_pos].trim_end().ends_with("//"),
        "first query report should be terminated before the second Query block"
    );
    let _ = std::fs::remove_file(query);
}

#[test]
fn jackhmmer_two_query_tabular_outputs_write_single_header_block() {
    let query = write_two_record_20aa_query();
    let target = test_path("hmmer/testsuite/20aa-alitest.fa");
    let (_stdout, tblout, domtblout) =
        run_jackhmmer_with_tblout_and_domtblout(query.to_str().unwrap(), &target, &["-N", "1"]);

    assert_eq!(tblout.matches("# target name").count(), 1);
    assert_eq!(domtblout.matches("# target name").count(), 1);
    let _ = std::fs::remove_file(query);
}

#[test]
fn jackhmmer_globins_converges_in_two_rounds_with_expected_round_profiles() {
    let stdout = run_jackhmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "2"],
    );
    let round1 = round_block(&stdout, 1);
    let round2 = round_block(&stdout, 2);

    assert!(stdout.contains("@@ New targets included:   45"));
    assert!(
        stdout.contains("@@ New alignment includes: 46 subseqs (was 1), including original query")
    );
    assert!(stdout.contains("@@ Continuing to next round."));
    assert_eq!(
        top_hit_rows(round1, 5),
        vec![
            (
                "2.7e-97".to_string(),
                "314.3".to_string(),
                "HBB_CALAR".to_string()
            ),
            (
                "4.3e-97".to_string(),
                "313.6".to_string(),
                "HBB_MANSP".to_string()
            ),
            (
                "2.4e-91".to_string(),
                "295.0".to_string(),
                "HBB_URSMA".to_string()
            ),
            (
                "3.6e-91".to_string(),
                "294.4".to_string(),
                "HBB_RABIT".to_string()
            ),
            (
                "5.4e-84".to_string(),
                "271.1".to_string(),
                "HBB_SUNMU".to_string()
            ),
        ],
        "jackhmmer round-1 globins hits changed"
    );

    assert_eq!(
        top_hit_rows(round2, 5),
        vec![
            (
                "1.4e-74".to_string(),
                "240.7".to_string(),
                "HBB_MANSP".to_string()
            ),
            (
                "5.6e-74".to_string(),
                "238.8".to_string(),
                "HBB_URSMA".to_string()
            ),
            (
                "6.8e-73".to_string(),
                "235.3".to_string(),
                "HBB_RABIT".to_string()
            ),
            (
                "1.9e-72".to_string(),
                "233.8".to_string(),
                "HBB_SUNMU".to_string()
            ),
            (
                "2.4e-72".to_string(),
                "233.5".to_string(),
                "HBB_CALAR".to_string()
            ),
        ],
        "jackhmmer round-2 globins hits changed"
    );
}

#[test]
fn jackhmmer_strict_thresholds_stop_after_empty_round_on_20aa() {
    let query = extract_fasta_record(&test_path("hmmer/testsuite/20aa-alitest.fa"), "test1");
    let stdout = run_jackhmmer(
        query.to_str().unwrap(),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["-N", "2", "-E", "1e-20", "--incE", "1e-20"],
    );
    let _ = std::fs::remove_file(query);

    assert!(stdout.contains("[No hits detected that satisfy reporting thresholds]"));
    assert!(stdout.contains("@@ New targets included:   0"));
    assert!(
        stdout.contains("@@ New alignment includes: 1 subseqs (was 1), including original query")
    );
    assert!(stdout.contains("@@ Continuing to next round."));
    assert!(stdout.contains("@@ Round:                  2"));
    assert!(stdout
        .contains("@@ Included in MSA:        1 subsequences (query + 0 subseqs from 0 targets)"));
    assert!(!stdout.contains("Query:       test1-i1  [M=20]"));
}

#[test]
fn jackhmmer_z_override_does_not_change_reported_target_count() {
    let stdout = run_jackhmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "1", "-Z", "123"],
    );
    assert!(stdout.contains("Target sequences:                         45  ("));
    assert!(!stdout.contains("Target sequences:                        123  ("));
}

#[test]
fn jackhmmer_globins_strict_inc_threshold_changes_round2_profile() {
    let stdout = run_jackhmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "2", "-E", "1e-20", "--incE", "1e-20"],
    );
    let round1 = round_block(&stdout, 1);
    let round2 = round_block(&stdout, 2);

    assert!(stdout.contains("@@ New targets included:   38"));
    assert!(
        stdout.contains("@@ New alignment includes: 39 subseqs (was 1), including original query")
    );
    assert_eq!(
        top_hit_rows(round1, 5),
        vec![
            (
                "2.7e-97".to_string(),
                "314.3".to_string(),
                "HBB_CALAR".to_string()
            ),
            (
                "4.3e-97".to_string(),
                "313.6".to_string(),
                "HBB_MANSP".to_string()
            ),
            (
                "2.4e-91".to_string(),
                "295.0".to_string(),
                "HBB_URSMA".to_string()
            ),
            (
                "3.6e-91".to_string(),
                "294.4".to_string(),
                "HBB_RABIT".to_string()
            ),
            (
                "5.4e-84".to_string(),
                "271.1".to_string(),
                "HBB_SUNMU".to_string()
            ),
        ],
        "jackhmmer strict-threshold round-1 globins hits changed"
    );

    assert_eq!(
        top_hit_rows(round2, 5),
        vec![
            (
                "2.5e-79".to_string(),
                "256.1".to_string(),
                "HBB_MANSP".to_string()
            ),
            (
                "1.6e-78".to_string(),
                "253.4".to_string(),
                "HBB_URSMA".to_string()
            ),
            (
                "6.1e-78".to_string(),
                "251.5".to_string(),
                "HBB_RABIT".to_string()
            ),
            (
                "3.8e-77".to_string(),
                "249.0".to_string(),
                "HBB_CALAR".to_string()
            ),
            (
                "4.6e-77".to_string(),
                "248.7".to_string(),
                "HBB_SUNMU".to_string()
            ),
        ],
        "jackhmmer strict-threshold round-2 globins hits changed"
    );
}

#[test]
fn jackhmmer_tblout_20aa_reports_final_round_rows() {
    let query = extract_fasta_record(&test_path("hmmer/testsuite/20aa-alitest.fa"), "test1");
    let (stdout, tblout) = run_jackhmmer_with_tblout(
        query.to_str().unwrap(),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["-N", "1"],
    );
    let _ = std::fs::remove_file(query);

    assert!(stdout.contains("[ok]"));
    assert_eq!(
        tblout_rows(&tblout, 4),
        vec![
            ("test1".to_string(), "2e-16".to_string(), "44.3".to_string()),
            (
                "test4".to_string(),
                "5.1e-16".to_string(),
                "43.2".to_string()
            ),
            ("test2".to_string(), "5e-11".to_string(), "28.8".to_string()),
            (
                "test3".to_string(),
                "1.8e-10".to_string(),
                "27.2".to_string()
            ),
        ],
        "jackhmmer --tblout 20aa rows changed"
    );
}

#[test]
fn jackhmmer_tblout_globins_uses_final_converged_round() {
    let (stdout, tblout) = run_jackhmmer_with_tblout(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "2"],
    );

    assert!(stdout.contains("@@ CONVERGED (in 2 rounds). "));
    assert_eq!(
        tblout_rows(&tblout, 5),
        vec![
            (
                "HBB_MANSP".to_string(),
                "1.4e-74".to_string(),
                "240.7".to_string()
            ),
            (
                "HBB_URSMA".to_string(),
                "5.6e-74".to_string(),
                "238.8".to_string()
            ),
            (
                "HBB_RABIT".to_string(),
                "6.8e-73".to_string(),
                "235.3".to_string()
            ),
            (
                "HBB_SUNMU".to_string(),
                "1.9e-72".to_string(),
                "233.8".to_string()
            ),
            (
                "HBB_CALAR".to_string(),
                "2.4e-72".to_string(),
                "233.5".to_string()
            ),
        ],
        "jackhmmer --tblout should reflect the final converged round"
    );
    assert!(tblout.contains("# Program:         jackhmmer\n"));
    assert!(tblout.contains("# Pipeline mode:   SEARCH\n"));
    assert!(tblout.contains("# Option settings: jackhmmer -N 2 --tblout "));
    assert!(tblout.ends_with("# [ok]\n"));
}

#[test]
fn jackhmmer_domtblout_20aa_reports_round1_domain_rows() {
    let query = extract_fasta_record(&test_path("hmmer/testsuite/20aa-alitest.fa"), "test1");
    let (stdout, domtblout) = run_jackhmmer_with_domtblout(
        query.to_str().unwrap(),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["-N", "1"],
    );
    let _ = std::fs::remove_file(query);

    assert!(stdout.contains("[ok]"));
    let rows = domtblout_rows(&domtblout, 4);
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0][0], "test1");
    assert_eq!(rows[0][3], "test1");
    assert_eq!(rows[0][5], "20");
    assert_eq!(rows[0][6], "2e-16");
    assert_eq!(rows[0][13], "44.3");
    assert_eq!(&rows[0][15..21], ["1", "20", "1", "20", "1", "20"]);

    assert_eq!(rows[1][0], "test4");
    assert_eq!(rows[1][6], "5.1e-16");
    assert_eq!(rows[1][13], "43.0");
    assert_eq!(rows[2][0], "test2");
    assert_eq!(rows[2][13], "28.5");
    assert_eq!(rows[3][0], "test3");
    assert_eq!(rows[3][13], "25.3");
}

#[test]
fn jackhmmer_domtblout_globins_uses_final_model_length_with_original_query_name() {
    let (stdout, domtblout) = run_jackhmmer_with_domtblout(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "2"],
    );

    assert!(stdout.contains("@@ CONVERGED (in 2 rounds). "));
    let rows = domtblout_rows(&domtblout, 5);
    assert_eq!(rows.len(), 5);

    assert_eq!(rows[0][0], "HBB_MANSP");
    assert_eq!(rows[0][3], "HBB_HUMAN");
    assert_eq!(rows[0][5], "146");
    assert_eq!(rows[0][6], "1.4e-74");
    assert_eq!(rows[0][13], "240.6");
    assert_eq!(&rows[0][15..21], ["1", "146", "1", "146", "1", "146"]);

    assert_eq!(rows[1][0], "HBB_URSMA");
    assert_eq!(rows[1][13], "238.6");
    assert_eq!(rows[2][0], "HBB_RABIT");
    assert_eq!(rows[2][13], "235.1");
    assert_eq!(rows[3][0], "HBB_SUNMU");
    assert_eq!(rows[3][13], "233.7");
    assert_eq!(rows[4][0], "HBB_CALAR");
    assert_eq!(rows[4][13], "233.4");
    assert!(domtblout.contains("# Program:         jackhmmer\n"));
    assert!(domtblout.contains("# Pipeline mode:   SEARCH\n"));
    assert!(domtblout.contains("# Option settings: jackhmmer -N 2 --domtblout "));
    assert!(domtblout.ends_with("# [ok]\n"));
}

#[test]
fn jackhmmer_round1_nonull2_zeroes_bias_and_raises_globins_scores() {
    let stdout = run_jackhmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "1", "--nonull2"],
    );
    let round1 = round_block(&stdout, 1);

    assert_eq!(
        top_hit_rows(round1, 4),
        vec![
            (
                "1.5e-97".to_string(),
                "315.1".to_string(),
                "HBB_CALAR".to_string()
            ),
            (
                "2.9e-97".to_string(),
                "314.2".to_string(),
                "HBB_MANSP".to_string()
            ),
            (
                "1.8e-91".to_string(),
                "295.4".to_string(),
                "HBB_URSMA".to_string()
            ),
            (
                "2.5e-91".to_string(),
                "294.9".to_string(),
                "HBB_RABIT".to_string()
            ),
        ],
        "jackhmmer --nonull2 round-1 globins top rows changed"
    );
    assert!(round1.contains(" 1.5e-97  315.1   0.0"));
}

#[test]
fn jackhmmer_round1_nobias_is_accepted_on_globins_fixture() {
    let stdout = run_jackhmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "1", "--nobias"],
    );
    let round1 = round_block(&stdout, 1);

    assert!(stdout.contains("[ok]"));
    assert_eq!(
        top_hit_rows(round1, 3)
            .iter()
            .map(|(_, _, name)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["HBB_CALAR", "HBB_MANSP", "HBB_URSMA"],
        "jackhmmer --nobias round-1 globins top-hit ordering changed"
    );
}

#[test]
fn jackhmmer_chkhmm_writes_per_round_hmm_checkpoints() {
    let prefix = unique_prefix("chkhmm", "prefix");
    let output = Command::new(binary_path("hmmer"))
        .arg("jackhmmer")
        .arg("-N")
        .arg("2")
        .arg("--chkhmm")
        .arg(&prefix)
        .args([
            test_path("hmmer/tutorial/HBB_HUMAN"),
            test_path("hmmer/tutorial/globins45.fa"),
        ])
        .output()
        .expect("failed to run hmmer jackhmmer --chkhmm");

    assert!(
        output.status.success(),
        "hmmer jackhmmer --chkhmm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let hmm1 = std::fs::read_to_string(format!("{}-1.hmm", prefix.display()))
        .expect("missing round-1 HMM checkpoint");
    let hmm2 = std::fs::read_to_string(format!("{}-2.hmm", prefix.display()))
        .expect("missing round-2 HMM checkpoint");

    assert!(hmm1.contains("NAME  HBB_HUMAN"));
    assert!(hmm1.contains("LENG  146"));
    assert!(hmm2.contains("NAME  HBB_HUMAN-i1"));
    assert!(hmm2.contains("LENG  146"));

    let _ = std::fs::remove_file(format!("{}-1.hmm", prefix.display()));
    let _ = std::fs::remove_file(format!("{}-2.hmm", prefix.display()));
}

#[test]
fn jackhmmer_chkali_writes_per_round_alignment_checkpoints() {
    let prefix = unique_prefix("chkali", "prefix");
    let output = Command::new(binary_path("hmmer"))
        .arg("jackhmmer")
        .arg("-N")
        .arg("2")
        .arg("--chkali")
        .arg(&prefix)
        .args([
            test_path("hmmer/tutorial/HBB_HUMAN"),
            test_path("hmmer/tutorial/globins45.fa"),
        ])
        .output()
        .expect("failed to run hmmer jackhmmer --chkali");

    assert!(
        output.status.success(),
        "hmmer jackhmmer --chkali failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let sto1 = std::fs::read_to_string(format!("{}-1.sto", prefix.display()))
        .expect("missing round-1 alignment checkpoint");
    let sto2 = std::fs::read_to_string(format!("{}-2.sto", prefix.display()))
        .expect("missing round-2 alignment checkpoint");

    assert!(sto1.contains("# STOCKHOLM 1.0"));
    assert!(sto1.contains("#=GF ID HBB_HUMAN-i1"));
    assert!(sto1.contains("HBB_HUMAN"));
    assert!(sto1.contains("#=GC RF"));

    assert!(sto2.contains("# STOCKHOLM 1.0"));
    assert!(sto2.contains("#=GF ID HBB_HUMAN-i2"));
    assert!(sto2.contains("HBB_MANSP"));
    assert!(sto2.contains("#=GC RF"));

    let _ = std::fs::remove_file(format!("{}-1.sto", prefix.display()));
    let _ = std::fs::remove_file(format!("{}-2.sto", prefix.display()));
}

#[test]
fn jackhmmer_a_writes_final_included_alignment() {
    let ali_out = unique_prefix("ali", "sto");
    let output = Command::new(binary_path("hmmer"))
        .arg("jackhmmer")
        .arg("-N")
        .arg("1")
        .arg("-A")
        .arg(&ali_out)
        .args([
            test_path("hmmer/tutorial/HBB_HUMAN"),
            test_path("hmmer/tutorial/globins45.fa"),
        ])
        .output()
        .expect("failed to run hmmer jackhmmer -A");

    assert!(
        output.status.success(),
        "hmmer jackhmmer -A failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# MSA of hits saved to file:"));
    assert!(stdout.contains("# Alignment of "));
    assert!(stdout.contains(" saved to:"));

    let sto = std::fs::read_to_string(&ali_out).expect("missing final alignment output");
    assert!(sto.starts_with("# STOCKHOLM 1.0\n"));
    assert!(sto.contains("#=GF ID HBB_HUMAN-i1\n"));
    assert!(sto.contains("#=GF AU jackhmmer (HMMER 3.4)\n"));
    assert!(sto.contains("HBB_HUMAN"));
    assert!(sto.contains("#=GC RF"));

    let _ = std::fs::remove_file(&ali_out);
}

#[test]
fn jackhmmer_globins_round2_chkali_matches_bundled_c_exactly() {
    let (rust_stdout, rust_msas) = run_jackhmmer_with_chkali(
        &binary_path("hmmer"),
        true,
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "2", "--cpu", "1"],
    );
    let (_c_stdout, c_msas) = run_jackhmmer_with_chkali(
        std::path::Path::new(&test_path("hmmer/src/jackhmmer")),
        false,
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "2", "--cpu", "1"],
    );

    assert!(rust_stdout.contains("@@ New targets included:   45"));
    assert_eq!(rust_msas.len(), 2);
    assert_eq!(c_msas.len(), 2);
    assert_eq!(
        normalized_stockholm_for_exact_parity(&rust_msas[1]),
        normalized_stockholm_for_exact_parity(&c_msas[1])
    );
}

#[test]
fn jackhmmer_globins_round2_chkhmm_matches_bundled_c_exactly() {
    let (rust_stdout, rust_hmms) = run_jackhmmer_with_chkhmm(
        &binary_path("hmmer"),
        true,
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "2", "--cpu", "1"],
    );
    let (_c_stdout, c_hmms) = run_jackhmmer_with_chkhmm(
        std::path::Path::new(&test_path("hmmer/src/jackhmmer")),
        false,
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "2", "--cpu", "1"],
    );

    assert!(rust_stdout.contains("@@ New targets included:   45"));
    assert_eq!(rust_hmms.len(), 2);
    assert_eq!(c_hmms.len(), 2);
    assert_eq!(
        normalized_hmm_for_exact_parity(&rust_hmms[1]),
        normalized_hmm_for_exact_parity(&c_hmms[1])
    );
}

#[test]
fn jackhmmer_post_round_model_construction_options_match_bundled_c() {
    let args = [
        "-N",
        "2",
        "--cpu",
        "1",
        "--wnone",
        "--enone",
        "--fragthresh",
        "0.0",
    ];
    let (rust_stdout, rust_hmms) = run_jackhmmer_with_chkhmm(
        &binary_path("hmmer"),
        true,
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &args,
    );
    let (_c_stdout, c_hmms) = run_jackhmmer_with_chkhmm(
        std::path::Path::new(&test_path("hmmer/src/jackhmmer")),
        false,
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &args,
    );

    assert!(rust_stdout.contains("# relative weighting scheme:       none"));
    assert!(rust_stdout.contains("# effective seq number scheme:     none"));
    assert!(rust_stdout.contains("# define fragments if <= x*alen:   0.000"));
    assert_eq!(rust_hmms.len(), 2);
    assert_eq!(c_hmms.len(), 2);
    for key in ["NAME", "LENG", "ALPH", "RF", "CONS", "MAP", "NSEQ", "CKSUM"] {
        assert_eq!(
            hmm_header_value(&rust_hmms[1], key),
            hmm_header_value(&c_hmms[1], key),
            "round-2 HMM header field {} drifted",
            key
        );
    }
    assert_hmm_float_field_close(&rust_hmms[1], &c_hmms[1], "EFFN", 1.0e-6);
    assert_eq!(hmm_stats_lines(&rust_hmms[1]), hmm_stats_lines(&c_hmms[1]));
}

#[test]
fn jackhmmer_globins_round2_tables_match_bundled_c_exactly() {
    let (rust_stdout, _rust_hmms, rust_tblout, rust_domtblout) =
        run_jackhmmer_with_chkhmm_and_tables(
            &binary_path("hmmer"),
            true,
            &test_path("hmmer/tutorial/HBB_HUMAN"),
            &test_path("hmmer/tutorial/globins45.fa"),
            &["-N", "2", "--cpu", "1"],
        );
    let (_c_stdout, _c_hmms, c_tblout, c_domtblout) = run_jackhmmer_with_chkhmm_and_tables(
        std::path::Path::new(&test_path("hmmer/src/jackhmmer")),
        false,
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["-N", "2", "--cpu", "1"],
    );

    assert!(rust_stdout.contains("@@ New targets included:   45"));
    assert_eq!(
        normalized_data_rows(&rust_tblout),
        normalized_data_rows(&c_tblout)
    );
    assert_eq!(
        normalized_data_rows(&rust_domtblout),
        normalized_data_rows(&c_domtblout)
    );
}

#[test]
fn jackhmmer_medium_realworld_round2_matches_expected_tbl_and_dom_counts() {
    let db = test_path("test_data/human_swissprot_2k.fasta");
    let query = extract_fasta_record(&db, "sp|O43739|CYH3_HUMAN");
    let (stdout, tblout, domtblout) =
        run_jackhmmer_with_tblout_and_domtblout(query.to_str().unwrap(), &db, &["-N", "2"]);
    let _ = std::fs::remove_file(query);

    assert!(stdout.contains("@@ New targets included:   55"));
    assert!(
        stdout.contains("@@ New alignment includes: 59 subseqs (was 1), including original query")
    );
    assert_eq!(
        tblout
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .count(),
        162
    );
    assert_eq!(
        domtblout
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .count(),
        235
    );

    let top = tblout_rows(&tblout, 5);
    assert_eq!(top[0].0, "sp|O43739|CYH3_HUMAN");
    assert_eq!(top[1].0, "sp|Q15438|CYH1_HUMAN");
    assert_eq!(top[2].0, "sp|Q99418|CYH2_HUMAN");
    assert_eq!(top[3].0, "sp|Q9UIA0|CYH4_HUMAN");
}

#[test]
fn jackhmmer_haptoglobin_realworld_round2_matches_expected_tbl_and_dom_counts() {
    let db = test_path("test_data/human_swissprot_2k.fasta");
    let query = extract_fasta_record(&db, "sp|P00738|HPT_HUMAN");
    let (stdout, tblout, domtblout) =
        run_jackhmmer_with_tblout_and_domtblout(query.to_str().unwrap(), &db, &["-N", "2"]);
    let _ = std::fs::remove_file(query);

    assert!(stdout.contains("@@ New targets included:   107"));
    assert!(
        stdout.contains("@@ New alignment includes: 114 subseqs (was 1), including original query")
    );
    assert_eq!(
        tblout
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .count(),
        127
    );
    assert_eq!(
        domtblout
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .count(),
        156
    );

    let top = tblout_rows(&tblout, 5);
    assert_eq!(top[0].0, "sp|Q7Z410|TMPS9_HUMAN");
    assert_eq!(top[1].0, "sp|P00738|HPT_HUMAN");
    assert_eq!(top[2].0, "sp|P00739|HPTR_HUMAN");
    assert_eq!(top[3].0, "sp|Q7RTY7|OVCH1_HUMAN");
    assert_eq!(top[4].0, "sp|Q5K4E3|POLS2_HUMAN");
}

#[test]
fn jackhmmer_medium_realworld_round2_checkpoint_and_tables_match_c() {
    let db = test_path("test_data/human_swissprot_2k.fasta");
    let query = extract_fasta_record(&db, "sp|O43739|CYH3_HUMAN");

    let (rust_stdout, rust_hmms, rust_tblout, rust_domtblout) =
        run_jackhmmer_with_chkhmm_and_tables(
            &binary_path("hmmer"),
            true,
            query.to_str().unwrap(),
            &db,
            &["-N", "2", "--cpu", "1"],
        );
    let (_c_stdout, c_hmms, c_tblout, c_domtblout) = run_jackhmmer_with_chkhmm_and_tables(
        std::path::Path::new(&test_path("hmmer/src/jackhmmer")),
        false,
        query.to_str().unwrap(),
        &db,
        &["-N", "2", "--cpu", "1"],
    );
    let _ = std::fs::remove_file(query);

    assert!(rust_stdout.contains("@@ New targets included:   55"));
    assert!(rust_stdout
        .contains("@@ New alignment includes: 59 subseqs (was 1), including original query"));
    assert_round2_checkpoint_and_table_parity(
        &rust_hmms,
        &c_hmms,
        &rust_tblout,
        &c_tblout,
        &rust_domtblout,
        &c_domtblout,
    );
}

#[test]
fn jackhmmer_haptoglobin_realworld_round2_checkpoint_and_tables_match_c() {
    let db = test_path("test_data/human_swissprot_2k.fasta");
    let query = extract_fasta_record(&db, "sp|P00738|HPT_HUMAN");

    let (rust_stdout, rust_hmms, rust_tblout, rust_domtblout) =
        run_jackhmmer_with_chkhmm_and_tables(
            &binary_path("hmmer"),
            true,
            query.to_str().unwrap(),
            &db,
            &["-N", "2", "--cpu", "1"],
        );
    let (_c_stdout, c_hmms, c_tblout, c_domtblout) = run_jackhmmer_with_chkhmm_and_tables(
        std::path::Path::new(&test_path("hmmer/src/jackhmmer")),
        false,
        query.to_str().unwrap(),
        &db,
        &["-N", "2", "--cpu", "1"],
    );
    let _ = std::fs::remove_file(query);

    assert!(rust_stdout.contains("@@ New targets included:   107"));
    assert!(rust_stdout
        .contains("@@ New alignment includes: 114 subseqs (was 1), including original query"));
    assert_round2_checkpoint_and_table_parity(
        &rust_hmms,
        &c_hmms,
        &rust_tblout,
        &c_tblout,
        &rust_domtblout,
        &c_domtblout,
    );
}
