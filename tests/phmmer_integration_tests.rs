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

fn run_phmmer(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> std::process::Output {
    let output = Command::new(binary_path("hmmer"))
        .arg("phmmer")
        .args(extra_args)
        .args([seqfile, seqdb])
        .output()
        .expect("failed to run hmmer phmmer");

    assert!(
        output.status.success(),
        "hmmer phmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    output
}

fn run_c_phmmer_tblout(seqfile: &str, seqdb: &str, extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("out.tbl");
    let output = Command::new(test_path("hmmer/src/phmmer"))
        .args(extra_args)
        .arg("--tblout")
        .arg(&tblout)
        .args([seqfile, seqdb])
        .output()
        .expect("failed to run bundled C phmmer");

    assert!(
        output.status.success(),
        "bundled C phmmer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::read_to_string(tblout).unwrap()
}

fn normalize_phmmer_stdout(stdout: &str) -> Vec<String> {
    let root_prefix = format!("{}/", env!("CARGO_MANIFEST_DIR"));
    stdout
        .lines()
        .filter(|line| !line.starts_with("# CPU time:") && !line.starts_with("# Mc/sec:"))
        .map(|line| line.replace(&root_prefix, ""))
        .collect()
}

fn parse_tblout_rows(content: &str) -> Vec<(String, String, String, String)> {
    content
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 19 || fields[1] != "-" || fields[3] != "-" {
                return None;
            }
            Some((
                fields[0].to_string(),
                fields[2].to_string(),
                fields[4].to_string(),
                fields[5].to_string(),
            ))
        })
        .collect()
}

fn top_hit_rows_from_stdout(stdout: &str, n: usize) -> Vec<(String, String, String)> {
    let mut rows = Vec::new();
    let mut in_hits = false;
    for line in stdout.lines() {
        if line.contains("E-value  score  bias") && line.contains("Sequence") {
            in_hits = true;
            continue;
        }
        if !in_hits {
            continue;
        }
        if rows.len() > 0 && line.trim().is_empty() {
            break;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 9
            && (fields[0].chars().next().unwrap().is_ascii_digit() || fields[0].contains('e'))
        {
            rows.push((
                fields[0].to_string(),
                fields[1].to_string(),
                fields[8].to_string(),
            ));
            if rows.len() == n {
                break;
            }
        }
    }
    rows
}

#[test]
fn phmmer_20aa_stdout_preserves_expected_query_summaries() {
    let output = run_phmmer(
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &[],
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let normalized = normalize_phmmer_stdout(&stdout);

    assert!(normalized.contains(&"Query:       test1  [L=20]".to_string()));
    assert!(normalized.contains(&"Query:       test2  [L=25]".to_string()));
    assert!(normalized.contains(&"Query:       test3  [L=28]".to_string()));
    assert!(normalized.contains(&"Query:       test4  [L=26]".to_string()));

    let top_rows = top_hit_rows_from_stdout(&stdout, 4);
    assert_eq!(
        top_rows,
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
        "phmmer 20aa top rows changed for test1 query"
    );
}

#[test]
fn phmmer_20aa_tblout_preserves_expected_rows() {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("out.tbl");
    let query = test_path("hmmer/testsuite/20aa-alitest.fa");
    let target = test_path("hmmer/testsuite/20aa-alitest.fa");
    let output = run_phmmer(&query, &target, &["--tblout", tblout.to_str().unwrap()]);
    assert!(String::from_utf8(output.stdout).unwrap().contains("[ok]"));

    let tbl = std::fs::read_to_string(tblout).unwrap();
    let c_tbl = run_c_phmmer_tblout(&query, &target, &[]);
    assert_eq!(
        parse_tblout_rows(&tbl),
        parse_tblout_rows(&c_tbl),
        "phmmer 20aa tblout rows drifted from bundled C"
    );
    assert_eq!(
        tbl.matches("# target name").count(),
        1,
        "phmmer multi-query tblout should use a single shared header"
    );
    assert!(tbl.contains("# Program:         phmmer\n"));
    assert!(tbl.contains("# Pipeline mode:   SEARCH\n"));
    assert!(tbl.contains(&format!(
        "# Query file:      {}\n",
        test_path("hmmer/testsuite/20aa-alitest.fa")
    )));
    assert!(tbl.contains(&format!(
        "# Target file:     {}\n",
        test_path("hmmer/testsuite/20aa-alitest.fa")
    )));
    assert!(tbl.contains("# Option settings: phmmer --tblout "));
    assert!(tbl.ends_with("# [ok]\n"));
}

#[test]
fn phmmer_seed_and_calibration_tuning_match_bundled_c_rows() {
    let query = test_path("hmmer/testsuite/20aa-alitest.fa");
    let target = test_path("hmmer/testsuite/20aa-alitest.fa");
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("rust.tbl");
    let args = [
        "--seed",
        "7",
        "--EmN",
        "20",
        "--EvN",
        "20",
        "--EfN",
        "20",
        "--tblout",
        tblout.to_str().unwrap(),
    ];

    let output = run_phmmer(&query, &target, &args);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("# random number seed set to:       7\n"));
    assert!(stdout.contains("# seq number for MSV Gumbel mu fit: 20\n"));
    assert!(stdout.contains("# seq number for Vit Gumbel mu fit: 20\n"));
    assert!(stdout.contains("# seq number for Fwd exp tau fit:   20\n"));

    let rust_tbl = std::fs::read_to_string(tblout).unwrap();
    let c_tbl = run_c_phmmer_tblout(
        &query,
        &target,
        &["--seed", "7", "--EmN", "20", "--EvN", "20", "--EfN", "20"],
    );
    assert_eq!(parse_tblout_rows(&rust_tbl), parse_tblout_rows(&c_tbl));
}

#[test]
fn phmmer_domtblout_has_c_style_footer() {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("out.domtbl");
    let output = run_phmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["--domtblout", domtblout.to_str().unwrap()],
    );
    assert!(String::from_utf8(output.stdout).unwrap().contains("[ok]"));

    let domtbl = std::fs::read_to_string(domtblout).unwrap();
    assert!(domtbl.contains("# Program:         phmmer\n"));
    assert!(domtbl.contains("# Pipeline mode:   SEARCH\n"));
    assert!(domtbl.contains(&format!(
        "# Query file:      {}\n",
        test_path("hmmer/tutorial/HBB_HUMAN")
    )));
    assert!(domtbl.contains(&format!(
        "# Target file:     {}\n",
        test_path("hmmer/tutorial/globins45.fa")
    )));
    assert!(domtbl.contains("# Option settings: phmmer --domtblout "));
    assert!(domtbl.ends_with("# [ok]\n"));
}

#[test]
fn phmmer_globins45_matches_c_top_hit_order_and_score() {
    let output = run_phmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let top_hits = top_hit_rows_from_stdout(&stdout, 5);

    assert_eq!(
        top_hits
            .iter()
            .map(|(_, _, name)| name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "HBB_CALAR",
            "HBB_MANSP",
            "HBB_URSMA",
            "HBB_RABIT",
            "HBB_SUNMU",
        ],
        "phmmer globins top-hit ordering changed"
    );

    assert_eq!(top_hits[0].0, "2.7e-97");
    assert_eq!(top_hits[0].1, "314.3");
    let top_score: f64 = top_hits[0].1.parse().unwrap();
    assert!(
        (top_score - 314.3).abs() < 0.05,
        "phmmer globins top-score drift moved unexpectedly: Rust {:.1} vs C 314.3",
        top_score
    );
}

#[test]
fn phmmer_globins_tblout_matches_bundled_c_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("out.tbl");
    let output = run_phmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["--tblout", tblout.to_str().unwrap()],
    );
    assert!(String::from_utf8(output.stdout).unwrap().contains("[ok]"));
    let rust_tbl = std::fs::read_to_string(tblout).unwrap();
    let c_tbl = run_c_phmmer_tblout(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[],
    );

    assert_eq!(parse_tblout_rows(&rust_tbl), parse_tblout_rows(&c_tbl));
}

#[test]
fn phmmer_popen_pextend_and_search_space_overrides_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("out.tbl");
    let output = run_phmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &[
            "--popen",
            "0.03",
            "--pextend",
            "0.3",
            "-Z",
            "123",
            "--domZ",
            "5",
            "--tblout",
            tblout.to_str().unwrap(),
        ],
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Initial search space (Z):                123  [as set by --Z on cmdline]")
    );
    assert!(stdout
        .contains("Domain search space  (domZ):               5  [as set by --domZ on cmdline]"));
    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert!(tbl
        .lines()
        .any(|line| !line.starts_with('#') && !line.trim().is_empty()));
    assert!(tbl.contains("--popen 0.03 --pextend 0.3 -Z 123 --domZ 5 --tblout"));
}

#[test]
fn phmmer_nonull2_zeroes_bias_and_raises_globins_scores() {
    let output = run_phmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["--nonull2"],
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let top_hits = top_hit_rows_from_stdout(&stdout, 4);

    assert_eq!(
        top_hits,
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
        "phmmer --nonull2 globins top rows changed"
    );
    assert!(stdout.contains(" 1.5e-97  315.1   0.0"));
}

#[test]
fn phmmer_nonull2_globins_tblout_matches_bundled_c_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let tblout = dir.path().join("out.tbl");
    let output = run_phmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["--nonull2", "--tblout", tblout.to_str().unwrap()],
    );
    assert!(String::from_utf8(output.stdout).unwrap().contains("[ok]"));
    let rust_tbl = std::fs::read_to_string(tblout).unwrap();
    let c_tbl = run_c_phmmer_tblout(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["--nonull2"],
    );

    assert_eq!(parse_tblout_rows(&rust_tbl), parse_tblout_rows(&c_tbl));
}

#[test]
fn phmmer_nobias_is_accepted_on_globins_fixture() {
    let output = run_phmmer(
        &test_path("hmmer/tutorial/HBB_HUMAN"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["--nobias"],
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let top_hits = top_hit_rows_from_stdout(&stdout, 3);

    assert!(stdout.contains("[ok]"));
    assert_eq!(
        top_hits
            .iter()
            .map(|(_, _, name)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["HBB_CALAR", "HBB_MANSP", "HBB_URSMA"],
        "phmmer --nobias globins top-hit ordering changed"
    );
}
