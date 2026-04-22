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
            (
                "3.8e-16".to_string(),
                "44.3".to_string(),
                "test1".to_string()
            ),
            (
                "9.5e-16".to_string(),
                "43.2".to_string(),
                "test4".to_string()
            ),
            (
                "9.3e-11".to_string(),
                "28.8".to_string(),
                "test2".to_string()
            ),
            (
                "3.4e-10".to_string(),
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
    let output = run_phmmer(
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["--tblout", tblout.to_str().unwrap()],
    );
    assert!(String::from_utf8(output.stdout).unwrap().contains("[ok]"));

    let tbl = std::fs::read_to_string(tblout).unwrap();
    assert_eq!(
        parse_tblout_rows(&tbl),
        vec![
            (
                "test1".to_string(),
                "test1".to_string(),
                "3.8e-16".to_string(),
                "44.3".to_string()
            ),
            (
                "test4".to_string(),
                "test1".to_string(),
                "9.5e-16".to_string(),
                "43.2".to_string()
            ),
            (
                "test2".to_string(),
                "test1".to_string(),
                "9.3e-11".to_string(),
                "28.8".to_string()
            ),
            (
                "test3".to_string(),
                "test1".to_string(),
                "3.4e-10".to_string(),
                "27.2".to_string()
            ),
            (
                "test2".to_string(),
                "test2".to_string(),
                "4.1e-15".to_string(),
                "41.0".to_string()
            ),
            (
                "test4".to_string(),
                "test2".to_string(),
                "1.7e-11".to_string(),
                "30.7".to_string()
            ),
            (
                "test1".to_string(),
                "test2".to_string(),
                "3.1e-11".to_string(),
                "30.0".to_string()
            ),
            (
                "test3".to_string(),
                "test2".to_string(),
                "6.9e-11".to_string(),
                "29.0".to_string()
            ),
            (
                "test3".to_string(),
                "test3".to_string(),
                "1.5e-15".to_string(),
                "42.4".to_string()
            ),
            (
                "test2".to_string(),
                "test3".to_string(),
                "9.9e-11".to_string(),
                "28.6".to_string()
            ),
            (
                "test4".to_string(),
                "test3".to_string(),
                "1.7e-10".to_string(),
                "28.0".to_string()
            ),
            (
                "test1".to_string(),
                "test3".to_string(),
                "3.3e-10".to_string(),
                "27.2".to_string()
            ),
            (
                "test4".to_string(),
                "test4".to_string(),
                "6.8e-17".to_string(),
                "46.4".to_string()
            ),
            (
                "test1".to_string(),
                "test4".to_string(),
                "6.6e-16".to_string(),
                "43.6".to_string()
            ),
            (
                "test2".to_string(),
                "test4".to_string(),
                "3.7e-11".to_string(),
                "30.0".to_string()
            ),
            (
                "test3".to_string(),
                "test4".to_string(),
                "1.6e-10".to_string(),
                "28.2".to_string()
            ),
        ],
        "phmmer 20aa tblout rows changed"
    );
}

#[test]
fn phmmer_globins45_preserves_top_hit_order_and_known_score_inflation() {
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

    assert_eq!(top_hits[0].0, "4.7e-97");
    assert_eq!(top_hits[0].1, "314.3");
    let top_score: f64 = top_hits[0].1.parse().unwrap();
    assert!(
        (top_score - 314.3).abs() < 0.05,
        "phmmer globins top-score drift moved unexpectedly: Rust {:.1} vs C 314.3",
        top_score
    );
}
