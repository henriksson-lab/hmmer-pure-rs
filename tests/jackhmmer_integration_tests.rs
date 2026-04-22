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

fn round_block<'a>(stdout: &'a str, round: usize) -> &'a str {
    let marker = format!("@@ Round: {}", round);
    let start = stdout.find(&marker).unwrap();
    let rest = &stdout[start..];
    if let Some(next) = rest[marker.len()..].find("@@ Round:") {
        &rest[..marker.len() + next]
    } else {
        rest
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

#[test]
fn jackhmmer_round1_20aa_matches_current_single_sequence_baseline() {
    let stdout = run_jackhmmer(
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["-N", "1"],
    );
    let round1 = round_block(&stdout, 1);

    assert!(stdout.contains("[ok]"));
    assert!(round1.contains("Query:       test1  [M=20]"));
    assert_eq!(
        top_hit_rows(round1, 4),
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
        "jackhmmer round-1 20aa hits changed"
    );
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

    assert!(stdout.contains("@@ 45 included, 45 new. Continuing to next round."));
    assert!(stdout.contains("@@ CONVERGED (in 2 rounds)."));

    assert_eq!(
        top_hit_rows(round1, 5),
        vec![
            (
                "4.7e-97".to_string(),
                "314.3".to_string(),
                "HBB_CALAR".to_string()
            ),
            (
                "7.4e-97".to_string(),
                "313.6".to_string(),
                "HBB_MANSP".to_string()
            ),
            (
                "4.2e-91".to_string(),
                "295.0".to_string(),
                "HBB_URSMA".to_string()
            ),
            (
                "6.3e-91".to_string(),
                "294.4".to_string(),
                "HBB_RABIT".to_string()
            ),
            (
                "9.4e-84".to_string(),
                "271.1".to_string(),
                "HBB_SUNMU".to_string()
            ),
        ],
        "jackhmmer round-1 globins hits changed"
    );

    assert!(round2.contains("Query:       HBB_HUMAN-i2  [M=146]"));
    assert_eq!(
        top_hit_rows(round2, 5),
        vec![
            (
                "2.2e-33".to_string(),
                "103.0".to_string(),
                "HBB_RABIT".to_string()
            ),
            (
                "4.2e-33".to_string(),
                "102.1".to_string(),
                "HBB_URSMA".to_string()
            ),
            (
                "1.4e-32".to_string(),
                "100.6".to_string(),
                "HBB_MANSP".to_string()
            ),
            (
                "1.5e-32".to_string(),
                "100.4".to_string(),
                "HBB_CALAR".to_string()
            ),
            (
                "4.6e-32".to_string(),
                "99.0".to_string(),
                "HBE_PONPY".to_string()
            ),
        ],
        "jackhmmer round-2 globins hits changed"
    );
}

#[test]
fn jackhmmer_strict_thresholds_stop_after_empty_round_on_20aa() {
    let stdout = run_jackhmmer(
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["-N", "2", "-E", "1e-20", "--incE", "1e-20"],
    );

    assert!(stdout.contains("@@ Round: 1"));
    assert!(stdout.contains("[No hits detected that satisfy reporting thresholds]"));
    assert!(stdout.contains("@@ 0 included, 0 new. Continuing to next round."));
    assert!(stdout.contains("@@ Round: 2"));
    assert!(stdout.contains("@@ No hits to build MSA from. Stopping."));
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

    assert!(stdout.contains("@@ 38 included, 38 new. Continuing to next round."));
    assert!(stdout.contains("@@ CONVERGED (in 2 rounds)."));

    assert_eq!(
        top_hit_rows(round1, 5),
        vec![
            (
                "4.7e-97".to_string(),
                "314.3".to_string(),
                "HBB_CALAR".to_string()
            ),
            (
                "7.4e-97".to_string(),
                "313.6".to_string(),
                "HBB_MANSP".to_string()
            ),
            (
                "4.2e-91".to_string(),
                "295.0".to_string(),
                "HBB_URSMA".to_string()
            ),
            (
                "6.3e-91".to_string(),
                "294.4".to_string(),
                "HBB_RABIT".to_string()
            ),
            (
                "9.4e-84".to_string(),
                "271.1".to_string(),
                "HBB_SUNMU".to_string()
            ),
        ],
        "jackhmmer strict-threshold round-1 globins hits changed"
    );

    assert!(round2.contains("Query:       HBB_HUMAN-i2  [M=145]"));
    assert_eq!(
        top_hit_rows(round2, 5),
        vec![
            (
                "3.3e-45".to_string(),
                "141.7".to_string(),
                "HBB_RABIT".to_string()
            ),
            (
                "3.9e-45".to_string(),
                "141.5".to_string(),
                "HBB_URSMA".to_string()
            ),
            (
                "5.7e-45".to_string(),
                "140.9".to_string(),
                "HBB_MANSP".to_string()
            ),
            (
                "9.7e-45".to_string(),
                "140.2".to_string(),
                "HBB_CALAR".to_string()
            ),
            (
                "3.1e-44".to_string(),
                "138.7".to_string(),
                "HBE_PONPY".to_string()
            ),
        ],
        "jackhmmer strict-threshold round-2 globins hits changed"
    );
}

#[test]
fn jackhmmer_tblout_20aa_reports_final_round_rows() {
    let (stdout, tblout) = run_jackhmmer_with_tblout(
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["-N", "1"],
    );

    assert!(stdout.contains("[ok]"));
    assert_eq!(
        tblout_rows(&tblout, 4),
        vec![
            (
                "test1".to_string(),
                "3.8e-16".to_string(),
                "44.3".to_string()
            ),
            (
                "test4".to_string(),
                "9.5e-16".to_string(),
                "43.2".to_string()
            ),
            (
                "test2".to_string(),
                "9.3e-11".to_string(),
                "28.8".to_string()
            ),
            (
                "test3".to_string(),
                "3.4e-10".to_string(),
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

    assert!(stdout.contains("@@ CONVERGED (in 2 rounds)."));
    assert_eq!(
        tblout_rows(&tblout, 5),
        vec![
            (
                "HBB_RABIT".to_string(),
                "2.2e-33".to_string(),
                "103.0".to_string()
            ),
            (
                "HBB_URSMA".to_string(),
                "4.2e-33".to_string(),
                "102.1".to_string()
            ),
            (
                "HBB_MANSP".to_string(),
                "1.4e-32".to_string(),
                "100.6".to_string()
            ),
            (
                "HBB_CALAR".to_string(),
                "1.5e-32".to_string(),
                "100.4".to_string()
            ),
            (
                "HBE_PONPY".to_string(),
                "4.6e-32".to_string(),
                "99.0".to_string()
            ),
        ],
        "jackhmmer --tblout should reflect the final converged round"
    );
}

#[test]
fn jackhmmer_domtblout_20aa_reports_round1_domain_rows() {
    let (stdout, domtblout) = run_jackhmmer_with_domtblout(
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["-N", "1"],
    );

    assert!(stdout.contains("[ok]"));
    let rows = domtblout_rows(&domtblout, 4);
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0][0], "test1");
    assert_eq!(rows[0][3], "test1");
    assert_eq!(rows[0][5], "20");
    assert_eq!(rows[0][6], "3.8e-16");
    assert_eq!(rows[0][13], "44.3");
    assert_eq!(&rows[0][15..21], ["1", "20", "1", "20", "1", "20"]);

    assert_eq!(rows[1][0], "test4");
    assert_eq!(rows[1][6], "9.5e-16");
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

    assert!(stdout.contains("@@ CONVERGED (in 2 rounds)."));
    let rows = domtblout_rows(&domtblout, 5);
    assert_eq!(rows.len(), 5);

    assert_eq!(rows[0][0], "HBB_RABIT");
    assert_eq!(rows[0][3], "HBB_HUMAN");
    assert_eq!(rows[0][5], "146");
    assert_eq!(rows[0][6], "2.2e-33");
    assert_eq!(rows[0][13], "102.8");
    assert_eq!(&rows[0][15..21], ["1", "146", "1", "146", "1", "146"]);

    assert_eq!(rows[1][0], "HBB_URSMA");
    assert_eq!(rows[1][13], "102.0");
    assert_eq!(rows[2][0], "HBB_MANSP");
    assert_eq!(rows[2][13], "100.4");
    assert_eq!(rows[3][0], "HBB_CALAR");
    assert_eq!(rows[3][13], "100.3");
    assert_eq!(rows[4][0], "HBE_PONPY");
    assert_eq!(rows[4][13], "98.8");
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
                "2.6e-97".to_string(),
                "315.1".to_string(),
                "HBB_CALAR".to_string()
            ),
            (
                "5.1e-97".to_string(),
                "314.2".to_string(),
                "HBB_MANSP".to_string()
            ),
            (
                "3.2e-91".to_string(),
                "295.4".to_string(),
                "HBB_URSMA".to_string()
            ),
            (
                "4.3e-91".to_string(),
                "294.9".to_string(),
                "HBB_RABIT".to_string()
            ),
        ],
        "jackhmmer --nonull2 round-1 globins top rows changed"
    );
    assert!(round1.contains(" 2.6e-97  315.1   0.0"));
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
