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

fn dom_query_counts(rows: &[DomtblRow]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for row in rows {
        *counts.entry(row.query.clone()).or_insert(0) += 1;
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
    assert_eq!(
        query_hit_counts(&rust_rows),
        query_hit_counts(&golden_rows),
        "gecco_pfam5 per-query hit counts diverged"
    );
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
        query_hit_counts(&rust_rows),
        query_hit_counts(&golden_rows),
        "gecco_missed per-query hit counts diverged"
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
    let rows = parse_domtbl_rows(&run_hmmsearch_domtblout(
        &test_path("hmmer/testsuite/gecco_pfam5.hmm"),
        &test_path("hmmer/testsuite/gecco_proteins.faa"),
    ));
    let expected = BTreeMap::from([
        ("AAA".to_string(), 3usize),
        ("AAA_16".to_string(), 8usize),
        ("AAA_21".to_string(), 6usize),
        ("AAA_29".to_string(), 4usize),
        ("SMC_N".to_string(), 3usize),
    ]);
    assert_eq!(dom_query_counts(&rows), expected);
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
