//! GECCO-focused HMMER regression tests.
//!
//! GECCO consumes pyhmmer domain hits directly, including alignment and
//! envelope coordinates. These tests pin a small region from the GECCO
//! CP157504.1 fixture where domain-level differences change downstream CRF
//! probabilities and cluster calls.

use std::process::Command;

use hmmer_pure_rs as hmmer;

#[derive(Debug, Clone)]
struct DomainRow {
    target: String,
    domain: String,
    ali_from: usize,
    ali_to: usize,
    env_from: usize,
    env_to: usize,
    i_evalue: f64,
    pvalue: f64,
}

#[derive(Debug, Clone)]
struct FullPrecisionDomainRow {
    target: String,
    domain: String,
    ali_from: i64,
    ali_to: i64,
    env_from: i64,
    env_to: i64,
    i_evalue: f64,
    bitscore: f32,
}

fn binary_path() -> std::path::PathBuf {
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push("hmmer");
    path
}

fn test_path(relative: &str) -> String {
    format!("{}/{}", env!("CARGO_MANIFEST_DIR"), relative)
}

fn fixture_path_if_present(relative: &str) -> Option<String> {
    let path = test_path(relative);
    if std::path::Path::new(&path).exists() {
        Some(path)
    } else {
        eprintln!("skipping test; optional fixture is absent: {relative}");
        None
    }
}

fn c_hmmconvert_path() -> String {
    test_path("hmmer/src/hmmconvert")
}

fn c_hmmsearch_path() -> String {
    test_path("hmmer/src/hmmsearch")
}

fn convert_to_c_binary_hmm(text_hmm: &str, output_h3m: &std::path::Path) {
    let output = Command::new(c_hmmconvert_path())
        .args(["-b", text_hmm])
        .output()
        .expect("failed to run C hmmconvert");
    assert!(
        output.status.success(),
        "C hmmconvert failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::write(output_h3m, output.stdout).unwrap();
}

fn run_c_domtblout(hmm: &str, proteins: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("c.domtblout.txt");
    let output = Command::new(c_hmmsearch_path())
        .args([
            "--noali",
            "--cpu",
            "1",
            "-Z",
            "2766",
            "--domZ",
            "2766",
            "--domtblout",
            domtblout.to_str().unwrap(),
            hmm,
            proteins,
        ])
        .output()
        .expect("failed to run C hmmsearch");
    assert!(
        output.status.success(),
        "C hmmsearch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(domtblout).unwrap()
}

fn run_rust_domtblout(hmm: &str, proteins: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let domtblout = dir.path().join("domtblout.txt");
    let output = Command::new(binary_path())
        .args([
            "search",
            "--noali",
            "-Z",
            "2766",
            "--domZ",
            "2766",
            "--domtblout",
            domtblout.to_str().unwrap(),
            hmm,
            proteins,
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

fn parse_domtblout_fields(content: &str) -> Vec<Vec<String>> {
    content
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<String> = line
                .split_whitespace()
                .take(22)
                .map(str::to_string)
                .collect();
            assert_eq!(
                fields.len(),
                22,
                "domtblout line has too few fields: {line}"
            );
            fields
        })
        .collect()
}

fn parse_golden_tsv(path: &str) -> Vec<DomainRow> {
    let content = std::fs::read_to_string(path).unwrap();
    content
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            assert_eq!(fields.len(), 8, "bad golden row: {line}");
            DomainRow {
                target: fields[0].to_string(),
                domain: fields[1].to_string(),
                ali_from: fields[2].parse().unwrap(),
                ali_to: fields[3].parse().unwrap(),
                env_from: fields[4].parse().unwrap(),
                env_to: fields[5].parse().unwrap(),
                i_evalue: fields[6].parse().unwrap(),
                pvalue: fields[7].parse().unwrap(),
            }
        })
        .collect()
}

fn parse_rust_domtblout(content: &str) -> Vec<DomainRow> {
    let mut rows: Vec<DomainRow> = content
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert!(
                fields.len() >= 22,
                "domtblout line has too few fields: {line}"
            );
            let i_evalue: f64 = fields[12].parse().unwrap();
            DomainRow {
                target: fields[0].to_string(),
                domain: fields[4].split('.').next().unwrap().to_string(),
                ali_from: fields[17].parse().unwrap(),
                ali_to: fields[18].parse().unwrap(),
                env_from: fields[19].parse().unwrap(),
                env_to: fields[20].parse().unwrap(),
                i_evalue,
                // pyhmmer reports pvalue separately; with fixed domZ this is
                // derived from the independent E-value.
                pvalue: i_evalue / 2766.0,
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        a.target
            .cmp(&b.target)
            .then(a.domain.cmp(&b.domain))
            .then(a.ali_from.cmp(&b.ali_from))
            .then(a.ali_to.cmp(&b.ali_to))
            .then(a.env_from.cmp(&b.env_from))
            .then(a.env_to.cmp(&b.env_to))
    });
    rows
}

fn assert_rows_match_pyhmmer(expected: &[DomainRow], actual: &[DomainRow]) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "domain row count differs: expected {}, actual {}",
        expected.len(),
        actual.len()
    );

    for (idx, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(expected.target, actual.target, "row {idx} target");
        assert_eq!(expected.domain, actual.domain, "row {idx} domain");
        assert_eq!(expected.ali_from, actual.ali_from, "row {idx} ali_from");
        assert_eq!(expected.ali_to, actual.ali_to, "row {idx} ali_to");
        assert_eq!(expected.env_from, actual.env_from, "row {idx} env_from");
        assert_eq!(expected.env_to, actual.env_to, "row {idx} env_to");

        let i_rel = relative_error(expected.i_evalue, actual.i_evalue);
        assert!(
            i_rel <= 0.06,
            "row {idx} i-Evalue differs too much for {}:{}: expected {}, actual {}, relerr {}",
            expected.target,
            expected.domain,
            expected.i_evalue,
            actual.i_evalue,
            i_rel
        );

        let p_rel = relative_error(expected.pvalue, actual.pvalue);
        assert!(
            p_rel <= 0.06,
            "row {idx} pvalue differs too much for {}:{}: expected {}, actual {}, relerr {}",
            expected.target,
            expected.domain,
            expected.pvalue,
            actual.pvalue,
            p_rel
        );
    }
}

fn relative_error(expected: f64, actual: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        ((actual - expected) / expected).abs()
    }
}

fn assert_domtblout_matches_c_hmmer(hmm: &str, proteins: &str) {
    let expected = parse_domtblout_fields(&run_c_domtblout(hmm, proteins));
    let actual = parse_domtblout_fields(&run_rust_domtblout(hmm, proteins));
    assert_eq!(expected, actual);
}

fn run_api_full_precision(hmm_path: &str, proteins_path: &str) -> Vec<FullPrecisionDomainRow> {
    let hmms = hmmer::hmmfile_binary::read_binary_hmm_file(std::path::Path::new(hmm_path))
        .expect("failed to read binary HMM fixture");
    let abc = hmmer::Alphabet::amino();
    let mut sqf = hmmer::sequence::open_seq_file(std::path::Path::new(proteins_path), &abc)
        .expect("failed to read protein fixture");
    let mut seqs = Vec::new();
    let mut sq = hmmer::sequence::Sequence::new();
    while sqf.read(&mut sq).unwrap() {
        seqs.push(sq.clone());
        sq.reuse();
    }

    let mut rows = Vec::new();
    for hmm_profile in &hmms {
        let abc = hmmer::Alphabet::new(hmm_profile.abc_type);
        let mut bg = hmmer::Bg::new(&abc);
        let mut gm = hmmer::Profile::new(hmm_profile.m, &abc);
        hmmer::profile::profile_config(hmm_profile, &bg, &mut gm, 100, hmmer::profile::P7_LOCAL);
        bg.set_filter(hmm_profile.m, &hmm_profile.compo);
        let mut om = hmmer::OProfile::convert(&gm);
        let mut pli = hmmer::Pipeline::new();
        pli.new_model(&gm);
        pli.z = 2766.0;
        pli.domz = 2766.0;
        pli.z_setby = hmmer::pipeline::ZSetBy::Option;
        pli.domz_setby = hmmer::pipeline::ZSetBy::Option;
        pli.do_alignment_display = false;

        let mut hits = hmmer::TopHits::new();
        for seq in &seqs {
            pli.n_targets = 0;
            pli.n_past_msv = 0;
            pli.n_past_bias = 0;
            pli.n_past_vit = 0;
            pli.n_past_fwd = 0;
            let mut local_bg = bg.clone();
            local_bg.set_length(seq.n);
            let mut local_hits = hmmer::TopHits::new();
            if pli.run(
                &mut gm,
                &mut om,
                &local_bg,
                hmm_profile,
                seq,
                &mut local_hits,
            ) {
                hits.hits.extend(local_hits.hits.into_iter());
            }
        }
        hits.sort_by_sortkey();
        hits.threshold(&pli, 2766.0, 2766.0);

        let domain = hmm_profile
            .acc
            .as_deref()
            .unwrap_or(hmm_profile.name.as_str())
            .to_string();
        for hit in &hits.hits {
            for dom in &hit.dcl {
                if dom.is_reported {
                    rows.push(FullPrecisionDomainRow {
                        target: hit.name.clone(),
                        domain: domain.clone(),
                        ali_from: dom.iali,
                        ali_to: dom.jali,
                        env_from: dom.ienv,
                        env_to: dom.jenv,
                        i_evalue: 2766.0 * dom.lnp.exp(),
                        bitscore: dom.bitscore,
                    });
                }
            }
        }
    }

    rows.sort_by(|a, b| {
        a.target
            .cmp(&b.target)
            .then(a.domain.cmp(&b.domain))
            .then(a.ali_from.cmp(&b.ali_from))
            .then(a.ali_to.cmp(&b.ali_to))
            .then(a.env_from.cmp(&b.env_from))
            .then(a.env_to.cmp(&b.env_to))
    });
    rows
}

#[test]
fn gecco_cluster1_domain_rows_match_pyhmmer_projection() {
    let Some(proteins) = fixture_path_if_present("test_data/gecco_cluster1_proteins.faa") else {
        return;
    };
    let expected = parse_golden_tsv(&test_path("tests/golden/gecco_cluster1_pyhmmer.tsv"));
    let actual = parse_rust_domtblout(&run_rust_domtblout(
        &test_path("test_data/gecco_cluster1_hmms.hmm"),
        &proteins,
    ));

    assert_rows_match_pyhmmer(&expected, &actual);
}

#[test]
fn gecco_cluster1_default_filter_rows_match_pyhmmer_projection() {
    let Some(proteins) = fixture_path_if_present("test_data/gecco_cluster1_proteins.faa") else {
        return;
    };
    let expected: Vec<DomainRow> =
        parse_golden_tsv(&test_path("tests/golden/gecco_cluster1_pyhmmer.tsv"))
            .into_iter()
            .filter(|row| row.pvalue < 1e-9)
            .collect();
    let actual: Vec<DomainRow> = parse_rust_domtblout(&run_rust_domtblout(
        &test_path("test_data/gecco_cluster1_hmms.hmm"),
        &proteins,
    ))
    .into_iter()
    .filter(|row| row.pvalue < 1e-9)
    .collect();

    assert_rows_match_pyhmmer(&expected, &actual);
}

#[test]
fn gecco_cluster1_c_binary_h3m_rows_match_pyhmmer_projection() {
    let Some(proteins) = fixture_path_if_present("test_data/gecco_cluster1_proteins.faa") else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let h3m = dir.path().join("gecco_cluster1_hmms.h3m");
    convert_to_c_binary_hmm(&test_path("test_data/gecco_cluster1_hmms.hmm"), &h3m);

    let expected = parse_golden_tsv(&test_path("tests/golden/gecco_cluster1_pyhmmer.tsv"));
    let actual = parse_rust_domtblout(&run_rust_domtblout(h3m.to_str().unwrap(), &proteins));

    assert_rows_match_pyhmmer(&expected, &actual);
}

#[test]
fn gecco_cluster1_text_hmm_domtblout_matches_c_hmmer() {
    let Some(proteins) = fixture_path_if_present("test_data/gecco_cluster1_proteins.faa") else {
        return;
    };
    assert_domtblout_matches_c_hmmer(&test_path("test_data/gecco_cluster1_hmms.hmm"), &proteins);
}

#[test]
fn gecco_cluster1_c_binary_h3m_domtblout_matches_c_hmmer() {
    let Some(proteins) = fixture_path_if_present("test_data/gecco_cluster1_proteins.faa") else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let h3m = dir.path().join("gecco_cluster1_hmms.h3m");
    convert_to_c_binary_hmm(&test_path("test_data/gecco_cluster1_hmms.hmm"), &h3m);

    assert_domtblout_matches_c_hmmer(h3m.to_str().unwrap(), &proteins);
}

#[test]
fn gecco_full_pfam_selected_binary_h3m_rows_match_pyhmmer_projection() {
    let Some(hmms) = fixture_path_if_present("test_data/gecco_full_pfam_selected_hmms.h3m") else {
        return;
    };
    let expected = parse_golden_tsv(&test_path(
        "tests/golden/gecco_full_pfam_selected_pyhmmer.tsv",
    ));
    let actual = parse_rust_domtblout(&run_rust_domtblout(
        &hmms,
        &test_path("test_data/gecco_full_pfam_selected_proteins.faa"),
    ));

    assert_rows_match_pyhmmer(&expected, &actual);
}

#[test]
fn gecco_full_pfam_selected_binary_h3m_domtblout_matches_c_hmmer() {
    let Some(hmms) = fixture_path_if_present("test_data/gecco_full_pfam_selected_hmms.h3m") else {
        return;
    };
    assert_domtblout_matches_c_hmmer(
        &hmms,
        &test_path("test_data/gecco_full_pfam_selected_proteins.faa"),
    );
}

#[test]
fn gecco_full_pfam_selected_default_filter_rows_match_pyhmmer_projection() {
    let Some(hmms) = fixture_path_if_present("test_data/gecco_full_pfam_selected_hmms.h3m") else {
        return;
    };
    let expected: Vec<DomainRow> = parse_golden_tsv(&test_path(
        "tests/golden/gecco_full_pfam_selected_pyhmmer.tsv",
    ))
    .into_iter()
    .filter(|row| row.pvalue < 1e-9)
    .collect();
    let actual: Vec<DomainRow> = parse_rust_domtblout(&run_rust_domtblout(
        &hmms,
        &test_path("test_data/gecco_full_pfam_selected_proteins.faa"),
    ))
    .into_iter()
    .filter(|row| row.pvalue < 1e-9)
    .collect();

    assert_rows_match_pyhmmer(&expected, &actual);
}

#[test]
fn gecco_pfam_stochastic_domains_match_pyhmmer_full_precision() {
    let rows = run_api_full_precision(
        &test_path("test_data/gecco_pfam_stochastic_selected_hmms.h3m"),
        &test_path("test_data/gecco_pfam_stochastic_selected_proteins.faa"),
    );
    let expected = [
        (
            "CP157504.1_3476",
            "PF07690.19",
            38,
            374,
            35,
            379,
            4.418183328172337e-30,
            102.12395477294922_f32,
        ),
        (
            "CP157504.1_3476",
            "PF07690.19",
            325,
            460,
            324,
            473,
            4.6307416971601415e-9,
            32.98520278930664_f32,
        ),
        (
            "CP157504.1_67",
            "PF01434.21",
            413,
            603,
            413,
            603,
            4.3720603042074165e-81,
            268.666259765625_f32,
        ),
        (
            "CP157504.1_5311",
            "PF00912.25",
            62,
            237,
            61,
            238,
            1.4665650306276348e-66,
            220.62559509277344_f32,
        ),
    ];

    for expected in expected {
        let row = rows
            .iter()
            .find(|row| {
                row.target == expected.0
                    && row.domain == expected.1
                    && row.ali_from == expected.2
                    && row.ali_to == expected.3
                    && row.env_from == expected.4
                    && row.env_to == expected.5
            })
            .unwrap_or_else(|| panic!("missing expected full-precision row: {expected:?}"));
        assert_eq!(row.bitscore.to_bits(), expected.7.to_bits());
        assert_eq!(row.i_evalue.to_bits(), (expected.6 as f64).to_bits());
    }
}
