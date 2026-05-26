//! Pipeline sensitivity tests.
//!
//! The P7 pipeline (SSV → Viterbi → Forward → domain definition) correctly
//! finds hits when given appropriate sequence windows. The sensitivity gap
//! in genome-scale searches is in the SSV long-target filter missing some
//! windows, not in the pipeline rejecting true hits.
//!
//! These tests verify:
//! 1. Pipeline finds tRNAs in short sequences (76bp, 1000bp)
//! 2. Pipeline correctly handles windows from SSV
//! 3. SSV long-target sensitivity on E. coli genome

use std::io::{BufRead, BufReader};
use std::path::Path;

fn read_hmm_from_cm(cm_path: &Path) -> hmmer_pure_rs::hmm::Hmm {
    let file = std::fs::File::open(cm_path).unwrap();
    let reader = BufReader::new(file);
    let mut hmm_lines = Vec::new();
    let mut in_hmm = false;
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with("HMMER3/") {
            in_hmm = true;
        }
        if in_hmm {
            hmm_lines.push(line);
            if hmm_lines.last().map(|l| l.trim()) == Some("//") {
                break;
            }
        }
    }
    let text = hmm_lines.join("\n");
    let cursor = BufReader::new(std::io::Cursor::new(text.into_bytes()));
    hmmer_pure_rs::hmmfile::read_hmms(cursor)
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
}

fn run_pipeline(hmm: &hmmer_pure_rs::hmm::Hmm, seq: &[u8]) -> Vec<(usize, usize, f32)> {
    use hmmer_pure_rs::*;
    let abc = alphabet::Alphabet::new(hmm.abc_type);
    let bg_obj = bg::Bg::new(&abc);
    let mut gm = profile::Profile::new(hmm.m, &abc);
    profile::profile_config(hmm, &bg_obj, &mut gm, 200, profile::P7_LOCAL);
    let mut om = simd::oprofile::OProfile::convert(&gm);
    let mut pli = pipeline::Pipeline::new();
    pli.new_model(&gm);
    let mut th = tophits::TopHits::new();
    let dsq = abc.digitize(seq);
    let sq = sequence::Sequence {
        name: "target".to_string(),
        acc: String::new(),
        desc: String::new(),
        dsq,
        n: seq.len(),
        l: seq.len(),
        taxid: -1,
    };
    pli.run(&mut gm, &mut om, &bg_obj, hmm, &sq, &mut th);
    th.hits
        .iter()
        .flat_map(|h| {
            h.dcl
                .iter()
                .map(|d| (d.ienv as usize, d.jenv as usize, d.bitscore))
        })
        .collect()
}

fn infernal_testsuite() -> std::path::PathBuf {
    for p in [
        "/data/henriksson/github/claude/infernal-rs/infernal/testsuite",
        "../infernal-rs/infernal/testsuite",
    ] {
        let path = Path::new(p);
        if path.join("tRNA.c.cm").exists() {
            return path.to_path_buf();
        }
    }
    panic!("Cannot find Infernal testsuite directory");
}

#[test]
fn pipeline_finds_trna_76bp() {
    let hmm = read_hmm_from_cm(&infernal_testsuite().join("tRNA.c.cm"));
    let seq = b"GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
    let domains = run_pipeline(&hmm, seq);
    println!("Pipeline 76bp: {} domains", domains.len());
    for (i, j, sc) in &domains {
        println!("  {}-{}: {:.1}", i, j, sc);
    }
    assert!(!domains.is_empty(), "Pipeline should find tRNA in 76bp");
    assert!(
        domains[0].2 > 20.0,
        "Score should be >20, got {:.1}",
        domains[0].2
    );
}

#[test]
fn pipeline_finds_trna_1000bp() {
    let hmm = read_hmm_from_cm(&infernal_testsuite().join("tRNA.c.cm"));
    let fa_path = infernal_testsuite().join("1k-tRNA.fa");
    if !fa_path.exists() {
        return;
    }
    let content = std::fs::read_to_string(&fa_path).unwrap();
    let mut seq = Vec::new();
    for line in content.lines().skip(1) {
        if line.starts_with('>') {
            break;
        }
        seq.extend_from_slice(line.trim().as_bytes());
    }
    let domains = run_pipeline(&hmm, &seq);
    println!("Pipeline 1000bp: {} domains", domains.len());
    for (i, j, sc) in &domains {
        println!("  {}-{}: {:.1}", i, j, sc);
    }
    assert!(
        domains
            .iter()
            .any(|(i, _, sc)| *i >= 830 && *i <= 900 && *sc > 10.0),
        "Pipeline should find tRNA near position 860 in 1000bp"
    );
}

#[test]
fn ssv_sensitivity_ecoli_genome() {
    let hmm = read_hmm_from_cm(&infernal_testsuite().join("tRNA.c.cm"));
    let ecoli_path = Path::new("/tmp/ecoli_k12.fna");
    if !ecoli_path.exists() {
        println!("Skipping: /tmp/ecoli_k12.fna not found");
        return;
    }

    let content = std::fs::read_to_string(ecoli_path).unwrap();
    let mut genome = Vec::new();
    for line in content.lines() {
        if line.starts_with('>') {
            continue;
        }
        genome.extend_from_slice(line.trim().as_bytes());
    }

    use hmmer_pure_rs::*;
    let abc = alphabet::Alphabet::new(hmm.abc_type);
    let bg_obj = bg::Bg::new(&abc);
    let mut gm = profile::Profile::new(hmm.m, &abc);
    profile::profile_config(&hmm, &bg_obj, &mut gm, 200, profile::P7_LOCAL);
    let om = simd::oprofile::OProfile::convert(&gm);
    let dsq = abc.digitize(&genome);

    let windows = unsafe {
        simd::ssv_longtarget::ssv_filter_longtarget(
            &dsq,
            genome.len(),
            &om,
            &bg_obj,
            0.02,
            (hmm.m * 4) as i32,
        )
    };

    println!(
        "SSV on E. coli ({}bp): {} windows",
        genome.len(),
        windows.len()
    );

    // C Infernal finds 89 tRNAs. SSV should produce windows covering most of them.
    // Known SSV misses: ~4-5 positions out of 89.
    assert!(
        windows.len() >= 50,
        "SSV should find at least 50 windows in E. coli, got {}",
        windows.len()
    );
}
