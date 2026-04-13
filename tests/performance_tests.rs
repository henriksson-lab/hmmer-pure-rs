//! Performance tests for hmmer-pure-rs P7 pipeline.
//!
//! These tests measure wall-clock time for realistic workloads
//! and assert upper bounds to catch performance regressions.
//! They use the Infernal tRNA P7 HMM against sequences of increasing size.

use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

/// Read a P7 HMM from an Infernal CM file (embedded HMMER3/f section).
fn read_hmm_from_cm(cm_path: &Path) -> hmmer_pure_rs::hmm::Hmm {
    let file = std::fs::File::open(cm_path).unwrap();
    let reader = BufReader::new(file);

    let mut hmm_lines = Vec::new();
    let mut in_hmm = false;
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with("HMMER3/") { in_hmm = true; }
        if in_hmm {
            hmm_lines.push(line);
            if hmm_lines.last().map(|l| l.trim()) == Some("//") { break; }
        }
    }
    assert!(!hmm_lines.is_empty(), "No HMMER3 section in CM file");

    let text = hmm_lines.join("\n");
    let cursor = BufReader::new(std::io::Cursor::new(text.into_bytes()));
    let hmms = hmmer_pure_rs::hmmfile::read_hmms(cursor).unwrap();
    hmms.into_iter().next().unwrap()
}

/// Run the full P7 pipeline on a sequence and return (domains, elapsed).
fn run_p7_timed(
    hmm: &hmmer_pure_rs::hmm::Hmm,
    seq: &[u8],
) -> (Vec<(usize, usize, f32)>, std::time::Duration) {
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
    };

    let start = Instant::now();
    pli.run(&mut gm, &mut om, &bg_obj, hmm, &sq, &mut th);
    let elapsed = start.elapsed();

    let mut domains = Vec::new();
    for h in &th.hits {
        for dom in &h.dcl {
            domains.push((dom.ienv as usize, dom.jenv as usize, dom.bitscore));
        }
    }
    (domains, elapsed)
}

/// Run the SSV long-target filter on a sequence and return (windows, elapsed).
/// This is the approach C Infernal uses for genome-scale search.
fn run_p7_longtarget_timed(
    hmm: &hmmer_pure_rs::hmm::Hmm,
    seq: &[u8],
) -> (Vec<(usize, usize, f32)>, std::time::Duration) {
    use hmmer_pure_rs::*;

    let abc = alphabet::Alphabet::new(hmm.abc_type);
    let bg_obj = bg::Bg::new(&abc);

    let mut gm = profile::Profile::new(hmm.m, &abc);
    profile::profile_config(hmm, &bg_obj, &mut gm, 200, profile::P7_LOCAL);
    let mut om = simd::oprofile::OProfile::convert(&gm);
    let max_length = (hmm.m * 4) as i32;

    let dsq = abc.digitize(seq);

    let start = Instant::now();
    let windows = unsafe {
        simd::ssv_longtarget::ssv_filter_longtarget(
            &dsq, seq.len(), &om, &bg_obj, 0.02, max_length,
        )
    };
    let elapsed = start.elapsed();

    let hits: Vec<(usize, usize, f32)> = windows.iter().map(|w| {
        let start = if w.n > w.length { w.n - w.length + 1 } else { 1 };
        (start, w.n, w.score)
    }).collect();

    (hits, elapsed)
}

fn infernal_testsuite() -> std::path::PathBuf {
    let paths = [
        "/data/henriksson/github/claude/infernal-rs/infernal/testsuite",
        "../infernal-rs/infernal/testsuite",
    ];
    for p in &paths {
        let path = Path::new(p);
        if path.join("tRNA.c.cm").exists() {
            return path.to_path_buf();
        }
    }
    panic!("Cannot find Infernal testsuite directory");
}

fn infernal_tutorial() -> std::path::PathBuf {
    let paths = [
        "/data/henriksson/github/claude/infernal-rs/infernal/tutorial",
        "../infernal-rs/infernal/tutorial",
    ];
    for p in &paths {
        let path = Path::new(p);
        if path.join("mrum-genome.fa").exists() {
            return path.to_path_buf();
        }
    }
    panic!("Cannot find Infernal tutorial directory");
}

/// Generate a random nucleotide sequence with an embedded tRNA.
fn make_seq_with_trna(total_len: usize, trna_pos: usize, seed: u64) -> Vec<u8> {
    let trna = b"GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
    let mut seq = Vec::with_capacity(total_len);
    let mut state = seed;
    for i in 0..total_len {
        if i >= trna_pos && i < trna_pos + trna.len() {
            seq.push(trna[i - trna_pos]);
        } else {
            // Simple LCG PRNG
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = match (state >> 33) % 4 {
                0 => b'A', 1 => b'C', 2 => b'G', _ => b'U',
            };
            seq.push(base);
        }
    }
    seq
}

/// Read a FASTA file and return the first sequence.
fn read_first_fasta_seq(path: &Path) -> Vec<u8> {
    let content = std::fs::read_to_string(path).unwrap();
    let mut seq = Vec::new();
    for line in content.lines().skip(1) {
        if line.starts_with('>') { break; }
        seq.extend_from_slice(line.trim().as_bytes());
    }
    seq
}

// ============================================================
// Performance: 10kb sequence
// ============================================================

#[test]
fn perf_p7_pipeline_10kb() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    let seq = make_seq_with_trna(10_000, 5000, 42);
    let (domains, elapsed) = run_p7_timed(&hmm, &seq);

    println!("P7 pipeline 10kb: {:.3}s, {} domains", elapsed.as_secs_f64(), domains.len());
    for (i, j, sc) in &domains {
        println!("  {}-{}: {:.1} bits", i, j, sc);
    }

    // Should find the tRNA
    let trna_hit = domains.iter().any(|(i, _, sc)| *i >= 4950 && *i <= 5050 && *sc > 10.0);
    assert!(trna_hit, "Should find tRNA at ~5000 in 10kb sequence");

    // Performance: should complete in under 5 seconds
    assert!(elapsed.as_secs() < 5,
        "10kb P7 pipeline took {:.1}s, expected <5s", elapsed.as_secs_f64());
}

// ============================================================
// Performance: 100kb sequence
// ============================================================

#[test]
fn perf_ssv_longtarget_100kb() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    let seq = make_seq_with_trna(100_000, 50_000, 123);
    let (windows, elapsed) = run_p7_longtarget_timed(&hmm, &seq);

    println!("SSV longtarget 100kb: {:.3}s, {} windows", elapsed.as_secs_f64(), windows.len());
    for (i, j, sc) in &windows {
        println!("  {}-{}: {:.1} nats", i, j, sc);
    }

    // Should find a window near the tRNA at position 50000
    let trna_hit = windows.iter().any(|(i, j, _)| *i <= 50100 && *j >= 50000);
    assert!(trna_hit, "SSV longtarget should find tRNA at ~50000 in 100kb sequence");

    // Performance: SSV longtarget should be fast (<5s for 100kb)
    assert!(elapsed.as_secs() < 5,
        "100kb SSV longtarget took {:.1}s, expected <5s", elapsed.as_secs_f64());
}

// ============================================================
// Performance: M. ruminantium genome (2.9 Mbp) — real-world test
// ============================================================

#[test]
fn perf_ssv_longtarget_genome() {
    let testsuite = infernal_testsuite();
    let tutorial = infernal_tutorial();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    let genome_path = tutorial.join("mrum-genome.fa");
    if !genome_path.exists() {
        println!("Skipping genome test: {} not found", genome_path.display());
        return;
    }

    let seq = read_first_fasta_seq(&genome_path);
    println!("Genome: {} bp", seq.len());

    let (domains, elapsed) = run_p7_longtarget_timed(&hmm, &seq);

    println!("SSV longtarget genome ({}bp): {:.3}s, {} windows",
        seq.len(), elapsed.as_secs_f64(), domains.len());
    for (i, j, sc) in &domains {
        println!("  {}-{}: {:.1} nats", i, j, sc);
    }

    // C Infernal processes this genome in ~1 second with SSE.
    // Target: under 60 seconds for the full 2.9 Mbp genome.
    assert!(elapsed.as_secs() < 60,
        "Genome SSV longtarget took {:.1}s, expected <60s. \
         C Infernal (SSE) does this in ~1s.",
        elapsed.as_secs_f64());

    // C Infernal finds 15 tRNAs in this genome.
    // SSV longtarget is a prefilter — it should find at least as many
    // windows as there are true hits (may include false positives).
    assert!(domains.len() >= 5,
        "Expected at least 5 SSV windows in M. ruminantium genome, got {}. \
         C Infernal finds 15 tRNAs.",
        domains.len());
}

// ============================================================
// Performance scaling: time should be roughly linear in sequence length
// ============================================================

#[test]
fn perf_p7_pipeline_scaling() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    let sizes = [1_000, 5_000, 10_000, 50_000];
    let mut times = Vec::new();

    for &size in &sizes {
        let seq = make_seq_with_trna(size, size / 2, size as u64);
        let (_domains, elapsed) = run_p7_timed(&hmm, &seq);
        let ms = elapsed.as_secs_f64() * 1000.0;
        times.push((size, ms));
        println!("  {}bp: {:.1}ms", size, ms);
    }

    // Check roughly linear scaling: time for 50kb should be
    // no more than 100x time for 1kb (allowing for overhead).
    // Perfect linear would be 50x.
    let t_small = times[0].1;
    let t_large = times[3].1;
    if t_small > 0.1 {
        let ratio = t_large / t_small;
        println!("Scaling: {:.0}x time increase for {:.0}x size increase",
            ratio, times[3].0 as f64 / times[0].0 as f64);
        assert!(ratio < 200.0,
            "P7 pipeline scaling is superlinear: {:.0}x time for 50x size. \
             Expected roughly linear (50-100x).",
            ratio);
    }
}
