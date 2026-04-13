//! Tests for P7 domain envelope precision.
//!
//! When a short model (~70 positions) is searched against a long sequence
//! (~1000bp) containing one true hit, the P7 pipeline should return a
//! tight domain envelope around the hit, NOT the entire sequence.
//!
//! These tests use the Infernal tRNA CM's embedded P7 filter HMM
//! (HMMER3/f format, M=71) against known tRNA-containing sequences.
//!
//! Bug tracked: P7 domain definition returns ienv=1,jenv=L for the
//! entire sequence instead of narrowing to the hit region.

use std::io::BufReader;
use std::path::Path;

/// Read a P7 HMM from an Infernal CM file (embedded HMMER3/f section).
fn read_hmm_from_cm(cm_path: &Path) -> hmmer_pure_rs::hmm::Hmm {
    use std::io::BufRead;
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

/// Domain result with both hit-level and domain-level scores.
#[derive(Debug, Clone)]
struct DomainResult {
    ienv: usize,
    jenv: usize,
    dom_bitscore: f32,
    hit_score: f32,
}

/// Run the P7 pipeline on a sequence and return domain envelopes with scores.
fn run_p7_pipeline_full(hmm: &hmmer_pure_rs::hmm::Hmm, seq: &[u8]) -> Vec<DomainResult> {
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
        name: "test".to_string(),
        acc: String::new(),
        desc: String::new(),
        dsq,
        n: seq.len(),
        l: seq.len(),
    };

    pli.run(&mut gm, &mut om, &bg_obj, hmm, &sq, &mut th);

    let mut results = Vec::new();
    for h in &th.hits {
        for dom in &h.dcl {
            results.push(DomainResult {
                ienv: dom.ienv as usize,
                jenv: dom.jenv as usize,
                dom_bitscore: dom.bitscore,
                hit_score: h.score,
            });
        }
    }
    results
}

/// Run the P7 pipeline on a sequence and return domain envelopes.
fn run_p7_pipeline(hmm: &hmmer_pure_rs::hmm::Hmm, seq: &[u8]) -> Vec<(usize, usize, f32)> {
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
        name: "test".to_string(),
        acc: String::new(),
        desc: String::new(),
        dsq,
        n: seq.len(),
        l: seq.len(),
    };

    pli.run(&mut gm, &mut om, &bg_obj, hmm, &sq, &mut th);

    let mut domains = Vec::new();
    for h in &th.hits {
        for dom in &h.dcl {
            domains.push((dom.ienv as usize, dom.jenv as usize, dom.bitscore));
        }
    }
    domains
}

fn infernal_testsuite() -> std::path::PathBuf {
    // Try common locations for Infernal test data
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

// ============================================================
// Test 1: Short tRNA sequence (76bp) — envelope should be tight
// ============================================================

#[test]
fn domain_envelope_short_trna_is_tight() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    // Pure 76bp tRNA sequence
    let seq = b"GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
    let domains = run_p7_pipeline(&hmm, seq);

    println!("Short tRNA (76bp) domains:");
    for (i, j, sc) in &domains {
        println!("  ienv={} jenv={} score={:.1}", i, j, sc);
    }

    assert!(!domains.is_empty(), "P7 should find tRNA in 76bp sequence");

    let (ienv, jenv, _score) = &domains[0];
    // Envelope should cover most of the sequence since it IS the tRNA
    assert!(*ienv <= 10, "ienv should be near start, got {}", ienv);
    assert!(*jenv >= 50, "jenv should be near end of 76bp tRNA, got {}", jenv);
}

// ============================================================
// Test 2: tRNA embedded in 1000bp — envelope should NOT be 1-1000
// ============================================================

#[test]
fn domain_envelope_embedded_trna_is_narrow() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    // Read first sequence from 1k-tRNA.fa (tRNA at ~860-930 in 1000bp)
    let fa_path = testsuite.join("1k-tRNA.fa");
    if !fa_path.exists() { return; }

    let content = std::fs::read_to_string(&fa_path).unwrap();
    let mut seq = Vec::new();
    for line in content.lines().skip(1) {
        if line.starts_with('>') { break; }
        seq.extend_from_slice(line.trim().as_bytes());
    }
    assert!(seq.len() >= 900, "Sequence should be ~1000bp, got {}", seq.len());

    let domains = run_p7_pipeline(&hmm, &seq);

    println!("Embedded tRNA ({}bp) domains:", seq.len());
    for (i, j, sc) in &domains {
        println!("  ienv={} jenv={} score={:.1}", i, j, sc);
    }

    assert!(!domains.is_empty(),
        "P7 should find tRNA embedded in 1000bp sequence");

    // The critical test: envelope should be narrow around the hit,
    // NOT spanning the entire sequence.
    // The tRNA is at approximately positions 860-930.
    // A correct P7 domain definition should return an envelope
    // of roughly 100-200bp centered on the hit, not 1-1000.
    let (ienv, jenv, _score) = &domains[0];
    let envelope_len = jenv - ienv + 1;

    println!("Envelope: {}-{} ({}bp) for {}bp sequence",
        ienv, jenv, envelope_len, seq.len());

    // FAILING EXPECTATION: this is the bug we're tracking.
    // If envelope covers >50% of the sequence, the domain definition
    // is not narrowing properly.
    assert!(envelope_len < seq.len() / 2,
        "BUG: P7 domain envelope is too wide: {}-{} ({}bp) covers >50% of {}bp sequence. \
         Expected a narrow envelope (~100-200bp) around the tRNA hit at ~860-930.",
        ienv, jenv, envelope_len, seq.len());
}

// ============================================================
// Test 3: Random sequence — should have no domains
// ============================================================

#[test]
fn domain_envelope_random_sequence_no_hits() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    // 500bp of random nucleotides (no tRNA)
    let seq = b"AACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCGATCG\
                TTGCAACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCG\
                AACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCGATCG\
                TTGCAACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCG\
                AACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCGATCG\
                TTGCAACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCG\
                AACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCGATCG\
                TTGCAACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCG\
                AACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCGATCG\
                TTGCAACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCG";

    let domains = run_p7_pipeline(&hmm, seq);

    println!("Random sequence ({}bp) domains: {}", seq.len(), domains.len());
    // Random sequence may or may not pass — but if it does,
    // the score should be low
    for (i, j, sc) in &domains {
        println!("  ienv={} jenv={} score={:.1}", i, j, sc);
        assert!(*sc < 20.0,
            "Random sequence should not have high-scoring P7 hit: {:.1}", sc);
    }
}

// ============================================================
// Test 4: Multiple tRNAs in long sequence — should find multiple envelopes
// ============================================================

#[test]
fn domain_envelope_multiple_hits_separate_envelopes() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    // Construct: 200bp random + tRNA + 200bp random + tRNA + 200bp random
    let trna = b"GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
    let random = b"AACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCGATCG\
                   AACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCGATCG\
                   AACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCGATCG\
                   AACGTTGCAATCGGTACGATCGATCGATCGATCGATCGATCGATCGATCG";

    let mut seq = Vec::new();
    seq.extend_from_slice(random);
    let trna1_start = seq.len() + 1; // 1-based
    seq.extend_from_slice(trna);
    let trna1_end = seq.len();
    seq.extend_from_slice(random);
    let trna2_start = seq.len() + 1;
    seq.extend_from_slice(trna);
    let trna2_end = seq.len();
    seq.extend_from_slice(random);

    println!("Constructed {}bp sequence with tRNAs at {}-{} and {}-{}",
        seq.len(), trna1_start, trna1_end, trna2_start, trna2_end);

    let domains = run_p7_pipeline(&hmm, &seq);

    println!("Multi-tRNA domains:");
    for (i, j, sc) in &domains {
        println!("  ienv={} jenv={} score={:.1}", i, j, sc);
    }

    // Should find at least 2 domains
    assert!(domains.len() >= 2,
        "Should find at least 2 tRNA domains, got {}", domains.len());

    // Each envelope should be narrow, not spanning the whole sequence
    for (ienv, jenv, _sc) in &domains {
        let env_len = jenv - ienv + 1;
        assert!(env_len < seq.len() / 3,
            "BUG: Domain envelope {}-{} ({}bp) is too wide for {}bp sequence",
            ienv, jenv, env_len, seq.len());
    }
}

// ============================================================
// Test 5: Domain bit scores should be positive for true hits
// ============================================================

#[test]
fn domain_scores_positive_for_true_trna() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    // Pure tRNA sequence — C HMMER gives ~30 bits for this
    let seq = b"GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
    let results = run_p7_pipeline_full(&hmm, seq);

    println!("Score test — short tRNA (76bp):");
    for r in &results {
        println!("  ienv={} jenv={} dom_score={:.1} hit_score={:.1}",
            r.ienv, r.jenv, r.dom_bitscore, r.hit_score);
    }

    assert!(!results.is_empty(), "Should find tRNA hit");

    let r = &results[0];
    // Domain bitscore should be positive for a true tRNA hit.
    // C HMMER reports ~30 bits for this sequence against the tRNA P7 HMM.
    assert!(r.dom_bitscore > 0.0,
        "BUG: Domain bitscore should be positive for true tRNA hit, got {:.1}. \
         C HMMER gives ~30 bits.",
        r.dom_bitscore);

    // Hit-level score should also be positive
    assert!(r.hit_score > 0.0,
        "BUG: Hit score should be positive for true tRNA hit, got {:.1}",
        r.hit_score);
}

// ============================================================
// Test 6: Domain scores for embedded tRNA should be positive
// ============================================================

#[test]
fn domain_scores_positive_for_embedded_trna() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    // Read first sequence from 1k-tRNA.fa
    let fa_path = testsuite.join("1k-tRNA.fa");
    if !fa_path.exists() { return; }

    let content = std::fs::read_to_string(&fa_path).unwrap();
    let mut seq = Vec::new();
    for line in content.lines().skip(1) {
        if line.starts_with('>') { break; }
        seq.extend_from_slice(line.trim().as_bytes());
    }

    let results = run_p7_pipeline_full(&hmm, &seq);

    println!("Score test — embedded tRNA ({}bp):", seq.len());
    for r in &results {
        println!("  ienv={} jenv={} dom_score={:.1} hit_score={:.1}",
            r.ienv, r.jenv, r.dom_bitscore, r.hit_score);
    }

    // Find the domain near the tRNA position (~860-930)
    let trna_domain = results.iter().find(|r| r.ienv >= 800 && r.ienv <= 900);

    assert!(trna_domain.is_some(),
        "Should find a domain near position 860-930");

    let td = trna_domain.unwrap();
    assert!(td.dom_bitscore > 0.0,
        "BUG: Domain bitscore for true tRNA should be positive, got {:.1}. \
         C HMMER gives ~30 bits for the tRNA P7 HMM.",
        td.dom_bitscore);
}

// ============================================================
// Test 7: Hit score should match C HMMER within tolerance
// ============================================================

#[test]
fn hit_score_matches_c_hmmer_approximately() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    let seq = b"GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
    let results = run_p7_pipeline_full(&hmm, seq);

    assert!(!results.is_empty(), "Should find tRNA hit");

    let r = &results[0];
    println!("Hit score: {:.1}, Domain score: {:.1}", r.hit_score, r.dom_bitscore);

    // C HMMER hmmsearch with tRNA P7 HMM against this sequence gives
    // a hit score of approximately 30 bits (±5 for different configs).
    // Allow wide tolerance since exact value depends on profile config.
    assert!(r.hit_score > 10.0 && r.hit_score < 60.0,
        "BUG: Hit score {:.1} is outside expected range 10-60 bits. \
         C HMMER gives ~30 bits.",
        r.hit_score);

    assert!(r.dom_bitscore > 10.0 && r.dom_bitscore < 60.0,
        "BUG: Domain bitscore {:.1} is outside expected range 10-60 bits. \
         C HMMER gives ~30 bits.",
        r.dom_bitscore);
}

// ============================================================
// Test 8: No false-positive domains in random flanking regions
// ============================================================

#[test]
fn no_false_positive_domains_in_random_flanks() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    // Read first sequence from 1k-tRNA.fa (tRNA at ~860-930 in 1000bp)
    let fa_path = testsuite.join("1k-tRNA.fa");
    if !fa_path.exists() { return; }

    let content = std::fs::read_to_string(&fa_path).unwrap();
    let mut seq = Vec::new();
    for line in content.lines().skip(1) {
        if line.starts_with('>') { break; }
        seq.extend_from_slice(line.trim().as_bytes());
    }

    let results = run_p7_pipeline_full(&hmm, &seq);

    println!("False-positive test — embedded tRNA ({}bp):", seq.len());
    for r in &results {
        println!("  ienv={} jenv={} dom_score={:.1} hit_score={:.1}",
            r.ienv, r.jenv, r.dom_bitscore, r.hit_score);
    }

    // Only domains overlapping the true tRNA region (~860-930) should have
    // significant scores. Domains in random flanking regions should NOT be
    // reported, or if reported, should have very low scores.
    //
    // C HMMER does not report false-positive domains in random flanks.
    // Any domain outside the tRNA region with score > 10 bits is a
    // false positive that would cause unnecessary (expensive) CM scoring
    // in the Infernal pipeline.
    let false_positives: Vec<_> = results.iter()
        .filter(|r| r.ienv < 800 || r.ienv > 950) // outside tRNA region
        .filter(|r| r.dom_bitscore > 10.0)          // significant score
        .collect();

    assert!(false_positives.is_empty(),
        "BUG: {} false-positive domain(s) with score > 10 bits outside tRNA region. \
         These cause unnecessary CYK scoring in the Infernal pipeline, making \
         genome-scale search extremely slow. Domains: {:?}",
        false_positives.len(),
        false_positives.iter()
            .map(|r| format!("{}-{}: {:.1} bits", r.ienv, r.jenv, r.dom_bitscore))
            .collect::<Vec<_>>());
}

// ============================================================
// Test 9: Domain count should match C HMMER — no spurious domains
// ============================================================

#[test]
fn embedded_trna_domain_count_matches_c_hmmer() {
    let testsuite = infernal_testsuite();
    let hmm = read_hmm_from_cm(&testsuite.join("tRNA.c.cm"));

    // Read first sequence from 1k-tRNA.fa
    let fa_path = testsuite.join("1k-tRNA.fa");
    if !fa_path.exists() { return; }

    let content = std::fs::read_to_string(&fa_path).unwrap();
    let mut seq = Vec::new();
    for line in content.lines().skip(1) {
        if line.starts_with('>') { break; }
        seq.extend_from_slice(line.trim().as_bytes());
    }

    let results = run_p7_pipeline_full(&hmm, &seq);

    println!("Domain count test ({}bp seq with 1 tRNA):", seq.len());
    for r in &results {
        println!("  ienv={} jenv={} dom_score={:.1}", r.ienv, r.jenv, r.dom_bitscore);
    }

    // C HMMER finds exactly 1 domain for this sequence (the tRNA at ~860-930).
    // Reporting extra spurious domains wastes time in downstream CM scoring.
    let significant: Vec<_> = results.iter()
        .filter(|r| r.dom_bitscore > 10.0)
        .collect();

    assert_eq!(significant.len(), 1,
        "BUG: Expected 1 significant domain (the tRNA), got {}. \
         C HMMER reports exactly 1 domain for this sequence. \
         Extra domains: {:?}",
        significant.len(),
        significant.iter()
            .map(|r| format!("{}-{}: {:.1} bits", r.ienv, r.jenv, r.dom_bitscore))
            .collect::<Vec<_>>());

    // The one significant domain should be near 860-930
    let d = &significant[0];
    assert!(d.ienv >= 830 && d.ienv <= 890,
        "Domain start {} should be near 860", d.ienv);
    assert!(d.jenv >= 900 && d.jenv <= 960,
        "Domain end {} should be near 930", d.jenv);
}
