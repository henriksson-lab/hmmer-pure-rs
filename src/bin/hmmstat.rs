//! hmmstat — display summary statistics for each HMM in a file.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::hmm;

#[derive(Parser)]
#[command(name = "hmmstat", about = "Display summary statistics for each HMM")]
struct Args {
    /// HMM file
    hmmfile: PathBuf,
}

fn main() {
    let args = Args::parse();

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "# hmmstat :: display summary statistics for each HMM").unwrap();
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(out, "# Copyright (C) 2023 Howard Hughes Medical Institute.").unwrap();
    writeln!(out, "# Freely distributed under the BSD open source license.").unwrap();
    writeln!(out, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -").unwrap();

    writeln!(
        out,
        "# {:>3} {:<20} {:<12} {:>8} {:>8} {:>5} {:>8} {:>8} {:>8} {:>8}",
        "idx", "name", "accession", "nseq", "eff_nseq", "M", "relent", "info", "p relE", "compKL"
    ).unwrap();
    writeln!(
        out,
        "# {:>3} {:<20} {:<12} {:>8} {:>8} {:>5} {:>8} {:>8} {:>8} {:>8}",
        "---", "--------------------", "------------", "--------", "--------", "-----", "--------", "--------", "--------", "--------"
    ).unwrap();

    for (idx, h) in hmms.iter().enumerate() {
        let abc = Alphabet::new(h.abc_type);
        let bg = Bg::new(&abc);

        // Compute relative entropy and information content
        let relent = mean_match_relative_entropy(h, &bg);
        let info = mean_match_info(h, &bg);
        let p_rele = mean_position_relative_entropy(h, &bg);
        let comp_kl = composition_kld(h, &bg);

        writeln!(
            out,
            "  {:>3} {:<20} {:<12} {:>8} {:>8.2} {:>5} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            idx + 1,
            h.name,
            h.acc.as_deref().unwrap_or("-"),
            h.nseq,
            h.eff_nseq,
            h.m,
            relent,
            info,
            p_rele,
            comp_kl,
        ).unwrap();
    }

    writeln!(out, "#").unwrap();
    writeln!(out, "# [ok]").unwrap();
}

/// Mean relative entropy per match emission.
fn mean_match_relative_entropy(h: &hmmer_pure_rs::Hmm, bg: &Bg) -> f32 {
    let k = h.abc_k;
    let mut sum = 0.0_f32;
    for node in 1..=h.m {
        for x in 0..k {
            let p = h.mat[node][x];
            if p > 0.0 && bg.f[x] > 0.0 {
                sum += p * (p / bg.f[x]).log2();
            }
        }
    }
    sum / h.m as f32
}

/// Mean information content per match emission.
fn mean_match_info(h: &hmmer_pure_rs::Hmm, bg: &Bg) -> f32 {
    let k = h.abc_k;
    let mut sum = 0.0_f32;
    for node in 1..=h.m {
        let mut node_entropy = 0.0_f32;
        for x in 0..k {
            let p = h.mat[node][x];
            if p > 0.0 {
                node_entropy -= p * p.log2();
            }
        }
        let mut bg_entropy = 0.0_f32;
        for x in 0..k {
            if bg.f[x] > 0.0 {
                bg_entropy -= bg.f[x] * bg.f[x].log2();
            }
        }
        sum += bg_entropy - node_entropy;
    }
    sum / h.m as f32
}

/// Mean position-wise relative entropy.
fn mean_position_relative_entropy(h: &hmmer_pure_rs::Hmm, bg: &Bg) -> f32 {
    mean_match_relative_entropy(h, bg) // simplified: same as relent for now
}

/// KL divergence between model composition and background.
fn composition_kld(h: &hmmer_pure_rs::Hmm, bg: &Bg) -> f32 {
    let k = h.abc_k;
    if h.flags & hmm::P7H_COMPO == 0 {
        return 0.0;
    }
    let mut kl = 0.0_f32;
    for x in 0..k {
        let p = h.compo[x];
        if p > 0.0 && bg.f[x] > 0.0 {
            kl += p * (p / bg.f[x]).log2();
        }
    }
    kl
}
