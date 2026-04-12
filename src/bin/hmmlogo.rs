//! hmmlogo — generate data for HMM sequence logo visualization.
//! Outputs per-position information content and emission probabilities.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmmfile;

#[derive(Parser)]
#[command(name = "hmmlogo", about = "Generate HMM logo data for visualization")]
struct Args {
    /// HMM file
    hmmfile: PathBuf,
}

fn main() {
    let args = Args::parse();

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for hmm in &hmms {
        let abc = Alphabet::new(hmm.abc_type);
        let bg = Bg::new(&abc);
        let k = abc.k;

        writeln!(out, "# Logo data for: {}", hmm.name).unwrap();
        writeln!(out, "# M={} K={}", hmm.m, k).unwrap();

        // Header: position and alphabet symbols
        write!(out, "pos\tIC").unwrap();
        for x in 0..k {
            write!(out, "\t{}", abc.sym[x] as char).unwrap();
        }
        writeln!(out).unwrap();

        // For each match position, output information content and letter heights
        for node in 1..=hmm.m {
            // Information content (bits)
            let mut ic = 0.0_f32;
            for x in 0..k {
                let p = hmm.mat[node][x];
                if p > 0.0 && bg.f[x] > 0.0 {
                    ic += p * (p / bg.f[x]).log2();
                }
            }

            write!(out, "{}\t{:.3}", node, ic).unwrap();

            // Letter heights (probability * IC for weighted logo)
            for x in 0..k {
                write!(out, "\t{:.4}", hmm.mat[node][x]).unwrap();
            }
            writeln!(out).unwrap();
        }
        writeln!(out, "//").unwrap();
    }
}
