//! hmmemit — sample or emit sequences from an HMM.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::hmmfile;

#[derive(Parser)]
#[command(name = "hmmemit", about = "Sample or emit sequences from a profile HMM")]
struct Args {
    /// HMM file
    hmmfile: PathBuf,

    /// Emit consensus sequence
    #[arg(short = 'c', long)]
    consensus: bool,

    /// Number of sequences to emit
    #[arg(short = 'N', default_value = "1")]
    n: usize,

    /// Random number seed
    #[arg(long = "seed", default_value = "0")]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for h in &hmms {
        let abc = Alphabet::new(h.abc_type);

        if args.consensus {
            // Emit consensus sequence: pick highest-probability residue at each position
            let mut seq = Vec::with_capacity(h.m);
            for node in 1..=h.m {
                let mut best_x = 0;
                let mut best_p = 0.0_f32;
                for x in 0..abc.k {
                    if h.mat[node][x] > best_p {
                        best_p = h.mat[node][x];
                        best_x = x;
                    }
                }
                seq.push(abc.sym[best_x]);
            }
            writeln!(out, ">{}-consensus", h.name).unwrap();
            let seq_str: String = seq.iter().map(|&b| b as char).collect();
            for chunk in seq_str.as_bytes().chunks(60) {
                writeln!(out, "{}", std::str::from_utf8(chunk).unwrap()).unwrap();
            }
        } else {
            // Sample sequences from the model (simplified: emit from match states)
            for i in 0..args.n {
                let mut rng = simple_rng(args.seed.wrapping_add(i as u64));
                let mut seq = Vec::with_capacity(h.m);
                for node in 1..=h.m {
                    // Sample from match emission distribution
                    let x = sample_discrete(&h.mat[node][..abc.k], &mut rng);
                    seq.push(abc.sym[x]);
                }
                writeln!(out, ">{}-sample{}", h.name, i + 1).unwrap();
                let seq_str: String = seq.iter().map(|&b| b as char).collect();
                for chunk in seq_str.as_bytes().chunks(60) {
                    writeln!(out, "{}", std::str::from_utf8(chunk).unwrap()).unwrap();
                }
            }
        }
    }
}

/// Simple LCG random number generator.
fn simple_rng(seed: u64) -> u64 {
    if seed == 0 { 42 } else { seed }
}

/// Sample from a discrete probability distribution.
fn sample_discrete(probs: &[f32], rng: &mut u64) -> usize {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r = (*rng >> 33) as f32 / (u32::MAX as f32 / 2.0);
    let r = r.fract().abs(); // 0..1

    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}
