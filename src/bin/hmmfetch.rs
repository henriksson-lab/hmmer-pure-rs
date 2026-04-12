//! hmmfetch — retrieve an HMM from an HMM file by name.
//! Uses SSI-like index for fast lookup on large databases.

use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::ssi::Index;

#[derive(Parser)]
#[command(name = "hmmfetch", about = "Retrieve HMM(s) from an HMM file by name")]
struct Args {
    /// HMM file
    hmmfile: PathBuf,
    /// Name of HMM to retrieve (or --index N)
    key: Option<String>,

    /// Fetch by index number instead of name
    #[arg(long = "index")]
    index: Option<usize>,
}

fn main() {
    let args = Args::parse();

    if let Some(idx) = args.index {
        // Fetch by index: need to read all HMMs
        let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
            eprintln!("Error reading HMM file: {}", e);
            std::process::exit(1);
        });
        if idx == 0 || idx > hmms.len() {
            eprintln!("Error: index {} out of range (1-{})", idx, hmms.len());
            std::process::exit(1);
        }
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        hmmfile::write_hmm(&mut out, &hmms[idx - 1]).unwrap();
    } else if let Some(ref key) = args.key {
        // Build index for fast lookup
        let idx = Index::build_from_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
            eprintln!("Error building index: {}", e);
            std::process::exit(1);
        });

        if idx.lookup(key).is_some() {
            // Found in index — read the specific HMM
            let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
                eprintln!("Error reading HMM file: {}", e);
                std::process::exit(1);
            });
            let found = hmms.iter().find(|h| h.name == *key || h.acc.as_deref() == Some(key));
            match found {
                Some(hmm) => {
                    let stdout = std::io::stdout();
                    let mut out = stdout.lock();
                    hmmfile::write_hmm(&mut out, hmm).unwrap();
                }
                None => {
                    eprintln!("Error: HMM '{}' not found in {}", key, args.hmmfile.display());
                    std::process::exit(1);
                }
            }
        } else {
            eprintln!("Error: HMM '{}' not found in {}", key, args.hmmfile.display());
            std::process::exit(1);
        }
    } else {
        eprintln!("Usage: hmmfetch <hmmfile> <key>");
        std::process::exit(1);
    }
}
