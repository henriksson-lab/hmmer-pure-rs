//! hmmfetch — retrieve an HMM from an HMM file by name.

use std::path::PathBuf;

use clap::Parser;

use hmmer::hmmfile;

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

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    if let Some(idx) = args.index {
        if idx == 0 || idx > hmms.len() {
            eprintln!("Error: index {} out of range (1-{})", idx, hmms.len());
            std::process::exit(1);
        }
        hmmfile::write_hmm(&mut out, &hmms[idx - 1]).unwrap();
    } else if let Some(ref key) = args.key {
        let found = hmms.iter().find(|h| h.name == *key || h.acc.as_deref() == Some(key));
        match found {
            Some(hmm) => hmmfile::write_hmm(&mut out, hmm).unwrap(),
            None => {
                eprintln!("Error: HMM '{}' not found in {}", key, args.hmmfile.display());
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Usage: hmmfetch <hmmfile> <key>");
        std::process::exit(1);
    }
}
