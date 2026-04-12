//! hmmconvert — convert HMM files between formats.

use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::hmmfile;

#[derive(Parser)]
#[command(name = "hmmconvert", about = "Convert profile HMM file to HMMER3 format")]
struct Args {
    /// HMM file
    hmmfile: PathBuf,
}

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for hmm in &hmms {
        hmmfile::write_hmm(&mut out, hmm).unwrap_or_else(|e| {
            eprintln!("Error writing HMM: {}", e);
            std::process::exit(1);
        });
    }
    std::process::ExitCode::SUCCESS
}
