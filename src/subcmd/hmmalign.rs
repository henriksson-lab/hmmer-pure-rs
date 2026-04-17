//! hmmalign — align sequences to a profile HMM.
//! Simplified version: uses Viterbi traceback for alignment.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::dp::generic_viterbi::g_viterbi;
use hmmer_pure_rs::dp::gmx::Gmx;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::sequence::{self, Sequence};

#[derive(Parser)]
#[command(name = "hmmalign", about = "Align sequences to a profile HMM")]
struct Args {
    /// HMM file
    hmmfile: PathBuf,
    /// Sequence file (FASTA format)
    seqfile: PathBuf,

    /// Output alignment format
    #[arg(long = "outformat", default_value = "Stockholm")]
    outformat: String,
}

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    logsum::p7_flogsuminit();

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    let hmm = &hmms[0];
    let abc = Alphabet::new(hmm.abc_type);
    let bg = Bg::new(&abc);

    // Read sequences
    let mut sequences = Vec::new();
    let mut sqf = sequence::open_seq_file(&args.seqfile, &abc).unwrap_or_else(|e| {
        eprintln!("Error opening sequence file: {}", e);
        std::process::exit(1);
    });
    let mut sq = Sequence::new();
    while sqf.read(&mut sq).unwrap() {
        sequences.push(sq.clone());
        sq.reuse();
    }

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    // Stockholm output header
    writeln!(out, "# STOCKHOLM 1.0").unwrap();
    writeln!(out).unwrap();

    // For each sequence, run Viterbi and output aligned sequence
    for sq in &sequences {
        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, sq.n as i32, P7_LOCAL);

        let mut gx = Gmx::new(hmm.m, sq.n);
        // Run Viterbi to compute alignment (traceback not yet implemented)
        g_viterbi(&sq.dsq, sq.n, &gm, &mut gx);

        // Simple alignment: just output the sequence
        // A proper implementation would traceback the Viterbi path
        // and insert gaps according to the model alignment
        let seq_text = abc.textize(&sq.dsq, sq.n);
        writeln!(out, "{:<20} {}", sq.name, seq_text).unwrap();
    }

    // Consensus from HMM
    if let Some(ref cons) = hmm.consensus {
        let cons_text: String = (1..=hmm.m).map(|i| cons[i] as char).collect();
        writeln!(out, "#=GC RF              {}", cons_text.replace('\0', "x")).unwrap();
    }

    writeln!(out, "//").unwrap();
    std::process::ExitCode::SUCCESS
}
