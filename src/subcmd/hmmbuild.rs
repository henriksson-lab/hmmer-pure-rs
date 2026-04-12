//! hmmbuild — build profile HMM(s) from multiple sequence alignment(s).

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::builder;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::msa;

#[derive(Parser)]
#[command(name = "hmmbuild", about = "Build profile HMM(s) from multiple sequence alignment(s)")]
struct Args {
    /// Output HMM file
    hmmfile: PathBuf,
    /// Input alignment file (Stockholm format)
    msafile: PathBuf,

    /// Name the HMM
    #[arg(short = 'n')]
    name: Option<String>,

    /// Use DNA alphabet
    #[arg(long)]
    dna: bool,

    /// Use RNA alphabet
    #[arg(long)]
    rna: bool,

    /// Sym fraction threshold for match/insert (default 0.5)
    #[arg(long = "symfrac", default_value = "0.5")]
    symfrac: f32,
}

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    let abc = if args.dna {
        Alphabet::dna()
    } else if args.rna {
        Alphabet::rna()
    } else {
        Alphabet::amino()
    };
    let bg = Bg::new(&abc);

    let msas = msa::read_stockholm(&args.msafile).unwrap_or_else(|e| {
        eprintln!("Error reading MSA file: {}", e);
        std::process::exit(1);
    });

    let stderr = std::io::stderr();
    let mut err = stderr.lock();

    // Build output file
    let mut out_file = std::fs::File::create(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error creating HMM file: {}", e);
        std::process::exit(1);
    });

    writeln!(err, "# hmmbuild :: profile HMM construction from multiple sequence alignments").unwrap();
    writeln!(err, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(err, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -").unwrap();
    writeln!(err, "# input alignment file:            {}", args.msafile.display()).unwrap();
    writeln!(err, "# output HMM file:                 {}", args.hmmfile.display()).unwrap();
    writeln!(err, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -").unwrap();
    writeln!(err).unwrap();

    for (idx, alignment) in msas.iter().enumerate() {
        let mut hmm = builder::build_hmm_from_msa(alignment, &abc, &bg, args.symfrac);

        if let Some(ref name) = args.name {
            hmm.name = name.clone();
        }

        writeln!(
            err,
            "{:>3} {:<20} {:>5} {:>5} {:>8.2}",
            idx + 1,
            hmm.name,
            hmm.m,
            alignment.nseq,
            hmm.eff_nseq,
        ).unwrap();

        hmmfile::write_hmm(&mut out_file, &hmm).unwrap_or_else(|e| {
            eprintln!("Error writing HMM: {}", e);
            std::process::exit(1);
        });
    }

    writeln!(err, "#").unwrap();
    writeln!(err, "# [ok]").unwrap();
    std::process::ExitCode::SUCCESS
}
