//! hmmbuild — build profile HMM(s) from multiple sequence alignment(s).

use std::io::{BufReader, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use hmmer_pure_rs::alphabet::{Alphabet, AlphabetType};
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::builder;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::msa;

#[derive(Parser)]
#[command(
    name = "hmmbuild",
    about = "Build profile HMM(s) from multiple sequence alignment(s)"
)]
struct Args {
    /// Output HMM file
    hmmfile: PathBuf,
    /// Input alignment file (Stockholm format)
    msafile: PathBuf,

    /// Name the HMM
    #[arg(short = 'n')]
    name: Option<String>,

    /// Direct summary output to file, not stdout
    #[arg(short = 'o')]
    summary_out: Option<PathBuf>,

    /// Assert input alignment file format
    #[arg(long = "informat")]
    informat: Option<String>,

    /// Use DNA alphabet
    #[arg(long, action = ArgAction::SetTrue)]
    dna: bool,

    /// Use RNA alphabet
    #[arg(long, action = ArgAction::SetTrue)]
    rna: bool,

    /// Use protein alphabet
    #[arg(long, action = ArgAction::SetTrue)]
    amino: bool,

    /// Assign consensus columns from RF annotation
    #[arg(long, action = ArgAction::SetTrue)]
    hand: bool,

    /// Assign consensus columns by residue fraction
    #[arg(long, action = ArgAction::SetTrue)]
    fast: bool,

    /// Sym fraction threshold for match/insert (default 0.5)
    #[arg(long = "symfrac", default_value = "0.5")]
    symfrac: f32,
}

/// Entry point for `hmmbuild`: build profile HMM(s) from MSA(s) and write them
/// to a single output file.
///
/// Streams a Stockholm input, calls the builder pipeline per alignment, applies
/// optional name/alphabet overrides, and prints HMMER 3.4's per-MSA summary line
/// (`idx`, name, nodes, nseq, eff_nseq). Corresponds to `main()` /
/// `output_result()` in hmmer/src/hmmbuild.c (single-threaded path only).
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    if [args.amino, args.dna, args.rna]
        .into_iter()
        .filter(|v| *v)
        .count()
        > 1
    {
        eprintln!("Error: options --amino, --dna, and --rna are mutually exclusive");
        std::process::exit(1);
    }
    if args.hand && args.fast {
        eprintln!("Error: options --hand and --fast are mutually exclusive");
        std::process::exit(1);
    }
    if !(0.0..=1.0).contains(&args.symfrac) {
        eprintln!("Error: --symfrac must be between 0 and 1");
        std::process::exit(1);
    }
    if args.hmmfile == PathBuf::from("-") {
        eprintln!("Error: hmmbuild cannot write <hmmfile_out> to stdout; use a file path");
        std::process::exit(1);
    }
    if args.msafile == PathBuf::from("-") && args.informat.is_none() {
        println!("Must specify --informat to read <alifile> from stdin ('-')");
        std::process::exit(1);
    }
    if let Some(ref informat) = args.informat {
        if !informat.eq_ignore_ascii_case("stockholm") && !informat.eq_ignore_ascii_case("pfam") {
            eprintln!("{informat} is not a recognized input alignment file format");
            std::process::exit(1);
        }
    }

    let msas = read_stockholm_maybe_stdin(&args.msafile).unwrap_or_else(|e| {
        eprintln!("Error reading MSA file: {}", e);
        std::process::exit(1);
    });
    if args.name.is_some() && msas.len() > 1 {
        eprintln!("Error: You can't use -n with an alignment database");
        std::process::exit(1);
    }
    let Some(first_msa) = msas.first() else {
        eprintln!("Error: no alignments found in {}", args.msafile.display());
        std::process::exit(1);
    };
    let abc = if args.dna {
        Alphabet::dna()
    } else if args.rna {
        Alphabet::rna()
    } else if args.amino {
        Alphabet::amino()
    } else {
        Alphabet::new(guess_msa_alphabet(first_msa).unwrap_or_else(|e| {
            eprintln!("{e}; please specify --amino, --dna, or --rna");
            std::process::exit(1);
        }))
    };
    let bg = Bg::new(&abc);

    let mut summary_file = args.summary_out.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating summary output file: {}", e);
            std::process::exit(1);
        })
    });
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let summary: &mut dyn Write = match summary_file {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };

    // Build output file
    let mut out_file = std::fs::File::create(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error creating HMM file: {}", e);
        std::process::exit(1);
    });

    writeln!(
        summary,
        "# hmmbuild :: profile HMM construction from multiple sequence alignments"
    )
    .unwrap();
    writeln!(summary, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(
        summary,
        "# Copyright (C) 2023 Howard Hughes Medical Institute."
    )
    .unwrap();
    writeln!(
        summary,
        "# Freely distributed under the BSD open source license."
    )
    .unwrap();
    writeln!(
        summary,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(
        summary,
        "# input alignment file:             {}",
        args.msafile.display()
    )
    .unwrap();
    writeln!(
        summary,
        "# output HMM file:                  {}",
        args.hmmfile.display()
    )
    .unwrap();
    if let Some(ref path) = args.summary_out {
        writeln!(
            summary,
            "# output directed to file:          {}",
            path.display()
        )
        .unwrap();
    }
    if args.amino {
        writeln!(summary, "# input alignment is asserted as:  protein").unwrap();
    }
    if args.dna {
        writeln!(summary, "# input alignment is asserted as:  DNA").unwrap();
    }
    if args.rna {
        writeln!(summary, "# input alignment is asserted as:  RNA").unwrap();
    }
    if args.hand {
        writeln!(
            summary,
            "# model architecture construction:  hand-specified by RF annotation"
        )
        .unwrap();
    }
    if args.symfrac != 0.5 {
        writeln!(
            summary,
            "# sym fraction for model structure: {:.3}",
            args.symfrac
        )
        .unwrap();
    }
    writeln!(
        summary,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(summary).unwrap();
    if abc.abc_type == AlphabetType::Amino {
        writeln!(
            summary,
            "# idx name                  nseq  alen  mlen eff_nseq re/pos description"
        )
        .unwrap();
        writeln!(
            summary,
            "#---- -------------------- ----- ----- ----- -------- ------ -----------"
        )
        .unwrap();
    } else {
        writeln!(
            summary,
            "# idx name                  nseq  alen  mlen     W eff_nseq re/pos description"
        )
        .unwrap();
        writeln!(
            summary,
            "#---- -------------------- ----- ----- ----- ----- -------- ------ -----------"
        )
        .unwrap();
    }

    for (idx, alignment) in msas.iter().enumerate() {
        if args.hand && alignment.rf.is_none() {
            eprintln!("Model file does not contain an RF line, required for --hand.");
            std::process::exit(1);
        }
        let mut hmm = builder::build_hmm_from_msa(alignment, &abc, &bg, args.symfrac, args.hand);

        if let Some(ref name) = args.name {
            hmm.name = name.clone();
        }

        let rel_entropy = mean_match_relative_entropy(&hmm, &bg);
        let description = hmm.desc.as_deref().unwrap_or("");
        if abc.abc_type == AlphabetType::Amino {
            writeln!(
                summary,
                "{:<5} {:<20} {:>5} {:>5} {:>5} {:>8.2} {:>6.3} {}",
                idx + 1,
                hmm.name,
                alignment.nseq,
                alignment.alen,
                hmm.m,
                hmm.eff_nseq,
                rel_entropy,
                description,
            )
            .unwrap();
        } else {
            writeln!(
                summary,
                "{:<5} {:<20} {:>5} {:>5} {:>5} {:>5} {:>8.2} {:>6.3} {}",
                idx + 1,
                hmm.name,
                alignment.nseq,
                alignment.alen,
                hmm.m,
                hmm.max_length,
                hmm.eff_nseq,
                rel_entropy,
                description,
            )
            .unwrap();
        }

        hmmfile::write_hmm(&mut out_file, &hmm).unwrap_or_else(|e| {
            eprintln!("Error writing HMM: {}", e);
            std::process::exit(1);
        });
    }

    std::process::ExitCode::SUCCESS
}

fn read_stockholm_maybe_stdin(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<msa::Msa>> {
    if path == std::path::Path::new("-") {
        let stdin = std::io::stdin();
        msa::read_stockholm_from_reader(BufReader::new(stdin.lock()))
    } else {
        msa::read_stockholm(path)
    }
}

fn guess_msa_alphabet(msa: &msa::Msa) -> Result<AlphabetType, String> {
    let mut counts = [0usize; 26];
    for row in &msa.aseq {
        for &ch in row {
            if ch.is_ascii_alphabetic() {
                counts[(ch.to_ascii_uppercase() - b'A') as usize] += 1;
            }
        }
    }
    let n: usize = counts.iter().sum();
    if n <= 10 {
        return Err("could not determine alignment alphabet from <=10 residues".to_string());
    }

    let idx = |ch: u8| (ch - b'A') as usize;
    let amino_only = b"EFIJLOPQZ"
        .iter()
        .map(|&ch| counts[idx(ch)])
        .sum::<usize>();
    if amino_only > 0 {
        return Ok(AlphabetType::Amino);
    }

    let dna_core = b"ACGTN".iter().map(|&ch| counts[idx(ch)]).sum::<usize>();
    let rna_core = b"ACGUN".iter().map(|&ch| counts[idx(ch)]).sum::<usize>();
    let frac = |x: usize| x as f64 / n as f64;
    let t = counts[idx(b'T')];
    let u = counts[idx(b'U')];
    if frac(dna_core) >= 0.98 && u == 0 {
        return Ok(AlphabetType::Dna);
    }
    if frac(rna_core) >= 0.98 && t == 0 {
        return Ok(AlphabetType::Rna);
    }

    let distinct = counts.iter().filter(|&&c| c > 0).count();
    if frac(dna_core.max(rna_core)) < 0.98 && distinct >= 15 {
        return Ok(AlphabetType::Amino);
    }

    Err("could not determine alignment alphabet".to_string())
}

fn mean_match_relative_entropy(hmm: &hmmer_pure_rs::Hmm, bg: &Bg) -> f32 {
    let mut sum = 0.0_f32;
    for node in 1..=hmm.m {
        for x in 0..hmm.abc_k {
            let p = hmm.mat[node][x];
            if p > 0.0 && bg.f[x] > 0.0 {
                sum += p * (p / bg.f[x]).log2();
            }
        }
    }
    sum / hmm.m as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmmbuild_parses_summary_output_and_alphabet_assertions() {
        let args = Args::try_parse_from([
            "hmmbuild",
            "-o",
            "summary.txt",
            "--amino",
            "out.hmm",
            "in.sto",
        ])
        .unwrap();

        assert_eq!(args.summary_out, Some(PathBuf::from("summary.txt")));
        assert!(args.amino);
        assert_eq!(args.hmmfile, PathBuf::from("out.hmm"));
        assert_eq!(args.msafile, PathBuf::from("in.sto"));
    }
}
