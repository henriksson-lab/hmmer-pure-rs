//! alimask — add/modify mask annotation on a Stockholm alignment.

use std::io::{BufReader, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use hmmer_pure_rs::msa;

#[derive(Parser)]
#[command(
    name = "alimask",
    about = "Add mask annotation to a Stockholm alignment"
)]
struct Args {
    /// Input alignment file (Stockholm)
    msafile: PathBuf,

    /// Output alignment file
    outfile: Option<PathBuf>,

    /// Assert input alignment format
    #[arg(long = "informat")]
    informat: Option<String>,

    /// Output alignment format
    #[arg(long = "outformat", default_value = "Stockholm")]
    outformat: String,

    /// Mask alignment-coordinate range(s), for example 3..8,12..15
    #[arg(long = "alirange")]
    alirange: Option<String>,

    /// Mask model-coordinate range(s)
    #[arg(long = "modelrange")]
    modelrange: Option<String>,

    /// Map model-coordinate range(s) to alignment coordinates
    #[arg(long = "model2ali")]
    model2ali: Option<String>,

    /// Map alignment-coordinate range(s) to model coordinates
    #[arg(long = "ali2model")]
    ali2model: Option<String>,

    /// Add to existing mask annotation instead of replacing it
    #[arg(long = "appendmask", action = ArgAction::SetTrue)]
    appendmask: bool,

    /// Model mask to apply (string of 'm' and '.' characters)
    #[arg(long = "modelmask")]
    modelmask: Option<String>,
}

/// Entry point for the `alimask` subcommand: append a modelmask line to a Stockholm MSA.
///
/// Parses the command line, reads the input alignment, and writes a Stockholm output
/// optionally augmented with a `#=GC MM` model mask line. Corresponds to `main()` in
/// hmmer/src/alimask.c, with the Rust port focused on the simple mask-annotation flow.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);
    let mode_count = [
        args.alirange.is_some(),
        args.modelrange.is_some(),
        args.model2ali.is_some(),
        args.ali2model.is_some(),
    ]
    .into_iter()
    .filter(|enabled| *enabled)
    .count();
    if mode_count == 0 {
        eprintln!("Must specify one of --modelrange, --alirange, --model2ali, or --ali2model");
        std::process::exit(1);
    }
    if mode_count > 1 {
        eprintln!(
            "Options --modelrange, --alirange, --model2ali, and --ali2model are mutually exclusive"
        );
        std::process::exit(1);
    }
    if args.modelrange.is_some() {
        eprintln!("alimask --modelrange is not implemented");
        std::process::exit(1);
    }
    if args.model2ali.is_some() {
        eprintln!("alimask --model2ali is not implemented");
        std::process::exit(1);
    }
    if args.ali2model.is_some() {
        eprintln!("alimask --ali2model is not implemented");
        std::process::exit(1);
    }
    let Some(ref outfile) = args.outfile else {
        eprintln!("alimask --alirange requires <postmsafile>");
        std::process::exit(1);
    };
    if args.msafile == PathBuf::from("-") && args.informat.is_none() {
        println!("Must specify --informat to read <alifile> from stdin ('-')");
        std::process::exit(1);
    }
    if let Some(ref informat) = args.informat {
        if !informat.eq_ignore_ascii_case("stockholm") && !informat.eq_ignore_ascii_case("pfam") {
            eprintln!("{informat} is not a recognized input sequence file format");
            std::process::exit(1);
        }
    }
    if !args.outformat.eq_ignore_ascii_case("stockholm")
        && !args.outformat.eq_ignore_ascii_case("pfam")
    {
        eprintln!(
            "{} is not a recognized output MSA file format",
            args.outformat
        );
        std::process::exit(1);
    }

    let msas = read_stockholm_maybe_stdin(&args.msafile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    let ali_ranges = args.alirange.as_deref().map(parse_ranges).transpose();
    let ali_ranges = ali_ranges.unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let mut out = std::fs::File::create(outfile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    for alignment in &msas {
        let name_width = alignment
            .sqname
            .iter()
            .map(|name| name.len())
            .max()
            .unwrap_or(0);
        if let Some(ref mask) = args.modelmask {
            if mask.len() != alignment.alen {
                eprintln!(
                    "Model mask length {} does not match alignment length {}",
                    mask.len(),
                    alignment.alen
                );
                std::process::exit(1);
            }
            if !mask.bytes().all(|b| b == b'm' || b == b'.') {
                eprintln!("Model mask may only contain 'm' and '.' characters");
                std::process::exit(1);
            }
        }

        let mut range_mask = ali_ranges.as_ref().map(|ranges| {
            let mut mask = vec![b'.'; alignment.alen];
            for &(start, end) in ranges {
                if start == 0 {
                    eprintln!("Mask ranges can not start before position 1; start 0 is invalid");
                    std::process::exit(1);
                }
                if end > alignment.alen {
                    eprintln!(
                        "Maximum mask range {} exceeds alignment length {}",
                        end, alignment.alen
                    );
                    std::process::exit(1);
                }
                for pos in start..=end {
                    mask[pos - 1] = b'm';
                }
            }
            mask
        });

        if args.appendmask {
            if let (Some(range_mask), Some(modelmask)) = (&mut range_mask, &args.modelmask) {
                for (dst, src) in range_mask.iter_mut().zip(modelmask.bytes()) {
                    if src == b'm' {
                        *dst = b'm';
                    }
                }
            }
        } else if range_mask.is_none() {
            range_mask = args.modelmask.as_ref().map(|mask| mask.as_bytes().to_vec());
        }

        writeln!(out, "# STOCKHOLM 1.0").unwrap();
        if !alignment.name.is_empty() {
            writeln!(out, "#=GF ID {}", alignment.name).unwrap();
        }
        writeln!(out).unwrap();

        // Write sequences
        for (i, name) in alignment.sqname.iter().enumerate() {
            let seq: String = alignment.aseq[i].iter().map(|&b| b as char).collect();
            writeln!(out, "{:<width$} {}", name, seq, width = name_width + 3).unwrap();
        }

        // Write RF if present
        if let Some(ref rf) = alignment.rf {
            let rf_str: String = rf.iter().map(|&b| b as char).collect();
            writeln!(out, "#=GC RF {}", rf_str).unwrap();
        }

        if let Some(mask) = range_mask {
            let mask = String::from_utf8(mask).unwrap();
            writeln!(out, "#=GC MM {}", mask).unwrap();
        }

        writeln!(out, "//").unwrap();
    }

    eprintln!("Wrote {} alignment(s) to {}", msas.len(), outfile.display());
    std::process::ExitCode::SUCCESS
}

fn parse_ranges(spec: &str) -> Result<Vec<(usize, usize)>, String> {
    let mut ranges = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        let Some((start, end)) = part.split_once("..") else {
            return Err(format!(
                "Range flags take coords <from>..<to>; {part} not recognized"
            ));
        };
        let start: usize = start
            .parse()
            .map_err(|_| format!("Failed to find <from> or <to> coord in {part}"))?;
        let end: usize = end
            .parse()
            .map_err(|_| format!("Failed to find <from> or <to> coord in {part}"))?;
        if start > end {
            return Err(format!(
                "In range ({part}) <from> can not be larger than <to>"
            ));
        }
        ranges.push((start, end));
    }
    Ok(ranges)
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
