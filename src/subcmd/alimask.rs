//! alimask — add/modify mask annotation on a Stockholm alignment.

use std::io::{BufReader, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use hmmer_pure_rs::alphabet::{Alphabet, AlphabetType};
use hmmer_pure_rs::builder::{self, RelativeWeighting};
use hmmer_pure_rs::msa;
use hmmer_pure_rs::output::fmt_fixed3;

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

    /// Direct summary output to file, not stdout
    #[arg(short = 'o')]
    summary_out: Option<PathBuf>,

    /// Assert input alignment format
    #[arg(long = "informat")]
    informat: Option<String>,

    /// Output alignment format
    #[arg(long = "outformat", default_value = "Stockholm")]
    outformat: String,

    /// Use protein alphabet
    #[arg(long, action = ArgAction::SetTrue)]
    amino: bool,

    /// Use DNA alphabet
    #[arg(long, action = ArgAction::SetTrue)]
    dna: bool,

    /// Use RNA alphabet
    #[arg(long, action = ArgAction::SetTrue)]
    rna: bool,

    /// Assign columns with >= symfrac residues as consensus
    #[arg(long = "fast", conflicts_with = "hand")]
    fast: bool,

    /// Use RF annotation as consensus columns
    #[arg(long = "hand", conflicts_with = "fast")]
    hand: bool,

    /// Consensus residue fraction for --fast construction
    #[arg(long = "symfrac", default_value = "0.5", value_parser = parse_fraction)]
    symfrac: f64,

    /// Sequence is called a fragment if L <= x*alignment_length
    #[arg(long = "fragthresh", default_value = "0.5", value_parser = parse_fraction)]
    fragthresh: f64,

    /// Random number seed
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Henikoff position-based weights for model-coordinate construction
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wgsc", "wblosum", "wnone", "wgiven"])]
    wpb: bool,

    /// Gerstein/Sonnhammer/Chothia tree weights for model-coordinate construction
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wblosum", "wnone", "wgiven"])]
    wgsc: bool,

    /// Henikoff simple filter weights for model-coordinate construction
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wnone", "wgiven"])]
    wblosum: bool,

    /// No relative sequence weighting for model-coordinate construction
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wblosum", "wgiven"])]
    wnone: bool,

    /// Use weights as given in the MSA file for model-coordinate construction
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wblosum", "wnone"])]
    wgiven: bool,

    /// For --wblosum: set identity cutoff
    #[arg(long = "wid", default_value = "0.62", value_parser = parse_fraction)]
    wid: f64,

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

fn parse_fraction(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid fraction: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("value must be between 0 and 1".to_string())
    }
}

/// Entry point for the `alimask` subcommand: append a modelmask line to a Stockholm MSA.
///
/// Parses the command line, reads the input alignment, and writes a Stockholm output
/// optionally augmented with a `#=GC MM` model mask line. Corresponds to `main()` in
/// hmmer/src/alimask.c, with the Rust port focused on the simple mask-annotation flow.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let wid_on_cmdline = args
        .iter()
        .any(|arg| arg == "--wid" || arg.starts_with("--wid="));
    let seed_on_cmdline = args
        .iter()
        .any(|arg| arg == "--seed" || arg.starts_with("--seed="));
    let args = Args::parse_from(&args);
    // C uses esl_opt_IsUsed("--seed"/"--wid"), which is FALSE when the supplied
    // value equals the option default (42 / 0.62 respectively). So `--seed 42`
    // and `--wid 0.62` print no header line, matching the no-flag case.
    let seed_was_requested = seed_on_cmdline && args.seed != 42;
    let wid_was_requested = wid_on_cmdline && args.wid != 0.62;
    if [args.amino, args.dna, args.rna]
        .into_iter()
        .filter(|v| *v)
        .count()
        > 1
    {
        eprintln!("Error: options --amino, --dna, and --rna are mutually exclusive");
        std::process::exit(1);
    }
    let mode_count = [
        args.alirange.is_some(),
        args.modelrange.is_some(),
        args.model2ali.is_some(),
        args.ali2model.is_some(),
    ]
    .into_iter()
    .filter(|enabled| *enabled)
    .count();
    if mode_count == 0 && args.modelmask.is_none() {
        eprintln!(
            "Must specify one of --modelrange, --alirange, --model2ali, --ali2model, or --modelmask"
        );
        std::process::exit(1);
    }
    if mode_count > 1 {
        eprintln!(
            "Options --modelrange, --alirange, --model2ali, and --ali2model are mutually exclusive"
        );
        std::process::exit(1);
    }
    if wid_was_requested && !args.wblosum {
        eprintln!("Error: --wid only works in combination with --wblosum");
        std::process::exit(1);
    }
    let mapping_mode = args.model2ali.is_some() || args.ali2model.is_some();
    let outfile = args.outfile.as_ref();
    if !mapping_mode && outfile.is_none() {
        eprintln!("alimask --alirange requires <postmsafile>");
        std::process::exit(1);
    }
    if args.msafile == std::path::Path::new("-") && args.informat.is_none() {
        println!("Must specify --informat to read <alifile> from stdin ('-')");
        std::process::exit(1);
    }
    if let Some(ref informat) = args.informat {
        if !is_stockholm_format(informat) {
            eprintln!("{informat} is not a recognized input sequence file format");
            std::process::exit(1);
        }
    }
    if !is_stockholm_format(&args.outformat) {
        eprintln!(
            "{} is not a recognized output MSA file format",
            args.outformat
        );
        std::process::exit(1);
    }

    let (msas, full_msas) = read_stockholm_maybe_stdin(&args.msafile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    let needs_model_map = args.modelrange.is_some() || mapping_mode;
    let inferred_abc = if needs_model_map && !args.hand {
        Some(if args.dna {
            Alphabet::dna()
        } else if args.rna {
            Alphabet::rna()
        } else if args.amino {
            Alphabet::amino()
        } else {
            let Some(first_msa) = msas.first() else {
                eprintln!("Error: no alignments found in {}", args.msafile.display());
                std::process::exit(1);
            };
            Alphabet::new(guess_msa_alphabet(&first_msa.msa).unwrap_or_else(|e| {
                eprintln!("{e}; please specify --amino, --dna, or --rna");
                std::process::exit(1);
            }))
        })
    } else {
        None
    };
    let weighting_strategy = if args.wnone {
        RelativeWeighting::None
    } else if args.wgsc {
        RelativeWeighting::Gsc
    } else if args.wblosum {
        RelativeWeighting::Blosum {
            identity_cutoff: args.wid,
        }
    } else if args.wgiven {
        RelativeWeighting::Given
    } else {
        RelativeWeighting::PositionBased
    };
    let coordinate_ranges = args
        .alirange
        .as_deref()
        .or(args.modelrange.as_deref())
        .or(args.model2ali.as_deref())
        .or(args.ali2model.as_deref())
        .map(parse_ranges)
        .transpose();
    let coordinate_ranges = coordinate_ranges.unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    if mapping_mode {
        write_summary_report(&args, seed_was_requested, wid_was_requested).unwrap_or_else(|e| {
            eprintln!("Error writing summary output: {}", e);
            std::process::exit(1);
        });
        for stockholm in &msas {
            let alignment = &stockholm.msa;
            let model_to_ali = make_model_to_alignment_map(
                alignment,
                inferred_abc.as_ref(),
                args.hand,
                args.symfrac,
                args.fragthresh,
                weighting_strategy,
            )
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });
            let ranges = coordinate_ranges.as_deref().unwrap_or(&[]);
            if args.model2ali.is_some() {
                write_model_to_alignment_map(ranges, &model_to_ali).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });
            } else {
                write_alignment_to_model_map(ranges, &model_to_ali, alignment.alen).unwrap_or_else(
                    |e| {
                        eprintln!("{e}");
                        std::process::exit(1);
                    },
                );
            }
        }
        return std::process::ExitCode::SUCCESS;
    }

    let outfile = outfile.expect("checked above");
    let mut out = std::fs::File::create(outfile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    write_summary_report(&args, seed_was_requested, wid_was_requested).unwrap_or_else(|e| {
        eprintln!("Error writing summary output: {}", e);
        std::process::exit(1);
    });

    for (stockholm, full) in msas.iter().zip(full_msas.iter()) {
        let alignment = &stockholm.msa;
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

        let ali_ranges = if args.modelrange.is_some() {
            let model_to_ali = make_model_to_alignment_map(
                alignment,
                inferred_abc.as_ref(),
                args.hand,
                args.symfrac,
                args.fragthresh,
                weighting_strategy,
            )
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });
            coordinate_ranges
                .as_ref()
                .map(|ranges| model_ranges_to_alignment_ranges(ranges, &model_to_ali))
                .transpose()
                .unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                })
        } else {
            coordinate_ranges.clone()
        };

        let mut range_mask = ali_ranges.as_ref().map(|ranges| {
            // C (alimask.c:368-374): keep_mm = IsUsed("--appendmask"). If the
            // input MSA has no #=GC MM line, msa->mm is freshly allocated and
            // keep_mm forced FALSE. When !keep_mm the mask is reset to all '.'.
            // So with --appendmask AND an existing MM line, start from the
            // existing mask and OR the new range columns on top; otherwise
            // start from a fresh all-'.' mask.
            let mut mask = match (args.appendmask, alignment.mm.as_ref()) {
                (true, Some(existing)) if existing.len() == alignment.alen => existing.clone(),
                _ => vec![b'.'; alignment.alen],
            };
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

        // C re-serializes the whole MSA via esl_msafile_Write(...
        // eslMSAFILE_STOCKHOLM) -> stockholm_write(fp, msa, cpl=200). C opens
        // the input through esl_msafile_Open(&abc, ...) with an explicit or
        // autodetected alphabet (digital mode), so stockholm_write textizes the
        // sequences: gap chars normalize to '-' and residues become canonical.
        // For model-coordinate modes, C first calls esl_msa_MarkFragments_old(),
        // which rewrites leading/trailing non-residues in short fragments to
        // missing data '~'. We apply the same mutation to the full-fidelity
        // text model before writing.
        let write_abc = inferred_abc.clone().unwrap_or_else(|| {
            if args.dna {
                Alphabet::dna()
            } else if args.rna {
                Alphabet::rna()
            } else if args.amino {
                Alphabet::amino()
            } else {
                Alphabet::new(guess_msa_alphabet(alignment).unwrap_or_else(|e| {
                    eprintln!("{e}; please specify --amino, --dna, or --rna");
                    std::process::exit(1);
                }))
            }
        });
        let mut full = full.clone();
        if needs_model_map {
            mark_fragments_old_full(&mut full, &write_abc, args.fragthresh);
        }
        if let Some(mask) = range_mask.as_deref() {
            full.mm = Some(mask.to_vec());
        }
        // C (alimask.c:380-396): in --modelrange mode the MSA is relative-weighted
        // before the map is built. --wpb/--wgsc/--wblosum raise eslMSA_HASWGTS, so
        // the output MSA carries #=GS ... WT lines; --wnone/--wgiven do not.
        if args.modelrange.is_some() {
            if let Some(weights) =
                compute_output_weights(alignment, &write_abc, args.hand, weighting_strategy)
            {
                full.has_wgts = true;
                full.wgt = weights;
            }
        }
        let cpl = if is_pfam_format(&args.outformat) {
            full.alen
        } else {
            200
        };
        msa::write_stockholm_full(&mut out, &full, cpl, Some(&write_abc)).unwrap();
    }

    std::process::ExitCode::SUCCESS
}

fn parse_ranges(spec: &str) -> Result<Vec<(usize, usize)>, String> {
    let mut ranges = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        let Some((start, end)) = part.split_once("..").or_else(|| part.split_once('-')) else {
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

fn write_summary_report(
    args: &Args,
    seed_was_requested: bool,
    wid_was_requested: bool,
) -> std::io::Result<()> {
    let mut file;
    let stdout;
    let out: &mut dyn Write = if let Some(ref path) = args.summary_out {
        file = std::fs::File::create(path)?;
        &mut file
    } else {
        stdout = std::io::stdout();
        let mut lock = stdout.lock();
        write_summary_report_to(&mut lock, args, seed_was_requested, wid_was_requested)?;
        lock.flush()?;
        return Ok(());
    };

    write_summary_report_to(out, args, seed_was_requested, wid_was_requested)
}

fn write_summary_report_to<W: Write + ?Sized>(
    out: &mut W,
    args: &Args,
    seed_was_requested: bool,
    wid_was_requested: bool,
) -> std::io::Result<()> {
    writeln!(
        out,
        "# alimask :: append modelmask line to a multiple sequence alignment"
    )?;
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/")?;
    writeln!(out, "# Copyright (C) 2023 Howard Hughes Medical Institute.")?;
    writeln!(
        out,
        "# Freely distributed under the BSD open source license."
    )?;
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )?;
    writeln!(
        out,
        "# input alignment file:             {}",
        args.msafile.display()
    )?;
    if (args.alirange.is_some() || args.modelrange.is_some()) && args.outfile.is_some() {
        writeln!(
            out,
            "# output alignment file:            {}",
            args.outfile.as_ref().unwrap().display()
        )?;
    }
    if let Some(ref range) = args.alirange {
        writeln!(out, "# alignment range:                  {range}")?;
    }
    if let Some(ref range) = args.modelrange {
        writeln!(out, "# model range:                      {range}")?;
    }
    if args.appendmask {
        writeln!(out, "# add to existing mask:             [on]")?;
    }
    if let Some(ref range) = args.model2ali {
        writeln!(out, "# ali ranges for model range:      {range}")?;
    }
    if let Some(ref range) = args.ali2model {
        writeln!(out, "# model ranges for ali range:      {range}")?;
    }
    if let Some(ref path) = args.summary_out {
        writeln!(
            out,
            "# output directed to file:          {}",
            path.display()
        )?;
    }
    if args.amino {
        writeln!(out, "# input alignment is asserted as:   protein")?;
    }
    if args.dna {
        writeln!(out, "# input alignment is asserted as:   DNA")?;
    }
    if args.rna {
        writeln!(out, "# input alignment is asserted as:   RNA")?;
    }
    if args.hand {
        writeln!(
            out,
            "# model architecture construction:  hand-specified by RF annotation"
        )?;
    }
    if args.symfrac != 0.5 {
        writeln!(
            out,
            "# sym fraction for model structure: {}",
            fmt_fixed3(args.symfrac)
        )?;
    }
    if args.fragthresh != 0.5 {
        writeln!(
            out,
            "# seq called frag if L <= x*alen:   {}",
            fmt_fixed3(args.fragthresh)
        )?;
    }
    // C (alimask.c:171-174): gated on IsUsed("--seed"), which is FALSE when the
    // supplied value equals the default (42). For seed 0 C prints BOTH lines:
    // the first `if (seed==0 && fprintf(...)<0)` actually evaluates the fprintf
    // (printing "one-time arbitrary"), the condition is false, so the `else if`
    // fprintf("set to: 0") also fires.
    if seed_was_requested {
        if args.seed == 0 {
            writeln!(
                out,
                "# random number seed:               one-time arbitrary"
            )?;
            writeln!(out, "# random number seed set to:        0")?;
        } else {
            writeln!(out, "# random number seed set to:        {}", args.seed)?;
        }
    }
    // C (alimask.c) gates the scheme line on esl_opt_IsUsed of a NON-default
    // weighting: --wpb is the default, so it never prints a scheme line (even
    // when given explicitly); only --wgsc/--wblosum/--wnone do.
    if args.wgsc {
        writeln!(out, "# relative weighting scheme:        G/S/C")?;
    }
    if args.wblosum {
        writeln!(out, "# relative weighting scheme:        BLOSUM filter")?;
    }
    if args.wnone {
        writeln!(out, "# relative weighting scheme:        none")?;
    }
    // C (alimask.c:170): the BLOSUM identity-cutoff line is gated on
    // IsUsed("--wid"), i.e. printed only when --wid is explicitly supplied,
    // NOT on --wblosum. (--wgiven has no header line in C at all, F6.)
    if wid_was_requested {
        writeln!(out, "# frac id cutoff for BLOSUM wgts:   {:.6}", args.wid)?;
    }
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )?;
    writeln!(out)?;
    Ok(())
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

fn mark_fragments_old_full(msa: &mut msa::FullStockholm, abc: &Alphabet, fragthresh: f64) {
    for row in &mut msa.aseq {
        let rlen = row
            .iter()
            .filter(|&&ch| {
                let code = abc.digitize_symbol(ch);
                code != hmmer_pure_rs::alphabet::DSQ_ILLEGAL
                    && code != hmmer_pure_rs::alphabet::DSQ_IGNORED
                    && abc.is_residue(code)
            })
            .count();
        if (rlen as f64) > fragthresh * (msa.alen as f64) {
            continue;
        }

        for ch in row.iter_mut() {
            let code = abc.digitize_symbol(*ch);
            if code != hmmer_pure_rs::alphabet::DSQ_ILLEGAL
                && code != hmmer_pure_rs::alphabet::DSQ_IGNORED
                && abc.is_residue(code)
            {
                break;
            }
            *ch = b'~';
        }
        for ch in row.iter_mut().rev() {
            let code = abc.digitize_symbol(*ch);
            if code != hmmer_pure_rs::alphabet::DSQ_ILLEGAL
                && code != hmmer_pure_rs::alphabet::DSQ_IGNORED
                && abc.is_residue(code)
            {
                break;
            }
            *ch = b'~';
        }
    }
}

fn model_ranges_to_alignment_ranges(
    ranges: &[(usize, usize)],
    model_to_ali: &[usize],
) -> Result<Vec<(usize, usize)>, String> {
    if model_to_ali.is_empty() {
        return Err("alimask could not infer any model consensus columns".to_string());
    }

    let mut ali_ranges = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        if start == 0 {
            return Err(
                "Mask ranges can not start before position 1; start 0 is invalid".to_string(),
            );
        }
        if end > model_to_ali.len() {
            return Err(format!(
                "Maximum model mask range {} exceeds model length {}",
                end,
                model_to_ali.len()
            ));
        }
        ali_ranges.push((model_to_ali[start - 1], model_to_ali[end - 1]));
    }
    Ok(ali_ranges)
}

fn make_model_to_alignment_map(
    alignment: &msa::Msa,
    abc: Option<&Alphabet>,
    hand: bool,
    symfrac: f64,
    fragthresh: f64,
    weighting_strategy: RelativeWeighting,
) -> Result<Vec<usize>, String> {
    if hand {
        let Some(ref rf) = alignment.rf else {
            return Err("Model file does not contain an RF line, required for --hand.".to_string());
        };
        let map: Vec<usize> = rf
            .iter()
            .enumerate()
            .filter_map(|(idx, &sym)| is_rf_consensus(sym).then_some(idx + 1))
            .collect();
        if map.is_empty() {
            return Err("alimask --hand requires at least one RF consensus column".to_string());
        }
        return Ok(map);
    }

    let abc = abc.expect("alphabet is required for fast model construction");
    let mask = builder::model_mask_from_msa(
        alignment,
        abc,
        symfrac as f32,
        fragthresh as f32,
        false,
        weighting_strategy,
    );
    let map: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(idx, &sym)| (sym == b'x').then_some(idx + 1))
        .collect();
    if map.is_empty() {
        return Err("alimask could not infer any model consensus columns".to_string());
    }
    Ok(map)
}

fn write_model_to_alignment_map(
    ranges: &[(usize, usize)],
    model_to_ali: &[usize],
) -> Result<(), String> {
    println!("model coordinates     alignment coordinates");
    for &(start, end) in ranges {
        if start == 0 {
            return Err(
                "Mask ranges can not start before position 1; start 0 is invalid".to_string(),
            );
        }
        if end > model_to_ali.len() {
            return Err(format!(
                "Maximum mask range {} exceeds computed model length {}",
                end,
                model_to_ali.len()
            ));
        }
        println!(
            "{:8}..{:<8} -> {:8}..{:<8}",
            start,
            end,
            model_to_ali[start - 1],
            model_to_ali[end - 1]
        );
    }
    Ok(())
}

fn write_alignment_to_model_map(
    ranges: &[(usize, usize)],
    model_to_ali: &[usize],
    alen: usize,
) -> Result<(), String> {
    println!("alignment coordinates     model coordinates");
    for &(start, end) in ranges {
        if start == 0 {
            return Err(
                "Mask ranges can not start before position 1; start 0 is invalid".to_string(),
            );
        }
        if end > alen {
            return Err(format!(
                "Maximum mask range {} exceeds alignment length {}",
                end, alen
            ));
        }
        let first = model_to_ali
            .iter()
            .position(|&apos| apos >= start && apos <= end);
        let Some(first) = first else {
            println!("   {:8}..{:<8} ->       -..-  (no map)", start, end);
            continue;
        };
        let last = model_to_ali
            .iter()
            .rposition(|&apos| apos >= start && apos <= end)
            .unwrap_or(first);
        println!(
            "   {:8}..{:<8} -> {:8}..{:<8}",
            model_to_ali[first],
            model_to_ali[last],
            first + 1,
            last + 1
        );
    }
    Ok(())
}

/// Compute the relative weights that C's `alimask` would write as `#=GS ... WT`
/// in `--modelrange` mode. Returns `Some` only for the weighting schemes that
/// raise `eslMSA_HASWGTS` in C (`--wpb`/`--wgsc`/`--wblosum`); `--wnone` and
/// `--wgiven` return `None` (no WT lines written), matching C.
fn compute_output_weights(
    alignment: &msa::Msa,
    abc: &Alphabet,
    hand: bool,
    weighting_strategy: RelativeWeighting,
) -> Option<Vec<f64>> {
    match weighting_strategy {
        RelativeWeighting::PositionBased => Some(builder::pb_weights(alignment, abc, !hand)),
        RelativeWeighting::Gsc => Some(builder::gsc_weights(alignment, abc)),
        RelativeWeighting::Blosum { identity_cutoff } => {
            Some(builder::blosum_weights(alignment, abc, identity_cutoff))
        }
        RelativeWeighting::None | RelativeWeighting::Given => None,
    }
}

fn is_rf_consensus(sym: u8) -> bool {
    sym != b'.' && sym != b'-' && sym != b'_' && sym != b'~'
}

/// Read the input alignment(s) twice: once into the simplified `StockholmMsa`
/// model (used for the coordinate/weighting/masking math) and once into the
/// full-fidelity `FullStockholm` model (used to re-serialize the masked MSA
/// byte-for-byte like C's `esl_msafile_Write`). When the input is stdin, the
/// bytes are buffered so both parses see identical content.
fn read_stockholm_maybe_stdin(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<(Vec<msa::StockholmMsa>, Vec<msa::FullStockholm>)> {
    if path == std::path::Path::new("-") {
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin().lock(), &mut bytes)
            .map_err(hmmer_pure_rs::errors::HmmerError::Io)?;
        let preserved = msa::read_stockholm_preserved_from_reader(BufReader::new(&bytes[..]))?;
        let full = msa::read_stockholm_full_from_reader(BufReader::new(&bytes[..]))?;
        Ok((preserved, full))
    } else {
        let preserved = msa::read_stockholm_preserved(path)?;
        let full = msa::read_stockholm_full(path)?;
        Ok((preserved, full))
    }
}

fn is_stockholm_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("stockholm")
        || format.eq_ignore_ascii_case("sto")
        || format.eq_ignore_ascii_case("pfam")
}

fn is_pfam_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("pfam")
}
