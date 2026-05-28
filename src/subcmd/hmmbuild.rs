//! hmmbuild — build profile HMM(s) from multiple sequence alignment(s).

use std::io::{BufReader, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use hmmer_pure_rs::alphabet::{Alphabet, AlphabetType};
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::builder;
use hmmer_pure_rs::calibrate::CalibrationConfig;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::msa;
use hmmer_pure_rs::output::{fmt_fixed3, fmt_width6_3, fmt_width8_2};
use hmmer_pure_rs::prior::PriorStrategy;
use hmmer_pure_rs::seqmodel;
use hmmer_pure_rs::util::cmath::{c_log_f64, ESL_CONST_LOG2R};

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

    /// Resave processed input alignment to file
    #[arg(short = 'O')]
    msa_out: Option<PathBuf>,

    /// Assert input alignment file format
    #[arg(long = "informat")]
    informat: Option<String>,

    /// Use DNA alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["rna", "amino"])]
    dna: bool,

    /// Use RNA alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["dna", "amino"])]
    rna: bool,

    /// Use protein alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["dna", "rna"])]
    amino: bool,

    /// Assign consensus columns from RF annotation
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "fast")]
    hand: bool,

    /// Assign consensus columns by residue fraction
    #[arg(long, action = ArgAction::SetTrue)]
    fast: bool,

    /// Henikoff position-based weights (default; accepted for C compatibility)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wgsc", "wblosum", "wnone", "wgiven"])]
    wpb: bool,

    /// Gerstein/Sonnhammer/Chothia tree weights
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wblosum", "wnone", "wgiven"])]
    wgsc: bool,

    /// Henikoff simple filter weights
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wnone", "wgiven"])]
    wblosum: bool,

    /// No relative sequence weighting; set all weights to 1
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wblosum", "wgiven"])]
    wnone: bool,

    /// Use weights as given in MSA file
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wblosum", "wnone"])]
    wgiven: bool,

    /// For --wblosum: set identity cutoff
    #[arg(long = "wid", default_value = "0.62", value_parser = parse_unit_f64)]
    wid: f64,

    /// Entropy effective sequence weighting (default; accepted for C compatibility)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eentexp", "eclust", "enone", "eset"])]
    eent: bool,

    /// Entropy effective sequence weighting with exponent-based scaling
    #[arg(long = "eentexp", action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eclust", "enone", "eset"])]
    eentexp: bool,

    /// Set effective sequence number to single-linkage cluster count
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eentexp", "enone", "eset"])]
    eclust: bool,

    /// For --eclust: set identity cutoff
    #[arg(long = "eid", default_value = "0.62", value_parser = parse_unit_f64)]
    eid: f64,

    /// No effective sequence weighting; use the raw sequence count
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eentexp", "eclust", "eset"])]
    enone: bool,

    /// Set effective sequence number for all models
    #[arg(long = "eset", conflicts_with_all = ["eent", "eentexp", "eclust", "enone"])]
    eset: Option<f32>,

    /// No prior; use observed counts only
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "plaplace")]
    pnone: bool,

    /// Use a Laplace +1 prior
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "pnone")]
    plaplace: bool,

    /// Minimum relative entropy target for entropy effective sequence weighting
    #[arg(long = "ere", value_parser = parse_positive_f64)]
    ere: Option<f64>,

    /// Entropy target sigma parameter
    #[arg(long = "esigma", default_value = "45.0", value_parser = parse_positive_f64)]
    esigma: f64,

    /// Random number seed for E-value calibration
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Number of parallel CPU workers (accepted for C compatibility)
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    #[arg(
        long = "stall",
        action = ArgAction::SetTrue,
        help = "arrest after start: for attaching debugger to process"
    )]
    stall: bool,

    /// Sym fraction threshold for match/insert (default 0.5)
    #[arg(long = "symfrac", default_value = "0.5", value_parser = parse_unit_f32, conflicts_with = "hand")]
    symfrac: f32,

    /// Sequence is called a fragment if L <= x*alignment_length
    #[arg(long = "fragthresh", default_value = "0.5", value_parser = parse_unit_f32)]
    fragthresh: f32,

    /// Use single-sequence builder path for one-sequence alignments
    #[arg(long = "singlemx", action = ArgAction::SetTrue)]
    singlemx: bool,

    /// Substitution score matrix choice for --singlemx
    #[arg(long = "mx", default_value = "BLOSUM62", conflicts_with = "mxfile")]
    matrix: String,

    /// Read substitution score matrix from file
    #[arg(long = "mxfile", conflicts_with = "matrix")]
    mxfile: Option<PathBuf>,

    /// Gap open probability for --singlemx
    #[arg(long = "popen", default_value = "0.02", value_parser = parse_popen)]
    popen: f32,

    /// Gap extend probability for --singlemx
    #[arg(long = "pextend", default_value = "0.4", value_parser = parse_pextend)]
    pextend: f32,

    /// Length of sequences for MSV Gumbel mu fit
    #[arg(long = "EmL", default_value = "200", value_parser = parse_positive_usize)]
    em_l: usize,

    /// Number of sequences for MSV Gumbel mu fit
    #[arg(long = "EmN", default_value = "200", value_parser = parse_positive_usize)]
    em_n: usize,

    /// Length of sequences for Viterbi Gumbel mu fit
    #[arg(long = "EvL", default_value = "200", value_parser = parse_positive_usize)]
    ev_l: usize,

    /// Number of sequences for Viterbi Gumbel mu fit
    #[arg(long = "EvN", default_value = "200", value_parser = parse_positive_usize)]
    ev_n: usize,

    /// Length of sequences for Forward exp tail tau fit
    #[arg(long = "EfL", default_value = "100", value_parser = parse_positive_usize)]
    ef_l: usize,

    /// Number of sequences for Forward exp tail tau fit
    #[arg(long = "EfN", default_value = "200", value_parser = parse_positive_usize)]
    ef_n: usize,

    /// Tail mass for Forward exponential tail tau fit
    #[arg(long = "Eft", default_value = "0.04", value_parser = parse_open_unit_f64)]
    eft: f64,

    /// Tail mass for deriving nucleotide model window length
    #[arg(long = "w_beta", value_parser = parse_window_beta)]
    w_beta: Option<f64>,

    /// Nucleotide model window length
    #[arg(long = "w_length", value_parser = parse_window_length)]
    w_length: Option<usize>,

    /// Pretend all inserts are length <= n
    #[arg(long = "maxinsertlen", value_parser = parse_maxinsertlen)]
    max_insert_len: Option<usize>,
}

struct BuildAlignment {
    msa: msa::Msa,
    cutoffs: msa::StockholmCutoffs,
    force_hand_arch: bool,
}

impl std::ops::Deref for BuildAlignment {
    type Target = msa::Msa;

    fn deref(&self) -> &Self::Target {
        &self.msa
    }
}

fn parse_positive_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid positive number: {e}"))?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
    }
}

fn parse_positive_usize(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid positive integer: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
    }
}

fn parse_open_unit_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid probability: {e}"))?;
    if value > 0.0 && value < 1.0 {
        Ok(value)
    } else {
        Err("value must be > 0 and < 1".to_string())
    }
}

fn parse_window_beta(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid window-length beta value: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("Invalid window-length beta value".to_string())
    }
}

fn parse_window_length(s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|e| format!("invalid window length: {e}"))
}

fn parse_maxinsertlen(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid max insert length: {e}"))?;
    if value >= 5 {
        Ok(value)
    } else {
        Err("value must be >= 5".to_string())
    }
}

fn parse_unit_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid probability: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("value must be >= 0 and <= 1".to_string())
    }
}

/// Parse an f32 in C's `0<=x<=1` range (used for --symfrac and --fragthresh).
fn parse_unit_f32(s: &str) -> Result<f32, String> {
    let value = s
        .parse::<f32>()
        .map_err(|e| format!("invalid real value: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("value must be >= 0 and <= 1".to_string())
    }
}

fn parse_popen(s: &str) -> Result<f32, String> {
    let value = s
        .parse::<f32>()
        .map_err(|e| format!("invalid gap open probability: {e}"))?;
    if (0.0..0.5).contains(&value) {
        Ok(value)
    } else {
        Err("--popen must be >= 0 and < 0.5".to_string())
    }
}

fn parse_pextend(s: &str) -> Result<f32, String> {
    let value = s
        .parse::<f32>()
        .map_err(|e| format!("invalid gap extend probability: {e}"))?;
    if (0.0..1.0).contains(&value) {
        Ok(value)
    } else {
        Err("--pextend must be >= 0 and < 1".to_string())
    }
}

/// Entry point for `hmmbuild`: build profile HMM(s) from MSA(s) and write them
/// to a single output file.
///
/// Streams a Stockholm input, calls the builder pipeline per alignment, applies
/// optional name/alphabet overrides, and prints HMMER 3.4's per-MSA summary line
/// (`idx`, name, nodes, nseq, eff_nseq). Corresponds to `main()` /
/// `output_result()` in hmmer/src/hmmbuild.c (single-threaded path only).
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let seed_was_requested = args
        .iter()
        .any(|arg| arg == "--seed" || arg.starts_with("--seed="));
    let cpu_was_requested = args
        .iter()
        .any(|arg| arg == "--cpu" || arg.starts_with("--cpu="));
    let wid_was_requested = args
        .iter()
        .any(|arg| arg == "--wid" || arg.starts_with("--wid="));
    let eid_was_requested = args
        .iter()
        .any(|arg| arg == "--eid" || arg.starts_with("--eid="));
    let em_l_was_requested = args
        .iter()
        .any(|arg| arg == "--EmL" || arg.starts_with("--EmL="));
    let em_n_was_requested = args
        .iter()
        .any(|arg| arg == "--EmN" || arg.starts_with("--EmN="));
    let ev_l_was_requested = args
        .iter()
        .any(|arg| arg == "--EvL" || arg.starts_with("--EvL="));
    let ev_n_was_requested = args
        .iter()
        .any(|arg| arg == "--EvN" || arg.starts_with("--EvN="));
    let ef_l_was_requested = args
        .iter()
        .any(|arg| arg == "--EfL" || arg.starts_with("--EfL="));
    let ef_n_was_requested = args
        .iter()
        .any(|arg| arg == "--EfN" || arg.starts_with("--EfN="));
    let eft_was_requested = args
        .iter()
        .any(|arg| arg == "--Eft" || arg.starts_with("--Eft="));
    let popen_was_requested = args
        .iter()
        .any(|arg| arg == "--popen" || arg.starts_with("--popen="));
    let pextend_was_requested = args
        .iter()
        .any(|arg| arg == "--pextend" || arg.starts_with("--pextend="));
    let mx_was_requested = args
        .iter()
        .any(|arg| arg == "--mx" || arg.starts_with("--mx="));
    let args = Args::parse_from(&args);
    let _stall_compat = args.stall;

    // Alphabet (--amino/--dna/--rna) and construction (--fast/--hand) mutual
    // exclusion, the --symfrac/--fragthresh 0<=x<=1 range, and "--symfrac
    // requires --fast" (clap: conflicts_with --hand) are all enforced at parse
    // time by clap, matching C's ALPHOPTS/CONOPTS toggles, range columns, and
    // --symfrac reqs --fast in hmmbuild.c.
    if wid_was_requested && !args.wblosum {
        eprintln!("Error: --wid only works in combination with --wblosum");
        std::process::exit(1);
    }
    if eid_was_requested && !args.eclust {
        eprintln!("Error: --eid only works in combination with --eclust");
        std::process::exit(1);
    }
    if matches!(args.w_length, Some(1..=3)) {
        eprintln!("Invalid window length value");
        std::process::exit(1);
    }
    if args.hmmfile == std::path::Path::new("-") {
        eprintln!("Error: hmmbuild cannot write <hmmfile_out> to stdout; use a file path");
        std::process::exit(1);
    }
    if !args.singlemx && (mx_was_requested || args.mxfile.is_some()) {
        eprintln!("Error: hmmbuild --mx and --mxfile currently require --singlemx");
        std::process::exit(1);
    }
    if args.singlemx {
        if let Some(path) = args.mxfile.as_ref() {
            std::fs::File::open(path).unwrap_or_else(|e| {
                eprintln!(
                    "Error: failed to read score matrix file {}: {e}",
                    path.display()
                );
                std::process::exit(1);
            });
        } else if mx_was_requested && !seqmodel::is_known_builtin_score_matrix_name(&args.matrix) {
            eprintln!(
                "Error: unknown built-in protein score matrix {}; supported matrices are PAM30, PAM70, PAM120, PAM240, BLOSUM45, BLOSUM50, BLOSUM62, BLOSUM80, BLOSUM90",
                args.matrix
            );
            std::process::exit(1);
        }
    }
    if args.msafile == std::path::Path::new("-") && args.informat.is_none() {
        println!("Must specify --informat to read <alifile> from stdin ('-')");
        std::process::exit(1);
    }
    if let Some(ref informat) = args.informat {
        if !is_supported_msa_format(informat) {
            eprintln!("{informat} is not a recognized input alignment file format");
            std::process::exit(1);
        }
    }

    let alignments = read_build_alignments_maybe_stdin(&args.msafile, args.informat.as_deref())
        .unwrap_or_else(|e| {
            eprintln!("Error reading MSA file: {}", e);
            std::process::exit(1);
        });
    if args.name.is_some() && alignments.len() > 1 {
        eprintln!("Error: You can't use -n with an alignment database");
        std::process::exit(1);
    }
    let Some(first_alignment) = alignments.first() else {
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
        Alphabet::new(
            guess_msa_alphabet(&first_alignment.msa).unwrap_or_else(|e| {
                eprintln!("{e}; please specify --amino, --dna, or --rna");
                std::process::exit(1);
            }),
        )
    };
    for alignment in &alignments {
        if let Err(e) = alignment.validate_digitizable(&abc) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
    let bg = Bg::new(&abc);
    let nucleotide_singlemx =
        args.singlemx && matches!(abc.abc_type, AlphabetType::Dna | AlphabetType::Rna);
    let singlemx_popen = if nucleotide_singlemx && !popen_was_requested {
        0.03125
    } else {
        args.popen
    };
    let singlemx_pextend = if nucleotide_singlemx && !pextend_was_requested {
        0.75
    } else {
        args.pextend
    };
    let score_matrix = if args.singlemx {
        if let Some(path) = args.mxfile.as_ref() {
            seqmodel::ScoreMatrix::from_file_for_alphabet(path, &abc).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            })
        } else {
            let matrix_name = if nucleotide_singlemx && !mx_was_requested {
                "DNA1"
            } else {
                args.matrix.as_str()
            };
            seqmodel::ScoreMatrix::builtin_for_alphabet(matrix_name, abc.abc_type).unwrap_or_else(
                |e| {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                },
            )
        }
    } else {
        seqmodel::ScoreMatrix::blosum62()
    };

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
    if let Some(ref path) = args.msa_out {
        writeln!(
            summary,
            "# processed alignment resaved to:   {}",
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
            "# sym fraction for model structure: {}",
            fmt_fixed3(args.symfrac as f64)
        )
        .unwrap();
    }
    if args.fragthresh != 0.5 {
        writeln!(
            summary,
            "# seq called frag if L <= x*alen:  {}",
            fmt_fixed3(args.fragthresh as f64)
        )
        .unwrap();
    }
    if args.singlemx {
        writeln!(
            summary,
            "# single sequence builder:          substitution matrix"
        )
        .unwrap();
        writeln!(summary, "# use score matrix for 1-seq MSAs:  on").unwrap();
        writeln!(
            summary,
            "# substitution score matrix:        {}",
            score_matrix.name()
        )
        .unwrap();
    }
    if popen_was_requested {
        writeln!(
            summary,
            "# gap open probability:             {:.6}",
            args.popen
        )
        .unwrap();
    }
    if pextend_was_requested {
        writeln!(
            summary,
            "# gap extend probability:           {:.6}",
            args.pextend
        )
        .unwrap();
    }
    if args.wnone {
        writeln!(summary, "# relative weighting scheme:        none").unwrap();
    }
    if args.wgsc {
        writeln!(summary, "# relative weighting scheme:        G/S/C").unwrap();
    }
    if args.wblosum {
        writeln!(summary, "# relative weighting scheme:        BLOSUM filter").unwrap();
    }
    if args.wgiven {
        writeln!(summary, "# relative weighting scheme:        given").unwrap();
    }
    if args.wblosum {
        writeln!(
            summary,
            "# frac id cutoff for BLOSUM wgts:   {:.6}",
            args.wid
        )
        .unwrap();
    }
    if args.enone {
        writeln!(summary, "# effective seq number scheme:      none").unwrap();
    }
    if args.eentexp {
        writeln!(
            summary,
            "# effective seq number scheme:      entropy weighting using exponent-based scaling"
        )
        .unwrap();
    }
    if args.eclust {
        writeln!(
            summary,
            "# effective seq number scheme:      single linkage clusters"
        )
        .unwrap();
    }
    if let Some(eset) = args.eset {
        writeln!(
            summary,
            "# effective seq number:             set to {:.6}",
            eset
        )
        .unwrap();
    }
    if args.eclust {
        writeln!(
            summary,
            "# frac id cutoff for --eclust:      {:.6}",
            args.eid
        )
        .unwrap();
    }
    if args.pnone {
        writeln!(summary, "# prior scheme:                     none").unwrap();
    }
    if args.plaplace {
        writeln!(summary, "# prior scheme:                     Laplace").unwrap();
    }
    if let Some(ere) = args.ere {
        writeln!(
            summary,
            "# minimum rel entropy target:       {:.6} bits",
            ere
        )
        .unwrap();
    }
    if args.esigma != 45.0 {
        writeln!(
            summary,
            "# entropy target sigma parameter:   {:.6} bits",
            args.esigma
        )
        .unwrap();
    }
    if seed_was_requested {
        if args.seed == 0 {
            writeln!(
                summary,
                "# random number seed:               one-time arbitrary"
            )
            .unwrap();
        } else {
            writeln!(summary, "# random number seed set to:        {}", args.seed).unwrap();
        }
    }
    if cpu_was_requested {
        writeln!(summary, "# number of worker threads:         {}", args.cpu).unwrap();
    }
    if em_l_was_requested {
        writeln!(summary, "# seq length for MSV Gumbel mu fit: {}", args.em_l).unwrap();
    }
    if em_n_was_requested {
        writeln!(summary, "# seq number for MSV Gumbel mu fit: {}", args.em_n).unwrap();
    }
    if ev_l_was_requested {
        writeln!(summary, "# seq length for Vit Gumbel mu fit: {}", args.ev_l).unwrap();
    }
    if ev_n_was_requested {
        writeln!(summary, "# seq number for Vit Gumbel mu fit: {}", args.ev_n).unwrap();
    }
    if ef_l_was_requested {
        writeln!(summary, "# seq length for Fwd exp tau fit:   {}", args.ef_l).unwrap();
    }
    if ef_n_was_requested {
        writeln!(summary, "# seq number for Fwd exp tau fit:   {}", args.ef_n).unwrap();
    }
    if eft_was_requested {
        writeln!(
            summary,
            "# tail mass for Fwd exp tau fit:    {:.6}",
            args.eft
        )
        .unwrap();
    }
    if let Some(max_insert_len) = args.max_insert_len {
        writeln!(
            summary,
            "# max insert length:                {}",
            max_insert_len
        )
        .unwrap();
    }
    if let Some(w_beta) = args.w_beta {
        writeln!(
            summary,
            "# window length tail mass:          {} bits",
            w_beta
        )
        .unwrap();
    }
    if let Some(w_length) = args.w_length {
        writeln!(summary, "# window length :                   {}", w_length).unwrap();
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

    let mut msa_out = args.msa_out.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating processed alignment output file: {}", e);
            std::process::exit(1);
        })
    });
    let calibration_config = CalibrationConfig {
        em_l: args.em_l,
        em_n: args.em_n,
        ev_l: args.ev_l,
        ev_n: args.ev_n,
        ef_l: args.ef_l,
        ef_n: args.ef_n,
        eft: args.eft,
    };

    for (idx, alignment) in alignments.iter().enumerate() {
        let hand_arch = args.hand || alignment.force_hand_arch;
        if hand_arch && alignment.rf.is_none() {
            eprintln!("Model file does not contain an RF line, required for --hand.");
            std::process::exit(1);
        }
        if args.singlemx && alignment.nseq != 1 {
            eprintln!("Error: hmmbuild --singlemx requires alignments with exactly one sequence");
            std::process::exit(1);
        }
        let weighting_strategy = if args.wnone {
            builder::RelativeWeighting::None
        } else if args.wgsc {
            builder::RelativeWeighting::Gsc
        } else if args.wblosum {
            builder::RelativeWeighting::Blosum {
                identity_cutoff: args.wid,
            }
        } else if args.wgiven {
            builder::RelativeWeighting::Given
        } else {
            builder::RelativeWeighting::PositionBased
        };
        if let Some(ref mut out) = msa_out {
            let model_mask = builder::model_mask_from_msa(
                alignment,
                &abc,
                args.symfrac,
                args.fragthresh,
                hand_arch,
                weighting_strategy,
            );
            write_processed_msa(out, alignment, &model_mask);
        }
        let mut hmm = if args.singlemx {
            build_single_sequence_hmm(
                alignment,
                &abc,
                &bg,
                &score_matrix,
                singlemx_popen,
                singlemx_pextend,
            )
        } else {
            builder::build_hmm_from_msa_with_prior_and_max_insert(
                alignment,
                &abc,
                &bg,
                args.symfrac,
                args.fragthresh,
                hand_arch,
                weighting_strategy,
                if args.enone {
                    builder::EffectiveSeqNumber::None
                } else if args.eclust {
                    builder::EffectiveSeqNumber::Cluster {
                        identity_cutoff: args.eid,
                    }
                } else if let Some(eset) = args.eset {
                    builder::EffectiveSeqNumber::Set(eset)
                } else if args.eentexp {
                    builder::EffectiveSeqNumber::EntropyExp {
                        target_re: args.ere,
                        target_sigma: Some(args.esigma),
                    }
                } else {
                    builder::EffectiveSeqNumber::Entropy {
                        target_re: args.ere,
                        target_sigma: Some(args.esigma),
                    }
                },
                if args.pnone {
                    PriorStrategy::None
                } else if args.plaplace {
                    PriorStrategy::Laplace
                } else {
                    PriorStrategy::Default
                },
                calibration_config,
                args.seed,
                args.max_insert_len,
            )
        };
        if args.singlemx {
            hmmer_pure_rs::calibrate::calibrate_with_config(
                &mut hmm,
                &abc,
                &bg,
                args.seed,
                calibration_config,
            );
            apply_window_length_options(&mut hmm, abc.abc_type, args.w_length, args.w_beta);
        } else if popen_was_requested || pextend_was_requested {
            apply_window_length_options(&mut hmm, abc.abc_type, args.w_length, args.w_beta);
            apply_fixed_gap_params(
                &mut hmm,
                popen_was_requested.then_some(args.popen),
                pextend_was_requested.then_some(args.pextend),
            );
            hmmer_pure_rs::calibrate::calibrate_with_config(
                &mut hmm,
                &abc,
                &bg,
                args.seed,
                calibration_config,
            );
        } else {
            apply_window_length_options(&mut hmm, abc.abc_type, args.w_length, args.w_beta);
        }
        builder::copy_stockholm_cutoffs_to_hmm(alignment.cutoffs, &mut hmm);

        if let Some(ref name) = args.name {
            hmm.name = name.clone();
        }

        let rel_entropy = mean_match_relative_entropy(&hmm, &bg);
        let description = hmm.desc.as_deref().unwrap_or("");
        if abc.abc_type == AlphabetType::Amino {
            writeln!(
                summary,
                "{:<5} {:<20} {:>5} {:>5} {:>5} {} {} {}",
                idx + 1,
                hmm.name,
                alignment.nseq,
                alignment.alen,
                hmm.m,
                fmt_width8_2(hmm.eff_nseq as f64),
                fmt_width6_3(rel_entropy as f64),
                description,
            )
            .unwrap();
        } else {
            writeln!(
                summary,
                "{:<5} {:<20} {:>5} {:>5} {:>5} {:>5} {} {} {}",
                idx + 1,
                hmm.name,
                alignment.nseq,
                alignment.alen,
                hmm.m,
                hmm.max_length,
                fmt_width8_2(hmm.eff_nseq as f64),
                fmt_width6_3(rel_entropy as f64),
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

fn read_build_alignments_maybe_stdin(
    path: &std::path::Path,
    format: Option<&str>,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<BuildAlignment>> {
    if format.is_some_and(is_a2m_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_a2m_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_a2m(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: true,
            })
            .collect())
    } else if format.is_some_and(is_afa_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_afa_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_afa(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
            })
            .collect())
    } else if format.is_some_and(is_psiblast_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_psiblast_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_psiblast(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
            })
            .collect())
    } else if format.is_some_and(is_clustal_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_clustal_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_clustal(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
            })
            .collect())
    } else if format.is_some_and(is_selex_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_selex_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_selex(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
            })
            .collect())
    } else if format.is_some_and(is_phylip_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_phylip_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_phylip(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
            })
            .collect())
    } else if format.is_some_and(is_phylips_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_phylips_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_phylips(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
            })
            .collect())
    } else if path == std::path::Path::new("-") {
        let stdin = std::io::stdin();
        Ok(
            msa::read_stockholm_preserved_from_reader(BufReader::new(stdin.lock()))?
                .into_iter()
                .map(|record| BuildAlignment {
                    msa: record.msa,
                    cutoffs: record.cutoffs,
                    force_hand_arch: false,
                })
                .collect(),
        )
    } else {
        Ok(msa::read_stockholm_preserved(path)?
            .into_iter()
            .map(|record| BuildAlignment {
                msa: record.msa,
                cutoffs: record.cutoffs,
                force_hand_arch: false,
            })
            .collect())
    }
}

fn is_supported_msa_format(format: &str) -> bool {
    is_stockholm_msa_format(format)
        || is_a2m_msa_format(format)
        || is_afa_msa_format(format)
        || is_psiblast_msa_format(format)
        || is_clustal_msa_format(format)
        || is_selex_msa_format(format)
        || is_phylip_msa_format(format)
        || is_phylips_msa_format(format)
}

fn is_stockholm_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("stockholm")
        || format.eq_ignore_ascii_case("sto")
        || format.eq_ignore_ascii_case("pfam")
}

fn is_a2m_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("a2m")
}

fn is_afa_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("afa")
        || format.eq_ignore_ascii_case("afasta")
        || format.eq_ignore_ascii_case("alignedfasta")
        || format.eq_ignore_ascii_case("aligned-fasta")
}

fn is_psiblast_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("psiblast")
}

fn is_clustal_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("clustal") || format.eq_ignore_ascii_case("clustallike")
}

fn is_selex_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("selex")
}

fn is_phylip_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("phylip")
}

fn is_phylips_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("phylips")
}

fn build_single_sequence_hmm(
    msa: &msa::Msa,
    abc: &Alphabet,
    bg: &Bg,
    matrix: &seqmodel::ScoreMatrix,
    popen: f32,
    pextend: f32,
) -> hmmer_pure_rs::Hmm {
    let mut dsq = vec![hmmer_pure_rs::alphabet::DSQ_SENTINEL];
    for &sym in &msa.aseq[0] {
        if matches!(sym, b'-' | b'.' | b'_' | b'~') || sym.is_ascii_whitespace() {
            continue;
        }
        let code = abc.digitize_symbol(sym);
        if code == hmmer_pure_rs::alphabet::DSQ_IGNORED || !abc.is_residue(code) {
            eprintln!(
                "Error: hmmbuild --singlemx sequence contains non-residue '{}'",
                sym as char
            );
            std::process::exit(1);
        }
        dsq.push(code);
    }
    if dsq.len() == 1 {
        eprintln!("Error: hmmbuild --singlemx requires at least one residue");
        std::process::exit(1);
    }
    dsq.push(hmmer_pure_rs::alphabet::DSQ_SENTINEL);
    let name = if !msa.name.is_empty() {
        msa.name.as_str()
    } else {
        msa.sqname[0].as_str()
    };
    let mut hmm = seqmodel::build_single_seq_hmm_with_matrix(
        name,
        &dsq,
        dsq.len() - 2,
        abc,
        bg,
        matrix,
        popen,
        pextend,
    )
    .unwrap_or_else(|e| {
        eprintln!("Error: hmmbuild --singlemx failed to build score matrix model: {e}");
        std::process::exit(1);
    });
    if let Some(ref acc) = msa.acc {
        hmm.acc = Some(acc.clone());
        hmm.flags |= hmmer_pure_rs::hmm::P7H_ACC;
    }
    if let Some(ref desc) = msa.desc {
        hmm.desc = Some(desc.clone());
        hmm.flags |= hmmer_pure_rs::hmm::P7H_DESC;
    }
    hmm
}

fn apply_window_length_options(
    hmm: &mut hmmer_pure_rs::Hmm,
    alphabet_type: AlphabetType,
    w_length: Option<usize>,
    w_beta: Option<f64>,
) {
    if !matches!(alphabet_type, AlphabetType::Dna | AlphabetType::Rna) {
        return;
    }
    if let Some(w_length) = w_length {
        hmm.max_length = w_length.min(i32::MAX as usize) as i32;
    } else if let Some(w_beta) = w_beta {
        if w_beta == 0.0 {
            hmm.max_length = (hmm.m.saturating_mul(4)).min(i32::MAX as usize) as i32;
        } else {
            builder::set_max_length_from_beta(hmm, w_beta);
        }
    }
}

fn apply_fixed_gap_params(hmm: &mut hmmer_pure_rs::Hmm, popen: Option<f32>, pextend: Option<f32>) {
    use hmmer_pure_rs::hmm::{DD, DM, II, IM, MD, MI, MM};

    for node in 0..=hmm.m {
        if let Some(popen) = popen {
            hmm.t[node][MM] = 1.0 - 2.0 * popen;
            hmm.t[node][MI] = popen;
            hmm.t[node][MD] = popen;
        }
        if let Some(pextend) = pextend {
            hmm.t[node][IM] = 1.0 - pextend;
            hmm.t[node][II] = pextend;
            hmm.t[node][DM] = 1.0 - pextend;
            hmm.t[node][DD] = pextend;
        }
    }

    if let Some(popen) = popen {
        hmm.t[hmm.m][MM] = 1.0 - popen;
    }
    hmm.t[hmm.m][MD] = 0.0;
    hmm.t[hmm.m][DM] = 1.0;
    hmm.t[hmm.m][DD] = 0.0;
}

fn write_processed_msa(out: &mut dyn Write, msa: &msa::Msa, model_mask: &[u8]) {
    let name_width = msa.sqname.iter().map(|name| name.len()).max().unwrap_or(0);
    writeln!(out, "# STOCKHOLM 1.0").unwrap();
    if !msa.name.is_empty() {
        writeln!(out, "#=GF ID {}", msa.name).unwrap();
    }
    if let Some(ref acc) = msa.acc {
        writeln!(out, "#=GF AC {}", acc).unwrap();
    }
    if let Some(ref desc) = msa.desc {
        writeln!(out, "#=GF DE {}", desc).unwrap();
    }
    writeln!(out).unwrap();

    for (idx, name) in msa.sqname.iter().enumerate() {
        let seq = String::from_utf8_lossy(&msa.aseq[idx]);
        writeln!(out, "{:<width$} {}", name, seq, width = name_width).unwrap();
    }
    let rf = String::from_utf8_lossy(model_mask);
    writeln!(out, "#=GC RF {}", rf).unwrap();
    writeln!(out, "//").unwrap();
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
                sum += p * c_log_f64((p / bg.f[x]) as f64) as f32 * (ESL_CONST_LOG2R as f32);
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

    #[test]
    fn hmmbuild_rejects_conflicting_alphabet_flags() {
        // C: ALPHOPTS toggle group -> "Options --amino and --dna conflict".
        assert!(
            Args::try_parse_from(["hmmbuild", "--amino", "--dna", "out.hmm", "in.sto"]).is_err()
        );
        assert!(Args::try_parse_from(["hmmbuild", "--dna", "--rna", "out.hmm", "in.sto"]).is_err());
        assert!(
            Args::try_parse_from(["hmmbuild", "--amino", "--rna", "out.hmm", "in.sto"]).is_err()
        );
    }

    #[test]
    fn hmmbuild_rejects_fast_and_hand_together() {
        // C: CONOPTS toggle group -> "Options --fast and --hand conflict".
        assert!(
            Args::try_parse_from(["hmmbuild", "--fast", "--hand", "out.hmm", "in.sto"]).is_err()
        );
    }

    #[test]
    fn hmmbuild_rejects_out_of_range_symfrac_and_fragthresh() {
        // C: --symfrac/--fragthresh range "0<=x<=1".
        assert!(Args::try_parse_from(["hmmbuild", "--symfrac", "2", "out.hmm", "in.sto"]).is_err());
        assert!(
            Args::try_parse_from(["hmmbuild", "--symfrac", "-0.1", "out.hmm", "in.sto"]).is_err()
        );
        assert!(
            Args::try_parse_from(["hmmbuild", "--fragthresh", "2", "out.hmm", "in.sto"]).is_err()
        );
        assert!(
            Args::try_parse_from(["hmmbuild", "--fragthresh", "-1", "out.hmm", "in.sto"]).is_err()
        );
        // Boundary values are accepted (0<=x<=1 is inclusive).
        assert!(Args::try_parse_from(["hmmbuild", "--symfrac", "0", "out.hmm", "in.sto"]).is_ok());
        assert!(Args::try_parse_from(["hmmbuild", "--symfrac", "1", "out.hmm", "in.sto"]).is_ok());
    }

    #[test]
    fn hmmbuild_symfrac_requires_fast_not_hand() {
        // C: --symfrac reqs --fast; with --hand (which toggles --fast off) it errors.
        assert!(Args::try_parse_from([
            "hmmbuild",
            "--hand",
            "--symfrac",
            "0.6",
            "out.hmm",
            "in.sto"
        ])
        .is_err());
        // --symfrac alone (fast is default-on) is fine.
        assert!(
            Args::try_parse_from(["hmmbuild", "--symfrac", "0.6", "out.hmm", "in.sto"]).is_ok()
        );
    }
}
