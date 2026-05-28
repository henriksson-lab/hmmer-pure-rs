//! jackhmmer — iteratively search a protein sequence against a protein database.
//! Builds single-seq HMM, searches, collects hits into MSA, rebuilds, repeats.

use std::io::Write;
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::builder;
use hmmer_pure_rs::calibrate::CalibrationConfig;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::msa::Msa;
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::prior::PriorStrategy;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::seqmodel;
use hmmer_pure_rs::sequence::{self, Sequence, SequenceFormat};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::{Hit, TopHits};
use hmmer_pure_rs::trace::{State, Trace};

const TARGET_BATCH_SIZE: usize = 256;

#[derive(Parser)]
#[command(
    name = "jackhmmer",
    about = "Iteratively search a protein sequence against a protein database"
)]
struct Args {
    /// Direct output to file <f>, not stdout
    #[arg(short = 'o')]
    output: Option<PathBuf>,

    /// Query sequence file (FASTA)
    seqfile: PathBuf,
    /// Target sequence database (FASTA)
    seqdb: PathBuf,

    /// Assert query sequence file format
    #[arg(long = "qformat")]
    qformat: Option<String>,

    /// Assert target sequence database format
    #[arg(long = "tformat")]
    tformat: Option<String>,

    /// Maximum number of iterations
    #[arg(short = 'N', default_value = "5", value_parser = parse_nonzero_usize)]
    max_iterations: usize,

    /// Report sequences <= this E-value threshold
    #[arg(
        short = 'E',
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "score_threshold"
    )]
    e_value: f64,

    /// Report sequences >= this score threshold
    #[arg(short = 'T', conflicts_with = "e_value", allow_hyphen_values = true)]
    score_threshold: Option<f64>,

    /// Report domains <= this E-value threshold
    #[arg(
        long = "domE",
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "dom_t"
    )]
    dom_e: f64,

    /// Report domains >= this score threshold
    #[arg(long = "domT", conflicts_with = "dom_e", allow_hyphen_values = true)]
    dom_t: Option<f64>,

    /// Include sequences <= this E-value threshold
    #[arg(
        long = "incE",
        default_value = "0.001",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_t"
    )]
    inc_e: f64,

    /// Include sequences >= this score threshold
    #[arg(long = "incT", conflicts_with = "inc_e", allow_hyphen_values = true)]
    inc_t: Option<f64>,

    /// Include domains <= this E-value threshold
    #[arg(
        long = "incdomE",
        default_value = "0.001",
        value_parser = parse_positive_f64,
        conflicts_with = "incdom_t"
    )]
    incdom_e: f64,

    /// Include domains >= this score threshold
    #[arg(long = "incdomT", conflicts_with = "incdom_e", allow_hyphen_values = true)]
    incdom_t: Option<f64>,

    /// Hidden C-compatible model gathering cutoff flag
    #[arg(long = "cut_ga", hide = true)]
    cut_ga: bool,

    /// Hidden C-compatible model noise cutoff flag
    #[arg(long = "cut_nc", hide = true)]
    cut_nc: bool,

    /// Hidden C-compatible model trusted cutoff flag
    #[arg(long = "cut_tc", hide = true)]
    cut_tc: bool,

    /// Turn all heuristic filters off
    #[arg(long = "max", conflicts_with_all = ["f1", "f2", "f3", "nobias"])]
    max: bool,

    /// MSV filter threshold
    #[arg(long = "F1", default_value = "0.02", allow_hyphen_values = true)]
    f1: f64,

    /// Viterbi filter threshold
    #[arg(long = "F2", default_value = "0.001", allow_hyphen_values = true)]
    f2: f64,

    /// Forward filter threshold
    #[arg(long = "F3", default_value = "1e-5", allow_hyphen_values = true)]
    f3: f64,

    /// Gap open probability for the single-sequence query model
    #[arg(long = "popen", default_value = "0.02", value_parser = parse_popen)]
    popen: f32,

    /// Gap extend probability for the single-sequence query model
    #[arg(long = "pextend", default_value = "0.4", value_parser = parse_pextend)]
    pextend: f32,

    /// Substitution score matrix choice
    #[arg(long = "mx", default_value = "BLOSUM62", conflicts_with = "mxfile")]
    matrix: String,

    /// Read substitution score matrix from file
    #[arg(long = "mxfile")]
    mxfile: Option<PathBuf>,

    /// Hidden C-compatible fast architecture option, prohibited by jackhmmer
    #[arg(long = "fast", hide = true)]
    fast: bool,

    /// Hidden C-compatible hand architecture option, prohibited by jackhmmer
    #[arg(long = "hand", hide = true)]
    hand: bool,

    /// Hidden C-compatible fast architecture threshold, prohibited by jackhmmer
    #[arg(long = "symfrac", hide = true)]
    symfrac: Option<f64>,

    /// Henikoff position-based weights for post-round rebuilds (default)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wgsc", "wblosum", "wnone"])]
    wpb: bool,

    /// Gerstein/Sonnhammer/Chothia tree weights for post-round rebuilds
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wblosum", "wnone"])]
    wgsc: bool,

    /// Henikoff simple filter weights for post-round rebuilds
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wnone"])]
    wblosum: bool,

    /// No relative sequence weighting in post-round rebuilds
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wblosum"])]
    wnone: bool,

    /// Hidden C-compatible given-weight option, prohibited by jackhmmer
    #[arg(long = "wgiven", hide = true)]
    wgiven: bool,

    /// For --wblosum: set identity cutoff
    #[arg(long = "wid", default_value = "0.62", value_parser = parse_unit_f64)]
    wid: f64,

    /// Entropy effective sequence weighting for post-round rebuilds (default)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eentexp", "eclust", "enone", "eset"])]
    eent: bool,

    /// Entropy effective sequence weighting with exponent-based scaling
    #[arg(long = "eentexp", action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eclust", "enone", "eset"])]
    eentexp: bool,

    /// Set post-round effective sequence number to single-linkage cluster count
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eentexp", "enone", "eset"])]
    eclust: bool,

    /// For --eclust: set identity cutoff
    #[arg(long = "eid", default_value = "0.62", value_parser = parse_unit_f64)]
    eid: f64,

    /// No post-round effective sequence weighting; use the raw sequence count
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eentexp", "eclust", "eset"])]
    enone: bool,

    /// Set post-round effective sequence number for all models
    #[arg(long = "eset", conflicts_with_all = ["eent", "eentexp", "eclust", "enone"])]
    eset: Option<f32>,

    /// No post-round prior; use observed counts only
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "plaplace")]
    pnone: bool,

    /// Use a post-round Laplace +1 prior
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "pnone")]
    plaplace: bool,

    /// Minimum relative entropy target for post-round entropy weighting
    #[arg(long = "ere", value_parser = parse_positive_f64)]
    ere: Option<f64>,

    /// Entropy target sigma parameter for post-round entropy weighting
    #[arg(long = "esigma", default_value = "45.0", value_parser = parse_positive_f64)]
    esigma: f64,

    /// Sequence is called a fragment if L <= x*alignment_length
    #[arg(long = "fragthresh", default_value = "0.5", value_parser = parse_unit_f32)]
    fragthresh: f32,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    nobias: bool,

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    nonull2: bool,

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    domz_value: Option<f64>,

    /// Random number seed
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Length of sequences for MSV Gumbel mu fit
    #[arg(long = "EmL", default_value = "200", value_parser = parse_nonzero_usize)]
    em_l: usize,

    /// Number of sequences for MSV Gumbel mu fit
    #[arg(long = "EmN", default_value = "200", value_parser = parse_nonzero_usize)]
    em_n: usize,

    /// Length of sequences for Viterbi Gumbel mu fit
    #[arg(long = "EvL", default_value = "200", value_parser = parse_nonzero_usize)]
    ev_l: usize,

    /// Number of sequences for Viterbi Gumbel mu fit
    #[arg(long = "EvN", default_value = "200", value_parser = parse_nonzero_usize)]
    ev_n: usize,

    /// Length of sequences for Forward exp tail tau fit
    #[arg(long = "EfL", default_value = "100", value_parser = parse_nonzero_usize)]
    ef_l: usize,

    /// Number of sequences for Forward exp tail tau fit
    #[arg(long = "EfN", default_value = "200", value_parser = parse_nonzero_usize)]
    ef_n: usize,

    /// Tail mass for Forward exponential tail tau fit
    #[arg(long = "Eft", default_value = "0.04", value_parser = parse_open_unit_f64)]
    eft: f64,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    domtblout: Option<PathBuf>,

    /// Save multiple alignment of hits to file
    #[arg(short = 'A')]
    ali_outfile: Option<PathBuf>,

    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    show_acc: bool,

    /// Omit alignments from the main output
    #[arg(long = "noali")]
    noali: bool,

    /// Do not line-wrap the main text output
    #[arg(long = "notextw", conflicts_with = "textw")]
    notextw: bool,

    /// Set the target line width for the main text output
    #[arg(long = "textw", default_value = "120", value_parser = parse_textw)]
    textw: usize,

    /// Save HMM checkpoints to files <f>-<iteration>.hmm
    #[arg(long = "chkhmm")]
    chkhmm: Option<PathBuf>,

    /// Save alignment checkpoints to files <f>-<iteration>.sto
    #[arg(long = "chkali")]
    chkali: Option<PathBuf>,
}

fn parse_nonzero_usize(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid positive integer: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
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

fn parse_unit_f32(s: &str) -> Result<f32, String> {
    let value = s
        .parse::<f32>()
        .map_err(|e| format!("invalid probability: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("value must be >= 0 and <= 1".to_string())
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

fn parse_textw(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid text width: {e}"))?;
    if value >= 120 {
        Ok(value)
    } else {
        Err("--textw must be >= 120".to_string())
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

/// Entry point for `jackhmmer`: iteratively search a protein query against a
/// protein database, rebuilding the HMM each round from included hits until
/// convergence or `-N` iterations.
///
/// Round 1 builds a single-sequence HMM (phmmer-style); subsequent rounds
/// reuse the previous round's included-hit MSA augmented with the original
/// query to rebuild the model. Each round runs the standard pipeline against
/// the target DB in bounded parallel batches, applies E/incE thresholds,
/// reports the canonical per-sequence tabular block, and optionally writes
/// `--chkhmm`/`--chkali`/`--tblout`/`--domtblout` checkpoints. Corresponds to
/// `serial_master()` in hmmer/src/jackhmmer.c.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let cpu_was_requested = args
        .iter()
        .any(|arg| arg == "--cpu" || arg.starts_with("--cpu="))
        || std::env::var_os("HMMER_NCPU").is_some();
    let matrix_was_requested = args
        .iter()
        .any(|arg| arg == "--mx" || arg.starts_with("--mx="));
    let mxfile_was_requested = args
        .iter()
        .any(|arg| arg == "--mxfile" || arg.starts_with("--mxfile="));
    let popen_was_requested = args
        .iter()
        .any(|arg| arg == "--popen" || arg.starts_with("--popen="));
    let pextend_was_requested = args
        .iter()
        .any(|arg| arg == "--pextend" || arg.starts_with("--pextend="));
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
    let args = hmmer_pure_rs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);
    if args.cut_ga {
        println!("Failed to parse command line: jackhmmer does not accept a --cut-ga option");
        std::process::exit(1);
    }
    if args.cut_nc {
        println!("Failed to parse command line: jackhmmer does not accept a --cut-nc option");
        std::process::exit(1);
    }
    if args.cut_tc {
        println!("Failed to parse command line: jackhmmer does not accept a --cut-tc option");
        std::process::exit(1);
    }
    if args.fast {
        println!("Failed to parse command line: jackhmmer does not accept a --fast option");
        std::process::exit(1);
    }
    if args.hand {
        println!("Failed to parse command line: jackhmmer does not accept a --hand option");
        std::process::exit(1);
    }
    if args.symfrac.is_some() {
        println!("Failed to parse command line: jackhmmer does not accept a --symfrac option");
        std::process::exit(1);
    }
    if args.wgiven {
        println!("Failed to parse command line: jackhmmer does not accept a --wgiven option");
        std::process::exit(1);
    }
    if wid_was_requested && !args.wblosum {
        eprintln!("Error: --wid only works in combination with --wblosum");
        std::process::exit(1);
    }
    if eid_was_requested && !args.eclust {
        eprintln!("Error: --eid only works in combination with --eclust");
        std::process::exit(1);
    }
    validate_sequence_format("jackhmmer --qformat", args.qformat.as_deref());
    validate_sequence_format("jackhmmer --tformat", args.tformat.as_deref());
    let rebuild_weighting = if args.wnone {
        builder::RelativeWeighting::None
    } else if args.wgsc {
        builder::RelativeWeighting::Gsc
    } else if args.wblosum {
        builder::RelativeWeighting::Blosum {
            identity_cutoff: args.wid,
        }
    } else {
        builder::RelativeWeighting::PositionBased
    };
    let rebuild_effn = if args.enone {
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
    };
    let rebuild_prior = if args.pnone {
        PriorStrategy::None
    } else if args.plaplace {
        PriorStrategy::Laplace
    } else {
        PriorStrategy::Default
    };
    let score_matrix = if let Some(mxfile) = args.mxfile.as_ref() {
        seqmodel::ScoreMatrix::from_file(mxfile)
    } else {
        seqmodel::ScoreMatrix::builtin(&args.matrix)
    }
    .unwrap_or_else(|e| {
        eprintln!("Error: jackhmmer {e}");
        std::process::exit(1);
    });
    if args.seqdb == PathBuf::from("-") {
        eprintln!("Error: target sequence database may not be '-' for jackhmmer");
        std::process::exit(1);
    }

    logsum::p7_flogsuminit();

    if args.cpu > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.cpu)
            .start_handler(|_| hmmer_pure_rs::util::simd_env::init())
            .build_global()
            .ok();
    }

    let abc = Alphabet::amino();
    let bg = Bg::new(&abc);

    let mut output_file = args.output.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open output file {path} for writing",
        )
    });
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let mut out: &mut dyn Write = match output_file {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };
    let mut tblout_file = args.tblout.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open tabular per-seq output file {path} for writing",
        )
    });
    let mut domtblout_file = args.domtblout.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open tabular per-dom output file {path} for writing",
        )
    });
    let mut ali_file = args.ali_outfile.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open alignment output file {path} for writing",
        )
    });

    writeln!(
        out,
        "# jackhmmer :: iteratively search a protein sequence against a protein database"
    )
    .unwrap();
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(out, "# Copyright (C) 2023 Howard Hughes Medical Institute.").unwrap();
    writeln!(
        out,
        "# Freely distributed under the BSD open source license."
    )
    .unwrap();
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(
        out,
        "# query sequence file:             {}",
        args.seqfile.display()
    )
    .unwrap();
    if let Some(format) = args.qformat.as_deref() {
        writeln!(out, "# query <seqfile> format asserted: {format}").unwrap();
    }
    writeln!(
        out,
        "# target sequence database:        {}",
        args.seqdb.display()
    )
    .unwrap();
    if let Some(format) = args.tformat.as_deref() {
        writeln!(out, "# target <seqdb> format asserted:  {format}").unwrap();
    }
    writeln!(
        out,
        "# maximum iterations set to:       {}",
        args.max_iterations
    )
    .unwrap();
    if let Some(output) = &args.output {
        writeln!(
            out,
            "# output directed to file:         {}",
            output.display()
        )
        .unwrap();
    }
    if let Some(tblout) = &args.tblout {
        writeln!(
            out,
            "# per-seq hits tabular output:     {}",
            tblout.display()
        )
        .unwrap();
    }
    if let Some(domtblout) = &args.domtblout {
        writeln!(
            out,
            "# per-dom hits tabular output:     {}",
            domtblout.display()
        )
        .unwrap();
    }
    if let Some(ali_outfile) = &args.ali_outfile {
        writeln!(
            out,
            "# MSA of hits saved to file:       {}",
            ali_outfile.display()
        )
        .unwrap();
    }
    if args.show_acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
    }
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    if matrix_was_requested {
        writeln!(
            out,
            "# subst score matrix (built-in):   {}",
            score_matrix.name()
        )
        .unwrap();
    }
    if mxfile_was_requested {
        writeln!(
            out,
            "# subst score matrix (file):       {}",
            args.mxfile.as_ref().unwrap().display()
        )
        .unwrap();
    }
    if popen_was_requested {
        writeln!(out, "# gap open probability:            {:.6}", args.popen).unwrap();
    }
    if pextend_was_requested {
        writeln!(
            out,
            "# gap extend probability:          {:.6}",
            args.pextend
        )
        .unwrap();
    }
    write_jackhmmer_option_header(&mut out, &args, &cmdline);
    let textw = if args.notextw { 0 } else { args.textw };
    if args.notextw {
        writeln!(out, "# max ASCII text line length:      unlimited").unwrap();
    } else if args.textw != 120 {
        writeln!(out, "# max ASCII text line length:      {}", args.textw).unwrap();
    }
    if args.max {
        writeln!(
            out,
            "# Max sensitivity mode:            on [all heuristic filters off]"
        )
        .unwrap();
    }
    if args.fragthresh != 0.5 {
        writeln!(
            out,
            "# define fragments if <= x*alen:   {:.3}",
            args.fragthresh
        )
        .unwrap();
    }
    if args.wpb {
        writeln!(out, "# relative weighting scheme:       Henikoff PB").unwrap();
    }
    if args.wgsc {
        writeln!(out, "# relative weighting scheme:       G/S/C").unwrap();
    }
    if args.wblosum {
        writeln!(out, "# relative weighting scheme:       BLOSUM filter").unwrap();
        writeln!(out, "# frac id cutoff for BLOSUM wgts:  {:.6}", args.wid).unwrap();
    }
    if args.wnone {
        writeln!(out, "# relative weighting scheme:       none").unwrap();
    }
    if args.eent {
        writeln!(out, "# effective seq number scheme:     entropy weighting").unwrap();
    }
    if args.eentexp {
        writeln!(
            out,
            "# effective seq number scheme:     entropy weighting using exponent-based scaling"
        )
        .unwrap();
    }
    if args.eclust {
        writeln!(
            out,
            "# effective seq number scheme:     single linkage clusters"
        )
        .unwrap();
        writeln!(out, "# frac id cutoff for --eclust:     {:.6}", args.eid).unwrap();
    }
    if args.enone {
        writeln!(out, "# effective seq number scheme:     none").unwrap();
    }
    if let Some(eset) = args.eset {
        writeln!(out, "# effective seq number:            set to {:.6}", eset).unwrap();
    }
    if let Some(ere) = args.ere {
        writeln!(out, "# minimum rel entropy target:      {:.6} bits", ere).unwrap();
    }
    if args.esigma != 45.0 {
        writeln!(
            out,
            "# entropy target sigma parameter:  {:.6} bits",
            args.esigma
        )
        .unwrap();
    }
    if args.pnone {
        writeln!(out, "# prior scheme:                    none").unwrap();
    }
    if args.plaplace {
        writeln!(out, "# prior scheme:                    Laplace +1").unwrap();
    }
    if cpu_was_requested {
        writeln!(out, "# number of worker threads:        {}", args.cpu).unwrap();
    }
    if em_l_was_requested {
        writeln!(out, "# seq length, MSV Gumbel mu fit:   {}", args.em_l).unwrap();
    }
    if em_n_was_requested {
        writeln!(out, "# seq number, MSV Gumbel mu fit:   {}", args.em_n).unwrap();
    }
    if ev_l_was_requested {
        writeln!(out, "# seq length, Vit Gumbel mu fit:   {}", args.ev_l).unwrap();
    }
    if ev_n_was_requested {
        writeln!(out, "# seq number, Vit Gumbel mu fit:   {}", args.ev_n).unwrap();
    }
    if ef_l_was_requested {
        writeln!(out, "# seq length, Fwd exp tau fit:     {}", args.ef_l).unwrap();
    }
    if ef_n_was_requested {
        writeln!(out, "# seq number, Fwd exp tau fit:     {}", args.ef_n).unwrap();
    }
    if eft_was_requested {
        writeln!(out, "# tail mass for Fwd exp tau fit:   {:.6}", args.eft).unwrap();
    }
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();

    let table_cmdline = normalize_jackhmmer_table_cmdline(&cmdline);
    let calibration_config = CalibrationConfig {
        em_l: args.em_l,
        em_n: args.em_n,
        ev_l: args.ev_l,
        ev_n: args.ev_n,
        ef_l: args.ef_l,
        ef_n: args.ef_n,
        eft: args.eft,
    };

    // Read query sequences. C jackhmmer runs the full iterative search once
    // for each FASTA record and keeps the outer report/footer shared.
    let mut query_sqf = open_sequence_file(&args.seqfile, &abc, args.qformat.as_deref())
        .unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });
    let mut query_sq = Sequence::new();
    let mut saw_query = false;
    let mut wrote_tblout_header = false;
    let mut wrote_domtblout_header = false;
    loop {
        query_sq.reuse();
        if !query_sqf.read(&mut query_sq).unwrap_or_else(|e| {
            eprintln!("Error reading query file: {}", e);
            std::process::exit(1);
        }) {
            if !saw_query {
                eprintln!("Error: no query sequence found");
                std::process::exit(1);
            }
            break;
        }
        saw_query = true;

        writeln!(out, "Query:       {}  [L={}]", query_sq.name, query_sq.n).unwrap();
        if !query_sq.acc.is_empty() {
            writeln!(out, "Accession:   {}", query_sq.acc).unwrap();
        }
        if !query_sq.desc.is_empty() {
            writeln!(out, "Description: {}", query_sq.desc).unwrap();
        }
        writeln!(out).unwrap();

        let query_tr = exact_match_query_trace(query_sq.n);

        let mut prev_included_names: Vec<String> = Vec::new();
        let mut prev_msa_nseq = 1usize;
        let mut prev_hmm: Option<hmmer_pure_rs::hmm::Hmm> = None;
        let mut final_hits: Option<TopHits> = None;
        let mut final_z: Option<f64> = None;
        let mut final_domz: Option<f64> = None;
        let mut final_model_len: Option<usize> = None;
        // MSA of included hits from the previous round. Built once at the
        // bottom of each iteration (matching C jackhmmer.c:683) and reused at
        // the top of the next iteration to rebuild the HMM, by `--chkali` at
        // the bottom of the current iteration, and (after the loop) by `-A`.
        // C builds the same `msa` once per round (jackhmmer.c:683-691, 723).
        let mut final_msa: Option<Msa> = None;

        for iteration in 1..=args.max_iterations {
            // Build HMM for this iteration
            let hmm = if iteration == 1 {
                // First iteration: single-sequence HMM (phmmer-style)
                seqmodel::build_single_seq_hmm_with_matrix_and_calibration(
                    &query_sq.name,
                    &query_sq.dsq,
                    query_sq.n,
                    &abc,
                    &bg,
                    &score_matrix,
                    args.popen,
                    args.pextend,
                    args.seed,
                    calibration_config,
                )
                .unwrap_or_else(|e| {
                    eprintln!("Error: jackhmmer failed to set single query seq score system: {e}");
                    std::process::exit(1);
                })
            } else {
                // C jackhmmer.c:683 builds the round's MSA once at end of
                // iteration; the same `msa` object is then handed to
                // `p7_Builder` at the top of the next round (line 605) and
                // freed at line 620. Mirror that: the MSA built at end of
                // the prior iteration is stashed in `final_msa`; take it
                // here so the rebuild consumes it and the storage is freed
                // before the new round's pipeline allocates `all_hits`.
                let msa = final_msa.take();
                // Match C jackhmmer.c:595's "destroy info->th at top of round":
                // the prior round's tophits (with all alidisplay payloads) is
                // no longer needed once its MSA has been materialized. Drop it
                // before the new round's all_hits accumulator and pipeline
                // scratch start growing.
                final_hits = None;

                let Some(msa) = msa else {
                    writeln!(out, "@@ No hits to build MSA from. Stopping.").unwrap();
                    break;
                };
                let prev_included_count = prev_included_names.len();
                let _ = prev_hmm
                    .as_ref()
                    .expect("jackhmmer rebuild requested without previous-round HMM");
                let mut hmm = builder::build_hmm_from_msa_with_prior(
                    &msa,
                    &abc,
                    &bg,
                    0.5,
                    args.fragthresh,
                    true,
                    rebuild_weighting,
                    rebuild_effn,
                    rebuild_prior,
                    // jackhmmer.c uses one P7_BUILDER (created once with the
                    // user's --EmL/--EmN/--EvL/--EvN/--EfL/--EfN/--Eft) for
                    // every round, so round 2+ rebuilds must calibrate with
                    // the same config as round 1, not Easel defaults.
                    calibration_config,
                    args.seed,
                );
                // Stash counts needed for the header banner, then drop the
                // MSA — C does the equivalent at jackhmmer.c:620 right after
                // the builder consumes it.
                let msa_nseq_for_banner = msa.nseq;
                drop(msa);
                if !query_sq.desc.is_empty() {
                    hmm.desc = Some(query_sq.desc.clone());
                    hmm.flags |= hmmer_pure_rs::hmm::P7H_DESC;
                }
                writeln!(out, "@@").unwrap();
                writeln!(out, "@@ Round:                  {}", iteration).unwrap();
                writeln!(
                out,
                "@@ Included in MSA:        {} subsequences (query + {} subseqs from {} targets)",
                msa_nseq_for_banner,
                msa_nseq_for_banner.saturating_sub(1),
                prev_included_count
            )
                .unwrap();
                writeln!(out, "@@ Model size:             {} positions", hmm.m).unwrap();
                writeln!(out, "@@").unwrap();
                writeln!(out).unwrap();
                hmm
            };

            if let Some(prefix) = &args.chkhmm {
                write_hmm_checkpoint(prefix, iteration, &hmm);
            }

            // Configure profile and search
            let mut local_bg = bg.clone();
            local_bg.set_filter(hmm.m, &hmm.compo);
            let mut gm = Profile::new(hmm.m, &abc);
            profile::profile_config(&hmm, &local_bg, &mut gm, 400, P7_LOCAL);
            let om = OProfile::convert(&gm);

            // Search the target DB in bounded batches so RSS scales with
            // the batch size instead of the full database size.
            use rayon::prelude::*;
            let mut all_hits = Vec::new();
            let mut z = 0usize;
            let mut total_residues = 0u64;
            let mut stats = PipelineStats::default();
            {
                let mut sqf = open_sequence_file(&args.seqdb, &abc, args.tformat.as_deref())
                    .unwrap_or_else(|e| {
                        eprintln!("Error: {}", e);
                        std::process::exit(1);
                    });
                let mut sq = Sequence::new();
                let mut batch = Vec::with_capacity(TARGET_BATCH_SIZE);

                loop {
                    while batch.len() < TARGET_BATCH_SIZE {
                        match sqf.read(&mut sq) {
                            Ok(true) => {
                                total_residues += sq.n as u64;
                                batch.push(sq.clone());
                                z += 1;
                                sq.reuse();
                            }
                            Ok(false) => break,
                            Err(e) => {
                                eprintln!("Error: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }

                    if batch.is_empty() {
                        break;
                    }

                    let batch_hits: Vec<(Option<Hit>, u64, u64, u64, u64)> = batch
                        .par_iter()
                        .map_init(
                            || {
                                hmmer_pure_rs::util::simd_env::init();
                                let lb = local_bg.clone();
                                let lgm = gm.clone();
                                let lom = om.clone();
                                let mut lpli = Pipeline::new();
                                configure_pipeline(&mut lpli, &args);
                                lpli.do_biasfilter = !args.nobias;
                                lpli.do_null2 = !args.nonull2;
                                lpli.seed = args.seed;
                                lpli.new_model(&lgm);
                                (lb, lgm, lom, lpli)
                            },
                            |(lb, lgm, lom, lpli), sq| {
                                // Mirror C's `p7_pipeline_Reuse`: zero the per-target
                                // counters before each target so the returned counts
                                // reflect only this sequence (the outer loop sums them
                                // into `stats`). DP scratch / profile / oprofile / bg
                                // are reused across all of this worker's targets.
                                lpli.n_targets = 0;
                                lpli.n_past_msv = 0;
                                lpli.n_past_bias = 0;
                                lpli.n_past_vit = 0;
                                lpli.n_past_fwd = 0;

                                lb.set_length(sq.n);

                                let mut lth = TopHits::new();
                                let hit = if lpli.run(lgm, lom, lb, &hmm, sq, &mut lth) {
                                    lth.hits.into_iter().next()
                                } else {
                                    None
                                };
                                (
                                    hit,
                                    lpli.n_past_msv,
                                    lpli.n_past_bias,
                                    lpli.n_past_vit,
                                    lpli.n_past_fwd,
                                )
                            },
                        )
                        .collect();
                    for (hit, msv, bias, vit, fwd) in batch_hits {
                        stats.n_past_msv += msv;
                        stats.n_past_bias += bias;
                        stats.n_past_vit += vit;
                        stats.n_past_fwd += fwd;
                        if let Some(hit) = hit {
                            all_hits.push(hit);
                        }
                    }
                    batch.clear();
                }
            }

            let mut th = TopHits::new();
            th.hits = all_hits;
            let n_targets = z as u64;
            let z = args.z_value.unwrap_or(n_targets as f64);
            final_z = Some(z);
            th.sort_by_sortkey();
            {
                let mut tmp_pli = Pipeline::new();
                configure_pipeline(&mut tmp_pli, &args);
                tmp_pli.do_biasfilter = !args.nobias;
                tmp_pli.do_null2 = !args.nonull2;
                tmp_pli.seed = args.seed;
                th.threshold(&tmp_pli, z, z);
                let domz = args.domz_value.unwrap_or(th.nreported as f64);
                if domz != z {
                    th.threshold(&tmp_pli, z, domz);
                }
                final_domz = Some(domz);
            }
            for hit in &mut th.hits {
                if hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED != 0
                    && !prev_included_names.iter().any(|prev| prev == &hit.name)
                {
                    hit.flags |= hmmer_pure_rs::tophits::P7_IS_NEW;
                } else {
                    hit.flags &= !hmmer_pure_rs::tophits::P7_IS_NEW;
                }
            }
            crate::subcmd::hmmsearch::write_standard_stdout_tables(
                &mut out,
                &th,
                &hmm.name,
                hmm.acc.as_deref().unwrap_or(""),
                hmm.m,
                hmm.cs.as_deref(),
                z,
                final_domz.unwrap_or(z),
                textw,
                args.show_acc,
                args.noali,
                true,
            );
            write_pipeline_stats(
                &mut out,
                hmm.m,
                n_targets,
                total_residues,
                th.nreported as u64,
                z,
                final_domz.unwrap_or(z),
                args.z_value.is_some(),
                args.domz_value.is_some(),
                &stats,
                args.f1,
                args.f2,
                args.f3,
            );
            writeln!(out).unwrap();
            let new_included_names: Vec<String> = th
                .hits
                .iter()
                .filter(|hit| hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED != 0)
                .map(|hit| hit.name.clone())
                .collect();
            let msa_nseq = th
                .hits
                .iter()
                .filter(|hit| hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED != 0)
                .map(|hit| hit.dcl.iter().filter(|dom| dom.is_included).count())
                .sum::<usize>()
                + 1;
            final_hits = Some(th);
            final_model_len = Some(hmm.m);
            prev_hmm = Some(hmm.clone());

            // Build the round MSA ONCE (matching C jackhmmer.c:683's single
            // `p7_tophits_Alignment` call per round). The same object is then
            // reused by `--chkali` below, by the next iteration's HMM build
            // (taken via `final_msa.take()` at the top of the loop), and by
            // `-A` after the loop ends. The previous round's MSA was either
            // consumed by this round's rebuild or dropped at the top of this
            // iteration, so the stash holds at most one MSA at a time.
            let msa = hmmer_pure_rs::tophits::included_alignment(
                final_hits.as_ref().unwrap(),
                &abc,
                hmm.m,
                Some((&query_sq, &query_tr)),
                &format!("{}-i{}", query_sq.name, iteration),
            )
            .map(|mut msa| {
                if !query_sq.desc.is_empty() {
                    msa.desc = Some(query_sq.desc.clone());
                }
                msa.author = Some("jackhmmer (HMMER 3.4)".to_string());
                msa
            });

            if let (Some(prefix), Some(ref msa)) = (&args.chkali, &msa) {
                write_msa_checkpoint(prefix, iteration, msa);
            }

            final_msa = msa;

            // Check convergence
            let n_new = new_included_names
                .iter()
                .filter(|name| !prev_included_names.iter().any(|prev| prev == *name))
                .count();
            writeln!(out, "@@ New targets included:   {}", n_new).unwrap();
            writeln!(
                out,
                "@@ New alignment includes: {} subseqs (was {}), including original query",
                msa_nseq, prev_msa_nseq
            )
            .unwrap();

            // Convergence test, faithful to jackhmmer.c serial_master() line 702:
            //   `if (nnew_targets == 0 && msa->nseq <= prv_msa_nseq)`.
            // C compares the new MSA's subsequence count against the previous
            // round's; it has no `iteration > 1` guard (round 1 with no
            // included hits converges immediately, msa->nseq==1<=prv==1).
            if n_new == 0 && msa_nseq <= prev_msa_nseq {
                writeln!(out, "@@").unwrap();
                writeln!(out, "@@ CONVERGED (in {} rounds). ", iteration).unwrap();
                writeln!(out, "@@").unwrap();
                writeln!(out).unwrap();
                break;
            }

            if iteration < args.max_iterations {
                writeln!(out, "@@ Continuing to next round.").unwrap();
                writeln!(out).unwrap();
            }

            prev_included_names = new_included_names;
            prev_msa_nseq = msa_nseq;
        }

        if let Some(ref mut f) = tblout_file {
            if let (Some(ref th), Some(z)) = (&final_hits, final_z) {
                let show_header = !wrote_tblout_header;
                crate::subcmd::hmmsearch::write_tblout(
                    f,
                    &query_sq.name,
                    Some(&query_sq.acc),
                    th,
                    z,
                    show_header,
                );
                wrote_tblout_header = true;
            }
        }
        if let Some(ref mut f) = domtblout_file {
            if let (Some(ref th), Some(z), Some(domz), Some(qlen)) =
                (&final_hits, final_z, final_domz, final_model_len)
            {
                let show_header = !wrote_domtblout_header;
                crate::subcmd::hmmsearch::write_domtblout(
                    f,
                    &query_sq.name,
                    Some(&query_sq.acc),
                    qlen,
                    th,
                    z,
                    domz,
                    show_header,
                );
                wrote_domtblout_header = true;
            }
        }
        if let Some(ref mut f) = ali_file {
            // Reuse the MSA already built at the end of the final iteration
            // (matching C jackhmmer.c:723's reuse of the single per-round
            // `msa`). When no rounds ran or no hits were found, `final_msa`
            // is None and no `-A` block is emitted.
            if let Some(ref msa) = final_msa {
                write_stockholm_msa(f, msa);
                writeln!(
                    out,
                    "# Alignment of {} hits satisfying inclusion thresholds saved to: {}",
                    msa.nseq,
                    args.ali_outfile.as_ref().unwrap().display()
                )
                .unwrap();
            }
            f.flush().unwrap();
        }

        writeln!(out, "//").unwrap();
    }

    if let Some(ref mut f) = tblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "jackhmmer",
            "SEARCH",
            &args.seqfile,
            &args.seqdb,
            &table_cmdline,
        );
        f.flush().unwrap();
    }
    if let Some(ref mut f) = domtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "jackhmmer",
            "SEARCH",
            &args.seqfile,
            &args.seqdb,
            &table_cmdline,
        );
        f.flush().unwrap();
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

fn normalize_jackhmmer_table_cmdline(cmdline: &str) -> String {
    let mut tokens = cmdline.split_whitespace();
    match (tokens.next(), tokens.next()) {
        (Some(_wrapper), Some("jackhmmer")) => {
            let rest = tokens.collect::<Vec<_>>();
            if rest.is_empty() {
                "jackhmmer".to_string()
            } else {
                format!("jackhmmer {}", rest.join(" "))
            }
        }
        _ => cmdline.to_string(),
    }
}

fn write_jackhmmer_option_header(out: &mut dyn Write, args: &Args, cmdline: &str) {
    if command_line_has_option(cmdline, "-E") {
        writeln!(
            out,
            "# sequence reporting threshold:    E-value <= {}",
            hmmer_pure_rs::output::fmt_g(args.e_value)
        )
        .unwrap();
    }
    if let Some(score) = args.score_threshold {
        writeln!(
            out,
            // C jackhmmer prints "<=" here, despite score thresholds being minimums.
            "# sequence reporting threshold:    score <= {}",
            hmmer_pure_rs::output::fmt_g(score as f64)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--domE") {
        writeln!(
            out,
            "# domain reporting threshold:      E-value <= {}",
            hmmer_pure_rs::output::fmt_g(args.dom_e)
        )
        .unwrap();
    }
    if let Some(score) = args.dom_t {
        writeln!(
            out,
            // C jackhmmer prints "<=" here, despite score thresholds being minimums.
            "# domain reporting threshold:      score <= {}",
            hmmer_pure_rs::output::fmt_g(score as f64)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--incE") {
        writeln!(
            out,
            "# sequence inclusion threshold:    E-value <= {}",
            hmmer_pure_rs::output::fmt_g(args.inc_e)
        )
        .unwrap();
    }
    if let Some(score) = args.inc_t {
        writeln!(
            out,
            "# sequence inclusion threshold:    score >= {}",
            hmmer_pure_rs::output::fmt_g(score as f64)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--incdomE") {
        writeln!(
            out,
            "# domain inclusion threshold:      E-value <= {}",
            hmmer_pure_rs::output::fmt_g(args.incdom_e)
        )
        .unwrap();
    }
    if let Some(score) = args.incdom_t {
        writeln!(
            out,
            "# domain inclusion threshold:      score >= {}",
            hmmer_pure_rs::output::fmt_g(score as f64)
        )
        .unwrap();
    }
    if args.max {
        writeln!(
            out,
            "# Max sensitivity mode:            on [all heuristic filters off]"
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--F1") {
        writeln!(
            out,
            "# MSV filter P threshold:       <= {}",
            hmmer_pure_rs::output::fmt_g(args.f1)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--F2") {
        writeln!(
            out,
            "# Vit filter P threshold:       <= {}",
            hmmer_pure_rs::output::fmt_g(args.f2)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--F3") {
        writeln!(
            out,
            "# Fwd filter P threshold:       <= {}",
            hmmer_pure_rs::output::fmt_g(args.f3)
        )
        .unwrap();
    }
    if args.nobias {
        writeln!(out, "# biased composition HMM filter:   off").unwrap();
    }
    if args.nonull2 {
        writeln!(out, "# null2 bias corrections:          off").unwrap();
    }
    if let Some(z) = args.z_value {
        writeln!(
            out,
            "# sequence search space set to:    {}",
            hmmer_pure_rs::output::fmt_fixed0(z)
        )
        .unwrap();
    }
    if let Some(domz) = args.domz_value {
        writeln!(
            out,
            "# domain search space set to:      {}",
            hmmer_pure_rs::output::fmt_fixed0(domz)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--seed") {
        if args.seed == 0 {
            writeln!(out, "# random number seed:              one-time arbitrary").unwrap();
        } else {
            writeln!(out, "# random number seed set to:       {}", args.seed).unwrap();
        }
    }
}

fn command_line_has_option(cmdline: &str, option: &str) -> bool {
    let compact_short = option
        .strip_prefix('-')
        .filter(|rest| !rest.starts_with('-') && rest.chars().count() == 1)
        .map(|_| option);
    cmdline.split_whitespace().any(|token| {
        token == option
            || token.starts_with(&format!("{option}="))
            || compact_short
                .is_some_and(|short| token.starts_with(short) && token.len() > short.len())
    })
}

fn validate_sequence_format(label: &str, format: Option<&str>) {
    if let Some(format) = format {
        if SequenceFormat::from_name(format).is_none() {
            eprintln!("{label}={format} is not a recognized input sequence file format");
            std::process::exit(1);
        }
    }
}

fn open_sequence_file(
    path: &std::path::Path,
    abc: &Alphabet,
    format: Option<&str>,
) -> hmmer_pure_rs::errors::HmmerResult<sequence::SeqFile<Box<dyn std::io::Read>>> {
    if let Some(format) = format {
        sequence::open_seq_file_with_format(path, abc, SequenceFormat::from_name(format).unwrap())
    } else {
        sequence::open_seq_file(path, abc)
    }
}

#[derive(Default)]
struct PipelineStats {
    n_past_msv: u64,
    n_past_bias: u64,
    n_past_vit: u64,
    n_past_fwd: u64,
}

fn write_pipeline_stats<W: Write>(
    out: &mut W,
    model_len: usize,
    n_targets: u64,
    total_residues: u64,
    _nreported: u64,
    z: f64,
    domz: f64,
    z_from_option: bool,
    domz_from_option: bool,
    stats: &PipelineStats,
    f1: f64,
    f2: f64,
    f3: f64,
) {
    let expected_msv = (f1 * n_targets as f64).max(0.0);
    let expected_vit = (f2 * n_targets as f64).max(0.0);
    let expected_fwd = (f3 * n_targets as f64).max(0.0);
    let frac = |n: u64| {
        if n_targets > 0 {
            n as f64 / n_targets as f64
        } else {
            0.0
        }
    };

    writeln!(out, "Internal pipeline statistics summary:").unwrap();
    writeln!(out, "-------------------------------------").unwrap();
    writeln!(
        out,
        "Query model(s):                  {:>11}  ({} nodes)",
        1, model_len
    )
    .unwrap();
    writeln!(
        out,
        "Target sequences:                {:>11}  ({} residues searched)",
        n_targets, total_residues
    )
    .unwrap();
    writeln!(
        out,
        "Passed MSV filter:               {:>11}  ({}); expected {} ({})",
        stats.n_past_msv,
        hmmer_pure_rs::output::fmt_g(frac(stats.n_past_msv)),
        hmmer_pure_rs::output::fmt_fixed1(expected_msv),
        hmmer_pure_rs::output::fmt_g(f1)
    )
    .unwrap();
    writeln!(
        out,
        "Passed bias filter:              {:>11}  ({}); expected {} ({})",
        stats.n_past_bias,
        hmmer_pure_rs::output::fmt_g(frac(stats.n_past_bias)),
        hmmer_pure_rs::output::fmt_fixed1(expected_msv),
        hmmer_pure_rs::output::fmt_g(f1)
    )
    .unwrap();
    writeln!(
        out,
        "Passed Vit filter:               {:>11}  ({}); expected {} ({})",
        stats.n_past_vit,
        hmmer_pure_rs::output::fmt_g(frac(stats.n_past_vit)),
        hmmer_pure_rs::output::fmt_fixed1(expected_vit),
        hmmer_pure_rs::output::fmt_g(f2)
    )
    .unwrap();
    writeln!(
        out,
        "Passed Fwd filter:               {:>11}  ({}); expected {} ({})",
        stats.n_past_fwd,
        hmmer_pure_rs::output::fmt_g(frac(stats.n_past_fwd)),
        hmmer_pure_rs::output::fmt_fixed1(expected_fwd),
        hmmer_pure_rs::output::fmt_g(f3)
    )
    .unwrap();
    writeln!(
        out,
        "Initial search space (Z):        {}  {}",
        hmmer_pure_rs::output::fmt_width11_0(z),
        if z_from_option {
            "[as set by --Z on cmdline]"
        } else {
            "[actual number of targets]"
        }
    )
    .unwrap();
    writeln!(
        out,
        "Domain search space  (domZ):     {}  {}",
        hmmer_pure_rs::output::fmt_width11_0(domz),
        if domz_from_option {
            "[as set by --domZ on cmdline]"
        } else {
            "[number of targets reported over threshold]"
        }
    )
    .unwrap();
}

/// Build a degenerate trace for the round-1 single-sequence query that simply
/// walks B -> M1..Mn -> E. Used so the query can be re-included verbatim when
/// constructing the included-hit MSA for the next round.
fn exact_match_query_trace(model_len: usize) -> Trace {
    let mut tr = Trace::new();
    tr.append(State::B, 0, 0);
    for i in 1..=model_len {
        tr.append(State::M, i, i);
    }
    tr.append(State::E, 0, 0);
    tr.m = model_len;
    tr.l = model_len;
    tr
}

/// Write the HMM produced at `iteration` to `<prefix>-<iteration>.hmm`.
fn write_hmm_checkpoint(prefix: &PathBuf, iteration: usize, hmm: &hmmer_pure_rs::hmm::Hmm) {
    let path = checkpoint_path(prefix, iteration, "hmm");
    let mut file = std::fs::File::create(&path).unwrap_or_else(|e| {
        eprintln!("Error creating HMM checkpoint {}: {}", path.display(), e);
        std::process::exit(1);
    });
    hmmer_pure_rs::hmmfile::write_hmm(&mut file, hmm).unwrap_or_else(|e| {
        eprintln!("Error writing HMM checkpoint {}: {}", path.display(), e);
        std::process::exit(1);
    });
}

/// Write the included-hit alignment for `iteration` to `<prefix>-<iteration>.sto`.
fn write_msa_checkpoint(prefix: &PathBuf, iteration: usize, msa: &Msa) {
    let path = checkpoint_path(prefix, iteration, "sto");
    let mut file = std::fs::File::create(&path).unwrap_or_else(|e| {
        eprintln!(
            "Error creating alignment checkpoint {}: {}",
            path.display(),
            e
        );
        std::process::exit(1);
    });
    write_stockholm_msa(&mut file, msa);
}

/// Compose a checkpoint filename of the form `<prefix>-<iteration>.<ext>`.
fn checkpoint_path(prefix: &PathBuf, iteration: usize, ext: &str) -> PathBuf {
    let mut os = prefix.as_os_str().to_os_string();
    os.push(format!("-{}.{}", iteration, ext));
    PathBuf::from(os)
}

/// Render an in-memory `Msa` as a Stockholm 1.0 file, emitting `#=GF`,
/// per-sequence `#=GS` descriptions, residue rows, per-row `#=GR ... PP`,
/// and `#=GC PP_cons`/`#=GC RF` consensus annotation.
///
/// Column widths are computed exactly as Easel's `stockholm_write()`
/// (hmmer/easel/esl_msafile_stockholm.c:1069): the left margin is
/// `max(maxname+1, maxgc+6, maxname+maxgr+7)`, `#=GF` tags pad to `maxgf`
/// (>=2), `#=GS` name fields to `maxname`, residue-row names to `margin-1`,
/// `#=GR` to `maxname` + a `margin-maxname-7` tag field, and `#=GC` to
/// `margin-6`. This is written as a single unwrapped block, i.e. equivalent to
/// Easel's Pfam format (`cpl == alen`); Easel's default Stockholm wraps at 200
/// aligned residues per block, so for alignments wider than 200 columns the
/// block layout would differ (the per-line annotation is identical).
///
/// When `fold_inserts_to_match` is `true`, residue rows are uppercased and
/// insert-gap '.' characters are rewritten to '-'. This reproduces how
/// `jackhmmer` renders its `-A`/checkpoint MSAs: jackhmmer's MSA carries the
/// real Viterbi-traced query (`p7_tophits_Alignment(th, abc, &qsq, &qtr, 1,
/// ...)`), whose presence collapses the round's insert columns into the
/// model's consensus (match) columns, so C emits '-'/uppercase there. phmmer
/// and hmmsearch build the same MSA without the query (`inc_n == 0`) and keep
/// the raw insert representation ('.'/lowercase), so they pass `false`.
pub(crate) fn write_stockholm_msa(out: &mut dyn Write, msa: &Msa) {
    write_stockholm_msa_inner(out, msa, true)
}

/// Same Stockholm writer, but emitting the raw `p7_tophits_Alignment` text
/// (insert columns kept as '.'/lowercase). Used by phmmer and hmmsearch `-A`,
/// which build the MSA without the query (`inc_n == 0`) and so match C's
/// `esl_msafile_Write` byte-for-byte without any insert folding.
pub(crate) fn write_tophits_alignment_msa(out: &mut dyn Write, msa: &Msa) {
    write_stockholm_msa_inner(out, msa, false)
}

fn write_stockholm_msa_inner(out: &mut dyn Write, msa: &Msa, fold_inserts_to_match: bool) {
    let maxname = msa.sqname.iter().map(|name| name.len()).max().unwrap_or(0);
    // maxgf: longest #=GF tag. We only ever emit ID/AC/DE/AU, all 2 chars,
    // and Easel floors maxgf at 2.
    let maxgf = 2usize;
    // maxgc: PP_cons => 7, RF => 2.
    let mut maxgc = 0usize;
    if msa.pp_cons.is_some() {
        maxgc = maxgc.max(7);
    }
    if msa.rf.is_some() {
        maxgc = maxgc.max(2);
    }
    // maxgr: PP rows => 2.
    let maxgr = if msa.pp.iter().any(|pp| pp.is_some()) {
        2usize
    } else {
        0
    };
    let mut margin = maxname + 1;
    if maxgc > 0 && maxgc + 6 > margin {
        margin = maxgc + 6;
    }
    if maxgr > 0 && maxname + maxgr + 7 > margin {
        margin = maxname + maxgr + 7;
    }

    writeln!(out, "# STOCKHOLM 1.0").unwrap();
    if !msa.name.is_empty() {
        writeln!(out, "#=GF {:<maxgf$} {}", "ID", msa.name).unwrap();
    }
    if let Some(acc) = &msa.acc {
        if !acc.is_empty() {
            writeln!(out, "#=GF {:<maxgf$} {}", "AC", acc).unwrap();
        }
    }
    if let Some(desc) = &msa.desc {
        if !desc.is_empty() {
            writeln!(out, "#=GF {:<maxgf$} {}", "DE", desc).unwrap();
        }
    }
    if let Some(author) = &msa.author {
        if !author.is_empty() {
            writeln!(out, "#=GF {:<maxgf$} {}", "AU", author).unwrap();
        }
    }
    writeln!(out).unwrap();

    if !msa.sqdesc.iter().all(|desc| desc.is_empty()) {
        for (name, desc) in msa.sqname.iter().zip(msa.sqdesc.iter()) {
            if !desc.is_empty() {
                writeln!(out, "#=GS {:<maxname$} DE {}", name, desc).unwrap();
            }
        }
        writeln!(out).unwrap();
    }

    let gr_tag_width = margin.saturating_sub(maxname + 7);
    for ((name, row), pp) in msa.sqname.iter().zip(msa.aseq.iter()).zip(msa.pp.iter()) {
        // The builder (`tophits::included_alignment`) emits Easel's digital-MSA
        // text form, where match-column gaps are '-', insert-column gaps are
        // '.', and inserted residues are lowercase. C's `esl_msafile_Write`
        // textizes those bytes unchanged (phmmer/hmmsearch path). jackhmmer's
        // MSA instead collapses inserts into the consensus, so we optionally
        // uppercase and fold '.'->'-' to match its C rendering.
        let rendered: std::borrow::Cow<'_, [u8]> = if fold_inserts_to_match {
            std::borrow::Cow::Owned(
                row.iter()
                    .map(|&ch| match ch {
                        b'.' => b'-',
                        b'a'..=b'z' => ch.to_ascii_uppercase(),
                        _ => ch,
                    })
                    .collect(),
            )
        } else {
            std::borrow::Cow::Borrowed(row.as_slice())
        };
        writeln!(
            out,
            "{:<width$} {}",
            name,
            String::from_utf8_lossy(&rendered),
            width = margin - 1
        )
        .unwrap();
        if let Some(pp) = pp {
            writeln!(
                out,
                "#=GR {:<maxname$} {:<gr_tag_width$} {}",
                name,
                "PP",
                String::from_utf8_lossy(pp),
            )
            .unwrap();
        }
    }
    if let Some(pp_cons) = &msa.pp_cons {
        writeln!(
            out,
            "#=GC {:<width$} {}",
            "PP_cons",
            String::from_utf8_lossy(pp_cons),
            width = margin - 6
        )
        .unwrap();
    }
    if let Some(rf) = &msa.rf {
        writeln!(
            out,
            "#=GC {:<width$} {}",
            "RF",
            String::from_utf8_lossy(rf),
            width = margin - 6
        )
        .unwrap();
    }
    writeln!(out, "//").unwrap();
}

fn configure_pipeline(pli: &mut Pipeline, args: &Args) {
    pli.e_value_threshold = args.e_value;
    pli.dom_e_value_threshold = args.dom_e;
    pli.inc_e = args.inc_e;
    pli.inc_dome = args.incdom_e;
    pli.do_max = args.max;
    pli.f1 = args.f1;
    pli.f2 = args.f2;
    pli.f3 = args.f3;

    if let Some(t) = args.score_threshold {
        pli.t = Some(t);
        pli.by_e = false;
    }
    if let Some(t) = args.dom_t {
        pli.dom_t = Some(t);
        pli.dom_by_e = false;
    }
    if let Some(t) = args.inc_t {
        pli.inc_t = Some(t);
        pli.inc_by_e = false;
    }
    if let Some(t) = args.incdom_t {
        pli.inc_dom_t = Some(t);
        pli.incdom_by_e = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jackhmmer_parses_c_output_file_option() {
        let args =
            Args::try_parse_from(["jackhmmer", "-o", "out.txt", "query.fa", "targets.fa"]).unwrap();
        assert_eq!(args.output, Some(PathBuf::from("out.txt")));
        assert_eq!(args.seqfile, PathBuf::from("query.fa"));
        assert_eq!(args.seqdb, PathBuf::from("targets.fa"));
    }

    #[test]
    fn jackhmmer_parses_c_alignment_output_option() {
        let args = Args::try_parse_from(["jackhmmer", "-A", "hits.sto", "query.fa", "targets.fa"])
            .unwrap();
        assert_eq!(args.ali_outfile, Some(PathBuf::from("hits.sto")));
        assert_eq!(args.seqfile, PathBuf::from("query.fa"));
        assert_eq!(args.seqdb, PathBuf::from("targets.fa"));
    }

    #[test]
    fn jackhmmer_parses_c_threshold_and_acceleration_options() {
        let args = Args::try_parse_from([
            "jackhmmer",
            "-T",
            "25",
            "--domT",
            "3",
            "--incT",
            "20",
            "--incdomT",
            "4",
            "-Z",
            "99",
            "--domZ",
            "7",
            "--seed",
            "5",
            "query.fa",
            "targets.fa",
        ])
        .unwrap();
        assert_eq!(args.score_threshold, Some(25.0));
        assert_eq!(args.dom_t, Some(3.0));
        assert_eq!(args.inc_t, Some(20.0));
        assert_eq!(args.incdom_t, Some(4.0));
        assert_eq!(args.z_value, Some(99.0));
        assert_eq!(args.domz_value, Some(7.0));
        assert_eq!(args.seed, 5);
        assert_eq!(args.em_l, 200);

        let args = Args::try_parse_from([
            "jackhmmer",
            "--F1",
            "0.1",
            "--F2",
            "0.2",
            "--F3",
            "0.3",
            "query.fa",
            "targets.fa",
        ])
        .unwrap();
        assert_eq!(args.f1, 0.1);
        assert_eq!(args.f2, 0.2);
        assert_eq!(args.f3, 0.3);
    }

    #[test]
    fn jackhmmer_accepts_negative_space_separated_f_values() {
        // C --F1/--F2/--F3 have no range; the space-separated negative form is
        // accepted. allow_hyphen_values matches that instead of rejecting
        // "-0.5" as an unknown flag.
        let args = Args::try_parse_from([
            "jackhmmer", "--F1", "-0.5", "--F2", "-1e-3", "--F3", "-2", "query.fa", "targets.fa",
        ])
        .unwrap();
        assert_eq!(args.f1, -0.5);
        assert_eq!(args.f2, -1e-3);
        assert_eq!(args.f3, -2.0);
    }

    #[test]
    fn jackhmmer_parses_single_sequence_calibration_options() {
        let args = Args::try_parse_from([
            "jackhmmer",
            "--seed",
            "7",
            "--EmL",
            "101",
            "--EmN",
            "20",
            "--EvL",
            "102",
            "--EvN",
            "21",
            "--EfL",
            "103",
            "--EfN",
            "22",
            "--Eft",
            "0.03",
            "query.fa",
            "targets.fa",
        ])
        .unwrap();

        assert_eq!(args.seed, 7);
        assert_eq!(args.em_l, 101);
        assert_eq!(args.em_n, 20);
        assert_eq!(args.ev_l, 102);
        assert_eq!(args.ev_n, 21);
        assert_eq!(args.ef_l, 103);
        assert_eq!(args.ef_n, 22);
        assert_eq!(args.eft, 0.03);
    }
}
