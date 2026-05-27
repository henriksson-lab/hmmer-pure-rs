//! phmmer — search a protein sequence against a protein database.
//! Builds a single-sequence HMM and searches with the hmmsearch pipeline.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::calibrate::CalibrationConfig;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::seqmodel;
use hmmer_pure_rs::sequence::{self, Sequence, SequenceFormat};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::{Hit, TopHits};

#[derive(Parser)]
#[command(
    name = "phmmer",
    about = "Search a protein sequence against a protein database"
)]
struct Args {
    /// Query sequence file (FASTA)
    seqfile: PathBuf,
    /// Target sequence database (FASTA)
    seqdb: PathBuf,

    /// Direct output to file, not stdout
    #[arg(short = 'o')]
    output: Option<PathBuf>,

    /// Assert query sequence file format
    #[arg(long = "qformat")]
    qformat: Option<String>,

    /// Assert target sequence file format
    #[arg(long = "tformat")]
    tformat: Option<String>,

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
        default_value = "0.01",
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
        default_value = "0.01",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_dom_t"
    )]
    inc_dome: f64,

    /// Include domains >= this score threshold
    #[arg(long = "incdomT", conflicts_with = "inc_dome", allow_hyphen_values = true)]
    inc_dom_t: Option<f64>,

    /// Hidden C-compatible model gathering cutoff flag
    #[arg(
        long = "cut_ga",
        hide = true,
        conflicts_with_all = [
            "cut_tc",
            "cut_nc",
            "e_value",
            "score_threshold",
            "dom_e",
            "dom_t",
            "inc_e",
            "inc_t",
            "inc_dome",
            "inc_dom_t"
        ]
    )]
    cut_ga: bool,

    /// Hidden C-compatible model noise cutoff flag
    #[arg(
        long = "cut_nc",
        hide = true,
        conflicts_with_all = [
            "cut_ga",
            "cut_tc",
            "e_value",
            "score_threshold",
            "dom_e",
            "dom_t",
            "inc_e",
            "inc_t",
            "inc_dome",
            "inc_dom_t"
        ]
    )]
    cut_nc: bool,

    /// Hidden C-compatible model trusted cutoff flag
    #[arg(
        long = "cut_tc",
        hide = true,
        conflicts_with_all = [
            "cut_ga",
            "cut_nc",
            "e_value",
            "score_threshold",
            "dom_e",
            "dom_t",
            "inc_e",
            "inc_t",
            "inc_dome",
            "inc_dom_t"
        ]
    )]
    cut_tc: bool,

    /// Gap open probability
    #[arg(long = "popen", default_value = "0.02", value_parser = parse_popen)]
    popen: f32,

    /// Gap extend probability
    #[arg(long = "pextend", default_value = "0.4", value_parser = parse_pextend)]
    pextend: f32,

    /// Substitution score matrix choice
    #[arg(long = "mx", default_value = "BLOSUM62", conflicts_with = "mxfile")]
    matrix: String,

    /// Read substitution score matrix from file
    #[arg(long = "mxfile")]
    mxfile: Option<PathBuf>,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Turn all heuristic filters off
    #[arg(long = "max", conflicts_with_all = ["f1", "f2", "f3", "nobias"])]
    max: bool,

    /// MSV threshold
    #[arg(long = "F1", default_value = "0.02", allow_hyphen_values = true)]
    f1: f64,

    /// Viterbi threshold
    #[arg(long = "F2", default_value = "0.001", allow_hyphen_values = true)]
    f2: f64,

    /// Forward threshold
    #[arg(long = "F3", default_value = "1e-5", allow_hyphen_values = true)]
    f3: f64,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    nobias: bool,

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    nonull2: bool,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    domtblout: Option<PathBuf>,

    /// Save Pfam-style table of hits and domains
    #[arg(long = "pfamtblout")]
    pfamtblout: Option<PathBuf>,

    /// Save multiple alignment of hits to file
    #[arg(short = 'A')]
    ali_outfile: Option<PathBuf>,

    /// Omit alignments from the main output
    #[arg(long = "noali")]
    noali: bool,

    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    show_acc: bool,

    /// Do not line-wrap the main text output
    #[arg(long = "notextw", conflicts_with = "textw")]
    notextw: bool,

    /// Set the target line width for the main text output
    #[arg(long = "textw", default_value = "120", value_parser = parse_textw)]
    textw: usize,

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    domz_value: Option<f64>,

    /// Set RNG seed
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Start restricted target search at this sequence key
    #[arg(long = "restrictdb_stkey", hide = true)]
    restrictdb_stkey: Option<String>,

    /// Search only this many target sequences from the restricted start
    #[arg(long = "restrictdb_n", value_parser = parse_positive_usize, hide = true)]
    restrictdb_n: Option<usize>,

    /// SSI index file for C-compatible restricted database options
    #[arg(long = "ssifile", hide = true)]
    ssifile: Option<PathBuf>,

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

/// Entry point for `phmmer`: search each query protein against a protein
/// database via a single-sequence HMM (no iteration, no MSA).
///
/// For each query: builds a length-1 HMM with `seqmodel::build_single_seq_hmm`
/// using the supplied gap-open/extend probabilities, configures local profile,
/// runs the standard pipeline against the full target DB in parallel (rayon),
/// then sorts/thresholds and prints the canonical per-sequence tabular block.
/// Optional `--tblout` is delegated to `hmmsearch::write_tblout`. Corresponds
/// to `serial_master()` in hmmer/src/phmmer.c.
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
    validate_sequence_format("phmmer --qformat", args.qformat.as_deref());
    validate_sequence_format("phmmer --tformat", args.tformat.as_deref());
    let score_matrix = if let Some(mxfile) = args.mxfile.as_ref() {
        seqmodel::ScoreMatrix::from_file(mxfile)
    } else {
        seqmodel::ScoreMatrix::builtin(&args.matrix)
    }
    .unwrap_or_else(|e| {
        eprintln!("Error: phmmer {e}");
        std::process::exit(1);
    });

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
    if args.seqfile == PathBuf::from("-") && args.seqdb == PathBuf::from("-") {
        eprintln!("Error: Either <seqfile> or <seqdb> may be '-' but not both");
        std::process::exit(1);
    }
    if args.seqdb == PathBuf::from("-") && count_query_sequences(&args.seqfile, &abc) > 1 {
        eprintln!(
            "Error: target sequence file - isn't rewindable; can't search it with multiple queries"
        );
        std::process::exit(1);
    }

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
    let mut pfamtblout_file = args.pfamtblout.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open pfam-style tabular output file {path} for writing",
        )
    });
    let mut ali_file = args.ali_outfile.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open alignment output file {path} for writing",
        )
    });
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

    writeln!(
        out,
        "# phmmer :: search a protein sequence against a protein database"
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
    writeln!(
        out,
        "# target sequence database:        {}",
        args.seqdb.display()
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
    if let Some(format) = args.qformat.as_deref() {
        writeln!(out, "# query <seqfile> format asserted: {format}").unwrap();
    }
    if let Some(format) = args.tformat.as_deref() {
        writeln!(out, "# target <seqdb> format asserted:  {format}").unwrap();
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
    if let Some(pfamtblout) = &args.pfamtblout {
        writeln!(
            out,
            "# pfam-style tabular hit output:   {}",
            pfamtblout.display()
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
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    if args.show_acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
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
    write_phmmer_option_header(&mut out, &args, &cmdline);
    let textw = if args.notextw { 0 } else { args.textw };
    if args.notextw {
        writeln!(out, "# max ASCII text line length:      unlimited").unwrap();
    } else if args.textw != 120 {
        writeln!(out, "# max ASCII text line length:      {}", args.textw).unwrap();
    }
    if cpu_was_requested {
        writeln!(out, "# number of worker threads:        {}", args.cpu).unwrap();
    }
    if let Some(stkey) = &args.restrictdb_stkey {
        writeln!(out, "# Restrict db to start at seq key: {}", stkey).unwrap();
    }
    if let Some(n) = args.restrictdb_n {
        writeln!(out, "# Restrict db to # target seqs:    {}", n).unwrap();
    }
    if let Some(ssifile) = &args.ssifile {
        writeln!(
            out,
            "# Override ssi file to:            {}",
            ssifile.display()
        )
        .unwrap();
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
    out.flush().unwrap_or_else(|e| {
        eprintln!("Error writing output: {}", e);
        std::process::exit(1);
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

    // Read query sequences
    let mut query_sqf = open_sequence_file(&args.seqfile, &abc, args.qformat.as_deref())
        .unwrap_or_else(|e| {
            eprintln!("Error opening query file: {}", e);
            std::process::exit(1);
        });

    let mut query_sq = Sequence::new();
    let mut n_queries = 0usize;
    while query_sqf.read(&mut query_sq).unwrap_or_else(|e| {
        eprintln!("Error reading query file: {}", e);
        std::process::exit(1);
    }) {
        n_queries += 1;
        // Build HMM from query sequence
        let hmm = seqmodel::build_single_seq_hmm_with_matrix_and_calibration(
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
            eprintln!("Error: phmmer failed to set single query seq score system: {e}");
            std::process::exit(1);
        });

        let mut local_bg = bg.clone();
        local_bg.set_filter(hmm.m, &hmm.compo);

        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(&hmm, &local_bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        writeln!(out, "Query:       {}  [L={}]", query_sq.name, query_sq.n).unwrap();
        if !query_sq.acc.is_empty() {
            writeln!(out, "Accession:   {}", query_sq.acc).unwrap();
        }
        if !query_sq.desc.is_empty() {
            writeln!(out, "Description: {}", query_sq.desc).unwrap();
        }
        out.flush().unwrap_or_else(|e| {
            eprintln!("Error writing output: {}", e);
            std::process::exit(1);
        });

        // Read target sequences
        let mut sequences = Vec::new();
        let mut total_residues: u64 = 0; // used for statistics
        {
            let mut sqf = open_sequence_file(&args.seqdb, &abc, args.tformat.as_deref())
                .unwrap_or_else(|e| {
                    eprintln!("Error opening target database: {}", e);
                    std::process::exit(1);
                });
            let mut restrict_started = args.restrictdb_stkey.is_none();
            if let Some(stkey) = args.restrictdb_stkey.as_deref() {
                sqf = crate::subcmd::hmmsearch::open_restricted_target_seq_file(
                    &args.seqdb,
                    &abc,
                    args.tformat.as_deref(),
                    stkey,
                    args.ssifile.as_deref(),
                )
                .unwrap_or_else(|e| {
                    eprintln!("Error opening restricted target database: {}", e);
                    std::process::exit(1);
                });
                restrict_started = true;
            }
            let mut sq = Sequence::new();
            let mut restrict_seen = 0usize;
            while sqf.read(&mut sq).unwrap_or_else(|e| {
                eprintln!("Error reading target database: {}", e);
                std::process::exit(1);
            }) {
                if !restrict_started {
                    if args
                        .restrictdb_stkey
                        .as_deref()
                        .is_some_and(|key| sq.name == key)
                    {
                        restrict_started = true;
                    } else {
                        sq.reuse();
                        continue;
                    }
                }
                if args
                    .restrictdb_n
                    .is_some_and(|limit| restrict_seen >= limit)
                {
                    break;
                }
                restrict_seen += 1;
                total_residues += sq.n as u64;
                sequences.push(sq.clone());
                sq.reuse();
            }
        }
        if sequences.is_empty() {
            eprintln!("Error: no sequences found in {}", args.seqdb.display());
            std::process::exit(1);
        }

        // Search in parallel
        use rayon::prelude::*;
        let scored: Vec<(Option<Hit>, u64, u64, u64, u64)> = sequences
            .par_iter()
            .map(|sq| {
                hmmer_pure_rs::util::simd_env::init();
                let mut lb = local_bg.clone();
                let mut lgm = gm.clone();
                let mut lom = om.clone();
                let mut lpli = Pipeline::new();
                lpli.f1 = args.f1;
                lpli.f2 = args.f2;
                lpli.f3 = args.f3;
                lpli.do_max = args.max;
                lpli.do_biasfilter = !args.nobias;
                lpli.do_null2 = !args.nonull2;
                let wants_alignment_file = args.ali_outfile.is_some();
                lpli.do_alignment = !args.noali
                    || args.domtblout.is_some()
                    || args.pfamtblout.is_some()
                    || wants_alignment_file;
                lpli.do_alignment_display = !args.noali || wants_alignment_file;
                lpli.seed = args.seed;
                lpli.new_model(&lgm);
                configure_thresholds(&mut lpli, &args);

                lb.set_length(sq.n);

                let mut lth = TopHits::new();
                let hit = if lpli.run(&mut lgm, &mut lom, &lb, &hmm, sq, &mut lth) {
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
            })
            .collect();

        let mut th = TopHits::new();
        let mut stats = PipelineStats::default();
        th.hits = scored
            .into_iter()
            .filter_map(|(hit, msv, bias, vit, fwd)| {
                stats.n_past_msv += msv;
                stats.n_past_bias += bias;
                stats.n_past_vit += vit;
                stats.n_past_fwd += fwd;
                hit
            })
            .collect();
        let z = args.z_value.unwrap_or(sequences.len() as f64);
        th.sort_by_sortkey();
        {
            let mut tmp_pli = Pipeline::new();
            configure_thresholds(&mut tmp_pli, &args);
            tmp_pli.do_biasfilter = !args.nobias;
            tmp_pli.do_null2 = !args.nonull2;
            th.threshold(&tmp_pli, z, z);
            let domz = args.domz_value.unwrap_or(th.nreported as f64);
            if domz != z {
                th.threshold(&tmp_pli, z, domz);
            }
        }
        let domz = args.domz_value.unwrap_or(th.nreported as f64);

        crate::subcmd::hmmsearch::write_standard_stdout_tables(
            &mut out,
            &th,
            &hmm.name,
            hmm.acc.as_deref().unwrap_or(""),
            hmm.m,
            hmm.cs.as_deref(),
            z,
            domz,
            textw,
            args.show_acc,
            args.noali,
            false,
        );
        write_pipeline_stats(
            &mut out,
            hmm.m,
            sequences.len() as u64,
            total_residues,
            th.nreported as u64,
            z,
            domz,
            args.z_value.is_some(),
            args.domz_value.is_some(),
            &stats,
            args.f1,
            args.f2,
            args.f3,
        );
        writeln!(out, "//").unwrap();

        if let Some(ref mut f) = tblout_file {
            crate::subcmd::hmmsearch::write_tblout(
                f,
                &query_sq.name,
                Some(&query_sq.acc),
                &th,
                z,
                n_queries == 1,
            );
        }
        if let Some(ref mut f) = domtblout_file {
            crate::subcmd::hmmsearch::write_domtblout(
                f,
                &query_sq.name,
                Some(&query_sq.acc),
                query_sq.n,
                &th,
                z,
                domz,
                n_queries == 1,
            );
        }
        if let Some(ref mut f) = pfamtblout_file {
            crate::subcmd::hmmsearch::write_pfamtblout(
                f,
                &query_sq.name,
                Some(&query_sq.acc),
                query_sq.n,
                &th,
                z,
                domz,
            );
        }
        if let Some(ref mut f) = ali_file {
            // C phmmer.c:620-638: build the included-domain MSA (named after the
            // model/query, with the query's acc/desc when present), write it,
            // then echo the confirmation line carrying the MSA's nseq.
            let nseq = crate::subcmd::hmmsearch::write_ali_output(
                f,
                &abc,
                hmm.m,
                &hmm.name,
                Some(query_sq.acc.as_str()),
                Some(query_sq.desc.as_str()),
                "phmmer (HMMER 3.4)",
                &th,
                textw,
            );
            match nseq {
                Some(nseq) => writeln!(
                    out,
                    "# Alignment of {} hits satisfying inclusion thresholds saved to: {}",
                    nseq,
                    args.ali_outfile.as_ref().unwrap().display()
                )
                .unwrap(),
                None => writeln!(
                    out,
                    "# No hits satisfy inclusion thresholds; no alignment saved"
                )
                .unwrap(),
            }
        }

        query_sq.reuse();
    }
    if n_queries == 0 {
        eprintln!("Error: no sequences found in {}", args.seqfile.display());
        std::process::exit(1);
    }

    let table_cmdline = normalize_phmmer_table_cmdline(&cmdline);
    if let Some(ref mut f) = tblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "phmmer",
            "SEARCH",
            &args.seqfile,
            &args.seqdb,
            &table_cmdline,
        );
    }
    if let Some(ref mut f) = domtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "phmmer",
            "SEARCH",
            &args.seqfile,
            &args.seqdb,
            &table_cmdline,
        );
    }
    if let Some(ref mut f) = pfamtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "phmmer",
            "SEARCH",
            &args.seqfile,
            &args.seqdb,
            &table_cmdline,
        );
    }

    if let Some(ref mut f) = tblout_file {
        f.flush().unwrap();
    }
    if let Some(ref mut f) = domtblout_file {
        f.flush().unwrap();
    }
    if let Some(ref mut f) = pfamtblout_file {
        f.flush().unwrap();
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

fn write_phmmer_option_header(out: &mut dyn Write, args: &Args, cmdline: &str) {
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
            "# sequence reporting threshold:    score >= {}",
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
            "# domain reporting threshold:      score >= {}",
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
            hmmer_pure_rs::output::fmt_g(args.inc_dome)
        )
        .unwrap();
    }
    if let Some(score) = args.inc_dom_t {
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

fn normalize_phmmer_table_cmdline(cmdline: &str) -> String {
    let mut tokens = cmdline.split_whitespace();
    match (tokens.next(), tokens.next()) {
        (Some(_wrapper), Some("phmmer")) => {
            let rest = tokens.collect::<Vec<_>>();
            if rest.is_empty() {
                "phmmer".to_string()
            } else {
                format!("phmmer {}", rest.join(" "))
            }
        }
        _ => cmdline.to_string(),
    }
}

fn count_query_sequences(path: &std::path::Path, abc: &Alphabet) -> usize {
    let mut sqf = sequence::open_seq_file(path, abc).unwrap_or_else(|e| {
        eprintln!("Error opening query file: {}", e);
        std::process::exit(1);
    });
    let mut sq = Sequence::new();
    let mut count = 0usize;
    while sqf.read(&mut sq).unwrap_or_else(|e| {
        eprintln!("Error reading query file: {}", e);
        std::process::exit(1);
    }) {
        count += 1;
        if count > 1 {
            break;
        }
        sq.reuse();
    }
    count
}

fn configure_thresholds(pli: &mut Pipeline, args: &Args) {
    pli.e_value_threshold = args.e_value;
    pli.dom_e_value_threshold = args.dom_e;
    pli.inc_e = args.inc_e;
    pli.inc_dome = args.inc_dome;
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
    if let Some(t) = args.inc_dom_t {
        pli.inc_dom_t = Some(t);
        pli.incdom_by_e = false;
    }
    // Model-specific thresholding (--cut_ga/--cut_tc/--cut_nc) is unused in phmmer:
    // the query model is built from a single sequence and never carries GA/TC/NC
    // cutoffs. C (hmmer/src/phmmer.c) still configures the pipeline for these
    // options (p7_pipeline.c:293-316): it sets by_E=FALSE and T=incT=domT=incdomT=0.0
    // and use_bit_cutoffs=p7H_*, then calls p7_pli_NewModel() WITHOUT checking its
    // return status (phmmer.c:565). The subsequent p7_pli_NewModelThresholds() fails
    // silently with eslEINVAL ("bit thresholds unavailable"), leaving T at 0.0, so the
    // effective behavior is "report/include by score >= 0.0". We replicate that exactly
    // here, and (unlike hmmsearch/hmmscan) phmmer never calls new_model_thresholds().
    let cutoff = if args.cut_ga {
        Some(hmmer_pure_rs::pipeline::BitCutoff::GA)
    } else if args.cut_tc {
        Some(hmmer_pure_rs::pipeline::BitCutoff::TC)
    } else if args.cut_nc {
        Some(hmmer_pure_rs::pipeline::BitCutoff::NC)
    } else {
        None
    };
    if let Some(bc) = cutoff {
        pli.use_bit_cutoffs = bc;
        pli.by_e = false;
        pli.dom_by_e = false;
        pli.inc_by_e = false;
        pli.incdom_by_e = false;
        pli.t = Some(0.0);
        pli.dom_t = Some(0.0);
        pli.inc_t = Some(0.0);
        pli.inc_dom_t = Some(0.0);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phmmer_parses_single_sequence_calibration_options() {
        let args = Args::try_parse_from([
            "phmmer",
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

    #[test]
    fn phmmer_accepts_negative_space_separated_f_values() {
        // C --F1/--F2/--F3 have no range; C accepts the space-separated
        // negative form. allow_hyphen_values matches that.
        let args = Args::try_parse_from([
            "phmmer", "--F1", "-0.5", "--F2", "-1e-3", "--F3", "-2", "query.fa", "targets.fa",
        ])
        .unwrap();
        assert_eq!(args.f1, -0.5);
        assert_eq!(args.f2, -1e-3);
        assert_eq!(args.f3, -2.0);
    }
}
