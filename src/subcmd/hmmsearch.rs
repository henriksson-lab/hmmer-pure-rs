//! Pure Rust hmmsearch — uses generic DP algorithms.
//! Progressively replacing C hmmsearch functionality.

use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::output::{fmt_bias, fmt_fixed1, fmt_fixed2, fmt_score, fmt_width5_1};
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::sequence::{self, Sequence, SequenceFormat};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::{Domain, Hit, TopHits};
use hmmer_pure_rs::util::cmath::c_exp_f64;

const TARGET_BATCH_SIZE: usize = 4096;

#[derive(Parser)]
#[command(
    name = "hmmsearch",
    about = "Search profile(s) against a sequence database"
)]
struct Args {
    /// HMM file
    hmmfile: PathBuf,
    /// Sequence database (FASTA format)
    seqdb: PathBuf,

    // --- Output options ---
    /// Direct output to file, not stdout
    #[arg(short = 'o')]
    outfile: Option<PathBuf>,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    domtblout: Option<PathBuf>,

    /// Don't output alignments
    #[arg(long = "noali")]
    noali: bool,

    // --- Reporting thresholds ---
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
    score_threshold: Option<f32>,

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
    dom_t: Option<f32>,

    // --- Inclusion thresholds ---
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
    inc_t: Option<f32>,

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
    inc_dom_t: Option<f32>,

    // --- Model-specific cutoffs ---
    /// Use model's GA gathering cutoffs to set all thresholding
    #[arg(
        long = "cut_ga",
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

    /// Use model's NC noise cutoffs to set all thresholding
    #[arg(
        long = "cut_nc",
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

    /// Use model's TC trusted cutoffs to set all thresholding
    #[arg(
        long = "cut_tc",
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

    // --- Acceleration heuristics ---
    /// Skip all filters (run everything through Forward)
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

    // --- Other expert options ---
    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    nonull2: bool,

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    domz_value: Option<f64>,

    /// Set RNG seed (0: one-time arbitrary seed)
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Start restricted target search at this sequence key
    #[arg(long = "restrictdb_stkey", hide = true)]
    restrictdb_stkey: Option<String>,

    /// Search only this many target sequences from the restricted start
    #[arg(long = "restrictdb_n", value_parser = parse_positive_usize, hide = true)]
    restrictdb_n: Option<usize>,

    /// SSI index file for C-compatible restricted database options
    #[arg(long = "ssifile", hide = true)]
    ssifile: Option<PathBuf>,

    // --- Output formatting ---
    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    show_acc: bool,

    /// Unlimit ASCII text output line width
    #[arg(long = "notextw", conflicts_with = "textw")]
    notextw: bool,

    /// Set max width of ASCII text output lines
    #[arg(long = "textw", default_value = "120", value_parser = parse_textw)]
    textw: usize,

    /// Save table of hits in Pfam format
    #[arg(long = "pfamtblout")]
    pfamtblout: Option<PathBuf>,

    /// Save multiple alignment of all hits to file
    #[arg(short = 'A')]
    ali_outfile: Option<PathBuf>,

    /// Assert target sequence file format (currently FASTA only)
    #[arg(long = "tformat")]
    tformat: Option<String>,
}

fn parse_textw(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid text width: {e}"))?;
    if value < 120 {
        Err("--textw must be >= 120".to_string())
    } else {
        Ok(value)
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

fn copy_reporting_thresholds(dst: &mut Pipeline, src: &Pipeline) {
    dst.e_value_threshold = src.e_value_threshold;
    dst.dom_e_value_threshold = src.dom_e_value_threshold;
    dst.inc_e = src.inc_e;
    dst.inc_dome = src.inc_dome;
    dst.t = src.t;
    dst.dom_t = src.dom_t;
    dst.inc_t = src.inc_t;
    dst.inc_dom_t = src.inc_dom_t;
    dst.by_e = src.by_e;
    dst.dom_by_e = src.dom_by_e;
    dst.inc_by_e = src.inc_by_e;
    dst.incdom_by_e = src.incdom_by_e;
    dst.use_bit_cutoffs = src.use_bit_cutoffs;
}

/// Entry point for `hmmsearch`: search profile(s) against a sequence database.
///
/// Equivalent to the C `main` + `serial_master` + `output_header` in
/// `hmmer/src/hmmsearch.c`. Parses CLI args, reads HMM(s), initializes the
/// pipeline (filter thresholds, reporting/inclusion thresholds, model-specific
/// cutoffs, Z/domZ overrides), iterates target sequences (serial when
/// `--cpu` is 0 or 1, otherwise rayon-parallel over a target batch), thresholds
/// and emits the standard hmmsearch text report plus optional --tblout,
/// --domtblout, --pfamtblout, and -A (Stockholm) outputs.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let cpu_was_requested = args
        .iter()
        .any(|arg| arg == "--cpu" || arg.starts_with("--cpu="))
        || std::env::var_os("HMMER_NCPU").is_some();
    let args = hmmer_pure_rs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);
    if let Some(format) = args.tformat.as_deref() {
        if SequenceFormat::from_name(format).is_none() {
            eprintln!(
                "hmmsearch --tformat={format} is not a recognized input sequence file format",
            );
            std::process::exit(1);
        }
    }

    // `--cpu 0` is HMMER's explicit serial/threading-off mode.
    if args.cpu > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.cpu)
            .start_handler(|_| hmmer_pure_rs::util::simd_env::init())
            .build_global()
            .ok();
    }

    logsum::p7_flogsuminit();

    if args.hmmfile == PathBuf::from("-") && args.seqdb == PathBuf::from("-") {
        eprintln!("Error: Either <hmmfile> or <seqdb> may be '-' but not both");
        std::process::exit(1);
    }

    // Read HMM(s)
    let hmms = read_hmms(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });
    if hmms.len() > 1 && args.seqdb == PathBuf::from("-") {
        eprintln!(
            "Error: target sequence file - isn't rewindable; can't search it with multiple queries"
        );
        std::process::exit(1);
    }

    // Output destination: -o file or stdout
    let outfile_handle;
    let stdout;
    let mut out: Box<dyn std::io::Write> = if let Some(ref path) = args.outfile {
        outfile_handle =
            create_output_file_or_exit(path, "Failed to open output file {path} for writing");
        Box::new(BufWriter::new(outfile_handle))
    } else {
        stdout = std::io::stdout();
        Box::new(BufWriter::new(stdout.lock()))
    };

    // Print header
    writeln!(
        out,
        "# hmmsearch :: search profile(s) against a sequence database"
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
        "# query HMM file:                  {}",
        args.hmmfile.display()
    )
    .unwrap();
    writeln!(
        out,
        "# target sequence database:        {}",
        args.seqdb.display()
    )
    .unwrap();
    if let Some(format) = args.tformat.as_deref() {
        writeln!(out, "# targ <seqfile> format asserted:  {format}").unwrap();
    }
    if let Some(ref path) = args.outfile {
        writeln!(out, "# output directed to file:         {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.tblout {
        writeln!(out, "# per-seq hits tabular output:     {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.domtblout {
        writeln!(out, "# per-dom hits tabular output:     {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.pfamtblout {
        writeln!(out, "# pfam-style tabular hit output:   {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.ali_outfile {
        writeln!(out, "# MSA of all hits saved to file:   {}", path.display()).unwrap();
    }
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    if args.show_acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
    }
    write_hmmsearch_option_header(&mut out, &args, &cmdline);
    if cpu_was_requested {
        writeln!(out, "# number of worker threads:        {}", args.cpu).unwrap();
    }
    if args.notextw {
        writeln!(out, "# max ASCII text line length:      unlimited").unwrap();
    } else if args.textw != 120 {
        writeln!(out, "# max ASCII text line length:      {}", args.textw).unwrap();
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

    // Open tblout/domtblout files if requested
    let mut tblout_file = args.tblout.as_ref().map(|p| {
        let file = create_output_file_or_exit(
            p,
            "Failed to open tabular per-seq output file {path} for writing",
        );
        BufWriter::new(file)
    });
    let mut domtblout_file = args.domtblout.as_ref().map(|p| {
        let file = create_output_file_or_exit(
            p,
            "Failed to open tabular per-dom output file {path} for writing",
        );
        BufWriter::new(file)
    });
    let mut pfamtblout_file = args.pfamtblout.as_ref().map(|p| {
        let file = create_output_file_or_exit(
            p,
            "Failed to open pfam-style tabular output file {path} for writing",
        );
        BufWriter::new(file)
    });
    let mut ali_outfile = args.ali_outfile.as_ref().map(|p| {
        let file =
            create_output_file_or_exit(p, "Failed to open alignment file {path} for writing");
        BufWriter::new(file)
    });

    let textw = if args.notextw { 0 } else { args.textw };

    for (query_idx, hmm) in hmms.iter().enumerate() {
        let abc = Alphabet::new(hmm.abc_type);
        let mut bg = Bg::new(&abc);
        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, 100, P7_LOCAL);
        // Configure bias filter with model composition
        bg.set_filter(hmm.m, &hmm.compo);

        let om = OProfile::convert(&gm);

        let mut pli = Pipeline::new();
        pli.new_model(&gm);

        let effective_f1 = if args.max { 1.0 } else { args.f1 };
        let effective_f2 = if args.max { 1.0 } else { args.f2 };
        let effective_f3 = if args.max { 1.0 } else { args.f3 };
        let effective_nobias = args.max || args.nobias;

        // Filter thresholds
        pli.f1 = effective_f1;
        pli.f2 = effective_f2;
        pli.f3 = effective_f3;
        pli.do_max = args.max;
        if effective_nobias {
            pli.do_biasfilter = false;
        }
        if args.nonull2 {
            pli.do_null2 = false;
        }
        pli.seed = args.seed;

        // Reporting thresholds
        pli.e_value_threshold = args.e_value;
        pli.dom_e_value_threshold = args.dom_e;
        pli.inc_e = args.inc_e;
        pli.inc_dome = args.inc_dome;

        // Score-based thresholds (override E-value based)
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

        // Model-specific cutoffs (override both E-value and score thresholds)
        if args.cut_ga {
            pli.use_bit_cutoffs = hmmer_pure_rs::pipeline::BitCutoff::GA;
        } else if args.cut_tc {
            pli.use_bit_cutoffs = hmmer_pure_rs::pipeline::BitCutoff::TC;
        } else if args.cut_nc {
            pli.use_bit_cutoffs = hmmer_pure_rs::pipeline::BitCutoff::NC;
        }
        if let Err(e) = pli.new_model_thresholds(&hmm.cutoff) {
            eprintln!("Error: {} for model {}", e, hmm.name);
            std::process::exit(1);
        }

        // Database size overrides
        if let Some(z) = args.z_value {
            pli.z = z;
            pli.z_setby = hmmer_pure_rs::pipeline::ZSetBy::Option;
        }
        if let Some(dz) = args.domz_value {
            pli.domz = dz;
            pli.domz_setby = hmmer_pure_rs::pipeline::ZSetBy::Option;
        }

        // Score sequences. For --cpu 0/1, stream the target database like C
        // HMMER instead of cloning the entire sequence set before scoring.
        let f1 = effective_f1;
        let f2 = effective_f2;
        let f3 = effective_f3;
        let do_max = args.max;
        let nobias = effective_nobias;
        let nonull2 = args.nonull2;
        let seed = args.seed;
        let wants_alignment_file = args.ali_outfile.is_some();
        let do_alignment = !args.noali
            || args.domtblout.is_some()
            || args.pfamtblout.is_some()
            || wants_alignment_file;
        let do_alignment_display = !args.noali || wants_alignment_file;
        let mut total_residues: u64 = 0;
        let mut n_targets: u64 = 0;

        let mut th = TopHits::new();
        writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.m).unwrap();
        if let Some(ref acc) = hmm.acc {
            if !acc.is_empty() {
                writeln!(out, "Accession:   {}", acc).unwrap();
            }
        }
        if let Some(ref desc) = hmm.desc {
            if !desc.is_empty() {
                writeln!(out, "Description: {}", desc).unwrap();
            }
        }
        out.flush().unwrap_or_else(|e| {
            eprintln!("Error writing output: {}", e);
            std::process::exit(1);
        });
        if args.cpu <= 1 {
            let mut local_bg = bg.clone();
            let mut local_gm = gm.clone();
            let mut local_om = om.clone();
            let mut local_pli = Pipeline::new();
            local_pli.new_model(&local_gm);
            copy_reporting_thresholds(&mut local_pli, &pli);
            if local_pli.use_bit_cutoffs != hmmer_pure_rs::pipeline::BitCutoff::None {
                local_pli
                    .new_model_thresholds(&hmm.cutoff)
                    .unwrap_or_else(|e| {
                        eprintln!("Error: {} for model {}", e, hmm.name);
                        std::process::exit(1);
                    });
            }
            local_pli.f1 = f1;
            local_pli.f2 = f2;
            local_pli.f3 = f3;
            local_pli.do_max = do_max;
            if nobias {
                local_pli.do_biasfilter = false;
            }
            if nonull2 {
                local_pli.do_null2 = false;
            }
            local_pli.do_alignment = do_alignment;
            local_pli.do_alignment_display = do_alignment_display;
            local_pli.seed = seed;
            local_pli.z = pli.z;
            local_pli.z_setby = pli.z_setby;
            local_pli.domz = pli.domz;
            local_pli.domz_setby = pli.domz_setby;

            let mut sqf = open_target_seq_file(&args.seqdb, &abc, args.tformat.as_deref())
                .unwrap_or_else(|e| {
                    eprintln!("Error opening sequence file: {}", e);
                    std::process::exit(1);
                });
            let mut restrict_started = args.restrictdb_stkey.is_none();
            if let Some(stkey) = args.restrictdb_stkey.as_deref() {
                sqf = open_restricted_target_seq_file(
                    &args.seqdb,
                    &abc,
                    args.tformat.as_deref(),
                    stkey,
                    args.ssifile.as_deref(),
                )
                .unwrap_or_else(|e| {
                    eprintln!("Error opening restricted sequence file: {}", e);
                    std::process::exit(1);
                });
                restrict_started = true;
            }
            let mut sq = Sequence::new();
            let mut restrict_seen = 0usize;
            while sqf.read(&mut sq).unwrap_or_else(|e| {
                eprintln!("Error reading sequence file: {}", e);
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
                n_targets += 1;

                local_pli.n_targets = 0;
                local_pli.n_past_msv = 0;
                local_pli.n_past_bias = 0;
                local_pli.n_past_vit = 0;
                local_pli.n_past_fwd = 0;
                local_bg.set_length(sq.n);

                let mut local_th = TopHits::new();
                if local_pli.run(
                    &mut local_gm,
                    &mut local_om,
                    &local_bg,
                    hmm,
                    &sq,
                    &mut local_th,
                ) {
                    th.hits.extend(local_th.hits.into_iter());
                }
                pli.n_past_msv += local_pli.n_past_msv;
                pli.n_past_bias += local_pli.n_past_bias;
                pli.n_past_vit += local_pli.n_past_vit;
                pli.n_past_fwd += local_pli.n_past_fwd;
                sq.reuse();
            }
        } else {
            use rayon::prelude::*;
            use std::sync::Arc;
            let shared_gm = Arc::new(gm.clone());
            let shared_om = Arc::new(om.clone());

            let mut sqf = open_target_seq_file(&args.seqdb, &abc, args.tformat.as_deref())
                .unwrap_or_else(|e| {
                    eprintln!("Error opening sequence file: {}", e);
                    std::process::exit(1);
                });
            let mut restrict_started = args.restrictdb_stkey.is_none();
            if let Some(stkey) = args.restrictdb_stkey.as_deref() {
                sqf = open_restricted_target_seq_file(
                    &args.seqdb,
                    &abc,
                    args.tformat.as_deref(),
                    stkey,
                    args.ssifile.as_deref(),
                )
                .unwrap_or_else(|e| {
                    eprintln!("Error opening restricted sequence file: {}", e);
                    std::process::exit(1);
                });
                restrict_started = true;
            }
            let mut sq = Sequence::new();
            let mut batch = Vec::with_capacity(TARGET_BATCH_SIZE);
            let mut restrict_seen = 0usize;

            loop {
                while batch.len() < TARGET_BATCH_SIZE {
                    match sqf.read(&mut sq) {
                        Ok(true) => {
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
                            n_targets += 1;
                            batch.push(sq.clone());
                            sq.reuse();
                        }
                        Ok(false) => break,
                        Err(e) => {
                            eprintln!("Error opening sequence file: {}", e);
                            std::process::exit(1);
                        }
                    }
                }

                if batch.is_empty() {
                    break;
                }

                let results: Vec<(Option<hmmer_pure_rs::tophits::Hit>, u64, u64, u64, u64)> = batch
                    .par_iter()
                    .map_init(
                        || {
                            hmmer_pure_rs::util::simd_env::init();
                            let local_gm = (*shared_gm).clone();
                            let mut local_pli = Pipeline::new();
                            local_pli.new_model(&local_gm);
                            copy_reporting_thresholds(&mut local_pli, &pli);
                            if local_pli.use_bit_cutoffs != hmmer_pure_rs::pipeline::BitCutoff::None
                            {
                                local_pli
                                    .new_model_thresholds(&hmm.cutoff)
                                    .unwrap_or_else(|e| {
                                        eprintln!("Error: {} for model {}", e, hmm.name);
                                        std::process::exit(1);
                                    });
                            }
                            local_pli.f1 = f1;
                            local_pli.f2 = f2;
                            local_pli.f3 = f3;
                            local_pli.do_max = do_max;
                            if nobias {
                                local_pli.do_biasfilter = false;
                            }
                            if nonull2 {
                                local_pli.do_null2 = false;
                            }
                            local_pli.do_alignment = do_alignment;
                            local_pli.do_alignment_display = do_alignment_display;
                            local_pli.seed = seed;
                            local_pli.z = pli.z;
                            local_pli.z_setby = pli.z_setby;
                            local_pli.domz = pli.domz;
                            local_pli.domz_setby = pli.domz_setby;
                            (bg.clone(), local_gm, (*shared_om).clone(), local_pli)
                        },
                        |(local_bg, local_gm, local_om, local_pli), sq| {
                            local_pli.n_targets = 0;
                            local_pli.n_past_msv = 0;
                            local_pli.n_past_bias = 0;
                            local_pli.n_past_vit = 0;
                            local_pli.n_past_fwd = 0;

                            local_bg.set_length(sq.n);

                            let mut local_th = TopHits::new();
                            let hit = if local_pli.run(
                                local_gm,
                                local_om,
                                local_bg,
                                hmm,
                                sq,
                                &mut local_th,
                            ) {
                                local_th.hits.into_iter().next()
                            } else {
                                None
                            };
                            (
                                hit,
                                local_pli.n_past_msv,
                                local_pli.n_past_bias,
                                local_pli.n_past_vit,
                                local_pli.n_past_fwd,
                            )
                        },
                    )
                    .collect();

                for (hit, msv, bias, vit, fwd) in results {
                    pli.n_past_msv += msv;
                    pli.n_past_bias += bias;
                    pli.n_past_vit += vit;
                    pli.n_past_fwd += fwd;
                    if let Some(h) = hit {
                        th.hits.push(h);
                    }
                }

                batch.clear();
            }
        }

        if n_targets == 0 {
            eprintln!("Error: no sequences found in {}", args.seqdb.display());
            std::process::exit(1);
        }

        pli.n_targets = n_targets;

        // Set Z (database size)
        let z = match pli.z_setby {
            hmmer_pure_rs::pipeline::ZSetBy::Option => pli.z,
            hmmer_pure_rs::pipeline::ZSetBy::Ntargets => pli.n_targets as f64,
        };

        // Sort and threshold (first pass with domz = z for sequence-level reporting)
        th.sort_by_sortkey();
        th.threshold(
            &pli, z, z, // temporary domz = z for first pass
        );

        // Set domz: user-specified, or auto = nreported from first pass
        let domz = match pli.domz_setby {
            hmmer_pure_rs::pipeline::ZSetBy::Option => pli.domz,
            hmmer_pure_rs::pipeline::ZSetBy::Ntargets => th.nreported as f64,
        };
        // Re-threshold with correct domz for domain-level E-values
        if domz != z {
            th.threshold(&pli, z, domz);
        }

        let show_acc = args.show_acc;
        let target_namew = th
            .hits
            .iter()
            .map(|hit| shown_hit_name(hit, show_acc).len())
            .max()
            .unwrap_or(0)
            .max(8);
        let target_descw = if textw > 0 {
            textw.saturating_sub(target_namew + 61).max(32)
        } else {
            0
        };

        // Per-sequence hit table
        writeln!(
            out,
            "Scores for complete sequences (score includes all domains):"
        )
        .unwrap();
        writeln!(
            out,
            "   --- full sequence ---   --- best 1 domain ---    -#dom-"
        )
        .unwrap();
        writeln!(
            out,
            "    E-value  score  bias    E-value  score  bias    exp  N  {:<target_namew$} Description",
            "Sequence"
        )
        .unwrap();
        writeln!(
            out,
            "    ------- ------ -----    ------- ------ -----   ---- --  {:<target_namew$} -----------",
            "--------"
        )
        .unwrap();

        let mut any_reported = false;
        let mut have_printed_incthresh = false;
        for hit in &th.hits {
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
                continue;
            }
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED == 0 && !have_printed_incthresh {
                writeln!(out, "  ------ inclusion threshold ------").unwrap();
                have_printed_incthresh = true;
            }
            any_reported = true;
            let evalue = evalue_from_lnp(z, hit.lnp);
            let best_dom = best_domain(hit);
            let dom_evalue = best_dom
                .map(|d| evalue_from_lnp(z, d.lnp))
                .unwrap_or(evalue);
            let dom_score = best_dom.map(|d| d.bitscore).unwrap_or(hit.score);
            let dom_bias = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);
            let desc = if hit.desc.is_empty() { "" } else { &hit.desc };
            let desc = truncate_for_textw(desc, target_descw);

            writeln!(
                out,
                "  {} {} {}  {} {} {}  {} {:2}  {:<target_namew$}  {}",
                hmmer_pure_rs::output::fmt_evalue(evalue),
                hmmer_pure_rs::output::fmt_score(hit.score),
                hmmer_pure_rs::output::fmt_bias(hit.bias),
                hmmer_pure_rs::output::fmt_evalue(dom_evalue),
                hmmer_pure_rs::output::fmt_score(dom_score),
                hmmer_pure_rs::output::fmt_bias(dom_bias),
                fmt_width5_1(hit.nexpected as f64),
                hit.nreported,
                shown_hit_name(hit, show_acc),
                desc,
            )
            .unwrap();
        }

        if !any_reported {
            writeln!(
                out,
                "\n   [No hits detected that satisfy reporting thresholds]"
            )
            .unwrap();
        }

        writeln!(out).unwrap();
        writeln!(out).unwrap();

        // Domain annotation for each sequence. C HMMER still prints the domain
        // table under --noali; only the alignment blocks are suppressed.
        if args.noali {
            writeln!(out, "Domain annotation for each sequence:").unwrap();
        } else {
            writeln!(out, "Domain annotation for each sequence (and alignments):").unwrap();
        }

        for hit in &th.hits {
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
                continue;
            }

            let domain_descw = if textw > 0 {
                textw
                    .saturating_sub(shown_hit_name(hit, show_acc).len() + 5)
                    .max(32)
            } else {
                0
            };
            let desc = truncate_for_textw(&hit.desc, domain_descw);
            writeln!(out, ">> {}  {}", shown_hit_name(hit, show_acc), desc).unwrap();

            if hit.nreported == 0 {
                writeln!(
                    out,
                    "   [No individual domains that satisfy reporting thresholds (although complete target did)]"
                )
                .unwrap();
                writeln!(out).unwrap();
                continue;
            }

            writeln!(out, "   #    score  bias  c-Evalue  i-Evalue hmmfrom  hmm to    alifrom  ali to    envfrom  env to     acc").unwrap();
            writeln!(out, " ---   ------ ----- --------- --------- ------- -------    ------- -------    ------- -------    ----").unwrap();

            let mut reported_idx = 0usize;
            for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
                reported_idx += 1;
                let c_evalue = evalue_from_lnp(domz, dom.lnp);
                let i_evalue = evalue_from_lnp(z, dom.lnp);
                let indicator = if dom.is_included {
                    '!'
                } else if dom.is_reported {
                    '?'
                } else {
                    '?'
                };

                let (hf, ht) = if let Some(ref ad) = dom.ad {
                    (ad.hmmfrom, ad.hmmto)
                } else {
                    (1, hmm.m)
                };
                // Boundary indicators
                let hmm_left = if hf == 1 { '[' } else { '.' };
                let hmm_right = if ht == hmm.m { ']' } else { '.' };
                let seq_left = if dom.iali == 1 { '[' } else { '.' };
                let seq_right = if dom.jali == hit.n as i64 { ']' } else { '.' };
                let env_left = if dom.ienv == 1 { '[' } else { '.' };
                let env_right = if dom.jenv == hit.n as i64 { ']' } else { '.' };
                let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);

                writeln!(
                    out,
                    " {:3} {} {} {} {} {} {:7} {:7} {}{} {:7} {:7} {}{} {:7} {:7} {}{} {}",
                    reported_idx,
                    indicator,
                    hmmer_pure_rs::output::fmt_score(dom.bitscore),
                    hmmer_pure_rs::output::fmt_bias(dom.dombias),
                    hmmer_pure_rs::output::fmt_evalue(c_evalue),
                    hmmer_pure_rs::output::fmt_evalue(i_evalue),
                    hf,
                    ht,
                    hmm_left,
                    hmm_right,
                    dom.iali,
                    dom.jali,
                    seq_left,
                    seq_right,
                    dom.ienv,
                    dom.jenv,
                    env_left,
                    env_right,
                    hmmer_pure_rs::output::fmt_fixed2(acc as f64),
                )
                .unwrap();
            }

            writeln!(out).unwrap();

            // Text alignments for each domain
            if !args.noali {
                writeln!(out, "  Alignments for each domain:").unwrap();
                let mut reported_idx = 0usize;
                for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
                    reported_idx += 1;
                    writeln!(
                        out,
                        "  == domain {}  score: {} bits;  conditional E-value: {}",
                        reported_idx,
                        hmmer_pure_rs::output::fmt_fixed1(dom.bitscore as f64),
                        hmmer_pure_rs::output::fmt_evalue(evalue_from_lnp(domz, dom.lnp)).trim()
                    )
                    .unwrap();

                    if let Some(ref ad) = dom.ad {
                        // Build CS line from hmm.cs over the alignment span.
                        let cs_line = hmm.cs.as_ref().map(|cs| {
                            let mut s = String::with_capacity(ad.model.len());
                            let mut cs_idx = ad.hmmfrom;
                            for ch in ad.model.chars() {
                                if ch == '.' {
                                    s.push('.');
                                } else {
                                    s.push(if cs_idx < cs.len() {
                                        cs[cs_idx] as char
                                    } else {
                                        ' '
                                    });
                                    cs_idx += 1;
                                }
                            }
                            s
                        });
                        hmmer_pure_rs::tophits::print_alidisplay_blocks_acc(
                            &mut out,
                            &hmm.name,
                            hmm.acc.as_deref().unwrap_or(""),
                            &hit.name,
                            &hit.acc,
                            ad,
                            cs_line.as_deref(),
                            textw,
                            show_acc,
                        );
                    }
                    writeln!(out).unwrap();
                }
            }
        }

        writeln!(out).unwrap();
        writeln!(out).unwrap();

        // Statistics
        let expected_msv = (pli.f1 * pli.n_targets as f64).max(0.0);
        let expected_vit = (pli.f2 * pli.n_targets as f64).max(0.0);
        let expected_fwd = (pli.f3 * pli.n_targets as f64).max(0.0);
        let frac_msv = if pli.n_targets > 0 {
            pli.n_past_msv as f64 / pli.n_targets as f64
        } else {
            0.0
        };
        let frac_vit = if pli.n_targets > 0 {
            pli.n_past_vit as f64 / pli.n_targets as f64
        } else {
            0.0
        };
        let frac_fwd = if pli.n_targets > 0 {
            pli.n_past_fwd as f64 / pli.n_targets as f64
        } else {
            0.0
        };
        let frac_bias = if pli.n_targets > 0 {
            pli.n_past_bias as f64 / pli.n_targets as f64
        } else {
            0.0
        };

        writeln!(out, "Internal pipeline statistics summary:").unwrap();
        writeln!(out, "-------------------------------------").unwrap();
        writeln!(
            out,
            "Query model(s):                  {:>11}  ({} nodes)",
            1, hmm.m
        )
        .unwrap();
        writeln!(
            out,
            "Target sequences:                {:>11}  ({} residues searched)",
            pli.n_targets, total_residues
        )
        .unwrap();
        writeln!(
            out,
            "Passed MSV filter:               {:>11}  ({}); expected {} ({})",
            pli.n_past_msv,
            hmmer_pure_rs::output::fmt_g(frac_msv),
            hmmer_pure_rs::output::fmt_fixed1(expected_msv),
            hmmer_pure_rs::output::fmt_g(pli.f1)
        )
        .unwrap();
        writeln!(
            out,
            "Passed bias filter:              {:>11}  ({}); expected {} ({})",
            pli.n_past_bias,
            hmmer_pure_rs::output::fmt_g(frac_bias),
            hmmer_pure_rs::output::fmt_fixed1(expected_msv),
            hmmer_pure_rs::output::fmt_g(pli.f1)
        )
        .unwrap();
        writeln!(
            out,
            "Passed Vit filter:               {:>11}  ({}); expected {} ({})",
            pli.n_past_vit,
            hmmer_pure_rs::output::fmt_g(frac_vit),
            hmmer_pure_rs::output::fmt_fixed1(expected_vit),
            hmmer_pure_rs::output::fmt_g(pli.f2)
        )
        .unwrap();
        writeln!(
            out,
            "Passed Fwd filter:               {:>11}  ({}); expected {} ({})",
            pli.n_past_fwd,
            hmmer_pure_rs::output::fmt_g(frac_fwd),
            hmmer_pure_rs::output::fmt_fixed1(expected_fwd),
            hmmer_pure_rs::output::fmt_g(pli.f3)
        )
        .unwrap();
        writeln!(
            out,
            "Initial search space (Z):        {}  {}",
            hmmer_pure_rs::output::fmt_width11_0(z),
            if pli.z_setby == hmmer_pure_rs::pipeline::ZSetBy::Option {
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
            if pli.domz_setby == hmmer_pure_rs::pipeline::ZSetBy::Option {
                "[as set by --domZ on cmdline]"
            } else {
                "[number of targets reported over threshold]"
            }
        )
        .unwrap();

        // Write tabular output
        if let Some(ref mut f) = tblout_file {
            write_tblout(f, &hmm.name, hmm.acc.as_deref(), &th, z, query_idx == 0);
        }
        if let Some(ref mut f) = domtblout_file {
            write_domtblout(
                f,
                &hmm.name,
                hmm.acc.as_deref(),
                hmm.m,
                &th,
                z,
                domz,
                query_idx == 0,
            );
        }
        if let Some(ref mut f) = pfamtblout_file {
            write_pfamtblout_with_pipeline(
                f,
                &hmm.name,
                hmm.acc.as_deref(),
                hmm.m,
                &th,
                &pli,
                z,
                domz,
            );
        }
        writeln!(out, "//").unwrap();
        if let Some(ref mut f) = ali_outfile {
            // C hmmsearch.c:554-572: after the "//" line, build the
            // included-domain MSA, write it, then echo the confirmation line
            // (carrying the MSA's nseq) to stdout.
            let nseq = write_ali_output(
                f,
                &abc,
                hmm.m,
                &hmm.name,
                hmm.acc.as_deref(),
                hmm.desc.as_deref(),
                "hmmsearch (HMMER 3.4)",
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
    }

    if let Some(ref mut f) = tblout_file {
        write_table_footer(
            f,
            "hmmsearch",
            "SEARCH",
            &args.hmmfile,
            &args.seqdb,
            &cmdline,
        );
    }
    if let Some(ref mut f) = domtblout_file {
        write_table_footer(
            f,
            "hmmsearch",
            "SEARCH",
            &args.hmmfile,
            &args.seqdb,
            &cmdline,
        );
    }
    if let Some(ref mut f) = pfamtblout_file {
        write_table_footer(
            f,
            "hmmsearch",
            "SEARCH",
            &args.hmmfile,
            &args.seqdb,
            &cmdline,
        );
    }

    writeln!(out, "[ok]").unwrap();
    out.flush().unwrap();
    if let Some(ref mut f) = tblout_file {
        f.flush().unwrap();
    }
    if let Some(ref mut f) = domtblout_file {
        f.flush().unwrap();
    }
    if let Some(ref mut f) = pfamtblout_file {
        f.flush().unwrap();
    }
    if let Some(ref mut f) = ali_outfile {
        f.flush().unwrap();
    }
    std::process::ExitCode::SUCCESS
}

pub(crate) fn create_output_file_or_exit(
    path: &std::path::Path,
    message_template: &str,
) -> std::fs::File {
    std::fs::File::create(path).unwrap_or_else(|_| {
        eprintln!(
            "{}",
            message_template.replace("{path}", &path.display().to_string())
        );
        std::process::exit(1);
    })
}

fn write_hmmsearch_option_header(out: &mut dyn Write, args: &Args, cmdline: &str) {
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
    if args.cut_ga {
        writeln!(out, "# model-specific thresholding:     GA cutoffs").unwrap();
    }
    if args.cut_nc {
        writeln!(out, "# model-specific thresholding:     NC cutoffs").unwrap();
    }
    if args.cut_tc {
        writeln!(out, "# model-specific thresholding:     TC cutoffs").unwrap();
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

/// Read one or more profile HMMs from `path`, auto-dispatching ASCII vs binary
/// by magic bytes like C `p7_hmmfile_Open`.
fn read_hmms(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<hmmer_pure_rs::Hmm>> {
    if path == std::path::Path::new("-") {
        hmmfile::read_hmms_auto(BufReader::new(std::io::stdin().lock()))
    } else {
        hmmfile::read_hmm_file_auto(path)
    }
}

/// Write per-sequence tabular output (`--tblout`).
///
/// Mirrors the non-long_target branch of C `p7_tophits_TabularTargets`
/// (`hmmer/src/p7_tophits.c`): a header trio (column widths sized to the
/// longest target name / accession and the query name / accession,
/// minimums 20/10/20/10) followed by one row per reported hit with full
/// sequence and best-domain stats.
pub fn write_tblout<W: Write>(
    f: &mut W,
    qname: &str,
    qacc: Option<&str>,
    th: &TopHits,
    z: f64,
    show_header: bool,
) {
    let tnamew = th
        .hits
        .iter()
        .map(|h| h.name.len())
        .max()
        .unwrap_or(0)
        .max(20);
    let taccw = th
        .hits
        .iter()
        .map(|h| if h.acc.is_empty() { 1 } else { h.acc.len() })
        .max()
        .unwrap_or(0)
        .max(10);
    let qnamew = qname.len().max(20);
    let tname_hdrw = tnamew - 1;
    let qacc_s = qacc.filter(|s| !s.is_empty()).unwrap_or("-");
    let qaccw = qacc_s.len().max(10);

    if show_header {
        writeln!(
            f,
            "#{:>w$} {:>22} {:>22} {:>33}",
            "",
            "--- full sequence ----",
            "--- best 1 domain ----",
            "--- domain number estimation ----",
            w = tnamew + qnamew + taccw + qaccw + 2
        )
        .unwrap();
        writeln!(
            f,
            "#{:<tname_hdrw$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:>9} {:>6} {:>5} {:>9} {:>6} {:>5} {:>5} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {}",
            " target name",
            "accession",
            "query name",
            "accession",
            "  E-value",
            " score",
            " bias",
            "  E-value",
            " score",
            " bias",
            "exp",
            "reg",
            "clu",
            " ov",
            "env",
            "dom",
            "rep",
            "inc",
            "description of target"
        )
        .unwrap();
        writeln!(
            f,
            "#{:>tname_hdrw$} {:>taccw$} {:>qnamew$} {:>qaccw$} {:>9} {:>6} {:>5} {:>9} {:>6} {:>5} {:>5} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {}",
            "-------------------",
            "----------",
            "--------------------",
            "----------",
            "---------",
            "------",
            "-----",
            "---------",
            "------",
            "-----",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---------------------"
        )
        .unwrap();
    }

    for hit in &th.hits {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        let evalue = evalue_from_lnp(z, hit.lnp);
        let best_dom = best_domain(hit);
        let dom_evalue = best_dom
            .map(|d| evalue_from_lnp(z, d.lnp))
            .unwrap_or(evalue);
        let dom_score = best_dom.map(|d| d.bitscore).unwrap_or(hit.score);
        let dom_bias = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);

        writeln!(
            f,
            "{:<tnamew$} {:<taccw$} {:<qnamew$} {:<qaccw$} {} {} {} {} {} {} {:>5} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {}",
            hit.name,
            if hit.acc.is_empty() { "-" } else { &hit.acc },
            qname,
            qacc_s,
            hmmer_pure_rs::output::fmt_evalue(evalue),
            hmmer_pure_rs::output::fmt_score(hit.score),
            hmmer_pure_rs::output::fmt_bias(hit.bias),
            hmmer_pure_rs::output::fmt_evalue(dom_evalue),
            hmmer_pure_rs::output::fmt_score(dom_score),
            hmmer_pure_rs::output::fmt_bias(dom_bias),
            hmmer_pure_rs::output::fmt_width5_1(hit.nexpected as f64),
            hit.nregions,
            hit.nclustered,
            hit.noverlaps,
            hit.nenvelopes,
            hit.ndom,
            hit.nreported,
            hit.nincluded,
            if hit.desc.is_empty() { "-" } else { &hit.desc },
        ).unwrap();
    }
}

pub fn best_domain(hit: &Hit) -> Option<&Domain> {
    // C p7_pipeline.c:1110 uses a strict `>` scan, so the FIRST domain with the
    // maximum bitscore wins on a tie. `Iterator::max_by` keeps the LAST max, so
    // fold manually replacing only on a strict increase.
    hit.dcl.iter().fold(None, |best: Option<&Domain>, d| match best {
        Some(b) if b.bitscore >= d.bitscore => Some(b),
        _ => Some(d),
    })
}

#[inline]
fn evalue_from_lnp(scale: f64, lnp: f64) -> f64 {
    scale * c_exp_f64(lnp)
}

fn shown_hit_name(hit: &Hit, show_acc: bool) -> &str {
    if show_acc && !hit.acc.is_empty() {
        &hit.acc
    } else {
        &hit.name
    }
}

fn truncate_for_textw(s: &str, width: usize) -> String {
    if width == 0 {
        return s.to_string();
    }
    s.chars().take(width).collect()
}

pub fn write_standard_stdout_tables<W: Write>(
    out: &mut W,
    th: &TopHits,
    model_name: &str,
    model_acc: &str,
    model_len: usize,
    model_cs: Option<&[u8]>,
    z: f64,
    domz: f64,
    textw: usize,
    show_acc: bool,
    noali: bool,
    show_hit_status: bool,
) {
    let target_namew = th
        .hits
        .iter()
        .map(|hit| shown_hit_name(hit, show_acc).len())
        .max()
        .unwrap_or(0)
        .max(8);
    let target_descw = if textw > 0 {
        textw.saturating_sub(target_namew + 61).max(32)
    } else {
        0
    };

    writeln!(
        out,
        "Scores for complete sequences (score includes all domains):"
    )
    .unwrap();
    writeln!(
        out,
        "   --- full sequence ---   --- best 1 domain ---    -#dom-"
    )
    .unwrap();
    writeln!(
        out,
        "    E-value  score  bias    E-value  score  bias    exp  N  {:<target_namew$} Description",
        "Sequence"
    )
    .unwrap();
    writeln!(
        out,
        "    ------- ------ -----    ------- ------ -----   ---- --  {:<target_namew$} -----------",
        "--------"
    )
    .unwrap();

    let mut any_reported = false;
    let mut have_printed_incthresh = false;
    for hit in &th.hits {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED == 0 && !have_printed_incthresh {
            writeln!(out, "  ------ inclusion threshold ------").unwrap();
            have_printed_incthresh = true;
        }
        any_reported = true;
        let evalue = evalue_from_lnp(z, hit.lnp);
        let best_dom = best_domain(hit);
        let dom_evalue = best_dom
            .map(|d| evalue_from_lnp(z, d.lnp))
            .unwrap_or(evalue);
        let dom_score = best_dom.map(|d| d.bitscore).unwrap_or(hit.score);
        let dom_bias = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);
        let desc = truncate_for_textw(&hit.desc, target_descw);
        let status = if show_hit_status
            && hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED != 0
            && hit.flags & hmmer_pure_rs::tophits::P7_IS_NEW != 0
        {
            '+'
        } else if show_hit_status && hit.flags & hmmer_pure_rs::tophits::P7_IS_DROPPED != 0 {
            '-'
        } else {
            ' '
        };

        writeln!(
            out,
            "{} {} {} {}  {} {} {}  {} {:2}  {:<target_namew$}  {}",
            status,
            hmmer_pure_rs::output::fmt_evalue(evalue),
            fmt_score(hit.score),
            fmt_bias(hit.bias),
            hmmer_pure_rs::output::fmt_evalue(dom_evalue),
            fmt_score(dom_score),
            fmt_bias(dom_bias),
            fmt_width5_1(hit.nexpected as f64),
            hit.nreported,
            shown_hit_name(hit, show_acc),
            desc,
        )
        .unwrap();
    }

    if !any_reported {
        writeln!(
            out,
            "\n   [No hits detected that satisfy reporting thresholds]"
        )
        .unwrap();
    }

    writeln!(out).unwrap();
    writeln!(out).unwrap();

    if noali {
        writeln!(out, "Domain annotation for each sequence:").unwrap();
    } else {
        writeln!(out, "Domain annotation for each sequence (and alignments):").unwrap();
    }

    for hit in &th.hits {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }

        let domain_descw = if textw > 0 {
            textw
                .saturating_sub(shown_hit_name(hit, show_acc).len() + 5)
                .max(32)
        } else {
            0
        };
        let desc = truncate_for_textw(&hit.desc, domain_descw);
        writeln!(out, ">> {}  {}", shown_hit_name(hit, show_acc), desc).unwrap();

        if hit.nreported == 0 {
            writeln!(
                out,
                "   [No individual domains that satisfy reporting thresholds (although complete target did)]"
            )
            .unwrap();
            writeln!(out).unwrap();
            continue;
        }

        writeln!(out, "   #    score  bias  c-Evalue  i-Evalue hmmfrom  hmm to    alifrom  ali to    envfrom  env to     acc").unwrap();
        writeln!(out, " ---   ------ ----- --------- --------- ------- -------    ------- -------    ------- -------    ----").unwrap();

        let mut reported_idx = 0usize;
        for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
            reported_idx += 1;
            let c_evalue = evalue_from_lnp(domz, dom.lnp);
            let i_evalue = evalue_from_lnp(z, dom.lnp);
            let indicator = if dom.is_included { '!' } else { '?' };

            let (hf, ht) = if let Some(ref ad) = dom.ad {
                (ad.hmmfrom, ad.hmmto)
            } else {
                (1, model_len)
            };
            let hmm_left = if hf == 1 { '[' } else { '.' };
            let hmm_right = if ht == model_len { ']' } else { '.' };
            let seq_left = if dom.iali == 1 { '[' } else { '.' };
            let seq_right = if dom.jali == hit.n as i64 { ']' } else { '.' };
            let env_left = if dom.ienv == 1 { '[' } else { '.' };
            let env_right = if dom.jenv == hit.n as i64 { ']' } else { '.' };
            let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);

            writeln!(
                out,
                " {:3} {} {} {} {} {} {:7} {:7} {}{} {:7} {:7} {}{} {:7} {:7} {}{} {}",
                reported_idx,
                indicator,
                fmt_score(dom.bitscore),
                fmt_bias(dom.dombias),
                hmmer_pure_rs::output::fmt_evalue(c_evalue),
                hmmer_pure_rs::output::fmt_evalue(i_evalue),
                hf,
                ht,
                hmm_left,
                hmm_right,
                dom.iali,
                dom.jali,
                seq_left,
                seq_right,
                dom.ienv,
                dom.jenv,
                env_left,
                env_right,
                fmt_fixed2(acc as f64),
            )
            .unwrap();
        }

        writeln!(out).unwrap();

        if !noali {
            writeln!(out, "  Alignments for each domain:").unwrap();
            let mut reported_idx = 0usize;
            for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
                reported_idx += 1;
                writeln!(
                    out,
                    "  == domain {}  score: {} bits;  conditional E-value: {}",
                    reported_idx,
                    fmt_fixed1(dom.bitscore as f64),
                    hmmer_pure_rs::output::fmt_evalue(evalue_from_lnp(domz, dom.lnp)).trim()
                )
                .unwrap();

                if let Some(ref ad) = dom.ad {
                    let mut ad_for_print = ad.clone();
                    if ad_for_print.rfline.is_empty() {
                        ad_for_print.model = ad_for_print.model.to_ascii_lowercase();
                        ad_for_print.mline = ad_for_print.mline.to_ascii_lowercase();
                    }
                    let cs_line = model_cs.map(|cs| {
                        let mut s = String::with_capacity(ad.model.len());
                        let mut cs_idx = ad.hmmfrom;
                        for ch in ad.model.chars() {
                            if ch == '.' {
                                s.push('.');
                            } else {
                                s.push(if cs_idx < cs.len() {
                                    cs[cs_idx] as char
                                } else {
                                    ' '
                                });
                                cs_idx += 1;
                            }
                        }
                        s
                    });
                    // Honor --acc on both the model and sequence sides
                    // (C p7_alidisplay.c:1176-1180): each side shows its
                    // accession when --acc is set and the accession is non-empty,
                    // else falls back to the name.
                    hmmer_pure_rs::tophits::print_alidisplay_blocks_acc(
                        out,
                        model_name,
                        model_acc,
                        &hit.name,
                        &hit.acc,
                        &ad_for_print,
                        cs_line.as_deref(),
                        textw,
                        show_acc,
                    );
                }
                writeln!(out).unwrap();
            }
        }
    }

    writeln!(out).unwrap();
    writeln!(out).unwrap();
}

/// Write per-domain tabular output (`--domtblout`).
///
/// Mirrors C `p7_tophits_TabularDomains` (`hmmer/src/p7_tophits.c`). Emits one
/// row per reported domain with full-sequence stats plus per-domain c-Evalue,
/// i-Evalue, score, bias, and hmm/ali/env coordinate triples. Column widths
/// auto-fit target names and accessions, with 20/10 minimums.
pub fn write_domtblout<W: Write>(
    f: &mut W,
    qname: &str,
    qacc: Option<&str>,
    qlen: usize,
    th: &TopHits,
    z: f64,
    domz: f64,
    show_header: bool,
) {
    let tnamew = th
        .hits
        .iter()
        .map(|h| h.name.len())
        .max()
        .unwrap_or(0)
        .max(20);
    let taccw = th
        .hits
        .iter()
        .map(|h| if h.acc.is_empty() { 1 } else { h.acc.len() })
        .max()
        .unwrap_or(0)
        .max(10);
    let qnamew = qname.len().max(20);
    let tname_hdrw = tnamew - 1;
    let qacc_s = qacc.filter(|s| !s.is_empty()).unwrap_or("-");
    let qaccw = qacc_s.len().max(10);

    if show_header {
        writeln!(
            f,
            "#{:>w$} {:>22} {:>40} {:>11} {:>11} {:>11}",
            "",
            "--- full sequence ---",
            "-------------- this domain -------------",
            "hmm coord",
            "ali coord",
            "env coord",
            w = tnamew + qnamew + 15 + taccw + qaccw - 1
        )
        .unwrap();
        writeln!(
            f,
            "#{:<tname_hdrw$} {:<taccw$} {:>5} {:<qnamew$} {:<qaccw$} {:>5} {:>9} {:>6} {:>5} {:>3} {:>3} {:>9} {:>9} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>4} {}",
            " target name",
            "accession",
            "tlen",
            "query name",
            "accession",
            "qlen",
            "E-value",
            "score",
            "bias",
            "#",
            "of",
            "c-Evalue",
            "i-Evalue",
            "score",
            "bias",
            "from",
            "to",
            "from",
            "to",
            "from",
            "to",
            "acc",
            "description of target"
        )
        .unwrap();
        writeln!(
            f,
            "#{:>tname_hdrw$} {:>taccw$} {:>5} {:>qnamew$} {:>qaccw$} {:>5} {:>9} {:>6} {:>5} {:>3} {:>3} {:>9} {:>9} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>4} {}",
            "-------------------",
            "----------",
            "-----",
            "--------------------",
            "----------",
            "-----",
            "---------",
            "------",
            "-----",
            "---",
            "---",
            "---------",
            "---------",
            "------",
            "-----",
            "-----",
            "-----",
            "-----",
            "-----",
            "-----",
            "-----",
            "----",
            "---------------------"
        )
        .unwrap();
    }

    for hit in &th.hits {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        let evalue = evalue_from_lnp(z, hit.lnp);

        let mut reported_idx = 0usize;
        for dom in &hit.dcl {
            if !dom.is_reported {
                continue;
            }
            reported_idx += 1;
            let c_evalue = evalue_from_lnp(domz, dom.lnp);
            let i_evalue = evalue_from_lnp(z, dom.lnp);
            let (hmmfrom, hmmto, acc) = if let Some(ref ad) = dom.ad {
                let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);
                (ad.hmmfrom, ad.hmmto, acc)
            } else {
                (1, qlen, 0.0)
            };

            writeln!(
            f,
                "{:<tnamew$} {:<taccw$} {:>5} {:<qnamew$} {:<qaccw$} {:>5} {} {} {} {:>3} {:>3} {} {} {} {} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {} {}",
                hit.name,
                if hit.acc.is_empty() { "-" } else { &hit.acc },
                hit.n,
                qname,
                qacc_s,
                qlen,
                hmmer_pure_rs::output::fmt_evalue(evalue),
                hmmer_pure_rs::output::fmt_score(hit.score),
                hmmer_pure_rs::output::fmt_bias(hit.bias),
                reported_idx,
                hit.nreported,
                hmmer_pure_rs::output::fmt_evalue(c_evalue),
                hmmer_pure_rs::output::fmt_evalue(i_evalue),
                hmmer_pure_rs::output::fmt_score(dom.bitscore),
                hmmer_pure_rs::output::fmt_bias(dom.dombias),
                hmmfrom,
                hmmto,
                dom.iali,
                dom.jali,
                dom.ienv,
                dom.jenv,
                hmmer_pure_rs::output::fmt_width4_2(acc as f64),
                if hit.desc.is_empty() { "-" } else { &hit.desc },
            ).unwrap();
        }
    }
}

/// Write Pfam-format tabular output (`--pfamtblout`).
///
/// Two sections: sequence scores then domain scores, both restricted to
/// reported hits. Domain rows are re-sorted as C pseudo-hits, using the full
/// sort-key comparator from `p7_tophits_TabularXfam`.
pub fn write_pfamtblout<W: Write>(
    f: &mut W,
    _qname: &str,
    _qacc: Option<&str>,
    qlen: usize,
    th: &TopHits,
    z: f64,
    _domz: f64,
) {
    let pli = Pipeline::new();
    write_pfamtblout_with_pipeline(f, _qname, _qacc, qlen, th, &pli, z, _domz);
}

pub(crate) fn write_pfamtblout_with_pipeline<W: Write>(
    f: &mut W,
    _qname: &str,
    _qacc: Option<&str>,
    qlen: usize,
    th: &TopHits,
    pli: &Pipeline,
    z: f64,
    _domz: f64,
) {
    let tnamew = th
        .hits
        .iter()
        .map(|h| h.name.len())
        .max()
        .unwrap_or(0)
        .max(20);

    writeln!(f, "# Sequence scores").unwrap();
    writeln!(f, "# ---------------").unwrap();
    writeln!(f, "#").unwrap();
    writeln!(
        f,
        "# {:<tname_hdrw$} {:>6} {:>9} {:>3} {:>5} {:>5}    {}",
        "name",
        " bits",
        "  E-value",
        "n",
        "exp",
        " bias",
        "description",
        tname_hdrw = tnamew - 1,
    )
    .unwrap();
    writeln!(
        f,
        "# {:>tname_hdrw$} {:>6} {:>9} {:>3} {:>5} {:>5}    {}",
        "-------------------",
        "------",
        "---------",
        "---",
        "-----",
        "-----",
        "---------------------",
        tname_hdrw = tnamew - 1,
    )
    .unwrap();

    for hit in &th.hits {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        let evalue = evalue_from_lnp(z, hit.lnp);
        writeln!(
            f,
            "{:<tnamew$}  {} {:>9} {:>3} {:>5} {}    {}",
            hit.name,
            hmmer_pure_rs::output::fmt_score(hit.score),
            hmmer_pure_rs::output::fmt_evalue(evalue),
            hit.ndom,
            hmmer_pure_rs::output::fmt_width5_1(hit.nexpected as f64),
            hmmer_pure_rs::output::fmt_bias(hit.bias),
            if hit.desc.is_empty() { "-" } else { &hit.desc },
        )
        .unwrap();
    }

    writeln!(f).unwrap();
    writeln!(f, "# Domain scores").unwrap();
    writeln!(f, "# -------------").unwrap();
    writeln!(f, "#").unwrap();
    writeln!(
        f,
        "# {:<tname_hdrw$} {:>6} {:>9} {:>5} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}     {}",
        " name",
        "bits",
        "E-value",
        "hit",
        "bias",
        "env-st",
        "env-en",
        "ali-st",
        "ali-en",
        "hmm-st",
        "hmm-en",
        "description",
        tname_hdrw = tnamew - 1,
    )
    .unwrap();
    writeln!(
        f,
        "# {:>tname_hdrw$} {:>6} {:>9} {:>5} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}      {}",
        "-------------------",
        "------",
        "---------",
        "-----",
        "-----",
        "------",
        "------",
        "------",
        "------",
        "------",
        "------",
        "---------------------",
        tname_hdrw = tnamew - 1,
    )
    .unwrap();

    // Build one pseudo-hit per *reported* domain, recording `ndom_reported`:
    // the 1-based ordinal of this domain among the reported domains of its
    // parent hit (C p7_tophits.c:1897-1906, `domhit->ndom = ndomReported`).
    // This is the value printed in the `hit` column — NOT the absolute dcl
    // index (which would skip over any unreported intervening domains).
    let mut reported_domains = Vec::new();
    for (hit_idx, hit) in th.hits.iter().enumerate() {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        let mut ndom_reported = 0usize;
        for (_dom_idx, dom) in hit.dcl.iter().enumerate() {
            if dom.is_reported {
                ndom_reported += 1;
                reported_domains.push((hit_idx, ndom_reported, hit, dom));
            }
        }
    }
    reported_domains.sort_by(|a, b| compare_pfamtblout_domain_pseudo_hits(a, b, pli));

    for (_hit_idx, ndom_reported, hit, dom) in reported_domains {
        let i_evalue = evalue_from_lnp(z, dom.lnp);
        let (hmmfrom, hmmto) = if let Some(ref ad) = dom.ad {
            (ad.hmmfrom, ad.hmmto)
        } else {
            (1, qlen)
        };
        writeln!(
            f,
            "{:<tnamew$}  {} {:>9} {:>5} {} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}     {}",
            hit.name,
            hmmer_pure_rs::output::fmt_score(dom.bitscore),
            hmmer_pure_rs::output::fmt_evalue(i_evalue),
            ndom_reported,
            hmmer_pure_rs::output::fmt_bias(dom.dombias),
            dom.ienv,
            dom.jenv,
            dom.iali,
            dom.jali,
            hmmfrom,
            hmmto,
            if hit.desc.is_empty() { "-" } else { &hit.desc },
        )
        .unwrap();
    }
}

fn compare_pfamtblout_domain_pseudo_hits(
    a: &(usize, usize, &Hit, &Domain),
    b: &(usize, usize, &Hit, &Domain),
    pli: &Pipeline,
) -> std::cmp::Ordering {
    // C p7_tophits.c:1910 keys the pfamtblout domain pseudo-hits purely on
    // pli->inc_by_E: `sortkey = inc_by_E ? -lnP : bitscore`, sorted descending.
    // Expressed as an ascending key (matching this module's convention): use
    // `lnp` when inc_by_e (asc lnp == desc -lnP) else `-bitscore` (asc -bits ==
    // desc bits). This differs from the shared `pli.hit_sortkey`, which keys on
    // the broader `score_sort_active()` predicate; we key locally on inc_by_e to
    // avoid touching pipeline.rs.
    let pfam_key = |bitscore: f32, lnp: f64| -> f64 {
        if pli.inc_by_e {
            lnp
        } else {
            -(bitscore as f64)
        }
    };
    let a_key = pfam_key(a.3.bitscore, a.3.lnp);
    let b_key = pfam_key(b.3.bitscore, b.3.lnp);
    a_key
        .partial_cmp(&b_key)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.2.name.cmp(&b.2.name))
        .then_with(|| {
            let a_dir = if a.3.iali < a.3.jali { 1 } else { -1 };
            let b_dir = if b.3.iali < b.3.jali { 1 } else { -1 };
            if a_dir != b_dir {
                b_dir.cmp(&a_dir)
            } else {
                a.3.iali.cmp(&b.3.iali)
            }
        })
}

pub fn write_table_footer<W: Write>(
    f: &mut W,
    program: &str,
    pipeline_mode: &str,
    query_file: &std::path::Path,
    target_file: &std::path::Path,
    cmdline: &str,
) {
    writeln!(f, "#").unwrap();
    writeln!(f, "# Program:         {}", program).unwrap();
    writeln!(f, "# Version:         3.4 (Aug 2023)").unwrap();
    writeln!(f, "# Pipeline mode:   {}", pipeline_mode).unwrap();
    writeln!(f, "# Query file:      {}", query_file.display()).unwrap();
    writeln!(f, "# Target file:     {}", target_file.display()).unwrap();
    writeln!(f, "# Option settings: {} ", cmdline).unwrap();
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| String::new());
    writeln!(f, "# Current dir:     {}", cwd).unwrap();
    writeln!(
        f,
        "# Date:            {}",
        hmmer_pure_rs::output::format_hmmer_date(std::time::SystemTime::now())
    )
    .unwrap();
    writeln!(f, "# [ok]").unwrap();
}

/// Build and write the `-A` multiple alignment of all included domains.
///
/// Faithful port of the `-A` block in `hmmer/src/hmmsearch.c:554-572` and
/// `hmmer/src/phmmer.c:620-638`: it calls `p7_tophits_Alignment(th, abc,
/// NULL,NULL, 0, p7_ALL_CONSENSUS_COLS, &msa)` to assemble a real
/// column-aligned MSA of every included domain (sequence names `name/from-to`,
/// `#=GS DE [subseq from]` lines, `#=GC RF`, `#=GR PP`, all aligned to the model
/// consensus columns), sets `#=GF ID/AC/DE/AU`, then writes Stockholm
/// (`textw > 0`) or Pfam (`textw == 0`). Returns the number of sequences in the
/// MSA when one was produced (so the caller can print the matching
/// `# Alignment of N hits ... saved to:` line), or `None` if no hits satisfy
/// the inclusion thresholds.
///
/// `name`/`acc`/`desc` are the per-tool GF annotation (hmmsearch: model
/// name/acc/desc; phmmer: the model/query name plus optional query acc/desc),
/// and `author` is e.g. `"hmmsearch (HMMER 3.4)"` / `"phmmer (HMMER 3.4)"`.
///
/// Reuses the same `tophits::included_alignment` + Stockholm writer that
/// `jackhmmer` uses; jackhmmer's writer emits a single, unwrapped block
/// (equivalent to C's Pfam `cpl == alen`). C wraps default Stockholm at 200
/// aligned residues per block, so for alignments wider than 200 columns the
/// Stockholm (`textw > 0`) blocking would differ; for the tool fixtures here
/// (alen <= ~175) it is byte-identical to C.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ali_output<W: Write>(
    f: &mut W,
    abc: &Alphabet,
    model_len: usize,
    name: &str,
    acc: Option<&str>,
    desc: Option<&str>,
    author: &str,
    th: &TopHits,
    _textw: usize,
) -> Option<usize> {
    let mut msa = hmmer_pure_rs::tophits::included_alignment(th, abc, model_len, None, name)?;
    msa.acc = acc.filter(|s| !s.is_empty()).map(|s| s.to_string());
    msa.desc = desc.filter(|s| !s.is_empty()).map(|s| s.to_string());
    msa.author = Some(author.to_string());
    let nseq = msa.nseq;
    crate::subcmd::jackhmmer::write_tophits_alignment_msa(f, &msa);
    Some(nseq)
}

fn open_target_seq_file(
    path: &Path,
    abc: &Alphabet,
    tformat: Option<&str>,
) -> hmmer_pure_rs::errors::HmmerResult<sequence::SeqFile<Box<dyn std::io::Read>>> {
    if let Some(format) = tformat {
        sequence::open_seq_file_with_format(path, abc, SequenceFormat::from_name(format).unwrap())
    } else {
        sequence::open_seq_file(path, abc)
    }
}

pub(crate) fn open_restricted_target_seq_file(
    path: &Path,
    abc: &Alphabet,
    tformat: Option<&str>,
    stkey: &str,
    ssifile: Option<&Path>,
) -> Result<sequence::SeqFile<Box<dyn std::io::Read>>, String> {
    if path == Path::new("-") {
        return Err("can't open an SSI index for standard input".to_string());
    }
    if path.extension().is_some_and(|ext| ext == "gz") {
        return Err("can't open an SSI index for a .gz compressed seq file".to_string());
    }
    let ssi_path = ssifile
        .map(Path::to_path_buf)
        .unwrap_or_else(|| hmmer_pure_rs::ssi::path_with_appended_suffix(path, ".ssi"));
    let offset = lookup_ssi_primary_offset(&ssi_path, stkey)?;
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("failed to open sequence file {}: {e}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("failed to seek sequence file {}: {e}", path.display()))?;
    let reader: Box<dyn Read> = Box::new(file);
    let sqf = sequence::SeqFile::new(reader, abc.clone());
    Ok(if let Some(format) = tformat {
        sqf.with_format(SequenceFormat::from_name(format).unwrap())
    } else {
        sqf
    })
}

fn lookup_ssi_primary_offset(ssi_path: &Path, key: &str) -> Result<u64, String> {
    let mut file = std::fs::File::open(ssi_path)
        .map_err(|e| format!("failed to open SSI index {}: {e}", ssi_path.display()))?;
    let magic = read_be_u32(&mut file)?;
    if magic != 0xd3d3c9b3 {
        return Err(format!("bad SSI magic in {}", ssi_path.display()));
    }
    let _flags = read_be_u32(&mut file)?;
    let offsz = read_be_u32(&mut file)?;
    let nfiles = read_be_u16(&mut file)?;
    let nprimary = read_be_u64(&mut file)?;
    let nsecondary = read_be_u64(&mut file)?;
    let flen = read_be_u32(&mut file)? as usize;
    let plen = read_be_u32(&mut file)? as usize;
    let slen = read_be_u32(&mut file)? as usize;
    let frecsize = read_be_u32(&mut file)? as usize;
    let precsize = read_be_u32(&mut file)? as usize;
    let srecsize = read_be_u32(&mut file)? as usize;
    let _foffset = read_be_offset(&mut file, offsz)?;
    let poffset = read_be_offset(&mut file, offsz)?;
    let soffset = read_be_offset(&mut file, offsz)?;
    if (offsz != 4 && offsz != 8) || nfiles != 1 {
        return Err(format!("unsupported SSI header in {}", ssi_path.display()));
    }
    if frecsize != flen + 16
        || precsize != plen + 2 + 2 * offsz as usize + 8
        || srecsize != slen + plen
    {
        return Err(format!(
            "SSI index {} has inconsistent record sizes",
            ssi_path.display()
        ));
    }

    let mut primary_offsets = std::collections::HashMap::new();
    file.seek(SeekFrom::Start(poffset))
        .map_err(|e| format!("failed to read SSI index {}: {e}", ssi_path.display()))?;
    for _ in 0..nprimary {
        let primary = read_fixed_string(&mut file, plen)?;
        let file_idx = read_be_u16(&mut file)?;
        let offset = read_be_offset(&mut file, offsz)?;
        let _data_offset = read_be_offset(&mut file, offsz)?;
        let _record_len = read_be_i64(&mut file)?;
        if file_idx == 0 {
            primary_offsets.insert(primary, offset);
        }
    }
    if let Some(offset) = primary_offsets.get(key).copied() {
        return Ok(offset);
    }

    file.seek(SeekFrom::Start(soffset))
        .map_err(|e| format!("failed to read SSI index {}: {e}", ssi_path.display()))?;
    for _ in 0..nsecondary {
        let secondary = read_fixed_string(&mut file, slen)?;
        let primary = read_fixed_string(&mut file, plen)?;
        if secondary == key {
            return primary_offsets.get(&primary).copied().ok_or_else(|| {
                format!(
                    "SSI index {} secondary key {key} references missing primary {primary}",
                    ssi_path.display()
                )
            });
        }
    }
    Err(format!(
        "sequence {key} not found in SSI index {}",
        ssi_path.display()
    ))
}

fn read_fixed_string<R: Read>(reader: &mut R, len: usize) -> Result<String, String> {
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("failed to read SSI index: {e}"))?;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(String::from_utf8_lossy(&buf[..end]).to_string())
}

fn read_be_u16<R: Read>(reader: &mut R) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("failed to read SSI index: {e}"))?;
    Ok(u16::from_be_bytes(buf))
}

fn read_be_u32<R: Read>(reader: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("failed to read SSI index: {e}"))?;
    Ok(u32::from_be_bytes(buf))
}

fn read_be_u64<R: Read>(reader: &mut R) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("failed to read SSI index: {e}"))?;
    Ok(u64::from_be_bytes(buf))
}

fn read_be_offset<R: Read>(reader: &mut R, offsz: u32) -> Result<u64, String> {
    match offsz {
        4 => read_be_u32(reader).map(u64::from),
        8 => read_be_u64(reader),
        _ => Err(format!("unsupported SSI offset size {offsz}")),
    }
}

fn read_be_i64<R: Read>(reader: &mut R) -> Result<i64, String> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("failed to read SSI index: {e}"))?;
    Ok(i64::from_be_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmmer_pure_rs::tophits::{P7_IS_INCLUDED, P7_IS_REPORTED};

    fn domain(bitscore: f32, reported: bool) -> Domain {
        Domain {
            iali: 1,
            jali: 10,
            ienv: 1,
            jenv: 10,
            bitscore,
            lnp: -1.0,
            dombias: 0.0,
            oasc: 0.0,
            envsc: 0.0,
            domcorrection: 0.0,
            is_reported: reported,
            is_included: reported,
            ad: None,
        }
    }

    fn domain_with_alignment(bitscore: f32, reported: bool, iali: i64, jali: i64) -> Domain {
        let mut dom = domain(bitscore, reported);
        dom.iali = iali;
        dom.jali = jali;
        dom.ienv = iali.min(jali);
        dom.jenv = iali.max(jali);
        dom
    }

    fn hit_with_name(name: &str, domains: Vec<Domain>, flags: u32) -> Hit {
        Hit {
            name: name.to_string(),
            acc: String::new(),
            desc: String::new(),
            n: 100,
            sortkey: -10.0,
            score: 10.0,
            bias: 0.0,
            pre_score: 10.0,
            sum_score: 10.0,
            lnp: -1.0,
            pre_lnp: -1.0,
            sum_lnp: -1.0,
            nexpected: 1.0,
            nregions: 0,
            nclustered: 0,
            noverlaps: 0,
            nenvelopes: domains.len(),
            ndom: domains.len(),
            nreported: domains.iter().filter(|dom| dom.is_reported).count(),
            nincluded: domains.iter().filter(|dom| dom.is_included).count(),
            dcl: domains,
            flags,
            seqidx: 0,
            subseq_start: 0,
        }
    }

    fn hit(domains: Vec<Domain>) -> Hit {
        hit_with_name("target", domains, P7_IS_REPORTED | P7_IS_INCLUDED)
    }

    #[test]
    fn best_domain_summary_uses_true_best_domain_even_if_unreported() {
        let hit = hit(vec![domain(40.0, false), domain(12.0, true)]);

        let best = best_domain(&hit).expect("domain");

        assert_eq!(best.bitscore, 40.0);
    }

    #[test]
    fn tblout_width_includes_unreported_hits() {
        let mut th = TopHits::new();
        th.hits.push(hit_with_name(
            "reported",
            vec![domain(12.0, true)],
            P7_IS_REPORTED | P7_IS_INCLUDED,
        ));
        th.hits.push(hit_with_name(
            "unreported-target-name-widens-columns",
            vec![domain(10.0, false)],
            0,
        ));

        let mut out = Vec::new();
        write_tblout(&mut out, "query", None, &th, 1.0, true);
        let out = String::from_utf8(out).unwrap();

        assert!(out.contains("reported                              -"));
    }

    #[test]
    fn pfamtblout_uses_dynamic_name_width_and_ndom_count() {
        let mut th = TopHits::new();
        th.hits.push(hit_with_name(
            "long-reported-target-name",
            vec![domain(20.0, true), domain(10.0, false)],
            P7_IS_REPORTED | P7_IS_INCLUDED,
        ));

        let mut out = Vec::new();
        let pli = Pipeline::new();
        write_pfamtblout_with_pipeline(&mut out, "query", None, 42, &th, &pli, 1.0, 1.0);
        let out = String::from_utf8(out).unwrap();

        assert!(out.contains("long-reported-target-name    10.0"));
        assert!(out.contains("long-reported-target-name    20.0"));
        assert!(out.lines().any(|line| {
            line.starts_with("long-reported-target-name")
                && line.split_whitespace().nth(3) == Some("2")
        }));
    }

    #[test]
    fn pfamtblout_domain_pseudo_hits_use_c_tie_breakers() {
        let mut th = TopHits::new();
        th.hits.push(hit_with_name(
            "target",
            vec![
                domain_with_alignment(20.0, true, 30, 20),
                domain_with_alignment(20.0, true, 5, 15),
            ],
            P7_IS_REPORTED | P7_IS_INCLUDED,
        ));

        let mut out = Vec::new();
        let pli = Pipeline::new();
        write_pfamtblout_with_pipeline(&mut out, "query", None, 42, &th, &pli, 1.0, 1.0);
        let out = String::from_utf8(out).unwrap();
        let rows: Vec<&str> = out
            .lines()
            .filter(|line| line.starts_with("target"))
            .collect();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1].split_whitespace().nth(3), Some("2"));
        assert_eq!(rows[2].split_whitespace().nth(3), Some("1"));
    }

    #[test]
    fn pfamtblout_domain_pseudo_hits_use_live_score_sort_mode() {
        let mut low_lnp_high_score = domain(50.0, true);
        low_lnp_high_score.lnp = -1.0;
        let mut high_lnp_low_score = domain(10.0, true);
        high_lnp_low_score.lnp = -100.0;

        let mut th = TopHits::new();
        th.hits.push(hit_with_name(
            "target",
            vec![low_lnp_high_score, high_lnp_low_score],
            P7_IS_REPORTED | P7_IS_INCLUDED,
        ));

        let mut out = Vec::new();
        let mut pli = Pipeline::new();
        pli.inc_t = Some(20.0);
        pli.inc_by_e = false;
        write_pfamtblout_with_pipeline(&mut out, "query", None, 42, &th, &pli, 1.0, 1.0);
        let out = String::from_utf8(out).unwrap();
        let rows: Vec<&str> = out
            .lines()
            .filter(|line| line.starts_with("target"))
            .collect();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1].split_whitespace().nth(1), Some("50.0"));
        assert_eq!(rows[2].split_whitespace().nth(1), Some("10.0"));
    }

    #[test]
    fn hmmsearch_accepts_negative_space_separated_f_values() {
        // C --F1/--F2/--F3 are eslARG_REAL with no range, so C accepts the
        // space-separated negative form. allow_hyphen_values keeps clap from
        // treating "-0.5" as an unknown flag.
        let args = Args::try_parse_from([
            "hmmsearch", "--F1", "-0.5", "--F2", "-1e-3", "--F3", "-2", "model.hmm", "targets.fa",
        ])
        .unwrap();
        assert_eq!(args.f1, -0.5);
        assert_eq!(args.f2, -1e-3);
        assert_eq!(args.f3, -2.0);
    }
}
