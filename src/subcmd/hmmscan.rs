//! hmmscan — search sequence(s) against an HMM database.
//! Reverses hmmsearch: query=sequences, targets=HMMs.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmmfile_binary;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::output::{fmt_g, fmt_width5_1};
use hmmer_pure_rs::pipeline::{BitCutoff, Pipeline};
use hmmer_pure_rs::pressed;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::sequence::{self, Sequence, SequenceFormat};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::TopHits;
use hmmer_pure_rs::util::cmath::c_exp_f64;

#[derive(Parser)]
#[command(
    name = "hmmscan",
    about = "Search sequence(s) against a profile HMM database"
)]
struct Args {
    /// Direct output to file <f>, not stdout
    #[arg(short = 'o')]
    output: Option<PathBuf>,

    /// HMM database file
    hmmdb: PathBuf,
    /// Query sequence file (FASTA)
    seqfile: PathBuf,

    /// Assert query sequence file format
    #[arg(long = "qformat")]
    qformat: Option<String>,

    /// Report profiles <= this E-value threshold
    #[arg(
        short = 'E',
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "score_threshold"
    )]
    e_value: f64,

    /// Report profiles >= this score threshold
    #[arg(short = 'T', conflicts_with = "e_value")]
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
    #[arg(long = "domT", conflicts_with = "dom_e")]
    dom_t: Option<f32>,

    /// Include profiles <= this E-value threshold
    #[arg(
        long = "incE",
        default_value = "0.01",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_t"
    )]
    inc_e: f64,

    /// Include profiles >= this score threshold
    #[arg(long = "incT", conflicts_with = "inc_e")]
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
    #[arg(long = "incdomT", conflicts_with = "inc_dome")]
    inc_dom_t: Option<f32>,

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

    /// Turn all heuristic filters off
    #[arg(long = "max", conflicts_with_all = ["f1", "f2", "f3", "nobias"])]
    max: bool,

    /// MSV filter threshold
    #[arg(long = "F1", default_value = "0.02")]
    f1: f64,

    /// Viterbi filter threshold
    #[arg(long = "F2", default_value = "0.001")]
    f2: f64,

    /// Forward filter threshold
    #[arg(long = "F3", default_value = "1e-5")]
    f3: f64,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    nobias: bool,

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    nonull2: bool,

    /// Random number seed
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "0")]
    cpu: usize,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    domtblout: Option<PathBuf>,

    /// Save Pfam-style table of hits and domains
    #[arg(long = "pfamtblout")]
    pfamtblout: Option<PathBuf>,

    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    acc: bool,

    /// Do not output alignments
    #[arg(long = "noali")]
    noali: bool,

    /// Unlimit ASCII text output line width
    #[arg(long = "notextw", conflicts_with = "textw")]
    notextw: bool,

    /// Set max width of ASCII text output lines
    #[arg(long = "textw", value_parser = parse_textw)]
    textw: Option<usize>,

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    domz_value: Option<f64>,
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

/// Entry point for `hmmscan`: scan each query sequence against every HMM in a
/// database and report a ranked, E-value-thresholded hit table.
///
/// Per-query, profiles are built once-per-HMM in parallel via rayon, the
/// pipeline (MSV -> Viterbi -> Forward + domain decoding) is run, the best
/// hit per HMM is collected, and the top-hits list is sorted, thresholded, and
/// printed in the canonical HMMER per-sequence tabular layout. Corresponds to
/// `serial_master()` in hmmer/src/hmmscan.c. MPI is omitted; pressed HMM
/// databases are required and read through the `.h3*` sidecar path.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let args = hmmer_pure_rs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);

    validate_sequence_format("hmmscan --qformat", args.qformat.as_deref());

    logsum::p7_flogsuminit();

    let use_parallel = args.cpu > 1;
    if use_parallel {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.cpu)
            .start_handler(|_| hmmer_pure_rs::util::simd_env::init())
            .build_global()
            .ok();
    }

    // C hmmscan requires a pressed HMM database and refuses plain ASCII HMMs.
    let pressed_available =
        pressed::pressed_db_sidecars_complete(&args.hmmdb).unwrap_or_else(|e| {
            eprintln!("Error opening pressed HMM database: {}", e);
            std::process::exit(1);
        });
    if !pressed_available {
        eprintln!(
            "Error opening pressed HMM database: {} is not pressed; use hmmpress first",
            args.hmmdb.display()
        );
        std::process::exit(1);
    }
    let h3m_path = pressed::pressed_h3m_path(&args.hmmdb);
    let hmms = hmmfile_binary::read_binary_hmm_file(&h3m_path).unwrap_or_else(|e| {
        eprintln!("Error reading HMM database: {}", e);
        std::process::exit(1);
    });
    let pressed_oprofiles = pressed::read_pressed_oprofiles(&args.hmmdb).unwrap_or_else(|e| {
        eprintln!("Error reading pressed optimized profiles: {}", e);
        std::process::exit(1);
    });
    if let Err(e) = pressed::validate_pressed_oprofiles_match_hmms(&hmms, &pressed_oprofiles) {
        eprintln!("Error reading pressed optimized profiles: {}", e);
        std::process::exit(1);
    };
    let bit_cutoff = selected_bit_cutoff(&args);
    if let Some(cutoff) = bit_cutoff {
        for hmm in &hmms {
            let mut pli = Pipeline::new();
            pli.use_bit_cutoffs = cutoff;
            if let Err(e) = pli.new_model_thresholds(&hmm.cutoff) {
                eprintln!("Error: {} for model {}", e, hmm.name);
                std::process::exit(1);
            }
        }
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

    let mut output_file = args.output.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open output file {path} for writing",
        )
    });
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let out: &mut dyn Write = match output_file {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };

    writeln!(
        out,
        "# hmmscan :: search sequence(s) against a profile database"
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
        writeln!(out, "# input seqfile format asserted:   {format}").unwrap();
    }
    writeln!(
        out,
        "# target HMM database:             {}",
        args.hmmdb.display()
    )
    .unwrap();
    if let Some(ref path) = args.output {
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
    if args.acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
    }
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    if command_line_has_option(&cmdline, "--cpu") || std::env::var_os("HMMER_NCPU").is_some() {
        if args.cpu == 0 {
            writeln!(out, "# multithread parallelization:     off").unwrap();
        } else {
            writeln!(
                out,
                "# multithread parallelization:     {} workers",
                args.cpu
            )
            .unwrap();
        }
    }
    if args.notextw {
        writeln!(out, "# max ASCII text line length:      unlimited").unwrap();
    }
    if let Some(textw) = args.textw {
        writeln!(out, "# max ASCII text line length:      {}", textw).unwrap();
    }
    write_hmmscan_option_header(out, &args, &cmdline);
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();

    // For each query sequence, search all HMMs
    let abc = alphabet_from_hmms(&hmms).unwrap_or_else(|e| {
        eprintln!("Error reading HMM database: {}", e);
        std::process::exit(1);
    });
    let bg = Bg::new(&abc);

    let mut sqf =
        open_query_seq_file(&args.seqfile, &abc, args.qformat.as_deref()).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });

    let mut sq = Sequence::new();
    let mut query_idx = 0usize;
    while sqf.read(&mut sq).unwrap_or_else(|e| {
        eprintln!("Error reading sequence file: {}", e);
        std::process::exit(1);
    }) {
        query_idx += 1;
        writeln!(out, "Query:       {}  [L={}]", sq.name, sq.n).unwrap();
        if !sq.acc.is_empty() {
            writeln!(out, "Accession:   {}", sq.acc).unwrap();
        }
        if !sq.desc.is_empty() {
            writeln!(out, "Description: {}", sq.desc).unwrap();
        }

        let scored: Vec<(Option<hmmer_pure_rs::tophits::Hit>, PipelineStats)> = if !use_parallel {
            hmms.iter()
                .enumerate()
                .map(|(idx, hmm)| {
                    score_hmmscan_model(
                        hmm,
                        &pressed_oprofiles[idx],
                        &bg,
                        &abc,
                        &sq,
                        &args,
                        bit_cutoff,
                    )
                })
                .collect()
        } else {
            use rayon::prelude::*;
            hmms.par_iter()
                .enumerate()
                .map(|(idx, hmm)| {
                    score_hmmscan_model(
                        hmm,
                        &pressed_oprofiles[idx],
                        &bg,
                        &abc,
                        &sq,
                        &args,
                        bit_cutoff,
                    )
                })
                .collect()
        };

        let mut th = TopHits::new();
        let mut stats = PipelineStats::default();
        th.hits = scored
            .into_iter()
            .filter_map(|(hit, s)| {
                stats.n_past_msv += s.n_past_msv;
                stats.n_past_bias += s.n_past_bias;
                stats.n_past_vit += s.n_past_vit;
                stats.n_past_fwd += s.n_past_fwd;
                hit
            })
            .collect();
        let z = args.z_value.unwrap_or(hmms.len() as f64);
        th.sort_by_sortkey();
        let mut output_pli = Pipeline::new();
        configure_thresholds(&mut output_pli, &args);
        if let Some(cutoff) = bit_cutoff {
            output_pli.use_bit_cutoffs = cutoff;
        }
        {
            th.threshold(&output_pli, z, z);
        }
        let domz = args.domz_value.unwrap_or(th.nreported as f64);
        if domz != z {
            th.threshold(&output_pli, z, domz);
        }

        if let Some(ref mut f) = tblout_file {
            crate::subcmd::hmmsearch::write_tblout(
                f,
                &sq.name,
                Some(&sq.acc),
                &th,
                z,
                query_idx == 1,
            );
        }
        if let Some(ref mut f) = domtblout_file {
            crate::subcmd::hmmsearch::write_domtblout(
                f,
                &sq.name,
                Some(&sq.acc),
                sq.n,
                &th,
                z,
                domz,
                query_idx == 1,
            );
        }
        if let Some(ref mut f) = pfamtblout_file {
            crate::subcmd::hmmsearch::write_pfamtblout_with_pipeline(
                f,
                &sq.name,
                Some(&sq.acc),
                sq.n,
                &th,
                &output_pli,
                z,
                domz,
            );
        }

        let textw = if args.notextw {
            0
        } else {
            args.textw.unwrap_or(120)
        };
        write_hmmscan_score_table(out, &th, z, args.acc, textw);
        write_hmmscan_domain_annotation(
            out, &th, z, domz, &sq.name, sq.n, &hmms, args.acc, args.noali, textw,
        );
        write_scan_pipeline_stats(
            out,
            hmms.len() as u64,
            hmms.iter().map(|hmm| hmm.m as u64).sum(),
            sq.n as u64,
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

        sq.reuse();
    }

    if let Some(ref mut f) = tblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "hmmscan",
            "SCAN",
            &args.seqfile,
            &args.hmmdb,
            &cmdline,
        );
    }
    if let Some(ref mut f) = domtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "hmmscan",
            "SCAN",
            &args.seqfile,
            &args.hmmdb,
            &cmdline,
        );
    }
    if let Some(ref mut f) = pfamtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "hmmscan",
            "SEARCH",
            &args.seqfile,
            &args.hmmdb,
            &cmdline,
        );
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

fn display_name(hit: &hmmer_pure_rs::tophits::Hit, prefer_acc: bool) -> &str {
    if prefer_acc && !hit.acc.is_empty() {
        &hit.acc
    } else {
        &hit.name
    }
}

fn write_hmmscan_option_header(out: &mut dyn Write, args: &Args, cmdline: &str) {
    if command_line_has_option(cmdline, "-E") {
        writeln!(
            out,
            "# profile reporting threshold:     E-value <= {}",
            fmt_g(args.e_value)
        )
        .unwrap();
    }
    if let Some(score) = args.score_threshold {
        writeln!(
            out,
            "# profile reporting threshold:     score >= {}",
            fmt_g(score as f64)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--domE") {
        writeln!(
            out,
            "# domain reporting threshold:      E-value <= {}",
            fmt_g(args.dom_e)
        )
        .unwrap();
    }
    if let Some(score) = args.dom_t {
        writeln!(
            out,
            "# domain reporting threshold:      score >= {}",
            fmt_g(score as f64)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--incE") {
        writeln!(
            out,
            "# profile inclusion threshold:     E-value <= {}",
            fmt_g(args.inc_e)
        )
        .unwrap();
    }
    if let Some(score) = args.inc_t {
        writeln!(
            out,
            "# profile inclusion threshold:     score >= {}",
            fmt_g(score as f64)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--incdomE") {
        writeln!(
            out,
            "# domain inclusion threshold:      E-value <= {}",
            fmt_g(args.inc_dome)
        )
        .unwrap();
    }
    if let Some(score) = args.inc_dom_t {
        writeln!(
            out,
            "# domain inclusion threshold:      score >= {}",
            fmt_g(score as f64)
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
        writeln!(out, "# MSV filter P threshold:       <= {}", fmt_g(args.f1)).unwrap();
    }
    if command_line_has_option(cmdline, "--F2") {
        writeln!(out, "# Vit filter P threshold:       <= {}", fmt_g(args.f2)).unwrap();
    }
    if command_line_has_option(cmdline, "--F3") {
        writeln!(out, "# Fwd filter P threshold:       <= {}", fmt_g(args.f3)).unwrap();
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

fn open_query_seq_file(
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

fn truncate_for_textw(s: &str, width: usize) -> String {
    if width == 0 || s.chars().count() <= width {
        return s.to_string();
    }
    s.chars().take(width).collect()
}

fn write_hmmscan_score_table<W: Write + ?Sized>(
    out: &mut W,
    th: &TopHits,
    z: f64,
    prefer_acc: bool,
    textw: usize,
) {
    use hmmer_pure_rs::tophits::P7_IS_REPORTED;

    let model_namew = th
        .hits
        .iter()
        .map(|hit| display_name(hit, prefer_acc).len())
        .max()
        .unwrap_or(0)
        .max(8);
    let descw = if textw > 0 {
        textw.saturating_sub(model_namew + 61).max(32)
    } else {
        0
    };

    writeln!(
        out,
        "Scores for complete sequence (score includes all domains):"
    )
    .unwrap();
    writeln!(
        out,
        "   --- full sequence ---   --- best 1 domain ---    -#dom-"
    )
    .unwrap();
    writeln!(
        out,
        "    E-value  score  bias    E-value  score  bias    exp  N  {:<model_namew$} Description",
        "Model"
    )
    .unwrap();
    writeln!(
        out,
        "    ------- ------ -----    ------- ------ -----   ---- --  {:<model_namew$} -----------",
        "--------"
    )
    .unwrap();

    let mut any_reported = false;
    let mut have_printed_incthresh = false;
    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED == 0 && !have_printed_incthresh {
            writeln!(out, "  ------ inclusion threshold ------").unwrap();
            have_printed_incthresh = true;
        }
        any_reported = true;
        let evalue = z * c_exp_f64(hit.lnp);
        let best_dom = crate::subcmd::hmmsearch::best_domain(hit);
        let dom_evalue = best_dom.map(|d| z * c_exp_f64(d.lnp)).unwrap_or(evalue);
        let dom_score = best_dom.map(|d| d.bitscore).unwrap_or(hit.score);
        let dom_bias = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);
        writeln!(
            out,
            "  {} {} {}  {} {} {}  {} {:2}  {:<model_namew$}  {}",
            hmmer_pure_rs::output::fmt_evalue(evalue),
            hmmer_pure_rs::output::fmt_score(hit.score),
            hmmer_pure_rs::output::fmt_bias(hit.bias),
            hmmer_pure_rs::output::fmt_evalue(dom_evalue),
            hmmer_pure_rs::output::fmt_score(dom_score),
            hmmer_pure_rs::output::fmt_bias(dom_bias),
            fmt_width5_1(hit.nexpected as f64),
            hit.nreported,
            display_name(hit, prefer_acc),
            truncate_for_textw(&hit.desc, descw),
        )
        .unwrap();
    }

    if !any_reported {
        writeln!(
            out,
            "\n   [No targets detected that satisfy reporting thresholds]"
        )
        .unwrap();
    }
}

fn write_hmmscan_domain_annotation(
    out: &mut dyn Write,
    th: &TopHits,
    z: f64,
    domz: f64,
    qname: &str,
    qlen: usize,
    hmms: &[hmmer_pure_rs::Hmm],
    prefer_acc: bool,
    noali: bool,
    textw: usize,
) {
    use hmmer_pure_rs::tophits::P7_IS_REPORTED;

    writeln!(out).unwrap();
    writeln!(out).unwrap();
    if noali {
        writeln!(out, "Domain annotation for each model:").unwrap();
    } else {
        writeln!(out, "Domain annotation for each model (and alignments):").unwrap();
    }

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }

        let desc = if hit.desc.is_empty() { "-" } else { &hit.desc };
        writeln!(out, ">> {}  {}", display_name(hit, prefer_acc), desc).unwrap();
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
            let indicator = if dom.is_included { '!' } else { '?' };
            let (hf, ht) = if let Some(ref ad) = dom.ad {
                (ad.hmmfrom, ad.hmmto)
            } else {
                (1, hit.n)
            };
            let hmm_left = if hf == 1 { '[' } else { '.' };
            let hmm_right = if ht == hit.n { ']' } else { '.' };
            let ali_left = if dom.iali == 1 { '[' } else { '.' };
            let ali_right = if dom.jali == qlen as i64 { ']' } else { '.' };
            let env_left = if dom.ienv == 1 { '[' } else { '.' };
            let env_right = if dom.jenv == qlen as i64 { ']' } else { '.' };
            let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);
            writeln!(
                out,
                " {:3} {} {} {} {:>9} {:>9} {:7} {:7} {}{} {:7} {:7} {}{} {:7} {:7} {}{} {}",
                reported_idx,
                indicator,
                hmmer_pure_rs::output::fmt_score(dom.bitscore),
                hmmer_pure_rs::output::fmt_bias(dom.dombias),
                hmmer_pure_rs::output::fmt_evalue(domz * c_exp_f64(dom.lnp)),
                hmmer_pure_rs::output::fmt_evalue(z * c_exp_f64(dom.lnp)),
                hf,
                ht,
                hmm_left,
                hmm_right,
                dom.iali,
                dom.jali,
                ali_left,
                ali_right,
                dom.ienv,
                dom.jenv,
                env_left,
                env_right,
                hmmer_pure_rs::output::fmt_fixed2(acc as f64),
            )
            .unwrap();
        }
        writeln!(out).unwrap();

        if !noali {
            writeln!(out, "  Alignments for each domain:").unwrap();
            let model_cs = hmmscan_model_cs_for_hit(hmms, hit);
            let mut reported_idx = 0usize;
            for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
                reported_idx += 1;
                writeln!(
                    out,
                    "  == domain {}  score: {} bits;  conditional E-value: {}",
                    reported_idx,
                    hmmer_pure_rs::output::fmt_fixed1(dom.bitscore as f64),
                    hmmer_pure_rs::output::fmt_evalue(domz * c_exp_f64(dom.lnp)).trim()
                )
                .unwrap();

                if let Some(ref ad) = dom.ad {
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
                    hmmer_pure_rs::tophits::print_alidisplay_blocks(
                        out,
                        display_name(hit, prefer_acc),
                        qname,
                        ad,
                        cs_line.as_deref(),
                        textw,
                    );
                }
                writeln!(out).unwrap();
            }
        }
    }
    writeln!(out).unwrap();
}

fn hmmscan_model_cs_for_hit<'a>(
    hmms: &'a [hmmer_pure_rs::Hmm],
    hit: &hmmer_pure_rs::tophits::Hit,
) -> Option<&'a [u8]> {
    hmms.iter()
        .find(|hmm| {
            hmm.name == hit.name
                || hmm
                    .acc
                    .as_deref()
                    .is_some_and(|acc| !hit.acc.is_empty() && acc == hit.acc)
        })
        .and_then(|hmm| hmm.cs.as_deref())
}

fn score_hmmscan_model(
    hmm: &hmmer_pure_rs::Hmm,
    oprofile: &OProfile,
    bg: &Bg,
    abc: &Alphabet,
    sq: &Sequence,
    args: &Args,
    bit_cutoff: Option<BitCutoff>,
) -> (Option<hmmer_pure_rs::tophits::Hit>, PipelineStats) {
    let mut local_bg = bg.clone();
    local_bg.set_filter(hmm.m, &hmm.compo);
    local_bg.set_length(sq.n);

    let mut gm = Profile::new(hmm.m, abc);
    profile::profile_config(hmm, &local_bg, &mut gm, sq.n as i32, P7_LOCAL);
    let mut om = oprofile.clone();

    let mut pli = Pipeline::new();
    pli.new_model(&gm);
    configure_thresholds(&mut pli, args);
    configure_acceleration(&mut pli, args);
    if let Some(z) = args.z_value {
        pli.z = z;
        pli.z_setby = hmmer_pure_rs::pipeline::ZSetBy::Option;
    }
    if let Some(domz) = args.domz_value {
        pli.domz = domz;
        pli.domz_setby = hmmer_pure_rs::pipeline::ZSetBy::Option;
    }
    if let Some(cutoff) = bit_cutoff {
        pli.use_bit_cutoffs = cutoff;
        if pli.new_model_thresholds(&hmm.cutoff).is_err() {
            return (None, PipelineStats::default());
        }
    }
    pli.do_alignment = true;
    pli.do_alignment_display = !args.noali;

    let mut th = TopHits::new();
    let hit = if pli.run(&mut gm, &mut om, &local_bg, hmm, sq, &mut th) {
        th.hits.into_iter().next().map(|mut hit| {
            hit.name = hmm.name.clone();
            hit.acc = hmm.acc.clone().unwrap_or_default();
            hit.desc = hmm.desc.clone().unwrap_or_default();
            hit.n = hmm.m;
            hit
        })
    } else {
        None
    };
    (
        hit,
        PipelineStats {
            n_past_msv: pli.n_past_msv,
            n_past_bias: pli.n_past_bias,
            n_past_vit: pli.n_past_vit,
            n_past_fwd: pli.n_past_fwd,
        },
    )
}

#[derive(Default)]
struct PipelineStats {
    n_past_msv: u64,
    n_past_bias: u64,
    n_past_vit: u64,
    n_past_fwd: u64,
}

fn write_scan_pipeline_stats<W: Write + ?Sized>(
    out: &mut W,
    n_targets: u64,
    total_nodes: u64,
    query_residues: u64,
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

    writeln!(out).unwrap();
    writeln!(out, "Internal pipeline statistics summary:").unwrap();
    writeln!(out, "-------------------------------------").unwrap();
    writeln!(
        out,
        "Query sequence(s):               {:>11}  ({} residues searched)",
        1, query_residues
    )
    .unwrap();
    writeln!(
        out,
        "Target model(s):                 {:>11}  ({} nodes)",
        n_targets, total_nodes
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

fn alphabet_from_hmms(hmms: &[hmmer_pure_rs::Hmm]) -> Result<Alphabet, String> {
    let hmm = hmms
        .first()
        .ok_or_else(|| "pressed HMM database contains no models".to_string())?;
    Ok(Alphabet::new(hmm.abc_type))
}

fn selected_bit_cutoff(args: &Args) -> Option<BitCutoff> {
    if args.cut_ga {
        Some(BitCutoff::GA)
    } else if args.cut_tc {
        Some(BitCutoff::TC)
    } else if args.cut_nc {
        Some(BitCutoff::NC)
    } else {
        None
    }
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
}

fn configure_acceleration(pli: &mut Pipeline, args: &Args) {
    pli.do_max = args.max;
    pli.f1 = args.f1;
    pli.f2 = args.f2;
    pli.f3 = args.f3;
    pli.do_biasfilter = !args.nobias;
    pli.do_null2 = !args.nonull2;
    pli.seed = args.seed;
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmmer_pure_rs::alphabet::AlphabetType;
    use hmmer_pure_rs::Hmm;

    #[test]
    fn hmmscan_positionals_match_c_order() {
        let args = Args::try_parse_from(["hmmscan", "models.hmm", "queries.fa"]).unwrap();

        assert_eq!(args.hmmdb, PathBuf::from("models.hmm"));
        assert_eq!(args.seqfile, PathBuf::from("queries.fa"));
    }

    #[test]
    fn hmmscan_parses_c_output_options() {
        let args = Args::try_parse_from([
            "hmmscan",
            "-o",
            "out.txt",
            "--tblout",
            "hits.tbl",
            "--domtblout",
            "domains.tbl",
            "--pfamtblout",
            "pfam.tbl",
            "--acc",
            "--noali",
            "--textw",
            "140",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();

        assert_eq!(args.output, Some(PathBuf::from("out.txt")));
        assert_eq!(args.tblout, Some(PathBuf::from("hits.tbl")));
        assert_eq!(args.domtblout, Some(PathBuf::from("domains.tbl")));
        assert_eq!(args.pfamtblout, Some(PathBuf::from("pfam.tbl")));
        assert!(args.acc);
        assert!(args.noali);
        assert_eq!(args.textw, Some(140));
    }

    #[test]
    fn hmmscan_parses_model_specific_cutoff_options() {
        let args =
            Args::try_parse_from(["hmmscan", "--cut_ga", "models.hmm", "queries.fa"]).unwrap();
        assert!(args.cut_ga);
        assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::GA));

        let args =
            Args::try_parse_from(["hmmscan", "--cut_tc", "models.hmm", "queries.fa"]).unwrap();
        assert!(args.cut_tc);
        assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::TC));

        let args =
            Args::try_parse_from(["hmmscan", "--cut_nc", "models.hmm", "queries.fa"]).unwrap();
        assert!(args.cut_nc);
        assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::NC));
    }

    #[test]
    fn hmmscan_parses_c_threshold_options() {
        let args = Args::try_parse_from([
            "hmmscan",
            "-T",
            "42",
            "--domT",
            "3",
            "--incT",
            "40",
            "--incdomT",
            "2",
            "-Z",
            "99",
            "--domZ",
            "7",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();
        assert_eq!(args.score_threshold, Some(42.0));
        assert_eq!(args.dom_t, Some(3.0));
        assert_eq!(args.inc_t, Some(40.0));
        assert_eq!(args.inc_dom_t, Some(2.0));
        assert_eq!(args.z_value, Some(99.0));
        assert_eq!(args.domz_value, Some(7.0));
    }

    #[test]
    fn hmmscan_parses_c_acceleration_options() {
        let args = Args::try_parse_from([
            "hmmscan",
            "--max",
            "--nonull2",
            "--seed",
            "7",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();
        assert!(args.max);
        assert!(args.nonull2);
        assert_eq!(args.seed, 7);

        let args = Args::try_parse_from([
            "hmmscan",
            "--F1",
            "0.1",
            "--F2",
            "0.2",
            "--F3",
            "0.3",
            "--nobias",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();
        assert_eq!(args.f1, 0.1);
        assert_eq!(args.f2, 0.2);
        assert_eq!(args.f3, 0.3);
        assert!(args.nobias);
    }

    #[test]
    fn hmmscan_uses_pressed_database_alphabet() {
        let hmms = vec![Hmm::new(4, AlphabetType::Dna, 4)];
        let abc = alphabet_from_hmms(&hmms).unwrap();
        assert_eq!(abc.abc_type, AlphabetType::Dna);
    }
}
