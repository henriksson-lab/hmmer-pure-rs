//! nhmmscan — search sequence(s) against an HMM database.
//! Reverses hmmsearch: query=sequences, targets=HMMs.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::builder::{self, DEFAULT_WINDOW_BETA};
use hmmer_pure_rs::hmmfile_binary;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::output::{fmt_bias, fmt_fixed0, fmt_g, fmt_score};
use hmmer_pure_rs::pipeline::{BitCutoff, Pipeline};
use hmmer_pure_rs::pressed;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::sequence::{self, Sequence, SequenceFormat};
use hmmer_pure_rs::tophits::TopHits;
use hmmer_pure_rs::util::cmath::c_exp_f64;

#[derive(Parser)]
#[command(
    name = "nhmmscan",
    about = "Search nucleotide sequence(s) against a DNA HMM database"
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

    /// Report models <= this E-value threshold
    #[arg(
        short = 'E',
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "score_threshold"
    )]
    e_value: f64,

    /// Report models >= this score threshold
    #[arg(short = 'T', conflicts_with = "e_value", allow_hyphen_values = true)]
    score_threshold: Option<f64>,

    /// Include models <= this E-value threshold
    #[arg(
        long = "incE",
        default_value = "0.01",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_t"
    )]
    inc_e: f64,

    /// Include models >= this score threshold
    #[arg(long = "incT", conflicts_with = "inc_e", allow_hyphen_values = true)]
    inc_t: Option<f64>,

    /// Use model's GA gathering cutoffs to set all thresholding
    #[arg(
        long = "cut_ga",
        conflicts_with_all = ["cut_tc", "cut_nc", "e_value", "score_threshold", "inc_e", "inc_t"]
    )]
    cut_ga: bool,

    /// Use model's NC noise cutoffs to set all thresholding
    #[arg(
        long = "cut_nc",
        conflicts_with_all = ["cut_ga", "cut_tc", "e_value", "score_threshold", "inc_e", "inc_t"]
    )]
    cut_nc: bool,

    /// Use model's TC trusted cutoffs to set all thresholding
    #[arg(
        long = "cut_tc",
        conflicts_with_all = ["cut_ga", "cut_nc", "e_value", "score_threshold", "inc_e", "inc_t"]
    )]
    cut_tc: bool,

    /// Turn all heuristic filters off (less speed, more power)
    #[arg(long = "max", conflicts_with_all = ["f1", "f2", "f3", "nobias"])]
    max: bool,

    /// Stage 1 (SSV) threshold
    #[arg(long = "F1", default_value = "0.02", allow_hyphen_values = true)]
    f1: f64,

    /// Stage 2 (Vit) threshold
    #[arg(long = "F2", default_value = "3e-3", allow_hyphen_values = true)]
    f2: f64,

    /// Stage 3 (Fwd) threshold
    #[arg(long = "F3", default_value = "3e-5", allow_hyphen_values = true)]
    f3: f64,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    nobias: bool,

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    nonull2: bool,

    /// Set number of comparisons done for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Random number seed
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Only search the top strand
    #[arg(long = "watson", conflicts_with = "crick")]
    watson: bool,

    /// Only search the bottom strand
    #[arg(long = "crick", conflicts_with = "watson")]
    crick: bool,

    /// Window length (max expected hit length)
    #[arg(long = "w_length", value_parser = parse_window_length)]
    w_length: Option<i32>,

    /// Tail mass for deriving window length
    #[arg(long = "w_beta", value_parser = parse_window_beta)]
    w_beta: Option<f64>,

    /// Window length for biased-composition modifier at MSV stage
    #[arg(long = "B1", default_value = "110", conflicts_with_all = ["max", "nobias"], hide = true)]
    b1: usize,

    /// Window length for biased-composition modifier at Viterbi stage
    #[arg(long = "B2", default_value = "240", conflicts_with_all = ["max", "nobias"], hide = true)]
    b2: usize,

    /// Window length for biased-composition modifier at Forward stage
    #[arg(long = "B3", default_value = "1000", conflicts_with_all = ["max", "nobias"], hide = true)]
    b3: usize,

    /// Override default background probabilities; accepted for compatibility
    #[arg(long = "bgfile", hide = true)]
    bgfile: Option<PathBuf>,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "0")]
    cpu: usize,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save Dfam-style table of hits and domains
    #[arg(long = "dfamtblout")]
    dfamtblout: Option<PathBuf>,

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

fn parse_window_length(s: &str) -> Result<i32, String> {
    let value = s
        .parse::<i32>()
        .map_err(|e| format!("invalid window length: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("--w_length must be > 0 and fit in a 32-bit signed integer".to_string())
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

/// Entry point for `nhmmscan`: scan each nucleotide query against every HMM in
/// a DNA HMM database and report a ranked, E-value-thresholded hit table.
///
/// Per-query, profiles are built once-per-HMM in parallel via rayon and scored
/// through the nhmmer long-target path on the requested strand(s). Hits are
/// adjusted into scan E-value space, deduplicated by model/alignment position,
/// thresholded, and printed with HMM names in the target column. Corresponds
/// to `serial_master()`/`serial_loop()` in `hmmer/src/nhmmscan.c`.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let args = hmmer_pure_rs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);
    validate_sequence_format("nhmmscan --qformat", args.qformat.as_deref());
    if matches!(args.w_length, Some(0..=3)) {
        eprintln!("Invalid window length value");
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

    // C nhmmscan requires a pressed HMM database and refuses plain ASCII HMMs.
    let pressed_available = pressed::pressed_db_available(&args.hmmdb).unwrap_or_else(|e| {
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
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating tblout file: {}", e);
            std::process::exit(1);
        })
    });
    let mut dfamtblout_file = args.dfamtblout.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating dfamtblout file: {}", e);
            std::process::exit(1);
        })
    });

    let mut output_file = args.output.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating output file: {}", e);
            std::process::exit(1);
        })
    });
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let out: &mut dyn Write = match output_file {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };

    writeln!(
        out,
        "# nhmmscan :: search DNA sequence(s) against a DNA profile database"
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
    if let Some(w_beta) = args.w_beta {
        writeln!(out, "# window length beta value:        {}", w_beta).unwrap();
    }
    if let Some(w_length) = args.w_length {
        writeln!(out, "# window length :                  {}", w_length).unwrap();
    }
    if args.b1 != 110 {
        writeln!(out, "# biased comp MSV window len:      {}", args.b1).unwrap();
    }
    if args.b2 != 240 {
        writeln!(out, "# biased comp Viterbi window len:  {}", args.b2).unwrap();
    }
    if args.b3 != 1000 {
        writeln!(out, "# biased comp Forward window len:  {}", args.b3).unwrap();
    }
    if let Some(path) = &args.bgfile {
        writeln!(out, "# file with custom bg probs:       {}", path.display()).unwrap();
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
    if let Some(ref path) = args.dfamtblout {
        writeln!(out, "# hits output in Dfam format:      {}", path.display()).unwrap();
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
    if let Some(z) = args.z_value {
        writeln!(out, "# sequence search space set to:    {}", fmt_fixed0(z)).unwrap();
    }
    write_nhmmscan_option_header(out, &args, &cmdline);
    if command_line_has_option(&cmdline, "--seed") {
        if args.seed == 0 {
            writeln!(out, "# random number seed:              one-time arbitrary").unwrap();
        } else {
            writeln!(out, "# random number seed set to:       {}", args.seed).unwrap();
        }
    }
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();

    // For each query sequence, search all HMMs
    let abc = nucleotide_alphabet_from_hmms(&hmms).unwrap_or_else(|e| {
        eprintln!("Error reading HMM database: {}", e);
        std::process::exit(1);
    });
    let mut bg = Bg::new(&abc);
    if let Some(path) = &args.bgfile {
        if let Err(e) = bg.read_file(&abc, path) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    let mut sqf =
        open_query_seq_file(&args.seqfile, &abc, args.qformat.as_deref()).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });

    let mut sq = Sequence::new();
    let mut query_idx = 0usize;
    let search_space_ln = hmmer_pure_rs::util::cmath::c_log_f64(hmms.len() as f64);
    while sqf.read(&mut sq).unwrap_or_else(|e| {
        eprintln!("Error reading sequence file: {}", e);
        std::process::exit(1);
    }) {
        query_idx += 1;
        writeln!(out, "Query:       {}  [L={}]", sq.name, sq.n).unwrap();

        let scan_model = |idx: usize, hmm: &hmmer_pure_rs::hmm::Hmm| {
            hmmer_pure_rs::util::simd_env::init();
            let mut local_bg = bg.clone();
            local_bg.set_filter(hmm.m, &hmm.compo);
            local_bg.set_length(sq.n);

            let mut gm = Profile::new(hmm.m, &abc);
            profile::profile_config(hmm, &local_bg, &mut gm, sq.n as i32, P7_LOCAL);
            let om = pressed_oprofiles[idx].clone();
            let mut threshold_pli = Pipeline::new();
            threshold_pli.new_model(&gm);
            configure_thresholds(&mut threshold_pli, &args);
            if let Some(cutoff) = bit_cutoff {
                threshold_pli.use_bit_cutoffs = cutoff;
                if threshold_pli.new_model_thresholds(&hmm.cutoff).is_err() {
                    return Ok(ScanModelResult {
                        hits: Vec::new(),
                        stats: PipelineStats::default(),
                    });
                }
            }
            let threshold_config =
                crate::subcmd::nhmmer::NhmmerThresholdConfig::from_pipeline(&threshold_pli);
            let bias_windows = crate::subcmd::nhmmer::NhmmerBiasWindowLengths {
                b1: args.b1,
                b2: args.b2,
                b3: args.b3,
            };

            let max_length = nhmmscan_max_length(hmm, args.w_length, args.w_beta);
            let effective_f1 = if args.max { 0.3 } else { args.f1 };
            let effective_f2 = if args.max { 1.0 } else { args.f2 };
            let effective_f3 = if args.max { 1.0 } else { args.f3 };
            let effective_nobias = args.max || args.nobias;
            let do_watson = !args.crick;
            let do_crick = !args.watson;
            let mut hits = Vec::new();
            let msv_counter = std::sync::atomic::AtomicU64::new(0);
            let bias_counter = std::sync::atomic::AtomicU64::new(0);
            let vit_counter = std::sync::atomic::AtomicU64::new(0);
            let fwd_counter = std::sync::atomic::AtomicU64::new(0);

            if do_crick && abc.complement.is_some() {
                let mut rc_dsq = sq.dsq.clone();
                abc.revcomp(&mut rc_dsq, sq.n);
                let rc_sq = Sequence {
                    name: sq.name.clone(),
                    acc: sq.acc.clone(),
                    desc: sq.desc.clone(),
                    dsq: rc_dsq,
                    n: sq.n,
                    l: sq.l,
                    taxid: -1,
                };
                let mut rc_hits = crate::subcmd::nhmmer::search_sequence(
                    &rc_sq,
                    hmm,
                    &gm,
                    &om,
                    &local_bg,
                    max_length,
                    effective_f1,
                    effective_f2,
                    effective_f3,
                    args.max,
                    effective_nobias,
                    args.nonull2,
                    args.seed,
                    bias_windows,
                    threshold_config,
                    true,
                    &msv_counter,
                    &bias_counter,
                    &vit_counter,
                    &fwd_counter,
                    None,
                )?;
                convert_crick_hits_to_forward(&mut rc_hits, sq.n);
                hits.extend(rc_hits);
            }

            if do_watson {
                hits.extend(crate::subcmd::nhmmer::search_sequence(
                    &sq,
                    hmm,
                    &gm,
                    &om,
                    &local_bg,
                    max_length,
                    effective_f1,
                    effective_f2,
                    effective_f3,
                    args.max,
                    effective_nobias,
                    args.nonull2,
                    args.seed,
                    bias_windows,
                    threshold_config,
                    false,
                    &msv_counter,
                    &bias_counter,
                    &vit_counter,
                    &fwd_counter,
                    None,
                )?);
            }

            let strands_searched = (do_watson as usize) + (do_crick as usize);
            let seq_len = sq.n * strands_searched.max(1);
            let nhmmscan_ln =
                nhmmscan_long_target_ln_adjustment(seq_len, max_length, search_space_ln);
            for hit in &mut hits {
                hit.lnp += nhmmscan_ln;
                hit.sortkey = hit.lnp;
                if let Some(dom0) = hit.dcl.first_mut() {
                    dom0.lnp = hit.lnp;
                }
                hit.name = hmm.name.clone();
                hit.acc = hmm.acc.clone().unwrap_or_default();
                hit.desc = hmm.desc.clone().unwrap_or_default();
                hit.n = hmm.m;
            }
            if let Some(cutoff) = bit_cutoff {
                let mut local_th = TopHits::new();
                local_th.hits = hits;
                let mut tmp_pli = Pipeline::new();
                tmp_pli.long_target = true;
                configure_thresholds(&mut tmp_pli, &args);
                tmp_pli.use_bit_cutoffs = cutoff;
                if tmp_pli.new_model_thresholds(&hmm.cutoff).is_ok() {
                    local_th.threshold(&tmp_pli, 1.0, 1.0);
                }
                hits = local_th.hits;
            }
            Ok(ScanModelResult {
                hits,
                stats: PipelineStats {
                    n_past_msv: msv_counter.load(std::sync::atomic::Ordering::Relaxed),
                    n_past_bias: bias_counter.load(std::sync::atomic::Ordering::Relaxed),
                    n_past_vit: vit_counter.load(std::sync::atomic::Ordering::Relaxed),
                    n_past_fwd: fwd_counter.load(std::sync::atomic::Ordering::Relaxed),
                },
            })
        };
        let scan_results: Vec<Result<ScanModelResult, String>> = if args.cpu == 0 {
            hmms.iter()
                .enumerate()
                .map(|(idx, hmm)| scan_model(idx, hmm))
                .collect()
        } else {
            use rayon::prelude::*;
            hmms.par_iter()
                .enumerate()
                .map(|(idx, hmm)| scan_model(idx, hmm))
                .collect()
        };
        let mut stats = PipelineStats::default();
        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = scan_results
            .into_iter()
            .flat_map(|result| {
                let result = result.unwrap_or_else(|err| {
                    eprintln!("Error: {err}");
                    std::process::exit(1);
                });
                stats.n_past_msv += result.stats.n_past_msv;
                stats.n_past_bias += result.stats.n_past_bias;
                stats.n_past_vit += result.stats.n_past_vit;
                stats.n_past_fwd += result.stats.n_past_fwd;
                result.hits
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        let evalue_scale = 1.0;
        th.sort_by_modelname_and_alipos();
        th.remove_duplicates();
        th.sort_by_sortkey();
        {
            let mut tmp_pli = Pipeline::new();
            tmp_pli.long_target = true;
            configure_thresholds(&mut tmp_pli, &args);
            if let Some(cutoff) = bit_cutoff {
                tmp_pli.use_bit_cutoffs = cutoff;
            }
            th.threshold(&tmp_pli, evalue_scale, evalue_scale);
        }

        if let Some(ref mut f) = tblout_file {
            crate::subcmd::nhmmer::write_nhmmer_tblout(
                f,
                "nhmmscan",
                "SCAN",
                &sq.name,
                Some(&sq.acc),
                &th,
                &args.seqfile,
                &args.hmmdb,
                &cmdline,
                query_idx == 1,
                false,
                evalue_scale,
                " modlen",
            );
        }
        if let Some(ref mut f) = dfamtblout_file {
            crate::subcmd::nhmmer::write_nhmmer_dfamtblout(
                f,
                &sq.name,
                None,
                "modlen",
                &th,
                evalue_scale,
            );
        }

        let textw = if args.notextw {
            0
        } else {
            args.textw.unwrap_or(120)
        };
        write_nhmmscan_stdout_hits(
            out,
            &th,
            evalue_scale,
            &sq.name,
            args.acc,
            args.noali,
            textw,
        );
        let strands_searched = ((!args.crick) as usize) + ((!args.watson) as usize);
        let pos_output = reported_aligned_residues(&th);
        write_nhmmscan_pipeline_stats(
            out,
            (sq.n * strands_searched.max(1)) as u64,
            hmms.len() as u64,
            hmms.iter().map(|hmm| hmm.m as u64).sum(),
            th.nreported as u64,
            pos_output,
            &stats,
            if args.max { 0.3 } else { args.f1 },
            if args.max { 1.0 } else { args.f2 },
            if args.max { 1.0 } else { args.f3 },
        );
        writeln!(out, "//").unwrap();

        sq.reuse();
    }

    if query_idx == 0 {
        eprintln!("Error: no sequences found in {}", args.seqfile.display());
        std::process::exit(1);
    }

    if let Some(ref mut f) = tblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "nhmmscan",
            "SCAN",
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

fn write_nhmmscan_option_header(out: &mut dyn Write, args: &Args, cmdline: &str) {
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
            fmt_g(score)
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
            fmt_g(score)
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
    if args.watson {
        writeln!(out, "# search only top strand:          on").unwrap();
    }
    if args.crick {
        writeln!(out, "# search only bottom strand:       on").unwrap();
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

fn nhmmscan_max_length(
    hmm: &hmmer_pure_rs::hmm::Hmm,
    w_length: Option<i32>,
    w_beta: Option<f64>,
) -> i32 {
    if let Some(w_length) = w_length {
        w_length
    } else if let Some(w_beta) = w_beta.filter(|beta| *beta > 0.0) {
        builder::max_length_from_beta(hmm, w_beta)
    } else if hmm.max_length > 0 {
        hmm.max_length
    } else if w_beta == Some(0.0) {
        (hmm.m * 4) as i32
    } else {
        builder::max_length_from_beta(hmm, DEFAULT_WINDOW_BETA)
    }
}

fn nhmmscan_long_target_ln_adjustment(
    seq_len: usize,
    max_length: i32,
    search_space_ln: f64,
) -> f64 {
    let nw_ratio = (seq_len as f32) / (max_length as f32);
    hmmer_pure_rs::util::cmath::c_log_f64(nw_ratio as f64) + search_space_ln
}

#[derive(Default)]
struct PipelineStats {
    n_past_msv: u64,
    n_past_bias: u64,
    n_past_vit: u64,
    n_past_fwd: u64,
}

struct ScanModelResult {
    hits: Vec<hmmer_pure_rs::tophits::Hit>,
    stats: PipelineStats,
}

fn write_nhmmscan_stdout_hits(
    out: &mut dyn Write,
    th: &TopHits,
    z: f64,
    qname: &str,
    prefer_acc: bool,
    noali: bool,
    textw: usize,
) {
    use hmmer_pure_rs::tophits::{P7_IS_INCLUDED, P7_IS_REPORTED};

    writeln!(out, "Scores for complete hit:").unwrap();
    writeln!(
        out,
        "    E-value  score  bias  Model     start    end  Description"
    )
    .unwrap();
    writeln!(
        out,
        "    ------- ------ -----  --------  -----  -----  -----------"
    )
    .unwrap();

    let mut have_printed_incthresh = false;
    let mut any_reported = false;
    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        any_reported = true;
        if hit.flags & P7_IS_INCLUDED == 0 && !have_printed_incthresh {
            writeln!(out, "  ------ inclusion threshold ------").unwrap();
            have_printed_incthresh = true;
        }
        let best_dom = hit
            .dcl
            .iter()
            .max_by(|a, b| a.bitscore.total_cmp(&b.bitscore));
        let (start, end) = best_dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
        writeln!(
            out,
            "  {:>9} {} {}  {:<8} {:>6} {:>6} {}",
            hmmer_pure_rs::output::fmt_evalue(z * c_exp_f64(hit.lnp)),
            fmt_score(hit.score),
            fmt_bias(hit.bias),
            display_name(hit, prefer_acc),
            start,
            end,
            if hit.desc.is_empty() { "" } else { &hit.desc },
        )
        .unwrap();
    }

    if !any_reported {
        writeln!(
            out,
            "   [No hits detected that satisfy reporting thresholds]"
        )
        .unwrap();
    }

    writeln!(out).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Annotation for each hit{}:",
        if noali { " " } else { "  (and alignments)" }
    )
    .unwrap();
    if !any_reported {
        writeln!(
            out,
            "\n   [No targets detected that satisfy reporting thresholds]"
        )
        .unwrap();
        return;
    }

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        writeln!(
            out,
            ">> {}  {}",
            display_name(hit, prefer_acc),
            if hit.desc.is_empty() { "" } else { &hit.desc }
        )
        .unwrap();
        if hit.nreported == 0 {
            writeln!(
                out,
                "   [No individual domains that satisfy reporting thresholds (although complete target did)]"
            )
            .unwrap();
            writeln!(out).unwrap();
            continue;
        }
        writeln!(
            out,
            "    score  bias    Evalue   hmmfrom    hmm to     alifrom    ali to      envfrom    env to      mod len      acc"
        )
        .unwrap();
        writeln!(
            out,
            "   ------ ----- ---------   -------   -------    --------- ---------    --------- ---------    ---------    ----"
        )
        .unwrap();
        for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
            let (hmmfrom, hmmto) = dom
                .ad
                .as_ref()
                .map(|ad| (ad.hmmfrom, ad.hmmto))
                .unwrap_or((0, 0));
            let indicator = if dom.is_included { "!" } else { "?" };
            let hmm_left = if hmmfrom == 1 { '[' } else { '.' };
            let hmm_right = if hmmto == hit.n { ']' } else { '.' };
            let ali_left = '.';
            let ali_right = '.';
            let env_left = '.';
            let env_right = '.';
            let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);
            writeln!(
                out,
                " {} {} {} {:>9}   {:>7}   {:>7} {}{} {:>9} {:>9} {}{} {:>9} {:>9} {}{} {:>9}    {}",
                indicator,
                hmmer_pure_rs::output::fmt_score(dom.bitscore),
                hmmer_pure_rs::output::fmt_bias(dom.dombias),
                hmmer_pure_rs::output::fmt_evalue(z * c_exp_f64(dom.lnp)),
                hmmfrom,
                hmmto,
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
                hit.n,
                hmmer_pure_rs::output::fmt_fixed2(acc as f64),
            )
            .unwrap();
        }
        writeln!(out).unwrap();

        if !noali {
            writeln!(out, "  Alignment:").unwrap();
            for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
                writeln!(
                    out,
                    "  score: {} bits",
                    hmmer_pure_rs::output::fmt_fixed1(dom.bitscore as f64)
                )
                .unwrap();
                if let Some(ref ad) = dom.ad {
                    hmmer_pure_rs::tophits::print_alidisplay_blocks(
                        out,
                        display_name(hit, prefer_acc),
                        qname,
                        ad,
                        None,
                        textw,
                    );
                }
                writeln!(out).unwrap();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_nhmmscan_pipeline_stats<W: Write + ?Sized>(
    out: &mut W,
    query_residues: u64,
    n_models: u64,
    total_nodes: u64,
    nreported: u64,
    pos_output: u64,
    stats: &PipelineStats,
    f1: f64,
    f2: f64,
    f3: f64,
) {
    let residues_scanned = query_residues.saturating_mul(n_models);
    let frac = |n: u64| {
        if residues_scanned > 0 {
            n as f64 / residues_scanned as f64
        } else {
            0.0
        }
    };
    writeln!(out).unwrap();
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
        n_models, total_nodes
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing SSV filter:     {:>11}  ({}); expected ({})",
        stats.n_past_msv,
        hmmer_pure_rs::output::fmt_g3(frac(stats.n_past_msv)),
        hmmer_pure_rs::output::fmt_g3(f1)
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing bias filter:    {:>11}  ({}); expected ({})",
        stats.n_past_bias,
        hmmer_pure_rs::output::fmt_g3(frac(stats.n_past_bias)),
        hmmer_pure_rs::output::fmt_g3(f1)
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing Vit filter:     {:>11}  ({}); expected ({})",
        stats.n_past_vit,
        hmmer_pure_rs::output::fmt_g3(frac(stats.n_past_vit)),
        hmmer_pure_rs::output::fmt_g3(f2)
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing Fwd filter:     {:>11}  ({}); expected ({})",
        stats.n_past_fwd,
        hmmer_pure_rs::output::fmt_g3(frac(stats.n_past_fwd)),
        hmmer_pure_rs::output::fmt_g3(f3)
    )
    .unwrap();
    writeln!(
        out,
        "Total number of hits:            {:>11}  ({})",
        nreported,
        hmmer_pure_rs::output::fmt_g3(frac(pos_output))
    )
    .unwrap();
    // Run-time footer (`# CPU time:` / `# Mc/sec:`) is intentionally
    // suppressed for determinism, matching the hmmsearch/phmmer/hmmscan/
    // jackhmmer Rust programs (which also omit these non-deterministic timing
    // lines). C HMMER prints them, but the Rust ports do not.
}

fn reported_aligned_residues(th: &TopHits) -> u64 {
    th.hits
        .iter()
        .filter(|hit| hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED != 0)
        .flat_map(|hit| hit.dcl.iter())
        .filter(|dom| dom.is_reported)
        .map(|dom| ((dom.jali - dom.iali).abs() + 1) as u64)
        .sum()
}

fn nucleotide_alphabet_from_hmms(hmms: &[hmmer_pure_rs::Hmm]) -> Result<Alphabet, String> {
    let hmm = hmms
        .first()
        .ok_or_else(|| "pressed HMM database contains no models".to_string())?;
    match hmm.abc_type {
        hmmer_pure_rs::alphabet::AlphabetType::Dna => Ok(Alphabet::dna()),
        hmmer_pure_rs::alphabet::AlphabetType::Rna => Ok(Alphabet::rna()),
        other => Err(format!(
            "nhmmscan requires a DNA or RNA HMM database; found {:?}",
            other
        )),
    }
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
    pli.inc_e = args.inc_e;
    if let Some(t) = args.score_threshold {
        pli.t = Some(t);
        pli.by_e = false;
    }
    if let Some(t) = args.inc_t {
        pli.inc_t = Some(t);
        pli.inc_by_e = false;
    }
}

fn convert_crick_hits_to_forward(hits: &mut [hmmer_pure_rs::tophits::Hit], n: usize) {
    for hit in hits {
        for dom in &mut hit.dcl {
            let ali_hi = n as i64 - dom.iali + 1;
            let ali_lo = n as i64 - dom.jali + 1;
            dom.iali = ali_hi;
            dom.jali = ali_lo;
            let env_hi = n as i64 - dom.ienv + 1;
            let env_lo = n as i64 - dom.jenv + 1;
            dom.ienv = env_hi;
            dom.jenv = env_lo;
            if let Some(ref mut ad) = dom.ad {
                let ad_hi = n.saturating_sub(ad.sqfrom) + 1;
                let ad_lo = n.saturating_sub(ad.sqto) + 1;
                ad.sqfrom = ad_hi;
                ad.sqto = ad_lo;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmmer_pure_rs::alphabet::AlphabetType;
    use hmmer_pure_rs::Hmm;

    #[test]
    fn nhmmscan_positionals_match_c_order() {
        let args = Args::try_parse_from(["nhmmscan", "models.hmm", "queries.fa"]).unwrap();

        assert_eq!(args.hmmdb, PathBuf::from("models.hmm"));
        assert_eq!(args.seqfile, PathBuf::from("queries.fa"));
    }

    #[test]
    fn nhmmscan_parses_c_output_options() {
        let args = Args::try_parse_from([
            "nhmmscan",
            "-o",
            "out.txt",
            "--tblout",
            "hits.tbl",
            "--dfamtblout",
            "dfam.tbl",
            "--acc",
            "--noali",
            "--notextw",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();

        assert_eq!(args.output, Some(PathBuf::from("out.txt")));
        assert_eq!(args.tblout, Some(PathBuf::from("hits.tbl")));
        assert_eq!(args.dfamtblout, Some(PathBuf::from("dfam.tbl")));
        assert!(args.acc);
        assert!(args.noali);
        assert!(args.notextw);
    }

    #[test]
    fn nhmmscan_parses_c_threshold_options() {
        let args = Args::try_parse_from([
            "nhmmscan",
            "-T",
            "42",
            "--incT",
            "40",
            "-Z",
            "45000000",
            "--seed",
            "7",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();
        assert_eq!(args.score_threshold, Some(42.0));
        assert_eq!(args.inc_t, Some(40.0));
        assert_eq!(args.z_value, Some(45_000_000.0));
        assert_eq!(args.seed, 7);

        let args =
            Args::try_parse_from(["nhmmscan", "--cut_ga", "models.hmm", "queries.fa"]).unwrap();
        assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::GA));

        assert!(
            Args::try_parse_from(["nhmmscan", "--domZ", "1", "models.hmm", "queries.fa"]).is_err()
        );
        assert!(
            Args::try_parse_from(["nhmmscan", "--domE", "1", "models.hmm", "queries.fa"]).is_err()
        );
    }

    #[test]
    fn nhmmscan_accepts_negative_space_separated_f_values() {
        // C --F1/--F2/--F3 are eslARG_REAL with no range; the space-separated
        // negative form is accepted. allow_hyphen_values matches C instead of
        // treating "-0.5" as an unknown flag.
        let args = Args::try_parse_from([
            "nhmmscan",
            "--F1",
            "-0.5",
            "--F2",
            "-1e-3",
            "--F3",
            "-2",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();
        assert_eq!(args.f1, -0.5);
        assert_eq!(args.f2, -1e-3);
        assert_eq!(args.f3, -2.0);
    }

    #[test]
    fn nhmmscan_long_target_adjustment_folds_model_search_space_into_lnp() {
        let search_space_ln = hmmer_pure_rs::util::cmath::c_log_f64(12.0);
        let got = nhmmscan_long_target_ln_adjustment(800, 200, search_space_ln);
        let expected = hmmer_pure_rs::util::cmath::c_log_f64(48.0);

        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn nhmmscan_uses_pressed_database_rna_alphabet() {
        let hmms = vec![Hmm::new(4, AlphabetType::Rna, 4)];
        let abc = nucleotide_alphabet_from_hmms(&hmms).unwrap();
        assert_eq!(abc.abc_type, AlphabetType::Rna);
    }

    #[test]
    fn nhmmscan_rejects_protein_pressed_database_alphabet() {
        let hmms = vec![Hmm::new(4, AlphabetType::Amino, 20)];
        let err = nucleotide_alphabet_from_hmms(&hmms).unwrap_err();
        assert!(err.contains("requires a DNA or RNA HMM database"));
    }
}
