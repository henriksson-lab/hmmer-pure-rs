//! Pure Rust hmmsearch — uses generic DP algorithms.
//! Progressively replacing C hmmsearch functionality.

use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::sequence::{self, Sequence};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::{Domain, Hit, TopHits};
use hmmer_pure_rs::{hmmfile, hmmfile_binary};

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
    #[arg(long = "F1", default_value = "0.02")]
    f1: f64,

    /// Viterbi threshold
    #[arg(long = "F2", default_value = "0.001")]
    f2: f64,

    /// Forward threshold
    #[arg(long = "F3", default_value = "1e-5")]
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

    /// Assert target sequence file format (not implemented; auto-detection is used)
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

/// Entry point for `hmmsearch`: search profile(s) against a sequence database.
///
/// Equivalent to the C `main` + `serial_master` + `output_header` in
/// `hmmer/src/hmmsearch.c`. Parses CLI args, reads HMM(s), initializes the
/// pipeline (filter thresholds, reporting/inclusion thresholds, model-specific
/// cutoffs, Z/domZ overrides), iterates target sequences (serial when
/// `--cpu 1`, otherwise rayon-parallel over a target batch), thresholds and
/// emits the standard hmmsearch text report plus optional --tblout,
/// --domtblout, --pfamtblout, and -A (Stockholm) outputs.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let args = Args::parse_from(&args);

    if let Some(ref format) = args.tformat {
        if format.eq_ignore_ascii_case("fasta") {
            // FASTA is already accepted by the sequence reader's auto-detection.
        } else {
            eprintln!(
                "hmmsearch --tformat={} is not implemented: only FASTA target format assertions are supported",
            format
        );
            std::process::exit(1);
        }
    }

    // Configure thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.cpu)
        .start_handler(|_| hmmer_pure_rs::util::simd_env::init())
        .build_global()
        .ok();

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
        outfile_handle = std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Error creating output file: {}", e);
            std::process::exit(1);
        });
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
    if args
        .tformat
        .as_deref()
        .is_some_and(|format| format.eq_ignore_ascii_case("fasta"))
    {
        writeln!(out, "# targ <seqfile> format asserted:  fasta").unwrap();
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
        writeln!(out, "# pfam-style tabular output:       {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.ali_outfile {
        writeln!(out, "# MSA of hits saved to file:       {}", path.display()).unwrap();
    }
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    if args.show_acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
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
        let file = std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating tblout file: {}", e);
            std::process::exit(1);
        });
        BufWriter::new(file)
    });
    let mut domtblout_file = args.domtblout.as_ref().map(|p| {
        let file = std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating domtblout file: {}", e);
            std::process::exit(1);
        });
        BufWriter::new(file)
    });
    let mut pfamtblout_file = args.pfamtblout.as_ref().map(|p| {
        let file = std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating pfamtblout file: {}", e);
            std::process::exit(1);
        });
        BufWriter::new(file)
    });
    let mut ali_outfile = args.ali_outfile.as_ref().map(|p| {
        let file = std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating alignment output file: {}", e);
            std::process::exit(1);
        });
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

        // Filter thresholds
        pli.f1 = args.f1;
        pli.f2 = args.f2;
        pli.f3 = args.f3;
        pli.do_max = args.max;
        if args.nobias {
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

        // Score sequences. For --cpu 1, stream the target database like C HMMER
        // instead of cloning the entire sequence set before scoring.
        let f1 = args.f1;
        let f2 = args.f2;
        let f3 = args.f3;
        let do_max = args.max;
        let nobias = args.nobias;
        let nonull2 = args.nonull2;
        let seed = args.seed;
        let do_alignment = !args.noali || args.domtblout.is_some() || args.pfamtblout.is_some();
        let do_alignment_display = !args.noali;
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
        if args.cpu == 1 {
            let mut local_bg = bg.clone();
            let mut local_gm = gm.clone();
            let mut local_om = om.clone();
            let mut local_pli = Pipeline::new();
            local_pli.new_model(&local_gm);
            local_pli.use_bit_cutoffs = pli.use_bit_cutoffs;
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

            let mut sqf = open_target_seq_file(&args.seqdb, &abc, args.tformat.as_deref())
                .unwrap_or_else(|e| {
                    eprintln!("Error opening sequence file: {}", e);
                    std::process::exit(1);
                });
            let mut sq = Sequence::new();
            while sqf.read(&mut sq).unwrap_or_else(|e| {
                eprintln!("Error reading sequence file: {}", e);
                std::process::exit(1);
            }) {
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
            let mut sq = Sequence::new();
            let mut batch = Vec::with_capacity(TARGET_BATCH_SIZE);

            loop {
                while batch.len() < TARGET_BATCH_SIZE {
                    match sqf.read(&mut sq) {
                        Ok(true) => {
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
                            local_pli.use_bit_cutoffs = pli.use_bit_cutoffs;
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
            hmmer_pure_rs::pipeline::ZSetBy::Ntargets => th.nreported.max(1) as f64,
        };
        // Re-threshold with correct domz for domain-level E-values
        if domz != z {
            th.threshold(&pli, z, domz);
        }

        let show_acc = args.show_acc;

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
            "    E-value  score  bias    E-value  score  bias    exp  N  Sequence Description"
        )
        .unwrap();
        writeln!(
            out,
            "    ------- ------ -----    ------- ------ -----   ---- --  -------- -----------"
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
            let evalue = z * hit.lnp.exp();
            let best_dom = best_domain(hit);
            let dom_evalue = best_dom.map(|d| z * d.lnp.exp()).unwrap_or(evalue);
            let dom_score = best_dom.map(|d| d.bitscore).unwrap_or(hit.score);
            let dom_bias = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);

            writeln!(
                out,
                "  {} {:6.1} {:5.1}  {} {:6.1} {:5.1}  {:4.1} {:2}  {} {}",
                hmmer_pure_rs::output::fmt_evalue(evalue),
                hit.score,
                hit.bias,
                hmmer_pure_rs::output::fmt_evalue(dom_evalue),
                dom_score,
                dom_bias,
                hit.nexpected,
                hit.nreported,
                if show_acc && !hit.acc.is_empty() {
                    &hit.acc
                } else {
                    &hit.name
                },
                if hit.desc.is_empty() { "" } else { &hit.desc },
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

        // Domain annotation for each sequence
        if !args.noali {
            writeln!(out, "Domain annotation for each sequence (and alignments):").unwrap();

            for hit in &th.hits {
                if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
                    continue;
                }

                writeln!(
                    out,
                    ">> {}  {}",
                    if show_acc && !hit.acc.is_empty() {
                        &hit.acc
                    } else {
                        &hit.name
                    },
                    hit.desc
                )
                .unwrap();
                writeln!(out, "   #    score  bias  c-Evalue  i-Evalue hmmfrom  hmm to    alifrom  ali to    envfrom  env to     acc").unwrap();
                writeln!(out, " ---   ------ ----- --------- --------- ------- -------    ------- -------    ------- -------    ----").unwrap();

                let mut reported_idx = 0usize;
                for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
                    reported_idx += 1;
                    let c_evalue = domz * dom.lnp.exp();
                    let i_evalue = z * dom.lnp.exp();
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
                    let seq_left = if dom.iali == dom.ienv { hmm_left } else { '.' };
                    let seq_right = if dom.jali == dom.jenv { hmm_right } else { '.' };
                    let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);

                    writeln!(
                        out,
                        " {:3} {} {:6.1} {:5.1} {} {} {:7} {:7} {}{} {:7} {:7} {}{} {:7} {:7} {}{} {:.2}",
                        reported_idx,
                        indicator,
                        dom.bitscore,
                        dom.dombias,
                        hmmer_pure_rs::output::fmt_evalue(c_evalue),
                        hmmer_pure_rs::output::fmt_evalue(i_evalue),
                        hf, ht,
                        hmm_left, hmm_right,
                        dom.iali, dom.jali,
                        seq_left, seq_right,
                        dom.ienv, dom.jenv,
                        seq_left, seq_right,
                        acc,
                    ).unwrap();
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
                            "  == domain {}  score: {:.1} bits;  conditional E-value: {}",
                            reported_idx,
                            dom.bitscore,
                            hmmer_pure_rs::output::fmt_evalue(domz * dom.lnp.exp()).trim()
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
                            hmmer_pure_rs::tophits::print_alidisplay_blocks(
                                &mut out,
                                &hmm.name,
                                &hit.name,
                                ad,
                                cs_line.as_deref(),
                                textw,
                            );
                        }
                        writeln!(out).unwrap();
                    }
                }
            }
        }

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

        writeln!(out, "Internal pipeline statistics summary:").unwrap();
        writeln!(out, "-------------------------------------").unwrap();
        writeln!(
            out,
            "Query model(s):                  {:>10}  ({} nodes)",
            1, hmm.m
        )
        .unwrap();
        writeln!(
            out,
            "Target sequences:                {:>10}  ({} residues searched)",
            pli.n_targets, total_residues
        )
        .unwrap();
        writeln!(
            out,
            "Passed MSV filter:               {:>10}  ({:.4}); expected {:.1} ({:.2})",
            pli.n_past_msv, frac_msv, expected_msv, pli.f1
        )
        .unwrap();
        writeln!(
            out,
            "Passed bias filter:              {:>10}  ({:.4}); expected {:.1} ({:.2})",
            pli.n_past_bias, frac_msv, expected_msv, pli.f1
        )
        .unwrap();
        writeln!(
            out,
            "Passed Vit filter:               {:>10}  ({:.4}); expected {:.1} ({:.4})",
            pli.n_past_vit, frac_vit, expected_vit, pli.f2
        )
        .unwrap();
        writeln!(
            out,
            "Passed Fwd filter:               {:>10}  ({:.4}); expected {:.1} ({:.0e})",
            pli.n_past_fwd, frac_fwd, expected_fwd, pli.f3
        )
        .unwrap();
        writeln!(
            out,
            "Initial search space (Z):        {:>10}  [actual number of targets]",
            pli.n_targets
        )
        .unwrap();
        writeln!(
            out,
            "Domain search space  (domZ):     {:>10}  [number of targets reported over threshold]",
            th.nreported
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
            write_pfamtblout(f, &hmm.name, hmm.acc.as_deref(), hmm.m, &th, z, domz);
        }
        if let Some(ref mut f) = ali_outfile {
            write_ali_output(f, hmm, &th, domz, textw);
        }
        writeln!(out, "//").unwrap();
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

/// Read one or more profile HMMs from `path`, auto-dispatching by extension.
///
/// `.h3m` files use the binary HMMER3 reader; everything else uses the ASCII
/// reader. Corresponds to C `p7_hmmfile_Read` (binary or ASCII selected
/// internally by `p7_hmmfile_Open`).
fn read_hmms(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<hmmer_pure_rs::Hmm>> {
    if path == std::path::Path::new("-") {
        hmmfile::read_hmms(BufReader::new(std::io::stdin().lock()))
    } else if path.extension().is_some_and(|ext| ext == "h3m") {
        hmmfile_binary::read_binary_hmm_file(path)
    } else {
        hmmfile::read_hmm_file(path)
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
        let evalue = z * hit.lnp.exp();
        let best_dom = best_domain(hit);
        let dom_evalue = best_dom.map(|d| z * d.lnp.exp()).unwrap_or(evalue);
        let dom_score = best_dom.map(|d| d.bitscore).unwrap_or(hit.score);
        let dom_bias = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);

        writeln!(
            f,
            "{:<tnamew$} {:<taccw$} {:<qnamew$} {:<qaccw$} {} {:>6.1} {:>5.1} {} {:>6.1} {:>5.1} {:>5.1} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {}",
            hit.name,
            if hit.acc.is_empty() { "-" } else { &hit.acc },
            qname,
            qacc_s,
            hmmer_pure_rs::output::fmt_evalue(evalue),
            hit.score,
            hit.bias,
            hmmer_pure_rs::output::fmt_evalue(dom_evalue),
            dom_score,
            dom_bias,
            hit.nexpected,
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
    hit.dcl
        .iter()
        .max_by(|a, b| a.bitscore.total_cmp(&b.bitscore))
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
        let evalue = z * hit.lnp.exp();

        let mut reported_idx = 0usize;
        for dom in &hit.dcl {
            if !dom.is_reported {
                continue;
            }
            reported_idx += 1;
            let c_evalue = domz * dom.lnp.exp();
            let i_evalue = z * dom.lnp.exp();
            let (hmmfrom, hmmto, acc) = if let Some(ref ad) = dom.ad {
                let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);
                (ad.hmmfrom, ad.hmmto, acc)
            } else {
                (1, qlen, 0.0)
            };

            writeln!(
                f,
                "{:<tnamew$} {:<taccw$} {:>5} {:<qnamew$} {:<qaccw$} {:>5} {} {:>6.1} {:>5.1} {:>3} {:>3} {} {} {:>6.1} {:>5.1} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>4.2} {}",
                hit.name,
                if hit.acc.is_empty() { "-" } else { &hit.acc },
                hit.n,
                qname,
                qacc_s,
                qlen,
                hmmer_pure_rs::output::fmt_evalue(evalue),
                hit.score,
                hit.bias,
                reported_idx,
                hit.nreported,
                hmmer_pure_rs::output::fmt_evalue(c_evalue),
                hmmer_pure_rs::output::fmt_evalue(i_evalue),
                dom.bitscore,
                dom.dombias,
                hmmfrom,
                hmmto,
                dom.iali,
                dom.jali,
                dom.ienv,
                dom.jenv,
                acc,
                if hit.desc.is_empty() { "-" } else { &hit.desc },
            ).unwrap();
        }
    }
}

/// Write Pfam-format tabular output (`--pfamtblout`).
///
/// Two sections: sequence scores then domain scores, both restricted to
/// reported hits. Domain rows are re-sorted by bitscore (descending), with
/// hit/domain indices as tie-breakers. Matches the Pfam-style block emitted
/// by C `p7_tophits_TabularXfam` in `hmmer/src/p7_tophits.c`.
pub fn write_pfamtblout<W: Write>(
    f: &mut W,
    _qname: &str,
    _qacc: Option<&str>,
    qlen: usize,
    th: &TopHits,
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
        let evalue = z * hit.lnp.exp();
        writeln!(
            f,
            "{:<tnamew$}  {:>6.1} {:>9} {:>3} {:>5.1} {:>5.1}    {}",
            hit.name,
            hit.score,
            hmmer_pure_rs::output::fmt_evalue(evalue),
            hit.ndom,
            hit.nexpected,
            hit.bias,
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

    let mut reported_domains = Vec::new();
    for (hit_idx, hit) in th.hits.iter().enumerate() {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        for (dom_idx, dom) in hit.dcl.iter().enumerate() {
            if dom.is_reported {
                reported_domains.push((hit_idx, dom_idx, hit, dom));
            }
        }
    }
    reported_domains.sort_by(|a, b| {
        b.3.bitscore
            .total_cmp(&a.3.bitscore)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });

    for (_hit_idx, dom_idx, hit, dom) in reported_domains {
        let i_evalue = z * dom.lnp.exp();
        let (hmmfrom, hmmto) = if let Some(ref ad) = dom.ad {
            (ad.hmmfrom, ad.hmmto)
        } else {
            (1, qlen)
        };
        writeln!(
            f,
            "{:<tnamew$}  {:>6.1} {:>9} {:>5} {:>5.1} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}     {}",
            hit.name,
            dom.bitscore,
            hmmer_pure_rs::output::fmt_evalue(i_evalue),
            dom_idx + 1,
            dom.dombias,
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

/// Write a minimal Stockholm MSA of all included domain hits (`-A`).
///
/// Emits `#=GF ID / AC / DE` annotation lines from the query HMM followed by
/// each included domain's aligned sequence (`AliDisplay::aseq`). The C
/// equivalent is `p7_tophits_Alignment` → `esl_msafile_Write` in
/// `hmmer/src/p7_tophits.c`; this is a lightweight stand-in.
fn write_ali_output<W: Write>(
    f: &mut W,
    hmm: &hmmer_pure_rs::hmm::Hmm,
    th: &TopHits,
    _domz: f64,
    _textw: usize,
) {
    // Write a minimal Stockholm-format MSA of included hits
    writeln!(f, "# STOCKHOLM 1.0").unwrap();
    writeln!(f, "#=GF ID   {}", hmm.name).unwrap();
    if let Some(ref acc) = hmm.acc {
        writeln!(f, "#=GF AC   {}", acc).unwrap();
    }
    if let Some(ref desc) = hmm.desc {
        writeln!(f, "#=GF DE   {}", desc).unwrap();
    }

    // Collect included domains and output their aligned sequences
    for hit in &th.hits {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED == 0 {
            continue;
        }
        for dom in &hit.dcl {
            if !dom.is_included {
                continue;
            }
            if let Some(ref ad) = dom.ad {
                writeln!(f, "{:<30} {}", hit.name, ad.aseq).unwrap();
            }
        }
    }
    writeln!(f, "//").unwrap();
}

fn open_target_seq_file(
    path: &Path,
    abc: &Alphabet,
    tformat: Option<&str>,
) -> hmmer_pure_rs::errors::HmmerResult<sequence::SeqFile<Box<dyn std::io::Read>>> {
    if tformat.is_some_and(|format| format.eq_ignore_ascii_case("fasta")) {
        sequence::open_fasta_seq_file(path, abc)
    } else {
        sequence::open_seq_file(path, abc)
    }
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
        write_pfamtblout(&mut out, "query", None, 42, &th, 1.0, 1.0);
        let out = String::from_utf8(out).unwrap();

        assert!(out.contains("long-reported-target-name    10.0"));
        assert!(out.contains("long-reported-target-name    20.0"));
        assert!(out.lines().any(|line| {
            line.starts_with("long-reported-target-name")
                && line.split_whitespace().nth(3) == Some("2")
        }));
    }
}
