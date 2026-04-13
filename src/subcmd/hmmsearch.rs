//! Pure Rust hmmsearch — uses generic DP algorithms.
//! Progressively replacing C hmmsearch functionality.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::sequence::{self, Sequence};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::TopHits;

#[derive(Parser)]
#[command(name = "hmmsearch", about = "Search profile(s) against a sequence database")]
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
    #[arg(short = 'E', default_value = "10.0")]
    e_value: f64,

    /// Report sequences >= this score threshold
    #[arg(short = 'T')]
    score_threshold: Option<f32>,

    /// Report domains <= this E-value threshold
    #[arg(long = "domE", default_value = "10.0")]
    dom_e: f64,

    /// Report domains >= this score threshold
    #[arg(long = "domT")]
    dom_t: Option<f32>,

    // --- Inclusion thresholds ---

    /// Include sequences <= this E-value threshold
    #[arg(long = "incE", default_value = "0.01")]
    inc_e: f64,

    /// Include sequences >= this score threshold
    #[arg(long = "incT")]
    inc_t: Option<f32>,

    /// Include domains <= this E-value threshold
    #[arg(long = "incdomE", default_value = "0.01")]
    inc_dome: f64,

    /// Include domains >= this score threshold
    #[arg(long = "incdomT")]
    inc_dom_t: Option<f32>,

    // --- Model-specific cutoffs ---

    /// Use model's GA gathering cutoffs to set all thresholding
    #[arg(long = "cut_ga")]
    cut_ga: bool,

    /// Use model's NC noise cutoffs to set all thresholding
    #[arg(long = "cut_nc")]
    cut_nc: bool,

    /// Use model's TC trusted cutoffs to set all thresholding
    #[arg(long = "cut_tc")]
    cut_tc: bool,

    // --- Acceleration heuristics ---

    /// Skip all filters (run everything through Forward)
    #[arg(long = "max")]
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
    #[arg(short = 'Z')]
    z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ")]
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
    #[arg(long = "notextw")]
    notextw: bool,

    /// Set max width of ASCII text output lines
    #[arg(long = "textw", default_value = "120")]
    textw: usize,

    /// Save table of hits in Pfam format
    #[arg(long = "pfamtblout")]
    pfamtblout: Option<PathBuf>,

    /// Save multiple alignment of all hits to file
    #[arg(short = 'A')]
    ali_outfile: Option<PathBuf>,

    /// Assert target sequence file format (e.g. fasta, embl, genbank, uniprot)
    #[arg(long = "tformat")]
    tformat: Option<String>,
}

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    // Configure thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.cpu)
        .build_global()
        .ok();

    logsum::p7_flogsuminit();

    // Read HMM(s)
    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    // Output destination: -o file or stdout
    let outfile_handle;
    let stdout;
    let mut out: Box<dyn std::io::Write> = if let Some(ref path) = args.outfile {
        outfile_handle = std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Error creating output file: {}", e);
            std::process::exit(1);
        });
        Box::new(std::io::BufWriter::new(outfile_handle))
    } else {
        stdout = std::io::stdout();
        Box::new(stdout.lock())
    };

    // Print header
    writeln!(out, "# hmmsearch :: search profile(s) against a sequence database").unwrap();
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(out, "# Copyright (C) 2023 Howard Hughes Medical Institute.").unwrap();
    writeln!(out, "# Freely distributed under the BSD open source license.").unwrap();
    writeln!(out, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -").unwrap();
    writeln!(out, "# query HMM file:                  {}", args.hmmfile.display()).unwrap();
    writeln!(out, "# target sequence database:        {}", args.seqdb.display()).unwrap();
    writeln!(out, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -").unwrap();
    writeln!(out).unwrap();

    // Open tblout/domtblout files if requested
    let mut tblout_file = args.tblout.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating tblout file: {}", e);
            std::process::exit(1);
        })
    });
    let mut domtblout_file = args.domtblout.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating domtblout file: {}", e);
            std::process::exit(1);
        })
    });
    let mut pfamtblout_file = args.pfamtblout.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating pfamtblout file: {}", e);
            std::process::exit(1);
        })
    });
    let mut ali_outfile = args.ali_outfile.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating alignment output file: {}", e);
            std::process::exit(1);
        })
    });

    let textw = if args.notextw { 0 } else { args.textw };

    for hmm in &hmms {
        let abc = Alphabet::new(hmm.abc_type);
        let mut bg = Bg::new(&abc);
        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
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
        if args.nobias { pli.do_biasfilter = false; }
        if args.nonull2 { pli.do_null2 = false; }
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
            continue;
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

        // Read all sequences first
        let mut sequences = Vec::new();
        let mut total_residues: u64 = 0;
        {
            let mut sqf = sequence::open_seq_file(&args.seqdb, &abc).unwrap_or_else(|e| {
                eprintln!("Error opening sequence file: {}", e);
                std::process::exit(1);
            });
            let mut sq = Sequence::new();
            while sqf.read(&mut sq).unwrap() {
                total_residues += sq.n as u64;
                sequences.push(sq.clone());
                sq.reuse();
            }
        }

        // Score sequences in parallel using rayon
        // Share base profile/oprofile via Arc to avoid expensive clones
        use rayon::prelude::*;
        use std::sync::Arc;
        let shared_gm = Arc::new(gm.clone());
        let shared_om = Arc::new(om.clone());
        let f1 = args.f1;
        let f2 = args.f2;
        let f3 = args.f3;
        let do_max = args.max;

        let nobias = args.nobias;
        let nonull2 = args.nonull2;
        let seed = args.seed;

        let results: Vec<(Option<hmmer_pure_rs::tophits::Hit>, u64, u64, u64, u64)> = sequences
            .par_iter()
            .map(|sq| {
                let mut local_bg = bg.clone();
                let mut local_gm = (*shared_gm).clone();
                let mut local_om = (*shared_om).clone();
                let mut local_pli = Pipeline::new();
                local_pli.new_model(&local_gm);
                local_pli.f1 = f1;
                local_pli.f2 = f2;
                local_pli.f3 = f3;
                local_pli.do_max = do_max;
                if nobias { local_pli.do_biasfilter = false; }
                if nonull2 { local_pli.do_null2 = false; }
                local_pli.seed = seed;

                local_bg.set_length(sq.n);
                profile::reconfig_length(&mut local_gm, sq.n as i32);
                local_om.reconfig_length(sq.n as i32);

                let mut local_th = TopHits::new();
                let hit = if local_pli.run(&local_gm, &local_om, &local_bg, hmm, sq, &mut local_th) {
                    local_th.hits.into_iter().next()
                } else {
                    None
                };
                (hit, local_pli.n_past_msv, local_pli.n_past_bias, local_pli.n_past_vit, local_pli.n_past_fwd)
            })
            .collect();

        let mut th = TopHits::new();
        pli.n_targets = sequences.len() as u64;
        pli.n_past_msv = 0;
        pli.n_past_bias = 0;
        pli.n_past_vit = 0;
        pli.n_past_fwd = 0;
        for (hit, msv, bias, vit, fwd) in results {
            pli.n_past_msv += msv;
            pli.n_past_bias += bias;
            pli.n_past_vit += vit;
            pli.n_past_fwd += fwd;
            if let Some(h) = hit {
                th.hits.push(h);
            }
        }

        // Set Z (database size)
        let z = match pli.z_setby {
            hmmer_pure_rs::pipeline::ZSetBy::Option => pli.z,
            hmmer_pure_rs::pipeline::ZSetBy::Ntargets => pli.n_targets as f64,
        };

        // Sort and threshold (first pass with domz = z for sequence-level reporting)
        th.sort_by_sortkey();
        th.threshold(
            &pli,
            z,
            z, // temporary domz = z for first pass
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

        // Output query header
        writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.m).unwrap();

        let show_acc = args.show_acc;

        // Per-sequence hit table
        writeln!(out, "Scores for complete sequences (score includes all domains):").unwrap();
        writeln!(out, "   --- full sequence ---   --- best 1 domain ---    -#dom-").unwrap();
        writeln!(out, "    E-value  score  bias    E-value  score  bias    exp  N  Sequence Description").unwrap();
        writeln!(out, "    ------- ------ -----    ------- ------ -----   ---- --  -------- -----------").unwrap();

        let mut any_reported = false;
        for hit in &th.hits {
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
                continue;
            }
            any_reported = true;
            let evalue = z * hit.lnp.exp();
            let dom_evalue = if !hit.dcl.is_empty() {
                domz * hit.dcl[0].lnp.exp()
            } else {
                evalue
            };
            let dom_score = if !hit.dcl.is_empty() {
                hit.dcl[0].bitscore
            } else {
                hit.score
            };

            writeln!(
                out,
                "  {} {:6.1} {:5.1}  {} {:6.1} {:5.1}  {:4.1} {:2}  {} {}",
                hmmer_pure_rs::output::fmt_evalue(evalue),
                hit.score,
                hit.bias,
                hmmer_pure_rs::output::fmt_evalue(dom_evalue),
                dom_score,
                hit.bias,
                hit.nexpected,
                hit.ndom,
                if show_acc && !hit.acc.is_empty() { &hit.acc } else { &hit.name },
                if hit.desc.is_empty() { "" } else { &hit.desc },
            ).unwrap();
        }

        if !any_reported {
            writeln!(out, "\n   [No hits detected that satisfy reporting thresholds]").unwrap();
        }

        writeln!(out).unwrap();

        // Domain annotation for each sequence
        if !args.noali {
            writeln!(out, "Domain annotation for each sequence (and alignments):").unwrap();

            for hit in &th.hits {
                if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
                    continue;
                }

                writeln!(out, ">> {}  {}", if show_acc && !hit.acc.is_empty() { &hit.acc } else { &hit.name }, hit.desc).unwrap();
                writeln!(out, "   #    score  bias  c-Evalue  i-Evalue hmmfrom  hmm to    alifrom  ali to    envfrom  env to     acc").unwrap();
                writeln!(out, " ---   ------ ----- --------- --------- ------- -------    ------- -------    ------- -------    ----").unwrap();

                for (di, dom) in hit.dcl.iter().enumerate() {
                    let dom_evalue = domz * dom.lnp.exp();
                    let c_evalue = dom.lnp.exp(); // conditional E-value (per-domain Z=1 for display)
                    let indicator = if dom.is_included {
                        '!'
                    } else if dom.is_reported {
                        '?'
                    } else {
                        '!'  // default to included
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

                    writeln!(
                        out,
                        " {:3} {} {:6.1} {:5.1} {} {} {:7} {:7} {}{} {:7} {:7} {}{} {:7} {:7} {}{} {:.2}",
                        di + 1,
                        indicator,
                        dom.bitscore,
                        dom.dombias,
                        hmmer_pure_rs::output::fmt_evalue(c_evalue),
                        hmmer_pure_rs::output::fmt_evalue(dom_evalue),
                        hf, ht,
                        hmm_left, hmm_right,
                        dom.iali, dom.jali,
                        seq_left, seq_right,
                        dom.ienv, dom.jenv,
                        seq_left, seq_right,
                        0.95_f32,
                    ).unwrap();
                }

                writeln!(out).unwrap();

                // Text alignments for each domain
                if !args.noali {
                    writeln!(out, "  Alignments for each domain:").unwrap();
                    for (di, dom) in hit.dcl.iter().enumerate() {
                        writeln!(out, "  == domain {}  score: {:.1} bits;  conditional E-value: {}",
                            di + 1, dom.bitscore, hmmer_pure_rs::output::fmt_evalue(dom.lnp.exp()).trim()).unwrap();

                        if let Some(ref ad) = dom.ad {
                            let name_width = hmm.name.len().max(hit.name.len()).max(5);
                            // Model line
                            writeln!(out, "  {:>width$} {:>3} {} {:>3}",
                                hmm.name, ad.hmmfrom, ad.model, ad.hmmto, width = name_width).unwrap();
                            // Match line
                            writeln!(out, "  {:>width$}     {}",
                                "", ad.mline, width = name_width).unwrap();
                            // Target sequence line
                            writeln!(out, "  {:>width$} {:>3} {} {:>3}",
                                hit.name, ad.sqfrom, ad.aseq, ad.sqto, width = name_width).unwrap();
                            // PP annotation line
                            writeln!(out, "  {:>width$}     {} PP",
                                "", ad.ppline, width = name_width).unwrap();
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
        let frac_msv = if pli.n_targets > 0 { pli.n_past_msv as f64 / pli.n_targets as f64 } else { 0.0 };
        let frac_vit = if pli.n_targets > 0 { pli.n_past_vit as f64 / pli.n_targets as f64 } else { 0.0 };
        let frac_fwd = if pli.n_targets > 0 { pli.n_past_fwd as f64 / pli.n_targets as f64 } else { 0.0 };

        writeln!(out, "Internal pipeline statistics summary:").unwrap();
        writeln!(out, "-------------------------------------").unwrap();
        writeln!(out, "Query model(s):                  {:>10}  ({} nodes)", 1, hmm.m).unwrap();
        writeln!(out, "Target sequences:                {:>10}  ({} residues searched)", pli.n_targets, total_residues).unwrap();
        writeln!(out, "Passed MSV filter:               {:>10}  ({:.4}); expected {:.1} ({:.2})", pli.n_past_msv, frac_msv, expected_msv, pli.f1).unwrap();
        writeln!(out, "Passed bias filter:              {:>10}  ({:.4}); expected {:.1} ({:.2})", pli.n_past_bias, frac_msv, expected_msv, pli.f1).unwrap();
        writeln!(out, "Passed Vit filter:               {:>10}  ({:.4}); expected {:.1} ({:.4})", pli.n_past_vit, frac_vit, expected_vit, pli.f2).unwrap();
        writeln!(out, "Passed Fwd filter:               {:>10}  ({:.4}); expected {:.1} ({:.0e})", pli.n_past_fwd, frac_fwd, expected_fwd, pli.f3).unwrap();
        writeln!(out, "Initial search space (Z):        {:>10}  [actual number of targets]", pli.n_targets).unwrap();
        writeln!(out, "Domain search space  (domZ):     {:>10}  [number of targets reported over threshold]", th.nreported).unwrap();

        // Write tabular output
        if let Some(ref mut f) = tblout_file {
            write_tblout(f, &hmm.name, hmm.acc.as_deref(), &th, z);
        }
        if let Some(ref mut f) = domtblout_file {
            write_domtblout(f, &hmm.name, hmm.acc.as_deref(), &th, z, domz);
        }
        if let Some(ref mut f) = pfamtblout_file {
            write_pfamtblout(f, &hmm.name, hmm.acc.as_deref(), &th, z, domz);
        }
        if let Some(ref mut f) = ali_outfile {
            write_ali_output(f, hmm, &th, domz, textw);
        }
    }

    writeln!(out, "//").unwrap();
    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

fn write_tblout(f: &mut std::fs::File, qname: &str, qacc: Option<&str>, th: &TopHits, z: f64) {
    writeln!(f, "#                                                               --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----").unwrap();
    writeln!(f, "# target name        accession  query name           accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target").unwrap();
    writeln!(f, "#------------------- ---------- -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------").unwrap();

    for hit in &th.hits {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        let evalue = z * hit.lnp.exp();
        let dom_evalue = if !hit.dcl.is_empty() {
            z * hit.dcl[0].lnp.exp()
        } else {
            evalue
        };
        let dom_score = if !hit.dcl.is_empty() {
            hit.dcl[0].bitscore
        } else {
            hit.score
        };

        writeln!(
            f,
            "{:<19} {:<10} {:<20} {:<10} {:>9.2e} {:>6.1} {:>5.1} {:>9.2e} {:>6.1} {:>5.1} {:>5.1} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {}",
            hit.name,
            if hit.acc.is_empty() { "-" } else { &hit.acc },
            qname,
            qacc.unwrap_or("-"),
            evalue,
            hit.score,
            0.0_f32,
            dom_evalue,
            dom_score,
            0.0_f32,
            hit.nexpected,
            hit.ndom,
            0, 0,
            hit.ndom,
            hit.ndom,
            hit.nreported,
            hit.nincluded,
            if hit.desc.is_empty() { "-" } else { &hit.desc },
        ).unwrap();
    }
}

fn write_domtblout(f: &mut std::fs::File, qname: &str, qacc: Option<&str>, th: &TopHits, z: f64, domz: f64) {
    writeln!(f, "#                                                                            --- full sequence --- -------------- this domain -------------   hmm coord   ali coord   env coord").unwrap();
    writeln!(f, "# target name        accession   tlen query name           accession   qlen   E-value  score  bias   #  of  c-Evalue  i-Evalue  score  bias  from    to  from    to  from    to  acc description of target").unwrap();
    writeln!(f, "#------------------- ---------- ----- -------------------- ---------- ----- --------- ------ ----- --- --- --------- --------- ------ ----- ----- ----- ----- ----- ----- ----- ---- ---------------------").unwrap();

    for hit in &th.hits {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        let evalue = z * hit.lnp.exp();

        for (di, dom) in hit.dcl.iter().enumerate() {
            let dom_evalue = domz * dom.lnp.exp();

            writeln!(
                f,
                "{:<19} {:<10} {:>5} {:<20} {:<10} {:>5} {:>9.2e} {:>6.1} {:>5.1} {:>3} {:>3} {:>9.2e} {:>9.2e} {:>6.1} {:>5.1} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:.2} {}",
                hit.name,
                if hit.acc.is_empty() { "-" } else { &hit.acc },
                0, // target length (not tracked yet)
                qname,
                qacc.unwrap_or("-"),
                0, // query length
                evalue,
                hit.score,
                0.0_f32,
                di + 1,
                hit.ndom,
                dom_evalue / z, // conditional E-value
                dom_evalue,
                dom.bitscore,
                0.0_f32,
                1,
                0,
                dom.iali,
                dom.jali,
                dom.ienv,
                dom.jenv,
                0.0_f32,
                if hit.desc.is_empty() { "-" } else { &hit.desc },
            ).unwrap();
        }
    }
}

/// Write Pfam-format tabular output (--pfamtblout).
/// Two sections: sequence scores, then domain scores.
fn write_pfamtblout(f: &mut std::fs::File, _qname: &str, _qacc: Option<&str>, th: &TopHits, z: f64, _domz: f64) {
    use std::io::Write;
    // Sequence scores section
    for hit in &th.hits {
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 { continue; }
        let evalue = z * hit.lnp.exp();
        let bias = hit.pre_score - hit.score;
        writeln!(f, "{:<20} {:6.1} {:9.2e} {:3} {:5.1} {:5.1}    {}",
            hit.name, hit.score, evalue, hit.ndom, hit.nexpected, bias,
            if hit.desc.is_empty() { "-" } else { &hit.desc },
        ).unwrap();
    }
}

/// Write alignment output in Stockholm format (-A).
fn write_ali_output(f: &mut std::fs::File, hmm: &hmmer_pure_rs::hmm::Hmm, th: &TopHits, _domz: f64, _textw: usize) {
    use std::io::Write;
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
        if hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED == 0 { continue; }
        for dom in &hit.dcl {
            if !dom.is_included { continue; }
            if let Some(ref ad) = dom.ad {
                writeln!(f, "{:<30} {}", hit.name, ad.aseq).unwrap();
            }
        }
    }
    writeln!(f, "//").unwrap();
}
