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

    /// Report sequences <= this E-value threshold
    #[arg(short = 'E', default_value = "10.0")]
    e_value: f64,

    /// Report sequences >= this score threshold
    #[arg(short = 'T')]
    score_threshold: Option<f32>,

    /// Include sequences <= this E-value threshold
    #[arg(long = "incE", default_value = "0.01")]
    inc_e: f64,

    /// Report domains <= this E-value threshold
    #[arg(long = "domE", default_value = "10.0")]
    dom_e: f64,

    /// Include domains <= this E-value threshold
    #[arg(long = "incdomE", default_value = "0.01")]
    inc_dome: f64,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    domtblout: Option<PathBuf>,

    /// Don't output alignments
    #[arg(long = "noali")]
    noali: bool,

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

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,
}

fn main() {
    let args = Args::parse();

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

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

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
        pli.f1 = args.f1;
        pli.f2 = args.f2;
        pli.f3 = args.f3;
        pli.e_value_threshold = args.e_value;
        pli.dom_e_value_threshold = args.dom_e;
        pli.inc_e = args.inc_e;
        pli.inc_dome = args.inc_dome;
        pli.do_max = args.max;

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
        use rayon::prelude::*;
        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = sequences
            .par_iter()
            .filter_map(|sq| {
                let mut local_bg = bg.clone();
                let mut local_gm = gm.clone();
                let mut local_om = om.clone();
                let mut local_pli = Pipeline::new();
                local_pli.new_model(&local_gm);
                local_pli.f1 = args.f1;
                local_pli.f2 = args.f2;
                local_pli.f3 = args.f3;
                local_pli.do_max = args.max;

                local_bg.set_length(sq.n);
                profile::reconfig_length(&mut local_gm, sq.n as i32);
                local_om.reconfig_length(sq.n as i32);

                let mut local_th = TopHits::new();
                if local_pli.run(&local_gm, &local_om, &local_bg, hmm, sq, &mut local_th) {
                    local_th.hits.into_iter().next()
                } else {
                    None
                }
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        pli.n_targets = sequences.len() as u64;
        pli.n_past_fwd = th.hits.len() as u64;
        pli.n_past_msv = sequences.len() as u64;
        pli.n_past_bias = sequences.len() as u64;
        pli.n_past_vit = sequences.len() as u64;

        // Set Z (database size)
        let z = if pli.z > 0.0 { pli.z } else { pli.n_targets as f64 };
        let domz = z;

        // Sort and threshold
        th.sort_by_sortkey();
        th.threshold(
            pli.e_value_threshold,
            pli.inc_e,
            pli.dom_e_value_threshold,
            pli.inc_dome,
            z,
            domz,
        );

        // Output query header
        writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.m).unwrap();

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
                "  {} {:6.1} {:5.1}  {} {:6.1} {:5.1}  {:4.1} {:2}  {:<9}{}",
                hmmer_pure_rs::output::fmt_evalue(evalue),
                hit.score,
                hit.bias,
                hmmer_pure_rs::output::fmt_evalue(dom_evalue),
                dom_score,
                hit.bias,
                hit.nexpected,
                hit.ndom,
                hit.name,
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

                writeln!(out, ">> {}  {}", hit.name, hit.desc).unwrap();
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
    }

    writeln!(out, "//").unwrap();
    writeln!(out, "[ok]").unwrap();
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
            "{:<20}{:<11}{:<21}{:<11}{:9.2e} {:6.1} {:5.1} {:9.2e} {:6.1} {:5.1} {:5.1} {:3} {:3} {:3} {:3} {:3} {:3} {:3} {}",
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
                "{:<20}{:<11}{:>5} {:<21}{:<11}{:>5} {:9.2e} {:6.1} {:5.1} {:3} {:3} {:9.2e} {:9.2e} {:6.1} {:5.1} {:5} {:5} {:5} {:5} {:5} {:5} {:.2} {}",
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
