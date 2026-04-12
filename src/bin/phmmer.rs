//! phmmer — search a protein sequence against a protein database.
//! Builds a single-sequence HMM and searches with the hmmsearch pipeline.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::seqmodel;
use hmmer_pure_rs::sequence::{self, Sequence};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::TopHits;

#[derive(Parser)]
#[command(name = "phmmer", about = "Search a protein sequence against a protein database")]
struct Args {
    /// Query sequence file (FASTA)
    seqfile: PathBuf,
    /// Target sequence database (FASTA)
    seqdb: PathBuf,

    /// Report sequences <= this E-value threshold
    #[arg(short = 'E', default_value = "10.0")]
    e_value: f64,

    /// Include sequences <= this E-value threshold
    #[arg(long = "incE", default_value = "0.01")]
    inc_e: f64,

    /// Gap open probability
    #[arg(long = "popen", default_value = "0.02")]
    popen: f32,

    /// Gap extend probability
    #[arg(long = "pextend", default_value = "0.4")]
    pextend: f32,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    logsum::p7_flogsuminit();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.cpu)
        .build_global()
        .ok();

    let abc = Alphabet::amino();
    let bg = Bg::new(&abc);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "# phmmer :: search a protein sequence against a protein database").unwrap();
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(out, "# Copyright (C) 2023 Howard Hughes Medical Institute.").unwrap();
    writeln!(out, "# Freely distributed under the BSD open source license.").unwrap();
    writeln!(out, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -").unwrap();
    writeln!(out, "# query sequence file:             {}", args.seqfile.display()).unwrap();
    writeln!(out, "# target sequence database:        {}", args.seqdb.display()).unwrap();
    writeln!(out, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -").unwrap();
    writeln!(out).unwrap();

    // Read query sequences
    let mut query_sqf = sequence::open_seq_file(&args.seqfile, &abc).unwrap_or_else(|e| {
        eprintln!("Error opening query file: {}", e);
        std::process::exit(1);
    });

    let mut query_sq = Sequence::new();
    while query_sqf.read(&mut query_sq).unwrap() {
        // Build HMM from query sequence
        let hmm = seqmodel::build_single_seq_hmm(
            &query_sq.name,
            &query_sq.dsq,
            query_sq.n,
            &abc,
            &bg,
            args.popen,
            args.pextend,
        );

        let mut local_bg = bg.clone();
        local_bg.set_filter(hmm.m, &hmm.compo);

        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(&hmm, &local_bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        // Read target sequences
        let mut sequences = Vec::new();
        let mut total_residues: u64 = 0; // used for statistics
        {
            let mut sqf = sequence::open_seq_file(&args.seqdb, &abc).unwrap_or_else(|e| {
                eprintln!("Error opening target database: {}", e);
                std::process::exit(1);
            });
            let mut sq = Sequence::new();
            while sqf.read(&mut sq).unwrap() {
                total_residues += sq.n as u64;
                sequences.push(sq.clone());
                sq.reuse();
            }
        }

        // Search in parallel
        use rayon::prelude::*;
        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = sequences
            .par_iter()
            .filter_map(|sq| {
                let mut lb = local_bg.clone();
                let mut lgm = gm.clone();
                let mut lom = om.clone();
                let mut lpli = Pipeline::new();
                lpli.new_model(&lgm);

                lb.set_length(sq.n);
                profile::reconfig_length(&mut lgm, sq.n as i32);
                lom.reconfig_length(sq.n as i32);

                let mut lth = TopHits::new();
                if lpli.run(&lgm, &lom, &lb, &hmm, sq, &mut lth) {
                    lth.hits.into_iter().next()
                } else {
                    None
                }
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        let z = sequences.len() as f64;
        th.sort_by_sortkey();
        th.threshold(args.e_value, args.inc_e, args.e_value, args.inc_e, z, z);

        // Output
        writeln!(out, "Query:       {}  [L={}]", query_sq.name, query_sq.n).unwrap();
        writeln!(out, "Scores for complete sequences (score includes all domains):").unwrap();
        writeln!(out, "   --- full sequence ---   --- best 1 domain ---    -#dom-").unwrap();
        writeln!(out, "    E-value  score  bias    E-value  score  bias    exp  N  Sequence Description").unwrap();
        writeln!(out, "    ------- ------ -----    ------- ------ -----   ---- --  -------- -----------").unwrap();

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

        if th.nreported == 0 {
            writeln!(out, "\n   [No hits detected that satisfy reporting thresholds]").unwrap();
        }

        writeln!(out).unwrap();
        writeln!(out, "# Searched {} sequences ({} residues)", sequences.len(), total_residues).unwrap();
        writeln!(out, "//").unwrap();
        query_sq.reuse();
    }

    writeln!(out, "[ok]").unwrap();
}
