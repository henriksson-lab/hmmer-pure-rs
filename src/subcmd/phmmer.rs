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
#[command(
    name = "phmmer",
    about = "Search a protein sequence against a protein database"
)]
struct Args {
    /// Query sequence file (FASTA)
    seqfile: PathBuf,
    /// Target sequence database (FASTA)
    seqdb: PathBuf,

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

    /// Gap open probability
    #[arg(long = "popen", default_value = "0.02", value_parser = parse_popen)]
    popen: f32,

    /// Gap extend probability
    #[arg(long = "pextend", default_value = "0.4", value_parser = parse_pextend)]
    pextend: f32,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Turn all heuristic filters off
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

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    domz_value: Option<f64>,

    /// Set RNG seed
    #[arg(long = "seed", default_value = "42")]
    seed: u32,
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
    let args = Args::parse_from(&args);

    logsum::p7_flogsuminit();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.cpu)
        .start_handler(|_| hmmer_pure_rs::util::simd_env::init())
        .build_global()
        .ok();

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

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

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

    // Read query sequences
    let mut query_sqf = sequence::open_seq_file(&args.seqfile, &abc).unwrap_or_else(|e| {
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
            let mut sqf = sequence::open_seq_file(&args.seqdb, &abc).unwrap_or_else(|e| {
                eprintln!("Error opening target database: {}", e);
                std::process::exit(1);
            });
            let mut sq = Sequence::new();
            while sqf.read(&mut sq).unwrap_or_else(|e| {
                eprintln!("Error reading target database: {}", e);
                std::process::exit(1);
            }) {
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
        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = sequences
            .par_iter()
            .filter_map(|sq| {
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
                lpli.seed = args.seed;
                lpli.new_model(&lgm);

                lb.set_length(sq.n);

                let mut lth = TopHits::new();
                if lpli.run(&mut lgm, &mut lom, &lb, &hmm, sq, &mut lth) {
                    lth.hits.into_iter().next()
                } else {
                    None
                }
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        let z = args.z_value.unwrap_or(sequences.len() as f64);
        th.sort_by_sortkey();
        {
            let mut tmp_pli = Pipeline::new();
            configure_thresholds(&mut tmp_pli, &args);
            tmp_pli.do_biasfilter = !args.nobias;
            tmp_pli.do_null2 = !args.nonull2;
            th.threshold(&tmp_pli, z, z);
            let domz = args.domz_value.unwrap_or(th.nreported.max(1) as f64);
            if domz != z {
                th.threshold(&tmp_pli, z, domz);
            }
        }
        let domz = args.domz_value.unwrap_or(th.nreported.max(1) as f64);

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

        let mut have_printed_incthresh = false;
        for hit in &th.hits {
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
                continue;
            }
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED == 0 && !have_printed_incthresh {
                writeln!(out, "  ------ inclusion threshold ------").unwrap();
                have_printed_incthresh = true;
            }
            let evalue = z * hit.lnp.exp();
            let best_dom = crate::subcmd::hmmsearch::best_domain(hit);
            let dom_evalue = best_dom.map(|d| z * d.lnp.exp()).unwrap_or(evalue);
            let dom_score = best_dom.map(|d| d.bitscore).unwrap_or(hit.score);
            let dom_bias = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);
            writeln!(
                out,
                "  {} {:6.1} {:5.1}  {} {:6.1} {:5.1}  {:4.1} {:2}  {:<9}{}",
                hmmer_pure_rs::output::fmt_evalue(evalue),
                hit.score,
                hit.bias,
                hmmer_pure_rs::output::fmt_evalue(dom_evalue),
                dom_score,
                dom_bias,
                hit.nexpected,
                hit.nreported,
                hit.name,
                if hit.desc.is_empty() { "" } else { &hit.desc },
            )
            .unwrap();
        }

        if th.nreported == 0 {
            writeln!(
                out,
                "\n   [No hits detected that satisfy reporting thresholds]"
            )
            .unwrap();
        }

        writeln!(out).unwrap();
        writeln!(
            out,
            "# Searched {} sequences ({} residues)",
            sequences.len(),
            total_residues
        )
        .unwrap();
        writeln!(out, "//").unwrap();

        if let Some(ref mut f) = tblout_file {
            crate::subcmd::hmmsearch::write_tblout(
                f,
                &query_sq.name,
                Some(&query_sq.acc),
                &th,
                z,
                true,
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
                true,
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

        query_sq.reuse();
    }
    if n_queries == 0 {
        eprintln!("Error: no sequences found in {}", args.seqfile.display());
        std::process::exit(1);
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
}
