//! nhmmscan — search sequence(s) against an HMM database.
//! Reverses hmmsearch: query=sequences, targets=HMMs.

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
#[command(name = "nhmmscan", about = "Search nucleotide sequence(s) against a DNA HMM database")]
struct Args {
    /// Query sequence file (FASTA)
    seqfile: PathBuf,
    /// HMM database file
    hmmdb: PathBuf,

    /// Report sequences <= this E-value threshold
    #[arg(short = 'E', default_value = "10.0")]
    e_value: f64,

    /// Include sequences <= this E-value threshold
    #[arg(long = "incE", default_value = "0.01")]
    inc_e: f64,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,
}

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    logsum::p7_flogsuminit();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.cpu)
        .build_global()
        .ok();

    // Read HMM database
    let hmms = hmmfile::read_hmm_file(&args.hmmdb).unwrap_or_else(|e| {
        eprintln!("Error reading HMM database: {}", e);
        std::process::exit(1);
    });

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "# nhmmscan :: search nucleotide sequence(s) against a DNA profile database").unwrap();
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(out, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -").unwrap();
    writeln!(out).unwrap();

    // For each query sequence, search all HMMs
    let abc = Alphabet::dna(); // assume amino for now
    let bg = Bg::new(&abc);

    let mut sqf = sequence::open_seq_file(&args.seqfile, &abc).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let mut sq = Sequence::new();
    while sqf.read(&mut sq).unwrap() {
        writeln!(out, "Query:       {}  [L={}]", sq.name, sq.n).unwrap();

        // Pre-build profiles for all HMMs (could be cached)
        use rayon::prelude::*;
        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = hmms
            .par_iter()
            .filter_map(|hmm| {
                let mut local_bg = bg.clone();
                local_bg.set_filter(hmm.m, &hmm.compo);
                local_bg.set_length(sq.n);

                let mut gm = Profile::new(hmm.m, &abc);
                profile::profile_config(hmm, &local_bg, &mut gm, sq.n as i32, P7_LOCAL);
                let mut om = OProfile::convert(&gm);

                let mut pli = Pipeline::new();
                pli.new_model(&gm);

                let mut th = TopHits::new();
                if pli.run(&mut gm, &mut om, &local_bg, hmm, &sq, &mut th) {
                    // Use the HMM name for the hit (in nhmmscan, targets are HMMs)
                    th.hits.into_iter().next().map(|mut hit| {
                        // Swap: in nhmmscan output, "target" is the HMM name
                        hit.name = hmm.name.clone();
                        hit.acc = hmm.acc.clone().unwrap_or_default();
                        hit.desc = hmm.desc.clone().unwrap_or_default();
                        hit
                    })
                } else {
                    None
                }
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        let z = hmms.len() as f64;
        th.sort_by_sortkey();
        {
            let mut tmp_pli = Pipeline::new();
            tmp_pli.e_value_threshold = args.e_value;
            tmp_pli.inc_e = args.inc_e;
            th.threshold(&tmp_pli, z, z);
        }

        writeln!(out, "Scores for complete sequence (score includes all domains):").unwrap();
        writeln!(out, "   --- full sequence ---   --- best 1 domain ---    -#dom-").unwrap();
        writeln!(out, "    E-value  score  bias    E-value  score  bias    exp  N  Model    Description").unwrap();
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
                hit.score, hit.bias,
                hmmer_pure_rs::output::fmt_evalue(dom_evalue),
                dom_score, hit.bias,
                hit.nexpected, hit.ndom,
                hit.name, hit.desc,
            ).unwrap();
        }

        if th.nreported == 0 {
            writeln!(out, "   [No targets detected that satisfy reporting thresholds]").unwrap();
        }
        writeln!(out, "\n//").unwrap();

        sq.reuse();
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}
