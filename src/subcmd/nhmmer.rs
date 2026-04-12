//! nhmmer — search DNA/RNA HMM(s) against a nucleotide sequence database.
//! Simplified version: uses standard pipeline without FM-index.

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
#[command(name = "nhmmer", about = "Search DNA/RNA HMM(s) against a nucleotide sequence database")]
struct Args {
    /// HMM file or query sequence
    hmmfile: PathBuf,
    /// Target sequence database (FASTA)
    seqdb: PathBuf,

    /// Report sequences <= this E-value threshold
    #[arg(short = 'E', default_value = "10.0")]
    e_value: f64,

    /// Include sequences <= this E-value threshold
    #[arg(long = "incE", default_value = "0.01")]
    inc_e: f64,

    /// Use DNA alphabet
    #[arg(long)]
    dna: bool,

    /// Use RNA alphabet
    #[arg(long)]
    rna: bool,

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

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "# nhmmer :: search a DNA model against a DNA database").unwrap();
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(out, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -").unwrap();
    writeln!(out).unwrap();

    for hmm in &hmms {
        // Determine alphabet from HMM or flags
        let abc = match hmm.abc_type {
            hmmer_pure_rs::alphabet::AlphabetType::Dna => Alphabet::dna(),
            hmmer_pure_rs::alphabet::AlphabetType::Rna => Alphabet::rna(),
            _ => {
                if args.rna {
                    Alphabet::rna()
                } else {
                    Alphabet::dna()
                }
            }
        };
        let mut bg = Bg::new(&abc);
        bg.set_filter(hmm.m, &hmm.compo);

        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        let mut pli = Pipeline::new();
        pli.new_model(&gm);

        // Read targets
        let mut sequences = Vec::new();
        {
            let mut sqf = sequence::open_seq_file(&args.seqdb, &abc).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            });
            let mut sq = Sequence::new();
            while sqf.read(&mut sq).unwrap() {
                sequences.push(sq.clone());
                sq.reuse();
            }
        }

        // For nhmmer: use windowed scanning for long sequences
        use rayon::prelude::*;
        let window_size = (hmm.m * 4).max(1000); // window = 4x model length, min 1000
        let overlap = hmm.m * 2; // overlap between windows

        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = sequences
            .par_iter()
            .flat_map(|sq| {
                let mut hits = Vec::new();

                if sq.n <= window_size {
                    // Short sequence: score whole thing
                    let mut lb = bg.clone();
                    let mut lgm = gm.clone();
                    let mut lom = om.clone();
                    let mut lpli = Pipeline::new();
                    lpli.new_model(&lgm);
                    lb.set_length(sq.n);
                    profile::reconfig_length(&mut lgm, sq.n as i32);
                    lom.reconfig_length(sq.n as i32);
                    let mut lth = TopHits::new();
                    if lpli.run(&lgm, &lom, &lb, hmm, sq, &mut lth) {
                        hits.extend(lth.hits);
                    }
                } else {
                    // Long sequence: scan in windows
                    let mut pos = 0;
                    while pos < sq.n {
                        let win_end = (pos + window_size).min(sq.n);
                        let win_len = win_end - pos;

                        // Create window sub-sequence
                        let mut win_dsq = vec![hmmer_pure_rs::alphabet::DSQ_SENTINEL];
                        win_dsq.extend_from_slice(&sq.dsq[pos + 1..=win_end]);
                        win_dsq.push(hmmer_pure_rs::alphabet::DSQ_SENTINEL);

                        let win_sq = Sequence {
                            name: format!("{}/{}-{}", sq.name, pos + 1, win_end),
                            acc: sq.acc.clone(),
                            desc: sq.desc.clone(),
                            dsq: win_dsq,
                            n: win_len,
                            l: win_len,
                        };

                        let mut lb = bg.clone();
                        let mut lgm = gm.clone();
                        let mut lom = om.clone();
                        let mut lpli = Pipeline::new();
                        lpli.new_model(&lgm);
                        lb.set_length(win_len);
                        profile::reconfig_length(&mut lgm, win_len as i32);
                        lom.reconfig_length(win_len as i32);
                        let mut lth = TopHits::new();
                        if lpli.run(&lgm, &lom, &lb, hmm, &win_sq, &mut lth) {
                            // Adjust hit coordinates to global position
                            for hit in &mut lth.hits {
                                for dom in &mut hit.dcl {
                                    dom.iali += pos as i64;
                                    dom.jali += pos as i64;
                                    dom.ienv += pos as i64;
                                    dom.jenv += pos as i64;
                                }
                                hit.name = sq.name.clone();
                                hit.desc = sq.desc.clone();
                            }
                            hits.extend(lth.hits);
                        }

                        if pos + window_size >= sq.n {
                            break;
                        }
                        pos += window_size - overlap;
                    }
                }
                hits
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        let z = sequences.len() as f64;
        th.sort_by_sortkey();
        th.threshold(args.e_value, args.inc_e, args.e_value, args.inc_e, z, z);

        // Output
        writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.m).unwrap();
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
                hit.score, hit.bias,
                hmmer_pure_rs::output::fmt_evalue(dom_evalue),
                dom_score, hit.bias,
                hit.nexpected, hit.ndom,
                hit.name,
                if hit.desc.is_empty() { "" } else { &hit.desc },
            ).unwrap();
        }

        if th.nreported == 0 {
            writeln!(out, "   [No hits detected that satisfy reporting thresholds]").unwrap();
        }
        writeln!(out, "\n//").unwrap();
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}
