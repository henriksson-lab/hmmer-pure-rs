//! jackhmmer — iteratively search a protein sequence against a protein database.
//! Builds single-seq HMM, searches, collects hits into MSA, rebuilds, repeats.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::builder;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::msa::Msa;
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::seqmodel;
use hmmer_pure_rs::sequence::{self, Sequence};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::TopHits;
use hmmer_pure_rs::trace::{State, Trace};

const TARGET_BATCH_SIZE: usize = 256;

#[derive(Parser)]
#[command(
    name = "jackhmmer",
    about = "Iteratively search a protein sequence against a protein database"
)]
struct Args {
    /// Direct output to file <f>, not stdout
    #[arg(short = 'o')]
    output: Option<PathBuf>,

    /// Query sequence file (FASTA)
    seqfile: PathBuf,
    /// Target sequence database (FASTA)
    seqdb: PathBuf,

    /// Maximum number of iterations
    #[arg(short = 'N', default_value = "5", value_parser = parse_nonzero_usize)]
    max_iterations: usize,

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
        default_value = "0.001",
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
        default_value = "0.001",
        value_parser = parse_positive_f64,
        conflicts_with = "incdom_t"
    )]
    incdom_e: f64,

    /// Include domains >= this score threshold
    #[arg(long = "incdomT", conflicts_with = "incdom_e")]
    incdom_t: Option<f32>,

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

    /// Gap open probability for the single-sequence query model
    #[arg(long = "popen", default_value = "0.02", value_parser = parse_popen)]
    popen: f32,

    /// Gap extend probability for the single-sequence query model
    #[arg(long = "pextend", default_value = "0.4", value_parser = parse_pextend)]
    pextend: f32,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    nobias: bool,

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    nonull2: bool,

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    domz_value: Option<f64>,

    /// Random number seed
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    domtblout: Option<PathBuf>,

    /// Save HMM checkpoints to files <f>-<iteration>.hmm
    #[arg(long = "chkhmm")]
    chkhmm: Option<PathBuf>,

    /// Save alignment checkpoints to files <f>-<iteration>.sto
    #[arg(long = "chkali")]
    chkali: Option<PathBuf>,
}

fn parse_nonzero_usize(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid positive integer: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
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

/// Entry point for `jackhmmer`: iteratively search a protein query against a
/// protein database, rebuilding the HMM each round from included hits until
/// convergence or `-N` iterations.
///
/// Round 1 builds a single-sequence HMM (phmmer-style); subsequent rounds
/// reuse the previous round's included-hit MSA augmented with the original
/// query to rebuild the model. Each round runs the standard pipeline against
/// the target DB in bounded parallel batches, applies E/incE thresholds,
/// reports the canonical per-sequence tabular block, and optionally writes
/// `--chkhmm`/`--chkali`/`--tblout`/`--domtblout` checkpoints. Corresponds to
/// `serial_master()` in hmmer/src/jackhmmer.c.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);
    if args.seqdb == PathBuf::from("-") {
        eprintln!("Error: target sequence database may not be '-' for jackhmmer");
        std::process::exit(1);
    }

    logsum::p7_flogsuminit();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.cpu)
        .start_handler(|_| hmmer_pure_rs::util::simd_env::init())
        .build_global()
        .ok();

    let abc = Alphabet::amino();
    let bg = Bg::new(&abc);
    if args.seqfile != PathBuf::from("-") && count_query_sequences(&args.seqfile, &abc) > 1 {
        eprintln!(
            "Error: jackhmmer multi-query sequence input is not implemented yet; split the query FASTA or use C jackhmmer"
        );
        std::process::exit(1);
    }

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

    writeln!(
        out,
        "# jackhmmer :: iteratively search a protein sequence against a protein database"
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
        "# maximum iterations set to:       {}",
        args.max_iterations
    )
    .unwrap();
    if let Some(tblout) = &args.tblout {
        writeln!(
            out,
            "# per-seq hits tabular output:     {}",
            tblout.display()
        )
        .unwrap();
    }
    if let Some(domtblout) = &args.domtblout {
        writeln!(
            out,
            "# per-dom hits tabular output:     {}",
            domtblout.display()
        )
        .unwrap();
    }
    if args.max {
        writeln!(
            out,
            "# Max sensitivity mode:            on [all heuristic filters off]"
        )
        .unwrap();
    }
    if args.seed != 42 {
        if args.seed == 0 {
            writeln!(out, "# random number seed:              one-time arbitrary").unwrap();
        } else {
            writeln!(out, "# random number seed set to:       {}", args.seed).unwrap();
        }
    }
    writeln!(out, "# number of worker threads:        {}", args.cpu).unwrap();
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();

    // Read query sequence
    let mut query_sqf = sequence::open_seq_file(&args.seqfile, &abc).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    let mut query_sq = Sequence::new();
    if !query_sqf.read(&mut query_sq).unwrap_or_else(|e| {
        eprintln!("Error reading query file: {}", e);
        std::process::exit(1);
    }) {
        eprintln!("Error: no query sequence found");
        std::process::exit(1);
    }

    writeln!(out, "Query:       {}  [L={}]", query_sq.name, query_sq.n).unwrap();
    if !query_sq.acc.is_empty() {
        writeln!(out, "Accession:   {}", query_sq.acc).unwrap();
    }
    if !query_sq.desc.is_empty() {
        writeln!(out, "Description: {}", query_sq.desc).unwrap();
    }
    writeln!(out).unwrap();

    let query_tr = exact_match_query_trace(query_sq.n);

    let mut prev_included_names: Vec<String> = Vec::new();
    let mut prev_msa_nseq = 1usize;
    let mut prev_hmm: Option<hmmer_pure_rs::hmm::Hmm> = None;
    let mut final_hits: Option<TopHits> = None;
    let mut final_z: Option<f64> = None;
    let mut final_domz: Option<f64> = None;
    let mut final_model_len: Option<usize> = None;

    for iteration in 1..=args.max_iterations {
        // Build HMM for this iteration
        let hmm = if iteration == 1 {
            // First iteration: single-sequence HMM (phmmer-style)
            seqmodel::build_single_seq_hmm(
                &query_sq.name,
                &query_sq.dsq,
                query_sq.n,
                &abc,
                &bg,
                args.popen,
                args.pextend,
            )
        } else {
            let prev_hmm = prev_hmm
                .as_ref()
                .expect("jackhmmer rebuild requested without previous-round HMM");
            let prev_hits = final_hits
                .as_ref()
                .expect("jackhmmer rebuild requested without previous-round hits");
            let msa = hmmer_pure_rs::tophits::included_alignment(
                prev_hits,
                &abc,
                prev_hmm.m,
                Some((&query_sq, &query_tr)),
                &format!("{}-i{}", query_sq.name, iteration - 1),
            );

            let Some(mut msa) = msa else {
                writeln!(out, "@@ No hits to build MSA from. Stopping.").unwrap();
                break;
            };
            if !query_sq.desc.is_empty() {
                msa.desc = Some(query_sq.desc.clone());
            }
            msa.author = Some("jackhmmer (HMMER 3.4)".to_string());
            let mut hmm = builder::build_hmm_from_msa(&msa, &abc, &bg, 0.5, true);
            if !query_sq.desc.is_empty() {
                hmm.desc = Some(query_sq.desc.clone());
                hmm.flags |= hmmer_pure_rs::hmm::P7H_DESC;
            }
            writeln!(out, "@@").unwrap();
            writeln!(out, "@@ Round:                  {}", iteration).unwrap();
            writeln!(
                out,
                "@@ Included in MSA:        {} subsequences (query + {} subseqs from {} targets)",
                msa.nseq,
                msa.nseq.saturating_sub(1),
                prev_included_names.len()
            )
            .unwrap();
            writeln!(out, "@@ Model size:             {} positions", hmm.m).unwrap();
            writeln!(out, "@@").unwrap();
            writeln!(out).unwrap();
            hmm
        };

        if let Some(prefix) = &args.chkhmm {
            write_hmm_checkpoint(prefix, iteration, &hmm);
        }

        // Configure profile and search
        let mut local_bg = bg.clone();
        local_bg.set_filter(hmm.m, &hmm.compo);
        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(&hmm, &local_bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        // Search the target DB in bounded batches so RSS scales with
        // the batch size instead of the full database size.
        use rayon::prelude::*;
        let mut all_hits = Vec::new();
        let mut z = 0usize;
        {
            let mut sqf = sequence::open_seq_file(&args.seqdb, &abc).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            });
            let mut sq = Sequence::new();
            let mut batch = Vec::with_capacity(TARGET_BATCH_SIZE);

            loop {
                while batch.len() < TARGET_BATCH_SIZE {
                    match sqf.read(&mut sq) {
                        Ok(true) => {
                            batch.push(sq.clone());
                            z += 1;
                            sq.reuse();
                        }
                        Ok(false) => break,
                        Err(e) => {
                            eprintln!("Error: {}", e);
                            std::process::exit(1);
                        }
                    }
                }

                if batch.is_empty() {
                    break;
                }

                let batch_hits: Vec<hmmer_pure_rs::tophits::Hit> = batch
                    .par_iter()
                    .filter_map(|sq| {
                        let mut lb = local_bg.clone();
                        let mut lgm = gm.clone();
                        let mut lom = om.clone();
                        let mut lpli = Pipeline::new();
                        configure_pipeline(&mut lpli, &args);
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
                all_hits.extend(batch_hits);
                batch.clear();
            }
        }

        let mut th = TopHits::new();
        th.hits = all_hits;
        let z = args.z_value.unwrap_or(z as f64);
        final_z = Some(z);
        th.sort_by_sortkey();
        {
            let mut tmp_pli = Pipeline::new();
            configure_pipeline(&mut tmp_pli, &args);
            tmp_pli.do_biasfilter = !args.nobias;
            tmp_pli.do_null2 = !args.nonull2;
            tmp_pli.seed = args.seed;
            th.threshold(&tmp_pli, z, z);
            let domz = args.domz_value.unwrap_or(th.nreported.max(1) as f64);
            if domz != z {
                th.threshold(&tmp_pli, z, domz);
            }
            final_domz = Some(domz);
        }
        // Output results for this round
        if iteration > 1 {
            writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.m).unwrap();
            if !query_sq.acc.is_empty() {
                writeln!(out, "Accession:   {}", query_sq.acc).unwrap();
            }
            if !query_sq.desc.is_empty() {
                writeln!(out, "Description: {}", query_sq.desc).unwrap();
            }
            writeln!(out).unwrap();
        }
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

        let mut new_included_names = Vec::new();
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

            if hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED != 0 {
                new_included_names.push(hit.name.clone());
            }
        }

        if th.nreported == 0 {
            writeln!(
                out,
                "   [No hits detected that satisfy reporting thresholds]"
            )
            .unwrap();
        }
        writeln!(out).unwrap();
        final_hits = Some(th);
        final_model_len = Some(hmm.m);
        prev_hmm = Some(hmm.clone());

        if let Some(prefix) = &args.chkali {
            if let Some(mut msa) = hmmer_pure_rs::tophits::included_alignment(
                final_hits.as_ref().unwrap(),
                &abc,
                hmm.m,
                Some((&query_sq, &query_tr)),
                &format!("{}-i{}", query_sq.name, iteration),
            ) {
                if !query_sq.desc.is_empty() {
                    msa.desc = Some(query_sq.desc.clone());
                }
                msa.author = Some("jackhmmer (HMMER 3.4)".to_string());
                write_msa_checkpoint(prefix, iteration, &msa);
            }
        }

        // Check convergence
        let n_new = new_included_names
            .iter()
            .filter(|name| !prev_included_names.iter().any(|prev| prev == *name))
            .count();
        let msa_nseq = new_included_names.len() + 1;

        writeln!(out, "@@ New targets included:   {}", n_new).unwrap();
        writeln!(
            out,
            "@@ New alignment includes: {} subseqs (was {}), including original query",
            msa_nseq, prev_msa_nseq
        )
        .unwrap();

        if n_new == 0 && new_included_names.len() <= prev_included_names.len() && iteration > 1 {
            writeln!(out, "@@").unwrap();
            writeln!(out, "@@ CONVERGED (in {} rounds). ", iteration).unwrap();
            writeln!(out, "@@").unwrap();
            writeln!(out).unwrap();
            break;
        }

        if iteration < args.max_iterations {
            writeln!(out, "@@ Continuing to next round.").unwrap();
            writeln!(out).unwrap();
        }

        prev_included_names = new_included_names;
        prev_msa_nseq = msa_nseq;
    }

    if let Some(ref mut f) = tblout_file {
        if let (Some(ref th), Some(z)) = (&final_hits, final_z) {
            crate::subcmd::hmmsearch::write_tblout(
                f,
                &query_sq.name,
                Some(&query_sq.acc),
                th,
                z,
                true,
            );
        }
        f.flush().unwrap();
    }
    if let Some(ref mut f) = domtblout_file {
        if let (Some(ref th), Some(z), Some(domz), Some(qlen)) =
            (&final_hits, final_z, final_domz, final_model_len)
        {
            crate::subcmd::hmmsearch::write_domtblout(
                f,
                &query_sq.name,
                Some(&query_sq.acc),
                qlen,
                th,
                z,
                domz,
                true,
            );
        }
        f.flush().unwrap();
    }

    writeln!(out, "//").unwrap();
    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

/// Build a degenerate trace for the round-1 single-sequence query that simply
/// walks B -> M1..Mn -> E. Used so the query can be re-included verbatim when
/// constructing the included-hit MSA for the next round.
fn exact_match_query_trace(model_len: usize) -> Trace {
    let mut tr = Trace::new();
    tr.append(State::B, 0, 0);
    for i in 1..=model_len {
        tr.append(State::M, i, i);
    }
    tr.append(State::E, 0, 0);
    tr.m = model_len;
    tr.l = model_len;
    tr
}

/// Write the HMM produced at `iteration` to `<prefix>-<iteration>.hmm`.
fn write_hmm_checkpoint(prefix: &PathBuf, iteration: usize, hmm: &hmmer_pure_rs::hmm::Hmm) {
    let path = checkpoint_path(prefix, iteration, "hmm");
    let mut file = std::fs::File::create(&path).unwrap_or_else(|e| {
        eprintln!("Error creating HMM checkpoint {}: {}", path.display(), e);
        std::process::exit(1);
    });
    hmmer_pure_rs::hmmfile::write_hmm(&mut file, hmm).unwrap_or_else(|e| {
        eprintln!("Error writing HMM checkpoint {}: {}", path.display(), e);
        std::process::exit(1);
    });
}

/// Write the included-hit alignment for `iteration` to `<prefix>-<iteration>.sto`.
fn write_msa_checkpoint(prefix: &PathBuf, iteration: usize, msa: &Msa) {
    let path = checkpoint_path(prefix, iteration, "sto");
    let mut file = std::fs::File::create(&path).unwrap_or_else(|e| {
        eprintln!(
            "Error creating alignment checkpoint {}: {}",
            path.display(),
            e
        );
        std::process::exit(1);
    });
    write_stockholm_msa(&mut file, msa);
}

/// Compose a checkpoint filename of the form `<prefix>-<iteration>.<ext>`.
fn checkpoint_path(prefix: &PathBuf, iteration: usize, ext: &str) -> PathBuf {
    let mut os = prefix.as_os_str().to_os_string();
    os.push(format!("-{}.{}", iteration, ext));
    PathBuf::from(os)
}

/// Render an in-memory `Msa` as a Stockholm 1.0 file, emitting `#=GF`,
/// per-sequence `#=GS` descriptions, residue rows, per-row `#=GR ... PP`,
/// and `#=GC PP_cons`/`#=GC RF` consensus annotation.
fn write_stockholm_msa(out: &mut dyn Write, msa: &Msa) {
    let name_width = msa
        .sqname
        .iter()
        .map(|name| name.len())
        .max()
        .unwrap_or(0)
        .max("#=GC RF".len())
        .max("#=GC PP_cons".len())
        .max(12);
    let gs_name_width = "#=GS ".len() + name_width;
    let gr_name_width = "#=GR ".len() + name_width + " PP".len();

    writeln!(out, "# STOCKHOLM 1.0").unwrap();
    if !msa.name.is_empty() {
        writeln!(out, "#=GF ID {}", msa.name).unwrap();
    }
    if let Some(desc) = &msa.desc {
        if !desc.is_empty() {
            writeln!(out, "#=GF DE {}", desc).unwrap();
        }
    }
    if let Some(author) = &msa.author {
        if !author.is_empty() {
            writeln!(out, "#=GF AU {}", author).unwrap();
        }
    }
    writeln!(out).unwrap();

    for (name, desc) in msa.sqname.iter().zip(msa.sqdesc.iter()) {
        if !desc.is_empty() {
            writeln!(
                out,
                "{:<width$} DE {}",
                format!("#=GS {}", name),
                desc,
                width = gs_name_width
            )
            .unwrap();
        }
    }
    if !msa.sqdesc.iter().all(|desc| desc.is_empty()) {
        writeln!(out).unwrap();
    }

    for ((name, row), pp) in msa.sqname.iter().zip(msa.aseq.iter()).zip(msa.pp.iter()) {
        let rendered_row: Vec<u8> = row
            .iter()
            .map(|&ch| match ch {
                b'.' => b'-',
                b'a'..=b'z' => ch.to_ascii_uppercase(),
                _ => ch,
            })
            .collect();
        writeln!(
            out,
            "{:<width$} {}",
            name,
            String::from_utf8_lossy(&rendered_row),
            width = name_width
        )
        .unwrap();
        if let Some(pp) = pp {
            writeln!(
                out,
                "{:<width$} {}",
                format!("#=GR {} PP", name),
                String::from_utf8_lossy(pp),
                width = gr_name_width
            )
            .unwrap();
        }
    }
    if let Some(pp_cons) = &msa.pp_cons {
        writeln!(
            out,
            "{:<width$} {}",
            "#=GC PP_cons",
            String::from_utf8_lossy(pp_cons),
            width = name_width
        )
        .unwrap();
    }
    if let Some(rf) = &msa.rf {
        writeln!(
            out,
            "{:<width$} {}",
            "#=GC RF",
            String::from_utf8_lossy(rf),
            width = name_width
        )
        .unwrap();
    }
    writeln!(out, "//").unwrap();
}

fn configure_pipeline(pli: &mut Pipeline, args: &Args) {
    pli.e_value_threshold = args.e_value;
    pli.dom_e_value_threshold = args.dom_e;
    pli.inc_e = args.inc_e;
    pli.inc_dome = args.incdom_e;
    pli.do_max = args.max;
    pli.f1 = args.f1;
    pli.f2 = args.f2;
    pli.f3 = args.f3;

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
    if let Some(t) = args.incdom_t {
        pli.inc_dom_t = Some(t);
        pli.incdom_by_e = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jackhmmer_parses_c_output_file_option() {
        let args =
            Args::try_parse_from(["jackhmmer", "-o", "out.txt", "query.fa", "targets.fa"]).unwrap();
        assert_eq!(args.output, Some(PathBuf::from("out.txt")));
        assert_eq!(args.seqfile, PathBuf::from("query.fa"));
        assert_eq!(args.seqdb, PathBuf::from("targets.fa"));
    }

    #[test]
    fn jackhmmer_parses_c_threshold_and_acceleration_options() {
        let args = Args::try_parse_from([
            "jackhmmer",
            "-T",
            "25",
            "--domT",
            "3",
            "--incT",
            "20",
            "--incdomT",
            "4",
            "-Z",
            "99",
            "--domZ",
            "7",
            "--seed",
            "5",
            "query.fa",
            "targets.fa",
        ])
        .unwrap();
        assert_eq!(args.score_threshold, Some(25.0));
        assert_eq!(args.dom_t, Some(3.0));
        assert_eq!(args.inc_t, Some(20.0));
        assert_eq!(args.incdom_t, Some(4.0));
        assert_eq!(args.z_value, Some(99.0));
        assert_eq!(args.domz_value, Some(7.0));
        assert_eq!(args.seed, 5);

        let args = Args::try_parse_from([
            "jackhmmer",
            "--F1",
            "0.1",
            "--F2",
            "0.2",
            "--F3",
            "0.3",
            "query.fa",
            "targets.fa",
        ])
        .unwrap();
        assert_eq!(args.f1, 0.1);
        assert_eq!(args.f2, 0.2);
        assert_eq!(args.f3, 0.3);
    }
}
