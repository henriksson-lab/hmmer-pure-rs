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

#[derive(Parser)]
#[command(
    name = "jackhmmer",
    about = "Iteratively search a protein sequence against a protein database"
)]
struct Args {
    /// Query sequence file (FASTA)
    seqfile: PathBuf,
    /// Target sequence database (FASTA)
    seqdb: PathBuf,

    /// Maximum number of iterations
    #[arg(short = 'N', default_value = "5")]
    max_iterations: usize,

    /// Report sequences <= this E-value threshold
    #[arg(short = 'E', default_value = "10.0")]
    e_value: f64,

    /// Include sequences <= this E-value threshold
    #[arg(long = "incE", default_value = "0.001")]
    inc_e: f64,

    /// Include domains <= this E-value threshold
    #[arg(long = "incdomE", default_value = "0.001")]
    incdom_e: f64,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

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

    /// Save HMM checkpoints to files <f>-<iteration>.hmm
    #[arg(long = "chkhmm")]
    chkhmm: Option<PathBuf>,

    /// Save alignment checkpoints to files <f>-<iteration>.sto
    #[arg(long = "chkali")]
    chkali: Option<PathBuf>,
}

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

    writeln!(
        out,
        "# jackhmmer :: iteratively search a protein sequence against a database"
    )
    .unwrap();
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
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
    if !query_sqf.read(&mut query_sq).unwrap() {
        eprintln!("Error: no query sequence found");
        std::process::exit(1);
    }

    // Read all target sequences
    let mut targets = Vec::new();
    {
        let mut sqf = sequence::open_seq_file(&args.seqdb, &abc).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });
        let mut sq = Sequence::new();
        while sqf.read(&mut sq).unwrap() {
            targets.push(sq.clone());
            sq.reuse();
        }
    }
    let z = targets.len() as f64;
    let query_tr = exact_match_query_trace(query_sq.n);

    let mut prev_included_names: Vec<String> = Vec::new();
    let mut prev_hmm: Option<hmmer_pure_rs::hmm::Hmm> = None;
    let mut final_hits: Option<TopHits> = None;
    let mut final_domz: Option<f64> = None;
    let mut final_model_len: Option<usize> = None;

    for iteration in 1..=args.max_iterations {
        writeln!(out, "@@ Round: {}", iteration).unwrap();

        // Build HMM for this iteration
        let hmm = if iteration == 1 {
            // First iteration: single-sequence HMM (phmmer-style)
            seqmodel::build_single_seq_hmm(
                &query_sq.name,
                &query_sq.dsq,
                query_sq.n,
                &abc,
                &bg,
                0.02,
                0.4,
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

            let Some(msa) = msa else {
                writeln!(out, "@@ No hits to build MSA from. Stopping.").unwrap();
                break;
            };
            builder::build_hmm_from_msa(&msa, &abc, &bg, 0.5, true)
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

        // Search in parallel
        use rayon::prelude::*;
        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = targets
            .par_iter()
            .filter_map(|sq| {
                let mut lb = local_bg.clone();
                let mut lgm = gm.clone();
                let mut lom = om.clone();
                let mut lpli = Pipeline::new();
                lpli.do_biasfilter = !args.nobias;
                lpli.do_null2 = !args.nonull2;
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
        th.sort_by_sortkey();
        {
            let mut tmp_pli = Pipeline::new();
            tmp_pli.do_biasfilter = !args.nobias;
            tmp_pli.do_null2 = !args.nonull2;
            tmp_pli.e_value_threshold = args.e_value;
            tmp_pli.inc_e = args.inc_e;
            tmp_pli.inc_dome = args.incdom_e;
            th.threshold(&tmp_pli, z, z);
            let domz = th.nreported.max(1) as f64;
            if domz != z {
                th.threshold(&tmp_pli, z, domz);
            }
            final_domz = Some(domz);
        }
        // Output results for this round
        writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.m).unwrap();
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
        for hit in &th.hits {
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
                continue;
            }
            let evalue = z * hit.lnp.exp();
            let best_dom = hit.dcl.iter().min_by(|a, b| a.lnp.total_cmp(&b.lnp));
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
            if let Some(msa) = hmmer_pure_rs::tophits::included_alignment(
                final_hits.as_ref().unwrap(),
                &abc,
                hmm.m,
                Some((&query_sq, &query_tr)),
                &format!("{}-i{}", query_sq.name, iteration),
            ) {
                write_msa_checkpoint(prefix, iteration, &msa);
            }
        }

        // Check convergence
        let n_new = new_included_names
            .iter()
            .filter(|name| !prev_included_names.iter().any(|prev| prev == *name))
            .count();

        if n_new == 0 && new_included_names.len() <= prev_included_names.len() && iteration > 1 {
            writeln!(out, "@@ CONVERGED (in {} rounds).", iteration).unwrap();
            break;
        }

        if iteration < args.max_iterations {
            writeln!(
                out,
                "@@ {} included, {} new. Continuing to next round.",
                new_included_names.len(),
                n_new
            )
            .unwrap();
        }

        prev_included_names = new_included_names;
    }

    if let Some(ref mut f) = tblout_file {
        if let Some(ref th) = final_hits {
            crate::subcmd::hmmsearch::write_tblout(f, &query_sq.name, None, th, z);
        }
        f.flush().unwrap();
    }
    if let Some(ref mut f) = domtblout_file {
        if let (Some(ref th), Some(domz), Some(qlen)) = (&final_hits, final_domz, final_model_len) {
            crate::subcmd::hmmsearch::write_domtblout(f, &query_sq.name, None, qlen, th, z, domz);
        }
        f.flush().unwrap();
    }

    writeln!(out, "//").unwrap();
    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

fn exact_match_query_trace(model_len: usize) -> Trace {
    let mut tr = Trace::new();
    tr.append(State::S, 0, 0);
    tr.append(State::N, 0, 0);
    tr.append(State::B, 0, 0);
    for i in 1..=model_len {
        tr.append(State::M, i, i);
    }
    tr.append(State::E, 0, 0);
    tr.append(State::C, 0, 0);
    tr.append(State::T, 0, 0);
    tr
}

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

fn write_msa_checkpoint(prefix: &PathBuf, iteration: usize, msa: &Msa) {
    let path = checkpoint_path(prefix, iteration, "sto");
    let mut file = std::fs::File::create(&path).unwrap_or_else(|e| {
        eprintln!("Error creating alignment checkpoint {}: {}", path.display(), e);
        std::process::exit(1);
    });
    write_stockholm_msa(&mut file, msa);
}

fn checkpoint_path(prefix: &PathBuf, iteration: usize, ext: &str) -> PathBuf {
    let mut os = prefix.as_os_str().to_os_string();
    os.push(format!("-{}.{}", iteration, ext));
    PathBuf::from(os)
}

fn write_stockholm_msa(out: &mut dyn Write, msa: &Msa) {
    let name_width = msa
        .sqname
        .iter()
        .map(|name| name.len())
        .max()
        .unwrap_or(0)
        .max("#=GC RF".len())
        .max(12);

    writeln!(out, "# STOCKHOLM 1.0").unwrap();
    writeln!(out).unwrap();
    if !msa.name.is_empty() {
        writeln!(out, "#=GF ID {}", msa.name).unwrap();
    }

    for (name, row) in msa.sqname.iter().zip(msa.aseq.iter()) {
        writeln!(
            out,
            "{:<width$} {}",
            name,
            String::from_utf8_lossy(row),
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
