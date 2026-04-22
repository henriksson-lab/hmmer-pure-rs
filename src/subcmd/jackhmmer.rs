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
    #[arg(long = "incE", default_value = "0.01")]
    inc_e: f64,

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

    let mut prev_included: Vec<String> = Vec::new();
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
            // Subsequent iterations: build from MSA of included hits
            // Create MSA from query + included hits
            let mut aseqs = Vec::new();
            let mut sqnames = Vec::new();

            // Add query
            let query_text = abc.textize(&query_sq.dsq, query_sq.n);
            aseqs.push(query_text.into_bytes());
            sqnames.push(query_sq.name.clone());

            // Add included hits (their raw sequences)
            for target in &targets {
                if prev_included.contains(&target.name) {
                    let text = abc.textize(&target.dsq, target.n);
                    aseqs.push(text.into_bytes());
                    sqnames.push(target.name.clone());
                }
            }

            if aseqs.len() <= 1 {
                writeln!(out, "@@ No hits to build MSA from. Stopping.").unwrap();
                break;
            }

            // Pad sequences to same length for MSA
            let max_len = aseqs.iter().map(|s| s.len()).max().unwrap_or(0);
            for seq in &mut aseqs {
                seq.resize(max_len, b'-');
            }

            let msa = Msa {
                name: format!("{}-i{}", query_sq.name, iteration),
                sqname: sqnames,
                aseq: aseqs,
                nseq: prev_included.len() + 1,
                alen: max_len,
                rf: None,
            };

            builder::build_hmm_from_msa(&msa, &abc, &bg, 0.5)
        };

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

        let mut new_included = Vec::new();
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
                new_included.push(hit.name.clone());
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

        // Check convergence
        let n_new = new_included
            .iter()
            .filter(|name| !prev_included.contains(name))
            .count();

        if n_new == 0 && new_included.len() <= prev_included.len() && iteration > 1 {
            writeln!(out, "@@ CONVERGED (in {} rounds).", iteration).unwrap();
            break;
        }

        if iteration < args.max_iterations {
            writeln!(
                out,
                "@@ {} included, {} new. Continuing to next round.",
                new_included.len(),
                n_new
            )
            .unwrap();
        }

        prev_included = new_included;
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
