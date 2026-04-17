//! nhmmer — search DNA/RNA HMM(s) against a nucleotide sequence database.
//! Uses SSV long-target filter for genome-scale sequences, matching C HMMER's nhmmer.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::{Alphabet, DSQ_SENTINEL};
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::sequence::{self, Sequence};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::TopHits;

#[derive(Parser)]
#[command(
    name = "nhmmer",
    about = "Search DNA/RNA HMM(s) against a nucleotide sequence database"
)]
struct Args {
    /// HMM file, alignment file, or query sequence
    hmmfile: PathBuf,
    /// Target sequence database (FASTA)
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

    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    show_acc: bool,

    /// Unlimit ASCII text output line width
    #[arg(long = "notextw")]
    notextw: bool,

    /// Set max width of ASCII text output lines
    #[arg(long = "textw", default_value = "120")]
    textw: usize,

    // --- Reporting thresholds ---
    /// Report sequences <= this E-value threshold
    #[arg(short = 'E', default_value = "10.0")]
    e_value: f64,

    /// Report sequences >= this score threshold
    #[arg(short = 'T')]
    score_threshold: Option<f32>,

    /// Include sequences <= this E-value threshold
    #[arg(long = "incE", default_value = "0.01")]
    inc_e: f64,

    /// Include sequences >= this score threshold
    #[arg(long = "incT")]
    inc_t: Option<f32>,

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
    /// Turn all heuristic filters off (less speed, more power)
    #[arg(long = "max")]
    max: bool,

    /// Stage 1 (SSV) threshold
    #[arg(long = "F1", default_value = "0.02")]
    f1: f64,

    /// Stage 2 (Vit) threshold
    #[arg(long = "F2", default_value = "3e-3")]
    f2: f64,

    /// Stage 3 (Fwd) threshold
    #[arg(long = "F3", default_value = "3e-5")]
    f3: f64,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    nobias: bool,

    // --- Alphabet selection ---
    /// Use DNA alphabet
    #[arg(long)]
    dna: bool,

    /// Use RNA alphabet
    #[arg(long)]
    rna: bool,

    // --- Expert options ---
    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    nonull2: bool,

    /// Set database size (Megabases) for E-value calculation
    #[arg(short = 'Z')]
    z_value: Option<f64>,

    /// Set RNG seed (0: one-time arbitrary seed)
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Only search the top strand
    #[arg(long = "watson")]
    watson: bool,

    /// Only search the bottom strand
    #[arg(long = "crick")]
    crick: bool,

    /// Window length (max expected hit length)
    #[arg(long = "w_length")]
    w_length: Option<usize>,

    /// Assert target sequence file format
    #[arg(long = "tformat")]
    tformat: Option<String>,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,
}

/// Threshold for using long-target SSV filter vs standard pipeline.
/// Sequences longer than this use SSV longtarget windowed scanning.
const LONG_TARGET_THRESHOLD: usize = 4000;

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    logsum::p7_flogsuminit();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.cpu)
        .start_handler(|_| hmmer_pure_rs::util::simd_env::init())
        .build_global()
        .ok();

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    // Output destination
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

    writeln!(out, "# nhmmer :: search a DNA model against a DNA database").unwrap();
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();

    // Strand selection
    let do_watson = !args.crick; // top strand unless --crick only
    let do_crick = !args.watson; // bottom strand unless --watson only

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
        let bg = Bg::new(&abc);

        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        let mut pli = Pipeline::new();
        pli.new_model(&gm);
        pli.f1 = args.f1;
        pli.f2 = args.f2;
        pli.f3 = args.f3;
        pli.e_value_threshold = args.e_value;
        pli.inc_e = args.inc_e;
        pli.do_max = args.max;
        pli.seed = args.seed;
        if args.nobias {
            pli.do_biasfilter = false;
        }
        if args.nonull2 {
            pli.do_null2 = false;
        }

        if let Some(t) = args.score_threshold {
            pli.t = Some(t);
            pli.by_e = false;
        }
        if let Some(t) = args.inc_t {
            pli.inc_t = Some(t);
            pli.inc_by_e = false;
        }

        // Model-specific cutoffs
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

        // Database size override (nhmmer uses Megabases)
        if let Some(z) = args.z_value {
            pli.z = z * 1_000_000.0; // convert Mb to bases
            pli.z_setby = hmmer_pure_rs::pipeline::ZSetBy::Option;
        }

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

        use rayon::prelude::*;

        let max_length = if hmm.max_length > 0 {
            hmm.max_length
        } else {
            (hmm.m * 4) as i32
        };
        let _w_length = args.w_length.unwrap_or(0);
        let f1 = args.f1;
        let do_max = args.max;
        let nobias = args.nobias;
        let nonull2 = args.nonull2;
        let seed = args.seed;

        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = sequences
            .par_iter()
            .flat_map(|sq| {
                let mut hits = Vec::new();

                // Search top strand (watson)
                if do_watson {
                    hits.extend(search_sequence(
                        sq, hmm, &gm, &om, &bg, max_length, f1, do_max, nobias, nonull2, seed,
                        false,
                    ));
                }

                // Search complement strand (crick)
                if do_crick && abc.complement.is_some() {
                    let mut rc_dsq = sq.dsq.clone();
                    abc.revcomp(&mut rc_dsq, sq.n);
                    let rc_sq = Sequence {
                        name: sq.name.clone(),
                        acc: sq.acc.clone(),
                        desc: sq.desc.clone(),
                        dsq: rc_dsq,
                        n: sq.n,
                        l: sq.l,
                    };
                    let mut rc_hits = search_sequence(
                        &rc_sq, hmm, &gm, &om, &bg, max_length, f1, do_max, nobias, nonull2, seed,
                        true,
                    );
                    // Convert complement coordinates back to forward strand
                    for hit in &mut rc_hits {
                        for dom in &mut hit.dcl {
                            let orig_i = sq.n as i64 - dom.jali + 1;
                            let orig_j = sq.n as i64 - dom.iali + 1;
                            dom.iali = orig_i;
                            dom.jali = orig_j;
                            let orig_ienv = sq.n as i64 - dom.jenv + 1;
                            let orig_jenv = sq.n as i64 - dom.ienv + 1;
                            dom.ienv = orig_ienv;
                            dom.jenv = orig_jenv;
                        }
                    }
                    hits.extend(rc_hits);
                }

                hits
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        pli.n_targets = sequences.len() as u64;

        // nhmmer E-value space is in residues, not sequences. Count each strand
        // searched once (C HMMER reports this as "residues searched").
        let total_residues: usize = sequences.iter().map(|s| s.n).sum();
        let strand_multiplier = (do_watson as usize) + (do_crick as usize);
        let residues_searched = (total_residues * strand_multiplier.max(1)) as f64;

        let z = match pli.z_setby {
            hmmer_pure_rs::pipeline::ZSetBy::Option => pli.z,
            hmmer_pure_rs::pipeline::ZSetBy::Ntargets => residues_searched,
        };
        if pli.z_setby == hmmer_pure_rs::pipeline::ZSetBy::Ntargets {
            pli.z = residues_searched;
        }
        th.sort_by_sortkey();
        th.threshold(&pli, z, z);

        // Output
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
                "  {} {:6.1} {:5.1}  {} {:6.1} {:5.1}  {:4.1} {:2}  {} {}",
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
                "   [No hits detected that satisfy reporting thresholds]"
            )
            .unwrap();
        }

        // Tblout
        if let Some(ref mut f) = tblout_file {
            hmmer_pure_rs::output::write_tblout(f, &hmm.name, hmm.acc.as_deref(), &th, z);
        }
        if let Some(ref mut f) = domtblout_file {
            hmmer_pure_rs::output::write_domtblout(f, &hmm.name, hmm.acc.as_deref(), &th, z, z);
        }

        writeln!(out, "\n//").unwrap();
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

/// Search a single sequence (short or long) and return hits.
fn search_sequence(
    sq: &Sequence,
    hmm: &hmmer_pure_rs::hmm::Hmm,
    gm: &Profile,
    om: &OProfile,
    bg: &Bg,
    max_length: i32,
    f1: f64,
    do_max: bool,
    nobias: bool,
    nonull2: bool,
    seed: u32,
    _is_complement: bool,
) -> Vec<hmmer_pure_rs::tophits::Hit> {
    if sq.n <= LONG_TARGET_THRESHOLD {
        // Short sequence: standard pipeline
        let mut lgm = gm.clone();
        let mut lom = om.clone();
        let lb = bg.clone();
        let mut lpli = Pipeline::new();
        lpli.new_model(&lgm);
        lpli.f1 = f1;
        lpli.do_max = do_max;
        lpli.seed = seed;
        if nobias {
            lpli.do_biasfilter = false;
        }
        if nonull2 {
            lpli.do_null2 = false;
        }
        let mut lth = TopHits::new();
        if lpli.run(&mut lgm, &mut lom, &lb, hmm, sq, &mut lth) {
            return lth.hits;
        }
        return Vec::new();
    }

    // Long sequence: SSV longtarget
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            return search_longtarget(
                sq, hmm, gm, om, bg, max_length, f1, do_max, nobias, nonull2, seed,
            );
        }
    }
    Vec::new()
}

/// Search a long target sequence using SSV longtarget filter.
/// Returns hits found in windows identified by the SSV filter.
#[cfg(target_arch = "x86_64")]
fn search_longtarget(
    sq: &Sequence,
    hmm: &hmmer_pure_rs::hmm::Hmm,
    gm: &Profile,
    om: &OProfile,
    bg: &Bg,
    max_length: i32,
    _f1: f64,
    _do_max: bool,
    _nobias: bool,
    _nonull2: bool,
    _seed: u32,
) -> Vec<hmmer_pure_rs::tophits::Hit> {
    use hmmer_pure_rs::simd::ssv_longtarget;

    // Phase 1: SSV longtarget filter — find candidate windows
    let mut windows =
        unsafe { ssv_longtarget::ssv_filter_longtarget(&sq.dsq, sq.n, om, bg, 0.02, max_length) };

    if windows.is_empty() {
        return Vec::new();
    }

    // Extend and merge overlapping windows
    let ml = if max_length > 0 {
        max_length as usize
    } else {
        hmm.m * 4
    };
    ssv_longtarget::extend_and_merge_windows(&mut windows, ml, sq.n);

    // Phase 2: Run full pipeline on each window
    let mut all_hits = Vec::new();

    for win in &windows {
        let win_start = win.n;
        let win_end = (win.n + win.length - 1).min(sq.n);
        let win_len = win_end - win_start + 1;
        if win_len < hmm.m {
            continue;
        }

        // Create window sub-sequence
        let mut win_dsq = vec![DSQ_SENTINEL];
        win_dsq.extend_from_slice(&sq.dsq[win_start..=win_end]);
        win_dsq.push(DSQ_SENTINEL);

        let win_sq = Sequence {
            name: sq.name.clone(),
            acc: sq.acc.clone(),
            desc: sq.desc.clone(),
            dsq: win_dsq,
            n: win_len,
            l: win_len,
        };

        let mut lgm = gm.clone();
        let mut lom = om.clone();
        let lb = bg.clone();
        let mut lpli = Pipeline::new();
        lpli.new_model(&lgm);
        // SSV already ran; still apply Viterbi (F2) and Forward (F3) filters
        // per window so we don't emit one Hit for every SSV candidate.

        let mut lth = TopHits::new();
        if lpli.run(&mut lgm, &mut lom, &lb, hmm, &win_sq, &mut lth) {
            // Adjust coordinates to global sequence position
            for hit in &mut lth.hits {
                hit.name = sq.name.clone();
                hit.desc = sq.desc.clone();
                for dom in &mut hit.dcl {
                    dom.iali += (win_start - 1) as i64;
                    dom.jali += (win_start - 1) as i64;
                    dom.ienv += (win_start - 1) as i64;
                    dom.jenv += (win_start - 1) as i64;
                    if let Some(ref mut ad) = dom.ad {
                        ad.sqfrom += win_start - 1;
                        ad.sqto += win_start - 1;
                    }
                }
            }
            all_hits.extend(lth.hits);
        }
    }

    all_hits
}
