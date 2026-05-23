//! nhmmscan — search sequence(s) against an HMM database.
//! Reverses hmmsearch: query=sequences, targets=HMMs.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmmfile_binary;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::pipeline::{BitCutoff, Pipeline};
use hmmer_pure_rs::pressed;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::sequence::{self, Sequence};
use hmmer_pure_rs::tophits::TopHits;

#[derive(Parser)]
#[command(
    name = "nhmmscan",
    about = "Search nucleotide sequence(s) against a DNA HMM database"
)]
struct Args {
    /// Direct output to file <f>, not stdout
    #[arg(short = 'o')]
    output: Option<PathBuf>,

    /// HMM database file
    hmmdb: PathBuf,
    /// Query sequence file (FASTA)
    seqfile: PathBuf,

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

    /// Use model's GA gathering cutoffs to set all thresholding
    #[arg(
        long = "cut_ga",
        conflicts_with_all = ["cut_tc", "cut_nc", "e_value", "score_threshold", "inc_e", "inc_t"]
    )]
    cut_ga: bool,

    /// Use model's NC noise cutoffs to set all thresholding
    #[arg(
        long = "cut_nc",
        conflicts_with_all = ["cut_ga", "cut_tc", "e_value", "score_threshold", "inc_e", "inc_t"]
    )]
    cut_nc: bool,

    /// Use model's TC trusted cutoffs to set all thresholding
    #[arg(
        long = "cut_tc",
        conflicts_with_all = ["cut_ga", "cut_nc", "e_value", "score_threshold", "inc_e", "inc_t"]
    )]
    cut_tc: bool,

    /// Turn all heuristic filters off (less speed, more power)
    #[arg(long = "max", conflicts_with_all = ["f1", "f2", "f3", "nobias"])]
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

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    nonull2: bool,

    /// Set number of comparisons done for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Random number seed
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Only search the top strand
    #[arg(long = "watson", conflicts_with = "crick")]
    watson: bool,

    /// Only search the bottom strand
    #[arg(long = "crick", conflicts_with = "watson")]
    crick: bool,

    /// Window length (max expected hit length)
    #[arg(long = "w_length")]
    w_length: Option<usize>,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save Dfam-style table of hits and domains
    #[arg(long = "dfamtblout")]
    dfamtblout: Option<PathBuf>,

    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    acc: bool,

    /// Do not output alignments
    #[arg(long = "noali")]
    noali: bool,

    /// Unlimit ASCII text output line width
    #[arg(long = "notextw", conflicts_with = "textw")]
    notextw: bool,

    /// Set max width of ASCII text output lines
    #[arg(long = "textw", value_parser = parse_textw)]
    textw: Option<usize>,
}

fn parse_textw(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid text width: {e}"))?;
    if value < 120 {
        Err("--textw must be >= 120".to_string())
    } else {
        Ok(value)
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

/// Entry point for `nhmmscan`: scan each nucleotide query against every HMM in
/// a DNA HMM database and report a ranked, E-value-thresholded hit table.
///
/// Per-query, profiles are built once-per-HMM in parallel via rayon and scored
/// through the nhmmer long-target path on the requested strand(s). Hits are
/// adjusted into scan E-value space, deduplicated by model/alignment position,
/// thresholded, and printed with HMM names in the target column. Corresponds
/// to `serial_master()`/`serial_loop()` in `hmmer/src/nhmmscan.c`.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let args = Args::parse_from(&args);
    if matches!(args.w_length, Some(0..=3)) {
        eprintln!("Invalid window length value");
        std::process::exit(1);
    }

    logsum::p7_flogsuminit();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.cpu)
        .start_handler(|_| hmmer_pure_rs::util::simd_env::init())
        .build_global()
        .ok();

    // C nhmmscan requires a pressed HMM database and refuses plain ASCII HMMs.
    let pressed_available = pressed::pressed_db_available(&args.hmmdb).unwrap_or_else(|e| {
        eprintln!("Error opening pressed HMM database: {}", e);
        std::process::exit(1);
    });
    if !pressed_available {
        eprintln!(
            "Error opening pressed HMM database: {} is not pressed; use hmmpress first",
            args.hmmdb.display()
        );
        std::process::exit(1);
    }
    let h3m_path = pressed::pressed_h3m_path(&args.hmmdb);
    let hmms = hmmfile_binary::read_binary_hmm_file(&h3m_path).unwrap_or_else(|e| {
        eprintln!("Error reading HMM database: {}", e);
        std::process::exit(1);
    });
    let pressed_oprofiles = pressed::read_pressed_oprofiles(&args.hmmdb).unwrap_or_else(|e| {
        eprintln!("Error reading pressed optimized profiles: {}", e);
        std::process::exit(1);
    });
    if let Err(e) = pressed::validate_pressed_oprofiles_match_hmms(&hmms, &pressed_oprofiles) {
        eprintln!("Error reading pressed optimized profiles: {}", e);
        std::process::exit(1);
    };
    let bit_cutoff = selected_bit_cutoff(&args);
    if let Some(cutoff) = bit_cutoff {
        for hmm in &hmms {
            let mut pli = Pipeline::new();
            pli.use_bit_cutoffs = cutoff;
            if let Err(e) = pli.new_model_thresholds(&hmm.cutoff) {
                eprintln!("Error: {} for model {}", e, hmm.name);
                std::process::exit(1);
            }
        }
    }

    let mut tblout_file = args.tblout.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating tblout file: {}", e);
            std::process::exit(1);
        })
    });
    let mut dfamtblout_file = args.dfamtblout.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating dfamtblout file: {}", e);
            std::process::exit(1);
        })
    });

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

    writeln!(
        out,
        "# nhmmscan :: search nucleotide sequence(s) against a DNA profile database"
    )
    .unwrap();
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
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
        "# target HMM database:             {}",
        args.hmmdb.display()
    )
    .unwrap();
    if let Some(ref path) = args.output {
        writeln!(out, "# output directed to file:         {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.tblout {
        writeln!(out, "# per-seq hits tabular output:     {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.dfamtblout {
        writeln!(out, "# Dfam-style tabular output:       {}", path.display()).unwrap();
    }
    if args.acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
    }
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    if args.notextw {
        writeln!(out, "# max ASCII text line length:      unlimited").unwrap();
    }
    if let Some(textw) = args.textw {
        writeln!(out, "# max ASCII text line length:      {}", textw).unwrap();
    }
    if let Some(z) = args.z_value {
        writeln!(out, "# sequence search space set to:    {:.0}", z).unwrap();
    }
    if args.seed != 42 {
        if args.seed == 0 {
            writeln!(out, "# random number seed:              one-time arbitrary").unwrap();
        } else {
            writeln!(out, "# random number seed set to:       {}", args.seed).unwrap();
        }
    }
    writeln!(out).unwrap();

    // For each query sequence, search all HMMs
    let abc = nucleotide_alphabet_from_hmms(&hmms).unwrap_or_else(|e| {
        eprintln!("Error reading HMM database: {}", e);
        std::process::exit(1);
    });
    let bg = Bg::new(&abc);

    let mut sqf = sequence::open_seq_file(&args.seqfile, &abc).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let mut sq = Sequence::new();
    let mut query_idx = 0usize;
    while sqf.read(&mut sq).unwrap_or_else(|e| {
        eprintln!("Error reading sequence file: {}", e);
        std::process::exit(1);
    }) {
        query_idx += 1;
        writeln!(out, "Query:       {}  [L={}]", sq.name, sq.n).unwrap();

        use rayon::prelude::*;
        let all_hits: Vec<Result<Vec<hmmer_pure_rs::tophits::Hit>, String>> = hmms
            .par_iter()
            .enumerate()
            .map(|(idx, hmm)| {
                let mut local_bg = bg.clone();
                local_bg.set_filter(hmm.m, &hmm.compo);
                local_bg.set_length(sq.n);

                let mut gm = Profile::new(hmm.m, &abc);
                profile::profile_config(hmm, &local_bg, &mut gm, sq.n as i32, P7_LOCAL);
                let om = pressed_oprofiles[idx].clone();
                let mut threshold_pli = Pipeline::new();
                threshold_pli.new_model(&gm);
                configure_thresholds(&mut threshold_pli, &args);
                if let Some(cutoff) = bit_cutoff {
                    threshold_pli.use_bit_cutoffs = cutoff;
                    if threshold_pli.new_model_thresholds(&hmm.cutoff).is_err() {
                        return Ok(Vec::new());
                    }
                }
                let threshold_config =
                    crate::subcmd::nhmmer::NhmmerThresholdConfig::from_pipeline(&threshold_pli);

                let max_length = if let Some(w_length) = args.w_length {
                    w_length as i32
                } else if hmm.max_length > 0 {
                    hmm.max_length
                } else {
                    (hmm.m * 4) as i32
                };
                let do_watson = !args.crick;
                let do_crick = !args.watson;
                let mut hits = Vec::new();
                let msv_counter = std::sync::atomic::AtomicU64::new(0);
                let bias_counter = std::sync::atomic::AtomicU64::new(0);
                let vit_counter = std::sync::atomic::AtomicU64::new(0);
                let fwd_counter = std::sync::atomic::AtomicU64::new(0);

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
                    let mut rc_hits = crate::subcmd::nhmmer::search_sequence(
                        &rc_sq,
                        hmm,
                        &gm,
                        &om,
                        &local_bg,
                        max_length,
                        args.f1,
                        args.f2,
                        args.f3,
                        args.max,
                        args.nobias,
                        args.nonull2,
                        args.seed,
                        threshold_config,
                        true,
                        &msv_counter,
                        &bias_counter,
                        &vit_counter,
                        &fwd_counter,
                    )?;
                    convert_crick_hits_to_forward(&mut rc_hits, sq.n);
                    hits.extend(rc_hits);
                }

                if do_watson {
                    hits.extend(crate::subcmd::nhmmer::search_sequence(
                        &sq,
                        hmm,
                        &gm,
                        &om,
                        &local_bg,
                        max_length,
                        args.f1,
                        args.f2,
                        args.f3,
                        args.max,
                        args.nobias,
                        args.nonull2,
                        args.seed,
                        threshold_config,
                        false,
                        &msv_counter,
                        &bias_counter,
                        &vit_counter,
                        &fwd_counter,
                    )?);
                }

                let strands_searched = (do_watson as usize) + (do_crick as usize);
                let seq_len = sq.n * strands_searched.max(1);
                let nhmmscan_ln = ((seq_len as f32) / (max_length as f32)).ln() as f64;
                for hit in &mut hits {
                    hit.lnp += nhmmscan_ln;
                    hit.sortkey = hit.lnp;
                    if let Some(dom0) = hit.dcl.first_mut() {
                        dom0.lnp = hit.lnp;
                    }
                    hit.name = hmm.name.clone();
                    hit.acc = hmm.acc.clone().unwrap_or_default();
                    hit.desc = hmm.desc.clone().unwrap_or_default();
                    hit.n = hmm.m;
                }
                if let Some(cutoff) = bit_cutoff {
                    let mut local_th = TopHits::new();
                    local_th.hits = hits;
                    let mut tmp_pli = Pipeline::new();
                    tmp_pli.long_target = true;
                    configure_thresholds(&mut tmp_pli, &args);
                    tmp_pli.use_bit_cutoffs = cutoff;
                    if tmp_pli.new_model_thresholds(&hmm.cutoff).is_ok() {
                        local_th.threshold(&tmp_pli, 1.0, 1.0);
                    }
                    hits = local_th.hits;
                }
                Ok(hits)
            })
            .collect();
        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = all_hits
            .into_iter()
            .flat_map(|result| {
                result.unwrap_or_else(|err| {
                    eprintln!("Error: {err}");
                    std::process::exit(1);
                })
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        let z = 1.0_f64;
        apply_nhmmscan_model_evalue_scale(&mut th.hits, hmms.len());
        th.sort_by_modelname_and_alipos();
        th.remove_duplicates();
        th.sort_by_sortkey();
        {
            let mut tmp_pli = Pipeline::new();
            tmp_pli.long_target = true;
            configure_thresholds(&mut tmp_pli, &args);
            if let Some(cutoff) = bit_cutoff {
                tmp_pli.use_bit_cutoffs = cutoff;
            }
            th.threshold(&tmp_pli, z, z);
        }

        if let Some(ref mut f) = tblout_file {
            crate::subcmd::nhmmer::write_nhmmer_tblout(
                f,
                "nhmmscan",
                "SCAN",
                &sq.name,
                Some(&sq.acc),
                &th,
                &args.seqfile,
                &args.hmmdb,
                &cmdline,
                query_idx == 1,
                false,
            );
        }
        if let Some(ref mut f) = dfamtblout_file {
            crate::subcmd::nhmmer::write_nhmmer_dfamtblout(f, &sq.name, None, "modlen", &th);
        }

        writeln!(
            out,
            "Scores for complete sequence (score includes all domains):"
        )
        .unwrap();
        writeln!(
            out,
            "   --- full sequence ---   --- best 1 domain ---    -#dom-"
        )
        .unwrap();
        writeln!(
            out,
            "    E-value  score  bias    E-value  score  bias    exp  N  Model    Description"
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
            let best_dom = hit
                .dcl
                .iter()
                .max_by(|a, b| a.bitscore.total_cmp(&b.bitscore));
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
                display_name(hit, args.acc),
                hit.desc,
            )
            .unwrap();
        }

        if th.nreported == 0 {
            writeln!(
                out,
                "   [No targets detected that satisfy reporting thresholds]"
            )
            .unwrap();
        }
        writeln!(out, "\n//").unwrap();

        sq.reuse();
    }

    if query_idx == 0 {
        eprintln!("Error: no sequences found in {}", args.seqfile.display());
        std::process::exit(1);
    }

    if let Some(ref mut f) = tblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "nhmmscan",
            "SCAN",
            &args.seqfile,
            &args.hmmdb,
            &cmdline,
        );
    }
    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

fn display_name(hit: &hmmer_pure_rs::tophits::Hit, prefer_acc: bool) -> &str {
    if prefer_acc && !hit.acc.is_empty() {
        &hit.acc
    } else {
        &hit.name
    }
}

fn nucleotide_alphabet_from_hmms(hmms: &[hmmer_pure_rs::Hmm]) -> Result<Alphabet, String> {
    let hmm = hmms
        .first()
        .ok_or_else(|| "pressed HMM database contains no models".to_string())?;
    match hmm.abc_type {
        hmmer_pure_rs::alphabet::AlphabetType::Dna => Ok(Alphabet::dna()),
        hmmer_pure_rs::alphabet::AlphabetType::Rna => Ok(Alphabet::rna()),
        other => Err(format!(
            "nhmmscan requires a DNA or RNA HMM database; found {:?}",
            other
        )),
    }
}

fn selected_bit_cutoff(args: &Args) -> Option<BitCutoff> {
    if args.cut_ga {
        Some(BitCutoff::GA)
    } else if args.cut_tc {
        Some(BitCutoff::TC)
    } else if args.cut_nc {
        Some(BitCutoff::NC)
    } else {
        None
    }
}

fn apply_nhmmscan_model_evalue_scale(hits: &mut [hmmer_pure_rs::tophits::Hit], nmodels: usize) {
    let scale = (nmodels as f32).ln() as f64;
    for hit in hits {
        hit.lnp += scale;
        hit.sortkey = hit.lnp;
        if let Some(dom0) = hit.dcl.first_mut() {
            dom0.lnp = hit.lnp;
        }
    }
}

fn configure_thresholds(pli: &mut Pipeline, args: &Args) {
    pli.e_value_threshold = args.e_value;
    pli.inc_e = args.inc_e;
    if let Some(t) = args.score_threshold {
        pli.t = Some(t);
        pli.by_e = false;
    }
    if let Some(t) = args.inc_t {
        pli.inc_t = Some(t);
        pli.inc_by_e = false;
    }
}

fn convert_crick_hits_to_forward(hits: &mut [hmmer_pure_rs::tophits::Hit], n: usize) {
    for hit in hits {
        for dom in &mut hit.dcl {
            let ali_hi = n as i64 - dom.iali + 1;
            let ali_lo = n as i64 - dom.jali + 1;
            dom.iali = ali_hi;
            dom.jali = ali_lo;
            let env_hi = n as i64 - dom.ienv + 1;
            let env_lo = n as i64 - dom.jenv + 1;
            dom.ienv = env_hi;
            dom.jenv = env_lo;
            if let Some(ref mut ad) = dom.ad {
                let ad_hi = n.saturating_sub(ad.sqfrom) + 1;
                let ad_lo = n.saturating_sub(ad.sqto) + 1;
                ad.sqfrom = ad_hi;
                ad.sqto = ad_lo;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmmer_pure_rs::alphabet::AlphabetType;
    use hmmer_pure_rs::Hmm;

    #[test]
    fn nhmmscan_positionals_match_c_order() {
        let args = Args::try_parse_from(["nhmmscan", "models.hmm", "queries.fa"]).unwrap();

        assert_eq!(args.hmmdb, PathBuf::from("models.hmm"));
        assert_eq!(args.seqfile, PathBuf::from("queries.fa"));
    }

    #[test]
    fn nhmmscan_parses_c_output_options() {
        let args = Args::try_parse_from([
            "nhmmscan",
            "-o",
            "out.txt",
            "--tblout",
            "hits.tbl",
            "--dfamtblout",
            "dfam.tbl",
            "--acc",
            "--noali",
            "--notextw",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();

        assert_eq!(args.output, Some(PathBuf::from("out.txt")));
        assert_eq!(args.tblout, Some(PathBuf::from("hits.tbl")));
        assert_eq!(args.dfamtblout, Some(PathBuf::from("dfam.tbl")));
        assert!(args.acc);
        assert!(args.noali);
        assert!(args.notextw);
    }

    #[test]
    fn nhmmscan_parses_c_threshold_options() {
        let args = Args::try_parse_from([
            "nhmmscan",
            "-T",
            "42",
            "--incT",
            "40",
            "-Z",
            "45000000",
            "--seed",
            "7",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();
        assert_eq!(args.score_threshold, Some(42.0));
        assert_eq!(args.inc_t, Some(40.0));
        assert_eq!(args.z_value, Some(45_000_000.0));
        assert_eq!(args.seed, 7);

        let args =
            Args::try_parse_from(["nhmmscan", "--cut_ga", "models.hmm", "queries.fa"]).unwrap();
        assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::GA));
    }

    #[test]
    fn nhmmscan_evalues_apply_post_merge_model_count_scale() {
        let base_lnp = 0.25_f64.ln();
        let mut hits = vec![hmmer_pure_rs::tophits::Hit {
            name: "model".to_string(),
            acc: String::new(),
            desc: String::new(),
            n: 100,
            sortkey: base_lnp,
            score: 10.0,
            bias: 0.0,
            pre_score: 10.0,
            sum_score: 10.0,
            lnp: base_lnp,
            pre_lnp: base_lnp,
            sum_lnp: base_lnp,
            nexpected: 1.0,
            nregions: 1,
            nclustered: 1,
            noverlaps: 0,
            nenvelopes: 1,
            ndom: 1,
            nreported: 0,
            nincluded: 0,
            dcl: vec![hmmer_pure_rs::tophits::Domain {
                iali: 1,
                jali: 10,
                ienv: 1,
                jenv: 10,
                bitscore: 10.0,
                lnp: base_lnp,
                dombias: 0.0,
                oasc: 0.0,
                envsc: 0.0,
                domcorrection: 0.0,
                is_reported: false,
                is_included: false,
                ad: None,
            }],
            flags: 0,
            seqidx: 0,
            subseq_start: 0,
        }];

        apply_nhmmscan_model_evalue_scale(&mut hits, 4);

        let expected = base_lnp + (4.0_f32).ln() as f64;
        assert!((hits[0].lnp - expected).abs() < 1e-12);
        assert!((hits[0].dcl[0].lnp - expected).abs() < 1e-12);
        assert!((hits[0].sortkey - expected).abs() < 1e-12);
    }

    #[test]
    fn nhmmscan_uses_pressed_database_rna_alphabet() {
        let hmms = vec![Hmm::new(4, AlphabetType::Rna, 4)];
        let abc = nucleotide_alphabet_from_hmms(&hmms).unwrap();
        assert_eq!(abc.abc_type, AlphabetType::Rna);
    }

    #[test]
    fn nhmmscan_rejects_protein_pressed_database_alphabet() {
        let hmms = vec![Hmm::new(4, AlphabetType::Amino, 20)];
        let err = nucleotide_alphabet_from_hmms(&hmms).unwrap_err();
        assert!(err.contains("requires a DNA or RNA HMM database"));
    }
}
