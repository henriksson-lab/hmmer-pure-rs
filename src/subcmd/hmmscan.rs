//! hmmscan — search sequence(s) against an HMM database.
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
    name = "hmmscan",
    about = "Search sequence(s) against a profile HMM database"
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

    /// Use model's GA gathering cutoffs to set all thresholding
    #[arg(
        long = "cut_ga",
        conflicts_with_all = [
            "cut_tc",
            "cut_nc",
            "e_value",
            "score_threshold",
            "dom_e",
            "dom_t",
            "inc_e",
            "inc_t",
            "inc_dome",
            "inc_dom_t"
        ]
    )]
    cut_ga: bool,

    /// Use model's NC noise cutoffs to set all thresholding
    #[arg(
        long = "cut_nc",
        conflicts_with_all = [
            "cut_ga",
            "cut_tc",
            "e_value",
            "score_threshold",
            "dom_e",
            "dom_t",
            "inc_e",
            "inc_t",
            "inc_dome",
            "inc_dom_t"
        ]
    )]
    cut_nc: bool,

    /// Use model's TC trusted cutoffs to set all thresholding
    #[arg(
        long = "cut_tc",
        conflicts_with_all = [
            "cut_ga",
            "cut_nc",
            "e_value",
            "score_threshold",
            "dom_e",
            "dom_t",
            "inc_e",
            "inc_t",
            "inc_dome",
            "inc_dom_t"
        ]
    )]
    cut_tc: bool,

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

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    nobias: bool,

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    nonull2: bool,

    /// Random number seed
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    domtblout: Option<PathBuf>,

    /// Save Pfam-style table of hits and domains
    #[arg(long = "pfamtblout")]
    pfamtblout: Option<PathBuf>,

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

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    domz_value: Option<f64>,
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

/// Entry point for `hmmscan`: scan each query sequence against every HMM in a
/// database and report a ranked, E-value-thresholded hit table.
///
/// Per-query, profiles are built once-per-HMM in parallel via rayon, the
/// pipeline (MSV -> Viterbi -> Forward + domain decoding) is run, the best
/// hit per HMM is collected, and the top-hits list is sorted, thresholded, and
/// printed in the canonical HMMER per-sequence tabular layout. Corresponds to
/// `serial_master()` in hmmer/src/hmmscan.c (MPI and pressed-db paths omitted).
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let args = Args::parse_from(&args);

    logsum::p7_flogsuminit();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.cpu)
        .start_handler(|_| hmmer_pure_rs::util::simd_env::init())
        .build_global()
        .ok();

    // C hmmscan requires a pressed HMM database and refuses plain ASCII HMMs.
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
        "# hmmscan :: search sequence(s) against a profile database"
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
    if let Some(ref path) = args.domtblout {
        writeln!(out, "# per-dom hits tabular output:     {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.pfamtblout {
        writeln!(out, "# pfam-style tabular output:       {}", path.display()).unwrap();
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
    writeln!(out).unwrap();

    // For each query sequence, search all HMMs
    let abc = alphabet_from_hmms(&hmms).unwrap_or_else(|e| {
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

        // Pre-build profiles for all HMMs (could be cached)
        use rayon::prelude::*;
        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = hmms
            .par_iter()
            .enumerate()
            .filter_map(|(idx, hmm)| {
                let mut local_bg = bg.clone();
                local_bg.set_filter(hmm.m, &hmm.compo);
                local_bg.set_length(sq.n);

                let mut gm = Profile::new(hmm.m, &abc);
                profile::profile_config(hmm, &local_bg, &mut gm, sq.n as i32, P7_LOCAL);
                let mut om = pressed_oprofiles[idx].clone();

                let mut pli = Pipeline::new();
                pli.new_model(&gm);
                configure_thresholds(&mut pli, &args);
                configure_acceleration(&mut pli, &args);
                if let Some(cutoff) = bit_cutoff {
                    pli.use_bit_cutoffs = cutoff;
                    pli.new_model_thresholds(&hmm.cutoff).ok()?;
                }
                pli.do_alignment =
                    !args.noali || args.domtblout.is_some() || args.pfamtblout.is_some();
                pli.do_alignment_display = !args.noali;

                let mut th = TopHits::new();
                if pli.run(&mut gm, &mut om, &local_bg, hmm, &sq, &mut th) {
                    // Use the HMM name for the hit (in hmmscan, targets are HMMs)
                    th.hits.into_iter().next().map(|mut hit| {
                        // Swap: in hmmscan output, "target" is the HMM name
                        hit.name = hmm.name.clone();
                        hit.acc = hmm.acc.clone().unwrap_or_default();
                        hit.desc = hmm.desc.clone().unwrap_or_default();
                        hit.n = hmm.m;
                        hit
                    })
                } else {
                    None
                }
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        let z = args.z_value.unwrap_or(hmms.len() as f64);
        th.sort_by_sortkey();
        {
            let mut tmp_pli = Pipeline::new();
            configure_thresholds(&mut tmp_pli, &args);
            if let Some(cutoff) = bit_cutoff {
                tmp_pli.use_bit_cutoffs = cutoff;
            }
            th.threshold(&tmp_pli, z, z);
        }
        let domz = args.domz_value.unwrap_or(th.nreported.max(1) as f64);
        if domz != z {
            let mut tmp_pli = Pipeline::new();
            configure_thresholds(&mut tmp_pli, &args);
            if let Some(cutoff) = bit_cutoff {
                tmp_pli.use_bit_cutoffs = cutoff;
            }
            th.threshold(&tmp_pli, z, domz);
        }

        if let Some(ref mut f) = tblout_file {
            crate::subcmd::hmmsearch::write_tblout(
                f,
                &sq.name,
                Some(&sq.acc),
                &th,
                z,
                query_idx == 1,
            );
        }
        if let Some(ref mut f) = domtblout_file {
            crate::subcmd::hmmsearch::write_domtblout(
                f,
                &sq.name,
                Some(&sq.acc),
                sq.n,
                &th,
                z,
                domz,
                query_idx == 1,
            );
        }
        if let Some(ref mut f) = pfamtblout_file {
            crate::subcmd::hmmsearch::write_pfamtblout(
                f,
                &sq.name,
                Some(&sq.acc),
                sq.n,
                &th,
                z,
                domz,
            );
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

    if let Some(ref mut f) = tblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "hmmscan",
            "SCAN",
            &args.seqfile,
            &args.hmmdb,
            &cmdline,
        );
    }
    if let Some(ref mut f) = domtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "hmmscan",
            "SCAN",
            &args.seqfile,
            &args.hmmdb,
            &cmdline,
        );
    }
    if let Some(ref mut f) = pfamtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "hmmscan",
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

fn alphabet_from_hmms(hmms: &[hmmer_pure_rs::Hmm]) -> Result<Alphabet, String> {
    let hmm = hmms
        .first()
        .ok_or_else(|| "pressed HMM database contains no models".to_string())?;
    Ok(Alphabet::new(hmm.abc_type))
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

fn configure_acceleration(pli: &mut Pipeline, args: &Args) {
    pli.do_max = args.max;
    pli.f1 = args.f1;
    pli.f2 = args.f2;
    pli.f3 = args.f3;
    pli.do_biasfilter = !args.nobias;
    pli.do_null2 = !args.nonull2;
    pli.seed = args.seed;
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmmer_pure_rs::alphabet::AlphabetType;
    use hmmer_pure_rs::Hmm;

    #[test]
    fn hmmscan_positionals_match_c_order() {
        let args = Args::try_parse_from(["hmmscan", "models.hmm", "queries.fa"]).unwrap();

        assert_eq!(args.hmmdb, PathBuf::from("models.hmm"));
        assert_eq!(args.seqfile, PathBuf::from("queries.fa"));
    }

    #[test]
    fn hmmscan_parses_c_output_options() {
        let args = Args::try_parse_from([
            "hmmscan",
            "-o",
            "out.txt",
            "--tblout",
            "hits.tbl",
            "--domtblout",
            "domains.tbl",
            "--pfamtblout",
            "pfam.tbl",
            "--acc",
            "--noali",
            "--textw",
            "140",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();

        assert_eq!(args.output, Some(PathBuf::from("out.txt")));
        assert_eq!(args.tblout, Some(PathBuf::from("hits.tbl")));
        assert_eq!(args.domtblout, Some(PathBuf::from("domains.tbl")));
        assert_eq!(args.pfamtblout, Some(PathBuf::from("pfam.tbl")));
        assert!(args.acc);
        assert!(args.noali);
        assert_eq!(args.textw, Some(140));
    }

    #[test]
    fn hmmscan_parses_model_specific_cutoff_options() {
        let args =
            Args::try_parse_from(["hmmscan", "--cut_ga", "models.hmm", "queries.fa"]).unwrap();
        assert!(args.cut_ga);
        assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::GA));

        let args =
            Args::try_parse_from(["hmmscan", "--cut_tc", "models.hmm", "queries.fa"]).unwrap();
        assert!(args.cut_tc);
        assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::TC));

        let args =
            Args::try_parse_from(["hmmscan", "--cut_nc", "models.hmm", "queries.fa"]).unwrap();
        assert!(args.cut_nc);
        assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::NC));
    }

    #[test]
    fn hmmscan_parses_c_threshold_options() {
        let args = Args::try_parse_from([
            "hmmscan",
            "-T",
            "42",
            "--domT",
            "3",
            "--incT",
            "40",
            "--incdomT",
            "2",
            "-Z",
            "99",
            "--domZ",
            "7",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();
        assert_eq!(args.score_threshold, Some(42.0));
        assert_eq!(args.dom_t, Some(3.0));
        assert_eq!(args.inc_t, Some(40.0));
        assert_eq!(args.inc_dom_t, Some(2.0));
        assert_eq!(args.z_value, Some(99.0));
        assert_eq!(args.domz_value, Some(7.0));
    }

    #[test]
    fn hmmscan_parses_c_acceleration_options() {
        let args = Args::try_parse_from([
            "hmmscan",
            "--max",
            "--nonull2",
            "--seed",
            "7",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();
        assert!(args.max);
        assert!(args.nonull2);
        assert_eq!(args.seed, 7);

        let args = Args::try_parse_from([
            "hmmscan",
            "--F1",
            "0.1",
            "--F2",
            "0.2",
            "--F3",
            "0.3",
            "--nobias",
            "models.hmm",
            "queries.fa",
        ])
        .unwrap();
        assert_eq!(args.f1, 0.1);
        assert_eq!(args.f2, 0.2);
        assert_eq!(args.f3, 0.3);
        assert!(args.nobias);
    }

    #[test]
    fn hmmscan_uses_pressed_database_alphabet() {
        let hmms = vec![Hmm::new(4, AlphabetType::Dna, 4)];
        let abc = alphabet_from_hmms(&hmms).unwrap();
        assert_eq!(abc.abc_type, AlphabetType::Dna);
    }
}
