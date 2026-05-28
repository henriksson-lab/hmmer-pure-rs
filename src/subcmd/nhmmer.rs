//! nhmmer — search DNA/RNA HMMs, alignments, or sequences against nucleotide
//! sequence databases.
//!
//! Uses the SSV long-target filter for genome-scale FASTA targets, matching C
//! HMMER's nhmmer. HMM query files, Stockholm/Pfam/AFA/A2M/PSIBLAST/CLUSTAL nucleotide MSAs, and
//! FASTA/UniProt/GenBank single-sequence queries are supported; current
//! makehmmerdb FM-index containers are loaded by reconstructing targets for the
//! long-target path, with container FM records used to seed candidate windows
//! when exact consensus seeds can be mapped back to sequence coordinates.

use std::io::{BufReader, Cursor, Read, Write};
use std::path::{Path, PathBuf};

use clap::Parser;

use hmmer_pure_rs::alphabet::{Alphabet, AlphabetType, DSQ_SENTINEL};
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::builder::{self, DEFAULT_WINDOW_BETA};
use hmmer_pure_rs::calibrate::CalibrationConfig;
use hmmer_pure_rs::fm_index::{FmIndex, FmInterval};
use hmmer_pure_rs::hmm::{self as p7hmm, Hmm};
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::msa;
use hmmer_pure_rs::output::{fmt_evalue, fmt_g, fmt_g3};
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::prior::PriorStrategy;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::seqmodel;
use hmmer_pure_rs::sequence::{self, Sequence};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::TopHits;
use hmmer_pure_rs::util::cmath::{c_exp_f64, c_log_f64};

const MAKEHMMERDB_INDEX_MAGIC: &[u8] = b"HMMERDB_INDEXES\0";
const NHMMER_FM_SEED_MAX_DEPTH: usize = 15;
const NHMMER_FM_SEED_SCORE_THRESHOLD_BITS: f32 = 14.0;
const NHMMER_FM_SEED_SCORE_DENSITY_BITS: f32 = 0.75;
const NHMMER_FM_SEED_DROP_MAX_LEN: usize = 4;
const NHMMER_FM_SEED_DROP_LIMIT_BITS: f32 = 0.3;
const NHMMER_FM_SEED_CONSEC_POS_REQ: usize = 5;
const NHMMER_FM_SCORE_SEED_LIMIT: usize = 256;
const NHMMER_FM_SCORE_TRIE_NORMAL_START_LIMIT: usize = 512;
const NHMMER_FM_SCORE_TRIE_NORMAL_TEXT_LIMIT: usize = 4096;
const NHMMER_FM_CONSENSUS_MATCH_REQ: usize = 11;
const NHMMER_FM_SEED_SSV_LENGTH: usize = 100;
const NHMMER_DEFAULT_BLOCK_LENGTH: usize = 1024 * 256;
const MAKEHMMERDB_INDEX_MIN_RECORD_BYTES: usize = 64 + 4 + 24 + 256 * 8;
const MAKEHMMERDB_C_MIN_SEQUENCE_META_BYTES: usize = 4 + 8 + 4 + 4 + 2 + 2 + 2 + 2 + 4;
const MAKEHMMERDB_C_AMBIGUITY_BYTES: usize = 8;
const MAKEHMMERDB_MAX_IN_MEMORY_BYTES: u64 = 16 * 1024 * 1024 * 1024;
#[cfg(test)]
const NHMMER_FM_WINDOW_LIMIT_REGRESSION_SIZE: usize = 4096;
const NHMMER_FM_WINDOW_SOURCE_LIMIT: usize = 4096;
const NHMMER_FM_MIN_EXTENDED_DIAG_SCORE_BITS: f32 = 0.0;

#[derive(Debug, Clone, Copy)]
struct NhmmerFmSeedConfig {
    max_depth: usize,
    score_threshold_bits: f32,
    score_density_bits: f32,
    drop_max_len: usize,
    drop_lim_bits: f32,
    consec_pos_req: usize,
    consensus_match_req: usize,
    ssv_length: usize,
}

impl Default for NhmmerFmSeedConfig {
    fn default() -> Self {
        Self {
            max_depth: NHMMER_FM_SEED_MAX_DEPTH,
            score_threshold_bits: NHMMER_FM_SEED_SCORE_THRESHOLD_BITS,
            score_density_bits: NHMMER_FM_SEED_SCORE_DENSITY_BITS,
            drop_max_len: NHMMER_FM_SEED_DROP_MAX_LEN,
            drop_lim_bits: NHMMER_FM_SEED_DROP_LIMIT_BITS,
            consec_pos_req: NHMMER_FM_SEED_CONSEC_POS_REQ,
            consensus_match_req: NHMMER_FM_CONSENSUS_MATCH_REQ,
            ssv_length: NHMMER_FM_SEED_SSV_LENGTH,
        }
    }
}

#[derive(Parser)]
#[command(
    name = "nhmmer",
    about = "Search DNA/RNA HMM(s) against a nucleotide sequence database"
)]
struct Args {
    /// HMM file, alignment file, or query sequence
    hmmfile: PathBuf,
    /// Target sequence database (FASTA or makehmmerdb FM-index)
    seqdb: PathBuf,

    // --- Output options ---
    /// Direct output to file, not stdout
    #[arg(short = 'o')]
    outfile: Option<PathBuf>,

    /// Save multiple alignment of all hits to file
    #[arg(short = 'A')]
    ali_outfile: Option<PathBuf>,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save hits to Dfam-style tabular file
    #[arg(long = "dfamtblout")]
    dfamtblout: Option<PathBuf>,

    /// Save scores for each position in each alignment to file
    #[arg(long = "aliscoresout")]
    aliscoresout: Option<PathBuf>,

    /// Write query HMMs to file
    #[arg(long = "hmmout")]
    hmmout: Option<PathBuf>,

    /// Don't output alignments
    #[arg(long = "noali")]
    noali: bool,

    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    show_acc: bool,

    /// Unlimit ASCII text output line width
    #[arg(long = "notextw", conflicts_with = "textw")]
    notextw: bool,

    /// Set max width of ASCII text output lines
    #[arg(long = "textw", default_value = "120", value_parser = parse_textw)]
    textw: usize,

    // --- Reporting thresholds ---
    /// Report sequences <= this E-value threshold
    #[arg(
        short = 'E',
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "score_threshold"
    )]
    e_value: f64,

    /// Report sequences >= this score threshold
    // `allow_hyphen_values` so negative bit-score thresholds parse in the
    // space-separated form C/Easel uses (e.g. `-T -20`); clap otherwise treats
    // `-20` as flags. The `-T=-20` form also still works.
    #[arg(short = 'T', conflicts_with = "e_value", allow_hyphen_values = true)]
    score_threshold: Option<f64>,

    /// Include sequences <= this E-value threshold
    #[arg(
        long = "incE",
        default_value = "0.01",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_t"
    )]
    inc_e: f64,

    /// Include sequences >= this score threshold
    #[arg(long = "incT", conflicts_with = "inc_e", allow_hyphen_values = true)]
    inc_t: Option<f64>,

    // --- Model-specific cutoffs ---
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

    // --- Acceleration heuristics ---
    /// Turn all heuristic filters off (less speed, more power)
    #[arg(long = "max", conflicts_with_all = ["f1", "f2", "f3", "nobias"])]
    max: bool,

    /// Stage 1 (SSV) threshold
    #[arg(long = "F1", default_value = "0.02", allow_hyphen_values = true)]
    f1: f64,

    /// Stage 2 (Vit) threshold
    #[arg(long = "F2", default_value = "3e-3", allow_hyphen_values = true)]
    f2: f64,

    /// Stage 3 (Fwd) threshold
    #[arg(long = "F3", default_value = "3e-5", allow_hyphen_values = true)]
    f3: f64,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    nobias: bool,

    /// Gap open probability for single-sequence query models
    #[arg(long = "popen", default_value = "0.03125", value_parser = parse_gap_open)]
    popen: f32,

    /// Gap extend probability for single-sequence query models
    #[arg(long = "pextend", default_value = "0.75", value_parser = parse_gap_extend)]
    pextend: f32,

    /// Use substitution score matrix for single-sequence MSA-format inputs
    #[arg(long = "singlemx")]
    singlemx: bool,

    /// Substitution score matrix choice for --singlemx-compatible inputs
    #[arg(
        long = "mx",
        default_value = "DNA1",
        conflicts_with = "mxfile",
        hide = true
    )]
    matrix: String,

    /// Read substitution score matrix from file
    #[arg(long = "mxfile", conflicts_with = "matrix")]
    mxfile: Option<PathBuf>,

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
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Retained for C command-line compatibility; not used by nhmmer output
    #[arg(long = "domZ", value_parser = parse_positive_f64, hide = true)]
    domz_value: Option<f64>,

    /// Retained for C command-line compatibility; not used by nhmmer output
    #[arg(long = "domE", default_value = "10.0", value_parser = parse_positive_f64, conflicts_with = "dom_t", hide = true)]
    dom_e: f64,

    /// Retained for C command-line compatibility; not used by nhmmer output
    #[arg(
        long = "domT",
        conflicts_with = "dom_e",
        hide = true,
        allow_hyphen_values = true
    )]
    dom_t: Option<f64>,

    /// Retained for C command-line compatibility; not used by nhmmer output
    #[arg(long = "incdomE", default_value = "0.01", value_parser = parse_positive_f64, conflicts_with = "inc_dom_t", hide = true)]
    inc_dome: f64,

    /// Retained for C command-line compatibility; not used by nhmmer output
    #[arg(
        long = "incdomT",
        conflicts_with = "inc_dome",
        hide = true,
        allow_hyphen_values = true
    )]
    inc_dom_t: Option<f64>,

    /// Set RNG seed (0: one-time arbitrary seed)
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Only search the top strand
    #[arg(long = "watson", conflicts_with = "crick")]
    watson: bool,

    /// Only search the bottom strand
    #[arg(long = "crick", conflicts_with = "watson")]
    crick: bool,

    /// Window length (max expected hit length)
    #[arg(long = "w_length", value_parser = parse_window_length)]
    w_length: Option<i32>,

    /// Length of blocks read from target database in threaded C nhmmer
    #[arg(long = "block_length", value_parser = parse_block_length)]
    block_length: Option<usize>,

    /// Tail mass for deriving window length
    #[arg(long = "w_beta", value_parser = parse_window_beta)]
    w_beta: Option<f64>,

    /// Assert target sequence file format (FASTA or FM-index)
    #[arg(long = "tformat")]
    tformat: Option<String>,

    /// Assert query file format (HMM auto-detected; Stockholm/AFA MSA supported)
    #[arg(long = "qformat")]
    qformat: Option<String>,

    /// Force query to be read as individual sequences, even if MSA-looking
    #[arg(long = "qsingle_seqs")]
    qsingle_seqs: bool,

    /// Window length for biased-composition modifier at SSV stage
    #[arg(long = "B1", default_value = "110", conflicts_with_all = ["max", "nobias"], hide = true)]
    b1: usize,

    /// Window length for biased-composition modifier at Viterbi stage
    #[arg(long = "B2", default_value = "240", conflicts_with_all = ["max", "nobias"], hide = true)]
    b2: usize,

    /// Window length for biased-composition modifier at Forward stage
    #[arg(long = "B3", default_value = "1000", conflicts_with_all = ["max", "nobias"], hide = true)]
    b3: usize,

    /// Override default background probabilities; accepted for compatibility
    #[arg(long = "bgfile", hide = true)]
    bgfile: Option<PathBuf>,

    /// FM seed length at which bit threshold must be met
    #[arg(long = "seed_max_depth", default_value = "15")]
    seed_max_depth: usize,

    /// Required FM seed score in bits
    #[arg(long = "seed_sc_thresh", default_value = "14")]
    seed_sc_thresh: f32,

    /// Required FM seed score density in bits per position
    #[arg(long = "seed_sc_density", default_value = "0.75")]
    seed_sc_density: f32,

    /// Maximum run length with score under max minus drop limit
    #[arg(long = "seed_drop_max_len", default_value = "4")]
    seed_drop_max_len: usize,

    /// Maximum drop in a low-growth FM seed run
    #[arg(long = "seed_drop_lim", default_value = "0.3")]
    seed_drop_lim: f32,

    /// Minimum consecutive positive scores in FM seed
    #[arg(long = "seed_req_pos", default_value = "5")]
    seed_req_pos: usize,

    /// Consecutive consensus matches that override FM score threshold
    #[arg(long = "seed_consens_match", default_value = "11")]
    seed_consens_match: usize,

    /// Window length around FM seed for full SSV diagonal
    #[arg(long = "seed_ssv_length", default_value = "100")]
    seed_ssv_length: usize,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Start restricted target search at this sequence key
    #[arg(long = "restrictdb_stkey", hide = true)]
    restrictdb_stkey: Option<String>,

    /// Search only this many target sequences from the restricted start
    #[arg(long = "restrictdb_n", value_parser = parse_positive_usize, hide = true)]
    restrictdb_n: Option<usize>,

    /// SSI index file for C-compatible restricted database options
    #[arg(long = "ssifile", hide = true)]
    ssifile: Option<PathBuf>,
}

impl NhmmerFmSeedConfig {
    fn from_args(args: &Args) -> Self {
        Self {
            max_depth: args.seed_max_depth,
            score_threshold_bits: args.seed_sc_thresh,
            score_density_bits: args.seed_sc_density,
            drop_max_len: args.seed_drop_max_len,
            drop_lim_bits: args.seed_drop_lim,
            consec_pos_req: args.seed_req_pos,
            consensus_match_req: args.seed_consens_match,
            ssv_length: args.seed_ssv_length,
        }
    }
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

fn parse_positive_usize(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid positive integer: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
    }
}

fn parse_window_length(s: &str) -> Result<i32, String> {
    let value = s
        .parse::<i32>()
        .map_err(|e| format!("invalid window length: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("--w_length must be > 0 and fit in a 32-bit signed integer".to_string())
    }
}

fn parse_window_beta(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid window-length beta value: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("Invalid window-length beta value".to_string())
    }
}

fn parse_block_length(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid block length: {e}"))?;
    if value >= 50_000 {
        Ok(value)
    } else {
        Err("value must be >= 50000".to_string())
    }
}

fn parse_gap_open(s: &str) -> Result<f32, String> {
    let value = s
        .parse::<f32>()
        .map_err(|e| format!("invalid gap open probability: {e}"))?;
    if (0.0..0.5).contains(&value) {
        Ok(value)
    } else {
        Err("--popen must be >= 0 and < 0.5".to_string())
    }
}

fn parse_gap_extend(s: &str) -> Result<f32, String> {
    let value = s
        .parse::<f32>()
        .map_err(|e| format!("invalid gap extend probability: {e}"))?;
    if (0.0..1.0).contains(&value) {
        Ok(value)
    } else {
        Err("--pextend must be >= 0 and < 1".to_string())
    }
}

/// Write C-nhmmer-style per-hit tabular output (`--tblout`).
///
/// Mirrors the `long_targets=TRUE` branch of `p7_tophits_TabularTargets` in
/// `hmmer/src/p7_tophits.c:1610`. Emits the header trio, one row per reported
/// hit (target/query name+acc, hmm/ali/env coords, seq length, strand,
/// E-value, score, bias, description), then a footer block with program /
/// version / query file / target file / option settings / cwd / date.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_nhmmer_tblout<W: std::io::Write>(
    f: &mut W,
    program: &str,
    pipeline_mode: &str,
    qname: &str,
    qacc: Option<&str>,
    th: &hmmer_pure_rs::tophits::TopHits,
    hmmfile: &std::path::Path,
    seqdb: &std::path::Path,
    cmdline: &str,
    show_header: bool,
    write_footer: bool,
    evalue_scale: f64,
    length_header: &str,
) {
    use hmmer_pure_rs::tophits::P7_IS_REPORTED;
    // Column widths sized dynamically to mirror C p7_tophits.c:1612-1616:
    //   tnamew = ESL_MAX(20, GetMaxNameLength)
    //   qnamew = ESL_MAX(20, strlen(qname))
    //   qaccw  = (qacc != NULL) ? ESL_MAX(10, strlen(qacc)) : 10
    //   taccw  = ESL_MAX(10, GetMaxAccessionLength)
    //   posw   = ESL_MAX(7, GetMaxPositionLength)   (long_targets always true here)
    // All GetMax* helpers iterate over *all* hits (h->unsrt), not just reported.
    let namew = th
        .hits
        .iter()
        .map(|h| h.name.len())
        .max()
        .unwrap_or(0)
        .max(20);
    let qw = qname.len().max(20);
    let qaccw = qacc.map(|s| s.len()).unwrap_or(0).max(10);
    let taccw = th
        .hits
        .iter()
        .map(|h| h.acc.len())
        .max()
        .unwrap_or(0)
        .max(10);
    // p7_tophits_GetMaxPositionLength: max decimal-digit length of iali/jali
    // over hits with dcl[0].iali > 0, min 7.
    let posw = th
        .hits
        .iter()
        .filter_map(|h| h.dcl.first())
        .filter(|d| d.iali > 0)
        .map(|d| {
            let a = d.iali.to_string().len();
            let b = d.jali.to_string().len();
            a.max(b)
        })
        .max()
        .unwrap_or(0)
        .max(7);

    // Mirror C p7_tophits.c:1623: `#%-*s ...` where width=namew-1, value has
    // leading space (` target name`). Net effect: `#` + 19-char field = 20.
    if show_header {
        // C p7_tophits.c:1623 long_targets header:
        //   "#%-*s %-*s %-*s %-*s %s %s %*s %*s %*s %*s %*s %6s %9s %6s %5s  %s\n"
        // widths: tnamew-1, taccw, qnamew, qaccw, then "hmmfrom"/"hmm to" plain,
        // then posw x5, then %6s strand, %9s E-value, %6s score, %5s bias.
        writeln!(
            f,
            "#{tname:<tnw$} {tacc:<taccw$} {qn:<qnw$} {qacc:<qaccw$} hmmfrom hmm to {af:>posw$} {at:>posw$} {ef:>posw$} {et:>posw$} {sl:>posw$} {strand:>6} {ev:>9} {sc:>6} {bi:>5}  description of target",
            tname = " target name",
            tacc = "accession",
            qn = "query name",
            qacc = "accession",
            af = "alifrom",
            at = " ali to",
            ef = "envfrom",
            et = " env to",
            sl = length_header,
            strand = "strand",
            ev = "  E-value",
            sc = " score",
            bi = " bias",
            tnw = namew - 1,
            qnw = qw,
            taccw = taccw,
            qaccw = qaccw,
            posw = posw,
        )
        .unwrap();
        // C p7_tophits.c:1626 long_targets dash line:
        //   "#%*s %*s %*s %*s %s %s %*s %*s %*s %*s %*s %6s %9s %6s %5s %s\n"
        // Right-justified (%*s) FIXED-LENGTH dash literals padded with spaces to
        // the column width; NOT fill-character dashes. The literals are:
        //   tnamew-1 -> 19 dashes, taccw -> 10, qnamew -> 20, qaccw -> 10,
        //   hmmfrom/hmm to (%s) -> 7 each (plain), posw x5 -> 7 each,
        //   %6s -> 6, %9s -> 9, %6s -> 6, %5s -> 5, %s -> 21.
        writeln!(
            f,
            "#{tname:>tnw$} {tacc:>taccw$} {qn:>qnw$} {qacc:>qaccw$} ------- ------- {af:>posw$} {at:>posw$} {ef:>posw$} {et:>posw$} {sl:>posw$} {strand:>6} {ev:>9} {sc:>6} {bi:>5} ---------------------",
            tname = "-------------------",
            tacc = "----------",
            qn = "--------------------",
            qacc = "----------",
            af = "-------",
            at = "-------",
            ef = "-------",
            et = "-------",
            sl = "-------",
            strand = "------",
            ev = "---------",
            sc = "------",
            bi = "-----",
            tnw = namew - 1,
            qnw = qw,
            taccw = taccw,
            qaccw = qaccw,
            posw = posw,
        )
        .unwrap();
    }

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        let best = match hit.dcl.first() {
            Some(d) => d,
            None => continue,
        };
        let (hmmfrom, hmm_to, alifrom, ali_to) = if let Some(ref ad) = best.ad {
            (ad.hmmfrom, ad.hmmto, ad.sqfrom, ad.sqto)
        } else {
            (0, 0, 0, 0)
        };
        // C passes the 6-char literal "   +  " / "   -  " to %6s.
        let strand_field = if best.iali <= best.jali {
            "   +  "
        } else {
            "   -  "
        };
        let acc_s = hit.acc.as_str();
        let acc_display: &str = if acc_s.is_empty() { "-" } else { acc_s };
        let qacc_display = qacc.filter(|s| !s.is_empty()).unwrap_or("-");
        let desc_display = if hit.desc.is_empty() {
            "-"
        } else {
            hit.desc.as_str()
        };
        let ev_str = fmt_evalue(evalue_scale * c_exp_f64(hit.lnp));
        // C p7_tophits.c:1649 long_targets row:
        //   "%-*s %-*s %-*s %-*s %7d %7d %*PRId64 x5 %6s %9.2g %6.1f %5.1f  %s"
        writeln!(
            f,
            "{tname:<namew$} {tacc:<taccw$} {qn:<qw$} {qacc:<qaccw$} {hf:>7} {ht:>7} {af:>posw$} {at:>posw$} {ef:>posw$} {et:>posw$} {sl:>posw$} {strand} {ev:>9} {sc} {bi}  {desc}",
            tname = hit.name,
            tacc = acc_display,
            qn = qname,
            qacc = qacc_display,
            hf = hmmfrom,
            ht = hmm_to,
            af = alifrom,
            at = ali_to,
            ef = best.ienv,
            et = best.jenv,
            sl = hit.n,
            strand = strand_field,
            ev = ev_str,
            sc = hmmer_pure_rs::output::fmt_score(hit.score),
            bi = hmmer_pure_rs::output::fmt_bias(best.dombias),
            desc = desc_display,
            namew = namew,
            qw = qw,
            taccw = taccw,
            qaccw = qaccw,
            posw = posw,
        )
        .unwrap();
    }
    if !write_footer {
        return;
    }

    // Footer
    writeln!(f, "#").unwrap();
    writeln!(f, "# Program:         {}", program).unwrap();
    writeln!(f, "# Version:         3.4 (Aug 2023)").unwrap();
    writeln!(f, "# Pipeline mode:   {}", pipeline_mode).unwrap();
    writeln!(f, "# Query file:      {}", hmmfile.display()).unwrap();
    writeln!(f, "# Target file:     {}", seqdb.display()).unwrap();
    writeln!(f, "# Option settings: {} ", cmdline).unwrap();
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| String::new());
    writeln!(f, "# Current dir:     {}", cwd).unwrap();
    let date_str = hmmer_pure_rs::output::format_hmmer_date(std::time::SystemTime::now());
    writeln!(f, "# Date:            {}", date_str).unwrap();
    writeln!(f, "# [ok]").unwrap();
}

/// Write C-nhmmscan-style Dfam tabular output (`--dfamtblout`).
///
/// This is the long-target/DNA table used by C `nhmmscan`, not the protein
/// Pfam-style two-section table used by `hmmscan --pfamtblout`.
pub(crate) fn write_nhmmer_dfamtblout<W: std::io::Write>(
    f: &mut W,
    qname: &str,
    qacc: Option<&str>,
    length_header: &str,
    th: &hmmer_pure_rs::tophits::TopHits,
    evalue_scale: f64,
) {
    use hmmer_pure_rs::tophits::P7_IS_REPORTED;

    let tnamew = th
        .hits
        .iter()
        .map(|h| h.name.len())
        .max()
        .unwrap_or(0)
        .max(20);
    let taccw = th
        .hits
        .iter()
        .map(|h| if h.acc.is_empty() { 1 } else { h.acc.len() })
        .max()
        .unwrap_or(0)
        .max(20);
    let qnamew = qname.len().max(20);
    let posw = th
        .hits
        .iter()
        .flat_map(|hit| {
            hit.dcl.iter().flat_map(|dom| {
                [
                    dom.iali.unsigned_abs() as usize,
                    dom.jali.unsigned_abs() as usize,
                    dom.ienv.unsigned_abs() as usize,
                    dom.jenv.unsigned_abs() as usize,
                    hit.n,
                ]
            })
        })
        .map(|pos| pos.to_string().len())
        .max()
        .unwrap_or(0)
        .max(7);
    let tname_hdrw = tnamew - 1;

    writeln!(f, "# hit scores").unwrap();
    writeln!(f, "# ----------").unwrap();
    writeln!(f, "#").unwrap();
    writeln!(
        f,
        "# {:<tname_hdrw$} {:<taccw$} {:<qnamew$} {:>6} {:>9} {:>5}  hmm-st  hmm-en {:>6} {:>posw$} {:>posw$} {:>posw$} {:>posw$} {:>posw$}   description of target",
        "target name",
        "acc",
        "query name",
        "bits",
        "  e-value",
        " bias",
        "strand",
        "ali-st",
        "ali-en",
        "env-st",
        "env-en",
        length_header
    )
    .unwrap();
    writeln!(
        f,
        "# {:<tname_hdrw$} {:<taccw$} {:<qnamew$} {:>6} {:>9} {:>5} ------- ------- {:>6} {:>posw$} {:>posw$} {:>posw$} {:>posw$} {:>posw$}   ---------------------",
        "-------------------",
        "-------------------",
        "-------------------",
        "------",
        "---------",
        "-----",
        "------",
        "-------",
        "-------",
        "-------",
        "-------",
        "-------"
    )
    .unwrap();

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        let Some(dom) = hit.dcl.first() else {
            continue;
        };
        let (hmmfrom, hmmto, alifrom, alito) = if let Some(ref ad) = dom.ad {
            (ad.hmmfrom, ad.hmmto, ad.sqfrom, ad.sqto)
        } else {
            (0, 0, dom.iali as usize, dom.jali as usize)
        };
        let strand = if dom.iali <= dom.jali {
            "   +  "
        } else {
            "   -  "
        };
        let acc = qacc
            .filter(|acc| !acc.is_empty())
            .or_else(|| (!hit.acc.is_empty()).then_some(hit.acc.as_str()))
            .unwrap_or("-");
        let desc = if hit.desc.is_empty() {
            "-"
        } else {
            hit.desc.as_str()
        };
        writeln!(
            f,
            "{:<tnamew$}  {:<taccw$} {:<qnamew$} {} {:>9} {} {:>7} {:>7} {} {:>posw$} {:>posw$} {:>posw$} {:>posw$} {:>posw$}   {}",
            hit.name,
            acc,
            qname,
            hmmer_pure_rs::output::fmt_score(hit.score),
            fmt_evalue(evalue_scale * c_exp_f64(hit.lnp)),
            hmmer_pure_rs::output::fmt_bias(dom.dombias),
            hmmfrom,
            hmmto,
            strand,
            alifrom,
            alito,
            dom.ienv,
            dom.jenv,
            hit.n,
            desc,
        )
        .unwrap();
    }
}

fn search_dfam_query_accession(qacc: Option<&str>) -> &str {
    qacc.filter(|acc| !acc.is_empty()).unwrap_or("-")
}

/// Write nhmmer `--aliscoresout` rows.
///
/// C stores these values in `Domain::scores_per_pos`; keep nhmmer's side-channel
/// local by recomputing the same displayed-column contribution from the
/// alignment display and configured profile when the artifact is written.
fn write_nhmmer_aliscoresout<W: std::io::Write>(
    f: &mut W,
    qname: &str,
    th: &hmmer_pure_rs::tophits::TopHits,
    gm: &Profile,
    abc: &Alphabet,
) {
    use hmmer_pure_rs::tophits::P7_IS_REPORTED;

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        let Some(dom) = hit.dcl.first() else {
            continue;
        };
        let Some(ad) = dom.ad.as_ref() else {
            continue;
        };
        write!(f, "{} {} {} {} :", qname, hit.name, dom.iali, dom.jali).unwrap();
        for score in nhmmer_alignment_scores_per_pos(ad, gm, abc) {
            match score {
                Some(score) => write!(f, " {:.3}", score).unwrap(),
                None => write!(f, " >").unwrap(),
            }
        }
        writeln!(f).unwrap();
    }
}

fn nhmmer_alignment_scores_per_pos(
    ad: &hmmer_pure_rs::tophits::AliDisplay,
    gm: &Profile,
    abc: &Alphabet,
) -> Vec<Option<f32>> {
    use hmmer_pure_rs::profile::{P7P_DD, P7P_DM, P7P_II, P7P_IM, P7P_MD, P7P_MI, P7P_MM};

    let model = ad.model.as_bytes();
    let aseq = ad.aseq.as_bytes();
    let n = model.len().min(aseq.len());
    let mut scores = vec![Some(0.0_f32); n];
    let mut j = ad.hmmfrom.saturating_sub(1);
    let mut k = 0usize;

    while k < n {
        if model[k] != b'.' && aseq[k] != b'-' {
            j += 1;
            let x = residue_code_for_alignment_score(abc, aseq[k]) as usize;
            let emission = gm.msc(j, x);
            let mm = if j == 1 { 0.0 } else { gm.tsc(j - 1, P7P_MM) };
            scores[k] = finite_aliscore_bits(emission + mm);
            k += 1;
        } else if model[k] == b'.' {
            scores[k] = None;
            let mut sc = gm.tsc(j, P7P_MI);
            k += 1;
            while k < n && model[k] == b'.' {
                scores[k] = None;
                sc += gm.tsc(j, P7P_II);
                k += 1;
            }
            sc += gm.tsc(j, P7P_IM) - gm.tsc(j, P7P_MM);
            scores[k - 1] = finite_aliscore_bits(sc);
        } else if aseq[k] == b'-' {
            scores[k] = None;
            let mut sc = gm.tsc(j, P7P_MD);
            j += 1;
            k += 1;
            while k < n && aseq[k] == b'-' {
                scores[k] = None;
                sc += gm.tsc(j, P7P_DD);
                j += 1;
                k += 1;
            }
            sc += gm.tsc(j, P7P_DM) - gm.tsc(j, P7P_MM);
            scores[k - 1] = finite_aliscore_bits(sc);
        } else {
            scores[k] = None;
            k += 1;
        }
    }

    scores
}

fn finite_aliscore_bits(score_nats: f32) -> Option<f32> {
    use hmmer_pure_rs::util::cmath::ESL_CONST_LOG2R;

    if score_nats.is_finite() {
        Some(score_nats * ESL_CONST_LOG2R as f32)
    } else {
        None
    }
}

fn residue_code_for_alignment_score(abc: &Alphabet, residue: u8) -> u8 {
    let code = abc.digitize_symbol(residue);
    if (code as usize) < abc.kp {
        code
    } else {
        abc.unknown_code()
    }
}

fn write_stockholm_msa<W: std::io::Write>(out: &mut W, msa: &hmmer_pure_rs::msa::Msa) {
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
    if let Some(acc) = &msa.acc {
        if !acc.is_empty() {
            writeln!(out, "#=GF AC {}", acc).unwrap();
        }
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
        writeln!(
            out,
            "{:<width$} {}",
            name,
            String::from_utf8_lossy(row),
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

/// Entry point for `nhmmer`: search nucleotide HMM(s) against a DNA/RNA
/// sequence database using the long-target SSV pipeline.
///
/// Equivalent to the C `main` + `serial_master` + `output_header` in
/// `hmmer/src/nhmmer.c`. Parses CLI args, reads HMM(s), configures the
/// pipeline (filter thresholds, reporting/inclusion thresholds, model-specific
/// cutoffs, Z override in Megabases), then for each HMM scans every target
/// sequence (rayon-parallel) on watson and/or crick strands via
/// `search_sequence`. Adjusts hit lnP for nhmmer's residue-based E-value
/// space (lnP += ln(N/W)), deduplicates, thresholds, and emits the
/// long_target text report plus optional `--tblout`.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let table_cmdline = normalize_nhmmer_table_cmdline(&cmdline);
    let matrix_was_requested = args
        .iter()
        .any(|arg| arg == "--mx" || arg.starts_with("--mx="));
    let mxfile_was_requested = args
        .iter()
        .any(|arg| arg == "--mxfile" || arg.starts_with("--mxfile="));
    let seed_max_depth_was_requested = args
        .iter()
        .any(|arg| arg == "--seed_max_depth" || arg.starts_with("--seed_max_depth="));
    let seed_sc_thresh_was_requested = args
        .iter()
        .any(|arg| arg == "--seed_sc_thresh" || arg.starts_with("--seed_sc_thresh="));
    let seed_sc_density_was_requested = args
        .iter()
        .any(|arg| arg == "--seed_sc_density" || arg.starts_with("--seed_sc_density="));
    let seed_drop_max_len_was_requested = args
        .iter()
        .any(|arg| arg == "--seed_drop_max_len" || arg.starts_with("--seed_drop_max_len="));
    let seed_drop_lim_was_requested = args
        .iter()
        .any(|arg| arg == "--seed_drop_lim" || arg.starts_with("--seed_drop_lim="));
    let seed_req_pos_was_requested = args
        .iter()
        .any(|arg| arg == "--seed_req_pos" || arg.starts_with("--seed_req_pos="));
    let seed_consens_match_was_requested = args
        .iter()
        .any(|arg| arg == "--seed_consens_match" || arg.starts_with("--seed_consens_match="));
    let seed_ssv_length_was_requested = args
        .iter()
        .any(|arg| arg == "--seed_ssv_length" || arg.starts_with("--seed_ssv_length="));
    let args = hmmer_pure_rs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);

    if let Some(ref format) = args.tformat {
        if format.eq_ignore_ascii_case("fasta") || is_fmindex_target_format(format) {
            // FASTA is already accepted by the sequence reader's auto-detection.
        } else {
            eprintln!(
                "nhmmer --tformat={} is not implemented: supported target format assertions are fasta and fmindex",
            format
        );
            std::process::exit(1);
        }
    }
    if let Some(ref format) = args.qformat {
        if is_stockholm_query_format(format) || is_text_msa_query_format(format) {
            // Supported below by the query loader.
        } else if query_sequence_format(format).is_some() {
            // Supported below by the single-sequence query loader.
        } else {
            eprintln!(
                "nhmmer --qformat={} is not implemented: supported query format assertions are fasta, uniprot, genbank, embl, ddbj, afa, a2m, psiblast, clustal, clustallike, selex, phylip, phylips, stockholm, and pfam",
                format
            );
            std::process::exit(1);
        }
    }
    if args.dna && args.rna {
        eprintln!("Error: options --dna and --rna are mutually exclusive");
        std::process::exit(1);
    }
    if args.qsingle_seqs
        && args
            .qformat
            .as_deref()
            .is_some_and(|f| f.eq_ignore_ascii_case("hmm"))
    {
        eprintln!("--qsingle_seqs flag is incompatible with an hmm-formatted query file");
        std::process::exit(1);
    }
    if matches!(args.w_length, Some(0..=3)) {
        eprintln!("Invalid window length value");
        std::process::exit(1);
    }

    logsum::p7_flogsuminit();

    if args.cpu > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.cpu)
            .start_handler(|_| hmmer_pure_rs::util::simd_env::init())
            .build_global()
            .ok();
    }

    if args.hmmfile.as_path() == Path::new("-") && args.seqdb.as_path() == Path::new("-") {
        eprintln!("Error: Either <query file> or <seqdb> may be '-' but not both");
        std::process::exit(1);
    }

    let hmms = read_query_hmms(&args).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if hmms.len() > 1 && args.seqdb.as_path() == Path::new("-") {
        eprintln!(
            "Error: target sequence file - isn't rewindable; can't search it with multiple queries"
        );
        std::process::exit(1);
    }

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

    writeln!(
        out,
        "# nhmmer :: search a DNA model, alignment, or sequence against a DNA database"
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
        "# query file:                      {}",
        args.hmmfile.display()
    )
    .unwrap();
    writeln!(
        out,
        "# target sequence database:        {}",
        args.seqdb.display()
    )
    .unwrap();
    if args
        .tformat
        .as_deref()
        .is_some_and(|format| format.eq_ignore_ascii_case("fasta"))
    {
        writeln!(out, "# target format asserted:          fasta").unwrap();
    }
    if args
        .tformat
        .as_deref()
        .is_some_and(is_fmindex_target_format)
    {
        writeln!(out, "# target format asserted:          fmindex").unwrap();
    }
    if args.tblout.is_some() {
        writeln!(
            out,
            "# hits tabular output:             {}",
            args.tblout.as_ref().unwrap().display()
        )
        .unwrap();
    }
    if args.dfamtblout.is_some() {
        writeln!(
            out,
            "# hits output in Dfam format:      {}",
            args.dfamtblout.as_ref().unwrap().display()
        )
        .unwrap();
    }
    if let Some(path) = &args.ali_outfile {
        writeln!(out, "# MSA of all hits saved to file:   {}", path.display()).unwrap();
    }
    if let Some(path) = &args.aliscoresout {
        writeln!(out, "# alignment scores output:         {}", path.display()).unwrap();
    }
    if let Some(path) = &args.hmmout {
        writeln!(out, "# hmm output:                      {}", path.display()).unwrap();
    }
    if args.show_acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
    }
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    write_nhmmer_option_header(&mut out, &args, &cmdline);
    if args.notextw {
        writeln!(out, "# max ASCII text line length:      unlimited").unwrap();
    } else if args.textw != 120 {
        writeln!(out, "# max ASCII text line length:      {}", args.textw).unwrap();
    }
    if args.singlemx {
        writeln!(out, "# Use score matrix for 1-seq MSAs:  on").unwrap();
    }
    if matrix_was_requested {
        writeln!(out, "# subst score matrix (built-in):   {}", args.matrix).unwrap();
    }
    if mxfile_was_requested {
        writeln!(
            out,
            "# subst score matrix (file):       {}",
            args.mxfile.as_ref().unwrap().display()
        )
        .unwrap();
    }
    // Mirror C nhmmer.c:341-342 — emit alphabet assertion when flag is used.
    if args.dna {
        writeln!(out, "# input query is asserted as:      DNA").unwrap();
    }
    if args.rna {
        writeln!(out, "# input query is asserted as:      RNA").unwrap();
    }
    if let Some(format) = &args.qformat {
        writeln!(out, "# query format asserted:           {}", format).unwrap();
    }
    if let Some(w_beta) = args.w_beta {
        writeln!(out, "# window length beta value:        {}", w_beta).unwrap();
    }
    if let Some(w_length) = args.w_length {
        writeln!(out, "# window length :                  {}", w_length).unwrap();
    }
    if let Some(block_length) = args.block_length {
        writeln!(out, "# block length :                   {}", block_length).unwrap();
    }
    if args.qsingle_seqs {
        writeln!(out, "# query contains individual seqs:  on").unwrap();
    }
    if args.b1 != 110 {
        writeln!(out, "# biased comp SSV window len:      {}", args.b1).unwrap();
    }
    if args.b2 != 240 {
        writeln!(out, "# biased comp Viterbi window len:  {}", args.b2).unwrap();
    }
    if args.b3 != 1000 {
        writeln!(out, "# biased comp Forward window len:  {}", args.b3).unwrap();
    }
    if let Some(path) = &args.bgfile {
        writeln!(out, "# file with custom bg probs:       {}", path.display()).unwrap();
    }
    if seed_max_depth_was_requested {
        writeln!(
            out,
            "# FM Seed length:                  {}",
            args.seed_max_depth
        )
        .unwrap();
    }
    if seed_sc_thresh_was_requested {
        writeln!(
            out,
            "# FM score threshold (bits):       {}",
            args.seed_sc_thresh
        )
        .unwrap();
    }
    if seed_sc_density_was_requested {
        writeln!(
            out,
            "# FM score density (bits/pos):     {}",
            args.seed_sc_density
        )
        .unwrap();
    }
    if seed_drop_max_len_was_requested {
        writeln!(
            out,
            "# FM max neg-growth length:        {}",
            args.seed_drop_max_len
        )
        .unwrap();
    }
    if seed_drop_lim_was_requested {
        writeln!(
            out,
            "# FM max run drop:                 {}",
            args.seed_drop_lim
        )
        .unwrap();
    }
    if seed_req_pos_was_requested {
        writeln!(
            out,
            "# FM req positive run length:      {}",
            args.seed_req_pos
        )
        .unwrap();
    }
    if seed_consens_match_was_requested {
        writeln!(
            out,
            "# FM consec consensus match req:   {}",
            args.seed_consens_match
        )
        .unwrap();
    }
    if seed_ssv_length_was_requested {
        writeln!(
            out,
            "# FM len used for Vit window:      {}",
            args.seed_ssv_length
        )
        .unwrap();
    }
    writeln!(out, "# number of worker threads:        {}", args.cpu).unwrap();
    if let Some(stkey) = &args.restrictdb_stkey {
        writeln!(out, "# Restrict db to start at seq key: {}", stkey).unwrap();
    }
    if let Some(n) = args.restrictdb_n {
        writeln!(out, "# Restrict db to # target seqs:    {}", n).unwrap();
    }
    if let Some(ssifile) = &args.ssifile {
        writeln!(
            out,
            "# Override ssi file to:            {}",
            ssifile.display()
        )
        .unwrap();
    }
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();

    // Alignment display line width. Mirrors C nhmmer.c:571-572:
    // `textw=0` means unlimited (single-block display), otherwise --textw.
    let linewidth: usize = if args.notextw { 0 } else { args.textw };

    // Strand selection
    let do_watson = !args.crick; // top strand unless --crick only
    let do_crick = !args.watson; // bottom strand unless --watson only

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
    let mut ali_file = args.ali_outfile.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating alignment output file: {}", e);
            std::process::exit(1);
        })
    });
    let mut aliscores_file = args.aliscoresout.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating alignment scores output file: {}", e);
            std::process::exit(1);
        })
    });
    let mut hmmout_file = args.hmmout.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating hmm output file: {}", e);
            std::process::exit(1);
        })
    });
    for (query_idx, hmm) in hmms.iter().enumerate() {
        let abc = match (args.dna, args.rna, hmm.abc_type) {
            (true, _, hmmer_pure_rs::alphabet::AlphabetType::Dna) => Alphabet::dna(),
            (_, true, hmmer_pure_rs::alphabet::AlphabetType::Rna) => Alphabet::rna(),
            (true, _, _) => {
                eprintln!(
                    "Error reading hmm from file {}: expected DNA query HMM",
                    args.hmmfile.display()
                );
                std::process::exit(1);
            }
            (_, true, _) => {
                eprintln!(
                    "Error reading hmm from file {}: expected RNA query HMM",
                    args.hmmfile.display()
                );
                std::process::exit(1);
            }
            (_, _, hmmer_pure_rs::alphabet::AlphabetType::Dna) => Alphabet::dna(),
            (_, _, hmmer_pure_rs::alphabet::AlphabetType::Rna) => Alphabet::rna(),
            _ => {
                eprintln!("Error: Invalid alphabet type in query for nhmmer. Expect DNA or RNA.");
                std::process::exit(1);
            }
        };
        let mut bg = Bg::new(&abc);
        if let Some(path) = &args.bgfile {
            if let Err(e) = bg.read_file(&abc, path) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        // Configure bias filter HMM with model composition — matches C
        // nhmmer.c which calls p7_bg_SetFilter(bg, om->M, om->compo).
        // Without this, bg.fhmm_* are default values and filter_score
        // produces nonsense for long_target bias calculations.
        bg.set_filter(hmm.m, &hmm.compo);

        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        let mut pli = Pipeline::new();
        pli.new_model(&gm);
        // F1 is resolved below, after the target DB is read (it depends on
        // FM-index vs FASTA and on whether --F1 was given). Per C
        // `nhmmer.c:1020-1029`, when --F1 is not on the command line F1 = 0.03
        // (FM-index) / 0.02 (FASTA); and `--max` is mutually exclusive with
        // `--F1`, so `--max` never raises F1 to 0.3 — it only forces F2=F3=1.0.
        let effective_f2 = if args.max { 1.0 } else { args.f2 };
        let effective_f3 = if args.max { 1.0 } else { args.f3 };
        let effective_nobias = args.max || args.nobias;

        pli.f2 = effective_f2;
        pli.f3 = effective_f3;
        pli.e_value_threshold = args.e_value;
        pli.inc_e = args.inc_e;
        pli.do_max = args.max;
        pli.seed = args.seed;
        if effective_nobias {
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
            std::process::exit(1);
        }

        // Database size override (nhmmer uses Megabases)
        if let Some(z) = args.z_value {
            pli.z = z * 1_000_000.0; // convert Mb to bases
            pli.z_setby = hmmer_pure_rs::pipeline::ZSetBy::Option;
        }

        writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.m).unwrap();
        if let Some(acc) = hmm.acc.as_deref() {
            if !acc.is_empty() {
                writeln!(out, "Accession:   {}", acc).unwrap();
            }
        }
        if let Some(desc) = hmm.desc.as_deref() {
            if !desc.is_empty() {
                writeln!(out, "Description: {}", desc).unwrap();
            }
        }
        out.flush().unwrap_or_else(|e| {
            eprintln!("Error writing output: {}", e);
            std::process::exit(1);
        });

        let target_db = read_target_database(&args, &abc).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });
        let sequences = &target_db.sequences;
        if sequences.is_empty() {
            eprintln!("Error: no sequences found in {}", args.seqdb.display());
            std::process::exit(1);
        }

        use rayon::prelude::*;

        let max_length = nhmmer_max_length(hmm, args.w_length, args.w_beta);
        if let Some(ref mut f) = hmmout_file {
            let mut saved_hmm = hmm.clone();
            saved_hmm.max_length = max_length;
            hmmer_pure_rs::hmmfile::write_hmm(f, &saved_hmm).unwrap_or_else(|e| {
                eprintln!("Error writing hmm output: {}", e);
                std::process::exit(1);
            });
        }
        // C `nhmmer.c:1020-1029`: when --F1 isn't on the command line, F1 is the
        // method default (0.03 for FM-index targets, 0.02 for FASTA), regardless
        // of --max.
        let f1 = if command_line_has_option(&cmdline, "--F1") {
            args.f1
        } else if !target_db.fm_index_records.is_empty() {
            0.03
        } else {
            0.02
        };
        pli.f1 = f1;
        let f2 = effective_f2;
        let f3 = effective_f3;
        let do_max = args.max;
        let nobias = effective_nobias;
        let nonull2 = args.nonull2;
        let seed = args.seed;
        let bias_windows = NhmmerBiasWindowLengths {
            b1: args.b1,
            b2: args.b2,
            b3: args.b3,
        };
        let threshold_config = NhmmerThresholdConfig::from_pipeline(&pli);
        let fm_seed_config = NhmmerFmSeedConfig::from_args(&args);
        let effective_block_length = args.block_length.unwrap_or(NHMMER_DEFAULT_BLOCK_LENGTH);

        // Atomic counters for pipeline filter residues, aggregated across
        // threads and per-window sub-pipelines.
        use std::sync::atomic::AtomicU64;
        let msv_counter = AtomicU64::new(0);
        let bias_counter = AtomicU64::new(0);
        let vit_counter = AtomicU64::new(0);
        let fwd_counter = AtomicU64::new(0);

        let all_hits: Vec<Result<Vec<hmmer_pure_rs::tophits::Hit>, String>> = sequences
            .par_iter()
            .enumerate()
            .map(|(seq_idx, sq)| {
                let mut hits = Vec::new();

                // Search top strand (watson)
                if do_watson {
                    let strand_msv_counter = AtomicU64::new(0);
                    let fm_windows = fm_seed_candidate_windows_with_config(
                        &target_db,
                        seq_idx,
                        hmm,
                        &bg,
                        max_length,
                        f1,
                        false,
                        fm_seed_config,
                    );
                    let initial_windows = if target_db.fm_index_records.is_empty() {
                        fm_windows.as_deref()
                    } else {
                        Some(fm_windows.as_deref().unwrap_or(&[]))
                    };
                    hits.extend(search_sequence(
                        sq,
                        hmm,
                        &gm,
                        &om,
                        &bg,
                        max_length,
                        f1,
                        f2,
                        f3,
                        do_max,
                        nobias,
                        nonull2,
                        seed,
                        bias_windows,
                        threshold_config,
                        false,
                        &strand_msv_counter,
                        &bias_counter,
                        &vit_counter,
                        &fwd_counter,
                        initial_windows,
                    )?);
                    let strand_msv_count =
                        strand_msv_counter.load(std::sync::atomic::Ordering::Relaxed);
                    let strand_msv_count = if target_db.fm_index_records.is_empty()
                        && !do_max
                        && sq.n > effective_block_length
                    {
                        c_style_blocked_msv_residue_count(
                            sq,
                            hmm,
                            &om,
                            &bg,
                            max_length,
                            f1,
                            effective_block_length,
                        )
                    } else {
                        strand_msv_count
                    };
                    msv_counter.fetch_add(strand_msv_count, std::sync::atomic::Ordering::Relaxed);
                }

                // Search complement strand (crick)
                if do_crick && abc.complement.is_some() {
                    let strand_msv_counter = AtomicU64::new(0);
                    let mut rc_dsq = sq.dsq.clone();
                    abc.revcomp(&mut rc_dsq, sq.n);
                    let rc_sq = Sequence {
                        name: sq.name.clone(),
                        acc: sq.acc.clone(),
                        desc: sq.desc.clone(),
                        dsq: rc_dsq,
                        n: sq.n,
                        l: sq.l,
                        taxid: -1,
                    };
                    let fm_windows = fm_seed_candidate_windows_with_config(
                        &target_db,
                        seq_idx,
                        hmm,
                        &bg,
                        max_length,
                        f1,
                        true,
                        fm_seed_config,
                    );
                    let initial_windows = if target_db.fm_index_records.is_empty() {
                        fm_windows.as_deref()
                    } else {
                        Some(fm_windows.as_deref().unwrap_or(&[]))
                    };
                    let mut rc_hits = search_sequence(
                        &rc_sq,
                        hmm,
                        &gm,
                        &om,
                        &bg,
                        max_length,
                        f1,
                        f2,
                        f3,
                        do_max,
                        nobias,
                        nonull2,
                        seed,
                        bias_windows,
                        threshold_config,
                        true,
                        &strand_msv_counter,
                        &bias_counter,
                        &vit_counter,
                        &fwd_counter,
                        initial_windows,
                    )?;
                    let strand_msv_count =
                        strand_msv_counter.load(std::sync::atomic::Ordering::Relaxed);
                    let strand_msv_count = if target_db.fm_index_records.is_empty()
                        && !do_max
                        && rc_sq.n > effective_block_length
                    {
                        c_style_blocked_msv_residue_count(
                            &rc_sq,
                            hmm,
                            &om,
                            &bg,
                            max_length,
                            f1,
                            effective_block_length,
                        )
                    } else {
                        strand_msv_count
                    };
                    msv_counter.fetch_add(strand_msv_count, std::sync::atomic::Ordering::Relaxed);
                    // Convert complement-strand coordinates back to forward
                    // strand. nhmmer convention (C p7_pipeline.c:1449-1456):
                    // for minus-strand hits, iali/jali and ienv/jenv are
                    // emitted in reverse order (iali > jali) so the `strand`
                    // column can be derived as iali < jali ? '+' : '-'.
                    //
                    // LOW-2 (audit 02-nhmmer-longtarget): C folds the strand
                    // flip, `seq_start`, and `window_start` into one expression
                    //   iali = seq_start - (window_start + iali) + 2
                    // inside p7_pli_postViterbi_LongTarget (p7_pipeline.c:1445).
                    // The Rust driver instead reverse-complements the target up
                    // front, applies the window->sequence offset inside
                    // `search_longtarget` (`+= win_start - 1`), then does the
                    // whole-sequence `sq.n - x + 1` flip here. These are
                    // algebraically identical to C precisely when
                    // `seq_start == sq.start == 1`, which is ALWAYS true for the
                    // Rust driver: it reads whole sequences (no blocked
                    // `esl_sqio_ReadWindow` streaming — see LOW-3 below), so
                    // `seq_start` never differs from 1. Verified bit-identical
                    // to C on MADE1/3box and on multi-segment FM DBs, both
                    // strands. Adopting C's single expression here would change
                    // nothing observable (seq_start is constant 1) while risking
                    // the currently-green crick coordinate parity, so the
                    // verified-equivalent two-step form is kept deliberately.
                    for hit in &mut rc_hits {
                        for dom in &mut hit.dcl {
                            // Both forward coords; ali higher first, lower second.
                            let ali_hi = sq.n as i64 - dom.iali + 1;
                            let ali_lo = sq.n as i64 - dom.jali + 1;
                            dom.iali = ali_hi;
                            dom.jali = ali_lo;
                            let env_hi = sq.n as i64 - dom.ienv + 1;
                            let env_lo = sq.n as i64 - dom.jenv + 1;
                            dom.ienv = env_hi;
                            dom.jenv = env_lo;
                            if let Some(ref mut ad) = dom.ad {
                                let ad_hi = sq.n.saturating_sub(ad.sqfrom) + 1;
                                let ad_lo = sq.n.saturating_sub(ad.sqto) + 1;
                                ad.sqfrom = ad_hi;
                                ad.sqto = ad_lo;
                            }
                        }
                    }
                    hits.extend(rc_hits);
                }

                for hit in &mut hits {
                    hit.seqidx = seq_idx as i64;
                }
                Ok(hits)
            })
            .collect();
        let all_hits: Vec<hmmer_pure_rs::tophits::Hit> = all_hits
            .into_iter()
            .flat_map(|result: Result<Vec<_>, String>| {
                result.unwrap_or_else(|err| {
                    eprintln!("Error: {err}");
                    std::process::exit(1);
                })
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
        let evalue_residues = args
            .z_value
            .map(|z_mb| {
                let strands = if do_watson && do_crick { 2.0 } else { 1.0 };
                z_mb * 1_000_000.0 * strands
            })
            .unwrap_or(residues_searched);

        // C p7_tophits_ComputeNhmmerEvalues: lnP += log(N/W) per hit, where
        // N = total residues searched (or -Z megabases, strand-adjusted) and W = window length
        // (= om->max_length). After this, evalue = exp(lnP) directly — the
        // database size is folded into lnP.
        let window_length = max_length as f64;
        // Match C p7_tophits_ComputeNhmmerEvalues exactly (hmmer/src/p7_tophits.c:796):
        //   hit.lnP += log((float)N / (float)W);  // note (float) casts before divide
        //   hit.dcl[0].lnP = hit.lnP;             // SET (not +=) dcl[0]
        //   hit.sortkey    = -1.0 * hit.lnP;      // sortkey is negative lnP
        // pre_lnP, sum_lnP, and dcl[1..] are NOT touched in C.
        //
        // Rust's sort_by_sortkey sorts ascending; C's hit_sorter_by_sortkey
        // sorts descending. Since we use sortkey = hit.lnp (not -lnp) so the
        // ascending sort puts most-significant first, we preserve that here
        // but still skip pre_lnp/sum_lnp/dcl[1..] to match C semantics.
        let nw_ratio = (evalue_residues as f32) / (window_length as f32);
        let nhmmer_ln_nw = c_log_f64(nw_ratio as f64);
        for hit in &mut th.hits {
            hit.lnp += nhmmer_ln_nw;
            hit.sortkey = hit.lnp;
            if let Some(dom0) = hit.dcl.first_mut() {
                dom0.lnp = hit.lnp;
            }
        }

        // After nhmmer adjustment, evalue = exp(lnp) directly (no × Z).
        let z = 1.0_f64;
        pli.z = residues_searched; // preserve residue count for stats output

        // Mirror C nhmmer.c flow: SortBySeqidxAndAlipos → RemoveDuplicates
        // → sort by sortkey → threshold. The positional sort lets
        // RemoveDuplicates detect adjacent-duplicate pairs.
        th.sort_by_seqidx_and_alipos();
        th.remove_duplicates();
        th.sort_by_sortkey();
        th.threshold(&pli, z, z);

        // Output — nhmmer-specific long_target format (C p7_tophits_Targets
        // with pli->long_targets = TRUE).
        let namew = th
            .hits
            .iter()
            .filter(|h| h.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED != 0)
            .map(|h| h.name.len())
            .max()
            .unwrap_or(8)
            .max(8);
        let posw = th
            .hits
            .iter()
            .filter(|h| h.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED != 0)
            .flat_map(|h| {
                h.dcl.iter().flat_map(|d| {
                    [d.iali, d.jali, d.ienv, d.jenv]
                        .into_iter()
                        .map(|v| v.abs().to_string().len())
                })
            })
            .max()
            .unwrap_or(6)
            .max(6);

        writeln!(out, "Scores for complete hits:").unwrap();
        writeln!(
            out,
            "  {:>9} {:>6} {:>5}  {:<namew$} {:>posw$} {:>posw$}  Description",
            "E-value",
            " score",
            " bias",
            "Sequence",
            "start",
            "end",
            namew = namew,
            posw = posw,
        )
        .unwrap();
        writeln!(
            out,
            "  {:>9} {:>6} {:>5}  {:<namew$} {:>posw$} {:>posw$}  -----------",
            "-------",
            "------",
            "-----",
            "--------",
            "-----",
            "-----",
            namew = namew,
            posw = posw,
        )
        .unwrap();

        let mut have_printed_incthresh = false;
        for hit in &th.hits {
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
                continue;
            }
            // Print "inclusion threshold" separator when transitioning from
            // included to reported-but-not-included hits (match C
            // p7_tophits.c:1181).
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED == 0 && !have_printed_incthresh {
                writeln!(out, "  ------ inclusion threshold ------").unwrap();
                have_printed_incthresh = true;
            }
            let best_dom = hit.dcl.first();
            let (iali, jali) = best_dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
            let dom_bias_bits = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);
            // Match C's "%c %9.2g %6.1f %5.1f  %-*s %*ld %*ld" — newness,
            // then E-value in %g format.
            writeln!(
                out,
                "  {} {} {}  {:<namew$} {:>posw$} {:>posw$} {}",
                fmt_evalue(c_exp_f64(hit.lnp)),
                hmmer_pure_rs::output::fmt_score(hit.score),
                hmmer_pure_rs::output::fmt_bias(dom_bias_bits),
                if args.show_acc && !hit.acc.is_empty() {
                    &hit.acc
                } else {
                    &hit.name
                },
                iali,
                jali,
                if hit.desc.is_empty() { "" } else { &hit.desc },
                namew = namew,
                posw = posw,
            )
            .unwrap();
        }

        if th.nreported == 0 {
            // C p7_tophits_Targets (p7_tophits.c:1245) writes "No hits".
            // The separate "No targets detected" message is emitted later
            // after the Annotation section header (p7_tophits.c:1471).
            writeln!(
                out,
                "\n   [No hits detected that satisfy reporting thresholds]"
            )
            .unwrap();
        }

        // Per-hit annotation block (C p7_tophits_Domains with long_targets).
        writeln!(out).unwrap();
        writeln!(out).unwrap();
        writeln!(
            out,
            // C p7_tophits.c:1277: "Annotation for each hit %s:\n" where %s is
            // " (and alignments)" or "". The literal space after "hit" stays, so
            // the --noali line is "...hit :" (space before colon) and the
            // alignment line is "...hit  (and alignments):" (colon, no space).
            "Annotation for each hit {}:",
            if args.noali { "" } else { " (and alignments)" }
        )
        .unwrap();
        // C p7_tophits.c:1471: emit "No targets detected" when nothing was
        // reported. Matches C's separate no-targets message in the Domains
        // section (distinct from the "No hits detected" message in the
        // Scores/Targets section above).
        if th.nreported == 0 {
            writeln!(
                out,
                "\n   [No targets detected that satisfy reporting thresholds]"
            )
            .unwrap();
        }
        for hit in &th.hits {
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
                continue;
            }
            writeln!(
                out,
                ">> {}  {}",
                if args.show_acc && !hit.acc.is_empty() {
                    &hit.acc
                } else {
                    &hit.name
                },
                if hit.desc.is_empty() { "" } else { &hit.desc }
            )
            .unwrap();
            if hit.nreported == 0 {
                writeln!(
                    out,
                    "   [No individual domains that satisfy reporting thresholds (although complete target did)]"
                )
                .unwrap();
                writeln!(out).unwrap();
                continue;
            }
            writeln!(
                out,
                "   {:>6} {:>5} {:>9} {:>9} {:>9} {:>2} {:>9} {:>9} {:>2} {:>9} {:>9}    {:>9} {:>2} {:>4}",
                "score",
                "bias",
                "  Evalue",
                "hmmfrom",
                "hmm to",
                "  ",
                " alifrom ",
                " ali to ",
                "  ",
                " envfrom ",
                " env to ",
                "  sq len ",
                "  ",
                "acc"
            )
            .unwrap();
            writeln!(
                out,
                "   {:>6} {:>5} {:>9} {:>9} {:>9} {:>2} {:>9} {:>9} {:>2} {:>9} {:>9}    {:>9} {:>2} {:>4}",
                "------",
                "-----",
                "---------",
                "-------",
                "-------",
                "  ",
                "---------",
                "---------",
                "  ",
                "---------",
                "---------",
                "---------",
                "  ",
                "----"
            )
            .unwrap();

            for dom in &hit.dcl {
                if !dom.is_reported {
                    continue;
                }
                let (hmmfrom, hmmto, sqfrom, sqto) = if let Some(ref ad) = dom.ad {
                    (ad.hmmfrom, ad.hmmto, ad.sqfrom, ad.sqto)
                } else {
                    (0, 0, 0, 0)
                };
                let hmmfrom_bracket = if hmmfrom == 1 { '[' } else { '.' };
                let hmmto_bracket = if hmmto == hmm.m { ']' } else { '.' };
                let sqfrom_bracket = if sqfrom == 1 { '[' } else { '.' };
                let sqto_bracket = if sqto as i64 == hit.n as i64 {
                    ']'
                } else {
                    '.'
                };
                let envfrom_bracket = if dom.ienv == 1 { '[' } else { '.' };
                let envto_bracket = if dom.jenv == hit.n as i64 { ']' } else { '.' };
                let env_span = (dom.jenv - dom.ienv).abs() as f32;
                let acc_val = dom.oasc / (1.0 + env_span);
                writeln!(
                    out,
                    " {} {} {} {} {:>9} {:>9} {}{} {:>9} {:>9} {}{} {:>9} {:>9} {}{} {:>9}    {}",
                    if dom.is_included { '!' } else { '?' },
                    hmmer_pure_rs::output::fmt_score(dom.bitscore),
                    hmmer_pure_rs::output::fmt_bias(dom.dombias),
                    fmt_evalue(c_exp_f64(dom.lnp)),
                    hmmfrom,
                    hmmto,
                    hmmfrom_bracket,
                    hmmto_bracket,
                    sqfrom,
                    sqto,
                    sqfrom_bracket,
                    sqto_bracket,
                    dom.ienv,
                    dom.jenv,
                    envfrom_bracket,
                    envto_bracket,
                    hit.n,
                    hmmer_pure_rs::output::fmt_width4_2(acc_val as f64),
                )
                .unwrap();
            }

            // Alignment block.
            if !args.noali {
                for dom in &hit.dcl {
                    if !dom.is_reported {
                        continue;
                    }
                    if let Some(ref ad) = dom.ad {
                        if !ad.model.is_empty() {
                            writeln!(out).unwrap();
                            writeln!(out, "  Alignment:").unwrap();
                            writeln!(
                                out,
                                "  score: {} bits",
                                hmmer_pure_rs::output::fmt_fixed1(dom.bitscore as f64)
                            )
                            .unwrap();
                            // Build CS line from hmm.cs over positions hmmfrom..=hmmto.
                            let cs_line = hmm.cs.as_deref().map(|cs: &[u8]| {
                                let mut s = String::with_capacity(ad.model.len());
                                let mut cs_idx = ad.hmmfrom;
                                for ch in ad.model.chars() {
                                    if ch == '.' {
                                        s.push('.');
                                    } else {
                                        s.push(if cs_idx < cs.len() {
                                            cs[cs_idx] as char
                                        } else {
                                            ' '
                                        });
                                        cs_idx += 1;
                                    }
                                }
                                s
                            });
                            hmmer_pure_rs::tophits::print_alidisplay_blocks(
                                &mut out,
                                &hmm.name,
                                if args.show_acc && !hit.acc.is_empty() {
                                    &hit.acc
                                } else {
                                    &hit.name
                                },
                                ad,
                                cs_line.as_deref(),
                                linewidth,
                            );
                        }
                    }
                }
            }
            writeln!(out).unwrap();
        }

        // Internal pipeline statistics summary.
        writeln!(out).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "Internal pipeline statistics summary:").unwrap();
        writeln!(out, "-------------------------------------").unwrap();
        writeln!(
            out,
            "Query model(s):                            {}  ({} nodes)",
            1, hmm.m
        )
        .unwrap();
        writeln!(
            out,
            "Target sequences:                          {}  ({} residues searched)",
            sequences.len(),
            residues_searched as u64
        )
        .unwrap();
        let msv_count = msv_counter.load(std::sync::atomic::Ordering::Relaxed);
        let bias_count = bias_counter.load(std::sync::atomic::Ordering::Relaxed);
        let vit_count = vit_counter.load(std::sync::atomic::Ordering::Relaxed);
        let fwd_count = fwd_counter.load(std::sync::atomic::Ordering::Relaxed);
        // pos_output = sum of aligned residues across reported domains
        // (matches C p7_pipeline.c:1985 `pli->pos_output / nres`).
        let mut pos_output: u64 = 0;
        for hit in &th.hits {
            if hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED == 0 {
                continue;
            }
            for dom in &hit.dcl {
                if dom.is_reported {
                    pos_output += ((dom.jali - dom.iali).abs() + 1) as u64;
                }
            }
        }
        let denom = residues_searched.max(1.0);
        let g3 = |v: f64| -> String { fmt_g3(v) };
        writeln!(
            out,
            "Residues passing SSV filter: {:>15}  ({}); expected ({})",
            msv_count,
            g3(msv_count as f64 / denom),
            g3(pli.f1)
        )
        .unwrap();
        writeln!(
            out,
            "Residues passing bias filter:{:>15}  ({}); expected ({})",
            bias_count,
            g3(bias_count as f64 / denom),
            g3(pli.f1)
        )
        .unwrap();
        writeln!(
            out,
            "Residues passing Vit filter: {:>15}  ({}); expected ({})",
            vit_count,
            g3(vit_count as f64 / denom),
            g3(pli.f2)
        )
        .unwrap();
        writeln!(
            out,
            "Residues passing Fwd filter: {:>15}  ({}); expected ({})",
            fwd_count,
            g3(fwd_count as f64 / denom),
            g3(pli.f3)
        )
        .unwrap();
        writeln!(
            out,
            "Total number of hits:        {:>15}  ({})",
            th.nreported,
            g3(pos_output as f64 / denom)
        )
        .unwrap();
        // Run-time footer (`# CPU time:` / `# Mc/sec:`) is intentionally
        // suppressed for determinism, matching the hmmsearch/phmmer/hmmscan/
        // jackhmmer Rust programs (which also omit these non-deterministic
        // timing lines). C HMMER prints them, but the Rust ports do not.

        // Tblout
        if let Some(ref mut f) = tblout_file {
            write_nhmmer_tblout(
                f,
                "nhmmer",
                "SEARCH",
                &hmm.name,
                hmm.acc.as_deref(),
                &th,
                &args.hmmfile,
                &args.seqdb,
                &table_cmdline,
                query_idx == 0,
                query_idx + 1 == hmms.len(),
                1.0,
                " sq len",
            );
        }
        if let Some(ref mut f) = dfamtblout_file {
            let qacc = search_dfam_query_accession(hmm.acc.as_deref());
            write_nhmmer_dfamtblout(f, &hmm.name, Some(qacc), "sq-len", &th, 1.0);
        }
        if let Some(ref mut f) = aliscores_file {
            write_nhmmer_aliscoresout(f, &hmm.name, &th, &gm, &abc);
        }
        if let Some(ref mut f) = ali_file {
            if let Some(mut msa) =
                hmmer_pure_rs::tophits::included_alignment(&th, &abc, hmm.m, None, &hmm.name)
            {
                msa.acc = hmm.acc.clone();
                msa.desc = hmm.desc.clone();
                msa.author = Some("nhmmer (HMMER 3.4)".to_string());
                write_stockholm_msa(f, &msa);
                writeln!(
                    out,
                    "# Alignment of {} hits satisfying inclusion thresholds saved to: {}",
                    msa.nseq,
                    args.ali_outfile.as_ref().unwrap().display()
                )
                .unwrap();
            } else {
                writeln!(
                    out,
                    "# No hits satisfy inclusion thresholds; no alignment saved"
                )
                .unwrap();
            }
        }
        writeln!(out, "//").unwrap();
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

fn write_nhmmer_option_header(out: &mut dyn Write, args: &Args, cmdline: &str) {
    if command_line_has_option(cmdline, "-E") {
        writeln!(
            out,
            "# sequence reporting threshold:    E-value <= {}",
            fmt_g(args.e_value)
        )
        .unwrap();
    }
    if let Some(score) = args.score_threshold {
        writeln!(
            out,
            "# sequence reporting threshold:    score >= {}",
            fmt_g(score)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--incE") {
        writeln!(
            out,
            "# sequence inclusion threshold:    E-value <= {}",
            fmt_g(args.inc_e)
        )
        .unwrap();
    }
    if let Some(score) = args.inc_t {
        writeln!(
            out,
            "# sequence inclusion threshold:    score >= {}",
            fmt_g(score)
        )
        .unwrap();
    }
    if args.cut_ga {
        writeln!(out, "# model-specific thresholding:     GA cutoffs").unwrap();
    }
    if args.cut_nc {
        writeln!(out, "# model-specific thresholding:     NC cutoffs").unwrap();
    }
    if args.cut_tc {
        writeln!(out, "# model-specific thresholding:     TC cutoffs").unwrap();
    }
    if args.max {
        writeln!(
            out,
            "# Max sensitivity mode:            on [all heuristic filters off]"
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--F1") {
        writeln!(out, "# SSV filter P threshold:       <= {}", fmt_g(args.f1)).unwrap();
    }
    if command_line_has_option(cmdline, "--F2") {
        writeln!(out, "# Vit filter P threshold:       <= {}", fmt_g(args.f2)).unwrap();
    }
    if command_line_has_option(cmdline, "--F3") {
        writeln!(out, "# Fwd filter P threshold:       <= {}", fmt_g(args.f3)).unwrap();
    }
    if args.nobias {
        writeln!(out, "# biased composition HMM filter:   off").unwrap();
    }
    if args.nonull2 {
        writeln!(out, "# null2 bias corrections:          off").unwrap();
    }
    if args.watson {
        writeln!(out, "# search only top strand:          on").unwrap();
    }
    if args.crick {
        writeln!(out, "# search only bottom strand:       on").unwrap();
    }
    if let Some(z) = args.z_value {
        writeln!(out, "# database size is set to:         {:.1} Mb", z).unwrap();
    }
    if command_line_has_option(cmdline, "--seed") {
        if args.seed == 0 {
            writeln!(out, "# random number seed:              one-time arbitrary").unwrap();
        }
        writeln!(out, "# random number seed set to:       {}", args.seed).unwrap();
    }
}

fn command_line_has_option(cmdline: &str, option: &str) -> bool {
    let compact_short = option
        .strip_prefix('-')
        .filter(|rest| !rest.starts_with('-') && rest.chars().count() == 1)
        .map(|_| option);
    cmdline.split_whitespace().any(|token| {
        token == option
            || token.starts_with(&format!("{option}="))
            || compact_short
                .is_some_and(|short| token.starts_with(short) && token.len() > short.len())
    })
}

fn normalize_nhmmer_table_cmdline(cmdline: &str) -> String {
    let mut tokens = cmdline.split_whitespace();
    match (tokens.next(), tokens.next()) {
        (Some(_wrapper), Some("nhmmer")) => {
            let rest = tokens.collect::<Vec<_>>();
            if rest.is_empty() {
                "nhmmer".to_string()
            } else {
                format!("nhmmer {}", rest.join(" "))
            }
        }
        _ => cmdline.to_string(),
    }
}

fn read_hmms(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<hmmer_pure_rs::Hmm>> {
    if path == std::path::Path::new("-") {
        hmmfile::read_hmms_auto(BufReader::new(std::io::stdin().lock()))
    } else {
        hmmfile::read_hmm_file_auto(path)
    }
}

fn read_query_hmms(args: &Args) -> Result<Vec<hmmer_pure_rs::Hmm>, String> {
    if args.qsingle_seqs {
        return match read_query_sequence_hmms(
            args,
            args.qformat.as_deref().and_then(query_sequence_format),
            true,
        ) {
            Ok(hmms) => Ok(hmms),
            Err(seq_err) => match read_hmms(&args.hmmfile) {
                Ok(_) => Err(
                    "--qsingle_seqs flag is incompatible with an hmm-formatted query file"
                        .to_string(),
                ),
                Err(_) => Err(seq_err),
            },
        };
    }

    match args.qformat.as_deref() {
        Some(format) if is_stockholm_query_format(format) => read_query_msa_hmms(args),
        Some(format) if is_text_msa_query_format(format) => read_query_msa_hmms(args),
        Some(format) if query_sequence_format(format).is_some() => {
            read_query_sequence_hmms(args, query_sequence_format(format), true)
        }
        Some(format) => Err(format!(
            "nhmmer --qformat={} is not implemented: supported query format assertions are fasta, uniprot, genbank, embl, ddbj, afa, a2m, psiblast, clustal, clustallike, selex, phylip, phylips, stockholm, and pfam",
            format
        )),
        None => match read_hmms(&args.hmmfile) {
            Ok(hmms) => Ok(hmms),
            Err(hmm_err) => {
                if args.hmmfile == std::path::Path::new("-") {
                    Err(
                        "Must specify query file format (--qformat) to read <query file> from stdin ('-')"
                            .to_string(),
                    )
                } else {
                    match read_query_msa_hmms(args) {
                        Ok(hmms) => Ok(hmms),
                        Err(msa_err) => read_query_sequence_hmms(args, None, false).map_err(
                            |seq_err| {
                                format!("Error reading query file: {hmm_err}; {msa_err}; {seq_err}")
                            },
                        ),
                    }
                }
            }
        },
    }
}

fn read_query_sequence_hmms(
    args: &Args,
    format: Option<sequence::SequenceFormat>,
    explicit_format: bool,
) -> Result<Vec<Hmm>, String> {
    let bytes = read_query_file_bytes(&args.hmmfile)
        .map_err(|e| format!("Error reading query sequence file: {e}"))?;
    let abc_type = if args.dna {
        AlphabetType::Dna
    } else if args.rna {
        AlphabetType::Rna
    } else if format == Some(sequence::SequenceFormat::Fasta) || format.is_none() {
        guess_fasta_alphabet(&bytes)?
    } else {
        return Err(
            "Unable to guess alphabet for query sequence file; please specify --dna or --rna"
                .to_string(),
        );
    };
    if abc_type != AlphabetType::Dna && abc_type != AlphabetType::Rna {
        return Err(
            "Error: Invalid alphabet type in query for nhmmer. Expect DNA or RNA.".to_string(),
        );
    }

    let abc = Alphabet::new(abc_type);
    let mut sqf = sequence::SeqFile::new(std::io::Cursor::new(bytes), abc.clone());
    if let Some(format) = format {
        sqf = sqf.with_format(format);
    } else {
        sqf = sqf.with_fasta_only();
    }
    let format_name = format.map_or("FASTA", |format| match format {
        sequence::SequenceFormat::Fasta => "FASTA",
        sequence::SequenceFormat::UniProt => "UniProt",
        sequence::SequenceFormat::GenBank => "GenBank",
        sequence::SequenceFormat::Embl => "EMBL",
        sequence::SequenceFormat::Ddbj => "DDBJ",
        sequence::SequenceFormat::Stockholm => "Stockholm",
    });
    let mut sequences = Vec::new();
    let mut sq = Sequence::new();
    while sqf
        .read(&mut sq)
        .map_err(|e| format!("Error reading {format_name} query file: {e}"))?
    {
        sequences.push(sq.clone());
        sq.reuse();
    }
    if sequences.is_empty() {
        return Err(format!(
            "Error reading {format_name} query file: no sequences found in {}",
            args.hmmfile.display()
        ));
    }
    if !explicit_format && fasta_queries_are_alignment_ambiguous(&sequences) {
        return Err(
            "Query file type could be either aligned or unaligned; please specify (--qformat [afa|fasta])"
                .to_string(),
        );
    }

    let bg = Bg::new(&abc);
    sequences
        .iter()
        .map(|seq| build_nhmmer_single_sequence_hmm(seq, &abc, &bg, args.popen, args.pextend))
        .collect()
}

fn read_query_file_bytes(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    if path == std::path::Path::new("-") {
        std::io::stdin().lock().read_to_end(&mut bytes)?;
    } else {
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    }
    Ok(bytes)
}

fn fasta_queries_are_alignment_ambiguous(sequences: &[Sequence]) -> bool {
    sequences.len() > 1
        && sequences
            .first()
            .is_some_and(|first| sequences.iter().all(|seq| seq.n == first.n))
}

fn guess_fasta_alphabet(bytes: &[u8]) -> Result<AlphabetType, String> {
    let mut counts = [0usize; 256];
    let mut total = 0usize;
    let mut in_header = false;
    let mut at_line_start = true;

    for &raw in bytes {
        if at_line_start && raw == b'>' {
            in_header = true;
            at_line_start = false;
            continue;
        }
        if raw == b'\n' || raw == b'\r' {
            in_header = false;
            at_line_start = true;
            continue;
        }
        if in_header {
            at_line_start = false;
            continue;
        }
        at_line_start = false;
        if raw.is_ascii_whitespace() {
            continue;
        }
        if raw == b'-' || raw == b'.' || raw == b'_' || raw == b'~' {
            continue;
        }
        if raw.is_ascii_alphabetic() {
            let ch = raw.to_ascii_uppercase();
            counts[ch as usize] += 1;
            total += 1;
        }
    }

    if total == 0 {
        return Err("Unable to guess alphabet for empty query sequence file".to_string());
    }

    let idx = |ch: u8| ch as usize;
    let a = counts[idx(b'A')];
    let c = counts[idx(b'C')];
    let g = counts[idx(b'G')];
    let t = counts[idx(b'T')];
    let u = counts[idx(b'U')];
    let n = counts[idx(b'N')] + counts[idx(b'X')];
    let dna_iupac = b"RYMKSWHBVD"
        .iter()
        .map(|&ch| counts[idx(ch)])
        .sum::<usize>();
    let dna_core = a + c + g + t + n + dna_iupac;
    let rna_core = a + c + g + u + n + dna_iupac;
    let frac = |count: usize| count as f64 / total as f64;

    if frac(dna_core) >= 0.98 && u == 0 {
        return Ok(AlphabetType::Dna);
    }
    if frac(rna_core) >= 0.98 && t == 0 {
        return Ok(AlphabetType::Rna);
    }
    Err(
        "Unable to guess alphabet for query sequence file; please specify --dna or --rna"
            .to_string(),
    )
}

fn is_stockholm_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("stockholm")
        || format.eq_ignore_ascii_case("sto")
        || format.eq_ignore_ascii_case("pfam")
}

fn query_sequence_format(format: &str) -> Option<sequence::SequenceFormat> {
    sequence::SequenceFormat::from_name(format)
}

fn build_nhmmer_single_sequence_hmm(
    seq: &Sequence,
    abc: &Alphabet,
    bg: &Bg,
    popen: f32,
    pextend: f32,
) -> Result<Hmm, String> {
    let cond = dna1_conditional_probabilities(abc, &bg.f)?;
    let mut hmm = Hmm::new(seq.n, abc.abc_type, abc.k);
    hmm.name = seq.name.clone();

    for node in 0..=seq.n {
        if node > 0 {
            let residue = seq.dsq[node] as usize;
            if residue < cond.len() {
                hmm.mat[node][..abc.k].copy_from_slice(&cond[residue][..abc.k]);
            } else {
                hmm.mat[node][..abc.k].copy_from_slice(&bg.f[..abc.k]);
            }
        }

        hmm.ins[node][..abc.k].copy_from_slice(&bg.f[..abc.k]);
        hmm.t[node][p7hmm::MM] = 1.0 - 2.0 * popen;
        hmm.t[node][p7hmm::MI] = popen;
        hmm.t[node][p7hmm::MD] = popen;
        hmm.t[node][p7hmm::IM] = 1.0 - pextend;
        hmm.t[node][p7hmm::II] = pextend;
        hmm.t[node][p7hmm::DM] = 1.0 - pextend;
        hmm.t[node][p7hmm::DD] = pextend;
    }

    hmm.t[seq.n][p7hmm::MM] = 1.0 - popen;
    hmm.t[seq.n][p7hmm::MD] = 0.0;
    hmm.t[seq.n][p7hmm::DM] = 1.0;
    hmm.t[seq.n][p7hmm::DD] = 0.0;

    set_nhmmer_single_sequence_composition(&mut hmm);
    set_nhmmer_single_sequence_consensus(&mut hmm, seq, abc);
    hmmer_pure_rs::calibrate::calibrate(&mut hmm, abc, bg);
    hmm.nseq = 1;
    hmm.eff_nseq = 1.0;
    Ok(hmm)
}

const DNA1_CANONICAL_SCORES: [[i32; 4]; 4] = [
    [41, -32, -26, -26],
    [-32, 39, -38, -17],
    [-26, -38, 46, -31],
    [-26, -17, -31, 39],
];

fn dna1_conditional_probabilities(abc: &Alphabet, bg_f: &[f32]) -> Result<Vec<Vec<f32>>, String> {
    if abc.k != 4 {
        return Err("nhmmer single-sequence query models require DNA or RNA alphabet".to_string());
    }
    let lambda = solve_nhmmer_dna1_lambda(bg_f)?;
    let mut joint = vec![vec![0.0_f64; abc.kp]; abc.kp];
    for a in 0..abc.k {
        for b in 0..abc.k {
            joint[a][b] = (bg_f[a] as f64)
                * (bg_f[b] as f64)
                * c_exp_f64(lambda * DNA1_CANONICAL_SCORES[a][b] as f64);
        }
    }

    for row in joint.iter_mut().take(abc.k) {
        for jp in abc.k + 1..abc.kp - 2 {
            row[jp] = (0..abc.k)
                .filter(|&j| abc.degen[jp][j])
                .map(|j| row[j])
                .sum();
        }
    }
    for ip in abc.k + 1..abc.kp - 2 {
        let canonical_cols: Vec<f64> = (0..abc.k)
            .map(|j| {
                (0..abc.k)
                    .filter(|&i| abc.degen[ip][i])
                    .map(|i| joint[i][j])
                    .sum()
            })
            .collect();
        for (j, &value) in canonical_cols.iter().enumerate() {
            joint[ip][j] = value;
        }
        for jp in abc.k + 1..abc.kp - 2 {
            joint[ip][jp] = (0..abc.k)
                .filter(|&j| abc.degen[jp][j])
                .map(|j| joint[ip][j])
                .sum();
        }
    }

    let any = abc.unknown_code() as usize;
    let mut cond = vec![vec![0.0_f32; abc.k]; abc.kp];
    for residue in 0..abc.kp - 2 {
        let denom = joint[residue][any];
        if denom > 0.0 {
            for b in 0..abc.k {
                cond[residue][b] = (joint[residue][b] / denom) as f32;
            }
        } else {
            cond[residue][..abc.k].copy_from_slice(&bg_f[..abc.k]);
        }
    }
    cond[abc.kp - 2][..abc.k].copy_from_slice(&bg_f[..abc.k]);
    cond[abc.kp - 1][..abc.k].copy_from_slice(&bg_f[..abc.k]);
    Ok(cond)
}

fn solve_nhmmer_dna1_lambda(bg_f: &[f32]) -> Result<f64, String> {
    let max_score = DNA1_CANONICAL_SCORES
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .max()
        .unwrap_or(0) as f64;
    if max_score <= 0.0 {
        return Err("DNA1 score matrix has no positive scores".to_string());
    }

    let mut hi = 1.0 / max_score;
    while hi < 50.0 && nhmmer_dna1_lambda_f(bg_f, hi) <= 0.0 {
        hi *= 2.0;
    }
    if nhmmer_dna1_lambda_f(bg_f, hi) <= 0.0 {
        return Err("failed to bracket lambda root for DNA1 score matrix".to_string());
    }

    let mut lo = 0.0_f64;
    for _ in 0..80 {
        let mid = (lo + hi) * 0.5;
        if nhmmer_dna1_lambda_f(bg_f, mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok((lo + hi) * 0.5)
}

fn nhmmer_dna1_lambda_f(bg_f: &[f32], lambda: f64) -> f64 {
    let mut fx = -1.0_f64;
    for a in 0..4 {
        for b in 0..4 {
            fx += (bg_f[a] as f64)
                * (bg_f[b] as f64)
                * c_exp_f64(lambda * DNA1_CANONICAL_SCORES[a][b] as f64);
        }
    }
    fx
}

fn set_nhmmer_single_sequence_composition(hmm: &mut Hmm) {
    let mut mocc = vec![0.0_f32; hmm.m + 1];
    let mut iocc = vec![0.0_f32; hmm.m + 1];

    if hmm.m > 0 {
        mocc[1] = hmm.t[0][p7hmm::MI] + hmm.t[0][p7hmm::MM];
        for k in 2..=hmm.m {
            mocc[k] = mocc[k - 1] * (hmm.t[k - 1][p7hmm::MM] + hmm.t[k - 1][p7hmm::MI])
                + (1.0 - mocc[k - 1]) * hmm.t[k - 1][p7hmm::DM];
        }
    }

    iocc[0] = hmm.t[0][p7hmm::MI] / hmm.t[0][p7hmm::IM];
    for k in 1..=hmm.m {
        iocc[k] = mocc[k] * hmm.t[k][p7hmm::MI] / hmm.t[k][p7hmm::IM];
    }

    for x in 0..hmm.abc_k.min(p7hmm::MAXABET) {
        hmm.compo[x] = hmm.ins[0][x] * iocc[0];
    }
    for k in 1..=hmm.m {
        for x in 0..hmm.abc_k.min(p7hmm::MAXABET) {
            hmm.compo[x] += hmm.mat[k][x] * mocc[k] + hmm.ins[k][x] * iocc[k];
        }
    }

    let sum: f32 = hmm.compo[..hmm.abc_k.min(p7hmm::MAXABET)].iter().sum();
    if sum > 0.0 {
        for x in 0..hmm.abc_k.min(p7hmm::MAXABET) {
            hmm.compo[x] /= sum;
        }
    }
    hmm.flags |= p7hmm::P7H_COMPO;
}

fn set_nhmmer_single_sequence_consensus(hmm: &mut Hmm, seq: &Sequence, abc: &Alphabet) {
    let mut cons = vec![b' '; hmm.m + 2];
    for (node, cons_byte) in cons.iter_mut().enumerate().take(hmm.m + 1).skip(1) {
        let residue = seq.dsq[node];
        if abc.is_residue(residue) {
            *cons_byte = abc.sym[residue as usize];
        }
    }
    hmm.consensus = Some(cons);
    hmm.flags |= p7hmm::P7H_CONS;
}

fn read_query_msa_hmms(args: &Args) -> Result<Vec<hmmer_pure_rs::Hmm>, String> {
    let mut msas = match args.qformat.as_deref() {
        Some(format) if is_text_msa_query_format(format) => {
            read_text_msa_query(format, &args.hmmfile)
                .map_err(|e| format!("Error reading MSA query file: {e}"))?
        }
        _ => read_stockholm_query(&args.hmmfile)
            .map_err(|e| format!("Error reading MSA query file: {e}"))?,
    };
    if msas.is_empty() {
        return Err(format!(
            "Error reading MSA query file: no alignments found in {}",
            args.hmmfile.display()
        ));
    }

    let msa_count = msas.len();
    let mut unnamed = 0usize;
    for (idx, alignment) in msas.iter_mut().enumerate() {
        if alignment.name.is_empty() {
            if msa_count > 1 {
                return Err(format!(
                    "Name annotation is required for each alignment in a multi MSA query file; failed on #{}",
                    idx + 1
                ));
            }
            alignment.name = query_name_from_path(&args.hmmfile);
            unnamed += 1;
        }
    }
    if unnamed > 1 {
        return Err("Error assigning names to query alignments".to_string());
    }

    let first_abc_type = if args.dna {
        AlphabetType::Dna
    } else if args.rna {
        AlphabetType::Rna
    } else {
        guess_msa_alphabet(&msas[0])?
    };
    if first_abc_type != AlphabetType::Dna && first_abc_type != AlphabetType::Rna {
        return Err(
            "Error: Invalid alphabet type in query for nhmmer. Expect DNA or RNA.".to_string(),
        );
    }
    let abc = Alphabet::new(first_abc_type);
    let bg = Bg::new(&abc);
    let score_matrix = if args.singlemx || args.mxfile.is_some() {
        Some(nhmmer_score_matrix(args, &abc)?)
    } else {
        None
    };

    let mut hmms = Vec::with_capacity(msas.len());
    for alignment in &msas {
        if args.dna && guess_msa_alphabet(alignment)? != AlphabetType::Dna {
            return Err(format!(
                "Error reading alignment from file {}: expected DNA query alignment",
                args.hmmfile.display()
            ));
        }
        if args.rna && guess_msa_alphabet(alignment)? != AlphabetType::Rna {
            return Err(format!(
                "Error reading alignment from file {}: expected RNA query alignment",
                args.hmmfile.display()
            ));
        }
        let hmm = if args.singlemx && alignment.nseq == 1 {
            build_nhmmer_singlemx_msa_hmm(
                alignment,
                &abc,
                &bg,
                score_matrix.as_ref().unwrap(),
                args.popen,
                args.pextend,
                args.seed,
            )?
        } else {
            builder::build_hmm_from_msa_with_prior(
                alignment,
                &abc,
                &bg,
                0.5,
                0.5,
                args.qformat.as_deref().is_some_and(is_a2m_query_format),
                builder::RelativeWeighting::PositionBased,
                builder::EffectiveSeqNumber::Entropy {
                    target_re: None,
                    target_sigma: Some(45.0),
                },
                PriorStrategy::Default,
                CalibrationConfig::default(),
                args.seed,
            )
        };
        hmms.push(hmm);
    }
    Ok(hmms)
}

fn nhmmer_score_matrix(args: &Args, abc: &Alphabet) -> Result<seqmodel::ScoreMatrix, String> {
    if let Some(path) = args.mxfile.as_ref() {
        seqmodel::ScoreMatrix::from_file_for_alphabet(path, abc)
            .map_err(|e| format!("Error: nhmmer {e}"))
    } else {
        seqmodel::ScoreMatrix::builtin_for_alphabet(&args.matrix, abc.abc_type)
            .map_err(|e| format!("Error: nhmmer {e}"))
    }
}

fn build_nhmmer_singlemx_msa_hmm(
    alignment: &msa::Msa,
    abc: &Alphabet,
    bg: &Bg,
    matrix: &seqmodel::ScoreMatrix,
    popen: f32,
    pextend: f32,
    seed: u32,
) -> Result<Hmm, String> {
    let mut dsq = vec![DSQ_SENTINEL];
    for &sym in &alignment.aseq[0] {
        if matches!(sym, b'-' | b'.' | b'_' | b'~') || sym.is_ascii_whitespace() {
            continue;
        }
        let code = abc.digitize_symbol(sym);
        if !abc.is_residue(code) {
            return Err(format!(
                "Error: nhmmer --singlemx sequence contains non-residue '{}'",
                sym as char
            ));
        }
        dsq.push(code);
    }
    if dsq.len() == 1 {
        return Err("Error: nhmmer --singlemx requires at least one residue".to_string());
    }
    dsq.push(DSQ_SENTINEL);
    let name = if !alignment.name.is_empty() {
        alignment.name.as_str()
    } else {
        alignment.sqname[0].as_str()
    };
    let mut hmm = seqmodel::build_single_seq_hmm_with_matrix_and_calibration(
        name,
        &dsq,
        dsq.len() - 2,
        abc,
        bg,
        matrix,
        popen,
        pextend,
        seed,
        CalibrationConfig::default(),
    )
    .map_err(|e| format!("Error: nhmmer --singlemx failed to build score matrix model: {e}"))?;
    if let Some(ref acc) = alignment.acc {
        hmm.acc = Some(acc.clone());
        hmm.flags |= p7hmm::P7H_ACC;
    }
    if let Some(ref desc) = alignment.desc {
        hmm.desc = Some(desc.clone());
        hmm.flags |= p7hmm::P7H_DESC;
    }
    Ok(hmm)
}

fn is_afa_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("afa")
        || format.eq_ignore_ascii_case("afasta")
        || format.eq_ignore_ascii_case("alignedfasta")
        || format.eq_ignore_ascii_case("aligned-fasta")
}

fn is_a2m_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("a2m")
}

fn is_psiblast_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("psiblast")
}

fn is_clustal_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("clustal") || format.eq_ignore_ascii_case("clustallike")
}

fn is_selex_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("selex")
}

fn is_phylip_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("phylip")
}

fn is_phylips_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("phylips")
}

fn is_text_msa_query_format(format: &str) -> bool {
    is_afa_query_format(format)
        || is_a2m_query_format(format)
        || is_psiblast_query_format(format)
        || is_clustal_query_format(format)
        || is_selex_query_format(format)
        || is_phylip_query_format(format)
        || is_phylips_query_format(format)
}

fn read_stockholm_query(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<msa::Msa>> {
    if path == std::path::Path::new("-") {
        msa::read_stockholm_from_reader(BufReader::new(std::io::stdin().lock()))
    } else {
        msa::read_stockholm(path)
    }
}

fn read_text_msa_query(format: &str, path: &std::path::Path) -> Result<Vec<msa::Msa>, String> {
    let bytes = read_query_file_bytes(path)
        .map_err(|e| format!("Error reading {} file: {e}", format.to_ascii_uppercase()))?;
    let mut reader = std::io::Cursor::new(bytes);
    let name = query_name_from_path(path);
    let result = if is_a2m_query_format(format) {
        msa::read_a2m_from_reader(&mut reader, name)
    } else if is_afa_query_format(format) {
        msa::read_afa_from_reader(&mut reader, name)
    } else if is_psiblast_query_format(format) {
        msa::read_psiblast_from_reader(&mut reader, name)
    } else if is_clustal_query_format(format) {
        msa::read_clustal_from_reader(&mut reader, name)
    } else if is_selex_query_format(format) {
        msa::read_selex_from_reader(&mut reader, name)
    } else if is_phylip_query_format(format) {
        msa::read_phylip_from_reader(&mut reader, name)
    } else if is_phylips_query_format(format) {
        msa::read_phylips_from_reader(&mut reader, name)
    } else {
        unreachable!("unsupported text MSA query format {format}")
    };
    result.map_err(|e| e.to_string())
}

fn query_name_from_path(path: &std::path::Path) -> String {
    if path == std::path::Path::new("-") {
        return "Query".to_string();
    }
    path.file_stem()
        .or_else(|| path.file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "Query".to_string())
}

fn guess_msa_alphabet(msa: &msa::Msa) -> Result<AlphabetType, String> {
    let mut counts = [0usize; 256];
    let mut total = 0usize;
    for row in &msa.aseq {
        for &raw in row {
            if matches!(raw, b'-' | b'.' | b'_' | b'~') || raw.is_ascii_whitespace() {
                continue;
            }
            let ch = raw.to_ascii_uppercase();
            counts[ch as usize] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return Err("Unable to guess alphabet for empty query alignment".to_string());
    }

    let idx = |ch: u8| ch as usize;
    let a = counts[idx(b'A')];
    let c = counts[idx(b'C')];
    let g = counts[idx(b'G')];
    let t = counts[idx(b'T')];
    let u = counts[idx(b'U')];
    let n = counts[idx(b'N')] + counts[idx(b'X')];
    let dna_iupac = b"RYMKSWHBVD"
        .iter()
        .map(|&ch| counts[idx(ch)])
        .sum::<usize>();
    let dna_core = a + c + g + t + n + dna_iupac;
    let rna_core = a + c + g + u + n + dna_iupac;
    let frac = |count: usize| count as f64 / total as f64;

    if frac(dna_core) >= 0.98 && u == 0 {
        return Ok(AlphabetType::Dna);
    }
    if frac(rna_core) >= 0.98 && t == 0 {
        return Ok(AlphabetType::Rna);
    }
    Err("Unable to guess alphabet for query alignment; please specify --dna or --rna".to_string())
}

fn open_target_seq_file(
    path: &Path,
    abc: &Alphabet,
    tformat: Option<&str>,
) -> hmmer_pure_rs::errors::HmmerResult<sequence::SeqFile<Box<dyn std::io::Read>>> {
    if tformat.is_some_and(|format| format.eq_ignore_ascii_case("fasta")) {
        sequence::open_fasta_seq_file(path, abc)
    } else {
        sequence::open_seq_file(path, abc)
    }
}

struct NhmmerTargetDb {
    sequences: Vec<Sequence>,
    fm_sequence_meta: Vec<NhmmerFmSequenceMeta>,
    fm_ambiguities: Vec<(usize, usize)>,
    fm_index_records: Vec<NhmmerFmIndexRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NhmmerCandidateWindow {
    n: usize,
    length: usize,
    k: usize,
}

#[derive(Clone, Debug)]
struct NhmmerScoredCandidateWindow {
    window: NhmmerCandidateWindow,
    score_bits: f32,
}

fn read_target_database(args: &Args, abc: &Alphabet) -> Result<NhmmerTargetDb, String> {
    let path = &args.seqdb;
    let tformat = args.tformat.as_deref();
    if tformat.is_some_and(is_fmindex_target_format) {
        if args.restrictdb_stkey.is_some() || args.restrictdb_n.is_some() || args.ssifile.is_some()
        {
            return Err("restricted database options require FASTA targets".to_string());
        }
        reject_fmindex_max(args.max)?;
        return read_makehmmerdb_target_database(path, abc);
    }

    let restrict_requested =
        args.restrictdb_stkey.is_some() || args.restrictdb_n.is_some() || args.ssifile.is_some();
    if !restrict_requested && tformat.is_none() && path != Path::new("-") {
        let mut magic = [0u8; 8];
        if std::fs::File::open(path)
            .and_then(|mut file| file.read_exact(&mut magic))
            .is_ok()
            && magic == *b"HMMERDB\0"
        {
            reject_fmindex_max(args.max)?;
            return read_makehmmerdb_target_database(path, abc);
        }

        if is_likely_raw_makehmmerdb_c_stream(&magic) {
            reject_fmindex_max(args.max)?;
            return read_makehmmerdb_target_database(path, abc);
        }
    }

    let mut sqf = if let Some(stkey) = args.restrictdb_stkey.as_deref() {
        crate::subcmd::hmmsearch::open_restricted_target_seq_file(
            path,
            abc,
            tformat,
            stkey,
            args.ssifile.as_deref(),
        )?
    } else {
        open_target_seq_file(path, abc, tformat).map_err(|e| e.to_string())?
    };
    let mut sequences = Vec::new();
    let mut sq = Sequence::new();
    let mut restrict_seen = 0usize;
    while sqf.read(&mut sq).map_err(|e| e.to_string())? {
        if args
            .restrictdb_n
            .is_some_and(|limit| restrict_seen >= limit)
        {
            break;
        }
        restrict_seen += 1;
        sequences.push(sq.clone());
        sq.reuse();
    }
    Ok(NhmmerTargetDb {
        sequences,
        fm_sequence_meta: Vec::new(),
        fm_ambiguities: Vec::new(),
        fm_index_records: Vec::new(),
    })
}

fn reject_fmindex_max(do_max: bool) -> Result<(), String> {
    if do_max {
        Err("--max flag is incompatible with the fmindex target type".to_string())
    } else {
        Ok(())
    }
}

fn is_fmindex_target_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("fmindex")
        || format.eq_ignore_ascii_case("fm")
        || format.eq_ignore_ascii_case("hmmerdb")
}

fn is_likely_raw_makehmmerdb_c_stream(prefix: &[u8]) -> bool {
    matches!(prefix.first(), Some(0 | 1))
        && prefix.get(1) == Some(&0)
        && prefix.get(2) == Some(&4)
        && prefix.get(3) == Some(&2)
}

fn read_makehmmerdb_target_database(path: &Path, abc: &Alphabet) -> Result<NhmmerTargetDb, String> {
    if path == Path::new("-") {
        return Err("nhmmer --tformat fmindex does not support stdin targets yet".to_string());
    }
    let bytes = read_makehmmerdb_database_bytes(path)?;
    parse_makehmmerdb_target_database(&bytes, abc)
}

fn read_makehmmerdb_database_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path).map_err(|e| {
        format!(
            "failed to open FM-index target database {}: {e}",
            path.display()
        )
    })?;
    let metadata = file.metadata().map_err(|e| {
        format!(
            "failed to stat FM-index target database {}: {e}",
            path.display()
        )
    })?;
    let len = metadata.len();
    if len > MAKEHMMERDB_MAX_IN_MEMORY_BYTES {
        return Err(format!(
            "FM-index target database {} is too large for the current in-memory reader ({} bytes > {} bytes)",
            path.display(),
            len,
            MAKEHMMERDB_MAX_IN_MEMORY_BYTES
        ));
    }
    let capacity = u64_to_usize(len, "FM-index target database size")?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|e| {
        format!(
            "failed to reserve memory for FM-index target database {}: {e}",
            path.display()
        )
    })?;
    let mut limited = file.take(MAKEHMMERDB_MAX_IN_MEMORY_BYTES + 1);
    limited.read_to_end(&mut bytes).map_err(|e| {
        format!(
            "failed to read FM-index target database {}: {e}",
            path.display()
        )
    })?;
    if bytes.len() as u64 > MAKEHMMERDB_MAX_IN_MEMORY_BYTES {
        return Err(format!(
            "FM-index target database {} is too large for the current in-memory reader (exceeds {} bytes)",
            path.display(),
            MAKEHMMERDB_MAX_IN_MEMORY_BYTES
        ));
    }
    Ok(bytes)
}

#[derive(Debug)]
struct NhmmerFmSequenceMeta {
    name: String,
    acc: String,
    desc: String,
    fm_start: usize,
    length: usize,
}

fn parse_makehmmerdb_target_database(
    bytes: &[u8],
    abc: &Alphabet,
) -> Result<NhmmerTargetDb, String> {
    let is_container = bytes.starts_with(b"HMMERDB\0");
    let container_index_records = if is_container {
        parse_makehmmerdb_index_extension(bytes)?
    } else {
        Vec::new()
    };
    let payload_bytes = if is_container {
        let stream_start = find_bytes(bytes, b"HMMERDB_C_STREAM\0")
            .ok_or_else(|| "makehmmerdb file is missing the C FM stream extension".to_string())?;
        let mut cursor = Cursor::new(&bytes[stream_start + b"HMMERDB_C_STREAM\0".len()..]);
        let version = read_u32(&mut cursor)?;
        if version != 1 {
            return Err(format!(
                "unsupported makehmmerdb C stream version {version}"
            ));
        }
        let payload_len = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb C stream payload length",
        )?;
        let payload_start = stream_start + b"HMMERDB_C_STREAM\0".len() + 12;
        let payload_end = payload_start
            .checked_add(payload_len)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| "makehmmerdb C stream payload is truncated".to_string())?;
        &bytes[payload_start..payload_end]
    } else {
        bytes
    };
    let mut payload = Cursor::new(payload_bytes);

    let (fwd_only, freq_sa, freq_cnt_b, block_count, sequences, ambiguities) =
        read_makehmmerdb_c_metadata_payload(&mut payload)?;
    let mut records_by_block = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        let record = read_makehmmerdb_c_fm_record(&mut payload, freq_sa, freq_cnt_b, true)?;
        records_by_block.push(record);
        if !fwd_only {
            let _ = read_makehmmerdb_c_fm_record(&mut payload, freq_sa, freq_cnt_b, false)?;
        }
    }
    if makehmmerdb_remaining(&payload) != 0 {
        return Err("trailing bytes after makehmmerdb C stream records".to_string());
    }
    if records_by_block.is_empty() && !sequences.is_empty() {
        return Err("makehmmerdb C stream has sequence metadata but no FM records".to_string());
    }
    validate_makehmmerdb_fm_records(&records_by_block, ambiguities.len())?;
    if !container_index_records.is_empty() {
        validate_makehmmerdb_index_records(
            &container_index_records,
            &records_by_block,
            fwd_only,
            ambiguities.len(),
        )?;
    }
    let fm_index_records = if !container_index_records.is_empty() {
        container_index_records
    } else if !is_container {
        build_raw_c_stream_fm_index_records(&records_by_block, fwd_only)?
    } else {
        Vec::new()
    };

    let mut out = Vec::with_capacity(sequences.len());
    for (seq_idx, seq_meta) in sequences.iter().enumerate() {
        let block = records_by_block
            .iter()
            .find(|record| {
                seq_idx >= record.seq_offset
                    && seq_idx < record.seq_offset + record.seq_count
                    && seq_meta.fm_start + seq_meta.length <= record.text.len()
            })
            .ok_or_else(|| {
                format!(
                    "sequence {} is outside all makehmmerdb FM text blocks",
                    seq_meta.name
                )
            })?;
        let text = &block.text[seq_meta.fm_start..seq_meta.fm_start + seq_meta.length];
        let mut seq = Sequence::new();
        seq.name = seq_meta.name.clone();
        seq.acc = seq_meta.acc.clone();
        seq.desc = seq_meta.desc.clone();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();
        out.push(seq);
    }
    Ok(NhmmerTargetDb {
        sequences: out,
        fm_sequence_meta: sequences,
        fm_ambiguities: ambiguities,
        fm_index_records,
    })
}

#[derive(Debug)]
struct NhmmerFmIndexRecord {
    block_id: usize,
    kind: u32,
    text_start: usize,
    text_len: usize,
    seq_offset: usize,
    seq_count: usize,
    ambig_offset: usize,
    ambig_count: usize,
    overlap_bases: usize,
    fm: FmIndex,
}

fn parse_makehmmerdb_index_extension(bytes: &[u8]) -> Result<Vec<NhmmerFmIndexRecord>, String> {
    let Some(index_start) = find_bytes(bytes, MAKEHMMERDB_INDEX_MAGIC) else {
        return Ok(Vec::new());
    };
    let mut cursor = Cursor::new(&bytes[index_start + MAKEHMMERDB_INDEX_MAGIC.len()..]);
    let version = read_u32(&mut cursor)?;
    if version != 1 {
        return Err(format!(
            "unsupported makehmmerdb FM-index extension version {version}"
        ));
    }
    let _fwd_only = read_u32(&mut cursor)? != 0;
    let record_count = u64_to_usize(
        read_u64(&mut cursor)?,
        "makehmmerdb FM-index record table count",
    )?;
    ensure_makehmmerdb_remaining(
        &cursor,
        record_count,
        MAKEHMMERDB_INDEX_MIN_RECORD_BYTES,
        "makehmmerdb FM-index record table",
    )?;
    let mut records = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        let block_id = u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index block id")?;
        let text_start = u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index text start")?;
        let text_len = u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index text length")?;
        let seq_offset = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index sequence offset",
        )?;
        let seq_count = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index sequence count",
        )?;
        let ambig_offset = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index ambiguity offset",
        )?;
        let ambig_count = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index ambiguity count",
        )?;
        let overlap_bases =
            u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index overlap bases")?;
        let kind = read_u32(&mut cursor)?;
        let bwt_len = u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index BWT length")?;
        let sa_len = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index suffix array length",
        )?;
        let c_len = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index C table length",
        )?;
        if c_len != 256 {
            return Err(format!(
                "makehmmerdb FM-index C table has {c_len} entries, expected 256"
            ));
        }
        let bwt_bytes = bwt_len;
        let sa_bytes = checked_count_bytes(sa_len, 4, "makehmmerdb FM-index suffix array")?;
        let c_bytes = checked_count_bytes(c_len, 8, "makehmmerdb FM-index C table")?;
        let remaining_record_bytes = bwt_bytes
            .checked_add(sa_bytes)
            .and_then(|value| value.checked_add(c_bytes))
            .ok_or_else(|| "makehmmerdb FM-index record byte span overflows usize".to_string())?;
        if makehmmerdb_remaining(&cursor) < remaining_record_bytes {
            return Err("makehmmerdb FM-index record payload is truncated".to_string());
        }

        let mut bwt = vec![0u8; bwt_len];
        cursor
            .read_exact(&mut bwt)
            .map_err(|e| format!("truncated makehmmerdb FM-index BWT: {e}"))?;

        let mut sa = Vec::with_capacity(sa_len);
        for _ in 0..sa_len {
            sa.push(read_i32(&mut cursor)?);
        }

        let mut c = [0usize; 256];
        for value in &mut c {
            *value = u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index C table value")?;
        }

        let fm = FmIndex::from_parts(bwt, sa, c, text_len)?;
        records.push(NhmmerFmIndexRecord {
            block_id,
            kind,
            text_start,
            text_len,
            seq_offset,
            seq_count,
            ambig_offset,
            ambig_count,
            overlap_bases,
            fm,
        });
    }
    Ok(records)
}

fn validate_makehmmerdb_fm_records(
    records_by_block: &[NhmmerFmRecord],
    ambiguity_count: usize,
) -> Result<(), String> {
    for record in records_by_block {
        if record.seq_offset.checked_add(record.seq_count).is_none() {
            return Err("makehmmerdb C stream sequence span overflows usize".to_string());
        }
        let Some(ambig_end) = record.ambig_offset.checked_add(record.ambig_count) else {
            return Err("makehmmerdb C stream ambiguity span overflows usize".to_string());
        };
        if ambig_end > ambiguity_count {
            return Err(
                "makehmmerdb C stream ambiguity span is outside metadata table".to_string(),
            );
        }
        if record.overlap_bases > record.text_bases_len {
            return Err("makehmmerdb C stream overlap exceeds block text length".to_string());
        }
    }
    Ok(())
}

fn validate_makehmmerdb_index_records(
    records: &[NhmmerFmIndexRecord],
    records_by_block: &[NhmmerFmRecord],
    fwd_only: bool,
    ambiguity_count: usize,
) -> Result<(), String> {
    let block_count = records_by_block.len();
    let mut expected_starts = Vec::with_capacity(block_count);
    let mut text_start = 0usize;
    for block in records_by_block {
        expected_starts.push(text_start);
        text_start = text_start
            .checked_add(block.text_bases_len)
            .ok_or_else(|| "makehmmerdb FM-index text span overflows usize".to_string())?;
    }
    let mut seen = vec![[false; 2]; block_count];
    for record in records {
        if record.block_id >= block_count {
            return Err(format!(
                "makehmmerdb FM-index record references block {} but only {block_count} block(s) exist",
                record.block_id
            ));
        }
        if record.kind > 1 {
            return Err(format!(
                "makehmmerdb FM-index record has unsupported strand kind {}",
                record.kind
            ));
        }
        if record.fm.n != record.text_len {
            return Err("makehmmerdb FM-index text length metadata is inconsistent".to_string());
        }
        if record.text_start.checked_add(record.text_len).is_none() {
            return Err("makehmmerdb FM-index text span overflows usize".to_string());
        }
        if record.seq_offset.checked_add(record.seq_count).is_none() {
            return Err("makehmmerdb FM-index sequence span overflows usize".to_string());
        }
        if record
            .ambig_offset
            .checked_add(record.ambig_count)
            .is_none()
        {
            return Err("makehmmerdb FM-index ambiguity span overflows usize".to_string());
        }
        if record.overlap_bases > record.text_len {
            return Err("makehmmerdb FM-index overlap exceeds block text length".to_string());
        }
        let block = &records_by_block[record.block_id];
        if record.text_start != expected_starts[record.block_id]
            || record.text_len != block.text_bases_len
            || record.seq_offset != block.seq_offset
            || record.seq_count != block.seq_count
            || record.ambig_offset != block.ambig_offset
            || record.ambig_count != block.ambig_count
            || record.overlap_bases != block.overlap_bases
        {
            return Err(
                "makehmmerdb FM-index record metadata does not match C stream block".to_string(),
            );
        }
        let expected_text = &block.text[..block.text_bases_len];
        let expected_fm = if record.kind == 0 {
            let reversed_text: Vec<u8> = expected_text.iter().rev().copied().collect();
            FmIndex::build(&reversed_text)
        } else {
            FmIndex::build(expected_text)
        };
        if record.fm.bwt != expected_fm.bwt
            || record.fm.sa != expected_fm.sa
            || record.fm.c != expected_fm.c
        {
            return Err(
                "makehmmerdb FM-index record does not match C stream block text".to_string(),
            );
        }
        let Some(ambig_end) = record.ambig_offset.checked_add(record.ambig_count) else {
            return Err("makehmmerdb FM-index ambiguity span overflows usize".to_string());
        };
        if ambig_end > ambiguity_count {
            return Err(
                "makehmmerdb FM-index ambiguity span is outside metadata table".to_string(),
            );
        }
        let kind = record.kind as usize;
        if seen[record.block_id][kind] {
            return Err("makehmmerdb FM-index contains duplicate block/strand record".to_string());
        }
        seen[record.block_id][kind] = true;
    }
    for (block_id, block_seen) in seen.iter().enumerate() {
        if !block_seen[0] {
            return Err(format!(
                "makehmmerdb FM-index is missing forward-strand record for block {block_id}"
            ));
        }
        if !fwd_only && !block_seen[1] {
            return Err(format!(
                "makehmmerdb FM-index is missing reverse-strand record for block {block_id}"
            ));
        }
    }
    Ok(())
}

fn build_raw_c_stream_fm_index_records(
    records_by_block: &[NhmmerFmRecord],
    fwd_only: bool,
) -> Result<Vec<NhmmerFmIndexRecord>, String> {
    let record_passes = if fwd_only { 1 } else { 2 };
    let mut records = Vec::with_capacity(records_by_block.len() * record_passes);
    let mut text_start = 0usize;

    for (block_id, block) in records_by_block.iter().enumerate() {
        let text_len = block.text_bases_len;
        if text_len == 0 {
            text_start = text_start
                .checked_add(text_len)
                .ok_or_else(|| "raw makehmmerdb C-stream text span overflows usize".to_string())?;
            continue;
        }

        let block_text = &block.text[..text_len];
        let reversed_text: Vec<u8> = block_text.iter().rev().copied().collect();
        records.push(NhmmerFmIndexRecord {
            block_id,
            kind: 0,
            text_start,
            text_len,
            seq_offset: block.seq_offset,
            seq_count: block.seq_count,
            ambig_offset: block.ambig_offset,
            ambig_count: block.ambig_count,
            overlap_bases: block.overlap_bases,
            fm: FmIndex::build(&reversed_text),
        });

        if !fwd_only {
            records.push(NhmmerFmIndexRecord {
                block_id,
                kind: 1,
                text_start,
                text_len,
                seq_offset: block.seq_offset,
                seq_count: block.seq_count,
                ambig_offset: block.ambig_offset,
                ambig_count: block.ambig_count,
                overlap_bases: block.overlap_bases,
                fm: FmIndex::build(block_text),
            });
        }

        text_start = text_start
            .checked_add(text_len)
            .ok_or_else(|| "raw makehmmerdb C-stream text span overflows usize".to_string())?;
    }

    Ok(records)
}

#[cfg(test)]
fn fm_seed_candidate_windows(
    target_db: &NhmmerTargetDb,
    seq_idx: usize,
    hmm: &Hmm,
    bg: &Bg,
    max_length: i32,
    is_complement: bool,
) -> Option<Vec<NhmmerCandidateWindow>> {
    fm_seed_candidate_windows_with_config(
        target_db,
        seq_idx,
        hmm,
        bg,
        max_length,
        0.02, // default --F1
        is_complement,
        NhmmerFmSeedConfig::default(),
    )
}

// FM-index candidate-window generation. Two complementary mechanisms produce
// the candidate windows handed to `search_longtarget`:
//
//   1. A seed-then-rescore stage (the consensus/score-threshold tries below):
//      enumerates candidate model k-mers, locates them in the FM index, extends
//      and merges. Fast but only finds windows around *exact* high-scoring
//      model k-mers.
//   2. `fm_ssv_augment_windows`: the faithful two-sweep port of C
//      `p7_SSVFM_longlarget` / `FM_Recurse` (`src/simd/fm_ssv.rs`), which walks
//      the FM trie scoring SSV diagonals directly over FM intervals and so
//      recovers weak diagonals with no exact k-mer match — closing the
//      sensitivity gap vs C. Its extended diagonals are filtered by C's
//      Gumbel-derived `sc_thresh` (matching `p7_SSVFM_longlarget`) before
//      windowing. Both strands are covered.
//
// The union of windows is deduped/merged in `fm_finalize_seed_windows`, and the
// real SSV/MSV/Viterbi/Forward scoring re-runs downstream in
// `search_longtarget`, which derives the final hit coordinates.
#[allow(clippy::too_many_arguments)]
fn fm_seed_candidate_windows_with_config(
    target_db: &NhmmerTargetDb,
    seq_idx: usize,
    hmm: &Hmm,
    bg: &Bg,
    max_length: i32,
    f1: f64,
    is_complement: bool,
    config: NhmmerFmSeedConfig,
) -> Option<Vec<NhmmerCandidateWindow>> {
    if target_db.fm_index_records.is_empty() {
        return None;
    }
    let seq_meta = target_db.fm_sequence_meta.get(seq_idx)?;
    let seq = target_db.sequences.get(seq_idx)?;
    let seeds = nhmmer_seed_strings_with_config(hmm, bg, config);
    if seeds.is_empty() {
        return None;
    }

    let desired_len = config.ssv_length.max(1).min(seq.n);
    if desired_len == 0 {
        return None;
    }

    let mut windows = Vec::new();
    // When the FM-augment runs in its C-faithful `FM_extendSeed` mode, its
    // windows alone reproduce C `p7_SSVFM_longlarget`'s list; the Rust-only seed
    // sources below are then dropped (and the seed pre-merge / window gate
    // skipped) so the FM residue counters match C. See `fm_ssv_augment_windows`.
    let mut augment_c_faithful = false;
    for record in &target_db.fm_index_records {
        let expected_kind = if is_complement { 1 } else { 0 };
        if record.kind != expected_kind
            || seq_idx < record.seq_offset
            || seq_idx >= record.seq_offset + record.seq_count
        {
            continue;
        }

        let seed_queries =
            fm_seed_queries_for_record(&seeds, record, seq.n, is_complement, hmm, bg);
        for (seed, fm_pos) in
            fm_locate_seed_queries(&record.fm, &seed_queries, NHMMER_FM_WINDOW_SOURCE_LIMIT)
        {
            fm_push_seed_window(
                &mut windows,
                &target_db.fm_ambiguities,
                seq_meta,
                seq,
                record,
                hmm,
                bg,
                is_complement,
                seed,
                fm_pos,
                desired_len,
            );
        }
        for (seed, fm_pos) in fm_locate_model_consensus_seeds(
            &record.fm,
            hmm,
            bg,
            is_complement,
            record.text_len,
            seq.n,
            NHMMER_FM_WINDOW_SOURCE_LIMIT,
            config,
        ) {
            fm_push_seed_window(
                &mut windows,
                &target_db.fm_ambiguities,
                seq_meta,
                seq,
                record,
                hmm,
                bg,
                is_complement,
                &seed,
                fm_pos,
                desired_len,
            );
        }
        let score_trie_limit = if record.text_len <= NHMMER_FM_SCORE_TRIE_NORMAL_TEXT_LIMIT
            && seq.n <= NHMMER_FM_SCORE_TRIE_NORMAL_TEXT_LIMIT
        {
            NHMMER_FM_SCORE_SEED_LIMIT
                .min(NHMMER_FM_WINDOW_SOURCE_LIMIT.saturating_sub(windows.len()))
        } else {
            0
        };
        if score_trie_limit > 0 {
            for (seed, fm_pos) in fm_locate_model_score_seeds_with_start_limit(
                &record.fm,
                hmm,
                bg,
                is_complement,
                record.text_len,
                seq.n,
                score_trie_limit,
                NHMMER_FM_SCORE_TRIE_NORMAL_START_LIMIT,
                config,
            ) {
                fm_push_seed_window(
                    &mut windows,
                    &target_db.fm_ambiguities,
                    seq_meta,
                    seq,
                    record,
                    hmm,
                    bg,
                    is_complement,
                    &seed,
                    fm_pos,
                    desired_len,
                );
            }
        }

        // Exact two-sweep SSV-over-FM augmentation. The kernel needs the block's
        // kind=0 (reversed-text, fmf) and kind=1 (forward-text, fmb) indices as a
        // bi-directional pair. `FmIndex` carries C's `occCnts_sb`/`occCnts_b`
        // sampled rank, so `occ` is O(FREQ_CNT_B) and the exhaustive `FM_Recurse`
        // traversal is tractable on genome-scale blocks.
        // Exact two-sweep SSV-over-FM augmentation, both strands. `record` is the
        // current strand's own record (kind=0 Watson, kind=1 Crick); the kernel
        // always needs the bi-directional pair fmf=kind0 (reversed text) +
        // fmb=kind1 (forward text). Extended diagonals are filtered by C's
        // `sc_thresh` inside the augment, so the kernel's extra seeds don't leak
        // through the lenient seed-then-rescore window gate.
        let sibling_kind = if is_complement { 0 } else { 1 };
        if let Some(sibling) = target_db
            .fm_index_records
            .iter()
            .find(|r| r.block_id == record.block_id && r.kind == sibling_kind)
        {
            let (fmf, fmb) = if is_complement {
                (&sibling.fm, &record.fm)
            } else {
                (&record.fm, &sibling.fm)
            };
            let mut augment_windows = Vec::new();
            let c_faithful = fm_ssv_augment_windows(
                fmf,
                fmb,
                record,
                &target_db.fm_ambiguities,
                seq_meta,
                seq,
                hmm,
                bg,
                config,
                is_complement,
                max_length,
                f1,
                desired_len,
                &mut augment_windows,
            );
            if c_faithful && !augment_windows.is_empty() {
                // The C-faithful augment alone reproduces C `p7_SSVFM_longlarget`'s
                // window list, so discard the Rust-only seed sources for this
                // block. (When the augment legitimately finds nothing — e.g. a
                // model too short for the FM seeding constraints — we keep the
                // Rust-only seeds rather than forcing an empty list, preserving
                // pre-existing behavior for those degenerate cases.)
                augment_c_faithful = true;
                windows = augment_windows;
            } else {
                windows.extend(augment_windows);
            }
        }
    }

    if windows.is_empty() {
        return None;
    }
    fm_finalize_seed_windows(windows, desired_len, seq, hmm, bg, augment_c_faithful)
}

/// Reduce the raw FM seed windows to the candidate-window list handed to the
/// long-target pipeline.
///
/// `c_faithful` is `true` when the windows came from the C-faithful FM-augment
/// (`FM_getSeeds` + `FM_extendSeed`); in that case the windows already match C
/// `p7_SSVFM_longlarget`'s extended-diagonal list, so the Rust-specific seed
/// pre-merge (`fm_merge_seed_windows`) and the lenient window gate
/// (`fm_extended_diagonal_passes_window_gate`) are skipped — the downstream
/// `extend_and_merge_windows_with_scoredata(.., 0.0)` does C's
/// `p7_pli_ExtendAndMergeWindows(.., 0)` merge, which makes the FM residue
/// counters match C. For the fallback seed-then-rescore path (`c_faithful ==
/// false`) those Rust-only guards stay in place to keep the hit set correct.
fn fm_finalize_seed_windows(
    windows: Vec<NhmmerScoredCandidateWindow>,
    desired_len: usize,
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    c_faithful: bool,
) -> Option<Vec<NhmmerCandidateWindow>> {
    let mut windows = if c_faithful {
        windows
    } else {
        fm_merge_seed_windows(windows, desired_len, seq, hmm, bg)
    };
    if !c_faithful {
        windows.retain(|scored| {
            fm_extended_diagonal_passes_window_gate(seq, hmm, bg, &scored.window, scored.score_bits)
        });
    }
    if windows.is_empty() {
        return None;
    }
    windows.sort_by(|a, b| {
        (a.window.n, a.window.length, a.window.k)
            .cmp(&(b.window.n, b.window.length, b.window.k))
            .then_with(|| b.score_bits.total_cmp(&a.score_bits))
    });
    let mut deduped = Vec::with_capacity(windows.len());
    for scored in windows {
        if deduped
            .last()
            .is_some_and(|prev: &NhmmerScoredCandidateWindow| prev.window == scored.window)
        {
            continue;
        }
        deduped.push(scored);
    }
    let mut windows = deduped;
    windows.sort_by(|a, b| {
        b.score_bits
            .total_cmp(&a.score_bits)
            .then_with(|| a.window.n.cmp(&b.window.n))
            .then_with(|| a.window.length.cmp(&b.window.length))
            .then_with(|| a.window.k.cmp(&b.window.k))
    });
    windows.sort_by_key(|w| (w.window.n, w.window.length, w.window.k));
    Some(windows.into_iter().map(|w| w.window).collect())
}

fn fm_merge_seed_windows(
    mut windows: Vec<NhmmerScoredCandidateWindow>,
    ssv_length: usize,
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
) -> Vec<NhmmerScoredCandidateWindow> {
    if windows.len() < 2 {
        return windows;
    }

    windows.sort_by(|a, b| {
        fm_window_diag_key(&a.window)
            .cmp(&fm_window_diag_key(&b.window))
            .then_with(|| a.window.n.cmp(&b.window.n))
            .then_with(|| fm_window_model_start(&a.window).cmp(&fm_window_model_start(&b.window)))
    });

    let mut merged = Vec::with_capacity(windows.len());
    let mut current = windows[0].window.clone();
    for scored in windows.into_iter().skip(1) {
        let next = scored.window;
        let current_diag = fm_window_diag_key(&current);
        let next_diag = fm_window_diag_key(&next);
        let current_start0 = current.n - 1;
        let current_model_start = fm_window_model_start(&current);
        let current_end0 = current.n - 1 + current.length - 1;
        let next_end0 = next.n - 1 + next.length - 1;
        if current_diag == next_diag
            && next.n - 1 + next.length < current.n - 1 + current.length + ssv_length
        {
            if next_end0 > current_end0 {
                current.length = next_end0 - current_start0 + 1;
                current.k = current_model_start + current.length - 1;
            }
        } else {
            merged.push(fm_rescore_seed_window(current, seq, hmm, bg));
            current = next;
        }
    }
    merged.push(fm_rescore_seed_window(current, seq, hmm, bg));
    merged
}

fn fm_window_model_start(window: &NhmmerCandidateWindow) -> usize {
    window.k.saturating_sub(window.length).saturating_add(1)
}

fn fm_window_diag_key(window: &NhmmerCandidateWindow) -> isize {
    (window.n as isize - 1) - fm_window_model_start(window) as isize
}

fn fm_rescore_seed_window(
    window: NhmmerCandidateWindow,
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
) -> NhmmerScoredCandidateWindow {
    let model_start = fm_window_model_start(&window);
    let score_bits = fm_diagonal_lod_bits(seq, hmm, bg, model_start, window.n - 1, window.length);
    NhmmerScoredCandidateWindow { window, score_bits }
}

fn fm_extended_diagonal_passes_window_gate(
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    window: &NhmmerCandidateWindow,
    score_bits: f32,
) -> bool {
    if !score_bits.is_finite() {
        return false;
    }
    if score_bits > NHMMER_FM_MIN_EXTENDED_DIAG_SCORE_BITS {
        return true;
    }
    !fm_diagonal_has_complete_lod(seq, hmm, bg, window)
}

fn fm_diagonal_has_complete_lod(
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    window: &NhmmerCandidateWindow,
) -> bool {
    let model_start = fm_window_model_start(window);
    (0..window.length).all(|offset| {
        fm_match_lod_bits(seq, hmm, bg, model_start + offset, window.n + offset).is_some()
    })
}

#[allow(clippy::too_many_arguments)]
fn fm_push_seed_window(
    windows: &mut Vec<NhmmerScoredCandidateWindow>,
    ambiguities: &[(usize, usize)],
    seq_meta: &NhmmerFmSequenceMeta,
    seq: &Sequence,
    record: &NhmmerFmIndexRecord,
    hmm: &Hmm,
    bg: &Bg,
    is_complement: bool,
    seed: &NhmmerConsensusSeed,
    fm_pos: usize,
    ssv_length: usize,
) {
    if fm_pos + seed.bases.len() > record.text_len {
        return;
    }
    let block_seed_start = if is_complement {
        fm_pos
    } else {
        record.text_len - fm_pos - seed.bases.len()
    };
    let Some(block_seed_end) = block_seed_start.checked_add(seed.bases.len()) else {
        return;
    };
    if fm_seed_is_wholly_in_leading_overlap(record, block_seed_end) {
        return;
    }
    if fm_seed_overlaps_ambiguity(
        ambiguities,
        record.ambig_offset,
        record.ambig_count,
        block_seed_start,
        block_seed_end,
    ) {
        return;
    }
    let forward_seq_seed_start = match block_seed_start.checked_sub(seq_meta.fm_start) {
        Some(pos) if pos + seed.bases.len() <= seq_meta.length => pos,
        _ => return,
    };
    let seq_seed_start = if is_complement {
        seq.n - forward_seq_seed_start - seed.bases.len()
    } else {
        forward_seq_seed_start
    };
    let (window_start0, window_len, model_end, score_bits) =
        fm_extend_seed_diagonal(seq, hmm, bg, seed, seq_seed_start, ssv_length);
    let window = NhmmerCandidateWindow {
        n: window_start0 + 1,
        length: window_len,
        k: model_end,
    };
    if !fm_extended_diagonal_passes_window_gate(seq, hmm, bg, &window, score_bits) {
        return;
    }
    windows.push(NhmmerScoredCandidateWindow { window, score_bits });
}

/// Emit a candidate window for a diagonal that has ALREADY been extended by the
/// faithful `fm_ssv::fm_extend_seed` (C `FM_extendSeed`), rather than re-running
/// the local sequence-space extension. This is the FM-augment path: the kernel
/// already produced an extended diagonal `(diag.n, diag.k, diag.length,
/// diag.score)` in C's coordinate/score frame, so we only do the FM→sequence
/// coordinate mapping (mirroring `fm_push_seed_window`'s leading-overlap and
/// ambiguity guards) and credit the kernel's own bit score. `seed_len` is the
/// extended diagonal length, `fm_pos` is `text_len - diag.n - diag.length`,
/// `model_end` is `diag.k + diag.length - 1`, and `score_bits` is `diag.score`.
#[allow(clippy::too_many_arguments)]
fn fm_push_extended_diag_window(
    windows: &mut Vec<NhmmerScoredCandidateWindow>,
    ambiguities: &[(usize, usize)],
    seq_meta: &NhmmerFmSequenceMeta,
    seq: &Sequence,
    record: &NhmmerFmIndexRecord,
    is_complement: bool,
    seed_len: usize,
    model_end: usize,
    fm_pos: usize,
    score_bits: f32,
) {
    if seed_len == 0 || fm_pos + seed_len > record.text_len {
        return;
    }
    let block_seed_start = if is_complement {
        fm_pos
    } else {
        record.text_len - fm_pos - seed_len
    };
    let Some(block_seed_end) = block_seed_start.checked_add(seed_len) else {
        return;
    };
    if fm_seed_is_wholly_in_leading_overlap(record, block_seed_end) {
        return;
    }
    if fm_seed_overlaps_ambiguity(
        ambiguities,
        record.ambig_offset,
        record.ambig_count,
        block_seed_start,
        block_seed_end,
    ) {
        return;
    }
    let forward_seq_seed_start = match block_seed_start.checked_sub(seq_meta.fm_start) {
        Some(pos) if pos + seed_len <= seq_meta.length => pos,
        _ => return,
    };
    let seq_seed_start = if is_complement {
        seq.n - forward_seq_seed_start - seed_len
    } else {
        forward_seq_seed_start
    };
    if !score_bits.is_finite() {
        return;
    }
    let window = NhmmerCandidateWindow {
        n: seq_seed_start + 1,
        length: seed_len,
        k: model_end,
    };
    windows.push(NhmmerScoredCandidateWindow { window, score_bits });
}

/// Augment the FM candidate windows with the exact two-sweep SSV-over-FM kernel
/// (`src/simd/fm_ssv.rs`), a faithful port of C `p7_SSVFM_longlarget` /
/// `FM_Recurse`. The seed-then-rescore loops above only find *exact* model
/// k-mers in the FM index; this walks the FM index accumulating SSV score over
/// the *actual* target characters, so it recovers weak diagonals with no exact
/// high-scoring k-mer match — the historical sensitivity gap vs C.
///
/// All score-valued inputs are in NATS (`ln(p/q)`; bit thresholds * ln2),
/// mirroring C's `ssv_scores_f` / `fm_cfg` units. The kernel only emits seed
/// *coordinates*; they are turned into windows by the existing validated
/// `fm_push_seed_window` (which re-scores in bits and whose downstream MSV/Vit/
/// Fwd gates filter false positives), so the unit choice stays isolated to the
/// kernel.
///
/// `is_complement` selects the strand. The caller passes `fmf` = the block's
/// kind=0 (reversed-text) index and `fmb` = its kind=1 (forward-text) index for
/// either strand, and `record` = the strand's own record (kind=0 for Watson,
/// kind=1 for Crick) used for the coordinate/overlap/ambiguity mapping exactly
/// as the seed-then-rescore path uses it. The Crick coordinate matches C's
/// complement `fm_getOriginalPosition` conversion (validated against bundled C).
///
/// To mirror C `p7_SSVFM_longlarget`, the extended diagonals are filtered by the
/// Gumbel-derived `sc_thresh` (computed from `max_length`, `f1`, and the model's
/// MSV evalue parameters) before becoming windows — without it, the exhaustive
/// kernel's extra raw seeds would survive the lenient seed-then-rescore `>0`
/// window gate and produce hits C never reports.
///
/// Returns `true` when the C-faithful `FM_extendSeed` extension path was taken
/// (single-segment block whose FM text frame we can reconstruct). In that mode
/// the augment alone reproduces C `p7_SSVFM_longlarget`'s window list, so the
/// caller drops the Rust-only seed sources and skips the seed pre-merge / window
/// gate (`fm_finalize_seed_windows`), matching C's residue counters. Returns
/// `false` for the fallback (seed-then-rescore) path, where those Rust-specific
/// guards stay in place.
#[allow(clippy::too_many_arguments)]
fn fm_ssv_augment_windows(
    fmf: &FmIndex,
    fmb: &FmIndex,
    record: &NhmmerFmIndexRecord,
    ambiguities: &[(usize, usize)],
    seq_meta: &NhmmerFmSequenceMeta,
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    config: NhmmerFmSeedConfig,
    is_complement: bool,
    max_length: i32,
    f1: f64,
    desired_len: usize,
    windows: &mut Vec<NhmmerScoredCandidateWindow>,
) -> bool {
    use crate::simd::fm_ssv;
    let m = hmm.m;
    if m == 0 {
        return false;
    }
    let kp = 4usize; // ACGT
    let ln2 = std::f32::consts::LN_2;

    // ssv_scores_f in NATS: ln(mat[k][x] / bg[x]), per model node k and base x.
    let mut scores = fm_ssv::SsvScores::new(m as i32, kp);
    let mut best_sc_sum = 0.0f64;
    for k in 1..=m {
        let mut best = 0.0f32;
        for x in 0..kp {
            let p = hmm
                .mat
                .get(k)
                .and_then(|r| r.get(x))
                .copied()
                .unwrap_or(0.0);
            let q = bg.f.get(x).copied().unwrap_or(0.0);
            let v = if p > 0.0 && q > 0.0 {
                ((p as f64) / (q as f64)).ln() as f32
            } else {
                -1.0e9
            };
            scores.set(k as i32, x, v);
            if v > best {
                best = v;
            }
        }
        best_sc_sum += best as f64;
    }

    // Consensus codes (0..3) from the model's consensus residues.
    let cons_bytes = nhmmer_consensus_bytes(hmm);
    let mut consensus = vec![0usize; m + 1];
    for (i, &b) in cons_bytes.iter().enumerate().take(m) {
        consensus[i + 1] = match b {
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 0,
        };
    }

    // sc_thresh_ratio (C nhmmer.c:996-1007): best_sc_avg = sum(max match score)/
    // sqrt(M) in nats, floored at 5; ratio = min(best_sc_avg/7, 1). Then
    // sc_threshFM = scthreshFM(14 bits) * ratio, in nats.
    let best_sc_avg = (best_sc_sum / (m as f64).sqrt()).max(5.0);
    let ratio = (best_sc_avg / 7.0).min(1.0) as f32;
    let sc_thresh_fm = NHMMER_FM_SEED_SCORE_THRESHOLD_BITS * ratio * ln2;

    let fm_config = fm_ssv::FmSsvConfig {
        max_depth: NHMMER_FM_SEED_MAX_DEPTH,
        drop_max_len: NHMMER_FM_SEED_DROP_MAX_LEN,
        drop_lim: NHMMER_FM_SEED_DROP_LIMIT_BITS * ln2,
        consec_pos_req: NHMMER_FM_SEED_CONSEC_POS_REQ,
        consensus_match_req: config.consensus_match_req,
        score_density_req: NHMMER_FM_SEED_SCORE_DENSITY_BITS * ln2,
        ssv_length: config.ssv_length.max(1),
    };

    // C `p7_SSVFM_longlarget` window threshold `sc_thresh`: the score an
    // *extended* SSV diagonal must reach to become a window. In nats:
    //   invP = gumbel_invsurv(F1, MMU, MLAMBDA)
    //   sc_thresh = invP*ln2 + nullsc - (tmove + tloop_total + tmove + tbmk + tec)
    // with the length model reconfigured to max_length (nu = 2.0). We compare it
    // against `fm_extend_seed_diagonal`'s bit score, so convert to bits.
    let sc_thresh_bits = {
        let ml = max_length.max(1) as f32;
        let tloop_total = (ml / (ml + 3.0)).ln() * ml;
        let tmove = (3.0 / (ml + 3.0)).ln();
        let tbmk = (2.0 / (m as f32 * (m as f32 + 1.0))).ln();
        let tec = (1.0f32 / 2.0).ln(); // nu = 2.0
        let mut bg_ml = bg.clone();
        bg_ml.set_length(ml as usize);
        let nullsc = bg_ml.null_one(ml as usize);
        // evparam[0]=P7_MMU, evparam[1]=P7_MLAMBDA.
        let mmu = hmm.evparam[p7hmm::P7_MMU] as f64;
        let mlambda = hmm.evparam[p7hmm::P7_MLAMBDA] as f64;
        let inv_p = hmmer_pure_rs::stats::gumbel::invsurv(f1, mmu, mlambda) as f32;
        let sc_thresh_nats = inv_p * ln2 + nullsc - (tmove + tloop_total + tmove + tbmk + tec);
        sc_thresh_nats / ln2
    };

    let want = if is_complement {
        fm_ssv::Complementarity::Complement
    } else {
        fm_ssv::Complementarity::NoComplement
    };
    let alph = fm_ssv::FmAlphabet::dna();
    let mut ssv_windows: Vec<NhmmerScoredCandidateWindow> = Vec::new();
    let diags = fm_ssv::fm_get_seeds(
        fmf,
        fmb,
        &alph,
        &scores,
        &consensus,
        sc_thresh_fm,
        &fm_config,
        /* top_only (Watson / NoComplement) */ !is_complement,
        /* bottom_only (Crick / Complement) */ is_complement,
    );

    // Build the strand-resolved FM-text codes that C `FM_extendSeed` extends
    // against via `fm_convertRange2DSQ(fm, ..., complementarity, ...)`. The kernel
    // diagonals' `diag.n` are in the FM coordinate frame (full-text 0-based);
    // `fm_extend_seed` reads `target[abs]` where `abs = target_start + offset`
    // (1-indexed, target[1] = FM position 0). For the single-segment DBs we
    // support, the record text covers `seq` exactly (fm_start == 0); for the
    // Watson (kind=0) frame the FM text is the forward sequence, and for the
    // Crick (kind=1) frame `fm_convertRange2DSQ` yields the reverse-complement,
    // which equals the revcomp of the forward sequence.
    let fm_text_codes: Option<Vec<usize>> = if seq_meta.fm_start == 0
        && seq_meta.length == seq.n
        && record.text_len >= seq.n
        && record.fm.n == record.text_len
    {
        let mut codes = vec![0usize; record.text_len + 2];
        if is_complement {
            // Crick: complement base at reversed position.
            for i in 0..seq.n {
                let fwd = seq.dsq[seq.n - i] as usize;
                codes[i + 1] = match fwd {
                    0 => 3,
                    1 => 2,
                    2 => 1,
                    3 => 0,
                    _ => 0,
                };
            }
        } else {
            for i in 0..seq.n {
                codes[i + 1] = seq.dsq[i + 1] as usize;
            }
        }
        Some(codes)
    } else {
        None
    };

    let c_faithful = fm_text_codes.is_some();
    let text_len = record.text_len as i64;
    // C's FM_DATA.N is the BWT length, including the terminal symbol. Rust
    // `FmIndex::n` stores only the biological text length, so use the serialized
    // BWT length for faithful FM_extendSeed coordinates.
    let fm_n = record.fm.bwt.len() as i64;
    for mut diag in diags {
        if diag.complementarity != want || diag.length <= 0 {
            continue;
        }
        if diag.k < 1 {
            continue;
        }

        if let Some(target) = fm_text_codes.as_ref() {
            // Faithful C `FM_extendSeed`: extend the raw kernel diagonal against
            // the FM text using the SSV (bit) score table, then window from the
            // extended coordinates. This reproduces C's window count/length —
            // the seed-then-rescore re-extension (`fm_extend_seed_diagonal`)
            // systematically produced fewer/shorter windows.
            fm_ssv::fm_extend_seed(&mut diag, target, &scores, &fm_config, fm_n);
            // `scores`/`diag.score` are in NATS; `sc_thresh_bits` is in BITS.
            // C compares the extended diagonal's score against `sc_thresh`
            // (nats), so convert the threshold to nats here. Store the window's
            // score in bits to match the seed-then-rescore path.
            if diag.length <= 0 || diag.score < sc_thresh_bits * ln2 {
                continue;
            }
            let model_end = (diag.k + diag.length - 1) as usize;
            if model_end == 0 || model_end > hmm.m {
                continue;
            }
            let fm_pos = text_len - diag.n - diag.length as i64;
            if fm_pos < 0 {
                continue;
            }
            fm_push_extended_diag_window(
                &mut ssv_windows,
                ambiguities,
                seq_meta,
                seq,
                record,
                is_complement,
                diag.length as usize,
                model_end,
                fm_pos as usize,
                diag.score / ln2,
            );
        } else {
            // Fallback (multi-segment / coordinate frame we don't reconstruct
            // here): the original seed-then-rescore window construction.
            let model_end = (diag.k + diag.length - 1) as usize;
            if model_end == 0 || model_end > hmm.m {
                continue;
            }
            let fm_pos = text_len - diag.n - diag.length as i64;
            if fm_pos < 0 {
                continue;
            }
            let synthetic = NhmmerConsensusSeed {
                bases: vec![0u8; diag.length as usize],
                model_end,
            };
            fm_push_seed_window(
                &mut ssv_windows,
                ambiguities,
                seq_meta,
                seq,
                record,
                hmm,
                bg,
                is_complement,
                &synthetic,
                fm_pos as usize,
                desired_len,
            );
        }
    }

    // C filters extended diagonals by `sc_thresh` before windowing. The
    // seed-then-rescore path's lenient `>0` gate (already applied inside
    // `fm_push_seed_window`) is not enough for the exhaustive kernel's extra
    // seeds, so apply the Gumbel-derived bit threshold here.
    ssv_windows.retain(|w| w.score_bits >= sc_thresh_bits);
    windows.extend(ssv_windows);
    c_faithful
}

struct NhmmerFmSeedQuery<'a> {
    fm_bases: Vec<u8>,
    seed: &'a NhmmerConsensusSeed,
    score_bits: f32,
}

fn fm_seed_queries_for_record<'a>(
    seeds: &'a [NhmmerConsensusSeed],
    record: &NhmmerFmIndexRecord,
    seq_len: usize,
    is_complement: bool,
    hmm: &Hmm,
    bg: &Bg,
) -> Vec<NhmmerFmSeedQuery<'a>> {
    let mut queries = Vec::with_capacity(seeds.len());
    for seed in seeds {
        if seed.bases.len() > record.text_len || seed.bases.len() > seq_len {
            continue;
        }
        let fm_bases = if is_complement {
            reverse_complement_dna_seed(&seed.bases)
        } else {
            let mut reversed_seed = seed.bases.clone();
            reversed_seed.reverse();
            reversed_seed
        };
        queries.push(NhmmerFmSeedQuery {
            fm_bases,
            seed,
            score_bits: fm_seed_model_lod_bits(hmm, bg, seed),
        });
    }
    queries
}

fn fm_locate_seed_queries<'a>(
    fm: &FmIndex,
    queries: &'a [NhmmerFmSeedQuery<'a>],
    limit: usize,
) -> Vec<(&'a NhmmerConsensusSeed, usize)> {
    if limit == 0 {
        return Vec::new();
    }
    let mut indexed_queries: Vec<usize> = (0..queries.len()).collect();
    indexed_queries.sort_by(|&a, &b| {
        queries[a]
            .fm_bases
            .iter()
            .rev()
            .cmp(queries[b].fm_bases.iter().rev())
            .then_with(|| queries[a].fm_bases.cmp(&queries[b].fm_bases))
    });

    let mut located = Vec::new();
    fm_locate_seed_queries_recurse(
        fm,
        queries,
        &indexed_queries,
        0,
        fm.root_interval(),
        &mut located,
        limit,
    );
    located
}

fn fm_locate_seed_queries_recurse<'a>(
    fm: &FmIndex,
    queries: &'a [NhmmerFmSeedQuery<'a>],
    query_ids: &[usize],
    depth: usize,
    interval: FmInterval,
    located: &mut Vec<(&'a NhmmerConsensusSeed, usize)>,
    limit: usize,
) {
    if located.len() >= limit {
        return;
    }
    let mut child_groups = Vec::new();
    let mut child_start = 0usize;
    while child_start < query_ids.len() {
        let query = &queries[query_ids[child_start]];
        if depth == query.fm_bases.len() {
            let mut terminal_end = child_start + 1;
            while terminal_end < query_ids.len()
                && depth == queries[query_ids[terminal_end]].fm_bases.len()
            {
                terminal_end += 1;
            }
            let mut terminals: Vec<usize> = query_ids[child_start..terminal_end].to_vec();
            terminals.sort_by(|&a, &b| {
                queries[b]
                    .score_bits
                    .total_cmp(&queries[a].score_bits)
                    .then_with(|| queries[a].seed.model_end.cmp(&queries[b].seed.model_end))
                    .then_with(|| queries[a].seed.bases.cmp(&queries[b].seed.bases))
            });
            let positions = fm.locate_interval(interval);
            for query_id in terminals {
                let terminal_query = &queries[query_id];
                for &fm_pos in &positions {
                    located.push((terminal_query.seed, fm_pos));
                    if located.len() >= limit {
                        return;
                    }
                }
            }
            child_start = terminal_end;
            continue;
        }

        let ch = query.fm_bases[query.fm_bases.len() - 1 - depth];
        let mut child_end = child_start + 1;
        while child_end < query_ids.len() {
            let child_query = &queries[query_ids[child_end]];
            if depth == child_query.fm_bases.len()
                || child_query.fm_bases[child_query.fm_bases.len() - 1 - depth] != ch
            {
                break;
            }
            child_end += 1;
        }

        let max_score = query_ids[child_start..child_end]
            .iter()
            .map(|&query_id| queries[query_id].score_bits)
            .fold(f32::NEG_INFINITY, f32::max);
        child_groups.push((child_start, child_end, ch, max_score));
        child_start = child_end;
    }

    child_groups.sort_by(|a, b| {
        b.3.total_cmp(&a.3)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });
    for (child_start, child_end, ch, _) in child_groups {
        if let Some(child_interval) = fm.prepend_interval(interval, ch) {
            fm_locate_seed_queries_recurse(
                fm,
                queries,
                &query_ids[child_start..child_end],
                depth + 1,
                child_interval,
                located,
                limit,
            );
        }
        if located.len() >= limit {
            return;
        }
    }
}

fn fm_extend_seed_diagonal(
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    seed: &NhmmerConsensusSeed,
    seq_seed_start0: usize,
    ssv_length: usize,
) -> (usize, usize, usize, f32) {
    let seed_len = seed.bases.len();
    let seed_model_start = seed.model_end.saturating_sub(seed_len).saturating_add(1);
    if seed_len == 0 || seed_model_start == 0 || seed.model_end > hmm.m {
        return (seq_seed_start0, seed_len, seed.model_end.min(hmm.m), 0.0);
    }

    let extend = 10usize.max(ssv_length.saturating_sub(seed_len));
    // NOTE: this is the seed-then-rescore approximation's extension window, NOT a
    // 1:1 port of C `FM_extendSeed`; the downstream window extension + MSV
    // rescore recompute final coordinates. Adding C's `model_start = k-extend+1`
    // "+ 1" here was tried and is observably neutral on the current fixtures, so
    // the simpler long-standing no-`+1` form is kept.
    let mut model_start = seed_model_start.saturating_sub(extend).max(1);
    let mut model_end = (seed_model_start + seed_len + extend - 1).min(hmm.m);
    let mut target_start = seq_seed_start0 as isize - (seed_model_start - model_start) as isize;
    let mut target_end = seq_seed_start0 as isize + (model_end - seed_model_start) as isize;

    if target_start < 0 {
        let shift = (-target_start) as usize;
        model_start = model_start.saturating_add(shift);
        target_start = 0;
    }
    let last_target = seq.n.saturating_sub(1) as isize;
    if target_end > last_target {
        let shift = (target_end - last_target) as usize;
        model_end = model_end.saturating_sub(shift);
        target_end = last_target;
    }
    if seq.n == 0 || model_start > model_end || target_start > target_end {
        return (seq_seed_start0, seed_len, seed.model_end, 0.0);
    }

    let mut score = 0.0_f32;
    let mut best_score = 0.0_f32;
    let mut hit_start = 0usize;
    let mut best_start = 0usize;
    let mut best_end = 0usize;
    let scan_len = (model_end - model_start + 1).min((target_end - target_start + 1) as usize);
    for offset in 0..scan_len {
        let model_pos = model_start + offset;
        let seq_pos = target_start as usize + offset + 1;
        let Some(delta) = fm_match_lod_bits(seq, hmm, bg, model_pos, seq_pos) else {
            score = 0.0;
            hit_start = offset + 1;
            continue;
        };
        score += delta;
        if score < 0.0 {
            score = 0.0;
            hit_start = offset + 1;
        } else if score > best_score {
            best_score = score;
            best_start = hit_start;
            best_end = offset;
        }
    }

    if best_score <= 0.0 {
        return (seq_seed_start0, seed_len, seed.model_end, best_score);
    }

    let window_start0 = target_start as usize + best_start;
    let window_len = best_end - best_start + 1;
    let model_end = model_start + best_end;
    (window_start0, window_len, model_end, best_score)
}

fn fm_diagonal_lod_bits(
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    model_start: usize,
    seq_start0: usize,
    len: usize,
) -> f32 {
    let mut score = 0.0_f32;
    for offset in 0..len {
        let Some(delta) =
            fm_match_lod_bits(seq, hmm, bg, model_start + offset, seq_start0 + 1 + offset)
        else {
            break;
        };
        score += delta;
    }
    score
}

fn fm_seed_model_lod_bits(hmm: &Hmm, bg: &Bg, seed: &NhmmerConsensusSeed) -> f32 {
    let seed_len = seed.bases.len();
    let seed_model_start = seed.model_end.saturating_sub(seed_len).saturating_add(1);
    if seed_len == 0 || seed_model_start == 0 || seed.model_end > hmm.m {
        return 0.0;
    }

    let abc = Alphabet::new(hmm.abc_type);
    let mut score = 0.0_f32;
    for (offset, &base) in seed.bases.iter().enumerate() {
        let Some(x) = abc.inmap.get(base as usize).copied() else {
            return score;
        };
        let x = x as usize;
        if x >= bg.f.len() {
            return score;
        }
        let model_pos = seed_model_start + offset;
        if model_pos == 0 || model_pos > hmm.m || x >= hmm.mat[model_pos].len() {
            return score;
        }
        let p = hmm.mat[model_pos][x];
        let q = bg.f[x];
        if p <= 0.0 || q <= 0.0 {
            return score;
        }
        score += ((p as f64) / (q as f64)).log2() as f32;
    }
    score
}

fn fm_match_lod_bits(
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    model_pos: usize,
    seq_pos: usize,
) -> Option<f32> {
    let x = *seq.dsq.get(seq_pos)? as usize;
    if model_pos == 0 || model_pos > hmm.m || x >= bg.f.len() || x >= hmm.mat[model_pos].len() {
        return None;
    }
    let p = hmm.mat[model_pos][x];
    let q = bg.f[x];
    (p > 0.0 && q > 0.0).then(|| ((p as f64) / (q as f64)).log2() as f32)
}

fn fm_seed_is_wholly_in_leading_overlap(record: &NhmmerFmIndexRecord, seed_end: usize) -> bool {
    record.overlap_bases > 0 && seed_end <= record.overlap_bases
}

fn fm_seed_overlaps_ambiguity(
    ambiguities: &[(usize, usize)],
    ambig_offset: usize,
    ambig_count: usize,
    seed_start: usize,
    seed_end: usize,
) -> bool {
    let Some(ambig_end) = ambig_offset.checked_add(ambig_count) else {
        return true;
    };
    if ambig_end > ambiguities.len() {
        return true;
    }
    for &(lower, upper) in &ambiguities[ambig_offset..ambig_end] {
        let ambiguity_end = upper.saturating_add(1);
        if seed_start < ambiguity_end && seed_end > lower {
            return true;
        }
    }
    false
}

fn reverse_complement_dna_seed(seed: &[u8]) -> Vec<u8> {
    seed.iter()
        .rev()
        .map(|&ch| match ch {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' | b'U' => b'A',
            _ => ch,
        })
        .collect()
}

fn complement_dna_base(ch: u8) -> u8 {
    match ch {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' | b'U' => b'A',
        _ => ch,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NhmmerConsensusSeed {
    bases: Vec<u8>,
    model_end: usize,
}

#[derive(Debug, Clone, Copy)]
struct NhmmerFmScoreState {
    score_bits: f32,
    max_score_bits: f32,
    score_peak_len: usize,
    consec_pos: usize,
    max_consec_pos: usize,
    consec_consensus: usize,
}

impl NhmmerFmScoreState {
    fn new() -> Self {
        Self {
            score_bits: 0.0,
            max_score_bits: 0.0,
            score_peak_len: 0,
            consec_pos: 0,
            max_consec_pos: 0,
            consec_consensus: 0,
        }
    }

    fn extend(
        self,
        delta_bits: f32,
        depth: usize,
        consensus_codes: &[Option<usize>],
        node: usize,
        code: usize,
        config: NhmmerFmSeedConfig,
    ) -> Self {
        let score_bits = self.score_bits + delta_bits;
        let consec_pos = if delta_bits > 0.0 {
            self.consec_pos + 1
        } else {
            0
        };
        let consec_consensus =
            nhmmer_next_consec_consensus(consensus_codes, node, code, self.consec_consensus);
        let (max_score_bits, score_peak_len) = if score_bits > self.max_score_bits {
            (score_bits, depth)
        } else if score_bits >= self.max_score_bits - config.drop_lim_bits {
            (self.max_score_bits, depth)
        } else {
            (self.max_score_bits, self.score_peak_len)
        };
        Self {
            score_bits,
            max_score_bits,
            score_peak_len,
            consec_pos,
            max_consec_pos: self.max_consec_pos.max(consec_pos),
            consec_consensus,
        }
    }

    fn reaches_seed(self, config: NhmmerFmSeedConfig) -> bool {
        self.score_bits >= config.score_threshold_bits
            || (config.consensus_match_req > 0
                && self.consec_consensus == config.consensus_match_req)
    }

    fn can_reach_consensus_seed(
        self,
        remaining_including_next: usize,
        config: NhmmerFmSeedConfig,
    ) -> bool {
        config.consensus_match_req > 0
            && self.consec_consensus + remaining_including_next >= config.consensus_match_req
    }

    fn should_prune_below_threshold(
        self,
        depth: usize,
        max_len: usize,
        config: NhmmerFmSeedConfig,
    ) -> bool {
        self.score_bits <= 0.0
            || depth == max_len
            || (config.drop_max_len > 0 && depth == self.score_peak_len + config.drop_max_len)
            || (self.score_bits < config.score_threshold_bits
                && depth > 4
                && depth > self.consec_consensus
                && self.score_bits / (depth as f32) < config.score_density_bits)
            || (config.consec_pos_req > 0
                && self.max_consec_pos < config.consec_pos_req
                && config.consec_pos_req.saturating_sub(self.consec_pos)
                    == max_len.saturating_sub(depth).saturating_add(1))
    }
}

fn nhmmer_seed_strings_with_config(
    hmm: &Hmm,
    bg: &Bg,
    config: NhmmerFmSeedConfig,
) -> Vec<NhmmerConsensusSeed> {
    let consensus = nhmmer_consensus_bytes(hmm);
    let seed_len = consensus.len().min(config.max_depth).max(1);
    let mut seeds = Vec::new();
    nhmmer_push_consensus_seeds(&consensus, seed_len, &mut seeds);

    // C's FM trie also accepts a seed when it accumulates
    // `--seed_consens_match` consecutive consensus matches (default 10), even
    // before the short diagonal reaches the score threshold. Add that concrete
    // seed tier so FM-index targets do not fall back solely because no exact
    // max-depth consensus seed occurs in the target.
    if config.consensus_match_req > 0 && seed_len > config.consensus_match_req {
        nhmmer_push_consensus_seeds(&consensus, config.consensus_match_req, &mut seeds);
    }
    nhmmer_push_score_threshold_seeds(hmm, bg, &mut seeds, config);
    seeds
}

fn nhmmer_push_consensus_seeds(
    consensus: &[u8],
    seed_len: usize,
    seeds: &mut Vec<NhmmerConsensusSeed>,
) {
    if seed_len == 0 {
        return;
    }
    for (start0, window) in consensus.windows(seed_len).enumerate() {
        if !window
            .iter()
            .all(|&ch| matches!(ch, b'A' | b'C' | b'G' | b'T'))
        {
            continue;
        }
        let model_end = start0 + seed_len;
        if !seeds
            .iter()
            .any(|seed| seed.model_end == model_end && seed.bases == window)
        {
            seeds.push(NhmmerConsensusSeed {
                bases: window.to_vec(),
                model_end,
            });
            if seeds.len() >= 64 {
                break;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fm_locate_model_consensus_seeds(
    fm: &FmIndex,
    hmm: &Hmm,
    bg: &Bg,
    is_complement: bool,
    text_len: usize,
    seq_len: usize,
    limit: usize,
    config: NhmmerFmSeedConfig,
) -> Vec<(NhmmerConsensusSeed, usize)> {
    let consensus = nhmmer_consensus_bytes(hmm);
    if consensus.is_empty() || text_len == 0 || seq_len == 0 || limit == 0 {
        return Vec::new();
    }

    let full_seed_len = consensus
        .len()
        .min(config.max_depth)
        .min(text_len)
        .min(seq_len);
    let short_seed_len = if config.consensus_match_req > 0 {
        config
            .consensus_match_req
            .min(consensus.len())
            .min(text_len)
            .min(seq_len)
    } else {
        full_seed_len
    };
    if full_seed_len == 0 {
        return Vec::new();
    }

    let mut located = Vec::new();
    let mut starts: Vec<(usize, f32)> = (0..consensus.len())
        .map(|start0| {
            let seed_len = full_seed_len.min(consensus.len() - start0);
            (
                start0,
                fm_consensus_seed_model_lod_bits(&consensus, start0, seed_len, hmm, bg),
            )
        })
        .collect();
    starts.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (start0, _) in starts {
        if located.len() >= limit {
            break;
        }
        let remaining = consensus.len() - start0;
        let max_len = full_seed_len.min(remaining);
        if max_len == 0 {
            continue;
        }
        fm_model_consensus_seed_dfs(
            fm,
            &consensus,
            start0,
            max_len,
            full_seed_len,
            short_seed_len,
            fm.root_interval(),
            is_complement,
            &mut Vec::with_capacity(max_len),
            &mut located,
            limit,
        );
    }
    located
}

fn fm_consensus_seed_model_lod_bits(
    consensus: &[u8],
    start0: usize,
    len: usize,
    hmm: &Hmm,
    bg: &Bg,
) -> f32 {
    let Some(seed_end) = start0.checked_add(len) else {
        return f32::NEG_INFINITY;
    };
    if seed_end > consensus.len() || seed_end > hmm.m {
        return f32::NEG_INFINITY;
    }
    let seed = NhmmerConsensusSeed {
        bases: consensus[start0..seed_end].to_vec(),
        model_end: seed_end,
    };
    fm_seed_model_lod_bits(hmm, bg, &seed)
}

#[allow(clippy::too_many_arguments)]
fn fm_model_consensus_seed_dfs(
    fm: &FmIndex,
    consensus: &[u8],
    start0: usize,
    max_len: usize,
    full_seed_len: usize,
    short_seed_len: usize,
    interval: FmInterval,
    is_complement: bool,
    current: &mut Vec<u8>,
    located: &mut Vec<(NhmmerConsensusSeed, usize)>,
    limit: usize,
) {
    if current.len() == max_len || located.len() >= limit {
        return;
    }

    let node0 = start0 + current.len();
    let ch = consensus[node0];
    if !matches!(ch, b'A' | b'C' | b'G' | b'T') {
        return;
    }
    let fm_ch = if is_complement {
        complement_dna_base(ch)
    } else {
        ch
    };
    let Some(child_interval) = fm.prepend_interval(interval, fm_ch) else {
        return;
    };

    current.push(ch);
    let len = current.len();
    if len == full_seed_len || (full_seed_len > short_seed_len && len == short_seed_len) {
        let seed = NhmmerConsensusSeed {
            bases: current.clone(),
            model_end: start0 + len,
        };
        for fm_pos in fm.locate_interval(child_interval) {
            located.push((seed.clone(), fm_pos));
            if located.len() >= limit {
                current.pop();
                return;
            }
        }
    }

    fm_model_consensus_seed_dfs(
        fm,
        consensus,
        start0,
        max_len,
        full_seed_len,
        short_seed_len,
        child_interval,
        is_complement,
        current,
        located,
        limit,
    );
    current.pop();
}

fn nhmmer_push_score_threshold_seeds(
    hmm: &Hmm,
    bg: &Bg,
    seeds: &mut Vec<NhmmerConsensusSeed>,
    config: NhmmerFmSeedConfig,
) {
    let Some((abc, max_depth, lod_bits, best_suffix, best_prefix, consensus_codes)) =
        nhmmer_fm_lod_tables(hmm, bg, config)
    else {
        return;
    };

    let before = seeds.len();
    let seed_limit = before + NHMMER_FM_SCORE_SEED_LIMIT;
    let mut starts: Vec<(bool, usize, f32)> = (1..=hmm.m)
        .filter_map(|start| {
            let max_len = max_depth.min(hmm.m - start + 1);
            (max_len > 0).then_some((false, start, best_suffix[start][max_len]))
        })
        .collect();
    starts.extend((1..=hmm.m).filter_map(|end| {
        let max_len = max_depth.min(end);
        (max_len > 0).then_some((true, end, best_prefix[end][max_len]))
    }));
    starts.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });

    for (reverse, anchor, _) in starts {
        if seeds.len() >= before + NHMMER_FM_SCORE_SEED_LIMIT {
            break;
        }
        if reverse {
            let max_len = max_depth.min(anchor);
            let mut current_rev = Vec::with_capacity(max_len);
            nhmmer_score_seed_reverse_dfs(
                anchor,
                anchor,
                max_len,
                &mut current_rev,
                &lod_bits,
                &best_prefix,
                &consensus_codes,
                &abc,
                seeds,
                seed_limit,
                NhmmerFmScoreState::new(),
                config,
            );
        } else {
            let max_len = max_depth.min(hmm.m - anchor + 1);
            let mut current = Vec::with_capacity(max_len);
            nhmmer_score_seed_dfs(
                anchor,
                anchor,
                max_len,
                &mut current,
                &lod_bits,
                &best_suffix,
                &consensus_codes,
                &abc,
                seeds,
                seed_limit,
                NhmmerFmScoreState::new(),
                config,
            );
        }
    }
}

#[allow(clippy::type_complexity)]
fn nhmmer_fm_lod_tables(
    hmm: &Hmm,
    bg: &Bg,
    config: NhmmerFmSeedConfig,
) -> Option<(
    Alphabet,
    usize,
    Vec<[f32; 4]>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Option<usize>>,
)> {
    if !matches!(hmm.abc_type, AlphabetType::Dna | AlphabetType::Rna) || hmm.m == 0 {
        return None;
    }

    let abc = Alphabet::new(hmm.abc_type);
    let max_depth = hmm.m.min(config.max_depth);
    if max_depth == 0 {
        return None;
    }

    let mut lod_bits = vec![[f32::NEG_INFINITY; 4]; hmm.m + 1];
    let mut best_suffix = vec![vec![0.0_f32; max_depth + 1]; hmm.m + 2];
    let mut best_prefix = vec![vec![0.0_f32; max_depth + 1]; hmm.m + 1];
    for (node, lod_row) in lod_bits.iter_mut().enumerate().take(hmm.m + 1).skip(1) {
        for (x, lod_cell) in lod_row.iter_mut().enumerate().take(abc.k.min(4)) {
            if hmm.mat[node][x] > 0.0 && bg.f[x] > 0.0 {
                *lod_cell = ((hmm.mat[node][x] as f64) / (bg.f[x] as f64)).log2() as f32;
            }
        }
    }
    for node in (1..=hmm.m).rev() {
        let best_here = lod_bits[node]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0);
        for len in 1..=max_depth {
            best_suffix[node][len] = best_here + best_suffix[node + 1][len - 1];
        }
    }
    for node in 1..=hmm.m {
        let best_here = lod_bits[node]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0);
        for len in 1..=max_depth {
            best_prefix[node][len] = best_here + best_prefix[node - 1][len - 1];
        }
    }

    let consensus_codes = nhmmer_fm_consensus_codes(hmm, &abc);

    Some((
        abc,
        max_depth,
        lod_bits,
        best_suffix,
        best_prefix,
        consensus_codes,
    ))
}

#[cfg(test)]
fn fm_locate_model_score_seeds(
    fm: &FmIndex,
    hmm: &Hmm,
    bg: &Bg,
    is_complement: bool,
    text_len: usize,
    seq_len: usize,
    limit: usize,
) -> Vec<(NhmmerConsensusSeed, usize)> {
    fm_locate_model_score_seeds_with_start_limit(
        fm,
        hmm,
        bg,
        is_complement,
        text_len,
        seq_len,
        limit,
        usize::MAX,
        NhmmerFmSeedConfig::default(),
    )
}

#[allow(clippy::too_many_arguments)]
fn fm_locate_model_score_seeds_with_start_limit(
    fm: &FmIndex,
    hmm: &Hmm,
    bg: &Bg,
    is_complement: bool,
    text_len: usize,
    seq_len: usize,
    limit: usize,
    start_limit: usize,
    config: NhmmerFmSeedConfig,
) -> Vec<(NhmmerConsensusSeed, usize)> {
    let Some((abc, max_depth, lod_bits, best_suffix, best_prefix, consensus_codes)) =
        nhmmer_fm_lod_tables(hmm, bg, config)
    else {
        return Vec::new();
    };
    if text_len == 0 || seq_len == 0 || limit == 0 {
        return Vec::new();
    }

    let mut located = Vec::new();
    let mut starts: Vec<(bool, usize, f32)> = (1..=hmm.m)
        .filter_map(|start| {
            let max_len = max_depth.min(hmm.m - start + 1).min(text_len).min(seq_len);
            (max_len > 0).then_some((false, start, best_suffix[start][max_len]))
        })
        .collect();
    starts.extend((1..=hmm.m).filter_map(|end| {
        let max_len = max_depth.min(end).min(text_len).min(seq_len);
        (max_len > 0).then_some((true, end, best_prefix[end][max_len]))
    }));
    starts.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });
    starts.truncate(start_limit);
    for (reverse, anchor, _) in starts {
        if located.len() >= limit {
            break;
        }
        if reverse {
            let max_len = max_depth.min(anchor).min(text_len).min(seq_len);
            let mut current_rev = Vec::with_capacity(max_len);
            fm_model_score_seed_reverse_dfs(
                fm,
                anchor,
                anchor,
                max_len,
                is_complement,
                &mut current_rev,
                &lod_bits,
                &best_prefix,
                &consensus_codes,
                &abc,
                &mut located,
                limit,
                NhmmerFmScoreState::new(),
                config,
            );
        } else {
            let max_len = max_depth.min(hmm.m - anchor + 1).min(text_len).min(seq_len);
            let mut current = Vec::with_capacity(max_len);
            fm_model_score_seed_dfs(
                fm,
                anchor,
                anchor,
                max_len,
                fm.root_interval(),
                is_complement,
                &mut current,
                &lod_bits,
                &best_suffix,
                &consensus_codes,
                &abc,
                &mut located,
                limit,
                NhmmerFmScoreState::new(),
                config,
            );
        }
    }
    located
}

#[allow(clippy::too_many_arguments)]
fn fm_model_score_seed_dfs(
    fm: &FmIndex,
    start: usize,
    node: usize,
    max_len: usize,
    interval: FmInterval,
    is_complement: bool,
    current: &mut Vec<u8>,
    lod_bits: &[[f32; 4]],
    best_suffix: &[Vec<f32>],
    consensus_codes: &[Option<usize>],
    abc: &Alphabet,
    located: &mut Vec<(NhmmerConsensusSeed, usize)>,
    limit: usize,
    state: NhmmerFmScoreState,
    config: NhmmerFmSeedConfig,
) {
    if current.len() == max_len || located.len() >= limit {
        return;
    }

    let depth = current.len() + 1;
    let remaining_after = max_len - depth;
    if state.score_bits + best_suffix[node][remaining_after + 1] < config.score_threshold_bits
        && !state.can_reach_consensus_seed(remaining_after + 1, config)
    {
        return;
    }

    let mut choices: Vec<(usize, f32)> = (0..abc.k.min(4))
        .filter_map(|x| {
            let sc = lod_bits[node][x];
            sc.is_finite().then_some((x, sc))
        })
        .collect();
    choices.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    for (x, delta) in choices {
        let next_state = state.extend(delta, depth, consensus_codes, node, x, config);
        if next_state.should_prune_below_threshold(depth, max_len, config)
            && !next_state.reaches_seed(config)
        {
            continue;
        }
        let tail_len = max_len - depth;
        if next_state.score_bits + best_suffix[node + 1][tail_len] < config.score_threshold_bits
            && !next_state.can_reach_consensus_seed(tail_len, config)
        {
            continue;
        }

        let ch = match abc.sym[x].to_ascii_uppercase() {
            b'U' => b'T',
            ch => ch,
        };
        let fm_ch = if is_complement {
            complement_dna_base(ch)
        } else {
            ch
        };
        let Some(child_interval) = fm.prepend_interval(interval, fm_ch) else {
            continue;
        };

        current.push(ch);
        let len = current.len();
        if next_state.reaches_seed(config) {
            let seed = NhmmerConsensusSeed {
                bases: current.clone(),
                model_end: start + len - 1,
            };
            for fm_pos in fm.locate_interval(child_interval) {
                located.push((seed.clone(), fm_pos));
                if located.len() >= limit {
                    current.pop();
                    return;
                }
            }
        }

        fm_model_score_seed_dfs(
            fm,
            start,
            node + 1,
            max_len,
            child_interval,
            is_complement,
            current,
            lod_bits,
            best_suffix,
            consensus_codes,
            abc,
            located,
            limit,
            next_state,
            config,
        );
        current.pop();
    }
}

#[allow(clippy::too_many_arguments)]
fn fm_model_score_seed_reverse_dfs(
    fm: &FmIndex,
    end: usize,
    node: usize,
    max_len: usize,
    is_complement: bool,
    current_rev: &mut Vec<u8>,
    lod_bits: &[[f32; 4]],
    best_prefix: &[Vec<f32>],
    consensus_codes: &[Option<usize>],
    abc: &Alphabet,
    located: &mut Vec<(NhmmerConsensusSeed, usize)>,
    limit: usize,
    state: NhmmerFmScoreState,
    config: NhmmerFmSeedConfig,
) {
    if current_rev.len() == max_len || located.len() >= limit || node == 0 {
        return;
    }

    let depth = current_rev.len() + 1;
    let remaining_after = max_len - depth;
    if state.score_bits + best_prefix[node][remaining_after + 1] < config.score_threshold_bits
        && !state.can_reach_consensus_seed(remaining_after + 1, config)
    {
        return;
    }

    let mut choices: Vec<(usize, f32)> = (0..abc.k.min(4))
        .filter_map(|x| {
            let sc = lod_bits[node][x];
            sc.is_finite().then_some((x, sc))
        })
        .collect();
    choices.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    for (x, delta) in choices {
        let next_state = state.extend(delta, depth, consensus_codes, node, x, config);
        if next_state.should_prune_below_threshold(depth, max_len, config)
            && !next_state.reaches_seed(config)
        {
            continue;
        }
        let tail_len = max_len - depth;
        if next_state.score_bits + best_prefix[node - 1][tail_len] < config.score_threshold_bits
            && !next_state.can_reach_consensus_seed(tail_len, config)
        {
            continue;
        }

        let ch = match abc.sym[x].to_ascii_uppercase() {
            b'U' => b'T',
            ch => ch,
        };
        current_rev.push(ch);
        let Some(child_interval) =
            fm_interval_for_reverse_model_prefix(fm, current_rev, is_complement)
        else {
            current_rev.pop();
            continue;
        };

        if next_state.reaches_seed(config) {
            let seed = NhmmerConsensusSeed {
                bases: current_rev.iter().rev().copied().collect(),
                model_end: end,
            };
            for fm_pos in fm.locate_interval(child_interval) {
                located.push((seed.clone(), fm_pos));
                if located.len() >= limit {
                    current_rev.pop();
                    return;
                }
            }
        }

        fm_model_score_seed_reverse_dfs(
            fm,
            end,
            node - 1,
            max_len,
            is_complement,
            current_rev,
            lod_bits,
            best_prefix,
            consensus_codes,
            abc,
            located,
            limit,
            next_state,
            config,
        );
        current_rev.pop();
    }
}

#[cfg(test)]
fn fm_interval_for_model_seed(
    fm: &FmIndex,
    bases: &[u8],
    is_complement: bool,
) -> Option<FmInterval> {
    let mut interval = fm.root_interval();
    for &ch in bases {
        let fm_ch = if is_complement {
            complement_dna_base(ch)
        } else {
            ch
        };
        interval = fm.prepend_interval(interval, fm_ch)?;
    }
    Some(interval)
}

fn fm_interval_for_reverse_model_prefix(
    fm: &FmIndex,
    current_rev: &[u8],
    is_complement: bool,
) -> Option<FmInterval> {
    let mut interval = fm.root_interval();
    for &ch in current_rev.iter().rev() {
        let fm_ch = if is_complement {
            complement_dna_base(ch)
        } else {
            ch
        };
        interval = fm.prepend_interval(interval, fm_ch)?;
    }
    Some(interval)
}

#[allow(clippy::too_many_arguments)]
fn nhmmer_score_seed_dfs(
    start: usize,
    node: usize,
    max_len: usize,
    current: &mut Vec<u8>,
    lod_bits: &[[f32; 4]],
    best_suffix: &[Vec<f32>],
    consensus_codes: &[Option<usize>],
    abc: &Alphabet,
    seeds: &mut Vec<NhmmerConsensusSeed>,
    seed_limit: usize,
    state: NhmmerFmScoreState,
    config: NhmmerFmSeedConfig,
) {
    if current.len() == max_len || seeds.len() >= seed_limit {
        return;
    }

    let depth = current.len() + 1;
    let remaining_after = max_len - depth;
    if state.score_bits + best_suffix[node][remaining_after + 1] < config.score_threshold_bits
        && !state.can_reach_consensus_seed(remaining_after + 1, config)
    {
        return;
    }

    let mut choices: Vec<(usize, f32)> = (0..abc.k.min(4))
        .filter_map(|x| {
            let sc = lod_bits[node][x];
            sc.is_finite().then_some((x, sc))
        })
        .collect();
    choices.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    for (x, delta) in choices {
        let next_state = state.extend(delta, depth, consensus_codes, node, x, config);
        if next_state.should_prune_below_threshold(depth, max_len, config)
            && !next_state.reaches_seed(config)
        {
            continue;
        }
        let tail_len = max_len - depth;
        if next_state.score_bits + best_suffix[node + 1][tail_len] < config.score_threshold_bits
            && !next_state.can_reach_consensus_seed(tail_len, config)
        {
            continue;
        }

        let ch = match abc.sym[x].to_ascii_uppercase() {
            b'U' => b'T',
            ch => ch,
        };
        current.push(ch);

        let len = current.len();
        if next_state.reaches_seed(config) {
            let model_end = start + len - 1;
            let seed = NhmmerConsensusSeed {
                bases: current.clone(),
                model_end,
            };
            if nhmmer_push_unique_score_seed(seeds, seed, seed_limit) {
                current.pop();
                return;
            }
        }

        nhmmer_score_seed_dfs(
            start,
            node + 1,
            max_len,
            current,
            lod_bits,
            best_suffix,
            consensus_codes,
            abc,
            seeds,
            seed_limit,
            next_state,
            config,
        );
        current.pop();
    }
}

#[allow(clippy::too_many_arguments)]
fn nhmmer_score_seed_reverse_dfs(
    end: usize,
    node: usize,
    max_len: usize,
    current_rev: &mut Vec<u8>,
    lod_bits: &[[f32; 4]],
    best_prefix: &[Vec<f32>],
    consensus_codes: &[Option<usize>],
    abc: &Alphabet,
    seeds: &mut Vec<NhmmerConsensusSeed>,
    seed_limit: usize,
    state: NhmmerFmScoreState,
    config: NhmmerFmSeedConfig,
) {
    if current_rev.len() == max_len || seeds.len() >= seed_limit || node == 0 {
        return;
    }

    let depth = current_rev.len() + 1;
    let remaining_after = max_len - depth;
    if state.score_bits + best_prefix[node][remaining_after + 1] < config.score_threshold_bits
        && !state.can_reach_consensus_seed(remaining_after + 1, config)
    {
        return;
    }

    let mut choices: Vec<(usize, f32)> = (0..abc.k.min(4))
        .filter_map(|x| {
            let sc = lod_bits[node][x];
            sc.is_finite().then_some((x, sc))
        })
        .collect();
    choices.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    for (x, delta) in choices {
        let next_state = state.extend(delta, depth, consensus_codes, node, x, config);
        if next_state.should_prune_below_threshold(depth, max_len, config)
            && !next_state.reaches_seed(config)
        {
            continue;
        }
        let tail_len = max_len - depth;
        if next_state.score_bits + best_prefix[node - 1][tail_len] < config.score_threshold_bits
            && !next_state.can_reach_consensus_seed(tail_len, config)
        {
            continue;
        }

        let ch = match abc.sym[x].to_ascii_uppercase() {
            b'U' => b'T',
            ch => ch,
        };
        current_rev.push(ch);

        if next_state.reaches_seed(config) {
            let seed = NhmmerConsensusSeed {
                bases: current_rev.iter().rev().copied().collect(),
                model_end: end,
            };
            if nhmmer_push_unique_score_seed(seeds, seed, seed_limit) {
                current_rev.pop();
                return;
            }
        }

        nhmmer_score_seed_reverse_dfs(
            end,
            node - 1,
            max_len,
            current_rev,
            lod_bits,
            best_prefix,
            consensus_codes,
            abc,
            seeds,
            seed_limit,
            next_state,
            config,
        );
        current_rev.pop();
    }
}

fn nhmmer_fm_consensus_codes(hmm: &Hmm, abc: &Alphabet) -> Vec<Option<usize>> {
    let consensus = nhmmer_consensus_bytes(hmm);
    let mut codes = vec![None; hmm.m + 1];
    for node in 1..=hmm.m.min(consensus.len()) {
        let ch = match consensus[node - 1].to_ascii_uppercase() {
            b'U' => b'T',
            ch => ch,
        };
        let code = abc.inmap[ch as usize];
        if code < abc.k as u8 && (code as usize) < 4 {
            codes[node] = Some(code as usize);
        }
    }
    codes
}

fn nhmmer_next_consec_consensus(
    consensus_codes: &[Option<usize>],
    node: usize,
    x: usize,
    previous: usize,
) -> usize {
    if consensus_codes.get(node).copied().flatten() == Some(x) {
        previous + 1
    } else {
        0
    }
}

fn nhmmer_push_unique_score_seed(
    seeds: &mut Vec<NhmmerConsensusSeed>,
    seed: NhmmerConsensusSeed,
    seed_limit: usize,
) -> bool {
    if !seeds.iter().any(|existing| {
        existing.model_end == seed.model_end && existing.bases.as_slice() == seed.bases.as_slice()
    }) {
        seeds.push(seed);
    }
    seeds.len() >= seed_limit
}

fn nhmmer_consensus_bytes(hmm: &Hmm) -> Vec<u8> {
    if let Some(consensus) = hmm.consensus.as_ref() {
        let bytes: Vec<u8> = consensus
            .iter()
            .skip(1)
            .take(hmm.m)
            .map(|&ch| ch.to_ascii_uppercase())
            .map(|ch| if ch == b'U' { b'T' } else { ch })
            .collect();
        if bytes
            .iter()
            .any(|&ch| matches!(ch, b'A' | b'C' | b'G' | b'T'))
        {
            return bytes;
        }
    }

    let abc = Alphabet::new(hmm.abc_type);
    (1..=hmm.m)
        .map(|node| {
            let best = hmm.mat[node]
                .iter()
                .take(abc.k)
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let ch = abc.sym[best].to_ascii_uppercase();
            if ch == b'U' {
                b'T'
            } else {
                ch
            }
        })
        .collect()
}

#[derive(Debug)]
struct NhmmerFmRecord {
    seq_offset: usize,
    seq_count: usize,
    ambig_offset: usize,
    ambig_count: usize,
    overlap_bases: usize,
    text_bases_len: usize,
    text: Vec<u8>,
}

#[allow(clippy::type_complexity)]
fn read_makehmmerdb_c_metadata_payload(
    cursor: &mut Cursor<&[u8]>,
) -> Result<
    (
        bool,
        usize,
        usize,
        usize,
        Vec<NhmmerFmSequenceMeta>,
        Vec<(usize, usize)>,
    ),
    String,
> {
    let fwd_only = read_u8(cursor)? != 0;
    let alph_type = read_u8(cursor)?;
    let alph_size = read_u8(cursor)?;
    let char_bits = read_u8(cursor)?;
    if (alph_type, alph_size, char_bits) != (0, 4, 2) {
        return Err("makehmmerdb FM-index is not a DNA 2-bit database".to_string());
    }
    let freq_sa = read_u32(cursor)? as usize;
    let _freq_cnt_sb = read_u32(cursor)? as usize;
    let freq_cnt_b = read_u32(cursor)? as usize;
    if freq_sa == 0 {
        return Err("makehmmerdb FM-index suffix-array sampling frequency is zero".to_string());
    }
    if freq_cnt_b == 0 {
        return Err("makehmmerdb FM-index occurrence sampling frequency is zero".to_string());
    }
    let block_count = read_u16(cursor)? as usize;
    let seq_count = read_u32(cursor)? as usize;
    let ambig_count = read_u32(cursor)? as usize;
    let _char_count = u64_to_usize(read_u64(cursor)?, "makehmmerdb metadata character count")?;

    let sequence_bytes = checked_count_bytes(
        seq_count,
        MAKEHMMERDB_C_MIN_SEQUENCE_META_BYTES,
        "makehmmerdb sequence metadata",
    )?;
    let ambiguity_bytes = checked_count_bytes(
        ambig_count,
        MAKEHMMERDB_C_AMBIGUITY_BYTES,
        "makehmmerdb ambiguity metadata",
    )?;
    let min_payload_bytes = sequence_bytes
        .checked_add(ambiguity_bytes)
        .ok_or_else(|| "makehmmerdb metadata payload byte span overflows usize".to_string())?;
    if makehmmerdb_remaining(cursor) < min_payload_bytes {
        return Err("makehmmerdb metadata payload is truncated".to_string());
    }

    let mut sequences = Vec::with_capacity(seq_count);
    for _ in 0..seq_count {
        let _target_id = read_u32(cursor)?;
        let _target_start = read_u64(cursor)?;
        let fm_start = read_u32(cursor)? as usize;
        let length = read_u32(cursor)? as usize;
        let name_len = read_u16(cursor)? as usize;
        let acc_len = read_u16(cursor)? as usize;
        let source_len = read_u16(cursor)? as usize;
        let desc_len = read_u16(cursor)? as usize;
        let name = read_nul_string(cursor, name_len)?;
        let acc = read_nul_string(cursor, acc_len)?;
        let _source = read_nul_string(cursor, source_len)?;
        let desc = read_nul_string(cursor, desc_len)?;
        sequences.push(NhmmerFmSequenceMeta {
            name,
            acc,
            desc,
            fm_start,
            length,
        });
    }
    ensure_makehmmerdb_remaining(
        cursor,
        ambig_count,
        MAKEHMMERDB_C_AMBIGUITY_BYTES,
        "makehmmerdb ambiguity metadata",
    )?;
    let mut ambiguities = Vec::with_capacity(ambig_count);
    for _ in 0..ambig_count {
        let lower = read_i32(cursor)?;
        let upper = read_i32(cursor)?;
        if lower < 0 || upper < lower {
            return Err("makehmmerdb ambiguity metadata has invalid coordinates".to_string());
        }
        ambiguities.push((lower as usize, upper as usize));
    }
    Ok((
        fwd_only,
        freq_sa,
        freq_cnt_b,
        block_count,
        sequences,
        ambiguities,
    ))
}

fn read_makehmmerdb_c_fm_record(
    cursor: &mut Cursor<&[u8]>,
    freq_sa: usize,
    freq_cnt_b: usize,
    has_text_and_sa: bool,
) -> Result<NhmmerFmRecord, String> {
    if freq_sa == 0 {
        return Err("makehmmerdb FM-index suffix-array sampling frequency is zero".to_string());
    }
    if freq_cnt_b == 0 {
        return Err("makehmmerdb FM-index occurrence sampling frequency is zero".to_string());
    }
    let n = u64_to_usize(read_u64(cursor)?, "makehmmerdb FM-index record length")?;
    let _term_loc = read_u32(cursor)?;
    let seq_offset = read_u32(cursor)? as usize;
    let ambig_offset = read_u32(cursor)? as usize;
    let overlap_bases = read_u32(cursor)? as usize;
    let seq_count = read_u32(cursor)? as usize;
    let ambig_count = read_u32(cursor)? as usize;

    let mut text = Vec::new();
    let packed_len = n.div_ceil(4);
    let text_bytes = if has_text_and_sa { packed_len } else { 0 };
    let bwt_bytes = packed_len;
    let sa_count = if has_text_and_sa {
        1usize
            .checked_add(n / freq_sa)
            .ok_or_else(|| "makehmmerdb FM-index suffix-array span overflows usize".to_string())?
    } else {
        0
    };
    let sa_bytes = checked_count_bytes(sa_count, 4, "makehmmerdb FM-index suffix array")?;
    let occ_b_count = 1usize
        .checked_add(n.div_ceil(freq_cnt_b))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| "makehmmerdb FM-index occurrence table span overflows usize".to_string())?;
    let occ_b_bytes = checked_count_bytes(occ_b_count, 2, "makehmmerdb FM-index occurrence table")?;
    let occ_sb_count = 1usize
        .checked_add(n.div_ceil(65_536))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| {
            "makehmmerdb FM-index superblock occurrence table span overflows usize".to_string()
        })?;
    let occ_sb_bytes = checked_count_bytes(
        occ_sb_count,
        4,
        "makehmmerdb FM-index superblock occurrence table",
    )?;
    let record_payload_bytes = text_bytes
        .checked_add(bwt_bytes)
        .and_then(|value| value.checked_add(sa_bytes))
        .and_then(|value| value.checked_add(occ_b_bytes))
        .and_then(|value| value.checked_add(occ_sb_bytes))
        .ok_or_else(|| "makehmmerdb FM-index record byte span overflows usize".to_string())?;
    if makehmmerdb_remaining(cursor) < record_payload_bytes {
        return Err("makehmmerdb FM-index record payload is truncated".to_string());
    }

    if has_text_and_sa {
        text = unpack_dna_quads(cursor, n)?;
        if n == 0 {
            return Err("makehmmerdb FM-index text record is missing terminal symbol".to_string());
        }
    }
    let _bwt = unpack_dna_quads(cursor, n)?;
    if has_text_and_sa {
        for _ in 0..sa_count {
            let _ = read_u32(cursor)?;
        }
    }
    for _ in 0..occ_b_count {
        let _ = read_u16(cursor)?;
    }
    for _ in 0..occ_sb_count {
        let _ = read_u32(cursor)?;
    }

    if has_text_and_sa {
        Ok(NhmmerFmRecord {
            seq_offset,
            seq_count,
            ambig_offset,
            ambig_count,
            overlap_bases,
            text_bases_len: n,
            text,
        })
    } else {
        Ok(NhmmerFmRecord {
            seq_offset,
            seq_count,
            ambig_offset,
            ambig_count,
            overlap_bases,
            text_bases_len: 0,
            text: Vec::new(),
        })
    }
}

fn unpack_dna_quads<R: Read>(cursor: &mut R, len: usize) -> Result<Vec<u8>, String> {
    let mut packed = vec![0u8; len.div_ceil(4)];
    cursor
        .read_exact(&mut packed)
        .map_err(|e| format!("truncated packed FM DNA text: {e}"))?;
    let mut out = Vec::with_capacity(len);
    for byte in packed {
        for shift in [6, 4, 2, 0] {
            if out.len() == len {
                break;
            }
            out.push(match (byte >> shift) & 0b11 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                3 => b'T',
                _ => unreachable!(),
            });
        }
    }
    Ok(out)
}

fn read_nul_string<R: Read>(cursor: &mut R, len: usize) -> Result<String, String> {
    let mut bytes = vec![0u8; len + 1];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| format!("truncated makehmmerdb string: {e}"))?;
    if bytes[len] != 0 {
        return Err("makehmmerdb string is missing NUL terminator".to_string());
    }
    if bytes[..len].contains(&0) {
        return Err("makehmmerdb string contains embedded NUL byte".to_string());
    }
    String::from_utf8(bytes[..len].to_vec())
        .map_err(|e| format!("makehmmerdb string is not UTF-8: {e}"))
}

fn makehmmerdb_remaining(cursor: &Cursor<&[u8]>) -> usize {
    cursor
        .get_ref()
        .len()
        .saturating_sub(cursor.position() as usize)
}

fn u64_to_usize(value: u64, what: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{what} exceeds usize"))
}

fn checked_count_bytes(count: usize, element_size: usize, what: &str) -> Result<usize, String> {
    count
        .checked_mul(element_size)
        .ok_or_else(|| format!("{what} byte span overflows usize"))
}

fn ensure_makehmmerdb_remaining(
    cursor: &Cursor<&[u8]>,
    count: usize,
    element_size: usize,
    what: &str,
) -> Result<(), String> {
    let bytes = checked_count_bytes(count, element_size, what)?;
    if makehmmerdb_remaining(cursor) < bytes {
        return Err(format!("{what} is truncated"));
    }
    Ok(())
}

fn read_u8<R: Read>(cursor: &mut R) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("truncated makehmmerdb byte: {e}"))?;
    Ok(buf[0])
}

fn read_u16<R: Read>(cursor: &mut R) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("truncated makehmmerdb u16: {e}"))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(cursor: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("truncated makehmmerdb u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(cursor: &mut R) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("truncated makehmmerdb i32: {e}"))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(cursor: &mut R) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("truncated makehmmerdb u64: {e}"))?;
    Ok(u64::from_le_bytes(buf))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn nhmmer_max_length(
    hmm: &hmmer_pure_rs::hmm::Hmm,
    w_length: Option<i32>,
    w_beta: Option<f64>,
) -> i32 {
    if let Some(w_length) = w_length {
        w_length
    } else if let Some(w_beta) = w_beta.filter(|beta| *beta > 0.0) {
        builder::max_length_from_beta(hmm, w_beta)
    } else if hmm.max_length > 0 {
        hmm.max_length
    } else if w_beta == Some(0.0) {
        (hmm.m * 4).max(1) as i32
    } else {
        builder::max_length_from_beta(hmm, DEFAULT_WINDOW_BETA)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct NhmmerBiasWindowLengths {
    pub(crate) b1: usize,
    pub(crate) b2: usize,
    pub(crate) b3: usize,
}

impl Default for NhmmerBiasWindowLengths {
    fn default() -> Self {
        Self {
            b1: 110,
            b2: 240,
            b3: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_dfam_query_accession_uses_dash_when_absent() {
        assert_eq!(search_dfam_query_accession(None), "-");
        assert_eq!(search_dfam_query_accession(Some("")), "-");
        assert_eq!(
            search_dfam_query_accession(Some("DF0000629.2")),
            "DF0000629.2"
        );
    }

    #[test]
    fn makehmmerdb_c_string_rejects_embedded_nul() {
        let err = read_nul_string(&mut &b"ab\0cd\0"[..], 5).unwrap_err();
        assert!(err.contains("embedded NUL"));
    }

    #[test]
    fn dfamtblout_uses_dynamic_name_and_position_widths() {
        use hmmer_pure_rs::tophits::{AliDisplay, Domain, Hit, TopHits, P7_IS_REPORTED};

        let mut th = TopHits::new();
        th.hits.push(Hit {
            name: "very-long-target-name-for-dfam-width".to_string(),
            acc: "very-long-model-accession".to_string(),
            desc: "description".to_string(),
            n: 1_234_567,
            sortkey: -10.0,
            score: 42.0,
            bias: 0.0,
            pre_score: 42.0,
            sum_score: 42.0,
            lnp: (1.0e-12_f64).ln(),
            pre_lnp: (1.0e-12_f64).ln(),
            sum_lnp: (1.0e-12_f64).ln(),
            nexpected: 1.0,
            nregions: 1,
            nclustered: 1,
            noverlaps: 0,
            nenvelopes: 1,
            ndom: 1,
            nreported: 1,
            nincluded: 1,
            dcl: vec![Domain {
                iali: 1_234_560,
                jali: 1_234_500,
                ienv: 1_234_490,
                jenv: 1_234_567,
                bitscore: 42.0,
                lnp: (1.0e-12_f64).ln(),
                dombias: 0.5,
                oasc: 0.0,
                envsc: 0.0,
                domcorrection: 0.0,
                is_reported: true,
                is_included: true,
                ad: Some(AliDisplay {
                    hmmfrom: 12,
                    hmmto: 345,
                    sqfrom: 1_234_560,
                    sqto: 1_234_500,
                    ..Default::default()
                }),
            }],
            flags: P7_IS_REPORTED,
            seqidx: 0,
            subseq_start: 0,
        });

        let mut out = Vec::new();
        write_nhmmer_dfamtblout(
            &mut out,
            "very-long-query-name-for-dfam-width",
            Some("DF0000001.1"),
            "sq-len",
            &th,
            1.0,
        );
        let out = String::from_utf8(out).unwrap();

        assert!(out.contains("very-long-target-name-for-dfam-width  DF0000001.1"));
        assert!(out.contains("very-long-query-name-for-dfam-width"));
        let row = out
            .lines()
            .find(|line| !line.starts_with('#') && !line.trim().is_empty())
            .unwrap();
        let fields: Vec<&str> = row.split_whitespace().collect();
        assert_eq!(
            &fields[9..14],
            &["1234560", "1234500", "1234490", "1234567", "1234567"]
        );
    }

    #[test]
    fn nhmmer_rejects_domtblout_like_c() {
        let err = match Args::try_parse_from([
            "nhmmer",
            "--domtblout",
            "domains.tbl",
            "query.hmm",
            "targets.fa",
        ]) {
            Ok(_) => panic!("nhmmer unexpectedly accepted --domtblout"),
            Err(err) => err,
        };

        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
    }

    #[test]
    fn nhmmer_parses_window_length_override() {
        let args = Args::try_parse_from(["nhmmer", "--w_length", "50", "query.hmm", "targets.fa"])
            .unwrap();

        assert_eq!(args.w_length, Some(50));
    }

    #[test]
    fn nhmmer_reads_binary_h3m_query_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("query.h3m");
        let mut hmm = hmmer_pure_rs::Hmm::new(4, hmmer_pure_rs::alphabet::AlphabetType::Dna, 4);
        hmm.name = "dna_query".to_string();
        {
            let mut file = std::fs::File::create(&path).unwrap();
            hmmer_pure_rs::hmmfile_binary::write_binary_hmm(&mut file, &hmm).unwrap();
        }

        let hmms = read_hmms(&path).unwrap();
        assert_eq!(hmms.len(), 1);
        assert_eq!(hmms[0].name, "dna_query");
        assert_eq!(hmms[0].abc_type, hmmer_pure_rs::alphabet::AlphabetType::Dna);
    }

    #[test]
    fn nhmmer_loads_makehmmerdb_container_index_records_for_seed_lookup() {
        let reversed_text = b"TGCATGCA";
        let fm = FmIndex::build(reversed_text);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HMMERDB\0");
        bytes.extend_from_slice(MAKEHMMERDB_INDEX_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        for value in [0u64, 0, fm.n as u64, 0, 1, 0, 0, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&(fm.bwt.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(fm.sa.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&256u64.to_le_bytes());
        bytes.extend_from_slice(&fm.bwt);
        for &pos in &fm.sa {
            bytes.extend_from_slice(&pos.to_le_bytes());
        }
        for &value in &fm.c {
            bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }

        let records = parse_makehmmerdb_index_extension(&bytes).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].block_id, 0);
        assert_eq!(records[0].kind, 0);
        assert_eq!(records[0].text_start, 0);
        assert_eq!(records[0].text_len, 8);
        assert_eq!(records[0].seq_offset, 0);
        assert_eq!(records[0].seq_count, 1);
        assert_eq!(records[0].ambig_offset, 0);
        assert_eq!(records[0].ambig_count, 0);
        assert_eq!(records[0].overlap_bases, 0);
        let mut positions = records[0].fm.locate(b"TGC");
        positions.sort();
        assert_eq!(positions, vec![0, 4]);
        let blocks = vec![NhmmerFmRecord {
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            text_bases_len: 8,
            text: b"ACGTACGT".to_vec(),
        }];
        validate_makehmmerdb_index_records(&records, &blocks, true, 0).unwrap();
    }

    #[test]
    fn makehmmerdb_index_rejects_huge_record_count_before_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HMMERDB\0");
        bytes.extend_from_slice(MAKEHMMERDB_INDEX_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());

        let err = parse_makehmmerdb_index_extension(&bytes).unwrap_err();
        assert!(
            err.contains("makehmmerdb FM-index record table"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn makehmmerdb_index_rejects_huge_bwt_before_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HMMERDB\0");
        bytes.extend_from_slice(MAKEHMMERDB_INDEX_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        for value in [0u64, 0, 8, 0, 1, 0, 0, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&4096u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&256u64.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0, 256 * 8));

        let err = parse_makehmmerdb_index_extension(&bytes).unwrap_err();
        assert!(
            err.contains("makehmmerdb FM-index record payload"),
            "unexpected error: {err}"
        );
    }

    fn makehmmerdb_c_metadata_header(freq_sa: u32, freq_cnt_b: u32, seq_count: u32) -> Vec<u8> {
        let mut bytes = vec![1, 0, 4, 2];
        bytes.extend_from_slice(&freq_sa.to_le_bytes());
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.extend_from_slice(&freq_cnt_b.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&seq_count.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes
    }

    #[test]
    fn makehmmerdb_c_metadata_rejects_zero_sampling_frequencies() {
        let zero_sa_bytes = makehmmerdb_c_metadata_header(0, 8, 0);
        let mut zero_sa = Cursor::new(zero_sa_bytes.as_slice());
        let err = read_makehmmerdb_c_metadata_payload(&mut zero_sa).unwrap_err();
        assert!(
            err.contains("suffix-array sampling frequency is zero"),
            "unexpected error: {err}"
        );

        let zero_cnt_bytes = makehmmerdb_c_metadata_header(4, 0, 0);
        let mut zero_cnt = Cursor::new(zero_cnt_bytes.as_slice());
        let err = read_makehmmerdb_c_metadata_payload(&mut zero_cnt).unwrap_err();
        assert!(
            err.contains("occurrence sampling frequency is zero"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn makehmmerdb_c_metadata_rejects_short_huge_sequence_payload_before_allocation() {
        let bytes = makehmmerdb_c_metadata_header(4, 8, u32::MAX);
        let mut cursor = Cursor::new(bytes.as_slice());

        let err = read_makehmmerdb_c_metadata_payload(&mut cursor).unwrap_err();
        assert!(
            err.contains("makehmmerdb metadata payload is truncated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn makehmmerdb_c_metadata_rejects_invalid_ambiguity_coordinates() {
        let mut bytes = makehmmerdb_c_metadata_header(4, 8, 0);
        let ambig_count_offset = 4 + 4 + 4 + 4 + 2 + 4;
        bytes[ambig_count_offset..ambig_count_offset + 4].copy_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(-1i32).to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        let mut cursor = Cursor::new(bytes.as_slice());

        let err = read_makehmmerdb_c_metadata_payload(&mut cursor).unwrap_err();
        assert!(
            err.contains("invalid coordinates"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn makehmmerdb_c_fm_record_rejects_zero_sampling_frequencies() {
        let mut empty = Cursor::new([].as_slice());
        let err = read_makehmmerdb_c_fm_record(&mut empty, 0, 8, true).unwrap_err();
        assert!(
            err.contains("suffix-array sampling frequency is zero"),
            "unexpected error: {err}"
        );

        let mut empty = Cursor::new([].as_slice());
        let err = read_makehmmerdb_c_fm_record(&mut empty, 4, 0, true).unwrap_err();
        assert!(
            err.contains("occurrence sampling frequency is zero"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn makehmmerdb_c_fm_record_tracks_terminal_symbol_separately_from_bases() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&5u64.to_le_bytes());
        for value in [0u32, 0, 0, 0, 1, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0b00011011, 0]);
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0u8, 8 * 2));
        bytes.extend(std::iter::repeat_n(0u8, 8 * 4));
        let mut cursor = Cursor::new(bytes.as_slice());

        let record = read_makehmmerdb_c_fm_record(&mut cursor, 4, 8, true).unwrap();
        assert_eq!(record.text_bases_len, 5);
        assert_eq!(&record.text[..4], b"ACGT");
    }

    #[test]
    fn makehmmerdb_c_fm_record_rejects_short_huge_payload_before_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1024u64.to_le_bytes());
        for value in [0u32, 0, 0, 0, 0, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut cursor = Cursor::new(bytes.as_slice());

        let err = read_makehmmerdb_c_fm_record(&mut cursor, 4, 8, true).unwrap_err();
        assert!(
            err.contains("makehmmerdb FM-index record payload"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn nhmmer_fm_seed_hits_create_candidate_windows() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"TTACGTACAA";
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 0,
                kind: 0,
                text_start: 0,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).unwrap();
        assert_eq!(
            windows,
            vec![NhmmerCandidateWindow {
                n: 3,
                length: 4,
                k: 4
            }]
        );
    }

    #[test]
    fn nhmmer_fm_seed_ssv_length_changes_candidate_window_span() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"ACGTTGCAACGATCGTACGAGTCCATGCTAGGATCCGTAACGTTG";
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(text.len(), AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b'N'; hmm.m + 2]);
        hmm.consensus.as_mut().unwrap()[0] = b' ';
        hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
            let ch = text[node - 1];
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
            if (16..=30).contains(&node) {
                hmm.consensus.as_mut().unwrap()[node] = ch;
            }
        }

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 0,
                kind: 0,
                text_start: 0,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        let base_config = NhmmerFmSeedConfig {
            score_threshold_bits: 999.0,
            consensus_match_req: 0,
            ..NhmmerFmSeedConfig::default()
        };
        let short_windows = fm_seed_candidate_windows_with_config(
            &target_db,
            0,
            &hmm,
            &Bg::new(&abc),
            100,
            0.02,
            false,
            NhmmerFmSeedConfig {
                ssv_length: 16,
                ..base_config
            },
        )
        .unwrap();
        let long_windows = fm_seed_candidate_windows_with_config(
            &target_db,
            0,
            &hmm,
            &Bg::new(&abc),
            100,
            0.02,
            false,
            NhmmerFmSeedConfig {
                ssv_length: text.len(),
                ..base_config
            },
        )
        .unwrap();

        assert_eq!(short_windows.len(), 1);
        assert_eq!(long_windows.len(), 1);
        assert!(
            short_windows[0].length < long_windows[0].length,
            "seed_ssv_length should bound FM seed extension before window creation: short={short_windows:?} long={long_windows:?}"
        );
    }

    #[test]
    fn nhmmer_fm_seed_hits_use_configured_consensus_match_run() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"TTACGTACGTACAA";
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(13, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![
            b' ', b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'G', b'G',
            b' ',
        ]);

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 0,
                kind: 0,
                text_start: 0,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        let windows = fm_seed_candidate_windows_with_config(
            &target_db,
            0,
            &hmm,
            &Bg::new(&abc),
            10,
            0.02,
            false,
            NhmmerFmSeedConfig {
                consensus_match_req: 10,
                ..NhmmerFmSeedConfig::default()
            },
        )
        .unwrap();
        assert_eq!(
            windows,
            vec![NhmmerCandidateWindow {
                n: 3,
                length: 10,
                k: 10
            }]
        );
    }

    #[test]
    fn nhmmer_fm_model_consensus_trie_reaches_late_capped_seed() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let motif = b"CCGTAACGTTAC";
        let text = [b"GG".as_slice(), motif, b"GG".as_slice()].concat();
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(&text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(180, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b'A'; hmm.m + 2]);
        hmm.consensus.as_mut().unwrap()[0] = b' ';
        hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
        for (offset, &ch) in motif.iter().enumerate() {
            hmm.consensus.as_mut().unwrap()[150 + offset] = ch;
        }

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 0,
                kind: 0,
                text_start: 0,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 20, false).unwrap();
        assert!(
            windows
                .iter()
                .any(|window| window.n == 3 && window.length == motif.len() && window.k == 161),
            "FM-pruned consensus recursion should reach a late materialized-seed-capped motif: {windows:?}"
        );
    }

    #[test]
    fn nhmmer_fm_model_consensus_trie_augments_materialized_seed_hits() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let early_motif = b"CCCCCCCCCCCC";
        let late_motif = b"CCGTAACGTTAC";
        let text = [
            b"GG".as_slice(),
            early_motif,
            b"GG".as_slice(),
            late_motif,
            b"GG".as_slice(),
        ]
        .concat();
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(&text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(180, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b'A'; hmm.m + 2]);
        hmm.consensus.as_mut().unwrap()[0] = b' ';
        hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
        for (offset, &ch) in early_motif.iter().enumerate() {
            hmm.consensus.as_mut().unwrap()[1 + offset] = ch;
        }
        for (offset, &ch) in late_motif.iter().enumerate() {
            hmm.consensus.as_mut().unwrap()[150 + offset] = ch;
        }

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 0,
                kind: 0,
                text_start: 0,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 24, false).unwrap();
        assert!(
            windows.iter().any(|window| window.n == 3
                && window.length == early_motif.len()
                && window.k == 12),
            "materialized consensus seed should still create the early window: {windows:?}"
        );
        assert!(
            windows.iter().any(|window| {
                window.n == 17 && window.length == late_motif.len() && window.k == 161
            }),
            "recursive consensus traversal should augment, not wait for an empty materialized result: {windows:?}"
        );
    }

    #[test]
    fn nhmmer_fm_model_consensus_trie_ranks_before_source_cap() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let low_motif = b"AAAAAAAAAAAA";
        let high_motif = b"CCGTAACGTTAC";
        let text = [b"GG".as_slice(), low_motif, b"GG".as_slice(), high_motif].concat();
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();

        let mut hmm = Hmm::new(32, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b'A'; hmm.m + 2]);
        hmm.consensus.as_mut().unwrap()[0] = b' ';
        hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.24;
            }
        }
        for (offset, &ch) in high_motif.iter().enumerate() {
            let node = 21 + offset;
            hmm.consensus.as_mut().unwrap()[node] = ch;
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let hits = fm_locate_model_consensus_seeds(
            &FmIndex::build(&reversed_text),
            &hmm,
            &Bg::new(&abc),
            false,
            text.len(),
            text.len(),
            1,
            NhmmerFmSeedConfig {
                consensus_match_req: 10,
                ..NhmmerFmSeedConfig::default()
            },
        );
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].0.model_end >= 30,
            "consensus trie should spend a tight source cap on the strongest target-supported seed first: {hits:?}"
        );
    }

    #[test]
    fn nhmmer_fm_seed_extension_scores_diagonal_flanks() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"TTACGTAA";
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(6, AlphabetType::Dna, abc.k);
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
        }
        for (node, ch) in [
            (1, b'A'),
            (2, b'C'),
            (3, b'G'),
            (4, b'T'),
            (5, b'A'),
            (6, b'C'),
        ] {
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let seed = NhmmerConsensusSeed {
            bases: b"CGT".to_vec(),
            model_end: 4,
        };
        let (start, len, model_end, score_bits) =
            fm_extend_seed_diagonal(&seq, &hmm, &Bg::new(&abc), &seed, 3, 6);
        assert_eq!((start, len, model_end), (2, 5, 5));
        assert!(
            (score_bits - 9.780284).abs() < 1e-5,
            "unexpected extended diagonal score: {score_bits}"
        );
    }

    #[test]
    fn nhmmer_fm_seed_extension_selects_best_bounded_subdiagonal_like_c() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"ACGT";
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
        }
        for (node, ch, p) in [
            (1, b'A', 0.97),
            (2, b'C', 0.01),
            (3, b'G', 0.97),
            (4, b'T', 0.97),
        ] {
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = p;
        }

        let seed = NhmmerConsensusSeed {
            bases: text.to_vec(),
            model_end: 4,
        };
        let (start, len, model_end, score_bits) =
            fm_extend_seed_diagonal(&seq, &hmm, &Bg::new(&abc), &seed, 0, 6);
        assert_eq!(
            (start, len, model_end),
            (2, 2, 4),
            "C FM_extendSeed keeps the highest-scoring clipped sub-diagonal before FM_window_from_diag derives k+length-1"
        );
        assert!(score_bits > 0.0);
    }

    #[test]
    fn nhmmer_fm_seed_window_emission_filters_nonpositive_extended_diagonal() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"ACGT";
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let seq_meta = NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            fm_start: 0,
            length: text.len(),
        };
        let record = NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(&text.iter().rev().copied().collect::<Vec<_>>()),
        };
        let seed = NhmmerConsensusSeed {
            bases: text.to_vec(),
            model_end: text.len(),
        };

        let mut low_hmm = Hmm::new(text.len(), AlphabetType::Dna, abc.k);
        low_hmm.name = "query".to_string();
        for node in 1..=low_hmm.m {
            for x in 0..abc.k {
                low_hmm.mat[node][x] = 0.25;
            }
        }

        let bg = Bg::new(&abc);
        let mut windows = Vec::new();
        fm_push_seed_window(
            &mut windows,
            &[],
            &seq_meta,
            &seq,
            &record,
            &low_hmm,
            &bg,
            false,
            &seed,
            0,
            text.len(),
        );
        assert!(
            windows.is_empty(),
            "C FM_extendSeed only sends SSV-passing positive diagonals to FM_window_from_diag: {windows:?}"
        );

        let mut high_hmm = low_hmm;
        for (offset, &ch) in text.iter().enumerate() {
            let node = offset + 1;
            for x in 0..abc.k {
                high_hmm.mat[node][x] = 0.01;
            }
            let code = abc.inmap[ch as usize] as usize;
            high_hmm.mat[node][code] = 0.97;
        }

        fm_push_seed_window(
            &mut windows,
            &[],
            &seq_meta,
            &seq,
            &record,
            &high_hmm,
            &bg,
            false,
            &seed,
            0,
            text.len(),
        );
        assert_eq!(windows.len(), 1);
        assert_eq!(
            windows[0].window,
            NhmmerCandidateWindow {
                n: 1,
                length: text.len(),
                k: text.len(),
            }
        );
        assert!(windows[0].score_bits > 0.0);
    }

    #[test]
    fn nhmmer_fm_seed_windows_merge_close_same_diagonal_runs() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"TTACGTAC";
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(6, AlphabetType::Dna, abc.k);
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
        }
        for (node, ch) in [
            (1, b'A'),
            (2, b'C'),
            (3, b'G'),
            (4, b'T'),
            (5, b'A'),
            (6, b'C'),
        ] {
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let windows = fm_merge_seed_windows(
            vec![
                NhmmerScoredCandidateWindow {
                    window: NhmmerCandidateWindow {
                        n: 3,
                        length: 4,
                        k: 4,
                    },
                    score_bits: 0.0,
                },
                NhmmerScoredCandidateWindow {
                    window: NhmmerCandidateWindow {
                        n: 5,
                        length: 4,
                        k: 6,
                    },
                    score_bits: 0.0,
                },
            ],
            6,
            &seq,
            &hmm,
            &Bg::new(&abc),
        );

        assert_eq!(windows.len(), 1);
        assert_eq!(
            windows[0].window,
            NhmmerCandidateWindow {
                n: 3,
                length: 6,
                k: 6,
            }
        );
        assert!(
            (windows[0].score_bits - 11.736341).abs() < 1e-5,
            "merged diagonal should be rescored over its full span: {windows:?}"
        );
    }

    #[test]
    fn nhmmer_fm_seed_hits_use_score_threshold_seed_strings() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let seed_text = b"ACGTACGTACGTACG";
        let text = b"TTACGTACGTACGTACGAA";
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(seed_text.len(), AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b'T'; seed_text.len() + 2]);
        hmm.consensus.as_mut().unwrap()[0] = b' ';
        hmm.consensus.as_mut().unwrap()[seed_text.len() + 1] = b' ';
        for (node, &ch) in seed_text.iter().enumerate() {
            let node = node + 1;
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 0,
                kind: 0,
                text_start: 0,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 15, false).unwrap();
        assert!(
            windows
                .iter()
                .any(|window| window.n == 3 && window.length == seed_text.len() && window.k >= 8),
            "score-threshold seed should create a bounded FM candidate window: {windows:?}"
        );
    }

    #[test]
    fn nhmmer_fm_model_score_trie_reaches_target_pruned_late_model_seed() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let motif = b"CCGTAACGTTACGTA";
        let text = [b"GG".as_slice(), motif, b"GG".as_slice()].concat();
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(&text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(290, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b'T'; hmm.m + 2]);
        hmm.consensus.as_mut().unwrap()[0] = b' ';
        hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
            let ch = if (270..270 + motif.len()).contains(&node) {
                motif[node - 270]
            } else {
                b'A'
            };
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 0,
                kind: 0,
                text_start: 0,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 20, false).unwrap();
        assert!(
            windows.iter().any(|window| {
                let window_start0 = window.n - 1;
                let window_end0 = window_start0 + window.length;
                window_start0 <= 2 && window_end0 >= 2 + motif.len() && window.k >= 277
            }),
            "FM-pruned model score recursion should reach a late target-supported seed: {windows:?}"
        );
    }

    #[test]
    fn nhmmer_fm_model_score_trie_augments_existing_consensus_windows() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let early_motif = b"ACGTACGTACGT";
        let late_motif = b"CCGTAACGTTACGTA";
        let text = [
            b"GG".as_slice(),
            early_motif.as_slice(),
            b"GG".as_slice(),
            late_motif.as_slice(),
            b"GG".as_slice(),
        ]
        .concat();
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(&text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(300, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b'T'; hmm.m + 2]);
        hmm.consensus.as_mut().unwrap()[0] = b' ';
        hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
            let code = abc.inmap[b'A' as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }
        for (offset, &ch) in early_motif.iter().enumerate() {
            let node = 1 + offset;
            hmm.consensus.as_mut().unwrap()[node] = ch;
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }
        for (offset, &ch) in late_motif.iter().enumerate() {
            let node = 280 + offset;
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 0,
                kind: 0,
                text_start: 0,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 20, false).unwrap();
        assert!(
            windows.iter().any(|window| window.n == 3
                && window.length >= early_motif.len()
                && window.k <= 20),
            "consensus tier should still create the early FM window: {windows:?}"
        );
        assert!(
            windows.iter().any(|window| {
                let window_start0 = window.n - 1;
                let window_end0 = window_start0 + window.length;
                window_start0 <= 16 && window_end0 >= 16 + late_motif.len() && window.k >= 294
            }),
            "normal score-trie tier should add the late score-only FM window without waiting for empty earlier tiers: {windows:?}"
        );
    }

    #[test]
    fn nhmmer_fm_model_score_trie_ranks_starts_before_source_cap() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let weak_motif = b"ACGTACGTACGTACG";
        let strong_motif = b"CCGTAACGTTACGTA";
        let text = [
            b"GG".as_slice(),
            weak_motif,
            b"GG".as_slice(),
            strong_motif,
            b"GG".as_slice(),
        ]
        .concat();
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();

        let mut hmm = Hmm::new(45, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b'A'; hmm.m + 2]);
        hmm.consensus.as_mut().unwrap()[0] = b' ';
        hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.25;
            }
        }
        for (offset, &ch) in weak_motif.iter().enumerate() {
            let node = 1 + offset;
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.75;
        }
        for (offset, &ch) in strong_motif.iter().enumerate() {
            let node = 25 + offset;
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let hits = fm_locate_model_score_seeds(
            &FmIndex::build(&reversed_text),
            &hmm,
            &Bg::new(&abc),
            false,
            text.len(),
            text.len(),
            1,
        );
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].0.model_end >= 32,
            "tight score-trie source cap should prefer the stronger late seed: {hits:?}"
        );
    }

    #[test]
    fn nhmmer_fm_final_windows_keep_more_than_4096_plausible_seed_windows() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let window_count = NHMMER_FM_WINDOW_LIMIT_REGRESSION_SIZE + 2;
        let text = vec![b'A'; window_count];
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(&text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(1, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b' ']);
        let bg = Bg::new(&abc);

        let scored_windows: Vec<NhmmerScoredCandidateWindow> = (0..window_count)
            .map(|idx| NhmmerScoredCandidateWindow {
                window: NhmmerCandidateWindow {
                    n: idx + 1,
                    length: 1,
                    k: 1,
                },
                score_bits: (window_count - idx) as f32,
            })
            .collect();

        let windows = fm_finalize_seed_windows(scored_windows, 1, &seq, &hmm, &bg, false).unwrap();
        assert_eq!(windows.len(), window_count);
        assert_eq!(windows.first().unwrap().n, 1);
        assert_eq!(windows.last().unwrap().n, window_count);
        assert!(
            windows
                .iter()
                .any(|window| window.n == NHMMER_FM_WINDOW_LIMIT_REGRESSION_SIZE + 1),
            "late plausible FM seed windows must not be dropped before long-target scanning"
        );
    }

    #[test]
    fn nhmmer_fm_consensus_trie_ranks_short_suffix_starts_before_source_cap() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let bg = Bg::new(&abc);
        let early_seed = b"ACGTACGTAC";
        let late_seed = b"CCGTAACGTT";
        let consensus = [early_seed.as_slice(), b"A".as_slice(), late_seed.as_slice()].concat();
        let text = [
            early_seed.as_slice(),
            b"GG".as_slice(),
            late_seed.as_slice(),
        ]
        .concat();
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();

        let mut hmm = Hmm::new(consensus.len(), AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' '; hmm.m + 2]);
        for (offset, &ch) in consensus.iter().enumerate() {
            hmm.consensus.as_mut().unwrap()[offset + 1] = ch;
        }
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
            let code = abc.inmap[consensus[node - 1] as usize] as usize;
            hmm.mat[node][code] = if node <= early_seed.len() { 0.35 } else { 0.97 };
        }

        let hits = fm_locate_model_consensus_seeds(
            &FmIndex::build(&reversed_text),
            &hmm,
            &bg,
            false,
            text.len(),
            text.len(),
            1,
            NhmmerFmSeedConfig {
                consensus_match_req: 10,
                ..NhmmerFmSeedConfig::default()
            },
        );

        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].0.model_end,
            consensus.len(),
            "tight consensus-trie source cap should rank the high-scoring 10-mer suffix before an earlier weaker seed: {hits:?}"
        );
        assert_eq!(hits[0].0.bases, late_seed);
    }

    #[test]
    fn nhmmer_fm_model_score_reverse_trie_locates_target_supported_seed() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let motif = b"CCGTAACGTTACGTA";
        let text = [b"GG".as_slice(), motif, b"TT".as_slice()].concat();
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();

        let mut hmm = Hmm::new(motif.len(), AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b'A'; hmm.m + 2]);
        hmm.consensus.as_mut().unwrap()[0] = b' ';
        hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
            let code = abc.inmap[motif[node - 1] as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let bg = Bg::new(&abc);
        let (abc, max_depth, lod_bits, _best_suffix, best_prefix, consensus_codes) =
            nhmmer_fm_lod_tables(&hmm, &bg, NhmmerFmSeedConfig::default()).unwrap();
        let fm = FmIndex::build(&reversed_text);
        let mut hits = Vec::new();
        let mut current_rev = Vec::new();
        fm_model_score_seed_reverse_dfs(
            &fm,
            hmm.m,
            hmm.m,
            max_depth,
            false,
            &mut current_rev,
            &lod_bits,
            &best_prefix,
            &consensus_codes,
            &abc,
            &mut hits,
            64,
            NhmmerFmScoreState::new(),
            NhmmerFmSeedConfig::default(),
        );

        assert!(
            hits.iter().any(|(seed, fm_pos)| {
                seed.model_end == motif.len()
                    && seed.bases == motif
                    && *fm_pos == text.len() - 2 - motif.len()
            }),
            "reverse model-direction score trie should locate the full target-supported seed: {hits:?}"
        );
    }

    #[test]
    fn nhmmer_fm_reverse_score_trie_prefix_interval_matches_seed_orientation() {
        let text = b"TTACGTAA";
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let fm = FmIndex::build(&reversed_text);

        let full_seed_interval = fm_interval_for_model_seed(&fm, b"ACGT", false).unwrap();
        let reverse_prefix_interval =
            fm_interval_for_reverse_model_prefix(&fm, b"TGCA", false).unwrap();
        let mut full_positions = fm.locate_interval(full_seed_interval);
        let mut reverse_prefix_positions = fm.locate_interval(reverse_prefix_interval);
        full_positions.sort();
        reverse_prefix_positions.sort();
        assert_eq!(reverse_prefix_positions, full_positions);

        assert!(
            fm_interval_for_reverse_model_prefix(&fm, b"TC", false).is_none(),
            "reverse score recursion should prune a target-absent suffix before deeper model traversal"
        );
    }

    #[test]
    fn nhmmer_score_threshold_seed_strings_include_reverse_model_direction() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let motif = b"CCGTAACGTTACGTA";
        let mut hmm = Hmm::new(motif.len(), AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
            let code = abc.inmap[motif[node - 1] as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let bg = Bg::new(&abc);
        let (abc, max_depth, lod_bits, _best_suffix, best_prefix, consensus_codes) =
            nhmmer_fm_lod_tables(&hmm, &bg, NhmmerFmSeedConfig::default()).unwrap();
        let mut seeds = Vec::new();
        let mut current_rev = Vec::new();
        nhmmer_score_seed_reverse_dfs(
            hmm.m,
            hmm.m,
            max_depth,
            &mut current_rev,
            &lod_bits,
            &best_prefix,
            &consensus_codes,
            &abc,
            &mut seeds,
            NHMMER_FM_SCORE_SEED_LIMIT,
            NhmmerFmScoreState::new(),
            NhmmerFmSeedConfig::default(),
        );

        assert!(
            seeds
                .iter()
                .any(|seed| seed.model_end == motif.len() && seed.bases == motif),
            "materialized reverse model-direction score seeds should include the full motif: {seeds:?}"
        );
    }

    #[test]
    fn nhmmer_fm_score_trie_prunes_nonpositive_prefixes_like_c() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let motif = b"ACCGTAACGTTACGT";
        let mut hmm = Hmm::new(motif.len(), AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        for x in 0..abc.k {
            hmm.mat[1][x] = 0.25;
        }
        for (offset, &ch) in motif[1..].iter().enumerate() {
            let node = offset + 2;
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let bg = Bg::new(&abc);
        let (abc, max_depth, lod_bits, best_suffix, _best_prefix, consensus_codes) =
            nhmmer_fm_lod_tables(&hmm, &bg, NhmmerFmSeedConfig::default()).unwrap();

        let mut materialized = Vec::new();
        let mut current = Vec::new();
        nhmmer_score_seed_dfs(
            1,
            1,
            max_depth,
            &mut current,
            &lod_bits,
            &best_suffix,
            &consensus_codes,
            &abc,
            &mut materialized,
            NHMMER_FM_SCORE_SEED_LIMIT,
            NhmmerFmScoreState::new(),
            NhmmerFmSeedConfig::default(),
        );
        assert!(
            materialized.is_empty(),
            "C FM_Recurse prunes branches whose accumulated score is <= 0: {materialized:?}"
        );

        let text = motif;
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let fm = FmIndex::build(&reversed_text);
        let mut located = Vec::new();
        let mut current = Vec::new();
        fm_model_score_seed_dfs(
            &fm,
            1,
            1,
            max_depth,
            fm.root_interval(),
            false,
            &mut current,
            &lod_bits,
            &best_suffix,
            &consensus_codes,
            &abc,
            &mut located,
            NHMMER_FM_SCORE_SEED_LIMIT,
            NhmmerFmScoreState::new(),
            NhmmerFmSeedConfig::default(),
        );
        assert!(
            located.is_empty(),
            "FM-backed score recursion should apply the same nonpositive-score pruning: {located:?}"
        );
    }

    #[test]
    fn nhmmer_fm_score_trie_prunes_low_density_unless_consensus_supported() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let motif = b"ACGTAACGTTACGTA";

        let mut low_density_hmm = Hmm::new(motif.len(), AlphabetType::Dna, abc.k);
        low_density_hmm.name = "query".to_string();
        low_density_hmm.consensus = Some(vec![b'T'; low_density_hmm.m + 2]);
        low_density_hmm.consensus.as_mut().unwrap()[0] = b' ';
        low_density_hmm.consensus.as_mut().unwrap()[low_density_hmm.m + 1] = b' ';
        for (offset, &ch) in motif.iter().enumerate() {
            let node = offset + 1;
            for x in 0..abc.k {
                low_density_hmm.mat[node][x] = 0.01;
            }
            let code = abc.inmap[ch as usize] as usize;
            low_density_hmm.mat[node][code] = if node <= 5 { 0.36 } else { 0.97 };
        }

        let bg = Bg::new(&abc);
        let (abc, max_depth, lod_bits, best_suffix, _best_prefix, consensus_codes) =
            nhmmer_fm_lod_tables(&low_density_hmm, &bg, NhmmerFmSeedConfig::default()).unwrap();
        let reversed_text: Vec<u8> = motif.iter().rev().copied().collect();
        let fm = FmIndex::build(&reversed_text);

        let mut materialized = Vec::new();
        let mut current = Vec::new();
        nhmmer_score_seed_dfs(
            1,
            1,
            max_depth,
            &mut current,
            &lod_bits,
            &best_suffix,
            &consensus_codes,
            &abc,
            &mut materialized,
            NHMMER_FM_SCORE_SEED_LIMIT,
            NhmmerFmScoreState::new(),
            NhmmerFmSeedConfig::default(),
        );
        assert!(
            materialized.is_empty(),
            "C FM_Recurse prunes non-consensus branches whose score density falls too low: {materialized:?}"
        );

        let mut located = Vec::new();
        let mut current = Vec::new();
        fm_model_score_seed_dfs(
            &fm,
            1,
            1,
            max_depth,
            fm.root_interval(),
            false,
            &mut current,
            &lod_bits,
            &best_suffix,
            &consensus_codes,
            &abc,
            &mut located,
            NHMMER_FM_SCORE_SEED_LIMIT,
            NhmmerFmScoreState::new(),
            NhmmerFmSeedConfig::default(),
        );
        assert!(
            located.is_empty(),
            "FM-backed recursion should apply the same C score-density prune: {located:?}"
        );

        let mut consensus_hmm = low_density_hmm;
        consensus_hmm.consensus = Some(vec![b' '; consensus_hmm.m + 2]);
        for (offset, &ch) in motif.iter().enumerate() {
            consensus_hmm.consensus.as_mut().unwrap()[offset + 1] = ch;
        }
        let (abc, max_depth, lod_bits, best_suffix, _best_prefix, consensus_codes) =
            nhmmer_fm_lod_tables(&consensus_hmm, &bg, NhmmerFmSeedConfig::default()).unwrap();
        let mut located = Vec::new();
        let mut current = Vec::new();
        fm_model_score_seed_dfs(
            &fm,
            1,
            1,
            max_depth,
            fm.root_interval(),
            false,
            &mut current,
            &lod_bits,
            &best_suffix,
            &consensus_codes,
            &abc,
            &mut located,
            NHMMER_FM_SCORE_SEED_LIMIT,
            NhmmerFmScoreState::new(),
            NhmmerFmSeedConfig::default(),
        );
        assert!(
            located
                .iter()
                .any(|(seed, _)| seed.model_end == motif.len() && seed.bases == motif),
            "a continuous consensus-supported path should bypass the C density prune and remain extendable: {located:?}"
        );
    }

    #[test]
    fn nhmmer_fm_seed_queries_share_intervals_and_prune_missing_branches() {
        let text = b"ACGTACGTAAAA";
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let fm = FmIndex::build(&reversed_text);
        let seed_a = NhmmerConsensusSeed {
            bases: b"ACGT".to_vec(),
            model_end: 4,
        };
        let seed_b = NhmmerConsensusSeed {
            bases: b"TCGT".to_vec(),
            model_end: 4,
        };
        let seed_c = NhmmerConsensusSeed {
            bases: b"AAAA".to_vec(),
            model_end: 4,
        };
        let seeds = vec![seed_a, seed_b, seed_c];
        let abc = Alphabet::new(AlphabetType::Dna);
        let hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
        let bg = Bg::new(&abc);
        let record = NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: 12,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm,
        };
        let queries = fm_seed_queries_for_record(&seeds, &record, 12, false, &hmm, &bg);
        let mut located: Vec<(Vec<u8>, usize)> =
            fm_locate_seed_queries(&record.fm, &queries, usize::MAX)
                .into_iter()
                .map(|(seed, pos)| (seed.bases.clone(), pos))
                .collect();
        located.sort();

        assert_eq!(
            located,
            vec![
                (b"AAAA".to_vec(), 0),
                (b"ACGT".to_vec(), 4),
                (b"ACGT".to_vec(), 8)
            ]
        );
    }

    #[test]
    fn nhmmer_fm_seed_queries_rank_identical_terminal_seeds_before_source_cap() {
        let text = b"ACGTACGT";
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let fm = FmIndex::build(&reversed_text);
        let abc = Alphabet::new(AlphabetType::Dna);
        let bg = Bg::new(&abc);
        let mut hmm = Hmm::new(8, AlphabetType::Dna, abc.k);
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.24;
            }
        }
        for (node, ch, p) in [
            (1, b'A', 0.35),
            (2, b'C', 0.35),
            (3, b'G', 0.35),
            (4, b'T', 0.35),
            (5, b'A', 0.97),
            (6, b'C', 0.97),
            (7, b'G', 0.97),
            (8, b'T', 0.97),
        ] {
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = p;
        }

        let low_seed = NhmmerConsensusSeed {
            bases: b"ACGT".to_vec(),
            model_end: 4,
        };
        let high_seed = NhmmerConsensusSeed {
            bases: b"ACGT".to_vec(),
            model_end: 8,
        };
        let record = NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm,
        };
        let seeds = vec![low_seed, high_seed];
        let queries = fm_seed_queries_for_record(&seeds, &record, text.len(), false, &hmm, &bg);

        let located = fm_locate_seed_queries(&record.fm, &queries, 1);
        assert_eq!(located.len(), 1);
        assert_eq!(
            located[0].0.model_end, 8,
            "tight FM source cap should prefer the higher-scoring model coordinate"
        );
    }

    #[test]
    fn nhmmer_fm_seed_hits_skip_ambiguous_spans() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"TTACGTACAA";
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: vec![(3, 4)],
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 0,
                kind: 0,
                text_start: 0,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 1,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        assert!(fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).is_none());
    }

    #[test]
    fn nhmmer_fm_seed_hits_include_block_text_start() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"TTACGTACAA";
        let block_text = &text[2..];
        let reversed_block_text: Vec<u8> = block_text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 1,
                kind: 0,
                text_start: 2,
                text_len: block_text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_block_text),
            }],
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).unwrap();
        assert_eq!(
            windows,
            vec![NhmmerCandidateWindow {
                n: 1,
                length: 4,
                k: 4
            }]
        );
    }

    #[test]
    fn nhmmer_fm_seed_hits_use_block_local_ambiguity_coordinates() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"TTACGTACAA";
        let block_text = &text[2..];
        let reversed_block_text: Vec<u8> = block_text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(block_text);
        seq.n = block_text.len();
        seq.l = block_text.len();

        let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: block_text.len(),
            }],
            fm_ambiguities: vec![(1, 2)],
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 1,
                kind: 0,
                text_start: 2,
                text_len: block_text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 1,
                overlap_bases: 0,
                fm: FmIndex::build(&reversed_block_text),
            }],
        };

        assert!(fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).is_none());
    }

    #[test]
    fn nhmmer_fm_seed_hits_skip_wholly_overlapped_block_prefix() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"ACGTACAA";
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 1,
                kind: 0,
                text_start: 7,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 4,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        assert!(fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).is_none());
    }

    #[test]
    fn nhmmer_fm_seed_hits_keep_boundary_spanning_overlap_hits() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"TACGTCAA";
        let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 1,
                kind: 0,
                text_start: 7,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 2,
                fm: FmIndex::build(&reversed_text),
            }],
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).unwrap();
        assert_eq!(
            windows,
            vec![NhmmerCandidateWindow {
                n: 2,
                length: 4,
                k: 4
            }]
        );
    }

    #[test]
    fn nhmmer_fm_reverse_seed_hits_create_crick_candidate_windows() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"AACCTTAC";
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(3, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b'A', b'G', b' ']);

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records: vec![NhmmerFmIndexRecord {
                block_id: 0,
                kind: 1,
                text_start: 0,
                text_len: text.len(),
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                fm: FmIndex::build(text),
            }],
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 5, true).unwrap();
        assert_eq!(
            windows,
            vec![NhmmerCandidateWindow {
                n: 3,
                length: 3,
                k: 3
            }]
        );
        assert!(fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 5, false).is_none());
    }

    #[test]
    fn nhmmer_raw_c_stream_records_create_seed_indexes() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"TTACGTACAA";
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

        let fm_index_records = build_raw_c_stream_fm_index_records(
            &[NhmmerFmRecord {
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                text_bases_len: text.len(),
                text: text.to_vec(),
            }],
            false,
        )
        .unwrap();
        assert_eq!(fm_index_records.len(), 2);
        assert_eq!(fm_index_records[0].kind, 0);
        assert_eq!(fm_index_records[1].kind, 1);

        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records,
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).unwrap();
        assert_eq!(
            windows,
            vec![NhmmerCandidateWindow {
                n: 3,
                length: 4,
                k: 4
            }]
        );
    }

    #[test]
    fn nhmmer_raw_c_stream_reverse_record_creates_crick_seed_windows() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = b"AACCTTAC";
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(text);
        seq.n = text.len();
        seq.l = text.len();

        let mut hmm = Hmm::new(3, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b'A', b'G', b' ']);

        let fm_index_records = build_raw_c_stream_fm_index_records(
            &[NhmmerFmRecord {
                seq_offset: 0,
                seq_count: 1,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
                text_bases_len: text.len(),
                text: text.to_vec(),
            }],
            false,
        )
        .unwrap();
        let target_db = NhmmerTargetDb {
            sequences: vec![seq],
            fm_sequence_meta: vec![NhmmerFmSequenceMeta {
                name: "target".to_string(),
                acc: String::new(),
                desc: String::new(),
                fm_start: 0,
                length: text.len(),
            }],
            fm_ambiguities: Vec::new(),
            fm_index_records,
        };

        let windows =
            fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 5, true).unwrap();
        assert_eq!(
            windows,
            vec![NhmmerCandidateWindow {
                n: 3,
                length: 3,
                k: 3
            }]
        );
    }

    #[test]
    fn nhmmer_reverse_complements_dna_seed() {
        assert_eq!(reverse_complement_dna_seed(b"AAGT"), b"ACTT");
    }

    #[test]
    fn nhmmer_threshold_config_preserves_model_cutoff_thresholds() {
        let mut outer = Pipeline::new();
        outer.use_bit_cutoffs = hmmer_pure_rs::pipeline::BitCutoff::GA;
        outer.by_e = false;
        outer.dom_by_e = false;
        outer.inc_by_e = false;
        outer.incdom_by_e = false;
        outer.t = Some(10.0);
        outer.dom_t = Some(8.0);
        outer.inc_t = Some(10.0);
        outer.inc_dom_t = Some(8.0);

        let mut window = Pipeline::new();
        NhmmerThresholdConfig::from_pipeline(&outer).apply_to(&mut window);

        assert_eq!(
            window.use_bit_cutoffs,
            hmmer_pure_rs::pipeline::BitCutoff::GA
        );
        assert!(!window.by_e);
        assert!(!window.dom_by_e);
        assert!(!window.inc_by_e);
        assert!(!window.incdom_by_e);
        assert_eq!(window.t, Some(10.0));
        assert_eq!(window.dom_t, Some(8.0));
        assert_eq!(window.inc_t, Some(10.0));
        assert_eq!(window.inc_dom_t, Some(8.0));
    }

    fn test_longtarget_domain(reported: bool, included: bool) -> hmmer_pure_rs::tophits::Domain {
        hmmer_pure_rs::tophits::Domain {
            iali: 1,
            jali: 8,
            ienv: 1,
            jenv: 8,
            bitscore: 10.0,
            lnp: -10.0,
            dombias: 1.0,
            oasc: 0.0,
            envsc: 0.0,
            domcorrection: 0.0,
            is_reported: reported,
            is_included: included,
            ad: None,
        }
    }

    #[test]
    fn nhmmer_longtarget_split_uses_per_domain_bit_cutoff_flags() {
        use hmmer_pure_rs::tophits::{Hit, P7_IS_DROPPED, P7_IS_INCLUDED, P7_IS_REPORTED};

        let mut sq = Sequence::new();
        sq.name = "target".to_string();
        sq.n = 100;
        sq.l = 100;

        let src_hit = Hit {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            n: 80,
            sortkey: -1.0,
            score: 20.0,
            bias: 0.0,
            pre_score: 20.0,
            sum_score: 20.0,
            lnp: -20.0,
            pre_lnp: -20.0,
            sum_lnp: -20.0,
            nexpected: 2.0,
            nregions: 1,
            nclustered: 1,
            noverlaps: 0,
            nenvelopes: 2,
            ndom: 2,
            nreported: 1,
            nincluded: 1,
            dcl: Vec::new(),
            flags: P7_IS_REPORTED | P7_IS_INCLUDED,
            seqidx: 7,
            subseq_start: 0,
        };
        let mut cutoff_config = NhmmerThresholdConfig::from_pipeline(&Pipeline::new());
        cutoff_config.use_bit_cutoffs = hmmer_pure_rs::pipeline::BitCutoff::GA;
        cutoff_config.t = Some(10.0);
        cutoff_config.inc_t = Some(10.0);

        let failed_domain_hit = longtarget_domain_hit(
            &sq,
            &src_hit,
            test_longtarget_domain(false, false),
            cutoff_config,
        );

        assert_eq!(failed_domain_hit.flags & P7_IS_REPORTED, 0);
        assert_eq!(failed_domain_hit.flags & P7_IS_INCLUDED, 0);
        assert!(failed_domain_hit.flags & P7_IS_DROPPED != 0);
        assert_eq!(failed_domain_hit.nreported, 0);
        assert_eq!(failed_domain_hit.nincluded, 0);

        let included_domain_hit = longtarget_domain_hit(
            &sq,
            &src_hit,
            test_longtarget_domain(true, true),
            cutoff_config,
        );

        assert!(included_domain_hit.flags & P7_IS_REPORTED != 0);
        assert!(included_domain_hit.flags & P7_IS_INCLUDED != 0);
        assert_eq!(included_domain_hit.flags & P7_IS_DROPPED, 0);
        assert_eq!(included_domain_hit.nreported, 1);
        assert_eq!(included_domain_hit.nincluded, 1);

        cutoff_config.t = Some(11.0);
        cutoff_config.inc_t = Some(11.0);
        let target_cutoff_failed_hit = longtarget_domain_hit(
            &sq,
            &src_hit,
            test_longtarget_domain(true, true),
            cutoff_config,
        );
        assert_eq!(target_cutoff_failed_hit.flags & P7_IS_REPORTED, 0);
        assert_eq!(target_cutoff_failed_hit.flags & P7_IS_INCLUDED, 0);
        assert!(target_cutoff_failed_hit.flags & P7_IS_DROPPED != 0);
        assert_eq!(target_cutoff_failed_hit.nreported, 0);
        assert_eq!(target_cutoff_failed_hit.nincluded, 0);
    }

    #[test]
    fn nhmmer_fm_empty_window_list_does_not_fallback_to_full_sequence_scan() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let mut seq = Sequence::new();
        seq.name = "target".to_string();
        seq.dsq = abc.digitize(b"ACGTACGTACGT");
        seq.n = 12;
        seq.l = 12;

        let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
        hmm.name = "query".to_string();
        hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);
        for node in 1..=hmm.m {
            for x in 0..abc.k {
                hmm.mat[node][x] = 0.01;
            }
            let ch = hmm.consensus.as_ref().unwrap()[node];
            let code = abc.inmap[ch as usize] as usize;
            hmm.mat[node][code] = 0.97;
        }

        let mut bg = Bg::new(&abc);
        bg.set_filter(hmm.m, &hmm.compo);
        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);
        let om = OProfile::convert(&gm);
        let threshold_config = NhmmerThresholdConfig::from_pipeline(&Pipeline::new());
        let msv_counter = std::sync::atomic::AtomicU64::new(0);
        let bias_counter = std::sync::atomic::AtomicU64::new(0);
        let vit_counter = std::sync::atomic::AtomicU64::new(0);
        let fwd_counter = std::sync::atomic::AtomicU64::new(0);

        let hits = search_longtarget(
            &seq,
            &hmm,
            &gm,
            &om,
            &bg,
            20,
            0.02,
            0.003,
            0.00003,
            false,
            false,
            false,
            42,
            NhmmerBiasWindowLengths::default(),
            threshold_config,
            false,
            &msv_counter,
            &bias_counter,
            &vit_counter,
            &fwd_counter,
            Some(&[]),
        );

        assert!(
            hits.is_empty(),
            "an explicit empty FM window list should not trigger reconstructed full-sequence scanning"
        );
    }

    #[test]
    fn nhmmer_longtarget_dispatch_fails_loudly_without_sse2_acceleration() {
        let err = ensure_nhmmer_longtarget_accel_available_with(false).unwrap_err();

        assert!(err.contains("x86_64/SSE2"));
    }
}

/// Search a single target sequence (one strand) and return its hits.
///
/// Counterpart to one strand-pass inside C `serial_loop`
/// (`hmmer/src/nhmmer.c:1330`), which calls `p7_Pipeline_LongTarget` on every
/// sequence regardless of length. Dispatches to `search_longtarget` when the
/// SSE2 SIMD path is available; fails loudly otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search_sequence(
    sq: &Sequence,
    hmm: &hmmer_pure_rs::hmm::Hmm,
    gm: &Profile,
    om: &OProfile,
    bg: &Bg,
    max_length: i32,
    f1: f64,
    f2: f64,
    f3: f64,
    do_max: bool,
    nobias: bool,
    nonull2: bool,
    seed: u32,
    bias_windows: NhmmerBiasWindowLengths,
    threshold_config: NhmmerThresholdConfig,
    is_complement: bool,
    msv_counter: &std::sync::atomic::AtomicU64,
    bias_counter: &std::sync::atomic::AtomicU64,
    vit_counter: &std::sync::atomic::AtomicU64,
    fwd_counter: &std::sync::atomic::AtomicU64,
    initial_windows: Option<&[NhmmerCandidateWindow]>,
) -> Result<Vec<hmmer_pure_rs::tophits::Hit>, String> {
    // Always use longtarget path: C HMMER's nhmmer calls p7_Pipeline_LongTarget
    // for every sequence regardless of length.
    ensure_nhmmer_longtarget_accel_available()?;
    #[cfg(target_arch = "x86_64")]
    {
        return Ok(search_longtarget(
            sq,
            hmm,
            gm,
            om,
            bg,
            max_length,
            f1,
            f2,
            f3,
            do_max,
            nobias,
            nonull2,
            seed,
            bias_windows,
            threshold_config,
            is_complement,
            msv_counter,
            bias_counter,
            vit_counter,
            fwd_counter,
            initial_windows,
        ));
    }
    #[allow(unreachable_code)]
    Err(nhmmer_longtarget_unsupported_message())
}

#[cfg(target_arch = "x86_64")]
fn c_style_blocked_msv_residue_count(
    sq: &Sequence,
    hmm: &hmmer_pure_rs::hmm::Hmm,
    om: &OProfile,
    bg: &Bg,
    max_length: i32,
    f1: f64,
    block_length: usize,
) -> u64 {
    if sq.n <= block_length {
        return msv_residue_count_for_longtarget_window(sq, hmm, om, bg, max_length, f1);
    }

    let overlap = max_length.max(0) as usize;
    let mut total = 0_u64;
    let mut start = 1_usize;
    loop {
        let end = (start + block_length - 1).min(sq.n);
        let mut block_dsq = Vec::with_capacity(end - start + 3);
        block_dsq.push(DSQ_SENTINEL);
        block_dsq.extend_from_slice(&sq.dsq[start..=end]);
        block_dsq.push(DSQ_SENTINEL);
        let block_sq = Sequence {
            name: sq.name.clone(),
            acc: sq.acc.clone(),
            desc: sq.desc.clone(),
            n: end - start + 1,
            l: end - start + 1,
            dsq: block_dsq,
            taxid: -1,
        };
        total += msv_residue_count_for_longtarget_window(&block_sq, hmm, om, bg, max_length, f1);

        if end == sq.n {
            break;
        }
        start = if overlap == 0 {
            end + 1
        } else {
            end.saturating_sub(overlap).saturating_add(1).max(1)
        };
    }
    total
}

#[cfg(target_arch = "x86_64")]
fn msv_residue_count_for_longtarget_window(
    sq: &Sequence,
    hmm: &hmmer_pure_rs::hmm::Hmm,
    om: &OProfile,
    bg: &Bg,
    max_length: i32,
    f1: f64,
) -> u64 {
    use hmmer_pure_rs::simd::ssv_longtarget;

    let mut om_ssv = om.clone();
    om_ssv.reconfig_msv_length(max_length);
    let om = &om_ssv;

    let mut windows =
        unsafe { ssv_longtarget::ssv_filter_longtarget(&sq.dsq, sq.n, om, bg, f1, max_length) };
    if windows.is_empty() {
        return 0;
    }

    let ml = if max_length > 0 {
        max_length as usize
    } else {
        hmm.m * 4
    };
    let (prefix_lens, suffix_lens) = ssv_longtarget::compute_prefix_suffix_lengths_from_om(om);
    ssv_longtarget::extend_and_merge_windows_with_scoredata(
        &mut windows,
        ml,
        sq.n,
        0.0,
        &prefix_lens,
        &suffix_lens,
    );

    let mut total = 0_u64;
    for win in &windows {
        let win_start = win.n;
        let win_end = (win.n + win.length - 1).min(sq.n);
        if win_end < win_start || (win_end - win_start + 1) < hmm.m {
            continue;
        }
        let win_len = win_end - win_start + 1;
        let mut sub_dsq = vec![DSQ_SENTINEL];
        sub_dsq.extend_from_slice(&sq.dsq[win_start..=win_end]);
        sub_dsq.push(DSQ_SENTINEL);

        let mut msv_om = om.clone();
        msv_om.reconfig_msv_length(win_len as i32);
        let mut bg_win = bg.clone();
        bg_win.set_length(win_len);
        let nullsc = bg_win.null_one(win_len);
        let usc_result =
            unsafe { hmmer_pure_rs::simd::msv_filter::msv_filter(&sub_dsq, win_len, &msv_om) };
        let usc = match usc_result {
            hmmer_pure_rs::simd::msv_filter::MsvResult::Ok(sc) => sc,
            hmmer_pure_rs::simd::msv_filter::MsvResult::Overflow => f32::INFINITY,
        };
        let msv_pval = hmmer_pure_rs::pipeline::msv_pvalue(usc, nullsc, &hmm.evparam);
        if msv_pval <= f1 {
            total += win_len as u64;
        }
    }
    total
}

fn nhmmer_longtarget_accel_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sse2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn nhmmer_longtarget_unsupported_message() -> String {
    "nhmmer/nhmmscan long-target heuristic filters require x86_64/SSE2; use an x86_64/SSE2 build"
        .to_string()
}

fn ensure_nhmmer_longtarget_accel_available() -> Result<(), String> {
    ensure_nhmmer_longtarget_accel_available_with(nhmmer_longtarget_accel_available())
}

fn ensure_nhmmer_longtarget_accel_available_with(available: bool) -> Result<(), String> {
    if available {
        Ok(())
    } else {
        Err(nhmmer_longtarget_unsupported_message())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct NhmmerThresholdConfig {
    use_bit_cutoffs: hmmer_pure_rs::pipeline::BitCutoff,
    by_e: bool,
    dom_by_e: bool,
    inc_by_e: bool,
    incdom_by_e: bool,
    e_value_threshold: f64,
    dom_e_value_threshold: f64,
    inc_e: f64,
    inc_dome: f64,
    t: Option<f64>,
    dom_t: Option<f64>,
    inc_t: Option<f64>,
    inc_dom_t: Option<f64>,
}

impl NhmmerThresholdConfig {
    pub(crate) fn from_pipeline(pli: &Pipeline) -> Self {
        Self {
            use_bit_cutoffs: pli.use_bit_cutoffs,
            by_e: pli.by_e,
            dom_by_e: pli.dom_by_e,
            inc_by_e: pli.inc_by_e,
            incdom_by_e: pli.incdom_by_e,
            e_value_threshold: pli.e_value_threshold,
            dom_e_value_threshold: pli.dom_e_value_threshold,
            inc_e: pli.inc_e,
            inc_dome: pli.inc_dome,
            t: pli.t,
            dom_t: pli.dom_t,
            inc_t: pli.inc_t,
            inc_dom_t: pli.inc_dom_t,
        }
    }

    fn apply_to(self, pli: &mut Pipeline) {
        pli.use_bit_cutoffs = self.use_bit_cutoffs;
        pli.by_e = self.by_e;
        pli.dom_by_e = self.dom_by_e;
        pli.inc_by_e = self.inc_by_e;
        pli.incdom_by_e = self.incdom_by_e;
        pli.e_value_threshold = self.e_value_threshold;
        pli.dom_e_value_threshold = self.dom_e_value_threshold;
        pli.inc_e = self.inc_e;
        pli.inc_dome = self.inc_dome;
        pli.t = self.t;
        pli.dom_t = self.dom_t;
        pli.inc_t = self.inc_t;
        pli.inc_dom_t = self.inc_dom_t;
    }
}

/// Search one strand of a target sequence using the long-target SSV pipeline.
///
/// Mirrors C `p7_Pipeline_LongTarget` (`hmmer/src/p7_pipeline.c:1763`) end to
/// end: (1) SSV longtarget scan finds candidate windows; (2) windows are
/// extended/merged using per-k prefix/suffix lengths; (3) per-window MSV F1
/// gate; (4) post-SSV bias and Viterbi F2 gates (`p7_pli_postSSV_LongTarget`)
/// produce finer vit sub-windows; (5) long windows (>80 kb) are split with
/// 40 kb overlap for numerical stability; (6) Forward+null2+domain definition
/// runs per window via `Pipeline::run`, producing one `Hit` per emitted
/// domain. Maintains the residue counters (msv/bias/vit/fwd) and the
/// inter-window `fwd_overlap` carry exactly like C. Returned coordinates are
/// in the strand-local (input `sq`) frame; the caller flips them for crick.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn search_longtarget(
    sq: &Sequence,
    hmm: &hmmer_pure_rs::hmm::Hmm,
    gm: &Profile,
    om: &OProfile,
    bg: &Bg,
    max_length: i32,
    f1: f64,
    f2: f64,
    f3: f64,
    do_max: bool,
    nobias: bool,
    nonull2: bool,
    seed: u32,
    bias_windows: NhmmerBiasWindowLengths,
    threshold_config: NhmmerThresholdConfig,
    is_complement: bool,
    msv_counter: &std::sync::atomic::AtomicU64,
    bias_counter: &std::sync::atomic::AtomicU64,
    vit_counter: &std::sync::atomic::AtomicU64,
    fwd_counter: &std::sync::atomic::AtomicU64,
    initial_windows: Option<&[NhmmerCandidateWindow]>,
) -> Vec<hmmer_pure_rs::tophits::Hit> {
    use hmmer_pure_rs::simd::ssv_longtarget;

    let _ = bias_windows.b3;

    // Match C p7_Pipeline_LongTarget (p7_pipeline.c:1763): reconfigure MSV
    // length to max_length BEFORE SSV scan. This sets om->tjb_b based on
    // max_length (e.g. 253 for tRNA), not the full sequence length. Without
    // this, Rust's SSV sc_thresh is too high and misses weak peaks that C
    // catches.
    let mut om_ssv = om.clone();
    om_ssv.reconfig_msv_length(max_length);
    let om = &om_ssv;

    // Phase 1: SSV longtarget filter — find candidate windows.
    let mut windows = if let Some(seed_windows) = initial_windows {
        seed_windows
            .iter()
            .map(|w| ssv_longtarget::HmmWindow {
                n: w.n,
                k: w.k,
                length: w.length,
                score: 0.0,
                target_len: sq.n,
                complement: is_complement,
                // Per-segment FM path: each searched `sq` IS one FM segment, so
                // id is constant within this call and windows are already in
                // segment-local (RC-local for crick) coordinates. id=0 (single
                // segment per call) and fm_n=-1 (no concatenated-FM frame, so
                // no complement extension flip — see HmmWindow docs).
                id: 0,
                fm_n: -1,
            })
            .collect()
    } else {
        unsafe { ssv_longtarget::ssv_filter_longtarget(&sq.dsq, sq.n, om, bg, f1, max_length) }
    };

    // C p7_Pipeline_LongTarget has NO do_max special-case here: --max sets
    // F1=0.3, F2=F3=1.0 (p7_pipeline.c:355-356) and runs the identical SSV
    // windowing + MSV/Vit/Fwd pipeline. The F2/F3=1.0 gates pass everything,
    // but the F1=0.3 SSV gate still segments windows, and the per-stage
    // counters are still credited. (The previous Rust code collapsed --max to a
    // single full-sequence window and skipped the stages, which both lost C's
    // multi-window hits and left pos_past_msv/bias/vit at 0.)
    if windows.is_empty() {
        return Vec::new();
    }

    // Extend and merge overlapping windows. C uses pct_overlap = 0 for the
    // msv (SSV) stage (any overlap merges) and 0.5 for the vit stage
    // (hmmer/src/p7_pipeline.c:1794 and 1614). We also use per-k
    // prefix_lengths / suffix_lengths from p7_hmm_ScoreDataComputeRest to
    // compute window-specific extension sizes.
    let ml = if max_length > 0 {
        max_length as usize
    } else {
        hmm.m * 4
    };
    #[cfg(target_arch = "x86_64")]
    let (prefix_lens, suffix_lens) = ssv_longtarget::compute_prefix_suffix_lengths_from_om(om);
    #[cfg(not(target_arch = "x86_64"))]
    let (prefix_lens, suffix_lens) = ssv_longtarget::compute_prefix_suffix_lengths(hmm);
    {
        ssv_longtarget::extend_and_merge_windows_with_scoredata(
            &mut windows,
            ml,
            sq.n,
            0.0,
            &prefix_lens,
            &suffix_lens,
        );
    }

    // Phase 1a: Per-window standard MSV filter (F1 gate). Mirrors C
    // p7_pipeline.c:1862-1866 which calls p7_oprofile_ReconfigMSVLength(om,
    // window->length), p7_MSVFilter, then `if (P > pli->F1) continue;`. This
    // rejects SSV peaks that don't also pass a full MSV p-value threshold.
    // Without this, Rust finds SSV peaks that C rejects at this stage.
    let f1_thresh = f1;
    {
        let mut msv_filtered: Vec<ssv_longtarget::HmmWindow> = Vec::new();
        for win in &windows {
            let win_start = win.n;
            let win_end = (win.n + win.length - 1).min(sq.n);
            if win_end < win_start || (win_end - win_start + 1) < hmm.m {
                continue;
            }
            let win_len = win_end - win_start + 1;
            let mut sub_dsq = vec![DSQ_SENTINEL];
            sub_dsq.extend_from_slice(&sq.dsq[win_start..=win_end]);
            sub_dsq.push(DSQ_SENTINEL);

            // Reconfig MSV length for this window, compute null, run MSVFilter,
            // check p-value against F1.
            let mut msv_om = om.clone();
            msv_om.reconfig_msv_length(win_len as i32);
            let mut bg_win = bg.clone();
            bg_win.set_length(win_len);
            let nullsc = bg_win.null_one(win_len);
            let usc_result =
                unsafe { hmmer_pure_rs::simd::msv_filter::msv_filter(&sub_dsq, win_len, &msv_om) };
            let usc = match usc_result {
                hmmer_pure_rs::simd::msv_filter::MsvResult::Ok(sc) => sc,
                hmmer_pure_rs::simd::msv_filter::MsvResult::Overflow => f32::INFINITY,
            };
            let msv_pval = hmmer_pure_rs::pipeline::msv_pvalue(usc, nullsc, &hmm.evparam);
            if msv_pval > f1_thresh {
                continue;
            }
            // Mirror C p7_pipeline.c:1867 — accumulate residues after MSV F1 gate.
            use std::sync::atomic::Ordering;
            msv_counter.fetch_add(win_len as u64, Ordering::Relaxed);
            // Store the MSV score (usc) on the window so postSSV's F1-bias gate
            // can use it (C passes `usc` to p7_pli_postSSV_LongTarget).
            let mut filtered = win.clone();
            filtered.score = usc;
            msv_filtered.push(filtered);
        }
        windows = msv_filtered;
    }

    // Phase 1b: Vit_longtarget per merged SSV window to produce finer-grained
    // vit sub-windows. Mirrors C p7_pli_postSSV_LongTarget → ViterbiFilter_longtarget.
    // For each SSV window, we run Vit_longtarget on its subseq to find positions
    // where the E-state Vit score exceeds the F2 threshold; those positions get
    // re-extended and merged before going to Forward.
    let f2_p_thresh = f2;
    let max_length_usize = if max_length > 0 {
        max_length as usize
    } else {
        hmm.max_length.max((hmm.m * 4) as i32) as usize
    };
    let mut vit_windows: Vec<ssv_longtarget::HmmWindow> = Vec::new();
    {
        for msv in &windows {
            let msv_start = msv.n;
            let msv_end = (msv.n + msv.length - 1).min(sq.n);
            if msv_end < msv_start || (msv_end - msv_start + 1) < hmm.m {
                continue;
            }
            let subseq_len = msv_end - msv_start + 1;
            let mut vit_sub_dsq = vec![DSQ_SENTINEL];
            vit_sub_dsq.extend_from_slice(&sq.dsq[msv_start..=msv_end]);
            vit_sub_dsq.push(DSQ_SENTINEL);

            // Match C p7_pli_postSSV_LongTarget exactly (p7_pipeline.c:1580-1612):
            //   (a) bias filter on full window_len, with F1_L=min(100,win_len) scaling
            //   (b) compute loc_window_len = min(win_len, max_length), recompute nullsc
            //   (c) F2 filtersc = nullsc + bias_delta * F2_L/win_len (F2_L=min(240,win_len))
            //   (d) reconfig profile to loc_window_len
            //   (e) run Vit_longtarget with this filtersc
            // nhmmer-specific defaults from hmmer/src/nhmmer.c:160-162.
            // The generic p7_pipeline defaults are 100/240/1000 (p7_pipeline.c:341-344),
            // but nhmmer overrides `--B1` to 110.
            let f1_l = subseq_len.min(bias_windows.b1);
            let f2_l = subseq_len.min(bias_windows.b2);
            let loc_window_len = subseq_len.min(max_length_usize);

            // Compute bias_delta once (based on full win_len): bias_delta = filter_score - nullsc_win.
            let mut bg_win = bg.clone();
            bg_win.set_length(subseq_len);
            let nullsc_full = bg_win.null_one(subseq_len);
            let bias_delta = if !nobias {
                bg_win.filter_score(&vit_sub_dsq, subseq_len) - nullsc_full
            } else {
                0.0
            };

            // Mirror C p7_pipeline.c:1580-1587 bias F1 gate. Uses the MSV score
            // (usc) stored on the window above. If bias-corrected MSV p-value
            // exceeds F1 = 0.02, skip this SSV window.
            if !nobias {
                let f1_scale = if f1_l == subseq_len {
                    1.0_f32
                } else {
                    f1_l as f32 / subseq_len as f32
                };
                let filtersc_f1 = nullsc_full + bias_delta * f1_scale;
                let usc = msv.score;
                let bias_pval = hmmer_pure_rs::pipeline::msv_pvalue(usc, filtersc_f1, &hmm.evparam);
                if bias_pval > f1_thresh {
                    continue;
                }
            }
            // After bias F1 gate: count these residues toward pos_past_bias.
            use std::sync::atomic::Ordering;
            bias_counter.fetch_add(subseq_len as u64, Ordering::Relaxed);

            // Recompute nullsc at loc_window_len (may be shorter than win_len).
            let mut bg_loc = bg.clone();
            bg_loc.set_length(loc_window_len);
            let nullsc_win = bg_loc.null_one(loc_window_len);

            // F2 filtersc: scale bias_delta by F2_L/win_len, add nullsc_loc.
            let f2_scale = if f2_l == subseq_len {
                1.0_f32
            } else {
                f2_l as f32 / subseq_len as f32
            };
            let filtersc = nullsc_win + bias_delta * f2_scale;

            // Reconfig profile to loc_window_len (matches C p7_pipeline.c:1606).
            let mut vit_om = om.clone();
            vit_om.reconfig_length(loc_window_len as i32);
            let mut sub_windows = unsafe {
                hmmer_pure_rs::simd::vit_filter::viterbi_filter_longtarget(
                    &vit_sub_dsq,
                    subseq_len,
                    &vit_om,
                    filtersc,
                    f2_p_thresh,
                )
            };
            // Extend and merge Vit windows WITHIN the subseq (target_len =
            // subseq_len), matching C p7_pipeline.c:1614 which does
            // `p7_pli_ExtendAndMergeWindows(om, data, vit_windowlist, 0.5)` with
            // vit_windowlist entries having target_len=window_len (subseq_len).
            // This bounds extension to subseq bounds; without it, extension
            // could reach beyond the SSV window.
            ssv_longtarget::extend_and_merge_windows_with_scoredata(
                &mut sub_windows,
                ml,
                subseq_len,
                0.5,
                &prefix_lens,
                &suffix_lens,
            );
            // Mirror C p7_pipeline.c:1637-1641: pos_past_vit += length, subtract
            // overlap with previous vit window. C does NOT clamp the net
            // contribution to 0 — if overlap > length, pos_past_vit can receive
            // a net-negative delta. Rust mirrors this via wrapping u64 add so
            // parallel aggregation still gives the same final value.
            let mut prev_end: Option<usize> = None;
            for w in sub_windows {
                let abs_n = msv_start + w.n - 1;
                let mut add = w.length as i64;
                if let Some(pe) = prev_end {
                    if pe > abs_n {
                        add -= (pe - abs_n) as i64;
                    }
                }
                vit_counter.fetch_add(add as u64, Ordering::Relaxed);
                prev_end = Some(abs_n + w.length);
                vit_windows.push(ssv_longtarget::HmmWindow {
                    n: abs_n,
                    k: w.k,
                    length: w.length,
                    score: w.score,
                    target_len: sq.n,
                    complement: is_complement,
                    id: 0,
                    fm_n: -1,
                });
            }
        }

        // Vit windows are already extended+merged within each SSV subseq above
        // (matching C p7_pipeline.c:1614). Just aggregate them.
        windows = vit_windows;
        if windows.is_empty() {
            return Vec::new();
        }
    }

    // Window splitting for numerical stability in Forward: mirror C
    // p7_pli_postSSV_LongTarget (p7_pipeline.c:1620-1634). If a merged window
    // is longer than 80000 residues, split it into overlapping sub-windows of
    // length at most 80000 with overlap = min(40000, max_length).
    // LOW-1: split via the faithful do/while port (see
    // ssv_longtarget::split_long_windows; mirrors p7_pipeline.c:1620-1634).
    const MAX_WINDOW_LEN: usize = 80000;
    let overlap_len = 40000_usize.min(ml);
    windows = ssv_longtarget::split_long_windows(&windows, MAX_WINDOW_LEN, overlap_len);

    // Phase 2: Run full pipeline on each window
    let mut all_hits = Vec::new();
    // Track fwd overlap carried from previous F3-passing window, matching
    // C p7_pipeline.c:1335-1337 + 1649-1654.
    let mut fwd_overlap: usize = 0;

    for (win_idx, win) in windows.iter().enumerate() {
        let win_start = win.n;
        let win_end = (win.n + win.length - 1).min(sq.n);
        let win_len = win_end - win_start + 1;
        if win_len < hmm.m {
            fwd_overlap = 0;
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
            taxid: -1,
        };

        let mut lgm = gm.clone();
        let mut lom = om.clone();
        // Match C postViterbi_LongTarget (p7_pipeline.c:1316): SetLength(bg,
        // window_len) before NullOne. Without this, Pipeline::run's call to
        // `bg.null_one(l)` uses whatever `p1` the outer bg was initialized
        // with (default 350/351), instead of `window_len/(window_len+1)`.
        let mut lb = bg.clone();
        lb.set_length(win_len);
        let mut lpli = Pipeline::new();
        lpli.new_model(&lgm);
        lpli.long_target = true;
        threshold_config.apply_to(&mut lpli);
        // Propagate top-level F3 threshold (nhmmer default 3e-5, not pipeline
        // default 1e-5). Matches C nhmmer.c:114.
        lpli.f3 = f3;
        lpli.do_max = do_max;
        lpli.seed = seed;
        // Propagate --nobias/--nonull2 so the long-target bias scaling and
        // null2 correction inside Pipeline::run honor the user's flags.
        // Without this, Pipeline::new()'s defaults (do_biasfilter=true,
        // do_null2=true) override the user's intent.
        lpli.do_biasfilter = !nobias;
        lpli.do_null2 = !nonull2;
        lpli.long_target_b3 = bias_windows.b3;
        // SSV already ran; still apply Viterbi (F2) and Forward (F3) filters
        // per window so we don't emit one Hit for every SSV candidate.

        let mut lth = TopHits::new();
        let ran = lpli.run(&mut lgm, &mut lom, &lb, hmm, &win_sq, &mut lth);
        // msv/bias/vit counters already updated upstream. fwd_counter
        // += win_len - fwd_overlap (C p7_pipeline.c:1335). On F3 pass, carry
        // overlap with NEXT window; on F3 fail, reset fwd_overlap to 0.
        use std::sync::atomic::Ordering;
        if lpli.n_past_fwd > 0 {
            // C p7_pipeline.c:1335 `pli->pos_past_fwd += window_len - *overlap;`
            // does not clamp to zero. Use wrapping u64 add to allow net-negative
            // contributions, matching C's counter arithmetic exactly.
            let add = (win_len as i64) - (fwd_overlap as i64);
            fwd_counter.fetch_add(add as u64, Ordering::Relaxed);
            // Compute overlap with next window if any.
            if win_idx + 1 < windows.len() {
                let next = &windows[win_idx + 1];
                let this_end = win.n + win.length;
                fwd_overlap = this_end.saturating_sub(next.n);
            } else {
                fwd_overlap = 0;
            }
        } else {
            fwd_overlap = 0;
        }
        if ran {
            // nhmmer (long_target) creates one Hit per domain (C postViterbi
            // line 1433: loop `for (d = 0; d < ddef->ndom; d++)` creates one
            // hit per domain). Our Pipeline::run emits one hit per window
            // with all domains stuffed into hit.dcl, so we split here.
            for src_hit in lth.hits {
                for src_dom in &src_hit.dcl {
                    let mut dom = src_dom.clone();
                    // Adjust coordinates to global sequence position.
                    dom.iali += (win_start - 1) as i64;
                    dom.jali += (win_start - 1) as i64;
                    dom.ienv += (win_start - 1) as i64;
                    dom.jenv += (win_start - 1) as i64;
                    if let Some(ref mut ad) = dom.ad {
                        ad.sqfrom += win_start - 1;
                        ad.sqto += win_start - 1;
                    }
                    all_hits.push(longtarget_domain_hit(sq, &src_hit, dom, threshold_config));
                }
            }
        }
    }

    all_hits
}

fn longtarget_domain_hit(
    sq: &Sequence,
    src_hit: &hmmer_pure_rs::tophits::Hit,
    dom: hmmer_pure_rs::tophits::Domain,
    threshold_config: NhmmerThresholdConfig,
) -> hmmer_pure_rs::tophits::Hit {
    let score = dom.bitscore;
    // C prints the raw (occasionally negative) bias correction in
    // tblout/domtblout and the human tables (p7_tophits.c:1664, unclamped);
    // do not clamp to 0.0. Matches the pipeline.rs / domaindef.rs unclamping.
    let bias = dom.dombias;
    let mut flags = src_hit.flags;
    let mut nreported = 1;
    let mut nincluded = 1;
    if threshold_config.use_bit_cutoffs != hmmer_pure_rs::pipeline::BitCutoff::None {
        flags &= !(hmmer_pure_rs::tophits::P7_IS_REPORTED
            | hmmer_pure_rs::tophits::P7_IS_INCLUDED
            | hmmer_pure_rs::tophits::P7_IS_DROPPED);
        let target_reported = threshold_config
            .t
            .map(|threshold| score as f64 >= threshold)
            .unwrap_or(src_hit.flags & hmmer_pure_rs::tophits::P7_IS_REPORTED != 0);
        let target_included = threshold_config
            .inc_t
            .map(|threshold| score as f64 >= threshold)
            .unwrap_or(src_hit.flags & hmmer_pure_rs::tophits::P7_IS_INCLUDED != 0);
        let reported = dom.is_reported && target_reported;
        let included = dom.is_included && reported && target_included;
        nreported = usize::from(reported);
        nincluded = usize::from(included);
        if reported {
            flags |= hmmer_pure_rs::tophits::P7_IS_REPORTED;
        } else {
            flags |= hmmer_pure_rs::tophits::P7_IS_DROPPED;
        }
        if included {
            flags |= hmmer_pure_rs::tophits::P7_IS_INCLUDED;
        }
    }

    hmmer_pure_rs::tophits::Hit {
        name: sq.name.clone(),
        acc: sq.acc.clone(),
        desc: sq.desc.clone(),
        n: sq.n,
        score,
        bias,
        pre_score: score + bias,
        sum_score: score,
        lnp: dom.lnp,
        pre_lnp: dom.lnp,
        sum_lnp: dom.lnp,
        sortkey: dom.lnp,
        nexpected: src_hit.nexpected,
        nregions: src_hit.nregions,
        nclustered: src_hit.nclustered,
        noverlaps: src_hit.noverlaps,
        nenvelopes: src_hit.nenvelopes,
        ndom: 1,
        nreported,
        nincluded,
        flags,
        seqidx: src_hit.seqidx,
        subseq_start: src_hit.subseq_start,
        dcl: vec![dom],
    }
}
