//! nhmmer — search DNA/RNA HMM(s) against a nucleotide sequence database.
//! Uses SSV long-target filter for genome-scale sequences, matching C HMMER's nhmmer.

use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

use clap::Parser;

use hmmer_pure_rs::alphabet::{Alphabet, DSQ_SENTINEL};
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::sequence::{self, Sequence};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::TopHits;
use hmmer_pure_rs::{hmmfile, hmmfile_binary};

extern "C" {
    #[link_name = "log"]
    fn c_log(x: f64) -> f64;
}

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

    /// Save hits to Dfam-style tabular file
    #[arg(long = "dfamtblout")]
    dfamtblout: Option<PathBuf>,

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
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

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
    #[arg(long = "w_length")]
    w_length: Option<usize>,

    /// Assert target sequence file format (not implemented; auto-detection is used)
    #[arg(long = "tformat")]
    tformat: Option<String>,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,
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

/// Write C-nhmmer-style per-hit tabular output (`--tblout`).
///
/// Mirrors the `long_targets=TRUE` branch of `p7_tophits_TabularTargets` in
/// `hmmer/src/p7_tophits.c:1610`. Emits the header trio, one row per reported
/// hit (target/query name+acc, hmm/ali/env coords, seq length, strand,
/// E-value, score, bias, description), then a footer block with program /
/// version / query file / target file / option settings / cwd / date.
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
) {
    use hmmer_pure_rs::tophits::P7_IS_REPORTED;
    // Column widths taken from C's format string (static minimum 20/10/20/10/7/7/...).
    let namew = th
        .hits
        .iter()
        .filter(|h| h.flags & P7_IS_REPORTED != 0)
        .map(|h| h.name.len())
        .max()
        .unwrap_or(20)
        .max(20);
    let qw = qname.len().max(20);

    // Mirror C p7_tophits.c:1623: `#%-*s ...` where width=namew-1, value has
    // leading space (` target name`). Net effect: `#` + 19-char field = 20.
    if show_header {
        writeln!(
            f,
            "#{tname:<tnw$} {tacc:<10} {qn:<qnw$} {qacc:<10} {hf} {ht} {af:>7} {at:>7} {ef:>7} {et:>7} {sl:>7} {strand:>6} {ev:>9} {sc:>6} {bi:>5}  {desc}",
            tname = " target name",
            tacc = "accession",
            qn = "query name",
            qacc = "accession",
            hf = "hmmfrom",
            ht = "hmm to",
            af = "alifrom",
            at = " ali to",
            ef = "envfrom",
            et = " env to",
            sl = " sq len",
            strand = "strand",
            ev = "  E-value",
            sc = " score",
            bi = " bias",
            desc = "description of target",
            tnw = namew - 1,
            qnw = qw,
        )
        .unwrap();
        writeln!(
            f,
            "#{tname:-<tnw$} {tacc:-<10} {qn:-<qnw$} {qacc:-<10} {hf} {ht} {af:->7} {at:->7} {ef:->7} {et:->7} {sl:->7} {strand:->6} {ev:->9} {sc:->6} {bi:->5} {desc:-<21}",
            tname = "",
            tacc = "",
            qn = "",
            qacc = "",
            hf = "-------",
            ht = "-------",
            af = "",
            at = "",
            ef = "",
            et = "",
            sl = "",
            strand = "",
            ev = "",
            sc = "",
            bi = "",
            desc = "",
            tnw = namew - 1,
            qnw = qw,
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
        let ev_str = format_pct2g(hit.lnp.exp(), 9);
        writeln!(
            f,
            "{tname:<namew$} {tacc:<10} {qn:<qw$} {qacc:<10} {hf:>7} {ht:>7} {af:>7} {at:>7} {ef:>7} {et:>7} {sl:>7} {strand} {ev:>9} {sc:>6.1} {bi:>5.1}  {desc}",
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
            sc = hit.score,
            bi = best.dombias,
            desc = desc_display,
            namew = namew,
            qw = qw,
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
    let now = std::time::SystemTime::now();
    let date_str = format_date(now);
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
) {
    use hmmer_pure_rs::tophits::P7_IS_REPORTED;

    writeln!(f, "# hit scores").unwrap();
    writeln!(f, "# ----------").unwrap();
    writeln!(f, "#").unwrap();
    writeln!(
        f,
        "# target name         acc                  query name             bits   e-value  bias  hmm-st  hmm-en strand  ali-st  ali-en  env-st  env-en  {length_header}   description of target"
    )
    .unwrap();
    writeln!(
        f,
        "# ------------------- -------------------  -------------------  ------ --------- ----- ------- ------- ------ ------- ------- ------- ------- -------   ---------------------"
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
            "{:<21} {:<20} {:<20} {:>6.1} {:>9} {:>5.1} {:>7} {:>7} {} {:>7} {:>7} {:>7} {:>7} {:>7}   {}",
            hit.name,
            acc,
            qname,
            hit.score,
            format_pct2g(hit.lnp.exp(), 9),
            dom.dombias,
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

/// Format a `SystemTime` as a `ctime`-style `Day Mon DD HH:MM:SS YYYY` string
/// for the tblout footer's `# Date:` line.
fn format_date(t: std::time::SystemTime) -> String {
    use std::time::UNIX_EPOCH;
    let secs = t
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Very minimal date; mirror `Mon Day HH:MM:SS YYYY` shape. Using a rough
    // ctime-style format via time-of-day arithmetic.
    let (sec, min, hour, day, month, year) = broken_down_time(secs);
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    let dow = (((secs / 86400) + 4) % 7) as usize;
    format!(
        "{} {} {:>2} {:02}:{:02}:{:02} {}",
        days[dow],
        months[(month - 1) as usize],
        day,
        hour,
        min,
        sec,
        year
    )
}

/// Decompose Unix epoch seconds into `(sec, min, hour, day, month, year)` UTC.
///
/// Used by `format_date` to avoid a `chrono`/`time` dependency. Handles leap
/// years and is correct for years from 1970 forward.
fn broken_down_time(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let sec = (secs % 60) as u32;
    let min = ((secs / 60) % 60) as u32;
    let hour = ((secs / 3600) % 24) as u32;
    let mut days = (secs / 86400) as u64;
    let mut year: u32 = 1970;
    loop {
        let leap = is_leap(year);
        let yd = if leap { 366 } else { 365 };
        if days < yd {
            break;
        }
        days -= yd;
        year += 1;
    }
    let mdays = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month: u32 = 1;
    for &m in &mdays {
        if days < m {
            break;
        }
        days -= m;
        month += 1;
    }
    let day = (days + 1) as u32;
    (sec, min, hour, day, month, year)
}

/// Return true if `y` is a Gregorian leap year. Helper for `broken_down_time`.
fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Format a float like C `printf("%.3g", v)`: 3 significant digits, no padding.
///
/// Follows C stdio `%g` rules: trailing zeros stripped, scientific notation
/// when the magnitude falls outside `[1e-4, 1e3)`. Used in the pipeline
/// statistics summary and the `# Mc/sec:` line.
fn format_g3(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let abs = v.abs();
    let exp = abs.log10().floor() as i32;
    let scientific = exp < -4 || exp >= 3;
    if scientific {
        let mantissa = v / 10f64.powi(exp);
        let mant_rounded = (mantissa * 100.0).round() / 100.0;
        let carry = if mant_rounded.abs() >= 10.0 { 1 } else { 0 };
        let final_mant = mant_rounded / 10f64.powi(carry);
        let final_exp = exp + carry;
        let mant_str = strip_trailing_zeros(&format!("{}", final_mant));
        if final_exp >= 0 {
            format!("{}e+{:02}", mant_str, final_exp)
        } else {
            format!("{}e-{:02}", mant_str, -final_exp)
        }
    } else {
        let places = (3 - exp - 1).max(0) as usize;
        strip_trailing_zeros(&format!("{:.*}", places, v))
    }
}

/// Trim trailing zeros (and a dangling decimal point) from a fixed-point
/// numeric string — the C `%g` cleanup step shared by `format_g3` /
/// `format_pct2g`.
fn strip_trailing_zeros(s: &str) -> String {
    if let Some(dot) = s.find('.') {
        let mut end = s.len();
        while end > dot + 1 && s.as_bytes()[end - 1] == b'0' {
            end -= 1;
        }
        if end > 0 && s.as_bytes()[end - 1] == b'.' {
            end -= 1;
        }
        s[..end].to_string()
    } else {
        s.to_string()
    }
}

/// Format a float like C `printf("%*.2g", width, v)`: 2 significant digits,
/// right-aligned in `width` columns. Used for E-values in the per-hit and
/// per-domain tables. Switches to scientific when magnitude is outside
/// `[1e-4, 1e2)`.
fn format_pct2g(v: f64, width: usize) -> String {
    if v == 0.0 {
        return format!("{:>1$}", "0", width);
    }
    let abs = v.abs();
    let exp = abs.log10().floor() as i32;
    let scientific = exp < -4 || exp >= 2;
    let formatted = if scientific {
        let mantissa = v / 10f64.powi(exp);
        let mant_rounded = (mantissa * 10.0).round() / 10.0;
        let carry = if mant_rounded.abs() >= 10.0 { 1 } else { 0 };
        let final_mant = mant_rounded / 10f64.powi(carry);
        let final_exp = exp + carry;
        let mant_str = if final_mant.fract() == 0.0 {
            format!("{}", final_mant as i64)
        } else {
            format!("{}", final_mant)
        };
        if final_exp >= 0 {
            format!("{}e+{:02}", mant_str, final_exp)
        } else {
            format!("{}e-{:02}", mant_str, -final_exp)
        }
    } else {
        let places = (2 - exp - 1).max(0) as usize;
        // C %g strips trailing zeros after the decimal point.
        strip_trailing_zeros(&format!("{:.*}", places, v))
    };
    format!("{:>1$}", formatted, width)
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
    let args = Args::parse_from(&args);

    if let Some(ref format) = args.tformat {
        if format.eq_ignore_ascii_case("fasta") {
            // FASTA is already accepted by the sequence reader's auto-detection.
        } else {
            eprintln!(
                "nhmmer --tformat={} is not implemented: only FASTA target format assertions are supported",
            format
        );
            std::process::exit(1);
        }
    }
    if args.dna && args.rna {
        eprintln!("Error: options --dna and --rna are mutually exclusive");
        std::process::exit(1);
    }
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

    let hmms = read_hmms(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });
    if hmms.len() > 1 && args.seqdb == PathBuf::from("-") {
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
    // Mirror C nhmmer.c:341-342 — emit alphabet assertion when flag is used.
    if args.dna {
        writeln!(out, "# input query is asserted as:      DNA").unwrap();
    }
    if args.rna {
        writeln!(out, "# input query is asserted as:      RNA").unwrap();
    }
    writeln!(out, "# number of worker threads:        {}", args.cpu).unwrap();
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
    for (query_idx, hmm) in hmms.iter().enumerate() {
        let search_start = std::time::Instant::now();
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
            std::process::exit(1);
        }

        // Database size override (nhmmer uses Megabases)
        if let Some(z) = args.z_value {
            pli.z = z * 1_000_000.0; // convert Mb to bases
            pli.z_setby = hmmer_pure_rs::pipeline::ZSetBy::Option;
        }

        writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.m).unwrap();
        if let Some(ref acc) = hmm.acc {
            if !acc.is_empty() {
                writeln!(out, "Accession:   {}", acc).unwrap();
            }
        }
        if let Some(ref desc) = hmm.desc {
            if !desc.is_empty() {
                writeln!(out, "Description: {}", desc).unwrap();
            }
        }
        out.flush().unwrap_or_else(|e| {
            eprintln!("Error writing output: {}", e);
            std::process::exit(1);
        });

        // Read targets
        let mut sequences = Vec::new();
        {
            let mut sqf = open_target_seq_file(&args.seqdb, &abc, args.tformat.as_deref())
                .unwrap_or_else(|e| {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                });
            let mut sq = Sequence::new();
            while sqf.read(&mut sq).unwrap_or_else(|e| {
                eprintln!("Error reading sequence file: {}", e);
                std::process::exit(1);
            }) {
                sequences.push(sq.clone());
                sq.reuse();
            }
        }
        if sequences.is_empty() {
            eprintln!("Error: no sequences found in {}", args.seqdb.display());
            std::process::exit(1);
        }

        use rayon::prelude::*;

        let max_length = if let Some(w_length) = args.w_length {
            w_length as i32
        } else if hmm.max_length > 0 {
            hmm.max_length
        } else {
            (hmm.m * 4) as i32
        };
        let f1 = args.f1;
        let f2 = args.f2;
        let f3 = args.f3;
        let do_max = args.max;
        let nobias = args.nobias;
        let nonull2 = args.nonull2;
        let seed = args.seed;
        let threshold_config = NhmmerThresholdConfig::from_pipeline(&pli);

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
                        threshold_config,
                        false,
                        &msv_counter,
                        &bias_counter,
                        &vit_counter,
                        &fwd_counter,
                    )?);
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
                        threshold_config,
                        true,
                        &msv_counter,
                        &bias_counter,
                        &vit_counter,
                        &fwd_counter,
                    )?;
                    // Convert complement-strand coordinates back to forward
                    // strand. nhmmer convention (C p7_pipeline.c:1449-1456):
                    // for minus-strand hits, iali/jali and ienv/jenv are
                    // emitted in reverse order (iali > jali) so the `strand`
                    // column can be derived as iali < jali ? '+' : '-'.
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
        let nhmmer_ln_nw = unsafe { c_log(nw_ratio as f64) };
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
        if std::env::var("NHMMER_NO_DEDUP").is_err() {
            th.sort_by_seqidx_and_alipos();
            th.remove_duplicates();
        }
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
            "  {:>9} {:>6} {:>5}  {:<namew$} {:>posw$} {:>posw$}  {}",
            "E-value",
            " score",
            " bias",
            "Sequence",
            "start",
            "end",
            "Description",
            namew = namew,
            posw = posw,
        )
        .unwrap();
        writeln!(
            out,
            "  {:>9} {:>6} {:>5}  {:<namew$} {:>posw$} {:>posw$}  {}",
            "-------",
            "------",
            "-----",
            "--------",
            "-----",
            "-----",
            "-----------",
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
                "{} {} {:>6.1} {:>5.1}  {:<namew$} {:>posw$} {:>posw$} {}",
                ' ',
                format_pct2g(hit.lnp.exp(), 9),
                hit.score,
                dom_bias_bits,
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
            "Annotation for each hit{}:",
            if args.noali { "" } else { "  (and alignments)" }
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
                    " {} {:>6.1} {:>5.1} {} {:>9} {:>9} {}{} {:>9} {:>9} {}{} {:>9} {:>9} {}{} {:>9}    {:>4.2}",
                    if dom.is_included { '!' } else { '?' },
                    dom.bitscore,
                    dom.dombias,
                    format_pct2g(dom.lnp.exp(), 9),
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
                    acc_val,
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
                            writeln!(out, "  score: {:.1} bits", dom.bitscore).unwrap();
                            // Build CS line from hmm.cs over positions hmmfrom..=hmmto.
                            let cs_line = hmm.cs.as_ref().map(|cs| {
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
        let g3 = |v: f64| -> String { format_g3(v) };
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
        // CPU time / Mc/sec lines. Elapsed wall-clock is recorded; we don't
        // split user/sys so those remain 0.
        let elapsed = search_start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        let total_h = (elapsed_secs / 3600.0) as u64;
        let total_m = ((elapsed_secs / 60.0) as u64) % 60;
        let total_s = elapsed_secs - (total_h * 3600 + total_m * 60) as f64;
        writeln!(
            out,
            "# CPU time: 0.00u 0.00s 00:00:00.00 Elapsed: {:02}:{:02}:{:05.2}",
            total_h, total_m, total_s
        )
        .unwrap();
        let mc_per_sec = if elapsed_secs > 0.0 {
            (residues_searched * hmm.m as f64) / 1_000_000.0 / elapsed_secs
        } else {
            0.0
        };
        writeln!(out, "# Mc/sec: {}", format_g3(mc_per_sec)).unwrap();

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
                &std::env::args().collect::<Vec<_>>().join(" "),
                query_idx == 0,
                query_idx + 1 == hmms.len(),
            );
        }
        if let Some(ref mut f) = dfamtblout_file {
            write_nhmmer_dfamtblout(f, &hmm.name, hmm.acc.as_deref(), "sq-len", &th);
        }
        writeln!(out, "//").unwrap();
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

fn read_hmms(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<hmmer_pure_rs::Hmm>> {
    if path == std::path::Path::new("-") {
        hmmfile::read_hmms(BufReader::new(std::io::stdin().lock()))
    } else if hmmfile_binary::looks_like_binary_hmm_file(path)?
        || path.extension().is_some_and(|ext| ext == "h3m")
    {
        hmmfile_binary::read_binary_hmm_file(path)
    } else {
        hmmfile::read_hmm_file(path)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
            hmmfile_binary::write_binary_hmm(&mut file, &hmm).unwrap();
        }

        let hmms = read_hmms(&path).unwrap();
        assert_eq!(hmms.len(), 1);
        assert_eq!(hmms[0].name, "dna_query");
        assert_eq!(hmms[0].abc_type, hmmer_pure_rs::alphabet::AlphabetType::Dna);
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
    threshold_config: NhmmerThresholdConfig,
    is_complement: bool,
    msv_counter: &std::sync::atomic::AtomicU64,
    bias_counter: &std::sync::atomic::AtomicU64,
    vit_counter: &std::sync::atomic::AtomicU64,
    fwd_counter: &std::sync::atomic::AtomicU64,
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
            threshold_config,
            is_complement,
            msv_counter,
            bias_counter,
            vit_counter,
            fwd_counter,
        ));
    }
    #[allow(unreachable_code)]
    Err(nhmmer_longtarget_unsupported_message())
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
    t: Option<f32>,
    dom_t: Option<f32>,
    inc_t: Option<f32>,
    inc_dom_t: Option<f32>,
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
    threshold_config: NhmmerThresholdConfig,
    is_complement: bool,
    msv_counter: &std::sync::atomic::AtomicU64,
    bias_counter: &std::sync::atomic::AtomicU64,
    vit_counter: &std::sync::atomic::AtomicU64,
    fwd_counter: &std::sync::atomic::AtomicU64,
) -> Vec<hmmer_pure_rs::tophits::Hit> {
    use hmmer_pure_rs::simd::ssv_longtarget;

    // Match C p7_Pipeline_LongTarget (p7_pipeline.c:1763): reconfigure MSV
    // length to max_length BEFORE SSV scan. This sets om->tjb_b based on
    // max_length (e.g. 253 for tRNA), not the full sequence length. Without
    // this, Rust's SSV sc_thresh is too high and misses weak peaks that C
    // catches.
    let mut om_ssv = om.clone();
    om_ssv.reconfig_msv_length(max_length);
    let om = &om_ssv;

    // Phase 1: SSV longtarget filter — find candidate windows. --max turns
    // heuristic filters off, so search one full-length candidate window.
    let mut windows = if do_max {
        vec![ssv_longtarget::HmmWindow {
            n: 1,
            k: 0,
            length: sq.n,
            score: 0.0,
            target_len: sq.n,
            complement: is_complement,
        }]
    } else {
        unsafe { ssv_longtarget::ssv_filter_longtarget(&sq.dsq, sq.n, om, bg, f1, max_length) }
    };

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
    if std::env::var("DEBUG_PRE_MERGE_RUST").is_ok() {
        for (i, w) in windows.iter().enumerate() {
            eprintln!(
                "DEBUG_RUST pre_merge idx={} n={} len={} k={}",
                i, w.n, w.length, w.k
            );
        }
    }
    if !do_max {
        ssv_longtarget::extend_and_merge_windows_with_scoredata(
            &mut windows,
            ml,
            sq.n,
            0.0,
            &prefix_lens,
            &suffix_lens,
        );
    }
    if std::env::var("DEBUG_POST_MERGE_RUST").is_ok() {
        for (i, w) in windows.iter().enumerate() {
            eprintln!(
                "DEBUG_RUST post_merge idx={} n={} len={} k={}",
                i, w.n, w.length, w.k
            );
        }
    }

    // Phase 1a: Per-window standard MSV filter (F1 gate). Mirrors C
    // p7_pipeline.c:1862-1866 which calls p7_oprofile_ReconfigMSVLength(om,
    // window->length), p7_MSVFilter, then `if (P > pli->F1) continue;`. This
    // rejects SSV peaks that don't also pass a full MSV p-value threshold.
    // Without this, Rust finds SSV peaks that C rejects at this stage.
    let f1_thresh = f1;
    let mut windows = if do_max {
        windows
    } else {
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
        msv_filtered
    };

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
    if !do_max {
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
            const B1: usize = 110;
            const B2: usize = 240;
            let f1_l = subseq_len.min(B1);
            let f2_l = subseq_len.min(B2);
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
                });
            }
        }

        // Vit windows are already extended+merged within each SSV subseq above
        // (matching C p7_pipeline.c:1614). Just aggregate them.
        if !vit_windows.is_empty() {
            windows = vit_windows;
        }
    }

    // Window splitting for numerical stability in Forward: mirror C
    // p7_pli_postSSV_LongTarget (p7_pipeline.c:1620-1634). If a merged window
    // is longer than 80000 residues, split it into overlapping sub-windows of
    // length at most 80000 with overlap = min(40000, max_length).
    const MAX_WINDOW_LEN: usize = 80000;
    let overlap_len = 40000_usize.min(ml);
    let mut split_windows: Vec<ssv_longtarget::HmmWindow> = Vec::new();
    for w in &windows {
        if w.length <= MAX_WINDOW_LEN {
            split_windows.push(w.clone());
            continue;
        }
        // Trim current window to MAX_WINDOW_LEN, then emit tail windows.
        let mut head = w.clone();
        head.length = MAX_WINDOW_LEN;
        split_windows.push(head);
        let mut new_n = w.n;
        let mut new_len = w.length;
        loop {
            let shift = MAX_WINDOW_LEN - overlap_len;
            new_n += shift;
            new_len = new_len.saturating_sub(shift);
            if new_len == 0 {
                break;
            }
            let chunk = new_len.min(MAX_WINDOW_LEN);
            split_windows.push(ssv_longtarget::HmmWindow {
                n: new_n,
                k: 0,
                length: chunk,
                score: 0.0,
                target_len: w.target_len,
                complement: w.complement,
            });
            if new_len <= MAX_WINDOW_LEN {
                break;
            }
        }
    }
    windows = split_windows;

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
                fwd_overlap = if next.n < this_end {
                    this_end - next.n
                } else {
                    0
                };
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
                for mut dom in src_hit.dcl {
                    // Adjust coordinates to global sequence position.
                    dom.iali += (win_start - 1) as i64;
                    dom.jali += (win_start - 1) as i64;
                    dom.ienv += (win_start - 1) as i64;
                    dom.jenv += (win_start - 1) as i64;
                    if let Some(ref mut ad) = dom.ad {
                        ad.sqfrom += win_start - 1;
                        ad.sqto += win_start - 1;
                    }
                    let mut new_hit = hmmer_pure_rs::tophits::Hit {
                        name: sq.name.clone(),
                        acc: sq.acc.clone(),
                        desc: sq.desc.clone(),
                        n: sq.n,
                        score: dom.bitscore,
                        bias: dom.dombias.max(0.0),
                        pre_score: dom.bitscore + dom.dombias.max(0.0),
                        sum_score: dom.bitscore,
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
                        nreported: 1,
                        nincluded: 1,
                        flags: src_hit.flags,
                        seqidx: src_hit.seqidx,
                        subseq_start: src_hit.subseq_start,
                        dcl: vec![dom],
                    };
                    let _ = &mut new_hit;
                    all_hits.push(new_hit);
                }
            }
        }
    }

    all_hits
}
