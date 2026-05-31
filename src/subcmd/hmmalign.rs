//! hmmalign — align sequences to a profile HMM.

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::dp::gmx::Gmx;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::msa;
use hmmer_pure_rs::profile::{self, Profile, P7_UNILOCAL};
use hmmer_pure_rs::sequence::{self, Sequence, SequenceFormat};
use hmmer_pure_rs::trace::State;
#[cfg(not(target_arch = "x86_64"))]
use hmmer_pure_rs::{
    dp::generic_backward::g_backward,
    dp::generic_decoding::g_decoding,
    dp::generic_fwdback::g_forward,
    dp::generic_optacc::{g_oa_trace, g_optimal_accuracy},
};
#[cfg(target_arch = "x86_64")]
use hmmer_pure_rs::{
    simd::oprofile::{OProfile, P7O_C, P7O_J, P7O_LOOP, P7O_N},
    simd::probmx::{p_decoding_to_gmx, ProbMx},
};

#[derive(Parser)]
#[command(name = "hmmalign", about = "Align sequences to a profile HMM")]
struct Args {
    /// Output alignment to file instead of stdout
    #[arg(short = 'o')]
    output: Option<PathBuf>,

    /// Trim terminal tails of nonaligned residues from the alignment
    #[arg(long = "trim")]
    trim: bool,

    /// Include alignment in file <f> that the HMM was originally built from
    #[arg(long = "mapali")]
    mapali: Option<PathBuf>,

    /// HMM file
    hmmfile: PathBuf,
    /// Sequence file (FASTA format)
    seqfile: PathBuf,

    /// Output alignment format
    #[arg(long = "outformat", default_value = "Stockholm")]
    outformat: String,

    /// Assert input sequence file format
    #[arg(long = "informat")]
    informat: Option<String>,

    /// Assert protein alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["dna", "rna"])]
    amino: bool,

    /// Assert DNA alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["amino", "rna"])]
    dna: bool,

    /// Assert RNA alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["amino", "dna"])]
    rna: bool,
}

struct AlignmentRow {
    name: String,
    acc: Option<String>,
    desc: Option<String>,
    aseq: String,
    ppline: Option<String>,
}

struct TextMsa {
    rows: Vec<AlignmentRow>,
    rfline: String,
    pp_cons: String,
}

/// Entry point for `hmmalign`: align sequences to a profile HMM.
///
/// Reads an HMM and a sequence file, runs Forward/Backward + posterior decoding +
/// optimal-accuracy traceback per sequence, then assembles a multiple alignment
/// (Stockholm or A2M). Optionally merges a `--mapali` source alignment via the
/// HMM's map and checksum. Corresponds to `main()` in hmmer/src/hmmalign.c.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);
    if args.hmmfile == std::path::Path::new("-") && args.seqfile == std::path::Path::new("-") {
        eprintln!(
            "ERROR: Either <hmmfile> or <seqfile> may be '-' (to read from stdin), but not both."
        );
        std::process::exit(1);
    }
    // --amino/--dna/--rna mutual exclusion (C: ALPHOPTS toggle group) is
    // enforced at parse time by clap.
    let informat = args.informat.as_ref().map(|informat| {
        SequenceFormat::from_name(informat).unwrap_or_else(|| {
            eprintln!("{informat} is not a recognized input sequence file format");
            std::process::exit(1);
        })
    });

    enum OutFormat {
        Stockholm,
        Pfam,
        A2m,
        Psiblast,
        Afa,
        Clustal,
        ClustalLike,
        Selex,
        Phylip,
        Phylips,
    }
    // Mirrors the format names accepted by Easel's esl_msafile_EncodeFormat()
    // that hmmalign passes through esl_msafile_Write(). "pfam" is the Stockholm
    // single-block variant: Easel writes Stockholm in 200-column blocks but Pfam
    // in a single block spanning the whole alignment.
    let outformat = if args.outformat.eq_ignore_ascii_case("stockholm") {
        OutFormat::Stockholm
    } else if args.outformat.eq_ignore_ascii_case("pfam") {
        OutFormat::Pfam
    } else if args.outformat.eq_ignore_ascii_case("a2m") {
        OutFormat::A2m
    } else if args.outformat.eq_ignore_ascii_case("psiblast") {
        OutFormat::Psiblast
    } else if args.outformat.eq_ignore_ascii_case("afa") {
        OutFormat::Afa
    } else if args.outformat.eq_ignore_ascii_case("clustal") {
        OutFormat::Clustal
    } else if args.outformat.eq_ignore_ascii_case("clustallike") {
        OutFormat::ClustalLike
    } else if args.outformat.eq_ignore_ascii_case("selex") {
        OutFormat::Selex
    } else if args.outformat.eq_ignore_ascii_case("phylip") {
        OutFormat::Phylip
    } else if args.outformat.eq_ignore_ascii_case("phylips") {
        OutFormat::Phylips
    } else {
        eprintln!(
            "{} is not a recognized output MSA file format",
            args.outformat
        );
        return std::process::ExitCode::from(2);
    };

    logsum::p7_flogsuminit();

    let hmms = read_hmms_maybe_stdin(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });
    if hmms.is_empty() {
        eprintln!("Error: no HMMs found in {}", args.hmmfile.display());
        std::process::exit(1);
    }
    if hmms.len() != 1 {
        eprintln!(
            "Error: HMM file {} does not contain just one HMM",
            args.hmmfile.display()
        );
        std::process::exit(1);
    }

    let hmm = &hmms[0];
    let abc = if args.amino {
        Alphabet::amino()
    } else if args.dna {
        Alphabet::dna()
    } else if args.rna {
        Alphabet::rna()
    } else {
        Alphabet::new(hmm.abc_type)
    };
    let bg = Bg::new(&abc);

    let mut sequences = Vec::new();
    let mut sqf = open_seq_file_maybe_stdin(&args.seqfile, &abc, informat).unwrap_or_else(|e| {
        eprintln!("Error opening sequence file: {}", e);
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

    let msa = if let Some(mapali_path) = &args.mapali {
        let mapped_msas = msa::read_stockholm(mapali_path).unwrap_or_else(|e| {
            eprintln!(
                "Error reading mapped alignment {}: {}",
                mapali_path.display(),
                e
            );
            std::process::exit(1);
        });
        let mapped = mapped_msas.first().unwrap_or_else(|| {
            eprintln!(
                "Mapped alignment {} contained no alignment blocks",
                mapali_path.display()
            );
            std::process::exit(1);
        });
        build_text_msa_with_mapali(hmm, &abc, &bg, &sequences, mapped, args.trim)
    } else {
        build_text_msa(hmm, &abc, &bg, &sequences, args.trim)
    };

    let mut output_file = args.output.as_ref().map(|path| {
        File::create(path).unwrap_or_else(|e| {
            eprintln!("Error opening output file {}: {}", path.display(), e);
            std::process::exit(1);
        })
    });
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let out: &mut dyn Write = match output_file.as_mut() {
        Some(file) => file,
        None => &mut stdout_lock,
    };
    match outformat {
        OutFormat::Stockholm => write_stockholm(out, &msa),
        OutFormat::Pfam => write_stockholm_blocked(out, &msa, msa.rfline.len().max(1)),
        OutFormat::A2m => write_a2m(
            out,
            &msa,
            hmm.abc_type == hmmer_pure_rs::alphabet::AlphabetType::Amino,
        ),
        OutFormat::Psiblast => {
            msa::write_psiblast(out, &write_rows(&msa), Some(&msa.rfline)).unwrap();
        }
        OutFormat::Afa => {
            msa::write_afa(out, &write_rows(&msa)).unwrap();
        }
        OutFormat::Clustal => {
            msa::write_clustal(out, &write_rows(&msa), false, EASEL_VERSION).unwrap();
        }
        OutFormat::ClustalLike => {
            msa::write_clustal(out, &write_rows(&msa), true, EASEL_VERSION).unwrap();
        }
        OutFormat::Selex => {
            msa::write_selex(out, &write_rows(&msa), Some(&msa.rfline)).unwrap();
        }
        OutFormat::Phylip => {
            msa::write_phylip(out, &write_rows(&msa), false).unwrap();
        }
        OutFormat::Phylips => {
            msa::write_phylip(out, &write_rows(&msa), true).unwrap();
        }
    }
    std::process::ExitCode::SUCCESS
}

fn read_hmms_maybe_stdin(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<hmmer_pure_rs::Hmm>> {
    if path == std::path::Path::new("-") {
        let stdin = std::io::stdin();
        hmmfile::read_hmms_auto(BufReader::new(stdin.lock()))
    } else {
        hmmfile::read_hmm_file_auto(path)
    }
}

fn open_seq_file_maybe_stdin(
    path: &std::path::Path,
    abc: &Alphabet,
    format: Option<SequenceFormat>,
) -> hmmer_pure_rs::errors::HmmerResult<sequence::SeqFile<Box<dyn Read>>> {
    if path == std::path::Path::new("-") {
        let sqf = sequence::SeqFile::new(Box::new(std::io::stdin()) as Box<dyn Read>, abc.clone());
        Ok(if let Some(format) = format {
            sqf.with_format(format)
        } else {
            sqf
        })
    } else if let Some(format) = format {
        sequence::open_seq_file_with_format(path, abc, format)
    } else {
        sequence::open_seq_file(path, abc)
    }
}

/// Emit a Stockholm 1.0 representation of the assembled alignment, including
/// per-row PP annotation and `#=GC PP_cons` / `#=GC RF` consensus lines.
///
/// Faithful port of Easel's `stockholm_write` (`esl_msafile_stockholm.c`) for
/// the subset of annotation that `hmmalign` produces: per-row `#=GR <name> PP`,
/// `#=GC PP_cons`, `#=GC RF`, and optional `#=GS <name> AC/DE`. Stockholm format
/// wraps the alignment in 200-column blocks; the left margin width matches C's
/// `margin` computation so sequence, GR, and GC lines stay in register.
///
/// ## Intentional annotation scope — why no SS/SA/MM/#=GF lines
///
/// `hmmalign` builds its MSA via `p7_tracealign_Seqs()` in tracealign.c, which
/// calls three annotation helpers and then sets per-sequence names/accessions/descs:
///
///   1. `annotate_rf()` → sets `msa->rf` ('x'/'.' column mask) — **emitted** as `#=GC RF`
///   2. `annotate_mm()` → sets `msa->mm` only when `hmm->mm != NULL` (model mask).
///      Standard HMMs from `hmmbuild` never carry a model mask, so `msa->mm` is NULL
///      in practice for all `hmmalign` runs; the `#=GC MM` line is therefore never
///      produced. **Not emitted** (correctly).
///   3. `annotate_posterior_probability()` → sets `msa->pp[i]` per sequence and
///      `msa->pp_cons` — **emitted** as `#=GR <name> PP` and `#=GC PP_cons`.
///
/// Fields that `p7_tracealign_Seqs()` never sets on the output MSA for `hmmalign`:
///   - `msa->name/acc/desc/au` → NULL  → no `#=GF ID/AC/DE/AU` lines (correct)
///   - `msa->ss_cons/sa_cons`  → NULL  → no `#=GC SS_cons/SA_cons` lines (correct)
///   - `msa->ss[i]/sa[i]`      → NULL  → no `#=GR <name> SS/SA` lines (correct)
///   - `msa->cutset[]`         → unset → no `#=GF GA/TC/NC` lines (correct)
///
/// The per-sequence `sqacc` and `sqdesc` arrays are set from `sq[i]->acc` and
/// `sq[i]->desc`, which come from the input FASTA/sequence headers — **emitted** as
/// `#=GS <name> AC` / `#=GS <name> DE` when non-empty.
///
/// Conclusion: the Rust writers emit exactly what C's `hmmalign` + `stockholm_write`
/// would emit given the annotation that `p7_tracealign_Seqs()` places on the MSA.
/// The "missing" SS/SA/MM/GF-cutoff lines are absent because `hmmalign` itself never
/// produces them — this is a non-gap, not an omission in the Rust port.
fn write_stockholm(out: &mut dyn Write, msa: &TextMsa) {
    write_stockholm_blocked(out, msa, 200);
}

/// Shared Stockholm/Pfam writer; `cpl` is the residues-per-line block width
/// (200 for Stockholm, alen for Pfam — Easel's `stockholm_write` <cpl> arg).
fn write_stockholm_blocked(out: &mut dyn Write, msa: &TextMsa, cpl: usize) {
    let maxname = msa.rows.iter().map(|row| row.name.len()).max().unwrap_or(0);

    // maxgc: GC tags emitted here are PP_cons (7) and RF (2) => 7.
    let mut maxgc = 0usize;
    maxgc = maxgc.max(7); // PP_cons
    maxgc = maxgc.max(2); // RF
                          // maxgr: GR tag emitted here is PP (2).
    let maxgr = 2usize;

    let mut margin = maxname + 1;
    if maxgc > 0 && maxgc + 6 > margin {
        margin = maxgc + 6;
    }
    if maxgr > 0 && maxname + maxgr + 7 > margin {
        margin = maxname + maxgr + 7;
    }

    writeln!(out, "# STOCKHOLM 1.0").unwrap();

    // GF section: hmmalign's MSA carries no #=GF annotation, but Easel always
    // closes the GF section with a blank line.
    writeln!(out).unwrap();

    // GS section: per-sequence AC and DE blocks, each terminated by a blank line
    // when present (matching Easel's separate msa->sqacc / msa->sqdesc loops).
    if msa.rows.iter().any(|row| row.acc.is_some()) {
        for row in &msa.rows {
            if let Some(acc) = &row.acc {
                if !acc.is_empty() {
                    writeln!(out, "#=GS {:<width$} AC {}", row.name, acc, width = maxname).unwrap();
                }
            }
        }
        writeln!(out).unwrap();
    }
    if msa.rows.iter().any(|row| row.desc.is_some()) {
        for row in &msa.rows {
            if let Some(desc) = &row.desc {
                if !desc.is_empty() {
                    writeln!(
                        out,
                        "#=GS {:<width$} DE {}",
                        row.name,
                        desc,
                        width = maxname
                    )
                    .unwrap();
                }
            }
        }
        writeln!(out).unwrap();
    }

    // Alignment section, in <cpl>-column blocks.
    let alen = msa.rfline.len();
    let mut currpos = 0usize;
    while currpos < alen {
        let end = (currpos + cpl).min(alen);
        if currpos > 0 {
            writeln!(out).unwrap();
        }
        for row in &msa.rows {
            writeln!(
                out,
                "{:<width$} {}",
                row.name,
                &row.aseq[currpos..end],
                width = margin - 1
            )
            .unwrap();
            if let Some(ppline) = &row.ppline {
                writeln!(
                    out,
                    "#=GR {:<namew$} {:<tagw$} {}",
                    row.name,
                    "PP",
                    &ppline[currpos..end],
                    namew = maxname,
                    tagw = margin - maxname - 7
                )
                .unwrap();
            }
        }
        writeln!(
            out,
            "#=GC {:<tagw$} {}",
            "PP_cons",
            &msa.pp_cons[currpos..end],
            tagw = margin - 6
        )
        .unwrap();
        writeln!(
            out,
            "#=GC {:<tagw$} {}",
            "RF",
            &msa.rfline[currpos..end],
            tagw = margin - 6
        )
        .unwrap();
        currpos = end;
    }
    writeln!(out, "//").unwrap();
}

/// Emit the alignment in A2M format: consensus columns uppercase / `-`,
/// insert columns lowercase (insert gaps suppressed); `O`/`o` mapped to `X`/`x`
/// for amino alphabets.
fn write_a2m(out: &mut dyn Write, msa: &TextMsa, is_amino: bool) {
    for row in &msa.rows {
        if let Some(desc) = row.desc.as_deref().filter(|desc| !desc.is_empty()) {
            writeln!(out, ">{} {}", row.name, desc).unwrap();
        } else {
            writeln!(out, ">{}", row.name).unwrap();
        }
        let mut seq = String::with_capacity(row.aseq.len());
        for (ch, rf) in row.aseq.chars().zip(msa.rfline.chars()) {
            let is_consensus = rf.is_ascii_alphanumeric();
            let out_ch = if is_consensus {
                if ch.is_ascii_alphabetic() {
                    let mut up = ch.to_ascii_uppercase();
                    if is_amino && up == 'O' {
                        up = 'X';
                    }
                    Some(up)
                } else {
                    Some('-')
                }
            } else if ch.is_ascii_alphabetic() {
                let mut low = ch.to_ascii_lowercase();
                if is_amino && low == 'o' {
                    low = 'x';
                }
                Some(low)
            } else {
                None
            };
            if let Some(ch) = out_ch {
                seq.push(ch);
            }
        }
        for chunk in seq.as_bytes().chunks(60) {
            writeln!(out, "{}", std::str::from_utf8(chunk).unwrap()).unwrap();
        }
    }
}

/// Easel version string, used in the CLUSTAL-like header line. Matches the
/// `EASEL_VERSION` substituted by `hmmer/configure.ac` (currently "0.49").
const EASEL_VERSION: &str = "0.49";

/// Borrow the assembled `TextMsa` rows as the generic `msa::WriteRow` view used
/// by the reusable text-MSA writers in `src/msa.rs`.
fn write_rows(msa: &TextMsa) -> Vec<msa::WriteRow<'_>> {
    msa.rows
        .iter()
        .map(|row| msa::WriteRow {
            name: &row.name,
            acc: row.acc.as_deref(),
            desc: row.desc.as_deref(),
            aseq: &row.aseq,
        })
        .collect()
}

/// Align every input sequence to `hmm` (Forward/Backward + posterior decoding +
/// optimal accuracy traceback) and stitch the per-sequence traces into a single
/// text MSA, computing per-column PP_cons. Counterpart to the alignment block of
/// `main()` in hmmalign.c when no `--mapali` is supplied.
/// A per-sequence alignment result: the optimal-accuracy traceback and a small
/// per-trace-position posterior vector (`pp_z[z]` = the posterior for the residue
/// emitted at trace position `z`; 0 for non-emitting states).
type AlignedTrace = (hmmer_pure_rs::trace::Trace, Vec<f32>);

/// Forward/Backward + posterior decoding + optimal-accuracy traceback for every
/// sequence. Mirrors the per-sequence loop in C `p7_tracealign_Seqs`: the DP
/// matrices (and the profile) are allocated ONCE and reused (grown) across
/// sequences via `p7_gmx_GrowTo`-style reuse, so only the trace and a small
/// O(L) posterior vector are retained per sequence — never the O(M*L) matrices.
fn align_traces(
    hmm: &hmmer_pure_rs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
) -> Vec<AlignedTrace> {
    #[cfg(target_arch = "x86_64")]
    {
        return align_traces_optimized(hmm, abc, bg, sequences);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        return align_traces_generic(hmm, abc, bg, sequences);
    }
}

#[cfg(target_arch = "x86_64")]
fn align_traces_optimized(
    hmm: &hmmer_pure_rs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
) -> Vec<AlignedTrace> {
    let mut gm = Profile::new(hmm.m, abc);
    profile::profile_config(hmm, bg, &mut gm, 100, P7_UNILOCAL);
    let mut om = OProfile::convert(&gm);
    let mut fwd = ProbMx::new_full(hmm.m, 0);
    let mut bck = ProbMx::new_full(hmm.m, 0);
    let mut pp = ProbMx::new_full(hmm.m, 0);
    let mut oa = ProbMx::new_full(hmm.m, 0);
    let mut pp_gmx = Gmx::new(hmm.m, 0);

    let mut out = Vec::with_capacity(sequences.len());
    for sq in sequences {
        om.reconfig_length(sq.n as i32);
        fwd.resize_full(hmm.m, sq.n);
        bck.resize_full(hmm.m, sq.n);
        pp.resize_full(hmm.m, sq.n);
        oa.resize_full(hmm.m, sq.n);
        pp_gmx.grow_to(hmm.m, sq.n);

        let fwd_sc = unsafe {
            hmmer_pure_rs::simd::fwd_filter::forward_parser_pmx(&sq.dsq, sq.n, &om, &mut fwd)
        };
        unsafe {
            hmmer_pure_rs::simd::bck_filter::backward_parser_pmx(
                &sq.dsq, sq.n, &om, fwd_sc, &mut bck,
            );
            hmmer_pure_rs::simd::optacc::posterior_decoding_pmx(&fwd, &bck, &om, &mut pp);
            hmmer_pure_rs::simd::optacc::optimal_accuracy_pmx(&om, &pp, &mut oa);
        }
        let tr = unsafe { hmmer_pure_rs::simd::optacc::oa_trace_pmx(&om, &pp, &oa) };
        let njc_loop = [
            om.xf[P7O_N][P7O_LOOP],
            om.xf[P7O_J][P7O_LOOP],
            om.xf[P7O_C][P7O_LOOP],
        ];
        p_decoding_to_gmx(&fwd, &bck, hmm.m, njc_loop, &mut pp_gmx);
        let pp_z = trace_posteriors(&tr, &pp_gmx);
        out.push((tr, pp_z));
    }
    out
}

#[cfg(not(target_arch = "x86_64"))]
fn align_traces_generic(
    hmm: &hmmer_pure_rs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
) -> Vec<AlignedTrace> {
    // Configure the (L-independent) profile scores ONCE; only the length model
    // is updated per sequence via reconfig_length. Mirrors C p7_tracealign_Seqs,
    // which calls p7_ProfileConfig once and p7_ReconfigLength per target.
    let mut gm = Profile::new(hmm.m, abc);
    profile::profile_config(hmm, bg, &mut gm, 100, P7_UNILOCAL);
    let mut fwd = Gmx::new(hmm.m, 0);
    let mut bck = Gmx::new(hmm.m, 0);
    let mut pp = Gmx::new(hmm.m, 0);
    let mut oa = Gmx::new(hmm.m, 0);

    let mut out = Vec::with_capacity(sequences.len());
    for sq in sequences {
        profile::reconfig_length(&mut gm, sq.n as i32);
        fwd.grow_to(hmm.m, sq.n);
        bck.grow_to(hmm.m, sq.n);
        pp.grow_to(hmm.m, sq.n);
        oa.grow_to(hmm.m, sq.n);

        g_forward(&sq.dsq, sq.n, &gm, &mut fwd);
        g_backward(&sq.dsq, sq.n, &gm, &mut bck);
        g_decoding(&gm, &fwd, &bck, &mut pp);
        g_optimal_accuracy(&gm, &pp, &mut oa);
        let tr = g_oa_trace(&gm, &pp, &oa);
        let pp_z = trace_posteriors(&tr, &pp);
        out.push((tr, pp_z));
    }
    out
}

/// Extract the per-residue posterior probability at each trace position, while
/// the posterior matrix is still alive, so the O(M*L) matrix can be dropped.
fn trace_posteriors(tr: &hmmer_pure_rs::trace::Trace, pp: &Gmx) -> Vec<f32> {
    let mut v = vec![0.0_f32; tr.n];
    for (z, value) in v.iter_mut().enumerate().take(tr.n) {
        *value = match tr.st[z] {
            State::M => pp.mmx(tr.i[z], tr.k[z]),
            State::I => pp.imx(tr.i[z], tr.k[z]),
            State::N | State::C => pp.xmx(tr.i[z], state_pp_index(tr.st[z])),
            _ => 0.0,
        };
    }
    v
}

fn build_text_msa(
    hmm: &hmmer_pure_rs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
    trim: bool,
) -> TextMsa {
    let traces = align_traces(hmm, abc, bg, sequences);

    let (inscount, matuse, matmap, alen) = map_new_msa(hmm.m, &traces, trim);
    let mut rows = Vec::with_capacity(sequences.len());
    let mut pp_cons_sum = vec![0.0_f32; alen];
    let mut pp_cons_n = vec![0usize; alen];

    for (sq, (tr, pp_z)) in sequences.iter().zip(traces.iter()) {
        let (aseq, ppline) = make_text_row(abc, sq, tr, pp_z, &matuse, &matmap, alen, trim);
        for z in 0..tr.n {
            if tr.st[z] == State::M {
                let apos = matmap[tr.k[z]] - 1;
                pp_cons_sum[apos] += pp_z[z].min(1.0);
                pp_cons_n[apos] += 1;
            }
        }
        rows.push(AlignmentRow {
            name: sq.name.clone(),
            acc: (!sq.acc.is_empty()).then(|| sq.acc.clone()),
            desc: (!sq.desc.is_empty()).then(|| sq.desc.clone()),
            aseq,
            ppline: Some(ppline),
        });
    }

    for row in &mut rows {
        if let Some(ppline) = &mut row.ppline {
            rejustify_insertions_text(&mut row.aseq, ppline, &inscount, &matmap, &matuse, hmm.m);
        }
    }

    let mut rfline = vec![b'.'; alen];
    for k in 1..=hmm.m {
        if matuse[k] {
            rfline[matmap[k] - 1] = b'x';
        }
    }

    let mut pp_cons = String::with_capacity(alen);
    for apos in 0..alen {
        if pp_cons_n[apos] > 0 {
            pp_cons.push(pp_to_char(pp_cons_sum[apos] / pp_cons_n[apos] as f32));
        } else {
            pp_cons.push('.');
        }
    }

    TextMsa {
        rows,
        rfline: String::from_utf8(rfline).unwrap(),
        pp_cons,
    }
}

/// Variant of `build_text_msa` that also folds in a previously built `--mapali`
/// MSA. Verifies the HMM checksum/map, computes per-column insert widths spanning
/// both the source MSA and the new traces, then emits a merged alignment. Mirrors
/// the `map_alignment()` + main-loop path in hmmalign.c.
fn build_text_msa_with_mapali(
    hmm: &hmmer_pure_rs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
    mapped: &msa::Msa,
    trim: bool,
) -> TextMsa {
    if hmm.flags & hmmer_pure_rs::hmm::P7H_CHKSUM == 0 {
        eprintln!("HMM has no checksum. --mapali unreliable without it.");
        std::process::exit(1);
    }
    if hmm.flags & hmmer_pure_rs::hmm::P7H_MAP == 0 || hmm.map.is_none() {
        eprintln!("HMM has no map. --mapali can't work without it.");
        std::process::exit(1);
    }
    let checksum = msa::checksum(mapped, abc);
    if checksum != hmm.checksum {
        eprintln!("--mapali MSA isn't same as the one HMM came from (checksum mismatch)");
        std::process::exit(1);
    }

    let map = hmm.map.as_ref().unwrap();
    let mapped_pairs = faux_from_mapped_msa(mapped, map, abc, hmm.m);
    let traces = align_traces(hmm, abc, bg, sequences);

    let mut all_traces = Vec::with_capacity(mapped_pairs.len() + traces.len());
    all_traces.extend(mapped_pairs.iter().map(|(_, tr)| (tr.clone(), Vec::new())));
    all_traces.extend(traces.iter().cloned());

    let (inscount, matuse, matmap, alen) = map_new_msa(hmm.m, &all_traces, trim);

    let mut rows = Vec::with_capacity(mapped.nseq + sequences.len());
    for (idx, (sq, tr)) in mapped_pairs.iter().enumerate() {
        let dummy_pp = vec![0.0_f32; tr.n];
        let (mut aseq, mut ppline) =
            make_text_row(abc, sq, tr, &dummy_pp, &matuse, &matmap, alen, trim);
        rejustify_insertions_text(&mut aseq, &mut ppline, &inscount, &matmap, &matuse, hmm.m);
        rows.push(AlignmentRow {
            name: mapped.sqname[idx].clone(),
            acc: (!mapped.sqacc[idx].is_empty()).then(|| mapped.sqacc[idx].clone()),
            desc: (!mapped.sqdesc[idx].is_empty()).then(|| mapped.sqdesc[idx].clone()),
            aseq,
            ppline: None,
        });
    }

    let mut pp_cons_sum = vec![0.0_f32; alen];
    let mut pp_cons_n = vec![0usize; alen];
    for (sq, (tr, pp_z)) in sequences.iter().zip(traces.iter()) {
        let (mut aseq, mut ppline) = make_text_row(abc, sq, tr, pp_z, &matuse, &matmap, alen, trim);
        rejustify_insertions_text(&mut aseq, &mut ppline, &inscount, &matmap, &matuse, hmm.m);
        for z in 0..tr.n {
            if tr.st[z] == State::M {
                let apos = matmap[tr.k[z]] - 1;
                pp_cons_sum[apos] += pp_z[z].min(1.0);
                pp_cons_n[apos] += 1;
            }
        }
        rows.push(AlignmentRow {
            name: sq.name.clone(),
            acc: (!sq.acc.is_empty()).then(|| sq.acc.clone()),
            desc: (!sq.desc.is_empty()).then(|| sq.desc.clone()),
            aseq,
            ppline: Some(ppline),
        });
    }

    let mut rfline = vec![b'.'; alen];
    for k in 1..=hmm.m {
        if matuse[k] {
            rfline[matmap[k] - 1] = b'x';
        }
    }
    let mut pp_cons = String::with_capacity(alen);
    for apos in 0..alen {
        if pp_cons_n[apos] > 0 {
            pp_cons.push(pp_to_char(pp_cons_sum[apos] / pp_cons_n[apos] as f32));
        } else {
            pp_cons.push('.');
        }
    }

    let mut msa = TextMsa {
        rows,
        rfline: String::from_utf8(rfline).unwrap(),
        pp_cons,
    };
    if trim {
        trim_terminal_residueless_columns(&mut msa);
    }
    msa
}

fn faux_from_mapped_msa(
    mapped: &msa::Msa,
    map: &[i32],
    abc: &Alphabet,
    m: usize,
) -> Vec<(Sequence, hmmer_pure_rs::trace::Trace)> {
    let mut matassign = vec![false; mapped.alen + 1];
    for k in 1..=m {
        matassign[map[k] as usize] = true;
    }

    (0..mapped.nseq)
        .map(|idx| {
            let mut sq = Sequence::new();
            sq.name = mapped.sqname[idx].clone();
            sq.acc = mapped.sqacc[idx].clone();
            sq.desc = mapped.sqdesc[idx].clone();

            let mut tr = hmmer_pure_rs::trace::Trace::new();
            tr.append(State::B, 0, 0);

            let mut k = 0usize;
            let mut rpos = 1usize;
            for (apos0, &ch) in mapped.aseq[idx].iter().enumerate() {
                let apos = apos0 + 1;
                let code = abc.digitize_symbol(ch);
                let is_residue = abc.is_residue(code) || code == abc.nonresidue_code();
                let is_gap = matches!(ch, b'-' | b'.' | b'_');
                let is_missing = code == abc.missing_code();

                if matassign[apos] {
                    k += 1;
                    if is_residue {
                        sq.dsq.push(code);
                        tr.append(State::M, k, rpos);
                        rpos += 1;
                    } else if is_gap {
                        tr.append(State::D, k, 0);
                    } else if is_missing {
                        if tr.st.last().copied() != Some(State::X) {
                            tr.append(State::X, k, 0);
                        }
                    }
                } else if is_residue {
                    sq.dsq.push(code);
                    tr.append(State::I, k, rpos);
                    rpos += 1;
                } else if is_missing && tr.st.last().copied() != Some(State::X) {
                    tr.append(State::X, k, 0);
                }
            }

            tr.append(State::E, 0, 0);
            tr.m = k;
            tr.l = mapped.alen;
            sq.n = sq.dsq.len() - 1;
            sq.l = sq.n;
            sq.dsq.push(hmmer_pure_rs::alphabet::DSQ_SENTINEL);

            (sq, tr)
        })
        .collect()
}

fn trim_terminal_residueless_columns(msa: &mut TextMsa) {
    let alen = msa.rfline.len();
    let first = (0..alen).find(|&apos| {
        msa.rows
            .iter()
            .any(|row| row.aseq.as_bytes()[apos].is_ascii_alphabetic())
    });
    let Some(first) = first else {
        return;
    };
    let last = (0..alen)
        .rfind(|&apos| {
            msa.rows
                .iter()
                .any(|row| row.aseq.as_bytes()[apos].is_ascii_alphabetic())
        })
        .unwrap();
    if first == 0 && last + 1 == alen {
        return;
    }

    for row in &mut msa.rows {
        row.aseq = row.aseq[first..=last].to_string();
        if let Some(ppline) = row.ppline.as_mut() {
            *ppline = ppline[first..=last].to_string();
        }
    }
    msa.rfline = msa.rfline[first..=last].to_string();
    msa.pp_cons = msa.pp_cons[first..=last].to_string();
}

/// Build the merged-MSA coordinate system (insert widths, matuse flags, matmap,
/// total `alen`) from a set of traces. Mirrors Easel's `esl_msa_MapNew` /
/// HMMER's `p7_tracealign_Seqs()` map-construction step.
fn map_new_msa(
    m: usize,
    traces: &[AlignedTrace],
    trim: bool,
) -> (Vec<usize>, Vec<bool>, Vec<usize>, usize) {
    let mut inscount = vec![0usize; m + 1];
    let mut matuse = vec![true; m + 1];
    matuse[0] = false;
    let mut insnum = vec![0usize; m + 1];

    for (tr, _) in traces {
        insnum.fill(0);
        for z in 1..tr.n {
            match tr.st[z] {
                State::I => insnum[tr.k[z]] += 1,
                State::N if tr.st[z - 1] == State::N => insnum[0] += 1,
                State::C if tr.st[z - 1] == State::C => insnum[m] += 1,
                State::M => matuse[tr.k[z]] = true,
                State::J => panic!("J state unsupported in hmmalign MSA construction"),
                _ => {}
            }
        }
        for k in 0..=m {
            inscount[k] = inscount[k].max(insnum[k]);
        }
    }

    if trim {
        inscount[0] = 0;
        inscount[m] = 0;
    }

    let mut matmap = vec![0usize; m + 1];
    let mut alen = inscount[0];
    for k in 1..=m {
        if matuse[k] {
            matmap[k] = alen + 1;
            alen += 1 + inscount[k];
        } else {
            matmap[k] = alen;
            alen += inscount[k];
        }
    }

    (inscount, matuse, matmap, alen)
}

/// Render one sequence's trace as text-aligned residues plus the matching PP
/// (posterior-probability) annotation row, using `matmap` to place match-state
/// residues and contiguous insert positions for I/N/C states.
#[allow(clippy::too_many_arguments)]
fn make_text_row(
    abc: &Alphabet,
    sq: &Sequence,
    tr: &hmmer_pure_rs::trace::Trace,
    pp_z: &[f32],
    matuse: &[bool],
    matmap: &[usize],
    alen: usize,
    trim: bool,
) -> (String, String) {
    let mut aseq = vec![b'.'; alen];
    let mut ppline = vec![b'.'; alen];
    for k in 1..matuse.len() {
        if matuse[k] {
            aseq[matmap[k] - 1] = b'-';
        }
    }

    let mut apos = 0usize;
    for z in 0..tr.n {
        match tr.st[z] {
            State::M => {
                let idx = matmap[tr.k[z]] - 1;
                aseq[idx] = (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_uppercase() as u8;
                ppline[idx] = pp_to_char(pp_z[z].min(1.0)) as u8;
                apos = matmap[tr.k[z]];
            }
            State::D => {
                if matuse[tr.k[z]] {
                    aseq[matmap[tr.k[z]] - 1] = b'-';
                }
                apos = matmap[tr.k[z]];
            }
            State::I => {
                if apos < alen {
                    aseq[apos] =
                        (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_lowercase() as u8;
                    ppline[apos] = pp_to_char(pp_z[z].min(1.0)) as u8;
                    apos += 1;
                }
            }
            State::N | State::C => {
                if !trim && tr.i[z] > 0 && apos < alen {
                    aseq[apos] =
                        (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_lowercase() as u8;
                    ppline[apos] = pp_to_char(pp_z[z].min(1.0)) as u8;
                    apos += 1;
                }
            }
            State::E => {
                apos = matmap[matmap.len() - 1];
            }
            _ => {}
        }
    }

    (
        String::from_utf8(aseq).unwrap(),
        String::from_utf8(ppline).unwrap(),
    )
}

/// Map an N/C special state to the matching column index in the special-states
/// (XMX) portion of the posterior decoding matrix.
fn state_pp_index(st: State) -> usize {
    match st {
        State::N => hmmer_pure_rs::dp::gmx::P7G_N,
        State::C => hmmer_pure_rs::dp::gmx::P7G_C,
        _ => unreachable!(),
    }
}

/// Right-justify the second half of each insert region in `aseq`/`ppline` so
/// inserts pack toward the flanking match columns. Replicates HMMER's
/// `rejustify_inserts_text` behavior.
fn rejustify_insertions_text(
    aseq: &mut String,
    ppline: &mut String,
    inserts: &[usize],
    matmap: &[usize],
    matuse: &[bool],
    m: usize,
) {
    fn is_text_gap(c: u8) -> bool {
        matches!(c, b'.' | b'-' | b'~')
    }

    let mut aseq_bytes = aseq.as_bytes().to_vec();
    let mut pp_bytes = ppline.as_bytes().to_vec();

    for k in 0..m {
        if inserts[k] <= 1 {
            continue;
        }

        let start = matmap[k];
        let end = matmap[k + 1] - usize::from(matuse[k + 1]);
        let mut nins = (start..end)
            .filter(|&apos| aseq_bytes[apos].is_ascii_alphabetic())
            .count();
        if k == 0 {
            nins = 0;
        } else {
            nins /= 2;
        }

        let floor = (start + nins) as isize;
        let mut opos = end as isize - 1;
        let mut npos = end as isize - 1;
        while opos >= floor {
            if is_text_gap(aseq_bytes[opos as usize]) {
                opos -= 1;
                continue;
            }
            aseq_bytes[npos as usize] = aseq_bytes[opos as usize];
            pp_bytes[npos as usize] = pp_bytes[opos as usize];
            opos -= 1;
            npos -= 1;
        }
        while npos >= floor {
            aseq_bytes[npos as usize] = b'.';
            pp_bytes[npos as usize] = b'.';
            npos -= 1;
        }
    }

    *aseq = String::from_utf8(aseq_bytes).unwrap();
    *ppline = String::from_utf8(pp_bytes).unwrap();
}

/// Convert a posterior probability in [0,1] to its single-character bin
/// (`0`-`9`, `*`, or `.`) used in Stockholm PP annotation lines.
fn pp_to_char(pp: f32) -> char {
    let p = pp.clamp(0.0, 1.0);
    if p >= 0.95 {
        '*'
    } else if p >= 0.85 {
        '9'
    } else if p >= 0.75 {
        '8'
    } else if p >= 0.65 {
        '7'
    } else if p >= 0.55 {
        '6'
    } else if p >= 0.45 {
        '5'
    } else if p >= 0.35 {
        '4'
    } else if p >= 0.25 {
        '3'
    } else if p >= 0.15 {
        '2'
    } else if p >= 0.05 {
        '1'
    } else if p > 0.0 {
        '0'
    } else {
        '.'
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmmalign_rejects_conflicting_alphabet_flags() {
        // C: ALPHOPTS toggle group -> "Options --amino and --dna conflict".
        assert!(Args::try_parse_from(["hmmalign", "--amino", "--dna", "m.hmm", "s.fa"]).is_err());
        assert!(Args::try_parse_from(["hmmalign", "--dna", "--rna", "m.hmm", "s.fa"]).is_err());
        assert!(Args::try_parse_from(["hmmalign", "--amino", "--rna", "m.hmm", "s.fa"]).is_err());
        // A single alphabet flag is fine.
        assert!(Args::try_parse_from(["hmmalign", "--dna", "m.hmm", "s.fa"]).is_ok());
    }
}
