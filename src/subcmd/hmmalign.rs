//! hmmalign — align sequences to a profile HMM.

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::dp::generic_backward::g_backward;
use hmmer_pure_rs::dp::generic_decoding::g_decoding;
use hmmer_pure_rs::dp::generic_fwdback::g_forward;
use hmmer_pure_rs::dp::generic_optacc::{g_oa_trace, g_optimal_accuracy};
use hmmer_pure_rs::dp::gmx::Gmx;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::msa;
use hmmer_pure_rs::profile::{self, Profile, P7_UNILOCAL};
use hmmer_pure_rs::sequence::{self, Sequence, SequenceFormat};
use hmmer_pure_rs::trace::State;

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
    #[arg(long, action = ArgAction::SetTrue)]
    amino: bool,

    /// Assert DNA alphabet
    #[arg(long, action = ArgAction::SetTrue)]
    dna: bool,

    /// Assert RNA alphabet
    #[arg(long, action = ArgAction::SetTrue)]
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
    if args.hmmfile == PathBuf::from("-") && args.seqfile == PathBuf::from("-") {
        eprintln!(
            "ERROR: Either <hmmfile> or <seqfile> may be '-' (to read from stdin), but not both."
        );
        std::process::exit(1);
    }
    if [args.amino, args.dna, args.rna]
        .into_iter()
        .filter(|v| *v)
        .count()
        > 1
    {
        eprintln!("Error: options --amino, --dna, and --rna are mutually exclusive");
        std::process::exit(1);
    }
    let informat = args.informat.as_ref().map(|informat| {
        SequenceFormat::from_name(informat).unwrap_or_else(|| {
            eprintln!("{informat} is not a recognized input sequence file format");
            std::process::exit(1);
        })
    });

    enum OutFormat {
        Stockholm,
        A2m,
        Psiblast,
    }
    let outformat = if args.outformat.eq_ignore_ascii_case("stockholm")
        || args.outformat.eq_ignore_ascii_case("pfam")
    {
        OutFormat::Stockholm
    } else if args.outformat.eq_ignore_ascii_case("a2m") {
        OutFormat::A2m
    } else if args.outformat.eq_ignore_ascii_case("psiblast") {
        OutFormat::Psiblast
    } else {
        eprintln!(
            "unsupported --outformat {:?}; implemented: Stockholm, Pfam, A2M, PSIBLAST",
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
        OutFormat::A2m => write_a2m(
            out,
            &msa,
            hmm.abc_type == hmmer_pure_rs::alphabet::AlphabetType::Amino,
        ),
        OutFormat::Psiblast => write_psiblast(out, &msa),
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
fn write_stockholm(out: &mut dyn Write, msa: &TextMsa) {
    let name_width = msa
        .rows
        .iter()
        .map(|row| row.name.len())
        .max()
        .unwrap_or(0)
        .max("#=GC PP_cons".len())
        .max("#=GC RF".len())
        .max(13);

    writeln!(out, "# STOCKHOLM 1.0").unwrap();
    writeln!(out).unwrap();

    for row in &msa.rows {
        if let Some(acc) = &row.acc {
            if !acc.is_empty() {
                writeln!(
                    out,
                    "#=GS {:<width$} AC {}",
                    row.name,
                    acc,
                    width = name_width - 5
                )
                .unwrap();
            }
        }
        if let Some(desc) = &row.desc {
            if !desc.is_empty() {
                writeln!(
                    out,
                    "#=GS {:<width$} DE {}",
                    row.name,
                    desc,
                    width = name_width - 5
                )
                .unwrap();
            }
        }
    }
    if msa
        .rows
        .iter()
        .any(|row| row.acc.is_some() || row.desc.is_some())
    {
        writeln!(out).unwrap();
    }

    for row in &msa.rows {
        writeln!(out, "{:<width$} {}", row.name, row.aseq, width = name_width).unwrap();
        if let Some(ppline) = &row.ppline {
            writeln!(
                out,
                "{:<width$} {}",
                format!("#=GR {} PP", row.name),
                ppline,
                width = name_width
            )
            .unwrap();
        }
    }

    writeln!(
        out,
        "{:<width$} {}",
        "#=GC PP_cons",
        msa.pp_cons,
        width = name_width
    )
    .unwrap();
    writeln!(
        out,
        "{:<width$} {}",
        "#=GC RF",
        msa.rfline,
        width = name_width
    )
    .unwrap();
    writeln!(out, "//").unwrap();
}

/// Emit the alignment in A2M format: consensus columns uppercase / `-`,
/// insert columns lowercase (insert gaps suppressed); `O`/`o` mapped to `X`/`x`
/// for amino alphabets.
fn write_a2m(out: &mut dyn Write, msa: &TextMsa, is_amino: bool) {
    for row in &msa.rows {
        writeln!(out, ">{}", row.name).unwrap();
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

/// Emit PSIBLAST-style aligned rows. Easel renders nonresidue alignment
/// placeholders as `-` in this format while preserving inserted residues.
fn write_psiblast(out: &mut dyn Write, msa: &TextMsa) {
    let name_width = msa.rows.iter().map(|row| row.name.len()).max().unwrap_or(0);
    for row in &msa.rows {
        let seq: String = row
            .aseq
            .chars()
            .map(|ch| if ch == '.' { '-' } else { ch })
            .collect();
        writeln!(out, "{:<width$}  {}", row.name, seq, width = name_width).unwrap();
    }
}

/// Align every input sequence to `hmm` (Forward/Backward + posterior decoding +
/// optimal accuracy traceback) and stitch the per-sequence traces into a single
/// text MSA, computing per-column PP_cons. Counterpart to the alignment block of
/// `main()` in hmmalign.c when no `--mapali` is supplied.
fn build_text_msa(
    hmm: &hmmer_pure_rs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
    trim: bool,
) -> TextMsa {
    let traces: Vec<_> = sequences
        .iter()
        .map(|sq| {
            let mut gm = Profile::new(hmm.m, abc);
            profile::profile_config(hmm, bg, &mut gm, sq.n as i32, P7_UNILOCAL);

            let mut fwd = Gmx::new(hmm.m, sq.n);
            let mut bck = Gmx::new(hmm.m, sq.n);
            let mut pp = Gmx::new(hmm.m, sq.n);
            let mut oa = Gmx::new(hmm.m, sq.n);

            g_forward(&sq.dsq, sq.n, &gm, &mut fwd);
            g_backward(&sq.dsq, sq.n, &gm, &mut bck);
            g_decoding(&gm, &fwd, &bck, &mut pp);
            g_optimal_accuracy(&gm, &pp, &mut oa);
            let tr = g_oa_trace(&gm, &pp, &oa);

            (gm, pp, tr)
        })
        .collect();

    let (inscount, matuse, matmap, alen) = map_new_msa(hmm.m, &traces, trim);
    let mut rows = Vec::with_capacity(sequences.len());
    let mut pp_cons_sum = vec![0.0_f32; alen];
    let mut pp_cons_n = vec![0usize; alen];

    for (sq, (_gm, pp, tr)) in sequences.iter().zip(traces.iter()) {
        let (aseq, ppline) = make_text_row(abc, sq, tr, pp, &matuse, &matmap, alen, trim);
        for z in 0..tr.n {
            if tr.st[z] == State::M {
                let apos = matmap[tr.k[z]] - 1;
                pp_cons_sum[apos] += pp.mmx(tr.i[z], tr.k[z]).min(1.0);
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
    let mut inscount = mapped_insert_widths(mapped, map, hmm.m);
    if trim {
        inscount[0] = 0;
        inscount[hmm.m] = 0;
    }

    let traces: Vec<_> = sequences
        .iter()
        .map(|sq| {
            let mut gm = Profile::new(hmm.m, abc);
            profile::profile_config(hmm, bg, &mut gm, sq.n as i32, P7_UNILOCAL);

            let mut fwd = Gmx::new(hmm.m, sq.n);
            let mut bck = Gmx::new(hmm.m, sq.n);
            let mut pp = Gmx::new(hmm.m, sq.n);
            let mut oa = Gmx::new(hmm.m, sq.n);

            g_forward(&sq.dsq, sq.n, &gm, &mut fwd);
            g_backward(&sq.dsq, sq.n, &gm, &mut bck);
            g_decoding(&gm, &fwd, &bck, &mut pp);
            g_optimal_accuracy(&gm, &pp, &mut oa);
            let tr = g_oa_trace(&gm, &pp, &oa);
            (pp, tr)
        })
        .collect();

    for (_, tr) in &traces {
        update_insert_widths_from_trace(&mut inscount, tr, hmm.m);
    }

    let matuse = vec![true; hmm.m + 1];
    let matmap = compute_matmap(&inscount, &matuse, hmm.m);
    let alen = alignment_len_from_map(&inscount, &matuse, hmm.m);

    let mut rows = Vec::with_capacity(mapped.nseq + sequences.len());
    for (idx, name) in mapped.sqname.iter().enumerate() {
        rows.push(AlignmentRow {
            name: name.clone(),
            acc: None,
            desc: (!mapped.sqdesc[idx].is_empty()).then(|| mapped.sqdesc[idx].clone()),
            aseq: expand_mapped_row(&mapped.aseq[idx], map, &matmap, &inscount, hmm.m, trim),
            ppline: mapped.pp[idx]
                .as_ref()
                .map(|pp| expand_mapped_annotation(pp, map, &matmap, &inscount, hmm.m, trim)),
        });
    }

    let mut pp_cons_sum = vec![0.0_f32; alen];
    let mut pp_cons_n = vec![0usize; alen];
    for (sq, (pp, tr)) in sequences.iter().zip(traces.iter()) {
        let (mut aseq, mut ppline) = make_text_row(abc, sq, tr, pp, &matuse, &matmap, alen, trim);
        rejustify_insertions_text(&mut aseq, &mut ppline, &inscount, &matmap, &matuse, hmm.m);
        for z in 0..tr.n {
            if tr.st[z] == State::M {
                let apos = matmap[tr.k[z]] - 1;
                pp_cons_sum[apos] += pp.mmx(tr.i[z], tr.k[z]).min(1.0);
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
        rfline[matmap[k] - 1] = b'x';
    }
    let pp_cons = mapped
        .pp_cons
        .as_ref()
        .map(|pp| expand_mapped_annotation(pp, map, &matmap, &inscount, hmm.m, trim))
        .unwrap_or_else(|| ".".repeat(alen));
    let mut pp_cons_bytes = pp_cons.into_bytes();
    for (apos, ch) in pp_cons_bytes.iter_mut().enumerate() {
        if pp_cons_n[apos] > 0 {
            *ch = pp_to_char(pp_cons_sum[apos] / pp_cons_n[apos] as f32) as u8;
        }
    }
    let pp_cons = String::from_utf8(pp_cons_bytes).unwrap();

    TextMsa {
        rows,
        rfline: String::from_utf8(rfline).unwrap(),
        pp_cons,
    }
}

/// Compute the per-node insert width vector (length M+1) implied by the source
/// MSA's column-to-model `map`. Used as the initial insert budget when merging a
/// `--mapali` alignment with newly aligned sequences.
fn mapped_insert_widths(mapped: &msa::Msa, map: &[i32], m: usize) -> Vec<usize> {
    let mut ins = vec![0usize; m + 1];
    ins[0] = (map[1] - 1).max(0) as usize;
    for k in 1..m {
        ins[k] = (map[k + 1] - map[k] - 1).max(0) as usize;
    }
    ins[m] = (mapped.alen as i32 - map[m]).max(0) as usize;
    ins
}

/// Walk a trace, count I/N/C insertions per model node, and update the global
/// per-node insert-width vector to the maximum across all traces.
fn update_insert_widths_from_trace(
    inscount: &mut [usize],
    tr: &hmmer_pure_rs::trace::Trace,
    m: usize,
) {
    let mut insnum = vec![0usize; m + 1];
    for z in 1..tr.n {
        match tr.st[z] {
            State::I => insnum[tr.k[z]] += 1,
            State::N if tr.st[z - 1] == State::N => insnum[0] += 1,
            State::C if tr.st[z - 1] == State::C => insnum[m] += 1,
            _ => {}
        }
    }
    for k in 0..=m {
        inscount[k] = inscount[k].max(insnum[k]);
    }
}

/// Build `matmap[1..=M]`: the alignment column (1-based) assigned to each
/// consensus model node, accounting for retained insert widths and per-node
/// `matuse` flags.
fn compute_matmap(inscount: &[usize], matuse: &[bool], m: usize) -> Vec<usize> {
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
    matmap
}

/// Total alignment length implied by per-node insert widths plus retained
/// consensus columns (`matuse`).
fn alignment_len_from_map(inscount: &[usize], matuse: &[bool], m: usize) -> usize {
    let mut alen = inscount[0];
    for k in 1..=m {
        alen += inscount[k] + usize::from(matuse[k]);
    }
    alen
}

/// Expand a single mapped-MSA row into the merged alignment's coordinate system:
/// place consensus residues at `matmap[k]` columns and copy the original insert
/// stretches into the wider insert buckets.
fn expand_mapped_row(
    row: &[u8],
    map: &[i32],
    matmap: &[usize],
    inscount: &[usize],
    m: usize,
    trim: bool,
) -> String {
    let all_match = vec![true; m + 1];
    let mut out = vec![b'.'; alignment_len_from_map(inscount, &all_match, m)];
    for k in 1..=m {
        out[matmap[k] - 1] = b'-';
    }
    if !(trim && m == 0) {
        copy_insert_slice(
            &mut out,
            0,
            &row[0..(map[1] - 1).max(0) as usize],
            matmap,
            m,
            trim,
        );
    }
    for k in 1..=m {
        let src_cons = (map[k] - 1) as usize;
        out[matmap[k] - 1] = normalize_mapped_consensus_char(row[src_cons]);
        let next = if k == m {
            row.len()
        } else {
            (map[k + 1] - 1) as usize
        };
        copy_insert_slice(&mut out, k, &row[src_cons + 1..next], matmap, m, trim);
    }

    String::from_utf8(out).unwrap()
}

fn expand_mapped_annotation(
    row: &[u8],
    map: &[i32],
    matmap: &[usize],
    inscount: &[usize],
    m: usize,
    trim: bool,
) -> String {
    let all_match = vec![true; m + 1];
    let mut out = vec![b'.'; alignment_len_from_map(inscount, &all_match, m)];
    if !(trim && m == 0) {
        copy_insert_slice(
            &mut out,
            0,
            &row[0..(map[1] - 1).max(0) as usize],
            matmap,
            m,
            trim,
        );
    }
    for k in 1..=m {
        let src_cons = (map[k] - 1) as usize;
        out[matmap[k] - 1] = row[src_cons];
        let next = if k == m {
            row.len()
        } else {
            (map[k + 1] - 1) as usize
        };
        copy_insert_slice(&mut out, k, &row[src_cons + 1..next], matmap, m, trim);
    }
    String::from_utf8(out).unwrap()
}

/// Copy an insert-bucket slice from a source row into the destination alignment
/// starting at `matmap[bucket]` (or column 0 for bucket 0). Skips the N/C
/// terminal buckets when `trim` is set.
fn copy_insert_slice(
    out: &mut [u8],
    bucket: usize,
    slice: &[u8],
    matmap: &[usize],
    m: usize,
    trim: bool,
) {
    if trim && (bucket == 0 || bucket == m) {
        return;
    }
    let dst_start = if bucket == 0 { 0 } else { matmap[bucket] };
    for (offset, &ch) in slice.iter().enumerate() {
        out[dst_start + offset] = ch;
    }
}

/// Replace insert-style gaps (`.`, `_`) with the consensus-column gap (`-`).
fn normalize_mapped_consensus_char(ch: u8) -> u8 {
    match ch {
        b'.' | b'_' => b'-',
        other => other,
    }
}

/// Build the merged-MSA coordinate system (insert widths, matuse flags, matmap,
/// total `alen`) from a set of traces. Mirrors Easel's `esl_msa_MapNew` /
/// HMMER's `p7_tracealign_Seqs()` map-construction step.
fn map_new_msa(
    m: usize,
    traces: &[(Profile, Gmx, hmmer_pure_rs::trace::Trace)],
    trim: bool,
) -> (Vec<usize>, Vec<bool>, Vec<usize>, usize) {
    let mut inscount = vec![0usize; m + 1];
    let mut matuse = vec![true; m + 1];
    matuse[0] = false;
    let mut insnum = vec![0usize; m + 1];

    for (_, _, tr) in traces {
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
fn make_text_row(
    abc: &Alphabet,
    sq: &Sequence,
    tr: &hmmer_pure_rs::trace::Trace,
    pp: &Gmx,
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
                ppline[idx] = pp_to_char(pp.mmx(tr.i[z], tr.k[z]).min(1.0)) as u8;
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
                    ppline[apos] = pp_to_char(pp.imx(tr.i[z], tr.k[z]).min(1.0)) as u8;
                    apos += 1;
                }
            }
            State::N | State::C => {
                if !trim && tr.i[z] > 0 && apos < alen {
                    aseq[apos] =
                        (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_lowercase() as u8;
                    ppline[apos] =
                        pp_to_char(pp.xmx(tr.i[z], state_pp_index(tr.st[z])).min(1.0)) as u8;
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
