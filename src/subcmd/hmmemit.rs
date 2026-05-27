//! hmmemit — sample or emit sequences from an HMM.

use std::io::{BufReader, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::{ArgAction, FromArgMatches, Parser};

use hmmer_pure_rs::alphabet::{Alphabet, Dsq, DSQ_SENTINEL};
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmm::{Hmm, DD, DM, II, IM, MD, MM};
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::profile::{
    profile_config, Profile, P7P_BM, P7P_C, P7P_E, P7P_J, P7P_LOOP, P7P_MOVE, P7P_N, P7_GLOCAL,
    P7_LOCAL, P7_UNIGLOCAL, P7_UNILOCAL,
};
use hmmer_pure_rs::sequence::Sequence;
use hmmer_pure_rs::trace::{State, Trace};
use hmmer_pure_rs::util::cmath::c_exp_to_f32;
use hmmer_pure_rs::util::random::MersenneTwister;

/// Easel's interleaved Stockholm writer wraps residues at this column count
/// (`stockholm_write(fp, msa, 200)` for `eslMSAFILE_STOCKHOLM`,
/// `esl_msafile_stockholm.c`).
const STOCKHOLM_CPL: usize = 200;

#[derive(Parser)]
#[command(
    name = "hmmemit",
    about = "Sample or emit sequences from a profile HMM"
)]
struct Args {
    /// Direct output to file, not stdout
    #[arg(short = 'o')]
    outfile: Option<PathBuf>,

    /// HMM file
    hmmfile: PathBuf,

    /// Emit alignment
    #[arg(short = 'a', action = ArgAction::SetTrue,
          conflicts_with_all = ["consensus", "fancy_consensus", "profile"])]
    alignment: bool,

    /// Emit consensus sequence
    // C (hmmemit.c:32): -c is short-only (no long form). Do NOT add `long`.
    #[arg(short = 'c', action = ArgAction::SetTrue,
          conflicts_with_all = ["alignment", "fancy_consensus", "profile"])]
    consensus: bool,

    /// Emit fancy consensus sequence
    #[arg(short = 'C', action = ArgAction::SetTrue,
          conflicts_with_all = ["alignment", "consensus", "profile"])]
    fancy_consensus: bool,

    /// Sample sequences from profile, not core model
    #[arg(short = 'p', action = ArgAction::SetTrue,
          conflicts_with_all = ["alignment", "consensus", "fancy_consensus"])]
    profile: bool,

    /// Expected sequence length for profile emission
    /// (requires -p; enforced in run() so the default value alone does not trip it)
    #[arg(short = 'L', default_value = "400")]
    length: usize,

    /// Configure profile in multihit local mode
    #[arg(long = "local", action = ArgAction::SetTrue, requires = "profile",
          conflicts_with_all = ["unilocal", "glocal", "uniglocal"])]
    local: bool,

    /// Configure profile in unihit local mode
    #[arg(long = "unilocal", action = ArgAction::SetTrue, requires = "profile",
          conflicts_with_all = ["local", "glocal", "uniglocal"])]
    unilocal: bool,

    /// Configure profile in multihit glocal mode
    #[arg(long = "glocal", action = ArgAction::SetTrue, requires = "profile",
          conflicts_with_all = ["local", "unilocal", "uniglocal"])]
    glocal: bool,

    /// Configure profile in unihit glocal mode
    #[arg(long = "uniglocal", action = ArgAction::SetTrue, requires = "profile",
          conflicts_with_all = ["local", "unilocal", "glocal"])]
    uniglocal: bool,

    /// Fancy consensus: use any-residue unless best residue probability is at least this
    /// (range 0<=x<=1; requires -C, enforced in run())
    #[arg(long = "minl", default_value = "0.0", value_parser = parse_unit_interval)]
    minl: f32,

    /// Fancy consensus: uppercase best residue when probability is at least this
    /// (range 0<=x<=1; requires -C, enforced in run())
    #[arg(long = "minu", default_value = "0.0", value_parser = parse_unit_interval)]
    minu: f32,

    /// Number of sequences to emit
    #[arg(short = 'N', conflicts_with_all = ["consensus", "fancy_consensus"])]
    n: Option<NonZeroUsize>,

    /// Random number seed
    #[arg(long = "seed", default_value = "0")]
    seed: u64,
}

/// clap value parser for `--minl`/`--minu`: a real number in the closed range
/// `0 <= x <= 1`, matching C's `eslARG_REAL` range `"0<=x<=1"`.
fn parse_unit_interval(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .parse()
        .map_err(|_| format!("takes real-valued arg in range 0<=x<=1; got {s}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("takes real-valued arg in range 0<=x<=1; got {s}"))
    }
}

/// Entry point for `hmmemit`: emit consensus or sampled sequences from each HMM
/// in the input file.
///
/// With `-c`/`-C`, writes one consensus record per HMM. Otherwise samples
/// `-N` independent sequences from the core or configured profile model.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    use clap::parser::ValueSource;
    use clap::CommandFactory;

    // Parse once via ArgMatches so we can tell apart values that were given on
    // the command line from those that came from a clap `default_value`. This is
    // needed to faithfully reproduce C's `reqs` semantics on `-L`/`--minl`/`--minu`,
    // which have non-NULL defaults but whose `reqs` (esl_opt_IsOn) fire on
    // *presence*, not value. The toggle groups (-a/-c/-C/-p, the four mode flags)
    // and `-N` incompatibilities are enforced by clap attributes on `Args`.
    let matches = match Args::command().try_get_matches_from(&args) {
        Ok(m) => m,
        Err(e) => {
            e.print().ok();
            std::process::exit(2);
        }
    };
    let given = |id: &str| matches.value_source(id) == Some(ValueSource::CommandLine);
    let args = match Args::from_arg_matches(&matches) {
        Ok(a) => a,
        Err(e) => {
            e.print().ok();
            std::process::exit(2);
        }
    };

    // C: `-L` reqs `-p`; `--minl`/`--minu` reqs `-C`. Enforced on presence.
    if given("length") && !args.profile {
        eprintln!("Error: option -L requires (or has no effect without) option -p");
        std::process::exit(1);
    }
    if (given("minl") || given("minu")) && !args.fancy_consensus {
        eprintln!("Error: options --minl and --minu require (or have no effect without) option -C");
        std::process::exit(1);
    }

    let hmms = read_hmms_maybe_stdin(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });
    if hmms.is_empty() {
        eprintln!(
            "Empty HMM file {}? No HMM data found.",
            args.hmmfile.display()
        );
        std::process::exit(1);
    }

    let mut outfile = args.outfile.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating output file: {}", e);
            std::process::exit(1);
        })
    });
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let out: &mut dyn Write = match outfile {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };

    let n = args.n.map(NonZeroUsize::get).unwrap_or(1);
    let mut rng = MersenneTwister::new(args.seed as u32);

    for h in &hmms {
        let abc = Alphabet::new(h.abc_type);
        let bg = Bg::new(&abc);

        if args.consensus {
            let seq = simple_consensus(h, &abc);
            write_fasta(out, &format!("{}-consensus", h.name), &seq);
        } else if args.fancy_consensus {
            let seq = fancy_consensus(h, &abc, args.minl, args.minu);
            write_fasta(out, &format!("{}-consensus", h.name), &seq);
        } else if args.alignment {
            let mut samples = Vec::with_capacity(n);
            for i in 0..n {
                let (seq, trace) = core_emit(h, &abc, &mut rng);
                samples.push((format!("{}-sample{}", h.name, i + 1), seq, trace));
            }
            write_stockholm_alignment(out, h, &abc, &samples);
        } else {
            let mut gm = if args.profile {
                let mut profile = Profile::new(h.m, &abc);
                profile_config(
                    h,
                    &bg,
                    &mut profile,
                    args.length as i32,
                    profile_mode(&args),
                );
                Some(profile)
            } else {
                None
            };
            for i in 0..n {
                let (seq, _) = if let Some(profile) = gm.as_mut() {
                    profile_emit(h, profile, &bg, &mut rng)
                } else {
                    core_emit(h, &abc, &mut rng)
                };
                write_fasta(
                    out,
                    &format!("{}-sample{}", h.name, i + 1),
                    &sequence_text(&abc, &seq),
                );
            }
        }
    }
    std::process::ExitCode::SUCCESS
}

fn read_hmms_maybe_stdin(path: &std::path::Path) -> hmmer_pure_rs::errors::HmmerResult<Vec<Hmm>> {
    if path == std::path::Path::new("-") {
        let stdin = std::io::stdin();
        hmmfile::read_hmms_auto(BufReader::new(stdin.lock()))
    } else {
        hmmfile::read_hmm_file_auto(path)
    }
}

fn profile_mode(args: &Args) -> i32 {
    if args.unilocal {
        P7_UNILOCAL
    } else if args.glocal {
        P7_GLOCAL
    } else if args.uniglocal {
        P7_UNIGLOCAL
    } else {
        P7_LOCAL
    }
}

fn simple_consensus(hmm: &Hmm, abc: &Alphabet) -> Vec<u8> {
    let mut seq = Vec::with_capacity(hmm.m);
    for node in 1..=hmm.m {
        if masked_node(hmm, node) {
            seq.push(abc.sym[abc.unknown_code() as usize]);
        } else {
            seq.push(abc.sym[argmax(&hmm.mat[node][..abc.k])]);
        }
    }
    seq
}

fn fancy_consensus(hmm: &Hmm, abc: &Alphabet, min_lower: f32, min_upper: f32) -> Vec<u8> {
    let unknown = (abc.sym[abc.unknown_code() as usize] as char).to_ascii_lowercase() as u8;
    let mut seq = Vec::with_capacity(hmm.m);
    for node in 1..=hmm.m {
        if masked_node(hmm, node) {
            seq.push(unknown);
        } else {
            let x = argmax(&hmm.mat[node][..abc.k]);
            let p = hmm.mat[node][x];
            let c = abc.sym[x] as char;
            if p < min_lower {
                seq.push(unknown);
            } else if p >= min_upper {
                seq.push(c.to_ascii_uppercase() as u8);
            } else {
                seq.push(c.to_ascii_lowercase() as u8);
            }
        }
    }
    seq
}

fn core_emit(hmm: &Hmm, abc: &Alphabet, rng: &mut MersenneTwister) -> (Sequence, Trace) {
    let mut sq = Sequence::new();
    let mut tr = Trace::new();
    let mut k = 0usize;
    let mut i = 0usize;
    let mut state = State::B;

    tr.append(state, k, i);
    while state != State::E {
        state = match state {
            State::B | State::M => match rng.sample_discrete(&hmm.t[k][MM..=MD]) {
                0 => State::M,
                1 => State::I,
                _ => State::D,
            },
            State::I => match rng.sample_discrete(&hmm.t[k][IM..=II]) {
                0 => State::M,
                _ => State::I,
            },
            State::D => match rng.sample_discrete(&hmm.t[k][DM..=DD]) {
                0 => State::M,
                _ => State::D,
            },
            _ => panic!("invalid core emit state"),
        };

        if matches!(state, State::M | State::D) {
            k += 1;
        }
        if matches!(state, State::M | State::I) {
            i += 1;
        }
        if k == hmm.m + 1 {
            if state == State::M {
                state = State::E;
                k = 0;
            } else {
                panic!("core emitter failed to reach E from terminal match state");
            }
        }

        let residue: Option<Dsq> = match state {
            State::M => Some(rng.sample_residue(&hmm.mat[k][..abc.k])),
            State::I => Some(rng.sample_residue(&hmm.ins[k][..abc.k])),
            _ => None,
        };
        tr.append(state, k, i);
        if let Some(x) = residue {
            sq.dsq.push(x);
        }
    }
    sq.dsq.push(DSQ_SENTINEL);
    sq.n = sq.dsq.len().saturating_sub(2);
    sq.l = sq.n;
    tr.m = hmm.m;
    tr.l = i;
    (sq, tr)
}

fn profile_emit(hmm: &Hmm, gm: &Profile, bg: &Bg, rng: &mut MersenneTwister) -> (Sequence, Trace) {
    let mut sq = Sequence::new();
    let mut tr = Trace::new();
    let mut k = 0usize;
    let mut i = 0usize;
    let mut kend = hmm.m;
    let mut state = State::N;

    tr.append(State::S, k, i);
    tr.append(State::N, k, i);
    while state != State::T {
        let previous = state;
        state = match state {
            State::B => {
                if gm.is_local() {
                    let (kstart, end) = sample_endpoints(rng, gm);
                    k = kstart;
                    kend = end;
                    State::M
                } else {
                    match rng.sample_discrete(&hmm.t[0][MM..=MD]) {
                        0 => {
                            k = 1;
                            State::M
                        }
                        1 => {
                            k = 0;
                            State::I
                        }
                        _ => {
                            k = 1;
                            State::D
                        }
                    }
                }
            }
            State::M => {
                if k == kend {
                    State::E
                } else {
                    match rng.sample_discrete(&hmm.t[k][MM..=MD]) {
                        0 => State::M,
                        1 => State::I,
                        _ => State::D,
                    }
                }
            }
            State::D => {
                if k == kend {
                    State::E
                } else if rng.sample_discrete(&hmm.t[k][DM..=DD]) == 0 {
                    State::M
                } else {
                    State::D
                }
            }
            State::I => {
                if rng.sample_discrete(&hmm.t[k][IM..=II]) == 0 {
                    State::M
                } else {
                    State::I
                }
            }
            State::N => {
                if sample_special(rng, gm, P7P_N) == P7P_MOVE {
                    State::B
                } else {
                    State::N
                }
            }
            State::E => {
                if sample_special(rng, gm, P7P_E) == P7P_MOVE {
                    State::C
                } else {
                    State::J
                }
            }
            State::C => {
                if sample_special(rng, gm, P7P_C) == P7P_MOVE {
                    State::T
                } else {
                    State::C
                }
            }
            State::J => {
                if sample_special(rng, gm, P7P_J) == P7P_MOVE {
                    State::B
                } else {
                    State::J
                }
            }
            _ => panic!("invalid profile emit state"),
        };

        if state == State::E {
            k = 0;
        } else if state == State::M && previous != State::B {
            k += 1;
        } else if state == State::D {
            k += 1;
        }

        let residue = if state == State::M {
            Some(rng.sample_residue(&hmm.mat[k][..gm.abc_k]))
        } else if state == State::I {
            Some(rng.sample_residue(&hmm.ins[k][..gm.abc_k]))
        } else if matches!(state, State::N | State::C | State::J) && previous == state {
            Some(rng.sample_residue(&bg.f[..gm.abc_k]))
        } else {
            None
        };

        if let Some(x) = residue {
            i += 1;
            sq.dsq.push(x);
        }
        tr.append(state, k, i);
    }

    sq.dsq.push(DSQ_SENTINEL);
    sq.n = sq.dsq.len().saturating_sub(2);
    sq.l = sq.n;
    tr.m = hmm.m;
    tr.l = i;
    (sq, tr)
}

fn sample_special(rng: &mut MersenneTwister, gm: &Profile, state: usize) -> usize {
    let probs = [
        c_exp_to_f32(gm.xsc[state][P7P_LOOP] as f64),
        c_exp_to_f32(gm.xsc[state][P7P_MOVE] as f64),
    ];
    rng.sample_discrete(&probs)
}

fn sample_endpoints(rng: &mut MersenneTwister, gm: &Profile) -> (usize, usize) {
    let mut pstart = vec![0.0_f32; gm.m + 1];
    for (k, pk) in pstart.iter_mut().enumerate().take(gm.m + 1).skip(1) {
        *pk = c_exp_to_f32(gm.tsc(k - 1, P7P_BM) as f64) * (gm.m - k + 1) as f32;
    }
    let kstart = rng.sample_discrete(&pstart);
    let kend = kstart + rng.roll(gm.m - kstart + 1);
    (kstart, kend)
}

fn write_fasta(out: &mut dyn Write, name: &str, seq: &[u8]) {
    writeln!(out, ">{}", name).unwrap();
    for chunk in seq.chunks(60) {
        writeln!(out, "{}", std::str::from_utf8(chunk).unwrap()).unwrap();
    }
}

fn sequence_text(abc: &Alphabet, seq: &Sequence) -> Vec<u8> {
    seq.dsq[1..=seq.n]
        .iter()
        .map(|&x| abc.sym[x as usize])
        .collect()
}

/// Write the sampled core-model alignment as interleaved Stockholm, matching
/// C's `emit_alignment()` (`hmmemit.c`) which builds an MSA via
/// `p7_tracealign_Seqs(..., p7_ALL_CONSENSUS_COLS, ...)` and writes it with
/// `esl_msafile_Write(ofp, msa, eslMSAFILE_STOCKHOLM)`.
///
/// Reproduces the Easel Stockholm writer layout (`esl_msafile_stockholm.c`,
/// `stockholm_write`): residues wrapped at [`STOCKHOLM_CPL`] columns into
/// interleaved blocks separated by a blank line, a left margin sized to the
/// widest of `maxname+1` and `maxgc+6` (`maxgc == 2` for the RF tag), and a
/// `#=GC RF` consensus line per block. Inserts are rejustified to match Easel's
/// `rejustify_insertions_digital()` (`tracealign.c`).
fn write_stockholm_alignment(
    out: &mut dyn Write,
    hmm: &Hmm,
    abc: &Alphabet,
    samples: &[(String, Sequence, Trace)],
) {
    let traces: Vec<&Trace> = samples.iter().map(|(_, _, tr)| tr).collect();
    let (inscount, matmap, alen) = map_core_alignment(hmm.m, &traces);

    // Build full-width rows, then rejustify inserts exactly as Easel does.
    let rows: Vec<Vec<u8>> = samples
        .iter()
        .map(|(_, seq, trace)| {
            let mut row = render_core_alignment_row(abc, seq, trace, &inscount, &matmap, alen);
            rejustify_insertions(abc, &mut row, hmm.m, &inscount, &matmap);
            row
        })
        .collect();
    let rf = core_rf_line(hmm.m, &inscount, &matmap, alen);

    // Easel margin = max(maxname+1, maxgc+6). maxgc = 2 (the "RF" tag, clamped
    // to a minimum of 2 when msa->rf is present). Sequence lines are `%-*s `
    // with width margin-1; the `#=GC RF` line is `#=GC %-*s ` with the RF tag
    // padded to margin-6 — both land the residues at column `margin`.
    let maxname = samples
        .iter()
        .map(|(name, _, _)| name.len())
        .max()
        .unwrap_or(0);
    let maxgc = 2usize;
    let margin = (maxname + 1).max(maxgc + 6);

    writeln!(out, "# STOCKHOLM 1.0").unwrap();
    writeln!(out).unwrap();

    let mut currpos = 0usize;
    loop {
        let acpl = (alen - currpos).min(STOCKHOLM_CPL);
        if currpos > 0 {
            writeln!(out).unwrap();
        }
        for ((name, _, _), row) in samples.iter().zip(rows.iter()) {
            writeln!(
                out,
                "{:<width$} {}",
                name,
                std::str::from_utf8(&row[currpos..currpos + acpl]).unwrap(),
                width = margin - 1
            )
            .unwrap();
        }
        writeln!(
            out,
            "#=GC {:<width$} {}",
            "RF",
            std::str::from_utf8(&rf[currpos..currpos + acpl]).unwrap(),
            width = margin - 6
        )
        .unwrap();

        currpos += acpl;
        if currpos >= alen {
            break;
        }
    }
    writeln!(out, "//").unwrap();
}

fn map_core_alignment(m: usize, traces: &[&Trace]) -> (Vec<usize>, Vec<usize>, usize) {
    let mut inscount = vec![0usize; m + 1];
    let mut insnum = vec![0usize; m + 1];
    for tr in traces {
        insnum.fill(0);
        for z in 1..tr.n {
            if tr.st[z] == State::I {
                insnum[tr.k[z]] += 1;
            }
        }
        for k in 0..=m {
            inscount[k] = inscount[k].max(insnum[k]);
        }
    }

    let mut matmap = vec![0usize; m + 1];
    let mut alen = inscount[0];
    for k in 1..=m {
        matmap[k] = alen;
        alen += 1 + inscount[k];
    }
    (inscount, matmap, alen)
}

fn render_core_alignment_row(
    abc: &Alphabet,
    sq: &Sequence,
    tr: &Trace,
    inscount: &[usize],
    matmap: &[usize],
    alen: usize,
) -> Vec<u8> {
    let mut row = vec![b'.'; alen];
    for k in 1..matmap.len() {
        row[matmap[k]] = b'-';
    }
    let mut used_insert = vec![0usize; inscount.len()];
    for z in 0..tr.n {
        match tr.st[z] {
            State::M => row[matmap[tr.k[z]]] = abc.sym[sq.dsq[tr.i[z]] as usize],
            State::D => row[matmap[tr.k[z]]] = b'-',
            State::I => {
                let base = if tr.k[z] == 0 { 0 } else { matmap[tr.k[z]] + 1 };
                let col = base + used_insert[tr.k[z]];
                row[col] = (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_lowercase() as u8;
                used_insert[tr.k[z]] += 1;
            }
            _ => {}
        }
    }
    row
}

/// Rejustify the inserted residues of a single alignment row, mirroring Easel's
/// `rejustify_insertions_digital()` (`tracealign.c`). For each node `k` whose
/// insert column run is longer than 1, residues are split in half: the first
/// `nres/2` stay left-justified and the remainder are pushed flush-right within
/// the run (gaps fall in the middle). The N-terminal run (`k == 0`) is fully
/// right-justified (`nres -> 0` left), and the C-terminal run (`k == m`) is left
/// untouched (the C loop runs `k = 0 .. m-1`).
///
/// Columns are 0-based here: match node `k` (`k >= 1`) occupies `matmap[k]` and
/// its insert run is `matmap[k]+1 ..= matmap[k]+inscount[k]`; the N-terminal run
/// is `0 ..= inscount[0]-1`.
fn rejustify_insertions(
    abc: &Alphabet,
    row: &mut [u8],
    m: usize,
    inscount: &[usize],
    matmap: &[usize],
) {
    let is_residue = |c: u8| {
        let up = c.to_ascii_uppercase();
        up != b'.' && up != b'-' && abc.sym[..abc.k].contains(&up)
    };

    for k in 0..m {
        if inscount[k] <= 1 {
            continue;
        }
        // Insert run [lo, hi] inclusive, 0-based.
        let (lo, hi) = if k == 0 {
            (0usize, inscount[0] - 1)
        } else {
            (matmap[k] + 1, matmap[k] + inscount[k])
        };

        let nres = row[lo..=hi].iter().filter(|&&c| is_residue(c)).count();
        // N-terminus is fully right-justified; otherwise split in half (the
        // number of residues that stay left-justified).
        let nleft = if k == 0 { 0 } else { nres / 2 };
        let split = lo + nleft; // first column that receives right-justified content

        // Pack residues toward the right end (hi down to split), then gap-fill.
        let mut npos = hi as isize;
        let mut opos = hi as isize;
        while opos >= split as isize {
            if !is_residue(row[opos as usize]) {
                opos -= 1;
            } else {
                row[npos as usize] = row[opos as usize];
                npos -= 1;
                opos -= 1;
            }
        }
        while npos >= split as isize {
            row[npos as usize] = b'.';
            npos -= 1;
        }
    }
}

fn core_rf_line(m: usize, inscount: &[usize], matmap: &[usize], alen: usize) -> Vec<u8> {
    let mut rf = vec![b'.'; alen];
    for k in 1..=m {
        rf[matmap[k]] = b'x';
        for col in matmap[k] + 1..matmap[k] + 1 + inscount[k] {
            rf[col] = b'.';
        }
    }
    rf
}

fn masked_node(hmm: &Hmm, node: usize) -> bool {
    hmm.mm
        .as_ref()
        .and_then(|mm| mm.get(node))
        .is_some_and(|&c| c == b'm')
}

fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > best_v {
            best_i = i;
            best_v = x;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmmemit_accepts_c_mode_options_but_rejects_incompatible_pairs() {
        assert!(Args::try_parse_from(["hmmemit", "-c", "model.hmm"]).is_ok());
        assert!(Args::try_parse_from(["hmmemit", "-C", "--minl", "0.4", "model.hmm"]).is_ok());
        assert!(Args::try_parse_from(["hmmemit", "-a", "-N", "3", "model.hmm"]).is_ok());
        assert!(
            Args::try_parse_from(["hmmemit", "-p", "-L", "20", "--unilocal", "model.hmm"]).is_ok()
        );
        let args = Args::try_parse_from(["hmmemit", "-N", "3", "model.hmm"]).unwrap();
        assert_eq!(args.n.map(NonZeroUsize::get), Some(3));
    }

    #[test]
    fn hmmemit_c_is_short_only_no_consensus_alias() {
        // C (hmmemit.c:32): -c is short-only. The short form works; the
        // previously-erroneous --consensus long alias must be rejected.
        assert!(Args::try_parse_from(["hmmemit", "-c", "model.hmm"]).is_ok());
        assert!(Args::try_parse_from(["hmmemit", "--consensus", "model.hmm"]).is_err());
    }

    #[test]
    fn hmmemit_rejects_nonpositive_n() {
        assert!(Args::try_parse_from(["hmmemit", "-N", "0", "model.hmm"]).is_err());
    }

    #[test]
    fn hmmemit_emit_mode_toggle_is_mutually_exclusive() {
        // C EMITOPTS "-a,-c,-C,-p": all pairs conflict at parse time.
        assert!(Args::try_parse_from(["hmmemit", "-a", "-c", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "-a", "-C", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "-a", "-p", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "-c", "-C", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "-c", "-p", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "-C", "-p", "model.hmm"]).is_err());
    }

    #[test]
    fn hmmemit_n_incompatible_with_consensus_modes() {
        // C: -N incomp "-c,-C". -N with -a / -p / alone is allowed.
        assert!(Args::try_parse_from(["hmmemit", "-N", "3", "-c", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "-N", "3", "-C", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "-N", "3", "-a", "model.hmm"]).is_ok());
        assert!(Args::try_parse_from(["hmmemit", "-N", "3", "-p", "model.hmm"]).is_ok());
    }

    #[test]
    fn hmmemit_mode_flags_toggle_and_require_profile() {
        // C MODEOPTS: the four mode flags are mutually exclusive...
        assert!(
            Args::try_parse_from(["hmmemit", "-p", "--local", "--glocal", "model.hmm"]).is_err()
        );
        assert!(
            Args::try_parse_from(["hmmemit", "-p", "--unilocal", "--uniglocal", "model.hmm"])
                .is_err()
        );
        // ...and require -p.
        assert!(Args::try_parse_from(["hmmemit", "--local", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "--glocal", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "-p", "--glocal", "model.hmm"]).is_ok());
    }

    #[test]
    fn hmmemit_minl_minu_range_is_validated() {
        // C range "0<=x<=1".
        assert!(Args::try_parse_from(["hmmemit", "-C", "--minl", "5", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "-C", "--minu", "-0.1", "model.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmemit", "-C", "--minl", "0.0", "model.hmm"]).is_ok());
        assert!(Args::try_parse_from(["hmmemit", "-C", "--minu", "1.0", "model.hmm"]).is_ok());
    }

    #[test]
    fn rejustify_splits_inserts_like_easel() {
        // One node (k=1) with an insert run of width 3 holding two residues.
        // matmap[1]=0 (match col), insert run = cols 1..=3, match2 at col 4.
        // Easel: nres=2 -> nleft=1; left residue stays at col1, the other
        // flushes right to col3, gap in the middle: "s.s".
        let abc = Alphabet::amino();
        let inscount = vec![0usize, 3usize, 0usize];
        let matmap = vec![0usize, 0usize, 4usize];
        let mut row = b"Ass.-".to_vec(); // left-justified inserts before rejustify
        rejustify_insertions(&abc, &mut row, 2, &inscount, &matmap);
        assert_eq!(&row, b"As.s-");
    }

    #[test]
    fn simple_consensus_uses_masked_unknown_code() {
        let mut hmm = Hmm::new(2, hmmer_pure_rs::alphabet::AlphabetType::Amino, 20);
        hmm.name = "x".to_string();
        hmm.mat[1][0] = 1.0;
        hmm.mat[2][1] = 1.0;
        hmm.mm = Some(vec![b' ', b'.', b'm', b'\0']);
        let abc = Alphabet::amino();
        assert_eq!(simple_consensus(&hmm, &abc), b"AX");
    }

    #[test]
    fn fancy_consensus_applies_thresholds_and_case() {
        let mut hmm = Hmm::new(3, hmmer_pure_rs::alphabet::AlphabetType::Amino, 20);
        hmm.mat[1][0] = 0.2;
        hmm.mat[2][1] = 0.5;
        hmm.mat[3][2] = 0.9;
        let abc = Alphabet::amino();
        assert_eq!(fancy_consensus(&hmm, &abc, 0.4, 0.8), b"xcD");
    }

    #[test]
    fn core_alignment_renders_stockholm_shape() {
        let mut hmm = Hmm::new(2, hmmer_pure_rs::alphabet::AlphabetType::Amino, 20);
        hmm.t[0][MM] = 1.0;
        hmm.t[1][MM] = 1.0;
        hmm.t[2][MM] = 1.0;
        hmm.mat[1][0] = 1.0;
        hmm.mat[2][1] = 1.0;
        let abc = Alphabet::amino();
        let mut rng = MersenneTwister::new(1);
        let (seq, trace) = core_emit(&hmm, &abc, &mut rng);
        let mut out = Vec::new();
        write_stockholm_alignment(
            &mut out,
            &hmm,
            &abc,
            &[("h-sample1".to_string(), seq, trace)],
        );
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("# STOCKHOLM 1.0\n\n"));
        assert!(text.contains("h-sample1 AC"));
        assert!(text.contains("#=GC RF   xx"));
        assert!(text.ends_with("//\n"));
    }

    #[test]
    fn profile_emit_matches_c_fixed_seed_fn3_modes() {
        let hmm = hmmfile::read_hmm_file(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/fn3.hmm"
        )))
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        let abc = Alphabet::new(hmm.abc_type);
        let bg = Bg::new(&abc);

        let cases = [
            (
                P7_LOCAL,
                b"DIYCALVTEVALHMLTKFRHDAEEAKIHVTLLKTMEYGLQGTQSPHPSGISESHLVALWFRSIGMILGKKAYYKRETQQLVNEFTIAAQGRRYTQEELTEDDNTSTEYEVHQQWVANTTEI".as_slice(),
            ),
            (
                P7_UNILOCAL,
                b"DIYCALVTEVALHMLTKFRHDGADGAGHFTVDKIVPPYAADGQGTQSPHPSGISESHLVALWFRSIGMILGKKAYYKRETQQLVNEFTIAAQGRRYTQEELTEDLDMVR".as_slice(),
            ),
            (
                P7_GLOCAL,
                b"DIYCALVTEVALHMLTKFRHDIDKPTILKSKEAHERELTLQWSPSQYSGGSRDSMFKVTYSAFNSSKSQKITVEEKGPQYAITYLNAEVGFALKVQTVRDEGTGDWHMVRTMEVAHHVVVCNVTDNKVYVSWAKARAPNARNTFYRLVYKPSNSMHMWKERIRKSNHGTSQAVSDEGLLEGEQYGIKVSAVTPNLPQPQSRWLMVKPHMEELG".as_slice(),
            ),
            (
                P7_UNIGLOCAL,
                b"DIYCALVTEVALHMLTKFRHDGADGAGHFTVDKIVPPYAADGQGTQSPKETSDPQALVVYSAKSVTLNWNHPEEGIRNYSGFYYSLDEVEAAPGPNSTQDEETEDTFGDGVLKVAGLVVVANYTFKLTYVSGADFRNA".as_slice(),
            ),
        ];

        for (mode, expected) in cases {
            let mut gm = Profile::new(hmm.m, &abc);
            profile_config(&hmm, &bg, &mut gm, 25, mode);
            let mut rng = MersenneTwister::new(7);
            let (seq, trace) = profile_emit(&hmm, &gm, &bg, &mut rng);
            assert_eq!(sequence_text(&abc, &seq), expected);
            assert_eq!(trace.st.first(), Some(&State::S));
            assert_eq!(trace.st.last(), Some(&State::T));
        }
    }
}
