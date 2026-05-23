//! hmmsim — simulation/benchmarking tool.
//! Generates random sequences and scores them against an HMM for calibration analysis.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::dp::generic_fwdback::g_forward;
use hmmer_pure_rs::dp::generic_msv::g_msv;
use hmmer_pure_rs::dp::generic_viterbi::g_viterbi;
use hmmer_pure_rs::dp::gmx::Gmx;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::profile::{self, Profile, P7_GLOCAL, P7_LOCAL, P7_UNIGLOCAL, P7_UNILOCAL};
use hmmer_pure_rs::util::random::MersenneTwister;

#[derive(Parser)]
#[command(name = "hmmsim", about = "Score random sequences against an HMM")]
struct Args {
    hmmfile: PathBuf,

    #[arg(short = 'N', default_value = "1000")]
    n: usize,

    #[arg(short = 'L', default_value = "100")]
    l: usize,

    #[arg(long = "seed", default_value = "0")]
    seed: u32,

    #[arg(short = 'o')]
    outfile: Option<PathBuf>,

    #[arg(long = "vit", alias = "viterbi")]
    viterbi: bool,

    #[arg(long = "fwd", alias = "forward")]
    forward: bool,

    #[arg(long = "hyb")]
    hybrid: bool,

    #[arg(long = "msv")]
    msv: bool,

    #[arg(long = "fs")]
    fs: bool,

    #[arg(long = "sw")]
    sw: bool,

    #[arg(long = "ls")]
    ls: bool,

    #[arg(long = "s")]
    s: bool,
}

/// Entry point for `hmmsim`: generate `N` iid random sequences of length `L`
/// (sampled from the background distribution) and score each one against the
/// first HMM in the input file, emitting bit scores to stdout.
///
/// Scoring kernel is selectable: default `--vit`, `--fwd`, or `--msv`.
/// Used to calibrate E-value parameters and to benchmark DP engines.
/// Corresponds to `process_workunit()`/`main()` in hmmer/src/hmmsim.c, much
/// abbreviated (no MPI master/worker, no multi-HMM batch).
pub fn run(args: Vec<String>) -> ExitCode {
    let args = Args::parse_from(&args);

    if args.n == 0 {
        eprintln!("Invalid number of samples: -N must be > 0");
        return ExitCode::FAILURE;
    }
    if args.l == 0 {
        eprintln!("Invalid sequence length: -L must be > 0");
        return ExitCode::FAILURE;
    }
    let score_modes = [args.viterbi, args.forward, args.hybrid, args.msv]
        .iter()
        .filter(|&&enabled| enabled)
        .count();
    if score_modes > 1 {
        eprintln!("hmmsim scoring options --vit, --fwd, --hyb, and --msv are mutually exclusive");
        return ExitCode::FAILURE;
    }
    if args.hybrid {
        eprintln!("hmmsim --hyb is not implemented");
        return ExitCode::FAILURE;
    }
    let alignment_modes = [args.fs, args.sw, args.ls, args.s]
        .iter()
        .filter(|&&enabled| enabled)
        .count();
    if alignment_modes > 1 {
        eprintln!("hmmsim alignment options --fs, --sw, --ls, and --s are mutually exclusive");
        return ExitCode::FAILURE;
    }

    hmmer_pure_rs::logsum::p7_flogsuminit();

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    if hmms.is_empty() {
        eprintln!("Error: no HMMs found in {}", args.hmmfile.display());
        return ExitCode::FAILURE;
    }

    let hmm = &hmms[0];
    let abc = Alphabet::new(hmm.abc_type);
    let bg = Bg::new(&abc);
    let mut gm = Profile::new(hmm.m, &abc);
    let mode = if args.sw {
        P7_UNILOCAL
    } else if args.ls {
        P7_GLOCAL
    } else if args.s {
        P7_UNIGLOCAL
    } else {
        P7_LOCAL
    };
    profile::profile_config(hmm, &bg, &mut gm, args.l as i32, mode);

    let mut rng = MersenneTwister::new(args.seed);

    let mut out: Box<dyn Write> = match args.outfile {
        Some(ref path) => Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Error creating output {}: {}", path.display(), e);
            std::process::exit(1);
        })),
        None => Box::new(std::io::stdout()),
    };

    writeln!(
        out,
        "# hmmsim: {} random sequences of length {} against {}",
        args.n, args.l, hmm.name
    )
    .unwrap();
    writeln!(out, "# seed={}", args.seed).unwrap();

    for _ in 0..args.n {
        let mut dsq = vec![hmmer_pure_rs::alphabet::DSQ_SENTINEL];
        for _ in 0..args.l {
            dsq.push(rng.sample_residue(&bg.f));
        }
        dsq.push(hmmer_pure_rs::alphabet::DSQ_SENTINEL);

        let null_sc = bg.null_one(args.l);
        let mut gx = Gmx::new(gm.m, args.l);

        if args.forward {
            let sc = g_forward(&dsq, args.l, &gm, &mut gx);
            let bits = (sc - null_sc) / std::f32::consts::LN_2;
            writeln!(out, "{:.4}", bits).unwrap();
        } else if args.msv {
            let sc = g_msv(&dsq, args.l, &gm, &mut gx, 2.0);
            let bits = (sc - null_sc) / std::f32::consts::LN_2;
            writeln!(out, "{:.4}", bits).unwrap();
        } else {
            let sc = g_viterbi(&dsq, args.l, &gm, &mut gx);
            let bits = (sc - null_sc) / std::f32::consts::LN_2;
            writeln!(out, "{:.4}", bits).unwrap();
        }
    }

    ExitCode::SUCCESS
}
