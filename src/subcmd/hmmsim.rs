//! hmmsim — simulation/benchmarking tool.
//! Generates random sequences and scores them against an HMM for calibration analysis.

use std::io::Write;
use std::process::ExitCode;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::dp::generic_fwdback::g_forward;
use hmmer_pure_rs::dp::generic_msv::g_msv;
use hmmer_pure_rs::dp::generic_viterbi::g_viterbi;
use hmmer_pure_rs::dp::gmx::Gmx;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::util::random::MersenneTwister;

#[derive(Parser)]
#[command(name = "hmmsim", about = "Score random sequences against an HMM")]
struct Args {
    hmmfile: PathBuf,

    #[arg(short = 'N', default_value = "1000")]
    n: usize,

    #[arg(short = 'L', default_value = "400")]
    l: usize,

    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    #[arg(long = "viterbi")]
    viterbi: bool,

    #[arg(long = "forward")]
    forward: bool,
}

pub fn run(args: Vec<String>) -> ExitCode {
    let args = Args::parse_from(&args);

    hmmer_pure_rs::logsum::p7_flogsuminit();

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let hmm = &hmms[0];
    let abc = Alphabet::new(hmm.abc_type);
    let bg = Bg::new(&abc);
    let mut gm = Profile::new(hmm.m, &abc);
    profile::profile_config(hmm, &bg, &mut gm, args.l as i32, P7_LOCAL);

    let mut rng = MersenneTwister::new(args.seed);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "# hmmsim: {} random sequences of length {} against {}", args.n, args.l, hmm.name).unwrap();
    writeln!(out, "# seed={}", args.seed).unwrap();

    for _ in 0..args.n {
        let mut dsq = vec![hmmer_pure_rs::alphabet::DSQ_SENTINEL];
        for _ in 0..args.l {
            dsq.push(rng.sample_residue(&bg.f));
        }
        dsq.push(hmmer_pure_rs::alphabet::DSQ_SENTINEL);

        let null_sc = bg.null_one(args.l);
        let mut gx = Gmx::new(gm.m, args.l);

        if args.viterbi {
            let sc = g_viterbi(&dsq, args.l, &gm, &mut gx);
            let bits = (sc - null_sc) / std::f32::consts::LN_2;
            writeln!(out, "{:.4}", bits).unwrap();
        } else if args.forward {
            let sc = g_forward(&dsq, args.l, &gm, &mut gx);
            let bits = (sc - null_sc) / std::f32::consts::LN_2;
            writeln!(out, "{:.4}", bits).unwrap();
        } else {
            let sc = g_msv(&dsq, args.l, &gm, &mut gx, 2.0);
            let bits = (sc - null_sc) / std::f32::consts::LN_2;
            writeln!(out, "{:.4}", bits).unwrap();
        }
    }

    ExitCode::SUCCESS
}
