//! hmmsim — simulation/benchmarking tool.
//!
//! Generates random sequences and scores them against an HMM for calibration
//! analysis. Score/statistical artifacts and Forward tail-mass controls are
//! implemented, including C-shaped main-summary rows for Forward,
//! Viterbi/MSV, and hybrid modes.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::calibrate::p7_lambda;
use hmmer_pure_rs::dp::generic_fwdback::{g_forward, g_hybrid};
use hmmer_pure_rs::dp::generic_msv::g_msv;
use hmmer_pure_rs::dp::generic_viterbi::g_viterbi;
use hmmer_pure_rs::dp::gmx::Gmx;
use hmmer_pure_rs::hmm::Hmm;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::output::{fmt_fixed4, fmt_fixed6, fmt_g};
use hmmer_pure_rs::profile::{
    self, Profile, P7P_C, P7P_J, P7P_LOOP, P7P_MOVE, P7P_N, P7_GLOCAL, P7_LOCAL, P7_UNIGLOCAL,
    P7_UNILOCAL,
};
use hmmer_pure_rs::stats::{exponential, gumbel};
use hmmer_pure_rs::trace::{g_trace, State};
use hmmer_pure_rs::util::cmath::{c_log_f64, c_logf_to_f32, c_sqrt_f64, ESL_CONST_LOG2};
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

    #[arg(short = 'a')]
    alignment_stats: bool,

    #[arg(short = 'v')]
    verbose: bool,

    #[arg(
        long = "stall",
        help = "arrest after start: for debugging MPI under gdb"
    )]
    stall: bool,

    #[arg(long = "pfile")]
    pfile: Option<PathBuf>,

    #[arg(long = "efile")]
    efile: Option<PathBuf>,

    #[arg(long = "ffile")]
    ffile: Option<PathBuf>,

    #[arg(long = "xfile")]
    xfile: Option<PathBuf>,

    #[arg(long = "afile")]
    afile: Option<PathBuf>,

    #[arg(long = "pthresh", default_value = "0.02")]
    pthresh: f64,

    #[arg(long = "tmin", default_value = "0.02", value_parser = parse_positive_f64)]
    tmin: f64,

    #[arg(long = "tmax", default_value = "0.02", value_parser = parse_positive_f64)]
    tmax: f64,

    #[arg(long = "tpoints", default_value = "1", value_parser = parse_positive_usize)]
    tpoints: usize,

    #[arg(long = "tlinear")]
    tlinear: bool,

    #[arg(long = "vit", alias = "viterbi")]
    viterbi: bool,

    #[arg(long = "fwd", alias = "forward")]
    forward: bool,

    #[arg(long = "hyb")]
    hybrid: bool,

    #[arg(long = "msv")]
    msv: bool,

    /// Use optimized scoring kernels when available (accepted as compatibility no-op)
    #[arg(long = "fast")]
    fast: bool,

    #[arg(long = "bgflat")]
    bgflat: bool,

    #[arg(long = "bgcomp")]
    bgcomp: bool,

    #[arg(long = "x-no-lengthmodel")]
    no_lengthmodel: bool,

    #[arg(long = "nu")]
    nu: Option<f32>,

    #[arg(long = "EmL", default_value = "200", value_parser = parse_positive_usize)]
    em_l: usize,

    #[arg(long = "EmN", default_value = "200", value_parser = parse_positive_usize)]
    em_n: usize,

    #[arg(long = "EvL", default_value = "200", value_parser = parse_positive_usize)]
    ev_l: usize,

    #[arg(long = "EvN", default_value = "200", value_parser = parse_positive_usize)]
    ev_n: usize,

    #[arg(long = "EfL", default_value = "100", value_parser = parse_positive_usize)]
    ef_l: usize,

    #[arg(long = "EfN", default_value = "200", value_parser = parse_positive_usize)]
    ef_n: usize,

    #[arg(long = "Eft", default_value = "0.04", value_parser = parse_open_unit_f64)]
    eft: f64,

    #[arg(long = "fs")]
    fs: bool,

    #[arg(long = "sw")]
    sw: bool,

    #[arg(long = "ls")]
    ls: bool,

    #[arg(long = "s")]
    s: bool,
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

fn parse_open_unit_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid probability: {e}"))?;
    if value > 0.0 && value < 1.0 {
        Ok(value)
    } else {
        Err("value must be > 0 and < 1".to_string())
    }
}

fn parse_positive_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid positive real: {e}"))?;
    if value > 0.0 && value.is_finite() {
        Ok(value)
    } else {
        Err("value must be finite and > 0".to_string())
    }
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
    let args_raw = args.clone();
    let args = Args::parse_from(&args);
    let _fast_compat = args.fast;
    let _verbose_compat = args.verbose;
    let _stall_compat = args.stall;

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
    let alignment_modes = [args.fs, args.sw, args.ls, args.s]
        .iter()
        .filter(|&&enabled| enabled)
        .count();
    if alignment_modes > 1 {
        eprintln!("hmmsim alignment options --fs, --sw, --ls, and --s are mutually exclusive");
        return ExitCode::FAILURE;
    }
    if args.alignment_stats && score_modes > 0 && !args.viterbi {
        eprintln!("hmmsim -a requires Viterbi scoring");
        return ExitCode::FAILURE;
    }
    if args.afile.is_some() && !args.alignment_stats {
        eprintln!("hmmsim --afile requires -a");
        return ExitCode::FAILURE;
    }
    if arg_was_used(&args_raw, "--pthresh") && args.ffile.is_none() {
        eprintln!("hmmsim --pthresh requires --ffile");
        return ExitCode::FAILURE;
    }
    if args.nu.is_some() && !args.msv {
        eprintln!("hmmsim --nu is only supported with --msv");
        return ExitCode::FAILURE;
    }
    if args.nu.is_some() && args.fast {
        eprintln!("hmmsim --nu cannot be used with --fast");
        return ExitCode::FAILURE;
    }
    let nu = args.nu.unwrap_or(2.0);
    if nu <= 1.0 || !nu.is_finite() {
        eprintln!("hmmsim --nu must be finite and > 1.0");
        return ExitCode::FAILURE;
    }
    if !(0.0..=1.0).contains(&args.pthresh) || !args.pthresh.is_finite() {
        eprintln!("hmmsim --pthresh must be finite and between 0.0 and 1.0");
        return ExitCode::FAILURE;
    }

    hmmer_pure_rs::logsum::p7_flogsuminit();

    let hmms = hmmfile::read_hmm_file_auto(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    if hmms.is_empty() {
        eprintln!("Error: no HMMs found in {}", args.hmmfile.display());
        return ExitCode::FAILURE;
    }

    let hmm = &hmms[0];
    let abc = Alphabet::new(hmm.abc_type);
    let mut bg = if args.bgflat {
        Bg::new_uniform(&abc)
    } else {
        Bg::new(&abc)
    };
    if args.bgcomp {
        bg.f = profile::hmm_average_match_composition(hmm);
        bg.fhmm_e0 = bg.f.clone();
        bg.fhmm_e1 = bg.f.clone();
    }
    bg.set_length(args.l);

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
    if args.no_lengthmodel {
        elide_length_model(&mut gm, &mut bg);
    }

    let mut rng = MersenneTwister::new(args.seed);

    let score_mode = if args.forward {
        ScoreMode::Forward
    } else if args.hybrid {
        ScoreMode::Hybrid
    } else if args.msv {
        ScoreMode::Msv
    } else {
        ScoreMode::Viterbi
    };
    let model_params = calibrated_model_params(score_mode, hmm, &abc, &bg, &args, &mut rng);

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

    let mut scores = Vec::with_capacity(args.n);
    let mut alilens = Vec::with_capacity(args.n);

    for _ in 0..args.n {
        let mut dsq = vec![hmmer_pure_rs::alphabet::DSQ_SENTINEL];
        for _ in 0..args.l {
            dsq.push(rng.sample_residue(&bg.f));
        }
        dsq.push(hmmer_pure_rs::alphabet::DSQ_SENTINEL);

        let null_sc = bg.null_one(args.l);
        let mut gx = Gmx::new(gm.m, args.l);

        let bits = if args.forward {
            let sc = g_forward(&dsq, args.l, &gm, &mut gx);
            ((sc - null_sc) as f64 / ESL_CONST_LOG2) as f32
        } else if args.hybrid {
            let sc = g_hybrid(&dsq, args.l, &gm, &mut gx);
            ((sc - null_sc) as f64 / ESL_CONST_LOG2) as f32
        } else if args.msv {
            let sc = g_msv(&dsq, args.l, &gm, &mut gx, nu);
            ((sc - null_sc) as f64 / ESL_CONST_LOG2) as f32
        } else {
            let sc = g_viterbi(&dsq, args.l, &gm, &mut gx);
            ((sc - null_sc) as f64 / ESL_CONST_LOG2) as f32
        };

        let alilen = if score_mode == ScoreMode::Viterbi {
            let tr = g_trace(&dsq, args.l, &gm, &gx);
            tr.st
                .iter()
                .filter(|&&state| matches!(state, State::M | State::D | State::I))
                .count()
        } else {
            args.l
        };
        scores.push(bits as f64);
        alilens.push(alilen);
        writeln!(out, "{}", fmt_fixed4(bits as f64)).unwrap();
    }

    let histogram = ScoreHistogram::from_scores(&scores, -50.0, 50.0, 0.2);
    match score_mode {
        ScoreMode::Forward => {
            write_forward_tail_summary(&args, &mut out, &hmm.name, model_params, &histogram)
                .unwrap();
        }
        ScoreMode::Viterbi | ScoreMode::Hybrid | ScoreMode::Msv => {
            write_gumbel_summary(&mut out, &hmm.name, model_params, &histogram).unwrap();
        }
    }

    if let Err(e) = write_optional_outputs(
        &args,
        &hmm.name,
        model_params,
        score_mode,
        &scores,
        &alilens,
    ) {
        eprintln!("Error writing hmmsim output: {}", e);
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn arg_was_used(args: &[String], flag: &str) -> bool {
    args.iter()
        .any(|arg| arg == flag || arg.starts_with(&format!("{flag}=")))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScoreMode {
    Viterbi,
    Forward,
    Hybrid,
    Msv,
}

#[derive(Clone, Copy)]
struct ModelParams {
    mu: f64,
    lambda: f64,
}

fn calibrated_model_params(
    score_mode: ScoreMode,
    hmm: &Hmm,
    abc: &Alphabet,
    bg: &Bg,
    args: &Args,
    rng: &mut MersenneTwister,
) -> ModelParams {
    let lambda = p7_lambda(hmm, bg) as f64;
    match score_mode {
        ScoreMode::Msv => ModelParams {
            mu: calibrate_msv_mu(hmm, abc, bg, args.em_l, args.em_n, lambda, rng),
            lambda,
        },
        ScoreMode::Viterbi => ModelParams {
            mu: calibrate_viterbi_mu(hmm, abc, bg, args.ev_l, args.ev_n, lambda, rng),
            lambda,
        },
        ScoreMode::Forward => ModelParams {
            mu: calibrate_forward_tau(hmm, abc, bg, args.ef_l, args.ef_n, args.eft, lambda, rng),
            lambda,
        },
        ScoreMode::Hybrid => ModelParams { mu: 0.0, lambda },
    }
}

fn calibrate_msv_mu(
    hmm: &Hmm,
    abc: &Alphabet,
    bg: &Bg,
    l: usize,
    n: usize,
    lambda: f64,
    rng: &mut MersenneTwister,
) -> f64 {
    let scores = simulate_calibration_scores(hmm, abc, bg, l, n, rng, |dsq, gm, gx| {
        g_msv(dsq, l, gm, gx, 2.0)
    });
    gumbel::fit_complete_loc(&scores, lambda).unwrap_or_else(|_| fallback_gumbel(&scores).0)
}

fn calibrate_viterbi_mu(
    hmm: &Hmm,
    abc: &Alphabet,
    bg: &Bg,
    l: usize,
    n: usize,
    lambda: f64,
    rng: &mut MersenneTwister,
) -> f64 {
    let scores = simulate_calibration_scores(hmm, abc, bg, l, n, rng, |dsq, gm, gx| {
        g_viterbi(dsq, l, gm, gx)
    });
    gumbel::fit_complete_loc(&scores, lambda).unwrap_or_else(|_| fallback_gumbel(&scores).0)
}

fn calibrate_forward_tau(
    hmm: &Hmm,
    abc: &Alphabet,
    bg: &Bg,
    l: usize,
    n: usize,
    tailp: f64,
    lambda: f64,
    rng: &mut MersenneTwister,
) -> f64 {
    let scores = simulate_calibration_scores(hmm, abc, bg, l, n, rng, |dsq, gm, gx| {
        g_forward(dsq, l, gm, gx)
    });
    let (gmu, glam) = gumbel::fit_complete(&scores).unwrap_or_else(|_| fallback_gumbel(&scores));
    let tau = gumbel::invcdf(1.0 - tailp, gmu, glam) + c_log_f64(tailp) / lambda;
    if tau.is_finite() {
        tau
    } else {
        fallback_exponential(&scores).0
    }
}

fn simulate_calibration_scores<F>(
    hmm: &Hmm,
    abc: &Alphabet,
    bg: &Bg,
    l: usize,
    n: usize,
    rng: &mut MersenneTwister,
    mut score_fn: F,
) -> Vec<f64>
where
    F: FnMut(&[u8], &Profile, &mut Gmx) -> f32,
{
    let mut bg = bg.clone();
    bg.set_length(l);
    let mut gm = Profile::new(hmm.m, abc);
    profile::profile_config(hmm, &bg, &mut gm, l as i32, P7_LOCAL);
    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        let mut dsq = Vec::with_capacity(l + 2);
        dsq.push(hmmer_pure_rs::alphabet::DSQ_SENTINEL);
        for _ in 0..l {
            dsq.push(rng.sample_residue(&bg.f));
        }
        dsq.push(hmmer_pure_rs::alphabet::DSQ_SENTINEL);

        let mut gx = Gmx::new(gm.m, l);
        let sc = score_fn(&dsq, &gm, &mut gx);
        let null_sc = bg.null_one(l);
        let bits = (sc as f64 - null_sc as f64) / ESL_CONST_LOG2;
        if bits.is_finite() {
            scores.push(bits);
        }
    }

    if scores.is_empty() {
        scores.push(0.0);
    }
    scores
}

fn write_optional_outputs(
    args: &Args,
    hmm_name: &str,
    model_params: ModelParams,
    score_mode: ScoreMode,
    scores: &[f64],
    alilens: &[usize],
) -> std::io::Result<()> {
    let fit = ScoreFit::fit(score_mode, scores, model_params);
    let histogram = ScoreHistogram::from_scores(scores, -50.0, 50.0, 0.2);

    if let Some(path) = &args.xfile {
        let mut file = std::fs::File::create(path)?;
        for &score in scores {
            file.write_all(&score.to_ne_bytes())?;
        }
    }

    if let Some(path) = &args.afile {
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "# {}", hmm_name)?;
        writeln!(file, "# alilen bitscore")?;
        for (&alilen, &score) in alilens.iter().zip(scores.iter()) {
            writeln!(file, "{} {}", alilen, fmt_fixed4(score))?;
        }
    }

    if let Some(path) = &args.ffile {
        let mut npass = 0usize;
        for &score in scores {
            if fit.model_survival(score) <= args.pthresh {
                npass += 1;
            }
        }
        let mut file = std::fs::File::create(path)?;
        writeln!(
            file,
            "{}\t{}\t{}",
            hmm_name,
            npass,
            fmt_fixed4(npass as f64 / scores.len() as f64)
        )?;
    }

    if let Some(path) = &args.efile {
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "# {}", hmm_name)?;
        for rank in 1..=scores.len().min(1000) {
            let score = histogram.rank(rank);
            writeln!(
                file,
                "{} {}",
                rank,
                fmt_g(scores.len() as f64 * fit.model_survival(score))
            )?;
        }
        writeln!(file, "&")?;
    }

    if let Some(path) = &args.pfile {
        let mut file = std::fs::File::create(path)?;
        histogram.write_survival_plot(&mut file)?;
        match score_mode {
            ScoreMode::Forward => {
                let fits = forward_tail_fits(args, &histogram);
                let (mu, lambda) = fits
                    .last()
                    .map(|fit| (fit.mu, fit.lambda))
                    .unwrap_or_else(|| fallback_exponential(scores));
                write_exponential_plot(&mut file, mu, lambda, mu, histogram.xmax + 5.0, 0.1)?;
                write_exponential_plot(&mut file, mu, 0.693147, mu, histogram.xmax + 5.0, 0.1)?;
            }
            ScoreMode::Viterbi | ScoreMode::Hybrid | ScoreMode::Msv => {
                let (mu, lambda) =
                    gumbel::fit_complete(scores).unwrap_or_else(|_| fallback_gumbel(scores));
                let mufix = gumbel::fit_complete_loc(scores, 0.693147).unwrap_or_else(|_| mu);
                write_gumbel_plot(
                    &mut file,
                    mu,
                    lambda,
                    histogram.xmin - 5.0,
                    histogram.xmax + 5.0,
                    0.1,
                )?;
                write_gumbel_plot(
                    &mut file,
                    mufix,
                    0.693147,
                    histogram.xmin - 5.0,
                    histogram.xmax + 5.0,
                    0.1,
                )?;
            }
        }
    }

    Ok(())
}

#[derive(Clone)]
struct ScoreHistogram {
    obs: Vec<u64>,
    w: f64,
    bmin: f64,
    bmax: f64,
    imin: usize,
    imax: usize,
    xmin: f64,
    xmax: f64,
    sorted: Vec<f64>,
}

impl ScoreHistogram {
    fn from_scores(scores: &[f64], bmin: f64, bmax: f64, w: f64) -> Self {
        let nb = ((bmax - bmin) / w) as usize;
        let mut histogram = Self {
            obs: vec![0; nb],
            w,
            bmin,
            bmax,
            imin: nb,
            imax: 0,
            xmin: f64::INFINITY,
            xmax: f64::NEG_INFINITY,
            sorted: Vec::with_capacity(scores.len()),
        };
        for &score in scores {
            if score.is_finite() {
                histogram.add(score);
            }
        }
        histogram.sorted.sort_by(|a, b| a.total_cmp(b));
        histogram
    }

    fn add(&mut self, score: f64) {
        let mut bin = (((score - self.bmin) / self.w) - 1.0).ceil() as isize;
        if bin < 0 {
            let nnew = (-bin * 2) as usize;
            let mut obs = vec![0; nnew + self.obs.len()];
            obs[nnew..].copy_from_slice(&self.obs);
            self.obs = obs;
            self.bmin -= nnew as f64 * self.w;
            if !self.sorted.is_empty() {
                self.imin += nnew;
                self.imax += nnew;
            }
            bin += nnew as isize;
        } else if bin as usize >= self.obs.len() {
            let nnew = (bin as usize - self.obs.len() + 1) * 2;
            self.obs.resize(self.obs.len() + nnew, 0);
            self.bmax += nnew as f64 * self.w;
        }

        let bin = bin as usize;
        self.obs[bin] += 1;
        if self.sorted.is_empty() {
            self.imin = bin;
            self.imax = bin;
        } else {
            self.imin = self.imin.min(bin);
            self.imax = self.imax.max(bin);
        }
        self.xmin = self.xmin.min(score);
        self.xmax = self.xmax.max(score);
        self.sorted.push(score);
    }

    fn rank(&self, rank: usize) -> f64 {
        self.sorted[self.sorted.len() - rank]
    }

    fn tail_by_mass(&self, mass: f64) -> &[f64] {
        let n = ((self.sorted.len() as f64) * mass.min(1.0)).ceil() as usize;
        let n = n.max(1).min(self.sorted.len());
        &self.sorted[self.sorted.len() - n..]
    }

    fn bin_lbound(&self, bin: usize) -> f64 {
        self.w * bin as f64 + self.bmin
    }

    fn write_survival_plot(&self, out: &mut dyn Write) -> std::io::Result<()> {
        let mut cumulative = 0_u64;
        let n = self.sorted.len() as f64;
        if self.obs[self.imax] > 1 {
            writeln!(out, "{}\t{}", fmt_fixed6(self.xmax), fmt_g(1.0 / n))?;
        }
        for bin in (self.imin..=self.imax).rev() {
            if self.obs[bin] > 0 {
                cumulative += self.obs[bin];
                writeln!(
                    out,
                    "{}\t{}",
                    fmt_fixed6(self.bin_lbound(bin)),
                    fmt_g(cumulative as f64 / n)
                )?;
            }
        }
        writeln!(out, "&")
    }
}

#[derive(Clone, Copy)]
struct ForwardTailFit {
    tailp: f64,
    mu: f64,
    lambda: f64,
}

fn forward_tail_fits(args: &Args, histogram: &ScoreHistogram) -> Vec<ForwardTailFit> {
    let mut fits = Vec::new();
    let mut tailp = args.tmin;
    loop {
        if tailp > 1.0 {
            tailp = 1.0;
        }
        let tail = histogram.tail_by_mass(tailp);
        let fit = if tail.is_empty() {
            fallback_exponential(&histogram.sorted)
        } else {
            let (mu, lambda) = exponential::fit_complete(tail);
            if lambda.is_finite() && lambda > 0.0 {
                (mu, lambda)
            } else {
                fallback_exponential(&histogram.sorted)
            }
        };
        fits.push(ForwardTailFit {
            tailp,
            mu: fit.0,
            lambda: fit.1,
        });

        if args.tpoints == 1 {
            break;
        }
        if args.tlinear {
            tailp += (args.tmax - args.tmin) / (args.tpoints - 1) as f64;
        } else {
            tailp *= (c_log_f64(args.tmax / args.tmin) / (args.tpoints - 1) as f64).exp();
        }
        if tailp > args.tmax + 1e-7 {
            break;
        }
    }
    fits
}

fn write_forward_tail_summary(
    args: &Args,
    out: &mut dyn Write,
    hmm_name: &str,
    model_params: ModelParams,
    histogram: &ScoreHistogram,
) -> std::io::Result<()> {
    let rank = histogram.sorted.len().min(10).max(1);
    let x10 = histogram.rank(rank);
    let fits = forward_tail_fits(args, histogram);
    for fit in fits {
        let e10 =
            histogram.sorted.len() as f64 * fit.tailp * exponential::surv(x10, fit.mu, fit.lambda);
        let e10fix =
            histogram.sorted.len() as f64 * fit.tailp * exponential::surv(x10, fit.mu, 0.693147);
        let e10p = histogram.sorted.len() as f64
            * exponential::surv(x10, model_params.mu, model_params.lambda);
        writeln!(
            out,
            "{:<20}  {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4}",
            hmm_name,
            fit.tailp,
            fit.mu,
            fit.lambda,
            e10,
            fit.mu,
            e10fix,
            model_params.mu,
            model_params.lambda,
            e10p
        )?;
    }
    Ok(())
}

fn write_gumbel_summary(
    out: &mut dyn Write,
    hmm_name: &str,
    model_params: ModelParams,
    histogram: &ScoreHistogram,
) -> std::io::Result<()> {
    let rank = histogram.sorted.len().min(10).max(1);
    let x10 = histogram.rank(rank);

    let (mu, lambda) = gumbel::fit_complete(&histogram.sorted)
        .unwrap_or_else(|_| fallback_gumbel(&histogram.sorted));
    let e10 = histogram.sorted.len() as f64 * gumbel::surv(x10, mu, lambda);

    let mufix = gumbel::fit_complete_loc(&histogram.sorted, 0.693147).unwrap_or_else(|_| mu);
    let e10fix = histogram.sorted.len() as f64 * gumbel::surv(x10, mufix, 0.693147);

    let mufix2 = gumbel::fit_complete_loc(&histogram.sorted, model_params.lambda)
        .unwrap_or_else(|_| model_params.mu);
    let e10fix2 = histogram.sorted.len() as f64 * gumbel::surv(x10, mufix2, model_params.lambda);

    let e10p =
        histogram.sorted.len() as f64 * gumbel::surv(x10, model_params.mu, model_params.lambda);

    writeln!(
        out,
        "{:<20}  {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4}",
        hmm_name,
        1.0,
        mu,
        lambda,
        e10,
        mufix,
        e10fix,
        mufix2,
        e10fix2,
        model_params.mu,
        model_params.lambda,
        e10p
    )
}

fn write_gumbel_plot(
    out: &mut dyn Write,
    mu: f64,
    lambda: f64,
    xmin: f64,
    xmax: f64,
    xstep: f64,
) -> std::io::Result<()> {
    let mut x = xmin;
    while x <= xmax {
        writeln!(
            out,
            "{}\t{}",
            fmt_fixed6(x),
            fmt_g(gumbel::surv(x, mu, lambda))
        )?;
        x += xstep;
    }
    writeln!(out, "&")
}

fn write_exponential_plot(
    out: &mut dyn Write,
    mu: f64,
    lambda: f64,
    xmin: f64,
    xmax: f64,
    xstep: f64,
) -> std::io::Result<()> {
    let mut x = xmin;
    while x <= xmax {
        writeln!(
            out,
            "{}\t{}",
            fmt_fixed6(x),
            fmt_g(exponential::surv(x, mu, lambda))
        )?;
        x += xstep;
    }
    writeln!(out, "&")
}

#[derive(Clone, Copy)]
struct ScoreFit {
    kind: FitKind,
    model_mu: f64,
    model_lambda: f64,
}

#[derive(Clone, Copy)]
enum FitKind {
    Gumbel,
    Exponential,
}

impl ScoreFit {
    fn fit(score_mode: ScoreMode, scores: &[f64], model_params: ModelParams) -> Self {
        let kind = match score_mode {
            ScoreMode::Forward | ScoreMode::Hybrid => FitKind::Exponential,
            ScoreMode::Viterbi | ScoreMode::Msv => FitKind::Gumbel,
        };
        let (model_mu, model_lambda) = (model_params.mu, model_params.lambda);
        let (fit_mu, fit_lambda) = match kind {
            FitKind::Gumbel => {
                gumbel::fit_complete(scores).unwrap_or_else(|_| fallback_gumbel(scores))
            }
            FitKind::Exponential => fit_exponential_tail(scores, 0.02),
        };
        let (model_mu, model_lambda) =
            sanitize_model_params(model_mu, model_lambda, fit_mu, fit_lambda);

        Self {
            kind,
            model_mu,
            model_lambda,
        }
    }

    fn model_survival(&self, score: f64) -> f64 {
        match self.kind {
            FitKind::Gumbel => gumbel::surv(score, self.model_mu, self.model_lambda),
            FitKind::Exponential => exponential::surv(score, self.model_mu, self.model_lambda),
        }
    }
}

fn fit_exponential_tail(scores: &[f64], tail_mass: f64) -> (f64, f64) {
    let mut ranked = ranked_scores(scores);
    let tail_n = ((ranked.len() as f64 * tail_mass).ceil() as usize)
        .max(2)
        .min(ranked.len());
    ranked.truncate(tail_n);
    let (mu, lambda) = exponential::fit_complete(&ranked);
    if lambda.is_finite() && lambda > 0.0 {
        (mu, lambda)
    } else {
        fallback_exponential(scores)
    }
}

fn fallback_gumbel(scores: &[f64]) -> (f64, f64) {
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let var = sample_variance(scores, mean).max(1e-6);
    let lambda = std::f64::consts::PI / c_sqrt_f64(6.0 * var);
    let euler_gamma = 0.5772156649015329_f64;
    (mean - euler_gamma / lambda, lambda)
}

fn fallback_exponential(scores: &[f64]) -> (f64, f64) {
    let mu = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let lambda = 1.0 / (mean - mu).max(1e-6);
    (mu, lambda)
}

fn sample_variance(scores: &[f64], mean: f64) -> f64 {
    if scores.len() < 2 {
        return 0.0;
    }
    scores
        .iter()
        .map(|score| {
            let delta = score - mean;
            delta * delta
        })
        .sum::<f64>()
        / (scores.len() - 1) as f64
}

fn sanitize_model_params(mu: f64, lambda: f64, fit_mu: f64, fit_lambda: f64) -> (f64, f64) {
    if mu.is_finite() && lambda.is_finite() && lambda > 0.0 {
        (mu, lambda)
    } else {
        (fit_mu, fit_lambda)
    }
}

fn ranked_scores(scores: &[f64]) -> Vec<f64> {
    let mut ranked = scores.to_vec();
    ranked.sort_by(|a, b| b.total_cmp(a));
    ranked
}

fn elide_length_model(gm: &mut Profile, bg: &mut Bg) {
    bg.p1 = 350.0 / 351.0;
    let loop_sc = c_logf_to_f32(bg.p1);
    let move_sc = c_logf_to_f32(1.0 - bg.p1);
    gm.xsc[P7P_N][P7P_LOOP] = loop_sc;
    gm.xsc[P7P_C][P7P_LOOP] = loop_sc;
    gm.xsc[P7P_J][P7P_LOOP] = loop_sc;
    gm.xsc[P7P_N][P7P_MOVE] = move_sc;
    gm.xsc[P7P_C][P7P_MOVE] = move_sc;
    gm.xsc[P7P_J][P7P_MOVE] = move_sc;
}
