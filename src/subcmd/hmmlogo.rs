//! hmmlogo — generate data for HMM sequence logo visualization.
//! Outputs C-style residue-height and indel-value tables.

#![allow(clippy::needless_range_loop)]

use std::io::{BufReader, Write};
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmm::{II, MI};
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::output::{fmt_fixed2, fmt_width6_3};
use hmmer_pure_rs::profile;
use hmmer_pure_rs::util::cmath::{c_log_f64, ESL_CONST_LOG2R};

#[derive(Parser)]
#[command(name = "hmmlogo", about = "Generate HMM logo data for visualization")]
struct Args {
    /// Total height = relative entropy; all letters shown
    #[arg(long = "height_relent_all")]
    height_relent_all: bool,

    /// Total height = relative entropy; only letters above background shown
    #[arg(long = "height_relent_abovebg")]
    height_relent_abovebg: bool,

    /// Total height = sums of scores; residue height = score
    #[arg(long = "height_score")]
    height_score: bool,

    /// Do not provide indel rate values
    #[arg(long = "no_indel")]
    no_indel: bool,

    /// HMM file
    hmmfile: PathBuf,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LogoMode {
    RelentAll,
    RelentAboveBg,
    Score,
}

/// Entry point for `hmmlogo`: dump C HMMER logo residue heights and,
/// by default, per-position insert probability / expected insert length /
/// match occupancy values. Mirrors the default branch of `main()` in
/// hmmer/src/hmmlogo.c.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    let mode_count = [
        args.height_relent_all,
        args.height_relent_abovebg,
        args.height_score,
    ]
    .into_iter()
    .filter(|v| *v)
    .count();
    if mode_count > 1 {
        eprintln!(
            "Error: options --height_relent_all, --height_relent_abovebg, and --height_score are mutually exclusive"
        );
        std::process::exit(1);
    }
    let mode = if args.height_score {
        LogoMode::Score
    } else if args.height_relent_abovebg {
        LogoMode::RelentAboveBg
    } else {
        LogoMode::RelentAll
    };

    let hmms = read_hmms_maybe_stdin(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let Some(hmm) = hmms.first() else {
        eprintln!("Error: no HMMs found in {}", args.hmmfile.display());
        std::process::exit(1);
    };
    let abc = Alphabet::new(hmm.abc_type);
    let bg = Bg::new(&abc);
    let k = abc.k;

    if mode != LogoMode::Score {
        writeln!(
            out,
            "max expected height = {}",
            fmt_fixed2(max_height(&bg) as f64)
        )
        .unwrap();
    }
    writeln!(out, "Residue heights").unwrap();
    for node in 1..=hmm.m {
        let (relent, heights) = residue_heights(hmm, &bg, node, mode);
        write!(out, "{}: ", node).unwrap();
        for height in heights.iter().take(k) {
            write!(out, "{} ", fmt_width6_3(*height as f64)).unwrap();
        }
        if mode != LogoMode::Score {
            write!(out, " ({})", fmt_width6_3(relent as f64)).unwrap();
        }
        writeln!(out).unwrap();
    }

    if !args.no_indel {
        let occupancy = profile::hmm_calculate_occupancy(hmm);
        writeln!(out, "Indel values").unwrap();
        for node in 1..=hmm.m {
            let (insert_p, insert_exp_l) = if node == hmm.m {
                (0.0, 0.0)
            } else {
                let insert_p = hmm.t[node][MI];
                let insert_exp_l = 1.0 / (1.0 - hmm.t[node][II]);
                (insert_p, insert_exp_l)
            };
            writeln!(
                out,
                "{}: {} {} {}",
                node,
                fmt_width6_3(insert_p as f64),
                fmt_width6_3(insert_exp_l as f64),
                fmt_width6_3(occupancy[node] as f64)
            )
            .unwrap();
        }
    }
    std::process::ExitCode::SUCCESS
}

fn read_hmms_maybe_stdin(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<hmmer_pure_rs::Hmm>> {
    if path == std::path::Path::new("-") {
        let stdin = std::io::stdin();
        hmmfile::read_first_hmm(BufReader::new(stdin.lock())).map(|hmm| vec![hmm])
    } else {
        hmmfile::read_first_hmm_file_auto(path).map(|hmm| vec![hmm])
    }
}

fn max_height(bg: &Bg) -> f32 {
    let min_p =
        bg.f.iter()
            .copied()
            .filter(|p| *p > 0.0)
            .fold(1.0_f32, f32::min);
    (c_log_f64(1.0 / min_p as f64) * ESL_CONST_LOG2R) as f32
}

fn residue_heights(
    hmm: &hmmer_pure_rs::Hmm,
    bg: &Bg,
    node: usize,
    mode: LogoMode,
) -> (f32, Vec<f32>) {
    let k = hmm.abc_k;
    let mut relent = 0.0_f32;
    let mut above_bg_prob_sum = 0.0_f32;

    for x in 0..k {
        let p = hmm.mat[node][x];
        if p > 0.0 && bg.f[x] > 0.0 {
            let logodds = (c_log_f64((p / bg.f[x]) as f64) * ESL_CONST_LOG2R) as f32;
            relent += p * logodds;
            if logodds > 0.0 {
                above_bg_prob_sum += p;
            }
        }
    }

    let mut heights = vec![0.0_f32; k];
    for (x, height) in heights.iter_mut().enumerate().take(k) {
        let p = hmm.mat[node][x];
        // C's hmmlogo_ScoreHeights (hmmlogo.c:108-116) computes log(p/bg)
        // UNCONDITIONALLY — no guard on p — so a zero match emission yields
        // log(0) = -inf, which prints as "  -inf" under %6.3f. The two relent
        // modes (hmmlogo.c:48-50, 86-92) instead leave height at 0.0 when
        // p == 0, so they keep the guard.
        if mode == LogoMode::Score {
            *height = (c_log_f64((p / bg.f[x]) as f64) * ESL_CONST_LOG2R) as f32;
            continue;
        }
        if p <= 0.0 || bg.f[x] <= 0.0 {
            continue;
        }
        let logodds = (c_log_f64((p / bg.f[x]) as f64) * ESL_CONST_LOG2R) as f32;
        *height = match mode {
            LogoMode::RelentAll => relent * p,
            LogoMode::RelentAboveBg => {
                if logodds > 0.0 && above_bg_prob_sum > 0.0 {
                    relent * p / above_bg_prob_sum
                } else {
                    0.0
                }
            }
            LogoMode::Score => unreachable!("score mode handled above"),
        };
    }

    (relent, heights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmmlogo_parses_c_output_modes() {
        let args = Args::try_parse_from(["hmmlogo", "--height_score", "--no_indel", "models.hmm"])
            .unwrap();
        assert!(args.height_score);
        assert!(args.no_indel);
        assert_eq!(args.hmmfile, PathBuf::from("models.hmm"));

        let err = Args::try_parse_from([
            "hmmlogo",
            "--height_score",
            "--height_relent_abovebg",
            "models.hmm",
        ])
        .unwrap();
        let count = [
            err.height_relent_all,
            err.height_relent_abovebg,
            err.height_score,
        ]
        .into_iter()
        .filter(|v| *v)
        .count();
        assert_eq!(count, 2);
    }

    /// hmmlogo F1: in Score mode a zero match emission must yield -inf
    /// (C's hmmlogo_ScoreHeights computes log(p/bg) unguarded). The two relent
    /// modes keep the p<=0 guard and leave the height at 0.0, matching C.
    #[test]
    fn hmmlogo_score_mode_zero_emission_is_neg_inf() {
        use hmmer_pure_rs::alphabet::AlphabetType;

        let abc = Alphabet::new(AlphabetType::Amino);
        let bg = Bg::new(&abc);
        let k = abc.k;

        // One node, residue 0 has zero emission, residue 1 has a nonzero prob.
        let mut hmm = hmmer_pure_rs::Hmm::new(1, AlphabetType::Amino, k);
        hmm.mat[1][0] = 0.0;
        hmm.mat[1][1] = 1.0;

        let (_relent, heights) = residue_heights(&hmm, &bg, 1, LogoMode::Score);
        assert!(
            heights[0].is_infinite() && heights[0] < 0.0,
            "score-mode zero emission should be -inf, got {}",
            heights[0]
        );
        assert!(heights[1].is_finite());

        // Relent modes keep the guard: zero emission stays 0.0 (finite).
        let (_r, h_all) = residue_heights(&hmm, &bg, 1, LogoMode::RelentAll);
        assert_eq!(h_all[0], 0.0);
        let (_r, h_bg) = residue_heights(&hmm, &bg, 1, LogoMode::RelentAboveBg);
        assert_eq!(h_bg[0], 0.0);
    }
}
