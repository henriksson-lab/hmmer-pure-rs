//! hmmlogo — generate data for HMM sequence logo visualization.
//! Outputs per-position information content and emission probabilities.

use std::io::{BufReader, Write};
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmm::{II, MI};
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::profile;

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
        writeln!(out, "max expected height = {:.2}", max_height(&bg)).unwrap();
    }
    writeln!(out, "Residue heights").unwrap();
    for node in 1..=hmm.m {
        let (relent, heights) = residue_heights(hmm, &bg, node, mode);
        write!(out, "{}: ", node).unwrap();
        for height in heights.iter().take(k) {
            write!(out, "{:6.3} ", height).unwrap();
        }
        if mode != LogoMode::Score {
            write!(out, " ({:6.3})", relent).unwrap();
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
                "{}: {:6.3} {:6.3} {:6.3}",
                node, insert_p, insert_exp_l, occupancy[node]
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
        hmmfile::read_hmms(BufReader::new(stdin.lock()))
    } else {
        hmmfile::read_hmm_file(path)
    }
}

fn max_height(bg: &Bg) -> f32 {
    let min_p =
        bg.f.iter()
            .copied()
            .filter(|p| *p > 0.0)
            .fold(1.0_f32, f32::min);
    (1.0 / min_p).log2()
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
            let logodds = (p / bg.f[x]).log2();
            relent += p * logodds;
            if logodds > 0.0 {
                above_bg_prob_sum += p;
            }
        }
    }

    let mut heights = vec![0.0_f32; k];
    for (x, height) in heights.iter_mut().enumerate().take(k) {
        let p = hmm.mat[node][x];
        if p <= 0.0 || bg.f[x] <= 0.0 {
            continue;
        }
        let logodds = (p / bg.f[x]).log2();
        *height = match mode {
            LogoMode::RelentAll => relent * p,
            LogoMode::RelentAboveBg => {
                if logodds > 0.0 && above_bg_prob_sum > 0.0 {
                    relent * p / above_bg_prob_sum
                } else {
                    0.0
                }
            }
            LogoMode::Score => logodds,
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
}
