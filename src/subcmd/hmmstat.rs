//! hmmstat — display summary statistics for each HMM in a file.

use std::io::{BufReader, Write};
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmm::{DM, IM, MI, MM};
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::profile;
use hmmer_pure_rs::util::cmath::{c_log_f64, ESL_CONST_LOG2, ESL_CONST_LOG2R};

#[derive(Parser)]
#[command(name = "hmmstat", about = "Display summary statistics for each HMM")]
struct Args {
    /// HMM file
    hmmfile: PathBuf,
}

/// Entry point for `hmmstat`: print one summary row per HMM in the input file.
///
/// Columns: idx, name, accession, nseq, eff_nseq, M (model length), mean match
/// relative entropy, mean information content, mean position-wise relative
/// entropy, and composition-vs-background KL divergence. Mirrors the default
/// (`-h` aside) tabular path of `main()` in hmmer/src/hmmstat.c.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    let hmms = read_hmms_maybe_stdin(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(
        out,
        "# hmmstat :: display summary statistics for a profile file"
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
    writeln!(out, "#").unwrap();

    writeln!(
        out,
        "# idx  name                 accession        nseq eff_nseq      M relent   info p relE compKL"
    )
    .unwrap();
    writeln!(
        out,
        "# ---- -------------------- ------------ -------- -------- ------ ------ ------ ------ ------"
    )
    .unwrap();

    for (idx, h) in hmms.iter().enumerate() {
        let abc = Alphabet::new(h.abc_type);
        let bg = Bg::new(&abc);

        // Compute relative entropy and information content
        let relent = mean_match_relative_entropy(h, &bg);
        let info = mean_match_info(h, &bg);
        let p_rele = mean_position_relative_entropy(h, &bg);
        let comp_kl = composition_kld(h, &bg);

        write_stat_row(
            &mut out,
            idx + 1,
            &h.name,
            h.acc.as_deref().unwrap_or("-"),
            h.nseq,
            h.eff_nseq,
            h.m,
            relent,
            info,
            p_rele,
            comp_kl,
        );
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

#[allow(clippy::too_many_arguments)]
fn write_stat_row<W: Write>(
    out: &mut W,
    idx: usize,
    name: &str,
    acc: &str,
    nseq: i32,
    eff_nseq: f32,
    m: usize,
    relent: f32,
    info: f32,
    p_rele: f32,
    comp_kl: f32,
) {
    writeln!(
        out,
        "{:<6} {:<20} {:<12} {:>8} {} {:>6} {} {} {} {}",
        idx,
        name,
        acc,
        nseq,
        hmmer_pure_rs::output::fmt_width8_2(eff_nseq as f64),
        m,
        hmmer_pure_rs::output::fmt_width6_2(relent as f64),
        hmmer_pure_rs::output::fmt_width6_2(info as f64),
        hmmer_pure_rs::output::fmt_width6_2(p_rele as f64),
        hmmer_pure_rs::output::fmt_width6_2(comp_kl as f64)
    )
    .unwrap();
}

/// Mean relative entropy per match emission.
fn mean_match_relative_entropy(h: &hmmer_pure_rs::Hmm, bg: &Bg) -> f32 {
    let k = h.abc_k;
    let mut sum = 0.0_f32;
    for node in 1..=h.m {
        for x in 0..k {
            let p = h.mat[node][x];
            if p > 0.0 && bg.f[x] > 0.0 {
                sum += p * c_log_f64((p / bg.f[x]) as f64) as f32 * (ESL_CONST_LOG2R as f32);
            }
        }
    }
    sum / h.m as f32
}

/// Mean information content per match emission.
fn mean_match_info(h: &hmmer_pure_rs::Hmm, bg: &Bg) -> f32 {
    let k = h.abc_k;
    let mut sum = 0.0_f32;
    for node in 1..=h.m {
        let mut node_entropy = 0.0_f32;
        for x in 0..k {
            let p = h.mat[node][x];
            if p > 0.0 {
                node_entropy -= p * c_log_f64(p as f64) as f32 * (ESL_CONST_LOG2R as f32);
            }
        }
        let mut bg_entropy = 0.0_f32;
        for x in 0..k {
            if bg.f[x] > 0.0 {
                bg_entropy -= bg.f[x] * c_log_f64(bg.f[x] as f64) as f32 * (ESL_CONST_LOG2R as f32);
            }
        }
        sum += bg_entropy - node_entropy;
    }
    sum / h.m as f32
}

/// Mean position-wise relative entropy.
fn mean_position_relative_entropy(h: &hmmer_pure_rs::Hmm, bg: &Bg) -> f32 {
    let mocc = profile::hmm_calculate_occupancy(h);
    let occ_sum: f32 = mocc[1..=h.m].iter().sum();
    if occ_sum <= 0.0 {
        return 0.0;
    }

    let mut mre = 0.0_f64;
    for (node, &occ) in mocc.iter().enumerate().take(h.m + 1).skip(1) {
        mre += (occ as f64) * f_rel_entropy_c(&h.mat[node][..h.abc_k], &bg.f[..h.abc_k]) as f64;
    }
    mre /= fsum_c(&mocc[1..=h.m]) as f64;

    if h.m < 2 {
        return mre as f32;
    }

    let trans_occ_sum = fsum_c(&mocc[2..=h.m]);
    if trans_occ_sum <= 0.0 {
        return mre as f32;
    }

    let mut tre = 0.0_f64;
    for node in 2..=h.m {
        let prev_occ = mocc[node - 1] as f64;
        let p1 = bg.p1 as f64;
        let mm = h.t[node - 1][MM] as f64;
        let mi = h.t[node - 1][MI] as f64;
        let im = h.t[node - 1][IM] as f64;
        let dm = h.t[node - 1][DM] as f64;

        let xm = prev_occ * log_ratio_term(mm, p1);
        let xi = prev_occ * mi * (safe_ln_ratio(mm, p1) + safe_ln_ratio(im, p1));
        let xd = (1.0 - prev_occ) * log_ratio_term(dm, p1);
        tre += (xm + xi + xd) / ESL_CONST_LOG2;
    }
    tre /= trans_occ_sum as f64;

    (mre + tre) as f32
}

/// KL divergence between model composition and background.
fn composition_kld(h: &hmmer_pure_rs::Hmm, bg: &Bg) -> f32 {
    let mocc = profile::hmm_calculate_occupancy(h);
    let mut avg = vec![0.0_f32; h.abc_k];
    for (node, &occ) in mocc.iter().enumerate().take(h.m + 1).skip(1) {
        for (x, value) in avg.iter_mut().enumerate().take(h.abc_k) {
            *value += h.mat[node][x] * occ;
        }
    }

    let sum: f32 = avg.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    for p in &mut avg {
        *p /= sum;
    }

    rel_entropy(&avg, &bg.f[..h.abc_k]) as f32
}

fn fsum_c(values: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let mut c = 0.0_f32;
    for &value in values {
        let y = value - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

fn f_rel_entropy_c(p: &[f32], q: &[f32]) -> f32 {
    let mut kl = 0.0_f32;
    for (&px, &qx) in p.iter().zip(q.iter()) {
        if px > 0.0 {
            if qx == 0.0 {
                return f32::INFINITY;
            }
            kl = (kl as f64 + (px as f64) * (c_log_f64((px / qx) as f64) / ESL_CONST_LOG2)) as f32;
        }
    }
    kl
}

fn rel_entropy(p: &[f32], q: &[f32]) -> f64 {
    p.iter()
        .zip(q.iter())
        .filter_map(|(&px, &qx)| {
            if px > 0.0 && qx > 0.0 {
                Some((px as f64) * c_log_f64((px as f64) / (qx as f64)) * ESL_CONST_LOG2R)
            } else {
                None
            }
        })
        .sum()
}

fn log_ratio_term(p: f64, q: f64) -> f64 {
    if p > 0.0 && q > 0.0 {
        p * c_log_f64(p / q)
    } else {
        0.0
    }
}

fn safe_ln_ratio(p: f64, q: f64) -> f64 {
    if p > 0.0 && q > 0.0 {
        c_log_f64(p / q)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmmer_pure_rs::alphabet::AlphabetType;
    use hmmer_pure_rs::hmm::{DD, II, MD};
    use hmmer_pure_rs::Hmm;

    fn two_node_hmm() -> Hmm {
        let mut h = Hmm::new(2, AlphabetType::Amino, 20);
        h.name = "toy".to_string();
        h.nseq = 2;
        h.eff_nseq = 2.0;
        h.t[0][MM] = 0.8;
        h.t[0][MI] = 0.0;
        h.t[0][MD] = 0.2;
        h.t[1][MM] = 0.7;
        h.t[1][MI] = 0.1;
        h.t[1][MD] = 0.2;
        h.t[1][IM] = 0.6;
        h.t[1][II] = 0.4;
        h.t[1][DM] = 0.5;
        h.t[1][DD] = 0.5;

        for node in 1..=2 {
            for x in 0..20 {
                h.mat[node][x] = 0.01;
                h.ins[node][x] = 0.05;
            }
        }
        h.mat[1][0] = 0.81;
        h.mat[2][1] = 0.81;
        h
    }

    #[test]
    fn position_relative_entropy_is_not_plain_match_average_when_transitions_matter() {
        let abc = Alphabet::new(AlphabetType::Amino);
        let bg = Bg::new(&abc);
        let h = two_node_hmm();

        let match_re = mean_match_relative_entropy(&h, &bg);
        let pos_re = mean_position_relative_entropy(&h, &bg);

        assert!((pos_re - match_re).abs() > 0.001);
    }

    #[test]
    fn composition_kld_recomputes_occupancy_weighted_match_composition() {
        let abc = Alphabet::new(AlphabetType::Amino);
        let bg = Bg::new(&abc);
        let h = two_node_hmm();

        let kld = composition_kld(&h, &bg);

        assert!(kld > 0.0);
    }

    #[test]
    fn stat_row_uses_c_hmmstat_spacing_and_precision() {
        let mut out = Vec::new();
        write_stat_row(
            &mut out,
            1,
            "fn3",
            "PF00041.13",
            106,
            11.42,
            86,
            0.6613,
            0.6341,
            0.5718,
            0.0392,
        );

        assert_eq!(
            String::from_utf8(out).unwrap(),
            "1      fn3                  PF00041.13        106    11.42     86   0.66   0.63   0.57   0.04\n"
        );
    }

    #[test]
    fn stat_row_idx_uses_c_minus_6d_field_width_for_large_indices() {
        // C body format is "%-6d %-20s ...": the idx is left-justified in a
        // fixed width-6 field followed by a single space. For idx >= 10000 the
        // 5-digit value still occupies the 6-wide field, so it is "10000 "
        // (5 digits + 1 pad + the separating space => "10000  "), not the
        // old Rust "{:<4}   " which produced "10000   " (5 + 3 spaces).
        let mut out = Vec::new();
        write_stat_row(&mut out, 10000, "fn3", "-", 1, 1.0, 1, 0.0, 0.0, 0.0, 0.0);
        let s = String::from_utf8(out).unwrap();
        // idx field + name field start: "10000 " (6-wide left-just) + " " sep.
        assert!(
            s.starts_with("10000  fn3"),
            "idx field must match C %-6d (width 6, left-justified): got {s:?}"
        );
        // And a 7-digit index simply overflows the field, same as C %-6d.
        let mut out2 = Vec::new();
        write_stat_row(
            &mut out2, 1234567, "fn3", "-", 1, 1.0, 1, 0.0, 0.0, 0.0, 0.0,
        );
        let s2 = String::from_utf8(out2).unwrap();
        assert!(
            s2.starts_with("1234567 fn3"),
            "over-wide idx must overflow like C %-6d: got {s2:?}"
        );
    }
}
