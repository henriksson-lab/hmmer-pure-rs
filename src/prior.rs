//! Dirichlet mixture priors for HMM parameterization.
//! Simplified port of p7_prior.c.

use crate::util::cmath::{c_exp_f64, c_log_f64};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorStrategy {
    Default,
    None,
    Laplace,
}

/// Add Dirichlet prior pseudocounts and return the posterior mean estimate.
///
/// `counts[a]` are the observed pseudocounts and `alpha[a]` the Dirichlet
/// parameters (`K` of each); result is `(counts[a] + alpha[a]) / sum` then
/// renormalised. Thin `f32` wrapper around the `f64` core; used wherever a
/// single Dirichlet component is enough (e.g. transitions).
pub fn apply_dirichlet(counts: &[f32], alpha: &[f32]) -> Vec<f32> {
    let alpha_d: Vec<f64> = alpha.iter().map(|&v| v as f64).collect();
    apply_dirichlet_f64(counts, &alpha_d)
}

/// `f64` core of [`apply_dirichlet`]; takes counts and an `f64` alpha vector
/// and returns the normalised `f32` posterior mean.
fn apply_dirichlet_f64(counts: &[f32], alpha: &[f64]) -> Vec<f32> {
    let k = counts.len();
    let mut result = vec![0.0_f64; k];
    let count_sum: f64 = counts.iter().map(|&v| v as f64).sum();
    let alpha_sum: f64 = alpha.iter().sum();
    let total = count_sum + alpha_sum;

    if total > 0.0 {
        for i in 0..k {
            result[i] = (counts[i] as f64 + alpha[i]) / total;
        }
    }

    dnorm(&mut result);

    result.iter().map(|&v| v as f32).collect()
}

/// Sjolander 9-component mixture Dirichlet for amino acid match emissions.
/// Trained on Blocks9 database [Sjolander96].
const SJOLANDER_Q: [f64; 9] = [
    0.178091, 0.056591, 0.0960191, 0.0781233, 0.0834977, 0.0904123, 0.114468, 0.0682132, 0.234585,
];

const SJOLANDER_ALPHA: [[f64; 20]; 9] = [
    [
        0.270671, 0.039848, 0.017576, 0.016415, 0.014268, 0.131916, 0.012391, 0.022599, 0.020358,
        0.030727, 0.015315, 0.048298, 0.053803, 0.020662, 0.023612, 0.216147, 0.147226, 0.065438,
        0.003758, 0.009621,
    ],
    [
        0.021465, 0.010300, 0.011741, 0.010883, 0.385651, 0.016416, 0.076196, 0.035329, 0.013921,
        0.093517, 0.022034, 0.028593, 0.013086, 0.023011, 0.018866, 0.029156, 0.018153, 0.036100,
        0.071770, 0.419641,
    ],
    [
        0.561459, 0.045448, 0.438366, 0.764167, 0.087364, 0.259114, 0.214940, 0.145928, 0.762204,
        0.247320, 0.118662, 0.441564, 0.174822, 0.530840, 0.465529, 0.583402, 0.445586, 0.227050,
        0.029510, 0.121090,
    ],
    [
        0.070143, 0.011140, 0.019479, 0.094657, 0.013162, 0.048038, 0.077000, 0.032939, 0.576639,
        0.072293, 0.028240, 0.080372, 0.037661, 0.185037, 0.506783, 0.073732, 0.071587, 0.042532,
        0.011254, 0.028723,
    ],
    [
        0.041103, 0.014794, 0.005610, 0.010216, 0.153602, 0.007797, 0.007175, 0.299635, 0.010849,
        0.999446, 0.210189, 0.006127, 0.013021, 0.019798, 0.014509, 0.012049, 0.035799, 0.180085,
        0.012744, 0.026466,
    ],
    [
        0.115607, 0.037381, 0.012414, 0.018179, 0.051778, 0.017255, 0.004911, 0.796882, 0.017074,
        0.285858, 0.075811, 0.014548, 0.015092, 0.011382, 0.012696, 0.027535, 0.088333, 0.944340,
        0.004373, 0.016741,
    ],
    [
        0.093461, 0.004737, 0.387252, 0.347841, 0.010822, 0.105877, 0.049776, 0.014963, 0.094276,
        0.027761, 0.010040, 0.187869, 0.050018, 0.110039, 0.038668, 0.119471, 0.065802, 0.025430,
        0.003215, 0.018742,
    ],
    [
        0.452171, 0.114613, 0.062460, 0.115702, 0.284246, 0.140204, 0.100358, 0.550230, 0.143995,
        0.700649, 0.276580, 0.118569, 0.097470, 0.126673, 0.143634, 0.278983, 0.358482, 0.661750,
        0.061533, 0.199373,
    ],
    [
        0.005193, 0.004039, 0.006722, 0.006121, 0.003468, 0.016931, 0.003647, 0.002184, 0.005019,
        0.005990, 0.001473, 0.004158, 0.009055, 0.003630, 0.006583, 0.003172, 0.003690, 0.002967,
        0.002772, 0.002686,
    ],
];

/// Posterior mean of an amino-acid count vector under the Sjolander
/// 9-component mixture Dirichlet (Blocks9 amino acid match prior).
///
/// Computes per-component log posteriors (with `q_k` mixture weights and
/// `LogGamma` Bayes factors), normalises them, and forms the mixture-weighted
/// mean of each component's posterior mean. Implements the amino-acid match
/// branch of C's `p7_ParameterEstimation` / `esl_mixdchlet_MPParameters`.
pub fn apply_mixture_dirichlet(counts: &[f32]) -> Vec<f32> {
    let k = counts.len().min(20);
    let counts_d: Vec<f64> = counts[..k].iter().map(|&c| c as f64).collect();
    let count_sum: f64 = counts_d.iter().sum();

    // Compute posterior probability of each mixture component
    let mut log_pk = [0.0_f64; 9];
    for comp in 0..9 {
        let alpha_sum: f64 = SJOLANDER_ALPHA[comp][..k].iter().sum();
        // log P(counts | component) ∝ log(q_k) + log Γ(alpha_sum) - log Γ(count_sum + alpha_sum)
        //   + sum_a [ log Γ(c_a + alpha_a) - log Γ(alpha_a) ]
        // Simplified: use log-gamma approximation
        let mut log_p = c_log_f64(SJOLANDER_Q[comp]);
        log_p += ln_gamma(alpha_sum) - ln_gamma(count_sum + alpha_sum);
        for a in 0..k {
            log_p += ln_gamma(counts_d[a] + SJOLANDER_ALPHA[comp][a])
                - ln_gamma(SJOLANDER_ALPHA[comp][a]);
        }
        log_pk[comp] = log_p;
    }

    let mut pk = log_pk;
    dlog_norm(&mut pk);

    // Compute mean posterior: p[a] = sum_k P(k|c) * (c[a] + alpha_k[a]) / (sum_c + sum_alpha_k)
    let mut result = vec![0.0_f64; k];
    for comp in 0..9 {
        let alpha_sum: f64 = SJOLANDER_ALPHA[comp][..k].iter().sum();
        let denom = count_sum + alpha_sum;
        if denom > 0.0 {
            for a in 0..k {
                result[a] += pk[comp] * (counts_d[a] + SJOLANDER_ALPHA[comp][a]) / denom;
            }
        }
    }

    // Normalize
    dnorm(&mut result);

    result.iter().map(|&v| v as f32).collect()
}

/// Lanczos-style approximation of `log Γ(x)` for `x > 0`,
/// matching Easel's `esl_stats_LogGamma()`.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    const COF: [f64; 11] = [
        4.694580336184385e+04,
        -1.560605207784446e+05,
        2.065049568014106e+05,
        -1.388934775095388e+05,
        5.031796415085709e+04,
        -9.601592329182778e+03,
        8.785_855_930_895_25e2,
        -3.155153906098611e+01,
        2.908143421162229e-01,
        -2.319827630494973e-04,
        1.251639670050933e-10,
    ];
    let xx = x - 1.0;
    let tx = xx + 11.0;
    let mut tmp = tx;
    let mut value = 1.0;
    for cof in COF.iter().rev() {
        value += cof / tmp;
        tmp -= 1.0;
    }
    c_log_f64(value) + 0.918938533 + (xx + 0.5) * c_log_f64(tx + 0.5) - (tx + 0.5)
}

/// Numerically stable `log(sum(exp(values)))` ("log-sum-exp"), terms with
/// `value < max - 500` are dropped to avoid underflow.
fn dlog_sum(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::INFINITY {
        return f64::INFINITY;
    }
    let mut sum = 0.0;
    for &value in values {
        if value > max - 500.0 {
            sum += c_exp_f64(value - max);
        }
    }
    c_log_f64(sum) + max
}

/// Normalise a vector of log-probabilities in place to a probability
/// distribution (exponentiate after subtracting the log-sum-exp, then re-norm).
fn dlog_norm(values: &mut [f64]) {
    let denom = dlog_sum(values);
    for value in values.iter_mut() {
        *value = c_exp_f64(*value - denom);
    }
    dnorm(values);
}

/// Normalise a non-negative vector to sum to 1, falling back to the uniform
/// distribution when the sum is zero (mirrors `esl_vec_DNorm`).
fn dnorm(values: &mut [f64]) {
    let sum: f64 = values.iter().sum();
    if sum != 0.0 {
        for value in values {
            *value /= sum;
        }
    } else if !values.is_empty() {
        let uniform = 1.0 / values.len() as f64;
        for value in values {
            *value = uniform;
        }
    }
}

/// Default transition Dirichlet prior parameters.
pub struct TransitionPrior {
    /// Match transitions: MM, MI, MD
    pub tm: [f64; 3],
    /// Insert transitions: IM, II
    pub ti: [f64; 2],
    /// Delete transitions: DM, DD
    pub td: [f64; 2],
}

impl Default for TransitionPrior {
    /// HMMER3 default transition Dirichlet (Mitchison, trained on Pfam);
    /// the same numbers as `p7_prior_CreateAmino`.
    fn default() -> Self {
        TransitionPrior::amino()
    }
}

impl TransitionPrior {
    /// Amino-acid transition Dirichlet (Mitchison, trained on Pfam);
    /// `p7_prior_CreateAmino` (`p7_prior.c:74-85`).
    pub fn amino() -> Self {
        TransitionPrior {
            tm: [0.7939, 0.0278, 0.0135], // Mitchison, trained on Pfam
            ti: [0.1551, 0.1331],
            td: [0.9002, 0.5630],
        }
    }

    /// Nucleic-acid transition Dirichlet (trained on rmark benchmark);
    /// `p7_prior_CreateNucleic` (`p7_prior.c:190-201`).
    pub fn nucleic() -> Self {
        TransitionPrior {
            tm: [2.0, 0.1, 0.1], // TMM, TMI, TMD
            ti: [0.12, 0.4],     // TIM, TII
            td: [0.5, 1.0],      // TDM, TDD
        }
    }

    /// Select the transition prior matching the alphabet size, mirroring
    /// C's `p7_builder.c:134-137` (nucleic for eslDNA/eslRNA, amino otherwise).
    pub fn for_alphabet(abc_k: usize) -> Self {
        if abc_k == 4 {
            TransitionPrior::nucleic()
        } else {
            TransitionPrior::amino()
        }
    }
}

/// HMMER3 amino-acid insert-emission Dirichlet (single component) from
/// `p7_prior_CreateAmino`.
fn amino_insert_alpha() -> [f64; 20] {
    [
        681.0, 120.0, 623.0, 651.0, 313.0, 902.0, 241.0, 371.0, 687.0, 676.0, 143.0, 548.0, 647.0,
        415.0, 551.0, 926.0, 623.0, 505.0, 102.0, 269.0,
    ]
}

/// HMMER3 nucleic-acid match-emission 4-component Dirichlet mixture
/// (`p7_prior_CreateNucleic`): returns (mixture weights, per-component alphas).
fn nucleic_match_alpha() -> (&'static [f64], &'static [[f64; 4]]) {
    static Q: [f64; 4] = [0.24, 0.26, 0.08, 0.42];
    static ALPHA: [[f64; 4]; 4] = [
        [0.16, 0.45, 0.12, 0.39],
        [0.09, 0.03, 0.09, 0.04],
        [1.29, 0.40, 6.58, 0.51],
        [1.74, 1.49, 1.57, 1.95],
    ];
    (&Q, &ALPHA)
}

/// HMMER3 nucleic-acid insert-emission Dirichlet (uniform Laplace prior).
fn nucleic_insert_alpha() -> [f64; 4] {
    [1.0, 1.0, 1.0, 1.0]
}

/// Generalised mixture-Dirichlet posterior mean for arbitrary mixture weights
/// `q` and per-component alphas (used for nucleic match emissions).
fn apply_mixture_dirichlet_with_components(
    counts: &[f32],
    q: &[f64],
    alpha: &[Vec<f64>],
) -> Vec<f32> {
    let k = counts.len();
    let counts_d: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    let count_sum: f64 = counts_d.iter().sum();

    let mut log_pk = vec![0.0_f64; q.len()];
    for comp in 0..q.len() {
        let alpha_sum: f64 = alpha[comp].iter().sum();
        let mut log_p = c_log_f64(q[comp]);
        log_p += ln_gamma(alpha_sum) - ln_gamma(count_sum + alpha_sum);
        for a in 0..k {
            log_p += ln_gamma(counts_d[a] + alpha[comp][a]) - ln_gamma(alpha[comp][a]);
        }
        log_pk[comp] = log_p;
    }

    let mut pk = log_pk;
    dlog_norm(&mut pk);

    let mut result = vec![0.0_f64; k];
    for comp in 0..q.len() {
        let alpha_sum: f64 = alpha[comp].iter().sum();
        let denom = count_sum + alpha_sum;
        if denom > 0.0 {
            for a in 0..k {
                result[a] += pk[comp] * (counts_d[a] + alpha[comp][a]) / denom;
            }
        }
    }

    dnorm(&mut result);

    result.iter().map(|&v| v as f32).collect()
}

/// Apply HMMER3 Dirichlet priors to an HMM's weighted counts, in place,
/// turning it into a parameterised probabilistic model (port of
/// `p7_ParameterEstimation`).
///
/// For every node: match transitions (MM/MI/MD) and insert transitions
/// (IM/II) are estimated via the single-component transition prior; delete
/// transitions (DM/DD) are estimated for 1..M-1 and conventionally fixed at
/// the model termini. Match emissions use the Sjolander 9-component prior
/// for amino acids, the 4-component nucleic prior, or a Laplace fallback;
/// insert emissions use the simple `amino_insert_alpha` / `nucleic_insert_alpha`.
pub fn apply_priors(hmm: &mut crate::hmm::Hmm) {
    apply_priors_with_strategy(hmm, PriorStrategy::Default);
}

pub fn apply_priors_with_strategy(hmm: &mut crate::hmm::Hmm, strategy: PriorStrategy) {
    match strategy {
        PriorStrategy::Default => apply_default_priors(hmm),
        PriorStrategy::None => apply_none_priors(hmm),
        PriorStrategy::Laplace => apply_laplace_priors(hmm),
    }
}

fn apply_default_priors(hmm: &mut crate::hmm::Hmm) {
    use crate::hmm::*;
    let k = hmm.abc_k;
    let m = hmm.m;
    let trans = TransitionPrior::for_alphabet(k);

    for node in 0..=m {
        let mt_counts = [hmm.t[node][MM], hmm.t[node][MI], hmm.t[node][MD]];
        let mt_probs = apply_dirichlet_f64(&mt_counts, &trans.tm);
        hmm.t[node][MM] = mt_probs[0];
        hmm.t[node][MI] = mt_probs[1];
        hmm.t[node][MD] = mt_probs[2];

        let it_counts = [hmm.t[node][IM], hmm.t[node][II]];
        let it_probs = apply_dirichlet_f64(&it_counts, &trans.ti);
        hmm.t[node][IM] = it_probs[0];
        hmm.t[node][II] = it_probs[1];
    }
    hmm.t[m][MD] = 0.0;
    let msum = hmm.t[m][MM] + hmm.t[m][MI] + hmm.t[m][MD];
    if msum > 0.0 {
        hmm.t[m][MM] /= msum;
        hmm.t[m][MI] /= msum;
        hmm.t[m][MD] /= msum;
    }

    for node in 1..m {
        let dt_counts = [hmm.t[node][DM], hmm.t[node][DD]];
        let dt_probs = apply_dirichlet_f64(&dt_counts, &trans.td);
        hmm.t[node][DM] = dt_probs[0];
        hmm.t[node][DD] = dt_probs[1];
    }
    hmm.t[0][DM] = 1.0;
    hmm.t[0][DD] = 0.0;
    hmm.t[m][DM] = 1.0;
    hmm.t[m][DD] = 0.0;

    for node in 1..=m {
        let match_counts: Vec<f32> = hmm.mat[node][..k].to_vec();
        let match_probs = if k == 20 {
            apply_mixture_dirichlet(&match_counts)
        } else if k == 4 {
            let (q, alpha_arr) = nucleic_match_alpha();
            let alpha: Vec<Vec<f64>> = alpha_arr.iter().map(|row| row.to_vec()).collect();
            apply_mixture_dirichlet_with_components(&match_counts, q, &alpha)
        } else {
            apply_dirichlet_f64(&match_counts, &vec![1.0_f64; k])
        };
        hmm.mat[node][..k].copy_from_slice(&match_probs);
    }
    hmm.mat[0][..k].fill(0.0);
    if k > 0 {
        hmm.mat[0][0] = 1.0;
    }

    let ins_alpha: Vec<f64> = if k == 20 {
        amino_insert_alpha().to_vec()
    } else if k == 4 {
        nucleic_insert_alpha().to_vec()
    } else {
        vec![1.0_f64; k]
    };
    for node in 0..=m {
        let ins_counts: Vec<f32> = hmm.ins[node][..k].to_vec();
        let ins_probs = apply_dirichlet_f64(&ins_counts, &ins_alpha);
        hmm.ins[node][..k].copy_from_slice(&ins_probs);
    }
}

/// NULL-prior path: C's `p7_ParameterEstimation` returns `p7_hmm_Renormalize`
/// when `pri==NULL` (`p7_prior.c:307`). Port `p7_hmm_Renormalize`
/// (`p7_hmm.c:855-883`) exactly: per-group `esl_vec_FNorm`, then the two
/// terminal-state fixups at node M.
fn apply_none_priors(hmm: &mut crate::hmm::Hmm) {
    use crate::hmm::*;
    let k = hmm.abc_k;
    let m = hmm.m;

    for node in 0..=m {
        fnorm(&mut hmm.mat[node][..k]);
        fnorm(&mut hmm.ins[node][..k]);
        // Transition groups: TMAT (MM,MI,MD), TINS (IM,II), TDEL (DM,DD).
        fnorm_t(&mut hmm.t[node], MM, 3);
        fnorm_t(&mut hmm.t[node], IM, 2);
        fnorm_t(&mut hmm.t[node], DM, 2);
    }

    // If t[M][TD*] was all zeros, FNorm made TDD nonzero (0.5/0.5). Re-enforce.
    hmm.t[m][DM] = 1.0;
    hmm.t[m][DD] = 0.0;

    // Rare: if t[M][TM*] was all zeros, the nonexistent M_M->D_M+1 became
    // nonzero. Fix that too.
    if hmm.t[m][MD] > 0.0 {
        hmm.t[m][MD] = 0.0;
        hmm.t[m][MM] = 0.5;
        hmm.t[m][MI] = 0.5;
    }
}

/// `esl_vec_FNorm`: normalise a non-negative vector to sum to 1; if the sum is
/// zero, set the uniform distribution.
fn fnorm(values: &mut [f32]) {
    let sum: f32 = values.iter().sum();
    if sum != 0.0 {
        for v in values.iter_mut() {
            *v /= sum;
        }
    } else if !values.is_empty() {
        let uniform = 1.0 / values.len() as f32;
        for v in values.iter_mut() {
            *v = uniform;
        }
    }
}

/// FNorm a contiguous transition group `t[start..start+n]` (indices follow the
/// p7H_* layout: MM..MD, IM..II, DM..DD).
fn fnorm_t(t: &mut [f32], start: usize, n: usize) {
    fnorm(&mut t[start..start + n]);
}

fn apply_laplace_priors(hmm: &mut crate::hmm::Hmm) {
    parameterize_with_uniform_alpha(hmm, 1.0);
}

fn parameterize_with_uniform_alpha(hmm: &mut crate::hmm::Hmm, alpha: f64) {
    use crate::hmm::*;
    let k = hmm.abc_k;
    let m = hmm.m;

    let mt_alpha = [alpha; 3];
    let it_alpha = [alpha; 2];
    let dt_alpha = [alpha; 2];

    for node in 0..=m {
        let mt_counts = [hmm.t[node][MM], hmm.t[node][MI], hmm.t[node][MD]];
        let mt_probs = apply_dirichlet_f64(&mt_counts, &mt_alpha);
        hmm.t[node][MM] = mt_probs[0];
        hmm.t[node][MI] = mt_probs[1];
        hmm.t[node][MD] = mt_probs[2];

        let it_counts = [hmm.t[node][IM], hmm.t[node][II]];
        let it_probs = apply_dirichlet_f64(&it_counts, &it_alpha);
        hmm.t[node][IM] = it_probs[0];
        hmm.t[node][II] = it_probs[1];
    }
    hmm.t[m][MD] = 0.0;
    let msum = hmm.t[m][MM] + hmm.t[m][MI] + hmm.t[m][MD];
    if msum > 0.0 {
        hmm.t[m][MM] /= msum;
        hmm.t[m][MI] /= msum;
        hmm.t[m][MD] /= msum;
    }

    for node in 1..m {
        let dt_counts = [hmm.t[node][DM], hmm.t[node][DD]];
        let dt_probs = apply_dirichlet_f64(&dt_counts, &dt_alpha);
        hmm.t[node][DM] = dt_probs[0];
        hmm.t[node][DD] = dt_probs[1];
    }
    hmm.t[0][DM] = 1.0;
    hmm.t[0][DD] = 0.0;
    hmm.t[m][DM] = 1.0;
    hmm.t[m][DD] = 0.0;

    let emission_alpha = vec![alpha; k];
    for node in 1..=m {
        let match_counts: Vec<f32> = hmm.mat[node][..k].to_vec();
        let match_probs = apply_dirichlet_f64(&match_counts, &emission_alpha);
        hmm.mat[node][..k].copy_from_slice(&match_probs);
    }
    hmm.mat[0][..k].fill(0.0);
    if k > 0 {
        hmm.mat[0][0] = 1.0;
    }

    for node in 0..=m {
        let ins_counts: Vec<f32> = hmm.ins[node][..k].to_vec();
        let ins_probs = apply_dirichlet_f64(&ins_counts, &emission_alpha);
        hmm.ins[node][..k].copy_from_slice(&ins_probs);
    }
}
