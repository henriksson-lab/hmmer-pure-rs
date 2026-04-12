//! Dirichlet mixture priors for HMM parameterization.
//! Simplified port of p7_prior.c.

/// Apply Dirichlet prior pseudocounts to a count vector.
/// `counts` is the observed counts [0..K-1].
/// `alpha` is the Dirichlet parameter vector [0..K-1].
/// Returns the posterior mean estimate.
pub fn apply_dirichlet(counts: &[f32], alpha: &[f32]) -> Vec<f32> {
    let k = counts.len();
    let mut result = vec![0.0_f32; k];
    let count_sum: f32 = counts.iter().sum();
    let alpha_sum: f32 = alpha.iter().sum();
    let total = count_sum + alpha_sum;

    if total > 0.0 {
        for i in 0..k {
            result[i] = (counts[i] + alpha[i]) / total;
        }
    }
    result
}

/// Sjolander 9-component mixture Dirichlet for amino acid match emissions.
/// Trained on Blocks9 database [Sjolander96].
const SJOLANDER_Q: [f64; 9] = [
    0.178091, 0.056591, 0.0960191, 0.0781233, 0.0834977,
    0.0904123, 0.114468, 0.0682132, 0.234585,
];

const SJOLANDER_ALPHA: [[f64; 20]; 9] = [
    [0.270671, 0.039848, 0.017576, 0.016415, 0.014268, 0.131916, 0.012391, 0.022599, 0.020358, 0.030727, 0.015315, 0.048298, 0.053803, 0.020662, 0.023612, 0.216147, 0.147226, 0.065438, 0.003758, 0.009621],
    [0.021465, 0.010300, 0.011741, 0.010883, 0.385651, 0.016416, 0.076196, 0.035329, 0.013921, 0.093517, 0.022034, 0.028593, 0.013086, 0.023011, 0.018866, 0.029156, 0.018153, 0.036100, 0.071770, 0.419641],
    [0.561459, 0.045448, 0.438366, 0.764167, 0.087364, 0.259114, 0.214940, 0.145928, 0.762204, 0.247320, 0.118662, 0.441564, 0.174822, 0.530840, 0.465529, 0.583402, 0.445586, 0.227050, 0.029510, 0.121090],
    [0.070143, 0.011140, 0.019479, 0.094657, 0.013162, 0.048038, 0.077000, 0.032939, 0.576639, 0.072293, 0.028240, 0.080372, 0.037661, 0.185037, 0.506783, 0.073732, 0.071587, 0.042532, 0.011254, 0.028723],
    [0.041103, 0.014794, 0.005610, 0.010216, 0.153602, 0.007797, 0.007175, 0.299635, 0.010849, 0.999446, 0.210189, 0.006127, 0.013021, 0.019798, 0.014509, 0.012049, 0.035799, 0.180085, 0.012744, 0.026466],
    [0.115607, 0.037381, 0.012414, 0.018179, 0.051778, 0.017255, 0.004911, 0.796882, 0.017074, 0.285858, 0.075811, 0.014548, 0.015092, 0.011382, 0.012696, 0.027535, 0.088333, 0.944340, 0.004373, 0.016741],
    [0.093461, 0.004737, 0.387252, 0.347841, 0.010822, 0.105877, 0.049776, 0.014963, 0.094276, 0.027761, 0.010040, 0.187869, 0.050018, 0.110039, 0.038668, 0.119471, 0.065802, 0.025430, 0.003215, 0.018742],
    [0.452171, 0.114613, 0.062460, 0.115702, 0.284246, 0.140204, 0.100358, 0.550230, 0.143995, 0.700649, 0.276580, 0.118569, 0.097470, 0.126673, 0.143634, 0.278983, 0.358482, 0.661750, 0.061533, 0.199373],
    [0.005193, 0.004039, 0.006722, 0.006121, 0.003468, 0.016931, 0.003647, 0.002184, 0.005019, 0.005990, 0.001473, 0.004158, 0.009055, 0.003630, 0.006583, 0.003172, 0.003690, 0.002967, 0.002772, 0.002686],
];

/// Apply mixture Dirichlet prior to a count vector.
/// Returns posterior mean estimate using 9-component Sjolander mixture.
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
        let mut log_p = SJOLANDER_Q[comp].ln();
        log_p += ln_gamma(alpha_sum) - ln_gamma(count_sum + alpha_sum);
        for a in 0..k {
            log_p += ln_gamma(counts_d[a] + SJOLANDER_ALPHA[comp][a])
                - ln_gamma(SJOLANDER_ALPHA[comp][a]);
        }
        log_pk[comp] = log_p;
    }

    // Normalize posterior component probabilities
    let max_log = log_pk.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut pk = [0.0_f64; 9];
    let mut pk_sum = 0.0_f64;
    for comp in 0..9 {
        pk[comp] = (log_pk[comp] - max_log).exp();
        pk_sum += pk[comp];
    }
    for comp in 0..9 {
        pk[comp] /= pk_sum;
    }

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
    let rsum: f64 = result.iter().sum();
    if rsum > 0.0 {
        for a in 0..k {
            result[a] /= rsum;
        }
    }

    result.iter().map(|&v| v as f32).collect()
}

/// Log-gamma using Lanczos approximation (accurate for all positive x).
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Lanczos coefficients (g=7, n=9)
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut sum = C[0];
    for i in 1..9 {
        sum += C[i] / (x + i as f64);
    }
    let t = x + 7.5; // g + 0.5
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Default transition Dirichlet prior parameters.
pub struct TransitionPrior {
    /// Match transitions: MM, MI, MD
    pub tm: [f32; 3],
    /// Insert transitions: IM, II
    pub ti: [f32; 2],
    /// Delete transitions: DM, DD
    pub td: [f32; 2],
}

impl Default for TransitionPrior {
    fn default() -> Self {
        TransitionPrior {
            tm: [0.7939, 0.0278, 0.0135], // Mitchison, trained on Pfam
            ti: [0.1551, 0.1331],
            td: [0.9002, 0.5630],
        }
    }
}

/// Apply Dirichlet priors to an HMM's emission and transition counts.
/// Modifies the HMM in place, converting counts to probabilities.
pub fn apply_priors(hmm: &mut crate::hmm::Hmm) {
    use crate::hmm::*;
    let k = hmm.abc_k;
    let m = hmm.m;
    let trans = TransitionPrior::default();

    // Background for insert emissions
    let bg_alpha: Vec<f32> = (0..k).map(|_| 1.0).collect(); // uniform

    for node in 1..=m {
        // Apply match emission prior (9-component Sjolander mixture for amino acids)
        let match_counts: Vec<f32> = hmm.mat[node][..k].to_vec();
        let match_probs = if k == 20 {
            apply_mixture_dirichlet(&match_counts)
        } else {
            apply_dirichlet(&match_counts, &[0.5_f32; 20][..k])
        };
        hmm.mat[node][..k].copy_from_slice(&match_probs);

        // Apply insert emission prior (uniform)
        let ins_counts: Vec<f32> = hmm.ins[node][..k].to_vec();
        let ins_probs = apply_dirichlet(&ins_counts, &bg_alpha);
        hmm.ins[node][..k].copy_from_slice(&ins_probs);

        // Apply transition priors
        if node < m {
            // Match transitions: MM, MI, MD
            let mt_counts = [hmm.t[node][MM], hmm.t[node][MI], hmm.t[node][MD]];
            let mt_probs = apply_dirichlet(&mt_counts, &trans.tm);
            hmm.t[node][MM] = mt_probs[0];
            hmm.t[node][MI] = mt_probs[1];
            hmm.t[node][MD] = mt_probs[2];

            // Insert transitions: IM, II
            let it_counts = [hmm.t[node][IM], hmm.t[node][II]];
            let it_probs = apply_dirichlet(&it_counts, &trans.ti);
            hmm.t[node][IM] = it_probs[0];
            hmm.t[node][II] = it_probs[1];

            // Delete transitions: DM, DD
            let dt_counts = [hmm.t[node][DM], hmm.t[node][DD]];
            let dt_probs = apply_dirichlet(&dt_counts, &trans.td);
            hmm.t[node][DM] = dt_probs[0];
            hmm.t[node][DD] = dt_probs[1];
        }
    }
}
