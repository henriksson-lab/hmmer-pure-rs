//! P7_BG - Null/background model for statistical significance.
//! Direct port of p7_bg.c.

use crate::alphabet::{Alphabet, AlphabetType, Dsq};

/// Default amino acid background frequencies from Swiss-Prot 50.8 (Oct 2006).
/// Order: ACDEFGHIKLMNPQRSTVWY (Easel canonical amino acid order).
pub const AMINO_FREQUENCIES: [f32; 20] = [
    0.0787945, // A
    0.0151600, // C
    0.0535222, // D
    0.0668298, // E
    0.0397062, // F
    0.0695071, // G
    0.0229198, // H
    0.0590092, // I
    0.0594422, // K
    0.0963728, // L
    0.0237718, // M
    0.0414386, // N
    0.0482904, // P
    0.0395639, // Q
    0.0540978, // R
    0.0683364, // S
    0.0540687, // T
    0.0673417, // V
    0.0114135, // W
    0.0304133, // Y
];

/// Background/null model for HMMER.
#[derive(Debug, Clone)]
pub struct Bg {
    /// Background residue frequencies [0..K-1]
    pub f: Vec<f32>,
    /// Null1 geometric transition probability: L/(L+1)
    pub p1: f32,
    /// Prior weight for null2/null3 models (constant 1/256)
    pub omega: f32,
    /// Alphabet size (K)
    pub k: usize,
    /// Alphabet type
    pub abc_type: AlphabetType,

    // Bias filter HMM (simplified: 2-state)
    /// Filter HMM state 0 emission probabilities [0..K-1] (background)
    pub fhmm_e0: Vec<f32>,
    /// Filter HMM state 1 emission probabilities [0..K-1] (biased composition)
    pub fhmm_e1: Vec<f32>,
    /// Filter HMM transitions: t[state][to_state] for 2-state model
    pub fhmm_t: [[f32; 3]; 2], // [2 states][3: state0, state1, end]
    /// Filter HMM initial state probabilities
    pub fhmm_pi: [f32; 2],
}

impl Bg {
    /// Create a new background model for the given alphabet.
    pub fn new(abc: &Alphabet) -> Self {
        let k = abc.k;
        let mut f = vec![1.0 / k as f32; k]; // uniform default

        if abc.abc_type == AlphabetType::Amino {
            f.copy_from_slice(&AMINO_FREQUENCIES);
        }

        let p1 = 350.0 / 351.0;
        let omega = 1.0 / 256.0;

        // Initialize filter HMM with background frequencies
        let fhmm_e0 = f.clone();
        let fhmm_e1 = f.clone();

        Bg {
            f,
            p1,
            omega,
            k,
            abc_type: abc.abc_type,
            fhmm_e0,
            fhmm_e1,
            fhmm_t: [
                [p1, 1.0 - p1, 1.0], // state 0 transitions
                [p1, 1.0 - p1, 1.0], // state 1 transitions
            ],
            fhmm_pi: [0.999, 0.001],
        }
    }

    /// Set the null model's expected sequence length to L.
    pub fn set_length(&mut self, l: usize) {
        self.p1 = l as f32 / (l as f32 + 1.0);
        self.fhmm_t[0][0] = self.p1;
        self.fhmm_t[0][1] = 1.0 - self.p1;
    }

    /// Calculate the null1 log-odds score for a sequence of length L.
    /// This is the length-dependent component of the null model.
    pub fn null_one(&self, l: usize) -> f32 {
        l as f32 * self.p1.ln() + (1.0 - self.p1).ln()
    }

    /// Configure the bias filter HMM with model composition.
    /// `compo` is the model's residue composition [0..K-1].
    /// `m` is the model length.
    pub fn set_filter(&mut self, m: usize, compo: &[f32]) {
        let l0 = 400.0_f32; // expected length in background state
        let l1 = (m as f32 / 8.0).max(1.0); // expected length in biased state

        // State 0: background (iid)
        self.fhmm_t[0][0] = l0 / (l0 + 1.0);
        self.fhmm_t[0][1] = 1.0 / (l0 + 1.0);
        self.fhmm_t[0][2] = 1.0;

        // State 1: biased composition
        self.fhmm_t[1][0] = 1.0 / (l1 + 1.0);
        self.fhmm_t[1][1] = l1 / (l1 + 1.0);
        self.fhmm_t[1][2] = 1.0;

        // Emissions as odds ratios (e[x] / bg_freq[x]), matching C's esl_hmm_Configure()
        self.fhmm_e0 = vec![1.0_f32; self.k]; // bg/bg = 1.0
        self.fhmm_e1 = compo[..self.k].iter().zip(self.f[..self.k].iter())
            .map(|(&c, &f)| if f > 0.0 { c / f } else { 0.0 })
            .collect();

        // Initial probabilities
        self.fhmm_pi = [0.999, 0.001];
    }

    /// Calculate the bias filter score for a digital sequence.
    /// `dsq` is 1-based digital sequence.
    /// Returns the filter score (null model log-likelihood in nats).
    pub fn filter_score(&self, dsq: &[Dsq], l: usize) -> f32 {
        // Forward algorithm on the 2-state filter HMM
        let k = self.k;

        // dp[state] = forward probability at current position
        let mut dp = [0.0_f32; 2];

        // Initialize: position 0
        dp[0] = self.fhmm_pi[0];
        dp[1] = self.fhmm_pi[1];

        let mut total_logsc = 0.0_f32;

        for i in 1..=l {
            let x = dsq[i] as usize;
            if x >= k {
                // Non-canonical residue: treat as background
                continue;
            }

            let prev = dp;

            // Transition + emission for state 0
            dp[0] = (prev[0] * self.fhmm_t[0][0] + prev[1] * self.fhmm_t[1][0])
                * self.fhmm_e0[x];
            // Transition + emission for state 1
            dp[1] = (prev[0] * self.fhmm_t[0][1] + prev[1] * self.fhmm_t[1][1])
                * self.fhmm_e1[x];

            // Rescale to prevent underflow
            let scale = dp[0] + dp[1];
            if scale > 0.0 {
                dp[0] /= scale;
                dp[1] /= scale;
                total_logsc += scale.ln();
            }
        }

        // Terminate: multiply by exit probabilities
        let term = dp[0] * self.fhmm_t[0][2] + dp[1] * self.fhmm_t[1][2];
        let nullsc = total_logsc + term.ln();

        // Apply length distribution
        nullsc + l as f32 * self.p1.ln() + (1.0 - self.p1).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;

    #[test]
    fn test_bg_create_amino() {
        let abc = Alphabet::amino();
        let bg = Bg::new(&abc);
        assert_eq!(bg.k, 20);
        assert!((bg.f[0] - 0.0787945).abs() < 1e-6); // A
        assert!((bg.p1 - 350.0 / 351.0).abs() < 1e-6);
        assert!((bg.omega - 1.0 / 256.0).abs() < 1e-8);
    }

    #[test]
    fn test_bg_set_length() {
        let abc = Alphabet::amino();
        let mut bg = Bg::new(&abc);
        bg.set_length(400);
        assert!((bg.p1 - 400.0 / 401.0).abs() < 1e-6);
    }

    #[test]
    fn test_bg_null_one() {
        let abc = Alphabet::amino();
        let mut bg = Bg::new(&abc);
        bg.set_length(100);
        let sc = bg.null_one(100);
        // Score should be L * ln(p1) + ln(1-p1)
        let expected = 100.0 * bg.p1.ln() + (1.0 - bg.p1).ln();
        assert!((sc - expected).abs() < 1e-6);
    }
}
