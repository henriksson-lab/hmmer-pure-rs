//! P7_BG - Null/background model for statistical significance.
//! Direct port of p7_bg.c.

use crate::alphabet::{Alphabet, AlphabetType, Dsq};
use crate::errors::{HmmerError, HmmerResult};
use crate::util::cmath::{c_log_f64, c_log_to_f32, c_logf_to_f32};
use std::io::Read;
use std::path::Path;

const MAX_BG_FILE_BYTES: usize = 1024 * 1024;

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
    /// Filter HMM transitions: `t[state][to_state]` for 2-state model
    pub fhmm_t: [[f32; 3]; 2], // [2 states][3: state0, state1, end]
    /// Filter HMM initial state probabilities
    pub fhmm_pi: [f32; 2],
}

impl Bg {
    /// Allocate a P7_BG null model for the given digital alphabet.
    ///
    /// For amino acid alphabets, sets iid background frequencies to average
    /// Swiss-Prot residue composition; for DNA/RNA, sets uniform frequencies.
    /// The bias-filter HMM is not configured here — call `set_filter()`
    /// after this and before using the filter score.
    /// Counterpart to C's `p7_bg_Create()`.
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

    /// Allocate a P7_BG null model with uniform residue frequencies.
    ///
    /// Counterpart to C's `p7_bg_CreateUniform()`. This is primarily used by
    /// experimental calibration/simulation commands that deliberately avoid
    /// HMMER's default amino-acid background composition.
    pub fn new_uniform(abc: &Alphabet) -> Self {
        let mut bg = Self::new(abc);
        let p = 1.0 / abc.k as f32;
        bg.f.fill(p);
        bg.fhmm_e0 = bg.f.clone();
        bg.fhmm_e1 = bg.f.clone();
        bg
    }

    /// Read replacement residue background frequencies from a C HMMER
    /// `p7_bg_Read()` style file.
    pub fn read_file<P: AsRef<Path>>(&mut self, abc: &Alphabet, path: P) -> HmmerResult<()> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                HmmerError::NotFound(format!(
                    "couldn't open bg file  {} for reading",
                    path.display()
                ))
            } else {
                HmmerError::Io(e)
            }
        })?;
        let mut bytes = Vec::new();
        file.by_ref()
            .take((MAX_BG_FILE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(HmmerError::Io)?;
        if bytes.len() > MAX_BG_FILE_BYTES {
            return Err(HmmerError::Format(format!(
                "bg file {} exceeds {} bytes",
                path.display(),
                MAX_BG_FILE_BYTES
            )));
        }
        let text = String::from_utf8(bytes).map_err(|e| {
            HmmerError::Format(format!("invalid UTF-8 in bg file {}: {e}", path.display()))
        })?;
        let mut records = Vec::new();
        for (line_idx, raw_line) in text.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            records.push((line_idx + 1, line.split_whitespace().collect::<Vec<_>>()));
        }
        let Some((line_no, first)) = records.first() else {
            return Err(HmmerError::Format(format!(
                "premature end of file [line 0 of bgfile {}]",
                path.display()
            )));
        };
        if first.len() != 1 {
            return Err(HmmerError::Format(format!(
                "extra unexpected data found [line {} of bgfile {}]",
                line_no,
                path.display()
            )));
        }
        let file_type = parse_bg_alphabet(first[0]).ok_or_else(|| {
            HmmerError::Format(format!(
                "expected alphabet type but saw \"{}\" [line {} of bgfile {}]",
                first[0],
                line_no,
                path.display()
            ))
        })?;
        if file_type != abc.abc_type {
            return Err(HmmerError::Format(format!(
                "bg file's alphabet is {}; expected {} [line {}, {}]",
                first[0],
                bg_alphabet_name(abc.abc_type),
                line_no,
                path.display()
            )));
        }

        let mut fq = vec![-1.0_f32; abc.k];
        let mut n = 0usize;
        for (line_no, fields) in records.iter().skip(1) {
            if fields.len() != 2 {
                return Err(HmmerError::Format(format!(
                    "extra unexpected data found [line {} of bgfile {}]",
                    line_no,
                    path.display()
                )));
            }
            let residue = fields[0].as_bytes();
            if residue.len() != 1 {
                return Err(HmmerError::Format(format!(
                    "expected to parse a residue letter; saw {} [line {} of bgfile {}]",
                    fields[0],
                    line_no,
                    path.display()
                )));
            }
            let x = abc.digitize_symbol(residue[0]);
            if !abc.is_canonical(x) {
                return Err(HmmerError::Format(format!(
                    "expected to parse a residue letter; saw {} [line {} of bgfile {}]",
                    fields[0],
                    line_no,
                    path.display()
                )));
            }
            let x = x as usize;
            if fq[x] != -1.0 {
                return Err(HmmerError::Format(format!(
                    "already parsed probability of {} [line {} of bgfile {}]",
                    abc.sym[x] as char,
                    line_no,
                    path.display()
                )));
            }
            fq[x] = fields[1].parse::<f32>().map_err(|_| {
                HmmerError::Format(format!(
                    "expected a probability, saw {} [line {} of bgfile {}]",
                    fields[1],
                    line_no,
                    path.display()
                ))
            })?;
            n += 1;
        }
        if n != abc.k {
            return Err(HmmerError::Format(format!(
                "expected {} residue frequencies, but found {} in bgfile {}",
                abc.k,
                n,
                path.display()
            )));
        }
        let sum: f32 = fq.iter().sum();
        if (sum - 1.0).abs() > 0.001 {
            return Err(HmmerError::Format(format!(
                "residue frequencies do not sum to 1.0 in bgfile {}",
                path.display()
            )));
        }
        for f in &mut fq {
            *f /= sum;
        }
        self.f = fq;
        self.fhmm_e0 = self.f.clone();
        self.fhmm_e1 = self.f.clone();
        Ok(())
    }

    /// Set the geometric null-model length distribution to a mean of `l` residues.
    /// Counterpart to C's `p7_bg_SetLength()`.
    pub fn set_length(&mut self, l: usize) {
        self.p1 = l as f32 / (l as f32 + 1.0);
        self.fhmm_t[0][0] = self.p1;
        self.fhmm_t[0][1] = 1.0 - self.p1;
    }

    /// Calculate the null1 log-odds score for a sequence of length `l`.
    ///
    /// Because the null1 residue composition matches the background used in
    /// profile scoring, only the null model transitions contribute:
    /// `L*log(p1) + log(1-p1)`. Counterpart to C's `p7_bg_NullOne()`.
    pub fn null_one(&self, l: usize) -> f32 {
        (l as f64 * c_log_f64(self.p1 as f64) + c_log_f64(1.0_f64 - self.p1 as f64)) as f32
    }

    /// Configure the two-state bias-filter HMM from a model's residue composition.
    ///
    /// State 0 is the iid background; state 1 emits the biased composition `compo`
    /// (length K). `m` is the model length, used to set state-1 dwell time
    /// (`L1 = M/8`). The expected total filter length defaults to ~400; a later
    /// `set_length()` call adjusts it to the target sequence length.
    /// Counterpart to C's `p7_bg_SetFilter()`.
    pub fn set_filter(&mut self, m: usize, compo: &[f32]) {
        let l1 = m as f32 / 8.0; // expected length in biased state

        // State 0: background (iid). C p7_bg_SetFilter() resets this to the
        // default expected filter length before any later SetLength() call.
        let l0 = 400.0_f32;
        self.fhmm_t[0][0] = l0 / (l0 + 1.0);
        self.fhmm_t[0][1] = 1.0 / (l0 + 1.0);
        self.fhmm_t[0][2] = 1.0;

        // State 1: biased composition
        self.fhmm_t[1][0] = 1.0 / (l1 + 1.0);
        self.fhmm_t[1][1] = l1 / (l1 + 1.0);
        self.fhmm_t[1][2] = 1.0;

        // Emissions as odds ratios (e[x] / bg_freq[x]), matching C's esl_hmm_Configure()
        let kp = match self.abc_type {
            AlphabetType::Dna | AlphabetType::Rna => 18,
            AlphabetType::Amino => 29,
            AlphabetType::Unknown => self.k,
        };
        self.fhmm_e0 = vec![1.0_f32; kp]; // bg/bg = 1.0, including gap/nonres/missing
        self.fhmm_e1 = vec![1.0_f32; kp];
        for (x, &c) in compo.iter().enumerate().take(self.k) {
            self.fhmm_e1[x] = if self.f[x] > 0.0 { c / self.f[x] } else { 0.0 };
        }

        match self.abc_type {
            AlphabetType::Amino => {
                self.set_degenerate_filter_emission(21, &[11, 2], compo); // B = N or D
                self.set_degenerate_filter_emission(22, &[7, 9], compo); // J = I or L
                self.set_degenerate_filter_emission(23, &[13, 3], compo); // Z = Q or E
                self.set_degenerate_filter_emission(24, &[8], compo); // O = K
                self.set_degenerate_filter_emission(25, &[1], compo); // U = C
                let all: Vec<usize> = (0..self.k).collect();
                self.set_degenerate_filter_emission(26, &all, compo); // X = any residue
            }
            AlphabetType::Dna | AlphabetType::Rna => {
                self.set_degenerate_filter_emission(5, &[0, 2], compo); // R
                self.set_degenerate_filter_emission(6, &[1, 3], compo); // Y
                self.set_degenerate_filter_emission(7, &[0, 1], compo); // M
                self.set_degenerate_filter_emission(8, &[2, 3], compo); // K
                self.set_degenerate_filter_emission(9, &[1, 2], compo); // S
                self.set_degenerate_filter_emission(10, &[0, 3], compo); // W
                self.set_degenerate_filter_emission(11, &[0, 1, 3], compo); // H
                self.set_degenerate_filter_emission(12, &[1, 2, 3], compo); // B
                self.set_degenerate_filter_emission(13, &[0, 1, 2], compo); // V
                self.set_degenerate_filter_emission(14, &[0, 2, 3], compo); // D
                self.set_degenerate_filter_emission(15, &[0, 1, 2, 3], compo); // N
            }
            AlphabetType::Unknown => {}
        }

        // Initial probabilities
        self.fhmm_pi = [0.999, 0.001];
    }

    /// Calculate the bias-filter null log-likelihood for a digital sequence.
    ///
    /// Scores the two-state filter HMM against `dsq[1..=l]` using rescaled
    /// forward DP, then adds the geometric length term. Returns log-likelihood
    /// in nats. Counterpart to C's `p7_bg_FilterScore()`.
    pub fn filter_score(&self, dsq: &[Dsq], l: usize) -> f32 {
        if l == 0 {
            let term = self.fhmm_pi[0] * self.fhmm_t[0][2] + self.fhmm_pi[1] * self.fhmm_t[1][2];
            return c_log_to_f32(term as f64) + c_logf_to_f32(1.0 - self.p1);
        }

        let (e0, e1) = self.filter_emissions(dsq[1] as usize);
        let mut dp = [e0 * self.fhmm_pi[0], e1 * self.fhmm_pi[1]];
        let mut max = dp[0].max(dp[1]);
        dp[0] /= max;
        dp[1] /= max;
        let mut nullsc = c_log_to_f32(max as f64);

        for &x in dsq.iter().take(l + 1).skip(2) {
            let prev = dp;
            let (e0, e1) = self.filter_emissions(x as usize);
            dp[0] = (prev[0] * self.fhmm_t[0][0] + prev[1] * self.fhmm_t[1][0]) * e0;
            dp[1] = (prev[0] * self.fhmm_t[0][1] + prev[1] * self.fhmm_t[1][1]) * e1;
            max = dp[0].max(dp[1]);
            dp[0] /= max;
            dp[1] /= max;
            nullsc += c_log_to_f32(max as f64);
        }

        let term = dp[0] * self.fhmm_t[0][2] + dp[1] * self.fhmm_t[1][2];
        nullsc += c_log_to_f32(term as f64);

        // Apply length distribution
        nullsc + l as f32 * c_logf_to_f32(self.p1) + c_logf_to_f32(1.0 - self.p1)
    }

    /// Look up state-0 and state-1 filter emission odds for digital code `x`.
    #[inline]
    fn filter_emissions(&self, x: usize) -> (f32, f32) {
        if x < self.fhmm_e0.len() {
            (self.fhmm_e0[x], self.fhmm_e1[x])
        } else {
            (1.0, 1.0)
        }
    }

    /// Set state-1 emission odds for a degenerate code `x` as the
    /// composition-weighted average over its canonical members.
    fn set_degenerate_filter_emission(&mut self, x: usize, members: &[usize], compo: &[f32]) {
        if x >= self.fhmm_e1.len() {
            return;
        }
        let mut numer = 0.0_f32;
        let mut denom = 0.0_f32;
        for &y in members {
            numer += compo[y];
            denom += self.f[y];
        }
        self.fhmm_e1[x] = if denom > 0.0 { numer / denom } else { 0.0 };
    }
}

fn parse_bg_alphabet(token: &str) -> Option<AlphabetType> {
    if token.eq_ignore_ascii_case("dna") {
        Some(AlphabetType::Dna)
    } else if token.eq_ignore_ascii_case("rna") {
        Some(AlphabetType::Rna)
    } else if token.eq_ignore_ascii_case("amino") {
        Some(AlphabetType::Amino)
    } else {
        None
    }
}

fn bg_alphabet_name(abc_type: AlphabetType) -> &'static str {
    match abc_type {
        AlphabetType::Dna => "DNA",
        AlphabetType::Rna => "RNA",
        AlphabetType::Amino => "amino",
        AlphabetType::Unknown => "unknown",
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
    fn set_filter_resets_state0_to_default_length() {
        let abc = Alphabet::amino();
        let mut bg = Bg::new(&abc);
        bg.set_length(100);
        let compo = bg.f.clone();

        bg.set_filter(80, &compo);

        assert!((bg.fhmm_t[0][0] - 400.0 / 401.0).abs() < 1e-6);
        assert!((bg.fhmm_t[0][1] - 1.0 / 401.0).abs() < 1e-6);
    }

    #[test]
    fn test_bg_null_one() {
        let abc = Alphabet::amino();
        let mut bg = Bg::new(&abc);
        bg.set_length(100);
        let sc = bg.null_one(100);
        // Score should be L * ln(p1) + ln(1-p1)
        let expected = (100.0 * crate::util::cmath::c_log_f64(bg.p1 as f64)
            + crate::util::cmath::c_log_f64((1.0 - bg.p1) as f64)) as f32;
        assert!((sc - expected).abs() < 1e-6);
    }

    #[test]
    fn bg_reader_rejects_oversized_file_before_full_allocation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.bg");
        std::fs::write(&path, vec![b'A'; MAX_BG_FILE_BYTES + 1]).unwrap();

        let abc = Alphabet::amino();
        let mut bg = Bg::new(&abc);
        let err = bg.read_file(&abc, &path).unwrap_err();

        assert!(
            err.to_string().contains("exceeds"),
            "unexpected error: {err}"
        );
    }
}
