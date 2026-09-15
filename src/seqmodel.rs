//! Creating profile HMMs from single sequences.
//! Port of seqmodel.c p7_Seqmodel().
//!
//! This module is deliberately thin: it is only `p7_Seqmodel()`. The score
//! system that feeds it lives in [`crate::util::scorematrix`] (Easel's
//! `esl_scorematrix.c`), and the orchestration around it —
//! `p7_hmm_SetComposition()`, `p7_hmm_SetConsensus()`, calibration and the
//! nucleotide window length — lives in [`crate::builder::Builder`], mirroring
//! C's `p7_SingleBuilder()` (p7_builder.c:496).

use crate::alphabet::{Alphabet, Dsq};
use crate::bg::Bg;
use crate::builder::Builder;
use crate::calibrate::CalibrationConfig;
use crate::hmm::*;

pub use crate::util::scorematrix::{is_known_builtin_score_matrix_name, ScoreMatrix};

/// Port of `p7_Seqmodel()` (seqmodel.c:52).
///
/// Makes a profile HMM from a single sequence, for probabilistic
/// Smith/Waterman alignment, HMMER3-style.
///
/// The query is digital sequence `dsq` of length `m` residues in alphabet
/// `abc`, named `name`. The scoring system is given by `q`, `f`, `popen` and
/// `pextend`: `q` is a `Kp x Kp` matrix of conditional residue probabilities
/// `P(a | b)`, typically obtained by reverse engineering a score matrix; `f` is
/// the vector of `K` background frequencies; `popen` and `pextend` are the
/// probabilities assigned to gap-open (`t_MI`, `t_MD`) and gap-extend (`t_II`,
/// `t_DD`) transitions.
///
/// `popen`/`pextend` are `f64` because C's are: they come from
/// `esl_opt_GetReal()`, so `1.0 - 2 * popen` is evaluated in double and only
/// narrowed to `float` on assignment.
#[allow(clippy::too_many_arguments)]
pub fn seqmodel(
    abc: &Alphabet,
    dsq: &[Dsq],
    m: usize,
    name: &str,
    q: &[Vec<f64>],
    f: &[f32],
    popen: f64,
    pextend: f64,
) -> Hmm {
    let k = abc.k;
    let mut hmm = Hmm::new(m, abc.abc_type, k);

    for node in 0..=m {
        // Use rows of P matrix as source of match emission vectors.
        // C: esl_vec_D2F(Q->mx[(int) dsq[k]], abc->K, hmm->mat[k]).
        if node > 0 {
            let residue = dsq[node] as usize;
            if residue < q.len() {
                for x in 0..k {
                    hmm.mat[node][x] = q[residue][x] as f32;
                }
            } else {
                // Not reachable for a validly digitized sequence; every digital
                // code is < Kp. Kept so a malformed dsq degrades to background
                // rather than panicking.
                hmm.mat[node][..k].copy_from_slice(&f[..k]);
            }
        }

        // Set inserts to background for now. This will be improved.
        hmm.ins[node][..k].copy_from_slice(&f[..k]);

        hmm.t[node][MM] = (1.0 - 2.0 * popen) as f32;
        hmm.t[node][MI] = popen as f32;
        hmm.t[node][MD] = popen as f32;
        hmm.t[node][IM] = (1.0 - pextend) as f32;
        hmm.t[node][II] = pextend as f32;
        hmm.t[node][DM] = (1.0 - pextend) as f32;
        hmm.t[node][DD] = pextend as f32;
    }

    // Deal w/ special stuff at node M, overwriting a little of what we just
    // did (seqmodel.c:80-85). Note this overrides MM, MD, DM and DD only:
    // MI, IM and II keep their general-formula values from the loop above.
    hmm.t[m][MM] = (1.0 - popen) as f32;
    hmm.t[m][MD] = 0.0;
    hmm.t[m][DM] = 1.0;
    hmm.t[m][DD] = 0.0;

    // Add mandatory annotation (seqmodel.c:87-93).
    hmm.name = name.to_string();
    hmm.comlog = Some("[HMM created from a query sequence]".to_string());
    hmm.nseq = 1;
    hmm.checksum = 0;
    // C also calls p7_hmm_SetCtime() here. This port leaves `ctime` unset on
    // every build path, not just this one, so that written models stay
    // byte-reproducible; see TODO.md.

    hmm
}

/// Build a profile HMM from one query sequence using BLOSUM62 (amino) or DNA1
/// (nucleotide) scoring.
///
/// Compatibility wrapper kept for the published 0.7.x API; new code should use
/// [`Builder`] directly, which is the 1:1 counterpart of C's
/// `p7_builder_*` / `p7_SingleBuilder()` pair.
///
/// Unlike the 0.7.x version, this picks the default matrix by alphabet the way
/// `p7_builder_SetScoreSystem()` does (p7_builder.c:296-301) instead of
/// hardcoding BLOSUM62, so it no longer panics on a DNA/RNA alphabet.
pub fn build_single_seq_hmm(
    name: &str,
    dsq: &[Dsq],
    seq_len: usize,
    abc: &Alphabet,
    bg: &Bg,
    popen: f32,
    pextend: f32,
) -> Hmm {
    let matrix_name = ScoreMatrix::default_builtin_name(abc.abc_type);
    let matrix = ScoreMatrix::builtin_for_alphabet(matrix_name, abc)
        .expect("default score matrix should be a built-in for this alphabet");
    build_single_seq_hmm_with_matrix(name, dsq, seq_len, abc, bg, &matrix, popen, pextend)
        .expect("default score matrix should be valid")
}

/// As [`build_single_seq_hmm`], with an explicit score matrix.
/// Compatibility wrapper over [`Builder`].
#[allow(clippy::too_many_arguments)]
pub fn build_single_seq_hmm_with_matrix(
    name: &str,
    dsq: &[Dsq],
    seq_len: usize,
    abc: &Alphabet,
    bg: &Bg,
    matrix: &ScoreMatrix,
    popen: f32,
    pextend: f32,
) -> Result<Hmm, String> {
    build_single_seq_hmm_with_matrix_and_calibration(
        name,
        dsq,
        seq_len,
        abc,
        bg,
        matrix,
        popen,
        pextend,
        crate::builder::DEFAULT_BUILDER_SEED,
        CalibrationConfig::default(),
    )
}

/// As [`build_single_seq_hmm_with_matrix`], with an explicit calibration seed
/// and config. Compatibility wrapper over [`Builder`].
#[allow(clippy::too_many_arguments)]
pub fn build_single_seq_hmm_with_matrix_and_calibration(
    name: &str,
    dsq: &[Dsq],
    seq_len: usize,
    abc: &Alphabet,
    bg: &Bg,
    matrix: &ScoreMatrix,
    popen: f32,
    pextend: f32,
    calibration_seed: u32,
    calibration_config: CalibrationConfig,
) -> Result<Hmm, String> {
    let mut bld = Builder::new(abc.abc_type)
        .with_seed(calibration_seed)
        .with_calibration(calibration_config);
    bld.set_score_system_from_matrix(matrix.clone(), popen as f64, pextend as f64, bg, abc)?;
    let mut hmm = bld.single_builder(name, dsq, seq_len, abc, bg)?;
    // C's p7_SingleBuilder leaves eff_nseq at its p7_hmm_Create() sentinel; the
    // 0.7.x wrappers set it to 1.0 and callers/writers depend on that.
    hmm.eff_nseq = 1.0;
    Ok(hmm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::AlphabetType;

    /// F3: single-seq consensus is case-thresholded like C's
    /// `p7_hmm_SetConsensus` (`p7_hmm.c:720-721`): a residue whose own
    /// match-emission probability is below `mthresh` (0.5 amino) is
    /// lower-cased, otherwise upper-cased.
    #[test]
    fn single_seq_consensus_case_thresholded() {
        let abc = Alphabet::new(AlphabetType::Amino);
        let bg = Bg::new(&abc);
        let seq = b"ACDEFGHIKLMNPQRSTVWY";
        let mut dsq = vec![0u8; seq.len() + 2];
        for (i, &c) in seq.iter().enumerate() {
            dsq[i + 1] = abc.digitize_symbol(c);
        }
        let hmm = build_single_seq_hmm("q", &dsq, seq.len(), &abc, &bg, 0.02, 0.4);
        let cons = hmm.consensus.as_ref().expect("consensus set");
        for node in 1..=hmm.m {
            let residue = dsq[node] as usize;
            let p = hmm.mat[node][residue];
            let ch = cons[node];
            if p >= 0.5 {
                assert!(
                    ch.is_ascii_uppercase(),
                    "node {node} p={p} should be UPPER, got {}",
                    ch as char
                );
            } else {
                assert!(
                    ch.is_ascii_lowercase(),
                    "node {node} p={p} should be lower, got {}",
                    ch as char
                );
            }
        }
        assert!(
            (1..=hmm.m).any(|k| cons[k].is_ascii_lowercase()),
            "expected at least one lower-cased consensus residue"
        );
    }

    /// The regression behind issue #1: a DNA query must build without panicking
    /// and must carry a window length, as `p7_SingleBuilder()` sets at
    /// p7_builder.c:512-516. Before this port, `build_single_seq_hmm` hardcoded
    /// BLOSUM62 and panicked here, and no single-sequence path set `max_length`.
    #[test]
    fn single_seq_dna_builds_and_sets_max_length() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let bg = Bg::new(&abc);
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
        let mut dsq = vec![0u8; seq.len() + 2];
        for (i, &c) in seq.iter().enumerate() {
            dsq[i + 1] = abc.digitize_symbol(c);
        }
        let hmm = build_single_seq_hmm("q", &dsq, seq.len(), &abc, &bg, 0.03125, 0.75);
        assert_eq!(hmm.m, seq.len());
        assert!(
            hmm.max_length > 0,
            "DNA single-seq model must carry MAXL, got {}",
            hmm.max_length
        );
    }

    /// Amino models must *not* get a window length: C guards the block on
    /// `eslDNA || eslRNA` (p7_builder.c:512).
    #[test]
    fn single_seq_amino_leaves_max_length_unset() {
        let abc = Alphabet::new(AlphabetType::Amino);
        let bg = Bg::new(&abc);
        let seq = b"ACDEFGHIKLMNPQRSTVWY";
        let mut dsq = vec![0u8; seq.len() + 2];
        for (i, &c) in seq.iter().enumerate() {
            dsq[i + 1] = abc.digitize_symbol(c);
        }
        let hmm = build_single_seq_hmm("q", &dsq, seq.len(), &abc, &bg, 0.02, 0.4);
        assert_eq!(hmm.max_length, -1);
    }
}
