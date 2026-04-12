//! Build an HMM from a single sequence using a substitution matrix.
//! Simplified port of seqmodel.c p7_Seqmodel().

use crate::alphabet::{Alphabet, Dsq};
use crate::bg::Bg;
use crate::hmm::*;

/// BLOSUM62 scores for the 20 canonical amino acids (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y).
const BLOSUM62_20: [[i32; 20]; 20] = [
    [ 4, 0,-2,-1,-2, 0,-2,-1,-1,-1,-1,-2,-1,-1,-1, 1, 0, 0,-3,-2], // A
    [ 0, 9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2], // C
    [-2,-3, 6, 2,-3,-1,-1,-3,-1,-4,-3, 1,-1, 0,-2, 0,-1,-3,-4,-3], // D
    [-1,-4, 2, 5,-3,-2, 0,-3, 1,-3,-2, 0,-1, 2, 0, 0,-1,-2,-3,-2], // E
    [-2,-2,-3,-3, 6,-3,-1, 0,-3, 0, 0,-3,-4,-3,-3,-2,-2,-1, 1, 3], // F
    [ 0,-3,-1,-2,-3, 6,-2,-4,-2,-4,-3, 0,-2,-2,-2, 0,-2,-3,-2,-3], // G
    [-2,-3,-1, 0,-1,-2, 8,-3,-1,-3,-2, 1,-2, 0, 0,-1,-2,-3,-2, 2], // H
    [-1,-1,-3,-3, 0,-4,-3, 4,-3, 2, 1,-3,-3,-3,-3,-2,-1, 3,-3,-1], // I
    [-1,-3,-1, 1,-3,-2,-1,-3, 5,-2,-1, 0,-1, 1, 2, 0,-1,-2,-3,-2], // K
    [-1,-1,-4,-3, 0,-4,-3, 2,-2, 4, 2,-3,-3,-2,-2,-2,-1, 1,-2,-1], // L
    [-1,-1,-3,-2, 0,-3,-2, 1,-1, 2, 5,-2,-2, 0,-1,-1,-1, 1,-1,-1], // M
    [-2,-3, 1, 0,-3, 0, 1,-3, 0,-3,-2, 6,-2, 0, 0, 1, 0,-3,-4,-2], // N
    [-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2, 7,-1,-2,-1,-1,-2,-4,-3], // P
    [-1,-3, 0, 2,-3,-2, 0,-3, 1,-2, 0, 0,-1, 5, 1, 0,-1,-2,-2,-1], // Q
    [-1,-3,-2, 0,-3,-2, 0,-3, 2,-2,-1, 0,-2, 1, 5,-1,-1,-3,-3,-2], // R
    [ 1,-1, 0, 0,-2, 0,-1,-2, 0,-2,-1, 1,-1, 0,-1, 4, 1,-2,-3,-2], // S
    [ 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5, 0,-2,-2], // T
    [ 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-2,-3,-2, 0, 4,-3,-1], // V
    [-3,-2,-4,-3, 1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11, 2], // W
    [-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1, 2, 7], // Y
];

/// Convert a BLOSUM62 score matrix to conditional probability matrix P(b|a).
/// Returns a 20x20 matrix where row a gives P(target=b | query=a).
fn score_to_conditional(bg_f: &[f32]) -> Vec<Vec<f32>> {
    let k = 20;
    // Find lambda by searching for the value that makes sum(P_ij) = 1
    // For BLOSUM62 with standard amino acid frequencies, lambda ≈ 0.3466
    let lambda = 0.3466_f64;

    // Convert scores to joint probabilities: P_ij = fi * fj * exp(lambda * S_ij)
    let mut joint = vec![vec![0.0_f64; k]; k];
    let mut total = 0.0_f64;
    for a in 0..k {
        for b in 0..k {
            joint[a][b] = (bg_f[a] as f64) * (bg_f[b] as f64) * (lambda * BLOSUM62_20[a][b] as f64).exp();
            total += joint[a][b];
        }
    }
    // Normalize
    for a in 0..k {
        for b in 0..k {
            joint[a][b] /= total;
        }
    }

    // Convert joint to conditional: P(b|a) = P_ab / P(a) where P(a) = sum_b P_ab
    let mut cond = vec![vec![0.0_f32; k]; k];
    for a in 0..k {
        let row_sum: f64 = joint[a].iter().sum();
        for b in 0..k {
            cond[a][b] = if row_sum > 0.0 {
                (joint[a][b] / row_sum) as f32
            } else {
                bg_f[b]
            };
        }
    }
    cond
}

/// Build an HMM from a single sequence using BLOSUM62 scoring.
pub fn build_single_seq_hmm(
    name: &str,
    dsq: &[Dsq],
    seq_len: usize,
    abc: &Alphabet,
    bg: &Bg,
    popen: f32,
    pextend: f32,
) -> Hmm {
    let k = abc.k;
    let m = seq_len;

    // Get conditional probability matrix from BLOSUM62
    let cond = score_to_conditional(&bg.f);

    let mut hmm = Hmm::new(m, abc.abc_type, k);
    hmm.name = name.to_string();

    // Set transitions and emissions for each node
    for node in 1..=m {
        let residue = dsq[node] as usize;

        // Match emissions from conditional probability matrix
        if residue < k {
            for x in 0..k {
                hmm.mat[node][x] = cond[residue][x];
            }
        } else {
            // Unknown residue: use background
            for x in 0..k {
                hmm.mat[node][x] = bg.f[x];
            }
        }

        // Insert emissions = background
        for x in 0..k {
            hmm.ins[node][x] = bg.f[x];
        }

        // Transitions
        if node < m {
            hmm.t[node][MM] = 1.0 - 2.0 * popen;
            hmm.t[node][MI] = popen;
            hmm.t[node][MD] = popen;
            hmm.t[node][IM] = 1.0 - pextend;
            hmm.t[node][II] = pextend;
            hmm.t[node][DM] = 1.0 - pextend;
            hmm.t[node][DD] = pextend;
        } else {
            // Last node: no I or D transitions out
            hmm.t[node][MM] = 1.0;
            hmm.t[node][MI] = 0.0;
            hmm.t[node][MD] = 0.0;
            hmm.t[node][IM] = 1.0;
            hmm.t[node][II] = 0.0;
            hmm.t[node][DM] = 1.0;
            hmm.t[node][DD] = 0.0;
        }
    }

    // Node 0 (begin) transitions
    hmm.t[0][MM] = 1.0 - popen; // B->M1
    hmm.t[0][MI] = popen;       // B->I0 (actually not used)
    hmm.t[0][MD] = popen;       // B->D1
    hmm.t[0][IM] = 1.0 - pextend;
    hmm.t[0][II] = pextend;
    hmm.t[0][DM] = 1.0 - pextend;
    hmm.t[0][DD] = pextend;

    // Insert emissions at node 0
    for x in 0..k {
        hmm.ins[0][x] = bg.f[x];
    }

    // Set composition from background
    for x in 0..k.min(MAXABET) {
        hmm.compo[x] = bg.f[x];
    }

    // Set flags
    hmm.flags |= P7H_COMPO;

    // Set consensus from sequence
    let mut cons = vec![b' '; m + 2];
    for node in 1..=m {
        let residue = dsq[node] as usize;
        if residue < abc.kp {
            cons[node] = abc.sym[residue];
        }
    }
    hmm.consensus = Some(cons);
    hmm.flags |= P7H_CONS;

    // E-value calibration by simulation
    crate::calibrate::calibrate(&mut hmm, abc, bg);

    hmm.nseq = 1;
    hmm.eff_nseq = 1.0;

    hmm
}
