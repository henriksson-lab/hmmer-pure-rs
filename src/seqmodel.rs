//! Build an HMM from a single sequence using a substitution matrix.
//! Simplified port of seqmodel.c p7_Seqmodel().

use crate::alphabet::{Alphabet, Dsq};
use crate::bg::Bg;
use crate::hmm::*;

/// BLOSUM62 scores for the 20 canonical amino acids (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y).
const BLOSUM62_20: [[i32; 20]; 20] = [
    [
        4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2,
    ], // A
    [
        0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2,
    ], // C
    [
        -2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3,
    ], // D
    [
        -1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2,
    ], // E
    [
        -2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3,
    ], // F
    [
        0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3,
    ], // G
    [
        -2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2,
    ], // H
    [
        -1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1,
    ], // I
    [
        -1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2,
    ], // K
    [
        -1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1,
    ], // L
    [
        -1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1,
    ], // M
    [
        -2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2,
    ], // N
    [
        -1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3,
    ], // P
    [
        -1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1,
    ], // Q
    [
        -1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2,
    ], // R
    [
        1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2,
    ], // S
    [
        0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2,
    ], // T
    [
        0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1,
    ], // V
    [
        -3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2,
    ], // W
    [
        -2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7,
    ], // Y
];

/// Convert a BLOSUM62 score matrix to conditional probability matrix P(b|a).
/// Returns a 20x20 matrix where row a gives P(target=b | query=a).
fn score_to_conditional(bg_f: &[f32]) -> Vec<Vec<f32>> {
    let k = 20;
    let lambda = solve_lambda(bg_f);

    let mut joint = vec![vec![0.0_f64; k]; k];
    for a in 0..k {
        for b in 0..k {
            joint[a][b] =
                (bg_f[a] as f64) * (bg_f[b] as f64) * (lambda * BLOSUM62_20[a][b] as f64).exp();
        }
    }

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

fn lambda_f(bg_f: &[f32], lambda: f64) -> f64 {
    let mut fx = -1.0_f64;
    for a in 0..20 {
        for b in 0..20 {
            fx += (bg_f[a] as f64) * (bg_f[b] as f64) * (lambda * BLOSUM62_20[a][b] as f64).exp();
        }
    }
    fx
}

fn solve_lambda(bg_f: &[f32]) -> f64 {
    let max_score = BLOSUM62_20
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .max()
        .unwrap() as f64;
    let mut hi = 1.0 / max_score;
    while hi < 50.0 && lambda_f(bg_f, hi) <= 0.0 {
        hi *= 2.0;
    }
    let mut lo = 0.0_f64;
    for _ in 0..80 {
        let mid = (lo + hi) * 0.5;
        if lambda_f(bg_f, mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    (lo + hi) * 0.5
}

fn calculate_occupancy(hmm: &Hmm) -> (Vec<f32>, Vec<f32>) {
    let mut mocc = vec![0.0_f32; hmm.m + 1];
    let mut iocc = vec![0.0_f32; hmm.m + 1];

    mocc[0] = 0.0;
    mocc[1] = hmm.t[0][MI] + hmm.t[0][MM];
    for k in 2..=hmm.m {
        mocc[k] = mocc[k - 1] * (hmm.t[k - 1][MM] + hmm.t[k - 1][MI])
            + (1.0 - mocc[k - 1]) * hmm.t[k - 1][DM];
    }

    iocc[0] = hmm.t[0][MI] / hmm.t[0][IM];
    for k in 1..=hmm.m {
        iocc[k] = mocc[k] * hmm.t[k][MI] / hmm.t[k][IM];
    }

    (mocc, iocc)
}

fn set_composition(hmm: &mut Hmm) {
    let (mocc, iocc) = calculate_occupancy(hmm);
    for x in 0..hmm.abc_k.min(MAXABET) {
        hmm.compo[x] = hmm.ins[0][x] * iocc[0];
    }
    for k in 1..=hmm.m {
        for x in 0..hmm.abc_k.min(MAXABET) {
            hmm.compo[x] += hmm.mat[k][x] * mocc[k] + hmm.ins[k][x] * iocc[k];
        }
    }

    let sum: f32 = hmm.compo[..hmm.abc_k.min(MAXABET)].iter().sum();
    if sum > 0.0 {
        for x in 0..hmm.abc_k.min(MAXABET) {
            hmm.compo[x] /= sum;
        }
    }
    hmm.flags |= P7H_COMPO;
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

    // Mirror C p7_Seqmodel (hmmer/src/seqmodel.c:55) exactly: set transitions
    // for every node k in 0..=M with the same formula, then override a
    // subset of node M's transitions at the end. Rust previously hand-wrote
    // node 0 with t[0][MM]=1-popen and node M with all-zeroed I/D
    // transitions, producing ~21-bit score inflation vs C phmmer.
    for node in 0..=m {
        // Match emissions from conditional probability matrix (only for k>0).
        if node > 0 {
            let residue = dsq[node] as usize;
            if residue < k {
                for x in 0..k {
                    hmm.mat[node][x] = cond[residue][x];
                }
            } else {
                for x in 0..k {
                    hmm.mat[node][x] = bg.f[x];
                }
            }
        }

        // Insert emissions = background, for every node including 0.
        for x in 0..k {
            hmm.ins[node][x] = bg.f[x];
        }

        hmm.t[node][MM] = 1.0 - 2.0 * popen;
        hmm.t[node][MI] = popen;
        hmm.t[node][MD] = popen;
        hmm.t[node][IM] = 1.0 - pextend;
        hmm.t[node][II] = pextend;
        hmm.t[node][DM] = 1.0 - pextend;
        hmm.t[node][DD] = pextend;
    }

    // Special handling at node M (C seqmodel.c:85): overrides MM, MD, DM, DD
    // ONLY. MI, IM, II keep their general-formula values from the loop above.
    hmm.t[m][MM] = 1.0 - popen;
    hmm.t[m][MD] = 0.0;
    hmm.t[m][DM] = 1.0;
    hmm.t[m][DD] = 0.0;

    // Insert emissions at node 0
    for x in 0..k {
        hmm.ins[0][x] = bg.f[x];
    }

    set_composition(&mut hmm);

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
