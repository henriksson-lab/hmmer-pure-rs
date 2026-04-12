//! Null2 bias correction by expectation from posterior probabilities.
//! Simplified port of generic_null2.c p7_GNull2_ByExpectation().

use crate::dp::gmx::*;
use crate::hmm::Hmm;
use crate::profile::Profile;

/// Compute null2 odds ratios from a posterior probability matrix.
///
/// Returns a vector `null2[0..K-1]` where null2[x] = f'(x) / f(x) is the
/// odds ratio of the domain-specific composition vs background.
///
/// The "bias" is computed as sum of log(null2[dsq[i]]) over domain positions,
/// converted to bits.
pub fn null2_by_expectation(
    gm: &Profile,
    hmm: &Hmm,
    pp: &Gmx,
    bg_f: &[f32],
) -> Vec<f32> {
    let m = gm.m;
    let k = gm.abc_k;
    let l = pp.l;

    // 1. Compute expected state usage across all positions
    // For each state, sum the posterior probabilities across all positions
    let mut match_usage = vec![0.0_f32; m + 1];
    let mut insert_usage = vec![0.0_f32; m + 1];
    let mut n_usage = 0.0_f32;
    let mut c_usage = 0.0_f32;
    let mut j_usage = 0.0_f32;

    for i in 1..=l {
        for node in 1..=m {
            match_usage[node] += pp.mmx(i, node);
            insert_usage[node] += pp.imx(i, node);
        }
        n_usage += pp.xmx(i, P7G_N);
        c_usage += pp.xmx(i, P7G_C);
        j_usage += pp.xmx(i, P7G_J);
    }

    // 2. Compute expected composition for each residue
    let mut fpc = vec![0.0_f32; k]; // f'(x): domain-specific composition

    for x in 0..k {
        let mut sum = 0.0_f32;
        for node in 1..=m {
            // Match state contribution: usage * emission probability
            sum += match_usage[node] * hmm.mat[node][x];
            // Insert state contribution: usage * insert emission
            // Insert emissions are typically background (hmm.ins[node][x])
            if node < m {
                sum += insert_usage[node] * hmm.ins[node][x];
            }
        }
        // N, C, J states emit with background frequencies
        sum += (n_usage + c_usage + j_usage) * bg_f[x];
        fpc[x] = sum;
    }

    // Normalize fpc to sum to 1
    let fpc_sum: f32 = fpc.iter().sum();
    if fpc_sum > 0.0 {
        for x in 0..k {
            fpc[x] /= fpc_sum;
        }
    }

    // 3. Compute odds ratios: null2[x] = fpc[x] / bg[x]
    let mut null2 = vec![1.0_f32; k];
    for x in 0..k {
        if bg_f[x] > 0.0 {
            null2[x] = fpc[x] / bg_f[x];
        }
    }

    null2
}

/// Compute the null2 bias correction score in nats for a domain.
/// `dsq` is the 1-based digital sequence.
/// `ienv`, `jenv` are the envelope boundaries (1-based).
pub fn null2_score(null2: &[f32], dsq: &[u8], ienv: usize, jenv: usize) -> f32 {
    let mut score = 0.0_f32;
    for i in ienv..=jenv {
        let x = dsq[i] as usize;
        if x < null2.len() && null2[x] > 0.0 {
            score += null2[x].ln();
        }
    }
    score
}
