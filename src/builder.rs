//! HMM builder — construct profile HMMs from multiple sequence alignments.
//! Simplified port of p7_builder.c and build.c.

use crate::alphabet::Alphabet;
use crate::bg::Bg;
use crate::hmm::*;
use crate::msa::Msa;

/// Henikoff position-based sequence weighting.
/// Returns weights[0..nseq] that sum to nseq.
pub fn pb_weights(msa: &Msa, abc: &Alphabet) -> Vec<f32> {
    let nseq = msa.nseq;
    let k = abc.k;

    let ax = msa.digitize(abc);
    let mut weights = vec![0.0_f32; nseq];

    for col in 0..msa.alen {
        // Count distinct residues and per-residue counts at this column
        let mut counts = vec![0usize; k];

        for seq in 0..nseq {
            let pos = col + 1;
            if pos < ax[seq].len() - 1 {
                let residue = ax[seq][pos] as usize;
                if residue < k {
                    counts[residue] += 1;
                }
            }
        }

        let r = counts.iter().filter(|&&c| c > 0).count(); // number of distinct residues
        if r == 0 {
            continue;
        }

        // Weight each sequence's contribution at this column
        for seq in 0..nseq {
            let pos = col + 1;
            if pos < ax[seq].len() - 1 {
                let residue = ax[seq][pos] as usize;
                if residue < k && counts[residue] > 0 {
                    weights[seq] += 1.0 / (r as f32 * counts[residue] as f32);
                }
            }
        }
    }

    // Normalize: each weight divided by number of residues in that sequence
    for seq in 0..nseq {
        let mut n_res = 0;
        for col in 0..msa.alen {
            let pos = col + 1;
            if pos < ax[seq].len() - 1 {
                let residue = ax[seq][pos] as usize;
                if residue < k {
                    n_res += 1;
                }
            }
        }
        if n_res > 0 {
            weights[seq] /= n_res as f32;
        }
    }

    // Normalize weights to sum to nseq
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        let scale = nseq as f32 / sum;
        for w in &mut weights {
            *w *= scale;
        }
    }

    weights
}

/// Build an HMM from a multiple sequence alignment using fast model construction.
///
/// Uses the "fast" method: columns with >= `symfrac` occupancy become match states.
/// Default symfrac = 0.5.
pub fn build_hmm_from_msa(msa: &Msa, abc: &Alphabet, bg: &Bg, symfrac: f32) -> Hmm {
    let k = abc.k;
    let nseq = msa.nseq;

    // Digitize the alignment
    let ax = msa.digitize(abc);
    let gap = abc.gap_code();

    // Step 1: Determine which columns are match states (Fast model construction)
    let mut matassign = vec![false; msa.alen];
    for col in 0..msa.alen {
        let mut n_residues = 0;
        for seq in 0..nseq {
            // ax[seq] is 1-based; column col maps to position col+1
            let pos = col + 1;
            if pos < ax[seq].len() - 1 {
                let residue = ax[seq][pos];
                if residue != gap && abc.is_canonical(residue) {
                    n_residues += 1;
                }
            }
        }
        let occupancy = n_residues as f32 / nseq as f32;
        matassign[col] = occupancy >= symfrac;
    }

    // If RF annotation exists, use it instead
    if let Some(ref rf) = msa.rf {
        for col in 0..msa.alen.min(rf.len()) {
            matassign[col] = rf[col] != b'.' && rf[col] != b'-' && rf[col] != b' ';
        }
    }

    let m = matassign.iter().filter(|&&x| x).count();
    if m == 0 {
        // No match columns — return a trivial HMM
        let mut hmm = Hmm::new(1, abc.abc_type, k);
        hmm.name = msa.name.clone();
        return hmm;
    }

    let mut hmm = Hmm::new(m, abc.abc_type, k);
    hmm.name = msa.name.clone();
    hmm.nseq = nseq as i32;
    hmm.eff_nseq = nseq as f32;

    // Compute position-based sequence weights
    let weights = pb_weights(msa, abc);
    let eff_nseq: f32 = weights.iter().sum();
    hmm.eff_nseq = eff_nseq;

    // Step 2: Count weighted residues and transitions
    let pseudo = 1.0 / k as f32;
    for node in 0..=m {
        for x in 0..k {
            hmm.mat[node][x] = pseudo;
            hmm.ins[node][x] = pseudo;
        }
        for t in 0..NTRANSITIONS {
            hmm.t[node][t] = 0.01;
        }
    }

    // Count observed residues weighted by PB weights
    for seq in 0..nseq {
        let w = weights[seq];
        let mut node = 0;

        for col in 0..msa.alen {
            let pos = col + 1;
            if pos >= ax[seq].len() - 1 {
                break;
            }
            let residue = ax[seq][pos] as usize;

            if matassign[col] {
                // This column is a match state
                let prev_node = node;
                node += 1;

                if residue < k {
                    hmm.mat[node][residue] += w;
                    hmm.t[prev_node][MM] += w;
                } else if residue == gap as usize {
                    hmm.t[prev_node][MD] += w;
                }
            } else {
                // This column is an insert
                if residue < k && node > 0 {
                    hmm.ins[node][residue] += w;
                    hmm.t[node][MI] += w * 0.1;
                }
            }
        }
    }

    // Step 3: Normalize probabilities
    for node in 1..=m {
        // Match emissions
        let mat_sum: f32 = hmm.mat[node].iter().take(k).sum();
        if mat_sum > 0.0 {
            for x in 0..k {
                hmm.mat[node][x] /= mat_sum;
            }
        }

        // Insert emissions
        let ins_sum: f32 = hmm.ins[node].iter().take(k).sum();
        if ins_sum > 0.0 {
            for x in 0..k {
                hmm.ins[node][x] /= ins_sum;
            }
        }
    }

    // Node 0 insert emissions = background
    for x in 0..k {
        hmm.ins[0][x] = bg.f[x];
    }

    // Normalize transitions
    for node in 0..=m {
        // Match transitions: MM + MI + MD = 1
        let m_total = hmm.t[node][MM] + hmm.t[node][MI] + hmm.t[node][MD];
        if m_total > 0.0 {
            hmm.t[node][MM] /= m_total;
            hmm.t[node][MI] /= m_total;
            hmm.t[node][MD] /= m_total;
        } else {
            hmm.t[node][MM] = 1.0;
            hmm.t[node][MI] = 0.0;
            hmm.t[node][MD] = 0.0;
        }

        // Insert transitions: IM + II = 1
        let i_total = hmm.t[node][IM] + hmm.t[node][II];
        if i_total > 0.0 {
            hmm.t[node][IM] /= i_total;
            hmm.t[node][II] /= i_total;
        } else {
            hmm.t[node][IM] = 0.5;
            hmm.t[node][II] = 0.5;
        }

        // Delete transitions: DM + DD = 1
        let d_total = hmm.t[node][DM] + hmm.t[node][DD];
        if d_total > 0.0 {
            hmm.t[node][DM] /= d_total;
            hmm.t[node][DD] /= d_total;
        } else {
            hmm.t[node][DM] = 0.5;
            hmm.t[node][DD] = 0.5;
        }
    }

    // Last node: no transitions out
    hmm.t[m][MI] = 0.0;
    hmm.t[m][MD] = 0.0;
    hmm.t[m][MM] = 1.0;

    // Set composition
    for x in 0..k.min(MAXABET) {
        hmm.compo[x] = bg.f[x];
    }
    hmm.flags |= P7H_COMPO;

    // Set consensus
    let mut cons = vec![b' '; m + 2];
    let mut node = 0;
    for col in 0..msa.alen {
        if matassign[col] {
            node += 1;
            // Find highest-probability residue
            let mut best_x = 0;
            let mut best_p = 0.0;
            for x in 0..k {
                if hmm.mat[node][x] > best_p {
                    best_p = hmm.mat[node][x];
                    best_x = x;
                }
            }
            cons[node] = abc.sym[best_x];
        }
    }
    hmm.consensus = Some(cons);
    hmm.flags |= P7H_CONS;

    // Set RF from matassign
    if msa.rf.is_some() {
        let mut rf = vec![b' '; m + 2];
        let mut node = 0;
        for col in 0..msa.alen {
            if matassign[col] {
                node += 1;
                rf[node] = b'x';
            }
        }
        hmm.rf = Some(rf);
        hmm.flags |= P7H_RF;
    }

    // Apply Dirichlet priors to emission/transition counts
    crate::prior::apply_priors(&mut hmm);

    // Effective sequence number estimation
    crate::eweight::entropy_weight(&mut hmm, bg, None);

    // E-value calibration by simulation
    crate::calibrate::calibrate(&mut hmm, abc, bg);

    hmm
}
