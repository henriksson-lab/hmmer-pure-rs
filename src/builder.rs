//! HMM builder — construct profile HMMs from multiple sequence alignments.
//! Simplified port of p7_builder.c and build.c.

use crate::alphabet::{Alphabet, AlphabetType};
use crate::bg::Bg;
use crate::hmm::*;
use crate::msa::{self, Msa};
use crate::trace::{State as TraceState, Trace};

/// Henikoff position-based sequence weights, normalized to sum to `nseq`.
///
/// In default "fast" architecture mode, any input RF annotation is ignored;
/// `use_rf = true` (HMMER's `--hand` mode) restricts the column scan to RF
/// consensus columns. Counterpart to Easel's `esl_msaweight_PB()`.
pub fn pb_weights(msa: &Msa, abc: &Alphabet, use_rf: bool) -> Vec<f64> {
    let nseq = msa.nseq;
    let k = abc.k;

    let ax = msa.digitize(abc);
    let mut weights = vec![0.0_f64; nseq];

    for col in 0..msa.alen {
        if use_rf {
            let rf = msa.rf.as_ref().unwrap();
            if col >= rf.len() || rf[col] == b'.' || rf[col] == b'-' || rf[col] == b' ' {
                continue;
            }
        }

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
                    weights[seq] += 1.0 / (r * counts[residue]) as f64;
                }
            }
        }
    }

    // Normalize: each weight divided by number of residues in that sequence
    for seq in 0..nseq {
        let mut n_res = 0;
        for col in 0..msa.alen {
            if use_rf {
                let rf = msa.rf.as_ref().unwrap();
                if col >= rf.len() || rf[col] == b'.' || rf[col] == b'-' || rf[col] == b' ' {
                    continue;
                }
            }
            let pos = col + 1;
            if pos < ax[seq].len() - 1 {
                let residue = ax[seq][pos] as usize;
                if residue < k {
                    n_res += 1;
                }
            }
        }
        if n_res > 0 {
            weights[seq] /= n_res as f64;
        }
    }

    // Normalize weights to sum to nseq
    let sum: f64 = weights.iter().sum();
    if sum != 0.0 {
        let scale = nseq as f64 / sum;
        for w in &mut weights {
            *w *= scale;
        }
    } else if nseq > 0 {
        let uniform = 1.0 / nseq as f64;
        weights.fill(uniform);
        for w in &mut weights {
            *w *= nseq as f64;
        }
    }

    weights
}

/// Build a profile HMM from a multiple sequence alignment.
///
/// Pipeline: PB weights -> mark fragments -> assign match columns
/// (fast architecture, or `--hand` from RF) -> faux trace counts ->
/// effective Neff via entropy weighting -> Dirichlet priors ->
/// composition/consensus annotation -> E-value calibration.
/// Counterpart to C's `p7_Builder()` (with `build.c`'s `build_model()`).
pub fn build_hmm_from_msa(
    msa: &Msa,
    abc: &Alphabet,
    bg: &Bg,
    symfrac: f32,
    hand_arch: bool,
) -> Hmm {
    let k = abc.k;
    let nseq = msa.nseq;

    // Digitize the alignment, then mark fragment flanks as missing data
    // before architecture assignment and faux-trace counting, matching
    // esl_msa_MarkFragments_old() in the C builder flow.
    let mut ax = msa.digitize(abc);
    let gap = abc.gap_code();
    let missing = abc.missing_code();

    // Compute position-based sequence weights before architecture assignment.
    let weights = pb_weights(msa, abc, hand_arch && msa.rf.is_some());
    mark_fragments_old(&mut ax, abc, msa.alen, 0.5);

    // Step 1: Determine which columns are match states (Fast model construction)
    let mut matassign = vec![false; msa.alen];
    for col in 0..msa.alen {
        let mut residue_wt = 0.0_f32;
        let mut total_wt = 0.0_f32;
        for seq in 0..nseq {
            let pos = col + 1;
            if pos < ax[seq].len() - 1 {
                let residue = ax[seq][pos];
                if abc.is_residue(residue) {
                    residue_wt = (residue_wt as f64 + weights[seq]) as f32;
                    total_wt = (total_wt as f64 + weights[seq]) as f32;
                } else if residue == gap {
                    total_wt = (total_wt as f64 + weights[seq]) as f32;
                } else if residue == missing {
                    continue;
                }
            }
        }
        matassign[col] = residue_wt > 0.0 && total_wt > 0.0 && residue_wt / total_wt >= symfrac;
    }

    // Only `--hand` architecture uses input RF; fast architecture ignores it.
    if hand_arch {
        if let Some(ref rf) = msa.rf {
            for col in 0..msa.alen.min(rf.len()) {
                matassign[col] = rf[col] != b'.' && rf[col] != b'-' && rf[col] != b' ';
            }
        }
    }

    let m = matassign.iter().filter(|&&x| x).count();
    if m == 0 {
        // No match columns — return a trivial HMM
        let mut hmm = Hmm::new(1, abc.abc_type, k);
        hmm.name = msa.name.clone();
        if let Some(ref acc) = msa.acc {
            hmm.acc = Some(acc.clone());
            hmm.flags |= P7H_ACC;
        }
        if let Some(ref desc) = msa.desc {
            hmm.desc = Some(desc.clone());
            hmm.flags |= P7H_DESC;
        }
        return hmm;
    }

    let mut hmm = Hmm::new(m, abc.abc_type, k);
    hmm.name = msa.name.clone();
    if let Some(ref acc) = msa.acc {
        hmm.acc = Some(acc.clone());
        hmm.flags |= P7H_ACC;
    }
    if let Some(ref desc) = msa.desc {
        hmm.desc = Some(desc.clone());
        hmm.flags |= P7H_DESC;
    }
    hmm.nseq = nseq as i32;
    hmm.eff_nseq = nseq as f32;

    let eff_nseq: f32 = weights.iter().sum::<f64>() as f32;
    hmm.eff_nseq = eff_nseq;

    // Step 2: Count weighted residues and transitions
    for node in 0..=m {
        for x in 0..k {
            hmm.mat[node][x] = 0.0;
            hmm.ins[node][x] = 0.0;
        }
        hmm.t[node] = [0.0; NTRANSITIONS];
    }

    let traces = faux_trace_from_msa(&ax, &matassign, abc);

    // Count observed residues and transitions from doctored faux traces,
    // matching C's build.c -> p7_trace_FauxFromMSA() -> p7_trace_Doctor()
    // -> p7_trace_Count() flow.
    for seq in 0..nseq {
        let w = weights[seq] as f32;
        if traces[seq].n == 0 {
            continue;
        }
        count_trace(&mut hmm, &ax[seq], w, &traces[seq], abc);
    }

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

    // Store alignment column map for hmmalign --mapali.
    let mut map = vec![0i32; m + 1];
    let mut node = 0usize;
    for col in 0..msa.alen {
        if matassign[col] {
            node += 1;
            map[node] = (col + 1) as i32;
        }
    }
    hmm.map = Some(map);
    hmm.flags |= P7H_MAP;

    // Store the training alignment checksum for hmmalign --mapali verification.
    hmm.checksum = msa::checksum(msa, abc);
    hmm.flags |= P7H_CHKSUM;

    // Effective sequence number estimation acts on count HMMs, then the
    // resulting ratio rescales counts before final parameterization.
    let neff = crate::eweight::entropy_weight(&mut hmm, bg, None);
    let scale = if hmm.nseq > 0 {
        neff as f64 / hmm.nseq as f64
    } else {
        1.0
    };
    crate::eweight::scale_counts(&mut hmm, scale);

    // Apply Dirichlet priors to emission/transition counts
    crate::prior::apply_priors(&mut hmm);

    set_hmm_composition(&mut hmm);
    set_hmm_consensus(&mut hmm, abc);

    // E-value calibration by simulation
    crate::calibrate::calibrate(&mut hmm, abc, bg);
    if abc.abc_type != AlphabetType::Amino {
        hmm.max_length = (hmm.m * 4).max(1) as i32;
    }

    hmm
}

/// Compute the average residue composition implied by the HMM's match and
/// insert emissions, weighted by per-node occupancy, and store in `hmm.compo`.
/// Counterpart to C's `p7_hmm_SetComposition()`.
fn set_hmm_composition(hmm: &mut Hmm) {
    let mut mocc = vec![0.0_f32; hmm.m + 1];
    let mut iocc = vec![0.0_f32; hmm.m + 1];

    mocc[0] = 0.0;
    if hmm.m >= 1 {
        mocc[1] = hmm.t[0][MI] + hmm.t[0][MM];
    }
    for k in 2..=hmm.m {
        mocc[k] = mocc[k - 1] * (hmm.t[k - 1][MM] + hmm.t[k - 1][MI])
            + (1.0 - mocc[k - 1]) * hmm.t[k - 1][DM];
    }

    if hmm.t[0][IM] > 0.0 {
        iocc[0] = hmm.t[0][MI] / hmm.t[0][IM];
    }
    for k in 1..=hmm.m {
        if hmm.t[k][IM] > 0.0 {
            iocc[k] = mocc[k] * hmm.t[k][MI] / hmm.t[k][IM];
        }
    }

    hmm.compo.fill(0.0);
    for x in 0..hmm.abc_k {
        hmm.compo[x] += hmm.ins[0][x] * iocc[0];
    }
    for k in 1..=hmm.m {
        for x in 0..hmm.abc_k {
            hmm.compo[x] += hmm.mat[k][x] * mocc[k];
            hmm.compo[x] += hmm.ins[k][x] * iocc[k];
        }
    }

    let sum: f32 = hmm.compo[..hmm.abc_k].iter().sum();
    if sum > 0.0 {
        for x in 0..hmm.abc_k {
            hmm.compo[x] /= sum;
        }
    }
    hmm.flags |= P7H_COMPO;
}

/// Derive a consensus residue string from the most-probable match emission
/// per node. Uppercase if its probability >= threshold (0.9 for nucleic,
/// 0.5 for amino), otherwise lowercase. Counterpart to C's `p7_hmm_SetConsensus()`.
fn set_hmm_consensus(hmm: &mut Hmm, abc: &Alphabet) {
    let threshold = match hmm.abc_type {
        crate::alphabet::AlphabetType::Dna | crate::alphabet::AlphabetType::Rna => 0.9,
        _ => 0.5,
    };
    let mut cons = vec![b' '; hmm.m + 2];
    for (node, cons_slot) in cons.iter_mut().enumerate().take(hmm.m + 1).skip(1) {
        let mut best_x = 0usize;
        let mut best_p = f32::NEG_INFINITY;
        for x in 0..abc.k {
            if hmm.mat[node][x] > best_p {
                best_p = hmm.mat[node][x];
                best_x = x;
            }
        }
        let symbol = abc.sym[best_x];
        *cons_slot = if best_p >= threshold {
            symbol.to_ascii_uppercase()
        } else {
            symbol.to_ascii_lowercase()
        };
    }
    hmm.consensus = Some(cons);
    hmm.flags |= P7H_CONS;
}

/// Generate one faux traceback per MSA row consistent with the `matassign[]`
/// architecture, then run trace doctoring to remove D->I / I->D conflicts.
/// Counterpart to C's `p7_trace_FauxFromMSA()` + `p7_trace_Doctor()`.
fn faux_trace_from_msa(ax: &[Vec<u8>], matassign: &[bool], abc: &Alphabet) -> Vec<Trace> {
    let mut traces = Vec::with_capacity(ax.len());
    for row in ax {
        let mut tr = Trace::new();
        tr.append(TraceState::B, 0, 0);

        let mut k = 0usize;
        for apos in 1..=matassign.len() {
            let sym = row[apos];
            if matassign[apos - 1] {
                k += 1;
                if abc.is_residue(sym) || sym == abc.nonresidue_code() {
                    tr.append(TraceState::M, k, apos);
                } else if abc.is_gap(sym) {
                    tr.append(TraceState::D, k, 0);
                } else if abc.is_missing(sym) && tr.st.last().copied() != Some(TraceState::X) {
                    tr.append(TraceState::X, k, 0);
                }
            } else if abc.is_residue(sym) || sym == abc.nonresidue_code() {
                tr.append(TraceState::I, k, apos);
            } else if abc.is_missing(sym) && tr.st.last().copied() != Some(TraceState::X) {
                tr.append(TraceState::X, k, 0);
            }
        }
        tr.append(TraceState::E, 0, 0);
        tr.m = k;
        tr.l = matassign.len();
        doctor_trace(&mut tr);
        traces.push(tr);
    }
    traces
}

/// Mark short fragment sequences' leading/trailing gaps as missing-data
/// symbols so they don't contribute spurious counts at flanking positions.
/// Counterpart to Easel's `esl_msa_MarkFragments_old()`.
fn mark_fragments_old(ax: &mut [Vec<u8>], abc: &Alphabet, alen: usize, fragthresh: f32) {
    let missing = abc.missing_code();
    for row in ax {
        let rlen = row
            .iter()
            .skip(1)
            .take(alen)
            .filter(|&&sym| abc.is_residue(sym))
            .count();
        if (rlen as f32) <= fragthresh * alen as f32 {
            for pos in 1..=alen {
                if abc.is_residue(row[pos]) {
                    break;
                }
                row[pos] = missing;
            }
            for pos in (1..=alen).rev() {
                if abc.is_residue(row[pos]) {
                    break;
                }
                row[pos] = missing;
            }
        }
    }
}

/// Collapse adjacent D-I / I-D pairs in a faux trace into single M states,
/// matching the HMMER architecture's disallowed D<->I transitions.
/// Counterpart to C's `p7_trace_Doctor()`.
fn doctor_trace(tr: &mut Trace) {
    let mut new_st = Vec::with_capacity(tr.n);
    let mut new_k = Vec::with_capacity(tr.n);
    let mut new_i = Vec::with_capacity(tr.n);
    let mut opos = 0usize;

    while opos < tr.n {
        if opos + 1 < tr.n && tr.st[opos] == TraceState::D && tr.st[opos + 1] == TraceState::I {
            new_st.push(TraceState::M);
            new_k.push(tr.k[opos]);
            new_i.push(tr.i[opos + 1]);
            opos += 2;
        } else if opos + 1 < tr.n
            && tr.st[opos] == TraceState::I
            && tr.st[opos + 1] == TraceState::D
        {
            new_st.push(TraceState::M);
            new_k.push(tr.k[opos + 1]);
            new_i.push(tr.i[opos]);
            opos += 2;
        } else {
            new_st.push(tr.st[opos]);
            new_k.push(tr.k[opos]);
            new_i.push(tr.i[opos]);
            opos += 1;
        }
    }

    tr.st = new_st;
    tr.k = new_k;
    tr.i = new_i;
    tr.n = tr.st.len();
}

/// Accumulate weight `wt` for symbol `sym` into count vector `ct[0..K]`.
/// Distributes degenerate residues evenly over their canonical members.
/// Counterpart to Easel's `esl_abc_FCount()`.
fn fcount(abc: &Alphabet, ct: &mut [f32], sym: u8, wt: f32) {
    if abc.is_canonical(sym) || abc.is_gap(sym) {
        if let Some(slot) = ct.get_mut(sym as usize) {
            *slot += wt;
        }
    } else if abc.is_missing(sym) || sym == abc.nonresidue_code() {
    } else if abc.is_degenerate(sym) {
        let denom = abc.ndegen[sym as usize] as f32;
        if denom > 0.0 {
            for y in 0..abc.k {
                if abc.degen[sym as usize][y] {
                    ct[y] += wt / denom;
                }
            }
        }
    }
}

/// Accumulate weighted emission and transition counts from one trace into
/// `hmm`. Skips X (missing data) regions at the trace ends. Counterpart to
/// C's `p7_trace_Count()`.
fn count_trace(hmm: &mut Hmm, dsq: &[u8], wt: f32, tr: &Trace, abc: &Alphabet) {
    let mut z1 = 0usize;
    let mut z2 = tr.n.saturating_sub(1);

    if tr.n >= 2 && tr.st[0] == TraceState::B && tr.st[1] == TraceState::X {
        for z in 2..tr.n.saturating_sub(1) {
            if tr.st[z] == TraceState::M {
                z1 = z;
                break;
            }
        }
    }
    if tr.n >= 2 && tr.st[tr.n - 1] == TraceState::E && tr.st[tr.n - 2] == TraceState::X {
        for z in (1..=tr.n.saturating_sub(3)).rev() {
            if tr.st[z] == TraceState::M {
                z2 = z;
                break;
            }
        }
    }

    for z in z1..z2 {
        if tr.st[z] == TraceState::X {
            continue;
        }

        let st = tr.st[z];
        let st2 = tr.st[z + 1];
        let k = tr.k[z];
        let k2 = tr.k[z + 1];
        let i = tr.i[z];

        if st == TraceState::M {
            fcount(abc, &mut hmm.mat[k], dsq[i], wt);
        } else if st == TraceState::I {
            fcount(abc, &mut hmm.ins[k], dsq[i], wt);
        }

        if st2 == TraceState::X {
            continue;
        }

        if st == TraceState::B {
            if st2 == TraceState::M && k2 > 1 {
                hmm.t[0][MD] += wt;
                for ktmp in 1..k2.saturating_sub(1) {
                    hmm.t[ktmp][DD] += wt;
                }
                hmm.t[k2 - 1][DM] += wt;
            } else {
                match st2 {
                    TraceState::M => hmm.t[0][MM] += wt,
                    TraceState::I => hmm.t[0][MI] += wt,
                    TraceState::D => hmm.t[0][MD] += wt,
                    _ => {}
                }
            }
        } else if st == TraceState::M {
            match st2 {
                TraceState::M => hmm.t[k][MM] += wt,
                TraceState::I => hmm.t[k][MI] += wt,
                TraceState::D => hmm.t[k][MD] += wt,
                TraceState::E => hmm.t[k][MM] += wt,
                _ => {}
            }
        } else if st == TraceState::I {
            match st2 {
                TraceState::M => hmm.t[k][IM] += wt,
                TraceState::I => hmm.t[k][II] += wt,
                TraceState::E => hmm.t[k][IM] += wt,
                _ => {}
            }
        } else if st == TraceState::D {
            match st2 {
                TraceState::M => hmm.t[k][DM] += wt,
                TraceState::D => hmm.t[k][DD] += wt,
                TraceState::E => hmm.t[k][DM] += wt,
                _ => {}
            }
        }
    }
}
