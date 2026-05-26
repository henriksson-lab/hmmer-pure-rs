//! HMM builder — construct profile HMMs from multiple sequence alignments.
//! Simplified port of p7_builder.c and build.c.

use crate::alphabet::{Alphabet, AlphabetType};
use crate::bg::Bg;
use crate::calibrate::CalibrationConfig;
use crate::hmm::*;
use crate::msa::{self, Msa};
use crate::prior::PriorStrategy;
use crate::trace::{State as TraceState, Trace};

pub const DEFAULT_WINDOW_BETA: f64 = 1e-7;

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

fn normalize_weights_to_nseq(weights: &mut [f64], nseq: usize) {
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        let scale = nseq as f64 / sum;
        for weight in weights {
            *weight *= scale;
        }
    } else if nseq > 0 {
        weights.fill(1.0);
    }
}

/// Use MSA-supplied sequence weights exactly as parsed, or 1.0 defaults.
pub fn given_weights(msa: &Msa) -> Vec<f64> {
    msa.weights.clone().unwrap_or_else(|| vec![1.0; msa.nseq])
}

/// BLOSUM relative weights: single-linkage cluster at pairwise identity
/// `max_id`, then divide one unit of weight over each cluster.
pub fn blosum_weights(msa: &Msa, abc: &Alphabet, max_id: f64) -> Vec<f64> {
    let nseq = msa.nseq;
    if nseq <= 1 {
        return vec![1.0; nseq];
    }
    let ax = msa.digitize(abc);
    let mut dsu = DisjointSet::new(nseq);
    for i in 0..nseq {
        for j in i + 1..nseq {
            if pair_identity(abc, &ax[i], &ax[j]) >= max_id {
                dsu.union(i, j);
            }
        }
    }

    let mut cluster_id = vec![usize::MAX; nseq];
    let mut cluster_sizes = Vec::<usize>::new();
    for seq in 0..nseq {
        let root = dsu.find(seq);
        if cluster_id[root] == usize::MAX {
            cluster_id[root] = cluster_sizes.len();
            cluster_sizes.push(0);
        }
        cluster_sizes[cluster_id[root]] += 1;
    }

    let mut weights = vec![0.0; nseq];
    for (seq, weight) in weights.iter_mut().enumerate() {
        let cluster = cluster_id[dsu.find(seq)];
        *weight = 1.0 / cluster_sizes[cluster] as f64;
    }
    normalize_weights_to_nseq(&mut weights, nseq);
    weights
}

/// Number of single-linkage clusters at pairwise identity `max_id`.
///
/// This is the effective-sequence-number calculation used by hmmbuild
/// `--eclust`: any pair with identity >= cutoff links their clusters.
pub fn single_linkage_cluster_count(msa: &Msa, abc: &Alphabet, max_id: f64) -> usize {
    let nseq = msa.nseq;
    if nseq <= 1 {
        return nseq;
    }
    let ax = msa.digitize(abc);
    let mut dsu = DisjointSet::new(nseq);
    for i in 0..nseq {
        for j in i + 1..nseq {
            if pair_identity(abc, &ax[i], &ax[j]) >= max_id {
                dsu.union(i, j);
            }
        }
    }

    let mut nclusters = 0usize;
    for seq in 0..nseq {
        if dsu.find(seq) == seq {
            nclusters += 1;
        }
    }
    nclusters
}

/// Gerstein/Sonnhammer/Chothia tree weights, using UPGMA over fractional
/// pairwise differences, normalized to sum to `nseq`.
pub fn gsc_weights(msa: &Msa, abc: &Alphabet) -> Vec<f64> {
    let nseq = msa.nseq;
    if nseq <= 1 {
        return vec![1.0; nseq];
    }
    let ax = msa.digitize(abc);
    let mut distances = vec![vec![0.0_f64; nseq]; nseq];
    for i in 0..nseq {
        for j in i + 1..nseq {
            let distance = 1.0 - pair_identity(abc, &ax[i], &ax[j]);
            distances[i][j] = distance;
            distances[j][i] = distance;
        }
    }

    let tree = upgma_tree(distances);
    let mut clade_size = vec![0usize; nseq - 1];
    for node in (0..nseq - 1).rev() {
        clade_size[node] = child_clade_size(tree.left[node], &clade_size)
            + child_clade_size(tree.right[node], &clade_size);
    }

    let mut x = vec![0.0_f64; nseq - 1];
    for node in (0..nseq - 1).rev() {
        x[node] = tree.ld[node] + tree.rd[node];
        if tree.left[node] > 0 {
            x[node] += x[tree.left[node] as usize];
        }
        if tree.right[node] > 0 {
            x[node] += x[tree.right[node] as usize];
        }
    }

    let mut weights = vec![0.0_f64; nseq];
    x[0] = 0.0;
    for node in 0..nseq - 1 {
        let mut lw = tree.ld[node];
        if tree.left[node] > 0 {
            lw += x[tree.left[node] as usize];
        }
        let mut rw = tree.rd[node];
        if tree.right[node] > 0 {
            rw += x[tree.right[node] as usize];
        }

        let (lx, rx) = if lw + rw == 0.0 {
            let left_share =
                child_clade_size(tree.left[node], &clade_size) as f64 / clade_size[node] as f64;
            let right_share =
                child_clade_size(tree.right[node], &clade_size) as f64 / clade_size[node] as f64;
            (x[node] * left_share, x[node] * right_share)
        } else {
            (x[node] * lw / (lw + rw), x[node] * rw / (lw + rw))
        };

        assign_gsc_child_weight(tree.left[node], lx + tree.ld[node], &mut x, &mut weights);
        assign_gsc_child_weight(tree.right[node], rx + tree.rd[node], &mut x, &mut weights);
    }
    normalize_weights_to_nseq(&mut weights, nseq);
    weights
}

fn pair_identity(abc: &Alphabet, ax1: &[u8], ax2: &[u8]) -> f64 {
    let mut nid = 0usize;
    let mut len1 = 0usize;
    let mut len2 = 0usize;
    let mut pos = 1usize;
    while pos < ax1.len()
        && pos < ax2.len()
        && ax1[pos] != crate::alphabet::DSQ_SENTINEL
        && ax2[pos] != crate::alphabet::DSQ_SENTINEL
    {
        if abc.is_residue(ax1[pos]) {
            len1 += 1;
        }
        if abc.is_residue(ax2[pos]) {
            len2 += 1;
        }
        if abc.is_residue(ax1[pos]) && abc.is_residue(ax2[pos]) && ax1[pos] == ax2[pos] {
            nid += 1;
        }
        pos += 1;
    }
    let denom = len1.min(len2);
    if denom == 0 {
        0.0
    } else {
        nid as f64 / denom as f64
    }
}

#[derive(Clone)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        if self.rank[root_a] < self.rank[root_b] {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if self.rank[root_a] == self.rank[root_b] {
            self.rank[root_a] += 1;
        }
    }
}

struct UpgmaTree {
    left: Vec<i32>,
    right: Vec<i32>,
    ld: Vec<f64>,
    rd: Vec<f64>,
}

fn upgma_tree(mut distances: Vec<Vec<f64>>) -> UpgmaTree {
    let ntaxa = distances.len();
    let mut idx: Vec<i32> = (0..ntaxa).map(|i| -(i as i32)).collect();
    let mut nin = vec![1usize; ntaxa];
    let mut height = vec![0.0_f64; ntaxa - 1];
    let mut tree = UpgmaTree {
        left: vec![0; ntaxa - 1],
        right: vec![0; ntaxa - 1],
        ld: vec![0.0; ntaxa - 1],
        rd: vec![0.0; ntaxa - 1],
    };

    for n_active in (2..=ntaxa).rev() {
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        let mut min_distance = distances[0][1];
        for (row, row_vals) in distances.iter().enumerate().take(n_active) {
            for (col, &distance) in row_vals.iter().enumerate().take(n_active).skip(row + 1) {
                if distance < min_distance {
                    min_distance = distance;
                    best_i = row;
                    best_j = col;
                }
            }
        }

        let node = n_active - 2;
        tree.left[node] = idx[best_i];
        tree.right[node] = idx[best_j];
        height[node] = min_distance / 2.0;
        tree.ld[node] = height[node];
        tree.rd[node] = height[node];
        if idx[best_i] > 0 {
            tree.ld[node] = (tree.ld[node] - height[idx[best_i] as usize]).max(0.0);
        }
        if idx[best_j] > 0 {
            tree.rd[node] = (tree.rd[node] - height[idx[best_j] as usize]).max(0.0);
        }

        if best_j != n_active - 1 {
            swap_distance_index(&mut distances, best_j, n_active - 1, n_active);
            idx.swap(best_j, n_active - 1);
            nin.swap(best_j, n_active - 1);
        }
        if best_i != n_active - 2 {
            swap_distance_index(&mut distances, best_i, n_active - 2, n_active);
            idx.swap(best_i, n_active - 2);
            nin.swap(best_i, n_active - 2);
        }

        let i = n_active - 2;
        let j = n_active - 1;
        for col in 0..n_active {
            distances[i][col] = (nin[i] as f64 * distances[i][col]
                + nin[j] as f64 * distances[j][col])
                / (nin[i] + nin[j]) as f64;
            distances[col][i] = distances[i][col];
        }
        nin[i] += nin[j];
        idx[i] = node as i32;
    }

    tree
}

fn swap_distance_index(distances: &mut [Vec<f64>], a: usize, b: usize, n_active: usize) {
    for row in distances.iter_mut().take(n_active) {
        row.swap(a, b);
    }
    distances.swap(a, b);
}

fn child_clade_size(child: i32, clade_size: &[usize]) -> usize {
    if child > 0 {
        clade_size[child as usize]
    } else {
        1
    }
}

fn assign_gsc_child_weight(child: i32, value: f64, x: &mut [f64], weights: &mut [f64]) {
    if child > 0 {
        x[child as usize] = value;
    } else {
        weights[(-child) as usize] = value;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectiveSeqNumber {
    Entropy {
        target_re: Option<f64>,
        target_sigma: Option<f64>,
    },
    EntropyExp {
        target_re: Option<f64>,
        target_sigma: Option<f64>,
    },
    Cluster {
        identity_cutoff: f64,
    },
    None,
    Set(f32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RelativeWeighting {
    PositionBased,
    Gsc,
    Blosum { identity_cutoff: f64 },
    Given,
    None,
}

/// Derive the builder's alignment-column architecture assignment.
///
/// Returns one byte per alignment column: `x` for match columns and `.` for
/// insert columns. This is the same fragment-marking, weighting, and `--hand`
/// logic used by `build_hmm_from_msa_with_prior()`, exposed so `hmmbuild -O`
/// can write the post-processed MSA that corresponds to the built model.
pub fn model_mask_from_msa(
    msa: &Msa,
    abc: &Alphabet,
    symfrac: f32,
    fragthresh: f32,
    hand_arch: bool,
    weighting_strategy: RelativeWeighting,
) -> Vec<u8> {
    let mut ax = msa.digitize(abc);
    let weights = match weighting_strategy {
        RelativeWeighting::PositionBased => pb_weights(msa, abc, hand_arch && msa.rf.is_some()),
        RelativeWeighting::Gsc => gsc_weights(msa, abc),
        RelativeWeighting::Blosum { identity_cutoff } => blosum_weights(msa, abc, identity_cutoff),
        RelativeWeighting::Given => given_weights(msa),
        RelativeWeighting::None => vec![1.0; msa.nseq],
    };
    mark_fragments_old(&mut ax, abc, msa.alen, fragthresh);
    model_mask_from_digitized(msa, abc, &ax, &weights, symfrac, hand_arch)
}

/// Build a profile HMM from a multiple sequence alignment.
///
/// Pipeline: PB weights -> mark fragments -> assign match columns
/// (fast architecture, or `--hand` from RF) -> faux trace counts ->
/// effective Neff estimation -> Dirichlet priors ->
/// composition/consensus annotation -> E-value calibration.
/// Counterpart to C's `p7_Builder()` (with `build.c`'s `build_model()`).
pub fn build_hmm_from_msa(
    msa: &Msa,
    abc: &Alphabet,
    bg: &Bg,
    symfrac: f32,
    fragthresh: f32,
    hand_arch: bool,
    weighting_strategy: RelativeWeighting,
    effn_strategy: EffectiveSeqNumber,
    seed: u32,
) -> Hmm {
    build_hmm_from_msa_with_prior(
        msa,
        abc,
        bg,
        symfrac,
        fragthresh,
        hand_arch,
        weighting_strategy,
        effn_strategy,
        PriorStrategy::Default,
        CalibrationConfig::default(),
        seed,
    )
}

pub fn build_hmm_from_msa_with_prior(
    msa: &Msa,
    abc: &Alphabet,
    bg: &Bg,
    symfrac: f32,
    fragthresh: f32,
    hand_arch: bool,
    weighting_strategy: RelativeWeighting,
    effn_strategy: EffectiveSeqNumber,
    prior_strategy: PriorStrategy,
    calibration_config: CalibrationConfig,
    seed: u32,
) -> Hmm {
    build_hmm_from_msa_with_prior_and_max_insert(
        msa,
        abc,
        bg,
        symfrac,
        fragthresh,
        hand_arch,
        weighting_strategy,
        effn_strategy,
        prior_strategy,
        calibration_config,
        seed,
        None,
    )
}

pub fn build_hmm_from_msa_with_prior_and_max_insert(
    msa: &Msa,
    abc: &Alphabet,
    bg: &Bg,
    symfrac: f32,
    fragthresh: f32,
    hand_arch: bool,
    weighting_strategy: RelativeWeighting,
    effn_strategy: EffectiveSeqNumber,
    prior_strategy: PriorStrategy,
    calibration_config: CalibrationConfig,
    seed: u32,
    max_insert_len: Option<usize>,
) -> Hmm {
    let k = abc.k;
    let nseq = msa.nseq;

    // Digitize the alignment, then mark fragment flanks as missing data
    // before architecture assignment and faux-trace counting, matching
    // esl_msa_MarkFragments_old() in the C builder flow.
    let mut ax = msa.digitize(abc);
    let weights = match weighting_strategy {
        RelativeWeighting::PositionBased => pb_weights(msa, abc, hand_arch && msa.rf.is_some()),
        RelativeWeighting::Gsc => gsc_weights(msa, abc),
        RelativeWeighting::Blosum { identity_cutoff } => blosum_weights(msa, abc, identity_cutoff),
        RelativeWeighting::Given => given_weights(msa),
        RelativeWeighting::None => vec![1.0; nseq],
    };
    mark_fragments_old(&mut ax, abc, msa.alen, fragthresh);

    let matassign: Vec<bool> =
        model_mask_from_digitized(msa, abc, &ax, &weights, symfrac, hand_arch)
            .into_iter()
            .map(|sym| sym == b'x')
            .collect();

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

    if let Some(max_insert_len) = max_insert_len {
        clamp_insert_self_transition_counts(&mut hmm, max_insert_len);
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
    let uniform_neff = match effn_strategy {
        EffectiveSeqNumber::Entropy {
            target_re,
            target_sigma,
        } => Some(crate::eweight::entropy_weight(
            &mut hmm,
            bg,
            target_re,
            target_sigma,
        )),
        EffectiveSeqNumber::EntropyExp {
            target_re,
            target_sigma,
        } => {
            crate::eweight::entropy_weight_exp(&mut hmm, bg, target_re, target_sigma);
            None
        }
        EffectiveSeqNumber::Cluster { identity_cutoff } => {
            let nclusters = single_linkage_cluster_count(msa, abc, identity_cutoff) as f32;
            hmm.eff_nseq = nclusters;
            Some(nclusters)
        }
        EffectiveSeqNumber::None => {
            hmm.eff_nseq = hmm.nseq as f32;
            Some(hmm.nseq as f32)
        }
        EffectiveSeqNumber::Set(value) => {
            hmm.eff_nseq = value;
            Some(value)
        }
    };
    if let Some(neff) = uniform_neff {
        let scale = if hmm.nseq > 0 {
            neff as f64 / hmm.nseq as f64
        } else {
            1.0
        };
        crate::eweight::scale_counts(&mut hmm, scale);
    }

    // Parameterize emission/transition counts with the selected prior strategy.
    crate::prior::apply_priors_with_strategy(&mut hmm, prior_strategy);

    set_hmm_composition(&mut hmm);
    set_hmm_consensus(&mut hmm, abc);

    // E-value calibration by simulation
    crate::calibrate::calibrate_with_config(&mut hmm, abc, bg, seed, calibration_config);
    if abc.abc_type != AlphabetType::Amino {
        set_max_length_from_beta(&mut hmm, DEFAULT_WINDOW_BETA);
    }

    hmm
}

pub fn copy_stockholm_cutoffs_to_hmm(cutoffs: msa::StockholmCutoffs, hmm: &mut Hmm) {
    if let Some([seq_cutoff, dom_cutoff]) = cutoffs.ga {
        hmm.cutoff[P7_GA1] = seq_cutoff;
        hmm.cutoff[P7_GA2] = dom_cutoff;
        hmm.flags |= P7H_GA;
    }
    if let Some([seq_cutoff, dom_cutoff]) = cutoffs.tc {
        hmm.cutoff[P7_TC1] = seq_cutoff;
        hmm.cutoff[P7_TC2] = dom_cutoff;
        hmm.flags |= P7H_TC;
    }
    if let Some([seq_cutoff, dom_cutoff]) = cutoffs.nc {
        hmm.cutoff[P7_NC1] = seq_cutoff;
        hmm.cutoff[P7_NC2] = dom_cutoff;
        hmm.flags |= P7H_NC;
    }
}

fn clamp_insert_self_transition_counts(hmm: &mut Hmm, max_insert_len: usize) {
    let max_insert_len = max_insert_len as f32;
    for node in 1..hmm.m {
        let max_ii = max_insert_len * hmm.t[node][MI];
        if hmm.t[node][II] > max_ii {
            hmm.t[node][II] = max_ii;
        }
    }
}

/// Compute HMMER's beta-derived maximum likely emitted sequence length.
///
/// This is the Rust counterpart of C `p7_Builder_MaxLength()`: it accumulates
/// the emitted-length distribution for the core model until the remaining tail
/// mass is below `emit_thresh`, bounded by `min(20*M, 100000)` and at least `M`.
pub fn max_length_from_beta(hmm: &Hmm, emit_thresh: f64) -> i32 {
    let model_len = hmm.m;
    if model_len <= 1 {
        return 1;
    }

    let length_bound = model_len.max((20 * model_len).min(100_000));
    let mut max_length = length_bound as i32;

    let mut i_dp = vec![[0.0_f64; 2]; model_len + 1];
    let mut m_dp = vec![[0.0_f64; 2]; model_len + 1];
    let mut d_dp = vec![[0.0_f64; 2]; model_len + 1];

    m_dp[1][0] = 1.0;
    i_dp[1][0] = 0.0;
    d_dp[1][0] = 0.0;
    m_dp[2][0] = 0.0;
    i_dp[2][0] = 0.0;
    d_dp[2][0] = hmm.t[1][MD] as f64;
    for k in 3..=model_len {
        m_dp[k][0] = 0.0;
        i_dp[k][0] = 0.0;
        d_dp[k][0] = hmm.t[k - 1][DD] as f64 * d_dp[k - 1][0];
    }

    m_dp[1][1] = 0.0;
    d_dp[1][1] = 0.0;
    d_dp[2][1] = 0.0;
    i_dp[2][1] = 0.0;
    i_dp[1][1] = hmm.t[1][MI] as f64 * m_dp[1][0];
    m_dp[2][1] = hmm.t[1][MM] as f64 * m_dp[1][0];
    for k in 3..=model_len {
        m_dp[k][1] = hmm.t[k - 1][DM] as f64 * d_dp[k - 1][0];
        i_dp[k][1] = 0.0;
        d_dp[k][1] =
            hmm.t[k - 1][MD] as f64 * m_dp[k - 1][1] + hmm.t[k - 1][DD] as f64 * d_dp[k - 1][1];
    }

    let mut p_sum =
        m_dp[model_len][0] + m_dp[model_len][1] + d_dp[model_len][0] + d_dp[model_len][1];

    let mut col_ptr = 0usize;
    for col in 3..=length_bound {
        let prev_col_ptr = 1 - col_ptr;
        let mut surv = 0.0_f64;

        m_dp[1][col_ptr] = 0.0;
        d_dp[1][col_ptr] = 0.0;
        i_dp[1][col_ptr] = hmm.t[1][II] as f64 * i_dp[1][prev_col_ptr];
        surv += i_dp[1][col_ptr];

        for k in 2..=model_len {
            m_dp[k][col_ptr] = hmm.t[k - 1][MM] as f64 * m_dp[k - 1][prev_col_ptr]
                + hmm.t[k - 1][DM] as f64 * d_dp[k - 1][prev_col_ptr]
                + hmm.t[k - 1][IM] as f64 * i_dp[k - 1][prev_col_ptr];
            i_dp[k][col_ptr] = hmm.t[k][MI] as f64 * m_dp[k][prev_col_ptr]
                + hmm.t[k][II] as f64 * i_dp[k][prev_col_ptr];
            d_dp[k][col_ptr] = hmm.t[k - 1][MD] as f64 * m_dp[k - 1][col_ptr]
                + hmm.t[k - 1][DD] as f64 * d_dp[k - 1][col_ptr];

            surv += i_dp[k][col_ptr]
                + m_dp[k][col_ptr] * (1.0 - hmm.t[k][MD] as f64)
                + d_dp[k][col_ptr] * (1.0 - hmm.t[k][DD] as f64);
        }
        surv += m_dp[model_len][col_ptr] * hmm.t[model_len][MD] as f64
            + d_dp[model_len][col_ptr] * hmm.t[model_len][DD] as f64
            - i_dp[model_len][col_ptr];

        p_sum += m_dp[model_len][col_ptr] + d_dp[model_len][col_ptr];
        let denom = surv + p_sum;
        if denom > 0.0 {
            surv /= denom;
        }

        if surv < emit_thresh {
            max_length = col as i32;
            break;
        }

        col_ptr = 1 - col_ptr;
    }

    max_length
}

pub fn set_max_length_from_beta(hmm: &mut Hmm, emit_thresh: f64) {
    hmm.max_length = max_length_from_beta(hmm, emit_thresh);
}

fn model_mask_from_digitized(
    msa: &Msa,
    abc: &Alphabet,
    ax: &[Vec<u8>],
    weights: &[f64],
    symfrac: f32,
    hand_arch: bool,
) -> Vec<u8> {
    let gap = abc.gap_code();
    let missing = abc.missing_code();
    let mut mask = vec![b'.'; msa.alen];

    for col in 0..msa.alen {
        let mut residue_wt = 0.0_f32;
        let mut total_wt = 0.0_f32;
        for seq in 0..msa.nseq {
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
        if residue_wt > 0.0 && total_wt > 0.0 && residue_wt / total_wt >= symfrac {
            mask[col] = b'x';
        }
    }

    // Only `--hand` architecture uses input RF; fast architecture ignores it.
    if hand_arch {
        if let Some(ref rf) = msa.rf {
            for col in 0..msa.alen.min(rf.len()) {
                mask[col] = if rf[col] != b'.' && rf[col] != b'-' && rf[col] != b' ' {
                    b'x'
                } else {
                    b'.'
                };
            }
        }
    }

    mask
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_stockholm_cutoffs_to_hmm_flags() {
        let cutoffs = crate::msa::StockholmCutoffs {
            ga: Some([25.0, 24.0]),
            tc: Some([30.0, 29.0]),
            nc: Some([-1.0, -2.0]),
        };
        let mut hmm = Hmm::new(1, AlphabetType::Amino, 20);
        copy_stockholm_cutoffs_to_hmm(cutoffs, &mut hmm);

        assert_eq!(hmm.cutoff[P7_GA1], 25.0);
        assert_eq!(hmm.cutoff[P7_GA2], 24.0);
        assert_eq!(hmm.cutoff[P7_TC1], 30.0);
        assert_eq!(hmm.cutoff[P7_TC2], 29.0);
        assert_eq!(hmm.cutoff[P7_NC1], -1.0);
        assert_eq!(hmm.cutoff[P7_NC2], -2.0);
        assert!(hmm.flags & P7H_GA != 0);
        assert!(hmm.flags & P7H_TC != 0);
        assert!(hmm.flags & P7H_NC != 0);
    }

    #[test]
    fn max_length_from_beta_tracks_tail_mass() {
        let hmm =
            crate::hmmfile::read_hmm_file_auto(std::path::Path::new("hmmer/tutorial/MADE1.hmm"))
                .unwrap()
                .remove(0);

        let loose = max_length_from_beta(&hmm, 0.5);
        let default = max_length_from_beta(&hmm, DEFAULT_WINDOW_BETA);

        assert!(loose > 0);
        assert!(default > 0);
        assert!(loose < default);
    }
}
