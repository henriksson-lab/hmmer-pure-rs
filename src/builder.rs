//! HMM builder — construct profile HMMs from multiple sequence alignments.
//! Simplified port of p7_builder.c and build.c.

use crate::alphabet::{Alphabet, AlphabetType};
use crate::bg::Bg;
use crate::calibrate::CalibrationConfig;
use crate::hmm::*;
use crate::msa::{self, Msa};
use crate::prior::PriorStrategy;
use crate::trace::{State as TraceState, Trace};
use crate::util::random::{esl_rand64_deal, Rand64};

pub const DEFAULT_WINDOW_BETA: f64 = 1e-7;

/// Default PB-weighting config values, mirroring `ESL_MSAWEIGHT_CFG`
/// (esl_msaweight.h:30-37). `ignore_rf` is supplied per-call because the
/// HMMER builder flips it with `--hand` (p7_builder.c:814); the rest are the
/// fixed Easel defaults.
const PB_FRAGTHRESH: f32 = 0.5;
const PB_SYMFRAC: f32 = 0.5;
/// `eslMSAWEIGHT_ALLOW_SAMP` (TRUE): permit the deep-alignment subsampling
/// consensus path. (esl_msaweight.h:33)
const PB_ALLOW_SAMP: bool = true;
/// `eslMSAWEIGHT_SAMPTHRESH` (50000): if `nseq > sampthresh`, determine
/// consensus from a subsample rather than all sequences. (esl_msaweight.h:34)
const PB_SAMPTHRESH: usize = 50000;
/// `eslMSAWEIGHT_NSAMP` (10000): number of sequences in the subsample.
/// (esl_msaweight.h:35)
const PB_NSAMP: usize = 10000;
/// `eslMSAWEIGHT_MAXFRAG` (5000): if the subsample contains more than this
/// many fragments, reject sampling and fall back to all sequences.
/// (esl_msaweight.h:36)
const PB_MAXFRAG: usize = 5000;
/// `eslMSAWEIGHT_RNGSEED` (42): fixed RNG seed for reproducible subsampling.
/// (esl_msaweight.h:37)
const PB_RNGSEED: u64 = 42;

/// Henikoff position-based sequence weights, normalized to sum to `nseq`.
///
/// Faithful port of Easel's `esl_msaweight_PB_adv()` (esl_msaweight.c:182) as
/// invoked by the HMMER builder (`relative_weights`, p7_builder.c:809) with the
/// default config. Consensus columns are determined by RF annotation when
/// present and `ignore_rf` is false (i.e. `--hand`); otherwise by the symfrac
/// gap-fraction rule over fragment-aware counts (`consensus_by_all`). Counts
/// are collected with HMMER's fragment-span rule so fragments' external gaps
/// are excluded. The per-residue PB rule `1/(r·ct)`, per-sequence `/rlen`
/// normalization, and final scale-to-`nseq` all match the C exactly.
///
/// The deep-alignment subsampling path (`consensus_by_sample`, nseq > 50000) is
/// reproduced bit-faithfully: it samples `nsamp` sequence indices with Easel's
/// 64-bit Mersenne Twister (`esl_rand64`) + Vitter sequential sampling
/// (`esl_rand64_Deal`), marks fragments by the same span rule, and determines
/// consensus columns from the sample (or rejects when too many fragments are
/// seen, falling back to `consensus_by_all`).
pub fn pb_weights(msa: &Msa, abc: &Alphabet, ignore_rf: bool) -> Vec<f64> {
    let nseq = msa.nseq;
    let alen = msa.alen;
    let k = abc.k;
    let kp = abc.kp;

    // Contract: single-sequence MSA gets weight 1.0 (esl_msaweight.c:199).
    if nseq == 1 {
        return vec![1.0];
    }
    if nseq == 0 || alen == 0 {
        return vec![1.0; nseq];
    }

    let ax = msa.digitize(abc);

    // Count matrix ct[apos=0..=alen][a=0..Kp-1]; apos is 1-based, [0][] unused.
    let mut ct = vec![vec![0i32; kp]; alen + 1];
    // Consensus column indices (1-based), in increasing order.
    let mut conscols: Vec<usize> = Vec::with_capacity(alen);

    // Determine consensus columns early if we can. RF takes priority; else, on
    // a deep alignment, sample to determine consensus (esl_msaweight.c:205-207).
    // ncons stays 0 if neither path is used or the sample is rejected.
    if !ignore_rf && msa.rf.is_some() {
        consensus_by_rf(msa.rf.as_ref().unwrap(), abc, alen, &mut conscols);
    } else if PB_ALLOW_SAMP && nseq > PB_SAMPTHRESH {
        consensus_by_sample(&ax, abc, alen, nseq, &mut ct, &mut conscols);
    }

    // Collect count matrix (over all columns, or only consensus columns if
    // already known), excluding fragments' external gaps.
    collect_counts(&ax, abc, alen, nseq, &conscols, &mut ct);

    // If consensus columns weren't determined yet, do it now from <ct>.
    if conscols.is_empty() {
        consensus_by_all(&ct, abc, alen, &mut conscols);
    }

    // If still nothing, that's pathological -- use all columns.
    if conscols.is_empty() {
        conscols.extend(1..=alen);
    }

    let ncons = conscols.len();

    // Count how many different canonical residues are used in each consensus
    // column: r[j] (esl_msaweight.c:224-231).
    let mut r = vec![0i32; ncons];
    for (j, &apos) in conscols.iter().enumerate() {
        for a in 0..k {
            if ct[apos][a] > 0 {
                r[j] += 1;
            }
        }
    }

    // Bump sequence weights using the PB rule (esl_msaweight.c:234-246).
    let mut weights = vec![0.0_f64; nseq];
    for idx in 0..nseq {
        let mut rlen = 0i32;
        for (j, &apos) in conscols.iter().enumerate() {
            let a = ax[idx][apos] as usize;
            if a < k {
                weights[idx] += 1.0 / (r[j] * ct[apos][a]) as f64;
                rlen += 1;
            }
        }
        if rlen > 0 {
            weights[idx] /= rlen as f64;
        }
    }

    // Normalize to sum to 1, then scale to nseq (esl_vec_DNorm + DScale).
    let sum: f64 = weights.iter().sum();
    if sum != 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    } else {
        let uniform = 1.0 / nseq as f64;
        for w in &mut weights {
            *w = uniform;
        }
    }
    for w in &mut weights {
        *w *= nseq as f64;
    }

    weights
}

/// Use RF annotation to define consensus columns (esl_msaweight.c:271).
/// A column is consensus unless its RF character maps to the gap symbol
/// (`esl_abc_CIsGap`). `conscols` is filled with 1-based indices.
fn consensus_by_rf(rf: &[u8], abc: &Alphabet, alen: usize, conscols: &mut Vec<usize>) {
    let gap = abc.gap_code();
    for apos in 1..=alen {
        let c = rf.get(apos - 1).copied().unwrap_or(b' ');
        if abc.digitize_symbol(c) == gap {
            continue;
        }
        conscols.push(apos);
    }
}

/// Use counts from all sequences to determine consensus (esl_msaweight.c:400).
/// A column is consensus if its gap fraction `ct[K]/tot < symfrac`, where
/// `tot` sums symbol codes `0..Kp-2` (residues + gaps, incl. degeneracies,
/// excl. nonresidue/missing). Float arithmetic mirrors the C cast order.
fn consensus_by_all(ct: &[Vec<i32>], abc: &Alphabet, alen: usize, conscols: &mut Vec<usize>) {
    let k = abc.k;
    let kp = abc.kp;
    for apos in 1..=alen {
        let mut tot = 0i32;
        for a in 0..(kp - 2) {
            tot += ct[apos][a];
        }
        if (ct[apos][k] as f32 / tot as f32) < PB_SYMFRAC {
            conscols.push(apos);
        }
    }
}

/// Collect the observed symbol-count matrix `ct[apos][a]`, applying HMMER's
/// fragment-span rule so fragments' external (terminal) gaps are not counted
/// (esl_msaweight.c:434). A sequence is a fragment if its aligned span
/// `rpos-lpos+1 < ceil(fragthresh*alen)`; full-length sequences count columns
/// `1..=alen`, fragments only `lpos..=rpos`. If `conscols` is non-empty, only
/// those columns are counted (a pure optimization that matches C).
fn collect_counts(
    ax: &[Vec<u8>],
    abc: &Alphabet,
    alen: usize,
    nseq: usize,
    conscols: &[usize],
    ct: &mut [Vec<i32>],
) {
    // C re-zeros the whole matrix here (esl_mat_ISet, esl_msaweight.c:443), so
    // any counts left over from consensus_by_sample are discarded.
    for row in ct.iter_mut() {
        row.iter_mut().for_each(|c| *c = 0);
    }

    let minspan = (PB_FRAGTHRESH * alen as f32).ceil() as i64;

    let alen_i = alen as i64;
    for ax_seq in ax.iter().take(nseq) {
        // Leftmost / rightmost aligned residue (1..=alen), as 1-based signed
        // indices to mirror C's int loops (lpos may exit at alen+1, rpos at 0).
        let mut lpos: i64 = 1;
        while lpos <= alen_i && !abc.is_residue(ax_seq[lpos as usize]) {
            lpos += 1;
        }
        let mut rpos: i64 = alen_i;
        while rpos >= 1 && !abc.is_residue(ax_seq[rpos as usize]) {
            rpos -= 1;
        }

        // Fragment test: span = rpos-lpos+1. Full-length seqs reset to whole
        // alignment; fragments keep the [lpos,rpos] span. (span <= 0 for the
        // all-gap / empty case, which is < minspan, so treated as a fragment.)
        let span = rpos - lpos + 1;
        if span >= minspan {
            lpos = 1;
            rpos = alen_i;
        }

        if !conscols.is_empty() {
            for &apos in conscols {
                let apos = apos as i64;
                if apos > rpos {
                    break;
                }
                if apos < lpos {
                    continue;
                }
                let a = ax_seq[apos as usize] as usize;
                ct[apos as usize][a] += 1;
            }
        } else {
            let mut apos = lpos;
            while apos <= rpos {
                let a = ax_seq[apos as usize] as usize;
                ct[apos as usize][a] += 1;
                apos += 1;
            }
        }
    }
}

/// Determine consensus columns from a statistical subsample of sequences,
/// for deep alignments (esl_msaweight.c:321, `consensus_by_sample`).
///
/// Faithful port: samples `nsamp` sequence indices (0..nseq-1) without
/// replacement using Easel's 64-bit Mersenne Twister seeded with
/// `eslMSAWEIGHT_RNGSEED` (42) and the Vitter sequential-sampling `Deal`
/// algorithm, then collects observed symbol counts in `ct` over each sampled
/// sequence's fragment span (marking fragments by the `minspan = ceil(fragthresh*alen)`
/// rule). If at most `maxfrag` fragments are seen, consensus columns are those
/// with gap fraction `ct[K]/tot < symfrac`; otherwise sampling is rejected
/// (`conscols` left empty), and the caller falls back to `consensus_by_all`.
///
/// On entry `ct` is overwritten (zeroed then filled); on success `conscols`
/// holds the 1-based consensus-column indices. The `ct` populated here is
/// discarded by `collect_counts`, which re-zeros it — exactly as in C.
fn consensus_by_sample(
    ax: &[Vec<u8>],
    abc: &Alphabet,
    alen: usize,
    nseq: usize,
    ct: &mut [Vec<i32>],
    conscols: &mut Vec<usize>,
) {
    let k = abc.k;
    let kp = abc.kp;

    // Zero ct (esl_mat_ISet, esl_msaweight.c:340).
    for row in ct.iter_mut() {
        row.iter_mut().for_each(|c| *c = 0);
    }

    // Sample nsamp indices in 0..nseq-1, sorted ascending (esl_rand64_Deal).
    let mut rng = Rand64::new(PB_RNGSEED);
    let sampidx = esl_rand64_deal(&mut rng, PB_NSAMP as i64, nseq as i64);

    let minspan = (PB_FRAGTHRESH * alen as f32).ceil() as i64;
    let alen_i = alen as i64;
    let mut nfrag = 0usize;

    for &idx64 in &sampidx {
        let idx = idx64 as usize;
        let ax_seq = &ax[idx];

        let mut lpos: i64 = 1;
        while lpos <= alen_i && !abc.is_residue(ax_seq[lpos as usize]) {
            lpos += 1;
        }
        let mut rpos: i64 = alen_i;
        while rpos >= 1 && !abc.is_residue(ax_seq[rpos as usize]) {
            rpos -= 1;
        }
        if rpos - lpos + 1 < minspan {
            nfrag += 1;
        } else {
            lpos = 1;
            rpos = alen_i;
        }

        let mut apos = lpos;
        while apos <= rpos {
            let a = ax_seq[apos as usize] as usize;
            ct[apos as usize][a] += 1;
            apos += 1;
        }
    }

    if nfrag <= PB_MAXFRAG {
        for apos in 1..=alen {
            let mut tot = 0i32;
            for a in 0..(kp - 2) {
                tot += ct[apos][a];
            }
            if (ct[apos][k] as f32 / tot as f32) < PB_SYMFRAC {
                conscols.push(apos);
            }
        }
    }
    // else: too many fragments -> reject; conscols stays empty (eslFAIL path).
}

/// Apply the `#=GC MM` model mask to the digitized alignment in place
/// (`do_modelmask`, build.c:222). In every column marked `'m'`, each non-gap,
/// non-missing residue is rewritten to the degenerate "any" symbol (Kp-3).
/// `ax` rows are 1-based (`ax[seq][1..=alen]`); `msa.mm[apos-1]` is the mask.
fn do_modelmask(ax: &mut [Vec<u8>], abc: &Alphabet, msa: &Msa) {
    let Some(mm) = msa.mm.as_ref() else {
        return;
    };
    let any = abc.unknown_code(); // Kp-3
    let gap = abc.gap_code(); // K
    let missing = abc.missing_code(); // Kp-1
    for apos in 1..=msa.alen {
        if mm.get(apos - 1).copied() != Some(b'm') {
            continue;
        }
        for row in ax.iter_mut() {
            let c = row[apos];
            if c != gap && c != missing {
                row[apos] = any;
            }
        }
    }
}

/// Transfer rf/mm/cs/ca optional annotation from the MSA to the model
/// (`annotate_model`, build.c:338). Each line, if present in the MSA, is
/// emitted as a model array `hmm.X[0..=M]` where index 0 is a leading space
/// and indices 1..=M hold the source character at each successive match
/// column. The MM line maps `'.'` to `'-'` exactly as C does (build.c:360).
/// (The alignment column map is set separately by the caller, matching the
/// `p7H_MAP` portion of annotate_model.)
fn annotate_model(hmm: &mut Hmm, matassign: &[bool], msa: &Msa) {
    let alen = msa.alen;

    if let Some(rf) = msa.rf.as_ref() {
        let mut out = vec![b' '; hmm.m + 2];
        let mut k = 1usize;
        for apos in 1..=alen {
            if matassign[apos - 1] {
                out[k] = rf.get(apos - 1).copied().unwrap_or(b' ');
                k += 1;
            }
        }
        hmm.rf = Some(out);
        hmm.flags |= P7H_RF;
    }

    if let Some(mm) = msa.mm.as_ref() {
        let mut out = vec![b' '; hmm.m + 2];
        let mut k = 1usize;
        for apos in 1..=alen {
            if matassign[apos - 1] {
                let c = mm.get(apos - 1).copied().unwrap_or(b' ');
                out[k] = if c == b'.' { b'-' } else { c };
                k += 1;
            }
        }
        hmm.mm = Some(out);
        hmm.flags |= P7H_MMASK;
    }

    if let Some(ss) = msa.ss_cons.as_ref() {
        let mut out = vec![b' '; hmm.m + 2];
        let mut k = 1usize;
        for apos in 1..=alen {
            if matassign[apos - 1] {
                out[k] = ss.get(apos - 1).copied().unwrap_or(b' ');
                k += 1;
            }
        }
        hmm.cs = Some(out);
        hmm.flags |= P7H_CS;
    }

    if let Some(sa) = msa.sa_cons.as_ref() {
        let mut out = vec![b' '; hmm.m + 2];
        let mut k = 1usize;
        for apos in 1..=alen {
            if matassign[apos - 1] {
                out[k] = sa.get(apos - 1).copied().unwrap_or(b' ');
                k += 1;
            }
        }
        hmm.ca = Some(out);
        hmm.flags |= P7H_CA;
    }
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
        RelativeWeighting::PositionBased => pb_weights(msa, abc, !hand_arch),
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
        RelativeWeighting::PositionBased => pb_weights(msa, abc, !hand_arch),
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

    // Apply the #=GC MM model mask (do_modelmask, build.c:222): rewrite
    // non-gap residues in 'm'-marked columns to the degenerate "any" symbol
    // (Kp-3) before counting, so those columns get degenerate emission counts.
    // Architecture and weighting above used the unmasked residues, matching C
    // (do_modelmask runs at the top of matassign2hmm, after modelmaking).
    do_modelmask(&mut ax, abc, msa);

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

    // Transfer rf/mm/cs/ca annotation from the MSA (annotate_model, build.c:338):
    // for each annotation line present, copy the original character at every
    // match column into a 1-based model array hmm->X[1..M], with index 0 = ' '.
    annotate_model(&mut hmm, &matassign, msa);

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
            prior_strategy,
            target_re,
            target_sigma,
        )),
        EffectiveSeqNumber::EntropyExp {
            target_re,
            target_sigma,
        } => {
            crate::eweight::entropy_weight_exp(&mut hmm, bg, prior_strategy, target_re, target_sigma);
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
