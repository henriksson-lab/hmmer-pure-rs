//! Exact SSV-over-FM diagonal kernel for nhmmer's FM-index long-target path.
//!
//! This is an additive, faithful port of the core of C's
//! `p7_SSVFM_longlarget` pipeline (hmmer/src/fm_ssv.c): the single FM trie
//! traversal (`FM_getSeeds` -> `FM_Recurse`) that scores SSV diagonals
//! *directly* over FM-index intervals using the per-position match log-odds
//! score table (C's `ssvdata->ssv_scores_f[k*Kp + x]`), carrying the full
//! `FM_DP_PAIR` DP state and pruning exactly as C does, followed by
//! `FM_mergeSeeds` and `FM_extendSeed`.
//!
//! SCOPE / WHAT IS FAITHFUL vs WHAT IS NOT (read before wiring into production):
//!
//! * The Rust `FmIndex` (src/fm_index.rs) exposes only single-direction
//!   backward search via `prepend_interval`, which is exactly the primitive
//!   C uses for the `fm_direction == fm_forward` sweep
//!   (`fm_updateIntervalReverse(fmf, ...)`). That forward sweep is ported here
//!   faithfully, including the full DP-pair state and every pruning clause.
//!
//! * C also runs a second `fm_direction == fm_reverse` sweep that requires a
//!   *bi-directional* FM index (`fmb` plus a second interval, updated via
//!   `fm_updateIntervalForward`). The Rust `FmIndex` has no bi-directional
//!   support, so that half of `FM_Recurse` CANNOT be ported here without first
//!   extending `FmIndex`. This is reported as a dependency; this module covers
//!   the forward-on-model / forward-on-FM and the complement variants that the
//!   forward FM sweep can express.
//!
//! * The score table here is the per-position match LOD in bits, computed from
//!   the HMM emission and background (identical convention to C's
//!   `ssv_scores_f` for match states and to nhmmer.rs `fm_match_lod_bits`).
//!   No optimized-profile `ssv_scores_f` field exists on the Rust oprofile, so
//!   the table is constructed locally; the algorithm is validated against it.
//!
//! This module is unit-tested but is NOT (yet) wired into the production
//! nhmmer path, which keeps its verified seed-then-rescore approximation. See
//! the report accompanying this change.

use crate::fm_index::{FmIndex, FmInterval};

/// Forward / backward orientation of the diagonal over the *model*.
/// Port of C's `fm_forward` / `fm_backward` for `model_direction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelDirection {
    Forward,
    Backward,
}

/// Strand of the target. Port of C's `p7_NOCOMPLEMENT` / `p7_COMPLEMENT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complementarity {
    NoComplement,
    Complement,
}

/// Tunable thresholds and pruning parameters.
/// Mirrors the fields of C's `FM_CFG` / `FM_METADATA` that `FM_Recurse`,
/// `FM_mergeSeeds`, and `FM_extendSeed` read.
#[derive(Debug, Clone, Copy)]
pub struct FmSsvConfig {
    /// `fm_cfg->max_depth`: trie pruning length.
    pub max_depth: usize,
    /// `fm_cfg->drop_max_len`: max consecutive low-scoring run past the peak.
    pub drop_max_len: usize,
    /// `fm_cfg->drop_lim`: how close to the peak still counts as a new peak.
    pub drop_lim: f32,
    /// `fm_cfg->consec_pos_req`: required run of positive-scoring matches.
    pub consec_pos_req: usize,
    /// `fm_cfg->consensus_match_req`: consecutive consensus matches that force a seed.
    pub consensus_match_req: usize,
    /// `fm_cfg->score_density_req`: minimum bits/position.
    pub score_density_req: f32,
    /// `fm_cfg->ssv_length`: window length around a seed for `FM_mergeSeeds`/`FM_extendSeed`.
    pub ssv_length: usize,
}

impl Default for FmSsvConfig {
    fn default() -> Self {
        // Defaults mirror nhmmer.rs NhmmerFmSeedConfig::default / C fm defaults.
        Self {
            max_depth: 15,
            drop_max_len: 4,
            drop_lim: 0.3,
            consec_pos_req: 5,
            consensus_match_req: 11,
            score_density_req: 0.75,
            ssv_length: 100,
        }
    }
}

/// Compact per-diagonal DP state. Faithful port of C `FM_DP_PAIR`
/// (hmmer/src/hmmer.h). `pos` is the model position `k` reached on this path.
#[derive(Debug, Clone, Copy)]
pub struct FmDpPair {
    pub pos: i32,
    pub score: f32,
    pub max_score: f32,
    pub score_peak_len: i32,
    pub consec_pos: i16,
    pub max_consec_pos: i16,
    pub consec_consensus: i16,
    pub complementarity: Complementarity,
    pub model_direction: ModelDirection,
}

/// A diagonal seed. Faithful subset of C `FM_DIAG` (the fields the kernel and
/// `FM_mergeSeeds`/`FM_extendSeed` actually use).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmDiag {
    /// Target start position (FM coordinate), C `diag->n`.
    pub n: i64,
    /// Model position, C `diag->k`.
    pub k: i32,
    /// Diagonal length, C `diag->length`.
    pub length: i32,
    pub score: f32,
    pub complementarity: Complementarity,
}

impl FmDiag {
    /// C `sortkey`: groups by strand, then diagonal (n - k), then a fractional
    /// model-position tiebreak. Used by `FM_mergeSeeds`'s qsort.
    fn sortkey(&self, fm_n: i64, m: i32) -> f64 {
        let strand_base = if self.complementarity == Complementarity::Complement {
            (fm_n + 1) as f64
        } else {
            0.0
        };
        strand_base + (self.n - self.k as i64) as f64 + (self.k as f64) / ((m + 1) as f64)
    }
}

/// Per-position match score table. Index `[k * kp + x]` gives the LOD score in
/// bits for emitting alphabet symbol `x` at model node `k` (1..=M). This is the
/// Rust analog of C's `ssvdata->ssv_scores_f`. `kp` is the alphabet stride.
#[derive(Debug, Clone)]
pub struct SsvScores {
    pub scores: Vec<f32>,
    pub m: i32,
    pub kp: usize,
}

impl SsvScores {
    pub fn new(m: i32, kp: usize) -> Self {
        let len = ((m as usize) + 1) * kp;
        Self {
            scores: vec![0.0; len],
            m,
            kp,
        }
    }

    #[inline]
    pub fn get(&self, k: i32, x: usize) -> f32 {
        self.scores[(k as usize) * self.kp + x]
    }

    #[inline]
    pub fn set(&mut self, k: i32, x: usize, v: f32) {
        self.scores[(k as usize) * self.kp + x] = v;
    }
}

/// Metadata describing the (DNA) alphabet for the FM trie traversal.
/// Mirrors the bits of C's `FM_METADATA` the kernel reads: alphabet size and
/// complement map.
#[derive(Debug, Clone)]
pub struct FmAlphabet {
    /// Number of "real" characters to enumerate over the trie (C `alph_size`,
    /// e.g. 4 for ACGT).
    pub alph_size: usize,
    /// Complement code map, `compl_alph[c]` (e.g. A<->T, C<->G).
    pub compl_alph: Vec<usize>,
}

impl FmAlphabet {
    /// Standard DNA: codes 0=A,1=C,2=G,3=T with Watson-Crick complement.
    pub fn dna() -> Self {
        Self {
            alph_size: 4,
            // A<->T (0<->3), C<->G (1<->2)
            compl_alph: vec![3, 2, 1, 0],
        }
    }
}

/// Result of `FM_getPassingDiags`: each surviving FM interval entry becomes a
/// seed. Faithful to C's coordinate computation.
#[allow(clippy::too_many_arguments)]
fn get_passing_diags(
    fm: &FmIndex,
    k: i32,
    depth: i32,
    model_direction: ModelDirection,
    complementarity: Complementarity,
    interval: FmInterval,
    seeds: &mut Vec<FmDiag>,
) {
    // C iterates interval->lower..=interval->upper, backtracking each to a
    // target position. The Rust FmIndex resolves a whole interval to text
    // positions in one call (`locate_interval`), which is equivalent to C's
    // per-entry FM_backtrackSeed over the same SA range.
    let positions = fm.locate_interval(interval);
    let fm_n = fm.n as i64;
    for backtrack in positions {
        let backtrack = backtrack as i64;
        let mut seed = FmDiag {
            k,
            length: depth,
            // C: n = (NOCOMPLEMENT) ? N - backtrack - depth - 1 : backtrack
            n: if complementarity == Complementarity::NoComplement {
                fm_n - backtrack - depth as i64 - 1
            } else {
                backtrack
            },
            score: 0.0,
            complementarity,
        };
        // C: if model_direction == fm_forward, seed->k -= (depth - 1)
        if model_direction == ModelDirection::Forward {
            seed.k -= depth - 1;
        }
        seeds.push(seed);
    }
}

/// Heart of the kernel: faithful port of C `FM_Recurse` restricted to the
/// `fm_direction == fm_forward` sweep (single backward-search FM index).
///
/// `dp_pairs` is the shared scratch vector C indexes with `first..=last`;
/// surviving children are appended past `last` and recursed on. We mirror that
/// exactly using indices into a growable Vec.
#[allow(clippy::too_many_arguments)]
fn fm_recurse(
    depth: i32,
    fm: &FmIndex,
    alph: &FmAlphabet,
    scores: &SsvScores,
    consensus: &[usize],
    sc_thresh_fm: f32,
    config: &FmSsvConfig,
    dp_pairs: &mut Vec<FmDpPair>,
    first: usize,
    last: usize,
    interval_1: FmInterval,
    seeds: &mut Vec<FmDiag>,
) {
    let m = scores.m;

    for c in 0..alph.alph_size {
        // dppos tracks the last index appended for this character's column.
        let mut dppos = last;

        for i in first..=last {
            let pair = dp_pairs[i];
            let k = match pair.model_direction {
                ModelDirection::Forward => pair.pos + 1,
                ModelDirection::Backward => pair.pos - 1,
            };

            // next_score and consensus character, accounting for complement.
            let (next_score, cons_c) = match pair.complementarity {
                Complementarity::Complement => {
                    let cc = alph.compl_alph[c];
                    (scores.get(k, cc), alph.compl_alph[consensus[k as usize]])
                }
                Complementarity::NoComplement => (scores.get(k, c), consensus[k as usize]),
            };

            let sc = pair.score + next_score;
            let positive_run: i16 = if next_score > 0.0 {
                pair.consec_pos + 1
            } else {
                0
            };
            let consec_consensus: i16 = if c == cons_c {
                pair.consec_consensus + 1
            } else {
                0
            };

            if sc >= sc_thresh_fm
                || (config.consensus_match_req > 0
                    && consec_consensus as usize == config.consensus_match_req)
            {
                // A seed to extend: shrink the FM interval by prepending c and
                // record all matching target positions.
                if !interval_1.is_empty() {
                    if let Some(new_iv) = fm.prepend_interval(interval_1, fm_code_to_base(c)) {
                        get_passing_diags(
                            fm,
                            k,
                            depth,
                            pair.model_direction,
                            pair.complementarity,
                            new_iv,
                            seeds,
                        );
                    }
                }
            } else if sc <= 0.0
                || depth == config.max_depth as i32
                || (pair.model_direction == ModelDirection::Forward && k == m)
                || (pair.model_direction == ModelDirection::Backward && k == 1)
                || depth == pair.score_peak_len + config.drop_max_len as i32
                || (depth > 4
                    && depth > consec_consensus as i32
                    && (sc / depth as f32) < config.score_density_req)
                || ((pair.max_consec_pos as usize) < config.consec_pos_req
                    && config.consec_pos_req.saturating_sub(positive_run as usize)
                        == (config.max_depth as i32 - depth + 1) as usize)
            {
                // pruned - do nothing
            } else {
                // Extendable: append a new DP pair past `last`.
                dppos += 1;
                let (max_score, score_peak_len) = if sc > pair.max_score {
                    (sc, depth)
                } else if sc >= pair.max_score - config.drop_lim {
                    (pair.max_score, depth)
                } else {
                    (pair.max_score, pair.score_peak_len)
                };
                let new_pair = FmDpPair {
                    pos: k,
                    score: sc,
                    max_score,
                    score_peak_len,
                    consec_pos: positive_run,
                    max_consec_pos: positive_run.max(pair.max_consec_pos),
                    consec_consensus,
                    complementarity: pair.complementarity,
                    model_direction: pair.model_direction,
                };
                // dp_pairs grows; index dppos may be one past current end.
                if dppos < dp_pairs.len() {
                    dp_pairs[dppos] = new_pair;
                } else {
                    debug_assert_eq!(dppos, dp_pairs.len());
                    dp_pairs.push(new_pair);
                }
            }
        }

        if dppos > last {
            // At least one extendable diagonal: descend the trie on c.
            if interval_1.is_empty() {
                continue;
            }
            let Some(new_iv) = fm.prepend_interval(interval_1, fm_code_to_base(c)) else {
                continue;
            };
            fm_recurse(
                depth + 1,
                fm,
                alph,
                scores,
                consensus,
                sc_thresh_fm,
                config,
                dp_pairs,
                last + 1,
                dppos,
                new_iv,
                seeds,
            );
        }
    }
}

/// Map an FM alphabet code (0=A,1=C,2=G,3=T) to the byte the Rust `FmIndex`
/// expects. The Rust FM indexes are built over ASCII uppercase ACGT text.
#[inline]
fn fm_code_to_base(c: usize) -> u8 {
    match c {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'T',
        _ => 0,
    }
}

/// Faithful port of C `FM_getSeeds` (forward sweep only). Kickstarts a DP
/// column for each starting character `c`, seeding forward-on-model and
/// reverse-on-model diagonals (and their complement variants), then recurses.
///
/// `strands`: when `top_only`/`bottom_only` are both false, both strands are
/// considered (C `p7_STRAND_BOTH`).
#[allow(clippy::too_many_arguments)]
pub fn fm_get_seeds(
    fm: &FmIndex,
    alph: &FmAlphabet,
    scores: &SsvScores,
    consensus: &[usize],
    sc_thresh_fm: f32,
    config: &FmSsvConfig,
    top_only: bool,
    bottom_only: bool,
) -> Vec<FmDiag> {
    let m = scores.m;
    let mut seeds: Vec<FmDiag> = Vec::new();

    for c in 0..alph.alph_size {
        // Root interval for character c (whole interval that ends in c after
        // one backward step). The Rust FmIndex computes this via prepend on the
        // root interval.
        let Some(root_c) = fm.prepend_interval(fm.root_interval(), fm_code_to_base(c)) else {
            continue;
        };

        // Build the initial DP column (compressed: positive-scoring entries
        // only). C keeps four kinds; we keep forward-on-model and
        // reverse-on-model entries (and complement variants) here.
        let mut dp_pairs: Vec<FmDpPair> = Vec::new();

        for k in 1..=m {
            if !bottom_only {
                let sc = scores.get(k, c);
                if sc > 0.0 {
                    // fwd on model
                    if k < m - 3 {
                        dp_pairs.push(initial_pair(
                            k,
                            sc,
                            c == consensus[k as usize],
                            Complementarity::NoComplement,
                            ModelDirection::Forward,
                        ));
                    }
                    // rev on model
                    if k > 4 {
                        dp_pairs.push(initial_pair(
                            k,
                            sc,
                            c == consensus[k as usize],
                            Complementarity::NoComplement,
                            ModelDirection::Backward,
                        ));
                    }
                }
            }
            if !top_only {
                let cc = alph.compl_alph[c];
                let sc = scores.get(k, cc);
                if sc > 0.0 {
                    if k > 4 {
                        dp_pairs.push(initial_pair(
                            k,
                            sc,
                            c == consensus[k as usize],
                            Complementarity::Complement,
                            ModelDirection::Backward,
                        ));
                    }
                    if k < m - 3 {
                        dp_pairs.push(initial_pair(
                            k,
                            sc,
                            c == consensus[k as usize],
                            Complementarity::Complement,
                            ModelDirection::Forward,
                        ));
                    }
                }
            }
        }

        if dp_pairs.is_empty() {
            continue;
        }
        let last = dp_pairs.len() - 1;
        fm_recurse(
            2,
            fm,
            alph,
            scores,
            consensus,
            sc_thresh_fm,
            config,
            &mut dp_pairs,
            0,
            last,
            root_c,
            &mut seeds,
        );
    }

    fm_merge_seeds(&mut seeds, fm.n as i64, m, config.ssv_length as i32);
    seeds
}

#[inline]
fn initial_pair(
    k: i32,
    sc: f32,
    is_consensus: bool,
    complementarity: Complementarity,
    model_direction: ModelDirection,
) -> FmDpPair {
    FmDpPair {
        pos: k,
        score: sc,
        max_score: sc,
        score_peak_len: 1,
        consec_pos: 1,
        max_consec_pos: 1,
        consec_consensus: if is_consensus { 1 } else { 0 },
        complementarity,
        model_direction,
    }
}

/// Faithful port of C `FM_mergeSeeds`: sort by (strand, diagonal, model pos),
/// then merge overlapping/nearby diagonals on the same diagonal.
pub fn fm_merge_seeds(seeds: &mut Vec<FmDiag>, fm_n: i64, m: i32, ssv_length: i32) {
    if seeds.is_empty() {
        return;
    }

    seeds.sort_by(|a, b| {
        a.sortkey(fm_n, m)
            .partial_cmp(&b.sortkey(fm_n, m))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out: Vec<FmDiag> = Vec::with_capacity(seeds.len());

    let curr = seeds[0];
    let mut curr_complement = curr.complementarity == Complementarity::Complement;
    let mut curr_n = curr.n;
    let mut curr_k = curr.k;
    let mut curr_len = curr.length;
    let mut curr_end = curr_n + curr_len as i64 - 1;
    let mut curr_diagval = curr.n - curr.k as i64;

    for next in seeds.iter().skip(1).copied() {
        let next_complement = next.complementarity == Complementarity::Complement;
        if next_complement == curr_complement
            && (next.n - next.k as i64) == curr_diagval
            && next.n + (next.length as i64) < curr_n + curr_len as i64 + ssv_length as i64
        {
            let tmp = next.n + next.length as i64 - 1;
            if tmp > curr_end {
                curr_end = tmp;
                curr_len = (curr_end - curr_n + 1) as i32;
            }
        } else {
            out.push(FmDiag {
                n: curr_n,
                k: curr_k,
                length: (curr_end - curr_n + 1) as i32,
                score: 0.0,
                complementarity: if curr_complement {
                    Complementarity::Complement
                } else {
                    Complementarity::NoComplement
                },
            });
            curr_n = next.n;
            curr_k = next.k;
            curr_len = next.length;
            curr_end = curr_n + curr_len as i64 - 1;
            curr_diagval = next.n - next.k as i64;
            curr_complement = next_complement;
        }
    }
    out.push(FmDiag {
        n: curr_n,
        k: curr_k,
        length: (curr_end - curr_n + 1) as i32,
        score: 0.0,
        complementarity: if curr_complement {
            Complementarity::Complement
        } else {
            Complementarity::NoComplement
        },
    });

    *seeds = out;
}

/// Faithful port of C `FM_extendSeed`: extend the seed in both directions
/// within a bounded window and keep the best-scoring sub-diagonal.
///
/// `target` is the (already strand-resolved) target sequence in alphabet
/// codes, 1-indexed to match C's `tmp_sq->dsq` (index 0 is a sentinel).
/// `fm_n` is the FM text length `fm->N`.
pub fn fm_extend_seed(
    diag: &mut FmDiag,
    target: &[usize],
    scores: &SsvScores,
    config: &FmSsvConfig,
    fm_n: i64,
) {
    let m = scores.m;
    let kp = scores.kp;
    let extend = (config.ssv_length as i32 - diag.length).max(10);

    // C: model_start = max(1, diag->k - extend + 1); the "+1" is load-bearing.
    let mut model_start = (diag.k - extend + 1).max(1);
    let mut model_end = (diag.k + diag.length + extend - 1).min(m);
    let mut target_start: i64 = diag.n - (diag.k as i64 - model_start as i64);
    let mut target_end: i64 = diag.n + (model_end as i64 - diag.k as i64);

    if target_start < 0 {
        model_start -= target_start as i32;
        target_start = 0;
    }
    if target_end > fm_n - 2 {
        model_end -= (target_end - (fm_n - 2)) as i32;
        target_end = fm_n - 2;
    }
    if model_start > model_end || target_start > target_end {
        return;
    }

    // Walk model_start..=model_end against target[1..]. C reads tmp_sq->dsq[n]
    // with n starting at 1. We expect `target` to be laid out the same way
    // (target[0] is sentinel; the extracted range begins at index 1).
    let mut k = model_start;
    let mut n: usize = 1;
    let mut sc = 0.0_f32;
    let mut hit_start: i64 = n as i64;
    let mut max_sc = 0.0_f32;
    let mut max_hit_start: i64 = n as i64;
    let mut max_hit_end: i64 = n as i64;

    while k <= model_end {
        let cidx = (k as usize) * kp + target.get(n).copied().unwrap_or(0);
        let delta = scores.scores.get(cidx).copied().unwrap_or(0.0);
        sc += delta;
        if sc < 0.0 {
            sc = 0.0;
            hit_start = n as i64 + 1;
        } else if sc > max_sc {
            max_sc = sc;
            max_hit_start = hit_start;
            max_hit_end = n as i64;
        }
        k += 1;
        n += 1;
    }

    diag.n = target_start + max_hit_start - 1;
    diag.k = model_start + (max_hit_start - 1) as i32;
    diag.length = (max_hit_end - max_hit_start + 1) as i32;
    diag.score = max_sc;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a score table where the consensus base at node k scores +2 bits and
    // everything else scores -3 bits. kp=4 (ACGT only) for simplicity.
    fn consensus_scores(consensus_bases: &[usize]) -> (SsvScores, Vec<usize>) {
        let m = consensus_bases.len() as i32;
        let kp = 4;
        let mut scores = SsvScores::new(m, kp);
        let mut consensus = vec![0usize; (m as usize) + 1];
        for (i, &b) in consensus_bases.iter().enumerate() {
            let k = (i + 1) as i32;
            consensus[k as usize] = b;
            for x in 0..kp {
                scores.set(k, x, if x == b { 2.0 } else { -3.0 });
            }
        }
        (scores, consensus)
    }

    fn codes_to_text(codes: &[usize]) -> Vec<u8> {
        codes.iter().map(|&c| fm_code_to_base(c)).collect()
    }

    #[test]
    fn ssv_scores_index_roundtrips() {
        let mut s = SsvScores::new(3, 4);
        s.set(2, 1, 1.5);
        assert_eq!(s.get(2, 1), 1.5);
        assert_eq!(s.get(1, 0), 0.0);
        assert_eq!(s.scores.len(), 4 * 4);
    }

    #[test]
    fn extend_seed_uses_c_model_start_rule_and_clips_best_subdiagonal() {
        // Model consensus ACGTAC (codes 0,1,2,3,0,1), M=6.
        let (scores, _consensus) = consensus_scores(&[0, 1, 2, 3, 0, 1]);
        let config = FmSsvConfig {
            ssv_length: 6,
            ..FmSsvConfig::default()
        };
        // Target (1-indexed, sentinel at 0): perfect match to consensus.
        // text positions 0..=5 = ACGTAC
        let target_codes = [0usize, 0, 1, 2, 3, 0, 1]; // index 0 sentinel
        let fm_n = 8; // pretend FM text length
                      // Seed covering model k=4 length 2 (positions GT at model 3..4 -> diag.k=4)
        let mut diag = FmDiag {
            n: 3, // target start of seed (0-based fm coord) - aligns model pos
            k: 4,
            length: 2,
            score: 0.0,
            complementarity: Complementarity::NoComplement,
        };
        fm_extend_seed(&mut diag, &target_codes, &scores, &config, fm_n);
        // With a perfect-match target the extension should grow the diagonal and
        // produce a positive score equal to 2 bits per matched position.
        assert!(diag.score > 0.0);
        assert!(diag.length >= 2);
        // model_start rule: max(1, k - extend + 1) with extend=max(10,6-2)=10
        // => model_start = max(1, 4-10+1) = 1, so the window starts at model 1.
        assert!(diag.k >= 1 && diag.k <= 6);
    }

    #[test]
    fn merge_seeds_combines_overlapping_same_diagonal() {
        let mut seeds = vec![
            FmDiag {
                n: 10,
                k: 5,
                length: 4,
                score: 0.0,
                complementarity: Complementarity::NoComplement,
            },
            FmDiag {
                n: 12,
                k: 7,
                length: 4,
                score: 0.0,
                complementarity: Complementarity::NoComplement,
            },
        ];
        // Same diagonal (n-k = 5 for both), overlapping -> merge into one.
        fm_merge_seeds(&mut seeds, 1000, 20, 100);
        assert_eq!(seeds.len(), 1);
        let d = seeds[0];
        assert_eq!(d.n, 10);
        // merged end = max(10+4-1, 12+4-1) = 15 -> length = 15-10+1 = 6
        assert_eq!(d.length, 6);
    }

    #[test]
    fn merge_seeds_keeps_distinct_diagonals_separate() {
        let mut seeds = vec![
            FmDiag {
                n: 10,
                k: 5,
                length: 4,
                score: 0.0,
                complementarity: Complementarity::NoComplement,
            },
            // Different diagonal (n - k differs)
            FmDiag {
                n: 50,
                k: 5,
                length: 4,
                score: 0.0,
                complementarity: Complementarity::NoComplement,
            },
        ];
        fm_merge_seeds(&mut seeds, 1000, 20, 100);
        assert_eq!(seeds.len(), 2);
    }

    #[test]
    fn merge_seeds_separates_strands() {
        let mut seeds = vec![
            FmDiag {
                n: 10,
                k: 5,
                length: 4,
                score: 0.0,
                complementarity: Complementarity::NoComplement,
            },
            FmDiag {
                n: 10,
                k: 5,
                length: 4,
                score: 0.0,
                complementarity: Complementarity::Complement,
            },
        ];
        fm_merge_seeds(&mut seeds, 1000, 20, 100);
        assert_eq!(seeds.len(), 2);
    }

    #[test]
    fn recurse_finds_exact_consensus_seed_in_target() {
        // Model consensus = ACGTACGT (8 nodes). A target containing exactly that
        // string should yield a seed on the matching diagonal.
        let consensus_bases = vec![0usize, 1, 2, 3, 0, 1, 2, 3];
        let (scores, consensus) = consensus_scores(&consensus_bases);
        let alph = FmAlphabet::dna();

        // Target text: flanks + the consensus string embedded once.
        // Forward FM index in this codebase is built over the REVERSED text.
        let target_text = b"TTTTACGTACGTTTTT".to_vec();
        let reversed: Vec<u8> = target_text.iter().rev().copied().collect();
        let fm = FmIndex::build(&reversed);

        let config = FmSsvConfig {
            max_depth: 8,
            consec_pos_req: 1,
            consensus_match_req: 4,
            score_density_req: 0.0,
            drop_max_len: 8,
            ssv_length: 16,
            ..FmSsvConfig::default()
        };

        // Threshold low enough that the consensus run is found.
        let seeds = fm_get_seeds(&fm, &alph, &scores, &consensus, 6.0, &config, true, false);

        // There must be at least one forward (NoComplement) seed.
        assert!(
            seeds
                .iter()
                .any(|d| d.complementarity == Complementarity::NoComplement),
            "expected a forward-strand seed, got {seeds:?}"
        );

        // The embedded consensus starts at target position 4 (0-based) in the
        // forward text "TTTTACGTACGT...". At least one seed should map onto the
        // ACGTACGT region (target n in [4, 11]).
        assert!(
            seeds.iter().any(|d| {
                d.complementarity == Complementarity::NoComplement && d.n >= 0 && d.n <= 12
            }),
            "expected a seed near the embedded consensus, got {seeds:?}"
        );

        // Sanity: the located forward text should actually contain the consensus.
        let cons_text = codes_to_text(&consensus_bases);
        assert!(target_text
            .windows(cons_text.len())
            .any(|w| w == cons_text.as_slice()));
    }

    #[test]
    fn recurse_emits_no_seeds_for_absent_string() {
        // Consensus ACGTACGT but target has no such run -> no high-scoring seed.
        let consensus_bases = vec![0usize, 1, 2, 3, 0, 1, 2, 3];
        let (scores, consensus) = consensus_scores(&consensus_bases);
        let alph = FmAlphabet::dna();

        // Target with no ACGT runs (all A then all T).
        let target_text = b"AAAAAAAATTTTTTTT".to_vec();
        let reversed: Vec<u8> = target_text.iter().rev().copied().collect();
        let fm = FmIndex::build(&reversed);

        let config = FmSsvConfig {
            max_depth: 8,
            consec_pos_req: 5,
            consensus_match_req: 8,
            score_density_req: 0.75,
            ssv_length: 16,
            ..FmSsvConfig::default()
        };

        // High threshold: with a -3 mismatch penalty no diagonal in this target
        // can reach +12 bits over 8 positions.
        let seeds = fm_get_seeds(&fm, &alph, &scores, &consensus, 12.0, &config, true, false);
        assert!(
            seeds.is_empty(),
            "expected no seeds for absent consensus, got {seeds:?}"
        );
    }

    #[test]
    fn full_pipeline_recurse_merge_extend_runs_clean() {
        // End-to-end: get seeds, then extend each against the target and ensure
        // the kernel produces a coherent diagonal without panicking.
        let consensus_bases = vec![0usize, 1, 2, 3, 0, 1, 2, 3, 0, 1];
        let (scores, consensus) = consensus_scores(&consensus_bases);
        let alph = FmAlphabet::dna();

        let target_text = b"GGACGTACGTACGG".to_vec();
        let reversed: Vec<u8> = target_text.iter().rev().copied().collect();
        let fm = FmIndex::build(&reversed);

        let config = FmSsvConfig {
            max_depth: 10,
            consec_pos_req: 1,
            consensus_match_req: 4,
            score_density_req: 0.0,
            drop_max_len: 10,
            ssv_length: 14,
            ..FmSsvConfig::default()
        };

        let mut seeds = fm_get_seeds(&fm, &alph, &scores, &consensus, 6.0, &config, true, false);
        assert!(!seeds.is_empty());

        // Build a 1-indexed target in alphabet codes for forward extension.
        let mut target_codes = vec![0usize];
        for &b in &target_text {
            target_codes.push(match b {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                _ => 0,
            });
        }
        let fm_n = target_text.len() as i64 + 1;
        for diag in seeds.iter_mut() {
            if diag.complementarity == Complementarity::NoComplement {
                fm_extend_seed(diag, &target_codes, &scores, &config, fm_n);
                assert!(diag.length >= 0);
            }
        }
    }
}
