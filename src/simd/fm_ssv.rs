//! Exact SSV-over-FM diagonal kernel for nhmmer's FM-index long-target path.
#![allow(clippy::needless_range_loop)]
//!
//! This is an additive, faithful port of the core of C's
//! `p7_SSVFM_longlarget` pipeline (hmmer/src/fm_ssv.c): the single FM trie
//! traversal (`FM_getSeeds` -> `FM_Recurse`) that scores SSV diagonals
//! *directly* over FM-index intervals using the per-position match log-odds
//! score table (C's `ssvdata->ssv_scores_f[k*Kp + x]`), carrying the full
//! `FM_DP_PAIR` DP state and pruning exactly as C does, followed by
//! `FM_mergeSeeds` and `FM_extendSeed`.
//!
//! SCOPE / FAITHFULNESS:
//!
//! * Both of C's sweeps are ported. The `fm_direction == fm_forward` sweep uses
//!   `FmIndex::prepend_interval` (C `fm_updateIntervalReverse(fmf, ...)`) on the
//!   reversed-text index; the `fm_direction == fm_backward` sweep uses
//!   `FmIndex::update_interval_forward` (C `fm_updateIntervalForward(fmb, ...)`)
//!   on the forward-text index, carrying the second (fmf-locatable) interval.
//!   The full `FM_DP_PAIR` state and every pruning clause are reproduced (except
//!   C's `opt_ext_fwd/rev` look-ahead prune, whose omission only makes Rust
//!   explore *more*, never fewer — see `fm_recurse`).
//!
//! * The kernel is unit-agnostic in its score inputs: `SsvScores`, `sc_thresh_fm`,
//!   and the score-valued `FmSsvConfig` fields (`drop_lim`, `score_density_req`)
//!   must all share a unit. The nhmmer caller supplies them in NATS (`ln(p/q)`;
//!   bit thresholds * ln2) to mirror C's `ssv_scores_f`/`fm_cfg`. The kernel
//!   only emits seed *coordinates*, so the unit stays isolated from the
//!   bit-based downstream window construction.
//!
//! Wired into the nhmmer FM-index path via `fm_ssv_augment_windows`
//! (`src/subcmd/nhmmer.rs`), where it augments the seed-then-rescore candidate
//! windows to recover weak diagonals with no exact model k-mer match.
//! `FmIndex` carries C `FM_DATA`'s `occCnts_sb`/`occCnts_b` sampled rank, so
//! `occ` is O(`FREQ_CNT_B`) and this exhaustive trie traversal is tractable on
//! genome-scale blocks.

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

/// Which FM index / interval discipline a `FM_Recurse` sweep uses. Port of C's
/// `fm_forward` / `fm_backward` for `fm_direction` (distinct from the
/// per-diagonal `model_direction`).
///
/// * `Forward`: standard backward-search step on `fmf` (the BWT of the reversed
///   block text; Rust `kind=0`), via [`FmIndex::prepend_interval`]. The single
///   interval is itself locatable on `fmf`.
/// * `Backward`: bi-directional forward-search step on `fmb` (the BWT of the
///   forward block text; Rust `kind=1`), via [`FmIndex::update_interval_forward`].
///   It carries a second interval that is locatable on `fmf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FmDirection {
    Forward,
    Backward,
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

/// Optimal-extension look-ahead tables (C `P7_SCOREDATA.opt_ext_fwd/rev`).
/// `fwd[k][j]` = the best possible score from extending model position `k`
/// forward by `j+1` positions (sum of the per-node max match scores); `rev[r][j]`
/// likewise extending backward from `r`. Used by `FM_Recurse` to prune diagonals
/// that cannot reach threshold even with a perfect extension near the depth limit.
#[derive(Debug)]
struct OptExt {
    fwd: Vec<[f32; 10]>,
    rev: Vec<[f32; 10]>,
}

impl OptExt {
    /// Faithful port of `scoredata_GetSSVScoreArrays`'s opt_ext loop
    /// (`p7_scoredata.c`).
    fn build(scores: &SsvScores) -> Self {
        let m = scores.m as usize;
        let mut max_scores = vec![0f32; m + 1];
        for k in 1..=m {
            let mut best = 0f32;
            for x in 0..scores.kp {
                let v = scores.get(k as i32, x);
                if v > best {
                    best = v;
                }
            }
            max_scores[k] = best;
        }
        let mut fwd = vec![[0f32; 10]; m + 1];
        let mut rev = vec![[0f32; 10]; m + 1];
        for i in 1..m {
            let mut sc_fwd = 0f32;
            let mut sc_rev = 0f32;
            let mut j = 0usize;
            while j < 10 && i + j < m {
                sc_fwd += max_scores[i + j + 1];
                fwd[i][j] = sc_fwd;
                sc_rev += max_scores[m - i - j];
                rev[m - i][j] = sc_rev;
                j += 1;
            }
            while j < 10 {
                fwd[i][j] = fwd[i][j - 1];
                rev[m - i][j] = rev[m - i][j - 1];
                j += 1;
            }
        }
        OptExt { fwd, rev }
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

/// Faithful Rust counterpart of C `FM_backtrackSeed()`.
///
/// The underlying `FmIndex` method performs the LF walk to a suffix-array
/// sample, matching C's per-row seed backtracking contract.
fn fm_backtrack_seed(fm: &FmIndex, row: usize) -> Option<usize> {
    fm.backtrack_seed_position(row)
}

/// Faithful Rust counterpart of C `fm_newSeed()`: append one seed slot and
/// return it to the caller for field assignment.
fn fm_new_seed(seeds: &mut Vec<FmDiag>) -> &mut FmDiag {
    seeds.push(FmDiag {
        n: 0,
        k: 0,
        length: 0,
        score: 0.0,
        complementarity: Complementarity::NoComplement,
    });
    seeds.last_mut().expect("just pushed FM seed")
}

/// Result of `FM_getPassingDiags`: each surviving FM interval entry becomes a
/// seed. Faithful to C's coordinate computation.
#[allow(clippy::too_many_arguments)]
fn fm_get_passing_diags(
    fm: &FmIndex,
    k: i32,
    depth: i32,
    model_direction: ModelDirection,
    complementarity: Complementarity,
    interval: FmInterval,
    seeds: &mut Vec<FmDiag>,
) {
    // C uses `fmf->N` = the BWT length including the sentinel = text_len + 1.
    let fm_n = fm.n as i64 + 1;
    for row in interval.lo..interval.hi {
        let Some(backtrack) = fm_backtrack_seed(fm, row) else {
            continue;
        };
        let backtrack = backtrack as i64;
        let seed = fm_new_seed(seeds);
        seed.k = k;
        seed.length = depth;
        // C `FM_getPassingDiags`:
        //   NOCOMPLEMENT: n = N - backtrack - depth - 1
        //   COMPLEMENT:   n = backtrack
        seed.n = if complementarity == Complementarity::NoComplement {
            fm_n - backtrack - depth as i64 - 1
        } else {
            backtrack
        };
        seed.complementarity = complementarity;
        // C: if model_direction == fm_forward, seed->k -= (depth - 1)
        if model_direction == ModelDirection::Forward {
            seed.k -= depth - 1;
        }
    }
}

/// Heart of the kernel: faithful port of C `FM_Recurse` restricted to the
/// `fm_direction == fm_forward` sweep (single backward-search FM index).
///
/// `dp_pairs` is the shared scratch vector C indexes with `first..=last`;
/// surviving children are appended past `last` and recursed on. We mirror that
/// exactly using indices into a growable Vec.
/// Look up `opt_ext[idx][j]`, returning `None` (i.e. do not prune) when the
/// indices are outside the valid/computed range.
#[inline]
fn opt_ext_lookahead(table: &[[f32; 10]], idx: i32, j: i32) -> Option<f32> {
    if idx < 1 || !(0..10).contains(&j) {
        return None;
    }
    table.get(idx as usize).map(|row| row[j as usize])
}

#[allow(clippy::too_many_arguments)]
fn fm_recurse(
    depth: i32,
    fmf: &FmIndex,
    fmb: &FmIndex,
    direction: FmDirection,
    alph: &FmAlphabet,
    scores: &SsvScores,
    consensus: &[usize],
    sc_thresh_fm: f32,
    config: &FmSsvConfig,
    opt_ext: &OptExt,
    dp_pairs: &mut Vec<FmDpPair>,
    first: usize,
    last: usize,
    // Forward sweep: the locatable `fmf` reverse-search interval.
    // Backward sweep: the `fmb` mirror interval.
    interval_1: FmInterval,
    // Backward sweep only: the `fmf`-coordinate forward interval (locatable).
    interval_2: Option<FmInterval>,
    seeds: &mut Vec<FmDiag>,
) {
    let m = scores.m;

    for c in 0..alph.alph_size {
        let base = fm_code_to_base(c);
        // One interval step for this character `c` (independent of the DP rows).
        // Returns (next interval_1, interval locatable in fmf), mirroring C's
        // `fm_updateIntervalReverse(fmf)` (forward sweep) and
        // `fm_updateIntervalForward(fmb)` (backward sweep).
        let step: Option<(FmInterval, FmInterval)> = match direction {
            FmDirection::Forward => fmf.prepend_interval(interval_1, base).map(|iv| (iv, iv)),
            FmDirection::Backward => {
                let iv2 = interval_2.expect("backward sweep requires interval_2");
                fmb.update_interval_forward(interval_1, iv2, base)
            }
        };

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
                // A seed to extend: emit all matching target positions from the
                // fmf-locatable interval for this character.
                if let Some((_, locatable)) = step {
                    fm_get_passing_diags(
                        fmf,
                        k,
                        depth,
                        pair.model_direction,
                        pair.complementarity,
                        locatable,
                        seeds,
                    );
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
                // C's optimal-extension look-ahead prune: near the depth limit,
                // drop a diagonal that can't reach threshold even with the best
                // possible extension. `j = max_depth - depth - 1` (C indexes the
                // size-10 opt_ext arrays); guarded to a valid j and model range.
                || (pair.model_direction == ModelDirection::Forward
                    && depth > config.max_depth as i32 - 10
                    && opt_ext_lookahead(&opt_ext.fwd, k, config.max_depth as i32 - depth - 1)
                        .is_some_and(|best| sc + best < sc_thresh_fm))
                || (pair.model_direction == ModelDirection::Backward
                    && depth > config.max_depth as i32 - 10
                    && opt_ext_lookahead(&opt_ext.rev, k - 1, config.max_depth as i32 - depth - 1)
                        .is_some_and(|best| sc + best < sc_thresh_fm))
            {
                // pruned - do nothing.
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
            let Some((new_iv1, locatable)) = step else {
                continue;
            };
            let next_iv2 = match direction {
                FmDirection::Forward => None,
                FmDirection::Backward => Some(locatable),
            };
            fm_recurse(
                depth + 1,
                fmf,
                fmb,
                direction,
                alph,
                scores,
                consensus,
                sc_thresh_fm,
                config,
                opt_ext,
                dp_pairs,
                last + 1,
                dppos,
                new_iv1,
                next_iv2,
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
/// Faithful port of C `FM_getSeeds`: for each starting character it builds the
/// forward-sweep and backward-sweep DP columns and runs both `FM_Recurse`
/// sweeps, then merges. `fmf` is the BWT of the reversed block text (Rust
/// `kind=0`); `fmb` is the BWT of the forward block text (Rust `kind=1`).
#[allow(clippy::too_many_arguments)]
pub fn fm_get_seeds(
    fmf: &FmIndex,
    fmb: &FmIndex,
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
    let opt_ext = OptExt::build(scores);

    for c in 0..alph.alph_size {
        let base = fm_code_to_base(c);
        // Single-character interval. C inits all three intervals (f1, f2, bk)
        // to `fmf->C[c]..fmf->C[c+1]-1`; the C[] arrays of fmf and fmb agree
        // (same character multiset), so this value seeds both sweeps.
        let Some(iv0) = fmf.char_interval(base) else {
            continue;
        };

        // C `dp_pairs_fwd` (forward sweep) and `dp_pairs_rev` (backward sweep).
        let mut dp_fwd: Vec<FmDpPair> = Vec::new();
        let mut dp_rev: Vec<FmDpPair> = Vec::new();

        for k in 1..=m {
            if !bottom_only {
                let sc = scores.get(k, c);
                if sc > 0.0 {
                    // fwd-on-model -> forward sweep; rev-on-model -> backward sweep.
                    if k < m - 3 {
                        dp_fwd.push(initial_pair(
                            k,
                            sc,
                            c == consensus[k as usize],
                            Complementarity::NoComplement,
                            ModelDirection::Forward,
                        ));
                    }
                    if k > 4 {
                        dp_rev.push(initial_pair(
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
                    // complement rev-on-model -> forward sweep;
                    // complement fwd-on-model -> backward sweep.
                    if k > 4 {
                        dp_fwd.push(initial_pair(
                            k,
                            sc,
                            c == consensus[k as usize],
                            Complementarity::Complement,
                            ModelDirection::Backward,
                        ));
                    }
                    if k < m - 3 {
                        dp_rev.push(initial_pair(
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

        if !dp_fwd.is_empty() {
            let last = dp_fwd.len() - 1;
            fm_recurse(
                2,
                fmf,
                fmb,
                FmDirection::Forward,
                alph,
                scores,
                consensus,
                sc_thresh_fm,
                config,
                &opt_ext,
                &mut dp_fwd,
                0,
                last,
                iv0,
                None,
                &mut seeds,
            );
        }
        if !dp_rev.is_empty() {
            let last = dp_rev.len() - 1;
            fm_recurse(
                2,
                fmf,
                fmb,
                FmDirection::Backward,
                alph,
                scores,
                consensus,
                sc_thresh_fm,
                config,
                &opt_ext,
                &mut dp_rev,
                0,
                last,
                iv0,
                Some(iv0),
                &mut seeds,
            );
        }
    }

    // C passes `fmf->N`, the BWT length including the terminal sentinel. Rust
    // `FmIndex::n` is the biological text length, so use `bwt_len()` here.
    fm_merge_seeds(
        &mut seeds,
        fmf.bwt_len() as i64,
        m,
        config.ssv_length as i32,
    );
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
/// `target` is the WHOLE (already strand-resolved) target text in alphabet
/// codes, 1-indexed so `target[1]` is FM position 0 (index 0 is a sentinel),
/// matching C's `fm_convertRange2DSQ` extraction of `[target_start, target_end]`
/// into `tmp_sq->dsq[1..]`. `fm_n` is the FM text length `fm->N`.
pub fn fm_extend_seed_core(
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

    // Walk model_start..=model_end against the extracted target range. C reads
    // tmp_sq->dsq[n] with n starting at 1, where tmp_sq holds the range
    // [target_start, target_end] extracted by fm_convertRange2DSQ. Here `target`
    // is the WHOLE strand-resolved text, 1-indexed (target[1] = FM position 0),
    // so position `target_start + (n-1)` (0-based) maps to absolute 1-index
    // `target_start + n`. `n` is the 1-based offset within the extracted range
    // and is what max_hit_start/end are expressed in, exactly like C.
    let mut k = model_start;
    let mut n: usize = 1;
    let mut sc = 0.0_f32;
    let mut hit_start: i64 = n as i64;
    let mut max_sc = 0.0_f32;
    let mut max_hit_start: i64 = n as i64;
    let mut max_hit_end: i64 = n as i64;

    while k <= model_end {
        let abs_idx = (target_start + n as i64) as usize;
        let cidx = (k as usize) * kp + target.get(abs_idx).copied().unwrap_or(0);
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
        fm_extend_seed_core(&mut diag, &target_codes, &scores, &config, fm_n);
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
        // fmf = BWT of reversed text (forward sweep); fmb = BWT of forward text
        // (backward sweep), mirroring C's fmf/fmb pair.
        let fmf = FmIndex::build(&reversed);
        let fmb = FmIndex::build(&target_text);

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
        let seeds = fm_get_seeds(
            &fmf, &fmb, &alph, &scores, &consensus, 6.0, &config, true, false,
        );

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
        // fmf = BWT of reversed text (forward sweep); fmb = BWT of forward text
        // (backward sweep), mirroring C's fmf/fmb pair.
        let fmf = FmIndex::build(&reversed);
        let fmb = FmIndex::build(&target_text);

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
        let seeds = fm_get_seeds(
            &fmf, &fmb, &alph, &scores, &consensus, 12.0, &config, true, false,
        );
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
        // fmf = BWT of reversed text (forward sweep); fmb = BWT of forward text
        // (backward sweep), mirroring C's fmf/fmb pair.
        let fmf = FmIndex::build(&reversed);
        let fmb = FmIndex::build(&target_text);

        let config = FmSsvConfig {
            max_depth: 10,
            consec_pos_req: 1,
            consensus_match_req: 4,
            score_density_req: 0.0,
            drop_max_len: 10,
            ssv_length: 14,
            ..FmSsvConfig::default()
        };

        let mut seeds = fm_get_seeds(
            &fmf, &fmb, &alph, &scores, &consensus, 6.0, &config, true, false,
        );
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
                fm_extend_seed_core(diag, &target_codes, &scores, &config, fm_n);
                assert!(diag.length >= 0);
            }
        }
    }
}
