//! P7_PROFILE - Scoring profile for sequence comparison.
//! Direct port of p7_profile.c and modelconfig.c.

#![allow(clippy::needless_range_loop)]

use crate::alphabet::Alphabet;
use crate::bg::Bg;
use crate::errors::{HmmerError, HmmerResult};
use crate::hmm::*;
use crate::util::cmath::{c_log_f32_to_f32, c_log_f64, c_log_to_f32, ESL_CONST_LOG2};

/// Tracehash hook: record the per-node B→Mk entry-score derivation so it can
/// be diff'd against the C reference. Active only under the `tracehash` feature.
#[cfg(feature = "tracehash")]
fn trace_profile_entry_source(m: usize, k: usize, occ: f32, z: f32, ratio: f32, score: f32) {
    let mut th = tracehash::th_call!("profile_entry_source_bits");
    th.input_usize(m);
    th.input_usize(k);
    th.output_u64(occ.to_bits() as u64);
    th.output_u64(z.to_bits() as u64);
    th.output_u64(ratio.to_bits() as u64);
    th.output_u64(score.to_bits() as u64);
    th.finish();
}

// Profile transition indices (different ordering from HMM transitions)
pub const P7P_MM: usize = 0;
pub const P7P_IM: usize = 1;
pub const P7P_DM: usize = 2;
pub const P7P_BM: usize = 3;
pub const P7P_MD: usize = 4;
pub const P7P_DD: usize = 5;
pub const P7P_MI: usize = 6;
pub const P7P_II: usize = 7;
pub const P7P_NTRANS: usize = 8;

// Emission score indices
pub const P7P_MSC: usize = 0;
pub const P7P_ISC: usize = 1;
pub const P7P_NR: usize = 2;

// Special state indices
pub const P7P_E: usize = 0;
pub const P7P_N: usize = 1;
pub const P7P_J: usize = 2;
pub const P7P_C: usize = 3;
pub const P7P_NXSTATES: usize = 4;

// Special transition indices
pub const P7P_LOOP: usize = 0;
pub const P7P_MOVE: usize = 1;
pub const P7P_NXTRANS: usize = 2;

// Search modes
/// Plan7 local, multihit search: each domain may match part of the HMM,
/// and a target sequence may contain multiple domains.
/// `P7` refers to the Plan7 profile HMM architecture.
pub const P7_LOCAL: i32 = 1;
/// Plan7 glocal, multihit search: each domain spans the full HMM,
/// and a target sequence may contain multiple domains. Target flanks are allowed.
/// `P7` refers to the Plan7 profile HMM architecture.
pub const P7_GLOCAL: i32 = 2;
/// Plan7 local, unihit search: at most one domain per target sequence,
/// matching any part of the HMM. Target flanks are allowed.
/// `P7` refers to the Plan7 profile HMM architecture.
pub const P7_UNILOCAL: i32 = 3;
/// Plan7 glocal, unihit search: at most one domain per target sequence,
/// spanning the full HMM. Target flanks are allowed.
/// `P7` refers to the Plan7 profile HMM architecture.
pub const P7_UNIGLOCAL: i32 = 4;

fn check_distribution(probabilities: &[f32], label: &str) -> HmmerResult<()> {
    let sum: f64 = probabilities.iter().map(|&p| p as f64).sum();
    if probabilities
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || (sum - 1.0).abs() > 0.001
    {
        return Err(HmmerError::InvalidArg(format!(
            "{label} must be finite probabilities summing to one"
        )));
    }
    Ok(())
}

/// Alignment coverage and number of domains allowed per target.
///
/// Used by [`Profile::from_hmm`]; the existing integer `P7_*` modes remain
/// available for low-level APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Match part of the model; allow multiple domains per target.
    LocalMultihit,
    /// Span the full model; allow multiple domains per target.
    GlocalMultihit,
    /// Match part of the model; allow at most one domain per target.
    LocalUnihit,
    /// Span the full model; allow at most one domain per target.
    GlocalUnihit,
}

impl From<SearchMode> for i32 {
    fn from(mode: SearchMode) -> Self {
        match mode {
            SearchMode::LocalMultihit => P7_LOCAL,
            SearchMode::GlocalMultihit => P7_GLOCAL,
            SearchMode::LocalUnihit => P7_UNILOCAL,
            SearchMode::GlocalUnihit => P7_UNIGLOCAL,
        }
    }
}

/// Scoring profile derived from an HMM, with natural-log scores in nats.
///
/// Allocate with [`Profile::new`], then initialize with [`profile_config`].
/// Target sequence length and HMM model length are separate parameters.
#[derive(Debug, Clone)]
pub struct Profile {
    /// Transition scores: `tsc[k * P7P_NTRANS + s]` for node k, transition s
    pub tsc: Vec<f32>,
    /// Emission scores: `rsc[x]` is a vec of (M+1)*P7P_NR entries for residue code x.
    /// `rsc[x][k * P7P_NR + P7P_MSC]` = match emission score at node k for residue x
    /// `rsc[x][k * P7P_NR + P7P_ISC]` = insert emission score at node k for residue x
    pub rsc: Vec<Vec<f32>>,
    /// Special state transitions `[NECJ][LOOP,MOVE]`
    pub xsc: [[f32; P7P_NXTRANS]; P7P_NXSTATES],

    /// Search mode: one of the `P7_LOCAL`, `P7_GLOCAL`, or `P7_UNI*` constants.
    pub mode: i32,
    /// Configured target length in residues, excluding sentinels.
    pub l: i32,
    /// Number of model match nodes, independent of the target length.
    pub m: usize,
    /// Expected number of uses of the between-domain J state.
    pub nj: f32,

    // Annotation (copied from HMM)
    pub name: String,
    pub acc: Option<String>,
    pub desc: Option<String>,
    pub rf: Vec<u8>,
    pub mm: Vec<u8>,
    pub cs: Vec<u8>,
    pub consensus: Vec<u8>,

    pub evparam: [f32; NEVPARAM],
    pub cutoff: [f32; NCUTOFFS],
    pub compo: [f32; MAXABET],

    pub max_length: i32,

    /// Alphabet total size (Kp)
    pub abc_kp: usize,
    /// Alphabet canonical size (K)
    pub abc_k: usize,
}

impl Profile {
    /// Allocate and configure a scoring profile with checked inputs.
    ///
    /// `target_length` counts residues, excluding sentinels, and must fit an
    /// `i32`. `mode` selects model coverage and the permitted number of domains.
    /// Checks the alphabet, table dimensions, core probability distributions,
    /// background frequencies, and positive local match occupancy. Returns
    /// [`HmmerError::InvalidArg`] for invalid model/background data or
    /// [`HmmerError::Range`] for an unsupported length.
    ///
    /// Calibration is not required for constructing a profile. Use
    /// [`crate::Searcher`] for calibrated database searches. This does not
    /// change `bg`'s length model; low-level callers must set it separately.
    pub fn from_hmm(
        hmm: &Hmm,
        bg: &Bg,
        target_length: usize,
        mode: SearchMode,
    ) -> HmmerResult<Self> {
        let length = i32::try_from(target_length)
            .map_err(|_| HmmerError::Range("target length exceeds i32::MAX residues".into()))?;
        if hmm.m == 0 || hmm.m > i32::MAX as usize {
            return Err(HmmerError::Range(
                "model length must be in 1..=i32::MAX match states".into(),
            ));
        }
        if hmm.abc_type == crate::alphabet::AlphabetType::Unknown {
            return Err(HmmerError::InvalidArg("model alphabet is unknown".into()));
        }
        let abc = Alphabet::new(hmm.abc_type);
        if hmm.abc_k != abc.k || bg.abc_type != abc.abc_type || bg.k != abc.k {
            return Err(HmmerError::InvalidArg(
                "model and background alphabets must match".into(),
            ));
        }
        if bg.f.len() != abc.k || bg.f.iter().any(|p| *p <= 0.0) {
            return Err(HmmerError::InvalidArg(
                "background must have one positive frequency per canonical residue".into(),
            ));
        }
        check_distribution(&bg.f, "background frequencies")?;
        if hmm.t.len() <= hmm.m || hmm.mat.len() <= hmm.m {
            return Err(HmmerError::InvalidArg(
                "model probability tables are too short".into(),
            ));
        }
        for row in &hmm.mat[1..=hmm.m] {
            if row.len() != abc.k {
                return Err(HmmerError::InvalidArg(
                    "model emission row has wrong alphabet size".into(),
                ));
            }
            check_distribution(row, "match emissions")?;
        }
        for row in &hmm.t[..hmm.m] {
            check_distribution(&row[MM..=MD], "match transitions")?;
            check_distribution(&row[IM..=II], "insert transitions")?;
            check_distribution(&row[DM..=DD], "delete transitions")?;
        }
        if matches!(mode, SearchMode::LocalMultihit | SearchMode::LocalUnihit)
            && !hmm_calculate_occupancy(hmm).iter().any(|p| *p > 0.0)
        {
            return Err(HmmerError::InvalidArg(
                "model has no reachable match states".into(),
            ));
        }
        let mut profile = Self::new(hmm.m, &abc);
        profile_config(hmm, bg, &mut profile, length, mode.into());
        Ok(profile)
    }

    /// Allocate a profile sized for an HMM of up to `alloc_m` nodes
    /// (port of `p7_profile_Create`).
    ///
    /// All transition/emission scores are initialised to `-inf`; metadata
    /// (name/acc/desc/annotation arrays) is empty. The caller is expected to
    /// populate scores via [`profile_config`].
    pub fn new(alloc_m: usize, abc: &Alphabet) -> Self {
        let kp = abc.kp;
        let k = abc.k;
        let tsc = vec![f32::NEG_INFINITY; (alloc_m + 1) * P7P_NTRANS];
        let rsc = vec![vec![f32::NEG_INFINITY; (alloc_m + 2) * P7P_NR]; kp];
        let xsc = [[f32::NEG_INFINITY; P7P_NXTRANS]; P7P_NXSTATES];

        Profile {
            tsc,
            rsc,
            xsc,
            mode: 0,
            l: 0,
            m: 0,
            nj: 0.0,
            name: String::new(),
            acc: None,
            desc: None,
            rf: vec![0; alloc_m + 2],
            mm: vec![0; alloc_m + 2],
            cs: vec![0; alloc_m + 2],
            consensus: vec![0; alloc_m + 2],
            evparam: [EVPARAM_UNSET; NEVPARAM],
            cutoff: [CUTOFF_UNSET; NCUTOFFS],
            compo: [COMPO_UNSET; MAXABET],
            max_length: -1,
            abc_kp: kp,
            abc_k: k,
        }
    }

    /// Read transition score `tsc[k * P7P_NTRANS + s]` for node `k`,
    /// transition class `s` (e.g. `P7P_MM`, `P7P_BM`).
    #[inline]
    ///
    /// Returns a natural-log score (nats). `k` is a model table index; `s` must
    /// be a `P7P_*` transition index below [`P7P_NTRANS`]. For B→Mk entry,
    /// use `k - 1` with [`P7P_BM`]. Panics if the flattened index is out of bounds.
    pub fn tsc(&self, k: usize, s: usize) -> f32 {
        self.tsc[k * P7P_NTRANS + s]
    }

    /// Write transition score `tsc[k * P7P_NTRANS + s] = val`.
    #[inline]
    ///
    /// `val` is in nats; use negative infinity for an impossible transition.
    /// `k` and `s` use the indexing described by [`Self::tsc`]. Panics if the
    /// flattened index is outside the allocated transition table.
    pub fn set_tsc(&mut self, k: usize, s: usize, val: f32) {
        self.tsc[k * P7P_NTRANS + s] = val;
    }

    /// Read the match emission score at node `k` for residue code `x`.
    #[inline]
    ///
    /// `k` is a 1-based model node (`1..=self.m`); `x` is a digital residue
    /// code from the profile's alphabet, below `self.abc_kp`. Returns a score
    /// in nats, with negative infinity for an impossible emission. Panics if
    /// an index exceeds the allocated score tables.
    pub fn msc(&self, k: usize, x: usize) -> f32 {
        self.rsc[x][k * P7P_NR + P7P_MSC]
    }

    /// Read the insert emission score at node `k` for residue code `x`.
    #[inline]
    ///
    /// `k` is a 1-based model node (`1..=self.m`); `x` is a digital residue
    /// code from the profile's alphabet, below `self.abc_kp`. Returns a score
    /// in nats, with negative infinity for an impossible emission. Panics if
    /// an index exceeds the allocated score tables.
    pub fn isc(&self, k: usize, x: usize) -> f32 {
        self.rsc[x][k * P7P_NR + P7P_ISC]
    }

    /// True if the profile is in local (vs. glocal) alignment mode (port of
    /// `p7_profile_IsLocal`).
    ///
    /// Local alignment may begin/end inside the model; glocal alignment spans
    /// its full length. This checks `mode`, not the current special-state scores.
    pub fn is_local(&self) -> bool {
        self.mode == P7_LOCAL || self.mode == P7_UNILOCAL
    }

    /// True if the profile is configured for multihit alignment (port of
    /// `p7_profile_IsMultihit`).
    ///
    /// Multihit mode permits multiple domains in one target sequence. This checks
    /// `mode`; [`reconfig_unihit`] and [`reconfig_multihit`] do not update that field.
    pub fn is_multihit(&self) -> bool {
        self.mode == P7_LOCAL || self.mode == P7_GLOCAL
    }
}

/// Compute per-node match-state occupancy `occ[0..=M]` for a core HMM
/// (port of `p7_hmm_CalculateOccupancy`).
///
/// `occ[k]` is the probability that match state `Mk` is used at least once
/// in a path through the model. Used by [`profile_config`] to set the local
/// entry distribution (B→Mk) so that match states are entered in proportion
/// to their occupancy.
///
/// Returns `hmm.m + 1` probabilities, with index 0 unused. Requires a
/// nonempty core HMM with probability tables; a zero-node model panics.
pub fn hmm_calculate_occupancy(hmm: &Hmm) -> Vec<f32> {
    let mut mocc = vec![0.0_f32; hmm.m + 1];
    mocc[0] = 0.0;
    mocc[1] = hmm.t[0][MI] + hmm.t[0][MM]; // 1 - B->D_1
    for k in 2..=hmm.m {
        let prev = mocc[k - 1];
        let match_or_insert = prev * (hmm.t[k - 1][MM] + hmm.t[k - 1][MI]);
        let delete_entry = (1.0_f64 - prev as f64) * hmm.t[k - 1][DM] as f64;
        mocc[k] = (match_or_insert as f64 + delete_entry) as f32;
    }
    mocc
}

/// Compute the occupancy-weighted average match-state composition.
///
/// This is the composition returned by C's `p7_hmm_CompositionKLD()` when its
/// optional composition vector is requested. It intentionally uses match
/// emissions only, not insert emissions.
///
/// Returns `hmm.abc_k` values in canonical alphabet order, normalized when
/// the total match occupancy is positive. Requires a nonempty core HMM.
pub fn hmm_average_match_composition(hmm: &Hmm) -> Vec<f32> {
    let occ = hmm_calculate_occupancy(hmm);
    let mut avg = vec![0.0_f32; hmm.abc_k];
    for k in 1..=hmm.m {
        for x in 0..hmm.abc_k {
            avg[x] += hmm.mat[k][x] * occ[k];
        }
    }

    let sum: f32 = avg.iter().sum();
    if sum > 0.0 {
        for p in &mut avg {
            *p /= sum;
        }
    }
    avg
}

/// Expected score for a degenerate residue code `x`, weighting the canonical
/// scores `sc[0..K-1]` by background frequencies `p`. Mirrors
/// `esl_abc_FAvgScore` semantics for IUPAC degeneracies.
fn f_expect_score(abc: &Alphabet, x: usize, sc: &[f32], p: &[f32]) -> f32 {
    if !abc.is_residue(x as u8) {
        return 0.0;
    }
    let mut result = 0.0_f32;
    let mut denom = 0.0_f32;
    for i in 0..abc.k {
        if abc.degen[x][i] {
            result += sc[i] * p[i];
            denom += p[i];
        }
    }
    if denom > 0.0 {
        result / denom
    } else {
        0.0
    }
}

/// Populate the degenerate-residue entries (`K+1..=Kp-3`) of an emission
/// score vector from the canonical scores using [`f_expect_score`].
fn f_expect_sc_vec(abc: &Alphabet, sc: &mut [f32], p: &[f32]) {
    for x in (abc.k + 1)..=(abc.kp - 3) {
        sc[x] = f_expect_score(abc, x, sc, p);
    }
}

/// Configure a search profile from an HMM, background, mode and target
/// length (port of `p7_ProfileConfig`).
///
/// Computes lod-score transition and emission tables relative to `bg`,
/// installs B→Mk entry scores (local mode uses occupancy ratios, glocal mode
/// uses left-wing retraction), sets E-state multihit/unihit transitions, then
/// finalises the length distribution via [`reconfig_length`]. Annotation
/// (name/acc/desc/rf/mm/cs/cons/cutoff/compo/evparam) is copied across.
///
/// # Arguments
///
/// - `hmm`: Core HMM with probability tables and at least one match node.
/// - `bg`: Background residue frequencies for the same alphabet as `hmm`.
/// - `gm`: Output profile, allocated with [`Profile::new`] for this alphabet
///   and at least `hmm.m` nodes. Its scores and metadata are overwritten.
/// - `l`: Expected target sequence length in residues (amino acids or
///   nucleotides), excluding digital-sequence sentinels. Use the actual
///   sequence length when scoring a known target; this is distinct from
///   the model length `hmm.m`. Must be nonnegative.
/// - `mode`: [`P7_LOCAL`] or [`P7_GLOCAL`] for multiple domains per target,
///   or [`P7_UNILOCAL`] or [`P7_UNIGLOCAL`] for at most one domain.
///   Local modes allow partial model matches; glocal modes span the model.
///
/// When constructing a profile before reading targets, choose an initial
/// expected length, then call [`reconfig_length`] for each target. Set the
/// background length separately with [`Bg::set_length`]; this function does
/// not change `bg` or an already converted optimized profile.
///
/// # Panics
///
/// Panics if the output score buffers are too small or the HMM has no nodes.
/// Alphabet compatibility and valid mode/length values are caller prerequisites
/// and are not checked.
pub fn profile_config(hmm: &Hmm, bg: &Bg, gm: &mut Profile, l: i32, mode: i32) {
    let abc = Alphabet::new(hmm.abc_type);

    gm.m = hmm.m;
    gm.max_length = hmm.max_length;
    gm.mode = mode;

    // Copy metadata
    gm.name = hmm.name.clone();
    gm.acc = hmm.acc.clone();
    gm.desc = hmm.desc.clone();
    if let Some(ref rf) = hmm.rf {
        let len = rf.len().min(gm.rf.len());
        gm.rf[..len].copy_from_slice(&rf[..len]);
    }
    if let Some(ref mm) = hmm.mm {
        let len = mm.len().min(gm.mm.len());
        gm.mm[..len].copy_from_slice(&mm[..len]);
    }
    if let Some(ref cons) = hmm.consensus {
        let len = cons.len().min(gm.consensus.len());
        gm.consensus[..len].copy_from_slice(&cons[..len]);
    }
    if let Some(ref cs) = hmm.cs {
        let len = cs.len().min(gm.cs.len());
        gm.cs[..len].copy_from_slice(&cs[..len]);
    }
    gm.evparam = hmm.evparam;
    gm.cutoff = hmm.cutoff;
    gm.compo = hmm.compo;

    // Entry scores (B->Mk)
    if gm.is_local() {
        // Local mode: occ[k] / sum(occ[i] * (M-i+1))
        let occ = hmm_calculate_occupancy(hmm);
        let mut z = 0.0_f32;
        for k in 1..=hmm.m {
            z += occ[k] * (hmm.m - k + 1) as f32;
        }
        for k in 1..=hmm.m {
            let ratio = occ[k] / z;
            let score = c_log_f32_to_f32(ratio);
            #[cfg(feature = "tracehash")]
            trace_profile_entry_source(hmm.m, k, occ[k], z, ratio, score);
            gm.set_tsc(k - 1, P7P_BM, score);
        }
    } else {
        // Glocal mode: left wing retraction
        let mut z = c_log_f32_to_f32(hmm.t[0][MD]);
        gm.set_tsc(0, P7P_BM, c_log_to_f32(1.0 - hmm.t[0][MD] as f64));
        for k in 1..hmm.m {
            gm.set_tsc(
                k,
                P7P_BM,
                (z as f64 + c_log_f64(hmm.t[k][DM] as f64)) as f32,
            );
            z = (z as f64 + c_log_f64(hmm.t[k][DD] as f64)) as f32;
        }
    }

    // E state transitions
    if gm.is_multihit() {
        gm.xsc[P7P_E][P7P_MOVE] = -(ESL_CONST_LOG2 as f32); // -log(2)
        gm.xsc[P7P_E][P7P_LOOP] = -(ESL_CONST_LOG2 as f32);
        gm.nj = 1.0;
    } else {
        gm.xsc[P7P_E][P7P_MOVE] = 0.0;
        gm.xsc[P7P_E][P7P_LOOP] = f32::NEG_INFINITY;
        gm.nj = 0.0;
    }

    // Transition scores (nodes 1..M-1) — use f64 log to match C's log()
    for k in 1..gm.m {
        gm.set_tsc(k, P7P_MM, c_log_f32_to_f32(hmm.t[k][MM]));
        gm.set_tsc(k, P7P_MI, c_log_f32_to_f32(hmm.t[k][MI]));
        gm.set_tsc(k, P7P_MD, c_log_f32_to_f32(hmm.t[k][MD]));
        gm.set_tsc(k, P7P_IM, c_log_f32_to_f32(hmm.t[k][IM]));
        gm.set_tsc(k, P7P_II, c_log_f32_to_f32(hmm.t[k][II]));
        gm.set_tsc(k, P7P_DM, c_log_f32_to_f32(hmm.t[k][DM]));
        gm.set_tsc(k, P7P_DD, c_log_f32_to_f32(hmm.t[k][DD]));
    }

    // Match emission scores
    let mut sc = vec![0.0_f32; abc.kp];
    sc[abc.k] = f32::NEG_INFINITY; // gap
    sc[abc.kp - 2] = f32::NEG_INFINITY; // nonresidue
    sc[abc.kp - 1] = f32::NEG_INFINITY; // missing data

    for k in 1..=hmm.m {
        for x in 0..abc.k {
            // Match C: log((double)mat[k][x] / bg->f[x]) — double precision division + log
            sc[x] = c_log_to_f32((hmm.mat[k][x] as f64) / (bg.f[x] as f64));
        }
        f_expect_sc_vec(&abc, &mut sc, &bg.f);

        for x in 0..abc.kp {
            gm.rsc[x][k * P7P_NR + P7P_MSC] = sc[x];
        }
    }

    // Insert emission scores (hardwired to 0.0 = background)
    for x in 0..abc.kp {
        for k in 1..hmm.m {
            gm.rsc[x][k * P7P_NR + P7P_ISC] = 0.0;
        }
        gm.rsc[x][hmm.m * P7P_NR + P7P_ISC] = f32::NEG_INFINITY; // I_M impossible
    }
    // Gap, nonresidue, missing data insert emissions = -inf
    for k in 1..=hmm.m {
        gm.rsc[abc.k][k * P7P_NR + P7P_ISC] = f32::NEG_INFINITY;
        gm.rsc[abc.kp - 2][k * P7P_NR + P7P_ISC] = f32::NEG_INFINITY;
        gm.rsc[abc.kp - 1][k * P7P_NR + P7P_ISC] = f32::NEG_INFINITY;
    }

    // Configure length model
    gm.l = 0;
    reconfig_length(gm, l);
}

/// Reset the target-length component of a configured profile to mean `L`
/// (port of `p7_ReconfigLength`).
///
/// Updates the N/J/C loop/move transitions so they bear `L/(2+nj)` of the
/// unannotated length budget without recomputing the rest of the profile.
/// Designed to be cheap because the search pipeline calls it once per target.
/// The null model length must be reset separately via `Bg::set_length`.
///
/// `l` is a nonnegative target length in residues, excluding sentinels,
/// usually [`crate::sequence::Sequence::n`]. This updates `gm.l` and the
/// probabilities of consuming unrelated residues before, between, and after
/// matching domains. It does not change the core model length or emissions.
/// The profile must already have been initialized by [`profile_config`].
/// An optimized profile derived from `gm` requires its own length update.
pub fn reconfig_length(gm: &mut Profile, l: i32) {
    let pmove = (2.0 + gm.nj) / (l as f32 + 2.0 + gm.nj);
    let ploop = 1.0 - pmove;
    gm.xsc[P7P_N][P7P_LOOP] = c_log_f32_to_f32(ploop);
    gm.xsc[P7P_C][P7P_LOOP] = c_log_f32_to_f32(ploop);
    gm.xsc[P7P_J][P7P_LOOP] = c_log_f32_to_f32(ploop);
    gm.xsc[P7P_N][P7P_MOVE] = c_log_f32_to_f32(pmove);
    gm.xsc[P7P_C][P7P_MOVE] = c_log_f32_to_f32(pmove);
    gm.xsc[P7P_J][P7P_MOVE] = c_log_f32_to_f32(pmove);
    gm.l = l;
}

/// Flip an already-configured profile into multihit mode at target length `L`
/// (port of `p7_ReconfigMultihit`).
///
/// Sets the E-state loop/move transitions to `log(0.5)` and `nj = 1`, then
/// calls [`reconfig_length`] because the length model depends on `nj`. Used
/// inside the domain-definition pipeline to flip in and out of unihit mode.
///
/// `gm` must already be configured; `l` is a nonnegative target sequence
/// length in residues, excluding sentinels. Allows additional domains through E→J.
/// This changes special-state scores and `nj`, but does not update `gm.mode`,
/// the background model, or an optimized profile derived from `gm`.
pub fn reconfig_multihit(gm: &mut Profile, l: i32) {
    gm.xsc[P7P_E][P7P_MOVE] = -(ESL_CONST_LOG2 as f32);
    gm.xsc[P7P_E][P7P_LOOP] = -(ESL_CONST_LOG2 as f32);
    gm.nj = 1.0;
    reconfig_length(gm, l);
}

/// Flip a configured profile into unihit mode at target length `L`
/// (port of `p7_ReconfigUnihit`).
///
/// Sets E→C to 0 and E→J to `-inf` (so J is unreachable), zeroes `nj`,
/// then refreshes the length model via [`reconfig_length`].
///
/// `gm` must already be configured; `l` is a nonnegative target sequence
/// length in residues, excluding sentinels. Allows at most one domain through E→C.
/// This changes special-state scores and `nj`, but does not update `gm.mode`,
/// the background model, or an optimized profile derived from `gm`.
pub fn reconfig_unihit(gm: &mut Profile, l: i32) {
    gm.xsc[P7P_E][P7P_MOVE] = 0.0;
    gm.xsc[P7P_E][P7P_LOOP] = f32::NEG_INFINITY;
    gm.nj = 0.0;
    reconfig_length(gm, l);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Test helper: load the first HMM from `testsuite/20aa.hmm`.
    fn load_test_hmm() -> Hmm {
        crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
    }

    #[test]
    fn test_profile_config_basic() {
        let hmm = load_test_hmm();
        let abc = Alphabet::new(hmm.abc_type);
        let bg = Bg::new(&abc);
        let mut gm = Profile::new(hmm.m, &abc);

        profile_config(&hmm, &bg, &mut gm, 400, P7_LOCAL);

        assert_eq!(gm.m, 20);
        assert_eq!(gm.mode, P7_LOCAL);
        assert_eq!(gm.l, 400);
        assert_eq!(gm.name, "test");
    }
}
