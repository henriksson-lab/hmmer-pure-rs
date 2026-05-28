//! P7_PROFILE - Scoring profile for sequence comparison.
//! Direct port of p7_profile.c and modelconfig.c.

#![allow(clippy::needless_range_loop)]

use crate::alphabet::Alphabet;
use crate::bg::Bg;
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
pub const P7_LOCAL: i32 = 1;
pub const P7_GLOCAL: i32 = 2;
pub const P7_UNILOCAL: i32 = 3;
pub const P7_UNIGLOCAL: i32 = 4;

/// Scoring profile derived from an HMM.
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

    pub mode: i32,
    pub l: i32,   // configured target length
    pub m: usize, // model length (nodes)
    pub nj: f32,  // expected # J uses

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
    pub fn tsc(&self, k: usize, s: usize) -> f32 {
        self.tsc[k * P7P_NTRANS + s]
    }

    /// Write transition score `tsc[k * P7P_NTRANS + s] = val`.
    #[inline]
    pub fn set_tsc(&mut self, k: usize, s: usize, val: f32) {
        self.tsc[k * P7P_NTRANS + s] = val;
    }

    /// Read the match emission score at node `k` for residue code `x`.
    #[inline]
    pub fn msc(&self, k: usize, x: usize) -> f32 {
        self.rsc[x][k * P7P_NR + P7P_MSC]
    }

    /// Read the insert emission score at node `k` for residue code `x`.
    #[inline]
    pub fn isc(&self, k: usize, x: usize) -> f32 {
        self.rsc[x][k * P7P_NR + P7P_ISC]
    }

    /// True if the profile is in local (vs. glocal) alignment mode (port of
    /// `p7_profile_IsLocal`).
    pub fn is_local(&self) -> bool {
        self.mode == P7_LOCAL || self.mode == P7_UNILOCAL
    }

    /// True if the profile is configured for multihit alignment (port of
    /// `p7_profile_IsMultihit`).
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
