//! P7_PROFILE - Scoring profile for sequence comparison.
//! Direct port of p7_profile.c and modelconfig.c.

use crate::alphabet::Alphabet;
use crate::bg::Bg;
use crate::hmm::*;

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
    /// Transition scores: tsc[k * P7P_NTRANS + s] for node k, transition s
    pub tsc: Vec<f32>,
    /// Emission scores: rsc[x] is a vec of (M+1)*P7P_NR entries for residue code x.
    /// rsc[x][k * P7P_NR + P7P_MSC] = match emission score at node k for residue x
    /// rsc[x][k * P7P_NR + P7P_ISC] = insert emission score at node k for residue x
    pub rsc: Vec<Vec<f32>>,
    /// Special state transitions [NECJ][LOOP,MOVE]
    pub xsc: [[f32; P7P_NXTRANS]; P7P_NXSTATES],

    pub mode: i32,
    pub l: i32,      // configured target length
    pub m: usize,    // model length (nodes)
    pub nj: f32,     // expected # J uses

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
    /// Create a new empty profile for a model of up to `alloc_m` nodes.
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

    /// Access a transition score: tsc(k, s)
    #[inline]
    pub fn tsc(&self, k: usize, s: usize) -> f32 {
        self.tsc[k * P7P_NTRANS + s]
    }

    /// Set a transition score: tsc(k, s) = val
    #[inline]
    pub fn set_tsc(&mut self, k: usize, s: usize, val: f32) {
        self.tsc[k * P7P_NTRANS + s] = val;
    }

    /// Access a match emission score
    #[inline]
    pub fn msc(&self, k: usize, x: usize) -> f32 {
        self.rsc[x][k * P7P_NR + P7P_MSC]
    }

    /// Access an insert emission score
    #[inline]
    pub fn isc(&self, k: usize, x: usize) -> f32 {
        self.rsc[x][k * P7P_NR + P7P_ISC]
    }

    pub fn is_local(&self) -> bool {
        self.mode == P7_LOCAL || self.mode == P7_UNILOCAL
    }

    pub fn is_multihit(&self) -> bool {
        self.mode == P7_LOCAL || self.mode == P7_GLOCAL
    }
}

/// Calculate match occupancy for an HMM.
/// Returns occ[0..M] where occ[k] = probability of match state k being used.
pub fn hmm_calculate_occupancy(hmm: &Hmm) -> Vec<f32> {
    let mut mocc = vec![0.0_f32; hmm.m + 1];
    mocc[0] = 0.0;
    mocc[1] = hmm.t[0][MI] + hmm.t[0][MM]; // 1 - B->D_1
    for k in 2..=hmm.m {
        mocc[k] = mocc[k - 1] * (hmm.t[k - 1][MM] + hmm.t[k - 1][MI])
            + (1.0 - mocc[k - 1]) * hmm.t[k - 1][DM];
    }
    mocc
}

/// Expected score for a degenerate residue, weighted by background frequencies.
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

/// Fill in degenerate residue scores from canonical scores and background frequencies.
fn f_expect_sc_vec(abc: &Alphabet, sc: &mut [f32], p: &[f32]) {
    for x in (abc.k + 1)..=(abc.kp - 3) {
        sc[x] = f_expect_score(abc, x, sc, p);
    }
}

/// Configure a profile from an HMM, background model, mode, and target length.
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
        let z: f32 = (1..=hmm.m)
            .map(|k| occ[k] * (hmm.m - k + 1) as f32)
            .sum();
        for k in 1..=hmm.m {
            gm.set_tsc(k - 1, P7P_BM, (occ[k] / z).ln()); // off-by-one: entry at Mk stored as [k-1][BM]
        }
    } else {
        // Glocal mode: left wing retraction
        let mut z = hmm.t[0][MD].ln();
        gm.set_tsc(0, P7P_BM, (1.0 - hmm.t[0][MD]).ln());
        for k in 1..hmm.m {
            gm.set_tsc(k, P7P_BM, z + hmm.t[k][DM].ln());
            z += hmm.t[k][DD].ln();
        }
    }

    // E state transitions
    if gm.is_multihit() {
        gm.xsc[P7P_E][P7P_MOVE] = -(2.0_f32.ln()); // -log(2)
        gm.xsc[P7P_E][P7P_LOOP] = -(2.0_f32.ln());
        gm.nj = 1.0;
    } else {
        gm.xsc[P7P_E][P7P_MOVE] = 0.0;
        gm.xsc[P7P_E][P7P_LOOP] = f32::NEG_INFINITY;
        gm.nj = 0.0;
    }

    // Transition scores (nodes 1..M-1)
    for k in 1..gm.m {
        gm.set_tsc(k, P7P_MM, hmm.t[k][MM].ln());
        gm.set_tsc(k, P7P_MI, hmm.t[k][MI].ln());
        gm.set_tsc(k, P7P_MD, hmm.t[k][MD].ln());
        gm.set_tsc(k, P7P_IM, hmm.t[k][IM].ln());
        gm.set_tsc(k, P7P_II, hmm.t[k][II].ln());
        gm.set_tsc(k, P7P_DM, hmm.t[k][DM].ln());
        gm.set_tsc(k, P7P_DD, hmm.t[k][DD].ln());
    }

    // Match emission scores
    let mut sc = vec![0.0_f32; abc.kp];
    sc[abc.k] = f32::NEG_INFINITY;     // gap
    sc[abc.kp - 2] = f32::NEG_INFINITY; // nonresidue
    sc[abc.kp - 1] = f32::NEG_INFINITY; // missing data

    for k in 1..=hmm.m {
        for x in 0..abc.k {
            sc[x] = (hmm.mat[k][x] / bg.f[x]).ln();
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

/// Reconfigure the target sequence length distribution of a profile.
pub fn reconfig_length(gm: &mut Profile, l: i32) {
    let pmove = (2.0 + gm.nj) / (l as f32 + 2.0 + gm.nj);
    let ploop = 1.0 - pmove;
    gm.xsc[P7P_N][P7P_LOOP] = ploop.ln();
    gm.xsc[P7P_C][P7P_LOOP] = ploop.ln();
    gm.xsc[P7P_J][P7P_LOOP] = ploop.ln();
    gm.xsc[P7P_N][P7P_MOVE] = pmove.ln();
    gm.xsc[P7P_C][P7P_MOVE] = pmove.ln();
    gm.xsc[P7P_J][P7P_MOVE] = pmove.ln();
    gm.l = l;
}

/// Reconfigure into multihit mode for target length L.
pub fn reconfig_multihit(gm: &mut Profile, l: i32) {
    gm.xsc[P7P_E][P7P_MOVE] = -(2.0_f32.ln());
    gm.xsc[P7P_E][P7P_LOOP] = -(2.0_f32.ln());
    gm.nj = 1.0;
    reconfig_length(gm, l);
}

/// Reconfigure into unihit mode for target length L.
pub fn reconfig_unihit(gm: &mut Profile, l: i32) {
    gm.xsc[P7P_E][P7P_MOVE] = 0.0;
    gm.xsc[P7P_E][P7P_LOOP] = f32::NEG_INFINITY;
    gm.nj = 0.0;
    reconfig_length(gm, l);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::AlphabetType;
    use std::path::Path;

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
