//! P7_HMM - Core profile HMM model.
//! Direct port of HMMER's P7_HMM data structure.

use crate::alphabet::AlphabetType;

/// Number of statistical parameters stored in models
pub const NEVPARAM: usize = 6;
/// Number of Pfam score cutoffs stored in models
pub const NCUTOFFS: usize = 6;
/// Maximum alphabet size
pub const MAXABET: usize = 20;

/// Number of transitions per node
pub const NTRANSITIONS: usize = 7;

// Transition indices
pub const MM: usize = 0;
pub const MI: usize = 1;
pub const MD: usize = 2;
pub const IM: usize = 3;
pub const II: usize = 4;
pub const DM: usize = 5;
pub const DD: usize = 6;

// E-value parameter indices
pub const P7_MMU: usize = 0;
pub const P7_MLAMBDA: usize = 1;
pub const P7_VMU: usize = 2;
pub const P7_VLAMBDA: usize = 3;
pub const P7_FTAU: usize = 4;
pub const P7_FLAMBDA: usize = 5;

// Cutoff indices
pub const P7_GA1: usize = 0;
pub const P7_GA2: usize = 1;
pub const P7_TC1: usize = 2;
pub const P7_TC2: usize = 3;
pub const P7_NC1: usize = 4;
pub const P7_NC2: usize = 5;

// Flag constants
pub const P7H_HASBITS: u32 = 1 << 0;
pub const P7H_DESC: u32 = 1 << 1;
pub const P7H_RF: u32 = 1 << 2;
pub const P7H_CS: u32 = 1 << 3;
pub const P7H_STATS: u32 = 1 << 7;
pub const P7H_MAP: u32 = 1 << 8;
pub const P7H_ACC: u32 = 1 << 9;
pub const P7H_GA: u32 = 1 << 10;
pub const P7H_TC: u32 = 1 << 11;
pub const P7H_NC: u32 = 1 << 12;
pub const P7H_CA: u32 = 1 << 13;
pub const P7H_COMPO: u32 = 1 << 14;
pub const P7H_CHKSUM: u32 = 1 << 15;
pub const P7H_CONS: u32 = 1 << 16;
pub const P7H_MMASK: u32 = 1 << 17;

pub const EVPARAM_UNSET: f32 = -99999.0;
pub const CUTOFF_UNSET: f32 = -99999.0;
pub const COMPO_UNSET: f32 = -1.0;

/// Core profile HMM model.
#[derive(Debug, Clone)]
pub struct Hmm {
    /// Model length (number of nodes)
    pub m: usize,
    /// Alphabet type
    pub abc_type: AlphabetType,
    /// Alphabet size (K)
    pub abc_k: usize,

    /// Transition probabilities: `t[0..M][0..6]`
    /// `t[0]` = begin transitions, `t[1..M]` = node transitions
    pub t: Vec<[f32; NTRANSITIONS]>,
    /// Match emission probabilities: `mat[0..M][0..K-1]`
    /// `mat[0]` is unused (begins at 1)
    pub mat: Vec<Vec<f32>>,
    /// Insert emission probabilities: `ins[0..M][0..K-1]`
    pub ins: Vec<Vec<f32>>,

    // Annotation
    pub name: String,
    pub acc: Option<String>,
    pub desc: Option<String>,
    pub rf: Option<Vec<u8>>,        // 0..M+1
    pub mm: Option<Vec<u8>>,        // model mask, 0..M+1
    pub consensus: Option<Vec<u8>>, // 0..M+1
    pub cs: Option<Vec<u8>>,        // 0..M+1
    pub ca: Option<Vec<u8>>,        // 0..M+1

    // Metadata
    pub comlog: Option<String>,
    pub nseq: i32,
    pub eff_nseq: f32,
    pub max_length: i32,
    pub ctime: Option<String>,
    pub map: Option<Vec<i32>>, // 0..M+1
    pub checksum: u32,

    // Statistical parameters
    pub evparam: [f32; NEVPARAM],
    pub cutoff: [f32; NCUTOFFS],
    pub compo: [f32; MAXABET],

    // Flags
    pub flags: u32,
}

impl Hmm {
    /// Allocate a new P7_HMM with `m` nodes for an alphabet of size `abc_k`.
    ///
    /// All transition and emission tables are zeroed; metadata fields are set
    /// to their "unset" sentinels. Combines the C shell+body allocation:
    /// counterpart to `p7_hmm_Create()` / `p7_hmm_CreateBody()`.
    pub fn new(m: usize, abc_type: AlphabetType, abc_k: usize) -> Self {
        let t = vec![[0.0f32; NTRANSITIONS]; m + 1];
        let mat = vec![vec![0.0f32; abc_k]; m + 1];
        let ins = vec![vec![0.0f32; abc_k]; m + 1];

        Hmm {
            m,
            abc_type,
            abc_k,
            t,
            mat,
            ins,
            name: String::new(),
            acc: None,
            desc: None,
            rf: None,
            mm: None,
            consensus: None,
            cs: None,
            ca: None,
            comlog: None,
            nseq: -1,
            eff_nseq: -1.0,
            max_length: -1,
            ctime: None,
            map: None,
            checksum: 0,
            evparam: [EVPARAM_UNSET; NEVPARAM],
            cutoff: [CUTOFF_UNSET; NCUTOFFS],
            compo: [COMPO_UNSET; MAXABET],
            flags: 0,
        }
    }
}

/// Port of `p7_hmm_CalculateOccupancy()` (p7_hmm.c:1338).
///
/// `mocc[k]` is the expected occupancy of match state k; `iocc[k]` that of
/// insert state k. Both slices must be at least `hmm.m + 1` long.
///
/// The mixed precision is C's, not an accident: `1.0` is a `double` literal, so
/// `(1.0 - mocc[k-1]) * hmm->t[k-1][p7H_DM]` is evaluated in double and only
/// the final sum is narrowed back to `float`. The first product stays in
/// `float`. Computing the whole line in `f32` shifts results in the last bits.
pub fn calculate_occupancy(hmm: &Hmm, mocc: &mut [f32], iocc: Option<&mut [f32]>) {
    mocc[0] = 0.0; // no M_0 state
    if hmm.m >= 1 {
        mocc[1] = hmm.t[0][MI] + hmm.t[0][MM]; // initialize w/ 1 - B->D_1
    }
    for k in 2..=hmm.m {
        let prev = mocc[k - 1];
        let match_or_insert = prev * (hmm.t[k - 1][MM] + hmm.t[k - 1][MI]);
        let delete_entry = (1.0_f64 - prev as f64) * hmm.t[k - 1][DM] as f64;
        mocc[k] = (match_or_insert as f64 + delete_entry) as f32;
    }

    if let Some(iocc) = iocc {
        iocc[0] = hmm.t[0][MI] / hmm.t[0][IM];
        for k in 1..=hmm.m {
            iocc[k] = mocc[k] * hmm.t[k][MI] / hmm.t[k][IM];
        }
    }
}

/// Port of `p7_hmm_SetComposition()` (p7_hmm.c:625).
///
/// Sets `hmm.compo[]` to the model's expected residue composition, weighting
/// each match emission by its occupancy and each insert emission by its insert
/// occupancy, then normalizing. Raises `P7H_COMPO`.
pub fn set_composition(hmm: &mut Hmm) {
    let mut mocc = vec![0.0_f32; hmm.m + 1];
    let mut iocc = vec![0.0_f32; hmm.m + 1];
    calculate_occupancy(hmm, &mut mocc, Some(&mut iocc));

    let k = hmm.abc_k.min(MAXABET);
    crate::util::vectorops::f_set(&mut hmm.compo[..k], 0.0);
    crate::util::vectorops::f_add_scaled(&mut hmm.compo[..k], &hmm.ins[0][..k], iocc[0]);
    for node in 1..=hmm.m {
        let mocc_k = mocc[node];
        let iocc_k = iocc[node];
        for x in 0..k {
            hmm.compo[x] += hmm.mat[node][x] * mocc_k;
            hmm.compo[x] += hmm.ins[node][x] * iocc_k;
        }
    }

    crate::util::vectorops::f_norm(&mut hmm.compo[..k]);
    hmm.flags |= P7H_COMPO;
}

/// Port of `p7_hmm_SetConsensus()` (p7_hmm.c:702).
///
/// With `dsq` given (C's `sq` argument, used for single-sequence query models),
/// the consensus residue at node k is the query's own residue; otherwise it is
/// the most probable match emission. Either way the symbol is upper-cased when
/// its match-emission probability reaches `mthresh` and lower-cased otherwise.
/// `mthresh` is 0.5 for amino, 0.9 for DNA/RNA, 0.5 for anything else.
///
/// Deviation from C, deliberate: when `dsq[k]` is a *degenerate* code, C
/// evaluates `hmm->mat[k][x]` with `x >= K`, which reads past the end of that
/// node's emission vector into the next node's (P7_HMM allocates `mat` as one
/// contiguous `(M+1)*K` block). That is an out-of-bounds read whose value is an
/// artifact of the allocation, and at `k` near `M` it runs off the allocation
/// entirely. This port treats a degenerate residue as below threshold, i.e.
/// lower-cased. For DNA/RNA, where `mthresh` is 0.9 and single-sequence
/// self-emission probabilities are well under that, canonical residues are
/// lower-cased too, so the two agree in practice.
pub fn set_consensus(hmm: &mut Hmm, abc: &crate::alphabet::Alphabet, dsq: Option<&[u8]>) {
    let mthresh: f32 = match hmm.abc_type {
        AlphabetType::Dna | AlphabetType::Rna => 0.9,
        _ => 0.5,
    };

    let mut cons = vec![b' '; hmm.m + 2];
    for node in 1..=hmm.m {
        let x = match dsq {
            Some(dsq) => dsq[node] as usize,
            None => crate::util::vectorops::f_argmax(&hmm.mat[node][..abc.k]),
        };
        if x >= abc.kp {
            continue;
        }
        let sym = abc.sym[x];
        let p = if x < abc.k { hmm.mat[node][x] } else { 0.0 };
        cons[node] = if p >= mthresh {
            sym.to_ascii_uppercase()
        } else {
            sym.to_ascii_lowercase()
        };
    }
    hmm.consensus = Some(cons);
    hmm.flags |= P7H_CONS;
}
