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

    /// Transition probabilities: t[0..M][0..6]
    /// t[0] = begin transitions, t[1..M] = node transitions
    pub t: Vec<[f32; NTRANSITIONS]>,
    /// Match emission probabilities: mat[0..M][0..K-1]
    /// mat[0] is unused (begins at 1)
    pub mat: Vec<Vec<f32>>,
    /// Insert emission probabilities: ins[0..M][0..K-1]
    pub ins: Vec<Vec<f32>>,

    // Annotation
    pub name: String,
    pub acc: Option<String>,
    pub desc: Option<String>,
    pub rf: Option<Vec<u8>>,       // 0..M+1
    pub mm: Option<Vec<u8>>,       // model mask, 0..M+1
    pub consensus: Option<Vec<u8>>,// 0..M+1
    pub cs: Option<Vec<u8>>,       // 0..M+1
    pub ca: Option<Vec<u8>>,       // 0..M+1

    // Metadata
    pub comlog: Option<String>,
    pub nseq: i32,
    pub eff_nseq: f32,
    pub max_length: i32,
    pub ctime: Option<String>,
    pub map: Option<Vec<i32>>,     // 0..M+1
    pub checksum: u32,

    // Statistical parameters
    pub evparam: [f32; NEVPARAM],
    pub cutoff: [f32; NCUTOFFS],
    pub compo: [f32; MAXABET],

    // Flags
    pub flags: u32,
}

impl Hmm {
    /// Create a new HMM with the given model length and alphabet size.
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
