//! Biological sequence alphabets (DNA, RNA, amino acid).
//! Direct port of Easel's esl_alphabet.

/// Digital sequence residue type (unsigned 8-bit).
pub type Dsq = u8;

/// Sentinel value marking sequence boundaries in digital sequences.
/// Digital sequences are 1-based: dsq[0] = SENTINEL, dsq[1..L] = sequence, dsq[L+1] = SENTINEL.
pub const DSQ_SENTINEL: Dsq = 255;
pub const DSQ_ILLEGAL: Dsq = 254;
pub const DSQ_IGNORED: Dsq = 253;

/// Alphabet type codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum AlphabetType {
    Unknown = 0,
    Rna = 1,
    Dna = 2,
    Amino = 3,
}

// Canonical alphabet strings (ordering matters — matches Easel exactly)
const DNA_SYMS: &str = "ACGT-RYMKSWHBVDN*~";
const RNA_SYMS: &str = "ACGU-RYMKSWHBVDN*~";
const AMINO_SYMS: &str = "ACDEFGHIKLMNPQRSTVWY-BJZOUX*~";

/// A biological sequence alphabet.
#[derive(Debug, Clone)]
pub struct Alphabet {
    pub abc_type: AlphabetType,
    /// Number of canonical residues (4 for DNA/RNA, 20 for amino)
    pub k: usize,
    /// Total alphabet size including gap, degeneracies, special symbols
    pub kp: usize,
    /// Symbol string: sym[i] is the character for digital code i
    pub sym: Vec<u8>,
    /// Input map: ASCII char -> digital code (128 entries)
    pub inmap: [Dsq; 128],
    /// Degeneracy matrix: degen[code][canonical] = true if canonical is part of code
    pub degen: Vec<Vec<bool>>,
    /// Number of canonical residues per code
    pub ndegen: Vec<usize>,
    /// Complement map (DNA/RNA only): complement[code] = complementary code
    pub complement: Option<Vec<Dsq>>,
}

impl Alphabet {
    /// Create a standard alphabet (DNA, RNA, or Amino).
    pub fn new(abc_type: AlphabetType) -> Self {
        match abc_type {
            AlphabetType::Dna => Self::create_dna(),
            AlphabetType::Rna => Self::create_rna(),
            AlphabetType::Amino => Self::create_amino(),
            AlphabetType::Unknown => panic!("Cannot create unknown alphabet"),
        }
    }

    pub fn dna() -> Self {
        Self::new(AlphabetType::Dna)
    }

    pub fn rna() -> Self {
        Self::new(AlphabetType::Rna)
    }

    pub fn amino() -> Self {
        Self::new(AlphabetType::Amino)
    }

    fn create_dna() -> Self {
        let sym = DNA_SYMS.as_bytes().to_vec();
        let k = 4;
        let kp = 18;
        let mut abc = Self::init(AlphabetType::Dna, k, kp, sym);
        // Synonyms (must come before SetCaseInsensitive)
        abc.set_equiv(b'U', b'T');
        abc.set_equiv(b'X', b'N');
        abc.set_equiv(b'I', b'A');
        abc.set_equiv(b'_', b'-');
        abc.set_equiv(b'.', b'-');
        abc.set_case_insensitive();
        abc.set_dna_degeneracies();
        abc.set_complement_dna();
        abc
    }

    fn create_rna() -> Self {
        let sym = RNA_SYMS.as_bytes().to_vec();
        let k = 4;
        let kp = 18;
        let mut abc = Self::init(AlphabetType::Rna, k, kp, sym);
        abc.set_equiv(b'T', b'U');
        abc.set_equiv(b'X', b'N');
        abc.set_equiv(b'I', b'A');
        abc.set_equiv(b'_', b'-');
        abc.set_equiv(b'.', b'-');
        abc.set_case_insensitive();
        abc.set_rna_degeneracies();
        abc.set_complement_rna();
        abc
    }

    fn create_amino() -> Self {
        let sym = AMINO_SYMS.as_bytes().to_vec();
        let k = 20;
        let kp = 29;
        let mut abc = Self::init(AlphabetType::Amino, k, kp, sym);
        abc.set_equiv(b'_', b'-');
        abc.set_equiv(b'.', b'-');
        abc.set_case_insensitive();
        abc.set_amino_degeneracies();
        abc
    }

    fn init(abc_type: AlphabetType, k: usize, kp: usize, sym: Vec<u8>) -> Self {
        let mut inmap = [DSQ_ILLEGAL; 128];

        // Map each symbol character to its digital code
        for (i, &ch) in sym.iter().enumerate() {
            if (ch as usize) < 128 {
                inmap[ch as usize] = i as Dsq;
            }
        }

        // Initialize degeneracy: canonical residues map to themselves
        // Also set "any" character (Kp-3) to include all canonical residues
        let mut degen = vec![vec![false; k]; kp];
        let mut ndegen = vec![0usize; kp];
        for i in 0..k {
            degen[i][i] = true;
            ndegen[i] = 1;
        }
        // "any" character (Kp-3) includes all canonical residues
        ndegen[kp - 3] = k;
        for i in 0..k {
            degen[kp - 3][i] = true;
        }

        Alphabet {
            abc_type,
            k,
            kp,
            sym,
            inmap,
            degen,
            ndegen,
            complement: None,
        }
    }

    /// Set an equivalence: input character `sym` maps to the same code as `equiv`.
    fn set_equiv(&mut self, sym: u8, equiv: u8) {
        let code = self.inmap[equiv as usize];
        self.inmap[sym as usize] = code;
    }

    fn set_case_insensitive(&mut self) {
        for lc in b'a'..=b'z' {
            let uc = lc.to_ascii_uppercase();
            let lc_valid =
                self.inmap[lc as usize] != DSQ_ILLEGAL && self.inmap[lc as usize] != DSQ_IGNORED;
            let uc_valid =
                self.inmap[uc as usize] != DSQ_ILLEGAL && self.inmap[uc as usize] != DSQ_IGNORED;

            if uc_valid && !lc_valid {
                self.inmap[lc as usize] = self.inmap[uc as usize];
            } else if lc_valid && !uc_valid {
                self.inmap[uc as usize] = self.inmap[lc as usize];
            }
        }
    }

    fn set_degeneracy(&mut self, code: u8, members: &[u8]) {
        let code_idx = self.inmap[code as usize] as usize;
        for &m in members {
            let m_idx = self.inmap[m as usize] as usize;
            self.degen[code_idx][m_idx] = true;
        }
        self.ndegen[code_idx] = members.len();
    }

    fn set_dna_degeneracies(&mut self) {
        self.set_degeneracy(b'R', &[b'A', b'G']);
        self.set_degeneracy(b'Y', &[b'C', b'T']);
        self.set_degeneracy(b'M', &[b'A', b'C']);
        self.set_degeneracy(b'K', &[b'G', b'T']);
        self.set_degeneracy(b'S', &[b'C', b'G']);
        self.set_degeneracy(b'W', &[b'A', b'T']);
        self.set_degeneracy(b'H', &[b'A', b'C', b'T']);
        self.set_degeneracy(b'B', &[b'C', b'G', b'T']);
        self.set_degeneracy(b'V', &[b'A', b'C', b'G']);
        self.set_degeneracy(b'D', &[b'A', b'G', b'T']);
        // N = any
        self.set_degeneracy(b'N', &[b'A', b'C', b'G', b'T']);
        // * = nonresidue, ~ = missing: leave empty
    }

    fn set_rna_degeneracies(&mut self) {
        self.set_degeneracy(b'R', &[b'A', b'G']);
        self.set_degeneracy(b'Y', &[b'C', b'U']);
        self.set_degeneracy(b'M', &[b'A', b'C']);
        self.set_degeneracy(b'K', &[b'G', b'U']);
        self.set_degeneracy(b'S', &[b'C', b'G']);
        self.set_degeneracy(b'W', &[b'A', b'U']);
        self.set_degeneracy(b'H', &[b'A', b'C', b'U']);
        self.set_degeneracy(b'B', &[b'C', b'G', b'U']);
        self.set_degeneracy(b'V', &[b'A', b'C', b'G']);
        self.set_degeneracy(b'D', &[b'A', b'G', b'U']);
        self.set_degeneracy(b'N', &[b'A', b'C', b'G', b'U']);
    }

    fn set_amino_degeneracies(&mut self) {
        // B = N or D (Asx)
        self.set_degeneracy(b'B', &[b'N', b'D']);
        // J = I or L
        self.set_degeneracy(b'J', &[b'I', b'L']);
        // Z = Q or E (Glx)
        self.set_degeneracy(b'Z', &[b'Q', b'E']);
        // O = pyrrolysine -> maps to K
        self.set_degeneracy(b'O', &[b'K']);
        // U = selenocysteine -> maps to C
        self.set_degeneracy(b'U', &[b'C']);
        // X = any (all 20)
        let all: Vec<u8> = AMINO_SYMS.bytes().take(20).collect();
        self.set_degeneracy(b'X', &all);
    }

    fn set_complement_dna(&mut self) {
        let mut comp = vec![0u8; self.kp];
        // A<->T, C<->G
        comp[0] = 3;
        comp[1] = 2;
        comp[2] = 1;
        comp[3] = 0;
        comp[4] = 4; // gap -> gap
                     // R<->Y, M<->K, S<->S, W<->W, H<->D, B<->V
        comp[5] = 6;
        comp[6] = 5;
        comp[7] = 8;
        comp[8] = 7;
        comp[9] = 9;
        comp[10] = 10;
        comp[11] = 14;
        comp[12] = 13;
        comp[13] = 12;
        comp[14] = 11;
        comp[15] = 15; // N -> N
        comp[16] = 16; // * -> *
        comp[17] = 17; // ~ -> ~
        self.complement = Some(comp);
    }

    fn set_complement_rna(&mut self) {
        // Same as DNA complement (U maps same as T at position 3)
        self.set_complement_dna();
    }

    // ===== Query methods =====

    #[inline]
    pub fn is_canonical(&self, x: Dsq) -> bool {
        (x as usize) < self.k
    }

    #[inline]
    pub fn is_gap(&self, x: Dsq) -> bool {
        x as usize == self.k
    }

    #[inline]
    pub fn is_degenerate(&self, x: Dsq) -> bool {
        let xu = x as usize;
        xu > self.k && xu < self.kp - 2
    }

    #[inline]
    pub fn is_residue(&self, x: Dsq) -> bool {
        let xu = x as usize;
        xu < self.k || (xu > self.k && xu < self.kp - 2)
    }

    #[inline]
    pub fn is_missing(&self, x: Dsq) -> bool {
        x as usize == self.kp - 1
    }

    #[inline]
    pub fn gap_code(&self) -> Dsq {
        self.k as Dsq
    }

    #[inline]
    pub fn unknown_code(&self) -> Dsq {
        (self.kp - 3) as Dsq
    }

    #[inline]
    pub fn nonresidue_code(&self) -> Dsq {
        (self.kp - 2) as Dsq
    }

    #[inline]
    pub fn missing_code(&self) -> Dsq {
        (self.kp - 1) as Dsq
    }

    // ===== Digitization =====

    /// Digitize a single character.
    #[inline]
    pub fn digitize_symbol(&self, c: u8) -> Dsq {
        if (c as usize) < 128 {
            self.inmap[c as usize]
        } else {
            DSQ_ILLEGAL
        }
    }

    /// Digitize a text sequence into a new Vec<Dsq>.
    /// Returns a 1-based digital sequence: dsq[0] = SENTINEL, dsq[1..L] = sequence, dsq[L+1] = SENTINEL.
    pub fn digitize(&self, seq: &[u8]) -> Vec<Dsq> {
        let mut dsq = Vec::with_capacity(seq.len() + 2);
        dsq.push(DSQ_SENTINEL);
        for &c in seq {
            let code = self.digitize_symbol(c);
            if code != DSQ_IGNORED && code != DSQ_ILLEGAL {
                dsq.push(code);
            }
        }
        dsq.push(DSQ_SENTINEL);
        dsq
    }

    /// Convert a digital sequence back to text (ASCII).
    /// Input dsq is 1-based (dsq[1..=l] are the residues).
    pub fn textize(&self, dsq: &[Dsq], l: usize) -> String {
        let mut s = String::with_capacity(l);
        for i in 1..=l {
            let code = dsq[i] as usize;
            if code < self.kp {
                s.push(self.sym[code] as char);
            } else {
                s.push('?');
            }
        }
        s
    }

    /// Reverse complement a digital sequence in place (DNA/RNA only).
    /// dsq is 1-based: dsq[1..=n] are the residues.
    pub fn revcomp(&self, dsq: &mut [Dsq], n: usize) {
        let comp = self
            .complement
            .as_ref()
            .expect("revcomp requires DNA/RNA alphabet");
        // Reverse the sequence portion [1..=n]
        let mut i = 1;
        let mut j = n;
        while i < j {
            let tmp = dsq[i];
            dsq[i] = comp[dsq[j] as usize];
            dsq[j] = comp[tmp as usize];
            i += 1;
            j -= 1;
        }
        if i == j {
            dsq[i] = comp[dsq[i] as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dna_alphabet() {
        let abc = Alphabet::dna();
        assert_eq!(abc.k, 4);
        assert_eq!(abc.kp, 18);
        assert_eq!(abc.sym[0], b'A');
        assert_eq!(abc.sym[1], b'C');
        assert_eq!(abc.sym[2], b'G');
        assert_eq!(abc.sym[3], b'T');
    }

    #[test]
    fn test_amino_alphabet() {
        let abc = Alphabet::amino();
        assert_eq!(abc.k, 20);
        assert_eq!(abc.kp, 29);
        assert_eq!(abc.sym[0], b'A');
        assert_eq!(abc.sym[19], b'Y');
    }

    #[test]
    fn test_digitize_dna() {
        let abc = Alphabet::dna();
        let dsq = abc.digitize(b"ACGT");
        assert_eq!(dsq.len(), 6); // sentinel + 4 residues + sentinel
        assert_eq!(dsq[0], DSQ_SENTINEL);
        assert_eq!(dsq[1], 0); // A
        assert_eq!(dsq[2], 1); // C
        assert_eq!(dsq[3], 2); // G
        assert_eq!(dsq[4], 3); // T
        assert_eq!(dsq[5], DSQ_SENTINEL);
    }

    #[test]
    fn test_case_insensitive() {
        let abc = Alphabet::dna();
        let dsq = abc.digitize(b"acgt");
        assert_eq!(dsq[1], 0);
        assert_eq!(dsq[2], 1);
        assert_eq!(dsq[3], 2);
        assert_eq!(dsq[4], 3);
    }

    #[test]
    fn test_textize() {
        let abc = Alphabet::dna();
        let dsq = abc.digitize(b"ACGT");
        let text = abc.textize(&dsq, 4);
        assert_eq!(text, "ACGT");
    }

    #[test]
    fn test_revcomp() {
        let abc = Alphabet::dna();
        let mut dsq = abc.digitize(b"ACGT");
        abc.revcomp(&mut dsq, 4);
        let text = abc.textize(&dsq, 4);
        assert_eq!(text, "ACGT"); // ACGT revcomp is ACGT
    }

    #[test]
    fn test_revcomp2() {
        let abc = Alphabet::dna();
        let mut dsq = abc.digitize(b"AACG");
        abc.revcomp(&mut dsq, 4);
        let text = abc.textize(&dsq, 4);
        assert_eq!(text, "CGTT");
    }

    #[test]
    fn test_amino_degeneracy() {
        let abc = Alphabet::amino();
        // B = N or D
        let b_code = abc.digitize_symbol(b'B') as usize;
        let n_code = abc.digitize_symbol(b'N') as usize;
        let d_code = abc.digitize_symbol(b'D') as usize;
        assert!(abc.degen[b_code][n_code]);
        assert!(abc.degen[b_code][d_code]);
        assert_eq!(abc.ndegen[b_code], 2);
    }
}
