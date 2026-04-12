//! FM-index for fast pattern searching in DNA sequences.
//! Used by nhmmer for long-target scanning.

use divsufsort::sort_in_place;

/// A simple FM-index for DNA sequences.
pub struct FmIndex {
    /// Burrows-Wheeler transform
    pub bwt: Vec<u8>,
    /// Suffix array
    pub sa: Vec<i32>,
    /// Count of each character in BWT prefix (C array)
    pub c: [usize; 256],
    /// Length of the indexed text
    pub n: usize,
}

impl FmIndex {
    /// Build an FM-index from a DNA sequence (text mode, uppercase ACGT).
    pub fn build(text: &[u8]) -> Self {
        // Append sentinel
        let mut text_s = text.to_vec();
        text_s.push(0); // sentinel (smallest byte)
        let n = text_s.len();

        // Build suffix array using divsufsort
        let mut sa = vec![0i32; n];
        sort_in_place(&text_s, &mut sa);

        // Build BWT from suffix array
        let mut bwt = vec![0u8; n];
        for i in 0..n {
            let pos = sa[i] as usize;
            bwt[i] = if pos == 0 { 0 } else { text_s[pos - 1] };
        }

        // Build C array (count of characters < c in the text)
        let mut char_count = [0usize; 256];
        for &ch in &text_s {
            char_count[ch as usize] += 1;
        }
        let mut c = [0usize; 256];
        let mut cumsum = 0;
        for i in 0..256 {
            c[i] = cumsum;
            cumsum += char_count[i];
        }

        let orig_n = text.len();
        FmIndex { bwt, sa, c, n: orig_n }
    }

    /// Count occurrences of a pattern in the indexed text.
    /// Returns the number of matches.
    pub fn count(&self, pattern: &[u8]) -> usize {
        if pattern.is_empty() || self.n == 0 {
            return 0;
        }

        let bwt_len = self.bwt.len();
        let mut lo = 0usize;
        let mut hi = bwt_len;

        // Backward search
        for &ch in pattern.iter().rev() {
            // Count occurrences of ch in bwt[0..lo] and bwt[0..hi]
            let occ_lo = self.occ(ch, lo);
            let occ_hi = self.occ(ch, hi);

            lo = self.c[ch as usize] + occ_lo;
            hi = self.c[ch as usize] + occ_hi;

            if lo >= hi {
                return 0;
            }
        }

        hi - lo
    }

    /// Find positions of a pattern in the indexed text.
    /// Returns a vector of starting positions (0-based).
    pub fn locate(&self, pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() || self.n == 0 {
            return Vec::new();
        }

        let bwt_len = self.bwt.len();
        let mut lo = 0usize;
        let mut hi = bwt_len;

        for &ch in pattern.iter().rev() {
            let occ_lo = self.occ(ch, lo);
            let occ_hi = self.occ(ch, hi);
            lo = self.c[ch as usize] + occ_lo;
            hi = self.c[ch as usize] + occ_hi;
            if lo >= hi {
                return Vec::new();
            }
        }

        (lo..hi).map(|i| self.sa[i] as usize).collect()
    }

    /// Count occurrences of character `ch` in bwt[0..pos].
    fn occ(&self, ch: u8, pos: usize) -> usize {
        self.bwt[..pos].iter().filter(|&&b| b == ch).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fm_index_basic() {
        let text = b"ACGTACGTACGT";
        let fm = FmIndex::build(text);
        assert_eq!(fm.n, 12);

        // Count exact matches
        let count = fm.count(b"ACGT");
        assert_eq!(count, 3);

        let count = fm.count(b"GGGG");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_fm_index_locate() {
        let text = b"AACGTAACGT";
        let fm = FmIndex::build(text);

        let mut positions = fm.locate(b"AACGT");
        positions.sort();
        assert_eq!(positions, vec![0, 5]);
    }
}
