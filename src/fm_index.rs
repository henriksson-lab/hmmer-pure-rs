//! FM-index for fast pattern searching in DNA sequences.
//! Used by nhmmer for long-target scanning.

use divsufsort::sort_in_place;

/// A simple FM-index for DNA sequences.
#[derive(Debug)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FmInterval {
    pub lo: usize,
    pub hi: usize,
}

impl FmInterval {
    pub fn len(self) -> usize {
        self.hi.saturating_sub(self.lo)
    }

    pub fn is_empty(self) -> bool {
        self.lo >= self.hi
    }
}

impl FmIndex {
    /// Build an FM-index over `text` (typically uppercase ACGT, plus appended
    /// sentinel). Constructs the suffix array via divsufsort, derives the BWT
    /// from it, and tabulates the cumulative C[] array used for backward
    /// search. Counterpart to C's `fm_FM_read()` / makehmmerdb construction.
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
        FmIndex {
            bwt,
            sa,
            c,
            n: orig_n,
        }
    }

    /// Return the full suffix-array interval before any pattern characters are
    /// applied.
    pub fn root_interval(&self) -> FmInterval {
        FmInterval {
            lo: 0,
            hi: self.bwt.len(),
        }
    }

    /// Apply one backward-search step to an existing interval.
    ///
    /// This is the primitive C's FM trie search uses while recursively adding
    /// bases and pruning empty suffix-array intervals before exploring deeper
    /// seeds.
    pub fn prepend_interval(&self, interval: FmInterval, ch: u8) -> Option<FmInterval> {
        if ch == 0 || interval.is_empty() || interval.hi > self.bwt.len() {
            return None;
        }

        let lo = self.c[ch as usize] + self.occ(ch, interval.lo);
        let hi = self.c[ch as usize] + self.occ(ch, interval.hi);
        (lo < hi).then_some(FmInterval { lo, hi })
    }

    /// Reconstruct an FM-index from serialized components.
    ///
    /// `text_len` is the length of the indexed text before the internal
    /// sentinel. Serialized makehmmerdb container indexes store BWT/SA/C arrays
    /// separately, so nhmmer uses this constructor when loading those records.
    pub fn from_parts(
        bwt: Vec<u8>,
        sa: Vec<i32>,
        c: [usize; 256],
        text_len: usize,
    ) -> Result<Self, String> {
        let expected_len = text_len
            .checked_add(1)
            .ok_or_else(|| "FM-index text length overflows usize".to_string())?;
        if bwt.len() != expected_len {
            return Err(format!(
                "FM-index BWT length {} does not match text length {} plus sentinel",
                bwt.len(),
                text_len
            ));
        }
        if sa.len() != bwt.len() {
            return Err(format!(
                "FM-index suffix-array length {} does not match BWT length {}",
                sa.len(),
                bwt.len()
            ));
        }
        if c[0] != 0 {
            return Err("FM-index C table must start at zero".to_string());
        }
        if c.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err("FM-index C table is not monotonic".to_string());
        }
        if sa
            .iter()
            .any(|&pos| pos < 0 || pos as usize >= expected_len)
        {
            return Err("FM-index suffix-array entry is outside indexed text".to_string());
        }

        Ok(FmIndex {
            bwt,
            sa,
            c,
            n: text_len,
        })
    }

    /// Count exact occurrences of `pattern` in the indexed text by BWT
    /// backward search. Returns 0 when the pattern does not occur.
    /// Analog of C's `getSARangeReverse()` interval-shrinking loop.
    pub fn count(&self, pattern: &[u8]) -> usize {
        if pattern.is_empty() || self.n == 0 || pattern.contains(&0) {
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

    /// Locate all 0-based starting positions of `pattern` in the indexed text.
    /// Same backward search as `count()`, then maps the resulting suffix-array
    /// interval back to text positions. Analog of C's `fm_getOriginalPosition()`
    /// applied across the matching SA range.
    pub fn locate(&self, pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() || self.n == 0 || pattern.contains(&0) {
            return Vec::new();
        }

        let mut interval = self.root_interval();

        for &ch in pattern.iter().rev() {
            let Some(next) = self.prepend_interval(interval, ch) else {
                return Vec::new();
            };
            interval = next;
        }

        self.locate_interval(interval)
    }

    /// Locate all text positions in an already-computed suffix-array interval.
    pub fn locate_interval(&self, interval: FmInterval) -> Vec<usize> {
        if interval.is_empty() || interval.hi > self.sa.len() {
            return Vec::new();
        }

        (interval.lo..interval.hi)
            .map(|i| self.sa[i] as usize)
            .filter(|&pos| pos < self.n)
            .collect()
    }

    /// Linear-scan implementation of the Occ(ch, pos) function: counts `ch`
    /// in `bwt[0..pos]`. A real FM-index uses sampled rank tables; this
    /// minimal port is O(pos) per call.
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

    #[test]
    fn fm_index_does_not_expose_internal_sentinel() {
        let fm = FmIndex::build(b"ACGT");

        assert_eq!(fm.count(&[0]), 0);
        assert!(fm.locate(&[0]).is_empty());
    }

    #[test]
    fn fm_index_reconstructs_from_serialized_parts() {
        let original = FmIndex::build(b"ACGTACGT");
        let rebuilt = FmIndex::from_parts(
            original.bwt.clone(),
            original.sa.clone(),
            original.c,
            original.n,
        )
        .unwrap();

        let mut positions = rebuilt.locate(b"CGT");
        positions.sort();
        assert_eq!(positions, vec![1, 5]);
    }

    #[test]
    fn fm_index_rejects_inconsistent_serialized_parts() {
        let original = FmIndex::build(b"ACGT");
        let err = FmIndex::from_parts(original.bwt, original.sa, original.c, 99).unwrap_err();

        assert!(err.contains("BWT length"));
    }

    #[test]
    fn fm_index_interval_steps_match_locate_and_prune_missing_prefixes() {
        let fm = FmIndex::build(b"AACGTAACGT");
        let interval = fm
            .prepend_interval(fm.root_interval(), b'T')
            .and_then(|iv| fm.prepend_interval(iv, b'G'))
            .and_then(|iv| fm.prepend_interval(iv, b'C'))
            .and_then(|iv| fm.prepend_interval(iv, b'A'))
            .and_then(|iv| fm.prepend_interval(iv, b'A'))
            .unwrap();

        let mut via_interval = fm.locate_interval(interval);
        via_interval.sort();
        let mut via_pattern = fm.locate(b"AACGT");
        via_pattern.sort();
        assert_eq!(via_interval, via_pattern);

        assert_eq!(interval.len(), 2);
        assert!(fm.prepend_interval(interval, b'G').is_none());
    }
}
