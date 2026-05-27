//! FM-index for fast pattern searching in DNA sequences.
//! Used by nhmmer for long-target scanning.

use divsufsort::sort_in_place;

/// Superblock sampling interval for the cumulative occurrence counts (C
/// `FM_METADATA.freq_cnt_sb`; makehmmerdb default 65536).
const FREQ_CNT_SB: usize = 65536;
/// Block sampling interval for the within-superblock occurrence counts (C
/// `FM_METADATA.freq_cnt_b`; makehmmerdb default 256).
const FREQ_CNT_B: usize = 256;

/// Two-level sampled rank tables over the BWT, mirroring C `FM_DATA`'s
/// `occCnts_sb` (u32 superblock cumulative) and `occCnts_b` (u16 within-superblock
/// block cumulative). They make `Occ(c, pos)` O(`FREQ_CNT_B`) instead of O(pos).
/// Counts are kept only for the symbols actually present in the BWT.
#[derive(Debug)]
struct OccRank {
    /// Distinct BWT symbols, in byte order.
    symbols: Vec<u8>,
    /// byte -> index into `symbols`, or -1 if the byte never occurs.
    sym_of: Vec<i16>,
    /// `occ_sb[sb * n_sym + s]` = count of symbol `s` in `bwt[0 .. sb*FREQ_CNT_SB]`.
    occ_sb: Vec<u32>,
    /// `occ_b[b * n_sym + s]` = count of symbol `s` in `bwt[sb_start .. b*FREQ_CNT_B]`
    /// where `sb_start` is the start of the superblock containing block `b`.
    occ_b: Vec<u16>,
}

impl OccRank {
    fn build(bwt: &[u8]) -> Self {
        let mut present = [false; 256];
        for &b in bwt {
            present[b as usize] = true;
        }
        let mut sym_of = vec![-1i16; 256];
        let mut symbols = Vec::new();
        for (b, &p) in present.iter().enumerate() {
            if p {
                sym_of[b] = symbols.len() as i16;
                symbols.push(b as u8);
            }
        }
        let ns = symbols.len().max(1);
        let nb = bwt.len() / FREQ_CNT_B + 1;
        let nsb = bwt.len() / FREQ_CNT_SB + 1;
        let mut occ_sb = vec![0u32; nsb * ns];
        let mut occ_b = vec![0u16; nb * ns];
        let mut running = vec![0u32; ns];
        let mut sb_base = vec![0u32; ns];
        for pos in 0..bwt.len() {
            // Record samples for the count over bwt[0..pos] (exclusive of pos),
            // matching the half-open Occ semantics used everywhere else.
            if pos % FREQ_CNT_SB == 0 {
                let sb = pos / FREQ_CNT_SB;
                for s in 0..ns {
                    occ_sb[sb * ns + s] = running[s];
                    sb_base[s] = running[s];
                }
            }
            if pos % FREQ_CNT_B == 0 {
                let b = pos / FREQ_CNT_B;
                for s in 0..ns {
                    occ_b[b * ns + s] = (running[s] - sb_base[s]) as u16;
                }
            }
            let s = sym_of[bwt[pos] as usize];
            running[s as usize] += 1;
        }
        OccRank {
            symbols,
            sym_of,
            occ_sb,
            occ_b,
        }
    }

    #[inline]
    fn n_sym(&self) -> usize {
        self.symbols.len().max(1)
    }
}

/// A simple FM-index for DNA sequences. Field layout mirrors a reduced C
/// `FM_DATA`: the BWT, sampled suffix array, cumulative `C[]` table, and the
/// two-level sampled occurrence-count rank (`occ`, = C `occCnts_sb`/`occCnts_b`).
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
    /// Sampled rank tables (C `FM_DATA.occCnts_sb`/`occCnts_b`).
    occ_rank: OccRank,
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
        let occ_rank = OccRank::build(&bwt);
        FmIndex {
            bwt,
            sa,
            c,
            n: orig_n,
            occ_rank,
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

        let occ_rank = OccRank::build(&bwt);
        Ok(FmIndex {
            bwt,
            sa,
            c,
            n: text_len,
            occ_rank,
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

    /// `Occ(ch, pos)`: counts `ch` in `bwt[0..pos]`, using the two-level sampled
    /// rank tables (C `fm_getOccCount`): superblock cumulative + within-superblock
    /// block cumulative + a short scan of the trailing partial block. O(`FREQ_CNT_B`).
    fn occ(&self, ch: u8, pos: usize) -> usize {
        let rank = &self.occ_rank;
        let s = rank.sym_of[ch as usize];
        if s < 0 {
            return 0;
        }
        let s = s as usize;
        let ns = rank.n_sym();
        let sb = pos / FREQ_CNT_SB;
        let b = pos / FREQ_CNT_B;
        let mut cnt = rank.occ_sb[sb * ns + s] as usize + rank.occ_b[b * ns + s] as usize;
        let block_start = b * FREQ_CNT_B;
        for &x in &self.bwt[block_start..pos] {
            if x == ch {
                cnt += 1;
            }
        }
        cnt
    }

    /// `OccLT(ch, pos)`: number of BWT symbols *strictly less than* `ch` in
    /// `bwt[0..pos]` (the sentinel byte 0 and any smaller bases all count). This
    /// is the second value C's `fm_getOccCountLT()` returns and is the quantity
    /// the bi-directional forward interval update needs. Uses the same sampled
    /// rank tables, summed over the present symbols `< ch`.
    fn occ_lt(&self, ch: u8, pos: usize) -> usize {
        let rank = &self.occ_rank;
        let ns = rank.n_sym();
        let sb = pos / FREQ_CNT_SB;
        let b = pos / FREQ_CNT_B;
        let mut cnt = 0usize;
        for (s, &sym) in rank.symbols.iter().enumerate() {
            if sym < ch {
                cnt += rank.occ_sb[sb * ns + s] as usize + rank.occ_b[b * ns + s] as usize;
            }
        }
        let block_start = b * FREQ_CNT_B;
        for &x in &self.bwt[block_start..pos] {
            if x < ch {
                cnt += 1;
            }
        }
        cnt
    }

    /// Total occurrences of `ch` across the whole BWT (= count of `ch` in the
    /// indexed text). Used to seed a single-character interval.
    fn occ_total(&self, ch: u8) -> usize {
        self.occ(ch, self.bwt.len())
    }

    /// Half-open interval over the suffix array for a single character `ch`
    /// (the starting point of a forward or backward search). Empty if `ch`
    /// does not occur.
    pub fn char_interval(&self, ch: u8) -> Option<FmInterval> {
        if ch == 0 {
            return None;
        }
        let lo = self.c[ch as usize];
        let hi = lo + self.occ_total(ch);
        (lo < hi).then_some(FmInterval { lo, hi })
    }

    /// Apply one *forward* (left-to-right) search step of the bi-directional
    /// BWT, faithful port of C `fm_updateIntervalForward()` (Algorithm 4 of
    /// Simpson 2010, `hmmer/src/fm_general.c`).
    ///
    /// Maintains the synchronized pair of intervals over a single index:
    /// `bk` is the SA-range of `reverse(W)` (the mirror, updated by a normal
    /// backward-search step) and `fwd` is the SA-range of the forward pattern
    /// `W` itself (updated via the `OccLT` offset). Appending `ch` to the
    /// pattern (`W -> W·ch`) returns the new `(bk, fwd)` pair, where `fwd` is
    /// directly locatable via [`locate_interval`]. Returns `None` when the
    /// extended pattern does not occur.
    ///
    /// C uses inclusive `[lower, upper]` ranges; this port uses the crate's
    /// half-open `[lo, hi)` convention throughout (verified in the unit tests
    /// against a brute-force forward search).
    pub fn update_interval_forward(
        &self,
        bk: FmInterval,
        fwd: FmInterval,
        ch: u8,
    ) -> Option<(FmInterval, FmInterval)> {
        if ch == 0 || bk.is_empty() || bk.hi > self.bwt.len() {
            return None;
        }
        let occ_lo = self.occ(ch, bk.lo);
        let occ_hi = self.occ(ch, bk.hi);
        // New mirror (backward) interval: normal backward-search step on `ch`.
        let new_bk_lo = self.c[ch as usize] + occ_lo;
        let new_bk_hi = self.c[ch as usize] + occ_hi;
        if new_bk_lo >= new_bk_hi {
            return None;
        }
        // New forward interval: shift `fwd.lo` by the number of suffixes in the
        // current mirror range that carry a symbol strictly smaller than `ch`,
        // then take the same size as the new mirror range.
        let occ_lt_lo = self.occ_lt(ch, bk.lo);
        let occ_lt_hi = self.occ_lt(ch, bk.hi);
        let new_fwd_lo = fwd.lo + (occ_lt_hi - occ_lt_lo);
        let new_fwd_hi = new_fwd_lo + (new_bk_hi - new_bk_lo);
        Some((
            FmInterval {
                lo: new_bk_lo,
                hi: new_bk_hi,
            },
            FmInterval {
                lo: new_fwd_lo,
                hi: new_fwd_hi,
            },
        ))
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

    /// Forward (left-to-right) bi-directional search must locate the same text
    /// positions as the backward-search `locate()`, for several patterns and a
    /// pattern that is absent. This validates the half-open port of C's
    /// `fm_updateIntervalForward` against a brute-force oracle.
    ///
    /// Mirrors C's two-index setup: the interval math runs on the backward
    /// index `fmb` (BWT of the reversed text), while the synchronized forward
    /// interval is located on the forward index `fmf` (BWT of the text). See
    /// `fm_getSARangeForward` (`fm_general.c`).
    #[test]
    fn fm_index_forward_search_matches_locate() {
        let text = b"ACGTACGTTTACGGTACAACGTAC";
        let fmf = FmIndex::build(text);
        let rev: Vec<u8> = text.iter().rev().copied().collect();
        let fmb = FmIndex::build(&rev);

        let forward_locate = |pattern: &[u8]| -> Option<Vec<usize>> {
            let (&first, rest) = pattern.split_first().unwrap();
            // Single-char start: C[c]..C[c]+occ_total are identical on fmf/fmb
            // (same character multiset), so init both intervals there.
            let mut bk = fmb.char_interval(first)?;
            let mut fwd = bk;
            for &ch in rest {
                let (nb, nf) = fmb.update_interval_forward(bk, fwd, ch)?;
                bk = nb;
                fwd = nf;
            }
            // The forward interval lives in fmf's coordinate space.
            let mut pos = fmf.locate_interval(fwd);
            pos.sort();
            Some(pos)
        };

        for pattern in [
            &b"ACGT"[..],
            &b"ACG"[..],
            &b"TAC"[..],
            &b"AACGTAC"[..],
            &b"A"[..],
            &b"GGTA"[..],
        ] {
            let mut expected = fmf.locate(pattern);
            expected.sort();
            assert_eq!(
                forward_locate(pattern),
                Some(expected),
                "forward search disagrees with locate() for {:?}",
                std::str::from_utf8(pattern).unwrap()
            );
        }

        // Absent pattern: forward search must terminate with no interval.
        assert_eq!(forward_locate(b"ACGTACGTACGTAC"), None);
        assert!(fmf.locate(b"ACGTACGTACGTAC").is_empty());
    }
}
