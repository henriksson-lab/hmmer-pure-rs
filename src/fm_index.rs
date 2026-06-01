//! FM-index for fast pattern searching in DNA sequences.
//! Used by nhmmer for long-target scanning.

use divsufsort::sort_in_place;
use std::fmt;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;

/// Superblock sampling interval for the cumulative occurrence counts (C
/// `FM_METADATA.freq_cnt_sb`; makehmmerdb default 65536).
const FREQ_CNT_SB: usize = 65536;
/// Block sampling interval for the within-superblock occurrence counts (C
/// `FM_METADATA.freq_cnt_b`; makehmmerdb default 256).
const FREQ_CNT_B: usize = 256;
const DEFAULT_SA_SAMPLE_RATE: usize = 32;

const PACKED_BYTE_EQ_COUNTS: [[u8; 256]; 4] = build_packed_byte_eq_counts();
const PACKED_BYTE_LT_COUNTS: [[u8; 256]; 4] = build_packed_byte_lt_counts();

const fn build_packed_byte_eq_counts() -> [[u8; 256]; 4] {
    let mut table = [[0u8; 256]; 4];
    let mut code = 0usize;
    while code < 4 {
        let mut byte = 0usize;
        while byte < 256 {
            let b = byte as u8;
            table[code][byte] = (((b >> 6) & 0b11) == code as u8) as u8
                + (((b >> 4) & 0b11) == code as u8) as u8
                + (((b >> 2) & 0b11) == code as u8) as u8
                + ((b & 0b11) == code as u8) as u8;
            byte += 1;
        }
        code += 1;
    }
    table
}

const fn build_packed_byte_lt_counts() -> [[u8; 256]; 4] {
    let mut table = [[0u8; 256]; 4];
    let mut code = 0usize;
    while code < 4 {
        let mut byte = 0usize;
        while byte < 256 {
            let b = byte as u8;
            table[code][byte] = (((b >> 6) & 0b11) < code as u8) as u8
                + (((b >> 4) & 0b11) < code as u8) as u8
                + (((b >> 2) & 0b11) < code as u8) as u8
                + ((b & 0b11) < code as u8) as u8;
            byte += 1;
        }
        code += 1;
    }
    table
}

pub struct MmapBytes {
    ptr: *const u8,
    len: usize,
}

unsafe impl Send for MmapBytes {}
unsafe impl Sync for MmapBytes {}

impl MmapBytes {
    pub fn open(path: &Path, max_len: u64) -> Result<Arc<Self>, String> {
        let file = File::open(path).map_err(|e| {
            format!(
                "failed to open FM-index target database {}: {e}",
                path.display()
            )
        })?;
        let len = file
            .metadata()
            .map_err(|e| {
                format!(
                    "failed to stat FM-index target database {}: {e}",
                    path.display()
                )
            })?
            .len();
        if len > max_len {
            return Err(format!(
                "FM-index target database {} is too large for the current in-memory reader ({} bytes > {} bytes)",
                path.display(),
                len,
                max_len
            ));
        }
        let len = usize::try_from(len)
            .map_err(|_| "FM-index target database size overflows usize".to_string())?;
        if len == 0 {
            return Ok(Arc::new(Self {
                ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                len,
            }));
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(format!(
                "failed to mmap FM-index target database {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        Ok(Arc::new(Self {
            ptr: ptr.cast::<u8>(),
            len,
        }))
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MmapBytes {
    fn drop(&mut self) {
        if self.len != 0 {
            unsafe {
                libc::munmap(self.ptr.cast::<libc::c_void>().cast_mut(), self.len);
            }
        }
    }
}

impl fmt::Debug for MmapBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapBytes").field("len", &self.len).finish()
    }
}

#[derive(Debug)]
struct MappedSa {
    bytes: Arc<MmapBytes>,
    offset: usize,
    len: usize,
}

#[derive(Debug)]
struct MappedBwt {
    bytes: Arc<MmapBytes>,
    offset: usize,
    len: usize,
}

#[derive(Debug)]
struct PackedDnaBwt {
    bytes: Vec<u8>,
    mapped_bytes: Option<Arc<MmapBytes>>,
    mapped_offset: usize,
    byte_len: usize,
    len: usize,
    term_loc: usize,
}

impl PackedDnaBwt {
    #[inline]
    fn byte_at(&self, idx: usize) -> Option<u8> {
        if let Some(bytes) = &self.mapped_bytes {
            bytes
                .as_slice()
                .get(self.mapped_offset.checked_add(idx)?)
                .copied()
        } else {
            self.bytes.get(idx).copied()
        }
    }

    fn validate_span(&self) -> Result<(), String> {
        if self.byte_len != self.len.div_ceil(4) {
            return Err("FM-index packed BWT has unexpected length".to_string());
        }
        if let Some(bytes) = &self.mapped_bytes {
            self.mapped_offset
                .checked_add(self.byte_len)
                .filter(|&end| end <= bytes.as_slice().len())
                .ok_or_else(|| "FM-index mapped packed BWT is outside mapped file".to_string())?;
        } else if self.bytes.len() != self.byte_len {
            return Err("FM-index packed BWT has unexpected length".to_string());
        }
        Ok(())
    }

    #[inline]
    fn code_at(&self, pos: usize) -> u8 {
        (self
            .byte_at(pos / 4)
            .expect("validated packed BWT position must be in range")
            >> (6 - ((pos & 0x3) * 2)))
            & 0b11
    }

    #[inline]
    fn terminal_in_range(&self, start: usize, end: usize) -> usize {
        usize::from(start <= self.term_loc && self.term_loc < end)
    }

    #[inline]
    fn byte_count_eq(byte: u8, code: u8) -> usize {
        PACKED_BYTE_EQ_COUNTS[code as usize][byte as usize] as usize
    }

    #[inline]
    fn byte_count_lt(byte: u8, code: u8) -> usize {
        PACKED_BYTE_LT_COUNTS[code as usize][byte as usize] as usize
    }

    #[inline]
    fn count_code_eq(&self, mut start: usize, end: usize, code: u8) -> usize {
        let mut count = 0usize;
        while start < end && (start & 0x3) != 0 {
            count += usize::from(self.code_at(start) == code);
            start += 1;
        }
        while start + 4 <= end {
            count += Self::byte_count_eq(
                self.byte_at(start / 4)
                    .expect("validated packed BWT position must be in range"),
                code,
            );
            start += 4;
        }
        while start < end {
            count += usize::from(self.code_at(start) == code);
            start += 1;
        }
        count
    }

    #[inline]
    fn count_code_lt(&self, mut start: usize, end: usize, code: u8) -> usize {
        let mut count = 0usize;
        while start < end && (start & 0x3) != 0 {
            count += usize::from(self.code_at(start) < code);
            start += 1;
        }
        while start + 4 <= end {
            count += Self::byte_count_lt(
                self.byte_at(start / 4)
                    .expect("validated packed BWT position must be in range"),
                code,
            );
            start += 4;
        }
        while start < end {
            count += usize::from(self.code_at(start) < code);
            start += 1;
        }
        count
    }

    #[inline]
    fn count_eq(&self, start: usize, end: usize, ch: u8) -> usize {
        match ch {
            0 => self.terminal_in_range(start, end),
            b'A' => self.count_code_eq(start, end, 0) - self.terminal_in_range(start, end),
            b'C' => self.count_code_eq(start, end, 1),
            b'G' => self.count_code_eq(start, end, 2),
            b'T' => self.count_code_eq(start, end, 3),
            _ => 0,
        }
    }

    #[inline]
    fn count_lt(&self, start: usize, end: usize, ch: u8) -> usize {
        match ch {
            0 => 0,
            b'A' => self.terminal_in_range(start, end),
            b'C' => self.count_code_lt(start, end, 1),
            b'G' => self.count_code_lt(start, end, 2),
            b'T' => self.count_code_lt(start, end, 3),
            _ => end - start,
        }
    }
}

#[derive(Debug)]
enum RegularSaSamples {
    InMemory {
        positions: Vec<i32>,
        freq_sa: usize,
    },
    Mapped {
        bytes: Arc<MmapBytes>,
        offset: usize,
        count: usize,
        freq_sa: usize,
        text_len: usize,
    },
}

impl RegularSaSamples {
    fn position_for_row(&self, row: usize) -> Option<usize> {
        match self {
            RegularSaSamples::InMemory { positions, freq_sa } => {
                if row % *freq_sa != 0 {
                    return None;
                }
                positions.get(row / *freq_sa).map(|&pos| pos as usize)
            }
            RegularSaSamples::Mapped {
                bytes,
                offset,
                count,
                freq_sa,
                text_len,
            } => {
                if row % *freq_sa != 0 {
                    return None;
                }
                let idx = row / *freq_sa;
                if idx >= *count {
                    return None;
                }
                let offset = offset.checked_add(idx.checked_mul(4)?)?;
                let chunk = bytes.as_slice().get(offset..offset + 4)?;
                let pos = i32::from_le_bytes(chunk.try_into().ok()?);
                Some(if pos < 0 { *text_len } else { pos as usize })
            }
        }
    }
}

/// Two-level sampled rank tables over the BWT, mirroring C `FM_DATA`'s
/// `occCnts_sb` (superblock cumulative) and `occCnts_b` (u16 within-superblock
/// block cumulative). They make `Occ(c, pos)` O(`FREQ_CNT_B`) instead of O(pos).
/// Counts are kept only for the symbols actually present in the BWT.
#[derive(Debug)]
struct OccRank {
    /// Distinct BWT symbols, in byte order.
    symbols: Vec<u8>,
    /// byte -> index into `symbols`, or -1 if the byte never occurs.
    sym_of: Vec<i16>,
    /// `occ_sb[sb * n_sym + s]` = count of symbol `s` in `bwt[0 .. sb*FREQ_CNT_SB]`.
    occ_sb: Vec<usize>,
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
        let mut occ_sb = vec![0usize; nsb * ns];
        let mut occ_b = vec![0u16; nb * ns];
        let mut running = vec![0usize; ns];
        let mut sb_base = vec![0usize; ns];
        // Iterate inclusively to `bwt.len()` so the trailing sample row(s) at an
        // exact block/superblock boundary are written. Without this, when
        // `bwt.len()` is a multiple of FREQ_CNT_B the last block row stays 0 and
        // `occ(ch, bwt.len())` (used by the very first backward-search step from
        // the root interval) wrongly returns 0 — making every FM search return
        // no hits for texts of length ≡ 255 (mod 256).
        for pos in 0..=bwt.len() {
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
            if pos < bwt.len() {
                let s = sym_of[bwt[pos] as usize];
                running[s as usize] += 1;
            }
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

    fn from_makehmmerdb_dna_tables(
        bwt_len: usize,
        term_loc: usize,
        freq_cnt_b: usize,
        occ_b_codes: &[u16],
        occ_sb_codes: &[u32],
    ) -> Result<Self, String> {
        if freq_cnt_b == 0 {
            return Err("FM-index occurrence sampling frequency is zero".to_string());
        }
        let nb = 1 + bwt_len.div_ceil(freq_cnt_b);
        let nsb = 1 + bwt_len.div_ceil(FREQ_CNT_SB);
        if occ_b_codes.len() != nb * 4 {
            return Err("FM-index occurrence block table has unexpected length".to_string());
        }
        if occ_sb_codes.len() != nsb * 4 {
            return Err("FM-index occurrence superblock table has unexpected length".to_string());
        }

        let symbols = vec![0, b'A', b'C', b'G', b'T'];
        let mut sym_of = vec![-1i16; 256];
        for (idx, &sym) in symbols.iter().enumerate() {
            sym_of[sym as usize] = idx as i16;
        }

        let ns = symbols.len();
        let mut occ_b = vec![0u16; nb * ns];
        let mut occ_sb = vec![0usize; nsb * ns];

        for row in 0..nb {
            let pos = (row * freq_cnt_b).min(bwt_len);
            let sb_start = (pos / FREQ_CNT_SB) * FREQ_CNT_SB;
            let terminal_in_block = sb_start <= term_loc && term_loc < pos;
            occ_b[row * ns] = terminal_in_block as u16;
            let code0 = occ_b_codes[row * 4];
            occ_b[row * ns + 1] = code0.saturating_sub(terminal_in_block as u16);
            occ_b[row * ns + 2] = occ_b_codes[row * 4 + 1];
            occ_b[row * ns + 3] = occ_b_codes[row * 4 + 2];
            occ_b[row * ns + 4] = occ_b_codes[row * 4 + 3];
        }

        for row in 0..nsb {
            let pos = (row * FREQ_CNT_SB).min(bwt_len);
            let terminal_in_prefix = term_loc < pos;
            occ_sb[row * ns] = terminal_in_prefix as usize;
            let code0 = occ_sb_codes[row * 4] as usize;
            occ_sb[row * ns + 1] = code0.saturating_sub(terminal_in_prefix as usize);
            occ_sb[row * ns + 2] = occ_sb_codes[row * 4 + 1] as usize;
            occ_sb[row * ns + 3] = occ_sb_codes[row * 4 + 2] as usize;
            occ_sb[row * ns + 4] = occ_sb_codes[row * 4 + 3] as usize;
        }

        Ok(Self {
            symbols,
            sym_of,
            occ_sb,
            occ_b,
        })
    }
}

/// A simple FM-index for DNA sequences. Field layout mirrors a reduced C
/// `FM_DATA`: the BWT, sampled suffix array, cumulative `C[]` table, and the
/// two-level sampled occurrence-count rank (`occ`, = C `occCnts_sb`/`occCnts_b`).
#[derive(Debug)]
pub struct FmIndex {
    /// Burrows-Wheeler transform
    pub bwt: Vec<u8>,
    mapped_bwt: Option<MappedBwt>,
    packed_dna_bwt: Option<PackedDnaBwt>,
    /// Suffix array
    pub sa: Vec<i32>,
    /// Sparse suffix-array samples as `(BWT row, text position)`.
    sa_samples: Vec<(i32, i32)>,
    regular_sa_samples: Option<RegularSaSamples>,
    mapped_sa: Option<MappedSa>,
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
            mapped_bwt: None,
            packed_dna_bwt: None,
            sa,
            sa_samples: Vec::new(),
            regular_sa_samples: None,
            mapped_sa: None,
            c,
            n: orig_n,
            occ_rank,
        }
    }

    /// Build an FM-index over `text`, then retain only row-sampled suffix-array
    /// entries. This matches `build()` search semantics while avoiding a
    /// resident full `i32` suffix array for raw makehmmerdb C-stream targets.
    pub fn build_sampled(text: &[u8]) -> Self {
        Self::build_sampled_with_rate(text, DEFAULT_SA_SAMPLE_RATE)
    }

    pub fn build_sampled_with_rate(text: &[u8], sample_rate: usize) -> Self {
        let mut index = Self::build(text);
        let sample_rate = sample_rate.max(1);
        index.sa_samples = index
            .sa
            .iter()
            .enumerate()
            .filter_map(|(row, &pos)| {
                (row % sample_rate == 0).then(|| {
                    (
                        i32::try_from(row).expect("FM-index row exceeds i32 range"),
                        pos,
                    )
                })
            })
            .collect();
        index.regular_sa_samples = None;
        index.sa.clear();
        index.sa.shrink_to_fit();
        index
    }

    pub fn from_makehmmerdb_raw_parts(
        bwt: Vec<u8>,
        raw_sa_samples: Vec<i32>,
        term_loc: usize,
        freq_sa: usize,
        freq_cnt_b: usize,
        occ_b_codes: Vec<u16>,
        occ_sb_codes: Vec<u32>,
    ) -> Result<Self, String> {
        let bwt_len = bwt.len();
        if bwt_len == 0 {
            return Err("FM-index raw BWT is empty".to_string());
        }
        let text_len = bwt_len - 1;
        let mut char_count = [0usize; 256];
        for &ch in &bwt {
            char_count[ch as usize] += 1;
        }
        let mut c = [0usize; 256];
        let mut cumsum = 0usize;
        for i in 0..256 {
            c[i] = cumsum;
            cumsum += char_count[i];
        }
        Self::validate_parts_common(&bwt, c, text_len, bwt_len)?;

        let regular_sa_samples =
            Self::make_regular_sa_samples(raw_sa_samples, freq_sa, text_len, bwt_len)?;

        let occ_rank = OccRank::from_makehmmerdb_dna_tables(
            bwt_len,
            term_loc,
            freq_cnt_b,
            &occ_b_codes,
            &occ_sb_codes,
        )?;
        Ok(FmIndex {
            bwt,
            mapped_bwt: None,
            packed_dna_bwt: None,
            sa: Vec::new(),
            sa_samples: Vec::new(),
            regular_sa_samples: Some(regular_sa_samples),
            mapped_sa: None,
            c,
            n: text_len,
            occ_rank,
        })
    }

    fn make_regular_sa_samples(
        raw_sa_samples: Vec<i32>,
        freq_sa: usize,
        text_len: usize,
        bwt_len: usize,
    ) -> Result<RegularSaSamples, String> {
        if freq_sa == 0 {
            return Err("FM-index suffix-array sampling frequency is zero".to_string());
        }
        let mut positions = Vec::with_capacity(raw_sa_samples.len());
        for (idx, &pos) in raw_sa_samples.iter().enumerate() {
            let row = idx
                .checked_mul(freq_sa)
                .ok_or_else(|| "FM-index suffix-array sample row overflows usize".to_string())?;
            if row >= bwt_len {
                continue;
            }
            let pos = if pos < 0 {
                i32::try_from(text_len)
                    .map_err(|_| "FM-index suffix-array sample exceeds i32 range".to_string())?
            } else {
                pos
            };
            if pos as usize >= bwt_len {
                return Err("FM-index suffix-array sample is outside indexed text".to_string());
            }
            positions.push(pos);
        }
        Ok(RegularSaSamples::InMemory { positions, freq_sa })
    }

    fn make_mapped_regular_sa_samples(
        mapped_bytes: Arc<MmapBytes>,
        raw_sa_offset: usize,
        raw_sa_count: usize,
        freq_sa: usize,
        text_len: usize,
        bwt_len: usize,
    ) -> Result<RegularSaSamples, String> {
        if freq_sa == 0 {
            return Err("FM-index suffix-array sampling frequency is zero".to_string());
        }
        let raw_sa_bytes = raw_sa_count
            .checked_mul(4)
            .ok_or_else(|| "FM-index suffix-array sample span overflows usize".to_string())?;
        let slice = mapped_bytes
            .as_slice()
            .get(raw_sa_offset..raw_sa_offset + raw_sa_bytes)
            .ok_or_else(|| "FM-index suffix-array samples are outside mapped file".to_string())?;
        for (idx, chunk) in slice.chunks_exact(4).enumerate() {
            let row = idx
                .checked_mul(freq_sa)
                .ok_or_else(|| "FM-index suffix-array sample row overflows usize".to_string())?;
            if row >= bwt_len {
                continue;
            }
            let pos = i32::from_le_bytes(chunk.try_into().unwrap());
            let pos = if pos < 0 {
                i32::try_from(text_len)
                    .map_err(|_| "FM-index suffix-array sample exceeds i32 range".to_string())?
            } else {
                pos
            };
            if pos as usize >= bwt_len {
                return Err("FM-index suffix-array sample is outside indexed text".to_string());
            }
        }
        Ok(RegularSaSamples::Mapped {
            bytes: mapped_bytes,
            offset: raw_sa_offset,
            count: raw_sa_count,
            freq_sa,
            text_len,
        })
    }

    pub fn from_makehmmerdb_packed_dna_parts(
        packed_bwt: Vec<u8>,
        bwt_len: usize,
        raw_sa_samples: Vec<i32>,
        term_loc: usize,
        freq_sa: usize,
        freq_cnt_b: usize,
        occ_b_codes: Vec<u16>,
        occ_sb_codes: Vec<u32>,
    ) -> Result<Self, String> {
        if freq_cnt_b == 0 {
            return Err("FM-index occurrence sampling frequency is zero".to_string());
        }
        if bwt_len == 0 {
            return Err("FM-index raw BWT is empty".to_string());
        }
        let packed_bwt = PackedDnaBwt {
            byte_len: packed_bwt.len(),
            bytes: packed_bwt,
            mapped_bytes: None,
            mapped_offset: 0,
            len: bwt_len,
            term_loc,
        };
        packed_bwt.validate_span()?;
        if term_loc >= bwt_len {
            return Err("FM-index terminal location is outside BWT".to_string());
        }
        let text_len = bwt_len - 1;
        let nsb = 1 + bwt_len.div_ceil(FREQ_CNT_SB);
        let final_occ_sb_row = (nsb - 1) * 4;
        if occ_sb_codes.len() < final_occ_sb_row + 4 {
            return Err("FM-index occurrence superblock table has unexpected length".to_string());
        }
        let code0_total = occ_sb_codes[final_occ_sb_row] as usize;
        if code0_total == 0 {
            return Err("FM-index occurrence table is missing terminal symbol".to_string());
        }
        let counts = [
            1usize,
            code0_total - 1,
            occ_sb_codes[final_occ_sb_row + 1] as usize,
            occ_sb_codes[final_occ_sb_row + 2] as usize,
            occ_sb_codes[final_occ_sb_row + 3] as usize,
        ];
        if counts.iter().sum::<usize>() != bwt_len {
            return Err("FM-index occurrence table totals do not match BWT length".to_string());
        }
        let mut c = [0usize; 256];
        c[0] = 0;
        c[b'A' as usize] = counts[0];
        c[b'C' as usize] = counts[0] + counts[1];
        c[b'G' as usize] = counts[0] + counts[1] + counts[2];
        c[b'T' as usize] = counts[0] + counts[1] + counts[2] + counts[3];

        let regular_sa_samples =
            Self::make_regular_sa_samples(raw_sa_samples, freq_sa, text_len, bwt_len)?;

        let occ_rank = OccRank::from_makehmmerdb_dna_tables(
            bwt_len,
            term_loc,
            freq_cnt_b,
            &occ_b_codes,
            &occ_sb_codes,
        )?;
        Ok(FmIndex {
            bwt: Vec::new(),
            mapped_bwt: None,
            packed_dna_bwt: Some(packed_bwt),
            sa: Vec::new(),
            sa_samples: Vec::new(),
            regular_sa_samples: Some(regular_sa_samples),
            mapped_sa: None,
            c,
            n: text_len,
            occ_rank,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_makehmmerdb_packed_dna_parts_mapped_sa(
        packed_bwt: Vec<u8>,
        bwt_len: usize,
        mapped_bytes: Arc<MmapBytes>,
        raw_sa_offset: usize,
        raw_sa_count: usize,
        term_loc: usize,
        freq_sa: usize,
        freq_cnt_b: usize,
        occ_b_codes: Vec<u16>,
        occ_sb_codes: Vec<u32>,
    ) -> Result<Self, String> {
        let mut index = Self::from_makehmmerdb_packed_dna_parts(
            packed_bwt,
            bwt_len,
            Vec::new(),
            term_loc,
            freq_sa,
            freq_cnt_b,
            occ_b_codes,
            occ_sb_codes,
        )?;
        index.regular_sa_samples = Some(Self::make_mapped_regular_sa_samples(
            mapped_bytes,
            raw_sa_offset,
            raw_sa_count,
            freq_sa,
            index.n,
            bwt_len,
        )?);
        Ok(index)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_makehmmerdb_mapped_packed_dna_parts_mapped_sa(
        mapped_packed_bwt: Arc<MmapBytes>,
        packed_bwt_offset: usize,
        packed_bwt_bytes: usize,
        bwt_len: usize,
        mapped_sa_bytes: Arc<MmapBytes>,
        raw_sa_offset: usize,
        raw_sa_count: usize,
        term_loc: usize,
        freq_sa: usize,
        freq_cnt_b: usize,
        occ_b_codes: Vec<u16>,
        occ_sb_codes: Vec<u32>,
    ) -> Result<Self, String> {
        let packed_bwt = PackedDnaBwt {
            bytes: Vec::new(),
            mapped_bytes: Some(mapped_packed_bwt),
            mapped_offset: packed_bwt_offset,
            byte_len: packed_bwt_bytes,
            len: bwt_len,
            term_loc,
        };
        packed_bwt.validate_span()?;
        if freq_cnt_b == 0 {
            return Err("FM-index occurrence sampling frequency is zero".to_string());
        }
        if bwt_len == 0 {
            return Err("FM-index raw BWT is empty".to_string());
        }
        if term_loc >= bwt_len {
            return Err("FM-index terminal location is outside BWT".to_string());
        }
        let text_len = bwt_len - 1;
        let nsb = 1 + bwt_len.div_ceil(FREQ_CNT_SB);
        let final_occ_sb_row = (nsb - 1) * 4;
        if occ_sb_codes.len() < final_occ_sb_row + 4 {
            return Err("FM-index occurrence superblock table has unexpected length".to_string());
        }
        let code0_total = occ_sb_codes[final_occ_sb_row] as usize;
        if code0_total == 0 {
            return Err("FM-index occurrence table is missing terminal symbol".to_string());
        }
        let counts = [
            1usize,
            code0_total - 1,
            occ_sb_codes[final_occ_sb_row + 1] as usize,
            occ_sb_codes[final_occ_sb_row + 2] as usize,
            occ_sb_codes[final_occ_sb_row + 3] as usize,
        ];
        if counts.iter().sum::<usize>() != bwt_len {
            return Err("FM-index occurrence table totals do not match BWT length".to_string());
        }
        let mut c = [0usize; 256];
        c[0] = 0;
        c[b'A' as usize] = counts[0];
        c[b'C' as usize] = counts[0] + counts[1];
        c[b'G' as usize] = counts[0] + counts[1] + counts[2];
        c[b'T' as usize] = counts[0] + counts[1] + counts[2] + counts[3];

        let regular_sa_samples = Self::make_mapped_regular_sa_samples(
            mapped_sa_bytes,
            raw_sa_offset,
            raw_sa_count,
            freq_sa,
            text_len,
            bwt_len,
        )?;
        let occ_rank = OccRank::from_makehmmerdb_dna_tables(
            bwt_len,
            term_loc,
            freq_cnt_b,
            &occ_b_codes,
            &occ_sb_codes,
        )?;
        Ok(FmIndex {
            bwt: Vec::new(),
            mapped_bwt: None,
            packed_dna_bwt: Some(packed_bwt),
            sa: Vec::new(),
            sa_samples: Vec::new(),
            regular_sa_samples: Some(regular_sa_samples),
            mapped_sa: None,
            c,
            n: text_len,
            occ_rank,
        })
    }

    /// Return the full suffix-array interval before any pattern characters are
    /// applied.
    pub fn bwt_len(&self) -> usize {
        if let Some(mapped) = &self.mapped_bwt {
            mapped.len
        } else if let Some(packed) = &self.packed_dna_bwt {
            packed.len
        } else {
            self.bwt.len()
        }
    }

    fn bwt_slice(&self) -> &[u8] {
        if let Some(mapped) = &self.mapped_bwt {
            &mapped.bytes.as_slice()[mapped.offset..mapped.offset + mapped.len]
        } else {
            &self.bwt
        }
    }

    fn bwt_at(&self, pos: usize) -> Option<u8> {
        if let Some(packed) = &self.packed_dna_bwt {
            if pos >= packed.len {
                return None;
            }
            if pos == packed.term_loc {
                return Some(0);
            }
            let byte = packed.byte_at(pos / 4)?;
            return Some(match (byte >> (6 - ((pos & 0x3) * 2))) & 0b11 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                3 => b'T',
                _ => unreachable!(),
            });
        }
        self.bwt_slice().get(pos).copied()
    }

    pub fn root_interval(&self) -> FmInterval {
        FmInterval {
            lo: 0,
            hi: self.bwt_len(),
        }
    }

    /// Apply one backward-search step to an existing interval.
    ///
    /// This is the primitive C's FM trie search uses while recursively adding
    /// bases and pruning empty suffix-array intervals before exploring deeper
    /// seeds.
    pub fn prepend_interval(&self, interval: FmInterval, ch: u8) -> Option<FmInterval> {
        if ch == 0 || interval.is_empty() || interval.hi > self.bwt_len() {
            return None;
        }

        let lo = self.c[ch as usize].checked_add(self.occ(ch, interval.lo))?;
        let hi = self.c[ch as usize].checked_add(self.occ(ch, interval.hi))?;
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
        Self::from_parts_inner(bwt, sa, c, text_len, None)
    }

    /// Reconstruct an FM-index while retaining only sampled suffix-array rows.
    ///
    /// This preserves FM locate semantics and trades up to `sample_rate - 1`
    /// LF steps per located row for substantially lower resident memory when
    /// loading makehmmerdb container records.
    pub fn from_parts_sampled(
        bwt: Vec<u8>,
        sa: Vec<i32>,
        c: [usize; 256],
        text_len: usize,
    ) -> Result<Self, String> {
        Self::from_parts_inner(bwt, sa, c, text_len, Some(DEFAULT_SA_SAMPLE_RATE))
    }

    pub fn from_mapped_parts(
        bwt: Vec<u8>,
        sa_len: usize,
        mapped_bytes: Arc<MmapBytes>,
        sa_offset: usize,
        c: [usize; 256],
        text_len: usize,
    ) -> Result<Self, String> {
        let expected_len = text_len
            .checked_add(1)
            .ok_or_else(|| "FM-index text length overflows usize".to_string())?;
        Self::validate_parts_common(&bwt, c, text_len, expected_len)?;
        if sa_len != bwt.len() {
            return Err(format!(
                "FM-index suffix-array length {} does not match BWT length {}",
                sa_len,
                bwt.len()
            ));
        }
        let sa_bytes = sa_len
            .checked_mul(4)
            .ok_or_else(|| "FM-index mapped suffix-array span overflows usize".to_string())?;
        sa_offset
            .checked_add(sa_bytes)
            .filter(|&end| end <= mapped_bytes.as_slice().len())
            .ok_or_else(|| "FM-index mapped suffix-array is outside mapped file".to_string())?;
        let occ_rank = OccRank::build(&bwt);
        Ok(FmIndex {
            bwt,
            mapped_bwt: None,
            packed_dna_bwt: None,
            sa: Vec::new(),
            sa_samples: Vec::new(),
            regular_sa_samples: None,
            mapped_sa: Some(MappedSa {
                bytes: mapped_bytes,
                offset: sa_offset,
                len: sa_len,
            }),
            c,
            n: text_len,
            occ_rank,
        })
    }

    pub fn from_mapped_file_parts(
        mapped_bytes: Arc<MmapBytes>,
        bwt_offset: usize,
        bwt_len: usize,
        sa_offset: usize,
        sa_len: usize,
        c: [usize; 256],
        text_len: usize,
    ) -> Result<Self, String> {
        Self::from_mapped_file_parts_with_occ(
            mapped_bytes,
            bwt_offset,
            bwt_len,
            sa_offset,
            sa_len,
            c,
            text_len,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_mapped_file_parts_with_makehmmerdb_occ(
        mapped_bytes: Arc<MmapBytes>,
        bwt_offset: usize,
        bwt_len: usize,
        sa_offset: usize,
        sa_len: usize,
        c: [usize; 256],
        text_len: usize,
        term_loc: usize,
        freq_cnt_b: usize,
        occ_b_offset: usize,
        occ_b_count: usize,
        occ_sb_offset: usize,
        occ_sb_count: usize,
    ) -> Result<Self, String> {
        let mapped_slice = mapped_bytes.as_slice();
        let occ_b_bytes = occ_b_count
            .checked_mul(2)
            .ok_or_else(|| "FM-index occurrence block table span overflows usize".to_string())?;
        let occ_sb_bytes = occ_sb_count.checked_mul(4).ok_or_else(|| {
            "FM-index occurrence superblock table span overflows usize".to_string()
        })?;
        let occ_b_slice = mapped_slice
            .get(occ_b_offset..occ_b_offset + occ_b_bytes)
            .ok_or_else(|| "FM-index occurrence block table is outside mapped file".to_string())?;
        let occ_sb_slice = mapped_slice
            .get(occ_sb_offset..occ_sb_offset + occ_sb_bytes)
            .ok_or_else(|| {
                "FM-index occurrence superblock table is outside mapped file".to_string()
            })?;
        let mut occ_b = Vec::with_capacity(occ_b_count);
        for chunk in occ_b_slice.chunks_exact(2) {
            occ_b.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        let mut occ_sb = Vec::with_capacity(occ_sb_count);
        for chunk in occ_sb_slice.chunks_exact(4) {
            occ_sb.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let occ_rank =
            OccRank::from_makehmmerdb_dna_tables(bwt_len, term_loc, freq_cnt_b, &occ_b, &occ_sb)?;
        Self::from_mapped_file_parts_with_occ(
            mapped_bytes,
            bwt_offset,
            bwt_len,
            sa_offset,
            sa_len,
            c,
            text_len,
            Some(occ_rank),
        )
    }

    fn from_mapped_file_parts_with_occ(
        mapped_bytes: Arc<MmapBytes>,
        bwt_offset: usize,
        bwt_len: usize,
        sa_offset: usize,
        sa_len: usize,
        c: [usize; 256],
        text_len: usize,
        occ_rank: Option<OccRank>,
    ) -> Result<Self, String> {
        let expected_len = text_len
            .checked_add(1)
            .ok_or_else(|| "FM-index text length overflows usize".to_string())?;
        let mapped_slice = mapped_bytes.as_slice();
        let bwt = mapped_slice
            .get(bwt_offset..bwt_offset + bwt_len)
            .ok_or_else(|| "FM-index mapped BWT is outside mapped file".to_string())?;
        Self::validate_parts_common(bwt, c, text_len, expected_len)?;
        if sa_len != bwt_len {
            return Err(format!(
                "FM-index suffix-array length {} does not match BWT length {}",
                sa_len, bwt_len
            ));
        }
        let sa_bytes = sa_len
            .checked_mul(4)
            .ok_or_else(|| "FM-index mapped suffix-array span overflows usize".to_string())?;
        sa_offset
            .checked_add(sa_bytes)
            .filter(|&end| end <= mapped_slice.len())
            .ok_or_else(|| "FM-index mapped suffix-array is outside mapped file".to_string())?;
        let occ_rank = occ_rank.unwrap_or_else(|| OccRank::build(bwt));
        Ok(FmIndex {
            bwt: Vec::new(),
            mapped_bwt: Some(MappedBwt {
                bytes: Arc::clone(&mapped_bytes),
                offset: bwt_offset,
                len: bwt_len,
            }),
            packed_dna_bwt: None,
            sa: Vec::new(),
            sa_samples: Vec::new(),
            regular_sa_samples: None,
            mapped_sa: Some(MappedSa {
                bytes: mapped_bytes,
                offset: sa_offset,
                len: sa_len,
            }),
            c,
            n: text_len,
            occ_rank,
        })
    }

    fn from_parts_inner(
        bwt: Vec<u8>,
        sa: Vec<i32>,
        c: [usize; 256],
        text_len: usize,
        sample_rate: Option<usize>,
    ) -> Result<Self, String> {
        let expected_len = text_len
            .checked_add(1)
            .ok_or_else(|| "FM-index text length overflows usize".to_string())?;
        Self::validate_parts_common(&bwt, c, text_len, expected_len)?;
        if sa.len() != bwt.len() {
            return Err(format!(
                "FM-index suffix-array length {} does not match BWT length {}",
                sa.len(),
                bwt.len()
            ));
        }
        if sa
            .iter()
            .any(|&pos| pos < 0 || pos as usize >= expected_len)
        {
            return Err("FM-index suffix-array entry is outside indexed text".to_string());
        }
        #[cfg(debug_assertions)]
        {
            let mut seen = vec![false; expected_len];
            for &pos in &sa {
                let pos = pos as usize;
                if seen[pos] {
                    return Err(format!(
                        "FM-index suffix-array is not a permutation: duplicate position {pos}"
                    ));
                }
                seen[pos] = true;
            }
        }

        let sa_samples = if let Some(sample_rate) = sample_rate {
            if sample_rate == 0 {
                return Err("FM-index suffix-array sample rate is zero".to_string());
            }
            let mut samples = Vec::with_capacity(sa.len().div_ceil(sample_rate));
            for (row, &pos) in sa.iter().enumerate() {
                if row % sample_rate == 0 {
                    samples.push((
                        i32::try_from(row)
                            .map_err(|_| "FM-index row exceeds i32 range".to_string())?,
                        pos,
                    ));
                }
            }
            samples
        } else {
            Vec::new()
        };
        let sa = if sample_rate.is_some() {
            Vec::new()
        } else {
            sa
        };
        let occ_rank = OccRank::build(&bwt);
        Ok(FmIndex {
            bwt,
            mapped_bwt: None,
            packed_dna_bwt: None,
            sa,
            sa_samples,
            regular_sa_samples: None,
            mapped_sa: None,
            c,
            n: text_len,
            occ_rank,
        })
    }

    fn validate_parts_common(
        bwt: &[u8],
        c: [usize; 256],
        text_len: usize,
        expected_len: usize,
    ) -> Result<(), String> {
        if bwt.len() != expected_len {
            return Err(format!(
                "FM-index BWT length {} does not match text length {} plus sentinel",
                bwt.len(),
                text_len
            ));
        }
        if c[0] != 0 {
            return Err("FM-index C table must start at zero".to_string());
        }
        if c.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err("FM-index C table is not monotonic".to_string());
        }
        let mut counts = [0usize; 256];
        for &ch in bwt {
            counts[ch as usize] += 1;
        }
        if counts[0] != 1 {
            return Err(format!(
                "FM-index BWT must contain exactly one sentinel byte, found {}",
                counts[0]
            ));
        }
        let mut cumulative = 0usize;
        for ch in 0..256 {
            if c[ch] != cumulative {
                return Err(format!(
                    "FM-index C table entry {ch} is {}, expected {cumulative}",
                    c[ch]
                ));
            }
            cumulative = cumulative
                .checked_add(counts[ch])
                .ok_or_else(|| "FM-index C table cumulative count overflows usize".to_string())?;
        }
        if cumulative != bwt.len() {
            return Err(format!(
                "FM-index C table covers {cumulative} symbols, expected {}",
                bwt.len()
            ));
        }
        Ok(())
    }

    /// Count exact occurrences of `pattern` in the indexed text by BWT
    /// backward search. Returns 0 when the pattern does not occur.
    /// Analog of C's `getSARangeReverse()` interval-shrinking loop.
    pub fn count(&self, pattern: &[u8]) -> usize {
        if pattern.is_empty() || self.n == 0 || pattern.contains(&0) {
            return 0;
        }

        let bwt_len = self.bwt_len();
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
        if interval.is_empty() || interval.hi > self.bwt_len() {
            return Vec::new();
        }

        (interval.lo..interval.hi)
            .filter_map(|i| self.suffix_position(i))
            .filter(|&pos| pos < self.n)
            .collect()
    }

    /// Locate at most `limit` text positions in an already-computed suffix-array
    /// interval. This mirrors `locate_interval()` order without materializing
    /// very large FM intervals that callers are going to truncate anyway.
    pub fn locate_interval_limited(&self, interval: FmInterval, limit: usize) -> Vec<usize> {
        if limit == 0 || interval.is_empty() || interval.hi > self.bwt_len() {
            return Vec::new();
        }

        let mut positions = Vec::new();
        for row in interval.lo..interval.hi {
            if let Some(pos) = self.suffix_position(row).filter(|&pos| pos < self.n) {
                positions.push(pos);
                if positions.len() >= limit {
                    break;
                }
            }
        }
        positions
    }

    /// Faithful public counterpart of C `FM_backtrackSeed()` for one BWT row.
    ///
    /// `suffix_position()` already performs the same LF backtracking to a
    /// sampled suffix-array row when only samples are resident, and returns the
    /// text position represented by this row.
    pub fn backtrack_seed_position(&self, row: usize) -> Option<usize> {
        self.suffix_position(row).filter(|&pos| pos < self.n)
    }

    fn suffix_position(&self, row: usize) -> Option<usize> {
        if !self.sa.is_empty() {
            return self.sa.get(row).map(|&pos| pos as usize);
        }
        if let Some(mapped) = &self.mapped_sa {
            return self.mapped_suffix_position(mapped, row);
        }

        let mut row = row;
        let mut steps = 0usize;
        let limit = self.bwt_len();
        while steps <= limit {
            if let Some(samples) = &self.regular_sa_samples {
                if let Some(sampled_pos) = samples.position_for_row(row) {
                    return Some((sampled_pos + steps) % self.bwt_len());
                }
            }
            let row_i32 = i32::try_from(row).ok()?;
            if let Ok(idx) = self
                .sa_samples
                .binary_search_by_key(&row_i32, |&(sample_row, _)| sample_row)
            {
                let sampled_pos = self.sa_samples[idx].1 as usize;
                return Some((sampled_pos + steps) % self.bwt_len());
            }
            let ch = self.bwt_at(row)?;
            row = self.c[ch as usize].checked_add(self.occ(ch, row))?;
            steps += 1;
        }
        None
    }

    fn mapped_suffix_position(&self, mapped: &MappedSa, row: usize) -> Option<usize> {
        if row >= mapped.len {
            return None;
        }
        let offset = mapped.offset.checked_add(row.checked_mul(4)?)?;
        let bytes = mapped.bytes.as_slice().get(offset..offset + 4)?;
        let pos = i32::from_le_bytes(bytes.try_into().ok()?);
        (pos >= 0).then_some(pos as usize)
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
        let mut cnt = rank.occ_sb[sb * ns + s];
        if b != sb * (FREQ_CNT_SB / FREQ_CNT_B) {
            cnt += rank.occ_b[b * ns + s] as usize;
        }
        let block_start = b * FREQ_CNT_B;
        if let Some(packed) = &self.packed_dna_bwt {
            return cnt + packed.count_eq(block_start, pos, ch);
        }
        for idx in block_start..pos {
            if self.bwt_at(idx) == Some(ch) {
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
                cnt += rank.occ_sb[sb * ns + s];
                if b != sb * (FREQ_CNT_SB / FREQ_CNT_B) {
                    cnt += rank.occ_b[b * ns + s] as usize;
                }
            }
        }
        let block_start = b * FREQ_CNT_B;
        if let Some(packed) = &self.packed_dna_bwt {
            return cnt + packed.count_lt(block_start, pos, ch);
        }
        for idx in block_start..pos {
            if self.bwt_at(idx).is_some_and(|x| x < ch) {
                cnt += 1;
            }
        }
        cnt
    }

    /// Total occurrences of `ch` across the whole BWT (= count of `ch` in the
    /// indexed text). Used to seed a single-character interval.
    fn occ_total(&self, ch: u8) -> usize {
        self.occ(ch, self.bwt_len())
    }

    /// Half-open interval over the suffix array for a single character `ch`
    /// (the starting point of a forward or backward search). Empty if `ch`
    /// does not occur.
    pub fn char_interval(&self, ch: u8) -> Option<FmInterval> {
        if ch == 0 {
            return None;
        }
        let lo = self.c[ch as usize];
        let hi = lo.checked_add(self.occ_total(ch))?;
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
        if ch == 0
            || bk.is_empty()
            || fwd.is_empty()
            || bk.hi > self.bwt_len()
            || fwd.hi > self.bwt_len()
        {
            return None;
        }
        let occ_lo = self.occ(ch, bk.lo);
        let occ_hi = self.occ(ch, bk.hi);
        // New mirror (backward) interval: normal backward-search step on `ch`.
        let new_bk_lo = self.c[ch as usize].checked_add(occ_lo)?;
        let new_bk_hi = self.c[ch as usize].checked_add(occ_hi)?;
        if new_bk_lo >= new_bk_hi {
            return None;
        }
        // New forward interval: shift `fwd.lo` by the number of suffixes in the
        // current mirror range that carry a symbol strictly smaller than `ch`,
        // then take the same size as the new mirror range.
        let occ_lt_lo = self.occ_lt(ch, bk.lo);
        let occ_lt_hi = self.occ_lt(ch, bk.hi);
        let fwd_delta = occ_lt_hi.checked_sub(occ_lt_lo)?;
        let interval_len = new_bk_hi.checked_sub(new_bk_lo)?;
        let new_fwd_lo = fwd.lo.checked_add(fwd_delta)?;
        let new_fwd_hi = new_fwd_lo.checked_add(interval_len)?;
        if new_fwd_hi > self.bwt_len() {
            return None;
        }
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
    fn sampled_fm_index_locates_like_full_suffix_array() {
        let original = FmIndex::build(b"ACGTACGTGCAACGTACGT");
        let sampled = FmIndex::from_parts_sampled(
            original.bwt.clone(),
            original.sa.clone(),
            original.c,
            original.n,
        )
        .unwrap();

        for pattern in [b"ACG".as_slice(), b"TGC".as_slice(), b"CGTA".as_slice()] {
            let mut expected = original.locate(pattern);
            let mut got = sampled.locate(pattern);
            expected.sort_unstable();
            got.sort_unstable();
            assert_eq!(got, expected, "sampled locate mismatch for {pattern:?}");
        }
    }

    #[test]
    fn sampled_fm_index_samples_suffix_array_by_bwt_row() {
        let original = FmIndex::build(b"ACGTACGTGCAACGTACGT");
        let sample_rate = 3;
        let sampled = FmIndex::from_parts_inner(
            original.bwt.clone(),
            original.sa.clone(),
            original.c,
            original.n,
            Some(sample_rate),
        )
        .unwrap();

        let expected: Vec<(i32, i32)> = original
            .sa
            .iter()
            .enumerate()
            .filter(|(row, _)| row % sample_rate == 0)
            .map(|(row, &pos)| (row as i32, pos))
            .collect();
        let position_sampled: Vec<(i32, i32)> = original
            .sa
            .iter()
            .enumerate()
            .filter(|(_, &pos)| (pos as usize) % sample_rate == 0)
            .map(|(row, &pos)| (row as i32, pos))
            .collect();

        assert_ne!(position_sampled, expected);
        assert_eq!(sampled.sa_samples, expected);
    }

    #[test]
    fn fm_index_rejects_inconsistent_serialized_parts() {
        let original = FmIndex::build(b"ACGT");
        let err = FmIndex::from_parts(original.bwt, original.sa, original.c, 99).unwrap_err();

        assert!(err.contains("BWT length"));
    }

    #[test]
    fn fm_index_rejects_c_table_that_does_not_match_bwt_counts() {
        let original = FmIndex::build(b"ACGT");
        let mut c = original.c;
        c[b'C' as usize] += 1;

        let err = FmIndex::from_parts(original.bwt, original.sa, c, original.n).unwrap_err();
        assert!(err.contains("C table entry"));
    }

    #[test]
    fn fm_index_rejects_bwt_without_exactly_one_sentinel() {
        let original = FmIndex::build(b"ACGT");
        let mut bwt = original.bwt;
        let sentinel = bwt.iter().position(|&ch| ch == 0).unwrap();
        bwt[sentinel] = b'A';

        let err = FmIndex::from_parts(bwt, original.sa, original.c, original.n).unwrap_err();
        assert!(err.contains("exactly one sentinel"));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn fm_index_rejects_suffix_array_duplicate_positions() {
        let original = FmIndex::build(b"ACGT");
        let mut sa = original.sa;
        sa[1] = sa[0];

        let err = FmIndex::from_parts(original.bwt, sa, original.c, original.n).unwrap_err();
        assert!(err.contains("not a permutation"));
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

    #[test]
    fn fm_index_forward_update_rejects_invalid_forward_interval() {
        let text = b"ACGTACGT";
        let fmb = FmIndex::build(&text.iter().rev().copied().collect::<Vec<_>>());
        let bk = fmb.char_interval(b'A').unwrap();
        let invalid_fwd = FmInterval {
            lo: 0,
            hi: fmb.bwt_len() + 1,
        };

        assert!(fmb.update_interval_forward(bk, invalid_fwd, b'C').is_none());
    }

    /// Regression: `occ`/`count`/`locate` must be correct at exact block/
    /// superblock boundaries. Text lengths ≡ 255 (mod 256) make `bwt.len()` a
    /// multiple of FREQ_CNT_B; a missing trailing sample row previously made
    /// every search return 0 hits at those lengths.
    #[test]
    fn fm_index_occ_correct_at_block_boundaries() {
        let bases = [b'A', b'C', b'G', b'T'];
        // Cover text_len around the 256/512 BWT-length boundaries (bwt = len+1).
        for &text_len in &[254usize, 255, 256, 257, 510, 511, 512, 767] {
            let text: Vec<u8> = (0..text_len).map(|i| bases[(i * 7 + 3) % 4]).collect();
            let fm = FmIndex::build(&text);
            for pat in [
                &b"ACGT"[..],
                &b"A"[..],
                &b"GT"[..],
                &b"TAC"[..],
                &b"ACGTAC"[..],
            ] {
                if pat.len() > text_len {
                    continue;
                }
                // Brute-force count of overlapping occurrences.
                let expected = text.windows(pat.len()).filter(|w| *w == pat).count();
                assert_eq!(
                    fm.count(pat),
                    expected,
                    "count({:?}) wrong at text_len={text_len}",
                    std::str::from_utf8(pat).unwrap()
                );
                let mut got = fm.locate(pat);
                got.sort_unstable();
                let mut want: Vec<usize> = (0..text_len)
                    .filter(|&i| i + pat.len() <= text_len && &text[i..i + pat.len()] == pat)
                    .collect();
                want.sort_unstable();
                assert_eq!(
                    got,
                    want,
                    "locate({:?}) wrong at text_len={text_len}",
                    std::str::from_utf8(pat).unwrap()
                );
            }
        }
    }

    #[test]
    fn makehmmerdb_occ_tables_do_not_double_count_superblock_boundary() {
        let text: Vec<u8> = (0..70_000)
            .map(|i| [b'A', b'C', b'G', b'T'][(i * 11 + 1) % 4])
            .collect();
        let full = FmIndex::build(&text);
        let term_loc = full.bwt.iter().position(|&ch| ch == 0).unwrap();
        let raw_sa_samples = full
            .sa
            .iter()
            .enumerate()
            .filter_map(|(row, &pos)| {
                (row % 8 == 0).then_some(if pos as usize == text.len() { -1 } else { pos })
            })
            .collect::<Vec<_>>();

        let nb = 1 + full.bwt.len().div_ceil(FREQ_CNT_B);
        let nsb = 1 + full.bwt.len().div_ceil(FREQ_CNT_SB);
        let mut occ_b = vec![0u16; nb * 4];
        let mut occ_sb = vec![0u32; nsb * 4];
        let mut cnt_b = [0u16; 4];
        let mut cnt_sb = [0u32; 4];
        for (j, &base) in full.bwt.iter().enumerate() {
            let code = match base {
                0 | b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                _ => unreachable!(),
            };
            cnt_b[code] += 1;
            cnt_sb[code] += 1;
            let joffset = j + 1;
            if joffset % FREQ_CNT_B == 0 {
                let row = joffset / FREQ_CNT_B;
                occ_b[row * 4..row * 4 + 4].copy_from_slice(&cnt_b);
                if joffset % FREQ_CNT_SB == 0 {
                    let row = joffset / FREQ_CNT_SB;
                    occ_sb[row * 4..row * 4 + 4].copy_from_slice(&cnt_sb);
                    cnt_b = [0; 4];
                }
            }
        }
        occ_b[(nb - 1) * 4..nb * 4].copy_from_slice(&cnt_b);
        occ_sb[(nsb - 1) * 4..nsb * 4].copy_from_slice(&cnt_sb);

        let raw = FmIndex::from_makehmmerdb_raw_parts(
            full.bwt.clone(),
            raw_sa_samples,
            term_loc,
            8,
            FREQ_CNT_B,
            occ_b,
            occ_sb,
        )
        .unwrap();

        for pat in [&b"ACGT"[..], &b"CGTA"[..], &b"TACG"[..], &b"A"[..]] {
            assert_eq!(
                raw.count(pat),
                full.count(pat),
                "count mismatch for {pat:?}"
            );
            let mut got = raw.locate(pat);
            got.sort_unstable();
            let mut want = full.locate(pat);
            want.sort_unstable();
            assert_eq!(got, want, "locate mismatch for {pat:?}");
        }
    }
}
