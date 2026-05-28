//! Easel dsqdata format: a binary sequence database for fast reading.
//!
//! This is a faithful port of the **on-disk byte format** of Easel's
//! `esl_dsqdata.c` (HMMER/Easel 3.4). Files written here are byte-compatible
//! with Easel's `esl_dsqdata_Write()`, and files written by Easel can be read
//! back by [`read_dsqdata`]. See `hmmer/easel/esl_dsqdata.c` for the
//! source-of-truth.

#![allow(clippy::needless_late_init)]
//!
//! ## On-disk layout (four files)
//!
//! A dsqdata "database" is a stub file `<basename>` plus three binary files:
//!   - `<basename>.dsqi` — index file
//!   - `<basename>.dsqm` — metadata file
//!   - `<basename>.dsqs` — packed sequence file
//!
//! The four files are linked by a 32-bit `uniquetag`.
//!
//! ### Stub file (`<basename>`, text)
//! First line is machine-parsed:
//! ```text
//! Easel dsqdata v1 x<uniquetag>
//! ```
//! followed by human-readable lines (original file/format, type, counts).
//!
//! ### Index file (`.dsqi`)
//! Header: 7 × `uint32` then 3 × `uint64`, little-endian:
//!   `magic, uniquetag, alphatype, flags, max_namelen, max_acclen, max_desclen,`
//!   `max_seqlen, nseq, nres`.
//! Then `nseq` records of `ESL_DSQDATA_RECORD` = two `int64`:
//!   `metadata_end`, `psq_end` — cumulative END offsets (inclusive, i.e. the
//!   offset of this record's *last* element). `metadata_end` is a byte offset
//!   into `.dsqm` (after the 8-byte header); `psq_end` is a *packet* (uint32)
//!   offset into `.dsqs` (after the 8-byte header). Both can be -1 for the
//!   first record if it is empty.
//!
//! ### Metadata file (`.dsqm`)
//! Header: 2 × `uint32` (`magic, uniquetag`). Then per sequence:
//!   `name\0 acc\0 desc\0` followed by an `int32` taxid (-1 = none).
//!
//! ### Sequence file (`.dsqs`)
//! Header: 2 × `uint32` (`magic, uniquetag`). Then packed residues as a stream
//! of `uint32` packets (see packing below).
//!
//! ## Packing
//! Each packet is a `uint32`:
//! ```text
//!  [31]=EOD  [30]=5bit  [29..0]= 6×5-bit residues OR 15×2-bit residues
//! ```
//! 2-bit packing is used for nucleic alphabets (k<=4) where 15 consecutive
//! canonical residues are available; 5-bit packing is used otherwise (and
//! always for amino). The EOD (end-of-data) packet terminates a sequence and
//! may be a partial 5-bit packet whose unused residue slots are 0x1f. A
//! zero-length sequence is a single EOD packet `0xFFFFFFFF`.
//!
//! ## Uniquetag
//! The C `esl_dsqdata_Write()` generates the uniquetag via `esl_randomness_Create(0)`
//! (which seeds Mersenne Twister from `time()+pid+clock()`) followed by one call
//! to `esl_random_uint32()` — so C's uniquetag is **non-deterministic**. Any nonzero
//! uint32 is a valid tag as long as all four files agree. This Rust port instead
//! derives a **deterministic** tag by hashing the dataset's basic statistics
//! (`nseq`, `nres`, `max_seqlen`) with a fixed seed. Databases written by Rust and
//! by C are interoperable regardless of the differing tag derivation methods.
//!
//! ## Taxid
//! The metadata format stores a per-sequence `int32` taxid field (`-1` = none),
//! matching C `esl_dsqdata`'s `chu->taxid[i]`. It is read into and written from
//! [`Sequence::taxid`], so a write→read round-trip preserves it.
//!
//! ## Threaded loader
//! Easel's reader uses a loader thread (disk I/O) and several unpacker threads
//! (CPU decompression) running in a producer/consumer pipeline. This port
//! implements a **two-thread pipeline**: a background loader thread streams raw
//! packed bytes off disk in fixed-size chunks and sends them over a channel;
//! the main thread receives each raw chunk and unpacks it while the loader
//! prefetches the next one. The bytes processed and the sequences produced are
//! format-identical to the synchronous approach; only the overlap of I/O and
//! CPU work differs.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::mpsc;

use crate::alphabet::{Alphabet, AlphabetType, Dsq, DSQ_SENTINEL};
use crate::errors::{HmmerError, HmmerResult};
use crate::sequence::Sequence;

/// Maximum number of sequences per chunk in the threaded loader, matching
/// `eslDSQDATA_CHUNK_MAXSEQ` from the C implementation.
const CHUNK_MAXSEQ: usize = 4096;

/// A raw (still-packed) chunk sent from the loader thread to the main thread.
/// Contains the raw packed uint32 words, raw metadata bytes, and the index
/// records for each sequence in the chunk so the unpacker knows the boundaries.
struct RawChunk {
    /// Packed sequence data (uint32 packets) for this chunk's sequences.
    packed: Vec<u32>,
    /// Raw metadata bytes (name\0 acc\0 desc\0 taxid×4) for this chunk's sequences.
    metadata: Vec<u8>,
    /// Index records for each sequence in this chunk. Each record carries
    /// the *chunk-relative* cumulative end offsets (0-based within this chunk).
    records: Vec<IndexRecord>,
}

/// "dsq1" + 0x80808080. Detects format and byte order (port of `eslDSQDATA_MAGIC_V1`).
const MAGIC_V1: u32 = 0xc4d3d1b1;
/// Byte-swapped magic (port of `eslDSQDATA_MAGIC_V1SWAP`); read support unimplemented (matches C).
const MAGIC_V1SWAP: u32 = 0xb1d1d3c4;

/// Control bit: last packet in a packed sequence (port of `eslDSQDATA_EOD`).
const EOD_BIT: u32 = 1 << 31;
/// Control bit: packet is 5-bit packed (port of `eslDSQDATA_5BIT`).
const FIVEBIT_BIT: u32 = 1 << 30;
const DSQDATA_INDEX_HEADER_LEN: u64 = 7 * 4 + 3 * 8;
const DSQDATA_SIDECAR_HEADER_LEN: u64 = 8;
const DSQDATA_INDEX_RECORD_LEN: u64 = 16;
const MAX_DSQDATA_STUB_LINE_LEN: usize = 4096;

#[inline]
fn is_eod(v: u32) -> bool {
    v & EOD_BIT != 0
}
#[inline]
fn is_5bit(v: u32) -> bool {
    v & FIVEBIT_BIT != 0
}

/// One `.dsqi` index record: cumulative inclusive END offsets.
/// `metadata_end` is a byte offset into `.dsqm`; `psq_end` is a uint32-packet
/// offset into `.dsqs`. Stored on disk as two little-endian `int64`s, in the
/// order (metadata_end, psq_end) matching the C struct field order.
#[derive(Clone, Copy, Debug)]
struct IndexRecord {
    metadata_end: i64,
    psq_end: i64,
}

// ---------------------------------------------------------------------------
// Packing (port of dsqdata_pack5 / dsqdata_pack2)
// ---------------------------------------------------------------------------

/// 5-bit pack a digital sequence `dsq[1..=n]` (1-based, sentinel at 0). Returns
/// the packets. Port of `dsqdata_pack5()`.
fn pack5(dsq: &[Dsq], n: usize) -> Vec<u32> {
    let mut psq: Vec<u32> = Vec::new();
    let mut r = 1usize; // position in dsq (1-based)
    while r <= n {
        let mut v = FIVEBIT_BIT;
        let mut b: i32 = 25;
        while b >= 0 && r <= n {
            v |= (dsq[r] as u32) << b;
            r += 1;
            b -= 5;
        }
        while b >= 0 {
            v |= 31u32 << b;
            b -= 5;
        }
        if r > n {
            v |= EOD_BIT;
        }
        psq.push(v);
    }
    // n == 0: a single empty EOD sentinel packet (all bits set).
    if psq.is_empty() {
        psq.push(!0u32);
    }
    psq
}

/// 2-bit (+ 5-bit for noncanonicals) pack a digital nucleic sequence.
/// Port of `dsqdata_pack2()`.
fn pack2(dsq: &[Dsq], n: usize) -> Vec<u32> {
    let mut psq: Vec<u32> = Vec::new();
    let mut d = 0usize; // position of next degenerate residue (1..=n), n+1 if none
    let mut r = 1usize; // position in dsq (1-based)
    while r <= n {
        // Slide the "next degenerate residue" detector.
        if d < r {
            d = r;
            while d <= n {
                if dsq[d] > 3 {
                    break;
                }
                d += 1;
            }
        }

        let v;
        // Can we 2-bit pack the next 15 residues r..r+14?
        if n - r + 1 >= 15 && d > r + 14 {
            let mut vv = 0u32;
            let mut b: i32 = 28;
            while b >= 0 {
                vv |= (dsq[r] as u32) << b;
                r += 1;
                b -= 2;
            }
            v = vv;
        } else {
            let mut vv = FIVEBIT_BIT;
            let mut b: i32 = 25;
            while b >= 0 && r <= n {
                vv |= (dsq[r] as u32) << b;
                r += 1;
                b -= 5;
            }
            while b >= 0 {
                vv |= 31u32 << b;
                b -= 5;
            }
            v = vv;
        }

        let v = if r > n { v | EOD_BIT } else { v };
        psq.push(v);
    }
    if psq.is_empty() {
        psq.push(!0u32);
    }
    psq
}

// ---------------------------------------------------------------------------
// Unpacking (port of dsqdata_unpack5 / dsqdata_unpack2)
// ---------------------------------------------------------------------------

/// Unpack one 5-bit-encoded sequence starting at `psq[start..]`. Appends
/// residues to `out` (without sentinels). Returns (L, packets_consumed).
/// Port of `dsqdata_unpack5()`.
fn unpack5(psq: &[u32], start: usize, out: &mut Vec<Dsq>) -> HmmerResult<(usize, usize)> {
    let mut pos = start;
    let mut l = 0usize;
    let mut v = *psq
        .get(pos)
        .ok_or_else(|| HmmerError::Format("dsqdata: truncated packet stream".into()))?;
    pos += 1;

    while !is_eod(v) {
        for b in [25, 20, 15, 10, 5, 0] {
            out.push(((v >> b) & 31) as Dsq);
            l += 1;
        }
        v = *psq
            .get(pos)
            .ok_or_else(|| HmmerError::Format("dsqdata: truncated packet stream".into()))?;
        pos += 1;
    }
    // EOD packet, possibly partial. Stop at first 0x1f sentinel slot.
    let mut b: i32 = 25;
    while b >= 0 && ((v >> b) & 31) != 31 {
        out.push(((v >> b) & 31) as Dsq);
        l += 1;
        b -= 5;
    }
    Ok((l, pos - start))
}

/// Unpack one 2-bit (mixed 5-bit) encoded sequence. Port of `dsqdata_unpack2()`.
fn unpack2(psq: &[u32], start: usize, out: &mut Vec<Dsq>) -> HmmerResult<(usize, usize)> {
    let mut pos = start;
    let mut l = 0usize;
    let mut v = *psq
        .get(pos)
        .ok_or_else(|| HmmerError::Format("dsqdata: truncated packet stream".into()))?;
    pos += 1;

    while !is_eod(v) {
        if is_5bit(v) {
            for b in [25, 20, 15, 10, 5, 0] {
                out.push(((v >> b) & 31) as Dsq);
                l += 1;
            }
        } else {
            for b in [28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0] {
                out.push(((v >> b) & 3) as Dsq);
                l += 1;
            }
        }
        v = *psq
            .get(pos)
            .ok_or_else(|| HmmerError::Format("dsqdata: truncated packet stream".into()))?;
        pos += 1;
    }
    // EOD packet. 2-bit EOD is full; 5-bit EOD may be partial.
    if is_5bit(v) {
        let mut b: i32 = 25;
        while b >= 0 && ((v >> b) & 31) != 31 {
            out.push(((v >> b) & 31) as Dsq);
            l += 1;
            b -= 5;
        }
    } else {
        for b in [28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0] {
            out.push(((v >> b) & 3) as Dsq);
            l += 1;
        }
    }
    Ok((l, pos - start))
}

// ---------------------------------------------------------------------------
// Write (port of esl_dsqdata_Write)
// ---------------------------------------------------------------------------

/// Write a digital sequence database in Easel dsqdata format.
///
/// Creates the four files `<basename>`, `<basename>.dsqi`, `<basename>.dsqm`,
/// and `<basename>.dsqs`. Faithful port of `esl_dsqdata_Write()`. The
/// `alphabet` must be nucleic (DNA/RNA) or amino; nucleic uses 2-bit packing,
/// amino uses 5-bit packing, matching C.
///
/// Each sequence's `taxid` is written from [`Sequence::taxid`] (`-1` = none).
pub fn write_dsqdata(
    basename: &Path,
    sequences: &[Sequence],
    alphabet: &Alphabet,
) -> HmmerResult<()> {
    let alphatype = alphabet.abc_type;
    let do_pack5 = match alphatype {
        AlphabetType::Amino => true,
        AlphabetType::Dna | AlphabetType::Rna => false,
        AlphabetType::Unknown => {
            return Err(HmmerError::Format(
                "dsqdata: alphabet must be protein or nucleic".into(),
            ))
        }
    };

    // First pass: statistics.
    let mut max_namelen = 0u32;
    let mut max_acclen = 0u32;
    let mut max_desclen = 0u32;
    let mut max_seqlen = 0u64;
    let mut nres = 0u64;
    for sq in sequences {
        validate_dsq(sq)?;
        nres += sq.n as u64;
        max_seqlen = max_seqlen.max(sq.n as u64);
        max_namelen = max_namelen.max(sq.name.len() as u32);
        max_acclen = max_acclen.max(sq.acc.len() as u32);
        max_desclen = max_desclen.max(sq.desc.len() as u32);
    }
    let nseq = sequences.len() as u64;

    // A deterministic uniquetag. C uses a random uint32 from its RNG; any value
    // works as long as the four files agree, which they do here.
    let uniquetag: u32 = derive_uniquetag(nseq, nres, max_seqlen);

    let p = basename;
    let mut ifp = BufWriter::new(create(&with_ext(p, "dsqi"))?);
    let mut mfp = BufWriter::new(create(&with_ext(p, "dsqm"))?);
    let mut sfp = BufWriter::new(create(&with_ext(p, "dsqs"))?);
    let mut stubfp = BufWriter::new(create(p)?);

    // Index header: 7 u32 then 3 u64.
    write_u32(&mut ifp, MAGIC_V1)?;
    write_u32(&mut ifp, uniquetag)?;
    write_u32(&mut ifp, alphatype as u32)?;
    write_u32(&mut ifp, 0)?; // flags
    write_u32(&mut ifp, max_namelen)?;
    write_u32(&mut ifp, max_acclen)?;
    write_u32(&mut ifp, max_desclen)?;
    write_u64(&mut ifp, max_seqlen)?;
    write_u64(&mut ifp, nseq)?;
    write_u64(&mut ifp, nres)?;

    // Metadata + sequence headers: magic, uniquetag.
    write_u32(&mut mfp, MAGIC_V1)?;
    write_u32(&mut mfp, uniquetag)?;
    write_u32(&mut sfp, MAGIC_V1)?;
    write_u32(&mut sfp, uniquetag)?;

    // Second pass: per-sequence index/metadata/packed records.
    let mut spos: i64 = 0; // running count of packets written to .dsqs
    let mut mpos: i64 = 0; // running count of bytes written to .dsqm
    for sq in sequences {
        let psq = if do_pack5 {
            pack5(&sq.dsq, sq.n)
        } else {
            pack2(&sq.dsq, sq.n)
        };
        for &v in &psq {
            write_u32(&mut sfp, v)?;
        }
        spos += psq.len() as i64;

        // Metadata: name\0 acc\0 desc\0 taxid(int32; -1 = none)
        mpos += write_cstr(&mut mfp, sq.name.as_bytes())? as i64;
        mpos += write_cstr(&mut mfp, sq.acc.as_bytes())? as i64;
        mpos += write_cstr(&mut mfp, sq.desc.as_bytes())? as i64;
        write_i32(&mut mfp, sq.taxid)?;
        mpos += 4;

        // Index record: inclusive END offsets (hence -1).
        let idx = IndexRecord {
            metadata_end: mpos - 1,
            psq_end: spos - 1,
        };
        write_i64(&mut ifp, idx.metadata_end)?;
        write_i64(&mut ifp, idx.psq_end)?;
    }

    // Stub file (human-readable; first line machine-parsed).
    writeln!(stubfp, "Easel dsqdata v1 x{}", uniquetag).map_err(HmmerError::Io)?;
    writeln!(stubfp).map_err(HmmerError::Io)?;
    writeln!(stubfp, "Original file:   (rust hmmer-rs)").map_err(HmmerError::Io)?;
    writeln!(stubfp, "Original format: unknown").map_err(HmmerError::Io)?;
    writeln!(stubfp, "Type:            {}", decode_type(alphatype)).map_err(HmmerError::Io)?;
    writeln!(stubfp, "Sequences:       {}", nseq).map_err(HmmerError::Io)?;
    writeln!(stubfp, "Residues:        {}", nres).map_err(HmmerError::Io)?;

    ifp.flush().map_err(HmmerError::Io)?;
    mfp.flush().map_err(HmmerError::Io)?;
    sfp.flush().map_err(HmmerError::Io)?;
    stubfp.flush().map_err(HmmerError::Io)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Read (threaded loader + synchronous unpacker)
// ---------------------------------------------------------------------------

/// Open and read an entire dsqdata database from `<basename>` and its three
/// `.dsq?` companion files.
///
/// Implements a two-thread pipeline mirroring Easel's loader/unpacker design:
/// a background **loader thread** streams raw packed bytes and metadata off
/// disk in chunks of up to [`CHUNK_MAXSEQ`] sequences and sends each
/// [`RawChunk`] over a channel; the main thread receives each chunk and
/// unpacks (decompresses) it while the loader prefetches the next one. This
/// overlaps disk I/O with CPU decompression, faithfully reflecting C's
/// threaded pipeline (which uses more threads but the same two-stage
/// loader→unpacker structure).
///
/// Magic/uniquetag validation is identical to C. The alphabet is determined
/// from the index header's `alphatype` (as C does when the caller passes a
/// NULL alphabet).
///
/// Returns the sequences with their leading/trailing [`DSQ_SENTINEL`] bytes
/// restored. Accession/description are populated from the metadata file.
/// Taxid is read (and format-validated) but not stored; see module docs.
pub fn read_dsqdata(basename: &Path) -> HmmerResult<Vec<Sequence>> {
    let p = basename;

    // ---- Stub file: parse "Easel dsqdata v1 x<tag>" first line. ----
    let stub_file = open(p)?;
    let mut stub_reader = BufReader::new(stub_file);
    let first = crate::hmmfile::read_capped_text_line(&mut stub_reader, MAX_DSQDATA_STUB_LINE_LEN)?
        .ok_or_else(|| HmmerError::Format("dsqdata stub file is empty".into()))?;
    let mut toks = first.split_whitespace();
    if toks.next() != Some("Easel") || toks.next() != Some("dsqdata") {
        return Err(HmmerError::Format("dsqdata stub has bad format".into()));
    }
    let vtok = toks
        .next()
        .ok_or_else(|| HmmerError::Format("dsqdata stub missing version".into()))?;
    if !vtok.starts_with('v') || vtok[1..].parse::<u64>().is_err() {
        return Err(HmmerError::Format("dsqdata stub has bad version".into()));
    }
    let xtok = toks
        .next()
        .ok_or_else(|| HmmerError::Format("dsqdata stub missing tag".into()))?;
    if !xtok.starts_with('x') {
        return Err(HmmerError::Format("dsqdata stub has bad tag".into()));
    }
    let stub_tag: u32 = xtok[1..]
        .parse()
        .map_err(|_| HmmerError::Format("dsqdata stub has non-integer tag".into()))?;

    // ---- Index file header. ----
    let dsqi_path = with_ext(p, "dsqi");
    let dsqm_path = with_ext(p, "dsqm");
    let dsqs_path = with_ext(p, "dsqs");

    let ifile = open(&dsqi_path)?;
    let ifile_len = ifile.metadata().map_err(HmmerError::Io)?.len();
    let mut ifp = BufReader::new(ifile);
    let magic = read_u32(&mut ifp)?;
    let tag = read_u32(&mut ifp)?;
    let alphatype = read_u32(&mut ifp)?;
    let _flags = read_u32(&mut ifp)?;
    let _max_namelen = read_u32(&mut ifp)?;
    let _max_acclen = read_u32(&mut ifp)?;
    let _max_desclen = read_u32(&mut ifp)?;
    let _max_seqlen = read_u64(&mut ifp)?;
    let nseq = read_u64(&mut ifp)?;
    let _nres = read_u64(&mut ifp)?;

    if tag != stub_tag {
        return Err(HmmerError::Format(
            "dsqdata index file has bad tag, doesn't match stub".into(),
        ));
    }
    if magic == MAGIC_V1SWAP {
        return Err(HmmerError::Format(
            "dsqdata: cannot read byte-swapped data (unimplemented, as in C)".into(),
        ));
    }
    if magic != MAGIC_V1 {
        return Err(HmmerError::Format(
            "dsqdata index file has bad magic".into(),
        ));
    }
    let alphabet = match alphatype {
        1 => Alphabet::rna(),
        2 => Alphabet::dna(),
        3 => Alphabet::amino(),
        _ => {
            return Err(HmmerError::Format(format!(
                "dsqdata: invalid/unsupported alphabet type {}",
                alphatype
            )))
        }
    };
    let do_pack5 = alphabet.abc_type == AlphabetType::Amino;

    let nseq_usize = validate_dsqdata_nseq(nseq, ifile_len)?;

    // ---- Read all index records into memory (small: 16 bytes × nseq). ----
    // We read the full index up front so the loader thread can seek freely.
    let mut all_records: Vec<IndexRecord> = Vec::with_capacity(nseq_usize);
    for _ in 0..nseq_usize {
        let metadata_end = read_i64(&mut ifp)?;
        let psq_end = read_i64(&mut ifp)?;
        all_records.push(IndexRecord {
            metadata_end,
            psq_end,
        });
    }
    drop(ifp); // index fully consumed

    // ---- Validate metadata and sequence file headers. ----
    let mfile = open(&dsqm_path)?;
    let mfile_len = mfile.metadata().map_err(HmmerError::Io)?.len();
    let mut mfp = BufReader::new(mfile);
    let m_magic = read_u32(&mut mfp)?;
    let m_tag = read_u32(&mut mfp)?;
    if m_magic != magic {
        return Err(HmmerError::Format(
            "dsqdata metadata file has bad magic".into(),
        ));
    }
    if m_tag != stub_tag {
        return Err(HmmerError::Format(
            "dsqdata metadata file has bad tag, doesn't match stub".into(),
        ));
    }

    let sfile = open(&dsqs_path)?;
    let sfile_len = sfile.metadata().map_err(HmmerError::Io)?.len();
    let mut sfp = BufReader::new(sfile);
    let s_magic = read_u32(&mut sfp)?;
    let s_tag = read_u32(&mut sfp)?;
    if s_magic != magic {
        return Err(HmmerError::Format(
            "dsqdata sequence file has bad magic".into(),
        ));
    }
    if s_tag != stub_tag {
        return Err(HmmerError::Format(
            "dsqdata sequence file has bad tag, doesn't match stub".into(),
        ));
    }

    validate_dsqdata_index_extents(&all_records, mfile_len, sfile_len)?;

    // ---- Spawn the loader thread. ----
    // The loader reads raw bytes in chunks of CHUNK_MAXSEQ sequences and sends
    // RawChunk values over the channel. The main thread receives and unpacks
    // each chunk while the loader prefetches the next one.
    //
    // Channel is bounded to 1 so the loader stays at most one chunk ahead,
    // bounding memory use while still achieving I/O/CPU overlap.
    let (tx, rx) = mpsc::sync_channel::<Result<RawChunk, String>>(1);

    let loader_handle = std::thread::spawn(move || {
        let mut psq_last: i64 = -1;
        let mut meta_last: i64 = -1;
        let mut i = 0usize;
        let nseq_total = all_records.len();

        while i < nseq_total {
            // Take up to CHUNK_MAXSEQ index records for this chunk.
            let chunk_end = (i + CHUNK_MAXSEQ).min(nseq_total);
            let chunk_records = &all_records[i..chunk_end];

            let last_rec = chunk_records[chunk_records.len() - 1];

            // Number of packets in this chunk.
            let n_packets = (last_rec.psq_end - psq_last) as usize;
            // Number of metadata bytes in this chunk.
            let n_meta = (last_rec.metadata_end - meta_last) as usize;

            // Read packed sequence data.
            let mut packed_bytes = vec![0u8; n_packets * 4];
            if let Err(e) = sfp.read_exact(&mut packed_bytes) {
                let _ = tx.send(Err(format!("dsqdata loader: seq read error: {}", e)));
                return;
            }
            let packed: Vec<u32> = packed_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // Read metadata.
            let mut metadata = vec![0u8; n_meta];
            if let Err(e) = mfp.read_exact(&mut metadata) {
                let _ = tx.send(Err(format!("dsqdata loader: meta read error: {}", e)));
                return;
            }

            // Adjust index records to be chunk-relative (subtract the offsets
            // that were consumed before this chunk started).
            let psq_base = psq_last;
            let meta_base = meta_last;
            let records: Vec<IndexRecord> = chunk_records
                .iter()
                .map(|r| IndexRecord {
                    psq_end: r.psq_end - psq_base - 1, // convert to 0-based packet index within chunk
                    metadata_end: r.metadata_end - meta_base - 1, // 0-based byte index within chunk
                })
                .collect();

            psq_last = last_rec.psq_end;
            meta_last = last_rec.metadata_end;
            i = chunk_end;

            if tx
                .send(Ok(RawChunk {
                    packed,
                    metadata,
                    records,
                }))
                .is_err()
            {
                // Receiver dropped (main thread errored out). Stop loading.
                return;
            }
        }
        // Loader done; channel closes automatically when `tx` is dropped here.
    });

    // ---- Main thread: receive and unpack each chunk. ----
    let mut sequences: Vec<Sequence> = Vec::with_capacity(nseq_usize);

    for raw in rx {
        let chunk = raw.map_err(HmmerError::Format)?;
        unpack_chunk(&chunk, do_pack5, &mut sequences)?;
    }

    // Join the loader thread to propagate any panics.
    loader_handle
        .join()
        .map_err(|_| HmmerError::Format("dsqdata loader thread panicked".into()))?;

    Ok(sequences)
}

/// Unpack a [`RawChunk`] into [`Sequence`]s, appending to `out`.
///
/// This is the "unpacker" side of the two-thread pipeline. It mirrors
/// `dsqdata_unpack_chunk()` in the C implementation but operates on
/// chunk-relative offsets pre-computed by the loader thread.
fn unpack_chunk(chunk: &RawChunk, do_pack5: bool, out: &mut Vec<Sequence>) -> HmmerResult<()> {
    let psq = &chunk.packed;
    let metadata = &chunk.metadata;

    // The chunk's records have chunk-relative cumulative end offsets
    // (0-based, inclusive). Walk them to recover per-sequence slices.
    let mut psq_last: i64 = -1;
    let mut meta_last: i64 = -1;

    for rec in &chunk.records {
        // Packet range for this sequence within the chunk.
        let pstart = (psq_last + 1) as usize;
        let pn = (rec.psq_end - psq_last) as usize;
        if pn == 0 {
            return Err(HmmerError::Format(
                "dsqdata: index record encodes zero packets (every seq needs an EOD packet)".into(),
            ));
        }

        let mut residues = Vec::new();
        let (l, consumed) = if do_pack5 {
            unpack5(psq, pstart, &mut residues)?
        } else {
            unpack2(psq, pstart, &mut residues)?
        };
        if consumed != pn {
            return Err(HmmerError::Format(format!(
                "dsqdata: packet count mismatch (index says {}, unpacked {})",
                pn, consumed
            )));
        }
        psq_last = rec.psq_end;

        // Metadata range for this sequence within the chunk.
        let mstart = (meta_last + 1) as usize;
        let mend = (rec.metadata_end + 1) as usize; // exclusive
        if mend > metadata.len() || mstart > mend {
            return Err(HmmerError::Format(
                "dsqdata: metadata offset out of range".into(),
            ));
        }
        let block = &metadata[mstart..mend];
        // name\0 acc\0 desc\0 taxid(int32)
        let (name, rest) = take_cstr(block)?;
        let (acc, rest) = take_cstr(rest)?;
        let (desc, rest) = take_cstr(rest)?;
        if rest.len() != 4 {
            return Err(HmmerError::Format(
                "dsqdata: metadata record must end with exactly one taxid".into(),
            ));
        }
        // taxid: a per-sequence NCBI taxonomy id (int32; -1 = none), matching
        // C `esl_dsqdata`'s `chu->taxid[i]`.
        let taxid = i32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
        meta_last = rec.metadata_end;

        let mut dsq = Vec::with_capacity(l + 2);
        dsq.push(DSQ_SENTINEL);
        dsq.extend_from_slice(&residues);
        dsq.push(DSQ_SENTINEL);

        out.push(Sequence {
            name: String::from_utf8_lossy(name).into_owned(),
            acc: String::from_utf8_lossy(acc).into_owned(),
            desc: String::from_utf8_lossy(desc).into_owned(),
            dsq,
            n: l,
            l,
            taxid,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_dsq(sq: &Sequence) -> HmmerResult<()> {
    for (field, value) in [
        ("name", sq.name.as_bytes()),
        ("accession", sq.acc.as_bytes()),
        ("description", sq.desc.as_bytes()),
    ] {
        if value.contains(&0) {
            return Err(HmmerError::Format(format!(
                "Sequence {} {field} contains an embedded NUL byte",
                sq.name
            )));
        }
    }
    if sq.dsq.len() < sq.n + 2 {
        return Err(HmmerError::Format(format!(
            "Sequence {} digital data is shorter than declared length {}",
            sq.name, sq.n
        )));
    }
    if sq.dsq.first() != Some(&DSQ_SENTINEL) || sq.dsq.get(sq.n + 1) != Some(&DSQ_SENTINEL) {
        return Err(HmmerError::Format(format!(
            "Sequence {} is missing digital sentinels",
            sq.name
        )));
    }
    Ok(())
}

fn validate_dsqdata_nseq(nseq: u64, index_file_len: u64) -> HmmerResult<usize> {
    if index_file_len < DSQDATA_INDEX_HEADER_LEN {
        return Err(HmmerError::Format("dsqdata index file is truncated".into()));
    }
    let index_payload = index_file_len - DSQDATA_INDEX_HEADER_LEN;
    let max_records = index_payload / DSQDATA_INDEX_RECORD_LEN;
    if nseq > max_records {
        return Err(HmmerError::Format(format!(
            "dsqdata index declares {nseq} sequences but only {max_records} index records fit in the file"
        )));
    }
    usize::try_from(nseq).map_err(|_| {
        HmmerError::Format(format!(
            "dsqdata index declares too many sequences for this platform: {nseq}"
        ))
    })
}

fn validate_dsqdata_index_extents(
    records: &[IndexRecord],
    metadata_file_len: u64,
    sequence_file_len: u64,
) -> HmmerResult<()> {
    if metadata_file_len < DSQDATA_SIDECAR_HEADER_LEN
        || sequence_file_len < DSQDATA_SIDECAR_HEADER_LEN
    {
        return Err(HmmerError::Format(
            "dsqdata sidecar file is truncated".into(),
        ));
    }
    let metadata_payload_len = metadata_file_len - DSQDATA_SIDECAR_HEADER_LEN;
    let sequence_payload_bytes = sequence_file_len - DSQDATA_SIDECAR_HEADER_LEN;
    if !sequence_payload_bytes.is_multiple_of(4) {
        return Err(HmmerError::Format(
            "dsqdata sequence file has a partial trailing packet".into(),
        ));
    }
    let sequence_packets = sequence_payload_bytes / 4;
    if records.is_empty() {
        if metadata_payload_len != 0 || sequence_packets != 0 {
            return Err(HmmerError::Format(
                "dsqdata sidecar payload has trailing bytes with zero sequences".into(),
            ));
        }
        return Ok(());
    }

    let mut prev_meta = -1i64;
    let mut prev_psq = -1i64;
    for (idx, rec) in records.iter().enumerate() {
        if rec.metadata_end < prev_meta || rec.psq_end <= prev_psq {
            return Err(HmmerError::Format(format!(
                "dsqdata index record {} is not monotonic",
                idx + 1
            )));
        }
        if rec.metadata_end < 0 {
            return Err(HmmerError::Format(format!(
                "dsqdata index record {} has negative metadata extent",
                idx + 1
            )));
        }
        let metadata_end = u64::try_from(rec.metadata_end).map_err(|_| {
            HmmerError::Format(format!(
                "dsqdata index record {} has invalid metadata extent",
                idx + 1
            ))
        })?;
        let psq_end = u64::try_from(rec.psq_end).map_err(|_| {
            HmmerError::Format(format!(
                "dsqdata index record {} has invalid packet extent",
                idx + 1
            ))
        })?;
        if metadata_end >= metadata_payload_len {
            return Err(HmmerError::Format(format!(
                "dsqdata index record {} metadata extent exceeds .dsqm payload",
                idx + 1
            )));
        }
        if psq_end >= sequence_packets {
            return Err(HmmerError::Format(format!(
                "dsqdata index record {} packet extent exceeds .dsqs payload",
                idx + 1
            )));
        }
        prev_meta = rec.metadata_end;
        prev_psq = rec.psq_end;
    }
    let final_metadata_len = u64::try_from(prev_meta + 1)
        .map_err(|_| HmmerError::Format("dsqdata final metadata extent is invalid".into()))?;
    let final_sequence_packets = u64::try_from(prev_psq + 1)
        .map_err(|_| HmmerError::Format("dsqdata final packet extent is invalid".into()))?;
    if final_metadata_len != metadata_payload_len {
        return Err(HmmerError::Format(
            "dsqdata metadata sidecar has trailing bytes after final index record".into(),
        ));
    }
    if final_sequence_packets != sequence_packets {
        return Err(HmmerError::Format(
            "dsqdata sequence sidecar has trailing packets after final index record".into(),
        ));
    }
    Ok(())
}

fn with_ext(base: &Path, ext: &str) -> std::path::PathBuf {
    let mut s = base.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    std::path::PathBuf::from(s)
}

fn create(path: &Path) -> HmmerResult<std::fs::File> {
    std::fs::File::create(path).map_err(HmmerError::Io)
}
fn open(path: &Path) -> HmmerResult<std::fs::File> {
    std::fs::File::open(path).map_err(HmmerError::Io)
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> HmmerResult<()> {
    w.write_all(&v.to_le_bytes()).map_err(HmmerError::Io)
}
fn write_i32<W: Write>(w: &mut W, v: i32) -> HmmerResult<()> {
    w.write_all(&v.to_le_bytes()).map_err(HmmerError::Io)
}
fn write_u64<W: Write>(w: &mut W, v: u64) -> HmmerResult<()> {
    w.write_all(&v.to_le_bytes()).map_err(HmmerError::Io)
}
fn write_i64<W: Write>(w: &mut W, v: i64) -> HmmerResult<()> {
    w.write_all(&v.to_le_bytes()).map_err(HmmerError::Io)
}
/// Write a NUL-terminated C string; returns total bytes written (len+1).
fn write_cstr<W: Write>(w: &mut W, s: &[u8]) -> HmmerResult<usize> {
    w.write_all(s).map_err(HmmerError::Io)?;
    w.write_all(&[0u8]).map_err(HmmerError::Io)?;
    Ok(s.len() + 1)
}

fn read_u32<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(HmmerError::Io)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> HmmerResult<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(HmmerError::Io)?;
    Ok(u64::from_le_bytes(b))
}
fn read_i64<R: Read>(r: &mut R) -> HmmerResult<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(HmmerError::Io)?;
    Ok(i64::from_le_bytes(b))
}

/// Split off a NUL-terminated C string from the front of `buf`. Returns the
/// string bytes (excluding the NUL) and the remainder (after the NUL).
fn take_cstr(buf: &[u8]) -> HmmerResult<(&[u8], &[u8])> {
    match buf.iter().position(|&b| b == 0) {
        Some(i) => Ok((&buf[..i], &buf[i + 1..])),
        None => Err(HmmerError::Format(
            "dsqdata: metadata string not NUL-terminated".into(),
        )),
    }
}

fn decode_type(t: AlphabetType) -> &'static str {
    match t {
        AlphabetType::Rna => "RNA",
        AlphabetType::Dna => "DNA",
        AlphabetType::Amino => "amino",
        AlphabetType::Unknown => "unknown",
    }
}

/// Derive a deterministic uniquetag from dataset statistics.
///
/// The C `esl_dsqdata_Write()` generates the tag by calling
/// `esl_randomness_Create(0)` (which seeds a Mersenne Twister from
/// `time() + pid + clock()`, making it non-deterministic / different on every
/// run) and then drawing one `uint32_t`. Any nonzero `uint32` is valid as a
/// tag, provided all four files agree. This Rust implementation instead
/// produces a **deterministic** tag by mixing the dataset's basic statistics
/// with a fixed nonzero starting constant. Files written by Rust and files
/// written by C are interoperable regardless of the different tag derivation
/// methods, because the reader only checks that all four files carry the same
/// tag — it does not mandate a specific derivation method.
fn derive_uniquetag(nseq: u64, nres: u64, max_seqlen: u64) -> u32 {
    let mut h = 0x9E3779B9u32;
    for v in [nseq, nres, max_seqlen] {
        h = h
            .wrapping_mul(2654435761)
            .wrapping_add((v as u32) ^ ((v >> 32) as u32));
    }
    h | 1 // ensure nonzero
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        dir.join(format!("hmmerrs_dsqdata_{}_{}", std::process::id(), name))
    }

    fn cleanup(base: &Path) {
        std::fs::remove_file(base).ok();
        std::fs::remove_file(with_ext(base, "dsqi")).ok();
        std::fs::remove_file(with_ext(base, "dsqm")).ok();
        std::fs::remove_file(with_ext(base, "dsqs")).ok();
    }

    fn write_le_i64_at(buf: &mut [u8], offset: usize, value: i64) {
        buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn roundtrip_amino_5bit() {
        let abc = Alphabet::amino();
        let seqs = vec![
            Sequence {
                name: "seq1".into(),
                acc: "ACC1".into(),
                desc: "first seq".into(),
                dsq: abc.digitize(b"ACDEFGHIKLMNPQRSTVWY"),
                n: 20,
                l: 20,
                taxid: -1,
            },
            Sequence {
                name: "seq2".into(),
                acc: String::new(),
                desc: String::new(),
                dsq: abc.digitize(b"MKV"),
                n: 3,
                l: 3,
                taxid: -1,
            },
        ];

        let base = tmp("amino");
        write_dsqdata(&base, &seqs, &abc).unwrap();
        let got = read_dsqdata(&base).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name, "seq1");
        assert_eq!(got[0].acc, "ACC1");
        assert_eq!(got[0].desc, "first seq");
        assert_eq!(got[0].n, 20);
        assert_eq!(got[0].dsq, seqs[0].dsq);
        assert_eq!(got[1].name, "seq2");
        assert_eq!(got[1].n, 3);
        assert_eq!(got[1].dsq, seqs[1].dsq);
        cleanup(&base);
    }

    #[test]
    fn roundtrip_dna_2bit_with_degenerates() {
        let abc = Alphabet::dna();
        // Mix of canonical runs and degenerate residues to exercise both 2-bit
        // and 5-bit packets within one sequence.
        let raw = b"ACGTACGTACGTACGTNNACGTNRYACGTACGTACGT";
        let seqs = vec![Sequence {
            name: "dna1".into(),
            acc: String::new(),
            desc: String::new(),
            dsq: abc.digitize(raw),
            n: raw.len(),
            l: raw.len(),
            taxid: -1,
        }];

        let base = tmp("dna");
        write_dsqdata(&base, &seqs, &abc).unwrap();
        let got = read_dsqdata(&base).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].n, raw.len());
        assert_eq!(got[0].dsq, seqs[0].dsq);
        cleanup(&base);
    }

    #[test]
    fn roundtrip_empty_sequence() {
        let abc = Alphabet::amino();
        let seqs = vec![Sequence {
            name: "empty".into(),
            acc: String::new(),
            desc: String::new(),
            dsq: vec![DSQ_SENTINEL, DSQ_SENTINEL],
            n: 0,
            l: 0,
            taxid: -1,
        }];
        let base = tmp("empty");
        write_dsqdata(&base, &seqs, &abc).unwrap();
        let got = read_dsqdata(&base).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].n, 0);
        assert_eq!(got[0].name, "empty");
        assert_eq!(got[0].dsq, vec![DSQ_SENTINEL, DSQ_SENTINEL]);
        cleanup(&base);
    }

    #[test]
    fn rejects_dsqdata_trailing_sidecar_payload_after_final_record() {
        let records = [IndexRecord {
            metadata_end: 3,
            psq_end: 0,
        }];
        let err = validate_dsqdata_index_extents(
            &records,
            DSQDATA_SIDECAR_HEADER_LEN + 5,
            DSQDATA_SIDECAR_HEADER_LEN + 4,
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("metadata sidecar has trailing bytes"));

        let err = validate_dsqdata_index_extents(
            &records,
            DSQDATA_SIDECAR_HEADER_LEN + 4,
            DSQDATA_SIDECAR_HEADER_LEN + 8,
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("sequence sidecar has trailing packets"));
    }

    #[test]
    fn rejects_zero_sequence_dsqdata_with_payload_bytes() {
        let err = validate_dsqdata_index_extents(
            &[],
            DSQDATA_SIDECAR_HEADER_LEN + 1,
            DSQDATA_SIDECAR_HEADER_LEN,
        )
        .unwrap_err();
        assert!(err.to_string().contains("zero sequences"));
    }

    #[test]
    fn rejects_dsqdata_metadata_padding_after_taxid() {
        let abc = Alphabet::amino();
        let dsq = abc.digitize(b"");
        let packed = pack5(&dsq, 0);
        let metadata = b"empty\0\0\0\xff\xff\xff\xffx".to_vec();
        let metadata_end = metadata.len() as i64 - 1;
        let chunk = RawChunk {
            packed,
            metadata,
            records: vec![IndexRecord {
                metadata_end,
                psq_end: 0,
            }],
        };
        let mut out = Vec::new();
        let err = unpack_chunk(&chunk, true, &mut out).unwrap_err();
        assert!(err
            .to_string()
            .contains("metadata record must end with exactly one taxid"));
    }

    #[test]
    fn pack5_roundtrip_lengths() {
        let abc = Alphabet::amino();
        for len in [0usize, 1, 5, 6, 7, 12, 13, 100] {
            let raw: Vec<u8> = (0..len).map(|i| b"ACDEFGHIKLMNPQRSTVWY"[i % 20]).collect();
            let dsq = abc.digitize(&raw);
            let psq = pack5(&dsq, len);
            let mut out = Vec::new();
            let (l, consumed) = unpack5(&psq, 0, &mut out).unwrap();
            assert_eq!(l, len, "len {}", len);
            assert_eq!(consumed, psq.len());
            assert_eq!(out, &dsq[1..=len]);
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let base = tmp("badmagic");
        // Minimal stub + index with wrong magic.
        std::fs::write(&base, "Easel dsqdata v1 x12345\n").unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&0xdeadbeefu32.to_le_bytes()); // bad magic
        buf.extend_from_slice(&12345u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4 * 5 + 8 * 3]);
        std::fs::write(with_ext(&base, "dsqi"), buf).unwrap();
        std::fs::write(with_ext(&base, "dsqm"), Vec::<u8>::new()).unwrap();
        std::fs::write(with_ext(&base, "dsqs"), Vec::<u8>::new()).unwrap();
        let err = read_dsqdata(&base).unwrap_err();
        assert!(err.to_string().contains("bad magic"));
        cleanup(&base);
    }

    #[test]
    fn reads_only_first_dsqdata_stub_line() {
        let base = tmp("stub_first_line_only");
        cleanup(&base);
        let mut stub = b"Easel dsqdata v1 x12345\n".to_vec();
        stub.extend(std::iter::repeat_n(b'x', MAX_DSQDATA_STUB_LINE_LEN + 1));
        std::fs::write(&base, stub).unwrap();

        let err = read_dsqdata(&base).unwrap_err();
        assert!(!err.to_string().contains("maximum supported length"));
        cleanup(&base);
    }

    #[test]
    fn rejects_dsqdata_nseq_that_exceeds_index_records_before_allocation() {
        let base = tmp("bad_nseq");
        cleanup(&base);
        std::fs::write(&base, "Easel dsqdata v1 x12345\n").unwrap();
        let mut idx = Vec::new();
        idx.extend_from_slice(&MAGIC_V1.to_le_bytes());
        idx.extend_from_slice(&12345u32.to_le_bytes());
        idx.extend_from_slice(&(AlphabetType::Amino as u32).to_le_bytes());
        idx.extend_from_slice(&0u32.to_le_bytes());
        idx.extend_from_slice(&0u32.to_le_bytes());
        idx.extend_from_slice(&0u32.to_le_bytes());
        idx.extend_from_slice(&0u32.to_le_bytes());
        idx.extend_from_slice(&0u64.to_le_bytes());
        idx.extend_from_slice(&u64::MAX.to_le_bytes());
        idx.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(with_ext(&base, "dsqi"), idx).unwrap();

        let err = read_dsqdata(&base).unwrap_err();
        assert!(err.to_string().contains("only 0 index records fit"));
        cleanup(&base);
    }

    #[test]
    fn rejects_dsqdata_nonmonotonic_index_before_loader_allocation() {
        let abc = Alphabet::amino();
        let seqs = vec![
            Sequence {
                name: "seq1".into(),
                acc: String::new(),
                desc: String::new(),
                dsq: abc.digitize(b"ACD"),
                n: 3,
                l: 3,
                taxid: -1,
            },
            Sequence {
                name: "seq2".into(),
                acc: String::new(),
                desc: String::new(),
                dsq: abc.digitize(b"EFG"),
                n: 3,
                l: 3,
                taxid: -1,
            },
        ];

        let base = tmp("bad_index_order");
        write_dsqdata(&base, &seqs, &abc).unwrap();
        let mut idx = std::fs::read(with_ext(&base, "dsqi")).unwrap();
        write_le_i64_at(&mut idx, DSQDATA_INDEX_HEADER_LEN as usize + 16, 0);
        std::fs::write(with_ext(&base, "dsqi"), idx).unwrap();

        let err = read_dsqdata(&base).unwrap_err();
        assert!(err.to_string().contains("not monotonic"));
        cleanup(&base);
    }

    #[test]
    fn rejects_dsqdata_packet_extent_past_sequence_file() {
        let abc = Alphabet::amino();
        let seqs = vec![Sequence {
            name: "seq1".into(),
            acc: String::new(),
            desc: String::new(),
            dsq: abc.digitize(b"ACD"),
            n: 3,
            l: 3,
            taxid: -1,
        }];

        let base = tmp("bad_psq_extent");
        write_dsqdata(&base, &seqs, &abc).unwrap();
        let mut idx = std::fs::read(with_ext(&base, "dsqi")).unwrap();
        write_le_i64_at(&mut idx, DSQDATA_INDEX_HEADER_LEN as usize + 8, 999);
        std::fs::write(with_ext(&base, "dsqi"), idx).unwrap();

        let err = read_dsqdata(&base).unwrap_err();
        assert!(err.to_string().contains("packet extent exceeds"));
        cleanup(&base);
    }

    /// Full taxid storage round-trip: write sequences carrying distinct
    /// (non-(-1) and -1) taxids, read them back, and confirm each taxid is
    /// preserved along with the surrounding name/acc/desc/sequence metadata.
    /// Exercises both `write_dsqdata` (emits `sq.taxid`) and `unpack_chunk`
    /// (stores the parsed int32 into `Sequence::taxid`).
    #[test]
    fn taxid_metadata_roundtrip() {
        let abc = Alphabet::amino();
        let seqs = vec![
            Sequence {
                name: "taxseq1".into(),
                acc: "AC001".into(),
                desc: "organism A".into(),
                dsq: abc.digitize(b"MKVLWA"),
                n: 6,
                l: 6,
                taxid: 9606,
            },
            Sequence {
                name: "taxseq2".into(),
                acc: String::new(),
                desc: "organism B".into(),
                dsq: abc.digitize(b"ACDEF"),
                n: 5,
                l: 5,
                taxid: -1,
            },
        ];

        let base = tmp("taxid");
        write_dsqdata(&base, &seqs, &abc).unwrap();
        let got = read_dsqdata(&base).unwrap();

        // Metadata (incl. taxid) must survive the write→read cycle intact.
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name, "taxseq1");
        assert_eq!(got[0].acc, "AC001");
        assert_eq!(got[0].desc, "organism A");
        assert_eq!(got[0].n, 6);
        assert_eq!(got[0].dsq, seqs[0].dsq);
        assert_eq!(got[0].taxid, 9606);
        assert_eq!(got[1].name, "taxseq2");
        assert_eq!(got[1].acc, "");
        assert_eq!(got[1].desc, "organism B");
        assert_eq!(got[1].n, 5);
        assert_eq!(got[1].dsq, seqs[1].dsq);
        assert_eq!(got[1].taxid, -1);

        cleanup(&base);
    }

    /// Stress the threaded loader by writing and reading more sequences than
    /// CHUNK_MAXSEQ in a single database. This exercises the chunked pipeline
    /// that spans multiple loader→channel→unpack iterations.
    #[test]
    fn threaded_loader_multi_chunk() {
        let abc = Alphabet::amino();
        // CHUNK_MAXSEQ is 4096; write 2×CHUNK_MAXSEQ+7 sequences to force
        // three complete loader chunks.
        let n_seqs = CHUNK_MAXSEQ * 2 + 7;
        let amino_chars = b"ACDEFGHIKLMNPQRSTVWY";
        let seqs: Vec<Sequence> = (0..n_seqs)
            .map(|i| {
                let raw: Vec<u8> = (0..((i % 20) + 1))
                    .map(|j| amino_chars[(i + j) % 20])
                    .collect();
                let n = raw.len();
                Sequence {
                    name: format!("seq{}", i),
                    acc: String::new(),
                    desc: String::new(),
                    dsq: abc.digitize(&raw),
                    n,
                    l: n,
                    taxid: -1,
                }
            })
            .collect();

        let base = tmp("multichunk");
        write_dsqdata(&base, &seqs, &abc).unwrap();
        let got = read_dsqdata(&base).unwrap();
        assert_eq!(got.len(), n_seqs);
        for (i, (expected, actual)) in seqs.iter().zip(got.iter()).enumerate() {
            assert_eq!(actual.name, expected.name, "name mismatch at seq {}", i);
            assert_eq!(actual.n, expected.n, "length mismatch at seq {}", i);
            assert_eq!(actual.dsq, expected.dsq, "dsq mismatch at seq {}", i);
        }
        cleanup(&base);
    }
}
