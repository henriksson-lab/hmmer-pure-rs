//! Easel dsqdata format: a binary sequence database for fast reading.
//!
//! This is a faithful port of the **on-disk byte format** of Easel's
//! `esl_dsqdata.c` (HMMER/Easel 3.4). Files written here are byte-compatible
//! with Easel's `esl_dsqdata_Write()`, and files written by Easel can be read
//! back by [`read_dsqdata`]. See `hmmer/easel/esl_dsqdata.c` for the
//! source-of-truth.
//!
//! ## On-disk layout (four files)
//!
//! A dsqdata "database" is a stub file `<basename>` plus three binary files:
//!   - `<basename>.dsqi` — index file
//!   - `<basename>.dsqm` — metadata file
//!   - `<basename>.dsqs` — packed sequence file
//!
//! The four files are linked by a random 32-bit `uniquetag`.
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
//! ## Simplification vs. C
//! Easel's reader uses a threaded producer/consumer pipeline (a loader thread
//! and several unpacker threads) purely for throughput. This port implements a
//! simple **synchronous** reader/writer. The bytes read and written are
//! format-identical to C; only the concurrency optimization is omitted.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::alphabet::{Alphabet, AlphabetType, Dsq, DSQ_SENTINEL};
use crate::errors::{HmmerError, HmmerResult};
use crate::sequence::Sequence;

/// "dsq1" + 0x80808080. Detects format and byte order (port of `eslDSQDATA_MAGIC_V1`).
const MAGIC_V1: u32 = 0xc4d3d1b1;
/// Byte-swapped magic (port of `eslDSQDATA_MAGIC_V1SWAP`); read support unimplemented (matches C).
const MAGIC_V1SWAP: u32 = 0xb1d1d3c4;

/// Control bit: last packet in a packed sequence (port of `eslDSQDATA_EOD`).
const EOD_BIT: u32 = 1 << 31;
/// Control bit: packet is 5-bit packed (port of `eslDSQDATA_5BIT`).
const FIVEBIT_BIT: u32 = 1 << 30;

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
/// Taxids are written as -1 (none), since [`Sequence`] carries no taxid field.
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
        max_namelen = max_namelen.max(sq.name.as_bytes().len() as u32);
        max_acclen = max_acclen.max(sq.acc.as_bytes().len() as u32);
        max_desclen = max_desclen.max(sq.desc.as_bytes().len() as u32);
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

        // Metadata: name\0 acc\0 desc\0 taxid(int32 = -1)
        mpos += write_cstr(&mut mfp, sq.name.as_bytes())? as i64;
        mpos += write_cstr(&mut mfp, sq.acc.as_bytes())? as i64;
        mpos += write_cstr(&mut mfp, sq.desc.as_bytes())? as i64;
        write_i32(&mut mfp, -1)?;
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
// Read (synchronous port of esl_dsqdata_Open + loader + unpacker)
// ---------------------------------------------------------------------------

/// Open and read an entire dsqdata database from `<basename>` and its three
/// `.dsq?` companion files.
///
/// This is a synchronous replacement for Easel's threaded reader (see module
/// docs). It validates the magic/uniquetag linkage exactly as C does, then
/// unpacks every sequence. The alphabet is determined from the index header's
/// `alphatype` (as C does when the caller passes a NULL alphabet).
///
/// Returns the sequences with their leading/trailing [`DSQ_SENTINEL`] bytes
/// restored. Accession/description are populated from the metadata file.
pub fn read_dsqdata(basename: &Path) -> HmmerResult<Vec<Sequence>> {
    let p = basename;

    // ---- Stub file: parse "Easel dsqdata v1 x<tag>" first line. ----
    let stub = std::fs::read_to_string(p).map_err(HmmerError::Io)?;
    let first = stub
        .lines()
        .next()
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
    let mut ifp = BufReader::new(open(&with_ext(p, "dsqi"))?);
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
        return Err(HmmerError::Format("dsqdata index file has bad magic".into()));
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

    // ---- Read all index records. ----
    let mut records = Vec::with_capacity(nseq as usize);
    for _ in 0..nseq {
        let metadata_end = read_i64(&mut ifp)?;
        let psq_end = read_i64(&mut ifp)?;
        records.push(IndexRecord {
            metadata_end,
            psq_end,
        });
    }

    // ---- Metadata file: verify header, slurp body. ----
    let mut mfp = BufReader::new(open(&with_ext(p, "dsqm"))?);
    let m_magic = read_u32(&mut mfp)?;
    let m_tag = read_u32(&mut mfp)?;
    if m_magic != magic {
        return Err(HmmerError::Format("dsqdata metadata file has bad magic".into()));
    }
    if m_tag != stub_tag {
        return Err(HmmerError::Format(
            "dsqdata metadata file has bad tag, doesn't match stub".into(),
        ));
    }
    let mut metadata = Vec::new();
    mfp.read_to_end(&mut metadata).map_err(HmmerError::Io)?;

    // ---- Sequence file: verify header, slurp packed body as u32 packets. ----
    let mut sfp = BufReader::new(open(&with_ext(p, "dsqs"))?);
    let s_magic = read_u32(&mut sfp)?;
    let s_tag = read_u32(&mut sfp)?;
    if s_magic != magic {
        return Err(HmmerError::Format("dsqdata sequence file has bad magic".into()));
    }
    if s_tag != stub_tag {
        return Err(HmmerError::Format(
            "dsqdata sequence file has bad tag, doesn't match stub".into(),
        ));
    }
    let mut sbytes = Vec::new();
    sfp.read_to_end(&mut sbytes).map_err(HmmerError::Io)?;
    if sbytes.len() % 4 != 0 {
        return Err(HmmerError::Format(
            "dsqdata sequence file body is not a whole number of packets".into(),
        ));
    }
    let psq: Vec<u32> = sbytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // ---- Walk records, unpacking metadata + sequence per record. ----
    let mut sequences = Vec::with_capacity(nseq as usize);
    let mut psq_last: i64 = -1; // last packet offset consumed (inclusive)
    let mut meta_last: i64 = -1; // last metadata byte offset consumed (inclusive)
    for rec in &records {
        // Packet range for this sequence: (psq_last+1 ..= rec.psq_end).
        let pstart = (psq_last + 1) as usize;
        let pn = (rec.psq_end - psq_last) as usize;
        if pn == 0 {
            return Err(HmmerError::Format(
                "dsqdata: index record encodes zero packets (every seq needs an EOD packet)".into(),
            ));
        }
        let mut residues = Vec::new();
        let (l, consumed) = if do_pack5 {
            unpack5(&psq, pstart, &mut residues)?
        } else {
            unpack2(&psq, pstart, &mut residues)?
        };
        if consumed != pn {
            return Err(HmmerError::Format(format!(
                "dsqdata: packet count mismatch (index says {}, unpacked {})",
                pn, consumed
            )));
        }
        psq_last = rec.psq_end;

        // Metadata range for this sequence: bytes (meta_last+1 ..= rec.metadata_end).
        let mstart = (meta_last + 1) as usize;
        let mend = (rec.metadata_end + 1) as usize; // exclusive
        if mend > metadata.len() || mstart > mend {
            return Err(HmmerError::Format("dsqdata: metadata offset out of range".into()));
        }
        let block = &metadata[mstart..mend];
        // name\0 acc\0 desc\0 taxid(int32)
        let (name, rest) = take_cstr(block)?;
        let (acc, rest) = take_cstr(rest)?;
        let (desc, rest) = take_cstr(rest)?;
        if rest.len() < 4 {
            return Err(HmmerError::Format("dsqdata: metadata record truncated (taxid)".into()));
        }
        // taxid is read but not stored (Sequence has no taxid field).
        let _taxid = i32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
        meta_last = rec.metadata_end;

        let mut dsq = Vec::with_capacity(l + 2);
        dsq.push(DSQ_SENTINEL);
        dsq.extend_from_slice(&residues);
        dsq.push(DSQ_SENTINEL);

        sequences.push(Sequence {
            name: String::from_utf8_lossy(name).into_owned(),
            acc: String::from_utf8_lossy(acc).into_owned(),
            desc: String::from_utf8_lossy(desc).into_owned(),
            dsq,
            n: l,
            l,
        });
    }

    Ok(sequences)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_dsq(sq: &Sequence) -> HmmerResult<()> {
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

/// Derive a deterministic uniquetag. C uses a random uint32, but any value is
/// valid provided the four files agree. We mix the dataset's basic stats with a
/// fixed nonzero seed so the tag is nonzero and stable for a given dataset.
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
            },
            Sequence {
                name: "seq2".into(),
                acc: String::new(),
                desc: String::new(),
                dsq: abc.digitize(b"MKV"),
                n: 3,
                l: 3,
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
}
