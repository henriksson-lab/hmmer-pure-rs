//! Binary HMM file I/O — reading C HMMER's .h3m format.
//! Enables interoperability with C hmmpress output.
#![allow(clippy::while_let_loop)]

use std::io::{BufReader, ErrorKind, Read};
use std::path::Path;

use crate::alphabet::AlphabetType;
use crate::errors::{HmmerError, HmmerResult};
use crate::hmm::*;
use crate::hmmfile::HmmAsciiFormat;

// Magic numbers for HMMER3 binary format versions
const MAGIC_3A: u32 = 0xe8ededb6;
const MAGIC_3B: u32 = 0xe8ededb7;
const MAGIC_3C: u32 = 0xe8ededb8;
const MAGIC_3D: u32 = 0xe8ededb9;
const MAGIC_3E: u32 = 0xe8ededb0;
const MAGIC_3F: u32 = 0xe8ededba;
const MAX_BINARY_HMM_MODEL_LENGTH: i32 = 1_000_000;
const MAX_BINARY_HMM_STRING_LEN: usize = 1 << 20;

/// Return true if `magic` is one of the supported native-endian HMMER3 binary
/// HMM magic numbers.
pub fn is_binary_hmm_magic(magic: u32) -> bool {
    matches!(
        magic,
        MAGIC_3A | MAGIC_3B | MAGIC_3C | MAGIC_3D | MAGIC_3E | MAGIC_3F
    )
}

/// Return true if a path starts with a supported HMMER3 binary HMM magic.
pub fn looks_like_binary_hmm_file(path: &Path) -> HmmerResult<bool> {
    let mut file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut magic_buf = [0u8; 4];
    match file.read_exact(&mut magic_buf) {
        Ok(()) => {
            let magic = u32::from_ne_bytes(magic_buf);
            Ok(is_binary_hmm_magic(magic))
        }
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(HmmerError::Io(e)),
    }
}

/// Read a little/native-endian `u32` from `r` (matches C's raw `fread` of `uint32_t`).
fn read_u32<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u32::from_ne_bytes(buf))
}

/// Read a native-endian `i32` from `r`.
fn read_i32<R: Read>(r: &mut R) -> HmmerResult<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(i32::from_ne_bytes(buf))
}

/// Read a native-endian `f32` from `r`.
fn read_f32<R: Read>(r: &mut R) -> HmmerResult<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(f32::from_ne_bytes(buf))
}

/// Read a length-prefixed C-string: `i32` length including trailing NUL, then bytes.
/// Matches C `read_bin_string`, which treats any `len <= 0` as a NULL string
/// with no error (the writer's NULL convention emits `len = 0`).
fn read_string<R: Read>(r: &mut R) -> HmmerResult<String> {
    let len = read_i32(r)?;
    if len <= 0 {
        return Ok(String::new()); // len <= 0 = NULL/empty (C: s = NULL)
    }
    let len = len as usize;
    if len > MAX_BINARY_HMM_STRING_LEN {
        return Err(HmmerError::Format(format!(
            "Binary HMM string length is too large: {len}"
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    if buf.last() != Some(&0) {
        return Err(HmmerError::Format(
            "Binary HMM string is missing trailing NUL terminator".to_string(),
        ));
    }
    if buf[..len - 1].contains(&0) {
        return Err(HmmerError::Format(
            "Binary HMM string contains embedded NUL byte".to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&buf[..len - 1]).to_string())
}

/// Like [`read_string`] but returns `None` for empty/absent strings.
fn read_string_optional<R: Read>(r: &mut R) -> HmmerResult<Option<String>> {
    let s = read_string(r)?;
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s))
    }
}

/// Read one binary HMMER3 HMM record from `r` (port of `read_bin30hmm`).
///
/// Dispatches on the leading magic to pick the file format (3/a..3/f), then
/// reads flags, M, alphabet code, the match/insert/transition probability
/// blocks, name, optional annotation strings/arrays, comlog, nseq, eff_nseq,
/// maxlen (3/c+), ctime, map (if `P7H_MAP`), checksum, E-value parameters
/// (full array in 3/b+, the legacy three scalars in 3/a), Pfam cutoffs, and
/// optional COMPO composition vector. Returns `Ok(None)` on clean EOF.
pub fn read_binary_hmm<R: Read>(r: &mut R) -> HmmerResult<Option<Hmm>> {
    // Read magic number
    let mut magic_buf = [0u8; 4];
    match r.read(&mut magic_buf[..1]) {
        Ok(0) => return Ok(None),
        Ok(1) => {}
        Ok(_) => unreachable!(),
        Err(e) => return Err(HmmerError::Io(e)),
    }
    if let Err(e) = r.read_exact(&mut magic_buf[1..]) {
        return Err(HmmerError::Io(e));
    }
    let magic = u32::from_ne_bytes(magic_buf);

    let (has_maxl, has_modern_evparams) = match magic {
        MAGIC_3A => (false, false),
        MAGIC_3B => (false, true),
        MAGIC_3C | MAGIC_3D | MAGIC_3E | MAGIC_3F => (true, true),
        _ => {
            return Err(HmmerError::Format(format!(
                "Bad binary HMM magic: {:#x}",
                magic
            )))
        }
    };

    let flags = read_i32(r)? as u32;
    let m_i32 = read_i32(r)?;
    if !(1..=MAX_BINARY_HMM_MODEL_LENGTH).contains(&m_i32) {
        return Err(HmmerError::Format(format!(
            "Invalid binary HMM model length: {m_i32}"
        )));
    }
    let m = m_i32 as usize;
    let abc_type_int = read_i32(r)?;
    let abc_type = match abc_type_int {
        1 => AlphabetType::Rna,
        2 => AlphabetType::Dna,
        3 => AlphabetType::Amino,
        _ => {
            return Err(HmmerError::Format(format!(
                "Unknown binary HMM alphabet code: {abc_type_int}"
            )))
        }
    };

    let k = match abc_type {
        AlphabetType::Amino => 20,
        AlphabetType::Dna | AlphabetType::Rna => 4,
        _ => 20,
    };

    let mut hmm = Hmm::new(m, abc_type, k);
    hmm.flags = flags;

    // Read match emissions: mat[1..M][0..K-1]
    for node in 1..=m {
        for x in 0..k {
            hmm.mat[node][x] = read_f32(r)?;
        }
    }

    // Read insert emissions: ins[0..M][0..K-1]
    for node in 0..=m {
        for x in 0..k {
            hmm.ins[node][x] = read_f32(r)?;
        }
    }

    // Read transitions: t[0..M][0..6]
    for node in 0..=m {
        for t in 0..NTRANSITIONS {
            hmm.t[node][t] = read_f32(r)?;
        }
    }

    // Read name
    hmm.name = read_string(r)?;
    if hmm.name.is_empty() {
        return Err(HmmerError::Format(
            "Binary HMM record has empty required NAME".to_string(),
        ));
    }

    // Optional fields based on flags
    if flags & P7H_ACC != 0 {
        hmm.acc = read_string_optional(r)?;
    }
    if flags & P7H_DESC != 0 {
        hmm.desc = read_string_optional(r)?;
    }
    if flags & P7H_RF != 0 {
        hmm.rf = Some(read_annotation(r, m)?);
    }
    if flags & P7H_MMASK != 0 {
        hmm.mm = Some(read_annotation(r, m)?);
    }
    if flags & P7H_CONS != 0 {
        hmm.consensus = Some(read_annotation(r, m)?);
    }
    if flags & P7H_CS != 0 {
        hmm.cs = Some(read_annotation(r, m)?);
    }
    if flags & P7H_CA != 0 {
        hmm.ca = Some(read_annotation(r, m)?);
    }

    // Command log
    hmm.comlog = read_string_optional(r)?;

    // nseq, eff_nseq
    hmm.nseq = read_i32(r)?;
    hmm.eff_nseq = read_f32(r)?;

    // max_length (format 3c+)
    if has_maxl {
        hmm.max_length = read_i32(r)?;
    }

    // Creation time
    hmm.ctime = read_string_optional(r)?;

    // Map
    if flags & P7H_MAP != 0 {
        let mut map = vec![0i32; m + 1];
        for item in map.iter_mut().take(m + 1) {
            *item = read_i32(r)?;
        }
        hmm.map = Some(map);
    }

    // Checksum
    hmm.checksum = read_u32(r)?;

    // E-value parameters. HMMER 3/b+ stores these unconditionally; the
    // P7H_STATS flag records that they are valid, not whether bytes exist.
    if has_modern_evparams {
        for i in 0..NEVPARAM {
            hmm.evparam[i] = read_f32(r)?;
        }
    } else {
        // 3/a stored only MLAMBDA, MMU, FTAU and C HMMER expands them.
        let lambda = read_f32(r)?;
        let mu = read_f32(r)?;
        let tau = read_f32(r)?;
        hmm.evparam[P7_MLAMBDA] = lambda;
        hmm.evparam[P7_MMU] = mu;
        hmm.evparam[P7_FTAU] = tau;
        hmm.evparam[P7_FLAMBDA] = lambda;
        hmm.evparam[P7_VLAMBDA] = lambda;
        hmm.evparam[P7_VMU] = mu;
    }

    // Pfam cutoffs are present as a full array in the binary stream even if no
    // GA/TC/NC flags are set.
    for i in 0..NCUTOFFS {
        hmm.cutoff[i] = read_f32(r)?;
    }

    // Composition
    if flags & P7H_COMPO != 0 {
        for i in 0..k.min(MAXABET) {
            hmm.compo[i] = read_f32(r)?;
        }
    }

    Ok(Some(hmm))
}

/// Read an `m + 2` byte per-node annotation array (RF/MM/CONS/CS/CA) as raw bytes.
fn read_annotation<R: Read>(r: &mut R, m: usize) -> HmmerResult<Vec<u8>> {
    let mut buf = vec![0u8; m + 2];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(buf)
}

/// Open a `.h3m` binary HMM database and read every contained HMM.
pub fn read_binary_hmm_file(path: &Path) -> HmmerResult<Vec<Hmm>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    let mut hmms = Vec::new();
    let mut expected_abc = None;

    loop {
        match read_binary_hmm(&mut reader)? {
            Some(hmm) => {
                if let Some(expected) = expected_abc {
                    if hmm.abc_type != expected {
                        return Err(HmmerError::Format(format!(
                            "Binary HMM file contains mixed alphabets: first record is {:?}, record {} is {:?}",
                            expected,
                            hmms.len() + 1,
                            hmm.abc_type
                        )));
                    }
                } else {
                    expected_abc = Some(hmm.abc_type);
                }
                hmms.push(hmm);
            }
            None => break,
        }
    }

    Ok(hmms)
}

/// Write a single HMM in HMMER3/f binary format (port of `p7_hmmfile_WriteBinary`).
///
/// Emits magic, flags, M, alphabet code, all match/insert/transition
/// probability blocks, name, optional acc/desc, RF/MM/CONS/CS/CA annotation
/// arrays, comlog, nseq/eff_nseq, max_length, ctime, optional map, checksum,
/// E-value parameter array, Pfam cutoffs, and optional COMPO. Output is
/// byte-compatible with C `hmmpress`/`hmmconvert -b`.
pub fn write_binary_hmm<W: std::io::Write>(w: &mut W, hmm: &Hmm) -> HmmerResult<()> {
    write_binary_hmm_with_format(w, hmm, HmmAsciiFormat::Hmmer3f)
}

/// Write a single HMM in the selected HMMER3 binary format (3/a..3/f).
pub fn write_binary_hmm_with_format<W: std::io::Write>(
    w: &mut W,
    hmm: &Hmm,
    format: HmmAsciiFormat,
) -> HmmerResult<()> {
    let k = hmm.abc_k;
    let mut flags = hmm.flags;
    if hmm.acc.is_some() {
        flags |= P7H_ACC;
    } else {
        flags &= !P7H_ACC;
    }
    if hmm.desc.is_some() {
        flags |= P7H_DESC;
    } else {
        flags &= !P7H_DESC;
    }

    let magic = match format {
        HmmAsciiFormat::Hmmer3a => MAGIC_3A,
        HmmAsciiFormat::Hmmer3b => MAGIC_3B,
        HmmAsciiFormat::Hmmer3c => MAGIC_3C,
        HmmAsciiFormat::Hmmer3d => MAGIC_3D,
        HmmAsciiFormat::Hmmer3e => MAGIC_3E,
        HmmAsciiFormat::Hmmer3f => MAGIC_3F,
    };
    w.write_all(&magic.to_ne_bytes()).map_err(HmmerError::Io)?;

    // Flags
    w.write_all(&(flags as i32).to_ne_bytes())
        .map_err(HmmerError::Io)?;

    // M
    w.write_all(&checked_i32_usize(hmm.m, "Binary HMM model length")?.to_ne_bytes())
        .map_err(HmmerError::Io)?;

    // Alphabet type
    let abc_int: i32 = match hmm.abc_type {
        AlphabetType::Rna => 1,
        AlphabetType::Dna => 2,
        AlphabetType::Amino => 3,
        _ => 3,
    };
    w.write_all(&abc_int.to_ne_bytes())
        .map_err(HmmerError::Io)?;

    // Match emissions
    for node in 1..=hmm.m {
        for x in 0..k {
            w.write_all(&hmm.mat[node][x].to_ne_bytes())
                .map_err(HmmerError::Io)?;
        }
    }

    // Insert emissions
    for node in 0..=hmm.m {
        for x in 0..k {
            w.write_all(&hmm.ins[node][x].to_ne_bytes())
                .map_err(HmmerError::Io)?;
        }
    }

    // Transitions
    for node in 0..=hmm.m {
        for t in 0..NTRANSITIONS {
            w.write_all(&hmm.t[node][t].to_ne_bytes())
                .map_err(HmmerError::Io)?;
        }
    }

    // Name
    write_string(w, &hmm.name)?;

    // Optional fields
    if flags & P7H_ACC != 0 {
        write_optional_string(w, hmm.acc.as_deref())?;
    }
    if flags & P7H_DESC != 0 {
        write_optional_string(w, hmm.desc.as_deref())?;
    }

    if flags & P7H_RF != 0 {
        write_annotation(w, &hmm.rf, hmm.m)?;
    }
    if flags & P7H_MMASK != 0 {
        write_annotation(w, &hmm.mm, hmm.m)?;
    }
    if flags & P7H_CONS != 0 {
        write_annotation(w, &hmm.consensus, hmm.m)?;
    }
    if flags & P7H_CS != 0 {
        write_annotation(w, &hmm.cs, hmm.m)?;
    }
    if flags & P7H_CA != 0 {
        write_annotation(w, &hmm.ca, hmm.m)?;
    }

    // Command log
    write_optional_string(w, hmm.comlog.as_deref())?;

    // nseq, eff_nseq
    w.write_all(&hmm.nseq.to_ne_bytes())
        .map_err(HmmerError::Io)?;
    w.write_all(&hmm.eff_nseq.to_ne_bytes())
        .map_err(HmmerError::Io)?;

    if format >= HmmAsciiFormat::Hmmer3c {
        w.write_all(&hmm.max_length.to_ne_bytes())
            .map_err(HmmerError::Io)?;
    }

    // Creation time
    write_optional_string(w, hmm.ctime.as_deref())?;

    // Map
    if flags & P7H_MAP != 0 {
        if let Some(ref map) = hmm.map {
            for node in 0..=hmm.m {
                let value = map.get(node).copied().unwrap_or(0);
                w.write_all(&value.to_ne_bytes()).map_err(HmmerError::Io)?;
            }
        } else {
            for _ in 0..=hmm.m {
                w.write_all(&0i32.to_ne_bytes()).map_err(HmmerError::Io)?;
            }
        }
    }

    // Checksum
    w.write_all(&hmm.checksum.to_ne_bytes())
        .map_err(HmmerError::Io)?;

    if format == HmmAsciiFormat::Hmmer3a {
        for i in [P7_MLAMBDA, P7_MMU, P7_FTAU] {
            w.write_all(&hmm.evparam[i].to_ne_bytes())
                .map_err(HmmerError::Io)?;
        }
    } else {
        for i in 0..NEVPARAM {
            w.write_all(&hmm.evparam[i].to_ne_bytes())
                .map_err(HmmerError::Io)?;
        }
    }

    // Cutoffs
    for i in 0..NCUTOFFS {
        w.write_all(&hmm.cutoff[i].to_ne_bytes())
            .map_err(HmmerError::Io)?;
    }

    // Composition
    if flags & P7H_COMPO != 0 {
        for i in 0..k.min(MAXABET) {
            w.write_all(&hmm.compo[i].to_ne_bytes())
                .map_err(HmmerError::Io)?;
        }
    }

    Ok(())
}

/// Write an `m + 2` byte per-node annotation array, padding with spaces and a
/// trailing NUL if `data` is `None` or shorter than the expected length.
fn write_annotation<W: std::io::Write>(
    w: &mut W,
    data: &Option<Vec<u8>>,
    m: usize,
) -> HmmerResult<()> {
    let len = m + 2;
    if let Some(d) = data {
        // C keeps these arrays as: index 0 = ' ', 1..M = data, M+1 = '\0'
        // (see p7_hmmfile.c read_asc30hmm / read_bin30hmm, ~lines 1560-1562),
        // and p7_hmmfile_WriteBinary dumps the full m+2 bytes verbatim. Our
        // reader leaves index M+1 as a space, so force the trailing byte to a
        // NUL terminator here to match C's binary output byte-for-byte.
        let mut buf = vec![0u8; len];
        let n = d.len().min(len);
        buf[..n].copy_from_slice(&d[..n]);
        buf[len - 1] = 0;
        w.write_all(&buf).map_err(HmmerError::Io)?;
    } else {
        let mut empty = vec![b' '; len];
        empty[len - 1] = 0;
        w.write_all(&empty).map_err(HmmerError::Io)?;
    }
    Ok(())
}

/// Write a non-NULL length-prefixed C-string: length (including NUL), bytes,
/// terminating NUL. Mirrors C `write_bin_string` for a non-NULL `s`.
fn write_string<W: std::io::Write>(w: &mut W, s: &str) -> HmmerResult<()> {
    let bytes = s.as_bytes();
    if bytes.contains(&0) {
        return Err(HmmerError::Format(
            "Binary HMM string contains embedded NUL byte".to_string(),
        ));
    }
    let stored_len = bytes
        .len()
        .checked_add(1)
        .ok_or_else(|| HmmerError::Format("Binary HMM string length overflows usize".into()))?;
    let len = checked_i32_usize(stored_len, "Binary HMM string length")?;
    w.write_all(&len.to_ne_bytes()).map_err(HmmerError::Io)?;
    w.write_all(bytes).map_err(HmmerError::Io)?;
    w.write_all(&[0u8]).map_err(HmmerError::Io)?; // null terminator
    Ok(())
}

fn checked_i32_usize(value: usize, field: &str) -> HmmerResult<i32> {
    i32::try_from(value)
        .map_err(|_| HmmerError::Format(format!("{field} exceeds i32 range: {value}")))
}

/// Write an optional length-prefixed C-string, matching C `write_bin_string`:
/// for `None` (a NULL string) write just `int32 = 0` and no body; for `Some(s)`
/// write `strlen(s)+1`, the bytes, and a terminating NUL.
fn write_optional_string<W: std::io::Write>(w: &mut W, s: Option<&str>) -> HmmerResult<()> {
    match s {
        None => {
            w.write_all(&0i32.to_ne_bytes()).map_err(HmmerError::Io)?;
            Ok(())
        }
        Some(s) => write_string(w, s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::process::Command;

    fn test_path(relative: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
    }

    /// Test helper: run the C `hmmconvert -b` binary on an ASCII HMM and
    /// stage the resulting `.h3m` in `dir` for the Rust reader to consume.
    fn c_binary_hmm_from_text(text_hmm: &Path, dir: &tempfile::TempDir) -> std::path::PathBuf {
        let output = Command::new(test_path("hmmer/src/hmmconvert"))
            .arg("-b")
            .arg(text_hmm)
            .output()
            .expect("failed to run C hmmconvert");
        assert!(
            output.status.success(),
            "C hmmconvert failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let path = dir.path().join("converted.h3m");
        std::fs::write(&path, output.stdout).unwrap();
        path
    }

    #[test]
    fn rejects_unknown_binary_alphabet_code() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC_3F.to_ne_bytes());
        buf.extend_from_slice(&0i32.to_ne_bytes()); // flags
        buf.extend_from_slice(&1i32.to_ne_bytes()); // M
        buf.extend_from_slice(&99i32.to_ne_bytes()); // invalid alphabet code

        let err = read_binary_hmm(&mut Cursor::new(buf)).unwrap_err();
        assert!(
            matches!(err, HmmerError::Format(msg) if msg.contains("Unknown binary HMM alphabet code"))
        );
    }

    #[test]
    fn rejects_nonpositive_binary_model_length_before_allocating() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC_3F.to_ne_bytes());
        buf.extend_from_slice(&0i32.to_ne_bytes()); // flags
        buf.extend_from_slice(&(-1i32).to_ne_bytes()); // invalid M
        buf.extend_from_slice(&3i32.to_ne_bytes()); // amino

        let err = read_binary_hmm(&mut Cursor::new(buf)).unwrap_err();
        assert!(
            matches!(err, HmmerError::Format(msg) if msg.contains("Invalid binary HMM model length"))
        );
    }

    #[test]
    fn rejects_excessive_binary_model_length_before_allocating() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC_3F.to_ne_bytes());
        buf.extend_from_slice(&0i32.to_ne_bytes()); // flags
        buf.extend_from_slice(&(MAX_BINARY_HMM_MODEL_LENGTH + 1).to_ne_bytes()); // invalid M
        buf.extend_from_slice(&3i32.to_ne_bytes()); // amino

        let err = read_binary_hmm(&mut Cursor::new(buf)).unwrap_err();
        assert!(
            matches!(err, HmmerError::Format(msg) if msg.contains("Invalid binary HMM model length"))
        );
    }

    #[test]
    fn rejects_invalid_binary_string_length() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC_3F.to_ne_bytes());
        buf.extend_from_slice(&0i32.to_ne_bytes()); // flags
        buf.extend_from_slice(&1i32.to_ne_bytes()); // M
        buf.extend_from_slice(&3i32.to_ne_bytes()); // amino
        for _ in 0..20 {
            buf.extend_from_slice(&0f32.to_ne_bytes()); // mat[1]
        }
        for _ in 0..40 {
            buf.extend_from_slice(&0f32.to_ne_bytes()); // ins[0..1]
        }
        for _ in 0..14 {
            buf.extend_from_slice(&0f32.to_ne_bytes()); // t[0..1]
        }
        buf.extend_from_slice(&(-2i32).to_ne_bytes()); // non-positive len: C read_bin_string returns NULL

        // C treats any len<=0 as a NULL string (no error at the string level), then
        // rejects the record because the required NAME is absent (p7_hmmfile.c:1565,1654).
        let err = read_binary_hmm(&mut Cursor::new(buf)).unwrap_err();
        assert!(matches!(err, HmmerError::Format(msg) if msg.contains("empty required NAME")));
    }

    #[test]
    fn rejects_binary_string_without_trailing_nul() {
        let mut buf = Cursor::new([3i32.to_ne_bytes().as_slice(), b"abc"].concat());

        let err = read_string(&mut buf).unwrap_err();
        assert!(err.to_string().contains("missing trailing NUL"));
    }

    #[test]
    fn rejects_binary_string_with_embedded_nul() {
        let mut buf = Cursor::new([4i32.to_ne_bytes().as_slice(), b"a\0b\0"].concat());

        let err = read_string(&mut buf).unwrap_err();
        assert!(err.to_string().contains("embedded NUL"));
    }

    #[test]
    fn writer_rejects_binary_string_with_embedded_nul() {
        let mut hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .remove(0);
        hmm.name = "bad\0name".to_string();

        let err = write_binary_hmm(&mut Vec::new(), &hmm).unwrap_err();
        assert!(err.to_string().contains("embedded NUL"));
    }

    #[test]
    fn rejects_excessive_binary_string_length_before_allocation() {
        let mut buf = Cursor::new(((MAX_BINARY_HMM_STRING_LEN + 1) as i32).to_ne_bytes());

        let err = read_string(&mut buf).unwrap_err();
        assert!(err.to_string().contains("too large"));
    }

    #[test]
    fn rejects_binary_hmm_with_empty_required_name() {
        let mut hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .remove(0);
        hmm.name.clear();
        let mut buf = Vec::new();
        write_binary_hmm(&mut buf, &hmm).unwrap();

        let err = read_binary_hmm(&mut Cursor::new(buf)).unwrap_err();
        assert!(err.to_string().contains("empty required NAME"));
    }

    #[test]
    fn test_roundtrip_binary() {
        // Read an HMM from ASCII, write to binary, read back
        let hmms = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap();
        let hmm = &hmms[0];

        // Write to binary
        let mut buf = Vec::new();
        write_binary_hmm(&mut buf, hmm).unwrap();

        // Read back
        let mut cursor = Cursor::new(&buf);
        let hmm2 = read_binary_hmm(&mut cursor).unwrap().unwrap();

        assert_eq!(hmm.name, hmm2.name);
        assert_eq!(hmm.m, hmm2.m);
        assert_eq!(hmm.abc_type, hmm2.abc_type);

        // Compare match emissions
        for node in 1..=hmm.m {
            for x in 0..hmm.abc_k {
                assert!(
                    (hmm.mat[node][x] - hmm2.mat[node][x]).abs() < 1e-6,
                    "mat[{}][{}] mismatch",
                    node,
                    x
                );
            }
        }
    }

    /// Regression: the Rust binary writer must be byte-for-byte identical to
    /// C `hmmconvert -b`. A prior bug wrote a trailing space (0x20) instead of
    /// the NUL terminator at index M+1 of the per-node annotation arrays
    /// (rf/mm/cons/cs/ca), diverging by one byte just before the
    /// comlog/nseq/ctime region. C keeps array[M+1] == '\0' (p7_hmmfile.c:1560-1562)
    /// and dumps the full M+2 bytes verbatim (p7_hmmfile.c:1037-1041).
    #[test]
    fn binary_write_matches_c_hmmconvert_byte_for_byte() {
        // Exercise models with CONS/CS/RF/MAP/COMPO and ctime/comlog/nseq fields.
        for rel in [
            "test_data/Pkinase_pfam.hmm", // CONS annotation, exposed the trailing-NUL bug
            "hmmer/tutorial/fn3.hmm",     // CS + MAP + DATE + NSEQ + EFFN
            "hmmer/tutorial/MADE1.hmm",   // DNA, RF + CONS + max_length (3/c+)
        ] {
            let text_path = test_path(rel);
            if !text_path.exists() {
                continue;
            }
            let dir = tempfile::tempdir().unwrap();
            let c_h3m = std::fs::read(c_binary_hmm_from_text(&text_path, &dir)).unwrap();

            let hmms = crate::hmmfile::read_hmm_file(&text_path).unwrap();
            let mut rust_h3m = Vec::new();
            for hmm in &hmms {
                write_binary_hmm(&mut rust_h3m, hmm).unwrap();
            }

            assert_eq!(
                rust_h3m.len(),
                c_h3m.len(),
                "{rel}: length differs (rust {} vs C {})",
                rust_h3m.len(),
                c_h3m.len()
            );
            if let Some(off) = rust_h3m.iter().zip(&c_h3m).position(|(a, b)| a != b) {
                panic!(
                    "{rel}: first byte divergence at offset {off}: rust=0x{:02x} c=0x{:02x}",
                    rust_h3m[off], c_h3m[off]
                );
            }
        }
    }

    #[test]
    fn read_binary_hmm_file_rejects_mixed_alphabet_records() {
        let amino = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .remove(0);
        let dna = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_data/mapali/ecori-rebuilt.hmm"
        )))
        .unwrap()
        .remove(0);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.h3m");
        let mut file = std::fs::File::create(&path).unwrap();
        write_binary_hmm(&mut file, &amino).unwrap();
        write_binary_hmm(&mut file, &dna).unwrap();
        drop(file);

        let err = read_binary_hmm_file(&path).unwrap_err();

        assert!(err.to_string().contains("mixed alphabets"));
    }

    #[test]
    fn binary_writer_normalizes_acc_and_desc_flags_from_fields() {
        let mut hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .remove(0);
        hmm.acc = Some("ACC123".to_string());
        hmm.desc = Some("description".to_string());
        hmm.flags &= !(P7H_ACC | P7H_DESC);

        let mut buf = Vec::new();
        write_binary_hmm(&mut buf, &hmm).unwrap();
        let read = read_binary_hmm(&mut Cursor::new(buf)).unwrap().unwrap();
        assert_eq!(read.acc.as_deref(), Some("ACC123"));
        assert_eq!(read.desc.as_deref(), Some("description"));

        let mut no_meta = hmm;
        no_meta.acc = None;
        no_meta.desc = None;
        no_meta.flags |= P7H_ACC | P7H_DESC;
        let mut buf = Vec::new();
        write_binary_hmm(&mut buf, &no_meta).unwrap();
        let read = read_binary_hmm(&mut Cursor::new(buf)).unwrap().unwrap();
        assert_eq!(read.acc, None);
        assert_eq!(read.desc, None);
    }

    #[test]
    fn reads_c_hmmer_binary_with_raw_annotation_layout() {
        let text_path = test_path("test_data/gecco_cluster1_hmms.hmm");
        let expected = crate::hmmfile::read_hmm_file(&text_path).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let h3m_path = c_binary_hmm_from_text(&text_path, &dir);
        let actual = read_binary_hmm_file(&h3m_path).unwrap();

        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert_eq!(expected.name, actual.name);
            assert_eq!(expected.acc, actual.acc);
            assert_eq!(expected.m, actual.m);
            assert_eq!(expected.abc_type, actual.abc_type);
            assert_eq!(
                expected.consensus.as_ref().map(Vec::len),
                actual.consensus.as_ref().map(Vec::len)
            );
            assert_eq!(
                expected.rf.as_ref().map(Vec::len),
                actual.rf.as_ref().map(Vec::len)
            );
            assert_eq!(
                expected.map.as_ref().map(Vec::len),
                actual.map.as_ref().map(Vec::len)
            );

            for node in 1..=expected.m {
                for x in 0..expected.abc_k {
                    assert!(
                        (expected.mat[node][x] - actual.mat[node][x]).abs() < 1e-6,
                        "{} mat[{}][{}] mismatch",
                        expected.name,
                        node,
                        x
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires GECCO's full Pfam.h3m fixture"]
    fn reads_gecco_full_pfam_h3m() {
        let path = Path::new("/data/henriksson/github/claude/gecco-rs/data/Pfam.h3m");
        let hmms = read_binary_hmm_file(path).unwrap();
        assert_eq!(hmms.len(), 2766);
        assert!(hmms.iter().all(|hmm| !hmm.name.is_empty()));
        assert!(hmms.iter().all(|hmm| hmm.m > 0));
    }
}
