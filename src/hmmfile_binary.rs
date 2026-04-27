//! Binary HMM file I/O — reading C HMMER's .h3m format.
//! Enables interoperability with C hmmpress output.

use std::io::{BufReader, Read};
use std::path::Path;

use crate::alphabet::AlphabetType;
use crate::errors::{HmmerError, HmmerResult};
use crate::hmm::*;

// Magic numbers for HMMER3 binary format versions
const MAGIC_3A: u32 = 0xe8ededb6;
const MAGIC_3B: u32 = 0xe8ededb7;
const MAGIC_3C: u32 = 0xe8ededb8;
const MAGIC_3D: u32 = 0xe8ededb9;
const MAGIC_3E: u32 = 0xe8ededb0;
const MAGIC_3F: u32 = 0xe8ededba;

fn read_u32<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u32::from_ne_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> HmmerResult<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(i32::from_ne_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> HmmerResult<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(f32::from_ne_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> HmmerResult<String> {
    let len = read_i32(r)?;
    if len <= 0 {
        return Ok(String::new()); // -1 = absent, 0 = empty
    }
    let len = len as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    // Remove null terminator if present
    if buf.last() == Some(&0) {
        buf.pop();
    }
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_string_optional<R: Read>(r: &mut R) -> HmmerResult<Option<String>> {
    let s = read_string(r)?;
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s))
    }
}

/// Read a single HMM from a binary .h3m stream.
pub fn read_binary_hmm<R: Read>(r: &mut R) -> HmmerResult<Option<Hmm>> {
    // Read magic number
    let magic = match read_u32(r) {
        Ok(m) => m,
        Err(_) => return Ok(None), // EOF
    };

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
    let m = read_i32(r)? as usize;
    let abc_type_int = read_i32(r)?;
    let abc_type = match abc_type_int {
        1 => AlphabetType::Rna,
        2 => AlphabetType::Dna,
        3 => AlphabetType::Amino,
        _ => AlphabetType::Amino,
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
        for node in 0..=m {
            map[node] = read_i32(r)?;
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

fn read_annotation<R: Read>(r: &mut R, m: usize) -> HmmerResult<Vec<u8>> {
    let mut buf = vec![0u8; m + 2];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(buf)
}

/// Read all HMMs from a binary .h3m file.
pub fn read_binary_hmm_file(path: &Path) -> HmmerResult<Vec<Hmm>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    let mut hmms = Vec::new();

    loop {
        match read_binary_hmm(&mut reader)? {
            Some(hmm) => hmms.push(hmm),
            None => break,
        }
    }

    Ok(hmms)
}

/// Write a single HMM in C-compatible binary .h3m format.
pub fn write_binary_hmm<W: std::io::Write>(w: &mut W, hmm: &Hmm) -> HmmerResult<()> {
    let k = hmm.abc_k;

    // Magic number (3/f format)
    w.write_all(&MAGIC_3F.to_ne_bytes())
        .map_err(HmmerError::Io)?;

    // Flags
    w.write_all(&(hmm.flags as i32).to_ne_bytes())
        .map_err(HmmerError::Io)?;

    // M
    w.write_all(&(hmm.m as i32).to_ne_bytes())
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
    if hmm.flags & P7H_ACC != 0 {
        write_string(w, hmm.acc.as_deref().unwrap_or(""))?;
    }
    if hmm.flags & P7H_DESC != 0 {
        write_string(w, hmm.desc.as_deref().unwrap_or(""))?;
    }

    if hmm.flags & P7H_RF != 0 {
        write_annotation(w, &hmm.rf, hmm.m)?;
    }
    if hmm.flags & P7H_MMASK != 0 {
        write_annotation(w, &hmm.mm, hmm.m)?;
    }
    if hmm.flags & P7H_CONS != 0 {
        write_annotation(w, &hmm.consensus, hmm.m)?;
    }
    if hmm.flags & P7H_CS != 0 {
        write_annotation(w, &hmm.cs, hmm.m)?;
    }
    if hmm.flags & P7H_CA != 0 {
        write_annotation(w, &hmm.ca, hmm.m)?;
    }

    // Command log
    write_string(w, hmm.comlog.as_deref().unwrap_or(""))?;

    // nseq, eff_nseq
    w.write_all(&hmm.nseq.to_ne_bytes())
        .map_err(HmmerError::Io)?;
    w.write_all(&hmm.eff_nseq.to_ne_bytes())
        .map_err(HmmerError::Io)?;

    // max_length
    w.write_all(&hmm.max_length.to_ne_bytes())
        .map_err(HmmerError::Io)?;

    // Creation time
    write_string(w, hmm.ctime.as_deref().unwrap_or(""))?;

    // Map
    if hmm.flags & P7H_MAP != 0 {
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

    // E-value params
    for i in 0..NEVPARAM {
        w.write_all(&hmm.evparam[i].to_ne_bytes())
            .map_err(HmmerError::Io)?;
    }

    // Cutoffs
    for i in 0..NCUTOFFS {
        w.write_all(&hmm.cutoff[i].to_ne_bytes())
            .map_err(HmmerError::Io)?;
    }

    // Composition
    if hmm.flags & P7H_COMPO != 0 {
        for i in 0..k.min(MAXABET) {
            w.write_all(&hmm.compo[i].to_ne_bytes())
                .map_err(HmmerError::Io)?;
        }
    }

    Ok(())
}

fn write_annotation<W: std::io::Write>(
    w: &mut W,
    data: &Option<Vec<u8>>,
    m: usize,
) -> HmmerResult<()> {
    let len = m + 2;
    if let Some(d) = data {
        if d.len() >= len {
            w.write_all(&d[..len]).map_err(HmmerError::Io)?;
        } else {
            w.write_all(d).map_err(HmmerError::Io)?;
            for _ in d.len()..len {
                w.write_all(&[0u8]).map_err(HmmerError::Io)?;
            }
        }
    } else {
        let mut empty = vec![b' '; len];
        empty[len - 1] = 0;
        w.write_all(&empty).map_err(HmmerError::Io)?;
    }
    Ok(())
}

fn write_string<W: std::io::Write>(w: &mut W, s: &str) -> HmmerResult<()> {
    let bytes = s.as_bytes();
    let len = (bytes.len() + 1) as i32; // include null terminator
    w.write_all(&len.to_ne_bytes()).map_err(HmmerError::Io)?;
    w.write_all(bytes).map_err(HmmerError::Io)?;
    w.write_all(&[0u8]).map_err(HmmerError::Io)?; // null terminator
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::process::Command;

    fn test_path(relative: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
    }

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
