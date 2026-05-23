//! Simple sequence/HMM index for fast random access by name.
//!
//! The in-memory index is used for direct Rust lookup. `write_hmm_ssi()` emits
//! Easel SSI v3 files compatible with C HMMER's `hmmfetch --index`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Seek, Write};
use std::path::{Path, PathBuf};

use crate::errors::{HmmerError, HmmerResult};

const SSI_V30_MAGIC: u32 = 0xd3d3c9b3;
const SSI_OFFSZ: u32 = 8;

#[derive(Debug, Clone)]
struct PrimaryKey {
    key: String,
    offset: u64,
}

#[derive(Debug, Clone)]
struct SecondaryKey {
    key: String,
    primary_key: String,
}

/// An index mapping names to file offsets.
#[derive(Debug)]
pub struct Index {
    /// Map from name to byte offset in the file
    pub name_to_offset: HashMap<String, u64>,
    /// Map from accession to byte offset
    pub acc_to_offset: HashMap<String, u64>,
    primary_keys: Vec<PrimaryKey>,
    secondary_keys: Vec<SecondaryKey>,
}

impl Index {
    /// Build an in-memory name/accession index from an ASCII HMM file.
    /// Scans for `HMMER3/` record starts, then `NAME` and `ACC` lines, and
    /// records each one's byte offset. Simpler in-memory alternative to
    /// the on-disk SSI files built by Easel's `esl_newssi_*` functions.
    pub fn build_from_hmm_file(path: &Path) -> HmmerResult<Self> {
        let records = scan_hmm_records(path)?;
        let mut name_to_offset = HashMap::new();
        let mut acc_to_offset = HashMap::new();
        let mut primary_keys = Vec::with_capacity(records.len());
        let mut secondary_keys = Vec::new();

        for record in records {
            if name_to_offset
                .insert(record.name.clone(), record.offset)
                .is_some()
            {
                return Err(HmmerError::Format(format!(
                    "Duplicate HMM name '{}' cannot be indexed",
                    record.name
                )));
            }
            primary_keys.push(PrimaryKey {
                key: record.name.clone(),
                offset: record.offset,
            });
            if let Some(acc) = record.acc {
                if acc_to_offset.insert(acc.clone(), record.offset).is_some() {
                    return Err(HmmerError::Format(format!(
                        "Duplicate HMM accession '{}' cannot be indexed",
                        acc
                    )));
                }
                secondary_keys.push(SecondaryKey {
                    key: acc,
                    primary_key: record.name,
                });
            }
        }

        Ok(Index {
            name_to_offset,
            acc_to_offset,
            primary_keys,
            secondary_keys,
        })
    }

    /// Look up `key` first as a name, then as an accession, and return the
    /// file offset of the containing HMM record (or `None` if not found).
    /// Analog of `esl_ssi_FindName()` from Easel's `esl_ssi.c`.
    pub fn lookup(&self, key: &str) -> Option<u64> {
        self.name_to_offset
            .get(key)
            .or_else(|| self.acc_to_offset.get(key))
            .copied()
    }

    /// Number of indexed primary names (excludes accession-only aliases).
    pub fn len(&self) -> usize {
        self.name_to_offset.len()
    }

    pub fn accession_len(&self) -> usize {
        self.acc_to_offset.len()
    }
}

struct HmmRecord {
    name: String,
    acc: Option<String>,
    offset: u64,
}

fn scan_hmm_records(path: &Path) -> HmmerResult<Vec<HmmRecord>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut line = String::new();
    let mut record_offset: Option<u64> = None;
    let mut name: Option<String> = None;
    let mut acc: Option<String> = None;

    loop {
        let offset = reader.stream_position().map_err(HmmerError::Io)?;
        line.clear();
        let n = reader.read_line(&mut line).map_err(HmmerError::Io)?;
        if n == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.starts_with("HMMER3/") {
            if record_offset.is_some() {
                return Err(HmmerError::Format(
                    "Missing // terminator before next HMM record".to_string(),
                ));
            }
            record_offset = Some(offset);
        } else if trimmed.starts_with("NAME ") {
            name = Some(trimmed[5..].trim().to_string());
        } else if trimmed.starts_with("ACC ") {
            acc = Some(trimmed[4..].trim().to_string());
        } else if trimmed == "//" {
            finish_record(&mut records, record_offset.take(), name.take(), acc.take())?;
        }
    }

    if record_offset.is_some() {
        return Err(HmmerError::Format(
            "Missing // terminator at end of HMM file".to_string(),
        ));
    }
    Ok(records)
}

fn finish_record(
    records: &mut Vec<HmmRecord>,
    record_offset: Option<u64>,
    name: Option<String>,
    acc: Option<String>,
) -> HmmerResult<()> {
    let Some(offset) = record_offset else {
        return Ok(());
    };
    let Some(name) = name else {
        return Err(HmmerError::Format(
            "Every HMM must have a name to be indexed".to_string(),
        ));
    };
    records.push(HmmRecord { name, acc, offset });
    Ok(())
}

/// Write a C/Easel-compatible SSI v3 index for an ASCII HMM file.
///
/// HMM records have no data offset or length in C HMMER's `hmmfetch --index`,
/// so primary keys store only record offsets and accessions are secondary keys.
pub fn write_hmm_ssi(hmm_path: &Path) -> HmmerResult<(PathBuf, usize, usize)> {
    let index = Index::build_from_hmm_file(hmm_path)?;
    let ssi_path = PathBuf::from(format!("{}.ssi", hmm_path.display()));
    if ssi_path.exists() {
        return Err(HmmerError::Format(format!(
            "SSI index {} already exists; delete or rename it",
            ssi_path.display()
        )));
    }

    write_hmm_ssi_records(
        hmm_path,
        &ssi_path,
        index.primary_keys.iter().map(|p| {
            let acc = index
                .secondary_keys
                .iter()
                .find(|s| s.primary_key == p.key)
                .map(|s| s.key.clone());
            (p.key.clone(), acc, p.offset)
        }),
        false,
    )
}

/// Write an Easel SSI v3 index from already-known HMM record offsets.
///
/// This is used by `hmmpress`, whose index points at generated `.h3m` binary
/// records rather than the original ASCII input file.
pub fn write_hmm_ssi_records<I>(
    indexed_path: &Path,
    ssi_path: &Path,
    records: I,
    overwrite: bool,
) -> HmmerResult<(PathBuf, usize, usize)>
where
    I: IntoIterator<Item = (String, Option<String>, u64)>,
{
    if ssi_path.exists() && !overwrite {
        return Err(HmmerError::Format(format!(
            "SSI index {} already exists; delete or rename it",
            ssi_path.display()
        )));
    }

    let mut primary = Vec::new();
    let mut secondary = Vec::new();
    for (name, acc, offset) in records {
        primary.push(PrimaryKey {
            key: name.clone(),
            offset,
        });
        if let Some(acc) = acc {
            if !acc.is_empty() {
                secondary.push(SecondaryKey {
                    key: acc,
                    primary_key: name,
                });
            }
        }
    }

    primary.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
    secondary.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));

    let flen = fixed_len(indexed_path.to_string_lossy().as_ref());
    let plen = primary.iter().map(|p| fixed_len(&p.key)).max().unwrap_or(0);
    let slen = secondary
        .iter()
        .map(|s| fixed_len(&s.key))
        .max()
        .unwrap_or(0);

    let frecsize = flen + 4 * std::mem::size_of::<u32>();
    let precsize = plen + std::mem::size_of::<u16>() + 2 * SSI_OFFSZ as usize + 8;
    let srecsize = slen + plen;
    let foffset = 9 * std::mem::size_of::<u32>()
        + 2 * std::mem::size_of::<u64>()
        + std::mem::size_of::<u16>()
        + 3 * SSI_OFFSZ as usize;
    let poffset = foffset + frecsize;
    let soffset = poffset + precsize * primary.len();

    let mut out = std::fs::File::create(ssi_path).map_err(HmmerError::Io)?;
    write_u32(&mut out, SSI_V30_MAGIC)?;
    write_u32(&mut out, 0)?;
    write_u32(&mut out, SSI_OFFSZ)?;
    write_u16(&mut out, 1)?;
    write_u64(&mut out, primary.len() as u64)?;
    write_u64(&mut out, secondary.len() as u64)?;
    write_u32(&mut out, flen as u32)?;
    write_u32(&mut out, plen as u32)?;
    write_u32(&mut out, slen as u32)?;
    write_u32(&mut out, frecsize as u32)?;
    write_u32(&mut out, precsize as u32)?;
    write_u32(&mut out, srecsize as u32)?;
    write_u64(&mut out, foffset as u64)?;
    write_u64(&mut out, poffset as u64)?;
    write_u64(&mut out, soffset as u64)?;

    write_fixed_str(&mut out, indexed_path.to_string_lossy().as_ref(), flen)?;
    write_u32(&mut out, 0)?; // HMM file format code
    write_u32(&mut out, 0)?; // file flags
    write_u32(&mut out, 0)?; // bpl
    write_u32(&mut out, 0)?; // rpl

    let mut previous: Option<&str> = None;
    for p in &primary {
        if previous == Some(p.key.as_str()) {
            return Err(HmmerError::Format(format!(
                "Duplicate HMM name '{}' cannot be indexed",
                p.key
            )));
        }
        previous = Some(&p.key);
        write_fixed_str(&mut out, &p.key, plen)?;
        write_u16(&mut out, 0)?;
        write_u64(&mut out, p.offset)?;
        write_u64(&mut out, 0)?;
        write_i64(&mut out, 0)?;
    }

    previous = None;
    for s in &secondary {
        if previous == Some(s.key.as_str()) {
            return Err(HmmerError::Format(format!(
                "Duplicate HMM accession '{}' cannot be indexed",
                s.key
            )));
        }
        previous = Some(&s.key);
        write_fixed_str(&mut out, &s.key, slen)?;
        write_fixed_str(&mut out, &s.primary_key, plen)?;
    }

    Ok((ssi_path.to_path_buf(), primary.len(), secondary.len()))
}

fn fixed_len(s: &str) -> usize {
    s.len() + 1
}

fn write_fixed_str<W: Write>(w: &mut W, s: &str, len: usize) -> HmmerResult<()> {
    let mut buf = vec![0u8; len];
    let bytes = s.as_bytes();
    if bytes.len() >= len {
        return Err(HmmerError::Format(format!(
            "SSI string '{}' exceeds fixed field length {}",
            s, len
        )));
    }
    buf[..bytes.len()].copy_from_slice(bytes);
    w.write_all(&buf).map_err(HmmerError::Io)
}

fn write_u16<W: Write>(w: &mut W, v: u16) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}

fn write_i64<W: Write>(w: &mut W, v: i64) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: building an index for `minipfam.hmm` should index >=10 HMMs
    /// and let us look up `14-3-3` by name.
    #[test]
    fn test_build_index() {
        let idx = Index::build_from_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/minipfam.hmm"
        )))
        .unwrap();
        assert!(
            idx.len() >= 10,
            "minipfam should have at least 10 HMMs, got {}",
            idx.len()
        );
        assert!(idx.lookup("14-3-3").is_some());
    }

    #[test]
    fn write_hmm_ssi_emits_easel_v3_header_and_keys() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("two.hmm");
        std::fs::write(
            &hmm_path,
            b"HMMER3/f\nNAME  b\nACC   PF00002\nLENG  1\n//\nHMMER3/f\nNAME  a\nACC   PF00001\nLENG  1\n//\n",
        )
        .unwrap();

        let (ssi_path, nprimary, nsecondary) = write_hmm_ssi(&hmm_path).unwrap();
        assert_eq!(nprimary, 2);
        assert_eq!(nsecondary, 2);

        let bytes = std::fs::read(ssi_path).unwrap();
        assert_eq!(
            u32::from_be_bytes(bytes[0..4].try_into().unwrap()),
            SSI_V30_MAGIC
        );
        assert_eq!(
            u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
            SSI_OFFSZ
        );
        assert_eq!(u16::from_be_bytes(bytes[12..14].try_into().unwrap()), 1);
        assert_eq!(u64::from_be_bytes(bytes[14..22].try_into().unwrap()), 2);
        assert_eq!(u64::from_be_bytes(bytes[22..30].try_into().unwrap()), 2);

        let poffset = u64::from_be_bytes(bytes[62..70].try_into().unwrap()) as usize;
        assert_eq!(&bytes[poffset..poffset + 2], b"a\0");
    }

    #[test]
    fn index_rejects_missing_record_terminator() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("bad.hmm");
        std::fs::write(&hmm_path, b"HMMER3/f\nNAME  bad\nLENG  1\n").unwrap();

        let err = Index::build_from_hmm_file(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("Missing // terminator"));
    }
}
