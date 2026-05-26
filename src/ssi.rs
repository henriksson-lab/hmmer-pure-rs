//! Simple sequence/HMM index for fast random access by name.
//!
//! The in-memory index is used for direct Rust lookup. `write_hmm_ssi()` emits
//! Easel SSI v3 files compatible with C HMMER's `hmmfetch --index`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::errors::{HmmerError, HmmerResult};
use crate::hmmfile;
use crate::hmmfile_binary;

const SSI_V30_MAGIC: u32 = 0xd3d3c9b3;
const SSI_OFFSZ: u32 = 8;
const SSI_OFFSZ_32: u32 = 4;

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

/// Return `<path><suffix>` without formatting the path through UTF-8.
pub fn path_with_appended_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut path = path.as_os_str().to_os_string();
    path.push(suffix);
    PathBuf::from(path)
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

/// An Easel SSI v3 index loaded from disk.
#[derive(Debug)]
pub struct OnDiskIndex {
    pub indexed_path: PathBuf,
    primary_offsets: HashMap<String, u64>,
    secondary_to_primary: HashMap<String, String>,
}

impl OnDiskIndex {
    /// Look up `key` first as a primary HMM name, then as a secondary accession.
    pub fn lookup(&self, key: &str) -> Option<u64> {
        self.primary_offsets.get(key).copied().or_else(|| {
            self.secondary_to_primary
                .get(key)
                .and_then(|primary| self.primary_offsets.get(primary))
                .copied()
        })
    }
}

impl Index {
    /// Build an in-memory name/accession index from an ASCII or binary HMM file.
    /// Records each HMM's byte offset for SSI-style random access. Simpler
    /// in-memory alternative to the on-disk SSI files built by Easel's
    /// `esl_newssi_*` functions.
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
    if hmmfile_binary::looks_like_binary_hmm_file(path)? {
        return scan_binary_hmm_records(path);
    }
    scan_ascii_hmm_records(path)
}

fn scan_ascii_hmm_records(path: &Path) -> HmmerResult<Vec<HmmRecord>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut line = String::new();
    let mut record_offset: Option<u64> = None;
    let mut record_text = String::new();

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
            record_text.clear();
        }
        if record_offset.is_some() {
            record_text.push_str(&line);
        }
        if trimmed == "//" {
            finish_ascii_record(&mut records, record_offset.take(), &record_text)?;
            record_text.clear();
        }
    }

    if record_offset.is_some() {
        return Err(HmmerError::Format(
            "Missing // terminator at end of HMM file".to_string(),
        ));
    }
    Ok(records)
}

fn scan_binary_hmm_records(path: &Path) -> HmmerResult<Vec<HmmRecord>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();

    loop {
        let offset = reader.stream_position().map_err(HmmerError::Io)?;
        let Some(hmm) = hmmfile_binary::read_binary_hmm(&mut reader)? else {
            break;
        };
        records.push(HmmRecord {
            name: hmm.name,
            acc: hmm.acc,
            offset,
        });
    }

    Ok(records)
}

fn finish_ascii_record(
    records: &mut Vec<HmmRecord>,
    record_offset: Option<u64>,
    record_text: &str,
) -> HmmerResult<()> {
    let Some(offset) = record_offset else {
        return Ok(());
    };
    let mut hmms =
        hmmfile::read_hmms(BufReader::new(Cursor::new(record_text.as_bytes().to_vec())))?;
    if hmms.len() != 1 {
        return Err(HmmerError::Format(format!(
            "Expected one HMM record at offset {offset}, found {}",
            hmms.len()
        )));
    }
    let hmm = hmms.remove(0);
    let name = hmm.name;
    if name.is_empty() {
        return Err(HmmerError::Format(
            "Every HMM must have a name to be indexed".to_string(),
        ));
    }
    let acc = hmm.acc;
    records.push(HmmRecord { name, acc, offset });
    Ok(())
}

/// Write a C/Easel-compatible SSI v3 index for an ASCII HMM file.
///
/// HMM records have no data offset or length in C HMMER's `hmmfetch --index`,
/// so primary keys store only record offsets and accessions are secondary keys.
pub fn write_hmm_ssi(hmm_path: &Path) -> HmmerResult<(PathBuf, usize, usize)> {
    let index = Index::build_from_hmm_file(hmm_path)?;
    let ssi_path = path_with_appended_suffix(hmm_path, ".ssi");
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

/// Load the Easel SSI v3 sidecar for an HMM file, if it exists.
pub fn read_hmm_ssi(hmm_path: &Path) -> HmmerResult<Option<OnDiskIndex>> {
    let ssi_path = path_with_appended_suffix(hmm_path, ".ssi");
    if !ssi_path.exists() {
        return Ok(None);
    }
    read_hmm_ssi_path(&ssi_path).map(Some)
}

fn read_hmm_ssi_path(ssi_path: &Path) -> HmmerResult<OnDiskIndex> {
    let mut file = std::fs::File::open(ssi_path).map_err(HmmerError::Io)?;
    let magic = read_u32(&mut file)?;
    if magic != SSI_V30_MAGIC {
        return Err(HmmerError::Format(format!(
            "Bad SSI magic in {}: {magic:#x}",
            ssi_path.display()
        )));
    }
    let _flags = read_u32(&mut file)?;
    let offsz = read_u32(&mut file)?;
    let nfiles = read_u16(&mut file)?;
    let nprimary = read_u64(&mut file)?;
    let nsecondary = read_u64(&mut file)?;
    let flen = read_u32(&mut file)? as usize;
    let plen = read_u32(&mut file)? as usize;
    let slen = read_u32(&mut file)? as usize;
    let frecsize = read_u32(&mut file)? as usize;
    let precsize = read_u32(&mut file)? as usize;
    let srecsize = read_u32(&mut file)? as usize;
    let foffset = read_offset(&mut file, offsz)?;
    let poffset = read_offset(&mut file, offsz)?;
    let soffset = read_offset(&mut file, offsz)?;

    if (offsz != SSI_OFFSZ && offsz != SSI_OFFSZ_32) || nfiles != 1 {
        return Err(HmmerError::Format(format!(
            "Unsupported SSI header in {}: offsz={offsz} nfiles={nfiles}",
            ssi_path.display()
        )));
    }
    let expected_frecsize = flen + 4 * std::mem::size_of::<u32>();
    let expected_precsize = plen + std::mem::size_of::<u16>() + 2 * offsz as usize + 8;
    let expected_srecsize = slen + plen;
    if frecsize != expected_frecsize
        || precsize != expected_precsize
        || srecsize != expected_srecsize
    {
        return Err(HmmerError::Format(format!(
            "SSI index {} has inconsistent record sizes",
            ssi_path.display()
        )));
    }

    file.seek(SeekFrom::Start(foffset))
        .map_err(HmmerError::Io)?;
    let indexed_path = read_fixed_path(&mut file, flen)?;
    skip_exact(&mut file, 4 * std::mem::size_of::<u32>())?;

    let mut primary_offsets = HashMap::new();
    file.seek(SeekFrom::Start(poffset))
        .map_err(HmmerError::Io)?;
    for _ in 0..nprimary {
        let key = read_fixed_string(&mut file, plen)?;
        let file_idx = read_u16(&mut file)?;
        let offset = read_offset(&mut file, offsz)?;
        let data_offset = read_offset(&mut file, offsz)?;
        let record_len = read_i64(&mut file)?;
        if file_idx != 0 || data_offset != 0 || record_len != 0 {
            return Err(HmmerError::Format(format!(
                "SSI index {} contains unsupported primary record for {key}",
                ssi_path.display()
            )));
        }
        if primary_offsets.insert(key.clone(), offset).is_some() {
            return Err(HmmerError::Format(format!(
                "SSI index {} contains duplicate primary key {key}",
                ssi_path.display()
            )));
        }
    }

    let mut secondary_to_primary = HashMap::new();
    file.seek(SeekFrom::Start(soffset))
        .map_err(HmmerError::Io)?;
    for _ in 0..nsecondary {
        let key = read_fixed_string(&mut file, slen)?;
        let primary = read_fixed_string(&mut file, plen)?;
        if !primary_offsets.contains_key(&primary) {
            return Err(HmmerError::Format(format!(
                "SSI index {} secondary key {key} references missing primary {primary}",
                ssi_path.display()
            )));
        }
        if secondary_to_primary.insert(key.clone(), primary).is_some() {
            return Err(HmmerError::Format(format!(
                "SSI index {} contains duplicate secondary key {key}",
                ssi_path.display()
            )));
        }
    }

    Ok(OnDiskIndex {
        indexed_path,
        primary_offsets,
        secondary_to_primary,
    })
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
    write_hmm_ssi_records_with_stored_path(indexed_path, indexed_path, ssi_path, records, overwrite)
}

/// Write an SSI v3 index while allowing the fixed file-table width and stored
/// file-table name to differ. Easel sizes the field from the path passed to
/// `esl_newssi_AddFile()`, but stores only its basename.
pub fn write_hmm_ssi_records_with_stored_path<I>(
    indexed_path: &Path,
    stored_path: &Path,
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

    let flen = fixed_path_len(indexed_path);
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

    write_fixed_path(&mut out, stored_path, flen)?;
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

fn fixed_path_len(path: &Path) -> usize {
    path_bytes(path).len() + 1
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

fn write_fixed_path<W: Write>(w: &mut W, path: &Path, len: usize) -> HmmerResult<()> {
    let mut buf = vec![0u8; len];
    let bytes = path_bytes(path);
    if bytes.len() >= len {
        return Err(HmmerError::Format(format!(
            "SSI path '{}' exceeds fixed field length {}",
            path.display(),
            len
        )));
    }
    buf[..bytes.len()].copy_from_slice(&bytes);
    w.write_all(&buf).map_err(HmmerError::Io)
}

#[cfg(unix)]
fn path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;

    path.as_os_str().as_bytes().to_vec()
}

pub fn path_file_name(path: &Path) -> PathBuf {
    path.file_name()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(""))
}

#[cfg(not(unix))]
fn path_bytes(path: &Path) -> Vec<u8> {
    path.to_string_lossy().as_bytes().to_vec()
}

#[cfg(unix)]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStringExt;

    PathBuf::from(std::ffi::OsString::from_vec(bytes.to_vec()))
}

#[cfg(not(unix))]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(bytes).into_owned())
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

fn read_fixed_string<R: Read>(r: &mut R, len: usize) -> HmmerResult<String> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(String::from_utf8_lossy(&buf[..end]).to_string())
}

fn read_fixed_path<R: Read>(r: &mut R, len: usize) -> HmmerResult<PathBuf> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(path_from_bytes(&buf[..end]))
}

fn skip_exact<R: Read>(r: &mut R, len: usize) -> HmmerResult<()> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)
}

fn read_u16<R: Read>(r: &mut R) -> HmmerResult<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> HmmerResult<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u64::from_be_bytes(buf))
}

fn read_offset<R: Read>(r: &mut R, offsz: u32) -> HmmerResult<u64> {
    match offsz {
        SSI_OFFSZ_32 => read_u32(r).map(u64::from),
        SSI_OFFSZ => read_u64(r),
        _ => Err(HmmerError::Format(format!(
            "Unsupported SSI offset size {offsz}"
        ))),
    }
}

fn read_i64<R: Read>(r: &mut R) -> HmmerResult<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(i64::from_be_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fn3_record_with(name: &str, acc: &str) -> Vec<u8> {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/fn3.hmm"
        ))
        .unwrap();
        text.replace("NAME  fn3", &format!("NAME  {name}"))
            .replace("ACC   PF00041.13", &format!("ACC   {acc}"))
            .into_bytes()
    }

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
        let mut data = fn3_record_with("b", "PF00002");
        data.extend_from_slice(&fn3_record_with("a", "PF00001"));
        std::fs::write(&hmm_path, data).unwrap();

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
    fn read_hmm_ssi_accepts_32_bit_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let ssi_path = dir.path().join("small.ssi");

        let stored_path = "small.hmm";
        let primary = "alpha";
        let secondary = "PF00001";
        let flen = fixed_len(stored_path);
        let plen = fixed_len(primary);
        let slen = fixed_len(secondary);
        let frecsize = flen + 4 * std::mem::size_of::<u32>();
        let precsize = plen + std::mem::size_of::<u16>() + 2 * SSI_OFFSZ_32 as usize + 8;
        let srecsize = slen + plen;
        let foffset = 9 * std::mem::size_of::<u32>()
            + 2 * std::mem::size_of::<u64>()
            + std::mem::size_of::<u16>()
            + 3 * SSI_OFFSZ_32 as usize;
        let poffset = foffset + frecsize;
        let soffset = poffset + precsize;

        let mut out = std::fs::File::create(&ssi_path).unwrap();
        write_u32(&mut out, SSI_V30_MAGIC).unwrap();
        write_u32(&mut out, 0).unwrap();
        write_u32(&mut out, SSI_OFFSZ_32).unwrap();
        write_u16(&mut out, 1).unwrap();
        write_u64(&mut out, 1).unwrap();
        write_u64(&mut out, 1).unwrap();
        write_u32(&mut out, flen as u32).unwrap();
        write_u32(&mut out, plen as u32).unwrap();
        write_u32(&mut out, slen as u32).unwrap();
        write_u32(&mut out, frecsize as u32).unwrap();
        write_u32(&mut out, precsize as u32).unwrap();
        write_u32(&mut out, srecsize as u32).unwrap();
        write_u32(&mut out, foffset as u32).unwrap();
        write_u32(&mut out, poffset as u32).unwrap();
        write_u32(&mut out, soffset as u32).unwrap();

        write_fixed_str(&mut out, stored_path, flen).unwrap();
        write_u32(&mut out, 0).unwrap();
        write_u32(&mut out, 0).unwrap();
        write_u32(&mut out, 0).unwrap();
        write_u32(&mut out, 0).unwrap();

        write_fixed_str(&mut out, primary, plen).unwrap();
        write_u16(&mut out, 0).unwrap();
        write_u32(&mut out, 1234).unwrap();
        write_u32(&mut out, 0).unwrap();
        write_i64(&mut out, 0).unwrap();

        write_fixed_str(&mut out, secondary, slen).unwrap();
        write_fixed_str(&mut out, primary, plen).unwrap();
        drop(out);

        let on_disk = read_hmm_ssi_path(&ssi_path).unwrap();
        assert_eq!(on_disk.indexed_path, PathBuf::from(stored_path));
        assert_eq!(on_disk.lookup(primary), Some(1234));
        assert_eq!(on_disk.lookup(secondary), Some(1234));
    }

    #[cfg(unix)]
    #[test]
    fn write_hmm_ssi_preserves_non_utf8_path_bytes() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir
            .path()
            .join(std::ffi::OsString::from_vec(b"two-\xff.hmm".to_vec()));
        let mut data = fn3_record_with("b", "PF00002");
        data.extend_from_slice(&fn3_record_with("a", "PF00001"));
        std::fs::write(&hmm_path, data).unwrap();

        let (ssi_path, nprimary, nsecondary) = write_hmm_ssi(&hmm_path).unwrap();
        assert_eq!(nprimary, 2);
        assert_eq!(nsecondary, 2);
        assert!(ssi_path.exists());
        assert_eq!(ssi_path, path_with_appended_suffix(&hmm_path, ".ssi"));
        assert!(ssi_path.as_os_str().as_bytes().contains(&0xff));

        let on_disk = read_hmm_ssi(&hmm_path).unwrap().unwrap();
        assert_eq!(on_disk.indexed_path, hmm_path);
        assert!(on_disk.indexed_path.as_os_str().as_bytes().contains(&0xff));
    }

    #[test]
    fn index_rejects_missing_record_terminator() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("bad.hmm");
        std::fs::write(&hmm_path, b"HMMER3/f\nNAME  bad\nLENG  1\n").unwrap();

        let err = Index::build_from_hmm_file(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("Missing // terminator"));
    }

    #[test]
    fn index_rejects_terminated_record_that_full_parser_rejects() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("bad.hmm");
        std::fs::write(&hmm_path, b"HMMER3/f\nNAME  bad\nLENG  1\n//\n").unwrap();

        let err = write_hmm_ssi(&hmm_path).unwrap_err();
        assert!(
            err.to_string().contains("Unexpected EOF in HMM header")
                || err.to_string().contains("Missing ALPH")
        );
        assert!(!path_with_appended_suffix(&hmm_path, ".ssi").exists());
    }
}
