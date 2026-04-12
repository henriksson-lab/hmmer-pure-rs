//! Binary sequence database format for fast reading.
//! Simplified port of esl_dsqdata — stores digital sequences in a compact binary format.

use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;

#[cfg(test)]
use crate::alphabet::Alphabet;
use crate::alphabet::DSQ_SENTINEL;
use crate::errors::{HmmerError, HmmerResult};
use crate::sequence::Sequence;

const DSQDATA_MAGIC: u32 = 0xD5AD474A;

/// Write a sequence database in binary format.
pub fn write_dsqdata(path: &Path, sequences: &[Sequence]) -> HmmerResult<()> {
    let file = std::fs::File::create(path).map_err(HmmerError::Io)?;
    let mut w = BufWriter::new(file);

    // Header
    w.write_all(&DSQDATA_MAGIC.to_le_bytes()).map_err(HmmerError::Io)?;
    w.write_all(&(sequences.len() as u64).to_le_bytes()).map_err(HmmerError::Io)?;

    // Write each sequence
    for sq in sequences {
        // Name length + name
        let name_bytes = sq.name.as_bytes();
        w.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(HmmerError::Io)?;
        w.write_all(name_bytes).map_err(HmmerError::Io)?;

        // Sequence length + digital sequence (excluding sentinels)
        w.write_all(&(sq.n as u64).to_le_bytes()).map_err(HmmerError::Io)?;
        w.write_all(&sq.dsq[1..=sq.n]).map_err(HmmerError::Io)?;
    }

    Ok(())
}

/// Read sequences from a binary dsqdata file.
pub fn read_dsqdata(path: &Path) -> HmmerResult<Vec<Sequence>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut r = BufReader::new(file);

    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4).map_err(HmmerError::Io)?;
    let magic = u32::from_le_bytes(buf4);
    if magic != DSQDATA_MAGIC {
        return Err(HmmerError::Format("Bad dsqdata magic".to_string()));
    }

    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8).map_err(HmmerError::Io)?;
    let nseq = u64::from_le_bytes(buf8) as usize;

    let mut sequences = Vec::with_capacity(nseq);

    for _ in 0..nseq {
        r.read_exact(&mut buf4).map_err(HmmerError::Io)?;
        let name_len = u32::from_le_bytes(buf4) as usize;
        let mut name_buf = vec![0u8; name_len];
        r.read_exact(&mut name_buf).map_err(HmmerError::Io)?;
        let name = String::from_utf8_lossy(&name_buf).to_string();

        r.read_exact(&mut buf8).map_err(HmmerError::Io)?;
        let n = u64::from_le_bytes(buf8) as usize;

        let mut seq_buf = vec![0u8; n];
        r.read_exact(&mut seq_buf).map_err(HmmerError::Io)?;

        let mut dsq = Vec::with_capacity(n + 2);
        dsq.push(DSQ_SENTINEL);
        dsq.extend_from_slice(&seq_buf);
        dsq.push(DSQ_SENTINEL);

        sequences.push(Sequence {
            name,
            acc: String::new(),
            desc: String::new(),
            dsq,
            n,
            l: n,
        });
    }

    Ok(sequences)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsqdata_roundtrip() {
        let abc = Alphabet::amino();
        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
        let seqs = vec![Sequence {
            name: "test".to_string(),
            acc: String::new(),
            desc: String::new(),
            dsq,
            n: 20,
            l: 20,
        }];

        let path = std::env::temp_dir().join("test_dsqdata.bin");
        write_dsqdata(&path, &seqs).unwrap();
        let read_seqs = read_dsqdata(&path).unwrap();

        assert_eq!(read_seqs.len(), 1);
        assert_eq!(read_seqs[0].name, "test");
        assert_eq!(read_seqs[0].n, 20);
        assert_eq!(read_seqs[0].dsq, seqs[0].dsq);

        std::fs::remove_file(&path).ok();
    }
}
