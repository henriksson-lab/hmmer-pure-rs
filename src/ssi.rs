//! Simple sequence/HMM index for fast random access by name.
//! Simplified alternative to Easel's esl_ssi.c.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Seek};
use std::path::Path;

use crate::errors::{HmmerError, HmmerResult};

/// An index mapping names to file offsets.
#[derive(Debug)]
pub struct Index {
    /// Map from name to byte offset in the file
    pub name_to_offset: HashMap<String, u64>,
    /// Map from accession to byte offset
    pub acc_to_offset: HashMap<String, u64>,
}

impl Index {
    /// Build an index from an HMM file by scanning for NAME/ACC lines.
    pub fn build_from_hmm_file(path: &Path) -> HmmerResult<Self> {
        let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
        let mut reader = BufReader::new(file);
        let mut name_to_offset = HashMap::new();
        let mut acc_to_offset = HashMap::new();

        let mut line = String::new();
        let mut record_offset: u64 = 0;
        // current_name tracks the most recent NAME for association with ACC

        loop {
            let offset = reader.stream_position().map_err(HmmerError::Io)?;
            line.clear();
            let n = reader.read_line(&mut line).map_err(HmmerError::Io)?;
            if n == 0 {
                break;
            }

            let trimmed = line.trim();

            if trimmed.starts_with("HMMER3/") {
                record_offset = offset;
            } else if trimmed.starts_with("NAME ") {
                let name = trimmed[5..].trim().to_string();
                name_to_offset.insert(name, record_offset);
            } else if trimmed.starts_with("ACC ") {
                let acc = trimmed[5..].trim().to_string();
                acc_to_offset.insert(acc, record_offset);
            }
        }

        Ok(Index {
            name_to_offset,
            acc_to_offset,
        })
    }

    /// Look up a key (name or accession) and return the file offset.
    pub fn lookup(&self, key: &str) -> Option<u64> {
        self.name_to_offset
            .get(key)
            .or_else(|| self.acc_to_offset.get(key))
            .copied()
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.name_to_offset.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_index() {
        let idx = Index::build_from_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/minipfam.hmm"
        )))
        .unwrap();
        assert!(idx.len() >= 10, "minipfam should have at least 10 HMMs, got {}", idx.len());
        assert!(idx.lookup("14-3-3").is_some());
    }
}
