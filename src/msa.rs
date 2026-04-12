//! Multiple Sequence Alignment (MSA) I/O — Stockholm format.

use crate::alphabet::{Alphabet, Dsq};
use crate::errors::{HmmerError, HmmerResult};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// A multiple sequence alignment.
#[derive(Debug, Clone)]
pub struct Msa {
    /// Alignment name (from #=GF ID)
    pub name: String,
    /// Sequence names
    pub sqname: Vec<String>,
    /// Aligned sequences (text, with gap characters)
    pub aseq: Vec<Vec<u8>>,
    /// Number of sequences
    pub nseq: usize,
    /// Alignment length (columns)
    pub alen: usize,
    /// Reference annotation (#=GC RF)
    pub rf: Option<Vec<u8>>,
}

impl Msa {
    /// Digitize all sequences in the alignment.
    /// Returns a vector of digitized sequences (1-based, with sentinels).
    /// Gap characters are represented as the alphabet's gap code.
    pub fn digitize(&self, abc: &Alphabet) -> Vec<Vec<Dsq>> {
        let gap = abc.gap_code();
        self.aseq
            .iter()
            .map(|seq| {
                let mut dsq = Vec::with_capacity(self.alen + 2);
                dsq.push(crate::alphabet::DSQ_SENTINEL);
                for &ch in seq {
                    if ch == b'-' || ch == b'.' {
                        dsq.push(gap);
                    } else {
                        let code = abc.digitize_symbol(ch);
                        if code != crate::alphabet::DSQ_IGNORED {
                            dsq.push(code);
                        }
                    }
                }
                dsq.push(crate::alphabet::DSQ_SENTINEL);
                dsq
            })
            .collect()
    }
}

/// Read a Stockholm format MSA from a file.
pub fn read_stockholm(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let reader = BufReader::new(file);
    read_stockholm_from_reader(reader)
}

/// Read Stockholm MSAs from a reader.
pub fn read_stockholm_from_reader<R: Read>(reader: BufReader<R>) -> HmmerResult<Vec<Msa>> {
    let mut msas = Vec::new();
    let mut lines: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(HmmerError::Io)?;
        lines.push(line);
    }

    let mut i = 0;
    while i < lines.len() {
        // Find start of Stockholm block
        if lines[i].starts_with("# STOCKHOLM") {
            let start = i;
            // Find end
            while i < lines.len() && lines[i].trim() != "//" {
                i += 1;
            }
            let end = i;
            if let Some(msa) = parse_stockholm_block(&lines[start..=end.min(lines.len() - 1)])? {
                msas.push(msa);
            }
        }
        i += 1;
    }

    Ok(msas)
}

fn parse_stockholm_block(lines: &[String]) -> HmmerResult<Option<Msa>> {
    let mut name = String::new();
    let mut seq_order: Vec<String> = Vec::new();
    let mut seq_data: HashMap<String, Vec<u8>> = HashMap::new();
    let mut rf: Option<Vec<u8>> = None;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "//" || trimmed.starts_with("# STOCKHOLM") {
            continue;
        }

        if trimmed.starts_with("#=GF ID") {
            name = trimmed[7..].trim().to_string();
        } else if trimmed.starts_with("#=GC RF") {
            let rf_str = trimmed[7..].trim();
            match &mut rf {
                Some(existing) => existing.extend_from_slice(rf_str.as_bytes()),
                None => rf = Some(rf_str.as_bytes().to_vec()),
            }
        } else if trimmed.starts_with('#') {
            // Other annotation — skip
            continue;
        } else {
            // Sequence line: name  sequence
            let parts: Vec<&str> = trimmed.splitn(2, char::is_whitespace).collect();
            if parts.len() == 2 {
                let sqname = parts[0].to_string();
                let sqdata = parts[1].trim().as_bytes();

                if !seq_data.contains_key(&sqname) {
                    seq_order.push(sqname.clone());
                    seq_data.insert(sqname, sqdata.to_vec());
                } else {
                    seq_data.get_mut(&sqname).unwrap().extend_from_slice(sqdata);
                }
            }
        }
    }

    if seq_order.is_empty() {
        return Ok(None);
    }

    let alen = seq_data.values().map(|v| v.len()).max().unwrap_or(0);
    let nseq = seq_order.len();

    let aseq: Vec<Vec<u8>> = seq_order
        .iter()
        .map(|name| {
            let mut seq = seq_data.remove(name).unwrap_or_default();
            seq.resize(alen, b'-');
            seq
        })
        .collect();

    Ok(Some(Msa {
        name,
        sqname: seq_order,
        aseq,
        nseq,
        alen,
        rf,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_20aa_stockholm() {
        let msas = read_stockholm(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.sto"
        )))
        .unwrap();
        assert_eq!(msas.len(), 1);
        let msa = &msas[0];
        assert_eq!(msa.name, "test");
        assert_eq!(msa.nseq, 10);
        assert_eq!(msa.alen, 20);
        assert!(msa.rf.is_some());
    }

    #[test]
    fn test_read_globins4_stockholm() {
        let msas = read_stockholm(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/globins4.sto"
        )))
        .unwrap();
        assert_eq!(msas.len(), 1);
        let msa = &msas[0];
        assert_eq!(msa.nseq, 4);
        assert!(msa.alen > 100);
    }
}
