//! Biological sequence type and FASTA I/O.
//! Simplified port of esl_sq and esl_sqio_ascii for FASTA format.

use crate::alphabet::{Alphabet, Dsq, DSQ_SENTINEL};
use crate::errors::{HmmerError, HmmerResult};
use std::io::{BufRead, BufReader, Read};

/// A biological sequence (digital or text mode).
#[derive(Debug, Clone)]
pub struct Sequence {
    pub name: String,
    pub acc: String,
    pub desc: String,
    /// Digital sequence, 1-based: dsq[0] = SENTINEL, dsq[1..=n] = seq, dsq[n+1] = SENTINEL
    pub dsq: Vec<Dsq>,
    /// Length of the sequence
    pub n: usize,
    /// Full source length (same as n for complete sequences)
    pub l: usize,
}

impl Sequence {
    pub fn new() -> Self {
        Sequence {
            name: String::new(),
            acc: String::new(),
            desc: String::new(),
            dsq: vec![DSQ_SENTINEL],
            n: 0,
            l: 0,
        }
    }

    /// Reuse the sequence object for a new read (avoids reallocation).
    pub fn reuse(&mut self) {
        self.name.clear();
        self.acc.clear();
        self.desc.clear();
        self.dsq.clear();
        self.dsq.push(DSQ_SENTINEL);
        self.n = 0;
        self.l = 0;
    }
}

/// FASTA sequence file reader.
pub struct SeqFile<R: Read> {
    reader: BufReader<R>,
    abc: Alphabet,
    /// Buffered line for look-ahead
    pending_header: Option<String>,
    at_eof: bool,
}

impl<R: Read> SeqFile<R> {
    pub fn new(reader: R, abc: Alphabet) -> Self {
        SeqFile {
            reader: BufReader::new(reader),
            abc,
            pending_header: None,
            at_eof: false,
        }
    }

    /// Read the next sequence. Returns Ok(true) if a sequence was read, Ok(false) at EOF.
    /// Supports FASTA (>name) and UniProt/SwissProt (ID/SQ) formats.
    pub fn read(&mut self, sq: &mut Sequence) -> HmmerResult<bool> {
        if self.at_eof {
            return Ok(false);
        }

        sq.reuse();

        // Find the start of a sequence record
        let first_line = if let Some(h) = self.pending_header.take() {
            h
        } else {
            let mut line = String::new();
            loop {
                line.clear();
                let n = self.reader.read_line(&mut line).map_err(HmmerError::Io)?;
                if n == 0 {
                    self.at_eof = true;
                    return Ok(false);
                }
                let trimmed = line.trim();
                if trimmed.starts_with('>')
                    || trimmed.starts_with("ID ")
                    || trimmed.starts_with("LOCUS ")
                {
                    break;
                }
            }
            line
        };

        let trimmed = first_line.trim();

        if trimmed.starts_with('>') {
            // FASTA format
            let after_gt = &trimmed[1..];
            let parts: Vec<&str> = after_gt.splitn(2, char::is_whitespace).collect();
            sq.name = parts[0].to_string();
            if let Some(desc) = parts.get(1) {
                sq.desc = desc.trim().to_string();
            }

            let mut line = String::new();
            loop {
                line.clear();
                let n = self.reader.read_line(&mut line).map_err(HmmerError::Io)?;
                if n == 0 {
                    self.at_eof = true;
                    break;
                }
                let trimmed = line.trim();
                if trimmed.starts_with('>') {
                    self.pending_header = Some(line.clone());
                    break;
                }
                for &ch in trimmed.as_bytes() {
                    let code = self.abc.digitize_symbol(ch);
                    if code != crate::alphabet::DSQ_IGNORED && code != crate::alphabet::DSQ_ILLEGAL
                    {
                        sq.dsq.push(code);
                    }
                }
            }
        } else if trimmed.starts_with("ID ") {
            // UniProt/SwissProt format
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                sq.name = parts[1].to_string();
            }

            // Read until SQ line, collecting DE and AC
            let mut line = String::new();
            let mut in_seq = false;
            loop {
                line.clear();
                let n = self.reader.read_line(&mut line).map_err(HmmerError::Io)?;
                if n == 0 {
                    self.at_eof = true;
                    break;
                }
                let trimmed = line.trim();

                if trimmed == "//" {
                    break; // end of record
                }

                if trimmed.starts_with("DE ") && sq.desc.is_empty() {
                    sq.desc = trimmed[5..].trim().to_string();
                } else if trimmed.starts_with("AC ") && sq.acc.is_empty() {
                    let acc = trimmed[5..].trim().trim_end_matches(';');
                    sq.acc = acc.split(';').next().unwrap_or("").trim().to_string();
                } else if trimmed.starts_with("SQ ") {
                    in_seq = true;
                    continue;
                }

                if in_seq {
                    // Sequence data: letters and spaces, ending with //
                    for &ch in trimmed.as_bytes() {
                        if ch.is_ascii_alphabetic() {
                            let code = self.abc.digitize_symbol(ch);
                            if code != crate::alphabet::DSQ_IGNORED
                                && code != crate::alphabet::DSQ_ILLEGAL
                            {
                                sq.dsq.push(code);
                            }
                        }
                    }
                }
            }
        }

        if trimmed.starts_with("LOCUS ") {
            // GenBank format
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                sq.name = parts[1].to_string();
            }

            let mut line = String::new();
            let mut in_seq = false;
            loop {
                line.clear();
                let n = self.reader.read_line(&mut line).map_err(HmmerError::Io)?;
                if n == 0 {
                    self.at_eof = true;
                    break;
                }
                let trimmed = line.trim();

                if trimmed == "//" {
                    break;
                }

                if trimmed.starts_with("DEFINITION") && sq.desc.is_empty() {
                    sq.desc = trimmed[10..].trim().to_string();
                } else if trimmed.starts_with("ACCESSION") && sq.acc.is_empty() {
                    sq.acc = trimmed[9..]
                        .trim()
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_string();
                } else if trimmed == "ORIGIN" {
                    in_seq = true;
                    continue;
                }

                if in_seq {
                    for &ch in trimmed.as_bytes() {
                        if ch.is_ascii_alphabetic() {
                            let code = self.abc.digitize_symbol(ch);
                            if code != crate::alphabet::DSQ_IGNORED
                                && code != crate::alphabet::DSQ_ILLEGAL
                            {
                                sq.dsq.push(code);
                            }
                        }
                    }
                }
            }
        }

        sq.dsq.push(DSQ_SENTINEL);
        sq.n = sq.dsq.len() - 2;
        sq.l = sq.n;

        Ok(sq.n > 0)
    }
}

/// Open a FASTA file for reading with the given alphabet.
/// Automatically detects and decompresses `.gz` files.
pub fn open_seq_file(
    path: &std::path::Path,
    abc: &Alphabet,
) -> HmmerResult<SeqFile<Box<dyn Read>>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let reader: Box<dyn Read> = if path.extension().map_or(false, |e| e == "gz") {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(SeqFile::new(reader, abc.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use std::path::Path;

    #[test]
    fn test_read_fasta() {
        let abc = Alphabet::amino();
        let mut sqf = open_seq_file(
            Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/hmmer/testsuite/20aa-alitest.fa"
            )),
            &abc,
        )
        .unwrap();

        let mut sq = Sequence::new();
        let mut count = 0;
        while sqf.read(&mut sq).unwrap() {
            count += 1;
            assert!(sq.n > 0);
            assert!(!sq.name.is_empty());
            sq.reuse();
        }
        assert!(count > 0, "Should read at least one sequence");
    }

    #[test]
    fn test_read_uniprot() {
        let abc = Alphabet::amino();
        let mut sqf = open_seq_file(
            Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/hmmer/tutorial/7LESS_DROME"
            )),
            &abc,
        )
        .unwrap();

        let mut sq = Sequence::new();
        assert!(sqf.read(&mut sq).unwrap());
        assert_eq!(sq.name, "7LESS_DROME");
        assert!(
            sq.n > 2500,
            "7LESS_DROME should be >2500 residues, got {}",
            sq.n
        );
    }

    #[test]
    fn test_read_globins() {
        let abc = Alphabet::amino();
        let mut sqf = open_seq_file(
            Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/hmmer/tutorial/globins45.fa"
            )),
            &abc,
        )
        .unwrap();

        let mut sq = Sequence::new();
        let mut count = 0;
        while sqf.read(&mut sq).unwrap() {
            count += 1;
            assert!(
                sq.n > 100,
                "Globin seq should be >100 residues, got {}",
                sq.n
            );
            sq.reuse();
        }
        assert!(
            count >= 40,
            "Should read ~45 globin sequences, got {}",
            count
        );
    }
}
