//! Biological sequence type and FASTA I/O.
//! Simplified port of esl_sq and esl_sqio_ascii for FASTA format.

use crate::alphabet::{Alphabet, Dsq, DSQ_IGNORED, DSQ_ILLEGAL, DSQ_SENTINEL};
use crate::errors::{HmmerError, HmmerResult};
use std::io::{BufRead, BufReader, Read};

/// A biological sequence (digital or text mode).
#[derive(Debug, Clone)]
pub struct Sequence {
    pub name: String,
    pub acc: String,
    pub desc: String,
    /// Digital sequence, 1-based: `dsq[0]` = SENTINEL, `dsq[1..=n]` = seq, `dsq[n+1]` = SENTINEL
    pub dsq: Vec<Dsq>,
    /// Length of the sequence
    pub n: usize,
    /// Full source length (same as n for complete sequences)
    pub l: usize,
}

impl Sequence {
    /// Create a new empty digital `Sequence` (port of `esl_sq_Create`).
    ///
    /// The dsq starts with a single `DSQ_SENTINEL`; subsequent reads append
    /// residues at positions `1..=n` and add a trailing sentinel.
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

    /// Reset the object so a new sequence may be read into it
    /// (port of `esl_sq_Reuse`).
    ///
    /// Clears metadata and the digital sequence buffer (re-seeding it with
    /// the leading sentinel) without freeing capacity, so a hot read loop
    /// avoids reallocation between records.
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
    fasta_only: bool,
    /// Buffered line for look-ahead
    pending_header: Option<String>,
    at_eof: bool,
}

impl<R: Read> SeqFile<R> {
    /// Wrap a `Read` source as a buffered sequence file using alphabet `abc`.
    /// Format autodetection happens on the first line of [`Self::read`].
    pub fn new(reader: R, abc: Alphabet) -> Self {
        SeqFile {
            reader: BufReader::new(reader),
            abc,
            fasta_only: false,
            pending_header: None,
            at_eof: false,
        }
    }

    /// Require FASTA input instead of using sequence format autodetection.
    pub fn with_fasta_only(mut self) -> Self {
        self.fasta_only = true;
        self
    }

    /// Read the next sequence record into `sq` (port of the common
    /// `esl_sqio_Read` entry point).
    ///
    /// Auto-detects FASTA (`>name`), UniProt/Swiss-Prot (`ID … // SQ …`), or
    /// GenBank (`LOCUS … // ORIGIN …`) format from the first non-empty line
    /// and parses one record. Residues are digitised through the alphabet,
    /// preserving ignored whitespace and rejecting illegal sequence symbols.
    /// Returns `Ok(false)` at EOF.
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
                    || (!self.fasta_only
                        && (trimmed.starts_with("ID ") || trimmed.starts_with("LOCUS ")))
                {
                    break;
                }
                if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    return Err(HmmerError::Format(format!(
                        "unrecognized sequence file record start: {}",
                        trimmed
                    )));
                }
            }
            line
        };

        let trimmed = first_line.trim();

        if trimmed.starts_with('>') {
            // FASTA format
            let after_gt = trimmed[1..].trim_start();
            if after_gt.is_empty() {
                return Err(HmmerError::Format("no FASTA name found".to_string()));
            }
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
                    if let Some(code) = self.digitize_fasta_residue(ch, &sq.name)? {
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

                if trimmed.starts_with("DE ") {
                    let de = trimmed[5..].trim();
                    if !de.is_empty() {
                        if !sq.desc.is_empty() {
                            sq.desc.push(' ');
                        }
                        sq.desc.push_str(de);
                    }
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
                        if ch.is_ascii_alphabetic()
                            || matches!(ch, b'*' | b'~' | b'-' | b'.' | b'_')
                        {
                            if let Some(code) =
                                self.digitize_sequence_residue(ch, "UniProt sequence", &sq.name)?
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
            let mut in_definition = false;
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

                if in_definition
                    && !in_seq
                    && (line.starts_with(' ') || line.starts_with('\t'))
                    && !trimmed.is_empty()
                {
                    if !sq.desc.is_empty() {
                        sq.desc.push(' ');
                    }
                    sq.desc.push_str(trimmed);
                    continue;
                }
                in_definition = false;

                if trimmed.starts_with("DEFINITION") && sq.desc.is_empty() {
                    sq.desc = trimmed[10..].trim().to_string();
                    in_definition = true;
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
                        if ch.is_ascii_alphabetic()
                            || matches!(ch, b'*' | b'~' | b'-' | b'.' | b'_')
                        {
                            if let Some(code) =
                                self.digitize_sequence_residue(ch, "GenBank sequence", &sq.name)?
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

        if sq.n == 0 {
            return Err(HmmerError::Format(format!(
                "zero-length sequence record '{}'",
                sq.name
            )));
        }
        Ok(true)
    }

    fn digitize_fasta_residue(&self, ch: u8, seq_name: &str) -> HmmerResult<Option<Dsq>> {
        self.digitize_sequence_residue(ch, "FASTA sequence", seq_name)
    }

    fn digitize_sequence_residue(
        &self,
        ch: u8,
        format_name: &str,
        seq_name: &str,
    ) -> HmmerResult<Option<Dsq>> {
        let code = self.abc.digitize_symbol(ch);
        if code == DSQ_IGNORED {
            return Ok(None);
        }
        if code == DSQ_ILLEGAL || (!self.abc.is_residue(code) && code != self.abc.nonresidue_code())
        {
            let display = if ch.is_ascii_graphic() || ch == b' ' {
                (ch as char).to_string()
            } else {
                format!("\\x{ch:02x}")
            };
            let where_text = if seq_name.is_empty() {
                format_name.to_string()
            } else {
                format!("{format_name} '{seq_name}'")
            };
            return Err(HmmerError::Format(format!(
                "Illegal symbol '{display}' in {where_text}"
            )));
        }
        Ok(Some(code))
    }
}

/// Open a sequence file for reading with `abc` (port of `esl_sqfile_Open`).
///
/// Transparently wraps `.gz` files with a `flate2` decompressor; the
/// underlying reader is type-erased into `Box<dyn Read>`. Returns
/// [`HmmerError::Io`] if the path cannot be opened.
pub fn open_seq_file(
    path: &std::path::Path,
    abc: &Alphabet,
) -> HmmerResult<SeqFile<Box<dyn Read>>> {
    open_seq_file_inner(path, abc, false)
}

/// Open a sequence file and require FASTA records.
pub fn open_fasta_seq_file(
    path: &std::path::Path,
    abc: &Alphabet,
) -> HmmerResult<SeqFile<Box<dyn Read>>> {
    open_seq_file_inner(path, abc, true)
}

fn open_seq_file_inner(
    path: &std::path::Path,
    abc: &Alphabet,
    fasta_only: bool,
) -> HmmerResult<SeqFile<Box<dyn Read>>> {
    if path == std::path::Path::new("-") {
        let reader: Box<dyn Read> = Box::new(std::io::stdin());
        let sqf = SeqFile::new(reader, abc.clone());
        return Ok(if fasta_only {
            sqf.with_fasta_only()
        } else {
            sqf
        });
    }
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let reader: Box<dyn Read> = if path.extension().map_or(false, |e| e == "gz") {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let sqf = SeqFile::new(reader, abc.clone());
    Ok(if fasta_only {
        sqf.with_fasta_only()
    } else {
        sqf
    })
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
        assert_eq!(sq.acc, "P13368");
        assert_eq!(sq.desc, "RecName: Full=Protein sevenless; EC=2.7.10.1;");
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

    fn read_fasta_text(text: &str) -> HmmerResult<Sequence> {
        let abc = Alphabet::amino();
        let mut sqf = SeqFile::new(text.as_bytes(), abc);
        let mut sq = Sequence::new();
        sqf.read(&mut sq)?;
        Ok(sq)
    }

    #[test]
    fn test_read_fasta_rejects_hash() {
        let err = read_fasta_text(">seq\nAC#D\n").unwrap_err();
        assert!(matches!(err, HmmerError::Format(_)));
    }

    #[test]
    fn test_read_fasta_rejects_digit() {
        let err = read_fasta_text(">seq\nACD1EF\n").unwrap_err();
        assert!(matches!(err, HmmerError::Format(_)));
    }

    #[test]
    fn test_read_fasta_rejects_gap() {
        let err = read_fasta_text(">seq\nAC-D\n").unwrap_err();
        assert!(matches!(err, HmmerError::Format(_)));
    }

    #[test]
    fn test_read_fasta_accepts_nonresidue_star() {
        let sq = read_fasta_text(">seq\nAC*D\n").unwrap();
        assert_eq!(sq.n, 4);
    }

    #[test]
    fn test_read_fasta_ignores_whitespace() {
        let sq = read_fasta_text(">seq\nAC D\tEF\n").unwrap();
        assert_eq!(sq.n, 5);
    }

    #[test]
    fn test_read_fasta_rejects_preamble_text() {
        let fasta = b"not a sequence record\n>seq1\nACDE\n";
        let abc = Alphabet::amino();
        let mut reader = SeqFile::new(&fasta[..], abc);
        let mut sq = Sequence::new();
        let err = reader.read(&mut sq).unwrap_err();
        assert!(err
            .to_string()
            .contains("unrecognized sequence file record start"));
    }

    #[test]
    fn test_read_fasta_rejects_empty_record() {
        let err = read_fasta_text(">empty\n").unwrap_err();
        assert!(err.to_string().contains("zero-length sequence record"));
    }

    #[test]
    fn test_read_genbank_multiline_definition() {
        let genbank = b"LOCUS       seq1 4 bp DNA\nDEFINITION  first line\n            second line\nACCESSION   ABC123\nORIGIN\n        1 acgt\n//\n";
        let abc = Alphabet::dna();
        let mut reader = SeqFile::new(&genbank[..], abc);
        let mut sq = Sequence::new();
        assert!(reader.read(&mut sq).unwrap());
        assert_eq!(sq.desc, "first line second line");
        assert_eq!(sq.acc, "ABC123");
        assert_eq!(sq.n, 4);
    }
}
