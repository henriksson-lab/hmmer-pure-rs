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
    /// Alignment accession (from #=GF AC)
    pub acc: Option<String>,
    /// Alignment description (from #=GF DE)
    pub desc: Option<String>,
    /// Alignment author/provenance (from #=GF AU)
    pub author: Option<String>,
    /// Sequence names
    pub sqname: Vec<String>,
    /// Per-sequence descriptions (`#=GS <seq> DE`)
    pub sqdesc: Vec<String>,
    /// Aligned sequences (text, with gap characters)
    pub aseq: Vec<Vec<u8>>,
    /// Per-sequence posterior probability annotation (`#=GR <seq> PP`)
    pub pp: Vec<Option<Vec<u8>>>,
    /// Number of sequences
    pub nseq: usize,
    /// Alignment length (columns)
    pub alen: usize,
    /// Reference annotation (#=GC RF)
    pub rf: Option<Vec<u8>>,
    /// Consensus posterior probability annotation (#=GC PP_cons)
    pub pp_cons: Option<Vec<u8>>,
}

impl Msa {
    /// Digitize a text-mode alignment into Easel-style digital rows
    /// (port of `esl_msa_Digitize`, returning the rows instead of
    /// mutating in place).
    ///
    /// Each row is 1-based with `DSQ_SENTINEL` bytes flanking the
    /// alignment columns. Aligned gap characters (`-` or `.`) are mapped
    /// to the alphabet's gap code; symbols that the alphabet ignores
    /// (e.g. whitespace) are skipped silently.
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

/// Compute the Easel 32-bit alignment checksum (`esl_msa_Checksum`).
///
/// Considers only alignment data (digital symbols, columns 1..alen of every
/// sequence), so two MSAs with identical columns but different annotation
/// hash the same. Used to verify that an alignment matches a known
/// reference, e.g. when `hmmalign --mapali` is mapping new sequences onto
/// the seed alignment an HMM was built from. Implements the variant of
/// Jenkins' hash from `esl_keyhash`.
pub fn checksum(msa: &Msa, abc: &Alphabet) -> u32 {
    let mut val = 0u32;
    for row in msa.digitize(abc) {
        for &sym in row.iter().skip(1).take(msa.alen) {
            val = val.wrapping_add(sym as u32);
            val = val.wrapping_add(val << 10);
            val ^= val >> 6;
        }
    }
    val = val.wrapping_add(val << 3);
    val ^= val >> 11;
    val = val.wrapping_add(val << 15);
    val
}

/// Read every Stockholm alignment in `path` (convenience wrapper that opens
/// the file and dispatches to [`read_stockholm_from_reader`]).
pub fn read_stockholm(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let reader = BufReader::new(file);
    read_stockholm_from_reader(reader)
}

/// Read all Stockholm-format alignments from an open reader.
///
/// Scans for `# STOCKHOLM` block headers and dispatches each block (terminated
/// by `//`) to `parse_stockholm_block`. Concatenated multi-MSA files are
/// supported. Lightweight Rust port of the Stockholm subset of
/// `esl_msafile_stockholm.c`.
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
            if i == lines.len() {
                return Err(HmmerError::Format(
                    "missing // terminator after MSA".to_string(),
                ));
            }
            let end = i;
            if let Some(msa) = parse_stockholm_block(&lines[start..=end])? {
                msas.push(msa);
            }
        }
        i += 1;
    }

    Ok(msas)
}

/// Parse one Stockholm block (between `# STOCKHOLM` and `//`) into an [`Msa`].
///
/// Recognises common GF/GS/GR/GC metadata plus bare `name sequence` rows.
/// Returns `Ok(None)` if the block contained no sequences.
fn parse_stockholm_block(lines: &[String]) -> HmmerResult<Option<Msa>> {
    let mut name = String::new();
    let mut acc: Option<String> = None;
    let mut desc: Option<String> = None;
    let mut seq_order: Vec<String> = Vec::new();
    let mut seq_data: HashMap<String, Vec<u8>> = HashMap::new();
    let mut sqdesc: HashMap<String, String> = HashMap::new();
    let mut pp: HashMap<String, Vec<u8>> = HashMap::new();
    let mut rf: Option<Vec<u8>> = None;
    let mut pp_cons: Option<Vec<u8>> = None;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "//" || trimmed.starts_with("# STOCKHOLM") {
            continue;
        }

        if trimmed.starts_with("#=GF ID") {
            name = trimmed[7..].trim().to_string();
        } else if trimmed.starts_with("#=GF AC") {
            acc = Some(trimmed[7..].trim().to_string());
        } else if trimmed.starts_with("#=GF DE") {
            let line = trimmed[7..].trim();
            if !line.is_empty() {
                desc.get_or_insert_with(String::new);
                if let Some(desc) = &mut desc {
                    if !desc.is_empty() {
                        desc.push(' ');
                    }
                    desc.push_str(line);
                }
            }
        } else if trimmed.starts_with("#=GC RF") {
            let rf_str = trimmed[7..].trim();
            match &mut rf {
                Some(existing) => existing.extend_from_slice(rf_str.as_bytes()),
                None => rf = Some(rf_str.as_bytes().to_vec()),
            }
        } else if trimmed.starts_with("#=GC PP_cons") {
            let pp_str = trimmed[12..].trim();
            match &mut pp_cons {
                Some(existing) => existing.extend_from_slice(pp_str.as_bytes()),
                None => pp_cons = Some(pp_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GS ") {
            let fields: Vec<&str> = rest.splitn(3, char::is_whitespace).collect();
            if fields.len() == 3 && fields[1] == "DE" {
                sqdesc
                    .entry(fields[0].to_string())
                    .and_modify(|desc| {
                        if !desc.is_empty() {
                            desc.push(' ');
                        }
                        desc.push_str(fields[2].trim());
                    })
                    .or_insert_with(|| fields[2].trim().to_string());
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GR ") {
            let fields: Vec<&str> = rest.splitn(3, char::is_whitespace).collect();
            if fields.len() == 3 && fields[1] == "PP" {
                pp.entry(fields[0].to_string())
                    .and_modify(|line| line.extend_from_slice(fields[2].trim().as_bytes()))
                    .or_insert_with(|| fields[2].trim().as_bytes().to_vec());
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

    let alen = seq_data
        .get(&seq_order[0])
        .map(|v| v.len())
        .unwrap_or_default();
    for name in &seq_order {
        let len = seq_data.get(name).map(|v| v.len()).unwrap_or_default();
        if len != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm sequence {} has aligned length {}, expected {}",
                name, len, alen
            )));
        }
    }
    if let Some(ref rf) = rf {
        if rf.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GC RF annotation has length {}, expected {}",
                rf.len(),
                alen
            )));
        }
    }
    if let Some(ref pp_cons) = pp_cons {
        if pp_cons.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GC PP_cons annotation has length {}, expected {}",
                pp_cons.len(),
                alen
            )));
        }
    }
    for (name, line) in &pp {
        if !seq_order.iter().any(|seq_name| seq_name == name) {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GR {} PP annotation refers to unknown sequence",
                name
            )));
        }
        if line.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GR {} PP annotation has length {}, expected {}",
                name,
                line.len(),
                alen
            )));
        }
        validate_pp_annotation(line, &format!("#=GR {} PP", name))?;
    }
    for name in sqdesc.keys() {
        if !seq_order.iter().any(|seq_name| seq_name == name) {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GS {} DE annotation refers to unknown sequence",
                name
            )));
        }
    }
    if let Some(ref pp_cons) = pp_cons {
        validate_pp_annotation(pp_cons, "#=GC PP_cons")?;
    }
    let nseq = seq_order.len();

    let aseq: Vec<Vec<u8>> = seq_order
        .iter()
        .map(|name| seq_data.remove(name).unwrap_or_default())
        .collect();
    let sqdesc_vec: Vec<String> = seq_order
        .iter()
        .map(|name| sqdesc.remove(name).unwrap_or_default())
        .collect();
    let pp_vec: Vec<Option<Vec<u8>>> = seq_order.iter().map(|name| pp.remove(name)).collect();

    Ok(Some(Msa {
        name,
        acc,
        desc,
        author: None,
        sqname: seq_order,
        sqdesc: sqdesc_vec,
        pp: pp_vec,
        aseq,
        nseq,
        alen,
        rf,
        pp_cons,
    }))
}

fn validate_pp_annotation(line: &[u8], label: &str) -> HmmerResult<()> {
    if let Some(&bad) = line
        .iter()
        .find(|&&ch| !(ch == b'.' || ch == b'*' || ch.is_ascii_digit()))
    {
        return Err(HmmerError::Format(format!(
            "Stockholm {} annotation contains invalid PP character '{}'",
            label, bad as char
        )));
    }
    Ok(())
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

    #[test]
    fn rejects_unterminated_stockholm_block() {
        let input = b"# STOCKHOLM 1.0\nseq AC\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("missing // terminator"));
    }

    #[test]
    fn rejects_mismatched_stockholm_row_lengths() {
        let input = b"# STOCKHOLM 1.0\nseq1 AC\nseq2 A\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("aligned length"));
    }

    #[test]
    fn parses_stockholm_description_and_pp_annotations() {
        let input =
            b"# STOCKHOLM 1.0\n#=GS s1 DE first desc\ns1 AC\n#=GR s1 PP 9*\n#=GC PP_cons 8*\n//\n";
        let msas = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap();
        let msa = &msas[0];
        assert_eq!(msa.sqdesc[0], "first desc");
        assert_eq!(msa.pp[0].as_deref(), Some(&b"9*"[..]));
        assert_eq!(msa.pp_cons.as_deref(), Some(&b"8*"[..]));
    }

    #[test]
    fn rejects_invalid_stockholm_pp_annotations() {
        let input = b"# STOCKHOLM 1.0\ns1 AC\n#=GR s1 PP 9A\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("invalid PP character"));

        let input = b"# STOCKHOLM 1.0\ns1 AC\n#=GR missing PP 9*\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("unknown sequence"));
    }
}
